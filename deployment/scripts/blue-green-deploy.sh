#!/bin/bash

# Blue-Green Deployment Script for Claude TUI
# Implements zero-downtime deployment with automated rollback

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-claude-tui}"
SERVICE_NAME="${SERVICE_NAME:-claude-tui}"
DEPLOYMENT_TIMEOUT="${DEPLOYMENT_TIMEOUT:-300}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-60}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Get current active deployment (blue or green)
get_active_deployment() {
    local active_selector=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "")
    
    if [[ "$active_selector" == "blue" ]]; then
        echo "blue"
    elif [[ "$active_selector" == "green" ]]; then
        echo "green"
    else
        # Default to blue if no active deployment
        echo "blue"
    fi
}

# Get inactive deployment (opposite of active)
get_inactive_deployment() {
    local active="$1"
    if [[ "$active" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Deploy to inactive environment
deploy_inactive() {
    local inactive="$1"
    local image="$2"
    
    log_info "Deploying to $inactive environment..."
    
    # Update the inactive deployment with new image
    kubectl patch deployment "claude-tui-$inactive" -n "$NAMESPACE" \
        -p '{"spec":{"template":{"spec":{"containers":[{"name":"claude-tui","image":"'"$image"'"}]}}}}'
    
    # Wait for rollout to complete
    log_info "Waiting for $inactive deployment to complete..."
    if ! kubectl rollout status deployment/"claude-tui-$inactive" -n "$NAMESPACE" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "Deployment to $inactive failed"
        return 1
    fi
    
    log_success "$inactive deployment completed"
    return 0
}

# Health check function
health_check() {
    local deployment="$1"
    local retries=0
    local max_retries=$((HEALTH_CHECK_TIMEOUT / 5))
    
    log_info "Running health checks for $deployment deployment..."
    
    # Get pod IP for direct health check
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui,version="$deployment" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$pod_name" ]]; then
        log_error "No pods found for $deployment deployment"
        return 1
    fi
    
    while [[ $retries -lt $max_retries ]]; do
        log_info "Health check attempt $((retries + 1))/$max_retries..."
        
        # Check if pod is ready
        if kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' | grep -q "True"; then
            # Additional application-level health check
            if kubectl exec "$pod_name" -n "$NAMESPACE" -- curl -f http://localhost:8080/health &>/dev/null; then
                log_success "Health check passed for $deployment"
                return 0
            fi
        fi
        
        sleep 5
        ((retries++))
    done
    
    log_error "Health check failed for $deployment after $max_retries attempts"
    return 1
}

# Performance validation
performance_validation() {
    local deployment="$1"
    
    log_info "Running performance validation for $deployment deployment..."
    
    # Get service endpoint
    local service_endpoint=$(kubectl get service "$SERVICE_NAME-$deployment" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [[ -z "$service_endpoint" ]]; then
        log_warning "Could not get service endpoint, skipping performance validation"
        return 0
    fi
    
    # Run basic performance test
    local response_time=$(curl -w "%{time_total}" -s -o /dev/null "http://$service_endpoint/health" || echo "999")
    
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
        log_error "Performance validation failed: Response time $response_time seconds exceeds threshold"
        return 1
    fi
    
    log_success "Performance validation passed: Response time $response_time seconds"
    return 0
}

# Switch traffic to new deployment
switch_traffic() {
    local new_active="$1"
    
    log_info "Switching traffic to $new_active deployment..."
    
    # Update service selector to point to new deployment
    kubectl patch service "$SERVICE_NAME" -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"version":"'"$new_active"'"}}}'
    
    # Verify switch
    local updated_selector=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}')
    
    if [[ "$updated_selector" == "$new_active" ]]; then
        log_success "Traffic successfully switched to $new_active"
        return 0
    else
        log_error "Failed to switch traffic to $new_active"
        return 1
    fi
}

# Rollback function
rollback() {
    local rollback_to="$1"
    
    log_warning "Initiating rollback to $rollback_to deployment..."
    
    # Switch service back
    kubectl patch service "$SERVICE_NAME" -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"version":"'"$rollback_to"'"}}}'
    
    log_success "Rollback completed to $rollback_to"
}

# Cleanup old deployment
cleanup_old_deployment() {
    local old_deployment="$1"
    
    log_info "Scaling down old $old_deployment deployment..."
    
    # Scale down old deployment to 0 replicas
    kubectl scale deployment "claude-tui-$old_deployment" -n "$NAMESPACE" --replicas=0
    
    log_success "Old $old_deployment deployment scaled down"
}

# Main deployment function
main() {
    local new_image="$1"
    
    if [[ -z "$new_image" ]]; then
        log_error "Usage: $0 <new-image>"
        exit 1
    fi
    
    log_info "Starting blue-green deployment for image: $new_image"
    
    # Check prerequisites
    check_prerequisites
    
    # Get current active deployment
    local active_deployment=$(get_active_deployment)
    local inactive_deployment=$(get_inactive_deployment "$active_deployment")
    
    log_info "Current active deployment: $active_deployment"
    log_info "Deploying to inactive deployment: $inactive_deployment"
    
    # Store original deployment for rollback
    local original_deployment="$active_deployment"
    
    # Deploy to inactive environment
    if ! deploy_inactive "$inactive_deployment" "$new_image"; then
        log_error "Deployment failed"
        exit 1
    fi
    
    # Run health checks on new deployment
    if ! health_check "$inactive_deployment"; then
        log_error "Health checks failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback "$original_deployment"
        fi
        exit 1
    fi
    
    # Run performance validation
    if ! performance_validation "$inactive_deployment"; then
        log_error "Performance validation failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback "$original_deployment"
        fi
        exit 1
    fi
    
    # Switch traffic to new deployment
    if ! switch_traffic "$inactive_deployment"; then
        log_error "Traffic switch failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback "$original_deployment"
        fi
        exit 1
    fi
    
    # Final health check after traffic switch
    sleep 10
    if ! health_check "$inactive_deployment"; then
        log_error "Post-switch health check failed"
        rollback "$original_deployment"
        exit 1
    fi
    
    # Cleanup old deployment
    cleanup_old_deployment "$active_deployment"
    
    log_success "Blue-green deployment completed successfully!"
    log_info "New active deployment: $inactive_deployment"
    log_info "Deployment image: $new_image"
    
    # Send deployment notification
    if command -v curl &> /dev/null && [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ Claude TUI production deployment completed successfully\nImage: $new_image\nActive: $inactive_deployment\"}" \
            "$SLACK_WEBHOOK_URL" || true
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi