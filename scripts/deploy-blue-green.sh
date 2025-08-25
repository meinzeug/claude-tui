#!/bin/bash

# Blue-Green Deployment Script for Claude-TIU
# Implements zero-downtime deployment strategy

set -euo pipefail

# Configuration
NAMESPACE="production"
APP_NAME="claude-tiu"
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_TIMEOUT=120

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to check if kubectl is available
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Function to get current active color
get_current_color() {
    local active_color
    active_color=$(kubectl get service "${APP_NAME}" -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "")
    
    if [[ -z "$active_color" ]]; then
        # If no color selector, check which deployment has more replicas
        local blue_replicas green_replicas
        blue_replicas=$(kubectl get deployment "${APP_NAME}-blue" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        green_replicas=$(kubectl get deployment "${APP_NAME}-green" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        
        if [[ "$blue_replicas" -gt "$green_replicas" ]]; then
            active_color="blue"
        else
            active_color="green"
        fi
    fi
    
    echo "$active_color"
}

# Function to determine new color
get_new_color() {
    local current_color="$1"
    if [[ "$current_color" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Function to update deployment image
update_deployment_image() {
    local color="$1"
    local image="$2"
    local deployment_name="${APP_NAME}-${color}"
    
    log "Updating ${deployment_name} with image ${image}"
    
    kubectl set image deployment/"$deployment_name" "$APP_NAME=$image" -n "$NAMESPACE"
    
    # Wait for rollout to complete
    if ! kubectl rollout status deployment/"$deployment_name" -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        error "Deployment rollout failed for $deployment_name"
        return 1
    fi
    
    log "Successfully updated $deployment_name"
}

# Function to perform health check
health_check() {
    local color="$1"
    local deployment_name="${APP_NAME}-${color}"
    
    log "Performing health check for ${deployment_name}..."
    
    # Wait for pods to be ready
    if ! kubectl wait --for=condition=ready pod -l "app=${APP_NAME},color=${color}" -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        error "Pods failed to become ready for $deployment_name"
        return 1
    fi
    
    # Get pod IP for direct health check
    local pod_name
    pod_name=$(kubectl get pods -l "app=${APP_NAME},color=${color}" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [[ -z "$pod_name" ]]; then
        error "No pods found for $deployment_name"
        return 1
    fi
    
    # Perform health check via port-forward
    log "Testing health endpoint for $pod_name"
    
    # Start port-forward in background
    kubectl port-forward "$pod_name" 8080:8000 -n "$NAMESPACE" &
    local pf_pid=$!
    
    # Wait for port-forward to establish
    sleep 5
    
    # Perform health check
    local health_check_passed=false
    for i in {1..10}; do
        if curl -f -s -m 10 "http://localhost:8080/health" > /dev/null 2>&1; then
            health_check_passed=true
            break
        fi
        warn "Health check attempt $i failed, retrying..."
        sleep 5
    done
    
    # Clean up port-forward
    kill $pf_pid 2>/dev/null || true
    
    if [[ "$health_check_passed" == "true" ]]; then
        log "Health check passed for $deployment_name"
        return 0
    else
        error "Health check failed for $deployment_name"
        return 1
    fi
}

# Function to switch traffic
switch_traffic() {
    local new_color="$1"
    
    log "Switching traffic to $new_color environment"
    
    # Update service selector to point to new color
    kubectl patch service "$APP_NAME" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"color\":\"$new_color\"}}}"
    
    log "Traffic switched to $new_color environment"
}

# Function to scale down old environment
scale_down_old_environment() {
    local old_color="$1"
    local deployment_name="${APP_NAME}-${old_color}"
    
    log "Scaling down old environment: $deployment_name"
    
    kubectl scale deployment "$deployment_name" --replicas=0 -n "$NAMESPACE"
    
    log "Old environment scaled down"
}

# Function to scale up new environment
scale_up_new_environment() {
    local new_color="$1"
    local replicas="${2:-3}"
    local deployment_name="${APP_NAME}-${new_color}"
    
    log "Scaling up new environment: $deployment_name to $replicas replicas"
    
    kubectl scale deployment "$deployment_name" --replicas="$replicas" -n "$NAMESPACE"
    
    # Wait for deployment to be ready
    if ! kubectl rollout status deployment/"$deployment_name" -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        error "Failed to scale up $deployment_name"
        return 1
    fi
    
    log "New environment scaled up successfully"
}

# Function to rollback deployment
rollback_deployment() {
    local current_color="$1"
    local old_color
    
    if [[ "$current_color" == "blue" ]]; then
        old_color="green"
    else
        old_color="blue"
    fi
    
    error "Performing rollback to $old_color environment"
    
    # Scale up old environment
    scale_up_new_environment "$old_color"
    
    # Switch traffic back
    switch_traffic "$old_color"
    
    # Scale down failed environment
    scale_down_old_environment "$current_color"
    
    log "Rollback completed successfully"
}

# Function to create deployment backup
create_backup() {
    local backup_name="pre-deployment-backup-$(date +%Y%m%d-%H%M%S)"
    local backup_dir="backups"
    
    log "Creating deployment backup: $backup_name"
    
    mkdir -p "$backup_dir"
    
    # Backup current deployments
    kubectl get deployment "${APP_NAME}-blue" -n "$NAMESPACE" -o yaml > "$backup_dir/${backup_name}-blue-deployment.yaml" 2>/dev/null || true
    kubectl get deployment "${APP_NAME}-green" -n "$NAMESPACE" -o yaml > "$backup_dir/${backup_name}-green-deployment.yaml" 2>/dev/null || true
    kubectl get service "$APP_NAME" -n "$NAMESPACE" -o yaml > "$backup_dir/${backup_name}-service.yaml" 2>/dev/null || true
    
    log "Backup created in $backup_dir/$backup_name"
}

# Function to perform post-deployment validation
post_deployment_validation() {
    local new_color="$1"
    
    log "Performing post-deployment validation..."
    
    # Check service is routing correctly
    local service_color
    service_color=$(kubectl get service "$APP_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}')
    
    if [[ "$service_color" != "$new_color" ]]; then
        error "Service is not routing to the correct color: expected $new_color, got $service_color"
        return 1
    fi
    
    # Check deployment status
    local ready_replicas desired_replicas
    ready_replicas=$(kubectl get deployment "${APP_NAME}-${new_color}" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    desired_replicas=$(kubectl get deployment "${APP_NAME}-${new_color}" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    
    if [[ "$ready_replicas" != "$desired_replicas" ]]; then
        error "Deployment not ready: $ready_replicas/$desired_replicas replicas ready"
        return 1
    fi
    
    log "Post-deployment validation passed"
}

# Main deployment function
deploy() {
    local image="$1"
    local replicas="${2:-3}"
    
    log "Starting Blue-Green deployment for Claude-TIU"
    log "Image: $image"
    log "Target replicas: $replicas"
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup
    create_backup
    
    # Determine current and new colors
    local current_color new_color
    current_color=$(get_current_color)
    new_color=$(get_new_color "$current_color")
    
    log "Current active environment: $current_color"
    log "Deploying to environment: $new_color"
    
    # Scale up new environment
    if ! scale_up_new_environment "$new_color" "$replicas"; then
        error "Failed to scale up new environment"
        exit 1
    fi
    
    # Update new environment with new image
    if ! update_deployment_image "$new_color" "$image"; then
        error "Failed to update deployment image"
        scale_down_old_environment "$new_color"
        exit 1
    fi
    
    # Perform health check on new environment
    if ! health_check "$new_color"; then
        error "Health check failed for new environment"
        scale_down_old_environment "$new_color"
        exit 1
    fi
    
    # Switch traffic to new environment
    switch_traffic "$new_color"
    
    # Perform post-deployment validation
    if ! post_deployment_validation "$new_color"; then
        error "Post-deployment validation failed"
        rollback_deployment "$new_color"
        exit 1
    fi
    
    # Scale down old environment
    scale_down_old_environment "$current_color"
    
    log "Blue-Green deployment completed successfully!"
    log "Active environment: $new_color"
    log "Image: $image"
}

# Function to show current status
status() {
    log "Claude-TIU Blue-Green Deployment Status"
    
    local current_color
    current_color=$(get_current_color)
    
    echo
    echo "Current Active Environment: $current_color"
    echo
    
    # Show deployment status
    echo "Deployment Status:"
    kubectl get deployments -l "app=$APP_NAME" -n "$NAMESPACE" -o wide
    
    echo
    echo "Service Status:"
    kubectl get service "$APP_NAME" -n "$NAMESPACE" -o wide
    
    echo
    echo "Pod Status:"
    kubectl get pods -l "app=$APP_NAME" -n "$NAMESPACE" -o wide
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    deploy <image> [replicas]  Deploy new image using blue-green strategy
    status                     Show current deployment status
    rollback                   Rollback to previous environment
    help                       Show this help message

Examples:
    $0 deploy ghcr.io/claude-tiu/claude-tiu:v1.2.3
    $0 deploy ghcr.io/claude-tiu/claude-tiu:latest 5
    $0 status
    $0 rollback

Environment Variables:
    NAMESPACE              Kubernetes namespace (default: production)
    APP_NAME              Application name (default: claude-tiu)
    HEALTH_CHECK_TIMEOUT  Health check timeout in seconds (default: 300)

EOF
}

# Main script logic
case "${1:-}" in
    deploy)
        if [[ -z "${2:-}" ]]; then
            error "Image parameter is required"
            usage
            exit 1
        fi
        deploy "$2" "${3:-3}"
        ;;
    status)
        status
        ;;
    rollback)
        current_color=$(get_current_color)
        rollback_deployment "$current_color"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        error "Unknown command: ${1:-}"
        usage
        exit 1
        ;;
esac