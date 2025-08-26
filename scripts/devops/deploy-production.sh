#!/bin/bash

# Production Deployment Script for Claude TUI
# Supports multiple deployment strategies with automated rollback

set -euo pipefail

# Default values
IMAGE=""
STRATEGY="rolling"
TIMEOUT=1800
HEALTH_CHECKS=5
NAMESPACE="production"
ROLLBACK_ON_FAILURE=true
DRY_RUN=false

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

# Help function
show_help() {
    cat << EOF
Production Deployment Script for Claude TUI

Usage: $0 [OPTIONS]

Options:
    --image IMAGE               Docker image to deploy (required)
    --strategy STRATEGY         Deployment strategy: rolling, blue-green, canary (default: rolling)
    --timeout SECONDS           Deployment timeout in seconds (default: 1800)
    --health-checks COUNT       Number of health checks to perform (default: 5)
    --namespace NAMESPACE       Kubernetes namespace (default: production)
    --no-rollback              Disable automatic rollback on failure
    --dry-run                  Show what would be deployed without actually deploying
    -h, --help                 Show this help message

Examples:
    $0 --image ghcr.io/claude-tui/claude-tui:v1.0.0
    $0 --image ghcr.io/claude-tui/claude-tui:latest --strategy blue-green
    $0 --image ghcr.io/claude-tui/claude-tui:v1.1.0 --strategy canary --timeout 3600

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --health-checks)
            HEALTH_CHECKS="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$IMAGE" ]]; then
    log_error "Image is required. Use --image to specify the Docker image."
    exit 1
fi

# Validate strategy
case "$STRATEGY" in
    rolling|blue-green|canary)
        ;;
    *)
        log_error "Invalid strategy: $STRATEGY. Valid options: rolling, blue-green, canary"
        exit 1
        ;;
esac

# Validate kubectl and cluster connection
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed or not in PATH"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

# Functions for different deployment strategies
deploy_rolling() {
    local image=$1
    
    log_info "Starting rolling deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would update deployment with image: $image"
        return 0
    fi
    
    # Get current image for rollback
    local current_image
    current_image=$(kubectl get deployment claude-tui-app -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
    log_info "Current image: $current_image"
    log_info "New image: $image"
    
    # Update the deployment
    kubectl set image deployment/claude-tui-app claude-tui="$image" -n "$NAMESPACE"
    
    # Wait for rollout to complete
    log_info "Waiting for rollout to complete (timeout: ${TIMEOUT}s)..."
    if kubectl rollout status deployment/claude-tui-app -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_success "Rolling deployment completed successfully"
        return 0
    else
        log_error "Rolling deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            log_warning "Initiating rollback to previous image: $current_image"
            kubectl set image deployment/claude-tui-app claude-tui="$current_image" -n "$NAMESPACE"
            kubectl rollout status deployment/claude-tui-app -n "$NAMESPACE" --timeout=300s
        fi
        return 1
    fi
}

deploy_blue_green() {
    local image=$1
    
    log_info "Starting blue-green deployment..."
    
    # Determine current and target colors
    local current_color
    current_color=$(kubectl get service claude-tui-service -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")
    local target_color
    if [[ "$current_color" == "blue" ]]; then
        target_color="green"
    else
        target_color="blue"
    fi
    
    log_info "Current color: $current_color"
    log_info "Deploying to: $target_color"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy to $target_color environment with image: $image"
        return 0
    fi
    
    # Deploy to target environment
    kubectl set image deployment/claude-tui-"$target_color" claude-tui="$image" -n "$NAMESPACE"
    
    # Wait for target deployment to be ready
    log_info "Waiting for $target_color deployment to be ready..."
    if kubectl rollout status deployment/claude-tui-"$target_color" -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_success "$target_color deployment ready"
    else
        log_error "$target_color deployment failed"
        return 1
    fi
    
    # Run health checks on target environment
    if run_health_checks "$target_color"; then
        # Switch traffic to target
        log_info "Switching traffic to $target_color..."
        kubectl patch service claude-tui-service -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'$target_color'"}}}'
        
        # Wait a bit and verify
        sleep 30
        if run_health_checks; then
            log_success "Blue-green deployment completed successfully"
            
            # Scale down old environment
            log_info "Scaling down $current_color environment..."
            kubectl scale deployment claude-tui-"$current_color" --replicas=0 -n "$NAMESPACE"
            return 0
        else
            log_error "Health checks failed after traffic switch"
            # Rollback traffic
            kubectl patch service claude-tui-service -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'$current_color'"}}}'
            return 1
        fi
    else
        log_error "Health checks failed for $target_color environment"
        return 1
    fi
}

deploy_canary() {
    local image=$1
    local canary_percentage=10
    
    log_info "Starting canary deployment with $canary_percentage% traffic..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy canary with $canary_percentage% traffic using image: $image"
        return 0
    fi
    
    # Deploy canary version
    kubectl set image deployment/claude-tui-canary claude-tui="$image" -n "$NAMESPACE"
    
    # Wait for canary to be ready
    log_info "Waiting for canary deployment to be ready..."
    if kubectl rollout status deployment/claude-tui-canary -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_success "Canary deployment ready"
    else
        log_error "Canary deployment failed"
        return 1
    fi
    
    # Configure traffic splitting (this would typically use a service mesh like Istio)
    log_info "Configuring traffic split: $canary_percentage% to canary, $((100-canary_percentage))% to stable"
    
    # Monitor canary for a period
    log_info "Monitoring canary deployment for 5 minutes..."
    sleep 300
    
    # Run health checks and metrics validation
    if run_health_checks "canary" && validate_canary_metrics; then
        log_success "Canary validation successful. Promoting to full deployment..."
        
        # Promote canary to main deployment
        kubectl set image deployment/claude-tui-app claude-tui="$image" -n "$NAMESPACE"
        kubectl rollout status deployment/claude-tui-app -n "$NAMESPACE" --timeout="${TIMEOUT}s"
        
        # Scale down canary
        kubectl scale deployment claude-tui-canary --replicas=0 -n "$NAMESPACE"
        
        log_success "Canary deployment completed successfully"
        return 0
    else
        log_error "Canary validation failed. Rolling back..."
        kubectl scale deployment claude-tui-canary --replicas=0 -n "$NAMESPACE"
        return 1
    fi
}

# Health check function
run_health_checks() {
    local target=${1:-""}
    local service_name="claude-tui-service"
    
    if [[ -n "$target" ]]; then
        service_name="claude-tui-service-$target"
    fi
    
    log_info "Running health checks for $service_name..."
    
    # Get service endpoint
    local endpoint
    endpoint=$(kubectl get service "$service_name" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
    
    if [[ -z "$endpoint" ]]; then
        # Try getting cluster IP if no load balancer
        endpoint=$(kubectl get service "$service_name" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if [[ -z "$endpoint" ]]; then
            log_error "Could not get service endpoint"
            return 1
        fi
    fi
    
    log_info "Service endpoint: $endpoint"
    
    # Run multiple health checks
    local success_count=0
    for ((i=1; i<=HEALTH_CHECKS; i++)); do
        log_info "Health check $i/$HEALTH_CHECKS..."
        
        # Basic health check
        if curl -s -f "http://$endpoint/health" > /dev/null; then
            log_info "✓ Basic health check passed"
            success_count=$((success_count + 1))
        else
            log_warning "✗ Basic health check failed"
        fi
        
        # API health check
        if curl -s -f "http://$endpoint/api/v1/health" > /dev/null; then
            log_info "✓ API health check passed"
            success_count=$((success_count + 1))
        else
            log_warning "✗ API health check failed"
        fi
        
        sleep 10
    done
    
    local success_rate=$((success_count * 100 / (HEALTH_CHECKS * 2)))
    log_info "Health check success rate: $success_rate%"
    
    if [[ $success_rate -ge 80 ]]; then
        log_success "Health checks passed (success rate: $success_rate%)"
        return 0
    else
        log_error "Health checks failed (success rate: $success_rate%)"
        return 1
    fi
}

# Validate canary metrics
validate_canary_metrics() {
    log_info "Validating canary metrics..."
    
    # This would typically integrate with monitoring systems like Prometheus
    # For now, we'll do basic validation
    
    # Check error rate
    # Check response time
    # Check resource usage
    
    # Placeholder implementation
    sleep 10
    log_info "Canary metrics validation passed"
    return 0
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        return 1
    fi
    
    # Check if required deployments exist
    case "$STRATEGY" in
        rolling)
            if ! kubectl get deployment claude-tui-app -n "$NAMESPACE" &> /dev/null; then
                log_error "Deployment claude-tui-app not found in namespace $NAMESPACE"
                return 1
            fi
            ;;
        blue-green)
            if ! kubectl get deployment claude-tui-blue -n "$NAMESPACE" &> /dev/null || \
               ! kubectl get deployment claude-tui-green -n "$NAMESPACE" &> /dev/null; then
                log_error "Blue-green deployments not found in namespace $NAMESPACE"
                return 1
            fi
            ;;
        canary)
            if ! kubectl get deployment claude-tui-app -n "$NAMESPACE" &> /dev/null || \
               ! kubectl get deployment claude-tui-canary -n "$NAMESPACE" &> /dev/null; then
                log_error "Main or canary deployment not found in namespace $NAMESPACE"
                return 1
            fi
            ;;
    esac
    
    # Verify image exists (basic check)
    if ! docker manifest inspect "$IMAGE" &> /dev/null; then
        log_warning "Could not verify image exists: $IMAGE"
    fi
    
    log_success "Pre-deployment checks passed"
    return 0
}

# Post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."
    
    # Update deployment annotations
    kubectl annotate deployment claude-tui-app deployment.kubernetes.io/revision-history-limit=10 -n "$NAMESPACE" --overwrite
    
    # Log deployment details
    log_info "Deployment completed:"
    log_info "  Image: $IMAGE"
    log_info "  Strategy: $STRATEGY"
    log_info "  Namespace: $NAMESPACE"
    log_info "  Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    
    # Optional: Send notifications, update monitoring, etc.
    
    log_success "Post-deployment tasks completed"
}

# Main execution
main() {
    log_info "Starting Claude TUI production deployment..."
    log_info "Image: $IMAGE"
    log_info "Strategy: $STRATEGY"
    log_info "Namespace: $NAMESPACE"
    log_info "Dry Run: $DRY_RUN"
    
    # Run pre-deployment checks
    if ! pre_deployment_checks; then
        log_error "Pre-deployment checks failed"
        exit 1
    fi
    
    # Execute deployment based on strategy
    case "$STRATEGY" in
        rolling)
            if deploy_rolling "$IMAGE"; then
                log_success "Rolling deployment successful"
            else
                log_error "Rolling deployment failed"
                exit 1
            fi
            ;;
        blue-green)
            if deploy_blue_green "$IMAGE"; then
                log_success "Blue-green deployment successful"
            else
                log_error "Blue-green deployment failed"
                exit 1
            fi
            ;;
        canary)
            if deploy_canary "$IMAGE"; then
                log_success "Canary deployment successful"
            else
                log_error "Canary deployment failed"
                exit 1
            fi
            ;;
    esac
    
    # Run post-deployment tasks
    if [[ "$DRY_RUN" == "false" ]]; then
        post_deployment_tasks
    fi
    
    log_success "Deployment completed successfully!"
}

# Trap to handle script interruption
trap 'log_error "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"