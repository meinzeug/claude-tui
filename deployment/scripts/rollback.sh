#!/bin/bash

# Automated Rollback Script for Claude TUI
# Provides fast, reliable rollback capabilities with safety checks

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-claude-tui}"
SERVICE_NAME="${SERVICE_NAME:-claude-tui}"
ROLLBACK_TIMEOUT="${ROLLBACK_TIMEOUT:-180}"
VALIDATE_ROLLBACK="${VALIDATE_ROLLBACK:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
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

# Environment from argument
ENVIRONMENT="${1:-production}"

# Get deployment history
get_deployment_history() {
    local deployment="$1"
    kubectl rollout history deployment/"$deployment" -n "$NAMESPACE" --no-headers | tail -n 5
}

# Get current revision
get_current_revision() {
    local deployment="$1"
    kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}'
}

# Get previous revision
get_previous_revision() {
    local deployment="$1"
    local current_revision=$(get_current_revision "$deployment")
    local previous_revision=$((current_revision - 1))
    
    # Verify previous revision exists
    if kubectl rollout history deployment/"$deployment" -n "$NAMESPACE" --revision="$previous_revision" &>/dev/null; then
        echo "$previous_revision"
    else
        log_error "Previous revision $previous_revision not found for deployment $deployment"
        return 1
    fi
}

# Rollback deployment
rollback_deployment() {
    local deployment="$1"
    local revision="${2:-}"
    
    log_info "Rolling back deployment $deployment..."
    
    if [[ -n "$revision" ]]; then
        log_info "Rolling back to specific revision: $revision"
        kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE" --to-revision="$revision"
    else
        log_info "Rolling back to previous revision"
        kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE"
    fi
    
    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    if kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"; then
        log_success "Rollback completed for $deployment"
        return 0
    else
        log_error "Rollback failed for $deployment"
        return 1
    fi
}

# Get active deployment color (blue/green)
get_active_deployment_color() {
    local active_selector=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "")
    
    if [[ "$active_selector" == "blue" ]]; then
        echo "blue"
    elif [[ "$active_selector" == "green" ]]; then
        echo "green"
    else
        # Try to determine from deployments
        local blue_replicas=$(kubectl get deployment "claude-tui-blue" -n "$NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
        local green_replicas=$(kubectl get deployment "claude-tui-green" -n "$NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
        
        if [[ $blue_replicas -gt 0 ]]; then
            echo "blue"
        elif [[ $green_replicas -gt 0 ]]; then
            echo "green"
        else
            echo "unknown"
        fi
    fi
}

# Switch to other deployment color
switch_to_other_color() {
    local current_color="$1"
    local target_color
    
    if [[ "$current_color" == "blue" ]]; then
        target_color="green"
    elif [[ "$current_color" == "green" ]]; then
        target_color="blue"
    else
        log_error "Cannot determine deployment color to switch to"
        return 1
    fi
    
    log_info "Switching traffic from $current_color to $target_color..."
    
    # Check if target deployment exists and is ready
    if ! kubectl get deployment "claude-tui-$target_color" -n "$NAMESPACE" &>/dev/null; then
        log_error "Target deployment claude-tui-$target_color does not exist"
        return 1
    fi
    
    # Scale up target deployment if needed
    local target_replicas=$(kubectl get deployment "claude-tui-$target_color" -n "$NAMESPACE" -o jsonpath='{.status.replicas}' || echo "0")
    
    if [[ $target_replicas -eq 0 ]]; then
        log_info "Scaling up $target_color deployment..."
        kubectl scale deployment "claude-tui-$target_color" -n "$NAMESPACE" --replicas=3
        
        if ! kubectl rollout status deployment/"claude-tui-$target_color" -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"; then
            log_error "Failed to scale up $target_color deployment"
            return 1
        fi
    fi
    
    # Switch service to target color
    kubectl patch service "$SERVICE_NAME" -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"version":"'"$target_color"'"}}}'
    
    # Verify switch
    local updated_selector=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}')
    
    if [[ "$updated_selector" == "$target_color" ]]; then
        log_success "Successfully switched traffic to $target_color deployment"
        return 0
    else
        log_error "Failed to switch traffic to $target_color deployment"
        return 1
    fi
}

# Validate rollback
validate_rollback() {
    log_info "Validating rollback..."
    
    # Run health checks
    if command -v /home/tekkadmin/claude-tui/deployment/scripts/health-checks.sh &>/dev/null; then
        if /home/tekkadmin/claude-tui/deployment/scripts/health-checks.sh "$ENVIRONMENT"; then
            log_success "Rollback validation passed"
            return 0
        else
            log_error "Rollback validation failed"
            return 1
        fi
    else
        log_warning "Health check script not found, performing basic validation..."
        
        # Basic pod readiness check
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | tr ' ' '\n' | grep -c "True" || echo "0")
        
        if [[ $ready_pods -gt 0 ]]; then
            log_success "Basic validation passed: $ready_pods pods ready"
            return 0
        else
            log_error "Basic validation failed: no ready pods"
            return 1
        fi
    fi
}

# Show deployment status
show_deployment_status() {
    log_info "Current deployment status:"
    
    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE" -l app=claude-tui
    
    echo ""
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app=claude-tui
    
    echo ""
    echo "Service:"
    kubectl get service "$SERVICE_NAME" -n "$NAMESPACE"
    
    # Show active deployment color if using blue-green
    local active_color=$(get_active_deployment_color)
    if [[ "$active_color" != "unknown" ]]; then
        echo ""
        log_info "Active deployment color: $active_color"
    fi
}

# Emergency rollback (fastest possible)
emergency_rollback() {
    log_warning "Initiating emergency rollback..."
    
    # Get active deployment color
    local active_color=$(get_active_deployment_color)
    
    if [[ "$active_color" == "unknown" ]]; then
        log_error "Cannot determine active deployment for emergency rollback"
        return 1
    fi
    
    # Switch to other color immediately
    if switch_to_other_color "$active_color"; then
        log_success "Emergency rollback completed"
        return 0
    else
        log_error "Emergency rollback failed"
        return 1
    fi
}

# Standard rollback with revision
standard_rollback() {
    local revision="${1:-}"
    
    # Determine which deployments to rollback
    local deployments=()
    
    # Check if using blue-green deployment
    local active_color=$(get_active_deployment_color)
    
    if [[ "$active_color" != "unknown" ]]; then
        deployments=("claude-tui-$active_color")
    else
        # Single deployment model
        if kubectl get deployment claude-tui -n "$NAMESPACE" &>/dev/null; then
            deployments=("claude-tui")
        else
            log_error "No deployments found to rollback"
            return 1
        fi
    fi
    
    log_info "Rolling back deployments: ${deployments[*]}"
    
    # Rollback each deployment
    for deployment in "${deployments[@]}"; do
        if ! rollback_deployment "$deployment" "$revision"; then
            log_error "Failed to rollback deployment $deployment"
            return 1
        fi
    done
    
    return 0
}

# Display rollback options
show_rollback_options() {
    log_info "Available rollback options:"
    
    # Get deployment names
    local deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[*].metadata.name}')
    
    for deployment in $deployments; do
        echo ""
        echo "Deployment: $deployment"
        echo "Current revision: $(get_current_revision "$deployment")"
        echo "Rollout history:"
        get_deployment_history "$deployment" | head -5
    done
}

# Send rollback notification
send_notification() {
    local status="$1"
    local message="$2"
    
    if command -v curl &>/dev/null && [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local emoji
        case "$status" in
            "success") emoji="✅" ;;
            "warning") emoji="⚠️" ;;
            "error") emoji="❌" ;;
            *) emoji="ℹ️" ;;
        esac
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji Claude TUI Rollback ($ENVIRONMENT): $message\"}" \
            "$SLACK_WEBHOOK_URL" &>/dev/null || true
    fi
}

# Main function
main() {
    local action="${2:-auto}"
    local revision="${3:-}"
    
    log_info "Starting rollback process for $ENVIRONMENT environment..."
    log_info "Action: $action"
    
    # Show current status
    show_deployment_status
    
    case "$action" in
        "emergency")
            log_warning "Emergency rollback requested"
            if emergency_rollback; then
                send_notification "success" "Emergency rollback completed successfully"
                log_success "Emergency rollback completed successfully!"
            else
                send_notification "error" "Emergency rollback failed"
                log_error "Emergency rollback failed!"
                exit 1
            fi
            ;;
        "revision")
            if [[ -z "$revision" ]]; then
                log_error "Revision number required for revision rollback"
                show_rollback_options
                exit 1
            fi
            
            log_info "Rolling back to revision $revision"
            if standard_rollback "$revision"; then
                send_notification "success" "Rollback to revision $revision completed"
                log_success "Rollback to revision $revision completed!"
            else
                send_notification "error" "Rollback to revision $revision failed"
                log_error "Rollback to revision $revision failed!"
                exit 1
            fi
            ;;
        "show")
            show_rollback_options
            exit 0
            ;;
        "auto"|*)
            log_info "Automatic rollback to previous version"
            if standard_rollback; then
                send_notification "success" "Automatic rollback completed"
                log_success "Automatic rollback completed!"
            else
                send_notification "error" "Automatic rollback failed"
                log_error "Automatic rollback failed!"
                exit 1
            fi
            ;;
    esac
    
    # Validate rollback if enabled
    if [[ "$VALIDATE_ROLLBACK" == "true" ]]; then
        sleep 10  # Wait for services to stabilize
        if validate_rollback; then
            log_success "Rollback validation passed"
        else
            log_warning "Rollback validation failed - manual verification recommended"
        fi
    fi
    
    # Show final status
    echo ""
    log_info "Final deployment status:"
    show_deployment_status
}

# Help function
show_help() {
    echo "Usage: $0 <environment> <action> [revision]"
    echo ""
    echo "Arguments:"
    echo "  environment  Target environment (staging|production)"
    echo "  action       Rollback action:"
    echo "               - auto: Rollback to previous revision (default)"
    echo "               - emergency: Emergency rollback using blue-green switch"
    echo "               - revision: Rollback to specific revision"
    echo "               - show: Show rollback options"
    echo "  revision     Specific revision number (required for 'revision' action)"
    echo ""
    echo "Examples:"
    echo "  $0 production auto"
    echo "  $0 production emergency"
    echo "  $0 production revision 5"
    echo "  $0 production show"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE           Kubernetes namespace (default: claude-tui)"
    echo "  SERVICE_NAME        Service name (default: claude-tui)"
    echo "  ROLLBACK_TIMEOUT    Rollback timeout in seconds (default: 180)"
    echo "  VALIDATE_ROLLBACK   Validate after rollback (default: true)"
    echo "  SLACK_WEBHOOK_URL   Slack webhook for notifications"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    main "$@"
fi