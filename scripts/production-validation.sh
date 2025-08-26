#!/bin/bash

# Production deployment validation script
# Validates Claude-TIU deployment after production rollout

set -euo pipefail

NAMESPACE=${NAMESPACE:-claude-tui-production}
SERVICE_NAME=${SERVICE_NAME:-claude-tui-service}
TIMEOUT=${TIMEOUT:-300}
RETRY_INTERVAL=${RETRY_INTERVAL:-10}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to wait for condition with timeout
wait_for_condition() {
    local condition=$1
    local description=$2
    local timeout=$3
    local interval=${4:-5}
    
    log "Waiting for $description..."
    local elapsed=0
    
    while ! eval "$condition" >/dev/null 2>&1; do
        if [ $elapsed -ge $timeout ]; then
            error "Timeout waiting for $description"
            return 1
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    log "$description completed successfully"
}

# Check kubectl connectivity
check_kubectl() {
    log "Checking kubectl connectivity..."
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        return 1
    fi
    log "kubectl connectivity verified"
}

# Validate namespace exists
check_namespace() {
    log "Checking namespace $NAMESPACE..."
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        error "Namespace $NAMESPACE does not exist"
        return 1
    fi
    log "Namespace $NAMESPACE exists"
}

# Check deployment status
check_deployment() {
    log "Checking deployment status..."
    
    # Wait for deployment to be available
    wait_for_condition \
        "kubectl get deployment claude-tui-app -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' | grep -q '[1-9]'" \
        "deployment to have ready replicas" \
        $TIMEOUT
    
    # Check rollout status
    if ! kubectl rollout status deployment/claude-tui-app -n "$NAMESPACE" --timeout=300s; then
        error "Deployment rollout failed"
        return 1
    fi
    
    # Verify minimum replicas
    local ready_replicas
    ready_replicas=$(kubectl get deployment claude-tui-app -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    
    if [ "$ready_replicas" -lt 3 ]; then
        error "Insufficient ready replicas: $ready_replicas (expected: ≥3)"
        return 1
    fi
    
    log "Deployment validation passed: $ready_replicas replicas ready"
}

# Check pod health
check_pods() {
    log "Checking pod health..."
    
    # Get all pods
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=claude-tui -o name)
    
    if [ -z "$pods" ]; then
        error "No pods found"
        return 1
    fi
    
    # Check each pod status
    local failed_pods=0
    for pod in $pods; do
        local pod_name
        pod_name=$(echo "$pod" | sed 's|pod/||')
        
        # Check pod is running
        local phase
        phase=$(kubectl get "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
        
        if [ "$phase" != "Running" ]; then
            error "Pod $pod_name is not running (phase: $phase)"
            ((failed_pods++))
            continue
        fi
        
        # Check all containers are ready
        local ready_containers
        local total_containers
        ready_containers=$(kubectl get "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}')
        total_containers=$(kubectl get "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[*].name}' | wc -w)
        
        if [ "$ready_containers" != "true" ]; then
            error "Pod $pod_name containers not ready"
            ((failed_pods++))
        fi
    done
    
    if [ $failed_pods -gt 0 ]; then
        error "$failed_pods pods failed health check"
        return 1
    fi
    
    log "All pods are healthy"
}

# Check service connectivity
check_service() {
    log "Checking service connectivity..."
    
    # Verify service exists
    if ! kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
        error "Service $SERVICE_NAME not found"
        return 1
    fi
    
    # Get service endpoint
    local endpoints
    endpoints=$(kubectl get endpoints "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.subsets[0].addresses[*].ip}')
    
    if [ -z "$endpoints" ]; then
        error "Service has no endpoints"
        return 1
    fi
    
    log "Service has endpoints: $endpoints"
}

# Health endpoint validation
check_health_endpoint() {
    log "Checking health endpoint..."
    
    # Port forward to test health endpoint
    local port=8080
    kubectl port-forward -n "$NAMESPACE" service/"$SERVICE_NAME" $port:80 >/dev/null 2>&1 &
    local pf_pid=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Test health endpoint
    local health_check_passed=false
    for i in {1..10}; do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            health_check_passed=true
            break
        fi
        sleep 2
    done
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
    
    if [ "$health_check_passed" = false ]; then
        error "Health endpoint check failed"
        return 1
    fi
    
    log "Health endpoint validation passed"
}

# Performance validation
check_performance() {
    log "Running performance validation..."
    
    # Port forward for performance tests
    local port=8081
    kubectl port-forward -n "$NAMESPACE" service/"$SERVICE_NAME" $port:80 >/dev/null 2>&1 &
    local pf_pid=$!
    
    sleep 5
    
    # Simple performance test
    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}\n' "http://localhost:$port/health" || echo "999")
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
    
    # Check if response time is under threshold (500ms = 0.5s)
    if (( $(echo "$response_time > 0.5" | bc -l) )); then
        warn "Response time $response_time seconds exceeds 500ms threshold"
        return 1
    fi
    
    log "Performance validation passed: ${response_time}s response time"
}

# Check HPA status
check_hpa() {
    log "Checking Horizontal Pod Autoscaler..."
    
    if ! kubectl get hpa claude-tui-hpa -n "$NAMESPACE" >/dev/null 2>&1; then
        warn "HPA not found"
        return 0
    fi
    
    # Check HPA is functioning
    local hpa_status
    hpa_status=$(kubectl get hpa claude-tui-hpa -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="AbleToScale")].status}')
    
    if [ "$hpa_status" != "True" ]; then
        warn "HPA is not able to scale"
        return 1
    fi
    
    log "HPA validation passed"
}

# Check resource usage
check_resources() {
    log "Checking resource usage..."
    
    # Get pods and check resource usage
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=claude-tui -o name)
    
    for pod in $pods; do
        local pod_name
        pod_name=$(echo "$pod" | sed 's|pod/||')
        
        # Get CPU and memory usage (requires metrics server)
        if kubectl top pod "$pod_name" -n "$NAMESPACE" >/dev/null 2>&1; then
            local usage
            usage=$(kubectl top pod "$pod_name" -n "$NAMESPACE" --no-headers)
            log "Pod $pod_name resource usage: $usage"
        else
            warn "Cannot get resource metrics for $pod_name (metrics server may not be available)"
        fi
    done
}

# Validate ingress (if enabled)
check_ingress() {
    log "Checking ingress configuration..."
    
    if kubectl get ingress -n "$NAMESPACE" >/dev/null 2>&1; then
        local ingress_ready
        ingress_ready=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}')
        
        if [ -n "$ingress_ready" ]; then
            log "Ingress is ready with IP: $ingress_ready"
        else
            warn "Ingress IP not yet assigned"
        fi
    else
        log "No ingress configured"
    fi
}

# Main validation function
main() {
    log "Starting Claude-TIU production deployment validation..."
    
    local validation_steps=(
        "check_kubectl"
        "check_namespace"
        "check_deployment"
        "check_pods"
        "check_service"
        "check_health_endpoint"
        "check_performance"
        "check_hpa"
        "check_resources"
        "check_ingress"
    )
    
    local failed_steps=0
    
    for step in "${validation_steps[@]}"; do
        if ! $step; then
            ((failed_steps++))
        fi
        echo # Add spacing between checks
    done
    
    if [ $failed_steps -eq 0 ]; then
        log "🎉 All validation checks passed! Claude-TIU is ready for production traffic."
        return 0
    else
        error "❌ $failed_steps validation checks failed. Please review and fix issues before serving production traffic."
        return 1
    fi
}

# Check for required tools
for tool in kubectl curl bc; do
    if ! command -v $tool >/dev/null 2>&1; then
        error "Required tool '$tool' is not installed"
        exit 1
    fi
done

# Run main validation
main "$@"