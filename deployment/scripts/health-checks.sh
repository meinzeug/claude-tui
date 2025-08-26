#!/bin/bash

# Comprehensive Health Check Script for Claude TUI
# Validates application health, dependencies, and performance

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-claude-tui}"
SERVICE_NAME="${SERVICE_NAME:-claude-tui}"
TIMEOUT="${TIMEOUT:-60}"
VERBOSE="${VERBOSE:-false}"

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

# Get environment from argument
ENVIRONMENT="${1:-staging}"

# Health check functions
check_kubernetes_resources() {
    log_info "Checking Kubernetes resources..."
    
    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_error "Namespace $NAMESPACE not found"
        return 1
    fi
    
    # Check deployments
    local deployments=$(kubectl get deployments -n "$NAMESPACE" --no-headers | wc -l)
    if [[ $deployments -eq 0 ]]; then
        log_error "No deployments found in namespace $NAMESPACE"
        return 1
    fi
    
    # Check deployment status
    local ready_deployments=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].status.conditions[?(@.type=="Available")].status}' | tr ' ' '\n' | grep -c "True" || echo "0")
    
    if [[ $ready_deployments -eq 0 ]]; then
        log_error "No ready deployments found"
        return 1
    fi
    
    # Check pods
    local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui --no-headers | wc -l)
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | tr ' ' '\n' | grep -c "True" || echo "0")
    
    if [[ $ready_pods -eq 0 ]]; then
        log_error "No ready pods found"
        return 1
    fi
    
    log_success "Kubernetes resources: $ready_deployments/$deployments deployments ready, $ready_pods/$total_pods pods ready"
    return 0
}

check_service_endpoints() {
    log_info "Checking service endpoints..."
    
    # Get service endpoint
    local service_exists=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" &>/dev/null && echo "true" || echo "false")
    
    if [[ "$service_exists" == "false" ]]; then
        log_error "Service $SERVICE_NAME not found"
        return 1
    fi
    
    # Check if service has endpoints
    local endpoints=$(kubectl get endpoints "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
    
    if [[ $endpoints -eq 0 ]]; then
        log_error "Service has no endpoints"
        return 1
    fi
    
    log_success "Service endpoints: $endpoints active"
    return 0
}

check_application_health() {
    log_info "Checking application health endpoints..."
    
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$pod_name" ]]; then
        log_error "No pods found for health check"
        return 1
    fi
    
    # Health check endpoint
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- curl -f -s http://localhost:8080/health &>/dev/null; then
        log_success "Health endpoint responsive"
    else
        log_error "Health endpoint failed"
        return 1
    fi
    
    # Readiness check endpoint
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- curl -f -s http://localhost:8080/ready &>/dev/null; then
        log_success "Readiness endpoint responsive"
    else
        log_error "Readiness endpoint failed"
        return 1
    fi
    
    # API endpoint check
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- curl -f -s http://localhost:8080/api/v1/health &>/dev/null; then
        log_success "API endpoint responsive"
    else
        log_warning "API endpoint not responsive (may be expected)"
    fi
    
    return 0
}

check_database_connectivity() {
    log_info "Checking database connectivity..."
    
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$pod_name" ]]; then
        log_error "No pods found for database check"
        return 1
    fi
    
    # Database connectivity check
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- python3 -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" &>/dev/null; then
        log_success "Database connectivity verified"
    else
        log_error "Database connectivity failed"
        return 1
    fi
    
    return 0
}

check_redis_connectivity() {
    log_info "Checking Redis connectivity..."
    
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$pod_name" ]]; then
        log_error "No pods found for Redis check"
        return 1
    fi
    
    # Redis connectivity check
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- python3 -c "
import os
import redis
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
" &>/dev/null; then
        log_success "Redis connectivity verified"
    else
        log_error "Redis connectivity failed"
        return 1
    fi
    
    return 0
}

check_external_dependencies() {
    log_info "Checking external dependencies..."
    
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$pod_name" ]]; then
        log_error "No pods found for external dependency check"
        return 1
    fi
    
    # Check Claude API connectivity (if API key is available)
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- curl -s -o /dev/null -w "%{http_code}" https://api.anthropic.com &>/dev/null; then
        log_success "Claude API reachable"
    else
        log_warning "Claude API not reachable (may be network policy)"
    fi
    
    # Check Claude Flow connectivity
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- curl -s -o /dev/null -w "%{http_code}" https://api.claude-flow.com &>/dev/null; then
        log_success "Claude Flow API reachable"
    else
        log_warning "Claude Flow API not reachable (may be network policy)"
    fi
    
    return 0
}

check_resource_usage() {
    log_info "Checking resource usage..."
    
    local pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[*].metadata.name}')
    
    for pod in $pods; do
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Resource usage for pod $pod:"
            kubectl top pod "$pod" -n "$NAMESPACE" --no-headers | awk '{print "  CPU: " $2 ", Memory: " $3}'
        fi
        
        # Check if pod is consuming excessive resources (basic threshold check)
        local cpu_usage=$(kubectl top pod "$pod" -n "$NAMESPACE" --no-headers | awk '{print $2}' | sed 's/m//' || echo "0")
        local memory_usage=$(kubectl top pod "$pod" -n "$NAMESPACE" --no-headers | awk '{print $3}' | sed 's/Mi//' || echo "0")
        
        # Basic thresholds (adjust as needed)
        if [[ $cpu_usage -gt 1500 ]]; then
            log_warning "High CPU usage detected: ${cpu_usage}m"
        fi
        
        if [[ $memory_usage -gt 800 ]]; then
            log_warning "High memory usage detected: ${memory_usage}Mi"
        fi
    done
    
    log_success "Resource usage check completed"
    return 0
}

check_logs_for_errors() {
    log_info "Checking recent logs for errors..."
    
    local pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[*].metadata.name}')
    local error_count=0
    
    for pod in $pods; do
        # Check for ERROR level logs in the last 5 minutes
        local errors=$(kubectl logs "$pod" -n "$NAMESPACE" --since=5m | grep -i "error\|exception\|fatal" | wc -l || echo "0")
        
        if [[ $errors -gt 0 ]]; then
            ((error_count += errors))
            if [[ "$VERBOSE" == "true" ]]; then
                log_warning "Found $errors errors in pod $pod"
                kubectl logs "$pod" -n "$NAMESPACE" --since=5m | grep -i "error\|exception\|fatal" | head -5
            fi
        fi
    done
    
    if [[ $error_count -gt 10 ]]; then
        log_warning "High error count in logs: $error_count errors"
        return 1
    elif [[ $error_count -gt 0 ]]; then
        log_warning "Some errors found in logs: $error_count errors"
    else
        log_success "No critical errors found in recent logs"
    fi
    
    return 0
}

run_smoke_tests() {
    log_info "Running basic smoke tests..."
    
    # Get service endpoint for external testing
    local service_type=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.type}')
    local service_endpoint=""
    
    if [[ "$service_type" == "LoadBalancer" ]]; then
        service_endpoint=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    fi
    
    if [[ -z "$service_endpoint" ]]; then
        # Use port-forward for testing
        kubectl port-forward service/"$SERVICE_NAME" -n "$NAMESPACE" 8080:8080 &
        local port_forward_pid=$!
        sleep 5
        service_endpoint="localhost:8080"
    fi
    
    # Basic smoke tests
    local tests_passed=0
    local tests_total=3
    
    # Test 1: Health endpoint
    if curl -f -s "http://$service_endpoint/health" &>/dev/null; then
        log_success "Smoke test 1/3 passed: Health endpoint"
        ((tests_passed++))
    else
        log_error "Smoke test 1/3 failed: Health endpoint"
    fi
    
    # Test 2: API version endpoint
    if curl -f -s "http://$service_endpoint/api/v1/version" &>/dev/null; then
        log_success "Smoke test 2/3 passed: API version endpoint"
        ((tests_passed++))
    else
        log_warning "Smoke test 2/3 failed: API version endpoint (may be expected)"
    fi
    
    # Test 3: Basic performance test
    local response_time=$(curl -w "%{time_total}" -s -o /dev/null "http://$service_endpoint/health" 2>/dev/null || echo "999")
    if (( $(echo "$response_time < 2.0" | bc -l) )); then
        log_success "Smoke test 3/3 passed: Response time ($response_time seconds)"
        ((tests_passed++))
    else
        log_error "Smoke test 3/3 failed: Response time too slow ($response_time seconds)"
    fi
    
    # Clean up port-forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill $port_forward_pid &>/dev/null || true
    fi
    
    if [[ $tests_passed -eq $tests_total ]]; then
        log_success "All smoke tests passed ($tests_passed/$tests_total)"
        return 0
    else
        log_warning "Some smoke tests failed ($tests_passed/$tests_total)"
        return 1
    fi
}

# Main health check function
main() {
    log_info "Starting comprehensive health checks for $ENVIRONMENT environment..."
    
    local checks_passed=0
    local checks_total=8
    local critical_failures=0
    
    # Run all health checks
    if check_kubernetes_resources; then ((checks_passed++)); else ((critical_failures++)); fi
    if check_service_endpoints; then ((checks_passed++)); else ((critical_failures++)); fi
    if check_application_health; then ((checks_passed++)); else ((critical_failures++)); fi
    if check_database_connectivity; then ((checks_passed++)); else ((critical_failures++)); fi
    if check_redis_connectivity; then ((checks_passed++)); fi
    if check_external_dependencies; then ((checks_passed++)); fi
    if check_resource_usage; then ((checks_passed++)); fi
    if check_logs_for_errors; then ((checks_passed++)); fi
    
    # Run smoke tests separately
    local smoke_tests_passed=false
    if run_smoke_tests; then
        smoke_tests_passed=true
    fi
    
    # Summary
    echo ""
    log_info "Health Check Summary:"
    echo "  Environment: $ENVIRONMENT"
    echo "  Health checks passed: $checks_passed/$checks_total"
    echo "  Critical failures: $critical_failures"
    echo "  Smoke tests: $([ "$smoke_tests_passed" = true ] && echo "PASSED" || echo "FAILED")"
    
    # Determine overall result
    if [[ $critical_failures -eq 0 && "$smoke_tests_passed" = true ]]; then
        log_success "All health checks passed! System is healthy."
        exit 0
    elif [[ $critical_failures -eq 0 ]]; then
        log_warning "Health checks mostly passed, but some smoke tests failed."
        exit 1
    else
        log_error "Critical health check failures detected!"
        exit 2
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi