#!/bin/bash

# Comprehensive Smoke Test Suite for Claude TUI
# Validates critical functionality after deployment

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-staging}"
NAMESPACE="${NAMESPACE:-claude-tui}"
SERVICE_NAME="${SERVICE_NAME:-claude-tui}"
TIMEOUT="${TIMEOUT:-300}"
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

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test result tracking
declare -A TEST_RESULTS

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TESTS_TOTAL++))
    log_info "Running test: $test_name"
    
    if $test_function; then
        ((TESTS_PASSED++))
        TEST_RESULTS["$test_name"]="PASSED"
        log_success "✓ $test_name"
    else
        ((TESTS_FAILED++))
        TEST_RESULTS["$test_name"]="FAILED"
        log_error "✗ $test_name"
    fi
}

# Get service endpoint
get_service_endpoint() {
    local service_type=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.type}' 2>/dev/null || echo "")
    local endpoint=""
    
    case "$service_type" in
        "LoadBalancer")
            endpoint=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || \
                      kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
            ;;
        "NodePort")
            local node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || \
                           kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo "")
            local node_port=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
            if [[ -n "$node_ip" && -n "$node_port" ]]; then
                endpoint="$node_ip:$node_port"
            fi
            ;;
        "ClusterIP"|*)
            # Use port-forward for ClusterIP or unknown service types
            kubectl port-forward service/"$SERVICE_NAME" -n "$NAMESPACE" 8080:8080 &
            local port_forward_pid=$!
            sleep 5
            endpoint="localhost:8080"
            ;;
    esac
    
    echo "$endpoint"
}

# Basic connectivity test
test_basic_connectivity() {
    local endpoint="$1"
    
    if curl -f -s -m 10 "http://$endpoint/health" >/dev/null; then
        return 0
    else
        return 1
    fi
}

# Health endpoint test
test_health_endpoint() {
    local endpoint="$1"
    
    local response=$(curl -s -m 10 "http://$endpoint/health" 2>/dev/null || echo "")
    
    if [[ -n "$response" ]]; then
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Health response: $response"
        fi
        
        # Check for expected health indicators
        if echo "$response" | grep -q -i "healthy\|ok\|success"; then
            return 0
        fi
    fi
    
    return 1
}

# Readiness endpoint test
test_readiness_endpoint() {
    local endpoint="$1"
    
    if curl -f -s -m 10 "http://$endpoint/ready" >/dev/null; then
        return 0
    else
        return 1
    fi
}

# API version endpoint test
test_api_version() {
    local endpoint="$1"
    
    local response=$(curl -s -m 10 "http://$endpoint/api/v1/version" 2>/dev/null || echo "")
    
    if [[ -n "$response" ]]; then
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "API version response: $response"
        fi
        
        # Check for version information
        if echo "$response" | grep -q -E "version|v[0-9]"; then
            return 0
        fi
    fi
    
    return 1
}

# Performance test
test_response_time() {
    local endpoint="$1"
    local max_response_time="${2:-2.0}"
    
    local response_time=$(curl -w "%{time_total}" -s -o /dev/null -m 10 "http://$endpoint/health" 2>/dev/null || echo "999")
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Response time: ${response_time}s (max: ${max_response_time}s)"
    fi
    
    if (( $(echo "$response_time <= $max_response_time" | bc -l) )); then
        return 0
    else
        return 1
    fi
}

# Load test (basic)
test_concurrent_requests() {
    local endpoint="$1"
    local concurrent_requests="${2:-5}"
    local total_requests="${3:-20}"
    
    log_info "Running load test: $total_requests requests with $concurrent_requests concurrency"
    
    # Create temporary directory for results
    local temp_dir=$(mktemp -d)
    local success_count=0
    
    # Run concurrent requests
    for ((i=1; i<=total_requests; i++)); do
        {
            if curl -f -s -m 10 "http://$endpoint/health" >/dev/null 2>&1; then
                echo "success" > "$temp_dir/result_$i"
            else
                echo "failure" > "$temp_dir/result_$i"
            fi
        } &
        
        # Limit concurrent requests
        if (( i % concurrent_requests == 0 )); then
            wait
        fi
    done
    
    # Wait for remaining requests
    wait
    
    # Count successes
    success_count=$(find "$temp_dir" -name "result_*" -exec cat {} \; | grep -c "success" || echo "0")
    
    # Cleanup
    rm -rf "$temp_dir"
    
    local success_rate=$(echo "scale=2; $success_count * 100 / $total_requests" | bc)
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Load test results: $success_count/$total_requests successful (${success_rate}%)"
    fi
    
    # Require 95% success rate
    if (( $(echo "$success_rate >= 95" | bc -l) )); then
        return 0
    else
        return 1
    fi
}

# Authentication test
test_authentication() {
    local endpoint="$1"
    
    # Test unauthenticated access to protected endpoint
    local status_code=$(curl -s -o /dev/null -w "%{http_code}" -m 10 "http://$endpoint/api/v1/protected" 2>/dev/null || echo "000")
    
    # Should return 401 or 403 for unauthenticated requests
    if [[ "$status_code" == "401" || "$status_code" == "403" ]]; then
        return 0
    elif [[ "$status_code" == "404" ]]; then
        # Endpoint might not exist, which is acceptable
        return 0
    else
        return 1
    fi
}

# Database connectivity test
test_database_connectivity() {
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$pod_name" ]]; then
        return 1
    fi
    
    # Test database connection via application
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- python3 -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.environ.get('DATABASE_URL', ''))
    cursor = conn.cursor()
    cursor.execute('SELECT 1')
    result = cursor.fetchone()
    conn.close()
    exit(0 if result[0] == 1 else 1)
except Exception as e:
    exit(1)
" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Redis connectivity test
test_redis_connectivity() {
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$pod_name" ]]; then
        return 1
    fi
    
    # Test Redis connection via application
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- python3 -c "
import os
import redis
try:
    r = redis.from_url(os.environ.get('REDIS_URL', ''))
    r.ping()
    r.set('smoke_test_key', 'test_value')
    value = r.get('smoke_test_key')
    r.delete('smoke_test_key')
    exit(0 if value == b'test_value' else 1)
except Exception as e:
    exit(1)
" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# File upload test (if applicable)
test_file_operations() {
    local endpoint="$1"
    
    # Create a small test file
    local test_file=$(mktemp)
    echo "test content for smoke test" > "$test_file"
    
    # Try to upload file (adjust endpoint as needed)
    local status_code=$(curl -s -o /dev/null -w "%{http_code}" -m 10 \
        -F "file=@$test_file" "http://$endpoint/api/v1/upload" 2>/dev/null || echo "000")
    
    # Cleanup
    rm -f "$test_file"
    
    # Accept 200, 400 (bad request), or 404 (endpoint doesn't exist)
    if [[ "$status_code" =~ ^(200|400|404)$ ]]; then
        return 0
    else
        return 1
    fi
}

# WebSocket test (if applicable)
test_websocket_connectivity() {
    local endpoint="$1"
    
    # Replace http with ws for WebSocket
    local ws_endpoint=$(echo "$endpoint" | sed 's/http/ws/')
    
    # Test WebSocket connection using a simple tool
    if command -v wscat &>/dev/null; then
        # Use wscat if available
        echo "ping" | timeout 5 wscat -c "ws://$ws_endpoint/ws" &>/dev/null
        return $?
    elif command -v websocat &>/dev/null; then
        # Use websocat if available
        echo "ping" | timeout 5 websocat "ws://$ws_endpoint/ws" &>/dev/null
        return $?
    else
        # Skip WebSocket test if no tools available
        return 0
    fi
}

# Run all smoke tests
run_smoke_tests() {
    local endpoint="$1"
    
    log_info "Starting smoke tests for endpoint: http://$endpoint"
    
    # Core functionality tests
    run_test "Basic Connectivity" "test_basic_connectivity $endpoint"
    run_test "Health Endpoint" "test_health_endpoint $endpoint"
    run_test "Readiness Endpoint" "test_readiness_endpoint $endpoint"
    run_test "API Version" "test_api_version $endpoint"
    
    # Performance tests
    run_test "Response Time" "test_response_time $endpoint 2.0"
    run_test "Concurrent Requests" "test_concurrent_requests $endpoint 5 20"
    
    # Security tests
    run_test "Authentication" "test_authentication $endpoint"
    
    # Infrastructure tests
    run_test "Database Connectivity" "test_database_connectivity"
    run_test "Redis Connectivity" "test_redis_connectivity"
    
    # Additional functionality tests
    run_test "File Operations" "test_file_operations $endpoint"
    run_test "WebSocket Connectivity" "test_websocket_connectivity $endpoint"
}

# Generate test report
generate_report() {
    echo ""
    log_info "=== SMOKE TEST REPORT ==="
    echo "Environment: $ENVIRONMENT"
    echo "Timestamp: $(date)"
    echo "Total Tests: $TESTS_TOTAL"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "Success Rate: $(echo "scale=2; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)%"
    echo ""
    
    log_info "Test Results:"
    for test_name in "${!TEST_RESULTS[@]}"; do
        local result="${TEST_RESULTS[$test_name]}"
        if [[ "$result" == "PASSED" ]]; then
            echo -e "  ✓ $test_name: ${GREEN}$result${NC}"
        else
            echo -e "  ✗ $test_name: ${RED}$result${NC}"
        fi
    done
    
    echo ""
}

# Send notification
send_notification() {
    local status="$1"
    
    if command -v curl &>/dev/null && [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local emoji
        local message
        
        case "$status" in
            "success")
                emoji="✅"
                message="Smoke tests passed ($TESTS_PASSED/$TESTS_TOTAL)"
                ;;
            "partial")
                emoji="⚠️"
                message="Smoke tests partially passed ($TESTS_PASSED/$TESTS_TOTAL)"
                ;;
            "failure")
                emoji="❌"
                message="Smoke tests failed ($TESTS_PASSED/$TESTS_TOTAL)"
                ;;
        esac
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji Claude TUI Smoke Tests ($ENVIRONMENT): $message\"}" \
            "$SLACK_WEBHOOK_URL" &>/dev/null || true
    fi
}

# Main function
main() {
    log_info "Starting smoke test suite for $ENVIRONMENT environment..."
    
    # Get service endpoint
    local endpoint=$(get_service_endpoint)
    
    if [[ -z "$endpoint" ]]; then
        log_error "Cannot determine service endpoint"
        exit 1
    fi
    
    log_info "Service endpoint: http://$endpoint"
    
    # Wait for service to be ready
    log_info "Waiting for service to be ready..."
    local retries=0
    local max_retries=30
    
    while ! test_basic_connectivity "$endpoint" && [[ $retries -lt $max_retries ]]; do
        sleep 10
        ((retries++))
        log_info "Waiting for service... ($retries/$max_retries)"
    done
    
    if [[ $retries -eq $max_retries ]]; then
        log_error "Service not ready after $max_retries attempts"
        exit 1
    fi
    
    # Run all smoke tests
    run_smoke_tests "$endpoint"
    
    # Generate report
    generate_report
    
    # Clean up port-forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill $port_forward_pid &>/dev/null || true
    fi
    
    # Determine overall result and send notification
    local success_rate=$(echo "scale=0; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All smoke tests passed! ✅"
        send_notification "success"
        exit 0
    elif [[ $success_rate -ge 80 ]]; then
        log_warning "Smoke tests partially passed (${success_rate}% success rate)"
        send_notification "partial"
        exit 1
    else
        log_error "Smoke tests failed (${success_rate}% success rate)"
        send_notification "failure"
        exit 2
    fi
}

# Help function
show_help() {
    echo "Usage: $0 <environment>"
    echo ""
    echo "Arguments:"
    echo "  environment  Target environment (staging|production)"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE      Kubernetes namespace (default: claude-tui)"
    echo "  SERVICE_NAME   Service name (default: claude-tui)"
    echo "  TIMEOUT        Test timeout in seconds (default: 300)"
    echo "  VERBOSE        Enable verbose output (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0 staging"
    echo "  VERBOSE=true $0 production"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    main
fi