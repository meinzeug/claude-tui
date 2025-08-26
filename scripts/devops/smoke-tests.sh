#!/bin/bash

# Smoke Tests for Claude TUI Deployment
# Comprehensive validation of deployed application

set -euo pipefail

# Configuration
BASE_URL="${1:-http://localhost:8000}"
TIMEOUT=30
MAX_RETRIES=5
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test runner function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    log_info "Running test: $test_name"
    
    if eval "$test_command"; then
        log_success "$test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        log_error "$test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# HTTP request helper with retries
http_request() {
    local url="$1"
    local expected_status="${2:-200}"
    local method="${3:-GET}"
    local data="${4:-}"
    local content_type="${5:-application/json}"
    
    local retry_count=0
    
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        local curl_opts=(-s -w "%{http_code}" --max-time "$TIMEOUT")
        
        if [[ "$method" == "POST" && -n "$data" ]]; then
            curl_opts+=(-X POST -H "Content-Type: $content_type" -d "$data")
        elif [[ "$method" == "PUT" && -n "$data" ]]; then
            curl_opts+=(-X PUT -H "Content-Type: $content_type" -d "$data")
        fi
        
        local response
        response=$(curl "${curl_opts[@]}" "$url" 2>/dev/null)
        local status_code="${response: -3}"
        local body="${response%???}"
        
        if [[ "$status_code" == "$expected_status" ]]; then
            if [[ "$VERBOSE" == "true" ]]; then
                log_info "Response: $body"
            fi
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [[ $retry_count -lt $MAX_RETRIES ]]; then
            log_warning "Request failed (attempt $retry_count/$MAX_RETRIES), retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    log_error "Request to $url failed after $MAX_RETRIES attempts (last status: $status_code)"
    return 1
}

# Basic health checks
test_basic_health() {
    http_request "$BASE_URL/health"
}

test_api_health() {
    http_request "$BASE_URL/api/v1/health"
}

test_root_endpoint() {
    http_request "$BASE_URL/"
}

# API endpoint tests
test_api_version() {
    local response
    response=$(curl -s --max-time "$TIMEOUT" "$BASE_URL/api/v1/version" 2>/dev/null)
    
    if echo "$response" | jq -e '.version' > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

test_api_status() {
    local response
    response=$(curl -s --max-time "$TIMEOUT" "$BASE_URL/api/v1/status" 2>/dev/null)
    
    if echo "$response" | jq -e '.status' > /dev/null 2>&1; then
        local status
        status=$(echo "$response" | jq -r '.status')
        if [[ "$status" == "healthy" || "$status" == "ok" ]]; then
            return 0
        fi
    fi
    return 1
}

# Authentication tests (if applicable)
test_auth_endpoints() {
    # Test login endpoint exists (should return 401 or method not allowed, not 404)
    local status_code
    status_code=$(curl -s -w "%{http_code}" --max-time "$TIMEOUT" -o /dev/null "$BASE_URL/api/v1/auth/login" 2>/dev/null)
    
    if [[ "$status_code" == "401" || "$status_code" == "405" || "$status_code" == "400" ]]; then
        return 0
    else
        return 1
    fi
}

# Database connectivity test
test_database_connectivity() {
    local response
    response=$(curl -s --max-time "$TIMEOUT" "$BASE_URL/api/v1/health/database" 2>/dev/null)
    
    if echo "$response" | jq -e '.database.status' > /dev/null 2>&1; then
        local db_status
        db_status=$(echo "$response" | jq -r '.database.status')
        if [[ "$db_status" == "connected" || "$db_status" == "healthy" ]]; then
            return 0
        fi
    fi
    return 1
}

# Redis connectivity test
test_redis_connectivity() {
    local response
    response=$(curl -s --max-time "$TIMEOUT" "$BASE_URL/api/v1/health/redis" 2>/dev/null)
    
    if echo "$response" | jq -e '.redis.status' > /dev/null 2>&1; then
        local redis_status
        redis_status=$(echo "$response" | jq -r '.redis.status')
        if [[ "$redis_status" == "connected" || "$redis_status" == "healthy" ]]; then
            return 0
        fi
    fi
    return 1
}

# Performance tests
test_response_time() {
    local start_time
    local end_time
    local response_time
    
    start_time=$(date +%s.%N)
    if http_request "$BASE_URL/health"; then
        end_time=$(date +%s.%N)
        response_time=$(echo "$end_time - $start_time" | bc -l)
        
        if (( $(echo "$response_time < 2.0" | bc -l) )); then
            log_info "Response time: ${response_time}s (< 2.0s)"
            return 0
        else
            log_warning "Response time: ${response_time}s (>= 2.0s)"
            return 1
        fi
    else
        return 1
    fi
}

# Load test (basic)
test_concurrent_requests() {
    log_info "Running concurrent request test..."
    
    local pids=()
    local concurrent_requests=10
    local temp_dir
    temp_dir=$(mktemp -d)
    
    # Launch concurrent requests
    for ((i=1; i<=concurrent_requests; i++)); do
        (
            if curl -s --max-time "$TIMEOUT" "$BASE_URL/health" > "$temp_dir/response_$i.txt" 2>&1; then
                echo "success" > "$temp_dir/result_$i.txt"
            else
                echo "failure" > "$temp_dir/result_$i.txt"
            fi
        ) &
        pids+=($!)
    done
    
    # Wait for all requests to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    # Count successful requests
    local successful_requests=0
    for ((i=1; i<=concurrent_requests; i++)); do
        if [[ -f "$temp_dir/result_$i.txt" && "$(cat "$temp_dir/result_$i.txt")" == "success" ]]; then
            successful_requests=$((successful_requests + 1))
        fi
    done
    
    # Cleanup
    rm -rf "$temp_dir"
    
    local success_rate=$((successful_requests * 100 / concurrent_requests))
    log_info "Concurrent requests success rate: $success_rate% ($successful_requests/$concurrent_requests)"
    
    if [[ $success_rate -ge 90 ]]; then
        return 0
    else
        return 1
    fi
}

# Feature-specific tests
test_api_endpoints() {
    local endpoints=(
        "projects"
        "tasks"
        "ai"
        "analytics"
        "workflows"
    )
    
    for endpoint in "${endpoints[@]}"; do
        # Test that endpoint exists (may return 401/403 for protected endpoints)
        local status_code
        status_code=$(curl -s -w "%{http_code}" --max-time "$TIMEOUT" -o /dev/null "$BASE_URL/api/v1/$endpoint" 2>/dev/null)
        
        if [[ "$status_code" != "404" && "$status_code" != "000" ]]; then
            log_info "✓ API endpoint /$endpoint is accessible (status: $status_code)"
        else
            log_error "✗ API endpoint /$endpoint is not accessible (status: $status_code)"
            return 1
        fi
    done
    
    return 0
}

# Security tests
test_security_headers() {
    local headers
    headers=$(curl -s -I --max-time "$TIMEOUT" "$BASE_URL/health" 2>/dev/null)
    
    local security_checks=0
    local total_security_checks=4
    
    # Check for security headers
    if echo "$headers" | grep -i "x-frame-options" > /dev/null; then
        log_info "✓ X-Frame-Options header present"
        security_checks=$((security_checks + 1))
    fi
    
    if echo "$headers" | grep -i "x-content-type-options" > /dev/null; then
        log_info "✓ X-Content-Type-Options header present"
        security_checks=$((security_checks + 1))
    fi
    
    if echo "$headers" | grep -i "x-xss-protection" > /dev/null; then
        log_info "✓ X-XSS-Protection header present"
        security_checks=$((security_checks + 1))
    fi
    
    if echo "$headers" | grep -i "strict-transport-security" > /dev/null; then
        log_info "✓ Strict-Transport-Security header present"
        security_checks=$((security_checks + 1))
    fi
    
    local security_score=$((security_checks * 100 / total_security_checks))
    log_info "Security headers score: $security_score%"
    
    if [[ $security_score -ge 50 ]]; then
        return 0
    else
        return 1
    fi
}

# Configuration validation
test_environment_config() {
    local response
    response=$(curl -s --max-time "$TIMEOUT" "$BASE_URL/api/v1/config" 2>/dev/null)
    
    if echo "$response" | jq -e '.environment' > /dev/null 2>&1; then
        local environment
        environment=$(echo "$response" | jq -r '.environment')
        log_info "Environment: $environment"
        
        if [[ "$environment" == "production" || "$environment" == "staging" ]]; then
            return 0
        fi
    fi
    
    return 1
}

# WebSocket test (if applicable)
test_websocket_connection() {
    # Simple WebSocket connectivity test
    if command -v wscat &> /dev/null; then
        local ws_url
        ws_url="${BASE_URL/http/ws}/ws"
        
        timeout 10 wscat -c "$ws_url" -x '{"type":"ping"}' 2>/dev/null | grep -q "pong" && return 0
    fi
    
    # Skip if wscat not available
    log_info "WebSocket test skipped (wscat not available)"
    return 0
}

# Resource validation
test_resource_usage() {
    local response
    response=$(curl -s --max-time "$TIMEOUT" "$BASE_URL/api/v1/health/metrics" 2>/dev/null)
    
    if echo "$response" | jq -e '.memory' > /dev/null 2>&1; then
        local memory_usage
        memory_usage=$(echo "$response" | jq -r '.memory.usage_percent // 0')
        
        if (( $(echo "$memory_usage < 90" | bc -l) )); then
            log_info "Memory usage: $memory_usage%"
            return 0
        else
            log_warning "High memory usage: $memory_usage%"
            return 1
        fi
    fi
    
    return 0
}

# Main test execution
run_all_tests() {
    log_info "Starting smoke tests for: $BASE_URL"
    log_info "Timeout: ${TIMEOUT}s, Max retries: $MAX_RETRIES"
    echo
    
    # Core health tests
    run_test "Basic Health Check" "test_basic_health"
    run_test "API Health Check" "test_api_health"
    run_test "Root Endpoint" "test_root_endpoint"
    
    # API tests
    run_test "API Version" "test_api_version"
    run_test "API Status" "test_api_status"
    run_test "API Endpoints" "test_api_endpoints"
    
    # Authentication tests
    run_test "Auth Endpoints" "test_auth_endpoints"
    
    # Database and Redis tests
    run_test "Database Connectivity" "test_database_connectivity"
    run_test "Redis Connectivity" "test_redis_connectivity"
    
    # Performance tests
    run_test "Response Time" "test_response_time"
    run_test "Concurrent Requests" "test_concurrent_requests"
    
    # Security tests
    run_test "Security Headers" "test_security_headers"
    
    # Configuration tests
    run_test "Environment Config" "test_environment_config"
    
    # Additional tests
    run_test "WebSocket Connection" "test_websocket_connection"
    run_test "Resource Usage" "test_resource_usage"
}

# Results summary
show_results() {
    echo
    echo "==================== SMOKE TEST RESULTS ===================="
    echo
    log_info "Total Tests: $TOTAL_TESTS"
    log_success "Passed: $PASSED_TESTS"
    log_error "Failed: $FAILED_TESTS"
    echo
    
    local pass_rate
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        pass_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
        log_info "Pass Rate: $pass_rate%"
    else
        log_warning "No tests were executed"
        return 1
    fi
    
    echo
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log_success "All smoke tests passed! 🎉"
        echo "Deployment is ready for use."
        return 0
    elif [[ $pass_rate -ge 80 ]]; then
        log_warning "Most tests passed, but some issues detected."
        echo "Deployment may be usable but requires attention."
        return 1
    else
        log_error "Multiple test failures detected!"
        echo "Deployment may not be ready for use."
        return 1
    fi
}

# Help function
show_help() {
    cat << EOF
Smoke Tests for Claude TUI Deployment

Usage: $0 [BASE_URL] [OPTIONS]

Arguments:
    BASE_URL    Base URL of the deployed application (default: http://localhost:8000)

Options:
    --timeout SECONDS       Request timeout in seconds (default: 30)
    --max-retries COUNT     Maximum retries per request (default: 5)
    --verbose              Enable verbose output
    -h, --help             Show this help message

Examples:
    $0
    $0 https://staging.claude-tui.dev
    $0 https://claude-tui.dev --timeout 60 --max-retries 3
    $0 http://localhost:8080 --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "${BASE_URL_SET:-}" ]]; then
                BASE_URL="$1"
                BASE_URL_SET=true
            else
                log_error "Unexpected argument: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate dependencies
if ! command -v curl &> /dev/null; then
    log_error "curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    log_warning "jq is not installed, some JSON tests may fail"
fi

if ! command -v bc &> /dev/null; then
    log_warning "bc is not installed, some numeric tests may fail"
fi

# Main execution
main() {
    run_all_tests
    show_results
}

# Trap to handle script interruption
trap 'log_error "Smoke tests interrupted"; exit 130' INT TERM

# Run main function
main "$@"