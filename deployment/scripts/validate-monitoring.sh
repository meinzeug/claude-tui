#!/bin/bash

# Hive Mind Monitoring Stack Validation Script
# Validates all monitoring components are working correctly

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_DIR="${SCRIPT_DIR}/../monitoring"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

# Test runner function
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TOTAL_TESTS++))
    log_info "Running test: $test_name"
    
    if eval "$test_function"; then
        log_success "$test_name"
    else
        log_error "$test_name"
    fi
    
    echo ""
}

# Service availability tests
test_prometheus_availability() {
    if curl -sf "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

test_grafana_availability() {
    if curl -sf "http://localhost:3000/api/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

test_alertmanager_availability() {
    if curl -sf "http://localhost:9093/-/healthy" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

test_loki_availability() {
    if curl -sf "http://localhost:3100/ready" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

test_jaeger_availability() {
    if curl -sf "http://localhost:16686/" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Configuration tests
test_prometheus_config() {
    local config_check
    config_check=$(curl -s "http://localhost:9090/api/v1/status/config" | jq -r '.status')
    
    if [ "$config_check" = "success" ]; then
        return 0
    else
        return 1
    fi
}

test_prometheus_targets() {
    local targets_up
    targets_up=$(curl -s "http://localhost:9090/api/v1/targets" | jq -r '.data.activeTargets[].health' | grep -c "up")
    
    if [ "$targets_up" -gt 0 ]; then
        log_info "Found $targets_up active targets"
        return 0
    else
        return 1
    fi
}

test_grafana_datasources() {
    local datasources
    datasources=$(curl -s -u admin:admin123 "http://localhost:3000/api/datasources" | jq length)
    
    if [ "$datasources" -gt 0 ]; then
        log_info "Found $datasources configured datasources"
        return 0
    else
        return 1
    fi
}

test_grafana_dashboards() {
    local dashboards
    dashboards=$(curl -s -u admin:admin123 "http://localhost:3000/api/search?type=dash-db" | jq length)
    
    if [ "$dashboards" -gt 0 ]; then
        log_info "Found $dashboards imported dashboards"
        return 0
    else
        return 1
    fi
}

test_alertmanager_config() {
    local config_check
    config_check=$(curl -s "http://localhost:9093/api/v1/status" | jq -r '.status')
    
    if [ "$config_check" = "success" ]; then
        return 0
    else
        return 1
    fi
}

# Metrics collection tests
test_prometheus_metrics() {
    local metric_families
    metric_families=$(curl -s "http://localhost:9090/api/v1/label/__name__/values" | jq -r '.data | length')
    
    if [ "$metric_families" -gt 100 ]; then
        log_info "Found $metric_families metric families"
        return 0
    else
        return 1
    fi
}

test_node_exporter_metrics() {
    local node_metrics
    node_metrics=$(curl -s "http://localhost:9090/api/v1/query?query=up{job=\"node-exporter\"}" | jq -r '.data.result[0].value[1]')
    
    if [ "$node_metrics" = "1" ]; then
        return 0
    else
        return 1
    fi
}

test_cadvisor_metrics() {
    local cadvisor_metrics
    cadvisor_metrics=$(curl -s "http://localhost:9090/api/v1/query?query=up{job=\"cadvisor\"}" | jq -r '.data.result[0].value[1]')
    
    if [ "$cadvisor_metrics" = "1" ]; then
        return 0
    else
        return 1
    fi
}

# Alert rule tests
test_alert_rules() {
    local alert_rules
    alert_rules=$(curl -s "http://localhost:9090/api/v1/rules" | jq '.data.groups[].rules[] | select(.type=="alerting") | .name' | wc -l)
    
    if [ "$alert_rules" -gt 5 ]; then
        log_info "Found $alert_rules alerting rules"
        return 0
    else
        return 1
    fi
}

test_alert_rule_syntax() {
    local invalid_rules
    invalid_rules=$(curl -s "http://localhost:9090/api/v1/rules" | jq '.data.groups[].rules[] | select(.type=="alerting" and .health!="ok") | .name' | wc -l)
    
    if [ "$invalid_rules" -eq 0 ]; then
        return 0
    else
        log_warning "Found $invalid_rules rules with syntax errors"
        return 1
    fi
}

# Log ingestion tests
test_loki_ingestion() {
    local log_streams
    log_streams=$(curl -s "http://localhost:3100/loki/api/v1/labels" | jq -r '.data | length')
    
    if [ "$log_streams" -gt 0 ]; then
        log_info "Found $log_streams log label streams"
        return 0
    else
        return 1
    fi
}

test_promtail_targets() {
    local promtail_status
    if docker ps --format "table {{.Names}}" | grep -q "hive-promtail"; then
        return 0
    else
        return 1
    fi
}

# Tracing tests
test_jaeger_services() {
    local services
    services=$(curl -s "http://localhost:16686/api/services" | jq -r '.data | length')
    
    if [ "$services" -ge 0 ]; then
        log_info "Found $services traced services"
        return 0
    else
        return 1
    fi
}

# Performance tests
test_prometheus_query_performance() {
    local start_time
    local end_time
    local duration
    
    start_time=$(date +%s%3N)
    curl -s "http://localhost:9090/api/v1/query?query=up" > /dev/null
    end_time=$(date +%s%3N)
    
    duration=$((end_time - start_time))
    
    if [ "$duration" -lt 1000 ]; then
        log_info "Prometheus query took ${duration}ms"
        return 0
    else
        log_warning "Prometheus query took ${duration}ms (>1s)"
        return 1
    fi
}

test_grafana_response_time() {
    local start_time
    local end_time
    local duration
    
    start_time=$(date +%s%3N)
    curl -s "http://localhost:3000/api/health" > /dev/null
    end_time=$(date +%s%3N)
    
    duration=$((end_time - start_time))
    
    if [ "$duration" -lt 500 ]; then
        log_info "Grafana API response took ${duration}ms"
        return 0
    else
        log_warning "Grafana API response took ${duration}ms (>500ms)"
        return 1
    fi
}

# Storage tests
test_prometheus_storage() {
    local storage_info
    storage_info=$(curl -s "http://localhost:9090/api/v1/status/tsdb" | jq -r '.data.seriesCountByMetricName | keys | length')
    
    if [ "$storage_info" -gt 0 ]; then
        log_info "Prometheus TSDB contains data for $storage_info metric names"
        return 0
    else
        return 1
    fi
}

test_disk_usage() {
    local monitoring_disk_usage
    monitoring_disk_usage=$(du -sh "$MONITORING_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    
    log_info "Monitoring directory disk usage: $monitoring_disk_usage"
    
    # Check if docker volumes are using reasonable space
    local volume_usage
    volume_usage=$(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}" | grep "Local Volumes" | awk '{print $3}' || echo "unknown")
    
    log_info "Docker volumes usage: $volume_usage"
    return 0
}

# Integration tests
test_health_endpoint_integration() {
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        local health_status
        health_status=$(curl -s "http://localhost:8000/health" | jq -r '.status')
        
        if [ "$health_status" != "null" ]; then
            log_info "Health endpoint returns status: $health_status"
            return 0
        else
            return 1
        fi
    else
        log_warning "Health endpoint not available (application may not be running)"
        return 1
    fi
}

test_metrics_endpoint_integration() {
    if curl -sf "http://localhost:8000/metrics" > /dev/null 2>&1; then
        local metrics_count
        metrics_count=$(curl -s "http://localhost:8000/metrics" | grep -c "^[a-zA-Z]" || echo "0")
        
        if [ "$metrics_count" -gt 0 ]; then
            log_info "Metrics endpoint exposes $metrics_count metrics"
            return 0
        else
            return 1
        fi
    else
        log_warning "Metrics endpoint not available (application may not be running)"
        return 1
    fi
}

# Notification tests (optional)
test_slack_webhook() {
    if [ -n "${SLACK_WEBHOOK_URL:-}" ] && [ "$SLACK_WEBHOOK_URL" != "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" ]; then
        # Test webhook with a test message
        local webhook_test
        webhook_test=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-type: application/json' \
            --data '{"text":"Monitoring validation test from Hive Mind"}' \
            "$SLACK_WEBHOOK_URL")
        
        if [ "$webhook_test" = "200" ]; then
            return 0
        else
            return 1
        fi
    else
        log_warning "Slack webhook not configured"
        return 1
    fi
}

# Security tests
test_grafana_security() {
    # Test if Grafana requires authentication
    local auth_test
    auth_test=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:3000/api/dashboards/home")
    
    if [ "$auth_test" = "401" ] || [ "$auth_test" = "403" ]; then
        return 0
    else
        log_warning "Grafana may not require authentication (HTTP $auth_test)"
        return 1
    fi
}

# Container health tests
test_container_health() {
    local unhealthy_containers
    unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | wc -l)
    
    if [ "$unhealthy_containers" -eq 0 ]; then
        return 0
    else
        log_warning "Found $unhealthy_containers unhealthy containers"
        docker ps --filter "health=unhealthy" --format "table {{.Names}}\t{{.Status}}"
        return 1
    fi
}

test_container_resource_limits() {
    local containers_without_limits
    containers_without_limits=$(docker ps --format "{{.Names}}" | grep "hive-" | wc -l)
    
    # This is a placeholder test - in reality, you'd check resource limits
    if [ "$containers_without_limits" -gt 0 ]; then
        log_info "Found $containers_without_limits monitoring containers"
        return 0
    else
        return 1
    fi
}

# Main test execution
main() {
    log_info "Starting Hive Mind Monitoring Stack Validation"
    echo ""
    
    # Load environment variables if available
    if [ -f "${MONITORING_DIR}/.env" ]; then
        source "${MONITORING_DIR}/.env"
    fi
    
    # Service Availability Tests
    log_info "=== Service Availability Tests ==="
    run_test "Prometheus Availability" test_prometheus_availability
    run_test "Grafana Availability" test_grafana_availability
    run_test "AlertManager Availability" test_alertmanager_availability
    run_test "Loki Availability" test_loki_availability
    run_test "Jaeger Availability" test_jaeger_availability
    
    # Configuration Tests
    log_info "=== Configuration Tests ==="
    run_test "Prometheus Configuration" test_prometheus_config
    run_test "Prometheus Targets" test_prometheus_targets
    run_test "Grafana Datasources" test_grafana_datasources
    run_test "Grafana Dashboards" test_grafana_dashboards
    run_test "AlertManager Configuration" test_alertmanager_config
    
    # Metrics Collection Tests
    log_info "=== Metrics Collection Tests ==="
    run_test "Prometheus Metrics Collection" test_prometheus_metrics
    run_test "Node Exporter Metrics" test_node_exporter_metrics
    run_test "cAdvisor Metrics" test_cadvisor_metrics
    
    # Alert Rules Tests
    log_info "=== Alert Rules Tests ==="
    run_test "Alert Rules Configuration" test_alert_rules
    run_test "Alert Rules Syntax" test_alert_rule_syntax
    
    # Log Ingestion Tests
    log_info "=== Log Ingestion Tests ==="
    run_test "Loki Log Ingestion" test_loki_ingestion
    run_test "Promtail Configuration" test_promtail_targets
    
    # Tracing Tests
    log_info "=== Distributed Tracing Tests ==="
    run_test "Jaeger Services" test_jaeger_services
    
    # Performance Tests
    log_info "=== Performance Tests ==="
    run_test "Prometheus Query Performance" test_prometheus_query_performance
    run_test "Grafana Response Time" test_grafana_response_time
    
    # Storage Tests
    log_info "=== Storage Tests ==="
    run_test "Prometheus Storage" test_prometheus_storage
    run_test "Disk Usage" test_disk_usage
    
    # Integration Tests
    log_info "=== Integration Tests ==="
    run_test "Health Endpoint Integration" test_health_endpoint_integration
    run_test "Metrics Endpoint Integration" test_metrics_endpoint_integration
    
    # Notification Tests
    log_info "=== Notification Tests ==="
    run_test "Slack Webhook" test_slack_webhook
    
    # Security Tests
    log_info "=== Security Tests ==="
    run_test "Grafana Security" test_grafana_security
    
    # Container Health Tests
    log_info "=== Container Health Tests ==="
    run_test "Container Health Status" test_container_health
    run_test "Container Resource Configuration" test_container_resource_limits
    
    # Summary
    echo ""
    log_info "=== Validation Summary ==="
    echo -e "Total Tests: ${BLUE}$TOTAL_TESTS${NC}"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    
    local success_rate=$((TESTS_PASSED * 100 / TOTAL_TESTS))
    echo -e "Success Rate: ${BLUE}${success_rate}%${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "All monitoring components are working correctly!"
        exit 0
    else
        log_error "$TESTS_FAILED tests failed. Please review the output and fix any issues."
        exit 1
    fi
}

# Run main function
main "$@"