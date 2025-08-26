#!/bin/bash

# 🩺 Quantum Intelligence Health Check Script
# Comprehensive health validation for all quantum modules and system components
# Author: CI/CD Engineer - Hive Mind Team
# Version: 2.0.0

set -euo pipefail

# 🎨 Colors and formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'
readonly BOLD='\033[1m'

# 📊 Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly LOGS_DIR="${PROJECT_ROOT}/deployment/logs"

# 🔧 Default values
ENVIRONMENT="staging"
NAMESPACE=""
TIMEOUT="300"
QUANTUM_VALIDATION="true"
PERFORMANCE_VALIDATION="false"
SECURITY_VALIDATION="false"
SLA_VALIDATION="false"
VERBOSE="false"
OUTPUT_FORMAT="text"
REPORT_FILE=""

# 📊 Health check metrics
declare -A HEALTH_METRICS=(
    ["total_checks"]=0
    ["passed_checks"]=0
    ["failed_checks"]=0
    ["warning_checks"]=0
    ["quantum_checks"]=0
    ["performance_checks"]=0
    ["security_checks"]=0
    ["sla_checks"]=0
)

# 🧠 Quantum Intelligence modules to check
declare -A QUANTUM_MODULES=(
    ["neural-swarm"]="Neural Swarm Evolution Engine"
    ["adaptive-topology"]="Adaptive Topology Manager"
    ["emergent-behavior"]="Emergent Behavior Engine"
    ["meta-learning"]="Meta Learning Coordinator"
    ["quantum-orchestrator"]="Quantum Intelligence Orchestrator"
)

# 📈 SLA thresholds
declare -A SLA_THRESHOLDS=(
    ["response_time_ms"]=500
    ["availability_percent"]=99.9
    ["error_rate_percent"]=0.1
    ["memory_usage_percent"]=80
    ["cpu_usage_percent"]=70
    ["disk_usage_percent"]=85
)

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")    echo -e "${timestamp} ${BLUE}[INFO]${NC} $message" ;;
        "WARN")    echo -e "${timestamp} ${YELLOW}[WARN]${NC} $message" ;;
        "ERROR")   echo -e "${timestamp} ${RED}[ERROR]${NC} $message" >&2 ;;
        "SUCCESS") echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" ;;
        "DEBUG")   [[ "$VERBOSE" == "true" ]] && echo -e "${timestamp} ${PURPLE}[DEBUG]${NC} $message" ;;
    esac
    
    # Log to file
    mkdir -p "$LOGS_DIR"
    echo "${timestamp} [${level}] $message" >> "${LOGS_DIR}/health-checks-$(date '+%Y%m%d').log"
}

show_help() {
    cat << EOF
🩺 Quantum Intelligence Health Check Script

USAGE:
    $(basename "$0") [ENVIRONMENT] [OPTIONS]

ARGUMENTS:
    ENVIRONMENT                   Deployment environment (staging|production|canary)

OPTIONS:
    -n, --namespace NAMESPACE     Kubernetes namespace [default: claude-tui-ENV]
    -t, --timeout SECONDS         Health check timeout [default: 300]
    --quantum-validation          Enable quantum intelligence validation [default: true]
    --performance-validation      Enable performance validation [default: false]
    --security-validation         Enable security validation [default: false]
    --sla-validation             Enable SLA validation [default: false]
    --output-format FORMAT        Output format (text|json|html) [default: text]
    --report-file FILE           Save report to file
    -v, --verbose                Enable verbose logging
    -h, --help                   Show this help message

EXAMPLES:
    # Basic health check for staging
    $(basename "$0") staging

    # Comprehensive production validation
    $(basename "$0") production --quantum-validation --performance-validation --security-validation --sla-validation

    # Generate JSON report
    $(basename "$0") production --output-format json --report-file health-report.json

    # Extended timeout for performance tests
    $(basename "$0") staging --timeout 600 --performance-validation

HEALTH CHECK CATEGORIES:
    🧠 Quantum Intelligence: Neural modules, swarm coordination, adaptation
    🚄 Performance:         Response times, throughput, resource usage
    🔒 Security:           Authentication, authorization, encryption
    📊 SLA Compliance:      Availability, reliability, service levels

For more information, visit: https://github.com/claude-tui/claude-tui/docs/health-checks
EOF
}

parse_args() {
    if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
        ENVIRONMENT="$1"
        shift
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --quantum-validation)
                QUANTUM_VALIDATION="true"
                shift
                ;;
            --performance-validation)
                PERFORMANCE_VALIDATION="true"
                shift
                ;;
            --security-validation)
                SECURITY_VALIDATION="true"
                shift
                ;;
            --sla-validation)
                SLA_VALIDATION="true"
                shift
                ;;
            --output-format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --report-file)
                REPORT_FILE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                show_help >&2
                exit 1
                ;;
        esac
    done
}

validate_environment() {
    case "$ENVIRONMENT" in
        staging|production|canary) ;;
        *)
            log "ERROR" "Invalid environment: $ENVIRONMENT. Must be staging, production, or canary."
            exit 1
            ;;
    esac
    
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="claude-tui-${ENVIRONMENT}"
    fi
}

check_kubernetes_connection() {
    log "INFO" "🔍 Checking Kubernetes connection..."
    
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log "ERROR" "Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log "ERROR" "Namespace $NAMESPACE not found"
        return 1
    fi
    
    log "SUCCESS" "Kubernetes connection validated"
    return 0
}

increment_metric() {
    local metric="$1"
    HEALTH_METRICS["$metric"]=$((HEALTH_METRICS["$metric"] + 1))
}

run_health_check() {
    local check_name="$1"
    local check_function="$2"
    local category="$3"
    
    log "INFO" "Running health check: $check_name"
    increment_metric "total_checks"
    
    if [[ -n "$category" ]]; then
        increment_metric "${category}_checks"
    fi
    
    local start_time=$(date +%s)
    local result=0
    
    if $check_function; then
        increment_metric "passed_checks"
        log "SUCCESS" "✅ $check_name: PASSED"
    else
        result=$?
        if [[ $result -eq 2 ]]; then
            increment_metric "warning_checks"
            log "WARN" "⚠️  $check_name: WARNING"
        else
            increment_metric "failed_checks"
            log "ERROR" "❌ $check_name: FAILED"
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log "DEBUG" "Health check $check_name completed in ${duration}s"
    
    return $result
}

# 🧠 Quantum Intelligence Health Checks
check_quantum_orchestrator() {
    log "DEBUG" "Checking quantum orchestrator health..."
    
    local pod_name
    pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui,component=quantum-orchestrator --no-headers -o custom-columns=":metadata.name" | head -n1)
    
    if [[ -z "$pod_name" ]]; then
        log "ERROR" "No quantum orchestrator pods found"
        return 1
    fi
    
    # Check pod readiness
    if ! kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' | grep -q "True"; then
        log "ERROR" "Quantum orchestrator pod not ready: $pod_name"
        return 1
    fi
    
    # Test quantum orchestrator endpoint
    if kubectl exec "$pod_name" -n "$NAMESPACE" -- curl -f -s http://localhost:8000/quantum/orchestrator/health >/dev/null 2>&1; then
        log "DEBUG" "Quantum orchestrator health endpoint responsive"
        return 0
    else
        log "ERROR" "Quantum orchestrator health endpoint unresponsive"
        return 1
    fi
}

check_neural_swarm_evolution() {
    log "DEBUG" "Checking neural swarm evolution engine..."
    
    local response
    if response=$(kubectl run health-check-temp --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" --timeout=30s -- \
                 curl -s -f "http://claude-tui.${NAMESPACE}.svc.cluster.local/quantum/neural-swarm/health" 2>/dev/null); then
        
        # Parse response for detailed health info
        if echo "$response" | grep -q '"status":"healthy"'; then
            log "DEBUG" "Neural swarm evolution engine healthy"
            return 0
        else
            log "WARN" "Neural swarm evolution engine reporting degraded performance"
            return 2
        fi
    else
        log "ERROR" "Neural swarm evolution engine health check failed"
        return 1
    fi
}

check_adaptive_topology_manager() {
    log "DEBUG" "Checking adaptive topology manager..."
    
    local topology_status
    if topology_status=$(kubectl run health-check-temp --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" --timeout=30s -- \
                        curl -s -f "http://claude-tui.${NAMESPACE}.svc.cluster.local/quantum/adaptive-topology/status" 2>/dev/null); then
        
        local active_nodes
        active_nodes=$(echo "$topology_status" | grep -o '"active_nodes":[0-9]*' | cut -d: -f2)
        
        if [[ "$active_nodes" -gt 0 ]]; then
            log "DEBUG" "Adaptive topology manager active with $active_nodes nodes"
            return 0
        else
            log "WARN" "Adaptive topology manager has no active nodes"
            return 2
        fi
    else
        log "ERROR" "Adaptive topology manager health check failed"
        return 1
    fi
}

check_emergent_behavior_engine() {
    log "DEBUG" "Checking emergent behavior engine..."
    
    if kubectl run health-check-temp --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" --timeout=30s -- \
       curl -s -f "http://claude-tui.${NAMESPACE}.svc.cluster.local/quantum/emergent-behavior/metrics" >/dev/null 2>&1; then
        log "DEBUG" "Emergent behavior engine responding to metrics requests"
        return 0
    else
        log "ERROR" "Emergent behavior engine health check failed"
        return 1
    fi
}

check_meta_learning_coordinator() {
    log "DEBUG" "Checking meta learning coordinator..."
    
    local learning_status
    if learning_status=$(kubectl run health-check-temp --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" --timeout=30s -- \
                        curl -s -f "http://claude-tui.${NAMESPACE}.svc.cluster.local/quantum/meta-learning/status" 2>/dev/null); then
        
        if echo "$learning_status" | grep -q '"learning_active":true'; then
            log "DEBUG" "Meta learning coordinator actively learning"
            return 0
        else
            log "WARN" "Meta learning coordinator not actively learning"
            return 2
        fi
    else
        log "ERROR" "Meta learning coordinator health check failed"
        return 1
    fi
}

# 🚄 Performance Health Checks
check_response_time() {
    log "DEBUG" "Checking API response time..."
    
    local start_time end_time response_time
    start_time=$(date +%s%N)
    
    if kubectl run perf-check-temp --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" --timeout=10s -- \
       curl -s -f "http://claude-tui.${NAMESPACE}.svc.cluster.local/health" >/dev/null 2>&1; then
        
        end_time=$(date +%s%N)
        response_time=$(((end_time - start_time) / 1000000)) # Convert to milliseconds
        
        log "DEBUG" "API response time: ${response_time}ms"
        
        if [[ $response_time -lt ${SLA_THRESHOLDS["response_time_ms"]} ]]; then
            return 0
        else
            log "WARN" "API response time ${response_time}ms exceeds threshold ${SLA_THRESHOLDS["response_time_ms"]}ms"
            return 2
        fi
    else
        log "ERROR" "API response time check failed"
        return 1
    fi
}

check_memory_usage() {
    log "DEBUG" "Checking memory usage..."
    
    local pods memory_usage
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name")
    
    for pod in $pods; do
        if memory_usage=$(kubectl top pod "$pod" -n "$NAMESPACE" --no-headers | awk '{print $3}' | sed 's/Mi//' 2>/dev/null); then
            # Get memory limit
            local memory_limit
            memory_limit=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].resources.limits.memory}' | sed 's/Mi//')
            
            if [[ -n "$memory_limit" && "$memory_limit" != "null" ]]; then
                local usage_percent=$((memory_usage * 100 / memory_limit))
                log "DEBUG" "Pod $pod memory usage: ${usage_percent}%"
                
                if [[ $usage_percent -gt ${SLA_THRESHOLDS["memory_usage_percent"]} ]]; then
                    log "WARN" "Pod $pod memory usage ${usage_percent}% exceeds threshold"
                    return 2
                fi
            fi
        else
            log "WARN" "Could not get memory usage for pod $pod"
            return 2
        fi
    done
    
    return 0
}

check_cpu_usage() {
    log "DEBUG" "Checking CPU usage..."
    
    local pods cpu_usage
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name")
    
    for pod in $pods; do
        if cpu_usage=$(kubectl top pod "$pod" -n "$NAMESPACE" --no-headers | awk '{print $2}' | sed 's/m//' 2>/dev/null); then
            # Get CPU limit
            local cpu_limit
            cpu_limit=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].resources.limits.cpu}' | sed 's/m//')
            
            if [[ -n "$cpu_limit" && "$cpu_limit" != "null" ]]; then
                local usage_percent=$((cpu_usage * 100 / cpu_limit))
                log "DEBUG" "Pod $pod CPU usage: ${usage_percent}%"
                
                if [[ $usage_percent -gt ${SLA_THRESHOLDS["cpu_usage_percent"]} ]]; then
                    log "WARN" "Pod $pod CPU usage ${usage_percent}% exceeds threshold"
                    return 2
                fi
            fi
        else
            log "WARN" "Could not get CPU usage for pod $pod"
            return 2
        fi
    done
    
    return 0
}

# 🔒 Security Health Checks
check_pod_security_context() {
    log "DEBUG" "Checking pod security contexts..."
    
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name")
    
    for pod in $pods; do
        # Check if pod runs as non-root
        local run_as_non_root
        run_as_non_root=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.runAsNonRoot}')
        
        if [[ "$run_as_non_root" != "true" ]]; then
            log "ERROR" "Pod $pod not configured to run as non-root"
            return 1
        fi
        
        # Check if containers have read-only root filesystem
        local read_only_root_fs
        read_only_root_fs=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].securityContext.readOnlyRootFilesystem}')
        
        if [[ "$read_only_root_fs" != "true" ]]; then
            log "WARN" "Pod $pod container does not have read-only root filesystem"
            return 2
        fi
    done
    
    log "DEBUG" "Pod security contexts validated"
    return 0
}

check_tls_certificates() {
    log "DEBUG" "Checking TLS certificate validity..."
    
    local ingress_host
    ingress_host=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "")
    
    if [[ -n "$ingress_host" ]]; then
        if echo | openssl s_client -servername "$ingress_host" -connect "$ingress_host:443" 2>/dev/null | \
           openssl x509 -checkend 604800 -noout >/dev/null 2>&1; then
            log "DEBUG" "TLS certificate valid for $ingress_host"
            return 0
        else
            log "WARN" "TLS certificate for $ingress_host expires within 7 days or is invalid"
            return 2
        fi
    else
        log "DEBUG" "No ingress found, skipping TLS certificate check"
        return 0
    fi
}

# 📊 SLA Health Checks
check_service_availability() {
    log "DEBUG" "Checking service availability..."
    
    local available_replicas desired_replicas
    available_replicas=$(kubectl get deployment -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].status.availableReplicas}' 2>/dev/null || echo "0")
    desired_replicas=$(kubectl get deployment -n "$NAMESPACE" -l app=claude-tui -o jsonpath='{.items[0].spec.replicas}' 2>/dev/null || echo "1")
    
    local availability_percent=$((available_replicas * 100 / desired_replicas))
    log "DEBUG" "Service availability: ${availability_percent}% (${available_replicas}/${desired_replicas} replicas)"
    
    if [[ $availability_percent -ge ${SLA_THRESHOLDS["availability_percent"]} ]]; then
        return 0
    else
        log "ERROR" "Service availability ${availability_percent}% below SLA threshold"
        return 1
    fi
}

check_error_rate() {
    log "DEBUG" "Checking error rate from metrics..."
    
    # This would typically query Prometheus metrics
    # For now, we'll simulate by checking recent logs
    local error_count total_count error_rate
    
    error_count=$(kubectl logs -n "$NAMESPACE" -l app=claude-tui --since=5m 2>/dev/null | grep -c "ERROR" || echo "0")
    total_count=$(kubectl logs -n "$NAMESPACE" -l app=claude-tui --since=5m 2>/dev/null | wc -l || echo "1")
    
    if [[ $total_count -gt 0 ]]; then
        error_rate=$((error_count * 100 / total_count))
        log "DEBUG" "Error rate: ${error_rate}% (${error_count}/${total_count} log entries)"
        
        if [[ $error_rate -le ${SLA_THRESHOLDS["error_rate_percent"]} ]]; then
            return 0
        else
            log "WARN" "Error rate ${error_rate}% exceeds SLA threshold"
            return 2
        fi
    else
        log "DEBUG" "No log entries found, assuming no errors"
        return 0
    fi
}

generate_report() {
    local report_data
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$OUTPUT_FORMAT" in
        json)
            report_data=$(cat << EOF
{
  "timestamp": "$timestamp",
  "environment": "$ENVIRONMENT",
  "namespace": "$NAMESPACE",
  "health_check_summary": {
    "total_checks": ${HEALTH_METRICS["total_checks"]},
    "passed_checks": ${HEALTH_METRICS["passed_checks"]},
    "failed_checks": ${HEALTH_METRICS["failed_checks"]},
    "warning_checks": ${HEALTH_METRICS["warning_checks"]},
    "success_rate": "$((HEALTH_METRICS["passed_checks"] * 100 / HEALTH_METRICS["total_checks"]))%"
  },
  "validation_categories": {
    "quantum_intelligence": ${HEALTH_METRICS["quantum_checks"]},
    "performance": ${HEALTH_METRICS["performance_checks"]},
    "security": ${HEALTH_METRICS["security_checks"]},
    "sla_compliance": ${HEALTH_METRICS["sla_checks"]}
  },
  "overall_status": "$([ ${HEALTH_METRICS["failed_checks"]} -eq 0 ] && echo "HEALTHY" || echo "DEGRADED")"
}
EOF
            )
            ;;
        *)
            report_data=$(cat << EOF

🩺 QUANTUM INTELLIGENCE HEALTH REPORT
=====================================

Timestamp: $timestamp
Environment: $ENVIRONMENT
Namespace: $NAMESPACE

📊 SUMMARY
----------
Total Checks: ${HEALTH_METRICS["total_checks"]}
✅ Passed: ${HEALTH_METRICS["passed_checks"]}
❌ Failed: ${HEALTH_METRICS["failed_checks"]}
⚠️  Warnings: ${HEALTH_METRICS["warning_checks"]}
📈 Success Rate: $((HEALTH_METRICS["passed_checks"] * 100 / HEALTH_METRICS["total_checks"]))%

🧠 QUANTUM INTELLIGENCE: ${HEALTH_METRICS["quantum_checks"]} checks
🚄 PERFORMANCE: ${HEALTH_METRICS["performance_checks"]} checks
🔒 SECURITY: ${HEALTH_METRICS["security_checks"]} checks
📊 SLA COMPLIANCE: ${HEALTH_METRICS["sla_checks"]} checks

Overall Status: $([ ${HEALTH_METRICS["failed_checks"]} -eq 0 ] && echo "🟢 HEALTHY" || echo "🟡 DEGRADED")
EOF
            )
            ;;
    esac
    
    if [[ -n "$REPORT_FILE" ]]; then
        echo "$report_data" > "$REPORT_FILE"
        log "INFO" "Report saved to: $REPORT_FILE"
    else
        echo "$report_data"
    fi
}

main() {
    log "INFO" "🩺 Starting Quantum Intelligence Health Checks..."
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Namespace: $NAMESPACE"
    log "INFO" "Timeout: ${TIMEOUT}s"
    
    # Validate environment and connectivity
    validate_environment
    if ! check_kubernetes_connection; then
        log "ERROR" "Failed to connect to Kubernetes"
        exit 1
    fi
    
    # Basic system health checks
    run_health_check "Kubernetes Connection" "check_kubernetes_connection" ""
    
    # Quantum Intelligence health checks
    if [[ "$QUANTUM_VALIDATION" == "true" ]]; then
        log "INFO" "🧠 Running Quantum Intelligence validation..."
        run_health_check "Quantum Orchestrator" "check_quantum_orchestrator" "quantum"
        run_health_check "Neural Swarm Evolution" "check_neural_swarm_evolution" "quantum"
        run_health_check "Adaptive Topology Manager" "check_adaptive_topology_manager" "quantum"
        run_health_check "Emergent Behavior Engine" "check_emergent_behavior_engine" "quantum"
        run_health_check "Meta Learning Coordinator" "check_meta_learning_coordinator" "quantum"
    fi
    
    # Performance health checks
    if [[ "$PERFORMANCE_VALIDATION" == "true" ]]; then
        log "INFO" "🚄 Running Performance validation..."
        run_health_check "API Response Time" "check_response_time" "performance"
        run_health_check "Memory Usage" "check_memory_usage" "performance"
        run_health_check "CPU Usage" "check_cpu_usage" "performance"
    fi
    
    # Security health checks
    if [[ "$SECURITY_VALIDATION" == "true" ]]; then
        log "INFO" "🔒 Running Security validation..."
        run_health_check "Pod Security Context" "check_pod_security_context" "security"
        run_health_check "TLS Certificates" "check_tls_certificates" "security"
    fi
    
    # SLA health checks
    if [[ "$SLA_VALIDATION" == "true" ]]; then
        log "INFO" "📊 Running SLA validation..."
        run_health_check "Service Availability" "check_service_availability" "sla"
        run_health_check "Error Rate" "check_error_rate" "sla"
    fi
    
    # Generate and display report
    generate_report
    
    # Exit with appropriate code
    if [[ ${HEALTH_METRICS["failed_checks"]} -eq 0 ]]; then
        log "SUCCESS" "🎉 All health checks passed!"
        exit 0
    else
        log "ERROR" "❌ ${HEALTH_METRICS["failed_checks"]} health check(s) failed"
        exit 1
    fi
}

# Script execution
trap 'log "ERROR" "Health check script failed at line $LINENO"' ERR
parse_args "$@"
main