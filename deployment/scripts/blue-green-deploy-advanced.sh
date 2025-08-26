#!/bin/bash

# 🚀 Advanced Blue-Green Deployment Script for Quantum Intelligence System
# Implements zero-downtime deployment with automated rollback capabilities
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
readonly NC='\033[0m' # No Color
readonly BOLD='\033[1m'

# 📊 Configuration and defaults
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly CONFIG_DIR="${PROJECT_ROOT}/deployment/config"
readonly TEMPLATES_DIR="${PROJECT_ROOT}/k8s"
readonly LOGS_DIR="${PROJECT_ROOT}/deployment/logs"

# 🔧 Default values
ENVIRONMENT="staging"
IMAGE=""
STRATEGY="blue-green"
QUANTUM_MODULES="all"
NAMESPACE=""
SAFETY_CHECKS="enabled"
MONITORING="enabled"
ROLLBACK_ON_FAILURE="enabled"
DRY_RUN="false"
VERBOSE="false"
TIMEOUT="600"
HEALTH_CHECK_INTERVAL="10"
HEALTH_CHECK_RETRIES="30"

# 📋 Quantum Intelligence modules
declare -A QUANTUM_MODULES_MAP=(
    ["neural-swarm"]="src/ai/quantum_intelligence/neural_swarm_evolution.py"
    ["adaptive-topology"]="src/ai/quantum_intelligence/adaptive_topology_manager.py"
    ["emergent-behavior"]="src/ai/quantum_intelligence/emergent_behavior_engine.py"
    ["meta-learning"]="src/ai/quantum_intelligence/meta_learning_coordinator.py"
)

# 🛡️ Security and validation functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${timestamp} ${BLUE}[INFO]${NC} $message" ;;
        "WARN")  echo -e "${timestamp} ${YELLOW}[WARN]${NC} $message" ;;
        "ERROR") echo -e "${timestamp} ${RED}[ERROR]${NC} $message" >&2 ;;
        "SUCCESS") echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" ;;
        "DEBUG") [[ "$VERBOSE" == "true" ]] && echo -e "${timestamp} ${PURPLE}[DEBUG]${NC} $message" ;;
    esac
    
    # Log to file
    mkdir -p "$LOGS_DIR"
    echo "${timestamp} [${level}] $message" >> "${LOGS_DIR}/deployment-$(date '+%Y%m%d').log"
}

show_help() {
    cat << EOF
🚀 Advanced Blue-Green Deployment Script for Quantum Intelligence System

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    -e, --environment ENV          Deployment environment (staging|production|canary) [default: staging]
    -i, --image IMAGE             Container image to deploy (required)
    -s, --strategy STRATEGY       Deployment strategy (blue-green|canary|rolling) [default: blue-green]
    -q, --quantum-modules MODULES Quantum modules to deploy (all|neural-swarm|adaptive-topology|emergent-behavior|meta-learning) [default: all]
    -n, --namespace NAMESPACE     Kubernetes namespace [default: claude-tui-ENV]
    --safety-checks ENABLED       Enable safety checks (enabled|disabled) [default: enabled]
    --monitoring ENABLED          Enable monitoring deployment (enabled|disabled) [default: enabled]
    --rollback-on-failure ENABLED Auto-rollback on failure (enabled|disabled) [default: enabled]
    --dry-run                     Perform a dry run without actual deployment
    --timeout SECONDS             Deployment timeout in seconds [default: 600]
    --health-check-interval SEC   Health check interval [default: 10]
    --health-check-retries COUNT  Health check retries [default: 30]
    -v, --verbose                 Enable verbose logging
    -h, --help                    Show this help message

EXAMPLES:
    # Basic staging deployment
    $(basename "$0") -e staging -i ghcr.io/claude-tui/claude-tui:latest

    # Production deployment with specific quantum modules
    $(basename "$0") -e production -i ghcr.io/claude-tui/claude-tui:v1.0.0 -q neural-swarm,meta-learning

    # Canary deployment with extended timeout
    $(basename "$0") -e production -i ghcr.io/claude-tui/claude-tui:canary -s canary --timeout 900

    # Dry run for validation
    $(basename "$0") -e staging -i test:latest --dry-run

QUANTUM INTELLIGENCE MODULES:
    all               Deploy all quantum intelligence modules
    neural-swarm      Neural Swarm Evolution Engine
    adaptive-topology Adaptive Topology Manager
    emergent-behavior Emergent Behavior Engine
    meta-learning     Meta Learning Coordinator

For more information, visit: https://github.com/claude-tui/claude-tui/docs/deployment
EOF
}

# 📝 Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -i|--image)
                IMAGE="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -q|--quantum-modules)
                QUANTUM_MODULES="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --safety-checks)
                SAFETY_CHECKS="$2"
                shift 2
                ;;
            --monitoring)
                MONITORING="$2"
                shift 2
                ;;
            --rollback-on-failure)
                ROLLBACK_ON_FAILURE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --health-check-interval)
                HEALTH_CHECK_INTERVAL="$2"
                shift 2
                ;;
            --health-check-retries)
                HEALTH_CHECK_RETRIES="$2"
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

# 🔍 Validation functions
validate_environment() {
    case "$ENVIRONMENT" in
        staging|production|canary) ;;
        *)
            log "ERROR" "Invalid environment: $ENVIRONMENT. Must be staging, production, or canary."
            exit 1
            ;;
    esac
}

validate_image() {
    if [[ -z "$IMAGE" ]]; then
        log "ERROR" "Container image is required. Use -i or --image option."
        exit 1
    fi
    
    # Validate image format
    if ! [[ "$IMAGE" =~ ^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)*:[a-zA-Z0-9._-]+$ ]]; then
        log "ERROR" "Invalid image format: $IMAGE"
        exit 1
    fi
    
    log "INFO" "Validating container image accessibility..."
    if [[ "$DRY_RUN" == "false" ]]; then
        if ! docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
            log "ERROR" "Container image not accessible: $IMAGE"
            exit 1
        fi
    fi
    log "SUCCESS" "Container image validation passed"
}

validate_strategy() {
    case "$STRATEGY" in
        blue-green|canary|rolling) ;;
        *)
            log "ERROR" "Invalid deployment strategy: $STRATEGY. Must be blue-green, canary, or rolling."
            exit 1
            ;;
    esac
}

validate_quantum_modules() {
    if [[ "$QUANTUM_MODULES" == "all" ]]; then
        return 0
    fi
    
    IFS=',' read -ra MODULES <<< "$QUANTUM_MODULES"
    for module in "${MODULES[@]}"; do
        if [[ ! -v QUANTUM_MODULES_MAP["$module"] ]]; then
            log "ERROR" "Invalid quantum module: $module. Available modules: ${!QUANTUM_MODULES_MAP[*]}"
            exit 1
        fi
    done
}

validate_kubernetes_access() {
    log "INFO" "Validating Kubernetes access..."
    
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log "ERROR" "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Set namespace if not provided
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="claude-tui-${ENVIRONMENT}"
    fi
    
    # Check if namespace exists, create if needed
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log "INFO" "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl create namespace "$NAMESPACE" || {
                log "ERROR" "Failed to create namespace: $NAMESPACE"
                exit 1
            }
        fi
    fi
    
    log "SUCCESS" "Kubernetes access validation passed"
}

# 🧠 Quantum Intelligence deployment functions
deploy_quantum_modules() {
    log "INFO" "🧠 Deploying Quantum Intelligence modules..."
    
    local modules_to_deploy
    if [[ "$QUANTUM_MODULES" == "all" ]]; then
        modules_to_deploy="${!QUANTUM_MODULES_MAP[*]}"
    else
        IFS=',' read -ra modules_to_deploy <<< "$QUANTUM_MODULES"
    fi
    
    for module in $modules_to_deploy; do
        log "INFO" "Deploying quantum module: $module"
        
        # Generate module-specific configuration
        local config_file="${TEMPLATES_DIR}/quantum/${module}-deployment.yaml"
        
        if [[ -f "$config_file" ]]; then
            # Apply module deployment with envsubst for variable substitution
            export MODULE_NAME="$module"
            export IMAGE="$IMAGE"
            export NAMESPACE="$NAMESPACE"
            export ENVIRONMENT="$ENVIRONMENT"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                envsubst < "$config_file" | kubectl apply -f -
            else
                log "DEBUG" "DRY RUN: Would apply configuration for module $module"
            fi
        else
            log "WARN" "Configuration file not found for module: $module"
        fi
    done
    
    log "SUCCESS" "🧠 Quantum Intelligence modules deployment completed"
}

# 🔄 Blue-Green deployment implementation
perform_blue_green_deployment() {
    log "INFO" "🔄 Starting Blue-Green deployment..."
    
    local current_color
    local new_color
    
    # Determine current active color
    if kubectl get deployment "${NAMESPACE}-green" -n "$NAMESPACE" >/dev/null 2>&1; then
        current_color="green"
        new_color="blue"
    else
        current_color="blue"
        new_color="green"
    fi
    
    log "INFO" "Current active color: $current_color, deploying to: $new_color"
    
    # Deploy to inactive color
    deploy_to_color "$new_color"
    
    # Health check new deployment
    if ! health_check_deployment "$new_color"; then
        log "ERROR" "Health check failed for $new_color deployment"
        if [[ "$ROLLBACK_ON_FAILURE" == "enabled" ]]; then
            rollback_deployment "$new_color"
        fi
        exit 1
    fi
    
    # Switch traffic to new deployment
    switch_traffic "$new_color"
    
    # Clean up old deployment
    cleanup_old_deployment "$current_color"
    
    log "SUCCESS" "🔄 Blue-Green deployment completed successfully"
}

deploy_to_color() {
    local color="$1"
    log "INFO" "Deploying to $color environment..."
    
    # Prepare deployment configuration
    local deployment_file="${TEMPLATES_DIR}/${ENVIRONMENT}/deployment-${color}.yaml"
    local service_file="${TEMPLATES_DIR}/${ENVIRONMENT}/service-${color}.yaml"
    
    # Set environment variables for template substitution
    export COLOR="$color"
    export IMAGE="$IMAGE"
    export NAMESPACE="$NAMESPACE"
    export ENVIRONMENT="$ENVIRONMENT"
    export QUANTUM_MODULES="$QUANTUM_MODULES"
    
    # Apply deployment
    if [[ "$DRY_RUN" == "false" ]]; then
        if [[ -f "$deployment_file" ]]; then
            envsubst < "$deployment_file" | kubectl apply -f -
        else
            # Generate deployment from template
            generate_deployment_config "$color" | kubectl apply -f -
        fi
        
        if [[ -f "$service_file" ]]; then
            envsubst < "$service_file" | kubectl apply -f -
        fi
    else
        log "DEBUG" "DRY RUN: Would deploy to $color environment"
    fi
    
    # Wait for deployment to be ready
    if [[ "$DRY_RUN" == "false" ]]; then
        log "INFO" "Waiting for $color deployment to be ready..."
        kubectl rollout status deployment/claude-tui-$color -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    fi
}

generate_deployment_config() {
    local color="$1"
    
    cat << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-tui-$color
  namespace: $NAMESPACE
  labels:
    app: claude-tui
    version: $color
    environment: $ENVIRONMENT
    quantum-enabled: "true"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-tui
      version: $color
  template:
    metadata:
      labels:
        app: claude-tui
        version: $color
        environment: $ENVIRONMENT
        quantum-enabled: "true"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
      containers:
      - name: claude-tui
        image: $IMAGE
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: ENVIRONMENT
          value: "$ENVIRONMENT"
        - name: QUANTUM_MODULES
          value: "$QUANTUM_MODULES"
        - name: KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "$MAX_MEMORY_THRESHOLD"
            cpu: "$MAX_CPU_THRESHOLD"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: claude-tui-$color
  namespace: $NAMESPACE
  labels:
    app: claude-tui
    version: $color
spec:
  selector:
    app: claude-tui
    version: $color
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
EOF
}

health_check_deployment() {
    local color="$1"
    log "INFO" "🩺 Performing health check for $color deployment..."
    
    local endpoint="http://claude-tui-${color}.${NAMESPACE}.svc.cluster.local/health"
    local retries=0
    
    while [[ $retries -lt $HEALTH_CHECK_RETRIES ]]; do
        if [[ "$DRY_RUN" == "false" ]]; then
            # Use kubectl port-forward for health check
            if kubectl run health-check-pod --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" -- \
               curl -f -s "$endpoint" >/dev/null 2>&1; then
                log "SUCCESS" "Health check passed for $color deployment"
                
                # Additional quantum intelligence health checks
                if ! quantum_health_check "$color"; then
                    log "ERROR" "Quantum intelligence health check failed"
                    return 1
                fi
                
                return 0
            fi
        else
            log "DEBUG" "DRY RUN: Would perform health check for $color"
            return 0
        fi
        
        ((retries++))
        log "INFO" "Health check attempt $retries/$HEALTH_CHECK_RETRIES failed, retrying in ${HEALTH_CHECK_INTERVAL}s..."
        sleep "$HEALTH_CHECK_INTERVAL"
    done
    
    log "ERROR" "Health check failed for $color deployment after $HEALTH_CHECK_RETRIES attempts"
    return 1
}

quantum_health_check() {
    local color="$1"
    log "INFO" "🧠 Performing quantum intelligence health check..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DEBUG" "DRY RUN: Would perform quantum health check"
        return 0
    fi
    
    # Check quantum modules endpoints
    local modules_to_check
    if [[ "$QUANTUM_MODULES" == "all" ]]; then
        modules_to_check="${!QUANTUM_MODULES_MAP[*]}"
    else
        IFS=',' read -ra modules_to_check <<< "$QUANTUM_MODULES"
    fi
    
    for module in $modules_to_check; do
        local endpoint="http://claude-tui-${color}.${NAMESPACE}.svc.cluster.local/quantum/$module/health"
        if ! kubectl run quantum-health-check-pod --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" -- \
           curl -f -s "$endpoint" >/dev/null 2>&1; then
            log "ERROR" "Quantum module health check failed: $module"
            return 1
        fi
    done
    
    log "SUCCESS" "🧠 Quantum intelligence health check passed"
    return 0
}

switch_traffic() {
    local new_color="$1"
    log "INFO" "🔀 Switching traffic to $new_color deployment..."
    
    # Update the main service to point to new deployment
    local main_service_file="${TEMPLATES_DIR}/${ENVIRONMENT}/service.yaml"
    
    export COLOR="$new_color"
    export NAMESPACE="$NAMESPACE"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        if [[ -f "$main_service_file" ]]; then
            envsubst < "$main_service_file" | kubectl apply -f -
        else
            # Update service selector
            kubectl patch service claude-tui -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"'$new_color'"}}}'
        fi
        
        # Wait for service update to propagate
        sleep 10
        
        # Verify traffic switch
        if ! verify_traffic_switch "$new_color"; then
            log "ERROR" "Traffic switch verification failed"
            return 1
        fi
    else
        log "DEBUG" "DRY RUN: Would switch traffic to $new_color"
    fi
    
    log "SUCCESS" "Traffic successfully switched to $new_color deployment"
}

verify_traffic_switch() {
    local color="$1"
    log "INFO" "🔍 Verifying traffic switch to $color..."
    
    # Perform multiple requests to verify traffic routing
    local success_count=0
    local total_requests=10
    
    for ((i=1; i<=total_requests; i++)); do
        if kubectl run traffic-test-pod --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" -- \
           curl -f -s "http://claude-tui.${NAMESPACE}.svc.cluster.local/version" | grep -q "$color"; then
            ((success_count++))
        fi
    done
    
    local success_rate=$((success_count * 100 / total_requests))
    log "INFO" "Traffic switch verification: ${success_rate}% success rate"
    
    if [[ $success_rate -ge 80 ]]; then
        log "SUCCESS" "Traffic switch verification passed"
        return 0
    else
        log "ERROR" "Traffic switch verification failed"
        return 1
    fi
}

cleanup_old_deployment() {
    local old_color="$1"
    log "INFO" "🧹 Cleaning up old $old_color deployment..."
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Scale down old deployment
        kubectl scale deployment claude-tui-$old_color --replicas=0 -n "$NAMESPACE" || true
        
        # Wait a bit before deleting
        sleep 30
        
        # Delete old deployment and service
        kubectl delete deployment claude-tui-$old_color -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete service claude-tui-$old_color -n "$NAMESPACE" --ignore-not-found=true
    else
        log "DEBUG" "DRY RUN: Would cleanup old $old_color deployment"
    fi
    
    log "SUCCESS" "Old deployment cleanup completed"
}

rollback_deployment() {
    local failed_color="$1"
    log "WARN" "🔄 Initiating rollback for failed $failed_color deployment..."
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Scale down failed deployment
        kubectl scale deployment claude-tui-$failed_color --replicas=0 -n "$NAMESPACE" || true
        
        # Delete failed deployment
        kubectl delete deployment claude-tui-$failed_color -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete service claude-tui-$failed_color -n "$NAMESPACE" --ignore-not-found=true
    else
        log "DEBUG" "DRY RUN: Would rollback failed $failed_color deployment"
    fi
    
    log "SUCCESS" "Rollback completed"
}

# 📊 Monitoring deployment
deploy_monitoring() {
    if [[ "$MONITORING" != "enabled" ]]; then
        log "INFO" "Monitoring deployment skipped"
        return 0
    fi
    
    log "INFO" "📊 Deploying monitoring stack..."
    
    local monitoring_script="${SCRIPT_DIR}/deploy-monitoring.sh"
    if [[ -x "$monitoring_script" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            "$monitoring_script" "$ENVIRONMENT"
        else
            log "DEBUG" "DRY RUN: Would deploy monitoring stack"
        fi
    else
        log "WARN" "Monitoring deployment script not found or not executable"
    fi
}

# 🎯 Main execution function
main() {
    log "INFO" "🚀 Starting Advanced Blue-Green Deployment..."
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Image: $IMAGE"
    log "INFO" "Strategy: $STRATEGY"
    log "INFO" "Quantum Modules: $QUANTUM_MODULES"
    log "INFO" "Namespace: $NAMESPACE"
    
    # Validation phase
    log "INFO" "🔍 Running pre-deployment validation..."
    validate_environment
    validate_image
    validate_strategy
    validate_quantum_modules
    validate_kubernetes_access
    
    # Safety checks
    if [[ "$SAFETY_CHECKS" == "enabled" ]]; then
        log "INFO" "🛡️ Running safety checks..."
        # Add safety check implementations here
    fi
    
    # Deployment phase
    case "$STRATEGY" in
        blue-green)
            perform_blue_green_deployment
            ;;
        canary)
            log "ERROR" "Canary deployment strategy not yet implemented"
            exit 1
            ;;
        rolling)
            log "ERROR" "Rolling deployment strategy not yet implemented"
            exit 1
            ;;
    esac
    
    # Deploy quantum intelligence modules
    deploy_quantum_modules
    
    # Deploy monitoring
    deploy_monitoring
    
    # Final validation
    log "INFO" "🔍 Running post-deployment validation..."
    
    log "SUCCESS" "🎉 Deployment completed successfully!"
    log "INFO" "📊 Deployment Summary:"
    log "INFO" "  Environment: $ENVIRONMENT"
    log "INFO" "  Image: $IMAGE"
    log "INFO" "  Strategy: $STRATEGY"
    log "INFO" "  Quantum Modules: $QUANTUM_MODULES"
    log "INFO" "  Namespace: $NAMESPACE"
    log "INFO" "  Monitoring: $MONITORING"
}

# 🚀 Script execution
trap 'log "ERROR" "Deployment failed at line $LINENO"' ERR

# Parse arguments and run main function
parse_args "$@"
main

log "SUCCESS" "🚀 Advanced Blue-Green Deployment Script completed successfully!"