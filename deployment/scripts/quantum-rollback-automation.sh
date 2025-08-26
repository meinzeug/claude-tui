#!/bin/bash

# 🔄 Quantum Intelligence Rollback Automation & Incident Response
# Advanced rollback system with automated incident response and recovery procedures
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
readonly BACKUP_DIR="${PROJECT_ROOT}/deployment/backups"
readonly INCIDENT_DIR="${PROJECT_ROOT}/deployment/incidents"

# 🔧 Default values
ENVIRONMENT="staging"
NAMESPACE=""
REASON="unknown"
PRESERVE_LOGS="true"
NOTIFICATION_CHANNELS="slack"
AUTO_SCALE_DOWN="true"
BACKUP_BEFORE_ROLLBACK="true"
INCIDENT_ID=""
ROLLBACK_TIMEOUT="300"
HEALTH_CHECK_ATTEMPTS="10"
VERBOSE="false"
DRY_RUN="false"

# 📊 Rollback tracking
declare -A ROLLBACK_METRICS=(
    ["start_time"]=$(date +%s)
    ["steps_completed"]=0
    ["steps_failed"]=0
    ["services_affected"]=0
    ["data_preserved"]="unknown"
    ["downtime_seconds"]=0
)

# 🚨 Incident severity levels
declare -A SEVERITY_LEVELS=(
    ["critical"]="P0 - System Down"
    ["high"]="P1 - Major Impact"
    ["medium"]="P2 - Moderate Impact"
    ["low"]="P3 - Minor Impact"
    ["info"]="P4 - Informational"
)

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")      echo -e "${timestamp} ${BLUE}[INFO]${NC} $message" ;;
        "WARN")      echo -e "${timestamp} ${YELLOW}[WARN]${NC} $message" ;;
        "ERROR")     echo -e "${timestamp} ${RED}[ERROR]${NC} $message" >&2 ;;
        "SUCCESS")   echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" ;;
        "DEBUG")     [[ "$VERBOSE" == "true" ]] && echo -e "${timestamp} ${PURPLE}[DEBUG]${NC} $message" ;;
        "ROLLBACK")  echo -e "${timestamp} ${CYAN}[ROLLBACK]${NC} $message" ;;
        "INCIDENT")  echo -e "${timestamp} ${RED}[INCIDENT]${NC} $message" ;;
    esac
    
    # Log to file
    mkdir -p "$LOGS_DIR"
    echo "${timestamp} [${level}] $message" >> "${LOGS_DIR}/rollback-$(date '+%Y%m%d').log"
    
    # Log incidents to separate file
    if [[ "$level" == "INCIDENT" || "$level" == "ERROR" ]]; then
        mkdir -p "$INCIDENT_DIR"
        echo "${timestamp} [${level}] $message" >> "${INCIDENT_DIR}/incident-${INCIDENT_ID:-$(date +%s)}.log"
    fi
}

show_help() {
    cat << EOF
🔄 Quantum Intelligence Rollback Automation & Incident Response

USAGE:
    $(basename "$0") [ENVIRONMENT] [OPTIONS]

ARGUMENTS:
    ENVIRONMENT                   Deployment environment (staging|production|canary)

OPTIONS:
    -n, --namespace NAMESPACE     Kubernetes namespace [default: claude-tui-ENV]
    -r, --reason REASON           Rollback reason (deployment-failure|security-incident|performance-degradation) [default: unknown]
    --preserve-logs              Preserve logs during rollback [default: true]
    --notification-channels LIST  Notification channels (slack,email,pagerduty) [default: slack]
    --auto-scale-down            Auto scale down failed deployment [default: true]
    --backup-before-rollback     Create backup before rollback [default: true]
    --incident-id ID             Incident tracking ID
    --rollback-timeout SECONDS   Rollback operation timeout [default: 300]
    --health-check-attempts NUM   Health check retry attempts [default: 10]
    --dry-run                    Perform dry run without actual rollback
    -v, --verbose                Enable verbose logging
    -h, --help                   Show this help message

ROLLBACK REASONS:
    deployment-failure           Failed deployment or unhealthy services
    security-incident           Security vulnerability or breach detected
    performance-degradation      Unacceptable performance regression
    data-corruption              Data integrity issues detected
    external-dependency-failure  External service dependency failures
    manual-rollback             Manually initiated rollback

EXAMPLES:
    # Basic rollback for failed deployment
    $(basename "$0") staging --reason deployment-failure

    # Production incident rollback with full logging
    $(basename "$0") production --reason security-incident --incident-id INC-2024-001 --preserve-logs

    # Performance rollback with custom timeout
    $(basename "$0") staging --reason performance-degradation --rollback-timeout 600

    # Dry run to validate rollback procedure
    $(basename "$0") staging --reason deployment-failure --dry-run

INCIDENT RESPONSE FEATURES:
    🔄 Automated Rollback:       Intelligent rollback to last known good state
    📊 Health Monitoring:        Continuous health checks during rollback
    💾 Data Preservation:        Backup critical data before rollback
    🚨 Incident Tracking:        Comprehensive incident logging and metrics
    📢 Multi-channel Alerts:     Slack, email, PagerDuty notifications
    🔍 Root Cause Analysis:      Automated log collection and analysis

For more information, visit: https://github.com/claude-tui/claude-tui/docs/incident-response
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
            -r|--reason)
                REASON="$2"
                shift 2
                ;;
            --preserve-logs)
                PRESERVE_LOGS="true"
                shift
                ;;
            --no-preserve-logs)
                PRESERVE_LOGS="false"
                shift
                ;;
            --notification-channels)
                NOTIFICATION_CHANNELS="$2"
                shift 2
                ;;
            --auto-scale-down)
                AUTO_SCALE_DOWN="true"
                shift
                ;;
            --no-auto-scale-down)
                AUTO_SCALE_DOWN="false"
                shift
                ;;
            --backup-before-rollback)
                BACKUP_BEFORE_ROLLBACK="true"
                shift
                ;;
            --no-backup-before-rollback)
                BACKUP_BEFORE_ROLLBACK="false"
                shift
                ;;
            --incident-id)
                INCIDENT_ID="$2"
                shift 2
                ;;
            --rollback-timeout)
                ROLLBACK_TIMEOUT="$2"
                shift 2
                ;;
            --health-check-attempts)
                HEALTH_CHECK_ATTEMPTS="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
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
            log "ERROR" "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="claude-tui-${ENVIRONMENT}"
    fi
    
    # Generate incident ID if not provided
    if [[ -z "$INCIDENT_ID" ]]; then
        INCIDENT_ID="AUTO-$(date +%Y%m%d-%H%M%S)-${ENVIRONMENT}"
    fi
}

check_rollback_prerequisites() {
    log "INFO" "🔍 Checking rollback prerequisites..."
    
    # Check kubectl access
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log "ERROR" "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log "ERROR" "Namespace $NAMESPACE not found"
        exit 1
    fi
    
    # Check if any deployments exist
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=claude-tui --no-headers 2>/dev/null | wc -l)
    
    if [[ $deployments -eq 0 ]]; then
        log "WARN" "No Claude-TUI deployments found in namespace $NAMESPACE"
    fi
    
    log "SUCCESS" "Prerequisites validated"
}

create_incident_record() {
    log "INCIDENT" "📝 Creating incident record: $INCIDENT_ID"
    
    mkdir -p "$INCIDENT_DIR"
    local incident_file="${INCIDENT_DIR}/${INCIDENT_ID}.json"
    
    cat > "$incident_file" << EOF
{
  "incident_id": "$INCIDENT_ID",
  "timestamp": "$(date -Iseconds)",
  "environment": "$ENVIRONMENT",
  "namespace": "$NAMESPACE",
  "reason": "$REASON",
  "severity": "$(determine_incident_severity)",
  "status": "investigating",
  "rollback_initiated": true,
  "affected_services": [],
  "timeline": [
    {
      "timestamp": "$(date -Iseconds)",
      "event": "Incident detected and rollback initiated",
      "details": "Automated rollback triggered for reason: $REASON"
    }
  ],
  "metrics": {
    "detection_time": "$(date -Iseconds)",
    "rollback_start_time": "$(date -Iseconds)",
    "estimated_impact": "unknown",
    "users_affected": "unknown"
  }
}
EOF
    
    log "SUCCESS" "Incident record created: $incident_file"
}

determine_incident_severity() {
    case "$REASON" in
        security-incident|data-corruption)
            echo "critical"
            ;;
        deployment-failure|external-dependency-failure)
            echo "high"
            ;;
        performance-degradation)
            echo "medium"
            ;;
        *)
            echo "low"
            ;;
    esac
}

backup_current_state() {
    if [[ "$BACKUP_BEFORE_ROLLBACK" != "true" ]]; then
        log "INFO" "Skipping backup (disabled)"
        return 0
    fi
    
    log "ROLLBACK" "💾 Creating backup before rollback..."
    
    mkdir -p "$BACKUP_DIR"
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="${BACKUP_DIR}/pre-rollback-${ENVIRONMENT}-${backup_timestamp}.tar.gz"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DEBUG" "DRY RUN: Would create backup at $backup_file"
        return 0
    fi
    
    # Export current Kubernetes resources
    local temp_dir
    temp_dir=$(mktemp -d)
    
    # Backup deployments
    kubectl get deployments -n "$NAMESPACE" -o yaml > "${temp_dir}/deployments.yaml" 2>/dev/null || true
    
    # Backup services
    kubectl get services -n "$NAMESPACE" -o yaml > "${temp_dir}/services.yaml" 2>/dev/null || true
    
    # Backup configmaps
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "${temp_dir}/configmaps.yaml" 2>/dev/null || true
    
    # Backup secrets (names only for security)
    kubectl get secrets -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" > "${temp_dir}/secret-names.txt" 2>/dev/null || true
    
    # Backup logs if requested
    if [[ "$PRESERVE_LOGS" == "true" ]]; then
        log "DEBUG" "Backing up pod logs..."
        kubectl logs -l app=claude-tui -n "$NAMESPACE" --all-containers=true --prefix=true > "${temp_dir}/pod-logs.txt" 2>/dev/null || true
    fi
    
    # Create compressed backup
    tar -czf "$backup_file" -C "$temp_dir" . 2>/dev/null
    rm -rf "$temp_dir"
    
    ROLLBACK_METRICS["data_preserved"]="true"
    log "SUCCESS" "Backup created: $backup_file"
}

identify_rollback_targets() {
    log "ROLLBACK" "🎯 Identifying rollback targets..."
    
    local deployments services configmaps
    
    # Find Claude-TUI deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name" 2>/dev/null | tr '\n' ' ')
    
    # Find related services
    services=$(kubectl get services -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name" 2>/dev/null | tr '\n' ' ')
    
    # Find related configmaps
    configmaps=$(kubectl get configmaps -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name" 2>/dev/null | tr '\n' ' ')
    
    log "DEBUG" "Rollback targets identified:"
    log "DEBUG" "  Deployments: ${deployments:-none}"
    log "DEBUG" "  Services: ${services:-none}"
    log "DEBUG" "  ConfigMaps: ${configmaps:-none}"
    
    # Count affected services
    local deployment_count service_count
    deployment_count=$(echo "${deployments}" | wc -w)
    service_count=$(echo "${services}" | wc -w)
    ROLLBACK_METRICS["services_affected"]=$((deployment_count + service_count))
}

perform_quantum_aware_rollback() {
    log "ROLLBACK" "🧠 Performing quantum-aware rollback..."
    
    # Scale down current deployments first to prevent split-brain scenarios
    if [[ "$AUTO_SCALE_DOWN" == "true" ]]; then
        scale_down_current_deployments
    fi
    
    # Rollback each deployment
    rollback_deployments
    
    # Wait for rollback to stabilize
    wait_for_rollback_completion
    
    # Perform quantum-specific recovery
    recover_quantum_intelligence_state
}

scale_down_current_deployments() {
    log "ROLLBACK" "⬇️  Scaling down current deployments..."
    
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for deployment in $deployments; do
        log "DEBUG" "Scaling down deployment: $deployment"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl scale deployment "$deployment" --replicas=0 -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s" >/dev/null 2>&1 || {
                log "WARN" "Failed to scale down deployment: $deployment"
                ROLLBACK_METRICS["steps_failed"]=$((ROLLBACK_METRICS["steps_failed"] + 1))
            }
        else
            log "DEBUG" "DRY RUN: Would scale down deployment $deployment"
        fi
        
        ROLLBACK_METRICS["steps_completed"]=$((ROLLBACK_METRICS["steps_completed"] + 1))
    done
}

rollback_deployments() {
    log "ROLLBACK" "🔄 Rolling back deployments to previous versions..."
    
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for deployment in $deployments; do
        log "DEBUG" "Rolling back deployment: $deployment"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            # Check if there's a previous revision to rollback to
            local rollout_history
            rollout_history=$(kubectl rollout history deployment/"$deployment" -n "$NAMESPACE" 2>/dev/null | wc -l)
            
            if [[ $rollout_history -gt 1 ]]; then
                kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s" >/dev/null 2>&1 || {
                    log "ERROR" "Failed to rollback deployment: $deployment"
                    ROLLBACK_METRICS["steps_failed"]=$((ROLLBACK_METRICS["steps_failed"] + 1))
                    continue
                }
                
                # Wait for rollback to complete
                kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s" >/dev/null 2>&1 || {
                    log "WARN" "Rollback status check timed out for deployment: $deployment"
                }
            else
                log "WARN" "No previous revision found for deployment: $deployment"
                ROLLBACK_METRICS["steps_failed"]=$((ROLLBACK_METRICS["steps_failed"] + 1))
            fi
        else
            log "DEBUG" "DRY RUN: Would rollback deployment $deployment"
        fi
        
        ROLLBACK_METRICS["steps_completed"]=$((ROLLBACK_METRICS["steps_completed"] + 1))
    done
}

wait_for_rollback_completion() {
    log "ROLLBACK" "⏳ Waiting for rollback completion..."
    
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for deployment in $deployments; do
        log "DEBUG" "Waiting for deployment readiness: $deployment"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            local ready=false
            local attempts=0
            
            while [[ $attempts -lt $HEALTH_CHECK_ATTEMPTS && "$ready" == "false" ]]; do
                if kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q "True"; then
                    ready=true
                    log "DEBUG" "Deployment $deployment is ready"
                else
                    log "DEBUG" "Deployment $deployment not ready yet (attempt $((attempts + 1))/$HEALTH_CHECK_ATTEMPTS)"
                    sleep 10
                fi
                
                attempts=$((attempts + 1))
            done
            
            if [[ "$ready" == "false" ]]; then
                log "ERROR" "Deployment $deployment failed to become ready after rollback"
                ROLLBACK_METRICS["steps_failed"]=$((ROLLBACK_METRICS["steps_failed"] + 1))
            fi
        else
            log "DEBUG" "DRY RUN: Would wait for deployment $deployment readiness"
        fi
    done
}

recover_quantum_intelligence_state() {
    log "ROLLBACK" "🧠 Recovering quantum intelligence state..."
    
    # Specific recovery procedures for quantum modules
    if [[ "$DRY_RUN" == "false" ]]; then
        # Reset neural swarm evolution state
        reset_neural_swarm_state
        
        # Recalibrate adaptive topology
        recalibrate_adaptive_topology
        
        # Reinitialize emergent behavior patterns
        reinitialize_emergent_behavior
        
        # Restart meta-learning coordination
        restart_meta_learning
    else
        log "DEBUG" "DRY RUN: Would recover quantum intelligence state"
    fi
    
    log "SUCCESS" "Quantum intelligence state recovery completed"
}

reset_neural_swarm_state() {
    log "DEBUG" "Resetting neural swarm evolution state..."
    
    # Send reset command to neural swarm pods
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui,component=neural-swarm --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for pod in $pods; do
        kubectl exec "$pod" -n "$NAMESPACE" -- curl -X POST http://localhost:8000/quantum/neural-swarm/reset >/dev/null 2>&1 || {
            log "WARN" "Failed to reset neural swarm state in pod: $pod"
        }
    done
}

recalibrate_adaptive_topology() {
    log "DEBUG" "Recalibrating adaptive topology manager..."
    
    # Send recalibration command to topology manager pods
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui,component=topology-manager --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for pod in $pods; do
        kubectl exec "$pod" -n "$NAMESPACE" -- curl -X POST http://localhost:8000/quantum/adaptive-topology/recalibrate >/dev/null 2>&1 || {
            log "WARN" "Failed to recalibrate topology in pod: $pod"
        }
    done
}

reinitialize_emergent_behavior() {
    log "DEBUG" "Reinitializing emergent behavior patterns..."
    
    # Send reinitialization command to behavior engine pods
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui,component=behavior-engine --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for pod in $pods; do
        kubectl exec "$pod" -n "$NAMESPACE" -- curl -X POST http://localhost:8000/quantum/emergent-behavior/reinitialize >/dev/null 2>&1 || {
            log "WARN" "Failed to reinitialize emergent behavior in pod: $pod"
        }
    done
}

restart_meta_learning() {
    log "DEBUG" "Restarting meta-learning coordination..."
    
    # Send restart command to meta-learning coordinator pods
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=claude-tui,component=meta-learning --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for pod in $pods; do
        kubectl exec "$pod" -n "$NAMESPACE" -- curl -X POST http://localhost:8000/quantum/meta-learning/restart >/dev/null 2>&1 || {
            log "WARN" "Failed to restart meta-learning in pod: $pod"
        }
    done
}

validate_rollback_success() {
    log "ROLLBACK" "✅ Validating rollback success..."
    
    local validation_failures=0
    
    # Check deployment health
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=claude-tui --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for deployment in $deployments; do
        local available_replicas desired_replicas
        available_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.availableReplicas}' 2>/dev/null || echo "0")
        desired_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")
        
        if [[ $available_replicas -lt $desired_replicas ]]; then
            log "ERROR" "Deployment $deployment not fully available after rollback ($available_replicas/$desired_replicas)"
            validation_failures=$((validation_failures + 1))
        else
            log "DEBUG" "Deployment $deployment is healthy ($available_replicas/$desired_replicas)"
        fi
    done
    
    # Test quantum intelligence endpoints
    validate_quantum_endpoints || validation_failures=$((validation_failures + 1))
    
    return $validation_failures
}

validate_quantum_endpoints() {
    log "DEBUG" "Validating quantum intelligence endpoints..."
    
    local service_endpoint
    service_endpoint=$(kubectl get service claude-tui -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "claude-tui.${NAMESPACE}.svc.cluster.local")
    
    # Test each quantum module endpoint
    local quantum_modules=("neural-swarm" "adaptive-topology" "emergent-behavior" "meta-learning")
    local endpoint_failures=0
    
    for module in "${quantum_modules[@]}"; do
        if kubectl run endpoint-test --rm -i --restart=Never --image=curlimages/curl:latest -n "$NAMESPACE" --timeout=30s -- \
           curl -f -s "http://${service_endpoint}/quantum/${module}/health" >/dev/null 2>&1; then
            log "DEBUG" "Quantum module $module endpoint is healthy"
        else
            log "WARN" "Quantum module $module endpoint validation failed"
            endpoint_failures=$((endpoint_failures + 1))
        fi
    done
    
    return $endpoint_failures
}

send_notifications() {
    log "INFO" "📢 Sending rollback notifications..."
    
    local severity
    severity=$(determine_incident_severity)
    local status
    status=$([ ${ROLLBACK_METRICS["steps_failed"]} -eq 0 ] && echo "SUCCESS" || echo "PARTIAL_FAILURE")
    
    IFS=',' read -ra channels <<< "$NOTIFICATION_CHANNELS"
    
    for channel in "${channels[@]}"; do
        case "$channel" in
            slack)
                send_slack_notification "$status" "$severity"
                ;;
            email)
                send_email_notification "$status" "$severity"
                ;;
            pagerduty)
                send_pagerduty_notification "$status" "$severity"
                ;;
            *)
                log "WARN" "Unknown notification channel: $channel"
                ;;
        esac
    done
}

send_slack_notification() {
    local status="$1"
    local severity="$2"
    
    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        log "WARN" "SLACK_WEBHOOK_URL not configured"
        return 0
    fi
    
    local color emoji message
    
    case "$status" in
        SUCCESS)
            color="good"
            emoji="✅"
            message="Rollback completed successfully"
            ;;
        PARTIAL_FAILURE)
            color="warning"
            emoji="⚠️"
            message="Rollback completed with some failures"
            ;;
        *)
            color="danger"
            emoji="❌"
            message="Rollback failed"
            ;;
    esac
    
    local payload
    payload=$(cat << EOF
{
  "attachments": [
    {
      "color": "$color",
      "title": "$emoji Quantum Intelligence Rollback - $status",
      "fields": [
        {
          "title": "Environment",
          "value": "$ENVIRONMENT",
          "short": true
        },
        {
          "title": "Incident ID",
          "value": "$INCIDENT_ID",
          "short": true
        },
        {
          "title": "Reason",
          "value": "$REASON",
          "short": true
        },
        {
          "title": "Severity",
          "value": "${SEVERITY_LEVELS[$severity]}",
          "short": true
        },
        {
          "title": "Steps Completed",
          "value": "${ROLLBACK_METRICS[steps_completed]}",
          "short": true
        },
        {
          "title": "Steps Failed",
          "value": "${ROLLBACK_METRICS[steps_failed]}",
          "short": true
        },
        {
          "title": "Services Affected",
          "value": "${ROLLBACK_METRICS[services_affected]}",
          "short": true
        }
      ],
      "text": "$message",
      "footer": "Quantum Intelligence Rollback System",
      "ts": $(date +%s)
    }
  ]
}
EOF
    )
    
    curl -X POST -H 'Content-type: application/json' \
         --data "$payload" \
         "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || {
        log "WARN" "Failed to send Slack notification"
    }
}

send_email_notification() {
    local status="$1"
    local severity="$2"
    
    log "DEBUG" "Email notification would be sent (not implemented in demo)"
}

send_pagerduty_notification() {
    local status="$1"
    local severity="$2"
    
    log "DEBUG" "PagerDuty notification would be sent (not implemented in demo)"
}

generate_rollback_report() {
    log "INFO" "📋 Generating rollback report..."
    
    local end_time=$(date +%s)
    local duration=$((end_time - ROLLBACK_METRICS["start_time"]))
    ROLLBACK_METRICS["downtime_seconds"]=$duration
    
    local report_file="${INCIDENT_DIR}/rollback-report-${INCIDENT_ID}.json"
    
    cat > "$report_file" << EOF
{
  "rollback_summary": {
    "incident_id": "$INCIDENT_ID",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "reason": "$REASON",
    "start_time": "$(date -d @${ROLLBACK_METRICS["start_time"]} -Iseconds)",
    "end_time": "$(date -Iseconds)",
    "duration_seconds": $duration,
    "status": "$([ ${ROLLBACK_METRICS["steps_failed"]} -eq 0 ] && echo "success" || echo "partial_failure")"
  },
  "metrics": {
    "steps_completed": ${ROLLBACK_METRICS["steps_completed"]},
    "steps_failed": ${ROLLBACK_METRICS["steps_failed"]},
    "services_affected": ${ROLLBACK_METRICS["services_affected"]},
    "data_preserved": "${ROLLBACK_METRICS["data_preserved"]}",
    "downtime_seconds": $duration
  },
  "configuration": {
    "preserve_logs": "$PRESERVE_LOGS",
    "auto_scale_down": "$AUTO_SCALE_DOWN",
    "backup_before_rollback": "$BACKUP_BEFORE_ROLLBACK",
    "notification_channels": "$NOTIFICATION_CHANNELS",
    "rollback_timeout": $ROLLBACK_TIMEOUT,
    "health_check_attempts": $HEALTH_CHECK_ATTEMPTS
  }
}
EOF
    
    log "SUCCESS" "Rollback report generated: $report_file"
    echo "$report_file"
}

main() {
    log "INFO" "🔄 Starting Quantum Intelligence Rollback Automation..."
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Namespace: $NAMESPACE"
    log "INFO" "Reason: $REASON"
    log "INFO" "Incident ID: $INCIDENT_ID"
    
    # Validation and setup
    validate_environment
    check_rollback_prerequisites
    create_incident_record
    
    # Pre-rollback procedures
    backup_current_state
    identify_rollback_targets
    
    # Rollback execution
    perform_quantum_aware_rollback
    
    # Validation and reporting
    local validation_failures=0
    validate_rollback_success || validation_failures=$?
    
    # Notifications
    send_notifications
    
    # Generate final report
    local report_file
    report_file=$(generate_rollback_report)
    
    # Final status
    if [[ $validation_failures -eq 0 && ${ROLLBACK_METRICS["steps_failed"]} -eq 0 ]]; then
        log "SUCCESS" "🎉 Rollback completed successfully!"
        log "INFO" "📄 Report: $report_file"
        exit 0
    else
        log "ERROR" "⚠️  Rollback completed with issues (validation failures: $validation_failures, step failures: ${ROLLBACK_METRICS["steps_failed"]})"
        log "INFO" "📄 Report: $report_file"
        exit 1
    fi
}

# Script execution
trap 'log "ERROR" "Rollback script failed at line $LINENO"' ERR
parse_args "$@"
main