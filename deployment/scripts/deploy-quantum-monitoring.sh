#!/bin/bash

# 📊 Quantum Intelligence Monitoring Deployment Script
# Comprehensive monitoring stack with Prometheus, Grafana, Jaeger, and custom quantum metrics
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
readonly MONITORING_DIR="${PROJECT_ROOT}/deployment/monitoring"
readonly K8S_DIR="${PROJECT_ROOT}/k8s"
readonly LOGS_DIR="${PROJECT_ROOT}/deployment/logs"

# 🔧 Default values
ENVIRONMENT="staging"
NAMESPACE=""
MONITORING_NAMESPACE="monitoring"
PROMETHEUS="true"
GRAFANA="true"
JAEGER="true"
LOKI="true"
ALERTMANAGER="true"
QUANTUM_DASHBOARDS="true"
PERSISTENCE="true"
EXTERNAL_ACCESS="false"
HELM_TIMEOUT="600s"
VERBOSE="false"

# 📊 Monitoring components configuration
declare -A COMPONENT_VERSIONS=(
    ["prometheus"]="v2.47.0"
    ["grafana"]="10.1.0"
    ["jaeger"]="1.49.0"
    ["loki"]="2.9.0"
    ["alertmanager"]="v0.26.0"
    ["node-exporter"]="1.6.1"
    ["kube-state-metrics"]="2.10.0"
)

declare -A MONITORING_PORTS=(
    ["prometheus"]=9090
    ["grafana"]=3000
    ["jaeger"]=16686
    ["loki"]=3100
    ["alertmanager"]=9093
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
        "MONITOR") echo -e "${timestamp} ${CYAN}[MONITOR]${NC} $message" ;;
    esac
    
    # Log to file
    mkdir -p "$LOGS_DIR"
    echo "${timestamp} [${level}] $message" >> "${LOGS_DIR}/monitoring-deployment-$(date '+%Y%m%d').log"
}

show_help() {
    cat << EOF
📊 Quantum Intelligence Monitoring Deployment Script

USAGE:
    $(basename "$0") [ENVIRONMENT] [OPTIONS]

ARGUMENTS:
    ENVIRONMENT                   Deployment environment (staging|production|canary)

OPTIONS:
    -n, --namespace NAMESPACE     Application namespace [default: claude-tui-ENV]
    -m, --monitoring-namespace NS Monitoring namespace [default: monitoring]
    --prometheus                  Deploy Prometheus (default: true)
    --grafana                     Deploy Grafana (default: true)  
    --jaeger                      Deploy Jaeger tracing (default: true)
    --loki                        Deploy Loki logging (default: true)
    --alertmanager               Deploy Alertmanager (default: true)
    --quantum-dashboards         Deploy quantum intelligence dashboards (default: true)
    --persistence                Enable persistent storage (default: true)
    --external-access            Enable external access via LoadBalancer (default: false)
    --helm-timeout DURATION      Helm operation timeout [default: 600s]
    -v, --verbose                Enable verbose logging
    -h, --help                   Show this help message

EXAMPLES:
    # Basic monitoring stack for staging
    $(basename "$0") staging

    # Full production monitoring with external access
    $(basename "$0") production --external-access --persistence

    # Minimal monitoring for development
    $(basename "$0") staging --no-jaeger --no-loki --no-persistence

    # Custom monitoring namespace
    $(basename "$0") production --monitoring-namespace observability

MONITORING COMPONENTS:
    📊 Prometheus:     Metrics collection and storage
    📈 Grafana:        Visualization and dashboards  
    🔍 Jaeger:         Distributed tracing
    📝 Loki:           Log aggregation
    🚨 Alertmanager:   Alert routing and management
    🧠 Quantum Dashboards: Custom quantum intelligence metrics

For more information, visit: https://github.com/claude-tui/claude-tui/docs/monitoring
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
            -m|--monitoring-namespace)
                MONITORING_NAMESPACE="$2"
                shift 2
                ;;
            --prometheus)
                PROMETHEUS="true"
                shift
                ;;
            --no-prometheus)
                PROMETHEUS="false"
                shift
                ;;
            --grafana)
                GRAFANA="true"
                shift
                ;;
            --no-grafana)
                GRAFANA="false"
                shift
                ;;
            --jaeger)
                JAEGER="true"
                shift
                ;;
            --no-jaeger)
                JAEGER="false"
                shift
                ;;
            --loki)
                LOKI="true"
                shift
                ;;
            --no-loki)
                LOKI="false"
                shift
                ;;
            --alertmanager)
                ALERTMANAGER="true"
                shift
                ;;
            --no-alertmanager)
                ALERTMANAGER="false"
                shift
                ;;
            --quantum-dashboards)
                QUANTUM_DASHBOARDS="true"
                shift
                ;;
            --no-quantum-dashboards)
                QUANTUM_DASHBOARDS="false"
                shift
                ;;
            --persistence)
                PERSISTENCE="true"
                shift
                ;;
            --no-persistence)
                PERSISTENCE="false"
                shift
                ;;
            --external-access)
                EXTERNAL_ACCESS="true"
                shift
                ;;
            --helm-timeout)
                HELM_TIMEOUT="$2"
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
            log "ERROR" "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="claude-tui-${ENVIRONMENT}"
    fi
}

check_dependencies() {
    log "INFO" "🔍 Checking dependencies..."
    
    # Check kubectl
    if ! command -v kubectl >/dev/null 2>&1; then
        log "ERROR" "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm >/dev/null 2>&1; then
        log "ERROR" "helm not found. Please install Helm 3."
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log "ERROR" "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Add Helm repositories
    add_helm_repositories
    
    log "SUCCESS" "Dependencies validated"
}

add_helm_repositories() {
    log "INFO" "📦 Adding Helm repositories..."
    
    # Add required Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
    helm repo add grafana https://grafana.github.io/helm-charts >/dev/null 2>&1 || true
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts >/dev/null 2>&1 || true
    helm repo add bitnami https://charts.bitnami.com/bitnami >/dev/null 2>&1 || true
    
    # Update repositories
    helm repo update >/dev/null 2>&1
    
    log "SUCCESS" "Helm repositories updated"
}

create_namespaces() {
    log "INFO" "🏗️ Creating namespaces..."
    
    # Create monitoring namespace
    if ! kubectl get namespace "$MONITORING_NAMESPACE" >/dev/null 2>&1; then
        kubectl create namespace "$MONITORING_NAMESPACE"
        log "SUCCESS" "Created namespace: $MONITORING_NAMESPACE"
    fi
    
    # Create application namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        kubectl create namespace "$NAMESPACE"
        log "SUCCESS" "Created namespace: $NAMESPACE"
    fi
    
    # Label namespaces for monitoring
    kubectl label namespace "$NAMESPACE" monitoring=enabled --overwrite >/dev/null 2>&1 || true
    kubectl label namespace "$MONITORING_NAMESPACE" monitoring=enabled --overwrite >/dev/null 2>&1 || true
}

deploy_prometheus() {
    if [[ "$PROMETHEUS" != "true" ]]; then
        log "INFO" "Skipping Prometheus deployment"
        return 0
    fi
    
    log "MONITOR" "📊 Deploying Prometheus..."
    
    # Generate Prometheus values file
    local values_file="/tmp/prometheus-values.yaml"
    generate_prometheus_values > "$values_file"
    
    # Deploy Prometheus using Helm
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --values "$values_file" \
        --timeout "$HELM_TIMEOUT" \
        --wait
    
    log "SUCCESS" "Prometheus deployed successfully"
}

generate_prometheus_values() {
    cat << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

prometheus:
  prometheusSpec:
    retention: 30d
    resources:
      requests:
        memory: 2Gi
        cpu: 1000m
      limits:
        memory: 4Gi
        cpu: 2000m
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp2
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi
    additionalScrapeConfigs:
      - job_name: 'quantum-intelligence'
        kubernetes_sd_configs:
          - role: endpoints
            namespaces:
              names:
                - $NAMESPACE
        relabel_configs:
          - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_quantum]
            action: keep
            regex: true

grafana:
  enabled: true
  adminPassword: $(openssl rand -base64 32 | head -c 16)
  persistence:
    enabled: $PERSISTENCE
    size: 10Gi
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: 'quantum-dashboards'
          orgId: 1
          folder: 'Quantum Intelligence'
          type: file
          disableDeletion: false
          editable: true
          allowUiUpdates: true
          options:
            path: /var/lib/grafana/dashboards/quantum

alertmanager:
  enabled: $ALERTMANAGER
  alertmanagerSpec:
    resources:
      requests:
        memory: 256Mi
        cpu: 100m
      limits:
        memory: 512Mi
        cpu: 200m
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: gp2
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 5Gi

nodeExporter:
  enabled: true

kubeStateMetrics:
  enabled: true

coreDns:
  enabled: false

kubeDns:
  enabled: false

kubeEtcd:
  enabled: false

kubeScheduler:
  enabled: false

kubeControllerManager:
  enabled: false
EOF
}

deploy_grafana_dashboards() {
    if [[ "$QUANTUM_DASHBOARDS" != "true" ]]; then
        log "INFO" "Skipping quantum dashboards deployment"
        return 0
    fi
    
    log "MONITOR" "📈 Deploying Quantum Intelligence dashboards..."
    
    # Create ConfigMaps for dashboards
    create_quantum_dashboard_configmaps
    
    # Restart Grafana to load new dashboards
    kubectl rollout restart deployment/prometheus-grafana -n "$MONITORING_NAMESPACE" >/dev/null 2>&1 || true
    
    log "SUCCESS" "Quantum dashboards deployed"
}

create_quantum_dashboard_configmaps() {
    # Neural Swarm Evolution Dashboard
    kubectl create configmap quantum-neural-swarm-dashboard \
        --from-file="${MONITORING_DIR}/grafana/dashboards/neural-swarm-dashboard.json" \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || {
        
        # Create a basic neural swarm dashboard if file doesn't exist
        create_basic_neural_swarm_dashboard
    }
    
    # Adaptive Topology Dashboard
    kubectl create configmap quantum-topology-dashboard \
        --from-literal="topology-dashboard.json"="$(generate_topology_dashboard)" \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
    
    # System Overview Dashboard
    kubectl create configmap quantum-overview-dashboard \
        --from-literal="overview-dashboard.json"="$(generate_overview_dashboard)" \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
}

create_basic_neural_swarm_dashboard() {
    local dashboard_json='
{
  "dashboard": {
    "id": null,
    "title": "Neural Swarm Evolution",
    "tags": ["quantum", "neural-swarm"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Neural Swarm Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "quantum_neural_swarm_evolution_duration_seconds",
            "legendFormat": "Evolution Duration"
          }
        ]
      }
    ]
  }
}'
    
    kubectl create configmap quantum-neural-swarm-dashboard \
        --from-literal="neural-swarm-dashboard.json"="$dashboard_json" \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
}

generate_topology_dashboard() {
    cat << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Adaptive Topology Manager",
    "tags": ["quantum", "topology"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Active Nodes",
        "type": "stat",
        "targets": [
          {
            "expr": "quantum_topology_active_nodes",
            "legendFormat": "Active Nodes"
          }
        ]
      },
      {
        "id": 2,
        "title": "Topology Optimization Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(quantum_topology_optimizations_total[5m])",
            "legendFormat": "Optimizations/sec"
          }
        ]
      }
    ]
  }
}
EOF
}

generate_overview_dashboard() {
    cat << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Quantum Intelligence Overview",
    "tags": ["quantum", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"quantum-intelligence\"}",
            "legendFormat": "System Status"
          }
        ]
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
EOF
}

deploy_jaeger() {
    if [[ "$JAEGER" != "true" ]]; then
        log "INFO" "Skipping Jaeger deployment"
        return 0
    fi
    
    log "MONITOR" "🔍 Deploying Jaeger tracing..."
    
    # Generate Jaeger values file
    local values_file="/tmp/jaeger-values.yaml"
    generate_jaeger_values > "$values_file"
    
    # Deploy Jaeger using Helm
    helm upgrade --install jaeger jaegertracing/jaeger \
        --namespace "$MONITORING_NAMESPACE" \
        --values "$values_file" \
        --timeout "$HELM_TIMEOUT" \
        --wait
    
    log "SUCCESS" "Jaeger deployed successfully"
}

generate_jaeger_values() {
    cat << EOF
provisionDataStore:
  cassandra: false
  elasticsearch: true

elasticsearch:
  replicas: 1
  minimumMasterNodes: 1
  resources:
    requests:
      memory: 1Gi
      cpu: 500m
    limits:
      memory: 2Gi
      cpu: 1000m

storage:
  type: elasticsearch
  elasticsearch:
    host: jaeger-elasticsearch-master
    port: 9200

query:
  service:
    type: $([ "$EXTERNAL_ACCESS" == "true" ] && echo "LoadBalancer" || echo "ClusterIP")
  resources:
    requests:
      memory: 256Mi
      cpu: 100m
    limits:
      memory: 512Mi
      cpu: 200m

collector:
  service:
    grpc:
      port: 14250
    http:
      port: 14268
  resources:
    requests:
      memory: 256Mi
      cpu: 100m
    limits:
      memory: 512Mi
      cpu: 200m

agent:
  resources:
    requests:
      memory: 128Mi
      cpu: 50m
    limits:
      memory: 256Mi
      cpu: 100m
EOF
}

deploy_loki() {
    if [[ "$LOKI" != "true" ]]; then
        log "INFO" "Skipping Loki deployment"
        return 0
    fi
    
    log "MONITOR" "📝 Deploying Loki logging..."
    
    # Generate Loki values file
    local values_file="/tmp/loki-values.yaml"
    generate_loki_values > "$values_file"
    
    # Deploy Loki using Helm
    helm upgrade --install loki grafana/loki-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --values "$values_file" \
        --timeout "$HELM_TIMEOUT" \
        --wait
    
    log "SUCCESS" "Loki deployed successfully"
}

generate_loki_values() {
    cat << EOF
loki:
  enabled: true
  persistence:
    enabled: $PERSISTENCE
    size: 50Gi
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m

promtail:
  enabled: true
  resources:
    requests:
      memory: 128Mi
      cpu: 100m
    limits:
      memory: 256Mi
      cpu: 200m

fluent-bit:
  enabled: false

filebeat:
  enabled: false

logstash:
  enabled: false
EOF
}

configure_service_monitors() {
    log "MONITOR" "🎯 Configuring service monitors..."
    
    # Create ServiceMonitor for Quantum Intelligence modules
    kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: quantum-intelligence
  namespace: $MONITORING_NAMESPACE
  labels:
    app: quantum-intelligence
spec:
  selector:
    matchLabels:
      app: claude-tui
      monitoring: enabled
  namespaceSelector:
    matchNames:
    - $NAMESPACE
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
    relabelings:
    - sourceLabels: [__meta_kubernetes_service_annotation_prometheus_io_quantum]
      action: keep
      regex: true
EOF
    
    log "SUCCESS" "Service monitors configured"
}

configure_alerting_rules() {
    if [[ "$ALERTMANAGER" != "true" ]]; then
        log "INFO" "Skipping alerting rules configuration"
        return 0
    fi
    
    log "MONITOR" "🚨 Configuring alerting rules..."
    
    # Create PrometheusRule for Quantum Intelligence alerts
    kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: quantum-intelligence-alerts
  namespace: $MONITORING_NAMESPACE
  labels:
    app: quantum-intelligence
    prometheus: kube-prometheus
    role: alert-rules
spec:
  groups:
  - name: quantum-intelligence
    rules:
    - alert: QuantumModuleDown
      expr: up{job="quantum-intelligence"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Quantum Intelligence module is down"
        description: "Quantum Intelligence module has been down for more than 5 minutes."
    
    - alert: NeuralSwarmHighLatency
      expr: quantum_neural_swarm_evolution_duration_seconds > 5
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "Neural Swarm Evolution high latency"
        description: "Neural Swarm Evolution is taking more than 5 seconds."
    
    - alert: AdaptiveTopologyOptimizationFailures
      expr: rate(quantum_topology_optimization_failures_total[5m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High topology optimization failure rate"
        description: "Adaptive Topology Manager is experiencing high failure rates."
    
    - alert: QuantumSystemHighMemoryUsage
      expr: container_memory_usage_bytes{pod=~"claude-tui-.*"} / container_spec_memory_limit_bytes > 0.8
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage in Quantum Intelligence system"
        description: "Memory usage is above 80% for Quantum Intelligence pods."
    
    - alert: QuantumSystemHighCPUUsage
      expr: rate(container_cpu_usage_seconds_total{pod=~"claude-tui-.*"}[5m]) > 0.8
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High CPU usage in Quantum Intelligence system"
        description: "CPU usage is above 80% for Quantum Intelligence pods."
EOF
    
    log "SUCCESS" "Alerting rules configured"
}

setup_external_access() {
    if [[ "$EXTERNAL_ACCESS" != "true" ]]; then
        log "INFO" "Skipping external access setup"
        return 0
    fi
    
    log "MONITOR" "🌐 Setting up external access..."
    
    # Create Ingress for monitoring services
    kubectl apply -f - << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: $MONITORING_NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - monitoring.${ENVIRONMENT}.claude-tui.quantum.ai
    secretName: monitoring-tls
  rules:
  - host: monitoring.${ENVIRONMENT}.claude-tui.quantum.ai
    http:
      paths:
      - path: /grafana
        pathType: Prefix
        backend:
          service:
            name: prometheus-grafana
            port:
              number: 80
      - path: /prometheus
        pathType: Prefix
        backend:
          service:
            name: prometheus-kube-prometheus-prometheus
            port:
              number: 9090
      - path: /jaeger
        pathType: Prefix
        backend:
          service:
            name: jaeger-query
            port:
              number: 16686
EOF
    
    log "SUCCESS" "External access configured"
}

validate_monitoring_deployment() {
    log "INFO" "🔍 Validating monitoring deployment..."
    
    local failed_components=()
    
    # Check Prometheus
    if [[ "$PROMETHEUS" == "true" ]]; then
        if ! kubectl get deployment prometheus-kube-prometheus-operator -n "$MONITORING_NAMESPACE" >/dev/null 2>&1; then
            failed_components+=("Prometheus")
        fi
    fi
    
    # Check Grafana
    if [[ "$GRAFANA" == "true" ]]; then
        if ! kubectl get deployment prometheus-grafana -n "$MONITORING_NAMESPACE" >/dev/null 2>&1; then
            failed_components+=("Grafana")
        fi
    fi
    
    # Check Jaeger
    if [[ "$JAEGER" == "true" ]]; then
        if ! kubectl get deployment jaeger-query -n "$MONITORING_NAMESPACE" >/dev/null 2>&1; then
            failed_components+=("Jaeger")
        fi
    fi
    
    # Check Loki
    if [[ "$LOKI" == "true" ]]; then
        if ! kubectl get statefulset loki -n "$MONITORING_NAMESPACE" >/dev/null 2>&1; then
            failed_components+=("Loki")
        fi
    fi
    
    if [[ ${#failed_components[@]} -eq 0 ]]; then
        log "SUCCESS" "All monitoring components deployed successfully"
        return 0
    else
        log "ERROR" "Failed to deploy components: ${failed_components[*]}"
        return 1
    fi
}

display_access_information() {
    log "INFO" "📋 Monitoring Access Information:"
    
    if [[ "$PROMETHEUS" == "true" ]]; then
        local prometheus_port
        prometheus_port=$(kubectl get svc prometheus-kube-prometheus-prometheus -n "$MONITORING_NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
        log "INFO" "📊 Prometheus: kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n $MONITORING_NAMESPACE $prometheus_port:9090"
    fi
    
    if [[ "$GRAFANA" == "true" ]]; then
        local grafana_port
        grafana_port=$(kubectl get svc prometheus-grafana -n "$MONITORING_NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
        local grafana_password
        grafana_password=$(kubectl get secret prometheus-grafana -n "$MONITORING_NAMESPACE" -o jsonpath='{.data.admin-password}' | base64 -d)
        log "INFO" "📈 Grafana: kubectl port-forward svc/prometheus-grafana -n $MONITORING_NAMESPACE $grafana_port:80"
        log "INFO" "   Username: admin, Password: $grafana_password"
    fi
    
    if [[ "$JAEGER" == "true" ]]; then
        local jaeger_port
        jaeger_port=$(kubectl get svc jaeger-query -n "$MONITORING_NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "16686")
        log "INFO" "🔍 Jaeger: kubectl port-forward svc/jaeger-query -n $MONITORING_NAMESPACE $jaeger_port:16686"
    fi
    
    if [[ "$EXTERNAL_ACCESS" == "true" ]]; then
        log "INFO" "🌐 External Access: https://monitoring.${ENVIRONMENT}.claude-tui.quantum.ai"
    fi
}

main() {
    log "INFO" "📊 Starting Quantum Intelligence Monitoring Deployment..."
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Application Namespace: $NAMESPACE"
    log "INFO" "Monitoring Namespace: $MONITORING_NAMESPACE"
    
    # Validation and setup
    validate_environment
    check_dependencies
    create_namespaces
    
    # Deploy monitoring components
    deploy_prometheus
    deploy_grafana_dashboards
    deploy_jaeger
    deploy_loki
    
    # Configure monitoring
    configure_service_monitors
    configure_alerting_rules
    setup_external_access
    
    # Validation
    if validate_monitoring_deployment; then
        log "SUCCESS" "🎉 Monitoring deployment completed successfully!"
        display_access_information
        exit 0
    else
        log "ERROR" "❌ Monitoring deployment failed"
        exit 1
    fi
}

# Script execution
trap 'log "ERROR" "Monitoring deployment script failed at line $LINENO"' ERR
parse_args "$@"
main