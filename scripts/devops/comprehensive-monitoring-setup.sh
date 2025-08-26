#!/bin/bash
# Comprehensive Monitoring Setup for Claude-TUI
# Monitoring Engineer Implementation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MONITORING_DIR="${PROJECT_ROOT}/monitoring"
CONFIG_DIR="${PROJECT_ROOT}/config/monitoring"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        ERROR)
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            ;;
        WARN)
            echo -e "${YELLOW}[$timestamp] WARN: $message${NC}"
            ;;
        INFO)
            echo -e "${GREEN}[$timestamp] INFO: $message${NC}"
            ;;
        DEBUG)
            echo -e "${BLUE}[$timestamp] DEBUG: $message${NC}"
            ;;
    esac
}

# Monitoring Engineer: Advanced Alerting Rules
setup_advanced_alerting() {
    log INFO "Setting up advanced alerting rules..."
    
    mkdir -p "${CONFIG_DIR}/rules"
    
    cat > "${CONFIG_DIR}/rules/claude-tui-sla-alerts.yml" << 'EOF'
groups:
- name: claude-tui-sla
  rules:
  # 99.9% Uptime SLA Monitoring
  - alert: SLAViolationCritical
    expr: (1 - rate(http_requests_total{code!~"5.."}[5m]) / rate(http_requests_total[5m])) > 0.001
    for: 2m
    labels:
      severity: critical
      sla: "99.9%"
    annotations:
      summary: "SLA violation - Error rate exceeds 0.1%"
      description: "Application error rate is {{ $value | humanizePercentage }} for the last 2 minutes"
      runbook_url: "https://docs.claude-tui.dev/runbooks/sla-violation"

  # Response Time SLA
  - alert: ResponseTimeSLAViolation
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2.0
    for: 3m
    labels:
      severity: warning
      sla: "response-time"
    annotations:
      summary: "95th percentile response time exceeds 2 seconds"
      description: "95th percentile response time is {{ $value }}s"

  # Database Connection SLA
  - alert: DatabaseSLAViolation
    expr: up{job="postgres-exporter"} == 0
    for: 1m
    labels:
      severity: critical
      sla: "database"
    annotations:
      summary: "Database is down - SLA violation"
      description: "PostgreSQL database has been down for more than 1 minute"

- name: claude-tui-business-metrics
  rules:
  # AI Request Success Rate
  - alert: AIRequestFailureRate
    expr: rate(claude_tui_ai_requests_total{status="error"}[5m]) / rate(claude_tui_ai_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
      component: "ai-integration"
    annotations:
      summary: "High AI request failure rate"
      description: "AI request failure rate is {{ $value | humanizePercentage }} over the last 5 minutes"

  # User Experience Alerts
  - alert: UserSessionFailures
    expr: rate(user_session_failures_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
      component: "user-experience"
    annotations:
      summary: "High user session failure rate"
      description: "User session failures are at {{ $value }} per second"

- name: claude-tui-security
  rules:
  # Security Breach Detection
  - alert: SecurityBreach
    expr: rate(security_events_total{type="breach"}[1m]) > 0
    for: 0m
    labels:
      severity: critical
      component: "security"
    annotations:
      summary: "Security breach detected"
      description: "Security breach event detected: {{ $labels.source }}"

  # Anomalous Authentication Attempts
  - alert: AnomalousAuthAttempts
    expr: rate(auth_attempts_total{status="failed"}[5m]) > 10
    for: 2m
    labels:
      severity: warning
      component: "authentication"
    annotations:
      summary: "High number of failed authentication attempts"
      description: "{{ $value }} failed authentication attempts per second"
EOF

    log INFO "Advanced alerting rules created"
}

# Setup comprehensive dashboards
setup_comprehensive_dashboards() {
    log INFO "Setting up comprehensive monitoring dashboards..."
    
    mkdir -p "${CONFIG_DIR}/grafana/dashboards"
    
    # Main application dashboard
    cat > "${CONFIG_DIR}/grafana/dashboards/claude-tui-executive.json" << 'EOF'
{
  "dashboard": {
    "title": "Claude-TUI Executive Dashboard",
    "tags": ["executive", "sla", "business"],
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "title": "Uptime SLA",
        "type": "singlestat",
        "targets": [
          {
            "expr": "1 - (rate(http_requests_total{code=~\"5..\"}[24h]) / rate(http_requests_total[24h]))",
            "legendFormat": "Uptime %"
          }
        ],
        "valueMaps": [
          {
            "value": "null",
            "text": "100%"
          }
        ],
        "thresholds": "99.9,99.95",
        "colorBackground": true,
        "format": "percentunit"
      },
      {
        "title": "Response Time P95",
        "type": "singlestat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95 Response Time"
          }
        ],
        "unit": "s",
        "thresholds": "1,2",
        "colorBackground": true
      },
      {
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "claude_tui_active_users",
            "legendFormat": "Active Users"
          }
        ],
        "colorBackground": false
      },
      {
        "title": "AI Request Success Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(claude_tui_ai_requests_total{status=\"success\"}[5m]) / rate(claude_tui_ai_requests_total[5m])",
            "legendFormat": "Success Rate"
          }
        ],
        "format": "percentunit",
        "thresholds": "0.95,0.99",
        "colorBackground": true
      }
    ]
  }
}
EOF

    # Technical operations dashboard
    cat > "${CONFIG_DIR}/grafana/dashboards/claude-tui-operations.json" << 'EOF'
{
  "dashboard": {
    "title": "Claude-TUI Operations Dashboard",
    "tags": ["operations", "technical", "infrastructure"],
    "refresh": "15s",
    "panels": [
      {
        "title": "HTTP Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{code}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_memory_rss_bytes / 1024 / 1024",
            "legendFormat": "RSS Memory (MB)"
          },
          {
            "expr": "process_memory_heap_bytes / 1024 / 1024",
            "legendFormat": "Heap Memory (MB)"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Container Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m]) * 100",
            "legendFormat": "CPU Usage %"
          }
        ]
      }
    ]
  }
}
EOF

    log INFO "Comprehensive dashboards created"
}

# Setup log aggregation and analysis
setup_log_analysis() {
    log INFO "Setting up advanced log analysis..."
    
    mkdir -p "${CONFIG_DIR}/loki"
    
    cat > "${CONFIG_DIR}/loki/advanced-config.yaml" << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 168h

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules
  alertmanager_url: http://alertmanager:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true
EOF

    # Create log analysis rules
    cat > "${CONFIG_DIR}/loki/rules/claude-tui-log-alerts.yaml" << 'EOF'
groups:
- name: claude-tui-logs
  rules:
  - alert: HighErrorRate
    expr: |
      sum(rate({job="claude-tui"} |= "ERROR" [5m])) by (instance)
      /
      sum(rate({job="claude-tui"} [5m])) by (instance)
      > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate in logs"
      description: "Error rate is {{ $value | humanizePercentage }} on {{ $labels.instance }}"

  - alert: DatabaseConnectionError
    expr: |
      sum(rate({job="claude-tui"} |~ "database.*connection.*error" [5m])) > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection errors detected"
      description: "Database connection errors detected in application logs"

  - alert: SecurityIncident
    expr: |
      sum(rate({job="claude-tui"} |~ "security|breach|attack|unauthorized" [1m])) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Security incident detected in logs"
      description: "Potential security incident detected in application logs"
EOF

    log INFO "Advanced log analysis setup completed"
}

# Setup synthetic monitoring
setup_synthetic_monitoring() {
    log INFO "Setting up synthetic monitoring..."
    
    mkdir -p "${MONITORING_DIR}/synthetic"
    
    cat > "${MONITORING_DIR}/synthetic/health-monitor.py" << 'EOF'
#!/usr/bin/env python3
"""
Synthetic Health Monitoring for Claude-TUI
Performs end-to-end health checks and reports metrics to Prometheus
"""

import time
import requests
import json
import sys
from datetime import datetime
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway
import os

# Configuration
PUSHGATEWAY_URL = os.getenv('PUSHGATEWAY_URL', 'http://localhost:9091')
TARGET_URL = os.getenv('TARGET_URL', 'http://localhost:8000')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '30'))

# Metrics
registry = CollectorRegistry()
health_check_success = Gauge('claude_tui_health_check_success', 'Health check success (1=success, 0=failure)', registry=registry)
response_time = Gauge('claude_tui_health_check_response_time_seconds', 'Health check response time', registry=registry)
api_check_success = Gauge('claude_tui_api_check_success', 'API check success', registry=registry)
login_check_success = Gauge('claude_tui_login_check_success', 'Login flow check success', registry=registry)

def check_health():
    """Basic health check"""
    try:
        start_time = time.time()
        response = requests.get(f"{TARGET_URL}/health", timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            health_check_success.set(1)
            response_time.set(end_time - start_time)
            return True, end_time - start_time
        else:
            health_check_success.set(0)
            return False, 0
    except Exception as e:
        print(f"Health check failed: {e}")
        health_check_success.set(0)
        return False, 0

def check_api():
    """Check API endpoints"""
    try:
        # Check API status endpoint
        response = requests.get(f"{TARGET_URL}/api/v1/health", timeout=10)
        if response.status_code == 200:
            api_check_success.set(1)
            return True
        else:
            api_check_success.set(0)
            return False
    except Exception as e:
        print(f"API check failed: {e}")
        api_check_success.set(0)
        return False

def check_login_flow():
    """Check user login flow"""
    try:
        # This would implement actual login flow testing
        # For now, just check if the login page loads
        response = requests.get(f"{TARGET_URL}/auth/login", timeout=10)
        if response.status_code in [200, 302]:  # 302 for redirects
            login_check_success.set(1)
            return True
        else:
            login_check_success.set(0)
            return False
    except Exception as e:
        print(f"Login check failed: {e}")
        login_check_success.set(0)
        return False

def main():
    """Main monitoring loop"""
    print(f"Starting synthetic monitoring for {TARGET_URL}")
    
    while True:
        try:
            timestamp = datetime.now().isoformat()
            
            # Perform checks
            health_ok, resp_time = check_health()
            api_ok = check_api()
            login_ok = check_login_flow()
            
            # Log results
            print(f"[{timestamp}] Health: {'✓' if health_ok else '✗'} "
                  f"API: {'✓' if api_ok else '✗'} "
                  f"Login: {'✓' if login_ok else '✗'} "
                  f"Response: {resp_time:.3f}s")
            
            # Push metrics to Pushgateway
            push_to_gateway(PUSHGATEWAY_URL, job='claude-tui-synthetic', registry=registry)
            
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
EOF

    chmod +x "${MONITORING_DIR}/synthetic/health-monitor.py"
    
    # Create systemd service for synthetic monitoring
    cat > "${MONITORING_DIR}/synthetic/claude-tui-synthetic.service" << 'EOF'
[Unit]
Description=Claude-TUI Synthetic Monitoring
After=network.target

[Service]
Type=simple
User=monitoring
Group=monitoring
ExecStart=/usr/bin/python3 /opt/claude-tui/monitoring/synthetic/health-monitor.py
Restart=always
RestartSec=10
Environment=TARGET_URL=https://claude-tui.dev
Environment=PUSHGATEWAY_URL=http://localhost:9091
Environment=CHECK_INTERVAL=30

[Install]
WantedBy=multi-user.target
EOF

    log INFO "Synthetic monitoring setup completed"
}

# Setup monitoring automation
setup_monitoring_automation() {
    log INFO "Setting up monitoring automation..."
    
    mkdir -p "${MONITORING_DIR}/automation"
    
    # Auto-scaling script
    cat > "${MONITORING_DIR}/automation/auto-scale.sh" << 'EOF'
#!/bin/bash
# Auto-scaling based on metrics

set -euo pipefail

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
NAMESPACE="${NAMESPACE:-production}"
MIN_REPLICAS="${MIN_REPLICAS:-2}"
MAX_REPLICAS="${MAX_REPLICAS:-10}"
CPU_THRESHOLD="${CPU_THRESHOLD:-70}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-80}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

get_metric() {
    local query="$1"
    curl -s "${PROMETHEUS_URL}/api/v1/query?query=${query}" | \
        jq -r '.data.result[0].value[1] // "0"'
}

scale_deployment() {
    local deployment="$1"
    local replicas="$2"
    
    log "Scaling $deployment to $replicas replicas"
    kubectl scale deployment "$deployment" --replicas="$replicas" -n "$NAMESPACE"
}

main() {
    log "Starting auto-scaling check..."
    
    # Get current metrics
    cpu_usage=$(get_metric "avg(rate(container_cpu_usage_seconds_total{pod=~\"claude-tui-.*\"}[5m])) * 100")
    memory_usage=$(get_metric "avg(container_memory_working_set_bytes{pod=~\"claude-tui-.*\"} / container_spec_memory_limit_bytes * 100)")
    current_replicas=$(kubectl get deployment claude-tui-app -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    
    log "Current metrics - CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Replicas: $current_replicas"
    
    # Scaling logic
    if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )) || (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
        if (( current_replicas < MAX_REPLICAS )); then
            new_replicas=$((current_replicas + 1))
            scale_deployment "claude-tui-app" "$new_replicas"
            log "Scaled up due to high resource usage"
        else
            log "Already at maximum replicas ($MAX_REPLICAS)"
        fi
    elif (( $(echo "$cpu_usage < 30" | bc -l) )) && (( $(echo "$memory_usage < 40" | bc -l) )); then
        if (( current_replicas > MIN_REPLICAS )); then
            new_replicas=$((current_replicas - 1))
            scale_deployment "claude-tui-app" "$new_replicas"
            log "Scaled down due to low resource usage"
        else
            log "Already at minimum replicas ($MIN_REPLICAS)"
        fi
    else
        log "No scaling action needed"
    fi
}

main "$@"
EOF

    chmod +x "${MONITORING_DIR}/automation/auto-scale.sh"
    
    # Create cron jobs
    cat > "${MONITORING_DIR}/automation/monitoring-cron" << 'EOF'
# Claude-TUI Monitoring Automation
# Auto-scaling check every 2 minutes
*/2 * * * * /opt/claude-tui/monitoring/automation/auto-scale.sh >> /var/log/auto-scale.log 2>&1

# Health checks every minute
* * * * * /opt/claude-tui/monitoring/synthetic/health-monitor.py >> /var/log/synthetic-monitor.log 2>&1

# Backup monitoring data daily at 3 AM
0 3 * * * /opt/claude-tui/monitoring/scripts/backup.sh >> /var/log/monitoring-backup.log 2>&1

# Clean old logs weekly
0 2 * * 0 find /var/log -name "*.log" -mtime +7 -delete
EOF

    log INFO "Monitoring automation setup completed"
}

# Main execution
main() {
    log INFO "Starting Comprehensive Monitoring setup..."
    
    setup_advanced_alerting
    setup_comprehensive_dashboards
    setup_log_analysis
    setup_synthetic_monitoring
    setup_monitoring_automation
    
    log INFO "Comprehensive Monitoring setup completed!"
    
    echo
    echo "🎉 Comprehensive Monitoring is ready!"
    echo
    echo "Features configured:"
    echo "✅ Advanced SLA alerting"
    echo "✅ Executive and operations dashboards"
    echo "✅ Log analysis and alerting"
    echo "✅ Synthetic monitoring"
    echo "✅ Auto-scaling automation"
    echo
    echo "Next steps:"
    echo "1. Deploy monitoring stack: docker-compose -f docker-compose.monitoring.yml up -d"
    echo "2. Install synthetic monitoring: systemctl enable claude-tui-synthetic"
    echo "3. Setup cron jobs: crontab monitoring/automation/monitoring-cron"
    echo "4. Configure alerting endpoints in Grafana"
    echo
}

# Run main function
main "$@"