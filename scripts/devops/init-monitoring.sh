#!/bin/bash

# Claude TIU Monitoring Stack Initialization
# Sets up Prometheus, Grafana, and alerting infrastructure

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

create_monitoring_directories() {
    log INFO "Creating monitoring directories..."
    
    # Create directory structure
    mkdir -p "${MONITORING_DIR}"/{prometheus/{data,rules},grafana/{data,dashboards,provisioning/{datasources,dashboards,notifiers}},alertmanager/{data,templates},loki/data}
    
    # Set permissions
    chmod 755 "${MONITORING_DIR}"/*/data
    
    log INFO "Monitoring directories created"
}

setup_prometheus_config() {
    log INFO "Setting up Prometheus configuration..."
    
    # Copy rules
    cp -r "${CONFIG_DIR}/rules" "${MONITORING_DIR}/prometheus/"
    
    # Create prometheus data directory with correct permissions
    sudo chown -R 65534:65534 "${MONITORING_DIR}/prometheus/data" || log WARN "Could not set Prometheus data ownership"
    
    log INFO "Prometheus configuration complete"
}

setup_grafana_config() {
    log INFO "Setting up Grafana configuration..."
    
    # Create Grafana provisioning configs
    cat > "${MONITORING_DIR}/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

deleteDatasources:
  - name: Prometheus
    orgId: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "15s"
      queryTimeout: "60s"
      httpMethod: "POST"
    version: 1

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    isDefault: false
    editable: true
    jsonData:
      maxLines: 1000
    version: 1
EOF

    cat > "${MONITORING_DIR}/grafana/provisioning/dashboards/default.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    # Copy dashboards
    cp -r "${CONFIG_DIR}/grafana/dashboards/"* "${MONITORING_DIR}/grafana/dashboards/"
    
    # Set Grafana permissions
    sudo chown -R 472:472 "${MONITORING_DIR}/grafana" || log WARN "Could not set Grafana ownership"
    
    log INFO "Grafana configuration complete"
}

setup_alertmanager_config() {
    log INFO "Setting up Alertmanager configuration..."
    
    cat > "${MONITORING_DIR}/alertmanager/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@claude-tiu.dev'
  smtp_auth_username: ''
  smtp_auth_password: ''
  
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default-receiver'
  routes:
    - match:
        severity: critical
      receiver: 'critical-receiver'
      continue: true
    - match:
        severity: warning
      receiver: 'warning-receiver'

receivers:
  - name: 'default-receiver'
    webhook_configs:
      - url: 'http://claude-tiu:8000/api/v1/alerts/webhook'
        send_resolved: true
  
  - name: 'critical-receiver'
    email_configs:
      - to: 'ops-critical@claude-tiu.dev'
        subject: '🚨 CRITICAL Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        body: |
          {{ range .Alerts }}
          **Alert:** {{ .Annotations.summary }}
          **Description:** {{ .Annotations.description }}
          **Severity:** {{ .Labels.severity }}
          **Service:** {{ .Labels.service }}
          **Instance:** {{ .Labels.instance }}
          **Time:** {{ .StartsAt }}
          {{ end }}
    webhook_configs:
      - url: 'http://claude-tiu:8000/api/v1/alerts/critical'
        send_resolved: true
  
  - name: 'warning-receiver'
    email_configs:
      - to: 'ops-warnings@claude-tiu.dev'
        subject: '⚠️ Warning Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
EOF

    # Set Alertmanager permissions
    sudo chown -R 65534:65534 "${MONITORING_DIR}/alertmanager/data" || log WARN "Could not set Alertmanager data ownership"
    
    log INFO "Alertmanager configuration complete"
}

setup_loki_config() {
    log INFO "Setting up Loki configuration..."
    
    # Loki config is already in the main config directory
    # Just set up data directory permissions
    sudo chown -R 10001:10001 "${MONITORING_DIR}/loki/data" || log WARN "Could not set Loki data ownership"
    
    log INFO "Loki configuration complete"
}

create_docker_monitoring_compose() {
    log INFO "Creating monitoring Docker Compose override..."
    
    cat > "${PROJECT_ROOT}/docker-compose.monitoring.yml" << 'EOF'
version: '3.8'

services:
  # Prometheus - Metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: claude-tiu-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/rules:/etc/prometheus/rules:ro
      - ./monitoring/prometheus/data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - claude-network
    restart: unless-stopped
    depends_on:
      - claude-tiu

  # Grafana - Visualization
  grafana:
    image: grafana/grafana:latest
    container_name: claude-tiu-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin123}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - ./monitoring/grafana/data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      - claude-network
    restart: unless-stopped
    depends_on:
      - prometheus

  # Alertmanager - Alert handling
  alertmanager:
    image: prom/alertmanager:latest
    container_name: claude-tiu-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - ./monitoring/alertmanager/data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
      - '--cluster.advertise-address=0.0.0.0:9093'
    networks:
      - claude-network
    restart: unless-stopped

  # Loki - Log aggregation
  loki:
    image: grafana/loki:latest
    container_name: claude-tiu-loki
    ports:
      - "3100:3100"
    volumes:
      - ./config/monitoring/loki.yml:/etc/loki/local-config.yaml:ro
      - ./monitoring/loki/data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - claude-network
    restart: unless-stopped

  # Node Exporter - System metrics
  node-exporter:
    image: prom/node-exporter:latest
    container_name: claude-tiu-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - claude-network
    restart: unless-stopped

  # cAdvisor - Container metrics
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: claude-tiu-cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg:/dev/kmsg
    networks:
      - claude-network
    restart: unless-stopped

  # Redis Exporter
  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: claude-tiu-redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://cache:6379
    networks:
      - claude-network
    restart: unless-stopped
    depends_on:
      - cache

  # Postgres Exporter
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: claude-tiu-postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://claude_user:${POSTGRES_PASSWORD:-claude_secure_pass}@db:5432/claude_tiu?sslmode=disable
    networks:
      - claude-network
    restart: unless-stopped
    depends_on:
      - db

networks:
  claude-network:
    external: true
EOF

    log INFO "Monitoring Docker Compose file created"
}

generate_monitoring_readme() {
    log INFO "Generating monitoring documentation..."
    
    cat > "${MONITORING_DIR}/README.md" << 'EOF'
# Claude TIU Monitoring Stack

This directory contains the monitoring infrastructure for Claude TIU.

## Components

### Prometheus (Port 9090)
- **Purpose**: Metrics collection and alerting
- **Configuration**: `prometheus/prometheus.yml`
- **Rules**: `prometheus/rules/`
- **Data**: `prometheus/data/`

### Grafana (Port 3000)
- **Purpose**: Visualization and dashboards
- **Default Login**: admin / admin123 (change in production)
- **Dashboards**: `grafana/dashboards/`
- **Data**: `grafana/data/`

### Alertmanager (Port 9093)
- **Purpose**: Alert routing and notification
- **Configuration**: `alertmanager/alertmanager.yml`
- **Data**: `alertmanager/data/`

### Loki (Port 3100)
- **Purpose**: Log aggregation and querying
- **Configuration**: `loki/loki.yml`
- **Data**: `loki/data/`

## Usage

### Start Monitoring Stack
```bash
# Start all monitoring services
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Or using make
make monitoring
```

### Access Services
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **Loki**: http://localhost:3100

### Key Metrics

#### Application Metrics
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `claude_tiu_active_users` - Active users
- `claude_tiu_ai_requests_total` - AI API requests
- `claude_tiu_errors_total` - Application errors

#### System Metrics
- `node_cpu_seconds_total` - CPU usage
- `node_memory_MemAvailable_bytes` - Available memory
- `node_filesystem_avail_bytes` - Disk space
- `node_load1` - System load

#### Database Metrics
- `pg_stat_database_numbackends` - Database connections
- `pg_stat_database_xact_commit` - Database transactions
- `redis_connected_clients` - Redis connections
- `redis_memory_used_bytes` - Redis memory usage

### Alerting Rules

#### Critical Alerts
- Application down for >2 minutes
- Error rate >10% for >3 minutes
- System out of memory (>90% usage)
- Database connection failures

#### Warning Alerts
- High CPU usage (>80% for >5 minutes)
- High latency (>2s 95th percentile)
- Low disk space (<15%)
- High memory usage (>80%)

### Backup and Restore

#### Backup Monitoring Data
```bash
# Backup Prometheus data
docker-compose exec prometheus tar -czf /prometheus/backup-$(date +%Y%m%d).tar.gz /prometheus

# Backup Grafana dashboards
docker-compose exec grafana tar -czf /var/lib/grafana/backup-$(date +%Y%m%d).tar.gz /var/lib/grafana
```

#### Restore Monitoring Data
```bash
# Restore Prometheus data
docker-compose exec prometheus tar -xzf /prometheus/backup-YYYYMMDD.tar.gz -C /

# Restore Grafana dashboards
docker-compose exec grafana tar -xzf /var/lib/grafana/backup-YYYYMMDD.tar.gz -C /
```

### Troubleshooting

#### Common Issues

1. **Permission Denied Errors**
   ```bash
   # Fix data directory permissions
   sudo chown -R 65534:65534 monitoring/prometheus/data
   sudo chown -R 472:472 monitoring/grafana/data
   sudo chown -R 10001:10001 monitoring/loki/data
   ```

2. **High Memory Usage**
   ```bash
   # Check container memory usage
   docker stats

   # Adjust retention policies in prometheus.yml
   --storage.tsdb.retention.time=15d
   ```

3. **Missing Metrics**
   ```bash
   # Check if application is exposing metrics
   curl http://localhost:8000/metrics

   # Check Prometheus targets
   # Visit http://localhost:9090/targets
   ```

### Configuration Updates

After updating configurations:
```bash
# Reload Prometheus configuration
docker-compose exec prometheus kill -HUP 1

# Restart Grafana (for dashboard updates)
docker-compose restart grafana

# Reload Alertmanager configuration
docker-compose exec alertmanager kill -HUP 1
```

### Performance Tuning

#### Prometheus
- Adjust scrape intervals based on needs
- Configure retention policies for data storage
- Use recording rules for expensive queries

#### Grafana
- Enable caching for dashboards
- Optimize query time ranges
- Use variables for dynamic dashboards

#### Loki
- Configure appropriate retention periods
- Use log stream labels efficiently
- Enable compression for log storage
EOF

    log INFO "Monitoring documentation generated"
}

create_monitoring_scripts() {
    log INFO "Creating monitoring utility scripts..."
    
    # Create monitoring utility scripts directory
    mkdir -p "${MONITORING_DIR}/scripts"
    
    # Health check script
    cat > "${MONITORING_DIR}/scripts/health-check.sh" << 'EOF'
#!/bin/bash
# Monitoring stack health check

echo "Checking monitoring stack health..."

services=("prometheus:9090" "grafana:3000" "alertmanager:9093" "loki:3100")
all_healthy=true

for service in "${services[@]}"; do
    name=${service%:*}
    port=${service#*:}
    
    if curl -s -f "http://localhost:${port}/api/v1/query?query=up" > /dev/null 2>&1 || \
       curl -s -f "http://localhost:${port}" > /dev/null 2>&1; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is unhealthy"
        all_healthy=false
    fi
done

if $all_healthy; then
    echo "✅ All monitoring services are healthy"
    exit 0
else
    echo "❌ Some monitoring services are unhealthy"
    exit 1
fi
EOF

    chmod +x "${MONITORING_DIR}/scripts/health-check.sh"
    
    # Backup script
    cat > "${MONITORING_DIR}/scripts/backup.sh" << 'EOF'
#!/bin/bash
# Backup monitoring data

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating monitoring backup in $BACKUP_DIR..."

# Backup Prometheus data
docker-compose exec -T prometheus tar -czf - /prometheus 2>/dev/null > "$BACKUP_DIR/prometheus.tar.gz"

# Backup Grafana data
docker-compose exec -T grafana tar -czf - /var/lib/grafana 2>/dev/null > "$BACKUP_DIR/grafana.tar.gz"

# Backup Alertmanager data
docker-compose exec -T alertmanager tar -czf - /alertmanager 2>/dev/null > "$BACKUP_DIR/alertmanager.tar.gz"

# Backup configurations
tar -czf "$BACKUP_DIR/configs.tar.gz" config/monitoring/

echo "Backup completed: $BACKUP_DIR"
EOF

    chmod +x "${MONITORING_DIR}/scripts/backup.sh"
    
    log INFO "Monitoring utility scripts created"
}

main() {
    log INFO "Initializing Claude TIU monitoring stack..."
    
    create_monitoring_directories
    setup_prometheus_config
    setup_grafana_config
    setup_alertmanager_config
    setup_loki_config
    create_docker_monitoring_compose
    generate_monitoring_readme
    create_monitoring_scripts
    
    log INFO "Monitoring stack initialization complete!"
    
    echo
    echo "🎉 Monitoring stack is ready!"
    echo
    echo "Next steps:"
    echo "1. Review configuration files in config/monitoring/"
    echo "2. Update environment variables in .env file"
    echo "3. Start monitoring stack: make monitoring"
    echo "4. Access Grafana: http://localhost:3000 (admin/admin123)"
    echo "5. Check Prometheus: http://localhost:9090"
    echo
    echo "For more information, see monitoring/README.md"
}

# Run main function
main "$@"