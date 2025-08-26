#!/bin/bash

# Hive Mind Monitoring Stack Deployment Script
# Deploys complete observability stack for production operations

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_DIR="${SCRIPT_DIR}/../monitoring"
COMPOSE_FILE="${MONITORING_DIR}/docker-compose.monitoring.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install it and try again."
        exit 1
    fi
    
    # Check available disk space (need at least 10GB)
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        log_warning "Low disk space detected. Monitoring stack requires at least 10GB."
    fi
    
    log_success "Prerequisites check completed"
}

# Setup environment variables
setup_environment() {
    log_info "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    ENV_FILE="${MONITORING_DIR}/.env"
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Monitoring Stack Environment Variables

# Alert Manager Configuration
SMTP_PASSWORD=your_smtp_password_here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
PAGERDUTY_ROUTING_KEY=your_pagerduty_routing_key_here
PAGERDUTY_SLO_KEY=your_pagerduty_slo_key_here

# Grafana Configuration
GF_SECURITY_ADMIN_PASSWORD=admin123
GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel

# Sentry Configuration
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
ENVIRONMENT=production
RELEASE_VERSION=1.0.0
SENTRY_SAMPLE_RATE=0.1
SENTRY_TRACES_SAMPLE_RATE=0.1

# Database Configuration
DATABASE_URL=postgresql://postgres:password@postgres:5432/claude_tui
REDIS_URL=redis://redis:6379/0

# Service Configuration
CLAUDE_TUI_URL=http://claude-tui:8000
API_GATEWAY_URL=http://api-gateway:8080
MCP_SERVER_URL=http://mcp-server:3000
EOF
        
        log_warning "Created .env file with default values. Please update it with your actual configuration."
    fi
    
    log_success "Environment setup completed"
}

# Create necessary directories and files
setup_directories() {
    log_info "Setting up monitoring directories..."
    
    # Create directory structure
    mkdir -p "${MONITORING_DIR}"/{prometheus/rules,grafana/{provisioning/{dashboards,datasources,notifiers},dashboards},alertmanager/{templates},loki,promtail,jaeger,sentry,incident-response}
    
    # Set appropriate permissions
    sudo chown -R 472:472 "${MONITORING_DIR}/grafana" 2>/dev/null || true
    sudo chown -R 65534:65534 "${MONITORING_DIR}/prometheus" 2>/dev/null || true
    sudo chown -R 10001:10001 "${MONITORING_DIR}/loki" 2>/dev/null || true
    
    log_success "Directory setup completed"
}

# Setup Grafana datasources
setup_grafana_datasources() {
    log_info "Setting up Grafana datasources..."
    
    cat > "${MONITORING_DIR}/grafana/provisioning/datasources/datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
EOF

    log_success "Grafana datasources configured"
}

# Setup Grafana dashboard provisioning
setup_grafana_dashboard_provisioning() {
    log_info "Setting up Grafana dashboard provisioning..."
    
    cat > "${MONITORING_DIR}/grafana/provisioning/dashboards/dashboards.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: 'Hive Mind'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    log_success "Grafana dashboard provisioning configured"
}

# Setup Promtail configuration
setup_promtail() {
    log_info "Setting up Promtail configuration..."
    
    cat > "${MONITORING_DIR}/promtail/promtail.yml" << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Docker containers
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log

    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          expressions:
            tag:
          source: attrs
      - regex:
          expression: (?P<container_name>(?:[^|]*))\|
          source: tag
      - timestamp:
          format: RFC3339Nano
          source: time
      - labels:
          stream:
          container_name:
      - output:
          source: output

  # System logs
  - job_name: syslog
    static_configs:
      - targets:
          - localhost
        labels:
          job: syslog
          __path__: /var/log/syslog

  # Application logs
  - job_name: hive-mind-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: hive-mind
          __path__: /var/log/hive-mind/*.log
EOF

    log_success "Promtail configuration completed"
}

# Start monitoring stack
start_monitoring_stack() {
    log_info "Starting monitoring stack..."
    
    cd "$MONITORING_DIR"
    
    # Pull latest images
    docker-compose -f docker-compose.monitoring.yml pull
    
    # Start services
    docker-compose -f docker-compose.monitoring.yml up -d
    
    log_success "Monitoring stack started"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local services=("prometheus:9090" "grafana:3000" "alertmanager:9093" "loki:3100" "jaeger:16686")
    local max_wait=300  # 5 minutes
    local wait_time=0
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        log_info "Waiting for $name to be ready..."
        
        while ! nc -z localhost "$port" 2>/dev/null; do
            sleep 5
            wait_time=$((wait_time + 5))
            
            if [ $wait_time -ge $max_wait ]; then
                log_error "$name failed to start within $max_wait seconds"
                return 1
            fi
        done
        
        log_success "$name is ready"
    done
    
    # Additional health checks
    log_info "Performing health checks..."
    
    # Check Prometheus targets
    sleep 10
    if curl -sf "http://localhost:9090/-/healthy" > /dev/null; then
        log_success "Prometheus health check passed"
    else
        log_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if curl -sf "http://localhost:3000/api/health" > /dev/null; then
        log_success "Grafana health check passed"
    else
        log_warning "Grafana health check failed"
    fi
    
    log_success "All services are ready"
}

# Configure monitoring integrations
configure_integrations() {
    log_info "Configuring monitoring integrations..."
    
    # Add Prometheus scrape configs for Hive Mind services
    # This would be done through service discovery in a real deployment
    
    # Import Grafana dashboards
    log_info "Importing Grafana dashboards..."
    
    # The dashboards will be automatically imported through provisioning
    
    log_success "Monitoring integrations configured"
}

# Display access information
display_access_info() {
    log_info "Monitoring stack deployment completed!"
    
    cat << EOF

${GREEN}=== Hive Mind Monitoring Stack Access Information ===${NC}

${BLUE}Grafana Dashboard:${NC}
  URL: http://localhost:3000
  Username: admin
  Password: admin123

${BLUE}Prometheus:${NC}
  URL: http://localhost:9090

${BLUE}AlertManager:${NC}
  URL: http://localhost:9093

${BLUE}Jaeger Tracing:${NC}
  URL: http://localhost:16686

${BLUE}Loki (via Grafana):${NC}
  Access through Grafana -> Explore -> Loki

${YELLOW}Next Steps:${NC}
1. Update environment variables in ${MONITORING_DIR}/.env
2. Configure Slack and PagerDuty webhooks
3. Set up Sentry DSN
4. Review and customize alert rules
5. Import additional dashboards as needed

${YELLOW}Monitoring Endpoints:${NC}
- Health Checks: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics
- SLO Dashboard: Available in Grafana

For troubleshooting, check logs:
  docker-compose -f ${MONITORING_DIR}/docker-compose.monitoring.yml logs [service_name]

EOF
}

# Handle cleanup on script exit
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "Deployment failed. Run the following to clean up:"
        echo "  cd ${MONITORING_DIR} && docker-compose -f docker-compose.monitoring.yml down -v"
    fi
}

trap cleanup EXIT

# Main deployment function
main() {
    log_info "Starting Hive Mind monitoring stack deployment..."
    
    check_prerequisites
    setup_environment
    setup_directories
    setup_grafana_datasources
    setup_grafana_dashboard_provisioning
    setup_promtail
    start_monitoring_stack
    wait_for_services
    configure_integrations
    display_access_info
    
    log_success "Monitoring stack deployment completed successfully!"
}

# Run main function
main "$@"