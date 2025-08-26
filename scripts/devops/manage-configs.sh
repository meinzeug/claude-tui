#!/bin/bash

# Environment Configuration Management Script
# Manages environment-specific configurations for Claude TUI

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"
ENVIRONMENTS_DIR="$CONFIG_DIR/environments"
TEMPLATES_DIR="$CONFIG_DIR/templates"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Help function
show_help() {
    cat << EOF
Environment Configuration Management for Claude TUI

Usage: $0 COMMAND [OPTIONS]

Commands:
    init                    Initialize configuration structure
    validate ENV            Validate environment configuration
    deploy ENV              Deploy configuration to environment
    generate ENV            Generate configuration from template
    backup ENV              Backup current environment configuration
    restore ENV BACKUP      Restore configuration from backup
    diff ENV1 ENV2          Compare configurations between environments
    list                    List available environments
    template NAME           Create new configuration template

Options:
    --dry-run              Show what would be done without executing
    --force                Force operation without confirmation
    --verbose              Enable verbose output
    -h, --help             Show this help message

Environments:
    development            Local development environment
    testing                Testing/CI environment  
    staging                Staging environment
    production             Production environment

Examples:
    $0 init
    $0 validate production
    $0 generate staging
    $0 deploy production --dry-run
    $0 diff staging production

EOF
}

# Initialize configuration structure
init_config_structure() {
    log_info "Initializing configuration structure..."
    
    # Create directories
    mkdir -p "$ENVIRONMENTS_DIR"/{development,testing,staging,production}
    mkdir -p "$TEMPLATES_DIR"
    mkdir -p "$CONFIG_DIR"/{secrets,backups}
    
    # Create base configuration files
    create_base_configs
    create_templates
    
    log_success "Configuration structure initialized"
}

# Create base configuration files
create_base_configs() {
    # Development configuration
    cat > "$ENVIRONMENTS_DIR/development/app.yaml" << 'EOF'
environment: development
debug: true
log_level: DEBUG

database:
  url: sqlite:///./claude_tui_dev.db
  pool_size: 5
  echo: true

redis:
  url: redis://localhost:6379/0
  max_connections: 10

security:
  secret_key: dev-secret-key-change-in-production
  enable_cors: true
  cors_origins: ["http://localhost:3000", "http://127.0.0.1:3000"]

api:
  rate_limit: 1000/minute
  enable_docs: true
  
performance:
  cache_ttl: 300
  async_workers: 2

monitoring:
  enable_metrics: false
  enable_tracing: false
EOF

    # Testing configuration
    cat > "$ENVIRONMENTS_DIR/testing/app.yaml" << 'EOF'
environment: testing
debug: false
log_level: INFO

database:
  url: sqlite:///./test.db
  pool_size: 5
  echo: false

redis:
  url: redis://localhost:6379/1
  max_connections: 10

security:
  secret_key: test-secret-key
  enable_cors: true
  cors_origins: ["*"]

api:
  rate_limit: 10000/minute
  enable_docs: true

performance:
  cache_ttl: 60
  async_workers: 1

monitoring:
  enable_metrics: false
  enable_tracing: false
EOF

    # Staging configuration
    cat > "$ENVIRONMENTS_DIR/staging/app.yaml" << 'EOF'
environment: staging
debug: false
log_level: INFO

database:
  url: postgresql://user:password@staging-db:5432/claude_tui_staging
  pool_size: 20
  max_connections: 50

redis:
  url: redis://staging-redis:6379/0
  max_connections: 50

security:
  secret_key: ${SECRET_KEY}
  enable_cors: true
  cors_origins: ["https://staging.claude-tui.dev"]
  enable_https_redirect: true

api:
  rate_limit: 1000/minute
  enable_docs: true

performance:
  cache_ttl: 3600
  async_workers: 4

monitoring:
  enable_metrics: true
  enable_tracing: true
  metrics_port: 9090
EOF

    # Production configuration
    cat > "$ENVIRONMENTS_DIR/production/app.yaml" << 'EOF'
environment: production
debug: false
log_level: WARNING

database:
  url: postgresql://user:password@prod-db:5432/claude_tui_production
  pool_size: 50
  max_connections: 100
  ssl_mode: require

redis:
  url: redis://prod-redis:6379/0
  max_connections: 100
  ssl: true

security:
  secret_key: ${SECRET_KEY}
  enable_cors: false
  enable_https_redirect: true
  enable_hsts: true
  session_timeout: 3600

api:
  rate_limit: 500/minute
  enable_docs: false

performance:
  cache_ttl: 7200
  async_workers: 8
  enable_compression: true

monitoring:
  enable_metrics: true
  enable_tracing: true
  enable_profiling: false
  metrics_port: 9090
  
backup:
  enable_auto_backup: true
  backup_schedule: "0 2 * * *"
  retention_days: 30
EOF

    # Logging configurations
    for env in development testing staging production; do
        cat > "$ENVIRONMENTS_DIR/$env/logging.yaml" << EOF
version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
  detailed:
    format: '[%(asctime)s] %(levelname)s in %(name)s.%(funcName)s:%(lineno)d: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: $([ "$env" = "development" ] && echo "DEBUG" || echo "INFO")
    formatter: $([ "$env" = "development" ] && echo "detailed" || echo "default")
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: $([ "$env" = "production" ] && echo "WARNING" || echo "INFO")
    formatter: detailed
    filename: /app/logs/claude-tui.log
    maxBytes: 10485760
    backupCount: 5

root:
  level: $([ "$env" = "development" ] && echo "DEBUG" || echo "INFO")
  handlers: [console, file]

loggers:
  claude_tui:
    level: $([ "$env" = "development" ] && echo "DEBUG" || echo "INFO")
    handlers: [console, file]
    propagate: no
  
  uvicorn:
    level: INFO
    handlers: [console, file]
    propagate: no
EOF
    done
    
    log_info "Base configuration files created"
}

# Create configuration templates
create_templates() {
    # Application template
    cat > "$TEMPLATES_DIR/app.yaml.template" << 'EOF'
environment: {{ ENVIRONMENT }}
debug: {{ DEBUG | default(false) }}
log_level: {{ LOG_LEVEL | default("INFO") }}

database:
  url: {{ DATABASE_URL }}
  pool_size: {{ DB_POOL_SIZE | default(20) }}
  max_connections: {{ DB_MAX_CONNECTIONS | default(50) }}
  
redis:
  url: {{ REDIS_URL }}
  max_connections: {{ REDIS_MAX_CONNECTIONS | default(50) }}

security:
  secret_key: {{ SECRET_KEY }}
  enable_cors: {{ ENABLE_CORS | default(false) }}
  cors_origins: {{ CORS_ORIGINS | default([]) }}

api:
  rate_limit: {{ RATE_LIMIT | default("1000/minute") }}
  enable_docs: {{ ENABLE_DOCS | default(false) }}

monitoring:
  enable_metrics: {{ ENABLE_METRICS | default(true) }}
  enable_tracing: {{ ENABLE_TRACING | default(true) }}
EOF

    # Kubernetes template
    cat > "$TEMPLATES_DIR/k8s-deployment.yaml.template" << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-tui-app
  namespace: {{ NAMESPACE }}
  labels:
    app: claude-tui
    environment: {{ ENVIRONMENT }}
spec:
  replicas: {{ REPLICAS }}
  selector:
    matchLabels:
      app: claude-tui
  template:
    metadata:
      labels:
        app: claude-tui
        environment: {{ ENVIRONMENT }}
    spec:
      containers:
      - name: claude-tui
        image: {{ IMAGE }}
        ports:
        - containerPort: 8000
        env:
        - name: CLAUDE_TIU_ENV
          value: {{ ENVIRONMENT }}
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: claude-tui-secrets
              key: secret-key
        resources:
          requests:
            memory: {{ MEMORY_REQUEST }}
            cpu: {{ CPU_REQUEST }}
          limits:
            memory: {{ MEMORY_LIMIT }}
            cpu: {{ CPU_LIMIT }}
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
EOF

    log_info "Configuration templates created"
}

# Validate environment configuration
validate_config() {
    local env="$1"
    local config_file="$ENVIRONMENTS_DIR/$env/app.yaml"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        return 1
    fi
    
    log_info "Validating configuration for environment: $env"
    
    # Check YAML syntax
    if command -v yq &> /dev/null; then
        if ! yq eval '.' "$config_file" > /dev/null; then
            log_error "Invalid YAML syntax in $config_file"
            return 1
        fi
    elif command -v python3 &> /dev/null; then
        if ! python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
            log_error "Invalid YAML syntax in $config_file"
            return 1
        fi
    else
        log_warning "No YAML validator found, skipping syntax check"
    fi
    
    # Validate required fields
    local required_fields=("environment" "database.url" "redis.url" "security.secret_key")
    for field in "${required_fields[@]}"; do
        if command -v yq &> /dev/null; then
            if [[ "$(yq eval ".$field" "$config_file")" == "null" ]]; then
                log_error "Required field missing: $field"
                return 1
            fi
        fi
    done
    
    # Environment-specific validations
    case "$env" in
        "production")
            # Production-specific checks
            if grep -q "dev-secret-key" "$config_file"; then
                log_error "Development secret key detected in production config"
                return 1
            fi
            
            if grep -q "debug: true" "$config_file"; then
                log_error "Debug mode enabled in production config"
                return 1
            fi
            ;;
        "staging")
            # Staging-specific checks
            if grep -q "localhost" "$config_file"; then
                log_warning "Localhost URL detected in staging config"
            fi
            ;;
    esac
    
    log_success "Configuration validation passed for $env"
    return 0
}

# Generate configuration from template
generate_config() {
    local env="$1"
    local template_file="$TEMPLATES_DIR/app.yaml.template"
    local output_file="$ENVIRONMENTS_DIR/$env/app.yaml"
    
    if [[ ! -f "$template_file" ]]; then
        log_error "Template file not found: $template_file"
        return 1
    fi
    
    log_info "Generating configuration for environment: $env"
    
    # Set environment-specific variables
    local vars_file="$ENVIRONMENTS_DIR/$env/variables.env"
    if [[ -f "$vars_file" ]]; then
        # shellcheck source=/dev/null
        source "$vars_file"
    fi
    
    # Use envsubst if available, otherwise basic substitution
    if command -v envsubst &> /dev/null; then
        envsubst < "$template_file" > "$output_file"
    else
        # Basic template substitution (limited functionality)
        cp "$template_file" "$output_file"
        log_warning "envsubst not available, template substitution may be incomplete"
    fi
    
    log_success "Configuration generated: $output_file"
}

# Deploy configuration to environment
deploy_config() {
    local env="$1"
    local dry_run="${2:-false}"
    
    log_info "Deploying configuration for environment: $env"
    
    # Validate configuration first
    if ! validate_config "$env"; then
        log_error "Configuration validation failed, aborting deployment"
        return 1
    fi
    
    local config_file="$ENVIRONMENTS_DIR/$env/app.yaml"
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN: Would deploy the following configuration:"
        cat "$config_file"
        return 0
    fi
    
    # Deploy based on environment
    case "$env" in
        "production"|"staging")
            # Deploy to Kubernetes
            deploy_to_kubernetes "$env"
            ;;
        "development"|"testing")
            # Copy to local config directory
            cp "$config_file" "$PROJECT_ROOT/claude_tui_config.yaml"
            log_success "Configuration deployed locally"
            ;;
    esac
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    local env="$1"
    local namespace="$env"
    
    # Create ConfigMap
    kubectl create configmap claude-tui-config \
        --from-file="$ENVIRONMENTS_DIR/$env/" \
        --namespace="$namespace" \
        --dry-run=client \
        -o yaml | kubectl apply -f -
    
    # Restart deployment to pick up new config
    kubectl rollout restart deployment/claude-tui-app -n "$namespace"
    
    log_success "Configuration deployed to Kubernetes namespace: $namespace"
}

# Backup configuration
backup_config() {
    local env="$1"
    local backup_dir="$CONFIG_DIR/backups/$env"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$backup_dir/backup_$timestamp.tar.gz"
    
    mkdir -p "$backup_dir"
    
    log_info "Creating backup for environment: $env"
    
    tar -czf "$backup_file" -C "$ENVIRONMENTS_DIR" "$env"
    
    log_success "Backup created: $backup_file"
    
    # Keep only last 10 backups
    ls -t "$backup_dir"/backup_*.tar.gz | tail -n +11 | xargs -r rm
    
    echo "$backup_file"
}

# Restore configuration from backup
restore_config() {
    local env="$1"
    local backup_file="$2"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "Restoring configuration for environment: $env"
    
    # Create backup of current config
    local current_backup
    current_backup=$(backup_config "$env")
    log_info "Current configuration backed up to: $current_backup"
    
    # Restore from backup
    tar -xzf "$backup_file" -C "$ENVIRONMENTS_DIR"
    
    log_success "Configuration restored from: $backup_file"
}

# Compare configurations
diff_configs() {
    local env1="$1"
    local env2="$2"
    
    local config1="$ENVIRONMENTS_DIR/$env1/app.yaml"
    local config2="$ENVIRONMENTS_DIR/$env2/app.yaml"
    
    if [[ ! -f "$config1" ]] || [[ ! -f "$config2" ]]; then
        log_error "Configuration file(s) not found"
        return 1
    fi
    
    log_info "Comparing configurations: $env1 vs $env2"
    
    if command -v diff &> /dev/null; then
        diff -u "$config1" "$config2" || true
    else
        log_warning "diff command not available"
    fi
}

# List available environments
list_environments() {
    log_info "Available environments:"
    
    if [[ -d "$ENVIRONMENTS_DIR" ]]; then
        for env_dir in "$ENVIRONMENTS_DIR"/*; do
            if [[ -d "$env_dir" ]]; then
                local env_name=$(basename "$env_dir")
                local config_file="$env_dir/app.yaml"
                if [[ -f "$config_file" ]]; then
                    echo "  ✓ $env_name"
                else
                    echo "  ✗ $env_name (no config file)"
                fi
            fi
        done
    else
        log_warning "Environments directory not found. Run 'init' first."
    fi
}

# Parse command line arguments
COMMAND=""
ENVIRONMENT=""
DRY_RUN=false
FORCE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        init|validate|deploy|generate|backup|restore|diff|list|template)
            COMMAND="$1"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$ENVIRONMENT" ]]; then
                ENVIRONMENT="$1"
            else
                # Additional arguments for specific commands
                case "$COMMAND" in
                    restore)
                        BACKUP_FILE="$1"
                        ;;
                    diff)
                        ENVIRONMENT2="$1"
                        ;;
                    template)
                        TEMPLATE_NAME="$1"
                        ;;
                esac
            fi
            shift
            ;;
    esac
done

# Validate command
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Execute command
case "$COMMAND" in
    init)
        init_config_structure
        ;;
    validate)
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for validate command"
            exit 1
        fi
        validate_config "$ENVIRONMENT"
        ;;
    deploy)
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for deploy command"
            exit 1
        fi
        deploy_config "$ENVIRONMENT" "$DRY_RUN"
        ;;
    generate)
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for generate command"
            exit 1
        fi
        generate_config "$ENVIRONMENT"
        ;;
    backup)
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for backup command"
            exit 1
        fi
        backup_config "$ENVIRONMENT"
        ;;
    restore)
        if [[ -z "$ENVIRONMENT" ]] || [[ -z "$BACKUP_FILE" ]]; then
            log_error "Environment and backup file required for restore command"
            exit 1
        fi
        restore_config "$ENVIRONMENT" "$BACKUP_FILE"
        ;;
    diff)
        if [[ -z "$ENVIRONMENT" ]] || [[ -z "$ENVIRONMENT2" ]]; then
            log_error "Two environments required for diff command"
            exit 1
        fi
        diff_configs "$ENVIRONMENT" "$ENVIRONMENT2"
        ;;
    list)
        list_environments
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac