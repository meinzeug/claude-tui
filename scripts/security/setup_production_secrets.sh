#!/bin/bash

# Production Secret Setup Script for Claude-TUI
# This script helps generate and configure secure secrets for production deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SECRETS_DIR="/etc/claude-tui/secrets"
BACKUP_DIR="/var/backups/claude-tui/secrets"
NAMESPACE="claude-tui-prod"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root for production setup"
        exit 1
    fi
}

# Generate a secure random string
generate_secret() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d '=' | head -c $length
}

# Generate JWT secret key
generate_jwt_secret() {
    print_status "Generating JWT secret key..."
    openssl rand -base64 64 | tr -d '\n'
}

# Generate application secret key
generate_app_secret() {
    print_status "Generating application secret key..."
    openssl rand -base64 64 | tr -d '\n'
}

# Generate encryption key
generate_encryption_key() {
    print_status "Generating encryption key..."
    openssl rand -base64 32 | tr -d '\n'
}

# Generate CSRF secret
generate_csrf_secret() {
    print_status "Generating CSRF secret key..."
    openssl rand -base64 32 | tr -d '\n'
}

# Generate database password
generate_db_password() {
    print_status "Generating database password..."
    openssl rand -base64 24 | tr -d '=' | head -c 24
}

# Generate Redis password
generate_redis_password() {
    print_status "Generating Redis password..."
    openssl rand -base64 24 | tr -d '=' | head -c 24
}

# Create secrets directory
setup_secrets_directory() {
    print_status "Setting up secrets directory..."
    
    # Create directories
    mkdir -p "$SECRETS_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Set secure permissions
    chmod 700 "$SECRETS_DIR"
    chmod 700 "$BACKUP_DIR"
    
    print_status "Secrets directory created at $SECRETS_DIR"
}

# Generate all secrets
generate_all_secrets() {
    print_header "GENERATING PRODUCTION SECRETS"
    
    local secrets_file="$SECRETS_DIR/production.env"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    
    # Backup existing secrets if they exist
    if [[ -f "$secrets_file" ]]; then
        print_warning "Backing up existing secrets..."
        cp "$secrets_file" "$BACKUP_DIR/production.env.$timestamp.backup"
    fi
    
    # Generate new secrets
    cat > "$secrets_file" << EOF
# Production Secrets for Claude-TUI
# Generated on $(date)
# DO NOT COMMIT TO VERSION CONTROL

# JWT Authentication
JWT_SECRET_KEY=$(generate_jwt_secret)
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Application Secrets
SECRET_KEY=$(generate_app_secret)
ENCRYPTION_KEY=$(generate_encryption_key)
CSRF_SECRET_KEY=$(generate_csrf_secret)

# Database Configuration
DATABASE_URL=postgresql://claude_tui:$(generate_db_password)@postgres:5432/claude_tui_prod
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=$(generate_redis_password)
REDIS_SESSION_DB=1
REDIS_AUDIT_DB=2
REDIS_RATELIMIT_DB=3
REDIS_BLACKLIST_DB=4

# Security Settings
ENVIRONMENT=production
DEBUG=false
TWO_FACTOR_ENABLED=true

# Rate Limiting
LOGIN_RATE_LIMIT=5
API_RATE_LIMIT_AUTHENTICATED=1000
API_RATE_LIMIT_ANONYMOUS=100

# Audit Logging
AUDIT_LOG_LEVEL=INFO
AUDIT_RETENTION_DAYS=90

# Generated secrets timestamp
SECRETS_GENERATED=$(date '+%Y-%m-%d %H:%M:%S')
EOF

    # Set secure permissions
    chmod 600 "$secrets_file"
    
    print_status "Secrets generated and saved to $secrets_file"
}

# Create Kubernetes secrets
create_k8s_secrets() {
    print_header "CREATING KUBERNETES SECRETS"
    
    local secrets_file="$SECRETS_DIR/production.env"
    
    if [[ ! -f "$secrets_file" ]]; then
        print_error "Secrets file not found. Run generate_all_secrets first."
        return 1
    fi
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl not found. Skipping Kubernetes secret creation."
        return 0
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_status "Creating namespace $NAMESPACE..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Delete existing secret if it exists
    if kubectl get secret claude-tui-secrets -n "$NAMESPACE" &> /dev/null; then
        print_warning "Deleting existing Kubernetes secret..."
        kubectl delete secret claude-tui-secrets -n "$NAMESPACE"
    fi
    
    # Create new secret from env file
    print_status "Creating Kubernetes secret..."
    kubectl create secret generic claude-tui-secrets \
        --from-env-file="$secrets_file" \
        --namespace="$NAMESPACE"
    
    print_status "Kubernetes secret created successfully"
}

# Setup database with generated credentials
setup_database() {
    print_header "SETTING UP DATABASE"
    
    local secrets_file="$SECRETS_DIR/production.env"
    
    if [[ ! -f "$secrets_file" ]]; then
        print_error "Secrets file not found. Run generate_all_secrets first."
        return 1
    fi
    
    # Source the secrets
    source "$secrets_file"
    
    # Extract password from DATABASE_URL
    local db_password=$(echo "$DATABASE_URL" | sed -n 's/.*:\([^@]*\)@.*/\1/p')
    
    print_status "Database setup instructions:"
    echo "1. Create database user:"
    echo "   CREATE USER claude_tui WITH PASSWORD '$db_password';"
    echo "2. Create database:"
    echo "   CREATE DATABASE claude_tui_prod OWNER claude_tui;"
    echo "3. Grant privileges:"
    echo "   GRANT ALL PRIVILEGES ON DATABASE claude_tui_prod TO claude_tui;"
    
    print_warning "Please run these SQL commands on your PostgreSQL server"
}

# Setup Redis with generated credentials
setup_redis() {
    print_header "SETTING UP REDIS"
    
    local secrets_file="$SECRETS_DIR/production.env"
    
    if [[ ! -f "$secrets_file" ]]; then
        print_error "Secrets file not found. Run generate_all_secrets first."
        return 1
    fi
    
    # Source the secrets
    source "$secrets_file"
    
    print_status "Redis configuration:"
    echo "requirepass $REDIS_PASSWORD"
    
    print_warning "Add this to your Redis configuration and restart Redis"
}

# Validate generated secrets
validate_secrets() {
    print_header "VALIDATING SECRETS"
    
    local secrets_file="$SECRETS_DIR/production.env"
    local valid=true
    
    if [[ ! -f "$secrets_file" ]]; then
        print_error "Secrets file not found."
        return 1
    fi
    
    # Source secrets
    source "$secrets_file"
    
    # Check JWT secret
    if [[ ${#JWT_SECRET_KEY} -lt 32 ]]; then
        print_error "JWT_SECRET_KEY is too short (minimum 32 characters)"
        valid=false
    fi
    
    # Check application secret
    if [[ ${#SECRET_KEY} -lt 32 ]]; then
        print_error "SECRET_KEY is too short (minimum 32 characters)"
        valid=false
    fi
    
    # Check encryption key
    if [[ ${#ENCRYPTION_KEY} -lt 32 ]]; then
        print_error "ENCRYPTION_KEY is too short (minimum 32 characters)"
        valid=false
    fi
    
    # Check environment
    if [[ "$ENVIRONMENT" != "production" ]]; then
        print_error "ENVIRONMENT is not set to 'production'"
        valid=false
    fi
    
    # Check debug mode
    if [[ "$DEBUG" != "false" ]]; then
        print_error "DEBUG mode is enabled (should be false in production)"
        valid=false
    fi
    
    if $valid; then
        print_status "All secrets validation passed ✅"
    else
        print_error "Secrets validation failed ❌"
        return 1
    fi
}

# Show security checklist
show_security_checklist() {
    print_header "PRODUCTION SECURITY CHECKLIST"
    
    echo "✅ Secrets Management:"
    echo "   - Secrets generated with cryptographically secure random values"
    echo "   - Secrets stored with restricted permissions (600)"
    echo "   - Kubernetes secrets created from environment file"
    echo ""
    
    echo "🔍 Still Required:"
    echo "   - [ ] Configure firewall rules"
    echo "   - [ ] Setup SSL/TLS certificates"
    echo "   - [ ] Configure reverse proxy (nginx/apache)"
    echo "   - [ ] Setup monitoring and alerting"
    echo "   - [ ] Configure backup strategies"
    echo "   - [ ] Setup log aggregation"
    echo "   - [ ] Configure intrusion detection"
    echo "   - [ ] Setup security scanning"
    echo ""
    
    echo "📋 Database Security:"
    echo "   - [ ] Enable SSL for database connections"
    echo "   - [ ] Configure database firewall rules"
    echo "   - [ ] Setup database backups"
    echo "   - [ ] Enable database audit logging"
    echo ""
    
    echo "🔒 Redis Security:"
    echo "   - [ ] Enable Redis AUTH with generated password"
    echo "   - [ ] Configure Redis SSL/TLS"
    echo "   - [ ] Setup Redis persistence and backups"
    echo "   - [ ] Configure Redis firewall rules"
}

# Main function
main() {
    print_header "CLAUDE-TUI PRODUCTION SECURITY SETUP"
    
    # Check prerequisites
    check_root
    
    # Setup
    setup_secrets_directory
    generate_all_secrets
    
    # Optional Kubernetes setup
    read -p "Do you want to create Kubernetes secrets? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_k8s_secrets
    fi
    
    # Validate
    validate_secrets
    
    # Show setup instructions
    setup_database
    setup_redis
    
    # Show checklist
    show_security_checklist
    
    print_status "Production security setup completed!"
    print_warning "Remember to:"
    print_warning "1. Backup your secrets securely"
    print_warning "2. Rotate secrets regularly"
    print_warning "3. Monitor for security events"
    print_warning "4. Keep dependencies updated"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi