#!/bin/bash
# Claude TUI Deployment Script
# Automated deployment for staging and production environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_ENV="${1:-staging}"
VERSION="${2:-latest}"

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
    exit 1
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    case ${DEPLOYMENT_TYPE} in
        "docker"|"docker-compose")
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed"
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is not installed"
            fi
            ;;
        "kubernetes"|"k8s")
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
            fi
            ;;
    esac
    
    log_success "All dependencies are installed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment for ${ENVIRONMENT}..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env file..."
        cat > .env << EOF
# Claude-TIU Environment Configuration
CLAUDE_API_KEY=your-claude-api-key-here
CLAUDE_FLOW_API_KEY=your-claude-flow-api-key-here
POSTGRES_DB=claude_tui
POSTGRES_USER=claude_user
POSTGRES_PASSWORD=claude_secure_pass
DATABASE_URL=postgresql://claude_user:claude_secure_pass@db:5432/claude_tui
REDIS_URL=redis://cache:6379
CLAUDE_TIU_ENV=${ENVIRONMENT}
EOF
        log_warning "Please update the .env file with your actual API keys"
    fi
    
    # Create necessary directories
    mkdir -p data logs backups .swarm/memory
    
    log_success "Environment setup complete"
}

# Docker deployment
deploy_docker() {
    log_info "Building Docker image..."
    docker build -t claude-tui:latest .
    
    log_info "Running Docker container..."
    docker run -d \
        --name claude-tui-${ENVIRONMENT} \
        --env-file .env \
        -v "${PWD}/data:/app/data" \
        -v "${PWD}/logs:/app/logs" \
        -v "${PWD}/projects:/app/projects" \
        -p 8080:8080 \
        claude-tui:latest
    
    log_success "Docker deployment complete"
}

# Docker Compose deployment
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    case ${ENVIRONMENT} in
        "production")
            if [ -f docker-compose.prod.yml ]; then
                docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
            else
                log_warning "Production compose file not found, using default"
                docker-compose up -d --build
            fi
            ;;
        "development")
            docker-compose --profile dev up -d --build
            ;;
        *)
            docker-compose up -d --build
            ;;
    esac
    
    log_success "Docker Compose deployment complete"
    
    # Show status
    log_info "Container status:"
    docker-compose ps
}

# Kubernetes deployment
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Create ConfigMap and Secrets
    kubectl apply -f k8s/configmap.yaml -n ${NAMESPACE}
    
    if [ -f k8s/secrets-${ENVIRONMENT}.yaml ]; then
        kubectl apply -f k8s/secrets-${ENVIRONMENT}.yaml -n ${NAMESPACE}
    else
        log_warning "Using template secrets - update k8s/secrets.yaml with real values"
        kubectl apply -f k8s/secrets.yaml -n ${NAMESPACE}
    fi
    
    # Create storage
    kubectl apply -f k8s/storage.yaml -n ${NAMESPACE}
    
    # Deploy application
    kubectl apply -f k8s/deployment.yaml -n ${NAMESPACE}
    kubectl apply -f k8s/service.yaml -n ${NAMESPACE}
    kubectl apply -f k8s/hpa.yaml -n ${NAMESPACE}
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/claude-tui -n ${NAMESPACE}
    
    log_success "Kubernetes deployment complete"
    
    # Show status
    log_info "Deployment status:"
    kubectl get all -n ${NAMESPACE}
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    case ${DEPLOYMENT_TYPE} in
        "docker")
            if docker ps | grep -q "claude-tui-${ENVIRONMENT}"; then
                log_success "Container is running"
            else
                log_error "Container is not running"
            fi
            ;;
        "docker-compose")
            if docker-compose ps | grep -q "Up"; then
                log_success "All services are running"
            else
                log_error "Some services are not running"
            fi
            ;;
        "kubernetes"|"k8s")
            if kubectl get pods -n ${NAMESPACE} | grep -q "Running"; then
                log_success "Pods are running"
            else
                log_error "Pods are not running"
            fi
            ;;
    esac
}

# Show logs
show_logs() {
    log_info "Showing application logs..."
    
    case ${DEPLOYMENT_TYPE} in
        "docker")
            docker logs -f claude-tui-${ENVIRONMENT}
            ;;
        "docker-compose")
            docker-compose logs -f claude-tui
            ;;
        "kubernetes"|"k8s")
            kubectl logs -f deployment/claude-tui -n ${NAMESPACE}
            ;;
    esac
}

# Cleanup deployment
cleanup() {
    log_info "Cleaning up deployment..."
    
    case ${DEPLOYMENT_TYPE} in
        "docker")
            docker stop claude-tui-${ENVIRONMENT} || true
            docker rm claude-tui-${ENVIRONMENT} || true
            ;;
        "docker-compose")
            docker-compose down -v
            ;;
        "kubernetes"|"k8s")
            kubectl delete namespace ${NAMESPACE} || true
            ;;
    esac
    
    log_success "Cleanup complete"
}

# Show usage
show_usage() {
    echo "Usage: $0 [DEPLOYMENT_TYPE] [ENVIRONMENT] [NAMESPACE]"
    echo ""
    echo "DEPLOYMENT_TYPE:"
    echo "  docker          - Single Docker container"
    echo "  docker-compose  - Docker Compose (default)"
    echo "  kubernetes|k8s  - Kubernetes deployment"
    echo ""
    echo "ENVIRONMENT:"
    echo "  development (default)"
    echo "  staging"
    echo "  production"
    echo ""
    echo "Commands:"
    echo "  deploy          - Deploy the application"
    echo "  health          - Check deployment health"
    echo "  logs           - Show application logs"
    echo "  cleanup        - Remove deployment"
    echo ""
    echo "Examples:"
    echo "  $0 docker-compose development"
    echo "  $0 kubernetes production claude-tui"
    echo "  $0 logs"
    exit 0
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            check_dependencies
            setup_environment
            
            case ${DEPLOYMENT_TYPE} in
                "docker")
                    deploy_docker
                    ;;
                "docker-compose")
                    deploy_docker_compose
                    ;;
                "kubernetes"|"k8s")
                    deploy_kubernetes
                    ;;
                *)
                    log_error "Unknown deployment type: ${DEPLOYMENT_TYPE}"
                    ;;
            esac
            
            health_check
            ;;
        "health")
            health_check
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            ;;
    esac
}

# Handle special case for command as first argument
if [[ "${1:-}" =~ ^(health|logs|cleanup|help|-h|--help)$ ]]; then
    main "$1"
else
    main "deploy"
fi