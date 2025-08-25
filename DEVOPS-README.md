# DevOps Infrastructure - Claude-TIU

This document provides comprehensive guidance for deploying, managing, and maintaining the Claude-TIU project infrastructure.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for Claude Flow)
- Python 3.11+
- Git

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository>
cd claude-tui

# Install dependencies
make install-dev

# Setup environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Development Deployment
```bash
# Using Docker Compose (recommended)
make docker-dev

# Or using the deployment script
./scripts/devops/deploy.sh docker-compose development

# View logs
make logs
```

### 3. Production Deployment
```bash
# Docker Compose
./scripts/devops/deploy.sh docker-compose production

# Kubernetes
./scripts/devops/deploy.sh kubernetes production
```

## 📁 Infrastructure Components

### Docker Configuration
- **Dockerfile**: Multi-stage build with security hardening
- **docker-compose.yml**: Development and production services
- **Services**: App, PostgreSQL, Redis, monitoring tools

### Kubernetes Manifests (`k8s/`)
- **namespace.yaml**: Namespace and resource quotas
- **deployment.yaml**: Application deployment with security policies
- **service.yaml**: Load balancer and ingress configuration
- **configmap.yaml**: Application configuration
- **secrets.yaml**: Secure credential management
- **storage.yaml**: Persistent volume configuration
- **hpa.yaml**: Auto-scaling configuration

### CI/CD Pipelines (`.github/workflows/`)
- **ci-cd.yml**: Main build, test, and deployment pipeline
- **security.yml**: Security scanning and vulnerability assessment

## 🛠️ Available Commands

### Makefile Commands
```bash
# Development
make setup           # Complete development setup
make dev            # Start development environment
make test           # Run tests
make test-coverage  # Run tests with coverage

# Code Quality
make lint           # Run linters
make format         # Format code
make security       # Security scans

# Docker Operations
make docker-build   # Build Docker image
make docker-dev     # Run development environment
make docker-prod    # Run production environment

# Database Operations
make db-init        # Initialize database
make db-backup      # Backup database
make db-restore     # Restore database

# Utilities
make clean          # Clean build artifacts
make clean-all      # Clean everything including Docker
make check-env      # Check environment setup
```

### Deployment Script
```bash
# Basic deployment
./scripts/devops/deploy.sh [TYPE] [ENV] [NAMESPACE]

# Examples
./scripts/devops/deploy.sh docker-compose development
./scripts/devops/deploy.sh kubernetes production
./scripts/devops/deploy.sh docker staging

# Management
./scripts/devops/deploy.sh health    # Health check
./scripts/devops/deploy.sh logs      # View logs
./scripts/devops/deploy.sh cleanup   # Clean deployment
```

## 🔒 Security Features

### Multi-layered Security
1. **Container Security**
   - Non-root user execution
   - Read-only root filesystem
   - Security contexts and policies
   - Resource limitations

2. **Network Security**
   - Network policies for pod communication
   - TLS termination at ingress
   - Rate limiting and DDoS protection

3. **Secrets Management**
   - Kubernetes secrets for credentials
   - External secret management integration ready
   - API key rotation procedures

4. **Code Security**
   - Pre-commit hooks for security scanning
   - Automated security testing in CI/CD
   - Dependency vulnerability scanning

### Security Scanning
```bash
# Run comprehensive security audit
python scripts/devops/security_audit.py

# Individual security checks
make security
bandit -r src/
safety check
```

## 📊 Monitoring & Observability

### Health Checks
- Container health checks
- Kubernetes liveness/readiness probes
- Database health monitoring
- API endpoint monitoring

### Metrics Collection
- Prometheus metrics scraping
- Application performance metrics
- System resource monitoring
- Custom business metrics

### Logging
- Structured JSON logging
- Centralized log collection
- Log retention policies
- Error tracking and alerting

## 🏗️ Deployment Environments

### Development
- Local Docker Compose setup
- Hot reload for development
- Debug tools and utilities
- Test database with sample data

### Staging
- Production-like environment
- Integration testing
- Security validation
- Performance testing

### Production
- High availability configuration
- Auto-scaling enabled
- Security hardened
- Monitoring and alerting
- Backup and disaster recovery

## 📋 Operations Procedures

### Daily Operations
```bash
# Check system health
kubectl get pods -n claude-tui
docker-compose ps

# View logs
make logs
kubectl logs -f deployment/claude-tui -n claude-tui

# Monitor resources
kubectl top pods -n claude-tui
docker stats
```

### Backup Procedures
```bash
# Database backup
make db-backup

# Full system backup
kubectl exec -n claude-tui deployment/claude-tui -- /app/scripts/backup.sh

# Verify backup integrity
./scripts/devops/verify-backup.sh
```

### Update Procedures
```bash
# Rolling update (Kubernetes)
kubectl set image deployment/claude-tui claude-tui=claude-tui:v1.2.0 -n claude-tui

# Blue-green deployment
./scripts/devops/blue-green-deploy.sh

# Rollback if needed
kubectl rollout undo deployment/claude-tui -n claude-tui
```

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment claude-tui --replicas=5 -n claude-tui

# Auto-scaling configuration
kubectl autoscale deployment claude-tui --cpu-percent=70 --min=2 --max=10 -n claude-tui

# Check scaling status
kubectl get hpa -n claude-tui
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker logs claude-tui-container
kubectl logs -f pod/claude-tui-xxx -n claude-tui

# Check configuration
kubectl describe pod claude-tui-xxx -n claude-tui
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/claude-tui -n claude-tui -- python -c "import psycopg2; print('DB OK')"

# Check database status
kubectl exec -it claude-tui-db -n claude-tui -- pg_isready
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n claude-tui
docker stats

# Review metrics
kubectl port-forward service/claude-tui-service 9090:9090 -n claude-tui
# Access http://localhost:9090/metrics
```

### Emergency Procedures

#### Security Incident Response
1. Isolate affected components
2. Collect logs and evidence
3. Apply security patches
4. Rotate compromised credentials
5. Document and report

#### System Recovery
1. Assess damage and data loss
2. Restore from most recent backup
3. Validate data integrity
4. Gradually restore services
5. Monitor for issues

## 🔧 Configuration Management

### Environment Variables
```bash
# Required variables
CLAUDE_API_KEY          # Claude AI API key
CLAUDE_FLOW_API_KEY     # Claude Flow API key
DATABASE_URL            # PostgreSQL connection string
REDIS_URL              # Redis connection string

# Optional variables
CLAUDE_TIU_ENV         # Environment (development/staging/production)
LOG_LEVEL              # Logging level (DEBUG/INFO/WARN/ERROR)
MAX_WORKERS            # Number of worker processes
RATE_LIMIT             # API rate limiting configuration
```

### Configuration Files
- `config/app.yaml`: Application configuration
- `config/logging.yaml`: Logging configuration
- `config/prometheus.yml`: Monitoring configuration
- `.env`: Environment-specific variables

## 📚 Additional Resources

### Documentation
- [Security Architecture](docs/security.md)
- [Deployment Guide](docs/deployment.md)
- [API Specification](docs/api-specification.md)
- [Developer Guide](docs/developer-guide.md)

### External Tools
- [Claude Flow Documentation](https://github.com/ruvnet/claude-flow)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Support Contacts
- Development Team: dev@claude-tui.local
- DevOps Team: devops@claude-tui.local
- Security Team: security@claude-tui.local

## 📈 Performance Benchmarks

### Expected Performance
- **API Response Time**: < 200ms (95th percentile)
- **Database Query Time**: < 50ms (average)
- **Container Startup Time**: < 30s
- **Memory Usage**: < 2GB per container
- **CPU Usage**: < 70% under normal load

### Scaling Thresholds
- **CPU**: Scale up at 70% utilization
- **Memory**: Scale up at 80% utilization
- **Response Time**: Scale up if > 500ms for 2 minutes
- **Error Rate**: Alert if > 1% for 1 minute

---

**Last Updated**: August 2025
**Version**: 1.0.0
**Maintained by**: DevOps Team