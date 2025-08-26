# Claude TUI Production Deployment Infrastructure

This directory contains the complete production deployment infrastructure for Claude TUI, implementing modern DevOps best practices with zero-downtime deployments, comprehensive monitoring, and automated rollback capabilities.

## 🚀 Quick Start

```bash
# 1. Set up infrastructure
kubectl apply -f ../k8s/namespace.yaml
kubectl apply -f ../k8s/secrets.yaml
kubectl apply -f ../k8s/configmap.yaml

# 2. Deploy using blue-green strategy
./scripts/blue-green-deploy.sh ghcr.io/your-org/claude-tui:latest

# 3. Verify deployment
./scripts/health-checks.sh production

# 4. Run smoke tests
./scripts/smoke-tests.sh production
```

## 📁 Directory Structure

```
deployment/
├── scripts/                    # Deployment automation scripts
│   ├── blue-green-deploy.sh   # Zero-downtime blue-green deployment
│   ├── health-checks.sh       # Comprehensive health validation
│   ├── rollback.sh            # Automated rollback procedures
│   └── smoke-tests.sh         # Post-deployment smoke tests
├── monitoring/                 # Monitoring and alerting configuration
│   ├── prometheus-alerts.yaml # Production alert rules
│   └── grafana-dashboard.json # Operational dashboard
└── README.md                  # This file
```

## 🔧 Core Features

### 🔵🟢 Blue-Green Deployment
- **Zero-downtime deployments** with traffic switching
- **Automated health checks** and validation
- **Instant rollback** capabilities
- **Performance validation** before traffic switch

### 🔍 Comprehensive Monitoring
- **Prometheus metrics** collection
- **Grafana dashboards** for visualization  
- **Multi-level alerting** (critical, warning, info)
- **SLA monitoring** and violation detection

### 🛡️ Security & Compliance
- **Container security** hardening
- **Vulnerability scanning** with Trivy/Snyk
- **RBAC implementation** with minimal privileges
- **Network policies** for pod isolation

### 📊 Health & Validation
- **Multi-tier health checks** (startup, liveness, readiness)
- **Database connectivity** validation
- **External API** status monitoring
- **Performance regression** detection

## 🎯 CI/CD Pipeline Features

### Pipeline Stages
1. **Code Quality**: Formatting, linting, type checking
2. **Security Scanning**: Dependency vulnerabilities, code analysis
3. **Testing**: Unit, integration, performance tests
4. **Building**: Multi-architecture Docker images
5. **Staging**: Automated staging deployment
6. **Production**: Tag-based production deployment
7. **Monitoring**: Performance validation and alerting

### GitHub Actions Workflow
- **Parallel execution** for faster builds
- **Matrix testing** across Python versions
- **Security scanning** integration
- **Automated rollback** on failures
- **Slack notifications** for deployment status

## 📋 Deployment Scripts

### `blue-green-deploy.sh`
Implements zero-downtime blue-green deployment strategy:

```bash
# Usage
./scripts/blue-green-deploy.sh <new-image>

# Features
- Automatic active/inactive detection
- Health validation before traffic switch
- Performance validation
- Automatic rollback on failure
- Slack notifications
```

### `health-checks.sh` 
Comprehensive health validation:

```bash
# Usage
./scripts/health-checks.sh <environment>

# Validates
- Kubernetes resources status
- Application health endpoints
- Database connectivity
- External API dependencies
- Resource usage patterns
```

### `rollback.sh`
Automated rollback procedures:

```bash
# Emergency rollback (fastest)
./scripts/rollback.sh production emergency

# Rollback to specific revision
./scripts/rollback.sh production revision 5

# Show rollback options
./scripts/rollback.sh production show
```

### `smoke-tests.sh`
Post-deployment validation:

```bash
# Usage
./scripts/smoke-tests.sh <environment>

# Tests
- Basic connectivity
- API endpoints
- Authentication
- Performance thresholds
- External dependencies
```

## 📈 Monitoring Configuration

### Prometheus Alerts
Configured in `monitoring/prometheus-alerts.yaml`:

- **Critical**: Service down, high error rates, SLA violations
- **Warning**: High latency, resource usage, pod issues
- **Info**: Long-running pods, low user activity

### Grafana Dashboard
Pre-configured dashboard (`monitoring/grafana-dashboard.json`):

- System overview and status
- Request metrics and error rates
- Resource usage monitoring
- Database and external API status

## 🔐 Security Configuration

### Container Security
- Non-root user execution (UID 1000)
- Read-only root filesystem
- Dropped capabilities
- Security contexts

### Network Security  
- Network policies for pod isolation
- TLS encryption for external communications
- Kubernetes secrets management
- RBAC with minimal permissions

### Image Security
- Multi-stage builds for minimal images
- Alpine Linux base for reduced attack surface
- Automated vulnerability scanning
- Regular dependency updates

## 🚨 Troubleshooting

### Common Issues

1. **Deployment Failures**
   ```bash
   kubectl get pods -n claude-tui
   kubectl logs -f deployment/claude-tui-blue -n claude-tui
   ```

2. **Database Connectivity**
   ```bash
   kubectl exec -it <pod> -n claude-tui -- psql $DATABASE_URL -c "SELECT 1"
   ```

3. **Resource Issues**
   ```bash
   kubectl top pods -n claude-tui --sort-by=memory
   ```

### Debug Commands
```bash
# Pod debugging
kubectl exec -it <pod-name> -n claude-tui -- /bin/bash

# Network debugging
kubectl exec -it <pod-name> -n claude-tui -- curl http://claude-tui/health

# Configuration debugging
kubectl get configmap claude-config -n claude-tui -o yaml
```

## 📊 Performance Tuning

### Resource Optimization
- **Memory**: 768Mi requests, 1.5Gi limits
- **CPU**: 500m requests, 2000m limits
- **Storage**: SSD-backed persistent volumes

### Application Tuning
- Database connection pooling
- Redis caching optimization
- Python memory management
- Graceful shutdown handling

## 🎛️ Environment Variables

### Production Configuration
```yaml
CLAUDE_TUI_ENV: production
LOG_LEVEL: INFO
MAX_WORKERS: 4
GRACEFUL_TIMEOUT: 30
METRICS_ENABLED: true
```

### Performance Settings
```yaml
WORKER_CONNECTIONS: 1000
MAX_REQUESTS: 10000
PROMETHEUS_MULTIPROC_DIR: /tmp/prometheus
```

## 📞 Support

### Contact Information
- **Platform Team**: platform-team@company.com
- **Documentation**: `/docs/PRODUCTION_DEPLOYMENT_GUIDE.md`
- **Runbooks**: Internal wiki

### Escalation Procedures
1. **L1**: Platform engineer investigation
2. **L2**: Senior platform engineer + Dev team  
3. **L3**: Engineering management

## 🔄 Maintenance

### Regular Tasks
- **Weekly**: Review alert noise and tune thresholds
- **Monthly**: Security scan reviews and updates
- **Quarterly**: Performance baseline updates
- **Annually**: Disaster recovery testing

### Automated Maintenance
- Dependency updates via Renovate/Dependabot
- Security scanning in CI/CD pipeline
- Performance regression testing
- Automated backup procedures

---

For detailed deployment instructions, see [`/docs/PRODUCTION_DEPLOYMENT_GUIDE.md`](../docs/PRODUCTION_DEPLOYMENT_GUIDE.md)