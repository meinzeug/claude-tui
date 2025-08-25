# Claude-TIU Production Deployment Guide

## 🚀 Complete Production Infrastructure

This guide covers the complete production deployment infrastructure for Claude-TIU, optimized for 1000+ concurrent users with 99.9% uptime SLA.

## 📋 Infrastructure Overview

### 🐳 Docker Production Setup
- **Multi-stage builds** with optimized layer caching
- **Security hardened** with non-root user and minimal attack surface
- **Resource optimized** (400MB memory limit, 1000m CPU limit)
- **Health checks** and graceful shutdown procedures

### ☸️ Kubernetes Production Manifests
- **Horizontal Pod Autoscaler**: 3-20 replicas based on CPU/memory
- **Rolling deployments** with zero-downtime strategy
- **Resource limits** aligned with performance optimization results
- **Security contexts** and network policies

### 🔄 Enhanced GitHub Actions CI/CD
- **Security scanning** with Trivy and CodeQL
- **Performance regression testing** with automated benchmarks
- **Blue-green deployments** with automated rollback
- **Multi-environment support** (staging → production)

### 🏗️ Infrastructure as Code
- **Terraform** for AWS EKS cluster provisioning
- **Helm charts** for application deployment
- **Database migration** scripts for PostgreSQL
- **Secrets management** with Kubernetes secrets + external operators

### 📊 Monitoring & Alerting
- **Prometheus rules** for SLA monitoring (<500ms response time)
- **Grafana dashboards** for real-time metrics
- **Automated alerting** for critical issues
- **Performance tracking** and bottleneck analysis

## 🚀 Quick Deploy Commands

### 1. Infrastructure Provisioning
```bash
# Deploy AWS infrastructure with Terraform
cd terraform
terraform init
terraform plan -var="cluster_name=claude-tui-production"
terraform apply

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name claude-tui-production
```

### 2. Application Deployment
```bash
# Deploy with Helm
helm install claude-tui ./helm/claude-tui \
  --namespace claude-tui-production \
  --create-namespace \
  --values helm/claude-tui/values.yaml

# Or deploy with kubectl
kubectl apply -f k8s/
```

### 3. Database Setup
```bash
# Run database migrations
kubectl exec -it deployment/claude-tui-app -n claude-tui-production -- \
  psql $DATABASE_URL -f /app/scripts/migration/database-migration.sql
```

### 4. Monitoring Setup
```bash
# Deploy monitoring stack
kubectl apply -f monitoring/prometheus-rules.yaml
curl -X POST "$GRAFANA_API_URL/api/dashboards/db" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @monitoring/grafana-dashboard.json
```

## 📊 Production SLA Targets

| Metric | Target | Monitoring |
|--------|--------|------------|
| **Uptime** | 99.9% | Prometheus alerts |
| **Response Time** | <500ms (95th percentile) | Continuous monitoring |
| **Error Rate** | <1% | Real-time alerting |
| **Scaling** | 3-20 replicas | HPA automation |
| **Deployment Time** | <2 minutes | CI/CD pipeline |
| **Rollback Time** | <30 seconds | Automated rollback |

## 🔒 Security Features

### Container Security
- Non-root user execution
- Read-only root filesystem
- Dropped ALL capabilities
- Resource limits enforcement

### Network Security
- Network policies for pod isolation
- TLS encryption for all traffic
- Ingress rate limiting (100 req/min)
- RBAC with minimal permissions

### Secrets Management
- Kubernetes secrets for sensitive data
- External Secret Operator support
- Vault integration ready
- Encrypted database connections

## 📈 Performance Optimizations

### Resource Management
- **Memory**: 256Mi request, 400Mi limit
- **CPU**: 500m request, 1000m limit
- **Storage**: 10Gi persistent volumes
- **Ephemeral**: 2Gi temp storage

### Scaling Configuration
- **Min replicas**: 3 (high availability)
- **Max replicas**: 20 (1000+ user capacity)
- **Scale up**: 100% increase every 30s
- **Scale down**: 10% decrease every 5m

### Health Checks
- **Startup probe**: 10s initial, 10s interval
- **Readiness probe**: 30s initial, 10s interval
- **Liveness probe**: 60s initial, 30s interval

## 🚨 Monitoring & Alerting

### Critical Alerts (PagerDuty/Slack)
- Service down for >1 minute
- Error rate >5% for >2 minutes
- Response time >500ms for >3 minutes
- Memory usage >80% for >5 minutes

### Warning Alerts (Slack)
- CPU usage >70% for >5 minutes
- Disk usage >80% for >10 minutes
- HPA at maximum replicas for >10 minutes

### Dashboards
- **Real-time metrics**: Request rate, response time, error rate
- **Resource monitoring**: CPU, memory, network, disk
- **Business metrics**: Active users, task completion rates
- **Infrastructure health**: Pod status, node resources

## 🔄 Deployment Process

### Staging Deployment (Automatic)
1. **Trigger**: Push to `main` branch
2. **Security scan**: Trivy + CodeQL analysis
3. **Build**: Multi-arch Docker image
4. **Deploy**: Staging environment
5. **Smoke tests**: Automated validation
6. **Notification**: Slack deployment status

### Production Deployment (Manual/Tag)
1. **Trigger**: Git tag `v*` or manual dispatch
2. **Pre-deployment**: Database backup
3. **Blue-green deploy**: Zero-downtime strategy
4. **Validation**: Health checks + performance tests
5. **Monitoring**: Real-time metrics tracking
6. **Rollback**: Automated on failure

## 📁 File Structure

```
claude-tui/
├── docker/
│   ├── Dockerfile.production     # Multi-stage production build
│   └── .dockerignore            # Build optimization
├── k8s/
│   ├── namespace.yaml           # Production namespace
│   ├── deployment.yaml          # Application deployment
│   ├── service.yaml             # Service configuration
│   ├── hpa.yaml                 # Horizontal pod autoscaler
│   ├── ingress.yaml             # Ingress with SSL
│   ├── configmap.yaml           # Application configuration
│   ├── secrets.yaml             # Secret templates
│   └── rbac.yaml                # Security & permissions
├── terraform/
│   ├── main.tf                  # EKS cluster infrastructure
│   └── userdata.sh              # Node initialization
├── helm/claude-tui/
│   ├── Chart.yaml               # Helm chart metadata
│   ├── values.yaml              # Configuration values
│   └── templates/               # Kubernetes templates
├── monitoring/
│   ├── prometheus-rules.yaml    # Alert rules
│   └── grafana-dashboard.json   # Monitoring dashboard
├── scripts/
│   ├── production-validation.sh # Deployment validation
│   ├── check_performance_regression.py # Performance testing
│   └── migration/
│       └── database-migration.sql # Database schema
└── .github/workflows/
    └── production-deployment.yml # CI/CD pipeline
```

## 🛠️ Operations Runbook

### Health Check Commands
```bash
# Check deployment status
kubectl get deployments -n claude-tui-production

# Check pod health
kubectl get pods -n claude-tui-production

# Check HPA status  
kubectl get hpa -n claude-tui-production

# View logs
kubectl logs -f deployment/claude-tui-app -n claude-tui-production

# Run validation script
./scripts/production-validation.sh
```

### Troubleshooting
```bash
# Check resource usage
kubectl top pods -n claude-tui-production

# Describe problematic pod
kubectl describe pod <pod-name> -n claude-tui-production

# Check events
kubectl get events -n claude-tui-production --sort-by='.lastTimestamp'

# Port forward for debugging
kubectl port-forward service/claude-tui-service 8080:80 -n claude-tui-production
```

### Emergency Procedures
```bash
# Scale up immediately
kubectl scale deployment claude-tui-app --replicas=10 -n claude-tui-production

# Rollback to previous version
kubectl rollout undo deployment/claude-tui-app -n claude-tui-production

# Emergency restart
kubectl rollout restart deployment/claude-tui-app -n claude-tui-production
```

## 📞 Support & Contacts

- **Production Issues**: PagerDuty escalation
- **Deployment Questions**: DevOps team (#devops-claude-tui)
- **Performance Issues**: Performance team (#perf-claude-tui)
- **Security Concerns**: Security team (#security-alerts)

## 🔗 Related Documentation

- [Performance Optimization Results](./PERFORMANCE_OPTIMIZATION_SUMMARY.md)
- [Security Assessment](./SECURITY_ASSESSMENT.md)
- [API Documentation](./API_DOCUMENTATION.md)
- [Disaster Recovery Plan](./DISASTER_RECOVERY.md)

---

**Status**: ✅ PRODUCTION READY - Launch Window is NOW!

**Next Steps**: Execute deployment pipeline and begin production traffic routing.