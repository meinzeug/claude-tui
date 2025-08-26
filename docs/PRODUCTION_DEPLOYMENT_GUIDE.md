# Claude TUI Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Claude TUI to production using modern DevOps practices including blue-green deployments, automated CI/CD pipelines, and comprehensive monitoring.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Blue-Green Deployment](#blue-green-deployment)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Security Configuration](#security-configuration)
7. [Rollback Procedures](#rollback-procedures)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
- Kubernetes cluster (v1.24+)
- Docker (v20.10+)
- kubectl (v1.24+)
- Helm (v3.8+)
- GitHub CLI (for CI/CD setup)

### Infrastructure Requirements
- **Minimum**: 3 nodes, 8GB RAM each, 4 CPU cores
- **Recommended**: 5 nodes, 16GB RAM each, 8 CPU cores
- **Storage**: SSD-backed persistent volumes
- **Network**: Load balancer support

### External Dependencies
- PostgreSQL database (v13+)
- Redis cache (v6+)
- Claude API access
- Claude Flow API access (optional)

## Infrastructure Setup

### 1. Kubernetes Namespace Creation

```bash
kubectl create namespace claude-tui
kubectl label namespace claude-tui name=claude-tui
```

### 2. Secrets Configuration

Create the secrets file and apply:

```bash
# Create secrets
kubectl create secret generic claude-tui-secrets \
  --from-literal=claude-api-key="your-claude-api-key" \
  --from-literal=claude-flow-api-key="your-claude-flow-api-key" \
  --from-literal=database-url="postgresql://user:pass@host:5432/dbname" \
  --from-literal=redis-url="redis://host:6379/0" \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=encryption-key="your-encryption-key" \
  -n claude-tui
```

### 3. ConfigMap Setup

```bash
kubectl apply -f k8s/configmap.yaml -n claude-tui
```

### 4. Storage Configuration

```bash
kubectl apply -f k8s/storage.yaml -n claude-tui
```

### 5. RBAC Setup

```bash
kubectl apply -f k8s/rbac.yaml -n claude-tui
```

## CI/CD Pipeline

### GitHub Actions Setup

1. **Repository Secrets Configuration**

Add the following secrets to your GitHub repository:

```
CLAUDE_API_KEY=your-claude-api-key
CLAUDE_FLOW_API_KEY=your-claude-flow-api-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
SLACK_WEBHOOK_URL=your-slack-webhook-url
SNYK_TOKEN=your-snyk-token
GRAFANA_PASSWORD=your-grafana-password
```

2. **Workflow Triggers**

The CI/CD pipeline triggers on:
- Push to `main` branch (staging deployment)
- Tag creation with `v*` pattern (production deployment)
- Pull requests (testing only)
- Manual workflow dispatch

3. **Pipeline Stages**

```yaml
# Stages executed in parallel where possible:
1. Code Quality & Security Checks
2. Multi-version Testing (Python 3.10, 3.11, 3.12)
3. Docker Image Build & Security Scan
4. Staging Deployment
5. Production Deployment (tag-based)
6. Performance Monitoring
```

### Pipeline Features

- **Parallel Execution**: Multiple test types run concurrently
- **Security Scanning**: Trivy, Snyk, Bandit, Safety
- **Multi-architecture Builds**: AMD64 and ARM64
- **Comprehensive Testing**: Unit, integration, performance
- **Automated Rollback**: On deployment failures
- **Notifications**: Slack integration

## Blue-Green Deployment

### Architecture

The blue-green deployment strategy provides zero-downtime deployments by maintaining two identical production environments:

- **Blue Environment**: Currently active production
- **Green Environment**: New version for deployment
- **Main Service**: Routes traffic between blue/green

### Deployment Process

1. **Automated Deployment**:
   ```bash
   ./deployment/scripts/blue-green-deploy.sh <new-image-tag>
   ```

2. **Manual Steps**:
   ```bash
   # Deploy to inactive environment
   kubectl apply -f k8s/production/deployment-green.yaml
   
   # Update image
   kubectl set image deployment/claude-tui-green \
     claude-tui=ghcr.io/your-org/claude-tui:v1.2.3 \
     -n claude-tui
   
   # Switch traffic
   kubectl patch service claude-tui -n claude-tui \
     -p '{"spec":{"selector":{"version":"green"}}}'
   ```

### Traffic Switching

The main service selector is updated to switch traffic:

```yaml
spec:
  selector:
    app: claude-tui
    version: blue  # Changed to 'green' during deployment
```

## Monitoring and Alerting

### Prometheus Metrics

The application exposes metrics on port 9090:

```
# Application metrics
claude_tui_requests_total
claude_tui_request_duration_seconds
claude_tui_database_connections_active
claude_tui_external_api_status

# System metrics
container_memory_usage_bytes
container_cpu_usage_seconds_total
kube_pod_status_ready
```

### Grafana Dashboards

Import the dashboard from `deployment/monitoring/grafana-dashboard.json`:

1. **System Overview**: Service status, pod health
2. **Performance**: Request rate, response times, error rates
3. **Resources**: CPU, memory, storage usage
4. **Dependencies**: Database, Redis, external APIs

### Alert Rules

Critical alerts configured in `deployment/monitoring/prometheus-alerts.yaml`:

- **Service Down**: Immediate notification
- **High Error Rate**: >5% for 5 minutes
- **High Latency**: >2s 95th percentile
- **Resource Usage**: >85% memory, >80% CPU
- **Pod Issues**: Crash loops, OOM kills
- **SLA Violations**: <99% availability

### Setting Up Monitoring

```bash
# Deploy monitoring stack
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f deployment/monitoring/prometheus-alerts.yaml

# Configure Grafana
kubectl apply -f monitoring/grafana-dashboard.json
```

## Security Configuration

### Container Security

- **Non-root user**: UID 1000
- **Read-only filesystem**: Prevents runtime modifications
- **Security contexts**: Dropped capabilities
- **Resource limits**: Prevent resource exhaustion

### Network Security

- **Network policies**: Restrict pod-to-pod communication
- **TLS encryption**: All external communications
- **Secret management**: Kubernetes secrets with rotation

### Image Security

- **Multi-stage builds**: Minimal production images
- **Vulnerability scanning**: Trivy and Snyk integration
- **Base image**: Alpine Linux for minimal attack surface
- **Regular updates**: Automated dependency updates

### Access Control

- **RBAC**: Role-based access control
- **Service accounts**: Minimal required permissions
- **Pod security standards**: Restricted security profile

## Rollback Procedures

### Automated Rollback

Rollbacks are triggered automatically on:
- Health check failures
- Performance validation failures
- High error rates post-deployment

### Manual Rollback Options

1. **Quick Rollback** (Blue-Green):
   ```bash
   ./deployment/scripts/rollback.sh production emergency
   ```

2. **Revision Rollback**:
   ```bash
   ./deployment/scripts/rollback.sh production revision 5
   ```

3. **Kubernetes Native**:
   ```bash
   kubectl rollout undo deployment/claude-tui-blue -n claude-tui
   ```

### Rollback Validation

Post-rollback checks include:
- Health endpoint verification
- Performance validation
- Error rate monitoring
- Smoke test execution

## Troubleshooting

### Common Issues

#### 1. Deployment Failures

**Symptoms**: Pods not starting, ImagePullBackOff
**Solutions**:
```bash
# Check pod status
kubectl get pods -n claude-tui
kubectl describe pod <pod-name> -n claude-tui

# Check deployment status
kubectl rollout status deployment/claude-tui-blue -n claude-tui

# Check logs
kubectl logs -f deployment/claude-tui-blue -n claude-tui
```

#### 2. Database Connectivity Issues

**Symptoms**: Connection timeouts, authentication failures
**Solutions**:
```bash
# Test database connectivity
kubectl exec -it <pod-name> -n claude-tui -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check database secrets
kubectl get secret claude-tui-secrets -n claude-tui -o yaml

# Verify network policies
kubectl get networkpolicies -n claude-tui
```

#### 3. High Memory Usage

**Symptoms**: OOMKilled pods, performance degradation
**Solutions**:
```bash
# Check resource usage
kubectl top pods -n claude-tui

# Increase memory limits
kubectl patch deployment claude-tui-blue -n claude-tui -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"claude-tui","resources":{"limits":{"memory":"2Gi"}}}]}}}}'

# Enable memory profiling
kubectl set env deployment/claude-tui-blue -n claude-tui \
  MEMORY_PROFILING=true
```

#### 4. Load Balancer Issues

**Symptoms**: External access failures, timeout errors
**Solutions**:
```bash
# Check service status
kubectl get svc claude-tui -n claude-tui
kubectl describe svc claude-tui -n claude-tui

# Check ingress configuration
kubectl get ingress -n claude-tui
kubectl describe ingress claude-tui -n claude-tui

# Test internal connectivity
kubectl exec -it <pod-name> -n claude-tui -- curl http://claude-tui:80/health
```

### Debug Commands

```bash
# Pod debugging
kubectl exec -it <pod-name> -n claude-tui -- /bin/bash
kubectl logs -f <pod-name> -n claude-tui --previous

# Resource monitoring
kubectl top nodes
kubectl top pods -n claude-tui --sort-by=memory

# Network debugging
kubectl exec -it <pod-name> -n claude-tui -- nslookup claude-tui
kubectl exec -it <pod-name> -n claude-tui -- netstat -tlnp

# Configuration debugging
kubectl get configmap claude-config -n claude-tui -o yaml
kubectl get secret claude-tui-secrets -n claude-tui -o yaml
```

### Health Check Endpoints

- **Health**: `GET /health` - Basic health status
- **Readiness**: `GET /ready` - Ready to accept traffic
- **Startup**: `GET /startup` - Application started
- **Metrics**: `GET /metrics` - Prometheus metrics

### Performance Tuning

#### Database Optimization

```yaml
# Connection pooling configuration
DATABASE_POOL_SIZE: 20
DATABASE_MAX_OVERFLOW: 30
DATABASE_POOL_TIMEOUT: 30
DATABASE_POOL_RECYCLE: 3600
```

#### Memory Optimization

```yaml
# Python memory settings
PYTHONMALLOC: malloc
MALLOC_ARENA_MAX: 2
MALLOC_MMAP_THRESHOLD_: 131072
```

#### Container Resources

```yaml
resources:
  requests:
    memory: "768Mi"
    cpu: "500m"
  limits:
    memory: "1.5Gi"
    cpu: "2000m"
```

## Best Practices

### Deployment Best Practices

1. **Always test in staging first**
2. **Use blue-green deployments for zero downtime**
3. **Implement comprehensive health checks**
4. **Monitor deployment metrics**
5. **Have rollback procedures ready**

### Security Best Practices

1. **Regular security scans**
2. **Principle of least privilege**
3. **Secret rotation**
4. **Network segmentation**
5. **Audit logging**

### Monitoring Best Practices

1. **Monitor business metrics**
2. **Set up proactive alerts**
3. **Use SLI/SLO approach**
4. **Regular alert tuning**
5. **Incident response procedures**

## Support and Escalation

### Contact Information

- **Platform Team**: platform-team@company.com
- **On-call Rotation**: Use PagerDuty escalation
- **Emergency Hotline**: +1-xxx-xxx-xxxx

### Escalation Procedures

1. **Level 1**: Platform engineer investigation
2. **Level 2**: Senior platform engineer + Dev team
3. **Level 3**: Engineering management involvement

### Incident Response

1. **Immediate**: Assess impact and containment
2. **Short-term**: Implement workaround/rollback
3. **Long-term**: Root cause analysis and prevention

---

For additional support or questions, please refer to the team documentation or contact the platform team.