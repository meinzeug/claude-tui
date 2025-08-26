# Hive Mind Monitoring and Incident Response Runbook

## Overview

This runbook provides comprehensive procedures for monitoring, troubleshooting, and responding to incidents in the Hive Mind collective system.

## Quick Reference

### Emergency Contacts
- **On-call Engineer**: Slack #incidents-critical
- **PagerDuty**: [PagerDuty Console](https://hivemind.pagerduty.com)
- **Security Team**: security@hive-mind.ai

### System Access
- **Grafana**: http://monitoring.hive-mind.ai:3000
- **Prometheus**: http://monitoring.hive-mind.ai:9090
- **AlertManager**: http://monitoring.hive-mind.ai:9093
- **Jaeger**: http://monitoring.hive-mind.ai:16686

## Service Level Objectives (SLOs)

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| **Availability** | 99.9% | HTTP 2xx/3xx responses | < 99.9% over 5 minutes |
| **Latency** | P99 < 2s | Response time | P99 > 2s for 5 minutes |
| **Throughput** | > 100 RPS | Requests per second | < 100 RPS for 10 minutes |
| **Error Rate** | < 0.1% | 5xx responses | > 0.1% over 5 minutes |

## Monitoring Stack Components

### Prometheus
- **Purpose**: Metrics collection and alerting
- **Port**: 9090
- **Config**: `/deployment/monitoring/prometheus/prometheus.yml`
- **Data Retention**: 30 days

### Grafana
- **Purpose**: Visualization and dashboards
- **Port**: 3000
- **Login**: admin/admin123
- **Key Dashboards**:
  - Hive Mind - System Overview
  - SLO/SLA Monitoring Dashboard

### AlertManager
- **Purpose**: Alert routing and notification
- **Port**: 9093
- **Config**: `/deployment/monitoring/alertmanager/alertmanager.yml`

### Loki
- **Purpose**: Log aggregation
- **Port**: 3100
- **Retention**: 30 days

### Jaeger
- **Purpose**: Distributed tracing
- **Port**: 16686
- **Storage**: Badger (local) / Elasticsearch (production)

## Alert Response Procedures

### Critical Alerts

#### ServiceDown
**Trigger**: Service is not responding (up == 0)
**Severity**: Critical
**Response Time**: Immediate (< 5 minutes)

**Steps:**
1. Check service status in Grafana dashboard
2. Verify if it's a single instance or complete service failure
3. Check recent deployments in deployment history
4. Restart service if single instance failure:
   ```bash
   kubectl rollout restart deployment/<service-name> -n production
   ```
5. If complete service failure, check:
   - Resource constraints (CPU/Memory)
   - Database connectivity
   - External dependencies
6. Rollback recent deployment if necessary:
   ```bash
   kubectl rollout undo deployment/<service-name> -n production
   ```
7. Update incident status in PagerDuty

#### ClaudeTUIHighErrorRate
**Trigger**: Error rate > 10% for 2 minutes
**Severity**: Critical
**Response Time**: < 10 minutes

**Steps:**
1. Check error distribution by endpoint in Grafana
2. Review recent error logs in Loki:
   ```
   {job="claude-tui"} |= "ERROR"
   ```
3. Check if specific endpoint or global issue
4. Scale up service if needed:
   ```bash
   kubectl scale deployment claude-tui --replicas=6 -n production
   ```
5. Enable circuit breaker for failing external dependencies
6. Check database performance and connections
7. Review and rollback recent changes if necessary

#### PostgreSQLDown
**Trigger**: Database is unreachable
**Severity**: Critical
**Response Time**: Immediate

**Steps:**
1. Check database pod status:
   ```bash
   kubectl get pods -l app=postgresql -n production
   ```
2. Check database logs:
   ```bash
   kubectl logs -l app=postgresql -n production --tail=100
   ```
3. Verify persistent volume and storage
4. Check connection limits and active connections
5. Restart database if needed (coordinate with team)
6. Verify backup and recovery procedures
7. Check for disk space issues

### Warning Alerts

#### HighMemoryUsage
**Trigger**: Memory usage > 85% for 5 minutes
**Severity**: Warning
**Response Time**: < 30 minutes

**Steps:**
1. Identify which pods/containers are consuming memory
2. Check for memory leaks in application logs
3. Review memory usage trends over time
4. Scale horizontally if possible:
   ```bash
   kubectl scale deployment <service> --replicas=<new-count>
   ```
5. Enable memory optimization features
6. Plan for vertical scaling if needed
7. Review memory allocation limits

#### HighCPUUsage
**Trigger**: CPU usage > 80% for 5 minutes
**Severity**: Warning
**Response Time**: < 30 minutes

**Steps:**
1. Identify CPU-intensive processes
2. Check for inefficient queries or algorithms
3. Review CPU usage patterns
4. Scale horizontally:
   ```bash
   kubectl scale deployment <service> --replicas=<new-count>
   ```
5. Enable CPU throttling if necessary
6. Review and optimize hot paths in code

#### ClaudeTUIHighLatency
**Trigger**: P95 latency > 2s for 5 minutes
**Severity**: Warning
**Response Time**: < 30 minutes

**Steps:**
1. Check latency by endpoint in Grafana
2. Review database query performance
3. Check external API response times
4. Look for network issues or timeouts
5. Enable caching if appropriate
6. Scale up resources if needed
7. Review recent code changes for performance regressions

## Health Check Procedures

### Manual Health Checks

**System Health Check:**
```bash
curl -s http://localhost:8000/health | jq
```

**Database Connection Check:**
```bash
curl -s http://localhost:8000/health/readiness
```

**Individual Component Checks:**
```bash
# Prometheus
curl -s http://localhost:9090/-/healthy

# Grafana
curl -s http://localhost:3000/api/health

# AlertManager
curl -s http://localhost:9093/-/healthy

# Loki
curl -s http://localhost:3100/ready
```

### Automated Health Monitoring

The system includes comprehensive health checks:

1. **Liveness Probes**: Detect if service is running
2. **Readiness Probes**: Detect if service can handle traffic
3. **Startup Probes**: Detect if service has started properly

## Troubleshooting Guide

### Common Issues and Solutions

#### High Error Rates

**Symptoms:**
- Increased 5xx responses
- User complaints about failures
- Alert notifications

**Investigation:**
1. Check error logs in Loki
2. Identify error patterns and frequency
3. Review recent deployments
4. Check external service dependencies

**Solutions:**
- Rollback recent deployment
- Scale up services
- Enable circuit breakers
- Fix identified bugs

#### Performance Degradation

**Symptoms:**
- Increased response times
- Timeouts
- User complaints about slowness

**Investigation:**
1. Check response time percentiles
2. Review database query performance
3. Check resource utilization
4. Analyze distributed traces in Jaeger

**Solutions:**
- Scale resources horizontally
- Optimize database queries
- Add caching layers
- Review and optimize algorithms

#### Service Discovery Issues

**Symptoms:**
- Services cannot find each other
- Connection refused errors
- Intermittent failures

**Investigation:**
1. Check service mesh configuration
2. Verify DNS resolution
3. Check network policies
4. Review service registration

**Solutions:**
- Restart affected services
- Update service discovery configuration
- Fix network policies
- Re-register services

### Log Analysis

**Error Log Analysis:**
```bash
# Recent errors
kubectl logs -l app=claude-tui --tail=1000 | grep ERROR

# Specific time range
kubectl logs -l app=claude-tui --since=1h | grep ERROR

# Pattern matching
kubectl logs -l app=claude-tui | grep -E "(timeout|connection|refused)"
```

**Loki Query Examples:**
```
# All errors in last hour
{job="claude-tui"} |= "ERROR" | json

# Specific service errors
{job="hive-mind-coordinator"} |= "ERROR" 

# Performance issues
{job="claude-tui"} |= "slow" or "timeout" or "latency"

# Database errors
{job="claude-tui"} |= "database" |= "error"
```

## Incident Response Workflow

### 1. Detection and Alerting
- Automated alerts via AlertManager
- Monitoring dashboard notifications
- User reports or external monitoring

### 2. Initial Response
- Acknowledge alert in PagerDuty
- Join incident channel (#incidents-critical)
- Begin triage and assessment

### 3. Investigation
- Check monitoring dashboards
- Review recent changes and deployments
- Analyze logs and metrics
- Identify root cause

### 4. Mitigation
- Apply immediate fixes (restart, scale, rollback)
- Implement workarounds if needed
- Monitor recovery progress

### 5. Resolution
- Confirm service restoration
- Update incident status
- Document lessons learned

### 6. Post-Incident Review
- Conduct blameless post-mortem
- Identify improvement opportunities
- Implement preventive measures
- Update runbooks and procedures

## Maintenance Procedures

### Regular Maintenance Tasks

**Daily:**
- Review monitoring dashboards
- Check alert status and resolve any issues
- Monitor resource utilization trends
- Review error rates and performance metrics

**Weekly:**
- Analyze performance trends
- Review capacity planning metrics
- Update alert thresholds as needed
- Check backup and recovery procedures

**Monthly:**
- Review and update runbooks
- Conduct disaster recovery testing
- Update monitoring configurations
- Review SLO/SLA compliance

### Monitoring Stack Maintenance

**Prometheus:**
- Monitor disk usage and retention
- Review and optimize query performance
- Update recording rules as needed
- Backup configuration and rules

**Grafana:**
- Update dashboards and panels
- Review and organize dashboard folders
- Update data sources and plugins
- Backup dashboards and configuration

**AlertManager:**
- Review and update alert rules
- Test notification channels
- Update routing and grouping rules
- Review silence and inhibition rules

## Escalation Procedures

### Level 1: On-Call Engineer
- Initial response and triage
- Basic troubleshooting and fixes
- Monitoring and status updates

### Level 2: Senior Engineer/Tech Lead
- Complex troubleshooting
- Architecture-level decisions
- Coordination with other teams

### Level 3: Engineering Manager/CTO
- Major incident coordination
- External communication
- Resource allocation decisions

### Communication Channels

**Internal:**
- Slack: #incidents-critical, #monitoring
- PagerDuty: Incident management
- Email: Critical updates and summaries

**External:**
- Status page updates
- Customer communication
- Vendor escalation

## Monitoring Best Practices

### Alert Design
1. **Actionable**: Every alert should require human action
2. **Meaningful**: Alerts should indicate real problems
3. **Timely**: Alerts should fire before users are impacted
4. **Proportional**: Alert severity should match impact

### Dashboard Design
1. **Purpose-built**: Each dashboard serves a specific role
2. **Hierarchical**: High-level overview with drill-down capability
3. **Consistent**: Common layouts and conventions
4. **Up-to-date**: Regular maintenance and updates

### Metric Collection
1. **Comprehensive**: Cover all critical system components
2. **Efficient**: Minimize monitoring overhead
3. **Standardized**: Consistent naming and labeling
4. **Documented**: Clear metric definitions and usage

## Useful Queries and Commands

### Prometheus Queries

**Service availability:**
```promql
up{job="claude-tui"}
```

**Request rate:**
```promql
rate(http_requests_total[5m])
```

**Error rate:**
```promql
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

**Response time percentiles:**
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Kubernetes Commands

**Pod status:**
```bash
kubectl get pods -A | grep -v Running
```

**Resource usage:**
```bash
kubectl top pods -A --sort-by=cpu
kubectl top pods -A --sort-by=memory
```

**Event monitoring:**
```bash
kubectl get events --sort-by=.metadata.creationTimestamp
```

**Log streaming:**
```bash
kubectl logs -f deployment/claude-tui -n production
```

## Contact Information

| Role | Contact | Method |
|------|---------|---------|
| On-Call Engineer | @oncall | Slack |
| DevOps Team | #devops | Slack |
| Security Team | security@hive-mind.ai | Email |
| Management | #incidents-critical | Slack |

---

**Last Updated**: 2024-01-01
**Next Review**: 2024-02-01
**Version**: 1.0.0