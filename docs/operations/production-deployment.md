# Production Deployment Guide

This comprehensive guide covers deploying Claude-TUI's intelligent development brain to production environments with high availability, scalability, and security.

## 🎯 Deployment Overview

### Architecture Patterns

Claude-TUI supports multiple deployment architectures:

1. **Single Instance**: Development and small teams
2. **Load Balanced**: Medium-scale deployments  
3. **Microservices**: Enterprise-scale deployments
4. **Hybrid Cloud**: Multi-cloud deployments
5. **Edge Computing**: Distributed global deployments

### Production Requirements

#### Minimum Production Setup
- **Compute**: 4 vCPU, 8GB RAM, 100GB SSD
- **Database**: PostgreSQL 15+ with 50GB storage
- **Cache**: Redis cluster with 4GB memory
- **Load Balancer**: NGINX or equivalent
- **Monitoring**: Prometheus + Grafana stack

#### Enterprise Setup
- **Compute**: Auto-scaling cluster (8-32 vCPUs per node)
- **Database**: PostgreSQL cluster with read replicas
- **Cache**: Redis Cluster with high availability
- **Container Orchestration**: Kubernetes cluster
- **Monitoring**: Full observability stack

## 🐳 Container Deployment

### Production Docker Configuration

#### Multi-stage Dockerfile

```dockerfile
# Build stage
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git curl nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt requirements-prod.txt ./
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
    -r requirements.txt -r requirements-prod.txt

# Production stage
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r claudetui && useradd -r -g claudetui claudetui

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git curl nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels and install
COPY --from=builder /wheels /wheels
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache --no-index --find-links /wheels \
    -r requirements.txt -r requirements-prod.txt \
    && rm -rf /wheels

# Copy application
WORKDIR /app
COPY . .
RUN pip install --no-deps -e .

# Set up directories and permissions
RUN mkdir -p /app/{projects,logs,cache,tmp} && \
    chown -R claudetui:claudetui /app && \
    chmod -R 755 /app

USER claudetui

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health').raise_for_status()" \
    || exit 1

# Security configurations
ENV PYTHONPATH=/app/src
ENV PYTHONHASHSEED=random
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Production startup command
CMD ["python", "-m", "uvicorn", "api.main:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

#### Production Docker Compose

```yaml
version: '3.8'

services:
  claude-tui-api:
    build:
      context: .
      dockerfile: Dockerfile.production
    image: claude-tui:${VERSION:-latest}
    container_name: claude-tui-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
      - REDIS_URL=redis://redis-cluster:6379
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - WORKERS=4
      - MAX_MEMORY=2G
    volumes:
      - ./data/projects:/app/projects
      - ./data/logs:/app/logs
      - ./data/cache:/app/cache
      - ./config/production.yaml:/app/config/production.yaml:ro
    networks:
      - claude-network
    depends_on:
      - postgres
      - redis-cluster
      - prometheus
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15-alpine
    container_name: claude-tui-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./scripts/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - claude-network
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis-cluster:
    image: redis:7-alpine
    container_name: claude-tui-redis
    restart: unless-stopped
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    volumes:
      - redis_data:/data
      - ./config/redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    networks:
      - claude-network
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: claude-tui-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./config/nginx/sites-enabled:/etc/nginx/sites-enabled:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./static:/var/www/static:ro
    networks:
      - claude-network
    depends_on:
      - claude-tui-api
    deploy:
      resources:
        limits:
          memory: 128M
          cpus: '0.25'

  prometheus:
    image: prom/prometheus:latest
    container_name: claude-tui-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./config/prometheus/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    networks:
      - claude-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: claude-tui-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - claude-network

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /var/lib/claude-tui/postgres
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /var/lib/claude-tui/redis
  prometheus_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /var/lib/claude-tui/prometheus
  grafana_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /var/lib/claude-tui/grafana

networks:
  claude-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
```

### Environment Configuration

#### Production Environment File

```bash
# .env.production

# Application Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-super-secret-key-here

# Database Configuration
DATABASE_URL=postgresql://claude_user:secure_password@postgres:5432/claude_tui
DATABASE_POOL_SIZE=20
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Redis Configuration  
REDIS_URL=redis://redis-cluster:6379/0
REDIS_POOL_SIZE=10
REDIS_SOCKET_TIMEOUT=5

# AI Services
CLAUDE_API_KEY=your-claude-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_URL=https://api.anthropic.com

# Security Configuration
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS Configuration
CORS_ORIGINS=["https://yourdomain.com", "https://app.yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true

# Performance Configuration
WORKERS=4
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50
PRELOAD_APP=true
TIMEOUT=30
KEEPALIVE=2

# Resource Limits
MAX_MEMORY=2G
MAX_CPU=2.0
MAX_TASKS_PER_AGENT=5
CACHE_TTL=3600

# Monitoring Configuration
PROMETHEUS_METRICS=true
GRAFANA_ENABLED=true
HEALTH_CHECK_INTERVAL=30

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_RETENTION_DAYS=30
```

## ☸️ Kubernetes Deployment

### Namespace and Configuration

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: claude-tui-prod
  labels:
    name: claude-tui-prod
    environment: production

---
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claude-tui-config
  namespace: claude-tui-prod
data:
  production.yaml: |
    api:
      host: "0.0.0.0"
      port: 8080
      workers: 4
      
    database:
      pool_size: 20
      pool_timeout: 30
      
    redis:
      pool_size: 10
      socket_timeout: 5
      
    monitoring:
      prometheus_enabled: true
      metrics_port: 9000
      
    security:
      cors_origins:
        - "https://yourdomain.com"
        - "https://app.yourdomain.com"
      
    performance:
      max_memory: "2G"
      max_cpu: 2.0
      cache_ttl: 3600

---
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: claude-tui-secrets
  namespace: claude-tui-prod
type: Opaque
stringData:
  database-url: "postgresql://claude_user:secure_password@postgres:5432/claude_tui"
  redis-url: "redis://redis-cluster:6379/0"
  claude-api-key: "your-claude-api-key"
  jwt-secret-key: "your-jwt-secret-key"
  secret-key: "your-super-secret-key"
```

### Application Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-tui-api
  namespace: claude-tui-prod
  labels:
    app: claude-tui-api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: claude-tui-api
  template:
    metadata:
      labels:
        app: claude-tui-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: claude-tui-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: claude-tui-api
        image: claude-tui:v1.0.0
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9000
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: claude-tui-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: claude-tui-secrets
              key: redis-url
        - name: CLAUDE_API_KEY
          valueFrom:
            secretKeyRef:
              name: claude-tui-secrets
              key: claude-api-key
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: claude-tui-secrets
              key: jwt-secret-key
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: projects-storage
          mountPath: /app/projects
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 2
      volumes:
      - name: config
        configMap:
          name: claude-tui-config
      - name: projects-storage
        persistentVolumeClaim:
          claimName: claude-tui-projects-pvc
      - name: logs
        emptyDir: {}
      nodeSelector:
        kubernetes.io/arch: amd64
      tolerations:
      - key: "node.kubernetes.io/not-ready"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 30

---
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: claude-tui-api-hpa
  namespace: claude-tui-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: claude-tui-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30

---
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: claude-tui-api-service
  namespace: claude-tui-prod
  labels:
    app: claude-tui-api
spec:
  type: ClusterIP
  selector:
    app: claude-tui-api
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  - name: metrics
    port: 9000
    targetPort: metrics
    protocol: TCP

---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: claude-tui-api-ingress
  namespace: claude-tui-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://yourdomain.com"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: claude-tui-api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: claude-tui-api-service
            port:
              number: 80
```

### Database Deployment

```yaml
# kubernetes/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: claude-tui-prod
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: "claude_tui"
        - name: POSTGRES_USER
          value: "claude_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secrets
              key: postgres-password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        ports:
        - containerPort: 5432
          name: postgres
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - claude_user
            - -d
            - claude_tui
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - claude_user
            - -d
            - claude_tui
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
      storageClassName: fast-ssd

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: claude-tui-prod
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

## 🔧 Configuration Management

### Production Configuration File

```yaml
# config/production.yaml
api:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  worker_class: "uvicorn.workers.UvicornWorker"
  max_requests: 1000
  max_requests_jitter: 50
  preload_app: true
  timeout: 30
  keepalive: 2

database:
  url: "${DATABASE_URL}"
  pool_size: 20
  pool_timeout: 30
  pool_recycle: 3600
  echo: false
  
redis:
  url: "${REDIS_URL}"
  pool_size: 10
  socket_timeout: 5
  socket_connect_timeout: 5
  retry_on_timeout: true
  
ai_services:
  claude:
    api_key: "${CLAUDE_API_KEY}"
    model: "claude-3-sonnet-20241022"
    timeout: 60
    max_retries: 3
    rate_limit: 100
    
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    timeout: 30
    
security:
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    access_token_expire_minutes: 30
    refresh_token_expire_days: 7
    
  cors:
    allow_origins: 
      - "https://yourdomain.com"
      - "https://app.yourdomain.com"
    allow_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: ["*"]
    allow_credentials: true
    
  rate_limiting:
    default: "1000/hour"
    auth: "100/hour"
    ai_operations: "500/hour"
    
monitoring:
  prometheus:
    enabled: true
    port: 9000
    path: "/metrics"
    
  logging:
    level: "INFO"
    format: "json"
    handlers:
      - type: "console"
      - type: "file"
        filename: "/app/logs/claude-tui.log"
        max_size: "100MB"
        backup_count: 10
        
performance:
  caching:
    enabled: true
    default_ttl: 3600
    max_size: "500MB"
    
  resource_limits:
    max_memory: "2G"
    max_cpu: 2.0
    max_tasks_per_agent: 5
    
  optimization:
    enable_compression: true
    connection_pool_size: 20
    async_pool_size: 100
    
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  storage_path: "/var/lib/claude-tui/backups"
  compress: true
  
alerts:
  email:
    enabled: true
    smtp_host: "smtp.yourdomain.com"
    smtp_port: 587
    from_address: "alerts@yourdomain.com"
    
  webhooks:
    enabled: true
    endpoints:
      - name: "slack"
        url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
      - name: "pagerduty"
        url: "https://events.pagerduty.com/v2/enqueue"
```

## 🔍 Monitoring & Observability

### Prometheus Configuration

```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'claude-tui-prod'
    
rule_files:
  - "rules/*.yml"
  
scrape_configs:
  - job_name: 'claude-tui-api'
    static_configs:
      - targets: ['claude-tui-api-service:9000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
      
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093
```

### Alert Rules

```yaml
# config/prometheus/rules/claude-tui-alerts.yml
groups:
- name: claude-tui-alerts
  rules:
  
  # High Error Rate
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} requests per second"
      
  # High Response Time
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"
      
  # Database Connection Issues
  - alert: DatabaseConnectionHigh
    expr: db_connections_active / db_connections_max > 0.8
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Database connection pool nearly exhausted"
      description: "Active connections: {{ $value }}"
      
  # Memory Usage
  - alert: HighMemoryUsage
    expr: (process_resident_memory_bytes / 1024 / 1024 / 1024) > 1.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }} GB"
      
  # AI Service Availability
  - alert: AIServiceDown
    expr: ai_service_availability < 1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "AI service unavailable"
      description: "{{ $labels.service }} is not responding"
```

### Grafana Dashboards

```json
// config/grafana/dashboards/claude-tui-overview.json
{
  "dashboard": {
    "id": null,
    "title": "Claude-TUI Production Overview",
    "tags": ["claude-tui", "production"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m]))",
            "legendFormat": "Requests/sec"
          }
        ],
        "yAxes": [
          {
            "label": "Requests per second"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active AI Agents",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(ai_agents_active)",
            "legendFormat": "Active Agents"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          },
          {
            "expr": "rate(process_cpu_seconds_total[5m]) * 100",
            "legendFormat": "CPU %"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## 🚨 Incident Response

### Automated Recovery Scripts

```bash
#!/bin/bash
# scripts/production/auto-recovery.sh

set -e

NAMESPACE="claude-tui-prod"
SERVICE="claude-tui-api"
THRESHOLD_CPU=80
THRESHOLD_MEMORY=85
MAX_RESTART_COUNT=3

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_health() {
    local service=$1
    kubectl get pods -n $NAMESPACE -l app=$service -o json | \
        jq -r '.items[] | select(.status.phase != "Running" or .status.containerStatuses[].restartCount > '$MAX_RESTART_COUNT') | .metadata.name'
}

restart_unhealthy_pods() {
    local unhealthy_pods=$(check_health $SERVICE)
    
    if [ -n "$unhealthy_pods" ]; then
        log "Found unhealthy pods: $unhealthy_pods"
        
        for pod in $unhealthy_pods; do
            log "Restarting pod: $pod"
            kubectl delete pod -n $NAMESPACE $pod
            
            # Wait for replacement pod
            sleep 30
            
            # Verify new pod is healthy
            kubectl wait --for=condition=Ready pod -l app=$SERVICE -n $NAMESPACE --timeout=300s
        done
    fi
}

scale_on_load() {
    local current_replicas=$(kubectl get deployment $SERVICE -n $NAMESPACE -o jsonpath='{.status.replicas}')
    local cpu_usage=$(kubectl top pods -n $NAMESPACE -l app=$SERVICE --no-headers | awk '{sum+=$2} END {print sum/NR}' | sed 's/[^0-9]//g')
    
    if [ "$cpu_usage" -gt "$THRESHOLD_CPU" ] && [ "$current_replicas" -lt "10" ]; then
        log "High CPU usage detected ($cpu_usage%). Scaling up..."
        kubectl scale deployment $SERVICE -n $NAMESPACE --replicas=$((current_replicas + 1))
    fi
}

main() {
    log "Starting automated recovery check..."
    
    restart_unhealthy_pods
    scale_on_load
    
    log "Recovery check completed"
}

main "$@"
```

### Health Check Endpoint

```python
# src/api/health.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import psutil
import time
import asyncio
from datetime import datetime
from typing import Dict, Any

from api.dependencies.database import get_database
from core.ai_interface import AIInterface

router = APIRouter()

@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database check
    try:
        async with get_database() as db:
            await db.execute("SELECT 1")
            health_status["checks"]["database"] = {
                "status": "healthy",
                "response_time_ms": 0  # Measure actual response time
            }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Redis check
    try:
        from core.cache import redis_client
        await redis_client.ping()
        health_status["checks"]["redis"] = {
            "status": "healthy"
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # AI service check
    try:
        ai_interface = AIInterface()
        test_result = await ai_interface.test_connectivity()
        health_status["checks"]["ai_services"] = {
            "status": "healthy" if test_result else "unhealthy",
            "claude": test_result
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["ai_services"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # System resources check
    memory_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent(interval=1)
    disk_percent = psutil.disk_usage('/').percent
    
    health_status["checks"]["system"] = {
        "status": "healthy" if memory_percent < 90 and cpu_percent < 90 else "degraded",
        "memory_percent": memory_percent,
        "cpu_percent": cpu_percent,
        "disk_percent": disk_percent
    }
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    
    try:
        # Quick checks for essential services
        async with get_database() as db:
            await db.execute("SELECT 1")
        
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")
```

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] **Environment Setup**
  - [ ] Production environment configured
  - [ ] Secrets properly stored and encrypted
  - [ ] SSL certificates installed
  - [ ] DNS configured correctly

- [ ] **Infrastructure**
  - [ ] Database cluster operational
  - [ ] Redis cluster configured
  - [ ] Load balancer configured
  - [ ] Monitoring stack deployed

- [ ] **Security**
  - [ ] Security scan completed
  - [ ] Penetration testing performed
  - [ ] Access controls configured
  - [ ] Firewall rules implemented

- [ ] **Testing**
  - [ ] All tests passing
  - [ ] Load testing completed
  - [ ] Integration tests verified
  - [ ] Security tests passed

### Deployment

- [ ] **Application Deployment**
  - [ ] Code deployed to production
  - [ ] Database migrations executed
  - [ ] Configuration validated
  - [ ] Health checks passing

- [ ] **Verification**
  - [ ] All endpoints responding
  - [ ] AI services connected
  - [ ] Database connectivity verified
  - [ ] Cache functionality working

### Post-Deployment

- [ ] **Monitoring**
  - [ ] Metrics collection active
  - [ ] Alerts configured
  - [ ] Dashboards accessible
  - [ ] Log aggregation working

- [ ] **Documentation**
  - [ ] Deployment notes updated
  - [ ] Runbooks updated
  - [ ] Team notified
  - [ ] Change requests closed

## 🔄 Maintenance & Updates

### Rolling Updates

```bash
#!/bin/bash
# scripts/production/rolling-update.sh

NAMESPACE="claude-tui-prod"
DEPLOYMENT="claude-tui-api"
NEW_IMAGE="claude-tui:v1.1.0"

# Update deployment image
kubectl set image deployment/$DEPLOYMENT \
    claude-tui-api=$NEW_IMAGE \
    -n $NAMESPACE

# Watch rollout status
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# Verify new pods are healthy
kubectl get pods -n $NAMESPACE -l app=claude-tui-api

echo "Rolling update completed successfully"
```

### Backup Procedures

```bash
#!/bin/bash
# scripts/production/backup.sh

BACKUP_DIR="/var/lib/claude-tui/backups"
DATE=$(date +%Y%m%d_%H%M%S)
NAMESPACE="claude-tui-prod"

# Database backup
kubectl exec -n $NAMESPACE postgres-0 -- pg_dump \
    -U claude_user -d claude_tui \
    | gzip > "$BACKUP_DIR/database_$DATE.sql.gz"

# Configuration backup
kubectl get configmaps -n $NAMESPACE -o yaml > "$BACKUP_DIR/configmaps_$DATE.yaml"
kubectl get secrets -n $NAMESPACE -o yaml > "$BACKUP_DIR/secrets_$DATE.yaml"

# Projects backup
kubectl exec -n $NAMESPACE \
    $(kubectl get pods -n $NAMESPACE -l app=claude-tui-api -o jsonpath='{.items[0].metadata.name}') \
    -- tar czf - /app/projects | cat > "$BACKUP_DIR/projects_$DATE.tar.gz"

echo "Backup completed: $DATE"
```

---

*This production deployment guide ensures your Claude-TUI intelligent development brain runs reliably, securely, and efficiently in production environments. For specific deployment scenarios or troubleshooting, see our [operations documentation](../operations/) and [support resources](../support/).*