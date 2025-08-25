# Claude-TIU Installation & Deployment Guide

## 🚀 Complete Installation & Deployment Documentation

This comprehensive guide covers all installation methods and deployment options for Claude-TIU, from local development to enterprise-scale production deployments.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Local Development Setup](#local-development-setup)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Provider Deployments](#cloud-provider-deployments)
7. [Enterprise Configuration](#enterprise-configuration)
8. [Monitoring & Observability](#monitoring--observability)
9. [Security Hardening](#security-hardening)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
| Component | Specification | Notes |
|-----------|---------------|--------|
| **OS** | Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ | WSL2 recommended for Windows |
| **CPU** | 2 cores, 2.0 GHz | Multi-core recommended for AI workloads |
| **RAM** | 4 GB | 8 GB+ recommended for optimal performance |
| **Storage** | 10 GB free space | SSD recommended |
| **Network** | Stable internet connection | Required for AI service APIs |
| **Python** | 3.9+ | Python 3.11+ recommended |
| **Node.js** | 16+ | Required for Claude Flow |

### Recommended Requirements (Production)
| Component | Specification | Notes |
|-----------|---------------|--------|
| **CPU** | 8+ cores, 3.0+ GHz | For concurrent AI operations |
| **RAM** | 32 GB+ | Supports multiple projects and caching |
| **Storage** | 100 GB+ NVMe SSD | Fast I/O for project files and caching |
| **Network** | Gigabit connection | Low latency to AI services |
| **Database** | PostgreSQL 14+ | For persistent data and analytics |
| **Cache** | Redis 6+ | For session and response caching |

### Enterprise Scale Requirements
| Component | Specification | Notes |
|-----------|---------------|--------|
| **CPU** | 16+ cores, 3.5+ GHz | Multi-node cluster support |
| **RAM** | 64 GB+ per node | High-concurrency workloads |
| **Storage** | 500 GB+ per node | Distributed storage recommended |
| **Load Balancer** | HAProxy/NGINX | For high availability |
| **Monitoring** | Prometheus + Grafana | Full observability stack |

---

## Quick Installation

### One-Line Install (Recommended)
```bash
# Install Claude-TIU with all dependencies
curl -sSL https://install.claude-tiu.dev/quick | bash

# Or using wget
wget -qO- https://install.claude-tiu.dev/quick | bash
```

This installer will:
- ✅ Check system compatibility
- ✅ Install Python dependencies
- ✅ Setup Claude Flow integration
- ✅ Configure default settings
- ✅ Create desktop shortcuts
- ✅ Run initial setup wizard

### Verify Installation
```bash
# Check Claude-TIU version
claude-tiu --version
# Expected: claude-tiu 1.0.0

# Run system check
claude-tiu doctor
# Should show all green checkmarks

# Test AI connectivity
claude-tiu test-connection
# Verifies Claude API access
```

---

## Local Development Setup

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv claude-tiu-env
source claude-tiu-env/bin/activate  # Linux/macOS
# claude-tiu-env\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install from Source
```bash
# Clone repository
git clone https://github.com/claude-tiu/claude-tiu.git
cd claude-tiu

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .[dev]
```

### Step 3: Configure Environment
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

**Required Environment Variables:**
```bash
# API Configuration
CLAUDE_API_KEY=sk-your-claude-api-key-here
CLAUDE_FLOW_ENDPOINT=http://localhost:3000

# Application Settings
DEBUG=True
LOG_LEVEL=DEBUG
MAX_CONCURRENT_TASKS=5

# Database Configuration (Optional for development)
DATABASE_URL=sqlite:///./claude_tiu.db
# DATABASE_URL=postgresql://user:pass@localhost:5432/claude_tiu

# Cache Configuration (Optional)
REDIS_URL=redis://localhost:6379/0

# Security Settings
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
```

### Step 4: Initialize Database
```bash
# Initialize database
claude-tiu db init

# Run migrations
claude-tiu db upgrade

# Create admin user (optional)
claude-tiu create-user --email admin@example.com --role admin
```

### Step 5: Start Development Services
```bash
# Terminal 1: Start main application
claude-tiu run --debug

# Terminal 2: Start Claude Flow (if using)
npx claude-flow@alpha serve --port 3000

# Terminal 3: Start Redis (optional)
redis-server

# Terminal 4: Start PostgreSQL (optional)
pg_ctl -D /usr/local/var/postgres start
```

### Development Workflow
```bash
# Run tests
pytest tests/

# Code formatting
black src/
isort src/

# Type checking
mypy src/

# Security scan
bandit -r src/

# Full development check
make dev-check
```

---

## Docker Deployment

### Single Container (Development)
```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Flow
RUN npm install -g claude-flow@alpha

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Expose port
EXPOSE 8000

# Start command
CMD ["claude-tiu", "run", "--host", "0.0.0.0"]
```

```bash
# Build and run development container
docker build -f Dockerfile.dev -t claude-tiu:dev .
docker run -p 8000:8000 -e CLAUDE_API_KEY=your-key claude-tiu:dev
```

### Multi-Container Production (Docker Compose)
```yaml
# docker-compose.yml
version: '3.8'

services:
  claude-tiu:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/claude_tiu
      - REDIS_URL=redis://redis:6379/0
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - ./projects:/app/projects
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=claude_tiu
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - claude-tiu
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

**Production Dockerfile:**
```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Production image
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r claude && useradd --no-log-init -r -g claude claude

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Flow
RUN npm install -g claude-flow@alpha

# Copy wheels and install
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy application
COPY . .
RUN pip install -e . && \
    chown -R claude:claude /app

# Switch to non-root user
USER claude

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "src.api.main:app"]
```

### Deploy with Docker Compose
```bash
# Create environment file
cp .env.example .env
nano .env  # Configure your settings

# Deploy stack
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale claude-tiu=3

# Update services
docker-compose pull
docker-compose up -d
```

---

## Kubernetes Deployment

### Namespace and Configuration
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: claude-tiu
  labels:
    name: claude-tiu
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claude-tiu-config
  namespace: claude-tiu
data:
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  MAX_CONCURRENT_TASKS: "10"
  DATABASE_URL: "postgresql://postgres:password@postgres:5432/claude_tiu"
  REDIS_URL: "redis://redis:6379/0"
```

### Secrets Management
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: claude-tiu-secrets
  namespace: claude-tiu
type: Opaque
data:
  claude-api-key: <base64-encoded-key>
  secret-key: <base64-encoded-secret>
  jwt-secret-key: <base64-encoded-jwt-secret>
  db-password: <base64-encoded-password>
```

```bash
# Create secrets from command line
kubectl create secret generic claude-tiu-secrets \
  --from-literal=claude-api-key=sk-your-key-here \
  --from-literal=secret-key=your-secret-key \
  --from-literal=jwt-secret-key=your-jwt-secret \
  --namespace=claude-tiu
```

### PostgreSQL Deployment
```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: claude-tiu
spec:
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
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: claude_tiu
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: db-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: claude-tiu
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: claude-tiu
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### Redis Deployment
```yaml
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: claude-tiu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: claude-tiu
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: claude-tiu
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

### Claude-TIU Application Deployment
```yaml
# k8s/claude-tiu.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-tiu
  namespace: claude-tiu
  labels:
    app: claude-tiu
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-tiu
  template:
    metadata:
      labels:
        app: claude-tiu
    spec:
      containers:
      - name: claude-tiu
        image: claude-tiu/claude-tiu:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: CLAUDE_API_KEY
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: claude-api-key
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: secret-key
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: jwt-secret-key
        envFrom:
        - configMapRef:
            name: claude-tiu-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: projects-storage
          mountPath: /app/projects
        - name: logs-storage
          mountPath: /app/logs
      volumes:
      - name: projects-storage
        persistentVolumeClaim:
          claimName: projects-pvc
      - name: logs-storage
        persistentVolumeClaim:
          claimName: logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: claude-tiu-service
  namespace: claude-tiu
spec:
  selector:
    app: claude-tiu
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: projects-pvc
  namespace: claude-tiu
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: logs-pvc
  namespace: claude-tiu
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
```

### Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: claude-tiu-ingress
  namespace: claude-tiu
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
spec:
  tls:
  - hosts:
    - claude-tiu.your-domain.com
    secretName: claude-tiu-tls
  rules:
  - host: claude-tiu.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: claude-tiu-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: claude-tiu-hpa
  namespace: claude-tiu
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: claude-tiu
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
```

### Deploy to Kubernetes
```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n claude-tiu

# View logs
kubectl logs -f deployment/claude-tiu -n claude-tiu

# Scale deployment
kubectl scale deployment claude-tiu --replicas=5 -n claude-tiu

# Port forward for testing
kubectl port-forward service/claude-tiu-service 8080:80 -n claude-tiu
```

---

## Cloud Provider Deployments

### Amazon Web Services (AWS)

#### ECS Deployment
```json
{
  "family": "claude-tiu",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "claude-tiu",
      "image": "your-account.dkr.ecr.region.amazonaws.com/claude-tiu:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/claude-tiu",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://username:password@your-rds-endpoint:5432/claude_tiu"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://your-elasticache-endpoint:6379/0"
        }
      ],
      "secrets": [
        {
          "name": "CLAUDE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:claude-tiu/api-key"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### EKS with Helm Chart
```yaml
# helm-chart/values.yaml
replicaCount: 3

image:
  repository: your-account.dkr.ecr.region.amazonaws.com/claude-tiu
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
  hosts:
    - host: claude-tiu.your-domain.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: false
  external:
    host: your-rds-endpoint
    database: claude_tiu

redis:
  enabled: false
  external:
    host: your-elasticache-endpoint
```

### Google Cloud Platform (GCP)

#### Cloud Run Deployment
```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: claude-tiu
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 10
      timeoutSeconds: 600
      containers:
      - image: gcr.io/your-project/claude-tiu:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://username:password@your-cloud-sql-ip:5432/claude_tiu"
        - name: REDIS_URL
          value: "redis://your-memorystore-ip:6379/0"
        - name: CLAUDE_API_KEY
          valueFrom:
            secretKeyRef:
              key: claude-api-key
              name: claude-tiu-secrets
        resources:
          limits:
            cpu: "2000m"
            memory: "4Gi"
          requests:
            cpu: "1000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
```

#### GKE Deployment
```bash
# Create GKE cluster
gcloud container clusters create claude-tiu-cluster \
  --zone=us-central1-a \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --machine-type=n1-standard-2

# Deploy application
kubectl apply -f k8s/

# Setup ingress with SSL
gcloud compute ssl-certificates create claude-tiu-ssl \
  --domains=claude-tiu.your-domain.com
```

### Microsoft Azure

#### Container Instances
```yaml
# azure-container-instances.yaml
apiVersion: 2021-09-01
location: East US
name: claude-tiu-container-group
properties:
  containers:
  - name: claude-tiu
    properties:
      image: your-registry.azurecr.io/claude-tiu:latest
      resources:
        requests:
          cpu: 1.0
          memoryInGB: 2
      ports:
      - protocol: TCP
        port: 8000
      environmentVariables:
      - name: DATABASE_URL
        value: "postgresql://username@your-postgres:password@your-postgres.postgres.database.azure.com:5432/claude_tiu"
      - name: REDIS_URL
        value: "redis://your-redis.redis.cache.windows.net:6380"
      - name: CLAUDE_API_KEY
        secureValue: "your-claude-api-key"
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 80
  restartPolicy: Always
type: Microsoft.ContainerInstance/containerGroups
```

#### AKS Deployment
```bash
# Create AKS cluster
az aks create \
  --resource-group claude-tiu-rg \
  --name claude-tiu-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group claude-tiu-rg --name claude-tiu-aks

# Deploy application
kubectl apply -f k8s/
```

---

## Enterprise Configuration

### High Availability Setup
```yaml
# ha-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claude-tiu-ha-config
data:
  # Load balancing
  LOAD_BALANCER_ALGORITHM: "round_robin"
  STICKY_SESSIONS: "true"
  HEALTH_CHECK_INTERVAL: "30"
  
  # Database clustering
  DATABASE_CLUSTER_NODES: "postgres-primary,postgres-replica1,postgres-replica2"
  DATABASE_READ_REPLICA_COUNT: "2"
  DATABASE_CONNECTION_POOL_SIZE: "20"
  
  # Cache clustering
  REDIS_CLUSTER_NODES: "redis-node1,redis-node2,redis-node3"
  REDIS_REPLICATION_FACTOR: "2"
  
  # Application clustering
  APPLICATION_CLUSTER_SIZE: "5"
  AUTO_SCALING_MIN_REPLICAS: "3"
  AUTO_SCALING_MAX_REPLICAS: "20"
  AUTO_SCALING_CPU_THRESHOLD: "70"
  AUTO_SCALING_MEMORY_THRESHOLD: "80"
```

### Multi-Region Deployment
```bash
#!/bin/bash
# multi-region-deploy.sh

# Define regions
REGIONS=("us-east-1" "us-west-2" "eu-west-1" "ap-southeast-1")

# Deploy to each region
for region in "${REGIONS[@]}"; do
    echo "Deploying to $region..."
    
    # Set region context
    export AWS_REGION=$region
    
    # Deploy infrastructure
    terraform apply -var="region=$region" -auto-approve
    
    # Deploy application
    kubectl apply -f k8s/ --context="cluster-$region"
    
    # Configure cross-region replication
    aws rds create-db-cluster-snapshot \
        --db-cluster-snapshot-identifier "claude-tiu-snapshot-$(date +%Y%m%d)" \
        --db-cluster-identifier "claude-tiu-cluster-$region"
    
    echo "Deployment to $region completed"
done

# Setup global load balancer
aws route53 create-hosted-zone \
    --name claude-tiu.com \
    --caller-reference "$(date +%s)"

# Configure failover routing
for region in "${REGIONS[@]}"; do
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z123456789 \
        --change-batch file://dns-config-$region.json
done
```

### Enterprise Security Configuration
```yaml
# security-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claude-tiu-security-config
data:
  # Authentication
  JWT_ALGORITHM: "RS256"
  JWT_EXPIRATION: "3600"
  REFRESH_TOKEN_EXPIRATION: "604800"
  PASSWORD_MIN_LENGTH: "12"
  MFA_REQUIRED: "true"
  
  # Authorization
  RBAC_ENABLED: "true"
  PERMISSION_CACHE_TTL: "300"
  SESSION_TIMEOUT: "1800"
  
  # Network Security
  ALLOWED_ORIGINS: "https://claude-tiu.com,https://app.claude-tiu.com"
  RATE_LIMIT_REQUESTS: "1000"
  RATE_LIMIT_WINDOW: "3600"
  
  # Data Protection
  ENCRYPTION_AT_REST: "true"
  ENCRYPTION_IN_TRANSIT: "true"
  DATA_RETENTION_DAYS: "90"
  AUDIT_LOG_ENABLED: "true"
  
  # API Security
  API_KEY_ROTATION_DAYS: "30"
  REQUEST_SIZE_LIMIT: "50MB"
  RESPONSE_SIZE_LIMIT: "100MB"
```

---

## Monitoring & Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "claude-tiu-rules.yml"

scrape_configs:
  - job_name: 'claude-tiu'
    static_configs:
      - targets: ['claude-tiu:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Claude-TIU Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "AI Request Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "ai_request_duration_seconds",
            "legendFormat": "AI Response Time"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration
```yaml
# logging/fluent-bit.conf
[SERVICE]
    Flush         1
    Log_Level     info
    Daemon        off
    HTTP_Server   On
    HTTP_Listen   0.0.0.0
    HTTP_Port     2020

[INPUT]
    Name              tail
    Path              /app/logs/*.log
    Parser            json
    Tag               claude-tiu.*
    Refresh_Interval  5

[FILTER]
    Name                kubernetes
    Match               claude-tiu.*
    Use_Journal         Off
    Merge_Log           On
    K8S-Logging.Parser  On
    K8S-Logging.Exclude Off

[OUTPUT]
    Name  es
    Match *
    Host  elasticsearch
    Port  9200
    Index claude-tiu-logs
    Type  _doc
```

### Health Checks
```python
# health_checks.py
from typing import Dict, Any
import asyncio
import aioredis
import asyncpg
from datetime import datetime

async def comprehensive_health_check() -> Dict[str, Any]:
    """Comprehensive system health check."""
    checks = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "components": {}
    }
    
    # Database health
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute("SELECT 1")
        await conn.close()
        checks["components"]["database"] = {
            "status": "healthy",
            "response_time_ms": 10
        }
    except Exception as e:
        checks["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        checks["overall_status"] = "degraded"
    
    # Redis health
    try:
        redis = aioredis.from_url(REDIS_URL)
        await redis.ping()
        await redis.close()
        checks["components"]["redis"] = {
            "status": "healthy",
            "response_time_ms": 5
        }
    except Exception as e:
        checks["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        checks["overall_status"] = "degraded"
    
    # AI API health
    try:
        # Test Claude API connectivity
        response = await test_claude_api()
        checks["components"]["claude_api"] = {
            "status": "healthy",
            "response_time_ms": response.get("response_time", 0)
        }
    except Exception as e:
        checks["components"]["claude_api"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        checks["overall_status"] = "degraded"
    
    return checks
```

---

## Security Hardening

### SSL/TLS Configuration
```nginx
# nginx/ssl.conf
server {
    listen 443 ssl http2;
    server_name claude-tiu.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/claude-tiu.crt;
    ssl_certificate_key /etc/ssl/private/claude-tiu.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://claude-tiu-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout configuration
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer configuration
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name claude-tiu.your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### Firewall Configuration
```bash
#!/bin/bash
# firewall-setup.sh

# UFW (Ubuntu)
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP
ufw allow 443/tcp     # HTTPS
ufw allow 5432/tcp    # PostgreSQL (internal only)
ufw allow 6379/tcp    # Redis (internal only)
ufw --force enable

# Fail2ban configuration
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[claude-tiu]
enabled = true
port = http,https
filter = claude-tiu
logpath = /app/logs/access.log
maxretry = 10
EOF

# Create Claude-TIU filter
cat > /etc/fail2ban/filter.d/claude-tiu.conf << EOF
[Definition]
failregex = ^.*"(POST|GET|PUT|DELETE).*" 4[0-9]{2} .*$
ignoreregex =
EOF

systemctl restart fail2ban
```

### Secret Management
```bash
#!/bin/bash
# secrets-setup.sh

# Using HashiCorp Vault
vault kv put secret/claude-tiu/production \
    claude_api_key="sk-your-secure-key" \
    secret_key="your-super-secret-key" \
    jwt_secret="your-jwt-secret" \
    db_password="your-db-password"

# Using AWS Secrets Manager
aws secretsmanager create-secret \
    --name "claude-tiu/production" \
    --description "Claude-TIU production secrets" \
    --secret-string '{
        "claude_api_key": "sk-your-secure-key",
        "secret_key": "your-super-secret-key",
        "jwt_secret": "your-jwt-secret",
        "db_password": "your-db-password"
    }'

# Using Kubernetes Secrets
kubectl create secret generic claude-tiu-secrets \
    --from-literal=claude-api-key="sk-your-secure-key" \
    --from-literal=secret-key="your-super-secret-key" \
    --from-literal=jwt-secret="your-jwt-secret" \
    --from-literal=db-password="your-db-password" \
    --namespace=claude-tiu
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Installation Issues
```bash
# Permission denied
sudo chown -R $USER:$USER ~/.local/bin/claude-tiu
chmod +x ~/.local/bin/claude-tiu

# Python version conflicts
pyenv install 3.11.0
pyenv global 3.11.0
pip install --upgrade pip

# Missing dependencies
sudo apt-get update
sudo apt-get install -y python3-dev build-essential
```

#### 2. API Connection Issues
```bash
# Test Claude API connectivity
curl -H "Authorization: Bearer $CLAUDE_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello"}' \
     https://api.anthropic.com/v1/messages

# Check network connectivity
ping api.anthropic.com
nslookup api.anthropic.com

# Verify SSL certificates
openssl s_client -connect api.anthropic.com:443 -servername api.anthropic.com
```

#### 3. Database Connection Issues
```bash
# Test PostgreSQL connection
psql $DATABASE_URL -c "SELECT version();"

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Reset database
claude-tiu db reset --confirm
claude-tiu db init
```

#### 4. Performance Issues
```bash
# Check system resources
htop
free -h
df -h

# Monitor Claude-TIU processes
ps aux | grep claude-tiu

# Check application logs
tail -f /app/logs/claude-tiu.log

# Clear caches
claude-tiu cache clear
redis-cli flushall
```

#### 5. Docker Issues
```bash
# Check container status
docker ps -a
docker logs claude-tiu

# Restart containers
docker-compose restart

# Clean up Docker
docker system prune -a
```

#### 6. Kubernetes Issues
```bash
# Check pod status
kubectl get pods -n claude-tiu
kubectl describe pod <pod-name> -n claude-tiu

# Check logs
kubectl logs -f deployment/claude-tiu -n claude-tiu

# Debug networking
kubectl exec -it <pod-name> -n claude-tiu -- curl localhost:8000/health
```

### Performance Tuning

#### Application Tuning
```python
# performance_config.py
PERFORMANCE_CONFIG = {
    # Worker configuration
    "workers": 4,  # Number of worker processes
    "worker_connections": 1000,
    "worker_class": "uvicorn.workers.UvicornWorker",
    
    # Memory optimization
    "max_memory_per_worker": "512MB",
    "gc_threshold": (700, 10, 10),
    
    # Connection pooling
    "db_pool_size": 20,
    "db_max_overflow": 10,
    "redis_pool_size": 50,
    
    # Caching
    "cache_ttl": 3600,
    "response_cache_size": "100MB",
    
    # AI optimization
    "ai_request_timeout": 30,
    "ai_max_concurrent": 10,
    "ai_retry_attempts": 3,
}
```

#### Database Tuning
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Create indexes
CREATE INDEX CONCURRENTLY idx_projects_user_id ON projects(user_id);
CREATE INDEX CONCURRENTLY idx_tasks_status ON tasks(status);
CREATE INDEX CONCURRENTLY idx_tasks_created_at ON tasks(created_at DESC);

-- Analyze tables
ANALYZE projects;
ANALYZE tasks;
ANALYZE users;
```

#### Redis Tuning
```redis
# Redis optimization
maxmemory 1gb
maxmemory-policy allkeys-lru
timeout 300
tcp-keepalive 300
save 900 1
save 300 10
save 60 10000
```

### Backup and Recovery

#### Database Backup
```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups/claude-tiu"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/claude_tiu_$TIMESTAMP.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump $DATABASE_URL > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3 (optional)
aws s3 cp $BACKUP_FILE.gz s3://your-backup-bucket/claude-tiu/

# Keep only last 30 days of backups
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

#### Application Backup
```bash
#!/bin/bash
# backup-application.sh

BACKUP_DIR="/backups/claude-tiu"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup projects directory
tar -czf $BACKUP_DIR/projects_$TIMESTAMP.tar.gz /app/projects/

# Backup configuration
tar -czf $BACKUP_DIR/config_$TIMESTAMP.tar.gz /app/config/

# Backup logs (last 7 days)
find /app/logs -name "*.log" -mtime -7 | tar -czf $BACKUP_DIR/logs_$TIMESTAMP.tar.gz -T -

echo "Application backup completed"
```

#### Recovery Procedures
```bash
#!/bin/bash
# recovery.sh

# Restore database
gunzip -c claude_tiu_20240115_120000.sql.gz | psql $DATABASE_URL

# Restore projects
tar -xzf projects_20240115_120000.tar.gz -C /

# Restore configuration
tar -xzf config_20240115_120000.tar.gz -C /

# Restart services
docker-compose restart
# or
kubectl rollout restart deployment/claude-tiu -n claude-tiu

echo "Recovery completed"
```

---

## Conclusion

This comprehensive installation and deployment guide provides everything needed to run Claude-TIU from local development to enterprise-scale production deployments. Choose the deployment method that best fits your requirements:

- **Local Development**: Use the quick installation or source setup
- **Small Teams**: Docker Compose deployment  
- **Production**: Kubernetes with proper monitoring and security
- **Enterprise**: Multi-region with high availability and comprehensive monitoring

For additional support:
- 📚 [Full Documentation](https://docs.claude-tiu.dev)
- 💬 [Community Forum](https://community.claude-tiu.dev)
- 📧 [Enterprise Support](mailto:enterprise@claude-tiu.dev)

---

**Ready to deploy Claude-TIU at scale! 🚀**