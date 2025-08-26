# Installation Guide

This guide covers all installation methods for Claude-TUI, from simple pip install to advanced Docker and Kubernetes deployments.

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.11+ (3.12 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Terminal**: Support for 256+ colors

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS, macOS 13+, or Windows 11
- **Python**: 3.12+
- **RAM**: 16GB for optimal performance
- **Storage**: 5GB free space (for caches and projects)
- **Terminal**: Modern terminal with emoji support

### Dependencies
- **Git**: Version control operations
- **Node.js**: 18+ for Claude Flow integration
- **Docker**: Optional, for containerized deployment

## 🚀 Installation Methods

### Method 1: PyPI Installation (Recommended)

The simplest way to install Claude-TUI:

```bash
# Install from PyPI
pip install claude-tui

# Verify installation
claude-tui --version
claude-tui health-check
```

### Method 2: Development Installation

For contributors and advanced users:

```bash
# Clone the repository
git clone https://github.com/your-org/claude-tui.git
cd claude-tui

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests to verify installation
pytest
```

### Method 3: Docker Installation

For isolated and consistent environments:

```bash
# Pull the latest image
docker pull claude-tui:latest

# Run interactively
docker run -it \
  -v $(pwd)/projects:/app/projects \
  -v ~/.claude-tui:/app/.claude-tui \
  -e CLAUDE_API_KEY=your-key-here \
  claude-tui:latest

# Create an alias for easier use
echo 'alias claude-tui="docker run -it -v $(pwd):/workspace claude-tui:latest"' >> ~/.bashrc
source ~/.bashrc
```

### Method 4: Build from Source

For the latest features and custom builds:

```bash
# Clone and navigate
git clone https://github.com/your-org/claude-tui.git
cd claude-tui

# Install build tools
pip install build wheel

# Build the package
python -m build

# Install the built package
pip install dist/claude_tui-*.whl
```

## 🔧 Configuration

### Initial Setup

After installation, run the setup wizard:

```bash
claude-tui configure
```

This interactive setup will guide you through:

1. **API Key Configuration**
   - Claude API key from Anthropic
   - Optional: OpenAI, GitHub, etc.

2. **Directory Setup**
   - Workspace directory
   - Template directory
   - Cache location

3. **Performance Settings**
   - Memory limits
   - Concurrent agents
   - Cache sizes

### Manual Configuration

#### Environment Variables

Create a `.env` file or set system environment variables:

```bash
# Core Configuration
export CLAUDE_API_KEY="your-claude-api-key-here"
export CLAUDE_TUI_WORKSPACE_PATH="$HOME/claude-projects"
export CLAUDE_TUI_LOG_LEVEL="INFO"

# Performance Tuning
export CLAUDE_TUI_MAX_AGENTS="10"
export CLAUDE_TUI_MEMORY_LIMIT="2048"
export CLAUDE_TUI_CACHE_SIZE="1000"

# Feature Flags
export CLAUDE_TUI_ENABLE_TELEMETRY="true"
export CLAUDE_TUI_AUTO_UPDATE="true"
```

#### Configuration File

Create `~/.claude-tui/config.yaml`:

```yaml
# Core API Configuration
api:
  claude:
    key: "${CLAUDE_API_KEY}"
    model: "claude-3-sonnet-20241022"
    timeout: 60
    max_retries: 3
  
  openai:  # Optional
    key: "${OPENAI_API_KEY}"
    model: "gpt-4"
  
  github:  # Optional
    token: "${GITHUB_TOKEN}"

# AI Agent Configuration
agents:
  max_concurrent: 5
  memory_per_agent: "512MB"
  timeout: 300
  auto_spawn: true
  
  # Agent-specific settings
  backend_dev:
    preferred_languages: ["python", "javascript", "go"]
    frameworks: ["fastapi", "express", "gin"]
  
  frontend_dev:
    preferred_frameworks: ["react", "vue", "svelte"]
    build_tools: ["vite", "webpack", "rollup"]

# Anti-Hallucination Engine
validation:
  enabled: true
  precision_threshold: 0.95
  auto_fix: true
  deep_scan: true
  cross_validate: true

# User Interface
ui:
  theme: "dark"  # "dark", "light", "auto"
  animations: true
  show_metrics: true
  auto_save: true
  
  # Keyboard shortcuts
  shortcuts:
    new_project: "ctrl+n"
    command_palette: "ctrl+p"
    quick_search: "ctrl+f"

# Performance Optimization
performance:
  cache:
    enabled: true
    size_limit: "1GB"
    ttl_default: 3600  # 1 hour
    
  memory:
    gc_threshold: 0.8
    max_usage: "4GB"
    
  network:
    connection_pool_size: 20
    request_timeout: 30

# Project Settings
projects:
  default_template: "python-package"
  auto_git_init: true
  auto_install_deps: true
  
  templates_path: "~/.claude-tui/templates"
  workspace_path: "~/claude-projects"

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "~/.claude-tui/logs/claude-tui.log"
  rotation: "1 week"
  max_size: "100MB"
  
# Security
security:
  encrypt_api_keys: true
  sandbox_code_execution: true
  network_restrictions: true
  
# Updates
updates:
  auto_check: true
  auto_install_patches: true
  channel: "stable"  # "stable", "beta", "nightly"
```

## 🐳 Docker Deployment

### Simple Docker Setup

```dockerfile
# Dockerfile.simple
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8080
CMD ["claude-tui", "serve"]
```

### Production Docker Setup

```dockerfile
# Dockerfile.production
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git curl nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Production stage
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r claudetui && useradd -r -g claudetui claudetui

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git curl nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy application
COPY . .
RUN pip install -e .

# Set up directories
RUN mkdir -p /app/projects /app/logs /app/cache && \
    chown -R claudetui:claudetui /app

USER claudetui

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD claude-tui health-check || exit 1

EXPOSE 8080
CMD ["claude-tui", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  claude-tui:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./projects:/app/projects
      - ./config:/app/config
      - logs:/app/logs
    environment:
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/claudetui
    depends_on:
      - redis
      - postgres
    networks:
      - claude-network

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    networks:
      - claude-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=claudetui
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - claude-network

volumes:
  redis_data:
  postgres_data:
  logs:

networks:
  claude-network:
    driver: bridge
```

## ☸️ Kubernetes Deployment

### Basic Kubernetes Manifests

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: claude-tui
  labels:
    name: claude-tui

---
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claude-tui-config
  namespace: claude-tui
data:
  config.yaml: |
    api:
      claude:
        model: "claude-3-sonnet-20241022"
        timeout: 60
    agents:
      max_concurrent: 5
    validation:
      precision_threshold: 0.95

---
# kubernetes/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: claude-tui-secrets
  namespace: claude-tui
type: Opaque
data:
  claude-api-key: base64-encoded-key-here
  postgres-password: base64-encoded-password-here

---
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-tui
  namespace: claude-tui
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-tui
  template:
    metadata:
      labels:
        app: claude-tui
    spec:
      containers:
      - name: claude-tui
        image: claude-tui:latest
        ports:
        - containerPort: 8080
        env:
        - name: CLAUDE_API_KEY
          valueFrom:
            secretKeyRef:
              name: claude-tui-secrets
              key: claude-api-key
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: projects
          mountPath: /app/projects
        resources:
          limits:
            cpu: 1000m
            memory: 2Gi
          requests:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: claude-tui-config
      - name: projects
        persistentVolumeClaim:
          claimName: claude-tui-projects

---
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: claude-tui-service
  namespace: claude-tui
spec:
  selector:
    app: claude-tui
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: claude-tui-ingress
  namespace: claude-tui
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - claude-tui.yourdomain.com
    secretName: claude-tui-tls
  rules:
  - host: claude-tui.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: claude-tui-service
            port:
              number: 80
```

### Helm Chart

```yaml
# helm/claude-tui/Chart.yaml
apiVersion: v2
name: claude-tui
description: A Helm chart for Claude-TUI
type: application
version: 1.0.0
appVersion: "1.0.0"

# helm/claude-tui/values.yaml
replicaCount: 3

image:
  repository: claude-tui
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: claude-tui.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: claude-tui-tls
      hosts:
        - claude-tui.yourdomain.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

config:
  api:
    claude:
      model: "claude-3-sonnet-20241022"
  agents:
    max_concurrent: 5

secrets:
  claude_api_key: "your-api-key-here"
```

## 🔍 Verification

### Installation Verification

```bash
# Check version
claude-tui --version

# Run health check
claude-tui health-check

# Verify configuration
claude-tui config show

# Test AI connectivity
claude-tui test-ai

# Run diagnostic
claude-tui diagnose
```

### Performance Testing

```bash
# Run performance benchmarks
claude-tui benchmark

# Memory usage test
claude-tui memory-test

# Load testing (if running as server)
claude-tui load-test --concurrent-users 10
```

## 🛠️ Troubleshooting

### Common Issues

#### Issue: "claude-tui command not found"
```bash
# Solution: Ensure PATH includes pip install location
pip show -f claude-tui
export PATH="$PATH:$(python -m site --user-base)/bin"
```

#### Issue: "API key not configured"
```bash
# Solution: Set up API key
claude-tui configure
# or
export CLAUDE_API_KEY="your-key-here"
```

#### Issue: "Permission denied"
```bash
# Solution: Fix permissions
chmod +x $(which claude-tui)
# or for Docker
docker run --user $(id -u):$(id -g) ...
```

### Performance Issues

#### High Memory Usage
```bash
# Reduce agent limit
export CLAUDE_TUI_MAX_AGENTS=3

# Clear cache
claude-tui cache clear

# Optimize configuration
claude-tui optimize
```

#### Slow Response Times
```bash
# Check network connectivity
claude-tui test-connectivity

# Update to latest version
pip install --upgrade claude-tui

# Use local caching
claude-tui cache enable
```

## 🔄 Updates

### Automatic Updates
```bash
# Enable auto-updates
claude-tui config set updates.auto_check true

# Check for updates
claude-tui update check

# Install updates
claude-tui update install
```

### Manual Updates
```bash
# Update via pip
pip install --upgrade claude-tui

# Update Docker image
docker pull claude-tui:latest

# Update from source
git pull origin main
pip install -e .
```

## 🎉 Next Steps

After successful installation:

1. **[Configure API Keys](getting-started.md#initial-setup)**
2. **[Create Your First Project](getting-started.md#your-first-project)**
3. **[Explore Features](../user-guide.md)**
4. **[Join the Community](../contributing.md)**

---

*You're now ready to experience the future of AI-assisted development!* 🚀