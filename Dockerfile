# Multi-stage Docker build for Claude TUI - Production-ready with security hardening
# Stage 1: Builder - Install and compile dependencies
FROM python:3.11-slim as builder

# Security: Update packages and install minimal build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Node.js LTS for Claude Flow
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g npm@latest

WORKDIR /build

# Copy dependency files first for better layer caching
COPY pyproject.toml setup.py requirements*.txt ./

# Build Python wheels for better performance and security
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt \
    && if [ -f requirements-dev.txt ]; then pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements-dev.txt; fi

# Stage 2: Production - Secure runtime image
FROM python:3.11-slim as production

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    nodejs \
    npm \
    curl \
    ca-certificates \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user with restricted permissions
RUN groupadd -r -g 1000 claude && \
    useradd -r -u 1000 -g claude -d /home/claude -m -s /bin/bash claude && \
    mkdir -p /app /home/claude/.claude-tui /app/data /app/logs /app/config && \
    chown -R claude:claude /app /home/claude

WORKDIR /app

# Copy and install Python wheels from builder stage
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-index --find-links=/tmp/wheels /tmp/wheels/* \
    && rm -rf /tmp/wheels /root/.cache/pip

# Install Claude Flow with specific version for consistency
RUN npm install -g claude-flow@alpha --production \
    && npm cache clean --force

# Copy application source code
COPY --chown=claude:claude src/ ./src/
COPY --chown=claude:claude pyproject.toml setup.py ./
COPY --chown=claude:claude config/ ./config/

# Install the application in production mode
RUN pip install --no-deps --no-cache-dir -e .

# Switch to non-root user for security
USER claude

# Create application directories with proper permissions
RUN mkdir -p .swarm logs data backups coordination memory && \
    chmod 755 .swarm logs data backups coordination memory

# Comprehensive health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import claude_tui; print('Health check passed')" || exit 1

# Environment variables for production
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLAUDE_TUI_ENV=production \
    CLAUDE_TUI_CONFIG_DIR=/app/config \
    CLAUDE_TUI_DATA_DIR=/app/data \
    CLAUDE_TUI_LOG_DIR=/app/logs

# Expose application port
EXPOSE 8000

# Use dumb-init for proper signal handling
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "-m", "claude_tui.main"]

# Stage 3: Development - Extended image with dev tools
FROM production as development

# Switch back to root for installing dev dependencies
USER root

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    net-tools \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Switch back to claude user
USER claude

# Development environment variables
ENV CLAUDE_TUI_ENV=development \
    PYTHONDONTWRITEBYTECODE=0

# Development command with hot reload
CMD ["python", "-m", "claude_tui.main", "--dev", "--reload"]

# Stage 4: Testing - Image for running tests in CI
FROM development as testing

USER root

# Copy all source code and tests
COPY --chown=claude:claude . .

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio pytest-mock

USER claude

# Test environment variables
ENV CLAUDE_TUI_ENV=testing \
    PYTHONPATH=/app/src:/app/tests

# Default test command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=claude_tui"]