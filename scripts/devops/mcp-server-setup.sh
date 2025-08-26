#!/bin/bash
# MCP Server Infrastructure Setup for Claude-TUI
# Infrastructure Engineer Implementation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MCP_DIR="${PROJECT_ROOT}/mcp-server"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        ERROR)
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            ;;
        WARN)
            echo -e "${YELLOW}[$timestamp] WARN: $message${NC}"
            ;;
        INFO)
            echo -e "${GREEN}[$timestamp] INFO: $message${NC}"
            ;;
        DEBUG)
            echo -e "${BLUE}[$timestamp] DEBUG: $message${NC}"
            ;;
    esac
}

# Infrastructure Engineer: MCP Server Setup
setup_mcp_server_infrastructure() {
    log INFO "Setting up MCP server infrastructure..."
    
    # Create MCP server directory structure
    mkdir -p "${MCP_DIR}"/{
        src,
        config,
        data/{persistent,cache,logs},
        security/{certs,secrets},
        monitoring,
        backup
    }
    
    # Create MCP server Dockerfile
    cat > "${MCP_DIR}/Dockerfile" << 'EOF'
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS runtime

# Security: Create non-root user
RUN addgroup -g 1001 -S mcpuser && \
    adduser -S -D -H -u 1001 -s /sbin/nologin -G mcpuser mcpuser

# Install security updates
RUN apk update && apk upgrade && \
    apk add --no-cache tini curl && \
    rm -rf /var/cache/apk/*

WORKDIR /app

# Copy application files
COPY --from=builder /app/node_modules ./node_modules
COPY --chown=mcpuser:mcpuser src ./src
COPY --chown=mcpuser:mcpuser config ./config
COPY --chown=mcpuser:mcpuser package*.json ./

# Create data directories with proper permissions
RUN mkdir -p data/{persistent,cache,logs} && \
    chown -R mcpuser:mcpuser data

# Security hardening
USER mcpuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["node", "src/server.js"]

EXPOSE 3000
EOF

    # Create MCP server package.json
    cat > "${MCP_DIR}/package.json" << 'EOF'
{
  "name": "claude-tui-mcp-server",
  "version": "1.0.0",
  "description": "MCP Server for Claude-TUI Integration",
  "main": "src/server.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js",
    "test": "jest",
    "health-check": "curl -f http://localhost:3000/health"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "compression": "^1.7.4",
    "winston": "^3.10.0",
    "prom-client": "^14.2.0",
    "dotenv": "^16.3.1",
    "joi": "^17.9.2",
    "redis": "^4.6.7",
    "pg": "^8.11.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.6.2",
    "supertest": "^6.3.3"
  },
  "engines": {
    "node": ">=18.0.0"
  },
  "keywords": ["mcp", "claude", "ai", "integration"]
}
EOF

    log INFO "MCP server infrastructure setup completed"
}

# Main execution
main() {
    log INFO "Starting MCP Server Infrastructure setup..."
    
    setup_mcp_server_infrastructure
    
    log INFO "MCP Server Infrastructure setup completed!"
    
    echo
    echo "🎉 MCP Server Infrastructure is ready!"
    echo
    echo "Next steps:"
    echo "1. Update API keys in config/.env.production"
    echo "2. Build Docker image: docker build -t claude-tui-mcp mcp-server/"
    echo "3. Deploy with: docker-compose up -d"
    echo
}

# Run main function
main "$@"