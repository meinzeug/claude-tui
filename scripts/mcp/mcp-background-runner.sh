#!/bin/bash

# MCP Background Runner - Enhanced for Production
# Handles proper backgrounding, monitoring, and process management

set -euo pipefail

# Configuration
MCP_LOG_DIR="/home/tekkadmin/claude-tui/logs/mcp"
MCP_PID_FILE="/home/tekkadmin/claude-tui/scripts/mcp/mcp-server.pid"
MCP_LOCK_FILE="/home/tekkadmin/claude-tui/scripts/mcp/mcp-server.lock"
NODE_PATH="/home/tekkadmin/.npm/_npx/7cfa166e65244432/node_modules"
MCP_SERVER_JS="$NODE_PATH/claude-flow/src/mcp/mcp-server.js"
RESTART_INTERVAL=5
MAX_RESTART_ATTEMPTS=10
MEMORY_LIMIT_MB=512

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MCP_LOG_DIR/runner.log"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$MCP_LOG_DIR/runner.log" >&2
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$MCP_LOG_DIR/runner.log"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$MCP_LOG_DIR/runner.log"
}

# Initialize directories
init_directories() {
    mkdir -p "$MCP_LOG_DIR"
    mkdir -p "$(dirname "$MCP_PID_FILE")"
    
    # Set up log rotation
    if [ ! -f "$MCP_LOG_DIR/logrotate.conf" ]; then
        cat > "$MCP_LOG_DIR/logrotate.conf" << EOF
$MCP_LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
    fi
}

# Check if MCP server is running
is_mcp_running() {
    if [ ! -f "$MCP_PID_FILE" ]; then
        return 1
    fi
    
    local pid=$(cat "$MCP_PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        # Additional check: is it actually the MCP server?
        if ps -p "$pid" -o cmd --no-headers | grep -q "mcp-server.js"; then
            return 0
        else
            # PID exists but not MCP server, clean up
            rm -f "$MCP_PID_FILE"
            return 1
        fi
    else
        # PID doesn't exist, clean up
        rm -f "$MCP_PID_FILE"
        return 1
    fi
}

# Get MCP server PID
get_mcp_pid() {
    if [ -f "$MCP_PID_FILE" ]; then
        cat "$MCP_PID_FILE"
    else
        echo ""
    fi
}

# Memory optimization for Node.js
optimize_node_memory() {
    # Set Node.js memory options for better performance
    export NODE_OPTIONS="--max-old-space-size=$MEMORY_LIMIT_MB --optimize-for-size --gc-interval=100"
    export UV_THREADPOOL_SIZE=4
    export NODE_ENV=production
}

# Start MCP server with enhanced error handling
start_mcp_server() {
    log "Starting MCP server..."
    
    # Check if already running
    if is_mcp_running; then
        local pid=$(get_mcp_pid)
        warning "MCP server already running with PID $pid"
        return 0
    fi
    
    # Acquire lock
    if ! (
        flock -n 200
        
        # Double-check after acquiring lock
        if is_mcp_running; then
            log "MCP server started by another process"
            return 0
        fi
        
        # Optimize memory settings
        optimize_node_memory
        
        # Start server with proper backgrounding
        cd "$(dirname "$MCP_SERVER_JS")"
        
        nohup node "$MCP_SERVER_JS" \
            > "$MCP_LOG_DIR/mcp-server-stdout.log" \
            2> "$MCP_LOG_DIR/mcp-server-stderr.log" &
        
        local pid=$!
        echo $pid > "$MCP_PID_FILE"
        
        # Wait a moment and verify it started
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            success "MCP server started successfully with PID $pid"
            
            # Store startup info
            echo "$(date '+%Y-%m-%d %H:%M:%S'): Started with PID $pid" >> "$MCP_LOG_DIR/startup.log"
            
            return 0
        else
            error "MCP server failed to start"
            rm -f "$MCP_PID_FILE"
            return 1
        fi
        
    ) 200> "$MCP_LOCK_FILE"; then
        error "Could not acquire lock to start MCP server"
        return 1
    fi
}

# Stop MCP server gracefully
stop_mcp_server() {
    log "Stopping MCP server..."
    
    if ! is_mcp_running; then
        warning "MCP server is not running"
        return 0
    fi
    
    local pid=$(get_mcp_pid)
    log "Attempting graceful shutdown of PID $pid"
    
    # Try graceful shutdown first
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            success "MCP server stopped gracefully"
            rm -f "$MCP_PID_FILE"
            return 0
        fi
        sleep 1
    done
    
    # Force kill if necessary
    warning "Graceful shutdown failed, force killing..."
    kill -KILL "$pid" 2>/dev/null || true
    
    # Final verification
    for i in {1..5}; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            success "MCP server force killed"
            rm -f "$MCP_PID_FILE"
            return 0
        fi
        sleep 1
    done
    
    error "Failed to stop MCP server"
    return 1
}

# Restart MCP server
restart_mcp_server() {
    log "Restarting MCP server..."
    stop_mcp_server
    sleep 2
    start_mcp_server
}

# Monitor and auto-restart with backoff
monitor_mcp_server() {
    log "Starting MCP server monitoring..."
    local restart_count=0
    local last_restart=0
    
    while true; do
        if ! is_mcp_running; then
            local current_time=$(date +%s)
            
            # Implement exponential backoff for restarts
            if [ $restart_count -gt 0 ] && [ $((current_time - last_restart)) -lt $((RESTART_INTERVAL * restart_count)) ]; then
                log "Waiting for backoff period..."
                sleep $((RESTART_INTERVAL * restart_count))
            fi
            
            if [ $restart_count -ge $MAX_RESTART_ATTEMPTS ]; then
                error "Maximum restart attempts ($MAX_RESTART_ATTEMPTS) reached. Stopping monitor."
                exit 1
            fi
            
            warning "MCP server not running. Attempting restart #$((restart_count + 1))"
            
            if start_mcp_server; then
                restart_count=0
                success "MCP server restarted successfully"
            else
                restart_count=$((restart_count + 1))
                last_restart=$(date +%s)
                error "Failed to restart MCP server (attempt $restart_count/$MAX_RESTART_ATTEMPTS)"
            fi
        else
            # Server is running, check memory usage
            local pid=$(get_mcp_pid)
            local memory_mb=$(ps -p "$pid" -o rss --no-headers 2>/dev/null | awk '{print int($1/1024)}')
            
            if [ ! -z "$memory_mb" ] && [ "$memory_mb" -gt $((MEMORY_LIMIT_MB * 2)) ]; then
                warning "MCP server memory usage high: ${memory_mb}MB (limit: ${MEMORY_LIMIT_MB}MB)"
                log "Restarting server to prevent memory issues..."
                restart_mcp_server
            fi
        fi
        
        sleep "$RESTART_INTERVAL"
    done
}

# Get status
status() {
    if is_mcp_running; then
        local pid=$(get_mcp_pid)
        local memory_mb=$(ps -p "$pid" -o rss --no-headers 2>/dev/null | awk '{print int($1/1024)}')
        local cpu_percent=$(ps -p "$pid" -o %cpu --no-headers 2>/dev/null | awk '{print $1}')
        
        success "MCP server is running"
        echo "  PID: $pid"
        echo "  Memory: ${memory_mb}MB"
        echo "  CPU: ${cpu_percent}%"
        echo "  Log: $MCP_LOG_DIR/mcp-server-stdout.log"
        
        # Show recent activity
        echo -e "\n${BLUE}Recent activity:${NC}"
        tail -n 5 "$MCP_LOG_DIR/mcp-server-stdout.log" 2>/dev/null || echo "No recent logs"
    else
        error "MCP server is not running"
        return 1
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    rm -f "$MCP_LOCK_FILE"
}

# Signal handlers
trap cleanup EXIT
trap 'log "Received SIGTERM, shutting down..."; stop_mcp_server; exit 0' TERM
trap 'log "Received SIGINT, shutting down..."; stop_mcp_server; exit 0' INT

# Main execution
main() {
    init_directories
    
    case "${1:-}" in
        "start")
            start_mcp_server
            ;;
        "stop")
            stop_mcp_server
            ;;
        "restart")
            restart_mcp_server
            ;;
        "status")
            status
            ;;
        "monitor")
            monitor_mcp_server
            ;;
        "logs")
            echo -e "${BLUE}MCP Server Logs:${NC}"
            echo "STDOUT: $MCP_LOG_DIR/mcp-server-stdout.log"
            echo "STDERR: $MCP_LOG_DIR/mcp-server-stderr.log"
            echo "RUNNER: $MCP_LOG_DIR/runner.log"
            echo ""
            tail -f "$MCP_LOG_DIR/mcp-server-stdout.log"
            ;;
        "health")
            # Run health check
            "$0" status && \
            node "$(dirname "$0")/health-check.js" && \
            success "Health check passed"
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|monitor|logs|health}"
            echo ""
            echo "Commands:"
            echo "  start   - Start MCP server"
            echo "  stop    - Stop MCP server"
            echo "  restart - Restart MCP server"
            echo "  status  - Show server status"
            echo "  monitor - Start monitoring with auto-restart"
            echo "  logs    - Show and follow logs"
            echo "  health  - Run health check"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"