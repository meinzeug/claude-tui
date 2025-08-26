#!/bin/bash

# Auto-Restart Script for MCP Server - Production Resilience
# Implements intelligent restart strategies, failure analysis, and recovery

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_RUNNER="$SCRIPT_DIR/mcp-background-runner.sh"
HEALTH_CHECKER="$SCRIPT_DIR/health-check.js"
LOG_DIR="/home/tekkadmin/claude-tui/logs/mcp"
RESTART_LOG="$LOG_DIR/auto-restart.log"
FAILURE_LOG="$LOG_DIR/failure-analysis.log"
RECOVERY_STATE_FILE="$LOG_DIR/recovery-state.json"

# Restart strategy configuration
MAX_RAPID_RESTARTS=3
RAPID_RESTART_WINDOW=300  # 5 minutes
EXPONENTIAL_BACKOFF_BASE=30
MAX_BACKOFF_SECONDS=3600  # 1 hour
HEALTH_CHECK_INTERVAL=60
FAILURE_ANALYSIS_ENABLED=true
AUTO_RECOVERY_ENABLED=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local color=""
    
    case "$level" in
        "INFO") color="$BLUE" ;;
        "SUCCESS") color="$GREEN" ;;
        "WARNING") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
        "CRITICAL") color="$PURPLE" ;;
        *) color="$NC" ;;
    esac
    
    echo -e "${color}[$timestamp] $level: $message${NC}" | tee -a "$RESTART_LOG"
}

# Initialize directories and state
init_recovery_system() {
    mkdir -p "$LOG_DIR"
    
    if [ ! -f "$RECOVERY_STATE_FILE" ]; then
        cat > "$RECOVERY_STATE_FILE" << EOF
{
    "startTime": $(date +%s),
    "totalRestarts": 0,
    "rapidRestarts": 0,
    "lastRestartTime": 0,
    "currentBackoffLevel": 0,
    "consecutiveFailures": 0,
    "lastHealthyTime": $(date +%s),
    "recoveryStrategies": {
        "memoryOptimization": false,
        "processCleanup": false,
        "dependencyCheck": false,
        "systemReboot": false
    }
}
EOF
    fi
    
    log "INFO" "Auto-restart system initialized"
}

# Load recovery state
load_recovery_state() {
    if [ -f "$RECOVERY_STATE_FILE" ]; then
        cat "$RECOVERY_STATE_FILE"
    else
        echo "{}"
    fi
}

# Save recovery state
save_recovery_state() {
    local state="$1"
    echo "$state" > "$RECOVERY_STATE_FILE"
}

# Update recovery state field
update_recovery_state() {
    local field="$1"
    local value="$2"
    local current_state=$(load_recovery_state)
    local updated_state=$(echo "$current_state" | jq ".$field = $value")
    save_recovery_state "$updated_state"
}

# Calculate exponential backoff delay
calculate_backoff() {
    local failure_count="$1"
    local backoff_seconds=$((EXPONENTIAL_BACKOFF_BASE * (2 ** failure_count)))
    if [ $backoff_seconds -gt $MAX_BACKOFF_SECONDS ]; then
        backoff_seconds=$MAX_BACKOFF_SECONDS
    fi
    echo $backoff_seconds
}

# Check if we're in rapid restart scenario
is_rapid_restart() {
    local state=$(load_recovery_state)
    local current_time=$(date +%s)
    local last_restart=$(echo "$state" | jq -r '.lastRestartTime // 0')
    local rapid_restarts=$(echo "$state" | jq -r '.rapidRestarts // 0')
    
    # Reset rapid restart counter if outside window
    if [ $((current_time - last_restart)) -gt $RAPID_RESTART_WINDOW ]; then
        update_recovery_state "rapidRestarts" 0
        rapid_restarts=0
    fi
    
    [ $rapid_restarts -ge $MAX_RAPID_RESTARTS ]
}

# Perform failure analysis
analyze_failure() {
    if [ "$FAILURE_ANALYSIS_ENABLED" != "true" ]; then
        return
    fi
    
    log "INFO" "Performing failure analysis..."
    
    local analysis_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local analysis_report="$LOG_DIR/failure-analysis-$(date +%s).json"
    
    # Collect system information
    local system_info=$(cat << EOF
{
    "timestamp": "$analysis_timestamp",
    "systemLoad": "$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 $2 $3 }')",
    "memoryUsage": $(free -m | awk 'NR==2{printf "%.2f", $3*100/$2 }'),
    "diskUsage": $(df -h / | awk 'NR==2{print $5}' | sed 's/%//'),
    "processCount": $(ps aux | wc -l),
    "nodeProcesses": $(pgrep -f node | wc -l)
}
EOF
)
    
    # Check for common failure patterns
    local failure_patterns=""
    
    # Memory exhaustion
    if [ "$(free -m | awk 'NR==2{print ($3*100/$2 > 90)}')" == "1" ]; then
        failure_patterns="${failure_patterns}\"memory_exhaustion\","
    fi
    
    # Too many Node.js processes
    if [ "$(pgrep -f node | wc -l)" -gt 10 ]; then
        failure_patterns="${failure_patterns}\"excessive_node_processes\","
    fi
    
    # Disk space issues
    if [ "$(df / | awk 'NR==2 {print $5}' | sed 's/%//')" -gt 90 ]; then
        failure_patterns="${failure_patterns}\"disk_space_low\","
    fi
    
    # Remove trailing comma
    failure_patterns=$(echo "$failure_patterns" | sed 's/,$//')
    
    # Generate analysis report
    cat > "$analysis_report" << EOF
{
    "timestamp": "$analysis_timestamp",
    "systemInfo": $system_info,
    "failurePatterns": [$failure_patterns],
    "logSnippets": {
        "mcpStdout": "$(tail -n 20 "$LOG_DIR/mcp-server-stdout.log" 2>/dev/null | jq -Rs . || echo '""')",
        "mcpStderr": "$(tail -n 20 "$LOG_DIR/mcp-server-stderr.log" 2>/dev/null | jq -Rs . || echo '""')"
    },
    "recommendedActions": $(generate_recovery_recommendations "$failure_patterns")
}
EOF
    
    log "INFO" "Failure analysis completed: $analysis_report"
    
    # Log critical patterns
    if echo "$failure_patterns" | grep -q "memory_exhaustion"; then
        log "CRITICAL" "Memory exhaustion detected - may need system reboot"
    fi
    if echo "$failure_patterns" | grep -q "excessive_node_processes"; then
        log "WARNING" "Excessive Node.js processes detected - cleaning up"
        cleanup_node_processes
    fi
}

# Generate recovery recommendations
generate_recovery_recommendations() {
    local patterns="$1"
    local recommendations='[]'
    
    if echo "$patterns" | grep -q "memory_exhaustion"; then
        recommendations=$(echo "$recommendations" | jq '. += ["restart_system", "enable_memory_optimization"]')
    fi
    
    if echo "$patterns" | grep -q "excessive_node_processes"; then
        recommendations=$(echo "$recommendations" | jq '. += ["cleanup_node_processes", "check_process_leaks"]')
    fi
    
    if echo "$patterns" | grep -q "disk_space_low"; then
        recommendations=$(echo "$recommendations" | jq '. += ["cleanup_logs", "expand_disk_space"]')
    fi
    
    if [ "$recommendations" == "[]" ]; then
        recommendations='["standard_restart", "check_configuration"]'
    fi
    
    echo "$recommendations"
}

# Cleanup excessive Node.js processes
cleanup_node_processes() {
    log "INFO" "Cleaning up excessive Node.js processes..."
    
    # Kill orphaned claude-flow processes
    pgrep -f "claude-flow" | while read pid; do
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
            log "INFO" "Terminating orphaned claude-flow process: $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 5
    
    # Force kill if still running
    pgrep -f "claude-flow" | while read pid; do
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
            log "WARNING" "Force killing stubborn process: $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

# Implement recovery strategies
implement_recovery_strategy() {
    local strategy="$1"
    local state=$(load_recovery_state)
    
    case "$strategy" in
        "memoryOptimization")
            log "INFO" "Implementing memory optimization recovery..."
            # Set memory limits and cleanup
            export NODE_OPTIONS="--max-old-space-size=256 --optimize-for-size"
            cleanup_node_processes
            # Clear system caches
            sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
            update_recovery_state "recoveryStrategies.memoryOptimization" true
            ;;
            
        "processCleanup")
            log "INFO" "Implementing process cleanup recovery..."
            cleanup_node_processes
            # Clear temporary files
            find /tmp -name "*claude*" -type f -mtime +1 -delete 2>/dev/null || true
            update_recovery_state "recoveryStrategies.processCleanup" true
            ;;
            
        "dependencyCheck")
            log "INFO" "Implementing dependency check recovery..."
            # Verify Node.js and npm
            node --version && npm --version || {
                log "ERROR" "Node.js/npm verification failed"
                return 1
            }
            # Check if claude-flow is accessible
            npx claude-flow@alpha --version || {
                log "ERROR" "claude-flow accessibility check failed"
                return 1
            }
            update_recovery_state "recoveryStrategies.dependencyCheck" true
            ;;
            
        "systemReboot")
            log "CRITICAL" "System reboot required - high failure rate detected"
            # This would require sudo access and should be carefully considered
            # For now, just log the recommendation
            echo "SYSTEM REBOOT RECOMMENDED" >> "$FAILURE_LOG"
            update_recovery_state "recoveryStrategies.systemReboot" true
            return 1
            ;;
    esac
}

# Intelligent restart with recovery strategies
intelligent_restart() {
    local state=$(load_recovery_state)
    local current_time=$(date +%s)
    local consecutive_failures=$(echo "$state" | jq -r '.consecutiveFailures // 0')
    
    # Update restart statistics
    local total_restarts=$(echo "$state" | jq -r '.totalRestarts // 0')
    local rapid_restarts=$(echo "$state" | jq -r '.rapidRestarts // 0')
    local last_restart=$(echo "$state" | jq -r '.lastRestartTime // 0')
    
    total_restarts=$((total_restarts + 1))
    consecutive_failures=$((consecutive_failures + 1))
    
    # Check if this is a rapid restart
    if [ $((current_time - last_restart)) -lt $RAPID_RESTART_WINDOW ]; then
        rapid_restarts=$((rapid_restarts + 1))
    else
        rapid_restarts=1
    fi
    
    # Update state
    local updated_state=$(echo "$state" | jq \
        ".totalRestarts = $total_restarts | \
         .rapidRestarts = $rapid_restarts | \
         .lastRestartTime = $current_time | \
         .consecutiveFailures = $consecutive_failures")
    save_recovery_state "$updated_state"
    
    log "INFO" "Intelligent restart #$total_restarts (consecutive failures: $consecutive_failures)"
    
    # Perform failure analysis
    analyze_failure
    
    # Check if we need to implement recovery strategies
    if [ $consecutive_failures -ge 3 ]; then
        log "WARNING" "High failure rate detected, implementing recovery strategies..."
        
        if [ $consecutive_failures -eq 3 ]; then
            implement_recovery_strategy "memoryOptimization"
        elif [ $consecutive_failures -eq 5 ]; then
            implement_recovery_strategy "processCleanup"
        elif [ $consecutive_failures -eq 8 ]; then
            implement_recovery_strategy "dependencyCheck"
        elif [ $consecutive_failures -ge 12 ]; then
            implement_recovery_strategy "systemReboot"
            return 1
        fi
    fi
    
    # Calculate backoff delay
    local backoff_seconds=$(calculate_backoff $consecutive_failures)
    
    if [ $backoff_seconds -gt 0 ]; then
        log "INFO" "Applying exponential backoff: ${backoff_seconds}s"
        sleep $backoff_seconds
    fi
    
    # Attempt restart
    log "INFO" "Starting MCP server with recovery optimizations..."
    
    if "$MCP_RUNNER" start; then
        log "SUCCESS" "MCP server restarted successfully"
        
        # Reset consecutive failures on successful start
        update_recovery_state "consecutiveFailures" 0
        update_recovery_state "lastHealthyTime" $current_time
        return 0
    else
        log "ERROR" "Failed to restart MCP server"
        return 1
    fi
}

# Health monitoring loop
monitor_and_restart() {
    log "INFO" "Starting intelligent auto-restart monitor..."
    
    while true; do
        # Check server health
        if node "$HEALTH_CHECKER" check > /dev/null 2>&1; then
            # Server is healthy
            local current_time=$(date +%s)
            update_recovery_state "lastHealthyTime" $current_time
            update_recovery_state "consecutiveFailures" 0
        else
            # Server is unhealthy or failed
            log "ERROR" "MCP server health check failed, initiating intelligent restart..."
            
            if ! intelligent_restart; then
                log "CRITICAL" "Intelligent restart failed, backing off..."
                sleep 300  # 5-minute cooldown on critical failure
            fi
        fi
        
        sleep $HEALTH_CHECK_INTERVAL
    done
}

# Generate status report
generate_status_report() {
    local state=$(load_recovery_state)
    local current_time=$(date +%s)
    local start_time=$(echo "$state" | jq -r '.startTime // 0')
    local uptime=$((current_time - start_time))
    local uptime_hours=$((uptime / 3600))
    local uptime_minutes=$(((uptime % 3600) / 60))
    
    cat << EOF
${CYAN}=== MCP Auto-Restart Status Report ===${NC}
${BLUE}Monitor Uptime:${NC} ${uptime_hours}h ${uptime_minutes}m
${BLUE}Total Restarts:${NC} $(echo "$state" | jq -r '.totalRestarts // 0')
${BLUE}Consecutive Failures:${NC} $(echo "$state" | jq -r '.consecutiveFailures // 0')
${BLUE}Last Healthy Time:${NC} $(date -d "@$(echo "$state" | jq -r '.lastHealthyTime // 0')" 2>/dev/null || echo "Unknown")

${CYAN}Recovery Strategies Applied:${NC}
$(echo "$state" | jq -r '.recoveryStrategies | to_entries[] | select(.value == true) | "  ✓ " + .key' || echo "  None")

${CYAN}Recent Health Status:${NC}
$(if node "$HEALTH_CHECKER" check > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ MCP Server is healthy${NC}"
else
    echo -e "  ${RED}✗ MCP Server is unhealthy${NC}"
fi)
EOF
}

# Signal handlers
cleanup() {
    log "INFO" "Auto-restart monitor shutting down..."
    exit 0
}

trap cleanup EXIT TERM INT

# Main execution
main() {
    init_recovery_system
    
    case "${1:-monitor}" in
        "monitor")
            monitor_and_restart
            ;;
        "restart")
            intelligent_restart
            ;;
        "status")
            generate_status_report
            ;;
        "analyze")
            analyze_failure
            ;;
        "reset")
            log "INFO" "Resetting recovery state..."
            rm -f "$RECOVERY_STATE_FILE"
            init_recovery_system
            log "SUCCESS" "Recovery state reset"
            ;;
        *)
            echo "Usage: $0 {monitor|restart|status|analyze|reset}"
            echo ""
            echo "Commands:"
            echo "  monitor  - Start continuous monitoring with auto-restart"
            echo "  restart  - Perform single intelligent restart"
            echo "  status   - Show current status and statistics"
            echo "  analyze  - Perform failure analysis"
            echo "  reset    - Reset recovery state"
            exit 1
            ;;
    esac
}

# Ensure jq is available for JSON processing
if ! command -v jq >/dev/null 2>&1; then
    log "ERROR" "jq is required but not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y jq
fi

# Run main function
main "$@"