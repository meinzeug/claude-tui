#!/bin/bash

# Claude-Flow Control Script - Automated Version
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

COMMAND=${1:-help}
shift

# Function to check Claude authentication
check_claude_auth() {
    if ! claude doctor &>/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Claude authentication required${NC}"
        echo "Please run: claude"
        echo "This will guide you through OAuth authentication"
        return 1
    fi
    return 0
}

# Function to run npx commands with auto-yes
run_npx() {
    yes | npx -y "$@" 2>/dev/null || npx -y "$@"
}

case $COMMAND in
    start)
        echo "🚀 Starting Claude-Flow..."
        
        # Start MCP servers (placeholder)
        cd .mcp && ./start-servers.sh && cd ..
        
        # Initialize Hive Mind with auto-yes
        echo "Initializing Hive Mind..."
        run_npx claude-flow@alpha hive-mind init --force || true
        
        # Show status
        run_npx claude-flow@alpha status || echo "Status check skipped"
        
        echo -e "${GREEN}✅ Claude-Flow is ready!${NC}"
        echo "Note: Run 'claude' first if authentication is needed"
        ;;
        
    stop)
        echo "🛑 Stopping Claude-Flow..."
        cd .mcp && ./stop-servers.sh && cd ..
        echo -e "${GREEN}✅ Claude-Flow stopped${NC}"
        ;;
        
    status)
        run_npx claude-flow@alpha status
        run_npx claude-flow@alpha memory stats 2>/dev/null || true
        run_npx claude-flow@alpha hive-mind status 2>/dev/null || true
        ;;
        
    swarm)
        TASK="$@"
        if [ -z "$TASK" ]; then
            echo "Usage: ./claude-flow.sh swarm 'your development task'"
            exit 1
        fi
        if ! check_claude_auth; then
            echo -e "${RED}Please authenticate Claude first: run 'claude'${NC}"
            exit 1
        fi
        run_npx claude-flow@alpha swarm "$TASK"
        ;;
        
    sparc)
        TASK="$@"
        if ! check_claude_auth; then
            echo -e "${RED}Please authenticate Claude first: run 'claude'${NC}"
            exit 1
        fi
        run_npx claude-flow@alpha sparc "$TASK"
        ;;
        
    hive)
        run_npx claude-flow@alpha hive-mind "$@"
        ;;
        
    memory)
        run_npx claude-flow@alpha memory "$@"
        ;;
        
    agent)
        run_npx claude-flow@alpha agent "$@"
        ;;
        
    auth)
        echo "🔐 Starting Claude OAuth authentication..."
        echo ""
        echo -e "${CYAN}This will open your browser for authentication${NC}"
        echo "Follow the OAuth flow and return here when complete"
        echo ""
        claude
        ;;
        
    doctor)
        echo "🏥 Running diagnostics..."
        echo ""
        echo "Claude Code:"
        if claude doctor 2>/dev/null; then
            echo "  ✅ Authenticated"
        else
            echo "  ❌ Not authenticated (run: ./claude-flow.sh auth)"
        fi
        echo ""
        echo "Claude-Flow:"
        run_npx claude-flow@alpha --version 2>/dev/null || echo "  ⚠️  Not found"
        echo ""
        echo "Project Structure:"
        [ -d .hive-mind ] && echo "  ✅ .hive-mind/" || echo "  ❌ .hive-mind/"
        [ -d .swarm ] && echo "  ✅ .swarm/" || echo "  ❌ .swarm/"
        [ -d logs ] && echo "  ✅ logs/" || echo "  ❌ logs/"
        ;;
        
    init)
        echo "🔧 Initializing Claude-Flow..."
        run_npx claude-flow@alpha init --force
        run_npx claude-flow@alpha hive-mind init --force
        run_npx claude-flow@alpha memory init
        echo -e "${GREEN}✅ Initialization complete${NC}"
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    logs)
        if [ -d logs ] && [ "$(ls -A logs 2>/dev/null)" ]; then
            tail -f logs/*.log 2>/dev/null || echo "No logs available yet"
        else
            echo "No logs available yet"
        fi
        ;;
        
    help|*)
        cat << EOF
${BOLD}Claude-Flow Control Script${NC}

Usage: ./claude-flow.sh [command] [options]

${CYAN}Commands:${NC}
  ${GREEN}start${NC}     - Start Claude-Flow and services
  ${GREEN}stop${NC}      - Stop all services
  ${GREEN}status${NC}    - Show current status
  ${GREEN}restart${NC}   - Restart all services
  ${GREEN}init${NC}      - Initialize/reinitialize Claude-Flow
  
  ${YELLOW}swarm${NC}     - Run a swarm task (requires auth)
  ${YELLOW}sparc${NC}     - Run SPARC methodology (requires auth)
  ${YELLOW}hive${NC}      - Hive-mind operations
  ${YELLOW}memory${NC}    - Memory operations
  ${YELLOW}agent${NC}     - Agent management
  
  ${CYAN}auth${NC}      - Setup Claude OAuth authentication
  ${CYAN}doctor${NC}    - Run system diagnostics
  ${CYAN}logs${NC}      - Show live logs
  ${CYAN}help${NC}      - Show this help

${BOLD}Examples:${NC}
  ./claude-flow.sh start
  ./claude-flow.sh auth                           # First-time setup
  ./claude-flow.sh swarm "Build a REST API"       # Run task
  ./claude-flow.sh sparc "Research frameworks"    # Research
  ./claude-flow.sh doctor                         # Check health
  ./claude-flow.sh stop

${YELLOW}Note:${NC} First run 'claude' or './claude-flow.sh auth' to authenticate
EOF
        ;;
esac
