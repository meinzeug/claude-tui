#!/bin/bash

# ============================================
# Claude-Flow Fully Automated Setup Script
# Version: 3.0.0 - Non-Interactive
# ============================================

set -e  # Exit on error

# Configuration - Set these before running
INTERACTIVE_MODE=${INTERACTIVE_MODE:-false}  # Set to true for interactive mode
AUTO_YES=${AUTO_YES:-true}                   # Auto-answer yes to all prompts
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}           # Project directory
REPO_NAME=${REPO_NAME:-$(basename $PROJECT_DIR)}
REPO_VISIBILITY=${REPO_VISIBILITY:-private}
SKIP_MCP=${SKIP_MCP:-true}                  # Skip MCP packages (not available)
SKIP_GITHUB=${SKIP_GITHUB:-false}           # Skip GitHub setup
SKIP_AUTH=${SKIP_AUTH:-false}               # Skip Claude auth (do manually later)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ASCII Art Header
clear
echo -e "${CYAN}"
cat << "EOF"
   ____ _                 _        _____ _               
  / ___| | __ _ _   _  __| | ___  |  ___| | _____      __
 | |   | |/ _` | | | |/ _` |/ _ \ | |_  | |/ _ \ \ /\ / /
 | |___| | (_| | |_| | (_| |  __/ |  _| | | (_) \ V  V / 
  \____|_|\__,_|\__,_|\__,_|\___| |_|   |_|\___/ \_/\_/  
                                                          
         🐝 Hive Mind AI Orchestration Platform 🐝
EOF
echo -e "${NC}"
echo -e "${BOLD}Fully Automated Setup - Version 3.0${NC}"
echo "========================================================"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Project Dir: $PROJECT_DIR"
echo "  Repo Name: $REPO_NAME"
echo "  Auto Mode: Enabled"
echo ""
sleep 2

# Function to check command exists
check_command() {
    command -v $1 &> /dev/null
}

# Function to print step
print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}▶ STEP: $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to print info
print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Non-interactive yes/no (always returns based on AUTO_YES)
ask_yes_no() {
    if [ "$INTERACTIVE_MODE" = true ]; then
        while true; do
            read -p "$1 (y/n): " yn
            case $yn in
                [Yy]* ) return 0;;
                [Nn]* ) return 1;;
                * ) echo "Please answer yes (y) or no (n).";;
            esac
        done
    else
        echo "$1 [AUTO: ${AUTO_YES}]"
        [ "$AUTO_YES" = true ] && return 0 || return 1
    fi
}

# ============================================
# STEP 1: Check Prerequisites
# ============================================
print_step "Checking System Prerequisites"

MISSING_DEPS=false

# Check Node.js
if check_command node; then
    NODE_VERSION=$(node -v)
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
    if [ $NODE_MAJOR -lt 18 ]; then
        print_error "Node.js version must be 18 or higher (found: $NODE_VERSION)"
        MISSING_DEPS=true
    else
        print_success "Node.js $NODE_VERSION"
    fi
else
    print_error "Node.js is not installed"
    MISSING_DEPS=true
fi

# Check npm
if check_command npm; then
    NPM_VERSION=$(npm -v)
    print_success "npm $NPM_VERSION"
else
    print_error "npm is not installed"
    MISSING_DEPS=true
fi

# Check git
if check_command git; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    print_success "Git $GIT_VERSION"
else
    print_error "Git is not installed"
    MISSING_DEPS=true
fi

# Check GitHub CLI (optional)
if check_command gh; then
    GH_VERSION=$(gh --version 2>/dev/null | head -n1 | cut -d' ' -f3)
    print_success "GitHub CLI $GH_VERSION (optional)"
else
    print_warning "GitHub CLI not found (optional)"
fi

if [ "$MISSING_DEPS" = true ]; then
    print_error "Missing required dependencies. Please install them first."
    exit 1
fi

# ============================================
# STEP 2: Project Setup
# ============================================
print_step "Setting Up Project Directory"

# Create and enter project directory
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
print_success "Working directory: $PROJECT_DIR"

# Initialize git if needed
if [ ! -d .git ]; then
    print_info "Initializing git repository..."
    git init
    print_success "Git repository initialized"
else
    print_success "Git repository already exists"
fi

# ============================================
# STEP 3: Install Claude Code
# ============================================
print_step "Installing Claude Code CLI"

if ! check_command claude; then
    print_info "Installing @anthropic-ai/claude-code globally..."
    npm install -g @anthropic-ai/claude-code --silent
    print_success "Claude Code installed successfully"
else
    CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
    print_success "Claude Code already installed (version: $CLAUDE_VERSION)"
fi

# ============================================
# STEP 4: Claude Authentication Note
# ============================================
print_step "Claude Authentication Setup"

if [ "$SKIP_AUTH" = true ]; then
    print_warning "Skipping Claude authentication (manual setup required)"
    echo ""
    echo -e "${YELLOW}To authenticate Claude later, run:${NC}"
    echo "  claude"
    echo ""
    echo "Or use the control script:"
    echo "  ./claude-flow.sh auth"
    echo ""
    
    # Create auth reminder file
    cat > .claude-auth-required << 'EOF'
AUTHENTICATION REQUIRED
=======================
Claude Code requires OAuth authentication before use.

To authenticate:
1. Run: claude
2. Follow the browser OAuth flow
3. Or run: ./claude-flow.sh auth

This is a one-time setup per machine.
EOF
else
    print_info "Claude authentication will be handled on first run"
    echo "The 'claude' command will open browser for OAuth when needed"
fi

# ============================================
# STEP 5: Initialize Claude-Flow Structure
# ============================================
print_step "Creating Claude-Flow Project Structure"

print_info "Creating directory structure..."

# Create all necessary directories
mkdir -p .hive-mind
mkdir -p .swarm
mkdir -p .mcp/servers
mkdir -p memory
mkdir -p coordination
mkdir -p logs
mkdir -p hooks
mkdir -p agents

# Create hive-mind configuration
print_info "Creating Hive Mind configuration..."
cat > .hive-mind/config.json << 'EOL'
{
  "version": "2.0.0",
  "topology": "hierarchical",
  "agents": {
    "queen": {
      "role": "orchestrator",
      "model": "claude-3-opus",
      "temperature": 0.3
    },
    "workers": {
      "count": 5,
      "roles": ["coder", "tester", "reviewer", "researcher", "architect"],
      "model": "claude-3-sonnet",
      "temperature": 0.5
    }
  },
  "memory": {
    "type": "sqlite",
    "path": ".swarm/memory.db",
    "persistence": true
  },
  "neural": {
    "enabled": true,
    "models": 27,
    "wasm": true
  },
  "auth": {
    "type": "oauth",
    "provider": "anthropic"
  }
}
EOL
print_success "Hive-mind configuration created"

# ============================================
# STEP 6: MCP Server Setup (Simplified)
# ============================================
print_step "Setting Up MCP Server Configuration"

if [ "$SKIP_MCP" = true ]; then
    print_warning "Skipping MCP server packages (not publicly available yet)"
else
    print_info "Creating MCP configuration..."
fi

# Create basic MCP configuration
cat > .mcp/config.json << 'EOL'
{
  "version": "1.0.0",
  "servers": {
    "filesystem": {
      "enabled": false,
      "path": "@modelcontextprotocol/server-filesystem",
      "args": ["--root", "./"]
    },
    "github": {
      "enabled": false,
      "path": "@modelcontextprotocol/server-github",
      "env": {
        "GITHUB_TOKEN": ""
      }
    },
    "web-search": {
      "enabled": false,
      "path": "@modelcontextprotocol/server-web-search"
    },
    "memory": {
      "enabled": false,
      "path": "@modelcontextprotocol/server-memory",
      "args": ["--db", ".swarm/memory.db"]
    }
  },
  "note": "MCP servers disabled - packages not yet publicly available"
}
EOL

# Create placeholder MCP scripts
cat > .mcp/start-servers.sh << 'EOL'
#!/bin/bash
echo "🔍 MCP servers are not yet available on public npm"
echo "📝 Claude-Flow will work without them"
echo "✅ Continuing with core functionality..."
EOL
chmod +x .mcp/start-servers.sh

cat > .mcp/stop-servers.sh << 'EOL'
#!/bin/bash
echo "✅ MCP server shutdown (no servers running)"
EOL
chmod +x .mcp/stop-servers.sh

print_success "MCP configuration created (placeholder mode)"

# ============================================
# STEP 7: GitHub Setup (Optional)
# ============================================
print_step "GitHub Integration Setup"

if [ "$SKIP_GITHUB" = false ] && check_command gh; then
    print_info "Setting up GitHub integration..."
    
    # Check if gh is authenticated
    if gh auth status &>/dev/null; then
        print_success "GitHub CLI authenticated"
        
        # Try to create repo if no remote exists
        if ! git remote get-url origin &>/dev/null; then
            print_info "Creating GitHub repository: $REPO_NAME"
            gh repo create "$REPO_NAME" --$REPO_VISIBILITY --clone=false 2>/dev/null || {
                print_warning "Could not create repo (may already exist)"
            }
            
            # Try to add remote
            REPO_URL="git@github.com:$(gh api user -q .login)/$REPO_NAME.git"
            git remote add origin "$REPO_URL" 2>/dev/null || true
        fi
    else
        print_warning "GitHub CLI not authenticated. Run 'gh auth login' to setup"
    fi
    
    # Create GitHub Actions workflow
    mkdir -p .github/workflows
    cat > .github/workflows/claude-code.yml << 'EOACTION'
name: Claude Code CI

on:
  issue_comment:
    types: [created]
  pull_request:
    types: [opened, synchronize]

jobs:
  claude-code:
    if: contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - name: Claude Code Action
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          echo "Claude Code action placeholder"
          echo "Add ANTHROPIC_API_KEY to repository secrets"
EOACTION
    
    print_success "GitHub Actions workflow created"
else
    print_info "Skipping GitHub integration"
fi

# ============================================
# STEP 8: Create Control Scripts
# ============================================
print_step "Creating Control Scripts"

print_info "Creating main control script..."

# Main launcher script
cat > claude-flow.sh << 'EOLAUNCHER'
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
EOLAUNCHER

chmod +x claude-flow.sh
print_success "Control script created: ./claude-flow.sh"

# ============================================
# STEP 9: Create Documentation Files
# ============================================
print_step "Creating Documentation"

# Create CLAUDE.md
cat > CLAUDE.md << 'EOCLAUDE'
# Claude Code Configuration

This file configures Claude Code behavior for this project.

## Project Context

Claude-Flow managed project with Hive Mind orchestration.
OAuth authentication via Anthropic Console.

## Development Standards

### Code Style
- Consistent indentation (2 spaces JS/TS, 4 spaces Python)
- Language-specific conventions
- Comprehensive comments for complex logic
- Descriptive naming

### Git Workflow
- Feature branches for new work
- Clear commit messages (conventional commits)
- Test before committing

### Testing
- Tests for all features
- 80% minimum coverage
- TDD when appropriate
- Unit and integration tests

### Security
- No secrets in code
- Environment variables for config
- Input validation
- OWASP guidelines

## Agent Roles

- **Architect**: System design
- **Coder**: Implementation
- **Tester**: Testing
- **Reviewer**: Code review
- **Researcher**: Best practices

## Authentication

Run `claude` to authenticate via OAuth.
EOCLAUDE

# Create README.md
cat > README.md << 'EOREADME'
# Claude-Flow Project

AI-powered development with Claude-Flow v2.0.0 and Hive Mind orchestration.

## 🚀 Quick Start

```bash
# 1. Authenticate Claude (first time only)
./claude-flow.sh auth

# 2. Start Claude-Flow
./claude-flow.sh start

# 3. Run a task
./claude-flow.sh swarm "Build a feature"

# 4. Check status
./claude-flow.sh status
```

## 📚 Commands

| Command | Description |
|---------|-------------|
| `start` | Start all services |
| `stop` | Stop all services |
| `status` | Check system status |
| `swarm <task>` | Execute swarm task |
| `sparc <research>` | Research with SPARC |
| `auth` | Setup OAuth authentication |
| `doctor` | Run diagnostics |
| `init` | Initialize/reset |

## 🐝 Features

- **Hive Mind** orchestration with Queen/Worker pattern
- **27+ Neural Models** with WASM acceleration
- **SQLite Memory** for persistent context
- **OAuth Authentication** via Anthropic Console
- **Auto-checkpointing** for safe rollback

## 📁 Structure

```
.hive-mind/     # Hive configuration
.swarm/         # Memory database
.mcp/           # MCP servers (future)
memory/         # Agent memories
logs/           # System logs
CLAUDE.md       # Claude configuration
```

## 🔐 Authentication

First time setup:
```bash
claude  # Opens browser for OAuth
```

## 📖 Links

- [Claude-Flow](https://github.com/ruvnet/claude-flow)
- [Claude Code](https://docs.anthropic.com/claude-code)
- [Anthropic Console](https://console.anthropic.com)
EOREADME

print_success "Documentation created"

# Create .gitignore
cat > .gitignore << 'EOGITIGNORE'
# Environment
.env
.env.local
*.env

# Authentication
.claude/
.claude_authenticated
.claude-auth-required

# Claude-Flow
.swarm/
.mcp/servers/*.pid
memory/
coordination/
logs/
*.log

# Dependencies
node_modules/
package-lock.json
yarn.lock
pnpm-lock.yaml

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Temp
*.tmp
*.temp
*.bak
/tmp/
EOGITIGNORE

print_success ".gitignore created"

# ============================================
# STEP 10: Initialize Claude-Flow
# ============================================
print_step "Initializing Claude-Flow Components"

print_info "Running Claude-Flow initialization..."

# Initialize Claude-Flow with auto-yes
echo "Installing Claude-Flow (this may take a moment)..."
yes | npx -y claude-flow@alpha init --force 2>/dev/null || {
    print_warning "Claude-Flow initialization will complete on first run"
}

# Try to initialize memory
yes | npx -y claude-flow@alpha memory init 2>/dev/null || {
    print_info "Memory will be initialized on first use"
}

# Try to initialize hive-mind
yes | npx -y claude-flow@alpha hive-mind init 2>/dev/null || {
    print_info "Hive-mind will be initialized on first use"
}

print_success "Claude-Flow components initialized"

# ============================================
# STEP 11: Git Initial Commit
# ============================================
if [ -d .git ]; then
    print_step "Creating Initial Git Commit"
    
    git add . 2>/dev/null || true
    git commit -m "Initial Claude-Flow setup (automated)" 2>/dev/null || {
        print_info "Git commit skipped (no changes or already committed)"
    }
fi

# ============================================
# STEP 12: Installation Summary
# ============================================
print_step "Installation Complete!"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${GREEN}✅ Claude-Flow Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check authentication status
AUTH_STATUS="Not checked"
if claude doctor &>/dev/null 2>&1; then
    AUTH_STATUS="✅ Authenticated"
else
    AUTH_STATUS="⚠️  Not authenticated"
fi

echo -e "${CYAN}📊 Setup Summary:${NC}"
echo "  Location: $PROJECT_DIR"
echo "  Repository: $REPO_NAME"
echo "  Claude Auth: $AUTH_STATUS"
echo "  MCP Servers: Placeholder (packages not available)"
echo ""

if [ "$AUTH_STATUS" = "⚠️  Not authenticated" ]; then
    echo -e "${YELLOW}⚠️  Important: Claude Authentication Required${NC}"
    echo ""
    echo "Before using Claude-Flow, you must authenticate:"
    echo -e "  ${BOLD}claude${NC}"
    echo ""
    echo "This will open your browser for OAuth login."
    echo "This is a one-time setup per machine."
    echo ""
fi

echo -e "${CYAN}🎯 Next Steps:${NC}"
echo ""
if [ "$AUTH_STATUS" = "⚠️  Not authenticated" ]; then
    echo "1. Authenticate Claude:"
    echo -e "   ${BOLD}claude${NC}"
    echo ""
    echo "2. Start Claude-Flow:"
else
    echo "1. Start Claude-Flow:"
fi
echo -e "   ${BOLD}./claude-flow.sh start${NC}"
echo ""
echo "2. Run your first task:"
echo -e "   ${BOLD}./claude-flow.sh swarm 'Create a REST API'${NC}"
echo ""
echo "3. Check diagnostics:"
echo -e "   ${BOLD}./claude-flow.sh doctor${NC}"
echo ""

echo -e "${MAGENTA}📚 Quick Reference:${NC}"
echo "• Help: ./claude-flow.sh help"
echo "• Logs: ./claude-flow.sh logs"
echo "• Status: ./claude-flow.sh status"
echo ""

echo -e "${GREEN}✨ Tips:${NC}"
echo "• Use 'swarm' for coding tasks"
echo "• Use 'sparc' for research"
echo "• Memory persists between sessions"
echo "• Check logs/ for debugging"
echo ""

# Create success marker
touch .claude-flow-installed

echo -e "${BOLD}${GREEN}🐝 Happy coding with Claude-Flow!${NC}"
echo ""

# Final message about authentication
if [ "$AUTH_STATUS" = "⚠️  Not authenticated" ]; then
    echo -e "${YELLOW}Remember: Run 'claude' first to authenticate!${NC}"
fi

# Exit successfully
exit 0
