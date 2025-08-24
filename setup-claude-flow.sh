#!/bin/bash

# Claude Flow User Setup Script
# Runs in user context after server provisioning

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🚀 Setting up Claude-Flow environment for user..."

# Configure NPM for user
print_info "Configuring NPM for user environment..."
npm config set prefix ~/.npm-global
mkdir -p ~/.npm-global
export PATH=~/.npm-global/bin:$PATH
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc

# Install Claude tools in user space (backup if global install failed)
print_info "Ensuring Claude tools are available..."
if ! which claude &>/dev/null; then
    print_warning "Claude not found in PATH, installing in user space..."
    npm install -g @anthropic-ai/claude-code
fi

# Verify Claude Code is working
if claude --version &>/dev/null; then
    print_status "Claude Code: $(claude --version)"
else
    print_error "Claude Code installation failed"
    exit 1
fi

# Verify npx and claude-flow are available
if npx claude-flow@alpha --version &>/dev/null; then
    print_status "Claude Flow available via npx: $(npx claude-flow@alpha --version)"
else
    print_warning "Claude Flow might need to be downloaded on first use"
fi

# Initialize Claude Flow with comprehensive fixes
print_info "Initializing Claude Flow with comprehensive schema fixes..."

# Create backup directory
mkdir -p backups

# Clean slate initialization
if [ -d ".claude" ]; then mv .claude "backups/.claude.$(date +%s)"; fi
if [ -d ".swarm" ]; then mv .swarm "backups/.swarm.$(date +%s)"; fi
if [ -d ".hive-mind" ]; then mv .hive-mind "backups/.hive-mind.$(date +%s)"; fi

# Initialize Claude Flow
npx claude-flow@alpha init --force
npx claude-flow@alpha hive-mind init --force

print_status "Claude Flow initialized"

# Apply comprehensive database schema fixes
print_info "Applying comprehensive database schema fixes..."

if [ -f ".hive-mind/hive.db" ]; then
    # Check existing columns to avoid duplicates
    EXISTING_SESSION_COLS=$(sqlite3 .hive-mind/hive.db "PRAGMA table_info(sessions);" | cut -d'|' -f2)
    EXISTING_AGENT_COLS=$(sqlite3 .hive-mind/hive.db "PRAGMA table_info(agents);" | cut -d'|' -f2)

    # Add missing columns to sessions table
    echo "$EXISTING_SESSION_COLS" | grep -q "swarm_name" || sqlite3 .hive-mind/hive.db "ALTER TABLE sessions ADD COLUMN swarm_name TEXT;"
    echo "$EXISTING_SESSION_COLS" | grep -q "objective" || sqlite3 .hive-mind/hive.db "ALTER TABLE sessions ADD COLUMN objective TEXT;"
    echo "$EXISTING_SESSION_COLS" | grep -q "checkpoint_data" || sqlite3 .hive-mind/hive.db "ALTER TABLE sessions ADD COLUMN checkpoint_data TEXT DEFAULT '{}';"
    echo "$EXISTING_SESSION_COLS" | grep -q "workflow_state" || sqlite3 .hive-mind/hive.db "ALTER TABLE sessions ADD COLUMN workflow_state TEXT DEFAULT '{}';"
    echo "$EXISTING_SESSION_COLS" | grep -q "description" || sqlite3 .hive-mind/hive.db "ALTER TABLE sessions ADD COLUMN description TEXT;"
    echo "$EXISTING_SESSION_COLS" | grep -q "config" || sqlite3 .hive-mind/hive.db "ALTER TABLE sessions ADD COLUMN config TEXT DEFAULT '{}';"
    echo "$EXISTING_SESSION_COLS" | grep -q "topology" || sqlite3 .hive-mind/hive.db "ALTER TABLE sessions ADD COLUMN topology TEXT DEFAULT 'mesh';"

    # Add missing columns to agents table
    echo "$EXISTING_AGENT_COLS" | grep -q "role" || sqlite3 .hive-mind/hive.db "ALTER TABLE agents ADD COLUMN role TEXT DEFAULT 'worker';"
    echo "$EXISTING_AGENT_COLS" | grep -q "specialization" || sqlite3 .hive-mind/hive.db "ALTER TABLE agents ADD COLUMN specialization TEXT;"

    # Create missing tables for auto-save functionality
    sqlite3 .hive-mind/hive.db << 'EOSQL'
CREATE TABLE IF NOT EXISTS session_checkpoints (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    checkpoint_data TEXT,
    checkpoint_type TEXT DEFAULT 'auto',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);

CREATE TABLE IF NOT EXISTS workflow_state (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    workflow_data TEXT,
    checkpoint_data TEXT,
    state TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
EOSQL

    print_status "Database schema fixes applied"
fi

# Create helpful aliases
cat >> ~/.bashrc << 'EOFBASH'

# Claude-Flow Development Aliases
alias cf="npx claude-flow@alpha"
alias cfh="npx claude-flow@alpha hive-mind"
alias cfs="npx claude-flow@alpha swarm"
alias cfm="npx claude-flow@alpha memory"
alias cc="claude --dangerously-skip-permissions"
alias cc-safe="claude"
EOFBASH

source ~/.bashrc
print_status "Development aliases configured"

# Configure Claude MCP servers
print_info "Configuring Claude MCP servers..."
claude mcp remove claude-flow 2>/dev/null || true
claude mcp remove ruv-swarm 2>/dev/null || true
claude mcp add claude-flow "npx claude-flow@alpha mcp start" || print_warning "MCP setup may need manual configuration"
claude mcp add ruv-swarm "npx ruv-swarm mcp start" || print_warning "MCP setup may need manual configuration"

print_status "Claude-Flow environment setup complete"

# Test the installation
print_info "Testing installation..."
npx claude-flow@alpha memory stats || print_warning "Memory test failed"
npx claude-flow@alpha hive-mind status || print_warning "Hive-mind test failed"

print_status "🎉 User environment ready!"
echo
print_info "Next steps:"
echo "  1. Connect your Claude account in the interactive session"
echo "  2. Documentation will be auto-generated from your concept"
echo "  3. Use work.sh for ongoing development"
echo
