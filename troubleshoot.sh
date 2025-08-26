#!/bin/bash

# Troubleshooting Script for Claude-Flow Environment
# Helps diagnose and fix common issues

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

clear
echo "🔧 Claude-Flow Environment Troubleshooter"
echo "========================================"
echo

# Check Node.js
print_info "Checking Node.js..."
if command -v node &>/dev/null; then
    print_status "Node.js installed: $(node --version)"
else
    print_error "Node.js not found!"
fi

# Check NPM
print_info "Checking NPM..."
if command -v npm &>/dev/null; then
    print_status "NPM installed: $(npm --version)"
else
    print_error "NPM not found!"
fi

# Check Claude Code
print_info "Checking Claude Code..."
if command -v claude &>/dev/null; then
    if claude --version &>/dev/null; then
        print_status "Claude Code installed: $(claude --version)"
    else
        print_warning "Claude Code found but version check failed"
    fi
else
    print_error "Claude Code not found!"
    print_info "Try: npm install -g @anthropic-ai/claude-code"
fi

# Check Claude-Flow
print_info "Checking Claude-Flow..."
if npx claude-flow@alpha --version &>/dev/null; then
    print_status "Claude-Flow available: $(npx claude-flow@alpha --version)"
else
    print_warning "Claude-Flow not responding properly"
    print_info "Try: npm install -g claude-flow@alpha"
fi

# Check SQLite
print_info "Checking SQLite..."
if command -v sqlite3 &>/dev/null; then
    print_status "SQLite installed: $(sqlite3 --version | cut -d' ' -f1)"
else
    print_error "SQLite not found!"
fi

# Check Claude-Flow directories
print_info "Checking Claude-Flow directories..."
for dir in .claude .swarm .hive-mind; do
    if [ -d "$dir" ]; then
        print_status "$dir directory exists"
    else
        print_warning "$dir directory missing - run: npx claude-flow@alpha init"
    fi
done

# Check database
if [ -f ".hive-mind/hive.db" ]; then
    print_status "Hive database exists"
    print_info "Database tables:"
    sqlite3 .hive-mind/hive.db ".tables" | sed 's/^/  • /'
else
    print_warning "Hive database missing - run: npx claude-flow@alpha hive-mind init"
fi

echo
print_info "Quick fixes for common issues:"
echo "  • Reinitialize Claude-Flow: npx claude-flow@alpha init --force"
echo "  • Fix database schema: ./setup-claude-flow.sh"
echo "  • Clear NPM cache: npm cache clean --force"
echo "  • Reinstall Claude Code: npm install -g @anthropic-ai/claude-code"
echo "  • Check PATH: echo \$PATH"
echo
