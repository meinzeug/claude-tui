#!/bin/bash

# Project Startup Script
# Starts Claude Code for account connection, then generates docs and starts development

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_header() { echo -e "${PURPLE}🚀 $1${NC}"; }

clear
print_header "Claude Development Environment Startup"
echo "======================================"
echo

print_info "Project: $(basename "$PWD")"
print_info "Concept available in: docs/konzept.md"
echo

print_header "Step 1: Claude Code Account Connection"
print_warning "In Claude Code:"
print_warning "  1. Connect your Anthropic account (/login)"
print_warning "  2. Verify your API access"
print_warning "  3. Exit Claude Code (Ctrl+C or /exit) when ready"
echo

print_info "Starting Claude Code with dangerous permissions for account setup..."
claude --dangerously-skip-permissions

print_status "Claude Code session completed"
echo

print_header "Step 2: Generating Project Documentation"
print_info "Running documentation generator..."

# Run docs generator
./docs-generator.sh

print_status "Documentation generation completed"
echo

print_header "Step 3: Starting Continuous Development"
print_info "Launching ongoing development session..."

# Run work script
./work.sh

print_status "🎉 Complete development environment started!"
