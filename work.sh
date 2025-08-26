#!/bin/bash

# Ongoing Development Work Script
# Uses Claude-Flow hive-mind for continuous development based on docs and codebase

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
print_header "Claude-Flow Continuous Development Session"
echo "========================================"
echo

# Check if documentation exists
if [ ! -d "docs" ] || [ -z "$(ls -A docs/ 2>/dev/null)" ]; then
    print_warning "No documentation found in docs/ directory"
    print_info "Run ./docs-generator.sh first to generate documentation"
    exit 1
fi

print_info "Project: $(basename "$PWD")"
print_info "Documentation files found:"
ls -la docs/*.md 2>/dev/null | awk '{print "  • " $9}' || print_warning "No .md files in docs/"

# Check codebase
if [ -d "src" ] && [ "$(ls -A src/ 2>/dev/null)" ]; then
    print_info "Existing codebase found in src/"
    CODEBASE_STATUS="with existing codebase"
else
    print_info "No existing codebase - starting fresh development"
    CODEBASE_STATUS="fresh development"
fi

echo

# Collect all documentation content
print_info "Reading all documentation files..."
ALL_DOCS=""
for doc_file in docs/*.md; do
    if [ -f "$doc_file" ]; then
        print_info "Loading: $(basename "$doc_file")"
        ALL_DOCS="$ALL_DOCS

=== $(basename "$doc_file") ===
$(cat "$doc_file")
"
    fi
done

# Collect codebase summary if exists
CODEBASE_SUMMARY=""
if [ -d "src" ] && [ "$(ls -A src/ 2>/dev/null)" ]; then
    print_info "Analyzing existing codebase..."
    CODEBASE_SUMMARY="

=== CURRENT CODEBASE STRUCTURE ===
$(find src/ -type f -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.html" -o -name "*.css" 2>/dev/null | head -20)

=== RECENT CHANGES ===
$(git log --oneline -10 2>/dev/null || echo "No git history available")
"
fi

# Create comprehensive development prompt
WORK_PROMPT="Analysiere die vollständige Projektdokumentation und entwickle das Projekt weiter. Arbeite iterativ und systematisch basierend auf:

PROJEKTDOKUMENTATION:
$ALL_DOCS

AKTUELLE CODEBASE:
$CODEBASE_SUMMARY

ENTWICKLUNGSAUFGABEN:
1. Analysiere alle Markdown-Dateien in docs/ um den aktuellen Projektstand zu verstehen
2. Prüfe die bestehende Codebase in src/ (falls vorhanden)
3. Identifiziere die nächsten logischen Entwicklungsschritte basierend auf der Roadmap
4. Implementiere Features systematisch according to architecture.md
5. Erstelle/aktualisiere Tests basierend auf testing-strategy.md
6. Halte dich an die Spezifikationen in requirements.md und api-specification.md
7. Dokumentiere alle Änderungen und aktualisiere relevante .md Dateien

STATUS: $CODEBASE_STATUS

Arbeite systematisch und dokumentiere alle Schritte. Fokussiere auf qualitativ hochwertigen, produktionstauglichen Code."

print_info "Starting Claude-Flow hive-mind for continuous development..."
print_warning "This will start an interactive development session"
print_warning "Press Ctrl+C to pause and save progress at any time"

echo
read -p "🚀 Start development session? (Y/n): " START_CONFIRM
if [[ "$START_CONFIRM" =~ ^[Nn]$ ]]; then
    print_info "Development session cancelled"
    exit 0
fi

# Execute hive-mind spawn with the work prompt
npx claude-flow@alpha hive-mind spawn "$WORK_PROMPT" --claude --auto-spawn

print_status "🎉 Development session completed!"
