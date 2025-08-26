#!/bin/bash

# Documentation Generator Script
# Generates project documentation from konzept.md using Claude-Flow hive-mind

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
echo "📚 Generating Project Documentation from Concept"
echo "==============================================="
echo

if [ ! -f "docs/konzept.md" ]; then
    print_error "docs/konzept.md not found!"
    exit 1
fi

CONCEPT_CONTENT=$(cat docs/konzept.md)
CONCEPT_SIZE=$(echo "$CONCEPT_CONTENT" | wc -c)
print_info "Loaded concept: $CONCEPT_SIZE characters"

print_info "Starting Claude-Flow hive-mind to generate comprehensive documentation..."

# Initial prompt for documentation generation
INITIAL_PROMPT="Analysiere das Projektkonzept in docs/konzept.md vollständig und erstelle eine umfassende Projektdokumentation. Erstelle folgende Markdown-Dateien im docs/ Verzeichnis basierend auf dem Konzept:

1. docs/roadmap.md - Detaillierte Projekt-Roadmap mit Meilensteinen und Zeitplan
2. docs/architecture.md - Technische Architektur und Systemdesign
3. docs/requirements.md - Funktionale und non-funktionale Anforderungen
4. docs/api-specification.md - API-Design und Endpunkt-Spezifikationen
5. docs/database-schema.md - Datenbankdesign und Schema
6. docs/deployment.md - Deployment-Strategie und DevOps-Konzept
7. docs/security.md - Sicherheitskonzept und Best Practices
8. docs/testing-strategy.md - Test-Strategie und Qualitätssicherung
9. docs/user-guide.md - Benutzerhandbuch und Dokumentation
10. docs/developer-guide.md - Entwickler-Setup und Contributing Guidelines

Konzept-Inhalt:
$CONCEPT_CONTENT

Analysiere das Konzept gründlich und erstelle professionelle, detaillierte Markdown-Dokumentation die als Basis für die Entwicklung dient. Jede Datei soll vollständig und implementierungsbereit sein."

# Execute hive-mind spawn with the initial prompt
npx claude-flow@alpha hive-mind spawn "$INITIAL_PROMPT" --claude --auto-spawn

print_status "🎉 Documentation generation started!"
print_info "Monitor progress with: npx claude-flow@alpha hive-mind status"
