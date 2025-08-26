#!/bin/bash
# Claude-TUI Documentation Build and Deployment Pipeline
# ====================================================
#
# This script builds comprehensive documentation for Claude-TUI including:
# - Auto-generated API docs from OpenAPI specs
# - Code documentation from source analysis
# - Interactive tutorials
# - Multi-format output (HTML, PDF, Markdown)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/src"
DOCS_DIR="$SCRIPT_DIR/docs" 
BUILD_DIR="$DOCS_DIR/_build"
SITE_DIR="$BUILD_DIR/site"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Required Python packages
    python3 -c "import mkdocs" 2>/dev/null || missing_deps+=("mkdocs")
    python3 -c "import jinja2" 2>/dev/null || missing_deps+=("jinja2")
    python3 -c "import yaml" 2>/dev/null || missing_deps+=("pyyaml")
    python3 -c "import markdown" 2>/dev/null || missing_deps+=("markdown")
    
    # Optional dependencies
    if ! command -v pandoc &> /dev/null; then
        log_warning "pandoc not found - PDF generation will be disabled"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Install with: pip install ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

# Setup build environment
setup_build_env() {
    log_info "Setting up build environment..."
    
    # Create build directories
    mkdir -p "$BUILD_DIR"
    mkdir -p "$BUILD_DIR/api"
    mkdir -p "$BUILD_DIR/modules"
    mkdir -p "$BUILD_DIR/tutorials"
    mkdir -p "$BUILD_DIR/images"
    mkdir -p "$BUILD_DIR/assets"
    
    # Copy static assets
    if [ -d "$DOCS_DIR/images" ]; then
        cp -r "$DOCS_DIR/images"/* "$BUILD_DIR/images/" 2>/dev/null || true
    fi
    
    if [ -d "$DOCS_DIR/assets" ]; then
        cp -r "$DOCS_DIR/assets"/* "$BUILD_DIR/assets/" 2>/dev/null || true
    fi
    
    log_success "Build environment ready"
}

# Generate API documentation from OpenAPI spec
generate_api_docs() {
    log_info "Generating API documentation..."
    
    if [ ! -f "$DOCS_DIR/openapi-specification.yaml" ]; then
        log_warning "OpenAPI specification not found - skipping API docs"
        return
    fi
    
    # Use Python script to generate API docs
    python3 << 'EOF'
import yaml
import json
from pathlib import Path

def generate_api_md(spec_path, output_path):
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    
    md_content = f"""# API Reference

## {spec['info']['title']} v{spec['info']['version']}

{spec['info']['description']}

## Base URLs

"""
    
    for server in spec.get('servers', []):
        md_content += f"- **{server['description']}**: `{server['url']}`\n"
    
    md_content += "\n## Endpoints\n\n"
    
    for path, methods in spec.get('paths', {}).items():
        md_content += f"### {path}\n\n"
        
        for method, details in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                md_content += f"#### {method.upper()} {path}\n\n"
                
                if 'summary' in details:
                    md_content += f"**{details['summary']}**\n\n"
                
                if 'description' in details:
                    md_content += f"{details['description']}\n\n"
                
                md_content += "---\n\n"
    
    with open(output_path, 'w') as f:
        f.write(md_content)

generate_api_md('docs/openapi-specification.yaml', 'docs/_build/api-reference-generated.md')
EOF
    
    log_success "API documentation generated"
}

# Copy existing documentation
copy_existing_docs() {
    log_info "Copying existing documentation..."
    
    # Main documentation files
    local main_docs=(
        "index.md"
        "architecture.md" 
        "roadmap.md"
        "CONTRIBUTING.md"
    )
    
    for doc in "${main_docs[@]}"; do
        if [ -f "$DOCS_DIR/$doc" ]; then
            cp "$DOCS_DIR/$doc" "$BUILD_DIR/"
        fi
    done
    
    # Copy subdirectories
    local subdirs=(
        "user-guide"
        "developer-guide" 
        "operations"
        "tutorials"
        "api-reference"
        "architecture"
    )
    
    for subdir in "${subdirs[@]}"; do
        if [ -d "$DOCS_DIR/$subdir" ]; then
            cp -r "$DOCS_DIR/$subdir" "$BUILD_DIR/"
        fi
    done
    
    log_success "Documentation files copied"
}

# Generate navigation structure
generate_navigation() {
    log_info "Generating navigation structure..."
    
    cat > "$BUILD_DIR/mkdocs.yml" << 'EOF'
site_name: Claude-TUI Documentation
site_description: Comprehensive documentation for Claude-TUI intelligent development brain
site_author: Claude-TUI Team
site_url: https://docs.claude-tui.dev

# Repository
repo_name: claude-tui/claude-tui
repo_url: https://github.com/claude-tui/claude-tui
edit_uri: edit/main/docs/

# Configuration
theme:
  name: material
  language: en
  palette:
    # Palette toggle for dark mode
    - scheme: slate
      primary: blue
      accent: blue
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
    # Palette toggle for light mode  
    - scheme: default
      primary: blue
      accent: blue
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
  
  features:
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.expand
    - navigation.top
    - navigation.tracking
    - search.highlight
    - search.share
    - search.suggest
    - content.code.annotate
    - content.code.copy
    - content.tooltips

  icon:
    repo: fontawesome/brands/github
    edit: material/pencil
    view: material/eye

# Extensions
markdown_extensions:
  - admonition
  - attr_list
  - codehilite:
      guess_lang: false
  - toc:
      permalink: true
  - md_in_html
  - pymdownx.arithmatex
  - pymdownx.betterem:
      smart_enable: all
  - pymdownx.caret
  - pymdownx.critic
  - pymdownx.details
  - pymdownx.emoji:
      emoji_index: !!python/name:materialx.emoji.twemoji
      emoji_generator: !!python/name:materialx.emoji.to_svg
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.keys
  - pymdownx.magiclink
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.tilde

# Plugins
plugins:
  - search:
      lang: en
  - minify:
      minify_html: true
  - git-revision-date-localized:
      type: date
      fallback_to_build_date: true

# Navigation
nav:
  - Home: index.md
  - Getting Started:
    - Overview: user-guide/getting-started.md  
    - Installation: user-guide/installation.md
    - First Tutorial: tutorials/getting-started-tutorial.md
  - User Guide:
    - Introduction: user-guide/getting-started.md
    - Installation: user-guide/installation.md
    - Basic Usage: user-guide/basic-usage.md
    - Advanced Features: user-guide/advanced-features.md
    - Project Templates: user-guide/project-templates.md
    - AI Agents: user-guide/ai-agents.md
    - Troubleshooting: troubleshooting-faq.md
  - Developer Guide:
    - Architecture: developer-guide/architecture-deep-dive.md
    - Contributing: developer-guide/contributing.md
    - Custom Agents: developer-guide/custom-agents.md
    - Plugin Development: developer-guide/plugin-development.md
    - API Integration: developer-guide/api-integration.md
  - API Reference:
    - Overview: api-reference/comprehensive-api-guide.md
    - Authentication: api-reference/authentication.md
    - Projects: api-reference/projects.md
    - Tasks: api-reference/tasks.md
    - AI Agents: api-reference/agents.md
    - WebSocket: api-reference/websocket.md
  - Tutorials:
    - Getting Started: tutorials/getting-started-tutorial.md
    - Advanced Features: tutorials/advanced-features-tutorial.md
    - API Integration: tutorials/api-integration-tutorial.md
    - Custom Agents: tutorials/custom-agent-tutorial.md
  - Operations:
    - Deployment: operations/production-deployment.md
    - Monitoring: operations/monitoring.md
    - Scaling: operations/scaling.md
    - Security: operations/security.md
    - Backup: operations/backup.md
  - Architecture:
    - System Overview: architecture.md
    - Neural Design: architecture/neural-design.md
    - Performance: architecture/performance.md
    - Scalability: architecture/scalability-plan.md
  - About:
    - Roadmap: roadmap.md
    - Changelog: CHANGELOG.md
    - Contributing: CONTRIBUTING.md
    - License: LICENSE.md

# Extra
extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/claude-tui/claude-tui
    - icon: fontawesome/brands/discord
      link: https://discord.gg/claude-tui
    - icon: fontawesome/brands/twitter  
      link: https://twitter.com/claude_tui
    
  version:
    provider: mike
    default: stable

# Copyright
copyright: Copyright &copy; 2024 Claude-TUI Team

# Analytics
google_analytics:
  - 'G-XXXXXXXXXX'
  - 'auto'
EOF
    
    log_success "Navigation structure generated"
}

# Build HTML documentation
build_html() {
    log_info "Building HTML documentation..."
    
    cd "$BUILD_DIR"
    
    if ! mkdocs build --clean --strict; then
        log_error "MkDocs build failed"
        return 1
    fi
    
    log_success "HTML documentation built successfully"
    cd "$SCRIPT_DIR"
}

# Generate PDF documentation (optional)
generate_pdf() {
    if ! command -v pandoc &> /dev/null; then
        log_warning "Pandoc not available - skipping PDF generation"
        return
    fi
    
    log_info "Generating PDF documentation..."
    
    # Combine all markdown files
    cat > "$BUILD_DIR/complete-guide.md" << 'EOF'
# Claude-TUI: Complete Documentation

## Table of Contents

EOF
    
    # Add all documentation files
    for md_file in $(find "$BUILD_DIR" -name "*.md" -not -name "complete-guide.md" | sort); do
        echo "$(cat "$md_file")" >> "$BUILD_DIR/complete-guide.md"
        echo -e "\n---\n" >> "$BUILD_DIR/complete-guide.md"
    done
    
    # Generate PDF
    pandoc "$BUILD_DIR/complete-guide.md" \
           --pdf-engine=xelatex \
           --toc \
           --toc-depth=3 \
           --number-sections \
           --highlight-style=github \
           -V geometry:margin=1in \
           -V fontsize=11pt \
           -V documentclass=article \
           -o "$BUILD_DIR/claude-tui-documentation.pdf" \
           2>/dev/null || {
        log_warning "PDF generation failed - continuing without PDF"
        return
    }
    
    log_success "PDF documentation generated"
}

# Run auto-documentation generator
run_auto_generator() {
    if [ -f "$SCRIPT_DIR/docs-generator-config.py" ]; then
        log_info "Running auto-documentation generator..."
        
        python3 "$SCRIPT_DIR/docs-generator-config.py" build \
                --source-dir "$SOURCE_DIR" \
                --docs-dir "$DOCS_DIR" \
                --output-dir "$BUILD_DIR/auto-generated" \
                --validate
        
        # Merge auto-generated content
        if [ -d "$BUILD_DIR/auto-generated" ]; then
            cp -r "$BUILD_DIR/auto-generated"/* "$BUILD_DIR/" 2>/dev/null || true
        fi
        
        log_success "Auto-documentation completed"
    else
        log_warning "Auto-documentation generator not found - skipping"
    fi
}

# Deploy documentation (optional)
deploy_docs() {
    local deploy_target="${1:-local}"
    
    case "$deploy_target" in
        "github")
            log_info "Deploying to GitHub Pages..."
            cd "$BUILD_DIR"
            if mkdocs gh-deploy --force; then
                log_success "Deployed to GitHub Pages"
            else
                log_error "GitHub Pages deployment failed"
            fi
            cd "$SCRIPT_DIR"
            ;;
        "netlify")
            log_info "Preparing for Netlify deployment..."
            echo "# Netlify deployment ready" > "$SITE_DIR/_headers"
            echo "/*" >> "$SITE_DIR/_headers"  
            echo "  X-Frame-Options: DENY" >> "$SITE_DIR/_headers"
            echo "  X-XSS-Protection: 1; mode=block" >> "$SITE_DIR/_headers"
            log_success "Netlify deployment files prepared"
            ;;
        "local"|*)
            log_info "Documentation available locally at: file://$SITE_DIR/index.html"
            ;;
    esac
}

# Serve documentation locally
serve_docs() {
    log_info "Starting documentation server..."
    cd "$BUILD_DIR"
    mkdocs serve --dev-addr=0.0.0.0:8000
}

# Main build process
main() {
    local action="${1:-build}"
    local deploy_target="${2:-local}"
    
    log_info "Claude-TUI Documentation Build Pipeline"
    log_info "======================================="
    
    case "$action" in
        "build")
            check_dependencies
            setup_build_env
            copy_existing_docs
            run_auto_generator
            generate_api_docs
            generate_navigation
            build_html
            generate_pdf
            deploy_docs "$deploy_target"
            ;;
        "serve")
            check_dependencies
            setup_build_env
            copy_existing_docs
            generate_navigation
            serve_docs
            ;;
        "clean")
            log_info "Cleaning build directory..."
            rm -rf "$BUILD_DIR"
            log_success "Build directory cleaned"
            ;;
        "deploy")
            deploy_docs "$deploy_target"
            ;;
        *)
            echo "Usage: $0 {build|serve|clean|deploy} [target]"
            echo ""
            echo "Actions:"
            echo "  build   - Build complete documentation"
            echo "  serve   - Build and serve documentation locally"
            echo "  clean   - Clean build directory"
            echo "  deploy  - Deploy to specified target"
            echo ""
            echo "Deploy targets:"
            echo "  local   - Local deployment (default)"
            echo "  github  - GitHub Pages"
            echo "  netlify - Netlify"
            exit 1
            ;;
    esac
}

# Performance monitoring
if [ "$DOCS_BUILD_PROFILE" = "true" ]; then
    exec 3>&2 2> >(tee /tmp/docs-build-profile.log >&3)
    set -x
fi

# Run main function
main "$@"

# Clean up
if [ "$DOCS_BUILD_PROFILE" = "true" ]; then
    set +x
    exec 2>&3 3>&-
    log_info "Profile log saved to /tmp/docs-build-profile.log"
fi

log_success "Documentation build pipeline completed!"
echo ""
echo "📚 Documentation built successfully!"
echo "🌐 HTML: $SITE_DIR/index.html" 
if [ -f "$BUILD_DIR/claude-tui-documentation.pdf" ]; then
    echo "📄 PDF: $BUILD_DIR/claude-tui-documentation.pdf"
fi
echo ""
echo "Next steps:"
echo "  • Review the generated documentation"
echo "  • Test all links and examples"  
echo "  • Deploy to your hosting platform"
echo "  • Share with your team and community"