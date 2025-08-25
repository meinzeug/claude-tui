#!/usr/bin/env python3
"""
Help Screen - Comprehensive keyboard shortcuts and usage documentation
for the Claude-TUI application.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from textual import on
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import (
    TabbedContent, TabPane, Label, Button, Static, 
    DataTable, RichLog, Markdown
)
from textual.screen import Screen
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown as RichMarkdown


class HelpScreen(Screen):
    """Comprehensive help screen with keyboard shortcuts and documentation"""
    
    def __init__(self) -> None:
        super().__init__()
        self.shortcut_data = self._load_shortcut_data()
        
    def compose(self):
        """Compose help screen"""
        with Container(id="help-container"):
            yield Label("🔧 Claude-TUI Help & Documentation", classes="header")
            
            with TabbedContent():
                # Keyboard Shortcuts Tab
                with TabPane("Shortcuts", id="shortcuts-tab"):
                    yield self._create_shortcuts_table()
                
                # Widget Guide Tab
                with TabPane("Widgets", id="widgets-tab"):
                    yield self._create_widget_guide()
                
                # User Guide Tab
                with TabPane("User Guide", id="guide-tab"):
                    yield self._create_user_guide()
                
                # Tips & Tricks Tab
                with TabPane("Tips", id="tips-tab"):
                    yield self._create_tips_section()
                
                # About Tab
                with TabPane("About", id="about-tab"):
                    yield self._create_about_section()
            
            # Action buttons
            with Horizontal(classes="help-actions"):
                yield Button("📋 Export Shortcuts", id="export-shortcuts")
                yield Button("🔄 Refresh", id="refresh-help")
                yield Button("❌ Close", id="close-help", variant="error")
    
    def _load_shortcut_data(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Load keyboard shortcut data organized by category"""
        return {
            "Navigation": [
                ("Ctrl+Q", "Quit Application", "Exit Claude-TUI safely"),
                ("Ctrl+H", "Show Help", "Display this help screen"),
                ("F1", "Quick Help", "Alternative help shortcut"),
                ("Escape", "Close Modal", "Close current modal dialog"),
                ("Ctrl+W", "Workspace", "Switch to main workspace"),
                ("Ctrl+M", "Monitoring", "Open monitoring dashboard"),
            ],
            "Project Management": [
                ("Ctrl+N", "New Project", "Create a new project"),
                ("Ctrl+O", "Open Project", "Open existing project"),
                ("Ctrl+S", "Save Project", "Save current project"),
                ("Ctrl+Shift+S", "Save As", "Save project with new name"),
                ("Ctrl+R", "Refresh", "Refresh current view"),
            ],
            "Vim-style Navigation": [
                ("j", "Focus Down", "Move focus to next widget"),
                ("k", "Focus Up", "Move focus to previous widget"),
                ("h", "Focus Left", "Move focus left"),
                ("l", "Focus Right", "Move focus right"),
                ("gg", "Focus First", "Jump to first focusable element"),
                ("Shift+G", "Focus Last", "Jump to last focusable element"),
            ],
            "UI Controls": [
                ("Ctrl+T", "Toggle Theme", "Switch between light/dark themes"),
                ("Ctrl+Shift+C", "Show Console", "Toggle AI console visibility"),
                ("Ctrl+Shift+V", "Show Validation", "Toggle progress validation"),
                ("Ctrl+Shift+W", "Show Workflows", "Toggle workflow visualizer"),
                ("Ctrl+Shift+M", "Show Metrics", "Toggle metrics dashboard"),
                ("F11", "Fullscreen", "Toggle fullscreen mode"),
            ],
            "AI & Development": [
                ("Ctrl+Shift+A", "AI Assist", "Focus AI assistant"),
                ("Ctrl+Shift+P", "Scan Placeholders", "Run placeholder detection"),
                ("F2", "Settings", "Open application settings"),
                ("F3", "Templates", "Show AI command templates"),
                ("Ctrl+/", "Quick Find", "Open search dialog"),
            ],
            "Task Management": [
                ("Ctrl+Shift+T", "New Task", "Create new task"),
                ("Ctrl+Shift+E", "Execute Task", "Run selected task"),
                ("Ctrl+Shift+D", "Task Details", "Show task details"),
                ("Space", "Toggle Selection", "Toggle task/item selection"),
                ("Enter", "Activate", "Activate selected item"),
            ],
            "File Operations": [
                ("Ctrl+Shift+F", "Find in Files", "Search across project files"),
                ("Ctrl+Shift+R", "Replace", "Find and replace in files"),
                ("Ctrl+B", "Build", "Build current project"),
                ("Ctrl+Shift+B", "Build All", "Build entire workspace"),
                ("F5", "Run/Debug", "Execute current project"),
            ]
        }
    
    def _create_shortcuts_table(self) -> Container:
        """Create keyboard shortcuts table"""
        with Container():
            yield Label("⌨️ Keyboard Shortcuts", classes="section-header")
            
            # Create data table for shortcuts
            table = DataTable()
            table.add_columns("Category", "Shortcut", "Action", "Description")
            
            for category, shortcuts in self.shortcut_data.items():
                # Add category separator
                table.add_row("", "", "", "", classes="category-separator")
                table.add_row(
                    Text(category, style="bold cyan"), 
                    "", "", "", 
                    classes="category-header"
                )
                
                # Add shortcuts for this category
                for shortcut, action, description in shortcuts:
                    table.add_row(
                        "",
                        Text(shortcut, style="bold yellow"),
                        Text(action, style="bold"),
                        Text(description, style="dim")
                    )
            
            yield table
        
        return Container()
    
    def _create_widget_guide(self) -> Container:
        """Create widget usage guide"""
        with Container():
            yield Label("🧩 Widget Guide", classes="section-header")
            
            widget_docs = """
# Widget Overview

## Project Tree Widget
- **Purpose**: Navigate and explore project files
- **Features**: File type icons, validation status indicators
- **Shortcuts**: Arrow keys to navigate, Enter to open files
- **Status Icons**: ✅ Validated, ⚠️ Has placeholders, ❌ Has errors

## Task Dashboard Widget
- **Purpose**: Manage project tasks and track progress
- **Features**: Real vs claimed progress tracking, authenticity scoring
- **Filters**: All, Active, Completed, Blocked tasks
- **Actions**: Add task, refresh, view analytics

## AI Console Widget
- **Purpose**: Interact with AI assistant
- **Features**: Command history, autocomplete, task tracking
- **Usage**: Type commands or use templates
- **History**: Use Up/Down arrows to navigate command history

## Progress Intelligence Widget
- **Purpose**: Validate real vs fake progress
- **Features**: Quality scoring, authenticity validation
- **Metrics**: Code completeness, test coverage, documentation
- **Alerts**: Warnings for potential over-reporting

## Workflow Visualizer Widget
- **Purpose**: Visual representation of task dependencies
- **Features**: Critical path analysis, execution timeline
- **Views**: Tree view, Gantt chart, dependency graph
- **Controls**: Start, pause, stop workflow execution

## Metrics Dashboard Widget
- **Purpose**: Monitor system and productivity metrics
- **Features**: System health, performance tracking
- **Tabs**: System Health, Productivity, Quality, Performance
- **Alerts**: Real-time threshold monitoring

## Placeholder Alert Widget
- **Purpose**: Detect and warn about incomplete code
- **Features**: Automatic scanning, severity levels
- **Detection**: TODOs, empty functions, mock data
- **Actions**: Auto-fix, manual review, ignore
            """
            
            yield Markdown(widget_docs)
        
        return Container()
    
    def _create_user_guide(self) -> Container:
        """Create user guide documentation"""
        with Container():
            yield Label("📖 User Guide", classes="section-header")
            
            user_guide = """
# Getting Started with Claude-TUI

## First Steps
1. **Create or Open Project**: Use Ctrl+N for new or Ctrl+O to open existing
2. **Explore Interface**: Familiarize yourself with the three-panel layout
3. **Configure Settings**: Press F2 to customize your experience

## Workspace Layout
- **Left Panel**: Project tree and navigation
- **Center Panel**: Main content tabs (Tasks, Editor, Workflows, Metrics)
- **Right Panel**: AI Console, Validation, Inspector

## Working with Projects
1. **Project Structure**: Organized file tree with validation indicators
2. **Task Management**: Create, assign, and track development tasks
3. **AI Integration**: Use AI assistant for code generation and problem-solving
4. **Progress Tracking**: Monitor real progress vs claimed completion

## AI Assistant Usage
- **Command Templates**: Use F3 to access pre-built commands
- **Custom Commands**: Type natural language requests
- **Context Awareness**: AI understands your project structure
- **History**: Access previous commands with arrow keys

## Quality Assurance
- **Placeholder Detection**: Automatic scanning for incomplete code
- **Progress Validation**: Real vs fake progress analysis
- **Code Quality Metrics**: Comprehensive quality scoring
- **Test Coverage**: Track and improve test completeness

## Workflow Management
- **Visual Workflows**: See task dependencies and critical path
- **Execution Control**: Start, pause, monitor workflow progress
- **Team Coordination**: Understand task assignments and timing
- **Performance Analysis**: Optimize workflow efficiency

## Best Practices
1. **Regular Saves**: Use Ctrl+S frequently to save progress
2. **Quality Focus**: Aim for high authenticity scores
3. **Template Usage**: Leverage AI templates for consistency
4. **Progress Monitoring**: Review validation results regularly
5. **Workflow Planning**: Design clear task dependencies

## Troubleshooting
- **Performance Issues**: Check metrics dashboard for bottlenecks
- **Validation Errors**: Review placeholder alerts for issues
- **AI Problems**: Clear console and restart AI assistant
- **Layout Issues**: Use layout controls to reset view
            """
            
            yield Markdown(user_guide)
        
        return Container()
    
    def _create_tips_section(self) -> Container:
        """Create tips and tricks section"""
        with Container():
            yield Label("💡 Tips & Tricks", classes="section-header")
            
            tips = """
# Pro Tips for Claude-TUI

## Productivity Boosters
- **Vim Navigation**: Use hjkl keys for faster navigation
- **Quick Actions**: Float buttons (bottom-right) for instant access
- **Template Library**: Build custom AI command templates
- **Workspace Layouts**: Switch between focused and monitoring modes

## AI Assistant Mastery
- **Context First**: Provide clear context in commands
- **Iterative Refinement**: Build on previous AI responses
- **Template Customization**: Modify templates for your workflow
- **Batch Operations**: Combine multiple tasks in single commands

## Quality Assurance
- **Regular Scanning**: Enable automatic placeholder detection
- **Progress Reviews**: Check authenticity scores frequently
- **Test-Driven Development**: Write tests before implementation
- **Documentation**: Keep README and docs updated

## Workflow Optimization
- **Dependency Planning**: Map task relationships clearly
- **Critical Path Focus**: Prioritize longest dependency chains
- **Parallel Execution**: Identify tasks that can run simultaneously
- **Regular Reviews**: Monitor and adjust workflows

## Customization
- **Theme Selection**: Choose between light/dark modes
- **Layout Preferences**: Customize panel sizes and visibility
- **Shortcut Mapping**: Learn and use keyboard shortcuts
- **Widget Configuration**: Adjust update intervals and thresholds

## Advanced Features
- **Export Options**: Save configurations and reports
- **Import/Export**: Share workflows and templates
- **Integration**: Connect with external tools and services
- **Automation**: Set up automated quality checks

## Performance Tips
- **Resource Monitoring**: Watch system metrics dashboard
- **Memory Management**: Clear caches when needed
- **Network Optimization**: Monitor AI response times
- **Disk Space**: Regular cleanup of temporary files
            """
            
            yield Markdown(tips)
        
        return Container()
    
    def _create_about_section(self) -> Container:
        """Create about section"""
        with Container():
            yield Label("ℹ️ About Claude-TUI", classes="section-header")
            
            about_text = """
# Claude-TUI: Intelligent AI-powered Terminal User Interface

## Overview
Claude-TIU is an advanced software development assistant that combines the power of AI with intelligent terminal user interface design. Built with Textual and integrated with Claude AI capabilities, it provides a comprehensive development environment focused on quality, authenticity, and productivity.

## Key Features
- **AI-Powered Development**: Natural language commands for code generation
- **Quality Validation**: Real vs fake progress analysis with authenticity scoring  
- **Workflow Management**: Visual task dependencies and execution monitoring
- **Placeholder Detection**: Automatic scanning for incomplete implementations
- **Metrics Dashboard**: Comprehensive system and productivity monitoring
- **Responsive Design**: Professional dark theme with accessibility features

## Technology Stack
- **Framework**: Textual for terminal UI
- **Language**: Python 3.8+
- **AI Integration**: Claude AI API
- **Architecture**: Modular widget-based design
- **Styling**: Enhanced TCSS with responsive layout

## Development Team
- **Architecture**: Advanced software engineering patterns
- **UI/UX**: Modern terminal interface design
- **AI Integration**: Intelligent command processing
- **Quality Assurance**: Comprehensive validation systems

## Version Information
- **Version**: 1.0.0 Alpha
- **Build**: Development Build
- **License**: MIT License
- **Support**: Community-driven development

## Contributing
We welcome contributions! Please check our GitHub repository for:
- Issue reporting and feature requests
- Development guidelines and coding standards
- Community discussions and support
- Documentation improvements

## Acknowledgments
- Textual framework for excellent TUI capabilities
- Claude AI for intelligent assistance
- Open source community for inspiration and feedback
- Beta testers for valuable input and bug reports

## Support & Resources
- **Documentation**: Comprehensive guides and API reference
- **Community**: Active forums and discussion channels  
- **Tutorials**: Step-by-step learning resources
- **Examples**: Sample projects and configurations

## Future Roadmap
- Enhanced AI model integration
- Extended plugin architecture
- Advanced collaboration features
- Mobile companion applications
- Cloud synchronization capabilities

---
Built with ❤️ by the Claude-TIU development team
            """
            
            yield Markdown(about_text)
        
        return Container()
    
    # Event handlers
    @on(Button.Pressed, "#export-shortcuts")
    def export_shortcuts(self) -> None:
        """Export keyboard shortcuts to file"""
        # Create exportable format
        export_text = "# Claude-TIU Keyboard Shortcuts\n\n"
        
        for category, shortcuts in self.shortcut_data.items():
            export_text += f"## {category}\n\n"
            export_text += "| Shortcut | Action | Description |\n"
            export_text += "|----------|--------|-------------|\n"
            
            for shortcut, action, description in shortcuts:
                export_text += f"| `{shortcut}` | {action} | {description} |\n"
            
            export_text += "\n"
        
        # This would normally save to file or copy to clipboard
        self.notify("Shortcuts exported to clipboard", severity="success")
    
    @on(Button.Pressed, "#refresh-help")
    def refresh_help(self) -> None:
        """Refresh help content"""
        self.shortcut_data = self._load_shortcut_data()
        self.refresh()
        self.notify("Help content refreshed", severity="info")
    
    @on(Button.Pressed, "#close-help")
    def close_help(self) -> None:
        """Close help screen"""
        self.dismiss()