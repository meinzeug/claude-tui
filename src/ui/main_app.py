#!/usr/bin/env python3
"""
Claude-TIU Main Application
Intelligent AI-powered Terminal User Interface for project management
with anti-hallucination and progress validation capabilities.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Static, Button, Input, 
    DirectoryTree, RichLog, ProgressBar, Label
)
from textual.message import Message
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Import custom widgets
from .widgets.project_tree import ProjectTree
from .widgets.task_dashboard import TaskDashboard
from .widgets.progress_intelligence import ProgressIntelligence
from .widgets.console_widget import ConsoleWidget
from .widgets.placeholder_alert import PlaceholderAlert
from .widgets.notification_system import NotificationSystem
from .screens import (
    ProjectWizardScreen, SettingsScreen, CreateProjectMessage,
    SettingsSavedMessage
)

# Import core components (will be created/integrated)
try:
    from ..core.project_manager import ProjectManager
    from ..core.ai_interface import AIInterface  
    from ..core.validation_engine import ValidationEngine
except ImportError:
    # Fallback imports for development
    class ProjectManager:
        def __init__(self):
            self.current_project = None
        def initialize(self):
            pass
        def save_current_project(self):
            pass
    
    class AIInterface:
        def __init__(self):
            pass
        def initialize(self):
            pass
        async def execute_task(self, task_description, context):
            return f"Mock AI result for: {task_description}"
    
    class ValidationEngine:
        def __init__(self):
            pass
        def initialize(self):
            pass
        async def analyze_project(self, project_path):
            from .widgets.progress_intelligence import ProgressReport
            return ProgressReport(
                real_progress=0.7,
                claimed_progress=0.9,
                fake_progress=0.2,
                quality_score=7.5,
                authenticity_score=0.78,
                placeholders_found=3,
                todos_found=5
            )


class MainWorkspace(Container):
    """Main workspace container with responsive layout"""
    
    def __init__(self, app_instance: 'ClaudeTIUApp') -> None:
        super().__init__()
        self.app_instance = app_instance
        self.project_tree: Optional[ProjectTree] = None
        self.task_dashboard: Optional[TaskDashboard] = None
        self.progress_widget: Optional[ProgressIntelligence] = None
        self.console_widget: Optional[ConsoleWidget] = None
        
    def compose(self) -> ComposeResult:
        """Compose the main workspace layout"""
        with Horizontal():
            # Left panel - Project Explorer & Validation
            with Vertical(classes="left-panel"):
                self.project_tree = ProjectTree(self.app_instance.project_manager)
                yield self.project_tree
                
                self.progress_widget = ProgressIntelligence()
                yield self.progress_widget
                
            # Right panel - Main workspace
            with Vertical(classes="main-panel"):
                # Task dashboard
                self.task_dashboard = TaskDashboard(self.app_instance.project_manager)
                yield self.task_dashboard
                
                # Console for AI interaction
                self.console_widget = ConsoleWidget(self.app_instance.ai_interface)
                yield self.console_widget

    def on_mount(self) -> None:
        """Initialize workspace after mounting"""
        self.start_progress_monitoring()
        
    @work(exclusive=True)
    async def start_progress_monitoring(self) -> None:
        """Start continuous progress monitoring"""
        while True:
            try:
                if self.app_instance.project_manager.current_project:
                    # Update progress intelligence
                    validation_results = await self.app_instance.validation_engine.analyze_project(
                        self.app_instance.project_manager.current_project.path
                    )
                    
                    if self.progress_widget:
                        self.progress_widget.update_validation(validation_results)
                    
                    # Check for placeholders and alert if found
                    if validation_results.fake_progress > 20:  # More than 20% fake progress
                        self.app_instance.show_placeholder_alert(validation_results)
                        
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.app_instance.notify(f"Progress monitoring error: {e}", "error")
                await asyncio.sleep(60)  # Wait longer on errors


class ClaudeTIUApp(App[None]):
    """Main Claude-TIU Application with anti-hallucination capabilities"""
    
    TITLE = "Claude-TIU - Intelligent AI Project Manager"
    SUB_TITLE = "SPARC Development with Progress Validation"
    
    CSS_PATH = "styles/main.css"
    
    BINDINGS = [
        Binding("ctrl+n", "new_project", "New Project"),
        Binding("ctrl+o", "open_project", "Open Project"),
        Binding("ctrl+s", "save_project", "Save Project"),
        Binding("ctrl+p", "project_wizard", "Project Wizard"),
        Binding("ctrl+t", "toggle_task_panel", "Toggle Tasks"),
        Binding("ctrl+c", "toggle_console", "Toggle Console"),
        Binding("ctrl+v", "toggle_validation", "Toggle Validation"),
        Binding("ctrl+comma", "settings", "Settings"),
        Binding("f1", "help", "Help"),
        Binding("f5", "refresh", "Refresh"),
        Binding("f12", "debug_mode", "Debug"),
        Binding("ctrl+q", "quit", "Quit"),
        # Vim-style navigation
        Binding("h", "focus_left", "Focus Left", show=False),
        Binding("j", "focus_down", "Focus Down", show=False),
        Binding("k", "focus_up", "Focus Up", show=False),
        Binding("l", "focus_right", "Focus Right", show=False),
    ]
    
    def __init__(self) -> None:
        super().__init__()
        self.project_manager = ProjectManager()
        self.ai_interface = AIInterface()
        self.validation_engine = ValidationEngine()
        self.notification_system: Optional[NotificationSystem] = None
        self.placeholder_alert: Optional[PlaceholderAlert] = None
        self.workspace: Optional[MainWorkspace] = None
        
        # Application state
        self.current_screen = "main"
        self.debug_mode = False
        self.validation_enabled = True
        
    def compose(self) -> ComposeResult:
        """Compose the main application layout"""
        yield Header(show_clock=True)
        
        # Main workspace
        self.workspace = MainWorkspace(self)
        yield self.workspace
        
        # Notification system (overlay)
        self.notification_system = NotificationSystem()
        yield self.notification_system
        
        # Placeholder alert system (overlay)
        self.placeholder_alert = PlaceholderAlert()
        yield self.placeholder_alert
        
        yield Footer()
        
    def on_mount(self) -> None:
        """Initialize application after mounting"""
        self.title = self.TITLE
        self.sub_title = self.SUB_TITLE
        
        # Initialize core systems
        self.init_core_systems()
        
        # Show welcome message
        self.notify("Claude-TIU initialized. Press Ctrl+P for Project Wizard.", "info")
        
    def init_core_systems(self) -> None:
        """Initialize core application systems"""
        try:
            # Initialize project manager
            self.project_manager.initialize()
            
            # Initialize AI interface
            self.ai_interface.initialize()
            
            # Initialize validation engine
            self.validation_engine.initialize()
            
            self.notify("All systems initialized successfully", "success")
            
        except Exception as e:
            self.notify(f"System initialization failed: {e}", "error")
    
    # Action handlers
    def action_new_project(self) -> None:
        """Create a new project"""
        self.push_screen(ProjectWizardScreen(self.project_manager))
    
    def action_open_project(self) -> None:
        """Open an existing project"""
        # TODO: Implement project selection dialog
        self.notify("Open project functionality coming soon", "info")
    
    def action_save_project(self) -> None:
        """Save current project"""
        if self.project_manager.current_project:
            self.project_manager.save_current_project()
            self.notify("Project saved successfully", "success")
        else:
            self.notify("No project to save", "warning")
    
    def action_project_wizard(self) -> None:
        """Launch project wizard"""
        self.push_screen(ProjectWizardScreen(self.project_manager))
    
    def action_settings(self) -> None:
        """Open settings screen"""
        self.push_screen(SettingsScreen())
    
    def action_help(self) -> None:
        """Show help information"""
        help_content = self._generate_help_content()
        self.notify("Help: Use Ctrl+? for detailed help", "info")
    
    def action_refresh(self) -> None:
        """Refresh all widgets and data"""
        if self.workspace:
            if self.workspace.project_tree:
                self.workspace.project_tree.refresh()
            if self.workspace.task_dashboard:
                self.workspace.task_dashboard.refresh()
        self.notify("Interface refreshed", "info")
    
    def action_debug_mode(self) -> None:
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        status = "enabled" if self.debug_mode else "disabled"
        self.notify(f"Debug mode {status}", "info")
    
    # Vim-style navigation
    def action_focus_left(self) -> None:
        """Focus widget to the left"""
        self.screen.focus_previous()
    
    def action_focus_right(self) -> None:
        """Focus widget to the right"""
        self.screen.focus_next()
    
    def action_focus_up(self) -> None:
        """Focus widget above"""
        # Implementation depends on current focus context
        pass
    
    def action_focus_down(self) -> None:
        """Focus widget below"""
        # Implementation depends on current focus context
        pass
    
    def action_toggle_task_panel(self) -> None:
        """Toggle task dashboard visibility"""
        if self.workspace and self.workspace.task_dashboard:
            current_display = self.workspace.task_dashboard.display
            self.workspace.task_dashboard.display = not current_display
    
    def action_toggle_console(self) -> None:
        """Toggle console widget visibility"""
        if self.workspace and self.workspace.console_widget:
            current_display = self.workspace.console_widget.display
            self.workspace.console_widget.display = not current_display
    
    def action_toggle_validation(self) -> None:
        """Toggle validation system"""
        self.validation_enabled = not self.validation_enabled
        status = "enabled" if self.validation_enabled else "disabled"
        self.notify(f"Validation system {status}", "info")
    
    # Core functionality
    def notify(self, message: str, notification_type: str = "info") -> None:
        """Send notification to user"""
        if self.notification_system:
            self.notification_system.add_notification(message, notification_type)
        else:
            # Fallback to console output
            print(f"[{notification_type.upper()}] {message}")
    
    def show_placeholder_alert(self, validation_results) -> None:
        """Show placeholder detection alert"""
        if self.placeholder_alert and validation_results.placeholders_found:
            self.placeholder_alert.show_alert(validation_results)
    
    @work(exclusive=True)
    async def execute_ai_task(self, task_description: str, context: Dict[str, Any]) -> None:
        """Execute AI task with validation"""
        try:
            # Execute AI task
            result = await self.ai_interface.execute_task(task_description, context)
            
            # Validate result if validation is enabled
            if self.validation_enabled:
                validation = await self.validation_engine.validate_ai_output(
                    result, context
                )
                
                if not validation.is_authentic:
                    # Trigger auto-completion workflow
                    completed_result = await self.ai_interface.complete_placeholder_code(
                        result, validation.completion_suggestions
                    )
                    result = completed_result
            
            # Update UI with results
            self.notify("AI task completed successfully", "success")
            
        except Exception as e:
            self.notify(f"AI task failed: {e}", "error")
    
    # Handle messages from screens and widgets
    @on(CreateProjectMessage)
    def handle_create_project(self, message: CreateProjectMessage) -> None:
        """Handle project creation from wizard"""
        try:
            # Create project using project manager
            result = self.project_manager.create_project_from_config(message.config)
            if result:
                self.notify(f"Project '{message.config.name}' created successfully!", "success")
                # Update UI to show new project
                if self.workspace and self.workspace.project_tree:
                    self.workspace.project_tree.set_project(str(message.config.path))
            else:
                self.notify("Failed to create project", "error")
        except Exception as e:
            self.notify(f"Project creation failed: {e}", "error")
    
    @on(SettingsSavedMessage)
    def handle_settings_saved(self, message: SettingsSavedMessage) -> None:
        """Handle settings saved"""
        self.notify("Settings saved successfully", "success")
        # Apply new settings
        self._apply_settings(message.settings)
    
    def _apply_settings(self, settings) -> None:
        """Apply new settings to the application"""
        # Update validation settings
        self.validation_enabled = settings.validation_enabled
        
        # Update theme (would need theme system implementation)
        # self.theme = settings.theme
        
        # Update other application settings
        if hasattr(settings, 'debug_mode'):
            self.debug_mode = settings.debug_mode
    
    def _generate_help_content(self) -> str:
        """Generate help content for the application"""
        return """
        Claude-TIU Help:
        
        Navigation:
        - h/j/k/l: Vim-style navigation
        - Tab/Shift+Tab: Focus next/previous
        - Enter: Activate focused element
        
        Project Management:
        - Ctrl+N: New project
        - Ctrl+O: Open project  
        - Ctrl+P: Project wizard
        - Ctrl+S: Save project
        
        Interface:
        - Ctrl+T: Toggle task panel
        - Ctrl+C: Toggle console
        - Ctrl+V: Toggle validation
        - F5: Refresh interface
        
        System:
        - F1: Help
        - F12: Debug mode
        - Ctrl+,: Settings
        - Ctrl+Q: Quit
        """


def run_app() -> None:
    """Entry point for running the Claude-TIU application"""
    app = ClaudeTIUApp()
    app.run()


if __name__ == "__main__":
    run_app()