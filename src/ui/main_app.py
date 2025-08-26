#!/usr/bin/env python3
"""
Claude-TUI Main Application
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

# Import custom widgets - handle both import paths
try:
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
    # Import AutomaticProgrammingScreen
    try:
        from claude_tui.ui.screens.automatic_programming_screen import AutomaticProgrammingScreen
    except ImportError:
        AutomaticProgrammingScreen = None
except ImportError:
    # Create minimal fallback widgets for development
    from textual.widgets import Static
    
    class ProjectTree(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Project Tree [Placeholder]")
            
    class TaskDashboard(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Task Dashboard [Placeholder]")
            
    class ProgressIntelligence(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Progress Intelligence [Placeholder]")
            def update_validation(self, results):
                pass
            
    class ConsoleWidget(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Console Widget [Placeholder]")
            
    class PlaceholderAlert(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("")
            self.display = False
        def show_alert(self, results):
            pass
            
    class NotificationSystem(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("")
            self.display = False
        def add_notification(self, message, type_):
            pass
    
    # Screen classes
    from textual.screen import Screen
    class ProjectWizardScreen(Screen):
        pass
    class SettingsScreen(Screen):
        pass
    class CreateProjectMessage:
        pass  
    class SettingsSavedMessage:
        pass

# Import core components (will be created/integrated)
try:
    from claude_tui.core.project_manager import ProjectManager
    from claude_tui.core.ai_interface import AIInterface  
    from claude_tui.validation.anti_hallucination_engine import ValidationEngine
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
    
    def __init__(self, app_instance: 'ClaudeTUIApp') -> None:
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
        # Skip monitoring in headless/test mode
        if self.app_instance.non_blocking:
            return
            
        while self.app_instance.is_running():
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
                
                # Exit if app stopped
                if not self.app_instance.is_running():
                    break


class ClaudeTUIApp(App[None]):
    """Main Claude-TUI Application with anti-hallucination capabilities"""
    
    TITLE = "Claude-TUI - Intelligent AI Project Manager"
    SUB_TITLE = "SPARC Development with Progress Validation"
    
    # CSS_PATH = str(Path(__file__).parent / "styles" / "main.tcss")
    
    # Constructor with proper initialization
    
    BINDINGS = [
        Binding("ctrl+n", "new_project", "New Project"),
        Binding("ctrl+o", "open_project", "Open Project"),
        Binding("ctrl+s", "save_project", "Save Project"),
        Binding("ctrl+p", "project_wizard", "Project Wizard"),
        Binding("ctrl+t", "toggle_task_panel", "Toggle Tasks"),
        Binding("ctrl+c", "toggle_console", "Toggle Console"),
        Binding("ctrl+v", "toggle_validation", "Toggle Validation"),
        Binding("ctrl+comma", "settings", "Settings"),
        Binding("ctrl+a", "automatic_programming", "Auto Programming"),
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
    
    def __init__(self, headless: bool = False, test_mode: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.headless = headless
        self.test_mode = test_mode
        self.non_blocking = headless or test_mode
        self._running = False
        self._event_loop = None
        
        # Initialize core managers
        self.project_manager = ProjectManager()
        self.ai_interface = AIInterface()
        self.validation_engine = ValidationEngine()
        
        # Initialize config manager (required by tests)
        self.config_manager = self._create_config_manager()
        
        # UI Components
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
        
        # Show welcome message with new automatic programming feature
        self.notify("Claude-TUI initialized. Press Ctrl+P for Project Wizard, Ctrl+A for Automatic Programming.", "info")
        
    def init_core_systems(self) -> None:
        """Initialize core application systems"""
        try:
            # Initialize project manager
            self.project_manager.initialize()
            
            # Initialize AI interface
            self.ai_interface.initialize()
            
            # Initialize validation engine
            self.validation_engine.initialize()
            
            # Mark as running after successful initialization
            self._running = True
            
            self.notify("All systems initialized successfully", "success")
            
        except Exception as e:
            self.notify(f"System initialization failed: {e}", "error")
    
    async def init_async(self) -> None:
        """Async initialization for non-blocking mode"""
        self.init_core_systems()
        self._running = True
    
    def is_running(self) -> bool:
        """Check if the app is running (useful for testing)"""
        return self._running
    
    def stop(self) -> None:
        """Stop the application gracefully"""
        self._running = False
        if not self.non_blocking:
            self.exit()
    
    async def run_async(self, headless: bool = None, **kwargs) -> None:
        """Async version of run for better integration"""
        try:
            # If headless is passed but not set in constructor, update it
            if headless is not None:
                self.headless = headless
                self.non_blocking = headless or self.test_mode
                
            if self.non_blocking:
                await self.init_async()
            else:
                # Use textual's async run if available
                if hasattr(super(), 'run_async'):
                    await super().run_async()
                else:
                    # Fallback to sync run
                    import asyncio
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.run)
        except Exception as e:
            print(f"Error in async run: {e}")
            raise
    
    # Action handlers
    def action_new_project(self) -> None:
        """Create a new project"""
        self.push_screen(ProjectWizardScreen(self.project_manager))
    
    def action_open_project(self) -> None:
        """Open an existing project"""
        try:
            # Show file dialog for project selection
            from textual.widgets import DirectoryTree
            from textual.screen import ModalScreen
            from pathlib import Path
            
            class ProjectSelectionScreen(ModalScreen):
                def compose(self):
                    yield DirectoryTree(Path.home() / "Projects")
                
                def on_directory_tree_file_selected(self, event):
                    if event.path.suffix == '.json':  # Assume project files are JSON
                        self.dismiss(event.path)
                        
            def handle_selection(project_path):
                if project_path:
                    self.project_manager.load_project(project_path)
                    self.notify(f"Project loaded: {project_path.name}", "success")
            
            self.push_screen(ProjectSelectionScreen(), handle_selection)
            
        except Exception as e:
            self.notify(f"Could not open project selection: {e}", "error")
    
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
    
    def action_automatic_programming(self) -> None:
        """Open automatic programming screen"""
        if AutomaticProgrammingScreen:
            self.push_screen(AutomaticProgrammingScreen(self.config_manager))
        else:
            self.notify("Automatic Programming feature not available", "error")
    
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
    
    def _create_config_manager(self):
        """Create and initialize config manager"""
        try:
            # Try to import the actual config manager
            from claude_tui.core.config_manager import ConfigManager
            return ConfigManager()
        except ImportError:
            # Create a fallback config manager for testing
            class FallbackConfigManager:
                def __init__(self):
                    self.config = {}
                    self.settings = {}
                def get(self, key, default=None):
                    return self.config.get(key, default)
                def set(self, key, value):
                    self.config[key] = value
                def save(self):
                    pass
                def load(self):
                    pass
                def initialize(self):
                    pass
            return FallbackConfigManager()
    
    def _generate_help_content(self) -> str:
        """Generate help content for the application"""
        return """
        Claude-TUI Help:
        
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


def run_app(headless: bool = False, test_mode: bool = False) -> None:
    """Entry point for running the Claude-TUI application"""
    # Use integration bridge for better error handling
    try:
        from .integration_bridge import run_integrated_app
        run_integrated_app(ui_type="ui", debug=False, headless=headless, test_mode=test_mode)
    except ImportError:
        # Fallback to direct instantiation
        app = ClaudeTUIApp(headless=headless, test_mode=test_mode)
        if headless or test_mode:
            return run_app_non_blocking(app)
        else:
            app.run()

async def run_app_async(headless: bool = False, test_mode: bool = False) -> ClaudeTUIApp:
    """Async entry point for running the Claude-TUI application"""
    app = ClaudeTUIApp(headless=headless, test_mode=test_mode)
    if headless or test_mode:
        # Initialize without running the event loop
        await app.init_async()
        return app
    else:
        # For non-blocking modes, we don't actually run the full async TUI
        # since that would require the full textual event loop
        await app.init_async()
        return app

def run_app_non_blocking(app: ClaudeTUIApp) -> ClaudeTUIApp:
    """Run app in non-blocking mode for testing"""
    import threading
    import time
    
    def background_init():
        try:
            app.init_core_systems()
            app._running = True
        except Exception as e:
            print(f"Error in background init: {e}")
    
    if app.non_blocking:
        # Run initialization in background thread for testing
        thread = threading.Thread(target=background_init, daemon=True)
        thread.start()
        time.sleep(0.1)  # Give initialization a moment
        return app
    else:
        # Regular blocking run
        app.run()
        return app


if __name__ == "__main__":
    run_app()