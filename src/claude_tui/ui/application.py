"""
Main Textual Application - The primary TUI application class.

Provides the main application framework with screen management,
keyboard shortcuts, and integration with core components.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button, Footer, Header, Input, Label, Log, 
    ProgressBar, Static, Tree
)
from textual import events
from rich.console import Console
from rich.text import Text

from claude_tui.core.config_manager import ConfigManager
from claude_tui.core.project_manager import ProjectManager
from claude_tui.core.task_engine import TaskEngine
from claude_tui.core.progress_validator import ProgressValidator
from claude_tui.integrations.ai_interface import AIInterface
try:
    from claude_tui.ui.screens import (
        WelcomeScreen, ProjectSetupScreen, MonitoringScreen
    )
    from claude_tui.ui.screens.workspace_screen import WorkspaceScreen
except ImportError:
    # Create fallback screens
    from textual.screen import Screen
    class WelcomeScreen(Screen):
        def compose(self):
            from textual.widgets import Static
            yield Static("Welcome to Claude-TUI")
    
    class ProjectSetupScreen(Screen):
        def setup_for_new_project(self):
            pass
        def compose(self):
            from textual.widgets import Static
            yield Static("Project Setup")
    
    class MonitoringScreen(Screen):
        def compose(self):
            from textual.widgets import Static
            yield Static("Monitoring Dashboard")
    
    class WorkspaceScreen(Screen):
        async def set_project(self, project):
            pass
        def compose(self):
            from textual.widgets import Static
            yield Static("Workspace")
from claude_tui.models.project import Project

# Import UI widgets with fallbacks
try:
    from claude_tui.ui.widgets.project_tree import ProjectTree
    from claude_tui.ui.widgets.task_dashboard import TaskDashboard
    from claude_tui.ui.widgets.console_widget import ConsoleWidget
    from claude_tui.ui.widgets.placeholder_alert import PlaceholderAlert
    from claude_tui.ui.widgets.progress_intelligence import ProgressIntelligence
    from claude_tui.ui.widgets.workflow_visualizer import WorkflowVisualizerWidget
    from claude_tui.ui.widgets.metrics_dashboard import MetricsDashboardWidget
    from claude_tui.ui.widgets.modal_dialogs import ConfigurationModal, CommandTemplatesModal
except ImportError:
    # Create basic fallback widgets
    from textual.widgets import Static
    
    class ProjectTree(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Project Tree [Loading...]")
    
    class TaskDashboard(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Task Dashboard [Loading...]")
    
    class ConsoleWidget(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Console [Loading...]")
    
    class PlaceholderAlert(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("")
    
    class ProgressIntelligence(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Progress Intelligence [Loading...]")
    
    class WorkflowVisualizerWidget(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Workflow Visualizer [Loading...]")
    
    class MetricsDashboardWidget(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Metrics Dashboard [Loading...]")
    
    class ConfigurationModal(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Configuration [Loading...]")
    
    class CommandTemplatesModal(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Command Templates [Loading...]")

# Import theme manager (create if doesn't exist)
try:
    from claude_tui.utils.theme_manager import ThemeManager
except ImportError:
    # Create a basic theme manager if it doesn't exist
    class ThemeManager:
        def __init__(self, config_manager):
            self.config_manager = config_manager
        
        async def get_theme(self, theme_name):
            return {"name": theme_name}

logger = logging.getLogger(__name__)


class ClaudeTUIApp(App):
    """
    Main Textual application for Claude-TUI.
    
    Provides the primary user interface with screen management,
    keyboard shortcuts, and integration with all core components.
    """
    
    TITLE = "Claude-TUI: Intelligent AI-powered Terminal User Interface"
    SUB_TITLE = "Advanced Software Development Assistant"
    
    # Application-wide key bindings
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+h", "help", "Help"),
        Binding("ctrl+p", "show_projects", "Projects"),
        Binding("ctrl+n", "new_project", "New Project"),
        Binding("ctrl+o", "open_project", "Open Project"),
        Binding("ctrl+s", "save_project", "Save Project"),
        Binding("ctrl+w", "workspace", "Workspace"),
        Binding("ctrl+m", "monitoring", "Monitoring"),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme"),
        Binding("f1", "help", "Help"),
        Binding("f2", "settings", "Settings"),
        Binding("f5", "refresh", "Refresh"),
        Binding("f11", "toggle_fullscreen", "Fullscreen"),
    ]
    
    # CSS styling for the application - load from external file
    CSS_PATH = "src/ui/styles/enhanced.tcss"
    
    def __init__(
        self,
        config_manager: ConfigManager,
        debug: bool = False,
        initial_project_dir: Optional[Path] = None
    ):
        """
        Initialize the ClaudeTUI application.
        
        Args:
            config_manager: Configuration management instance
            debug: Enable debug mode
            initial_project_dir: Initial project directory to load
        """
        super().__init__()
        
        # Core components
        self.config_manager = config_manager
        self.project_manager = ProjectManager(config_manager)
        self.task_engine = TaskEngine(config_manager)
        self.ai_interface = AIInterface(config_manager)
        self.progress_validator = ProgressValidator(config_manager)
        
        # UI components
        self.theme_manager = ThemeManager(config_manager)
        
        # Application state
        self.debug = debug
        self.initial_project_dir = initial_project_dir
        self.current_project: Optional[Project] = None
        
        # Screen instances
        self.welcome_screen = WelcomeScreen()
        self.project_setup_screen = ProjectSetupScreen()
        self.workspace_screen = WorkspaceScreen()
        self.monitoring_screen = MonitoringScreen()
        
        # Runtime state
        self._background_tasks: set = set()
        self._status_message = ""
        self._notifications: List[Dict[str, Any]] = []
        
        logger.info("ClaudeTUI application initialized")
    
    async def on_mount(self) -> None:
        """
        Called when the application is mounted.
        """
        logger.info("Mounting ClaudeTUI application")
        
        try:
            # Initialize core components
            await self.task_engine.initialize()
            await self.project_manager.initialize() if hasattr(self.project_manager, 'initialize') else None
            
            # Apply theme
            await self._apply_theme()
            
            # Load initial project if specified
            if self.initial_project_dir:
                try:
                    project = await self.project_manager.load_project(self.initial_project_dir)
                    self.current_project = project
                    await self.push_screen(self.workspace_screen)
                    self._set_status(f"Loaded project: {project.name}")
                except Exception as e:
                    logger.warning(f"Failed to load initial project: {e}")
                    await self.push_screen(self.welcome_screen)
            else:
                # Show welcome screen
                await self.push_screen(self.welcome_screen)
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("ClaudeTUI application mounted successfully")
            
        except Exception as e:
            logger.error(f"Failed to mount application: {e}")
            await self.action_quit()
    
    def compose(self) -> ComposeResult:
        """
        Compose the main application layout.
        """
        yield Header()
        
        with Container(classes="main-container"):
            # Main content area - screens will be pushed here
            yield Static("Initializing Claude-TUI...", id="main-content")
        
        # Status bar
        with Container(classes="status-bar"):
            yield Static(self._status_message, id="status")
        
        yield Footer()
    
    async def on_ready(self) -> None:
        """
        Called when the application is ready.
        """
        self._set_status("Claude-TUI ready - Press Ctrl+H for help")
    
    # Action handlers for key bindings
    
    async def action_quit(self) -> None:
        """
        Quit the application.
        """
        logger.info("Shutting down Claude-TUI application")
        
        try:
            # Save current project if exists
            if self.current_project:
                await self.project_manager.save_project(self.current_project)
            
            # Cleanup components
            await self._cleanup_background_tasks()
            await self.task_engine.cleanup()
            await self.project_manager.cleanup()
            await self.ai_interface.cleanup()
            
            logger.info("Application cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during application cleanup: {e}")
        
        finally:
            self.exit()
    
    async def action_help(self) -> None:
        """
        Show help information.
        """
        from claude_tui.ui.screens.help_screen import HelpScreen
        help_screen = HelpScreen()
        await self.push_screen(help_screen)
    
    async def action_new_project(self) -> None:
        """
        Create a new project.
        """
        self.project_setup_screen.setup_for_new_project()
        await self.push_screen(self.project_setup_screen)
    
    async def action_open_project(self) -> None:
        """
        Open an existing project.
        """
        from claude_tui.ui.screens.project_browser_screen import ProjectBrowserScreen
        browser_screen = ProjectBrowserScreen()
        await self.push_screen(browser_screen)
    
    async def action_save_project(self) -> None:
        """
        Save the current project.
        """
        if self.current_project:
            try:
                await self.project_manager.save_project(self.current_project)
                self._set_status(f"Project '{self.current_project.name}' saved")
                await self._show_notification("Project saved successfully", "success")
            except Exception as e:
                logger.error(f"Failed to save project: {e}")
                await self._show_notification(f"Save failed: {e}", "error")
        else:
            await self._show_notification("No project to save", "warning")
    
    async def action_workspace(self) -> None:
        """
        Show the main workspace.
        """
        if self.current_project:
            await self.switch_screen(self.workspace_screen)
        else:
            await self._show_notification("No project loaded. Create or open a project first.", "warning")
    
    async def action_monitoring(self) -> None:
        """
        Show the monitoring dashboard.
        """
        await self.push_screen(self.monitoring_screen)
    
    async def action_toggle_theme(self) -> None:
        """
        Toggle between light and dark themes.
        """
        current_theme = await self.config_manager.get_setting('ui_preferences.theme', 'dark')
        new_theme = 'light' if current_theme == 'dark' else 'dark'
        
        await self.config_manager.update_setting('ui_preferences.theme', new_theme)
        await self._apply_theme()
        
        self._set_status(f"Theme changed to {new_theme}")
    
    async def action_settings(self) -> None:
        """
        Show application settings.
        """
        from claude_tui.ui.screens.settings_screen import SettingsScreen
        settings_screen = SettingsScreen()
        await self.push_screen(settings_screen)
    
    async def action_refresh(self) -> None:
        """
        Refresh the current view.
        """
        current_screen = self.screen
        if hasattr(current_screen, 'refresh'):
            await current_screen.refresh()
        
        self._set_status("View refreshed")
    
    async def action_toggle_fullscreen(self) -> None:
        """
        Toggle fullscreen mode (if supported by terminal).
        """
        # This would interact with terminal capabilities if available
        self._set_status("Fullscreen toggle requested")
    
    # Public methods for screen interaction
    
    async def set_current_project(self, project: Project) -> None:
        """
        Set the current project and update UI.
        
        Args:
            project: The project to set as current
        """
        self.current_project = project
        self._set_status(f"Project loaded: {project.name}")
        
        # Update workspace screen with new project
        if hasattr(self.workspace_screen, 'set_project'):
            await self.workspace_screen.set_project(project)
        
        logger.info(f"Current project set to: {project.name}")
    
    async def show_notification(self, message: str, type_: str = "info") -> None:
        """
        Show a notification to the user.
        
        Args:
            message: Notification message
            type_: Notification type (info, success, warning, error)
        """
        await self._show_notification(message, type_)
    
    def get_current_project(self) -> Optional[Project]:
        """
        Get the currently loaded project.
        
        Returns:
            Current project or None if no project is loaded
        """
        return self.current_project
    
    # Private helper methods
    
    def _set_status(self, message: str) -> None:
        """
        Set the status bar message.
        
        Args:
            message: Status message to display
        """
        self._status_message = message
        status_widget = self.query_one("#status", Static)
        if status_widget:
            status_widget.update(message)
        
        logger.debug(f"Status: {message}")
    
    async def _show_notification(
        self, 
        message: str, 
        type_: str = "info",
        duration: int = 5
    ) -> None:
        """
        Show a temporary notification.
        
        Args:
            message: Notification message
            type_: Notification type
            duration: Duration in seconds to show notification
        """
        notification = {
            'message': message,
            'type': type_,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self._notifications.append(notification)
        
        # For now, just update status. In a full implementation,
        # this would show a proper notification widget
        status_prefix = {
            'info': '💬',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }.get(type_, '💬')
        
        self._set_status(f"{status_prefix} {message}")
        
        # Clear notification after duration
        async def clear_notification():
            await asyncio.sleep(duration)
            if self._status_message == f"{status_prefix} {message}":
                self._set_status("Ready")
        
        task = asyncio.create_task(clear_notification())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _apply_theme(self) -> None:
        """
        Apply the current theme to the application.
        """
        ui_prefs = self.config_manager.get_ui_preferences()
        theme_name = ui_prefs.theme
        
        theme = await self.theme_manager.get_theme(theme_name)
        if theme:
            # Apply theme colors and styling
            # This would update CSS variables or Textual theme settings
            logger.debug(f"Applied theme: {theme_name}")
    
    async def _start_background_tasks(self) -> None:
        """
        Start background monitoring tasks.
        """
        # Start periodic status updates
        task = asyncio.create_task(self._periodic_status_update())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        logger.debug("Background tasks started")
    
    async def _cleanup_background_tasks(self) -> None:
        """
        Cleanup background tasks.
        """
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.debug("Background tasks cleaned up")
    
    async def _periodic_status_update(self) -> None:
        """
        Periodically update application status.
        """
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Get system status
                if self.current_project:
                    status = await self.project_manager.get_project_status(self.current_project)
                    # Update UI with status information
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic status update: {e}")
                await asyncio.sleep(5)  # Short delay before retry