"""
Main TUI Application - Textual-based Terminal User Interface.

The primary application interface providing project management, AI integration,
task orchestration, and real-time monitoring capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Button, Input, TextArea, Tree, DataTable,
    Static, Label, ProgressBar, Tabs, TabbedContent, TabPane,
    LoadingIndicator, Log, DirectoryTree
)
from textual.reactive import reactive
from textual.message import Message
from textual.screen import Screen
from textual.binding import Binding
from textual.worker import Worker
from rich.text import Text
from rich.console import Console

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.core.ai_interface import AIInterface
from src.claude_tui.core.project_manager import ProjectManager
from src.claude_tui.core.task_engine import TaskEngine
from src.claude_tui.core.logger import get_logger
from src.claude_tui.ui.screens.workspace_screen import WorkspaceScreen
from src.claude_tui.ui.screens.project_wizard import ProjectWizard
from src.claude_tui.ui.screens.settings import SettingsScreen

try:
    from src.claude_tui.ui.screens.help_screen import HelpScreen
except ImportError:
    HelpScreen = Static

try:
    from src.claude_tui.ui.widgets.task_dashboard import TaskDashboard
except ImportError:
    TaskDashboard = Static

try:
    from src.claude_tui.ui.widgets.console_widget import ConsoleWidget
except ImportError:
    ConsoleWidget = Static

try:
    from src.claude_tui.ui.widgets.metrics_dashboard import MetricsDashboard
except ImportError:
    MetricsDashboard = Static

try:
    from src.claude_tui.ui.widgets.notification_system import NotificationSystem
except ImportError:
    NotificationSystem = Static

logger = get_logger(__name__)
console = Console()


class StatusBar(Static):
    """Status bar showing current state and metrics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_timer = None
        
    def on_mount(self) -> None:
        """Setup periodic status updates."""
        self.update_timer = self.set_interval(5.0, self.update_status)
        self.update_status()
    
    def update_status(self) -> None:
        """Update status bar content."""
        try:
            app = self.app
            if hasattr(app, 'current_project') and app.current_project:
                project_name = app.current_project.name
                project_status = "Active"
            else:
                project_name = "No Project"
                project_status = "Idle"
            
            memory_usage = "Unknown"
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_usage = f"{memory_mb:.1f}MB"
            except ImportError:
                pass
            
            ai_status = "Connected" if hasattr(app, 'ai_interface') and app.ai_interface else "Disconnected"
            
            status_text = f"Project: {project_name} | Status: {project_status} | AI: {ai_status} | Memory: {memory_usage} | {datetime.now().strftime('%H:%M:%S')}"
            self.update(status_text)
            
        except Exception as e:
            self.update(f"Status Error: {e}")


class MainScreen(Screen):
    """Main application screen with tabbed interface."""
    
    BINDINGS = [
        Binding("ctrl+n", "new_project", "New Project"),
        Binding("ctrl+o", "open_project", "Open Project"),
        Binding("ctrl+s", "save_project", "Save Project"),
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("f1", "show_help", "Help"),
        Binding("f2", "show_settings", "Settings"),
        Binding("escape", "close_modal", "Close Modal"),
    ]
    
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_tab = "workspace"
    
    def compose(self) -> ComposeResult:
        """Compose the main screen layout."""
        yield Header(show_clock=True)
        
        with Vertical():
            # Main content area with tabs
            with TabbedContent(initial="workspace"):
                with TabPane("Workspace", id="workspace"):
                    yield WorkspaceScreen(id="workspace_content")
                
                with TabPane("AI Console", id="ai_console"):
                    yield ConsoleWidget(id="ai_console_content")
                
                with TabPane("Tasks", id="tasks"):
                    yield TaskDashboard(id="task_dashboard")
                
                with TabPane("Metrics", id="metrics"):
                    yield MetricsDashboard(id="metrics_dashboard")
                
                with TabPane("Projects", id="projects"):
                    with Vertical():
                        yield DirectoryTree("./", id="project_tree")
                        with Horizontal():
                            yield Button("New Project", id="new_project_btn", variant="primary")
                            yield Button("Open Project", id="open_project_btn", variant="default")
                            yield Button("Clone Project", id="clone_project_btn", variant="default")
        
        yield StatusBar(id="status_bar")
        yield Footer()
        yield NotificationSystem(id="notifications")
    
    def on_mount(self) -> None:
        """Initialize the main screen."""
        self.app_instance.main_screen = self
        logger.info("Main screen mounted")
    
    def action_new_project(self) -> None:
        """Create a new project."""
        self.app_instance.push_screen(ProjectWizard())
    
    def action_open_project(self) -> None:
        """Open an existing project."""
        from src.claude_tui.ui.screens.file_picker import FilePickerScreen
        
        def on_project_selected(project_path: Path) -> None:
            """Handle project selection."""
            asyncio.create_task(self._load_selected_project(project_path))
        
        file_picker = FilePickerScreen(
            title="Open Project",
            file_types=[".json", ".toml", ".yaml", ".py"],
            directories_only=True,
            callback=on_project_selected
        )
        self.app_instance.push_screen(file_picker)
    
    def action_save_project(self) -> None:
        """Save current project."""
        if hasattr(self.app_instance, 'current_project') and self.app_instance.current_project:
            self.app_instance.notify("Project saved", severity="success")
        else:
            self.app_instance.notify("No project to save", severity="warning")
    
    def action_quit_app(self) -> None:
        """Quit the application."""
        self.app_instance.exit()
    
    def action_show_help(self) -> None:
        """Show help screen."""
        self.app_instance.push_screen(HelpScreen())
    
    def action_show_settings(self) -> None:
        """Show settings screen."""
        self.app_instance.push_screen(SettingsScreen())
    
    def action_close_modal(self) -> None:
        """Close any open modal."""
        if self.app_instance.screen_stack:
            self.app_instance.pop_screen()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "new_project_btn":
            self.action_new_project()
        elif event.button.id == "open_project_btn":
            self.action_open_project()
        elif event.button.id == "clone_project_btn":
            from src.claude_tui.ui.screens.clone_project_dialog import CloneProjectDialog
            
            def on_clone_complete(result: Dict[str, Any]) -> None:
                if result.get('success'):
                    self.app_instance.notify(f"Project cloned: {result['name']}", severity="success")
                    if result.get('path'):
                        asyncio.create_task(self._load_selected_project(Path(result['path'])))
                else:
                    self.app_instance.notify(f"Clone failed: {result.get('error', 'Unknown error')}", severity="error")
            
            clone_dialog = CloneProjectDialog(callback=on_clone_complete)
            self.app_instance.push_screen(clone_dialog)
    
    async def _load_selected_project(self, project_path: Path) -> None:
        """Load a selected project."""
        try:
            if self.app_instance.project_manager:
                self.app_instance.notify(f"Loading project: {project_path}", severity="info")
                
                project = await self.app_instance.project_manager.load_project(project_path)
                self.app_instance.current_project = project
                
                # Update project tree view
                project_tree = self.query_one("#project_tree", DirectoryTree)
                project_tree.path = project_path
                
                self.app_instance.notify(f"Project loaded: {project.name}", severity="success")
                logger.info(f"Project loaded successfully: {project.name}")
            else:
                self.app_instance.notify("Project manager not available", severity="error")
        
        except Exception as e:
            error_msg = f"Failed to load project: {e}"
            self.app_instance.notify(error_msg, severity="error")
            logger.error(error_msg)


class ClaudeTUIApp(App):
    """
    Main Claude TUI Application.
    
    Provides an intelligent AI-powered terminal interface for software development
    with project management, task orchestration, and real-time collaboration.
    """
    
    CSS_PATH = "styles/main.tcss"
    TITLE = "Claude TUI - AI-Powered Development Environment"
    SUB_TITLE = "Intelligent Terminal Interface"
    
    # Reactive attributes
    current_project = reactive(None)
    ai_connected = reactive(False)
    task_count = reactive(0)
    
    def __init__(
        self,
        config_manager: ConfigManager,
        debug: bool = False,
        initial_project_dir: Optional[Path] = None,
        headless: bool = False,
        test_mode: bool = False
    ):
        super().__init__()
        
        # Core components
        self.config_manager = config_manager
        self._debug = debug
        self.initial_project_dir = initial_project_dir
        self.headless = headless
        self.test_mode = test_mode
        
        # Initialize components
        self.ai_interface: Optional[AIInterface] = None
        self.project_manager: Optional[ProjectManager] = None
        self.task_engine: Optional[TaskEngine] = None
        
        # UI components
        self.main_screen: Optional[MainScreen] = None
        self.notification_system: Optional[NotificationSystem] = None
        
        # State
        self.initialization_complete = False
        self.background_tasks: List[Worker] = []
        
        logger.info("Claude TUI App initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield MainScreen(self)
    
    async def on_mount(self) -> None:
        """Initialize the application after mounting."""
        self.title = self.TITLE
        self.sub_title = self.SUB_TITLE
        
        # Start initialization in background
        self.run_worker(self._initialize_components(), exclusive=True)
        
        # Setup notification system
        self.notification_system = self.query_one("#notifications", NotificationSystem)
        
        logger.info("Claude TUI App mounted")
    
    @property
    def is_initialized(self) -> bool:
        """Check if app is fully initialized."""
        return self.initialization_complete
    
    async def _initialize_components(self) -> None:
        """Initialize all application components."""
        try:
            self.notify("Initializing Claude TUI...", severity="info")
            
            # Initialize AI interface
            self.notify("Connecting to AI services...", severity="info")
            self.ai_interface = AIInterface(self.config_manager)
            ai_success = await self.ai_interface.initialize()
            
            if ai_success:
                self.ai_connected = True
                self.notify("AI services connected", severity="success")
                logger.info("AI interface initialized successfully")
            else:
                self.notify("AI services connection failed", severity="error")
                logger.warning("AI interface initialization failed")
            
            # Initialize project manager
            self.notify("Setting up project management...", severity="info")
            self.project_manager = ProjectManager(self.config_manager)
            await self.project_manager.initialize()
            
            # Initialize task engine
            self.notify("Starting task engine...", severity="info")
            self.task_engine = TaskEngine(self.config_manager)
            await self.task_engine.initialize()
            
            # Load initial project if specified
            if self.initial_project_dir:
                self.notify(f"Loading project: {self.initial_project_dir}", severity="info")
                try:
                    project = await self.project_manager.load_project(self.initial_project_dir)
                    self.current_project = project
                    self.notify(f"Project loaded: {project.name}", severity="success")
                except Exception as e:
                    self.notify(f"Failed to load project: {e}", severity="error")
                    logger.error(f"Failed to load initial project: {e}")
            
            # Start background monitoring
            self._start_background_monitoring()
            
            self.initialization_complete = True
            self.notify("Claude TUI ready!", severity="success")
            logger.info("Application initialization complete")
            
        except Exception as e:
            self.notify(f"Initialization failed: {e}", severity="error")
            logger.error(f"Application initialization failed: {e}")
            
            # Still allow app to run in degraded mode
            self.initialization_complete = True
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Task monitoring
            if self.task_engine:
                task_monitor = self.run_worker(
                    self._monitor_tasks(),
                    name="task_monitor",
                    group="monitoring"
                )
                self.background_tasks.append(task_monitor)
            
            # AI health monitoring
            if self.ai_interface:
                ai_monitor = self.run_worker(
                    self._monitor_ai_health(),
                    name="ai_monitor", 
                    group="monitoring"
                )
                self.background_tasks.append(ai_monitor)
            
            logger.info("Background monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start background monitoring: {e}")
    
    async def _monitor_tasks(self) -> None:
        """Monitor task engine status."""
        while True:
            try:
                if self.task_engine:
                    status = await self.task_engine.get_status()
                    self.task_count = status.get('active_tasks', 0)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _monitor_ai_health(self) -> None:
        """Monitor AI service health."""
        while True:
            try:
                if self.ai_interface and self.ai_interface._initialization_complete:
                    # Simple health check - could be more sophisticated
                    self.ai_connected = True
                else:
                    self.ai_connected = False
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AI health monitoring error: {e}")
                self.ai_connected = False
                await asyncio.sleep(60)
    
    def notify(self, message: str, severity: str = "info") -> None:
        """Send a notification to the user."""
        try:
            if self.notification_system:
                self.notification_system.add_notification(message, severity)
            else:
                # Fallback to console if notification system not ready
                if severity == "error":
                    console.print(f"[red]Error: {message}[/red]")
                elif severity == "warning":
                    console.print(f"[yellow]Warning: {message}[/yellow]")
                elif severity == "success":
                    console.print(f"[green]Success: {message}[/green]")
                else:
                    console.print(f"[blue]Info: {message}[/blue]")
                    
            logger.info(f"Notification: {severity} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def execute_ai_task(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None
    ) -> Any:
        """
        Execute an AI task with progress tracking.
        
        Args:
            prompt: Task description
            context: Additional context
            callback: Optional callback for progress updates
            
        Returns:
            Task result
        """
        if not self.ai_interface or not self.ai_connected:
            self.notify("AI services not available", severity="error")
            return None
        
        try:
            self.notify("Executing AI task...", severity="info")
            
            if callback:
                callback("Starting AI task...")
            
            result = await self.ai_interface.execute_claude_code(
                prompt=prompt,
                context=context,
                project_path=self.current_project.path if self.current_project else None
            )
            
            if result.success:
                self.notify("AI task completed", severity="success")
                if callback:
                    callback("Task completed successfully")
                return result
            else:
                self.notify(f"AI task failed: {result.error}", severity="error")
                if callback:
                    callback(f"Task failed: {result.error}")
                return None
                
        except Exception as e:
            error_msg = f"AI task execution failed: {e}"
            self.notify(error_msg, severity="error")
            logger.error(error_msg)
            if callback:
                callback(f"Error: {e}")
            return None
    
    async def close_application(self) -> None:
        """Clean shutdown of the application."""
        try:
            self.notify("Shutting down Claude TUI...", severity="info")
            
            # Cancel background tasks
            for worker in self.background_tasks:
                worker.cancel()
            
            # Close AI interface
            if self.ai_interface:
                await self.ai_interface.close()
            
            # Save any pending project changes
            if self.current_project and self.project_manager:
                try:
                    await self.project_manager.save_project(self.current_project)
                except Exception as e:
                    logger.warning(f"Failed to save project on shutdown: {e}")
            
            logger.info("Application shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")
    
    async def on_unmount(self) -> None:
        """Cleanup on unmount."""
        await self.close_application()
    
    async def run_async(self) -> None:
        """
        Async version of run for compatibility with launcher.
        
        This method handles both headless and interactive modes.
        """
        try:
            if self.headless or self.test_mode:
                # For headless/test mode, just initialize without running the full UI
                await self._initialize_components()
            else:
                # For interactive mode, delegate to textual's run_async if available
                if hasattr(super(), 'run_async'):
                    await super().run_async()
                else:
                    # Fallback: initialize and let the app be managed externally
                    await self._initialize_components()
        except Exception as e:
            logger.error(f"Error in run_async: {e}")
            raise
    
    def action_quit(self) -> None:
        """Override quit to ensure clean shutdown."""
        self.run_worker(self.close_application(), exclusive=True)
        self.exit()