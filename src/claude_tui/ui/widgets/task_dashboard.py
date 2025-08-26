"""
Task Dashboard Widget - Displays task progress and management interface.

This widget provides a comprehensive view of development tasks with:
- Task status visualization
- Progress tracking
- Interactive task management
- Real-time updates
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass

try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import (
        DataTable, Static, Button, ProgressBar,
        Collapsible, Label, Container
    )
    from textual.containers import Grid, Horizontal, Vertical
    from textual.reactive import reactive
    from textual.message import Message
    from textual import work
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    # Fallback base classes and functions
    Widget = object
    ComposeResult = object
    Message = object
    def reactive(default):
        """Fallback reactive decorator."""
        return default
    def work(func):
        """Fallback work decorator."""
        return func

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Task information for dashboard display."""
    id: str
    name: str
    status: str
    progress: float
    priority: str
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    dependencies: Set[str] = None
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.tags is None:
            self.tags = set()


class TaskStatusWidget(Widget):
    """Individual task status display widget."""
    
    def __init__(self, task: TaskInfo, **kwargs):
        """Initialize task status widget."""
        super().__init__(**kwargs)
        self.task = task
        self.progress_bar = None
        self.status_label = None
        
    def compose(self) -> ComposeResult:
        """Compose the task status widget."""
        if not TEXTUAL_AVAILABLE:
            yield Static("Task status widget (Textual not available)")
            return
            
        with Container(classes="task-status-container"):
            yield Label(f"Task: {self.task.name}", classes="task-name")
            yield Label(f"Status: {self.task.status}", classes="task-status")
            
            # Progress bar
            self.progress_bar = ProgressBar(
                total=100,
                show_eta=True,
                classes="task-progress"
            )
            self.progress_bar.advance(self.task.progress)
            yield self.progress_bar
            
            # Additional info
            with Horizontal(classes="task-info"):
                yield Label(f"Priority: {self.task.priority}", classes="task-priority")
                if self.task.assigned_to:
                    yield Label(f"Assigned: {self.task.assigned_to}", classes="task-assignee")
    
    def update_progress(self, progress: float) -> None:
        """Update task progress."""
        self.task.progress = progress
        if self.progress_bar and TEXTUAL_AVAILABLE:
            self.progress_bar.advance(progress - self.progress_bar.progress)


class TaskDashboard(Widget):
    """
    Main task dashboard widget for displaying and managing development tasks.
    
    Features:
    - Task list with status and progress
    - Filter and search capabilities
    - Task creation and editing
    - Real-time updates
    - Dependency visualization
    """
    
    # Reactive attributes
    selected_task_id = reactive(None)
    filter_status = reactive("all")
    
    class TaskSelected(Message):
        """Message sent when a task is selected."""
        def __init__(self, task_id: str, task: TaskInfo) -> None:
            super().__init__()
            self.task_id = task_id
            self.task = task
    
    class TaskUpdated(Message):
        """Message sent when a task is updated."""
        def __init__(self, task_id: str, task: TaskInfo) -> None:
            super().__init__()
            self.task_id = task_id
            self.task = task
    
    def __init__(self, **kwargs):
        """Initialize the task dashboard."""
        super().__init__(**kwargs)
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_widgets: Dict[str, TaskStatusWidget] = {}
        self.data_table = None
        self.filter_buttons = {}
        self.update_callbacks: List[Callable] = []
        
    def compose(self) -> ComposeResult:
        """Compose the task dashboard."""
        if not TEXTUAL_AVAILABLE:
            yield Static("Task Dashboard (Textual not available)")
            return
            
        with Container(classes="task-dashboard"):
            # Header with filters
            with Horizontal(classes="dashboard-header"):
                yield Label("Task Dashboard", classes="dashboard-title")
                
                # Filter buttons
                with Horizontal(classes="filter-buttons"):
                    self.filter_buttons["all"] = Button("All", variant="primary", classes="filter-btn")
                    self.filter_buttons["pending"] = Button("Pending", classes="filter-btn")
                    self.filter_buttons["in_progress"] = Button("In Progress", classes="filter-btn")
                    self.filter_buttons["completed"] = Button("Completed", classes="filter-btn")
                    
                    for button in self.filter_buttons.values():
                        yield button
            
            # Main content area
            with Container(classes="dashboard-content"):
                # Task table
                self.data_table = DataTable(classes="task-table")
                self.data_table.add_columns("ID", "Name", "Status", "Progress", "Priority")
                yield self.data_table
                
                # Task details panel
                with Collapsible(title="Task Details", collapsed=True, classes="task-details"):
                    yield Container(id="task-details-content")
    
    def on_mount(self) -> None:
        """Handle widget mount."""
        if TEXTUAL_AVAILABLE:
            # Set up table selection
            self.data_table.cursor_type = "row"
            
            # Bind filter buttons
            for status, button in self.filter_buttons.items():
                button.action_press = lambda s=status: self.set_filter(s)
    
    def add_task(self, task: TaskInfo) -> None:
        """Add a new task to the dashboard."""
        self.tasks[task.id] = task
        self._refresh_table()
        
        # Create task widget
        task_widget = TaskStatusWidget(task)
        self.task_widgets[task.id] = task_widget
        
        logger.info(f"Added task: {task.name} ({task.id})")
    
    def update_task(self, task_id: str, task: TaskInfo) -> None:
        """Update an existing task."""
        if task_id in self.tasks:
            self.tasks[task_id] = task
            self._refresh_table()
            
            # Update widget
            if task_id in self.task_widgets:
                self.task_widgets[task_id].task = task
                self.task_widgets[task_id].update_progress(task.progress)
            
            # Notify listeners
            self.post_message(self.TaskUpdated(task_id, task))
            for callback in self.update_callbacks:
                callback(task_id, task)
                
            logger.info(f"Updated task: {task.name} ({task_id})")
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task from the dashboard."""
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            self._refresh_table()
            
            # Remove widget
            if task_id in self.task_widgets:
                del self.task_widgets[task_id]
                
            logger.info(f"Removed task: {task.name} ({task_id})")
    
    def set_filter(self, status: str) -> None:
        """Set the task filter status."""
        self.filter_status = status
        self._refresh_table()
        
        # Update button states
        for btn_status, button in self.filter_buttons.items():
            if btn_status == status:
                button.variant = "primary"
            else:
                button.variant = "default"
    
    def _refresh_table(self) -> None:
        """Refresh the task table display."""
        if not self.data_table or not TEXTUAL_AVAILABLE:
            return
            
        # Clear table
        self.data_table.clear()
        
        # Filter tasks
        filtered_tasks = self._get_filtered_tasks()
        
        # Add rows
        for task in filtered_tasks:
            progress_text = f"{task.progress:.1f}%"
            self.data_table.add_row(
                task.id,
                task.name,
                task.status,
                progress_text,
                task.priority
            )
    
    def _get_filtered_tasks(self) -> List[TaskInfo]:
        """Get tasks filtered by current status."""
        if self.filter_status == "all":
            return list(self.tasks.values())
        
        return [
            task for task in self.tasks.values()
            if task.status.lower() == self.filter_status.lower()
        ]
    
    def on_data_table_row_selected(self, event) -> None:
        """Handle task selection in table."""
        if not TEXTUAL_AVAILABLE:
            return
            
        row_key = event.row_key
        if row_key is not None:
            # Get task ID from first column
            task_id = self.data_table.get_row(row_key)[0]
            if task_id in self.tasks:
                self.selected_task_id = task_id
                task = self.tasks[task_id]
                self.post_message(self.TaskSelected(task_id, task))
                self._show_task_details(task)
    
    def _show_task_details(self, task: TaskInfo) -> None:
        """Show detailed information for selected task."""
        if not TEXTUAL_AVAILABLE:
            return
            
        details_container = self.query_one("#task-details-content")
        details_container.remove_children()
        
        # Add detailed task information
        details_container.mount(
            Label(f"Task ID: {task.id}"),
            Label(f"Name: {task.name}"),
            Label(f"Status: {task.status}"),
            Label(f"Progress: {task.progress:.1f}%"),
            Label(f"Priority: {task.priority}"),
        )
        
        if task.assigned_to:
            details_container.mount(Label(f"Assigned to: {task.assigned_to}"))
        
        if task.due_date:
            details_container.mount(Label(f"Due: {task.due_date.strftime('%Y-%m-%d %H:%M')}"))
        
        if task.dependencies:
            deps_text = ", ".join(task.dependencies)
            details_container.mount(Label(f"Dependencies: {deps_text}"))
        
        if task.tags:
            tags_text = ", ".join(task.tags)
            details_container.mount(Label(f"Tags: {tags_text}"))
    
    def add_update_callback(self, callback: Callable) -> None:
        """Add a callback for task updates."""
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable) -> None:
        """Remove an update callback."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    @work
    async def refresh_tasks_async(self) -> None:
        """Asynchronously refresh task data."""
        # This would typically fetch from a service
        # For now, just refresh the display
        self._refresh_table()


# Fallback implementations for when Textual is not available
class FallbackTaskDashboard:
    """Fallback task dashboard for when Textual is not available."""
    
    def __init__(self):
        """Initialize fallback dashboard."""
        self.tasks = {}
        logger.info("Using fallback task dashboard (Textual not available)")
    
    def add_task(self, task: TaskInfo) -> None:
        """Add task to fallback dashboard."""
        self.tasks[task.id] = task
        print(f"Task added: {task.name} ({task.status})")
    
    def update_task(self, task_id: str, task: TaskInfo) -> None:
        """Update task in fallback dashboard."""
        self.tasks[task_id] = task
        print(f"Task updated: {task.name} ({task.status}) - {task.progress:.1f}%")
    
    def list_tasks(self) -> None:
        """List all tasks in fallback mode."""
        print("\n=== Task Dashboard ===")
        for task in self.tasks.values():
            print(f"[{task.status}] {task.name} - {task.progress:.1f}% ({task.priority})")
        print("====================\n")


# Export the appropriate dashboard class
if TEXTUAL_AVAILABLE:
    __all__ = ["TaskDashboard", "TaskInfo", "TaskStatusWidget"]
else:
    TaskDashboard = FallbackTaskDashboard
    __all__ = ["TaskDashboard", "TaskInfo"]