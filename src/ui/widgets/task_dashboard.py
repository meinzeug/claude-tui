#!/usr/bin/env python3
"""
Task Dashboard Widget - Real-time task management and progress tracking
with intelligent progress validation and authenticity scoring.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import (
    Static, Label, Button, ProgressBar, 
    DataTable, ListView, ListItem
)
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.console import Console


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProjectTask:
    """Project task data structure"""
    id: str
    name: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    progress: float = 0.0
    real_progress: float = 0.0
    fake_progress: float = 0.0
    quality_score: float = 0.0
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = None
    assigned_agent: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
    
    @property
    def authenticity_score(self) -> float:
        """Calculate authenticity score based on real vs fake progress"""
        if self.progress == 0:
            return 1.0
        return min(1.0, self.real_progress / max(0.01, self.progress))
    
    @property
    def eta_minutes(self) -> Optional[int]:
        """Estimated time to completion in minutes"""
        if self.status == TaskStatus.COMPLETED:
            return 0
        if self.real_progress == 0:
            return None
        
        remaining_progress = 1.0 - self.real_progress
        time_per_progress = self.actual_hours / max(0.01, self.real_progress)
        remaining_hours = remaining_progress * time_per_progress
        return int(remaining_hours * 60)
    
    def get_status_icon(self) -> str:
        """Get icon for current task status"""
        icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.VALIDATING: "🔍",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫",
        }
        return icons.get(self.status, "❓")
    
    def get_priority_icon(self) -> str:
        """Get icon for task priority"""
        icons = {
            TaskPriority.LOW: "🟢",
            TaskPriority.MEDIUM: "🟡",
            TaskPriority.HIGH: "🟠",
            TaskPriority.CRITICAL: "🔴",
        }
        return icons.get(self.priority, "⚪")


class TaskProgressWidget(Container):
    """Individual task progress display with authenticity validation"""
    
    def __init__(self, task: ProjectTask) -> None:
        super().__init__()
        self.task = task
        self.progress_bar: Optional[ProgressBar] = None
        self.status_label: Optional[Label] = None
        self.authenticity_label: Optional[Label] = None
        
    def compose(self) -> ComposeResult:
        """Compose task progress widget"""
        with Vertical():
            # Task header
            header_text = f"{self.task.get_status_icon()} {self.task.name} {self.task.get_priority_icon()}"
            yield Label(header_text, classes="task-header")
            
            # Progress bars
            with Horizontal():
                yield Label("Real:", classes="progress-label")
                self.real_progress_bar = ProgressBar(
                    total=100, 
                    show_eta=False,
                    show_percentage=True
                )
                yield self.real_progress_bar
            
            with Horizontal():
                yield Label("Claimed:", classes="progress-label")
                self.claimed_progress_bar = ProgressBar(
                    total=100,
                    show_eta=False, 
                    show_percentage=True
                )
                yield self.claimed_progress_bar
            
            # Status information
            self.status_label = Label(
                f"Quality: {self.task.quality_score:.1f}/10 | Auth: {self.task.authenticity_score:.0%}",
                classes="task-status"
            )
            yield self.status_label
            
            # ETA information
            eta_text = f"ETA: {self.task.eta_minutes or 'Unknown'} min" if self.task.eta_minutes else "ETA: Calculating..."
            self.eta_label = Label(eta_text, classes="task-eta")
            yield self.eta_label
    
    def update_task(self, task: ProjectTask) -> None:
        """Update widget with new task data"""
        self.task = task
        
        # Update progress bars
        if hasattr(self, 'real_progress_bar'):
            self.real_progress_bar.update(progress=int(task.real_progress * 100))
        if hasattr(self, 'claimed_progress_bar'):
            self.claimed_progress_bar.update(progress=int(task.progress * 100))
        
        # Update status
        if self.status_label:
            self.status_label.update(
                f"Quality: {task.quality_score:.1f}/10 | Auth: {task.authenticity_score:.0%}"
            )
        
        # Update ETA
        if self.eta_label:
            eta_text = f"ETA: {task.eta_minutes} min" if task.eta_minutes else "ETA: Calculating..."
            self.eta_label.update(eta_text)


class TaskList(ListView):
    """Enhanced task list with filtering and sorting"""
    
    def __init__(self, tasks: List[ProjectTask]) -> None:
        super().__init__()
        self.tasks = tasks
        self.filter_status: Optional[TaskStatus] = None
        self.sort_by = "priority"
        
    def populate_list(self) -> None:
        """Populate list with filtered and sorted tasks"""
        self.clear()
        
        # Filter tasks
        filtered_tasks = self.tasks
        if self.filter_status:
            filtered_tasks = [t for t in filtered_tasks if t.status == self.filter_status]
        
        # Sort tasks
        if self.sort_by == "priority":
            priority_order = {p: i for i, p in enumerate(TaskPriority)}
            filtered_tasks.sort(key=lambda t: priority_order.get(t.priority, 99))
        elif self.sort_by == "progress":
            filtered_tasks.sort(key=lambda t: t.real_progress, reverse=True)
        elif self.sort_by == "created":
            filtered_tasks.sort(key=lambda t: t.created_at)
        
        # Add tasks to list
        for task in filtered_tasks:
            task_widget = TaskProgressWidget(task)
            self.append(ListItem(task_widget))
    
    def set_filter(self, status: Optional[TaskStatus]) -> None:
        """Set status filter"""
        self.filter_status = status
        self.populate_list()
    
    def set_sort(self, sort_by: str) -> None:
        """Set sort criteria"""
        self.sort_by = sort_by
        self.populate_list()
    
    def update_tasks(self, tasks: List[ProjectTask]) -> None:
        """Update task list with new data"""
        self.tasks = tasks
        self.populate_list()


class TaskDashboard(Vertical):
    """Main task dashboard with real-time updates and validation"""
    
    tasks: reactive[List[ProjectTask]] = reactive([])
    
    def __init__(self, backend_bridge) -> None:
        super().__init__()
        self.backend_bridge = backend_bridge
        self.task_list: Optional[TaskList] = None
        self.stats_widget: Optional[Static] = None
        self.filter_buttons: Dict[str, Button] = {}
        self.current_filter: Optional[TaskStatus] = None
        self.current_project = None
        
    def compose(self) -> ComposeResult:
        """Compose task dashboard"""
        yield Label("🎯 Task Dashboard", classes="header")
        
        # Task statistics
        self.stats_widget = Static(self._generate_stats(), id="task-stats")
        yield self.stats_widget
        
        # Filter buttons
        with Horizontal(classes="filter-bar"):
            self.filter_buttons["all"] = Button("All", id="filter-all")
            yield self.filter_buttons["all"]
            
            self.filter_buttons["active"] = Button("Active", id="filter-active")
            yield self.filter_buttons["active"]
            
            self.filter_buttons["completed"] = Button("Completed", id="filter-completed")
            yield self.filter_buttons["completed"]
            
            self.filter_buttons["blocked"] = Button("Blocked", id="filter-blocked")
            yield self.filter_buttons["blocked"]
        
        # Task list
        self.task_list = TaskList(self.tasks)
        yield self.task_list
        
        # Control buttons
        with Horizontal(classes="control-bar"):
            yield Button("➕ Add Task", id="add-task")
            yield Button("🔄 Refresh", id="refresh-tasks")
            yield Button("📊 Analytics", id="task-analytics")
    
    def on_mount(self) -> None:
        """Initialize dashboard after mounting"""
        self.start_task_monitoring()
    
    def watch_tasks(self, tasks: List[ProjectTask]) -> None:
        """React to task list changes"""
        if self.task_list:
            self.task_list.update_tasks(tasks)
        if self.stats_widget:
            self.stats_widget.update(self._generate_stats())
    
    @work(exclusive=True)
    async def start_task_monitoring(self) -> None:
        """Start real-time task monitoring via backend bridge"""
        while True:
            try:
                # Get latest tasks from backend bridge
                if self.backend_bridge and self.current_project:
                    backend_tasks = await self.backend_bridge.get_project_tasks(self.current_project)
                    
                    # Convert backend task data to ProjectTask objects
                    updated_tasks = []
                    for task_data in backend_tasks:
                        project_task = self._convert_backend_task(task_data)
                        if project_task:
                            updated_tasks.append(project_task)
                    
                    # Update reactive property with real data
                    self.tasks = updated_tasks
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                # Handle errors gracefully - continue in offline mode
                await asyncio.sleep(30)
    
    def _convert_backend_task(self, task_data: Dict[str, Any]) -> Optional[ProjectTask]:
        """Convert backend task data to ProjectTask object"""
        try:
            # Convert backend task format to ProjectTask
            task_id = task_data.get('task_id') or task_data.get('id', '')
            name = task_data.get('name', 'Unknown Task')
            description = task_data.get('description', '')
            status_str = task_data.get('status', 'pending')
            priority_str = task_data.get('priority', 'medium')
            
            # Map backend status to TaskStatus enum
            status_mapping = {
                'pending': TaskStatus.PENDING,
                'in_progress': TaskStatus.IN_PROGRESS,
                'running': TaskStatus.IN_PROGRESS,
                'validating': TaskStatus.VALIDATING,
                'completed': TaskStatus.COMPLETED,
                'failed': TaskStatus.FAILED,
                'blocked': TaskStatus.BLOCKED
            }
            status = status_mapping.get(status_str, TaskStatus.PENDING)
            
            # Map backend priority to TaskPriority enum
            priority_mapping = {
                'low': TaskPriority.LOW,
                'medium': TaskPriority.MEDIUM,
                'high': TaskPriority.HIGH,
                'critical': TaskPriority.CRITICAL
            }
            priority = priority_mapping.get(priority_str, TaskPriority.MEDIUM)
            
            # Extract progress and quality data from backend
            progress = task_data.get('progress_percentage', 0.0) / 100.0
            real_progress = task_data.get('real_progress', progress)
            quality_score = task_data.get('quality_score', 0.0)
            authenticity_score = task_data.get('authenticity_score', 100.0) / 100.0
            
            # Calculate fake progress
            fake_progress = max(0, progress - real_progress)
            
            # Parse timestamps
            created_at = datetime.now()
            if task_data.get('created_at'):
                try:
                    created_at = datetime.fromisoformat(task_data['created_at'].replace('Z', '+00:00')).replace(tzinfo=None)
                except:
                    pass
            
            started_at = None
            if task_data.get('started_at'):
                try:
                    started_at = datetime.fromisoformat(task_data['started_at'].replace('Z', '+00:00')).replace(tzinfo=None)
                except:
                    pass
            
            completed_at = None
            if task_data.get('completed_at'):
                try:
                    completed_at = datetime.fromisoformat(task_data['completed_at'].replace('Z', '+00:00')).replace(tzinfo=None)
                except:
                    pass
            
            # Create ProjectTask with real backend data
            project_task = ProjectTask(
                id=task_id,
                name=name,
                description=description,
                status=status,
                priority=priority,
                progress=progress,
                real_progress=real_progress,
                fake_progress=fake_progress,
                quality_score=quality_score,
                estimated_hours=task_data.get('estimated_duration', 0.0) / 60.0,  # Convert minutes to hours
                actual_hours=task_data.get('execution_time_seconds', 0.0) / 3600.0,  # Convert seconds to hours
                created_at=created_at,
                started_at=started_at,
                completed_at=completed_at,
                dependencies=task_data.get('dependencies', []),
                assigned_agent=task_data.get('assigned_agent'),
                validation_results=task_data.get('validation_results')
            )
            
            # Override authenticity score property calculation
            project_task._authenticity_score = authenticity_score
            
            return project_task
            
        except Exception as e:
            # Log conversion error but continue
            return None
    
    async def update_task_from_backend(self, task_data: Dict[str, Any]) -> None:
        """Update specific task from real-time backend data"""
        try:
            task_id = task_data.get('task_id')
            if not task_id:
                return
            
            # Find and update existing task
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    updated_task = self._convert_backend_task(task_data)
                    if updated_task:
                        # Update the task in place
                        tasks_copy = list(self.tasks)
                        tasks_copy[i] = updated_task
                        self.tasks = tasks_copy
                    break
            else:
                # Task not found, add it
                new_task = self._convert_backend_task(task_data)
                if new_task:
                    self.tasks = list(self.tasks) + [new_task]
                    
        except Exception as e:
            # Handle update errors gracefully
            pass
    
    def _generate_stats(self) -> Text:
        """Generate task statistics display"""
        if not self.tasks:
            return Text("No tasks available")
        
        # Calculate statistics
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        in_progress_tasks = len([t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS])
        blocked_tasks = len([t for t in self.tasks if t.status == TaskStatus.BLOCKED])
        
        # Calculate average authenticity
        authenticity_scores = [t.authenticity_score for t in self.tasks if t.progress > 0]
        avg_authenticity = sum(authenticity_scores) / len(authenticity_scores) if authenticity_scores else 1.0
        
        # Calculate overall progress
        if total_tasks > 0:
            overall_progress = completed_tasks / total_tasks * 100
        else:
            overall_progress = 0
        
        # Create stats text
        stats = Text()
        stats.append(f"📋 Tasks: {total_tasks} | ")
        stats.append(f"✅ Completed: {completed_tasks} | ")
        stats.append(f"🔄 Active: {in_progress_tasks} | ")
        stats.append(f"🚫 Blocked: {blocked_tasks}\n")
        stats.append(f"📈 Progress: {overall_progress:.1f}% | ")
        stats.append(f"🎯 Authenticity: {avg_authenticity:.1%}")
        
        return stats
    
    # Event handlers
    @on(Button.Pressed, "#filter-all")
    def filter_all_tasks(self) -> None:
        """Show all tasks"""
        self.current_filter = None
        if self.task_list:
            self.task_list.set_filter(None)
    
    @on(Button.Pressed, "#filter-active")
    def filter_active_tasks(self) -> None:
        """Show only active tasks"""
        self.current_filter = TaskStatus.IN_PROGRESS
        if self.task_list:
            self.task_list.set_filter(TaskStatus.IN_PROGRESS)
    
    @on(Button.Pressed, "#filter-completed")
    def filter_completed_tasks(self) -> None:
        """Show only completed tasks"""
        self.current_filter = TaskStatus.COMPLETED
        if self.task_list:
            self.task_list.set_filter(TaskStatus.COMPLETED)
    
    @on(Button.Pressed, "#filter-blocked")
    def filter_blocked_tasks(self) -> None:
        """Show only blocked tasks"""
        self.current_filter = TaskStatus.BLOCKED
        if self.task_list:
            self.task_list.set_filter(TaskStatus.BLOCKED)
    
    @on(Button.Pressed, "#add-task")
    def add_new_task(self) -> None:
        """Handle add task button with real backend integration"""
        if self.backend_bridge:
            # Create task via backend bridge
            asyncio.create_task(self._create_new_task())
        else:
            # Emit message for parent to handle if no backend
            self.post_message(AddTaskMessage())
    
    async def _create_new_task(self) -> None:
        """Create new task via backend"""
        try:
            # Basic task creation - in a real app, this would show a dialog
            task_id = await self.backend_bridge.create_task(
                name=f"New Task {datetime.now().strftime('%H:%M')}",
                description="Task created from TUI",
                task_type="general",
                priority="medium"
            )
            
            if task_id:
                # Refresh to show new task
                await self._force_refresh()
        except Exception:
            # Handle creation errors gracefully
            pass
    
    @on(Button.Pressed, "#refresh-tasks")
    def refresh_tasks(self) -> None:
        """Refresh task list"""
        self.refresh()
    
    @on(Button.Pressed, "#task-analytics")
    def show_analytics(self) -> None:
        """Show task analytics"""
        self.post_message(ShowAnalyticsMessage())
    
    def refresh(self, *, repaint: bool = True, layout: bool = False, **kwargs) -> None:
        """Refresh the dashboard with latest backend data"""
        try:
            # Force update of tasks from backend
            if self.backend_bridge:
                # This will trigger the monitoring loop to fetch fresh data
                asyncio.create_task(self._force_refresh())
            
            # Call parent refresh with all compatible parameters
            # Filter out incompatible kwargs to avoid TypeError
            compatible_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ['repaint', 'layout']}
            
            # Try calling with standard parameters first
            try:
                super().refresh(repaint=repaint, layout=layout, **compatible_kwargs)
            except TypeError as e:
                # Fallback to basic refresh if signature mismatch
                if 'repaint' in str(e) or 'layout' in str(e):
                    super().refresh(**compatible_kwargs)
                else:
                    raise
        except Exception as e:
            # Log error but continue operation
            try:
                from ...core.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Dashboard refresh error: {str(e)}")
            except:
                pass  # Continue without logging if logger unavailable
    
    async def _force_refresh(self) -> None:
        """Force refresh from backend"""
        try:
            if self.backend_bridge and self.current_project:
                backend_tasks = await self.backend_bridge.get_project_tasks(self.current_project)
                updated_tasks = []
                for task_data in backend_tasks:
                    project_task = self._convert_backend_task(task_data)
                    if project_task:
                        updated_tasks.append(project_task)
                self.tasks = updated_tasks
        except Exception:
            # Ignore refresh errors
            pass
    
    def set_project(self, project_id: str) -> None:
        """Set the current project for task monitoring"""
        self.current_project = project_id
        self.refresh()


class AddTaskMessage(Message):
    """Message for adding new task"""
    pass


class ShowAnalyticsMessage(Message):
    """Message for showing analytics"""
    pass