#!/usr/bin/env python3
"""
Workflow Visualizer Widget - Visual representation of task workflows,
dependencies, and execution flow with real-time updates.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from textual import on, work
from textual.containers import Vertical, Horizontal, Container, Center
from textual.widgets import Static, Label, Button, Tree, ListView, ListItem
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.tree import Tree as RichTree
from rich.table import Table
from rich.console import Console


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskNodeStatus(Enum):
    """Individual task node status"""
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """Task node in the workflow graph"""
    id: str
    name: str
    description: str
    status: TaskNodeStatus = TaskNodeStatus.WAITING
    dependencies: Set[str] = None
    estimated_duration: int = 0  # minutes
    actual_duration: int = 0
    progress: float = 0.0
    agent_assigned: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_files: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.output_files is None:
            self.output_files = []
    
    def get_status_icon(self) -> str:
        """Get icon for task status"""
        icons = {
            TaskNodeStatus.WAITING: "⏳",
            TaskNodeStatus.READY: "🟢",
            TaskNodeStatus.IN_PROGRESS: "🔄",
            TaskNodeStatus.COMPLETED: "✅",
            TaskNodeStatus.FAILED: "❌",
            TaskNodeStatus.SKIPPED: "⏭️"
        }
        return icons.get(self.status, "❓")
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task can be executed based on dependencies"""
        return self.dependencies.issubset(completed_tasks)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    nodes: Dict[str, TaskNode]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_progress: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks that are ready to execute"""
        completed = {
            node_id for node_id, node in self.nodes.items()
            if node.status == TaskNodeStatus.COMPLETED
        }
        
        ready_tasks = []
        for node in self.nodes.values():
            if (node.status == TaskNodeStatus.WAITING and 
                node.can_execute(completed)):
                ready_tasks.append(node)
        
        return ready_tasks
    
    def calculate_progress(self) -> float:
        """Calculate overall workflow progress"""
        if not self.nodes:
            return 0.0
        
        total_weight = len(self.nodes)
        completed_weight = sum(
            1 if node.status == TaskNodeStatus.COMPLETED else node.progress
            for node in self.nodes.values()
        )
        
        return completed_weight / total_weight
    
    def get_critical_path(self) -> List[str]:
        """Calculate critical path through the workflow"""
        # Simplified critical path calculation
        # In production, this would use proper graph algorithms
        path = []
        remaining_nodes = set(self.nodes.keys())
        
        while remaining_nodes:
            # Find nodes with no dependencies in remaining set
            ready = []
            for node_id in remaining_nodes:
                node = self.nodes[node_id]
                deps_in_remaining = node.dependencies & remaining_nodes
                if not deps_in_remaining:
                    ready.append((node_id, node.estimated_duration))
            
            if not ready:
                break  # Circular dependency or error
            
            # Pick the longest task
            longest_task = max(ready, key=lambda x: x[1])
            path.append(longest_task[0])
            remaining_nodes.remove(longest_task[0])
        
        return path


class WorkflowTreeWidget(Tree):
    """Tree widget showing workflow structure"""
    
    def __init__(self, workflow: WorkflowDefinition) -> None:
        super().__init__("Workflow")
        self.workflow = workflow
        self.node_to_tree_id = {}
        
    def populate_tree(self) -> None:
        """Populate tree with workflow nodes"""
        self.clear()
        self.node_to_tree_id.clear()
        
        # Build hierarchy - for now, show by status groups
        status_groups = {}
        for node in self.workflow.nodes.values():
            if node.status not in status_groups:
                status_groups[node.status] = []
            status_groups[node.status].append(node)
        
        for status, nodes in status_groups.items():
            if nodes:
                status_node = self.root.add(
                    f"{nodes[0].get_status_icon()} {status.value.title()} ({len(nodes)})"
                )
                
                for node in sorted(nodes, key=lambda n: n.name):
                    node_label = f"{node.get_status_icon()} {node.name}"
                    if node.progress > 0:
                        node_label += f" ({node.progress:.0%})"
                    
                    tree_node = status_node.add(node_label)
                    self.node_to_tree_id[node.id] = tree_node
    
    def update_node_status(self, node_id: str, status: TaskNodeStatus, progress: float = None) -> None:
        """Update node status in the tree"""
        if node_id in self.workflow.nodes:
            self.workflow.nodes[node_id].status = status
            if progress is not None:
                self.workflow.nodes[node_id].progress = progress
            
            # Rebuild tree to show updated status
            self.populate_tree()


class WorkflowExecutionPanel(Static):
    """Panel showing workflow execution details"""
    
    def __init__(self, workflow: WorkflowDefinition) -> None:
        super().__init__()
        self.workflow = workflow
        
    def render(self) -> Panel:
        """Render execution panel"""
        table = Table("Metric", "Value", "Details", show_header=True)
        
        # Overall status
        table.add_row(
            "Status", 
            f"{self._get_workflow_icon()} {self.workflow.status.value.title()}",
            self._get_status_details()
        )
        
        # Progress
        progress = self.workflow.calculate_progress()
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        table.add_row(
            "Progress",
            f"{progress:.1%}",
            f"[{progress_bar}]"
        )
        
        # Task counts
        status_counts = {}
        for node in self.workflow.nodes.values():
            status_counts[node.status] = status_counts.get(node.status, 0) + 1
        
        total_tasks = len(self.workflow.nodes)
        completed = status_counts.get(TaskNodeStatus.COMPLETED, 0)
        in_progress = status_counts.get(TaskNodeStatus.IN_PROGRESS, 0)
        failed = status_counts.get(TaskNodeStatus.FAILED, 0)
        
        table.add_row(
            "Tasks",
            f"{completed}/{total_tasks}",
            f"✅ {completed} | 🔄 {in_progress} | ❌ {failed}"
        )
        
        # Timing
        if self.workflow.started_at:
            elapsed = datetime.now() - self.workflow.started_at
            elapsed_str = f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"
            table.add_row("Elapsed", elapsed_str, "")
        
        # ETA calculation
        eta = self._calculate_eta()
        if eta:
            table.add_row("ETA", eta, "Based on current progress")
        
        return Panel(table, title="🚀 Workflow Execution", border_style="green")
    
    def _get_workflow_icon(self) -> str:
        """Get icon for workflow status"""
        icons = {
            WorkflowStatus.PENDING: "⏳",
            WorkflowStatus.RUNNING: "🔄",
            WorkflowStatus.PAUSED: "⏸️",
            WorkflowStatus.COMPLETED: "✅",
            WorkflowStatus.FAILED: "❌",
            WorkflowStatus.CANCELLED: "🚫"
        }
        return icons.get(self.workflow.status, "❓")
    
    def _get_status_details(self) -> str:
        """Get detailed status information"""
        if self.workflow.status == WorkflowStatus.RUNNING:
            in_progress = sum(
                1 for node in self.workflow.nodes.values()
                if node.status == TaskNodeStatus.IN_PROGRESS
            )
            return f"{in_progress} tasks running"
        elif self.workflow.status == WorkflowStatus.COMPLETED:
            duration = self.workflow.completed_at - self.workflow.started_at
            return f"Completed in {duration.seconds}s"
        return ""
    
    def _calculate_eta(self) -> Optional[str]:
        """Calculate estimated time to completion"""
        if self.workflow.status != WorkflowStatus.RUNNING:
            return None
        
        progress = self.workflow.calculate_progress()
        if progress == 0:
            return None
        
        elapsed = (datetime.now() - self.workflow.started_at).total_seconds()
        total_estimated = elapsed / progress
        remaining = total_estimated - elapsed
        
        if remaining < 60:
            return f"{int(remaining)}s"
        elif remaining < 3600:
            return f"{int(remaining / 60)}m"
        else:
            hours = int(remaining / 3600)
            minutes = int((remaining % 3600) / 60)
            return f"{hours}h {minutes}m"


class CriticalPathWidget(Static):
    """Widget showing critical path through workflow"""
    
    def __init__(self, workflow: WorkflowDefinition) -> None:
        super().__init__()
        self.workflow = workflow
        
    def render(self) -> Panel:
        """Render critical path"""
        critical_path = self.workflow.get_critical_path()
        
        content = Text()
        if not critical_path:
            content.append("No critical path calculated", style="dim")
        else:
            content.append("Critical Path (longest dependency chain):\n\n", style="bold")
            
            for i, node_id in enumerate(critical_path):
                node = self.workflow.nodes[node_id]
                
                # Add connector
                if i > 0:
                    content.append("    ↓\n", style="dim")
                
                # Add node
                content.append(
                    f"{i + 1}. {node.get_status_icon()} {node.name}",
                    style="bold" if node.status == TaskNodeStatus.IN_PROGRESS else "default"
                )
                
                if node.estimated_duration > 0:
                    content.append(f" ({node.estimated_duration}min)", style="dim")
                content.append("\n")
            
            # Total time
            total_time = sum(
                self.workflow.nodes[node_id].estimated_duration 
                for node_id in critical_path
            )
            if total_time > 0:
                content.append(f"\nTotal estimated time: {total_time} minutes", style="bold blue")
        
        return Panel(content, title="🎯 Critical Path", border_style="yellow")


class WorkflowVisualizerWidget(Vertical):
    """Main workflow visualizer widget"""
    
    current_workflow: reactive[Optional[WorkflowDefinition]] = reactive(None)
    
    def __init__(self, task_engine) -> None:
        super().__init__()
        self.task_engine = task_engine
        self.workflow_tree: Optional[WorkflowTreeWidget] = None
        self.execution_panel: Optional[WorkflowExecutionPanel] = None
        self.critical_path_widget: Optional[CriticalPathWidget] = None
        self.monitoring_active = False
        
    def compose(self):
        """Compose workflow visualizer"""
        yield Label("🔀 Workflow Visualizer", classes="header")
        
        with Horizontal():
            # Left panel - workflow tree
            with Vertical(classes="workflow-tree-panel"):
                yield Label("Workflow Structure", classes="panel-header")
                yield Static("No workflow loaded", id="workflow-placeholder")
            
            # Right panel - execution details
            with Vertical(classes="workflow-details-panel"):
                yield Label("Execution Details", classes="panel-header")
                yield Static("", id="execution-details")
                yield Static("", id="critical-path")
        
        # Control buttons
        with Horizontal(classes="workflow-controls"):
            yield Button("▶️ Start", id="start-workflow", variant="primary")
            yield Button("⏸️ Pause", id="pause-workflow")
            yield Button("⏹️ Stop", id="stop-workflow", variant="error")
            yield Button("🔄 Refresh", id="refresh-workflow")
            yield Button("📊 Export", id="export-workflow")
    
    def on_mount(self) -> None:
        """Initialize workflow monitoring"""
        self.start_monitoring()
    
    def watch_current_workflow(self, workflow: Optional[WorkflowDefinition]) -> None:
        """React to workflow changes"""
        if workflow:
            self.load_workflow(workflow)
        else:
            self.clear_workflow()
    
    def load_workflow(self, workflow: WorkflowDefinition) -> None:
        """Load and display workflow"""
        # Remove placeholder
        try:
            placeholder = self.query_one("#workflow-placeholder", Static)
            placeholder.remove()
        except:
            pass
        
        # Create workflow tree
        self.workflow_tree = WorkflowTreeWidget(workflow)
        self.workflow_tree.populate_tree()
        tree_panel = self.query_one(".workflow-tree-panel", Vertical)
        tree_panel.mount(self.workflow_tree)
        
        # Create execution panel
        self.execution_panel = WorkflowExecutionPanel(workflow)
        execution_container = self.query_one("#execution-details", Static)
        execution_container.remove()
        details_panel = self.query_one(".workflow-details-panel", Vertical)
        details_panel.mount(self.execution_panel)
        
        # Create critical path widget
        self.critical_path_widget = CriticalPathWidget(workflow)
        critical_container = self.query_one("#critical-path", Static)
        critical_container.remove()
        details_panel.mount(self.critical_path_widget)
    
    def clear_workflow(self) -> None:
        """Clear current workflow display"""
        if self.workflow_tree:
            self.workflow_tree.remove()
            self.workflow_tree = None
        
        if self.execution_panel:
            self.execution_panel.remove()
            self.execution_panel = None
        
        if self.critical_path_widget:
            self.critical_path_widget.remove()
            self.critical_path_widget = None
        
        # Restore placeholders
        tree_panel = self.query_one(".workflow-tree-panel", Vertical)
        tree_panel.mount(Static("No workflow loaded", id="workflow-placeholder"))
        
        details_panel = self.query_one(".workflow-details-panel", Vertical)
        details_panel.mount(Static("", id="execution-details"))
        details_panel.mount(Static("", id="critical-path"))
    
    @work(exclusive=True)
    async def start_monitoring(self) -> None:
        """Start workflow monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Update workflow status from task engine
                if self.current_workflow and self.task_engine:
                    await self._update_workflow_status()
                    
                    # Refresh UI components
                    if self.workflow_tree:
                        self.workflow_tree.populate_tree()
                    if self.execution_panel:
                        self.execution_panel.refresh()
                    if self.critical_path_widget:
                        self.critical_path_widget.refresh()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                # Log error and continue
                await asyncio.sleep(15)
    
    async def _update_workflow_status(self) -> None:
        """Update workflow status from task engine"""
        if not self.current_workflow:
            return
        
        # This would integrate with the actual task engine
        # For now, simulate status updates
        workflow = self.current_workflow
        
        # Update overall progress
        workflow.total_progress = workflow.calculate_progress()
        
        # Check if workflow is complete
        all_completed = all(
            node.status in [TaskNodeStatus.COMPLETED, TaskNodeStatus.SKIPPED]
            for node in workflow.nodes.values()
        )
        
        if all_completed and workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
    
    # Event handlers
    @on(Button.Pressed, "#start-workflow")
    def start_workflow(self) -> None:
        """Start workflow execution"""
        if self.current_workflow:
            self.current_workflow.status = WorkflowStatus.RUNNING
            if not self.current_workflow.started_at:
                self.current_workflow.started_at = datetime.now()
            
            self.post_message(StartWorkflowMessage(self.current_workflow))
    
    @on(Button.Pressed, "#pause-workflow")
    def pause_workflow(self) -> None:
        """Pause workflow execution"""
        if self.current_workflow and self.current_workflow.status == WorkflowStatus.RUNNING:
            self.current_workflow.status = WorkflowStatus.PAUSED
            self.post_message(PauseWorkflowMessage(self.current_workflow))
    
    @on(Button.Pressed, "#stop-workflow")
    def stop_workflow(self) -> None:
        """Stop workflow execution"""
        if self.current_workflow:
            self.current_workflow.status = WorkflowStatus.CANCELLED
            self.post_message(StopWorkflowMessage(self.current_workflow))
    
    @on(Button.Pressed, "#refresh-workflow")
    def refresh_workflow(self) -> None:
        """Refresh workflow display"""
        if self.current_workflow:
            self.load_workflow(self.current_workflow)
    
    @on(Button.Pressed, "#export-workflow")
    def export_workflow(self) -> None:
        """Export workflow definition"""
        if self.current_workflow:
            self.post_message(ExportWorkflowMessage(self.current_workflow))
    
    def set_workflow(self, workflow: WorkflowDefinition) -> None:
        """Set current workflow"""
        self.current_workflow = workflow
    
    def update_task_status(self, task_id: str, status: TaskNodeStatus, progress: float = None) -> None:
        """Update status of a specific task"""
        if self.current_workflow and task_id in self.current_workflow.nodes:
            node = self.current_workflow.nodes[task_id]
            node.status = status
            if progress is not None:
                node.progress = progress
            
            # Update tree if available
            if self.workflow_tree:
                self.workflow_tree.update_node_status(task_id, status, progress)
    
    def stop_monitoring(self) -> None:
        """Stop workflow monitoring"""
        self.monitoring_active = False


# Message classes for workflow events
class StartWorkflowMessage(Message):
    """Message to start workflow execution"""
    def __init__(self, workflow: WorkflowDefinition) -> None:
        super().__init__()
        self.workflow = workflow


class PauseWorkflowMessage(Message):
    """Message to pause workflow execution"""
    def __init__(self, workflow: WorkflowDefinition) -> None:
        super().__init__()
        self.workflow = workflow


class StopWorkflowMessage(Message):
    """Message to stop workflow execution"""
    def __init__(self, workflow: WorkflowDefinition) -> None:
        super().__init__()
        self.workflow = workflow


class ExportWorkflowMessage(Message):
    """Message to export workflow"""
    def __init__(self, workflow: WorkflowDefinition) -> None:
        super().__init__()
        self.workflow = workflow