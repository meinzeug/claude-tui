#!/usr/bin/env python3
"""
Enhanced Workspace Screen - Main development workspace with integrated widgets,
responsive layout, and advanced features.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from textual import on, work
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.widgets import TabbedContent, TabPane, Label, Button, Static
from textual.screen import Screen
from textual.message import Message
from textual.reactive import reactive

# Import all UI widgets
from claude_tiu.ui.widgets.project_tree import ProjectTree, FileSelectedMessage
from claude_tiu.ui.widgets.task_dashboard import TaskDashboard, AddTaskMessage, ShowAnalyticsMessage
from claude_tiu.ui.widgets.console_widget import ConsoleWidget, ShowCommandTemplatesMessage
from claude_tiu.ui.widgets.placeholder_alert import (
    PlaceholderAlert, PlaceholderAlertTriggeredMessage, AutoFixIssuesMessage,
    StartCompletionMessage, ExportPlaceholderReportMessage
)
from claude_tiu.ui.widgets.progress_intelligence import (
    ProgressIntelligence, ValidateNowMessage, ShowValidationDetailsMessage
)
from claude_tiu.ui.widgets.workflow_visualizer import (
    WorkflowVisualizerWidget, StartWorkflowMessage, PauseWorkflowMessage,
    StopWorkflowMessage, ExportWorkflowMessage
)
from claude_tiu.ui.widgets.metrics_dashboard import (
    MetricsDashboardWidget, ExportMetricsMessage, ShowMetricsSettingsMessage
)
from claude_tiu.ui.widgets.modal_dialogs import (
    ConfigurationModal, CommandTemplatesModal, TaskCreationModal,
    ConfirmationModal, UseTemplateMessage, CreateTaskMessage,
    SaveConfigMessage, ExportConfigMessage
)

# Import models
from claude_tiu.models.project import Project
from claude_tiu.models.task import Task


class WorkspaceLayout(Enum):
    """Workspace layout modes"""
    STANDARD = "standard"      # Classic 3-panel layout
    FOCUSED = "focused"        # Single main panel
    DEVELOPMENT = "development" # Code-focused layout
    MONITORING = "monitoring"   # Metrics and monitoring focused
    COLLABORATION = "collaboration"  # Team-oriented layout


class WorkspaceScreen(Screen):
    """Enhanced workspace screen with integrated widget management"""
    
    current_project: reactive[Optional[Project]] = reactive(None)
    layout_mode: reactive[WorkspaceLayout] = reactive(WorkspaceLayout.STANDARD)
    console_visible: reactive[bool] = reactive(True)
    validation_visible: reactive[bool] = reactive(True)
    workflows_visible: reactive[bool] = reactive(False)
    metrics_visible: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance
        
        # Initialize all widgets with proper dependencies
        self.project_tree = ProjectTree(app_instance.project_manager)
        self.task_dashboard = TaskDashboard(app_instance.project_manager)
        self.console_widget = ConsoleWidget(app_instance.ai_interface)
        self.placeholder_alert = PlaceholderAlert()
        self.progress_widget = ProgressIntelligence()
        self.workflow_visualizer = WorkflowVisualizerWidget(app_instance.task_engine)
        self.metrics_dashboard = MetricsDashboardWidget()
        
        # Layout state
        self.sidebar_width = 30
        self.bottom_panel_height = 15
        self.widget_panels: Dict[str, Container] = {}
        
    def compose(self):
        """Compose the workspace layout"""
        with Container(id="workspace-root"):
            # Header with project info and controls
            yield self._create_workspace_header()
            
            # Main workspace content
            with Container(id="workspace-content", classes="workspace-main"):
                # Left sidebar - project navigation and overview
                with Container(id="sidebar", classes="sidebar"):
                    yield self.project_tree
                    yield Static("", id="sidebar-spacer")
                
                # Center area - main content with tabs
                with Container(id="main-content", classes="main-content"):
                    with TabbedContent(id="main-tabs"):
                        # Task Dashboard Tab
                        with TabPane("Tasks", id="tasks-tab"):
                            yield self.task_dashboard
                        
                        # Code Editor Tab (placeholder for now)
                        with TabPane("Editor", id="editor-tab"):
                            yield Static("Code editor integration coming soon...", classes="coming-soon")
                        
                        # Workflow Visualizer Tab
                        with TabPane("Workflows", id="workflows-tab"):
                            yield self.workflow_visualizer
                        
                        # Metrics Dashboard Tab
                        with TabPane("Metrics", id="metrics-tab"):
                            yield self.metrics_dashboard
                
                # Right panel - dynamic content based on context
                with Container(id="right-panel", classes="right-panel"):
                    with TabbedContent(id="right-tabs"):
                        # AI Console
                        with TabPane("Console", id="console-tab"):
                            yield self.console_widget
                        
                        # Progress Intelligence
                        with TabPane("Validation", id="validation-tab"):
                            yield self.progress_widget
                        
                        # Project Inspector
                        with TabPane("Inspector", id="inspector-tab"):
                            yield Static("File inspector and properties", classes="coming-soon")
            
            # Bottom status and alerts area
            with Container(id="bottom-panel", classes="bottom-panel"):
                yield self.placeholder_alert
                yield Static("", id="status-info")
        
        # Quick action buttons
        yield self._create_quick_actions()
    
    def _create_workspace_header(self) -> Container:
        """Create workspace header with project info and controls"""
        with Container(classes="workspace-header"):
            # Project information
            with Horizontal(classes="project-info"):
                yield Label("📁 No Project", id="project-title", classes="project-name")
                yield Label("Ready", id="project-status", classes="project-status")
            
            # Layout controls
            with Horizontal(classes="layout-controls"):
                yield Button("📋 Standard", id="layout-standard", classes="layout-btn")
                yield Button("🔍 Focused", id="layout-focused", classes="layout-btn")
                yield Button("💻 Dev", id="layout-development", classes="layout-btn")
                yield Button("📊 Monitor", id="layout-monitoring", classes="layout-btn")
            
            # Quick actions
            with Horizontal(classes="quick-actions"):
                yield Button("💾 Save", id="quick-save", variant="primary")
                yield Button("▶️ Run", id="quick-run", variant="success")
                yield Button("🔧 Build", id="quick-build")
                yield Button("🧪 Test", id="quick-test")
        
        return Container()
    
    def _create_quick_actions(self) -> Container:
        """Create floating quick action buttons"""
        with Container(id="quick-actions-float", classes="floating-actions"):
            yield Button("🤖 AI", id="ai-assist", classes="quick-btn")
            yield Button("🔍 Find", id="quick-find", classes="quick-btn")
            yield Button("📋 Tasks", id="quick-tasks", classes="quick-btn")
            yield Button("⚙️ Settings", id="quick-settings", classes="quick-btn")
        
        return Container()
    
    async def on_mount(self) -> None:
        """Initialize workspace when mounted"""
        # Set initial focus
        self.project_tree.focus()
        
        # Start workspace monitoring
        self.start_workspace_monitoring()
    
    def watch_current_project(self, project: Optional[Project]) -> None:
        """React to project changes"""
        if project:
            self._update_project_ui(project)
            self.project_tree.set_project(str(project.root_path))
        else:
            self._clear_project_ui()
    
    def watch_layout_mode(self, layout: WorkspaceLayout) -> None:
        """React to layout mode changes"""
        self._apply_layout(layout)
    
    def _update_project_ui(self, project: Project) -> None:
        """Update UI with project information"""
        title_widget = self.query_one("#project-title", Label)
        title_widget.update(f"📁 {project.name}")
        
        status_widget = self.query_one("#project-status", Label)
        status_widget.update("Active")
        
        # Update widgets with project context
        self.task_dashboard.set_project(project)
        if hasattr(self.workflow_visualizer, 'set_project'):
            self.workflow_visualizer.set_project(project)
    
    def _clear_project_ui(self) -> None:
        """Clear project-specific UI elements"""
        title_widget = self.query_one("#project-title", Label)
        title_widget.update("📁 No Project")
        
        status_widget = self.query_one("#project-status", Label)
        status_widget.update("Ready")
    
    def _apply_layout(self, layout: WorkspaceLayout) -> None:
        """Apply workspace layout"""
        # This would adjust CSS classes and visibility of panels
        # For now, just log the layout change
        pass
    
    @work(exclusive=True)
    async def start_workspace_monitoring(self) -> None:
        """Start workspace monitoring tasks"""
        while True:
            try:
                # Update workspace status
                await self._update_workspace_status()
                
                # Check for project changes
                if self.current_project:
                    await self._check_project_health()
                
                await asyncio.sleep(15)  # Update every 15 seconds
                
            except Exception as e:
                await asyncio.sleep(30)
    
    async def _update_workspace_status(self) -> None:
        """Update workspace status information"""
        # This would collect and display various workspace metrics
        pass
    
    async def _check_project_health(self) -> None:
        """Check current project health and status"""
        if self.current_project:
            # This would integrate with project validation
            pass
    
    # Event handlers for widget messages
    
    @on(FileSelectedMessage)
    async def handle_file_selected(self, message: FileSelectedMessage) -> None:
        """Handle file selection from project tree"""
        # This would open the file in the editor tab
        # For now, just show in console
        if hasattr(self.console_widget, 'add_system_message'):
            self.console_widget.add_system_message(f"File selected: {message.file_path}")
    
    @on(AddTaskMessage)
    async def handle_add_task(self, message: AddTaskMessage) -> None:
        """Handle add task request"""
        if self.current_project:
            task_modal = TaskCreationModal(str(self.current_project.id))
            await self.app.push_screen(task_modal)
        else:
            self.notify("No project loaded", severity="warning")
    
    @on(ShowAnalyticsMessage)
    async def handle_show_analytics(self, message: ShowAnalyticsMessage) -> None:
        """Handle show analytics request"""
        # Switch to metrics tab
        main_tabs = self.query_one("#main-tabs", TabbedContent)
        main_tabs.active = "metrics-tab"
    
    @on(ShowCommandTemplatesMessage)
    async def handle_show_templates(self, message: ShowCommandTemplatesMessage) -> None:
        """Handle show command templates request"""
        templates_modal = CommandTemplatesModal()
        await self.app.push_screen(templates_modal)
    
    @on(UseTemplateMessage)
    async def handle_use_template(self, message: UseTemplateMessage) -> None:
        """Handle use template message"""
        # Execute the template command in AI console
        if hasattr(self.console_widget, 'execute_ai_command'):
            self.console_widget.execute_ai_command(message.command)
        
        # Switch to console tab
        right_tabs = self.query_one("#right-tabs", TabbedContent)
        right_tabs.active = "console-tab"
    
    @on(CreateTaskMessage)
    async def handle_create_task(self, message: CreateTaskMessage) -> None:
        """Handle create task message"""
        # This would integrate with the task engine to create the task
        task_data = message.task_data
        
        # For now, just add to console log
        if hasattr(self.console_widget, 'add_success_message'):
            self.console_widget.add_success_message(f"Task created: {task_data['name']}")
        
        # Refresh task dashboard
        self.task_dashboard.refresh()
    
    @on(PlaceholderAlertTriggeredMessage)
    async def handle_placeholder_alert(self, message: PlaceholderAlertTriggeredMessage) -> None:
        """Handle placeholder alert"""
        # This alert is already shown by the PlaceholderAlert widget
        # Just log it in console
        if hasattr(self.console_widget, 'add_warning_message'):
            self.console_widget.add_warning_message(
                f"Placeholder code detected: {len(message.issues)} issues found"
            )
    
    @on(ValidateNowMessage)
    async def handle_validate_now(self, message: ValidateNowMessage) -> None:
        """Handle immediate validation request"""
        # Trigger placeholder scan
        self.placeholder_alert.trigger_immediate_scan()
        
        # Update progress widget
        self.progress_widget.validate_now()
        
        if hasattr(self.console_widget, 'add_system_message'):
            self.console_widget.add_system_message("Validation initiated")
    
    @on(StartWorkflowMessage)
    async def handle_start_workflow(self, message: StartWorkflowMessage) -> None:
        """Handle workflow start request"""
        # This would integrate with the task engine
        if hasattr(self.console_widget, 'add_success_message'):
            self.console_widget.add_success_message(f"Workflow started: {message.workflow.name}")
    
    # Layout control handlers
    
    @on(Button.Pressed, "#layout-standard")
    def set_standard_layout(self) -> None:
        """Set standard layout"""
        self.layout_mode = WorkspaceLayout.STANDARD
    
    @on(Button.Pressed, "#layout-focused")
    def set_focused_layout(self) -> None:
        """Set focused layout"""
        self.layout_mode = WorkspaceLayout.FOCUSED
    
    @on(Button.Pressed, "#layout-development")
    def set_development_layout(self) -> None:
        """Set development layout"""
        self.layout_mode = WorkspaceLayout.DEVELOPMENT
    
    @on(Button.Pressed, "#layout-monitoring")
    def set_monitoring_layout(self) -> None:
        """Set monitoring layout"""
        self.layout_mode = WorkspaceLayout.MONITORING
    
    # Quick action handlers
    
    @on(Button.Pressed, "#quick-save")
    async def quick_save(self) -> None:
        """Quick save current work"""
        if self.current_project:
            await self.app_instance.action_save_project()
        else:
            self.notify("No project to save", severity="warning")
    
    @on(Button.Pressed, "#quick-run")
    async def quick_run(self) -> None:
        """Quick run/execute current task"""
        # This would execute the current task or run the project
        if hasattr(self.console_widget, 'add_system_message'):
            self.console_widget.add_system_message("Quick run initiated")
    
    @on(Button.Pressed, "#quick-build")
    async def quick_build(self) -> None:
        """Quick build project"""
        # This would trigger a build process
        if hasattr(self.console_widget, 'add_system_message'):
            self.console_widget.add_system_message("Build process started")
    
    @on(Button.Pressed, "#quick-test")
    async def quick_test(self) -> None:
        """Quick test execution"""
        # This would run tests
        if hasattr(self.console_widget, 'add_system_message'):
            self.console_widget.add_system_message("Test execution started")
    
    @on(Button.Pressed, "#ai-assist")
    async def ai_assist(self) -> None:
        """Focus AI assistant"""
        right_tabs = self.query_one("#right-tabs", TabbedContent)
        right_tabs.active = "console-tab"
        self.console_widget.focus()
    
    @on(Button.Pressed, "#quick-find")
    async def quick_find(self) -> None:
        """Quick find/search"""
        # This would open a search dialog
        self.notify("Quick find: Use Ctrl+F for now", severity="info")
    
    @on(Button.Pressed, "#quick-tasks")
    async def quick_tasks(self) -> None:
        """Quick task management"""
        main_tabs = self.query_one("#main-tabs", TabbedContent)
        main_tabs.active = "tasks-tab"
    
    @on(Button.Pressed, "#quick-settings")
    async def quick_settings(self) -> None:
        """Quick settings"""
        await self.app_instance.action_settings()
    
    # Public interface methods
    
    async def set_project(self, project: Project) -> None:
        """Set the current project"""
        self.current_project = project
    
    async def toggle_console(self) -> None:
        """Toggle console visibility"""
        self.console_visible = not self.console_visible
        # Implementation would show/hide the console panel
    
    async def toggle_validation(self) -> None:
        """Toggle validation panel visibility"""
        self.validation_visible = not self.validation_visible
        # Implementation would show/hide the validation panel
    
    async def toggle_workflows(self) -> None:
        """Toggle workflow visualizer visibility"""
        self.workflows_visible = not self.workflows_visible
        main_tabs = self.query_one("#main-tabs", TabbedContent)
        main_tabs.active = "workflows-tab"
    
    async def toggle_metrics(self) -> None:
        """Toggle metrics dashboard visibility"""
        self.metrics_visible = not self.metrics_visible
        main_tabs = self.query_one("#main-tabs", TabbedContent)
        main_tabs.active = "metrics-tab"
    
    async def focus_console(self) -> None:
        """Focus the AI console"""
        right_tabs = self.query_one("#right-tabs", TabbedContent)
        right_tabs.active = "console-tab"
        if hasattr(self.console_widget, 'input_widget') and self.console_widget.input_widget:
            self.console_widget.input_widget.focus()
    
    async def refresh(self) -> None:
        """Refresh all workspace components"""
        # Refresh all widgets
        self.project_tree.refresh()
        self.task_dashboard.refresh()
        self.progress_widget.validate_now()
        
        # Update UI
        if self.current_project:
            self._update_project_ui(self.current_project)
    
    def get_active_file(self) -> Optional[str]:
        """Get currently active file"""
        # This would return the file currently being edited
        return None
    
    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state for persistence"""
        return {
            "layout_mode": self.layout_mode.value,
            "console_visible": self.console_visible,
            "validation_visible": self.validation_visible,
            "workflows_visible": self.workflows_visible,
            "metrics_visible": self.metrics_visible,
            "sidebar_width": self.sidebar_width,
            "bottom_panel_height": self.bottom_panel_height
        }
    
    def restore_workspace_state(self, state: Dict[str, Any]) -> None:
        """Restore workspace state"""
        self.layout_mode = WorkspaceLayout(state.get("layout_mode", "standard"))
        self.console_visible = state.get("console_visible", True)
        self.validation_visible = state.get("validation_visible", True)
        self.workflows_visible = state.get("workflows_visible", False)
        self.metrics_visible = state.get("metrics_visible", False)
        self.sidebar_width = state.get("sidebar_width", 30)
        self.bottom_panel_height = state.get("bottom_panel_height", 15)