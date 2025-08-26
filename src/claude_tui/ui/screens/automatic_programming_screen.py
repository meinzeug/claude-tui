#!/usr/bin/env python3
"""
Automatic Programming Screen
===========================

Interactive TUI screen for automatic programming workflows.
Provides user interface for creating, executing, and monitoring AI-powered code generation.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Button, Input, Label, Static, 
    Select, TextArea, ProgressBar, RichLog, Tree,
    Collapsible, Checkbox, RadioButton, RadioSet
)
from textual.message import Message
from textual.reactive import reactive
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..integrations.automatic_programming_workflow import (
    AutomaticProgrammingWorkflow, WorkflowStatus, ProgressUpdate
)

logger = logging.getLogger(__name__)


class WorkflowStarted(Message):
    """Message sent when a workflow is started"""
    def __init__(self, workflow_id: str, workflow_name: str) -> None:
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        super().__init__()


class WorkflowCompleted(Message):
    """Message sent when a workflow is completed"""
    def __init__(self, workflow_id: str, result: Dict[str, Any]) -> None:
        self.workflow_id = workflow_id
        self.result = result
        super().__init__()


class AutomaticProgrammingScreen(Screen):
    """
    Main screen for automatic programming features
    """
    
    CSS = """
    AutomaticProgrammingScreen {
        layout: vertical;
    }
    
    .main-container {
        layout: horizontal;
        height: 1fr;
    }
    
    .left-panel {
        width: 30%;
        padding: 1;
        border: solid $primary;
    }
    
    .right-panel {
        width: 70%;
        padding: 1;
        border: solid $secondary;
    }
    
    .workflow-form {
        layout: vertical;
        height: auto;
        margin: 1;
    }
    
    .progress-section {
        height: 40%;
        border: solid $accent;
        margin: 1 0;
    }
    
    .results-section {
        height: 1fr;
        border: solid $success;
        margin: 1 0;
    }
    
    .template-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        height: auto;
        margin: 1 0;
    }
    
    .template-card {
        height: 6;
        border: solid $primary;
        padding: 1;
    }
    
    .status-indicator {
        width: 3;
        text-align: center;
    }
    
    .progress-bar {
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        ("escape", "back", "Back"),
        ("ctrl+n", "new_workflow", "New Workflow"),
        ("ctrl+s", "save_workflow", "Save Workflow"),
        ("ctrl+r", "run_workflow", "Run Workflow"),
        ("f5", "refresh", "Refresh"),
    ]
    
    def __init__(self, config_manager=None):
        super().__init__()
        self.config_manager = config_manager
        self.workflow_manager = AutomaticProgrammingWorkflow(config_manager)
        
        # Screen state
        self.current_workflow_id: Optional[str] = None
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_results: Dict[str, Dict[str, Any]] = {}
        
        # UI components
        self.project_name_input: Optional[Input] = None
        self.project_path_input: Optional[Input] = None
        self.workflow_select: Optional[Select] = None
        self.custom_prompt_area: Optional[TextArea] = None
        self.progress_bar: Optional[ProgressBar] = None
        self.progress_log: Optional[RichLog] = None
        self.results_log: Optional[RichLog] = None
        self.status_label: Optional[Label] = None
        
        # Add progress callback
        self.workflow_manager.add_progress_callback(self._handle_progress_update)
        
        logger.info("AutomaticProgrammingScreen initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the screen layout"""
        yield Header()
        
        with Container(classes="main-container"):
            # Left panel - Configuration and Templates
            with Vertical(classes="left-panel"):
                yield Label("🤖 Automatic Programming", classes="panel-title")
                
                # Workflow configuration form
                with Collapsible(title="Project Configuration", collapsed=False):
                    with Container(classes="workflow-form"):
                        yield Label("Project Name:")
                        self.project_name_input = Input(placeholder="my-awesome-project")
                        yield self.project_name_input
                        
                        yield Label("Project Path:")
                        self.project_path_input = Input(
                            placeholder="/path/to/project",
                            value=str(Path.home() / "Projects")
                        )
                        yield self.project_path_input
                
                # Template selection
                with Collapsible(title="Workflow Templates", collapsed=False):
                    self.workflow_select = Select([
                        ("FastAPI Application", "fastapi_app"),
                        ("React Dashboard", "react_dashboard"),
                        ("Custom Workflow", "custom")
                    ], value="fastapi_app")
                    yield self.workflow_select
                
                # Custom workflow configuration
                with Collapsible(title="Custom Requirements", collapsed=True, id="custom-section"):
                    yield Label("Describe what you want to build:")
                    self.custom_prompt_area = TextArea(
                        placeholder="Create a FastAPI application with user authentication, JWT tokens, and a database...",
                        show_line_numbers=False
                    )
                    yield self.custom_prompt_area
                
                # Action buttons
                with Horizontal():
                    yield Button("▶ Start Workflow", id="start-btn", variant="success")
                    yield Button("⏹ Cancel", id="cancel-btn", variant="error")
                    yield Button("📋 Templates", id="templates-btn")
            
            # Right panel - Progress and Results
            with Vertical(classes="right-panel"):
                # Status section
                with Container(classes="status-section"):
                    self.status_label = Label("Ready to start workflow")
                    yield self.status_label
                
                # Progress section
                with Collapsible(title="Workflow Progress", classes="progress-section"):
                    self.progress_bar = ProgressBar(show_eta=True, show_percentage=True)
                    yield self.progress_bar
                    
                    self.progress_log = RichLog(auto_scroll=True, markup=True)
                    yield self.progress_log
                
                # Results section  
                with Collapsible(title="Generated Code & Results", classes="results-section"):
                    self.results_log = RichLog(auto_scroll=True, markup=True)
                    yield self.results_log
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the screen after mounting"""
        self.title = "Automatic Programming"
        self.sub_title = "AI-Powered Code Generation"
        
        # Load available templates
        self._refresh_templates()
        
        # Initial status
        self._update_status("Ready to create your next project!", "info")
    
    @on(Select.Changed, "#workflow-select")
    def on_workflow_template_changed(self, event: Select.Changed) -> None:
        """Handle workflow template selection change"""
        template_id = event.value
        custom_section = self.query_one("#custom-section", Collapsible)
        
        if template_id == "custom":
            custom_section.collapsed = False
            self._update_status("Enter custom requirements below", "info")
        else:
            custom_section.collapsed = True
            templates = self.workflow_manager.get_available_templates()
            if template_id in templates:
                template = templates[template_id]
                self._update_status(f"Selected: {template['name']}", "info")
    
    @on(Button.Pressed, "#start-btn")
    def on_start_workflow(self, event: Button.Pressed) -> None:
        """Handle start workflow button press"""
        self._start_workflow()
    
    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_workflow(self, event: Button.Pressed) -> None:
        """Handle cancel workflow button press"""
        if self.current_workflow_id:
            self._cancel_current_workflow()
    
    @on(Button.Pressed, "#templates-btn")
    def on_show_templates(self, event: Button.Pressed) -> None:
        """Handle templates button press"""
        self._show_template_gallery()
    
    @work(exclusive=True)
    async def _start_workflow(self) -> None:
        """Start the selected workflow"""
        try:
            # Validate inputs
            project_name = self.project_name_input.value.strip()
            project_path = self.project_path_input.value.strip()
            
            if not project_name:
                self._update_status("Please enter a project name", "error")
                return
            
            if not project_path:
                self._update_status("Please enter a project path", "error")
                return
            
            # Create project path
            full_project_path = Path(project_path) / project_name
            
            # Get selected workflow
            workflow_type = self.workflow_select.value
            
            self._update_status("Creating workflow...", "info")
            
            # Create workflow based on type
            if workflow_type == "custom":
                custom_prompt = self.custom_prompt_area.text.strip()
                if not custom_prompt:
                    self._update_status("Please enter custom requirements", "error")
                    return
                
                workflow_id = await self.workflow_manager.create_custom_workflow(
                    name=f"Custom: {project_name}",
                    description=custom_prompt[:100] + "...",
                    prompt=custom_prompt,
                    project_path=full_project_path
                )
            else:
                # Use template
                workflow_id = await self.workflow_manager.create_workflow_from_template(
                    template_name=workflow_type,
                    project_name=project_name,
                    project_path=full_project_path
                )
            
            self.current_workflow_id = workflow_id
            self.active_workflows[workflow_id] = {
                "name": project_name,
                "started_at": datetime.now(),
                "status": "running"
            }
            
            # Update UI
            self._update_status(f"Starting workflow: {project_name}", "success")
            self.progress_log.write(f"🚀 Starting workflow: {workflow_id}")
            self.progress_log.write(f"📁 Project: {project_name}")
            self.progress_log.write(f"🎯 Template: {workflow_type}")
            self.progress_log.write(f"📂 Path: {full_project_path}")
            
            # Send workflow started message
            self.post_message(WorkflowStarted(workflow_id, project_name))
            
            # Execute workflow
            result = await self.workflow_manager.execute_workflow(workflow_id)
            
            # Handle completion
            self._handle_workflow_completion(workflow_id, result)
            
        except Exception as e:
            logger.error(f"Error starting workflow: {e}")
            self._update_status(f"Error starting workflow: {str(e)}", "error")
            self.progress_log.write(f"❌ Error: {str(e)}")
    
    async def _cancel_current_workflow(self) -> None:
        """Cancel the currently running workflow"""
        if not self.current_workflow_id:
            return
        
        try:
            success = await self.workflow_manager.cancel_workflow(self.current_workflow_id)
            if success:
                self._update_status("Workflow cancelled", "warning")
                self.progress_log.write(f"⏹ Workflow {self.current_workflow_id} cancelled")
                self.active_workflows[self.current_workflow_id]["status"] = "cancelled"
            else:
                self._update_status("Failed to cancel workflow", "error")
            
        except Exception as e:
            logger.error(f"Error cancelling workflow: {e}")
            self._update_status(f"Error cancelling workflow: {str(e)}", "error")
    
    def _handle_progress_update(self, update: ProgressUpdate) -> None:
        """Handle progress updates from workflow manager"""
        try:
            # Update progress bar
            if self.progress_bar:
                self.progress_bar.progress = update.progress * 100
            
            # Add to progress log
            status_emoji = {
                "starting": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(update.step_status, "⏳")
            
            timestamp = datetime.fromtimestamp(update.timestamp).strftime("%H:%M:%S")
            self.progress_log.write(
                f"[{timestamp}] {status_emoji} {update.step_name}: {update.message}"
            )
            
            # Update main status
            if update.step_status == "starting":
                self._update_status(f"Running: {update.step_name}", "info")
            elif update.step_status == "completed":
                self._update_status(f"Completed: {update.step_name}", "success")
            elif update.step_status == "failed":
                self._update_status(f"Failed: {update.step_name}", "error")
            
        except Exception as e:
            logger.error(f"Error handling progress update: {e}")
    
    def _handle_workflow_completion(self, workflow_id: str, result) -> None:
        """Handle workflow completion"""
        try:
            # Update workflow status
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = "completed"
                self.active_workflows[workflow_id]["completed_at"] = datetime.now()
            
            # Store results
            self.workflow_results[workflow_id] = {
                "result": result,
                "completed_at": datetime.now()
            }
            
            # Update UI
            if result.status == WorkflowStatus.COMPLETED:
                self._update_status(f"Workflow completed successfully! ✅", "success")
                self.progress_log.write("🎉 Workflow completed successfully!")
            else:
                self._update_status(f"Workflow completed with errors ⚠️", "warning")
                self.progress_log.write("⚠️ Workflow completed with errors")
            
            # Display results
            self._display_workflow_results(result)
            
            # Send completion message
            self.post_message(WorkflowCompleted(workflow_id, result.__dict__))
            
            # Reset current workflow
            self.current_workflow_id = None
            
        except Exception as e:
            logger.error(f"Error handling workflow completion: {e}")
            self._update_status(f"Error processing results: {str(e)}", "error")
    
    def _display_workflow_results(self, result) -> None:
        """Display workflow results in the results panel"""
        try:
            self.results_log.clear()
            
            # Summary
            self.results_log.write("[bold green]📊 Workflow Results Summary[/bold green]")
            self.results_log.write(f"Status: {result.status.value}")
            self.results_log.write(f"Steps completed: {result.steps_completed}/{result.steps_total}")
            self.results_log.write(f"Duration: {result.duration:.2f} seconds")
            
            # Created files
            if result.created_files:
                self.results_log.write("\n[bold blue]📄 Created Files:[/bold blue]")
                for file_path in result.created_files[:10]:  # Show first 10
                    self.results_log.write(f"  • {file_path}")
                if len(result.created_files) > 10:
                    self.results_log.write(f"  ... and {len(result.created_files) - 10} more")
            
            # Modified files
            if result.modified_files:
                self.results_log.write("\n[bold yellow]✏️ Modified Files:[/bold yellow]")
                for file_path in result.modified_files[:10]:  # Show first 10
                    self.results_log.write(f"  • {file_path}")
                if len(result.modified_files) > 10:
                    self.results_log.write(f"  ... and {len(result.modified_files) - 10} more")
            
            # Errors
            if result.errors:
                self.results_log.write("\n[bold red]❌ Errors:[/bold red]")
                for error in result.errors[:5]:  # Show first 5 errors
                    self.results_log.write(f"  • {error}")
                if len(result.errors) > 5:
                    self.results_log.write(f"  ... and {len(result.errors) - 5} more errors")
            
            # Step results (collapsed view)
            if result.results:
                self.results_log.write("\n[bold cyan]🔧 Step Details:[/bold cyan]")
                for step_id, step_result in result.results.items():
                    status = step_result.get("status", "unknown")
                    status_emoji = "✅" if status == "success" else "❌"
                    self.results_log.write(f"  {status_emoji} {step_id}: {status}")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            self.results_log.write(f"❌ Error displaying results: {str(e)}")
    
    def _refresh_templates(self) -> None:
        """Refresh the template selection"""
        try:
            templates = self.workflow_manager.get_available_templates()
            options = []
            
            for template_id, template_info in templates.items():
                options.append((template_info["name"], template_id))
            
            options.append(("Custom Workflow", "custom"))
            
            if self.workflow_select:
                self.workflow_select.set_options(options)
                
        except Exception as e:
            logger.error(f"Error refreshing templates: {e}")
    
    def _show_template_gallery(self) -> None:
        """Show template gallery with descriptions"""
        try:
            templates = self.workflow_manager.get_available_templates()
            
            self.results_log.clear()
            self.results_log.write("[bold green]📚 Available Workflow Templates[/bold green]")
            
            for template_id, template_info in templates.items():
                self.results_log.write(f"\n[bold blue]🔧 {template_info['name']}[/bold blue]")
                self.results_log.write(f"Description: {template_info['description']}")
                self.results_log.write(f"Steps: {template_info['steps_count']}")
                
        except Exception as e:
            logger.error(f"Error showing template gallery: {e}")
    
    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update the status label"""
        try:
            status_colors = {
                "info": "blue",
                "success": "green", 
                "warning": "yellow",
                "error": "red"
            }
            
            status_icons = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌"
            }
            
            color = status_colors.get(status_type, "white")
            icon = status_icons.get(status_type, "")
            
            if self.status_label:
                self.status_label.update(f"{icon} {message}")
                # You could also change the label's color/style here if needed
                
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def action_back(self) -> None:
        """Go back to main screen"""
        self.dismiss()
    
    def action_new_workflow(self) -> None:
        """Create new workflow"""
        if self.project_name_input:
            self.project_name_input.value = ""
        if self.custom_prompt_area:
            self.custom_prompt_area.text = ""
        self._update_status("Ready for new workflow", "info")
    
    def action_save_workflow(self) -> None:
        """Save current workflow configuration"""
        # TODO: Implement workflow saving
        self._update_status("Workflow save not yet implemented", "warning")
    
    def action_run_workflow(self) -> None:
        """Run the configured workflow"""
        self._start_workflow()
    
    def action_refresh(self) -> None:
        """Refresh the screen"""
        self._refresh_templates()
        self._update_status("Screen refreshed", "success")
    
    async def on_screen_suspend(self) -> None:
        """Handle screen suspension"""
        # Cancel any running workflows when switching screens
        if self.current_workflow_id:
            await self._cancel_current_workflow()
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Cancel any active workflows
            if self.current_workflow_id:
                await self._cancel_current_workflow()
            
            # Remove progress callback
            self.workflow_manager.remove_progress_callback(self._handle_progress_update)
            
            # Cleanup workflow manager
            await self.workflow_manager.cleanup()
            
            logger.info("AutomaticProgrammingScreen cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")