#!/usr/bin/env python3
"""
Automatic Programming UI Widgets
================================

Specialized widgets for displaying automatic programming workflows,
progress, and results with real-time updates.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Label, ProgressBar, Tree, RichLog, Button, 
    Collapsible, TextArea, Select, Input, Checkbox
)
from textual.reactive import reactive, var
from textual.message import Message

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.tree import Tree as RichTree


class WorkflowProgressWidget(Static):
    """
    Real-time progress display for automatic programming workflows
    """
    
    def __init__(self, workflow_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.workflow_id = workflow_id
        self.current_step = ""
        self.total_steps = 0
        self.completed_steps = 0
        self.step_progress = {}
        self.start_time = None
        
    def compose(self) -> ComposeResult:
        """Compose the progress widget"""
        with Vertical():
            # Overall progress
            yield Label("Workflow Progress", classes="widget-title")
            
            self.overall_progress = ProgressBar(
                total=100, 
                show_eta=True, 
                show_percentage=True,
                classes="overall-progress"
            )
            yield self.overall_progress
            
            # Current step info
            self.step_info = Label("Ready to start...", classes="step-info")
            yield self.step_info
            
            # Step-by-step progress
            with Collapsible(title="Step Details", collapsed=False):
                self.step_tree = Tree("Workflow Steps")
                yield self.step_tree
    
    def update_workflow_info(self, workflow_id: str, total_steps: int):
        """Update workflow information"""
        self.workflow_id = workflow_id
        self.total_steps = total_steps
        self.start_time = datetime.now()
        self.completed_steps = 0
        
        # Initialize step tree
        self.step_tree.clear()
        root = self.step_tree.root
        root.set_label(f"Workflow: {workflow_id[:8]}...")
    
    def update_step_progress(self, step_id: str, step_name: str, progress: float, status: str):
        """Update progress for a specific step"""
        self.current_step = step_name
        overall_progress = (self.completed_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        
        # Update overall progress
        self.overall_progress.progress = overall_progress
        
        # Update step info
        status_emoji = {
            "starting": "🔄",
            "in_progress": "⏳", 
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️"
        }.get(status, "❓")
        
        self.step_info.update(f"{status_emoji} {step_name} ({progress:.0%})")
        
        # Update step tree
        self._update_step_tree(step_id, step_name, progress, status)
        
        if status == "completed":
            self.completed_steps += 1
    
    def _update_step_tree(self, step_id: str, step_name: str, progress: float, status: str):
        """Update the step tree display"""
        try:
            # Find or create step node
            root = self.step_tree.root
            step_node = None
            
            for node in root.children:
                if node.data and node.data.get("step_id") == step_id:
                    step_node = node
                    break
            
            if not step_node:
                step_node = root.add(f"{step_name}", data={"step_id": step_id})
            
            # Update node label with status
            status_emoji = {
                "starting": "🔄",
                "in_progress": "⏳",
                "completed": "✅", 
                "failed": "❌",
                "skipped": "⏭️"
            }.get(status, "❓")
            
            step_node.set_label(f"{status_emoji} {step_name} ({progress:.0%})")
            
        except Exception as e:
            # Fallback: just update the step info
            pass
    
    def set_error(self, error_message: str):
        """Set error state"""
        self.step_info.update(f"❌ Error: {error_message}")
        self.overall_progress.progress = 0
    
    def set_completed(self, duration: float):
        """Set completed state"""
        self.step_info.update(f"✅ Completed in {duration:.1f}s")
        self.overall_progress.progress = 100


class CodeResultsViewer(Static):
    """
    Widget for viewing generated code with syntax highlighting and validation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generated_files = {}
        self.current_file = None
        
    def compose(self) -> ComposeResult:
        """Compose the code viewer"""
        with Vertical():
            yield Label("Generated Code", classes="widget-title")
            
            # File selector
            with Horizontal():
                self.file_select = Select([], prompt="Select file to view")
                yield self.file_select
                
                self.validation_btn = Button("✓ Validate", classes="validation-btn")
                yield self.validation_btn
            
            # Code display area
            with ScrollableContainer():
                self.code_display = RichLog(
                    auto_scroll=False,
                    markup=True,
                    classes="code-display"
                )
                yield self.code_display
            
            # Validation results
            with Collapsible(title="Validation Results", collapsed=True):
                self.validation_display = RichLog(
                    auto_scroll=True,
                    markup=True,
                    classes="validation-display"
                )
                yield self.validation_display
    
    @on(Select.Changed)
    def on_file_selected(self, event: Select.Changed):
        """Handle file selection"""
        if event.value and event.value in self.generated_files:
            self._display_file(event.value)
    
    @on(Button.Pressed, ".validation-btn")
    def on_validate_code(self, event: Button.Pressed):
        """Handle validation request"""
        if self.current_file:
            self._validate_current_file()
    
    def add_generated_file(self, file_path: str, content: str, file_type: str = "python"):
        """Add a generated file to the viewer"""
        self.generated_files[file_path] = {
            "content": content,
            "type": file_type,
            "created_at": datetime.now()
        }
        
        # Update file selector
        self._update_file_selector()
        
        # Auto-select first file if none selected
        if not self.current_file:
            self.current_file = file_path
            self._display_file(file_path)
    
    def _update_file_selector(self):
        """Update the file selection dropdown"""
        options = []
        for file_path in self.generated_files.keys():
            # Show just filename for cleaner display
            filename = Path(file_path).name
            options.append((filename, file_path))
        
        self.file_select.set_options(options)
    
    def _display_file(self, file_path: str):
        """Display the selected file with syntax highlighting"""
        if file_path not in self.generated_files:
            return
        
        self.current_file = file_path
        file_info = self.generated_files[file_path]
        content = file_info["content"]
        file_type = file_info["file_type"]
        
        # Clear previous content
        self.code_display.clear()
        
        # Show file header
        self.code_display.write(f"[bold blue]📄 {file_path}[/bold blue]")
        self.code_display.write(f"[dim]Type: {file_type} | Created: {file_info['created_at'].strftime('%H:%M:%S')}[/dim]")
        self.code_display.write("")
        
        # Display code with syntax highlighting
        try:
            # Determine lexer based on file extension
            file_ext = Path(file_path).suffix.lower()
            lexer_map = {
                '.py': 'python',
                '.js': 'javascript', 
                '.jsx': 'javascript',
                '.ts': 'typescript',
                '.tsx': 'typescript',
                '.html': 'html',
                '.css': 'css',
                '.json': 'json',
                '.md': 'markdown',
                '.yml': 'yaml',
                '.yaml': 'yaml'
            }
            
            lexer = lexer_map.get(file_ext, 'text')
            
            # Create syntax highlighted content
            syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
            self.code_display.write(syntax)
            
        except Exception as e:
            # Fallback to plain text
            self.code_display.write(f"[red]Error highlighting code: {e}[/red]")
            self.code_display.write(content)
    
    @work(exclusive=True)
    async def _validate_current_file(self):
        """Validate the currently displayed file"""
        if not self.current_file:
            return
        
        file_info = self.generated_files[self.current_file]
        content = file_info["content"]
        file_type = file_info["file_type"]
        
        self.validation_display.clear()
        self.validation_display.write(f"[yellow]🔍 Validating {self.current_file}...[/yellow]")
        
        try:
            validation_results = await self._perform_validation(content, file_type)
            
            # Display results
            if validation_results["valid"]:
                self.validation_display.write("[green]✅ Validation passed![/green]")
            else:
                self.validation_display.write("[red]❌ Validation failed![/red]")
            
            # Show details
            for issue in validation_results.get("issues", []):
                self.validation_display.write(f"[red]• {issue}[/red]")
            
            for warning in validation_results.get("warnings", []):
                self.validation_display.write(f"[yellow]• {warning}[/yellow]")
            
            # Show recommendations
            for rec in validation_results.get("recommendations", []):
                self.validation_display.write(f"[blue]💡 {rec}[/blue]")
                
        except Exception as e:
            self.validation_display.write(f"[red]❌ Validation error: {e}[/red]")
    
    async def _perform_validation(self, content: str, file_type: str) -> Dict[str, Any]:
        """Perform validation on the content"""
        results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            if file_type == "python":
                # Basic Python syntax validation
                try:
                    compile(content, "<string>", "exec")
                except SyntaxError as e:
                    results["valid"] = False
                    results["issues"].append(f"Syntax error at line {e.lineno}: {e.msg}")
                
                # Check for common patterns
                if "TODO" in content:
                    results["warnings"].append("Contains TODO comments")
                if "pass" in content and content.count("pass") > 2:
                    results["warnings"].append("Multiple 'pass' statements found")
                if not content.strip():
                    results["issues"].append("File is empty")
                
                # Recommendations
                if "import" not in content:
                    results["recommendations"].append("Consider adding necessary imports")
                if not content.startswith('"""') and not content.startswith("'''"):
                    results["recommendations"].append("Consider adding a module docstring")
            
            elif file_type in ["javascript", "typescript"]:
                # Basic JS/TS validation
                if not content.strip():
                    results["issues"].append("File is empty")
                if "TODO" in content:
                    results["warnings"].append("Contains TODO comments")
                if "console.log" in content:
                    results["warnings"].append("Contains debug console.log statements")
            
            # Simulate validation delay
            await asyncio.sleep(0.5)
            
        except Exception as e:
            results["valid"] = False
            results["issues"].append(f"Validation error: {str(e)}")
        
        return results
    
    def clear(self):
        """Clear all displayed content"""
        self.generated_files.clear()
        self.current_file = None
        self.code_display.clear()
        self.validation_display.clear()
        self.file_select.set_options([])


class AutomaticProgrammingDashboard(Static):
    """
    Main dashboard widget combining progress and results
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workflow_id = None
        
    def compose(self) -> ComposeResult:
        """Compose the dashboard"""
        with Horizontal():
            # Left panel - Progress
            with Vertical(classes="progress-panel"):
                self.progress_widget = WorkflowProgressWidget()
                yield self.progress_widget
                
                # Quick actions
                with Container(classes="quick-actions"):
                    yield Label("Quick Actions", classes="section-title")
                    yield Button("⏸️ Pause", id="pause-btn", classes="action-btn")
                    yield Button("⏹️ Stop", id="stop-btn", classes="action-btn") 
                    yield Button("🔄 Retry", id="retry-btn", classes="action-btn")
            
            # Right panel - Results
            with Vertical(classes="results-panel"):
                self.results_viewer = CodeResultsViewer()
                yield self.results_viewer
    
    def start_workflow(self, workflow_id: str, total_steps: int):
        """Start monitoring a workflow"""
        self.workflow_id = workflow_id
        self.progress_widget.update_workflow_info(workflow_id, total_steps)
        self.results_viewer.clear()
    
    def update_progress(self, step_id: str, step_name: str, progress: float, status: str):
        """Update workflow progress"""
        self.progress_widget.update_step_progress(step_id, step_name, progress, status)
    
    def add_generated_file(self, file_path: str, content: str, file_type: str = "python"):
        """Add a generated file to the results viewer"""
        self.results_viewer.add_generated_file(file_path, content, file_type)
    
    def set_error(self, error_message: str):
        """Set error state"""
        self.progress_widget.set_error(error_message)
    
    def set_completed(self, duration: float):
        """Set completed state"""
        self.progress_widget.set_completed(duration)
    
    @on(Button.Pressed, "#pause-btn")
    def on_pause(self, event: Button.Pressed):
        """Handle pause button"""
        # TODO: Implement pause functionality
        pass
    
    @on(Button.Pressed, "#stop-btn") 
    def on_stop(self, event: Button.Pressed):
        """Handle stop button"""
        # TODO: Implement stop functionality
        pass
    
    @on(Button.Pressed, "#retry-btn")
    def on_retry(self, event: Button.Pressed):
        """Handle retry button"""
        # TODO: Implement retry functionality
        pass


class WorkflowTemplateSelector(Static):
    """
    Widget for selecting and configuring workflow templates
    """
    
    def __init__(self, templates: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.templates = templates or {}
        self.selected_template = None
        
    def compose(self) -> ComposeResult:
        """Compose the template selector"""
        with Vertical():
            yield Label("Workflow Templates", classes="widget-title")
            
            # Template selection
            self.template_select = Select([], prompt="Choose a template")
            yield self.template_select
            
            # Template description
            self.description_area = TextArea(
                placeholder="Template description will appear here...",
                read_only=True,
                show_line_numbers=False
            )
            yield self.description_area
            
            # Template parameters (if any)
            with Collapsible(title="Template Parameters", collapsed=True):
                self.parameters_container = Vertical()
                yield self.parameters_container
    
    def set_templates(self, templates: Dict[str, Any]):
        """Set available templates"""
        self.templates = templates
        
        # Update select options
        options = []
        for template_id, template_info in templates.items():
            options.append((template_info["name"], template_id))
        
        self.template_select.set_options(options)
    
    @on(Select.Changed)
    def on_template_selected(self, event: Select.Changed):
        """Handle template selection"""
        if event.value and event.value in self.templates:
            self.selected_template = event.value
            self._display_template_info(event.value)
    
    def _display_template_info(self, template_id: str):
        """Display information about the selected template"""
        if template_id not in self.templates:
            return
        
        template = self.templates[template_id]
        
        # Update description
        description = f"{template['description']}\n\nSteps: {template['steps_count']}"
        self.description_area.text = description
        
        # TODO: Add parameter configuration UI if template has parameters
    
    def get_selected_template(self) -> Optional[str]:
        """Get the currently selected template ID"""
        return self.selected_template
    
    def get_template_parameters(self) -> Dict[str, Any]:
        """Get configured parameters for the selected template"""
        # TODO: Implement parameter collection
        return {}


# CSS for automatic programming widgets
AUTOMATIC_PROGRAMMING_CSS = """
/* Progress Widget Styles */
.progress-panel {
    width: 50%;
    padding: 1;
    border: solid $primary;
}

.results-panel {
    width: 50%;
    padding: 1;
    border: solid $secondary;
}

.overall-progress {
    margin: 1 0;
}

.step-info {
    margin: 1 0;
    text-style: bold;
}

.widget-title {
    text-style: bold;
    color: $accent;
    margin-bottom: 1;
}

.section-title {
    text-style: bold;
    color: $primary;
    margin: 1 0;
}

/* Code Viewer Styles */
.code-display {
    height: 1fr;
    border: solid $success;
    margin: 1 0;
}

.validation-display {
    height: 10;
    border: solid $warning;
    margin: 1 0;
}

.validation-btn {
    margin-left: 1;
}

/* Quick Actions */
.quick-actions {
    height: auto;
    margin: 1 0;
    padding: 1;
    border: solid $accent;
}

.action-btn {
    margin: 0 1 1 0;
    width: 1fr;
}
"""