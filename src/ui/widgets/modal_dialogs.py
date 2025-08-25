#!/usr/bin/env python3
"""
Modal Dialog Widgets - Configuration dialogs, command templates,
and other modal interactions for the TUI application.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from textual import on
from textual.containers import Vertical, Horizontal, Container, Grid
from textual.widgets import (
    Static, Label, Button, Input, TextArea, Select, Checkbox, 
    RadioSet, RadioButton, ListView, ListItem
)
from textual.screen import ModalScreen
from textual.message import Message
from textual.validation import Function, Number
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax


@dataclass
class ConfigOption:
    """Configuration option definition"""
    key: str
    name: str
    description: str
    value_type: type
    default_value: Any
    current_value: Any = None
    options: Optional[List[Any]] = None  # For select/radio options
    validation_func: Optional[Callable] = None
    category: str = "general"
    
    def __post_init__(self):
        if self.current_value is None:
            self.current_value = self.default_value


class ConfigurationModal(ModalScreen):
    """Modal screen for application configuration"""
    
    def __init__(self, config_options: List[ConfigOption]) -> None:
        super().__init__()
        self.config_options = config_options
        self.input_widgets: Dict[str, Any] = {}
        self.categories = list(set(opt.category for opt in config_options))
        
    def compose(self):
        """Compose configuration modal"""
        with Container(id="config-modal", classes="modal-container"):
            yield Label("⚙️ Application Settings", classes="modal-header")
            
            # Configuration tabs by category
            with Vertical(classes="config-content"):
                for category in sorted(self.categories):
                    category_options = [opt for opt in self.config_options if opt.category == category]
                    if category_options:
                        yield Label(f"📁 {category.title()}", classes="category-header")
                        
                        with Container(classes="category-content"):
                            for option in category_options:
                                yield self._create_option_widget(option)
            
            # Action buttons
            with Horizontal(classes="modal-actions"):
                yield Button("💾 Save", id="save-config", variant="primary")
                yield Button("🔄 Reset", id="reset-config")
                yield Button("📤 Export", id="export-config")
                yield Button("📥 Import", id="import-config")
                yield Button("❌ Cancel", id="cancel-config", variant="error")
    
    def _create_option_widget(self, option: ConfigOption) -> Container:
        """Create widget for a configuration option"""
        with Container(classes="config-option"):
            # Option label and description
            yield Label(option.name, classes="option-name")
            if option.description:
                yield Label(option.description, classes="option-description")
            
            # Input widget based on type
            if option.value_type == bool:
                widget = Checkbox(option.current_value, id=f"config-{option.key}")
            elif option.value_type == int:
                widget = Input(
                    str(option.current_value),
                    placeholder="Enter number...",
                    validators=[Number()],
                    id=f"config-{option.key}"
                )
            elif option.value_type == float:
                widget = Input(
                    str(option.current_value),
                    placeholder="Enter decimal number...",
                    validators=[Number()],
                    id=f"config-{option.key}"
                )
            elif option.options:
                # Select dropdown for predefined options
                widget = Select(
                    [(str(opt), opt) for opt in option.options],
                    value=option.current_value,
                    id=f"config-{option.key}"
                )
            else:
                # Text input for strings
                widget = Input(
                    str(option.current_value),
                    placeholder="Enter value...",
                    id=f"config-{option.key}"
                )
            
            self.input_widgets[option.key] = widget
            yield widget
        
        return Container()
    
    def _validate_inputs(self) -> Dict[str, Any]:
        """Validate all inputs and return values"""
        values = {}
        errors = []
        
        for option in self.config_options:
            widget = self.input_widgets.get(option.key)
            if widget:
                try:
                    if isinstance(widget, Checkbox):
                        values[option.key] = widget.value
                    elif isinstance(widget, Select):
                        values[option.key] = widget.value
                    elif isinstance(widget, Input):
                        raw_value = widget.value
                        if option.value_type == int:
                            values[option.key] = int(raw_value)
                        elif option.value_type == float:
                            values[option.key] = float(raw_value)
                        else:
                            values[option.key] = raw_value
                    
                    # Custom validation
                    if option.validation_func:
                        if not option.validation_func(values[option.key]):
                            errors.append(f"Invalid value for {option.name}")
                
                except (ValueError, TypeError) as e:
                    errors.append(f"Invalid {option.value_type.__name__} for {option.name}")
        
        if errors:
            # Show errors to user
            error_text = "\n".join(errors)
            self.notify(f"Validation errors:\n{error_text}", severity="error")
            return {}
        
        return values
    
    @on(Button.Pressed, "#save-config")
    def save_configuration(self) -> None:
        """Save configuration changes"""
        values = self._validate_inputs()
        if values:
            self.post_message(SaveConfigMessage(values))
            self.dismiss()
    
    @on(Button.Pressed, "#reset-config")
    def reset_configuration(self) -> None:
        """Reset all values to defaults"""
        for option in self.config_options:
            widget = self.input_widgets.get(option.key)
            if widget:
                if isinstance(widget, Checkbox):
                    widget.value = option.default_value
                elif isinstance(widget, Select):
                    widget.value = option.default_value
                elif isinstance(widget, Input):
                    widget.value = str(option.default_value)
    
    @on(Button.Pressed, "#export-config")
    def export_configuration(self) -> None:
        """Export current configuration"""
        values = self._validate_inputs()
        if values:
            self.post_message(ExportConfigMessage(values))
    
    @on(Button.Pressed, "#import-config")
    def import_configuration(self) -> None:
        """Import configuration from file"""
        self.post_message(ImportConfigMessage())
    
    @on(Button.Pressed, "#cancel-config")
    def cancel_configuration(self) -> None:
        """Cancel configuration changes"""
        self.dismiss()


class CommandTemplatesModal(ModalScreen):
    """Modal screen showing AI command templates"""
    
    def __init__(self) -> None:
        super().__init__()
        self.templates = self._load_command_templates()
        self.selected_template: Optional[Dict[str, str]] = None
        
    def compose(self):
        """Compose command templates modal"""
        with Container(id="templates-modal", classes="modal-container"):
            yield Label("📝 AI Command Templates", classes="modal-header")
            
            with Horizontal(classes="templates-content"):
                # Template categories and list
                with Vertical(classes="templates-list"):
                    yield Label("Available Templates", classes="section-header")
                    
                    with ListView(id="template-list"):
                        for category, templates in self.templates.items():
                            # Category header
                            yield ListItem(Label(f"📁 {category}", classes="category-item"))
                            
                            # Templates in category
                            for template in templates:
                                template_item = Label(f"  📄 {template['name']}")
                                yield ListItem(template_item, id=f"template-{template['id']}")
                
                # Template preview and details
                with Vertical(classes="template-preview"):
                    yield Label("Template Preview", classes="section-header")
                    yield Static("Select a template to preview", id="template-preview-content")
                    
                    # Template customization
                    yield Label("Customization", classes="section-header")
                    yield Input(placeholder="Custom parameters...", id="custom-params")
                    yield TextArea("Additional context or modifications...", id="custom-context")
            
            # Action buttons
            with Horizontal(classes="modal-actions"):
                yield Button("🚀 Use Template", id="use-template", variant="primary")
                yield Button("💾 Save Custom", id="save-custom")
                yield Button("📤 Export Templates", id="export-templates")
                yield Button("❌ Close", id="close-templates", variant="error")
    
    def _load_command_templates(self) -> Dict[str, List[Dict[str, str]]]:
        """Load command templates organized by category"""
        return {
            "Development": [
                {
                    "id": "create-feature",
                    "name": "Create New Feature",
                    "description": "Create a complete feature with tests and documentation",
                    "template": "Create a new feature for {feature_name} that {functionality}. Include:\n- Implementation in {language}\n- Unit tests with >80% coverage\n- API documentation\n- Error handling\n- Logging",
                    "parameters": ["feature_name", "functionality", "language"]
                },
                {
                    "id": "fix-bug",
                    "name": "Fix Bug",
                    "description": "Analyze and fix a reported bug",
                    "template": "Fix the bug in {file_path} where {bug_description}. Steps:\n1. Analyze the root cause\n2. Implement the fix\n3. Add regression tests\n4. Update documentation if needed",
                    "parameters": ["file_path", "bug_description"]
                },
                {
                    "id": "refactor-code",
                    "name": "Refactor Code",
                    "description": "Refactor code for better maintainability",
                    "template": "Refactor {component} to improve {improvement_goal}. Requirements:\n- Maintain existing functionality\n- Improve code readability\n- Follow {coding_standards}\n- Update tests accordingly",
                    "parameters": ["component", "improvement_goal", "coding_standards"]
                }
            ],
            "Testing": [
                {
                    "id": "write-tests",
                    "name": "Write Test Suite",
                    "description": "Create comprehensive tests for a component",
                    "template": "Write a comprehensive test suite for {component} covering:\n- Happy path scenarios\n- Edge cases and error conditions\n- Performance tests\n- Integration tests\nTarget coverage: {coverage_target}%",
                    "parameters": ["component", "coverage_target"]
                },
                {
                    "id": "test-analysis",
                    "name": "Analyze Test Coverage",
                    "description": "Analyze and improve test coverage",
                    "template": "Analyze test coverage for {module} and:\n1. Identify uncovered code paths\n2. Write missing tests\n3. Improve existing test quality\n4. Generate coverage report",
                    "parameters": ["module"]
                }
            ],
            "Documentation": [
                {
                    "id": "api-docs",
                    "name": "Generate API Documentation",
                    "description": "Create comprehensive API documentation",
                    "template": "Generate API documentation for {api_name} including:\n- Endpoint descriptions\n- Request/response schemas\n- Authentication details\n- Example usage\n- Error codes and handling",
                    "parameters": ["api_name"]
                },
                {
                    "id": "user-guide",
                    "name": "User Guide",
                    "description": "Create user-friendly documentation",
                    "template": "Create a user guide for {feature} that includes:\n- Getting started tutorial\n- Common use cases\n- Troubleshooting\n- FAQs\n- Best practices",
                    "parameters": ["feature"]
                }
            ],
            "DevOps": [
                {
                    "id": "deploy-setup",
                    "name": "Deployment Setup",
                    "description": "Set up deployment pipeline",
                    "template": "Set up deployment for {application} to {environment}:\n- CI/CD pipeline configuration\n- Environment-specific configs\n- Health checks and monitoring\n- Rollback procedures\n- Documentation",
                    "parameters": ["application", "environment"]
                }
            ]
        }
    
    @on(ListView.Highlighted)
    def template_selected(self, event: ListView.Highlighted) -> None:
        """Handle template selection"""
        if event.item and event.item.id and event.item.id.startswith("template-"):
            template_id = event.item.id.replace("template-", "")
            
            # Find template by ID
            for category_templates in self.templates.values():
                for template in category_templates:
                    if template["id"] == template_id:
                        self.selected_template = template
                        self._show_template_preview(template)
                        break
    
    def _show_template_preview(self, template: Dict[str, str]) -> None:
        """Show template preview in the preview pane"""
        preview_content = f"""Name: {template['name']}

Description: {template['description']}

Template:
{template['template']}

Parameters: {', '.join(template.get('parameters', []))}
"""
        
        preview_widget = self.query_one("#template-preview-content", Static)
        preview_widget.update(preview_content)
    
    @on(Button.Pressed, "#use-template")
    def use_template(self) -> None:
        """Use the selected template"""
        if self.selected_template:
            # Get custom parameters
            custom_params = self.query_one("#custom-params", Input).value
            custom_context = self.query_one("#custom-context", TextArea).value
            
            # Build final command
            command = self.selected_template["template"]
            
            # Add custom context if provided
            if custom_context.strip():
                command += f"\n\nAdditional Context:\n{custom_context}"
            
            self.post_message(UseTemplateMessage(
                template=self.selected_template,
                command=command,
                custom_params=custom_params,
                custom_context=custom_context
            ))
            self.dismiss()
        else:
            self.notify("Please select a template first", severity="warning")
    
    @on(Button.Pressed, "#save-custom")
    def save_custom_template(self) -> None:
        """Save custom template"""
        # This would open another modal for creating custom templates
        self.post_message(SaveCustomTemplateMessage())
    
    @on(Button.Pressed, "#export-templates")
    def export_templates(self) -> None:
        """Export all templates"""
        self.post_message(ExportTemplatesMessage(self.templates))
    
    @on(Button.Pressed, "#close-templates")
    def close_modal(self) -> None:
        """Close the modal"""
        self.dismiss()


class ConfirmationModal(ModalScreen):
    """Simple confirmation dialog"""
    
    def __init__(
        self,
        title: str,
        message: str,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
        variant: str = "primary"
    ) -> None:
        super().__init__()
        self.title = title
        self.message = message
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text
        self.variant = variant
        
    def compose(self):
        """Compose confirmation modal"""
        with Container(id="confirm-modal", classes="modal-container small-modal"):
            yield Label(self.title, classes="modal-header")
            yield Static(self.message, classes="confirm-message")
            
            with Horizontal(classes="modal-actions"):
                yield Button(self.confirm_text, id="confirm-action", variant=self.variant)
                yield Button(self.cancel_text, id="cancel-action", variant="default")
    
    @on(Button.Pressed, "#confirm-action")
    def confirm_action(self) -> None:
        """Handle confirmation"""
        self.post_message(ConfirmationMessage(True))
        self.dismiss()
    
    @on(Button.Pressed, "#cancel-action")
    def cancel_action(self) -> None:
        """Handle cancellation"""
        self.post_message(ConfirmationMessage(False))
        self.dismiss()


class TaskCreationModal(ModalScreen):
    """Modal for creating new tasks"""
    
    def __init__(self, project_id: str) -> None:
        super().__init__()
        self.project_id = project_id
        
    def compose(self):
        """Compose task creation modal"""
        with Container(id="task-modal", classes="modal-container"):
            yield Label("➕ Create New Task", classes="modal-header")
            
            with Vertical(classes="form-content"):
                # Basic task information
                yield Label("Task Name", classes="form-label")
                yield Input(placeholder="Enter task name...", id="task-name")
                
                yield Label("Description", classes="form-label")
                yield TextArea(placeholder="Describe the task...", id="task-description")
                
                # Task properties
                with Horizontal():
                    with Vertical():
                        yield Label("Priority", classes="form-label")
                        yield Select([
                            ("Low", "low"),
                            ("Medium", "medium"),
                            ("High", "high"),
                            ("Critical", "critical")
                        ], value="medium", id="task-priority")
                    
                    with Vertical():
                        yield Label("Estimated Hours", classes="form-label")
                        yield Input(placeholder="0.0", validators=[Number()], id="task-hours")
                
                # Dependencies
                yield Label("Dependencies (comma-separated task IDs)", classes="form-label")
                yield Input(placeholder="task-1, task-2", id="task-dependencies")
                
                # Assignment
                yield Label("Assign to Agent", classes="form-label")
                yield Select([
                    ("Auto-assign", ""),
                    ("Research Agent", "researcher"),
                    ("Code Agent", "coder"),
                    ("Test Agent", "tester"),
                    ("Review Agent", "reviewer")
                ], id="task-agent")
            
            # Action buttons
            with Horizontal(classes="modal-actions"):
                yield Button("✅ Create Task", id="create-task", variant="primary")
                yield Button("❌ Cancel", id="cancel-task", variant="error")
    
    @on(Button.Pressed, "#create-task")
    def create_task(self) -> None:
        """Create the task"""
        # Collect form data
        name = self.query_one("#task-name", Input).value
        description = self.query_one("#task-description", TextArea).value
        priority = self.query_one("#task-priority", Select).value
        hours = self.query_one("#task-hours", Input).value
        dependencies = self.query_one("#task-dependencies", Input).value
        agent = self.query_one("#task-agent", Select).value
        
        # Validate
        if not name.strip():
            self.notify("Task name is required", severity="error")
            return
        
        try:
            estimated_hours = float(hours) if hours else 0.0
        except ValueError:
            self.notify("Invalid number for estimated hours", severity="error")
            return
        
        # Parse dependencies
        deps = [dep.strip() for dep in dependencies.split(",") if dep.strip()]
        
        # Create task data
        task_data = {
            "name": name,
            "description": description,
            "priority": priority,
            "estimated_hours": estimated_hours,
            "dependencies": deps,
            "assigned_agent": agent if agent else None,
            "project_id": self.project_id
        }
        
        self.post_message(CreateTaskMessage(task_data))
        self.dismiss()
    
    @on(Button.Pressed, "#cancel-task")
    def cancel_task(self) -> None:
        """Cancel task creation"""
        self.dismiss()


# Message classes for modal dialogs
class SaveConfigMessage(Message):
    """Message sent when configuration is saved"""
    def __init__(self, values: Dict[str, Any]) -> None:
        super().__init__()
        self.values = values


class ExportConfigMessage(Message):
    """Message sent to export configuration"""
    def __init__(self, values: Dict[str, Any]) -> None:
        super().__init__()
        self.values = values


class ImportConfigMessage(Message):
    """Message sent to import configuration"""
    pass


class UseTemplateMessage(Message):
    """Message sent when using a command template"""
    def __init__(
        self, 
        template: Dict[str, str], 
        command: str, 
        custom_params: str = "", 
        custom_context: str = ""
    ) -> None:
        super().__init__()
        self.template = template
        self.command = command
        self.custom_params = custom_params
        self.custom_context = custom_context


class SaveCustomTemplateMessage(Message):
    """Message sent to save custom template"""
    pass


class ExportTemplatesMessage(Message):
    """Message sent to export templates"""
    def __init__(self, templates: Dict[str, List[Dict[str, str]]]) -> None:
        super().__init__()
        self.templates = templates


class ConfirmationMessage(Message):
    """Message sent with confirmation result"""
    def __init__(self, confirmed: bool) -> None:
        super().__init__()
        self.confirmed = confirmed


class CreateTaskMessage(Message):
    """Message sent to create new task"""
    def __init__(self, task_data: Dict[str, Any]) -> None:
        super().__init__()
        self.task_data = task_data