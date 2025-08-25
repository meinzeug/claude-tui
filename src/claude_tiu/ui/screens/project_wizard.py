"""
Project Wizard - Guided project creation interface.

Provides a step-by-step wizard for creating new projects with templates,
configuration options, and AI-assisted setup.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Button, Input, Select, RadioSet, RadioButton, 
    Checkbox, Label, ProgressBar, TextArea
)
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message

logger = logging.getLogger(__name__)


class ProjectTemplate:
    """Project template definition."""
    
    def __init__(
        self,
        name: str,
        description: str,
        category: str = "general",
        files: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        setup_commands: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.category = category
        self.files = files or {}
        self.dependencies = dependencies or []
        self.setup_commands = setup_commands or []


class ProjectWizard(ModalScreen):
    """
    Project creation wizard with step-by-step guidance.
    
    Steps:
    1. Project type and template selection
    2. Project configuration (name, location, etc.)
    3. Features and integrations
    4. Review and create
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
        Binding("ctrl+n", "next_step", "Next"),
        Binding("ctrl+p", "prev_step", "Previous"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_step = 1
        self.max_steps = 4
        self.project_data: Dict[str, Any] = {
            'template': None,
            'name': '',
            'location': Path.home() / "Projects",
            'description': '',
            'features': [],
            'ai_integration': True,
            'git_init': True,
            'create_venv': True
        }
        self.templates = self._get_available_templates()
    
    def compose(self) -> ComposeResult:
        """Compose the project wizard layout."""
        with Container(classes="wizard-container"):
            # Header
            with Horizontal(classes="wizard-header"):
                yield Label("🚀 New Project Wizard", classes="wizard-title")
                yield ProgressBar(total=self.max_steps, progress=self.current_step, 
                                show_percentage=True, id="progress")
            
            # Content area that changes based on step
            yield Container(id="step_content", classes="step-content")
            
            # Footer with navigation
            with Horizontal(classes="wizard-footer"):
                yield Button("Cancel", id="cancel_btn", variant="default")
                yield Button("Previous", id="prev_btn", variant="default", disabled=True)
                yield Button("Next", id="next_btn", variant="primary")
                yield Button("Create Project", id="create_btn", variant="success", 
                           disabled=True, classes="hidden")
    
    def on_mount(self) -> None:
        """Initialize the wizard."""
        self.update_step_content()
        logger.info("Project wizard opened")
    
    def _get_available_templates(self) -> List[ProjectTemplate]:
        """Get list of available project templates."""
        return [
            ProjectTemplate(
                name="Python CLI Application",
                description="Command-line application with Click framework",
                category="python",
                files={
                    "main.py": "#!/usr/bin/env python3\n\nimport click\n\n@click.command()\ndef main():\n    click.echo('Hello World!')\n\nif __name__ == '__main__':\n    main()",
                    "requirements.txt": "click>=8.0.0\n",
                    "README.md": "# {project_name}\n\nA Python CLI application.\n"
                },
                dependencies=["click"],
                setup_commands=["pip install -r requirements.txt"]
            ),
            ProjectTemplate(
                name="Python Web API",
                description="FastAPI web application with async support",
                category="python",
                files={
                    "main.py": "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\nasync def root():\n    return {'message': 'Hello World'}",
                    "requirements.txt": "fastapi>=0.100.0\nuvicorn>=0.20.0\n",
                    "README.md": "# {project_name}\n\nA FastAPI web application.\n"
                },
                dependencies=["fastapi", "uvicorn"],
                setup_commands=["pip install -r requirements.txt"]
            ),
            ProjectTemplate(
                name="React Application",
                description="Modern React application with TypeScript",
                category="javascript",
                files={
                    "package.json": '{\n  "name": "{project_name}",\n  "version": "1.0.0",\n  "dependencies": {\n    "react": "^18.0.0",\n    "typescript": "^4.9.0"\n  }\n}',
                    "README.md": "# {project_name}\n\nA React application.\n"
                },
                dependencies=["react", "typescript"],
                setup_commands=["npm install"]
            ),
            ProjectTemplate(
                name="Documentation Site",
                description="Documentation site with Markdown support",
                category="documentation",
                files={
                    "index.md": "# {project_name}\n\nProject documentation.\n",
                    "README.md": "# {project_name} Documentation\n\nBuilt with markdown.\n"
                }
            ),
            ProjectTemplate(
                name="Empty Project",
                description="Start with a blank project",
                category="general",
                files={
                    "README.md": "# {project_name}\n\nProject description.\n"
                }
            )
        ]
    
    def update_step_content(self) -> None:
        """Update content based on current step."""
        content_container = self.query_one("#step_content", Container)
        content_container.remove_children()
        
        # Update progress bar
        progress = self.query_one("#progress", ProgressBar)
        progress.progress = self.current_step
        
        if self.current_step == 1:
            content_container.compose_add_child(self._create_template_step())
        elif self.current_step == 2:
            content_container.compose_add_child(self._create_config_step())
        elif self.current_step == 3:
            content_container.compose_add_child(self._create_features_step())
        elif self.current_step == 4:
            content_container.compose_add_child(self._create_review_step())
        
        # Update button states
        self._update_navigation_buttons()
    
    def _create_template_step(self) -> Container:
        """Create template selection step."""
        container = Container(classes="step-container")
        
        with container:
            yield Label("Step 1: Choose a Project Template", classes="step-title")
            yield Label("Select the type of project you want to create:", classes="step-description")
            
            # Group templates by category
            categories = {}
            for template in self.templates:
                if template.category not in categories:
                    categories[template.category] = []
                categories[template.category].append(template)
            
            with ScrollableContainer(classes="template-list"):
                for category, templates in categories.items():
                    yield Label(f"📁 {category.title()}", classes="category-header")
                    
                    for template in templates:
                        with Container(classes="template-item"):
                            yield RadioButton(
                                f"[b]{template.name}[/b]\n{template.description}",
                                value=template.name,
                                name="template_selection"
                            )
        
        return container
    
    def _create_config_step(self) -> Container:
        """Create project configuration step."""
        container = Container(classes="step-container")
        
        with container:
            yield Label("Step 2: Project Configuration", classes="step-title")
            yield Label("Configure your new project:", classes="step-description")
            
            with Vertical(classes="config-form"):
                yield Label("Project Name:")
                yield Input(
                    placeholder="Enter project name...",
                    value=self.project_data.get('name', ''),
                    id="project_name"
                )
                
                yield Label("Project Location:")
                yield Input(
                    placeholder="Enter project path...",
                    value=str(self.project_data.get('location', '')),
                    id="project_location"
                )
                
                yield Label("Description (optional):")
                yield TextArea(
                    placeholder="Describe your project...",
                    text=self.project_data.get('description', ''),
                    id="project_description",
                    max_height=3
                )
        
        return container
    
    def _create_features_step(self) -> Container:
        """Create features selection step."""
        container = Container(classes="step-container")
        
        with container:
            yield Label("Step 3: Features & Integration", classes="step-title")
            yield Label("Choose additional features for your project:", classes="step-description")
            
            with Vertical(classes="features-list"):
                yield Checkbox(
                    "🤖 AI Integration (Claude Code)",
                    value=self.project_data.get('ai_integration', True),
                    id="ai_integration"
                )
                
                yield Checkbox(
                    "📝 Git Repository",
                    value=self.project_data.get('git_init', True),
                    id="git_init"
                )
                
                yield Checkbox(
                    "🐍 Python Virtual Environment",
                    value=self.project_data.get('create_venv', True),
                    id="create_venv"
                )
                
                yield Checkbox(
                    "🧪 Testing Framework",
                    value=False,
                    id="testing_framework"
                )
                
                yield Checkbox(
                    "📚 Documentation Template",
                    value=False,
                    id="docs_template"
                )
                
                yield Checkbox(
                    "🔧 Development Tools (linting, formatting)",
                    value=True,
                    id="dev_tools"
                )
        
        return container
    
    def _create_review_step(self) -> Container:
        """Create project review step."""
        container = Container(classes="step-container")
        
        with container:
            yield Label("Step 4: Review & Create", classes="step-title")
            yield Label("Review your project configuration:", classes="step-description")
            
            with ScrollableContainer(classes="review-content"):
                # Project summary
                template_name = self.project_data.get('template', 'None selected')
                project_name = self.project_data.get('name', 'Unnamed Project')
                location = self.project_data.get('location', 'Not specified')
                
                yield Static(f"""
[b]Project Summary[/b]

📝 Name: {project_name}
📋 Template: {template_name}
📂 Location: {location}
📄 Description: {self.project_data.get('description', 'No description')}

[b]Selected Features[/b]
🤖 AI Integration: {'Yes' if self.project_data.get('ai_integration') else 'No'}
📝 Git Repository: {'Yes' if self.project_data.get('git_init') else 'No'}
🐍 Virtual Environment: {'Yes' if self.project_data.get('create_venv') else 'No'}

[i]Click 'Create Project' to proceed with these settings.[/i]
                """.strip())
        
        return container
    
    def _update_navigation_buttons(self) -> None:
        """Update navigation button states."""
        prev_btn = self.query_one("#prev_btn", Button)
        next_btn = self.query_one("#next_btn", Button)
        create_btn = self.query_one("#create_btn", Button)
        
        # Previous button
        prev_btn.disabled = (self.current_step == 1)
        
        # Next/Create buttons
        if self.current_step == self.max_steps:
            next_btn.add_class("hidden")
            create_btn.remove_class("hidden")
            create_btn.disabled = not self._can_create_project()
        else:
            next_btn.remove_class("hidden")
            create_btn.add_class("hidden")
            next_btn.disabled = not self._can_proceed()
    
    def _can_proceed(self) -> bool:
        """Check if user can proceed to next step."""
        if self.current_step == 1:
            # Need template selected
            return self.project_data.get('template') is not None
        elif self.current_step == 2:
            # Need project name and location
            return (
                self.project_data.get('name', '').strip() != '' and
                self.project_data.get('location') is not None
            )
        elif self.current_step == 3:
            # Features step - can always proceed
            return True
        return True
    
    def _can_create_project(self) -> bool:
        """Check if project can be created."""
        return (
            self.project_data.get('template') is not None and
            self.project_data.get('name', '').strip() != '' and
            self.project_data.get('location') is not None
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle wizard button presses."""
        if event.button.id == "cancel_btn":
            self.dismiss()
        elif event.button.id == "prev_btn":
            self.action_prev_step()
        elif event.button.id == "next_btn":
            self.action_next_step()
        elif event.button.id == "create_btn":
            self.action_create_project()
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle template selection."""
        if event.pressed and event.pressed.name == "template_selection":
            self.project_data['template'] = event.pressed.value
            self._update_navigation_buttons()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input field changes."""
        if event.input.id == "project_name":
            self.project_data['name'] = event.value
        elif event.input.id == "project_location":
            self.project_data['location'] = Path(event.value) if event.value else None
        elif event.input.id == "project_description":
            self.project_data['description'] = event.value
        
        self._update_navigation_buttons()
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        checkbox_id = event.checkbox.id
        if checkbox_id:
            self.project_data[checkbox_id] = event.value
    
    def action_next_step(self) -> None:
        """Go to next step."""
        if self.current_step < self.max_steps and self._can_proceed():
            self.current_step += 1
            self.update_step_content()
    
    def action_prev_step(self) -> None:
        """Go to previous step."""
        if self.current_step > 1:
            self.current_step -= 1
            self.update_step_content()
    
    def action_create_project(self) -> None:
        """Create the project."""
        if self._can_create_project():
            try:
                # TODO: Implement actual project creation
                project_path = self.project_data['location'] / self.project_data['name']
                
                # For now, just show success message
                self.app.notify(
                    f"Project '{self.project_data['name']}' would be created at {project_path}",
                    severity="success"
                )
                
                logger.info(f"Project creation requested: {self.project_data}")
                self.dismiss()
                
            except Exception as e:
                logger.error(f"Project creation failed: {e}")
                self.app.notify(f"Failed to create project: {e}", severity="error")
    
    def action_dismiss(self) -> None:
        """Dismiss the wizard."""
        self.dismiss()