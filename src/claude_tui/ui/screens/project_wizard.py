"""
Project Wizard - Guided project creation interface.

Provides a step-by-step wizard for creating new projects with templates,
configuration options, and AI-assisted setup.
"""

import json
import logging
from datetime import datetime
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
                # Implement actual project creation
                project_path = Path(self.project_data['location']) / self.project_data['name']
                
                # Create project directory structure
                project_path.mkdir(parents=True, exist_ok=True)
                
                # Create basic project files based on template
                template = self.project_data.get('template', 'basic')
                
                if template == 'python':
                    self._create_python_project(project_path)
                elif template == 'web':
                    self._create_web_project(project_path)
                elif template == 'api':
                    self._create_api_project(project_path)
                else:
                    self._create_basic_project(project_path)
                
                # Create project configuration
                config_data = {
                    'name': self.project_data['name'],
                    'description': self.project_data.get('description', ''),
                    'template': template,
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0.0'
                }
                
                config_file = project_path / '.project.json'
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Notify user of successful creation
                self.app.notify(
                    f"Project '{self.project_data['name']}' created successfully at {project_path}",
                    severity="success"
                )
                
                logger.info(f"Project creation requested: {self.project_data}")
                self.dismiss()
                
            except Exception as e:
                logger.error(f"Project creation failed: {e}")
                self.app.notify(f"Failed to create project: {e}", severity="error")
    
    def _create_basic_project(self, project_path: Path) -> None:
        """Create a basic project structure."""
        (project_path / "src").mkdir(exist_ok=True)
        (project_path / "tests").mkdir(exist_ok=True)
        (project_path / "docs").mkdir(exist_ok=True)
        
        # Create basic README
        readme_content = f"""# {self.project_data['name']}

{self.project_data.get('description', 'A basic project created with Claude TUI')}

## Getting Started

TODO: Add setup instructions
"""
        (project_path / "README.md").write_text(readme_content)
        
        # Create basic .gitignore
        gitignore_content = """__pycache__/
*.py[cod]
*$py.class
.DS_Store
.vscode/
.idea/
"""
        (project_path / ".gitignore").write_text(gitignore_content)
    
    def _create_python_project(self, project_path: Path) -> None:
        """Create a Python project structure."""
        self._create_basic_project(project_path)
        
        # Create Python-specific structure
        package_name = self.project_data['name'].lower().replace('-', '_')
        src_dir = project_path / "src" / package_name
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        (src_dir / "__init__.py").write_text(f'"""{ self.project_data["name"] } package."""\n\n__version__ = "1.0.0"\n')
        
        # Create main.py
        main_content = '''"""Main module."""

def main():
    """Main entry point."""
    print("Hello from {}")

if __name__ == "__main__":
    main()
'''.format(self.project_data['name'])
        (src_dir / "main.py").write_text(main_content)
        
        # Create requirements.txt
        (project_path / "requirements.txt").write_text("# Add your dependencies here\n")
        
        # Create setup.py
        setup_content = f'''from setuptools import setup, find_packages

setup(
    name="{self.project_data['name']}",
    version="1.0.0",
    description="{self.project_data.get('description', '')}",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    python_requires=">=3.8",
    install_requires=[
        # Add dependencies here
    ],
    entry_points={{
        "console_scripts": [
            "{package_name}={package_name}.main:main",
        ],
    }},
)
'''
        (project_path / "setup.py").write_text(setup_content)
    
    def _create_web_project(self, project_path: Path) -> None:
        """Create a web project structure."""
        self._create_basic_project(project_path)
        
        # Create web-specific directories
        (project_path / "static").mkdir(exist_ok=True)
        (project_path / "templates").mkdir(exist_ok=True)
        (project_path / "static" / "css").mkdir(exist_ok=True)
        (project_path / "static" / "js").mkdir(exist_ok=True)
        
        # Create basic HTML template
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.project_data['name']}</title>
    <link rel="stylesheet" href="static/css/style.css">
</head>
<body>
    <h1>Welcome to {self.project_data['name']}</h1>
    <p>{self.project_data.get('description', 'A web project created with Claude TUI')}</p>
    <script src="static/js/app.js"></script>
</body>
</html>'''
        (project_path / "templates" / "index.html").write_text(html_content)
        
        # Create basic CSS
        css_content = """body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

h1 {
    color: #333;
    text-align: center;
}
"""
        (project_path / "static" / "css" / "style.css").write_text(css_content)
        
        # Create basic JS
        js_content = """document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded');
});
"""
        (project_path / "static" / "js" / "app.js").write_text(js_content)
    
    def _create_api_project(self, project_path: Path) -> None:
        """Create an API project structure."""
        self._create_python_project(project_path)
        
        # Add API-specific files
        package_name = self.project_data['name'].lower().replace('-', '_')
        src_dir = project_path / "src" / package_name
        
        # Create API main file
        api_content = '''"""API main module."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="{}",
    description="{}",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {{"message": "Welcome to {} API"}}

@app.get("/health")
async def health():
    """Health check endpoint.""" 
    return {{"status": "healthy"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''.format(self.project_data['name'], self.project_data.get('description', ''), self.project_data['name'])
        
        (src_dir / "api.py").write_text(api_content)
        
        # Update requirements with FastAPI
        requirements_content = """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
"""
        (project_path / "requirements.txt").write_text(requirements_content)
    
    def action_dismiss(self) -> None:
        """Dismiss the wizard."""
        self.dismiss()