#!/usr/bin/env python3
"""
Project Wizard Screen - Interactive project creation wizard
with template selection, configuration, and guided setup.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from textual import on, work
from textual.containers import Vertical, Horizontal, Container, Center, Middle
from textual.widgets import (
    Static, Label, Button, Input, Select, Checkbox, 
    RadioSet, RadioButton, TextArea, ProgressBar, 
    TabbedContent, TabPane, Tree, DirectoryTree
)
from textual.screen import ModalScreen
from textual.message import Message
from textual.reactive import reactive
from textual.validation import Function, ValidationResult, Validator
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree as RichTree
from rich.console import Console


class ProjectType(Enum):
    """Supported project types"""
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    NODE_JS = "nodejs"
    PYTHON = "python"
    DJANGO = "django"
    FLASK = "flask"
    JAVA = "java"
    SPRING_BOOT = "spring_boot"
    GO = "go"
    RUST = "rust"
    NEXT_JS = "nextjs"
    NUXT = "nuxt"
    FASTAPI = "fastapi"
    EXPRESS = "express"
    CUSTOM = "custom"


class BuildTool(Enum):
    """Build tools and bundlers"""
    WEBPACK = "webpack"
    VITE = "vite"
    PARCEL = "parcel"
    ROLLUP = "rollup"
    ESBUILD = "esbuild"
    MAVEN = "maven"
    GRADLE = "gradle"
    CARGO = "cargo"
    GO_MOD = "go_mod"
    POETRY = "poetry"
    PIP = "pip"
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"


class TestFramework(Enum):
    """Testing frameworks"""
    JEST = "jest"
    VITEST = "vitest"
    CYPRESS = "cypress"
    PLAYWRIGHT = "playwright"
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JUNIT = "junit"
    TESTNG = "testng"
    GO_TEST = "go_test"
    RUST_TEST = "rust_test"


@dataclass
class ProjectTemplate:
    """Project template definition"""
    id: str
    name: str
    description: str
    project_type: ProjectType
    technologies: List[str]
    build_tools: List[BuildTool]
    test_frameworks: List[TestFramework]
    features: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    directories: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def get_icon(self) -> str:
        """Get icon for project type"""
        icons = {
            ProjectType.REACT: "⚛️",
            ProjectType.VUE: "🟢",
            ProjectType.ANGULAR: "🔴",
            ProjectType.NODE_JS: "🟢",
            ProjectType.PYTHON: "🐍",
            ProjectType.DJANGO: "🐍",
            ProjectType.FLASK: "🐍",
            ProjectType.JAVA: "☕",
            ProjectType.SPRING_BOOT: "☕",
            ProjectType.GO: "🐹",
            ProjectType.RUST: "🦀",
            ProjectType.NEXT_JS: "▲",
            ProjectType.NUXT: "🟢",
            ProjectType.FASTAPI: "🐍",
            ProjectType.EXPRESS: "🟢",
            ProjectType.CUSTOM: "🔨"
        }
        return icons.get(self.project_type, "📁")


@dataclass
class ProjectConfig:
    """Project configuration from wizard"""
    name: str
    description: str
    path: Path
    template: ProjectTemplate
    features: List[str] = field(default_factory=list)
    build_tool: Optional[BuildTool] = None
    test_framework: Optional[TestFramework] = None
    package_manager: Optional[str] = None
    git_init: bool = True
    ai_prompts: List[str] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)


class ProjectNameValidator(Validator):
    """Validator for project names"""
    
    def validate(self, value: str) -> ValidationResult:
        """Validate project name"""
        if not value:
            return self.failure("Project name cannot be empty")
        
        if len(value) < 2:
            return self.failure("Project name must be at least 2 characters")
        
        if len(value) > 50:
            return self.failure("Project name cannot exceed 50 characters")
        
        # Check for invalid characters
        invalid_chars = set(value) & set('\\/:*?"<>|')
        if invalid_chars:
            return self.failure(f"Invalid characters: {', '.join(invalid_chars)}")
        
        return self.success()


class PathValidator(Validator):
    """Validator for project paths"""
    
    def validate(self, value: str) -> ValidationResult:
        """Validate project path"""
        if not value:
            return self.failure("Path cannot be empty")
        
        try:
            path = Path(value)
            
            # Check if path exists and is writable
            if path.exists() and not path.is_dir():
                return self.failure("Path exists but is not a directory")
            
            # Check if directory is empty if it exists
            if path.exists() and any(path.iterdir()):
                return self.failure("Directory is not empty")
            
            # Check if we can write to parent directory
            parent = path.parent
            if not parent.exists():
                return self.failure("Parent directory does not exist")
            
            if not parent.is_dir():
                return self.failure("Parent path is not a directory")
            
            return self.success()
            
        except Exception as e:
            return self.failure(f"Invalid path: {e}")


class TemplateSelector(Container):
    """Template selection widget"""
    
    def __init__(self, templates: List[ProjectTemplate]) -> None:
        super().__init__()
        self.templates = templates
        self.selected_template: Optional[ProjectTemplate] = None
        
    def compose(self):
        """Compose template selector"""
        yield Label("Choose a project template:", classes="section-header")
        
        with RadioSet(id="template-selector"):
            for template in self.templates:
                radio_button = RadioButton(
                    f"{template.get_icon()} {template.name}",
                    value=template.id
                )
                yield radio_button
        
        # Template details panel
        yield Static(
            "Select a template to see details",
            id="template-details",
            classes="template-details"
        )
    
    @on(RadioSet.Changed)
    def template_selected(self, event: RadioSet.Changed) -> None:
        """Handle template selection"""
        if event.pressed:
            template_id = event.pressed.value
            self.selected_template = next(
                (t for t in self.templates if t.id == template_id),
                None
            )
            
            if self.selected_template:
                self._update_template_details()
                self.post_message(TemplateSelectedMessage(self.selected_template))
    
    def _update_template_details(self) -> None:
        """Update template details display"""
        if not self.selected_template:
            return
        
        template = self.selected_template
        
        # Create details panel
        details = Text()
        details.append(f"{template.get_icon()} {template.name}\n", style="bold blue")
        details.append(f"{template.description}\n\n", style="dim")
        
        details.append("Technologies: ", style="bold")
        details.append(f"{', '.join(template.technologies)}\n")
        
        if template.features:
            details.append("Features: ", style="bold")
            details.append(f"{', '.join(template.features)}\n")
        
        details.append("Build Tools: ", style="bold")
        details.append(f"{', '.join([t.value for t in template.build_tools])}\n")
        
        details.append("Test Frameworks: ", style="bold")
        details.append(f"{', '.join([f.value for f in template.test_frameworks])}\n")
        
        # Update details panel
        details_panel = self.query_one("#template-details", Static)
        details_panel.update(Panel(details, title="Template Details"))


class ProjectConfigForm(Container):
    """Project configuration form"""
    
    def __init__(self) -> None:
        super().__init__()
        self.config_data: Dict[str, Any] = {}
        
    def compose(self):
        """Compose configuration form"""
        yield Label("Project Configuration:", classes="section-header")
        
        # Basic project info
        with Vertical(classes="form-section"):
            yield Label("Project Name:")
            yield Input(
                placeholder="Enter project name",
                id="project-name",
                validators=[ProjectNameValidator()]
            )
            
            yield Label("Description:")
            yield TextArea(
                placeholder="Enter project description (optional)",
                id="project-description"
            )
            
            yield Label("Project Path:")
            with Horizontal():
                yield Input(
                    placeholder="/path/to/project",
                    id="project-path",
                    validators=[PathValidator()]
                )
                yield Button("📂 Browse", id="browse-path")
        
        # Advanced options
        with Vertical(classes="form-section"):
            yield Label("Options:", classes="subsection-header")
            
            yield Checkbox("Initialize Git repository", id="git-init", value=True)
            yield Checkbox("Generate README.md", id="generate-readme", value=True)
            yield Checkbox("Setup CI/CD pipeline", id="setup-cicd", value=False)
            yield Checkbox("Add Docker configuration", id="add-docker", value=False)
            yield Checkbox("Enable TypeScript (if applicable)", id="enable-typescript", value=True)
    
    def get_config_data(self) -> Dict[str, Any]:
        """Get current form data"""
        try:
            return {
                'name': self.query_one("#project-name", Input).value,
                'description': self.query_one("#project-description", TextArea).text,
                'path': self.query_one("#project-path", Input).value,
                'git_init': self.query_one("#git-init", Checkbox).value,
                'generate_readme': self.query_one("#generate-readme", Checkbox).value,
                'setup_cicd': self.query_one("#setup-cicd", Checkbox).value,
                'add_docker': self.query_one("#add-docker", Checkbox).value,
                'enable_typescript': self.query_one("#enable-typescript", Checkbox).value,
            }
        except Exception:
            return {}
    
    def validate_form(self) -> List[str]:
        """Validate form data"""
        errors = []
        
        try:
            # Validate required fields
            name = self.query_one("#project-name", Input)
            if not name.value or not name.is_valid:
                errors.append("Invalid project name")
            
            path = self.query_one("#project-path", Input)
            if not path.value or not path.is_valid:
                errors.append("Invalid project path")
        
        except Exception as e:
            errors.append(f"Form validation error: {e}")
        
        return errors
    
    @on(Button.Pressed, "#browse-path")
    def browse_for_path(self) -> None:
        """Open path browser dialog"""
        self.post_message(BrowseForPathMessage())


class FeatureSelector(Container):
    """Feature selection widget for template customization"""
    
    def __init__(self, template: Optional[ProjectTemplate] = None) -> None:
        super().__init__()
        self.template = template
        self.available_features: List[str] = []
        self.selected_features: List[str] = []
        
    def compose(self):
        """Compose feature selector"""
        yield Label("Select Additional Features:", classes="section-header")
        
        if not self.template:
            yield Static("Select a template first", classes="placeholder")
        else:
            self._compose_features()
    
    def _compose_features(self) -> None:
        """Compose features based on template"""
        # Common features based on project type
        common_features = {
            ProjectType.REACT: [
                "Redux/State Management", "React Router", "Material-UI", 
                "Styled Components", "Storybook", "PWA Support"
            ],
            ProjectType.VUE: [
                "Vuex/Pinia", "Vue Router", "Vuetify", "Composition API", 
                "Vue CLI", "Nuxt.js Integration"
            ],
            ProjectType.PYTHON: [
                "FastAPI", "Flask", "Django", "SQLAlchemy", 
                "Pydantic", "Celery", "Redis", "PostgreSQL"
            ],
            ProjectType.NODE_JS: [
                "Express.js", "GraphQL", "MongoDB", "JWT Authentication",
                "Socket.io", "Passport.js", "Swagger/OpenAPI"
            ]
        }
        
        # Get features for current template
        self.available_features = common_features.get(
            self.template.project_type, 
            ["Authentication", "Database", "API Integration", "Testing"]
        )
        
        # Add feature checkboxes
        with Vertical(id="feature-list"):
            for feature in self.available_features:
                yield Checkbox(feature, id=f"feature-{feature.lower().replace(' ', '-')}")
    
    def update_template(self, template: ProjectTemplate) -> None:
        """Update features for new template"""
        self.template = template
        
        # Clear existing features
        feature_list = self.query_one("#feature-list", Vertical)
        feature_list.remove_children()
        
        # Add new features
        self._compose_features()
    
    def get_selected_features(self) -> List[str]:
        """Get currently selected features"""
        selected = []
        
        for feature in self.available_features:
            checkbox_id = f"feature-{feature.lower().replace(' ', '-')}"
            try:
                checkbox = self.query_one(f"#{checkbox_id}", Checkbox)
                if checkbox.value:
                    selected.append(feature)
            except:
                pass
        
        return selected


class ProjectPreview(Container):
    """Project preview widget showing final configuration"""
    
    def __init__(self) -> None:
        super().__init__()
        self.project_config: Optional[ProjectConfig] = None
        
    def compose(self):
        """Compose project preview"""
        yield Label("Project Preview:", classes="section-header")
        
        yield Static(
            "Complete configuration to see preview",
            id="preview-content",
            classes="preview-panel"
        )
        
        # File structure preview
        yield Label("File Structure:", classes="subsection-header")
        yield Static(
            "File structure will appear here",
            id="file-structure",
            classes="file-structure"
        )
    
    def update_preview(self, config: ProjectConfig) -> None:
        """Update preview with configuration"""
        self.project_config = config
        
        # Update preview content
        preview_content = self._generate_preview_content(config)
        preview_panel = self.query_one("#preview-content", Static)
        preview_panel.update(preview_content)
        
        # Update file structure
        file_structure = self._generate_file_structure(config)
        structure_panel = self.query_one("#file-structure", Static)
        structure_panel.update(file_structure)
    
    def _generate_preview_content(self, config: ProjectConfig) -> Panel:
        """Generate preview content panel"""
        content = Text()
        
        content.append(f"{config.template.get_icon()} {config.name}\n", style="bold blue")
        content.append(f"{config.description}\n\n", style="dim")
        
        content.append("Template: ", style="bold")
        content.append(f"{config.template.name}\n")
        
        content.append("Path: ", style="bold")
        content.append(f"{config.path}\n")
        
        if config.features:
            content.append("Features: ", style="bold")
            content.append(f"{', '.join(config.features)}\n")
        
        if config.build_tool:
            content.append("Build Tool: ", style="bold")
            content.append(f"{config.build_tool.value}\n")
        
        if config.test_framework:
            content.append("Test Framework: ", style="bold")
            content.append(f"{config.test_framework.value}\n")
        
        return Panel(content, title="Configuration Summary")
    
    def _generate_file_structure(self, config: ProjectConfig) -> Panel:
        """Generate file structure preview"""
        # Create a simple tree structure
        tree = RichTree(f"📁 {config.name}")
        
        # Add common directories and files based on template
        template = config.template
        
        # Add directories from template
        for directory in template.directories:
            tree.add(f"📁 {directory}")
        
        # Add common files
        common_files = {
            "README.md": "📝",
            ".gitignore": "🗺️",
            "package.json": "📦" if template.project_type in [ProjectType.REACT, ProjectType.NODE_JS] else None,
            "requirements.txt": "🐍" if template.project_type == ProjectType.PYTHON else None,
            "Cargo.toml": "🦀" if template.project_type == ProjectType.RUST else None,
            "go.mod": "🐹" if template.project_type == ProjectType.GO else None,
        }
        
        for filename, icon in common_files.items():
            if icon:  # Only add if relevant for project type
                tree.add(f"{icon} {filename}")
        
        # Add template-specific files
        for filename in template.files.keys():
            tree.add(f"📄 {filename}")
        
        return Panel(tree, title="Project Structure")


class TemplateSelectedMessage(Message):
    """Message sent when a template is selected"""
    
    def __init__(self, template: ProjectTemplate) -> None:
        super().__init__()
        self.template = template


class CreateProjectMessage(Message):
    """Message sent when project creation is completed"""
    
    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config


class ProjectWizardScreen(ModalScreen):
    """Main project wizard screen"""
    
    def __init__(self, project_manager) -> None:
        super().__init__()
        self.project_manager = project_manager
        self.templates = self._load_templates()
        self.current_template: Optional[ProjectTemplate] = None
        self.wizard_config: Dict[str, Any] = {}
        
    def compose(self):
        """Compose project wizard"""
        with Container(id="wizard-container"):
            # Header
            yield Label("🧜‍♂️ Project Wizard", classes="wizard-header")
            
            # Wizard content with tabs
            with TabbedContent("Template", "Configuration", "Features", "Preview"):
                # Template Selection Tab
                with TabPane("Template", id="template-tab"):
                    self.template_selector = TemplateSelector(self.templates)
                    yield self.template_selector
                
                # Configuration Tab
                with TabPane("Configuration", id="config-tab"):
                    self.config_form = ProjectConfigForm()
                    yield self.config_form
                
                # Features Tab
                with TabPane("Features", id="features-tab"):
                    self.feature_selector = FeatureSelector()
                    yield self.feature_selector
                
                # Preview Tab
                with TabPane("Preview", id="preview-tab"):
                    self.project_preview = ProjectPreview()
                    yield self.project_preview
            
            # Progress bar
            yield ProgressBar(id="wizard-progress", show_percentage=False)
            
            # Action buttons
            with Horizontal(classes="wizard-actions"):
                yield Button("← Back", id="back-button", disabled=True)
                yield Button("Next →", id="next-button")
                yield Button("🚀 Create Project", id="create-button", variant="primary", disabled=True)
                yield Button("❌ Cancel", id="cancel-button", variant="error")
    
    def _load_templates(self) -> List[ProjectTemplate]:
        """Load available project templates"""
        # This would load from a templates directory or configuration file
        # For now, return sample templates
        return [
            ProjectTemplate(
                id="react-typescript",
                name="React + TypeScript",
                description="Modern React application with TypeScript, Vite, and testing setup",
                project_type=ProjectType.REACT,
                technologies=["React", "TypeScript", "Vite", "ESLint", "Prettier"],
                build_tools=[BuildTool.VITE],
                test_frameworks=[TestFramework.JEST],
                features=["Hot Reload", "TypeScript", "ESLint", "Prettier"],
                dependencies={
                    "react": "^18.0.0",
                    "react-dom": "^18.0.0"
                },
                dev_dependencies={
                    "@types/react": "^18.0.0",
                    "@types/react-dom": "^18.0.0",
                    "typescript": "^5.0.0",
                    "vite": "^4.0.0"
                },
                directories=["src", "src/components", "src/hooks", "src/utils", "public", "tests"]
            ),
            ProjectTemplate(
                id="python-fastapi",
                name="Python FastAPI",
                description="FastAPI backend with async support, Pydantic, and testing",
                project_type=ProjectType.FASTAPI,
                technologies=["Python", "FastAPI", "Pydantic", "SQLAlchemy", "Pytest"],
                build_tools=[BuildTool.POETRY],
                test_frameworks=[TestFramework.PYTEST],
                features=["Async/Await", "Pydantic Models", "SQLAlchemy", "Auto-docs"],
                dependencies={
                    "fastapi": "^0.100.0",
                    "uvicorn": "^0.23.0",
                    "pydantic": "^2.0.0"
                },
                directories=["app", "app/api", "app/models", "app/services", "tests"]
            ),
            ProjectTemplate(
                id="nodejs-express",
                name="Node.js + Express",
                description="Express.js backend with TypeScript, JWT auth, and MongoDB",
                project_type=ProjectType.EXPRESS,
                technologies=["Node.js", "Express", "TypeScript", "MongoDB", "JWT"],
                build_tools=[BuildTool.NPM],
                test_frameworks=[TestFramework.JEST],
                features=["JWT Auth", "MongoDB", "TypeScript", "Middleware"],
                directories=["src", "src/controllers", "src/models", "src/routes", "src/middleware", "tests"]
            )
        ]
    
    @on(TemplateSelectedMessage)
    def handle_template_selected(self, message: TemplateSelectedMessage) -> None:
        """Handle template selection"""
        self.current_template = message.template
        self.feature_selector.update_template(message.template)
        
        # Enable next button
        next_button = self.query_one("#next-button", Button)
        next_button.disabled = False
        
        # Update progress
        progress = self.query_one("#wizard-progress", ProgressBar)
        progress.update(progress=25)
    
    @on(Button.Pressed, "#next-button")
    def next_step(self) -> None:
        """Go to next wizard step"""
        # Get current active tab
        tabbed_content = self.query_one(TabbedContent)
        current_tab = tabbed_content.active
        
        if current_tab == "template-tab":
            tabbed_content.active = "config-tab"
            progress = self.query_one("#wizard-progress", ProgressBar)
            progress.update(progress=50)
        elif current_tab == "config-tab":
            # Validate configuration
            errors = self.config_form.validate_form()
            if errors:
                # Show validation errors
                return
            
            tabbed_content.active = "features-tab"
            progress = self.query_one("#wizard-progress", ProgressBar)
            progress.update(progress=75)
        elif current_tab == "features-tab":
            tabbed_content.active = "preview-tab"
            
            # Generate final configuration and preview
            config = self._generate_project_config()
            self.project_preview.update_preview(config)
            
            progress = self.query_one("#wizard-progress", ProgressBar)
            progress.update(progress=100)
            
            # Enable create button
            create_button = self.query_one("#create-button", Button)
            create_button.disabled = False
        
        # Update back button state
        back_button = self.query_one("#back-button", Button)
        back_button.disabled = False
    
    @on(Button.Pressed, "#back-button")
    def back_step(self) -> None:
        """Go to previous wizard step"""
        tabbed_content = self.query_one(TabbedContent)
        current_tab = tabbed_content.active
        
        if current_tab == "config-tab":
            tabbed_content.active = "template-tab"
            back_button = self.query_one("#back-button", Button)
            back_button.disabled = True
            progress = self.query_one("#wizard-progress", ProgressBar)
            progress.update(progress=25)
        elif current_tab == "features-tab":
            tabbed_content.active = "config-tab"
            progress = self.query_one("#wizard-progress", ProgressBar)
            progress.update(progress=50)
        elif current_tab == "preview-tab":
            tabbed_content.active = "features-tab"
            create_button = self.query_one("#create-button", Button)
            create_button.disabled = True
            progress = self.query_one("#wizard-progress", ProgressBar)
            progress.update(progress=75)
    
    @on(Button.Pressed, "#create-button")
    def create_project(self) -> None:
        """Create the project"""
        config = self._generate_project_config()
        self.post_message(CreateProjectMessage(config))
        self.dismiss()
    
    @on(Button.Pressed, "#cancel-button")
    def cancel_wizard(self) -> None:
        """Cancel wizard"""
        self.dismiss()
    
    def _generate_project_config(self) -> ProjectConfig:
        """Generate final project configuration"""
        form_data = self.config_form.get_config_data()
        selected_features = self.feature_selector.get_selected_features()
        
        config = ProjectConfig(
            name=form_data['name'],
            description=form_data['description'],
            path=Path(form_data['path']) / form_data['name'],
            template=self.current_template,
            features=selected_features,
            git_init=form_data['git_init'],
            custom_config={
                'generate_readme': form_data.get('generate_readme', True),
                'setup_cicd': form_data.get('setup_cicd', False),
                'add_docker': form_data.get('add_docker', False),
                'enable_typescript': form_data.get('enable_typescript', True),
            }
        )
        
        return config


# Messages
class TemplateSelectedMessage(Message):
    """Message sent when template is selected"""
    
    def __init__(self, template: ProjectTemplate) -> None:
        super().__init__()
        self.template = template


class BrowseForPathMessage(Message):
    """Message to open path browser"""
    pass


class CreateProjectMessage(Message):
    """Message to create project"""
    
    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config