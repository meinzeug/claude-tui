# Claude TIU - Developer Guide

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure and Architecture](#project-structure-and-architecture)  
3. [Contributing Guidelines](#contributing-guidelines)
4. [Code Style and Conventions](#code-style-and-conventions)
5. [Adding New Features and Templates](#adding-new-features-and-templates)
6. [Plugin Development Guide](#plugin-development-guide)
7. [Testing Locally](#testing-locally)
8. [Debugging Techniques](#debugging-techniques)
9. [Release Process](#release-process)

---

## Development Environment Setup

### Prerequisites

#### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.9.0 or higher
- **Node.js**: 16.0 or higher (for Claude Flow integration)
- **Git**: Latest version
- **Memory**: 8GB RAM recommended for development
- **Storage**: 2GB free space

#### Required Tools
```bash
# Python development tools
pip install --upgrade pip setuptools wheel

# Development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Node.js tools for Claude Flow integration
npm install -g claude-flow@alpha

# Optional but recommended
pip install ipython  # Better REPL
pip install rich     # Enhanced terminal output
```

### Setting Up Development Environment

#### 1. Clone and Setup Repository
```bash
# Clone the repository
git clone https://github.com/your-org/claude-tiu.git
cd claude-tiu

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### 2. Development Configuration
```bash
# Create development configuration
cp config/development.yaml.template config/development.yaml

# Edit configuration with your settings
# - API keys (use test keys for development)
# - Debug settings
# - Local paths
```

#### 3. Install Claude TIU in Development Mode
```bash
# Install in editable mode
pip install -e .

# Verify installation
python -c "import claude_tiu; print('Installation successful')"

# Test CLI
python main.py --version
```

#### 4. IDE Setup

**Visual Studio Code**
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true
  },
  "extensions.recommendations": [
    "ms-python.python",
    "ms-python.flake8",
    "ms-python.black-formatter",
    "ms-python.mypy-type-checker"
  ]
}
```

**PyCharm**
- Configure Python interpreter to use the virtual environment
- Enable flake8 and black formatting
- Configure pytest as the test runner
- Set up run configurations for main.py

#### 5. Environment Variables
```bash
# .env file for development
CLAUDE_TIU_DEBUG=1
CLAUDE_TIU_LOG_LEVEL=DEBUG
CLAUDE_TIU_CONFIG_PATH=./config/development.yaml
CLAUDE_API_KEY=your_test_api_key
PYTHONPATH=./src
```

---

## Project Structure and Architecture

### High-Level Architecture

Claude TIU follows a modular, layered architecture designed for extensibility and maintainability:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TUI Layer     │    │   CLI Layer     │    │   API Layer     │
│   (Textual)     │    │   (Click)       │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Project   │  │   Workflow  │  │  Template   │            │
│  │  Manager    │  │  Engine     │  │   Manager   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
┌─────────────────────────────────────────────────────────────────┐
│                     Core Services                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ AI Interface│  │ Validation  │  │   Memory    │            │
│  │  (Claude)   │  │   Engine    │  │   Manager   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
┌─────────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │    File     │  │    Config   │  │   Plugin    │            │
│  │   System    │  │   Manager   │  │   System    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
claude-tiu/
├── src/                          # Source code
│   ├── claude_tiu/              # Main package
│   │   ├── __init__.py
│   │   ├── main.py              # Application entry point
│   │   ├── cli/                 # CLI interface
│   │   │   ├── __init__.py
│   │   │   ├── commands.py      # Click commands
│   │   │   └── utils.py         # CLI utilities
│   │   ├── tui/                 # Terminal UI
│   │   │   ├── __init__.py
│   │   │   ├── app.py           # Main Textual app
│   │   │   ├── screens/         # Application screens
│   │   │   │   ├── __init__.py
│   │   │   │   ├── welcome.py   # Welcome screen
│   │   │   │   ├── project.py   # Project management
│   │   │   │   ├── wizard.py    # Project wizard
│   │   │   │   └── settings.py  # Settings screen
│   │   │   └── widgets/         # Custom widgets
│   │   │       ├── __init__.py
│   │   │       ├── progress.py  # Progress widgets
│   │   │       ├── tree.py      # Project tree
│   │   │       └── validation.py # Validation widgets
│   │   ├── core/                # Core business logic
│   │   │   ├── __init__.py
│   │   │   ├── project_manager.py    # Project management
│   │   │   ├── ai_interface.py       # AI integration
│   │   │   ├── workflow_engine.py    # Workflow execution
│   │   │   ├── template_manager.py   # Template handling
│   │   │   ├── validation_engine.py  # Code validation
│   │   │   └── memory_manager.py     # State management
│   │   ├── integrations/        # External integrations
│   │   │   ├── __init__.py
│   │   │   ├── claude_code.py   # Claude Code integration
│   │   │   ├── claude_flow.py   # Claude Flow integration
│   │   │   ├── git_integration.py    # Git operations
│   │   │   └── editor_integration.py # Editor integration
│   │   ├── models/              # Data models
│   │   │   ├── __init__.py
│   │   │   ├── project.py       # Project model
│   │   │   ├── workflow.py      # Workflow model
│   │   │   ├── template.py      # Template model
│   │   │   └── validation.py    # Validation model
│   │   ├── utils/               # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── file_utils.py    # File operations
│   │   │   ├── config_utils.py  # Configuration
│   │   │   ├── logging_utils.py # Logging setup
│   │   │   └── async_utils.py   # Async helpers
│   │   └── plugins/             # Plugin system
│   │       ├── __init__.py
│   │       ├── base.py          # Base plugin class
│   │       ├── loader.py        # Plugin loader
│   │       └── registry.py      # Plugin registry
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   │   ├── test_core/          # Core logic tests
│   │   ├── test_tui/           # TUI tests
│   │   ├── test_cli/           # CLI tests
│   │   └── test_integrations/  # Integration tests
│   ├── integration/            # Integration tests
│   │   ├── test_workflows/     # Workflow tests
│   │   └── test_end_to_end/    # E2E tests
│   ├── fixtures/               # Test data
│   │   ├── projects/           # Sample projects
│   │   ├── templates/          # Test templates
│   │   └── workflows/          # Test workflows
│   └── conftest.py             # Pytest configuration
├── docs/                       # Documentation
│   ├── user-guide.md          # User documentation
│   ├── developer-guide.md     # This file
│   ├── api/                   # API documentation
│   └── examples/              # Usage examples
├── config/                     # Configuration files
│   ├── development.yaml.template
│   ├── production.yaml.template
│   └── testing.yaml
├── templates/                  # Built-in templates
│   ├── web/                   # Web development templates
│   ├── python/                # Python project templates
│   ├── mobile/                # Mobile app templates
│   └── ml/                    # ML project templates
├── workflows/                 # Built-in workflows
│   ├── web-development/       # Web development workflows
│   ├── api-development/       # API development workflows
│   └── testing/               # Testing workflows
├── scripts/                   # Development scripts
│   ├── setup-dev.sh          # Development setup
│   ├── run-tests.sh          # Test execution
│   ├── build.sh              # Build script
│   └── release.sh            # Release script
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Build configuration
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .github/                  # GitHub workflows
│   └── workflows/
│       ├── ci.yml            # Continuous integration
│       ├── release.yml       # Release workflow
│       └── docs.yml          # Documentation build
└── README.md                 # Project readme
```

### Core Components

#### 1. Project Manager (`core/project_manager.py`)
Handles all project lifecycle operations:

```python
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio

from claude_tiu.models.project import Project, ProjectConfig
from claude_tiu.core.workflow_engine import WorkflowEngine
from claude_tiu.core.validation_engine import ValidationEngine

class ProjectManager:
    """
    Central manager for all project operations.
    
    Responsibilities:
    - Project creation and initialization
    - Project state management
    - Workflow orchestration
    - Progress tracking and validation
    """
    
    def __init__(self, config_manager, ai_interface):
        self.config = config_manager
        self.ai = ai_interface
        self.workflow_engine = WorkflowEngine(ai_interface)
        self.validation_engine = ValidationEngine()
        self._active_projects: Dict[str, Project] = {}
    
    async def create_project(
        self, 
        name: str, 
        template: str, 
        config: ProjectConfig
    ) -> Project:
        """Create a new project from template."""
        # Implementation details...
        
    async def open_project(self, project_path: Path) -> Project:
        """Open existing project."""
        # Implementation details...
        
    async def execute_workflow(
        self, 
        project: Project, 
        workflow_name: str,
        variables: Dict[str, Any] = None
    ) -> None:
        """Execute workflow on project."""
        # Implementation details...
```

#### 2. AI Interface (`core/ai_interface.py`)
Manages communication with Claude Code and Claude Flow:

```python
import asyncio
import json
from typing import Dict, Any, List, Optional
import subprocess

from claude_tiu.models.ai_request import AIRequest, AIResponse
from claude_tiu.utils.async_utils import run_subprocess_async

class AIInterface:
    """
    Interface for AI operations using Claude Code and Claude Flow.
    
    Provides unified access to:
    - Claude Code for direct code generation
    - Claude Flow for workflow orchestration
    - Response validation and processing
    """
    
    def __init__(self, config):
        self.config = config
        self.claude_code_path = config.get('claude_code_path', 'claude')
        self.claude_flow_path = config.get('claude_flow_path', 'npx claude-flow@alpha')
    
    async def generate_code(
        self, 
        request: AIRequest
    ) -> AIResponse:
        """Generate code using Claude Code."""
        cmd = [
            self.claude_code_path,
            '--prompt', request.prompt,
            '--context', json.dumps(request.context),
            '--format', 'json'
        ]
        
        result = await run_subprocess_async(cmd)
        return AIResponse.from_json(result.stdout)
    
    async def execute_workflow(
        self, 
        workflow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow using Claude Flow."""
        # Implementation details...
```

#### 3. Validation Engine (`core/validation_engine.py`)
Implements the anti-hallucination system:

```python
import re
import ast
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from claude_tiu.models.validation import ValidationResult, ValidationRule

@dataclass
class ProgressReport:
    """Detailed analysis of project progress authenticity."""
    real_progress: float        # 0-100% of actual functional code
    claimed_progress: float     # 0-100% of what AI claims to have done
    quality_score: float        # 0-100 overall quality rating
    blocking_issues: List[str]  # Critical issues preventing progress
    placeholder_count: int      # Number of TODOs/placeholders found
    functionality_score: float # 0-100% of features actually working

class ValidationEngine:
    """
    Advanced validation system for detecting fake progress and placeholders.
    
    Features:
    - Multi-layer validation (syntax, semantic, functional)
    - Placeholder detection with pattern matching
    - AI-powered code quality assessment
    - Automatic issue fixing where possible
    """
    
    def __init__(self):
        self.placeholder_patterns = [
            r'TODO:|FIXME:|XXX:|HACK:',
            r'placeholder|dummy|mock|fake',
            r'NotImplemented|NotImplementedError',
            r'pass\s*#.*implement',
            r'console\.log\(["\']test["\']',
            r'function.*\{\s*\}',  # Empty functions
            r'def.*:\s*pass',      # Empty Python functions
        ]
        
    async def analyze_project_progress(
        self, 
        project_path: Path
    ) -> ProgressReport:
        """
        Comprehensive analysis of project progress authenticity.
        
        Combines multiple validation approaches:
        1. Static code analysis for placeholders
        2. Syntax validation for all files
        3. AI-powered semantic analysis
        4. Automated testing where possible
        """
        # Static analysis
        static_results = await self._static_analysis(project_path)
        
        # Semantic analysis using AI
        semantic_results = await self._semantic_analysis(project_path)
        
        # Functionality testing
        functional_results = await self._functionality_testing(project_path)
        
        return ProgressReport(
            real_progress=self._calculate_real_progress(
                static_results, semantic_results, functional_results
            ),
            claimed_progress=semantic_results.get('claimed_progress', 0),
            quality_score=self._calculate_quality_score(static_results),
            blocking_issues=static_results.get('blocking_issues', []),
            placeholder_count=static_results.get('placeholder_count', 0),
            functionality_score=functional_results.get('score', 0)
        )
```

#### 4. Workflow Engine (`core/workflow_engine.py`)
Executes complex multi-step workflows:

```python
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from claude_tiu.models.workflow import Workflow, Task, TaskStatus
from claude_tiu.core.ai_interface import AIInterface

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    outputs: List[str]
    execution_time: float
    validation_results: Optional[Dict[str, Any]] = None

class WorkflowEngine:
    """
    Orchestrates complex multi-task workflows.
    
    Features:
    - Dependency management between tasks
    - Parallel execution where possible
    - Error recovery and retry logic
    - Progress monitoring and reporting
    - Human approval points
    """
    
    def __init__(self, ai_interface: AIInterface):
        self.ai = ai_interface
        self._running_workflows: Dict[str, Workflow] = {}
        
    async def execute_workflow(
        self, 
        workflow: Workflow,
        variables: Dict[str, Any] = None
    ) -> Dict[str, TaskResult]:
        """Execute a complete workflow with all its phases and tasks."""
        
        workflow_id = workflow.id
        self._running_workflows[workflow_id] = workflow
        
        try:
            results = {}
            
            # Execute phases in order
            for phase in workflow.phases:
                if phase.parallel:
                    # Execute tasks in parallel
                    phase_results = await self._execute_phase_parallel(
                        phase, variables, results
                    )
                else:
                    # Execute tasks sequentially
                    phase_results = await self._execute_phase_sequential(
                        phase, variables, results
                    )
                
                results.update(phase_results)
                
                # Check if phase failed
                if any(r.status == TaskStatus.FAILED for r in phase_results.values()):
                    if not phase.continue_on_error:
                        break
                        
            return results
            
        finally:
            del self._running_workflows[workflow_id]
```

---

## Contributing Guidelines

### Getting Started with Contributions

#### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/claude-tiu.git
cd claude-tiu

# Add upstream remote
git remote add upstream https://github.com/original/claude-tiu.git
```

#### 2. Create Feature Branch
```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

#### 3. Development Workflow
```bash
# Make your changes
# Add tests for new functionality
# Run tests locally
./scripts/run-tests.sh

# Run linting and formatting
black src/ tests/
flake8 src/ tests/
mypy src/

# Commit changes
git add .
git commit -m "feat: add new feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Contribution Types

#### Bug Fixes
1. **Create Issue**: Report the bug with reproduction steps
2. **Write Test**: Add a test that demonstrates the bug
3. **Fix Bug**: Implement the fix
4. **Verify**: Ensure the test passes and no regressions

#### New Features
1. **Discussion**: Create issue to discuss the feature first
2. **Design**: Document the feature design and API
3. **Implementation**: Implement with comprehensive tests
4. **Documentation**: Update user and developer docs

#### Documentation
1. **User Documentation**: Improve user guides and tutorials
2. **API Documentation**: Add/improve docstrings and API docs
3. **Examples**: Create usage examples and tutorials
4. **Architecture**: Document system design and architecture

### Pull Request Process

#### 1. PR Preparation
```bash
# Sync with upstream
git fetch upstream
git rebase upstream/main

# Ensure all tests pass
./scripts/run-tests.sh

# Ensure code quality
./scripts/quality-check.sh
```

#### 2. PR Description Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Test coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

#### 3. Review Process
1. **Automated Checks**: CI/CD pipeline runs all tests
2. **Code Review**: Maintainers review the code
3. **Feedback**: Address review feedback
4. **Approval**: Get approval from maintainers
5. **Merge**: Maintainer merges the PR

### Community Guidelines

#### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

#### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time community chat
- **Email**: Direct contact with maintainers

---

## Code Style and Conventions

### Python Code Style

#### 1. PEP 8 Compliance
All Python code must follow PEP 8 with these specific preferences:

```python
# Line length: 88 characters (Black default)
# Indentation: 4 spaces
# Quote style: Double quotes preferred

# ✅ Good
def create_project(name: str, template: str) -> Project:
    """Create a new project from template."""
    return Project(name=name, template=template)

# ❌ Bad  
def create_project(name,template):
    return Project(name = name,template = template)
```

#### 2. Type Hints
Use type hints for all function signatures:

```python
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# ✅ Good
async def execute_workflow(
    self, 
    workflow_id: str,
    variables: Optional[Dict[str, Any]] = None
) -> List[TaskResult]:
    """Execute workflow with given variables."""
    pass

# ❌ Bad
async def execute_workflow(self, workflow_id, variables=None):
    pass
```

#### 3. Docstring Standards
Use Google-style docstrings:

```python
def validate_project_progress(
    self, 
    project_path: Path,
    validation_rules: Optional[List[ValidationRule]] = None
) -> ProgressReport:
    """
    Validate project progress and detect fake implementations.
    
    Args:
        project_path: Path to the project directory
        validation_rules: Optional custom validation rules
        
    Returns:
        ProgressReport containing authenticity analysis
        
    Raises:
        ValidationError: If project path is invalid
        
    Example:
        >>> validator = ValidationEngine()
        >>> report = validator.validate_project_progress(Path("./my-project"))
        >>> print(f"Real progress: {report.real_progress}%")
    """
    pass
```

#### 4. Error Handling
Use specific exception types and proper error handling:

```python
from claude_tiu.exceptions import (
    ProjectError, 
    ValidationError, 
    WorkflowError
)

# ✅ Good
async def create_project(self, name: str) -> Project:
    """Create new project."""
    try:
        # Validate name
        if not name or not name.strip():
            raise ProjectError("Project name cannot be empty")
            
        # Check if project exists
        if self._project_exists(name):
            raise ProjectError(f"Project '{name}' already exists")
            
        # Create project
        project = await self._create_project_impl(name)
        return project
        
    except OSError as e:
        raise ProjectError(f"Failed to create project: {e}") from e

# ❌ Bad
def create_project(self, name):
    try:
        # Create project
        pass
    except Exception:
        return None
```

#### 5. Class Design
Follow SOLID principles and composition over inheritance:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# Use protocols for interfaces
class ValidationRule(Protocol):
    """Protocol for validation rules."""
    
    def validate(self, code: str) -> bool:
        """Validate code against this rule."""
        ...

# Use dataclasses for data containers
@dataclass
class ProjectConfig:
    """Configuration for project creation."""
    name: str
    template: str
    framework: Optional[str] = None
    database: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Project name is required")

# Use composition in service classes
class ProjectManager:
    """Manages project lifecycle operations."""
    
    def __init__(
        self, 
        ai_interface: AIInterface,
        validation_engine: ValidationEngine,
        workflow_engine: WorkflowEngine
    ):
        self._ai = ai_interface
        self._validator = validation_engine
        self._workflow = workflow_engine
```

### JavaScript/TypeScript Style (for Claude Flow integration)

```typescript
// Use TypeScript for type safety
interface WorkflowConfig {
  name: string;
  version: string;
  tasks: Task[];
}

// Use async/await consistently
async function executeWorkflow(config: WorkflowConfig): Promise<WorkflowResult> {
  try {
    const result = await claudeFlow.execute(config);
    return result;
  } catch (error) {
    throw new WorkflowError(`Workflow execution failed: ${error.message}`);
  }
}

// Use proper error handling
class WorkflowError extends Error {
  constructor(message: string, public readonly cause?: Error) {
    super(message);
    this.name = 'WorkflowError';
  }
}
```

### YAML Configuration Style

```yaml
# Use consistent indentation (2 spaces)
# Use descriptive keys
# Include comments for complex configurations

project:
  name: "example-project"
  version: "1.0.0"
  
  # Template configuration
  template:
    type: "web-app"
    framework: "react"
    typescript: true
    
  # AI generation settings
  ai:
    model: "claude-3-opus"
    temperature: 0.1  # Lower for more consistent code
    max_tokens: 4000
    
    # Anti-hallucination settings
    validation:
      enabled: true
      strict_mode: true
      placeholder_threshold: 0.05  # 5% max placeholders
      
  # Workflow configuration
  workflow:
    phases:
      - name: "setup"
        parallel: false
        tasks:
          - name: "init-project"
            template: "project-init"
            
      - name: "development" 
        parallel: true  # Tasks can run in parallel
        tasks:
          - name: "frontend"
            template: "react-frontend"
          - name: "backend"
            template: "node-backend"
```

### File Naming Conventions

```
# Python files
snake_case.py

# JavaScript/TypeScript files  
kebab-case.js
kebab-case.ts

# Configuration files
kebab-case.yaml
kebab-case.json

# Documentation
kebab-case.md

# Templates
template-name.yaml

# Workflows
workflow-name.yaml

# Test files
test_module_name.py
module-name.test.ts
```

### Import Organization

```python
# Standard library imports
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party imports
import click
from textual.app import App
from rich.console import Console

# Local imports
from claude_tiu.core.project_manager import ProjectManager
from claude_tiu.models.project import Project
from claude_tiu.utils.logging_utils import get_logger

# Relative imports (avoid when possible)
from .validation_engine import ValidationEngine
```

---

## Adding New Features and Templates

### Adding New Features

#### 1. Feature Planning
Before implementing a new feature:

1. **Create Feature Issue**: Document the feature requirements
2. **Design API**: Define the public interface
3. **Consider Impact**: Assess impact on existing functionality
4. **Plan Testing**: Define test strategy

#### 2. Implementation Process

**Step 1: Core Logic**
```python
# src/claude_tiu/core/new_feature.py
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for the new feature."""
    enabled: bool = True
    options: Dict[str, Any] = None
    
class NewFeature:
    """Implementation of the new feature."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    async def execute(self, input_data: Any) -> Any:
        """Main feature execution logic."""
        if not self.config.enabled:
            return None
            
        # Implementation here
        pass
```

**Step 2: Integration**
```python
# src/claude_tiu/core/project_manager.py
from claude_tiu.core.new_feature import NewFeature

class ProjectManager:
    def __init__(self, ...):
        # ... existing code ...
        self.new_feature = NewFeature(config.new_feature)
        
    async def use_new_feature(self, data: Any) -> Any:
        """Use the new feature."""
        return await self.new_feature.execute(data)
```

**Step 3: CLI Integration**
```python
# src/claude_tiu/cli/commands.py
@click.command()
@click.option('--option', help='Feature option')
def new_feature_command(option: str):
    """Use the new feature via CLI."""
    # Implementation
    pass

# Add to main CLI group
main.add_command(new_feature_command)
```

**Step 4: TUI Integration**
```python
# src/claude_tiu/tui/screens/new_feature_screen.py
from textual.screen import Screen
from textual.widgets import Button, Input

class NewFeatureScreen(Screen):
    """Screen for the new feature."""
    
    def compose(self):
        yield Button("Execute Feature", id="execute")
        yield Input(placeholder="Enter parameters...")
        
    def on_button_pressed(self, event):
        if event.button.id == "execute":
            # Handle feature execution
            pass
```

#### 3. Testing New Features

**Unit Tests**
```python
# tests/unit/test_new_feature.py
import pytest
from claude_tiu.core.new_feature import NewFeature, FeatureConfig

class TestNewFeature:
    """Test suite for the new feature."""
    
    @pytest.fixture
    def feature(self):
        """Create feature instance for testing."""
        config = FeatureConfig(enabled=True)
        return NewFeature(config)
        
    async def test_execute_basic(self, feature):
        """Test basic feature execution."""
        result = await feature.execute("test data")
        assert result is not None
        
    async def test_execute_disabled(self):
        """Test feature execution when disabled."""
        config = FeatureConfig(enabled=False)
        feature = NewFeature(config)
        result = await feature.execute("test data")
        assert result is None
```

**Integration Tests**
```python
# tests/integration/test_new_feature_integration.py
import pytest
from claude_tiu.core.project_manager import ProjectManager

class TestNewFeatureIntegration:
    """Integration tests for new feature."""
    
    async def test_feature_in_project_workflow(self, project_manager):
        """Test feature integration in project workflow."""
        # Test implementation
        pass
```

### Adding New Templates

#### 1. Template Structure
Templates are defined in YAML format with specific structure:

```yaml
# templates/web/new-template.yaml
name: "new-template"
version: "1.0.0"
description: "Description of the new template"
category: "web"
tags: ["framework", "typescript", "modern"]

# Template metadata
metadata:
  author: "Your Name"
  license: "MIT"
  min_claude_tiu_version: "1.0.0"
  
# Variables for customization
variables:
  project_name:
    type: string
    description: "Name of the project"
    required: true
    validation: "^[a-zA-Z][a-zA-Z0-9-_]*$"
    
  framework:
    type: choice
    description: "Frontend framework to use"
    options: ["react", "vue", "angular"]
    default: "react"
    
  styling:
    type: multi_choice
    description: "Styling options"
    options: ["css-modules", "styled-components", "tailwind"]
    default: ["css-modules"]
    
  features:
    type: list
    description: "Additional features to include"
    items:
      - name: "authentication"
        description: "User authentication system"
      - name: "api-integration"
        description: "REST API integration"

# Main template content
template: |
  Create a {framework} application named {project_name}.
  
  Project Requirements:
  - Use TypeScript for type safety
  - Include {styling} for styling
  - Implement the following features: {features}
  
  Technical Specifications:
  - Follow industry best practices
  - Include comprehensive error handling
  - Add unit tests with high coverage
  - Implement proper logging
  - Use environment variables for configuration
  
  Project Structure:
  - Organize code in logical modules
  - Separate concerns appropriately
  - Include documentation
  - Set up development scripts
  
  Code Quality Requirements:
  - No TODO or placeholder comments
  - Complete implementation of all features
  - Proper error handling for edge cases
  - Input validation and sanitization
  - Security best practices

# File-specific templates (optional)
files:
  "package.json":
    template: |
      {
        "name": "{project_name}",
        "version": "1.0.0",
        "description": "Generated by Claude TIU",
        "main": "index.js",
        "scripts": {
          "start": "react-scripts start",
          "build": "react-scripts build",
          "test": "react-scripts test",
          "eject": "react-scripts eject"
        },
        "dependencies": {
          {% if framework == "react" %}
          "react": "^18.0.0",
          "react-dom": "^18.0.0",
          "react-scripts": "5.0.1"
          {% endif %}
        }
      }
      
  "src/App.{tsx|jsx}":
    condition: "framework == 'react'"
    template: |
      import React from 'react';
      {% if 'styled-components' in styling %}
      import styled from 'styled-components';
      {% endif %}
      
      function App() {
        return (
          <div className="App">
            <header className="App-header">
              <h1>Welcome to {project_name}</h1>
            </header>
          </div>
        );
      }
      
      export default App;

# Validation rules specific to this template
validation:
  rules:
    - name: "React best practices"
      applies_to: "src/**/*.{tsx,jsx}"
      checks:
        - "proper_hook_usage"
        - "component_naming"
        - "prop_types_or_typescript"
        
    - name: "No hardcoded values"
      pattern: "http://localhost"
      severity: "warning"
      message: "Use environment variables for API endpoints"

# Post-generation hooks
hooks:
  post_generation:
    - name: "install_dependencies"
      command: "npm install"
      working_dir: "{project_root}"
      
    - name: "initial_commit"
      command: "git init && git add . && git commit -m 'Initial commit from Claude TIU'"
      working_dir: "{project_root}"
      
    - name: "setup_development"
      command: "npm run setup:dev"
      working_dir: "{project_root}"
      condition: "development_mode == true"
```

#### 2. Template Categories

**Web Development Templates**
```
templates/web/
├── react-app/
│   ├── basic.yaml
│   ├── typescript.yaml
│   └── fullstack.yaml
├── vue-app/
│   ├── basic.yaml
│   └── composition-api.yaml
└── node-api/
    ├── express.yaml
    ├── fastify.yaml
    └── graphql.yaml
```

**Python Templates**
```
templates/python/
├── web-api/
│   ├── fastapi.yaml
│   ├── django.yaml
│   └── flask.yaml
├── cli-tool/
│   ├── click.yaml
│   └── argparse.yaml
└── data-science/
    ├── jupyter.yaml
    ├── pandas-analysis.yaml
    └── ml-pipeline.yaml
```

#### 3. Template Development Process

**Step 1: Create Template File**
```bash
# Create new template directory
mkdir -p templates/category/template-name

# Create template definition
touch templates/category/template-name/template.yaml

# Create example project (for testing)
mkdir -p templates/category/template-name/example
```

**Step 2: Test Template**
```bash
# Test template generation
python -m claude_tiu.cli create-project \
  --template category/template-name \
  --name test-project \
  --output ./test-output

# Validate generated project
python -m claude_tiu.cli validate-project ./test-output/test-project
```

**Step 3: Add Template Tests**
```python
# tests/unit/test_templates.py
import pytest
from claude_tiu.core.template_manager import TemplateManager

class TestNewTemplate:
    """Test suite for new template."""
    
    @pytest.fixture
    def template_manager(self):
        return TemplateManager()
        
    async def test_template_loading(self, template_manager):
        """Test template loads correctly."""
        template = template_manager.load_template("category/template-name")
        assert template.name == "template-name"
        
    async def test_template_generation(self, template_manager):
        """Test template generates valid project."""
        variables = {
            "project_name": "test-project",
            "framework": "react"
        }
        
        project = await template_manager.generate_project(
            "category/template-name", 
            variables
        )
        
        assert project.name == "test-project"
        # Add more assertions
```

#### 4. Template Best Practices

**Clear Variable Definitions**
```yaml
variables:
  api_base_url:
    type: string
    description: "Base URL for API endpoints"
    required: true
    default: "https://api.example.com"
    validation: "^https?://.*"
    example: "https://api.myapp.com"
    
  database_config:
    type: object
    description: "Database configuration"
    required: true
    properties:
      type:
        type: choice
        options: ["postgresql", "mysql", "mongodb"]
        default: "postgresql"
      host:
        type: string
        default: "localhost"
      port:
        type: integer
        default: 5432
```

**Comprehensive Templates**
```yaml
template: |
  You are creating a production-ready {framework} application.
  
  MANDATORY REQUIREMENTS:
  1. Complete implementation - no TODO comments or placeholders
  2. Error handling for all user inputs and external calls
  3. Input validation and sanitization
  4. Proper logging throughout the application
  5. Environment-based configuration
  6. Security best practices implementation
  7. Unit tests for core functionality
  8. Documentation for setup and usage
  
  PROJECT STRUCTURE:
  Follow industry-standard project organization with:
  - Clear separation of concerns
  - Modular architecture
  - Consistent naming conventions
  - Proper dependency management
  
  SPECIFIC FEATURES TO IMPLEMENT:
  {feature_list}
  
  QUALITY STANDARDS:
  - Code coverage: minimum 80%
  - Performance: response times under 200ms
  - Security: no known vulnerabilities
  - Maintainability: clean, readable code with comments
```

**Validation Integration**
```yaml
validation:
  post_generation:
    - name: "syntax_check"
      description: "Verify all generated files have valid syntax"
      
    - name: "dependency_check"
      description: "Ensure all dependencies are properly declared"
      
    - name: "security_scan"
      description: "Check for common security vulnerabilities"
      
    - name: "test_execution"
      description: "Run generated tests to verify functionality"
      
  quality_gates:
    - name: "no_placeholders"
      threshold: 0
      description: "No TODO or placeholder code allowed"
      
    - name: "test_coverage"
      threshold: 80
      description: "Minimum test coverage percentage"
```

---

## Plugin Development Guide

### Plugin Architecture

Claude TIU uses a flexible plugin architecture that allows extending functionality without modifying core code. Plugins can add new:

- Templates and workflows
- Validation rules  
- AI integrations
- TUI screens and widgets
- CLI commands
- File generators

#### Plugin Interface

```python
# src/claude_tiu/plugins/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class PluginMetadata:
    """Plugin metadata and information."""
    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: Optional[str] = None
    min_claude_tiu_version: str = "1.0.0"
    dependencies: List[str] = None

class Plugin(ABC):
    """Base class for all Claude TIU plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self, app_context: 'AppContext') -> None:
        """Initialize the plugin with application context."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of plugin resources."""
        pass
    
    def get_templates(self) -> Dict[str, Any]:
        """Return templates provided by this plugin."""
        return {}
    
    def get_workflows(self) -> Dict[str, Any]:
        """Return workflows provided by this plugin."""
        return {}
    
    def get_validation_rules(self) -> List[Any]:
        """Return validation rules provided by this plugin."""
        return []
    
    def get_cli_commands(self) -> List[Any]:
        """Return CLI commands provided by this plugin."""
        return []
    
    def get_tui_screens(self) -> Dict[str, Any]:
        """Return TUI screens provided by this plugin."""
        return {}
```

### Creating a Plugin

#### 1. Plugin Project Structure

```
my-claude-tiu-plugin/
├── src/
│   └── my_plugin/
│       ├── __init__.py
│       ├── plugin.py          # Main plugin class
│       ├── templates/         # Plugin templates
│       │   └── my-template.yaml
│       ├── workflows/         # Plugin workflows
│       │   └── my-workflow.yaml
│       ├── validation/        # Validation rules
│       │   └── rules.py
│       ├── cli/               # CLI extensions
│       │   └── commands.py
│       └── tui/               # TUI extensions
│           ├── screens.py
│           └── widgets.py
├── tests/                     # Plugin tests
├── docs/                      # Plugin documentation
├── setup.py                   # Installation script
├── requirements.txt           # Dependencies
└── README.md                  # Plugin readme
```

#### 2. Basic Plugin Implementation

```python
# src/my_plugin/plugin.py
import asyncio
from pathlib import Path
from typing import Dict, Any, List

from claude_tiu.plugins.base import Plugin, PluginMetadata
from claude_tiu.models.template import Template
from claude_tiu.models.workflow import Workflow

class MyPlugin(Plugin):
    """Example plugin for Claude TIU."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-claude-tiu-plugin",
            version="1.0.0",
            description="Example plugin demonstrating Claude TIU extensibility",
            author="Your Name",
            license="MIT",
            homepage="https://github.com/yourname/my-claude-tiu-plugin",
            min_claude_tiu_version="1.0.0"
        )
    
    async def initialize(self, app_context) -> None:
        """Initialize plugin with app context."""
        self.app_context = app_context
        self.logger = app_context.get_logger(f"plugin.{self.metadata.name}")
        
        self.logger.info(f"Initializing plugin {self.metadata.name}")
        
        # Load plugin resources
        await self._load_templates()
        await self._load_workflows()
        
    async def shutdown(self) -> None:
        """Clean shutdown."""
        self.logger.info(f"Shutting down plugin {self.metadata.name}")
        
    def get_templates(self) -> Dict[str, Template]:
        """Return plugin templates."""
        return self._templates
    
    def get_workflows(self) -> Dict[str, Workflow]:
        """Return plugin workflows."""
        return self._workflows
    
    async def _load_templates(self) -> None:
        """Load templates from plugin directory."""
        template_dir = Path(__file__).parent / "templates"
        self._templates = {}
        
        for template_file in template_dir.glob("*.yaml"):
            template = await self._load_template_file(template_file)
            self._templates[template.name] = template
            
    async def _load_workflows(self) -> None:
        """Load workflows from plugin directory."""
        workflow_dir = Path(__file__).parent / "workflows"
        self._workflows = {}
        
        for workflow_file in workflow_dir.glob("*.yaml"):
            workflow = await self._load_workflow_file(workflow_file)
            self._workflows[workflow.name] = workflow
```

#### 3. Template Plugin Example

```python
# src/my_plugin/templates.py
from claude_tiu.plugins.base import Plugin
from claude_tiu.models.template import Template

class CustomTemplatePlugin(Plugin):
    """Plugin providing custom project templates."""
    
    def get_templates(self) -> Dict[str, Template]:
        """Return custom templates."""
        return {
            "fastapi-microservice": self._create_fastapi_template(),
            "react-dashboard": self._create_dashboard_template(),
            "ml-pipeline": self._create_ml_template()
        }
    
    def _create_fastapi_template(self) -> Template:
        """Create FastAPI microservice template."""
        return Template(
            name="fastapi-microservice",
            description="Production-ready FastAPI microservice",
            category="api",
            variables={
                "service_name": {
                    "type": "string",
                    "required": True,
                    "description": "Name of the microservice"
                },
                "database": {
                    "type": "choice",
                    "options": ["postgresql", "mysql", "mongodb"],
                    "default": "postgresql"
                }
            },
            template="""
            Create a production-ready FastAPI microservice named {service_name}.
            
            Requirements:
            - Use {database} as the database
            - Include authentication and authorization
            - Add comprehensive API documentation
            - Implement health checks and metrics
            - Include Docker containerization
            - Add comprehensive tests
            - Implement proper logging and monitoring
            
            Architecture:
            - Follow clean architecture principles
            - Separate business logic from frameworks
            - Use dependency injection
            - Implement proper error handling
            """,
            files={
                "main.py": """
                # FastAPI application entry point
                from fastapi import FastAPI
                from fastapi.middleware.cors import CORSMiddleware
                
                app = FastAPI(
                    title="{service_name}",
                    description="Generated by Claude TIU",
                    version="1.0.0"
                )
                
                # Add CORS middleware
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"]
                )
                
                @app.get("/")
                async def root():
                    return {"message": "Welcome to {service_name}"}
                    
                @app.get("/health")
                async def health_check():
                    return {"status": "healthy"}
                """
            }
        )
```

#### 4. Validation Plugin Example

```python
# src/my_plugin/validation.py
from claude_tiu.plugins.base import Plugin
from claude_tiu.models.validation import ValidationRule

class SecurityValidationPlugin(Plugin):
    """Plugin providing security validation rules."""
    
    def get_validation_rules(self) -> List[ValidationRule]:
        """Return security validation rules."""
        return [
            self._create_secret_detection_rule(),
            self._create_sql_injection_rule(),
            self._create_xss_prevention_rule()
        ]
    
    def _create_secret_detection_rule(self) -> ValidationRule:
        """Create rule to detect hardcoded secrets."""
        return ValidationRule(
            name="hardcoded_secrets",
            description="Detect hardcoded secrets in code",
            pattern=r'(password|api_key|secret|token)\s*=\s*["\'][^"\']{8,}["\']',
            severity="critical",
            message="Hardcoded secrets detected. Use environment variables instead.",
            auto_fix=False,
            applies_to=["**/*.py", "**/*.js", "**/*.ts"]
        )
    
    def _create_sql_injection_rule(self) -> ValidationRule:
        """Create rule to prevent SQL injection vulnerabilities."""
        return ValidationRule(
            name="sql_injection_prevention", 
            description="Prevent SQL injection vulnerabilities",
            pattern=r'(SELECT|INSERT|UPDATE|DELETE).*\+.*WHERE',
            severity="high",
            message="Potential SQL injection vulnerability. Use parameterized queries.",
            suggestion="Use parameterized queries or ORM methods instead of string concatenation.",
            applies_to=["**/*.py", "**/*.js"]
        )
```

#### 5. CLI Extension Plugin

```python
# src/my_plugin/cli.py
import click
from claude_tiu.plugins.base import Plugin

class CLIExtensionPlugin(Plugin):
    """Plugin adding custom CLI commands."""
    
    def get_cli_commands(self) -> List[click.Command]:
        """Return custom CLI commands."""
        return [
            self.custom_analyze_command(),
            self.security_scan_command()
        ]
    
    def custom_analyze_command(self) -> click.Command:
        """Create custom project analysis command."""
        @click.command("analyze")
        @click.argument("project_path")
        @click.option("--format", default="json", help="Output format")
        def analyze(project_path: str, format: str):
            """Analyze project with custom metrics."""
            # Implementation here
            click.echo(f"Analyzing project at {project_path}")
            
        return analyze
    
    def security_scan_command(self) -> click.Command:
        """Create security scanning command."""
        @click.command("security-scan")
        @click.argument("project_path")
        @click.option("--severity", default="medium", help="Minimum severity level")
        def security_scan(project_path: str, severity: str):
            """Perform security scan on project."""
            # Implementation here
            click.echo(f"Running security scan on {project_path}")
            
        return security_scan
```

#### 6. TUI Extension Plugin

```python
# src/my_plugin/tui.py
from textual.screen import Screen
from textual.widgets import Button, Label
from claude_tiu.plugins.base import Plugin

class CustomScreen(Screen):
    """Custom TUI screen provided by plugin."""
    
    def compose(self):
        yield Label("Custom Plugin Screen")
        yield Button("Custom Action", id="custom_action")
        
    def on_button_pressed(self, event):
        if event.button.id == "custom_action":
            # Handle custom action
            self.notify("Custom action executed!")

class TUIExtensionPlugin(Plugin):
    """Plugin adding custom TUI screens."""
    
    def get_tui_screens(self) -> Dict[str, Screen]:
        """Return custom TUI screens."""
        return {
            "custom_screen": CustomScreen
        }
```

### Plugin Installation and Distribution

#### 1. Plugin Package Setup

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="my-claude-tiu-plugin",
    version="1.0.0",
    description="Custom plugin for Claude TIU",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "my_plugin": [
            "templates/*.yaml",
            "workflows/*.yaml",
        ]
    },
    install_requires=[
        "claude-tiu>=1.0.0",
    ],
    entry_points={
        "claude_tiu.plugins": [
            "my_plugin = my_plugin.plugin:MyPlugin",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
```

#### 2. Plugin Installation

```bash
# Install from PyPI
pip install my-claude-tiu-plugin

# Install from source
pip install git+https://github.com/yourname/my-claude-tiu-plugin.git

# Install in development mode
git clone https://github.com/yourname/my-claude-tiu-plugin.git
cd my-claude-tiu-plugin
pip install -e .
```

#### 3. Plugin Configuration

```yaml
# ~/.claude-tiu/config.yaml
plugins:
  enabled:
    - "my-claude-tiu-plugin"
    - "another-plugin"
    
  settings:
    my-claude-tiu-plugin:
      custom_setting: "value"
      enable_feature_x: true
      
    another-plugin:
      api_endpoint: "https://api.example.com"
```

### Plugin Testing

#### 1. Unit Tests

```python
# tests/test_plugin.py
import pytest
from my_plugin.plugin import MyPlugin
from claude_tiu.core.app_context import AppContext

class TestMyPlugin:
    """Test suite for the plugin."""
    
    @pytest.fixture
    async def plugin(self):
        """Create plugin instance for testing."""
        plugin = MyPlugin()
        app_context = AppContext()  # Mock context
        await plugin.initialize(app_context)
        return plugin
    
    async def test_plugin_metadata(self, plugin):
        """Test plugin metadata is correct."""
        metadata = plugin.metadata
        assert metadata.name == "my-claude-tiu-plugin"
        assert metadata.version == "1.0.0"
        
    async def test_templates_loaded(self, plugin):
        """Test templates are loaded correctly."""
        templates = plugin.get_templates()
        assert len(templates) > 0
        assert "my-template" in templates
```

#### 2. Integration Tests

```python
# tests/test_integration.py
import pytest
from claude_tiu.core.plugin_manager import PluginManager
from claude_tiu.core.app_context import AppContext

class TestPluginIntegration:
    """Integration tests for plugin with Claude TIU."""
    
    async def test_plugin_registration(self):
        """Test plugin registers correctly."""
        app_context = AppContext()
        plugin_manager = PluginManager(app_context)
        
        await plugin_manager.load_plugin("my-claude-tiu-plugin")
        
        assert "my-claude-tiu-plugin" in plugin_manager.loaded_plugins
        
    async def test_template_availability(self):
        """Test plugin templates are available in app."""
        # Test implementation
        pass
```

---

## Testing Locally

### Test Suite Organization

Claude TIU uses pytest for comprehensive testing across multiple layers:

```
tests/
├── unit/                      # Fast, isolated unit tests
│   ├── test_core/            # Core business logic
│   ├── test_models/          # Data models
│   ├── test_utils/           # Utility functions
│   └── test_plugins/         # Plugin system
├── integration/              # Integration tests
│   ├── test_ai_integration/  # AI service integration
│   ├── test_workflows/       # Workflow execution
│   └── test_file_operations/ # File system operations
├── e2e/                      # End-to-end tests
│   ├── test_project_creation/# Complete project workflows
│   ├── test_tui/            # TUI interaction tests
│   └── test_cli/            # CLI command tests
├── performance/              # Performance benchmarks
│   ├── test_generation_speed/# Code generation performance
│   └── test_validation_speed/# Validation performance
├── fixtures/                 # Test data and mocks
│   ├── projects/            # Sample projects
│   ├── templates/           # Test templates
│   └── responses/           # Mock AI responses
└── conftest.py              # Pytest configuration
```

### Running Tests

#### 1. Basic Test Execution

```bash
# Run all tests
./scripts/run-tests.sh

# Or manually
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests
pytest tests/e2e/                     # End-to-end tests

# Run tests with coverage
pytest --cov=claude_tiu --cov-report=html

# Run tests in parallel (faster)
pytest -n auto
```

#### 2. Test Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
    --cov=claude_tiu
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    ai_required: Tests that require AI API access
```

#### 3. Test Fixtures and Mocks

```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import tempfile
import shutil

from claude_tiu.core.project_manager import ProjectManager
from claude_tiu.core.ai_interface import AIInterface
from claude_tiu.models.project import Project

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def mock_ai_interface():
    """Create mock AI interface for testing."""
    mock = AsyncMock(spec=AIInterface)
    mock.generate_code.return_value = Mock(
        code="print('Hello, World!')",
        status="success"
    )
    return mock

@pytest.fixture
def sample_project(temp_dir):
    """Create sample project for testing."""
    project = Project(
        name="test-project",
        path=temp_dir / "test-project",
        template="python-cli"
    )
    project.path.mkdir()
    return project

@pytest.fixture
def project_manager(mock_ai_interface):
    """Create project manager with mocked dependencies."""
    return ProjectManager(ai_interface=mock_ai_interface)
```

### Unit Testing Examples

#### 1. Testing Core Logic

```python
# tests/unit/test_core/test_project_manager.py
import pytest
from unittest.mock import AsyncMock, Mock
from pathlib import Path

from claude_tiu.core.project_manager import ProjectManager
from claude_tiu.models.project import Project, ProjectConfig
from claude_tiu.exceptions import ProjectError

class TestProjectManager:
    """Test suite for ProjectManager."""
    
    async def test_create_project_success(self, project_manager, temp_dir):
        """Test successful project creation."""
        config = ProjectConfig(
            name="test-project",
            template="python-cli",
            output_dir=temp_dir
        )
        
        project = await project_manager.create_project(config)
        
        assert project.name == "test-project"
        assert project.path.exists()
        assert project.template == "python-cli"
        
    async def test_create_project_duplicate_name(self, project_manager, temp_dir):
        """Test error when creating project with existing name."""
        config = ProjectConfig(
            name="duplicate-project",
            template="python-cli", 
            output_dir=temp_dir
        )
        
        # Create first project
        await project_manager.create_project(config)
        
        # Try to create duplicate
        with pytest.raises(ProjectError, match="already exists"):
            await project_manager.create_project(config)
            
    async def test_create_project_invalid_template(self, project_manager, temp_dir):
        """Test error when using invalid template."""
        config = ProjectConfig(
            name="test-project",
            template="nonexistent-template",
            output_dir=temp_dir
        )
        
        with pytest.raises(ProjectError, match="Template not found"):
            await project_manager.create_project(config)
```

#### 2. Testing AI Integration

```python
# tests/unit/test_core/test_ai_interface.py
import pytest
from unittest.mock import patch, Mock
import json

from claude_tiu.core.ai_interface import AIInterface
from claude_tiu.models.ai_request import AIRequest

class TestAIInterface:
    """Test suite for AI integration."""
    
    @pytest.fixture
    def ai_interface(self):
        config = {"api_key": "test-key"}
        return AIInterface(config)
    
    @patch('claude_tiu.core.ai_interface.run_subprocess_async')
    async def test_generate_code_success(self, mock_subprocess, ai_interface):
        """Test successful code generation."""
        # Mock subprocess response
        mock_result = Mock()
        mock_result.stdout = json.dumps({
            "code": "print('Hello, World!')",
            "status": "success"
        })
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        request = AIRequest(
            prompt="Create a hello world program",
            context={"language": "python"}
        )
        
        response = await ai_interface.generate_code(request)
        
        assert response.code == "print('Hello, World!')"
        assert response.status == "success"
        
    @patch('claude_tiu.core.ai_interface.run_subprocess_async')
    async def test_generate_code_failure(self, mock_subprocess, ai_interface):
        """Test AI generation failure handling."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "API rate limit exceeded"
        mock_subprocess.return_value = mock_result
        
        request = AIRequest(prompt="Create a program")
        
        with pytest.raises(Exception, match="rate limit"):
            await ai_interface.generate_code(request)
```

#### 3. Testing Validation Engine

```python
# tests/unit/test_core/test_validation_engine.py
import pytest
from pathlib import Path

from claude_tiu.core.validation_engine import ValidationEngine
from claude_tiu.models.validation import ValidationResult

class TestValidationEngine:
    """Test suite for ValidationEngine."""
    
    @pytest.fixture
    def validator(self):
        return ValidationEngine()
    
    async def test_detect_placeholders(self, validator, temp_dir):
        """Test placeholder detection."""
        # Create file with placeholders
        test_file = temp_dir / "test.py"
        test_file.write_text("""
def example_function():
    # TODO: implement this function
    pass
    
def another_function():
    print("Hello, World!")
        """)
        
        result = await validator.analyze_project_progress(temp_dir)
        
        assert result.placeholder_count > 0
        assert result.real_progress < 100
        
    async def test_no_placeholders(self, validator, temp_dir):
        """Test validation with no placeholders."""
        # Create file without placeholders
        test_file = temp_dir / "test.py"
        test_file.write_text("""
def example_function():
    return "Hello, World!"
    
def another_function():
    return 42
        """)
        
        result = await validator.analyze_project_progress(temp_dir)
        
        assert result.placeholder_count == 0
        assert result.real_progress > 50  # Should be higher
```

### Integration Testing

#### 1. Workflow Integration Tests

```python
# tests/integration/test_workflows/test_workflow_execution.py
import pytest
from pathlib import Path

from claude_tiu.core.workflow_engine import WorkflowEngine
from claude_tiu.models.workflow import Workflow

class TestWorkflowIntegration:
    """Integration tests for workflow execution."""
    
    @pytest.fixture
    def workflow_engine(self, mock_ai_interface):
        return WorkflowEngine(mock_ai_interface)
    
    async def test_simple_workflow_execution(self, workflow_engine, temp_dir):
        """Test execution of simple workflow."""
        workflow = Workflow(
            name="simple-test",
            phases=[
                {
                    "name": "setup",
                    "tasks": [
                        {
                            "name": "create-file",
                            "ai_prompt": "Create a Python hello world script",
                            "outputs": ["hello.py"]
                        }
                    ]
                }
            ]
        )
        
        results = await workflow_engine.execute_workflow(
            workflow, 
            {"output_dir": temp_dir}
        )
        
        assert len(results) == 1
        assert "create-file" in results
        assert results["create-file"].status == "completed"
        
    async def test_workflow_with_dependencies(self, workflow_engine, temp_dir):
        """Test workflow with task dependencies."""
        workflow = Workflow(
            name="dependency-test",
            phases=[
                {
                    "name": "development",
                    "tasks": [
                        {
                            "name": "setup-project",
                            "ai_prompt": "Create project structure",
                            "outputs": ["src/", "tests/"]
                        },
                        {
                            "name": "implement-code",
                            "depends_on": ["setup-project"],
                            "ai_prompt": "Implement main functionality",
                            "outputs": ["src/main.py"]
                        }
                    ]
                }
            ]
        )
        
        results = await workflow_engine.execute_workflow(workflow)
        
        # Verify both tasks completed
        assert results["setup-project"].status == "completed"
        assert results["implement-code"].status == "completed"
        
        # Verify dependency order was respected
        setup_time = results["setup-project"].execution_time
        implement_time = results["implement-code"].execution_time
        # Implementation should start after setup
```

#### 2. File System Integration Tests

```python
# tests/integration/test_file_operations/test_project_generation.py
import pytest
from pathlib import Path

from claude_tiu.core.project_manager import ProjectManager
from claude_tiu.models.project import ProjectConfig

class TestProjectGeneration:
    """Integration tests for project file generation."""
    
    async def test_python_project_generation(self, project_manager, temp_dir):
        """Test complete Python project generation."""
        config = ProjectConfig(
            name="test-python-project",
            template="python-cli",
            output_dir=temp_dir,
            variables={
                "author": "Test Author",
                "license": "MIT"
            }
        )
        
        project = await project_manager.create_project(config)
        
        # Verify project structure
        assert project.path.exists()
        assert (project.path / "src").exists()
        assert (project.path / "tests").exists()
        assert (project.path / "README.md").exists()
        assert (project.path / "requirements.txt").exists()
        
        # Verify file contents
        readme = (project.path / "README.md").read_text()
        assert "test-python-project" in readme
        assert "Test Author" in readme
```

### End-to-End Testing

#### 1. CLI End-to-End Tests

```python
# tests/e2e/test_cli/test_project_creation_cli.py
import pytest
import subprocess
import json
from pathlib import Path

class TestCLIEndToEnd:
    """End-to-end tests for CLI interface."""
    
    def test_create_project_command(self, temp_dir):
        """Test project creation via CLI."""
        result = subprocess.run([
            "python", "-m", "claude_tiu.cli",
            "create-project",
            "--name", "cli-test-project",
            "--template", "python-cli",
            "--output", str(temp_dir),
            "--no-interactive"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Project created successfully" in result.stdout
        
        project_dir = temp_dir / "cli-test-project"
        assert project_dir.exists()
        
    def test_validate_project_command(self, temp_dir, sample_project):
        """Test project validation via CLI."""
        result = subprocess.run([
            "python", "-m", "claude_tiu.cli",
            "validate",
            str(sample_project.path),
            "--format", "json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        output = json.loads(result.stdout)
        assert "validation_results" in output
        assert "quality_score" in output
```

#### 2. TUI End-to-End Tests

```python
# tests/e2e/test_tui/test_project_wizard.py
import pytest
from textual.pilot import Pilot

from claude_tiu.tui.app import ClaudeTIUApp

class TestTUIEndToEnd:
    """End-to-end tests for TUI interface."""
    
    async def test_project_wizard_flow(self):
        """Test complete project wizard flow."""
        app = ClaudeTIUApp()
        
        async with app.run_test() as pilot:
            # Start wizard
            await pilot.press("n")  # New project
            
            # Fill in project details
            await pilot.type("test-project")
            await pilot.press("tab")  # Move to template selection
            await pilot.press("enter")  # Select template
            
            # Continue through wizard steps
            await pilot.press("tab", "enter")  # Framework selection
            await pilot.press("tab", "enter")  # Features selection
            
            # Start generation
            await pilot.press("tab", "enter")  # Generate button
            
            # Wait for completion (mock will complete instantly)
            await pilot.pause()
            
            # Verify success screen
            assert "Project created successfully" in pilot.app.screen.render()
```

### Performance Testing

#### 1. Generation Speed Tests

```python
# tests/performance/test_generation_speed.py
import pytest
import time
from pathlib import Path

from claude_tiu.core.project_manager import ProjectManager

class TestGenerationPerformance:
    """Performance benchmarks for project generation."""
    
    @pytest.mark.slow
    async def test_project_generation_speed(self, project_manager, temp_dir):
        """Test project generation speed benchmark."""
        start_time = time.time()
        
        config = ProjectConfig(
            name="performance-test",
            template="python-cli",
            output_dir=temp_dir
        )
        
        project = await project_manager.create_project(config)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Assert reasonable generation time (adjust based on expectations)
        assert generation_time < 30.0  # Should complete within 30 seconds
        
        # Verify project was created correctly
        assert project.path.exists()
        
        print(f"Project generation took {generation_time:.2f} seconds")
```

#### 2. Validation Performance Tests

```python
# tests/performance/test_validation_speed.py
import pytest
import time
from pathlib import Path

from claude_tiu.core.validation_engine import ValidationEngine

class TestValidationPerformance:
    """Performance benchmarks for validation."""
    
    @pytest.fixture
    def large_project(self, temp_dir):
        """Create large project for performance testing."""
        # Create multiple files with various content
        for i in range(100):
            file_path = temp_dir / f"module_{i}.py"
            file_path.write_text(f"""
def function_{i}():
    '''Function {i} implementation.'''
    result = {i} * 2
    return result

class Class_{i}:
    '''Class {i} implementation.'''
    
    def __init__(self):
        self.value = {i}
        
    def method(self):
        return self.value * 2
            """)
        return temp_dir
    
    @pytest.mark.slow
    async def test_large_project_validation_speed(self, large_project):
        """Test validation speed on large project."""
        validator = ValidationEngine()
        
        start_time = time.time()
        
        result = await validator.analyze_project_progress(large_project)
        
        end_time = time.time()
        validation_time = end_time - start_time
        
        # Assert reasonable validation time
        assert validation_time < 10.0  # Should complete within 10 seconds
        
        # Verify validation completed
        assert result.quality_score >= 0
        
        print(f"Validation took {validation_time:.2f} seconds for 100 files")
```

### Test Utilities and Helpers

#### 1. Test Utilities

```python
# tests/utils.py
import asyncio
from pathlib import Path
from typing import Any, Dict
import tempfile
import shutil

class TestProjectBuilder:
    """Utility for building test projects."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        
    def create_python_file(self, path: str, content: str) -> Path:
        """Create Python file with content."""
        file_path = self.temp_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
        
    def create_config_file(self, config: Dict[str, Any]) -> Path:
        """Create configuration file."""
        import yaml
        config_path = self.temp_dir / "config.yaml"
        config_path.write_text(yaml.dump(config))
        return config_path

async def wait_for_condition(condition, timeout: float = 5.0):
    """Wait for condition to become true with timeout."""
    start_time = asyncio.get_event_loop().time()
    
    while True:
        if condition():
            return True
            
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(f"Condition not met within {timeout} seconds")
            
        await asyncio.sleep(0.1)
```

#### 2. Mock Helpers

```python
# tests/mocks.py
from unittest.mock import Mock, AsyncMock
from claude_tiu.models.ai_request import AIResponse

class MockAIInterface:
    """Mock AI interface for testing."""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
        
    async def generate_code(self, request) -> AIResponse:
        """Mock code generation."""
        self.call_count += 1
        
        # Return predefined response if available
        if request.prompt in self.responses:
            return AIResponse(
                code=self.responses[request.prompt],
                status="success"
            )
        
        # Default response
        return AIResponse(
            code="# Generated code",
            status="success"
        )

def create_mock_project(name: str = "test-project") -> Mock:
    """Create mock project for testing."""
    project = Mock()
    project.name = name
    project.path = Path(f"/tmp/{name}")
    project.template = "test-template"
    return project
```

---

## Debugging Techniques

### Debugging Setup

#### 1. Debug Configuration

```python
# src/claude_tiu/utils/logging_utils.py
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_debug_logging(
    level: str = "DEBUG",
    log_file: Optional[Path] = None,
    include_ai_interactions: bool = True
) -> None:
    """Setup comprehensive debug logging."""
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file or Path.home() / ".claude-tiu" / "debug.log")
        ]
    )
    
    # Configure specific loggers
    loggers = [
        "claude_tiu.core",
        "claude_tiu.tui",
        "claude_tiu.cli",
        "claude_tiu.integrations"
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    if include_ai_interactions:
        # Special handler for AI interactions
        ai_logger = logging.getLogger("claude_tiu.ai_interactions")
        ai_handler = logging.FileHandler(
            log_file.parent / "ai_interactions.log" if log_file 
            else Path.home() / ".claude-tiu" / "ai_interactions.log"
        )
        ai_handler.setFormatter(
            logging.Formatter('%(asctime)s - AI - %(levelname)s - %(message)s')
        )
        ai_logger.addHandler(ai_handler)
        ai_logger.setLevel(logging.DEBUG)
```

#### 2. Debug Environment Variables

```bash
# .env.debug
CLAUDE_TIU_DEBUG=1
CLAUDE_TIU_LOG_LEVEL=DEBUG
CLAUDE_TIU_TRACE_AI_CALLS=1
CLAUDE_TIU_VALIDATION_VERBOSE=1
CLAUDE_TIU_PROFILE_PERFORMANCE=1

# Python debugging
PYTHONPATH=./src
PYTHONDONTWRITEBYTECODE=1
PYTHONASYNCIODEBUG=1
```

#### 3. Debug CLI Options

```python
# src/claude_tiu/cli/commands.py
import click
from claude_tiu.utils.logging_utils import setup_debug_logging

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--trace-ai', is_flag=True, help='Trace AI interactions')
@click.pass_context
def main(ctx, debug, verbose, trace_ai):
    """Claude TIU main CLI."""
    if debug:
        setup_debug_logging(
            level="DEBUG",
            include_ai_interactions=trace_ai
        )
        ctx.obj = {"debug": True}
    elif verbose:
        setup_debug_logging(level="INFO")
```

### Debugging Core Components

#### 1. Project Manager Debugging

```python
# src/claude_tiu/core/project_manager.py
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ProjectManager:
    """Project manager with debug instrumentation."""
    
    async def create_project(self, config: ProjectConfig) -> Project:
        """Create project with debug logging."""
        logger.debug(f"Creating project: {config.name}")
        logger.debug(f"Template: {config.template}")
        logger.debug(f"Variables: {config.variables}")
        
        try:
            # Validate configuration
            self._validate_config(config)
            logger.debug("Configuration validated successfully")
            
            # Create project directory
            project_path = config.output_dir / config.name
            logger.debug(f"Creating project directory: {project_path}")
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Load template
            template = await self._load_template(config.template)
            logger.debug(f"Template loaded: {template.name} v{template.version}")
            
            # Generate project files
            logger.debug("Starting project generation")
            project = await self._generate_project(template, config)
            logger.debug(f"Project generated with {len(project.files)} files")
            
            return project
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}", exc_info=True)
            raise
    
    def _validate_config(self, config: ProjectConfig) -> None:
        """Validate configuration with debug info."""
        logger.debug("Validating project configuration")
        
        if not config.name:
            logger.error("Project name is empty")
            raise ValueError("Project name required")
            
        if not config.template:
            logger.error("Template not specified")
            raise ValueError("Template required")
            
        logger.debug("Configuration validation passed")
```

#### 2. AI Interface Debugging

```python
# src/claude_tiu/core/ai_interface.py
import logging
import json
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)
ai_logger = logging.getLogger("claude_tiu.ai_interactions")

class AIInterface:
    """AI interface with comprehensive debugging."""
    
    async def generate_code(self, request: AIRequest) -> AIResponse:
        """Generate code with detailed logging."""
        start_time = time.time()
        request_id = id(request)
        
        ai_logger.info(f"AI Request {request_id} started")
        ai_logger.debug(f"Prompt: {request.prompt}")
        ai_logger.debug(f"Context: {json.dumps(request.context, indent=2)}")
        
        try:
            # Prepare command
            cmd = self._build_command(request)
            logger.debug(f"Claude command: {' '.join(cmd)}")
            
            # Execute command
            logger.debug("Executing Claude Code command")
            result = await self._execute_command(cmd)
            
            # Process response
            response = self._process_response(result)
            
            end_time = time.time()
            duration = end_time - start_time
            
            ai_logger.info(f"AI Request {request_id} completed in {duration:.2f}s")
            ai_logger.debug(f"Response status: {response.status}")
            ai_logger.debug(f"Generated code length: {len(response.code)} chars")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Generated code:\n{response.code}")
            
            return response
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            ai_logger.error(f"AI Request {request_id} failed after {duration:.2f}s: {e}")
            logger.error(f"AI generation failed: {e}", exc_info=True)
            raise
    
    async def _execute_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute command with detailed logging."""
        logger.debug(f"Executing: {' '.join(cmd)}")
        
        try:
            result = await run_subprocess_async(cmd)
            
            logger.debug(f"Command exit code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Command stdout: {result.stdout[:500]}...")
            if result.stderr:
                logger.debug(f"Command stderr: {result.stderr}")
                
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
```

#### 3. Validation Engine Debugging

```python
# src/claude_tiu/core/validation_engine.py
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)
validation_logger = logging.getLogger("claude_tiu.validation")

class ValidationEngine:
    """Validation engine with debug instrumentation."""
    
    async def analyze_project_progress(self, project_path: Path) -> ProgressReport:
        """Analyze project with comprehensive debugging."""
        validation_logger.info(f"Starting validation of {project_path}")
        
        try:
            # Static analysis
            logger.debug("Running static code analysis")
            static_results = await self._static_analysis(project_path)
            validation_logger.debug(f"Static analysis found {len(static_results.issues)} issues")
            
            # Placeholder detection
            logger.debug("Detecting placeholders")
            placeholder_results = await self._detect_placeholders(project_path)
            validation_logger.debug(f"Found {placeholder_results.count} placeholders")
            
            for placeholder in placeholder_results.placeholders:
                validation_logger.debug(f"Placeholder in {placeholder.file}:{placeholder.line}: {placeholder.content}")
            
            # Functionality testing
            logger.debug("Testing functionality")
            functional_results = await self._test_functionality(project_path)
            validation_logger.debug(f"Functionality score: {functional_results.score}")
            
            # Generate report
            report = self._generate_report(static_results, placeholder_results, functional_results)
            
            validation_logger.info(f"Validation completed. Real progress: {report.real_progress}%")
            
            return report
            
        except Exception as e:
            validation_logger.error(f"Validation failed: {e}", exc_info=True)
            raise
    
    async def _detect_placeholders(self, project_path: Path) -> Dict[str, Any]:
        """Detect placeholders with detailed logging."""
        logger.debug(f"Scanning for placeholders in {project_path}")
        
        placeholders = []
        
        for file_path in project_path.rglob("*.py"):
            logger.debug(f"Checking file: {file_path}")
            
            try:
                content = file_path.read_text()
                file_placeholders = self._scan_file_for_placeholders(file_path, content)
                placeholders.extend(file_placeholders)
                
                if file_placeholders:
                    logger.debug(f"Found {len(file_placeholders)} placeholders in {file_path}")
                    
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        logger.debug(f"Total placeholders found: {len(placeholders)}")
        
        return {
            "count": len(placeholders),
            "placeholders": placeholders
        }
```

### Debugging Workflows

#### 1. Workflow Execution Debugging

```python
# src/claude_tiu/core/workflow_engine.py
import logging
from typing import Dict, Any
import asyncio

logger = logging.getLogger(__name__)
workflow_logger = logging.getLogger("claude_tiu.workflows")

class WorkflowEngine:
    """Workflow engine with debug instrumentation."""
    
    async def execute_workflow(self, workflow: Workflow, variables: Dict[str, Any] = None) -> Dict[str, TaskResult]:
        """Execute workflow with comprehensive debugging."""
        workflow_id = workflow.id
        workflow_logger.info(f"Starting workflow execution: {workflow.name} ({workflow_id})")
        
        try:
            results = {}
            
            for phase_idx, phase in enumerate(workflow.phases):
                phase_logger = workflow_logger.getChild(f"phase_{phase_idx}")
                phase_logger.info(f"Starting phase: {phase.name}")
                
                if phase.parallel:
                    phase_logger.debug("Executing tasks in parallel")
                    phase_results = await self._execute_phase_parallel(phase, variables, results)
                else:
                    phase_logger.debug("Executing tasks sequentially")
                    phase_results = await self._execute_phase_sequential(phase, variables, results)
                
                results.update(phase_results)
                
                # Check for failures
                failed_tasks = [task_id for task_id, result in phase_results.items() 
                              if result.status == TaskStatus.FAILED]
                
                if failed_tasks:
                    phase_logger.error(f"Phase failed. Failed tasks: {failed_tasks}")
                    if not phase.continue_on_error:
                        workflow_logger.error(f"Workflow {workflow_id} terminated due to phase failure")
                        break
                else:
                    phase_logger.info(f"Phase completed successfully")
            
            workflow_logger.info(f"Workflow {workflow_id} execution completed")
            return results
            
        except Exception as e:
            workflow_logger.error(f"Workflow {workflow_id} execution failed: {e}", exc_info=True)
            raise
    
    async def _execute_task(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        """Execute individual task with debugging."""
        task_logger = workflow_logger.getChild(f"task_{task.name}")
        task_logger.info(f"Starting task execution: {task.name}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare AI request
            ai_request = self._prepare_ai_request(task, context)
            task_logger.debug(f"AI request prepared. Prompt length: {len(ai_request.prompt)}")
            
            # Execute AI request
            task_logger.debug("Sending request to AI")
            ai_response = await self.ai.generate_code(ai_request)
            
            # Validate response
            task_logger.debug("Validating AI response")
            validation_result = await self._validate_task_result(task, ai_response)
            
            if validation_result.is_valid:
                task_logger.info(f"Task {task.name} completed successfully")
                status = TaskStatus.COMPLETED
            else:
                task_logger.warning(f"Task {task.name} validation failed: {validation_result.issues}")
                status = TaskStatus.FAILED
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            return TaskResult(
                task_id=task.name,
                status=status,
                outputs=task.outputs,
                execution_time=execution_time,
                validation_results=validation_result.to_dict()
            )
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            task_logger.error(f"Task {task.name} execution failed: {e}", exc_info=True)
            
            return TaskResult(
                task_id=task.name,
                status=TaskStatus.FAILED,
                outputs=[],
                execution_time=execution_time
            )
```

### Debugging TUI Components

#### 1. TUI Screen Debugging

```python
# src/claude_tiu/tui/screens/project_wizard.py
import logging
from textual.screen import Screen
from textual.widgets import Input, Select, Button

logger = logging.getLogger(__name__)

class ProjectWizardScreen(Screen):
    """Project wizard with debug logging."""
    
    def compose(self):
        logger.debug("Composing project wizard screen")
        yield Input(placeholder="Project name", id="project_name")
        yield Select(options=self._get_templates(), id="template_select")
        yield Button("Create Project", id="create_button")
    
    def on_input_changed(self, event):
        """Handle input changes with debugging."""
        logger.debug(f"Input changed: {event.input.id} = {event.value}")
        
        if event.input.id == "project_name":
            self._validate_project_name(event.value)
    
    def on_button_pressed(self, event):
        """Handle button press with debugging."""
        logger.debug(f"Button pressed: {event.button.id}")
        
        if event.button.id == "create_button":
            self._handle_create_project()
    
    def _handle_create_project(self):
        """Handle project creation with debugging."""
        logger.debug("Starting project creation from wizard")
        
        try:
            # Get form data
            project_name = self.query_one("#project_name").value
            template = self.query_one("#template_select").value
            
            logger.debug(f"Project name: {project_name}")
            logger.debug(f"Selected template: {template}")
            
            # Validate inputs
            if not project_name:
                logger.warning("Project name is empty")
                self.notify("Project name is required", severity="error")
                return
            
            if not template:
                logger.warning("Template not selected")
                self.notify("Please select a template", severity="error")
                return
            
            # Start project creation
            logger.info(f"Creating project: {project_name} with template: {template}")
            self.app.push_screen("project_creation", project_name=project_name, template=template)
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}", exc_info=True)
            self.notify(f"Failed to create project: {e}", severity="error")
```

#### 2. Widget State Debugging

```python
# src/claude_tiu/tui/widgets/progress_widget.py
import logging
from textual.widget import Widget
from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

class ProgressWidget(Widget):
    """Progress widget with state debugging."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.progress = Progress()
        self.tasks = {}
        logger.debug("ProgressWidget initialized")
    
    def add_task(self, name: str, total: int) -> TaskID:
        """Add task with debugging."""
        logger.debug(f"Adding progress task: {name} (total: {total})")
        task_id = self.progress.add_task(name, total=total)
        self.tasks[name] = task_id
        self.refresh()
        return task_id
    
    def update_task(self, name: str, completed: int, total: int = None):
        """Update task progress with debugging."""
        logger.debug(f"Updating task {name}: completed={completed}, total={total}")
        
        if name not in self.tasks:
            logger.warning(f"Task {name} not found in progress widget")
            return
        
        task_id = self.tasks[name]
        
        if total is not None:
            self.progress.update(task_id, completed=completed, total=total)
        else:
            self.progress.update(task_id, completed=completed)
        
        self.refresh()
        
        # Log milestone progress
        task = self.progress.tasks[task_id]
        percentage = (task.completed / task.total) * 100 if task.total > 0 else 0
        logger.debug(f"Task {name} progress: {percentage:.1f}%")
    
    def complete_task(self, name: str):
        """Complete task with debugging."""
        logger.debug(f"Completing task: {name}")
        
        if name in self.tasks:
            task_id = self.tasks[name]
            task = self.progress.tasks[task_id]
            self.progress.update(task_id, completed=task.total)
            self.refresh()
            logger.info(f"Task {name} completed")
        else:
            logger.warning(f"Cannot complete task {name}: not found")
```

### Advanced Debugging Techniques

#### 1. Performance Profiling

```python
# src/claude_tiu/utils/profiling.py
import cProfile
import pstats
import functools
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def profile_function(output_file: Path = None):
    """Decorator to profile function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not output_file:
                # Simple timing
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
                return result
            else:
                # Full profiling
                pr = cProfile.Profile()
                pr.enable()
                
                result = func(*args, **kwargs)
                
                pr.disable()
                pr.dump_stats(output_file)
                
                # Log summary
                stats = pstats.Stats(pr)
                stats.sort_stats('cumulative')
                logger.debug(f"Profile saved to {output_file}")
                
                return result
        return wrapper
    return decorator

@profile_function()
async def debug_ai_generation(request):
    """Debug version of AI generation with profiling."""
    # Your AI generation code here
    pass
```

#### 2. Memory Usage Tracking

```python
# src/claude_tiu/utils/memory_debug.py
import psutil
import logging
from typing import Dict, Any
import functools

logger = logging.getLogger(__name__)

class MemoryTracker:
    """Track memory usage during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = self.get_memory_usage()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": self.process.memory_percent()
        }
    
    def log_memory_diff(self, operation: str):
        """Log memory difference from baseline."""
        current = self.get_memory_usage()
        diff = {
            key: current[key] - self.baseline[key] 
            for key in current.keys()
        }
        
        logger.debug(f"Memory usage after {operation}: {current}")
        logger.debug(f"Memory diff from baseline: {diff}")

def track_memory(operation_name: str):
    """Decorator to track memory usage of function."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracker = MemoryTracker()
            
            try:
                result = await func(*args, **kwargs)
                tracker.log_memory_diff(f"{operation_name} (success)")
                return result
            except Exception as e:
                tracker.log_memory_diff(f"{operation_name} (error)")
                raise
                
        return wrapper
    return decorator
```

#### 3. State Inspection Tools

```python
# src/claude_tiu/utils/debug_tools.py
import json
import logging
from typing import Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class StateInspector:
    """Tool for inspecting application state during debugging."""
    
    @staticmethod
    def dump_object_state(obj: Any, name: str = "object") -> None:
        """Dump object state to logs."""
        try:
            state = {}
            
            # Get object attributes
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if not callable(attr_value):
                            state[attr_name] = str(attr_value)
                    except Exception:
                        state[attr_name] = "<unable to access>"
            
            logger.debug(f"State of {name}:")
            logger.debug(json.dumps(state, indent=2, default=str))
            
        except Exception as e:
            logger.error(f"Failed to dump state of {name}: {e}")
    
    @staticmethod
    def save_state_snapshot(obj: Any, file_path: Path) -> None:
        """Save object state to file."""
        try:
            state = StateInspector._serialize_object(obj)
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.debug(f"State snapshot saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")
    
    @staticmethod
    def _serialize_object(obj: Any, max_depth: int = 3) -> Dict[str, Any]:
        """Recursively serialize object state."""
        if max_depth <= 0:
            return str(obj)
        
        try:
            if hasattr(obj, '__dict__'):
                return {
                    key: StateInspector._serialize_object(value, max_depth - 1)
                    for key, value in obj.__dict__.items()
                    if not key.startswith('_')
                }
            else:
                return str(obj)
        except Exception:
            return "<unable to serialize>"

# Usage in code
def debug_project_state(project_manager):
    """Debug helper to inspect project manager state."""
    StateInspector.dump_object_state(project_manager, "ProjectManager")
    
    # Save snapshot for later analysis
    snapshot_path = Path.home() / ".claude-tiu" / "debug" / "project_manager_state.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    StateInspector.save_state_snapshot(project_manager, snapshot_path)
```

This comprehensive developer guide provides everything needed to understand, contribute to, and extend Claude TIU effectively.