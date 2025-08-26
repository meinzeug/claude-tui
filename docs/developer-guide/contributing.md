# Contributing to Claude-TUI

Welcome to the Claude-TUI contributor community! We're excited to have you join us in building the future of AI-assisted development. This guide will help you get started with contributing to the world's first **Intelligent Development Brain**.

## 🌟 Ways to Contribute

### Code Contributions
- **Core Features**: Enhance the intelligence brain functionality
- **AI Agents**: Create new specialized AI agents
- **Validation Engine**: Improve the anti-hallucination system
- **UI/UX**: Enhance the Textual interface
- **Performance**: Optimize system performance

### Non-Code Contributions  
- **Documentation**: Improve guides and references
- **Testing**: Add test cases and scenarios
- **Bug Reports**: Help identify and report issues
- **Feature Requests**: Suggest new capabilities
- **Community**: Help other users and contributors

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** with pip and virtual environments
- **Node.js 18+** for Claude Flow integration
- **Git** for version control
- **Docker** (optional) for containerized development
- **Claude API Key** for AI functionality testing

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/claude-tui.git
   cd claude-tui
   
   # Add upstream remote
   git remote add upstream https://github.com/original-org/claude-tui.git
   ```

2. **Create Virtual Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   # Install development dependencies
   pip install -e .
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   
   # Install Claude Flow (optional)
   npm install -g claude-flow@alpha
   ```

4. **Configuration Setup**
   ```bash
   # Copy example configuration
   cp .env.example .env
   
   # Edit .env with your API keys
   nano .env
   ```

5. **Verify Setup**
   ```bash
   # Run tests to verify setup
   pytest
   
   # Run the application
   claude-tui --help
   
   # Run health check
   claude-tui health-check
   ```

### Project Structure Understanding

```
claude-tui/
├── src/                          # Source code
│   ├── claude_tui/              # Main package
│   │   ├── core/                # Core functionality
│   │   │   ├── project_manager.py  # Central orchestrator
│   │   │   ├── task_engine.py      # Workflow management
│   │   │   └── ai_interface.py     # AI integration
│   │   ├── ui/                  # User interface
│   │   │   ├── main_app.py         # Main TUI application
│   │   │   ├── screens/            # UI screens
│   │   │   └── widgets/            # UI components
│   │   ├── validation/          # Anti-hallucination engine
│   │   ├── integrations/        # External integrations
│   │   └── utils/               # Utility functions
│   └── api/                     # REST API
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   └── fixtures/                # Test data
├── docs/                        # Documentation
├── examples/                    # Example projects
└── scripts/                     # Development scripts
```

## 📝 Development Workflow

### Branch Strategy

We use the **GitHub Flow** branching model:

- `main`: Production-ready code
- `develop`: Integration branch for features  
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical production fixes

### Making Changes

1. **Create a Feature Branch**
   ```bash
   # Update main branch
   git checkout main
   git pull upstream main
   
   # Create feature branch
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   ```bash
   # Make your code changes
   # Follow the coding standards (see below)
   
   # Run tests frequently
   pytest tests/unit/
   
   # Run linting
   black src/ tests/
   isort src/ tests/
   mypy src/
   ```

3. **Commit Your Changes**
   ```bash
   # Add changes
   git add .
   
   # Commit with descriptive message
   git commit -m "feat: add intelligent cache management
   
   - Implement ML-powered cache invalidation
   - Add cache performance metrics
   - Optimize memory usage for large projects
   
   Closes #123"
   ```

4. **Push and Create Pull Request**
   ```bash
   # Push to your fork
   git push origin feature/your-feature-name
   
   # Create pull request on GitHub
   # Follow the PR template
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(ai): add neural response caching system
fix(validation): resolve placeholder detection edge case
docs(api): update authentication endpoint documentation
perf(core): optimize task scheduler memory usage
```

## 🎯 Coding Standards

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# Good example
class IntelligentProjectManager:
    """
    Central orchestrator for project management with AI capabilities.
    
    This class coordinates all aspects of project lifecycle including
    creation, AI agent coordination, and validation workflows.
    """
    
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.ai_interface = AIInterface()
        self.validator = AntiHallucinationEngine()
        
    async def create_project_with_ai(
        self, 
        requirements: ProjectRequirements,
        template: Optional[ProjectTemplate] = None
    ) -> ProjectCreationResult:
        """
        Create a new project with AI assistance and validation.
        
        Args:
            requirements: Project requirements and specifications
            template: Optional project template to use as base
            
        Returns:
            ProjectCreationResult with success status and project details
            
        Raises:
            ProjectCreationException: If project creation fails
            ValidationException: If AI output validation fails
        """
        try:
            # Validate requirements
            validated_requirements = await self._validate_requirements(requirements)
            
            # Generate project structure with AI
            project_structure = await self.ai_interface.generate_project_structure(
                validated_requirements, template
            )
            
            # Validate generated structure
            validation_result = await self.validator.validate_project_structure(
                project_structure
            )
            
            if not validation_result.is_valid:
                raise ValidationException(
                    f"Generated project structure failed validation: "
                    f"{validation_result.errors}"
                )
            
            # Create project files
            project = await self._create_project_files(
                project_structure, validated_requirements
            )
            
            return ProjectCreationResult(
                success=True,
                project=project,
                validation_score=validation_result.score
            )
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            raise ProjectCreationException(f"Failed to create project: {e}") from e
```

### Code Quality Standards

- **Line Length**: Maximum 88 characters (Black formatter default)
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Google-style docstrings for all public methods
- **Error Handling**: Explicit exception handling with meaningful messages
- **Logging**: Use structured logging with appropriate levels
- **Async/Await**: Prefer async/await for I/O operations

### Testing Standards

```python
import pytest
from unittest.mock import AsyncMock, patch
from claude_tui.core.project_manager import ProjectManager
from claude_tui.models.project import ProjectRequirements

class TestProjectManager:
    """Test suite for ProjectManager functionality."""
    
    @pytest.fixture
    async def project_manager(self):
        """Create ProjectManager instance with mocked dependencies."""
        manager = ProjectManager()
        manager.ai_interface = AsyncMock()
        manager.validator = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    async def test_create_project_success(self, project_manager):
        """Test successful project creation with AI assistance."""
        # Arrange
        requirements = ProjectRequirements(
            name="test-project",
            type="web_service",
            language="python",
            framework="fastapi"
        )
        
        project_manager.ai_interface.generate_project_structure.return_value = {
            "files": ["main.py", "requirements.txt"],
            "structure": {"src": ["app.py"], "tests": ["test_app.py"]}
        }
        
        project_manager.validator.validate_project_structure.return_value = {
            "is_valid": True,
            "score": 0.98
        }
        
        # Act
        result = await project_manager.create_project_with_ai(requirements)
        
        # Assert
        assert result.success
        assert result.validation_score > 0.95
        project_manager.ai_interface.generate_project_structure.assert_called_once()
        project_manager.validator.validate_project_structure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_project_validation_failure(self, project_manager):
        """Test project creation with validation failure."""
        # Arrange
        requirements = ProjectRequirements(name="invalid-project")
        
        project_manager.ai_interface.generate_project_structure.return_value = {
            "files": ["main.py"]
        }
        
        project_manager.validator.validate_project_structure.return_value = {
            "is_valid": False,
            "errors": ["Missing requirements.txt", "No test structure"]
        }
        
        # Act & Assert
        with pytest.raises(ValidationException) as exc_info:
            await project_manager.create_project_with_ai(requirements)
        
        assert "validation" in str(exc_info.value).lower()
```

## 🧠 AI Agent Development

### Creating Custom Agents

To create a new AI agent, follow this pattern:

```python
from claude_tui.ai.base_agent import BaseAgent
from claude_tui.ai.agent_capabilities import AgentCapability

class DatabaseOptimizationAgent(BaseAgent):
    """Specialized agent for database optimization tasks."""
    
    def __init__(self):
        super().__init__(
            agent_id="db-optimizer",
            name="Database Optimization Specialist", 
            description="Optimizes database schemas, queries, and performance",
            capabilities=[
                AgentCapability.DATABASE_DESIGN,
                AgentCapability.QUERY_OPTIMIZATION, 
                AgentCapability.PERFORMANCE_ANALYSIS,
                AgentCapability.INDEX_MANAGEMENT
            ],
            specializations=["postgresql", "mysql", "sqlite", "mongodb"]
        )
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute database optimization task."""
        
        # Log task start
        self.logger.info(f"Starting database optimization task: {task.type}")
        
        # Build context
        context = await self._build_database_context(task)
        
        # Route to specific handler
        if task.type == TaskType.SCHEMA_OPTIMIZATION:
            result = await self._optimize_schema(task, context)
        elif task.type == TaskType.QUERY_OPTIMIZATION:
            result = await self._optimize_queries(task, context)
        elif task.type == TaskType.INDEX_ANALYSIS:
            result = await self._analyze_indexes(task, context)
        else:
            result = await super().execute_task(task)
        
        # Validate result
        validated_result = await self._validate_optimization_result(result, task)
        
        return validated_result
    
    async def _optimize_schema(self, task: AgentTask, context: dict) -> AgentResult:
        """Optimize database schema based on usage patterns."""
        
        # Analyze current schema
        schema_analysis = await self._analyze_current_schema(context)
        
        # Generate optimization recommendations
        optimization_prompt = self._create_schema_optimization_prompt(
            schema_analysis, task.requirements
        )
        
        # Get AI recommendations
        ai_result = await self.ai_interface.generate_optimization_plan(
            optimization_prompt, context
        )
        
        # Validate recommendations
        validation = await self._validate_schema_changes(ai_result.recommendations)
        
        return AgentResult(
            success=validation.is_safe,
            output=ai_result.recommendations,
            metadata={
                "optimization_score": validation.improvement_score,
                "safety_score": validation.safety_score,
                "estimated_improvement": validation.estimated_improvement
            },
            recommendations=ai_result.additional_recommendations
        )
```

### Agent Registration

Register new agents in the agent registry:

```python
# src/claude_tui/ai/agent_registry.py

from .database_optimization_agent import DatabaseOptimizationAgent

AVAILABLE_AGENTS = {
    # Existing agents...
    "db-optimizer": DatabaseOptimizationAgent,
}
```

## 🔬 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **AI Tests**: Test AI agent behavior with mocked responses
5. **Performance Tests**: Test system performance and scalability

### Writing Good Tests

```python
# Good test example
class TestAntiHallucinationEngine:
    """Comprehensive tests for the anti-hallucination engine."""
    
    @pytest.fixture
    def validation_engine(self):
        """Create validation engine with test configuration."""
        engine = AntiHallucinationEngine()
        engine.config.precision_threshold = 0.95
        return engine
    
    @pytest.mark.parametrize("code_sample,expected_issues", [
        ("def hello(): pass  # TODO: implement", 1),
        ("def complete_function(): return 'done'", 0),
        ("class MyClass: pass  # FIXME: add methods", 1),
    ])
    async def test_placeholder_detection(
        self, 
        validation_engine, 
        code_sample, 
        expected_issues
    ):
        """Test placeholder detection with various code samples."""
        
        context = ValidationContext(language="python", auto_fix=False)
        result = await validation_engine.validate_code_authenticity(
            code_sample, context
        )
        
        placeholder_issues = [
            issue for issue in result.issues 
            if issue.type == ValidationIssueType.PLACEHOLDER
        ]
        
        assert len(placeholder_issues) == expected_issues
        
        if expected_issues > 0:
            assert result.authenticity_score < 0.95
        else:
            assert result.authenticity_score >= 0.95
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src/claude_tui --cov-report=html

# Run performance tests
pytest tests/performance/ --benchmark-only

# Run tests in parallel
pytest -n auto
```

## 📚 Documentation Guidelines

### Documentation Types

1. **API Documentation**: Docstrings in code
2. **User Guides**: Step-by-step instructions
3. **Developer Guides**: Technical implementation details
4. **Reference Documentation**: Comprehensive API reference
5. **Tutorials**: Learning-oriented guides

### Writing Documentation

```markdown
# Component Name

Brief description of what this component does and why it's important.

## Overview

Detailed explanation of the component's purpose, functionality, and how it fits into the overall system.

## Usage

### Basic Usage

```python
# Simple example
from claude_tui.component import Component

component = Component()
result = await component.do_something()
```

### Advanced Usage

```python
# Complex example with configuration
component = Component(config={
    "option1": "value1",
    "option2": True
})

async with component:
    result = await component.advanced_operation(
        param1="value", 
        param2=123
    )
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `option1` | str | `"default"` | Description of option1 |
| `option2` | bool | `False` | Description of option2 |

## Examples

### Example 1: Common Use Case
[Detailed example with explanation]

### Example 2: Advanced Use Case  
[Complex example with step-by-step breakdown]

## API Reference

### Class: `Component`

Main class for component functionality.

#### Methods

##### `do_something(param: str) -> Result`

Description of what this method does.

**Parameters:**
- `param` (str): Description of parameter

**Returns:**
- `Result`: Description of return value

**Raises:**
- `ComponentException`: When something goes wrong

## See Also

- [Related Component](../related-component.md)
- [User Guide](../user-guide.md)
```

## 🐛 Bug Reports

### Before Reporting

1. **Search Existing Issues**: Check if the bug is already reported
2. **Reproduce the Bug**: Ensure the bug is reproducible
3. **Gather Information**: System info, logs, and steps to reproduce

### Bug Report Template

```markdown
## Bug Description

Clear and concise description of what the bug is.

## Steps to Reproduce

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior

Clear description of what you expected to happen.

## Actual Behavior

Clear description of what actually happened.

## Environment

- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python Version: [e.g., 3.11.5]
- Claude-TUI Version: [e.g., 1.2.3]
- Claude API: [e.g., claude-3-sonnet-20241022]

## Additional Context

Add any other context about the problem here.
Include logs, screenshots, or error messages if applicable.

## Logs

```
[Paste relevant logs here]
```
```

## ✨ Feature Requests

### Feature Request Template

```markdown
## Feature Description

Clear and concise description of what you want to happen.

## Problem Statement

Is your feature request related to a problem? Please describe.
A clear description of what the problem is.

## Proposed Solution

Describe the solution you'd like.

## Alternatives Considered

Describe any alternative solutions or features you've considered.

## Implementation Ideas

If you have ideas about how this could be implemented, share them here.

## Additional Context

Add any other context or screenshots about the feature request here.
```

## 🎭 Code Review Process

### For Contributors

When submitting a pull request:

1. **Self-Review**: Review your own code first
2. **Testing**: Ensure all tests pass
3. **Documentation**: Update relevant documentation
4. **Commit Messages**: Follow commit message format
5. **PR Description**: Use the PR template

### For Reviewers

When reviewing pull requests:

1. **Code Quality**: Check for clarity, maintainability, and best practices
2. **Testing**: Verify adequate test coverage
3. **Performance**: Consider performance implications
4. **Security**: Check for security issues
5. **Documentation**: Ensure documentation is updated

### Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are added/updated and pass
- [ ] Documentation is updated
- [ ] No obvious security issues
- [ ] Performance impact is considered
- [ ] Breaking changes are documented
- [ ] Commit messages follow conventions

## 🏆 Recognition

We appreciate all contributions! Contributors are recognized in:

- **README Contributors Section**: All contributors listed
- **Release Notes**: Major contributions highlighted  
- **Hall of Fame**: Top contributors featured
- **Contributor Badges**: GitHub profile badges
- **Annual Report**: Contributor achievements

## 📞 Getting Help

### Community Support

- **GitHub Discussions**: General questions and community help
- **Discord**: Real-time chat with maintainers and community
- **Stack Overflow**: Tag questions with `claude-tui`

### Maintainer Contact

- **GitHub Issues**: Bug reports and feature requests
- **Email**: maintainers@claude-tui.dev
- **Office Hours**: Weekly community office hours

### Development Support

- **Architecture Questions**: Post in GitHub Discussions
- **Code Review**: Request review from maintainers
- **Mentoring**: We provide mentoring for new contributors

## 🔄 Release Process

### Development Cycle

1. **Feature Development**: Features developed in feature branches
2. **Integration Testing**: Features integrated in develop branch
3. **Release Preparation**: Release branch created from develop
4. **Testing Phase**: Comprehensive testing and bug fixes
5. **Release**: Tagged release and deployment to PyPI
6. **Post-Release**: Monitor, patch, and prepare next cycle

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Schedule

- **Major Releases**: Quarterly (every 3 months)
- **Minor Releases**: Monthly
- **Patch Releases**: As needed for critical bugs

## 🙏 Thank You

Thank you for contributing to Claude-TUI! Your contributions help make AI-assisted development accessible to developers worldwide. Together, we're building the future of intelligent software development.

---

*Ready to contribute? Start by reading our [Architecture Guide](architecture-deep-dive.md) and then dive into the codebase!* 🚀