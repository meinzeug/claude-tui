# Contributing to Claude TUI

Thank you for your interest in contributing to Claude TUI! This guide will help you get started with contributing to our AI-powered development tool.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Release Process](#release-process)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

### Our Pledge

We are committed to making participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- Git
- 8GB+ RAM (recommended for development)

### Quick Start

```bash
# Fork the repository on GitHub
git clone https://github.com/your-username/claude-tui.git
cd claude-tui

# Set up development environment
./scripts/setup-dev.sh

# Run tests to ensure everything works
pytest tests/ -v
```

## 🛠️ Development Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Install Claude Flow
npm install -g claude-flow@alpha
```

### 2. Configuration

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration with your API keys
CLAUDE_API_KEY=your_claude_api_key_here
CLAUDE_FLOW_ENDPOINT=http://localhost:3000
DEBUG=true
```

### 3. Verify Setup

```bash
# Run basic tests
pytest tests/unit/ -v

# Start development server
python -m claude_tui --debug

# Verify Claude Flow integration
npx claude-flow@alpha hive-mind status
```

## 📝 Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes or new features
- **Documentation**: Improve or add documentation
- **Testing**: Add or improve test coverage
- **Performance**: Optimize performance or resource usage

### Before You Start

1. **Check existing issues** - Someone might already be working on it
2. **Create an issue** - Discuss your idea before implementing
3. **Fork the repository** - Work on your own fork
4. **Create a feature branch** - Keep changes organized

### Coding Standards

#### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use type hints for all functions
def process_task(task: DevelopmentTask) -> TaskResult:
    """Process a development task with AI assistance."""
    
# Use descriptive variable names
user_authentication_config = AuthConfig(...)

# Keep functions focused and small (< 50 lines)
async def validate_user_input(input_data: str) -> ValidationResult:
    """Validate user input for security and format."""
    # Implementation...

# Use dataclasses/Pydantic for data structures
@dataclass
class ProjectMetrics:
    lines_of_code: int
    test_coverage: float
    complexity_score: float
```

#### Code Quality Tools

We use the following tools to maintain code quality:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
pylint src/

# Security scanning
bandit -r src/
```

#### Documentation Standards

```python
def create_project(self, config: ProjectConfig) -> Project:
    """
    Create a new project with AI assistance.
    
    This method orchestrates the complete project creation process,
    including template selection, dependency resolution, and initial
    code generation with validation.
    
    Args:
        config: Project configuration containing name, template,
               features, and AI behavior settings.
               
    Returns:
        Project instance representing the created project with
        all necessary metadata and file structure.
        
    Raises:
        ProjectCreationError: If project creation fails due to
                            template issues or validation failures.
        ValidationError: If configuration validation fails.
        
    Example:
        >>> config = ProjectConfig(name="my-app", template="react")
        >>> project = await tiu.create_project(config)
        >>> print(f"Created: {project.path}")
    """
```

### Commit Message Format

We use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(ai): add Claude Flow orchestration support

Add integration with Claude Flow for complex multi-agent 
development workflows.

- Implement ClaudeFlowClient for API communication
- Add workflow orchestration engine
- Include agent spawning and coordination
- Add comprehensive test coverage

Closes #123
```

## 🔄 Pull Request Process

### 1. Prepare Your Changes

```bash
# Create feature branch
git checkout -b feat/add-new-feature

# Make your changes
# ... code changes ...

# Add tests
# ... test files ...

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
isort src/ tests/
```

### 2. Submit Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feat/add-new-feature
   ```

2. **Create PR** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/demos if applicable
   - Checklist of changes

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No merge conflicts
   ```

### 3. Review Process

- **Automated checks** must pass
- **Code review** by maintainers
- **Testing** on different platforms
- **Documentation** review if applicable

### 4. After Approval

- Squash commits if requested
- Ensure CI passes
- Maintainer will merge

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_project.py
│   ├── test_ai_interface.py
│   └── test_validation.py
├── integration/          # Integration tests
│   ├── test_claude_code.py
│   ├── test_claude_flow.py
│   └── test_workflows.py
├── e2e/                 # End-to-end tests
│   ├── test_project_creation.py
│   └── test_full_workflows.py
└── fixtures/            # Test fixtures
    ├── projects/
    └── templates/
```

### Writing Tests

```python
import pytest
from claude_tui import ClaudeTIU, ProjectConfig

class TestProjectCreation:
    """Test project creation functionality."""
    
    @pytest.fixture
    async def tiu_client(self):
        """Create test TIU client."""
        return ClaudeTIU(debug=True)
    
    @pytest.fixture
    def sample_config(self):
        """Sample project configuration."""
        return ProjectConfig(
            name="test-project",
            template="python-basic",
            features=["testing"]
        )
    
    async def test_create_python_project(self, tiu_client, sample_config, tmp_path):
        """Test creating a Python project."""
        # Arrange
        sample_config.path = tmp_path
        
        # Act
        project = await tiu_client.create_project(sample_config)
        
        # Assert
        assert project.path.exists()
        assert (project.path / "src").exists()
        assert (project.path / "tests").exists()
        assert (project.path / "requirements.txt").exists()
    
    async def test_project_validation(self, tiu_client, sample_config, tmp_path):
        """Test project validation after creation."""
        # Arrange
        sample_config.path = tmp_path
        project = await tiu_client.create_project(sample_config)
        
        # Act
        validation_report = await project.validate_codebase()
        
        # Assert
        assert validation_report.is_valid
        assert validation_report.authenticity_score > 0.8
        assert len(validation_report.issues) == 0
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/performance/ --benchmark-only

# Run tests in parallel
pytest tests/ -n auto
```

### Mocking AI Services

```python
from unittest.mock import AsyncMock, patch
import pytest

@pytest.fixture
def mock_claude_client():
    """Mock Claude Code client."""
    with patch('claude_tui.ai.ClaudeCodeClient') as mock:
        mock.generate_code.return_value = AsyncMock(return_value={
            'code': 'def hello(): return "world"',
            'language': 'python',
            'quality_score': 0.95
        })
        yield mock

async def test_code_generation_with_mock(mock_claude_client):
    """Test code generation using mocked AI service."""
    # Test implementation using mock...
```

## 📚 Documentation

### Documentation Types

1. **Code Documentation** - Docstrings and type hints
2. **API Documentation** - Complete API reference
3. **User Guides** - How-to guides and tutorials
4. **Developer Guides** - Contributing and development
5. **Architecture** - System design and decisions

### Writing Documentation

```markdown
# Feature Documentation Template

## Overview
Brief description of the feature.

## Usage
Basic usage examples with code snippets.

## Configuration
Configuration options and parameters.

## Examples
Comprehensive examples showing different use cases.

## API Reference
Detailed API documentation with parameters and return values.

## Troubleshooting
Common issues and solutions.
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation locally
cd docs/
make html

# Serve documentation
python -m http.server 8000 --directory _build/html
```

## 🚢 Release Process

### Version Management

We use semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Steps

1. **Prepare Release**:
   ```bash
   # Update version
   bump2version minor  # or major/patch
   
   # Update CHANGELOG.md
   # Update documentation
   ```

2. **Create Release PR**:
   - Update version numbers
   - Update changelog
   - Update documentation

3. **Tag Release**:
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

4. **GitHub Release**:
   - Create GitHub release
   - Add release notes
   - Attach binaries if applicable

## 🆘 Getting Help

### Communication Channels

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and discussions
- **Developer Chat** - Real-time developer communication

### Asking for Help

When asking for help:

1. **Search existing issues** first
2. **Provide context** - What are you trying to achieve?
3. **Include details** - Error messages, code snippets, environment
4. **Be specific** - Vague questions get vague answers

### Example Good Issue

```markdown
**Title**: Claude Flow integration fails with timeout error

**Environment**: 
- OS: Ubuntu 20.04
- Python: 3.9.7  
- Claude TUI: v1.2.0

**Steps to reproduce**:
1. Install Claude Flow: `npm install -g claude-flow@alpha`
2. Initialize swarm: `npx claude-flow@alpha swarm init mesh`
3. Run: `claude-tui create --template react`

**Expected behavior**:
Project should be created successfully

**Actual behavior**:
TimeoutError after 30 seconds with message: "Failed to connect to Claude Flow service"

**Error logs**:
```
[ERROR] Connection timeout: http://localhost:3000/api/swarm/init
[ERROR] Retrying connection (attempt 2/3)...
```

**Additional context**:
Claude Flow service is running and accessible via curl
```

## 🏆 Recognition

Contributors are recognized in:
- **README.md** - Contributor list
- **CHANGELOG.md** - Release contributor credits
- **GitHub Contributors** - Automatic GitHub recognition
- **Annual Reports** - Major contributor highlights

Thank you for contributing to Claude TUI! Your efforts help make AI-powered development accessible to everyone.