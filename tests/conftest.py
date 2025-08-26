"""
Global pytest configuration and fixtures for claude-tui tests.

This module provides shared fixtures, configuration, and utilities
for the entire test suite.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
try:
    from faker import Faker
except ImportError:
    Faker = None
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
except ImportError:
    create_engine = None
    sessionmaker = None

# Import project modules (will be created)
# from claude_tui.core.project_manager import ProjectManager
# from claude_tui.core.ai_interface import ClaudeInterface
# from claude_tui.database.models import Base


# Configure faker for consistent test data if available
if Faker is not None:
    fake = Faker()
    Faker.seed(42)
else:
    fake = None


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with standard structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create standard directories
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "docs").mkdir()
    (project_dir / "config").mkdir()
    
    # Create basic files
    (project_dir / "README.md").write_text("# Test Project")
    (project_dir / "pyproject.toml").write_text("[build-system]\\nrequires = []")
    
    return project_dir


@pytest.fixture
def mock_ai_interface():
    """Mock AI interface for testing."""
    mock = Mock()
    mock.execute_claude_code = AsyncMock(return_value={
        "status": "success",
        "output": "Generated code successfully",
        "files_created": ["test.py"]
    })
    mock.execute_claude_flow = AsyncMock(return_value={
        "status": "completed",
        "tasks_completed": 3,
        "results": {"analysis": "Code looks good"}
    })
    mock.validate_project = Mock(return_value=True)
    mock.validate_code = AsyncMock(return_value={
        "authentic": True,
        "issues": []
    })
    return mock


@pytest.fixture
def mock_database_session():
    """Create mock database session for testing."""
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Import and create tables (when models exist)
    # Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def sample_project_data():
    """Generate sample project data for testing."""
    return {
        "name": fake.slug(),
        "description": fake.text(max_nb_chars=100),
        "template": "python",
        "author": fake.name(),
        "email": fake.email(),
        "version": "0.1.0"
    }


@pytest.fixture
def sample_task_data():
    """Generate sample task data for testing."""
    return {
        "name": fake.word(),
        "prompt": fake.sentence(),
        "type": "code_generation",
        "priority": "medium",
        "estimated_duration": 30
    }


@pytest.fixture
def sample_code_with_placeholders():
    """Sample code containing placeholders for validation testing."""
    return '''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    # TODO: implement actual sum logic
    pass

def multiply(x, y):
    """Multiply two numbers - complete implementation."""
    return x * y

def divide(a, b):
    """Divide two numbers."""
    raise NotImplementedError("Division not yet implemented")

def complex_calculation():
    """Complex calculation with placeholder."""
    # implement later
    console.log("test")  # JavaScript placeholder
    return 42
'''


@pytest.fixture
def sample_complete_code():
    """Sample code without placeholders for validation testing."""
    return '''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b

def multiply(x, y):
    """Multiply two numbers."""
    return x * y

def divide(a, b):
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class Calculator:
    """Simple calculator class."""
    
    @staticmethod
    def add(x, y):
        return x + y
    
    @staticmethod
    def subtract(x, y):
        return x - y
'''


@pytest.fixture
def mock_security_validator():
    """Mock security validator for testing."""
    validator = Mock()
    validator.is_safe = Mock(return_value=True)
    validator.is_safe_command = Mock(return_value=True)
    validator.scan_for_vulnerabilities = Mock(return_value=[])
    return validator


@pytest.fixture
def mock_code_sandbox():
    """Mock code sandbox for testing."""
    sandbox = Mock()
    sandbox.execute = Mock(return_value={
        "output": "Test output",
        "error": None,
        "execution_time": 0.1,
        "memory_usage": 1024
    })
    sandbox.is_safe_code = Mock(return_value=True)
    return sandbox


class AsyncContextManager:
    """Helper for async context manager testing."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, *args):
        pass


@pytest.fixture
def async_context_manager():
    """Factory for creating async context managers in tests."""
    return AsyncContextManager


# Performance testing utilities
@pytest.fixture
def performance_threshold():
    """Define performance thresholds for testing."""
    return {
        "max_execution_time": 1.0,  # seconds
        "max_memory_usage": 100 * 1024 * 1024,  # 100MB
        "min_throughput": 10  # operations per second
    }


# Test data factories
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_project(**kwargs):
        """Create test project data."""
        defaults = {
            "name": fake.slug(),
            "description": fake.text(max_nb_chars=100),
            "template": "python",
            "status": "initialized"
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def create_task(**kwargs):
        """Create test task data."""
        defaults = {
            "name": fake.word(),
            "prompt": fake.sentence(),
            "type": "code_generation",
            "status": "pending"
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def create_validation_result(**kwargs):
        """Create test validation result."""
        defaults = {
            "real_progress": 70,
            "fake_progress": 30,
            "placeholders": ["TODO in line 5"],
            "quality_score": 0.7,
            "auto_fix_available": True
        }
        defaults.update(kwargs)
        return defaults


@pytest.fixture
def test_factory():
    """Provide test data factory."""
    return TestDataFactory


# Cleanup utilities
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after tests."""
    temp_files = []
    
    def register_temp_file(filepath):
        temp_files.append(filepath)
    
    yield register_temp_file
    
    # Cleanup
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass  # Ignore cleanup errors


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    # Set test environment
    monkeypatch.setenv("CLAUDE_TIU_ENV", "test")
    monkeypatch.setenv("CLAUDE_TIU_DEBUG", "true")
    
    # Mock external services
    monkeypatch.setenv("CLAUDE_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    
    # Disable external requests
    monkeypatch.setenv("CLAUDE_TIU_OFFLINE_MODE", "true")


# Async test utilities
@pytest_asyncio.fixture
async def async_test_client():
    """Create async test client for API testing."""
    # Will be implemented when we have an API
    pass


# Coverage utilities
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "external: mark test as requiring external services")
    
    # Configure coverage if running
    if config.getoption("--cov"):
        print("\\n📊 Running tests with coverage analysis...")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Mark slow tests based on name patterns
        if "performance" in item.name or "load" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("\\n🧪 Starting claude-tui test suite...")
    print("📋 Test Configuration:")
    print(f"   • Python version: {session.config.getoption('--tb')}")
    print(f"   • Async mode: enabled")
    print(f"   • Coverage: {'enabled' if session.config.getoption('--cov') else 'disabled'}")


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    if exitstatus == 0:
        print("\\n✅ All tests passed successfully!")
    else:
        print(f"\\n❌ Tests finished with exit status: {exitstatus}")
    
    # Print coverage reminder if not running with coverage
    if not session.config.getoption("--cov"):
        print("💡 Run with --cov to see coverage report")


# Test result helpers
class TestResultHelper:
    """Helper class for analyzing test results."""
    
    @staticmethod
    def assert_success_response(response):
        """Assert that response indicates success."""
        assert response.get("status") in ["success", "completed"]
        assert "error" not in response or response["error"] is None
    
    @staticmethod
    def assert_validation_result(result, expected_real=None, expected_fake=None):
        """Assert validation result structure and optionally values."""
        required_keys = ["real_progress", "fake_progress", "placeholders", "quality_score"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        if expected_real is not None:
            assert result["real_progress"] == expected_real
        
        if expected_fake is not None:
            assert result["fake_progress"] == expected_fake
        
        assert 0 <= result["real_progress"] <= 100
        assert 0 <= result["fake_progress"] <= 100
        assert 0 <= result["quality_score"] <= 1.0


@pytest.fixture
def result_helper():
    """Provide test result helper."""
    return TestResultHelper