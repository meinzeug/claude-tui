"""
Global pytest configuration and fixtures for Claude-TIU Test Suite
Provides enhanced testing infrastructure for comprehensive coverage analysis.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, Optional, Generator, AsyncGenerator

import pytest
import pytest_asyncio

# Add src to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Environment setup for testing
os.environ["PYTEST_CURRENT_TEST"] = "true"
os.environ["CLAUDE_TIU_TESTING"] = "true"
os.environ["CLAUDE_TIU_LOG_LEVEL"] = "INFO"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root path."""
    return project_root


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        "claude": {
            "api_key": "test-api-key",
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 4096,
            "timeout": 30
        },
        "project": {
            "name": "test-project",
            "path": "/tmp/test-project",
            "language": "python"
        },
        "ui": {
            "theme": "dark",
            "auto_refresh": True,
            "refresh_interval": 1.0
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def mock_ai_client():
    """Provide a mock AI client for testing."""
    client = AsyncMock()
    
    # Mock standard responses
    client.send_message = AsyncMock(return_value={
        "content": [{"text": "Mock AI response"}],
        "usage": {"input_tokens": 10, "output_tokens": 20}
    })
    
    client.validate_code = AsyncMock(return_value={
        "is_valid": True,
        "confidence": 0.95,
        "issues": []
    })
    
    client.stream_response = AsyncMock()
    
    return client


@pytest.fixture
def mock_project_data():
    """Provide mock project data for testing."""
    return {
        "id": "test-project-123",
        "name": "Test Project",
        "description": "A test project for validation",
        "language": "python",
        "framework": "fastapi",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "settings": {
            "auto_validation": True,
            "code_style": "pep8",
            "test_framework": "pytest"
        },
        "files": [
            "main.py",
            "requirements.txt",
            "README.md"
        ]
    }


@pytest.fixture
def mock_task_data():
    """Provide mock task data for testing."""
    return {
        "id": "task-456",
        "project_id": "test-project-123",
        "title": "Implement user authentication",
        "description": "Add JWT-based authentication system",
        "status": "in_progress",
        "priority": "high",
        "assignee": "developer@example.com",
        "created_at": "2024-01-01T10:00:00Z",
        "due_date": "2024-01-07T00:00:00Z",
        "tags": ["backend", "security", "auth"],
        "metadata": {
            "estimated_hours": 8,
            "complexity": "medium",
            "dependencies": []
        }
    }


@pytest.fixture
async def mock_database_session():
    """Provide a mock database session."""
    session = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.scalar = AsyncMock()
    session.query = Mock()
    return session


class MockValidationEngine:
    """Mock validation engine for testing."""
    
    def __init__(self):
        self.accuracy = 0.958
        self.confidence_threshold = 0.85
        self.validation_count = 0
        
    async def validate_content(self, content: str, context: Optional[Dict] = None):
        """Mock content validation."""
        self.validation_count += 1
        return {
            "is_valid": True,
            "confidence": 0.92,
            "accuracy": self.accuracy,
            "issues": [],
            "suggestions": [],
            "metadata": {
                "validation_time": 0.1,
                "model_version": "test-v1.0",
                "context_used": bool(context)
            }
        }
        
    def get_accuracy_metrics(self):
        """Get accuracy metrics."""
        return {
            "current_accuracy": self.accuracy,
            "target_accuracy": 0.958,
            "validation_count": self.validation_count,
            "success_rate": 0.98
        }


@pytest.fixture
def mock_validation_engine():
    """Provide a mock validation engine."""
    return MockValidationEngine()


@pytest.fixture
def mock_file_system(temp_dir):
    """Create a mock file system for testing."""
    # Create test file structure
    test_files = {
        "main.py": "# Test main file\nprint('Hello, World!')",
        "config.json": '{"name": "test", "version": "1.0"}',
        "requirements.txt": "fastapi>=0.68.0\nuvicorn>=0.15.0",
        "src/utils.py": "def helper_function():\n    return 'helper'",
        "tests/test_main.py": "def test_main():\n    assert True",
        "README.md": "# Test Project\n\nThis is a test project."
    }
    
    for file_path, content in test_files.items():
        full_path = temp_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    return temp_dir


@pytest.fixture
def performance_threshold():
    """Define performance thresholds for benchmarking."""
    return {
        "config_load_time": 0.1,  # seconds
        "validation_time": 0.5,   # seconds
        "ui_render_time": 0.2,    # seconds
        "ai_response_time": 2.0,  # seconds
        "memory_usage_mb": 100,   # MB
        "cpu_usage_percent": 50   # %
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "CLAUDE_TIU_ENV": "test",
        "CLAUDE_TIU_LOG_LEVEL": "INFO",
        "CLAUDE_TIU_CONFIG_FILE": "",
        "CLAUDE_API_KEY": "test-api-key"
    }
    
    os.environ.update(test_env)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Pytest markers for test categorization
pytest_markers = [
    "unit: Unit tests for individual components",
    "integration: Integration tests for component interactions",
    "performance: Performance and benchmarking tests",
    "security: Security and vulnerability tests",
    "tui: Text User Interface tests",
    "validation: Anti-hallucination validation tests",
    "slow: Tests that take more than 5 seconds",
    "fast: Tests that complete in under 1 second",
    "edge_case: Edge case and error condition tests",
    "regression: Regression tests for known issues",
    "smoke: Basic functionality smoke tests",
    "critical: Critical path tests",
    "property_based: Property-based tests using Hypothesis",
    "anti_hallucination: Anti-hallucination testing",
    "benchmark: Benchmark tests",
    "hypothesis: Hypothesis-based property testing"
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify collected test items for better organization."""
    for item in items:
        # Auto-mark tests based on file location
        test_file = str(item.fspath)
        
        if "unit" in test_file:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_file:
            item.add_marker(pytest.mark.integration)
        elif "performance" in test_file:
            item.add_marker(pytest.mark.performance)
        elif "security" in test_file:
            item.add_marker(pytest.mark.security)
        elif "ui" in test_file or "tui" in test_file:
            item.add_marker(pytest.mark.tui)
        elif "validation" in test_file:
            item.add_marker(pytest.mark.validation)
        
        # Mark slow tests
        if hasattr(item, 'callspec') and 'slow' in str(item.callspec):
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that use hypothesis
        if "hypothesis" in test_file or "property" in test_file:
            item.add_marker(pytest.mark.hypothesis)


@pytest.fixture(scope="session")
def test_metrics():
    """Track test metrics across the session."""
    metrics = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "coverage_percentage": 0.0,
        "performance_benchmarks": [],
        "security_issues": [],
        "validation_accuracy": []
    }
    return metrics


# Helper functions for common test operations
def create_mock_response(content: str, status_code: int = 200, headers: Optional[Dict] = None):
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = status_code
    response.text = content
    response.json.return_value = {"content": content}
    response.headers = headers or {}
    return response


def assert_performance_within_threshold(actual_time: float, threshold: float, operation: str):
    """Assert that operation performance is within threshold."""
    assert actual_time <= threshold, (
        f"{operation} took {actual_time:.3f}s, "
        f"which exceeds threshold of {threshold:.3f}s"
    )


def assert_memory_usage_acceptable(memory_mb: float, max_memory_mb: float):
    """Assert that memory usage is within acceptable limits."""
    assert memory_mb <= max_memory_mb, (
        f"Memory usage {memory_mb:.2f}MB exceeds limit of {max_memory_mb:.2f}MB"
    )


# Test data generators for property-based testing
def generate_test_config():
    """Generate test configuration data."""
    return {
        "api_key": f"sk-test-{'x' * 40}",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 4096,
        "temperature": 0.7
    }


def generate_test_project():
    """Generate test project data."""
    return {
        "name": "test-project",
        "description": "A test project",
        "language": "python",
        "framework": "fastapi"
    }


# Async testing utilities
@pytest_asyncio.fixture
async def async_mock_client():
    """Provide an async mock client."""
    client = AsyncMock()
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock(return_value=True)
    client.send = AsyncMock(return_value={"status": "success"})
    return client


# Coverage tracking utilities
class CoverageTracker:
    """Track coverage metrics during test execution."""
    
    def __init__(self):
        self.modules_tested = set()
        self.functions_tested = set()
        self.lines_tested = set()
        
    def mark_module_tested(self, module_name: str):
        """Mark a module as tested."""
        self.modules_tested.add(module_name)
        
    def mark_function_tested(self, function_name: str):
        """Mark a function as tested."""
        self.functions_tested.add(function_name)
        
    def get_coverage_stats(self):
        """Get current coverage statistics."""
        return {
            "modules": len(self.modules_tested),
            "functions": len(self.functions_tested),
            "lines": len(self.lines_tested)
        }


@pytest.fixture(scope="session")
def coverage_tracker():
    """Provide a coverage tracker instance."""
    return CoverageTracker()


# Test result collectors
@pytest.fixture(autouse=True)
def collect_test_results(request, test_metrics):
    """Automatically collect test results."""
    test_metrics["tests_run"] += 1
    
    def finalize():
        if hasattr(request.node, 'rep_call'):
            if request.node.rep_call.passed:
                test_metrics["tests_passed"] += 1
            else:
                test_metrics["tests_failed"] += 1
    
    request.addfinalizer(finalize)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)