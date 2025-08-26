"""
claude-tui test suite.

This package contains all tests for the claude-tui project:
- Unit tests for individual components
- Integration tests for system interactions  
- UI tests for terminal interface components
- Validation tests for anti-hallucination features
- Performance and load tests
- Security and input validation tests

Test organization:
- tests/unit/ - Unit tests for core modules
- tests/integration/ - Integration tests for CLI and external services
- tests/ui/ - TUI component and screen tests
- tests/validation/ - Anti-hallucination and placeholder detection tests
- tests/performance/ - Performance, load, and scalability tests
- tests/security/ - Security, validation, and sandbox tests
- tests/fixtures/ - Shared test data and mock factories

Run all tests:
    pytest

Run with coverage:
    pytest --cov=src --cov-report=html --cov-report=term

Run specific test categories:
    pytest tests/unit/                    # Unit tests only
    pytest tests/integration/             # Integration tests only
    pytest -m "not performance"          # Skip performance tests
    pytest -m "security"                 # Security tests only
    pytest --slow                        # Include slow tests

Test markers:
- unit: Unit tests (fast, isolated)
- integration: Integration tests (slower, external dependencies)
- ui: TUI component tests
- performance: Performance and benchmarking tests
- security: Security validation tests
- slow: Tests that take longer to run
- benchmark: Benchmark tests using pytest-benchmark
"""

__version__ = "0.1.0"
__author__ = "Claude TUI Team"

# Test configuration constants
TEST_TIMEOUT = 30  # Default test timeout in seconds
PERFORMANCE_TIMEOUT = 300  # Performance test timeout in seconds
BENCHMARK_ROUNDS = 10  # Default benchmark rounds

# Test data directories
TEST_DATA_DIR = "tests/fixtures/data"
TEST_TEMP_DIR = "/tmp/claude-tui-tests"

# Coverage requirements by module
COVERAGE_REQUIREMENTS = {
    "overall": 80,
    "core_modules": {
        "project_manager": 90,
        "ai_interface": 85,
        "task_engine": 90,
        "validators": 95,  # Critical for anti-hallucination
    },
    "ui_modules": {
        "widgets": 75,
        "screens": 70,
        "app": 80,
    },
    "integration": {
        "cli_integration": 85,
        "database": 90,
    },
    "security": {
        "sandbox": 100,  # Critical security component
        "input_validators": 95,
    }
}

# Test result categories for reporting
TEST_CATEGORIES = {
    "PASS": "✅",
    "FAIL": "❌", 
    "SKIP": "⏭️",
    "ERROR": "🚫",
    "BENCHMARK": "📊"
}