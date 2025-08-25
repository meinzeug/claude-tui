# Testing Implementation Guide - claude-tiu

## Overview

This guide provides comprehensive documentation for implementing and running tests in the claude-tiu project. The testing strategy follows industry best practices with a focus on anti-hallucination validation, security, and performance.

## Table of Contents

1. [Test Architecture](#test-architecture)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Coverage Requirements](#coverage-requirements)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Security Testing](#security-testing)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Test Architecture

### Test Pyramid Structure

```
         /\
        /E2E\      <- 10% (Complete workflows)
       /------\
      /Integr. \   <- 30% (Component interactions)
     /----------\
    /   Unit     \ <- 60% (Individual components)
   /--------------\
```

### Directory Structure

```
tests/
├── unit/                    # Unit tests (60%)
│   ├── test_core_comprehensive.py
│   ├── test_ai_interface.py
│   ├── test_project_manager.py
│   └── test_validation_engine.py
├── integration/             # Integration tests (30%)
│   ├── test_comprehensive_integration.py
│   ├── test_api_endpoints.py
│   └── test_service_integration.py
├── e2e/                     # End-to-end tests (10%)
│   ├── test_complete_workflows.py
│   └── test_user_journeys.py
├── performance/             # Performance tests
│   ├── test_performance_benchmarks.py
│   └── test_load_testing.py
├── security/                # Security tests
│   ├── test_security_comprehensive.py
│   └── test_vulnerability_assessment.py
├── validation/              # Anti-hallucination tests
│   ├── test_anti_hallucination.py
│   └── test_placeholder_detection.py
├── ui/                      # TUI tests
│   ├── test_tui_components.py
│   └── test_textual_widgets.py
├── fixtures/                # Shared test fixtures
│   ├── __init__.py
│   ├── project_fixtures.py
│   └── validation_fixtures.py
├── mocks/                   # Mock objects and services
│   ├── __init__.py
│   └── mock_dependencies.py
├── conftest.py             # Global pytest configuration
└── test_runner.py          # Enhanced test runner
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation

**Characteristics**:
- Fast execution (< 100ms per test)
- No external dependencies
- High coverage (90%+ for core modules)
- Extensive mocking

**Example**:
```python
def test_project_creation_success(self, project_manager, sample_project_data):
    """Test successful project creation."""
    # Arrange
    project_data = sample_project_data
    
    # Act
    result = project_manager.create_project(project_data)
    
    # Assert
    assert result["id"] is not None
    assert result["name"] == project_data["name"]
    assert result["status"] == "initialized"
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and API endpoints

**Characteristics**:
- Medium execution time (< 5 seconds per test)
- Database and external service mocking
- Focus on data flow and communication
- Real component integration

**Example**:
```python
@pytest.mark.asyncio
async def test_project_lifecycle_api(self, test_client):
    """Test complete project lifecycle through API."""
    # Create -> Read -> Update -> Delete
    create_result = await test_client.post("/api/projects", json=project_data)
    assert create_result.status == 201
    
    # ... additional lifecycle operations
```

### 3. End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete user workflows from start to finish

**Characteristics**:
- Slower execution (< 60 seconds per test)
- Full system integration
- Real user scenarios
- Minimal mocking

**Example**:
```python
@pytest.mark.asyncio
async def test_complete_project_creation_workflow(self, workflow_components):
    """Test complete project creation from TUI to file generation."""
    # Step 1: User initiates project creation
    # Step 2: Project Manager creates structure
    # Step 3: AI generates files
    # Step 4: Validation checks
    # Step 5: File system operations
```

### 4. Performance Tests (`tests/performance/`)

**Purpose**: Validate performance requirements and identify bottlenecks

**Characteristics**:
- Load testing
- Memory usage monitoring
- Response time benchmarking
- Throughput measurements

**Key Metrics**:
- Project creation: < 200ms
- Task execution: < 1000ms
- API response: < 500ms
- Concurrent users: 10+ simultaneous

### 5. Security Tests (`tests/security/`)

**Purpose**: Identify security vulnerabilities and validate protections

**Test Areas**:
- SQL injection prevention
- XSS protection
- Command injection prevention
- Path traversal protection
- Input validation
- Authentication security

### 6. Validation Tests (`tests/validation/`)

**Purpose**: Anti-hallucination validation and code quality assessment

**Key Features**:
- Placeholder detection (TODO, NotImplementedError, etc.)
- Real vs fake progress calculation
- AI cross-validation
- Code authenticity verification

## Running Tests

### Quick Test Commands

```bash
# Run all tests with coverage
python tests/test_runner.py all --coverage

# Run specific test category
python tests/test_runner.py unit --verbose
python tests/test_runner.py integration
python tests/test_runner.py e2e
python tests/test_runner.py performance
python tests/test_runner.py security

# Run with quality checks
python tests/test_runner.py all --lint --security

# Run smoke tests only
python tests/test_runner.py smoke
```

### Using pytest directly

```bash
# Unit tests with coverage
pytest tests/unit/ --cov=src --cov-report=html --cov-report=term-missing

# Integration tests
pytest tests/integration/ -v --timeout=300

# Performance tests
pytest tests/performance/ -m performance

# Security tests
pytest tests/security/ -v

# Run tests in parallel
pytest tests/ -n auto

# Run specific test file
pytest tests/unit/test_project_manager.py -v

# Run specific test method
pytest tests/unit/test_project_manager.py::TestProjectManager::test_create_project_success -v
```

### Test Markers

Use pytest markers to categorize and filter tests:

```bash
# Run only fast tests
pytest -m "not slow"

# Run security tests
pytest -m security

# Run performance tests
pytest -m performance

# Run smoke tests
pytest -m smoke

# Run critical tests only
pytest -m critical

# Exclude external dependency tests
pytest -m "not external"
```

## Writing Tests

### Test Structure (AAA Pattern)

Follow the Arrange-Act-Assert pattern:

```python
def test_function_name(self, fixtures):
    """Test description explaining what is being tested."""
    # Arrange - Set up test data and mocks
    input_data = {"key": "value"}
    expected_result = {"status": "success"}
    
    # Act - Execute the function being tested
    result = function_under_test(input_data)
    
    # Assert - Verify the results
    assert result == expected_result
    assert "key" in result
```

### Fixture Usage

Leverage pytest fixtures for reusable test setup:

```python
@pytest.fixture
def mock_ai_interface():
    """Mock AI interface for testing."""
    mock = Mock()
    mock.execute_claude_code = AsyncMock(return_value={"status": "success"})
    return mock

def test_with_fixture(self, mock_ai_interface):
    """Test using the fixture."""
    result = mock_ai_interface.execute_claude_code("test prompt")
    assert result["status"] == "success"
```

### Async Test Writing

For async functionality:

```python
@pytest.mark.asyncio
async def test_async_function(self):
    """Test async functionality."""
    result = await async_function()
    assert result is not None
```

### Parametrized Tests

For testing multiple scenarios:

```python
@pytest.mark.parametrize("input_value,expected", [
    ("valid_input", True),
    ("invalid_input", False),
    ("", False),
])
def test_validation(self, input_value, expected):
    """Test validation with various inputs."""
    result = validate_input(input_value)
    assert result == expected
```

## CI/CD Pipeline

### GitHub Actions Workflow

The comprehensive testing pipeline includes:

1. **Quality Gates** (10 minutes)
   - Code formatting (Black)
   - Import sorting (isort)
   - Linting (flake8)
   - Type checking (mypy)
   - Security linting (bandit)
   - Dependency vulnerabilities (safety)

2. **Unit Tests** (20 minutes)
   - Python 3.9, 3.10, 3.11 matrix
   - Coverage reporting
   - JUnit XML output

3. **Integration Tests** (30 minutes)
   - PostgreSQL and Redis services
   - Database migrations
   - API endpoint testing

4. **E2E Tests** (45 minutes)
   - Full workflow testing
   - Claude Flow integration
   - TUI testing

5. **Performance Tests** (30 minutes)
   - Load testing
   - Memory profiling
   - Benchmark comparisons

6. **Security Tests** (20 minutes)
   - Vulnerability scanning
   - OWASP ZAP baseline scan
   - Trivy filesystem scan

### Triggering Tests

Tests are triggered by:

- **Push** to main/develop/feature branches
- **Pull Requests** to main/develop
- **Schedule** (daily at 2 AM UTC)
- **Manual dispatch** with test suite selection

### Quality Gates

Before deployment, all tests must pass:

- ✅ Unit tests: 80%+ coverage
- ✅ Integration tests: All passing
- ✅ E2E tests: Critical workflows working
- ✅ Security tests: No critical vulnerabilities
- ✅ Performance tests: Within thresholds

## Coverage Requirements

### Overall Coverage Targets

- **Overall**: 80%+ code coverage
- **Core modules**: 90%+ coverage
- **Security modules**: 95%+ coverage
- **UI modules**: 70%+ coverage (TUI harder to test)

### Module-Specific Requirements

```yaml
coverage_requirements:
  core_modules:
    project_manager: 90%
    ai_interface: 85%
    task_engine: 90%
    validators: 95%  # Critical for anti-hallucination
  
  ui_modules:
    widgets: 75%
    screens: 70%
    app: 80%
  
  security:
    sandbox: 100%  # Critical security component
    validators: 95%
```

### Coverage Commands

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Generate XML coverage report
pytest --cov=src --cov-report=xml

# Coverage with missing lines
pytest --cov=src --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=80
```

## Performance Benchmarks

### Key Performance Indicators

| Operation | Target Time | Threshold |
|-----------|-------------|-----------|
| Project creation | < 200ms | < 500ms |
| Task execution | < 1000ms | < 2000ms |
| Code validation | < 100ms | < 300ms |
| API response | < 500ms | < 1000ms |

### Load Testing

```bash
# Run performance benchmarks
python tests/test_runner.py performance

# Memory profiling
python -m memory_profiler tests/performance/test_memory_usage.py

# CPU profiling
python -m cProfile -o profile.stats tests/performance/test_cpu_usage.py
```

### Benchmark Assertions

```python
def test_project_creation_performance(self):
    """Test project creation meets performance requirements."""
    start_time = time.perf_counter()
    
    project = project_manager.create_project(project_data)
    
    execution_time = time.perf_counter() - start_time
    assert execution_time < 0.2  # 200ms threshold
    assert project["id"] is not None
```

## Security Testing

### Security Test Categories

1. **Input Validation**
   - SQL injection prevention
   - XSS protection
   - Command injection prevention
   - Path traversal protection

2. **Authentication & Authorization**
   - Password security
   - Session management
   - Rate limiting
   - Brute force protection

3. **Data Protection**
   - Encryption at rest
   - Encryption in transit
   - Sensitive data handling
   - Secure random generation

4. **API Security**
   - Request validation
   - Rate limiting
   - CORS configuration
   - Security headers

### Security Testing Commands

```bash
# Run security tests
python tests/test_runner.py security

# Security linting
bandit -r src -f json -o security-report.json

# Dependency vulnerabilities
safety check --json --output safety-report.json

# OWASP ZAP scanning (requires running application)
docker run -v $(pwd):/zap/wrk/:rw -t owasp/zap2docker-weekly zap-baseline.py -t http://localhost:8000
```

## Troubleshooting

### Common Test Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError
# Solution: Ensure PYTHONPATH includes src directory
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
pytest tests/
```

#### 2. Async Test Issues

```python
# Error: RuntimeError: There is no current event loop
# Solution: Use pytest-asyncio and proper fixture
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

#### 3. Database Test Issues

```python
# Error: Database connection failed
# Solution: Use proper test database setup
@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine
```

#### 4. Mock Issues

```python
# Error: Mock not called as expected
# Solution: Verify mock setup and assertions
mock_function.assert_called_once_with(expected_args)
assert mock_function.call_count == 1
```

### Debugging Tests

```bash
# Run with debugging
pytest --pdb tests/unit/test_failing_test.py

# Verbose output
pytest -vv tests/

# Show local variables on failure
pytest -l tests/

# Only first failure
pytest -x tests/

# Trace function calls
pytest --trace tests/unit/test_specific.py
```

### Performance Debugging

```bash
# Profile test execution
python -m cProfile -o test_profile.stats -m pytest tests/unit/

# Memory usage tracking
python -m memory_profiler tests/unit/test_memory_intensive.py

# Line-by-line profiling
kernprof -l -v tests/performance/test_performance.py
```

## Best Practices

### 1. Test Organization

- **One test class per module/class being tested**
- **Descriptive test names that explain the scenario**
- **Group related tests in the same class**
- **Use consistent naming conventions**

### 2. Test Independence

- **Each test should be independent**
- **No shared state between tests**
- **Clean up resources in teardown**
- **Use fresh fixtures for each test**

### 3. Test Data

- **Use factories for test data generation**
- **Keep test data simple and minimal**
- **Use realistic but anonymized data**
- **Avoid hardcoded values when possible**

### 4. Assertions

- **One logical concept per test**
- **Use descriptive assertion messages**
- **Test both positive and negative cases**
- **Verify error conditions and edge cases**

### 5. Mocking Strategy

- **Mock external dependencies**
- **Keep mocks simple and focused**
- **Verify mock interactions when relevant**
- **Don't over-mock - test real behavior when possible**

### 6. Performance Considerations

- **Keep unit tests fast (< 100ms)**
- **Use appropriate test markers**
- **Parallel test execution for large suites**
- **Regular cleanup of test artifacts**

### 7. Security Testing

- **Test with malicious inputs**
- **Verify sanitization functions**
- **Test authentication and authorization**
- **Regular security audits of test code**

### 8. Anti-Hallucination Testing

- **Test with known placeholder patterns**
- **Verify cross-validation with multiple AI instances**
- **Test auto-completion workflows**
- **Measure accuracy of progress detection**

### 9. Documentation

- **Document complex test scenarios**
- **Explain non-obvious test setup**
- **Keep test documentation up-to-date**
- **Include performance expectations**

### 10. Continuous Improvement

- **Regular test suite maintenance**
- **Remove obsolete tests**
- **Update tests with code changes**
- **Monitor and improve test performance**

## Conclusion

This comprehensive testing strategy ensures the reliability, security, and performance of the claude-tiu project. By following these guidelines and utilizing the provided tools and frameworks, developers can maintain high code quality and catch issues early in the development cycle.

The anti-hallucination focus, combined with traditional testing practices, provides unique value in AI-powered development environments where code authenticity and quality validation are critical success factors.

For questions or improvements to the testing strategy, please consult the team leads or contribute to the testing documentation through pull requests.