# Getting Started with Test Workflow Framework

Welcome to the **Test Workflow Framework** - a comprehensive, integrated testing solution designed for SPARC methodology (Specification, Pseudocode, Architecture, Refinement, Completion). This guide will help you get up and running quickly.

## 🎯 What is Test Workflow Framework?

The Test Workflow Framework is a cohesive testing solution that integrates:

- **Test Runner**: Executes tests with comprehensive orchestration
- **Assertion Library**: Provides detailed, fluent assertions with rich error messages
- **Mock Framework**: Advanced mocking, stubbing, and spying capabilities
- **Test Context**: Shared test data and fixture management
- **Reporter System**: Multiple output formats (Console, JSON, HTML, JUnit, Markdown)
- **CI/CD Integration**: Ready-to-use GitHub Actions and Jenkins pipelines

## 🚀 Quick Start

### Installation

```bash
# Install the framework
pip install test-workflow

# Or install from source
git clone https://github.com/your-org/test-workflow
cd test-workflow
pip install -e .
```

### Your First Test

Create a simple test file `test_example.py`:

```python
import asyncio
from test_workflow.core import create_test_framework, TestSuite

async def run_tests():
    # Create test framework
    framework = create_test_framework()
    
    # Create test suite
    suite = TestSuite("My First Tests")
    
    def test_basic_math(assertions, mocks, context):
        """Test basic mathematics"""
        assertions.equal(2 + 2, 4)
        assertions.greater_than(5, 3)
        assertions.is_true(10 > 5)
        
    def test_strings(assertions, mocks, context):
        """Test string operations"""
        text = "Hello, World!"
        assertions.contains(text, "World")
        assertions.length_equal(text, 13)
        assertions.regex_match(text, r"Hello.*World")
    
    # Add tests to suite
    suite.add_test(test_basic_math)
    suite.add_test(test_strings)
    
    # Run tests
    framework.add_suite(suite)
    results = await framework.run_all()
    
    print(f"Tests completed: {results['summary']['passed']}/{results['summary']['total']} passed")

if __name__ == "__main__":
    asyncio.run(run_tests())
```

Run your first test:

```bash
python test_example.py
```

## 📋 Core Concepts

### 1. Test Framework Creation

The framework provides several ways to create and configure your test environment:

```python
# Basic framework
framework = create_test_framework()

# Framework with specific reporting
framework = create_test_framework(
    console_output=True,
    json_output="results.json",
    html_output="report.html",
    junit_output="junit.xml"
)

# Advanced framework with custom configuration
from test_workflow.core import IntegratedTestRunner, ReportFormat

framework = IntegratedTestRunner()
framework.reporter.configure_reporter(
    ReportFormat.HTML,
    output_file="custom_report.html",
    include_details=True,
    include_stack_traces=True
)
```

### 2. Test Suites

Test suites group related tests and provide setup/teardown capabilities:

```python
from test_workflow.core import TestSuite

# Create suite
suite = TestSuite("User Management Tests")

# Add setup/teardown
def setup_database(context):
    context.set("database", create_test_database())

def cleanup_database(context):
    db = context.get("database")
    db.close()

suite.setup = setup_database
suite.teardown = cleanup_database

# Add before/after each test hooks
def reset_database(context):
    db = context.get("database")
    db.clear_all_tables()

suite.before_each = reset_database
```

### 3. Test Functions

Test functions receive three integrated parameters:

```python
def test_user_creation(assertions, mocks, context):
    """
    Test function signature:
    - assertions: Assertion framework for validations
    - mocks: Mock framework for test isolation
    - context: Shared test context and data
    """
    
    # Use assertions
    user = create_user("John", "john@example.com")
    assertions.is_not_none(user)
    assertions.equal(user.name, "John")
    
    # Use mocks
    email_service = mocks.create_mock("email_service")
    email_service.send_welcome_email.return_value = True
    
    send_welcome_email(user, email_service)
    mocks.assert_called_with("email_service", user)
    
    # Use context
    db = context.get("database")
    saved_user = db.find_user(user.id)
    assertions.equal(saved_user, user)
```

## 🧪 Assertion Examples

The framework provides comprehensive assertions with detailed error messages:

### Basic Assertions

```python
def test_basic_assertions(assertions, mocks, context):
    # Equality
    assertions.equal(actual, expected, "Custom message")
    assertions.not_equal(actual, unexpected)
    
    # Boolean
    assertions.is_true(condition)
    assertions.is_false(condition)
    
    # Null checks
    assertions.is_none(value)
    assertions.is_not_none(value)
    
    # Numeric comparisons
    assertions.greater_than(10, 5)
    assertions.greater_than_or_equal(10, 10)
    assertions.less_than(5, 10)
    assertions.less_than_or_equal(5, 5)
```

### Collection Assertions

```python
def test_collection_assertions(assertions, mocks, context):
    items = [1, 2, 3, 4, 5]
    
    # Membership
    assertions.contains(items, 3)
    assertions.not_contains(items, 10)
    
    # Length
    assertions.length_equal(items, 5)
    assertions.empty([])
    assertions.not_empty(items)
    
    # Type checking
    assertions.isinstance_of(items, list)
    
    # Pattern matching
    assertions.regex_match("test@example.com", r".*@.*\.com")
```

### Advanced Assertions

```python
def test_advanced_assertions(assertions, mocks, context):
    users = [
        {"name": "John", "active": True},
        {"name": "Jane", "active": True},
        {"name": "Bob", "active": False}
    ]
    
    # Collection predicates
    assertions.all_match(users, lambda u: "name" in u)
    assertions.any_match(users, lambda u: u["active"])
    
    # Dictionary subset
    user = {"id": 1, "name": "John", "email": "john@example.com"}
    expected_subset = {"name": "John", "email": "john@example.com"}
    assertions.dict_contains_subset(user, expected_subset)
```

### Fluent Interface

```python
def test_fluent_assertions(assertions, mocks, context):
    value = 42
    
    assertions.that(value).equals(42)
    assertions.that(value).is_greater_than(40)
    assertions.that(value).is_instance_of(int)
    
    text = "Hello World"
    assertions.that(text).contains("World")
    assertions.that(text).has_length(11)
    assertions.that(text).is_not_empty()
```

### Exception Assertions

```python
def test_exception_handling(assertions, mocks, context):
    # As context manager
    with assertions.raises(ValueError, "Should raise ValueError"):
        raise ValueError("Something went wrong")
    
    # Access the caught exception
    with assertions.raises(ValueError) as exc_context:
        raise ValueError("Custom error message")
    
    # Verify exception details
    exception = exc_context.exception_caught
    assertions.that(str(exception)).contains("Custom error message")
```

## 🎭 Mock Framework Usage

The mock framework provides comprehensive mocking capabilities:

### Basic Mocks

```python
def test_basic_mocking(assertions, mocks, context):
    # Create mock
    api_mock = mocks.create_mock("api_client")
    
    # Configure return value
    api_mock.get_user.return_value = {"id": 1, "name": "John"}
    
    # Use mock
    result = api_mock.get_user(1)
    
    # Verify
    assertions.equal(result, {"id": 1, "name": "John"})
    mocks.assert_called_with("api_client", 1)
    mocks.assert_called_once("api_client")
```

### Advanced Mocking

```python
def test_advanced_mocking(assertions, mocks, context):
    # Mock with side effects
    db_mock = mocks.create_mock("database")
    db_mock.execute.side_effect = [
        [{"id": 1}],  # First call
        [{"id": 2}],  # Second call
        RuntimeError("Connection lost")  # Third call raises exception
    ]
    
    # First two calls work
    result1 = db_mock.execute("SELECT * FROM users WHERE id = 1")
    result2 = db_mock.execute("SELECT * FROM users WHERE id = 2")
    
    assertions.equal(result1, [{"id": 1}])
    assertions.equal(result2, [{"id": 2}])
    
    # Third call raises exception
    with assertions.raises(RuntimeError):
        db_mock.execute("SELECT * FROM users WHERE id = 3")
```

### Spies

```python
def test_spying(assertions, mocks, context):
    # Create real object
    class EmailService:
        def send_email(self, to, subject, body):
            return f"Email sent to {to}: {subject}"
    
    service = EmailService()
    
    # Create spy
    spy = mocks.create_spy(service, "send_email")
    
    # Use real object (spy records calls)
    result = service.send_email("test@example.com", "Hello", "Test message")
    
    # Verify real functionality worked
    assertions.that(result).contains("Email sent to test@example.com")
    
    # Verify spy recorded the call
    mocks.assert_called_with(spy.name, "test@example.com", "Hello", "Test message")
```

### Async Mocks

```python
async def test_async_mocking(assertions, mocks, context):
    # Create async mock
    async_service = mocks.create_async_mock(
        "async_service",
        return_value="async_result"
    )
    
    # Use async mock
    result = await async_service()
    
    assertions.equal(result, "async_result")
    mocks.assert_called("async_service")
```

## 📊 Context Management

Test context provides shared data and fixture management:

### Basic Context Usage

```python
def test_context_basics(assertions, mocks, context):
    # Set values
    context.set("user_id", 123)
    context.set("config", {"debug": True, "timeout": 30})
    
    # Get values
    user_id = context.get("user_id")
    config = context.get("config")
    timeout = config["timeout"]
    
    assertions.equal(user_id, 123)
    assertions.equal(timeout, 30)
    
    # Check existence
    assertions.is_true(context.has("user_id"))
    assertions.is_false(context.has("nonexistent"))
```

### Fixtures

```python
def test_fixtures(assertions, mocks, context):
    # Add fixture
    context.add_fixture("database", create_test_database())
    
    # Use fixture
    db = context.get_fixture("database")
    user = db.create_user("John", "john@example.com")
    
    assertions.is_not_none(user)
```

### Temporary Files

```python
def test_temporary_files(assertions, mocks, context):
    # Create temp file
    temp_file = context.create_temp_file(
        content="test data",
        suffix=".txt",
        prefix="test_"
    )
    
    # Use temp file
    with open(temp_file, 'r') as f:
        content = f.read()
    
    assertions.equal(content, "test data")
    
    # File is automatically cleaned up after test
```

### Context Snapshots

```python
def test_context_snapshots(assertions, mocks, context):
    # Set initial state
    context.set("counter", 5)
    
    # Create snapshot
    snapshot = context.create_snapshot("initial_state")
    
    # Modify state
    context.set("counter", 10)
    context.set("new_key", "value")
    
    # Restore snapshot
    context.restore_snapshot(snapshot)
    
    # State is restored
    assertions.equal(context.get("counter"), 5)
    assertions.is_none(context.get("new_key"))
```

## 📈 Reporting

The framework supports multiple report formats:

### Console Output
Automatically displays colored results in the terminal with detailed failure information.

### JSON Report
```python
framework = create_test_framework(
    json_output="test_results.json"
)
```

### HTML Report
```python
framework = create_test_framework(
    html_output="test_report.html"
)
```

### JUnit XML
```python
framework = create_test_framework(
    junit_output="junit.xml"
)
```

### Multiple Reports
```python
framework = create_test_framework(
    console_output=True,
    json_output="results.json",
    html_output="report.html",
    junit_output="junit.xml"
)
```

## 🚀 Advanced Features

### Parallel Execution

```python
# Run test suites in parallel
results = await framework.run_all(parallel=True)
```

### Test Filtering

```python
# Add tags to tests
suite.add_test(test_function, tags=["unit", "fast"])
suite.add_test(slow_test, tags=["integration", "slow"])

# Run only specific tags
results = await framework.run_all(filter_tags=["unit"])
```

### Custom Hooks

```python
def before_run_hook(runner):
    print("Setting up test environment...")
    runner.global_context.set("start_time", time.time())

def after_run_hook(runner, results):
    print("Cleaning up...")
    
framework.register_hook('before_run', before_run_hook)
framework.register_hook('after_run', after_run_hook)
```

### Test Discovery

```python
# Discover tests from modules
discovered_suites = framework.discover_tests('my_test_module')
for suite in discovered_suites:
    framework.add_suite(suite)
```

## 🏗️ CI/CD Integration

### GitHub Actions

The framework includes a comprehensive GitHub Actions workflow:

```yaml
# Copy .github/workflows/test_workflow.yml to your project
name: Test Workflow CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install test-workflow
      - name: Run tests
        run: |
          python -m pytest --cov=src --html=report.html
```

### Jenkins Pipeline

```groovy
// Use the included Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'python -m pytest --junit-xml=junit.xml'
            }
        }
    }
    post {
        always {
            publishTestResults testResultsPattern: 'junit.xml'
        }
    }
}
```

## 🔧 Best Practices

### 1. Test Organization

```python
# Group related tests in suites
user_tests = TestSuite("User Management")
auth_tests = TestSuite("Authentication")
api_tests = TestSuite("API Endpoints")

# Use descriptive test names
def test_user_creation_with_valid_email(assertions, mocks, context):
    pass

def test_user_creation_fails_with_invalid_email(assertions, mocks, context):
    pass
```

### 2. Setup and Teardown

```python
# Use suite-level setup for expensive operations
def setup_database(context):
    context.set("database", create_test_database())

# Use before_each for test isolation
def reset_database(context):
    db = context.get("database")
    db.clear_all_data()

suite.setup = setup_database
suite.before_each = reset_database
```

### 3. Assertion Messages

```python
# Provide clear, actionable error messages
assertions.equal(
    actual_count, 
    expected_count,
    f"User count mismatch after creating {num_users} users"
)

# Use assertions that provide good error details
assertions.dict_contains_subset(
    actual_user,
    {"name": "John", "email": "john@example.com"},
    "Created user should have correct name and email"
)
```

### 4. Mock Strategy

```python
# Mock external dependencies
def test_user_service_with_external_api(assertions, mocks, context):
    # Mock external API
    api_mock = mocks.create_mock("external_api")
    api_mock.verify_email.return_value = True
    
    # Test your service
    service = UserService(api_client=api_mock)
    user = service.create_user("john@example.com")
    
    # Verify interaction
    mocks.assert_called_with("external_api", "john@example.com")
    assertions.is_not_none(user)
```

### 5. Context Usage

```python
# Share expensive fixtures across tests
def setup_suite(context):
    context.add_fixture("ml_model", load_trained_model())
    context.add_fixture("test_data", load_test_dataset())

# Use context for test isolation
def test_model_prediction(assertions, mocks, context):
    model = context.get_fixture("ml_model")
    test_data = context.get_fixture("test_data")
    
    # Test model without affecting other tests
    predictions = model.predict(test_data[:10])
    assertions.equal(len(predictions), 10)
```

## 📚 Examples Repository

Check out the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Simple test scenarios
- `advanced_usage.py` - Complex integration testing
- `performance_testing.py` - Performance benchmarks
- `async_testing.py` - Async test patterns
- `integration_testing.py` - Full integration scenarios

## 🆘 Troubleshooting

### Common Issues

**Q: Tests are failing with import errors**
```bash
# Make sure the framework is installed
pip install test-workflow

# Or install in development mode
pip install -e .
```

**Q: Async tests not working properly**
```python
# Make sure to use asyncio.run() in main
if __name__ == "__main__":
    asyncio.run(run_tests())

# Ensure test functions are properly async
async def test_async_operation(assertions, mocks, context):
    result = await some_async_function()
    assertions.is_not_none(result)
```

**Q: Mocks not being reset between tests**
```python
# The framework automatically resets mocks between tests
# If you need manual reset:
def before_each_test(context):
    # Mocks are automatically reset
    pass

suite.before_each = before_each_test
```

**Q: Context data persisting between tests**
```python
# Test contexts are automatically isolated
# Use suite.before_each for explicit cleanup:
def reset_context(context):
    # Only need to reset shared fixtures, not test data
    db = context.get_fixture("database")
    if db:
        db.clear_all_tables()
```

### Getting Help

- **Documentation**: Check the `docs/` directory
- **Examples**: Look at `examples/` for usage patterns
- **Issues**: Report bugs on GitHub
- **Community**: Join our Discord server

## 🎉 Next Steps

Now that you understand the basics:

1. **Run the examples**: Execute `basic_usage.py` and `advanced_usage.py`
2. **Write your first test suite**: Start with simple unit tests
3. **Explore advanced features**: Try parallel execution and custom reporters
4. **Set up CI/CD**: Use the provided GitHub Actions or Jenkins pipeline
5. **Join the community**: Share your experiences and get help

Happy testing! 🧪✨