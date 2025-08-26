#!/usr/bin/env python3
"""
Advanced Usage Example - Test Workflow Framework
Demonstrates advanced features including parallel execution, complex mocking,
context inheritance, and comprehensive reporting
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import tempfile

# Add test-workflow to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_workflow.core import (
    create_test_framework,
    TestSuite,
    TestContext,
    IntegratedTestRunner,
    ReportFormat
)


# Advanced example application
class DatabaseConnection:
    """Mock database connection"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
        self.transactions = []
        
    def connect(self):
        # Simulate connection delay
        time.sleep(0.01)
        self.connected = True
        return True
        
    def execute(self, query: str, params: Dict = None):
        if not self.connected:
            raise RuntimeError("Not connected to database")
        
        self.transactions.append({
            'query': query,
            'params': params or {},
            'timestamp': time.time()
        })
        
        # Mock query results
        if 'SELECT' in query:
            return [{'id': 1, 'name': 'test'}]
        elif 'INSERT' in query:
            return {'inserted_id': 123}
        elif 'UPDATE' in query:
            return {'affected_rows': 1}
        else:
            return {'success': True}
            
    def close(self):
        self.connected = False


class APIClient:
    """Mock API client"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.request_history = []
        
    async def get(self, endpoint: str, params: Dict = None):
        await self._simulate_network_delay()
        return self._make_request('GET', endpoint, params)
        
    async def post(self, endpoint: str, data: Dict):
        await self._simulate_network_delay()
        return self._make_request('POST', endpoint, data)
        
    async def _simulate_network_delay(self):
        await asyncio.sleep(0.05)  # Simulate network latency
        
    def _make_request(self, method: str, endpoint: str, data: Dict = None):
        request = {
            'method': method,
            'endpoint': endpoint,
            'data': data or {},
            'timestamp': time.time()
        }
        self.request_history.append(request)
        
        # Mock responses
        if endpoint == '/users':
            return {'users': [{'id': 1, 'name': 'John'}]}
        elif endpoint == '/users/create':
            return {'id': 2, 'name': data.get('name', 'New User')}
        else:
            return {'status': 'success', 'data': data}


class UserRepository:
    """User repository with database and API integration"""
    
    def __init__(self, db: DatabaseConnection, api: APIClient):
        self.db = db
        self.api = api
        
    async def create_user(self, user_data: Dict) -> Dict:
        """Create user with database and API integration"""
        
        # Validate input
        if not user_data.get('name'):
            raise ValueError("Name is required")
        
        # Save to database
        result = self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            {'name': user_data['name'], 'email': user_data.get('email', '')}
        )
        
        user_id = result['inserted_id']
        
        # Sync with API
        api_result = await self.api.post('/users/create', {
            'id': user_id,
            'name': user_data['name'],
            'email': user_data.get('email', '')
        })
        
        return {
            'id': user_id,
            'name': user_data['name'],
            'email': user_data.get('email', ''),
            'api_id': api_result.get('id')
        }
        
    async def get_user(self, user_id: int) -> Dict:
        """Get user from database"""
        
        users = self.db.execute(
            "SELECT * FROM users WHERE id = ?",
            {'id': user_id}
        )
        
        if not users:
            return None
            
        return users[0]
        
    async def list_users(self) -> List[Dict]:
        """List all users"""
        return self.db.execute("SELECT * FROM users")


async def run_advanced_examples():
    """Run advanced usage examples"""
    
    print("🚀 Running Test Workflow Framework - Advanced Usage Examples")
    print("=" * 70)
    
    # Create results directory
    results_dir = "examples/results/advanced"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create advanced test framework with comprehensive reporting
    framework = IntegratedTestRunner()
    
    # Configure multiple reporters
    framework.reporter.configure_reporter(
        ReportFormat.CONSOLE,
        include_details=True,
        include_stack_traces=True
    )
    
    framework.reporter.configure_reporter(
        ReportFormat.JSON,
        output_file=f"{results_dir}/advanced_results.json",
        include_coverage=True
    )
    
    framework.reporter.configure_reporter(
        ReportFormat.HTML,
        output_file=f"{results_dir}/advanced_report.html",
        include_details=True,
        include_stack_traces=True
    )
    
    framework.reporter.configure_reporter(
        ReportFormat.JUNIT,
        output_file=f"{results_dir}/junit.xml"
    )
    
    framework.reporter.configure_reporter(
        ReportFormat.MARKDOWN,
        output_file=f"{results_dir}/report.md"
    )
    
    # Create test suites
    integration_suite = create_integration_test_suite()
    performance_suite = create_performance_test_suite()
    complex_mock_suite = create_complex_mocking_suite()
    context_suite = create_context_management_suite()
    error_scenarios_suite = create_error_scenarios_suite()
    
    framework.add_suite(integration_suite)
    framework.add_suite(performance_suite)
    framework.add_suite(complex_mock_suite)
    framework.add_suite(context_suite)
    framework.add_suite(error_scenarios_suite)
    
    # Add hooks for comprehensive testing
    framework.register_hook('before_run', before_run_hook)
    framework.register_hook('after_run', after_run_hook)
    
    # Run tests with parallel execution
    print("\n📋 Running advanced test suites in parallel...")
    start_time = time.time()
    results = await framework.run_all(parallel=True)
    end_time = time.time()
    
    print(f"\n📊 Advanced Test Summary:")
    print(f"  Total Tests: {results['summary']['total']}")
    print(f"  Passed: {results['summary']['passed']} ✅")
    print(f"  Failed: {results['summary']['failed']} ❌")
    print(f"  Errors: {results['summary']['errors']} ⚠️")
    print(f"  Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"  Total Duration: {results['summary']['duration']:.3f}s")
    print(f"  Parallel Execution: {end_time - start_time:.3f}s")
    
    # Integration statistics
    stats = results.get('integration_stats', {})
    print(f"\n🔗 Integration Statistics:")
    print(f"  Total Assertions: {stats.get('total_assertions', 0)}")
    print(f"  Mocks Used: {stats.get('mocks_used', 0)}")
    print(f"  Context Keys: {stats.get('context_keys', 0)}")
    
    return results


def create_integration_test_suite():
    """Create comprehensive integration test suite"""
    
    suite = TestSuite("Integration Tests")
    
    def setup_integration_context(context):
        """Setup integration test environment"""
        context.set("db_connection", DatabaseConnection("sqlite:///test.db"))
        context.set("api_client", APIClient("https://api.example.com", "test-key"))
        context.set("temp_files", [])
        
    def teardown_integration_context(context):
        """Cleanup integration test environment"""
        db = context.get("db_connection")
        if db and db.connected:
            db.close()
        
        # Clean up temp files
        temp_files = context.get("temp_files", [])
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
                
    suite.setup = setup_integration_context
    suite.teardown = teardown_integration_context
    
    async def test_user_repository_create(assertions, mocks, context):
        """Test user repository creation with database and API integration"""
        
        db = context.get("db_connection")
        api = context.get("api_client")
        
        # Connect database
        db.connect()
        
        repo = UserRepository(db, api)
        
        user_data = {
            'name': 'John Doe',
            'email': 'john@example.com'
        }
        
        # Create user
        user = await repo.create_user(user_data)
        
        # Comprehensive assertions
        assertions.is_not_none(user)
        assertions.equal(user['name'], 'John Doe')
        assertions.equal(user['email'], 'john@example.com')
        assertions.isinstance_of(user['id'], int)
        assertions.is_not_none(user.get('api_id'))
        
        # Verify database transaction
        assertions.equal(len(db.transactions), 1)
        assertions.that(db.transactions[0]['query']).contains('INSERT INTO users')
        
        # Verify API call
        assertions.equal(len(api.request_history), 1)
        assertions.equal(api.request_history[0]['method'], 'POST')
        assertions.equal(api.request_history[0]['endpoint'], '/users/create')
        
    async def test_user_repository_get(assertions, mocks, context):
        """Test user repository retrieval"""
        
        db = context.get("db_connection")
        api = context.get("api_client")
        
        db.connect()
        repo = UserRepository(db, api)
        
        # Get user
        user = await repo.get_user(1)
        
        assertions.is_not_none(user)
        assertions.equal(user['id'], 1)
        assertions.equal(user['name'], 'test')
        
    async def test_user_repository_list(assertions, mocks, context):
        """Test user repository listing"""
        
        db = context.get("db_connection")
        api = context.get("api_client")
        
        db.connect()
        repo = UserRepository(db, api)
        
        # List users
        users = await repo.list_users()
        
        assertions.isinstance_of(users, list)
        assertions.greater_than(len(users), 0)
        
    async def test_error_scenarios(assertions, mocks, context):
        """Test error scenarios in integration"""
        
        db = context.get("db_connection")
        api = context.get("api_client")
        
        repo = UserRepository(db, api)
        
        # Test validation error
        with assertions.raises(ValueError, "Name is required"):
            await repo.create_user({})
        
        # Test database connection error
        with assertions.raises(RuntimeError, "Not connected to database"):
            await repo.get_user(1)
            
    # Add integration tests
    suite.add_test(test_user_repository_create, tags=["integration", "database", "api"])
    suite.add_test(test_user_repository_get, tags=["integration", "database"])
    suite.add_test(test_user_repository_list, tags=["integration", "database"])
    suite.add_test(test_error_scenarios, tags=["integration", "error_handling"])
    
    return suite


def create_performance_test_suite():
    """Create performance test suite"""
    
    suite = TestSuite("Performance Tests")
    
    async def test_database_operation_performance(assertions, mocks, context):
        """Test database operation performance"""
        
        db = DatabaseConnection("sqlite:///perf_test.db")
        db.connect()
        
        # Measure performance
        start_time = time.time()
        
        for i in range(100):
            db.execute(f"SELECT * FROM users WHERE id = {i}")
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assertions.less_than(duration, 1.0, "100 queries should complete within 1 second")
        
        # Verify operations completed
        assertions.equal(len(db.transactions), 100)
        
        context.set("db_performance", duration)
        
    async def test_api_client_performance(assertions, mocks, context):
        """Test API client performance"""
        
        api = APIClient("https://api.example.com", "perf-test-key")
        
        start_time = time.time()
        
        # Concurrent API calls
        tasks = []
        for i in range(20):
            task = api.get(f"/users/{i}")
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assertions.less_than(duration, 2.0, "20 concurrent API calls should complete within 2 seconds")
        assertions.equal(len(results), 20)
        assertions.equal(len(api.request_history), 20)
        
        context.set("api_performance", duration)
        
    async def test_memory_usage_simulation(assertions, mocks, context):
        """Test memory usage patterns"""
        
        # Simulate memory-intensive operation
        large_data = []
        
        for i in range(1000):
            large_data.append({
                'id': i,
                'data': f"test_data_{i}" * 100  # Create larger strings
            })
            
        # Test memory usage is reasonable
        import sys
        data_size = sys.getsizeof(large_data)
        
        assertions.less_than(data_size, 10 * 1024 * 1024, "Data should be less than 10MB")
        
        # Clean up
        del large_data
        
    # Add performance tests
    suite.add_test(test_database_operation_performance, tags=["performance", "database"])
    suite.add_test(test_api_client_performance, tags=["performance", "api", "concurrent"])
    suite.add_test(test_memory_usage_simulation, tags=["performance", "memory"])
    
    return suite


def create_complex_mocking_suite():
    """Create complex mocking scenarios test suite"""
    
    suite = TestSuite("Complex Mocking Scenarios")
    
    def test_database_mock_with_side_effects(assertions, mocks, context):
        """Test complex database mocking with side effects"""
        
        # Create mock with sequence of return values
        db_mock = mocks.create_mock("complex_db")
        
        # Configure mock with return value sequence
        db_mock.execute.side_effect = [
            [{'id': 1, 'name': 'First'}],
            [{'id': 2, 'name': 'Second'}],
            RuntimeError("Database connection lost")
        ]
        
        # First call
        result1 = db_mock.execute("SELECT * FROM users WHERE id = 1")
        assertions.equal(result1, [{'id': 1, 'name': 'First'}])
        
        # Second call
        result2 = db_mock.execute("SELECT * FROM users WHERE id = 2")
        assertions.equal(result2, [{'id': 2, 'name': 'Second'}])
        
        # Third call should raise exception
        with assertions.raises(RuntimeError):
            db_mock.execute("SELECT * FROM users WHERE id = 3")
        
        # Verify call count
        mocks.assert_call_count("complex_db", 3)
        
    def test_api_mock_with_conditional_responses(assertions, mocks, context):
        """Test API mock with conditional responses"""
        
        api_mock = mocks.create_mock("conditional_api")
        
        def conditional_response(*args, **kwargs):
            endpoint = args[0] if args else kwargs.get('endpoint', '')
            
            if '/users' in endpoint:
                return {'users': [{'id': 1}]}
            elif '/posts' in endpoint:
                return {'posts': [{'id': 1, 'title': 'Test Post'}]}
            else:
                return {'error': 'Not found'}
                
        api_mock.get.side_effect = conditional_response
        
        # Test different endpoints
        users_response = api_mock.get('/users')
        assertions.equal(users_response, {'users': [{'id': 1}]})
        
        posts_response = api_mock.get('/posts')
        assertions.equal(posts_response, {'posts': [{'id': 1, 'title': 'Test Post'}]})
        
        error_response = api_mock.get('/unknown')
        assertions.equal(error_response, {'error': 'Not found'})
        
    def test_spy_on_real_object(assertions, mocks, context):
        """Test spying on real objects"""
        
        # Create real object
        db = DatabaseConnection("sqlite:///spy_test.db")
        
        # Create spy on connect method
        connect_spy = mocks.create_spy(db, "connect")
        
        # Use real object
        db.connect()
        assertions.is_true(db.connected)
        
        # Verify spy recorded the call
        mocks.assert_called_once(connect_spy.name)
        
    async def test_async_mock_scenarios(assertions, mocks, context):
        """Test complex async mocking scenarios"""
        
        # Create async mock with delay simulation
        async_service = mocks.create_async_mock("delayed_service")
        
        # Configure to simulate delay
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate processing time
            return f"processed: {args[0] if args else 'default'}"
            
        async_service.return_value = asyncio.coroutine(delayed_response)()
        
        # Test async behavior
        start_time = time.time()
        result = await async_service("test_data")
        end_time = time.time()
        
        duration = end_time - start_time
        assertions.greater_than(duration, 0.01)
        assertions.that(result).contains("processed: test_data")
        
    # Add complex mocking tests
    suite.add_test(test_database_mock_with_side_effects, tags=["mocking", "complex"])
    suite.add_test(test_api_mock_with_conditional_responses, tags=["mocking", "api"])
    suite.add_test(test_spy_on_real_object, tags=["mocking", "spy"])
    suite.add_test(test_async_mock_scenarios, tags=["mocking", "async"])
    
    return suite


def create_context_management_suite():
    """Create context management test suite"""
    
    suite = TestSuite("Context Management")
    
    def setup_parent_context(context):
        """Setup parent context with shared data"""
        context.set("global_config", {"env": "test", "debug": True})
        context.set("shared_counter", 0)
        
        # Create temporary file for testing
        temp_file = context.create_temp_file("test content", suffix=".txt")
        context.set("temp_file_path", temp_file)
        
    suite.setup = setup_parent_context
    
    def test_context_inheritance(assertions, mocks, context):
        """Test context inheritance from parent"""
        
        # Should inherit from parent context
        config = context.get("global_config")
        assertions.is_not_none(config)
        assertions.equal(config["env"], "test")
        assertions.is_true(config["debug"])
        
        # Should inherit counter
        counter = context.get("shared_counter")
        assertions.equal(counter, 0)
        
    def test_context_isolation(assertions, mocks, context):
        """Test that test contexts are isolated"""
        
        # Modify local context
        context.set("test_specific_data", "local_value")
        counter = context.get("shared_counter")
        context.set("shared_counter", counter + 1)
        
        # Verify local changes
        assertions.equal(context.get("test_specific_data"), "local_value")
        assertions.equal(context.get("shared_counter"), 1)
        
    def test_context_isolation_verification(assertions, mocks, context):
        """Verify context isolation from previous test"""
        
        # Should not have test-specific data from previous test
        test_specific = context.get("test_specific_data")
        assertions.is_none(test_specific)
        
        # Should have original shared counter (contexts are isolated)
        counter = context.get("shared_counter")
        assertions.equal(counter, 0)
        
    def test_context_snapshots(assertions, mocks, context):
        """Test context snapshot functionality"""
        
        # Create initial state
        context.set("data", {"count": 1})
        
        # Create snapshot
        snapshot_name = context.create_snapshot("before_changes")
        
        # Modify data
        data = context.get("data")
        data["count"] = 10
        context.set("data", data)
        context.set("new_key", "new_value")
        
        # Verify changes
        assertions.equal(context.get("data")["count"], 10)
        assertions.equal(context.get("new_key"), "new_value")
        
        # Restore snapshot
        context.restore_snapshot(snapshot_name)
        
        # Verify restoration
        restored_data = context.get("data")
        assertions.equal(restored_data["count"], 1)
        assertions.is_none(context.get("new_key"))
        
    def test_temporary_file_management(assertions, mocks, context):
        """Test temporary file management"""
        
        # Context should have temp file from setup
        temp_file = context.get("temp_file_path")
        assertions.is_not_none(temp_file)
        assertions.is_true(os.path.exists(temp_file))
        
        # Create additional temp file
        additional_temp = context.create_temp_file("additional content")
        assertions.is_true(os.path.exists(additional_temp))
        
        # Read content
        with open(temp_file, 'r') as f:
            content = f.read()
        assertions.equal(content, "test content")
        
    def test_context_temporary_changes(assertions, mocks, context):
        """Test temporary context changes"""
        
        original_value = "original"
        context.set("temp_test_key", original_value)
        
        # Use temporary context changes
        with context.temporary_set("temp_test_key", "temporary"):
            # Inside context manager, value is temporary
            assertions.equal(context.get("temp_test_key"), "temporary")
            
        # Outside context manager, value is restored
        assertions.equal(context.get("temp_test_key"), original_value)
        
    # Add context management tests
    suite.add_test(test_context_inheritance, tags=["context", "inheritance"])
    suite.add_test(test_context_isolation, tags=["context", "isolation"])
    suite.add_test(test_context_isolation_verification, tags=["context", "isolation"])
    suite.add_test(test_context_snapshots, tags=["context", "snapshots"])
    suite.add_test(test_temporary_file_management, tags=["context", "files"])
    suite.add_test(test_context_temporary_changes, tags=["context", "temporary"])
    
    return suite


def create_error_scenarios_suite():
    """Create comprehensive error scenarios test suite"""
    
    suite = TestSuite("Error Scenarios")
    
    def test_assertion_failures_with_details(assertions, mocks, context):
        """Test detailed assertion failure messages"""
        
        # This test demonstrates assertion failures with detailed messages
        try:
            complex_object = {
                'name': 'John',
                'age': 30,
                'skills': ['Python', 'JavaScript'],
                'active': True
            }
            
            expected_object = {
                'name': 'Jane',  # Different name
                'age': 25,       # Different age
                'skills': ['Python', 'Java'],  # Different skills
                'active': False  # Different status
            }
            
            assertions.equal(complex_object, expected_object, 
                           "Complex object comparison should show detailed diff")
                           
        except Exception as e:
            # This assertion failure is expected for demonstration
            context.set("expected_failure", str(e))
            
        # Test that we captured the expected failure
        failure_message = context.get("expected_failure")
        assertions.is_not_none(failure_message)
        assertions.that(failure_message).contains("Complex object comparison")
        
    async def test_async_error_handling(assertions, mocks, context):
        """Test async error handling scenarios"""
        
        async def async_operation_that_fails():
            await asyncio.sleep(0.01)
            raise ConnectionError("Network connection failed")
            
        # Test async exception handling
        with assertions.raises(ConnectionError) as exc_info:
            await async_operation_that_fails()
            
        # Verify exception details
        exception = exc_info.exception_caught
        assertions.that(str(exception)).contains("Network connection failed")
        
    def test_mock_configuration_errors(assertions, mocks, context):
        """Test mock configuration error scenarios"""
        
        # Create mock with call count limit
        limited_mock = mocks.create_mock(
            "limited_mock", 
            call_count_limit=2
        )
        
        # First two calls should work
        limited_mock("call 1")
        limited_mock("call 2")
        
        # Third call should fail
        with assertions.raises(Exception) as exc_info:
            limited_mock("call 3")
            
        exception_message = str(exc_info.exception_caught)
        assertions.that(exception_message).contains("exceeded call count limit")
        
    def test_context_error_scenarios(assertions, mocks, context):
        """Test context-related error scenarios"""
        
        # Test snapshot restoration of non-existent snapshot
        with assertions.raises(ValueError) as exc_info:
            context.restore_snapshot("non_existent_snapshot")
            
        exception_message = str(exc_info.exception_caught)
        assertions.that(exception_message).contains("not found")
        
    def test_integration_failure_cascade(assertions, mocks, context):
        """Test how failures cascade through integrated components"""
        
        # Create a scenario where multiple components fail
        db_mock = mocks.create_mock("failing_db")
        db_mock.execute.side_effect = RuntimeError("Database failure")
        
        api_mock = mocks.create_mock("failing_api")
        api_mock.get.side_effect = ConnectionError("API failure")
        
        # Test that multiple failures are properly tracked
        errors = []
        
        try:
            db_mock.execute("SELECT * FROM users")
        except Exception as e:
            errors.append(f"DB: {e}")
            
        try:
            api_mock.get("/users")
        except Exception as e:
            errors.append(f"API: {e}")
            
        # Verify multiple errors were captured
        assertions.equal(len(errors), 2)
        assertions.that(errors[0]).contains("Database failure")
        assertions.that(errors[1]).contains("API failure")
        
        context.set("cascade_errors", errors)
        
    # Add error scenario tests
    suite.add_test(test_assertion_failures_with_details, tags=["error", "assertions"])
    suite.add_test(test_async_error_handling, tags=["error", "async"])
    suite.add_test(test_mock_configuration_errors, tags=["error", "mocking"])
    suite.add_test(test_context_error_scenarios, tags=["error", "context"])
    suite.add_test(test_integration_failure_cascade, tags=["error", "integration"])
    
    return suite


def before_run_hook(runner):
    """Hook executed before test run starts"""
    print("🔧 Setting up advanced test environment...")
    
    # Set global context
    runner.global_context.set("test_run_id", f"advanced_{int(time.time())}")
    runner.global_context.set("environment", "advanced_testing")
    
    # Create shared resources
    runner.global_context.set("shared_resources", {
        "temp_dir": tempfile.mkdtemp(),
        "start_time": time.time()
    })


def after_run_hook(runner, results):
    """Hook executed after test run completes"""
    print("🧹 Cleaning up advanced test environment...")
    
    # Cleanup shared resources
    shared_resources = runner.global_context.get("shared_resources", {})
    temp_dir = shared_resources.get("temp_dir")
    
    if temp_dir and os.path.exists(temp_dir):
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️ Failed to cleanup temp directory: {e}")
            
    # Calculate additional metrics
    start_time = shared_resources.get("start_time", 0)
    if start_time:
        total_time = time.time() - start_time
        print(f"📊 Total execution time: {total_time:.3f}s")


if __name__ == "__main__":
    # Run the advanced examples
    results = asyncio.run(run_advanced_examples())
    
    print(f"\n📄 Comprehensive reports generated in examples/results/advanced/:")
    print(f"  🖥️  Console: Displayed above")
    print(f"  📊 JSON: advanced_results.json")
    print(f"  🌐 HTML: advanced_report.html")
    print(f"  📋 JUnit: junit.xml")
    print(f"  📝 Markdown: report.md")
    
    print(f"\n🎉 Advanced examples completed!")
    
    # Return appropriate exit code
    if results['summary']['failed'] > 0 or results['summary']['errors'] > 0:
        print("⚠️ Some tests failed or had errors - check reports for details")
        sys.exit(1)
    else:
        print("✅ All tests passed successfully!")
        sys.exit(0)