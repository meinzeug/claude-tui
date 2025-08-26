"""
Integration Tests - Verify component interactions in test-workflow framework
Tests the integration between test runner, assertions, mocks, context, and reporting
"""

import asyncio
import tempfile
import json
import os
from pathlib import Path
import pytest

from test_workflow.core import (
    IntegratedTestRunner,
    TestSuite,
    TestContext,
    AssertionFramework,
    MockFramework,
    TestReporter,
    ReportFormat,
    create_test_framework
)


class TestFrameworkIntegration:
    """Test integration between all framework components"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = IntegratedTestRunner()
        
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @pytest.mark.asyncio
    async def test_runner_assertion_integration(self):
        """Test that test runner properly integrates with assertion framework"""
        
        # Create test suite with assertion usage
        suite = TestSuite("assertion_integration_suite")
        
        def test_assertions_work(assertions, mocks, context):
            """Test that uses assertions"""
            assertions.equal(2 + 2, 4)
            assertions.is_true(True)
            assertions.contains([1, 2, 3], 2)
            
        def test_assertion_failure(assertions, mocks, context):
            """Test that deliberately fails"""
            assertions.equal(2 + 2, 5, "This should fail")
            
        suite.add_test(test_assertions_work)
        suite.add_test(test_assertion_failure)
        
        self.runner.add_suite(suite)
        
        # Run tests
        results = await self.runner.run_all()
        
        # Verify integration worked
        assert results['summary']['total'] == 2
        assert results['summary']['passed'] == 1
        assert results['summary']['failed'] == 1
        
        # Check assertion counts
        test_results = results['results']
        passed_test = next(r for r in test_results if r['status'] == 'passed')
        failed_test = next(r for r in test_results if r['status'] == 'failed')
        
        assert passed_test['assertions_count'] == 3
        assert failed_test['assertions_count'] == 1
        assert 'This should fail' in failed_test['message']
        
    @pytest.mark.asyncio
    async def test_runner_mock_integration(self):
        """Test that test runner properly integrates with mock framework"""
        
        suite = TestSuite("mock_integration_suite")
        
        def test_mock_usage(assertions, mocks, context):
            """Test that uses mocks"""
            # Create and use a mock
            db_mock = mocks.create_mock("database")
            db_mock.execute.return_value = [{'id': 1, 'name': 'test'}]
            
            # Call the mock
            result = db_mock.execute("SELECT * FROM users")
            
            # Assert the mock worked
            assertions.equal(result, [{'id': 1, 'name': 'test'}])
            mocks.assert_called(db_mock, "execute")
            
        def test_spy_usage(assertions, mocks, context):
            """Test that uses spies"""
            # Create an object to spy on
            class TestService:
                def process_data(self, data):
                    return f"processed: {data}"
                    
            service = TestService()
            spy = mocks.create_spy(service, "process_data")
            
            # Use the service (spy records calls)
            result = service.process_data("test_data")
            
            # Verify spy recorded the call
            assertions.equal(result, "processed: test_data")
            mocks.assert_called_with(spy.name, "test_data")
            
        suite.add_test(test_mock_usage)
        suite.add_test(test_spy_usage)
        
        self.runner.add_suite(suite)
        results = await self.runner.run_all()
        
        # Verify mock integration
        assert results['summary']['total'] == 2
        assert results['summary']['passed'] == 2
        
        # Check mock usage tracking
        test_results = results['results']
        mock_test = next(r for r in test_results if 'mock_usage' in r['name'])
        spy_test = next(r for r in test_results if 'spy_usage' in r['name'])
        
        assert len(mock_test['mocks_used']) > 0
        assert len(spy_test['mocks_used']) > 0
        
    @pytest.mark.asyncio
    async def test_runner_context_integration(self):
        """Test that test runner properly integrates with test context"""
        
        suite = TestSuite("context_integration_suite")
        
        # Setup suite context
        suite.context = TestContext("suite_context")
        suite.context.set("shared_data", {"counter": 0})
        
        def test_context_inheritance(assertions, mocks, context):
            """Test that inherits from suite context"""
            shared = context.get("shared_data")
            assertions.is_not_none(shared)
            assertions.equal(shared["counter"], 0)
            
            # Modify in local context
            context.set("test_specific", "local_value")
            
        def test_context_isolation(assertions, mocks, context):
            """Test that contexts are properly isolated"""
            shared = context.get("shared_data")
            assertions.equal(shared["counter"], 0)  # Should still be 0
            
            # This should not exist (from previous test)
            test_specific = context.get("test_specific")
            assertions.is_none(test_specific)
            
        suite.add_test(test_context_inheritance)
        suite.add_test(test_context_isolation)
        
        self.runner.add_suite(suite)
        results = await self.runner.run_all()
        
        # Verify context integration
        assert results['summary']['total'] == 2
        assert results['summary']['passed'] == 2
        
        # Check context data was tracked
        test_results = results['results']
        for result in test_results:
            assert 'context_data' in result
            assert len(result['context_data']) > 0
            
    @pytest.mark.asyncio
    async def test_reporter_integration(self):
        """Test that test runner properly integrates with reporting system"""
        
        # Create runner with multiple reporters
        runner = IntegratedTestRunner()
        runner.reporter.configure_reporter(
            ReportFormat.JSON,
            output_file=os.path.join(self.temp_dir, "test_results.json")
        )
        runner.reporter.configure_reporter(
            ReportFormat.HTML,
            output_file=os.path.join(self.temp_dir, "test_results.html")
        )
        
        # Create test suite
        suite = TestSuite("reporter_integration_suite")
        
        def test_successful(assertions, mocks, context):
            assertions.is_true(True)
            
        def test_failed(assertions, mocks, context):
            assertions.equal(1, 2, "Deliberate failure")
            
        suite.add_test(test_successful)
        suite.add_test(test_failed)
        
        runner.add_suite(suite)
        
        # Run tests
        results = await runner.run_all()
        
        # Verify reports were generated
        json_file = os.path.join(self.temp_dir, "test_results.json")
        html_file = os.path.join(self.temp_dir, "test_results.html")
        
        assert os.path.exists(json_file)
        assert os.path.exists(html_file)
        
        # Verify JSON report content
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            
        assert json_data['test_framework'] == 'test-workflow'
        assert json_data['results']['summary']['total'] == 2
        assert json_data['results']['summary']['passed'] == 1
        assert json_data['results']['summary']['failed'] == 1
        
        # Verify HTML report content
        with open(html_file, 'r') as f:
            html_content = f.read()
            
        assert 'Test Results Report' in html_content
        assert 'test_successful' in html_content
        assert 'test_failed' in html_content
        
    @pytest.mark.asyncio
    async def test_async_test_integration(self):
        """Test that async tests work properly with all components"""
        
        suite = TestSuite("async_integration_suite")
        
        async def async_test_with_assertions(assertions, mocks, context):
            """Async test using assertions"""
            await asyncio.sleep(0.01)  # Simulate async work
            assertions.equal(await self._async_operation(5), 10)
            
        async def async_test_with_mocks(assertions, mocks, context):
            """Async test using mocks"""
            async_mock = mocks.create_async_mock("async_service", return_value="async_result")
            
            result = await async_mock()
            assertions.equal(result, "async_result")
            
        suite.add_test(async_test_with_assertions)
        suite.add_test(async_test_with_mocks)
        
        self.runner.add_suite(suite)
        results = await self.runner.run_all()
        
        # Verify async integration
        assert results['summary']['total'] == 2
        assert results['summary']['passed'] == 2
        
    async def _async_operation(self, value):
        """Helper async operation"""
        await asyncio.sleep(0.01)
        return value * 2
        
    @pytest.mark.asyncio
    async def test_parallel_execution_integration(self):
        """Test that parallel execution works with all components"""
        
        # Create multiple suites
        suite1 = TestSuite("parallel_suite_1")
        suite2 = TestSuite("parallel_suite_2")
        
        def slow_test_1(assertions, mocks, context):
            import time
            time.sleep(0.1)  # Simulate work
            context.set("suite_id", "suite_1")
            assertions.equal(1 + 1, 2)
            
        def slow_test_2(assertions, mocks, context):
            import time
            time.sleep(0.1)  # Simulate work
            context.set("suite_id", "suite_2")
            assertions.equal(2 + 2, 4)
            
        suite1.add_test(slow_test_1)
        suite2.add_test(slow_test_2)
        
        self.runner.add_suite(suite1)
        self.runner.add_suite(suite2)
        
        # Run in parallel
        import time
        start_time = time.time()
        results = await self.runner.run_all(parallel=True)
        end_time = time.time()
        
        # Verify parallel execution was faster than sequential
        duration = end_time - start_time
        assert duration < 0.15  # Should be less than sequential (0.2s)
        
        # Verify results
        assert results['summary']['total'] == 2
        assert results['summary']['passed'] == 2
        
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test that error handling works across all components"""
        
        suite = TestSuite("error_handling_suite")
        
        def test_with_exception(assertions, mocks, context):
            """Test that raises an unexpected exception"""
            raise ValueError("Unexpected error in test")
            
        def test_with_assertion_error(assertions, mocks, context):
            """Test that fails with assertion"""
            assertions.equal("actual", "expected", "Custom failure message")
            
        def test_with_mock_error(assertions, mocks, context):
            """Test that has mock-related error"""
            mock_obj = mocks.create_mock("error_mock")
            mock_obj.side_effect = RuntimeError("Mock error")
            
            # This should raise the mock error
            mock_obj()
            
        suite.add_test(test_with_exception)
        suite.add_test(test_with_assertion_error)
        suite.add_test(test_with_mock_error)
        
        self.runner.add_suite(suite)
        results = await self.runner.run_all()
        
        # Verify error handling
        assert results['summary']['total'] == 3
        assert results['summary']['passed'] == 0
        assert results['summary']['failed'] == 1  # Assertion failure
        assert results['summary']['errors'] == 2  # Exception and mock error
        
        # Check that error details are captured
        test_results = results['results']
        for result in test_results:
            assert result['error'] is not None
            assert result['traceback_str'] != ""
            
    @pytest.mark.asyncio
    async def test_complete_integration_workflow(self):
        """Test a complete workflow using all framework components"""
        
        # Create a comprehensive test framework
        framework = create_test_framework(
            console_output=True,
            json_output=os.path.join(self.temp_dir, "complete_results.json"),
            html_output=os.path.join(self.temp_dir, "complete_report.html")
        )
        
        # Create test suite with setup/teardown
        suite = TestSuite("complete_workflow_suite")
        
        # Suite setup
        def suite_setup(context):
            context.set("database", {"users": [], "posts": []})
            context.set("api_client", MockAPIClient())
            
        def suite_teardown(context):
            # Cleanup
            context.clear()
            
        suite.setup = suite_setup
        suite.teardown = suite_teardown
        
        # Test cases
        def test_user_creation(assertions, mocks, context):
            """Test user creation with database and API integration"""
            db = context.get("database")
            api_client = context.get("api_client")
            
            # Mock API call
            api_mock = mocks.create_mock("user_api")
            api_mock.create_user.return_value = {"id": 1, "name": "John"}
            
            # Simulate user creation
            user_data = api_mock.create_user({"name": "John"})
            db["users"].append(user_data)
            
            # Assertions
            assertions.equal(len(db["users"]), 1)
            assertions.equal(db["users"][0]["name"], "John")
            mocks.assert_called_once("user_api")
            
        def test_post_creation(assertions, mocks, context):
            """Test post creation with context sharing"""
            db = context.get("database")
            
            # Create a post
            post = {"id": 1, "title": "Test Post", "author_id": 1}
            db["posts"].append(post)
            
            # Verify
            assertions.equal(len(db["posts"]), 1)
            assertions.equal(db["posts"][0]["title"], "Test Post")
            
        async def test_async_operation(assertions, mocks, context):
            """Async test with all components"""
            await asyncio.sleep(0.01)
            
            async_mock = mocks.create_async_mock("async_service", return_value="success")
            result = await async_mock()
            
            assertions.equal(result, "success")
            context.set("async_result", result)
            
        suite.add_test(test_user_creation, tags=["integration", "database"])
        suite.add_test(test_post_creation, tags=["integration", "database"])
        suite.add_test(test_async_operation, tags=["async", "integration"])
        
        framework.add_suite(suite)
        
        # Run the complete workflow
        results = await framework.run_all()
        
        # Comprehensive verification
        assert results['summary']['total'] == 3
        assert results['summary']['passed'] == 3
        assert results['summary']['failed'] == 0
        assert results['summary']['errors'] == 0
        assert results['summary']['success_rate'] == 100.0
        
        # Verify integration statistics
        integration_stats = results['integration_stats']
        assert integration_stats['total_assertions'] > 0
        assert integration_stats['mocks_used'] > 0
        assert integration_stats['context_keys'] > 0
        
        # Verify reports were generated
        json_file = os.path.join(self.temp_dir, "complete_results.json")
        html_file = os.path.join(self.temp_dir, "complete_report.html")
        
        assert os.path.exists(json_file)
        assert os.path.exists(html_file)
        
        # Verify report contents
        with open(json_file, 'r') as f:
            json_results = json.load(f)
            
        assert json_results['test_framework'] == 'test-workflow'
        assert json_results['results']['summary']['total'] == 3


class MockAPIClient:
    """Mock API client for testing"""
    
    def __init__(self):
        self.requests = []
        
    def make_request(self, method, url, data=None):
        request = {"method": method, "url": url, "data": data}
        self.requests.append(request)
        return {"status": "success", "data": data}


class TestFrameworkUsability:
    """Test the usability and developer experience of the framework"""
    
    def test_framework_creation_shortcuts(self):
        """Test that framework creation shortcuts work properly"""
        
        # Test basic creation
        framework1 = create_test_framework()
        assert framework1 is not None
        assert hasattr(framework1, 'run_all')
        
        # Test with specific outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            framework2 = create_test_framework(
                console_output=True,
                json_output=f"{temp_dir}/results.json",
                html_output=f"{temp_dir}/report.html",
                junit_output=f"{temp_dir}/junit.xml"
            )
            assert framework2 is not None
            
    def test_decorator_based_test_definition(self):
        """Test that decorator-based test definition works"""
        
        from test_workflow.core import test_case, suite_setup, suite_teardown
        
        @test_case(tags=["unit", "fast"])
        def sample_test():
            pass
            
        @suite_setup
        def setup_function():
            pass
            
        @suite_teardown
        def teardown_function():
            pass
            
        # Verify decorators added metadata
        assert hasattr(sample_test, '_is_test_case')
        assert hasattr(sample_test, '_test_tags')
        assert sample_test._test_tags == ["unit", "fast"]
        
        assert hasattr(setup_function, '_is_suite_setup')
        assert hasattr(teardown_function, '_is_suite_teardown')
        
    @pytest.mark.asyncio
    async def test_fluent_assertion_interface(self):
        """Test that fluent assertion interface works properly"""
        
        framework = create_test_framework()
        suite = TestSuite("fluent_assertions_suite")
        
        def test_fluent_assertions(assertions, mocks, context):
            """Test using fluent assertion interface"""
            
            # Test fluent interface
            assertions.that(42).equals(42)
            assertions.that("hello").contains("ell")
            assertions.that([1, 2, 3]).has_length(3)
            assertions.that([]).is_empty()
            assertions.that([1]).is_not_empty()
            assertions.that(None).is_none()
            assertions.that("value").is_not_none()
            
        suite.add_test(test_fluent_assertions)
        framework.add_suite(suite)
        
        results = await framework.run_all()
        
        # Verify fluent assertions work
        assert results['summary']['total'] == 1
        assert results['summary']['passed'] == 1
        
    def test_error_messages_are_helpful(self):
        """Test that error messages provide helpful information"""
        
        assertions = AssertionFramework()
        assertions.reset_for_test("helpful_errors_test")
        
        # Test detailed error message
        try:
            assertions.equal("actual_value", "expected_value", "Custom message")
            assert False, "Should have raised AssertionError"
        except Exception as e:
            error_message = str(e)
            assert "Custom message" in error_message
            assert "actual_value" in error_message
            assert "expected_value" in error_message
            assert "Expected:" in error_message
            assert "Actual:" in error_message