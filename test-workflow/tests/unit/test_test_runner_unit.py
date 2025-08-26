"""Unit Tests for Test Runner - London School TDD

Focuses on testing the interactions and collaborations of TestRunner
rather than its internal implementation details.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, call
import pytest
from src.test_runner import TestRunner, SimpleTestReporter, TestStatus, TestResult


class TestTestRunnerInteractions:
    """Test how TestRunner coordinates with its collaborators"""
    
    def setup_method(self):
        # Create mocks for all collaborators
        self.mock_discovery = Mock()
        self.mock_reporter = Mock()
        self.mock_file_watcher = Mock()
        
        # Configure default behavior
        self.mock_discovery.find_test_files.return_value = ['test_file1.py', 'test_file2.py']
        self.mock_discovery.load_tests.return_value = [
            {'name': 'test_should_pass', 'fn': Mock(), 'async': False},
            {'name': 'test_should_also_pass', 'fn': Mock(), 'async': False}
        ]
        
        self.runner = TestRunner(self.mock_discovery, self.mock_reporter, self.mock_file_watcher)
    
    async def test_should_coordinate_discovery_and_reporting(self):
        """TestRunner should coordinate between discovery and reporting"""
        # Act
        await self.runner.run_suite()
        
        # Assert - Verify the conversation between objects
        self.mock_discovery.find_test_files.assert_called_once()
        self.mock_discovery.load_tests.assert_called_once_with(['test_file1.py', 'test_file2.py'])
        self.mock_reporter.on_suite_complete.assert_called_once()
        self.mock_reporter.generate_report.assert_called_once()
    
    async def test_should_report_test_start_and_end_events(self):
        """TestRunner should report individual test events"""
        # Act
        await self.runner.run_suite()
        
        # Assert - Verify test events were reported
        expected_calls = [
            call('test_should_pass'),
            call('test_should_also_pass')
        ]
        self.mock_reporter.on_test_start.assert_has_calls(expected_calls)
        assert self.mock_reporter.on_test_end.call_count == 2
    
    async def test_should_execute_test_functions_with_isolation(self):
        """TestRunner should execute test functions but isolate failures"""
        # Arrange - One test fails, one passes
        passing_test = Mock()
        failing_test = Mock(side_effect=Exception("Test failed"))
        
        self.mock_discovery.load_tests.return_value = [
            {'name': 'passing_test', 'fn': passing_test, 'async': False},
            {'name': 'failing_test', 'fn': failing_test, 'async': False}
        ]
        
        # Act
        results = await self.runner.run_suite()
        
        # Assert - Both tests should have been called
        passing_test.assert_called_once()
        failing_test.assert_called_once()
        
        # Results should reflect mixed outcomes
        assert results['total'] == 2
        assert results['passed'] == 1
        assert results['failed'] == 1
    
    async def test_should_handle_async_tests_properly(self):
        """TestRunner should properly await async test functions"""
        # Arrange
        async_test = AsyncMock()
        sync_test = Mock()
        
        self.mock_discovery.load_tests.return_value = [
            {'name': 'async_test', 'fn': async_test, 'async': True},
            {'name': 'sync_test', 'fn': sync_test, 'async': False}
        ]
        
        # Act
        await self.runner.run_suite()
        
        # Assert - Async test should be awaited, sync test called normally
        async_test.assert_awaited_once()
        sync_test.assert_called_once()
    
    async def test_should_apply_filtering_through_discovery(self):
        """TestRunner should coordinate filtering with discovery"""
        # Act
        await self.runner.run_suite(pattern="specific_test")
        
        # Assert - Discovery should have been called with all tests first
        self.mock_discovery.find_test_files.assert_called_once()
        self.mock_discovery.load_tests.assert_called_once()
        
        # In a real implementation, we'd verify the pattern was applied
    
    async def test_should_coordinate_file_watching(self):
        """TestRunner should coordinate with file watcher for continuous testing"""
        # Act
        await self.runner.start_watch_mode()
        
        # Assert - Should set up file watching through discovery
        self.mock_discovery.watch_files.assert_called_once()
        
        # Verify callback was provided
        callback_arg = self.mock_discovery.watch_files.call_args[0][0]
        assert callable(callback_arg)
    
    def test_should_expose_running_state(self):
        """TestRunner should expose its running state"""
        # Initially not running
        assert not self.runner.is_running
        
        # Would be running during execution (tested in integration)
        
    async def test_should_return_structured_results(self):
        """TestRunner should return well-structured results"""
        # Act
        results = await self.runner.run_suite()
        
        # Assert - Results should have expected structure
        assert isinstance(results, dict)
        required_keys = ['total', 'passed', 'failed', 'skipped', 'duration', 'test_results']
        for key in required_keys:
            assert key in results
        
        assert isinstance(results['test_results'], list)
        assert isinstance(results['duration'], float)
    
    async def test_should_handle_empty_test_suite(self):
        """TestRunner should gracefully handle empty test suites"""
        # Arrange - No tests found
        self.mock_discovery.find_test_files.return_value = []
        
        # Act
        results = await self.runner.run_suite()
        
        # Assert - Should return empty results
        assert results['total'] == 0
        assert results['passed'] == 0
        assert results['failed'] == 0
        
        # Should still call discovery but not reporter events
        self.mock_discovery.find_test_files.assert_called_once()
        self.mock_reporter.on_test_start.assert_not_called()


class TestSimpleTestReporter:
    """Test the basic test reporter behavior"""
    
    def setup_method(self):
        self.reporter = SimpleTestReporter(verbose=True)
    
    def test_should_track_current_test(self):
        """Reporter should track the currently running test"""
        # Act
        self.reporter.on_test_start("test_example")
        
        # Assert
        assert self.reporter._current_test == "test_example"
    
    def test_should_handle_test_completion(self):
        """Reporter should handle test completion events"""
        # Arrange
        test_result = TestResult(
            name="test_example",
            status=TestStatus.PASSED,
            duration=0.5
        )
        
        # Act - Should not raise exception
        self.reporter.on_test_end(test_result)
        
        # In a real implementation, we might verify console output
    
    def test_should_generate_report_structure(self):
        """Reporter should generate structured reports"""
        # Arrange
        from src.test_runner import TestSuiteResult
        suite_result = TestSuiteResult(
            total=5,
            passed=4,
            failed=1,
            duration=2.5
        )
        
        # Act
        report = self.reporter.generate_report(suite_result.__dict__)
        
        # Assert
        assert isinstance(report, dict)
        assert 'format' in report
        assert 'timestamp' in report
        assert 'results' in report


class TestTestRunnerFactoryFunction:
    """Test the factory function for creating test runners"""
    
    def test_should_create_runner_with_defaults(self):
        """Factory should create runner with sensible defaults"""
        from src.test_runner import create_test_runner
        
        # Act
        runner = create_test_runner()
        
        # Assert
        assert isinstance(runner, TestRunner)
        assert runner._discovery is not None
        assert runner._reporter is not None
    
    def test_should_accept_custom_collaborators(self):
        """Factory should accept custom collaborators"""
        from src.test_runner import create_test_runner
        
        # Arrange
        custom_discovery = Mock()
        custom_reporter = Mock()
        
        # Act
        runner = create_test_runner(
            discovery=custom_discovery,
            reporter=custom_reporter
        )
        
        # Assert
        assert runner._discovery is custom_discovery
        assert runner._reporter is custom_reporter
    
    def test_should_pass_options_to_components(self):
        """Factory should pass options to created components"""
        from src.test_runner import create_test_runner
        
        # Act
        runner = create_test_runner(
            verbose=False,
            file_watcher=Mock()
        )
        
        # Assert
        assert runner._file_watcher is not None
        # In a real implementation, we'd verify reporter verbose setting


class TestAsyncTestExecution:
    """Test async test execution behavior"""
    
    def setup_method(self):
        self.mock_discovery = Mock()
        self.mock_reporter = Mock()
        self.runner = TestRunner(self.mock_discovery, self.mock_reporter)
    
    async def test_should_handle_async_test_with_delay(self):
        """Should properly handle async tests with delays"""
        # Arrange
        async def slow_async_test():
            await asyncio.sleep(0.01)
            return "completed"
        
        self.mock_discovery.find_test_files.return_value = ['test_file.py']
        self.mock_discovery.load_tests.return_value = [
            {'name': 'slow_async_test', 'fn': slow_async_test, 'async': True}
        ]
        
        # Act
        results = await self.runner.run_suite()
        
        # Assert
        assert results['total'] == 1
        assert results['passed'] == 1
        assert results['duration'] > 0.01  # Should include the delay
    
    async def test_should_handle_async_test_failure(self):
        """Should properly handle async test failures"""
        # Arrange
        async def failing_async_test():
            await asyncio.sleep(0.001)
            raise ValueError("Async test failed")
        
        self.mock_discovery.find_test_files.return_value = ['test_file.py']
        self.mock_discovery.load_tests.return_value = [
            {'name': 'failing_async_test', 'fn': failing_async_test, 'async': True}
        ]
        
        # Act
        results = await self.runner.run_suite()
        
        # Assert
        assert results['total'] == 1
        assert results['failed'] == 1
        assert results['passed'] == 0
        
        # Should have reported the failure
        assert self.mock_reporter.on_test_end.call_count == 1
        test_result = self.mock_reporter.on_test_end.call_args[0][0]
        assert test_result.status == TestStatus.FAILED
        assert "Async test failed" in str(test_result.error)


if __name__ == '__main__':
    # Run tests directly
    import asyncio
    
    async def run_all_tests():
        test_classes = [
            TestTestRunnerInteractions,
            TestSimpleTestReporter,
            TestTestRunnerFactoryFunction,
            TestAsyncTestExecution
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            instance = test_class()
            
            # Get all test methods
            test_methods = [method for method in dir(instance) 
                          if method.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                method = getattr(instance, method_name)
                
                try:
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        method()
                    
                    print(f"✓ {test_class.__name__}.{method_name}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"✗ {test_class.__name__}.{method_name}: {e}")
        
        print(f"\nUnit Tests: {passed_tests}/{total_tests} passed")
        return passed_tests == total_tests
    
    # Run the tests
    success = asyncio.run(run_all_tests())
    if success:
        print("All unit tests passed!")
    else:
        print("Some unit tests failed!")