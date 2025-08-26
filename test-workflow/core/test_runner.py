"""
Test Runner Core - Integrates with assertion library and mock framework
Part of the comprehensive test-workflow framework for SPARC methodology
"""

import asyncio
import inspect
import time
import traceback
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from .assertions import AssertionFramework
from .mocks import MockFramework
from .context import TestContext
from .reporter import TestReporter


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"  
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Comprehensive test result with integration data"""
    name: str
    status: TestStatus
    duration: float
    message: str = ""
    error: Optional[Exception] = None
    traceback_str: str = ""
    assertions_count: int = 0
    mocks_used: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for reporting"""
        return {
            'name': self.name,
            'status': self.status.value,
            'duration': self.duration,
            'message': self.message,
            'error': str(self.error) if self.error else None,
            'traceback': self.traceback_str,
            'assertions_count': self.assertions_count,
            'mocks_used': self.mocks_used,
            'context_data': self.context_data,
            'tags': self.tags
        }


@dataclass
class TestSuite:
    """Test suite with integrated components"""
    name: str
    tests: List[Callable] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    before_each: Optional[Callable] = None
    after_each: Optional[Callable] = None
    context: Optional[TestContext] = None
    tags: List[str] = field(default_factory=list)
    
    def add_test(self, test_func: Callable, tags: List[str] = None) -> None:
        """Add test with optional tags"""
        if tags:
            test_func._test_tags = tags
        self.tests.append(test_func)


class IntegratedTestRunner:
    """
    Core test runner that integrates:
    - Assertion library for test validation
    - Mock framework for test isolation
    - Test context for shared data
    - Reporter system for comprehensive results
    """
    
    def __init__(
        self,
        assertion_framework: Optional[AssertionFramework] = None,
        mock_framework: Optional[MockFramework] = None,
        reporter: Optional[TestReporter] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.assertions = assertion_framework or AssertionFramework()
        self.mocks = mock_framework or MockFramework()
        self.reporter = reporter or TestReporter()
        self.logger = logger or self._setup_logger()
        
        self.suites: List[TestSuite] = []
        self.results: List[TestResult] = []
        self.global_context = TestContext("global")
        self.current_suite: Optional[TestSuite] = None
        self.current_test: Optional[str] = None
        
        # Integration hooks
        self.before_run_hooks: List[Callable] = []
        self.after_run_hooks: List[Callable] = []
        self.test_discovery_hooks: List[Callable] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup integrated logger"""
        logger = logging.getLogger('test_runner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def add_suite(self, suite: TestSuite) -> None:
        """Add test suite to runner"""
        if suite.context is None:
            suite.context = TestContext(f"suite_{len(self.suites)}")
        self.suites.append(suite)
        
    def create_suite(
        self, 
        name: str, 
        setup: Callable = None,
        teardown: Callable = None,
        tags: List[str] = None
    ) -> TestSuite:
        """Create and add new test suite"""
        suite = TestSuite(
            name=name,
            setup=setup,
            teardown=teardown,
            tags=tags or []
        )
        self.add_suite(suite)
        return suite
        
    def register_hook(self, hook_type: str, func: Callable) -> None:
        """Register integration hooks"""
        hooks_map = {
            'before_run': self.before_run_hooks,
            'after_run': self.after_run_hooks,
            'test_discovery': self.test_discovery_hooks
        }
        
        if hook_type in hooks_map:
            hooks_map[hook_type].append(func)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
            
    def discover_tests(
        self, 
        module_or_path: Union[str, Any],
        pattern: str = "test_*.py"
    ) -> List[TestSuite]:
        """Discover tests with integration hooks"""
        discovered_suites = []
        
        # Run discovery hooks
        for hook in self.test_discovery_hooks:
            try:
                hook_result = hook(module_or_path, pattern)
                if hook_result:
                    discovered_suites.extend(hook_result)
            except Exception as e:
                self.logger.warning(f"Discovery hook failed: {e}")
                
        return discovered_suites
        
    async def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run test suite with full integration"""
        suite_results = []
        self.current_suite = suite
        
        try:
            # Suite setup
            if suite.setup:
                await self._run_with_context(suite.setup, suite.context)
                
            # Run each test
            for test_func in suite.tests:
                result = await self._run_test(test_func, suite)
                suite_results.append(result)
                
        except Exception as e:
            self.logger.error(f"Suite {suite.name} failed: {e}")
            error_result = TestResult(
                name=f"{suite.name}_setup_error",
                status=TestStatus.ERROR,
                duration=0.0,
                error=e,
                traceback_str=traceback.format_exc()
            )
            suite_results.append(error_result)
            
        finally:
            # Suite teardown
            if suite.teardown:
                try:
                    await self._run_with_context(suite.teardown, suite.context)
                except Exception as e:
                    self.logger.error(f"Suite {suite.name} teardown failed: {e}")
                    
            self.current_suite = None
            
        return suite_results
        
    async def _run_test(self, test_func: Callable, suite: TestSuite) -> TestResult:
        """Run individual test with full integration support"""
        test_name = f"{suite.name}.{test_func.__name__}"
        self.current_test = test_name
        start_time = time.time()
        
        # Initialize test context
        test_context = TestContext(test_name)
        test_context.inherit_from(suite.context)
        test_context.inherit_from(self.global_context)
        
        # Reset assertions and mocks for this test
        self.assertions.reset_for_test(test_name)
        self.mocks.reset_for_test(test_name)
        
        try:
            # Before each hook
            if suite.before_each:
                await self._run_with_context(suite.before_each, test_context)
                
            # Run the actual test with integrated components
            if inspect.iscoroutinefunction(test_func):
                await test_func(
                    assertions=self.assertions,
                    mocks=self.mocks,
                    context=test_context
                )
            else:
                test_func(
                    assertions=self.assertions,
                    mocks=self.mocks,
                    context=test_context
                )
                
            # Check if any assertions failed
            assertion_results = self.assertions.get_results(test_name)
            if any(not result.passed for result in assertion_results):
                status = TestStatus.FAILED
                message = f"Assertions failed: {len([r for r in assertion_results if not r.passed])}"
            else:
                status = TestStatus.PASSED
                message = f"All {len(assertion_results)} assertions passed"
                
        except Exception as e:
            status = TestStatus.ERROR
            message = str(e)
            error = e
            traceback_str = traceback.format_exc()
            self.logger.error(f"Test {test_name} error: {e}")
            
        else:
            error = None
            traceback_str = ""
            
        finally:
            # After each hook
            if suite.after_each:
                try:
                    await self._run_with_context(suite.after_each, test_context)
                except Exception as e:
                    self.logger.warning(f"After each hook failed for {test_name}: {e}")
                    
            duration = time.time() - start_time
            self.current_test = None
            
        # Create comprehensive result
        result = TestResult(
            name=test_name,
            status=status,
            duration=duration,
            message=message,
            error=error,
            traceback_str=traceback_str,
            assertions_count=len(self.assertions.get_results(test_name)),
            mocks_used=list(self.mocks.get_used_mocks(test_name)),
            context_data=test_context.get_all(),
            tags=getattr(test_func, '_test_tags', [])
        )
        
        return result
        
    async def _run_with_context(self, func: Callable, context: TestContext) -> Any:
        """Run function with context injection"""
        if inspect.iscoroutinefunction(func):
            return await func(context=context)
        else:
            return func(context=context)
            
    async def run_all(
        self, 
        filter_tags: List[str] = None,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """Run all test suites with comprehensive integration"""
        
        # Run before run hooks
        for hook in self.before_run_hooks:
            try:
                if inspect.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            except Exception as e:
                self.logger.warning(f"Before run hook failed: {e}")
                
        start_time = time.time()
        all_results = []
        
        try:
            if parallel and len(self.suites) > 1:
                # Parallel execution
                tasks = []
                for suite in self.suites:
                    if self._should_run_suite(suite, filter_tags):
                        tasks.append(self.run_suite(suite))
                        
                suite_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for results in suite_results:
                    if isinstance(results, Exception):
                        self.logger.error(f"Suite execution failed: {results}")
                        continue
                    all_results.extend(results)
                    
            else:
                # Sequential execution
                for suite in self.suites:
                    if self._should_run_suite(suite, filter_tags):
                        results = await self.run_suite(suite)
                        all_results.extend(results)
                        
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            
        finally:
            duration = time.time() - start_time
            
            # Run after run hooks
            for hook in self.after_run_hooks:
                try:
                    if inspect.iscoroutinefunction(hook):
                        await hook(self, all_results)
                    else:
                        hook(self, all_results)
                except Exception as e:
                    self.logger.warning(f"After run hook failed: {e}")
                    
        self.results.extend(all_results)
        
        # Generate comprehensive report
        report = self._generate_report(all_results, duration)
        
        # Send to reporter
        await self.reporter.generate_report(report)
        
        return report
        
    def _should_run_suite(self, suite: TestSuite, filter_tags: List[str]) -> bool:
        """Check if suite should run based on tag filtering"""
        if not filter_tags:
            return True
            
        return any(tag in suite.tags for tag in filter_tags)
        
    def _generate_report(self, results: List[TestResult], duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(results)
        passed = len([r for r in results if r.status == TestStatus.PASSED])
        failed = len([r for r in results if r.status == TestStatus.FAILED])
        errors = len([r for r in results if r.status == TestStatus.ERROR])
        skipped = len([r for r in results if r.status == TestStatus.SKIPPED])
        
        return {
            'summary': {
                'total': total_tests,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'skipped': skipped,
                'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
                'duration': duration
            },
            'suites': len(self.suites),
            'results': [result.to_dict() for result in results],
            'integration_stats': {
                'total_assertions': sum(r.assertions_count for r in results),
                'mocks_used': len(set().union(*[r.mocks_used for r in results])),
                'context_keys': len(self.global_context.get_all())
            },
            'timestamp': time.time()
        }


# Integration helper functions
def create_integrated_runner(
    with_console_reporter: bool = True,
    with_json_reporter: bool = False,
    log_level: str = "INFO"
) -> IntegratedTestRunner:
    """Create fully integrated test runner with common configuration"""
    
    from .reporter import ConsoleReporter, JsonReporter, CompositeReporter
    
    reporters = []
    if with_console_reporter:
        reporters.append(ConsoleReporter())
    if with_json_reporter:
        reporters.append(JsonReporter())
        
    reporter = CompositeReporter(reporters) if len(reporters) > 1 else reporters[0]
    
    logger = logging.getLogger('integrated_test_runner')
    logger.setLevel(getattr(logging, log_level))
    
    return IntegratedTestRunner(
        assertion_framework=AssertionFramework(),
        mock_framework=MockFramework(),
        reporter=reporter,
        logger=logger
    )


# Decorator for easy test registration
def test_case(tags: List[str] = None):
    """Decorator to mark functions as test cases"""
    def decorator(func):
        func._is_test_case = True
        func._test_tags = tags or []
        return func
    return decorator


def suite_setup(func):
    """Decorator to mark suite setup function"""
    func._is_suite_setup = True
    return func


def suite_teardown(func):
    """Decorator to mark suite teardown function"""
    func._is_suite_teardown = True
    return func