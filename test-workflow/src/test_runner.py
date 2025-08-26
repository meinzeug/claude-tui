"""Test Runner Implementation - London School TDD

Implemented following outside-in TDD:
1. Started with acceptance tests defining behavior
2. Used mocks to define collaborator contracts
3. Focus on coordinating interactions between objects
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    status: TestStatus
    duration: float = 0.0
    error: Optional[Exception] = None
    output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Complete test suite results"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestRunner:
    """Async Test Runner with London School coordination patterns
    
    Focuses on coordinating with collaborators:
    - TestDiscovery: finds and loads tests
    - TestReporter: reports progress and results
    - FileWatcher: monitors file changes (optional)
    
    Responsibilities:
    1. Coordinate test discovery and execution
    2. Handle async and sync test functions
    3. Isolate test failures
    4. Report detailed results
    5. Support filtering and watching
    """
    
    def __init__(self, discovery, reporter, file_watcher=None):
        """Initialize with collaborators (London School dependency injection)"""
        self._discovery = discovery
        self._reporter = reporter
        self._file_watcher = file_watcher
        self._running = False
        self._watch_mode = False
    
    async def run_suite(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """Run complete test suite
        
        Coordinates with discovery and reporter to:
        1. Find test files
        2. Load test functions
        3. Execute tests (async and sync)
        4. Collect and report results
        """
        suite_start_time = time.time()
        self._running = True
        
        try:
            # Coordinate with discovery to find tests
            test_files = self._discovery.find_test_files()
            if not test_files:
                return self._empty_result()
            
            # Load tests from discovered files
            tests = self._discovery.load_tests(test_files)
            
            # Apply filtering if pattern provided
            if pattern:
                tests = [t for t in tests if pattern in t['name']]
            
            # Initialize suite result tracking
            suite_result = TestSuiteResult(total=len(tests))
            
            # Execute each test with isolation
            for test_spec in tests:
                result = await self._execute_single_test(test_spec)
                suite_result.test_results.append(result)
                
                # Update counters based on result
                if result.status == TestStatus.PASSED:
                    suite_result.passed += 1
                elif result.status == TestStatus.FAILED:
                    suite_result.failed += 1
                elif result.status == TestStatus.SKIPPED:
                    suite_result.skipped += 1
            
            # Calculate total duration
            suite_result.duration = time.time() - suite_start_time
            
            # Report completion to collaborator
            self._reporter.on_suite_complete(suite_result)
            
            # Generate final report
            self._reporter.generate_report(suite_result.__dict__)
            
            return {
                'total': suite_result.total,
                'passed': suite_result.passed,
                'failed': suite_result.failed,
                'skipped': suite_result.skipped,
                'duration': suite_result.duration,
                'test_results': [r.__dict__ for r in suite_result.test_results]
            }
            
        finally:
            self._running = False
    
    async def _execute_single_test(self, test_spec: Dict[str, Any]) -> TestResult:
        """Execute a single test with proper isolation and error handling"""
        test_name = test_spec['name']
        test_fn = test_spec['fn']
        is_async = test_spec.get('async', False)
        
        # Report test start to collaborator
        self._reporter.on_test_start(test_name)
        
        result = TestResult(name=test_name, status=TestStatus.RUNNING)
        start_time = time.time()
        
        try:
            # Execute test function (async or sync)
            if is_async or asyncio.iscoroutinefunction(test_fn):
                await test_fn()
            else:
                test_fn()
            
            result.status = TestStatus.PASSED
            
        except Exception as e:
            # Isolate failure - don't let it stop other tests
            result.status = TestStatus.FAILED
            result.error = e
            result.output = str(e)
        
        finally:
            result.duration = time.time() - start_time
            
            # Report test end to collaborator
            self._reporter.on_test_end(result)
        
        return result
    
    async def start_watch_mode(self) -> None:
        """Start file watching mode for continuous testing
        
        Coordinates with file watcher to re-run tests on changes.
        """
        if not self._file_watcher:
            raise RuntimeError("File watcher not configured")
        
        self._watch_mode = True
        
        # Define callback for file changes
        async def on_file_change(changed_files: List[str]) -> None:
            """Handle file changes by re-running tests"""
            if self._running:
                return  # Don't start new run if already running
            
            print(f"Files changed: {changed_files}")
            await self.run_suite()
        
        # Set up file watching through collaborator
        self._discovery.watch_files(on_file_change)
    
    def stop_watch_mode(self) -> None:
        """Stop file watching mode"""
        self._watch_mode = False
        if hasattr(self._discovery, 'stop_watching'):
            self._discovery.stop_watching()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when no tests found"""
        return {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'duration': 0.0,
            'test_results': []
        }
    
    @property
    def is_running(self) -> bool:
        """Check if test runner is currently executing"""
        return self._running
    
    @property
    def is_watching(self) -> bool:
        """Check if file watching is active"""
        return self._watch_mode


class SimpleTestReporter:
    """Basic test reporter implementation
    
    Provides simple console output and basic reporting.
    Can be replaced with more sophisticated reporters.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._current_test = None
    
    def on_test_start(self, test_name: str) -> None:
        """Called when a test starts"""
        self._current_test = test_name
        if self.verbose:
            print(f"Starting: {test_name}")
    
    def on_test_end(self, result: TestResult) -> None:
        """Called when a test completes"""
        if self.verbose:
            status_symbol = "✓" if result.status == TestStatus.PASSED else "✗"
            duration_ms = result.duration * 1000
            print(f"{status_symbol} {result.name} ({duration_ms:.2f}ms)")
            
            if result.error and result.status == TestStatus.FAILED:
                print(f"  Error: {result.error}")
    
    def on_suite_complete(self, suite_result: TestSuiteResult) -> None:
        """Called when entire suite completes"""
        print(f"\nSuite completed:")
        print(f"  Total: {suite_result.total}")
        print(f"  Passed: {suite_result.passed}")
        print(f"  Failed: {suite_result.failed}")
        print(f"  Duration: {suite_result.duration:.2f}s")
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report"""
        # In a real implementation, this might generate JSON, XML, HTML reports
        return {
            'format': 'simple',
            'timestamp': time.time(),
            'results': results
        }


# Factory function for easy test runner creation
def create_test_runner(discovery=None, reporter=None, **options) -> TestRunner:
    """Factory function to create test runner with sensible defaults
    
    Follows London School pattern of injecting dependencies
    while providing convenient defaults.
    """
    if reporter is None:
        reporter = SimpleTestReporter(verbose=options.get('verbose', True))
    
    if discovery is None:
        # Import here to avoid circular dependencies
        try:
            from .test_discovery import TestDiscovery
        except ImportError:
            from test_discovery import TestDiscovery
        discovery = TestDiscovery()
    
    file_watcher = options.get('file_watcher')
    
    return TestRunner(discovery, reporter, file_watcher)


# Convenience function for running tests quickly
async def run_tests(pattern: Optional[str] = None, **options) -> Dict[str, Any]:
    """Quick function to discover and run tests
    
    Usage:
        results = await run_tests()
        results = await run_tests(pattern='user')
        results = await run_tests(verbose=False)
    """
    runner = create_test_runner(**options)
    return await runner.run_suite(pattern=pattern)
