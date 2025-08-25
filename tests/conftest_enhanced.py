"""
Enhanced pytest configuration with comprehensive async support and coverage reporting.

This configuration extends the base conftest.py with:
- Advanced async testing support
- Enhanced coverage reporting
- Performance monitoring
- Custom test markers and hooks
- Test data management
- Error reporting and debugging
"""

import asyncio
import os
import sys
import time
import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import AsyncMock, MagicMock

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import project modules for testing
from tests.fixtures.external_service_mocks import (
    MockClaudeCodeIntegration,
    MockClaudeFlowIntegration,
    MockDatabaseService,
    create_mock_service_suite
)


# ============================================================================
# Async Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Configure event loop policy for all tests."""
    if sys.platform == "win32":
        # Use ProactorEventLoop on Windows for better async support
        return asyncio.WindowsProactorEventLoopPolicy()
    else:
        # Use default policy on Unix systems
        return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    """Create and configure the main event loop for testing."""
    asyncio.set_event_loop_policy(event_loop_policy)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Configure loop for testing
    loop.set_debug(True)  # Enable debug mode for better error reporting
    
    yield loop
    
    # Cleanup
    try:
        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop)
        if pending:
            for task in pending:
                task.cancel()
            # Wait for all tasks to complete cancellation
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.close()


# ============================================================================
# Performance Monitoring and Metrics
# ============================================================================

class TestPerformanceMonitor:
    """Monitor test performance and resource usage."""
    
    def __init__(self):
        self.test_metrics = {}
        self.slow_tests = []
        self.memory_usage = []
        
    def start_test(self, test_name: str):
        """Start monitoring a test."""
        import psutil
        process = psutil.Process(os.getpid())
        
        self.test_metrics[test_name] = {
            "start_time": time.time(),
            "start_memory": process.memory_info().rss,
            "start_cpu": process.cpu_percent()
        }
    
    def end_test(self, test_name: str):
        """End monitoring and record metrics."""
        if test_name not in self.test_metrics:
            return
        
        import psutil
        process = psutil.Process(os.getpid())
        
        metrics = self.test_metrics[test_name]
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        duration = end_time - metrics["start_time"]
        memory_delta = end_memory - metrics["start_memory"]
        
        metrics.update({
            "duration": duration,
            "memory_delta_mb": memory_delta / 1024 / 1024,
            "peak_memory_mb": end_memory / 1024 / 1024
        })
        
        # Track slow tests (>5 seconds)
        if duration > 5.0:
            self.slow_tests.append({
                "test_name": test_name,
                "duration": duration,
                "memory_delta_mb": metrics["memory_delta_mb"]
            })
        
        self.memory_usage.append(end_memory / 1024 / 1024)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.test_metrics:
            return {}
        
        durations = [m["duration"] for m in self.test_metrics.values() if "duration" in m]
        memory_deltas = [m["memory_delta_mb"] for m in self.test_metrics.values() if "memory_delta_mb" in m]
        
        return {
            "total_tests": len(self.test_metrics),
            "slow_tests_count": len(self.slow_tests),
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "average_memory_delta_mb": sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            "max_memory_usage_mb": max(self.memory_usage) if self.memory_usage else 0,
            "slow_tests": self.slow_tests
        }


@pytest.fixture(scope="session")
def performance_monitor():
    """Provide performance monitoring for all tests."""
    monitor = TestPerformanceMonitor()
    yield monitor
    
    # Print performance summary at the end
    summary = monitor.get_summary()
    if summary:
        print("\n" + "="*80)
        print("TEST PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Slow tests (>5s): {summary['slow_tests_count']}")
        print(f"Average duration: {summary['average_duration']:.3f}s")
        print(f"Max duration: {summary['max_duration']:.3f}s")
        print(f"Average memory delta: {summary['average_memory_delta_mb']:.2f}MB")
        print(f"Peak memory usage: {summary['max_memory_usage_mb']:.2f}MB")
        
        if summary['slow_tests']:
            print("\nSlowest tests:")
            for test in sorted(summary['slow_tests'], key=lambda x: x['duration'], reverse=True)[:5]:
                print(f"  {test['test_name']}: {test['duration']:.3f}s ({test['memory_delta_mb']:.2f}MB)")


# ============================================================================
# Enhanced Async Support
# ============================================================================

@pytest_asyncio.fixture
async def async_session_maker():
    """Create async session maker for database testing."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    
    # Create in-memory SQLite database for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        },
    )
    
    # Create all tables
    try:
        from src.database.models import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except ImportError:
        # If models not available, skip table creation
        pass
    
    # Create session maker
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    yield async_session
    
    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture
async def async_db_session(async_session_maker):
    """Provide async database session for individual tests."""
    async with async_session_maker() as session:
        yield session
        await session.rollback()


# ============================================================================
# Mock Service Configurations
# ============================================================================

@pytest.fixture(scope="session")
def mock_services_config():
    """Configuration for mock services."""
    return {
        "claude_code": {
            "simulate_errors": False,
            "response_delay": 0.01,  # Fast for tests
            "error_rate": 0.02  # 2% error rate
        },
        "claude_flow": {
            "simulate_errors": False,
            "response_delay": 0.02,
            "error_rate": 0.01  # 1% error rate
        },
        "database": {
            "simulate_errors": False,
            "connection_delay": 0.005,  # Very fast for tests
            "error_rate": 0.01
        }
    }


@pytest.fixture
def mock_service_suite(mock_services_config):
    """Provide complete suite of mock services."""
    return create_mock_service_suite(
        simulate_errors=mock_services_config.get("simulate_global_errors", False)
    )


@pytest.fixture
def mock_service_suite_with_errors(mock_services_config):
    """Provide mock services that simulate errors for error handling tests."""
    return create_mock_service_suite(simulate_errors=True)


# ============================================================================
# Test Data Management
# ============================================================================

@pytest.fixture(scope="session")
def test_data_manager():
    """Manage test data across test session."""
    class TestDataManager:
        def __init__(self):
            self.data_cache = {}
            self.cleanup_tasks = []
        
        def cache_data(self, key: str, data: Any):
            """Cache data for reuse across tests."""
            self.data_cache[key] = data
        
        def get_cached_data(self, key: str, default=None):
            """Get cached data."""
            return self.data_cache.get(key, default)
        
        def register_cleanup(self, cleanup_func):
            """Register cleanup function to run at session end."""
            self.cleanup_tasks.append(cleanup_func)
        
        def cleanup_all(self):
            """Run all cleanup tasks."""
            for cleanup_func in self.cleanup_tasks:
                try:
                    cleanup_func()
                except Exception as e:
                    print(f"Cleanup error: {e}")
    
    manager = TestDataManager()
    yield manager
    manager.cleanup_all()


# ============================================================================
# Enhanced Error Reporting
# ============================================================================

@pytest.fixture
def error_reporter():
    """Enhanced error reporting for tests."""
    class ErrorReporter:
        def __init__(self):
            self.errors = []
            self.warnings = []
        
        def record_error(self, test_name: str, error: Exception, context: Dict = None):
            """Record an error with context."""
            self.errors.append({
                "test_name": test_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "timestamp": time.time()
            })
        
        def record_warning(self, test_name: str, message: str, context: Dict = None):
            """Record a warning with context."""
            self.warnings.append({
                "test_name": test_name,
                "message": message,
                "context": context or {},
                "timestamp": time.time()
            })
        
        def get_report(self) -> Dict[str, Any]:
            """Get error report."""
            return {
                "errors": self.errors,
                "warnings": self.warnings,
                "error_count": len(self.errors),
                "warning_count": len(self.warnings)
            }
    
    return ErrorReporter()


# ============================================================================
# Custom Test Markers and Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "async_test: mark test as async")
    config.addinivalue_line("markers", "slow_test: mark test as slow running")
    config.addinivalue_line("markers", "requires_external: mark test as requiring external services")
    config.addinivalue_line("markers", "memory_intensive: mark test as memory intensive")
    config.addinivalue_line("markers", "cpu_intensive: mark test as CPU intensive")
    config.addinivalue_line("markers", "integration_test: mark as integration test")
    config.addinivalue_line("markers", "load_test: mark as load/performance test")
    
    # Enhanced coverage configuration
    if config.getoption("--cov"):
        print("\n🔍 Enhanced coverage reporting enabled")
        print("📊 Performance monitoring active")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers and organize tests."""
    for item in items:
        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)
            item.add_marker(pytest.mark.asyncio)
        
        # Auto-mark tests based on naming patterns
        test_name = item.name.lower()
        test_path = str(item.fspath).lower()
        
        if any(keyword in test_name for keyword in ["performance", "load", "benchmark", "stress"]):
            item.add_marker(pytest.mark.slow_test)
            item.add_marker(pytest.mark.load_test)
        
        if "integration" in test_path or "integration" in test_name:
            item.add_marker(pytest.mark.integration_test)
        
        if any(keyword in test_name for keyword in ["memory", "resource", "usage"]):
            item.add_marker(pytest.mark.memory_intensive)
        
        if any(keyword in test_name for keyword in ["concurrent", "parallel", "cpu"]):
            item.add_marker(pytest.mark.cpu_intensive)
        
        if any(keyword in test_name for keyword in ["external", "oauth", "email", "payment"]):
            item.add_marker(pytest.mark.requires_external)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Enhanced test reporting with performance tracking."""
    outcome = yield
    rep = outcome.get_result()
    
    # Add performance data to test report
    if hasattr(item.session, "performance_monitor"):
        monitor = item.session.performance_monitor
        
        if call.when == "setup":
            monitor.start_test(item.nodeid)
        elif call.when == "teardown":
            monitor.end_test(item.nodeid)


def pytest_sessionstart(session):
    """Enhanced session start with monitoring setup."""
    print("\n🚀 Starting enhanced claude-tiu test suite...")
    print("="*80)
    print("🔧 Configuration:")
    print(f"   • Async mode: enabled")
    print(f"   • Performance monitoring: enabled")
    print(f"   • Mock services: available")
    print(f"   • Enhanced error reporting: enabled")
    
    # Check for optional dependencies
    optional_deps = {
        "psutil": "Performance monitoring",
        "aiofiles": "Async file operations",
        "httpx": "Async HTTP testing",
        "pytest-benchmark": "Benchmarking support"
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            status = "✅"
        except ImportError:
            status = "❌"
        print(f"   • {description}: {status}")
    
    print("="*80)


def pytest_sessionfinish(session, exitstatus):
    """Enhanced session finish with comprehensive reporting."""
    print("\n" + "="*80)
    print("📋 TEST SUITE SUMMARY")
    print("="*80)
    
    # Test results summary
    if hasattr(session, "testscollected") and session.testscollected:
        failed = len([r for r in session._setupstate if getattr(r, "failed", False)])
        passed = session.testscollected - failed
        
        print(f"Tests collected: {session.testscollected}")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {failed}")
        
        if exitstatus == 0:
            print("🎉 All tests passed!")
        else:
            print(f"❌ Test suite finished with exit status: {exitstatus}")
    
    # Coverage reminder
    if not session.config.getoption("--cov"):
        print("\n💡 Tip: Run with --cov to see detailed coverage report")
        print("   Example: pytest --cov=src --cov-report=html")
    
    # Performance tips
    print("\n🚀 Performance Tips:")
    print("   • Use -n auto for parallel test execution")
    print("   • Use --benchmark-only for benchmark tests only")
    print("   • Use -m 'not slow' to skip slow tests during development")
    
    print("="*80)


# ============================================================================
# Async Test Utilities
# ============================================================================

@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 30.0  # 30 seconds


@pytest.fixture
async def async_test_context():
    """Provide async test context with utilities."""
    class AsyncTestContext:
        def __init__(self):
            self.cleanup_tasks = []
            self.background_tasks = []
        
        def add_cleanup(self, coro):
            """Add async cleanup task."""
            self.cleanup_tasks.append(coro)
        
        def add_background_task(self, coro):
            """Add background task."""
            task = asyncio.create_task(coro)
            self.background_tasks.append(task)
            return task
        
        async def cleanup(self):
            """Run all cleanup tasks."""
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Run cleanup tasks
            for cleanup_task in self.cleanup_tasks:
                try:
                    await cleanup_task
                except Exception as e:
                    print(f"Cleanup error: {e}")
    
    context = AsyncTestContext()
    yield context
    await context.cleanup()


# ============================================================================
# Coverage Configuration Enhancement
# ============================================================================

@pytest.fixture(autouse=True)
def coverage_context(request):
    """Provide coverage context information."""
    if hasattr(request.config, "pluginmanager"):
        cov_plugin = request.config.pluginmanager.get_plugin("_cov")
        if cov_plugin and cov_plugin.cov_controller:
            # Add test context to coverage
            cov_controller = cov_plugin.cov_controller
            test_id = request.node.nodeid
            
            # This helps with more detailed coverage reporting
            if hasattr(cov_controller, "cov") and hasattr(cov_controller.cov, "_data"):
                cov_controller.cov._data.set_context(f"test:{test_id}")


# ============================================================================
# Test Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, tmp_path):
    """Setup isolated test environment."""
    # Set test-specific environment variables
    test_env = {
        "CLAUDE_TIU_ENV": "test",
        "CLAUDE_TIU_DEBUG": "true",
        "CLAUDE_TIU_LOG_LEVEL": "INFO",
        "CLAUDE_TIU_DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "CLAUDE_TIU_REDIS_URL": "redis://localhost:6379/15",  # Use test database
        "CLAUDE_TIU_SECRET_KEY": "test-secret-key-not-for-production",
        "CLAUDE_TIU_DISABLE_AUTH": "false",
        "CLAUDE_TIU_CACHE_ENABLED": "true",
        "CLAUDE_TIU_METRICS_ENABLED": "true",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    # Setup test directories
    test_dirs = ["logs", "uploads", "temp", "cache"]
    for dirname in test_dirs:
        (tmp_path / dirname).mkdir(exist_ok=True)
        monkeypatch.setenv(f"CLAUDE_TIU_{dirname.upper()}_DIR", str(tmp_path / dirname))
    
    # Ensure clean state for each test
    monkeypatch.setattr("time.time", lambda: 1640995200.0)  # Fixed timestamp for consistent tests