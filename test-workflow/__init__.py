"""
Test Workflow Framework - Integrated Testing for SPARC Methodology

A comprehensive testing framework that integrates:
- Test Runner with orchestration
- Assertion Library with detailed error messages
- Mock Framework with advanced capabilities
- Test Context for shared data management
- Reporter System with multiple output formats
- CI/CD Integration ready pipelines
"""

__version__ = "1.0.0"
__title__ = "Test Workflow Framework"
__description__ = "Integrated testing framework for SPARC methodology"
__author__ = "Test Workflow Team"
__license__ = "MIT"

# Core imports for easy access
from .core import (
    # Test Runner
    IntegratedTestRunner,
    TestSuite,
    TestResult,
    TestStatus,
    create_integrated_runner,
    create_test_framework,
    test_case,
    suite_setup,
    suite_teardown,
    
    # Assertions
    AssertionFramework,
    AssertionResult,
    AssertionError,
    FluentAssertion,
    ComparisonType,
    
    # Mocks
    MockFramework,
    MockConfiguration,
    MockCall,
    MockType,
    
    # Context
    TestContext,
    ContextSnapshot,
    
    # Reporting
    TestReporter,
    BaseReporter,
    ConsoleReporter,
    JsonReporter,
    JUnitReporter,
    HtmlReporter,
    MarkdownReporter,
    CompositeReporter,
    ReportFormat,
    ReportConfiguration,
    
    # SPARC Integration
    run_sparc_tests
)

# Error handling and developer experience
from .core.exceptions import TestWorkflowError, ConfigurationError, ValidationError
from .core.developer_experience import setup_development_environment, validate_environment

__all__ = [
    # Core classes
    "IntegratedTestRunner",
    "TestSuite",
    "TestResult", 
    "TestStatus",
    "AssertionFramework",
    "MockFramework",
    "TestContext",
    "TestReporter",
    
    # Factory functions
    "create_integrated_runner",
    "create_test_framework",
    "run_sparc_tests",
    
    # Decorators
    "test_case",
    "suite_setup", 
    "suite_teardown",
    
    # Assertion types
    "AssertionResult",
    "AssertionError",
    "FluentAssertion",
    "ComparisonType",
    
    # Mock types
    "MockConfiguration",
    "MockCall",
    "MockType",
    
    # Context types
    "ContextSnapshot",
    
    # Reporter classes
    "BaseReporter",
    "ConsoleReporter",
    "JsonReporter",
    "JUnitReporter",
    "HtmlReporter",
    "MarkdownReporter",
    "CompositeReporter",
    "ReportFormat",
    "ReportConfiguration",
    
    # Error classes
    "TestWorkflowError",
    "ConfigurationError", 
    "ValidationError",
    
    # Developer experience
    "setup_development_environment",
    "validate_environment",
    
    # Version info
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__license__"
]


def get_version():
    """Get framework version"""
    return __version__


def get_info():
    """Get framework information"""
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__, 
        "author": __author__,
        "license": __license__
    }


# Framework initialization and validation
def _validate_installation():
    """Validate framework installation"""
    try:
        # Check core components
        from .core import IntegratedTestRunner
        from .core.assertions import AssertionFramework
        from .core.mocks import MockFramework
        from .core.context import TestContext
        from .core.reporter import TestReporter
        
        return True
    except ImportError as e:
        raise ImportError(f"Test Workflow Framework installation is incomplete: {e}")


def _setup_logging():
    """Setup default logging configuration"""
    import logging
    
    # Create logger
    logger = logging.getLogger('test_workflow')
    
    # Only add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


# Initialize framework on import
try:
    _validate_installation()
    _framework_logger = _setup_logging()
    _framework_logger.debug("Test Workflow Framework initialized successfully")
except Exception as e:
    import warnings
    warnings.warn(f"Test Workflow Framework initialization warning: {e}")


# Convenience function for quick framework usage
def quick_test(test_function, **kwargs):
    """
    Quick test execution for single test functions
    
    Args:
        test_function: Test function to execute
        **kwargs: Additional configuration options
        
    Returns:
        Test results dictionary
    """
    import asyncio
    
    async def run_quick_test():
        framework = create_test_framework(**kwargs)
        suite = TestSuite("Quick Test")
        suite.add_test(test_function)
        framework.add_suite(suite)
        return await framework.run_all()
        
    return asyncio.run(run_quick_test())
    

# Add quick_test to exports
__all__.append("quick_test")