"""
Test Workflow Core - Integrated testing framework for SPARC methodology
Provides comprehensive test runner, assertions, mocks, context, and reporting
"""

from .test_runner import (
    IntegratedTestRunner,
    TestSuite,
    TestResult,
    TestStatus,
    create_integrated_runner,
    test_case,
    suite_setup,
    suite_teardown
)

from .assertions import (
    AssertionFramework,
    AssertionResult,
    AssertionError,
    FluentAssertion,
    ComparisonType
)

from .mocks import (
    MockFramework,
    MockConfiguration,
    MockCall,
    MockType
)

from .context import (
    TestContext,
    ContextSnapshot
)

from .reporter import (
    TestReporter,
    BaseReporter,
    ConsoleReporter,
    JsonReporter,
    JUnitReporter,
    HtmlReporter,
    MarkdownReporter,
    CompositeReporter,
    ReportFormat,
    ReportConfiguration
)

__version__ = "1.0.0"
__all__ = [
    # Test Runner
    "IntegratedTestRunner",
    "TestSuite", 
    "TestResult",
    "TestStatus",
    "create_integrated_runner",
    "test_case",
    "suite_setup",
    "suite_teardown",
    
    # Assertions
    "AssertionFramework",
    "AssertionResult", 
    "AssertionError",
    "FluentAssertion",
    "ComparisonType",
    
    # Mocks
    "MockFramework",
    "MockConfiguration",
    "MockCall", 
    "MockType",
    
    # Context
    "TestContext",
    "ContextSnapshot",
    
    # Reporting
    "TestReporter",
    "BaseReporter",
    "ConsoleReporter",
    "JsonReporter", 
    "JUnitReporter",
    "HtmlReporter",
    "MarkdownReporter",
    "CompositeReporter",
    "ReportFormat",
    "ReportConfiguration"
]


# Convenience factory function
def create_test_framework(
    console_output: bool = True,
    json_output: str = None,
    html_output: str = None,
    junit_output: str = None,
    include_coverage: bool = False
) -> IntegratedTestRunner:
    """
    Create a fully configured test framework with common settings
    
    Args:
        console_output: Enable console reporting
        json_output: Path for JSON report (optional)
        html_output: Path for HTML report (optional) 
        junit_output: Path for JUnit XML report (optional)
        include_coverage: Include coverage information
        
    Returns:
        Configured IntegratedTestRunner
    """
    runner = create_integrated_runner(
        with_console_reporter=console_output,
        with_json_reporter=bool(json_output)
    )
    
    # Configure additional reporters
    if json_output:
        runner.reporter.configure_reporter(
            ReportFormat.JSON,
            output_file=json_output,
            include_coverage=include_coverage
        )
        
    if html_output:
        runner.reporter.configure_reporter(
            ReportFormat.HTML,
            output_file=html_output,
            include_coverage=include_coverage
        )
        
    if junit_output:
        runner.reporter.configure_reporter(
            ReportFormat.JUNIT,
            output_file=junit_output
        )
        
    return runner


# Integration shortcuts for SPARC methodology
async def run_sparc_tests(
    test_modules: list,
    output_dir: str = "test_results",
    parallel: bool = False,
    tags: list = None
) -> dict:
    """
    Run tests following SPARC methodology with integrated reporting
    
    Args:
        test_modules: List of test modules to run
        output_dir: Directory for test outputs
        parallel: Run tests in parallel
        tags: Filter tests by tags
        
    Returns:
        Comprehensive test results
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create integrated framework
    framework = create_test_framework(
        console_output=True,
        json_output=f"{output_dir}/results.json",
        html_output=f"{output_dir}/report.html", 
        junit_output=f"{output_dir}/junit.xml",
        include_coverage=True
    )
    
    # Discover and run tests
    for module in test_modules:
        discovered = framework.discover_tests(module)
        for suite in discovered:
            framework.add_suite(suite)
            
    # Execute tests
    results = await framework.run_all(
        filter_tags=tags,
        parallel=parallel
    )
    
    return results