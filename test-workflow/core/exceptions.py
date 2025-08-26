"""
Test Workflow Framework Exceptions
Custom exception classes for better error handling and developer experience
"""

from typing import Optional, Any, Dict, List
import traceback


class TestWorkflowError(Exception):
    """Base exception class for Test Workflow Framework"""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        context: Dict[str, Any] = None,
        suggestions: List[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "TWF_GENERIC_ERROR"
        self.context = context or {}
        self.suggestions = suggestions or []
        self.traceback_str = traceback.format_exc()
        
    def __str__(self) -> str:
        """Enhanced string representation with helpful information"""
        output = [f"TestWorkflowError ({self.error_code}): {self.message}"]
        
        if self.context:
            output.append("\nContext:")
            for key, value in self.context.items():
                output.append(f"  {key}: {value}")
                
        if self.suggestions:
            output.append("\nSuggestions:")
            for suggestion in self.suggestions:
                output.append(f"  • {suggestion}")
                
        return "\n".join(output)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "suggestions": self.suggestions,
            "traceback": self.traceback_str
        }


class ConfigurationError(TestWorkflowError):
    """Raised when there are configuration-related issues"""
    
    def __init__(
        self,
        message: str,
        config_key: str = None,
        config_value: Any = None,
        expected_type: str = None
    ):
        context = {}
        suggestions = []
        
        if config_key:
            context["config_key"] = config_key
            
        if config_value is not None:
            context["config_value"] = str(config_value)
            context["actual_type"] = type(config_value).__name__
            
        if expected_type:
            context["expected_type"] = expected_type
            suggestions.append(f"Ensure {config_key} is of type {expected_type}")
            
        suggestions.extend([
            "Check your configuration file or environment variables",
            "Refer to documentation for valid configuration options"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_CONFIGURATION_ERROR",
            context=context,
            suggestions=suggestions
        )


class ValidationError(TestWorkflowError):
    """Raised when validation fails"""
    
    def __init__(
        self,
        message: str,
        field_name: str = None,
        field_value: Any = None,
        validation_rule: str = None
    ):
        context = {}
        suggestions = []
        
        if field_name:
            context["field_name"] = field_name
            
        if field_value is not None:
            context["field_value"] = str(field_value)
            
        if validation_rule:
            context["validation_rule"] = validation_rule
            suggestions.append(f"Ensure {field_name} satisfies: {validation_rule}")
            
        suggestions.extend([
            "Check input data format and values",
            "Verify all required fields are provided"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_VALIDATION_ERROR",
            context=context,
            suggestions=suggestions
        )


class TestExecutionError(TestWorkflowError):
    """Raised when test execution fails"""
    
    def __init__(
        self,
        message: str,
        test_name: str = None,
        suite_name: str = None,
        error_phase: str = None
    ):
        context = {}
        suggestions = []
        
        if test_name:
            context["test_name"] = test_name
            
        if suite_name:
            context["suite_name"] = suite_name
            
        if error_phase:
            context["error_phase"] = error_phase
            
        suggestions.extend([
            "Check test function signature and parameters",
            "Verify test setup and teardown methods",
            "Review test dependencies and mocks"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_TEST_EXECUTION_ERROR",
            context=context,
            suggestions=suggestions
        )


class AssertionSetupError(TestWorkflowError):
    """Raised when assertion framework setup fails"""
    
    def __init__(
        self,
        message: str,
        assertion_type: str = None,
        test_context: str = None
    ):
        context = {}
        suggestions = []
        
        if assertion_type:
            context["assertion_type"] = assertion_type
            
        if test_context:
            context["test_context"] = test_context
            
        suggestions.extend([
            "Ensure assertion framework is properly initialized",
            "Check that test context is correctly set up",
            "Verify assertion method usage is correct"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_ASSERTION_SETUP_ERROR",
            context=context,
            suggestions=suggestions
        )


class MockSetupError(TestWorkflowError):
    """Raised when mock framework setup fails"""
    
    def __init__(
        self,
        message: str,
        mock_name: str = None,
        mock_type: str = None,
        configuration_issue: str = None
    ):
        context = {}
        suggestions = []
        
        if mock_name:
            context["mock_name"] = mock_name
            
        if mock_type:
            context["mock_type"] = mock_type
            
        if configuration_issue:
            context["configuration_issue"] = configuration_issue
            
        suggestions.extend([
            "Verify mock configuration parameters",
            "Check that mock target exists and is accessible",
            "Ensure mock framework is properly initialized"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_MOCK_SETUP_ERROR", 
            context=context,
            suggestions=suggestions
        )


class ContextError(TestWorkflowError):
    """Raised when test context operations fail"""
    
    def __init__(
        self,
        message: str,
        context_key: str = None,
        operation: str = None,
        context_state: str = None
    ):
        context_info = {}
        suggestions = []
        
        if context_key:
            context_info["context_key"] = context_key
            
        if operation:
            context_info["operation"] = operation
            
        if context_state:
            context_info["context_state"] = context_state
            
        suggestions.extend([
            "Check that context key exists before access",
            "Verify context setup and initialization",
            "Ensure proper context cleanup"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_CONTEXT_ERROR",
            context=context_info,
            suggestions=suggestions
        )


class ReportingError(TestWorkflowError):
    """Raised when reporting fails"""
    
    def __init__(
        self,
        message: str,
        report_format: str = None,
        output_path: str = None,
        report_data_issue: str = None
    ):
        context = {}
        suggestions = []
        
        if report_format:
            context["report_format"] = report_format
            
        if output_path:
            context["output_path"] = output_path
            
        if report_data_issue:
            context["report_data_issue"] = report_data_issue
            
        suggestions.extend([
            "Check output directory permissions",
            "Verify report format is supported",
            "Ensure test results data is valid"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_REPORTING_ERROR",
            context=context,
            suggestions=suggestions
        )


class IntegrationError(TestWorkflowError):
    """Raised when component integration fails"""
    
    def __init__(
        self,
        message: str,
        component_a: str = None,
        component_b: str = None,
        integration_point: str = None
    ):
        context = {}
        suggestions = []
        
        if component_a:
            context["component_a"] = component_a
            
        if component_b:
            context["component_b"] = component_b
            
        if integration_point:
            context["integration_point"] = integration_point
            
        suggestions.extend([
            "Check component compatibility versions",
            "Verify integration configuration",
            "Review component initialization order"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_INTEGRATION_ERROR",
            context=context,
            suggestions=suggestions
        )


class EnvironmentError(TestWorkflowError):
    """Raised when environment setup fails"""
    
    def __init__(
        self,
        message: str,
        missing_dependency: str = None,
        environment_variable: str = None,
        system_requirement: str = None
    ):
        context = {}
        suggestions = []
        
        if missing_dependency:
            context["missing_dependency"] = missing_dependency
            suggestions.append(f"Install missing dependency: pip install {missing_dependency}")
            
        if environment_variable:
            context["environment_variable"] = environment_variable
            suggestions.append(f"Set required environment variable: {environment_variable}")
            
        if system_requirement:
            context["system_requirement"] = system_requirement
            suggestions.append(f"Ensure system requirement is met: {system_requirement}")
            
        suggestions.extend([
            "Check framework installation completeness",
            "Verify system compatibility",
            "Review environment configuration"
        ])
        
        super().__init__(
            message=message,
            error_code="TWF_ENVIRONMENT_ERROR",
            context=context,
            suggestions=suggestions
        )


# Exception handling utilities
def handle_exception(
    exception: Exception,
    context: str = None,
    logger = None,
    raise_new: bool = True
) -> Optional[TestWorkflowError]:
    """
    Handle and potentially convert exceptions to TestWorkflowError
    
    Args:
        exception: Original exception
        context: Additional context information
        logger: Logger instance for error logging
        raise_new: Whether to raise converted exception
        
    Returns:
        TestWorkflowError instance if raise_new=False
    """
    
    # If already a TestWorkflowError, just re-raise or return
    if isinstance(exception, TestWorkflowError):
        if logger:
            logger.error(f"TestWorkflowError in {context}: {exception}")
        if raise_new:
            raise exception
        return exception
        
    # Convert to appropriate TestWorkflowError subclass
    error_message = str(exception)
    error_context = {"original_exception": type(exception).__name__}
    
    if context:
        error_context["context"] = context
        
    # Determine appropriate exception type based on original exception
    if isinstance(exception, (ImportError, ModuleNotFoundError)):
        converted = EnvironmentError(
            message=f"Environment setup issue: {error_message}",
            missing_dependency=str(exception).split()[-1] if "No module named" in str(exception) else None
        )
    elif isinstance(exception, (ValueError, TypeError)):
        converted = ValidationError(
            message=f"Validation failed: {error_message}"
        )
    elif isinstance(exception, (KeyError, AttributeError)):
        converted = ConfigurationError(
            message=f"Configuration issue: {error_message}"
        )
    else:
        converted = TestWorkflowError(
            message=f"Unexpected error: {error_message}",
            context=error_context
        )
        
    if logger:
        logger.error(f"Exception converted to TestWorkflowError: {converted}")
        
    if raise_new:
        raise converted
    return converted


def create_error_report(exception: TestWorkflowError) -> Dict[str, Any]:
    """
    Create a comprehensive error report from TestWorkflowError
    
    Args:
        exception: TestWorkflowError instance
        
    Returns:
        Dictionary containing error report
    """
    
    import time
    import platform
    import sys
    
    return {
        "error_report": {
            "timestamp": time.time(),
            "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": exception.to_dict(),
            "system_info": {
                "platform": platform.system(),
                "python_version": sys.version,
                "framework_version": "1.0.0"  # Would be imported from main module
            }
        }
    }


def format_exception_for_console(exception: TestWorkflowError) -> str:
    """
    Format exception for console display with colors and formatting
    
    Args:
        exception: TestWorkflowError instance
        
    Returns:
        Formatted string for console display
    """
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"🚨 TEST WORKFLOW FRAMEWORK ERROR")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Error Code: {exception.error_code}")
    lines.append(f"Message: {exception.message}")
    
    if exception.context:
        lines.append("")
        lines.append("Context:")
        for key, value in exception.context.items():
            lines.append(f"  {key}: {value}")
            
    if exception.suggestions:
        lines.append("")
        lines.append("💡 Suggestions:")
        for suggestion in exception.suggestions:
            lines.append(f"  • {suggestion}")
            
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)