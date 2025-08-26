"""
Comprehensive Exception System for claude-tui.

This module defines a hierarchy of custom exceptions with:
- Structured error information
- Error recovery strategies
- Security context awareness
- Performance impact tracking
- User-friendly error messages

Key Features:
- Exception hierarchy for different error categories
- Structured error metadata for debugging
- Automatic error reporting and logging
- Recovery strategy recommendations
- Context-aware error handling
"""

import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    AI_SERVICE = "ai_service"
    FILE_SYSTEM = "file_system"
    PARSING = "parsing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_INPUT = "user_input"
    INTEGRATION = "integration"


class RecoveryStrategy(Enum):
    """Recommended recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ABORT = "abort"
    USER_INTERVENTION = "user_intervention"
    SYSTEM_RESTART = "system_restart"
    IGNORE = "ignore"
    ESCALATE = "escalate"


class ClaudeTUIException(Exception):
    """
    Base exception class for all claude-tui specific errors.
    
    Provides structured error information and recovery guidance.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.ABORT,
        metadata: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            category: Error category for classification
            severity: Severity level for prioritization
            recovery_strategy: Recommended recovery approach
            metadata: Additional error metadata
            cause: Underlying exception that caused this error
            context: Context information (user, project, etc.)
        """
        super().__init__(message)
        
        self.error_id = str(uuid4())
        self.timestamp = datetime.utcnow()
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.category = category
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.metadata = metadata or {}
        self.cause = cause
        self.context = context or {}
        
        # Capture stack trace
        self.stack_trace = traceback.format_exc()
        
        # Auto-log critical and high severity errors
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self._auto_log_error()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'error_code': self.error_code,
            'category': self.category.value,
            'severity': self.severity.value,
            'recovery_strategy': self.recovery_strategy.value,
            'metadata': self.metadata,
            'context': self.context,
            'stack_trace': self.stack_trace,
            'cause': str(self.cause) if self.cause else None
        }
    
    def get_user_message(self) -> str:
        """Get user-friendly error message."""
        if self.severity == ErrorSeverity.CRITICAL:
            return f"Critical Error: {self.message}. Please contact support with error ID: {self.error_id}"
        elif self.severity == ErrorSeverity.HIGH:
            return f"Error: {self.message}. {self._get_recovery_guidance()}"
        else:
            return self.message
    
    def _get_recovery_guidance(self) -> str:
        """Get recovery guidance based on strategy."""
        guidance_map = {
            RecoveryStrategy.RETRY: "Please try again in a few moments.",
            RecoveryStrategy.FALLBACK: "Falling back to alternative approach.",
            RecoveryStrategy.USER_INTERVENTION: "Please check your input and try again.",
            RecoveryStrategy.SYSTEM_RESTART: "System restart may be required.",
            RecoveryStrategy.ESCALATE: "This issue requires technical attention.",
            RecoveryStrategy.ABORT: "Operation cannot be completed.",
            RecoveryStrategy.IGNORE: "This warning can be safely ignored."
        }
        return guidance_map.get(self.recovery_strategy, "")
    
    def _auto_log_error(self) -> None:
        """Auto-log high severity errors."""
        try:
            from .logger import get_logger
            logger = get_logger(__name__)
            logger.error(
                f"Exception {self.error_code}: {self.message}",
                extra={
                    'error_id': self.error_id,
                    'error_code': self.error_code,
                    'category': self.category.value,
                    'severity': self.severity.value,
                    'metadata': self.metadata,
                    'context': self.context
                }
            )
        except ImportError:
            # Logger not available, continue without logging
            pass


# Configuration and Setup Exceptions

# Legacy aliases for backward compatibility
ClaudeTIUException = ClaudeTUIException
DatabaseError = ClaudeTUIException  # Can be specialized later


class ConfigurationError(ClaudeTUIException):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_key: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        metadata = kwargs.get('metadata', {})
        metadata.update({
            'config_file': config_file,
            'config_key': config_key,
            'expected_value': expected_value,
            'actual_value': actual_value
        })
        
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            metadata=metadata,
            **kwargs
        )


class InvalidProjectConfigError(ConfigurationError):
    """Project configuration is invalid."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing."""
    
    def __init__(self, config_key: str, config_file: Optional[str] = None, **kwargs):
        super().__init__(
            f"Missing required configuration: {config_key}",
            config_key=config_key,
            config_file=config_file,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


# Validation and Input Exceptions

class ValidationError(ClaudeTUIException):
    """Input validation errors."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rules: Optional[List[str]] = None,
        **kwargs
    ):
        metadata = kwargs.get('metadata', {})
        metadata.update({
            'field_name': field_name,
            'field_value': field_value,
            'validation_rules': validation_rules
        })
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            metadata=metadata,
            **kwargs
        )


class PlaceholderDetectionError(ValidationError):
    """Anti-hallucination placeholder detection error."""
    
    def __init__(
        self,
        file_path: str,
        placeholder_count: int,
        placeholder_patterns: List[str],
        **kwargs
    ):
        super().__init__(
            f"Detected {placeholder_count} placeholders in {file_path}",
            metadata={
                'file_path': file_path,
                'placeholder_count': placeholder_count,
                'placeholder_patterns': placeholder_patterns
            },
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            **kwargs
        )


class SemanticValidationError(ValidationError):
    """Semantic validation failure in generated code."""
    
    def __init__(
        self,
        file_path: str,
        validation_failures: List[str],
        **kwargs
    ):
        super().__init__(
            f"Semantic validation failed for {file_path}",
            metadata={
                'file_path': file_path,
                'validation_failures': validation_failures
            },
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


# Authentication and Authorization Exceptions

class AuthenticationError(ClaudeTUIException):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            **kwargs
        )


class InvalidAPIKeyError(AuthenticationError):
    """API key is invalid or missing."""
    
    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Invalid or missing API key for {service}",
            metadata={'service': service},
            **kwargs
        )


class AuthorizationError(ClaudeTUIException):
    """Authorization-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            **kwargs
        )


# AI Service Exceptions

class AIServiceError(ClaudeTUIException):
    """AI service integration errors."""
    
    def __init__(
        self,
        message: str,
        service: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        metadata = kwargs.get('metadata', {})
        metadata.update({
            'service': service,
            'status_code': status_code,
            'response_data': response_data
        })
        
        super().__init__(
            message,
            category=ErrorCategory.AI_SERVICE,
            recovery_strategy=RecoveryStrategy.RETRY,
            metadata=metadata,
            **kwargs
        )


class ClaudeCodeError(AIServiceError):
    """Claude Code service specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service='claude_code', **kwargs)


class ClaudeFlowError(AIServiceError):
    """Claude Flow service specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service='claude_flow', **kwargs)


class AIResponseParsingError(AIServiceError):
    """Error parsing AI service response."""
    
    def __init__(
        self,
        service: str,
        response_text: str,
        expected_format: str,
        **kwargs
    ):
        super().__init__(
            f"Failed to parse {service} response: expected {expected_format}",
            service=service,
            metadata={
                'response_text': response_text[:1000],  # Truncate for logging
                'expected_format': expected_format
            },
            **kwargs
        )


class AIServiceTimeoutError(AIServiceError):
    """AI service request timeout."""
    
    def __init__(self, service: str, timeout_seconds: int, **kwargs):
        super().__init__(
            f"{service} request timed out after {timeout_seconds} seconds",
            service=service,
            metadata={'timeout_seconds': timeout_seconds},
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


# Network and Connectivity Exceptions

class NetworkError(ClaudeTUIException):
    """Network connectivity errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class ConnectionError(NetworkError):
    """Connection establishment failed."""
    pass


class TimeoutError(NetworkError):
    """Network operation timeout."""
    
    def __init__(self, operation: str, timeout_seconds: int, **kwargs):
        super().__init__(
            f"{operation} timed out after {timeout_seconds} seconds",
            metadata={'operation': operation, 'timeout_seconds': timeout_seconds},
            **kwargs
        )


# File System Exceptions

class FileSystemError(ClaudeTUIException):
    """File system operation errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        metadata = kwargs.get('metadata', {})
        metadata.update({
            'file_path': file_path,
            'operation': operation
        })
        
        super().__init__(
            message,
            category=ErrorCategory.FILE_SYSTEM,
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            metadata=metadata,
            **kwargs
        )


class FileNotFoundError(FileSystemError):
    """Required file not found."""
    
    def __init__(self, file_path: str, **kwargs):
        super().__init__(
            f"File not found: {file_path}",
            file_path=file_path,
            **kwargs
        )


class PermissionError(FileSystemError):
    """Insufficient file system permissions."""
    
    def __init__(self, file_path: str, operation: str, **kwargs):
        super().__init__(
            f"Permission denied: {operation} on {file_path}",
            file_path=file_path,
            operation=operation,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ProjectDirectoryError(FileSystemError):
    """Project directory structure issues."""
    
    def __init__(self, project_path: str, issue: str, **kwargs):
        super().__init__(
            f"Project directory issue: {issue}",
            file_path=project_path,
            metadata={'issue': issue},
            **kwargs
        )


# Security Exceptions

class NotFoundError(ClaudeTUIException):
    """Resource not found error."""
    
    def __init__(self, resource_type: str = "Resource", resource_id: str = None, **kwargs):
        message = f"{resource_type}"
        if resource_id:
            message += f" with ID '{resource_id}'"
        message += " not found"
        
        super().__init__(
            message=message,
            category=ErrorCategory.USER,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            **kwargs
        )


class WorkflowExecutionError(ClaudeTUIException):
    """Workflow execution error."""
    
    def __init__(self, message: str = "Workflow execution failed", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class ResourceNotFoundError(NotFoundError):
    """Resource not found error - alias for NotFoundError."""
    pass


class SecurityError(ClaudeTUIException):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            **kwargs
        )


class InputSanitizationError(SecurityError):
    """Input failed security sanitization."""
    
    def __init__(
        self,
        input_type: str,
        security_issue: str,
        sanitized_input: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            f"Security issue in {input_type}: {security_issue}",
            metadata={
                'input_type': input_type,
                'security_issue': security_issue,
                'sanitized_input': sanitized_input
            },
            **kwargs
        )


class MaliciousCodeDetectionError(SecurityError):
    """Detected potentially malicious code."""
    
    def __init__(
        self,
        file_path: str,
        detection_patterns: List[str],
        confidence_score: float,
        **kwargs
    ):
        super().__init__(
            f"Potentially malicious code detected in {file_path}",
            metadata={
                'file_path': file_path,
                'detection_patterns': detection_patterns,
                'confidence_score': confidence_score
            },
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


# Performance Exceptions

class PerformanceError(ClaudeTUIException):
    """Performance-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            **kwargs
        )


class MemoryLimitExceededError(PerformanceError):
    """Memory usage exceeded limits."""
    
    def __init__(
        self,
        current_usage_mb: int,
        limit_mb: int,
        operation: str,
        **kwargs
    ):
        super().__init__(
            f"Memory limit exceeded during {operation}: {current_usage_mb}MB > {limit_mb}MB",
            metadata={
                'current_usage_mb': current_usage_mb,
                'limit_mb': limit_mb,
                'operation': operation
            },
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class TaskExecutionTimeoutError(PerformanceError):
    """Task execution exceeded timeout."""
    
    def __init__(
        self,
        task_name: str,
        timeout_seconds: int,
        elapsed_seconds: int,
        **kwargs
    ):
        super().__init__(
            f"Task {task_name} timed out after {elapsed_seconds}s (limit: {timeout_seconds}s)",
            metadata={
                'task_name': task_name,
                'timeout_seconds': timeout_seconds,
                'elapsed_seconds': elapsed_seconds
            },
            **kwargs
        )


# Integration Exceptions

class IntegrationError(ClaudeTUIException):
    """External integration errors."""
    
    def __init__(
        self,
        message: str,
        integration: str,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.INTEGRATION,
            metadata={'integration': integration},
            **kwargs
        )


# AI Advanced Service Exceptions

class TaskError(ClaudeTUIException):
    """Task execution specific errors."""
    
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            metadata={'task_id': task_id},
            **kwargs
        )


class ResourceError(ClaudeTUIException):
    """Resource management specific errors."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            metadata={'resource_type': resource_type},
            **kwargs
        )


class SwarmError(AIServiceError):
    """Swarm orchestration specific errors."""
    
    def __init__(self, message: str, swarm_id: Optional[str] = None, **kwargs):
        super().__init__(
            message, 
            service='swarm_orchestrator',
            metadata={'swarm_id': swarm_id},
            **kwargs
        )


class AgentError(AIServiceError):
    """Agent coordination specific errors."""
    
    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            service='agent_coordinator', 
            metadata={'agent_id': agent_id},
            **kwargs
        )


class CoordinationError(AgentError):
    """Agent coordination system errors."""
    pass


class CommunicationError(AgentError):
    """Inter-agent communication errors."""
    pass


class NeuralTrainingError(AIServiceError):
    """Neural pattern training errors."""
    
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            service='neural_trainer',
            metadata={'model_id': model_id},
            **kwargs
        )


class ModelError(NeuralTrainingError):
    """Neural model specific errors."""
    pass


class OrchestrationError(ClaudeTUIException):
    """Orchestration system errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AI_SERVICE,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class GitIntegrationError(IntegrationError):
    """Git integration specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, integration='git', **kwargs)


class CLIToolError(IntegrationError):
    """CLI tool integration errors."""
    
    def __init__(
        self,
        tool_name: str,
        command: str,
        return_code: int,
        stderr: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            f"CLI tool {tool_name} failed: {command} (exit code: {return_code})",
            integration=tool_name,
            metadata={
                'tool_name': tool_name,
                'command': command,
                'return_code': return_code,
                'stderr': stderr
            },
            **kwargs
        )


class ComplianceError(ClaudeTUIException):
    """Compliance-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            **kwargs
        )


# Utility Functions for Exception Handling

def handle_exception(
    exception: Exception,
    logger: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None
) -> ClaudeTUIException:
    """
    Convert generic exceptions to structured ClaudeTIU exceptions.
    
    Args:
        exception: Original exception
        logger: Logger instance for automatic logging
        context: Additional context information
        
    Returns:
        Structured ClaudeTUIException
    """
    if isinstance(exception, ClaudeTUIException):
        return exception
    
    # Map common exception types
    exception_mapping = {
        ValueError: ValidationError,
        KeyError: ConfigurationError,
        FileNotFoundError: FileSystemError,
        PermissionError: FileSystemError,
        TimeoutError: NetworkError,
        ConnectionError: NetworkError,
    }
    
    exception_class = exception_mapping.get(type(exception), ClaudeTUIException)
    
    structured_exception = exception_class(
        str(exception),
        cause=exception,
        context=context or {}
    )
    
    if logger:
        logger.error(f"Exception converted: {structured_exception.error_code}")
    
    return structured_exception


def create_error_response(
    exception: ClaudeTUIException,
    include_stack_trace: bool = False
) -> Dict[str, Any]:
    """
    Create standardized error response from exception.
    
    Args:
        exception: ClaudeTIU exception instance
        include_stack_trace: Whether to include stack trace in response
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        'error': True,
        'error_id': exception.error_id,
        'error_code': exception.error_code,
        'message': exception.get_user_message(),
        'category': exception.category.value,
        'severity': exception.severity.value,
        'recovery_strategy': exception.recovery_strategy.value,
        'timestamp': exception.timestamp.isoformat()
    }
    
    if include_stack_trace:
        response['stack_trace'] = exception.stack_trace
    
    if exception.metadata:
        response['metadata'] = exception.metadata
    
    return response


def log_exception(
    exception: Exception,
    logger: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log exception with appropriate detail level.
    
    Args:
        exception: Exception to log
        logger: Logger instance (will create if None)
        context: Additional context for logging
    """
    if logger is None:
        try:
            from .logger import get_logger
            logger = get_logger(__name__)
        except ImportError:
            return  # Logger not available
    
    if isinstance(exception, ClaudeTUIException):
        logger.error(
            f"{exception.error_code}: {exception.message}",
            extra={
                'error_id': exception.error_id,
                'category': exception.category.value,
                'severity': exception.severity.value,
                'metadata': exception.metadata,
                'context': context or {}
            }
        )
    else:
        logger.error(
            f"Unhandled exception: {type(exception).__name__}: {str(exception)}",
            exc_info=True,
            extra={'context': context or {}}
        )