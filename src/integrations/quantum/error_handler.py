"""
Comprehensive Error Handling and Fallback Mechanisms
Robust error handling system for Universal Development Environment Intelligence
"""

import asyncio
import functools
import logging
import sys
import traceback
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable, Awaitable
import weakref
import threading
from collections import defaultdict, deque
import json
import time

from pydantic import BaseModel, Field, ConfigDict


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(str, Enum):
    """Error categories"""
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    STORAGE = "storage"
    PERMISSION = "permission"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Error context information"""
    component: str
    operation: str
    environment: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "operation": self.operation,
            "environment": self.environment,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "metadata": self.metadata
        }


@dataclass
class ErrorRecord:
    """Error record with complete information"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    exception_message: str
    traceback: str
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: bool = False
    retry_count: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "traceback": self.traceback,
            "context": self.context.to_dict(),
            "recovery_attempted": self.recovery_attempted,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "recovery_success": self.recovery_success,
            "retry_count": self.retry_count,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


class ErrorClassifier:
    """Classifies errors by category and severity"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Classification rules
        self._category_patterns = {
            ErrorCategory.NETWORK: [
                "connection", "timeout", "unreachable", "dns", "socket",
                "http", "ssl", "certificate", "proxy"
            ],
            ErrorCategory.AUTHENTICATION: [
                "authentication", "authorization", "token", "credential",
                "login", "permission denied", "access denied", "forbidden"
            ],
            ErrorCategory.CONFIGURATION: [
                "configuration", "config", "setting", "parameter",
                "environment", "variable", "missing"
            ],
            ErrorCategory.STORAGE: [
                "file", "directory", "disk", "storage", "database",
                "no such file", "permission", "full", "space"
            ],
            ErrorCategory.VALIDATION: [
                "validation", "invalid", "format", "schema",
                "required", "missing field", "type error"
            ],
            ErrorCategory.PERFORMANCE: [
                "timeout", "memory", "cpu", "performance", "slow",
                "overload", "capacity", "limit exceeded"
            ]
        }
        
        self._severity_patterns = {
            ErrorSeverity.FATAL: [
                "fatal", "critical system", "corrupt", "segmentation fault"
            ],
            ErrorSeverity.CRITICAL: [
                "critical", "security", "data loss", "system failure",
                "service unavailable"
            ],
            ErrorSeverity.HIGH: [
                "error", "failed", "exception", "broken", "unable"
            ],
            ErrorSeverity.MEDIUM: [
                "warning", "deprecated", "retry", "fallback"
            ],
            ErrorSeverity.LOW: [
                "info", "notice", "debug", "minor"
            ]
        }
        
    def classify_error(self, exception: Exception, message: str = "") -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity"""
        try:
            # Get full error information
            error_text = f"{str(exception)} {message} {type(exception).__name__}".lower()
            
            # Classify category
            category = ErrorCategory.UNKNOWN
            for cat, patterns in self._category_patterns.items():
                if any(pattern in error_text for pattern in patterns):
                    category = cat
                    break
                    
            # Classify severity
            severity = ErrorSeverity.MEDIUM  # Default
            for sev, patterns in self._severity_patterns.items():
                if any(pattern in error_text for pattern in patterns):
                    severity = sev
                    break
                    
            # Special case classifications
            if isinstance(exception, (ConnectionError, TimeoutError)):
                category = ErrorCategory.NETWORK
                severity = ErrorSeverity.HIGH
            elif isinstance(exception, PermissionError):
                category = ErrorCategory.PERMISSION
                severity = ErrorSeverity.HIGH
            elif isinstance(exception, FileNotFoundError):
                category = ErrorCategory.STORAGE
                severity = ErrorSeverity.MEDIUM
            elif isinstance(exception, ValueError):
                category = ErrorCategory.VALIDATION
                severity = ErrorSeverity.MEDIUM
            elif isinstance(exception, (KeyboardInterrupt, SystemExit)):
                category = ErrorCategory.SYSTEM
                severity = ErrorSeverity.CRITICAL
            elif isinstance(exception, MemoryError):
                category = ErrorCategory.PERFORMANCE
                severity = ErrorSeverity.CRITICAL
                
            return category, severity
            
        except Exception as e:
            self.logger.error(f"Error in error classification: {e}")
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class RetryStrategy:
    """Retry strategy with backoff and jitter"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Check if operation should be retried"""
        if attempt >= self.max_retries:
            return False
            
        # Don't retry certain exceptions
        if isinstance(exception, (KeyboardInterrupt, SystemExit, MemoryError)):
            return False
            
        # Don't retry validation errors
        if isinstance(exception, (ValueError, TypeError)) and "validation" in str(exception).lower():
            return False
            
        return True
        
    def get_delay(self, attempt: int) -> float:
        """Get delay for retry attempt"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
            
        return delay


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker"""
        with self._lock:
            if self._state == "open":
                if time.time() - (self._last_failure_time or 0) > self.timeout:
                    self._state = "half-open"
                else:
                    raise Exception("Circuit breaker is OPEN")
                    
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
            
    async def acall(self, func: Callable[..., Awaitable], *args, **kwargs) -> Any:
        """Async call function through circuit breaker"""
        with self._lock:
            if self._state == "open":
                if time.time() - (self._last_failure_time or 0) > self.timeout:
                    self._state = "half-open"
                else:
                    raise Exception("Circuit breaker is OPEN")
                    
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
            
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self._failure_count = 0
            self._state = "closed"
            
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                
    @property
    def state(self) -> str:
        return self._state


class FallbackHandler:
    """Handles fallback operations when primary fails"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._fallback_functions: Dict[str, List[Callable]] = defaultdict(list)
        
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register fallback function for operation"""
        self._fallback_functions[operation_name].append(fallback_func)
        
    async def execute_fallback(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute fallback functions for operation"""
        fallbacks = self._fallback_functions.get(operation_name, [])
        
        for fallback in fallbacks:
            try:
                if asyncio.iscoroutinefunction(fallback):
                    result = await fallback(*args, **kwargs)
                else:
                    result = fallback(*args, **kwargs)
                return result
            except Exception as e:
                self.logger.warning(f"Fallback function failed: {e}")
                continue
                
        raise Exception(f"All fallbacks failed for operation: {operation_name}")


class ErrorRecoveryEngine:
    """Orchestrates error recovery strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retry_strategy = RetryStrategy()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handler = FallbackHandler()
        
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
        
    async def recover_from_error(self, error_record: ErrorRecord, 
                                operation: Callable, *args, **kwargs) -> Any:
        """Attempt to recover from error"""
        strategy = self._select_recovery_strategy(error_record)
        error_record.recovery_strategy = strategy
        error_record.recovery_attempted = True
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._retry_operation(error_record, operation, *args, **kwargs)
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_operation(error_record, operation, *args, **kwargs)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._circuit_breaker_operation(error_record, operation, *args, **kwargs)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._graceful_degradation(error_record, operation, *args, **kwargs)
            elif strategy == RecoveryStrategy.FAIL_FAST:
                raise Exception(f"Fail-fast strategy for error: {error_record.message}")
            else:
                # IGNORE strategy
                self.logger.info(f"Ignoring error: {error_record.message}")
                return None
                
        except Exception as e:
            error_record.recovery_success = False
            raise
            
    def _select_recovery_strategy(self, error_record: ErrorRecord) -> RecoveryStrategy:
        """Select appropriate recovery strategy"""
        # Strategy selection based on error characteristics
        if error_record.severity == ErrorSeverity.FATAL:
            return RecoveryStrategy.FAIL_FAST
        elif error_record.category == ErrorCategory.NETWORK:
            if error_record.retry_count < 3:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.FALLBACK
        elif error_record.category == ErrorCategory.AUTHENTICATION:
            return RecoveryStrategy.RETRY
        elif error_record.category == ErrorCategory.PERFORMANCE:
            return RecoveryStrategy.CIRCUIT_BREAKER
        elif error_record.category == ErrorCategory.VALIDATION:
            return RecoveryStrategy.FAIL_FAST
        else:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
            
    async def _retry_operation(self, error_record: ErrorRecord, 
                              operation: Callable, *args, **kwargs) -> Any:
        """Retry operation with backoff"""
        for attempt in range(self.retry_strategy.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    
                error_record.retry_count = attempt + 1
                
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                    
                error_record.recovery_success = True
                return result
                
            except Exception as e:
                if not self.retry_strategy.should_retry(attempt, e):
                    raise
                    
        raise Exception(f"Retry failed after {self.retry_strategy.max_retries} attempts")
        
    async def _fallback_operation(self, error_record: ErrorRecord,
                                 operation: Callable, *args, **kwargs) -> Any:
        """Execute fallback operation"""
        operation_name = getattr(operation, '__name__', 'unknown')
        result = await self.fallback_handler.execute_fallback(operation_name, *args, **kwargs)
        error_record.recovery_success = True
        return result
        
    async def _circuit_breaker_operation(self, error_record: ErrorRecord,
                                        operation: Callable, *args, **kwargs) -> Any:
        """Execute operation through circuit breaker"""
        breaker_name = f"{error_record.context.component}_{error_record.context.operation}"
        breaker = self.get_circuit_breaker(breaker_name)
        
        if asyncio.iscoroutinefunction(operation):
            result = await breaker.acall(operation, *args, **kwargs)
        else:
            result = breaker.call(operation, *args, **kwargs)
            
        error_record.recovery_success = True
        return result
        
    async def _graceful_degradation(self, error_record: ErrorRecord,
                                   operation: Callable, *args, **kwargs) -> Any:
        """Graceful degradation - return partial/default result"""
        self.logger.warning(f"Graceful degradation for: {error_record.message}")
        
        # Return appropriate default based on expected return type
        if hasattr(operation, '__annotations__') and 'return' in operation.__annotations__:
            return_type = operation.__annotations__['return']
            if return_type == dict:
                return {}
            elif return_type == list:
                return []
            elif return_type == str:
                return ""
            elif return_type == bool:
                return False
            elif return_type in (int, float):
                return 0
                
        error_record.recovery_success = True
        return None


class ErrorHandler:
    """
    Comprehensive Error Handling and Fallback Mechanisms
    Central error handling system for Universal Development Environment Intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.classifier = ErrorClassifier()
        self.recovery_engine = ErrorRecoveryEngine()
        
        # Error tracking
        self._error_records: Dict[str, ErrorRecord] = {}
        self._error_history: deque = deque(maxlen=config.get("history_size", 1000))
        self._error_stats = defaultdict(int)
        
        # Event callbacks
        self._error_callbacks: List[Callable[[ErrorRecord], None]] = []
        self._recovery_callbacks: List[Callable[[ErrorRecord, bool], None]] = []
        
        # Configuration
        self.auto_recovery = config.get("auto_recovery", True)
        self.error_reporting = config.get("error_reporting", True)
        
        # Thread safety
        self._lock = threading.RLock()
        
    async def handle_error(self, exception: Exception, context: ErrorContext,
                          message: str = "", auto_recover: bool = True) -> Optional[ErrorRecord]:
        """Handle error with context and optional recovery"""
        try:
            # Create error record
            error_id = f"{context.component}_{int(time.time() * 1000000)}"
            category, severity = self.classifier.classify_error(exception, message)
            
            error_record = ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                message=message or str(exception),
                exception_type=type(exception).__name__,
                exception_message=str(exception),
                traceback=traceback.format_exc(),
                context=context
            )
            
            # Store error record
            with self._lock:
                self._error_records[error_id] = error_record
                self._error_history.append(error_record)
                self._error_stats[f"{category.value}_{severity.value}"] += 1
                
            # Log error
            self._log_error(error_record)
            
            # Trigger error callbacks
            for callback in self._error_callbacks:
                try:
                    callback(error_record)
                except Exception as e:
                    self.logger.error(f"Error in error callback: {e}")
                    
            # Attempt recovery if enabled
            if auto_recover and self.auto_recovery and severity != ErrorSeverity.FATAL:
                # Recovery would be handled by the caller with recover_from_error
                pass
                
            return error_record
            
        except Exception as e:
            self.logger.critical(f"Error in error handler itself: {e}")
            return None
            
    async def recover_from_error(self, error_record: ErrorRecord, 
                                operation: Callable, *args, **kwargs) -> Any:
        """Attempt to recover from error"""
        try:
            result = await self.recovery_engine.recover_from_error(
                error_record, operation, *args, **kwargs
            )
            
            # Mark error as resolved if recovery succeeded
            if error_record.recovery_success:
                error_record.resolved = True
                error_record.resolution_time = datetime.now()
                
            # Trigger recovery callbacks
            for callback in self._recovery_callbacks:
                try:
                    callback(error_record, error_record.recovery_success)
                except Exception as e:
                    self.logger.error(f"Error in recovery callback: {e}")
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Error recovery failed: {e}")
            raise
            
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register fallback function"""
        self.recovery_engine.fallback_handler.register_fallback(operation_name, fallback_func)
        
    def register_error_callback(self, callback: Callable[[ErrorRecord], None]):
        """Register error event callback"""
        self._error_callbacks.append(callback)
        
    def register_recovery_callback(self, callback: Callable[[ErrorRecord, bool], None]):
        """Register recovery event callback"""
        self._recovery_callbacks.append(callback)
        
    def get_error_record(self, error_id: str) -> Optional[ErrorRecord]:
        """Get error record by ID"""
        return self._error_records.get(error_id)
        
    def get_recent_errors(self, hours: int = 1, 
                         severity: Optional[ErrorSeverity] = None,
                         category: Optional[ErrorCategory] = None) -> List[ErrorRecord]:
        """Get recent error records"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_errors = []
        for error in self._error_history:
            if error.timestamp < cutoff_time:
                continue
            if severity and error.severity != severity:
                continue
            if category and error.category != category:
                continue
            filtered_errors.append(error)
            
        return filtered_errors
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self._lock:
            total_errors = sum(self._error_stats.values())
            
            # Calculate error rates by category and severity
            category_stats = defaultdict(int)
            severity_stats = defaultdict(int)
            
            for key, count in self._error_stats.items():
                if '_' in key:
                    category, severity = key.split('_', 1)
                    category_stats[category] += count
                    severity_stats[severity] += count
                    
            return {
                "total_errors": total_errors,
                "error_records_stored": len(self._error_records),
                "error_history_size": len(self._error_history),
                "errors_by_category": dict(category_stats),
                "errors_by_severity": dict(severity_stats),
                "circuit_breakers": {
                    name: {"state": cb.state, "failure_count": cb._failure_count}
                    for name, cb in self.recovery_engine.circuit_breakers.items()
                }
            }
            
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        try:
            stats = self.get_error_statistics()
            recent_errors = self.get_recent_errors(24)  # Last 24 hours
            
            # Top error patterns
            error_patterns = defaultdict(int)
            for error in recent_errors:
                pattern = f"{error.category.value}_{error.exception_type}"
                error_patterns[pattern] += 1
                
            top_patterns = sorted(
                error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Recovery success rates
            recovery_attempts = [e for e in recent_errors if e.recovery_attempted]
            recovery_success_rate = 0.0
            if recovery_attempts:
                successful = sum(1 for e in recovery_attempts if e.recovery_success)
                recovery_success_rate = (successful / len(recovery_attempts)) * 100
                
            return {
                "generated_at": datetime.now().isoformat(),
                "statistics": stats,
                "recent_errors_24h": len(recent_errors),
                "recovery_success_rate": recovery_success_rate,
                "top_error_patterns": top_patterns,
                "unresolved_errors": len([e for e in recent_errors if not e.resolved]),
                "critical_errors": len([
                    e for e in recent_errors 
                    if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate error report: {e}")
            return {}
            
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL
        }.get(error_record.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"[{error_record.error_id}] {error_record.category.value.upper()} error in "
            f"{error_record.context.component}.{error_record.context.operation}: "
            f"{error_record.message}"
        )
        
        if error_record.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self.logger.critical(f"Traceback for {error_record.error_id}:\n{error_record.traceback}")


# Decorators for error handling
def handle_errors(component: str, operation: str, environment: str = "default",
                 auto_recover: bool = True, fallback_result: Any = None):
    """Decorator for automatic error handling"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = getattr(args[0], '_error_handler', None) if args else None
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    context = ErrorContext(
                        component=component,
                        operation=operation,
                        environment=environment
                    )
                    
                    error_record = await error_handler.handle_error(e, context)
                    
                    if auto_recover and error_record:
                        try:
                            return await error_handler.recover_from_error(
                                error_record, func, *args, **kwargs
                            )
                        except:
                            pass
                            
                if fallback_result is not None:
                    return fallback_result
                else:
                    raise
                    
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = getattr(args[0], '_error_handler', None) if args else None
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    context = ErrorContext(
                        component=component,
                        operation=operation,
                        environment=environment
                    )
                    
                    # For sync functions, we can't do async recovery
                    asyncio.create_task(error_handler.handle_error(e, context))
                    
                if fallback_result is not None:
                    return fallback_result
                else:
                    raise
                    
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def with_retry(max_retries: int = 3, base_delay: float = 1.0, 
               exponential_base: float = 2.0, jitter: bool = True):
    """Decorator for automatic retry with backoff"""
    def decorator(func):
        retry_strategy = RetryStrategy(max_retries, base_delay, 60.0, exponential_base, jitter)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries or not retry_strategy.should_retry(attempt, e):
                        raise
                    
                    delay = retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries or not retry_strategy.should_retry(attempt, e):
                        raise
                    
                    delay = retry_strategy.get_delay(attempt)
                    time.sleep(delay)
                    
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def with_circuit_breaker(failure_threshold: int = 5, timeout: float = 60.0):
    """Decorator for circuit breaker pattern"""
    circuit_breaker = CircuitBreaker(failure_threshold, timeout)
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await circuit_breaker.acall(func, *args, **kwargs)
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Context manager for error handling
class ErrorContext:
    """Context manager for error handling scope"""
    
    def __init__(self, error_handler: ErrorHandler, component: str, operation: str):
        self.error_handler = error_handler
        self.component = component
        self.operation = operation
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            context = ErrorContext(
                component=self.component,
                operation=self.operation,
                environment="default"
            )
            await self.error_handler.handle_error(exc_val, context)
        return False  # Don't suppress exception


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "auto_recovery": True,
            "error_reporting": True,
            "history_size": 1000
        }
        
        error_handler = ErrorHandler(config)
        
        # Register error callback
        def on_error(error_record: ErrorRecord):
            print(f"Error occurred: {error_record.message}")
            
        error_handler.register_error_callback(on_error)
        
        # Example error handling
        try:
            raise ConnectionError("Network connection failed")
        except Exception as e:
            context = ErrorContext(
                component="example",
                operation="demo",
                environment="test"
            )
            
            error_record = await error_handler.handle_error(e, context)
            print(f"Error handled: {error_record.error_id}")
            
        # Get error statistics
        stats = error_handler.get_error_statistics()
        print(f"Error statistics: {stats}")
        
        # Generate error report
        report = error_handler.get_error_report()
        print(f"Error report: {json.dumps(report, indent=2)}")
        
    # Run example
    # asyncio.run(main())