"""
Advanced Decorator System for claude-tui.

This module provides sophisticated decorators for:
- Error handling with recovery strategies
- Performance monitoring and optimization
- Security validation and auditing  
- Async operation management
- Retry logic with exponential backoff
- Caching and memoization
- Resource management and cleanup

Key Features:
- Context-aware error handling
- Performance metrics collection
- Security audit logging
- Intelligent retry mechanisms
- Memory-efficient caching
- Resource leak prevention
"""

import asyncio
import functools
import hashlib
import inspect
import time
import warnings
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

import psutil

from .exceptions import (
    ClaudeTUIException,
    MemoryLimitExceededError,
    PerformanceError,
    SecurityError,
    TaskExecutionTimeoutError,
    handle_exception
)
from .types import CacheStrategy, LogLevel, Priority, RetryStrategy

F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])

# Context variables for tracking
operation_context: ContextVar[Dict[str, Any]] = ContextVar('operation_context', default={})
performance_context: ContextVar[Dict[str, Any]] = ContextVar('performance_context', default={})


class PerformanceTracker:
    """Track performance metrics for decorated functions."""
    
    def __init__(self):
        self.metrics = {}
        self.active_operations = {}
    
    def start_operation(self, operation_id: str, function_name: str) -> None:
        """Start tracking an operation."""
        self.active_operations[operation_id] = {
            'function_name': function_name,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss
        }
    
    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """End tracking and return metrics."""
        if operation_id not in self.active_operations:
            return {}
        
        operation = self.active_operations.pop(operation_id)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        metrics = {
            'function_name': operation['function_name'],
            'duration': end_time - operation['start_time'],
            'memory_delta': end_memory - operation['start_memory'],
            'success': success,
            'timestamp': datetime.utcnow(),
            'metadata': metadata or {}
        }
        
        # Store in metrics history
        func_name = operation['function_name']
        if func_name not in self.metrics:
            self.metrics[func_name] = []
        self.metrics[func_name].append(metrics)
        
        # Keep only last 100 entries per function
        if len(self.metrics[func_name]) > 100:
            self.metrics[func_name] = self.metrics[func_name][-100:]
        
        return metrics


class CacheManager:
    """Intelligent caching system for decorated functions."""
    
    def __init__(self):
        self.cache = {}
        self.access_times = {}
        self.hit_counts = {}
        self.max_size = 1000
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        # Create a deterministic key from function name and arguments
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        # Convert args and kwargs to hashable representation
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> tuple[bool, Any]:
        """Get value from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
            return True, self.cache[key]
        return False, None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        cache_entry = {
            'value': value,
            'created_at': time.time(),
            'ttl': ttl
        }
        
        self.cache[key] = cache_entry
        self.access_times[key] = time.time()
        self.hit_counts[key] = 0
    
    def is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.cache:
            return True
        
        entry = self.cache[key]
        if entry.get('ttl') is None:
            return False
        
        return (time.time() - entry['created_at']) > entry['ttl']
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.access_times:
            return
        
        # Remove oldest 10% of entries
        entries_to_remove = max(1, len(self.cache) // 10)
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_entries[:entries_to_remove]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
            self.hit_counts.pop(key, None)


# Global instances
_performance_tracker = PerformanceTracker()
_cache_manager = CacheManager()


def error_handler(
    recovery_strategy: Optional[str] = None,
    log_errors: bool = True,
    reraise: bool = True,
    fallback_value: Any = None,
    exception_types: Optional[tuple] = None
) -> Callable[[F], F]:
    """
    Comprehensive error handling decorator.
    
    Args:
        recovery_strategy: Strategy for error recovery
        log_errors: Whether to log errors automatically
        reraise: Whether to re-raise exceptions after handling
        fallback_value: Default value to return on error
        exception_types: Specific exception types to handle
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Convert to structured exception
                structured_exc = handle_exception(e, context={
                    'function': func.__qualname__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })
                
                # Log if requested
                if log_errors:
                    try:
                        from .logger import get_logger
                        logger = get_logger(func.__module__)
                        logger.error(
                            f"Error in {func.__qualname__}: {structured_exc.message}",
                            extra={'error_id': structured_exc.error_id}
                        )
                    except ImportError:
                        pass
                
                # Handle specific exception types
                if exception_types and not isinstance(e, exception_types):
                    raise
                
                # Apply recovery strategy
                if recovery_strategy == 'fallback' and fallback_value is not None:
                    return fallback_value
                elif recovery_strategy == 'ignore':
                    return None
                
                if reraise:
                    raise structured_exc
                
                return fallback_value
        
        return wrapper
    return decorator


def async_error_handler(
    recovery_strategy: Optional[str] = None,
    log_errors: bool = True,
    reraise: bool = True,
    fallback_value: Any = None,
    exception_types: Optional[tuple] = None
) -> Callable[[AsyncF], AsyncF]:
    """
    Async version of error handling decorator.
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Convert to structured exception
                structured_exc = handle_exception(e, context={
                    'function': func.__qualname__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })
                
                # Log if requested
                if log_errors:
                    try:
                        from .logger import get_logger
                        logger = get_logger(func.__module__)
                        logger.error(
                            f"Error in {func.__qualname__}: {structured_exc.message}",
                            extra={'error_id': structured_exc.error_id}
                        )
                    except ImportError:
                        pass
                
                # Handle specific exception types
                if exception_types and not isinstance(e, exception_types):
                    raise
                
                # Apply recovery strategy
                if recovery_strategy == 'fallback' and fallback_value is not None:
                    return fallback_value
                elif recovery_strategy == 'ignore':
                    return None
                
                if reraise:
                    raise structured_exc
                
                return fallback_value
        
        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retry_on: Optional[tuple] = None,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
) -> Callable[[F], F]:
    """
    Intelligent retry decorator with multiple strategies.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for exponential backoff
        max_delay: Maximum delay between retries
        retry_on: Exception types to retry on
        strategy: Retry strategy to use
    """
    retry_on = retry_on or (Exception,)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed, raise exception
                        break
                    
                    # Calculate next delay based on strategy
                    if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        current_delay = min(current_delay * backoff_factor, max_delay)
                    elif strategy == RetryStrategy.LINEAR_BACKOFF:
                        current_delay = min(delay * (attempt + 1), max_delay)
                    elif strategy == RetryStrategy.FIXED_DELAY:
                        current_delay = delay
                    
                    # Log retry attempt
                    try:
                        from .logger import get_logger
                        logger = get_logger(func.__module__)
                        logger.warning(
                            f"Retrying {func.__qualname__} (attempt {attempt + 2}/{max_attempts}) after {current_delay}s delay"
                        )
                    except ImportError:
                        pass
                    
                    time.sleep(current_delay)
                except Exception as e:
                    # Non-retryable exception
                    raise handle_exception(e)
            
            # All retries failed
            raise handle_exception(last_exception, context={
                'max_attempts': max_attempts,
                'final_attempt': True
            })
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retry_on: Optional[tuple] = None,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
) -> Callable[[AsyncF], AsyncF]:
    """
    Async version of retry decorator.
    """
    retry_on = retry_on or (Exception,)
    
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    # Calculate next delay based on strategy
                    if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        current_delay = min(current_delay * backoff_factor, max_delay)
                    elif strategy == RetryStrategy.LINEAR_BACKOFF:
                        current_delay = min(delay * (attempt + 1), max_delay)
                    elif strategy == RetryStrategy.FIXED_DELAY:
                        current_delay = delay
                    
                    # Log retry attempt
                    try:
                        from .logger import get_logger
                        logger = get_logger(func.__module__)
                        logger.warning(
                            f"Retrying {func.__qualname__} (attempt {attempt + 2}/{max_attempts}) after {current_delay}s delay"
                        )
                    except ImportError:
                        pass
                    
                    await asyncio.sleep(current_delay)
                except Exception as e:
                    # Non-retryable exception
                    raise handle_exception(e)
            
            # All retries failed
            raise handle_exception(last_exception, context={
                'max_attempts': max_attempts,
                'final_attempt': True
            })
        
        return wrapper
    return decorator


def performance_monitor(
    memory_limit_mb: Optional[int] = None,
    execution_timeout: Optional[int] = None,
    log_performance: bool = True,
    track_memory: bool = True
) -> Callable[[F], F]:
    """
    Performance monitoring decorator with limits and tracking.
    
    Args:
        memory_limit_mb: Maximum memory usage in MB
        execution_timeout: Maximum execution time in seconds
        log_performance: Whether to log performance metrics
        track_memory: Whether to track memory usage
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_id = str(uuid4())
            start_memory = psutil.Process().memory_info().rss if track_memory else 0
            
            # Start performance tracking
            _performance_tracker.start_operation(operation_id, func.__qualname__)
            
            try:
                # Check memory limit before execution
                if memory_limit_mb and track_memory:
                    current_memory_mb = start_memory // 1024 // 1024
                    if current_memory_mb > memory_limit_mb:
                        raise MemoryLimitExceededError(
                            current_memory_mb, memory_limit_mb, func.__qualname__
                        )
                
                # Execute with timeout if specified
                if execution_timeout:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TaskExecutionTimeoutError(
                            func.__qualname__, execution_timeout, execution_timeout
                        )
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(execution_timeout)
                
                result = func(*args, **kwargs)
                
                # Clear timeout
                if execution_timeout:
                    signal.alarm(0)
                
                # Track successful completion
                metrics = _performance_tracker.end_operation(operation_id, success=True)
                
                # Log performance if requested
                if log_performance and metrics:
                    try:
                        from .logger import get_performance_logger
                        perf_logger = get_performance_logger()
                        perf_logger.log_metrics(func.__qualname__, metrics)
                    except ImportError:
                        pass
                
                return result
                
            except Exception as e:
                # Clear timeout
                if execution_timeout:
                    signal.alarm(0)
                
                # Track failed completion
                _performance_tracker.end_operation(operation_id, success=False, metadata={
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                
                raise
        
        return wrapper
    return decorator


def async_performance_monitor(
    memory_limit_mb: Optional[int] = None,
    execution_timeout: Optional[int] = None,
    log_performance: bool = True,
    track_memory: bool = True
) -> Callable[[AsyncF], AsyncF]:
    """
    Async version of performance monitoring decorator.
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            operation_id = str(uuid4())
            start_memory = psutil.Process().memory_info().rss if track_memory else 0
            
            # Start performance tracking
            _performance_tracker.start_operation(operation_id, func.__qualname__)
            
            try:
                # Check memory limit before execution
                if memory_limit_mb and track_memory:
                    current_memory_mb = start_memory // 1024 // 1024
                    if current_memory_mb > memory_limit_mb:
                        raise MemoryLimitExceededError(
                            current_memory_mb, memory_limit_mb, func.__qualname__
                        )
                
                # Execute with timeout if specified
                if execution_timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=execution_timeout
                    )
                else:
                    result = await func(*args, **kwargs)
                
                # Track successful completion
                metrics = _performance_tracker.end_operation(operation_id, success=True)
                
                # Log performance if requested
                if log_performance and metrics:
                    try:
                        from .logger import get_performance_logger
                        perf_logger = get_performance_logger()
                        perf_logger.log_metrics(func.__qualname__, metrics)
                    except ImportError:
                        pass
                
                return result
                
            except asyncio.TimeoutError:
                _performance_tracker.end_operation(operation_id, success=False, metadata={
                    'error': 'timeout',
                    'timeout_seconds': execution_timeout
                })
                raise TaskExecutionTimeoutError(
                    func.__qualname__, execution_timeout, execution_timeout
                )
            except Exception as e:
                # Track failed completion
                _performance_tracker.end_operation(operation_id, success=False, metadata={
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                raise
        
        return wrapper
    return decorator


def cache(
    ttl: Optional[int] = None,
    max_size: Optional[int] = None,
    strategy: CacheStrategy = CacheStrategy.LRU,
    key_generator: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Intelligent caching decorator with multiple strategies.
    
    Args:
        ttl: Time to live in seconds
        max_size: Maximum cache size
        strategy: Caching strategy
        key_generator: Custom key generation function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(func, args, kwargs)
            else:
                cache_key = _cache_manager._generate_key(func, args, kwargs)
            
            # Check cache
            hit, cached_value = _cache_manager.get(cache_key)
            if hit and not _cache_manager.is_expired(cache_key):
                return cached_value['value']
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            _cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def security_audit(
    audit_input: bool = True,
    audit_output: bool = False,
    sensitive_params: Optional[List[str]] = None,
    sanitize_input: bool = True
) -> Callable[[F], F]:
    """
    Security auditing decorator.
    
    Args:
        audit_input: Whether to audit input parameters
        audit_output: Whether to audit output
        sensitive_params: Parameter names to treat as sensitive
        sanitize_input: Whether to sanitize input automatically
    """
    sensitive_params = sensitive_params or []
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from .logger import get_security_logger
                security_logger = get_security_logger()
                
                # Audit input if requested
                if audit_input:
                    # Create sanitized version of kwargs for logging
                    safe_kwargs = {}
                    for key, value in kwargs.items():
                        if key in sensitive_params:
                            safe_kwargs[key] = '[REDACTED]'
                        else:
                            safe_kwargs[key] = str(value)[:100]  # Truncate for safety
                    
                    security_logger.log_security_event(
                        'function_call',
                        f"Function {func.__qualname__} called",
                        severity='low',
                        metadata={
                            'function': func.__qualname__,
                            'args_count': len(args),
                            'kwargs': safe_kwargs
                        }
                    )
                
                # Input sanitization (basic example)
                if sanitize_input:
                    # This is a simplified example - real sanitization would be more sophisticated
                    for key, value in kwargs.items():
                        if isinstance(value, str) and any(pattern in value.lower() for pattern in ['<script', 'javascript:', 'data:']):
                            raise SecurityError(f"Potentially malicious input detected in parameter {key}")
                
                result = func(*args, **kwargs)
                
                # Audit output if requested
                if audit_output and result is not None:
                    security_logger.log_security_event(
                        'function_output',
                        f"Function {func.__qualname__} returned result",
                        severity='low',
                        metadata={
                            'function': func.__qualname__,
                            'output_type': type(result).__name__,
                            'output_size': len(str(result)) if result else 0
                        }
                    )
                
                return result
                
            except ImportError:
                # Security logger not available, proceed without auditing
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def deprecated(
    reason: str,
    version: Optional[str] = None,
    alternative: Optional[str] = None
) -> Callable[[F], F]:
    """
    Mark function as deprecated with informative warnings.
    
    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        alternative: Suggested alternative function/method
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"{func.__qualname__} is deprecated"
            if version:
                warning_msg += f" since version {version}"
            warning_msg += f": {reason}"
            if alternative:
                warning_msg += f". Use {alternative} instead."
            
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            
            # Also log the deprecation warning
            try:
                from .logger import get_logger
                logger = get_logger(func.__module__)
                logger.warning(f"Deprecated function used: {warning_msg}")
            except ImportError:
                pass
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_types(**type_hints) -> Callable[[F], F]:
    """
    Runtime type validation decorator.
    
    Args:
        **type_hints: Parameter name to type mappings
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' in {func.__qualname__} "
                            f"expected {expected_type.__name__}, got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(calls_per_second: float) -> Callable[[F], F]:
    """
    Rate limiting decorator.
    
    Args:
        calls_per_second: Maximum calls per second allowed
    """
    min_interval = 1.0 / calls_per_second
    last_called = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_key = func.__qualname__
            now = time.time()
            
            if func_key in last_called:
                elapsed = now - last_called[func_key]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    time.sleep(sleep_time)
            
            last_called[func_key] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def resource_cleanup(*resources) -> Callable[[F], F]:
    """
    Automatic resource cleanup decorator.
    
    Args:
        *resources: Resource objects that need cleanup (must have close() method)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                for resource in resources:
                    try:
                        if hasattr(resource, 'close'):
                            resource.close()
                        elif hasattr(resource, '__exit__'):
                            resource.__exit__(None, None, None)
                    except Exception as e:
                        try:
                            from .logger import get_logger
                            logger = get_logger(func.__module__)
                            logger.warning(f"Failed to cleanup resource: {e}")
                        except ImportError:
                            pass
        
        return wrapper
    return decorator


# Convenience combinations
def robust(
    max_retries: int = 3,
    memory_limit_mb: Optional[int] = None,
    timeout: Optional[int] = None,
    cache_ttl: Optional[int] = None
) -> Callable[[F], F]:
    """
    Combination decorator for robust function execution.
    
    Combines retry, performance monitoring, and caching.
    """
    def decorator(func: F) -> F:
        # Apply decorators in order
        decorated_func = func
        
        if cache_ttl:
            decorated_func = cache(ttl=cache_ttl)(decorated_func)
        
        decorated_func = performance_monitor(
            memory_limit_mb=memory_limit_mb,
            execution_timeout=timeout
        )(decorated_func)
        
        decorated_func = retry(max_attempts=max_retries)(decorated_func)
        decorated_func = error_handler()(decorated_func)
        
        return decorated_func
    
    return decorator


def async_robust(
    max_retries: int = 3,
    memory_limit_mb: Optional[int] = None,
    timeout: Optional[int] = None,
    cache_ttl: Optional[int] = None
) -> Callable[[AsyncF], AsyncF]:
    """
    Async version of robust decorator combination.
    """
    def decorator(func: AsyncF) -> AsyncF:
        # Apply decorators in order
        decorated_func = func
        
        decorated_func = async_performance_monitor(
            memory_limit_mb=memory_limit_mb,
            execution_timeout=timeout
        )(decorated_func)
        
        decorated_func = async_retry(max_attempts=max_retries)(decorated_func)
        decorated_func = async_error_handler()(decorated_func)
        
        return decorated_func
    
    return decorator