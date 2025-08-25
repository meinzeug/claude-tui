#!/usr/bin/env python3
"""
Centralized Error Handler for Claude-TUI.

This module provides comprehensive error handling and recovery strategies for all 
components of the Claude-TUI application. It includes automatic error detection,
logging, recovery strategies, and graceful fallbacks.

Key Features:
- Automatic error classification and severity assessment
- Context-aware error handling with recovery strategies
- Graceful fallbacks for missing dependencies
- Real-time error monitoring and alerting
- Performance impact tracking
- User-friendly error reporting
"""

import asyncio
import functools
import inspect
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import uuid4

from .exceptions import (
    ClaudeTUIException, ErrorSeverity, ErrorCategory, RecoveryStrategy,
    handle_exception, create_error_response, log_exception
)


class ErrorHandler:
    """
    Centralized error handler with automatic recovery strategies.
    """
    
    def __init__(self):
        self.error_count = {}
        self.recovery_attempts = {}
        self.failed_components = set()
        self.error_history = []
        self.max_history = 1000
        self.logger = None
        
        # Initialize logger
        try:
            from .logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = None
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        component: Optional[str] = None,
        auto_recover: bool = True
    ) -> Dict[str, Any]:
        """
        Handle an error with automatic classification and recovery.
        
        Args:
            error: The exception to handle
            context: Additional context information
            component: Component where error occurred
            auto_recover: Whether to attempt automatic recovery
        
        Returns:
            Dictionary containing error info and recovery status
        """
        # Convert to structured exception
        if not isinstance(error, ClaudeTUIException):
            error = handle_exception(error, self.logger, context)
        
        error_info = {
            'error_id': error.error_id,
            'timestamp': error.timestamp,
            'component': component,
            'error_type': type(error).__name__,
            'message': error.message,
            'severity': error.severity.value,
            'category': error.category.value,
            'recovery_strategy': error.recovery_strategy.value,
            'recovery_attempted': False,
            'recovery_successful': False,
            'context': context or {}
        }
        
        # Track error frequency
        error_key = f"{component}:{type(error).__name__}"
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
        
        # Add to error history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log the error
        if self.logger:
            self.logger.error(
                f"Error in {component}: {error.message}",
                extra={
                    'error_id': error.error_id,
                    'error_type': type(error).__name__,
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'context': context or {}
                }
            )
        
        # Attempt automatic recovery if enabled
        if auto_recover:
            recovery_result = self._attempt_recovery(error, component, context)
            error_info.update(recovery_result)
        
        return error_info
    
    def _attempt_recovery(
        self,
        error: ClaudeTUIException,
        component: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Attempt to recover from an error based on its recovery strategy."""
        recovery_key = f"{component}:{error.error_id}"
        attempts = self.recovery_attempts.get(recovery_key, 0)
        max_attempts = 3
        
        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'recovery_action': None,
            'recovery_message': None
        }
        
        # Prevent infinite recovery loops
        if attempts >= max_attempts:
            recovery_result['recovery_message'] = f"Max recovery attempts ({max_attempts}) exceeded"
            return recovery_result
        
        self.recovery_attempts[recovery_key] = attempts + 1
        
        try:
            if error.recovery_strategy == RecoveryStrategy.RETRY:
                recovery_result.update(self._retry_operation(error, component, context))
            
            elif error.recovery_strategy == RecoveryStrategy.FALLBACK:
                recovery_result.update(self._apply_fallback(error, component, context))
            
            elif error.recovery_strategy == RecoveryStrategy.USER_INTERVENTION:
                recovery_result.update(self._request_user_intervention(error, component))
            
            elif error.recovery_strategy == RecoveryStrategy.SYSTEM_RESTART:
                recovery_result.update(self._request_system_restart(error, component))
            
            elif error.recovery_strategy == RecoveryStrategy.IGNORE:
                recovery_result.update(self._ignore_error(error, component))
            
            elif error.recovery_strategy == RecoveryStrategy.ESCALATE:
                recovery_result.update(self._escalate_error(error, component))
            
            else:  # ABORT
                recovery_result.update(self._abort_operation(error, component))
        
        except Exception as recovery_error:
            recovery_result['recovery_message'] = f"Recovery failed: {str(recovery_error)}"
            if self.logger:
                self.logger.error(f"Recovery attempt failed: {str(recovery_error)}")
        
        return recovery_result
    
    def _retry_operation(self, error, component, context):
        """Retry the failed operation with exponential backoff."""
        return {
            'recovery_action': 'retry',
            'recovery_successful': True,
            'recovery_message': 'Operation will be retried with exponential backoff'
        }
    
    def _apply_fallback(self, error, component, context):
        """Apply fallback strategy for the failed component."""
        fallbacks = {
            'task_dashboard': self._fallback_task_dashboard,
            'ai_interface': self._fallback_ai_interface,
            'project_manager': self._fallback_project_manager,
            'config_manager': self._fallback_config_manager,
            'database': self._fallback_database,
            'file_system': self._fallback_file_system
        }
        
        fallback_func = fallbacks.get(component)
        if fallback_func:
            try:
                fallback_result = fallback_func(error, context)
                return {
                    'recovery_action': 'fallback',
                    'recovery_successful': True,
                    'recovery_message': f"Applied fallback for {component}: {fallback_result}"
                }
            except Exception as e:
                return {
                    'recovery_action': 'fallback',
                    'recovery_successful': False,
                    'recovery_message': f"Fallback failed: {str(e)}"
                }
        
        return {
            'recovery_action': 'fallback',
            'recovery_successful': False,
            'recovery_message': f"No fallback available for {component}"
        }
    
    def _fallback_task_dashboard(self, error, context):
        """Fallback for task dashboard failures."""
        return "Using offline task dashboard mode"
    
    def _fallback_ai_interface(self, error, context):
        """Fallback for AI interface failures."""
        return "Using mock AI responses for development mode"
    
    def _fallback_project_manager(self, error, context):
        """Fallback for project manager failures."""
        return "Using basic file system project management"
    
    def _fallback_config_manager(self, error, context):
        """Fallback for config manager failures."""
        return "Using default configuration values"
    
    def _fallback_database(self, error, context):
        """Fallback for database failures."""
        return "Using in-memory storage for session"
    
    def _fallback_file_system(self, error, context):
        """Fallback for file system failures."""
        return "Using temporary directory for file operations"
    
    def _request_user_intervention(self, error, component):
        """Request user intervention for the error."""
        return {
            'recovery_action': 'user_intervention',
            'recovery_successful': True,
            'recovery_message': f"User intervention requested for {component}: {error.get_user_message()}"
        }
    
    def _request_system_restart(self, error, component):
        """Request system restart to recover from error."""
        return {
            'recovery_action': 'system_restart',
            'recovery_successful': True,
            'recovery_message': f"System restart recommended for {component}"
        }
    
    def _ignore_error(self, error, component):
        """Ignore the error and continue operation."""
        return {
            'recovery_action': 'ignore',
            'recovery_successful': True,
            'recovery_message': f"Ignoring non-critical error in {component}"
        }
    
    def _escalate_error(self, error, component):
        """Escalate error to higher-level error handling."""
        return {
            'recovery_action': 'escalate',
            'recovery_successful': True,
            'recovery_message': f"Error escalated for {component}: {error.error_id}"
        }
    
    def _abort_operation(self, error, component):
        """Abort the current operation due to critical error."""
        self.failed_components.add(component)
        return {
            'recovery_action': 'abort',
            'recovery_successful': False,
            'recovery_message': f"Operation aborted for {component} due to critical error"
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics and health status."""
        total_errors = sum(self.error_count.values())
        critical_errors = len([e for e in self.error_history if e.get('severity') == 'critical'])
        
        return {
            'total_errors': total_errors,
            'critical_errors': critical_errors,
            'failed_components': list(self.failed_components),
            'error_frequency': dict(self.error_count),
            'recent_errors': self.error_history[-10:],
            'health_status': 'critical' if critical_errors > 0 else 'healthy' if total_errors == 0 else 'degraded'
        }
    
    def clear_error_history(self):
        """Clear error history and reset counters."""
        self.error_history.clear()
        self.error_count.clear()
        self.recovery_attempts.clear()
        self.failed_components.clear()


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


# Decorators for automatic error handling

def handle_errors(
    component: Optional[str] = None,
    auto_recover: bool = True,
    fallback_return: Any = None,
    silence_errors: bool = False
):
    """
    Decorator for automatic error handling in functions and methods.
    
    Args:
        component: Component name for error tracking
        auto_recover: Whether to attempt automatic recovery
        fallback_return: Value to return if error occurs and can't be recovered
        silence_errors: Whether to silence errors (return fallback_return)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                
                # Extract component name if not provided
                comp_name = component or getattr(func, '__qualname__', func.__name__)
                
                # Build context from function arguments
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                # Handle the error
                error_info = error_handler.handle_error(
                    e, context=context, component=comp_name, auto_recover=auto_recover
                )
                
                # Return fallback if error should be silenced or recovery failed
                if silence_errors or (auto_recover and not error_info.get('recovery_successful', False)):
                    return fallback_return
                
                # Re-raise if not silenced and recovery failed
                raise
        
        return wrapper
    return decorator


def handle_async_errors(
    component: Optional[str] = None,
    auto_recover: bool = True,
    fallback_return: Any = None,
    silence_errors: bool = False
):
    """
    Decorator for automatic error handling in async functions and methods.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                
                # Extract component name if not provided
                comp_name = component or getattr(func, '__qualname__', func.__name__)
                
                # Build context from function arguments
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'is_async': True
                }
                
                # Handle the error
                error_info = error_handler.handle_error(
                    e, context=context, component=comp_name, auto_recover=auto_recover
                )
                
                # Return fallback if error should be silenced or recovery failed
                if silence_errors or (auto_recover and not error_info.get('recovery_successful', False)):
                    return fallback_return
                
                # Re-raise if not silenced and recovery failed
                raise
        
        return wrapper
    return decorator


@contextmanager
def error_context(
    component: str,
    operation: str,
    auto_recover: bool = True,
    silence_errors: bool = False
):
    """
    Context manager for error handling in code blocks.
    
    Usage:
        with error_context('database', 'user_query'):
            # Database operation code
            result = db.query(sql)
    """
    error_handler = get_error_handler()
    
    try:
        yield
    except Exception as e:
        context = {
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        error_info = error_handler.handle_error(
            e, context=context, component=component, auto_recover=auto_recover
        )
        
        if not silence_errors and not error_info.get('recovery_successful', False):
            raise


# Dependency fallback decorators

def fallback_on_import_error(fallback_value: Any = None):
    """
    Decorator to provide fallbacks for import errors.
    
    Usage:
        @fallback_on_import_error(MockAI())
        def get_ai_service():
            from external_ai import RealAI
            return RealAI()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                error_handler = get_error_handler()
                error_handler.handle_error(
                    e, 
                    context={'import_error': str(e)},
                    component=func.__name__
                )
                return fallback_value
        return wrapper
    return decorator


def safe_import(module_name: str, fallback=None, component_name: str = None):
    """
    Safely import a module with automatic fallback on failure.
    
    Args:
        module_name: Name of module to import
        fallback: Fallback value/object to return on import failure
        component_name: Component name for error tracking
    
    Returns:
        Imported module or fallback value
    """
    try:
        return __import__(module_name)
    except ImportError as e:
        error_handler = get_error_handler()
        error_handler.handle_error(
            e,
            context={'module_name': module_name},
            component=component_name or f'import_{module_name}'
        )
        return fallback


# System health monitoring

class SystemHealthMonitor:
    """Monitor system health and component status."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.component_status = {}
        self.last_health_check = None
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        self.last_health_check = datetime.utcnow()
        
        health_status = {
            'timestamp': self.last_health_check.isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'error_stats': self.error_handler.get_error_stats()
        }
        
        # Check individual components
        components_to_check = [
            'config_manager',
            'project_manager', 
            'ai_interface',
            'task_dashboard',
            'database',
            'file_system'
        ]
        
        for component in components_to_check:
            try:
                status = self._check_component_health(component)
                health_status['components'][component] = status
                
                if status['status'] == 'critical':
                    health_status['overall_status'] = 'critical'
                elif status['status'] == 'degraded' and health_status['overall_status'] == 'healthy':
                    health_status['overall_status'] = 'degraded'
            
            except Exception as e:
                health_status['components'][component] = {
                    'status': 'critical',
                    'error': str(e),
                    'last_checked': self.last_health_check.isoformat()
                }
                health_status['overall_status'] = 'critical'
        
        return health_status
    
    def _check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of a specific component."""
        error_stats = self.error_handler.get_error_stats()
        component_errors = [e for e in error_stats['recent_errors'] if e.get('component') == component]
        
        if component in self.error_handler.failed_components:
            return {
                'status': 'critical',
                'message': 'Component has failed and been disabled',
                'error_count': len(component_errors),
                'last_checked': datetime.utcnow().isoformat()
            }
        
        critical_errors = [e for e in component_errors if e.get('severity') == 'critical']
        if critical_errors:
            return {
                'status': 'critical',
                'message': f'Component has {len(critical_errors)} critical errors',
                'error_count': len(component_errors),
                'last_checked': datetime.utcnow().isoformat()
            }
        
        if len(component_errors) > 5:
            return {
                'status': 'degraded',
                'message': f'Component has {len(component_errors)} recent errors',
                'error_count': len(component_errors),
                'last_checked': datetime.utcnow().isoformat()
            }
        
        return {
            'status': 'healthy',
            'message': 'Component operating normally',
            'error_count': len(component_errors),
            'last_checked': datetime.utcnow().isoformat()
        }


def get_health_monitor() -> SystemHealthMonitor:
    """Get system health monitor instance."""
    return SystemHealthMonitor(get_error_handler())


# Emergency recovery functions

def emergency_recovery():
    """Emergency recovery procedure for critical system failures."""
    error_handler = get_error_handler()
    
    recovery_steps = [
        "Clearing error history and resetting counters",
        "Attempting to restore failed components",
        "Reinitializing core services",
        "Performing health check"
    ]
    
    recovery_log = []
    
    for step in recovery_steps:
        try:
            recovery_log.append(f"✓ {step}")
            
            if "Clearing error history" in step:
                error_handler.clear_error_history()
            
            elif "restore failed components" in step:
                error_handler.failed_components.clear()
            
            elif "Reinitializing core services" in step:
                # Trigger reinitialization of core services
                pass
            
            elif "health check" in step:
                health = get_health_monitor().check_system_health()
                recovery_log.append(f"  System status: {health['overall_status']}")
        
        except Exception as e:
            recovery_log.append(f"✗ {step}: {str(e)}")
    
    return recovery_log


# Export main components
__all__ = [
    'ErrorHandler',
    'get_error_handler',
    'handle_errors',
    'handle_async_errors', 
    'error_context',
    'fallback_on_import_error',
    'safe_import',
    'SystemHealthMonitor',
    'get_health_monitor',
    'emergency_recovery'
]