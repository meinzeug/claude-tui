"""
Advanced Logging System for claude-tiu.

This module provides a sophisticated logging framework with:
- Structured logging with context
- Performance monitoring
- Security event tracking
- Anti-hallucination validation logging
- Real-time log streaming for TUI

Key Features:
- JSON structured logging for machine parsing
- Context-aware logging with request tracing
- Performance metrics integration
- Security audit trail
- Configurable log levels and outputs
- Memory-efficient circular buffer for UI logs
"""

import json
import logging
import logging.handlers
import sys
import time
from collections import deque
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import psutil
from rich.console import Console
from rich.logging import RichHandler

from .types import LogLevel, LogContext, PerformanceMetrics

# Context variable for request tracking
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


class ContextFilter(logging.Filter):
    """Add context information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        context = request_context.get({})
        
        # Add context fields
        record.request_id = context.get('request_id', 'unknown')
        record.user_id = context.get('user_id', 'anonymous')
        record.project_id = context.get('project_id', 'none')
        record.session_id = context.get('session_id', 'none')
        
        # Add performance metrics
        record.memory_usage = psutil.Process().memory_info().rss // 1024 // 1024  # MB
        record.cpu_percent = psutil.cpu_percent()
        
        return True


class SecurityAuditHandler(logging.Handler):
    """Special handler for security-related events."""
    
    def __init__(self, audit_file: Path):
        super().__init__()
        self.audit_file = audit_file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit security audit log."""
        if hasattr(record, 'security_event') and record.security_event:
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'event_type': getattr(record, 'event_type', 'unknown'),
                'user_id': getattr(record, 'user_id', 'anonymous'),
                'request_id': getattr(record, 'request_id', 'unknown'),
                'severity': getattr(record, 'severity', 'low'),
                'metadata': getattr(record, 'metadata', {})
            }
            
            with open(self.audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')


class TUILogBuffer:
    """Circular buffer for TUI log display with filtering."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.filters = {}
        
    def add_log(self, record: logging.LogRecord) -> None:
        """Add log record to buffer."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'request_id': getattr(record, 'request_id', 'unknown'),
            'project_id': getattr(record, 'project_id', 'none')
        }
        self.buffer.append(log_entry)
    
    def get_logs(
        self,
        max_count: Optional[int] = None,
        level_filter: Optional[LogLevel] = None,
        module_filter: Optional[str] = None,
        project_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered logs from buffer."""
        logs = list(self.buffer)
        
        # Apply filters
        if level_filter:
            logs = [log for log in logs if log['level'] == level_filter.upper()]
        
        if module_filter:
            logs = [log for log in logs if module_filter in log['module']]
            
        if project_filter:
            logs = [log for log in logs if log['project_id'] == project_filter]
        
        # Limit count
        if max_count:
            logs = logs[-max_count:]
        
        return logs


class TUILogHandler(logging.Handler):
    """Handler that feeds logs to TUI buffer."""
    
    def __init__(self, buffer: TUILogBuffer):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add record to TUI buffer."""
        self.buffer.add_log(record)


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers = {}
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = str(uuid4())
        self._timers[timer_id] = {
            'operation': operation,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss
        }
        return timer_id
    
    def end_timer(self, timer_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End timing and log performance metrics."""
        if timer_id not in self._timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return
        
        timer = self._timers.pop(timer_id)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        duration = end_time - timer['start_time']
        memory_delta = end_memory - timer['start_memory']
        
        perf_data = {
            'operation': timer['operation'],
            'duration_seconds': duration,
            'memory_delta_mb': memory_delta // 1024 // 1024,
            'start_memory_mb': timer['start_memory'] // 1024 // 1024,
            'end_memory_mb': end_memory // 1024 // 1024,
            'metadata': metadata or {}
        }
        
        self.logger.info(
            f"Performance: {timer['operation']} completed in {duration:.3f}s",
            extra={'performance_data': perf_data}
        )
    
    def log_metrics(self, operation: str, metrics: PerformanceMetrics) -> None:
        """Log structured performance metrics."""
        self.logger.info(
            f"Metrics: {operation}",
            extra={
                'performance_metrics': {
                    'operation': operation,
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'memory_used_mb': metrics.memory_used // 1024 // 1024,
                    'active_tasks': metrics.active_tasks,
                    'cache_hit_rate': metrics.cache_hit_rate,
                    'ai_response_time': metrics.ai_response_time,
                    'timestamp': metrics.timestamp.isoformat()
                }
            }
        )


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_security_event(
        self,
        event_type: str,
        message: str,
        severity: str = 'medium',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event."""
        self.logger.warning(
            message,
            extra={
                'security_event': True,
                'event_type': event_type,
                'severity': severity,
                'metadata': metadata or {}
            }
        )
    
    def log_authentication_attempt(
        self,
        success: bool,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """Log authentication attempt."""
        self.log_security_event(
            'authentication',
            f"Authentication {'successful' if success else 'failed'}",
            severity='low' if success else 'high',
            metadata={
                'success': success,
                'user_id': user_id,
                'ip_address': ip_address
            }
        )
    
    def log_api_key_access(self, service: str, operation: str) -> None:
        """Log API key access."""
        self.log_security_event(
            'api_key_access',
            f"API key {operation} for service {service}",
            severity='low',
            metadata={'service': service, 'operation': operation}
        )
    
    def log_input_validation_failure(
        self,
        input_type: str,
        validation_error: str,
        input_sample: Optional[str] = None
    ) -> None:
        """Log input validation failure."""
        self.log_security_event(
            'input_validation',
            f"Input validation failed for {input_type}: {validation_error}",
            severity='medium',
            metadata={
                'input_type': input_type,
                'validation_error': validation_error,
                'input_sample': input_sample[:100] if input_sample else None
            }
        )


class LoggerManager:
    """Central logging management system."""
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        enable_tui: bool = True
    ):
        """
        Initialize logging manager.
        
        Args:
            log_dir: Directory for log files
            log_level: Minimum log level
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_json: Enable structured JSON logging
            enable_tui: Enable TUI log buffer
        """
        self.log_dir = log_dir or Path.home() / '.claude-tiu' / 'logs'
        self.log_level = log_level
        self.tui_buffer = TUILogBuffer() if enable_tui else None
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add context filter
        context_filter = ContextFilter()
        self.root_logger.addFilter(context_filter)
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handler()
        
        if enable_json:
            self._setup_json_handler()
        
        if enable_tui and self.tui_buffer:
            self._setup_tui_handler()
        
        # Setup security audit handler
        self._setup_security_handler()
        
        # Create specialized loggers
        self.performance_logger = PerformanceLogger(logging.getLogger('performance'))
        self.security_logger = SecurityLogger(logging.getLogger('security'))
        
        # Log initialization
        self.root_logger.info("Logging system initialized", extra={
            'log_dir': str(self.log_dir),
            'log_level': log_level,
            'handlers': len(self.root_logger.handlers)
        })
    
    def _setup_console_handler(self) -> None:
        """Setup rich console handler."""
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        handler.setLevel(getattr(logging, self.log_level.upper()))
        
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        self.root_logger.addHandler(handler)
    
    def _setup_file_handler(self) -> None:
        """Setup rotating file handler."""
        log_file = self.log_dir / 'claude-tiu.log'
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setLevel(getattr(logging, self.log_level.upper()))
        
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        self.root_logger.addHandler(handler)
    
    def _setup_json_handler(self) -> None:
        """Setup structured JSON logging."""
        json_log_file = self.log_dir / 'claude-tiu.json'
        handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setLevel(getattr(logging, self.log_level.upper()))
        
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        
        self.root_logger.addHandler(handler)
    
    def _setup_tui_handler(self) -> None:
        """Setup TUI buffer handler."""
        handler = TUILogHandler(self.tui_buffer)
        handler.setLevel(getattr(logging, self.log_level.upper()))
        self.root_logger.addHandler(handler)
    
    def _setup_security_handler(self) -> None:
        """Setup security audit handler."""
        audit_file = self.log_dir / 'security-audit.log'
        handler = SecurityAuditHandler(audit_file)
        handler.setLevel(logging.WARNING)
        
        self.root_logger.addHandler(handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance for module."""
        return logging.getLogger(name)
    
    def get_performance_logger(self) -> PerformanceLogger:
        """Get performance logger instance."""
        return self.performance_logger
    
    def get_security_logger(self) -> SecurityLogger:
        """Get security logger instance."""
        return self.security_logger
    
    def get_tui_logs(
        self,
        max_count: Optional[int] = None,
        level_filter: Optional[LogLevel] = None,
        module_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get logs for TUI display."""
        if not self.tui_buffer:
            return []
        
        return self.tui_buffer.get_logs(
            max_count=max_count,
            level_filter=level_filter,
            module_filter=module_filter
        )
    
    def set_context(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Set logging context for current request."""
        context = {
            'request_id': request_id or str(uuid4()),
            'user_id': user_id or 'anonymous',
            'project_id': project_id or 'none',
            'session_id': session_id or 'none'
        }
        request_context.set(context)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'request_id': getattr(record, 'request_id', 'unknown'),
            'user_id': getattr(record, 'user_id', 'anonymous'),
            'project_id': getattr(record, 'project_id', 'none'),
            'session_id': getattr(record, 'session_id', 'none'),
            'memory_usage_mb': getattr(record, 'memory_usage', 0),
            'cpu_percent': getattr(record, 'cpu_percent', 0.0)
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))


# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: LogLevel = LogLevel.INFO,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True,
    enable_tui: bool = True
) -> logging.Logger:
    """
    Setup global logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum log level
        enable_console: Enable console output
        enable_file: Enable file logging
        enable_json: Enable structured JSON logging
        enable_tui: Enable TUI log buffer
        
    Returns:
        Root logger instance
    """
    global _logger_manager
    
    _logger_manager = LoggerManager(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_json=enable_json,
        enable_tui=enable_tui
    )
    
    return _logger_manager.root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    if _logger_manager is None:
        setup_logging()
    
    return _logger_manager.get_logger(name)


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    if _logger_manager is None:
        setup_logging()
    
    return _logger_manager.get_performance_logger()


def get_security_logger() -> SecurityLogger:
    """Get security logger instance."""
    if _logger_manager is None:
        setup_logging()
    
    return _logger_manager.get_security_logger()


def set_logging_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> None:
    """
    Set logging context for current request.
    
    Args:
        request_id: Unique request identifier
        user_id: User identifier
        project_id: Project identifier
        session_id: Session identifier
    """
    if _logger_manager is None:
        setup_logging()
    
    _logger_manager.set_context(
        request_id=request_id,
        user_id=user_id,
        project_id=project_id,
        session_id=session_id
    )


def get_tui_logs(
    max_count: Optional[int] = None,
    level_filter: Optional[LogLevel] = None,
    module_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get logs for TUI display.
    
    Args:
        max_count: Maximum number of logs to return
        level_filter: Filter by log level
        module_filter: Filter by module name
        
    Returns:
        List of log entries for TUI display
    """
    if _logger_manager is None:
        setup_logging()
    
    return _logger_manager.get_tui_logs(
        max_count=max_count,
        level_filter=level_filter,
        module_filter=module_filter
    )