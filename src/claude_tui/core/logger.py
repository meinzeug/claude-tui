"""
Logging Configuration - Rich-enhanced logging system.

Provides comprehensive logging with Rich formatting, structured output,
and integration with the TUI application for real-time log display.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import install as rich_traceback_install

# Configure rich traceback for better error display
rich_traceback_install(show_locals=True)

# Create console instance for logging
console = Console(stderr=True, width=120)


class TUILogHandler(logging.Handler):
    """Custom log handler for TUI application integration."""
    
    def __init__(self, log_callback=None):
        super().__init__()
        self.log_callback = log_callback
        self.logs = []
        
    def emit(self, record):
        """Emit a log record to the TUI interface."""
        try:
            msg = self.format(record)
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created),
                'level': record.levelname,
                'logger': record.name,
                'message': msg,
                'module': getattr(record, 'module', ''),
                'funcName': getattr(record, 'funcName', ''),
                'lineno': getattr(record, 'lineno', 0)
            }
            
            self.logs.append(log_entry)
            
            # Keep only last 1000 log entries to prevent memory issues
            if len(self.logs) > 1000:
                self.logs = self.logs[-500:]  # Keep last 500
                
            if self.log_callback:
                self.log_callback(log_entry)
                
        except Exception:
            self.handleError(record)
    
    def get_recent_logs(self, count: int = 100):
        """Get recent log entries."""
        return self.logs[-count:]


def setup_logging(
    debug: bool = False,
    log_file: Optional[Path] = None,
    log_to_file: bool = True,
    log_callback=None
) -> Dict[str, Any]:
    """
    Setup comprehensive logging configuration.
    
    Args:
        debug: Enable debug level logging
        log_file: Custom log file path
        log_to_file: Whether to log to file
        log_callback: Callback function for TUI log integration
        
    Returns:
        Dictionary with logging configuration details
    """
    
    # Determine log level
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)-30s | %(levelname)-8s | %(funcName)-20s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handlers_info = []
    
    # Console handler with Rich formatting
    if not sys.stderr.isatty():
        # For non-TTY environments, use standard console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)
    else:
        # Use Rich handler for TTY environments
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            enable_link_path=debug,
            rich_tracebacks=True,
            tracebacks_show_locals=debug
        )
    
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    handlers_info.append({
        'type': 'console',
        'level': logging.getLevelName(log_level),
        'formatter': 'rich' if sys.stderr.isatty() else 'standard'
    })
    
    # File handler setup
    if log_to_file:
        if log_file is None:
            log_file = Path.home() / ".claude-tui" / "logs" / "claude-tui.log"
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler to prevent log files from getting too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        file_handler.setLevel(logging.DEBUG)  # Always debug level for file
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        handlers_info.append({
            'type': 'file',
            'path': str(log_file),
            'level': 'DEBUG',
            'max_size': '10MB',
            'backup_count': 5
        })
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.debug(f"Log level: {logging.getLevelName(log_level)}")
    
    if log_to_file:
        logger.info(f"File logging enabled: {log_file}")
    
    return {
        'level': logging.getLevelName(log_level),
        'handlers': handlers_info,
        'debug_enabled': debug,
        'log_file': str(log_file) if log_file else None
    }


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


# Module-level logger for this file
logger = get_logger(__name__)