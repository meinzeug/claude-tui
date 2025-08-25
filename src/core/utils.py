"""
Utility functions and helpers for claude-tiu core modules.

This module provides common utilities for file operations, error handling,
logging setup, and other supporting functionality needed across core components.

Key Features:
- Safe file system operations with error handling
- Logging configuration and management
- Path utilities and validation
- Async helper functions
- Error handling decorators
"""

import asyncio
import functools
import hashlib
import logging
import logging.handlers
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
import sys

from .types import PathStr


# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class FileSystemError(Exception):
    """File system operation errors."""
    pass


class LoggingSetupError(Exception):
    """Logging setup errors."""
    pass


def setup_logging(
    level: str = "INFO",
    log_file: Optional[PathStr] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        enable_console: Enable console logging
        
    Returns:
        Configured root logger
    """
    # Create root logger
    logger = logging.getLogger('claude-tiu')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # Less verbose for console
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    return logger


def error_handler(
    reraise: bool = True,
    default_return: Any = None,
    log_errors: bool = True
) -> Callable[[F], F]:
    """
    Decorator for handling errors in functions.
    
    Args:
        reraise: Whether to reraise the exception after logging
        default_return: Default value to return on error (if not reraising)
        log_errors: Whether to log errors
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if log_errors:
                        logger = logging.getLogger(func.__module__)
                        logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                    
                    if reraise:
                        raise
                    return default_return
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_errors:
                        logger = logging.getLogger(func.__module__)
                        logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                    
                    if reraise:
                        raise
                    return default_return
            
            return sync_wrapper
    
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """
    Decorator for retrying functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exceptions that trigger retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt < max_attempts - 1:  # Don't delay after final attempt
                            logger = logging.getLogger(func.__module__)
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff_multiplier
                
                # Re-raise the last exception if all attempts failed
                if last_exception:
                    raise last_exception
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                import time
                
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt < max_attempts - 1:  # Don't delay after final attempt
                            logger = logging.getLogger(func.__module__)
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                            time.sleep(current_delay)
                            current_delay *= backoff_multiplier
                
                # Re-raise the last exception if all attempts failed
                if last_exception:
                    raise last_exception
            
            return sync_wrapper
    
    return decorator


class SafeFileOperations:
    """
    Safe file operations with error handling and validation.
    """
    
    @staticmethod
    @error_handler(reraise=True)
    def read_file(
        file_path: PathStr,
        encoding: str = 'utf-8',
        max_size: int = 50 * 1024 * 1024  # 50MB max
    ) -> str:
        """
        Safely read file content with size limits.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            max_size: Maximum file size to read
            
        Returns:
            File content as string
            
        Raises:
            FileSystemError: If file operation fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileSystemError(f"File does not exist: {path}")
        
        if not path.is_file():
            raise FileSystemError(f"Path is not a file: {path}")
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > max_size:
            raise FileSystemError(
                f"File too large ({file_size} bytes, max {max_size}): {path}"
            )
        
        try:
            return path.read_text(encoding=encoding, errors='ignore')
        except (IOError, OSError, UnicodeError) as e:
            raise FileSystemError(f"Failed to read file {path}: {e}") from e
    
    @staticmethod
    @error_handler(reraise=True)
    def write_file(
        file_path: PathStr,
        content: str,
        encoding: str = 'utf-8',
        create_dirs: bool = True,
        backup_existing: bool = False
    ) -> None:
        """
        Safely write content to file.
        
        Args:
            file_path: Path to file
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if they don't exist
            backup_existing: Create backup of existing file
            
        Raises:
            FileSystemError: If file operation fails
        """
        path = Path(file_path)
        
        # Create parent directories if requested
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if requested and file exists
        if backup_existing and path.exists():
            backup_path = path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}{path.suffix}')
            shutil.copy2(path, backup_path)
        
        try:
            # Write to temporary file first, then rename (atomic operation)
            temp_path = path.with_suffix(f'.tmp.{os.getpid()}')
            temp_path.write_text(content, encoding=encoding)
            temp_path.replace(path)
        except (IOError, OSError, UnicodeError) as e:
            # Clean up temporary file if it exists
            temp_path = path.with_suffix(f'.tmp.{os.getpid()}')
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise FileSystemError(f"Failed to write file {path}: {e}") from e
    
    @staticmethod
    @error_handler(reraise=True)
    def copy_file(
        source: PathStr,
        destination: PathStr,
        create_dirs: bool = True,
        overwrite: bool = False
    ) -> None:
        """
        Safely copy file with validation.
        
        Args:
            source: Source file path
            destination: Destination file path
            create_dirs: Create destination directories
            overwrite: Allow overwriting existing files
            
        Raises:
            FileSystemError: If copy operation fails
        """
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            raise FileSystemError(f"Source file does not exist: {src_path}")
        
        if not src_path.is_file():
            raise FileSystemError(f"Source is not a file: {src_path}")
        
        if dst_path.exists() and not overwrite:
            raise FileSystemError(f"Destination already exists: {dst_path}")
        
        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(src_path, dst_path)
        except (IOError, OSError) as e:
            raise FileSystemError(f"Failed to copy {src_path} to {dst_path}: {e}") from e
    
    @staticmethod
    @error_handler(reraise=True)
    def move_file(
        source: PathStr,
        destination: PathStr,
        create_dirs: bool = True,
        overwrite: bool = False
    ) -> None:
        """
        Safely move file with validation.
        
        Args:
            source: Source file path
            destination: Destination file path
            create_dirs: Create destination directories
            overwrite: Allow overwriting existing files
            
        Raises:
            FileSystemError: If move operation fails
        """
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            raise FileSystemError(f"Source file does not exist: {src_path}")
        
        if dst_path.exists() and not overwrite:
            raise FileSystemError(f"Destination already exists: {dst_path}")
        
        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(src_path), str(dst_path))
        except (IOError, OSError) as e:
            raise FileSystemError(f"Failed to move {src_path} to {dst_path}: {e}") from e
    
    @staticmethod
    @error_handler(reraise=True)
    def delete_file(file_path: PathStr, missing_ok: bool = True) -> None:
        """
        Safely delete file.
        
        Args:
            file_path: Path to file
            missing_ok: Don't raise error if file doesn't exist
            
        Raises:
            FileSystemError: If delete operation fails
        """
        path = Path(file_path)
        
        if not path.exists():
            if not missing_ok:
                raise FileSystemError(f"File does not exist: {path}")
            return
        
        if not path.is_file():
            raise FileSystemError(f"Path is not a file: {path}")
        
        try:
            path.unlink()
        except (IOError, OSError) as e:
            raise FileSystemError(f"Failed to delete file {path}: {e}") from e
    
    @staticmethod
    @error_handler(reraise=True)
    def create_directory(
        dir_path: PathStr,
        parents: bool = True,
        exist_ok: bool = True
    ) -> None:
        """
        Safely create directory.
        
        Args:
            dir_path: Directory path
            parents: Create parent directories
            exist_ok: Don't raise error if directory exists
            
        Raises:
            FileSystemError: If directory creation fails
        """
        path = Path(dir_path)
        
        try:
            path.mkdir(parents=parents, exist_ok=exist_ok)
        except (IOError, OSError) as e:
            raise FileSystemError(f"Failed to create directory {path}: {e}") from e
    
    @staticmethod
    @error_handler(reraise=True)
    def delete_directory(
        dir_path: PathStr,
        recursive: bool = False,
        missing_ok: bool = True
    ) -> None:
        """
        Safely delete directory.
        
        Args:
            dir_path: Directory path
            recursive: Delete recursively
            missing_ok: Don't raise error if directory doesn't exist
            
        Raises:
            FileSystemError: If deletion fails
        """
        path = Path(dir_path)
        
        if not path.exists():
            if not missing_ok:
                raise FileSystemError(f"Directory does not exist: {path}")
            return
        
        if not path.is_dir():
            raise FileSystemError(f"Path is not a directory: {path}")
        
        try:
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        except (IOError, OSError) as e:
            raise FileSystemError(f"Failed to delete directory {path}: {e}") from e
    
    @staticmethod
    def get_file_hash(file_path: PathStr, algorithm: str = 'md5') -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            Hexadecimal hash string
            
        Raises:
            FileSystemError: If hash calculation fails
        """
        path = Path(file_path)
        
        if not path.exists() or not path.is_file():
            raise FileSystemError(f"File does not exist: {path}")
        
        try:
            hash_obj = hashlib.new(algorithm)
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except (IOError, OSError, ValueError) as e:
            raise FileSystemError(f"Failed to calculate hash for {path}: {e}") from e
    
    @staticmethod
    def get_file_info(file_path: PathStr) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
            
        Raises:
            FileSystemError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileSystemError(f"File does not exist: {path}")
        
        try:
            stat = path.stat()
            return {
                'path': str(path.absolute()),
                'name': path.name,
                'suffix': path.suffix,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'accessed': datetime.fromtimestamp(stat.st_atime),
                'is_file': path.is_file(),
                'is_dir': path.is_dir(),
                'permissions': oct(stat.st_mode)[-3:]
            }
        except (IOError, OSError) as e:
            raise FileSystemError(f"Failed to get file info for {path}: {e}") from e


class AsyncFileOperations:
    """
    Async file operations for non-blocking I/O.
    """
    
    @staticmethod
    async def read_file_async(
        file_path: PathStr,
        encoding: str = 'utf-8',
        max_size: int = 50 * 1024 * 1024
    ) -> str:
        """
        Asynchronously read file content.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            max_size: Maximum file size
            
        Returns:
            File content as string
        """
        def _read():
            return SafeFileOperations.read_file(file_path, encoding, max_size)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read)
    
    @staticmethod
    async def write_file_async(
        file_path: PathStr,
        content: str,
        encoding: str = 'utf-8',
        create_dirs: bool = True,
        backup_existing: bool = False
    ) -> None:
        """
        Asynchronously write file content.
        
        Args:
            file_path: Path to file
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories
            backup_existing: Create backup of existing file
        """
        def _write():
            SafeFileOperations.write_file(
                file_path, content, encoding, create_dirs, backup_existing
            )
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write)
    
    @staticmethod
    async def copy_file_async(
        source: PathStr,
        destination: PathStr,
        create_dirs: bool = True,
        overwrite: bool = False
    ) -> None:
        """
        Asynchronously copy file.
        
        Args:
            source: Source file path
            destination: Destination file path
            create_dirs: Create destination directories
            overwrite: Allow overwriting existing files
        """
        def _copy():
            SafeFileOperations.copy_file(source, destination, create_dirs, overwrite)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _copy)


def validate_path(
    path: PathStr,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        path: Path to validate
        must_exist: Path must exist
        must_be_file: Path must be a file
        must_be_dir: Path must be a directory
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If validation fails
    """
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise ValueError(f"Invalid path type: {type(path)}")
    
    # Resolve to absolute path
    path = path.resolve()
    
    # Check existence
    if must_exist and not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if path.exists():
        # Check file/directory type
        if must_be_file and not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        if must_be_dir and not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
    
    # Check extension
    if allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise ValueError(f"Invalid file extension {path.suffix}. Allowed: {allowed_extensions}")
    
    return path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def create_temp_file(
    content: str = "",
    suffix: str = ".tmp",
    prefix: str = "claude-tiu-",
    encoding: str = 'utf-8'
) -> Path:
    """
    Create temporary file with content.
    
    Args:
        content: Initial content
        suffix: File suffix
        prefix: File prefix
        encoding: Text encoding
        
    Returns:
        Path to temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, text=True)
    
    try:
        with os.fdopen(fd, 'w', encoding=encoding) as f:
            f.write(content)
        return Path(temp_path)
    except Exception:
        # Clean up file descriptor if writing fails
        try:
            os.close(fd)
        except OSError:
            pass
        # Remove temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def create_temp_directory(prefix: str = "claude-tiu-") -> Path:
    """
    Create temporary directory.
    
    Args:
        prefix: Directory prefix
        
    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir)


class ContextTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Operation name for logging
            logger: Logger instance
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Enter context."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"{self.name} completed in {duration:.3f}s")
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Async context timer
class AsyncContextTimer:
    """Async context manager for timing operations."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        """
        Initialize async timer.
        
        Args:
            name: Operation name for logging
            logger: Logger instance
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    async def __aenter__(self):
        """Enter async context."""
        import time
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"{self.name} completed in {duration:.3f}s")
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Convenience functions

def ensure_directory(path: PathStr) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_project_root(marker_files: List[str] = None) -> Optional[Path]:
    """
    Find project root by looking for marker files.
    
    Args:
        marker_files: Files that indicate project root
        
    Returns:
        Project root path or None if not found
    """
    if marker_files is None:
        marker_files = ['.git', 'setup.py', 'pyproject.toml', 'package.json']
    
    current_path = Path.cwd()
    
    while current_path != current_path.parent:
        for marker in marker_files:
            if (current_path / marker).exists():
                return current_path
        current_path = current_path.parent
    
    return None


def normalize_path(path: PathStr, relative_to: Optional[PathStr] = None) -> str:
    """
    Normalize path for consistent representation.
    
    Args:
        path: Path to normalize
        relative_to: Make path relative to this directory
        
    Returns:
        Normalized path string
    """
    normalized = Path(path).resolve()
    
    if relative_to:
        try:
            relative_to_path = Path(relative_to).resolve()
            normalized = normalized.relative_to(relative_to_path)
        except ValueError:
            # Path is not relative to the specified directory
            pass
    
    return str(normalized).replace('\\', '/')  # Use forward slashes consistently


# Initialize default logger
_default_logger = None

def get_logger(name: str = 'claude-tiu') -> logging.Logger:
    """Get logger instance with default configuration."""
    global _default_logger
    
    if _default_logger is None:
        _default_logger = setup_logging()
    
    return logging.getLogger(name)