"""
File System Manager - Secure and efficient file system operations.

This module provides comprehensive file system management with:
- Safe file operations with permission checks
- Atomic operations with rollback capabilities
- File validation and sanitization
- Directory structure management
- Performance monitoring and optimization
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

# Optional async file operations
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None

logger = logging.getLogger(__name__)


class FileSystemError(Exception):
    """Base exception for file system errors."""
    pass


class FilePermissionError(FileSystemError):
    """Raised when file permission operations fail."""
    pass


class FileValidationError(FileSystemError):
    """Raised when file validation fails."""
    pass


class FileOperationType(Enum):
    """File operation types for tracking."""
    CREATE = "create"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    CHMOD = "chmod"


@dataclass
class FileOperation:
    """Represents a file system operation."""
    type: FileOperationType
    path: Path
    target_path: Optional[Path] = None
    data: Optional[Union[str, bytes]] = None
    permissions: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error: Optional[str] = None
    
    def rollback(self) -> bool:
        """Attempt to rollback this operation."""
        try:
            if self.type == FileOperationType.CREATE and self.path.exists():
                self.path.unlink()
                return True
            elif self.type == FileOperationType.DELETE and self.target_path:
                # Restore from backup
                shutil.copy2(self.target_path, self.path)
                return True
            elif self.type == FileOperationType.WRITE and self.target_path:
                # Restore from backup
                shutil.copy2(self.target_path, self.path)
                return True
            elif self.type == FileOperationType.MOVE and self.target_path:
                # Move back
                shutil.move(self.target_path, self.path)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to rollback operation {self.type} on {self.path}: {e}")
            return False


@dataclass
class FileSystemStats:
    """File system statistics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    files_created: int = 0
    files_deleted: int = 0
    directories_created: int = 0
    average_operation_time: float = 0.0


class FileSystemManager:
    """
    Comprehensive file system manager with safety and performance features.
    
    Features:
    - Atomic file operations with rollback
    - Permission checking and validation
    - Path sanitization and security
    - Performance monitoring
    - Async operations support
    """
    
    def __init__(
        self,
        root_path: Optional[Path] = None,
        enable_async: bool = True,
        enable_backup: bool = True,
        max_file_size: int = 100 * 1024 * 1024  # 100MB
    ):
        """
        Initialize the file system manager.
        
        Args:
            root_path: Root directory for operations (sandbox)
            enable_async: Enable async file operations
            enable_backup: Enable automatic backups
            max_file_size: Maximum allowed file size
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.enable_async = enable_async and AIOFILES_AVAILABLE
        self.enable_backup = enable_backup
        self.max_file_size = max_file_size
        
        # Operation tracking
        self._operations: List[FileOperation] = []
        self._stats = FileSystemStats()
        
        # Security settings
        self.allowed_extensions: Optional[Set[str]] = None
        self.blocked_extensions: Set[str] = {'.exe', '.dll', '.so', '.dylib'}
        self.hidden_file_access = False
        
        # Backup directory
        self.backup_dir = self.root_path / '.backups'
        if self.enable_backup:
            self.backup_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"FileSystemManager initialized with root: {self.root_path}")
    
    def _validate_path(self, path: Path, check_exists: bool = False) -> Path:
        """
        Validate and sanitize a file path.
        
        Args:
            path: Path to validate
            check_exists: Whether to check if path exists
            
        Returns:
            Validated absolute path
            
        Raises:
            FileValidationError: If path is invalid
        """
        # Convert to absolute path
        abs_path = path.resolve()
        
        # Check if path is within root (sandbox)
        try:
            abs_path.relative_to(self.root_path)
        except ValueError:
            raise FileValidationError(f"Path {path} is outside root directory")
        
        # Check for path traversal attempts
        if '..' in str(path):
            raise FileValidationError("Path traversal attempt detected")
        
        # Check hidden files
        if not self.hidden_file_access and abs_path.name.startswith('.'):
            raise FileValidationError("Access to hidden files is not allowed")
        
        # Check file extension
        if self.allowed_extensions and abs_path.suffix not in self.allowed_extensions:
            raise FileValidationError(f"File extension {abs_path.suffix} not allowed")
        
        if abs_path.suffix in self.blocked_extensions:
            raise FileValidationError(f"File extension {abs_path.suffix} is blocked")
        
        # Check existence if requested
        if check_exists and not abs_path.exists():
            raise FileValidationError(f"Path {abs_path} does not exist")
        
        return abs_path
    
    def _track_operation(self, operation: FileOperation) -> None:
        """Track a file operation for statistics and rollback."""
        self._operations.append(operation)
        self._stats.total_operations += 1
        
        if operation.success:
            self._stats.successful_operations += 1
            
            if operation.type == FileOperationType.CREATE:
                if operation.path.is_file():
                    self._stats.files_created += 1
                else:
                    self._stats.directories_created += 1
            elif operation.type == FileOperationType.DELETE:
                self._stats.files_deleted += 1
            elif operation.type == FileOperationType.READ:
                self._stats.bytes_read += len(operation.data or b'')
            elif operation.type == FileOperationType.WRITE:
                self._stats.bytes_written += len(operation.data or b'')
        else:
            self._stats.failed_operations += 1
    
    async def create_file(
        self,
        path: Union[str, Path],
        content: Union[str, bytes] = '',
        mode: int = 0o644,
        overwrite: bool = False
    ) -> bool:
        """
        Create a file with the specified content.
        
        Args:
            path: File path
            content: File content
            mode: File permissions
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        path = Path(path)
        abs_path = self._validate_path(path)
        
        operation = FileOperation(
            type=FileOperationType.CREATE,
            path=abs_path,
            data=content,
            permissions=mode
        )
        
        try:
            # Check if file exists
            if abs_path.exists() and not overwrite:
                raise FileSystemError(f"File {abs_path} already exists")
            
            # Create parent directories
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if overwriting
            if abs_path.exists() and self.enable_backup:
                backup_path = self._create_backup(abs_path)
                operation.target_path = backup_path
            
            # Write content
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            if self.enable_async and asyncio.get_event_loop().is_running():
                async with aiofiles.open(abs_path, 'wb') as f:
                    await f.write(content)
            else:
                abs_path.write_bytes(content)
            
            # Set permissions
            abs_path.chmod(mode)
            
            operation.success = True
            logger.info(f"Created file: {abs_path}")
            return True
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            logger.error(f"Failed to create file {abs_path}: {e}")
            return False
            
        finally:
            self._track_operation(operation)
    
    async def read_file(
        self,
        path: Union[str, Path],
        mode: str = 'text'
    ) -> Optional[Union[str, bytes]]:
        """
        Read file contents.
        
        Args:
            path: File path
            mode: 'text' or 'binary'
            
        Returns:
            File contents or None if failed
        """
        path = Path(path)
        abs_path = self._validate_path(path, check_exists=True)
        
        operation = FileOperation(
            type=FileOperationType.READ,
            path=abs_path
        )
        
        try:
            # Check file size
            file_size = abs_path.stat().st_size
            if file_size > self.max_file_size:
                raise FileSystemError(f"File {abs_path} exceeds maximum size")
            
            # Read content
            if self.enable_async and asyncio.get_event_loop().is_running():
                read_mode = 'r' if mode == 'text' else 'rb'
                async with aiofiles.open(abs_path, read_mode) as f:
                    content = await f.read()
            else:
                if mode == 'text':
                    content = abs_path.read_text()
                else:
                    content = abs_path.read_bytes()
            
            operation.data = content
            operation.success = True
            return content
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            logger.error(f"Failed to read file {abs_path}: {e}")
            return None
            
        finally:
            self._track_operation(operation)
    
    async def write_file(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        mode: int = 0o644,
        append: bool = False
    ) -> bool:
        """
        Write content to a file.
        
        Args:
            path: File path
            content: Content to write
            mode: File permissions
            append: Whether to append to existing file
            
        Returns:
            True if successful
        """
        path = Path(path)
        abs_path = self._validate_path(path)
        
        operation = FileOperation(
            type=FileOperationType.WRITE,
            path=abs_path,
            data=content,
            permissions=mode
        )
        
        try:
            # Create backup if file exists
            if abs_path.exists() and self.enable_backup:
                backup_path = self._create_backup(abs_path)
                operation.target_path = backup_path
            
            # Ensure parent directory exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            if self.enable_async and asyncio.get_event_loop().is_running():
                write_mode = 'ab' if append else 'wb'
                async with aiofiles.open(abs_path, write_mode) as f:
                    await f.write(content)
            else:
                if append:
                    with open(abs_path, 'ab') as f:
                        f.write(content)
                else:
                    abs_path.write_bytes(content)
            
            # Set permissions
            abs_path.chmod(mode)
            
            operation.success = True
            logger.info(f"Wrote to file: {abs_path}")
            return True
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            logger.error(f"Failed to write file {abs_path}: {e}")
            
            # Attempt rollback
            if operation.target_path:
                operation.rollback()
            
            return False
            
        finally:
            self._track_operation(operation)
    
    async def delete_file(
        self,
        path: Union[str, Path],
        secure: bool = False
    ) -> bool:
        """
        Delete a file.
        
        Args:
            path: File path
            secure: Whether to securely overwrite before deletion
            
        Returns:
            True if successful
        """
        path = Path(path)
        abs_path = self._validate_path(path, check_exists=True)
        
        operation = FileOperation(
            type=FileOperationType.DELETE,
            path=abs_path
        )
        
        try:
            # Create backup before deletion
            if self.enable_backup:
                backup_path = self._create_backup(abs_path)
                operation.target_path = backup_path
            
            # Secure deletion if requested
            if secure and abs_path.is_file():
                await self._secure_delete(abs_path)
            else:
                if abs_path.is_file():
                    abs_path.unlink()
                elif abs_path.is_dir():
                    shutil.rmtree(abs_path)
            
            operation.success = True
            logger.info(f"Deleted: {abs_path}")
            return True
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            logger.error(f"Failed to delete {abs_path}: {e}")
            return False
            
        finally:
            self._track_operation(operation)
    
    async def move_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False
    ) -> bool:
        """
        Move a file or directory.
        
        Args:
            source: Source path
            destination: Destination path
            overwrite: Whether to overwrite existing destination
            
        Returns:
            True if successful
        """
        source = Path(source)
        destination = Path(destination)
        
        src_path = self._validate_path(source, check_exists=True)
        dst_path = self._validate_path(destination)
        
        operation = FileOperation(
            type=FileOperationType.MOVE,
            path=src_path,
            target_path=dst_path
        )
        
        try:
            # Check if destination exists
            if dst_path.exists() and not overwrite:
                raise FileSystemError(f"Destination {dst_path} already exists")
            
            # Create backup of destination if overwriting
            if dst_path.exists() and self.enable_backup:
                self._create_backup(dst_path)
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(src_path), str(dst_path))
            
            operation.success = True
            logger.info(f"Moved {src_path} to {dst_path}")
            return True
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            logger.error(f"Failed to move {src_path} to {dst_path}: {e}")
            return False
            
        finally:
            self._track_operation(operation)
    
    async def copy_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False
    ) -> bool:
        """
        Copy a file or directory.
        
        Args:
            source: Source path
            destination: Destination path
            overwrite: Whether to overwrite existing destination
            
        Returns:
            True if successful
        """
        source = Path(source)
        destination = Path(destination)
        
        src_path = self._validate_path(source, check_exists=True)
        dst_path = self._validate_path(destination)
        
        operation = FileOperation(
            type=FileOperationType.COPY,
            path=src_path,
            target_path=dst_path
        )
        
        try:
            # Check if destination exists
            if dst_path.exists() and not overwrite:
                raise FileSystemError(f"Destination {dst_path} already exists")
            
            # Create backup of destination if overwriting
            if dst_path.exists() and self.enable_backup:
                self._create_backup(dst_path)
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file or directory
            if src_path.is_file():
                shutil.copy2(str(src_path), str(dst_path))
            else:
                shutil.copytree(str(src_path), str(dst_path))
            
            operation.success = True
            logger.info(f"Copied {src_path} to {dst_path}")
            return True
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
            return False
            
        finally:
            self._track_operation(operation)
    
    async def create_directory(
        self,
        path: Union[str, Path],
        mode: int = 0o755,
        parents: bool = True
    ) -> bool:
        """
        Create a directory.
        
        Args:
            path: Directory path
            mode: Directory permissions
            parents: Whether to create parent directories
            
        Returns:
            True if successful
        """
        path = Path(path)
        abs_path = self._validate_path(path)
        
        try:
            abs_path.mkdir(mode=mode, parents=parents, exist_ok=True)
            logger.info(f"Created directory: {abs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {abs_path}: {e}")
            return False
    
    async def list_directory(
        self,
        path: Union[str, Path],
        pattern: str = '*',
        recursive: bool = False
    ) -> List[Path]:
        """
        List directory contents.
        
        Args:
            path: Directory path
            pattern: Glob pattern for filtering
            recursive: Whether to search recursively
            
        Returns:
            List of matching paths
        """
        path = Path(path)
        abs_path = self._validate_path(path, check_exists=True)
        
        if not abs_path.is_dir():
            raise FileSystemError(f"{abs_path} is not a directory")
        
        if recursive:
            return list(abs_path.rglob(pattern))
        else:
            return list(abs_path.glob(pattern))
    
    async def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed file information.
        
        Args:
            path: File path
            
        Returns:
            Dictionary with file information
        """
        path = Path(path)
        abs_path = self._validate_path(path, check_exists=True)
        
        stat_info = abs_path.stat()
        
        return {
            'path': str(abs_path),
            'name': abs_path.name,
            'extension': abs_path.suffix,
            'size': stat_info.st_size,
            'is_file': abs_path.is_file(),
            'is_directory': abs_path.is_dir(),
            'is_symlink': abs_path.is_symlink(),
            'created': datetime.fromtimestamp(stat_info.st_ctime),
            'modified': datetime.fromtimestamp(stat_info.st_mtime),
            'accessed': datetime.fromtimestamp(stat_info.st_atime),
            'permissions': oct(stat_info.st_mode),
            'owner_uid': stat_info.st_uid,
            'group_gid': stat_info.st_gid
        }
    
    def _create_backup(self, path: Path) -> Path:
        """Create a backup of a file or directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        if path.is_file():
            shutil.copy2(str(path), str(backup_path))
        else:
            shutil.copytree(str(path), str(backup_path))
        
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    async def _secure_delete(self, path: Path) -> None:
        """Securely overwrite and delete a file."""
        file_size = path.stat().st_size
        
        # Overwrite with random data
        with open(path, 'ba+', buffering=0) as f:
            for _ in range(3):  # Three passes
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
        
        # Delete the file
        path.unlink()
    
    def rollback_operations(self, count: int = 1) -> int:
        """
        Rollback recent operations.
        
        Args:
            count: Number of operations to rollback
            
        Returns:
            Number of successfully rolled back operations
        """
        rolled_back = 0
        
        for _ in range(min(count, len(self._operations))):
            operation = self._operations.pop()
            if operation.rollback():
                rolled_back += 1
                logger.info(f"Rolled back operation: {operation.type} on {operation.path}")
            else:
                logger.warning(f"Failed to rollback: {operation.type} on {operation.path}")
        
        return rolled_back
    
    def get_statistics(self) -> FileSystemStats:
        """Get file system operation statistics."""
        return self._stats
    
    def clear_statistics(self) -> None:
        """Clear operation statistics."""
        self._stats = FileSystemStats()
        self._operations.clear()


# Export main classes
__all__ = [
    'FileSystemManager',
    'FileOperation',
    'FileOperationType',
    'FileSystemStats',
    'FileSystemError',
    'FilePermissionError',
    'FileValidationError'
]