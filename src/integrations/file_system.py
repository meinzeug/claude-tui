"""
File System Manager Integration Module

Provides secure file system operations with comprehensive safety features:
- Safe file operations with permission checks
- Atomic file operations with rollback capabilities
- File validation and sanitization
- Directory structure management
- Backup and recovery mechanisms
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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple, BinaryIO, TextIO
import aiofiles
import aiofiles.os
import filecmp
import fnmatch
import magic
import psutil

logger = logging.getLogger(__name__)


class FileSystemError(Exception):
    """Base exception for file system operation errors"""
    pass


class FilePermissionError(FileSystemError):
    """Raised when file permission operations fail"""
    pass


class FileValidationError(FileSystemError):
    """Raised when file validation fails"""
    pass


class DirectoryError(FileSystemError):
    """Raised when directory operations fail"""
    pass


class BackupError(FileSystemError):
    """Raised when backup operations fail"""
    pass


class FileOperationStatus(Enum):
    """File operation status"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"


class FileType(Enum):
    """File type classification"""
    TEXT = "text"
    BINARY = "binary"
    IMAGE = "image"
    ARCHIVE = "archive"
    EXECUTABLE = "executable"
    CONFIGURATION = "configuration"
    SOURCE_CODE = "source_code"
    UNKNOWN = "unknown"


class SafetyLevel(Enum):
    """File operation safety levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    PARANOID = "paranoid"


@dataclass
class FileOperationResult:
    """Result from file system operations"""
    operation: str
    status: FileOperationStatus
    path: str
    message: str = ""
    files_affected: List[str] = field(default_factory=list)
    size_bytes: int = 0
    execution_time: float = 0.0
    backup_path: Optional[str] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        return self.status == FileOperationStatus.SUCCESS
    
    @property
    def size_human(self) -> str:
        """Human readable size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.size_bytes < 1024.0:
                return f"{self.size_bytes:.1f} {unit}"
            self.size_bytes /= 1024.0
        return f"{self.size_bytes:.1f} TB"


@dataclass
class FileInfo:
    """Comprehensive file information"""
    path: Path
    name: str
    size: int
    file_type: FileType
    mime_type: str
    permissions: str
    owner: str
    group: str
    created: datetime
    modified: datetime
    accessed: datetime
    checksum: str
    is_symlink: bool
    is_hidden: bool
    extension: str
    encoding: Optional[str] = None


@dataclass
class DirectoryStats:
    """Directory statistics and analysis"""
    path: Path
    total_files: int
    total_directories: int
    total_size: int
    file_types: Dict[str, int]
    largest_files: List[Tuple[str, int]]
    oldest_files: List[Tuple[str, datetime]]
    newest_files: List[Tuple[str, datetime]]
    permissions_summary: Dict[str, int]
    depth_analysis: Dict[int, int]


class FileSystemManager:
    """
    Advanced file system operations manager with safety features
    
    Features:
    - Atomic file operations with rollback capabilities
    - Comprehensive validation and sanitization
    - Permission management and security checks
    - Backup and recovery mechanisms
    - Performance monitoring and optimization
    - Integration with Claude Flow hooks
    - Cross-platform compatibility
    """
    
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        safety_level: SafetyLevel = SafetyLevel.STANDARD,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        enable_backups: bool = True,
        backup_retention_days: int = 30,
        claude_flow_binary: str = "npx claude-flow@alpha"
    ):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.safety_level = safety_level
        self.max_file_size = max_file_size
        self.enable_backups = enable_backups
        self.backup_retention_days = backup_retention_days
        self.claude_flow_binary = claude_flow_binary
        
        # Create backup directory
        self.backup_dir = self.base_path / '.claude-tui-backups'
        self.backup_dir.mkdir(exist_ok=True)
        
        # Operation tracking
        self.operation_history: List[FileOperationResult] = []
        self.active_locks: Set[str] = set()
        self.backup_registry: Dict[str, str] = {}
        
        # Safety features
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js',
            '.jar', '.sh', '.ps1', '.msi', '.deb', '.rpm'
        }
        self.dangerous_paths = {
            '/etc', '/bin', '/sbin', '/usr/bin', '/usr/sbin',
            'C:\\Windows', 'C:\\Program Files'
        }
        
        # Performance metrics
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'bytes_processed': 0,
            'average_execution_time': 0.0,
            'backup_operations': 0
        }
        
        # File type detection
        try:
            self.mime_detector = magic.Magic(mime=True)
        except:
            self.mime_detector = None
            logger.warning("python-magic not available, MIME type detection disabled")
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for file system operations"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    async def read_file(
        self,
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        validate: bool = True,
        max_size_override: Optional[int] = None
    ) -> FileOperationResult:
        """
        Safely read file content with validation
        
        Args:
            file_path: Path to file to read
            encoding: Text encoding (ignored for binary files)
            validate: Whether to validate file before reading
            max_size_override: Override default max file size
            
        Returns:
            FileOperationResult with file content
        """
        start_time = time.time()
        path = Path(file_path)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        logger.info(f"Reading file: {path}")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Reading file {path}")
            
            # Validation checks
            if validate:
                validation_result = await self._validate_file_access(path, 'read')
                if not validation_result['valid']:
                    raise FileValidationError(validation_result['error'])
            
            # Check file size
            max_size = max_size_override or self.max_file_size
            if path.stat().st_size > max_size:
                raise FileSystemError(f"File too large: {path.stat().st_size} bytes (max: {max_size})")
            
            # Determine file type
            file_type = await self._detect_file_type(path)
            
            # Read file content
            if file_type == FileType.BINARY:
                async with aiofiles.open(path, 'rb') as f:
                    content = await f.read()
            else:
                async with aiofiles.open(path, 'r', encoding=encoding, errors='replace') as f:
                    content = await f.read()
            
            # Calculate checksum
            checksum = hashlib.sha256(
                content.encode() if isinstance(content, str) else content
            ).hexdigest()
            
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="read",
                status=FileOperationStatus.SUCCESS,
                path=str(path),
                message=f"Successfully read {len(content)} {'characters' if isinstance(content, str) else 'bytes'}",
                files_affected=[str(path)],
                size_bytes=len(content.encode() if isinstance(content, str) else content),
                execution_time=execution_time,
                checksum_after=checksum,
                metadata={
                    'content': content,
                    'encoding': encoding if file_type != FileType.BINARY else None,
                    'file_type': file_type.value,
                    'content_length': len(content)
                }
            )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"File {path} read successfully",
                memory_key=f"filesystem/read/{path.name}",
                file_path=str(path)
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"File read completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="read",
                status=FileOperationStatus.FAILED,
                path=str(path),
                message=f"Failed to read file: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to read file {path}: {e}")
            raise FileSystemError(f"Failed to read file: {e}") from e

    async def write_file(
        self,
        file_path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = 'utf-8',
        atomic: bool = True,
        create_backup: bool = None,
        validate: bool = True,
        permissions: Optional[int] = None
    ) -> FileOperationResult:
        """
        Safely write file content with atomic operations
        
        Args:
            file_path: Path to file to write
            content: Content to write
            encoding: Text encoding (ignored for bytes)
            atomic: Whether to use atomic write operations
            create_backup: Whether to create backup (uses default if None)
            validate: Whether to validate operation
            permissions: File permissions to set
            
        Returns:
            FileOperationResult with operation status
        """
        start_time = time.time()
        path = Path(file_path)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        logger.info(f"Writing file: {path}")
        
        # Use backup setting if not specified
        if create_backup is None:
            create_backup = self.enable_backups
        
        backup_path = None
        checksum_before = None
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Writing file {path}")
            
            # Validation checks
            if validate:
                validation_result = await self._validate_file_write(path, content)
                if not validation_result['valid']:
                    raise FileValidationError(validation_result['error'])
            
            # Create backup if file exists and backup is enabled
            if create_backup and path.exists():
                backup_path = await self._create_file_backup(path)
                checksum_before = await self._calculate_file_checksum(path)
            
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write operation
            if atomic:
                # Write to temporary file first
                temp_path = path.with_suffix(path.suffix + '.tmp')
                
                try:
                    if isinstance(content, bytes):
                        async with aiofiles.open(temp_path, 'wb') as f:
                            await f.write(content)
                    else:
                        async with aiofiles.open(temp_path, 'w', encoding=encoding) as f:
                            await f.write(content)
                    
                    # Set permissions if specified
                    if permissions:
                        temp_path.chmod(permissions)
                    
                    # Atomic move
                    temp_path.replace(path)
                    
                except Exception as e:
                    # Cleanup temporary file on failure
                    if temp_path.exists():
                        temp_path.unlink()
                    raise e
                    
            else:
                # Direct write (non-atomic)
                if isinstance(content, bytes):
                    async with aiofiles.open(path, 'wb') as f:
                        await f.write(content)
                else:
                    async with aiofiles.open(path, 'w', encoding=encoding) as f:
                        await f.write(content)
                
                # Set permissions if specified
                if permissions:
                    path.chmod(permissions)
            
            # Calculate checksum after write
            checksum_after = await self._calculate_file_checksum(path)
            
            # Verify write integrity
            if isinstance(content, str):
                expected_checksum = hashlib.sha256(content.encode(encoding)).hexdigest()
            else:
                expected_checksum = hashlib.sha256(content).hexdigest()
            
            if checksum_after != expected_checksum:
                raise FileSystemError("Write integrity check failed")
            
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="write",
                status=FileOperationStatus.SUCCESS,
                path=str(path),
                message=f"Successfully wrote {len(content)} {'characters' if isinstance(content, str) else 'bytes'}",
                files_affected=[str(path)],
                size_bytes=len(content.encode(encoding) if isinstance(content, str) else content),
                execution_time=execution_time,
                backup_path=str(backup_path) if backup_path else None,
                checksum_before=checksum_before,
                checksum_after=checksum_after,
                metadata={
                    'atomic': atomic,
                    'backup_created': backup_path is not None,
                    'encoding': encoding if isinstance(content, str) else None,
                    'permissions': permissions,
                    'content_length': len(content)
                }
            )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"File {path} written successfully",
                memory_key=f"filesystem/write/{path.name}",
                file_path=str(path)
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"File write completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Restore from backup if available
            if backup_path and Path(backup_path).exists():
                try:
                    shutil.copy2(backup_path, path)
                    logger.info(f"Restored file from backup: {backup_path}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore from backup: {restore_error}")
            
            result = FileOperationResult(
                operation="write",
                status=FileOperationStatus.FAILED,
                path=str(path),
                message=f"Failed to write file: {e}",
                execution_time=execution_time,
                backup_path=str(backup_path) if backup_path else None
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to write file {path}: {e}")
            raise FileSystemError(f"Failed to write file: {e}") from e

    async def create_directory(
        self,
        directory_path: Union[str, Path],
        parents: bool = True,
        exist_ok: bool = True,
        permissions: Optional[int] = None
    ) -> FileOperationResult:
        """
        Create directory with comprehensive safety checks
        
        Args:
            directory_path: Path to directory to create
            parents: Whether to create parent directories
            exist_ok: Whether to ignore if directory exists
            permissions: Directory permissions to set
            
        Returns:
            FileOperationResult with operation status
        """
        start_time = time.time()
        path = Path(directory_path)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        logger.info(f"Creating directory: {path}")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Creating directory {path}")
            
            # Safety checks
            if self.safety_level != SafetyLevel.MINIMAL:
                safety_result = await self._validate_directory_creation(path)
                if not safety_result['valid']:
                    raise DirectoryError(safety_result['error'])
            
            # Check if directory already exists
            if path.exists():
                if path.is_dir():
                    if exist_ok:
                        result = FileOperationResult(
                            operation="create_directory",
                            status=FileOperationStatus.SKIPPED,
                            path=str(path),
                            message="Directory already exists",
                            execution_time=time.time() - start_time
                        )
                        return result
                    else:
                        raise DirectoryError(f"Directory already exists: {path}")
                else:
                    raise DirectoryError(f"Path exists but is not a directory: {path}")
            
            # Create directory
            path.mkdir(parents=parents, exist_ok=exist_ok)
            
            # Set permissions if specified
            if permissions:
                path.chmod(permissions)
            
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="create_directory",
                status=FileOperationStatus.SUCCESS,
                path=str(path),
                message="Directory created successfully",
                files_affected=[str(path)],
                execution_time=execution_time,
                metadata={
                    'parents': parents,
                    'permissions': permissions,
                    'exist_ok': exist_ok
                }
            )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"Directory {path} created",
                memory_key=f"filesystem/directory/{path.name}"
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"Directory created in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="create_directory",
                status=FileOperationStatus.FAILED,
                path=str(path),
                message=f"Failed to create directory: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to create directory {path}: {e}")
            raise DirectoryError(f"Failed to create directory: {e}") from e

    async def delete_file(
        self,
        file_path: Union[str, Path],
        create_backup: bool = None,
        force: bool = False
    ) -> FileOperationResult:
        """
        Safely delete file with backup options
        
        Args:
            file_path: Path to file to delete
            create_backup: Whether to create backup before deletion
            force: Whether to force deletion (ignore safety checks)
            
        Returns:
            FileOperationResult with operation status
        """
        start_time = time.time()
        path = Path(file_path)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        logger.info(f"Deleting file: {path}")
        
        # Use backup setting if not specified
        if create_backup is None:
            create_backup = self.enable_backups
        
        backup_path = None
        checksum_before = None
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Deleting file {path}")
            
            # Check if file exists
            if not path.exists():
                result = FileOperationResult(
                    operation="delete_file",
                    status=FileOperationStatus.NOT_FOUND,
                    path=str(path),
                    message="File does not exist",
                    execution_time=time.time() - start_time
                )
                return result
            
            if not path.is_file():
                raise FileSystemError(f"Path is not a file: {path}")
            
            # Safety checks
            if not force and self.safety_level != SafetyLevel.MINIMAL:
                safety_result = await self._validate_file_deletion(path)
                if not safety_result['valid']:
                    raise FilePermissionError(safety_result['error'])
            
            # Get file info before deletion
            file_size = path.stat().st_size
            checksum_before = await self._calculate_file_checksum(path)
            
            # Create backup if enabled
            if create_backup:
                backup_path = await self._create_file_backup(path)
            
            # Delete file
            path.unlink()
            
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="delete_file",
                status=FileOperationStatus.SUCCESS,
                path=str(path),
                message="File deleted successfully",
                files_affected=[str(path)],
                size_bytes=file_size,
                execution_time=execution_time,
                backup_path=str(backup_path) if backup_path else None,
                checksum_before=checksum_before,
                metadata={
                    'backup_created': backup_path is not None,
                    'force': force,
                    'original_size': file_size
                }
            )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"File {path} deleted",
                memory_key=f"filesystem/delete/{path.name}"
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"File deleted in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="delete_file",
                status=FileOperationStatus.FAILED,
                path=str(path),
                message=f"Failed to delete file: {e}",
                execution_time=execution_time,
                backup_path=str(backup_path) if backup_path else None,
                checksum_before=checksum_before
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to delete file {path}: {e}")
            raise FileSystemError(f"Failed to delete file: {e}") from e

    async def copy_file(
        self,
        source_path: Union[str, Path],
        destination_path: Union[str, Path],
        preserve_metadata: bool = True,
        overwrite: bool = False,
        validate_integrity: bool = True
    ) -> FileOperationResult:
        """
        Copy file with integrity verification
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            preserve_metadata: Whether to preserve file metadata
            overwrite: Whether to overwrite existing files
            validate_integrity: Whether to validate copy integrity
            
        Returns:
            FileOperationResult with operation status
        """
        start_time = time.time()
        src_path = Path(source_path)
        dst_path = Path(destination_path)
        
        if not src_path.is_absolute():
            src_path = self.base_path / src_path
        if not dst_path.is_absolute():
            dst_path = self.base_path / dst_path
        
        logger.info(f"Copying file: {src_path} -> {dst_path}")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Copying file {src_path} to {dst_path}")
            
            # Validation checks
            if not src_path.exists():
                raise FileSystemError(f"Source file does not exist: {src_path}")
            
            if not src_path.is_file():
                raise FileSystemError(f"Source is not a file: {src_path}")
            
            if dst_path.exists() and not overwrite:
                raise FileSystemError(f"Destination exists and overwrite is False: {dst_path}")
            
            # Safety checks
            if self.safety_level != SafetyLevel.MINIMAL:
                safety_result = await self._validate_file_copy(src_path, dst_path)
                if not safety_result['valid']:
                    raise FilePermissionError(safety_result['error'])
            
            # Get source file info
            src_size = src_path.stat().st_size
            src_checksum = await self._calculate_file_checksum(src_path)
            
            # Create destination directory
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Perform copy
            if preserve_metadata:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
            
            # Validate integrity if enabled
            if validate_integrity:
                dst_checksum = await self._calculate_file_checksum(dst_path)
                if src_checksum != dst_checksum:
                    # Cleanup failed copy
                    if dst_path.exists():
                        dst_path.unlink()
                    raise FileSystemError("Copy integrity validation failed")
            
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="copy_file",
                status=FileOperationStatus.SUCCESS,
                path=str(src_path),
                message=f"File copied successfully to {dst_path}",
                files_affected=[str(src_path), str(dst_path)],
                size_bytes=src_size,
                execution_time=execution_time,
                checksum_before=src_checksum,
                checksum_after=dst_checksum if validate_integrity else None,
                metadata={
                    'source': str(src_path),
                    'destination': str(dst_path),
                    'preserve_metadata': preserve_metadata,
                    'overwrite': overwrite,
                    'integrity_validated': validate_integrity
                }
            )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"File copied: {src_path.name} to {dst_path.name}",
                memory_key=f"filesystem/copy/{dst_path.name}",
                file_path=str(dst_path)
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"File copy completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = FileOperationResult(
                operation="copy_file",
                status=FileOperationStatus.FAILED,
                path=str(src_path),
                message=f"Failed to copy file: {e}",
                execution_time=execution_time,
                metadata={
                    'source': str(src_path),
                    'destination': str(dst_path)
                }
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to copy file {src_path} to {dst_path}: {e}")
            raise FileSystemError(f"Failed to copy file: {e}") from e

    async def get_file_info(
        self,
        file_path: Union[str, Path],
        include_content_analysis: bool = False
    ) -> FileInfo:
        """
        Get comprehensive file information
        
        Args:
            file_path: Path to file to analyze
            include_content_analysis: Whether to include content analysis
            
        Returns:
            FileInfo with comprehensive file details
        """
        path = Path(file_path)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        if not path.exists():
            raise FileSystemError(f"File does not exist: {path}")
        
        try:
            stat_info = path.stat()
            
            # Basic file information
            file_info = FileInfo(
                path=path,
                name=path.name,
                size=stat_info.st_size,
                file_type=await self._detect_file_type(path),
                mime_type=await self._detect_mime_type(path),
                permissions=stat.filemode(stat_info.st_mode),
                owner=path.owner() if hasattr(path, 'owner') else 'unknown',
                group=path.group() if hasattr(path, 'group') else 'unknown',
                created=datetime.fromtimestamp(stat_info.st_ctime),
                modified=datetime.fromtimestamp(stat_info.st_mtime),
                accessed=datetime.fromtimestamp(stat_info.st_atime),
                checksum=await self._calculate_file_checksum(path),
                is_symlink=path.is_symlink(),
                is_hidden=path.name.startswith('.'),
                extension=path.suffix.lower(),
                encoding=await self._detect_encoding(path) if await self._detect_file_type(path) == FileType.TEXT else None
            )
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {path}: {e}")
            raise FileSystemError(f"Failed to get file info: {e}") from e

    async def analyze_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        max_depth: Optional[int] = None
    ) -> DirectoryStats:
        """
        Analyze directory structure and statistics
        
        Args:
            directory_path: Directory to analyze
            recursive: Whether to analyze recursively
            max_depth: Maximum recursion depth
            
        Returns:
            DirectoryStats with comprehensive analysis
        """
        path = Path(directory_path)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        if not path.exists() or not path.is_dir():
            raise DirectoryError(f"Directory does not exist: {path}")
        
        logger.info(f"Analyzing directory: {path}")
        
        try:
            total_files = 0
            total_directories = 0
            total_size = 0
            file_types = {}
            largest_files = []
            oldest_files = []
            newest_files = []
            permissions_summary = {}
            depth_analysis = {}
            
            # Walk directory tree
            if recursive:
                for root, dirs, files in os.walk(path):
                    current_depth = len(Path(root).relative_to(path).parts)
                    
                    # Respect max depth
                    if max_depth and current_depth > max_depth:
                        dirs.clear()  # Don't recurse deeper
                        continue
                    
                    depth_analysis[current_depth] = depth_analysis.get(current_depth, 0) + len(files)
                    total_directories += len(dirs)
                    
                    for file_name in files:
                        file_path = Path(root) / file_name
                        
                        try:
                            stat_info = file_path.stat()
                            file_size = stat_info.st_size
                            
                            total_files += 1
                            total_size += file_size
                            
                            # File type analysis
                            extension = file_path.suffix.lower() or 'no_extension'
                            file_types[extension] = file_types.get(extension, 0) + 1
                            
                            # Track largest files (top 10)
                            largest_files.append((str(file_path), file_size))
                            largest_files.sort(key=lambda x: x[1], reverse=True)
                            largest_files = largest_files[:10]
                            
                            # Track oldest and newest files (top 10 each)
                            mod_time = datetime.fromtimestamp(stat_info.st_mtime)
                            
                            oldest_files.append((str(file_path), mod_time))
                            oldest_files.sort(key=lambda x: x[1])
                            oldest_files = oldest_files[:10]
                            
                            newest_files.append((str(file_path), mod_time))
                            newest_files.sort(key=lambda x: x[1], reverse=True)
                            newest_files = newest_files[:10]
                            
                            # Permission analysis
                            perm_str = stat.filemode(stat_info.st_mode)
                            permissions_summary[perm_str] = permissions_summary.get(perm_str, 0) + 1
                            
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Could not analyze file {file_path}: {e}")
                            continue
            else:
                # Non-recursive analysis
                for item in path.iterdir():
                    if item.is_file():
                        total_files += 1
                        try:
                            stat_info = item.stat()
                            total_size += stat_info.st_size
                            
                            extension = item.suffix.lower() or 'no_extension'
                            file_types[extension] = file_types.get(extension, 0) + 1
                            
                        except (OSError, PermissionError):
                            continue
                    elif item.is_dir():
                        total_directories += 1
            
            stats = DirectoryStats(
                path=path,
                total_files=total_files,
                total_directories=total_directories,
                total_size=total_size,
                file_types=file_types,
                largest_files=largest_files,
                oldest_files=oldest_files,
                newest_files=newest_files,
                permissions_summary=permissions_summary,
                depth_analysis=depth_analysis
            )
            
            logger.info(f"Directory analysis completed: {total_files} files, {total_size} bytes")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to analyze directory {path}: {e}")
            raise DirectoryError(f"Failed to analyze directory: {e}") from e

    async def _validate_file_access(self, path: Path, operation: str) -> Dict[str, Any]:
        """Validate file access permissions and safety"""
        
        try:
            # Check if path is in dangerous locations
            if self.safety_level == SafetyLevel.PARANOID:
                for dangerous_path in self.dangerous_paths:
                    if str(path).startswith(dangerous_path):
                        return {
                            'valid': False,
                            'error': f'Operation not allowed in dangerous path: {dangerous_path}'
                        }
            
            # Check file extension for dangerous types
            if path.suffix.lower() in self.dangerous_extensions:
                if self.safety_level == SafetyLevel.PARANOID:
                    return {
                        'valid': False,
                        'error': f'Dangerous file extension: {path.suffix}'
                    }
                elif self.safety_level == SafetyLevel.STANDARD:
                    logger.warning(f"Potentially dangerous file: {path}")
            
            # Check permissions
            if operation == 'read':
                if not os.access(path, os.R_OK):
                    return {
                        'valid': False,
                        'error': 'Read permission denied'
                    }
            elif operation == 'write':
                if path.exists() and not os.access(path, os.W_OK):
                    return {
                        'valid': False,
                        'error': 'Write permission denied'
                    }
                elif not path.exists() and not os.access(path.parent, os.W_OK):
                    return {
                        'valid': False,
                        'error': 'Write permission denied to parent directory'
                    }
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {e}'
            }

    async def _validate_file_write(self, path: Path, content: Union[str, bytes]) -> Dict[str, Any]:
        """Validate file write operation"""
        
        # Check content size
        content_size = len(content.encode('utf-8') if isinstance(content, str) else content)
        if content_size > self.max_file_size:
            return {
                'valid': False,
                'error': f'Content too large: {content_size} bytes (max: {self.max_file_size})'
            }
        
        # Check available disk space
        try:
            disk_usage = psutil.disk_usage(path.parent)
            if disk_usage.free < content_size * 2:  # Require 2x content size free
                return {
                    'valid': False,
                    'error': f'Insufficient disk space: {disk_usage.free} bytes available'
                }
        except:
            pass  # Skip disk space check if unavailable
        
        # Check for malicious content patterns (basic)
        if isinstance(content, str) and self.safety_level != SafetyLevel.MINIMAL:
            malicious_patterns = [
                r'rm\s+-rf\s+/',  # Dangerous rm commands
                r'del\s+/[sq]',   # Dangerous Windows del commands
                r'format\s+c:',   # Format commands
                r'__import__\s*\(\s*[\'"]os[\'"]',  # Python os imports
                r'eval\s*\(',     # Eval functions
                r'exec\s*\(',     # Exec functions
            ]
            
            import re
            for pattern in malicious_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    if self.safety_level == SafetyLevel.PARANOID:
                        return {
                            'valid': False,
                            'error': f'Potentially malicious content detected: {pattern}'
                        }
                    else:
                        logger.warning(f"Potentially suspicious content in file: {path}")
        
        return {'valid': True}

    async def _validate_directory_creation(self, path: Path) -> Dict[str, Any]:
        """Validate directory creation operation"""
        
        # Check if path is in dangerous locations
        if self.safety_level == SafetyLevel.PARANOID:
            for dangerous_path in self.dangerous_paths:
                if str(path).startswith(dangerous_path):
                    return {
                        'valid': False,
                        'error': f'Directory creation not allowed in: {dangerous_path}'
                    }
        
        # Check parent directory permissions
        if path.parent.exists() and not os.access(path.parent, os.W_OK):
            return {
                'valid': False,
                'error': 'Write permission denied to parent directory'
            }
        
        return {'valid': True}

    async def _validate_file_deletion(self, path: Path) -> Dict[str, Any]:
        """Validate file deletion operation"""
        
        # Check if file is critical system file
        critical_files = {
            'passwd', 'shadow', 'hosts', 'fstab', 'sudoers',
            'boot.ini', 'ntldr', 'bootmgr'
        }
        
        if path.name.lower() in critical_files:
            return {
                'valid': False,
                'error': f'Cannot delete critical system file: {path.name}'
            }
        
        # Check delete permissions
        if not os.access(path, os.W_OK):
            return {
                'valid': False,
                'error': 'Delete permission denied'
            }
        
        return {'valid': True}

    async def _validate_file_copy(self, src_path: Path, dst_path: Path) -> Dict[str, Any]:
        """Validate file copy operation"""
        
        # Check if copying to dangerous location
        if self.safety_level == SafetyLevel.PARANOID:
            for dangerous_path in self.dangerous_paths:
                if str(dst_path).startswith(dangerous_path):
                    return {
                        'valid': False,
                        'error': f'Copy to dangerous location not allowed: {dangerous_path}'
                    }
        
        return {'valid': True}

    async def _detect_file_type(self, path: Path) -> FileType:
        """Detect file type based on content and extension"""
        
        try:
            # Check by extension first
            extension = path.suffix.lower()
            
            text_extensions = {
                '.txt', '.md', '.rst', '.py', '.js', '.html', '.css', '.json',
                '.xml', '.yml', '.yaml', '.ini', '.cfg', '.conf', '.log'
            }
            
            image_extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.ico'
            }
            
            archive_extensions = {
                '.zip', '.rar', '.tar', '.gz', '.bz2', '.xz', '.7z'
            }
            
            executable_extensions = {
                '.exe', '.bat', '.cmd', '.sh', '.ps1'
            }
            
            config_extensions = {
                '.ini', '.cfg', '.conf', '.config', '.toml'
            }
            
            source_extensions = {
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php'
            }
            
            if extension in text_extensions:
                return FileType.TEXT
            elif extension in image_extensions:
                return FileType.IMAGE
            elif extension in archive_extensions:
                return FileType.ARCHIVE
            elif extension in executable_extensions:
                return FileType.EXECUTABLE
            elif extension in config_extensions:
                return FileType.CONFIGURATION
            elif extension in source_extensions:
                return FileType.SOURCE_CODE
            
            # Check MIME type if available
            if self.mime_detector:
                try:
                    mime_type = self.mime_detector.from_file(str(path))
                    if mime_type.startswith('text/'):
                        return FileType.TEXT
                    elif mime_type.startswith('image/'):
                        return FileType.IMAGE
                    elif 'application' in mime_type and any(x in mime_type for x in ['zip', 'tar', 'gzip']):
                        return FileType.ARCHIVE
                    elif 'application/octet-stream' in mime_type:
                        return FileType.BINARY
                except:
                    pass
            
            # Try to detect by content
            try:
                with open(path, 'rb') as f:
                    header = f.read(512)
                    
                    # Check for text content
                    try:
                        header.decode('utf-8')
                        return FileType.TEXT
                    except UnicodeDecodeError:
                        pass
                    
                    # Check for common binary signatures
                    if header.startswith(b'\x50\x4B'):  # ZIP signature
                        return FileType.ARCHIVE
                    elif header.startswith(b'\xFF\xD8\xFF'):  # JPEG signature
                        return FileType.IMAGE
                    elif header.startswith(b'\x89PNG'):  # PNG signature
                        return FileType.IMAGE
                    
            except:
                pass
            
            return FileType.BINARY
            
        except Exception:
            return FileType.UNKNOWN

    async def _detect_mime_type(self, path: Path) -> str:
        """Detect MIME type of file"""
        
        if self.mime_detector:
            try:
                return self.mime_detector.from_file(str(path))
            except:
                pass
        
        # Fallback to extension-based detection
        extension = path.suffix.lower()
        mime_map = {
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.zip': 'application/zip',
            '.tar': 'application/x-tar'
        }
        
        return mime_map.get(extension, 'application/octet-stream')

    async def _detect_encoding(self, path: Path) -> Optional[str]:
        """Detect text file encoding"""
        
        try:
            import chardet
            
            with open(path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
                
        except ImportError:
            # Fallback to common encodings
            for encoding in ['utf-8', 'ascii', 'latin-1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        f.read(1024)
                    return encoding
                except UnicodeDecodeError:
                    continue
        except:
            pass
        
        return 'utf-8'  # Default fallback

    async def _calculate_file_checksum(self, path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        
        try:
            hash_func = hashlib.new(algorithm)
            
            async with aiofiles.open(path, 'rb') as f:
                while chunk := await f.read(8192):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {path}: {e}")
            return ""

    async def _create_file_backup(self, path: Path) -> Path:
        """Create backup of file before modification"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{path.name}.backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(path, backup_path)
            
            # Store in backup registry
            self.backup_registry[str(path)] = str(backup_path)
            
            # Clean old backups
            await self._cleanup_old_backups()
            
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup for {path}: {e}")
            raise BackupError(f"Backup creation failed: {e}") from e

    async def _cleanup_old_backups(self):
        """Clean up old backup files"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_retention_days)
            
            for backup_file in self.backup_dir.glob('*.backup_*'):
                try:
                    file_modified = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_modified < cutoff_date:
                        backup_file.unlink()
                        logger.debug(f"Removed old backup: {backup_file}")
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    async def _execute_hook(
        self,
        hook_type: str,
        message: str,
        memory_key: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        """Execute Claude Flow hooks for coordination"""
        
        try:
            if hook_type == "pre-task":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "pre-task",
                    "--description", message
                ]
            elif hook_type == "post-task":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "post-task",
                    "--task-id", "filesystem-operation"
                ]
            elif hook_type == "post-edit" and memory_key:
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "post-edit",
                    "--memory-key", memory_key
                ]
                if file_path:
                    cmd.extend(["--file", file_path])
            elif hook_type == "notify":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "notify",
                    "--message", message
                ]
            else:
                return
            
            # Execute hook command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await asyncio.wait_for(process.communicate(), timeout=10)
            
        except Exception as e:
            logger.debug(f"Hook execution failed ({hook_type}): {e}")

    def _update_metrics(self, result: FileOperationResult):
        """Update performance metrics"""
        
        self.metrics['total_operations'] += 1
        self.metrics['bytes_processed'] += result.size_bytes
        
        if result.is_success:
            self.metrics['successful_operations'] += 1
        else:
            self.metrics['failed_operations'] += 1
        
        if result.backup_path:
            self.metrics['backup_operations'] += 1
        
        # Update average execution time
        total_time = (
            self.metrics['average_execution_time'] * (self.metrics['total_operations'] - 1) +
            result.execution_time
        )
        self.metrics['average_execution_time'] = total_time / self.metrics['total_operations']

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        success_rate = 0.0
        if self.metrics['total_operations'] > 0:
            success_rate = self.metrics['successful_operations'] / self.metrics['total_operations']
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'operations_history_size': len(self.operation_history),
            'active_locks': len(self.active_locks),
            'backup_registry_size': len(self.backup_registry),
            'bytes_processed_human': self._human_readable_size(self.metrics['bytes_processed'])
        }

    def _human_readable_size(self, size: int) -> str:
        """Convert bytes to human readable format"""
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    async def health_check(self) -> Dict[str, Any]:
        """Check health of file system integration"""
        
        try:
            # Check base path accessibility
            base_accessible = self.base_path.exists() and os.access(self.base_path, os.R_OK | os.W_OK)
            
            # Check backup directory
            backup_accessible = self.backup_dir.exists() and os.access(self.backup_dir, os.R_OK | os.W_OK)
            
            # Check disk space
            try:
                disk_usage = psutil.disk_usage(self.base_path)
                disk_usage_info = {
                    'total': disk_usage.total,
                    'used': disk_usage.used,
                    'free': disk_usage.free,
                    'percent': (disk_usage.used / disk_usage.total) * 100
                }
            except:
                disk_usage_info = None
            
            # Count backup files
            backup_count = len(list(self.backup_dir.glob('*.backup_*'))) if backup_accessible else 0
            
            return {
                'status': 'healthy' if base_accessible and backup_accessible else 'degraded',
                'base_path': str(self.base_path),
                'base_path_accessible': base_accessible,
                'backup_dir': str(self.backup_dir),
                'backup_dir_accessible': backup_accessible,
                'backup_count': backup_count,
                'safety_level': self.safety_level.value,
                'max_file_size': self.max_file_size,
                'disk_usage': disk_usage_info,
                'metrics': self.get_metrics(),
                'mime_detector_available': self.mime_detector is not None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'base_path': str(self.base_path)
            }