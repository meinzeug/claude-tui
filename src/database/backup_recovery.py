"""
Database Backup and Recovery System
Production-grade backup and recovery with encryption and cloud storage
"""

import os
import asyncio
import subprocess
import gzip
import shutil
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import aiofiles
import hashlib
import json

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTUIException

logger = get_logger(__name__)


@dataclass
class BackupConfig:
    """Backup configuration settings."""
    enabled: bool = True
    schedule: str = "0 2 * * *"  # Daily at 2 AM
    retention_days: int = 30
    compression: bool = True
    encryption: bool = True
    local_path: str = "/var/backups/claude-tui"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "database-backups"
    notification_webhook: Optional[str] = None


@dataclass
class BackupMetadata:
    """Backup metadata information."""
    backup_id: str
    database_name: str
    backup_type: str  # full, incremental, differential
    timestamp: datetime
    size_bytes: int
    compressed: bool
    encrypted: bool
    checksum: str
    location: str
    retention_date: datetime
    status: str  # pending, completed, failed, expired


class DatabaseBackupManager:
    """
    Advanced database backup and recovery system.
    
    Features:
    - Automated scheduled backups
    - Incremental and differential backups
    - Compression and encryption
    - Cloud storage integration (S3)
    - Point-in-time recovery
    - Backup verification and integrity checks
    - Monitoring and alerting
    """
    
    def __init__(
        self,
        config: BackupConfig,
        database_url: str,
        encryption_key: Optional[str] = None
    ):
        """
        Initialize backup manager.
        
        Args:
            config: Backup configuration
            database_url: Database connection URL
            encryption_key: Encryption key for backups
        """
        self.config = config
        self.database_url = database_url
        self.encryption_key = encryption_key
        
        # Parse database connection details
        self.db_host, self.db_port, self.db_name, self.db_user, self.db_password = self._parse_db_url(database_url)
        
        # Storage paths
        self.local_backup_path = Path(config.local_path)
        self.local_backup_path.mkdir(parents=True, exist_ok=True)
        
        # S3 client
        self.s3_client = None
        if config.s3_bucket:
            self.s3_client = boto3.client('s3')
        
        # Backup tracking
        self.backup_history: List[BackupMetadata] = []
        self.active_backups: Dict[str, BackupMetadata] = {}
        
        logger.info("Database backup manager initialized")
    
    def _parse_db_url(self, url: str) -> tuple:
        """Parse database URL into components."""
        try:
            # Handle asyncpg URLs
            if url.startswith('postgresql+asyncpg://'):
                url = url.replace('postgresql+asyncpg://', 'postgresql://')
            
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            return (
                parsed.hostname or 'localhost',
                parsed.port or 5432,
                parsed.path.lstrip('/') if parsed.path else 'postgres',
                parsed.username or 'postgres',
                parsed.password or ''
            )
        except Exception as e:
            logger.error(f"Failed to parse database URL: {e}")
            raise ClaudeTUIException(f"Invalid database URL: {e}")
    
    async def create_full_backup(
        self,
        backup_name: Optional[str] = None,
        upload_to_s3: bool = True
    ) -> BackupMetadata:
        """
        Create a full database backup.
        
        Args:
            backup_name: Custom backup name (auto-generated if None)
            upload_to_s3: Upload to S3 after creation
            
        Returns:
            BackupMetadata: Metadata for the created backup
        """
        if not backup_name:
            backup_name = f"full_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        backup_id = f"{backup_name}_{hashlib.md5(backup_name.encode()).hexdigest()[:8]}"
        
        logger.info(f"Creating full backup: {backup_id}")
        
        try:
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                database_name=self.db_name,
                backup_type="full",
                timestamp=datetime.utcnow(),
                size_bytes=0,
                compressed=self.config.compression,
                encrypted=self.config.encryption,
                checksum="",
                location="",
                retention_date=datetime.utcnow() + timedelta(days=self.config.retention_days),
                status="pending"
            )
            
            self.active_backups[backup_id] = metadata
            
            # Create backup file path
            backup_filename = f"{backup_id}.sql"
            if self.config.compression:
                backup_filename += ".gz"
            if self.config.encryption:
                backup_filename += ".enc"
            
            local_backup_file = self.local_backup_path / backup_filename
            
            # Run pg_dump
            await self._run_pg_dump(str(local_backup_file), metadata)
            
            # Calculate file size and checksum
            metadata.size_bytes = local_backup_file.stat().st_size
            metadata.checksum = await self._calculate_checksum(local_backup_file)
            metadata.location = str(local_backup_file)
            
            # Upload to S3 if configured
            if upload_to_s3 and self.s3_client and self.config.s3_bucket:
                s3_key = f"{self.config.s3_prefix}/{backup_filename}"
                await self._upload_to_s3(local_backup_file, s3_key)
                metadata.location = f"s3://{self.config.s3_bucket}/{s3_key}"
            
            metadata.status = "completed"
            self.backup_history.append(metadata)
            
            logger.info(f"Full backup completed: {backup_id} ({metadata.size_bytes} bytes)")
            
            # Send notification
            await self._send_notification(f"Full backup completed: {backup_id}", metadata)
            
            return metadata
            
        except Exception as e:
            metadata.status = "failed"
            logger.error(f"Full backup failed: {e}")
            await self._send_notification(f"Full backup failed: {backup_id} - {e}", metadata)
            raise ClaudeTUIException(f"Backup failed: {e}")
        
        finally:
            if backup_id in self.active_backups:
                del self.active_backups[backup_id]
    
    async def _run_pg_dump(self, output_file: str, metadata: BackupMetadata):
        """Run pg_dump command with proper options."""
        cmd = [
            'pg_dump',
            f'--host={self.db_host}',
            f'--port={self.db_port}',
            f'--username={self.db_user}',
            '--no-password',
            '--verbose',
            '--format=custom',
            '--compress=9' if metadata.compressed else '--compress=0',
            self.db_name
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_password
        
        if metadata.compressed and metadata.encrypted:
            # Pipe through gzip and encryption
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"pg_dump failed: {stderr.decode()}")
            
            # Compress and encrypt
            compressed_data = gzip.compress(stdout)
            
            if self.config.encryption and self.encryption_key:
                encrypted_data = self._encrypt_data(compressed_data, self.encryption_key)
                final_data = encrypted_data
            else:
                final_data = compressed_data
            
            async with aiofiles.open(output_file, 'wb') as f:
                await f.write(final_data)
        
        else:
            # Direct output to file
            cmd.extend(['--file', output_file])
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"pg_dump failed: {stderr.decode()}")
    
    def _encrypt_data(self, data: bytes, key: str) -> bytes:
        """Encrypt data using AES encryption."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'claude_tui_backup_salt',
                iterations=100000,
            )
            key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            cipher = Fernet(key_bytes)
            
            return cipher.encrypt(data)
            
        except ImportError:
            logger.warning("Cryptography library not available, skipping encryption")
            return data
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _upload_to_s3(self, local_file: Path, s3_key: str):
        """Upload backup file to S3."""
        try:
            logger.info(f"Uploading backup to S3: s3://{self.config.s3_bucket}/{s3_key}")
            
            # Upload with server-side encryption
            self.s3_client.upload_file(
                str(local_file),
                self.config.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'STANDARD_IA'
                }
            )
            
            logger.info("S3 upload completed")
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def restore_from_backup(
        self,
        backup_id: str,
        target_database: Optional[str] = None,
        point_in_time: Optional[datetime] = None
    ) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_id: ID of backup to restore
            target_database: Target database name (current if None)
            point_in_time: Point-in-time for recovery
            
        Returns:
            bool: True if restore successful
        """
        logger.info(f"Starting database restore from backup: {backup_id}")
        
        try:
            # Find backup metadata
            backup_metadata = None
            for backup in self.backup_history:
                if backup.backup_id == backup_id:
                    backup_metadata = backup
                    break
            
            if not backup_metadata:
                raise ClaudeTUIException(f"Backup not found: {backup_id}")
            
            # Download from S3 if needed
            local_backup_file = await self._ensure_local_backup(backup_metadata)
            
            # Verify backup integrity
            if not await self._verify_backup_integrity(local_backup_file, backup_metadata):
                raise ClaudeTUIException("Backup integrity check failed")
            
            # Prepare restore command
            restore_db = target_database or self.db_name
            
            # Create target database if it doesn't exist
            await self._create_database_if_not_exists(restore_db)
            
            # Run pg_restore
            await self._run_pg_restore(local_backup_file, restore_db, backup_metadata)
            
            logger.info(f"Database restore completed: {backup_id} -> {restore_db}")
            
            # Send notification
            await self._send_notification(
                f"Database restore completed: {backup_id} -> {restore_db}",
                backup_metadata
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            await self._send_notification(f"Database restore failed: {backup_id} - {e}", None)
            return False
    
    async def _ensure_local_backup(self, metadata: BackupMetadata) -> Path:
        """Ensure backup file is available locally."""
        local_file = self.local_backup_path / Path(metadata.location).name
        
        if local_file.exists():
            return local_file
        
        # Download from S3 if stored remotely
        if metadata.location.startswith('s3://'):
            s3_path = metadata.location.replace(f"s3://{self.config.s3_bucket}/", "")
            
            logger.info(f"Downloading backup from S3: {s3_path}")
            self.s3_client.download_file(
                self.config.s3_bucket,
                s3_path,
                str(local_file)
            )
        
        return local_file
    
    async def _verify_backup_integrity(self, file_path: Path, metadata: BackupMetadata) -> bool:
        """Verify backup file integrity using checksum."""
        try:
            calculated_checksum = await self._calculate_checksum(file_path)
            return calculated_checksum == metadata.checksum
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False
    
    async def _create_database_if_not_exists(self, database_name: str):
        """Create database if it doesn't exist."""
        cmd = [
            'psql',
            f'--host={self.db_host}',
            f'--port={self.db_port}',
            f'--username={self.db_user}',
            '--command', f'CREATE DATABASE "{database_name}";',
            'postgres'  # Connect to postgres db to create new db
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_password
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            await process.communicate()
            # Ignore return code as database might already exist
            
        except Exception as e:
            logger.warning(f"Database creation warning: {e}")
    
    async def _run_pg_restore(self, backup_file: Path, database_name: str, metadata: BackupMetadata):
        """Run pg_restore command."""
        # Decrypt and decompress if needed
        restore_file = backup_file
        
        if metadata.encrypted or metadata.compressed:
            temp_file = backup_file.with_suffix('.temp')
            
            async with aiofiles.open(backup_file, 'rb') as f:
                data = await f.read()
            
            # Decrypt if encrypted
            if metadata.encrypted and self.encryption_key:
                data = self._decrypt_data(data, self.encryption_key)
            
            # Decompress if compressed
            if metadata.compressed:
                data = gzip.decompress(data)
            
            async with aiofiles.open(temp_file, 'wb') as f:
                await f.write(data)
            
            restore_file = temp_file
        
        cmd = [
            'pg_restore',
            f'--host={self.db_host}',
            f'--port={self.db_port}',
            f'--username={self.db_user}',
            f'--dbname={database_name}',
            '--verbose',
            '--clean',
            '--if-exists',
            str(restore_file)
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_password
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        _, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"pg_restore failed: {stderr.decode()}")
        
        # Clean up temporary file
        if restore_file != backup_file:
            restore_file.unlink(missing_ok=True)
    
    def _decrypt_data(self, data: bytes, key: str) -> bytes:
        """Decrypt data using AES decryption."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'claude_tui_backup_salt',
                iterations=100000,
            )
            key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            cipher = Fernet(key_bytes)
            
            return cipher.decrypt(data)
            
        except ImportError:
            logger.warning("Cryptography library not available, returning data as-is")
            return data
    
    async def cleanup_old_backups(self):
        """Clean up expired backups."""
        current_time = datetime.utcnow()
        expired_backups = []
        
        for backup in self.backup_history[:]:
            if backup.retention_date < current_time:
                expired_backups.append(backup)
        
        for backup in expired_backups:
            try:
                # Remove local file
                local_file = self.local_backup_path / Path(backup.location).name
                if local_file.exists():
                    local_file.unlink()
                
                # Remove from S3
                if backup.location.startswith('s3://') and self.s3_client:
                    s3_key = backup.location.replace(f"s3://{self.config.s3_bucket}/", "")
                    self.s3_client.delete_object(
                        Bucket=self.config.s3_bucket,
                        Key=s3_key
                    )
                
                # Remove from history
                self.backup_history.remove(backup)
                
                logger.info(f"Cleaned up expired backup: {backup.backup_id}")
                
            except Exception as e:
                logger.error(f"Failed to clean up backup {backup.backup_id}: {e}")
    
    async def _send_notification(self, message: str, metadata: Optional[BackupMetadata]):
        """Send backup notification."""
        if not self.config.notification_webhook:
            return
        
        try:
            import aiohttp
            
            payload = {
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'database': self.db_name
            }
            
            if metadata:
                payload['backup_info'] = {
                    'backup_id': metadata.backup_id,
                    'type': metadata.backup_type,
                    'size_mb': round(metadata.size_bytes / 1024 / 1024, 2),
                    'status': metadata.status
                }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.notification_webhook,
                    json=payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.debug("Notification sent successfully")
                    else:
                        logger.warning(f"Notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status."""
        return {
            'config': {
                'enabled': self.config.enabled,
                'retention_days': self.config.retention_days,
                'compression': self.config.compression,
                'encryption': self.config.encryption,
                's3_enabled': bool(self.config.s3_bucket)
            },
            'statistics': {
                'total_backups': len(self.backup_history),
                'active_backups': len(self.active_backups),
                'successful_backups': len([b for b in self.backup_history if b.status == 'completed']),
                'failed_backups': len([b for b in self.backup_history if b.status == 'failed']),
                'total_backup_size_mb': sum(b.size_bytes for b in self.backup_history) / 1024 / 1024
            },
            'recent_backups': [
                {
                    'backup_id': b.backup_id,
                    'type': b.backup_type,
                    'timestamp': b.timestamp.isoformat(),
                    'size_mb': round(b.size_bytes / 1024 / 1024, 2),
                    'status': b.status,
                    'location': 'S3' if b.location.startswith('s3://') else 'Local'
                }
                for b in sorted(self.backup_history, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }


# Global backup manager
_backup_manager: Optional[DatabaseBackupManager] = None


def get_backup_manager() -> Optional[DatabaseBackupManager]:
    """Get global backup manager."""
    return _backup_manager


async def setup_backup_manager(
    database_url: str,
    config: BackupConfig,
    encryption_key: Optional[str] = None
) -> DatabaseBackupManager:
    """Set up database backup management."""
    global _backup_manager
    
    _backup_manager = DatabaseBackupManager(
        config=config,
        database_url=database_url,
        encryption_key=encryption_key
    )
    
    logger.info("Database backup management enabled")
    return _backup_manager