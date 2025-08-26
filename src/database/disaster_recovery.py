"""
Disaster Recovery and Backup System
Comprehensive backup, restore, and disaster recovery procedures
"""

import asyncio
import shutil
import tarfile
import gzip
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os
import subprocess
import hashlib

from src.database.session import get_db
from src.core.config import get_settings
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class BackupType(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    CORRUPTED = "corrupted"


@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    file_path: str
    size_bytes: int
    checksum: str
    status: BackupStatus
    duration_seconds: float
    tables_backed_up: List[str]
    error_message: Optional[str] = None


@dataclass
class RestoreResult:
    """Result of a restore operation"""
    restore_id: str
    backup_id: str
    timestamp: datetime
    status: BackupStatus
    duration_seconds: float
    tables_restored: List[str]
    records_restored: int
    error_message: Optional[str] = None


class DisasterRecoveryManager:
    """Comprehensive disaster recovery and backup management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different backup types
        (self.backup_dir / "database").mkdir(exist_ok=True)
        (self.backup_dir / "config").mkdir(exist_ok=True)
        (self.backup_dir / "logs").mkdir(exist_ok=True)
        (self.backup_dir / "media").mkdir(exist_ok=True)
        
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        
    def generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}_{os.getpid()}"
    
    def generate_restore_id(self) -> str:
        """Generate unique restore ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"restore_{timestamp}_{os.getpid()}"
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def backup_database(self, backup_type: BackupType = BackupType.FULL) -> BackupMetadata:
        """Create database backup"""
        backup_id = self.generate_backup_id()
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting database backup: {backup_id} (type: {backup_type})")
        
        try:
            # Get database session
            db = next(get_db())
            
            # Determine tables to backup
            tables_to_backup = await self.get_database_tables(db)
            
            # Create backup directory
            backup_path = self.backup_dir / "database" / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # Backup each table
            backed_up_tables = []
            total_records = 0
            
            for table_name in tables_to_backup:
                try:
                    table_file = backup_path / f"{table_name}.sql"
                    records = await self.backup_table(db, table_name, table_file)
                    backed_up_tables.append(table_name)
                    total_records += records
                    logger.debug(f"Backed up table {table_name}: {records} records")
                except Exception as e:
                    logger.error(f"Failed to backup table {table_name}: {e}")
                    continue
            
            # Create backup archive
            archive_path = backup_path.parent / f"{backup_id}.tar.gz"
            await self.create_compressed_archive(backup_path, archive_path)
            
            # Remove uncompressed backup directory
            shutil.rmtree(backup_path)
            
            # Calculate metadata
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            file_size = archive_path.stat().st_size
            checksum = self.calculate_checksum(archive_path)
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=start_time,
                file_path=str(archive_path),
                size_bytes=file_size,
                checksum=checksum,
                status=BackupStatus.SUCCESS,
                duration_seconds=duration,
                tables_backed_up=backed_up_tables
            )
            
            # Save metadata
            await self.save_backup_metadata(metadata)
            
            logger.info(f"Database backup completed: {backup_id} - {len(backed_up_tables)} tables, {file_size/1024/1024:.2f}MB")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Database backup failed: {backup_id} - {e}")
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=start_time,
                file_path="",
                size_bytes=0,
                checksum="",
                status=BackupStatus.FAILED,
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                tables_backed_up=[],
                error_message=str(e)
            )
            
            await self.save_backup_metadata(metadata)
            return metadata
        
        finally:
            db.close()
    
    async def get_database_tables(self, db: Session) -> List[str]:
        """Get list of all tables in database"""
        try:
            if "sqlite" in self.settings.database_url.lower():
                result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                return [row[0] for row in result.fetchall()]
            elif "postgresql" in self.settings.database_url.lower():
                result = db.execute(text("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public'
                """))
                return [row[0] for row in result.fetchall()]
            else:
                # Generic approach
                result = db.execute(text("SHOW TABLES"))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Could not retrieve table list: {e}")
            return ["users", "projects", "tasks"]  # Default tables
    
    async def backup_table(self, db: Session, table_name: str, output_file: Path) -> int:
        """Backup a single table to SQL file"""
        try:
            # Get table data
            result = db.execute(text(f"SELECT * FROM {table_name}"))
            rows = result.fetchall()
            columns = result.keys()
            
            # Write SQL dump
            with open(output_file, 'w') as f:
                # Write table structure (simplified)
                f.write(f"-- Backup of table: {table_name}\n")
                f.write(f"-- Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
                
                if rows:
                    # Write INSERT statements
                    for row in rows:
                        values = []
                        for value in row:
                            if value is None:
                                values.append("NULL")
                            elif isinstance(value, str):
                                # Escape single quotes
                                escaped_value = value.replace("'", "''")
                                values.append(f"'{escaped_value}'")
                            elif isinstance(value, (int, float)):
                                values.append(str(value))
                            elif isinstance(value, datetime):
                                values.append(f"'{value.isoformat()}'")
                            else:
                                values.append(f"'{str(value)}'")
                        
                        columns_str = ", ".join(columns)
                        values_str = ", ".join(values)
                        f.write(f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_str});\n")
            
            return len(rows)
            
        except Exception as e:
            logger.error(f"Failed to backup table {table_name}: {e}")
            # Create empty file to indicate attempt
            output_file.touch()
            return 0
    
    async def create_compressed_archive(self, source_path: Path, archive_path: Path) -> None:
        """Create compressed tar.gz archive"""
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source_path, arcname=source_path.name)
    
    async def backup_configuration(self) -> BackupMetadata:
        """Backup application configuration files"""
        backup_id = self.generate_backup_id()
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting configuration backup: {backup_id}")
        
        try:
            backup_path = self.backup_dir / "config" / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # Configuration files to backup
            config_files = [
                ".env",
                "config/",
                "src/core/config.py",
                "requirements.txt",
                "Dockerfile",
                "docker-compose.yml"
            ]
            
            backed_up_files = []
            
            for config_file in config_files:
                source_path = Path(config_file)
                if source_path.exists():
                    if source_path.is_file():
                        dest_path = backup_path / source_path.name
                        shutil.copy2(source_path, dest_path)
                        backed_up_files.append(config_file)
                    elif source_path.is_dir():
                        dest_path = backup_path / source_path.name
                        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                        backed_up_files.append(config_file)
            
            # Create archive
            archive_path = backup_path.parent / f"{backup_id}.tar.gz"
            await self.create_compressed_archive(backup_path, archive_path)
            
            # Remove uncompressed backup
            shutil.rmtree(backup_path)
            
            # Calculate metadata
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            file_size = archive_path.stat().st_size
            checksum = self.calculate_checksum(archive_path)
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.FULL,
                timestamp=start_time,
                file_path=str(archive_path),
                size_bytes=file_size,
                checksum=checksum,
                status=BackupStatus.SUCCESS,
                duration_seconds=duration,
                tables_backed_up=backed_up_files  # Reusing field for config files
            )
            
            await self.save_backup_metadata(metadata)
            
            logger.info(f"Configuration backup completed: {backup_id} - {len(backed_up_files)} files")
            return metadata
            
        except Exception as e:
            logger.error(f"Configuration backup failed: {backup_id} - {e}")
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.FULL,
                timestamp=start_time,
                file_path="",
                size_bytes=0,
                checksum="",
                status=BackupStatus.FAILED,
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                tables_backed_up=[],
                error_message=str(e)
            )
            
            await self.save_backup_metadata(metadata)
            return metadata
    
    async def restore_database(self, backup_id: str) -> RestoreResult:
        """Restore database from backup"""
        restore_id = self.generate_restore_id()
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting database restore: {restore_id} from backup {backup_id}")
        
        try:
            # Find backup metadata
            metadata = await self.get_backup_metadata(backup_id)
            if not metadata:
                raise ValueError(f"Backup not found: {backup_id}")
            
            backup_file = Path(metadata.file_path)
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Verify backup integrity
            if not await self.verify_backup_integrity(metadata):
                raise ValueError(f"Backup integrity check failed: {backup_id}")
            
            # Extract backup archive
            extract_path = backup_file.parent / f"restore_{restore_id}"
            extract_path.mkdir(exist_ok=True)
            
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(extract_path)
            
            # Get database session
            db = next(get_db())
            
            # Restore each table
            restored_tables = []
            total_records = 0
            
            sql_files = list(extract_path.rglob("*.sql"))
            
            for sql_file in sql_files:
                try:
                    table_name = sql_file.stem
                    records = await self.restore_table(db, sql_file)
                    restored_tables.append(table_name)
                    total_records += records
                    logger.debug(f"Restored table {table_name}: {records} records")
                except Exception as e:
                    logger.error(f"Failed to restore table {sql_file.stem}: {e}")
                    continue
            
            # Cleanup extraction directory
            shutil.rmtree(extract_path)
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            result = RestoreResult(
                restore_id=restore_id,
                backup_id=backup_id,
                timestamp=start_time,
                status=BackupStatus.SUCCESS,
                duration_seconds=duration,
                tables_restored=restored_tables,
                records_restored=total_records
            )
            
            logger.info(f"Database restore completed: {restore_id} - {len(restored_tables)} tables, {total_records} records")
            return result
            
        except Exception as e:
            logger.error(f"Database restore failed: {restore_id} - {e}")
            
            result = RestoreResult(
                restore_id=restore_id,
                backup_id=backup_id,
                timestamp=start_time,
                status=BackupStatus.FAILED,
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                tables_restored=[],
                records_restored=0,
                error_message=str(e)
            )
            
            return result
        
        finally:
            if 'db' in locals():
                db.close()
    
    async def restore_table(self, db: Session, sql_file: Path) -> int:
        """Restore a single table from SQL file"""
        try:
            records_restored = 0
            
            with open(sql_file, 'r') as f:
                content = f.read()
                
                # Split into individual SQL statements
                statements = [stmt.strip() for stmt in content.split(';\n') if stmt.strip() and not stmt.strip().startswith('--')]
                
                for statement in statements:
                    if statement.upper().startswith('INSERT'):
                        try:
                            db.execute(text(statement))
                            records_restored += 1
                        except Exception as e:
                            logger.warning(f"Failed to execute statement: {e}")
                            continue
                
                db.commit()
            
            return records_restored
            
        except Exception as e:
            logger.error(f"Failed to restore from {sql_file}: {e}")
            db.rollback()
            return 0
    
    async def verify_backup_integrity(self, metadata: BackupMetadata) -> bool:
        """Verify backup file integrity using checksum"""
        try:
            backup_file = Path(metadata.file_path)
            if not backup_file.exists():
                return False
            
            current_checksum = self.calculate_checksum(backup_file)
            return current_checksum == metadata.checksum
            
        except Exception as e:
            logger.error(f"Backup integrity verification failed: {e}")
            return False
    
    async def save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to file"""
        try:
            # Load existing metadata
            existing_metadata = []
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            
            # Add new metadata
            metadata_dict = asdict(metadata)
            metadata_dict['timestamp'] = metadata.timestamp.isoformat()
            existing_metadata.append(metadata_dict)
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    async def get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Retrieve backup metadata by ID"""
        try:
            if not self.metadata_file.exists():
                return None
            
            with open(self.metadata_file, 'r') as f:
                metadata_list = json.load(f)
            
            for metadata_dict in metadata_list:
                if metadata_dict['backup_id'] == backup_id:
                    # Convert timestamp back to datetime
                    metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                    return BackupMetadata(**metadata_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve backup metadata: {e}")
            return None
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List all available backups"""
        try:
            if not self.metadata_file.exists():
                return []
            
            with open(self.metadata_file, 'r') as f:
                metadata_list = json.load(f)
            
            backups = []
            for metadata_dict in metadata_list:
                metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                backups.append(BackupMetadata(**metadata_dict))
            
            # Sort by timestamp, newest first
            backups.sort(key=lambda x: x.timestamp, reverse=True)
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    async def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up backups older than retention period"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            backups = await self.list_backups()
            
            cleaned_count = 0
            remaining_metadata = []
            
            for backup in backups:
                if backup.timestamp < cutoff_date:
                    # Remove backup file
                    backup_file = Path(backup.file_path)
                    if backup_file.exists():
                        backup_file.unlink()
                        cleaned_count += 1
                        logger.info(f"Removed old backup: {backup.backup_id}")
                else:
                    # Keep this backup metadata
                    remaining_metadata.append(asdict(backup))
            
            # Update metadata file
            if cleaned_count > 0:
                for metadata in remaining_metadata:
                    metadata['timestamp'] = metadata['timestamp'].isoformat() if isinstance(metadata['timestamp'], datetime) else metadata['timestamp']
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(remaining_metadata, f, indent=2)
            
            logger.info(f"Cleaned up {cleaned_count} old backups")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0
    
    async def create_full_system_backup(self) -> Dict[str, BackupMetadata]:
        """Create comprehensive system backup"""
        logger.info("Starting full system backup")
        
        results = {}
        
        # Database backup
        try:
            db_backup = await self.backup_database(BackupType.FULL)
            results['database'] = db_backup
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            results['database'] = None
        
        # Configuration backup
        try:
            config_backup = await self.backup_configuration()
            results['configuration'] = config_backup
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            results['configuration'] = None
        
        logger.info("Full system backup completed")
        return results
    
    async def test_disaster_recovery(self) -> Dict[str, Any]:
        """Test disaster recovery procedures"""
        logger.info("Starting disaster recovery test")
        
        test_results = {
            "test_id": f"dr_test_{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": {}
        }
        
        # Test 1: Create test backup
        try:
            backup_result = await self.backup_database(BackupType.FULL)
            test_results["tests"]["backup_creation"] = {
                "status": "pass" if backup_result.status == BackupStatus.SUCCESS else "fail",
                "backup_id": backup_result.backup_id,
                "duration": backup_result.duration_seconds
            }
        except Exception as e:
            test_results["tests"]["backup_creation"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Test 2: Verify backup integrity
        if "backup_creation" in test_results["tests"] and test_results["tests"]["backup_creation"]["status"] == "pass":
            try:
                backup_id = test_results["tests"]["backup_creation"]["backup_id"]
                metadata = await self.get_backup_metadata(backup_id)
                integrity_ok = await self.verify_backup_integrity(metadata)
                
                test_results["tests"]["backup_integrity"] = {
                    "status": "pass" if integrity_ok else "fail",
                    "checksum_match": integrity_ok
                }
            except Exception as e:
                test_results["tests"]["backup_integrity"] = {
                    "status": "fail",
                    "error": str(e)
                }
        
        # Test 3: List backups
        try:
            backups = await self.list_backups()
            test_results["tests"]["backup_listing"] = {
                "status": "pass",
                "backup_count": len(backups)
            }
        except Exception as e:
            test_results["tests"]["backup_listing"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Calculate overall test status
        passed_tests = sum(1 for test in test_results["tests"].values() if test.get("status") == "pass")
        total_tests = len(test_results["tests"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "pass" if passed_tests == total_tests else "fail"
        }
        
        logger.info(f"Disaster recovery test completed: {passed_tests}/{total_tests} tests passed")
        return test_results


# Global disaster recovery manager
disaster_recovery = DisasterRecoveryManager()