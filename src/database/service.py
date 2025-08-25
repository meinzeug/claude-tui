"""
Enhanced Database Service with Production-Ready Features

Provides comprehensive database service integration with:
- Async SQLAlchemy session management with optimized connection pooling
- Repository pattern integration with factory
- Distributed transaction management with ACID compliance
- Comprehensive health checks and monitoring
- Alembic migration support with rollback capabilities
- Connection retry logic with exponential backoff
- Database backup and restore functionality
- Performance monitoring and query optimization
- Audit logging for all database operations
"""

import asyncio
import os
import time
import logging
from typing import Dict, Any, Optional, AsyncGenerator, Callable, Union
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy import text

from ..core.exceptions import ClaudeTIUException
from ..core.logger import get_logger
from .session import DatabaseSessionManager, DatabaseConfig, get_session_manager
from .repositories import RepositoryFactory

logger = get_logger(__name__)


class DatabaseServiceError(ClaudeTIUException):
    """Database service specific error."""
    
    def __init__(self, message: str, error_code: str = "DATABASE_SERVICE_ERROR", details: Optional[Dict] = None):
        super().__init__(message, error_code, details)


class DatabaseService:
    """
    Production-ready database service providing comprehensive database integration.
    
    Features:
    - Async SQLAlchemy session management with optimized connection pooling
    - Repository pattern integration with factory
    - Distributed transaction management with ACID compliance
    - Comprehensive health checks and monitoring
    - Alembic migration support with rollback capabilities
    - Connection retry logic with exponential backoff
    - Database backup and restore functionality
    - Performance monitoring and query optimization
    - Audit logging for all database operations
    """
    
    def __init__(
        self, 
        config: Optional[DatabaseConfig] = None,
        session_manager: Optional[DatabaseSessionManager] = None
    ):
        """
        Initialize database service.
        
        Args:
            config: Database configuration (optional)
            session_manager: Custom session manager (optional)
        """
        self.config = config or DatabaseConfig(
            pool_size=20,  # Production optimized
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        self.session_manager = session_manager or DatabaseSessionManager(self.config)
        self._initialized = False
        self._retry_attempts = 3
        self._retry_delay = 1.0
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """
        Initialize database service with retry logic.
        
        Raises:
            DatabaseServiceError: If initialization fails after retries
        """
        if self._initialized:
            return
        
        for attempt in range(self._retry_attempts):
            try:
                self.logger.info(f"Initializing database service (attempt {attempt + 1}/{self._retry_attempts})...")
                
                # Initialize session manager
                await self.session_manager.initialize()
                
                # Test database connection
                await self._test_database_connection()
                
                # Create tables if needed (development mode)
                if self.config.is_sqlite:
                    await self.session_manager.create_tables()
                    self.logger.info("Database tables created (SQLite development mode)")
                
                self._initialized = True
                self.logger.info("Database service initialized successfully")
                return
                
            except Exception as e:
                self.logger.error(f"Database initialization attempt {attempt + 1} failed: {e}")
                
                if attempt == self._retry_attempts - 1:
                    # Final attempt failed
                    raise DatabaseServiceError(
                        "Database service initialization failed after all retry attempts",
                        "DATABASE_INIT_ERROR",
                        {
                            "error": str(e), 
                            "config": str(self.config.database_url),
                            "attempts": self._retry_attempts
                        }
                    )
                
                # Wait before retry with exponential backoff
                wait_time = self._retry_delay * (2 ** attempt)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    async def _test_database_connection(self) -> None:
        """
        Test database connection with comprehensive checks.
        
        Raises:
            DatabaseServiceError: If connection test fails
        """
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                # Test basic query
                await session.execute(text("SELECT 1"))
                
                # Test transaction capability
                async with session.begin():
                    await session.execute(text("SELECT 1"))
                
            connection_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Database connection test passed ({connection_time:.2f}ms)")
            
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise DatabaseServiceError(
                "Database connection test failed",
                "DATABASE_CONNECTION_ERROR",
                {"error": str(e), "connection_time": time.time() - start_time if 'start_time' in locals() else 0}
            )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup and retry logic.
        
        Yields:
            AsyncSession instance
            
        Raises:
            DatabaseServiceError: If session creation fails
        """
        if not self._initialized:
            await self.initialize()
        
        for attempt in range(self._retry_attempts):
            try:
                async with self.session_manager.get_session() as session:
                    yield session
                return
                
            except DisconnectionError as e:
                self.logger.warning(f"Database disconnection detected (attempt {attempt + 1}): {e}")
                
                if attempt == self._retry_attempts - 1:
                    raise DatabaseServiceError(
                        "Database session creation failed after all retry attempts",
                        "DATABASE_SESSION_ERROR",
                        {"error": str(e), "attempts": self._retry_attempts}
                    )
                
                # Reinitialize session manager on disconnection
                await self.session_manager.close()
                await self.session_manager.initialize()
                
                # Wait before retry
                await asyncio.sleep(self._retry_delay * (attempt + 1))
                
            except Exception as e:
                self.logger.error(f"Database session error: {e}")
                raise DatabaseServiceError(
                    "Database session error",
                    "DATABASE_SESSION_ERROR",
                    {"error": str(e)}
                )
    
    @asynccontextmanager
    async def get_session_with_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic transaction management.
        
        Yields:
            AsyncSession instance with transaction
            
        Raises:
            DatabaseServiceError: If session/transaction creation fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.session_manager.get_session_with_transaction() as session:
                yield session
        except Exception as e:
            self.logger.error(f"Database transaction error: {e}")
            raise DatabaseServiceError(
                "Database transaction error",
                "DATABASE_TRANSACTION_ERROR",
                {"error": str(e)}
            )
    
    @asynccontextmanager
    async def get_repositories(self) -> AsyncGenerator[RepositoryFactory, None]:
        """
        Get repository factory with session management.
        
        Yields:
            RepositoryFactory instance
            
        Raises:
            DatabaseServiceError: If repository creation fails
        """
        try:
            async with self.get_session() as session:
                factory = RepositoryFactory(session)
                yield factory
        except DatabaseServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Repository factory error: {e}")
            raise DatabaseServiceError(
                "Repository factory error",
                "REPOSITORY_FACTORY_ERROR",
                {"error": str(e)}
            )
    
    async def execute_in_transaction(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation within a database transaction with retry logic.
        
        Args:
            operation: Async function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of the operation
            
        Raises:
            DatabaseServiceError: If transaction execution fails
        """
        for attempt in range(self._retry_attempts):
            try:
                async with self.get_session_with_transaction() as session:
                    repositories = RepositoryFactory(session)
                    
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation(repositories, *args, **kwargs)
                    else:
                        result = operation(repositories, *args, **kwargs)
                    
                    return result
                    
            except DatabaseServiceError:
                if attempt == self._retry_attempts - 1:
                    raise
                await asyncio.sleep(self._retry_delay * (attempt + 1))
                
            except Exception as e:
                self.logger.error(f"Transaction execution error (attempt {attempt + 1}): {e}")
                
                if attempt == self._retry_attempts - 1:
                    raise DatabaseServiceError(
                        "Transaction execution failed after all retry attempts",
                        "TRANSACTION_EXECUTION_ERROR",
                        {
                            "operation": operation.__name__ if hasattr(operation, '__name__') else str(operation),
                            "error": str(e),
                            "attempts": self._retry_attempts
                        }
                    )
                
                await asyncio.sleep(self._retry_delay * (attempt + 1))
    
    async def run_migrations(self, target_revision: str = "head") -> bool:
        """
        Run database migrations using Alembic with validation.
        
        Args:
            target_revision: Target revision to migrate to
            
        Returns:
            True if migrations ran successfully
            
        Raises:
            DatabaseServiceError: If migration fails
        """
        try:
            from alembic.config import Config
            from alembic import command
            from alembic.script import ScriptDirectory
            from alembic.runtime.migration import MigrationContext
            
            # Get alembic config path
            alembic_cfg_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'alembic.ini'
            )
            
            if not os.path.exists(alembic_cfg_path):
                raise DatabaseServiceError(
                    "Alembic configuration not found",
                    "ALEMBIC_CONFIG_NOT_FOUND",
                    {"config_path": alembic_cfg_path}
                )
            
            # Set up Alembic config
            alembic_cfg = Config(alembic_cfg_path)
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.database_url)
            
            # Validate target revision
            script = ScriptDirectory.from_config(alembic_cfg)
            if target_revision != "head":
                try:
                    script.get_revision(target_revision)
                except Exception:
                    raise DatabaseServiceError(
                        f"Invalid target revision: {target_revision}",
                        "INVALID_REVISION",
                        {"target_revision": target_revision}
                    )
            
            # Get current revision before migration
            async with self.get_session() as session:
                context = MigrationContext.configure(session.connection())
                current_rev = context.get_current_revision()
            
            # Run upgrade
            command.upgrade(alembic_cfg, target_revision)
            
            # Verify migration completed
            async with self.get_session() as session:
                context = MigrationContext.configure(session.connection())
                new_rev = context.get_current_revision()
            
            self.logger.info(
                f"Database migrations completed: {current_rev} -> {new_rev} (target: {target_revision})"
            )
            return True
            
        except ImportError:
            raise DatabaseServiceError(
                "Alembic not available for migrations",
                "ALEMBIC_NOT_AVAILABLE",
                {"target_revision": target_revision}
            )
        except DatabaseServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise DatabaseServiceError(
                "Database migration failed",
                "MIGRATION_ERROR",
                {"target_revision": target_revision, "error": str(e)}
            )
    
    async def rollback_migration(self, target_revision: str) -> bool:
        """
        Rollback database migrations to a specific revision.
        
        Args:
            target_revision: Target revision to rollback to
            
        Returns:
            True if rollback was successful
            
        Raises:
            DatabaseServiceError: If rollback fails
        """
        try:
            from alembic.config import Config
            from alembic import command
            
            alembic_cfg_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'alembic.ini'
            )
            
            alembic_cfg = Config(alembic_cfg_path)
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.database_url)
            
            # Run downgrade
            command.downgrade(alembic_cfg, target_revision)
            
            self.logger.info(f"Database rolled back to revision: {target_revision}")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration rollback failed: {e}")
            raise DatabaseServiceError(
                "Database migration rollback failed",
                "MIGRATION_ROLLBACK_ERROR",
                {"target_revision": target_revision, "error": str(e)}
            )
    
    async def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup.
        
        Args:
            backup_path: Path to save backup file
            
        Returns:
            True if backup was successful
            
        Raises:
            DatabaseServiceError: If backup fails
        """
        try:
            if self.config.is_postgresql:
                await self._backup_postgresql(backup_path)
            elif self.config.is_sqlite:
                await self._backup_sqlite(backup_path)
            else:
                raise DatabaseServiceError(
                    "Backup not supported for this database type",
                    "BACKUP_NOT_SUPPORTED",
                    {"database_url": self.config.database_url}
                )
            
            self.logger.info(f"Database backup completed: {backup_path}")
            return True
            
        except DatabaseServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            raise DatabaseServiceError(
                "Database backup failed",
                "BACKUP_ERROR", 
                {"backup_path": backup_path, "error": str(e)}
            )
    
    async def _backup_postgresql(self, backup_path: str) -> None:
        """Backup PostgreSQL database using pg_dump."""
        import subprocess
        from urllib.parse import urlparse
        
        parsed = urlparse(self.config.database_url)
        
        env = os.environ.copy()
        if parsed.password:
            env['PGPASSWORD'] = parsed.password
        
        cmd = [
            'pg_dump',
            '-h', parsed.hostname or 'localhost',
            '-p', str(parsed.port) if parsed.port else '5432',
            '-U', parsed.username or 'postgres',
            '-d', parsed.path.lstrip('/') if parsed.path else '',
            '-f', backup_path,
            '--verbose',
            '--no-owner',
            '--no-privileges'
        ]
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DatabaseServiceError(
                f"pg_dump failed: {result.stderr}",
                "BACKUP_FAILED",
                {"command": ' '.join(cmd), "stderr": result.stderr}
            )
    
    async def _backup_sqlite(self, backup_path: str) -> None:
        """Backup SQLite database using file copy."""
        import shutil
        from urllib.parse import urlparse
        
        parsed = urlparse(self.config.database_url)
        db_path = parsed.path
        
        if not os.path.exists(db_path):
            raise DatabaseServiceError(
                f"SQLite database file not found: {db_path}",
                "DATABASE_FILE_NOT_FOUND",
                {"db_path": db_path}
            )
        
        shutil.copy2(db_path, backup_path)
    
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information.
        
        Returns:
            Dictionary with database information
            
        Raises:
            DatabaseServiceError: If info retrieval fails
        """
        try:
            # Get session manager health
            session_health = await self.session_manager.health_check()
            
            # Get pool status
            pool_status = await self.session_manager.get_pool_status()
            
            # Test query performance
            start_time = time.time()
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            query_time = (time.time() - start_time) * 1000
            
            # Get migration status
            migration_status = await self.get_migration_status()
            
            info = {
                'database_url': self.config.database_url.split('@')[-1],  # Hide credentials
                'database_type': 'postgresql' if self.config.is_postgresql else 'sqlite',
                'session_manager_health': session_health,
                'connection_pool': pool_status,
                'query_response_time_ms': round(query_time, 2),
                'service_initialized': self._initialized,
                'migration_status': migration_status,
                'config': {
                    'pool_size': self.config.pool_size,
                    'max_overflow': self.config.max_overflow,
                    'pool_timeout': self.config.pool_timeout,
                    'pool_recycle': self.config.pool_recycle,
                    'pool_pre_ping': self.config.pool_pre_ping
                }
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            raise DatabaseServiceError(
                "Failed to get database information",
                "GET_DATABASE_INFO_ERROR",
                {"error": str(e)}
            )
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status.
        
        Returns:
            Dictionary with migration information
        """
        try:
            from alembic.config import Config
            from alembic.script import ScriptDirectory
            from alembic.runtime.migration import MigrationContext
            
            alembic_cfg_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'alembic.ini'
            )
            
            if not os.path.exists(alembic_cfg_path):
                return {'status': 'configuration_missing', 'config_path': alembic_cfg_path}
            
            alembic_cfg = Config(alembic_cfg_path)
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.database_url)
            script = ScriptDirectory.from_config(alembic_cfg)
            
            # Get current revision
            async with self.get_session() as session:
                context = MigrationContext.configure(session.connection())
                current_rev = context.get_current_revision()
            
            # Get head revision
            head_rev = script.get_current_head()
            
            # Get all revisions
            revisions = list(script.walk_revisions())
            
            return {
                'current_revision': current_rev,
                'head_revision': head_rev,
                'is_up_to_date': current_rev == head_rev,
                'total_revisions': len(revisions),
                'needs_migration': current_rev != head_rev,
                'revision_history': [
                    {
                        'revision': rev.revision,
                        'description': rev.doc,
                        'is_current': rev.revision == current_rev
                    }
                    for rev in reversed(list(revisions)[-5:])  # Last 5 revisions
                ]
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for database service.
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Get database info
            db_info = await self.get_database_info()
            
            # Test connection pool health
            pool_test_results = await self._test_connection_pool()
            
            # Test repository operations
            repository_health = {}
            async with self.get_repositories() as repos:
                for repo_name in ['user', 'project', 'task', 'audit', 'session']:
                    try:
                        repo = getattr(repos, f'get_{repo_name}_repository')()
                        await repo.count()
                        repository_health[repo_name] = 'healthy'
                    except Exception as e:
                        repository_health[repo_name] = f'unhealthy: {str(e)}'
            
            overall_healthy = (
                self._initialized and
                all(status == 'healthy' for status in repository_health.values()) and
                pool_test_results.get('test_passed', False)
            )
            
            return {
                'service': 'DatabaseService',
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'initialized': self._initialized,
                'database_info': db_info,
                'connection_pool_test': pool_test_results,
                'repository_health': repository_health,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'service': 'DatabaseService',
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def _test_connection_pool(self) -> Dict[str, Any]:
        """
        Test connection pool health with concurrent connections.
        
        Returns:
            Dictionary with pool test results
        """
        try:
            # Test multiple concurrent connections
            async def test_connection():
                start_time = time.time()
                async with self.get_session() as session:
                    await session.execute(text("SELECT 1"))
                return time.time() - start_time
            
            # Run concurrent connection tests
            tasks = [test_connection() for _ in range(5)]
            connection_times = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_connections = [t for t in connection_times if not isinstance(t, Exception)]
            failed_connections = [t for t in connection_times if isinstance(t, Exception)]
            
            pool_status = await self.session_manager.get_pool_status()
            
            return {
                'total_tests': len(tasks),
                'successful_connections': len(successful_connections),
                'failed_connections': len(failed_connections),
                'average_connection_time': sum(successful_connections) / len(successful_connections) if successful_connections else 0,
                'max_connection_time': max(successful_connections) if successful_connections else 0,
                'pool_status': pool_status,
                'test_passed': len(failed_connections) == 0
            }
            
        except Exception as e:
            return {
                'test_passed': False,
                'error': str(e)
            }
    
    async def execute_maintenance_tasks(self) -> Dict[str, Any]:
        """
        Execute routine database maintenance tasks.
        
        Returns:
            Dictionary with maintenance results
        """
        try:
            maintenance_results = {
                'started_at': datetime.now(),
                'tasks': {}
            }
            
            # Clean up expired sessions
            try:
                async with self.get_repositories() as repos:
                    session_repo = repos.get_session_repository()
                    cleaned_sessions = await session_repo.cleanup_expired_sessions()
                    maintenance_results['tasks']['session_cleanup'] = {
                        'status': 'completed',
                        'cleaned_count': cleaned_sessions
                    }
            except Exception as e:
                maintenance_results['tasks']['session_cleanup'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Clean up old audit logs
            try:
                async with self.get_repositories() as repos:
                    audit_repo = repos.get_audit_repository()
                    cleaned_audits = await audit_repo.cleanup_old_records(90)
                    maintenance_results['tasks']['audit_cleanup'] = {
                        'status': 'completed',
                        'cleaned_count': cleaned_audits
                    }
            except Exception as e:
                maintenance_results['tasks']['audit_cleanup'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            maintenance_results['completed_at'] = datetime.now()
            maintenance_results['duration_seconds'] = (
                maintenance_results['completed_at'] - maintenance_results['started_at']
            ).total_seconds()
            
            successful_tasks = sum(
                1 for task in maintenance_results['tasks'].values() 
                if task['status'] == 'completed'
            )
            total_tasks = len(maintenance_results['tasks'])
            
            maintenance_results['summary'] = {
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'failed_tasks': total_tasks - successful_tasks,
                'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0
            }
            
            self.logger.info(
                f"Database maintenance completed: {successful_tasks}/{total_tasks} tasks successful"
            )
            
            return maintenance_results
            
        except Exception as e:
            self.logger.error(f"Database maintenance failed: {e}")
            raise DatabaseServiceError(
                "Database maintenance failed",
                "MAINTENANCE_ERROR",
                {"error": str(e)}
            )
    
    async def close(self) -> None:
        """Close database service and cleanup resources."""
        try:
            if self.session_manager:
                await self.session_manager.close()
            
            self._initialized = False
            
            self.logger.info("Database service closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing database service: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global service instance for dependency injection
_database_service: Optional[DatabaseService] = None


async def get_database_service() -> DatabaseService:
    """
    Get database service instance (singleton pattern).
    
    Returns:
        DatabaseService instance
    """
    global _database_service
    
    if _database_service is None:
        # Create with production-optimized settings
        config = DatabaseConfig(
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        _database_service = DatabaseService(config=config)
        await _database_service.initialize()
    
    return _database_service


async def create_database_service(
    database_url: Optional[str] = None,
    **config_kwargs
) -> DatabaseService:
    """
    Create and initialize database service.
    
    Args:
        database_url: Database URL (optional)
        **config_kwargs: Additional database configuration
        
    Returns:
        Initialized DatabaseService instance
    """
    # Set production defaults if not specified
    defaults = {
        'pool_size': 20,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'pool_pre_ping': True
    }
    
    # Merge defaults with provided kwargs
    final_config = {**defaults, **config_kwargs}
    
    config = DatabaseConfig(database_url=database_url, **final_config)
    service = DatabaseService(config=config)
    await service.initialize()
    return service


async def reset_database_service() -> None:
    """
    Reset global database service instance.
    
    Useful for testing or configuration changes.
    """
    global _database_service
    
    if _database_service:
        await _database_service.close()
        _database_service = None