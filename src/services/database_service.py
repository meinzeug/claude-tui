"""
Database Service Integration

Provides comprehensive database service integration with:
- Session management and connection pooling
- Repository pattern integration
- Transaction management
- Health checks and monitoring
- Migration support
"""

import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from .base import BaseService
from ..database.session import DatabaseSessionManager, DatabaseConfig
from ..database.repositories import RepositoryFactory
from ..database.repositories.user_repository import UserRepository
from ..database.repositories.project_repository import ProjectRepository
from ..database.repositories.task_repository import TaskRepository
from ..database.repositories.audit_repository import AuditRepository
from ..database.repositories.session_repository import SessionRepository
from ..core.exceptions import ClaudeTIUException
from ..core.logger import get_logger

logger = get_logger(__name__)


class DatabaseServiceError(ClaudeTIUException):
    """Database service specific error."""
    
    def __init__(self, message: str, error_code: str = "DATABASE_SERVICE_ERROR", details: Optional[Dict] = None):
        super().__init__(message, error_code, details)


class DatabaseService(BaseService):
    """
    Database service providing comprehensive database integration.
    
    Features:
    - Session management with connection pooling
    - Repository pattern integration
    - Transaction management
    - Health checks and monitoring
    - Migration support
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
        super().__init__()
        self.config = config or DatabaseConfig()
        self.session_manager = session_manager or DatabaseSessionManager(self.config)
        self._repository_factory: Optional[RepositoryFactory] = None
    
    async def _initialize_impl(self) -> None:
        """
        Initialize database service implementation.
        
        Raises:
            DatabaseServiceError: If initialization fails
        """
        try:
            self.logger.info("Initializing database service...")
            
            # Initialize session manager
            await self.session_manager.initialize()
            
            # Create tables if needed (for development)
            # In production, use migrations instead
            if self.config.database_url.startswith('sqlite'):
                await self.session_manager.create_tables()
                self.logger.info("Database tables created (SQLite development mode)")
            
            # Test database connection
            await self._test_database_connection()
            
            self.logger.info("Database service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database service: {e}")
            raise DatabaseServiceError(
                "Database service initialization failed",
                "DATABASE_INIT_ERROR",
                {"error": str(e), "config": str(self.config.database_url)}
            )
    
    async def _test_database_connection(self) -> None:
        """
        Test database connection.
        
        Raises:
            DatabaseServiceError: If connection test fails
        """
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            self.logger.debug("Database connection test passed")
        except Exception as e:
            raise DatabaseServiceError(
                "Database connection test failed",
                "DATABASE_CONNECTION_ERROR",
                {"error": str(e)}
            )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.
        
        Yields:
            AsyncSession instance
            
        Raises:
            DatabaseServiceError: If session creation fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.session_manager.get_session() as session:
                yield session
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
    
    async def get_repository_factory(self, session: AsyncSession) -> RepositoryFactory:
        """
        Get repository factory for the given session.
        
        Args:
            session: AsyncSession instance
            
        Returns:
            RepositoryFactory instance
        """
        return RepositoryFactory(session)
    
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
                factory = await self.get_repository_factory(session)
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
    
    async def execute_in_transaction(self, operation, *args, **kwargs) -> Any:
        """
        Execute operation within a database transaction.
        
        Args:
            operation: Async function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of the operation
            
        Raises:
            DatabaseServiceError: If transaction execution fails
        """
        try:
            async with self.get_session_with_transaction() as session:
                # Create repository factory for the operation
                repositories = await self.get_repository_factory(session)
                
                # Execute operation with repositories
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(repositories, *args, **kwargs)
                else:
                    result = operation(repositories, *args, **kwargs)
                
                return result
                
        except DatabaseServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Transaction execution error: {e}")
            raise DatabaseServiceError(
                "Transaction execution failed",
                "TRANSACTION_EXECUTION_ERROR",
                {"operation": operation.__name__ if hasattr(operation, '__name__') else str(operation), "error": str(e)}
            )
    
    async def run_migrations(self, target_revision: str = "head") -> bool:
        """
        Run database migrations using Alembic.
        
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
            import os
            
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
            
            # Run upgrade
            command.upgrade(alembic_cfg, target_revision)
            
            self.logger.info(f"Database migrations completed to revision: {target_revision}")
            return True
            
        except ImportError:
            raise DatabaseServiceError(
                "Alembic not available for migrations",
                "ALEMBIC_NOT_AVAILABLE",
                {"target_revision": target_revision}
            )
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise DatabaseServiceError(
                "Database migration failed",
                "MIGRATION_ERROR",
                {"target_revision": target_revision, "error": str(e)}
            )
    
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
            start_time = datetime.now()
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            
            info = {
                'database_url': self.config.database_url.split('@')[-1],  # Hide credentials
                'database_type': 'postgresql' if 'postgresql' in self.config.database_url else 'sqlite',
                'session_manager_health': session_health,
                'connection_pool': pool_status,
                'query_response_time_ms': round(query_time, 2),
                'service_initialized': self._initialized,
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
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for database service.
        
        Returns:
            Dictionary with health check results
        """
        base_health = await super().health_check()
        
        try:
            # Get database info
            db_info = await self.get_database_info()
            
            # Test repository operations
            repository_health = {}
            async with self.get_repositories() as repos:
                # Test each repository type
                for repo_name in ['user', 'project', 'task', 'audit', 'session']:
                    try:
                        repo = getattr(repos, f'get_{repo_name}_repository')()
                        await repo.count()  # Simple health check
                        repository_health[repo_name] = 'healthy'
                    except Exception as e:
                        repository_health[repo_name] = f'unhealthy: {str(e)}'
            
            # Combine health information
            health_status = {
                **base_health,
                'database_info': db_info,
                'repository_health': repository_health,
                'overall_status': 'healthy' if all(
                    status == 'healthy' for status in repository_health.values()
                ) and self._initialized else 'unhealthy'
            }
            
            return health_status
            
        except Exception as e:
            return {
                **base_health,
                'overall_status': 'unhealthy',
                'error': str(e)
            }
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired user sessions.
        
        Returns:
            Number of sessions cleaned up
            
        Raises:
            DatabaseServiceError: If cleanup fails
        """
        try:
            async with self.get_repositories() as repos:
                session_repo = repos.get_session_repository()
                cleaned_count = await session_repo.cleanup_expired_sessions()
                
                self.logger.info(f"Cleaned up {cleaned_count} expired sessions")
                return cleaned_count
                
        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")
            raise DatabaseServiceError(
                "Failed to cleanup expired sessions",
                "SESSION_CLEANUP_ERROR",
                {"error": str(e)}
            )
    
    async def cleanup_old_audit_logs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old audit log records.
        
        Args:
            days_to_keep: Number of days of audit logs to retain
            
        Returns:
            Number of records cleaned up
            
        Raises:
            DatabaseServiceError: If cleanup fails
        """
        try:
            async with self.get_repositories() as repos:
                audit_repo = repos.get_audit_repository()
                cleaned_count = await audit_repo.cleanup_old_records(days_to_keep)
                
                self.logger.info(f"Cleaned up {cleaned_count} old audit records")
                return cleaned_count
                
        except Exception as e:
            self.logger.error(f"Audit log cleanup failed: {e}")
            raise DatabaseServiceError(
                "Failed to cleanup old audit logs",
                "AUDIT_CLEANUP_ERROR",
                {"days_to_keep": days_to_keep, "error": str(e)}
            )
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Dictionary with database statistics
            
        Raises:
            DatabaseServiceError: If statistics retrieval fails
        """
        try:
            async with self.get_repositories() as repos:
                # Get repository-specific statistics
                user_repo = repos.get_user_repository()
                project_repo = repos.get_project_repository()
                task_repo = repos.get_task_repository()
                audit_repo = repos.get_audit_repository()
                session_repo = repos.get_session_repository()
                
                # Collect counts
                statistics = {
                    'total_users': await user_repo.count(),
                    'active_users': await user_repo.count({'is_active': True}),
                    'total_projects': await project_repo.count(),
                    'public_projects': await project_repo.count({'is_public': True}),
                    'total_tasks': await task_repo.count(),
                    'completed_tasks': await task_repo.count({'status': 'completed'}),
                    'active_sessions': await session_repo.count({
                        'is_active': True,
                        'expires_at__gt': datetime.now()
                    }),
                    'audit_records_7_days': await audit_repo.count({
                        'created_at__gte': datetime.now() - timedelta(days=7)
                    }),
                    'generated_at': datetime.now()
                }
                
                # Get detailed statistics
                task_stats = await task_repo.get_task_statistics()
                session_stats = await session_repo.get_session_statistics()
                
                statistics.update({
                    'task_statistics': task_stats,
                    'session_statistics': session_stats
                })
                
                return statistics
                
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            raise DatabaseServiceError(
                "Failed to get database statistics",
                "GET_DATABASE_STATISTICS_ERROR",
                {"error": str(e)}
            )
    
    async def close(self) -> None:
        """
        Close database service and cleanup resources.
        """
        try:
            if self.session_manager:
                await self.session_manager.close()
            
            self._repository_factory = None
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


# Convenience functions
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
    config = DatabaseConfig(database_url=database_url, **config_kwargs)
    service = DatabaseService(config=config)
    await service.initialize()
    return service


async def get_database_service() -> DatabaseService:
    """
    Get database service instance (singleton pattern).
    
    Returns:
        DatabaseService instance
    """
    # This could be enhanced with proper dependency injection
    # For now, create a new instance
    return await create_database_service()
