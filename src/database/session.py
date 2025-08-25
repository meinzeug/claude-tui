"""
Database Session Manager with AsyncSession and Connection Pooling

Provides comprehensive database session management with:
- SQLAlchemy 2.0 AsyncSession support
- Connection pooling configuration
- Context managers for sessions
- Transaction management
- Error handling and logging
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import event, text
import os
from urllib.parse import urlparse

from .models import Base
from ..core.config import get_settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class DatabaseConfig:
    """Database configuration with security and performance settings."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        echo: bool = False,
        echo_pool: bool = False
    ):
        self.database_url = database_url or self._get_database_url()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.echo = echo
        self.echo_pool = echo_pool
        
        # Validate configuration
        self._validate_config()
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or settings."""
        settings = get_settings()
        return getattr(settings, 'database_url', 'sqlite+aiosqlite:///./claude_tui.db')
    
    def _validate_config(self) -> None:
        """Validate database configuration."""
        if not self.database_url:
            raise ValueError("Database URL is required")
        
        # Parse URL to validate format
        try:
            parsed = urlparse(self.database_url)
            if not parsed.scheme:
                raise ValueError("Invalid database URL format")
        except Exception as e:
            raise ValueError(f"Invalid database URL: {e}")
        
        # Validate pool settings
        if self.pool_size <= 0:
            raise ValueError("Pool size must be positive")
        if self.max_overflow < 0:
            raise ValueError("Max overflow cannot be negative")
        if self.pool_timeout <= 0:
            raise ValueError("Pool timeout must be positive")
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return 'sqlite' in self.database_url.lower()
    
    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return 'postgresql' in self.database_url.lower()
    
    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get engine configuration kwargs."""
        kwargs = {
            'echo': self.echo,
            'echo_pool': self.echo_pool,
            'pool_pre_ping': self.pool_pre_ping,
            'pool_recycle': self.pool_recycle,
        }
        
        # SQLite uses StaticPool for simplicity
        if self.is_sqlite:
            kwargs.update({
                'poolclass': StaticPool,
                'connect_args': {'check_same_thread': False},
            })
        else:
            # PostgreSQL and other databases use QueuePool
            kwargs.update({
                'poolclass': QueuePool,
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
            })
        
        return kwargs


class DatabaseSessionManager:
    """Database session manager with connection pooling and lifecycle management."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False
        logger.info(f"Database session manager initialized with URL: {self.config.database_url}")
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing database engine...")
            
            # Create async engine
            engine_kwargs = self.config.get_engine_kwargs()
            self._engine = create_async_engine(
                self.config.database_url,
                **engine_kwargs
            )
            
            # Set up event listeners
            self._setup_event_listeners()
            
            # Create session factory
            self._sessionmaker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            logger.info("Database session manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            await self.close()
            raise
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring."""
        if not self._engine:
            return
        
        @event.listens_for(self._engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            logger.debug("Database connection established")
            
            # Enable foreign keys for SQLite
            if self.config.is_sqlite:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        
        @event.listens_for(self._engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug("Database connection checked out from pool")
        
        @event.listens_for(self._engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            logger.debug("Database connection checked in to pool")
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        try:
            logger.info("Creating database tables...")
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        try:
            logger.warning("Dropping database tables...")
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        if not self._sessionmaker:
            raise RuntimeError("Session manager not initialized")
        
        session = self._sessionmaker()
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_session_with_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic transaction management."""
        if not self._sessionmaker:
            raise RuntimeError("Session manager not initialized")
        
        session = self._sessionmaker()
        try:
            async with session.begin():
                yield session
        except Exception as e:
            logger.error(f"Database transaction error: {e}")
            # Rollback is automatic with session.begin()
            raise
        finally:
            await session.close()
    
    async def execute_raw_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL query with parameters."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text(query), parameters or {})
                return result
        except SQLAlchemyError as e:
            logger.error(f"Raw query execution failed: {e}")
            raise
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status information."""
        if not self._engine or not hasattr(self._engine.pool, 'size'):
            return {'status': 'not_initialized'}
        
        pool = self._engine.pool
        return {
            'pool_size': getattr(pool, 'size', lambda: 'N/A')(),
            'checked_in': getattr(pool, 'checkedin', lambda: 'N/A')(),
            'checked_out': getattr(pool, 'checkedout', lambda: 'N/A')(),
            'overflow': getattr(pool, 'overflow', lambda: 'N/A')(),
            'invalid': getattr(pool, 'invalid', lambda: 'N/A')(),
            'status': 'healthy' if self._initialized else 'not_initialized'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        health_status = {
            'initialized': self._initialized,
            'database_url': self.config.database_url.split('@')[-1],  # Hide credentials
            'engine_status': 'healthy' if self._engine else 'not_created',
        }
        
        if self._initialized and self._engine:
            try:
                # Test query
                async with self._engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                health_status['connection_test'] = 'passed'
                
                # Pool status
                health_status['pool_status'] = await self.get_pool_status()
                
            except Exception as e:
                health_status['connection_test'] = f'failed: {str(e)}'
                health_status['status'] = 'unhealthy'
                return health_status
        
        health_status['status'] = 'healthy' if self._initialized else 'not_initialized'
        return health_status
    
    async def close(self) -> None:
        """Close database connections and cleanup."""
        if self._engine:
            logger.info("Closing database connections...")
            await self._engine.dispose()
            self._engine = None
        
        self._sessionmaker = None
        self._initialized = False
        logger.info("Database session manager closed")
    
    def __del__(self):
        """Cleanup on object deletion."""
        if self._initialized:
            logger.warning("DatabaseSessionManager was not properly closed")


# Global session manager instance
_session_manager: Optional[DatabaseSessionManager] = None


def get_session_manager() -> DatabaseSessionManager:
    """Get global database session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = DatabaseSessionManager()
    return _session_manager


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Convenience function to get database session."""
    session_manager = get_session_manager()
    if not session_manager._initialized:
        await session_manager.initialize()
    
    async with session_manager.get_session() as session:
        yield session


async def get_database_session_with_transaction() -> AsyncGenerator[AsyncSession, None]:
    """Convenience function to get database session with transaction."""
    session_manager = get_session_manager()
    if not session_manager._initialized:
        await session_manager.initialize()
    
    async with session_manager.get_session_with_transaction() as session:
        yield session


# Context manager for database operations
class DatabaseContext:
    """Context manager for database operations with error handling."""
    
    def __init__(self, session_manager: Optional[DatabaseSessionManager] = None):
        self.session_manager = session_manager or get_session_manager()
        self.session: Optional[AsyncSession] = None
    
    async def __aenter__(self) -> AsyncSession:
        """Enter context and return session."""
        if not self.session_manager._initialized:
            await self.session_manager.initialize()
        
        self.session = self.session_manager._sessionmaker()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup session."""
        if self.session:
            if exc_type:
                await self.session.rollback()
                logger.error(f"Database operation failed: {exc_val}")
            else:
                try:
                    await self.session.commit()
                except Exception as e:
                    await self.session.rollback()
                    logger.error(f"Failed to commit transaction: {e}")
                    raise
            
            await self.session.close()
        
        return False  # Don't suppress exceptions
