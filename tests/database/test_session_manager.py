"""
Tests for Database Session Manager

Comprehensive tests for:
- Session creation and management
- Connection pooling
- Error handling
- Health checks
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from src.database.session import DatabaseSessionManager, DatabaseConfig
from src.database.models import Base


@pytest_asyncio.fixture
async def database_config():
    """Test database configuration."""
    return DatabaseConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        pool_size=5,
        max_overflow=10
    )


@pytest_asyncio.fixture
async def session_manager(database_config):
    """Test session manager."""
    manager = DatabaseSessionManager(database_config)
    await manager.initialize()
    yield manager
    await manager.close()


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = DatabaseConfig(
            database_url="sqlite+aiosqlite:///test.db",
            pool_size=5
        )
        assert config.database_url == "sqlite+aiosqlite:///test.db"
        assert config.pool_size == 5
    
    def test_invalid_config(self):
        """Test invalid configuration handling."""
        with pytest.raises(ValueError):
            DatabaseConfig(database_url="")
        
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=-1)
    
    def test_database_type_detection(self):
        """Test database type detection."""
        sqlite_config = DatabaseConfig("sqlite+aiosqlite:///test.db")
        assert sqlite_config.is_sqlite is True
        assert sqlite_config.is_postgresql is False
        
        pg_config = DatabaseConfig("postgresql+asyncpg://user:pass@localhost/db")
        assert pg_config.is_sqlite is False
        assert pg_config.is_postgresql is True


class TestDatabaseSessionManager:
    """Test database session manager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, database_config):
        """Test session manager initialization."""
        manager = DatabaseSessionManager(database_config)
        
        assert manager._initialized is False
        await manager.initialize()
        assert manager._initialized is True
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_session_creation(self, session_manager):
        """Test session creation."""
        async with session_manager.get_session() as session:
            assert session is not None
            # Test simple query
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_session_with_transaction(self, session_manager):
        """Test session with transaction."""
        async with session_manager.get_session_with_transaction() as session:
            assert session is not None
            # Transaction should be active
            assert session.in_transaction() is True
    
    @pytest.mark.asyncio
    async def test_create_tables(self, session_manager):
        """Test table creation."""
        await session_manager.create_tables()
        
        # Verify tables were created by checking metadata
        async with session_manager.get_session() as session:
            # This should not raise an error
            result = await session.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = result.fetchall()
            
            # Should have our model tables
            table_names = [table[0] for table in tables]
            assert 'users' in table_names
            assert 'projects' in table_names
            assert 'tasks' in table_names
    
    @pytest.mark.asyncio
    async def test_health_check(self, session_manager):
        """Test health check functionality."""
        health = await session_manager.health_check()
        
        assert 'initialized' in health
        assert 'database_url' in health
        assert 'engine_status' in health
        assert health['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_pool_status(self, session_manager):
        """Test connection pool status."""
        pool_status = await session_manager.get_pool_status()
        
        assert 'status' in pool_status
        # SQLite in-memory doesn't have traditional pooling
        assert pool_status['status'] in ['healthy', 'not_initialized']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, database_config):
        """Test error handling in session manager."""
        # Test with invalid database URL
        bad_config = DatabaseConfig("invalid://url")
        manager = DatabaseSessionManager(bad_config)
        
        with pytest.raises(Exception):
            await manager.initialize()


class TestSessionContext:
    """Test session context managers."""
    
    @pytest.mark.asyncio
    async def test_session_context_success(self, session_manager):
        """Test successful session context."""
        async with session_manager.get_session() as session:
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_session_context_error(self, session_manager):
        """Test session context with error."""
        with pytest.raises(Exception):
            async with session_manager.get_session() as session:
                # This should cause an error
                await session.execute("INVALID SQL")
    
    @pytest.mark.asyncio
    async def test_transaction_context_rollback(self, session_manager):
        """Test transaction rollback on error."""
        # Create tables first
        await session_manager.create_tables()
        
        with pytest.raises(Exception):
            async with session_manager.get_session_with_transaction() as session:
                # Insert some data
                await session.execute(
                    "INSERT INTO users (email, username, hashed_password) VALUES (?, ?, ?)",
                    ("test@example.com", "testuser", "hashedpass")
                )
                # Force an error
                raise Exception("Test error")
        
        # Verify transaction was rolled back
        async with session_manager.get_session() as session:
            result = await session.execute("SELECT COUNT(*) FROM users")
            count = result.scalar()
            assert count == 0


@pytest.mark.asyncio
async def test_global_session_functions():
    """Test global session utility functions."""
    from src.database.session import get_session_manager
    
    manager = get_session_manager()
    assert isinstance(manager, DatabaseSessionManager)