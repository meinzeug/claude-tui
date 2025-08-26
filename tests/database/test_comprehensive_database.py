"""
Comprehensive Database Integration Tests

Complete test suite covering all database components:
- Database service initialization and configuration
- Session management and connection pooling
- Repository pattern implementation
- Transaction management with rollback
- Migration support and management
- Health checks and monitoring
- Backup and restore functionality
- Performance testing under load
- Error handling and recovery
"""

import pytest
import asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4

from src.database.service import DatabaseService, DatabaseServiceError, get_database_service
from src.database.session import DatabaseConfig, DatabaseSessionManager
from src.database.repositories import RepositoryFactory
from src.database.models import User, Project, Task, AuditLog, UserSession
from src.core.exceptions import ClaudeTIUException


@pytest.fixture
async def test_db_config():
    """Create test database configuration with optimized settings."""
    return DatabaseConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        pool_size=10,
        max_overflow=5,
        pool_timeout=30,
        pool_pre_ping=True,
        echo=False
    )


@pytest.fixture
async def db_service(test_db_config):
    """Create and initialize database service for testing."""
    service = DatabaseService(config=test_db_config)
    await service.initialize()
    
    # Create tables for testing
    async with service.get_session() as session:
        # Tables should be created automatically for SQLite
        pass
    
    yield service
    await service.close()


@pytest.fixture
async def sample_user_data():
    """Sample user data for testing."""
    return {
        'email': 'test@example.com',
        'username': 'testuser',
        'password': 'TestPassword123!',
        'full_name': 'Test User'
    }


@pytest.fixture
async def sample_project_data():
    """Sample project data for testing."""
    return {
        'name': 'Test Project',
        'description': 'A test project for unit testing',
        'project_type': 'development',
        'is_public': False
    }


class TestDatabaseServiceInitialization:
    """Test database service initialization and configuration."""
    
    async def test_service_creation(self, test_db_config):
        """Test basic service creation."""
        service = DatabaseService(config=test_db_config)
        
        assert service.config == test_db_config
        assert service._initialized is False
        assert service._retry_attempts == 3
        assert service._retry_delay == 1.0
    
    async def test_service_initialization(self, test_db_config):
        """Test service initialization process."""
        service = DatabaseService(config=test_db_config)
        
        await service.initialize()
        
        assert service._initialized is True
        
        # Test re-initialization (should not fail)
        await service.initialize()
        assert service._initialized is True
        
        await service.close()
    
    async def test_initialization_with_retry_success(self, test_db_config):
        """Test initialization succeeds after retry."""
        mock_session_manager = Mock(spec=DatabaseSessionManager)
        mock_session_manager.initialize = AsyncMock(
            side_effect=[Exception("First failure"), None]
        )
        
        service = DatabaseService(
            config=test_db_config, 
            session_manager=mock_session_manager
        )
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await service.initialize()
        
        assert service._initialized is True
        assert mock_session_manager.initialize.call_count == 2
    
    async def test_initialization_failure_after_retries(self, test_db_config):
        """Test initialization fails after all retries."""
        mock_session_manager = Mock(spec=DatabaseSessionManager)
        mock_session_manager.initialize = AsyncMock(
            side_effect=Exception("Persistent failure")
        )
        
        service = DatabaseService(
            config=test_db_config,
            session_manager=mock_session_manager
        )
        service._retry_attempts = 2
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(DatabaseServiceError) as exc_info:
                await service.initialize()
            
            assert "initialization failed after all retry attempts" in str(exc_info.value)
    
    async def test_database_connection_test(self, db_service):
        """Test database connection validation."""
        # Connection test is performed during initialization
        assert db_service._initialized is True
        
        # Manual connection test
        await db_service._test_database_connection()
        
        # Test with invalid session manager
        db_service.session_manager = None
        with pytest.raises(DatabaseServiceError):
            await db_service._test_database_connection()


class TestDatabaseSessionManagement:
    """Test database session management and connection pooling."""
    
    async def test_get_session_basic(self, db_service):
        """Test basic session retrieval."""
        async with db_service.get_session() as session:
            assert session is not None
            
            # Test basic query
            result = await session.execute("SELECT 1 as test")
            assert result.scalar() == 1
    
    async def test_get_session_concurrent(self, db_service):
        """Test concurrent session access."""
        async def session_task():
            async with db_service.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar()
        
        # Run 10 concurrent session tasks
        tasks = [session_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert all(result == 1 for result in results)
    
    async def test_session_with_transaction(self, db_service):
        """Test session with transaction management."""
        async with db_service.get_session_with_transaction() as session:
            assert session is not None
            
            # Create a user within transaction
            user = User(
                email='transaction@test.com',
                username='transactionuser',
                hashed_password='hashed_password'
            )
            session.add(user)
            await session.flush()
            
            # User should have an ID now
            assert user.id is not None
    
    async def test_session_retry_on_disconnection(self, db_service):
        """Test session retry logic on connection issues."""
        from sqlalchemy.exc import DisconnectionError
        
        original_get_session = db_service.session_manager.get_session
        call_count = 0
        
        async def mock_get_session():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise DisconnectionError("Connection lost", None, None)
            else:
                return original_get_session()
        
        with patch.object(db_service.session_manager, 'get_session', mock_get_session), \
             patch.object(db_service.session_manager, 'close', new_callable=AsyncMock), \
             patch.object(db_service.session_manager, 'initialize', new_callable=AsyncMock), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            async with db_service.get_session() as session:
                result = await session.execute("SELECT 1")
                assert result.scalar() == 1
        
        assert call_count >= 2


class TestRepositoryIntegration:
    """Test repository pattern integration with database service."""
    
    async def test_get_repositories(self, db_service):
        """Test repository factory access."""
        async with db_service.get_repositories() as repos:
            assert isinstance(repos, RepositoryFactory)
            
            # Test accessing all repository types
            user_repo = repos.get_user_repository()
            project_repo = repos.get_project_repository()
            task_repo = repos.get_task_repository()
            audit_repo = repos.get_audit_repository()
            session_repo = repos.get_session_repository()
            
            assert user_repo is not None
            assert project_repo is not None
            assert task_repo is not None
            assert audit_repo is not None
            assert session_repo is not None
    
    async def test_repository_operations(self, db_service, sample_user_data):
        """Test basic repository operations."""
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            
            # Test create
            user = await user_repo.create_user(**sample_user_data)
            assert user is not None
            assert user.email == sample_user_data['email']
            assert user.username == sample_user_data['username']
            
            # Test get by ID
            retrieved_user = await user_repo.get_by_id(user.id)
            assert retrieved_user is not None
            assert retrieved_user.id == user.id
            
            # Test get by email
            email_user = await user_repo.get_by_email(sample_user_data['email'])
            assert email_user is not None
            assert email_user.id == user.id
            
            # Test count
            count = await user_repo.count()
            assert count >= 1
    
    async def test_repository_relationships(self, db_service, sample_user_data, sample_project_data):
        """Test repository operations with relationships."""
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            project_repo = repos.get_project_repository()
            task_repo = repos.get_task_repository()
            
            # Create user
            user = await user_repo.create_user(**sample_user_data)
            
            # Create project
            project = await project_repo.create_project(
                owner_id=user.id,
                **sample_project_data
            )
            assert project is not None
            assert project.owner_id == user.id
            
            # Create task
            task = await task_repo.create(
                title="Test Task",
                description="A test task",
                project_id=project.id,
                assigned_to=user.id
            )
            assert task is not None
            assert task.project_id == project.id
            assert task.assigned_to == user.id


class TestTransactionManagement:
    """Test transaction management and rollback functionality."""
    
    async def test_execute_in_transaction_success(self, db_service, sample_user_data):
        """Test successful transaction execution."""
        async def create_user_operation(repos):
            user_repo = repos.get_user_repository()
            return await user_repo.create_user(**sample_user_data)
        
        result = await db_service.execute_in_transaction(create_user_operation)
        
        assert result is not None
        assert result.email == sample_user_data['email']
        
        # Verify user was actually created
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            user = await user_repo.get_by_email(sample_user_data['email'])
            assert user is not None
    
    async def test_execute_in_transaction_rollback(self, db_service, sample_user_data):
        """Test transaction rollback on failure."""
        async def failing_operation(repos):
            user_repo = repos.get_user_repository()
            # Create user
            await user_repo.create_user(**sample_user_data)
            # Then fail
            raise Exception("Simulated failure")
        
        with pytest.raises(DatabaseServiceError):
            await db_service.execute_in_transaction(failing_operation)
        
        # Verify rollback - user should not exist
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            user = await user_repo.get_by_email(sample_user_data['email'])
            assert user is None
    
    async def test_complex_transaction_rollback(self, db_service, sample_user_data, sample_project_data):
        """Test rollback of complex multi-table transaction."""
        async def complex_operation(repos):
            user_repo = repos.get_user_repository()
            project_repo = repos.get_project_repository()
            task_repo = repos.get_task_repository()
            
            # Create user
            user = await user_repo.create_user(**sample_user_data)
            
            # Create project
            project = await project_repo.create_project(
                owner_id=user.id,
                **sample_project_data
            )
            
            # Create task
            await task_repo.create(
                title="Test Task",
                project_id=project.id,
                assigned_to=user.id
            )
            
            # Force failure after all operations
            raise Exception("Complex operation failure")
        
        with pytest.raises(DatabaseServiceError):
            await db_service.execute_in_transaction(complex_operation)
        
        # Verify complete rollback
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            project_repo = repos.get_project_repository()
            task_repo = repos.get_task_repository()
            
            user = await user_repo.get_by_email(sample_user_data['email'])
            assert user is None
            
            projects = await project_repo.get_all()
            assert len(projects) == 0
            
            tasks = await task_repo.get_all()
            assert len(tasks) == 0
    
    async def test_transaction_retry_logic(self, db_service, sample_user_data):
        """Test transaction retry on temporary failures."""
        call_count = 0
        
        async def flaky_operation(repos):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise Exception("Temporary failure")
            
            user_repo = repos.get_user_repository()
            return await user_repo.create_user(**sample_user_data)
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await db_service.execute_in_transaction(flaky_operation)
        
        assert result is not None
        assert result.email == sample_user_data['email']
        assert call_count == 2


class TestDatabaseMigrations:
    """Test database migration management."""
    
    async def test_get_migration_status(self, db_service):
        """Test migration status retrieval."""
        status = await db_service.get_migration_status()
        
        # In test environment, likely no alembic config
        assert 'status' in status
        assert status['status'] in ['configuration_missing', 'error', 'current_revision']
    
    @patch('src.database.service.command')
    @patch('src.database.service.Config')
    async def test_run_migrations(self, mock_config_class, mock_command, db_service):
        """Test running migrations."""
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        with patch('os.path.exists', return_value=True):
            result = await db_service.run_migrations("head")
        
        assert result is True
        mock_command.upgrade.assert_called_once_with(mock_config, "head")
    
    @patch('src.database.service.command')
    @patch('src.database.service.Config')
    async def test_rollback_migrations(self, mock_config_class, mock_command, db_service):
        """Test rolling back migrations."""
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        with patch('os.path.exists', return_value=True):
            result = await db_service.rollback_migration("abc123")
        
        assert result is True
        mock_command.downgrade.assert_called_once_with(mock_config, "abc123")
    
    async def test_migration_errors(self, db_service):
        """Test migration error handling."""
        # Test missing config
        with patch('os.path.exists', return_value=False):
            with pytest.raises(DatabaseServiceError) as exc_info:
                await db_service.run_migrations()
            assert "Alembic configuration not found" in str(exc_info.value)
        
        # Test import error
        with patch('src.database.service.command', side_effect=ImportError):
            with pytest.raises(DatabaseServiceError) as exc_info:
                await db_service.run_migrations()
            assert "Alembic not available" in str(exc_info.value)


class TestDatabaseBackupRestore:
    """Test database backup and restore functionality."""
    
    async def test_sqlite_backup(self, db_service):
        """Test SQLite database backup."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            backup_path = tmp_file.name
        
        try:
            with patch('shutil.copy2') as mock_copy:
                result = await db_service.backup_database(backup_path)
                assert result is True
                mock_copy.assert_called_once()
        finally:
            if os.path.exists(backup_path):
                os.unlink(backup_path)
    
    @patch('subprocess.run')
    async def test_postgresql_backup(self, mock_run, test_db_config):
        """Test PostgreSQL database backup."""
        # Modify config for PostgreSQL
        test_db_config.database_url = "postgresql+asyncpg://user:pass@localhost/testdb"
        
        service = DatabaseService(config=test_db_config)
        await service.initialize()
        
        # Mock successful pg_dump
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        try:
            result = await service.backup_database("/tmp/test_backup.sql")
            assert result is True
            mock_run.assert_called_once()
        finally:
            await service.close()
    
    @patch('subprocess.run')
    async def test_backup_failure(self, mock_run, test_db_config):
        """Test backup failure handling."""
        test_db_config.database_url = "postgresql+asyncpg://user:pass@localhost/testdb"
        
        service = DatabaseService(config=test_db_config)
        await service.initialize()
        
        # Mock failed pg_dump
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "Backup failed"
        mock_run.return_value = mock_process
        
        try:
            with pytest.raises(DatabaseServiceError) as exc_info:
                await service.backup_database("/tmp/test_backup.sql")
            assert "Backup failed" in str(exc_info.value)
        finally:
            await service.close()
    
    async def test_unsupported_database_backup(self, test_db_config):
        """Test backup error for unsupported database."""
        test_db_config.database_url = "oracle://user:pass@localhost/db"
        
        service = DatabaseService(config=test_db_config)
        await service.initialize()
        
        try:
            with pytest.raises(DatabaseServiceError) as exc_info:
                await service.backup_database("/tmp/backup.sql")
            assert "Backup not supported" in str(exc_info.value)
        finally:
            await service.close()


class TestDatabaseHealthChecks:
    """Test database health monitoring and diagnostics."""
    
    async def test_database_info(self, db_service):
        """Test database information retrieval."""
        info = await db_service.get_database_info()
        
        expected_keys = [
            'database_url', 'database_type', 'session_manager_health',
            'connection_pool', 'query_response_time_ms', 'service_initialized',
            'migration_status', 'config'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['database_type'] == 'sqlite'
        assert info['service_initialized'] is True
        assert isinstance(info['query_response_time_ms'], float)
        assert info['query_response_time_ms'] >= 0
    
    async def test_comprehensive_health_check(self, db_service):
        """Test comprehensive health check."""
        health = await db_service.health_check()
        
        expected_keys = [
            'service', 'status', 'initialized', 'database_info',
            'connection_pool_test', 'repository_health', 'timestamp'
        ]
        
        for key in expected_keys:
            assert key in health
        
        assert health['service'] == 'DatabaseService'
        assert health['initialized'] is True
        
        # Check repository health
        repo_health = health['repository_health']
        expected_repos = ['user', 'project', 'task', 'audit', 'session']
        
        for repo_name in expected_repos:
            assert repo_name in repo_health
            assert repo_health[repo_name] == 'healthy'
    
    async def test_connection_pool_health(self, db_service):
        """Test connection pool health testing."""
        pool_test = await db_service._test_connection_pool()
        
        expected_keys = [
            'total_tests', 'successful_connections', 'failed_connections',
            'average_connection_time', 'max_connection_time', 'pool_status',
            'test_passed'
        ]
        
        for key in expected_keys:
            assert key in pool_test
        
        assert pool_test['total_tests'] == 5
        assert pool_test['successful_connections'] >= 0
        assert pool_test['failed_connections'] >= 0
        assert pool_test['test_passed'] in [True, False]
        
        if pool_test['successful_connections'] > 0:
            assert isinstance(pool_test['average_connection_time'], float)
            assert isinstance(pool_test['max_connection_time'], float)
    
    async def test_health_check_with_failures(self, test_db_config):
        """Test health check when components are unhealthy."""
        service = DatabaseService(config=test_db_config)
        # Don't initialize service - should cause health check issues
        
        health = await service.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['initialized'] is False


class TestDatabaseMaintenance:
    """Test database maintenance and cleanup operations."""
    
    async def test_maintenance_tasks_execution(self, db_service):
        """Test maintenance task execution."""
        # Mock the repository operations
        with patch.object(db_service, 'get_repositories') as mock_repos:
            # Mock session cleanup
            mock_session_repo = AsyncMock()
            mock_session_repo.cleanup_expired_sessions = AsyncMock(return_value=5)
            
            # Mock audit cleanup
            mock_audit_repo = AsyncMock()
            mock_audit_repo.cleanup_old_records = AsyncMock(return_value=10)
            
            # Mock repository factory
            mock_repo_factory = AsyncMock()
            mock_repo_factory.get_session_repository = MagicMock(return_value=mock_session_repo)
            mock_repo_factory.get_audit_repository = MagicMock(return_value=mock_audit_repo)
            
            mock_repos.return_value.__aenter__ = AsyncMock(return_value=mock_repo_factory)
            mock_repos.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Execute maintenance tasks
            results = await db_service.execute_maintenance_tasks()
            
            expected_keys = [
                'started_at', 'completed_at', 'duration_seconds',
                'tasks', 'summary'
            ]
            
            for key in expected_keys:
                assert key in results
            
            # Check task results
            tasks = results['tasks']
            assert 'session_cleanup' in tasks
            assert 'audit_cleanup' in tasks
            
            assert tasks['session_cleanup']['status'] == 'completed'
            assert tasks['session_cleanup']['cleaned_count'] == 5
            
            assert tasks['audit_cleanup']['status'] == 'completed'
            assert tasks['audit_cleanup']['cleaned_count'] == 10
            
            # Check summary
            summary = results['summary']
            assert summary['total_tasks'] >= 2
            assert summary['successful_tasks'] >= 2
            assert summary['success_rate'] > 0
    
    async def test_session_cleanup(self, db_service):
        """Test expired session cleanup."""
        # Mock repository operations
        with patch.object(db_service, 'get_repositories') as mock_repos:
            mock_session_repo = AsyncMock()
            mock_session_repo.cleanup_expired_sessions = AsyncMock(return_value=3)
            
            mock_repo_factory = AsyncMock()
            mock_repo_factory.get_session_repository = MagicMock(return_value=mock_session_repo)
            
            mock_repos.return_value.__aenter__ = AsyncMock(return_value=mock_repo_factory)
            mock_repos.return_value.__aexit__ = AsyncMock(return_value=None)
            
            cleaned_count = await db_service.cleanup_expired_sessions()
            assert cleaned_count == 3
    
    async def test_audit_log_cleanup(self, db_service):
        """Test old audit log cleanup."""
        # Mock repository operations
        with patch.object(db_service, 'get_repositories') as mock_repos:
            mock_audit_repo = AsyncMock()
            mock_audit_repo.cleanup_old_records = AsyncMock(return_value=7)
            
            mock_repo_factory = AsyncMock()
            mock_repo_factory.get_audit_repository = MagicMock(return_value=mock_audit_repo)
            
            mock_repos.return_value.__aenter__ = AsyncMock(return_value=mock_repo_factory)
            mock_repos.return_value.__aexit__ = AsyncMock(return_value=None)
            
            cleaned_count = await db_service.cleanup_old_audit_logs(90)
            assert cleaned_count == 7


class TestDatabaseServiceFactory:
    """Test database service factory functions and configuration."""
    
    async def test_create_database_service(self):
        """Test database service creation with custom configuration."""
        from src.database.service import create_database_service
        
        service = await create_database_service(
            database_url="sqlite+aiosqlite:///:memory:",
            pool_size=15,
            max_overflow=5,
            pool_timeout=45
        )
        
        try:
            assert service._initialized is True
            assert service.config.pool_size == 15
            assert service.config.max_overflow == 5
            assert service.config.pool_timeout == 45
        finally:
            await service.close()
    
    @patch('src.database.service._database_service', None)
    async def test_get_database_service_singleton(self):
        """Test singleton database service getter."""
        # First call creates instance
        service1 = await get_database_service()
        
        # Second call returns same instance
        service2 = await get_database_service()
        
        assert service1 is service2
        
        try:
            await service1.close()
        except:
            pass
    
    async def test_database_service_context_manager(self, test_db_config):
        """Test database service as context manager."""
        async with DatabaseService(config=test_db_config) as service:
            assert service._initialized is True
            
            # Test using the service
            async with service.get_session() as session:
                result = await session.execute("SELECT 1")
                assert result.scalar() == 1
        
        # Service should be closed after context
        assert service._initialized is False


@pytest.mark.performance
class TestDatabaseServicePerformance:
    """Performance and load testing for database service."""
    
    async def test_concurrent_session_performance(self, db_service):
        """Test performance under concurrent session load."""
        async def session_task():
            async with db_service.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar()
        
        # Run 50 concurrent sessions
        tasks = [session_task() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert all(result == 1 for result in results)
        assert len(results) == 50
    
    async def test_concurrent_transaction_performance(self, db_service):
        """Test performance under concurrent transaction load."""
        async def transaction_task(i):
            async def operation(repos):
                user_repo = repos.get_user_repository()
                return await user_repo.create(
                    email=f"perf{i}@example.com",
                    username=f"perfuser{i}",
                    hashed_password="hashed"
                )
            
            return await db_service.execute_in_transaction(operation)
        
        # Run 25 concurrent transactions
        tasks = [transaction_task(i) for i in range(25)]
        users = await asyncio.gather(*tasks)
        
        assert len(users) == 25
        assert all(user is not None for user in users)
        
        # Verify all users are unique
        emails = set(user.email for user in users)
        assert len(emails) == 25
    
    async def test_bulk_operation_performance(self, db_service):
        """Test bulk operation performance."""
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            
            # Prepare bulk user data
            users_data = [
                {
                    'email': f'bulk{i}@example.com',
                    'username': f'bulkuser{i}',
                    'hashed_password': 'hashed'
                }
                for i in range(100)
            ]
            
            # Bulk create
            users = await user_repo.bulk_create(users_data)
            assert len(users) == 100
            
            # Verify count
            count = await user_repo.count()
            assert count >= 100
            
            # Test bulk retrieval with filters
            filtered_users = await user_repo.get_all(
                filters={'username__like': 'bulk'},
                limit=150
            )
            assert len(filtered_users) >= 100


class TestDatabaseErrorHandling:
    """Test comprehensive error handling and recovery."""
    
    async def test_service_initialization_errors(self):
        """Test various service initialization error scenarios."""
        # Invalid database URL
        bad_config = DatabaseConfig("invalid://url/path")
        service = DatabaseService(config=bad_config)
        
        with pytest.raises(DatabaseServiceError):
            await service.initialize()
    
    async def test_session_error_handling(self, db_service):
        """Test session error handling and recovery."""
        # Test session error propagation
        with pytest.raises(Exception):
            async with db_service.get_session() as session:
                # Force an invalid query
                await session.execute("INVALID SQL QUERY")
    
    async def test_transaction_error_handling(self, db_service):
        """Test transaction error handling and rollback."""
        async def failing_operation(repos):
            user_repo = repos.get_user_repository()
            
            # Create user successfully
            user = await user_repo.create(
                email="error@test.com",
                username="erroruser",
                hashed_password="hashed"
            )
            
            # Force an error
            raise ValueError("Transaction should rollback")
        
        with pytest.raises(DatabaseServiceError):
            await db_service.execute_in_transaction(failing_operation)
        
        # Verify rollback occurred
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            user = await user_repo.get_by_email("error@test.com")
            assert user is None
    
    async def test_repository_error_handling(self, db_service):
        """Test repository-level error handling."""
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            
            # Test duplicate email error
            user_data = {
                'email': 'duplicate@test.com',
                'username': 'user1',
                'password': 'TestPass123!'
            }
            
            # Create first user
            user1 = await user_repo.create_user(**user_data)
            assert user1 is not None
            
            # Try to create duplicate - should handle gracefully
            from src.database.repositories.base import RepositoryError
            with pytest.raises(RepositoryError):
                user_data['username'] = 'user2'  # Different username, same email
                await user_repo.create_user(**user_data)


class TestDatabaseServiceIntegration:
    """End-to-end integration tests."""
    
    async def test_complete_user_workflow(self, db_service, sample_user_data):
        """Test complete user management workflow."""
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            session_repo = repos.get_session_repository()
            audit_repo = repos.get_audit_repository()
            
            # 1. Create user
            user = await user_repo.create_user(**sample_user_data)
            assert user is not None
            
            # 2. Authenticate user
            auth_user = await user_repo.authenticate_user(
                sample_user_data['email'],
                sample_user_data['password']
            )
            assert auth_user is not None
            assert auth_user.id == user.id
            
            # 3. Create session
            session = await session_repo.create_session(
                user_id=user.id,
                session_token="test-session-123",
                ip_address="127.0.0.1"
            )
            assert session is not None
            
            # 4. Log audit event
            audit_log = await audit_repo.log_action(
                user_id=user.id,
                action="user_login",
                resource_type="user",
                resource_id=str(user.id),
                result="success"
            )
            assert audit_log is not None
            
            # 5. Verify session is active
            active_session = await session_repo.get_active_session("test-session-123")
            assert active_session is not None
            
            # 6. Get user activity
            activity = await audit_repo.get_user_activity(user.id, days=1)
            assert len(activity) >= 1
            
            # 7. Invalidate session
            result = await session_repo.invalidate_session(session.id)
            assert result is True
    
    async def test_project_task_workflow(self, db_service, sample_user_data, sample_project_data):
        """Test project and task management workflow."""
        async def workflow_operation(repos):
            user_repo = repos.get_user_repository()
            project_repo = repos.get_project_repository()
            task_repo = repos.get_task_repository()
            audit_repo = repos.get_audit_repository()
            
            # Create user
            user = await user_repo.create_user(**sample_user_data)
            
            # Create project
            project = await project_repo.create_project(
                owner_id=user.id,
                **sample_project_data
            )
            
            # Create multiple tasks
            tasks = []
            for i in range(3):
                task = await task_repo.create(
                    title=f"Task {i+1}",
                    description=f"Description for task {i+1}",
                    project_id=project.id,
                    assigned_to=user.id
                )
                tasks.append(task)
            
            # Log project creation
            await audit_repo.log_action(
                user_id=user.id,
                action="project_created",
                resource_type="project",
                resource_id=str(project.id),
                result="success"
            )
            
            return {
                'user': user,
                'project': project,
                'tasks': tasks
            }
        
        # Execute entire workflow in transaction
        result = await db_service.execute_in_transaction(workflow_operation)
        
        assert result['user'] is not None
        assert result['project'] is not None
        assert len(result['tasks']) == 3
        
        # Verify data persisted correctly
        async with db_service.get_repositories() as repos:
            project_repo = repos.get_project_repository()
            task_repo = repos.get_task_repository()
            
            # Check project exists
            project = await project_repo.get_by_id(result['project'].id)
            assert project is not None
            
            # Check tasks exist
            project_tasks = await task_repo.get_project_tasks(project.id)
            assert len(project_tasks) == 3
    
    async def test_statistics_and_monitoring(self, db_service):
        """Test database statistics and monitoring integration."""
        # Create some test data first
        async with db_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            
            # Create a few users for statistics
            for i in range(5):
                await user_repo.create_user(
                    email=f"stats{i}@example.com",
                    username=f"statsuser{i}",
                    password="TestPass123!"
                )
        
        # Get database statistics
        stats = await db_service.get_database_statistics()
        
        expected_keys = [
            'total_users', 'active_users', 'total_projects',
            'public_projects', 'total_tasks', 'completed_tasks',
            'active_sessions', 'total_audit_records', 'generated_at',
            'performance'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_users'] >= 5
        assert isinstance(stats['performance'], dict)
        
        # Test health check includes statistics
        health = await db_service.health_check()
        assert 'database_info' in health
        assert 'repository_health' in health