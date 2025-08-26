"""
Tests for Database Service Integration

Comprehensive tests for:
- Service initialization and configuration
- Session and transaction management
- Repository integration
- Health checks and monitoring
- Error handling and recovery
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.database_service import DatabaseService, DatabaseServiceError
from src.database.session import DatabaseConfig
from src.database.repositories import RepositoryFactory


@pytest_asyncio.fixture
async def database_config():
    """Test database configuration."""
    return DatabaseConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        pool_size=5,
        max_overflow=10
    )


@pytest_asyncio.fixture
async def database_service(database_config):
    """Test database service."""
    service = DatabaseService(config=database_config)
    await service.initialize()
    yield service
    await service.close()


class TestDatabaseService:
    """Test database service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, database_config):
        """Test database service initialization."""
        service = DatabaseService(config=database_config)
        
        assert service._initialized is False
        await service.initialize()
        assert service._initialized is True
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_session_management(self, database_service):
        """Test database session management."""
        async with database_service.get_session() as session:
            assert session is not None
            # Test simple query
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_transaction_management(self, database_service):
        """Test transaction management."""
        async with database_service.get_session_with_transaction() as session:
            assert session is not None
            assert session.in_transaction() is True
            
            # Test transaction works
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_repository_factory_integration(self, database_service):
        """Test repository factory integration."""
        async with database_service.get_repositories() as repos:
            assert isinstance(repos, RepositoryFactory)
            
            # Test repository access
            user_repo = repos.get_user_repository()
            assert user_repo is not None
            
            project_repo = repos.get_project_repository()
            assert project_repo is not None
    
    @pytest.mark.asyncio
    async def test_execute_in_transaction(self, database_service):
        """Test transaction execution wrapper."""
        async def test_operation(repositories):
            """Test operation to execute in transaction."""
            user_repo = repositories.get_user_repository()
            return await user_repo.count()
        
        result = await database_service.execute_in_transaction(test_operation)
        assert isinstance(result, int)
        assert result >= 0
    
    @pytest.mark.asyncio
    async def test_database_info(self, database_service):
        """Test database information retrieval."""
        info = await database_service.get_database_info()
        
        assert 'database_url' in info
        assert 'database_type' in info
        assert 'session_manager_health' in info
        assert 'connection_pool' in info
        assert 'service_initialized' in info
        assert info['service_initialized'] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, database_service):
        """Test comprehensive health check."""
        health = await database_service.health_check()
        
        assert 'service' in health
        assert 'status' in health
        assert 'database_info' in health
        assert 'repository_health' in health
        assert health['service'] == 'DatabaseService'
    
    @pytest.mark.asyncio
    async def test_database_statistics(self, database_service):
        """Test database statistics collection."""
        stats = await database_service.get_database_statistics()
        
        assert 'total_users' in stats
        assert 'total_projects' in stats
        assert 'total_tasks' in stats
        assert 'generated_at' in stats
        assert isinstance(stats['total_users'], int)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, database_config):
        """Test error handling in database service."""
        # Test with invalid database URL
        bad_config = DatabaseConfig("invalid://url")
        service = DatabaseService(config=bad_config)
        
        with pytest.raises(DatabaseServiceError):
            await service.initialize()


class TestDatabaseServiceContext:
    """Test database service as context manager."""
    
    @pytest.mark.asyncio
    async def test_context_manager(self, database_config):
        """Test database service as async context manager."""
        async with DatabaseService(config=database_config) as service:
            assert service._initialized is True
            
            # Test functionality within context
            async with service.get_session() as session:
                result = await session.execute("SELECT 1")
                assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_context_manager_error(self, database_config):
        """Test context manager with error."""
        with pytest.raises(Exception):
            async with DatabaseService(config=database_config) as service:
                # Force an error
                raise Exception("Test error")


class TestDatabaseServiceUtilities:
    """Test database service utility functions."""
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, database_service):
        """Test expired session cleanup."""
        cleaned_count = await database_service.cleanup_expired_sessions()
        assert isinstance(cleaned_count, int)
        assert cleaned_count >= 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_audit_logs(self, database_service):
        """Test old audit log cleanup."""
        cleaned_count = await database_service.cleanup_old_audit_logs(days_to_keep=30)
        assert isinstance(cleaned_count, int)
        assert cleaned_count >= 0


@pytest.mark.asyncio
async def test_convenience_functions():
    """Test convenience functions."""
    from src.services.database_service import create_database_service
    
    service = await create_database_service(
        database_url="sqlite+aiosqlite:///:memory:"
    )
    
    assert service._initialized is True
    assert isinstance(service, DatabaseService)
    
    await service.close()


class TestDatabaseServiceWithRealData:
    """Test database service with real data operations."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, database_service):
        """Test complete workflow with database service."""
        # Create user through transaction
        async def create_user(repositories):
            user_repo = repositories.get_user_repository()
            return await user_repo.create_user(
                email="workflow@test.com",
                username="workflowuser",
                password="TestPass123!",
                full_name="Workflow User"
            )
        
        user = await database_service.execute_in_transaction(create_user)
        assert user is not None
        assert user.email == "workflow@test.com"
        
        # Create project through transaction
        async def create_project(repositories):
            project_repo = repositories.get_project_repository()
            return await project_repo.create_project(
                owner_id=user.id,
                name="Workflow Project",
                description="Test workflow project"
            )
        
        project = await database_service.execute_in_transaction(create_project)
        assert project is not None
        assert project.name == "Workflow Project"
        
        # Create task through transaction
        async def create_task(repositories):
            task_repo = repositories.get_task_repository()
            return await task_repo.create_task(
                project_id=project.id,
                title="Workflow Task",
                description="Test workflow task",
                assigned_to=user.id,
                creator_user_id=user.id
            )
        
        task = await database_service.execute_in_transaction(create_task)
        assert task is not None
        assert task.title == "Workflow Task"
        
        # Log audit event
        async def log_audit(repositories):
            audit_repo = repositories.get_audit_repository()
            return await audit_repo.log_action(
                user_id=user.id,
                action="workflow_test",
                resource_type="task",
                resource_id=str(task.id),
                result="success"
            )
        
        audit_log = await database_service.execute_in_transaction(log_audit)
        assert audit_log is not None
        assert audit_log.action == "workflow_test"
        
        # Verify statistics include our data
        stats = await database_service.get_database_statistics()
        assert stats['total_users'] >= 1
        assert stats['total_projects'] >= 1
        assert stats['total_tasks'] >= 1
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, database_service):
        """Test transaction rollback on error."""
        async def failing_operation(repositories):
            user_repo = repositories.get_user_repository()
            # Create user
            await user_repo.create_user(
                email="rollback@test.com",
                username="rollbackuser",
                password="TestPass123!"
            )
            # Force an error
            raise Exception("Forced error for rollback test")
        
        # Operation should fail and rollback
        with pytest.raises(DatabaseServiceError):
            await database_service.execute_in_transaction(failing_operation)
        
        # Verify user was not created (rolled back)
        async with database_service.get_repositories() as repos:
            user_repo = repos.get_user_repository()
            user = await user_repo.get_by_email("rollback@test.com")
            assert user is None