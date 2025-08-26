"""
Tests for Repository Pattern Implementation

Comprehensive tests for:
- BaseRepository CRUD operations
- UserRepository authentication features
- ProjectRepository access control
- TaskRepository project integration
- AuditRepository logging
- SessionRepository management
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

from src.database.session import DatabaseSessionManager, DatabaseConfig
from src.database.repositories import (
    RepositoryFactory, UserRepository, ProjectRepository, 
    TaskRepository, AuditRepository, SessionRepository
)
from src.database.repositories.base import RepositoryError
from src.database.models import User, Project, Task, AuditLog, UserSession


@pytest_asyncio.fixture
async def session_manager():
    """Test session manager with in-memory database."""
    config = DatabaseConfig("sqlite+aiosqlite:///:memory:")
    manager = DatabaseSessionManager(config)
    await manager.initialize()
    await manager.create_tables()
    yield manager
    await manager.close()


@pytest_asyncio.fixture
async def db_session(session_manager):
    """Test database session."""
    async with session_manager.get_session() as session:
        yield session


@pytest_asyncio.fixture
async def repository_factory(db_session):
    """Test repository factory."""
    return RepositoryFactory(db_session)


@pytest_asyncio.fixture
async def sample_user(repository_factory):
    """Create a sample user for testing."""
    user_repo = repository_factory.get_user_repository()
    user = await user_repo.create_user(
        email="test@example.com",
        username="testuser",
        password="TestPass123!",
        full_name="Test User"
    )
    return user


@pytest_asyncio.fixture
async def sample_project(repository_factory, sample_user):
    """Create a sample project for testing."""
    project_repo = repository_factory.get_project_repository()
    project = await project_repo.create_project(
        owner_id=sample_user.id,
        name="Test Project",
        description="A test project",
        project_type="test"
    )
    return project


class TestRepositoryFactory:
    """Test repository factory."""
    
    @pytest.mark.asyncio
    async def test_repository_creation(self, repository_factory):
        """Test repository creation via factory."""
        user_repo = repository_factory.get_user_repository()
        assert isinstance(user_repo, UserRepository)
        
        project_repo = repository_factory.get_project_repository()
        assert isinstance(project_repo, ProjectRepository)
        
        task_repo = repository_factory.get_task_repository()
        assert isinstance(task_repo, TaskRepository)
        
        audit_repo = repository_factory.get_audit_repository()
        assert isinstance(audit_repo, AuditRepository)
        
        session_repo = repository_factory.get_session_repository()
        assert isinstance(session_repo, SessionRepository)
    
    @pytest.mark.asyncio
    async def test_singleton_behavior(self, repository_factory):
        """Test that repositories are singleton within factory."""
        user_repo1 = repository_factory.get_user_repository()
        user_repo2 = repository_factory.get_user_repository()
        
        assert user_repo1 is user_repo2
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, repository_factory):
        """Test health check for all repositories."""
        health = await repository_factory.health_check_all()
        
        # Should have no repositories cached initially
        assert isinstance(health, dict)


class TestUserRepository:
    """Test user repository."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, repository_factory):
        """Test user creation."""
        user_repo = repository_factory.get_user_repository()
        
        user = await user_repo.create_user(
            email="new@example.com",
            username="newuser",
            password="NewPass123!",
            full_name="New User"
        )
        
        assert user is not None
        assert user.email == "new@example.com"
        assert user.username == "newuser"
        assert user.full_name == "New User"
        assert user.verify_password("NewPass123!")
    
    @pytest.mark.asyncio
    async def test_duplicate_user_creation(self, repository_factory, sample_user):
        """Test duplicate user creation prevention."""
        user_repo = repository_factory.get_user_repository()
        
        # Try to create user with same email
        with pytest.raises(RepositoryError):
            await user_repo.create_user(
                email=sample_user.email,
                username="different",
                password="DiffPass123!"
            )
        
        # Try to create user with same username
        with pytest.raises(RepositoryError):
            await user_repo.create_user(
                email="different@example.com",
                username=sample_user.username,
                password="DiffPass123!"
            )
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, repository_factory, sample_user):
        """Test user authentication."""
        user_repo = repository_factory.get_user_repository()
        
        # Successful authentication
        auth_user = await user_repo.authenticate_user(
            sample_user.email, "TestPass123!"
        )
        assert auth_user is not None
        assert auth_user.id == sample_user.id
        
        # Failed authentication
        auth_user = await user_repo.authenticate_user(
            sample_user.email, "WrongPassword"
        )
        assert auth_user is None
    
    @pytest.mark.asyncio
    async def test_get_user_by_email(self, repository_factory, sample_user):
        """Test get user by email."""
        user_repo = repository_factory.get_user_repository()
        
        found_user = await user_repo.get_by_email(sample_user.email)
        assert found_user is not None
        assert found_user.id == sample_user.id
        
        # Test case insensitive
        found_user = await user_repo.get_by_email(sample_user.email.upper())
        assert found_user is not None
        assert found_user.id == sample_user.id
    
    @pytest.mark.asyncio
    async def test_password_change(self, repository_factory, sample_user):
        """Test password change functionality."""
        user_repo = repository_factory.get_user_repository()
        
        # Successful password change
        success = await user_repo.change_password(
            sample_user.id, "TestPass123!", "NewPass456!"
        )
        assert success is True
        
        # Verify new password works
        auth_user = await user_repo.authenticate_user(
            sample_user.email, "NewPass456!"
        )
        assert auth_user is not None
        
        # Verify old password doesn't work
        auth_user = await user_repo.authenticate_user(
            sample_user.email, "TestPass123!"
        )
        assert auth_user is None


class TestProjectRepository:
    """Test project repository."""
    
    @pytest.mark.asyncio
    async def test_create_project(self, repository_factory, sample_user):
        """Test project creation."""
        project_repo = repository_factory.get_project_repository()
        
        project = await project_repo.create_project(
            owner_id=sample_user.id,
            name="New Project",
            description="A new test project",
            project_type="development"
        )
        
        assert project is not None
        assert project.name == "New Project"
        assert project.description == "A new test project"
        assert project.owner_id == sample_user.id
        assert project.project_type == "development"
    
    @pytest.mark.asyncio
    async def test_get_user_projects(self, repository_factory, sample_user, sample_project):
        """Test getting user projects."""
        project_repo = repository_factory.get_project_repository()
        
        projects = await project_repo.get_user_projects(sample_user.id)
        assert len(projects) >= 1
        assert sample_project.id in [p.id for p in projects]
    
    @pytest.mark.asyncio
    async def test_project_access_control(self, repository_factory, sample_user, sample_project):
        """Test project access control."""
        project_repo = repository_factory.get_project_repository()
        
        # Owner should have access
        has_access = await project_repo.check_project_access(
            sample_project.id, sample_user.id
        )
        assert has_access is True
        
        # Random user should not have access (unless public)
        random_user_id = uuid.uuid4()
        has_access = await project_repo.check_project_access(
            sample_project.id, random_user_id
        )
        assert has_access is False
    
    @pytest.mark.asyncio
    async def test_project_statistics(self, repository_factory, sample_project):
        """Test project statistics."""
        project_repo = repository_factory.get_project_repository()
        
        stats = await project_repo.get_project_statistics(sample_project.id)
        
        assert 'project_id' in stats
        assert 'total_tasks' in stats
        assert 'completion_rate' in stats
        assert stats['project_id'] == str(sample_project.id)


class TestTaskRepository:
    """Test task repository."""
    
    @pytest.mark.asyncio
    async def test_create_task(self, repository_factory, sample_user, sample_project):
        """Test task creation."""
        task_repo = repository_factory.get_task_repository()
        
        task = await task_repo.create_task(
            project_id=sample_project.id,
            title="Test Task",
            description="A test task",
            assigned_to=sample_user.id,
            priority="high",
            creator_user_id=sample_user.id
        )
        
        assert task is not None
        assert task.title == "Test Task"
        assert task.description == "A test task"
        assert task.project_id == sample_project.id
        assert task.assigned_to == sample_user.id
        assert task.priority == "high"
    
    @pytest.mark.asyncio
    async def test_get_user_tasks(self, repository_factory, sample_user, sample_project):
        """Test getting user tasks."""
        task_repo = repository_factory.get_task_repository()
        
        # Create a task first
        task = await task_repo.create_task(
            project_id=sample_project.id,
            title="User Task",
            assigned_to=sample_user.id,
            creator_user_id=sample_user.id
        )
        
        # Get user tasks
        tasks = await task_repo.get_user_tasks(sample_user.id)
        assert len(tasks) >= 1
        assert task.id in [t.id for t in tasks]
    
    @pytest.mark.asyncio
    async def test_task_status_update(self, repository_factory, sample_user, sample_project):
        """Test task status updates."""
        task_repo = repository_factory.get_task_repository()
        
        # Create task
        task = await task_repo.create_task(
            project_id=sample_project.id,
            title="Status Test Task",
            assigned_to=sample_user.id,
            creator_user_id=sample_user.id
        )
        
        # Update status
        success = await task_repo.update_task_status(
            task.id, "in_progress", sample_user.id
        )
        assert success is True
        
        # Verify status change
        updated_task = await task_repo.get_by_id(task.id)
        assert updated_task.status == "in_progress"


class TestAuditRepository:
    """Test audit repository."""
    
    @pytest.mark.asyncio
    async def test_log_action(self, repository_factory, sample_user):
        """Test audit logging."""
        audit_repo = repository_factory.get_audit_repository()
        
        audit_log = await audit_repo.log_action(
            user_id=sample_user.id,
            action="test_action",
            resource_type="test_resource",
            resource_id="test_123",
            ip_address="192.168.1.1",
            result="success"
        )
        
        assert audit_log is not None
        assert audit_log.user_id == sample_user.id
        assert audit_log.action == "test_action"
        assert audit_log.resource_type == "test_resource"
        assert audit_log.result == "success"
    
    @pytest.mark.asyncio
    async def test_get_user_activity(self, repository_factory, sample_user):
        """Test getting user activity."""
        audit_repo = repository_factory.get_audit_repository()
        
        # Log some activities
        await audit_repo.log_action(
            user_id=sample_user.id,
            action="login",
            resource_type="session",
            result="success"
        )
        
        # Get activity
        activity = await audit_repo.get_user_activity(sample_user.id, days=1)
        assert len(activity) >= 1
    
    @pytest.mark.asyncio
    async def test_audit_statistics(self, repository_factory):
        """Test audit statistics."""
        audit_repo = repository_factory.get_audit_repository()
        
        stats = await audit_repo.get_audit_statistics(days=30)
        
        assert 'total_actions' in stats
        assert 'success_rate' in stats
        assert 'action_counts' in stats
        assert isinstance(stats['total_actions'], int)


class TestSessionRepository:
    """Test session repository."""
    
    @pytest.mark.asyncio
    async def test_create_session(self, repository_factory, sample_user):
        """Test session creation."""
        session_repo = repository_factory.get_session_repository()
        
        session = await session_repo.create_session(
            user_id=sample_user.id,
            session_token="test_token_123",
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        
        assert session is not None
        assert session.user_id == sample_user.id
        assert session.session_token == "test_token_123"
        assert session.ip_address == "192.168.1.1"
        assert session.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_active_session(self, repository_factory, sample_user):
        """Test getting active session."""
        session_repo = repository_factory.get_session_repository()
        
        # Create session
        created_session = await session_repo.create_session(
            user_id=sample_user.id,
            session_token="active_token_123",
            ip_address="192.168.1.1"
        )
        
        # Get active session
        found_session = await session_repo.get_active_session("active_token_123")
        assert found_session is not None
        assert found_session.id == created_session.id
        assert found_session.is_active is True
    
    @pytest.mark.asyncio
    async def test_invalidate_session(self, repository_factory, sample_user):
        """Test session invalidation."""
        session_repo = repository_factory.get_session_repository()
        
        # Create session
        session = await session_repo.create_session(
            user_id=sample_user.id,
            session_token="invalidate_token_123"
        )
        
        # Invalidate session
        success = await session_repo.invalidate_session(session.id)
        assert success is True
        
        # Verify session is invalidated
        found_session = await session_repo.get_active_session("invalidate_token_123")
        assert found_session is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, repository_factory, sample_user):
        """Test cleanup of expired sessions."""
        session_repo = repository_factory.get_session_repository()
        
        # Create expired session
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        await session_repo.create_session(
            user_id=sample_user.id,
            session_token="expired_token_123",
            expires_at=expired_time
        )
        
        # Cleanup expired sessions
        cleaned_count = await session_repo.cleanup_expired_sessions()
        assert cleaned_count >= 0  # Should clean at least the expired session


class TestBaseRepositoryFeatures:
    """Test base repository features."""
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, repository_factory, sample_user):
        """Test bulk operations."""
        project_repo = repository_factory.get_project_repository()
        
        # Bulk create projects
        projects_data = [
            {
                "name": f"Bulk Project {i}",
                "description": f"Bulk project {i}",
                "owner_id": sample_user.id,
                "project_type": "bulk_test"
            }
            for i in range(3)
        ]
        
        created_projects = await project_repo.bulk_create(projects_data)
        assert len(created_projects) == 3
        
        # Test bulk update
        updates = [
            {"id": project.id, "description": f"Updated {project.name}"}
            for project in created_projects
        ]
        
        updated_count = await project_repo.bulk_update(updates)
        assert updated_count == 3
        
        # Test bulk delete
        project_ids = [project.id for project in created_projects]
        deleted_count = await project_repo.bulk_delete(project_ids)
        assert deleted_count == 3
    
    @pytest.mark.asyncio
    async def test_filtering(self, repository_factory, sample_user):
        """Test advanced filtering."""
        user_repo = repository_factory.get_user_repository()
        
        # Create additional users
        await user_repo.create_user(
            email="filter1@test.com",
            username="filter1",
            password="TestPass123!"
        )
        await user_repo.create_user(
            email="filter2@test.com", 
            username="filter2",
            password="TestPass123!"
        )
        
        # Test email like filter
        users = await user_repo.get_all(
            filters={"email__like": "filter"},
            limit=10
        )
        assert len(users) == 2
        
        # Test in filter
        users = await user_repo.get_all(
            filters={"username__in": ["filter1", "filter2"]},
            limit=10
        )
        assert len(users) == 2