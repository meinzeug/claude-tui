"""
Database Integration Tests for claude-tui.

Comprehensive tests for database operations including:
- Session management and connection pooling
- Repository pattern implementation
- Transaction handling and rollbacks
- Database migrations and schema validation
- Concurrent access and performance
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from contextlib import asynccontextmanager

from src.database.models import Base, User, Project, Task, UserSession, Role, Permission
from src.database.repositories import UserRepository, ProjectRepository, TaskRepository
from src.core.exceptions import DatabaseError, ValidationError


@pytest.fixture(scope="session")
async def async_engine():
    """Create async database engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine):
    """Create async database session for testing."""
    async_session = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def sample_user(async_session: AsyncSession):
    """Create sample user for testing."""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User"
    )
    user.set_password("SecurePass123!")
    
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    
    return user


@pytest.fixture
async def sample_project(async_session: AsyncSession, sample_user: User):
    """Create sample project for testing."""
    project = Project(
        name="Test Project",
        description="A test project for database testing",
        owner_id=sample_user.id,
        project_type="web_app",
        status="active"
    )
    
    async_session.add(project)
    await async_session.commit()
    await async_session.refresh(project)
    
    return project


@pytest.mark.asyncio
class TestDatabaseConnection:
    """Test database connection and session management."""
    
    async def test_connection_establishment(self, async_engine):
        """Test that database connection can be established."""
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1 as test_value"))
            row = result.fetchone()
            assert row[0] == 1
    
    async def test_session_creation(self, async_session):
        """Test that database sessions can be created."""
        assert async_session is not None
        assert isinstance(async_session, AsyncSession)
    
    async def test_session_transaction_commit(self, async_session):
        """Test session transaction commit."""
        user = User(
            email="commit@test.com",
            username="commituser",
            full_name="Commit User"
        )
        user.set_password("Password123!")
        
        async_session.add(user)
        await async_session.commit()
        
        # Verify user was persisted
        result = await async_session.execute(
            text("SELECT email FROM users WHERE email = :email"),
            {"email": "commit@test.com"}
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "commit@test.com"
    
    async def test_session_transaction_rollback(self, async_session):
        """Test session transaction rollback."""
        user = User(
            email="rollback@test.com",
            username="rollbackuser",
            full_name="Rollback User"
        )
        user.set_password("Password123!")
        
        async_session.add(user)
        await async_session.rollback()
        
        # Verify user was not persisted
        result = await async_session.execute(
            text("SELECT email FROM users WHERE email = :email"),
            {"email": "rollback@test.com"}
        )
        row = result.fetchone()
        assert row is None
    
    async def test_concurrent_sessions(self, async_engine):
        """Test concurrent database sessions."""
        async_session = sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async def create_user(session_id: int):
            async with async_session() as session:
                user = User(
                    email=f"concurrent{session_id}@test.com",
                    username=f"concurrent{session_id}",
                    full_name=f"Concurrent User {session_id}"
                )
                user.set_password("Password123!")
                
                session.add(user)
                await session.commit()
                return user.id
        
        # Create multiple users concurrently
        tasks = [create_user(i) for i in range(5)]
        user_ids = await asyncio.gather(*tasks)
        
        # Verify all users were created
        assert len(user_ids) == 5
        assert all(user_id is not None for user_id in user_ids)


@pytest.mark.asyncio
class TestRepositoryPattern:
    """Test repository pattern implementation."""
    
    async def test_user_repository_creation(self, async_session):
        """Test user repository instantiation."""
        repo = UserRepository(async_session)
        assert repo is not None
        assert repo.session == async_session
    
    async def test_user_repository_create(self, async_session):
        """Test user creation through repository."""
        repo = UserRepository(async_session)
        
        user_data = {
            "email": "repo@test.com",
            "username": "repouser",
            "full_name": "Repository User",
            "password": "SecurePass123!"
        }
        
        user = await repo.create_user(user_data)
        
        assert user is not None
        assert user.email == "repo@test.com"
        assert user.username == "repouser"
        assert user.verify_password("SecurePass123!")
    
    async def test_user_repository_get_by_id(self, async_session, sample_user):
        """Test user retrieval by ID through repository."""
        repo = UserRepository(async_session)
        
        retrieved_user = await repo.get_by_id(sample_user.id)
        
        assert retrieved_user is not None
        assert retrieved_user.id == sample_user.id
        assert retrieved_user.email == sample_user.email
    
    async def test_user_repository_get_by_email(self, async_session, sample_user):
        """Test user retrieval by email through repository."""
        repo = UserRepository(async_session)
        
        retrieved_user = await repo.get_by_email(sample_user.email)
        
        assert retrieved_user is not None
        assert retrieved_user.id == sample_user.id
        assert retrieved_user.email == sample_user.email
    
    async def test_user_repository_update(self, async_session, sample_user):
        """Test user update through repository."""
        repo = UserRepository(async_session)
        
        update_data = {
            "full_name": "Updated Full Name",
            "is_verified": True
        }
        
        updated_user = await repo.update_user(sample_user.id, update_data)
        
        assert updated_user.full_name == "Updated Full Name"
        assert updated_user.is_verified is True
    
    async def test_user_repository_delete(self, async_session):
        """Test user deletion through repository."""
        repo = UserRepository(async_session)
        
        # Create user to delete
        user_data = {
            "email": "delete@test.com",
            "username": "deleteuser",
            "password": "Password123!"
        }
        user = await repo.create_user(user_data)
        user_id = user.id
        
        # Delete user
        success = await repo.delete_user(user_id)
        assert success is True
        
        # Verify user was deleted
        deleted_user = await repo.get_by_id(user_id)
        assert deleted_user is None
    
    async def test_project_repository_operations(self, async_session, sample_user):
        """Test project repository CRUD operations."""
        repo = ProjectRepository(async_session)
        
        # Create project
        project_data = {
            "name": "Repository Test Project",
            "description": "Testing project repository",
            "owner_id": sample_user.id,
            "project_type": "api"
        }
        
        project = await repo.create_project(project_data)
        assert project.name == "Repository Test Project"
        assert project.owner_id == sample_user.id
        
        # Get project by ID
        retrieved_project = await repo.get_by_id(project.id)
        assert retrieved_project is not None
        assert retrieved_project.id == project.id
        
        # Update project
        update_data = {"status": "completed"}
        updated_project = await repo.update_project(project.id, update_data)
        assert updated_project.status == "completed"
        
        # Get projects by owner
        owner_projects = await repo.get_projects_by_owner(sample_user.id)
        assert len(owner_projects) >= 1
        assert any(p.id == project.id for p in owner_projects)
    
    async def test_repository_error_handling(self, async_session):
        """Test repository error handling for invalid operations."""
        repo = UserRepository(async_session)
        
        # Test duplicate email creation
        user_data = {
            "email": "duplicate@test.com",
            "username": "user1",
            "password": "Password123!"
        }
        
        await repo.create_user(user_data)
        
        # Try to create another user with same email
        duplicate_data = {
            "email": "duplicate@test.com",
            "username": "user2",
            "password": "Password123!"
        }
        
        with pytest.raises((IntegrityError, ValidationError)):
            await repo.create_user(duplicate_data)


@pytest.mark.asyncio
class TestTransactionHandling:
    """Test database transaction handling and rollbacks."""
    
    async def test_transaction_success(self, async_session):
        """Test successful transaction with multiple operations."""
        async with async_session.begin():
            # Create user
            user = User(
                email="transaction@test.com",
                username="transactionuser",
                full_name="Transaction User"
            )
            user.set_password("Password123!")
            async_session.add(user)
            
            # Flush to get user ID
            await async_session.flush()
            
            # Create project for user
            project = Project(
                name="Transaction Project",
                description="Testing transactions",
                owner_id=user.id,
                project_type="test"
            )
            async_session.add(project)
            
            # Transaction will commit automatically
        
        # Verify both objects were persisted
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM users WHERE email = :email"),
            {"email": "transaction@test.com"}
        )
        assert result.scalar() == 1
        
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM projects WHERE name = :name"),
            {"name": "Transaction Project"}
        )
        assert result.scalar() == 1
    
    async def test_transaction_rollback_on_error(self, async_session):
        """Test transaction rollback when error occurs."""
        try:
            async with async_session.begin():
                # Create user
                user = User(
                    email="rollback@test.com",
                    username="rollbackuser",
                    full_name="Rollback User"
                )
                user.set_password("Password123!")
                async_session.add(user)
                
                # Force an error by violating a constraint
                duplicate_user = User(
                    email="rollback@test.com",  # Duplicate email
                    username="rollbackuser2",
                    full_name="Duplicate User"
                )
                duplicate_user.set_password("Password123!")
                async_session.add(duplicate_user)
                
                await async_session.flush()  # This should raise IntegrityError
        except IntegrityError:
            pass  # Expected error
        
        # Verify no users were persisted due to rollback
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM users WHERE email = :email"),
            {"email": "rollback@test.com"}
        )
        assert result.scalar() == 0
    
    async def test_nested_transactions(self, async_session):
        """Test nested transaction handling."""
        async with async_session.begin():
            # Outer transaction - create user
            user = User(
                email="nested@test.com",
                username="nesteduser",
                full_name="Nested User"
            )
            user.set_password("Password123!")
            async_session.add(user)
            await async_session.flush()
            
            try:
                async with async_session.begin_nested():  # Savepoint
                    # Inner transaction - create project
                    project = Project(
                        name="Nested Project",
                        owner_id=user.id,
                        project_type="test"
                    )
                    async_session.add(project)
                    
                    # Force error in nested transaction
                    invalid_project = Project(
                        name="" * 200,  # Too long name
                        owner_id=user.id,
                        project_type="test"
                    )
                    async_session.add(invalid_project)
                    await async_session.flush()
            except (ValidationError, SQLAlchemyError):
                # Nested transaction rolled back, outer continues
                pass
        
        # User should exist, but project should not due to nested rollback
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM users WHERE email = :email"),
            {"email": "nested@test.com"}
        )
        assert result.scalar() == 1
        
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM projects WHERE name = :name"),
            {"name": "Nested Project"}
        )
        assert result.scalar() == 0


@pytest.mark.asyncio
class TestConnectionPooling:
    """Test database connection pooling behavior."""
    
    async def test_connection_pool_limits(self, async_engine):
        """Test connection pool handles concurrent requests."""
        @asynccontextmanager
        async def get_connection():
            async with async_engine.connect() as conn:
                yield conn
        
        async def execute_query(query_id: int):
            async with get_connection() as conn:
                result = await conn.execute(
                    text("SELECT :query_id as id"),
                    {"query_id": query_id}
                )
                row = result.fetchone()
                return row[0]
        
        # Execute multiple queries concurrently
        tasks = [execute_query(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All queries should complete successfully
        assert len(results) == 10
        assert sorted(results) == list(range(10))
    
    async def test_connection_cleanup(self, async_engine):
        """Test that connections are properly cleaned up."""
        initial_pool_size = async_engine.pool.size()
        
        async def use_connection():
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        
        # Use connections and verify cleanup
        await use_connection()
        await use_connection()
        await use_connection()
        
        # Pool size should return to initial state
        final_pool_size = async_engine.pool.size()
        assert final_pool_size == initial_pool_size


@pytest.mark.asyncio
class TestDatabasePerformance:
    """Test database performance under various conditions."""
    
    @pytest.mark.slow
    async def test_bulk_insert_performance(self, async_session):
        """Test bulk insert performance."""
        import time
        
        start_time = time.time()
        
        # Create many users in batches
        batch_size = 100
        total_users = 500
        
        for batch_start in range(0, total_users, batch_size):
            users = []
            for i in range(batch_start, min(batch_start + batch_size, total_users)):
                user = User(
                    email=f"bulk{i}@test.com",
                    username=f"bulk{i}",
                    full_name=f"Bulk User {i}"
                )
                user.set_password("Password123!")
                users.append(user)
            
            async_session.add_all(users)
            await async_session.commit()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 5.0, f"Bulk insert took {execution_time} seconds"
        
        # Verify all users were created
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM users WHERE email LIKE 'bulk%@test.com'")
        )
        assert result.scalar() == total_users
    
    @pytest.mark.slow
    async def test_query_performance_with_indexes(self, async_session, async_engine):
        """Test query performance with database indexes."""
        # Create test data
        users = []
        for i in range(100):
            user = User(
                email=f"perf{i}@test.com",
                username=f"perf{i}",
                full_name=f"Performance User {i}"
            )
            user.set_password("Password123!")
            users.append(user)
        
        async_session.add_all(users)
        await async_session.commit()
        
        # Test indexed query performance
        import time
        start_time = time.time()
        
        for i in range(50):
            email = f"perf{i}@test.com"
            result = await async_session.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": email}
            )
            result.fetchone()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Indexed queries should be fast
        assert execution_time < 1.0, f"Indexed queries took {execution_time} seconds"


@pytest.mark.asyncio
class TestDataIntegrity:
    """Test data integrity and constraints."""
    
    async def test_foreign_key_constraints(self, async_session, sample_user):
        """Test foreign key constraint enforcement."""
        # Create project with valid user ID
        project = Project(
            name="FK Test Project",
            owner_id=sample_user.id,
            project_type="test"
        )
        async_session.add(project)
        await async_session.commit()
        
        # Try to create project with invalid user ID
        invalid_project = Project(
            name="Invalid FK Project",
            owner_id=uuid.uuid4(),  # Non-existent user
            project_type="test"
        )
        async_session.add(invalid_project)
        
        with pytest.raises(IntegrityError):
            await async_session.commit()
    
    async def test_unique_constraints(self, async_session):
        """Test unique constraint enforcement."""
        # Create first user
        user1 = User(
            email="unique@test.com",
            username="uniqueuser",
            full_name="Unique User 1"
        )
        user1.set_password("Password123!")
        async_session.add(user1)
        await async_session.commit()
        
        # Try to create second user with same email
        user2 = User(
            email="unique@test.com",  # Duplicate
            username="uniqueuser2",
            full_name="Unique User 2"
        )
        user2.set_password("Password123!")
        async_session.add(user2)
        
        with pytest.raises(IntegrityError):
            await async_session.commit()
    
    async def test_cascade_deletion(self, async_session, sample_user, sample_project):
        """Test cascade deletion behavior."""
        # Create task for project
        task = Task(
            title="Test Task",
            description="Task for cascade test",
            project_id=sample_project.id,
            status="pending"
        )
        async_session.add(task)
        await async_session.commit()
        
        # Delete project - should cascade to tasks
        await async_session.delete(sample_project)
        await async_session.commit()
        
        # Verify task was deleted
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM tasks WHERE project_id = :project_id"),
            {"project_id": str(sample_project.id)}
        )
        assert result.scalar() == 0
    
    async def test_check_constraints(self, async_session, sample_user):
        """Test check constraints validation."""
        # Test invalid project status
        project = Project(
            name="Check Constraint Test",
            owner_id=sample_user.id,
            project_type="test",
            status="invalid_status"  # Should fail validation
        )
        
        with pytest.raises(ValidationError):
            async_session.add(project)
            await async_session.commit()


@pytest.mark.asyncio 
class TestConcurrentAccess:
    """Test concurrent database access scenarios."""
    
    async def test_concurrent_user_creation(self, async_engine):
        """Test concurrent user creation without conflicts."""
        async_session = sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async def create_user(user_id: int):
            async with async_session() as session:
                user = User(
                    email=f"concurrent{user_id}@test.com",
                    username=f"concurrent{user_id}",
                    full_name=f"Concurrent User {user_id}"
                )
                user.set_password("Password123!")
                
                session.add(user)
                await session.commit()
                return user.id
        
        # Create users concurrently
        tasks = [create_user(i) for i in range(10)]
        user_ids = await asyncio.gather(*tasks)
        
        assert len(user_ids) == 10
        assert all(user_id is not None for user_id in user_ids)
    
    async def test_concurrent_read_write(self, async_engine, sample_user):
        """Test concurrent read and write operations."""
        async_session_factory = sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async def read_user():
            async with async_session_factory() as session:
                result = await session.execute(
                    text("SELECT full_name FROM users WHERE id = :user_id"),
                    {"user_id": str(sample_user.id)}
                )
                row = result.fetchone()
                return row[0] if row else None
        
        async def update_user():
            async with async_session_factory() as session:
                await session.execute(
                    text("UPDATE users SET full_name = :name WHERE id = :user_id"),
                    {"name": "Updated Name", "user_id": str(sample_user.id)}
                )
                await session.commit()
        
        # Execute concurrent read and write operations
        tasks = [
            read_user() for _ in range(5)
        ] + [
            update_user() for _ in range(2)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent operations failed: {exceptions}"