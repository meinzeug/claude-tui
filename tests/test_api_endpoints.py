"""
Tests for API endpoints implementation.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import asyncio
import uuid
from datetime import datetime, timedelta

from src.api.main import create_app
from src.api.models.base import Base, get_database
from src.api.models.user import User, UserSession
from src.api.models.project import Project, Task
from src.api.dependencies.auth import get_password_hash, create_access_token


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture
async def async_session():
    """Create test database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def test_client():
    """Create test client."""
    app = create_app()
    
    async def override_get_database():
        async with TestingSessionLocal() as session:
            yield session
    
    app.dependency_overrides[get_database] = override_get_database
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def test_user(async_session: AsyncSession):
    """Create test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash=get_password_hash("testpass123"),
        role="developer"
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    
    return user


@pytest.fixture
async def auth_headers(test_user):
    """Create authentication headers."""
    access_token = create_access_token(data={"sub": str(test_user.id)})
    return {"Authorization": f"Bearer {access_token}"}


class TestProjectEndpoints:
    """Test project management endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_project(self, test_client, auth_headers):
        """Test project creation endpoint."""
        project_data = {
            "name": "Test Project",
            "path": "/tmp/test-project",
            "project_type": "python",
            "description": "A test project",
            "initialize_git": True,
            "create_venv": True,
            "config": {"test": True}
        }
        
        response = test_client.post(
            "/api/v1/projects/",
            json=project_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["project_type"] == "python"
        assert "project_id" in data
    
    @pytest.mark.asyncio
    async def test_list_projects(self, test_client, auth_headers):
        """Test project listing endpoint."""
        response = test_client.get(
            "/api/v1/projects/",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
    
    @pytest.mark.asyncio
    async def test_get_project(self, test_client, auth_headers):
        """Test get project endpoint."""
        # First create a project
        project_data = {
            "name": "Test Project",
            "path": "/tmp/test-project2",
            "project_type": "python"
        }
        
        create_response = test_client.post(
            "/api/v1/projects/",
            json=project_data,
            headers=auth_headers
        )
        
        assert create_response.status_code == 201
        project_id = create_response.json()["project_id"]
        
        # Now get the project
        response = test_client.get(
            f"/api/v1/projects/{project_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == project_id
        assert data["name"] == "Test Project"


class TestTaskEndpoints:
    """Test task management endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_task(self, test_client, auth_headers):
        """Test task creation endpoint."""
        task_data = {
            "name": "Test Task",
            "description": "A test task",
            "task_type": "code_generation",
            "priority": "high",
            "timeout_seconds": 300,
            "dependencies": [],
            "ai_enabled": True,
            "config": {"test": True}
        }
        
        response = test_client.post(
            "/api/v1/tasks/",
            json=task_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Task"
        assert data["priority"] == "high"
        assert "task_id" in data
    
    @pytest.mark.asyncio
    async def test_list_tasks(self, test_client, auth_headers):
        """Test task listing endpoint."""
        response = test_client.get(
            "/api/v1/tasks/",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data
    
    @pytest.mark.asyncio
    async def test_execute_task(self, test_client, auth_headers):
        """Test task execution endpoint."""
        # First create a task
        task_data = {
            "name": "Test Task",
            "description": "A test task",
            "task_type": "code_generation",
            "priority": "medium"
        }
        
        create_response = test_client.post(
            "/api/v1/tasks/",
            json=task_data,
            headers=auth_headers
        )
        
        assert create_response.status_code == 201
        task_id = create_response.json()["task_id"]
        
        # Execute the task
        execute_data = {
            "execution_mode": "adaptive",
            "wait_for_dependencies": True
        }
        
        response = test_client.post(
            f"/api/v1/tasks/{task_id}/execute",
            json=execute_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id


class TestValidationEndpoints:
    """Test validation endpoints."""
    
    @pytest.mark.asyncio
    async def test_validate_code(self, test_client, auth_headers):
        """Test code validation endpoint."""
        code_data = {
            "code": "def hello_world():\n    print('Hello, World!')",
            "language": "python",
            "file_path": "hello.py",
            "validation_level": "standard",
            "check_placeholders": True,
            "check_syntax": True,
            "check_quality": True
        }
        
        response = test_client.post(
            "/api/v1/validation/code",
            json=code_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "score" in data
        assert "issues" in data
        assert "warnings" in data
        assert "suggestions" in data
    
    @pytest.mark.asyncio
    async def test_validate_response(self, test_client, auth_headers):
        """Test response validation endpoint."""
        response_data = {
            "response": "This is a test response",
            "response_type": "text",
            "validation_criteria": {"min_length": 10}
        }
        
        response = test_client.post(
            "/api/v1/validation/response",
            json=response_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "score" in data


class TestAIEndpoints:
    """Test AI integration endpoints."""
    
    @pytest.mark.asyncio
    async def test_generate_code(self, test_client, auth_headers):
        """Test code generation endpoint."""
        generation_data = {
            "prompt": "Create a simple hello world function in Python",
            "language": "python",
            "context": {"framework": "none"},
            "validate_response": True,
            "use_cache": False
        }
        
        response = test_client.post(
            "/api/v1/ai/code/generate",
            json=generation_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "code" in data
        assert "language" in data
        assert data["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_orchestrate_task(self, test_client, auth_headers):
        """Test task orchestration endpoint."""
        orchestration_data = {
            "task_description": "Create a simple web API with authentication",
            "requirements": {"framework": "fastapi"},
            "agents": [],
            "strategy": "adaptive"
        }
        
        response = test_client.post(
            "/api/v1/ai/orchestrate",
            json=orchestration_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "status" in data
        assert "strategy_used" in data
    
    @pytest.mark.asyncio
    async def test_ai_performance_metrics(self, test_client, auth_headers):
        """Test AI performance metrics endpoint."""
        response = test_client.get(
            "/api/v1/ai/performance",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "success_rate" in data


class TestAuthentication:
    """Test authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, test_client):
        """Test unauthorized access is rejected."""
        response = test_client.get("/api/v1/projects/")
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_invalid_token(self, test_client):
        """Test invalid token is rejected."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = test_client.get("/api/v1/projects/", headers=headers)
        assert response.status_code == 401


class TestHealthChecks:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_main_health_check(self, test_client):
        """Test main health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_service_health_checks(self, test_client, auth_headers):
        """Test service-specific health checks."""
        endpoints = [
            "/api/v1/tasks/health/service",
            "/api/v1/validation/health/service",
            "/api/v1/ai/providers/status"
        ]
        
        for endpoint in endpoints:
            response = test_client.get(endpoint, headers=auth_headers)
            assert response.status_code in [200, 503]  # 200 or 503 (service unavailable)


if __name__ == "__main__":
    pytest.main([__file__])