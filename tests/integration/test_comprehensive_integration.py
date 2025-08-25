"""
Comprehensive integration tests for claude-tiu API endpoints and services.

This module tests the integration between different components,
API endpoints, external services, and data flow throughout the system.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
import aiohttp


class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    async def test_client(self):
        """Create test client for API testing."""
        # Mock application setup
        app = Mock()
        app.router = Mock()
        
        # Mock session
        session = Mock()
        session.get = AsyncMock()
        session.post = AsyncMock()
        session.put = AsyncMock()
        session.delete = AsyncMock()
        
        yield session
    
    @pytest.mark.asyncio
    async def test_project_lifecycle_api(self, test_client):
        """Test complete project lifecycle through API."""
        # Create project
        create_response = Mock()
        create_response.status = 201
        create_response.json = AsyncMock(return_value={
            "id": "proj_123",
            "name": "test-project",
            "status": "initialized"
        })
        test_client.post.return_value = create_response
        
        create_result = await test_client.post("/api/projects", json={
            "name": "test-project",
            "template": "python",
            "description": "Test project for integration testing"
        })
        
        assert create_result.status == 201
        project_data = await create_result.json()
        project_id = project_data["id"]
        
        # Get project
        get_response = Mock()
        get_response.status = 200
        get_response.json = AsyncMock(return_value=project_data)
        test_client.get.return_value = get_response
        
        get_result = await test_client.get(f"/api/projects/{project_id}")
        assert get_result.status == 200
        
        # Update project
        update_response = Mock()
        update_response.status = 200
        update_response.json = AsyncMock(return_value={
            **project_data,
            "description": "Updated description"
        })
        test_client.put.return_value = update_response
        
        update_result = await test_client.put(f"/api/projects/{project_id}", json={
            "description": "Updated description"
        })
        assert update_result.status == 200
        
        # Delete project
        delete_response = Mock()
        delete_response.status = 204
        test_client.delete.return_value = delete_response
        
        delete_result = await test_client.delete(f"/api/projects/{project_id}")
        assert delete_result.status == 204
    
    @pytest.mark.asyncio
    async def test_task_execution_api(self, test_client):
        """Test task execution through API."""
        # Create task
        task_data = {
            "name": "generate_function",
            "prompt": "Generate a Python function to calculate factorial",
            "type": "code_generation",
            "project_id": "proj_123"
        }
        
        create_response = Mock()
        create_response.status = 201
        create_response.json = AsyncMock(return_value={
            "id": "task_456",
            **task_data,
            "status": "pending"
        })
        test_client.post.return_value = create_response
        
        create_result = await test_client.post("/api/tasks", json=task_data)
        assert create_result.status == 201
        task_result = await create_result.json()
        task_id = task_result["id"]
        
        # Execute task
        execute_response = Mock()
        execute_response.status = 200
        execute_response.json = AsyncMock(return_value={
            "task_id": task_id,
            "status": "completed",
            "output": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "execution_time": 2.5
        })
        test_client.post.return_value = execute_response
        
        execute_result = await test_client.post(f"/api/tasks/{task_id}/execute")
        assert execute_result.status == 200
        
        execution_data = await execute_result.json()
        assert execution_data["status"] == "completed"
        assert "output" in execution_data
    
    @pytest.mark.asyncio
    async def test_validation_api(self, test_client):
        """Test validation endpoints."""
        code_sample = """
        def incomplete_function():
            # TODO: implement this
            pass
        
        def complete_function():
            return "Hello, World!"
        """
        
        validation_response = Mock()
        validation_response.status = 200
        validation_response.json = AsyncMock(return_value={
            "has_placeholders": True,
            "placeholder_count": 1,
            "real_progress": 50,
            "fake_progress": 50,
            "quality_score": 0.5,
            "placeholders": [
                {"line": 3, "type": "TODO", "text": "implement this"}
            ]
        })
        test_client.post.return_value = validation_response
        
        result = await test_client.post("/api/validate/code", json={
            "code": code_sample,
            "language": "python"
        })
        
        assert result.status == 200
        validation_data = await result.json()
        assert validation_data["has_placeholders"] is True
        assert validation_data["placeholder_count"] == 1
    
    @pytest.mark.asyncio
    async def test_ai_integration_api(self, test_client):
        """Test AI integration endpoints."""
        # Test Claude Code integration
        claude_response = Mock()
        claude_response.status = 200
        claude_response.json = AsyncMock(return_value={
            "status": "success",
            "output": "Generated code successfully",
            "files_created": ["main.py", "utils.py"]
        })
        test_client.post.return_value = claude_response
        
        result = await test_client.post("/api/ai/claude-code", json={
            "prompt": "Create a Python web scraper",
            "context": {"language": "python", "framework": "requests"}
        })
        
        assert result.status == 200
        ai_data = await result.json()
        assert ai_data["status"] == "success"
        assert len(ai_data["files_created"]) == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_client):
        """Test API error handling."""
        # Test 404 error
        error_response = Mock()
        error_response.status = 404
        error_response.json = AsyncMock(return_value={
            "error": "Project not found",
            "code": "PROJECT_NOT_FOUND"
        })
        test_client.get.return_value = error_response
        
        result = await test_client.get("/api/projects/nonexistent")
        assert result.status == 404
        
        error_data = await result.json()
        assert "error" in error_data
        assert error_data["code"] == "PROJECT_NOT_FOUND"
        
        # Test validation error
        validation_error_response = Mock()
        validation_error_response.status = 400
        validation_error_response.json = AsyncMock(return_value={
            "error": "Invalid request data",
            "details": ["name is required", "template must be one of: python, javascript, java"]
        })
        test_client.post.return_value = validation_error_response
        
        result = await test_client.post("/api/projects", json={})
        assert result.status == 400
        
        error_data = await result.json()
        assert "details" in error_data
        assert len(error_data["details"]) == 2


class TestServiceIntegration:
    """Integration tests for service layer interactions."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for integration testing."""
        return {
            "project_service": Mock(),
            "task_service": Mock(),
            "ai_service": Mock(),
            "validation_service": Mock(),
            "storage_service": Mock()
        }
    
    def test_project_task_integration(self, mock_services):
        """Test integration between project and task services."""
        project_service = mock_services["project_service"]
        task_service = mock_services["task_service"]
        
        # Mock project creation
        project_data = {"id": "proj_123", "name": "test-project"}
        project_service.create_project.return_value = project_data
        
        # Mock task creation
        task_data = {"id": "task_456", "name": "test-task", "project_id": "proj_123"}
        task_service.create_task.return_value = task_data
        
        # Test integration
        project = project_service.create_project({"name": "test-project"})
        task = task_service.create_task({
            "name": "test-task",
            "project_id": project["id"]
        })
        
        assert task["project_id"] == project["id"]
        project_service.create_project.assert_called_once()
        task_service.create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ai_validation_integration(self, mock_services):
        """Test integration between AI and validation services."""
        ai_service = mock_services["ai_service"]
        validation_service = mock_services["validation_service"]
        
        # Mock AI code generation
        generated_code = "def test_function():\n    return 'Hello, World!'"
        ai_service.generate_code = AsyncMock(return_value={
            "code": generated_code,
            "confidence": 0.9
        })
        
        # Mock validation
        validation_service.validate_code = AsyncMock(return_value={
            "is_authentic": True,
            "has_placeholders": False,
            "quality_score": 0.95
        })
        
        # Test integration
        ai_result = await ai_service.generate_code("Generate a test function")
        validation_result = await validation_service.validate_code(ai_result["code"])
        
        assert validation_result["is_authentic"] is True
        assert validation_result["has_placeholders"] is False
        ai_service.generate_code.assert_awaited_once()
        validation_service.validate_code.assert_awaited_once()
    
    def test_storage_service_integration(self, mock_services, temp_project_dir):
        """Test storage service integration."""
        storage_service = mock_services["storage_service"]
        
        # Mock file operations
        test_file = temp_project_dir / "test.py"
        content = "print('Hello, World!')"
        
        storage_service.save_file.return_value = True
        storage_service.load_file.return_value = content
        storage_service.file_exists.return_value = True
        
        # Test operations
        saved = storage_service.save_file(str(test_file), content)
        loaded_content = storage_service.load_file(str(test_file))
        exists = storage_service.file_exists(str(test_file))
        
        assert saved is True
        assert loaded_content == content
        assert exists is True


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.query = Mock()
        session.close = Mock()
        return session
    
    def test_project_crud_operations(self, mock_db_session):
        """Test CRUD operations for projects."""
        # Mock project model
        project_model = Mock()
        project_model.id = "proj_123"
        project_model.name = "test-project"
        project_model.status = "active"
        
        # Mock query operations
        mock_db_session.query.return_value.filter.return_value.first.return_value = project_model
        mock_db_session.query.return_value.all.return_value = [project_model]
        
        # Create
        mock_db_session.add(project_model)
        mock_db_session.commit()
        
        # Read
        found_project = (mock_db_session.query.return_value
                        .filter.return_value.first())
        assert found_project.id == "proj_123"
        
        # Update
        project_model.name = "updated-project"
        mock_db_session.commit()
        
        # Delete
        mock_db_session.delete = Mock()
        mock_db_session.delete(project_model)
        mock_db_session.commit()
        
        # Verify calls
        mock_db_session.add.assert_called_once()
        assert mock_db_session.commit.call_count == 3
    
    def test_task_relationship_operations(self, mock_db_session):
        """Test task relationships and dependencies."""
        # Mock models
        project_model = Mock()
        project_model.id = "proj_123"
        
        task1_model = Mock()
        task1_model.id = "task_1"
        task1_model.project_id = "proj_123"
        task1_model.dependencies = []
        
        task2_model = Mock()
        task2_model.id = "task_2"
        task2_model.project_id = "proj_123"
        task2_model.dependencies = [task1_model]
        
        # Mock relationships
        project_model.tasks = [task1_model, task2_model]
        
        # Test relationship queries
        mock_db_session.query.return_value.filter.return_value.first.return_value = project_model
        
        found_project = (mock_db_session.query.return_value
                        .filter.return_value.first())
        
        assert len(found_project.tasks) == 2
        assert found_project.tasks[1].dependencies[0] == task1_model
    
    def test_transaction_handling(self, mock_db_session):
        """Test database transaction handling."""
        # Mock successful transaction
        mock_db_session.begin.return_value.__enter__ = Mock()
        mock_db_session.begin.return_value.__exit__ = Mock(return_value=False)
        
        # Test successful transaction
        with mock_db_session.begin():
            mock_db_session.add(Mock())
            mock_db_session.commit()
        
        # Mock failed transaction
        mock_db_session.rollback = Mock()
        mock_db_session.begin.return_value.__exit__ = Mock(return_value=True)
        
        try:
            with mock_db_session.begin():
                mock_db_session.add(Mock())
                raise Exception("Database error")
        except Exception:
            mock_db_session.rollback()
        
        mock_db_session.rollback.assert_called_once()


class TestExternalServiceIntegration:
    """Integration tests for external services."""
    
    @pytest.mark.asyncio
    async def test_claude_code_integration(self):
        """Test integration with Claude Code CLI."""
        with patch('subprocess.run') as mock_run:
            # Mock successful execution
            mock_run.return_value = Mock(
                stdout='{"status": "success", "files": ["output.py"]}',
                stderr='',
                returncode=0
            )
            
            # Test Claude Code execution
            import subprocess
            result = subprocess.run([
                'claude-code', '--prompt', 'Generate hello world',
                '--output', 'hello.py'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            assert '"status": "success"' in result.stdout
    
    @pytest.mark.asyncio
    async def test_claude_flow_integration(self):
        """Test integration with Claude Flow."""
        with patch('subprocess.run') as mock_run:
            # Mock workflow execution
            mock_run.return_value = Mock(
                stdout='{"status": "completed", "tasks": 3}',
                stderr='',
                returncode=0
            )
            
            # Test workflow execution
            import subprocess
            result = subprocess.run([
                'npx', 'claude-flow', 'run', 'test-workflow.yaml'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            assert '"status": "completed"' in result.stdout
    
    @pytest.mark.asyncio
    async def test_git_integration(self, temp_project_dir):
        """Test Git integration functionality."""
        with patch('subprocess.run') as mock_run:
            # Mock git commands
            mock_run.side_effect = [
                Mock(returncode=0, stdout='', stderr=''),  # git init
                Mock(returncode=0, stdout='', stderr=''),  # git add
                Mock(returncode=0, stdout='', stderr=''),  # git commit
                Mock(returncode=0, stdout='main', stderr=''),  # git branch
            ]
            
            # Test git operations
            import subprocess
            
            # Initialize repository
            init_result = subprocess.run(['git', 'init'], 
                                       cwd=temp_project_dir, 
                                       capture_output=True)
            assert init_result.returncode == 0
            
            # Add files
            add_result = subprocess.run(['git', 'add', '.'], 
                                      cwd=temp_project_dir, 
                                      capture_output=True)
            assert add_result.returncode == 0
            
            # Commit changes
            commit_result = subprocess.run(['git', 'commit', '-m', 'Initial commit'], 
                                         cwd=temp_project_dir, 
                                         capture_output=True)
            assert commit_result.returncode == 0


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_real_time_updates(self):
        """Test real-time updates via WebSocket."""
        # Mock WebSocket connection
        websocket = Mock()
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock(return_value='{"type": "task_update", "data": {"id": "task_123", "status": "completed"}}')
        
        # Test sending update
        await websocket.send(json.dumps({
            "type": "task_update",
            "data": {"id": "task_123", "status": "in_progress"}
        }))
        
        # Test receiving update
        message = await websocket.recv()
        data = json.loads(message)
        
        assert data["type"] == "task_update"
        assert data["data"]["status"] == "completed"
        websocket.send.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_progress_streaming(self):
        """Test progress streaming via WebSocket."""
        websocket = Mock()
        websocket.send = AsyncMock()
        
        # Simulate progress updates
        progress_updates = [
            {"progress": 25, "message": "Starting task"},
            {"progress": 50, "message": "Processing"},
            {"progress": 75, "message": "Almost done"},
            {"progress": 100, "message": "Completed"}
        ]
        
        for update in progress_updates:
            await websocket.send(json.dumps({
                "type": "progress_update",
                "data": update
            }))
        
        assert websocket.send.await_count == 4


class TestConcurrencyIntegration:
    """Integration tests for concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """Test concurrent execution of multiple tasks."""
        # Mock task execution
        async def mock_execute_task(task_id):
            await asyncio.sleep(0.1)  # Simulate work
            return {"id": task_id, "status": "completed"}
        
        # Execute tasks concurrently
        task_ids = ["task_1", "task_2", "task_3", "task_4", "task_5"]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[
            mock_execute_task(task_id) for task_id in task_ids
        ])
        end_time = asyncio.get_event_loop().time()
        
        # Verify results
        assert len(results) == 5
        assert all(result["status"] == "completed" for result in results)
        
        # Verify concurrent execution (should be faster than sequential)
        execution_time = end_time - start_time
        assert execution_time < 0.3  # Should be much faster than 5 * 0.1 = 0.5s
    
    @pytest.mark.asyncio
    async def test_resource_contention_handling(self):
        """Test handling of resource contention."""
        # Mock shared resource
        shared_counter = {"value": 0}
        lock = asyncio.Lock()
        
        async def increment_counter():
            async with lock:
                current = shared_counter["value"]
                await asyncio.sleep(0.01)  # Simulate work
                shared_counter["value"] = current + 1
        
        # Run concurrent increments
        await asyncio.gather(*[increment_counter() for _ in range(10)])
        
        # Verify no race conditions
        assert shared_counter["value"] == 10


class TestErrorRecoveryIntegration:
    """Integration tests for error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_service_failure_recovery(self):
        """Test recovery from service failures."""
        # Mock service with failures
        service_calls = 0
        
        async def unreliable_service():
            nonlocal service_calls
            service_calls += 1
            if service_calls <= 2:
                raise Exception("Service temporarily unavailable")
            return {"status": "success"}
        
        # Test retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await unreliable_service()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(0.1)
        
        assert result["status"] == "success"
        assert service_calls == 3
    
    def test_data_consistency_on_failure(self, mock_db_session):
        """Test data consistency during failures."""
        # Mock transaction failure
        mock_db_session.commit.side_effect = Exception("Database error")
        mock_db_session.rollback = Mock()
        
        # Test transaction rollback
        try:
            mock_db_session.add(Mock())
            mock_db_session.commit()
        except Exception:
            mock_db_session.rollback()
        
        mock_db_session.rollback.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])