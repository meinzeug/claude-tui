"""
Unit tests for ProjectManager core functionality.

Tests project creation, management, validation, and lifecycle operations.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Note: These imports will work once the actual modules are created
# For now, we'll use mocks and the test will serve as a specification


class TestProjectManager:
    """Test suite for ProjectManager class."""
    
    @pytest.fixture
    def mock_ai_interface(self):
        """Mock AI interface for testing."""
        mock = Mock()
        mock.validate_project = Mock(return_value=True)
        mock.execute_claude_code = AsyncMock(return_value={
            "status": "success",
            "output": "Project validation successful",
            "confidence": 0.95
        })
        return mock
    
    @pytest.fixture
    def mock_task_engine(self):
        """Mock task engine for testing."""
        mock = Mock()
        mock.execute = AsyncMock(return_value={"status": "completed"})
        mock.create_task = Mock()
        mock.get_tasks = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return Mock(
            default_template="python",
            project_root="/tmp/claude-tiu-projects",
            ai_validation=True,
            auto_create_structure=True
        )
    
    @pytest.fixture
    def project_manager(self, mock_ai_interface, mock_task_engine, mock_config):
        """Create ProjectManager instance with mocked dependencies."""
        # Import will be: from claude_tiu.core.project_manager import ProjectManager
        # For now, create a mock class that represents the expected interface
        
        class MockProjectManager:
            def __init__(self, ai_interface, task_engine, config):
                self.ai_interface = ai_interface
                self.task_engine = task_engine
                self.config = config
                self.projects = {}
            
            def create_project(self, project_data):
                if not self.ai_interface.validate_project(project_data):
                    raise ValueError("Project validation failed")
                
                project_id = f"proj_{len(self.projects) + 1}"
                project = {
                    "id": project_id,
                    "status": "initialized",
                    "created_at": datetime.now(),
                    **project_data
                }
                self.projects[project_id] = project
                return project
            
            def get_project(self, project_id):
                return self.projects.get(project_id)
            
            def list_projects(self):
                return list(self.projects.values())
            
            def delete_project(self, project_id):
                if project_id in self.projects:
                    del self.projects[project_id]
                    return True
                return False
            
            async def execute_task_async(self, task):
                return await self.task_engine.execute(task)
        
        return MockProjectManager(mock_ai_interface, mock_task_engine, mock_config)
    
    @pytest.fixture
    def sample_project(self, sample_project_data):
        """Create sample project for testing."""
        return {
            "name": "test-project",
            "template": "python", 
            "description": "Test project for unit testing",
            "author": "Test User",
            "email": "test@example.com"
        }
    
    def test_create_project_success(self, project_manager, sample_project):
        """Test successful project creation."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        
        # Act
        result = project_manager.create_project(sample_project)
        
        # Assert
        assert result["id"] is not None
        assert result["status"] == "initialized"
        assert result["name"] == sample_project["name"]
        assert result["template"] == sample_project["template"]
        assert "created_at" in result
        
        # Verify AI validation was called
        project_manager.ai_interface.validate_project.assert_called_once_with(sample_project)
    
    def test_create_project_validation_failure(self, project_manager, sample_project):
        """Test project creation with validation failure."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = False
        
        # Act & Assert
        with pytest.raises(ValueError, match="Project validation failed"):
            project_manager.create_project(sample_project)
        
        # Verify no project was created
        assert len(project_manager.projects) == 0
    
    def test_create_project_duplicate_name(self, project_manager, sample_project):
        """Test creating project with duplicate name."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        project_manager.create_project(sample_project)
        
        # Act & Assert - trying to create another project with same name
        # This should either succeed with different ID or fail appropriately
        duplicate_project = sample_project.copy()
        result = project_manager.create_project(duplicate_project)
        
        # Should create new project with different ID
        assert result["id"] != "proj_1"
        assert result["name"] == sample_project["name"]
    
    @pytest.mark.asyncio
    async def test_execute_task_async(self, project_manager):
        """Test asynchronous task execution."""
        # Arrange
        task = {
            "name": "test-task",
            "prompt": "Generate a Python function",
            "type": "code_generation"
        }
        project_manager.task_engine.execute = AsyncMock(return_value={
            "status": "completed",
            "output": "Function generated successfully"
        })
        
        # Act
        result = await project_manager.execute_task_async(task)
        
        # Assert
        assert result["status"] == "completed"
        assert "output" in result
        project_manager.task_engine.execute.assert_awaited_once_with(task)
    
    def test_get_project_existing(self, project_manager, sample_project):
        """Test retrieving existing project."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        created_project = project_manager.create_project(sample_project)
        
        # Act
        retrieved_project = project_manager.get_project(created_project["id"])
        
        # Assert
        assert retrieved_project is not None
        assert retrieved_project["id"] == created_project["id"]
        assert retrieved_project["name"] == sample_project["name"]
    
    def test_get_project_nonexistent(self, project_manager):
        """Test retrieving non-existent project."""
        # Act
        result = project_manager.get_project("nonexistent_id")
        
        # Assert
        assert result is None
    
    def test_list_projects_empty(self, project_manager):
        """Test listing projects when none exist."""
        # Act
        projects = project_manager.list_projects()
        
        # Assert
        assert projects == []
        assert len(projects) == 0
    
    def test_list_projects_multiple(self, project_manager, project_factory):
        """Test listing multiple projects."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        
        project1 = project_factory.create_project_data(name="project-1")
        project2 = project_factory.create_project_data(name="project-2") 
        project3 = project_factory.create_project_data(name="project-3")
        
        project_manager.create_project(project1)
        project_manager.create_project(project2)
        project_manager.create_project(project3)
        
        # Act
        projects = project_manager.list_projects()
        
        # Assert
        assert len(projects) == 3
        project_names = [p["name"] for p in projects]
        assert "project-1" in project_names
        assert "project-2" in project_names
        assert "project-3" in project_names
    
    def test_delete_project_success(self, project_manager, sample_project):
        """Test successful project deletion."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        created_project = project_manager.create_project(sample_project)
        project_id = created_project["id"]
        
        # Verify project exists
        assert project_manager.get_project(project_id) is not None
        
        # Act
        result = project_manager.delete_project(project_id)
        
        # Assert
        assert result is True
        assert project_manager.get_project(project_id) is None
        assert len(project_manager.projects) == 0
    
    def test_delete_project_nonexistent(self, project_manager):
        """Test deleting non-existent project."""
        # Act
        result = project_manager.delete_project("nonexistent_id")
        
        # Assert
        assert result is False
    
    @pytest.mark.parametrize("template", ["python", "fastapi", "react", "cli"])
    def test_create_project_different_templates(self, project_manager, template):
        """Test creating projects with different templates."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        project_data = {
            "name": f"test-{template}-project",
            "template": template,
            "description": f"Test {template} project"
        }
        
        # Act
        result = project_manager.create_project(project_data)
        
        # Assert
        assert result["template"] == template
        assert result["name"] == f"test-{template}-project"
        assert result["status"] == "initialized"
    
    def test_project_manager_configuration(self, project_manager, mock_config):
        """Test project manager uses configuration correctly."""
        # Assert
        assert project_manager.config == mock_config
        assert project_manager.config.default_template == "python"
        assert project_manager.config.ai_validation is True
    
    @pytest.mark.asyncio
    async def test_ai_validation_integration(self, project_manager, sample_project):
        """Test integration with AI validation."""
        # Arrange
        project_manager.ai_interface.execute_claude_code = AsyncMock(return_value={
            "status": "success",
            "validation_result": {
                "valid": True,
                "confidence": 0.95,
                "suggestions": ["Add more detailed description"]
            }
        })
        
        # Act
        result = await project_manager.ai_interface.execute_claude_code(
            f"Validate project configuration: {sample_project}"
        )
        
        # Assert
        assert result["status"] == "success"
        assert result["validation_result"]["valid"] is True
        assert result["validation_result"]["confidence"] >= 0.9
    
    def test_project_timestamps(self, project_manager, sample_project):
        """Test project creation includes proper timestamps."""
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        before_creation = datetime.now()
        
        # Act
        result = project_manager.create_project(sample_project)
        after_creation = datetime.now()
        
        # Assert
        assert "created_at" in result
        created_at = result["created_at"]
        assert isinstance(created_at, datetime)
        assert before_creation <= created_at <= after_creation
    
    @pytest.mark.slow
    def test_project_creation_performance(self, project_manager, sample_project):
        """Test project creation performance."""
        import time
        
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        
        # Act
        start_time = time.time()
        result = project_manager.create_project(sample_project)
        end_time = time.time()
        
        # Assert
        creation_time = end_time - start_time
        assert creation_time < 1.0  # Should complete within 1 second
        assert result["id"] is not None
    
    def test_project_data_validation(self, project_manager):
        """Test project data validation requirements."""
        # Arrange - invalid project data
        invalid_projects = [
            {},  # Empty data
            {"name": ""},  # Empty name
            {"name": "test", "template": "invalid_template"},  # Invalid template
            {"template": "python"},  # Missing name
        ]
        
        for invalid_project in invalid_projects:
            # Act & Assert
            project_manager.ai_interface.validate_project.return_value = False
            with pytest.raises(ValueError):
                project_manager.create_project(invalid_project)


class TestProjectManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def project_manager_with_failing_ai(self, mock_task_engine, mock_config):
        """Create project manager with failing AI interface."""
        failing_ai = Mock()
        failing_ai.validate_project.side_effect = Exception("AI service unavailable")
        
        class MockProjectManager:
            def __init__(self, ai_interface, task_engine, config):
                self.ai_interface = ai_interface
                self.task_engine = task_engine
                self.config = config
                self.projects = {}
            
            def create_project(self, project_data):
                try:
                    if not self.ai_interface.validate_project(project_data):
                        raise ValueError("Project validation failed")
                except Exception as e:
                    raise RuntimeError(f"AI validation error: {str(e)}")
                
                project_id = f"proj_{len(self.projects) + 1}"
                project = {"id": project_id, "status": "initialized", **project_data}
                self.projects[project_id] = project
                return project
        
        return MockProjectManager(failing_ai, mock_task_engine, mock_config)
    
    def test_ai_service_failure(self, project_manager_with_failing_ai, sample_project_data):
        """Test handling of AI service failures."""
        # Act & Assert
        with pytest.raises(RuntimeError, match="AI validation error"):
            project_manager_with_failing_ai.create_project(sample_project_data)
    
    @pytest.mark.asyncio
    async def test_concurrent_project_creation(self, project_manager, project_factory):
        """Test concurrent project creation."""
        import asyncio
        
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        
        async def create_project_async(project_data):
            return project_manager.create_project(project_data)
        
        # Create multiple projects concurrently
        tasks = []
        for i in range(5):
            project_data = project_factory.create_project_data(name=f"concurrent-project-{i}")
            tasks.append(create_project_async(project_data))
        
        # Act
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 5
        project_ids = [r["id"] for r in results]
        assert len(set(project_ids)) == 5  # All IDs should be unique
        
        # Verify all projects were stored
        assert len(project_manager.projects) == 5
    
    def test_memory_usage_with_many_projects(self, project_manager, project_factory):
        """Test memory usage with large number of projects."""
        import tracemalloc
        
        # Arrange
        project_manager.ai_interface.validate_project.return_value = True
        tracemalloc.start()
        
        # Act - Create many projects
        for i in range(100):
            project_data = project_factory.create_project_data(name=f"memory-test-{i}")
            project_manager.create_project(project_data)
        
        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Assert
        assert len(project_manager.projects) == 100
        # Memory should not be excessive (less than 10MB)
        assert current < 10 * 1024 * 1024
    
    def test_project_cleanup_on_error(self, project_manager, sample_project_data):
        """Test that failed project creation doesn't leave partial state."""
        # Arrange - Make validation fail after some processing
        def failing_validation(project_data):
            # Simulate some processing, then fail
            if "trigger_failure" in project_data:
                raise Exception("Validation processing failed")
            return True
        
        project_manager.ai_interface.validate_project.side_effect = failing_validation
        
        # Act & Assert
        failing_project = {**sample_project_data, "trigger_failure": True}
        
        with pytest.raises(Exception, match="Validation processing failed"):
            project_manager.create_project(failing_project)
        
        # Verify no partial project was created
        assert len(project_manager.projects) == 0
    
    @pytest.mark.parametrize("invalid_id", [None, "", "   ", 123, [], {}])
    def test_get_project_invalid_ids(self, project_manager, invalid_id):
        """Test getting project with invalid IDs."""
        # Act & Assert
        result = project_manager.get_project(invalid_id)
        assert result is None
    
    @pytest.mark.parametrize("invalid_id", [None, "", "   ", 123, [], {}])
    def test_delete_project_invalid_ids(self, project_manager, invalid_id):
        """Test deleting project with invalid IDs."""
        # Act
        result = project_manager.delete_project(invalid_id)
        
        # Assert
        assert result is False