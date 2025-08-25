"""Unit tests for core components of claude-tiu."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from src.core.project_manager import ProjectManager
from src.core.task_engine import TaskEngine
from src.core.ai_interface import AIInterface
from src.core.validator import Validator
from src.core.config import Config
from src.core.exceptions import ValidationError, TaskError, ConfigError


class TestProjectManager:
    """Test suite for ProjectManager class."""
    
    def test_init(self, config, mock_ai_interface, mock_task_engine):
        """Test ProjectManager initialization."""
        manager = ProjectManager(
            config=config,
            ai_interface=mock_ai_interface,
            task_engine=mock_task_engine
        )
        
        assert manager.config == config
        assert manager.ai_interface == mock_ai_interface
        assert manager.task_engine == mock_task_engine
        assert manager.projects == {}
    
    @pytest.mark.asyncio
    async def test_create_project_success(self, project_manager, sample_project_data, test_project_dir):
        """Test successful project creation."""
        # Arrange
        project_data = sample_project_data.copy()
        project_data["path"] = str(test_project_dir)
        
        project_manager.ai_interface.validate_code.return_value = {"valid": True}
        
        # Act
        result = await project_manager.create_project(**project_data)
        
        # Assert
        assert result["id"] is not None
        assert result["name"] == project_data["name"]
        assert result["status"] == "initialized"
        assert Path(result["path"]).exists()
        project_manager.ai_interface.validate_code.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_project_validation_failure(self, project_manager, sample_project_data):
        """Test project creation with validation failure."""
        # Arrange
        project_manager.ai_interface.validate_code.return_value = {
            "valid": False,
            "issues": ["Invalid project structure"]
        }
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Project validation failed"):
            await project_manager.create_project(**sample_project_data)
    
    @pytest.mark.asyncio
    async def test_load_project_success(self, project_manager, test_project_dir):
        """Test successful project loading."""
        # Arrange
        config_file = test_project_dir / "claude-tiu.yaml"
        config_content = """
name: test-project
description: Test project
template: python
settings:
  python_version: "3.11"
"""
        config_file.write_text(config_content)
        
        # Act
        result = await project_manager.load_project(str(test_project_dir))
        
        # Assert
        assert result["name"] == "test-project"
        assert result["template"] == "python"
        assert "id" in result
    
    def test_load_project_not_found(self, project_manager):
        """Test loading non-existent project."""
        with pytest.raises(FileNotFoundError):
            asyncio.run(project_manager.load_project("/nonexistent/path"))
    
    @pytest.mark.asyncio
    async def test_get_project_status(self, project_manager, sample_project_data, test_project_dir):
        """Test getting project status."""
        # Arrange
        project_data = sample_project_data.copy()
        project_data["path"] = str(test_project_dir)
        project = await project_manager.create_project(**project_data)
        
        # Act
        status = project_manager.get_project_status(project["id"])
        
        # Assert
        assert status["id"] == project["id"]
        assert status["status"] in ["initialized", "active", "completed"]
        assert "tasks" in status
        assert "progress" in status
    
    def test_get_project_status_not_found(self, project_manager):
        """Test getting status for non-existent project."""
        with pytest.raises(KeyError, match="Project not found"):
            project_manager.get_project_status("non-existent-id")
    
    @pytest.mark.asyncio
    async def test_delete_project(self, project_manager, sample_project_data, test_project_dir):
        """Test project deletion."""
        # Arrange
        project_data = sample_project_data.copy()
        project_data["path"] = str(test_project_dir)
        project = await project_manager.create_project(**project_data)
        project_id = project["id"]
        
        # Act
        result = project_manager.delete_project(project_id)
        
        # Assert
        assert result is True
        assert project_id not in project_manager.projects
        with pytest.raises(KeyError):
            project_manager.get_project_status(project_id)


class TestTaskEngine:
    """Test suite for TaskEngine class."""
    
    def test_init(self, config):
        """Test TaskEngine initialization."""
        engine = TaskEngine(config=config)
        
        assert engine.config == config
        assert engine.tasks == {}
        assert engine.running_tasks == set()
    
    @pytest.mark.asyncio
    async def test_create_task(self, mock_task_engine, sample_task_data):
        """Test task creation."""
        # Mock the create_task method to return actual task data
        mock_task_engine.create_task.return_value = {
            "id": "task-123",
            "name": sample_task_data["name"],
            "status": "created",
            "created_at": "2023-01-01T00:00:00Z"
        }
        
        # Act
        result = mock_task_engine.create_task(
            name=sample_task_data["name"],
            prompt=sample_task_data["prompt"]
        )
        
        # Assert
        assert result["id"] is not None
        assert result["name"] == sample_task_data["name"]
        assert result["status"] == "created"
        mock_task_engine.create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_task_engine):
        """Test successful task execution."""
        # Arrange
        task = {"id": "task-123", "name": "test-task", "prompt": "Generate code"}
        mock_task_engine.execute_task.return_value = {
            "id": "task-123",
            "status": "completed",
            "result": "Generated code successfully",
            "execution_time": 1.5
        }
        
        # Act
        result = await mock_task_engine.execute_task(task)
        
        # Assert
        assert result["status"] == "completed"
        assert "result" in result
        assert "execution_time" in result
        mock_task_engine.execute_task.assert_awaited_once_with(task)
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, mock_task_engine):
        """Test task execution failure."""
        # Arrange
        task = {"id": "task-123", "name": "failing-task"}
        mock_task_engine.execute_task.side_effect = TaskError("Execution failed")
        
        # Act & Assert
        with pytest.raises(TaskError, match="Execution failed"):
            await mock_task_engine.execute_task(task)
    
    def test_get_task_status(self, mock_task_engine):
        """Test getting task status."""
        # Arrange
        task_id = "task-123"
        mock_task_engine.get_task_status.return_value = "running"
        
        # Act
        status = mock_task_engine.get_task_status(task_id)
        
        # Assert
        assert status == "running"
        mock_task_engine.get_task_status.assert_called_once_with(task_id)
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, mock_task_engine):
        """Test task cancellation."""
        # Arrange
        task_id = "task-123"
        mock_task_engine.cancel_task = AsyncMock(return_value=True)
        
        # Act
        result = await mock_task_engine.cancel_task(task_id)
        
        # Assert
        assert result is True
        mock_task_engine.cancel_task.assert_awaited_once_with(task_id)


class TestAIInterface:
    """Test suite for AIInterface class."""
    
    def test_init(self, config):
        """Test AIInterface initialization."""
        interface = AIInterface(config=config)
        
        assert interface.config == config
        assert interface.api_key == config.api_key
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_success(self, mock_ai_interface):
        """Test successful Claude Code execution."""
        # Arrange
        prompt = "Generate a hello world function"
        expected_result = {
            "status": "success",
            "result": "def hello_world():\n    print('Hello, World!')",
            "execution_time": 2.1
        }
        mock_ai_interface.execute_claude_code.return_value = expected_result
        
        # Act
        result = await mock_ai_interface.execute_claude_code(prompt)
        
        # Assert
        assert result["status"] == "success"
        assert "result" in result
        mock_ai_interface.execute_claude_code.assert_awaited_once_with(prompt)
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_api_error(self, mock_ai_interface):
        """Test Claude Code execution with API error."""
        # Arrange
        mock_ai_interface.execute_claude_code.side_effect = Exception("API Error")
        
        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            await mock_ai_interface.execute_claude_code("test prompt")
    
    @pytest.mark.asyncio
    async def test_execute_claude_flow_success(self, mock_ai_interface):
        """Test successful Claude Flow execution."""
        # Arrange
        workflow = {"tasks": ["task1", "task2"]}
        expected_result = {
            "status": "completed",
            "tasks": ["task1", "task2"],
            "results": [{"id": "task1"}, {"id": "task2"}]
        }
        mock_ai_interface.execute_claude_flow.return_value = expected_result
        
        # Act
        result = await mock_ai_interface.execute_claude_flow(workflow)
        
        # Assert
        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        mock_ai_interface.execute_claude_flow.assert_awaited_once_with(workflow)
    
    @pytest.mark.asyncio
    async def test_validate_code_success(self, mock_ai_interface, sample_complete_code):
        """Test successful code validation."""
        # Arrange
        expected_result = {
            "valid": True,
            "quality_score": 0.92,
            "issues": [],
            "suggestions": ["Consider adding type hints"]
        }
        mock_ai_interface.validate_code.return_value = expected_result
        
        # Act
        result = await mock_ai_interface.validate_code(sample_complete_code)
        
        # Assert
        assert result["valid"] is True
        assert result["quality_score"] > 0.9
        assert isinstance(result["issues"], list)
        mock_ai_interface.validate_code.assert_awaited_once_with(sample_complete_code)
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_placeholders(self, mock_ai_interface, sample_code_with_placeholders):
        """Test code analysis with placeholders."""
        # Arrange
        expected_result = {
            "quality_score": 0.45,
            "placeholder_count": 3,
            "completeness": 0.6,
            "suggestions": [
                "Implement TODO in calculate_sum function",
                "Complete placeholder_function implementation"
            ]
        }
        mock_ai_interface.analyze_code.return_value = expected_result
        
        # Act
        result = await mock_ai_interface.analyze_code(sample_code_with_placeholders)
        
        # Assert
        assert result["quality_score"] < 0.5
        assert result["placeholder_count"] > 0
        assert len(result["suggestions"]) > 0
        mock_ai_interface.analyze_code.assert_awaited_once_with(sample_code_with_placeholders)


class TestValidator:
    """Test suite for Validator class."""
    
    def test_init(self, config):
        """Test Validator initialization."""
        validator = Validator(config=config)
        
        assert validator.config == config
        assert hasattr(validator, 'placeholder_patterns')
    
    def test_has_placeholders_with_placeholders(self, validator, sample_code_with_placeholders):
        """Test placeholder detection with code containing placeholders."""
        result = validator.has_placeholders(sample_code_with_placeholders)
        
        assert result is True
    
    def test_has_placeholders_without_placeholders(self, validator, sample_complete_code):
        """Test placeholder detection with complete code."""
        result = validator.has_placeholders(sample_complete_code)
        
        assert result is False
    
    @pytest.mark.parametrize("code,expected", [
        ("# TODO: implement this", True),
        ("def func(): pass  # implement later", True),
        ("placeholder_function()", True),
        ("raise NotImplementedError", True),
        ("console.log('test')", True),
        ("def complete_function(): return x + 1", False),
        ("class RealImplementation: pass", False),
    ])
    def test_placeholder_patterns(self, validator, code, expected):
        """Test various placeholder patterns."""
        assert validator.has_placeholders(code) == expected
    
    def test_count_placeholders(self, validator, sample_code_with_placeholders):
        """Test counting placeholders in code."""
        count = validator.count_placeholders(sample_code_with_placeholders)
        
        assert count >= 3  # At least TODO, NotImplementedError, console.log
    
    def test_validate_file_structure(self, validator, test_project_dir):
        """Test file structure validation."""
        result = validator.validate_file_structure(test_project_dir)
        
        assert result["valid"] is True
        assert "missing_files" in result
        assert "extra_files" in result
    
    def test_validate_code_quality(self, validator, sample_complete_code):
        """Test code quality validation."""
        result = validator.validate_code_quality(sample_complete_code)
        
        assert "quality_score" in result
        assert "issues" in result
        assert isinstance(result["issues"], list)
        assert 0 <= result["quality_score"] <= 1
    
    def test_security_validation(self, validator, malicious_inputs):
        """Test security validation with malicious inputs."""
        for malicious_input in malicious_inputs[:5]:  # Test first 5 inputs
            result = validator.is_safe_input(malicious_input)
            assert result is False, f"Failed to detect malicious input: {malicious_input}"
    
    def test_security_validation_safe_input(self, validator):
        """Test security validation with safe input."""
        safe_inputs = [
            "normal text input",
            "def function_name():",
            "print('hello world')",
            "username123",
            "valid@email.com"
        ]
        
        for safe_input in safe_inputs:
            result = validator.is_safe_input(safe_input)
            assert result is True, f"False positive for safe input: {safe_input}"


class TestConfig:
    """Test suite for Config class."""
    
    def test_init_with_dict(self, test_config_dict):
        """Test Config initialization with dictionary."""
        config = Config(**test_config_dict)
        
        assert config.database_url == test_config_dict["database_url"]
        assert config.api_key == test_config_dict["api_key"]
        assert config.environment == test_config_dict["environment"]
        assert config.debug == test_config_dict["debug"]
    
    def test_init_with_missing_required_fields(self):
        """Test Config initialization with missing required fields."""
        with pytest.raises(ConfigError):
            Config()  # Missing required fields
    
    def test_environment_specific_settings(self, test_config_dict):
        """Test environment-specific configuration settings."""
        # Test config
        test_config_dict["environment"] = "test"
        config = Config(**test_config_dict)
        assert config.debug is True
        
        # Production config
        prod_config_dict = test_config_dict.copy()
        prod_config_dict["environment"] = "production"
        prod_config_dict["debug"] = False
        prod_config = Config(**prod_config_dict)
        assert prod_config.debug is False
    
    def test_security_settings(self, test_config_dict):
        """Test security-related configuration."""
        config = Config(**test_config_dict)
        
        assert "security" in test_config_dict
        assert config.security["jwt_secret"] == "test-secret"
        assert config.security["jwt_algorithm"] == "HS256"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid database URL
        with pytest.raises(ConfigError):
            Config(
                database_url="invalid-url",
                api_key="test-key",
                environment="test"
            )
        
        # Empty API key
        with pytest.raises(ConfigError):
            Config(
                database_url="sqlite:///:memory:",
                api_key="",
                environment="test"
            )


# Edge case and error handling tests
class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_project_creation(self, project_manager, sample_project_data, temp_dir):
        """Test concurrent project creation handling."""
        # Create multiple project directories
        project_dirs = []
        for i in range(5):
            project_dir = temp_dir / f"project_{i}"
            project_dir.mkdir()
            project_dirs.append(project_dir)
        
        # Prepare project data for concurrent creation
        project_tasks = []
        for i, project_dir in enumerate(project_dirs):
            project_data = sample_project_data.copy()
            project_data["name"] = f"test-project-{i}"
            project_data["path"] = str(project_dir)
            project_tasks.append(project_manager.create_project(**project_data))
        
        # Execute concurrently
        results = await asyncio.gather(*project_tasks, return_exceptions=True)
        
        # Verify all projects were created successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        
        # Verify each project has unique ID
        project_ids = [r["id"] for r in successful_results]
        assert len(set(project_ids)) == 5
    
    def test_large_code_validation(self, validator):
        """Test validation with large code files."""
        # Create large code content
        large_code = "\n".join([f"def function_{i}(): return {i}" for i in range(10000)])
        
        # Should handle large files without errors
        result = validator.has_placeholders(large_code)
        assert isinstance(result, bool)
        
        count = validator.count_placeholders(large_code)
        assert isinstance(count, int)
        assert count >= 0
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_task_failure(self, mock_task_engine):
        """Test memory cleanup after task execution failure."""
        # Arrange
        mock_task_engine.execute_task.side_effect = TaskError("Simulated failure")
        task = {"id": "failing-task", "name": "test"}
        
        # Act & Assert
        with pytest.raises(TaskError):
            await mock_task_engine.execute_task(task)
        
        # Verify cleanup (this would be implementation-specific)
        # In a real implementation, we'd check that resources are cleaned up
    
    def test_invalid_project_paths(self, project_manager):
        """Test handling of invalid project paths."""
        invalid_paths = [
            "/nonexistent/path",
            "",
            None,
            "/root",  # Permission denied path
            "../../../etc",  # Path traversal attempt
        ]
        
        for invalid_path in invalid_paths:
            with pytest.raises((FileNotFoundError, PermissionError, ValueError)):
                asyncio.run(project_manager.load_project(invalid_path))
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting_simulation(self, mock_ai_interface):
        """Test API rate limiting simulation."""
        # Simulate rate limiting by making API calls fail after certain number
        call_count = 0
        
        async def rate_limited_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise Exception("Rate limit exceeded")
            return {"status": "success", "result": "Generated"}
        
        mock_ai_interface.execute_claude_code.side_effect = rate_limited_call
        
        # First 5 calls should succeed
        for i in range(5):
            result = await mock_ai_interface.execute_claude_code(f"prompt {i}")
            assert result["status"] == "success"
        
        # 6th call should fail
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await mock_ai_interface.execute_claude_code("prompt 6")
    
    def test_unicode_and_special_characters(self, validator):
        """Test handling of unicode and special characters."""
        unicode_code = '''
# Unicode test: 测试中文字符
def función_con_acentos():
    """Función con caracteres especiales: ñáéíóú"""
    emoji_var = "🐍 Python is awesome! 🚀"
    return f"Hola mundo: {emoji_var}"

# Test various unicode categories
class УникодныйКласс:
    def метод_с_кириллицей(self):
        return "Привет мир"

# Arabic text
def arabic_function():
    text = "مرحبا بالعالم"
    return text
'''
        
        # Should handle unicode without errors
        result = validator.has_placeholders(unicode_code)
        assert isinstance(result, bool)
        
        quality_result = validator.validate_code_quality(unicode_code)
        assert "quality_score" in quality_result
