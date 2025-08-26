"""
TDD London School Tests for ProjectManager

Following London School (mockist) approach:
- Outside-in development flow 
- Mock-driven development for isolation
- Behavior verification over state testing
- Focus on object interactions and collaborations
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List

from src.claude_tui.core.project_manager import (
    ProjectManager, 
    DevelopmentResult
)
from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.core.state_manager import StateManager
from src.claude_tui.core.task_engine import TaskEngine
from src.claude_tui.core.progress_validator import ProgressValidator
from src.claude_tui.integrations.ai_interface import AIInterface
from src.claude_tui.models.project import Project, ProjectConfig, ProjectTemplate
from src.claude_tui.models.task import DevelopmentTask, TaskResult


# Mock collaborator contracts - London School emphasis on defining interfaces
@pytest.fixture
def mock_config_manager():
    """Mock configuration manager with defined contract"""
    mock = Mock(spec=ConfigManager)
    mock.get_config = Mock(return_value={'test': 'config'})
    mock.update_config = Mock()
    return mock


@pytest.fixture
def mock_state_manager():
    """Mock state manager following London School contract definition"""
    mock = AsyncMock(spec=StateManager)
    mock.initialize_project = AsyncMock()
    mock.save_project = AsyncMock()
    mock.load_project = AsyncMock()
    return mock


@pytest.fixture
def mock_task_engine():
    """Mock task engine with behavior expectations"""
    mock = AsyncMock(spec=TaskEngine)
    mock.execute_tasks = AsyncMock()
    mock.get_status = AsyncMock(return_value={'running': 0, 'completed': 5})
    mock.cleanup = AsyncMock()
    return mock


@pytest.fixture
def mock_ai_interface():
    """Mock AI interface for testing object collaborations"""
    mock = AsyncMock(spec=AIInterface)
    mock.execute_development_task = AsyncMock()
    mock.analyze_requirements = AsyncMock()
    mock.correct_task_result = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock


@pytest.fixture
def mock_validator():
    """Mock validator with contract expectations"""
    mock = AsyncMock(spec=ProgressValidator)
    mock.validate_project = AsyncMock()
    mock.validate_task_result = AsyncMock()
    mock.auto_fix_issue = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock


@pytest.fixture
def mock_template_engine():
    """Mock template engine for project creation workflow"""
    mock = AsyncMock()
    mock.load_template = AsyncMock()
    return mock


@pytest.fixture
def mock_file_system():
    """Mock file system manager"""
    mock = AsyncMock()
    mock.create_directory_structure = AsyncMock()
    mock.cleanup_directory = AsyncMock()
    mock.get_directory_stats = AsyncMock(return_value={'files': 10, 'size': '1MB'})
    return mock


@pytest.fixture
def project_manager_with_mocks(
    mock_config_manager,
    mock_state_manager, 
    mock_task_engine,
    mock_ai_interface,
    mock_validator
):
    """ProjectManager instance with all dependencies mocked"""
    return ProjectManager(
        config_manager=mock_config_manager,
        state_manager=mock_state_manager,
        task_engine=mock_task_engine,
        ai_interface=mock_ai_interface,
        validator=mock_validator
    )


class TestProjectManagerInitialization:
    """Test ProjectManager initialization - London School contract verification"""
    
    def test_should_initialize_with_required_dependencies(self, mock_config_manager):
        """Verify ProjectManager establishes correct collaborator relationships"""
        # Act
        project_manager = ProjectManager(config_manager=mock_config_manager)
        
        # Assert - Verify object collaboration setup
        assert project_manager.config_manager == mock_config_manager
        assert project_manager.state_manager is not None
        assert project_manager.task_engine is not None  
        assert project_manager.ai_interface is not None
        assert project_manager.validator is not None
        assert project_manager.current_project is None
        
    def test_should_use_provided_collaborators_over_defaults(self, project_manager_with_mocks):
        """Verify dependency injection follows proper collaboration pattern"""
        # Assert - Verify all mocks are properly injected
        assert isinstance(project_manager_with_mocks.state_manager, AsyncMock)
        assert isinstance(project_manager_with_mocks.task_engine, AsyncMock)
        assert isinstance(project_manager_with_mocks.ai_interface, AsyncMock)
        assert isinstance(project_manager_with_mocks.validator, AsyncMock)


class TestProjectCreationWorkflow:
    """Test project creation workflow - London School outside-in approach"""
    
    @pytest.mark.asyncio
    async def test_should_orchestrate_complete_project_creation_workflow(
        self,
        project_manager_with_mocks,
        mock_template_engine,
        mock_file_system
    ):
        """Test complete project creation conversation between objects"""
        
        # Arrange - Set up mock expectations for the workflow
        template_name = "react-app"
        project_name = "my-project"
        output_dir = Path("/tmp/projects")
        
        mock_template = Mock(spec=ProjectTemplate)
        mock_template.name = template_name
        mock_template.structure = {'src': {}, 'tests': {}}
        mock_template.files = []
        
        mock_project = Mock(spec=Project)
        mock_project.name = project_name
        mock_project.path = output_dir / project_name
        
        # Mock the collaboration sequence
        with patch.object(project_manager_with_mocks, 'template_engine', mock_template_engine), \
             patch.object(project_manager_with_mocks, 'file_system', mock_file_system):
            
            mock_template_engine.load_template.return_value = mock_template
            project_manager_with_mocks.task_engine.execute_tasks.return_value = []
            
            # Mock validation success
            validation_result = Mock()
            validation_result.is_valid = True
            validation_result.issues = []
            project_manager_with_mocks.validator.validate_project.return_value = validation_result
            
            # Act
            with patch('src.claude_tui.core.project_manager.Project') as mock_project_class:
                mock_project_class.return_value = mock_project
                
                result = await project_manager_with_mocks.create_project(
                    template_name=template_name,
                    project_name=project_name,
                    output_directory=output_dir
                )
        
        # Assert - Verify the complete workflow conversation
        # Step 1: Template loading
        mock_template_engine.load_template.assert_called_once_with(template_name)
        
        # Step 2: Directory structure creation  
        mock_file_system.create_directory_structure.assert_called_once()
        
        # Step 3: Task execution
        project_manager_with_mocks.task_engine.execute_tasks.assert_called_once()
        
        # Step 4: Project state initialization
        project_manager_with_mocks.state_manager.initialize_project.assert_called_once()
        
        # Step 5: Validation
        project_manager_with_mocks.validator.validate_project.assert_called_once()
        
        # Verify result
        assert result == mock_project
        assert project_manager_with_mocks.current_project == mock_project
    
    @pytest.mark.asyncio 
    async def test_should_handle_template_not_found_gracefully(
        self,
        project_manager_with_mocks,
        mock_template_engine
    ):
        """Test error handling collaboration when template doesn't exist"""
        
        # Arrange
        with patch.object(project_manager_with_mocks, 'template_engine', mock_template_engine):
            mock_template_engine.load_template.return_value = None
            
            # Act & Assert
            with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
                await project_manager_with_mocks.create_project(
                    template_name="nonexistent",
                    project_name="test-project", 
                    output_directory=Path("/tmp")
                )
        
        # Verify template engine was called
        mock_template_engine.load_template.assert_called_once_with("nonexistent")
    
    @pytest.mark.asyncio
    async def test_should_cleanup_on_creation_failure(
        self,
        project_manager_with_mocks,
        mock_template_engine,
        mock_file_system
    ):
        """Test cleanup behavior when project creation fails"""
        
        # Arrange - Force failure during task execution
        mock_template = Mock(spec=ProjectTemplate)
        mock_template.structure = {}
        mock_template.files = []
        
        with patch.object(project_manager_with_mocks, 'template_engine', mock_template_engine), \
             patch.object(project_manager_with_mocks, 'file_system', mock_file_system):
            
            mock_template_engine.load_template.return_value = mock_template
            project_manager_with_mocks.task_engine.execute_tasks.side_effect = Exception("Task failed")
            
            # Act & Assert
            with pytest.raises(Exception):
                await project_manager_with_mocks.create_project(
                    template_name="test-template",
                    project_name="test-project",
                    output_directory=Path("/tmp")
                )
        
        # Verify cleanup was called
        mock_file_system.cleanup_directory.assert_called_once()


class TestDevelopmentOrchestration:
    """Test development workflow orchestration - London School behavior focus"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_development_workflow_with_ai_and_validation(
        self,
        project_manager_with_mocks
    ):
        """Test complete development workflow object interactions"""
        
        # Arrange
        requirements = {"feature": "user authentication", "priority": "high"}
        mock_project = Mock(spec=Project)
        mock_project.name = "test-project"
        project_manager_with_mocks.current_project = mock_project
        
        # Mock task breakdown
        mock_tasks = [
            Mock(spec=DevelopmentTask, name="implement_auth"),
            Mock(spec=DevelopmentTask, name="add_tests")
        ]
        
        # Mock analysis result
        analysis_result = Mock()
        analysis_result.tasks = mock_tasks
        project_manager_with_mocks.ai_interface.analyze_requirements.return_value = analysis_result
        
        # Mock task execution results
        mock_task_result = Mock(spec=TaskResult)
        project_manager_with_mocks.ai_interface.execute_development_task.return_value = mock_task_result
        
        # Mock validation success
        validation_result = Mock()
        validation_result.is_valid = True
        project_manager_with_mocks.validator.validate_task_result.return_value = validation_result
        
        # Mock final project validation
        final_validation = Mock()
        final_validation.is_valid = True
        final_validation.to_dict = Mock(return_value={'status': 'valid'})
        project_manager_with_mocks.validator.validate_project.return_value = final_validation
        
        # Act
        result = await project_manager_with_mocks.orchestrate_development(requirements)
        
        # Assert - Verify the complete workflow conversation
        
        # Step 1: Requirements analysis
        project_manager_with_mocks.ai_interface.analyze_requirements.assert_called_once_with(
            requirements=requirements,
            project=mock_project
        )
        
        # Step 2: Task execution for each task
        assert project_manager_with_mocks.ai_interface.execute_development_task.call_count == len(mock_tasks)
        
        # Step 3: Task validation for each result
        assert project_manager_with_mocks.validator.validate_task_result.call_count == len(mock_tasks)
        
        # Step 4: Final project validation
        project_manager_with_mocks.validator.validate_project.assert_called_once_with(mock_project)
        
        # Verify result structure
        assert isinstance(result, DevelopmentResult)
        assert result.success is True
        assert result.project == mock_project
        assert len(result.completed_tasks) == len(mock_tasks)
        assert len(result.failed_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_should_attempt_auto_correction_on_validation_failure(
        self,
        project_manager_with_mocks
    ):
        """Test auto-correction workflow when validation fails"""
        
        # Arrange
        requirements = {"feature": "broken_feature"}
        mock_project = Mock(spec=Project)
        project_manager_with_mocks.current_project = mock_project
        
        mock_task = Mock(spec=DevelopmentTask, name="broken_task")
        analysis_result = Mock()
        analysis_result.tasks = [mock_task]
        project_manager_with_mocks.ai_interface.analyze_requirements.return_value = analysis_result
        
        # Mock initial task result (will fail validation)
        failed_result = Mock(spec=TaskResult)
        project_manager_with_mocks.ai_interface.execute_development_task.return_value = failed_result
        
        # Mock validation failure, then success after correction
        failed_validation = Mock()
        failed_validation.is_valid = False
        failed_validation.issues = ["syntax error"]
        
        success_validation = Mock()  
        success_validation.is_valid = True
        
        project_manager_with_mocks.validator.validate_task_result.side_effect = [
            failed_validation,  # First validation fails
            success_validation  # After correction succeeds
        ]
        
        # Mock successful correction
        corrected_result = Mock(spec=TaskResult)
        project_manager_with_mocks.ai_interface.correct_task_result.return_value = corrected_result
        
        # Mock final validation
        final_validation = Mock()
        final_validation.is_valid = True 
        final_validation.to_dict = Mock(return_value={'status': 'valid'})
        project_manager_with_mocks.validator.validate_project.return_value = final_validation
        
        # Act
        result = await project_manager_with_mocks.orchestrate_development(requirements)
        
        # Assert - Verify auto-correction workflow
        
        # Initial validation should fail
        assert project_manager_with_mocks.validator.validate_task_result.call_count == 2
        
        # Auto-correction should be attempted
        project_manager_with_mocks.ai_interface.correct_task_result.assert_called_once_with(
            task=mock_task,
            result=failed_result,
            validation_issues=failed_validation.issues
        )
        
        # Task should end up in completed list after correction
        assert len(result.completed_tasks) == 1
        assert len(result.failed_tasks) == 0
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_should_fail_gracefully_when_no_project_available(
        self,
        project_manager_with_mocks
    ):
        """Test error handling when no project is set"""
        
        # Arrange - No current project
        project_manager_with_mocks.current_project = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="No project specified and no current project set"):
            await project_manager_with_mocks.orchestrate_development({"feature": "test"})


class TestProjectStateManagement:
    """Test project state management operations - London School focus"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_project_loading_with_validation(
        self,
        project_manager_with_mocks
    ):
        """Test project loading workflow with validation"""
        
        # Arrange
        project_path = Path("/projects/existing-project")
        mock_project = Mock(spec=Project)
        mock_project.name = "existing-project"
        
        project_manager_with_mocks.state_manager.load_project.return_value = mock_project
        
        validation_result = Mock()
        validation_result.is_valid = True
        validation_result.issues = []
        project_manager_with_mocks.validator.validate_project.return_value = validation_result
        
        # Act
        result = await project_manager_with_mocks.load_project(project_path)
        
        # Assert - Verify loading and validation sequence
        project_manager_with_mocks.state_manager.load_project.assert_called_once_with(project_path)
        project_manager_with_mocks.validator.validate_project.assert_called_once_with(mock_project)
        
        assert result == mock_project
        assert project_manager_with_mocks.current_project == mock_project
    
    @pytest.mark.asyncio
    async def test_should_save_current_project_state(
        self,
        project_manager_with_mocks
    ):
        """Test project saving workflow"""
        
        # Arrange
        mock_project = Mock(spec=Project)
        project_manager_with_mocks.current_project = mock_project
        
        # Act
        await project_manager_with_mocks.save_project()
        
        # Assert
        project_manager_with_mocks.state_manager.save_project.assert_called_once_with(mock_project)
    
    @pytest.mark.asyncio
    async def test_should_raise_error_when_saving_without_project(
        self,
        project_manager_with_mocks
    ):
        """Test error handling when no project to save"""
        
        # Arrange - No current project
        project_manager_with_mocks.current_project = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="No project to save"):
            await project_manager_with_mocks.save_project()


class TestProjectStatusReporting:
    """Test project status reporting - London School behavior verification"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_comprehensive_status_gathering(
        self,
        project_manager_with_mocks,
        mock_file_system
    ):
        """Test status gathering from all collaborators"""
        
        # Arrange
        mock_project = Mock(spec=Project)
        mock_project.name = "test-project"
        mock_project.path = Path("/projects/test")
        mock_project.created_at = datetime(2024, 1, 1)
        mock_project.template = Mock()
        mock_project.template.name = "react-app"
        
        project_manager_with_mocks.current_project = mock_project
        
        # Mock validation status
        validation_result = Mock()
        validation_result.to_dict = Mock(return_value={'status': 'valid', 'issues': []})
        project_manager_with_mocks.validator.validate_project.return_value = validation_result
        
        # Mock task status
        task_status = {'running': 2, 'completed': 10}
        project_manager_with_mocks.task_engine.get_status.return_value = task_status
        
        # Mock filesystem stats
        fs_stats = {'files': 25, 'size': '5MB'}
        
        with patch.object(project_manager_with_mocks, 'file_system', mock_file_system):
            mock_file_system.get_directory_stats.return_value = fs_stats
            
            # Act
            status = await project_manager_with_mocks.get_project_status()
        
        # Assert - Verify collaboration with all components
        project_manager_with_mocks.validator.validate_project.assert_called_once_with(mock_project)
        project_manager_with_mocks.task_engine.get_status.assert_called_once()
        mock_file_system.get_directory_stats.assert_called_once_with(mock_project.path)
        
        # Verify status structure
        assert 'project' in status
        assert 'validation' in status
        assert 'tasks' in status
        assert 'filesystem' in status
        assert status['project']['name'] == "test-project"
        assert status['validation'] == {'status': 'valid', 'issues': []}
        assert status['tasks'] == task_status
        assert status['filesystem'] == fs_stats


class TestResourceCleanup:
    """Test resource cleanup - London School contract fulfillment"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_complete_cleanup_workflow(
        self,
        project_manager_with_mocks
    ):
        """Test cleanup coordination with all collaborators"""
        
        # Arrange
        mock_project = Mock(spec=Project)
        project_manager_with_mocks.current_project = mock_project
        
        # Add some background tasks
        background_task1 = Mock()
        background_task2 = Mock()
        project_manager_with_mocks._background_tasks.add(background_task1)
        project_manager_with_mocks._background_tasks.add(background_task2)
        
        # Act
        await project_manager_with_mocks.cleanup()
        
        # Assert - Verify complete cleanup sequence
        
        # Background tasks should be cancelled
        background_task1.cancel.assert_called_once()
        background_task2.cancel.assert_called_once()
        
        # Current project should be saved
        project_manager_with_mocks.state_manager.save_project.assert_called_once_with(mock_project)
        
        # All components should be cleaned up
        project_manager_with_mocks.task_engine.cleanup.assert_called_once()
        project_manager_with_mocks.ai_interface.cleanup.assert_called_once()
        project_manager_with_mocks.validator.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_should_cleanup_without_current_project(
        self,
        project_manager_with_mocks
    ):
        """Test cleanup when no current project exists"""
        
        # Arrange - No current project
        project_manager_with_mocks.current_project = None
        
        # Act
        await project_manager_with_mocks.cleanup()
        
        # Assert - Save shouldn't be called but other cleanup should happen
        project_manager_with_mocks.state_manager.save_project.assert_not_called()
        project_manager_with_mocks.task_engine.cleanup.assert_called_once()
        project_manager_with_mocks.ai_interface.cleanup.assert_called_once()
        project_manager_with_mocks.validator.cleanup.assert_called_once()


class TestContractVerification:
    """Test contract verification - London School emphasis on interfaces"""
    
    def test_project_manager_should_satisfy_expected_interface(self, project_manager_with_mocks):
        """Verify ProjectManager exposes expected public interface"""
        
        # Assert public methods exist
        assert hasattr(project_manager_with_mocks, 'create_project')
        assert hasattr(project_manager_with_mocks, 'orchestrate_development')
        assert hasattr(project_manager_with_mocks, 'load_project')
        assert hasattr(project_manager_with_mocks, 'save_project')
        assert hasattr(project_manager_with_mocks, 'get_project_status')
        assert hasattr(project_manager_with_mocks, 'cleanup')
        
        # Verify methods are callable
        assert callable(getattr(project_manager_with_mocks, 'create_project'))
        assert callable(getattr(project_manager_with_mocks, 'orchestrate_development'))
        assert callable(getattr(project_manager_with_mocks, 'load_project'))
        assert callable(getattr(project_manager_with_mocks, 'save_project'))
        assert callable(getattr(project_manager_with_mocks, 'get_project_status'))
        assert callable(getattr(project_manager_with_mocks, 'cleanup'))
    
    def test_should_maintain_collaborator_contracts(self, project_manager_with_mocks):
        """Verify ProjectManager maintains proper contracts with collaborators"""
        
        # Verify collaborator types match expected interfaces
        assert hasattr(project_manager_with_mocks.state_manager, 'initialize_project')
        assert hasattr(project_manager_with_mocks.state_manager, 'load_project') 
        assert hasattr(project_manager_with_mocks.state_manager, 'save_project')
        
        assert hasattr(project_manager_with_mocks.task_engine, 'execute_tasks')
        assert hasattr(project_manager_with_mocks.task_engine, 'get_status')
        assert hasattr(project_manager_with_mocks.task_engine, 'cleanup')
        
        assert hasattr(project_manager_with_mocks.ai_interface, 'execute_development_task')
        assert hasattr(project_manager_with_mocks.ai_interface, 'analyze_requirements')
        assert hasattr(project_manager_with_mocks.ai_interface, 'cleanup')
        
        assert hasattr(project_manager_with_mocks.validator, 'validate_project')
        assert hasattr(project_manager_with_mocks.validator, 'validate_task_result')
        assert hasattr(project_manager_with_mocks.validator, 'cleanup')


# Performance test for London School testing
class TestProjectManagerPerformance:
    """Performance tests using pytest-benchmark"""
    
    def test_project_manager_initialization_performance(self, benchmark, mock_config_manager):
        """Benchmark ProjectManager initialization time"""
        
        def create_project_manager():
            return ProjectManager(config_manager=mock_config_manager)
        
        result = benchmark(create_project_manager)
        assert result is not None
        assert hasattr(result, 'config_manager')