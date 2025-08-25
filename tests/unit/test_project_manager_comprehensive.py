#!/usr/bin/env python3
"""
Comprehensive Unit Tests for ProjectManager.

Tests the core project management functionality including:
- Project creation and lifecycle management
- State persistence and recovery
- Configuration management
- Anti-hallucination validation integration
- Error handling and edge cases
"""

import asyncio
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime, timedelta

# Import the components under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.project_manager import (
    ProjectManager, Project, StateManager, ProjectManagerException
)
from core.types import ProjectState, ProgressMetrics, ValidationResult, Priority
from core.config_manager import ProjectConfig
from core.task_engine import ExecutionStrategy, create_task


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager."""
    mock = Mock()
    mock.get_template_config.return_value = {
        'name': 'Test Template',
        'directories': ['src', 'tests'],
        'files': {
            'README.md': '# {{PROJECT_NAME}}\n\n{{PROJECT_DESCRIPTION}}\n',
            'src/__init__.py': '"""{{PROJECT_NAME}} package."""\n'
        },
        'config': {
            'framework': 'python',
            'features': ['basic-structure']
        }
    }
    return mock


@pytest.fixture
def mock_task_engine():
    """Mock task engine."""
    mock = AsyncMock()
    mock.execute_workflow.return_value = Mock(
        success=True,
        tasks_executed=[uuid4()],
        files_generated=['src/main.py'],
        quality_metrics={'quality_score': 85.0, 'authenticity_rate': 92.0}
    )
    return mock


@pytest.fixture
def mock_ai_interface():
    """Mock AI interface."""
    mock = AsyncMock()
    mock.execute_task.return_value = "Mock AI response"
    return mock


@pytest.fixture
def mock_validator():
    """Mock progress validator."""
    mock = AsyncMock()
    mock.validate_codebase.return_value = ValidationResult(
        is_authentic=True,
        authenticity_score=88.5,
        real_progress=85.0,
        fake_progress=15.0,
        issues=[],
        suggestions=['Continue development'],
        next_actions=['implement-features']
    )
    return mock


@pytest.fixture
def sample_project_config():
    """Sample project configuration."""
    return ProjectConfig(
        name='Test Project',
        description='A test project for unit testing',
        framework='python',
        features=['web-api', 'database'],
        validation_config={'auto_fix_enabled': True}
    )


class TestProjectManager:
    """Comprehensive tests for ProjectManager class."""
    
    @pytest.mark.asyncio
    async def test_project_manager_initialization(self, temp_project_dir):
        """Test ProjectManager initialization."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        assert pm.config_manager is not None
        assert pm.state_manager is not None
        assert pm.task_engine is not None
        assert pm.ai_interface is not None
        assert len(pm._project_templates) > 0
        assert 'basic' in pm._project_templates
        assert 'fastapi' in pm._project_templates
        assert 'react' in pm._project_templates
    
    @pytest.mark.asyncio
    async def test_create_project_basic(self, temp_project_dir):
        """Test basic project creation."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        project = await pm.create_project(
            name="TestProject",
            template="basic",
            project_path=temp_project_dir / "test_project"
        )
        
        assert project is not None
        assert project.name == "TestProject"
        assert project.state == ProjectState.ACTIVE
        assert project.path == temp_project_dir / "test_project"
        assert project.path.exists()
        
        # Check generated files
        assert (project.path / "README.md").exists()
        assert (project.path / "src").exists()
        assert (project.path / "tests").exists()
        assert (project.path / "src" / "__init__.py").exists()
        
        # Verify content has been processed
        readme_content = (project.path / "README.md").read_text()
        assert "TestProject" in readme_content
    
    @pytest.mark.asyncio
    async def test_create_project_with_overrides(self, temp_project_dir):
        """Test project creation with configuration overrides."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        overrides = {
            'description': 'Custom description',
            'framework': 'fastapi',
            'features': ['api', 'database', 'testing']
        }
        
        project = await pm.create_project(
            name="CustomProject",
            template="basic",
            config_overrides=overrides,
            project_path=temp_project_dir / "custom_project"
        )
        
        assert project.description == 'Custom description'
        assert project.config.framework == 'fastapi'
        assert 'api' in project.config.features
    
    @pytest.mark.asyncio
    async def test_create_project_invalid_template(self, temp_project_dir):
        """Test project creation with invalid template."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Should fallback to basic template
        project = await pm.create_project(
            name="TestProject",
            template="nonexistent",
            project_path=temp_project_dir / "test_project"
        )
        
        assert project is not None
        assert project.name == "TestProject"
        assert (project.path / "README.md").exists()
    
    @pytest.mark.asyncio
    async def test_orchestrate_development(self, temp_project_dir, mock_task_engine, mock_validator):
        """Test development orchestration workflow."""
        pm = ProjectManager(config_dir=temp_project_dir)
        pm.task_engine = mock_task_engine
        pm.validator = mock_validator
        
        # Create a project first
        project = await pm.create_project(
            name="DevProject",
            project_path=temp_project_dir / "dev_project"
        )
        
        # Define development requirements
        requirements = {
            'description': 'Build a REST API',
            'features': ['authentication', 'user-management'],
            'setup_project': True,
            'include_tests': True
        }
        
        # Execute development workflow
        result = await pm.orchestrate_development(
            project.id,
            requirements,
            ExecutionStrategy.ADAPTIVE
        )
        
        assert result.success is True
        assert len(result.tasks_executed) > 0
        assert result.files_generated is not None
        
        # Verify project state was updated
        assert project.progress.authenticity_rate == 88.5  # From mock validator
        assert len(project.validation_history) == 1
        assert project.id not in pm._active_projects  # Workflow completed
    
    @pytest.mark.asyncio
    async def test_orchestrate_development_project_not_found(self, temp_project_dir):
        """Test development orchestration with non-existent project."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        fake_id = uuid4()
        requirements = {'description': 'Test'}
        
        with pytest.raises(ProjectManagerException, match="Project .* not found"):
            await pm.orchestrate_development(fake_id, requirements)
    
    @pytest.mark.asyncio
    async def test_get_progress_report(self, temp_project_dir, mock_validator):
        """Test progress report generation."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Create project with some progress data
        project = await pm.create_project(
            name="ProgressProject",
            project_path=temp_project_dir / "progress_project"
        )
        
        # Simulate some progress
        project.generated_files.add('src/main.py')
        project.modified_files.add('README.md')
        project.validation_history.append(ValidationResult(
            is_authentic=True,
            authenticity_score=90.0,
            real_progress=80.0,
            fake_progress=20.0
        ))
        
        # Get progress report
        report = await pm.get_progress_report(project.id)
        
        assert report.project_id == project.id
        assert report.metrics.authenticity_rate > 0
        assert report.validation.authenticity_score == 90.0
        assert len(report.recent_activities) >= 2  # Generated and modified files
        assert 'Generated 1 files' in report.recent_activities
        assert 'Modified 1 files' in report.recent_activities
    
    @pytest.mark.asyncio
    async def test_validate_project_authenticity(self, temp_project_dir, mock_validator):
        """Test project authenticity validation."""
        pm = ProjectManager(config_dir=temp_project_dir)
        pm.validator = mock_validator
        
        # Create project
        project = await pm.create_project(
            name="ValidateProject",
            project_path=temp_project_dir / "validate_project"
        )
        
        # Add some content to validate
        (project.path / "src" / "main.py").write_text(
            "def hello():\n    print('Hello, World!')\n"
        )
        
        # Run validation
        result = await pm.validate_project_authenticity(project.id)
        
        assert result.is_authentic is True
        assert result.authenticity_score == 88.5
        assert len(project.validation_history) == 1
        
        # Verify validator was called with correct path
        mock_validator.validate_codebase.assert_called_once_with(project.path)
    
    @pytest.mark.asyncio
    async def test_validate_project_authenticity_with_auto_fix(self, temp_project_dir):
        """Test validation with auto-fix enabled."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Mock validator that finds issues
        mock_validator = AsyncMock()
        mock_issues = [
            Mock(type='PLACEHOLDER', severity='HIGH', description='TODO found')
        ]
        mock_validator.validate_codebase.return_value = ValidationResult(
            is_authentic=False,
            authenticity_score=60.0,
            real_progress=60.0,
            fake_progress=40.0,
            issues=mock_issues
        )
        mock_validator.auto_fix_placeholders.return_value = {
            'fixed_count': 1,
            'failed_count': 0
        }
        
        pm.validator = mock_validator
        
        # Create project with auto-fix enabled
        project = await pm.create_project(
            name="AutoFixProject",
            config_overrides={'validation_config': {'auto_fix_enabled': True}},
            project_path=temp_project_dir / "autofix_project"
        )
        
        # Run validation
        result = await pm.validate_project_authenticity(project.id)
        
        # Verify auto-fix was triggered
        mock_validator.auto_fix_placeholders.assert_called_once_with(
            mock_issues, project.path
        )
    
    @pytest.mark.asyncio
    async def test_list_projects(self, temp_project_dir):
        """Test project listing."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Create multiple projects
        project1 = await pm.create_project(
            name="Project1",
            project_path=temp_project_dir / "project1"
        )
        project2 = await pm.create_project(
            name="Project2",
            project_path=temp_project_dir / "project2"
        )
        
        # List all projects
        projects = await pm.list_projects()
        
        assert len(projects) == 2
        project_names = [p['name'] for p in projects]
        assert 'Project1' in project_names
        assert 'Project2' in project_names
        
        # Test filtering by state
        active_projects = await pm.list_projects(state_filter=ProjectState.ACTIVE)
        assert len(active_projects) == 2
        
        # Mark one project as failed and test filtering
        project1.state = ProjectState.FAILED
        failed_projects = await pm.list_projects(state_filter=ProjectState.FAILED)
        assert len(failed_projects) == 1
        assert failed_projects[0]['name'] == 'Project1'
    
    @pytest.mark.asyncio
    async def test_get_project_details(self, temp_project_dir):
        """Test getting detailed project information."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Create project
        project = await pm.create_project(
            name="DetailProject",
            project_path=temp_project_dir / "detail_project"
        )
        
        # Get details
        details = await pm.get_project_details(project.id)
        
        assert details is not None
        assert details['id'] == str(project.id)
        assert details['name'] == 'DetailProject'
        assert details['state'] == ProjectState.ACTIVE.value
        assert 'progress' in details
        assert 'generated_files' in details
        
        # Test non-existent project
        fake_id = uuid4()
        details = await pm.get_project_details(fake_id)
        assert details is None
    
    @pytest.mark.asyncio
    async def test_delete_project(self, temp_project_dir):
        """Test project deletion."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Create project
        project = await pm.create_project(
            name="DeleteProject",
            project_path=temp_project_dir / "delete_project"
        )
        
        project_path = project.path
        project_id = project.id
        
        assert project_path.exists()
        assert project_id in pm._active_projects
        
        # Delete project without removing files
        success = await pm.delete_project(project_id, remove_files=False)
        
        assert success is True
        assert project_id not in pm._active_projects
        assert project_path.exists()  # Files should still exist
        
        # Create another project and delete with file removal
        project2 = await pm.create_project(
            name="DeleteProject2",
            project_path=temp_project_dir / "delete_project2"
        )
        
        project2_path = project2.path
        project2_id = project2.id
        
        success = await pm.delete_project(project2_id, remove_files=True)
        
        assert success is True
        assert project2_id not in pm._active_projects
        assert not project2_path.exists()  # Files should be removed
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_project(self, temp_project_dir):
        """Test deleting a non-existent project."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        fake_id = uuid4()
        success = await pm.delete_project(fake_id)
        
        assert success is False


class TestStateManager:
    """Tests for StateManager class."""
    
    @pytest.mark.asyncio
    async def test_state_manager_initialization(self, temp_project_dir):
        """Test StateManager initialization."""
        sm = StateManager(state_dir=temp_project_dir / "state")
        
        assert sm.state_dir.exists()
        assert sm._project_states == {}
    
    @pytest.mark.asyncio
    async def test_save_and_load_project_state(self, temp_project_dir):
        """Test saving and loading project state."""
        sm = StateManager(state_dir=temp_project_dir / "state")
        
        project_id = uuid4()
        state_data = {
            'id': str(project_id),
            'name': 'Test Project',
            'state': 'ACTIVE',
            'progress': {'real_progress': 75.0, 'authenticity_rate': 85.0}
        }
        
        # Save state
        await sm.save_project_state(project_id, state_data)
        
        # Verify file was created
        state_file = sm.state_dir / f'project_{project_id}.json'
        assert state_file.exists()
        
        # Load state
        loaded_state = await sm.load_project_state(project_id)
        
        assert loaded_state is not None
        assert loaded_state['id'] == str(project_id)
        assert loaded_state['name'] == 'Test Project'
        assert 'last_saved' in loaded_state
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_project_state(self, temp_project_dir):
        """Test loading state for non-existent project."""
        sm = StateManager(state_dir=temp_project_dir / "state")
        
        fake_id = uuid4()
        loaded_state = await sm.load_project_state(fake_id)
        
        assert loaded_state is None
    
    @pytest.mark.asyncio
    async def test_delete_project_state(self, temp_project_dir):
        """Test deleting project state."""
        sm = StateManager(state_dir=temp_project_dir / "state")
        
        project_id = uuid4()
        state_data = {'id': str(project_id), 'name': 'Test Project'}
        
        # Save state
        await sm.save_project_state(project_id, state_data)
        
        # Verify state exists
        state_file = sm.state_dir / f'project_{project_id}.json'
        assert state_file.exists()
        assert project_id in sm._project_states
        
        # Delete state
        await sm.delete_project_state(project_id)
        
        # Verify state is removed
        assert not state_file.exists()
        assert project_id not in sm._project_states
    
    @pytest.mark.asyncio
    async def test_state_persistence_cache(self, temp_project_dir):
        """Test state manager memory cache."""
        sm = StateManager(state_dir=temp_project_dir / "state")
        
        project_id = uuid4()
        state_data = {'id': str(project_id), 'name': 'Cached Project'}
        
        # Save state (should cache in memory)
        await sm.save_project_state(project_id, state_data)
        
        # Delete the file but keep memory cache
        state_file = sm.state_dir / f'project_{project_id}.json'
        state_file.unlink()
        
        # Should still load from memory cache
        loaded_state = await sm.load_project_state(project_id)
        assert loaded_state is not None
        assert loaded_state['name'] == 'Cached Project'


class TestProject:
    """Tests for Project model class."""
    
    def test_project_initialization(self, sample_project_config):
        """Test Project initialization."""
        project_id = uuid4()
        project = Project(
            id=project_id,
            name='Test Project',
            description='A test project',
            config=sample_project_config,
            path=Path('/tmp/test_project')
        )
        
        assert project.id == project_id
        assert project.name == 'Test Project'
        assert project.description == 'A test project'
        assert project.state == ProjectState.INITIALIZING
        assert project.path == Path('/tmp/test_project')
        assert isinstance(project.progress, ProgressMetrics)
        assert project.validation_history == []
        assert project.active_workflows == {}
        assert project.generated_files == set()
        assert project.modified_files == set()
    
    def test_update_progress(self, sample_project_config):
        """Test progress update functionality."""
        project = Project(
            id=uuid4(),
            name='Progress Project',
            config=sample_project_config
        )
        
        initial_updated_at = project.updated_at
        
        # Create new progress metrics
        new_metrics = ProgressMetrics(
            real_progress=75.0,
            fake_progress=25.0,
            authenticity_rate=88.0,
            quality_score=82.5,
            tasks_completed=8,
            tasks_total=10
        )
        
        # Update progress
        project.update_progress(new_metrics)
        
        assert project.progress.real_progress == 75.0
        assert project.progress.authenticity_rate == 88.0
        assert project.updated_at > initial_updated_at
    
    def test_add_validation_result(self, sample_project_config):
        """Test validation result history management."""
        project = Project(
            id=uuid4(),
            name='Validation Project',
            config=sample_project_config
        )
        
        # Add validation results
        for i in range(55):  # More than the 50 limit
            result = ValidationResult(
                is_authentic=True,
                authenticity_score=90.0 - i,
                real_progress=90.0,
                fake_progress=10.0
            )
            project.add_validation_result(result)
        
        # Should keep only last 50 results
        assert len(project.validation_history) == 50
        assert project.validation_history[0].authenticity_score == 45.0  # 90 - 45
        assert project.validation_history[-1].authenticity_score == -4.0  # 90 - 54
    
    def test_project_to_dict(self, sample_project_config):
        """Test project serialization to dictionary."""
        project_id = uuid4()
        project = Project(
            id=project_id,
            name='Serialization Project',
            description='Test serialization',
            config=sample_project_config,
            path=Path('/tmp/serialize_project')
        )
        
        # Add some data
        project.generated_files.add('src/main.py')
        project.modified_files.add('README.md')
        workflow_id = uuid4()
        project.completed_workflows.append(workflow_id)
        
        # Serialize
        project_dict = project.to_dict()
        
        assert project_dict['id'] == str(project_id)
        assert project_dict['name'] == 'Serialization Project'
        assert project_dict['state'] == ProjectState.INITIALIZING.value
        assert project_dict['path'] == '/tmp/serialize_project'
        assert 'progress' in project_dict
        assert 'src/main.py' in project_dict['generated_files']
        assert 'README.md' in project_dict['modified_files']
        assert str(workflow_id) in project_dict['completed_workflows']


class TestProjectManagerIntegration:
    """Integration tests for ProjectManager with real components."""
    
    @pytest.mark.asyncio
    async def test_full_project_lifecycle(self, temp_project_dir):
        """Test complete project lifecycle from creation to deletion."""
        pm = ProjectManager(
            config_dir=temp_project_dir,
            enable_validation=True,
            max_concurrent_tasks=3
        )
        
        # 1. Create project
        project = await pm.create_project(
            name="LifecycleProject",
            template="fastapi",
            config_overrides={
                'description': 'Full lifecycle test project',
                'features': ['api', 'database', 'testing']
            },
            project_path=temp_project_dir / "lifecycle_project"
        )
        
        assert project.state == ProjectState.ACTIVE
        assert (project.path / "app" / "main.py").exists()
        
        # 2. Run development workflow
        requirements = {
            'description': 'Implement user authentication system',
            'features': ['auth', 'user-management', 'jwt'],
            'setup_project': False,  # Already set up
            'include_tests': True
        }
        
        # Mock the task engine to avoid real AI calls
        mock_result = Mock(
            success=True,
            tasks_executed=[uuid4(), uuid4()],
            files_generated=['app/auth.py', 'tests/test_auth.py'],
            quality_metrics={
                'quality_score': 87.5,
                'authenticity_rate': 93.0,
                'success_rate': 100.0
            }
        )
        
        with patch.object(pm.task_engine, 'execute_workflow', return_value=mock_result):
            dev_result = await pm.orchestrate_development(
                project.id, requirements, ExecutionStrategy.ADAPTIVE
            )
        
        assert dev_result.success
        assert len(dev_result.tasks_executed) == 2
        
        # 3. Validate project authenticity
        with patch.object(pm.validator, 'validate_codebase') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_authentic=True,
                authenticity_score=92.0,
                real_progress=88.0,
                fake_progress=12.0,
                issues=[],
                suggestions=['Project looks good'],
                next_actions=['deploy-to-staging']
            )
            
            validation_result = await pm.validate_project_authenticity(project.id)
        
        assert validation_result.is_authentic
        assert validation_result.authenticity_score == 92.0
        
        # 4. Get progress report
        progress_report = await pm.get_progress_report(project.id)
        
        assert progress_report.project_id == project.id
        assert progress_report.metrics.authenticity_rate == 92.0
        assert len(progress_report.recent_activities) > 0
        
        # 5. Test project persistence across manager instances
        pm2 = ProjectManager(config_dir=temp_project_dir)
        project_details = await pm2.get_project_details(project.id)
        
        assert project_details is not None
        assert project_details['name'] == 'LifecycleProject'
        
        # 6. Delete project
        success = await pm.delete_project(project.id, remove_files=True)
        
        assert success
        assert not project.path.exists()
    
    @pytest.mark.asyncio
    async def test_concurrent_project_operations(self, temp_project_dir):
        """Test concurrent project operations."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Create multiple projects concurrently
        project_tasks = [
            pm.create_project(
                name=f"ConcurrentProject{i}",
                project_path=temp_project_dir / f"concurrent_project_{i}"
            )
            for i in range(5)
        ]
        
        projects = await asyncio.gather(*project_tasks)
        
        assert len(projects) == 5
        assert all(p.state == ProjectState.ACTIVE for p in projects)
        assert len(set(p.id for p in projects)) == 5  # All unique IDs
        
        # List all projects
        project_list = await pm.list_projects()
        assert len(project_list) == 5
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, temp_project_dir):
        """Test error handling and recovery mechanisms."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Test project creation in non-existent directory
        with pytest.raises(ProjectManagerException):
            await pm.create_project(
                name="BadProject",
                project_path=Path("/nonexistent/path/bad_project")
            )
        
        # Create a valid project
        project = await pm.create_project(
            name="RecoveryProject",
            project_path=temp_project_dir / "recovery_project"
        )
        
        # Simulate task engine failure
        with patch.object(pm.task_engine, 'execute_workflow') as mock_execute:
            mock_execute.side_effect = Exception("Task engine failed")
            
            with pytest.raises(ProjectManagerException, match="Development workflow failed"):
                await pm.orchestrate_development(
                    project.id,
                    {'description': 'Test requirements'},
                    ExecutionStrategy.SEQUENTIAL
                )
            
            # Project should be marked as failed
            assert project.state == ProjectState.FAILED
    
    @pytest.mark.asyncio
    async def test_template_processing_edge_cases(self, temp_project_dir):
        """Test template processing edge cases."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Create project with special characters in name
        project = await pm.create_project(
            name="Test-Project_123 (Special)",
            template="basic",
            project_path=temp_project_dir / "special_chars_project"
        )
        
        # Check that template variables were processed correctly
        readme_content = (project.path / "README.md").read_text()
        assert "Test-Project_123 (Special)" in readme_content
        
        init_content = (project.path / "src" / "__init__.py").read_text()
        assert "Test-Project_123 (Special)" in init_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
