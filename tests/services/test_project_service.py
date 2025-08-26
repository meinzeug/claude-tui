"""
Comprehensive unit tests for Project Service.

Tests cover:
- Project creation and initialization
- Project loading and validation
- Configuration management
- Directory structure handling
- Git integration
- Virtual environment management
- Error handling and edge cases
- Performance under various project sizes
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from uuid import uuid4

from services.project_service import ProjectService
from core.exceptions import (
    ConfigurationError, FileSystemError, ProjectDirectoryError, ValidationError
)


class TestProjectServiceInitialization:
    """Test Project Service initialization and setup."""
    
    @pytest.mark.unit
    async def test_service_initialization_success(self, mock_project_manager):
        """Test successful service initialization."""
        service = ProjectService()
        
        with patch('services.project_service.ProjectManager', return_value=mock_project_manager):
            await service.initialize()
            
            assert service._initialized is True
            assert service._project_manager is not None
            assert isinstance(service._active_projects, dict)
    
    @pytest.mark.unit
    async def test_service_initialization_failure(self):
        """Test service initialization failure."""
        service = ProjectService()
        
        with patch('services.project_service.ProjectManager') as mock_pm:
            mock_pm.side_effect = Exception("ProjectManager initialization failed")
            
            with pytest.raises(ConfigurationError) as excinfo:
                await service.initialize()
            
            assert "Project service initialization failed" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_health_check(self, project_service):
        """Test project service health check."""
        health = await project_service.health_check()
        
        assert isinstance(health, dict)
        assert health['service'] == 'ProjectService'
        assert health['status'] == 'healthy'
        assert 'project_manager_available' in health
        assert 'active_projects_count' in health
        assert 'cache_size' in health


class TestProjectCreation:
    """Test project creation functionality."""
    
    @pytest.mark.unit
    async def test_create_project_success(self, project_service, temp_directory):
        """Test successful project creation."""
        project_path = temp_directory / "new_project"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            mock_create.return_value = {
                'id': 'project-123',
                'name': 'New Project',
                'type': 'python',
                'created_at': '2025-01-01T00:00:00Z'
            }
            
            result = await project_service.create_project(
                name="New Project",
                path=project_path,
                project_type="python",
                initialize_git=False,
                create_venv=False
            )
            
            assert result['name'] == "New Project"
            assert result['type'] == "python"
            assert 'project_id' in result
            assert project_path.exists()
            
            # Check that project was added to active projects
            assert len(project_service._active_projects) > 0
    
    @pytest.mark.unit
    async def test_create_project_invalid_name(self, project_service, temp_directory):
        """Test project creation with invalid name."""
        with pytest.raises(ValidationError) as excinfo:
            await project_service.create_project(
                name="",  # Empty name
                path=temp_directory / "project"
            )
        
        assert "Project name cannot be empty" in str(excinfo.value)
        
        with pytest.raises(ValidationError) as excinfo:
            await project_service.create_project(
                name=None,  # None name
                path=temp_directory / "project"
            )
        
        assert "Project name cannot be empty" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_create_project_existing_directory(self, project_service, temp_directory):
        """Test project creation when directory already exists."""
        existing_dir = temp_directory / "existing"
        existing_dir.mkdir()
        
        with pytest.raises(ProjectDirectoryError) as excinfo:
            await project_service.create_project(
                name="Test Project",
                path=existing_dir
            )
        
        assert "Directory already exists" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_create_project_with_git_initialization(self, project_service, temp_directory):
        """Test project creation with git initialization."""
        project_path = temp_directory / "git_project"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_create.return_value = {'id': 'test', 'name': 'Test'}
                
                # Mock successful git init
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'', b'')
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                result = await project_service.create_project(
                    name="Git Project",
                    path=project_path,
                    initialize_git=True
                )
                
                assert result['config']['git_initialized'] is True
                mock_subprocess.assert_called_with(
                    'git', 'init',
                    cwd=project_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
    
    @pytest.mark.unit
    async def test_create_project_with_venv(self, project_service, temp_directory):
        """Test Python project creation with virtual environment."""
        project_path = temp_directory / "python_project"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_create.return_value = {'id': 'test', 'name': 'Test'}
                
                # Mock successful venv creation
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'', b'')
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                result = await project_service.create_project(
                    name="Python Project",
                    path=project_path,
                    project_type="python",
                    create_venv=True
                )
                
                assert result['config']['venv_created'] is True
                mock_subprocess.assert_called_with(
                    'python', '-m', 'venv', str(project_path / 'venv'),
                    cwd=project_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
    
    @pytest.mark.unit
    async def test_create_project_cleanup_on_failure(self, project_service, temp_directory):
        """Test project cleanup when creation fails."""
        project_path = temp_directory / "failed_project"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            mock_create.side_effect = Exception("Creation failed")
            
            with pytest.raises(ProjectDirectoryError):
                await project_service.create_project(
                    name="Failed Project",
                    path=project_path
                )
            
            # Directory should be cleaned up
            assert not project_path.exists()


class TestProjectLoading:
    """Test project loading functionality."""
    
    @pytest.mark.unit
    async def test_load_project_success(self, project_service, temp_directory):
        """Test successful project loading."""
        project_path = temp_directory / "existing_project"
        project_path.mkdir()
        
        # Create project config
        config_dir = project_path / '.claude-tui'
        config_dir.mkdir()
        config_file = config_dir / 'project.json'
        
        config_data = {
            'id': 'project-456',
            'name': 'Existing Project',
            'type': 'python',
            'created_at': '2025-01-01T00:00:00Z'
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        with patch.object(project_service._project_manager, 'load_project') as mock_load:
            mock_load.return_value = {
                'name': 'Existing Project',
                'type': 'python',
                'path': str(project_path)
            }
            
            result = await project_service.load_project(
                path=project_path,
                validate_structure=True
            )
            
            assert result['name'] == 'Existing Project'
            assert result['type'] == 'python'
            assert result['project_id'] == 'project-456'
            assert 'validation' in result
            
            # Check added to active projects
            assert 'project-456' in project_service._active_projects
    
    @pytest.mark.unit
    async def test_load_project_nonexistent(self, project_service):
        """Test loading nonexistent project."""
        nonexistent_path = Path("/nonexistent/project")
        
        with pytest.raises(ProjectDirectoryError) as excinfo:
            await project_service.load_project(nonexistent_path)
        
        assert "Project directory does not exist" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_load_project_not_directory(self, project_service, temp_directory):
        """Test loading project that is not a directory."""
        file_path = temp_directory / "not_a_directory.txt"
        file_path.write_text("This is a file")
        
        with pytest.raises(ProjectDirectoryError) as excinfo:
            await project_service.load_project(file_path)
        
        assert "Path is not a directory" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_load_project_without_config(self, project_service, temp_directory):
        """Test loading project without existing config."""
        project_path = temp_directory / "no_config_project"
        project_path.mkdir()
        
        with patch.object(project_service._project_manager, 'load_project') as mock_load:
            mock_load.return_value = {
                'name': 'No Config Project',
                'type': 'unknown'
            }
            
            result = await project_service.load_project(project_path)
            
            assert result['name'] == 'No Config Project'
            assert 'project_id' in result  # Should generate new ID
            assert len(result['project_id']) > 0


class TestProjectValidation:
    """Test project structure validation."""
    
    @pytest.mark.unit
    async def test_validate_project_structure_complete(self, project_service, temp_directory):
        """Test validation of complete project structure."""
        project_path = temp_directory / "complete_project"
        project_path.mkdir()
        
        # Create common project files
        (project_path / "README.md").write_text("# Complete Project")
        (project_path / "setup.py").write_text("from setuptools import setup")
        (project_path / "requirements.txt").write_text("pytest>=6.0")
        (project_path / ".gitignore").write_text("__pycache__/")
        
        # Create source structure
        src_dir = project_path / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        
        # Create tests
        tests_dir = project_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("")
        
        # Create config
        config_dir = project_path / "config"
        config_dir.mkdir()
        
        # Add to active projects first
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path)
            }
        }
        
        result = await project_service.validate_project(project_id)
        
        assert result['is_valid'] is True
        assert result['score'] > 0.8
        assert len(result['issues']) == 0
        assert result['metadata']['found_files']
        assert result['metadata']['has_src_structure'] is True
        assert result['metadata']['has_tests'] is True
        assert result['metadata']['has_config'] is True
    
    @pytest.mark.unit
    async def test_validate_project_structure_minimal(self, project_service, temp_directory):
        """Test validation of minimal project structure."""
        project_path = temp_directory / "minimal_project"
        project_path.mkdir()
        
        # Only create one file
        (project_path / "main.py").write_text("print('Hello World')")
        
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path)
            }
        }
        
        result = await project_service.validate_project(project_id)
        
        assert result['is_valid'] is True
        assert result['score'] < 1.0  # Should have recommendations
        assert len(result['recommendations']) > 0
        assert any('README' in rec for rec in result['recommendations'])


class TestProjectConfiguration:
    """Test project configuration management."""
    
    @pytest.mark.unit
    async def test_update_project_config_success(self, project_service, temp_directory):
        """Test successful project configuration update."""
        project_path = temp_directory / "config_project"
        project_path.mkdir()
        config_dir = project_path / '.claude-tui'
        config_dir.mkdir()
        
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path),
                'config': {'existing_key': 'existing_value'}
            }
        }
        
        config_updates = {
            'new_key': 'new_value',
            'environment': 'development'
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = await project_service.update_project_config(
                project_id,
                config_updates
            )
            
            assert result['config']['new_key'] == 'new_value'
            assert result['config']['environment'] == 'development'
            assert result['config']['existing_key'] == 'existing_value'
            assert 'updated_at' in result
            
            # Verify file was written
            mock_file.assert_called_once()
    
    @pytest.mark.unit
    async def test_update_project_config_nonexistent(self, project_service):
        """Test updating config for nonexistent project."""
        with pytest.raises(ValidationError) as excinfo:
            await project_service.update_project_config(
                "nonexistent-project-id",
                {'key': 'value'}
            )
        
        assert "Project nonexistent-project-id not found" in str(excinfo.value)


class TestProjectManagement:
    """Test project management operations."""
    
    @pytest.mark.unit
    async def test_get_project_info_success(self, project_service, temp_directory):
        """Test retrieving project information."""
        project_path = temp_directory / "info_project"
        project_path.mkdir()
        
        # Create some files to test runtime info
        (project_path / ".git").mkdir()
        (project_path / "venv").mkdir()
        (project_path / "file1.py").write_text("# File 1")
        (project_path / "file2.py").write_text("# File 2")
        
        project_id = str(uuid4())
        project_data = {
            'id': project_id,
            'name': 'Info Project',
            'path': str(project_path),
            'type': 'python'
        }
        
        project_service._active_projects[project_id] = {
            'project_data': project_data,
            'last_accessed': '2025-01-01T00:00:00Z',
            'status': 'active'
        }
        
        result = await project_service.get_project_info(project_id)
        
        assert result['name'] == 'Info Project'
        assert result['type'] == 'python'
        assert 'runtime_info' in result
        assert result['runtime_info']['exists'] is True
        assert result['runtime_info']['is_git_repo'] is True
        assert result['runtime_info']['has_venv'] is True
        assert result['runtime_info']['file_count'] > 0
    
    @pytest.mark.unit
    async def test_get_project_info_nonexistent(self, project_service):
        """Test retrieving info for nonexistent project."""
        with pytest.raises(ValidationError) as excinfo:
            await project_service.get_project_info("nonexistent-id")
        
        assert "Project nonexistent-id not found" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_list_active_projects(self, project_service):
        """Test listing active projects."""
        # Add some projects
        for i in range(3):
            project_id = str(uuid4())
            project_service._active_projects[project_id] = {
                'project_data': {
                    'id': project_id,
                    'name': f'Project {i}',
                    'type': 'python'
                },
                'last_accessed': f'2025-01-0{i+1}T00:00:00Z',
                'status': 'active'
            }
        
        projects = await project_service.list_active_projects()
        
        assert len(projects) == 3
        # Should be sorted by last_accessed (most recent first)
        assert projects[0]['name'] == 'Project 2'
        assert projects[1]['name'] == 'Project 1'
        assert projects[2]['name'] == 'Project 0'
    
    @pytest.mark.unit
    async def test_remove_project_success(self, project_service, temp_directory):
        """Test successful project removal."""
        project_path = temp_directory / "remove_project"
        project_path.mkdir()
        (project_path / "file.txt").write_text("content")
        
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path)
            }
        }
        
        result = await project_service.remove_project(
            project_id,
            delete_files=True
        )
        
        assert result['project_id'] == project_id
        assert result['files_deleted'] is True
        assert 'removed_at' in result
        
        # Project should be removed from active projects
        assert project_id not in project_service._active_projects
        
        # Files should be deleted
        assert not project_path.exists()
    
    @pytest.mark.unit
    async def test_remove_project_keep_files(self, project_service, temp_directory):
        """Test project removal while keeping files."""
        project_path = temp_directory / "keep_files_project"
        project_path.mkdir()
        
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path)
            }
        }
        
        result = await project_service.remove_project(
            project_id,
            delete_files=False
        )
        
        assert result['files_deleted'] is False
        assert project_id not in project_service._active_projects
        assert project_path.exists()  # Files should still exist


class TestProjectServiceErrorHandling:
    """Test Project Service error handling and edge cases."""
    
    @pytest.mark.unit
    async def test_project_manager_not_initialized(self):
        """Test operations when project manager is not initialized."""
        service = ProjectService()
        # Don't initialize service
        
        with pytest.raises(ConfigurationError):
            await service.create_project(
                name="Test",
                path="/tmp/test"
            )
        
        with pytest.raises(ConfigurationError):
            await service.load_project("/tmp/test")
    
    @pytest.mark.unit
    async def test_git_initialization_failure(self, project_service, temp_directory):
        """Test handling of git initialization failure."""
        project_path = temp_directory / "git_fail_project"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_create.return_value = {'id': 'test', 'name': 'Test'}
                
                # Mock failed git init
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'', b'Git not found')
                mock_process.returncode = 1
                mock_subprocess.return_value = mock_process
                
                result = await project_service.create_project(
                    name="Git Fail Project",
                    path=project_path,
                    initialize_git=True
                )
                
                # Project should still be created despite git failure
                assert result['name'] == "Git Fail Project"
                assert result['config']['git_initialized'] is False
    
    @pytest.mark.unit
    async def test_venv_creation_failure(self, project_service, temp_directory):
        """Test handling of virtual environment creation failure."""
        project_path = temp_directory / "venv_fail_project"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_create.return_value = {'id': 'test', 'name': 'Test'}
                
                # Mock failed venv creation
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'', b'Python not found')
                mock_process.returncode = 1
                mock_subprocess.return_value = mock_process
                
                result = await project_service.create_project(
                    name="Venv Fail Project",
                    path=project_path,
                    project_type="python",
                    create_venv=True
                )
                
                # Project should still be created despite venv failure
                assert result['name'] == "Venv Fail Project"
                assert result['config']['venv_created'] is False
    
    @pytest.mark.unit
    async def test_config_file_write_failure(self, project_service, temp_directory):
        """Test handling of config file write failure."""
        project_path = temp_directory / "config_fail_project"
        
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path),
                'config': {}
            }
        }
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(FileSystemError) as excinfo:
                await project_service.update_project_config(project_id, {'key': 'value'})
            
            assert "Failed to save project config" in str(excinfo.value)


class TestProjectServiceEdgeCases:
    """Test Project Service edge cases and boundary conditions."""
    
    @pytest.mark.edge_case
    async def test_very_long_project_name(self, project_service, temp_directory):
        """Test project creation with very long name."""
        long_name = "A" * 1000  # Very long project name
        project_path = temp_directory / "long_name_project"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            mock_create.return_value = {'id': 'test', 'name': long_name}
            
            result = await project_service.create_project(
                name=long_name,
                path=project_path
            )
            
            assert result['name'] == long_name
            assert len(result['name']) == 1000
    
    @pytest.mark.edge_case
    async def test_special_characters_in_path(self, project_service, temp_directory):
        """Test project creation with special characters in path."""
        special_path = temp_directory / "special chars & symbols!"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            mock_create.return_value = {'id': 'test', 'name': 'Special'}
            
            result = await project_service.create_project(
                name="Special Project",
                path=special_path
            )
            
            assert result['name'] == "Special Project"
            assert special_path.exists()
    
    @pytest.mark.edge_case
    async def test_deep_directory_structure(self, project_service, temp_directory):
        """Test project creation in deeply nested directory."""
        deep_path = temp_directory
        
        # Create deep directory structure
        for i in range(10):
            deep_path = deep_path / f"level_{i}"
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            mock_create.return_value = {'id': 'test', 'name': 'Deep'}
            
            result = await project_service.create_project(
                name="Deep Project",
                path=deep_path
            )
            
            assert result['name'] == "Deep Project"
            assert deep_path.exists()
    
    @pytest.mark.edge_case
    async def test_large_project_validation(self, project_service, temp_directory):
        """Test validation of project with many files."""
        project_path = temp_directory / "large_project"
        project_path.mkdir()
        
        # Create many files
        src_dir = project_path / "src"
        src_dir.mkdir()
        
        for i in range(100):
            (src_dir / f"file_{i}.py").write_text(f"# File {i}")
        
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path)
            }
        }
        
        result = await project_service.validate_project(project_id)
        
        assert result['is_valid'] is True
        assert result['metadata']['has_src_structure'] is True
    
    @pytest.mark.performance
    async def test_concurrent_project_operations(self, project_service, temp_directory):
        """Test concurrent project operations."""
        from tests.conftest import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Create multiple project paths
        project_paths = [
            temp_directory / f"concurrent_project_{i}"
            for i in range(10)
        ]
        
        with patch.object(project_service._project_manager, 'create_project') as mock_create:
            mock_create.return_value = {'id': 'test', 'name': 'Test'}
            
            monitor.start()
            
            # Create projects concurrently
            tasks = [
                project_service.create_project(
                    name=f"Project {i}",
                    path=path,
                    initialize_git=False,
                    create_venv=False
                )
                for i, path in enumerate(project_paths)
            ]
            
            results = await asyncio.gather(*tasks)
            
            monitor.stop()
            
            assert len(results) == 10
            assert all(result['name'].startswith("Project") for result in results)
            
            # Performance assertion
            monitor.assert_performance(5.0)  # Should complete within 5 seconds
    
    @pytest.mark.edge_case
    async def test_unicode_project_names(self, project_service, temp_directory):
        """Test project creation with Unicode names."""
        unicode_names = [
            "プロジェクト",  # Japanese
            "Проект",       # Russian  
            "Projet 🚀",    # French with emoji
            "项目测试",      # Chinese
        ]
        
        for i, name in enumerate(unicode_names):
            project_path = temp_directory / f"unicode_{i}"
            
            with patch.object(project_service._project_manager, 'create_project') as mock_create:
                mock_create.return_value = {'id': f'test-{i}', 'name': name}
                
                result = await project_service.create_project(
                    name=name,
                    path=project_path
                )
                
                assert result['name'] == name
                assert project_path.exists()
    
    @pytest.mark.edge_case
    async def test_project_info_with_broken_symlinks(self, project_service, temp_directory):
        """Test project info when directory contains broken symlinks."""
        project_path = temp_directory / "symlink_project" 
        project_path.mkdir()
        
        # Create a broken symlink (if possible on this system)
        try:
            broken_link = project_path / "broken_link"
            broken_link.symlink_to("/nonexistent/target")
        except OSError:
            pytest.skip("Cannot create symlinks on this system")
        
        project_id = str(uuid4())
        project_service._active_projects[project_id] = {
            'project_data': {
                'id': project_id,
                'path': str(project_path)
            }
        }
        
        # Should handle broken symlinks gracefully
        result = await project_service.get_project_info(project_id)
        
        assert result['path'] == str(project_path)
        assert 'runtime_info' in result