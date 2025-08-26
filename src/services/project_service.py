"""
Project Service for claude-tui.

Manages project lifecycle, configuration, and operations:
- Project creation and initialization
- Configuration management
- File structure management
- Project validation and health checks
- Integration with AI services for project assistance
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from ..core.exceptions import (
    ConfigurationError, FileSystemError, ProjectDirectoryError, ValidationError
)
from ..core.project_manager import ProjectManager
from .base import BaseService


class ProjectService(BaseService):
    """
    Project Operations Service.
    
    Provides high-level project management operations with
    dependency injection and service coordination.
    """
    
    def __init__(self):
        super().__init__()
        self._project_manager: Optional[ProjectManager] = None
        self._active_projects: Dict[str, Dict[str, Any]] = {}
        self._project_cache: Dict[str, Any] = {}
        
    async def _initialize_impl(self) -> None:
        """Initialize project service."""
        try:
            # Initialize project manager
            self._project_manager = ProjectManager()
            
            # Load active projects from configuration
            await self._load_active_projects()
            
            self.logger.info("Project service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize project service: {str(e)}")
            raise ConfigurationError(f"Project service initialization failed: {str(e)}")
    
    async def _load_active_projects(self) -> None:
        """Load active projects from storage."""
        try:
            # Load from project manager's active projects
            if self._project_manager:
                active_projects = await self._project_manager.get_active_projects()
                for project in active_projects:
                    project_id = str(project.get('id', uuid4()))
                    self._active_projects[project_id] = {
                        'project_data': project,
                        'last_accessed': datetime.utcnow().isoformat(),
                        'status': 'active'
                    }
                    
                self.logger.info(f"Loaded {len(self._active_projects)} active projects")
                
        except Exception as e:
            self.logger.warning(f"Could not load active projects: {str(e)}")
            # Continue without loading projects
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check with project-specific status."""
        base_health = await super().health_check()
        
        base_health.update({
            'project_manager_available': self._project_manager is not None,
            'active_projects_count': len(self._active_projects),
            'cache_size': len(self._project_cache)
        })
        
        return base_health
    
    async def create_project(
        self,
        name: str,
        path: Union[str, Path],
        project_type: str = 'python',
        template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        initialize_git: bool = True,
        create_venv: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new project with specified configuration.
        
        Args:
            name: Project name
            path: Project directory path
            project_type: Type of project (python, nodejs, etc.)
            template: Template to use for project structure
            config: Additional configuration options
            initialize_git: Initialize git repository
            create_venv: Create virtual environment (Python projects)
            
        Returns:
            Created project information
        """
        return await self.execute_with_monitoring(
            'create_project',
            self._create_project_impl,
            name=name,
            path=path,
            project_type=project_type,
            template=template,
            config=config,
            initialize_git=initialize_git,
            create_venv=create_venv
        )
    
    async def _create_project_impl(
        self,
        name: str,
        path: Union[str, Path],
        project_type: str,
        template: Optional[str],
        config: Optional[Dict[str, Any]],
        initialize_git: bool,
        create_venv: bool
    ) -> Dict[str, Any]:
        """Internal project creation implementation."""
        if not self._project_manager:
            raise ConfigurationError("Project manager not initialized")
        
        project_path = Path(path).resolve()
        
        # Validate project parameters
        if not name or not name.strip():
            raise ValidationError("Project name cannot be empty")
        
        if project_path.exists():
            raise ProjectDirectoryError(str(project_path), "Directory already exists")
        
        try:
            # Create project directory
            project_path.mkdir(parents=True, exist_ok=False)
            
            # Generate project ID
            project_id = str(uuid4())
            
            # Prepare project configuration
            project_config = {
                'id': project_id,
                'name': name,
                'path': str(project_path),
                'type': project_type,
                'template': template,
                'created_at': datetime.utcnow().isoformat(),
                'config': config or {},
                'git_initialized': False,
                'venv_created': False,
                'status': 'creating'
            }
            
            # Create project using project manager
            result = await self._project_manager.create_project(
                name=name,
                project_type=project_type,
                path=project_path,
                template=template
            )
            
            # Initialize git repository if requested
            if initialize_git:
                try:
                    await self._initialize_git_repo(project_path)
                    project_config['git_initialized'] = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize git repo: {str(e)}")
            
            # Create virtual environment if requested (Python projects)
            if create_venv and project_type.lower() == 'python':
                try:
                    await self._create_virtual_environment(project_path)
                    project_config['venv_created'] = True
                except Exception as e:
                    self.logger.warning(f"Failed to create virtual environment: {str(e)}")
            
            # Save project configuration
            await self._save_project_config(project_path, project_config)
            
            # Add to active projects
            self._active_projects[project_id] = {
                'project_data': project_config,
                'last_accessed': datetime.utcnow().isoformat(),
                'status': 'active'
            }
            
            project_config['status'] = 'created'
            
            self.logger.info(f"Successfully created project: {name} at {project_path}")
            
            return {
                'project_id': project_id,
                'name': name,
                'path': str(project_path),
                'type': project_type,
                'config': project_config,
                'created_at': project_config['created_at'],
                'manager_result': result
            }
            
        except Exception as e:
            # Cleanup on failure
            if project_path.exists():
                try:
                    shutil.rmtree(project_path)
                except Exception as cleanup_error:
                    self.logger.error(f"Failed to cleanup failed project: {cleanup_error}")
            
            if isinstance(e, (ConfigurationError, ValidationError, ProjectDirectoryError)):
                raise
            else:
                raise ProjectDirectoryError(str(project_path), f"Project creation failed: {str(e)}")
    
    async def load_project(
        self,
        path: Union[str, Path],
        validate_structure: bool = True
    ) -> Dict[str, Any]:
        """
        Load existing project from path.
        
        Args:
            path: Path to project directory
            validate_structure: Whether to validate project structure
            
        Returns:
            Loaded project information
        """
        return await self.execute_with_monitoring(
            'load_project',
            self._load_project_impl,
            path=path,
            validate_structure=validate_structure
        )
    
    async def _load_project_impl(
        self,
        path: Union[str, Path],
        validate_structure: bool
    ) -> Dict[str, Any]:
        """Internal project loading implementation."""
        if not self._project_manager:
            raise ConfigurationError("Project manager not initialized")
        
        project_path = Path(path).resolve()
        
        if not project_path.exists():
            raise ProjectDirectoryError(str(project_path), "Project directory does not exist")
        
        if not project_path.is_dir():
            raise ProjectDirectoryError(str(project_path), "Path is not a directory")
        
        try:
            # Load project using project manager
            project_data = await self._project_manager.load_project(project_path)
            
            # Load project configuration if exists
            config_path = project_path / '.claude-tiu' / 'project.json'
            project_config = {}
            
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        project_config = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load project config: {str(e)}")
            
            # Generate project ID if not exists
            project_id = project_config.get('id', str(uuid4()))
            
            # Validate project structure if requested
            validation_result = {}
            if validate_structure:
                validation_result = await self._validate_project_structure(project_path)
            
            # Update project data
            project_info = {
                'project_id': project_id,
                'name': project_data.get('name', project_path.name),
                'path': str(project_path),
                'type': project_data.get('type', 'unknown'),
                'config': project_config,
                'loaded_at': datetime.utcnow().isoformat(),
                'validation': validation_result,
                'manager_data': project_data
            }
            
            # Add to active projects
            self._active_projects[project_id] = {
                'project_data': project_info,
                'last_accessed': datetime.utcnow().isoformat(),
                'status': 'active'
            }
            
            self.logger.info(f"Successfully loaded project: {project_info['name']}")
            
            return project_info
            
        except Exception as e:
            if isinstance(e, (ConfigurationError, ProjectDirectoryError)):
                raise
            else:
                raise ProjectDirectoryError(str(project_path), f"Failed to load project: {str(e)}")
    
    async def get_project_info(self, project_id: str) -> Dict[str, Any]:
        """Get detailed project information."""
        if project_id not in self._active_projects:
            raise ValidationError(f"Project {project_id} not found in active projects")
        
        project_info = self._active_projects[project_id]['project_data'].copy()
        
        # Update last accessed
        self._active_projects[project_id]['last_accessed'] = datetime.utcnow().isoformat()
        
        # Add runtime information
        project_path = Path(project_info['path'])
        if project_path.exists():
            project_info['runtime_info'] = {
                'exists': True,
                'is_git_repo': (project_path / '.git').exists(),
                'has_venv': any([
                    (project_path / 'venv').exists(),
                    (project_path / '.venv').exists(),
                    (project_path / 'env').exists()
                ]),
                'file_count': len(list(project_path.rglob('*'))),
                'size_mb': sum(f.stat().st_size for f in project_path.rglob('*') if f.is_file()) / (1024 * 1024)
            }
        else:
            project_info['runtime_info'] = {'exists': False}
        
        return project_info
    
    async def list_active_projects(self) -> List[Dict[str, Any]]:
        """List all active projects."""
        projects = []
        
        for project_id, project_entry in self._active_projects.items():
            project_data = project_entry['project_data'].copy()
            project_data['last_accessed'] = project_entry['last_accessed']
            project_data['entry_status'] = project_entry['status']
            projects.append(project_data)
        
        # Sort by last accessed time
        projects.sort(key=lambda x: x['last_accessed'], reverse=True)
        
        return projects
    
    async def validate_project(self, project_id: str) -> Dict[str, Any]:
        """Validate project structure and configuration."""
        project_info = await self.get_project_info(project_id)
        project_path = Path(project_info['path'])
        
        return await self._validate_project_structure(project_path)
    
    async def _validate_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Internal project structure validation."""
        validation_result = {
            'is_valid': True,
            'score': 1.0,
            'issues': [],
            'recommendations': [],
            'metadata': {
                'path': str(project_path),
                'validated_at': datetime.utcnow().isoformat()
            }
        }
        
        try:
            if not project_path.exists():
                validation_result.update({
                    'is_valid': False,
                    'score': 0.0,
                    'issues': ['Project directory does not exist']
                })
                return validation_result
            
            # Check for common project files/directories
            common_files = [
                'README.md', 'README.rst', 'LICENSE', 'setup.py', 'pyproject.toml',
                'package.json', 'requirements.txt', 'Dockerfile', '.gitignore'
            ]
            
            found_files = []
            for file_name in common_files:
                if (project_path / file_name).exists():
                    found_files.append(file_name)
            
            if not found_files:
                validation_result['recommendations'].append('Consider adding README.md and other standard project files')
                validation_result['score'] *= 0.8
            
            # Check for source code structure
            src_directories = ['src', 'lib', 'app', 'source']
            has_src_structure = any((project_path / src_dir).exists() for src_dir in src_directories)
            
            if not has_src_structure:
                validation_result['recommendations'].append('Consider organizing source code in dedicated directories')
                validation_result['score'] *= 0.9
            
            # Check for tests
            test_patterns = ['test', 'tests', 'spec', 'specs']
            has_tests = any(
                list(project_path.glob(f'**/{pattern}*')) or 
                list(project_path.glob(f'*{pattern}*'))
                for pattern in test_patterns
            )
            
            if not has_tests:
                validation_result['recommendations'].append('Consider adding tests to your project')
                validation_result['score'] *= 0.9
            
            # Check for configuration management
            config_patterns = ['.claude-tiu', 'config', 'configs', 'settings']
            has_config = any((project_path / pattern).exists() for pattern in config_patterns)
            
            if not has_config:
                validation_result['recommendations'].append('Consider adding configuration management')
                validation_result['score'] *= 0.95
            
            validation_result['metadata']['found_files'] = found_files
            validation_result['metadata']['has_src_structure'] = has_src_structure
            validation_result['metadata']['has_tests'] = has_tests
            validation_result['metadata']['has_config'] = has_config
            
        except Exception as e:
            validation_result.update({
                'is_valid': False,
                'score': 0.0,
                'issues': [f'Validation error: {str(e)}']
            })
        
        return validation_result
    
    async def update_project_config(
        self,
        project_id: str,
        config_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update project configuration."""
        if project_id not in self._active_projects:
            raise ValidationError(f"Project {project_id} not found")
        
        project_data = self._active_projects[project_id]['project_data']
        project_path = Path(project_data['path'])
        
        # Update configuration
        project_data['config'].update(config_updates)
        project_data['updated_at'] = datetime.utcnow().isoformat()
        
        # Save updated configuration
        await self._save_project_config(project_path, project_data)
        
        self.logger.info(f"Updated configuration for project {project_id}")
        
        return project_data
    
    async def _save_project_config(self, project_path: Path, config: Dict[str, Any]) -> None:
        """Save project configuration to file."""
        config_dir = project_path / '.claude-tiu'
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / 'project.json'
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            raise FileSystemError(f"Failed to save project config: {str(e)}", str(config_path))
    
    async def _initialize_git_repo(self, project_path: Path) -> None:
        """Initialize git repository in project."""
        process = await asyncio.create_subprocess_exec(
            'git', 'init',
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise FileSystemError(
                f"Git initialization failed: {stderr.decode()}",
                str(project_path)
            )
    
    async def _create_virtual_environment(self, project_path: Path) -> None:
        """Create Python virtual environment."""
        venv_path = project_path / 'venv'
        
        process = await asyncio.create_subprocess_exec(
            'python', '-m', 'venv', str(venv_path),
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise FileSystemError(
                f"Virtual environment creation failed: {stderr.decode()}",
                str(venv_path)
            )
    
    async def remove_project(
        self,
        project_id: str,
        delete_files: bool = False
    ) -> Dict[str, Any]:
        """Remove project from active projects."""
        if project_id not in self._active_projects:
            raise ValidationError(f"Project {project_id} not found")
        
        project_data = self._active_projects[project_id]['project_data']
        project_path = Path(project_data['path'])
        
        # Remove from active projects
        removed_project = self._active_projects.pop(project_id)
        
        result = {
            'project_id': project_id,
            'removed_at': datetime.utcnow().isoformat(),
            'files_deleted': False
        }
        
        # Delete files if requested
        if delete_files and project_path.exists():
            try:
                shutil.rmtree(project_path)
                result['files_deleted'] = True
                self.logger.info(f"Deleted project files for {project_id}")
            except Exception as e:
                self.logger.error(f"Failed to delete project files: {str(e)}")
                result['deletion_error'] = str(e)
        
        self.logger.info(f"Removed project {project_id} from active projects")
        
        return result