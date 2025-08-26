"""
Project Manager - Central Orchestration System for claude-tiu.

This module provides the main orchestration logic for project management,
coordinating all core components to deliver intelligent AI-powered development
workflows with continuous validation and anti-hallucination checks.

Key Features:
- Central project lifecycle management
- AI workflow orchestration with Claude Code/Flow integration
- Real-time progress monitoring with authenticity validation
- Template-based project generation
- Resource coordination and state management
- Recovery and error handling mechanisms
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from .ai_interface import AIInterface, AIContext
from .config_manager import ConfigManager
from .task_engine import TaskEngine, ExecutionStrategy, ExecutionContext, create_task
from .types import (
    ProjectState, Task, Workflow, DevelopmentResult,
    ProgressReport, ProgressMetrics, ValidationResult, Priority,
    ProjectConfig
)
from .validator import ProgressValidator


logger = logging.getLogger(__name__)


class ProjectManagerException(Exception):
    """Project manager related errors."""
    pass


class StateManager:
    """Manages project state persistence and recovery."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory for state persistence
        """
        self.state_dir = state_dir or Path.home() / '.claude-tiu' / 'state'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._project_states: Dict[UUID, Dict[str, Any]] = {}
    
    async def save_project_state(self, project_id: UUID, state_data: Dict[str, Any]) -> None:
        """Save project state to disk."""
        state_file = self.state_dir / f'project_{project_id}.json'
        
        # Add timestamp
        state_data['last_saved'] = datetime.utcnow().isoformat()
        
        # Cache in memory
        self._project_states[project_id] = state_data
        
        # Save to disk
        try:
            import json
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save project state: {e}")
    
    async def load_project_state(self, project_id: UUID) -> Optional[Dict[str, Any]]:
        """Load project state from disk."""
        # Check memory cache first
        if project_id in self._project_states:
            return self._project_states[project_id]
        
        state_file = self.state_dir / f'project_{project_id}.json'
        if not state_file.exists():
            return None
        
        try:
            import json
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                self._project_states[project_id] = state_data
                return state_data
        except Exception as e:
            logger.error(f"Failed to load project state: {e}")
            return None
    
    async def delete_project_state(self, project_id: UUID) -> None:
        """Delete project state."""
        # Remove from memory
        self._project_states.pop(project_id, None)
        
        # Remove from disk
        state_file = self.state_dir / f'project_{project_id}.json'
        state_file.unlink(missing_ok=True)


@dataclass
class Project:
    """Project model with comprehensive state management and tracking."""
    
    id: UUID
    name: str
    description: str = ""
    path: Optional[Path] = None
    config: Optional[ProjectConfig] = None
    state: ProjectState = ProjectState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Progress and validation tracking
    progress: ProgressMetrics = field(default_factory=ProgressMetrics)
    validation_history: List[ValidationResult] = field(default_factory=list)
    
    # Active workflows and files
    active_workflows: Dict[UUID, Workflow] = field(default_factory=dict)
    completed_workflows: List[UUID] = field(default_factory=list)
    generated_files: Set[str] = field(default_factory=set)
    modified_files: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.config:
            self.config = ProjectConfig(name=self.name)
    
    def update_progress(self, new_metrics: ProgressMetrics) -> None:
        """Update project progress metrics."""
        self.progress = new_metrics
        self.updated_at = datetime.utcnow()
    
    def add_validation_result(self, result: ValidationResult) -> None:
        """Add validation result to history."""
        self.validation_history.append(result)
        # Keep only last 50 validation results
        if len(self.validation_history) > 50:
            self.validation_history = self.validation_history[-50:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary for serialization."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'path': str(self.path) if self.path else None,
            'config': self.config.dict() if self.config else None,
            'progress': {
                'real_progress': self.progress.real_progress,
                'fake_progress': self.progress.fake_progress,
                'authenticity_rate': self.progress.authenticity_rate,
                'quality_score': self.progress.quality_score,
                'tasks_completed': self.progress.tasks_completed,
                'tasks_total': self.progress.tasks_total
            },
            'generated_files': list(self.generated_files),
            'modified_files': list(self.modified_files),
            'active_workflows': [str(wid) for wid in self.active_workflows.keys()],
            'completed_workflows': [str(wid) for wid in self.completed_workflows]
        }


class ProjectManager:
    """
    Central orchestrator for project management operations.
    
    Coordinates all core components to provide intelligent AI-powered development
    workflows with continuous validation and progress monitoring.
    """
    
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        enable_validation: bool = True,
        max_concurrent_tasks: int = 5
    ):
        """
        Initialize project manager.
        
        Args:
            config_dir: Configuration directory
            enable_validation: Enable anti-hallucination validation
            max_concurrent_tasks: Maximum concurrent task execution
        """
        # Initialize core components
        self.config_manager = ConfigManager(config_dir)
        self.state_manager = StateManager()
        self.task_engine = TaskEngine(self.config_manager)
        self.ai_interface = AIInterface(self.config_manager)
        self.validator = ProgressValidator() if enable_validation else None
        
        # Project management
        self._active_projects: Dict[UUID, Project] = {}
        self._project_templates: Dict[str, Dict[str, Any]] = {}
        
        # Load project templates
        self._load_builtin_templates()
    
    async def create_project(
        self,
        name: str,
        template: str = "basic",
        config_overrides: Optional[Dict[str, Any]] = None,
        project_path: Optional[Path] = None
    ) -> Project:
        """
        Create new project with anti-hallucination validation.
        
        Args:
            name: Project name
            template: Project template to use
            config_overrides: Configuration overrides
            project_path: Optional project path
            
        Returns:
            Created project instance
        """
        project_id = uuid4()
        logger.info(f"Creating project '{name}' with template '{template}' (ID: {project_id})")
        
        try:
            # Load template configuration
            template_config = await self._load_template_config(template)
            
            # Create project configuration
            project_config = self._create_project_config(
                name, template_config, config_overrides
            )
            
            # Determine project path
            if not project_path:
                project_path = Path.cwd() / name.lower().replace(' ', '-')
            
            # Create project directory
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Create project instance
            project = Project(
                id=project_id,
                name=name,
                description=project_config.description,
                config=project_config,
                path=project_path
            )
            
            # Generate initial project structure
            await self._generate_project_structure(project, template_config)
            
            # Register project
            self._active_projects[project_id] = project
            
            # Save initial state
            await self.state_manager.save_project_state(
                project_id, project.to_dict()
            )
            
            project.state = ProjectState.ACTIVE
            logger.info(f"Project '{name}' created successfully")
            
            return project
            
        except Exception as e:
            logger.error(f"Failed to create project '{name}': {e}")
            raise ProjectManagerException(f"Project creation failed: {e}") from e
    
    async def orchestrate_development(
        self,
        project_id: UUID,
        requirements: Dict[str, Any],
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    ) -> DevelopmentResult:
        """
        Main development workflow with continuous validation.
        
        Args:
            project_id: Project identifier
            requirements: Development requirements
            strategy: Execution strategy
            
        Returns:
            Development result with authenticity metrics
        """
        project = await self._get_project(project_id)
        if not project:
            raise ProjectManagerException(f"Project {project_id} not found")
        
        logger.info(f"Starting development workflow for project '{project.name}'")
        
        try:
            # Create development workflow from requirements
            workflow = await self._create_development_workflow(project, requirements)
            
            # Add workflow to project
            project.active_workflows[workflow.id] = workflow
            
            # Create execution context
            context = ExecutionContext(
                project_path=project.path,
                strategy=strategy,
                validate_results=True
            )
            
            # Execute workflow with task engine
            result = await self.task_engine.execute_workflow(workflow, context)
            
            # Update project with results
            await self._update_project_with_results(project, result)
            
            # Run comprehensive validation
            if self.validator and result.success:
                validation_result = await self.validator.validate_codebase(project.path)
                project.add_validation_result(validation_result)
                
                # Update authenticity metrics
                project.progress.authenticity_rate = validation_result.authenticity_score
                project.progress.update_authenticity_rate()
            
            # Move workflow to completed
            if workflow.id in project.active_workflows:
                del project.active_workflows[workflow.id]
                project.completed_workflows.append(workflow.id)
            
            # Save updated state
            await self.state_manager.save_project_state(
                project_id, project.to_dict()
            )
            
            logger.info(f"Development workflow completed for project '{project.name}'")
            return result
            
        except Exception as e:
            logger.error(f"Development workflow failed: {e}")
            project.state = ProjectState.FAILED
            raise ProjectManagerException(f"Development workflow failed: {e}") from e
    
    async def get_progress_report(self, project_id: UUID) -> ProgressReport:
        """
        Get comprehensive progress report with authenticity validation.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Detailed progress report
        """
        project = await self._get_project(project_id)
        if not project:
            raise ProjectManagerException(f"Project {project_id} not found")
        
        # Get latest validation if available
        latest_validation = (
            project.validation_history[-1] if project.validation_history
            else ValidationResult(is_authentic=True, authenticity_score=100.0, real_progress=0.0, fake_progress=0.0)
        )
        
        # Collect recent activities
        recent_activities = []
        if project.generated_files:
            recent_activities.append(f"Generated {len(project.generated_files)} files")
        if project.modified_files:
            recent_activities.append(f"Modified {len(project.modified_files)} files")
        if project.active_workflows:
            recent_activities.append(f"{len(project.active_workflows)} active workflows")
        
        # Gather performance data
        performance_data = {
            'total_workflows': len(project.completed_workflows) + len(project.active_workflows),
            'completed_workflows': len(project.completed_workflows),
            'validation_history_count': len(project.validation_history),
            'project_age_hours': (datetime.utcnow() - project.created_at).total_seconds() / 3600
        }
        
        return ProgressReport(
            project_id=project_id,
            metrics=project.progress,
            validation=latest_validation,
            recent_activities=recent_activities,
            performance_data=performance_data
        )
    
    async def validate_project_authenticity(
        self,
        project_id: UUID,
        deep_validation: bool = False
    ) -> ValidationResult:
        """
        Validate project authenticity and detect placeholders.
        
        Args:
            project_id: Project identifier
            deep_validation: Enable comprehensive validation
            
        Returns:
            Validation result with authenticity metrics
        """
        project = await self._get_project(project_id)
        if not project or not self.validator:
            raise ProjectManagerException("Project not found or validation disabled")
        
        logger.info(f"Starting authenticity validation for project '{project.name}'")
        
        # Configure validator for deep validation if requested
        if deep_validation:
            validator = ProgressValidator(
                enable_cross_validation=True,
                enable_execution_testing=True
            )
        else:
            validator = self.validator
        
        # Run validation on project codebase
        validation_result = await validator.validate_codebase(project.path)
        
        # Add to project history
        project.add_validation_result(validation_result)
        
        # Update project progress with validation results
        project.progress.authenticity_rate = validation_result.authenticity_score
        project.progress.update_authenticity_rate()
        
        # Auto-fix detected issues if enabled
        if (project.config.validation_config.get('auto_fix_enabled', True) and
            validation_result.issues):
            
            logger.info(f"Auto-fixing {len(validation_result.issues)} detected issues")
            
            fix_result = await validator.auto_fix_placeholders(
                validation_result.issues, project.path
            )
            
            logger.info(f"Auto-fix completed: {fix_result['fixed_count']} fixed, "
                       f"{fix_result['failed_count']} failed")
        
        # Save updated state
        await self.state_manager.save_project_state(
            project_id, project.to_dict()
        )
        
        logger.info(f"Authenticity validation completed: {validation_result.authenticity_score:.1f}%")
        return validation_result
    
    async def list_projects(
        self,
        state_filter: Optional[ProjectState] = None
    ) -> List[Dict[str, Any]]:
        """
        List all projects with optional state filtering.
        
        Args:
            state_filter: Optional state to filter by
            
        Returns:
            List of project summaries
        """
        projects = []
        
        for project in self._active_projects.values():
            if state_filter is None or project.state == state_filter:
                projects.append({
                    'id': str(project.id),
                    'name': project.name,
                    'description': project.description,
                    'state': project.state.value,
                    'progress': project.progress.real_progress,
                    'authenticity_rate': project.progress.authenticity_rate,
                    'created_at': project.created_at.isoformat(),
                    'updated_at': project.updated_at.isoformat()
                })
        
        return sorted(projects, key=lambda p: p['updated_at'], reverse=True)
    
    async def get_project_details(self, project_id: UUID) -> Optional[Dict[str, Any]]:
        """Get detailed project information."""
        project = await self._get_project(project_id)
        if not project:
            return None
        
        return project.to_dict()
    
    async def delete_project(self, project_id: UUID, remove_files: bool = False) -> bool:
        """
        Delete project and optionally remove files.
        
        Args:
            project_id: Project identifier
            remove_files: Whether to remove project files
            
        Returns:
            True if deletion was successful
        """
        project = await self._get_project(project_id)
        if not project:
            return False
        
        try:
            # Remove from active projects
            if project_id in self._active_projects:
                del self._active_projects[project_id]
            
            # Remove project state
            await self.state_manager.delete_project_state(project_id)
            
            # Remove files if requested
            if remove_files and project.path and project.path.exists():
                shutil.rmtree(project.path)
                logger.info(f"Removed project files: {project.path}")
            
            logger.info(f"Deleted project '{project.name}' (ID: {project_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            return False
    
    async def _get_project(self, project_id: UUID) -> Optional[Project]:
        """Get project by ID, loading from state if necessary."""
        if project_id in self._active_projects:
            return self._active_projects[project_id]
        
        # Try to load from saved state
        state_data = await self.state_manager.load_project_state(project_id)
        if state_data:
            # Reconstruct project from state
            project = await self._reconstruct_project_from_state(state_data)
            self._active_projects[project_id] = project
            return project
        
        return None
    
    async def _reconstruct_project_from_state(self, state_data: Dict[str, Any]) -> Project:
        """Reconstruct project from saved state data."""
        project_id = UUID(state_data['id'])
        
        # Create project config if available
        config = None
        if state_data.get('config'):
            config = ProjectConfig(**state_data['config'])
        
        project = Project(
            id=project_id,
            name=state_data['name'],
            description=state_data.get('description', ''),
            config=config,
            path=Path(state_data['path']) if state_data.get('path') else None
        )
        
        # Restore state
        project.state = ProjectState(state_data['state'])
        project.created_at = datetime.fromisoformat(state_data['created_at'])
        project.updated_at = datetime.fromisoformat(state_data['updated_at'])
        
        # Restore progress
        progress_data = state_data.get('progress', {})
        project.progress = ProgressMetrics(
            real_progress=progress_data.get('real_progress', 0.0),
            fake_progress=progress_data.get('fake_progress', 0.0),
            authenticity_rate=progress_data.get('authenticity_rate', 100.0),
            quality_score=progress_data.get('quality_score', 0.0),
            tasks_completed=progress_data.get('tasks_completed', 0),
            tasks_total=progress_data.get('tasks_total', 0)
        )
        
        # Restore file sets
        project.generated_files = set(state_data.get('generated_files', []))
        project.modified_files = set(state_data.get('modified_files', []))
        project.completed_workflows = [UUID(wid) for wid in state_data.get('completed_workflows', [])]
        
        return project
    
    async def _load_template_config(self, template_name: str) -> Dict[str, Any]:
        """Load template configuration."""
        if template_name in self._project_templates:
            return self._project_templates[template_name]
        
        # Try to load from config manager
        try:
            return self.config_manager.get_template_config(template_name)
        except:
            # Fallback to basic template
            return self._project_templates.get('basic', {})
    
    def _create_project_config(
        self,
        name: str,
        template_config: Dict[str, Any],
        config_overrides: Optional[Dict[str, Any]]
    ) -> ProjectConfig:
        """Create project configuration from template and overrides."""
        # Start with template config
        config_data = template_config.get('config', {})
        config_data['name'] = name
        
        # Apply overrides
        if config_overrides:
            config_data.update(config_overrides)
        
        return ProjectConfig(**config_data)
    
    async def _generate_project_structure(
        self,
        project: Project,
        template_config: Dict[str, Any]
    ) -> None:
        """Generate initial project structure from template."""
        if not project.path:
            return
        
        # Create directory structure
        directories = template_config.get('directories', [])
        for directory in directories:
            dir_path = project.path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create initial files
        files = template_config.get('files', {})
        for file_path, content_template in files.items():
            full_path = project.path / file_path
            
            # Process template content
            content = self._process_template_content(content_template, project)
            
            # Write file
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            
            project.generated_files.add(file_path)
    
    def _process_template_content(self, template: str, project: Project) -> str:
        """Process template content with project variables."""
        replacements = {
            '{{PROJECT_NAME}}': project.name,
            '{{PROJECT_DESCRIPTION}}': project.description,
            '{{FRAMEWORK}}': project.config.framework if project.config else 'python',
            '{{TIMESTAMP}}': datetime.utcnow().isoformat()
        }
        
        content = template
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        
        return content
    
    async def _create_development_workflow(
        self,
        project: Project,
        requirements: Dict[str, Any]
    ) -> Workflow:
        """Create development workflow from requirements."""
        workflow = Workflow(
            name=f"Development workflow for {project.name}",
            description=requirements.get('description', '')
        )
        
        # Create tasks based on requirements
        tasks = []
        
        # Basic setup tasks
        if requirements.get('setup_project', True):
            setup_task = create_task(
                name="Project Setup",
                description="Set up project structure and configuration",
                priority=Priority.HIGH,
                estimated_duration=30
            )
            tasks.append(setup_task)
        
        # Feature implementation tasks
        features = requirements.get('features', [])
        for feature in features:
            feature_task = create_task(
                name=f"Implement {feature}",
                description=f"Implement {feature} functionality",
                priority=Priority.MEDIUM,
                estimated_duration=120
            )
            tasks.append(feature_task)
        
        # Testing tasks
        if requirements.get('include_tests', True):
            test_task = create_task(
                name="Create Tests",
                description="Create comprehensive test suite",
                priority=Priority.MEDIUM,
                estimated_duration=90
            )
            tasks.append(test_task)
        
        # Add tasks to workflow
        for task in tasks:
            workflow.add_task(task)
        
        return workflow
    
    async def _update_project_with_results(
        self,
        project: Project,
        result: DevelopmentResult
    ) -> None:
        """Update project with workflow execution results."""
        # Update generated/modified files
        if result.files_generated:
            project.generated_files.update(result.files_generated)
        
        # Update progress metrics
        if result.quality_metrics:
            project.progress.quality_score = result.quality_metrics.get('quality_score', 0.0)
            project.progress.authenticity_rate = result.quality_metrics.get('authenticity_rate', 100.0)
        
        # Update task counts
        project.progress.tasks_completed = len(result.tasks_executed)
        project.progress.tasks_total = len(result.tasks_executed)
        project.progress.update_authenticity_rate()
        
        project.updated_at = datetime.utcnow()
    
    def _load_builtin_templates(self) -> None:
        """Load built-in project templates."""
        self._project_templates = {
            'basic': {
                'name': 'Basic Python Project',
                'description': 'Basic Python project structure',
                'directories': ['src', 'tests', 'docs'],
                'files': {
                    'README.md': '# {{PROJECT_NAME}}\n\n{{PROJECT_DESCRIPTION}}\n',
                    'requirements.txt': '# Project dependencies\n',
                    'src/__init__.py': '"""{{PROJECT_NAME}} package."""\n',
                    'tests/__init__.py': '"""Tests for {{PROJECT_NAME}}."""\n'
                },
                'config': {
                    'framework': 'python',
                    'features': ['database']
                }
            },
            'fastapi': {
                'name': 'FastAPI Project',
                'description': 'FastAPI web application project',
                'directories': ['app', 'tests', 'docs', 'alembic'],
                'files': {
                    'README.md': '# {{PROJECT_NAME}}\n\nFastAPI application\n',
                    'requirements.txt': 'fastapi>=0.104.0\nuvicorn>=0.24.0\npydantic>=2.0.0\n',
                    'app/main.py': '''"""{{PROJECT_NAME}} FastAPI application."""
from fastapi import FastAPI

app = FastAPI(
    title="{{PROJECT_NAME}}",
    description="{{PROJECT_DESCRIPTION}}"
)

@app.get("/")
async def root():
    return {"message": "Hello {{PROJECT_NAME}}"}
''',
                    'app/__init__.py': '"""{{PROJECT_NAME}} application."""\n'
                },
                'config': {
                    'framework': 'fastapi',
                    'database': 'postgresql',
                    'features': ['api', 'database', 'testing']
                }
            },
            'react': {
                'name': 'React Application',
                'description': 'React frontend application',
                'directories': ['src', 'public', 'tests'],
                'files': {
                    'README.md': '# {{PROJECT_NAME}}\n\nReact application\n',
                    'package.json': '''{
  "name": "{{PROJECT_NAME}}",
  "version": "1.0.0",
  "description": "{{PROJECT_DESCRIPTION}}",
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test"
  },
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  }
}''',
                    'src/App.js': '''import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>{{PROJECT_NAME}}</h1>
      <p>{{PROJECT_DESCRIPTION}}</p>
    </div>
  );
}

export default App;
'''
                },
                'config': {
                    'framework': 'react',
                    'features': ['frontend', 'components', 'testing']
                }
            }
        }


# Utility functions for external use

async def create_simple_project(
    name: str,
    template: str = "basic",
    path: Optional[Path] = None
) -> Project:
    """
    Convenience function to create a simple project.
    
    Args:
        name: Project name
        template: Project template
        path: Optional project path
        
    Returns:
        Created project
    """
    manager = ProjectManager()
    return await manager.create_project(name, template, project_path=path)


async def validate_project_quality(project_path: Path) -> ValidationResult:
    """
    Convenience function to validate project quality.
    
    Args:
        project_path: Path to project directory
        
    Returns:
        Validation result
    """
    validator = ProgressValidator()
    return await validator.validate_codebase(project_path)