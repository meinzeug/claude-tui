"""
Project Manager - Core orchestration component for project lifecycle management.

This module implements the central ProjectManager class responsible for:
- Project creation and configuration
- Template management and scaffolding 
- Resource coordination and lifecycle management
- Integration with AI services and validation systems
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.core.state_manager import StateManager
from src.claude_tui.core.task_engine import TaskEngine
from src.claude_tui.core.progress_validator import ProgressValidator
from src.claude_tui.integrations.ai_interface import AIInterface
from src.claude_tui.utils.template_engine import TemplateEngine, TemplateContext
from src.claude_tui.utils.file_system import FileSystemManager
from src.claude_tui.models.project import Project, ProjectConfig, ProjectTemplate
from src.claude_tui.models.task import DevelopmentTask, TaskResult

logger = logging.getLogger(__name__)


@dataclass
class DevelopmentResult:
    """Result of a development workflow execution."""
    success: bool
    project: Project
    completed_tasks: List[DevelopmentTask]
    failed_tasks: List[DevelopmentTask]
    validation_results: Dict[str, Any]
    duration: float
    error_message: Optional[str] = None


class ProjectManager:
    """
    Central orchestrator for project management operations.
    
    The ProjectManager coordinates all aspects of project lifecycle including
    creation, configuration, development workflows, and resource management.
    It serves as the main integration point between UI, AI services, and 
    project operations.
    """
    
    def __init__(
        self, 
        config_manager: ConfigManager,
        state_manager: Optional[StateManager] = None,
        task_engine: Optional[TaskEngine] = None,
        ai_interface: Optional[AIInterface] = None,
        validator: Optional[ProgressValidator] = None
    ):
        """
        Initialize the ProjectManager with required components.
        
        Args:
            config_manager: Configuration management instance
            state_manager: Project state management (optional, will create if None)
            task_engine: Task execution engine (optional, will create if None) 
            ai_interface: AI service interface (optional, will create if None)
            validator: Progress validation system (optional, will create if None)
        """
        self.config_manager = config_manager
        self.state_manager = state_manager or StateManager()
        self.task_engine = task_engine or TaskEngine(config_manager)
        self.ai_interface = ai_interface or AIInterface(config_manager)
        self.validator = validator or ProgressValidator(config_manager)
        
        # Initialize supporting components
        # self.template_engine = TemplateEngine(config_manager)  # Module not implemented yet
        # self.file_system = FileSystemManager()  # Module not implemented yet
        
        # Runtime state
        self.current_project: Optional[Project] = None
        self._background_tasks: set = set()
        
        logger.info("ProjectManager initialized successfully")
    
    async def create_project(
        self, 
        template_name: str,
        project_name: str,
        output_directory: Path,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Project:
        """
        Create a new project with anti-hallucination validation.
        
        This method orchestrates the complete project creation workflow:
        1. Load and validate project template
        2. Generate project structure with AI assistance
        3. Apply anti-hallucination validation
        4. Initialize project state and configuration
        
        Args:
            template_name: Name of the project template to use
            project_name: Name for the new project
            output_directory: Directory where project will be created
            config_overrides: Optional configuration overrides
            
        Returns:
            Project: The created project instance
            
        Raises:
            ValueError: If template is invalid or project name conflicts
            IOError: If output directory is not accessible
            ValidationError: If generated project fails validation
        """
        start_time = datetime.now()
        logger.info(f"Creating project '{project_name}' from template '{template_name}'")
        
        try:
            # Step 1: Load and validate template
            template = await self.template_engine.load_template(template_name)
            if not template:
                raise ValueError(f"Template '{template_name}' not found")
            
            # Step 2: Prepare project configuration
            project_config = ProjectConfig(
                name=project_name,
                template=template_name,
                output_directory=output_directory,
                config_overrides=config_overrides or {}
            )
            
            # Step 3: Create project directory structure
            project_path = output_directory / project_name
            await self.file_system.create_directory_structure(
                base_path=project_path,
                structure=template.structure
            )
            
            # Step 4: Generate project files with AI
            generation_tasks = await self._create_generation_tasks(
                template, project_config, project_path
            )
            
            # Execute generation tasks with validation
            generation_results = await self.task_engine.execute_tasks(
                tasks=generation_tasks,
                validate=True
            )
            
            # Step 5: Create project instance
            project = Project(
                name=project_name,
                path=project_path,
                config=project_config,
                template=template,
                created_at=datetime.now()
            )
            
            # Step 6: Initialize project state
            await self.state_manager.initialize_project(project)
            
            # Step 7: Run post-creation validation
            validation_result = await self.validator.validate_project(project)
            if not validation_result.is_valid:
                logger.warning(f"Project validation issues: {validation_result.issues}")
                # Auto-fix issues if possible
                await self._auto_fix_project_issues(project, validation_result)
            
            # Step 8: Set as current project
            self.current_project = project
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Project '{project_name}' created successfully in {duration:.2f}s")
            
            return project
            
        except Exception as e:
            logger.error(f"Failed to create project '{project_name}': {e}")
            # Cleanup on failure
            if 'project_path' in locals():
                await self.file_system.cleanup_directory(project_path)
            raise
    
    async def orchestrate_development(
        self, 
        requirements: Dict[str, Any],
        project: Optional[Project] = None
    ) -> DevelopmentResult:
        """
        Main development workflow with continuous validation.
        
        This method coordinates complex development workflows that may involve
        multiple AI agents, validation checkpoints, and iterative refinement.
        
        Args:
            requirements: Development requirements and specifications
            project: Target project (uses current if None)
            
        Returns:
            DevelopmentResult: Complete workflow execution result
        """
        start_time = datetime.now()
        target_project = project or self.current_project
        
        if not target_project:
            raise ValueError("No project specified and no current project set")
        
        logger.info(f"Orchestrating development for project '{target_project.name}'")
        
        try:
            # Step 1: Analyze requirements and create task breakdown
            tasks = await self._analyze_and_create_tasks(
                requirements, target_project
            )
            
            # Step 2: Execute development workflow
            completed_tasks = []
            failed_tasks = []
            
            for task in tasks:
                try:
                    # Execute task with AI
                    result = await self.ai_interface.execute_development_task(
                        task=task,
                        project=target_project
                    )
                    
                    # Validate result immediately 
                    validation = await self.validator.validate_task_result(
                        task=task,
                        result=result,
                        project=target_project
                    )
                    
                    if validation.is_valid:
                        completed_tasks.append(task)
                        logger.debug(f"Task '{task.name}' completed successfully")
                    else:
                        # Attempt auto-correction
                        corrected_result = await self._auto_correct_task_result(
                            task, result, validation
                        )
                        
                        if corrected_result:
                            completed_tasks.append(task)
                            logger.info(f"Task '{task.name}' completed after auto-correction")
                        else:
                            failed_tasks.append(task)
                            logger.error(f"Task '{task.name}' failed validation")
                            
                except Exception as e:
                    logger.error(f"Task '{task.name}' execution failed: {e}")
                    failed_tasks.append(task)
            
            # Step 3: Final project validation
            final_validation = await self.validator.validate_project(
                target_project
            )
            
            # Step 4: Create result summary
            duration = (datetime.now() - start_time).total_seconds()
            success = len(failed_tasks) == 0 and final_validation.is_valid
            
            result = DevelopmentResult(
                success=success,
                project=target_project,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                validation_results={
                    'final_validation': final_validation.to_dict(),
                    'task_validations': []
                },
                duration=duration
            )
            
            if not success:
                result.error_message = f"Development workflow failed: {len(failed_tasks)} tasks failed"
            
            logger.info(f"Development workflow completed in {duration:.2f}s (success: {success})")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Development workflow failed: {e}")
            
            return DevelopmentResult(
                success=False,
                project=target_project,
                completed_tasks=[],
                failed_tasks=[],
                validation_results={},
                duration=duration,
                error_message=str(e)
            )
    
    async def load_project(self, project_path: Path) -> Project:
        """
        Load an existing project from disk.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Project: The loaded project instance
        """
        logger.info(f"Loading project from {project_path}")
        
        try:
            project = await self.state_manager.load_project(project_path)
            
            # Validate project integrity
            validation = await self.validator.validate_project(project)
            if not validation.is_valid:
                logger.warning(f"Loaded project has validation issues: {validation.issues}")
            
            self.current_project = project
            logger.info(f"Project '{project.name}' loaded successfully")
            
            return project
            
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            raise
    
    async def save_project(self, project: Optional[Project] = None) -> None:
        """
        Save project state to disk.
        
        Args:
            project: Project to save (uses current if None)
        """
        target_project = project or self.current_project
        if not target_project:
            raise ValueError("No project to save")
        
        await self.state_manager.save_project(target_project)
        logger.info(f"Project '{target_project.name}' saved successfully")
    
    async def get_project_status(self, project: Optional[Project] = None) -> Dict[str, Any]:
        """
        Get comprehensive project status information.
        
        Args:
            project: Project to check (uses current if None)
            
        Returns:
            Dict containing project status information
        """
        target_project = project or self.current_project
        if not target_project:
            return {'error': 'No project available'}
        
        try:
            # Get validation status
            validation = await self.validator.validate_project(target_project)
            
            # Get task engine status
            task_status = await self.task_engine.get_status()
            
            # Get file system stats
            fs_stats = await self.file_system.get_directory_stats(target_project.path)
            
            return {
                'project': {
                    'name': target_project.name,
                    'path': str(target_project.path),
                    'created_at': target_project.created_at.isoformat(),
                    'template': target_project.template.name if target_project.template else None
                },
                'validation': validation.to_dict(),
                'tasks': task_status,
                'filesystem': fs_stats,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get project status: {e}")
            return {'error': str(e)}
    
    async def cleanup(self) -> None:
        """
        Cleanup resources and save state.
        """
        logger.info("Cleaning up ProjectManager")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save current project if exists
        if self.current_project:
            await self.save_project()
        
        # Cleanup components
        await self.task_engine.cleanup()
        await self.ai_interface.cleanup()
        await self.validator.cleanup()
        
        logger.info("ProjectManager cleanup completed")
    
    # Private helper methods
    
    async def _create_generation_tasks(
        self, 
        template: ProjectTemplate,
        config: ProjectConfig,
        project_path: Path
    ) -> List[DevelopmentTask]:
        """
        Create file generation tasks from template.
        """
        tasks = []
        
        for file_spec in template.files:
            task = DevelopmentTask(
                name=f"generate_{file_spec.name}",
                description=f"Generate {file_spec.name} from template",
                file_path=project_path / file_spec.path,
                template_spec=file_spec,
                config=config
            )
            tasks.append(task)
        
        return tasks
    
    async def _analyze_and_create_tasks(
        self, 
        requirements: Dict[str, Any],
        project: Project
    ) -> List[DevelopmentTask]:
        """
        Analyze requirements and create development tasks.
        """
        # Use AI to analyze requirements and create task breakdown
        analysis_result = await self.ai_interface.analyze_requirements(
            requirements=requirements,
            project=project
        )
        
        return analysis_result.tasks
    
    async def _auto_fix_project_issues(
        self, 
        project: Project, 
        validation_result: Any
    ) -> None:
        """
        Attempt to automatically fix project issues.
        """
        for issue in validation_result.issues:
            if issue.auto_fixable:
                try:
                    await self.validator.auto_fix_issue(issue, project)
                    logger.info(f"Auto-fixed issue: {issue.description}")
                except Exception as e:
                    logger.warning(f"Failed to auto-fix issue '{issue.description}': {e}")
    
    async def _auto_correct_task_result(
        self, 
        task: DevelopmentTask,
        result: TaskResult,
        validation: Any
    ) -> Optional[TaskResult]:
        """
        Attempt to automatically correct task result issues.
        """
        try:
            correction_result = await self.ai_interface.correct_task_result(
                task=task,
                result=result,
                validation_issues=validation.issues
            )
            
            # Re-validate corrected result
            revalidation = await self.validator.validate_task_result(
                task, correction_result, task.project
            )
            
            if revalidation.is_valid:
                return correction_result
            else:
                logger.warning(f"Auto-correction failed for task '{task.name}'")
                return None
                
        except Exception as e:
            logger.error(f"Auto-correction error for task '{task.name}': {e}")
            return None