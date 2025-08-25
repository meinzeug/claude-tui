"""
Workflow Integration Manager - Live AI workflow integration with anti-hallucination.

Seamlessly integrates anti-hallucination detection into actual AI workflows:
- Claude Code API call interception
- Task engine validation integration  
- Project manager validation hooks
- Development workflow live scanning
- Performance-optimized validation pipeline
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.validation.anti_hallucination_engine import AntiHallucinationEngine
from claude_tiu.validation.real_time_validator import RealTimeValidator, ValidationMode
from claude_tiu.integrations.anti_hallucination_integration import AntiHallucinationIntegration
from claude_tiu.models.project import Project
from claude_tiu.models.task import DevelopmentTask, TaskResult

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages in the AI workflow where validation can be applied."""
    PRE_EXECUTION = "pre_execution"         # Before AI call
    DURING_GENERATION = "during_generation" # During streaming response
    POST_GENERATION = "post_generation"     # After AI response
    TASK_COMPLETION = "task_completion"     # When task completes
    PROJECT_UPDATE = "project_update"       # When project files change


@dataclass
class WorkflowValidationConfig:
    """Configuration for workflow validation integration."""
    enabled_stages: List[WorkflowStage] = field(default_factory=lambda: [
        WorkflowStage.PRE_EXECUTION,
        WorkflowStage.POST_GENERATION,
        WorkflowStage.TASK_COMPLETION
    ])
    intercept_api_calls: bool = True
    validate_streaming: bool = True
    auto_fix_on_validation_fail: bool = True
    block_invalid_commits: bool = True
    performance_monitoring: bool = True
    max_validation_time_ms: int = 200
    batch_validation: bool = True
    cache_validation_results: bool = True


@dataclass
class WorkflowValidationResult:
    """Result from workflow validation with context."""
    validation_id: str
    workflow_stage: WorkflowStage
    is_valid: bool
    authenticity_score: float
    processing_time_ms: float
    issues_detected: int
    auto_fixes_applied: int
    blocked_execution: bool = False
    validation_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class WorkflowIntegrationManager:
    """
    Manages integration of anti-hallucination validation into AI workflows.
    
    Provides seamless validation integration across the entire AI development
    workflow with performance optimization and automatic error correction.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        engine: AntiHallucinationEngine,
        integration: AntiHallucinationIntegration
    ):
        """Initialize the workflow integration manager."""
        self.config_manager = config_manager
        self.engine = engine
        self.integration = integration
        self.real_time_validator = RealTimeValidator(config_manager, engine)
        
        # Configuration
        self.config = WorkflowValidationConfig()
        
        # Workflow hooks and interceptors
        self.api_interceptors: Dict[str, Callable] = {}
        self.workflow_hooks: Dict[WorkflowStage, List[Callable]] = {
            stage: [] for stage in WorkflowStage
        }
        
        # Active validations tracking
        self.active_validations: Dict[str, WorkflowValidationResult] = {}
        
        # Performance metrics
        self.workflow_metrics = {
            'total_workflow_validations': 0,
            'validations_by_stage': {stage.value: 0 for stage in WorkflowStage},
            'avg_validation_time_by_stage': {stage.value: 0.0 for stage in WorkflowStage},
            'blocked_executions': 0,
            'auto_fixes_applied': 0
        }
        
        logger.info("Workflow Integration Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the workflow integration manager."""
        logger.info("Initializing Workflow Integration Manager")
        
        try:
            # Load configuration
            await self._load_config()
            
            # Initialize real-time validator
            await self.real_time_validator.initialize()
            
            # Setup API interceptors
            await self._setup_api_interceptors()
            
            # Setup workflow stage hooks
            await self._setup_workflow_hooks()
            
            # Initialize performance monitoring
            await self._initialize_performance_monitoring()
            
            logger.info("Workflow Integration Manager ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Workflow Integration Manager: {e}")
            raise
    
    async def intercept_claude_code_call(
        self,
        prompt: str,
        context: Dict[str, Any],
        project: Optional[Project] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Intercept Claude Code API call for pre-execution validation.
        
        Args:
            prompt: The prompt being sent to Claude Code
            context: Call context
            project: Associated project
            
        Returns:
            Tuple of (should_proceed, validation_metadata)
        """
        if WorkflowStage.PRE_EXECUTION not in self.config.enabled_stages:
            return True, {}
        
        validation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Intercepting Claude Code call: {validation_id}")
        
        try:
            # Pre-execution validation
            validation_context = {
                **context,
                'workflow_stage': 'pre_execution',
                'api_call_type': 'claude_code',
                'project_name': project.name if project else None
            }
            
            # Validate prompt quality
            prompt_validation = await self.real_time_validator.validate_live(
                prompt,
                validation_context,
                ValidationMode.API_CALL
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create workflow validation result
            workflow_result = WorkflowValidationResult(
                validation_id=validation_id,
                workflow_stage=WorkflowStage.PRE_EXECUTION,
                is_valid=prompt_validation.is_valid,
                authenticity_score=prompt_validation.authenticity_score,
                processing_time_ms=processing_time,
                issues_detected=len(prompt_validation.issues_detected),
                auto_fixes_applied=0,
                validation_data={
                    'prompt_length': len(prompt),
                    'context_size': len(str(context)),
                    'prompt_validation': prompt_validation.__dict__
                }
            )
            
            # Store active validation
            self.active_validations[validation_id] = workflow_result
            
            # Run pre-execution hooks
            await self._run_workflow_hooks(WorkflowStage.PRE_EXECUTION, {
                'validation_result': workflow_result,
                'prompt': prompt,
                'context': context,
                'project': project
            })
            
            # Update metrics
            self._update_workflow_metrics(workflow_result)
            
            # Determine if execution should proceed
            should_proceed = prompt_validation.is_valid or not self.config.block_invalid_commits
            
            if not should_proceed:
                workflow_result.blocked_execution = True
                self.workflow_metrics['blocked_executions'] += 1
                logger.warning(f"Blocked Claude Code call due to validation failure: {validation_id}")
            
            return should_proceed, {
                'validation_id': validation_id,
                'authenticity_score': prompt_validation.authenticity_score,
                'issues_detected': len(prompt_validation.issues_detected)
            }
            
        except Exception as e:
            logger.error(f"API interception failed: {e}")
            # Allow execution on validation failure
            return True, {'error': str(e)}
    
    async def validate_ai_response(
        self,
        response_content: str,
        context: Dict[str, Any],
        project: Optional[Project] = None,
        task: Optional[DevelopmentTask] = None
    ) -> WorkflowValidationResult:
        """
        Validate AI response after generation.
        
        Args:
            response_content: AI-generated content
            context: Response context
            project: Associated project
            task: Associated task
            
        Returns:
            WorkflowValidationResult with validation details
        """
        validation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Validating AI response: {validation_id}")
        
        try:
            # Validate AI response content
            validation_context = {
                **context,
                'workflow_stage': 'post_generation',
                'task_id': task.id if task else None,
                'project_name': project.name if project else None
            }
            
            response_validation = await self.real_time_validator.validate_live(
                response_content,
                validation_context,
                ValidationMode.API_CALL
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Apply auto-fixes if needed
            auto_fixes_applied = 0
            fixed_content = response_content
            
            if not response_validation.is_valid and self.config.auto_fix_on_validation_fail:
                fix_applied, fixed_content = await self.real_time_validator.apply_auto_fixes(
                    response_content,
                    response_validation,
                    validation_context
                )
                
                if fix_applied:
                    auto_fixes_applied = 1
                    logger.info(f"Auto-fixes applied to AI response: {validation_id}")
            
            # Create workflow validation result
            workflow_result = WorkflowValidationResult(
                validation_id=validation_id,
                workflow_stage=WorkflowStage.POST_GENERATION,
                is_valid=response_validation.is_valid,
                authenticity_score=response_validation.authenticity_score,
                processing_time_ms=processing_time,
                issues_detected=len(response_validation.issues_detected),
                auto_fixes_applied=auto_fixes_applied,
                validation_data={
                    'response_length': len(response_content),
                    'fixed_content': fixed_content if auto_fixes_applied else None,
                    'response_validation': response_validation.__dict__
                }
            )
            
            # Store active validation
            self.active_validations[validation_id] = workflow_result
            
            # Run post-generation hooks
            await self._run_workflow_hooks(WorkflowStage.POST_GENERATION, {
                'validation_result': workflow_result,
                'response_content': response_content,
                'fixed_content': fixed_content,
                'context': context,
                'project': project,
                'task': task
            })
            
            # Update metrics
            self._update_workflow_metrics(workflow_result)
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"AI response validation failed: {e}")
            
            # Return error result
            return WorkflowValidationResult(
                validation_id=validation_id,
                workflow_stage=WorkflowStage.POST_GENERATION,
                is_valid=False,
                authenticity_score=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                issues_detected=1,
                auto_fixes_applied=0,
                validation_data={'error': str(e)}
            )
    
    async def validate_task_completion(
        self,
        task: DevelopmentTask,
        task_result: TaskResult,
        project: Project
    ) -> WorkflowValidationResult:
        """
        Validate task completion with comprehensive analysis.
        
        Args:
            task: Completed development task
            task_result: Task execution result
            project: Associated project
            
        Returns:
            WorkflowValidationResult for task completion
        """
        validation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Validating task completion: {task.name} ({validation_id})")
        
        try:
            # Validate task result content
            validation_context = {
                'workflow_stage': 'task_completion',
                'task_name': task.name,
                'task_type': task.task_type.value,
                'project_name': project.name
            }
            
            # Use the integration's task result validation
            integration_result = await self.integration.validate_task_result(
                task, task_result, project
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create workflow validation result
            workflow_result = WorkflowValidationResult(
                validation_id=validation_id,
                workflow_stage=WorkflowStage.TASK_COMPLETION,
                is_valid=integration_result.is_valid,
                authenticity_score=integration_result.authenticity_score,
                processing_time_ms=processing_time,
                issues_detected=len(integration_result.issues),
                auto_fixes_applied=0,  # Auto-fixes are handled by integration
                validation_data={
                    'task_id': task.id,
                    'task_success': task_result.success,
                    'generated_content_length': len(task_result.generated_content or ''),
                    'integration_result': integration_result.__dict__
                }
            )
            
            # Store active validation
            self.active_validations[validation_id] = workflow_result
            
            # Run task completion hooks
            await self._run_workflow_hooks(WorkflowStage.TASK_COMPLETION, {
                'validation_result': workflow_result,
                'task': task,
                'task_result': task_result,
                'project': project,
                'integration_result': integration_result
            })
            
            # Update metrics
            self._update_workflow_metrics(workflow_result)
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Task completion validation failed: {e}")
            
            return WorkflowValidationResult(
                validation_id=validation_id,
                workflow_stage=WorkflowStage.TASK_COMPLETION,
                is_valid=False,
                authenticity_score=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                issues_detected=1,
                auto_fixes_applied=0,
                validation_data={'error': str(e)}
            )
    
    async def validate_project_changes(
        self,
        project: Project,
        changed_files: List[Path],
        change_context: Dict[str, Any] = None
    ) -> List[WorkflowValidationResult]:
        """
        Validate project file changes for continuous integration.
        
        Args:
            project: Project being changed
            changed_files: List of changed files
            change_context: Context about the changes
            
        Returns:
            List of WorkflowValidationResult for each validated file
        """
        logger.info(f"Validating project changes: {project.name} ({len(changed_files)} files)")
        
        results = []
        change_context = change_context or {}
        
        try:
            # Batch validation for performance
            if self.config.batch_validation:
                validation_tasks = []
                
                for file_path in changed_files:
                    task = self._validate_file_change(
                        project, file_path, change_context
                    )
                    validation_tasks.append(task)
                
                # Wait for all validations
                batch_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"File validation failed: {result}")
                        # Create error result
                        error_result = WorkflowValidationResult(
                            validation_id=str(uuid.uuid4()),
                            workflow_stage=WorkflowStage.PROJECT_UPDATE,
                            is_valid=False,
                            authenticity_score=0.0,
                            processing_time_ms=0.0,
                            issues_detected=1,
                            auto_fixes_applied=0,
                            validation_data={'error': str(result)}
                        )
                        results.append(error_result)
                    else:
                        results.append(result)
            else:
                # Sequential validation
                for file_path in changed_files:
                    result = await self._validate_file_change(
                        project, file_path, change_context
                    )
                    results.append(result)
            
            # Run project update hooks
            await self._run_workflow_hooks(WorkflowStage.PROJECT_UPDATE, {
                'project': project,
                'changed_files': changed_files,
                'validation_results': results,
                'change_context': change_context
            })
            
            # Update metrics for all results
            for result in results:
                self._update_workflow_metrics(result)
            
            valid_files = sum(1 for r in results if r.is_valid)
            logger.info(f"Project validation completed: {valid_files}/{len(results)} files valid")
            
            return results
            
        except Exception as e:
            logger.error(f"Project changes validation failed: {e}")
            return []
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get comprehensive workflow integration metrics."""
        # Get base metrics
        real_time_metrics = await self.real_time_validator.get_performance_metrics()
        integration_metrics = await self.integration.get_integration_metrics()
        
        # Add workflow-specific metrics
        workflow_metrics = {
            'workflow_integration': {
                'total_workflow_validations': self.workflow_metrics['total_workflow_validations'],
                'validations_by_stage': self.workflow_metrics['validations_by_stage'],
                'avg_validation_time_by_stage': self.workflow_metrics['avg_validation_time_by_stage'],
                'blocked_executions': self.workflow_metrics['blocked_executions'],
                'auto_fixes_applied': self.workflow_metrics['auto_fixes_applied'],
                'active_validations': len(self.active_validations),
                'api_interceptors': len(self.api_interceptors)
            },
            'configuration': {
                'enabled_stages': [stage.value for stage in self.config.enabled_stages],
                'intercept_api_calls': self.config.intercept_api_calls,
                'validate_streaming': self.config.validate_streaming,
                'auto_fix_on_validation_fail': self.config.auto_fix_on_validation_fail,
                'max_validation_time_ms': self.config.max_validation_time_ms,
                'batch_validation': self.config.batch_validation
            }
        }
        
        return {
            **real_time_metrics,
            **integration_metrics,
            **workflow_metrics
        }
    
    def add_workflow_hook(self, stage: WorkflowStage, hook: Callable) -> None:
        """Add hook for specific workflow stage."""
        self.workflow_hooks[stage].append(hook)
    
    def add_api_interceptor(self, api_name: str, interceptor: Callable) -> None:
        """Add API call interceptor."""
        self.api_interceptors[api_name] = interceptor
    
    async def cleanup(self) -> None:
        """Cleanup workflow integration manager."""
        logger.info("Cleaning up Workflow Integration Manager")
        
        # Cleanup real-time validator
        await self.real_time_validator.cleanup()
        
        # Clear active validations
        self.active_validations.clear()
        
        # Clear hooks and interceptors
        for stage in self.workflow_hooks:
            self.workflow_hooks[stage].clear()
        self.api_interceptors.clear()
        
        logger.info("Workflow Integration Manager cleanup completed")
    
    # Private implementation methods
    
    async def _load_config(self) -> None:
        """Load workflow integration configuration."""
        config = await self.config_manager.get_setting('workflow_integration', {})
        
        enabled_stages = config.get('enabled_stages', [
            'pre_execution', 'post_generation', 'task_completion'
        ])
        
        self.config = WorkflowValidationConfig(
            enabled_stages=[WorkflowStage(stage) for stage in enabled_stages],
            intercept_api_calls=config.get('intercept_api_calls', True),
            validate_streaming=config.get('validate_streaming', True),
            auto_fix_on_validation_fail=config.get('auto_fix_on_validation_fail', True),
            block_invalid_commits=config.get('block_invalid_commits', True),
            performance_monitoring=config.get('performance_monitoring', True),
            max_validation_time_ms=config.get('max_validation_time_ms', 200),
            batch_validation=config.get('batch_validation', True),
            cache_validation_results=config.get('cache_validation_results', True)
        )
    
    async def _setup_api_interceptors(self) -> None:
        """Setup API call interceptors."""
        if self.config.intercept_api_calls:
            # Setup Claude Code interceptor
            self.add_api_interceptor('claude_code', self.intercept_claude_code_call)
            logger.info("API interceptors configured")
    
    async def _setup_workflow_hooks(self) -> None:
        """Setup workflow stage hooks."""
        # Default hooks for monitoring and logging
        for stage in WorkflowStage:
            if stage in self.config.enabled_stages:
                # Add logging hook
                async def log_hook(stage_name=stage.value, **kwargs):
                    logger.debug(f"Workflow hook triggered: {stage_name}")
                
                self.add_workflow_hook(stage, log_hook)
        
        logger.info("Workflow hooks configured")
    
    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring for workflows."""
        if self.config.performance_monitoring:
            self.workflow_metrics = {
                'total_workflow_validations': 0,
                'validations_by_stage': {stage.value: 0 for stage in WorkflowStage},
                'avg_validation_time_by_stage': {stage.value: 0.0 for stage in WorkflowStage},
                'blocked_executions': 0,
                'auto_fixes_applied': 0
            }
            logger.info("Performance monitoring initialized")
    
    def _update_workflow_metrics(self, result: WorkflowValidationResult) -> None:
        """Update workflow performance metrics."""
        stage_name = result.workflow_stage.value
        
        self.workflow_metrics['total_workflow_validations'] += 1
        self.workflow_metrics['validations_by_stage'][stage_name] += 1
        self.workflow_metrics['auto_fixes_applied'] += result.auto_fixes_applied
        
        # Update average validation time for stage
        current_count = self.workflow_metrics['validations_by_stage'][stage_name]
        current_avg = self.workflow_metrics['avg_validation_time_by_stage'][stage_name]
        
        new_avg = ((current_avg * (current_count - 1)) + result.processing_time_ms) / current_count
        self.workflow_metrics['avg_validation_time_by_stage'][stage_name] = new_avg
    
    async def _run_workflow_hooks(self, stage: WorkflowStage, context: Dict[str, Any]) -> None:
        """Run hooks for specific workflow stage."""
        hooks = self.workflow_hooks.get(stage, [])
        
        for hook in hooks:
            try:
                await hook(**context)
            except Exception as e:
                logger.warning(f"Workflow hook failed for stage {stage.value}: {e}")
    
    async def _validate_file_change(
        self,
        project: Project,
        file_path: Path,
        change_context: Dict[str, Any]
    ) -> WorkflowValidationResult:
        """Validate a single file change."""
        validation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Read file content
            with open(project.path / file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Validate file content
            validation_context = {
                **change_context,
                'workflow_stage': 'project_update',
                'file_path': str(file_path),
                'project_name': project.name
            }
            
            file_validation = await self.real_time_validator.validate_live(
                content,
                validation_context,
                ValidationMode.PRE_COMMIT
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return WorkflowValidationResult(
                validation_id=validation_id,
                workflow_stage=WorkflowStage.PROJECT_UPDATE,
                is_valid=file_validation.is_valid,
                authenticity_score=file_validation.authenticity_score,
                processing_time_ms=processing_time,
                issues_detected=len(file_validation.issues_detected),
                auto_fixes_applied=0,  # Would be handled separately
                validation_data={
                    'file_path': str(file_path),
                    'file_size': len(content),
                    'file_validation': file_validation.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to validate file change {file_path}: {e}")
            
            return WorkflowValidationResult(
                validation_id=validation_id,
                workflow_stage=WorkflowStage.PROJECT_UPDATE,
                is_valid=False,
                authenticity_score=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                issues_detected=1,
                auto_fixes_applied=0,
                validation_data={'error': str(e), 'file_path': str(file_path)}
            )