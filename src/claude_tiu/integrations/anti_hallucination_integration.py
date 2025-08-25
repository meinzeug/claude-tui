"""
Anti-Hallucination Integration Module - Seamless integration with Claude-TIU.

Provides seamless integration between the Anti-Hallucination Engine
and the main Claude-TIU system:
- AI Interface integration
- Real-time validation hooks
- Task result validation
- Project-wide validation
- Performance monitoring
- Configuration management
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.models.project import Project
from claude_tiu.models.task import DevelopmentTask, TaskResult
from claude_tiu.validation.anti_hallucination_engine import (
    AntiHallucinationEngine, 
    CodeSample,
    ValidationPipelineResult
)
from claude_tiu.validation.performance_optimizer import PerformanceOptimizer
from claude_tiu.validation.training_data_generator import TrainingDataGenerator
from claude_tiu.validation.progress_validator import ValidationResult, ValidationSeverity
from claude_tiu.integrations.ai_interface import AIInterface

logger = logging.getLogger(__name__)


@dataclass
class ValidationHookConfig:
    """Configuration for validation hooks."""
    enabled: bool = True
    auto_fix: bool = True
    block_on_critical: bool = True
    async_validation: bool = True
    cache_results: bool = True
    performance_monitoring: bool = True


@dataclass
class IntegrationMetrics:
    """Integration performance and usage metrics."""
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    auto_fixes_applied: int = 0
    critical_issues_blocked: int = 0
    avg_validation_time: float = 0.0
    cache_hit_rate: float = 0.0
    model_accuracy: float = 0.0
    last_updated: datetime = None


class AntiHallucinationIntegration:
    """
    Core integration module for Anti-Hallucination Engine.
    
    Seamlessly integrates advanced anti-hallucination validation
    into the Claude-TIU system workflow.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the integration module."""
        self.config_manager = config_manager
        
        # Core components
        self.engine: Optional[AntiHallucinationEngine] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.training_generator: Optional[TrainingDataGenerator] = None
        
        # Integration state
        self.is_initialized = False
        self.hook_config = ValidationHookConfig()
        self.metrics = IntegrationMetrics()
        
        # Validation hooks
        self.pre_task_hooks: List[callable] = []
        self.post_task_hooks: List[callable] = []
        self.code_generation_hooks: List[callable] = []
        
        # Real-time validation queue
        self.validation_queue: asyncio.Queue = asyncio.Queue()
        self.validation_workers: List[asyncio.Task] = []
        
        logger.info("Anti-Hallucination Integration initialized")
    
    async def initialize(self) -> None:
        """Initialize the integration system."""
        if self.is_initialized:
            return
        
        logger.info("Initializing Anti-Hallucination Integration")
        
        try:
            # Load configuration
            await self._load_integration_config()
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Setup validation hooks
            await self._setup_validation_hooks()
            
            # Start background workers
            await self._start_validation_workers()
            
            # Initialize model training
            await self._initialize_model_training()
            
            self.is_initialized = True
            logger.info("Anti-Hallucination Integration ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anti-Hallucination Integration: {e}")
            raise
    
    async def validate_ai_generated_content(
        self,
        content: str,
        context: Dict[str, Any] = None,
        task: Optional[DevelopmentTask] = None,
        project: Optional[Project] = None
    ) -> ValidationResult:
        """
        Main entry point for validating AI-generated content.
        
        Args:
            content: Content to validate
            context: Additional context
            task: Associated development task
            project: Associated project
            
        Returns:
            ValidationResult with comprehensive analysis
        """
        if not self.is_initialized:
            await self.initialize()
        
        context = context or {}
        context.update({
            'task': task,
            'project': project,
            'timestamp': datetime.now().isoformat(),
            'integration_version': '1.0.0'
        })
        
        try:
            # Pre-validation hooks
            await self._run_pre_validation_hooks(content, context)
            
            # Core validation
            if self.hook_config.async_validation:
                validation_result = await self._async_validate_content(content, context)
            else:
                validation_result = await self.engine.validate_code_authenticity(content, context)
            
            # Post-validation processing
            validation_result = await self._process_validation_result(
                validation_result, content, context, task, project
            )
            
            # Post-validation hooks
            await self._run_post_validation_hooks(validation_result, content, context)
            
            # Update metrics
            self._update_integration_metrics(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            
            # Return error result
            error_result = ValidationResult(
                is_valid=False,
                overall_score=0.0,
                authenticity_score=0.0,
                completeness_score=0.0,
                quality_score=0.0,
                issues=[],
                summary=f"Validation failed: {e}",
                execution_time=0.0,
                validated_at=datetime.now()
            )
            
            self.metrics.failed_validations += 1
            return error_result
    
    async def validate_task_result(
        self,
        task: DevelopmentTask,
        task_result: TaskResult,
        project: Project
    ) -> ValidationResult:
        """
        Validate a completed task result.
        
        Args:
            task: Original development task
            task_result: Task execution result
            project: Associated project
            
        Returns:
            ValidationResult for the task
        """
        logger.info(f"Validating task result: {task.name}")
        
        context = {
            'task_type': task.task_type,
            'task_priority': getattr(task, 'priority', 'medium'),
            'project_name': project.name if project else None,
            'validation_type': 'task_result'
        }
        
        # Validate generated content
        if task_result.generated_content:
            content_validation = await self.validate_ai_generated_content(
                task_result.generated_content,
                context=context,
                task=task,
                project=project
            )
            
            # Check if critical issues should block task completion
            if self.hook_config.block_on_critical:
                critical_issues = [
                    issue for issue in content_validation.issues
                    if issue.severity == ValidationSeverity.CRITICAL
                ]
                
                if critical_issues:
                    logger.warning(f"Task {task.name} blocked due to {len(critical_issues)} critical issues")
                    self.metrics.critical_issues_blocked += 1
                    content_validation.is_valid = False
            
            return content_validation
        
        # No content to validate
        return ValidationResult(
            is_valid=True,
            overall_score=1.0,
            authenticity_score=1.0,
            completeness_score=1.0,
            quality_score=1.0,
            issues=[],
            summary="No content to validate",
            execution_time=0.0,
            validated_at=datetime.now()
        )
    
    async def validate_project_codebase(
        self,
        project: Project,
        incremental: bool = True
    ) -> Dict[str, ValidationResult]:
        """
        Validate entire project codebase.
        
        Args:
            project: Project to validate
            incremental: Only validate changed files
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        logger.info(f"Validating project codebase: {project.name}")
        
        results = {}
        
        try:
            # Get files to validate
            files_to_validate = await self._get_project_files(project, incremental)
            
            logger.info(f"Validating {len(files_to_validate)} files")
            
            # Batch validation for performance
            batch_size = 10
            for i in range(0, len(files_to_validate), batch_size):
                batch = files_to_validate[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = []
                for file_path in batch:
                    task = self._validate_project_file(file_path, project)
                    batch_tasks.append(task)
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect results
                for file_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to validate {file_path}: {result}")
                        # Create error result
                        results[str(file_path)] = ValidationResult(
                            is_valid=False,
                            overall_score=0.0,
                            authenticity_score=0.0,
                            completeness_score=0.0,
                            quality_score=0.0,
                            issues=[],
                            summary=f"Validation error: {result}",
                            execution_time=0.0,
                            validated_at=datetime.now()
                        )
                    else:
                        results[str(file_path)] = result
            
            # Generate project-wide summary
            await self._generate_project_validation_summary(project, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Project validation failed: {e}")
            return {}
    
    async def auto_fix_issues(
        self,
        validation_result: ValidationResult,
        content: str,
        project: Optional[Project] = None
    ) -> Tuple[bool, str]:
        """
        Automatically fix validation issues.
        
        Args:
            validation_result: Validation result with issues
            content: Original content
            project: Associated project
            
        Returns:
            Tuple of (success, fixed_content)
        """
        if not self.hook_config.auto_fix or not validation_result.issues:
            return False, content
        
        logger.info(f"Attempting to auto-fix {len(validation_result.issues)} issues")
        
        try:
            # Filter auto-fixable issues
            fixable_issues = [
                issue for issue in validation_result.issues
                if issue.auto_fixable
            ]
            
            if not fixable_issues:
                return False, content
            
            # Apply fixes using the engine's auto-completion system
            fixed_content = content
            fixes_applied = 0
            
            for issue in fixable_issues:
                fix_result = await self.engine.auto_completion_engine.fix_issue(
                    issue, fixed_content, project
                )
                
                if fix_result and fix_result != fixed_content:
                    fixed_content = fix_result
                    fixes_applied += 1
            
            if fixes_applied > 0:
                logger.info(f"Applied {fixes_applied} auto-fixes")
                self.metrics.auto_fixes_applied += fixes_applied
                return True, fixed_content
            
            return False, content
            
        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
            return False, content
    
    async def register_ai_interface_hooks(self, ai_interface: AIInterface) -> None:
        """
        Register validation hooks with AI interface.
        
        Args:
            ai_interface: AI interface to hook into
        """
        logger.info("Registering AI interface hooks")
        
        # Hook into code generation
        if hasattr(ai_interface, 'add_generation_hook'):
            ai_interface.add_generation_hook(self._on_code_generated)
        
        # Hook into task completion
        if hasattr(ai_interface, 'add_completion_hook'):
            ai_interface.add_completion_hook(self._on_task_completed)
        
        # Hook into error handling
        if hasattr(ai_interface, 'add_error_hook'):
            ai_interface.add_error_hook(self._on_ai_error)
    
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics."""
        engine_metrics = {}
        if self.engine:
            engine_metrics = await self.engine.get_performance_metrics()
        
        optimizer_metrics = {}
        if self.performance_optimizer:
            optimizer_metrics = self.performance_optimizer.get_performance_metrics()
        
        return {
            'integration_metrics': {
                'total_validations': self.metrics.total_validations,
                'successful_validations': self.metrics.successful_validations,
                'failed_validations': self.metrics.failed_validations,
                'success_rate': (
                    self.metrics.successful_validations / max(self.metrics.total_validations, 1)
                ),
                'auto_fixes_applied': self.metrics.auto_fixes_applied,
                'critical_issues_blocked': self.metrics.critical_issues_blocked,
                'avg_validation_time': self.metrics.avg_validation_time,
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'model_accuracy': self.metrics.model_accuracy,
                'last_updated': self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
            },
            'engine_metrics': engine_metrics,
            'optimizer_metrics': optimizer_metrics,
            'hook_config': {
                'enabled': self.hook_config.enabled,
                'auto_fix': self.hook_config.auto_fix,
                'block_on_critical': self.hook_config.block_on_critical,
                'async_validation': self.hook_config.async_validation,
                'cache_results': self.hook_config.cache_results,
                'performance_monitoring': self.hook_config.performance_monitoring
            }
        }
    
    async def retrain_models(
        self,
        additional_samples: List[CodeSample] = None,
        full_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Retrain anti-hallucination models.
        
        Args:
            additional_samples: New training samples to include
            full_retrain: Whether to do full retraining or incremental
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Retraining models (full_retrain={full_retrain})")
        
        try:
            # Generate or collect training data
            if full_retrain or not additional_samples:
                training_data = await self.training_generator.generate_training_dataset()
            else:
                training_data = additional_samples
            
            # Train pattern recognition model
            pattern_metrics = await self.engine.train_pattern_recognition_model(training_data)
            
            # Update metrics
            self.metrics.model_accuracy = pattern_metrics.accuracy
            
            results = {
                'success': True,
                'training_samples': len(training_data),
                'model_metrics': {
                    'accuracy': pattern_metrics.accuracy,
                    'precision': pattern_metrics.precision,
                    'recall': pattern_metrics.recall,
                    'f1_score': pattern_metrics.f1_score,
                    'cross_validation_mean': np.mean(pattern_metrics.cross_validation_scores),
                    'training_time': (
                        datetime.now() - pattern_metrics.last_trained
                    ).total_seconds()
                }
            }
            
            logger.info(f"Model retraining completed. Accuracy: {pattern_metrics.accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup integration resources."""
        logger.info("Cleaning up Anti-Hallucination Integration")
        
        # Stop validation workers
        for worker in self.validation_workers:
            worker.cancel()
        
        await asyncio.gather(*self.validation_workers, return_exceptions=True)
        self.validation_workers.clear()
        
        # Cleanup core components
        if self.engine:
            await self.engine.cleanup()
        
        if self.performance_optimizer:
            await self.performance_optimizer.cleanup()
        
        self.is_initialized = False
        logger.info("Anti-Hallucination Integration cleanup completed")
    
    # Private implementation methods
    
    async def _load_integration_config(self) -> None:
        """Load integration configuration."""
        config = await self.config_manager.get_setting('anti_hallucination_integration', {})
        
        self.hook_config = ValidationHookConfig(
            enabled=config.get('enabled', True),
            auto_fix=config.get('auto_fix', True),
            block_on_critical=config.get('block_on_critical', True),
            async_validation=config.get('async_validation', True),
            cache_results=config.get('cache_results', True),
            performance_monitoring=config.get('performance_monitoring', True)
        )
    
    async def _initialize_core_components(self) -> None:
        """Initialize core anti-hallucination components."""
        # Initialize Anti-Hallucination Engine
        self.engine = AntiHallucinationEngine(self.config_manager)
        await self.engine.initialize()
        
        # Initialize Performance Optimizer
        self.performance_optimizer = PerformanceOptimizer()
        await self.performance_optimizer.initialize()
        
        # Initialize Training Data Generator
        self.training_generator = TrainingDataGenerator()
        await self.training_generator.initialize()
    
    async def _setup_validation_hooks(self) -> None:
        """Setup validation hooks for different stages."""
        # Pre-task hooks
        self.pre_task_hooks.append(self._pre_validation_performance_check)
        self.pre_task_hooks.append(self._pre_validation_cache_check)
        
        # Post-task hooks
        self.post_task_hooks.append(self._post_validation_metrics_update)
        self.post_task_hooks.append(self._post_validation_cache_store)
        
        # Code generation hooks
        self.code_generation_hooks.append(self._on_code_generated)
    
    async def _start_validation_workers(self) -> None:
        """Start background validation workers."""
        if not self.hook_config.async_validation:
            return
        
        # Start worker tasks
        worker_count = 3
        for i in range(worker_count):
            worker = asyncio.create_task(self._validation_worker(f"worker_{i}"))
            self.validation_workers.append(worker)
    
    async def _initialize_model_training(self) -> None:
        """Initialize model training with basic data."""
        logger.info("Initializing model training")
        
        # Generate initial training data
        initial_data = await self.training_generator.generate_training_dataset()
        
        # Train initial models
        pattern_metrics = await self.engine.train_pattern_recognition_model(initial_data)
        
        # Update metrics
        self.metrics.model_accuracy = pattern_metrics.accuracy
        
        logger.info(f"Initial model training completed. Accuracy: {pattern_metrics.accuracy:.4f}")
    
    async def _async_validate_content(self, content: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate content asynchronously using queue."""
        # Create future for result
        result_future = asyncio.Future()
        
        # Add to validation queue
        await self.validation_queue.put({
            'content': content,
            'context': context,
            'result_future': result_future
        })
        
        # Wait for result
        return await result_future
    
    async def _validation_worker(self, worker_name: str) -> None:
        """Background worker for async validation."""
        logger.debug(f"Starting validation worker: {worker_name}")
        
        while True:
            try:
                # Get validation request
                request = await self.validation_queue.get()
                
                # Process validation
                result = await self.engine.validate_code_authenticity(
                    request['content'],
                    request['context']
                )
                
                # Set result
                request['result_future'].set_result(result)
                
                # Mark task done
                self.validation_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Validation worker {worker_name} error: {e}")
                if 'result_future' in locals() and not request['result_future'].done():
                    request['result_future'].set_exception(e)
    
    async def _process_validation_result(
        self,
        result: ValidationResult,
        content: str,
        context: Dict[str, Any],
        task: Optional[DevelopmentTask],
        project: Optional[Project]
    ) -> ValidationResult:
        """Process validation result with additional context."""
        # Apply auto-fixes if enabled and needed
        if self.hook_config.auto_fix and not result.is_valid:
            fix_applied, fixed_content = await self.auto_fix_issues(result, content, project)
            
            if fix_applied:
                # Re-validate fixed content
                fixed_result = await self.engine.validate_code_authenticity(fixed_content, context)
                
                # Update summary
                fixed_result.summary += f" (auto-fixes applied)"
                
                return fixed_result
        
        return result
    
    async def _run_pre_validation_hooks(self, content: str, context: Dict[str, Any]) -> None:
        """Run pre-validation hooks."""
        for hook in self.pre_task_hooks:
            try:
                await hook(content, context)
            except Exception as e:
                logger.warning(f"Pre-validation hook failed: {e}")
    
    async def _run_post_validation_hooks(
        self,
        result: ValidationResult,
        content: str,
        context: Dict[str, Any]
    ) -> None:
        """Run post-validation hooks."""
        for hook in self.post_task_hooks:
            try:
                await hook(result, content, context)
            except Exception as e:
                logger.warning(f"Post-validation hook failed: {e}")
    
    async def _pre_validation_performance_check(self, content: str, context: Dict[str, Any]) -> None:
        """Pre-validation performance check."""
        if self.hook_config.performance_monitoring and self.performance_optimizer:
            # Could implement performance pre-checks here
            pass
    
    async def _pre_validation_cache_check(self, content: str, context: Dict[str, Any]) -> None:
        """Pre-validation cache check."""
        if self.hook_config.cache_results:
            # Could implement cache warming here
            pass
    
    async def _post_validation_metrics_update(
        self,
        result: ValidationResult,
        content: str,
        context: Dict[str, Any]
    ) -> None:
        """Update metrics after validation."""
        self.metrics.total_validations += 1
        
        if result.is_valid:
            self.metrics.successful_validations += 1
        else:
            self.metrics.failed_validations += 1
        
        # Update average validation time
        if self.metrics.total_validations > 1:
            self.metrics.avg_validation_time = (
                (self.metrics.avg_validation_time * (self.metrics.total_validations - 1) +
                 result.execution_time) / self.metrics.total_validations
            )
        else:
            self.metrics.avg_validation_time = result.execution_time
        
        self.metrics.last_updated = datetime.now()
    
    async def _post_validation_cache_store(
        self,
        result: ValidationResult,
        content: str,
        context: Dict[str, Any]
    ) -> None:
        """Store validation result in cache."""
        if self.hook_config.cache_results:
            # Cache storage is handled by the engine
            pass
    
    async def _get_project_files(self, project: Project, incremental: bool) -> List[Path]:
        """Get list of files to validate in project."""
        project_path = Path(project.path)
        
        if not project_path.exists():
            return []
        
        # File extensions to validate
        valid_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
        
        files = []
        for ext in valid_extensions:
            files.extend(project_path.rglob(f'*{ext}'))
        
        # Filter out common directories to skip
        skip_dirs = {'__pycache__', 'node_modules', '.git', '.venv', 'venv'}
        filtered_files = []
        
        for file_path in files:
            if not any(part in skip_dirs for part in file_path.parts):
                filtered_files.append(file_path)
        
        return filtered_files
    
    async def _validate_project_file(self, file_path: Path, project: Project) -> ValidationResult:
        """Validate a single project file."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create context
            context = {
                'file_path': str(file_path),
                'project': project,
                'validation_type': 'project_file'
            }
            
            # Validate content
            return await self.validate_ai_generated_content(content, context, project=project)
            
        except Exception as e:
            logger.error(f"Failed to validate {file_path}: {e}")
            
            return ValidationResult(
                is_valid=False,
                overall_score=0.0,
                authenticity_score=0.0,
                completeness_score=0.0,
                quality_score=0.0,
                issues=[],
                summary=f"File validation error: {e}",
                execution_time=0.0,
                validated_at=datetime.now()
            )
    
    async def _generate_project_validation_summary(
        self,
        project: Project,
        results: Dict[str, ValidationResult]
    ) -> None:
        """Generate project-wide validation summary."""
        if not results:
            return
        
        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.is_valid)
        invalid_files = total_files - valid_files
        
        avg_authenticity = sum(r.authenticity_score for r in results.values()) / total_files
        avg_quality = sum(r.quality_score for r in results.values()) / total_files
        
        total_issues = sum(len(r.issues) for r in results.values())
        
        summary = {
            'project_name': project.name,
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'avg_authenticity_score': avg_authenticity,
            'avg_quality_score': avg_quality,
            'total_issues': total_issues,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Save summary to project directory
        summary_path = Path(project.path) / '.claude-tiu' / 'validation_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(
            f"Project validation summary: {valid_files}/{total_files} valid files, "
            f"avg authenticity: {avg_authenticity:.3f}, total issues: {total_issues}"
        )
    
    def _update_integration_metrics(self, result: ValidationResult) -> None:
        """Update integration-specific metrics."""
        # This is called by _post_validation_metrics_update
        # Additional integration-specific metrics can be updated here
        pass
    
    async def _on_code_generated(self, code: str, context: Dict[str, Any]) -> None:
        """Hook called when AI generates code."""
        if self.hook_config.enabled:
            # Queue for validation
            asyncio.create_task(self.validate_ai_generated_content(code, context))
    
    async def _on_task_completed(
        self,
        task: DevelopmentTask,
        result: TaskResult,
        project: Project
    ) -> None:
        """Hook called when AI task is completed."""
        if self.hook_config.enabled:
            # Validate task result
            validation_result = await self.validate_task_result(task, result, project)
            
            # Store validation result in task context
            if hasattr(result, 'validation_result'):
                result.validation_result = validation_result
    
    async def _on_ai_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Hook called when AI encounters an error."""
        # Log error for potential model improvement
        logger.warning(f"AI error for model improvement tracking: {error}")
        
        # Could collect error patterns for training data generation
        if hasattr(error, 'generated_content'):
            # Mark as potentially hallucinated content
            pass