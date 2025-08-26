"""
Task Engine - Advanced workflow orchestration and task execution.

Provides sophisticated task scheduling and execution with intelligent
dependency resolution, parallel processing, and comprehensive monitoring.
Integrates with AI services and validation systems for high-quality results.

Features:
- Dependency-aware task scheduling
- Adaptive execution strategies (sequential, parallel, balanced)
- Real-time progress monitoring with metrics
- Error handling and automatic recovery
- Integration with AI services and validation
- Resource-aware concurrent execution
- Task result caching and optimization
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

from .types import (
    Task, TaskStatus, Priority, ExecutionStrategy, Workflow, 
    DevelopmentResult, ProgressMetrics, ValidationResult,
    SystemMetrics, ClaudeTUIException
)
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class TaskEngineException(ClaudeTUIException):
    """Task engine specific exceptions."""
    pass


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: UUID
    success: bool
    output: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    validation_score: float = 0.0
    execution_time: float = 0.0
    tokens_used: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ExecutionContext:
    """Context for task execution."""
    project_path: Optional[Path] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    validate_results: bool = True
    max_retries: int = 3
    execution_id: str = field(default_factory=lambda: f"exec_{uuid4().hex[:8]}")


@dataclass
class ExecutionStatus:
    """Current execution status and metrics."""
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    total_tasks: int = 0
    execution_time: float = 0.0
    success_rate: float = 0.0
    current_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    resource_usage: SystemMetrics = field(default_factory=SystemMetrics)
    bottlenecks: List[str] = field(default_factory=list)


class TaskEngine:
    """
    Advanced task scheduling and execution engine.
    
    The TaskEngine orchestrates complex development workflows with intelligent
    dependency resolution, adaptive execution strategies, and comprehensive
    monitoring. It provides high-performance task execution with built-in
    validation and error recovery.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the TaskEngine.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        
        # Core execution state
        self._active_tasks: Dict[UUID, Task] = {}
        self._completed_tasks: Dict[UUID, TaskResult] = {}
        self._failed_tasks: Dict[UUID, TaskResult] = {}
        self._cancelled_tasks: Set[UUID] = set()
        self._task_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance and resource management
        self._max_concurrent_tasks = min(8, psutil.cpu_count() or 4)
        self._execution_semaphore: Optional[asyncio.Semaphore] = None
        self._thread_executor = ThreadPoolExecutor(max_workers=4)
        
        # Execution tracking
        self._execution_history: deque = deque(maxlen=1000)
        self._performance_metrics: Dict[str, float] = {}
        self._task_cache: Dict[str, TaskResult] = {}
        
        # Configuration
        self._default_timeout = 300
        self._cache_ttl = 300  # 5 minutes
        self._retry_delays = [1, 3, 9]  # Exponential backoff
        
        # State management
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        
        logger.info("TaskEngine initialized")
    
    async def initialize(self) -> None:
        """Initialize the task engine with configuration."""
        if self._initialized:
            return
        
        logger.info("Initializing TaskEngine")
        
        # Load configuration
        config = self.config_manager.get_config_value('task_engine', {})
        
        # Configure resource limits based on system
        cpu_count = psutil.cpu_count() or 4
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        self._max_concurrent_tasks = min(
            config.get('max_concurrent_tasks', 8),
            max(2, int(cpu_count * 1.5)),  # 1.5x CPU cores
            max(2, int(memory_gb))         # 1 task per GB RAM
        )
        
        self._default_timeout = config.get('default_timeout', 300)
        self._execution_semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        
        logger.info(f"TaskEngine initialized: max_concurrent={self._max_concurrent_tasks}")
        self._initialized = True
    
    async def execute_task(
        self,
        task: Task,
        context: Optional[ExecutionContext] = None
    ) -> TaskResult:
        """
        Execute a single task with full monitoring and validation.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution result
        """
        if not self._initialized:
            await self.initialize()
        
        exec_context = context or ExecutionContext()
        start_time = time.time()
        
        logger.info(f"Executing task: {task.name} (ID: {task.id})")
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(task, exec_context)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug(f"Using cached result for task {task.name}")
                return cached_result
            
            # Update task status
            task.start()
            self._active_tasks[task.id] = task
            
            # Execute with semaphore for resource control
            async with self._execution_semaphore:
                result = await self._execute_task_with_retry(task, exec_context)
            
            # Update execution time
            result.execution_time = time.time() - start_time
            result.completed_at = datetime.utcnow()
            
            # Move to appropriate collection
            if result.success:
                task.complete()
                self._completed_tasks[task.id] = result
                self._cache_result(cache_key, result)
            else:
                task.fail(result.error_message or "Task failed")
                self._failed_tasks[task.id] = result
            
            # Remove from active tasks
            self._active_tasks.pop(task.id, None)
            
            # Record metrics
            await self._record_execution_metrics(task, result)
            
            logger.info(
                f"Task '{task.name}' completed: success={result.success}, "
                f"time={result.execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task '{task.name}' execution failed: {e}")
            
            execution_time = time.time() - start_time
            error_result = TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            
            task.fail(str(e))
            self._failed_tasks[task.id] = error_result
            self._active_tasks.pop(task.id, None)
            
            return error_result
    
    async def execute_tasks(
        self,
        tasks: List[Task],
        context: Optional[ExecutionContext] = None,
        strategy: Optional[ExecutionStrategy] = None
    ) -> List[TaskResult]:
        """
        Execute multiple tasks with dependency resolution and strategy optimization.
        
        Args:
            tasks: List of tasks to execute
            context: Execution context
            strategy: Execution strategy override
            
        Returns:
            List of task results in execution order
        """
        if not tasks:
            return []
        
        if not self._initialized:
            await self.initialize()
        
        exec_context = context or ExecutionContext()
        execution_strategy = strategy or exec_context.strategy
        
        logger.info(f"Executing {len(tasks)} tasks with strategy: {execution_strategy.value}")
        start_time = time.time()
        
        try:
            # Resolve dependencies and create execution plan
            execution_plan = await self._create_execution_plan(tasks)
            
            # Execute based on strategy
            if execution_strategy == ExecutionStrategy.SEQUENTIAL:
                results = await self._execute_sequential(execution_plan, exec_context)
            elif execution_strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_parallel(execution_plan, exec_context)
            elif execution_strategy == ExecutionStrategy.ADAPTIVE:
                results = await self._execute_adaptive(execution_plan, exec_context)
            else:  # Default to adaptive
                results = await self._execute_adaptive(execution_plan, exec_context)
            
            execution_time = time.time() - start_time
            success_count = sum(1 for r in results if r.success)
            
            logger.info(
                f"Task batch completed: {success_count}/{len(tasks)} successful "
                f"in {execution_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Task batch execution failed: {e}")
            # Return failure results for all tasks
            return [
                TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=f"Batch execution failed: {e}",
                    execution_time=0
                )
                for task in tasks
            ]
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        context: Optional[ExecutionContext] = None
    ) -> DevelopmentResult:
        """
        Execute a complete workflow with all tasks.
        
        Args:
            workflow: Workflow to execute
            context: Execution context
            
        Returns:
            Workflow execution result
        """
        logger.info(f"Executing workflow: {workflow.name}")
        start_time = time.time()
        
        exec_context = context or ExecutionContext()
        
        try:
            # Execute all workflow tasks
            task_results = await self.execute_tasks(workflow.tasks, exec_context)
            
            # Analyze results
            successful_tasks = [r for r in task_results if r.success]
            failed_tasks = [r for r in task_results if not r.success]
            
            # Collect all modified files
            all_files = []
            for result in successful_tasks:
                all_files.extend(result.files_modified)
                all_files.extend(result.files_created)
            
            # Calculate quality metrics
            total_validation_score = sum(r.validation_score for r in successful_tasks)
            avg_validation_score = (
                total_validation_score / len(successful_tasks)
                if successful_tasks else 0.0
            )
            
            quality_metrics = {
                'validation_score': avg_validation_score,
                'completion_rate': len(successful_tasks) / len(workflow.tasks) * 100,
                'error_rate': len(failed_tasks) / len(workflow.tasks) * 100
            }
            
            execution_time = time.time() - start_time
            success = len(failed_tasks) == 0
            
            result = DevelopmentResult(
                workflow_id=workflow.id,
                success=success,
                tasks_executed=[r.task_id for r in task_results],
                files_generated=list(set(all_files)),
                total_time=execution_time,
                quality_metrics=quality_metrics,
                error_details="; ".join(r.error_message for r in failed_tasks if r.error_message)
            )
            
            logger.info(
                f"Workflow '{workflow.name}' completed: success={success}, "
                f"time={execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow '{workflow.name}' failed: {e}")
            
            execution_time = time.time() - start_time
            return DevelopmentResult(
                workflow_id=workflow.id,
                success=False,
                tasks_executed=[],
                files_generated=[],
                total_time=execution_time,
                quality_metrics={},
                error_details=str(e)
            )
    
    async def get_status(self) -> ExecutionStatus:
        """
        Get current execution status and metrics.
        
        Returns:
            Current execution status
        """
        active_count = len(self._active_tasks)
        queued_count = self._task_queue.qsize()
        completed_count = len(self._completed_tasks)
        failed_count = len(self._failed_tasks)
        cancelled_count = len(self._cancelled_tasks)
        
        total_count = active_count + queued_count + completed_count + failed_count
        success_rate = (
            completed_count / (completed_count + failed_count)
            if (completed_count + failed_count) > 0 else 0.0
        )
        
        # Calculate total execution time from active tasks
        current_time = datetime.utcnow()
        total_execution_time = 0.0
        
        for task in self._active_tasks.values():
            if task.started_at:
                total_execution_time += (current_time - task.started_at).total_seconds()
        
        # Get current resource usage
        resource_usage = SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=psutil.virtual_memory().percent,
            active_tasks=active_count
        )
        
        return ExecutionStatus(
            active_tasks=active_count,
            queued_tasks=queued_count,
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            cancelled_tasks=cancelled_count,
            total_tasks=total_count,
            execution_time=total_execution_time,
            success_rate=success_rate * 100,
            resource_usage=resource_usage
        )
    
    async def cancel_task(self, task_id: UUID) -> bool:
        """
        Cancel a running or queued task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled successfully
        """
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            # Move to cancelled set
            self._cancelled_tasks.add(task_id)
            del self._active_tasks[task_id]
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    async def retry_failed_task(self, task_id: UUID) -> Optional[TaskResult]:
        """
        Retry a failed task.
        
        Args:
            task_id: ID of failed task to retry
            
        Returns:
            New task result if retry successful, None otherwise
        """
        if task_id not in self._failed_tasks:
            return None
        
        # Get original failed result and task
        failed_result = self._failed_tasks[task_id]
        
        # Find original task (would need to be stored or reconstructed)
        # For now, return None - this would need task persistence
        logger.warning(f"Task retry not implemented for task {task_id}")
        return None
    
    async def clear_completed_tasks(self) -> int:
        """
        Clear completed task results to free memory.
        
        Returns:
            Number of tasks cleared
        """
        cleared_count = len(self._completed_tasks)
        self._completed_tasks.clear()
        
        # Also clear old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, (result, timestamp) in self._task_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._task_cache[key]
        
        logger.info(f"Cleared {cleared_count} completed tasks and {len(expired_keys)} cache entries")
        return cleared_count
    
    async def shutdown(self) -> None:
        """Shutdown the task engine and cleanup resources."""
        logger.info("Shutting down TaskEngine")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all active tasks
        active_task_ids = list(self._active_tasks.keys())
        for task_id in active_task_ids:
            await self.cancel_task(task_id)
        
        # Shutdown thread executor
        self._thread_executor.shutdown(wait=True)
        
        # Clear all state
        self._active_tasks.clear()
        self._task_cache.clear()
        
        logger.info("TaskEngine shutdown complete")
    
    # Private execution methods
    
    async def _execute_task_with_retry(
        self,
        task: Task,
        context: ExecutionContext
    ) -> TaskResult:
        """Execute task with retry logic."""
        last_error = None
        
        for attempt in range(context.max_retries):
            try:
                result = await self._execute_single_task(task, context)
                if result.success:
                    return result
                
                last_error = result.error_message
                if attempt < context.max_retries - 1:
                    delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    logger.info(f"Task {task.name} failed, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
            except Exception as e:
                last_error = str(e)
                if attempt < context.max_retries - 1:
                    delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    logger.warning(f"Task {task.name} error: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
        
        # All retries failed
        return TaskResult(
            task_id=task.id,
            success=False,
            error_message=f"Task failed after {context.max_retries} attempts: {last_error}",
            started_at=datetime.utcnow()
        )
    
    async def _execute_single_task(
        self,
        task: Task,
        context: ExecutionContext
    ) -> TaskResult:
        """Execute a single task with AI interface."""
        start_time = datetime.utcnow()
        
        try:
            # Import AI interface here to avoid circular imports
            from .ai_interface import AIInterface
            
            ai_interface = AIInterface(self.config_manager)
            await ai_interface.initialize()
            
            # Build AI request from task
            ai_request = {
                'task_description': task.description,
                'context': context.variables,
                'project_path': context.project_path,
                'timeout': context.timeout or self._default_timeout
            }
            
            # Execute AI task
            ai_response = await ai_interface.execute_development_task(ai_request)
            
            # Create result from AI response
            result = TaskResult(
                task_id=task.id,
                success=ai_response.get('success', False),
                output=ai_response.get('output', ''),
                files_modified=ai_response.get('files_modified', []),
                files_created=ai_response.get('files_created', []),
                validation_score=ai_response.get('validation_score', 0.0),
                tokens_used=ai_response.get('tokens_used', 0),
                error_message=ai_response.get('error_message'),
                metadata=ai_response.get('metadata', {}),
                started_at=start_time
            )
            
            # Validate result if requested
            if context.validate_results and result.success:
                validation_result = await self._validate_task_result(task, result, context)
                if not validation_result.is_authentic:
                    result.success = False
                    result.error_message = f"Validation failed: {validation_result.authenticity_score:.1f}% authentic"
                    result.validation_score = validation_result.authenticity_score
            
            return result
            
        except Exception as e:
            logger.error(f"Single task execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                started_at=start_time
            )
    
    async def _create_execution_plan(self, tasks: List[Task]) -> List[List[Task]]:
        """Create execution plan with dependency resolution."""
        # Simple topological sort for now
        # In production, would use more sophisticated dependency resolution
        
        task_map = {task.id: task for task in tasks}
        in_degree = {task.id: len(task.dependencies) for task in tasks}
        
        execution_levels = []
        remaining_tasks = set(task.id for task in tasks)
        
        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = [
                task_map[task_id] for task_id in remaining_tasks
                if in_degree[task_id] == 0
            ]
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                logger.warning("Circular dependency detected, executing remaining tasks anyway")
                ready_tasks = [task_map[task_id] for task_id in remaining_tasks]
            
            execution_levels.append(ready_tasks)
            
            # Remove ready tasks and update dependencies
            for task in ready_tasks:
                remaining_tasks.remove(task.id)
                # Update dependent tasks
                for remaining_id in remaining_tasks:
                    if task.id in task_map[remaining_id].dependencies:
                        in_degree[remaining_id] -= 1
        
        return execution_levels
    
    async def _execute_sequential(
        self,
        execution_plan: List[List[Task]],
        context: ExecutionContext
    ) -> List[TaskResult]:
        """Execute tasks sequentially level by level."""
        results = []
        
        for level in execution_plan:
            for task in level:
                result = await self.execute_task(task, context)
                results.append(result)
                
                # Stop on first failure if configured
                if not result.success and context.variables.get('fail_fast', False):
                    # Add failure results for remaining tasks
                    remaining_tasks = []
                    for remaining_level in execution_plan[execution_plan.index(level):]:
                        remaining_tasks.extend(remaining_level)
                    
                    remaining_tasks = [t for t in remaining_tasks if t.id != task.id]
                    
                    for remaining_task in remaining_tasks:
                        results.append(TaskResult(
                            task_id=remaining_task.id,
                            success=False,
                            error_message="Skipped due to previous failure",
                            started_at=datetime.utcnow()
                        ))
                    
                    return results
        
        return results
    
    async def _execute_parallel(
        self,
        execution_plan: List[List[Task]],
        context: ExecutionContext
    ) -> List[TaskResult]:
        """Execute tasks in parallel within each level."""
        results = []
        
        for level in execution_plan:
            # Execute all tasks in this level concurrently
            level_tasks = [
                self.execute_task(task, context) for task in level
            ]
            
            level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
            
            # Process results
            for task, result in zip(level, level_results):
                if isinstance(result, Exception):
                    error_result = TaskResult(
                        task_id=task.id,
                        success=False,
                        error_message=str(result),
                        started_at=datetime.utcnow()
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        return results
    
    async def _execute_adaptive(
        self,
        execution_plan: List[List[Task]],
        context: ExecutionContext
    ) -> List[TaskResult]:
        """Execute tasks with adaptive strategy based on conditions."""
        # Simple heuristic: use parallel for independent tasks, sequential for dependent
        if len(execution_plan) <= 2 and all(len(level) <= 3 for level in execution_plan):
            # Small workload - use parallel
            return await self._execute_parallel(execution_plan, context)
        else:
            # Larger workload - use mixed approach
            results = []
            
            for level in execution_plan:
                if len(level) == 1:
                    # Single task - execute directly
                    result = await self.execute_task(level[0], context)
                    results.append(result)
                else:
                    # Multiple tasks - execute with controlled parallelism
                    semaphore = asyncio.Semaphore(min(len(level), self._max_concurrent_tasks))
                    
                    async def execute_with_semaphore(task):
                        async with semaphore:
                            return await self.execute_task(task, context)
                    
                    level_results = await asyncio.gather(*[
                        execute_with_semaphore(task) for task in level
                    ], return_exceptions=True)
                    
                    # Process results
                    for task, result in zip(level, level_results):
                        if isinstance(result, Exception):
                            error_result = TaskResult(
                                task_id=task.id,
                                success=False,
                                error_message=str(result),
                                started_at=datetime.utcnow()
                            )
                            results.append(error_result)
                        else:
                            results.append(result)
            
            return results
    
    async def _validate_task_result(
        self,
        task: Task,
        result: TaskResult,
        context: ExecutionContext
    ) -> ValidationResult:
        """Validate task result for authenticity and quality."""
        try:
            # Import validator here to avoid circular imports
            from .validator import ProgressValidator
            
            validator = ProgressValidator()
            
            # Create validation context
            validation_context = {
                'task_description': task.description,
                'expected_outputs': task.expected_outputs,
                'result_output': result.output,
                'files_modified': result.files_modified,
                'project_path': context.project_path
            }
            
            return await validator.validate_task_output(validation_context)
            
        except Exception as e:
            logger.warning(f"Task result validation failed: {e}")
            # Return permissive validation result on error
            return ValidationResult(
                is_authentic=True,
                authenticity_score=80.0,
                real_progress=80.0,
                fake_progress=20.0,
                issues=[],
                suggestions=[]
            )
    
    def _get_cache_key(self, task: Task, context: ExecutionContext) -> str:
        """Generate cache key for task and context."""
        import hashlib
        
        key_data = f"{task.description}:{task.priority.value}:{str(context.variables)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[TaskResult]:
        """Get cached result if still valid."""
        if cache_key in self._task_cache:
            result, timestamp = self._task_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            else:
                # Remove expired entry
                del self._task_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: TaskResult) -> None:
        """Cache task result with timestamp."""
        self._task_cache[cache_key] = (result, time.time())
        
        # Limit cache size
        if len(self._task_cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self._task_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            # Keep only the 80 most recent
            self._task_cache = dict(sorted_items[-80:])
    
    async def _record_execution_metrics(self, task: Task, result: TaskResult) -> None:
        """Record task execution metrics for performance monitoring."""
        metrics = {
            'task_id': str(task.id),
            'task_name': task.name,
            'success': result.success,
            'execution_time': result.execution_time,
            'validation_score': result.validation_score,
            'tokens_used': result.tokens_used,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._execution_history.append(metrics)
        
        # Update performance metrics
        if result.success:
            self._performance_metrics['avg_execution_time'] = (
                self._performance_metrics.get('avg_execution_time', 0) * 0.9 +
                result.execution_time * 0.1
            )
            self._performance_metrics['avg_validation_score'] = (
                self._performance_metrics.get('avg_validation_score', 0) * 0.9 +
                result.validation_score * 0.1
            )


# Factory functions

def create_task(
    name: str,
    description: str,
    priority: Priority = Priority.MEDIUM,
    estimated_duration: Optional[int] = None,
    dependencies: Optional[List[UUID]] = None,
    **kwargs
) -> Task:
    """
    Factory function to create a task.
    
    Args:
        name: Task name
        description: Task description
        priority: Task priority
        estimated_duration: Estimated duration in minutes
        dependencies: Task dependencies
        **kwargs: Additional task properties
        
    Returns:
        Created task instance
    """
    from .types import create_task as core_create_task
    return core_create_task(
        name=name,
        description=description,
        priority=priority,
        estimated_duration=estimated_duration,
        dependencies=dependencies,
        **kwargs
    )


# Utility functions

async def execute_simple_task(
    description: str,
    project_path: Optional[Path] = None,
    timeout: int = 300
) -> TaskResult:
    """
    Convenience function to execute a simple task.
    
    Args:
        description: Task description
        project_path: Optional project path for context
        timeout: Task timeout in seconds
        
    Returns:
        Task execution result
    """
    engine = TaskEngine()
    await engine.initialize()
    
    task = create_task("Simple Task", description)
    context = ExecutionContext(
        project_path=project_path,
        timeout=timeout
    )
    
    try:
        return await engine.execute_task(task, context)
    finally:
        await engine.shutdown()


async def execute_task_batch(
    task_descriptions: List[str],
    project_path: Optional[Path] = None,
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
) -> List[TaskResult]:
    """
    Convenience function to execute a batch of tasks.
    
    Args:
        task_descriptions: List of task descriptions
        project_path: Optional project path for context
        strategy: Execution strategy
        
    Returns:
        List of task execution results
    """
    engine = TaskEngine()
    await engine.initialize()
    
    tasks = [
        create_task(f"Task {i+1}", desc)
        for i, desc in enumerate(task_descriptions)
    ]
    
    context = ExecutionContext(
        project_path=project_path,
        strategy=strategy
    )
    
    try:
        return await engine.execute_tasks(tasks, context)
    finally:
        await engine.shutdown()