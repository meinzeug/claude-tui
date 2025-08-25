"""
Task Engine - Advanced task scheduling and execution engine.

Provides sophisticated workflow orchestration with:
- Dependency resolution and parallel execution
- Progress tracking and real-time monitoring
- Error handling and automatic recovery
- Integration with AI services and validation systems
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.models.task import (
    DevelopmentTask, TaskResult, TaskStatus, TaskPriority, Workflow, WorkflowResult
)
from claude_tiu.models.project import Project
from claude_tiu.utils.dependency_resolver import DependencyResolver
from claude_tiu.utils.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Task execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    BALANCED = "balanced"


@dataclass
class TaskExecutionContext:
    """Context for task execution."""
    project: Project
    workflow: Optional[Workflow] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = field(default="")
    started_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[int] = None


@dataclass
class ExecutionStatus:
    """Current execution status."""
    active_tasks: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_tasks: int
    execution_time: float
    success_rate: float
    current_strategy: ExecutionStrategy
    resource_usage: Dict[str, float] = field(default_factory=dict)


class TaskEngine:
    """
    Advanced task scheduling and execution engine.
    
    The TaskEngine manages complex development workflows with intelligent
    dependency resolution, parallel execution, and comprehensive monitoring.
    It integrates with AI services and validation systems to ensure
    high-quality results.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the TaskEngine.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        
        # Core components
        self.dependency_resolver = DependencyResolver()
        self.metrics_collector = MetricsCollector()
        
        # Execution state
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._active_tasks: Dict[str, DevelopmentTask] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._failed_tasks: Dict[str, Exception] = {}
        self._task_results: Dict[str, TaskResult] = {}
        
        # Execution control
        self._max_concurrent_tasks = 5
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._execution_strategy = ExecutionStrategy.ADAPTIVE
        self._task_timeout = 300  # 5 minutes default
        
        # Monitoring and metrics
        self._execution_history: deque = deque(maxlen=1000)
        self._performance_metrics: Dict[str, float] = {}
        
        # Background workers
        self._workers: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info("TaskEngine initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the task engine.
        """
        logger.info("Initializing TaskEngine")
        
        # Load configuration
        config = await self.config_manager.get_setting('task_engine', {})
        self._max_concurrent_tasks = config.get('max_concurrent_tasks', 5)
        self._task_timeout = config.get('task_timeout', 300)
        
        strategy_name = config.get('execution_strategy', 'adaptive')
        self._execution_strategy = ExecutionStrategy(strategy_name)
        
        # Initialize semaphore
        self._semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        
        # Start background workers
        await self._start_workers()
        
        logger.info(f"TaskEngine initialized with {self._max_concurrent_tasks} max concurrent tasks")
    
    async def execute_tasks(
        self,
        tasks: List[DevelopmentTask],
        validate: bool = True,
        strategy: Optional[ExecutionStrategy] = None,
        context: Optional[TaskExecutionContext] = None
    ) -> List[TaskResult]:
        """
        Execute a list of development tasks.
        
        Args:
            tasks: List of tasks to execute
            validate: Whether to validate task results
            strategy: Execution strategy override
            context: Execution context
            
        Returns:
            List of task results
        """
        if not tasks:
            return []
        
        execution_strategy = strategy or self._execution_strategy
        exec_context = context or TaskExecutionContext(
            project=tasks[0].project if tasks else None,
            execution_id=f"exec_{datetime.now().timestamp()}"
        )
        
        logger.info(f"Executing {len(tasks)} tasks with strategy: {execution_strategy.value}")
        
        start_time = datetime.now()
        
        try:
            # Resolve dependencies
            ordered_tasks = await self.dependency_resolver.resolve_dependencies(tasks)
            
            # Execute based on strategy
            if execution_strategy == ExecutionStrategy.SEQUENTIAL:
                results = await self._execute_sequential(ordered_tasks, validate, exec_context)
            elif execution_strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_parallel(ordered_tasks, validate, exec_context)
            elif execution_strategy == ExecutionStrategy.ADAPTIVE:
                results = await self._execute_adaptive(ordered_tasks, validate, exec_context)
            else:  # BALANCED
                results = await self._execute_balanced(ordered_tasks, validate, exec_context)
            
            # Record metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            success_count = sum(1 for r in results if r.success)
            
            await self.metrics_collector.record_execution(
                task_count=len(tasks),
                success_count=success_count,
                execution_time=execution_time,
                strategy=execution_strategy
            )
            
            logger.info(
                f"Task execution completed: {success_count}/{len(tasks)} successful "
                f"in {execution_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            # Return failure results for all tasks
            return [
                TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=f"Execution failed: {e}",
                    execution_time=0
                )
                for task in tasks
            ]
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        variables: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute a complete workflow.
        
        Args:
            workflow: Workflow to execute
            variables: Workflow variables
            
        Returns:
            Workflow execution result
        """
        logger.info(f"Executing workflow: {workflow.name}")
        
        start_time = datetime.now()
        
        try:
            # Create execution context
            context = TaskExecutionContext(
                project=workflow.project,
                workflow=workflow,
                variables=variables or {},
                execution_id=f"workflow_{workflow.id}_{datetime.now().timestamp()}",
                timeout=workflow.timeout
            )
            
            # Execute workflow tasks
            results = await self.execute_tasks(
                tasks=workflow.tasks,
                validate=workflow.validate_results,
                strategy=workflow.execution_strategy,
                context=context
            )
            
            # Analyze results
            total_tasks = len(workflow.tasks)
            completed_tasks = sum(1 for r in results if r.success)
            failed_tasks = total_tasks - completed_tasks
            
            success = failed_tasks == 0 and workflow.success_criteria.is_met(results)
            duration = (datetime.now() - start_time).total_seconds()
            
            result = WorkflowResult(
                workflow_id=workflow.id,
                success=success,
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                duration=duration,
                task_results=results
            )
            
            if not success:
                result.error_message = f"Workflow failed: {failed_tasks} tasks failed"
            
            logger.info(
                f"Workflow '{workflow.name}' completed: success={success}, "
                f"duration={duration:.2f}s"
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Workflow '{workflow.name}' failed: {e}")
            
            return WorkflowResult(
                workflow_id=workflow.id,
                success=False,
                total_tasks=len(workflow.tasks),
                completed_tasks=0,
                failed_tasks=len(workflow.tasks),
                duration=duration,
                error_message=str(e),
                task_results=[]
            )
    
    async def execute_workflow_from_file(
        self,
        workflow_file: Path,
        variables: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Load and execute a workflow from file.
        
        Args:
            workflow_file: Path to workflow definition file
            variables: Workflow variables
            
        Returns:
            Workflow execution result
        """
        try:
            from claude_tiu.utils.workflow_loader import WorkflowLoader
            
            loader = WorkflowLoader()
            workflow = await loader.load_from_file(workflow_file)
            
            return await self.execute_workflow(workflow, variables)
            
        except Exception as e:
            logger.error(f"Failed to load/execute workflow from {workflow_file}: {e}")
            raise
    
    async def get_status(self) -> ExecutionStatus:
        """
        Get current execution status.
        
        Returns:
            Current execution status
        """
        active_count = len(self._active_tasks)
        queued_count = self._task_queue.qsize()
        completed_count = len(self._completed_tasks)
        failed_count = len(self._failed_tasks)
        total_count = active_count + queued_count + completed_count + failed_count
        
        success_rate = (
            completed_count / (completed_count + failed_count)
            if (completed_count + failed_count) > 0
            else 0.0
        )
        
        # Calculate execution time from active tasks
        current_time = datetime.now()
        execution_time = 0.0
        
        for task in self._active_tasks.values():
            if task.started_at:
                execution_time += (current_time - task.started_at).total_seconds()
        
        return ExecutionStatus(
            active_tasks=active_count,
            queued_tasks=queued_count,
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            total_tasks=total_count,
            execution_time=execution_time,
            success_rate=success_rate,
            current_strategy=self._execution_strategy,
            resource_usage=await self._get_resource_usage()
        )
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already complete
        """
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            # Remove from active tasks
            del self._active_tasks[task_id]
            
            logger.info(f"Task '{task_id}' cancelled")
            return True
        
        return False
    
    async def pause_execution(self) -> None:
        """
        Pause task execution.
        """
        logger.info("Pausing task execution")
        # Implementation would set a pause flag that workers check
        
    async def resume_execution(self) -> None:
        """
        Resume task execution.
        """
        logger.info("Resuming task execution")
        # Implementation would clear pause flag
    
    async def cleanup(self) -> None:
        """
        Cleanup the task engine.
        """
        logger.info("Cleaning up TaskEngine")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all active tasks
        for task_id in list(self._active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Stop workers
        for worker in self._workers:
            worker.cancel()
        
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Save metrics
        await self.metrics_collector.flush()
        
        logger.info("TaskEngine cleanup completed")
    
    # Private execution methods
    
    async def _execute_sequential(
        self,
        tasks: List[DevelopmentTask],
        validate: bool,
        context: TaskExecutionContext
    ) -> List[TaskResult]:
        """Execute tasks sequentially."""
        results = []
        
        for task in tasks:
            try:
                result = await self._execute_single_task(task, validate, context)
                results.append(result)
                
                # Stop on first failure if configured
                if not result.success and context.workflow and context.workflow.fail_fast:
                    break
                    
            except Exception as e:
                logger.error(f"Sequential execution failed for task {task.id}: {e}")
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=str(e),
                    execution_time=0
                ))
                break
        
        return results
    
    async def _execute_parallel(
        self,
        tasks: List[DevelopmentTask],
        validate: bool,
        context: TaskExecutionContext
    ) -> List[TaskResult]:
        """Execute tasks in parallel."""
        # Group tasks by dependency level
        dependency_levels = self.dependency_resolver.get_dependency_levels(tasks)
        results = []
        
        for level_tasks in dependency_levels:
            # Execute all tasks in this level concurrently
            level_results = await asyncio.gather(*[
                self._execute_single_task(task, validate, context)
                for task in level_tasks
            ], return_exceptions=True)
            
            # Process results
            for task, result in zip(level_tasks, level_results):
                if isinstance(result, Exception):
                    results.append(TaskResult(
                        task_id=task.id,
                        success=False,
                        error_message=str(result),
                        execution_time=0
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def _execute_adaptive(
        self,
        tasks: List[DevelopmentTask],
        validate: bool,
        context: TaskExecutionContext
    ) -> List[TaskResult]:
        """Execute tasks with adaptive strategy."""
        # Analyze tasks to determine optimal approach
        if len(tasks) <= 3:
            return await self._execute_sequential(tasks, validate, context)
        elif all(not task.dependencies for task in tasks):
            return await self._execute_parallel(tasks, validate, context)
        else:
            return await self._execute_balanced(tasks, validate, context)
    
    async def _execute_balanced(
        self,
        tasks: List[DevelopmentTask],
        validate: bool,
        context: TaskExecutionContext
    ) -> List[TaskResult]:
        """Execute tasks with balanced approach."""
        # Mix of parallel and sequential based on dependencies
        dependency_levels = self.dependency_resolver.get_dependency_levels(tasks)
        results = []
        
        for level_tasks in dependency_levels:
            if len(level_tasks) == 1:
                # Single task - execute directly
                result = await self._execute_single_task(level_tasks[0], validate, context)
                results.append(result)
            else:
                # Multiple tasks - execute with concurrency limit
                semaphore = asyncio.Semaphore(min(len(level_tasks), self._max_concurrent_tasks))
                
                async def execute_with_semaphore(task):
                    async with semaphore:
                        return await self._execute_single_task(task, validate, context)
                
                level_results = await asyncio.gather(*[
                    execute_with_semaphore(task) for task in level_tasks
                ], return_exceptions=True)
                
                # Process results
                for task, result in zip(level_tasks, level_results):
                    if isinstance(result, Exception):
                        results.append(TaskResult(
                            task_id=task.id,
                            success=False,
                            error_message=str(result),
                            execution_time=0
                        ))
                    else:
                        results.append(result)
        
        return results
    
    async def _execute_single_task(
        self,
        task: DevelopmentTask,
        validate: bool,
        context: TaskExecutionContext
    ) -> TaskResult:
        """Execute a single task."""
        task_start = datetime.now()
        
        try:
            # Add to active tasks
            self._active_tasks[task.id] = task
            task.status = TaskStatus.RUNNING
            task.started_at = task_start
            
            logger.debug(f"Executing task: {task.name}")
            
            # Import here to avoid circular imports
            from claude_tiu.integrations.ai_interface import AIInterface
            
            # Execute with AI interface
            ai_interface = AIInterface(self.config_manager)
            result = await ai_interface.execute_development_task(task, context.project)
            
            # Validate if requested
            if validate and result.success:
                from claude_tiu.core.progress_validator import ProgressValidator
                validator = ProgressValidator(self.config_manager)
                
                validation = await validator.validate_task_result(
                    task, result, context.project
                )
                
                if not validation.is_valid:
                    result.success = False
                    result.error_message = f"Validation failed: {validation.summary}"
            
            # Update execution time
            result.execution_time = (datetime.now() - task_start).total_seconds()
            
            # Move to completed
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            del self._active_tasks[task.id]
            
            if result.success:
                self._completed_tasks[task.id] = result
            else:
                self._failed_tasks[task.id] = Exception(result.error_message or "Task failed")
            
            logger.debug(f"Task '{task.name}' completed: success={result.success}")
            
            return result
            
        except Exception as e:
            logger.error(f"Task '{task.name}' execution failed: {e}")
            
            # Update task status
            task.status = TaskStatus.FAILED
            if task.id in self._active_tasks:
                del self._active_tasks[task.id]
            
            self._failed_tasks[task.id] = e
            
            execution_time = (datetime.now() - task_start).total_seconds()
            
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _start_workers(self) -> None:
        """Start background workers."""
        # For now, we'll use direct execution rather than background workers
        # This can be expanded later for more sophisticated task queuing
        pass
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics."""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'active_tasks': len(self._active_tasks),
            'max_concurrent': self._max_concurrent_tasks
        }