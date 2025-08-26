"""
Task Service for claude-tui.

Manages task lifecycle, execution, and coordination:
- Task creation and configuration
- Execution orchestration with AI services
- Progress monitoring and validation
- Task dependency management
- Performance tracking and optimization
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from ..core.exceptions import TaskExecutionTimeoutError, ValidationError, PerformanceError
from ..core.task_engine import TaskEngine, ExecutionStrategy, create_task
from ..core.types import Task, TaskState, Priority
from .base import BaseService
from .ai_service import AIService


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskExecutionMode(Enum):
    """Task execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"


class TaskService(BaseService):
    """
    Task Management Service.
    
    Provides high-level task orchestration with AI integration,
    dependency management, and performance monitoring.
    """
    
    def __init__(self):
        super().__init__()
        self._task_engine: Optional[TaskEngine] = None
        self._ai_service: Optional[AIService] = None
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_queue: List[Dict[str, Any]] = []
        self._execution_history: List[Dict[str, Any]] = []
        self._task_dependencies: Dict[str, Set[str]] = {}
        self._performance_metrics: Dict[str, Any] = {}
        
    async def _initialize_impl(self) -> None:
        """Initialize task service."""
        try:
            # Initialize task engine
            self._task_engine = TaskEngine()
            
            # Get AI service dependency (if available)
            try:
                self._ai_service = self.get_dependency(AIService)
                self.logger.info("AI service integration available for task execution")
            except Exception as e:
                self.logger.warning(f"AI service not available: {str(e)}")
            
            # Initialize performance tracking
            self._performance_metrics = {
                'total_tasks_executed': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'average_execution_time': 0.0,
                'last_reset': datetime.utcnow().isoformat()
            }
            
            self.logger.info("Task service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize task service: {str(e)}")
            raise ValidationError(f"Task service initialization failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check with task-specific status."""
        base_health = await super().health_check()
        
        base_health.update({
            'task_engine_available': self._task_engine is not None,
            'ai_service_available': self._ai_service is not None,
            'active_tasks_count': len(self._active_tasks),
            'queued_tasks_count': len(self._task_queue),
            'execution_history_size': len(self._execution_history),
            'performance_metrics': self._performance_metrics.copy()
        })
        
        return base_health
    
    async def create_task(
        self,
        name: str,
        description: str,
        task_type: str = 'general',
        priority: Priority = Priority.MEDIUM,
        timeout_seconds: Optional[int] = None,
        dependencies: Optional[List[str]] = None,
        ai_enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new task with specified configuration.
        
        Args:
            name: Task name
            description: Task description
            task_type: Type of task (code_generation, analysis, etc.)
            priority: Task priority level
            timeout_seconds: Task timeout in seconds
            dependencies: List of task IDs this task depends on
            ai_enabled: Whether AI assistance is enabled
            config: Additional task configuration
            
        Returns:
            Created task information
        """
        return await self.execute_with_monitoring(
            'create_task',
            self._create_task_impl,
            name=name,
            description=description,
            task_type=task_type,
            priority=priority,
            timeout_seconds=timeout_seconds,
            dependencies=dependencies,
            ai_enabled=ai_enabled,
            config=config
        )
    
    async def _create_task_impl(
        self,
        name: str,
        description: str,
        task_type: str,
        priority: Priority,
        timeout_seconds: Optional[int],
        dependencies: Optional[List[str]],
        ai_enabled: bool,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Internal task creation implementation."""
        if not self._task_engine:
            raise ValidationError("Task engine not initialized")
        
        # Generate task ID
        task_id = str(uuid4())
        
        # Validate dependencies
        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self._active_tasks and not self._is_task_completed(dep_id):
                    raise ValidationError(f"Dependency task {dep_id} not found or not completed")
        
        # Create task using task engine
        core_task = create_task(
            name=name,
            description=description,
            task_type=task_type,
            priority=priority
        )
        
        # Prepare task configuration
        task_config = {
            'id': task_id,
            'core_task': core_task,
            'name': name,
            'description': description,
            'type': task_type,
            'priority': priority.value,
            'status': TaskStatus.PENDING.value,
            'ai_enabled': ai_enabled,
            'timeout_seconds': timeout_seconds or 300,
            'dependencies': dependencies or [],
            'config': config or {},
            'created_at': datetime.utcnow().isoformat(),
            'started_at': None,
            'completed_at': None,
            'execution_time_seconds': None,
            'result': None,
            'error': None,
            'context': self.get_context()
        }
        
        # Store task dependencies
        if dependencies:
            self._task_dependencies[task_id] = set(dependencies)
        
        # Add to active tasks
        self._active_tasks[task_id] = task_config
        
        self.logger.info(f"Created task: {name} (ID: {task_id})")
        
        return {
            'task_id': task_id,
            'name': name,
            'type': task_type,
            'priority': priority.value,
            'status': TaskStatus.PENDING.value,
            'created_at': task_config['created_at'],
            'dependencies': dependencies or [],
            'ai_enabled': ai_enabled
        }
    
    async def execute_task(
        self,
        task_id: str,
        execution_mode: TaskExecutionMode = TaskExecutionMode.ADAPTIVE,
        wait_for_dependencies: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a specific task.
        
        Args:
            task_id: ID of task to execute
            execution_mode: How to execute the task
            wait_for_dependencies: Whether to wait for dependencies
            
        Returns:
            Task execution result
        """
        return await self.execute_with_monitoring(
            'execute_task',
            self._execute_task_impl,
            task_id=task_id,
            execution_mode=execution_mode,
            wait_for_dependencies=wait_for_dependencies
        )
    
    async def _execute_task_impl(
        self,
        task_id: str,
        execution_mode: TaskExecutionMode,
        wait_for_dependencies: bool
    ) -> Dict[str, Any]:
        """Internal task execution implementation."""
        if task_id not in self._active_tasks:
            raise ValidationError(f"Task {task_id} not found")
        
        task_config = self._active_tasks[task_id]
        
        if task_config['status'] != TaskStatus.PENDING.value:
            raise ValidationError(f"Task {task_id} is not in pending state")
        
        # Wait for dependencies if required
        if wait_for_dependencies and task_config['dependencies']:
            await self._wait_for_dependencies(task_id)
        
        # Update task status
        task_config['status'] = TaskStatus.RUNNING.value
        task_config['started_at'] = datetime.utcnow().isoformat()
        
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting execution of task: {task_config['name']}")
            
            # Execute based on mode and AI availability
            if task_config['ai_enabled'] and self._ai_service:
                result = await self._execute_with_ai(task_config)
            else:
                result = await self._execute_with_engine(task_config)
            
            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            # Update task status
            task_config.update({
                'status': TaskStatus.COMPLETED.value,
                'completed_at': end_time.isoformat(),
                'execution_time_seconds': execution_time,
                'result': result
            })
            
            # Update performance metrics
            self._update_performance_metrics(True, execution_time)
            
            # Add to execution history
            self._add_to_history(task_config, True)
            
            self.logger.info(f"Completed task: {task_config['name']} in {execution_time:.2f}s")
            
            return {
                'task_id': task_id,
                'status': TaskStatus.COMPLETED.value,
                'execution_time_seconds': execution_time,
                'result': result,
                'completed_at': task_config['completed_at']
            }
            
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            task_config.update({
                'status': TaskStatus.TIMEOUT.value,
                'completed_at': datetime.utcnow().isoformat(),
                'execution_time_seconds': execution_time,
                'error': 'Task execution timeout'
            })
            
            self._update_performance_metrics(False, execution_time)
            self._add_to_history(task_config, False)
            
            raise TaskExecutionTimeoutError(
                task_config['name'],
                task_config['timeout_seconds'],
                int(execution_time)
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            task_config.update({
                'status': TaskStatus.FAILED.value,
                'completed_at': datetime.utcnow().isoformat(),
                'execution_time_seconds': execution_time,
                'error': str(e)
            })
            
            self._update_performance_metrics(False, execution_time)
            self._add_to_history(task_config, False)
            
            self.logger.error(f"Task execution failed: {task_config['name']}: {str(e)}")
            raise
    
    async def _execute_with_ai(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with AI assistance."""
        if not self._ai_service:
            raise ValidationError("AI service not available")
        
        task_type = task_config['type']
        
        try:
            if task_type == 'code_generation':
                result = await self._ai_service.generate_code(
                    prompt=task_config['description'],
                    language=task_config['config'].get('language', 'python'),
                    context=task_config['config'].get('context'),
                    validate_response=True
                )
            elif task_type == 'task_orchestration':
                result = await self._ai_service.orchestrate_task(
                    task_description=task_config['description'],
                    requirements=task_config['config'].get('requirements'),
                    strategy=task_config['config'].get('strategy', 'adaptive')
                )
            else:
                # Generic AI task execution
                result = await self._execute_generic_ai_task(task_config)
            
            return {
                'type': 'ai_result',
                'data': result,
                'ai_provider': 'claude_integration',
                'execution_method': 'ai_assisted'
            }
            
        except Exception as e:
            self.logger.error(f"AI-assisted execution failed: {str(e)}")
            # Fallback to engine execution
            return await self._execute_with_engine(task_config)
    
    async def _execute_with_engine(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with task engine."""
        if not self._task_engine:
            raise ValidationError("Task engine not available")
        
        core_task = task_config['core_task']
        
        # Set timeout
        timeout = task_config['timeout_seconds']
        
        try:
            result = await asyncio.wait_for(
                self._task_engine.execute_task(core_task),
                timeout=timeout
            )
            
            return {
                'type': 'engine_result',
                'data': result,
                'execution_method': 'task_engine'
            }
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise ValidationError(f"Task engine execution failed: {str(e)}")
    
    async def _execute_generic_ai_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic AI task."""
        # For generic tasks, we can use AI service validation
        if self._ai_service:
            validation_result = await self._ai_service.validate_response(
                response={'content': task_config['description']},
                validation_type='text',
                criteria=task_config['config'].get('validation_criteria')
            )
            
            return {
                'validation': validation_result,
                'processed': True,
                'content': task_config['description']
            }
        
        # Fallback to simple processing
        return {
            'processed': True,
            'content': task_config['description'],
            'fallback': True
        }
    
    async def _wait_for_dependencies(self, task_id: str) -> None:
        """Wait for task dependencies to complete."""
        if task_id not in self._task_dependencies:
            return
        
        dependencies = self._task_dependencies[task_id]
        
        while dependencies:
            completed_deps = set()
            
            for dep_id in dependencies:
                if self._is_task_completed(dep_id):
                    completed_deps.add(dep_id)
                elif self._is_task_failed(dep_id):
                    raise ValidationError(f"Dependency task {dep_id} failed")
            
            # Remove completed dependencies
            dependencies -= completed_deps
            
            if dependencies:
                # Wait a bit before checking again
                await asyncio.sleep(0.5)
    
    def _is_task_completed(self, task_id: str) -> bool:
        """Check if task is completed."""
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]['status'] == TaskStatus.COMPLETED.value
        
        # Check execution history
        for task_record in self._execution_history:
            if task_record.get('id') == task_id:
                return task_record.get('status') == TaskStatus.COMPLETED.value
        
        return False
    
    def _is_task_failed(self, task_id: str) -> bool:
        """Check if task has failed."""
        if task_id in self._active_tasks:
            status = self._active_tasks[task_id]['status']
            return status in [TaskStatus.FAILED.value, TaskStatus.TIMEOUT.value]
        
        # Check execution history
        for task_record in self._execution_history:
            if task_record.get('id') == task_id:
                status = task_record.get('status')
                return status in [TaskStatus.FAILED.value, TaskStatus.TIMEOUT.value]
        
        return False
    
    def _update_performance_metrics(self, success: bool, execution_time: float) -> None:
        """Update performance metrics."""
        self._performance_metrics['total_tasks_executed'] += 1
        
        if success:
            self._performance_metrics['successful_tasks'] += 1
        else:
            self._performance_metrics['failed_tasks'] += 1
        
        # Update average execution time
        total = self._performance_metrics['total_tasks_executed']
        current_avg = self._performance_metrics['average_execution_time']
        self._performance_metrics['average_execution_time'] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def _add_to_history(self, task_config: Dict[str, Any], success: bool) -> None:
        """Add task to execution history."""
        history_entry = {
            'id': task_config['id'],
            'name': task_config['name'],
            'type': task_config['type'],
            'status': task_config['status'],
            'success': success,
            'execution_time_seconds': task_config['execution_time_seconds'],
            'created_at': task_config['created_at'],
            'completed_at': task_config['completed_at'],
            'ai_enabled': task_config['ai_enabled'],
            'error': task_config.get('error')
        }
        
        self._execution_history.append(history_entry)
        
        # Keep history size manageable
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-500:]
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current task status."""
        if task_id not in self._active_tasks:
            # Check execution history
            for task_record in self._execution_history:
                if task_record.get('id') == task_id:
                    return task_record
            
            raise ValidationError(f"Task {task_id} not found")
        
        return self._active_tasks[task_id].copy()
    
    async def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks."""
        tasks = []
        
        for task_id, task_config in self._active_tasks.items():
            task_info = task_config.copy()
            # Remove core_task object for serialization
            task_info.pop('core_task', None)
            tasks.append(task_info)
        
        # Sort by creation time
        tasks.sort(key=lambda x: x['created_at'], reverse=True)
        
        return tasks
    
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a pending or running task."""
        if task_id not in self._active_tasks:
            raise ValidationError(f"Task {task_id} not found")
        
        task_config = self._active_tasks[task_id]
        
        if task_config['status'] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
            raise ValidationError(f"Task {task_id} cannot be cancelled (current status: {task_config['status']})")
        
        # Update task status
        task_config.update({
            'status': TaskStatus.CANCELLED.value,
            'completed_at': datetime.utcnow().isoformat(),
            'error': 'Task cancelled by user'
        })
        
        # Add to history
        self._add_to_history(task_config, False)
        
        self.logger.info(f"Cancelled task: {task_config['name']}")
        
        return {
            'task_id': task_id,
            'status': TaskStatus.CANCELLED.value,
            'cancelled_at': task_config['completed_at']
        }
    
    async def get_execution_history(
        self,
        limit: Optional[int] = None,
        task_type_filter: Optional[str] = None,
        success_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get task execution history with optional filtering."""
        history = self._execution_history.copy()
        
        # Apply filters
        if task_type_filter:
            history = [task for task in history if task.get('type') == task_type_filter]
        
        if success_only is not None:
            history = [task for task in history if task.get('success') == success_only]
        
        # Sort by completion time (most recent first)
        history.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
        
        # Apply limit
        if limit:
            history = history[:limit]
        
        return history
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_tasks = self._performance_metrics['total_tasks_executed']
        successful_tasks = self._performance_metrics['successful_tasks']
        failed_tasks = self._performance_metrics['failed_tasks']
        
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        # Calculate recent performance (last 50 tasks)
        recent_history = self._execution_history[-50:] if len(self._execution_history) >= 50 else self._execution_history
        recent_success_rate = sum(1 for task in recent_history if task.get('success', False)) / len(recent_history) if recent_history else 0
        
        # Calculate average execution times by task type
        type_metrics = {}
        for task in self._execution_history:
            task_type = task.get('type', 'unknown')
            if task_type not in type_metrics:
                type_metrics[task_type] = {'count': 0, 'total_time': 0, 'success_count': 0}
            
            type_metrics[task_type]['count'] += 1
            type_metrics[task_type]['total_time'] += task.get('execution_time_seconds', 0)
            if task.get('success', False):
                type_metrics[task_type]['success_count'] += 1
        
        for task_type, metrics in type_metrics.items():
            metrics['average_time'] = metrics['total_time'] / metrics['count'] if metrics['count'] > 0 else 0
            metrics['success_rate'] = metrics['success_count'] / metrics['count'] if metrics['count'] > 0 else 0
        
        return {
            'overall_metrics': self._performance_metrics.copy(),
            'success_rate': success_rate,
            'recent_success_rate': recent_success_rate,
            'active_tasks_count': len(self._active_tasks),
            'history_size': len(self._execution_history),
            'task_type_metrics': type_metrics,
            'report_generated_at': datetime.utcnow().isoformat()
        }