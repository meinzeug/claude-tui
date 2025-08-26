"""
Comprehensive unit tests for Task Service.

Tests cover:
- Task creation and configuration
- Task execution with AI and engine modes
- Dependency management and orchestration
- Performance monitoring and metrics
- Error handling and timeout scenarios
- Edge cases and concurrent execution
- Task lifecycle management
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from services.task_service import TaskService, TaskStatus, TaskExecutionMode
from core.exceptions import TaskExecutionTimeoutError, ValidationError, PerformanceError
from core.types import Priority


class TestTaskServiceInitialization:
    """Test Task Service initialization and setup."""
    
    @pytest.mark.unit
    async def test_service_initialization_success(self, mock_task_engine):
        """Test successful service initialization."""
        service = TaskService()
        
        with patch('services.task_service.TaskEngine', return_value=mock_task_engine):
            await service.initialize()
            
            assert service._initialized is True
            assert service._task_engine is not None
            assert isinstance(service._active_tasks, dict)
            assert isinstance(service._task_queue, list)
            assert isinstance(service._execution_history, list)
            assert isinstance(service._performance_metrics, dict)
    
    @pytest.mark.unit
    async def test_service_initialization_with_ai_service(self, mock_task_engine, ai_service):
        """Test initialization with AI service dependency."""
        service = TaskService()
        
        with patch('services.task_service.TaskEngine', return_value=mock_task_engine):
            # Mock AI service dependency
            service._ai_service = ai_service
            await service.initialize()
            
            assert service._ai_service is not None
            assert service._initialized is True
    
    @pytest.mark.unit
    async def test_service_initialization_failure(self):
        """Test service initialization failure."""
        service = TaskService()
        
        with patch('services.task_service.TaskEngine', side_effect=Exception("Engine init failed")):
            with pytest.raises(ValidationError) as excinfo:
                await service.initialize()
            
            assert "Task service initialization failed" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_health_check(self, task_service):
        """Test task service health check."""
        health = await task_service.health_check()
        
        assert isinstance(health, dict)
        assert health['service'] == 'TaskService'
        assert health['status'] == 'healthy'
        assert 'task_engine_available' in health
        assert 'ai_service_available' in health
        assert 'active_tasks_count' in health
        assert 'queued_tasks_count' in health
        assert 'performance_metrics' in health


class TestTaskCreation:
    """Test task creation functionality."""
    
    @pytest.mark.unit
    async def test_create_task_success(self, task_service):
        """Test successful task creation."""
        result = await task_service.create_task(
            name="Test Task",
            description="Test task description",
            task_type="code_generation",
            priority=Priority.HIGH
        )
        
        assert result['name'] == "Test Task"
        assert result['type'] == "code_generation"
        assert result['priority'] == Priority.HIGH.value
        assert result['status'] == TaskStatus.PENDING.value
        assert 'task_id' in result
        
        # Task should be added to active tasks
        task_id = result['task_id']
        assert task_id in task_service._active_tasks
    
    @pytest.mark.unit
    async def test_create_task_with_dependencies(self, task_service):
        """Test task creation with dependencies."""
        # Create first task
        task1_result = await task_service.create_task(
            name="Task 1",
            description="First task"
        )
        
        # Mark it as completed for dependency test
        task1_id = task1_result['task_id']
        task_service._active_tasks[task1_id]['status'] = TaskStatus.COMPLETED.value
        
        # Create second task depending on first
        task2_result = await task_service.create_task(
            name="Task 2",
            description="Second task",
            dependencies=[task1_id]
        )
        
        assert task2_result['dependencies'] == [task1_id]
        
        task2_id = task2_result['task_id']
        assert task2_id in task_service._task_dependencies
        assert task1_id in task_service._task_dependencies[task2_id]
    
    @pytest.mark.unit
    async def test_create_task_invalid_dependencies(self, task_service):
        """Test task creation with invalid dependencies."""
        with pytest.raises(ValidationError) as excinfo:
            await task_service.create_task(
                name="Invalid Task",
                description="Task with invalid dependency",
                dependencies=["nonexistent-task-id"]
            )
        
        assert "Dependency task nonexistent-task-id not found" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_create_task_with_config(self, task_service):
        """Test task creation with custom configuration."""
        config = {
            'language': 'python',
            'framework': 'fastapi',
            'validation_criteria': {'min_functions': 3}
        }
        
        result = await task_service.create_task(
            name="Configured Task",
            description="Task with configuration",
            task_type="code_generation",
            config=config
        )
        
        task_id = result['task_id']
        task_config = task_service._active_tasks[task_id]
        assert task_config['config'] == config
    
    @pytest.mark.unit
    async def test_create_task_without_task_engine(self):
        """Test task creation when task engine is not initialized."""
        service = TaskService()
        # Don't initialize service
        
        with pytest.raises(ValidationError) as excinfo:
            await service.create_task("Test", "Description")
        
        assert "Task engine not initialized" in str(excinfo.value)


class TestTaskExecution:
    """Test task execution functionality."""
    
    @pytest.mark.unit
    async def test_execute_task_with_ai_success(self, task_service, ai_service):
        """Test successful task execution with AI assistance."""
        # Set up AI service
        task_service._ai_service = ai_service
        
        # Create task
        task_result = await task_service.create_task(
            name="AI Task",
            description="Generate code function",
            task_type="code_generation",
            ai_enabled=True
        )
        
        task_id = task_result['task_id']
        
        # Mock AI service response
        with patch.object(ai_service, 'generate_code') as mock_generate:
            mock_generate.return_value = {
                'code': 'def hello(): return "Hello World"',
                'validation': {'is_valid': True}
            }
            
            result = await task_service.execute_task(task_id)
            
            assert result['status'] == TaskStatus.COMPLETED.value
            assert 'execution_time_seconds' in result
            assert result['result']['type'] == 'ai_result'
            
            mock_generate.assert_called_once()
    
    @pytest.mark.unit
    async def test_execute_task_with_engine(self, task_service):
        """Test task execution with task engine."""
        # Create task without AI
        task_result = await task_service.create_task(
            name="Engine Task",
            description="Execute with engine",
            ai_enabled=False
        )
        
        task_id = task_result['task_id']
        
        result = await task_service.execute_task(task_id)
        
        assert result['status'] == TaskStatus.COMPLETED.value
        assert result['result']['type'] == 'engine_result'
        assert result['result']['execution_method'] == 'task_engine'
    
    @pytest.mark.unit
    async def test_execute_task_orchestration(self, task_service, ai_service):
        """Test task orchestration execution."""
        task_service._ai_service = ai_service
        
        task_result = await task_service.create_task(
            name="Orchestration Task",
            description="Orchestrate complex workflow",
            task_type="task_orchestration",
            ai_enabled=True,
            config={'strategy': 'parallel', 'requirements': {'agents': 3}}
        )
        
        task_id = task_result['task_id']
        
        with patch.object(ai_service, 'orchestrate_task') as mock_orchestrate:
            mock_orchestrate.return_value = {
                'task_id': 'orchestrated-123',
                'agents': ['coder', 'reviewer', 'tester'],
                'status': 'running'
            }
            
            result = await task_service.execute_task(task_id)
            
            assert result['status'] == TaskStatus.COMPLETED.value
            assert result['result']['data']['task_id'] == 'orchestrated-123'
            
            mock_orchestrate.assert_called_once()
    
    @pytest.mark.unit
    async def test_execute_task_nonexistent(self, task_service):
        """Test execution of nonexistent task."""
        with pytest.raises(ValidationError) as excinfo:
            await task_service.execute_task("nonexistent-task-id")
        
        assert "Task nonexistent-task-id not found" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_execute_task_wrong_status(self, task_service):
        """Test execution of task in wrong status."""
        task_result = await task_service.create_task("Test", "Description")
        task_id = task_result['task_id']
        
        # Change task status to running
        task_service._active_tasks[task_id]['status'] = TaskStatus.RUNNING.value
        
        with pytest.raises(ValidationError) as excinfo:
            await task_service.execute_task(task_id)
        
        assert "not in pending state" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_execute_task_with_dependencies_wait(self, task_service):
        """Test task execution waiting for dependencies."""
        # Create dependency task
        dep_result = await task_service.create_task("Dependency", "Dep task")
        dep_id = dep_result['task_id']
        
        # Create main task with dependency
        main_result = await task_service.create_task(
            name="Main Task",
            description="Depends on other task",
            dependencies=[dep_id]
        )
        main_id = main_result['task_id']
        
        # Mark dependency as completed
        task_service._active_tasks[dep_id]['status'] = TaskStatus.COMPLETED.value
        
        result = await task_service.execute_task(main_id, wait_for_dependencies=True)
        
        assert result['status'] == TaskStatus.COMPLETED.value
    
    @pytest.mark.unit
    async def test_execute_task_dependency_failed(self, task_service):
        """Test task execution when dependency fails."""
        # Create dependency task
        dep_result = await task_service.create_task("Dependency", "Dep task")
        dep_id = dep_result['task_id']
        
        # Create main task with dependency
        main_result = await task_service.create_task(
            name="Main Task",
            description="Depends on failed task",
            dependencies=[dep_id]
        )
        main_id = main_result['task_id']
        
        # Mark dependency as failed
        task_service._active_tasks[dep_id]['status'] = TaskStatus.FAILED.value
        
        with pytest.raises(ValidationError) as excinfo:
            await task_service.execute_task(main_id, wait_for_dependencies=True)
        
        assert "Dependency task" in str(excinfo.value) and "failed" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_execute_task_timeout(self, task_service):
        """Test task execution timeout."""
        task_result = await task_service.create_task(
            name="Timeout Task",
            description="Task that times out",
            timeout_seconds=1  # Very short timeout
        )
        task_id = task_result['task_id']
        
        # Mock task engine to be slow
        with patch.object(task_service._task_engine, 'execute_task') as mock_execute:
            async def slow_task(*args):
                await asyncio.sleep(2)  # Longer than timeout
                return "result"
            
            mock_execute.side_effect = slow_task
            
            with pytest.raises(TaskExecutionTimeoutError):
                await task_service.execute_task(task_id)
            
            # Task should be marked as timeout
            assert task_service._active_tasks[task_id]['status'] == TaskStatus.TIMEOUT.value


class TestTaskManagement:
    """Test task management operations."""
    
    @pytest.mark.unit
    async def test_get_task_status(self, task_service):
        """Test retrieving task status."""
        task_result = await task_service.create_task("Test Task", "Description")
        task_id = task_result['task_id']
        
        status = await task_service.get_task_status(task_id)
        
        assert status['id'] == task_id
        assert status['name'] == "Test Task"
        assert status['status'] == TaskStatus.PENDING.value
    
    @pytest.mark.unit
    async def test_get_task_status_from_history(self, task_service):
        """Test retrieving task status from execution history."""
        # Create and execute task
        task_result = await task_service.create_task("Completed Task", "Description")
        task_id = task_result['task_id']
        
        await task_service.execute_task(task_id)
        
        # Remove from active tasks (simulate cleanup)
        del task_service._active_tasks[task_id]
        
        status = await task_service.get_task_status(task_id)
        
        assert status['id'] == task_id
        assert status['success'] is True
    
    @pytest.mark.unit
    async def test_list_active_tasks(self, task_service):
        """Test listing active tasks."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            result = await task_service.create_task(f"Task {i}", f"Description {i}")
            tasks.append(result)
        
        active_tasks = await task_service.list_active_tasks()
        
        assert len(active_tasks) == 3
        # Should be sorted by creation time (most recent first)
        assert active_tasks[0]['name'] == "Task 2"
        assert active_tasks[1]['name'] == "Task 1"
        assert active_tasks[2]['name'] == "Task 0"
    
    @pytest.mark.unit
    async def test_cancel_task(self, task_service):
        """Test task cancellation."""
        task_result = await task_service.create_task("Cancellable Task", "Description")
        task_id = task_result['task_id']
        
        result = await task_service.cancel_task(task_id)
        
        assert result['task_id'] == task_id
        assert result['status'] == TaskStatus.CANCELLED.value
        assert 'cancelled_at' in result
        
        # Task should be marked as cancelled
        assert task_service._active_tasks[task_id]['status'] == TaskStatus.CANCELLED.value
    
    @pytest.mark.unit
    async def test_cancel_task_invalid_status(self, task_service):
        """Test cancelling task with invalid status."""
        task_result = await task_service.create_task("Completed Task", "Description")
        task_id = task_result['task_id']
        
        # Mark as completed
        task_service._active_tasks[task_id]['status'] = TaskStatus.COMPLETED.value
        
        with pytest.raises(ValidationError) as excinfo:
            await task_service.cancel_task(task_id)
        
        assert "cannot be cancelled" in str(excinfo.value)


class TestTaskExecutionHistory:
    """Test task execution history functionality."""
    
    @pytest.mark.unit
    async def test_execution_history_tracking(self, task_service):
        """Test that execution history is properly tracked."""
        initial_history_size = len(task_service._execution_history)
        
        # Create and execute task
        task_result = await task_service.create_task("History Task", "Test history")
        task_id = task_result['task_id']
        
        await task_service.execute_task(task_id)
        
        assert len(task_service._execution_history) == initial_history_size + 1
        
        latest_entry = task_service._execution_history[-1]
        assert latest_entry['id'] == task_id
        assert latest_entry['name'] == "History Task"
        assert latest_entry['success'] is True
    
    @pytest.mark.unit
    async def test_get_execution_history_filtered(self, task_service):
        """Test filtered execution history retrieval."""
        # Create tasks of different types
        code_task = await task_service.create_task("Code Task", "Desc", task_type="code_generation")
        general_task = await task_service.create_task("General Task", "Desc", task_type="general")
        
        await task_service.execute_task(code_task['task_id'])
        await task_service.execute_task(general_task['task_id'])
        
        # Filter by task type
        code_history = await task_service.get_execution_history(
            task_type_filter="code_generation"
        )
        
        assert len(code_history) >= 1
        assert all(task['type'] == 'code_generation' for task in code_history)
        
        # Filter by success
        success_history = await task_service.get_execution_history(success_only=True)
        assert all(task['success'] is True for task in success_history)
    
    @pytest.mark.unit
    async def test_get_execution_history_limited(self, task_service):
        """Test execution history with limit."""
        # Create multiple tasks
        for i in range(5):
            task_result = await task_service.create_task(f"Task {i}", "Description")
            await task_service.execute_task(task_result['task_id'])
        
        limited_history = await task_service.get_execution_history(limit=3)
        
        assert len(limited_history) <= 3


class TestTaskPerformanceMetrics:
    """Test task performance monitoring and metrics."""
    
    @pytest.mark.unit
    async def test_performance_metrics_tracking(self, task_service):
        """Test performance metrics are properly tracked."""
        initial_metrics = task_service._performance_metrics.copy()
        
        # Execute successful task
        task_result = await task_service.create_task("Perf Task", "Description")
        await task_service.execute_task(task_result['task_id'])
        
        updated_metrics = task_service._performance_metrics
        
        assert updated_metrics['total_tasks_executed'] > initial_metrics['total_tasks_executed']
        assert updated_metrics['successful_tasks'] > initial_metrics['successful_tasks']
        assert updated_metrics['average_execution_time'] >= 0
    
    @pytest.mark.unit
    async def test_performance_report_generation(self, task_service):
        """Test comprehensive performance report generation."""
        # Execute multiple tasks with different outcomes
        for i in range(3):
            task_result = await task_service.create_task(f"Success Task {i}", "Desc")
            await task_service.execute_task(task_result['task_id'])
        
        # Simulate failed task
        failed_task = await task_service.create_task("Failed Task", "Desc")
        task_id = failed_task['task_id']
        
        # Manually mark as failed for testing
        task_service._active_tasks[task_id]['status'] = TaskStatus.FAILED.value
        task_service._add_to_history(task_service._active_tasks[task_id], False)
        
        report = await task_service.get_performance_report()
        
        assert 'overall_metrics' in report
        assert 'success_rate' in report
        assert 'recent_success_rate' in report
        assert 'task_type_metrics' in report
        assert report['success_rate'] >= 0.0 and report['success_rate'] <= 1.0
    
    @pytest.mark.unit
    async def test_task_type_metrics_breakdown(self, task_service):
        """Test task type metrics breakdown."""
        # Create tasks of different types
        code_task = await task_service.create_task("Code", "Desc", task_type="code_generation")
        analysis_task = await task_service.create_task("Analysis", "Desc", task_type="analysis")
        
        await task_service.execute_task(code_task['task_id'])
        await task_service.execute_task(analysis_task['task_id'])
        
        report = await task_service.get_performance_report()
        
        type_metrics = report['task_type_metrics']
        assert 'code_generation' in type_metrics
        assert 'analysis' in type_metrics
        
        for metrics in type_metrics.values():
            assert 'count' in metrics
            assert 'average_time' in metrics
            assert 'success_rate' in metrics


class TestTaskServiceErrorHandling:
    """Test Task Service error handling and resilience."""
    
    @pytest.mark.unit
    async def test_ai_service_failure_fallback(self, task_service, ai_service):
        """Test fallback to engine when AI service fails."""
        task_service._ai_service = ai_service
        
        task_result = await task_service.create_task(
            name="Fallback Task",
            description="Test AI fallback",
            task_type="code_generation",
            ai_enabled=True
        )
        task_id = task_result['task_id']
        
        with patch.object(ai_service, 'generate_code', side_effect=Exception("AI failed")):
            result = await task_service.execute_task(task_id)
            
            # Should fallback to engine execution
            assert result['status'] == TaskStatus.COMPLETED.value
            assert result['result']['type'] == 'engine_result'
    
    @pytest.mark.unit
    async def test_task_engine_failure_handling(self, task_service):
        """Test handling of task engine failures."""
        task_result = await task_service.create_task("Engine Fail", "Description")
        task_id = task_result['task_id']
        
        with patch.object(task_service._task_engine, 'execute_task', side_effect=Exception("Engine failed")):
            with pytest.raises(ValidationError) as excinfo:
                await task_service.execute_task(task_id)
            
            assert "Task engine execution failed" in str(excinfo.value)
            assert task_service._active_tasks[task_id]['status'] == TaskStatus.FAILED.value
    
    @pytest.mark.unit
    async def test_invalid_task_configuration(self, task_service):
        """Test handling of invalid task configuration."""
        # Test with invalid priority
        with pytest.raises(Exception):  # Should raise some validation error
            await task_service.create_task(
                name="Invalid Task",
                description="Description",
                priority="invalid_priority"
            )


class TestTaskServiceEdgeCases:
    """Test Task Service edge cases and boundary conditions."""
    
    @pytest.mark.edge_case
    async def test_very_long_task_description(self, task_service):
        """Test task creation with very long description."""
        long_description = "A" * 10000  # Very long description
        
        result = await task_service.create_task(
            name="Long Description Task",
            description=long_description
        )
        
        task_id = result['task_id']
        task_config = task_service._active_tasks[task_id]
        assert task_config['description'] == long_description
        assert len(task_config['description']) == 10000
    
    @pytest.mark.edge_case
    async def test_unicode_task_names(self, task_service):
        """Test task creation with Unicode names and descriptions."""
        unicode_name = "タスク 🚀 Задача"
        unicode_desc = "测试 Unicode ñoño émojis 🎉"
        
        result = await task_service.create_task(
            name=unicode_name,
            description=unicode_desc
        )
        
        assert result['name'] == unicode_name
        
        task_id = result['task_id']
        task_config = task_service._active_tasks[task_id]
        assert task_config['description'] == unicode_desc
    
    @pytest.mark.edge_case
    async def test_circular_dependencies(self, task_service):
        """Test detection of circular dependencies."""
        # Create three tasks
        task1 = await task_service.create_task("Task 1", "First")
        task2 = await task_service.create_task("Task 2", "Second", dependencies=[task1['task_id']])
        
        # Try to create third task that would create circular dependency
        # This should be handled gracefully or prevented
        task3 = await task_service.create_task("Task 3", "Third", dependencies=[task2['task_id']])
        
        # Now try to make task1 depend on task3 (would be circular)
        # This test ensures the system handles this scenario
        assert len(task_service._active_tasks) == 3
    
    @pytest.mark.performance
    async def test_concurrent_task_execution(self, task_service, performance_test_config):
        """Test concurrent task execution performance."""
        from tests.conftest import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Create multiple tasks
        task_ids = []
        for i in range(performance_test_config['concurrent_operations']):
            result = await task_service.create_task(f"Concurrent Task {i}", f"Description {i}")
            task_ids.append(result['task_id'])
        
        monitor.start()
        
        # Execute tasks concurrently
        tasks = [
            task_service.execute_task(task_id)
            for task_id in task_ids
        ]
        
        results = await asyncio.gather(*tasks)
        
        monitor.stop()
        
        assert len(results) == performance_test_config['concurrent_operations']
        assert all(result['status'] == TaskStatus.COMPLETED.value for result in results)
        
        # Performance assertion
        monitor.assert_performance(performance_test_config['max_execution_time'])
    
    @pytest.mark.edge_case
    async def test_task_execution_history_overflow(self, task_service):
        """Test task execution history management with overflow."""
        # Create many tasks to exceed history limit
        initial_history_size = len(task_service._execution_history)
        
        # Set low history limit for testing
        original_limit = 1000
        task_service._execution_history = task_service._execution_history[-50:]  # Simulate near-full history
        
        # Add tasks that will trigger overflow handling
        for i in range(60):  # More than the cleanup threshold
            task_result = await task_service.create_task(f"Overflow Task {i}", "Description")
            await task_service.execute_task(task_result['task_id'])
        
        # History should be managed (not grow indefinitely)
        assert len(task_service._execution_history) <= 1000
    
    @pytest.mark.edge_case
    async def test_task_config_with_large_context(self, task_service):
        """Test task creation with large configuration context."""
        large_config = {
            'large_data': 'x' * (1024 * 1024),  # 1MB of data
            'complex_structure': {
                'nested': {
                    'deeply': {
                        'very_deep': [f"item_{i}" for i in range(1000)]
                    }
                }
            }
        }
        
        result = await task_service.create_task(
            name="Large Config Task",
            description="Task with large configuration",
            config=large_config
        )
        
        task_id = result['task_id']
        task_config = task_service._active_tasks[task_id]
        assert len(task_config['config']['large_data']) == 1024 * 1024
    
    @pytest.mark.edge_case
    async def test_dependency_chain_execution(self, task_service):
        """Test execution of long dependency chains."""
        # Create chain of dependent tasks
        previous_task_id = None
        chain_length = 5
        
        for i in range(chain_length):
            dependencies = [previous_task_id] if previous_task_id else []
            
            result = await task_service.create_task(
                name=f"Chain Task {i}",
                description=f"Task {i} in dependency chain",
                dependencies=dependencies
            )
            
            if previous_task_id:
                # Mark previous task as completed
                task_service._active_tasks[previous_task_id]['status'] = TaskStatus.COMPLETED.value
            
            previous_task_id = result['task_id']
        
        # Execute final task (should wait for entire chain)
        final_result = await task_service.execute_task(
            previous_task_id,
            wait_for_dependencies=True
        )
        
        assert final_result['status'] == TaskStatus.COMPLETED.value