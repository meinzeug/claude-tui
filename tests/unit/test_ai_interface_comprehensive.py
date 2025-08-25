#!/usr/bin/env python3
"""
Comprehensive Unit Tests for AIInterface.

Tests the AI integration system including:
- Claude Code/Flow integration and coordination
- Task complexity analysis and optimization
- Response validation and authenticity checking
- Error handling and retry mechanisms
- Performance monitoring and caching
"""

import asyncio
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime, timedelta

# Import the components under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from integrations.ai_interface import AIInterface
from core.types import (
    Task, TaskStatus, Priority, AITaskResult, ValidationResult
)


@pytest.fixture
def mock_claude_code():
    """Mock Claude Code integration."""
    mock = AsyncMock()
    mock.execute_command.return_value = {
        'success': True,
        'output': 'Mock Claude Code response',
        'files_modified': ['src/main.py'],
        'execution_time': 1.5
    }
    return mock


@pytest.fixture
def mock_claude_flow():
    """Mock Claude Flow integration."""
    mock = AsyncMock()
    mock.orchestrate_task.return_value = {
        'success': True,
        'result': 'Mock Claude Flow orchestration',
        'agents_used': ['coder', 'reviewer'],
        'coordination_time': 2.3
    }
    return mock


@pytest.fixture
def sample_ai_context():
    """Sample AI context for testing."""
    return {
        'project_path': '/tmp/test_project',
        'project_type': 'python',
        'framework': 'fastapi',
        'requirements': [
            'Create REST API endpoints',
            'Add authentication',
            'Include error handling'
        ],
        'existing_files': ['main.py', 'models.py'],
        'constraints': {
            'max_complexity': 'medium',
            'performance_target': 'high',
            'security_level': 'strict'
        }
    }


@pytest.fixture
def sample_tasks():
    """Sample tasks for testing."""
    return [
        Task(
            id=uuid4(),
            name="Create API endpoints",
            description="Implement REST API endpoints for user management",
            priority=Priority.HIGH,
            estimated_duration=120
        ),
        Task(
            id=uuid4(),
            name="Add authentication",
            description="Implement JWT-based authentication",
            priority=Priority.MEDIUM,
            estimated_duration=90,
            metadata={'complexity': 'high', 'security_critical': True}
        ),
        Task(
            id=uuid4(),
            name="Write tests",
            description="Create comprehensive test suite",
            priority=Priority.LOW,
            estimated_duration=60
        )
    ]


class TestAIInterface:
    """Comprehensive tests for AIInterface class."""
    
    @pytest.mark.asyncio
    async def test_ai_interface_initialization(self):
        """Test AIInterface initialization."""
        ai_interface = AIInterface(enable_validation=True)
        
        assert ai_interface.enable_validation is True
        assert ai_interface._claude_code is not None
        assert ai_interface._claude_flow is not None
        assert ai_interface._task_complexity_analyzer is not None
        assert ai_interface._response_cache is not None
        assert ai_interface._performance_metrics is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_simple(self, sample_tasks, sample_ai_context, mock_claude_code):
        """Test simple task execution."""
        ai_interface = AIInterface(enable_validation=False)
        ai_interface._claude_code = mock_claude_code
        
        task = sample_tasks[0]  # "Create API endpoints"
        
        result = await ai_interface.execute_task(task, sample_ai_context)
        
        assert isinstance(result, AITaskResult)
        assert result.task_id == task.id
        assert result.success is True
        assert result.generated_content is not None
        assert result.execution_time > 0
        
        # Verify Claude Code was called
        mock_claude_code.execute_command.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_complex_with_flow(self, sample_tasks, sample_ai_context, mock_claude_flow):
        """Test complex task execution using Claude Flow."""
        ai_interface = AIInterface(enable_validation=False)
        ai_interface._claude_flow = mock_claude_flow
        
        task = sample_tasks[1]  # "Add authentication" - marked as high complexity
        
        # Mock complexity analysis to trigger Flow usage
        with patch.object(ai_interface._task_complexity_analyzer, 'analyze_complexity') as mock_analyze:
            mock_analyze.return_value = {
                'complexity_score': 8.5,  # High complexity
                'recommendation': 'use_flow',
                'estimated_agents': 3,
                'coordination_required': True
            }
            
            result = await ai_interface.execute_task(task, sample_ai_context)
        
        assert result.success is True
        assert 'coordination_time' in result.metadata
        
        # Verify Claude Flow was used for complex task
        mock_claude_flow.orchestrate_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_with_validation(self, sample_tasks, sample_ai_context):
        """Test task execution with validation enabled."""
        ai_interface = AIInterface(enable_validation=True)
        
        # Mock the AI services
        mock_claude_code = AsyncMock()
        mock_claude_code.execute_command.return_value = {
            'success': True,
            'output': 'Generated code with TODO comments',
            'files_modified': ['src/api.py'],
            'execution_time': 1.0
        }
        ai_interface._claude_code = mock_claude_code
        
        # Mock validator to detect issues
        mock_validator = AsyncMock()
        mock_validator.validate_ai_response.return_value = ValidationResult(
            is_authentic=False,  # Found issues
            authenticity_score=65.0,
            real_progress=60.0,
            fake_progress=40.0,
            issues=[{'type': 'placeholder', 'description': 'TODO comment found'}],
            suggestions=['Complete TODO implementations'],
            next_actions=['auto-fix-placeholders']
        )
        ai_interface._validator = mock_validator
        
        # Mock auto-fix functionality
        mock_auto_fix = AsyncMock()
        mock_auto_fix.return_value = {
            'success': True,
            'fixed_content': 'Complete implementation without TODOs',
            'fixes_applied': 1
        }
        ai_interface._auto_fix_response = mock_auto_fix
        
        task = sample_tasks[0]
        
        result = await ai_interface.execute_task(task, sample_ai_context)
        
        assert result.success is True
        assert result.validation_result is not None
        assert result.validation_result.authenticity_score == 65.0
        
        # Verify validation and auto-fix were called
        mock_validator.validate_ai_response.assert_called_once()
        mock_auto_fix.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry_mechanism(self, sample_tasks, sample_ai_context):
        """Test task execution with retry mechanism."""
        ai_interface = AIInterface(enable_validation=False)
        
        # Mock Claude Code to fail first, then succeed
        mock_claude_code = AsyncMock()
        mock_claude_code.execute_command.side_effect = [
            Exception("Network error"),  # First attempt fails
            {  # Second attempt succeeds
                'success': True,
                'output': 'Successful response',
                'files_modified': ['src/main.py'],
                'execution_time': 1.2
            }
        ]
        ai_interface._claude_code = mock_claude_code
        
        task = sample_tasks[0]
        
        result = await ai_interface.execute_task(task, sample_ai_context)
        
        assert result.success is True
        assert 'retry_attempt' in result.metadata
        assert result.metadata['retry_attempt'] == 1
        
        # Verify retry was attempted
        assert mock_claude_code.execute_command.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_task_max_retries_exceeded(self, sample_tasks, sample_ai_context):
        """Test task execution when max retries are exceeded."""
        ai_interface = AIInterface(enable_validation=False, max_retries=2)
        
        # Mock Claude Code to always fail
        mock_claude_code = AsyncMock()
        mock_claude_code.execute_command.side_effect = Exception("Persistent error")
        ai_interface._claude_code = mock_claude_code
        
        task = sample_tasks[0]
        
        result = await ai_interface.execute_task(task, sample_ai_context)
        
        assert result.success is False
        assert "Persistent error" in result.error_message
        assert result.metadata['retry_count'] == 2
        
        # Verify all retries were attempted
        assert mock_claude_code.execute_command.call_count == 3  # Original + 2 retries
    
    @pytest.mark.asyncio
    async def test_batch_execute_tasks(self, sample_tasks, sample_ai_context, mock_claude_code):
        """Test batch task execution."""
        ai_interface = AIInterface(enable_validation=False)
        ai_interface._claude_code = mock_claude_code
        
        # Mock different response for each task
        mock_responses = [
            {
                'success': True,
                'output': f'Response for task {i}',
                'files_modified': [f'src/file_{i}.py'],
                'execution_time': 1.0 + i * 0.5
            }
            for i in range(len(sample_tasks))
        ]
        mock_claude_code.execute_command.side_effect = mock_responses
        
        results = await ai_interface.batch_execute_tasks(sample_tasks, sample_ai_context)
        
        assert len(results) == len(sample_tasks)
        assert all(result.success for result in results)
        assert all(isinstance(result, AITaskResult) for result in results)
        
        # Verify each task was processed
        assert mock_claude_code.execute_command.call_count == len(sample_tasks)
    
    @pytest.mark.asyncio
    async def test_batch_execute_with_mixed_results(self, sample_tasks, sample_ai_context):
        """Test batch execution with some successes and failures."""
        ai_interface = AIInterface(enable_validation=False)
        
        # Mock mixed responses
        mock_claude_code = AsyncMock()
        mock_responses = [
            {  # Task 0: Success
                'success': True,
                'output': 'Successful response',
                'files_modified': ['src/api.py'],
                'execution_time': 1.0
            },
            Exception("Task 1 failed"),  # Task 1: Failure
            {  # Task 2: Success
                'success': True,
                'output': 'Another successful response',
                'files_modified': ['tests/test_main.py'],
                'execution_time': 0.8
            }
        ]
        mock_claude_code.execute_command.side_effect = mock_responses
        ai_interface._claude_code = mock_claude_code
        
        results = await ai_interface.batch_execute_tasks(sample_tasks, sample_ai_context)
        
        assert len(results) == len(sample_tasks)
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True
        assert "Task 1 failed" in results[1].error_message
    
    @pytest.mark.asyncio
    async def test_response_caching(self, sample_tasks, sample_ai_context, mock_claude_code):
        """Test response caching functionality."""
        ai_interface = AIInterface(enable_validation=False, enable_caching=True)
        ai_interface._claude_code = mock_claude_code
        
        task = sample_tasks[0]
        
        # First execution - should call Claude Code
        result1 = await ai_interface.execute_task(task, sample_ai_context)
        
        # Second execution with same task and context - should use cache
        result2 = await ai_interface.execute_task(task, sample_ai_context)
        
        assert result1.success is True
        assert result2.success is True
        assert result2.metadata.get('from_cache') is True
        
        # Verify Claude Code was only called once
        assert mock_claude_code.execute_command.call_count == 1
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, sample_tasks, sample_ai_context, mock_claude_code):
        """Test performance monitoring and metrics collection."""
        ai_interface = AIInterface(enable_validation=False)
        ai_interface._claude_code = mock_claude_code
        
        # Execute multiple tasks to generate metrics
        for task in sample_tasks:
            await ai_interface.execute_task(task, sample_ai_context)
        
        # Get performance metrics
        metrics = ai_interface.get_performance_metrics()
        
        assert metrics is not None
        assert metrics['total_tasks_executed'] == len(sample_tasks)
        assert metrics['average_execution_time'] > 0
        assert 'success_rate' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'complexity_distribution' in metrics
    
    @pytest.mark.asyncio
    async def test_context_optimization(self, sample_tasks, sample_ai_context):
        """Test AI context optimization for better performance."""
        ai_interface = AIInterface(enable_validation=False)
        
        # Mock context optimizer
        with patch.object(ai_interface, '_optimize_context') as mock_optimize:
            optimized_context = sample_ai_context.copy()
            optimized_context['optimized'] = True
            optimized_context['relevant_files'] = ['main.py']  # Reduced file list
            mock_optimize.return_value = optimized_context
            
            mock_claude_code = AsyncMock()
            mock_claude_code.execute_command.return_value = {
                'success': True,
                'output': 'Optimized response',
                'files_modified': ['src/main.py'],
                'execution_time': 0.8  # Faster due to optimization
            }
            ai_interface._claude_code = mock_claude_code
            
            task = sample_tasks[0]
            
            result = await ai_interface.execute_task(task, sample_ai_context)
        
        assert result.success is True
        assert result.execution_time < 1.0  # Should be faster with optimization
        
        # Verify context was optimized
        mock_optimize.assert_called_once_with(sample_ai_context, task)
    
    @pytest.mark.asyncio
    async def test_task_complexity_analysis(self, sample_tasks):
        """Test task complexity analysis."""
        ai_interface = AIInterface()
        
        # Test different complexity levels
        simple_task = sample_tasks[2]  # "Write tests" - should be simple
        complex_task = sample_tasks[1]  # "Add authentication" - should be complex
        
        simple_analysis = ai_interface._task_complexity_analyzer.analyze_complexity(
            simple_task, {'project_type': 'python'}
        )
        complex_analysis = ai_interface._task_complexity_analyzer.analyze_complexity(
            complex_task, {'project_type': 'python'}
        )
        
        assert simple_analysis['complexity_score'] < complex_analysis['complexity_score']
        assert simple_analysis['recommendation'] == 'use_claude_code'
        assert complex_analysis['recommendation'] in ['use_flow', 'use_hybrid']
    
    @pytest.mark.asyncio
    async def test_error_handling_edge_cases(self, sample_tasks, sample_ai_context):
        """Test error handling for various edge cases."""
        ai_interface = AIInterface(enable_validation=False)
        
        # Test with invalid task
        invalid_task = Task(
            id=uuid4(),
            name="",  # Empty name
            description="",  # Empty description
            priority=Priority.LOW
        )
        
        result = await ai_interface.execute_task(invalid_task, sample_ai_context)
        
        assert result.success is False
        assert "Invalid task" in result.error_message
        
        # Test with invalid context
        invalid_context = {}  # Empty context
        
        result = await ai_interface.execute_task(sample_tasks[0], invalid_context)
        
        assert result.success is False
        assert "Invalid context" in result.error_message or result.success is True  # Fallback handling
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, sample_ai_context):
        """Test concurrent execution of multiple tasks."""
        ai_interface = AIInterface(enable_validation=False, max_concurrent_tasks=3)
        
        # Create many independent tasks
        tasks = []
        for i in range(10):
            task = Task(
                id=uuid4(),
                name=f"Task {i}",
                description=f"Concurrent task {i}",
                priority=Priority.MEDIUM
            )
            tasks.append(task)
        
        # Mock Claude Code with varying response times
        mock_claude_code = AsyncMock()
        async def mock_execute_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate work
            return {
                'success': True,
                'output': 'Concurrent response',
                'files_modified': ['src/file.py'],
                'execution_time': 0.1
            }
        
        mock_claude_code.execute_command = mock_execute_with_delay
        ai_interface._claude_code = mock_claude_code
        
        start_time = asyncio.get_event_loop().time()
        results = await ai_interface.batch_execute_tasks(tasks, sample_ai_context)
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == len(tasks)
        assert all(result.success for result in results)
        
        # Should complete faster than sequential execution due to concurrency
        # Sequential would take ~1 second (10 * 0.1), concurrent should be much faster
        execution_time = end_time - start_time
        assert execution_time < 0.8  # Should benefit from concurrency
    
    def test_cache_key_generation(self, sample_tasks, sample_ai_context):
        """Test cache key generation for consistent caching."""
        ai_interface = AIInterface()
        
        task = sample_tasks[0]
        
        # Generate cache key multiple times - should be consistent
        key1 = ai_interface._generate_cache_key(task, sample_ai_context)
        key2 = ai_interface._generate_cache_key(task, sample_ai_context)
        
        assert key1 == key2
        
        # Different task should generate different key
        different_task = sample_tasks[1]
        key3 = ai_interface._generate_cache_key(different_task, sample_ai_context)
        
        assert key1 != key3
        
        # Different context should generate different key
        different_context = sample_ai_context.copy()
        different_context['project_type'] = 'javascript'
        key4 = ai_interface._generate_cache_key(task, different_context)
        
        assert key1 != key4
    
    @pytest.mark.asyncio
    async def test_cleanup_and_resource_management(self, sample_tasks, sample_ai_context):
        """Test cleanup and resource management."""
        ai_interface = AIInterface(enable_validation=False)
        
        # Execute some tasks to create resources
        mock_claude_code = AsyncMock()
        mock_claude_code.execute_command.return_value = {
            'success': True,
            'output': 'Response',
            'files_modified': ['src/main.py'],
            'execution_time': 1.0
        }
        ai_interface._claude_code = mock_claude_code
        
        for task in sample_tasks:
            await ai_interface.execute_task(task, sample_ai_context)
        
        # Verify resources were created
        assert len(ai_interface._performance_metrics) > 0
        if hasattr(ai_interface, '_response_cache'):
            assert len(ai_interface._response_cache) >= 0
        
        # Test cleanup
        await ai_interface.cleanup()
        
        # Verify cleanup was performed
        metrics = ai_interface.get_performance_metrics()
        assert metrics['total_tasks_executed'] == len(sample_tasks)  # Metrics preserved
    
    @pytest.mark.asyncio
    async def test_ai_interface_configuration(self):
        """Test AIInterface configuration options."""
        # Test with custom configuration
        config = {
            'max_retries': 5,
            'retry_delay': 2.0,
            'max_concurrent_tasks': 10,
            'enable_caching': False,
            'cache_ttl': 3600,
            'complexity_threshold': 7.0
        }
        
        ai_interface = AIInterface(
            enable_validation=True,
            **config
        )
        
        assert ai_interface.max_retries == 5
        assert ai_interface.retry_delay == 2.0
        assert ai_interface.max_concurrent_tasks == 10
        assert ai_interface.enable_caching is False
        assert ai_interface.complexity_threshold == 7.0


class TestTaskComplexityAnalyzer:
    """Tests for TaskComplexityAnalyzer helper class."""
    
    def test_complexity_scoring_algorithm(self):
        """Test complexity scoring algorithm."""
        from integrations.ai_interface import TaskComplexityAnalyzer
        
        analyzer = TaskComplexityAnalyzer()
        
        # Test simple task
        simple_task = Task(
            id=uuid4(),
            name="Print hello world",
            description="Create a simple hello world function",
            priority=Priority.LOW,
            estimated_duration=15
        )
        
        simple_context = {'project_type': 'python', 'framework': 'basic'}
        simple_score = analyzer.analyze_complexity(simple_task, simple_context)
        
        # Test complex task
        complex_task = Task(
            id=uuid4(),
            name="Implement distributed authentication system",
            description="Create a scalable, secure authentication system with JWT, OAuth2, rate limiting, and audit logging",
            priority=Priority.CRITICAL,
            estimated_duration=480,
            metadata={
                'security_critical': True,
                'requires_external_services': True,
                'performance_critical': True
            }
        )
        
        complex_context = {
            'project_type': 'python',
            'framework': 'fastapi',
            'existing_complexity': 'high',
            'constraints': {'security_level': 'strict'}
        }
        complex_score = analyzer.analyze_complexity(complex_task, complex_context)
        
        # Complex task should have higher score
        assert complex_score['complexity_score'] > simple_score['complexity_score']
        assert simple_score['recommendation'] == 'use_claude_code'
        assert complex_score['recommendation'] in ['use_flow', 'use_hybrid']
    
    def test_complexity_factors(self):
        """Test individual complexity factors."""
        from integrations.ai_interface import TaskComplexityAnalyzer
        
        analyzer = TaskComplexityAnalyzer()
        
        # Test keyword-based complexity
        security_task = Task(
            id=uuid4(),
            name="Implement authentication",
            description="Add secure login with encryption and audit",
            priority=Priority.HIGH
        )
        
        base_task = Task(
            id=uuid4(),
            name="Add button",
            description="Add a simple button to the UI",
            priority=Priority.LOW
        )
        
        context = {'project_type': 'python'}
        
        security_analysis = analyzer.analyze_complexity(security_task, context)
        base_analysis = analyzer.analyze_complexity(base_task, context)
        
        # Security task should be more complex due to keywords
        assert security_analysis['complexity_score'] > base_analysis['complexity_score']
    
    def test_recommendation_logic(self):
        """Test recommendation logic based on complexity."""
        from integrations.ai_interface import TaskComplexityAnalyzer
        
        analyzer = TaskComplexityAnalyzer()
        
        # Create tasks with controlled complexity
        low_complexity_task = Task(
            id=uuid4(),
            name="Simple task",
            description="A simple task",
            priority=Priority.LOW,
            estimated_duration=30
        )
        
        high_complexity_task = Task(
            id=uuid4(),
            name="Complex distributed microservice architecture",
            description="Design and implement a complex system with multiple services",
            priority=Priority.CRITICAL,
            estimated_duration=600,
            metadata={'requires_coordination': True, 'multiple_domains': True}
        )
        
        context = {'project_type': 'python', 'framework': 'fastapi'}
        
        low_analysis = analyzer.analyze_complexity(low_complexity_task, context)
        high_analysis = analyzer.analyze_complexity(high_complexity_task, context)
        
        assert low_analysis['recommendation'] == 'use_claude_code'
        assert high_analysis['recommendation'] in ['use_flow', 'use_hybrid']
        assert high_analysis['estimated_agents'] > low_analysis['estimated_agents']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
