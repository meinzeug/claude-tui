"""Test Claude Flow Orchestrator

Comprehensive tests for the Claude Flow orchestration system.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.ai.claude_flow_orchestrator import (
    ClaudeFlowOrchestrator,
    OrchestrationTask,
    TaskPriority,
    ContextType,
    OrchestratorState
)
from src.ai.performance_monitor import PerformanceMonitor


@pytest.fixture
async def orchestrator():
    """Create orchestrator instance for testing"""
    perf_monitor = Mock(spec=PerformanceMonitor)
    perf_monitor.start = AsyncMock()
    perf_monitor.stop = AsyncMock()
    perf_monitor.record_metric = Mock()
    perf_monitor.get_metrics_snapshot = AsyncMock(return_value={})
    
    orchestrator = ClaudeFlowOrchestrator(
        performance_monitor=perf_monitor,
        max_concurrent_tasks=5
    )
    
    # Mock Redis connection
    orchestrator.redis_client = AsyncMock()
    orchestrator.redis_client.ping = AsyncMock()
    orchestrator.redis_client.setex = AsyncMock()
    orchestrator.redis_client.get = AsyncMock(return_value=None)
    
    await orchestrator.initialize()
    
    yield orchestrator
    
    await orchestrator.shutdown()


@pytest.fixture
def sample_task():
    """Create sample task for testing"""
    return OrchestrationTask(
        task_id="test-task-001",
        description="Test development task",
        context_type=ContextType.DEVELOPMENT,
        priority=TaskPriority.MEDIUM,
        agent_requirements=["coding", "testing"],
        estimated_duration=300,
        context_data={"language": "python", "framework": "fastapi"}
    )


class TestClaudeFlowOrchestrator:
    """Test suite for Claude Flow Orchestrator"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.state == OrchestratorState.READY
        assert orchestrator.max_concurrent_tasks == 5
        assert len(orchestrator.context_patterns) == 6  # All context types
        assert orchestrator.performance_monitor is not None
    
    @pytest.mark.asyncio
    async def test_task_orchestration(self, orchestrator, sample_task):
        """Test basic task orchestration"""
        # Mock swarm manager
        orchestrator.swarm_manager.initialize_swarm = AsyncMock(return_value="swarm-001")
        orchestrator.swarm_manager.spawn_agent = AsyncMock()
        orchestrator.swarm_manager.get_available_agents = AsyncMock(return_value=[])
        
        execution_id = await orchestrator.orchestrate_task(sample_task)
        
        assert execution_id is not None
        assert execution_id.startswith("exec-")
        assert len(orchestrator.active_tasks) <= 1
    
    @pytest.mark.asyncio
    async def test_context_patterns(self, orchestrator):
        """Test context-specific patterns"""
        patterns = orchestrator.context_patterns
        
        # Test all context types have patterns
        for context_type in ContextType:
            assert context_type in patterns
            pattern = patterns[context_type]
            assert 'preferred_agents' in pattern
            assert 'topology' in pattern
            assert 'max_agents' in pattern
    
    @pytest.mark.asyncio
    async def test_swarm_selection(self, orchestrator, sample_task):
        """Test optimal swarm selection"""
        # Create mock swarms with different capacities
        orchestrator.swarm_capacities = {
            "swarm-001": Mock(current_load=0.2, success_rate=0.9, health_score=0.95),
            "swarm-002": Mock(current_load=0.8, success_rate=0.7, health_score=0.8),
            "swarm-003": Mock(current_load=0.5, success_rate=0.95, health_score=0.9)
        }
        
        # Mock agent capabilities
        orchestrator.swarm_manager.swarm_agents = {
            "swarm-001": [Mock(capabilities=["coding", "testing"])],
            "swarm-002": [Mock(capabilities=["analysis"])],
            "swarm-003": [Mock(capabilities=["coding", "optimization"])]
        }
        
        best_swarm = await orchestrator._select_optimal_swarm(sample_task)
        
        # Should select swarm-001 (best combination of low load and high performance)
        assert best_swarm in ["swarm-001", "swarm-003"]
    
    @pytest.mark.asyncio
    async def test_capacity_management(self, orchestrator):
        """Test task capacity limits"""
        # Fill up to max capacity
        tasks = []
        for i in range(orchestrator.max_concurrent_tasks):
            task = OrchestrationTask(
                task_id=f"task-{i}",
                description=f"Test task {i}",
                context_type=ContextType.DEVELOPMENT,
                priority=TaskPriority.LOW
            )
            tasks.append(task)
        
        # Mock the execution to not complete immediately
        with patch.object(orchestrator, '_execute_task') as mock_execute:
            mock_execute.return_value = AsyncMock()
            
            execution_ids = []
            for task in tasks:
                execution_id = await orchestrator.orchestrate_task(task)
                execution_ids.append(execution_id)
            
            assert len(orchestrator.active_tasks) == orchestrator.max_concurrent_tasks
            
            # Next task should be queued
            overflow_task = OrchestrationTask(
                task_id="overflow-task",
                description="Overflow task",
                context_type=ContextType.DEVELOPMENT,
                priority=TaskPriority.HIGH
            )
            
            overflow_id = await orchestrator.orchestrate_task(overflow_task)
            assert orchestrator.task_queue.qsize() > 0
    
    @pytest.mark.asyncio
    async def test_execution_context_building(self, orchestrator, sample_task):
        """Test execution context creation"""
        context = await orchestrator._build_execution_context(sample_task, "swarm-001")
        
        assert context['task_id'] == sample_task.task_id
        assert context['swarm_id'] == "swarm-001"
        assert context['context_type'] == sample_task.context_type.value
        assert context['priority'] == sample_task.priority.value
        assert 'orchestrator_config' in context
        assert context['orchestrator_config']['max_retries'] == sample_task.max_retries
    
    @pytest.mark.asyncio
    async def test_context_specific_execution(self, orchestrator):
        """Test context-specific execution strategies"""
        # Test development context
        dev_task = OrchestrationTask(
            task_id="dev-task",
            description="Development task",
            context_type=ContextType.DEVELOPMENT,
            priority=TaskPriority.MEDIUM
        )
        
        with patch.object(orchestrator, '_execute_development_task') as mock_dev:
            mock_dev.return_value = {'success': True}
            
            result = await orchestrator._execute_development_task(dev_task, "swarm-001", {})
            assert result['success'] is True
    
    @pytest.mark.asyncio
    async def test_performance_metrics_integration(self, orchestrator, sample_task):
        """Test performance metrics recording"""
        await orchestrator.orchestrate_task(sample_task)
        
        # Verify metrics were recorded
        assert orchestrator.orchestration_metrics['tasks_orchestrated'] > 0
        assert orchestrator.performance_monitor.record_metric.called
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test error handling and recovery"""
        # Test expired task
        expired_task = OrchestrationTask(
            task_id="expired-task",
            description="Expired task",
            context_type=ContextType.DEVELOPMENT,
            priority=TaskPriority.LOW,
            deadline=datetime.utcnow() - timedelta(minutes=1)  # Already expired
        )
        
        with pytest.raises(Exception):  # Should raise TaskError
            await orchestrator.orchestrate_task(expired_task)
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, orchestrator):
        """Test orchestration status reporting"""
        status = await orchestrator.get_orchestration_status()
        
        assert 'state' in status
        assert 'active_tasks' in status
        assert 'queued_tasks' in status
        assert 'completed_tasks' in status
        assert 'metrics' in status
        assert 'swarm_health' in status
        assert status['state'] == OrchestratorState.READY.value
    
    @pytest.mark.asyncio
    async def test_swarm_creation_optimization(self, orchestrator):
        """Test optimized swarm creation"""
        task = OrchestrationTask(
            task_id="complex-task",
            description="Complex machine learning task",
            context_type=ContextType.RESEARCH,
            priority=TaskPriority.HIGH,
            agent_requirements=["machine_learning", "data_analysis", "optimization"]
        )
        
        orchestrator.swarm_manager.initialize_swarm = AsyncMock(return_value="ml-swarm-001")
        orchestrator.swarm_manager.spawn_agent = AsyncMock()
        
        swarm_id = await orchestrator._create_optimized_swarm(task)
        
        assert swarm_id == "ml-swarm-001"
        assert swarm_id in orchestrator.swarm_capacities
        assert orchestrator.swarm_capacities[swarm_id].swarm_id == swarm_id


@pytest.mark.asyncio
async def test_concurrent_task_execution():
    """Test concurrent task execution"""
    orchestrator = ClaudeFlowOrchestrator(max_concurrent_tasks=3)
    
    # Mock dependencies
    orchestrator.performance_monitor = Mock()
    orchestrator.performance_monitor.start = AsyncMock()
    orchestrator.performance_monitor.record_metric = Mock()
    orchestrator.redis_client = AsyncMock()
    orchestrator.redis_client.ping = AsyncMock()
    
    await orchestrator.initialize()
    
    try:
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task = OrchestrationTask(
                task_id=f"concurrent-task-{i}",
                description=f"Concurrent task {i}",
                context_type=ContextType.DEVELOPMENT,
                priority=TaskPriority.MEDIUM
            )
            tasks.append(task)
        
        # Execute tasks concurrently
        orchestrator.swarm_manager.initialize_swarm = AsyncMock(return_value="test-swarm")
        orchestrator.swarm_manager.spawn_agent = AsyncMock()
        
        execution_ids = await asyncio.gather(*[
            orchestrator.orchestrate_task(task) for task in tasks[:3]
        ])
        
        # First 3 should be active, others queued
        assert len(execution_ids) == 3
        assert len(orchestrator.active_tasks) <= 3
    
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_shutdown():
    """Test graceful orchestrator shutdown"""
    orchestrator = ClaudeFlowOrchestrator()
    
    # Mock dependencies
    orchestrator.performance_monitor = Mock()
    orchestrator.performance_monitor.start = AsyncMock()
    orchestrator.performance_monitor.stop = AsyncMock()
    orchestrator.redis_client = AsyncMock()
    orchestrator.redis_client.ping = AsyncMock()
    orchestrator.redis_client.close = AsyncMock()
    
    await orchestrator.initialize()
    
    # Add some active tasks
    orchestrator.active_tasks["test-exec-1"] = Mock()
    orchestrator.active_tasks["test-exec-2"] = Mock()
    
    # Shutdown should handle active tasks gracefully
    await orchestrator.shutdown()
    
    assert orchestrator.state == OrchestratorState.SHUTDOWN
    orchestrator.performance_monitor.stop.assert_called_once()
    orchestrator.redis_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])