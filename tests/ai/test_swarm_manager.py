"""Test Enhanced Swarm Manager

Tests for the enhanced swarm management system.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.ai.swarm_manager import (
    EnhancedSwarmManager,
    SwarmHealthStatus,
    AgentLifecycleState,
    SwarmConfiguration,
    SwarmResourceUsage,
    AgentMetrics,
    ResourceType
)
from src.integrations.claude_flow import SwarmManager as BaseSwarmManager, SwarmTopology, AgentType


@pytest.fixture
async def swarm_manager():
    """Create enhanced swarm manager for testing"""
    base_manager = Mock(spec=BaseSwarmManager)
    base_manager.initialize = AsyncMock()
    base_manager.initialize_swarm = AsyncMock(return_value="swarm-001")
    base_manager.spawn_agent = AsyncMock(return_value=Mock(id="agent-001"))
    base_manager.terminate_agent = AsyncMock()
    base_manager.get_swarm_status = AsyncMock(return_value={'status': 'active'})
    base_manager.shutdown = AsyncMock()
    
    perf_monitor = Mock()
    perf_monitor.start = AsyncMock()
    perf_monitor.stop = AsyncMock()
    perf_monitor.record_metric = Mock()
    
    manager = EnhancedSwarmManager(
        base_manager=base_manager,
        performance_monitor=perf_monitor,
        max_swarms=5,
        resource_monitoring_enabled=True
    )
    
    await manager.initialize()
    
    yield manager
    
    await manager.shutdown()


@pytest.fixture
def project_context():
    """Sample project context for testing"""
    return {
        'primary_domain': 'web_development',
        'features': ['authentication', 'api', 'database'],
        'integrations': ['redis', 'postgresql'],
        'timeline_days': 14,
        'team_size': 3,
        'estimated_files': 25
    }


class TestEnhancedSwarmManager:
    """Test suite for Enhanced Swarm Manager"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, swarm_manager):
        """Test swarm manager initialization"""
        assert swarm_manager.base_manager is not None
        assert swarm_manager.performance_monitor is not None
        assert swarm_manager.max_swarms == 5
        assert swarm_manager.resource_monitoring_enabled is True
        
        # Background tasks should be running
        assert swarm_manager.health_monitor_task is not None
        assert swarm_manager.resource_monitor_task is not None
        assert swarm_manager.optimization_task is not None
        assert swarm_manager.cleanup_task is not None
    
    @pytest.mark.asyncio
    async def test_optimized_swarm_creation(self, swarm_manager, project_context):
        """Test optimized swarm creation based on project context"""
        swarm_id = await swarm_manager.create_optimized_swarm(project_context)
        
        assert swarm_id is not None
        assert swarm_id in swarm_manager.swarm_configurations
        assert swarm_id in swarm_manager.swarm_health
        assert swarm_id in swarm_manager.resource_usage
        
        config = swarm_manager.swarm_configurations[swarm_id]
        assert isinstance(config, SwarmConfiguration)
        assert config.topology in [SwarmTopology.HIERARCHICAL, SwarmTopology.MESH, SwarmTopology.STAR]
        assert config.specialization_focus == 'web_development'
    
    @pytest.mark.asyncio
    async def test_complexity_calculation(self, swarm_manager, project_context):
        """Test project complexity calculation"""
        complexity = swarm_manager._calculate_project_complexity(project_context)
        
        # Should be a reasonable complexity score
        assert 10 <= complexity <= 100
        
        # More features should increase complexity
        complex_context = project_context.copy()
        complex_context['features'] = ['feature' + str(i) for i in range(10)]
        complex_context['integrations'] = ['service' + str(i) for i in range(5)]
        
        higher_complexity = swarm_manager._calculate_project_complexity(complex_context)
        assert higher_complexity > complexity
    
    @pytest.mark.asyncio
    async def test_topology_selection(self, swarm_manager):
        """Test optimal topology selection"""
        # Low complexity should prefer STAR
        low_complexity = 15
        topology = swarm_manager._select_optimal_topology(low_complexity)
        assert topology == SwarmTopology.STAR
        
        # Medium complexity should prefer HIERARCHICAL
        medium_complexity = 45
        topology = swarm_manager._select_optimal_topology(medium_complexity)
        assert topology == SwarmTopology.HIERARCHICAL
        
        # High complexity should prefer MESH
        high_complexity = 75
        topology = swarm_manager._select_optimal_topology(high_complexity)
        assert topology == SwarmTopology.MESH
    
    @pytest.mark.asyncio
    async def test_resource_requirements(self, swarm_manager, project_context):
        """Test resource requirements calculation"""
        complexity = 50
        resources = swarm_manager._calculate_resource_requirements(project_context, complexity)
        
        assert ResourceType.CPU in resources
        assert ResourceType.MEMORY in resources
        assert ResourceType.NETWORK in resources
        assert ResourceType.STORAGE in resources
        
        # Machine learning domain should require more resources
        ml_context = project_context.copy()
        ml_context['primary_domain'] = 'machine_learning'
        ml_resources = swarm_manager._calculate_resource_requirements(ml_context, complexity)
        
        assert ml_resources[ResourceType.CPU] > resources[ResourceType.CPU]
        assert ml_resources[ResourceType.MEMORY] > resources[ResourceType.MEMORY]
    
    @pytest.mark.asyncio
    async def test_agent_types_determination(self, swarm_manager, project_context):
        """Test required agent types determination"""
        agent_types = swarm_manager._determine_required_agent_types(project_context)
        
        assert AgentType.COORDINATOR in agent_types
        assert len(agent_types) >= 2  # Should have multiple types
        
        # Web development should include relevant agent types
        assert AgentType.CODER in agent_types or AgentType.SYSTEM_ARCHITECT in agent_types
    
    @pytest.mark.asyncio
    async def test_intelligent_agent_spawning(self, swarm_manager):
        """Test intelligent agent spawning"""
        # Create a swarm first
        swarm_config = SwarmConfiguration(
            swarm_id="test-swarm",
            topology=SwarmTopology.HIERARCHICAL,
            max_agents=5
        )
        swarm_manager.swarm_configurations["test-swarm"] = swarm_config
        swarm_manager.swarm_health["test-swarm"] = SwarmHealthStatus.HEALTHY
        swarm_manager.resource_usage["test-swarm"] = SwarmResourceUsage(
            swarm_id="test-swarm",
            cpu_limit=100.0,
            memory_limit=1024.0
        )
        
        # Spawn agent
        agent_id = await swarm_manager.spawn_intelligent_agent(
            "test-swarm",
            AgentType.CODER,
            context={'specialization': 'python'},
            custom_capabilities=['python', 'fastapi', 'testing']
        )
        
        assert agent_id is not None
        assert agent_id in swarm_manager.agent_metrics
        assert agent_id in swarm_manager.agent_states
        assert agent_id in swarm_manager.agent_task_queues
        
        # Check initial state
        assert swarm_manager.agent_states[agent_id] == AgentLifecycleState.SPAWNING
    
    @pytest.mark.asyncio
    async def test_resource_availability_check(self, swarm_manager):
        """Test resource availability checking"""
        # Create swarm with limited resources
        swarm_id = "resource-test-swarm"
        swarm_manager.resource_usage[swarm_id] = SwarmResourceUsage(
            swarm_id=swarm_id,
            cpu_limit=50.0,
            memory_limit=512.0,
            current_cpu=40.0,  # Already using 80%
            current_memory=400.0  # Already using ~78%
        )
        
        # Should reject new agent due to high CPU usage
        available = await swarm_manager._check_resource_availability(swarm_id, AgentType.CODER)
        assert available is False
        
        # Reduce current usage
        swarm_manager.resource_usage[swarm_id].current_cpu = 10.0
        swarm_manager.resource_usage[swarm_id].current_memory = 100.0
        
        available = await swarm_manager._check_resource_availability(swarm_id, AgentType.CODER)
        assert available is True
    
    @pytest.mark.asyncio
    async def test_task_assignment(self, swarm_manager):
        """Test intelligent task assignment"""
        # Setup swarm and agents
        swarm_id = "task-test-swarm"
        swarm_manager.swarm_configurations[swarm_id] = SwarmConfiguration(
            swarm_id=swarm_id,
            topology=SwarmTopology.STAR
        )
        
        # Create mock agents with different capabilities
        agent1_id = "agent-python-001"
        agent2_id = "agent-js-002"
        
        swarm_manager.agent_metrics[agent1_id] = AgentMetrics(
            agent_id=agent1_id,
            spawn_time=datetime.utcnow()
        )
        swarm_manager.agent_metrics[agent2_id] = AgentMetrics(
            agent_id=agent2_id,
            spawn_time=datetime.utcnow()
        )
        
        swarm_manager.agent_states[agent1_id] = AgentLifecycleState.READY
        swarm_manager.agent_states[agent2_id] = AgentLifecycleState.READY
        
        # Mock agent capabilities
        with patch.object(swarm_manager, '_get_available_agents') as mock_available:
            mock_available.return_value = [agent1_id, agent2_id]
            
            with patch.object(swarm_manager, '_calculate_agent_task_suitability') as mock_suitability:
                # Agent1 is better for Python tasks
                mock_suitability.side_effect = lambda aid, desc, reqs: 0.9 if aid == agent1_id else 0.4
                
                assigned_agent = await swarm_manager.assign_task_to_best_agent(
                    swarm_id,
                    "Develop Python API endpoints",
                    ["python", "api", "coding"]
                )
                
                assert assigned_agent == agent1_id
    
    @pytest.mark.asyncio
    async def test_swarm_health_monitoring(self, swarm_manager):
        """Test swarm health monitoring"""
        swarm_id = "health-test-swarm"
        
        # Setup swarm with metrics
        swarm_manager.swarm_configurations[swarm_id] = SwarmConfiguration(
            swarm_id=swarm_id,
            topology=SwarmTopology.MESH
        )
        
        # Create agent metrics with poor performance
        agent_id = "unhealthy-agent"
        metrics = AgentMetrics(agent_id=agent_id, spawn_time=datetime.utcnow())
        metrics.tasks_completed = 2
        metrics.tasks_failed = 8  # 20% success rate
        metrics.health_score = 0.3
        
        swarm_manager.agent_metrics[agent_id] = metrics
        swarm_manager.agent_states[agent_id] = AgentLifecycleState.READY
        
        # Mock resource usage
        swarm_manager.resource_usage[swarm_id] = SwarmResourceUsage(
            swarm_id=swarm_id,
            current_cpu=10.0,
            current_memory=100.0
        )
        
        # Monitor health
        await swarm_manager._monitor_swarm_health(swarm_id)
        
        # Health score should reflect poor agent performance
        overall_health = swarm_manager._calculate_overall_health_score(swarm_id)
        assert overall_health < 0.7  # Should be considered unhealthy
    
    @pytest.mark.asyncio
    async def test_swarm_scaling(self, swarm_manager):
        """Test intelligent swarm scaling"""
        swarm_id = "scaling-test-swarm"
        
        # Setup swarm configuration
        config = SwarmConfiguration(
            swarm_id=swarm_id,
            topology=SwarmTopology.HIERARCHICAL,
            max_agents=8,
            min_agents=2,
            auto_scaling=True
        )
        swarm_manager.swarm_configurations[swarm_id] = config
        swarm_manager.swarm_health[swarm_id] = SwarmHealthStatus.HEALTHY
        
        # Mock current agent count
        with patch.object(swarm_manager, '_get_agent_count') as mock_count:
            mock_count.return_value = 3
            
            # Mock high load metrics (should trigger scale up)
            with patch.object(swarm_manager, '_analyze_swarm_load') as mock_load:
                mock_load.return_value = {
                    'utilization': 0.85,  # High utilization
                    'queue_depth': 7  # High queue
                }
                
                with patch.object(swarm_manager, '_analyze_swarm_performance') as mock_perf:
                    mock_perf.return_value = 0.8  # Good performance
                    
                    with patch.object(swarm_manager, '_determine_scaling_agent_types') as mock_types:
                        mock_types.return_value = [AgentType.CODER, AgentType.OPTIMIZER]
                        
                        scaled = await swarm_manager.scale_swarm_intelligently(swarm_id)
                        
                        assert scaled is True
                        # Should spawn new agents
                        assert swarm_manager.base_manager.spawn_agent.called
    
    @pytest.mark.asyncio
    async def test_graceful_agent_termination(self, swarm_manager):
        """Test graceful agent termination"""
        agent_id = "termination-test-agent"
        
        # Setup agent
        swarm_manager.agent_states[agent_id] = AgentLifecycleState.READY
        swarm_manager.agent_task_queues[agent_id] = asyncio.Queue()
        
        # Add a task to the queue
        await swarm_manager.agent_task_queues[agent_id].put({"task": "test"})
        
        # Terminate agent
        success = await swarm_manager.terminate_agent_gracefully(agent_id)
        
        assert success is True
        assert swarm_manager.agent_states[agent_id] == AgentLifecycleState.TERMINATED
        assert agent_id in swarm_manager.termination_queue
    
    @pytest.mark.asyncio
    async def test_health_report_generation(self, swarm_manager):
        """Test comprehensive health report generation"""
        swarm_id = "report-test-swarm"
        
        # Setup swarm with complete data
        config = SwarmConfiguration(
            swarm_id=swarm_id,
            topology=SwarmTopology.MESH,
            max_agents=5,
            specialization_focus="data_science"
        )
        
        swarm_manager.swarm_configurations[swarm_id] = config
        swarm_manager.swarm_health[swarm_id] = SwarmHealthStatus.HEALTHY
        swarm_manager.resource_usage[swarm_id] = SwarmResourceUsage(
            swarm_id=swarm_id,
            cpu_limit=100.0,
            memory_limit=1024.0,
            current_cpu=45.0,
            current_memory=512.0,
            peak_cpu=78.0,
            peak_memory=768.0
        )
        
        # Add agent metrics
        agent_id = "report-agent-001"
        metrics = AgentMetrics(agent_id=agent_id, spawn_time=datetime.utcnow() - timedelta(hours=2))
        metrics.tasks_completed = 15
        metrics.tasks_failed = 2
        metrics.health_score = 0.9
        metrics.error_count = 1
        
        swarm_manager.agent_metrics[agent_id] = metrics
        swarm_manager.agent_states[agent_id] = AgentLifecycleState.READY
        
        # Mock agent retrieval
        with patch.object(swarm_manager, '_get_swarm_agent_ids') as mock_agents:
            mock_agents.return_value = [agent_id]
            
            report = await swarm_manager.get_swarm_health_report(swarm_id)
            
            assert report['swarm_id'] == swarm_id
            assert report['health_status'] == SwarmHealthStatus.HEALTHY.value
            assert 'overall_health_score' in report
            assert 'configuration' in report
            assert 'resource_usage' in report
            assert 'agent_health' in report
            assert agent_id in report['agent_health']
            
            # Check agent health details
            agent_health = report['agent_health'][agent_id]
            assert agent_health['health_score'] == 0.9
            assert agent_health['success_rate'] > 0.8  # Should be high
            assert agent_health['uptime_hours'] > 1  # Should show uptime


@pytest.mark.asyncio
async def test_background_tasks():
    """Test background task execution"""
    manager = EnhancedSwarmManager(resource_monitoring_enabled=True)
    
    # Mock dependencies
    manager.base_manager = Mock()
    manager.base_manager.initialize = AsyncMock()
    manager.performance_monitor = Mock()
    manager.performance_monitor.start = AsyncMock()
    
    await manager.initialize()
    
    # Check that background tasks are created
    assert manager.health_monitor_task is not None
    assert manager.resource_monitor_task is not None
    assert manager.optimization_task is not None
    assert manager.cleanup_task is not None
    
    # All tasks should be running
    assert not manager.health_monitor_task.done()
    assert not manager.resource_monitor_task.done()
    assert not manager.optimization_task.done()
    assert not manager.cleanup_task.done()
    
    await manager.shutdown()


@pytest.mark.asyncio
async def test_error_recovery():
    """Test error handling and recovery"""
    manager = EnhancedSwarmManager()
    
    # Mock failing base manager
    manager.base_manager = Mock()
    manager.base_manager.initialize = AsyncMock(side_effect=Exception("Initialization failed"))
    
    with pytest.raises(Exception):
        await manager.initialize()
    
    # Manager should handle the failure gracefully
    assert manager.base_manager is not None


if __name__ == "__main__":
    pytest.main([__file__])