"""
TDD London School Tests for SwarmOrchestrator

London School (mockist) approach emphasizing:
- Mock-driven development for complete isolation
- Behavior verification focusing on object interactions
- Contract testing for collaborator relationships
- Outside-in testing flow from user scenarios to implementation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid

from src.ai.swarm_orchestrator import (
    SwarmOrchestrator,
    SwarmState,
    LoadBalancingStrategy,
    SwarmMetrics,
    TaskRequest
)
from src.integrations.claude_flow import (
    SwarmManager, SwarmConfig, SwarmTopology, AgentType,
    WorkflowStatus, ClaudeFlowOrchestrator
)
from src.core.exceptions import SwarmError, AgentError, OrchestrationError


# Mock Contracts - London School emphasis on defining collaborator interfaces
@pytest.fixture
def mock_swarm_manager():
    """Mock SwarmManager with complete interface contract"""
    mock = AsyncMock(spec=SwarmManager)
    mock.initialize_swarm = AsyncMock(return_value="swarm-123")
    mock.spawn_agent = AsyncMock(return_value="agent-456")
    mock.get_swarm_status = AsyncMock(return_value={
        'id': 'swarm-123',
        'topology': 'mesh',
        'agents': 3,
        'status': 'active'
    })
    mock.swarm_agents = {'swarm-123': []}
    return mock


@pytest.fixture
def mock_claude_flow_orchestrator():
    """Mock ClaudeFlowOrchestrator for workflow execution"""
    mock = AsyncMock(spec=ClaudeFlowOrchestrator)
    
    # Mock successful workflow result
    workflow_result = Mock()
    workflow_result.is_success = True
    workflow_result.status = "completed"
    workflow_result.execution_time = 45.2
    
    mock.orchestrate_development_workflow = AsyncMock(return_value=workflow_result)
    return mock


@pytest.fixture
def swarm_orchestrator_with_mocks():
    """SwarmOrchestrator with mocked dependencies for isolated testing"""
    
    orchestrator = SwarmOrchestrator(
        max_swarms=3,
        max_agents_per_swarm=5,
        enable_auto_scaling=True,
        monitoring_interval=10
    )
    
    # Replace real collaborators with mocks
    orchestrator.swarm_manager = AsyncMock(spec=SwarmManager)
    orchestrator.orchestrator = AsyncMock(spec=ClaudeFlowOrchestrator)
    
    # Setup common mock behaviors
    orchestrator.swarm_manager.initialize_swarm = AsyncMock(return_value="swarm-test")
    orchestrator.swarm_manager.spawn_agent = AsyncMock()
    orchestrator.swarm_manager.get_swarm_status = AsyncMock(return_value={
        'id': 'swarm-test',
        'topology': 'mesh', 
        'agents': 2,
        'status': 'active'
    })
    orchestrator.swarm_manager.swarm_agents = {'swarm-test': []}
    
    # Mock successful workflow execution
    workflow_result = Mock()
    workflow_result.is_success = True
    workflow_result.status = "completed"
    workflow_result.execution_time = 30.0
    orchestrator.orchestrator.orchestrate_development_workflow = AsyncMock(return_value=workflow_result)
    
    return orchestrator


class TestSwarmOrchestrationInitialization:
    """Test SwarmOrchestrator initialization - London School contract setup"""
    
    def test_should_initialize_with_configuration_parameters(self):
        """Verify SwarmOrchestrator establishes proper initial state"""
        # Act
        orchestrator = SwarmOrchestrator(
            max_swarms=10,
            max_agents_per_swarm=15,
            enable_auto_scaling=False,
            monitoring_interval=60,
            load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED
        )
        
        # Assert - Verify configuration is properly set
        assert orchestrator.max_swarms == 10
        assert orchestrator.max_agents_per_swarm == 15
        assert orchestrator.enable_auto_scaling is False
        assert orchestrator.monitoring_interval == 60
        assert orchestrator.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED
        
        # Verify collaborators are initialized
        assert orchestrator.swarm_manager is not None
        assert orchestrator.orchestrator is not None
        
        # Verify tracking structures are initialized
        assert isinstance(orchestrator.active_swarms, dict)
        assert isinstance(orchestrator.swarm_configs, dict)
        assert isinstance(orchestrator.swarm_metrics, dict)
        assert isinstance(orchestrator.pending_tasks, dict)
        assert isinstance(orchestrator.executing_tasks, dict)
    
    def test_should_setup_proper_collaborator_relationships(self):
        """Verify proper dependency injection and collaboration setup"""
        # Act
        orchestrator = SwarmOrchestrator()
        
        # Assert - Verify collaborators are properly linked
        assert hasattr(orchestrator, 'swarm_manager')
        assert hasattr(orchestrator, 'orchestrator')
        assert orchestrator.orchestrator.swarm_manager == orchestrator.swarm_manager


class TestSwarmInitializationWorkflow:
    """Test swarm initialization workflow - London School outside-in approach"""
    
    @pytest.mark.asyncio
    async def test_should_orchestrate_complete_swarm_initialization_workflow(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test complete swarm initialization conversation between objects"""
        
        # Arrange
        project_spec = {
            'features': ['auth', 'api', 'dashboard'],
            'integrations': ['database', 'redis'],
            'estimated_files': 20,
            'requires_backend': True,
            'requires_testing': True
        }
        
        # Act
        swarm_id = await swarm_orchestrator_with_mocks.initialize_swarm(project_spec)
        
        # Assert - Verify complete initialization workflow
        
        # Step 1: Swarm manager should initialize swarm
        swarm_orchestrator_with_mocks.swarm_manager.initialize_swarm.assert_called_once()
        
        # Step 2: Initial agents should be spawned based on requirements
        spawn_calls = swarm_orchestrator_with_mocks.swarm_manager.spawn_agent.call_args_list
        assert len(spawn_calls) > 0  # At least some agents spawned
        
        # Verify swarm is tracked properly
        assert swarm_id in swarm_orchestrator_with_mocks.active_swarms
        assert swarm_orchestrator_with_mocks.active_swarms[swarm_id] == SwarmState.ACTIVE
        assert swarm_id in swarm_orchestrator_with_mocks.swarm_metrics
        
        # Verify metrics initialized
        metrics = swarm_orchestrator_with_mocks.swarm_metrics[swarm_id]
        assert metrics.swarm_id == swarm_id
        assert isinstance(metrics, SwarmMetrics)
    
    @pytest.mark.asyncio
    async def test_should_generate_optimal_configuration_based_on_complexity(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test configuration generation based on project complexity analysis"""
        
        # Arrange - High complexity project
        complex_project = {
            'features': ['auth', 'api', 'dashboard', 'analytics', 'notifications'],
            'integrations': ['database', 'redis', 'elasticsearch', 'kafka'],
            'estimated_files': 100,
            'team_size': 5,
            'timeline_days': 90
        }
        
        # Act
        with patch.object(
            swarm_orchestrator_with_mocks, 
            '_generate_optimal_config'
        ) as mock_generate_config:
            
            optimal_config = Mock(spec=SwarmConfig)
            optimal_config.topology = SwarmTopology.MESH
            optimal_config.max_agents = 8
            mock_generate_config.return_value = optimal_config
            
            await swarm_orchestrator_with_mocks.initialize_swarm(complex_project)
        
        # Assert - Verify configuration generation was called
        mock_generate_config.assert_called_once_with(complex_project)
        
        # Verify swarm manager received the generated config
        swarm_orchestrator_with_mocks.swarm_manager.initialize_swarm.assert_called_once_with(
            optimal_config
        )
    
    @pytest.mark.asyncio
    async def test_should_handle_swarm_initialization_failure_gracefully(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test error handling when swarm initialization fails"""
        
        # Arrange - Force swarm manager to fail
        swarm_orchestrator_with_mocks.swarm_manager.initialize_swarm.side_effect = Exception(
            "Swarm initialization failed"
        )
        
        # Act & Assert
        with pytest.raises(OrchestrationError, match="Swarm initialization failed"):
            await swarm_orchestrator_with_mocks.initialize_swarm({'simple': 'project'})
        
        # Verify swarm manager was called
        swarm_orchestrator_with_mocks.swarm_manager.initialize_swarm.assert_called_once()


class TestTaskExecutionOrchestration:
    """Test task execution workflow - London School behavior verification"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_complete_task_execution_workflow(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test complete task execution conversation between objects"""
        
        # Arrange
        task_request = TaskRequest(
            task_id="task-123",
            description="Implement user authentication system",
            priority="high",
            agent_requirements=["backend", "security"],
            estimated_complexity=8
        )
        
        # Initialize a swarm first
        swarm_id = "swarm-test"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        swarm_orchestrator_with_mocks.swarm_metrics[swarm_id] = SwarmMetrics(
            swarm_id=swarm_id, 
            topology="mesh"
        )
        
        # Act
        execution_id = await swarm_orchestrator_with_mocks.execute_task(
            task_request, 
            swarm_id=swarm_id
        )
        
        # Assert - Verify task is properly queued and tracked
        assert execution_id.startswith("exec-")
        assert execution_id in swarm_orchestrator_with_mocks.executing_tasks
        assert task_request.task_id in swarm_orchestrator_with_mocks.pending_tasks
        
        # Verify task tracking structure
        execution_data = swarm_orchestrator_with_mocks.executing_tasks[execution_id]
        assert execution_data['task_request'] == task_request
        assert execution_data['swarm_id'] == swarm_id
        assert execution_data['status'] == 'running'
        assert 'start_time' in execution_data
    
    @pytest.mark.asyncio
    async def test_should_select_optimal_swarm_when_none_specified(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test swarm selection algorithm based on load and capabilities"""
        
        # Arrange - Create multiple swarms with different states
        swarm1 = "swarm-busy"
        swarm2 = "swarm-optimal"
        
        swarm_orchestrator_with_mocks.active_swarms = {
            swarm1: SwarmState.ACTIVE,
            swarm2: SwarmState.ACTIVE
        }
        
        # Mock metrics to favor swarm2
        swarm_orchestrator_with_mocks.swarm_metrics = {
            swarm1: SwarmMetrics(swarm1, "mesh", utilization_rate=90),
            swarm2: SwarmMetrics(swarm2, "mesh", utilization_rate=20, efficiency_score=95)
        }
        
        # Mock agent capabilities
        swarm_orchestrator_with_mocks.swarm_manager.swarm_agents = {
            swarm1: [Mock(capabilities=["frontend"])],
            swarm2: [Mock(capabilities=["backend", "security"])]
        }
        
        task_request = TaskRequest(
            task_id="task-456",
            description="Backend security implementation",
            agent_requirements=["backend", "security"]
        )
        
        # Mock swarm selection
        with patch.object(
            swarm_orchestrator_with_mocks, 
            '_select_optimal_swarm',
            return_value=swarm2
        ) as mock_select:
            
            # Act
            execution_id = await swarm_orchestrator_with_mocks.execute_task(task_request)
        
        # Assert - Verify optimal swarm selection was called
        mock_select.assert_called_once_with(task_request)
        
        # Verify task was assigned to selected swarm
        execution_data = swarm_orchestrator_with_mocks.executing_tasks[execution_id]
        assert execution_data['swarm_id'] == swarm2
    
    @pytest.mark.asyncio 
    async def test_should_create_new_swarm_when_none_suitable_exists(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test new swarm creation when existing swarms are not suitable"""
        
        # Arrange - No existing swarms
        swarm_orchestrator_with_mocks.active_swarms = {}
        
        task_request = TaskRequest(
            task_id="task-789",
            description="Complex AI pipeline implementation",
            estimated_complexity=9
        )
        
        # Mock swarm creation
        with patch.object(
            swarm_orchestrator_with_mocks,
            'initialize_swarm',
            return_value="new-swarm-123"
        ) as mock_init_swarm:
            
            # Act
            execution_id = await swarm_orchestrator_with_mocks.execute_task(task_request)
        
        # Assert - Verify new swarm was created
        mock_init_swarm.assert_called_once()
        
        # Verify project spec was generated from task
        call_args = mock_init_swarm.call_args[0][0]
        assert 'description' in call_args
        assert call_args['description'] == task_request.description


class TestSwarmStatusAndMonitoring:
    """Test swarm status reporting and monitoring - London School contract verification"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_comprehensive_status_gathering(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test status gathering from all swarm components"""
        
        # Arrange
        swarm_id = "swarm-status-test"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        
        # Mock swarm metrics
        metrics = SwarmMetrics(
            swarm_id=swarm_id,
            topology="mesh",
            total_tasks=100,
            completed_tasks=85,
            failed_tasks=5,
            average_task_time=25.5,
            efficiency_score=92.3
        )
        swarm_orchestrator_with_mocks.swarm_metrics[swarm_id] = metrics
        
        # Mock base status from swarm manager
        base_status = {
            'id': swarm_id,
            'topology': 'mesh',
            'agents': 4,
            'active_agents': 3
        }
        swarm_orchestrator_with_mocks.swarm_manager.get_swarm_status.return_value = base_status
        
        # Mock metrics update
        with patch.object(
            swarm_orchestrator_with_mocks,
            '_update_swarm_metrics'
        ) as mock_update_metrics:
            
            # Act
            status = await swarm_orchestrator_with_mocks.get_swarm_status(swarm_id)
        
        # Assert - Verify complete status gathering workflow
        
        # Base status should be retrieved from swarm manager
        swarm_orchestrator_with_mocks.swarm_manager.get_swarm_status.assert_called_once_with(swarm_id)
        
        # Metrics should be updated
        mock_update_metrics.assert_called_once_with(swarm_id)
        
        # Verify comprehensive status structure
        assert status['id'] == swarm_id
        assert status['state'] == SwarmState.ACTIVE.value
        assert 'orchestrator_metrics' in status
        assert 'resource_usage' in status
        assert 'pending_tasks' in status
        assert 'executing_tasks' in status
        
        # Verify orchestrator metrics
        orchestrator_metrics = status['orchestrator_metrics']
        assert orchestrator_metrics['total_tasks'] == 100
        assert orchestrator_metrics['success_rate'] == metrics.success_rate
        assert orchestrator_metrics['efficiency_score'] == 92.3
    
    @pytest.mark.asyncio
    async def test_should_handle_nonexistent_swarm_gracefully(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test error handling for status requests of nonexistent swarms"""
        
        # Act & Assert
        with pytest.raises(SwarmError, match="Swarm nonexistent-swarm not found"):
            await swarm_orchestrator_with_mocks.get_swarm_status("nonexistent-swarm")


class TestSwarmScalingOrchestration:
    """Test swarm scaling operations - London School behavior focus"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_swarm_scale_up_workflow(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test complete scale up workflow coordination"""
        
        # Arrange
        swarm_id = "swarm-scale-test"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        swarm_orchestrator_with_mocks.swarm_manager.swarm_agents = {
            swarm_id: [Mock(), Mock()]  # Currently 2 agents
        }
        
        target_agents = 5
        agent_types = [AgentType.CODER, AgentType.TESTER, AgentType.ANALYST]
        
        # Mock metrics update
        with patch.object(
            swarm_orchestrator_with_mocks,
            '_update_swarm_metrics'
        ) as mock_update_metrics:
            
            # Act
            result = await swarm_orchestrator_with_mocks.scale_swarm(
                swarm_id=swarm_id,
                target_agents=target_agents,
                agent_types=agent_types
            )
        
        # Assert - Verify scaling workflow
        assert result is True
        
        # Verify swarm state transitions
        # Should go to SCALING during operation, then back to ACTIVE
        assert swarm_orchestrator_with_mocks.active_swarms[swarm_id] == SwarmState.ACTIVE
        
        # Verify agents were spawned (3 new agents needed)
        spawn_calls = swarm_orchestrator_with_mocks.swarm_manager.spawn_agent.call_args_list
        assert len(spawn_calls) == 3  # 5 target - 2 current = 3 new agents
        
        # Verify metrics were updated after scaling
        mock_update_metrics.assert_called_once_with(swarm_id)
    
    @pytest.mark.asyncio
    async def test_should_determine_optimal_agent_types_for_scaling(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test optimal agent type selection for scaling operations"""
        
        # Arrange
        swarm_id = "swarm-agent-types"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        swarm_orchestrator_with_mocks.swarm_manager.swarm_agents = {swarm_id: [Mock()]}
        
        # Mock optimal agent type determination
        with patch.object(
            swarm_orchestrator_with_mocks,
            '_determine_optimal_agent_types',
            return_value=[AgentType.CODER, AgentType.TESTER]
        ) as mock_determine_types:
            
            # Act
            await swarm_orchestrator_with_mocks.scale_swarm(
                swarm_id=swarm_id,
                target_agents=3  # Scale from 1 to 3
                # No agent_types specified, should use optimal determination
            )
        
        # Assert - Verify optimal agent type determination
        mock_determine_types.assert_called_once_with(swarm_id, 2)  # 2 agents to add
    
    @pytest.mark.asyncio
    async def test_should_handle_scaling_failure_gracefully(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test error handling during scaling operations"""
        
        # Arrange
        swarm_id = "swarm-scale-fail"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        
        # Force spawn_agent to fail
        swarm_orchestrator_with_mocks.swarm_manager.spawn_agent.side_effect = Exception(
            "Agent spawn failed"
        )
        
        # Act & Assert
        with pytest.raises(OrchestrationError, match="Swarm scaling failed"):
            await swarm_orchestrator_with_mocks.scale_swarm(
                swarm_id=swarm_id,
                target_agents=5
            )
        
        # Verify swarm state is set to ERROR after failure
        assert swarm_orchestrator_with_mocks.active_swarms[swarm_id] == SwarmState.ERROR


class TestAutoOptimizationWorkflow:
    """Test automatic optimization workflows - London School coordination testing"""
    
    @pytest.mark.asyncio
    async def test_should_auto_scale_up_on_high_utilization(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test automatic scale-up behavior when utilization is high"""
        
        # Arrange
        swarm_id = "swarm-auto-scale-up"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        
        # High utilization metrics
        high_util_metrics = SwarmMetrics(
            swarm_id=swarm_id,
            topology="mesh",
            total_agents=3,
            utilization_rate=95.0  # Very high utilization
        )
        swarm_orchestrator_with_mocks.swarm_metrics[swarm_id] = high_util_metrics
        
        # Mock scaling method
        with patch.object(
            swarm_orchestrator_with_mocks,
            'scale_swarm',
            return_value=True
        ) as mock_scale:
            
            # Act
            await swarm_orchestrator_with_mocks._optimize_swarm_performance(swarm_id)
        
        # Assert - Verify scale up was triggered
        mock_scale.assert_called_once_with(swarm_id, 4)  # Scale from 3 to 4
    
    @pytest.mark.asyncio
    async def test_should_auto_scale_down_on_low_utilization(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test automatic scale-down behavior when utilization is low"""
        
        # Arrange 
        swarm_id = "swarm-auto-scale-down"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        
        # Low utilization metrics
        low_util_metrics = SwarmMetrics(
            swarm_id=swarm_id,
            topology="mesh", 
            total_agents=5,
            utilization_rate=15.0  # Very low utilization
        )
        swarm_orchestrator_with_mocks.swarm_metrics[swarm_id] = low_util_metrics
        
        # Mock scaling method
        with patch.object(
            swarm_orchestrator_with_mocks,
            'scale_swarm',
            return_value=True
        ) as mock_scale:
            
            # Act
            await swarm_orchestrator_with_mocks._optimize_swarm_performance(swarm_id)
        
        # Assert - Verify scale down was triggered
        mock_scale.assert_called_once_with(swarm_id, 4)  # Scale from 5 to 4


class TestHealthMonitoringWorkflow:
    """Test health monitoring workflows - London School state transition testing"""
    
    @pytest.mark.asyncio
    async def test_should_detect_and_mark_degraded_swarms(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test detection and marking of degraded swarms"""
        
        # Arrange
        swarm_id = "swarm-degraded"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.ACTIVE
        
        # Metrics indicating degradation
        degraded_metrics = SwarmMetrics(
            swarm_id=swarm_id,
            topology="mesh",
            total_tasks=20,
            completed_tasks=8,  # Only 40% success rate
            failed_tasks=12,
            success_rate=40.0  # Below threshold
        )
        swarm_orchestrator_with_mocks.swarm_metrics[swarm_id] = degraded_metrics
        
        # Act
        await swarm_orchestrator_with_mocks._check_swarm_health(swarm_id)
        
        # Assert - Verify swarm marked as degraded
        assert swarm_orchestrator_with_mocks.active_swarms[swarm_id] == SwarmState.DEGRADED
    
    @pytest.mark.asyncio
    async def test_should_detect_swarm_recovery_from_degraded_state(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test detection of swarm recovery from degraded state"""
        
        # Arrange - Swarm starts in degraded state
        swarm_id = "swarm-recovery"
        swarm_orchestrator_with_mocks.active_swarms[swarm_id] = SwarmState.DEGRADED
        
        # Metrics indicating recovery
        recovered_metrics = SwarmMetrics(
            swarm_id=swarm_id,
            topology="mesh",
            total_tasks=30,
            completed_tasks=25,
            failed_tasks=5,
            active_agents=3,
            success_rate=83.3  # Above recovery threshold
        )
        swarm_orchestrator_with_mocks.swarm_metrics[swarm_id] = recovered_metrics
        
        # Act
        await swarm_orchestrator_with_mocks._check_swarm_health(swarm_id)
        
        # Assert - Verify swarm recovered to active state
        assert swarm_orchestrator_with_mocks.active_swarms[swarm_id] == SwarmState.ACTIVE


class TestGracefulShutdownWorkflow:
    """Test graceful shutdown coordination - London School cleanup verification"""
    
    @pytest.mark.asyncio
    async def test_should_coordinate_complete_shutdown_workflow(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test complete shutdown workflow coordination"""
        
        # Arrange - Add some executing tasks
        execution_id = "exec-shutdown-test"
        swarm_orchestrator_with_mocks.executing_tasks[execution_id] = {
            'task_request': Mock(),
            'swarm_id': 'swarm-test',
            'status': 'running'
        }
        
        # Mock monitoring tasks
        monitoring_task = Mock()
        monitoring_task.done.return_value = False
        optimization_task = Mock()  
        optimization_task.done.return_value = False
        
        swarm_orchestrator_with_mocks.monitoring_task = monitoring_task
        swarm_orchestrator_with_mocks.optimization_task = optimization_task
        
        # Act
        await swarm_orchestrator_with_mocks.shutdown()
        
        # Assert - Verify complete shutdown sequence
        
        # Monitoring tasks should be cancelled
        monitoring_task.cancel.assert_called_once()
        optimization_task.cancel.assert_called_once()
        
        # All tracking should be cleared
        assert len(swarm_orchestrator_with_mocks.active_swarms) == 0
        assert len(swarm_orchestrator_with_mocks.swarm_metrics) == 0
        assert len(swarm_orchestrator_with_mocks.pending_tasks) == 0
        assert len(swarm_orchestrator_with_mocks.executing_tasks) == 0


class TestContractCompliance:
    """Test contract compliance - London School interface verification"""
    
    def test_swarm_orchestrator_should_expose_expected_public_interface(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Verify SwarmOrchestrator exposes expected public methods"""
        
        # Assert public interface exists
        assert hasattr(swarm_orchestrator_with_mocks, 'initialize_swarm')
        assert hasattr(swarm_orchestrator_with_mocks, 'execute_task')
        assert hasattr(swarm_orchestrator_with_mocks, 'get_swarm_status')
        assert hasattr(swarm_orchestrator_with_mocks, 'scale_swarm')
        assert hasattr(swarm_orchestrator_with_mocks, 'shutdown')
        assert hasattr(swarm_orchestrator_with_mocks, 'get_global_metrics')
        
        # Verify methods are callable
        assert callable(getattr(swarm_orchestrator_with_mocks, 'initialize_swarm'))
        assert callable(getattr(swarm_orchestrator_with_mocks, 'execute_task'))
        assert callable(getattr(swarm_orchestrator_with_mocks, 'get_swarm_status'))
        assert callable(getattr(swarm_orchestrator_with_mocks, 'scale_swarm'))
        assert callable(getattr(swarm_orchestrator_with_mocks, 'shutdown'))
        assert callable(getattr(swarm_orchestrator_with_mocks, 'get_global_metrics'))
    
    def test_should_maintain_collaborator_contracts(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Verify SwarmOrchestrator maintains proper contracts with collaborators"""
        
        # Verify SwarmManager contract
        assert hasattr(swarm_orchestrator_with_mocks.swarm_manager, 'initialize_swarm')
        assert hasattr(swarm_orchestrator_with_mocks.swarm_manager, 'spawn_agent')
        assert hasattr(swarm_orchestrator_with_mocks.swarm_manager, 'get_swarm_status')
        
        # Verify ClaudeFlowOrchestrator contract
        assert hasattr(swarm_orchestrator_with_mocks.orchestrator, 'orchestrate_development_workflow')


class TestPerformanceOptimization:
    """Performance tests with pytest-benchmark for London School testing"""
    
    def test_swarm_orchestrator_initialization_performance(self, benchmark):
        """Benchmark SwarmOrchestrator initialization performance"""
        
        def create_orchestrator():
            return SwarmOrchestrator(max_swarms=5, max_agents_per_swarm=10)
        
        result = benchmark(create_orchestrator)
        assert result is not None
        assert hasattr(result, 'swarm_manager')
        assert hasattr(result, 'orchestrator')
    
    def test_metrics_calculation_performance(self, benchmark, swarm_orchestrator_with_mocks):
        """Benchmark metrics calculation performance"""
        
        # Setup test metrics
        swarm_id = "perf-test-swarm"
        metrics = SwarmMetrics(
            swarm_id=swarm_id,
            topology="mesh",
            total_tasks=1000,
            completed_tasks=850,
            failed_tasks=50
        )
        swarm_orchestrator_with_mocks.swarm_metrics[swarm_id] = metrics
        
        def calculate_global_metrics():
            return swarm_orchestrator_with_mocks.get_global_metrics()
        
        result = benchmark(calculate_global_metrics)
        assert result is not None
        assert 'average_swarm_efficiency' in result
        assert 'system_uptime_hours' in result


# Integration contract tests combining multiple London School patterns
class TestSwarmOrchestrationIntegrationContracts:
    """Integration tests verifying complete workflow contracts"""
    
    @pytest.mark.asyncio
    async def test_complete_swarm_lifecycle_workflow(
        self,
        swarm_orchestrator_with_mocks
    ):
        """Test complete swarm lifecycle from initialization to shutdown"""
        
        # Phase 1: Initialize swarm
        project_spec = {'features': ['auth'], 'requires_backend': True}
        swarm_id = await swarm_orchestrator_with_mocks.initialize_swarm(project_spec)
        
        # Phase 2: Execute task
        task_request = TaskRequest(
            task_id="lifecycle-task",
            description="Test lifecycle task",
            priority="medium"
        )
        execution_id = await swarm_orchestrator_with_mocks.execute_task(
            task_request, 
            swarm_id=swarm_id
        )
        
        # Phase 3: Check status
        status = await swarm_orchestrator_with_mocks.get_swarm_status(swarm_id)
        
        # Phase 4: Scale swarm
        await swarm_orchestrator_with_mocks.scale_swarm(swarm_id, target_agents=3)
        
        # Phase 5: Shutdown
        await swarm_orchestrator_with_mocks.shutdown()
        
        # Assert - Verify complete workflow executed
        assert swarm_id is not None
        assert execution_id is not None
        assert status is not None
        
        # Verify final state after shutdown
        assert len(swarm_orchestrator_with_mocks.active_swarms) == 0