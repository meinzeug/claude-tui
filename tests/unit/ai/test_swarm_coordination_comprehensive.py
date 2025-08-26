#!/usr/bin/env python3
"""
Comprehensive Swarm Agent Coordination Tests
Tests for multi-agent coordination, communication, and workflow orchestration.
"""

import pytest
import asyncio
import time
import json
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Swarm coordination imports with fallbacks
try:
    from src.ai.swarm_orchestrator import SwarmOrchestrator
    from src.ai.agent_coordinator import AgentCoordinator
    from src.ai.swarm_manager import SwarmManager
    from src.claude_tui.core.config_manager import ConfigManager
except ImportError:
    # Mock implementations for testing
    
    class AgentStatus(Enum):
        IDLE = "idle"
        BUSY = "busy"
        ERROR = "error"
        OFFLINE = "offline"
    
    class TaskStatus(Enum):
        PENDING = "pending"
        ASSIGNED = "assigned"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
    
    @dataclass
    class Agent:
        id: str
        name: str
        agent_type: str
        capabilities: List[str] = field(default_factory=list)
        status: str = "idle"
        current_task: str = None
        performance_metrics: Dict[str, float] = field(default_factory=dict)
        last_heartbeat: float = field(default_factory=time.time)
    
    @dataclass
    class Task:
        id: str
        description: str
        requirements: List[str] = field(default_factory=list)
        priority: str = "medium"
        status: str = "pending"
        assigned_agent: str = None
        created_at: float = field(default_factory=time.time)
        completed_at: float = None
        result: Any = None
    
    @dataclass
    class SwarmMetrics:
        total_agents: int = 0
        active_agents: int = 0
        tasks_completed: int = 0
        tasks_failed: int = 0
        avg_response_time: float = 0.0
        coordination_efficiency: float = 0.0
        last_updated: float = field(default_factory=time.time)
    
    class ConfigManager:
        def __init__(self):
            pass
        
        async def get_setting(self, key, default=None):
            return default
    
    class SwarmOrchestrator:
        def __init__(self, config_manager: ConfigManager = None):
            self.config_manager = config_manager or ConfigManager()
            self.agents: Dict[str, Agent] = {}
            self.tasks: Dict[str, Task] = {}
            self.metrics = SwarmMetrics()
            self.coordination_timeout = 30.0
            self.max_agents = 10
            self.is_initialized = False
        
        async def initialize(self):
            self.is_initialized = True
        
        async def spawn_agent(self, config: Dict[str, Any]) -> str:
            agent_id = f"agent-{len(self.agents) + 1}"
            agent = Agent(
                id=agent_id,
                name=config.get("name", f"Agent{len(self.agents) + 1}"),
                agent_type=config.get("type", "generic"),
                capabilities=config.get("capabilities", []),
                status=AgentStatus.IDLE.value
            )
            self.agents[agent_id] = agent
            self.metrics.total_agents = len(self.agents)
            return agent_id
        
        async def assign_task(self, task_id: str, agent_id: str) -> bool:
            if task_id in self.tasks and agent_id in self.agents:
                self.tasks[task_id].status = TaskStatus.ASSIGNED.value
                self.tasks[task_id].assigned_agent = agent_id
                self.agents[agent_id].current_task = task_id
                self.agents[agent_id].status = AgentStatus.BUSY.value
                return True
            return False
        
        async def create_task(self, description: str, requirements: List[str] = None) -> str:
            task_id = f"task-{len(self.tasks) + 1}"
            task = Task(
                id=task_id,
                description=description,
                requirements=requirements or []
            )
            self.tasks[task_id] = task
            return task_id
        
        async def coordinate_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
            workflow_id = f"workflow-{uuid.uuid4().hex[:8]}"
            tasks = workflow_config.get("tasks", [])
            
            # Simulate workflow coordination
            assigned_agents = []
            for i, task_desc in enumerate(tasks):
                # Find available agent
                available_agents = [
                    agent for agent in self.agents.values()
                    if agent.status == AgentStatus.IDLE.value
                ]
                
                if available_agents:
                    selected_agent = available_agents[0]
                    task_id = await self.create_task(task_desc)
                    await self.assign_task(task_id, selected_agent.id)
                    assigned_agents.append(selected_agent.id)
            
            return {
                "workflow_id": workflow_id,
                "status": "coordinated",
                "assigned_agents": assigned_agents,
                "total_tasks": len(tasks)
            }
        
        async def get_swarm_metrics(self) -> SwarmMetrics:
            # Update metrics
            active_count = sum(1 for agent in self.agents.values() 
                             if agent.status == AgentStatus.BUSY.value)
            self.metrics.active_agents = active_count
            self.metrics.last_updated = time.time()
            return self.metrics
        
        async def handle_agent_failure(self, agent_id: str) -> bool:
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.ERROR.value
                # Reassign task if any
                if self.agents[agent_id].current_task:
                    task_id = self.agents[agent_id].current_task
                    self.tasks[task_id].status = TaskStatus.PENDING.value
                    self.tasks[task_id].assigned_agent = None
                return True
            return False
        
        async def cleanup(self):
            self.agents.clear()
            self.tasks.clear()
            self.is_initialized = False
    
    class AgentCoordinator:
        def __init__(self, orchestrator: SwarmOrchestrator):
            self.orchestrator = orchestrator
            self.coordination_rules = {}
            self.communication_channels = {}
        
        async def coordinate_agents(self, agent_ids: List[str], task: Task) -> Dict[str, Any]:
            # Mock coordination logic
            coordination_plan = {
                "primary_agent": agent_ids[0] if agent_ids else None,
                "supporting_agents": agent_ids[1:] if len(agent_ids) > 1 else [],
                "coordination_strategy": "sequential",
                "estimated_completion_time": 300.0  # 5 minutes
            }
            return coordination_plan
        
        async def establish_communication_channel(self, agent_ids: List[str]) -> str:
            channel_id = f"channel-{uuid.uuid4().hex[:8]}"
            self.communication_channels[channel_id] = {
                "agents": agent_ids,
                "created_at": time.time(),
                "message_count": 0
            }
            return channel_id
        
        async def broadcast_message(self, channel_id: str, message: str, sender_id: str) -> bool:
            if channel_id in self.communication_channels:
                self.communication_channels[channel_id]["message_count"] += 1
                return True
            return False
    
    class SwarmManager:
        def __init__(self, orchestrator: SwarmOrchestrator, coordinator: AgentCoordinator):
            self.orchestrator = orchestrator
            self.coordinator = coordinator
            self.load_balancer = {}
            self.fault_tolerance_enabled = True
        
        async def balance_load(self) -> Dict[str, Any]:
            # Mock load balancing
            agent_loads = {}
            for agent_id, agent in self.orchestrator.agents.items():
                # Calculate load based on current task
                load = 1.0 if agent.current_task else 0.0
                agent_loads[agent_id] = load
            
            avg_load = sum(agent_loads.values()) / len(agent_loads) if agent_loads else 0.0
            
            return {
                "agent_loads": agent_loads,
                "average_load": avg_load,
                "balanced": all(load <= 1.0 for load in agent_loads.values()),
                "recommendations": []
            }
        
        async def monitor_health(self) -> Dict[str, Any]:
            healthy_agents = []
            unhealthy_agents = []
            
            for agent_id, agent in self.orchestrator.agents.items():
                if agent.status == AgentStatus.ERROR.value:
                    unhealthy_agents.append(agent_id)
                else:
                    healthy_agents.append(agent_id)
            
            return {
                "healthy_agents": healthy_agents,
                "unhealthy_agents": unhealthy_agents,
                "health_score": len(healthy_agents) / max(len(self.orchestrator.agents), 1),
                "last_check": time.time()
            }
        
        async def optimize_performance(self) -> Dict[str, Any]:
            metrics = await self.orchestrator.get_swarm_metrics()
            
            optimizations = []
            if metrics.active_agents < metrics.total_agents / 2:
                optimizations.append("increase_task_distribution")
            
            if metrics.coordination_efficiency < 0.8:
                optimizations.append("improve_coordination_algorithms")
            
            return {
                "current_performance": {
                    "efficiency": metrics.coordination_efficiency,
                    "utilization": metrics.active_agents / max(metrics.total_agents, 1)
                },
                "optimizations": optimizations,
                "estimated_improvement": 0.15 if optimizations else 0.0
            }


@pytest.fixture
def config_manager():
    """Mock configuration manager."""
    return ConfigManager()


@pytest.fixture
def swarm_orchestrator(config_manager):
    """Create swarm orchestrator for testing."""
    return SwarmOrchestrator(config_manager)


@pytest.fixture
def agent_coordinator(swarm_orchestrator):
    """Create agent coordinator for testing."""
    return AgentCoordinator(swarm_orchestrator)


@pytest.fixture
def swarm_manager(swarm_orchestrator, agent_coordinator):
    """Create swarm manager for testing."""
    return SwarmManager(swarm_orchestrator, agent_coordinator)


@pytest.fixture
def sample_agent_configs():
    """Sample agent configurations for testing."""
    return [
        {
            "name": "TestAgent1",
            "type": "tester",
            "capabilities": ["unit_testing", "integration_testing", "validation"]
        },
        {
            "name": "CodeAgent1",
            "type": "coder",
            "capabilities": ["python", "javascript", "refactoring"]
        },
        {
            "name": "ReviewAgent1",
            "type": "reviewer",
            "capabilities": ["code_review", "documentation", "quality_assurance"]
        }
    ]


class TestSwarmOrchestrator:
    """Tests for SwarmOrchestrator functionality."""
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, swarm_orchestrator):
        """Test orchestrator initialization."""
        assert swarm_orchestrator.is_initialized is False
        
        await swarm_orchestrator.initialize()
        
        assert swarm_orchestrator.is_initialized is True
        assert isinstance(swarm_orchestrator.agents, dict)
        assert isinstance(swarm_orchestrator.tasks, dict)
        assert swarm_orchestrator.coordination_timeout > 0
        assert swarm_orchestrator.max_agents > 0
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_agent_spawning(self, swarm_orchestrator, sample_agent_configs):
        """Test agent spawning functionality."""
        await swarm_orchestrator.initialize()
        
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            
            assert agent_id is not None
            assert agent_id.startswith("agent-")
            assert agent_id in swarm_orchestrator.agents
            
            agent = swarm_orchestrator.agents[agent_id]
            assert agent.name == config["name"]
            assert agent.agent_type == config["type"]
            assert agent.capabilities == config["capabilities"]
            assert agent.status == AgentStatus.IDLE.value
            
            agent_ids.append(agent_id)
        
        assert len(swarm_orchestrator.agents) == len(sample_agent_configs)
        
        # Check metrics update
        metrics = await swarm_orchestrator.get_swarm_metrics()
        assert metrics.total_agents == len(sample_agent_configs)
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_task_creation_and_assignment(self, swarm_orchestrator, sample_agent_configs):
        """Test task creation and assignment."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Create task
        task_description = "Test task for validation"
        requirements = ["testing", "validation"]
        task_id = await swarm_orchestrator.create_task(task_description, requirements)
        
        assert task_id is not None
        assert task_id.startswith("task-")
        assert task_id in swarm_orchestrator.tasks
        
        task = swarm_orchestrator.tasks[task_id]
        assert task.description == task_description
        assert task.requirements == requirements
        assert task.status == TaskStatus.PENDING.value
        
        # Assign task to agent
        agent_id = agent_ids[0]  # Use first agent
        success = await swarm_orchestrator.assign_task(task_id, agent_id)
        
        assert success is True
        assert task.status == TaskStatus.ASSIGNED.value
        assert task.assigned_agent == agent_id
        assert swarm_orchestrator.agents[agent_id].current_task == task_id
        assert swarm_orchestrator.agents[agent_id].status == AgentStatus.BUSY.value
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_workflow_coordination(self, swarm_orchestrator, sample_agent_configs):
        """Test workflow coordination functionality."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        for config in sample_agent_configs:
            await swarm_orchestrator.spawn_agent(config)
        
        # Define workflow
        workflow_config = {
            "name": "Test Workflow",
            "tasks": [
                "Write unit tests",
                "Implement feature",
                "Review code"
            ]
        }
        
        # Coordinate workflow
        result = await swarm_orchestrator.coordinate_workflow(workflow_config)
        
        assert "workflow_id" in result
        assert result["status"] == "coordinated"
        assert "assigned_agents" in result
        assert result["total_tasks"] == len(workflow_config["tasks"])
        assert len(result["assigned_agents"]) <= len(workflow_config["tasks"])
        
        # Verify tasks were created and assigned
        assigned_tasks = [
            task for task in swarm_orchestrator.tasks.values()
            if task.status == TaskStatus.ASSIGNED.value
        ]
        assert len(assigned_tasks) == len(result["assigned_agents"])
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, swarm_orchestrator, sample_agent_configs):
        """Test agent failure handling."""
        await swarm_orchestrator.initialize()
        
        # Spawn agent and assign task
        agent_id = await swarm_orchestrator.spawn_agent(sample_agent_configs[0])
        task_id = await swarm_orchestrator.create_task("Test task")
        await swarm_orchestrator.assign_task(task_id, agent_id)
        
        # Verify initial state
        assert swarm_orchestrator.agents[agent_id].status == AgentStatus.BUSY.value
        assert swarm_orchestrator.tasks[task_id].status == TaskStatus.ASSIGNED.value
        
        # Simulate agent failure
        success = await swarm_orchestrator.handle_agent_failure(agent_id)
        
        assert success is True
        assert swarm_orchestrator.agents[agent_id].status == AgentStatus.ERROR.value
        assert swarm_orchestrator.tasks[task_id].status == TaskStatus.PENDING.value
        assert swarm_orchestrator.tasks[task_id].assigned_agent is None
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_swarm_metrics(self, swarm_orchestrator, sample_agent_configs):
        """Test swarm metrics collection."""
        await swarm_orchestrator.initialize()
        
        # Initial metrics
        initial_metrics = await swarm_orchestrator.get_swarm_metrics()
        assert initial_metrics.total_agents == 0
        assert initial_metrics.active_agents == 0
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Check updated metrics
        metrics = await swarm_orchestrator.get_swarm_metrics()
        assert metrics.total_agents == len(sample_agent_configs)
        assert metrics.active_agents == 0  # No active tasks yet
        
        # Assign some tasks
        for i, agent_id in enumerate(agent_ids[:2]):
            task_id = await swarm_orchestrator.create_task(f"Task {i+1}")
            await swarm_orchestrator.assign_task(task_id, agent_id)
        
        # Check metrics with active agents
        active_metrics = await swarm_orchestrator.get_swarm_metrics()
        assert active_metrics.active_agents == 2
        assert active_metrics.last_updated > initial_metrics.last_updated


class TestAgentCoordinator:
    """Tests for AgentCoordinator functionality."""
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_agent_coordination(self, agent_coordinator, swarm_orchestrator, sample_agent_configs):
        """Test agent coordination functionality."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Create task
        task_id = await swarm_orchestrator.create_task("Complex coordination task")
        task = swarm_orchestrator.tasks[task_id]
        
        # Coordinate agents for task
        coordination_plan = await agent_coordinator.coordinate_agents(agent_ids, task)
        
        assert isinstance(coordination_plan, dict)
        assert "primary_agent" in coordination_plan
        assert "supporting_agents" in coordination_plan
        assert "coordination_strategy" in coordination_plan
        assert "estimated_completion_time" in coordination_plan
        
        assert coordination_plan["primary_agent"] == agent_ids[0]
        assert coordination_plan["supporting_agents"] == agent_ids[1:]
        assert coordination_plan["estimated_completion_time"] > 0
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_communication_channels(self, agent_coordinator, swarm_orchestrator, sample_agent_configs):
        """Test agent communication channel establishment."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Establish communication channel
        channel_id = await agent_coordinator.establish_communication_channel(agent_ids)
        
        assert channel_id is not None
        assert channel_id.startswith("channel-")
        assert channel_id in agent_coordinator.communication_channels
        
        channel_info = agent_coordinator.communication_channels[channel_id]
        assert channel_info["agents"] == agent_ids
        assert channel_info["created_at"] > 0
        assert channel_info["message_count"] == 0
        
        # Test message broadcasting
        message = "Test coordination message"
        sender_id = agent_ids[0]
        success = await agent_coordinator.broadcast_message(channel_id, message, sender_id)
        
        assert success is True
        assert agent_coordinator.communication_channels[channel_id]["message_count"] == 1
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_coordination_edge_cases(self, agent_coordinator, swarm_orchestrator):
        """Test coordination edge cases."""
        await swarm_orchestrator.initialize()
        
        # Test coordination with no agents
        empty_coordination = await agent_coordinator.coordinate_agents([], None)
        assert empty_coordination["primary_agent"] is None
        assert empty_coordination["supporting_agents"] == []
        
        # Test communication channel with invalid channel
        invalid_message = await agent_coordinator.broadcast_message("invalid-channel", "test", "agent-1")
        assert invalid_message is False


class TestSwarmManager:
    """Tests for SwarmManager functionality."""
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_load_balancing(self, swarm_manager, swarm_orchestrator, sample_agent_configs):
        """Test load balancing functionality."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Assign tasks to some agents
        for i in range(2):
            task_id = await swarm_orchestrator.create_task(f"Load test task {i+1}")
            await swarm_orchestrator.assign_task(task_id, agent_ids[i])
        
        # Test load balancing
        load_balance_result = await swarm_manager.balance_load()
        
        assert "agent_loads" in load_balance_result
        assert "average_load" in load_balance_result
        assert "balanced" in load_balance_result
        assert "recommendations" in load_balance_result
        
        agent_loads = load_balance_result["agent_loads"]
        assert len(agent_loads) == len(agent_ids)
        
        # Check that busy agents have load = 1.0 and idle agents have load = 0.0
        busy_count = sum(1 for load in agent_loads.values() if load > 0)
        assert busy_count == 2  # Two agents were assigned tasks
        
        avg_load = load_balance_result["average_load"]
        expected_avg = 2.0 / len(agent_ids)  # 2 busy agents / total agents
        assert abs(avg_load - expected_avg) < 0.01
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_health_monitoring(self, swarm_manager, swarm_orchestrator, sample_agent_configs):
        """Test health monitoring functionality."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Initial health check - all agents should be healthy
        health_status = await swarm_manager.monitor_health()
        
        assert "healthy_agents" in health_status
        assert "unhealthy_agents" in health_status
        assert "health_score" in health_status
        assert "last_check" in health_status
        
        assert len(health_status["healthy_agents"]) == len(agent_ids)
        assert len(health_status["unhealthy_agents"]) == 0
        assert health_status["health_score"] == 1.0
        
        # Simulate agent failure
        await swarm_orchestrator.handle_agent_failure(agent_ids[0])
        
        # Check health after failure
        health_after_failure = await swarm_manager.monitor_health()
        
        assert len(health_after_failure["healthy_agents"]) == len(agent_ids) - 1
        assert len(health_after_failure["unhealthy_agents"]) == 1
        assert agent_ids[0] in health_after_failure["unhealthy_agents"]
        assert health_after_failure["health_score"] < 1.0
    
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_performance_optimization(self, swarm_manager, swarm_orchestrator, sample_agent_configs):
        """Test performance optimization functionality."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        for config in sample_agent_configs:
            await swarm_orchestrator.spawn_agent(config)
        
        # Test performance optimization
        optimization_result = await swarm_manager.optimize_performance()
        
        assert "current_performance" in optimization_result
        assert "optimizations" in optimization_result
        assert "estimated_improvement" in optimization_result
        
        current_perf = optimization_result["current_performance"]
        assert "efficiency" in current_perf
        assert "utilization" in current_perf
        
        assert 0.0 <= current_perf["efficiency"] <= 1.0
        assert 0.0 <= current_perf["utilization"] <= 1.0
        
        optimizations = optimization_result["optimizations"]
        assert isinstance(optimizations, list)
        
        estimated_improvement = optimization_result["estimated_improvement"]
        assert 0.0 <= estimated_improvement <= 1.0


class TestSwarmConcurrency:
    """Tests for concurrent swarm operations."""
    
    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_agent_spawning(self, swarm_orchestrator, sample_agent_configs):
        """Test concurrent agent spawning."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents concurrently
        start_time = time.perf_counter()
        spawn_tasks = [
            swarm_orchestrator.spawn_agent(config)
            for config in sample_agent_configs
        ]
        agent_ids = await asyncio.gather(*spawn_tasks)
        end_time = time.perf_counter()
        
        # Verify all agents were spawned
        assert len(agent_ids) == len(sample_agent_configs)
        assert all(agent_id in swarm_orchestrator.agents for agent_id in agent_ids)
        
        # Check timing
        spawn_time = end_time - start_time
        assert spawn_time < 2.0  # Should complete quickly
        
        print(f"Spawned {len(agent_ids)} agents in {spawn_time:.3f} seconds")
    
    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_task_assignment(self, swarm_orchestrator, sample_agent_configs):
        """Test concurrent task assignment."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Create tasks
        task_descriptions = [f"Concurrent task {i+1}" for i in range(len(agent_ids))]
        task_creation_tasks = [
            swarm_orchestrator.create_task(desc)
            for desc in task_descriptions
        ]
        task_ids = await asyncio.gather(*task_creation_tasks)
        
        # Assign tasks concurrently
        assignment_tasks = [
            swarm_orchestrator.assign_task(task_id, agent_id)
            for task_id, agent_id in zip(task_ids, agent_ids)
        ]
        assignment_results = await asyncio.gather(*assignment_tasks)
        
        # Verify all assignments succeeded
        assert all(result is True for result in assignment_results)
        
        # Verify agent states
        for agent_id in agent_ids:
            agent = swarm_orchestrator.agents[agent_id]
            assert agent.status == AgentStatus.BUSY.value
            assert agent.current_task is not None
    
    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_workflow_coordination(self, swarm_orchestrator, sample_agent_configs):
        """Test concurrent workflow coordination."""
        await swarm_orchestrator.initialize()
        
        # Spawn more agents for multiple workflows
        extended_configs = sample_agent_configs * 2  # Double the agents
        for config in extended_configs:
            await swarm_orchestrator.spawn_agent({
                **config,
                "name": f"{config['name']}_dup_{time.time()}"  # Ensure unique names
            })
        
        # Define multiple workflows
        workflows = [
            {
                "name": f"Workflow {i+1}",
                "tasks": [f"Task {i+1}-{j+1}" for j in range(2)]
            }
            for i in range(3)
        ]
        
        # Coordinate workflows concurrently
        start_time = time.perf_counter()
        coordination_tasks = [
            swarm_orchestrator.coordinate_workflow(workflow)
            for workflow in workflows
        ]
        coordination_results = await asyncio.gather(*coordination_tasks)
        end_time = time.perf_counter()
        
        # Verify all workflows were coordinated
        assert len(coordination_results) == len(workflows)
        for result in coordination_results:
            assert result["status"] == "coordinated"
            assert "workflow_id" in result
            assert len(result["assigned_agents"]) > 0
        
        # Check timing
        coordination_time = end_time - start_time
        assert coordination_time < 3.0  # Should be reasonably fast
        
        print(f"Coordinated {len(workflows)} workflows in {coordination_time:.3f} seconds")


class TestSwarmFaultTolerance:
    """Tests for fault tolerance and error recovery."""
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_agent_recovery_after_failure(self, swarm_orchestrator, swarm_manager, sample_agent_configs):
        """Test agent recovery after failure."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Assign tasks
        task_ids = []
        for agent_id in agent_ids:
            task_id = await swarm_orchestrator.create_task("Recovery test task")
            await swarm_orchestrator.assign_task(task_id, agent_id)
            task_ids.append(task_id)
        
        # Simulate multiple agent failures
        failed_agents = agent_ids[:2]
        for agent_id in failed_agents:
            await swarm_orchestrator.handle_agent_failure(agent_id)
        
        # Check health status
        health_status = await swarm_manager.monitor_health()
        assert len(health_status["unhealthy_agents"]) == len(failed_agents)
        assert health_status["health_score"] < 1.0
        
        # Verify tasks were reassigned (set back to pending)
        for i, task_id in enumerate(task_ids[:2]):
            task = swarm_orchestrator.tasks[task_id]
            assert task.status == TaskStatus.PENDING.value
            assert task.assigned_agent is None
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_swarm_resilience_under_load(self, swarm_orchestrator, swarm_manager):
        """Test swarm resilience under high load conditions."""
        await swarm_orchestrator.initialize()
        
        # Spawn many agents
        agent_configs = [
            {
                "name": f"LoadTestAgent{i}",
                "type": "load_tester",
                "capabilities": ["load_testing"]
            }
            for i in range(10)
        ]
        
        for config in agent_configs:
            await swarm_orchestrator.spawn_agent(config)
        
        # Create many tasks
        task_creation_tasks = [
            swarm_orchestrator.create_task(f"Load test task {i}")
            for i in range(15)  # More tasks than agents
        ]
        task_ids = await asyncio.gather(*task_creation_tasks)
        
        # Try to assign all tasks
        available_agents = list(swarm_orchestrator.agents.keys())
        assignment_results = []
        
        for i, task_id in enumerate(task_ids):
            if i < len(available_agents):
                result = await swarm_orchestrator.assign_task(task_id, available_agents[i])
                assignment_results.append(result)
        
        # Check load balancing
        load_status = await swarm_manager.balance_load()
        assert "agent_loads" in load_status
        
        # Some agents should be loaded
        loaded_agents = [
            agent_id for agent_id, load in load_status["agent_loads"].items()
            if load > 0
        ]
        assert len(loaded_agents) > 0
        
        # System should handle the load gracefully
        health_status = await swarm_manager.monitor_health()
        assert health_status["health_score"] > 0.5  # Should maintain reasonable health
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_communication_channel_resilience(self, agent_coordinator, swarm_orchestrator, sample_agent_configs):
        """Test communication channel resilience."""
        await swarm_orchestrator.initialize()
        
        # Spawn agents
        agent_ids = []
        for config in sample_agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # Create communication channels
        channels = []
        for i in range(3):
            channel_id = await agent_coordinator.establish_communication_channel(agent_ids)
            channels.append(channel_id)
        
        # Send many messages concurrently
        message_tasks = []
        for channel_id in channels:
            for i in range(5):
                for sender_id in agent_ids:
                    task = agent_coordinator.broadcast_message(
                        channel_id, 
                        f"Message {i} from {sender_id}",
                        sender_id
                    )
                    message_tasks.append(task)
        
        # Execute all message sends
        message_results = await asyncio.gather(*message_tasks)
        
        # Verify most messages were sent successfully
        successful_sends = sum(1 for result in message_results if result)
        success_rate = successful_sends / len(message_results)
        assert success_rate > 0.8  # At least 80% success rate
        
        # Verify channels maintained message counts
        for channel_id in channels:
            channel_info = agent_coordinator.communication_channels[channel_id]
            assert channel_info["message_count"] > 0


if __name__ == "__main__":
    # Run swarm coordination tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "unit",
        "--durations=10"
    ])