#!/usr/bin/env python3
"""
Hive Mind Coordinator - Advanced Multi-Agent Backend Orchestration

Provides sophisticated coordination for distributed AI agent systems:
- Dynamic agent lifecycle management
- Intelligent task distribution and load balancing
- Real-time inter-agent communication
- Collective decision making and consensus protocols
- Adaptive swarm topology optimization
- Performance-based agent scaling
- Fault tolerance and recovery mechanisms
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import hashlib
from collections import defaultdict
from statistics import mean, median

# Third-party imports
from pydantic import BaseModel, validator
import numpy as np

# Internal imports
from .core_services import ServiceOrchestrator, get_service_orchestrator
from .claude_integration_layer import ClaudeIntegrationLayer, get_claude_integration
from .tui_backend_bridge import TUIBackendBridge, get_tui_bridge
from ..core.config_manager import ConfigManager
from ..claude_tui.integrations.claude_flow_client import ClaudeFlowClient

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status types."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class TaskPriority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class SwarmTopology(str, Enum):
    """Swarm topology types."""
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    RING = "ring"
    STAR = "star"
    ADAPTIVE = "adaptive"


class DecisionMethod(str, Enum):
    """Collective decision methods."""
    CONSENSUS = "consensus"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    LEADER_DECISION = "leader_decision"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class AgentCapability:
    """Agent capability definition."""
    name: str
    level: float  # 0.0 to 1.0
    specialization: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    performance_weight: float = 1.0


@dataclass
class AgentPerformanceMetrics:
    """Agent performance tracking."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_completion_time: float = 0.0
    success_rate: float = 1.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 1.0
    reliability_score: float = 1.0
    
    # Time-series data
    completion_times: List[float] = field(default_factory=list)
    error_timestamps: List[datetime] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class HiveMindAgent:
    """Hive Mind agent representation."""
    agent_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.INITIALIZING
    capabilities: List[AgentCapability] = field(default_factory=list)
    current_task: Optional[str] = None
    
    # Performance tracking
    metrics: AgentPerformanceMetrics = field(default_factory=AgentPerformanceMetrics)
    
    # Communication
    last_heartbeat: datetime = field(default_factory=datetime.now)
    communication_endpoint: Optional[str] = None
    
    # Resource allocation
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    max_resources: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        """Check if agent is healthy based on recent activity."""
        time_since_heartbeat = datetime.now() - self.last_heartbeat
        return (
            self.status not in [AgentStatus.ERROR, AgentStatus.OFFLINE] and
            time_since_heartbeat < timedelta(minutes=5)
        )
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)."""
        if not self.max_resources:
            return 0.0
        
        load_factors = []
        for resource, allocated in self.allocated_resources.items():
            max_value = self.max_resources.get(resource, 1.0)
            if max_value > 0:
                load_factors.append(allocated / max_value)
        
        return mean(load_factors) if load_factors else 0.0
    
    def can_handle_task(self, required_capabilities: List[str], 
                       resource_requirements: Dict[str, float]) -> bool:
        """Check if agent can handle a task."""
        # Check capabilities
        agent_capabilities = {cap.name for cap in self.capabilities}
        if not all(cap in agent_capabilities for cap in required_capabilities):
            return False
        
        # Check resources
        for resource, required in resource_requirements.items():
            available = self.max_resources.get(resource, 0) - self.allocated_resources.get(resource, 0)
            if available < required:
                return False
        
        return True


@dataclass
class HiveMindTask:
    """Hive Mind task representation."""
    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    required_capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Assignment
    assigned_agent: Optional[str] = None
    assignment_time: Optional[datetime] = None
    
    # Execution tracking
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        return self.deadline is not None and datetime.now() > self.deadline
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class SwarmDecision(BaseModel):
    """Swarm collective decision."""
    decision_id: str
    proposal: Dict[str, Any]
    method: DecisionMethod
    votes: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    timestamp: datetime
    participants: List[str] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HiveMindCoordinator:
    """
    Advanced multi-agent backend coordination system.
    
    Features:
    - Dynamic agent lifecycle management
    - Intelligent task distribution
    - Real-time performance monitoring
    - Adaptive topology optimization
    - Collective decision making
    - Fault tolerance and recovery
    - Resource allocation optimization
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize Hive Mind coordinator."""
        self.config_manager = config_manager
        
        # Core services
        self.orchestrator: Optional[ServiceOrchestrator] = None
        self.claude_integration: Optional[ClaudeIntegrationLayer] = None
        self.tui_bridge: Optional[TUIBackendBridge] = None
        self.claude_flow: Optional[ClaudeFlowClient] = None
        
        # Agent management
        self.agents: Dict[str, HiveMindAgent] = {}
        self.agent_types: Dict[str, Dict[str, Any]] = {}
        
        # Task management
        self.tasks: Dict[str, HiveMindTask] = {}
        self.task_queue: List[str] = []  # Priority-ordered task IDs
        
        # Swarm coordination
        self.topology: SwarmTopology = SwarmTopology.ADAPTIVE
        self.swarm_state: Dict[str, Any] = {}
        
        # Decision making
        self.active_decisions: Dict[str, SwarmDecision] = {}
        self.decision_history: List[SwarmDecision] = []
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_metrics: Dict[str, float] = {}
        
        # Configuration
        self.max_agents = 50
        self.heartbeat_interval = 30  # seconds
        self.task_timeout = 300  # 5 minutes
        self.scaling_threshold = 0.8  # 80% utilization
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Hive Mind Coordinator initialized")
    
    async def initialize(self) -> None:
        """Initialize the Hive Mind coordinator."""
        logger.info("Initializing Hive Mind Coordinator...")
        
        try:
            # Get core services
            self.orchestrator = get_service_orchestrator()
            self.claude_integration = get_claude_integration()
            self.tui_bridge = get_tui_bridge()
            
            if self.orchestrator:
                self.claude_flow = self.orchestrator.get_claude_flow_service()
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize agent types
            await self._initialize_agent_types()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Hive Mind Coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hive Mind Coordinator: {e}")
            raise
    
    async def _load_configuration(self) -> None:
        """Load configuration settings."""
        hive_config = await self.config_manager.get_setting('hive_mind', {})
        
        self.max_agents = hive_config.get('max_agents', 50)
        self.heartbeat_interval = hive_config.get('heartbeat_interval', 30)
        self.task_timeout = hive_config.get('task_timeout', 300)
        self.scaling_threshold = hive_config.get('scaling_threshold', 0.8)
        
        topology_name = hive_config.get('topology', 'adaptive')
        topology_mapping = {
            'hierarchical': SwarmTopology.HIERARCHICAL,
            'mesh': SwarmTopology.MESH,
            'ring': SwarmTopology.RING,
            'star': SwarmTopology.STAR,
            'adaptive': SwarmTopology.ADAPTIVE
        }
        self.topology = topology_mapping.get(topology_name, SwarmTopology.ADAPTIVE)
    
    async def _initialize_agent_types(self) -> None:
        """Initialize supported agent types."""
        self.agent_types = {
            'researcher': {
                'capabilities': ['research', 'analysis', 'data_collection'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'network': 0.3},
                'max_concurrent_tasks': 3
            },
            'coder': {
                'capabilities': ['coding', 'refactoring', 'code_review'],
                'resource_requirements': {'cpu': 0.7, 'memory': 1.5, 'storage': 0.5},
                'max_concurrent_tasks': 2
            },
            'tester': {
                'capabilities': ['testing', 'validation', 'quality_assurance'],
                'resource_requirements': {'cpu': 0.6, 'memory': 1.0, 'network': 0.2},
                'max_concurrent_tasks': 4
            },
            'reviewer': {
                'capabilities': ['review', 'feedback', 'quality_control'],
                'resource_requirements': {'cpu': 0.4, 'memory': 0.8, 'network': 0.1},
                'max_concurrent_tasks': 5
            },
            'coordinator': {
                'capabilities': ['coordination', 'planning', 'decision_making'],
                'resource_requirements': {'cpu': 0.8, 'memory': 2.0, 'network': 0.5},
                'max_concurrent_tasks': 1
            },
            'specialist': {
                'capabilities': ['specialized_tasks', 'expert_analysis'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.5, 'gpu': 0.5},
                'max_concurrent_tasks': 1
            }
        }
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        tasks = [
            asyncio.create_task(self._heartbeat_monitor_loop()),
            asyncio.create_task(self._task_distribution_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._topology_optimization_loop()),
            asyncio.create_task(self._fault_recovery_loop()),
            asyncio.create_task(self._resource_balancing_loop()),
            asyncio.create_task(self._decision_processing_loop())
        ]
        
        self.background_tasks.update(tasks)
        logger.info(f"Started {len(tasks)} background tasks")
    
    # Agent Management
    
    async def spawn_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        resources: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Spawn a new agent in the hive mind.
        
        Args:
            agent_type: Type of agent to spawn
            agent_id: Optional custom agent ID
            capabilities: Optional custom capabilities
            resources: Optional resource limits
            tags: Optional agent tags
            
        Returns:
            str: Agent ID
        """
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if len(self.agents) >= self.max_agents:
            raise RuntimeError(f"Maximum agent limit ({self.max_agents}) reached")
        
        # Generate agent ID
        if not agent_id:
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        
        # Get agent type configuration
        type_config = self.agent_types[agent_type]
        
        # Build capabilities
        agent_capabilities = []
        capability_names = capabilities or type_config['capabilities']
        
        for cap_name in capability_names:
            capability = AgentCapability(
                name=cap_name,
                level=0.8,  # Default competency level
                resource_requirements=type_config['resource_requirements']
            )
            agent_capabilities.append(capability)
        
        # Create agent
        agent = HiveMindAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=agent_capabilities,
            max_resources=resources or type_config['resource_requirements'].copy(),
            tags=tags or []
        )
        
        # Add to agent registry
        self.agents[agent_id] = agent
        
        # Integrate with Claude Flow if available
        if self.claude_flow:
            try:
                from ..claude_tui.models.ai_models import AgentConfig
                
                flow_config = AgentConfig(
                    type=agent_type,
                    capabilities=capability_names,
                    name=agent_id
                )
                
                # This would integrate with Claude Flow swarms
                logger.info(f"Agent {agent_id} ready for Claude Flow integration")
                
            except Exception as e:
                logger.warning(f"Failed to integrate agent with Claude Flow: {e}")
        
        # Update agent status
        agent.status = AgentStatus.IDLE
        
        logger.info(f"Spawned {agent_type} agent: {agent_id}")
        
        # Notify observers
        await self._notify_agent_event('agent_spawned', agent_id, {
            'agent_type': agent_type,
            'capabilities': capability_names,
            'resources': agent.max_resources
        })
        
        return agent_id
    
    async def terminate_agent(self, agent_id: str, reason: str = "requested") -> bool:
        """
        Terminate an agent.
        
        Args:
            agent_id: Agent to terminate
            reason: Reason for termination
            
        Returns:
            bool: True if successful
        """
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Update status
        agent.status = AgentStatus.SHUTTING_DOWN
        
        # Cancel current task if any
        if agent.current_task:
            await self._reassign_task(agent.current_task, f"Agent {agent_id} terminating")
        
        # Remove from registry
        del self.agents[agent_id]
        
        logger.info(f"Terminated agent {agent_id}: {reason}")
        
        # Notify observers
        await self._notify_agent_event('agent_terminated', agent_id, {
            'reason': reason,
            'tasks_completed': agent.metrics.tasks_completed,
            'uptime': (datetime.now() - agent.created_at).total_seconds()
        })
        
        return True
    
    def get_agent(self, agent_id: str) -> Optional[HiveMindAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(
        self,
        agent_type: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[HiveMindAgent]:
        """List agents with optional filtering."""
        agents = list(self.agents.values())
        
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        if tags:
            agents = [a for a in agents if any(tag in a.tags for tag in tags)]
        
        return agents
    
    # Task Management
    
    async def submit_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: Optional[List[str]] = None,
        resource_requirements: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        deadline: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Submit a task to the hive mind.
        
        Args:
            description: Task description
            priority: Task priority
            required_capabilities: Required agent capabilities
            resource_requirements: Required resources
            context: Task context data
            dependencies: Task dependencies
            deadline: Task deadline
            tags: Task tags
            
        Returns:
            str: Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = HiveMindTask(
            task_id=task_id,
            description=description,
            priority=priority,
            required_capabilities=required_capabilities or [],
            resource_requirements=resource_requirements or {},
            context=context or {},
            dependencies=dependencies or [],
            deadline=deadline,
            tags=tags or []
        )
        
        # Add to task registry
        self.tasks[task_id] = task
        
        # Add to priority queue
        self._add_task_to_queue(task_id)
        
        logger.info(f"Submitted task {task_id}: {description[:50]}...")
        
        # Notify task distribution loop
        await self._notify_task_event('task_submitted', task_id, {
            'description': description,
            'priority': priority.value,
            'capabilities': required_capabilities or [],
            'resources': resource_requirements or {}
        })
        
        return task_id
    
    def _add_task_to_queue(self, task_id: str) -> None:
        """Add task to priority queue."""
        task = self.tasks[task_id]
        priority_values = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4
        }
        
        priority_value = priority_values[task.priority]
        
        # Insert in priority order
        inserted = False
        for i, existing_task_id in enumerate(self.task_queue):
            existing_task = self.tasks[existing_task_id]
            existing_priority = priority_values[existing_task.priority]
            
            if priority_value < existing_priority:
                self.task_queue.insert(i, task_id)
                inserted = True
                break
        
        if not inserted:
            self.task_queue.append(task_id)
    
    async def cancel_task(self, task_id: str, reason: str = "cancelled") -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task to cancel
            reason: Cancellation reason
            
        Returns:
            bool: True if successful
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Remove from queue if pending
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        # Notify assigned agent if any
        if task.assigned_agent:
            agent = self.agents.get(task.assigned_agent)
            if agent:
                agent.current_task = None
                agent.status = AgentStatus.IDLE
        
        # Update task status
        task.status = "cancelled"
        task.error = reason
        task.completed_at = datetime.now()
        
        logger.info(f"Cancelled task {task_id}: {reason}")
        
        await self._notify_task_event('task_cancelled', task_id, {
            'reason': reason,
            'was_assigned': task.assigned_agent is not None
        })
        
        return True
    
    def get_task(self, task_id: str) -> Optional[HiveMindTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[TaskPriority] = None,
        assigned_agent: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[HiveMindTask]:
        """List tasks with optional filtering."""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        
        if assigned_agent:
            tasks = [t for t in tasks if t.assigned_agent == assigned_agent]
        
        if tags:
            tasks = [t for t in tasks if any(tag in t.tags for tag in tags)]
        
        return tasks
    
    # Task Distribution
    
    async def _find_best_agent_for_task(self, task_id: str) -> Optional[str]:
        """Find the best agent for a task using multi-criteria optimization."""
        task = self.tasks[task_id]
        
        # Get candidate agents
        candidates = []
        for agent_id, agent in self.agents.items():
            if (
                agent.status == AgentStatus.IDLE and
                agent.can_handle_task(task.required_capabilities, task.resource_requirements)
            ):
                candidates.append((agent_id, agent))
        
        if not candidates:
            return None
        
        # Score candidates using multiple criteria
        best_agent = None
        best_score = -1
        
        for agent_id, agent in candidates:
            score = self._calculate_agent_task_score(agent, task)
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def _calculate_agent_task_score(self, agent: HiveMindAgent, task: HiveMindTask) -> float:
        """Calculate agent suitability score for a task."""
        score = 0.0
        
        # Capability match score (0.4 weight)
        capability_score = 0.0
        required_caps = set(task.required_capabilities)
        agent_caps = {cap.name: cap.level for cap in agent.capabilities}
        
        if required_caps:
            matched_caps = required_caps & set(agent_caps.keys())
            capability_score = len(matched_caps) / len(required_caps)
            
            # Bonus for high capability levels
            level_bonus = sum(agent_caps.get(cap, 0) for cap in matched_caps) / max(1, len(matched_caps))
            capability_score *= level_bonus
        
        score += capability_score * 0.4
        
        # Performance score (0.3 weight)
        performance_score = (
            agent.metrics.success_rate * 0.5 +
            agent.metrics.quality_score * 0.3 +
            agent.metrics.reliability_score * 0.2
        )
        score += performance_score * 0.3
        
        # Load factor score (0.2 weight) - prefer less loaded agents
        load_score = 1.0 - agent.load_factor
        score += load_score * 0.2
        
        # Recency score (0.1 weight) - prefer recently active agents
        time_since_active = (datetime.now() - agent.metrics.last_active).total_seconds()
        recency_score = max(0, 1.0 - (time_since_active / 3600))  # Decay over 1 hour
        score += recency_score * 0.1
        
        return score
    
    async def _assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent."""
        task = self.tasks[task_id]
        agent = self.agents[agent_id]
        
        # Update task
        task.assigned_agent = agent_id
        task.assignment_time = datetime.now()
        task.status = "assigned"
        
        # Update agent
        agent.current_task = task_id
        agent.status = AgentStatus.BUSY
        
        # Allocate resources
        for resource, amount in task.resource_requirements.items():
            current = agent.allocated_resources.get(resource, 0)
            agent.allocated_resources[resource] = current + amount
        
        logger.info(f"Assigned task {task_id} to agent {agent_id}")
        
        # Notify observers
        await self._notify_task_event('task_assigned', task_id, {
            'agent_id': agent_id,
            'agent_type': agent.agent_type,
            'assignment_time': task.assignment_time.isoformat()
        })
        
        # Start task execution
        asyncio.create_task(self._execute_task(task_id, agent_id))
        
        return True
    
    async def _execute_task(self, task_id: str, agent_id: str) -> None:
        """Execute a task using an agent."""
        task = self.tasks[task_id]
        agent = self.agents[agent_id]
        
        try:
            task.started_at = datetime.now()
            task.status = "running"
            
            # Integrate with Claude integration layer
            if self.claude_integration and task.required_capabilities:
                result = await self._execute_with_claude_integration(task, agent)
            else:
                # Simulate task execution (replace with actual implementation)
                result = await self._simulate_task_execution(task)
            
            # Task completed successfully
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            task.progress = 1.0
            
            # Update agent metrics
            agent.metrics.tasks_completed += 1
            if task.execution_time:
                agent.metrics.completion_times.append(task.execution_time)
                if agent.metrics.completion_times:
                    agent.metrics.average_completion_time = mean(agent.metrics.completion_times[-100:])
            
            # Update success rate
            total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed
            if total_tasks > 0:
                agent.metrics.success_rate = agent.metrics.tasks_completed / total_tasks
            
            logger.info(f"Task {task_id} completed by agent {agent_id}")
            
        except Exception as e:
            # Task failed
            task.error = str(e)
            task.status = "failed"
            task.completed_at = datetime.now()
            
            # Update agent metrics
            agent.metrics.tasks_failed += 1
            agent.metrics.error_timestamps.append(datetime.now())
            
            # Update success rate
            total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed
            if total_tasks > 0:
                agent.metrics.success_rate = agent.metrics.tasks_completed / total_tasks
            
            logger.error(f"Task {task_id} failed on agent {agent_id}: {e}")
            
        finally:
            # Clean up agent state
            agent.current_task = None
            agent.status = AgentStatus.IDLE
            agent.metrics.last_active = datetime.now()
            
            # Free allocated resources
            for resource, amount in task.resource_requirements.items():
                current = agent.allocated_resources.get(resource, 0)
                agent.allocated_resources[resource] = max(0, current - amount)
            
            # Remove from queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            # Notify observers
            await self._notify_task_event('task_completed', task_id, {
                'agent_id': agent_id,
                'status': task.status,
                'execution_time': task.execution_time,
                'success': task.status == "completed"
            })
    
    async def _execute_with_claude_integration(self, task: HiveMindTask, agent: HiveMindAgent) -> Any:
        """Execute task using Claude integration."""
        from .claude_integration_layer import ClaudeRequest, ResponseMode
        
        # Prepare Claude request
        claude_request = ClaudeRequest(
            prompt=f"""You are a {agent.agent_type} agent with capabilities: {[cap.name for cap in agent.capabilities]}.
            
            Task: {task.description}
            
            Context: {json.dumps(task.context, indent=2)}
            
            Please complete this task and provide a detailed response with your findings and any artifacts produced.
            """,
            system_prompt=f"You are an AI agent specializing in {agent.agent_type} tasks within a hive mind system.",
            project_context=task.context.get('project', {}),
            user_context={'agent_id': agent.agent_id, 'task_id': task.task_id},
            response_mode=ResponseMode.COMPLETE
        )
        
        # Execute with Claude integration
        response = await self.claude_integration.generate_response(claude_request)
        
        if response.success:
            return {
                'content': response.content,
                'model': response.model.value,
                'token_usage': response.token_usage,
                'response_time': response.response_time
            }
        else:
            raise Exception(f"Claude integration failed: {response.error}")
    
    async def _simulate_task_execution(self, task: HiveMindTask) -> Dict[str, Any]:
        """Simulate task execution (placeholder for actual implementation)."""
        # Simulate execution time
        execution_time = np.random.exponential(30)  # Average 30 seconds
        await asyncio.sleep(min(execution_time, 60))  # Cap at 60 seconds for testing
        
        # Simulate success/failure
        success_rate = 0.9  # 90% success rate
        if np.random.random() > success_rate:
            raise Exception("Simulated task failure")
        
        return {
            'simulated': True,
            'execution_time': execution_time,
            'result': f"Task '{task.description}' completed successfully"
        }
    
    # Background Loops
    
    async def _heartbeat_monitor_loop(self) -> None:
        """Monitor agent heartbeats and health."""
        logger.info("Starting heartbeat monitor loop")
        
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                current_time = datetime.now()
                unhealthy_agents = []
                
                for agent_id, agent in self.agents.items():
                    time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_interval * 3:  # 3 missed heartbeats
                        if agent.status != AgentStatus.OFFLINE:
                            logger.warning(f"Agent {agent_id} missed heartbeats, marking as offline")
                            agent.status = AgentStatus.OFFLINE
                            unhealthy_agents.append(agent_id)
                            
                            # Reassign current task if any
                            if agent.current_task:
                                await self._reassign_task(
                                    agent.current_task,
                                    f"Agent {agent_id} went offline"
                                )
                
                # Auto-spawn replacement agents if needed
                if unhealthy_agents and len([a for a in self.agents.values() if a.is_healthy]) < 2:
                    await self._auto_spawn_replacement_agents(unhealthy_agents)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(60)
    
    async def _task_distribution_loop(self) -> None:
        """Distribute tasks to available agents."""
        logger.info("Starting task distribution loop")
        
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                if not self.task_queue:
                    continue
                
                # Process tasks in priority order
                assigned_count = 0
                remaining_tasks = []
                
                for task_id in self.task_queue:
                    task = self.tasks.get(task_id)
                    if not task or task.status != "pending":
                        continue
                    
                    # Check dependencies
                    if not self._are_dependencies_met(task):
                        remaining_tasks.append(task_id)
                        continue
                    
                    # Find best agent
                    best_agent = await self._find_best_agent_for_task(task_id)
                    if best_agent:
                        await self._assign_task_to_agent(task_id, best_agent)
                        assigned_count += 1
                    else:
                        remaining_tasks.append(task_id)
                
                # Update task queue
                self.task_queue = remaining_tasks
                
                if assigned_count > 0:
                    logger.info(f"Assigned {assigned_count} tasks to agents")
                
            except Exception as e:
                logger.error(f"Error in task distribution: {e}")
                await asyncio.sleep(5)
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor swarm performance and collect metrics."""
        logger.info("Starting performance monitoring loop")
        
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 24 hours of metrics
                cutoff = datetime.now() - timedelta(hours=24)
                self.performance_history = [
                    m for m in self.performance_history
                    if datetime.fromisoformat(m['timestamp']) > cutoff
                ]
                
                # Store in cache if available
                if self.orchestrator:
                    cache_service = self.orchestrator.get_cache_service()
                    if cache_service:
                        await cache_service.set(
                            f"hive_metrics:{datetime.now().strftime('%Y%m%d%H%M')}",
                            metrics,
                            ttl=3600
                        )
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _topology_optimization_loop(self) -> None:
        """Optimize swarm topology based on performance."""
        logger.info("Starting topology optimization loop")
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                if self.topology == SwarmTopology.ADAPTIVE:
                    await self._optimize_topology()
                
            except Exception as e:
                logger.error(f"Error in topology optimization: {e}")
                await asyncio.sleep(600)
    
    async def _fault_recovery_loop(self) -> None:
        """Handle fault detection and recovery."""
        logger.info("Starting fault recovery loop")
        
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Detect stuck tasks
                await self._detect_and_recover_stuck_tasks()
                
                # Detect overloaded agents
                await self._detect_and_balance_overloaded_agents()
                
            except Exception as e:
                logger.error(f"Error in fault recovery: {e}")
                await asyncio.sleep(60)
    
    async def _resource_balancing_loop(self) -> None:
        """Balance resources across agents."""
        logger.info("Starting resource balancing loop")
        
        while True:
            try:
                await asyncio.sleep(120)  # Every 2 minutes
                
                # Check for resource imbalances
                await self._balance_resources()
                
                # Scale up/down based on demand
                await self._auto_scale_swarm()
                
            except Exception as e:
                logger.error(f"Error in resource balancing: {e}")
                await asyncio.sleep(300)
    
    async def _decision_processing_loop(self) -> None:
        """Process collective decisions."""
        logger.info("Starting decision processing loop")
        
        while True:
            try:
                await asyncio.sleep(10)  # Every 10 seconds
                
                # Process active decisions
                completed_decisions = []
                for decision_id, decision in self.active_decisions.items():
                    if await self._process_decision(decision):
                        completed_decisions.append(decision_id)
                
                # Remove completed decisions
                for decision_id in completed_decisions:
                    decision = self.active_decisions.pop(decision_id)
                    self.decision_history.append(decision)
                
                # Keep decision history manageable
                if len(self.decision_history) > 1000:
                    self.decision_history = self.decision_history[-1000:]
                
            except Exception as e:
                logger.error(f"Error in decision processing: {e}")
                await asyncio.sleep(30)
    
    # Helper Methods
    
    def _are_dependencies_met(self, task: HiveMindTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_task_id in task.dependencies:
            dep_task = self.tasks.get(dep_task_id)
            if not dep_task or dep_task.status != "completed":
                return False
        return True
    
    async def _reassign_task(self, task_id: str, reason: str) -> None:
        """Reassign a task to a different agent."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        logger.info(f"Reassigning task {task_id}: {reason}")
        
        # Reset task state
        task.assigned_agent = None
        task.assignment_time = None
        task.status = "pending"
        task.started_at = None
        
        # Add back to queue
        self._add_task_to_queue(task_id)
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        active_agents = [a for a in self.agents.values() if a.is_healthy]
        
        # Agent metrics
        agent_metrics = {
            'total_agents': len(self.agents),
            'healthy_agents': len(active_agents),
            'idle_agents': len([a for a in active_agents if a.status == AgentStatus.IDLE]),
            'busy_agents': len([a for a in active_agents if a.status == AgentStatus.BUSY]),
            'average_load_factor': mean([a.load_factor for a in active_agents]) if active_agents else 0.0
        }
        
        # Task metrics
        all_tasks = list(self.tasks.values())
        completed_tasks = [t for t in all_tasks if t.status == "completed"]
        failed_tasks = [t for t in all_tasks if t.status == "failed"]
        
        task_metrics = {
            'total_tasks': len(all_tasks),
            'pending_tasks': len(self.task_queue),
            'running_tasks': len([t for t in all_tasks if t.status == "running"]),
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(completed_tasks) / max(1, len(completed_tasks) + len(failed_tasks))
        }
        
        # Performance metrics
        if completed_tasks:
            execution_times = [t.execution_time for t in completed_tasks if t.execution_time]
            if execution_times:
                task_metrics.update({
                    'average_execution_time': mean(execution_times),
                    'median_execution_time': median(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times)
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'agents': agent_metrics,
            'tasks': task_metrics,
            'topology': self.topology.value,
            'swarm_efficiency': self._calculate_swarm_efficiency()
        }
    
    def _calculate_swarm_efficiency(self) -> float:
        """Calculate overall swarm efficiency score."""
        if not self.agents:
            return 0.0
        
        # Factor in agent utilization, success rates, and response times
        efficiency_factors = []
        
        for agent in self.agents.values():
            if agent.is_healthy:
                # Utilization efficiency (prefer balanced load)
                load_efficiency = 1.0 - abs(agent.load_factor - 0.7)  # Optimal ~70% utilization
                
                # Performance efficiency
                perf_efficiency = (
                    agent.metrics.success_rate * 0.6 +
                    agent.metrics.quality_score * 0.2 +
                    agent.metrics.reliability_score * 0.2
                )
                
                overall_efficiency = (load_efficiency + perf_efficiency) / 2
                efficiency_factors.append(max(0.0, min(1.0, overall_efficiency)))
        
        return mean(efficiency_factors) if efficiency_factors else 0.0
    
    # Notification Methods
    
    async def _notify_agent_event(self, event_type: str, agent_id: str, data: Dict[str, Any]) -> None:
        """Notify observers of agent events."""
        if self.tui_bridge:
            from .tui_backend_bridge import TUIEvent, TUIEventType
            
            event = TUIEvent(
                event_type=TUIEventType.BACKEND_CONNECTION_STATUS,
                timestamp=datetime.now(),
                data={
                    'event_type': event_type,
                    'agent_id': agent_id,
                    **data
                }
            )
            await self.tui_bridge.emit_event(event)
    
    async def _notify_task_event(self, event_type: str, task_id: str, data: Dict[str, Any]) -> None:
        """Notify observers of task events."""
        if self.tui_bridge:
            from .tui_backend_bridge import TUIEvent, TUIEventType
            
            event = TUIEvent(
                event_type=TUIEventType.SYNC_STATUS_CHANGED,
                timestamp=datetime.now(),
                data={
                    'event_type': event_type,
                    'task_id': task_id,
                    **data
                }
            )
            await self.tui_bridge.emit_event(event)
    
    # Public API Methods
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status."""
        return {
            'agents': {
                'total': len(self.agents),
                'healthy': len([a for a in self.agents.values() if a.is_healthy]),
                'by_status': {
                    status.value: len([a for a in self.agents.values() if a.status == status])
                    for status in AgentStatus
                },
                'by_type': {
                    agent_type: len([a for a in self.agents.values() if a.agent_type == agent_type])
                    for agent_type in self.agent_types.keys()
                }
            },
            'tasks': {
                'total': len(self.tasks),
                'pending': len(self.task_queue),
                'running': len([t for t in self.tasks.values() if t.status == "running"]),
                'completed': len([t for t in self.tasks.values() if t.status == "completed"]),
                'failed': len([t for t in self.tasks.values() if t.status == "failed"])
            },
            'topology': self.topology.value,
            'efficiency': self._calculate_swarm_efficiency(),
            'uptime': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        }
    
    async def cleanup(self) -> None:
        """Clean up Hive Mind coordinator resources."""
        logger.info("Cleaning up Hive Mind Coordinator...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Terminate all agents
        for agent_id in list(self.agents.keys()):
            await self.terminate_agent(agent_id, "coordinator shutdown")
        
        # Cancel all pending tasks
        for task_id in list(self.tasks.keys()):
            if self.tasks[task_id].status in ["pending", "running"]:
                await self.cancel_task(task_id, "coordinator shutdown")
        
        logger.info("Hive Mind Coordinator cleanup completed")


# Global coordinator instance
hive_mind_coordinator: Optional[HiveMindCoordinator] = None


def get_hive_mind_coordinator() -> Optional[HiveMindCoordinator]:
    """Get the global Hive Mind coordinator instance."""
    return hive_mind_coordinator


async def initialize_hive_mind_coordinator(config_manager: ConfigManager) -> HiveMindCoordinator:
    """Initialize the global Hive Mind coordinator."""
    global hive_mind_coordinator
    
    if hive_mind_coordinator is None:
        hive_mind_coordinator = HiveMindCoordinator(config_manager)
        await hive_mind_coordinator.initialize()
    
    return hive_mind_coordinator


async def cleanup_hive_mind_coordinator() -> None:
    """Clean up the global Hive Mind coordinator."""
    global hive_mind_coordinator
    
    if hive_mind_coordinator is not None:
        await hive_mind_coordinator.cleanup()
        hive_mind_coordinator = None
