"""Swarm Manager Module

Advanced swarm management system for Claude Flow integration with:
- Dynamic agent lifecycle management
- Intelligent swarm topology optimization
- Resource allocation and capacity planning
- Health monitoring and recovery mechanisms
- Performance-based scaling decisions
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set, Callable
from pathlib import Path
import aiohttp
import psutil

from ..integrations.claude_flow import SwarmManager as BaseSwarmManager, SwarmConfig, SwarmTopology, AgentType, Agent
from ..core.exceptions import SwarmError, AgentError, ResourceError
from .performance_monitor import PerformanceMonitor, MetricType

logger = logging.getLogger(__name__)


class SwarmHealthStatus(Enum):
    """Swarm health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


class AgentLifecycleState(Enum):
    """Agent lifecycle states"""
    SPAWNING = "spawning"
    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    IDLE = "idle"
    OVERLOADED = "overloaded"
    FAILING = "failing"
    TERMINATED = "terminated"


class ResourceType(Enum):
    """Resource types for allocation"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics"""
    agent_id: str
    spawn_time: datetime
    last_activity: datetime = field(default_factory=datetime.utcnow)
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    health_score: float = 1.0
    specialization_scores: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate"""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 1.0
        return self.tasks_completed / total_tasks
    
    @property
    def uptime_hours(self) -> float:
        """Calculate agent uptime in hours"""
        return (datetime.utcnow() - self.spawn_time).total_seconds() / 3600


@dataclass
class SwarmResourceUsage:
    """Swarm resource usage tracking"""
    swarm_id: str
    cpu_limit: float = 100.0  # percentage
    memory_limit: float = 1024.0  # MB
    current_cpu: float = 0.0
    current_memory: float = 0.0
    peak_cpu: float = 0.0
    peak_memory: float = 0.0
    agent_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def cpu_utilization(self) -> float:
        """CPU utilization percentage"""
        return (self.current_cpu / self.cpu_limit) * 100 if self.cpu_limit > 0 else 0
    
    @property
    def memory_utilization(self) -> float:
        """Memory utilization percentage"""
        return (self.current_memory / self.memory_limit) * 100 if self.memory_limit > 0 else 0


@dataclass
class SwarmConfiguration:
    """Enhanced swarm configuration"""
    swarm_id: str
    topology: SwarmTopology
    max_agents: int = 10
    min_agents: int = 1
    auto_scaling: bool = True
    health_check_interval: int = 30
    resource_limits: Dict[ResourceType, float] = field(default_factory=dict)
    agent_types_allowed: Set[AgentType] = field(default_factory=set)
    specialization_focus: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class EnhancedSwarmManager:
    """
    Enhanced Swarm Manager with advanced lifecycle management
    
    Features:
    - Intelligent agent lifecycle management
    - Dynamic topology optimization
    - Resource-aware scaling decisions
    - Health monitoring and recovery
    - Performance-based agent selection
    - Specialization tracking and optimization
    """
    
    def __init__(
        self,
        base_manager: Optional[BaseSwarmManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        max_swarms: int = 20,
        default_health_check_interval: int = 30,
        resource_monitoring_enabled: bool = True
    ):
        self.base_manager = base_manager or BaseSwarmManager()
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.max_swarms = max_swarms
        self.default_health_check_interval = default_health_check_interval
        self.resource_monitoring_enabled = resource_monitoring_enabled
        
        # Enhanced tracking
        self.swarm_configurations: Dict[str, SwarmConfiguration] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.swarm_health: Dict[str, SwarmHealthStatus] = {}
        self.resource_usage: Dict[str, SwarmResourceUsage] = {}
        
        # Agent lifecycle management
        self.agent_states: Dict[str, AgentLifecycleState] = {}
        self.agent_assignment_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.agent_specializations: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Task queue and scheduling
        self.agent_task_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.pending_spawns: Dict[str, datetime] = {}
        self.termination_queue: Set[str] = set()
        
        # Performance and optimization
        self.topology_performance_history: Dict[SwarmTopology, List[float]] = defaultdict(list)
        self.agent_selection_cache: Dict[str, List[str]] = {}
        self.optimization_recommendations: Dict[str, List[str]] = defaultdict(list)
        
        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.resource_monitor_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure enhanced logging"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    async def initialize(self):
        """Initialize the enhanced swarm manager"""
        logger.info("Initializing Enhanced Swarm Manager")
        
        try:
            # Initialize base manager
            if hasattr(self.base_manager, 'initialize'):
                await self.base_manager.initialize()
            
            # Start performance monitoring
            await self.performance_monitor.start()
            
            # Start background tasks
            self.health_monitor_task = asyncio.create_task(self._health_monitoring_loop())
            
            if self.resource_monitoring_enabled:
                self.resource_monitor_task = asyncio.create_task(self._resource_monitoring_loop())
            
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Enhanced Swarm Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Swarm Manager: {e}")
            raise SwarmError(f"Initialization failed: {e}") from e

    async def create_optimized_swarm(
        self,
        project_context: Dict[str, Any],
        preferred_topology: Optional[SwarmTopology] = None,
        resource_limits: Optional[Dict[ResourceType, float]] = None
    ) -> str:
        """Create optimized swarm based on project context"""
        
        swarm_id = f"swarm-{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Creating optimized swarm {swarm_id} for project context")
        
        try:
            # Analyze project requirements
            complexity_score = self._calculate_project_complexity(project_context)
            optimal_topology = preferred_topology or self._select_optimal_topology(complexity_score)
            
            # Determine resource requirements
            if resource_limits is None:
                resource_limits = self._calculate_resource_requirements(project_context, complexity_score)
            
            # Create base swarm configuration
            base_config = SwarmConfig(
                topology=optimal_topology,
                max_agents=min(complexity_score // 10 + 3, 15),
                strategy="adaptive",
                enable_coordination=True,
                enable_learning=True,
                auto_scaling=True
            )
            
            # Initialize with base manager
            await self.base_manager.initialize_swarm(base_config, swarm_id)
            
            # Create enhanced configuration
            enhanced_config = SwarmConfiguration(
                swarm_id=swarm_id,
                topology=optimal_topology,
                max_agents=base_config.max_agents,
                min_agents=max(1, base_config.max_agents // 3),
                auto_scaling=True,
                resource_limits=resource_limits,
                specialization_focus=project_context.get('primary_domain'),
                agent_types_allowed=self._determine_required_agent_types(project_context)
            )
            
            # Store configurations
            self.swarm_configurations[swarm_id] = enhanced_config
            self.swarm_health[swarm_id] = SwarmHealthStatus.HEALTHY
            self.resource_usage[swarm_id] = SwarmResourceUsage(
                swarm_id=swarm_id,
                cpu_limit=resource_limits.get(ResourceType.CPU, 100.0),
                memory_limit=resource_limits.get(ResourceType.MEMORY, 1024.0)
            )
            
            # Spawn initial agents
            initial_agents = self._determine_initial_agents(project_context, enhanced_config)
            for agent_type in initial_agents:
                await self.spawn_intelligent_agent(swarm_id, agent_type, project_context)
            
            logger.info(f"Optimized swarm {swarm_id} created with {len(initial_agents)} initial agents")
            
            return swarm_id
            
        except Exception as e:
            logger.error(f"Failed to create optimized swarm: {e}")
            raise SwarmError(f"Swarm creation failed: {e}") from e

    async def spawn_intelligent_agent(
        self,
        swarm_id: str,
        agent_type: AgentType,
        context: Optional[Dict[str, Any]] = None,
        custom_capabilities: Optional[List[str]] = None
    ) -> str:
        """Spawn agent with intelligent configuration"""
        
        if swarm_id not in self.swarm_configurations:
            raise SwarmError(f"Swarm {swarm_id} not found")
        
        config = self.swarm_configurations[swarm_id]
        
        # Check resource limits
        if not await self._check_resource_availability(swarm_id, agent_type):
            raise ResourceError(f"Insufficient resources to spawn {agent_type.value} in swarm {swarm_id}")
        
        # Check agent type restrictions
        if config.agent_types_allowed and agent_type not in config.agent_types_allowed:
            logger.warning(f"Agent type {agent_type.value} not in allowed types for swarm {swarm_id}")
        
        try:
            agent_id = f"agent-{uuid.uuid4().hex[:8]}"
            
            # Mark as spawning
            self.agent_states[agent_id] = AgentLifecycleState.SPAWNING
            self.pending_spawns[agent_id] = datetime.utcnow()
            
            # Enhanced agent configuration
            agent_capabilities = custom_capabilities or self._generate_agent_capabilities(
                agent_type, context, config.specialization_focus
            )
            
            # Spawn via base manager
            agent = await self.base_manager.spawn_agent(
                swarm_id, 
                agent_type, 
                name=f"{agent_type.value}-{agent_id[-4:]}",
                capabilities=agent_capabilities
            )
            
            # Initialize enhanced tracking
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                spawn_time=datetime.utcnow()
            )
            
            self.agent_states[agent_id] = AgentLifecycleState.INITIALIZING
            self.agent_task_queues[agent_id] = asyncio.Queue()
            
            # Update resource usage
            await self._update_resource_usage(swarm_id)
            
            # Start agent initialization
            asyncio.create_task(self._initialize_agent(agent_id, swarm_id, agent_type, context))
            
            # Record performance metric
            self.performance_monitor.record_metric(
                "agent_spawned",
                1,
                MetricType.CUSTOM,
                {"swarm_id": swarm_id, "agent_type": agent_type.value}
            )
            
            logger.info(f"Spawned intelligent agent {agent_id} of type {agent_type.value} in swarm {swarm_id}")
            
            return agent_id
            
        except Exception as e:
            # Cleanup on failure
            self.agent_states.pop(agent_id, None)
            self.pending_spawns.pop(agent_id, None)
            
            logger.error(f"Failed to spawn agent in swarm {swarm_id}: {e}")
            raise AgentError(f"Agent spawn failed: {e}") from e

    async def assign_task_to_best_agent(
        self,
        swarm_id: str,
        task_description: str,
        task_requirements: List[str],
        priority: int = 5
    ) -> Optional[str]:
        """Assign task to the best available agent"""
        
        if swarm_id not in self.swarm_configurations:
            raise SwarmError(f"Swarm {swarm_id} not found")
        
        try:
            # Get available agents
            available_agents = await self._get_available_agents(swarm_id)
            
            if not available_agents:
                # Try to spawn appropriate agent
                suitable_type = self._determine_suitable_agent_type(task_requirements)
                if suitable_type:
                    agent_id = await self.spawn_intelligent_agent(
                        swarm_id, 
                        suitable_type, 
                        {"task_requirements": task_requirements}
                    )
                    # Wait briefly for initialization
                    await asyncio.sleep(1)
                    available_agents = [agent_id]
                else:
                    return None
            
            # Score agents for task suitability
            best_agent = None
            best_score = -1
            
            for agent_id in available_agents:
                score = await self._calculate_agent_task_suitability(
                    agent_id, task_description, task_requirements
                )
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
            
            if best_agent and best_score > 0.3:  # Minimum suitability threshold
                # Assign task
                await self._assign_task_to_agent(best_agent, {
                    'description': task_description,
                    'requirements': task_requirements,
                    'priority': priority,
                    'assigned_at': datetime.utcnow().isoformat()
                })
                
                return best_agent
            
            return None
            
        except Exception as e:
            logger.error(f"Task assignment failed in swarm {swarm_id}: {e}")
            raise SwarmError(f"Task assignment failed: {e}") from e

    async def optimize_swarm_topology(self, swarm_id: str) -> bool:
        """Optimize swarm topology based on performance data"""
        
        if swarm_id not in self.swarm_configurations:
            return False
        
        try:
            config = self.swarm_configurations[swarm_id]
            current_topology = config.topology
            
            # Analyze current performance
            current_performance = await self._analyze_swarm_performance(swarm_id)
            
            # Test alternative topologies
            best_topology = current_topology
            best_performance = current_performance
            
            for topology in SwarmTopology:
                if topology == current_topology:
                    continue
                
                # Estimate performance for alternative topology
                estimated_performance = await self._estimate_topology_performance(
                    swarm_id, topology
                )
                
                if estimated_performance > best_performance * 1.1:  # 10% improvement threshold
                    best_topology = topology
                    best_performance = estimated_performance
            
            # Apply topology change if beneficial
            if best_topology != current_topology:
                await self._change_swarm_topology(swarm_id, best_topology)
                
                self.optimization_recommendations[swarm_id].append(
                    f"Optimized topology from {current_topology.value} to {best_topology.value} "
                    f"for {((best_performance / current_performance - 1) * 100):.1f}% improvement"
                )
                
                logger.info(f"Optimized swarm {swarm_id} topology to {best_topology.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Topology optimization failed for swarm {swarm_id}: {e}")
            return False

    async def scale_swarm_intelligently(
        self,
        swarm_id: str,
        target_performance: Optional[float] = None
    ) -> bool:
        """Intelligently scale swarm based on performance requirements"""
        
        if swarm_id not in self.swarm_configurations:
            return False
        
        try:
            config = self.swarm_configurations[swarm_id]
            current_agents = await self._get_agent_count(swarm_id)
            
            # Analyze current load and performance
            load_metrics = await self._analyze_swarm_load(swarm_id)
            performance_metrics = await self._analyze_swarm_performance(swarm_id)
            
            # Determine scaling decision
            should_scale_up = (
                load_metrics.get('utilization', 0) > 0.8 or
                load_metrics.get('queue_depth', 0) > 5 or
                (target_performance and performance_metrics < target_performance * 0.9)
            )
            
            should_scale_down = (
                load_metrics.get('utilization', 1) < 0.3 and
                load_metrics.get('queue_depth', 0) == 0 and
                current_agents > config.min_agents
            )
            
            if should_scale_up and current_agents < config.max_agents:
                # Scale up
                needed_types = await self._determine_scaling_agent_types(swarm_id, "up")
                
                for agent_type in needed_types:
                    if current_agents < config.max_agents:
                        await self.spawn_intelligent_agent(swarm_id, agent_type)
                        current_agents += 1
                
                logger.info(f"Scaled up swarm {swarm_id} by {len(needed_types)} agents")
                return True
            
            elif should_scale_down:
                # Scale down
                agents_to_remove = await self._select_agents_for_removal(swarm_id)
                
                for agent_id in agents_to_remove[:1]:  # Remove one at a time
                    await self.terminate_agent_gracefully(agent_id)
                
                if agents_to_remove:
                    logger.info(f"Scaled down swarm {swarm_id} by {min(1, len(agents_to_remove))} agents")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Intelligent scaling failed for swarm {swarm_id}: {e}")
            return False

    async def terminate_agent_gracefully(self, agent_id: str) -> bool:
        """Gracefully terminate an agent"""
        
        if agent_id not in self.agent_states:
            return False
        
        try:
            # Mark for termination
            self.agent_states[agent_id] = AgentLifecycleState.TERMINATED
            self.termination_queue.add(agent_id)
            
            # Wait for current tasks to complete (with timeout)
            timeout = 30  # 30 seconds
            start_time = time.time()
            
            while (self.agent_task_queues[agent_id].qsize() > 0 and 
                   time.time() - start_time < timeout):
                await asyncio.sleep(1)
            
            # Terminate via base manager
            await self.base_manager.terminate_agent(agent_id)
            
            # Cleanup tracking data
            self._cleanup_agent_data(agent_id)
            
            logger.info(f"Gracefully terminated agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate agent {agent_id}: {e}")
            return False

    async def get_swarm_health_report(self, swarm_id: str) -> Dict[str, Any]:
        """Generate comprehensive swarm health report"""
        
        if swarm_id not in self.swarm_configurations:
            raise SwarmError(f"Swarm {swarm_id} not found")
        
        try:
            config = self.swarm_configurations[swarm_id]
            health_status = self.swarm_health[swarm_id]
            resource_usage = self.resource_usage[swarm_id]
            
            # Get agent health metrics
            swarm_agents = await self._get_swarm_agent_ids(swarm_id)
            agent_health = {}
            
            for agent_id in swarm_agents:
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    agent_health[agent_id] = {
                        'health_score': metrics.health_score,
                        'success_rate': metrics.success_rate,
                        'uptime_hours': metrics.uptime_hours,
                        'tasks_completed': metrics.tasks_completed,
                        'error_count': metrics.error_count,
                        'state': self.agent_states.get(agent_id, 'unknown').value
                    }
            
            # Calculate overall health score
            overall_health = self._calculate_overall_health_score(swarm_id)
            
            # Get performance trends
            performance_history = self.topology_performance_history.get(config.topology, [])
            recent_performance = performance_history[-10:] if performance_history else []
            
            report = {
                'swarm_id': swarm_id,
                'health_status': health_status.value,
                'overall_health_score': overall_health,
                'configuration': {
                    'topology': config.topology.value,
                    'max_agents': config.max_agents,
                    'min_agents': config.min_agents,
                    'auto_scaling': config.auto_scaling,
                    'specialization_focus': config.specialization_focus
                },
                'resource_usage': {
                    'cpu_utilization': resource_usage.cpu_utilization,
                    'memory_utilization': resource_usage.memory_utilization,
                    'current_cpu': resource_usage.current_cpu,
                    'current_memory': resource_usage.current_memory,
                    'peak_cpu': resource_usage.peak_cpu,
                    'peak_memory': resource_usage.peak_memory
                },
                'agent_health': agent_health,
                'agent_count': len(swarm_agents),
                'performance_trends': {
                    'recent_scores': recent_performance,
                    'trend_direction': self._calculate_trend_direction(recent_performance),
                    'average_performance': sum(recent_performance) / len(recent_performance) if recent_performance else 0
                },
                'optimization_recommendations': self.optimization_recommendations.get(swarm_id, []),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate health report for swarm {swarm_id}: {e}")
            raise SwarmError(f"Health report generation failed: {e}") from e

    async def _initialize_agent(
        self,
        agent_id: str,
        swarm_id: str,
        agent_type: AgentType,
        context: Optional[Dict[str, Any]]
    ):
        """Initialize agent with enhanced setup"""
        
        try:
            # Simulate initialization process
            await asyncio.sleep(1)  # Initialization time
            
            # Update state
            self.agent_states[agent_id] = AgentLifecycleState.READY
            
            # Initialize specialization scores based on context
            if context and 'primary_domain' in context:
                domain = context['primary_domain']
                self.agent_specializations[agent_id][domain] = 0.8
            
            # Set initial capabilities based on agent type
            base_capabilities = {
                AgentType.CODER: ['coding', 'debugging', 'testing'],
                AgentType.RESEARCHER: ['research', 'analysis', 'documentation'],
                AgentType.ANALYST: ['analysis', 'optimization', 'reporting'],
                AgentType.OPTIMIZER: ['optimization', 'performance', 'efficiency'],
                AgentType.COORDINATOR: ['coordination', 'planning', 'communication']
            }
            
            for capability in base_capabilities.get(agent_type, []):
                self.agent_specializations[agent_id][capability] = 0.7
            
            # Remove from pending spawns
            self.pending_spawns.pop(agent_id, None)
            
            logger.info(f"Agent {agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Agent initialization failed for {agent_id}: {e}")
            self.agent_states[agent_id] = AgentLifecycleState.FAILED

    def _calculate_project_complexity(self, project_context: Dict[str, Any]) -> int:
        """Calculate project complexity score"""
        
        complexity = 0
        
        # Base complexity factors
        complexity += len(project_context.get('features', [])) * 5
        complexity += len(project_context.get('integrations', [])) * 8
        complexity += len(project_context.get('dependencies', [])) * 3
        
        # Domain complexity
        domain_weights = {
            'web_development': 10,
            'data_science': 15,
            'machine_learning': 20,
            'system_architecture': 18,
            'devops': 12
        }
        
        primary_domain = project_context.get('primary_domain', 'web_development')
        complexity += domain_weights.get(primary_domain, 10)
        
        # Timeline pressure
        timeline_days = project_context.get('timeline_days', 30)
        if timeline_days < 7:
            complexity += 20
        elif timeline_days < 14:
            complexity += 10
        
        # Team size factor
        team_size = project_context.get('team_size', 1)
        if team_size > 5:
            complexity += team_size * 2
        
        return min(100, max(10, complexity))

    def _select_optimal_topology(self, complexity_score: int) -> SwarmTopology:
        """Select optimal topology based on complexity"""
        
        if complexity_score < 30:
            return SwarmTopology.STAR
        elif complexity_score < 60:
            return SwarmTopology.HIERARCHICAL
        else:
            return SwarmTopology.MESH

    def _calculate_resource_requirements(
        self,
        project_context: Dict[str, Any],
        complexity_score: int
    ) -> Dict[ResourceType, float]:
        """Calculate resource requirements for project"""
        
        base_cpu = 50.0  # 50% CPU per swarm
        base_memory = 512.0  # 512MB per swarm
        
        # Scale with complexity
        complexity_multiplier = 1 + (complexity_score / 100.0)
        
        cpu_limit = base_cpu * complexity_multiplier
        memory_limit = base_memory * complexity_multiplier
        
        # Domain-specific adjustments
        domain_multipliers = {
            'machine_learning': {'cpu': 1.8, 'memory': 2.5},
            'data_science': {'cpu': 1.5, 'memory': 2.0},
            'system_architecture': {'cpu': 1.3, 'memory': 1.5},
            'web_development': {'cpu': 1.0, 'memory': 1.0}
        }
        
        domain = project_context.get('primary_domain', 'web_development')
        multipliers = domain_multipliers.get(domain, {'cpu': 1.0, 'memory': 1.0})
        
        return {
            ResourceType.CPU: cpu_limit * multipliers['cpu'],
            ResourceType.MEMORY: memory_limit * multipliers['memory'],
            ResourceType.NETWORK: 100.0,  # 100 Mbps
            ResourceType.STORAGE: 1024.0  # 1GB
        }

    def _determine_required_agent_types(self, project_context: Dict[str, Any]) -> Set[AgentType]:
        """Determine required agent types for project"""
        
        required_types = {AgentType.COORDINATOR}  # Always need coordination
        
        domain = project_context.get('primary_domain', 'web_development')
        
        domain_agents = {
            'web_development': {AgentType.CODER, AgentType.SYSTEM_ARCHITECT},
            'data_science': {AgentType.ANALYST, AgentType.RESEARCHER},
            'machine_learning': {AgentType.RESEARCHER, AgentType.OPTIMIZER},
            'system_architecture': {AgentType.SYSTEM_ARCHITECT, AgentType.CODER},
            'devops': {AgentType.OPTIMIZER, AgentType.SYSTEM_ARCHITECT}
        }
        
        required_types.update(domain_agents.get(domain, {AgentType.CODER}))
        
        # Add based on project features
        features = project_context.get('features', [])
        
        if any('test' in f.lower() for f in features):
            required_types.add(AgentType.TEST_ENGINEER)
        
        if any('optim' in f.lower() for f in features):
            required_types.add(AgentType.OPTIMIZER)
        
        if any('research' in f.lower() or 'analysis' in f.lower() for f in features):
            required_types.add(AgentType.RESEARCHER)
        
        return required_types

    def _determine_initial_agents(
        self,
        project_context: Dict[str, Any],
        config: SwarmConfiguration
    ) -> List[AgentType]:
        """Determine initial agents to spawn"""
        
        required_types = list(config.agent_types_allowed or self._determine_required_agent_types(project_context))
        
        # Limit to reasonable initial count
        max_initial = min(config.max_agents // 2 + 1, 5)
        
        # Prioritize by importance
        priority_order = [
            AgentType.COORDINATOR,
            AgentType.SYSTEM_ARCHITECT,
            AgentType.CODER,
            AgentType.ANALYST,
            AgentType.RESEARCHER,
            AgentType.OPTIMIZER,
            AgentType.TEST_ENGINEER
        ]
        
        initial_agents = []
        for agent_type in priority_order:
            if agent_type in required_types and len(initial_agents) < max_initial:
                initial_agents.append(agent_type)
        
        # Ensure at least one agent
        if not initial_agents:
            initial_agents.append(AgentType.CODER)
        
        return initial_agents

    async def _check_resource_availability(self, swarm_id: str, agent_type: AgentType) -> bool:
        """Check if resources are available for new agent"""
        
        if swarm_id not in self.resource_usage:
            return True
        
        usage = self.resource_usage[swarm_id]
        
        # Estimate resource requirements for new agent
        agent_cpu_cost = 15.0  # 15% CPU per agent
        agent_memory_cost = 128.0  # 128MB per agent
        
        # Check CPU availability
        if usage.current_cpu + agent_cpu_cost > usage.cpu_limit:
            logger.warning(f"Insufficient CPU for new agent in swarm {swarm_id}")
            return False
        
        # Check memory availability
        if usage.current_memory + agent_memory_cost > usage.memory_limit:
            logger.warning(f"Insufficient memory for new agent in swarm {swarm_id}")
            return False
        
        return True

    def _generate_agent_capabilities(
        self,
        agent_type: AgentType,
        context: Optional[Dict[str, Any]],
        specialization_focus: Optional[str]
    ) -> List[str]:
        """Generate context-aware agent capabilities"""
        
        base_capabilities = {
            AgentType.CODER: ['coding', 'debugging', 'testing', 'documentation'],
            AgentType.RESEARCHER: ['research', 'analysis', 'documentation', 'investigation'],
            AgentType.ANALYST: ['analysis', 'reporting', 'optimization', 'metrics'],
            AgentType.OPTIMIZER: ['optimization', 'performance', 'efficiency', 'benchmarking'],
            AgentType.COORDINATOR: ['coordination', 'planning', 'communication', 'management'],
            AgentType.SYSTEM_ARCHITECT: ['architecture', 'design', 'integration', 'scalability']
        }
        
        capabilities = base_capabilities.get(agent_type, ['general'])
        
        # Add specialization focus
        if specialization_focus:
            capabilities.append(specialization_focus)
        
        # Add context-specific capabilities
        if context:
            domain = context.get('primary_domain')
            if domain:
                capabilities.append(domain)
            
            requirements = context.get('task_requirements', [])
            capabilities.extend(requirements)
        
        # Remove duplicates and limit count
        capabilities = list(set(capabilities))[:8]
        
        return capabilities

    async def _get_available_agents(self, swarm_id: str) -> List[str]:
        """Get list of available agents in swarm"""
        
        available_agents = []
        
        # Get all agents in swarm
        swarm_agent_ids = await self._get_swarm_agent_ids(swarm_id)
        
        for agent_id in swarm_agent_ids:
            state = self.agent_states.get(agent_id)
            
            if state in [AgentLifecycleState.READY, AgentLifecycleState.IDLE]:
                available_agents.append(agent_id)
        
        return available_agents

    async def _get_swarm_agent_ids(self, swarm_id: str) -> List[str]:
        """Get all agent IDs for a swarm"""
        
        # This would integrate with the base manager to get actual agent IDs
        # For now, return tracked agent IDs
        swarm_agents = []
        
        for agent_id, metrics in self.agent_metrics.items():
            # Check if agent belongs to swarm (would need proper tracking)
            if agent_id in self.agent_states and self.agent_states[agent_id] != AgentLifecycleState.TERMINATED:
                swarm_agents.append(agent_id)
        
        return swarm_agents

    def _determine_suitable_agent_type(self, task_requirements: List[str]) -> Optional[AgentType]:
        """Determine suitable agent type for task requirements"""
        
        requirement_mappings = {
            'coding': AgentType.CODER,
            'development': AgentType.CODER,
            'research': AgentType.RESEARCHER,
            'analysis': AgentType.ANALYST,
            'optimization': AgentType.OPTIMIZER,
            'coordination': AgentType.COORDINATOR,
            'architecture': AgentType.SYSTEM_ARCHITECT,
            'testing': AgentType.TEST_ENGINEER
        }
        
        for requirement in task_requirements:
            req_lower = requirement.lower()
            for keyword, agent_type in requirement_mappings.items():
                if keyword in req_lower:
                    return agent_type
        
        return AgentType.CODER  # Default fallback

    async def _calculate_agent_task_suitability(
        self,
        agent_id: str,
        task_description: str,
        task_requirements: List[str]
    ) -> float:
        """Calculate agent suitability score for task"""
        
        if agent_id not in self.agent_metrics:
            return 0.5  # Default score for unknown agents
        
        metrics = self.agent_metrics[agent_id]
        specializations = self.agent_specializations[agent_id]
        
        # Base score from success rate and health
        base_score = (metrics.success_rate + metrics.health_score) / 2
        
        # Specialization matching
        specialization_score = 0.0
        for requirement in task_requirements:
            req_lower = requirement.lower()
            
            for specialization, score in specializations.items():
                if req_lower in specialization.lower() or specialization.lower() in req_lower:
                    specialization_score = max(specialization_score, score)
        
        # Workload factor (prefer less busy agents)
        workload_factor = 1.0
        queue_size = self.agent_task_queues[agent_id].qsize()
        if queue_size > 0:
            workload_factor = 1.0 / (1 + queue_size * 0.2)
        
        # Combined score
        total_score = (base_score * 0.4 + specialization_score * 0.4 + workload_factor * 0.2)
        
        return min(1.0, max(0.0, total_score))

    async def _assign_task_to_agent(self, agent_id: str, task_data: Dict[str, Any]):
        """Assign task to specific agent"""
        
        if agent_id not in self.agent_task_queues:
            raise AgentError(f"Agent {agent_id} not found")
        
        # Add to agent's task queue
        await self.agent_task_queues[agent_id].put(task_data)
        
        # Update agent state
        self.agent_states[agent_id] = AgentLifecycleState.WORKING
        
        # Record assignment
        self.agent_assignment_history[agent_id].append({
            'task_description': task_data.get('description', ''),
            'requirements': task_data.get('requirements', []),
            'priority': task_data.get('priority', 5),
            'assigned_at': task_data.get('assigned_at')
        })
        
        logger.info(f"Assigned task to agent {agent_id}")

    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        
        while True:
            try:
                for swarm_id in list(self.swarm_configurations.keys()):
                    await self._monitor_swarm_health(swarm_id)
                    await self._monitor_agent_health(swarm_id)
                
                await asyncio.sleep(self.default_health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(self.default_health_check_interval)

    async def _resource_monitoring_loop(self):
        """Background resource monitoring loop"""
        
        while True:
            try:
                for swarm_id in list(self.swarm_configurations.keys()):
                    await self._update_resource_usage(swarm_id)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def _optimization_loop(self):
        """Background optimization loop"""
        
        while True:
            try:
                for swarm_id in list(self.swarm_configurations.keys()):
                    await self.optimize_swarm_topology(swarm_id)
                    await self.scale_swarm_intelligently(swarm_id)
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        
        while True:
            try:
                # Cleanup terminated agents
                for agent_id in list(self.termination_queue):
                    self._cleanup_agent_data(agent_id)
                    self.termination_queue.remove(agent_id)
                
                # Cleanup old assignment history
                for agent_id in list(self.agent_assignment_history.keys()):
                    history = self.agent_assignment_history[agent_id]
                    if len(history) > 100:  # Keep only recent 100 assignments
                        self.agent_assignment_history[agent_id] = history[-100:]
                
                await asyncio.sleep(600)  # Cleanup every 10 minutes
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(600)

    def _cleanup_agent_data(self, agent_id: str):
        """Cleanup all data for terminated agent"""
        
        self.agent_states.pop(agent_id, None)
        self.agent_metrics.pop(agent_id, None)
        self.agent_specializations.pop(agent_id, None)
        self.agent_task_queues.pop(agent_id, None)
        self.agent_assignment_history.pop(agent_id, None)

    async def _monitor_swarm_health(self, swarm_id: str):
        """Monitor individual swarm health"""
        
        try:
            if swarm_id not in self.swarm_configurations:
                return
            
            # Calculate health metrics
            overall_health = self._calculate_overall_health_score(swarm_id)
            
            # Update health status
            if overall_health > 0.8:
                self.swarm_health[swarm_id] = SwarmHealthStatus.HEALTHY
            elif overall_health > 0.6:
                self.swarm_health[swarm_id] = SwarmHealthStatus.DEGRADED
            elif overall_health > 0.3:
                self.swarm_health[swarm_id] = SwarmHealthStatus.CRITICAL
            else:
                self.swarm_health[swarm_id] = SwarmHealthStatus.FAILED
            
            # Record performance metric
            self.performance_monitor.record_metric(
                f"swarm_health_{swarm_id}",
                overall_health,
                MetricType.CUSTOM
            )
            
        except Exception as e:
            logger.error(f"Swarm health monitoring failed for {swarm_id}: {e}")

    async def _monitor_agent_health(self, swarm_id: str):
        """Monitor health of agents in swarm"""
        
        try:
            swarm_agent_ids = await self._get_swarm_agent_ids(swarm_id)
            
            for agent_id in swarm_agent_ids:
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    
                    # Update health based on various factors
                    health_factors = []
                    
                    # Success rate factor
                    health_factors.append(metrics.success_rate)
                    
                    # Error rate factor (inverted)
                    total_tasks = metrics.tasks_completed + metrics.tasks_failed
                    error_rate = metrics.error_count / max(total_tasks, 1)
                    health_factors.append(1.0 - min(error_rate, 1.0))
                    
                    # Responsiveness factor (based on last activity)
                    time_since_activity = (datetime.utcnow() - metrics.last_activity).total_seconds()
                    responsiveness = max(0.0, 1.0 - time_since_activity / 3600)  # Decay over 1 hour
                    health_factors.append(responsiveness)
                    
                    # Calculate overall health
                    metrics.health_score = sum(health_factors) / len(health_factors)
                    
        except Exception as e:
            logger.error(f"Agent health monitoring failed for swarm {swarm_id}: {e}")

    async def _update_resource_usage(self, swarm_id: str):
        """Update resource usage for swarm"""
        
        try:
            if swarm_id not in self.resource_usage:
                return
            
            usage = self.resource_usage[swarm_id]
            swarm_agent_ids = await self._get_swarm_agent_ids(swarm_id)
            
            # Calculate current resource usage
            total_cpu = 0.0
            total_memory = 0.0
            
            for agent_id in swarm_agent_ids:
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    total_cpu += metrics.cpu_usage
                    total_memory += metrics.memory_usage
            
            usage.current_cpu = total_cpu
            usage.current_memory = total_memory
            usage.agent_count = len(swarm_agent_ids)
            usage.last_updated = datetime.utcnow()
            
            # Update peaks
            usage.peak_cpu = max(usage.peak_cpu, total_cpu)
            usage.peak_memory = max(usage.peak_memory, total_memory)
            
        except Exception as e:
            logger.error(f"Resource usage update failed for swarm {swarm_id}: {e}")

    def _calculate_overall_health_score(self, swarm_id: str) -> float:
        """Calculate overall health score for swarm"""
        
        try:
            if swarm_id not in self.swarm_configurations:
                return 0.0
            
            health_scores = []
            
            # Get agent health scores
            for agent_id, metrics in self.agent_metrics.items():
                if self.agent_states.get(agent_id) != AgentLifecycleState.TERMINATED:
                    health_scores.append(metrics.health_score)
            
            if not health_scores:
                return 0.5  # Default for no agents
            
            # Resource utilization factor
            if swarm_id in self.resource_usage:
                usage = self.resource_usage[swarm_id]
                cpu_factor = 1.0 - min(usage.cpu_utilization / 100.0, 1.0)
                memory_factor = 1.0 - min(usage.memory_utilization / 100.0, 1.0)
                resource_factor = (cpu_factor + memory_factor) / 2
            else:
                resource_factor = 1.0
            
            # Combined health score
            agent_health = sum(health_scores) / len(health_scores)
            overall_health = (agent_health * 0.7 + resource_factor * 0.3)
            
            return min(1.0, max(0.0, overall_health))
            
        except Exception as e:
            logger.error(f"Health score calculation failed for swarm {swarm_id}: {e}")
            return 0.5

    async def shutdown(self):
        """Gracefully shutdown the enhanced swarm manager"""
        
        logger.info("Shutting down Enhanced Swarm Manager")
        
        try:
            # Cancel background tasks
            for task in [self.health_monitor_task, self.resource_monitor_task, 
                        self.optimization_task, self.cleanup_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Terminate all agents
            for agent_id in list(self.agent_states.keys()):
                if self.agent_states[agent_id] != AgentLifecycleState.TERMINATED:
                    await self.terminate_agent_gracefully(agent_id)
            
            # Stop performance monitoring
            await self.performance_monitor.stop()
            
            # Shutdown base manager
            if hasattr(self.base_manager, 'shutdown'):
                await self.base_manager.shutdown()
            
            logger.info("Enhanced Swarm Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Enhanced Swarm Manager shutdown: {e}")