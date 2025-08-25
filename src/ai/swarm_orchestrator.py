"""Swarm Orchestrator Module

Advanced swarm orchestration for Claude Flow integration with:
- Dynamic topology selection and optimization
- Agent lifecycle management and coordination
- Real-time monitoring and performance tracking
- Load balancing and auto-scaling capabilities
- Memory management and state persistence
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Callable
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil

from ..integrations.claude_flow import (
    SwarmManager, SwarmConfig, SwarmTopology, AgentType, 
    WorkflowStatus, ClaudeFlowOrchestrator
)
from ..core.exceptions import SwarmError, AgentError, OrchestrationError

logger = logging.getLogger(__name__)


class SwarmState(Enum):
    """Swarm operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    SCALING = "scaling"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    ERROR = "error"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for task distribution"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPACITY_BASED = "capacity_based"
    INTELLIGENT = "intelligent"


@dataclass
class SwarmMetrics:
    """Comprehensive swarm performance metrics"""
    swarm_id: str
    topology: str
    total_agents: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    busy_agents: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0  # tasks per minute
    efficiency_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def utilization_rate(self) -> float:
        if self.total_agents == 0:
            return 0.0
        return (self.busy_agents / self.total_agents) * 100


@dataclass
class TaskRequest:
    """Task request for swarm execution"""
    task_id: str
    description: str
    priority: str = "medium"  # low, medium, high, critical
    agent_requirements: List[str] = field(default_factory=list)
    estimated_complexity: int = 5  # 1-10 scale
    max_execution_time: int = 300  # seconds
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class SwarmOrchestrator:
    """
    Advanced Swarm Orchestrator with intelligent coordination
    
    Features:
    - Multi-topology swarm management
    - Dynamic agent provisioning and scaling
    - Intelligent task distribution and load balancing
    - Real-time performance monitoring
    - Adaptive optimization based on workload patterns
    - Fault tolerance and recovery mechanisms
    """
    
    def __init__(
        self,
        max_swarms: int = 5,
        max_agents_per_swarm: int = 10,
        enable_auto_scaling: bool = True,
        monitoring_interval: int = 30,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT
    ):
        self.max_swarms = max_swarms
        self.max_agents_per_swarm = max_agents_per_swarm
        self.enable_auto_scaling = enable_auto_scaling
        self.monitoring_interval = monitoring_interval
        self.load_balancing_strategy = load_balancing_strategy
        
        # Core components
        self.swarm_manager = SwarmManager()
        self.orchestrator = ClaudeFlowOrchestrator(self.swarm_manager)
        
        # Swarm tracking
        self.active_swarms: Dict[str, SwarmState] = {}
        self.swarm_configs: Dict[str, SwarmConfig] = {}
        self.swarm_metrics: Dict[str, SwarmMetrics] = {}
        self.swarm_tasks: Dict[str, List[TaskRequest]] = {}
        
        # Task queue and execution
        self.pending_tasks: Dict[str, TaskRequest] = {}
        self.executing_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.global_metrics = {
            'total_swarms_created': 0,
            'total_tasks_processed': 0,
            'average_swarm_efficiency': 0.0,
            'peak_concurrent_tasks': 0,
            'system_uptime': datetime.utcnow()
        }
        
        # Monitoring and optimization
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for swarm orchestration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    async def initialize_swarm(
        self,
        project_spec: Dict[str, Any],
        custom_config: Optional[SwarmConfig] = None
    ) -> str:
        """
        Initialize optimized swarm for project requirements
        
        Args:
            project_spec: Project specification for optimization
            custom_config: Optional custom swarm configuration
            
        Returns:
            Swarm ID for future operations
        """
        logger.info("Initializing optimized swarm for project")
        
        try:
            # Generate optimal configuration if not provided
            if custom_config is None:
                config = await self._generate_optimal_config(project_spec)
            else:
                config = custom_config
            
            # Initialize swarm via manager
            swarm_id = await self.swarm_manager.initialize_swarm(config)
            
            # Track swarm state
            self.active_swarms[swarm_id] = SwarmState.INITIALIZING
            self.swarm_configs[swarm_id] = config
            self.swarm_tasks[swarm_id] = []
            
            # Initialize metrics
            self.swarm_metrics[swarm_id] = SwarmMetrics(
                swarm_id=swarm_id,
                topology=config.topology.value
            )
            
            # Spawn initial agents based on project requirements
            initial_agents = await self._determine_initial_agents(project_spec)
            for agent_type in initial_agents:
                await self.swarm_manager.spawn_agent(swarm_id, agent_type)
            
            # Update swarm state
            self.active_swarms[swarm_id] = SwarmState.ACTIVE
            self.global_metrics['total_swarms_created'] += 1
            
            # Start monitoring if first swarm
            if len(self.active_swarms) == 1:
                await self._start_monitoring()
            
            logger.info(f"Swarm {swarm_id} initialized successfully with {len(initial_agents)} agents")
            return swarm_id
            
        except Exception as e:
            logger.error(f"Failed to initialize swarm: {e}")
            raise OrchestrationError(f"Swarm initialization failed: {e}") from e
    
    async def execute_task(
        self,
        task_request: TaskRequest,
        swarm_id: Optional[str] = None
    ) -> str:
        """
        Execute task using optimal swarm and agent selection
        
        Args:
            task_request: Task to execute
            swarm_id: Optional specific swarm ID
            
        Returns:
            Execution ID for tracking
        """
        execution_id = f"exec-{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Executing task {task_request.task_id} with execution {execution_id}")
        
        try:
            # Select optimal swarm if not specified
            if swarm_id is None:
                swarm_id = await self._select_optimal_swarm(task_request)
                if swarm_id is None:
                    # Create new swarm if needed
                    project_spec = self._task_to_project_spec(task_request)
                    swarm_id = await self.initialize_swarm(project_spec)
            
            # Validate swarm state
            if swarm_id not in self.active_swarms:
                raise OrchestrationError(f"Swarm {swarm_id} not found")
            
            if self.active_swarms[swarm_id] not in [SwarmState.ACTIVE, SwarmState.IDLE]:
                raise OrchestrationError(f"Swarm {swarm_id} not ready for execution")
            
            # Add to execution tracking
            self.executing_tasks[execution_id] = {
                'task_request': task_request,
                'swarm_id': swarm_id,
                'start_time': datetime.utcnow(),
                'status': 'running'
            }
            
            # Add to pending tasks queue
            self.pending_tasks[task_request.task_id] = task_request
            
            # Execute task asynchronously
            asyncio.create_task(
                self._execute_task_internal(execution_id, task_request, swarm_id)
            )
            
            # Update metrics
            self.global_metrics['total_tasks_processed'] += 1
            current_concurrent = len(self.executing_tasks)
            if current_concurrent > self.global_metrics['peak_concurrent_tasks']:
                self.global_metrics['peak_concurrent_tasks'] = current_concurrent
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Task execution setup failed: {e}")
            raise OrchestrationError(f"Task execution failed: {e}") from e
    
    async def _execute_task_internal(
        self,
        execution_id: str,
        task_request: TaskRequest,
        swarm_id: str
    ):
        """Internal task execution with comprehensive error handling"""
        
        try:
            # Update swarm metrics
            if swarm_id in self.swarm_metrics:
                metrics = self.swarm_metrics[swarm_id]
                metrics.total_tasks += 1
                
            # Create project specification from task
            project_spec = {
                'description': task_request.description,
                'requirements': task_request.context,
                'complexity_score': task_request.estimated_complexity * 10,
                'priority': task_request.priority
            }
            
            # Execute via orchestrator
            result = await self.orchestrator.orchestrate_development_workflow(
                project_spec
            )
            
            # Update execution tracking
            execution_data = self.executing_tasks.get(execution_id, {})
            execution_data.update({
                'status': 'completed' if result.is_success else 'failed',
                'end_time': datetime.utcnow(),
                'result': result,
                'execution_time': result.execution_time
            })
            
            # Move to completed tasks
            self.completed_tasks[execution_id] = execution_data
            
            # Update metrics
            if swarm_id in self.swarm_metrics:
                metrics = self.swarm_metrics[swarm_id]
                if result.is_success:
                    metrics.completed_tasks += 1
                else:
                    metrics.failed_tasks += 1
                
                # Update average task time
                if metrics.completed_tasks > 0:
                    total_time = (metrics.average_task_time * (metrics.completed_tasks - 1) + 
                                result.execution_time)
                    metrics.average_task_time = total_time / metrics.completed_tasks
            
            # Remove from pending and executing
            self.pending_tasks.pop(task_request.task_id, None)
            self.executing_tasks.pop(execution_id, None)
            
            logger.info(f"Task {task_request.task_id} completed with status: {result.status}")
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            
            # Update failure tracking
            if execution_id in self.executing_tasks:
                self.executing_tasks[execution_id].update({
                    'status': 'failed',
                    'error': str(e),
                    'end_time': datetime.utcnow()
                })
                self.completed_tasks[execution_id] = self.executing_tasks.pop(execution_id)
            
            # Update metrics
            if swarm_id in self.swarm_metrics:
                self.swarm_metrics[swarm_id].failed_tasks += 1
    
    async def get_swarm_status(self, swarm_id: str) -> Dict[str, Any]:
        """Get comprehensive swarm status with real-time metrics"""
        
        if swarm_id not in self.active_swarms:
            raise SwarmError(f"Swarm {swarm_id} not found")
        
        try:
            # Get base status from manager
            base_status = await self.swarm_manager.get_swarm_status(swarm_id)
            
            # Add orchestrator metrics
            metrics = self.swarm_metrics.get(swarm_id, SwarmMetrics(swarm_id, "unknown"))
            
            # Update real-time metrics
            await self._update_swarm_metrics(swarm_id)
            
            return {
                **base_status,
                'state': self.active_swarms[swarm_id].value,
                'orchestrator_metrics': {
                    'total_tasks': metrics.total_tasks,
                    'completed_tasks': metrics.completed_tasks,
                    'failed_tasks': metrics.failed_tasks,
                    'success_rate': metrics.success_rate,
                    'average_task_time': metrics.average_task_time,
                    'throughput': metrics.throughput,
                    'utilization_rate': metrics.utilization_rate,
                    'efficiency_score': metrics.efficiency_score
                },
                'resource_usage': {
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage
                },
                'pending_tasks': len([t for t in self.pending_tasks.values() 
                                    if t.task_id in [task.task_id for task in self.swarm_tasks.get(swarm_id, [])]]),
                'executing_tasks': len([e for e in self.executing_tasks.values() 
                                      if e.get('swarm_id') == swarm_id])
            }
            
        except Exception as e:
            logger.error(f"Failed to get swarm status: {e}")
            raise OrchestrationError(f"Status retrieval failed: {e}") from e
    
    async def scale_swarm(
        self,
        swarm_id: str,
        target_agents: int,
        agent_types: Optional[List[AgentType]] = None
    ) -> bool:
        """Scale swarm agents up or down based on workload"""
        
        if swarm_id not in self.active_swarms:
            raise SwarmError(f"Swarm {swarm_id} not found")
        
        try:
            current_agents = len(self.swarm_manager.swarm_agents.get(swarm_id, []))
            
            if target_agents == current_agents:
                return True
            
            logger.info(f"Scaling swarm {swarm_id} from {current_agents} to {target_agents} agents")
            
            self.active_swarms[swarm_id] = SwarmState.SCALING
            
            if target_agents > current_agents:
                # Scale up
                agents_to_add = target_agents - current_agents
                
                if agent_types is None:
                    # Determine optimal agent types based on current workload
                    agent_types = await self._determine_optimal_agent_types(
                        swarm_id, agents_to_add
                    )
                
                for i in range(agents_to_add):
                    agent_type = agent_types[i % len(agent_types)]
                    await self.swarm_manager.spawn_agent(swarm_id, agent_type)
                    
            else:
                # Scale down (implement agent removal logic if needed)
                logger.warning("Agent scale-down not implemented yet")
            
            self.active_swarms[swarm_id] = SwarmState.ACTIVE
            await self._update_swarm_metrics(swarm_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale swarm {swarm_id}: {e}")
            self.active_swarms[swarm_id] = SwarmState.ERROR
            raise OrchestrationError(f"Swarm scaling failed: {e}") from e
    
    async def _generate_optimal_config(self, project_spec: Dict[str, Any]) -> SwarmConfig:
        """Generate optimal swarm configuration based on project requirements"""
        
        # Analyze project complexity
        complexity_factors = {
            'features': len(project_spec.get('features', [])) * 2,
            'integrations': len(project_spec.get('integrations', [])) * 3,
            'estimated_files': project_spec.get('estimated_files', 10),
            'team_size': project_spec.get('team_size', 1) * 5,
            'timeline_days': min(project_spec.get('timeline_days', 30), 100)
        }
        
        complexity_score = sum(complexity_factors.values())
        
        # Select topology based on complexity
        if complexity_score < 20:
            topology = SwarmTopology.STAR
            max_agents = 3
        elif complexity_score < 50:
            topology = SwarmTopology.HIERARCHICAL 
            max_agents = 5
        elif complexity_score < 100:
            topology = SwarmTopology.MESH
            max_agents = 8
        else:
            topology = SwarmTopology.MESH
            max_agents = self.max_agents_per_swarm
        
        return SwarmConfig(
            topology=topology,
            max_agents=max_agents,
            strategy="adaptive",
            enable_coordination=True,
            enable_learning=complexity_score > 30,
            auto_scaling=self.enable_auto_scaling
        )
    
    async def _determine_initial_agents(
        self, 
        project_spec: Dict[str, Any]
    ) -> List[AgentType]:
        """Determine initial agent types based on project requirements"""
        
        agents = [AgentType.SYSTEM_ARCHITECT]  # Always include architect
        
        # Add agents based on project characteristics
        if project_spec.get('requires_backend', True):
            agents.append(AgentType.BACKEND_DEV)
            
        if project_spec.get('requires_frontend', False):
            agents.append(AgentType.FRONTEND_DEV)
            
        if project_spec.get('requires_database', False):
            agents.append(AgentType.DATABASE_ARCHITECT)
            
        if project_spec.get('requires_testing', True):
            agents.append(AgentType.TEST_ENGINEER)
            
        # Add coder for general development
        agents.append(AgentType.CODER)
        
        return agents
    
    async def _select_optimal_swarm(self, task_request: TaskRequest) -> Optional[str]:
        """Select optimal existing swarm for task execution"""
        
        if not self.active_swarms:
            return None
        
        best_swarm = None
        best_score = -1
        
        for swarm_id, state in self.active_swarms.items():
            if state not in [SwarmState.ACTIVE, SwarmState.IDLE]:
                continue
                
            # Calculate suitability score
            metrics = self.swarm_metrics.get(swarm_id)
            if not metrics:
                continue
                
            # Score based on multiple factors
            utilization_score = 100 - metrics.utilization_rate  # Lower utilization is better
            efficiency_score = metrics.efficiency_score
            success_rate_score = metrics.success_rate
            
            # Check agent capabilities match
            agents = self.swarm_manager.swarm_agents.get(swarm_id, [])
            capability_match = 0
            for requirement in task_request.agent_requirements:
                for agent in agents:
                    if requirement.lower() in [cap.lower() for cap in agent.capabilities]:
                        capability_match += 10
                        break
            
            total_score = (
                utilization_score * 0.3 +
                efficiency_score * 0.3 +
                success_rate_score * 0.2 +
                capability_match * 0.2
            )
            
            if total_score > best_score:
                best_score = total_score
                best_swarm = swarm_id
        
        return best_swarm
    
    async def _start_monitoring(self):
        """Start monitoring and optimization tasks"""
        
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitor_swarms())
            
        if self.optimization_task is None or self.optimization_task.done():
            self.optimization_task = asyncio.create_task(self._optimize_swarms())
    
    async def _monitor_swarms(self):
        """Continuous swarm monitoring loop"""
        
        while self.active_swarms:
            try:
                for swarm_id in list(self.active_swarms.keys()):
                    await self._update_swarm_metrics(swarm_id)
                    await self._check_swarm_health(swarm_id)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _optimize_swarms(self):
        """Continuous swarm optimization loop"""
        
        while self.active_swarms:
            try:
                for swarm_id in list(self.active_swarms.keys()):
                    await self._optimize_swarm_performance(swarm_id)
                
                await asyncio.sleep(self.monitoring_interval * 2)  # Less frequent optimization
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)
    
    async def _update_swarm_metrics(self, swarm_id: str):
        """Update real-time swarm metrics"""
        
        if swarm_id not in self.swarm_metrics:
            return
            
        try:
            metrics = self.swarm_metrics[swarm_id]
            
            # Get agent counts
            agents = self.swarm_manager.swarm_agents.get(swarm_id, [])
            metrics.total_agents = len(agents)
            metrics.active_agents = len([a for a in agents if a.status == "active"])
            metrics.idle_agents = len([a for a in agents if a.status == "idle"])
            metrics.busy_agents = len([a for a in agents if a.status == "busy"])
            
            # Calculate throughput (tasks per minute)
            if metrics.average_task_time > 0:
                metrics.throughput = 60.0 / metrics.average_task_time
            
            # Update system resource usage
            process = psutil.Process()
            metrics.cpu_usage = process.cpu_percent()
            metrics.memory_usage = process.memory_percent()
            
            # Calculate efficiency score
            if metrics.total_tasks > 0:
                efficiency_factors = {
                    'success_rate': metrics.success_rate / 100.0,
                    'utilization': metrics.utilization_rate / 100.0,
                    'throughput_normalized': min(metrics.throughput / 10.0, 1.0)  # Normalize to 0-1
                }
                
                metrics.efficiency_score = sum(efficiency_factors.values()) / len(efficiency_factors) * 100
            
            metrics.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.warning(f"Failed to update metrics for swarm {swarm_id}: {e}")
    
    async def _check_swarm_health(self, swarm_id: str):
        """Check and update swarm health status"""
        
        try:
            metrics = self.swarm_metrics.get(swarm_id)
            if not metrics:
                return
            
            current_state = self.active_swarms[swarm_id]
            
            # Health checks
            if metrics.success_rate < 50 and metrics.total_tasks > 5:
                if current_state != SwarmState.DEGRADED:
                    logger.warning(f"Swarm {swarm_id} degraded - low success rate: {metrics.success_rate}%")
                    self.active_swarms[swarm_id] = SwarmState.DEGRADED
                    
            elif metrics.active_agents == 0 and metrics.total_agents > 0:
                if current_state != SwarmState.ERROR:
                    logger.error(f"Swarm {swarm_id} error - no active agents")
                    self.active_swarms[swarm_id] = SwarmState.ERROR
                    
            elif current_state in [SwarmState.DEGRADED, SwarmState.ERROR]:
                # Check if swarm recovered
                if metrics.success_rate > 75 and metrics.active_agents > 0:
                    logger.info(f"Swarm {swarm_id} recovered")
                    self.active_swarms[swarm_id] = SwarmState.ACTIVE
            
        except Exception as e:
            logger.error(f"Health check failed for swarm {swarm_id}: {e}")
    
    async def _optimize_swarm_performance(self, swarm_id: str):
        """Optimize swarm performance based on metrics and workload"""
        
        if not self.enable_auto_scaling:
            return
            
        try:
            metrics = self.swarm_metrics.get(swarm_id)
            if not metrics:
                return
            
            # Auto-scaling decisions
            if metrics.utilization_rate > 90 and metrics.total_agents < self.max_agents_per_swarm:
                # Scale up
                new_target = min(metrics.total_agents + 1, self.max_agents_per_swarm)
                await self.scale_swarm(swarm_id, new_target)
                logger.info(f"Scaled up swarm {swarm_id} to {new_target} agents (high utilization)")
                
            elif metrics.utilization_rate < 20 and metrics.total_agents > 2:
                # Scale down 
                new_target = max(metrics.total_agents - 1, 2)
                await self.scale_swarm(swarm_id, new_target)
                logger.info(f"Scaled down swarm {swarm_id} to {new_target} agents (low utilization)")
            
        except Exception as e:
            logger.error(f"Performance optimization failed for swarm {swarm_id}: {e}")
    
    def _task_to_project_spec(self, task_request: TaskRequest) -> Dict[str, Any]:
        """Convert task request to project specification"""
        
        return {
            'description': task_request.description,
            'features': task_request.context.get('features', []),
            'complexity_score': task_request.estimated_complexity * 10,
            'requires_backend': 'backend' in task_request.description.lower(),
            'requires_frontend': 'frontend' in task_request.description.lower() or 'ui' in task_request.description.lower(),
            'requires_database': 'database' in task_request.description.lower() or 'db' in task_request.description.lower(),
            'requires_testing': True,  # Always require testing
            'estimated_files': task_request.estimated_complexity * 2,
            'priority': task_request.priority
        }
    
    async def _determine_optimal_agent_types(
        self, 
        swarm_id: str, 
        count: int
    ) -> List[AgentType]:
        """Determine optimal agent types for scaling"""
        
        # Analyze current workload and agent distribution
        pending_tasks = [t for t in self.pending_tasks.values() 
                        if t.task_id in [task.task_id for task in self.swarm_tasks.get(swarm_id, [])]]
        
        # Count requirements from pending tasks
        requirement_counts = {}
        for task in pending_tasks:
            for req in task.agent_requirements:
                requirement_counts[req] = requirement_counts.get(req, 0) + 1
        
        # Map requirements to agent types
        agent_types = []
        type_priority = [
            (AgentType.CODER, ['coding', 'development', 'implementation']),
            (AgentType.ANALYST, ['analysis', 'research', 'investigation']),
            (AgentType.OPTIMIZER, ['optimization', 'performance', 'efficiency']),
            (AgentType.TEST_ENGINEER, ['testing', 'qa', 'validation'])
        ]
        
        for agent_type, keywords in type_priority:
            if any(keyword in requirement_counts for keyword in keywords):
                agent_types.append(agent_type)
        
        # Fill remaining slots with general coders
        while len(agent_types) < count:
            agent_types.append(AgentType.CODER)
        
        return agent_types[:count]
    
    async def shutdown(self):
        """Gracefully shutdown orchestrator and all swarms"""
        
        logger.info("Shutting down swarm orchestrator")
        
        try:
            # Cancel monitoring tasks
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                
            if self.optimization_task and not self.optimization_task.done():
                self.optimization_task.cancel()
            
            # Wait for executing tasks to complete (with timeout)
            if self.executing_tasks:
                logger.info(f"Waiting for {len(self.executing_tasks)} executing tasks to complete")
                
                timeout = 60  # 60 second timeout
                start_time = time.time()
                
                while self.executing_tasks and (time.time() - start_time) < timeout:
                    await asyncio.sleep(1)
                
                if self.executing_tasks:
                    logger.warning(f"Timed out waiting for {len(self.executing_tasks)} tasks to complete")
            
            # Clear all tracking
            self.active_swarms.clear()
            self.swarm_metrics.clear()
            self.pending_tasks.clear()
            self.executing_tasks.clear()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Swarm orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global orchestration metrics"""
        
        # Calculate average efficiency across all swarms
        if self.swarm_metrics:
            total_efficiency = sum(m.efficiency_score for m in self.swarm_metrics.values())
            avg_efficiency = total_efficiency / len(self.swarm_metrics)
        else:
            avg_efficiency = 0.0
        
        uptime = datetime.utcnow() - self.global_metrics['system_uptime']
        
        return {
            **self.global_metrics,
            'active_swarms': len(self.active_swarms),
            'total_agents': sum(m.total_agents for m in self.swarm_metrics.values()),
            'pending_tasks': len(self.pending_tasks),
            'executing_tasks': len(self.executing_tasks),
            'completed_tasks': len(self.completed_tasks),
            'average_swarm_efficiency': avg_efficiency,
            'system_uptime_hours': uptime.total_seconds() / 3600,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
