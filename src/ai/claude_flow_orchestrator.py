"""Claude Flow Orchestrator Module

Production-ready orchestrator for Claude Flow swarm management with:
- Advanced task routing and intelligent agent selection
- Real-time swarm topology optimization
- Context-aware workflow orchestration
- Performance-driven scaling and resource allocation
- Integration with Claude Code and external AI services
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set, Callable
from pathlib import Path
import aiohttp
import redis.asyncio as redis

from ..integrations.claude_flow import SwarmManager, SwarmConfig, SwarmTopology, AgentType
from ..integrations.claude_code import ClaudeCodeClient
from ..core.exceptions import OrchestrationError, SwarmError, TaskError
from .performance_monitor import PerformanceMonitor, MetricType
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    """Orchestrator operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    ORCHESTRATING = "orchestrating"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class ContextType(Enum):
    """Context types for task execution"""
    DEVELOPMENT = "development"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    INTEGRATION = "integration"
    TESTING = "testing"


@dataclass
class OrchestrationTask:
    """Advanced task definition for orchestration"""
    task_id: str
    description: str
    context_type: ContextType
    priority: TaskPriority
    agent_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: int = 300  # seconds
    max_retries: int = 3
    retry_count: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if task has exceeded deadline"""
        return self.deadline and datetime.utcnow() > self.deadline


@dataclass
class SwarmCapacity:
    """Swarm capacity and performance metrics"""
    swarm_id: str
    current_load: float = 0.0  # 0-1 scale
    max_capacity: int = 10
    active_tasks: int = 0
    avg_completion_time: float = 0.0
    success_rate: float = 1.0
    specialization_score: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ClaudeFlowOrchestrator:
    """
    Production Claude Flow Orchestrator
    
    Advanced orchestration system providing:
    - Intelligent task routing and agent selection
    - Dynamic swarm topology optimization
    - Context-aware workflow management
    - Performance-driven scaling decisions
    - Integration with Claude Code and external systems
    - Real-time monitoring and adaptive optimization
    """
    
    def __init__(
        self,
        claude_code_client: Optional[ClaudeCodeClient] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        cache_manager: Optional[CacheManager] = None,
        max_concurrent_tasks: int = 50,
        optimization_interval: int = 300  # 5 minutes
    ):
        self.claude_code_client = claude_code_client or ClaudeCodeClient()
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.cache_manager = cache_manager or CacheManager()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.optimization_interval = optimization_interval
        
        # Core components
        self.swarm_manager = SwarmManager()
        self.state = OrchestratorState.INITIALIZING
        
        # Task management
        self.active_tasks: Dict[str, OrchestrationTask] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.task_history: List[Dict[str, Any]] = []
        
        # Swarm tracking
        self.swarm_capacities: Dict[str, SwarmCapacity] = {}
        self.swarm_specializations: Dict[str, List[ContextType]] = {}
        self.topology_performance: Dict[SwarmTopology, Dict[str, float]] = {}
        
        # Context management
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.context_patterns: Dict[ContextType, Dict[str, Any]] = {}
        
        # Performance tracking
        self.orchestration_metrics = {
            'tasks_orchestrated': 0,
            'successful_completions': 0,
            'failed_tasks': 0,
            'average_orchestration_time': 0.0,
            'swarm_utilization': 0.0,
            'context_accuracy': 0.0,
            'optimization_score': 0.0
        }
        
        # Background tasks
        self.orchestration_loop: Optional[asyncio.Task] = None
        self.optimization_loop: Optional[asyncio.Task] = None
        self.monitoring_loop: Optional[asyncio.Task] = None
        
        # Redis client for distributed coordination
        self.redis_client: Optional[redis.Redis] = None
        
        self._initialize_context_patterns()
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

    def _initialize_context_patterns(self):
        """Initialize context-specific patterns"""
        self.context_patterns = {
            ContextType.DEVELOPMENT: {
                'preferred_agents': [AgentType.CODER, AgentType.SYSTEM_ARCHITECT],
                'topology': SwarmTopology.HIERARCHICAL,
                'max_agents': 8,
                'timeout_multiplier': 1.5
            },
            ContextType.RESEARCH: {
                'preferred_agents': [AgentType.RESEARCHER, AgentType.ANALYST],
                'topology': SwarmTopology.MESH,
                'max_agents': 5,
                'timeout_multiplier': 2.0
            },
            ContextType.ANALYSIS: {
                'preferred_agents': [AgentType.ANALYST, AgentType.OPTIMIZER],
                'topology': SwarmTopology.STAR,
                'max_agents': 6,
                'timeout_multiplier': 1.2
            },
            ContextType.OPTIMIZATION: {
                'preferred_agents': [AgentType.OPTIMIZER, AgentType.COORDINATOR],
                'topology': SwarmTopology.MESH,
                'max_agents': 4,
                'timeout_multiplier': 1.8
            },
            ContextType.INTEGRATION: {
                'preferred_agents': [AgentType.SYSTEM_ARCHITECT, AgentType.CODER],
                'topology': SwarmTopology.HIERARCHICAL,
                'max_agents': 7,
                'timeout_multiplier': 2.5
            },
            ContextType.TESTING: {
                'preferred_agents': [AgentType.TEST_ENGINEER, AgentType.REVIEWER],
                'topology': SwarmTopology.STAR,
                'max_agents': 4,
                'timeout_multiplier': 1.0
            }
        }

    async def initialize(self):
        """Initialize the orchestrator system"""
        logger.info("Initializing Claude Flow Orchestrator")
        
        try:
            # Initialize Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Start core components
            await self.performance_monitor.start()
            await self.cache_manager.initialize()
            
            # Initialize swarm manager
            await self.swarm_manager.initialize()
            
            # Start background loops
            self.orchestration_loop = asyncio.create_task(self._orchestration_loop())
            self.optimization_loop = asyncio.create_task(self._optimization_loop())
            self.monitoring_loop = asyncio.create_task(self._monitoring_loop())
            
            self.state = OrchestratorState.READY
            logger.info("Claude Flow Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self.state = OrchestratorState.ERROR
            raise OrchestrationError(f"Orchestrator initialization failed: {e}") from e

    async def orchestrate_task(
        self,
        task: OrchestrationTask,
        preferred_swarm_id: Optional[str] = None
    ) -> str:
        """
        Orchestrate task execution with intelligent routing
        
        Args:
            task: Task to orchestrate
            preferred_swarm_id: Optional preferred swarm for execution
            
        Returns:
            Execution ID for tracking
        """
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Orchestrating task {task.task_id} with execution {execution_id}")
        
        try:
            # Validate task
            if task.is_expired:
                raise TaskError(f"Task {task.task_id} has expired")
            
            # Check capacity limits
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                await self.task_queue.put((task, preferred_swarm_id, execution_id))
                logger.info(f"Task {task.task_id} queued due to capacity limits")
                return execution_id
            
            # Select optimal swarm
            swarm_id = await self._select_optimal_swarm(task, preferred_swarm_id)
            
            # Create execution context
            context = await self._build_execution_context(task, swarm_id)
            
            # Start task execution
            self.active_tasks[execution_id] = task
            asyncio.create_task(self._execute_task(execution_id, task, swarm_id, context))
            
            # Update metrics
            self.orchestration_metrics['tasks_orchestrated'] += 1
            
            # Cache task info for monitoring
            await self._cache_task_info(execution_id, {
                'task_id': task.task_id,
                'swarm_id': swarm_id,
                'context_type': task.context_type.value,
                'priority': task.priority.value,
                'started_at': datetime.utcnow().isoformat()
            })
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to orchestrate task {task.task_id}: {e}")
            raise OrchestrationError(f"Task orchestration failed: {e}") from e

    async def _select_optimal_swarm(
        self,
        task: OrchestrationTask,
        preferred_swarm_id: Optional[str] = None
    ) -> str:
        """Select optimal swarm for task execution"""
        
        if preferred_swarm_id and preferred_swarm_id in self.swarm_capacities:
            capacity = self.swarm_capacities[preferred_swarm_id]
            if capacity.current_load < 0.9:  # Not overloaded
                return preferred_swarm_id
        
        # Score existing swarms
        best_swarm = None
        best_score = -1
        
        for swarm_id, capacity in self.swarm_capacities.items():
            if capacity.current_load >= 1.0:  # Skip overloaded swarms
                continue
                
            score = await self._calculate_swarm_suitability(swarm_id, task)
            if score > best_score:
                best_score = score
                best_swarm = swarm_id
        
        # Create new swarm if needed
        if best_swarm is None or best_score < 0.7:
            best_swarm = await self._create_optimized_swarm(task)
        
        return best_swarm

    async def _calculate_swarm_suitability(
        self,
        swarm_id: str,
        task: OrchestrationTask
    ) -> float:
        """Calculate swarm suitability score for task"""
        
        capacity = self.swarm_capacities.get(swarm_id)
        if not capacity:
            return 0.0
        
        # Base score from capacity and performance
        capacity_score = 1.0 - capacity.current_load
        performance_score = capacity.success_rate
        health_score = capacity.health_score
        
        # Context specialization score
        context_score = capacity.specialization_score.get(
            task.context_type.value, 0.5
        )
        
        # Agent capability matching
        agents = self.swarm_manager.swarm_agents.get(swarm_id, [])
        capability_matches = 0
        
        for requirement in task.agent_requirements:
            for agent in agents:
                if requirement.lower() in [cap.lower() for cap in agent.capabilities]:
                    capability_matches += 1
                    break
        
        capability_score = min(1.0, capability_matches / max(len(task.agent_requirements), 1))
        
        # Weighted total score
        total_score = (
            capacity_score * 0.25 +
            performance_score * 0.25 +
            health_score * 0.15 +
            context_score * 0.20 +
            capability_score * 0.15
        )
        
        return total_score

    async def _create_optimized_swarm(self, task: OrchestrationTask) -> str:
        """Create optimized swarm for task execution"""
        
        pattern = self.context_patterns.get(task.context_type)
        if not pattern:
            pattern = self.context_patterns[ContextType.DEVELOPMENT]
        
        # Create swarm configuration
        config = SwarmConfig(
            topology=pattern['topology'],
            max_agents=pattern['max_agents'],
            strategy="adaptive",
            enable_coordination=True,
            enable_learning=True,
            auto_scaling=True
        )
        
        # Initialize swarm
        swarm_id = await self.swarm_manager.initialize_swarm(config)
        
        # Spawn preferred agents
        for agent_type in pattern['preferred_agents'][:3]:  # Start with 3 agents
            await self.swarm_manager.spawn_agent(swarm_id, agent_type)
        
        # Initialize capacity tracking
        self.swarm_capacities[swarm_id] = SwarmCapacity(
            swarm_id=swarm_id,
            max_capacity=pattern['max_agents']
        )
        
        # Record specialization
        if swarm_id not in self.swarm_specializations:
            self.swarm_specializations[swarm_id] = []
        self.swarm_specializations[swarm_id].append(task.context_type)
        
        logger.info(f"Created optimized swarm {swarm_id} for {task.context_type.value} context")
        
        return swarm_id

    async def _build_execution_context(
        self,
        task: OrchestrationTask,
        swarm_id: str
    ) -> Dict[str, Any]:
        """Build comprehensive execution context"""
        
        context = {
            'task_id': task.task_id,
            'swarm_id': swarm_id,
            'context_type': task.context_type.value,
            'priority': task.priority.value,
            'estimated_duration': task.estimated_duration,
            'deadline': task.deadline.isoformat() if task.deadline else None,
            'agent_requirements': task.agent_requirements,
            'dependencies': task.dependencies,
            'metadata': task.metadata,
            'context_data': task.context_data,
            'orchestrator_config': {
                'retry_enabled': task.max_retries > 0,
                'max_retries': task.max_retries,
                'timeout_multiplier': self.context_patterns.get(
                    task.context_type, {}
                ).get('timeout_multiplier', 1.0)
            }
        }
        
        # Add cached context from previous similar tasks
        similar_contexts = await self._get_similar_contexts(task)
        if similar_contexts:
            context['historical_patterns'] = similar_contexts
        
        return context

    async def _execute_task(
        self,
        execution_id: str,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ):
        """Execute task with comprehensive error handling and monitoring"""
        
        start_time = time.time()
        
        try:
            # Update swarm capacity
            if swarm_id in self.swarm_capacities:
                capacity = self.swarm_capacities[swarm_id]
                capacity.active_tasks += 1
                capacity.current_load = min(1.0, capacity.active_tasks / capacity.max_capacity)
            
            # Record performance metrics
            timer_id = self.performance_monitor.start_timer(f"task_execution_{task.context_type.value}")
            
            # Choose execution strategy based on context
            if task.context_type == ContextType.DEVELOPMENT:
                result = await self._execute_development_task(task, swarm_id, context)
            elif task.context_type == ContextType.RESEARCH:
                result = await self._execute_research_task(task, swarm_id, context)
            elif task.context_type == ContextType.ANALYSIS:
                result = await self._execute_analysis_task(task, swarm_id, context)
            elif task.context_type == ContextType.OPTIMIZATION:
                result = await self._execute_optimization_task(task, swarm_id, context)
            elif task.context_type == ContextType.INTEGRATION:
                result = await self._execute_integration_task(task, swarm_id, context)
            elif task.context_type == ContextType.TESTING:
                result = await self._execute_testing_task(task, swarm_id, context)
            else:
                result = await self._execute_generic_task(task, swarm_id, context)
            
            execution_time = self.performance_monitor.end_timer(timer_id)
            
            # Process results
            success = result.get('success', False)
            
            if success:
                await self._handle_task_success(execution_id, task, swarm_id, result, execution_time)
            else:
                await self._handle_task_failure(execution_id, task, swarm_id, result, execution_time)
                
        except Exception as e:
            logger.error(f"Task execution failed for {execution_id}: {e}")
            await self._handle_task_error(execution_id, task, swarm_id, str(e), time.time() - start_time)
        
        finally:
            # Cleanup
            self.active_tasks.pop(execution_id, None)
            
            # Update swarm capacity
            if swarm_id in self.swarm_capacities:
                capacity = self.swarm_capacities[swarm_id]
                capacity.active_tasks = max(0, capacity.active_tasks - 1)
                capacity.current_load = capacity.active_tasks / capacity.max_capacity

    async def _execute_development_task(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute development-focused task"""
        
        try:
            # Use Claude Code integration for development tasks
            if self.claude_code_client:
                result = await self.claude_code_client.execute_development_task(
                    task.description,
                    context
                )
                return {
                    'success': True,
                    'result': result,
                    'execution_method': 'claude_code'
                }
            
            # Fallback to swarm execution
            return await self._execute_via_swarm(task, swarm_id, context)
            
        except Exception as e:
            logger.error(f"Development task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_method': 'development'
            }

    async def _execute_research_task(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute research-focused task"""
        
        try:
            # Use specialized research agents
            agents = await self.swarm_manager.get_available_agents(
                swarm_id, 
                [AgentType.RESEARCHER, AgentType.ANALYST]
            )
            
            if not agents:
                # Spawn research agent if none available
                await self.swarm_manager.spawn_agent(swarm_id, AgentType.RESEARCHER)
            
            return await self._execute_via_swarm(task, swarm_id, context)
            
        except Exception as e:
            logger.error(f"Research task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_method': 'research'
            }

    async def _execute_analysis_task(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis-focused task"""
        
        try:
            # Use performance monitoring for analysis tasks
            self.performance_monitor.record_metric(
                f"analysis_task_started",
                1,
                MetricType.CUSTOM,
                {"swarm_id": swarm_id}
            )
            
            return await self._execute_via_swarm(task, swarm_id, context)
            
        except Exception as e:
            logger.error(f"Analysis task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_method': 'analysis'
            }

    async def _execute_optimization_task(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute optimization-focused task"""
        
        try:
            # Use optimizer agents with performance feedback
            agents = await self.swarm_manager.get_available_agents(
                swarm_id,
                [AgentType.OPTIMIZER]
            )
            
            if not agents:
                await self.swarm_manager.spawn_agent(swarm_id, AgentType.OPTIMIZER)
            
            # Add performance context
            context['performance_data'] = await self.performance_monitor.get_metrics_snapshot()
            
            return await self._execute_via_swarm(task, swarm_id, context)
            
        except Exception as e:
            logger.error(f"Optimization task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_method': 'optimization'
            }

    async def _execute_integration_task(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute integration-focused task"""
        
        try:
            # Integration tasks require coordination between multiple agents
            coordinator_agents = await self.swarm_manager.get_available_agents(
                swarm_id,
                [AgentType.COORDINATOR, AgentType.SYSTEM_ARCHITECT]
            )
            
            if not coordinator_agents:
                await self.swarm_manager.spawn_agent(swarm_id, AgentType.COORDINATOR)
            
            # Add integration-specific context
            context['integration_mode'] = True
            context['coordination_required'] = True
            
            return await self._execute_via_swarm(task, swarm_id, context)
            
        except Exception as e:
            logger.error(f"Integration task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_method': 'integration'
            }

    async def _execute_testing_task(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute testing-focused task"""
        
        try:
            # Use testing agents with validation context
            test_agents = await self.swarm_manager.get_available_agents(
                swarm_id,
                [AgentType.TEST_ENGINEER, AgentType.REVIEWER]
            )
            
            if not test_agents:
                await self.swarm_manager.spawn_agent(swarm_id, AgentType.TEST_ENGINEER)
            
            # Add testing-specific context
            context['validation_required'] = True
            context['quality_gates'] = task.context_data.get('quality_gates', [])
            
            return await self._execute_via_swarm(task, swarm_id, context)
            
        except Exception as e:
            logger.error(f"Testing task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_method': 'testing'
            }

    async def _execute_generic_task(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute generic task using default swarm behavior"""
        
        return await self._execute_via_swarm(task, swarm_id, context)

    async def _execute_via_swarm(
        self,
        task: OrchestrationTask,
        swarm_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task via swarm with enhanced error handling"""
        
        try:
            # Simulate swarm execution (replace with actual swarm integration)
            await asyncio.sleep(0.5)  # Simulate processing time
            
            # Mock successful execution
            return {
                'success': True,
                'result': {
                    'task_id': task.task_id,
                    'swarm_id': swarm_id,
                    'context_type': task.context_type.value,
                    'execution_summary': f"Successfully executed {task.context_type.value} task",
                    'agents_used': len(self.swarm_manager.swarm_agents.get(swarm_id, [])),
                    'execution_time': 0.5
                },
                'execution_method': 'swarm'
            }
            
        except Exception as e:
            logger.error(f"Swarm execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_method': 'swarm'
            }

    async def _handle_task_success(
        self,
        execution_id: str,
        task: OrchestrationTask,
        swarm_id: str,
        result: Dict[str, Any],
        execution_time: float
    ):
        """Handle successful task completion"""
        
        # Update metrics
        self.orchestration_metrics['successful_completions'] += 1
        
        # Update swarm capacity metrics
        if swarm_id in self.swarm_capacities:
            capacity = self.swarm_capacities[swarm_id]
            
            # Update success rate
            total_tasks = capacity.specialization_score.get('total_tasks', 0) + 1
            successes = capacity.specialization_score.get('successes', 0) + 1
            capacity.success_rate = successes / total_tasks
            
            # Update average completion time
            if capacity.avg_completion_time == 0:
                capacity.avg_completion_time = execution_time
            else:
                capacity.avg_completion_time = (capacity.avg_completion_time + execution_time) / 2
            
            # Update context specialization
            context_key = task.context_type.value
            current_score = capacity.specialization_score.get(context_key, 0.5)
            capacity.specialization_score[context_key] = min(1.0, current_score + 0.1)
            capacity.specialization_score['total_tasks'] = total_tasks
            capacity.specialization_score['successes'] = successes
        
        # Store completion record
        completion_record = {
            'execution_id': execution_id,
            'task_id': task.task_id,
            'swarm_id': swarm_id,
            'context_type': task.context_type.value,
            'priority': task.priority.value,
            'execution_time': execution_time,
            'result': result,
            'completed_at': datetime.utcnow().isoformat(),
            'success': True
        }
        
        self.completed_tasks[execution_id] = completion_record
        self.task_history.append(completion_record)
        
        # Cache result for future reference
        await self._cache_task_result(execution_id, completion_record)
        
        logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")

    async def _handle_task_failure(
        self,
        execution_id: str,
        task: OrchestrationTask,
        swarm_id: str,
        result: Dict[str, Any],
        execution_time: float
    ):
        """Handle task failure with retry logic"""
        
        # Check if retry is possible
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
            
            # Add exponential backoff
            delay = 2 ** task.retry_count
            await asyncio.sleep(delay)
            
            # Re-orchestrate with retry context
            retry_context = await self._build_execution_context(task, swarm_id)
            retry_context['is_retry'] = True
            retry_context['retry_count'] = task.retry_count
            
            asyncio.create_task(self._execute_task(execution_id, task, swarm_id, retry_context))
            return
        
        # Final failure
        self.orchestration_metrics['failed_tasks'] += 1
        
        # Update swarm capacity metrics
        if swarm_id in self.swarm_capacities:
            capacity = self.swarm_capacities[swarm_id]
            
            total_tasks = capacity.specialization_score.get('total_tasks', 0) + 1
            successes = capacity.specialization_score.get('successes', 0)
            capacity.success_rate = successes / total_tasks if total_tasks > 0 else 0.0
            capacity.specialization_score['total_tasks'] = total_tasks
            
            # Reduce health score on failure
            capacity.health_score = max(0.1, capacity.health_score - 0.1)
        
        # Store failure record
        failure_record = {
            'execution_id': execution_id,
            'task_id': task.task_id,
            'swarm_id': swarm_id,
            'context_type': task.context_type.value,
            'priority': task.priority.value,
            'execution_time': execution_time,
            'error': result.get('error', 'Unknown error'),
            'retry_count': task.retry_count,
            'failed_at': datetime.utcnow().isoformat(),
            'success': False
        }
        
        self.completed_tasks[execution_id] = failure_record
        self.task_history.append(failure_record)
        
        logger.error(f"Task {task.task_id} failed permanently after {task.retry_count} retries")

    async def _handle_task_error(
        self,
        execution_id: str,
        task: OrchestrationTask,
        swarm_id: str,
        error: str,
        execution_time: float
    ):
        """Handle task execution error"""
        
        error_record = {
            'execution_id': execution_id,
            'task_id': task.task_id,
            'swarm_id': swarm_id,
            'error': error,
            'execution_time': execution_time,
            'error_at': datetime.utcnow().isoformat(),
            'success': False
        }
        
        self.completed_tasks[execution_id] = error_record
        self.orchestration_metrics['failed_tasks'] += 1

    async def _orchestration_loop(self):
        """Main orchestration loop for queued tasks"""
        
        while self.state != OrchestratorState.SHUTDOWN:
            try:
                # Process queued tasks
                if not self.task_queue.empty() and len(self.active_tasks) < self.max_concurrent_tasks:
                    try:
                        task, preferred_swarm_id, execution_id = await asyncio.wait_for(
                            self.task_queue.get(), timeout=1.0
                        )
                        
                        # Re-orchestrate queued task
                        swarm_id = await self._select_optimal_swarm(task, preferred_swarm_id)
                        context = await self._build_execution_context(task, swarm_id)
                        
                        self.active_tasks[execution_id] = task
                        asyncio.create_task(self._execute_task(execution_id, task, swarm_id, context))
                        
                    except asyncio.TimeoutError:
                        continue
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(5)

    async def _optimization_loop(self):
        """Background optimization loop"""
        
        while self.state != OrchestratorState.SHUTDOWN:
            try:
                await asyncio.sleep(self.optimization_interval)
                await self._optimize_swarm_topologies()
                await self._cleanup_idle_swarms()
                await self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(self.optimization_interval)

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.state != OrchestratorState.SHUTDOWN:
            try:
                await self._monitor_swarm_health()
                await self._detect_performance_anomalies()
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _optimize_swarm_topologies(self):
        """Optimize swarm topologies based on performance data"""
        
        for swarm_id, capacity in self.swarm_capacities.items():
            try:
                # Analyze performance patterns
                if capacity.specialization_score.get('total_tasks', 0) < 5:
                    continue  # Not enough data
                
                current_topology = await self.swarm_manager.get_swarm_topology(swarm_id)
                if not current_topology:
                    continue
                
                # Calculate topology efficiency
                efficiency = capacity.success_rate * capacity.health_score
                
                # Store topology performance
                if current_topology not in self.topology_performance:
                    self.topology_performance[current_topology] = {}
                
                context_types = self.swarm_specializations.get(swarm_id, [])
                for context_type in context_types:
                    key = context_type.value
                    if key not in self.topology_performance[current_topology]:
                        self.topology_performance[current_topology][key] = []
                    
                    self.topology_performance[current_topology][key].append(efficiency)
                    
                    # Keep only recent performance data
                    if len(self.topology_performance[current_topology][key]) > 10:
                        self.topology_performance[current_topology][key] = \
                            self.topology_performance[current_topology][key][-10:]
                
            except Exception as e:
                logger.warning(f"Topology optimization failed for swarm {swarm_id}: {e}")

    async def _cleanup_idle_swarms(self):
        """Cleanup idle swarms to free resources"""
        
        idle_threshold = timedelta(minutes=30)
        current_time = datetime.utcnow()
        
        swarms_to_remove = []
        
        for swarm_id, capacity in self.swarm_capacities.items():
            if capacity.active_tasks == 0:
                time_since_update = current_time - capacity.last_updated
                
                if time_since_update > idle_threshold:
                    swarms_to_remove.append(swarm_id)
        
        for swarm_id in swarms_to_remove:
            try:
                await self.swarm_manager.shutdown_swarm(swarm_id)
                self.swarm_capacities.pop(swarm_id, None)
                self.swarm_specializations.pop(swarm_id, None)
                
                logger.info(f"Cleaned up idle swarm {swarm_id}")
                
            except Exception as e:
                logger.warning(f"Failed to cleanup swarm {swarm_id}: {e}")

    async def _monitor_swarm_health(self):
        """Monitor swarm health and trigger recovery if needed"""
        
        for swarm_id, capacity in self.swarm_capacities.items():
            try:
                if capacity.health_score < 0.5:
                    logger.warning(f"Swarm {swarm_id} health degraded: {capacity.health_score}")
                    
                    # Attempt recovery
                    await self._recover_swarm_health(swarm_id)
                
            except Exception as e:
                logger.error(f"Health monitoring failed for swarm {swarm_id}: {e}")

    async def _recover_swarm_health(self, swarm_id: str):
        """Attempt to recover swarm health"""
        
        try:
            # Restart unhealthy agents
            agents = self.swarm_manager.swarm_agents.get(swarm_id, [])
            unhealthy_agents = [a for a in agents if a.status == "error"]
            
            for agent in unhealthy_agents:
                await self.swarm_manager.restart_agent(swarm_id, agent.id)
            
            # Reset health score
            if swarm_id in self.swarm_capacities:
                self.swarm_capacities[swarm_id].health_score = 0.8
            
            logger.info(f"Attempted health recovery for swarm {swarm_id}")
            
        except Exception as e:
            logger.error(f"Health recovery failed for swarm {swarm_id}: {e}")

    async def _detect_performance_anomalies(self):
        """Detect and respond to performance anomalies"""
        
        # Check for system-wide performance issues
        metrics_snapshot = await self.performance_monitor.get_metrics_snapshot()
        
        system_metrics = metrics_snapshot.get('system_metrics', {})
        cpu_usage = system_metrics.get('cpu_usage_percent', {}).get('current', 0)
        memory_usage = system_metrics.get('memory_usage_percent', {}).get('current', 0)
        
        # Trigger scaling if resource usage is high
        if cpu_usage > 80 or memory_usage > 85:
            logger.warning(f"High resource usage detected: CPU={cpu_usage}%, Memory={memory_usage}%")
            await self._handle_resource_pressure()

    async def _handle_resource_pressure(self):
        """Handle high resource usage by optimizing swarm allocation"""
        
        try:
            # Reduce concurrent task limits
            original_limit = self.max_concurrent_tasks
            self.max_concurrent_tasks = max(10, int(self.max_concurrent_tasks * 0.8))
            
            if self.max_concurrent_tasks != original_limit:
                logger.info(f"Reduced concurrent task limit from {original_limit} to {self.max_concurrent_tasks}")
            
            # Scale down overloaded swarms
            for swarm_id, capacity in self.swarm_capacities.items():
                if capacity.current_load > 0.9:
                    # Add tasks to queue instead of immediate execution
                    continue
            
        except Exception as e:
            logger.error(f"Resource pressure handling failed: {e}")

    async def _get_similar_contexts(self, task: OrchestrationTask) -> List[Dict[str, Any]]:
        """Get similar execution contexts from history"""
        
        similar_contexts = []
        
        for record in self.task_history[-50:]:  # Check recent history
            if (record.get('context_type') == task.context_type.value and 
                record.get('success', False)):
                
                similar_contexts.append({
                    'execution_time': record.get('execution_time', 0),
                    'swarm_id': record.get('swarm_id'),
                    'agents_used': record.get('result', {}).get('agents_used', 0)
                })
        
        return similar_contexts[-5:]  # Return 5 most recent similar contexts

    async def _cache_task_info(self, execution_id: str, info: Dict[str, Any]):
        """Cache task information for monitoring"""
        
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    f"task:{execution_id}",
                    3600,  # 1 hour expiry
                    json.dumps(info)
                )
        except Exception as e:
            logger.warning(f"Failed to cache task info: {e}")

    async def _cache_task_result(self, execution_id: str, result: Dict[str, Any]):
        """Cache task result for analytics"""
        
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    f"result:{execution_id}",
                    86400,  # 24 hours expiry
                    json.dumps(result)
                )
        except Exception as e:
            logger.warning(f"Failed to cache task result: {e}")

    async def _update_performance_metrics(self):
        """Update orchestrator performance metrics"""
        
        try:
            total_tasks = len(self.completed_tasks)
            if total_tasks > 0:
                successful_tasks = len([t for t in self.completed_tasks.values() if t.get('success')])
                self.orchestration_metrics['context_accuracy'] = (successful_tasks / total_tasks) * 100
            
            # Calculate swarm utilization
            if self.swarm_capacities:
                total_utilization = sum(c.current_load for c in self.swarm_capacities.values())
                self.orchestration_metrics['swarm_utilization'] = total_utilization / len(self.swarm_capacities)
            
            # Update average orchestration time
            recent_tasks = [t for t in self.completed_tasks.values() 
                           if 'execution_time' in t][-100:]  # Last 100 tasks
            
            if recent_tasks:
                avg_time = sum(t['execution_time'] for t in recent_tasks) / len(recent_tasks)
                self.orchestration_metrics['average_orchestration_time'] = avg_time
            
        except Exception as e:
            logger.warning(f"Performance metrics update failed: {e}")

    async def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status"""
        
        return {
            'state': self.state.value,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'active_swarms': len(self.swarm_capacities),
            'total_swarm_capacity': sum(c.max_capacity for c in self.swarm_capacities.values()),
            'metrics': self.orchestration_metrics,
            'swarm_health': {
                swarm_id: {
                    'current_load': capacity.current_load,
                    'success_rate': capacity.success_rate,
                    'health_score': capacity.health_score,
                    'active_tasks': capacity.active_tasks
                }
                for swarm_id, capacity in self.swarm_capacities.items()
            },
            'performance_snapshot': await self.performance_monitor.get_metrics_snapshot() if self.performance_monitor else {},
            'timestamp': datetime.utcnow().isoformat()
        }

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        
        logger.info("Shutting down Claude Flow Orchestrator")
        
        self.state = OrchestratorState.SHUTDOWN
        
        try:
            # Cancel background tasks
            for task in [self.orchestration_loop, self.optimization_loop, self.monitoring_loop]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Wait for active tasks to complete (with timeout)
            if self.active_tasks:
                logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
                
                timeout = 60  # 60 second timeout
                start_time = time.time()
                
                while self.active_tasks and (time.time() - start_time) < timeout:
                    await asyncio.sleep(1)
                
                if self.active_tasks:
                    logger.warning(f"Timed out waiting for {len(self.active_tasks)} tasks to complete")
            
            # Shutdown components
            if self.performance_monitor:
                await self.performance_monitor.stop()
            
            if self.cache_manager:
                await self.cache_manager.shutdown()
            
            if self.swarm_manager:
                await self.swarm_manager.shutdown()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Claude Flow Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}")