"""Advanced AI Services API Endpoints

Comprehensive REST API for AI services including:
- Swarm orchestration and management
- Agent coordination and communication
- Neural pattern training and inference
- Real-time monitoring and metrics
- WebSocket support for live updates
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis

from ...ai.claude_flow_orchestrator import ClaudeFlowOrchestrator, OrchestrationTask, TaskPriority, ContextType
from ...ai.swarm_manager import EnhancedSwarmManager, SwarmHealthStatus, AgentLifecycleState
from ...ai.agent_coordinator import AgentCoordinator, MessageType, ConsensusType
from ...ai.neural_trainer import NeuralPatternTrainer, PatternType, LearningStrategy
from ...ai.performance_monitor import PerformanceMonitor, MetricType
from ...integrations.claude_flow import SwarmConfig, SwarmTopology, AgentType
from ...integrations.claude_code import ClaudeCodeClient
from ...core.exceptions import SwarmError, AgentError, NeuralTrainingError, OrchestrationError
from ...auth.decorators import require_auth

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/ai/advanced", tags=["AI Advanced Services"])

# Global instances (would be dependency injected in production)
cloud_flow_orchestrator: Optional[ClaudeFlowOrchestrator] = None
enhanced_swarm_manager: Optional[EnhancedSwarmManager] = None
agent_coordinator: Optional[AgentCoordinator] = None
neural_trainer: Optional[NeuralPatternTrainer] = None
performance_monitor: Optional[PerformanceMonitor] = None
redis_client: Optional[redis.Redis] = None
websocket_connections: List[WebSocket] = []


# Pydantic Models for Request/Response
class SwarmInitRequest(BaseModel):
    """Request model for optimized swarm initialization"""
    project_context: Dict[str, Any] = Field(..., description="Project context for optimization")
    preferred_topology: Optional[str] = Field(default=None, description="Preferred topology (if any)")
    resource_limits: Optional[Dict[str, float]] = Field(default=None, description="Resource limits")
    max_agents: int = Field(default=10, ge=1, le=20, description="Maximum number of agents")
    enable_auto_scaling: bool = Field(default=True, description="Enable automatic scaling")
    specialization_focus: Optional[str] = Field(default=None, description="Domain specialization")
    
    @validator('topology')
    def validate_topology(cls, v):
        valid_topologies = [t.value for t in SwarmTopology]
        if v not in valid_topologies:
            raise ValueError(f"Invalid topology. Must be one of: {valid_topologies}")
        return v


class TaskExecutionRequest(BaseModel):
    """Request model for intelligent task orchestration"""
    description: str = Field(..., description="Task description")
    context_type: str = Field(..., description="Context type for execution")
    priority: str = Field(default="medium", description="Task priority level")
    agent_requirements: List[str] = Field(default_factory=list, description="Required agent capabilities")
    estimated_duration: int = Field(default=300, description="Estimated duration in seconds")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    preferred_swarm_id: Optional[str] = Field(default=None, description="Preferred swarm for execution")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Task-specific context data")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    deadline: Optional[str] = Field(default=None, description="Task deadline (ISO format)")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['low', 'medium', 'high', 'critical']:
            raise ValueError("Priority must be: low, medium, high, or critical")
        return v


class AgentSpawnRequest(BaseModel):
    """Request model for agent spawning"""
    swarm_id: str = Field(..., description="Target swarm ID")
    agent_type: str = Field(..., description="Type of agent to spawn")
    name: Optional[str] = Field(default=None, description="Custom agent name")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        valid_types = [t.value for t in AgentType]
        if v not in valid_types:
            raise ValueError(f"Invalid agent type. Must be one of: {valid_types}")
        return v


class NeuralModelRequest(BaseModel):
    """Request model for advanced neural model creation"""
    name: str = Field(..., description="Model name")
    pattern_type: str = Field(..., description="Pattern type to learn")
    strategy: str = Field(default="supervised", description="Learning strategy")
    architecture: Optional[Dict[str, Any]] = Field(default=None, description="Model architecture")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Hyperparameters")
    enable_distributed_training: bool = Field(default=False, description="Enable distributed training")
    transfer_learning_source: Optional[str] = Field(default=None, description="Source model for transfer learning")
    
    @validator('pattern_type')
    def validate_pattern_type(cls, v):
        valid_patterns = [p.value for p in PatternType]
        if v not in valid_patterns:
            raise ValueError(f"Invalid pattern type. Must be one of: {valid_patterns}")
        return v
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = [s.value for s in LearningStrategy]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        return v


class TrainingDataRequest(BaseModel):
    """Request model for training data addition"""
    pattern_type: str = Field(..., description="Pattern type")
    inputs: List[Any] = Field(..., description="Training inputs")
    outputs: List[Any] = Field(..., description="Training outputs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score")
    source: str = Field(default="api", description="Data source")


class TrainingRequest(BaseModel):
    """Request model for neural training"""
    model_id: str = Field(..., description="Model ID to train")
    epochs: Optional[int] = Field(default=None, ge=1, le=1000, description="Number of epochs")
    learning_rate: Optional[float] = Field(default=None, ge=0.0001, le=1.0, description="Learning rate")
    use_claude_flow: bool = Field(default=True, description="Use Claude Flow neural training")


class PredictionRequest(BaseModel):
    """Request model for neural predictions"""
    model_id: str = Field(..., description="Model ID for prediction")
    inputs: List[Any] = Field(..., description="Prediction inputs")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold")


class ConsensusRequest(BaseModel):
    """Request model for consensus proposal"""
    proposer_id: str = Field(..., description="Proposer agent ID")
    proposal: Dict[str, Any] = Field(..., description="Proposal content")
    consensus_type: str = Field(default="majority", description="Consensus mechanism")
    participants: Optional[List[str]] = Field(default=None, description="Specific participants")
    deadline_minutes: int = Field(default=5, ge=1, le=60, description="Deadline in minutes")
    
    @validator('consensus_type')
    def validate_consensus_type(cls, v):
        valid_types = [t.value for t in ConsensusType]
        if v not in valid_types:
            raise ValueError(f"Invalid consensus type. Must be one of: {valid_types}")
        return v


# Additional Request Models for Enhanced Features
class EnsembleRequest(BaseModel):
    """Request model for model ensemble creation"""
    ensemble_name: str = Field(..., description="Ensemble name")
    model_ids: List[str] = Field(..., description="List of model IDs to ensemble")
    voting_strategy: str = Field(default="majority", description="Voting strategy for ensemble")
    
    @validator('model_ids')
    def validate_model_ids(cls, v):
        if len(v) < 2:
            raise ValueError("Ensemble requires at least 2 models")
        return v


class TransferLearningRequest(BaseModel):
    """Request model for transfer learning setup"""
    source_model_id: str = Field(..., description="Source model ID for transfer learning")
    target_pattern_type: str = Field(..., description="Target pattern type")
    target_model_name: str = Field(..., description="Name for target model")
    freeze_layers: int = Field(default=2, ge=0, le=10, description="Number of layers to freeze")
    
    @validator('target_pattern_type')
    def validate_pattern_type(cls, v):
        valid_patterns = [p.value for p in PatternType]
        if v not in valid_patterns:
            raise ValueError(f"Invalid pattern type. Must be one of: {valid_patterns}")
        return v


class HyperparameterOptimizationRequest(BaseModel):
    """Request model for hyperparameter optimization"""
    model_id: str = Field(..., description="Model ID to optimize")
    parameter_ranges: Dict[str, List[float]] = Field(..., description="Parameter ranges to explore")
    optimization_trials: int = Field(default=20, ge=5, le=100, description="Number of optimization trials")
    optimization_metric: str = Field(default="accuracy", description="Metric to optimize")
    
    @validator('parameter_ranges')
    def validate_parameter_ranges(cls, v):
        for param, range_vals in v.items():
            if len(range_vals) != 2 or range_vals[0] >= range_vals[1]:
                raise ValueError(f"Invalid range for parameter {param}. Must be [min, max] with min < max")
        return v


class SwarmScalingRequest(BaseModel):
    """Request model for intelligent swarm scaling"""
    swarm_id: str = Field(..., description="Swarm ID to scale")
    target_performance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Target performance level")
    scaling_strategy: str = Field(default="intelligent", description="Scaling strategy")
    min_agents: Optional[int] = Field(default=None, ge=1, description="Minimum number of agents")
    max_agents: Optional[int] = Field(default=None, le=20, description="Maximum number of agents")


# Response Models
class SwarmResponse(BaseModel):
    """Response model for swarm operations"""
    swarm_id: str
    status: str
    message: str
    topology: Optional[str] = None
    agents_count: Optional[int] = None
    created_at: Optional[str] = None


class TaskResponse(BaseModel):
    """Response model for task operations"""
    task_id: str
    execution_id: str
    status: str
    message: str
    swarm_id: Optional[str] = None
    estimated_completion: Optional[str] = None


class ModelResponse(BaseModel):
    """Response model for neural model operations"""
    model_id: str
    status: str
    message: str
    pattern_type: Optional[str] = None
    created_at: Optional[str] = None


# Dependency functions
async def get_claude_flow_orchestrator() -> ClaudeFlowOrchestrator:
    """Get Claude Flow orchestrator instance"""
    global cloud_flow_orchestrator
    if cloud_flow_orchestrator is None:
        cloud_flow_orchestrator = ClaudeFlowOrchestrator()
        await cloud_flow_orchestrator.initialize()
    return cloud_flow_orchestrator


async def get_enhanced_swarm_manager() -> EnhancedSwarmManager:
    """Get enhanced swarm manager instance"""
    global enhanced_swarm_manager
    if enhanced_swarm_manager is None:
        enhanced_swarm_manager = EnhancedSwarmManager()
        await enhanced_swarm_manager.initialize()
    return enhanced_swarm_manager


async def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
        await performance_monitor.start()
    return performance_monitor


async def get_agent_coordinator() -> AgentCoordinator:
    """Get agent coordinator instance"""
    global agent_coordinator
    if agent_coordinator is None:
        agent_coordinator = AgentCoordinator()
        await agent_coordinator.start()
    return agent_coordinator


async def get_neural_trainer() -> NeuralPatternTrainer:
    """Get enhanced neural trainer instance"""
    global neural_trainer
    if neural_trainer is None:
        orchestrator = await get_claude_flow_orchestrator()
        perf_monitor = await get_performance_monitor()
        neural_trainer = NeuralPatternTrainer(
            orchestrator=orchestrator,
            performance_monitor=perf_monitor
        )
        await neural_trainer.start()
    return neural_trainer


async def get_redis_client() -> redis.Redis:
    """Get Redis client for caching"""
    global redis_client
    if redis_client is None:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url)
    return redis_client


# Swarm Management Endpoints
@router.post("/swarm/init", response_model=SwarmResponse)
@require_auth
async def initialize_swarm(
    request: SwarmInitRequest,
    orchestrator: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """Initialize a new swarm with specified configuration"""
    
    try:
        # Create swarm configuration
        swarm_config = SwarmConfig(
            topology=SwarmTopology(request.topology),
            max_agents=request.max_agents,
            strategy=request.strategy,
            enable_coordination=request.enable_coordination,
            enable_learning=request.enable_learning
        )
        
        # Use project spec if provided, otherwise create basic spec
        project_spec = request.project_spec or {
            'description': 'API-initiated swarm',
            'features': [],
            'estimated_complexity': 50
        }
        
        # Initialize swarm
        swarm_id = await orchestrator.initialize_swarm(project_spec, swarm_config)
        
        # Cache swarm info
        cache_key = f"swarm:{swarm_id}"
        await cache_swarm_info(cache_key, {
            'swarm_id': swarm_id,
            'topology': request.topology,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        })
        
        # Broadcast swarm creation
        await broadcast_update({
            'type': 'swarm_created',
            'swarm_id': swarm_id,
            'topology': request.topology,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return SwarmResponse(
            swarm_id=swarm_id,
            status="initialized",
            message=f"Swarm initialized with {request.topology} topology",
            topology=request.topology,
            agents_count=0,
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize swarm: {e}")
        raise HTTPException(status_code=500, detail=f"Swarm initialization failed: {str(e)}")


@router.get("/swarm/{swarm_id}/status")
@require_auth
async def get_swarm_status(
    swarm_id: str,
    orchestrator: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """Get comprehensive swarm status"""
    
    try:
        status = await orchestrator.get_swarm_status(swarm_id)
        
        # Add cached information
        cache_key = f"swarm:{swarm_id}"
        cached_info = await get_cached_info(cache_key)
        if cached_info:
            status.update(cached_info)
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get swarm status: {e}")
        raise HTTPException(status_code=404, detail=f"Swarm {swarm_id} not found or error: {str(e)}")


@router.post("/swarm/{swarm_id}/scale")
@require_auth
async def scale_swarm(
    swarm_id: str,
    target_agents: int = Field(..., ge=1, le=20),
    agent_types: Optional[List[str]] = None,
    orchestrator: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """Scale swarm agents up or down"""
    
    try:
        # Validate agent types if provided
        if agent_types:
            valid_types = [t.value for t in AgentType]
            invalid_types = [at for at in agent_types if at not in valid_types]
            if invalid_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid agent types: {invalid_types}"
                )
            agent_type_enums = [AgentType(at) for at in agent_types]
        else:
            agent_type_enums = None
        
        success = await orchestrator.scale_swarm(swarm_id, target_agents, agent_type_enums)
        
        if success:
            # Broadcast scaling update
            await broadcast_update({
                'type': 'swarm_scaled',
                'swarm_id': swarm_id,
                'target_agents': target_agents,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return {
                'swarm_id': swarm_id,
                'status': 'scaled',
                'target_agents': target_agents,
                'message': f'Swarm scaled to {target_agents} agents'
            }
        else:
            raise HTTPException(status_code=500, detail="Scaling operation failed")
            
    except SwarmError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to scale swarm: {e}")
        raise HTTPException(status_code=500, detail=f"Scaling failed: {str(e)}")


# Task Execution Endpoints
@router.post("/task/execute", response_model=TaskResponse)
@require_auth
async def execute_task(
    request: TaskExecutionRequest,
    background_tasks: BackgroundTasks,
    orchestrator: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """Execute a task using optimal swarm selection"""
    
    try:
        # Create task request
        task_request = TaskRequest(
            task_id=f"task-{int(time.time())}",
            description=request.description,
            priority=request.priority,
            agent_requirements=request.agent_requirements,
            estimated_complexity=request.estimated_complexity,
            max_execution_time=request.max_execution_time,
            context=request.context
        )
        
        # Execute task
        execution_id = await orchestrator.execute_task(task_request, request.swarm_id)
        
        # Estimate completion time
        estimated_completion = (
            datetime.utcnow() + timedelta(seconds=request.max_execution_time)
        ).isoformat()
        
        # Cache task info
        cache_key = f"task:{task_request.task_id}"
        await cache_task_info(cache_key, {
            'task_id': task_request.task_id,
            'execution_id': execution_id,
            'description': request.description,
            'status': 'running',
            'created_at': datetime.utcnow().isoformat()
        })
        
        # Add background monitoring
        background_tasks.add_task(monitor_task_execution, execution_id, task_request.task_id)
        
        # Broadcast task start
        await broadcast_update({
            'type': 'task_started',
            'task_id': task_request.task_id,
            'execution_id': execution_id,
            'description': request.description[:100],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return TaskResponse(
            task_id=task_request.task_id,
            execution_id=execution_id,
            status="running",
            message="Task execution started",
            swarm_id=request.swarm_id,
            estimated_completion=estimated_completion
        )
        
    except Exception as e:
        logger.error(f"Failed to execute task: {e}")
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@router.get("/task/{task_id}/status")
@require_auth
async def get_task_status(
    task_id: str,
    orchestrator: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """Get task execution status"""
    
    try:
        # Get from cache first
        cache_key = f"task:{task_id}"
        cached_info = await get_cached_info(cache_key)
        
        if cached_info:
            # Try to get real-time status from orchestrator
            # This would require implementing task status tracking
            return cached_info
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


# Agent Management Endpoints
@router.post("/agents/spawn")
@require_auth
async def spawn_agent(
    request: AgentSpawnRequest,
    orchestrator: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """Spawn a new agent in specified swarm"""
    
    try:
        agent = await orchestrator.swarm_manager.spawn_agent(
            request.swarm_id,
            AgentType(request.agent_type),
            request.name,
            request.capabilities
        )
        
        # Broadcast agent spawn
        await broadcast_update({
            'type': 'agent_spawned',
            'swarm_id': request.swarm_id,
            'agent_id': agent.id,
            'agent_type': request.agent_type,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return {
            'agent_id': agent.id,
            'swarm_id': request.swarm_id,
            'type': request.agent_type,
            'name': agent.name,
            'status': 'spawned',
            'capabilities': agent.capabilities,
            'created_at': agent.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to spawn agent: {e}")
        raise HTTPException(status_code=500, detail=f"Agent spawn failed: {str(e)}")


@router.get("/agents/{agent_id}/status")
@require_auth
async def get_agent_status(
    agent_id: str,
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Get comprehensive agent status"""
    
    try:
        status = await coordinator.get_agent_status(agent_id)
        return status
        
    except AgentError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


# Neural Training Endpoints
@router.post("/neural/models", response_model=ModelResponse)
@require_auth
async def create_neural_model(
    request: NeuralModelRequest,
    trainer: NeuralPatternTrainer = Depends(get_neural_trainer)
):
    """Create a new neural model"""
    
    try:
        model_id = await trainer.create_model(
            request.name,
            PatternType(request.pattern_type),
            LearningStrategy(request.strategy),
            request.architecture,
            request.hyperparameters
        )
        
        return ModelResponse(
            model_id=model_id,
            status="created",
            message=f"Neural model '{request.name}' created successfully",
            pattern_type=request.pattern_type,
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create neural model: {e}")
        raise HTTPException(status_code=500, detail=f"Model creation failed: {str(e)}")


@router.post("/neural/training-data")
@require_auth
async def add_training_data(
    request: TrainingDataRequest,
    trainer: NeuralPatternTrainer = Depends(get_neural_trainer)
):
    """Add training data for neural patterns"""
    
    try:
        data_id = await trainer.add_training_data(
            PatternType(request.pattern_type),
            request.inputs,
            request.outputs,
            request.metadata,
            request.quality_score,
            request.source
        )
        
        return {
            'data_id': data_id,
            'pattern_type': request.pattern_type,
            'samples_count': len(request.inputs),
            'quality_score': request.quality_score,
            'status': 'added',
            'created_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to add training data: {e}")
        raise HTTPException(status_code=500, detail=f"Training data addition failed: {str(e)}")


@router.post("/neural/train")
@require_auth
async def train_neural_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    trainer: NeuralPatternTrainer = Depends(get_neural_trainer)
):
    """Train a neural model"""
    
    try:
        # Start training in background
        background_tasks.add_task(
            execute_training,
            trainer,
            request.model_id,
            request.epochs,
            request.learning_rate,
            request.use_claude_flow
        )
        
        return {
            'model_id': request.model_id,
            'status': 'training_started',
            'message': 'Neural training started in background',
            'epochs': request.epochs,
            'learning_rate': request.learning_rate,
            'use_claude_flow': request.use_claude_flow,
            'started_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Training start failed: {str(e)}")


@router.post("/neural/predict")
@require_auth
async def neural_predict(
    request: PredictionRequest,
    trainer: NeuralPatternTrainer = Depends(get_neural_trainer)
):
    """Make predictions using trained neural model"""
    
    try:
        result = await trainer.predict(
            request.model_id,
            request.inputs,
            request.confidence_threshold
        )
        
        return {
            **result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/neural/models/{model_id}/status")
@require_auth
async def get_neural_model_status(
    model_id: str,
    trainer: NeuralPatternTrainer = Depends(get_neural_trainer)
):
    """Get neural model status and training progress"""
    
    try:
        status = await trainer.get_model_status(model_id)
        return status
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found or error: {str(e)}")


# Consensus and Coordination Endpoints
@router.post("/consensus/propose")
@require_auth
async def propose_consensus(
    request: ConsensusRequest,
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Propose a consensus decision to agents"""
    
    try:
        proposal_id = await coordinator.propose_consensus(
            request.proposer_id,
            request.proposal,
            ConsensusType(request.consensus_type),
            request.participants,
            request.deadline_minutes
        )
        
        return {
            'proposal_id': proposal_id,
            'proposer_id': request.proposer_id,
            'consensus_type': request.consensus_type,
            'deadline_minutes': request.deadline_minutes,
            'status': 'proposed',
            'created_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to propose consensus: {e}")
        raise HTTPException(status_code=500, detail=f"Consensus proposal failed: {str(e)}")


@router.post("/consensus/{proposal_id}/vote")
@require_auth
async def vote_on_consensus(
    proposal_id: str,
    agent_id: str,
    vote: bool,
    weight: float = 1.0,
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Submit a vote for consensus proposal"""
    
    try:
        success = await coordinator.vote_on_proposal(agent_id, proposal_id, vote, weight)
        
        if success:
            return {
                'proposal_id': proposal_id,
                'agent_id': agent_id,
                'vote': vote,
                'weight': weight,
                'status': 'voted',
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Vote submission failed")
            
    except Exception as e:
        logger.error(f"Failed to submit vote: {e}")
        raise HTTPException(status_code=500, detail=f"Vote submission failed: {str(e)}")


# Monitoring and Metrics Endpoints
@router.get("/metrics/swarm")
@require_auth
async def get_swarm_metrics(
    orchestrator: SwarmOrchestrator = Depends(get_swarm_orchestrator)
):
    """Get comprehensive swarm orchestration metrics"""
    
    try:
        metrics = orchestrator.get_global_metrics()
        return {
            **metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get swarm metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@router.get("/metrics/coordination")
@require_auth
async def get_coordination_metrics(
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Get agent coordination metrics"""
    
    try:
        metrics = coordinator.get_coordination_metrics()
        return {
            **metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get coordination metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@router.get("/metrics/neural")
@require_auth
async def get_neural_metrics(
    trainer: NeuralPatternTrainer = Depends(get_neural_trainer)
):
    """Get neural training metrics"""
    
    try:
        metrics = trainer.get_training_metrics()
        return {
            **metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get neural metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


# WebSocket Endpoint for Real-time Updates
@router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            'type': 'connection_established',
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'Real-time monitoring connected'
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping/pong, subscriptions, etc.)
                data = await websocket.receive_json()
                
                # Handle client requests
                if data.get('type') == 'ping':
                    await websocket.send_json({
                        'type': 'pong',
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    finally:
        # Remove from connections
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


# Streaming Response for Long-running Operations
@router.get("/stream/training/{model_id}")
@require_auth
async def stream_training_progress(
    model_id: str,
    trainer: NeuralPatternTrainer = Depends(get_neural_trainer)
):
    """Stream training progress for a neural model"""
    
    async def generate_progress():
        """Generate training progress updates"""
        try:
            while True:
                # Get current training status
                status = await trainer.get_model_status(model_id)
                
                # Format as Server-Sent Events
                data = json.dumps({
                    'model_id': model_id,
                    'status': status,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                yield f"data: {data}\n\n"
                
                # Check if training is complete
                if status.get('active_session') is None:
                    break
                    
                await asyncio.sleep(2)  # Update every 2 seconds
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {{\"error\": \"{str(e)}\", \"timestamp\": \"{datetime.utcnow().isoformat()}\"}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


# Helper Functions
async def cache_swarm_info(cache_key: str, info: Dict[str, Any]):
    """Cache swarm information"""
    try:
        redis_client = await get_redis_client()
        await redis_client.setex(cache_key, 3600, json.dumps(info))  # 1 hour expiry
    except Exception as e:
        logger.warning(f"Failed to cache swarm info: {e}")


async def cache_task_info(cache_key: str, info: Dict[str, Any]):
    """Cache task information"""
    try:
        redis_client = await get_redis_client()
        await redis_client.setex(cache_key, 7200, json.dumps(info))  # 2 hours expiry
    except Exception as e:
        logger.warning(f"Failed to cache task info: {e}")


async def get_cached_info(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached information"""
    try:
        redis_client = await get_redis_client()
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Failed to get cached info: {e}")
    return None


async def broadcast_update(update: Dict[str, Any]):
    """Broadcast update to all WebSocket connections"""
    if not websocket_connections:
        return
        
    # Send to all connected clients
    disconnected = []
    for websocket in websocket_connections:
        try:
            await websocket.send_json(update)
        except Exception:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        if ws in websocket_connections:
            websocket_connections.remove(ws)


async def monitor_task_execution(execution_id: str, task_id: str):
    """Background task to monitor task execution"""
    try:
        # Monitor task execution and update cache
        # This would integrate with the orchestrator to track progress
        await asyncio.sleep(10)  # Placeholder monitoring
        
        # Update cache with completion
        cache_key = f"task:{task_id}"
        await cache_task_info(cache_key, {
            'task_id': task_id,
            'execution_id': execution_id,
            'status': 'completed',
            'completed_at': datetime.utcnow().isoformat()
        })
        
        # Broadcast completion
        await broadcast_update({
            'type': 'task_completed',
            'task_id': task_id,
            'execution_id': execution_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Task monitoring error: {e}")


async def execute_training(
    trainer: NeuralPatternTrainer,
    model_id: str,
    epochs: Optional[int],
    learning_rate: Optional[float],
    use_claude_flow: bool
):
    """Background task to execute neural training"""
    try:
        session_id = await trainer.train_model(
            model_id, epochs, learning_rate, use_claude_flow
        )
        
        # Broadcast training completion
        await broadcast_update({
            'type': 'training_completed',
            'model_id': model_id,
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Training execution error: {e}")
        
        # Broadcast training failure
        await broadcast_update({
            'type': 'training_failed',
            'model_id': model_id,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
