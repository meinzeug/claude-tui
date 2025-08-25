"""
Workflow Orchestration REST API Endpoints.

Provides comprehensive workflow management and orchestration:
- Workflow creation and configuration
- Workflow execution and monitoring
- Progress tracking and results
- Agent coordination and task distribution
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..dependencies.auth import get_current_user
from ..middleware.rate_limiting import rate_limit
from ...services.task_service import TaskService
from ...core.exceptions import (
    ValidationError, WorkflowExecutionError, ResourceNotFoundError
)

# Initialize router
router = APIRouter()

# Enums for API
class WorkflowStatusEnum(str, Enum):
    """Workflow status enumeration."""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionStrategyEnum(str, Enum):
    """Execution strategy enumeration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"

class AgentTypeEnum(str, Enum):
    """Agent type enumeration."""
    COORDINATOR = "coordinator"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    ANALYZER = "analyzer"
    ARCHITECT = "architect"
    OPTIMIZER = "optimizer"

# Pydantic Models
class WorkflowStepCreate(BaseModel):
    """Model for creating workflow steps."""
    name: str = Field(..., min_length=1, max_length=200, description="Step name")
    description: str = Field(..., min_length=1, max_length=1000, description="Step description")
    step_type: str = Field(default="task", description="Step type")
    agent_type: AgentTypeEnum = Field(default=AgentTypeEnum.CODER, description="Required agent type")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600, description="Step timeout")
    retry_count: int = Field(3, ge=0, le=10, description="Retry attempts")
    
class WorkflowCreate(BaseModel):
    """Request model for workflow creation."""
    name: str = Field(..., min_length=1, max_length=200, description="Workflow name")
    description: str = Field(..., min_length=1, max_length=1000, description="Workflow description")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    execution_strategy: ExecutionStrategyEnum = Field(default=ExecutionStrategyEnum.ADAPTIVE)
    max_agents: int = Field(5, ge=1, le=20, description="Maximum concurrent agents")
    timeout_seconds: Optional[int] = Field(None, ge=60, le=7200, description="Workflow timeout")
    steps: List[WorkflowStepCreate] = Field(..., min_items=1, description="Workflow steps")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")

class WorkflowUpdate(BaseModel):
    """Request model for workflow updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    execution_strategy: Optional[ExecutionStrategyEnum] = None
    max_agents: Optional[int] = Field(None, ge=1, le=20)
    timeout_seconds: Optional[int] = Field(None, ge=60, le=7200)
    configuration: Optional[Dict[str, Any]] = None

class WorkflowExecuteRequest(BaseModel):
    """Request model for workflow execution."""
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    agent_preferences: Optional[List[AgentTypeEnum]] = Field(None, description="Preferred agent types")
    priority_override: Optional[str] = Field(None, description="Priority override")

class WorkflowStepResponse(BaseModel):
    """Response model for workflow steps."""
    step_id: str
    name: str
    description: str
    step_type: str
    agent_type: str
    status: str
    dependencies: List[str]
    parameters: Dict[str, Any]
    timeout_seconds: Optional[int]
    retry_count: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class WorkflowResponse(BaseModel):
    """Response model for workflow operations."""
    workflow_id: str
    name: str
    description: str
    project_id: Optional[str]
    status: WorkflowStatusEnum
    execution_strategy: str
    max_agents: int
    timeout_seconds: Optional[int]
    steps: List[WorkflowStepResponse]
    configuration: Dict[str, Any]
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_seconds: Optional[float] = None

class WorkflowListResponse(BaseModel):
    """Response model for workflow listing."""
    workflows: List[WorkflowResponse]
    total: int
    page: int
    page_size: int

class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution."""
    workflow_id: str
    execution_id: str
    status: WorkflowStatusEnum
    progress_percentage: float
    current_step: Optional[str]
    agents_active: int
    steps_completed: int
    steps_total: int
    started_at: str
    estimated_completion: Optional[str] = None

class WorkflowProgressResponse(BaseModel):
    """Response model for workflow progress."""
    workflow_id: str
    execution_id: str
    status: WorkflowStatusEnum
    progress_percentage: float
    current_step: Optional[str]
    agents_active: int
    steps_completed: int
    steps_total: int
    step_details: List[WorkflowStepResponse]
    execution_log: List[Dict[str, Any]]
    metrics: Dict[str, Any]

# Dependency injection
async def get_workflow_service():
    """Get workflow service dependency."""
    # This would be replaced with actual service injection
    from ...services.workflow_service import WorkflowService
    service = WorkflowService()
    await service.initialize()
    return service

# Routes
@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
@rate_limit(requests=10, window=60)
async def create_workflow(
    workflow_data: WorkflowCreate,
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Create a new workflow with orchestration configuration.
    
    Creates a workflow definition with steps, dependencies,
    and execution parameters for AI agent coordination.
    """
    try:
        result = await workflow_service.create_workflow(
            name=workflow_data.name,
            description=workflow_data.description,
            project_id=workflow_data.project_id,
            execution_strategy=workflow_data.execution_strategy.value,
            max_agents=workflow_data.max_agents,
            timeout_seconds=workflow_data.timeout_seconds,
            steps=[step.dict() for step in workflow_data.steps],
            configuration=workflow_data.configuration,
            created_by=current_user["id"]
        )
        
        return WorkflowResponse(
            workflow_id=result['workflow_id'],
            name=result['name'],
            description=result['description'],
            project_id=result.get('project_id'),
            status=WorkflowStatusEnum(result['status']),
            execution_strategy=result['execution_strategy'],
            max_agents=result['max_agents'],
            timeout_seconds=result.get('timeout_seconds'),
            steps=[
                WorkflowStepResponse(
                    step_id=step['step_id'],
                    name=step['name'],
                    description=step['description'],
                    step_type=step['step_type'],
                    agent_type=step['agent_type'],
                    status=step['status'],
                    dependencies=step['dependencies'],
                    parameters=step['parameters'],
                    timeout_seconds=step.get('timeout_seconds'),
                    retry_count=step['retry_count']
                )
                for step in result['steps']
            ],
            configuration=result['configuration'],
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )

@router.get("/", response_model=WorkflowListResponse)
@rate_limit(requests=20, window=60)
async def list_workflows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    status_filter: Optional[WorkflowStatusEnum] = Query(None, description="Filter by status"),
    execution_strategy: Optional[ExecutionStrategyEnum] = Query(None, description="Filter by strategy"),
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    List workflows with filtering and pagination.
    
    Returns a paginated list of workflows with optional filtering
    by project, status, and execution strategy.
    """
    try:
        workflows = await workflow_service.list_workflows(
            user_id=current_user["id"],
            project_id=project_id,
            status_filter=status_filter.value if status_filter else None,
            execution_strategy=execution_strategy.value if execution_strategy else None,
            page=page,
            page_size=page_size
        )
        
        return WorkflowListResponse(
            workflows=[
                WorkflowResponse(
                    workflow_id=workflow['workflow_id'],
                    name=workflow['name'],
                    description=workflow['description'],
                    project_id=workflow.get('project_id'),
                    status=WorkflowStatusEnum(workflow['status']),
                    execution_strategy=workflow['execution_strategy'],
                    max_agents=workflow['max_agents'],
                    timeout_seconds=workflow.get('timeout_seconds'),
                    steps=[
                        WorkflowStepResponse(**step) for step in workflow['steps']
                    ],
                    configuration=workflow['configuration'],
                    created_at=workflow['created_at'],
                    updated_at=workflow['updated_at'],
                    started_at=workflow.get('started_at'),
                    completed_at=workflow.get('completed_at'),
                    execution_time_seconds=workflow.get('execution_time_seconds')
                )
                for workflow in workflows['items']
            ],
            total=workflows['total'],
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )

@router.get("/{workflow_id}", response_model=WorkflowResponse)
@rate_limit(requests=30, window=60)
async def get_workflow(
    workflow_id: str,
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Get detailed information about a specific workflow.
    
    Returns comprehensive workflow information including
    steps, configuration, and execution history.
    """
    try:
        workflow = await workflow_service.get_workflow(workflow_id)
        
        return WorkflowResponse(
            workflow_id=workflow['workflow_id'],
            name=workflow['name'],
            description=workflow['description'],
            project_id=workflow.get('project_id'),
            status=WorkflowStatusEnum(workflow['status']),
            execution_strategy=workflow['execution_strategy'],
            max_agents=workflow['max_agents'],
            timeout_seconds=workflow.get('timeout_seconds'),
            steps=[
                WorkflowStepResponse(**step) for step in workflow['steps']
            ],
            configuration=workflow['configuration'],
            created_at=workflow['created_at'],
            updated_at=workflow['updated_at'],
            started_at=workflow.get('started_at'),
            completed_at=workflow.get('completed_at'),
            execution_time_seconds=workflow.get('execution_time_seconds')
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow: {str(e)}"
        )

@router.put("/{workflow_id}", response_model=WorkflowResponse)
@rate_limit(requests=10, window=60)
async def update_workflow(
    workflow_id: str,
    workflow_update: WorkflowUpdate,
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Update workflow configuration and settings.
    
    Updates workflow properties and configuration.
    Cannot modify running workflows.
    """
    try:
        result = await workflow_service.update_workflow(
            workflow_id=workflow_id,
            updates=workflow_update.dict(exclude_unset=True),
            updated_by=current_user["id"]
        )
        
        return WorkflowResponse(**result)
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Update validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow: {str(e)}"
        )

@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
@rate_limit(requests=5, window=60)
async def execute_workflow(
    workflow_id: str,
    execution_request: WorkflowExecuteRequest,
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Execute workflow with AI agent orchestration.
    
    Starts workflow execution with agent coordination,
    dependency resolution, and progress monitoring.
    """
    try:
        result = await workflow_service.execute_workflow(
            workflow_id=workflow_id,
            execution_context=execution_request.execution_context,
            agent_preferences=[
                pref.value for pref in (execution_request.agent_preferences or [])
            ],
            priority_override=execution_request.priority_override,
            executed_by=current_user["id"]
        )
        
        return WorkflowExecutionResponse(
            workflow_id=workflow_id,
            execution_id=result['execution_id'],
            status=WorkflowStatusEnum(result['status']),
            progress_percentage=result['progress_percentage'],
            current_step=result.get('current_step'),
            agents_active=result['agents_active'],
            steps_completed=result['steps_completed'],
            steps_total=result['steps_total'],
            started_at=result['started_at'],
            estimated_completion=result.get('estimated_completion')
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except WorkflowExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow execution error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute workflow: {str(e)}"
        )

@router.get("/{workflow_id}/progress", response_model=WorkflowProgressResponse)
@rate_limit(requests=50, window=60)
async def get_workflow_progress(
    workflow_id: str,
    execution_id: Optional[str] = Query(None, description="Specific execution ID"),
    include_logs: bool = Query(False, description="Include execution logs"),
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Get real-time workflow execution progress.
    
    Returns current execution status, progress metrics,
    and step-by-step details with optional logs.
    """
    try:
        progress = await workflow_service.get_workflow_progress(
            workflow_id=workflow_id,
            execution_id=execution_id,
            include_logs=include_logs
        )
        
        return WorkflowProgressResponse(
            workflow_id=workflow_id,
            execution_id=progress['execution_id'],
            status=WorkflowStatusEnum(progress['status']),
            progress_percentage=progress['progress_percentage'],
            current_step=progress.get('current_step'),
            agents_active=progress['agents_active'],
            steps_completed=progress['steps_completed'],
            steps_total=progress['steps_total'],
            step_details=[
                WorkflowStepResponse(**step) for step in progress['step_details']
            ],
            execution_log=progress.get('execution_log', []),
            metrics=progress.get('metrics', {})
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow progress: {str(e)}"
        )

@router.post("/{workflow_id}/pause")
@rate_limit(requests=10, window=60)
async def pause_workflow(
    workflow_id: str,
    execution_id: Optional[str] = Body(None, embed=True, description="Specific execution ID"),
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Pause workflow execution.
    
    Pauses a running workflow at the current step boundary.
    Agents complete current tasks before pausing.
    """
    try:
        result = await workflow_service.pause_workflow(
            workflow_id=workflow_id,
            execution_id=execution_id,
            paused_by=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Workflow paused successfully",
                "workflow_id": workflow_id,
                "execution_id": result['execution_id'],
                "status": result['status'],
                "paused_at": result['paused_at']
            }
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except WorkflowExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause workflow: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause workflow: {str(e)}"
        )

@router.post("/{workflow_id}/resume")
@rate_limit(requests=10, window=60)
async def resume_workflow(
    workflow_id: str,
    execution_id: Optional[str] = Body(None, embed=True, description="Specific execution ID"),
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Resume paused workflow execution.
    
    Resumes a paused workflow from the last completed step.
    """
    try:
        result = await workflow_service.resume_workflow(
            workflow_id=workflow_id,
            execution_id=execution_id,
            resumed_by=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Workflow resumed successfully",
                "workflow_id": workflow_id,
                "execution_id": result['execution_id'],
                "status": result['status'],
                "resumed_at": result['resumed_at']
            }
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except WorkflowExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume workflow: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume workflow: {str(e)}"
        )

@router.post("/{workflow_id}/cancel")
@rate_limit(requests=10, window=60)
async def cancel_workflow(
    workflow_id: str,
    execution_id: Optional[str] = Body(None, embed=True, description="Specific execution ID"),
    reason: Optional[str] = Body(None, embed=True, description="Cancellation reason"),
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Cancel workflow execution.
    
    Cancels a running or paused workflow. Agents are
    gracefully stopped and cleanup is performed.
    """
    try:
        result = await workflow_service.cancel_workflow(
            workflow_id=workflow_id,
            execution_id=execution_id,
            reason=reason,
            cancelled_by=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Workflow cancelled successfully",
                "workflow_id": workflow_id,
                "execution_id": result['execution_id'],
                "status": result['status'],
                "cancelled_at": result['cancelled_at'],
                "reason": result.get('reason')
            }
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel workflow: {str(e)}"
        )

@router.delete("/{workflow_id}")
@rate_limit(requests=5, window=60)
async def delete_workflow(
    workflow_id: str,
    force: bool = Query(False, description="Force delete even if executions exist"),
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    Delete a workflow definition.
    
    Deletes workflow and optionally all execution history.
    Cannot delete workflows with active executions unless forced.
    """
    try:
        result = await workflow_service.delete_workflow(
            workflow_id=workflow_id,
            force=force,
            deleted_by=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Workflow deleted successfully",
                "workflow_id": workflow_id,
                "deleted_at": result['deleted_at']
            }
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except WorkflowExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete workflow: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete workflow: {str(e)}"
        )

@router.get("/{workflow_id}/executions")
@rate_limit(requests=10, window=60)
async def list_workflow_executions(
    workflow_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Page size"),
    status_filter: Optional[WorkflowStatusEnum] = Query(None, description="Filter by status"),
    current_user: Dict = Depends(get_current_user),
    workflow_service = Depends(get_workflow_service)
):
    """
    List workflow execution history.
    
    Returns paginated list of workflow executions with
    status, timing, and summary information.
    """
    try:
        executions = await workflow_service.list_executions(
            workflow_id=workflow_id,
            status_filter=status_filter.value if status_filter else None,
            page=page,
            page_size=page_size
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "workflow_id": workflow_id,
                "executions": executions['items'],
                "total": executions['total'],
                "page": page,
                "page_size": page_size
            }
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list executions: {str(e)}"
        )