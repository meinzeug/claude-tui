"""
Task Management REST API Endpoints.

Provides comprehensive task orchestration and management:
- Task creation and configuration
- Task execution and monitoring
- Progress tracking and results
- Performance metrics and history
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..dependencies.auth import get_current_user
from ..middleware.rate_limiting import rate_limit
from ...services.task_service import TaskService, TaskStatus, TaskExecutionMode
from ...core.types import Priority
from ...core.exceptions import (
    TaskExecutionTimeoutError, ValidationError, PerformanceError
)

# Initialize router
router = APIRouter()

# Enums for API
class TaskStatusEnum(str, Enum):
    """Task status enumeration for API."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskExecutionModeEnum(str, Enum):
    """Task execution mode enumeration for API."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"

class TaskPriorityEnum(str, Enum):
    """Task priority enumeration for API."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Pydantic Models
class TaskCreateRequest(BaseModel):
    """Request model for task creation."""
    name: str = Field(..., min_length=1, max_length=200, description="Task name")
    description: str = Field(..., min_length=1, max_length=1000, description="Task description")
    task_type: str = Field(default="general", description="Task type")
    priority: TaskPriorityEnum = Field(default=TaskPriorityEnum.MEDIUM, description="Task priority")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600, description="Task timeout in seconds")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Task dependencies")
    ai_enabled: bool = Field(default=True, description="Enable AI assistance")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task configuration")
    
    @validator('task_type')
    def validate_task_type(cls, v):
        allowed_types = ['general', 'code_generation', 'analysis', 'testing', 'deployment', 'task_orchestration']
        if v not in allowed_types:
            raise ValueError(f'Task type must be one of: {", ".join(allowed_types)}')
        return v

class TaskExecuteRequest(BaseModel):
    """Request model for task execution."""
    execution_mode: TaskExecutionModeEnum = Field(default=TaskExecutionModeEnum.ADAPTIVE, description="Execution mode")
    wait_for_dependencies: bool = Field(default=True, description="Wait for dependencies to complete")

class TaskResponse(BaseModel):
    """Response model for task operations."""
    task_id: str
    name: str
    description: str
    task_type: str
    priority: str
    status: TaskStatusEnum
    ai_enabled: bool
    timeout_seconds: int
    dependencies: List[str]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TaskListResponse(BaseModel):
    """Response model for task listing."""
    tasks: List[TaskResponse]
    total: int
    page: int
    page_size: int

class TaskExecutionResponse(BaseModel):
    """Response model for task execution."""
    task_id: str
    status: TaskStatusEnum
    execution_time_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    completed_at: Optional[str] = None

class TaskHistoryResponse(BaseModel):
    """Response model for task execution history."""
    history: List[Dict[str, Any]]
    total: int
    success_rate: float

class TaskPerformanceResponse(BaseModel):
    """Response model for task performance metrics."""
    overall_metrics: Dict[str, Any]
    success_rate: float
    recent_success_rate: float
    active_tasks_count: int
    history_size: int
    task_type_metrics: Dict[str, Any]
    report_generated_at: str

# Dependency injection
async def get_task_service() -> TaskService:
    """Get task service dependency."""
    service = TaskService()
    await service.initialize()
    return service

def _priority_to_enum(priority_str: str) -> Priority:
    """Convert string priority to Priority enum."""
    mapping = {
        'low': Priority.LOW,
        'medium': Priority.MEDIUM,
        'high': Priority.HIGH,
        'critical': Priority.CRITICAL
    }
    return mapping.get(priority_str.lower(), Priority.MEDIUM)

def _execution_mode_to_enum(mode_str: str) -> TaskExecutionMode:
    """Convert string execution mode to TaskExecutionMode enum."""
    mapping = {
        'sequential': TaskExecutionMode.SEQUENTIAL,
        'parallel': TaskExecutionMode.PARALLEL,
        'priority_based': TaskExecutionMode.PRIORITY_BASED,
        'adaptive': TaskExecutionMode.ADAPTIVE
    }
    return mapping.get(mode_str.lower(), TaskExecutionMode.ADAPTIVE)

# Routes
@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
@rate_limit(requests=10, window=60)  # 10 requests per minute
async def create_task(
    task_data: TaskCreateRequest,
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Create a new task with AI-powered execution capabilities.
    
    Creates a new task with specified configuration including:
    - Task type and priority settings
    - AI assistance configuration
    - Dependency management
    - Timeout and execution parameters
    """
    try:
        result = await task_service.create_task(
            name=task_data.name,
            description=task_data.description,
            task_type=task_data.task_type,
            priority=_priority_to_enum(task_data.priority.value),
            timeout_seconds=task_data.timeout_seconds,
            dependencies=task_data.dependencies,
            ai_enabled=task_data.ai_enabled,
            config=task_data.config
        )
        
        return TaskResponse(
            task_id=result['task_id'],
            name=result['name'],
            task_type=result['type'],
            priority=result['priority'],
            status=TaskStatusEnum(result['status']),
            ai_enabled=result['ai_enabled'],
            timeout_seconds=task_data.timeout_seconds or 300,
            dependencies=result['dependencies'],
            created_at=result['created_at'],
            description=task_data.description
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )

@router.get("/", response_model=TaskListResponse)
@rate_limit(requests=20, window=60)
async def list_tasks(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    task_type_filter: Optional[str] = Query(None, description="Filter by task type"),
    status_filter: Optional[TaskStatusEnum] = Query(None, description="Filter by status"),
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    List active tasks with filtering and pagination.
    
    Returns a paginated list of tasks with optional filtering
    by task type and status.
    """
    try:
        tasks = await task_service.list_active_tasks()
        
        # Apply filters
        if task_type_filter:
            tasks = [task for task in tasks if task.get('type') == task_type_filter]
        
        if status_filter:
            tasks = [task for task in tasks if task.get('status') == status_filter.value]
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_tasks = tasks[start_idx:end_idx]
        
        task_responses = [
            TaskResponse(
                task_id=task['id'],
                name=task['name'],
                description=task['description'],
                task_type=task['type'],
                priority=task['priority'],
                status=TaskStatusEnum(task['status']),
                ai_enabled=task.get('ai_enabled', True),
                timeout_seconds=task.get('timeout_seconds', 300),
                dependencies=task.get('dependencies', []),
                created_at=task.get('created_at', ''),
                started_at=task.get('started_at'),
                completed_at=task.get('completed_at'),
                execution_time_seconds=task.get('execution_time_seconds'),
                result=task.get('result'),
                error=task.get('error')
            )
            for task in paginated_tasks
        ]
        
        return TaskListResponse(
            tasks=task_responses,
            total=len(tasks),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )

@router.get("/{task_id}", response_model=TaskResponse)
@rate_limit(requests=30, window=60)
async def get_task(
    task_id: str,
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Get detailed information about a specific task.
    
    Returns comprehensive task information including status,
    execution progress, and results.
    """
    try:
        task_info = await task_service.get_task_status(task_id)
        
        return TaskResponse(
            task_id=task_info['id'],
            name=task_info['name'],
            description=task_info['description'],
            task_type=task_info['type'],
            priority=task_info['priority'],
            status=TaskStatusEnum(task_info['status']),
            ai_enabled=task_info.get('ai_enabled', True),
            timeout_seconds=task_info.get('timeout_seconds', 300),
            dependencies=task_info.get('dependencies', []),
            created_at=task_info.get('created_at', ''),
            started_at=task_info.get('started_at'),
            completed_at=task_info.get('completed_at'),
            execution_time_seconds=task_info.get('execution_time_seconds'),
            result=task_info.get('result'),
            error=task_info.get('error')
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task: {str(e)}"
        )

@router.post("/{task_id}/execute", response_model=TaskExecutionResponse)
@rate_limit(requests=10, window=60)
async def execute_task(
    task_id: str,
    execution_request: TaskExecuteRequest,
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Execute a specific task with AI assistance.
    
    Starts task execution with the specified execution mode
    and dependency handling settings.
    """
    try:
        result = await task_service.execute_task(
            task_id=task_id,
            execution_mode=_execution_mode_to_enum(execution_request.execution_mode.value),
            wait_for_dependencies=execution_request.wait_for_dependencies
        )
        
        return TaskExecutionResponse(
            task_id=result['task_id'],
            status=TaskStatusEnum(result['status']),
            execution_time_seconds=result.get('execution_time_seconds'),
            result=result.get('result'),
            completed_at=result.get('completed_at')
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task execution error: {str(e)}"
        )
    except TaskExecutionTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Task execution timeout: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute task: {str(e)}"
        )

@router.post("/{task_id}/cancel")
@rate_limit(requests=10, window=60)
async def cancel_task(
    task_id: str,
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Cancel a pending or running task.
    
    Cancels task execution if it's in pending or running state.
    """
    try:
        result = await task_service.cancel_task(task_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Task cancelled successfully",
                "task_id": result['task_id'],
                "status": result['status'],
                "cancelled_at": result['cancelled_at']
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task cancellation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )

@router.get("/history/execution", response_model=TaskHistoryResponse)
@rate_limit(requests=10, window=60)
async def get_execution_history(
    limit: Optional[int] = Query(50, ge=1, le=1000, description="Limit number of results"),
    task_type_filter: Optional[str] = Query(None, description="Filter by task type"),
    success_only: Optional[bool] = Query(None, description="Show only successful tasks"),
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Get task execution history with filtering options.
    
    Returns historical task execution data with optional filtering
    by task type and success status.
    """
    try:
        history = await task_service.get_execution_history(
            limit=limit,
            task_type_filter=task_type_filter,
            success_only=success_only
        )
        
        total_tasks = len(history)
        successful_tasks = sum(1 for task in history if task.get('success', False))
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        return TaskHistoryResponse(
            history=history,
            total=total_tasks,
            success_rate=success_rate
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution history: {str(e)}"
        )

@router.get("/performance/report", response_model=TaskPerformanceResponse)
@rate_limit(requests=5, window=60)
async def get_performance_report(
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Get comprehensive task performance metrics and analytics.
    
    Returns detailed performance metrics including success rates,
    execution times, and task type breakdowns.
    """
    try:
        report = await task_service.get_performance_report()
        
        return TaskPerformanceResponse(
            overall_metrics=report['overall_metrics'],
            success_rate=report['success_rate'],
            recent_success_rate=report['recent_success_rate'],
            active_tasks_count=report['active_tasks_count'],
            history_size=report['history_size'],
            task_type_metrics=report['task_type_metrics'],
            report_generated_at=report['report_generated_at']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance report: {str(e)}"
        )

@router.post("/batch/create", response_model=List[TaskResponse])
@rate_limit(requests=3, window=60)
async def create_batch_tasks(
    tasks: List[TaskCreateRequest] = Body(..., max_items=10, description="List of tasks to create"),
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Create multiple tasks in a single batch operation.
    
    Creates up to 10 tasks simultaneously with dependency
    resolution and error handling.
    """
    try:
        created_tasks = []
        
        for task_data in tasks:
            try:
                result = await task_service.create_task(
                    name=task_data.name,
                    description=task_data.description,
                    task_type=task_data.task_type,
                    priority=_priority_to_enum(task_data.priority.value),
                    timeout_seconds=task_data.timeout_seconds,
                    dependencies=task_data.dependencies,
                    ai_enabled=task_data.ai_enabled,
                    config=task_data.config
                )
                
                created_tasks.append(TaskResponse(
                    task_id=result['task_id'],
                    name=result['name'],
                    task_type=result['type'],
                    priority=result['priority'],
                    status=TaskStatusEnum(result['status']),
                    ai_enabled=result['ai_enabled'],
                    timeout_seconds=task_data.timeout_seconds or 300,
                    dependencies=result['dependencies'],
                    created_at=result['created_at'],
                    description=task_data.description
                ))
                
            except Exception as e:
                # Log error but continue with other tasks
                continue
        
        if not created_tasks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No tasks could be created"
            )
        
        return created_tasks
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch tasks: {str(e)}"
        )

@router.get("/health/service")
@rate_limit(requests=20, window=60)
async def task_service_health(
    current_user: Dict = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Get task service health status and diagnostics.
    
    Returns service health information including component
    availability and performance metrics.
    """
    try:
        health = await task_service.health_check()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy" if health['status'] == 'healthy' else "degraded",
                "service": "task_service",
                "components": {
                    "task_engine": health.get('task_engine_available', False),
                    "ai_service": health.get('ai_service_available', False)
                },
                "metrics": {
                    "active_tasks": health.get('active_tasks_count', 0),
                    "queued_tasks": health.get('queued_tasks_count', 0),
                    "history_size": health.get('execution_history_size', 0)
                },
                "performance": health.get('performance_metrics', {}),
                "checked_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "task_service",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
        )