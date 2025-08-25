"""
Workflow-related Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums
class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStrategy(str, Enum):
    """Execution strategy enumeration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


class AgentType(str, Enum):
    """Agent type enumeration."""
    COORDINATOR = "coordinator"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    ANALYZER = "analyzer"
    ARCHITECT = "architect"
    OPTIMIZER = "optimizer"


# Request Schemas
class WorkflowStepCreate(BaseModel):
    """Schema for creating workflow steps."""
    name: str = Field(..., min_length=1, max_length=200, description="Step name")
    description: str = Field(..., min_length=1, max_length=1000, description="Step description")
    step_type: str = Field(default="task", description="Step type")
    agent_type: AgentType = Field(default=AgentType.CODER, description="Required agent type")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600, description="Step timeout")
    retry_count: int = Field(3, ge=0, le=10, description="Retry attempts")


class WorkflowCreate(BaseModel):
    """Schema for creating workflows."""
    name: str = Field(..., min_length=1, max_length=200, description="Workflow name")
    description: str = Field(..., min_length=1, max_length=1000, description="Workflow description")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    execution_strategy: ExecutionStrategy = Field(default=ExecutionStrategy.ADAPTIVE)
    max_agents: int = Field(5, ge=1, le=20, description="Maximum concurrent agents")
    timeout_seconds: Optional[int] = Field(None, ge=60, le=7200, description="Workflow timeout")
    steps: List[WorkflowStepCreate] = Field(..., min_items=1, description="Workflow steps")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")


class WorkflowUpdate(BaseModel):
    """Schema for updating workflows."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    execution_strategy: Optional[ExecutionStrategy] = None
    max_agents: Optional[int] = Field(None, ge=1, le=20)
    timeout_seconds: Optional[int] = Field(None, ge=60, le=7200)
    configuration: Optional[Dict[str, Any]] = None


class WorkflowExecuteRequest(BaseModel):
    """Schema for workflow execution requests."""
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    agent_preferences: Optional[List[AgentType]] = Field(None, description="Preferred agent types")
    priority_override: Optional[str] = Field(None, description="Priority override")


# Response Schemas
class WorkflowStepResponse(BaseModel):
    """Schema for workflow step responses."""
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
    """Schema for workflow responses."""
    workflow_id: str
    name: str
    description: str
    project_id: Optional[str]
    status: WorkflowStatus
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
    """Schema for workflow list responses."""
    workflows: List[WorkflowResponse]
    total: int
    page: int
    page_size: int


class WorkflowExecutionResponse(BaseModel):
    """Schema for workflow execution responses."""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    progress_percentage: float
    current_step: Optional[str]
    agents_active: int
    steps_completed: int
    steps_total: int
    started_at: str
    estimated_completion: Optional[str] = None


class WorkflowProgressResponse(BaseModel):
    """Schema for workflow progress responses."""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    progress_percentage: float
    current_step: Optional[str]
    agents_active: int
    steps_completed: int
    steps_total: int
    step_details: List[WorkflowStepResponse]
    execution_log: List[Dict[str, Any]]
    metrics: Dict[str, Any]