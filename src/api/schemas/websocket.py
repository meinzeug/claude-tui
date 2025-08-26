"""
WebSocket-related Pydantic schemas for real-time communication validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field
from enum import Enum


# Enums
class WebSocketEventType(str, Enum):
    """WebSocket event types."""
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PROGRESS = "workflow_progress"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    AGENT_SPAWNED = "agent_spawned"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    PROJECT_STATUS_CHANGED = "project_status_changed"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR_OCCURRED = "error_occurred"


class SubscriptionType(str, Enum):
    """Subscription types for filtering events."""
    ALL = "all"
    TASKS = "tasks"
    WORKFLOWS = "workflows"
    PROJECTS = "projects"
    AGENTS = "agents"
    SYSTEM = "system"


# Request Schemas
class SubscriptionRequest(BaseModel):
    """WebSocket subscription request schema."""
    subscription_type: SubscriptionType
    filters: Optional[Dict] = Field(default_factory=dict, description="Subscription filters")
    subscription_id: Optional[str] = Field(None, description="Custom subscription ID")


class BroadcastRequest(BaseModel):
    """Broadcast message request schema."""
    event_type: WebSocketEventType
    data: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = None


class UserMessageRequest(BaseModel):
    """User-specific message request schema."""
    user_id: str
    event_type: WebSocketEventType
    data: Dict[str, Any]


# Response Schemas
class WebSocketMessage(BaseModel):
    """WebSocket message structure schema."""
    event_type: WebSocketEventType
    timestamp: str
    data: Dict[str, Any]
    subscription_id: Optional[str] = None
    user_id: Optional[str] = None


class ConnectionInfo(BaseModel):
    """Connection information schema."""
    active_connections: int
    connection_ids: List[str]
    subscriptions: List[Dict[str, Any]]
    total_system_connections: int
    total_system_subscriptions: int


class WebSocketHealthResponse(BaseModel):
    """WebSocket service health response schema."""
    status: str
    service: str
    active_connections: int
    active_users: int
    active_subscriptions: int
    supported_events: List[str]
    subscription_types: List[str]
    checked_at: str


# Event Data Schemas
class TaskProgressData(BaseModel):
    """Task progress update data schema."""
    task_id: str
    progress_percentage: float
    current_step: Optional[str] = None
    estimated_remaining_seconds: Optional[int] = None
    status: str
    metrics: Optional[Dict[str, Any]] = None


class WorkflowProgressData(BaseModel):
    """Workflow progress update data schema."""
    workflow_id: str
    execution_id: str
    progress_percentage: float
    current_step: Optional[str] = None
    steps_completed: int
    steps_total: int
    agents_active: int
    status: str
    estimated_completion: Optional[str] = None


class AgentStatusData(BaseModel):
    """Agent status update data schema."""
    agent_id: str
    agent_type: str
    status: str
    current_task: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class ProjectStatusData(BaseModel):
    """Project status update data schema."""
    project_id: str
    status: str
    health_score: Optional[float] = None
    last_activity: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class SystemNotificationData(BaseModel):
    """System notification data schema."""
    notification_type: str
    severity: str
    title: str
    message: str
    action_required: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ErrorEventData(BaseModel):
    """Error event data schema."""
    error_id: str
    error_type: str
    severity: str
    component: str
    message: str
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None