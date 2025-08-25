"""API Schemas Package."""

from .auth import TokenResponse, LoginRequest, RegisterRequest
from .user import UserCreate, UserUpdate, UserResponse, UserProfile
from .command import CommandCreate, CommandUpdate, CommandResponse, CommandExecute
from .plugin import PluginCreate, PluginUpdate, PluginResponse, PluginInstall
from .theme import ThemeCreate, ThemeUpdate, ThemeResponse, ThemeApply

# New schemas for extended functionality
from .workflows import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse, WorkflowListResponse,
    WorkflowExecuteRequest, WorkflowExecutionResponse, WorkflowProgressResponse,
    WorkflowStepCreate, WorkflowStepResponse, WorkflowStatus, ExecutionStrategy, AgentType
)
from .analytics import (
    MetricQuery, ReportRequest, AnalyticsResponse, PerformanceMetrics, UsageMetrics,
    SystemMetrics, ProjectAnalytics, WorkflowAnalytics, UserAnalytics,
    TrendAnalysis, DashboardData, ReportResponse, MetricType, TimeRange, AggregationType
)
from .websocket import (
    WebSocketMessage, SubscriptionRequest, BroadcastRequest, UserMessageRequest,
    ConnectionInfo, WebSocketHealthResponse, TaskProgressData, WorkflowProgressData,
    AgentStatusData, ProjectStatusData, SystemNotificationData, ErrorEventData,
    WebSocketEventType, SubscriptionType
)

__all__ = [
    # Auth & User schemas
    "TokenResponse", "LoginRequest", "RegisterRequest",
    "UserCreate", "UserUpdate", "UserResponse", "UserProfile",
    
    # Command & Plugin schemas
    "CommandCreate", "CommandUpdate", "CommandResponse", "CommandExecute",
    "PluginCreate", "PluginUpdate", "PluginResponse", "PluginInstall",
    "ThemeCreate", "ThemeUpdate", "ThemeResponse", "ThemeApply",
    
    # Workflow schemas
    "WorkflowCreate", "WorkflowUpdate", "WorkflowResponse", "WorkflowListResponse",
    "WorkflowExecuteRequest", "WorkflowExecutionResponse", "WorkflowProgressResponse",
    "WorkflowStepCreate", "WorkflowStepResponse", "WorkflowStatus", "ExecutionStrategy", "AgentType",
    
    # Analytics schemas
    "MetricQuery", "ReportRequest", "AnalyticsResponse", "PerformanceMetrics", "UsageMetrics",
    "SystemMetrics", "ProjectAnalytics", "WorkflowAnalytics", "UserAnalytics",
    "TrendAnalysis", "DashboardData", "ReportResponse", "MetricType", "TimeRange", "AggregationType",
    
    # WebSocket schemas
    "WebSocketMessage", "SubscriptionRequest", "BroadcastRequest", "UserMessageRequest",
    "ConnectionInfo", "WebSocketHealthResponse", "TaskProgressData", "WorkflowProgressData",
    "AgentStatusData", "ProjectStatusData", "SystemNotificationData", "ErrorEventData",
    "WebSocketEventType", "SubscriptionType"
]