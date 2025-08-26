#!/usr/bin/env python3
"""
UI Widgets Module - All custom widgets for the Claude-TIU interface
"""

from .console_widget import (
    ConsoleWidget,
    MessageType,
    AITaskStatus,
    ConsoleMessage,
    AITask,
    ShowCommandTemplatesMessage
)

from .notification_system import (
    NotificationSystem,
    Notification,
    NotificationLevel,
    NotificationCategory,
    ExportHistoryMessage
)

from .placeholder_alert import (
    PlaceholderAlert,
    PlaceholderIssue,
    PlaceholderType,
    PlaceholderSeverity,
    PlaceholderAlertTriggeredMessage,
    AutoFixIssuesMessage,
    StartCompletionMessage,
    ExportPlaceholderReportMessage
)

from .progress_intelligence import (
    ProgressIntelligence,
    ProgressReport,
    ValidationStatus,
    ValidateNowMessage,
    ShowValidationDetailsMessage
)

from .project_tree import (
    ProjectTree,
    FileSelectedMessage,
    TreeRefreshedMessage
)

from .task_dashboard import (
    TaskDashboard,
    ProjectTask,
    TaskStatus,
    TaskPriority,
    AddTaskMessage,
    ShowAnalyticsMessage
)

# Import new widgets
from .workflow_visualizer import (
    WorkflowVisualizerWidget,
    WorkflowStatus,
    TaskNodeStatus, 
    TaskNode,
    WorkflowDefinition,
    StartWorkflowMessage,
    PauseWorkflowMessage,
    StopWorkflowMessage,
    ExportWorkflowMessage
)

from .metrics_dashboard import (
    MetricsDashboardWidget,
    MetricType,
    AlertLevel,
    Metric,
    SystemHealth,
    ProductivityMetrics,
    ExportMetricsMessage,
    ShowMetricsSettingsMessage
)

from .modal_dialogs import (
    ConfigurationModal,
    CommandTemplatesModal,
    ConfirmationModal,
    TaskCreationModal,
    ConfigOption,
    SaveConfigMessage,
    ExportConfigMessage,
    ImportConfigMessage,
    UseTemplateMessage,
    SaveCustomTemplateMessage,
    ExportTemplatesMessage,
    ConfirmationMessage,
    CreateTaskMessage
)

__all__ = [
    # Console Widget
    'ConsoleWidget',
    'MessageType',
    'AITaskStatus', 
    'ConsoleMessage',
    'AITask',
    'ShowCommandTemplatesMessage',
    
    # Notification System
    'NotificationSystem',
    'Notification',
    'NotificationLevel',
    'NotificationCategory',
    'ExportHistoryMessage',
    
    # Placeholder Alert
    'PlaceholderAlert',
    'PlaceholderIssue',
    'PlaceholderType',
    'PlaceholderSeverity',
    'PlaceholderAlertTriggeredMessage',
    'AutoFixIssuesMessage',
    'StartCompletionMessage',
    'ExportPlaceholderReportMessage',
    
    # Progress Intelligence
    'ProgressIntelligence',
    'ProgressReport',
    'ValidationStatus',
    'ValidateNowMessage',
    'ShowValidationDetailsMessage',
    
    # Project Tree
    'ProjectTree',
    'FileSelectedMessage',
    'TreeRefreshedMessage',
    
    # Task Dashboard
    'TaskDashboard',
    'ProjectTask',
    'TaskStatus',
    'TaskPriority',
    'AddTaskMessage',
    'ShowAnalyticsMessage',
    
    # Workflow Visualizer
    'WorkflowVisualizerWidget',
    'WorkflowStatus',
    'TaskNodeStatus',
    'TaskNode', 
    'WorkflowDefinition',
    'StartWorkflowMessage',
    'PauseWorkflowMessage',
    'StopWorkflowMessage',
    'ExportWorkflowMessage',
    
    # Metrics Dashboard
    'MetricsDashboardWidget',
    'MetricType',
    'AlertLevel',
    'Metric',
    'SystemHealth',
    'ProductivityMetrics',
    'ExportMetricsMessage',
    'ShowMetricsSettingsMessage',
    
    # Modal Dialogs
    'ConfigurationModal',
    'CommandTemplatesModal',
    'ConfirmationModal',
    'TaskCreationModal',
    'ConfigOption',
    'SaveConfigMessage',
    'ExportConfigMessage',
    'ImportConfigMessage',
    'UseTemplateMessage',
    'SaveCustomTemplateMessage',
    'ExportTemplatesMessage',
    'ConfirmationMessage',
    'CreateTaskMessage',
]