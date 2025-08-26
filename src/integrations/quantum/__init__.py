"""
Quantum Universal Development Environment Intelligence
Cross-Platform Integration System for Seamless Development Experience

This package provides a comprehensive suite of tools for integrating Claude-TUI
intelligence across all major development environments, platforms, and tools.
"""

from .universal_environment_adapter import (
    UniversalEnvironmentAdapter,
    EnvironmentType,
    IntegrationStatus,
    EnvironmentCapability,
    QuantumIntelligenceAPI
)

from .ide_intelligence_bridge import (
    IDEIntelligenceBridge,
    IDEType,
    IDEFeature,
    VSCodeAdapter,
    IntelliJAdapter,
    VimAdapter
)

from .cicd_intelligence_orchestrator import (
    CICDIntelligenceOrchestrator,
    CICDPlatform,
    PipelineStatus,
    JobType,
    GitHubActionsAdapter,
    GitLabCIAdapter,
    JenkinsAdapter
)

from .cloud_platform_connector import (
    CloudPlatformConnector,
    CloudProvider,
    CloudService,
    DeploymentTarget,
    AWSAdapter,
    GCPAdapter,
    AzureAdapter
)

from .sync.real_time_synchronizer import (
    RealTimeSynchronizer,
    SyncEventType,
    ConflictResolution,
    SyncPriority,
    SyncEvent,
    create_file_change_event,
    create_cursor_move_event,
    create_build_event
)

from .plugins.plugin_manager import (
    PluginManager,
    PluginStatus,
    PluginType,
    PluginPriority,
    BasePlugin,
    PluginMetadata
)

from .config.unified_config_manager import (
    UnifiedConfigManager,
    ConfigFormat,
    ConfigScope,
    ConfigPriority,
    ConfigurableComponent
)

from .performance_monitor import (
    PerformanceMonitor,
    PerformanceLevel,
    MetricType,
    AlertLevel,
    PerformanceTimer,
    monitor_performance
)

from .error_handler import (
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    ErrorRecord,
    handle_errors,
    with_retry,
    with_circuit_breaker
)

__version__ = "1.0.0"
__author__ = "Claude-TUI Development Team"
__description__ = "Universal Development Environment Intelligence"

__all__ = [
    # Universal Environment Adapter
    "UniversalEnvironmentAdapter",
    "EnvironmentType", 
    "IntegrationStatus",
    "EnvironmentCapability",
    "QuantumIntelligenceAPI",
    
    # IDE Intelligence Bridge
    "IDEIntelligenceBridge",
    "IDEType",
    "IDEFeature",
    "VSCodeAdapter",
    "IntelliJAdapter", 
    "VimAdapter",
    
    # CI/CD Intelligence Orchestrator
    "CICDIntelligenceOrchestrator",
    "CICDPlatform",
    "PipelineStatus",
    "JobType",
    "GitHubActionsAdapter",
    "GitLabCIAdapter",
    "JenkinsAdapter",
    
    # Cloud Platform Connector
    "CloudPlatformConnector",
    "CloudProvider",
    "CloudService",
    "DeploymentTarget",
    "AWSAdapter",
    "GCPAdapter",
    "AzureAdapter",
    
    # Real-Time Synchronizer
    "RealTimeSynchronizer",
    "SyncEventType",
    "ConflictResolution",
    "SyncPriority",
    "SyncEvent",
    "create_file_change_event",
    "create_cursor_move_event",
    "create_build_event",
    
    # Plugin Manager
    "PluginManager",
    "PluginStatus",
    "PluginType", 
    "PluginPriority",
    "BasePlugin",
    "PluginMetadata",
    
    # Configuration Manager
    "UnifiedConfigManager",
    "ConfigFormat",
    "ConfigScope",
    "ConfigPriority",
    "ConfigurableComponent",
    
    # Performance Monitor
    "PerformanceMonitor",
    "PerformanceLevel",
    "MetricType",
    "AlertLevel",
    "PerformanceTimer",
    "monitor_performance",
    
    # Error Handler
    "ErrorHandler",
    "ErrorSeverity",
    "ErrorCategory", 
    "RecoveryStrategy",
    "ErrorContext",
    "ErrorRecord",
    "handle_errors",
    "with_retry",
    "with_circuit_breaker"
]