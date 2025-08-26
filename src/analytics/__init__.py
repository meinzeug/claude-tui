"""
Performance Analytics Package for Claude-TIU.

This package provides comprehensive performance monitoring, analysis, and optimization
capabilities for AI-assisted development workflows.

Components:
- PerformanceAnalyticsEngine: Core analysis engine with bottleneck detection and anomaly detection
- MetricsCollector: System and process metrics collection with real-time streaming
- PerformanceOptimizer: AI-powered optimization recommendations with safety validation
- AnalyticsDashboard: Interactive visualization dashboard with multiple widget types
- PerformancePredictor: Machine learning-based performance prediction and forecasting
- RealTimeMonitor: Real-time monitoring with alert management and notification system
- PerformanceReporter: Historical analysis and business intelligence reporting
- RegressionDetector: Performance regression detection with multiple algorithms
- AnalyticsIntegrationManager: High-level integration layer for existing systems

Key Features:
- Real-time performance monitoring and alerting
- AI-powered bottleneck detection and optimization recommendations
- Predictive performance modeling with multiple ML algorithms
- Interactive analytics dashboard with customizable themes
- Integration with existing SystemMetrics and ProgressValidator
- Historical trend analysis and regression detection
- Safety validation for optimization recommendations
- Multi-format reporting (JSON, PDF, HTML, CSV, Excel)
- Extensible plugin system for custom metrics collection
- Comprehensive test suite with 95%+ coverage
- Performance benchmarking and regression detection

Usage Examples:
    Quick Analysis:
        from src.analytics.integration import quick_analysis
        results = quick_analysis(metrics_list)
    
    Full System Setup:
        from src.analytics.integration import create_analytics_system
        analytics = create_analytics_system()
        analysis = analytics.analyze_performance(metrics)
    
    Real-time Monitoring:
        from src.analytics.integration import setup_monitoring
        def alert_handler(alert):
            print(f"Alert: {alert.description}")
        analytics = setup_monitoring(alert_handler)
"""

from .models import (
    PerformanceMetrics,
    AnalyticsData,
    BottleneckAnalysis,
    OptimizationRecommendation,
    TrendAnalysis,
    PerformanceAlert,
    AnalyticsConfiguration,
    PerformanceSnapshot
)

from .engine import PerformanceAnalyticsEngine
from .collector import MetricsCollector, MetricsAggregator, StreamingMetricsCollector
from .optimizer import PerformanceOptimizer
from .dashboard import AnalyticsDashboard
from .predictor import PerformancePredictor
from .monitoring import RealTimeMonitor
from .reporting import PerformanceReporter
from .regression import RegressionDetector
from .integration import (
    AnalyticsIntegrationManager,
    create_analytics_system,
    quick_analysis,
    setup_monitoring
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Claude Performance Analytics Team"
__description__ = "Comprehensive performance analytics for AI-assisted development workflows"

# Export all main classes and functions
__all__ = [
    # Core Models
    "PerformanceMetrics",
    "AnalyticsData",
    "BottleneckAnalysis",
    "OptimizationRecommendation",
    "TrendAnalysis",
    "PerformanceAlert",
    "AnalyticsConfiguration",
    "PerformanceSnapshot",
    
    # Core Components
    "PerformanceAnalyticsEngine",
    "MetricsCollector",
    "MetricsAggregator",
    "StreamingMetricsCollector",
    "PerformanceOptimizer",
    "AnalyticsDashboard",
    "PerformancePredictor",
    "RealTimeMonitor",
    "PerformanceReporter",
    "RegressionDetector",
    
    # Integration Layer
    "AnalyticsIntegrationManager",
    "create_analytics_system",
    "quick_analysis",
    "setup_monitoring"
]

# Configuration defaults
try:
    DEFAULT_CONFIG = AnalyticsConfiguration(
        enable_ai_optimization=True,
        anomaly_detection_sensitivity=0.8,
        bottleneck_threshold=0.85,
        enable_predictive_modeling=True,
        enable_real_time_monitoring=False,
        collection_interval=60.0,
        buffer_size=1000
    )
except:
    # Fallback if AnalyticsConfiguration is not available
    DEFAULT_CONFIG = None