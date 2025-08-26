"""
Analytics Data Models for Performance Monitoring.

Extends the existing SystemMetrics with advanced analytics capabilities
and defines comprehensive data structures for monitoring and optimization.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from ..core.types import SystemMetrics, ProgressMetrics, Priority, Severity


class MetricType(str, Enum):
    """Types of performance metrics."""
    SYSTEM = "system"
    APPLICATION = "application"
    AI_MODEL = "ai_model"
    WORKFLOW = "workflow"
    USER_INTERACTION = "user_interaction"
    NETWORK = "network"
    DATABASE = "database"


class AlertType(str, Enum):
    """Types of performance alerts."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ANOMALY_DETECTED = "anomaly_detected"
    REGRESSION_DETECTED = "regression_detected"
    BOTTLENECK_IDENTIFIED = "bottleneck_identified"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"


class AnalyticsStatus(str, Enum):
    """Analytics processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationType(str, Enum):
    """Types of optimization recommendations."""
    RESOURCE_SCALING = "resource_scaling"
    CODE_OPTIMIZATION = "code_optimization"
    CACHING = "caching"
    ALGORITHM_IMPROVEMENT = "algorithm_improvement"
    CONFIGURATION_TUNING = "configuration_tuning"
    INFRASTRUCTURE_UPGRADE = "infrastructure_upgrade"


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics extending SystemMetrics."""
    
    # Base system metrics (inherited from SystemMetrics)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: int = 0
    disk_percent: float = 0.0
    active_tasks: int = 0
    cache_hit_rate: float = 0.0
    ai_response_time: float = 0.0
    
    # Enhanced analytics metrics
    session_id: str = field(default_factory=lambda: str(uuid4()))
    metric_type: MetricType = MetricType.APPLICATION
    
    # Performance metrics
    throughput: float = 0.0  # Operations per second
    latency_p50: float = 0.0  # 50th percentile latency (ms)
    latency_p95: float = 0.0  # 95th percentile latency (ms)
    latency_p99: float = 0.0  # 99th percentile latency (ms)
    error_rate: float = 0.0  # Percentage of failed operations
    
    # AI-specific metrics
    tokens_per_second: float = 0.0
    model_accuracy: float = 0.0
    context_window_usage: float = 0.0  # Percentage of context window used
    hallucination_rate: float = 0.0
    
    # Workflow metrics
    workflow_completion_rate: float = 0.0
    task_success_rate: float = 0.0
    average_task_duration: float = 0.0
    concurrent_workflows: int = 0
    
    # Resource utilization
    network_io_bytes: int = 0
    disk_io_bytes: int = 0
    gpu_utilization: float = 0.0
    gpu_memory_usage: float = 0.0
    
    # Quality metrics
    code_quality_score: float = 0.0
    validation_pass_rate: float = 0.0
    placeholder_detection_rate: float = 0.0
    
    # Metadata
    environment: str = "development"
    version: str = "1.0.0"
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    
    def to_system_metrics(self) -> SystemMetrics:
        """Convert to base SystemMetrics for compatibility."""
        return SystemMetrics(
            timestamp=self.timestamp,
            cpu_percent=self.cpu_percent,
            memory_percent=self.memory_percent,
            memory_used=self.memory_used,
            disk_percent=self.disk_percent,
            active_tasks=self.active_tasks,
            cache_hit_rate=self.cache_hit_rate,
            ai_response_time=self.ai_response_time
        )
    
    @classmethod
    def from_system_metrics(
        cls, 
        system_metrics: SystemMetrics,
        **kwargs
    ) -> "PerformanceMetrics":
        """Create from base SystemMetrics with additional data."""
        return cls(
            timestamp=system_metrics.timestamp,
            cpu_percent=system_metrics.cpu_percent,
            memory_percent=system_metrics.memory_percent,
            memory_used=system_metrics.memory_used,
            disk_percent=system_metrics.disk_percent,
            active_tasks=system_metrics.active_tasks,
            cache_hit_rate=system_metrics.cache_hit_rate,
            ai_response_time=system_metrics.ai_response_time,
            **kwargs
        )
    
    def calculate_composite_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        # Weighted scoring of various metrics
        weights = {
            'cpu': 0.2,
            'memory': 0.2,
            'throughput': 0.15,
            'latency': 0.15,
            'error_rate': 0.1,
            'quality': 0.1,
            'ai_performance': 0.1
        }
        
        # Normalize metrics to 0-100 scale
        cpu_score = max(0, 100 - self.cpu_percent)
        memory_score = max(0, 100 - self.memory_percent)
        throughput_score = min(100, self.throughput * 10)  # Assume 10 ops/sec = 100%
        latency_score = max(0, 100 - (self.latency_p95 / 10))  # 1000ms = 0%
        error_score = max(0, 100 - (self.error_rate * 100))
        quality_score = self.code_quality_score
        ai_score = (
            (100 - self.hallucination_rate * 100) + 
            self.model_accuracy * 100 + 
            (100 - self.context_window_usage)
        ) / 3
        
        composite_score = (
            cpu_score * weights['cpu'] +
            memory_score * weights['memory'] +
            throughput_score * weights['throughput'] +
            latency_score * weights['latency'] +
            error_score * weights['error_rate'] +
            quality_score * weights['quality'] +
            ai_score * weights['ai_performance']
        )
        
        return round(min(100.0, max(0.0, composite_score)), 2)


class AnalyticsData(BaseModel):
    """Comprehensive analytics data structure."""
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    data_type: str  # metrics, trends, bottlenecks, etc.
    
    # Core data
    metrics: Optional[PerformanceMetrics] = None
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    processed_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Context information
    environment: str = "development"
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Processing metadata
    processing_status: AnalyticsStatus = AnalyticsStatus.PENDING
    processing_duration: float = 0.0
    error_message: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    # Bottleneck identification
    bottleneck_type: str = "unknown"  # cpu, memory, io, network, algorithm
    severity: Severity = Severity.MEDIUM
    component: str = ""  # Which component has the bottleneck
    
    # Impact analysis
    performance_impact: float = 0.0  # Percentage impact on overall performance
    affected_workflows: List[str] = field(default_factory=list)
    affected_users: int = 0
    
    # Root cause analysis
    root_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    
    # Evidence and metrics
    evidence: Dict[str, Any] = field(default_factory=dict)
    threshold_violations: List[str] = field(default_factory=list)
    trend_analysis: Optional[str] = None
    
    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    long_term_solutions: List[str] = field(default_factory=list)
    estimated_fix_effort: str = "unknown"  # hours, days, weeks
    
    # Resolution tracking
    status: str = "open"  # open, investigating, fixing, resolved
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def calculate_priority(self) -> Priority:
        """Calculate priority based on severity and impact."""
        if self.severity == Severity.CRITICAL or self.performance_impact > 50:
            return Priority.CRITICAL
        elif self.severity == Severity.HIGH or self.performance_impact > 30:
            return Priority.HIGH
        elif self.severity == Severity.MEDIUM or self.performance_impact > 15:
            return Priority.MEDIUM
        else:
            return Priority.LOW


class OptimizationRecommendation(BaseModel):
    """AI-generated optimization recommendation."""
    
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Recommendation details
    optimization_type: OptimizationType
    title: str
    description: str
    rationale: str
    
    # Impact estimation
    estimated_improvement: float = Field(ge=0.0, le=100.0)  # Percentage improvement
    confidence_score: float = Field(ge=0.0, le=100.0)
    risk_level: str = Field(default="low")  # low, medium, high
    
    # Implementation details
    implementation_steps: List[str] = Field(default_factory=list)
    required_resources: Dict[str, Any] = Field(default_factory=dict)
    estimated_effort: str = "unknown"  # hours, days, weeks
    prerequisites: List[str] = Field(default_factory=list)
    
    # Targeting
    target_components: List[str] = Field(default_factory=list)
    applicable_environments: List[str] = Field(default_factory=list)
    
    # Evidence and analysis
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)
    current_metrics: Optional[PerformanceMetrics] = None
    projected_metrics: Optional[PerformanceMetrics] = None
    
    # Tracking
    priority: Priority = Priority.MEDIUM
    status: str = "proposed"  # proposed, approved, implementing, completed, rejected
    assigned_to: Optional[str] = None
    implemented_at: Optional[datetime] = None
    
    # Validation
    a_b_test_results: Optional[Dict[str, Any]] = None
    actual_improvement: Optional[float] = None
    
    @validator('estimated_improvement', allow_reuse=True)
    def validate_improvement(cls, v: float) -> float:
        """Validate improvement percentage."""
        return max(0.0, min(100.0, v))
    
    @validator('confidence_score', allow_reuse=True)
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence score."""
        return max(0.0, min(100.0, v))
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


@dataclass
class TrendAnalysis:
    """Performance trend analysis over time."""
    
    start_time: datetime
    end_time: datetime
    id: str = field(default_factory=lambda: str(uuid4()))
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    data_points: int = 0
    
    # Trend characteristics
    trend_direction: str = "stable"  # improving, degrading, stable, volatile
    trend_strength: float = 0.0  # 0-1, how strong the trend is
    seasonality_detected: bool = False
    cyclical_patterns: List[str] = field(default_factory=list)
    
    # Statistical analysis
    mean_value: float = 0.0
    median_value: float = 0.0
    std_deviation: float = 0.0
    coefficient_of_variation: float = 0.0
    
    # Change detection
    significant_changes: List[Dict[str, Any]] = field(default_factory=list)
    anomalies_detected: int = 0
    change_points: List[datetime] = field(default_factory=list)
    
    # Forecasting
    predicted_values: List[Tuple[datetime, float]] = field(default_factory=list)
    prediction_confidence: float = 0.0
    forecast_horizon: timedelta = field(default_factory=lambda: timedelta(hours=24))
    
    # Alerts and recommendations
    trend_alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    metric_name: str = ""
    analysis_method: str = "statistical"  # statistical, ml, hybrid
    model_accuracy: float = 0.0


class PerformanceAlert(BaseModel):
    """Performance monitoring alert."""
    
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Alert classification
    alert_type: AlertType
    severity: Severity
    priority: Priority = Priority.MEDIUM
    
    # Alert details
    title: str
    description: str
    affected_component: str
    
    # Trigger information
    metric_name: str
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    violation_duration: timedelta = field(default_factory=lambda: timedelta(0))
    
    # Context
    environment: str = "development"
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Associated data
    performance_metrics: Optional[PerformanceMetrics] = None
    bottleneck_analysis: Optional[BottleneckAnalysis] = None
    trend_data: Optional[TrendAnalysis] = None
    
    # Response and resolution
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    # Escalation
    escalation_level: int = 0
    escalated_at: Optional[datetime] = None
    escalated_to: Optional[str] = None
    
    # Suppression (to prevent alert spam)
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    suppress_until: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return (
            self.resolved_at is None and 
            not self.suppressed and
            (self.suppress_until is None or datetime.utcnow() > self.suppress_until)
        )
    
    def age(self) -> timedelta:
        """Get age of the alert."""
        return datetime.utcnow() - self.created_at
    
    def time_to_resolve(self) -> Optional[timedelta]:
        """Get time taken to resolve alert."""
        if self.resolved_at:
            return self.resolved_at - self.created_at
        return None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds()
        }


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str = ""
    
    # Core metrics
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    composite_score: float = 0.0
    
    # Analysis results
    bottlenecks: List[BottleneckAnalysis] = field(default_factory=list)
    active_alerts: List[PerformanceAlert] = field(default_factory=list)
    optimization_opportunities: List[OptimizationRecommendation] = field(default_factory=list)
    
    # Context
    environment: str = "development"
    workload_description: str = ""
    concurrent_users: int = 0
    
    def get_critical_issues(self) -> List[Union[BottleneckAnalysis, PerformanceAlert]]:
        """Get all critical performance issues."""
        issues = []
        
        # Critical bottlenecks
        issues.extend([b for b in self.bottlenecks if b.severity == Severity.CRITICAL])
        
        # Critical alerts
        issues.extend([a for a in self.active_alerts if a.severity == Severity.CRITICAL])
        
        return issues


@dataclass
class AnalyticsConfiguration:
    """Configuration for analytics collection and processing."""
    
    # Collection settings
    collection_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    retention_period: timedelta = field(default_factory=lambda: timedelta(days=30))
    aggregation_intervals: List[timedelta] = field(default_factory=lambda: [
        timedelta(minutes=1),
        timedelta(minutes=5),
        timedelta(minutes=15),
        timedelta(hours=1),
        timedelta(hours=6),
        timedelta(days=1)
    ])
    
    # Alert thresholds
    cpu_threshold: float = 80.0  # Percentage
    memory_threshold: float = 85.0  # Percentage
    disk_threshold: float = 90.0  # Percentage
    error_rate_threshold: float = 5.0  # Percentage
    latency_threshold: float = 2000.0  # Milliseconds
    
    # Analysis settings
    trend_analysis_window: timedelta = field(default_factory=lambda: timedelta(hours=6))
    anomaly_sensitivity: float = 2.0  # Standard deviations
    minimum_data_points: int = 10
    
    # Optimization settings
    optimization_check_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    minimum_improvement_threshold: float = 5.0  # Percentage
    
    # Storage settings
    storage_backend: str = "filesystem"  # filesystem, database, cloud
    storage_path: Optional[Path] = None
    compression_enabled: bool = True
    
    # Privacy settings
    anonymize_user_data: bool = True
    data_retention_compliance: str = "gdpr"  # gdpr, ccpa, none