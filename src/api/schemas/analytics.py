"""
Analytics-related Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums
class MetricType(str, Enum):
    """Metric type enumeration."""
    PERFORMANCE = "performance"
    USAGE = "usage" 
    SYSTEM = "system"
    USER = "user"
    PROJECT = "project"
    WORKFLOW = "workflow"
    AGENT = "agent"


class TimeRange(str, Enum):
    """Time range enumeration."""
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"
    YEAR = "365d"


class AggregationType(str, Enum):
    """Aggregation type enumeration."""
    SUM = "sum"
    AVERAGE = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"


# Request Schemas
class MetricQuery(BaseModel):
    """Schema for metric query parameters."""
    metric_type: MetricType
    time_range: TimeRange = Field(default=TimeRange.DAY)
    aggregation: AggregationType = Field(default=AggregationType.AVERAGE)
    filters: Dict[str, Any] = Field(default_factory=dict)
    group_by: Optional[List[str]] = Field(None, description="Fields to group by")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Limit results")


class ReportRequest(BaseModel):
    """Schema for custom report requests."""
    report_name: str = Field(..., min_length=1, max_length=200)
    metrics: List[MetricType] = Field(..., min_items=1, max_items=10)
    time_range: TimeRange = Field(default=TimeRange.DAY)
    filters: Dict[str, Any] = Field(default_factory=dict)
    format: str = Field(default="json", description="Output format: json, csv, pdf")
    schedule: Optional[str] = Field(None, description="Cron schedule for recurring reports")


# Response Schemas
class AnalyticsResponse(BaseModel):
    """Base analytics response schema."""
    metric_type: str
    time_range: str
    aggregation: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """Performance metrics data schema."""
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_mb: float
    response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    active_connections: int
    queue_length: int


class UsageMetrics(BaseModel):
    """Usage metrics data schema."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    unique_users: int
    active_sessions: int
    api_calls_by_endpoint: Dict[str, int]
    feature_usage: Dict[str, int]
    bandwidth_usage_mb: float


class SystemMetrics(BaseModel):
    """System health metrics schema."""
    uptime_seconds: int
    service_availability_percent: float
    database_connections: int
    cache_hit_rate_percent: float
    background_jobs_pending: int
    background_jobs_failed: int
    disk_space_available_gb: float
    log_errors_count: int


class ProjectAnalytics(BaseModel):
    """Project analytics data schema."""
    total_projects: int
    active_projects: int
    projects_created_today: int
    average_project_size_mb: float
    most_used_templates: List[Dict[str, Any]]
    project_health_distribution: Dict[str, int]
    completion_rates: Dict[str, float]


class WorkflowAnalytics(BaseModel):
    """Workflow analytics data schema."""
    total_workflows: int
    active_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_execution_time_seconds: float
    success_rate_percent: float
    most_used_agents: List[Dict[str, Any]]
    bottleneck_analysis: Dict[str, Any]


class UserAnalytics(BaseModel):
    """User analytics data schema."""
    total_users: int
    active_users_today: int
    active_users_week: int
    new_users_today: int
    user_retention_rate: float
    top_features_by_usage: List[Dict[str, Any]]
    geographic_distribution: Dict[str, int]
    device_types: Dict[str, int]


class TrendAnalysis(BaseModel):
    """Trend analysis data schema."""
    metric_name: str
    trend_direction: str  # "up", "down", "stable"
    trend_magnitude: float
    confidence_score: float
    forecast_7d: List[float]
    seasonal_patterns: Dict[str, Any]
    anomalies: List[Dict[str, Any]]


class DashboardData(BaseModel):
    """Dashboard data schema."""
    time_range: str
    last_updated: str
    summary: Dict[str, Any]
    performance: PerformanceMetrics
    usage: UsageMetrics
    projects: ProjectAnalytics
    workflows: WorkflowAnalytics
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)


class ReportResponse(BaseModel):
    """Report response schema."""
    report_id: str
    report_name: str
    format: str
    generated_at: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]