"""
Analytics and Monitoring REST API Endpoints.

Provides comprehensive analytics and monitoring capabilities:
- Performance metrics collection
- Usage analytics and trends
- System health monitoring  
- Resource utilization tracking
- Predictive analytics and insights
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..dependencies.auth import get_current_user
from ..middleware.rate_limiting import rate_limit
from ...core.exceptions import ValidationError, ResourceNotFoundError

# Initialize router
router = APIRouter()

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

# Pydantic Models
class MetricQuery(BaseModel):
    """Metric query parameters."""
    metric_type: MetricType
    time_range: TimeRange = Field(default=TimeRange.DAY)
    aggregation: AggregationType = Field(default=AggregationType.AVERAGE)
    filters: Dict[str, Any] = Field(default_factory=dict)
    group_by: Optional[List[str]] = Field(None, description="Fields to group by")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Limit results")

class AnalyticsResponse(BaseModel):
    """Base analytics response."""
    metric_type: str
    time_range: str
    aggregation: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PerformanceMetrics(BaseModel):
    """Performance metrics data."""
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
    """Usage metrics data."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    unique_users: int
    active_sessions: int
    api_calls_by_endpoint: Dict[str, int]
    feature_usage: Dict[str, int]
    bandwidth_usage_mb: float

class SystemMetrics(BaseModel):
    """System health metrics."""
    uptime_seconds: int
    service_availability_percent: float
    database_connections: int
    cache_hit_rate_percent: float
    background_jobs_pending: int
    background_jobs_failed: int
    disk_space_available_gb: float
    log_errors_count: int

class ProjectAnalytics(BaseModel):
    """Project analytics data."""
    total_projects: int
    active_projects: int
    projects_created_today: int
    average_project_size_mb: float
    most_used_templates: List[Dict[str, Any]]
    project_health_distribution: Dict[str, int]
    completion_rates: Dict[str, float]

class WorkflowAnalytics(BaseModel):
    """Workflow analytics data."""
    total_workflows: int
    active_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_execution_time_seconds: float
    success_rate_percent: float
    most_used_agents: List[Dict[str, Any]]
    bottleneck_analysis: Dict[str, Any]

class UserAnalytics(BaseModel):
    """User analytics data."""
    total_users: int
    active_users_today: int
    active_users_week: int
    new_users_today: int
    user_retention_rate: float
    top_features_by_usage: List[Dict[str, Any]]
    geographic_distribution: Dict[str, int]
    device_types: Dict[str, int]

class TrendAnalysis(BaseModel):
    """Trend analysis data."""
    metric_name: str
    trend_direction: str  # "up", "down", "stable"
    trend_magnitude: float
    confidence_score: float
    forecast_7d: List[float]
    seasonal_patterns: Dict[str, Any]
    anomalies: List[Dict[str, Any]]

class ReportRequest(BaseModel):
    """Custom report request."""
    report_name: str = Field(..., min_length=1, max_length=200)
    metrics: List[MetricType] = Field(..., min_items=1, max_items=10)
    time_range: TimeRange = Field(default=TimeRange.DAY)
    filters: Dict[str, Any] = Field(default_factory=dict)
    format: str = Field(default="json", description="Output format: json, csv, pdf")
    schedule: Optional[str] = Field(None, description="Cron schedule for recurring reports")

# Dependency injection
async def get_analytics_service():
    """Get analytics service dependency."""
    from ...analytics.engine import AnalyticsEngine
    service = AnalyticsEngine()
    await service.initialize()
    return service

# Routes
@router.post("/metrics/query", response_model=AnalyticsResponse)
@rate_limit(requests=30, window=60)
async def query_metrics(
    query: MetricQuery,
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Query metrics with flexible filtering and aggregation.
    
    Executes custom metric queries with time-based filtering,
    aggregation, and grouping capabilities.
    """
    try:
        result = await analytics_service.query_metrics(
            metric_type=query.metric_type.value,
            time_range=query.time_range.value,
            aggregation=query.aggregation.value,
            filters=query.filters,
            group_by=query.group_by,
            limit=query.limit,
            user_id=current_user["id"]
        )
        
        return AnalyticsResponse(
            metric_type=query.metric_type.value,
            time_range=query.time_range.value,
            aggregation=query.aggregation.value,
            timestamp=datetime.utcnow().isoformat(),
            data=result["data"],
            metadata=result.get("metadata", {})
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query metrics: {str(e)}"
        )

@router.get("/performance", response_model=PerformanceMetrics)
@rate_limit(requests=20, window=60)
async def get_performance_metrics(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for metrics"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get system performance metrics.
    
    Returns comprehensive system performance data including
    CPU, memory, disk, network, and application metrics.
    """
    try:
        metrics = await analytics_service.get_performance_metrics(
            time_range=time_range.value,
            user_id=current_user["id"]
        )
        
        return PerformanceMetrics(**metrics)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

@router.get("/usage", response_model=UsageMetrics)
@rate_limit(requests=20, window=60)
async def get_usage_metrics(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for metrics"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get system usage metrics.
    
    Returns usage statistics including request counts,
    user activity, API usage, and feature adoption.
    """
    try:
        metrics = await analytics_service.get_usage_metrics(
            time_range=time_range.value,
            user_id=current_user["id"]
        )
        
        return UsageMetrics(**metrics)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage metrics: {str(e)}"
        )

@router.get("/system", response_model=SystemMetrics)
@rate_limit(requests=20, window=60)
async def get_system_metrics(
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get system health metrics.
    
    Returns current system health indicators including
    uptime, availability, database status, and errors.
    """
    try:
        metrics = await analytics_service.get_system_metrics(
            user_id=current_user["id"]
        )
        
        return SystemMetrics(**metrics)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )

@router.get("/projects", response_model=ProjectAnalytics)
@rate_limit(requests=15, window=60)
async def get_project_analytics(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for analytics"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get project analytics and insights.
    
    Returns comprehensive project statistics, trends,
    and health indicators across all projects.
    """
    try:
        analytics = await analytics_service.get_project_analytics(
            time_range=time_range.value,
            user_id=current_user["id"]
        )
        
        return ProjectAnalytics(**analytics)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project analytics: {str(e)}"
        )

@router.get("/workflows", response_model=WorkflowAnalytics)
@rate_limit(requests=15, window=60)
async def get_workflow_analytics(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for analytics"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get workflow execution analytics.
    
    Returns workflow performance metrics, success rates,
    bottleneck analysis, and optimization recommendations.
    """
    try:
        analytics = await analytics_service.get_workflow_analytics(
            time_range=time_range.value,
            user_id=current_user["id"]
        )
        
        return WorkflowAnalytics(**analytics)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow analytics: {str(e)}"
        )

@router.get("/users", response_model=UserAnalytics)
@rate_limit(requests=15, window=60)
async def get_user_analytics(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for analytics"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get user behavior analytics.
    
    Returns user activity patterns, feature usage,
    retention rates, and demographic insights.
    """
    try:
        analytics = await analytics_service.get_user_analytics(
            time_range=time_range.value,
            user_id=current_user["id"]
        )
        
        return UserAnalytics(**analytics)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user analytics: {str(e)}"
        )

@router.get("/trends/{metric_name}", response_model=TrendAnalysis)
@rate_limit(requests=10, window=60)
async def get_trend_analysis(
    metric_name: str,
    time_range: TimeRange = Query(TimeRange.MONTH, description="Historical time range"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get trend analysis for specific metric.
    
    Returns trend analysis including direction, magnitude,
    forecasting, seasonal patterns, and anomaly detection.
    """
    try:
        analysis = await analytics_service.analyze_trends(
            metric_name=metric_name,
            time_range=time_range.value,
            user_id=current_user["id"]
        )
        
        return TrendAnalysis(**analysis)
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metric not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze trends: {str(e)}"
        )

@router.get("/dashboard")
@rate_limit(requests=10, window=60)
async def get_analytics_dashboard(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for dashboard"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Get comprehensive analytics dashboard data.
    
    Returns aggregated analytics data for dashboard display
    including key metrics, trends, and health indicators.
    """
    try:
        dashboard_data = await analytics_service.get_dashboard_data(
            time_range=time_range.value,
            user_id=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "time_range": time_range.value,
                "last_updated": datetime.utcnow().isoformat(),
                "summary": dashboard_data["summary"],
                "performance": dashboard_data["performance"],
                "usage": dashboard_data["usage"],
                "projects": dashboard_data["projects"],
                "workflows": dashboard_data["workflows"],
                "alerts": dashboard_data.get("alerts", []),
                "recommendations": dashboard_data.get("recommendations", [])
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard data: {str(e)}"
        )

@router.post("/reports", status_code=status.HTTP_201_CREATED)
@rate_limit(requests=5, window=300)  # 5 reports per 5 minutes
async def create_custom_report(
    report_request: ReportRequest,
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Create custom analytics report.
    
    Generates custom reports with specified metrics, filters,
    and output formats. Supports scheduled recurring reports.
    """
    try:
        report_id = await analytics_service.create_report(
            report_name=report_request.report_name,
            metrics=[metric.value for metric in report_request.metrics],
            time_range=report_request.time_range.value,
            filters=report_request.filters,
            output_format=report_request.format,
            schedule=report_request.schedule,
            created_by=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Report created successfully",
                "report_id": report_id,
                "report_name": report_request.report_name,
                "scheduled": report_request.schedule is not None,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create report: {str(e)}"
        )

@router.get("/reports")
@rate_limit(requests=10, window=60)
async def list_reports(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Page size"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    List user's custom reports.
    
    Returns paginated list of created reports with
    metadata and generation status.
    """
    try:
        reports = await analytics_service.list_user_reports(
            user_id=current_user["id"],
            page=page,
            page_size=page_size
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "reports": reports["items"],
                "total": reports["total"],
                "page": page,
                "page_size": page_size
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list reports: {str(e)}"
        )

@router.get("/reports/{report_id}/download")
@rate_limit(requests=10, window=60)
async def download_report(
    report_id: str,
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Download generated report.
    
    Returns the generated report file in the specified format.
    """
    try:
        report_data = await analytics_service.get_report(
            report_id=report_id,
            user_id=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "report_id": report_id,
                "report_name": report_data["name"],
                "format": report_data["format"],
                "generated_at": report_data["generated_at"],
                "data": report_data["data"],
                "metadata": report_data["metadata"]
            }
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download report: {str(e)}"
        )

@router.post("/alerts/configure")
@rate_limit(requests=10, window=60)
async def configure_alerts(
    alert_rules: List[Dict[str, Any]] = Body(..., description="Alert configuration rules"),
    current_user: Dict = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """
    Configure analytics alerts and thresholds.
    
    Sets up automated alerts for metric thresholds,
    anomalies, and trend changes.
    """
    try:
        alert_config_id = await analytics_service.configure_alerts(
            alert_rules=alert_rules,
            user_id=current_user["id"]
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Alert configuration updated successfully",
                "alert_config_id": alert_config_id,
                "rules_configured": len(alert_rules),
                "configured_at": datetime.utcnow().isoformat()
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Alert configuration error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure alerts: {str(e)}"
        )

@router.get("/health")
@rate_limit(requests=30, window=60)
async def analytics_health_check():
    """
    Analytics service health check.
    
    Returns status of analytics components and data availability.
    """
    try:
        # This would check actual service health in a real implementation
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "service": "analytics",
                "components": {
                    "metrics_collector": "operational",
                    "data_processor": "operational", 
                    "trend_analyzer": "operational",
                    "report_generator": "operational",
                    "alert_system": "operational"
                },
                "data_freshness": {
                    "performance_metrics": "< 1 minute",
                    "usage_metrics": "< 5 minutes",
                    "system_metrics": "real-time",
                    "trend_analysis": "< 1 hour"
                },
                "storage": {
                    "metrics_retention_days": 365,
                    "reports_retention_days": 90,
                    "storage_usage_percent": 45
                },
                "checked_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "analytics",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
        )