"""
Historical Trend Analysis and Reporting System.

Comprehensive reporting system for generating historical performance reports,
trend analysis, and business intelligence insights.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
from scipy import stats
import pandas as pd

from ..core.types import Severity, Priority
from .models import (
    PerformanceMetrics, BottleneckAnalysis, OptimizationRecommendation,
    TrendAnalysis, PerformanceAlert, AnalyticsConfiguration
)


class ReportType(str, Enum):
    """Types of performance reports."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Report output formats."""
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"
    EXCEL = "xlsx"


class TrendDirection(str, Enum):
    """Trend direction classifications."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class ReportMetrics:
    """Aggregated metrics for reporting."""
    # Performance metrics
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_memory_percent: float = 0.0
    max_memory_percent: float = 0.0
    avg_throughput: float = 0.0
    max_throughput: float = 0.0
    
    # Quality metrics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    avg_error_rate: float = 0.0
    max_error_rate: float = 0.0
    uptime_percentage: float = 100.0
    
    # AI-specific metrics
    avg_model_accuracy: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_context_usage: float = 0.0
    
    # Workflow metrics
    avg_task_success_rate: float = 0.0
    avg_workflow_completion_rate: float = 0.0
    total_tasks_completed: int = 0
    
    # Alert statistics
    total_alerts: int = 0
    critical_alerts: int = 0
    avg_resolution_time: float = 0.0
    
    # Optimization statistics
    optimizations_implemented: int = 0
    avg_improvement_achieved: float = 0.0
    
    # Data quality
    data_points: int = 0
    data_completeness: float = 0.0
    measurement_period: timedelta = field(default_factory=lambda: timedelta(0))


@dataclass
class TrendReport:
    """Trend analysis report for a specific metric."""
    metric_name: str
    time_period: timedelta
    
    # Trend characteristics
    trend_direction: TrendDirection
    trend_strength: float  # 0-1 scale
    trend_coefficient: float  # Slope of trend line
    correlation_coefficient: float  # R-squared value
    
    # Statistical analysis
    mean_value: float = 0.0
    median_value: float = 0.0
    std_deviation: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    
    # Change detection
    significant_changes: List[Dict[str, Any]] = field(default_factory=list)
    change_points: List[datetime] = field(default_factory=list)
    anomalies_detected: int = 0
    
    # Seasonality analysis
    seasonality_detected: bool = False
    seasonal_patterns: List[str] = field(default_factory=list)
    
    # Forecasting
    forecast_accuracy: float = 0.0
    forecast_confidence: float = 0.0
    
    # Recommendations
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    report_type: ReportType
    report_format: ReportFormat
    
    # Time period
    start_time: datetime
    end_time: datetime
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Executive summary
    executive_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregated metrics
    metrics: ReportMetrics = field(default_factory=ReportMetrics)
    
    # Trend analysis
    trend_reports: List[TrendReport] = field(default_factory=list)
    
    # Key findings
    top_bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    critical_issues: List[Dict[str, Any]] = field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance comparisons
    period_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recommendations and action items
    recommendations: List[str] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Appendices
    detailed_charts: List[Dict[str, Any]] = field(default_factory=list)
    raw_data_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Report metadata
    generated_by: str = "Performance Analytics Engine"
    report_version: str = "1.0"
    data_quality_score: float = 0.0


class PerformanceReporter:
    """
    Advanced performance reporting and trend analysis system.
    
    Features:
    - Automated report generation (daily, weekly, monthly, quarterly)
    - Historical trend analysis with statistical significance testing
    - Performance baseline comparison
    - Anomaly detection and highlighting
    - Interactive visualizations and charts
    - Multi-format output (PDF, HTML, JSON, Excel)
    - Executive summary generation
    - Actionable insights and recommendations
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        output_directory: Optional[Path] = None
    ):
        """Initialize the performance reporter."""
        self.config = config or AnalyticsConfiguration()
        self.output_directory = output_directory or Path("reports")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Report templates and configurations
        self.report_templates: Dict[ReportType, Dict[str, Any]] = {}
        self.baseline_data: Dict[str, Dict[str, float]] = {}
        
        # Cached analysis results
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Report generation queue
        self.report_queue: asyncio.Queue = asyncio.Queue()
        self.generation_task: Optional[asyncio.Task] = None
        
        # Statistics tracking
        self.generation_stats = {
            'reports_generated': 0,
            'last_generation_time': None,
            'avg_generation_time': 0.0,
            'failed_generations': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the reporting system."""
        try:
            self.logger.info("Initializing Performance Reporter...")
            
            # Create output directory
            self.output_directory.mkdir(parents=True, exist_ok=True)
            for format_dir in ["json", "pdf", "html", "csv", "xlsx"]:
                (self.output_directory / format_dir).mkdir(exist_ok=True)
            
            # Load report templates
            await self._load_report_templates()
            
            # Load baseline data
            await self._load_baseline_data()
            
            # Start report generation processor
            self.generation_task = asyncio.create_task(self._report_generation_loop())
            
            self.logger.info("Performance Reporter initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reporter: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the reporting system."""
        if self.generation_task:
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass
    
    async def generate_report(
        self,
        report_type: ReportType,
        start_time: datetime,
        end_time: datetime,
        metrics_data: List[PerformanceMetrics],
        bottlenecks: List[BottleneckAnalysis] = None,
        alerts: List[PerformanceAlert] = None,
        optimizations: List[OptimizationRecommendation] = None,
        output_format: ReportFormat = ReportFormat.JSON
    ) -> PerformanceReport:
        """Generate a comprehensive performance report."""
        try:
            self.logger.info(f"Generating {report_type.value} report from {start_time} to {end_time}")
            
            # Create report ID
            report_id = f"{report_type.value}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"
            
            # Calculate aggregated metrics
            aggregated_metrics = await self._calculate_aggregated_metrics(
                metrics_data, alerts or []
            )
            
            # Perform trend analysis
            trend_reports = await self._analyze_trends(metrics_data, start_time, end_time)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                aggregated_metrics, trend_reports, bottlenecks or [], alerts or []
            )
            
            # Identify key findings
            top_bottlenecks = await self._identify_top_bottlenecks(bottlenecks or [])
            critical_issues = await self._identify_critical_issues(alerts or [])
            optimization_ops = await self._identify_optimization_opportunities(optimizations or [])
            
            # Performance comparisons
            period_comparisons = await self._calculate_period_comparisons(
                aggregated_metrics, report_type, start_time
            )
            baseline_comparisons = await self._calculate_baseline_comparisons(aggregated_metrics)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                trend_reports, top_bottlenecks, critical_issues
            )
            
            # Create action items
            action_items = await self._create_action_items(
                critical_issues, optimization_ops, recommendations
            )
            
            # Generate visualizations
            detailed_charts = await self._generate_charts(metrics_data, trend_reports)
            
            # Create comprehensive report
            report = PerformanceReport(
                report_id=report_id,
                report_type=report_type,
                report_format=output_format,
                start_time=start_time,
                end_time=end_time,
                executive_summary=executive_summary,
                metrics=aggregated_metrics,
                trend_reports=trend_reports,
                top_bottlenecks=top_bottlenecks,
                critical_issues=critical_issues,
                optimization_opportunities=optimization_ops,
                period_comparisons=period_comparisons,
                baseline_comparisons=baseline_comparisons,
                recommendations=recommendations,
                action_items=action_items,
                detailed_charts=detailed_charts,
                data_quality_score=self._calculate_data_quality_score(metrics_data)
            )
            
            # Save report
            await self._save_report(report, output_format)
            
            # Update statistics
            self.generation_stats['reports_generated'] += 1
            self.generation_stats['last_generation_time'] = datetime.utcnow()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            self.generation_stats['failed_generations'] += 1
            raise
    
    async def generate_trend_analysis(
        self,
        metric_name: str,
        metrics_data: List[PerformanceMetrics],
        time_period: timedelta
    ) -> TrendReport:
        """Generate detailed trend analysis for a specific metric."""
        try:
            # Extract metric values and timestamps
            values = []
            timestamps = []
            
            for metric in metrics_data:
                if hasattr(metric, metric_name):
                    values.append(getattr(metric, metric_name))
                    timestamps.append(metric.timestamp)
            
            if len(values) < 10:
                raise ValueError(f"Insufficient data points for trend analysis: {len(values)}")
            
            # Statistical analysis
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            min_val = min(values)
            max_val = max(values)
            
            # Trend analysis using linear regression
            time_indices = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, values)
            
            # Determine trend direction and strength
            trend_strength = abs(r_value)
            
            if abs(slope) < std_val * 0.1:
                trend_direction = TrendDirection.STABLE
            elif slope > 0:
                trend_direction = TrendDirection.IMPROVING if 'error' not in metric_name.lower() else TrendDirection.DEGRADING
            else:
                trend_direction = TrendDirection.DEGRADING if 'error' not in metric_name.lower() else TrendDirection.IMPROVING
            
            # Detect change points
            change_points = await self._detect_change_points(values, timestamps)
            
            # Anomaly detection
            anomalies = await self._detect_anomalies_in_series(values)
            
            # Seasonality analysis
            seasonality_detected, seasonal_patterns = await self._analyze_seasonality(
                values, timestamps
            )
            
            # Generate insights and recommendations
            insights = await self._generate_trend_insights(
                metric_name, trend_direction, trend_strength, values
            )
            recommendations = await self._generate_trend_recommendations(
                metric_name, trend_direction, anomalies, change_points
            )
            
            return TrendReport(
                metric_name=metric_name,
                time_period=time_period,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_coefficient=slope,
                correlation_coefficient=r_value ** 2,
                mean_value=mean_val,
                median_value=median_val,
                std_deviation=std_val,
                min_value=min_val,
                max_value=max_val,
                change_points=change_points,
                anomalies_detected=len(anomalies),
                seasonality_detected=seasonality_detected,
                seasonal_patterns=seasonal_patterns,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {metric_name}: {e}")
            return TrendReport(
                metric_name=metric_name,
                time_period=time_period,
                trend_direction=TrendDirection.UNKNOWN
            )
    
    async def schedule_automated_report(
        self,
        report_type: ReportType,
        schedule_time: datetime,
        recipients: List[str] = None,
        output_formats: List[ReportFormat] = None
    ) -> str:
        """Schedule an automated report generation."""
        schedule_id = f"scheduled_{report_type.value}_{schedule_time.timestamp()}"
        
        schedule_info = {
            'schedule_id': schedule_id,
            'report_type': report_type,
            'schedule_time': schedule_time,
            'recipients': recipients or [],
            'output_formats': output_formats or [ReportFormat.JSON],
            'status': 'scheduled'
        }
        
        # Add to report queue
        await self.report_queue.put(schedule_info)
        
        self.logger.info(f"Scheduled {report_type.value} report for {schedule_time}")
        return schedule_id
    
    # Private methods
    
    async def _calculate_aggregated_metrics(
        self,
        metrics_data: List[PerformanceMetrics],
        alerts: List[PerformanceAlert]
    ) -> ReportMetrics:
        """Calculate aggregated metrics for the reporting period."""
        if not metrics_data:
            return ReportMetrics()
        
        # Extract values for aggregation
        cpu_values = [m.cpu_percent for m in metrics_data if m.cpu_percent is not None]
        memory_values = [m.memory_percent for m in metrics_data if m.memory_percent is not None]
        throughput_values = [m.throughput for m in metrics_data if m.throughput is not None]
        response_times = [m.ai_response_time for m in metrics_data if m.ai_response_time is not None]
        error_rates = [m.error_rate for m in metrics_data if m.error_rate is not None]
        
        # Calculate alert statistics
        critical_alerts = [a for a in alerts if a.severity == Severity.CRITICAL]
        resolved_alerts = [a for a in alerts if a.resolved_at is not None]
        
        resolution_times = []
        for alert in resolved_alerts:
            if alert.resolved_at and alert.created_at:
                resolution_time = (alert.resolved_at - alert.created_at).total_seconds()
                resolution_times.append(resolution_time)
        
        return ReportMetrics(
            avg_cpu_percent=statistics.mean(cpu_values) if cpu_values else 0.0,
            max_cpu_percent=max(cpu_values) if cpu_values else 0.0,
            avg_memory_percent=statistics.mean(memory_values) if memory_values else 0.0,
            max_memory_percent=max(memory_values) if memory_values else 0.0,
            avg_throughput=statistics.mean(throughput_values) if throughput_values else 0.0,
            max_throughput=max(throughput_values) if throughput_values else 0.0,
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            p95_response_time=np.percentile(response_times, 95) if response_times else 0.0,
            avg_error_rate=statistics.mean(error_rates) if error_rates else 0.0,
            max_error_rate=max(error_rates) if error_rates else 0.0,
            total_alerts=len(alerts),
            critical_alerts=len(critical_alerts),
            avg_resolution_time=statistics.mean(resolution_times) if resolution_times else 0.0,
            data_points=len(metrics_data),
            data_completeness=self._calculate_data_completeness(metrics_data),
            measurement_period=metrics_data[-1].timestamp - metrics_data[0].timestamp if len(metrics_data) > 1 else timedelta(0)
        )
    
    async def _analyze_trends(
        self,
        metrics_data: List[PerformanceMetrics],
        start_time: datetime,
        end_time: datetime
    ) -> List[TrendReport]:
        """Analyze trends for key metrics."""
        key_metrics = [
            'cpu_percent', 'memory_percent', 'throughput',
            'latency_p95', 'error_rate', 'ai_response_time'
        ]
        
        trend_reports = []
        time_period = end_time - start_time
        
        for metric_name in key_metrics:
            try:
                trend_report = await self.generate_trend_analysis(
                    metric_name, metrics_data, time_period
                )
                trend_reports.append(trend_report)
            except Exception as e:
                self.logger.warning(f"Error analyzing trend for {metric_name}: {e}")
        
        return trend_reports
    
    async def _generate_executive_summary(
        self,
        metrics: ReportMetrics,
        trends: List[TrendReport],
        bottlenecks: List[BottleneckAnalysis],
        alerts: List[PerformanceAlert]
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        # Overall health assessment
        health_score = 100.0
        health_factors = []
        
        if metrics.avg_cpu_percent > 80:
            health_score -= 20
            health_factors.append("High CPU utilization")
        
        if metrics.avg_memory_percent > 85:
            health_score -= 20
            health_factors.append("High memory utilization")
        
        if metrics.avg_error_rate > 0.05:
            health_score -= 30
            health_factors.append("Elevated error rate")
        
        if metrics.critical_alerts > 0:
            health_score -= 25
            health_factors.append("Critical alerts present")
        
        health_score = max(0, health_score)
        
        # Performance status
        if health_score >= 90:
            performance_status = "Excellent"
        elif health_score >= 75:
            performance_status = "Good"
        elif health_score >= 60:
            performance_status = "Fair"
        elif health_score >= 40:
            performance_status = "Poor"
        else:
            performance_status = "Critical"
        
        # Key highlights
        highlights = []
        if metrics.data_points > 1000:
            highlights.append(f"Analyzed {metrics.data_points:,} data points")
        
        if metrics.uptime_percentage >= 99.9:
            highlights.append("Excellent system uptime")
        
        improving_trends = [t for t in trends if t.trend_direction == TrendDirection.IMPROVING]
        if improving_trends:
            highlights.append(f"{len(improving_trends)} metrics showing improvement")
        
        return {
            'health_score': health_score,
            'performance_status': performance_status,
            'health_factors': health_factors,
            'key_highlights': highlights,
            'total_alerts': metrics.total_alerts,
            'critical_issues': len([b for b in bottlenecks if b.severity == Severity.CRITICAL]),
            'data_quality': metrics.data_completeness * 100,
            'reporting_period_days': metrics.measurement_period.days
        }
    
    async def _identify_top_bottlenecks(
        self,
        bottlenecks: List[BottleneckAnalysis]
    ) -> List[Dict[str, Any]]:
        """Identify top performance bottlenecks."""
        # Sort by performance impact
        sorted_bottlenecks = sorted(
            bottlenecks,
            key=lambda b: b.performance_impact,
            reverse=True
        )
        
        return [
            {
                'type': bottleneck.bottleneck_type,
                'component': bottleneck.component,
                'impact': bottleneck.performance_impact,
                'severity': bottleneck.severity.value,
                'root_cause': bottleneck.root_cause,
                'recommendations': bottleneck.immediate_actions[:3]  # Top 3 actions
            }
            for bottleneck in sorted_bottlenecks[:5]  # Top 5 bottlenecks
        ]
    
    async def _identify_critical_issues(
        self,
        alerts: List[PerformanceAlert]
    ) -> List[Dict[str, Any]]:
        """Identify critical performance issues."""
        critical_alerts = [a for a in alerts if a.severity == Severity.CRITICAL]
        
        return [
            {
                'title': alert.title,
                'description': alert.description,
                'component': alert.affected_component,
                'created_at': alert.created_at.isoformat(),
                'is_resolved': alert.resolved_at is not None,
                'resolution_time': (
                    (alert.resolved_at - alert.created_at).total_seconds()
                    if alert.resolved_at else None
                )
            }
            for alert in critical_alerts
        ]
    
    async def _identify_optimization_opportunities(
        self,
        optimizations: List[OptimizationRecommendation]
    ) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        # Sort by estimated improvement
        sorted_opts = sorted(
            optimizations,
            key=lambda o: o.estimated_improvement,
            reverse=True
        )
        
        return [
            {
                'title': opt.title,
                'description': opt.description,
                'type': opt.optimization_type.value,
                'estimated_improvement': opt.estimated_improvement,
                'confidence': opt.confidence_score,
                'priority': opt.priority.value,
                'implementation_effort': opt.estimated_effort
            }
            for opt in sorted_opts[:10]  # Top 10 opportunities
        ]
    
    def _calculate_data_completeness(self, metrics_data: List[PerformanceMetrics]) -> float:
        """Calculate data completeness percentage."""
        if not metrics_data:
            return 0.0
        
        # Check key fields for completeness
        key_fields = ['cpu_percent', 'memory_percent', 'throughput', 'ai_response_time']
        total_checks = len(metrics_data) * len(key_fields)
        complete_checks = 0
        
        for metric in metrics_data:
            for field in key_fields:
                if hasattr(metric, field) and getattr(metric, field) is not None:
                    complete_checks += 1
        
        return complete_checks / total_checks if total_checks > 0 else 0.0
    
    def _calculate_data_quality_score(self, metrics_data: List[PerformanceMetrics]) -> float:
        """Calculate overall data quality score."""
        if not metrics_data:
            return 0.0
        
        # Factors for data quality
        completeness = self._calculate_data_completeness(metrics_data)
        
        # Consistency check (simplified)
        consistency = 1.0  # Assume good consistency for now
        
        # Timeliness check
        if len(metrics_data) > 1:
            time_gaps = [
                (metrics_data[i].timestamp - metrics_data[i-1].timestamp).total_seconds()
                for i in range(1, len(metrics_data))
            ]
            expected_interval = 30  # 30 seconds
            avg_gap = statistics.mean(time_gaps)
            timeliness = max(0.0, 1.0 - abs(avg_gap - expected_interval) / expected_interval)
        else:
            timeliness = 1.0
        
        # Combined score
        return (completeness * 0.5 + consistency * 0.3 + timeliness * 0.2) * 100
    
    async def _detect_change_points(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> List[datetime]:
        """Detect significant change points in the data."""
        change_points = []
        
        # Simple change point detection using moving averages
        window_size = min(10, len(values) // 4)
        
        if window_size < 3:
            return change_points
        
        for i in range(window_size, len(values) - window_size):
            before_avg = statistics.mean(values[i-window_size:i])
            after_avg = statistics.mean(values[i:i+window_size])
            
            # Check for significant change (more than 2 standard deviations)
            if len(values) > window_size * 2:
                overall_std = statistics.stdev(values)
                if abs(after_avg - before_avg) > 2 * overall_std:
                    change_points.append(timestamps[i])
        
        return change_points
    
    async def _detect_anomalies_in_series(self, values: List[float]) -> List[int]:
        """Detect anomalies in time series data."""
        if len(values) < 10:
            return []
        
        anomalies = []
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        threshold = 2.5 * std_val  # 2.5 standard deviations
        
        for i, value in enumerate(values):
            if abs(value - mean_val) > threshold:
                anomalies.append(i)
        
        return anomalies
    
    async def _analyze_seasonality(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Tuple[bool, List[str]]:
        """Analyze seasonality patterns in the data."""
        # Simplified seasonality detection
        # In a real implementation, this would use more sophisticated methods
        
        if len(values) < 24:  # Need at least 24 hours of hourly data
            return False, []
        
        # Check for daily patterns
        hourly_averages = {}
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            if hour not in hourly_averages:
                hourly_averages[hour] = []
            hourly_averages[hour].append(values[i])
        
        # Calculate variance between hourly averages
        if len(hourly_averages) >= 12:  # Need data for at least half the day
            hour_means = [statistics.mean(vals) for vals in hourly_averages.values()]
            hour_variance = statistics.variance(hour_means) if len(hour_means) > 1 else 0
            overall_variance = statistics.variance(values) if len(values) > 1 else 0
            
            # If hourly variance is significant compared to overall variance
            seasonality_detected = hour_variance > overall_variance * 0.1
            
            patterns = []
            if seasonality_detected:
                patterns.append("Daily pattern detected")
            
            return seasonality_detected, patterns
        
        return False, []
    
    async def _generate_trend_insights(
        self,
        metric_name: str,
        trend_direction: TrendDirection,
        trend_strength: float,
        values: List[float]
    ) -> List[str]:
        """Generate insights based on trend analysis."""
        insights = []
        
        if trend_direction == TrendDirection.IMPROVING:
            insights.append(f"{metric_name} is showing improvement over the analysis period")
            if trend_strength > 0.7:
                insights.append("The improvement trend is strong and consistent")
        
        elif trend_direction == TrendDirection.DEGRADING:
            insights.append(f"{metric_name} is showing degradation over the analysis period")
            if trend_strength > 0.7:
                insights.append("The degradation trend is strong and requires immediate attention")
        
        elif trend_direction == TrendDirection.STABLE:
            insights.append(f"{metric_name} has remained relatively stable")
        
        elif trend_direction == TrendDirection.VOLATILE:
            insights.append(f"{metric_name} shows high volatility and unpredictable behavior")
        
        # Value-based insights
        if values:
            max_val = max(values)
            min_val = min(values)
            range_ratio = (max_val - min_val) / max(min_val, 0.001)
            
            if range_ratio > 2.0:
                insights.append(f"High variability detected with {range_ratio:.1f}x range")
        
        return insights
    
    async def _generate_trend_recommendations(
        self,
        metric_name: str,
        trend_direction: TrendDirection,
        anomalies: List[int],
        change_points: List[datetime]
    ) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if trend_direction == TrendDirection.DEGRADING:
            if 'cpu' in metric_name.lower():
                recommendations.append("Investigate CPU-intensive processes and consider optimization")
            elif 'memory' in metric_name.lower():
                recommendations.append("Review memory usage patterns and identify potential leaks")
            elif 'error' in metric_name.lower():
                recommendations.append("Investigate root causes of increasing error rates")
            elif 'latency' in metric_name.lower():
                recommendations.append("Analyze performance bottlenecks causing increased latency")
        
        if len(anomalies) > len(change_points) * 0.1:  # More than 10% anomalies
            recommendations.append("High number of anomalies detected - implement monitoring alerts")
        
        if change_points:
            recommendations.append(f"Investigate {len(change_points)} significant change points in the data")
        
        return recommendations
    
    async def _calculate_period_comparisons(
        self,
        metrics: ReportMetrics,
        report_type: ReportType,
        start_time: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comparisons with previous periods."""
        # This would compare with previous period data
        # For now, return placeholder comparisons
        return {
            'previous_period': {
                'cpu_change': 0.0,
                'memory_change': 0.0,
                'throughput_change': 0.0,
                'error_rate_change': 0.0
            }
        }
    
    async def _calculate_baseline_comparisons(
        self,
        metrics: ReportMetrics
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comparisons with baseline performance."""
        # This would compare with stored baseline data
        return {
            'baseline': {
                'cpu_deviation': 0.0,
                'memory_deviation': 0.0,
                'throughput_deviation': 0.0,
                'error_rate_deviation': 0.0
            }
        }
    
    async def _generate_recommendations(
        self,
        trends: List[TrendReport],
        bottlenecks: List[Dict[str, Any]],
        issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []
        
        # Based on trends
        degrading_trends = [t for t in trends if t.trend_direction == TrendDirection.DEGRADING]
        if degrading_trends:
            recommendations.append(f"Address {len(degrading_trends)} metrics showing degradation")
        
        # Based on bottlenecks
        if bottlenecks:
            recommendations.append(f"Prioritize resolution of {len(bottlenecks)} performance bottlenecks")
        
        # Based on critical issues
        if issues:
            recommendations.append(f"Resolve {len(issues)} critical performance issues")
        
        return recommendations
    
    async def _create_action_items(
        self,
        issues: List[Dict[str, Any]],
        optimizations: List[Dict[str, Any]],
        recommendations: List[str]
    ) -> List[Dict[str, Any]]:
        """Create actionable items."""
        action_items = []
        
        # From critical issues
        for issue in issues[:3]:  # Top 3 issues
            action_items.append({
                'title': f"Resolve: {issue['title']}",
                'priority': 'High',
                'category': 'Issue Resolution',
                'description': issue['description'],
                'due_date': (datetime.utcnow() + timedelta(days=1)).isoformat()
            })
        
        # From optimizations
        for opt in optimizations[:2]:  # Top 2 optimizations
            action_items.append({
                'title': f"Implement: {opt['title']}",
                'priority': opt['priority'],
                'category': 'Optimization',
                'description': opt['description'],
                'estimated_effort': opt.get('implementation_effort', 'Unknown'),
                'due_date': (datetime.utcnow() + timedelta(days=7)).isoformat()
            })
        
        return action_items
    
    async def _generate_charts(
        self,
        metrics_data: List[PerformanceMetrics],
        trends: List[TrendReport]
    ) -> List[Dict[str, Any]]:
        """Generate chart configurations for visualizations."""
        charts = []
        
        # Time series chart for key metrics
        if metrics_data:
            charts.append({
                'type': 'line',
                'title': 'Performance Metrics Over Time',
                'data': {
                    'labels': [m.timestamp.isoformat() for m in metrics_data],
                    'datasets': [
                        {
                            'label': 'CPU Usage (%)',
                            'data': [m.cpu_percent for m in metrics_data],
                            'color': '#ff6384'
                        },
                        {
                            'label': 'Memory Usage (%)',
                            'data': [m.memory_percent for m in metrics_data],
                            'color': '#36a2eb'
                        }
                    ]
                }
            })
        
        return charts
    
    async def _save_report(self, report: PerformanceReport, format: ReportFormat) -> None:
        """Save report in specified format."""
        filename = f"{report.report_id}.{format.value}"
        filepath = self.output_directory / format.value / filename
        
        if format == ReportFormat.JSON:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, default=str, ensure_ascii=False)
        
        elif format == ReportFormat.CSV:
            # Convert metrics to CSV format
            metrics_df = pd.DataFrame([asdict(report.metrics)])
            metrics_df.to_csv(filepath, index=False)
        
        # Add other format implementations as needed
        
        self.logger.info(f"Report saved: {filepath}")
    
    async def _report_generation_loop(self) -> None:
        """Process scheduled report generation."""
        while True:
            try:
                # Get scheduled report from queue
                schedule_info = await self.report_queue.get()
                
                # Process scheduled report
                await self._process_scheduled_report(schedule_info)
                
                # Mark task as done
                self.report_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in report generation loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_scheduled_report(self, schedule_info: Dict[str, Any]) -> None:
        """Process a scheduled report generation."""
        # This would fetch data and generate the scheduled report
        self.logger.info(f"Processing scheduled report: {schedule_info['schedule_id']}")
    
    async def _load_report_templates(self) -> None:
        """Load report templates."""
        # Placeholder for template loading
        pass
    
    async def _load_baseline_data(self) -> None:
        """Load baseline performance data."""
        # Placeholder for baseline data loading
        pass