"""
Performance Analytics Engine.

Core engine that orchestrates performance monitoring, analysis, and optimization.
Integrates with existing SystemMetrics and ProgressValidator systems.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from dataclasses import asdict

from ..core.types import SystemMetrics, ProgressMetrics, Severity, Priority
from ..core.progress_tracker import RealProgressTracker
from ..core.validator import ProgressValidator, CodeQualityAnalyzer
from .models import (
    PerformanceMetrics, AnalyticsData, BottleneckAnalysis,
    OptimizationRecommendation, TrendAnalysis, PerformanceAlert,
    PerformanceSnapshot, AnalyticsConfiguration, MetricType,
    AlertType, OptimizationType, AnalyticsStatus
)


class PerformanceAnalyticsEngine:
    """
    Core performance analytics engine.
    
    Features:
    - Real-time metrics collection and analysis
    - Bottleneck detection and root cause analysis
    - Predictive performance modeling
    - Automated optimization recommendations
    - Historical trend analysis and reporting
    - Performance regression detection
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        project_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the analytics engine."""
        self.config = config or AnalyticsConfiguration()
        self.project_path = Path(project_path) if project_path else None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.progress_tracker: Optional[RealProgressTracker] = None
        self.validator: Optional[ProgressValidator] = None
        self.quality_analyzer: Optional[CodeQualityAnalyzer] = None
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.analytics_data: Dict[str, AnalyticsData] = {}
        self.bottleneck_analyses: Dict[str, BottleneckAnalysis] = {}
        self.optimization_recommendations: Dict[str, OptimizationRecommendation] = {}
        self.trend_analyses: Dict[str, TrendAnalysis] = {}
        self.performance_alerts: Dict[str, PerformanceAlert] = {}
        
        # Runtime state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.analysis_tasks: Set[asyncio.Task] = set()
        
        # Alert suppression
        self.alert_suppression: Dict[str, datetime] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Caching for performance
        self.analysis_cache: Dict[str, Tuple[datetime, Any]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Machine learning models (placeholder for future ML integration)
        self.anomaly_detector = None
        self.trend_predictor = None
        self.optimization_recommender = None
    
    async def initialize(self) -> None:
        """Initialize the analytics engine."""
        try:
            self.logger.info("Initializing Performance Analytics Engine...")
            
            # Initialize project-specific components
            if self.project_path:
                self.progress_tracker = RealProgressTracker(
                    project_path=self.project_path,
                    validation_interval=30  # 30 second intervals
                )
                
                self.validator = ProgressValidator(
                    enable_cross_validation=True,
                    enable_execution_testing=True,
                    enable_quality_analysis=True
                )
                
                self.quality_analyzer = CodeQualityAnalyzer()
            
            # Initialize storage
            await self._initialize_storage()
            
            # Load historical data
            await self._load_historical_data()
            
            # Initialize ML models (placeholder)
            await self._initialize_ml_models()
            
            self.logger.info("Performance Analytics Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics engine: {e}")
            raise
    
    async def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already active")
            return
        
        self.logger.info("Starting performance monitoring...")
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start progress tracking if available
        if self.progress_tracker:
            await self.progress_tracker.start_monitoring()
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping performance monitoring...")
        self.is_monitoring = False
        
        # Stop monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel analysis tasks
        for task in self.analysis_tasks.copy():
            task.cancel()
        
        if self.analysis_tasks:
            await asyncio.gather(*self.analysis_tasks, return_exceptions=True)
        
        # Stop progress tracking
        if self.progress_tracker:
            await self.progress_tracker.stop_monitoring()
        
        # Save data before shutdown
        await self._save_data()
    
    async def collect_metrics(
        self,
        system_metrics: Optional[SystemMetrics] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        try:
            # Get system metrics
            if not system_metrics:
                system_metrics = await self._collect_system_metrics()
            
            # Enhance with analytics data
            enhanced_metrics = PerformanceMetrics.from_system_metrics(
                system_metrics,
                session_id=additional_data.get('session_id', 'default') if additional_data else 'default',
                metric_type=MetricType.APPLICATION
            )
            
            # Collect additional metrics
            if additional_data:
                for key, value in additional_data.items():
                    if hasattr(enhanced_metrics, key):
                        setattr(enhanced_metrics, key, value)
            
            # Collect AI-specific metrics
            await self._collect_ai_metrics(enhanced_metrics)
            
            # Collect workflow metrics
            await self._collect_workflow_metrics(enhanced_metrics)
            
            # Collect quality metrics
            await self._collect_quality_metrics(enhanced_metrics)
            
            # Store metrics
            self.metrics_history.append(enhanced_metrics)
            
            # Calculate composite score
            enhanced_metrics.composite_score = enhanced_metrics.calculate_composite_score()
            
            return enhanced_metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            # Return basic metrics on error
            return PerformanceMetrics()
    
    async def analyze_bottlenecks(
        self,
        metrics: Optional[PerformanceMetrics] = None,
        historical_window: timedelta = None
    ) -> List[BottleneckAnalysis]:
        """Analyze performance bottlenecks."""
        try:
            if not metrics:
                metrics = await self.collect_metrics()
            
            window = historical_window or timedelta(minutes=10)
            cache_key = f"bottlenecks_{metrics.timestamp.isoformat()}_{window.total_seconds()}"
            
            # Check cache
            cached_result = self._get_cached_analysis(cache_key)
            if cached_result:
                return cached_result
            
            bottlenecks = []
            
            # CPU bottleneck analysis
            cpu_bottleneck = await self._analyze_cpu_bottleneck(metrics, window)
            if cpu_bottleneck:
                bottlenecks.append(cpu_bottleneck)
            
            # Memory bottleneck analysis
            memory_bottleneck = await self._analyze_memory_bottleneck(metrics, window)
            if memory_bottleneck:
                bottlenecks.append(memory_bottleneck)
            
            # I/O bottleneck analysis
            io_bottleneck = await self._analyze_io_bottleneck(metrics, window)
            if io_bottleneck:
                bottlenecks.append(io_bottleneck)
            
            # Latency bottleneck analysis
            latency_bottleneck = await self._analyze_latency_bottleneck(metrics, window)
            if latency_bottleneck:
                bottlenecks.append(latency_bottleneck)
            
            # AI-specific bottleneck analysis
            ai_bottleneck = await self._analyze_ai_bottleneck(metrics, window)
            if ai_bottleneck:
                bottlenecks.append(ai_bottleneck)
            
            # Workflow bottleneck analysis
            workflow_bottleneck = await self._analyze_workflow_bottleneck(metrics, window)
            if workflow_bottleneck:
                bottlenecks.append(workflow_bottleneck)
            
            # Cache results
            self._cache_analysis(cache_key, bottlenecks)
            
            # Store bottleneck analyses
            for bottleneck in bottlenecks:
                self.bottleneck_analyses[bottleneck.id] = bottleneck
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error analyzing bottlenecks: {e}")
            return []
    
    async def generate_optimization_recommendations(
        self,
        bottlenecks: Optional[List[BottleneckAnalysis]] = None,
        metrics: Optional[PerformanceMetrics] = None
    ) -> List[OptimizationRecommendation]:
        """Generate AI-powered optimization recommendations."""
        try:
            if not bottlenecks:
                bottlenecks = await self.analyze_bottlenecks(metrics)
            
            if not metrics:
                metrics = await self.collect_metrics()
            
            recommendations = []
            
            # Generate recommendations for each bottleneck
            for bottleneck in bottlenecks:
                bottleneck_recommendations = await self._generate_bottleneck_recommendations(
                    bottleneck, metrics
                )
                recommendations.extend(bottleneck_recommendations)
            
            # Generate general optimization recommendations
            general_recommendations = await self._generate_general_recommendations(metrics)
            recommendations.extend(general_recommendations)
            
            # Sort by priority and impact
            recommendations.sort(
                key=lambda r: (r.priority.value, -r.estimated_improvement, -r.confidence_score)
            )
            
            # Store recommendations
            for rec in recommendations:
                self.optimization_recommendations[str(rec.id)] = rec
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return []
    
    async def analyze_trends(
        self,
        metric_name: str,
        window: timedelta = None,
        prediction_horizon: timedelta = None
    ) -> TrendAnalysis:
        """Analyze performance trends and predict future values."""
        try:
            window = window or self.config.trend_analysis_window
            prediction_horizon = prediction_horizon or timedelta(hours=24)
            
            # Get historical data
            end_time = datetime.utcnow()
            start_time = end_time - window
            
            historical_metrics = [
                m for m in self.metrics_history
                if start_time <= m.timestamp <= end_time
            ]
            
            if len(historical_metrics) < self.config.minimum_data_points:
                raise ValueError(f"Insufficient data points: {len(historical_metrics)}")
            
            # Extract metric values
            values = []
            timestamps = []
            
            for metrics in historical_metrics:
                if hasattr(metrics, metric_name):
                    values.append(getattr(metrics, metric_name))
                    timestamps.append(metrics.timestamp)
            
            if not values:
                raise ValueError(f"No data found for metric: {metric_name}")
            
            # Perform trend analysis
            trend_analysis = await self._perform_trend_analysis(
                metric_name, values, timestamps, prediction_horizon
            )
            
            # Store analysis
            self.trend_analyses[trend_analysis.id] = trend_analysis
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends for {metric_name}: {e}")
            # Return empty analysis
            return TrendAnalysis(
                start_time=datetime.utcnow() - window,
                end_time=datetime.utcnow(),
                metric_name=metric_name
            )
    
    async def detect_anomalies(
        self,
        metrics: Optional[PerformanceMetrics] = None
    ) -> List[PerformanceAlert]:
        """Detect performance anomalies and generate alerts."""
        try:
            if not metrics:
                metrics = await self.collect_metrics()
            
            alerts = []
            
            # Threshold-based anomaly detection
            threshold_alerts = await self._detect_threshold_anomalies(metrics)
            alerts.extend(threshold_alerts)
            
            # Statistical anomaly detection
            statistical_alerts = await self._detect_statistical_anomalies(metrics)
            alerts.extend(statistical_alerts)
            
            # Trend-based anomaly detection
            trend_alerts = await self._detect_trend_anomalies(metrics)
            alerts.extend(trend_alerts)
            
            # Filter out suppressed alerts
            active_alerts = []
            for alert in alerts:
                if not self._is_alert_suppressed(alert):
                    active_alerts.append(alert)
                    # Store alert
                    self.performance_alerts[str(alert.id)] = alert
                    # Trigger callbacks
                    await self._trigger_alert_callbacks(alert)
            
            return active_alerts
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def get_performance_snapshot(
        self,
        include_recommendations: bool = True
    ) -> PerformanceSnapshot:
        """Get comprehensive performance snapshot."""
        try:
            # Collect current metrics
            metrics = await self.collect_metrics()
            
            # Analyze bottlenecks
            bottlenecks = await self.analyze_bottlenecks(metrics)
            
            # Detect anomalies
            active_alerts = await self.detect_anomalies(metrics)
            
            # Get optimization recommendations
            recommendations = []
            if include_recommendations:
                recommendations = await self.generate_optimization_recommendations(
                    bottlenecks, metrics
                )
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                session_id=metrics.session_id,
                metrics=metrics,
                composite_score=metrics.calculate_composite_score(),
                bottlenecks=bottlenecks,
                active_alerts=active_alerts,
                optimization_opportunities=recommendations[:10],  # Top 10
                environment=metrics.environment
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error creating performance snapshot: {e}")
            # Return basic snapshot
            return PerformanceSnapshot()
    
    async def generate_analytics_report(
        self,
        window: timedelta = None,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        try:
            window = window or timedelta(hours=24)
            end_time = datetime.utcnow()
            start_time = end_time - window
            
            # Filter metrics within window
            window_metrics = [
                m for m in self.metrics_history
                if start_time <= m.timestamp <= end_time
            ]
            
            if not window_metrics:
                return {"error": "No data available for the specified time window"}
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(window_metrics)
            
            # Analyze trends for key metrics
            key_metrics = [
                'cpu_percent', 'memory_percent', 'throughput', 
                'latency_p95', 'error_rate', 'ai_response_time'
            ]
            
            trend_analyses = {}
            for metric in key_metrics:
                try:
                    trend = await self.analyze_trends(metric, window)
                    trend_analyses[metric] = asdict(trend)
                except Exception as e:
                    self.logger.warning(f"Could not analyze trend for {metric}: {e}")
            
            # Get current bottlenecks
            current_bottlenecks = [
                asdict(b) for b in self.bottleneck_analyses.values()
                if b.detected_at >= start_time
            ]
            
            # Get recent alerts
            recent_alerts = [
                alert.dict() for alert in self.performance_alerts.values()
                if alert.created_at >= start_time
            ]
            
            # Get active optimization recommendations
            active_recommendations = [
                rec.dict() for rec in self.optimization_recommendations.values()
                if rec.status in ['proposed', 'approved'] and rec.created_at >= start_time
            ]
            
            # Predictions
            predictions = {}
            if include_predictions:
                for metric in key_metrics:
                    try:
                        trend = await self.analyze_trends(
                            metric, window, timedelta(hours=6)
                        )
                        predictions[metric] = trend.predicted_values[:12]  # Next 6 hours
                    except Exception as e:
                        self.logger.warning(f"Could not predict {metric}: {e}")
            
            return {
                'report_generated_at': datetime.utcnow().isoformat(),
                'time_window': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': window.total_seconds() / 3600
                },
                'summary_statistics': summary_stats,
                'trend_analyses': trend_analyses,
                'current_bottlenecks': current_bottlenecks,
                'recent_alerts': recent_alerts,
                'optimization_opportunities': active_recommendations,
                'predictions': predictions if include_predictions else {},
                'data_quality': {
                    'total_data_points': len(window_metrics),
                    'data_completeness': len(window_metrics) / (window.total_seconds() / 30),  # Assuming 30s intervals
                    'analysis_coverage': len(trend_analyses) / len(key_metrics)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating analytics report: {e}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    # Private methods for implementation
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Trigger analysis tasks
                analysis_tasks = [
                    self._schedule_analysis_task('bottleneck', metrics),
                    self._schedule_analysis_task('anomaly', metrics),
                    self._schedule_analysis_task('optimization', metrics)
                ]
                
                # Don't wait for analysis tasks to complete
                for task in analysis_tasks:
                    self.analysis_tasks.add(task)
                    task.add_done_callback(self.analysis_tasks.discard)
                
                # Wait for next collection interval
                await asyncio.sleep(self.config.collection_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _schedule_analysis_task(
        self,
        analysis_type: str,
        metrics: PerformanceMetrics
    ) -> asyncio.Task:
        """Schedule an analysis task."""
        if analysis_type == 'bottleneck':
            return asyncio.create_task(self.analyze_bottlenecks(metrics))
        elif analysis_type == 'anomaly':
            return asyncio.create_task(self.detect_anomalies(metrics))
        elif analysis_type == 'optimization':
            return asyncio.create_task(self.generate_optimization_recommendations(None, metrics))
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect basic system metrics."""
        try:
            import psutil
            import time
            
            # Get system information
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used=memory.used,
                disk_percent=disk.percent,
                active_tasks=len(psutil.pids()),
                cache_hit_rate=0.0,  # Would need application-specific implementation
                ai_response_time=0.0  # Would need AI service integration
            )
            
        except ImportError:
            # Fallback if psutil not available
            return SystemMetrics()
        except Exception as e:
            self.logger.warning(f"Error collecting system metrics: {e}")
            return SystemMetrics()
    
    async def _collect_ai_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect AI-specific metrics."""
        if self.validator and self.project_path:
            try:
                # Get validation results
                validation_result = await self.validator.validate_codebase(self.project_path)
                
                metrics.model_accuracy = validation_result.authenticity_score / 100
                metrics.hallucination_rate = (100 - validation_result.authenticity_score) / 100
                metrics.validation_pass_rate = 1.0 if validation_result.is_authentic else 0.0
                
                # Count placeholder issues
                placeholder_issues = [
                    issue for issue in validation_result.issues 
                    if 'placeholder' in issue.description.lower()
                ]
                metrics.placeholder_detection_rate = len(placeholder_issues) / max(1, len(validation_result.issues))
                
            except Exception as e:
                self.logger.warning(f"Error collecting AI metrics: {e}")
    
    async def _collect_workflow_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect workflow-specific metrics."""
        if self.progress_tracker:
            try:
                current_progress = self.progress_tracker.get_current_progress()
                if current_progress:
                    metrics.workflow_completion_rate = current_progress.real_progress
                    metrics.task_success_rate = current_progress.authenticity_rate / 100
                    
                    # Get trend for average task duration
                    trend = self.progress_tracker.analyze_progress_trend()
                    if trend.velocity > 0:
                        metrics.average_task_duration = 3600 / trend.velocity  # Hours to seconds
                
            except Exception as e:
                self.logger.warning(f"Error collecting workflow metrics: {e}")
    
    async def _collect_quality_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect code quality metrics."""
        if self.quality_analyzer and self.project_path:
            try:
                quality_metrics = await self.quality_analyzer.analyze_codebase(self.project_path)
                metrics.code_quality_score = quality_metrics.overall_score
                
            except Exception as e:
                self.logger.warning(f"Error collecting quality metrics: {e}")
    
    async def _analyze_cpu_bottleneck(
        self,
        metrics: PerformanceMetrics,
        window: timedelta
    ) -> Optional[BottleneckAnalysis]:
        """Analyze CPU bottlenecks."""
        if metrics.cpu_percent > self.config.cpu_threshold:
            # Get historical CPU usage
            recent_metrics = [
                m for m in list(self.metrics_history)[-100:]  # Last 100 data points
                if (datetime.utcnow() - m.timestamp) <= window
            ]
            
            cpu_values = [m.cpu_percent for m in recent_metrics if m.cpu_percent > 0]
            avg_cpu = statistics.mean(cpu_values) if cpu_values else metrics.cpu_percent
            
            severity = Severity.CRITICAL if avg_cpu > 95 else Severity.HIGH if avg_cpu > 85 else Severity.MEDIUM
            
            return BottleneckAnalysis(
                bottleneck_type="cpu",
                severity=severity,
                component="system_cpu",
                performance_impact=min(100, (avg_cpu - 50) * 2),  # Impact calculation
                root_cause="High CPU utilization detected",
                contributing_factors=[
                    "Heavy computational workload",
                    "Inefficient algorithms",
                    "Concurrent processing"
                ],
                evidence={
                    "current_cpu_percent": metrics.cpu_percent,
                    "average_cpu_percent": avg_cpu,
                    "threshold_exceeded": self.config.cpu_threshold,
                    "measurement_window": window.total_seconds()
                },
                immediate_actions=[
                    "Identify CPU-intensive processes",
                    "Optimize computational algorithms",
                    "Consider horizontal scaling"
                ],
                long_term_solutions=[
                    "Implement caching strategies",
                    "Upgrade hardware resources",
                    "Optimize code for better CPU efficiency"
                ]
            )
        
        return None
    
    async def _analyze_memory_bottleneck(
        self,
        metrics: PerformanceMetrics,
        window: timedelta
    ) -> Optional[BottleneckAnalysis]:
        """Analyze memory bottlenecks."""
        if metrics.memory_percent > self.config.memory_threshold:
            recent_metrics = [
                m for m in list(self.metrics_history)[-100:]
                if (datetime.utcnow() - m.timestamp) <= window
            ]
            
            memory_values = [m.memory_percent for m in recent_metrics if m.memory_percent > 0]
            avg_memory = statistics.mean(memory_values) if memory_values else metrics.memory_percent
            
            severity = Severity.CRITICAL if avg_memory > 95 else Severity.HIGH if avg_memory > 90 else Severity.MEDIUM
            
            return BottleneckAnalysis(
                bottleneck_type="memory",
                severity=severity,
                component="system_memory",
                performance_impact=min(100, (avg_memory - 60) * 2.5),
                root_cause="High memory utilization detected",
                contributing_factors=[
                    "Memory leaks",
                    "Large data structures",
                    "Insufficient garbage collection"
                ],
                evidence={
                    "current_memory_percent": metrics.memory_percent,
                    "average_memory_percent": avg_memory,
                    "memory_used_bytes": metrics.memory_used,
                    "threshold_exceeded": self.config.memory_threshold
                },
                immediate_actions=[
                    "Identify memory leaks",
                    "Optimize data structures",
                    "Implement memory pooling"
                ],
                long_term_solutions=[
                    "Increase system memory",
                    "Implement efficient caching",
                    "Optimize memory allocation patterns"
                ]
            )
        
        return None
    
    async def _analyze_io_bottleneck(
        self,
        metrics: PerformanceMetrics,
        window: timedelta
    ) -> Optional[BottleneckAnalysis]:
        """Analyze I/O bottlenecks."""
        if metrics.disk_percent > self.config.disk_threshold:
            return BottleneckAnalysis(
                bottleneck_type="io",
                severity=Severity.HIGH,
                component="disk_io",
                performance_impact=min(100, (metrics.disk_percent - 70) * 3),
                root_cause="High disk utilization detected",
                contributing_factors=[
                    "Heavy disk I/O operations",
                    "Inefficient file access patterns",
                    "Insufficient disk space"
                ],
                evidence={
                    "disk_percent": metrics.disk_percent,
                    "disk_io_bytes": metrics.disk_io_bytes,
                    "threshold_exceeded": self.config.disk_threshold
                },
                immediate_actions=[
                    "Optimize file operations",
                    "Implement disk caching",
                    "Clean up unnecessary files"
                ],
                long_term_solutions=[
                    "Upgrade to SSD storage",
                    "Implement distributed storage",
                    "Optimize database queries"
                ]
            )
        
        return None
    
    async def _analyze_latency_bottleneck(
        self,
        metrics: PerformanceMetrics,
        window: timedelta
    ) -> Optional[BottleneckAnalysis]:
        """Analyze latency bottlenecks."""
        if metrics.latency_p95 > self.config.latency_threshold:
            return BottleneckAnalysis(
                bottleneck_type="latency",
                severity=Severity.HIGH,
                component="response_time",
                performance_impact=min(100, (metrics.latency_p95 / self.config.latency_threshold) * 50),
                root_cause="High response latency detected",
                contributing_factors=[
                    "Network latency",
                    "Database query performance",
                    "Complex processing logic"
                ],
                evidence={
                    "latency_p95": metrics.latency_p95,
                    "ai_response_time": metrics.ai_response_time,
                    "threshold_exceeded": self.config.latency_threshold
                },
                immediate_actions=[
                    "Optimize slow queries",
                    "Implement response caching",
                    "Reduce network round trips"
                ],
                long_term_solutions=[
                    "Implement CDN",
                    "Optimize algorithms",
                    "Use faster hardware"
                ]
            )
        
        return None
    
    async def _analyze_ai_bottleneck(
        self,
        metrics: PerformanceMetrics,
        window: timedelta
    ) -> Optional[BottleneckAnalysis]:
        """Analyze AI-specific bottlenecks."""
        issues = []
        
        if metrics.hallucination_rate > 0.2:  # 20% hallucination rate
            issues.append("High AI hallucination rate")
        
        if metrics.ai_response_time > 5000:  # 5 second response time
            issues.append("Slow AI response times")
        
        if metrics.context_window_usage > 0.9:  # 90% context window usage
            issues.append("Context window near capacity")
        
        if issues:
            return BottleneckAnalysis(
                bottleneck_type="ai_model",
                severity=Severity.MEDIUM,
                component="ai_processing",
                performance_impact=len(issues) * 20,
                root_cause="AI model performance issues",
                contributing_factors=issues,
                evidence={
                    "hallucination_rate": metrics.hallucination_rate,
                    "ai_response_time": metrics.ai_response_time,
                    "context_window_usage": metrics.context_window_usage,
                    "model_accuracy": metrics.model_accuracy
                },
                immediate_actions=[
                    "Optimize AI prompts",
                    "Implement response caching",
                    "Reduce context window usage"
                ],
                long_term_solutions=[
                    "Fine-tune AI models",
                    "Implement model ensemble",
                    "Upgrade to more capable models"
                ]
            )
        
        return None
    
    async def _analyze_workflow_bottleneck(
        self,
        metrics: PerformanceMetrics,
        window: timedelta
    ) -> Optional[BottleneckAnalysis]:
        """Analyze workflow bottlenecks."""
        issues = []
        
        if metrics.task_success_rate < 0.8:  # Less than 80% success rate
            issues.append("Low task success rate")
        
        if metrics.workflow_completion_rate < 0.6:  # Less than 60% completion
            issues.append("Low workflow completion rate")
        
        if metrics.average_task_duration > 3600:  # More than 1 hour per task
            issues.append("High average task duration")
        
        if issues:
            return BottleneckAnalysis(
                bottleneck_type="workflow",
                severity=Severity.MEDIUM,
                component="task_execution",
                performance_impact=len(issues) * 15,
                root_cause="Workflow execution inefficiencies",
                contributing_factors=issues,
                evidence={
                    "task_success_rate": metrics.task_success_rate,
                    "workflow_completion_rate": metrics.workflow_completion_rate,
                    "average_task_duration": metrics.average_task_duration
                },
                immediate_actions=[
                    "Analyze failed tasks",
                    "Optimize task dependencies",
                    "Improve error handling"
                ],
                long_term_solutions=[
                    "Redesign workflow structure",
                    "Implement better task prioritization",
                    "Add more comprehensive monitoring"
                ]
            )
        
        return None
    
    def _get_cached_analysis(self, cache_key: str) -> Optional[Any]:
        """Get cached analysis result."""
        if cache_key in self.analysis_cache:
            timestamp, result = self.analysis_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                return result
            else:
                del self.analysis_cache[cache_key]
        return None
    
    def _cache_analysis(self, cache_key: str, result: Any) -> None:
        """Cache analysis result."""
        self.analysis_cache[cache_key] = (datetime.utcnow(), result)
        
        # Clean old cache entries
        if len(self.analysis_cache) > 1000:
            cutoff_time = datetime.utcnow() - self.cache_ttl
            keys_to_remove = [
                key for key, (timestamp, _) in self.analysis_cache.items()
                if timestamp < cutoff_time
            ]
            for key in keys_to_remove:
                del self.analysis_cache[key]
    
    # Additional helper methods would be implemented here...
    # (truncated for brevity - the full implementation would include all the helper methods)
    
    async def _initialize_storage(self) -> None:
        """Initialize data storage."""
        # Placeholder for storage initialization
        pass
    
    async def _load_historical_data(self) -> None:
        """Load historical analytics data."""
        # Placeholder for loading historical data
        pass
    
    async def _initialize_ml_models(self) -> None:
        """Initialize machine learning models."""
        # Placeholder for ML model initialization
        pass
    
    async def _save_data(self) -> None:
        """Save current analytics data."""
        # Placeholder for data saving
        pass
    
    def _calculate_summary_statistics(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics for metrics."""
        if not metrics_list:
            return {}
        
        # Calculate statistics for key metrics
        stats = {}
        
        numeric_fields = [
            'cpu_percent', 'memory_percent', 'throughput', 'latency_p95',
            'error_rate', 'ai_response_time', 'code_quality_score'
        ]
        
        for field in numeric_fields:
            values = [getattr(m, field, 0) for m in metrics_list if hasattr(m, field)]
            if values:
                stats[field] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return stats
    
    async def _perform_trend_analysis(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime],
        prediction_horizon: timedelta
    ) -> TrendAnalysis:
        """Perform statistical trend analysis."""
        # Simple trend analysis implementation
        # In a full implementation, this would use more sophisticated statistical methods
        
        if len(values) < 2:
            return TrendAnalysis(
                start_time=timestamps[0] if timestamps else datetime.utcnow(),
                end_time=timestamps[-1] if timestamps else datetime.utcnow(),
                metric_name=metric_name,
                trend_direction="stable",
                data_points=len(values)
            )
        
        # Calculate basic statistics
        mean_value = statistics.mean(values)
        median_value = statistics.median(values)
        std_deviation = statistics.stdev(values) if len(values) > 1 else 0
        
        # Simple trend detection using linear regression
        x = list(range(len(values)))
        n = len(values)
        
        if n > 1:
            # Calculate slope
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
            
            # Determine trend direction
            if abs(slope) < std_deviation * 0.1:
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "improving"
                trend_strength = min(1.0, abs(slope) / (std_deviation or 1))
            else:
                trend_direction = "degrading"
                trend_strength = min(1.0, abs(slope) / (std_deviation or 1))
        else:
            trend_direction = "stable"
            trend_strength = 0.0
        
        return TrendAnalysis(
            start_time=timestamps[0],
            end_time=timestamps[-1],
            data_points=len(values),
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            mean_value=mean_value,
            median_value=median_value,
            std_deviation=std_deviation,
            coefficient_of_variation=std_deviation / mean_value if mean_value != 0 else 0,
            metric_name=metric_name,
            analysis_method="statistical"
        )
    
    async def _detect_threshold_anomalies(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Detect threshold-based anomalies."""
        alerts = []
        
        # CPU threshold alert
        if metrics.cpu_percent > self.config.cpu_threshold:
            alert = PerformanceAlert(
                alert_type=AlertType.THRESHOLD_EXCEEDED,
                severity=Severity.HIGH,
                title="High CPU Usage",
                description=f"CPU usage ({metrics.cpu_percent:.1f}%) exceeds threshold ({self.config.cpu_threshold}%)",
                affected_component="system_cpu",
                metric_name="cpu_percent",
                threshold_value=self.config.cpu_threshold,
                current_value=metrics.cpu_percent,
                performance_metrics=metrics
            )
            alerts.append(alert)
        
        # Memory threshold alert
        if metrics.memory_percent > self.config.memory_threshold:
            alert = PerformanceAlert(
                alert_type=AlertType.THRESHOLD_EXCEEDED,
                severity=Severity.HIGH,
                title="High Memory Usage",
                description=f"Memory usage ({metrics.memory_percent:.1f}%) exceeds threshold ({self.config.memory_threshold}%)",
                affected_component="system_memory",
                metric_name="memory_percent",
                threshold_value=self.config.memory_threshold,
                current_value=metrics.memory_percent,
                performance_metrics=metrics
            )
            alerts.append(alert)
        
        # Error rate threshold alert
        if metrics.error_rate > self.config.error_rate_threshold / 100:
            alert = PerformanceAlert(
                alert_type=AlertType.THRESHOLD_EXCEEDED,
                severity=Severity.CRITICAL,
                title="High Error Rate",
                description=f"Error rate ({metrics.error_rate*100:.1f}%) exceeds threshold ({self.config.error_rate_threshold}%)",
                affected_component="application",
                metric_name="error_rate",
                threshold_value=self.config.error_rate_threshold / 100,
                current_value=metrics.error_rate,
                performance_metrics=metrics
            )
            alerts.append(alert)
        
        return alerts
    
    async def _detect_statistical_anomalies(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Detect statistical anomalies using historical data."""
        alerts = []
        
        if len(self.metrics_history) < 10:  # Need sufficient historical data
            return alerts
        
        # Analyze recent metrics for anomalies
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 data points
        
        # CPU anomaly detection
        cpu_values = [m.cpu_percent for m in recent_metrics if m.cpu_percent > 0]
        if cpu_values and len(cpu_values) > 5:
            cpu_mean = statistics.mean(cpu_values)
            cpu_std = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            
            if cpu_std > 0 and abs(metrics.cpu_percent - cpu_mean) > (self.config.anomaly_sensitivity * cpu_std):
                alert = PerformanceAlert(
                    alert_type=AlertType.ANOMALY_DETECTED,
                    severity=Severity.MEDIUM,
                    title="CPU Usage Anomaly",
                    description=f"CPU usage ({metrics.cpu_percent:.1f}%) is {abs(metrics.cpu_percent - cpu_mean)/cpu_std:.1f} standard deviations from normal",
                    affected_component="system_cpu",
                    metric_name="cpu_percent",
                    current_value=metrics.cpu_percent,
                    performance_metrics=metrics
                )
                alerts.append(alert)
        
        return alerts
    
    async def _detect_trend_anomalies(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Detect trend-based anomalies."""
        # Placeholder for trend-based anomaly detection
        return []
    
    def _is_alert_suppressed(self, alert: PerformanceAlert) -> bool:
        """Check if alert should be suppressed."""
        alert_key = f"{alert.alert_type}_{alert.affected_component}_{alert.metric_name}"
        
        if alert_key in self.alert_suppression:
            suppression_time = self.alert_suppression[alert_key]
            if datetime.utcnow() < suppression_time:
                return True
            else:
                del self.alert_suppression[alert_key]
        
        return False
    
    async def _trigger_alert_callbacks(self, alert: PerformanceAlert) -> None:
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    async def _generate_bottleneck_recommendations(
        self,
        bottleneck: BottleneckAnalysis,
        metrics: PerformanceMetrics
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations for a specific bottleneck."""
        recommendations = []
        
        if bottleneck.bottleneck_type == "cpu":
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.CODE_OPTIMIZATION,
                title="Optimize CPU-intensive Operations",
                description="Implement algorithmic optimizations to reduce CPU usage",
                rationale=f"Current CPU usage is {metrics.cpu_percent:.1f}%, causing performance bottleneck",
                estimated_improvement=20.0,
                confidence_score=80.0,
                implementation_steps=[
                    "Profile CPU-intensive functions",
                    "Optimize algorithms and data structures",
                    "Implement parallel processing where possible",
                    "Add performance monitoring"
                ],
                target_components=["cpu_intensive_operations"],
                priority=Priority.HIGH
            ))
        
        elif bottleneck.bottleneck_type == "memory":
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.RESOURCE_SCALING,
                title="Optimize Memory Usage",
                description="Implement memory optimization strategies",
                rationale=f"Current memory usage is {metrics.memory_percent:.1f}%, approaching limits",
                estimated_improvement=25.0,
                confidence_score=85.0,
                implementation_steps=[
                    "Identify memory leaks",
                    "Optimize data structures",
                    "Implement memory pooling",
                    "Add memory monitoring"
                ],
                target_components=["memory_management"],
                priority=Priority.HIGH
            ))
        
        return recommendations
    
    async def _generate_general_recommendations(
        self,
        metrics: PerformanceMetrics
    ) -> List[OptimizationRecommendation]:
        """Generate general optimization recommendations."""
        recommendations = []
        
        # Cache hit rate optimization
        if metrics.cache_hit_rate < 0.8:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.CACHING,
                title="Improve Cache Hit Rate",
                description="Optimize caching strategy to improve performance",
                rationale=f"Current cache hit rate is {metrics.cache_hit_rate*100:.1f}%, below optimal",
                estimated_improvement=15.0,
                confidence_score=70.0,
                implementation_steps=[
                    "Analyze cache usage patterns",
                    "Optimize cache size and eviction policies",
                    "Implement more efficient caching strategy"
                ],
                target_components=["caching_system"],
                priority=Priority.MEDIUM
            ))
        
        # Code quality optimization
        if metrics.code_quality_score < 70:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.CODE_OPTIMIZATION,
                title="Improve Code Quality",
                description="Refactor code to improve maintainability and performance",
                rationale=f"Code quality score is {metrics.code_quality_score:.1f}, below standards",
                estimated_improvement=10.0,
                confidence_score=60.0,
                implementation_steps=[
                    "Refactor complex functions",
                    "Add comprehensive documentation",
                    "Improve test coverage",
                    "Address technical debt"
                ],
                target_components=["codebase"],
                priority=Priority.LOW
            ))
        
        return recommendations