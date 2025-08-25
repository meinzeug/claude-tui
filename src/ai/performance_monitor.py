"""AI Performance Monitor Module

Comprehensive performance monitoring and analytics for AI services with:
- Real-time metrics collection and analysis
- Performance dashboards and visualization
- Bottleneck detection and optimization recommendations
- Historical trend analysis and reporting
- Alerting and notification systems
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path

import psutil
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_DEPTH = "queue_depth"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TimeWindow(Enum):
    """Time window for aggregations"""
    SECOND = 1
    MINUTE = 60
    HOUR = 3600
    DAY = 86400


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    metric_type: MetricType
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    unit: str = ""
    description: str = ""
    
    def add_point(self, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Add a metric point"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        self.points.append(point)
    
    def get_recent_points(self, seconds: int = 300) -> List[MetricPoint]:
        """Get points from last N seconds"""
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        return [point for point in self.points if point.timestamp >= cutoff]
    
    def get_average(self, seconds: int = 300) -> float:
        """Get average value over time window"""
        recent_points = self.get_recent_points(seconds)
        if not recent_points:
            return 0.0
        return sum(point.value for point in recent_points) / len(recent_points)
    
    def get_percentile(self, percentile: float, seconds: int = 300) -> float:
        """Get percentile value over time window"""
        recent_points = self.get_recent_points(seconds)
        if not recent_points:
            return 0.0
        values = [point.value for point in recent_points]
        return np.percentile(values, percentile)


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    id: str
    name: str
    metric_name: str
    condition: str  # e.g., ">", "<", "=="
    threshold: float
    level: AlertLevel
    time_window: int = 300  # seconds
    cooldown: int = 300  # seconds between alerts
    message_template: str = ""
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    timestamp: datetime
    duration_seconds: int
    metrics_summary: Dict[str, Dict[str, float]]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    alerts_triggered: List[str]
    system_health_score: float


class PerformanceMonitor:
    """
    Advanced Performance Monitor for AI Services
    
    Features:
    - Real-time metrics collection and storage
    - Performance dashboards and visualization
    - Bottleneck detection and analysis
    - Alert system with configurable thresholds
    - Historical trend analysis
    - Optimization recommendations
    """
    
    def __init__(
        self,
        collection_interval: int = 10,  # seconds
        retention_hours: int = 24,
        enable_alerts: bool = True,
        alert_cooldown: int = 300,
        enable_profiling: bool = True
    ):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.enable_alerts = enable_alerts
        self.alert_cooldown = alert_cooldown
        self.enable_profiling = enable_profiling
        
        # Metrics storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.custom_metrics: Dict[str, MetricSeries] = {}
        
        # Alerting system
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Performance reports
        self.reports: List[PerformanceReport] = []
        
        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.component_timers: Dict[str, List[float]] = defaultdict(list)
        
        self._setup_logging()
        self._initialize_default_metrics()
        self._initialize_default_alerts()
    
    def _setup_logging(self):
        """Configure logging for performance monitor"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def _initialize_default_metrics(self):
        """Initialize default system metrics"""
        
        default_metrics = [
            ("cpu_usage_percent", MetricType.RESOURCE_USAGE, "%", "CPU usage percentage"),
            ("memory_usage_percent", MetricType.RESOURCE_USAGE, "%", "Memory usage percentage"),
            ("disk_usage_percent", MetricType.RESOURCE_USAGE, "%", "Disk usage percentage"),
            ("network_io_bytes", MetricType.RESOURCE_USAGE, "bytes", "Network I/O bytes"),
            ("disk_io_bytes", MetricType.RESOURCE_USAGE, "bytes", "Disk I/O bytes"),
            ("request_latency_ms", MetricType.LATENCY, "ms", "Request latency in milliseconds"),
            ("request_throughput", MetricType.THROUGHPUT, "req/s", "Requests per second"),
            ("error_rate_percent", MetricType.ERROR_RATE, "%", "Error rate percentage"),
            ("cache_hit_rate_percent", MetricType.CACHE_HIT_RATE, "%", "Cache hit rate percentage"),
            ("queue_depth", MetricType.QUEUE_DEPTH, "items", "Queue depth in items")
        ]
        
        for name, metric_type, unit, description in default_metrics:
            self.metrics[name] = MetricSeries(
                name=name,
                metric_type=metric_type,
                unit=unit,
                description=description
            )
    
    def _initialize_default_alerts(self):
        """Initialize default performance alerts"""
        
        default_alerts = [
            ("high_cpu_usage", "cpu_usage_percent", ">", 80.0, AlertLevel.WARNING,
             "High CPU usage: {value}% > {threshold}%"),
            ("critical_cpu_usage", "cpu_usage_percent", ">", 95.0, AlertLevel.CRITICAL,
             "Critical CPU usage: {value}% > {threshold}%"),
            ("high_memory_usage", "memory_usage_percent", ">", 85.0, AlertLevel.WARNING,
             "High memory usage: {value}% > {threshold}%"),
            ("critical_memory_usage", "memory_usage_percent", ">", 95.0, AlertLevel.CRITICAL,
             "Critical memory usage: {value}% > {threshold}%"),
            ("high_error_rate", "error_rate_percent", ">", 5.0, AlertLevel.ERROR,
             "High error rate: {value}% > {threshold}%"),
            ("high_latency", "request_latency_ms", ">", 5000.0, AlertLevel.WARNING,
             "High request latency: {value}ms > {threshold}ms"),
            ("low_cache_hit_rate", "cache_hit_rate_percent", "<", 70.0, AlertLevel.WARNING,
             "Low cache hit rate: {value}% < {threshold}%")
        ]
        
        for name, metric_name, condition, threshold, level, message in default_alerts:
            self.alerts[name] = PerformanceAlert(
                id=name,
                name=name.replace("_", " ").title(),
                metric_name=metric_name,
                condition=condition,
                threshold=threshold,
                level=level,
                message_template=message
            )
    
    async def start(self):
        """Start performance monitoring"""
        logger.info("Starting AI performance monitor")
        
        # Start collection tasks
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.enable_alerts:
            self.alert_task = asyncio.create_task(self._alert_loop())
        
        logger.info("AI performance monitor started")
    
    async def stop(self):
        """Stop performance monitoring"""
        logger.info("Stopping AI performance monitor")
        
        # Cancel tasks
        for task in [self.collection_task, self.cleanup_task, self.alert_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("AI performance monitor stopped")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.CUSTOM,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a custom metric value"""
        
        if name not in self.custom_metrics:
            self.custom_metrics[name] = MetricSeries(
                name=name,
                metric_type=metric_type
            )
        
        self.custom_metrics[name].add_point(value, tags, metadata)
    
    def start_timer(self, component: str) -> str:
        """Start timing a component operation"""
        timer_id = f"{component}_{uuid.uuid4().hex[:8]}"
        if not hasattr(self, '_active_timers'):
            self._active_timers = {}
        self._active_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing and record the duration"""
        if not hasattr(self, '_active_timers') or timer_id not in self._active_timers:
            return 0.0
        
        duration = time.time() - self._active_timers.pop(timer_id)
        
        # Extract component name from timer_id
        component = timer_id.rsplit('_', 1)[0]
        self.component_timers[component].append(duration)
        
        # Keep only recent timings
        self.component_timers[component] = self.component_timers[component][-1000:]
        
        # Record as metric
        self.record_metric(f"{component}_duration_ms", duration * 1000, MetricType.LATENCY)
        
        return duration
    
    async def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        
        snapshot = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
            'system_metrics': {},
            'custom_metrics': {},
            'component_timings': {}
        }
        
        # System metrics
        for name, series in self.metrics.items():
            if series.points:
                latest = series.points[-1]
                snapshot['system_metrics'][name] = {
                    'current': latest.value,
                    'average_5min': series.get_average(300),
                    'average_1hour': series.get_average(3600),
                    'p95_5min': series.get_percentile(95, 300),
                    'unit': series.unit
                }
        
        # Custom metrics
        for name, series in self.custom_metrics.items():
            if series.points:
                latest = series.points[-1]
                snapshot['custom_metrics'][name] = {
                    'current': latest.value,
                    'average_5min': series.get_average(300),
                    'p95_5min': series.get_percentile(95, 300)
                }
        
        # Component timings
        for component, timings in self.component_timers.items():
            if timings:
                snapshot['component_timings'][component] = {
                    'average_ms': np.mean(timings) * 1000,
                    'p95_ms': np.percentile(timings, 95) * 1000,
                    'p99_ms': np.percentile(timings, 99) * 1000,
                    'count': len(timings)
                }
        
        return snapshot
    
    async def generate_performance_report(
        self,
        duration_hours: int = 1
    ) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=duration_hours)
        duration_seconds = duration_hours * 3600
        
        # Collect metrics summary
        metrics_summary = {}
        for name, series in {**self.metrics, **self.custom_metrics}.items():
            recent_points = series.get_recent_points(duration_seconds)
            if recent_points:
                values = [point.value for point in recent_points]
                metrics_summary[name] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'count': len(values)
                }
        
        # Identify bottlenecks
        bottlenecks = await self._identify_bottlenecks(metrics_summary)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(metrics_summary, bottlenecks)
        
        # Get recent alerts
        recent_alerts = [
            alert['message'] for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= start_time
        ]
        
        # Calculate health score
        health_score = self._calculate_health_score(metrics_summary)
        
        report = PerformanceReport(
            timestamp=end_time,
            duration_seconds=duration_seconds,
            metrics_summary=metrics_summary,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            alerts_triggered=recent_alerts,
            system_health_score=health_score
        )
        
        self.reports.append(report)
        
        # Keep only recent reports
        self.reports = self.reports[-100:]
        
        return report
    
    def add_custom_alert(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        level: AlertLevel = AlertLevel.WARNING,
        message_template: str = "",
        time_window: int = 300
    ):
        """Add custom performance alert"""
        
        self.alerts[name] = PerformanceAlert(
            id=name,
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            level=level,
            message_template=message_template or f"{metric_name} {condition} {threshold}",
            time_window=time_window
        )
        
        logger.info(f"Added custom alert: {name}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        
        snapshot = await self.get_metrics_snapshot()
        
        # Get recent alerts
        recent_alerts = self.alert_history[-10:] if self.alert_history else []
        
        # Get active alerts
        active_alerts_list = [
            {
                'name': alert_name,
                'level': alert.level.value,
                'message': alert.message_template,
                'since': triggered_time.isoformat()
            }
            for alert_name, triggered_time in self.active_alerts.items()
            if alert_name in self.alerts
            for alert in [self.alerts[alert_name]]
        ]
        
        return {
            **snapshot,
            'recent_alerts': recent_alerts,
            'active_alerts': active_alerts_list,
            'health_score': self._calculate_health_score(snapshot.get('system_metrics', {})),
            'total_metrics': len(self.metrics) + len(self.custom_metrics),
            'total_alerts': len(self.alerts),
            'active_alert_count': len(self.active_alerts)
        }
    
    async def _collection_loop(self):
        """Background metrics collection loop"""
        
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage_percent'].add_point(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage_percent'].add_point(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics['disk_usage_percent'].add_point(disk_percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics['network_io_bytes'].add_point(net_io.bytes_sent + net_io.bytes_recv)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:  # May be None on some systems
                self.metrics['disk_io_bytes'].add_point(disk_io.read_bytes + disk_io.write_bytes)
            
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop for old metrics"""
        
        while True:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_metrics(self):
        """Remove old metric points beyond retention period"""
        
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        cleaned_count = 0
        
        # Clean system metrics
        for series in self.metrics.values():
            original_count = len(series.points)
            series.points = deque(
                (point for point in series.points if point.timestamp >= cutoff),
                maxlen=series.points.maxlen
            )
            cleaned_count += original_count - len(series.points)
        
        # Clean custom metrics
        for series in self.custom_metrics.values():
            original_count = len(series.points)
            series.points = deque(
                (point for point in series.points if point.timestamp >= cutoff),
                maxlen=series.points.maxlen
            )
            cleaned_count += original_count - len(series.points)
        
        # Clean alert history
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff
        ]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old metric points")
    
    async def _alert_loop(self):
        """Background alert checking loop"""
        
        while True:
            try:
                await self._check_alerts()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(60)
    
    async def _check_alerts(self):
        """Check all alerts against current metrics"""
        
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            try:
                await self._evaluate_alert(alert)
            except Exception as e:
                logger.error(f"Alert evaluation failed for {alert.name}: {e}")
    
    async def _evaluate_alert(self, alert: PerformanceAlert):
        """Evaluate a single alert condition"""
        
        # Get metric series
        series = self.metrics.get(alert.metric_name) or self.custom_metrics.get(alert.metric_name)
        if not series:
            return
        
        # Get recent average
        current_value = series.get_average(alert.time_window)
        if current_value == 0.0:  # No data
            return
        
        # Check condition
        triggered = False
        if alert.condition == ">": 
            triggered = current_value > alert.threshold
        elif alert.condition == "<":
            triggered = current_value < alert.threshold
        elif alert.condition == ">=":
            triggered = current_value >= alert.threshold
        elif alert.condition == "<=":
            triggered = current_value <= alert.threshold
        elif alert.condition == "==":
            triggered = abs(current_value - alert.threshold) < 0.01
        
        now = datetime.utcnow()
        
        if triggered:
            # Check cooldown
            if (alert.last_triggered and 
                (now - alert.last_triggered).total_seconds() < alert.cooldown):
                return
            
            # Trigger alert
            await self._trigger_alert(alert, current_value)
            
        else:
            # Clear active alert if it exists
            if alert.id in self.active_alerts:
                del self.active_alerts[alert.id]
    
    async def _trigger_alert(self, alert: PerformanceAlert, current_value: float):
        """Trigger an alert"""
        
        now = datetime.utcnow()
        
        # Format message
        message = alert.message_template.format(
            value=current_value,
            threshold=alert.threshold,
            metric=alert.metric_name
        )
        
        # Update alert tracking
        alert.last_triggered = now
        alert.trigger_count += 1
        self.active_alerts[alert.id] = now
        
        # Add to history
        alert_record = {
            'id': alert.id,
            'name': alert.name,
            'level': alert.level.value,
            'message': message,
            'value': current_value,
            'threshold': alert.threshold,
            'timestamp': now.isoformat()
        }
        
        self.alert_history.append(alert_record)
        
        # Keep only recent alerts
        self.alert_history = self.alert_history[-1000:]
        
        logger.warning(f"ALERT [{alert.level.value.upper()}] {alert.name}: {message}")
    
    async def _identify_bottlenecks(
        self,
        metrics_summary: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from metrics"""
        
        bottlenecks = []
        
        # CPU bottleneck
        cpu_metrics = metrics_summary.get('cpu_usage_percent', {})
        if cpu_metrics.get('p95', 0) > 80:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high' if cpu_metrics['p95'] > 95 else 'medium',
                'metric': 'cpu_usage_percent',
                'value': cpu_metrics['p95'],
                'description': f"High CPU usage (P95: {cpu_metrics['p95']:.1f}%)"
            })
        
        # Memory bottleneck
        memory_metrics = metrics_summary.get('memory_usage_percent', {})
        if memory_metrics.get('p95', 0) > 85:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high' if memory_metrics['p95'] > 95 else 'medium',
                'metric': 'memory_usage_percent',
                'value': memory_metrics['p95'],
                'description': f"High memory usage (P95: {memory_metrics['p95']:.1f}%)"
            })
        
        # Latency bottleneck
        latency_metrics = metrics_summary.get('request_latency_ms', {})
        if latency_metrics.get('p95', 0) > 2000:
            bottlenecks.append({
                'type': 'latency',
                'severity': 'high' if latency_metrics['p95'] > 5000 else 'medium',
                'metric': 'request_latency_ms',
                'value': latency_metrics['p95'],
                'description': f"High request latency (P95: {latency_metrics['p95']:.0f}ms)"
            })
        
        return bottlenecks
    
    async def _generate_recommendations(
        self,
        metrics_summary: Dict[str, Dict[str, float]],
        bottlenecks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # CPU recommendations
        if any(b['type'] == 'cpu' for b in bottlenecks):
            recommendations.append(
                "Consider optimizing CPU-intensive operations or adding more CPU resources"
            )
            recommendations.append(
                "Review async/await usage to prevent blocking operations"
            )
        
        # Memory recommendations
        if any(b['type'] == 'memory' for b in bottlenecks):
            recommendations.append(
                "Investigate memory leaks and optimize data structures"
            )
            recommendations.append(
                "Consider implementing memory pooling or caching strategies"
            )
        
        # Latency recommendations
        if any(b['type'] == 'latency' for b in bottlenecks):
            recommendations.append(
                "Optimize database queries and add appropriate indexes"
            )
            recommendations.append(
                "Implement caching for frequently accessed data"
            )
            recommendations.append(
                "Consider using CDN for static content delivery"
            )
        
        # Cache recommendations
        cache_hit_rate = metrics_summary.get('cache_hit_rate_percent', {}).get('mean', 100)
        if cache_hit_rate < 70:
            recommendations.append(
                f"Improve cache hit rate (currently {cache_hit_rate:.1f}%) by optimizing cache keys and TTL"
            )
        
        # Error rate recommendations
        error_rate = metrics_summary.get('error_rate_percent', {}).get('mean', 0)
        if error_rate > 1:
            recommendations.append(
                f"Investigate and fix errors (current rate: {error_rate:.1f}%)"
            )
        
        return recommendations
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)"""
        
        score = 100.0
        
        # CPU health (weight: 25%)
        cpu_usage = metrics.get('cpu_usage_percent', {}).get('current', 0)
        if cpu_usage > 95:
            score -= 25
        elif cpu_usage > 80:
            score -= 15
        elif cpu_usage > 60:
            score -= 5
        
        # Memory health (weight: 25%)
        memory_usage = metrics.get('memory_usage_percent', {}).get('current', 0)
        if memory_usage > 95:
            score -= 25
        elif memory_usage > 85:
            score -= 15
        elif memory_usage > 70:
            score -= 5
        
        # Error rate health (weight: 25%)
        error_rate = metrics.get('error_rate_percent', {}).get('current', 0)
        if error_rate > 10:
            score -= 25
        elif error_rate > 5:
            score -= 15
        elif error_rate > 1:
            score -= 5
        
        # Latency health (weight: 25%)
        latency = metrics.get('request_latency_ms', {}).get('current', 0)
        if latency > 10000:
            score -= 25
        elif latency > 5000:
            score -= 15
        elif latency > 2000:
            score -= 5
        
        return max(0.0, score)
