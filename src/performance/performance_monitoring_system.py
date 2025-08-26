#!/usr/bin/env python3
"""
Performance Monitoring and Alerting System

Real-time performance monitoring with intelligent alerting for <200ms API response targets.
Comprehensive system health monitoring with proactive optimization recommendations.

Features:
- Real-time performance metrics collection
- Intelligent threshold-based alerting
- Performance trend analysis and forecasting
- Automated optimization suggestions
- Dashboard integration and reporting
- Multi-tier alerting (WARNING, CRITICAL, EMERGENCY)
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading
import psutil
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_RESPONSE = "database_response"
    AGENT_UTILIZATION = "agent_utilization"


@dataclass
class MetricValue:
    """Individual metric measurement"""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass
class Alert:
    """Performance alert"""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    title: str
    description: str
    value: float
    threshold: float
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    actions_taken: List[str] = field(default_factory=list)
    
    @property
    def duration_minutes(self) -> float:
        end_time = self.resolved_at or time.time()
        return (end_time - self.timestamp) / 60


@dataclass
class PerformanceTarget:
    """Performance target configuration"""
    metric_type: MetricType
    target_value: float
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    unit: str = ""
    enabled: bool = True


class PerformanceMetricsCollector:
    """Collects performance metrics from various system components"""
    
    def __init__(self):
        self.metrics_buffer: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.collection_interval = 5.0  # seconds
        self.collecting = False
        self.collection_task: Optional[asyncio.Task] = None
    
    async def start_collection(self):
        """Start metrics collection"""
        if self.collecting:
            return
        
        self.collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Performance metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.collecting:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        timestamp = time.time()
        
        # Memory metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 ** 2)
        
        self.add_metric(MetricValue(
            MetricType.MEMORY_USAGE,
            memory_mb,
            "MB",
            timestamp,
            context={"process_id": process.pid}
        ))
        
        # CPU metrics
        cpu_percent = process.cpu_percent()
        if cpu_percent > 0:  # Only record when CPU usage is measured
            self.add_metric(MetricValue(
                MetricType.CPU_USAGE,
                cpu_percent,
                "%",
                timestamp
            ))
        
        # System metrics
        system_memory = psutil.virtual_memory()
        self.add_metric(MetricValue(
            MetricType.MEMORY_USAGE,
            system_memory.percent,
            "%",
            timestamp,
            context={"scope": "system"},
            tags=["system"]
        ))
    
    def add_metric(self, metric: MetricValue):
        """Add metric to buffer"""
        self.metrics_buffer.append(metric)
    
    def get_recent_metrics(
        self, 
        metric_type: Optional[MetricType] = None,
        minutes: int = 5
    ) -> List[MetricValue]:
        """Get recent metrics of specified type"""
        cutoff_time = time.time() - (minutes * 60)
        
        metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
        
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        
        return sorted(metrics, key=lambda x: x.timestamp)
    
    def get_metric_statistics(
        self, 
        metric_type: MetricType,
        minutes: int = 10
    ) -> Dict[str, float]:
        """Get statistical summary of metrics"""
        metrics = self.get_recent_metrics(metric_type, minutes)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'p95': values[int(len(values) * 0.95)] if len(values) > 20 else max(values),
            'p99': values[int(len(values) * 0.99)] if len(values) > 100 else max(values)
        }


class AlertingEngine:
    """Intelligent alerting engine with threshold monitoring"""
    
    def __init__(self, metrics_collector: PerformanceMetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Performance targets optimized for <200ms API responses
        self.targets = {
            MetricType.RESPONSE_TIME: PerformanceTarget(
                MetricType.RESPONSE_TIME,
                target_value=150.0,  # 150ms target
                warning_threshold=200.0,  # Warning at 200ms
                critical_threshold=500.0,  # Critical at 500ms
                emergency_threshold=1000.0,  # Emergency at 1s
                unit="ms"
            ),
            MetricType.MEMORY_USAGE: PerformanceTarget(
                MetricType.MEMORY_USAGE,
                target_value=100.0,  # 100MB target
                warning_threshold=150.0,  # Warning at 150MB
                critical_threshold=200.0,  # Critical at 200MB
                emergency_threshold=300.0,  # Emergency at 300MB
                unit="MB"
            ),
            MetricType.CPU_USAGE: PerformanceTarget(
                MetricType.CPU_USAGE,
                target_value=15.0,  # 15% target
                warning_threshold=50.0,  # Warning at 50%
                critical_threshold=80.0,  # Critical at 80%
                emergency_threshold=95.0,  # Emergency at 95%
                unit="%"
            ),
            MetricType.ERROR_RATE: PerformanceTarget(
                MetricType.ERROR_RATE,
                target_value=0.5,  # 0.5% target
                warning_threshold=1.0,  # Warning at 1%
                critical_threshold=5.0,  # Critical at 5%
                emergency_threshold=10.0,  # Emergency at 10%
                unit="%"
            ),
            MetricType.CACHE_HIT_RATE: PerformanceTarget(
                MetricType.CACHE_HIT_RATE,
                target_value=85.0,  # 85% target
                warning_threshold=70.0,  # Warning below 70%
                critical_threshold=50.0,  # Critical below 50%
                unit="%"
            )
        }
        
        # Alert suppression to prevent spam
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        self.last_alert_times: Dict[str, float] = {}
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    async def check_all_metrics(self):
        """Check all metrics against thresholds"""
        for metric_type, target in self.targets.items():
            if target.enabled:
                await self._check_metric_thresholds(metric_type, target)
    
    async def _check_metric_thresholds(self, metric_type: MetricType, target: PerformanceTarget):
        """Check specific metric against thresholds"""
        stats = self.metrics_collector.get_metric_statistics(metric_type, minutes=5)
        
        if not stats or stats['count'] < 3:
            return  # Not enough data
        
        # Use P95 for most metrics, mean for error rates
        check_value = stats['p95'] if metric_type != MetricType.ERROR_RATE else stats['mean']
        
        alert_level = None
        threshold_value = None
        
        # Determine alert level (reverse logic for cache hit rate)
        if metric_type == MetricType.CACHE_HIT_RATE:
            if check_value < target.critical_threshold:
                alert_level = AlertLevel.CRITICAL
                threshold_value = target.critical_threshold
            elif check_value < target.warning_threshold:
                alert_level = AlertLevel.WARNING
                threshold_value = target.warning_threshold
        else:
            if target.emergency_threshold and check_value > target.emergency_threshold:
                alert_level = AlertLevel.EMERGENCY
                threshold_value = target.emergency_threshold
            elif check_value > target.critical_threshold:
                alert_level = AlertLevel.CRITICAL
                threshold_value = target.critical_threshold
            elif check_value > target.warning_threshold:
                alert_level = AlertLevel.WARNING
                threshold_value = target.warning_threshold
        
        if alert_level:
            await self._trigger_alert(
                metric_type, alert_level, check_value, threshold_value, stats
            )
        else:
            # Check if we can resolve existing alerts
            await self._resolve_alerts(metric_type, check_value, target)
    
    async def _trigger_alert(
        self, 
        metric_type: MetricType, 
        level: AlertLevel, 
        value: float, 
        threshold: float,
        stats: Dict[str, float]
    ):
        """Trigger an alert"""
        alert_key = f"{metric_type.value}_{level.value}"
        
        # Check cooldown
        if alert_key in self.last_alert_times:
            time_since_last = time.time() - self.last_alert_times[alert_key]
            if time_since_last < self.alert_cooldown:
                return  # Still in cooldown
        
        # Check if alert already active
        if alert_key in self.active_alerts:
            return  # Alert already active
        
        alert_id = f"alert_{int(time.time())}_{alert_key}"
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            metric_type=metric_type,
            title=self._generate_alert_title(metric_type, level, value),
            description=self._generate_alert_description(metric_type, level, value, threshold, stats),
            value=value,
            threshold=threshold,
            timestamp=time.time()
        )
        
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = time.time()
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(f"ALERT TRIGGERED: {alert.title}")
    
    async def _resolve_alerts(self, metric_type: MetricType, current_value: float, target: PerformanceTarget):
        """Resolve alerts when metrics return to normal"""
        alerts_to_resolve = []
        
        for alert_key, alert in self.active_alerts.items():
            if alert.metric_type != metric_type:
                continue
            
            # Check if metric is back within acceptable range
            is_resolved = False
            
            if metric_type == MetricType.CACHE_HIT_RATE:
                # For cache hit rate, resolved when above warning threshold
                is_resolved = current_value >= target.warning_threshold
            else:
                # For other metrics, resolved when below warning threshold
                is_resolved = current_value <= target.warning_threshold
            
            if is_resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                alerts_to_resolve.append(alert_key)
                logger.info(f"ALERT RESOLVED: {alert.title} (duration: {alert.duration_minutes:.1f} min)")
        
        # Remove resolved alerts
        for alert_key in alerts_to_resolve:
            self.active_alerts.pop(alert_key, None)
    
    def _generate_alert_title(self, metric_type: MetricType, level: AlertLevel, value: float) -> str:
        """Generate alert title"""
        target = self.targets[metric_type]
        return f"{level.value.upper()}: {metric_type.value.replace('_', ' ').title()} - {value:.1f}{target.unit}"
    
    def _generate_alert_description(
        self, 
        metric_type: MetricType, 
        level: AlertLevel, 
        value: float, 
        threshold: float,
        stats: Dict[str, float]
    ) -> str:
        """Generate detailed alert description"""
        target = self.targets[metric_type]
        
        description = f"""
Performance Alert: {metric_type.value.replace('_', ' ').title()}

Current Value: {value:.2f}{target.unit}
Threshold: {threshold:.2f}{target.unit}
Target: {target.target_value:.2f}{target.unit}

Statistics (last 5 minutes):
- Count: {stats['count']} measurements
- Min: {stats['min']:.2f}{target.unit}
- Max: {stats['max']:.2f}{target.unit}
- Mean: {stats['mean']:.2f}{target.unit}
- P95: {stats['p95']:.2f}{target.unit}

Alert Level: {level.value.upper()}
"""
        
        # Add specific recommendations
        if metric_type == MetricType.RESPONSE_TIME:
            description += "\nRecommended Actions:\n- Check database query performance\n- Review cache hit rates\n- Analyze CPU and memory usage\n- Consider scaling up resources"
        elif metric_type == MetricType.MEMORY_USAGE:
            description += "\nRecommended Actions:\n- Run memory profiler\n- Check for memory leaks\n- Optimize garbage collection\n- Consider increasing memory limits"
        elif metric_type == MetricType.CPU_USAGE:
            description += "\nRecommended Actions:\n- Profile CPU-intensive operations\n- Optimize algorithms\n- Check for blocking operations\n- Consider horizontal scaling"
        
        return description.strip()
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by level"""
        alerts = list(self.active_alerts.values())
        if level:
            alerts = [a for a in alerts if a.level == level]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        active_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            active_counts[alert.level.value] += 1
        
        recent_resolved = [
            a for a in self.alert_history
            if a.resolved and a.resolved_at and (time.time() - a.resolved_at) < 3600  # Last hour
        ]
        
        return {
            'active_alerts': {
                'total': len(self.active_alerts),
                'emergency': active_counts['emergency'],
                'critical': active_counts['critical'],
                'warning': active_counts['warning'],
                'info': active_counts['info']
            },
            'recent_resolved': len(recent_resolved),
            'alert_rate_per_hour': len([
                a for a in self.alert_history
                if (time.time() - a.timestamp) < 3600
            ]),
            'system_health': self._calculate_system_health_score()
        }
    
    def _calculate_system_health_score(self) -> int:
        """Calculate overall system health score (0-100)"""
        score = 100
        
        # Deduct points for active alerts
        for alert in self.active_alerts.values():
            if alert.level == AlertLevel.EMERGENCY:
                score -= 30
            elif alert.level == AlertLevel.CRITICAL:
                score -= 20
            elif alert.level == AlertLevel.WARNING:
                score -= 10
            elif alert.level == AlertLevel.INFO:
                score -= 5
        
        return max(0, score)


class PerformanceMonitoringSystem:
    """Main performance monitoring system"""
    
    def __init__(self):
        self.metrics_collector = PerformanceMetricsCollector()
        self.alerting_engine = AlertingEngine(self.metrics_collector)
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Setup default alert handlers
        self.alerting_engine.add_alert_handler(self._log_alert_handler)
        
        # Performance dashboard data
        self.dashboard_data: Dict[str, Any] = {}
        
        logger.info("Performance monitoring system initialized")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            return
        
        await self.metrics_collector.start_collection()
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring system started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.metrics_collector.stop_collection()
        logger.info("Performance monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check all metrics against thresholds
                await self.alerting_engine.check_all_metrics()
                
                # Update dashboard data
                await self._update_dashboard_data()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _update_dashboard_data(self):
        """Update performance dashboard data"""
        self.dashboard_data = {
            'timestamp': time.time(),
            'system_health_score': self.alerting_engine._calculate_system_health_score(),
            'alert_summary': self.alerting_engine.get_alert_summary(),
            'metrics_summary': {},
            'performance_trends': {}
        }
        
        # Get metrics summaries
        for metric_type in MetricType:
            stats = self.metrics_collector.get_metric_statistics(metric_type, minutes=10)
            if stats:
                self.dashboard_data['metrics_summary'][metric_type.value] = stats
        
        # Calculate performance trends
        self.dashboard_data['performance_trends'] = await self._calculate_performance_trends()
    
    async def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends for key metrics"""
        trends = {}
        
        for metric_type in [MetricType.RESPONSE_TIME, MetricType.MEMORY_USAGE, MetricType.CPU_USAGE]:
            current_stats = self.metrics_collector.get_metric_statistics(metric_type, minutes=5)
            previous_stats = self.metrics_collector.get_metric_statistics(metric_type, minutes=10)
            
            if current_stats and previous_stats and previous_stats['mean'] > 0:
                change_percent = ((current_stats['mean'] - previous_stats['mean']) / previous_stats['mean']) * 100
                
                if change_percent > 10:
                    trends[metric_type.value] = "increasing"
                elif change_percent < -10:
                    trends[metric_type.value] = "decreasing"
                else:
                    trends[metric_type.value] = "stable"
            else:
                trends[metric_type.value] = "unknown"
        
        return trends
    
    def _log_alert_handler(self, alert: Alert):
        """Default log-based alert handler"""
        level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
            AlertLevel.EMERGENCY: logger.critical
        }
        
        log_func = level_map.get(alert.level, logger.info)
        log_func(f"PERFORMANCE ALERT: {alert.title} - {alert.description}")
    
    def add_email_alert_handler(self, smtp_server: str, smtp_port: int, 
                               username: str, password: str, 
                               recipients: List[str]):
        """Add email alert handler"""
        
        def email_handler(alert: Alert):
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                try:
                    self._send_alert_email(
                        alert, smtp_server, smtp_port, 
                        username, password, recipients
                    )
                except Exception as e:
                    logger.error(f"Failed to send alert email: {e}")
        
        self.alerting_engine.add_alert_handler(email_handler)
    
    def _send_alert_email(self, alert: Alert, smtp_server: str, smtp_port: int,
                         username: str, password: str, recipients: List[str]):
        """Send alert via email"""
        msg = MimeMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"[PERFORMANCE ALERT] {alert.title}"
        
        body = alert.description
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
    
    def record_api_response_time(self, endpoint: str, response_time_ms: float):
        """Record API response time metric"""
        self.metrics_collector.add_metric(MetricValue(
            MetricType.RESPONSE_TIME,
            response_time_ms,
            "ms",
            time.time(),
            context={"endpoint": endpoint}
        ))
    
    def record_error_rate(self, error_count: int, total_requests: int):
        """Record error rate metric"""
        error_rate = (error_count / max(total_requests, 1)) * 100
        self.metrics_collector.add_metric(MetricValue(
            MetricType.ERROR_RATE,
            error_rate,
            "%",
            time.time(),
            context={"error_count": error_count, "total_requests": total_requests}
        ))
    
    def record_cache_hit_rate(self, hit_rate_percent: float):
        """Record cache hit rate metric"""
        self.metrics_collector.add_metric(MetricValue(
            MetricType.CACHE_HIT_RATE,
            hit_rate_percent,
            "%",
            time.time()
        ))
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        return self.dashboard_data
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'system_health_score': self.alerting_engine._calculate_system_health_score(),
            'alert_summary': self.alerting_engine.get_alert_summary(),
            'active_alerts': [
                {
                    'title': alert.title,
                    'level': alert.level.value,
                    'metric_type': alert.metric_type.value,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'duration_minutes': alert.duration_minutes
                }
                for alert in self.alerting_engine.get_active_alerts()
            ],
            'metrics_summary': {},
            'performance_targets': {},
            'recommendations': []
        }
        
        # Add metrics summaries
        for metric_type in MetricType:
            stats = self.metrics_collector.get_metric_statistics(metric_type, minutes=30)
            if stats:
                report['metrics_summary'][metric_type.value] = stats
        
        # Add performance targets
        for metric_type, target in self.alerting_engine.targets.items():
            report['performance_targets'][metric_type.value] = {
                'target': target.target_value,
                'warning_threshold': target.warning_threshold,
                'critical_threshold': target.critical_threshold,
                'unit': target.unit
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_performance_recommendations()
        
        return report
    
    def _generate_performance_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check for high response times
        response_time_stats = self.metrics_collector.get_metric_statistics(
            MetricType.RESPONSE_TIME, minutes=15
        )
        if response_time_stats and response_time_stats['mean'] > 200:
            recommendations.append({
                'priority': 'high',
                'category': 'response_time',
                'title': 'Optimize API Response Times',
                'description': f"Average response time is {response_time_stats['mean']:.0f}ms, exceeding 200ms target",
                'actions': [
                    'Implement aggressive caching strategies',
                    'Optimize database queries',
                    'Add CDN for static assets',
                    'Consider horizontal scaling'
                ]
            })
        
        # Check for high memory usage
        memory_stats = self.metrics_collector.get_metric_statistics(
            MetricType.MEMORY_USAGE, minutes=15
        )
        if memory_stats and memory_stats['mean'] > 150:
            recommendations.append({
                'priority': 'medium',
                'category': 'memory',
                'title': 'Reduce Memory Usage',
                'description': f"Average memory usage is {memory_stats['mean']:.0f}MB, exceeding targets",
                'actions': [
                    'Run memory profiler to identify leaks',
                    'Implement object pooling',
                    'Optimize garbage collection settings',
                    'Review data structure usage'
                ]
            })
        
        # Check for low cache hit rates
        cache_stats = self.metrics_collector.get_metric_statistics(
            MetricType.CACHE_HIT_RATE, minutes=15
        )
        if cache_stats and cache_stats['mean'] < 70:
            recommendations.append({
                'priority': 'medium',
                'category': 'caching',
                'title': 'Improve Cache Performance',
                'description': f"Cache hit rate is {cache_stats['mean']:.1f}%, below 85% target",
                'actions': [
                    'Review cache key strategies',
                    'Increase cache TTL for stable data',
                    'Implement cache warming',
                    'Optimize cache size and eviction policies'
                ]
            })
        
        return recommendations


# Global monitoring system instance
_monitoring_system: Optional[PerformanceMonitoringSystem] = None


def get_monitoring_system() -> PerformanceMonitoringSystem:
    """Get global monitoring system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = PerformanceMonitoringSystem()
    return _monitoring_system


async def initialize_monitoring() -> PerformanceMonitoringSystem:
    """Initialize and start performance monitoring system"""
    monitoring_system = get_monitoring_system()
    await monitoring_system.start_monitoring()
    logger.info("Performance monitoring system initialized and started")
    return monitoring_system


# Convenience functions for metric recording
def record_api_response_time(endpoint: str, response_time_ms: float):
    """Record API response time"""
    system = get_monitoring_system()
    system.record_api_response_time(endpoint, response_time_ms)


def record_error_rate(error_count: int, total_requests: int):
    """Record error rate"""
    system = get_monitoring_system()
    system.record_error_rate(error_count, total_requests)


def record_cache_hit_rate(hit_rate_percent: float):
    """Record cache hit rate"""
    system = get_monitoring_system()
    system.record_cache_hit_rate(hit_rate_percent)


if __name__ == "__main__":
    # Example usage and testing
    async def test_monitoring_system():
        print("🔍 PERFORMANCE MONITORING SYSTEM - Testing")
        print("=" * 60)
        
        # Initialize monitoring system
        monitoring = await initialize_monitoring()
        
        # Simulate some metrics
        print("📊 Simulating performance metrics...")
        for i in range(10):
            # Simulate API response times
            response_time = 150 + (i * 20)  # Gradually increasing
            monitoring.record_api_response_time("/api/v1/test", response_time)
            
            # Simulate error rates
            error_rate = min(i * 0.5, 2.0)  # Gradually increasing error rate
            monitoring.record_error_rate(int(error_rate), 100)
            
            await asyncio.sleep(1)
        
        # Wait for monitoring to process
        await asyncio.sleep(2)
        
        # Check for alerts
        alerts = monitoring.alerting_engine.get_active_alerts()
        print(f"\\n🚨 Active Alerts: {len(alerts)}\")\n        for alert in alerts:\n            print(f\"   - {alert.title} ({alert.level.value})\")\n        \n        # Get performance report\n        report = monitoring.get_performance_report()\n        print(f\"\\n📈 Performance Report:\")\n        print(f\"   System Health Score: {report['system_health_score']}/100\")\n        print(f\"   Alert Summary: {report['alert_summary']['active_alerts']['total']} active alerts\")\n        print(f\"   Recommendations: {len(report['recommendations'])}\")\n        \n        for rec in report['recommendations'][:2]:\n            print(f\"\\n   🎯 {rec['title']} ({rec['priority']} priority)\")\n            print(f\"      {rec['description']}\")\n        \n        # Stop monitoring\n        await monitoring.stop_monitoring()\n        print(\"\\n✅ Performance monitoring test completed!\")\n    \n    # Run the test\n    asyncio.run(test_monitoring_system())