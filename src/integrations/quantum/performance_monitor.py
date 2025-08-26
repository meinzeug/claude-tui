"""
Performance Monitoring and Optimization Layer
High-performance monitoring and optimization for Universal Development Environment Intelligence
"""

import asyncio
import gc
import logging
import os
import psutil
import resource
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import weakref
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import tracemalloc

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class PerformanceLevel(str, Enum):
    """Performance level indicators"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Performance metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(str, Enum):
    """Performance alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "metadata": self.metadata
        }


@dataclass 
class PerformanceAlert:
    """Performance alert notification"""
    alert_id: str
    level: AlertLevel
    metric_name: str
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    environment: str = ""
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "metric_name": self.metric_name,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment,
            "resolved": self.resolved
        }


@dataclass
class SystemResources:
    """System resource utilization snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_available: int
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    process_count: int
    thread_count: int
    file_descriptors: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used": self.memory_used,
            "memory_available": self.memory_available,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "process_count": self.process_count,
            "thread_count": self.thread_count,
            "file_descriptors": self.file_descriptors
        }


class MetricCollector(ABC):
    """Abstract base class for metric collectors"""
    
    @abstractmethod
    async def collect(self) -> List[PerformanceMetric]:
        """Collect performance metrics"""
        pass
        
    @abstractmethod
    def get_collector_info(self) -> Dict[str, Any]:
        """Get collector information"""
        pass


class SystemMetricsCollector(MetricCollector):
    """System-level metrics collector"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        
    async def collect(self) -> List[PerformanceMetric]:
        """Collect system metrics"""
        metrics = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(PerformanceMetric(
                name="system.cpu.percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                labels={"collector": "system"}
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.extend([
                PerformanceMetric(
                    name="system.memory.percent",
                    value=memory.percent,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "system"}
                ),
                PerformanceMetric(
                    name="system.memory.used",
                    value=memory.used,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "system", "unit": "bytes"}
                ),
                PerformanceMetric(
                    name="system.memory.available",
                    value=memory.available,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "system", "unit": "bytes"}
                )
            ])
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.append(PerformanceMetric(
                name="system.disk.percent",
                value=(disk.used / disk.total) * 100,
                metric_type=MetricType.GAUGE,
                labels={"collector": "system", "mount": "/"}
            ))
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics.extend([
                PerformanceMetric(
                    name="system.network.bytes_sent",
                    value=network.bytes_sent,
                    metric_type=MetricType.COUNTER,
                    labels={"collector": "system"}
                ),
                PerformanceMetric(
                    name="system.network.bytes_recv",
                    value=network.bytes_recv,
                    metric_type=MetricType.COUNTER,
                    labels={"collector": "system"}
                )
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            
        return metrics
        
    def get_collector_info(self) -> Dict[str, Any]:
        return {
            "name": "SystemMetricsCollector",
            "description": "Collects system-level performance metrics",
            "metrics": [
                "system.cpu.percent",
                "system.memory.percent",
                "system.memory.used",
                "system.memory.available",
                "system.disk.percent",
                "system.network.bytes_sent",
                "system.network.bytes_recv"
            ]
        }


class ProcessMetricsCollector(MetricCollector):
    """Process-specific metrics collector"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        
    async def collect(self) -> List[PerformanceMetric]:
        """Collect process metrics"""
        metrics = []
        
        try:
            # Process CPU and memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            metrics.extend([
                PerformanceMetric(
                    name="process.cpu.percent",
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "process", "pid": str(self.process.pid)}
                ),
                PerformanceMetric(
                    name="process.memory.rss",
                    value=memory_info.rss,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "process", "pid": str(self.process.pid), "unit": "bytes"}
                ),
                PerformanceMetric(
                    name="process.memory.percent",
                    value=memory_percent,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "process", "pid": str(self.process.pid)}
                )
            ])
            
            # Thread and file descriptor counts
            num_threads = self.process.num_threads()
            num_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            
            metrics.extend([
                PerformanceMetric(
                    name="process.threads.count",
                    value=num_threads,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "process", "pid": str(self.process.pid)}
                ),
                PerformanceMetric(
                    name="process.file_descriptors.count",
                    value=num_fds,
                    metric_type=MetricType.GAUGE,
                    labels={"collector": "process", "pid": str(self.process.pid)}
                )
            ])
            
            # I/O counters
            try:
                io_counters = self.process.io_counters()
                metrics.extend([
                    PerformanceMetric(
                        name="process.io.read_bytes",
                        value=io_counters.read_bytes,
                        metric_type=MetricType.COUNTER,
                        labels={"collector": "process", "pid": str(self.process.pid)}
                    ),
                    PerformanceMetric(
                        name="process.io.write_bytes",
                        value=io_counters.write_bytes,
                        metric_type=MetricType.COUNTER,
                        labels={"collector": "process", "pid": str(self.process.pid)}
                    )
                ])
            except (psutil.AccessDenied, AttributeError):
                pass
                
        except Exception as e:
            self.logger.error(f"Failed to collect process metrics: {e}")
            
        return metrics
        
    def get_collector_info(self) -> Dict[str, Any]:
        return {
            "name": "ProcessMetricsCollector",
            "description": "Collects process-specific performance metrics",
            "metrics": [
                "process.cpu.percent",
                "process.memory.rss",
                "process.memory.percent",
                "process.threads.count",
                "process.file_descriptors.count",
                "process.io.read_bytes",
                "process.io.write_bytes"
            ]
        }


class PythonMetricsCollector(MetricCollector):
    """Python runtime metrics collector"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._gc_stats = {"collections": gc.get_stats()}
        
    async def collect(self) -> List[PerformanceMetric]:
        """Collect Python runtime metrics"""
        metrics = []
        
        try:
            # Memory usage with tracemalloc if available
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                metrics.extend([
                    PerformanceMetric(
                        name="python.memory.current",
                        value=current,
                        metric_type=MetricType.GAUGE,
                        labels={"collector": "python", "unit": "bytes"}
                    ),
                    PerformanceMetric(
                        name="python.memory.peak",
                        value=peak,
                        metric_type=MetricType.GAUGE,
                        labels={"collector": "python", "unit": "bytes"}
                    )
                ])
                
            # Garbage collection stats
            gc_stats = gc.get_stats()
            for generation, stats in enumerate(gc_stats):
                metrics.extend([
                    PerformanceMetric(
                        name="python.gc.collections",
                        value=stats.get('collections', 0),
                        metric_type=MetricType.COUNTER,
                        labels={"collector": "python", "generation": str(generation)}
                    ),
                    PerformanceMetric(
                        name="python.gc.collected",
                        value=stats.get('collected', 0),
                        metric_type=MetricType.COUNTER,
                        labels={"collector": "python", "generation": str(generation)}
                    ),
                    PerformanceMetric(
                        name="python.gc.uncollectable",
                        value=stats.get('uncollectable', 0),
                        metric_type=MetricType.COUNTER,
                        labels={"collector": "python", "generation": str(generation)}
                    )
                ])
                
            # Object counts
            objects = len(gc.get_objects())
            metrics.append(PerformanceMetric(
                name="python.objects.count",
                value=objects,
                metric_type=MetricType.GAUGE,
                labels={"collector": "python"}
            ))
            
            # Thread count
            thread_count = threading.active_count()
            metrics.append(PerformanceMetric(
                name="python.threads.count",
                value=thread_count,
                metric_type=MetricType.GAUGE,
                labels={"collector": "python"}
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect Python metrics: {e}")
            
        return metrics
        
    def get_collector_info(self) -> Dict[str, Any]:
        return {
            "name": "PythonMetricsCollector",
            "description": "Collects Python runtime performance metrics",
            "metrics": [
                "python.memory.current",
                "python.memory.peak", 
                "python.gc.collections",
                "python.gc.collected",
                "python.gc.uncollectable",
                "python.objects.count",
                "python.threads.count"
            ]
        }


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
    def get_metric(self) -> PerformanceMetric:
        """Get performance metric for this timer"""
        return PerformanceMetric(
            name=f"timer.{self.name}",
            value=self.duration or 0,
            metric_type=MetricType.TIMER,
            labels={**self.labels, "unit": "seconds"}
        )


class MetricAggregator:
    """Aggregates metrics over time windows"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.logger = logging.getLogger(__name__)
        
    def add_metric(self, metric: PerformanceMetric):
        """Add metric to aggregation"""
        key = self._get_metric_key(metric)
        self._metrics[key].append(metric)
        
    def get_aggregated_stats(self, metric_name: str, 
                           time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get aggregated statistics for a metric"""
        try:
            key = metric_name
            metrics = list(self._metrics[key])
            
            if not metrics:
                return {}
                
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                
            if not metrics:
                return {}
                
            values = [m.value for m in metrics]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "p50": statistics.median(values),
                "p90": statistics.quantiles(values, n=10)[8] if len(values) >= 10 else max(values),
                "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate aggregated stats for {metric_name}: {e}")
            return {}
            
    def get_rate(self, metric_name: str, time_window: timedelta) -> float:
        """Calculate rate of change for a metric"""
        try:
            key = metric_name
            metrics = list(self._metrics[key])
            
            if len(metrics) < 2:
                return 0.0
                
            # Filter by time window
            cutoff_time = datetime.now() - time_window
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if len(recent_metrics) < 2:
                return 0.0
                
            # Calculate rate (assuming counter metric)
            first = recent_metrics[0]
            last = recent_metrics[-1]
            
            value_diff = last.value - first.value
            time_diff = (last.timestamp - first.timestamp).total_seconds()
            
            return value_diff / time_diff if time_diff > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate rate for {metric_name}: {e}")
            return 0.0
            
    def _get_metric_key(self, metric: PerformanceMetric) -> str:
        """Get unique key for metric"""
        return metric.name


class PerformanceThresholdManager:
    """Manages performance thresholds and alerts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._thresholds: Dict[str, Dict[AlertLevel, float]] = {}
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
    def set_threshold(self, metric_name: str, level: AlertLevel, value: float):
        """Set performance threshold"""
        if metric_name not in self._thresholds:
            self._thresholds[metric_name] = {}
        self._thresholds[metric_name][level] = value
        
    def check_thresholds(self, metric: PerformanceMetric) -> List[PerformanceAlert]:
        """Check metric against thresholds"""
        alerts = []
        
        thresholds = self._thresholds.get(metric.name, {})
        if not thresholds:
            return alerts
            
        # Check each threshold level (from most severe to least)
        for level in [AlertLevel.CRITICAL, AlertLevel.ERROR, AlertLevel.WARNING, AlertLevel.INFO]:
            if level not in thresholds:
                continue
                
            threshold = thresholds[level]
            
            # Check if threshold is breached
            breached = False
            if level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
                breached = metric.value > threshold  # High values are bad
            else:
                breached = metric.value < threshold  # Low values might be concerning
                
            alert_id = f"{metric.name}.{level.value}"
            
            if breached:
                # Create new alert if not already active
                if alert_id not in self._active_alerts:
                    alert = PerformanceAlert(
                        alert_id=alert_id,
                        level=level,
                        metric_name=metric.name,
                        message=f"Metric {metric.name} breached {level.value} threshold: {metric.value} > {threshold}",
                        value=metric.value,
                        threshold=threshold,
                        environment=metric.labels.get("environment", "unknown")
                    )
                    
                    self._active_alerts[alert_id] = alert
                    alerts.append(alert)
                    
                    # Trigger callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            self.logger.error(f"Error in alert callback: {e}")
                            
            else:
                # Resolve alert if it was active
                if alert_id in self._active_alerts:
                    alert = self._active_alerts[alert_id]
                    alert.resolved = True
                    alerts.append(alert)
                    del self._active_alerts[alert_id]
                    
        return alerts
        
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return list(self._active_alerts.values())
        
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Register alert callback"""
        self._alert_callbacks.append(callback)
        
    def unregister_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Unregister alert callback"""
        try:
            self._alert_callbacks.remove(callback)
        except ValueError:
            pass


class PerformanceOptimizer:
    """Performance optimization engine"""
    
    def __init__(self, aggregator: MetricAggregator):
        self.aggregator = aggregator
        self.logger = logging.getLogger(__name__)
        self._optimization_rules: List[Callable[[Dict[str, Dict[str, float]]], List[str]]] = []
        
        self._register_default_rules()
        
    def _register_default_rules(self):
        """Register default optimization rules"""
        
        def memory_optimization_rule(stats: Dict[str, Dict[str, float]]) -> List[str]:
            """Memory optimization suggestions"""
            suggestions = []
            
            # Check system memory usage
            system_memory = stats.get("system.memory.percent", {})
            if system_memory.get("mean", 0) > 80:
                suggestions.append("System memory usage is high (>80%). Consider:")
                suggestions.append("- Increasing system RAM")
                suggestions.append("- Closing unused applications")
                suggestions.append("- Optimizing memory-intensive operations")
                
            # Check process memory usage
            process_memory = stats.get("process.memory.percent", {})
            if process_memory.get("mean", 0) > 50:
                suggestions.append("Process memory usage is high (>50%). Consider:")
                suggestions.append("- Implementing memory pooling")
                suggestions.append("- Using lazy loading for large objects")
                suggestions.append("- Running garbage collection more frequently")
                
            # Check Python object count growth
            python_objects = stats.get("python.objects.count", {})
            if python_objects.get("std_dev", 0) > python_objects.get("mean", 0) * 0.5:
                suggestions.append("Python object count is highly variable. Consider:")
                suggestions.append("- Using object pooling")
                suggestions.append("- Implementing proper cleanup in finally blocks")
                suggestions.append("- Reviewing object lifecycle management")
                
            return suggestions
            
        def cpu_optimization_rule(stats: Dict[str, Dict[str, float]]) -> List[str]:
            """CPU optimization suggestions"""
            suggestions = []
            
            # Check system CPU usage
            system_cpu = stats.get("system.cpu.percent", {})
            if system_cpu.get("mean", 0) > 70:
                suggestions.append("System CPU usage is high (>70%). Consider:")
                suggestions.append("- Implementing asynchronous processing")
                suggestions.append("- Using multiprocessing for CPU-bound tasks")
                suggestions.append("- Optimizing hot code paths")
                suggestions.append("- Adding caching to reduce computation")
                
            # Check process CPU usage
            process_cpu = stats.get("process.cpu.percent", {})
            if process_cpu.get("p95", 0) > 90:
                suggestions.append("Process CPU spikes detected (P95 >90%). Consider:")
                suggestions.append("- Profiling to identify bottlenecks")
                suggestions.append("- Breaking large operations into smaller chunks")
                suggestions.append("- Using background processing for heavy tasks")
                
            return suggestions
            
        def io_optimization_rule(stats: Dict[str, Dict[str, float]]) -> List[str]:
            """I/O optimization suggestions"""
            suggestions = []
            
            # Check file descriptor usage
            fd_count = stats.get("process.file_descriptors.count", {})
            if fd_count.get("max", 0) > 1000:
                suggestions.append("High file descriptor usage detected (>1000). Consider:")
                suggestions.append("- Implementing connection pooling")
                suggestions.append("- Ensuring proper file/connection cleanup")
                suggestions.append("- Using context managers for resource management")
                
            return suggestions
            
        self._optimization_rules.extend([
            memory_optimization_rule,
            cpu_optimization_rule,
            io_optimization_rule
        ])
        
    def generate_optimization_suggestions(self, 
                                        time_window: timedelta = timedelta(minutes=15)) -> List[str]:
        """Generate optimization suggestions based on recent metrics"""
        try:
            # Get aggregated stats for all metrics
            all_stats = {}
            
            # Common metrics to analyze
            metrics_to_analyze = [
                "system.cpu.percent",
                "system.memory.percent",
                "process.cpu.percent", 
                "process.memory.percent",
                "process.threads.count",
                "process.file_descriptors.count",
                "python.objects.count",
                "python.gc.collections"
            ]
            
            for metric_name in metrics_to_analyze:
                stats = self.aggregator.get_aggregated_stats(metric_name, time_window)
                if stats:
                    all_stats[metric_name] = stats
                    
            # Apply optimization rules
            all_suggestions = []
            for rule in self._optimization_rules:
                try:
                    suggestions = rule(all_stats)
                    all_suggestions.extend(suggestions)
                except Exception as e:
                    self.logger.error(f"Error in optimization rule: {e}")
                    
            return all_suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization suggestions: {e}")
            return []
            
    def register_optimization_rule(self, rule: Callable[[Dict[str, Dict[str, float]]], List[str]]):
        """Register custom optimization rule"""
        self._optimization_rules.append(rule)


class PerformanceMonitor:
    """
    Performance Monitoring and Optimization Layer
    High-performance monitoring and optimization for Universal Development Environment Intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self._collectors: List[MetricCollector] = []
        self._aggregator = MetricAggregator(
            window_size=config.get("metric_window_size", 1000)
        )
        self._threshold_manager = PerformanceThresholdManager()
        self._optimizer = PerformanceOptimizer(self._aggregator)
        
        # Monitoring state
        self._monitoring_active = False
        self._collection_task: Optional[asyncio.Task] = None
        self._collection_interval = config.get("collection_interval", 10.0)  # seconds
        
        # Performance data
        self._performance_history: deque = deque(
            maxlen=config.get("history_size", 1440)  # 24 hours at 1 minute intervals
        )
        
        # Initialize default collectors
        self._initialize_default_collectors()
        self._initialize_default_thresholds()
        
    def _initialize_default_collectors(self):
        """Initialize default metric collectors"""
        self._collectors = [
            SystemMetricsCollector(),
            ProcessMetricsCollector(),
            PythonMetricsCollector()
        ]
        
    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds"""
        # System thresholds
        self._threshold_manager.set_threshold("system.cpu.percent", AlertLevel.WARNING, 70.0)
        self._threshold_manager.set_threshold("system.cpu.percent", AlertLevel.CRITICAL, 90.0)
        self._threshold_manager.set_threshold("system.memory.percent", AlertLevel.WARNING, 80.0)
        self._threshold_manager.set_threshold("system.memory.percent", AlertLevel.CRITICAL, 95.0)
        
        # Process thresholds
        self._threshold_manager.set_threshold("process.memory.percent", AlertLevel.WARNING, 50.0)
        self._threshold_manager.set_threshold("process.memory.percent", AlertLevel.CRITICAL, 75.0)
        self._threshold_manager.set_threshold("process.file_descriptors.count", AlertLevel.WARNING, 1000)
        self._threshold_manager.set_threshold("process.file_descriptors.count", AlertLevel.CRITICAL, 2000)
        
    async def start_monitoring(self) -> bool:
        """Start performance monitoring"""
        try:
            if self._monitoring_active:
                return True
                
            self.logger.info("Starting performance monitoring")
            
            # Start tracemalloc for memory tracking
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                
            # Start collection task
            self._collection_task = asyncio.create_task(self._collection_loop())
            self._monitoring_active = True
            
            self.logger.info("Performance monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            return False
            
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            if not self._monitoring_active:
                return
                
            self.logger.info("Stopping performance monitoring")
            
            self._monitoring_active = False
            
            # Cancel collection task
            if self._collection_task:
                self._collection_task.cancel()
                try:
                    await self._collection_task
                except asyncio.CancelledError:
                    pass
                    
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping performance monitoring: {e}")
            
    def add_collector(self, collector: MetricCollector):
        """Add custom metric collector"""
        self._collectors.append(collector)
        
    def remove_collector(self, collector: MetricCollector):
        """Remove metric collector"""
        try:
            self._collectors.remove(collector)
        except ValueError:
            pass
            
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance snapshot"""
        try:
            # Get latest system resources
            resources = self._get_system_resources()
            
            # Get recent metric stats
            stats = {}
            for metric_name in ["system.cpu.percent", "system.memory.percent", 
                               "process.cpu.percent", "process.memory.percent"]:
                metric_stats = self._aggregator.get_aggregated_stats(
                    metric_name, timedelta(minutes=5)
                )
                if metric_stats:
                    stats[metric_name] = metric_stats
                    
            # Determine overall performance level
            performance_level = self._calculate_performance_level(stats)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "performance_level": performance_level.value,
                "system_resources": resources.to_dict(),
                "metric_stats": stats,
                "active_alerts": [alert.to_dict() for alert in self._threshold_manager.get_active_alerts()],
                "collectors": [collector.get_collector_info() for collector in self._collectors]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current performance: {e}")
            return {}
            
    def get_performance_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get performance history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            snapshot for snapshot in self._performance_history
            if datetime.fromisoformat(snapshot["timestamp"]) >= cutoff_time
        ]
        
    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions"""
        return self._optimizer.generate_optimization_suggestions()
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            current_perf = self.get_current_performance()
            history = self.get_performance_history(24)  # 24 hours
            suggestions = self.get_optimization_suggestions()
            
            # Calculate trends
            trends = self._calculate_performance_trends(history)
            
            return {
                "generated_at": datetime.now().isoformat(),
                "current_performance": current_perf,
                "performance_trends": trends,
                "optimization_suggestions": suggestions,
                "alert_summary": {
                    "active_alerts": len(self._threshold_manager.get_active_alerts()),
                    "alerts_by_level": self._get_alerts_by_level()
                },
                "monitoring_config": {
                    "collection_interval": self._collection_interval,
                    "collectors_count": len(self._collectors),
                    "monitoring_active": self._monitoring_active
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {}
            
    async def _collection_loop(self):
        """Main metric collection loop"""
        while self._monitoring_active:
            try:
                # Collect metrics from all collectors
                all_metrics = []
                for collector in self._collectors:
                    try:
                        metrics = await collector.collect()
                        all_metrics.extend(metrics)
                    except Exception as e:
                        self.logger.error(f"Error collecting metrics from {collector.__class__.__name__}: {e}")
                        
                # Process collected metrics
                for metric in all_metrics:
                    # Add to aggregator
                    self._aggregator.add_metric(metric)
                    
                    # Check thresholds
                    alerts = self._threshold_manager.check_thresholds(metric)
                    for alert in alerts:
                        if not alert.resolved:
                            self.logger.warning(f"Performance alert: {alert.message}")
                        else:
                            self.logger.info(f"Performance alert resolved: {alert.alert_id}")
                            
                # Take performance snapshot
                performance_snapshot = self.get_current_performance()
                if performance_snapshot:
                    self._performance_history.append(performance_snapshot)
                    
                await asyncio.sleep(self._collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance collection loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
                
    def _get_system_resources(self) -> SystemResources:
        """Get current system resource utilization"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk_usage = {}
            
            # Get disk usage for all mounted filesystems
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = (usage.used / usage.total) * 100
                except (PermissionError, FileNotFoundError):
                    continue
                    
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process information
            process = psutil.Process()
            
            return SystemResources(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used=memory.used,
                memory_available=memory.available,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=len(psutil.pids()),
                thread_count=process.num_threads(),
                file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system resources: {e}")
            return SystemResources(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                memory_used=0,
                memory_available=0,
                disk_usage={},
                network_io={},
                process_count=0,
                thread_count=0,
                file_descriptors=0
            )
            
    def _calculate_performance_level(self, stats: Dict[str, Dict[str, float]]) -> PerformanceLevel:
        """Calculate overall performance level"""
        try:
            # Weighted scoring system
            scores = []
            weights = []
            
            # CPU performance (higher is worse)
            cpu_stats = stats.get("system.cpu.percent", {})
            if cpu_stats:
                cpu_score = max(0, 100 - cpu_stats.get("mean", 0))
                scores.append(cpu_score)
                weights.append(0.3)
                
            # Memory performance (higher is worse)
            memory_stats = stats.get("system.memory.percent", {})
            if memory_stats:
                memory_score = max(0, 100 - memory_stats.get("mean", 0))
                scores.append(memory_score)
                weights.append(0.3)
                
            # Process CPU performance
            proc_cpu_stats = stats.get("process.cpu.percent", {})
            if proc_cpu_stats:
                proc_cpu_score = max(0, 100 - proc_cpu_stats.get("mean", 0))
                scores.append(proc_cpu_score)
                weights.append(0.2)
                
            # Process memory performance
            proc_mem_stats = stats.get("process.memory.percent", {})
            if proc_mem_stats:
                proc_mem_score = max(0, 100 - proc_mem_stats.get("mean", 0))
                scores.append(proc_mem_score)
                weights.append(0.2)
                
            if not scores:
                return PerformanceLevel.FAIR
                
            # Calculate weighted average
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            # Map to performance levels
            if weighted_score >= 80:
                return PerformanceLevel.EXCELLENT
            elif weighted_score >= 60:
                return PerformanceLevel.GOOD
            elif weighted_score >= 40:
                return PerformanceLevel.FAIR
            elif weighted_score >= 20:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL
                
        except Exception as e:
            self.logger.error(f"Failed to calculate performance level: {e}")
            return PerformanceLevel.FAIR
            
    def _calculate_performance_trends(self, history: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate performance trends from history"""
        trends = {}
        
        try:
            if len(history) < 2:
                return trends
                
            # Extract time series data
            cpu_values = []
            memory_values = []
            timestamps = []
            
            for snapshot in history:
                if "system_resources" in snapshot:
                    resources = snapshot["system_resources"]
                    cpu_values.append(resources.get("cpu_percent", 0))
                    memory_values.append(resources.get("memory_percent", 0))
                    timestamps.append(datetime.fromisoformat(snapshot["timestamp"]))
                    
            if len(cpu_values) >= 2:
                # Simple trend calculation (linear regression would be better)
                cpu_trend = "stable"
                if cpu_values[-1] > cpu_values[0] * 1.1:
                    cpu_trend = "increasing"
                elif cpu_values[-1] < cpu_values[0] * 0.9:
                    cpu_trend = "decreasing"
                trends["cpu_usage"] = cpu_trend
                
                memory_trend = "stable"
                if memory_values[-1] > memory_values[0] * 1.1:
                    memory_trend = "increasing"
                elif memory_values[-1] < memory_values[0] * 0.9:
                    memory_trend = "decreasing"
                trends["memory_usage"] = memory_trend
                
        except Exception as e:
            self.logger.error(f"Failed to calculate performance trends: {e}")
            
        return trends
        
    def _get_alerts_by_level(self) -> Dict[str, int]:
        """Get alert counts by level"""
        counts = defaultdict(int)
        
        for alert in self._threshold_manager.get_active_alerts():
            counts[alert.level.value] += 1
            
        return dict(counts)


# Utility decorators and context managers
def monitor_performance(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with PerformanceTimer(metric_name, labels) as timer:
                result = await func(*args, **kwargs)
            # In a real implementation, this would send the metric somewhere
            print(f"Performance metric: {timer.get_metric().to_dict()}")
            return result
            
        def sync_wrapper(*args, **kwargs):
            with PerformanceTimer(metric_name, labels) as timer:
                result = func(*args, **kwargs)
            # In a real implementation, this would send the metric somewhere
            print(f"Performance metric: {timer.get_metric().to_dict()}")
            return result
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "collection_interval": 5.0,
            "metric_window_size": 100,
            "history_size": 1440
        }
        
        monitor = PerformanceMonitor(config)
        
        try:
            await monitor.start_monitoring()
            
            # Let it collect some data
            await asyncio.sleep(15)
            
            # Get performance report
            report = monitor.get_performance_report()
            print(f"Performance report: {json.dumps(report, indent=2)}")
            
            # Get optimization suggestions
            suggestions = monitor.get_optimization_suggestions()
            print(f"Optimization suggestions: {suggestions}")
            
        finally:
            await monitor.stop_monitoring()
            
    # Run example
    # asyncio.run(main())