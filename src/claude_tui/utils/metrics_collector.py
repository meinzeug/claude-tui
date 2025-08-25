"""
Metrics collection utilities for Claude-TUI.

Provides performance monitoring, usage tracking, and analytics collection.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Individual metric data point."""
    name: str
    value: Union[int, float, str]
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance-specific metric."""
    operation: str
    duration_ms: float
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class UsageMetric:
    """Usage tracking metric."""
    feature: str
    action: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for Claude-TUI.
    
    Provides performance monitoring, usage tracking, and analytics
    with in-memory storage and optional persistence.
    """
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.performance_metrics: deque = deque(maxlen=max_metrics)
        self.usage_metrics: deque = deque(maxlen=max_metrics)
        
        # Aggregated statistics
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Session tracking
        self.session_start_time = datetime.now()
        self.session_id = f"session_{int(time.time())}"
        
        logger.info(f"MetricsCollector initialized with max_metrics={max_metrics}")
    
    def record_metric(
        self, 
        name: str, 
        value: Union[int, float, str],
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a general metric."""
        metric = MetricData(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Update aggregated stats
        if isinstance(value, (int, float)):
            self.gauges[name] = float(value)
        
        logger.debug(f"Recorded metric: {name}={value} {unit}")
    
    def record_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            context=context or {}
        )
        
        self.performance_metrics.append(metric)
        
        # Update aggregated stats
        self.timers[operation].append(duration_ms)
        if success:
            self.counters[f"{operation}_success"] += 1
        else:
            self.counters[f"{operation}_error"] += 1
        
        logger.debug(f"Recorded performance: {operation} took {duration_ms:.2f}ms")
    
    def record_usage(
        self,
        feature: str,
        action: str,
        user_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a usage metric."""
        metric = UsageMetric(
            feature=feature,
            action=action,
            user_id=user_id,
            session_id=self.session_id,
            parameters=parameters or {}
        )
        
        self.usage_metrics.append(metric)
        
        # Update counters
        self.counters[f"usage_{feature}_{action}"] += 1
        self.counters[f"usage_{feature}_total"] += 1
        
        logger.debug(f"Recorded usage: {feature}.{action}")
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] += value
        self.record_metric(name, self.counters[name], "count")
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric."""
        self.gauges[name] = value
        self.record_metric(name, value, "gauge")
    
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        return TimingContext(self, operation)
    
    def get_performance_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        if operation:
            # Stats for specific operation
            if operation not in self.timers:
                return {}
            
            durations = self.timers[operation]
            if not durations:
                return {}
            
            return {
                "operation": operation,
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "success_count": self.counters.get(f"{operation}_success", 0),
                "error_count": self.counters.get(f"{operation}_error", 0)
            }
        else:
            # Stats for all operations
            stats = {}
            for op in self.timers:
                stats[op] = self.get_performance_stats(op)
            return stats
    
    def get_usage_stats(self, feature: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics."""
        if feature:
            # Stats for specific feature
            stats = {
                "feature": feature,
                "total_usage": self.counters.get(f"usage_{feature}_total", 0)
            }
            
            # Get action breakdowns
            actions = {}
            for key, count in self.counters.items():
                if key.startswith(f"usage_{feature}_") and key != f"usage_{feature}_total":
                    action = key[len(f"usage_{feature}_"):]
                    actions[action] = count
            
            stats["actions"] = actions
            return stats
        else:
            # Stats for all features
            features = set()
            for key in self.counters:
                if key.startswith("usage_") and key.endswith("_total"):
                    feature = key[6:-6]  # Remove "usage_" and "_total"
                    features.add(feature)
            
            stats = {}
            for feat in features:
                stats[feat] = self.get_usage_stats(feat)
            
            return stats
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        now = datetime.now()
        uptime = (now - self.session_start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "uptime_seconds": uptime,
            "start_time": self.session_start_time.isoformat(),
            "current_time": now.isoformat(),
            "total_metrics": len(self.metrics),
            "total_performance_metrics": len(self.performance_metrics),
            "total_usage_metrics": len(self.usage_metrics),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, List[Dict]]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_general = []
        recent_performance = []
        recent_usage = []
        
        # Filter general metrics
        for metric in reversed(self.metrics):
            if metric.timestamp < cutoff_time:
                break
            recent_general.append(asdict(metric))
        
        # Filter performance metrics
        for metric in reversed(self.performance_metrics):
            if metric.timestamp < cutoff_time:
                break
            recent_performance.append(asdict(metric))
        
        # Filter usage metrics
        for metric in reversed(self.usage_metrics):
            if metric.timestamp < cutoff_time:
                break
            recent_usage.append(asdict(metric))
        
        return {
            "general": recent_general,
            "performance": recent_performance,
            "usage": recent_usage
        }
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict]:
        """Export all metrics in specified format."""
        data = {
            "system": self.get_system_metrics(),
            "performance": self.get_performance_stats(),
            "usage": self.get_usage_stats(),
            "recent_metrics": self.get_recent_metrics(60)  # Last hour
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.metrics.clear()
        self.performance_metrics.clear()
        self.usage_metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.timers.clear()
        
        self.session_start_time = datetime.now()
        self.session_id = f"session_{int(time.time())}"
        
        logger.info("Metrics collector reset")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        import sys
        
        total_size = 0
        total_size += sys.getsizeof(self.metrics)
        total_size += sys.getsizeof(self.performance_metrics) 
        total_size += sys.getsizeof(self.usage_metrics)
        total_size += sys.getsizeof(self.counters)
        total_size += sys.getsizeof(self.gauges)
        total_size += sys.getsizeof(self.timers)
        
        return total_size / (1024 * 1024)  # Convert to MB


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, operation: str):
        self.collector = collector
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            
            self.collector.record_performance(
                operation=self.operation,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message
            )


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def record_metric(name: str, value: Union[int, float, str], **kwargs) -> None:
    """Convenience function to record a metric."""
    get_metrics_collector().record_metric(name, value, **kwargs)


def record_performance(operation: str, duration_ms: float, **kwargs) -> None:
    """Convenience function to record performance."""
    get_metrics_collector().record_performance(operation, duration_ms, **kwargs)


def record_usage(feature: str, action: str, **kwargs) -> None:
    """Convenience function to record usage."""
    get_metrics_collector().record_usage(feature, action, **kwargs)


def time_operation(operation: str):
    """Convenience function for timing operations."""
    return get_metrics_collector().time_operation(operation)


# Decorator for automatic performance tracking
def track_performance(operation_name: Optional[str] = None):
    """Decorator to automatically track function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with time_operation(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator