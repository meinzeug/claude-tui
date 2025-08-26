"""
Analytics test package initialization.

Provides common test utilities and fixtures for analytics testing.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
import numpy as np

from src.analytics.models import PerformanceMetrics
from src.core.types import SystemMetrics


def create_sample_metrics(count: int = 10, start_time: datetime = None) -> List[PerformanceMetrics]:
    """Create sample performance metrics for testing."""
    if start_time is None:
        start_time = datetime.now() - timedelta(hours=count)
    
    metrics = []
    for i in range(count):
        timestamp = start_time + timedelta(hours=i)
        
        system_metrics = SystemMetrics(
            cpu_usage=50 + np.random.normal(0, 15),
            memory_usage=60 + np.random.normal(0, 10),
            disk_io=40 + np.random.normal(0, 12),
            network_io=30 + np.random.normal(0, 8),
            timestamp=timestamp
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.0 + np.random.normal(0, 0.5),
            throughput=100 + np.random.normal(0, 20),
            error_rate=max(0, 0.01 + np.random.normal(0, 0.005)),
            latency_p95=1.5 + np.random.normal(0, 0.3)
        )
        
        metrics.append(perf_metrics)
    
    return metrics


def create_bottleneck_scenario(bottleneck_type: str = "cpu") -> List[PerformanceMetrics]:
    """Create metrics that demonstrate a specific bottleneck type."""
    metrics = create_sample_metrics(5)
    
    if bottleneck_type == "cpu":
        for metric in metrics:
            metric.base_metrics.cpu_usage = 90 + np.random.normal(0, 5)
            metric.execution_time = 5.0 + np.random.normal(0, 1.0)
    
    elif bottleneck_type == "memory":
        for metric in metrics:
            metric.base_metrics.memory_usage = 85 + np.random.normal(0, 5)
            metric.queue_depth = 20 + np.random.normal(0, 5)
    
    elif bottleneck_type == "io":
        for metric in metrics:
            metric.base_metrics.disk_io = 90 + np.random.normal(0, 5)
            metric.latency_p95 = 5.0 + np.random.normal(0, 1.0)
    
    return metrics


@pytest.fixture
def sample_metrics():
    """Pytest fixture for sample metrics."""
    return create_sample_metrics()


@pytest.fixture
def cpu_bottleneck_metrics():
    """Pytest fixture for CPU bottleneck scenario."""
    return create_bottleneck_scenario("cpu")


@pytest.fixture 
def memory_bottleneck_metrics():
    """Pytest fixture for memory bottleneck scenario."""
    return create_bottleneck_scenario("memory")


@pytest.fixture
def io_bottleneck_metrics():
    """Pytest fixture for I/O bottleneck scenario."""
    return create_bottleneck_scenario("io")


# Common test constants
TEST_CONFIDENCE_THRESHOLD = 0.7
TEST_IMPROVEMENT_THRESHOLD = 10.0
TEST_PROCESSING_TIME_LIMIT = 30.0