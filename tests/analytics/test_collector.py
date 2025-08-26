"""
Test suite for the MetricsCollector and related components.

Tests metrics collection, aggregation, real-time streaming,
and integration with system monitoring.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import psutil
from typing import List, Dict, Any

from src.analytics.collector import (
    MetricsCollector, MetricsAggregator, StreamingMetricsCollector,
    CollectionConfiguration
)
from src.analytics.models import PerformanceMetrics
from src.core.types import SystemMetrics


class TestMetricsAggregator:
    """Test suite for the MetricsAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create a MetricsAggregator instance for testing."""
        return MetricsAggregator()

    @pytest.fixture
    def sample_metrics_list(self):
        """Create a list of sample metrics for aggregation testing."""
        metrics_list = []
        base_time = datetime.now()
        
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i)
            system_metrics = SystemMetrics(
                cpu_usage=50 + i * 2,  # Increasing CPU usage
                memory_usage=60 + i,   # Increasing memory usage
                disk_io=30 + i * 0.5,
                network_io=20 + i * 0.3,
                timestamp=timestamp
            )
            
            perf_metrics = PerformanceMetrics(
                base_metrics=system_metrics,
                execution_time=2.0 + i * 0.1,
                throughput=100 - i * 2,  # Decreasing throughput
                error_rate=0.01 + i * 0.001,
                latency_p95=1.5 + i * 0.05
            )
            metrics_list.append(perf_metrics)
        
        return metrics_list

    def test_aggregate_cpu_metrics(self, aggregator, sample_metrics_list):
        """Test CPU metrics aggregation."""
        result = aggregator.aggregate_metrics(sample_metrics_list, "1h")
        
        assert 'cpu_usage' in result
        assert 'avg' in result['cpu_usage']
        assert 'min' in result['cpu_usage']
        assert 'max' in result['cpu_usage']
        assert 'p95' in result['cpu_usage']
        
        # Verify calculations
        expected_avg = sum(m.base_metrics.cpu_usage for m in sample_metrics_list) / len(sample_metrics_list)
        assert abs(result['cpu_usage']['avg'] - expected_avg) < 0.1

    def test_aggregate_performance_metrics(self, aggregator, sample_metrics_list):
        """Test performance metrics aggregation."""
        result = aggregator.aggregate_metrics(sample_metrics_list, "1h")
        
        assert 'execution_time' in result
        assert 'throughput' in result
        assert 'error_rate' in result
        assert 'latency_p95' in result
        
        # Verify throughput aggregation
        expected_throughput_avg = sum(m.throughput for m in sample_metrics_list) / len(sample_metrics_list)
        assert abs(result['throughput']['avg'] - expected_throughput_avg) < 0.1

    def test_time_window_aggregation(self, aggregator, sample_metrics_list):
        """Test aggregation with different time windows."""
        # Test 5-minute window
        result_5m = aggregator.aggregate_metrics(sample_metrics_list, "5m")
        
        # Test 1-hour window
        result_1h = aggregator.aggregate_metrics(sample_metrics_list, "1h")
        
        # Both should have the same structure
        assert set(result_5m.keys()) == set(result_1h.keys())
        
        # Values might differ due to time window filtering
        for key in result_5m.keys():
            assert isinstance(result_5m[key], dict)
            assert isinstance(result_1h[key], dict)

    def test_percentile_calculations(self, aggregator, sample_metrics_list):
        """Test percentile calculations in aggregation."""
        result = aggregator.aggregate_metrics(sample_metrics_list, "1h")
        
        for metric_key in ['cpu_usage', 'memory_usage', 'execution_time']:
            if metric_key in result:
                assert 'p50' in result[metric_key]
                assert 'p95' in result[metric_key]
                assert 'p99' in result[metric_key]
                
                # P99 should be >= P95 should be >= P50
                assert result[metric_key]['p99'] >= result[metric_key]['p95']
                assert result[metric_key]['p95'] >= result[metric_key]['p50']

    def test_empty_metrics_handling(self, aggregator):
        """Test handling of empty metrics list."""
        result = aggregator.aggregate_metrics([], "1h")
        assert isinstance(result, dict)
        # Should return empty or default structure
        
    def test_single_metric_aggregation(self, aggregator):
        """Test aggregation with single metric."""
        system_metrics = SystemMetrics(
            cpu_usage=75.0,
            memory_usage=80.0,
            disk_io=50.0,
            network_io=30.0,
            timestamp=datetime.now()
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.5,
            throughput=120.0,
            error_rate=0.02
        )
        
        result = aggregator.aggregate_metrics([perf_metrics], "1h")
        
        # For single metric, avg should equal min and max
        assert result['cpu_usage']['avg'] == result['cpu_usage']['min']
        assert result['cpu_usage']['avg'] == result['cpu_usage']['max']


class TestMetricsCollector:
    """Test suite for the MetricsCollector class."""

    @pytest.fixture
    def collector_config(self):
        """Create collector configuration for testing."""
        return CollectionConfiguration(
            collection_interval=1.0,
            enable_system_metrics=True,
            enable_process_metrics=True,
            enable_custom_metrics=True,
            buffer_size=100
        )

    @pytest.fixture
    def collector(self, collector_config):
        """Create a MetricsCollector instance for testing."""
        return MetricsCollector(collector_config)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics(self, mock_net, mock_disk, mock_memory, mock_cpu, collector):
        """Test system metrics collection."""
        # Mock psutil responses
        mock_cpu.return_value = 75.5
        mock_memory.return_value.percent = 65.2
        mock_disk.return_value.read_bytes = 1024000
        mock_disk.return_value.write_bytes = 512000
        mock_net.return_value.bytes_sent = 204800
        mock_net.return_value.bytes_recv = 409600
        
        metrics = collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_usage == 75.5
        assert metrics.memory_usage == 65.2
        assert metrics.disk_io > 0
        assert metrics.network_io > 0
        assert isinstance(metrics.timestamp, datetime)

    @patch('psutil.Process')
    def test_collect_process_metrics(self, mock_process_class, collector):
        """Test process-specific metrics collection."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.0
        mock_process.memory_info.return_value.rss = 104857600  # 100MB
        mock_process.num_threads.return_value = 8
        mock_process_class.return_value = mock_process
        
        metrics = collector.collect_process_metrics()
        
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'thread_count' in metrics
        assert metrics['cpu_usage'] == 25.0
        assert metrics['memory_usage'] == 104857600

    def test_add_custom_metric(self, collector):
        """Test adding custom metrics."""
        # Add a custom metric
        collector.add_custom_metric("custom_counter", 42)
        collector.add_custom_metric("custom_gauge", 3.14)
        collector.add_custom_metric("custom_histogram", [1, 2, 3, 4, 5])
        
        custom_metrics = collector.get_custom_metrics()
        
        assert "custom_counter" in custom_metrics
        assert "custom_gauge" in custom_metrics
        assert "custom_histogram" in custom_metrics
        assert custom_metrics["custom_counter"] == 42
        assert custom_metrics["custom_gauge"] == 3.14
        assert custom_metrics["custom_histogram"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_collect_metrics_async(self, collector):
        """Test asynchronous metrics collection."""
        with patch.object(collector, 'collect_system_metrics') as mock_system, \
             patch.object(collector, 'collect_process_metrics') as mock_process:
            
            mock_system.return_value = SystemMetrics(
                cpu_usage=50.0, memory_usage=60.0, disk_io=30.0, 
                network_io=20.0, timestamp=datetime.now()
            )
            mock_process.return_value = {"cpu_usage": 15.0, "memory_usage": 52428800}
            
            metrics = await collector.collect_metrics_async()
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.base_metrics.cpu_usage == 50.0
            mock_system.assert_called_once()
            mock_process.assert_called_once()

    def test_plugin_system(self, collector):
        """Test plugin system for extensible metrics collection."""
        # Mock plugin
        def custom_plugin():
            return {"plugin_metric": 123.45}
        
        collector.register_plugin("test_plugin", custom_plugin)
        plugins = collector.get_plugins()
        
        assert "test_plugin" in plugins
        
        # Execute plugin
        plugin_result = collector.execute_plugin("test_plugin")
        assert plugin_result["plugin_metric"] == 123.45

    def test_metrics_filtering(self, collector):
        """Test metrics filtering functionality."""
        # Set up filters
        collector.add_filter("cpu_threshold", lambda m: m.base_metrics.cpu_usage < 90.0)
        collector.add_filter("memory_threshold", lambda m: m.base_metrics.memory_usage < 85.0)
        
        # Create test metrics
        high_cpu_metrics = PerformanceMetrics(
            base_metrics=SystemMetrics(
                cpu_usage=95.0, memory_usage=70.0, disk_io=40.0, 
                network_io=25.0, timestamp=datetime.now()
            ),
            execution_time=3.0, throughput=80.0, error_rate=0.03
        )
        
        normal_metrics = PerformanceMetrics(
            base_metrics=SystemMetrics(
                cpu_usage=65.0, memory_usage=70.0, disk_io=40.0, 
                network_io=25.0, timestamp=datetime.now()
            ),
            execution_time=2.0, throughput=100.0, error_rate=0.01
        )
        
        # Test filtering
        assert not collector.apply_filters(high_cpu_metrics)  # Should be filtered out
        assert collector.apply_filters(normal_metrics)  # Should pass


class TestStreamingMetricsCollector:
    """Test suite for the StreamingMetricsCollector class."""

    @pytest.fixture
    def streaming_collector(self):
        """Create a StreamingMetricsCollector instance for testing."""
        config = CollectionConfiguration(
            collection_interval=0.1,  # Fast collection for testing
            buffer_size=10
        )
        return StreamingMetricsCollector(config)

    @pytest.mark.asyncio
    async def test_start_streaming(self, streaming_collector):
        """Test starting the streaming collection."""
        collected_metrics = []
        
        def callback(metrics):
            collected_metrics.append(metrics)
        
        streaming_collector.add_callback(callback)
        
        # Start streaming for a short period
        await streaming_collector.start_streaming()
        await asyncio.sleep(0.3)  # Collect for 300ms
        await streaming_collector.stop_streaming()
        
        # Should have collected some metrics
        assert len(collected_metrics) >= 2  # At least 2 collections in 300ms with 100ms interval

    @pytest.mark.asyncio
    async def test_streaming_callbacks(self, streaming_collector):
        """Test callback system for streaming metrics."""
        callback_results = []
        
        async def async_callback(metrics):
            callback_results.append(f"async_{len(callback_results)}")
        
        def sync_callback(metrics):
            callback_results.append(f"sync_{len(callback_results)}")
        
        streaming_collector.add_callback(async_callback)
        streaming_collector.add_callback(sync_callback)
        
        # Simulate metric collection
        mock_metrics = PerformanceMetrics(
            base_metrics=SystemMetrics(
                cpu_usage=50.0, memory_usage=60.0, disk_io=30.0,
                network_io=20.0, timestamp=datetime.now()
            ),
            execution_time=2.0, throughput=100.0, error_rate=0.01
        )
        
        await streaming_collector._notify_callbacks(mock_metrics)
        
        # Both callbacks should have been called
        assert len(callback_results) == 2
        assert any("async" in result for result in callback_results)
        assert any("sync" in result for result in callback_results)

    def test_buffer_management(self, streaming_collector):
        """Test metrics buffer management."""
        # Fill buffer beyond capacity
        for i in range(15):  # Buffer size is 10
            metrics = PerformanceMetrics(
                base_metrics=SystemMetrics(
                    cpu_usage=50.0 + i, memory_usage=60.0, disk_io=30.0,
                    network_io=20.0, timestamp=datetime.now()
                ),
                execution_time=2.0, throughput=100.0, error_rate=0.01
            )
            streaming_collector.add_to_buffer(metrics)
        
        buffer = streaming_collector.get_buffer()
        
        # Buffer should be limited to configured size
        assert len(buffer) <= streaming_collector.config.buffer_size
        
        # Should contain the most recent metrics
        latest_cpu = buffer[-1].base_metrics.cpu_usage
        assert latest_cpu >= 60.0  # From the later iterations

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, streaming_collector):
        """Test error handling in streaming collection."""
        error_count = 0
        
        def error_callback(metrics):
            nonlocal error_count
            error_count += 1
            raise ValueError("Test error")
        
        def success_callback(metrics):
            pass  # This should still work
        
        streaming_collector.add_callback(error_callback)
        streaming_collector.add_callback(success_callback)
        
        # Simulate metric collection with error
        mock_metrics = PerformanceMetrics(
            base_metrics=SystemMetrics(
                cpu_usage=50.0, memory_usage=60.0, disk_io=30.0,
                network_io=20.0, timestamp=datetime.now()
            ),
            execution_time=2.0, throughput=100.0, error_rate=0.01
        )
        
        # Should handle error gracefully
        await streaming_collector._notify_callbacks(mock_metrics)
        
        assert error_count == 1

    def test_metrics_transformation(self, streaming_collector):
        """Test metrics transformation functionality."""
        def cpu_normalizer(metrics):
            # Normalize CPU to 0-1 range
            normalized_metrics = metrics.copy()
            normalized_metrics.base_metrics.cpu_usage = metrics.base_metrics.cpu_usage / 100.0
            return normalized_metrics
        
        streaming_collector.add_transformer(cpu_normalizer)
        
        original_metrics = PerformanceMetrics(
            base_metrics=SystemMetrics(
                cpu_usage=75.0, memory_usage=60.0, disk_io=30.0,
                network_io=20.0, timestamp=datetime.now()
            ),
            execution_time=2.0, throughput=100.0, error_rate=0.01
        )
        
        transformed_metrics = streaming_collector.apply_transformers(original_metrics)
        
        assert transformed_metrics.base_metrics.cpu_usage == 0.75
        assert transformed_metrics.base_metrics.memory_usage == 60.0  # Unchanged


@pytest.mark.integration
class TestCollectorIntegration:
    """Integration tests for metrics collection with other components."""
    
    @pytest.mark.asyncio
    async def test_collector_with_analytics_engine(self):
        """Test integration between collector and analytics engine."""
        from src.analytics.engine import PerformanceAnalyticsEngine
        from src.analytics.models import AnalyticsConfiguration
        
        # Create components
        collector_config = CollectionConfiguration(collection_interval=0.1, buffer_size=5)
        collector = MetricsCollector(collector_config)
        
        engine_config = AnalyticsConfiguration()
        engine = PerformanceAnalyticsEngine(engine_config)
        
        # Collect some metrics
        metrics_list = []
        for i in range(5):
            with patch.object(collector, 'collect_system_metrics') as mock_collect:
                mock_collect.return_value = SystemMetrics(
                    cpu_usage=60.0 + i * 5, memory_usage=70.0, disk_io=40.0,
                    network_io=25.0, timestamp=datetime.now()
                )
                metrics = await collector.collect_metrics_async()
                metrics_list.append(metrics)
        
        # Analyze with engine
        bottlenecks = engine.analyze_bottlenecks(metrics_list)
        anomalies = engine.detect_anomalies(metrics_list)
        
        # Should work without errors
        assert isinstance(bottlenecks, list)
        assert isinstance(anomalies, list)

    def test_real_system_metrics_collection(self):
        """Test actual system metrics collection (integration test)."""
        config = CollectionConfiguration(
            collection_interval=1.0,
            enable_system_metrics=True,
            enable_process_metrics=True
        )
        collector = MetricsCollector(config)
        
        # This test actually calls psutil
        system_metrics = collector.collect_system_metrics()
        
        assert isinstance(system_metrics, SystemMetrics)
        assert 0 <= system_metrics.cpu_usage <= 100
        assert 0 <= system_metrics.memory_usage <= 100
        assert system_metrics.disk_io >= 0
        assert system_metrics.network_io >= 0
        assert isinstance(system_metrics.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])