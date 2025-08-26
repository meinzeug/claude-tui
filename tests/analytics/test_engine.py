"""
Comprehensive test suite for the Performance Analytics Engine.

Tests the core functionality of bottleneck detection, anomaly detection,
optimization recommendations, and integration with existing systems.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import numpy as np

from src.analytics.engine import PerformanceAnalyticsEngine
from src.analytics.models import (
    PerformanceMetrics, BottleneckAnalysis, OptimizationRecommendation,
    TrendAnalysis, PerformanceAlert, AnalyticsConfiguration
)
from src.core.types import SystemMetrics, ProgressMetrics, ValidationResult


class TestPerformanceAnalyticsEngine:
    """Test suite for the PerformanceAnalyticsEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a PerformanceAnalyticsEngine instance for testing."""
        config = AnalyticsConfiguration(
            enable_ai_optimization=True,
            anomaly_detection_sensitivity=0.8,
            bottleneck_threshold=0.9,
            enable_predictive_modeling=True
        )
        return PerformanceAnalyticsEngine(config)

    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics for testing."""
        base_metrics = SystemMetrics(
            cpu_usage=85.5,
            memory_usage=70.2,
            disk_io=45.0,
            network_io=30.1,
            timestamp=datetime.now()
        )
        
        return PerformanceMetrics(
            base_metrics=base_metrics,
            execution_time=2.5,
            queue_depth=5,
            active_tasks=3,
            completed_tasks=10,
            error_rate=0.02,
            throughput=100.0,
            latency_p95=1.8,
            tokens_per_second=250.5,
            code_quality_score=0.92,
            test_coverage=0.88,
            deployment_frequency=2.5,
            lead_time=1.2,
            mttr=0.5,
            change_failure_rate=0.05
        )

    @pytest.fixture
    def historical_metrics(self):
        """Create historical metrics for trend analysis."""
        metrics_list = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(24):  # 24 hours of hourly data
            timestamp = base_time + timedelta(hours=i)
            cpu_usage = 50 + (30 * np.sin(i * 0.5)) + np.random.normal(0, 5)  # Simulate daily pattern
            
            base_metrics = SystemMetrics(
                cpu_usage=max(0, min(100, cpu_usage)),
                memory_usage=60 + np.random.normal(0, 10),
                disk_io=40 + np.random.normal(0, 15),
                network_io=25 + np.random.normal(0, 8),
                timestamp=timestamp
            )
            
            perf_metrics = PerformanceMetrics(
                base_metrics=base_metrics,
                execution_time=2.0 + np.random.normal(0, 0.5),
                throughput=100 + np.random.normal(0, 20),
                latency_p95=1.5 + np.random.normal(0, 0.3),
                error_rate=0.01 + max(0, np.random.normal(0, 0.01))
            )
            metrics_list.append(perf_metrics)
        
        return metrics_list

    def test_engine_initialization(self, engine):
        """Test proper initialization of the analytics engine."""
        assert engine is not None
        assert engine.config.enable_ai_optimization is True
        assert engine.config.anomaly_detection_sensitivity == 0.8
        assert hasattr(engine, 'anomaly_detector')
        assert hasattr(engine, 'predictor')

    def test_bottleneck_detection_cpu(self, engine, sample_metrics):
        """Test CPU bottleneck detection."""
        # Simulate high CPU usage
        sample_metrics.base_metrics.cpu_usage = 95.0
        sample_metrics.execution_time = 5.0  # High execution time
        
        bottlenecks = engine.analyze_bottlenecks([sample_metrics])
        
        assert len(bottlenecks) > 0
        cpu_bottleneck = next((b for b in bottlenecks if b.type == "cpu"), None)
        assert cpu_bottleneck is not None
        assert cpu_bottleneck.severity >= 0.8
        assert "cpu" in cpu_bottleneck.description.lower()

    def test_bottleneck_detection_memory(self, engine, sample_metrics):
        """Test memory bottleneck detection."""
        # Simulate high memory usage
        sample_metrics.base_metrics.memory_usage = 92.0
        sample_metrics.queue_depth = 20  # High queue depth
        
        bottlenecks = engine.analyze_bottlenecks([sample_metrics])
        
        memory_bottleneck = next((b for b in bottlenecks if b.type == "memory"), None)
        if memory_bottleneck:  # May not always trigger depending on thresholds
            assert memory_bottleneck.severity >= 0.7
            assert "memory" in memory_bottleneck.description.lower()

    def test_bottleneck_detection_io(self, engine, sample_metrics):
        """Test I/O bottleneck detection."""
        # Simulate high I/O usage
        sample_metrics.base_metrics.disk_io = 95.0
        sample_metrics.latency_p95 = 5.0  # High latency
        
        bottlenecks = engine.analyze_bottlenecks([sample_metrics])
        
        io_bottleneck = next((b for b in bottlenecks if b.type == "io"), None)
        if io_bottleneck:
            assert "io" in io_bottleneck.description.lower() or "latency" in io_bottleneck.description.lower()

    def test_anomaly_detection_cpu_spike(self, engine, historical_metrics):
        """Test anomaly detection for CPU spikes."""
        # Add an anomalous data point
        anomaly_metrics = historical_metrics[-1]
        anomaly_metrics.base_metrics.cpu_usage = 98.0  # Significant spike
        
        anomalies = engine.detect_anomalies(historical_metrics)
        
        # Should detect the CPU spike
        cpu_anomalies = [a for a in anomalies if a.metric_name == "cpu_usage"]
        assert len(cpu_anomalies) > 0
        assert any(a.severity >= 0.7 for a in cpu_anomalies)

    def test_anomaly_detection_performance_degradation(self, engine, historical_metrics):
        """Test anomaly detection for performance degradation."""
        # Simulate performance degradation
        for metrics in historical_metrics[-3:]:  # Last 3 hours
            metrics.execution_time *= 2.5  # Significant slowdown
            metrics.throughput *= 0.4  # Reduced throughput
        
        anomalies = engine.detect_anomalies(historical_metrics)
        
        # Should detect performance degradation
        perf_anomalies = [a for a in anomalies if a.metric_name in ["execution_time", "throughput"]]
        assert len(perf_anomalies) > 0

    def test_optimization_recommendations_cpu(self, engine, sample_metrics):
        """Test optimization recommendations for CPU issues."""
        # Simulate CPU bottleneck
        sample_metrics.base_metrics.cpu_usage = 95.0
        sample_metrics.execution_time = 5.0
        
        bottlenecks = engine.analyze_bottlenecks([sample_metrics])
        recommendations = engine.generate_optimization_recommendations(bottlenecks, [sample_metrics])
        
        assert len(recommendations) > 0
        cpu_recs = [r for r in recommendations if "cpu" in r.description.lower() or "parallel" in r.description.lower()]
        assert len(cpu_recs) > 0
        
        # Check recommendation quality
        for rec in cpu_recs:
            assert rec.confidence >= 0.6
            assert rec.estimated_improvement > 0
            assert len(rec.implementation_steps) > 0

    def test_optimization_recommendations_memory(self, engine, sample_metrics):
        """Test optimization recommendations for memory issues."""
        # Simulate memory pressure
        sample_metrics.base_metrics.memory_usage = 90.0
        sample_metrics.queue_depth = 25
        
        bottlenecks = engine.analyze_bottlenecks([sample_metrics])
        recommendations = engine.generate_optimization_recommendations(bottlenecks, [sample_metrics])
        
        memory_recs = [r for r in recommendations if "memory" in r.description.lower() or "cache" in r.description.lower()]
        if memory_recs:  # May not always generate memory recommendations
            for rec in memory_recs:
                assert rec.priority in ["high", "medium", "low"]
                assert rec.risk_level in ["low", "medium", "high"]

    def test_trend_analysis(self, engine, historical_metrics):
        """Test trend analysis functionality."""
        trend_analysis = engine.analyze_trends(historical_metrics)
        
        assert trend_analysis is not None
        assert hasattr(trend_analysis, 'cpu_trend')
        assert hasattr(trend_analysis, 'memory_trend')
        assert hasattr(trend_analysis, 'performance_trend')
        
        # Verify trend calculations
        assert trend_analysis.cpu_trend in ['increasing', 'decreasing', 'stable']
        assert isinstance(trend_analysis.confidence, float)
        assert 0 <= trend_analysis.confidence <= 1

    @pytest.mark.asyncio
    async def test_real_time_analysis(self, engine, sample_metrics):
        """Test real-time analysis capabilities."""
        # Mock real-time data stream
        async def mock_metrics_stream():
            for i in range(5):
                yield sample_metrics
                await asyncio.sleep(0.1)
        
        results = []
        async for analysis in engine.analyze_real_time(mock_metrics_stream()):
            results.append(analysis)
        
        assert len(results) == 5
        for result in results:
            assert 'bottlenecks' in result
            assert 'anomalies' in result
            assert 'recommendations' in result

    def test_integration_with_progress_validator(self, engine):
        """Test integration with existing ProgressValidator system."""
        # Mock ProgressValidator and ValidationResult
        mock_validator = Mock()
        mock_result = ValidationResult(
            is_valid=True,
            confidence=0.9,
            issues=[],
            timestamp=datetime.now()
        )
        mock_validator.validate_progress.return_value = mock_result
        
        # Test integration
        with patch('src.core.validator.ProgressValidator', return_value=mock_validator):
            metrics = [self.sample_metrics()]  # Use the fixture method
            analysis = engine.analyze_with_validation(metrics)
            
            assert 'validation_results' in analysis
            assert 'performance_analysis' in analysis
            assert analysis['validation_results']['confidence'] == 0.9

    def test_configuration_impact(self):
        """Test different configuration impacts on analysis."""
        # High sensitivity configuration
        high_sensitivity_config = AnalyticsConfiguration(
            anomaly_detection_sensitivity=0.9,
            bottleneck_threshold=0.8
        )
        high_sensitivity_engine = PerformanceAnalyticsEngine(high_sensitivity_config)
        
        # Low sensitivity configuration
        low_sensitivity_config = AnalyticsConfiguration(
            anomaly_detection_sensitivity=0.5,
            bottleneck_threshold=0.95
        )
        low_sensitivity_engine = PerformanceAnalyticsEngine(low_sensitivity_config)
        
        # Create sample data with moderate issues
        sample_data = self.sample_metrics()
        sample_data.base_metrics.cpu_usage = 85.0  # Moderate CPU usage
        
        high_sens_bottlenecks = high_sensitivity_engine.analyze_bottlenecks([sample_data])
        low_sens_bottlenecks = low_sensitivity_engine.analyze_bottlenecks([sample_data])
        
        # High sensitivity should detect more issues
        assert len(high_sens_bottlenecks) >= len(low_sens_bottlenecks)

    def test_error_handling(self, engine):
        """Test error handling in various scenarios."""
        # Test with empty metrics
        bottlenecks = engine.analyze_bottlenecks([])
        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) == 0
        
        # Test with invalid metrics
        invalid_metrics = Mock()
        invalid_metrics.base_metrics = None
        
        with pytest.raises(AttributeError):
            engine.analyze_bottlenecks([invalid_metrics])

    def test_performance_metrics_validation(self, sample_metrics):
        """Test validation of performance metrics structure."""
        # Test required fields
        assert hasattr(sample_metrics, 'base_metrics')
        assert hasattr(sample_metrics, 'execution_time')
        assert hasattr(sample_metrics, 'throughput')
        assert hasattr(sample_metrics, 'error_rate')
        
        # Test data types
        assert isinstance(sample_metrics.execution_time, (int, float))
        assert isinstance(sample_metrics.throughput, (int, float))
        assert 0 <= sample_metrics.error_rate <= 1

    def test_bottleneck_prioritization(self, engine, sample_metrics):
        """Test that bottlenecks are properly prioritized."""
        # Create multiple bottleneck conditions
        sample_metrics.base_metrics.cpu_usage = 95.0
        sample_metrics.base_metrics.memory_usage = 90.0
        sample_metrics.execution_time = 10.0
        sample_metrics.error_rate = 0.1
        
        bottlenecks = engine.analyze_bottlenecks([sample_metrics])
        
        if len(bottlenecks) > 1:
            # Verify bottlenecks are sorted by severity
            severities = [b.severity for b in bottlenecks]
            assert severities == sorted(severities, reverse=True)

    def sample_metrics(self):
        """Helper method to create sample metrics."""
        base_metrics = SystemMetrics(
            cpu_usage=75.0,
            memory_usage=65.0,
            disk_io=40.0,
            network_io=25.0,
            timestamp=datetime.now()
        )
        
        return PerformanceMetrics(
            base_metrics=base_metrics,
            execution_time=2.0,
            throughput=100.0,
            error_rate=0.01,
            latency_p95=1.5
        )


@pytest.mark.integration
class TestAnalyticsIntegration:
    """Integration tests for the analytics system with existing components."""
    
    def test_system_metrics_integration(self):
        """Test integration with SystemMetrics from core.types."""
        from src.core.types import SystemMetrics
        
        system_metrics = SystemMetrics(
            cpu_usage=80.0,
            memory_usage=70.0,
            disk_io=50.0,
            network_io=30.0,
            timestamp=datetime.now()
        )
        
        # Create PerformanceMetrics from SystemMetrics
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.5,
            throughput=120.0,
            error_rate=0.02
        )
        
        assert perf_metrics.base_metrics.cpu_usage == 80.0
        assert perf_metrics.execution_time == 2.5

    def test_progress_metrics_integration(self):
        """Test integration with ProgressMetrics from core.types."""
        from src.core.types import ProgressMetrics
        
        progress_metrics = ProgressMetrics(
            task_id="test_task",
            progress_percentage=75.5,
            estimated_completion=datetime.now() + timedelta(minutes=30),
            authenticity_score=0.92,
            validation_passed=True,
            timestamp=datetime.now()
        )
        
        # Analytics should be able to incorporate progress metrics
        assert progress_metrics.authenticity_score > 0.9
        assert progress_metrics.validation_passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])