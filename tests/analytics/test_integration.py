"""
Comprehensive integration tests for the Performance Analytics System.

Tests full end-to-end workflows, integration with existing systems,
and real-world usage scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
import tempfile
import os

from src.analytics.engine import PerformanceAnalyticsEngine
from src.analytics.collector import MetricsCollector, CollectionConfiguration
from src.analytics.optimizer import PerformanceOptimizer, OptimizationStrategy
from src.analytics.predictor import PerformancePredictor
from src.analytics.dashboard import AnalyticsDashboard
from src.analytics.monitoring import RealTimeMonitor
from src.analytics.reporting import PerformanceReporter
from src.analytics.regression import RegressionDetector
from src.analytics.models import (
    PerformanceMetrics, AnalyticsConfiguration, BottleneckAnalysis,
    OptimizationRecommendation
)
from src.core.types import SystemMetrics, ProgressMetrics, ValidationResult


@pytest.mark.integration
class TestAnalyticsSystemIntegration:
    """Integration tests for the complete analytics system."""

    @pytest.fixture
    def analytics_system(self):
        """Create a complete analytics system for testing."""
        config = AnalyticsConfiguration(
            enable_ai_optimization=True,
            enable_predictive_modeling=True,
            anomaly_detection_sensitivity=0.8,
            bottleneck_threshold=0.85,
            enable_real_time_monitoring=True
        )
        
        collector_config = CollectionConfiguration(
            collection_interval=1.0,
            enable_system_metrics=True,
            enable_process_metrics=True,
            buffer_size=100
        )
        
        return {
            'engine': PerformanceAnalyticsEngine(config),
            'collector': MetricsCollector(collector_config),
            'optimizer': PerformanceOptimizer(OptimizationStrategy.BALANCED),
            'predictor': PerformancePredictor(),
            'dashboard': AnalyticsDashboard(),
            'monitor': RealTimeMonitor(),
            'reporter': PerformanceReporter(),
            'regression_detector': RegressionDetector()
        }

    @pytest.fixture
    def historical_performance_data(self):
        """Generate realistic historical performance data."""
        data = []
        base_time = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            for hour in range(24):
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Simulate daily patterns and weekly trends
                hour_factor = 0.5 + 0.5 * abs(12 - hour) / 12  # Peak at noon/midnight
                day_factor = 0.8 + 0.4 * (day / 6)  # Increasing load over week
                
                base_cpu = 40 + 30 * hour_factor * day_factor
                base_memory = 50 + 25 * hour_factor * day_factor
                
                # Add some noise and occasional spikes
                import random
                cpu_noise = random.gauss(0, 5)
                memory_noise = random.gauss(0, 8)
                
                # Occasional performance spikes
                if random.random() < 0.05:  # 5% chance of spike
                    cpu_noise += 20
                    memory_noise += 15
                
                system_metrics = SystemMetrics(
                    cpu_usage=max(0, min(100, base_cpu + cpu_noise)),
                    memory_usage=max(0, min(100, base_memory + memory_noise)),
                    disk_io=30 + random.gauss(0, 10),
                    network_io=20 + random.gauss(0, 8),
                    timestamp=timestamp
                )
                
                perf_metrics = PerformanceMetrics(
                    base_metrics=system_metrics,
                    execution_time=2.0 + random.gauss(0, 0.5),
                    throughput=100 - (base_cpu / 2) + random.gauss(0, 10),
                    error_rate=max(0, 0.01 + random.gauss(0, 0.005)),
                    latency_p95=1.5 + (base_cpu / 50) + random.gauss(0, 0.2),
                    tokens_per_second=200 + random.gauss(0, 30),
                    code_quality_score=0.85 + random.gauss(0, 0.1),
                    test_coverage=0.8 + random.gauss(0, 0.15)
                )
                
                data.append(perf_metrics)
        
        return data

    def test_end_to_end_performance_analysis(self, analytics_system, historical_performance_data):
        """Test complete end-to-end performance analysis workflow."""
        engine = analytics_system['engine']
        optimizer = analytics_system['optimizer']
        predictor = analytics_system['predictor']
        
        # 1. Analyze historical data for bottlenecks
        bottlenecks = engine.analyze_bottlenecks(historical_performance_data)
        
        # 2. Detect anomalies in the data
        anomalies = engine.detect_anomalies(historical_performance_data)
        
        # 3. Generate optimization recommendations
        recommendations = optimizer.generate_recommendations(bottlenecks)
        
        # 4. Predict future performance
        prediction = predictor.predict_performance(historical_performance_data, hours_ahead=24)
        
        # Verify results
        assert isinstance(bottlenecks, list)
        assert isinstance(anomalies, list)
        assert isinstance(recommendations, list)
        assert prediction is not None
        
        # Should detect some patterns in the synthetic data
        if len(bottlenecks) > 0:
            assert any(b.severity > 0.5 for b in bottlenecks)
        
        if len(recommendations) > 0:
            assert all(r.confidence > 0 for r in recommendations)
            assert all(r.estimated_improvement > 0 for r in recommendations)

    @pytest.mark.asyncio
    async def test_real_time_monitoring_workflow(self, analytics_system):
        """Test real-time monitoring and alerting workflow."""
        monitor = analytics_system['monitor']
        engine = analytics_system['engine']
        
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_handler(alert_handler)
        
        # Simulate real-time metrics stream
        async def mock_metrics_stream():
            for i in range(10):
                # Simulate escalating performance issues
                cpu_usage = 50 + i * 5  # Gradually increasing CPU
                
                system_metrics = SystemMetrics(
                    cpu_usage=cpu_usage,
                    memory_usage=60 + i * 2,
                    disk_io=40 + i,
                    network_io=25,
                    timestamp=datetime.now()
                )
                
                perf_metrics = PerformanceMetrics(
                    base_metrics=system_metrics,
                    execution_time=2.0 + i * 0.2,
                    throughput=100 - i * 3,
                    error_rate=0.01 + i * 0.002
                )
                
                yield perf_metrics
                await asyncio.sleep(0.1)
        
        # Process real-time stream
        async for metrics in mock_metrics_stream():
            await monitor.process_metrics(metrics)
            
            # Check for bottlenecks in real-time
            bottlenecks = engine.analyze_bottlenecks([metrics])
            if bottlenecks:
                await monitor.trigger_alerts(bottlenecks)
        
        # Should have triggered some alerts for escalating performance
        assert len(alerts_received) > 0

    def test_dashboard_integration(self, analytics_system, historical_performance_data):
        """Test dashboard integration with analytics components."""
        dashboard = analytics_system['dashboard']
        engine = analytics_system['engine']
        
        # Generate analytics data
        bottlenecks = engine.analyze_bottlenecks(historical_performance_data)
        anomalies = engine.detect_anomalies(historical_performance_data)
        trend_analysis = engine.analyze_trends(historical_performance_data)
        
        # Create dashboard widgets
        metrics_widget = dashboard.create_metrics_widget(historical_performance_data[-24:])  # Last 24 hours
        bottleneck_widget = dashboard.create_bottleneck_widget(bottlenecks)
        trend_widget = dashboard.create_trend_widget(trend_analysis)
        
        # Verify widget creation
        assert metrics_widget is not None
        assert bottleneck_widget is not None
        assert trend_widget is not None
        
        # Generate dashboard HTML
        dashboard_html = dashboard.render_dashboard([metrics_widget, bottleneck_widget, trend_widget])
        assert isinstance(dashboard_html, str)
        assert len(dashboard_html) > 100  # Should be substantial HTML content

    def test_reporting_integration(self, analytics_system, historical_performance_data):
        """Test reporting integration with analytics data."""
        reporter = analytics_system['reporter']
        engine = analytics_system['engine']
        
        # Generate analysis data
        bottlenecks = engine.analyze_bottlenecks(historical_performance_data)
        trend_analysis = engine.analyze_trends(historical_performance_data)
        
        # Generate comprehensive report
        report = reporter.generate_performance_report(
            historical_performance_data,
            bottlenecks=bottlenecks,
            trend_analysis=trend_analysis,
            time_period="7d"
        )
        
        assert report is not None
        assert hasattr(report, 'executive_summary')
        assert hasattr(report, 'detailed_findings')
        assert hasattr(report, 'recommendations')
        
        # Export report in different formats
        json_report = reporter.export_report(report, format="json")
        html_report = reporter.export_report(report, format="html")
        
        assert isinstance(json_report, str)
        assert isinstance(html_report, str)
        
        # JSON should be valid
        json.loads(json_report)  # Should not raise exception

    def test_regression_detection_integration(self, analytics_system, historical_performance_data):
        """Test regression detection with historical data."""
        regression_detector = analytics_system['regression_detector']
        
        # Create baseline from early data
        baseline_data = historical_performance_data[:48]  # First 2 days
        regression_detector.establish_baseline(baseline_data)
        
        # Test recent data for regressions
        recent_data = historical_performance_data[-48:]  # Last 2 days
        regressions = regression_detector.detect_regressions(recent_data)
        
        assert isinstance(regressions, list)
        
        # If regressions detected, should have proper structure
        for regression in regressions:
            assert hasattr(regression, 'metric_name')
            assert hasattr(regression, 'severity')
            assert hasattr(regression, 'timestamp')
            assert hasattr(regression, 'description')

    def test_prediction_accuracy_validation(self, analytics_system, historical_performance_data):
        """Test prediction accuracy by using historical data."""
        predictor = analytics_system['predictor']
        
        # Use first 80% of data to train/predict
        train_size = int(len(historical_performance_data) * 0.8)
        train_data = historical_performance_data[:train_size]
        test_data = historical_performance_data[train_size:]
        
        if len(test_data) > 0:
            # Predict performance for test period
            prediction = predictor.predict_performance(train_data, hours_ahead=len(test_data))
            
            assert prediction is not None
            
            # Compare predictions with actual data (basic validation)
            if hasattr(prediction, 'predicted_metrics'):
                assert len(prediction.predicted_metrics) > 0
                
                # Verify prediction structure
                for pred_metric in prediction.predicted_metrics:
                    assert hasattr(pred_metric, 'timestamp')
                    assert hasattr(pred_metric, 'predicted_values')

    def test_optimization_impact_simulation(self, analytics_system, historical_performance_data):
        """Test optimization impact simulation with realistic data."""
        engine = analytics_system['engine']
        optimizer = analytics_system['optimizer']
        
        # Analyze current performance
        bottlenecks = engine.analyze_bottlenecks(historical_performance_data[-24:])  # Last day
        
        if len(bottlenecks) > 0:
            # Generate optimization recommendations
            recommendations = optimizer.generate_recommendations(bottlenecks)
            
            if len(recommendations) > 0:
                # Simulate impact of top recommendation
                top_recommendation = recommendations[0]
                baseline_metrics = historical_performance_data[-1]
                
                simulated_metrics = optimizer.simulate_impact(baseline_metrics, top_recommendation)
                
                # Should show improvement
                assert simulated_metrics.execution_time <= baseline_metrics.execution_time
                assert simulated_metrics.throughput >= baseline_metrics.throughput

    def test_memory_and_performance_efficiency(self, analytics_system, historical_performance_data):
        """Test system efficiency with large datasets."""
        engine = analytics_system['engine']
        
        # Process large dataset
        import time
        start_time = time.time()
        
        # Analyze large dataset
        bottlenecks = engine.analyze_bottlenecks(historical_performance_data)
        anomalies = engine.detect_anomalies(historical_performance_data)
        trends = engine.analyze_trends(historical_performance_data)
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 30.0  # 30 seconds max for this dataset
        
        # Results should be reasonable
        assert isinstance(bottlenecks, list)
        assert isinstance(anomalies, list)
        assert trends is not None

    def test_concurrent_analysis(self, analytics_system, historical_performance_data):
        """Test concurrent analysis operations."""
        engine = analytics_system['engine']
        
        async def concurrent_analysis():
            # Run multiple analyses concurrently
            tasks = [
                asyncio.create_task(engine.analyze_bottlenecks_async(historical_performance_data)),
                asyncio.create_task(engine.detect_anomalies_async(historical_performance_data)),
                asyncio.create_task(engine.analyze_trends_async(historical_performance_data))
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        # Run concurrent analysis
        results = asyncio.run(concurrent_analysis())
        
        # All tasks should complete successfully
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)

    def test_configuration_impact_on_analysis(self, historical_performance_data):
        """Test how different configurations affect analysis results."""
        # High sensitivity configuration
        high_sens_config = AnalyticsConfiguration(
            anomaly_detection_sensitivity=0.9,
            bottleneck_threshold=0.7,
            enable_ai_optimization=True
        )
        high_sens_engine = PerformanceAnalyticsEngine(high_sens_config)
        
        # Low sensitivity configuration
        low_sens_config = AnalyticsConfiguration(
            anomaly_detection_sensitivity=0.5,
            bottleneck_threshold=0.95,
            enable_ai_optimization=False
        )
        low_sens_engine = PerformanceAnalyticsEngine(low_sens_config)
        
        # Analyze with both configurations
        high_sens_bottlenecks = high_sens_engine.analyze_bottlenecks(historical_performance_data)
        low_sens_bottlenecks = low_sens_engine.analyze_bottlenecks(historical_performance_data)
        
        high_sens_anomalies = high_sens_engine.detect_anomalies(historical_performance_data)
        low_sens_anomalies = low_sens_engine.detect_anomalies(historical_performance_data)
        
        # High sensitivity should generally detect more issues
        assert len(high_sens_bottlenecks) >= len(low_sens_bottlenecks)
        assert len(high_sens_anomalies) >= len(low_sens_anomalies)


@pytest.mark.integration
class TestExistingSystemIntegration:
    """Test integration with existing Claude-TIU systems."""

    def test_progress_validator_integration(self):
        """Test integration with existing ProgressValidator."""
        from src.analytics.engine import PerformanceAnalyticsEngine
        from src.analytics.models import AnalyticsConfiguration
        
        config = AnalyticsConfiguration()
        engine = PerformanceAnalyticsEngine(config)
        
        # Mock existing ValidationResult
        mock_validation = ValidationResult(
            is_valid=True,
            confidence=0.9,
            issues=[],
            timestamp=datetime.now()
        )
        
        # Create performance metrics
        system_metrics = SystemMetrics(
            cpu_usage=75.0,
            memory_usage=60.0,
            disk_io=45.0,
            network_io=30.0,
            timestamp=datetime.now()
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.5,
            throughput=100.0,
            error_rate=0.02
        )
        
        # Should integrate validation results with performance analysis
        analysis = engine.analyze_with_validation([perf_metrics], mock_validation)
        
        assert 'performance_analysis' in analysis
        assert 'validation_results' in analysis
        assert analysis['validation_results']['confidence'] == 0.9

    def test_system_metrics_compatibility(self):
        """Test compatibility with existing SystemMetrics structure."""
        from src.core.types import SystemMetrics
        from src.analytics.models import PerformanceMetrics
        
        # Create SystemMetrics using existing structure
        existing_metrics = SystemMetrics(
            cpu_usage=80.0,
            memory_usage=70.0,
            disk_io=50.0,
            network_io=35.0,
            timestamp=datetime.now()
        )
        
        # Should seamlessly extend to PerformanceMetrics
        enhanced_metrics = PerformanceMetrics(
            base_metrics=existing_metrics,
            execution_time=3.0,
            throughput=90.0,
            error_rate=0.03
        )
        
        # Verify compatibility
        assert enhanced_metrics.base_metrics.cpu_usage == 80.0
        assert enhanced_metrics.execution_time == 3.0
        assert isinstance(enhanced_metrics.base_metrics, SystemMetrics)

    def test_progress_metrics_integration(self):
        """Test integration with existing ProgressMetrics."""
        from src.core.types import ProgressMetrics
        from src.analytics.engine import PerformanceAnalyticsEngine
        from src.analytics.models import AnalyticsConfiguration
        
        config = AnalyticsConfiguration()
        engine = PerformanceAnalyticsEngine(config)
        
        # Create ProgressMetrics
        progress = ProgressMetrics(
            task_id="test_task_123",
            progress_percentage=75.0,
            estimated_completion=datetime.now() + timedelta(minutes=30),
            authenticity_score=0.95,
            validation_passed=True,
            timestamp=datetime.now()
        )
        
        # Analytics should be able to incorporate progress data
        progress_analysis = engine.analyze_progress_metrics([progress])
        
        assert progress_analysis is not None
        assert 'progress_efficiency' in progress_analysis
        assert 'authenticity_assessment' in progress_analysis


@pytest.mark.integration
class TestPerformanceAndScalability:
    """Test performance and scalability of the analytics system."""

    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        from src.analytics.engine import PerformanceAnalyticsEngine
        from src.analytics.models import AnalyticsConfiguration
        
        config = AnalyticsConfiguration()
        engine = PerformanceAnalyticsEngine(config)
        
        # Create large dataset
        large_dataset = []
        for i in range(1000):  # 1000 data points
            system_metrics = SystemMetrics(
                cpu_usage=50 + (i % 50),
                memory_usage=60 + (i % 40),
                disk_io=40 + (i % 30),
                network_io=25 + (i % 20),
                timestamp=datetime.now() + timedelta(minutes=i)
            )
            
            perf_metrics = PerformanceMetrics(
                base_metrics=system_metrics,
                execution_time=2.0 + (i % 10) * 0.1,
                throughput=100 - (i % 50),
                error_rate=0.01 + (i % 100) * 0.0001
            )
            large_dataset.append(perf_metrics)
        
        # Time the processing
        import time
        start_time = time.time()
        
        bottlenecks = engine.analyze_bottlenecks(large_dataset)
        anomalies = engine.detect_anomalies(large_dataset)
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert processing_time < 60.0  # 60 seconds max
        assert isinstance(bottlenecks, list)
        assert isinstance(anomalies, list)

    def test_memory_usage_efficiency(self):
        """Test memory usage efficiency."""
        from src.analytics.collector import MetricsCollector, CollectionConfiguration
        import psutil
        import gc
        
        # Monitor memory before
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create collector and collect metrics
        config = CollectionConfiguration(buffer_size=1000)
        collector = MetricsCollector(config)
        
        # Fill buffer to capacity
        for i in range(1000):
            collector.add_custom_metric(f"metric_{i}", i * 1.5)
        
        # Check memory usage
        after_buffer_memory = process.memory_info().rss
        memory_increase = after_buffer_memory - initial_memory
        
        # Should not use excessive memory (adjust threshold as needed)
        assert memory_increase < 100 * 1024 * 1024  # 100MB max increase
        
        # Clean up
        del collector
        gc.collect()

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations."""
        from src.analytics.engine import PerformanceAnalyticsEngine
        from src.analytics.models import AnalyticsConfiguration
        
        config = AnalyticsConfiguration()
        engines = [PerformanceAnalyticsEngine(config) for _ in range(5)]
        
        # Create test data
        test_metrics = []
        for i in range(100):
            system_metrics = SystemMetrics(
                cpu_usage=50 + i % 50,
                memory_usage=60 + i % 40,
                disk_io=40,
                network_io=25,
                timestamp=datetime.now()
            )
            
            perf_metrics = PerformanceMetrics(
                base_metrics=system_metrics,
                execution_time=2.0,
                throughput=100.0,
                error_rate=0.01
            )
            test_metrics.append(perf_metrics)
        
        # Run concurrent analyses
        async def analyze_concurrently(engine, data):
            return await engine.analyze_bottlenecks_async(data)
        
        tasks = [analyze_concurrently(engine, test_metrics) for engine in engines]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])