"""
Analytics System Usage Examples.

Demonstrates how to use the performance analytics system with
various scenarios and integration patterns.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List
import json

# Import analytics system
from src.analytics.integration import (
    create_analytics_system, quick_analysis, setup_monitoring,
    AnalyticsIntegrationManager
)
from src.analytics.models import AnalyticsConfiguration, PerformanceMetrics
from src.analytics.optimizer import OptimizationStrategy
from src.core.types import SystemMetrics, ProgressMetrics, ValidationResult


def example_basic_analysis():
    """Example 1: Basic performance analysis."""
    print("=== Example 1: Basic Performance Analysis ===")
    
    # Create sample metrics (simulating collected data)
    metrics = []
    for i in range(24):  # 24 hours of data
        system_metrics = SystemMetrics(
            cpu_usage=50 + i * 2,  # Gradually increasing CPU
            memory_usage=60 + i,   # Increasing memory usage
            disk_io=40 + (i % 10) * 2,
            network_io=25 + (i % 5) * 3,
            timestamp=datetime.now() - timedelta(hours=24-i)
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.0 + i * 0.1,
            throughput=100 - i,  # Decreasing throughput
            error_rate=0.01 + i * 0.001,
            latency_p95=1.5 + i * 0.05
        )
        metrics.append(perf_metrics)
    
    # Perform quick analysis
    results = quick_analysis(metrics)
    
    print(f"Analysis completed at: {results['timestamp']}")
    print(f"Metrics analyzed: {results['metrics_analyzed']}")
    print(f"Bottlenecks detected: {len(results['bottlenecks'])}")
    print(f"Anomalies detected: {len(results['anomalies'])}")
    print(f"Overall health score: {results['overall_health_score']:.1f}/100")
    
    if results['bottlenecks']:
        print("\nTop bottleneck:")
        top_bottleneck = results['bottlenecks'][0]
        print(f"  Type: {top_bottleneck.type}")
        print(f"  Severity: {top_bottleneck.severity:.2f}")
        print(f"  Description: {top_bottleneck.description}")
    
    print()


def example_custom_configuration():
    """Example 2: Using custom analytics configuration."""
    print("=== Example 2: Custom Configuration ===")
    
    # Create custom configuration for high-sensitivity monitoring
    config = AnalyticsConfiguration(
        enable_ai_optimization=True,
        anomaly_detection_sensitivity=0.9,  # High sensitivity
        bottleneck_threshold=0.7,           # Lower threshold
        enable_predictive_modeling=True,
        enable_real_time_monitoring=True,
        collection_interval=0.5,            # Faster collection
        buffer_size=200
    )
    
    # Create analytics system with custom config
    analytics = create_analytics_system(config)
    
    # Check system status
    status = analytics.get_system_status()
    print(f"System initialized: {datetime.now()}")
    print(f"Monitoring active: {status['monitoring_active']}")
    print(f"Configuration:")
    for key, value in status['configuration'].items():
        print(f"  {key}: {value}")
    
    print()


def example_optimization_workflow():
    """Example 3: Complete optimization workflow."""
    print("=== Example 3: Optimization Workflow ===")
    
    # Create analytics system
    analytics = create_analytics_system()
    
    # Simulate performance problem
    problematic_metrics = []
    for i in range(10):
        system_metrics = SystemMetrics(
            cpu_usage=90 + i,      # High CPU usage
            memory_usage=85 + i,   # High memory usage
            disk_io=80,
            network_io=30,
            timestamp=datetime.now() - timedelta(hours=10-i)
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=5.0 + i * 0.2,  # Increasing execution time
            throughput=50 - i,             # Decreasing throughput
            error_rate=0.05 + i * 0.001,   # Increasing errors
            latency_p95=3.0 + i * 0.1
        )
        problematic_metrics.append(perf_metrics)
    
    # Get optimization plan
    optimization_plan = analytics.get_optimization_plan(
        problematic_metrics, 
        strategy=OptimizationStrategy.BALANCED
    )
    
    print(f"Optimization analysis completed: {optimization_plan['timestamp']}")
    print(f"Bottlenecks detected: {optimization_plan['bottlenecks_detected']}")
    print(f"Estimated improvement: {optimization_plan['estimated_improvement']:.1f}%")
    
    # Display recommendations
    recommendations = optimization_plan['recommendations']
    print(f"\nTop 3 Optimization Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec.description}")
        print(f"   Confidence: {rec.confidence:.2f}")
        print(f"   Estimated improvement: {rec.estimated_improvement:.1f}%")
        print(f"   Priority: {rec.priority}")
        print(f"   Risk level: {rec.risk_level}")
    
    # Display safety validation
    print(f"\nSafety Validation Results:")
    safety_results = optimization_plan['safety_validation']
    safe_count = sum(1 for result in safety_results if result['is_safe'])
    print(f"Safe recommendations: {safe_count}/{len(safety_results)}")
    
    print()


def example_dashboard_generation():
    """Example 4: Dashboard generation."""
    print("=== Example 4: Dashboard Generation ===")
    
    # Create analytics system
    analytics = create_analytics_system()
    
    # Generate sample metrics
    dashboard_metrics = []
    for i in range(48):  # 2 days of hourly data
        system_metrics = SystemMetrics(
            cpu_usage=40 + 30 * abs((i % 24) - 12) / 12,  # Daily pattern
            memory_usage=50 + 20 * (i / 48) + (i % 12) * 2,  # Trend + pattern
            disk_io=30 + (i % 8) * 5,
            network_io=20 + (i % 6) * 4,
            timestamp=datetime.now() - timedelta(hours=48-i)
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.0 + (i % 10) * 0.2,
            throughput=90 + 10 * (1 - i / 48),
            error_rate=0.01 + (i % 20) * 0.001
        )
        dashboard_metrics.append(perf_metrics)
    
    # Generate dashboard
    dashboard_html = analytics.generate_dashboard(dashboard_metrics, theme="dark")
    
    # Save dashboard to file
    with open("/tmp/analytics_dashboard.html", "w") as f:
        f.write(dashboard_html)
    
    print("Dashboard generated and saved to /tmp/analytics_dashboard.html")
    print(f"Dashboard size: {len(dashboard_html)} characters")
    
    # Generate report
    report_json = analytics.generate_report(dashboard_metrics, time_period="2d", format="json")
    
    # Save report
    with open("/tmp/performance_report.json", "w") as f:
        f.write(report_json)
    
    print("Performance report saved to /tmp/performance_report.json")
    
    print()


async def example_real_time_monitoring():
    """Example 5: Real-time monitoring with alerts."""
    print("=== Example 5: Real-Time Monitoring ===")
    
    # Alert handler
    def handle_alert(alert):
        print(f"🚨 ALERT: {alert.description}")
        print(f"   Severity: {alert.severity}")
        print(f"   Timestamp: {alert.timestamp}")
        print(f"   Affected metrics: {', '.join(alert.affected_metrics)}")
    
    # Set up monitoring
    config = AnalyticsConfiguration(
        enable_real_time_monitoring=True,
        collection_interval=1.0,
        anomaly_detection_sensitivity=0.8
    )
    analytics = setup_monitoring(handle_alert, config)
    
    print("Real-time monitoring started...")
    print("Simulating performance scenarios...")
    
    # Simulate various performance scenarios
    scenarios = [
        {"name": "Normal operation", "cpu": 50, "memory": 60, "duration": 3},
        {"name": "CPU spike", "cpu": 95, "memory": 60, "duration": 2},
        {"name": "Memory pressure", "cpu": 60, "memory": 90, "duration": 2},
        {"name": "Recovery", "cpu": 45, "memory": 55, "duration": 2}
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        for i in range(scenario['duration']):
            # Create metrics for current scenario
            system_metrics = SystemMetrics(
                cpu_usage=scenario['cpu'] + (i * 2),
                memory_usage=scenario['memory'] + (i * 1),
                disk_io=40,
                network_io=25,
                timestamp=datetime.now()
            )
            
            perf_metrics = PerformanceMetrics(
                base_metrics=system_metrics,
                execution_time=2.0 + (scenario['cpu'] / 50),
                throughput=100 - (scenario['cpu'] / 2),
                error_rate=0.01 * (scenario['cpu'] / 50)
            )
            
            # Process through monitoring system
            await analytics.monitor.process_metrics(perf_metrics)
            await asyncio.sleep(1)
    
    # Stop monitoring
    analytics.stop_monitoring()
    print("\nReal-time monitoring stopped.")
    print()


def example_regression_detection():
    """Example 6: Performance regression detection."""
    print("=== Example 6: Regression Detection ===")
    
    # Create analytics system
    analytics = create_analytics_system()
    
    # Generate baseline metrics (good performance)
    baseline_metrics = []
    for i in range(72):  # 3 days of baseline
        system_metrics = SystemMetrics(
            cpu_usage=45 + (i % 12) * 2,  # Stable with small variations
            memory_usage=55 + (i % 8) * 1,
            disk_io=35 + (i % 6) * 2,
            network_io=25 + (i % 4) * 1,
            timestamp=datetime.now() - timedelta(hours=96-i)
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.0 + (i % 10) * 0.05,
            throughput=105 - (i % 8) * 2,
            error_rate=0.005 + (i % 20) * 0.0001
        )
        baseline_metrics.append(perf_metrics)
    
    # Establish baseline
    success = analytics.establish_baseline(baseline_metrics)
    print(f"Baseline established: {success}")
    
    # Generate recent metrics with regression
    recent_metrics = []
    for i in range(24):  # 1 day of recent data with problems
        degradation_factor = 1 + (i / 24) * 0.5  # Gradual degradation
        
        system_metrics = SystemMetrics(
            cpu_usage=60 + i * 2 * degradation_factor,  # Increasing CPU
            memory_usage=70 + i * 1 * degradation_factor,  # Increasing memory
            disk_io=50 + i * 1.5,
            network_io=35 + i * 0.5,
            timestamp=datetime.now() - timedelta(hours=24-i)
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=3.0 + i * 0.1 * degradation_factor,
            throughput=80 - i * 2 * degradation_factor,
            error_rate=0.02 + i * 0.001 * degradation_factor
        )
        recent_metrics.append(perf_metrics)
    
    # Detect regressions
    regressions = analytics.detect_regressions(recent_metrics)
    
    print(f"Regressions detected: {len(regressions)}")
    for regression in regressions:
        print(f"  Metric: {regression['metric_name']}")
        print(f"  Severity: {regression['severity']:.2f}")
        print(f"  Description: {regression['description']}")
        print(f"  Timestamp: {regression['timestamp']}")
        print()


def example_integration_with_existing_systems():
    """Example 7: Integration with existing Claude-TIU systems."""
    print("=== Example 7: Integration with Existing Systems ===")
    
    # Create analytics system
    analytics = create_analytics_system()
    
    # Simulate existing system metrics
    system_metrics = SystemMetrics(
        cpu_usage=75.0,
        memory_usage=68.0,
        disk_io=45.0,
        network_io=32.0,
        timestamp=datetime.now()
    )
    
    # Simulate validation result from existing ProgressValidator
    validation_result = ValidationResult(
        is_valid=False,  # Some validation issues
        confidence=0.7,
        issues=[
            # Simulated issues
        ],
        timestamp=datetime.now()
    )
    
    # Create performance metrics from system metrics
    perf_metrics = [PerformanceMetrics(
        base_metrics=system_metrics,
        execution_time=3.2,
        throughput=85.0,
        error_rate=0.03,
        code_quality_score=0.75,
        test_coverage=0.82
    )]
    
    # Integrate analytics with validation
    integrated_analysis = analytics.integrate_with_progress_validator(
        perf_metrics, validation_result
    )
    
    print("Integrated Analysis Results:")
    print(f"Validation confidence: {integrated_analysis['validation_results']['confidence']}")
    print(f"Performance health score: {integrated_analysis['performance_analysis']['overall_health_score']:.1f}")
    
    # Analyze correlations
    correlations = integrated_analysis['correlation_insights']
    print(f"Validation-Performance correlation: {correlations['validation_confidence_vs_performance']}")
    print(f"Issue impact analysis: {correlations['issues_impact_on_performance']}")
    
    print()


def example_ai_learning_integration():
    """Example 8: AI learning system integration."""
    print("=== Example 8: AI Learning System Integration ===")
    
    # Create analytics system with AI optimization enabled
    config = AnalyticsConfiguration(
        enable_ai_optimization=True,
        enable_predictive_modeling=True,
        anomaly_detection_sensitivity=0.85
    )
    analytics = create_analytics_system(config)
    
    # Simulate AI training metrics
    training_metrics = []
    for epoch in range(50):
        # Simulate training progress
        loss = 1.0 * (1 - epoch / 50) + 0.1  # Decreasing loss
        accuracy = 0.5 + 0.4 * (epoch / 50)  # Increasing accuracy
        
        system_metrics = SystemMetrics(
            cpu_usage=70 + (epoch % 10) * 2,
            memory_usage=80 + epoch * 0.2,  # Gradually increasing
            disk_io=60,
            network_io=40,
            timestamp=datetime.now() - timedelta(minutes=50-epoch)
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=5.0 - epoch * 0.05,  # Improving over time
            throughput=60 + epoch * 0.5,
            error_rate=loss / 10,
            tokens_per_second=100 + epoch * 2,
            model_accuracy=accuracy,
            training_loss=loss
        )
        training_metrics.append(perf_metrics)
    
    # Analyze AI training performance
    analysis = analytics.analyze_performance(training_metrics)
    
    print("AI Training Performance Analysis:")
    print(f"Training epochs analyzed: {len(training_metrics)}")
    print(f"Performance trend: {analysis['trends'].performance_trend if analysis.get('trends') else 'N/A'}")
    print(f"Bottlenecks during training: {len(analysis['bottlenecks'])}")
    
    if analysis.get('predictions'):
        print("Predicted performance improvements:")
        prediction = analysis['predictions']
        print(f"  Confidence: {prediction.confidence:.2f}")
        print(f"  Trend: {prediction.trend}")
    
    print()


async def run_all_examples():
    """Run all examples in sequence."""
    print("🚀 Analytics System Usage Examples")
    print("=" * 50)
    
    # Run synchronous examples
    example_basic_analysis()
    example_custom_configuration()
    example_optimization_workflow()
    example_dashboard_generation()
    example_regression_detection()
    example_integration_with_existing_systems()
    example_ai_learning_integration()
    
    # Run asynchronous example
    await example_real_time_monitoring()
    
    print("✅ All examples completed successfully!")
    print("\nGenerated files:")
    print("  - /tmp/analytics_dashboard.html")
    print("  - /tmp/performance_report.json")


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())