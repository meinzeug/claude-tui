"""
Analytics System Integration Layer.

Provides high-level integration with existing Claude-TIU systems
and simplified APIs for common analytics operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging

from .engine import PerformanceAnalyticsEngine
from .collector import MetricsCollector, CollectionConfiguration
from .optimizer import PerformanceOptimizer, OptimizationStrategy
from .predictor import PerformancePredictor
from .dashboard import AnalyticsDashboard
from .monitoring import RealTimeMonitor
from .reporting import PerformanceReporter
from .regression import RegressionDetector
from .models import (
    PerformanceMetrics, AnalyticsConfiguration, BottleneckAnalysis,
    OptimizationRecommendation, TrendAnalysis, PerformanceAlert
)
from ..core.types import SystemMetrics, ProgressMetrics, ValidationResult

logger = logging.getLogger(__name__)


class AnalyticsIntegrationManager:
    """
    High-level manager for analytics system integration.
    
    Provides simplified APIs for common analytics operations and
    manages integration with existing Claude-TIU systems.
    """
    
    def __init__(self, config: Optional[AnalyticsConfiguration] = None):
        """Initialize the analytics integration manager."""
        self.config = config or AnalyticsConfiguration()
        
        # Initialize core components
        self.engine = PerformanceAnalyticsEngine(self.config)
        
        collector_config = CollectionConfiguration(
            collection_interval=self.config.collection_interval,
            enable_system_metrics=True,
            enable_process_metrics=True,
            buffer_size=self.config.buffer_size
        )
        self.collector = MetricsCollector(collector_config)
        
        self.optimizer = PerformanceOptimizer(OptimizationStrategy.BALANCED)
        self.predictor = PerformancePredictor()
        self.dashboard = AnalyticsDashboard()
        self.monitor = RealTimeMonitor()
        self.reporter = PerformanceReporter()
        self.regression_detector = RegressionDetector()
        
        # Integration state
        self._monitoring_active = False
        self._baseline_established = False
        
        logger.info("Analytics integration manager initialized")
    
    def initialize_system(self) -> bool:
        """Initialize the complete analytics system."""
        try:
            # Verify all components are ready
            components = [
                self.engine, self.collector, self.optimizer,
                self.predictor, self.dashboard, self.monitor,
                self.reporter, self.regression_detector
            ]
            
            for component in components:
                if hasattr(component, 'initialize'):
                    component.initialize()
            
            logger.info("Analytics system fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics system: {e}")
            return False
    
    def start_monitoring(self, alert_callback: Optional[callable] = None) -> bool:
        """Start real-time performance monitoring."""
        try:
            if alert_callback:
                self.monitor.add_alert_handler(alert_callback)
            
            # Start background monitoring
            asyncio.create_task(self._monitoring_loop())
            self._monitoring_active = True
            
            logger.info("Real-time monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop real-time performance monitoring."""
        try:
            self._monitoring_active = False
            logger.info("Real-time monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect current metrics
                metrics = await self.collector.collect_metrics_async()
                
                # Process metrics through analytics pipeline
                await self.monitor.process_metrics(metrics)
                
                # Check for bottlenecks and anomalies
                bottlenecks = self.engine.analyze_bottlenecks([metrics])
                anomalies = self.engine.detect_anomalies([metrics])
                
                # Trigger alerts if needed
                if bottlenecks:
                    await self.monitor.trigger_alerts(bottlenecks)
                
                if anomalies:
                    await self.monitor.trigger_anomaly_alerts(anomalies)
                
                # Wait for next collection interval
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    def analyze_performance(
        self,
        metrics: Union[List[PerformanceMetrics], List[SystemMetrics]],
        include_predictions: bool = True,
        include_optimizations: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive performance analysis.
        
        Args:
            metrics: Performance or system metrics to analyze
            include_predictions: Whether to include future performance predictions
            include_optimizations: Whether to include optimization recommendations
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Convert SystemMetrics to PerformanceMetrics if needed
            perf_metrics = self._ensure_performance_metrics(metrics)
            
            # Core analysis
            bottlenecks = self.engine.analyze_bottlenecks(perf_metrics)
            anomalies = self.engine.detect_anomalies(perf_metrics)
            trends = self.engine.analyze_trends(perf_metrics)
            
            results = {
                'timestamp': datetime.now(),
                'metrics_analyzed': len(perf_metrics),
                'bottlenecks': bottlenecks,
                'anomalies': anomalies,
                'trends': trends,
                'overall_health_score': self._calculate_health_score(perf_metrics, bottlenecks, anomalies)
            }
            
            # Add predictions if requested
            if include_predictions and len(perf_metrics) >= 5:
                prediction = self.predictor.predict_performance(perf_metrics, hours_ahead=24)
                results['predictions'] = prediction
            
            # Add optimization recommendations if requested
            if include_optimizations and bottlenecks:
                recommendations = self.optimizer.generate_recommendations(bottlenecks)
                results['optimization_recommendations'] = recommendations
            
            logger.info(f"Performance analysis completed: {len(bottlenecks)} bottlenecks, {len(anomalies)} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def establish_baseline(self, historical_metrics: List[PerformanceMetrics]) -> bool:
        """Establish performance baseline from historical data."""
        try:
            # Set baseline in regression detector
            self.regression_detector.establish_baseline(historical_metrics)
            
            # Train predictor with historical data
            self.predictor.train_model(historical_metrics)
            
            self._baseline_established = True
            logger.info(f"Performance baseline established with {len(historical_metrics)} data points")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish baseline: {e}")
            return False
    
    def detect_regressions(self, recent_metrics: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """Detect performance regressions against established baseline."""
        if not self._baseline_established:
            logger.warning("No baseline established for regression detection")
            return []
        
        try:
            regressions = self.regression_detector.detect_regressions(recent_metrics)
            
            logger.info(f"Regression detection completed: {len(regressions)} regressions found")
            return regressions
            
        except Exception as e:
            logger.error(f"Regression detection failed: {e}")
            return []
    
    def generate_dashboard(
        self,
        metrics: List[PerformanceMetrics],
        theme: str = "default"
    ) -> str:
        """Generate interactive analytics dashboard."""
        try:
            # Analyze data for dashboard
            analysis = self.analyze_performance(metrics, include_predictions=False)
            
            # Create dashboard widgets
            widgets = []
            
            # Metrics overview widget
            widgets.append(self.dashboard.create_metrics_widget(metrics[-24:] if len(metrics) > 24 else metrics))
            
            # Bottlenecks widget
            if 'bottlenecks' in analysis and analysis['bottlenecks']:
                widgets.append(self.dashboard.create_bottleneck_widget(analysis['bottlenecks']))
            
            # Trends widget
            if 'trends' in analysis and analysis['trends']:
                widgets.append(self.dashboard.create_trend_widget(analysis['trends']))
            
            # Alerts widget
            recent_alerts = self.monitor.get_recent_alerts(hours=24)
            if recent_alerts:
                widgets.append(self.dashboard.create_alerts_widget(recent_alerts))
            
            # Render dashboard
            dashboard_html = self.dashboard.render_dashboard(widgets, theme=theme)
            
            logger.info("Dashboard generated successfully")
            return dashboard_html
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return f"<html><body><h1>Dashboard Generation Error</h1><p>{e}</p></body></html>"
    
    def generate_report(
        self,
        metrics: List[PerformanceMetrics],
        time_period: str = "7d",
        format: str = "json"
    ) -> str:
        """Generate comprehensive performance report."""
        try:
            # Analyze metrics
            analysis = self.analyze_performance(metrics)
            
            # Generate report
            report = self.reporter.generate_performance_report(
                metrics,
                bottlenecks=analysis.get('bottlenecks', []),
                trend_analysis=analysis.get('trends'),
                time_period=time_period
            )
            
            # Export in requested format
            report_content = self.reporter.export_report(report, format=format)
            
            logger.info(f"Performance report generated in {format} format")
            return report_content
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f'{{"error": "{e}", "timestamp": "{datetime.now().isoformat()}"}}'
    
    def get_optimization_plan(
        self,
        metrics: List[PerformanceMetrics],
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> Dict[str, Any]:
        """Get comprehensive optimization plan."""
        try:
            # Set optimizer strategy
            self.optimizer = PerformanceOptimizer(strategy)
            
            # Analyze bottlenecks
            bottlenecks = self.engine.analyze_bottlenecks(metrics)
            
            if not bottlenecks:
                return {
                    'message': 'No significant bottlenecks detected',
                    'health_score': self._calculate_health_score(metrics, [], [])
                }
            
            # Generate recommendations
            recommendations = self.optimizer.generate_recommendations(bottlenecks)
            
            # Create optimization plan
            plan = self.optimizer.create_optimization_plan(recommendations)
            
            # Validate safety
            safety_results = []
            for rec in recommendations:
                is_safe, safety_score, issues = self.optimizer.safety_validator.validate_recommendation(rec)
                safety_results.append({
                    'recommendation_id': rec.id if hasattr(rec, 'id') else None,
                    'is_safe': is_safe,
                    'safety_score': safety_score,
                    'issues': [{'description': issue.description, 'severity': issue.severity} for issue in issues]
                })
            
            return {
                'timestamp': datetime.now(),
                'bottlenecks_detected': len(bottlenecks),
                'recommendations': recommendations,
                'optimization_plan': plan,
                'safety_validation': safety_results,
                'estimated_improvement': sum(rec.estimated_improvement for rec in recommendations) / len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Optimization plan generation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def integrate_with_progress_validator(
        self,
        metrics: List[PerformanceMetrics],
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """Integrate analytics with existing progress validation."""
        try:
            # Analyze performance
            performance_analysis = self.analyze_performance(metrics, include_predictions=False, include_optimizations=False)
            
            # Combine with validation results
            integrated_analysis = {
                'timestamp': datetime.now(),
                'validation_results': {
                    'is_valid': validation_result.is_valid,
                    'confidence': validation_result.confidence,
                    'issues_count': len(validation_result.issues),
                    'validation_timestamp': validation_result.timestamp
                },
                'performance_analysis': performance_analysis,
                'correlation_insights': self._analyze_validation_performance_correlation(validation_result, performance_analysis)
            }
            
            logger.info("Analytics integrated with progress validation")
            return integrated_analysis
            
        except Exception as e:
            logger.error(f"Integration with progress validator failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def _ensure_performance_metrics(
        self,
        metrics: Union[List[PerformanceMetrics], List[SystemMetrics]]
    ) -> List[PerformanceMetrics]:
        """Convert SystemMetrics to PerformanceMetrics if needed."""
        if not metrics:
            return []
        
        if isinstance(metrics[0], SystemMetrics):
            # Convert SystemMetrics to PerformanceMetrics
            perf_metrics = []
            for sys_metric in metrics:
                perf_metric = PerformanceMetrics(
                    base_metrics=sys_metric,
                    execution_time=2.0,  # Default value
                    throughput=100.0,    # Default value
                    error_rate=0.01      # Default value
                )
                perf_metrics.append(perf_metric)
            return perf_metrics
        
        return metrics
    
    def _calculate_health_score(
        self,
        metrics: List[PerformanceMetrics],
        bottlenecks: List[BottleneckAnalysis],
        anomalies: List[Any]
    ) -> float:
        """Calculate overall system health score (0-100)."""
        if not metrics:
            return 0.0
        
        # Base score from average metrics
        avg_cpu = sum(m.base_metrics.cpu_usage for m in metrics) / len(metrics)
        avg_memory = sum(m.base_metrics.memory_usage for m in metrics) / len(metrics)
        avg_error_rate = sum(m.error_rate for m in metrics) / len(metrics)
        
        # Calculate base health (lower resource usage = better health)
        cpu_health = max(0, 100 - avg_cpu)
        memory_health = max(0, 100 - avg_memory)
        error_health = max(0, 100 - (avg_error_rate * 10000))  # Scale error rate
        
        base_health = (cpu_health + memory_health + error_health) / 3
        
        # Penalize for bottlenecks and anomalies
        bottleneck_penalty = len(bottlenecks) * 10
        anomaly_penalty = len(anomalies) * 5
        
        final_health = max(0, base_health - bottleneck_penalty - anomaly_penalty)
        return min(100, final_health)
    
    def _analyze_validation_performance_correlation(
        self,
        validation_result: ValidationResult,
        performance_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between validation results and performance."""
        correlations = {
            'validation_confidence_vs_performance': None,
            'issues_impact_on_performance': None,
            'recommendations': []
        }
        
        # Analyze confidence correlation
        if validation_result.confidence and 'overall_health_score' in performance_analysis:
            health_score = performance_analysis['overall_health_score']
            correlation_strength = abs(validation_result.confidence * 100 - health_score)
            
            if correlation_strength < 20:
                correlations['validation_confidence_vs_performance'] = 'strong_positive'
            elif correlation_strength < 40:
                correlations['validation_confidence_vs_performance'] = 'moderate'
            else:
                correlations['validation_confidence_vs_performance'] = 'weak'
        
        # Analyze issue impact
        if validation_result.issues and 'bottlenecks' in performance_analysis:
            validation_issues = len(validation_result.issues)
            performance_bottlenecks = len(performance_analysis['bottlenecks'])
            
            if validation_issues > 0 and performance_bottlenecks > 0:
                correlations['issues_impact_on_performance'] = 'both_systems_detect_problems'
            elif validation_issues > 0:
                correlations['issues_impact_on_performance'] = 'validation_only'
            elif performance_bottlenecks > 0:
                correlations['issues_impact_on_performance'] = 'performance_only'
            else:
                correlations['issues_impact_on_performance'] = 'no_issues_detected'
        
        return correlations
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current analytics system status."""
        return {
            'timestamp': datetime.now(),
            'monitoring_active': self._monitoring_active,
            'baseline_established': self._baseline_established,
            'components_status': {
                'engine': 'operational',
                'collector': 'operational',
                'optimizer': 'operational',
                'predictor': 'operational',
                'dashboard': 'operational',
                'monitor': 'operational' if self._monitoring_active else 'idle',
                'reporter': 'operational',
                'regression_detector': 'operational' if self._baseline_established else 'no_baseline'
            },
            'configuration': {
                'collection_interval': self.config.collection_interval,
                'anomaly_sensitivity': self.config.anomaly_detection_sensitivity,
                'bottleneck_threshold': self.config.bottleneck_threshold,
                'ai_optimization_enabled': self.config.enable_ai_optimization,
                'predictive_modeling_enabled': self.config.enable_predictive_modeling
            }
        }


# Convenience function for quick analytics setup
def create_analytics_system(config: Optional[AnalyticsConfiguration] = None) -> AnalyticsIntegrationManager:
    """
    Create and initialize a complete analytics system.
    
    Args:
        config: Optional analytics configuration
        
    Returns:
        Fully initialized analytics integration manager
    """
    manager = AnalyticsIntegrationManager(config)
    manager.initialize_system()
    return manager


# Example usage patterns
def quick_analysis(metrics: List[Union[PerformanceMetrics, SystemMetrics]]) -> Dict[str, Any]:
    """Quick performance analysis with default settings."""
    manager = create_analytics_system()
    return manager.analyze_performance(metrics)


def setup_monitoring(alert_callback: callable, config: Optional[AnalyticsConfiguration] = None) -> AnalyticsIntegrationManager:
    """Set up real-time monitoring with alerts."""
    manager = create_analytics_system(config)
    manager.start_monitoring(alert_callback)
    return manager