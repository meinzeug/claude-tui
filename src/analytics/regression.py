"""
Performance Regression Detection System.

Advanced system for detecting performance regressions, identifying root causes,
and providing automated alerts when system performance degrades unexpectedly.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..core.types import Severity, Priority
from .models import (
    PerformanceMetrics, PerformanceAlert, TrendAnalysis,
    AnalyticsConfiguration, AlertType
)


class RegressionType(str, Enum):
    """Types of performance regressions."""
    SUDDEN_DROP = "sudden_drop"
    GRADUAL_DECLINE = "gradual_decline"
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_SURGE = "anomaly_surge"
    BASELINE_DEVIATION = "baseline_deviation"


class RegressionSeverity(str, Enum):
    """Regression severity levels."""
    MINOR = "minor"      # < 10% performance impact
    MODERATE = "moderate"  # 10-25% performance impact
    MAJOR = "major"       # 25-50% performance impact
    CRITICAL = "critical" # > 50% performance impact


class DetectionMethod(str, Enum):
    """Regression detection methods."""
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    THRESHOLD = "threshold"
    TREND_ANALYSIS = "trend_analysis"
    BASELINE_COMPARISON = "baseline_comparison"


@dataclass
class RegressionAlert:
    """Performance regression alert."""
    regression_id: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    # Regression details
    regression_type: RegressionType
    severity: RegressionSeverity
    affected_metric: str
    
    # Performance impact
    performance_impact: float  # Percentage degradation
    baseline_value: float
    current_value: float
    regression_start_time: Optional[datetime] = None
    
    # Detection details
    detection_method: DetectionMethod
    confidence_score: float = 0.0
    statistical_significance: float = 0.0
    
    # Context information
    environment: str = ""
    affected_components: List[str] = field(default_factory=list)
    
    # Root cause analysis
    potential_causes: List[str] = field(default_factory=list)
    correlated_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evidence
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    time_series_data: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Response tracking
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    metric_name: str
    baseline_period: timedelta
    
    # Statistical properties
    mean_value: float = 0.0
    std_deviation: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    
    # Seasonal adjustments
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    data_points: int = 0
    confidence_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Thresholds for regression detection
    regression_threshold: float = 0.2  # 20% degradation
    anomaly_threshold: float = 2.0     # 2 standard deviations
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegressionReport:
    """Comprehensive regression analysis report."""
    report_id: str
    analysis_period: timedelta
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Summary statistics
    total_regressions: int = 0
    critical_regressions: int = 0
    resolved_regressions: int = 0
    avg_detection_time: float = 0.0
    
    # Regression breakdown
    regressions_by_type: Dict[RegressionType, int] = field(default_factory=dict)
    regressions_by_severity: Dict[RegressionSeverity, int] = field(default_factory=dict)
    regressions_by_metric: Dict[str, int] = field(default_factory=dict)
    
    # Performance impact analysis
    total_performance_impact: float = 0.0
    most_impacted_components: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trend analysis
    regression_frequency_trend: str = "stable"  # increasing, decreasing, stable
    mean_time_to_detection: float = 0.0
    mean_time_to_resolution: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)


class RegressionDetector:
    """
    Advanced performance regression detection system.
    
    Features:
    - Multiple detection algorithms (statistical, ML-based, threshold)
    - Adaptive baseline learning
    - Root cause correlation analysis
    - Automated alert generation
    - False positive suppression
    - Historical regression tracking
    - Performance impact assessment
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        baseline_learning_period: timedelta = timedelta(days=14)
    ):
        """Initialize the regression detector."""
        self.config = config or AnalyticsConfiguration()
        self.baseline_learning_period = baseline_learning_period
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Performance baselines
        self.baselines: Dict[str, PerformanceBaseline] = {}
        
        # Regression tracking
        self.active_regressions: Dict[str, RegressionAlert] = {}
        self.regression_history: deque = deque(maxlen=10000)
        
        # Detection models
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Detection parameters
        self.detection_config = {
            'min_data_points': 30,
            'confidence_threshold': 0.8,
            'false_positive_rate': 0.05,
            'detection_sensitivity': 0.8,
            'baseline_update_interval': timedelta(hours=6)
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'false_positives': 0,
            'confirmed_regressions': 0,
            'avg_detection_accuracy': 0.0
        }
        
        # Suppression rules
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialize the regression detector."""
        try:
            self.logger.info("Initializing Regression Detector...")
            
            # Load existing baselines
            await self._load_baselines()
            
            # Initialize detection models
            await self._initialize_detection_models()
            
            # Load suppression rules
            await self._load_suppression_rules()
            
            self.logger.info("Regression Detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize regression detector: {e}")
            raise
    
    async def start_monitoring(self) -> None:
        """Start regression monitoring."""
        if self.is_monitoring:
            self.logger.warning("Regression monitoring already active")
            return
        
        self.logger.info("Starting regression monitoring...")
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop regression monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping regression monitoring...")
        self.is_monitoring = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def detect_regressions(
        self,
        current_metrics: PerformanceMetrics,
        historical_metrics: List[PerformanceMetrics]
    ) -> List[RegressionAlert]:
        """Detect performance regressions in current metrics."""
        try:
            regressions_detected = []
            
            # Key metrics to monitor for regressions
            monitored_metrics = [
                'cpu_percent', 'memory_percent', 'throughput', 
                'latency_p95', 'error_rate', 'ai_response_time'
            ]
            
            for metric_name in monitored_metrics:
                try:
                    # Get current value
                    current_value = getattr(current_metrics, metric_name, None)
                    if current_value is None:
                        continue
                    
                    # Detect regressions for this metric
                    metric_regressions = await self._detect_metric_regressions(
                        metric_name, current_value, current_metrics.timestamp,
                        historical_metrics
                    )
                    
                    regressions_detected.extend(metric_regressions)
                    
                except Exception as e:
                    self.logger.warning(f"Error detecting regressions for {metric_name}: {e}")
            
            # Filter and deduplicate regressions
            filtered_regressions = await self._filter_regressions(regressions_detected)
            
            # Store active regressions
            for regression in filtered_regressions:
                self.active_regressions[regression.regression_id] = regression
                self.detection_stats['total_detections'] += 1
            
            return filtered_regressions
            
        except Exception as e:
            self.logger.error(f"Error detecting regressions: {e}")
            return []
    
    async def update_baselines(
        self,
        historical_metrics: List[PerformanceMetrics]
    ) -> None:
        """Update performance baselines with new data."""
        try:
            if len(historical_metrics) < self.detection_config['min_data_points']:
                return
            
            # Group metrics by metric name
            metrics_by_name = self._group_metrics_by_name(historical_metrics)
            
            for metric_name, values in metrics_by_name.items():
                await self._update_metric_baseline(metric_name, values)
            
            self.logger.debug(f"Updated baselines for {len(metrics_by_name)} metrics")
            
        except Exception as e:
            self.logger.error(f"Error updating baselines: {e}")
    
    async def analyze_regression(
        self,
        regression_id: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform detailed analysis of a detected regression."""
        if regression_id not in self.active_regressions:
            return {'error': f'Regression {regression_id} not found'}
        
        regression = self.active_regressions[regression_id]
        
        try:
            analysis = {
                'regression_details': {
                    'id': regression.regression_id,
                    'type': regression.regression_type.value,
                    'severity': regression.severity.value,
                    'metric': regression.affected_metric,
                    'impact': regression.performance_impact,
                    'confidence': regression.confidence_score
                },
                'root_cause_analysis': await self._perform_root_cause_analysis(regression),
                'impact_assessment': await self._assess_regression_impact(regression),
                'correlation_analysis': await self._analyze_correlations(regression),
                'recommendations': await self._generate_regression_recommendations(regression),
                'timeline': self._create_regression_timeline(regression),
                'additional_context': additional_context or {}
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing regression {regression_id}: {e}")
            return {'error': str(e)}
    
    async def acknowledge_regression(
        self,
        regression_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """Acknowledge a detected regression."""
        if regression_id in self.active_regressions:
            regression = self.active_regressions[regression_id]
            regression.acknowledged = True
            regression.acknowledged_at = datetime.utcnow()
            regression.acknowledged_by = acknowledged_by
            
            if notes:
                if 'acknowledgment_notes' not in regression.supporting_data:
                    regression.supporting_data['acknowledgment_notes'] = []
                regression.supporting_data['acknowledgment_notes'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'notes': notes,
                    'by': acknowledged_by
                })
            
            self.logger.info(f"Regression {regression_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    async def resolve_regression(
        self,
        regression_id: str,
        resolved_by: str,
        resolution_notes: str
    ) -> bool:
        """Mark a regression as resolved."""
        if regression_id in self.active_regressions:
            regression = self.active_regressions[regression_id]
            regression.resolved = True
            regression.resolved_at = datetime.utcnow()
            regression.resolution_notes = resolution_notes
            
            # Move to history
            self.regression_history.append(regression)
            del self.active_regressions[regression_id]
            
            self.logger.info(f"Regression {regression_id} resolved by {resolved_by}")
            return True
        
        return False
    
    async def generate_regression_report(
        self,
        period: timedelta = timedelta(days=7)
    ) -> RegressionReport:
        """Generate comprehensive regression analysis report."""
        try:
            cutoff_time = datetime.utcnow() - period
            
            # Get regressions from the specified period
            period_regressions = [
                r for r in self.regression_history
                if r.detected_at >= cutoff_time
            ]
            
            # Add current active regressions
            period_regressions.extend(self.active_regressions.values())
            
            # Generate report
            report_id = f"regression_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            report = RegressionReport(
                report_id=report_id,
                analysis_period=period,
                total_regressions=len(period_regressions),
                critical_regressions=len([r for r in period_regressions if r.severity == RegressionSeverity.CRITICAL]),
                resolved_regressions=len([r for r in period_regressions if r.resolved])
            )
            
            # Analyze regressions by type
            for regression in period_regressions:
                regression_type = regression.regression_type
                if regression_type not in report.regressions_by_type:
                    report.regressions_by_type[regression_type] = 0
                report.regressions_by_type[regression_type] += 1
                
                # By severity
                severity = regression.severity
                if severity not in report.regressions_by_severity:
                    report.regressions_by_severity[severity] = 0
                report.regressions_by_severity[severity] += 1
                
                # By metric
                metric = regression.affected_metric
                if metric not in report.regressions_by_metric:
                    report.regressions_by_metric[metric] = 0
                report.regressions_by_metric[metric] += 1
            
            # Calculate performance impact
            report.total_performance_impact = sum(
                r.performance_impact for r in period_regressions
            )
            
            # Generate recommendations
            report.recommendations = await self._generate_report_recommendations(
                period_regressions
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating regression report: {e}")
            raise
    
    # Private methods
    
    async def _detect_metric_regressions(
        self,
        metric_name: str,
        current_value: float,
        timestamp: datetime,
        historical_metrics: List[PerformanceMetrics]
    ) -> List[RegressionAlert]:
        """Detect regressions for a specific metric."""
        regressions = []
        
        # Get baseline for the metric
        baseline = self.baselines.get(metric_name)
        if not baseline:
            # Create baseline if it doesn't exist
            baseline = await self._create_baseline(metric_name, historical_metrics)
            if not baseline:
                return regressions
        
        # Multiple detection methods
        detection_results = await asyncio.gather(
            self._statistical_detection(metric_name, current_value, baseline),
            self._threshold_detection(metric_name, current_value, baseline),
            self._trend_based_detection(metric_name, current_value, historical_metrics),
            self._anomaly_detection(metric_name, current_value, historical_metrics),
            return_exceptions=True
        )
        
        # Process detection results
        for i, result in enumerate(detection_results):
            if isinstance(result, Exception):
                self.logger.warning(f"Detection method {i} failed for {metric_name}: {result}")
                continue
            
            if result:  # Regression detected
                method_names = [
                    DetectionMethod.STATISTICAL,
                    DetectionMethod.THRESHOLD,
                    DetectionMethod.TREND_ANALYSIS,
                    DetectionMethod.MACHINE_LEARNING
                ]
                
                regression_alert = RegressionAlert(
                    regression_id=f"{metric_name}_{timestamp.timestamp()}_{i}",
                    regression_type=result['type'],
                    severity=result['severity'],
                    affected_metric=metric_name,
                    performance_impact=result['impact'],
                    baseline_value=baseline.mean_value,
                    current_value=current_value,
                    detection_method=method_names[i],
                    confidence_score=result['confidence'],
                    statistical_significance=result.get('significance', 0.0),
                    environment=getattr(historical_metrics[-1], 'environment', '') if historical_metrics else '',
                    supporting_data=result.get('evidence', {})
                )
                
                regressions.append(regression_alert)
        
        return regressions
    
    async def _statistical_detection(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline
    ) -> Optional[Dict[str, Any]]:
        """Statistical regression detection using z-score analysis."""
        if baseline.std_deviation == 0:
            return None
        
        # Calculate z-score
        z_score = abs(current_value - baseline.mean_value) / baseline.std_deviation
        
        # Check for regression (depends on metric type)
        is_regression = False
        impact = 0.0
        
        if self._is_lower_better_metric(metric_name):
            # For metrics where lower is better (error_rate, latency)
            if current_value > baseline.mean_value:
                is_regression = z_score > 2.0  # 2 standard deviations
                impact = ((current_value - baseline.mean_value) / baseline.mean_value) * 100
        else:
            # For metrics where higher is better (throughput, cache_hit_rate)
            if current_value < baseline.mean_value:
                is_regression = z_score > 2.0
                impact = ((baseline.mean_value - current_value) / baseline.mean_value) * 100
        
        if not is_regression:
            return None
        
        # Determine severity based on impact
        if impact > 50:
            severity = RegressionSeverity.CRITICAL
        elif impact > 25:
            severity = RegressionSeverity.MAJOR
        elif impact > 10:
            severity = RegressionSeverity.MODERATE
        else:
            severity = RegressionSeverity.MINOR
        
        # Determine regression type
        if z_score > 3.0:
            regression_type = RegressionType.SUDDEN_DROP
        else:
            regression_type = RegressionType.THRESHOLD_BREACH
        
        return {
            'type': regression_type,
            'severity': severity,
            'impact': impact,
            'confidence': min(1.0, z_score / 3.0),  # Normalize confidence
            'significance': 1 - stats.norm.cdf(z_score),  # P-value
            'evidence': {
                'z_score': z_score,
                'baseline_mean': baseline.mean_value,
                'baseline_std': baseline.std_deviation
            }
        }
    
    async def _threshold_detection(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline
    ) -> Optional[Dict[str, Any]]:
        """Threshold-based regression detection."""
        threshold = baseline.regression_threshold
        
        # Calculate deviation from baseline
        if baseline.mean_value == 0:
            return None
        
        deviation = abs(current_value - baseline.mean_value) / baseline.mean_value
        
        if deviation < threshold:
            return None
        
        # Determine if this is actually a regression
        is_regression = False
        if self._is_lower_better_metric(metric_name):
            is_regression = current_value > baseline.mean_value * (1 + threshold)
        else:
            is_regression = current_value < baseline.mean_value * (1 - threshold)
        
        if not is_regression:
            return None
        
        impact = deviation * 100
        
        # Determine severity
        if impact > 50:
            severity = RegressionSeverity.CRITICAL
        elif impact > 25:
            severity = RegressionSeverity.MAJOR
        elif impact > 10:
            severity = RegressionSeverity.MODERATE
        else:
            severity = RegressionSeverity.MINOR
        
        return {
            'type': RegressionType.THRESHOLD_BREACH,
            'severity': severity,
            'impact': impact,
            'confidence': min(1.0, deviation / threshold),
            'evidence': {
                'threshold': threshold,
                'deviation': deviation,
                'baseline_value': baseline.mean_value
            }
        }
    
    async def _trend_based_detection(
        self,
        metric_name: str,
        current_value: float,
        historical_metrics: List[PerformanceMetrics]
    ) -> Optional[Dict[str, Any]]:
        """Trend-based regression detection."""
        if len(historical_metrics) < 10:
            return None
        
        # Extract recent values (last 10 data points)
        recent_values = []
        for metric in historical_metrics[-10:]:
            if hasattr(metric, metric_name):
                value = getattr(metric, metric_name)
                if value is not None:
                    recent_values.append(value)
        
        if len(recent_values) < 5:
            return None
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
        
        # Determine if trend indicates regression
        is_degrading_trend = False
        if self._is_lower_better_metric(metric_name):
            is_degrading_trend = slope > 0  # Increasing trend is bad
        else:
            is_degrading_trend = slope < 0  # Decreasing trend is bad
        
        # Check if trend is statistically significant
        if not is_degrading_trend or abs(r_value) < 0.7 or p_value > 0.05:
            return None
        
        # Calculate impact based on trend extrapolation
        trend_value = slope * len(recent_values) + intercept
        baseline_value = recent_values[0] if recent_values else current_value
        
        if baseline_value == 0:
            return None
        
        impact = abs(trend_value - baseline_value) / baseline_value * 100
        
        if impact < 10:  # Less than 10% impact
            return None
        
        # Determine severity
        if impact > 40:
            severity = RegressionSeverity.CRITICAL
        elif impact > 20:
            severity = RegressionSeverity.MAJOR
        elif impact > 10:
            severity = RegressionSeverity.MODERATE
        else:
            severity = RegressionSeverity.MINOR
        
        return {
            'type': RegressionType.GRADUAL_DECLINE,
            'severity': severity,
            'impact': impact,
            'confidence': abs(r_value),
            'significance': 1 - p_value,
            'evidence': {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing'
            }
        }
    
    async def _anomaly_detection(
        self,
        metric_name: str,
        current_value: float,
        historical_metrics: List[PerformanceMetrics]
    ) -> Optional[Dict[str, Any]]:
        """ML-based anomaly detection for regression identification."""
        if len(historical_metrics) < 30:
            return None
        
        # Extract historical values
        historical_values = []
        for metric in historical_metrics:
            if hasattr(metric, metric_name):
                value = getattr(metric, metric_name)
                if value is not None:
                    historical_values.append(value)
        
        if len(historical_values) < 30:
            return None
        
        # Get or create anomaly detector
        detector = await self._get_anomaly_detector(metric_name, historical_values)
        
        # Check if current value is an anomaly
        is_anomaly = detector.predict([[current_value]])[0] == -1
        
        if not is_anomaly:
            return None
        
        # Calculate anomaly score
        anomaly_score = detector.decision_function([[current_value]])[0]
        
        # Check if anomaly represents a regression
        baseline_value = statistics.mean(historical_values[-20:])  # Recent baseline
        
        is_regression = False
        impact = 0.0
        
        if self._is_lower_better_metric(metric_name):
            is_regression = current_value > baseline_value
            impact = ((current_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
        else:
            is_regression = current_value < baseline_value
            impact = ((baseline_value - current_value) / baseline_value) * 100 if baseline_value > 0 else 0
        
        if not is_regression or impact < 5:
            return None
        
        # Determine severity
        if impact > 50:
            severity = RegressionSeverity.CRITICAL
        elif impact > 25:
            severity = RegressionSeverity.MAJOR
        elif impact > 10:
            severity = RegressionSeverity.MODERATE
        else:
            severity = RegressionSeverity.MINOR
        
        return {
            'type': RegressionType.ANOMALY_SURGE,
            'severity': severity,
            'impact': impact,
            'confidence': min(1.0, abs(anomaly_score)),
            'evidence': {
                'anomaly_score': anomaly_score,
                'baseline_value': baseline_value,
                'is_outlier': True
            }
        }
    
    def _is_lower_better_metric(self, metric_name: str) -> bool:
        """Determine if lower values are better for a metric."""
        lower_better_metrics = {
            'cpu_percent', 'memory_percent', 'disk_percent', 
            'error_rate', 'latency_p50', 'latency_p95', 'latency_p99',
            'ai_response_time', 'hallucination_rate'
        }
        return metric_name in lower_better_metrics
    
    def _group_metrics_by_name(
        self, 
        metrics: List[PerformanceMetrics]
    ) -> Dict[str, List[float]]:
        """Group metrics by metric name."""
        grouped = {}
        
        metric_names = [
            'cpu_percent', 'memory_percent', 'throughput',
            'latency_p95', 'error_rate', 'ai_response_time'
        ]
        
        for metric_name in metric_names:
            values = []
            for metric in metrics:
                if hasattr(metric, metric_name):
                    value = getattr(metric, metric_name)
                    if value is not None:
                        values.append(value)
            
            if values:
                grouped[metric_name] = values
        
        return grouped
    
    async def _create_baseline(
        self,
        metric_name: str,
        historical_metrics: List[PerformanceMetrics]
    ) -> Optional[PerformanceBaseline]:
        """Create performance baseline for a metric."""
        values = []
        for metric in historical_metrics:
            if hasattr(metric, metric_name):
                value = getattr(metric, metric_name)
                if value is not None:
                    values.append(value)
        
        if len(values) < self.detection_config['min_data_points']:
            return None
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_period=self.baseline_learning_period,
            mean_value=statistics.mean(values),
            std_deviation=statistics.stdev(values) if len(values) > 1 else 0,
            percentile_95=np.percentile(values, 95),
            percentile_99=np.percentile(values, 99),
            min_value=min(values),
            max_value=max(values),
            data_points=len(values),
            confidence_level=min(1.0, len(values) / 100)  # Confidence based on data points
        )
        
        self.baselines[metric_name] = baseline
        return baseline
    
    async def _update_metric_baseline(
        self,
        metric_name: str,
        values: List[float]
    ) -> None:
        """Update baseline for a specific metric."""
        if len(values) < self.detection_config['min_data_points']:
            return
        
        baseline = self.baselines.get(metric_name)
        if not baseline:
            baseline = PerformanceBaseline(
                metric_name=metric_name,
                baseline_period=self.baseline_learning_period
            )
            self.baselines[metric_name] = baseline
        
        # Update statistical properties
        baseline.mean_value = statistics.mean(values)
        baseline.std_deviation = statistics.stdev(values) if len(values) > 1 else 0
        baseline.percentile_95 = np.percentile(values, 95)
        baseline.percentile_99 = np.percentile(values, 99)
        baseline.min_value = min(values)
        baseline.max_value = max(values)
        baseline.data_points = len(values)
        baseline.last_updated = datetime.utcnow()
        baseline.confidence_level = min(1.0, len(values) / 100)
    
    async def _get_anomaly_detector(
        self,
        metric_name: str,
        historical_values: List[float]
    ) -> IsolationForest:
        """Get or create anomaly detector for a metric."""
        if metric_name in self.anomaly_detectors:
            return self.anomaly_detectors[metric_name]
        
        # Create and train new detector
        detector = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        
        # Prepare data
        X = np.array(historical_values).reshape(-1, 1)
        detector.fit(X)
        
        # Store detector
        self.anomaly_detectors[metric_name] = detector
        
        return detector
    
    async def _filter_regressions(
        self,
        regressions: List[RegressionAlert]
    ) -> List[RegressionAlert]:
        """Filter and deduplicate regression alerts."""
        if not regressions:
            return []
        
        # Sort by confidence score
        sorted_regressions = sorted(
            regressions,
            key=lambda r: r.confidence_score,
            reverse=True
        )
        
        # Remove duplicates for the same metric
        filtered = []
        seen_metrics = set()
        
        for regression in sorted_regressions:
            metric_key = f"{regression.affected_metric}_{regression.regression_type.value}"
            
            if metric_key not in seen_metrics:
                # Apply suppression rules
                if not await self._is_suppressed(regression):
                    filtered.append(regression)
                    seen_metrics.add(metric_key)
        
        return filtered
    
    async def _is_suppressed(self, regression: RegressionAlert) -> bool:
        """Check if regression should be suppressed."""
        # Check global suppression rules
        for rule_id, rule in self.suppression_rules.items():
            if (rule.get('metric') == regression.affected_metric and
                rule.get('min_impact', 0) <= regression.performance_impact):
                
                suppress_until = rule.get('suppress_until')
                if suppress_until and datetime.utcnow() < suppress_until:
                    return True
        
        return False
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for regression detection."""
        while self.is_monitoring:
            try:
                # Update baselines periodically
                if self.baselines:
                    last_update = min(
                        baseline.last_updated for baseline in self.baselines.values()
                    )
                    
                    if (datetime.utcnow() - last_update > 
                        self.detection_config['baseline_update_interval']):
                        
                        self.logger.debug("Updating performance baselines...")
                        # This would trigger baseline updates with fresh data
                
                # Clean up old regressions
                await self._cleanup_old_data()
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in regression monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old regression data."""
        # Remove very old regressions from history
        cutoff_time = datetime.utcnow() - timedelta(days=90)
        
        # Filter regression history
        self.regression_history = deque([
            r for r in self.regression_history
            if r.detected_at >= cutoff_time
        ], maxlen=10000)
    
    # Placeholder methods for advanced analysis
    async def _perform_root_cause_analysis(self, regression: RegressionAlert) -> Dict[str, Any]:
        """Perform root cause analysis for a regression."""
        return {
            'potential_causes': regression.potential_causes,
            'analysis_method': 'correlation_analysis',
            'confidence': 0.7
        }
    
    async def _assess_regression_impact(self, regression: RegressionAlert) -> Dict[str, Any]:
        """Assess the impact of a regression."""
        return {
            'performance_impact': regression.performance_impact,
            'affected_users': 'unknown',
            'business_impact': 'medium'
        }
    
    async def _analyze_correlations(self, regression: RegressionAlert) -> Dict[str, Any]:
        """Analyze correlations with other metrics."""
        return {
            'correlated_metrics': [],
            'correlation_strength': 0.0
        }
    
    async def _generate_regression_recommendations(
        self, 
        regression: RegressionAlert
    ) -> List[str]:
        """Generate recommendations for addressing a regression."""
        recommendations = []
        
        if regression.severity == RegressionSeverity.CRITICAL:
            recommendations.append("Immediately investigate and prioritize resolution")
        
        if regression.affected_metric in ['error_rate', 'latency_p95']:
            recommendations.append("Check recent deployments and configuration changes")
        
        return recommendations
    
    def _create_regression_timeline(self, regression: RegressionAlert) -> List[Dict[str, Any]]:
        """Create timeline of regression events."""
        timeline = [
            {
                'timestamp': regression.detected_at.isoformat(),
                'event': 'Regression detected',
                'details': f"Performance degradation in {regression.affected_metric}"
            }
        ]
        
        if regression.acknowledged_at:
            timeline.append({
                'timestamp': regression.acknowledged_at.isoformat(),
                'event': 'Regression acknowledged',
                'details': f"Acknowledged by {regression.acknowledged_by}"
            })
        
        if regression.resolved_at:
            timeline.append({
                'timestamp': regression.resolved_at.isoformat(),
                'event': 'Regression resolved',
                'details': regression.resolution_notes or "Resolved"
            })
        
        return timeline
    
    async def _generate_report_recommendations(
        self,
        regressions: List[RegressionAlert]
    ) -> List[str]:
        """Generate recommendations for the regression report."""
        recommendations = []
        
        if len(regressions) > 10:
            recommendations.append("High number of regressions - review deployment processes")
        
        critical_regressions = [r for r in regressions if r.severity == RegressionSeverity.CRITICAL]
        if critical_regressions:
            recommendations.append(f"Address {len(critical_regressions)} critical regressions immediately")
        
        return recommendations
    
    async def _load_baselines(self) -> None:
        """Load existing performance baselines."""
        # Placeholder for baseline loading from storage
        pass
    
    async def _initialize_detection_models(self) -> None:
        """Initialize machine learning detection models."""
        # Placeholder for ML model initialization
        pass
    
    async def _load_suppression_rules(self) -> None:
        """Load regression suppression rules."""
        # Placeholder for loading suppression rules
        pass