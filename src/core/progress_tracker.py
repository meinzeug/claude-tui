"""
Real vs Fake Progress Tracking System.

This module implements comprehensive progress tracking that can distinguish
between authentic code completion and superficial "fake" progress that appears
complete but lacks substance or functionality.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .types import (
    Issue, IssueType, Severity, ValidationResult, ProgressMetrics, 
    TaskStatus, PathStr
)
from .validator import ProgressValidator, CodeQualityAnalyzer
from .logger import get_logger


class ProgressType(str, Enum):
    """Types of progress tracking."""
    REAL = "real"
    FAKE = "fake" 
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProgressConfidence(str, Enum):
    """Confidence levels for progress assessments."""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 80-89%
    MEDIUM = "medium"       # 60-79%
    LOW = "low"            # 40-59%
    VERY_LOW = "very_low"  # 0-39%


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a specific point in time."""
    timestamp: datetime
    real_progress: float
    fake_progress: float
    total_progress: float
    authenticity_score: float
    quality_score: float
    files_analyzed: int
    issues_found: int
    confidence: ProgressConfidence
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def authenticity_rate(self) -> float:
        """Calculate authenticity rate."""
        if self.total_progress == 0:
            return 100.0
        return (self.real_progress / self.total_progress) * 100


@dataclass
class ProgressTrend:
    """Progress trend analysis over time."""
    direction: str  # "improving", "declining", "stable"
    velocity: float  # Rate of change
    consistency: float  # How consistent the progress is
    predicted_completion: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class ProgressAlert:
    """Alert for progress anomalies."""
    id: str
    timestamp: datetime
    alert_type: str
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class RealProgressTracker:
    """
    Advanced progress tracking system that distinguishes real from fake progress.
    
    Features:
    - Real-time progress monitoring
    - Authenticity scoring
    - Trend analysis
    - Anomaly detection
    - Progress prediction
    - Alert generation
    """
    
    def __init__(
        self, 
        project_path: PathStr,
        validation_interval: int = 30,  # seconds
        history_size: int = 100
    ):
        self.project_path = Path(project_path)
        self.validation_interval = validation_interval
        self.history_size = history_size
        
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.validator = ProgressValidator(enable_quality_analysis=True)
        self.quality_analyzer = CodeQualityAnalyzer()
        
        # Progress tracking data
        self.snapshots: List[ProgressSnapshot] = []
        self.alerts: List[ProgressAlert] = []
        self.baseline_metrics: Optional[ProgressSnapshot] = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'authenticity_drop': 20.0,  # Alert if authenticity drops by this much
            'fake_progress_spike': 30.0,  # Alert if fake progress increases rapidly
            'quality_degradation': 15.0,  # Alert if quality score drops
            'stagnation_time': 300,  # Alert if no progress for this many seconds
        }
    
    async def start_monitoring(self) -> None:
        """Start real-time progress monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.logger.info("Starting progress monitoring...")
        self.is_monitoring = True
        
        # Take baseline measurement
        await self._take_baseline_snapshot()
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop progress monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping progress monitoring...")
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def take_snapshot(self) -> ProgressSnapshot:
        """Take a progress snapshot at the current moment."""
        
        # Validate current state
        validation_result = await self.validator.validate_codebase(self.project_path)
        
        # Analyze quality metrics
        quality_metrics = await self.quality_analyzer.analyze_codebase(self.project_path)
        
        # Calculate progress metrics
        real_progress, fake_progress = self._calculate_progress_split(validation_result)
        total_progress = real_progress + fake_progress
        
        # Determine confidence level
        confidence = self._calculate_confidence(validation_result, quality_metrics)
        
        snapshot = ProgressSnapshot(
            timestamp=datetime.utcnow(),
            real_progress=real_progress,
            fake_progress=fake_progress,
            total_progress=total_progress,
            authenticity_score=validation_result.authenticity_score,
            quality_score=quality_metrics.overall_score,
            files_analyzed=len(self._get_source_files()),
            issues_found=len(validation_result.issues),
            confidence=confidence,
            details={
                'validation_result': validation_result.dict(),
                'quality_metrics': {
                    'lines_of_code': quality_metrics.lines_of_code,
                    'complexity': quality_metrics.cyclomatic_complexity,
                    'test_coverage': quality_metrics.test_coverage,
                    'documentation_coverage': quality_metrics.documentation_coverage,
                    'maintainability_index': quality_metrics.maintainability_index,
                    'technical_debt_minutes': quality_metrics.technical_debt_minutes,
                    'duplication_percentage': quality_metrics.duplication_percentage,
                    'security_hotspots': quality_metrics.security_hotspots
                }
            }
        )
        
        # Add to history
        self.snapshots.append(snapshot)
        
        # Maintain history size limit
        if len(self.snapshots) > self.history_size:
            self.snapshots = self.snapshots[-self.history_size:]
        
        # Check for alerts
        await self._check_for_alerts(snapshot)
        
        return snapshot
    
    def get_current_progress(self) -> Optional[ProgressSnapshot]:
        """Get the most recent progress snapshot."""
        return self.snapshots[-1] if self.snapshots else None
    
    def get_progress_history(
        self, 
        hours: int = 24
    ) -> List[ProgressSnapshot]:
        """Get progress history for the specified time period."""
        if not self.snapshots:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            snapshot for snapshot in self.snapshots 
            if snapshot.timestamp >= cutoff_time
        ]
    
    def analyze_progress_trend(
        self, 
        hours: int = 2
    ) -> ProgressTrend:
        """Analyze progress trend over the specified time period."""
        
        history = self.get_progress_history(hours)
        
        if len(history) < 2:
            return ProgressTrend(
                direction="stable",
                velocity=0.0,
                consistency=100.0,
                confidence=0.0
            )
        
        # Calculate trend metrics
        real_progress_values = [s.real_progress for s in history]
        timestamps = [(s.timestamp - history[0].timestamp).total_seconds() for s in history]
        
        # Calculate velocity (progress per hour)
        if len(real_progress_values) >= 2:
            time_diff = timestamps[-1] - timestamps[0]
            progress_diff = real_progress_values[-1] - real_progress_values[0]
            velocity = (progress_diff / time_diff) * 3600 if time_diff > 0 else 0.0  # per hour
        else:
            velocity = 0.0
        
        # Determine direction
        if velocity > 1.0:
            direction = "improving"
        elif velocity < -1.0:
            direction = "declining"
        else:
            direction = "stable"
        
        # Calculate consistency (lower variance = higher consistency)
        if len(real_progress_values) > 1:
            variance = sum(
                (x - sum(real_progress_values) / len(real_progress_values)) ** 2 
                for x in real_progress_values
            ) / len(real_progress_values)
            consistency = max(0.0, 100.0 - variance)
        else:
            consistency = 100.0
        
        # Predict completion time
        predicted_completion = None
        if velocity > 0 and real_progress_values[-1] < 100.0:
            remaining_progress = 100.0 - real_progress_values[-1]
            hours_to_completion = remaining_progress / velocity
            predicted_completion = datetime.utcnow() + timedelta(hours=hours_to_completion)
        
        # Calculate confidence
        confidence = min(100.0, (len(history) / 10) * consistency / 100 * 100)
        
        return ProgressTrend(
            direction=direction,
            velocity=velocity,
            consistency=consistency,
            predicted_completion=predicted_completion,
            confidence=confidence
        )
    
    def get_alerts(
        self, 
        severity: Optional[Severity] = None,
        unresolved_only: bool = True
    ) -> List[ProgressAlert]:
        """Get progress alerts."""
        alerts = self.alerts
        
        if unresolved_only:
            alerts = [alert for alert in alerts if not alert.resolved]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive progress report."""
        current = self.get_current_progress()
        
        if not current:
            return {
                'status': 'no_data',
                'message': 'No progress data available'
            }
        
        trend = self.analyze_progress_trend()
        recent_history = self.get_progress_history(hours=4)
        active_alerts = self.get_alerts(unresolved_only=True)
        
        return {
            'status': 'active',
            'current_progress': {
                'real_progress': current.real_progress,
                'fake_progress': current.fake_progress,
                'total_progress': current.total_progress,
                'authenticity_rate': current.authenticity_rate,
                'quality_score': current.quality_score,
                'confidence': current.confidence.value
            },
            'trend_analysis': {
                'direction': trend.direction,
                'velocity': round(trend.velocity, 2),
                'consistency': round(trend.consistency, 2),
                'predicted_completion': trend.predicted_completion.isoformat() if trend.predicted_completion else None,
                'confidence': round(trend.confidence, 2)
            },
            'statistics': {
                'files_analyzed': current.files_analyzed,
                'issues_found': current.issues_found,
                'snapshots_taken': len(self.snapshots),
                'monitoring_duration': (
                    (current.timestamp - self.snapshots[0].timestamp).total_seconds() 
                    if len(self.snapshots) > 1 else 0
                )
            },
            'alerts': {
                'total': len(active_alerts),
                'by_severity': {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in Severity
                },
                'recent': [
                    {
                        'id': alert.id,
                        'type': alert.alert_type,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in active_alerts[:5]  # Last 5 alerts
                ]
            },
            'history': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'real_progress': snapshot.real_progress,
                    'fake_progress': snapshot.fake_progress,
                    'authenticity_rate': snapshot.authenticity_rate,
                    'quality_score': snapshot.quality_score
                }
                for snapshot in recent_history[-20:]  # Last 20 data points
            ]
        }
    
    async def export_progress_data(self, file_path: PathStr) -> None:
        """Export progress tracking data to file."""
        data = {
            'snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'real_progress': s.real_progress,
                    'fake_progress': s.fake_progress,
                    'total_progress': s.total_progress,
                    'authenticity_score': s.authenticity_score,
                    'quality_score': s.quality_score,
                    'files_analyzed': s.files_analyzed,
                    'issues_found': s.issues_found,
                    'confidence': s.confidence.value,
                    'details': s.details
                }
                for s in self.snapshots
            ],
            'alerts': [
                {
                    'id': a.id,
                    'timestamp': a.timestamp.isoformat(),
                    'alert_type': a.alert_type,
                    'severity': a.severity.value,
                    'message': a.message,
                    'details': a.details,
                    'resolved': a.resolved
                }
                for a in self.alerts
            ],
            'metadata': {
                'project_path': str(self.project_path),
                'validation_interval': self.validation_interval,
                'history_size': self.history_size,
                'export_timestamp': datetime.utcnow().isoformat()
            }
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self.take_snapshot()
                await asyncio.sleep(self.validation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.validation_interval)
    
    async def _take_baseline_snapshot(self) -> None:
        """Take baseline measurement."""
        self.baseline_metrics = await self.take_snapshot()
        self.logger.info(f"Baseline established: {self.baseline_metrics.real_progress}% real progress")
    
    def _calculate_progress_split(
        self, 
        validation_result: ValidationResult
    ) -> Tuple[float, float]:
        """Calculate real vs fake progress split."""
        
        # Base progress from validation result
        real_progress = validation_result.real_progress
        fake_progress = validation_result.fake_progress
        
        # Apply additional heuristics based on issues
        critical_issues = sum(1 for issue in validation_result.issues if issue.severity == Severity.CRITICAL)
        high_issues = sum(1 for issue in validation_result.issues if issue.severity == Severity.HIGH)
        
        # Reduce real progress based on issue severity
        real_progress_penalty = (critical_issues * 10) + (high_issues * 5)
        real_progress = max(0.0, real_progress - real_progress_penalty)
        
        # Adjust fake progress accordingly
        fake_progress = min(100.0, fake_progress + real_progress_penalty)
        
        return real_progress, fake_progress
    
    def _calculate_confidence(
        self,
        validation_result: ValidationResult,
        quality_metrics
    ) -> ProgressConfidence:
        """Calculate confidence level for progress assessment."""
        
        # Base confidence on authenticity score
        base_confidence = validation_result.authenticity_score
        
        # Adjust based on quality metrics
        if hasattr(quality_metrics, 'overall_score'):
            quality_factor = quality_metrics.overall_score / 100
            base_confidence = (base_confidence + quality_metrics.overall_score) / 2
        
        # Adjust based on number of issues
        issue_count = len(validation_result.issues)
        if issue_count > 10:
            base_confidence *= 0.8
        elif issue_count > 5:
            base_confidence *= 0.9
        
        # Convert to confidence enum
        if base_confidence >= 90:
            return ProgressConfidence.VERY_HIGH
        elif base_confidence >= 80:
            return ProgressConfidence.HIGH
        elif base_confidence >= 60:
            return ProgressConfidence.MEDIUM
        elif base_confidence >= 40:
            return ProgressConfidence.LOW
        else:
            return ProgressConfidence.VERY_LOW
    
    async def _check_for_alerts(self, snapshot: ProgressSnapshot) -> None:
        """Check for progress anomalies and generate alerts."""
        
        if len(self.snapshots) < 2:
            return  # Need at least 2 snapshots for comparison
        
        previous = self.snapshots[-2]
        
        # Check for authenticity drop
        authenticity_drop = previous.authenticity_rate - snapshot.authenticity_rate
        if authenticity_drop > self.alert_thresholds['authenticity_drop']:
            await self._create_alert(
                alert_type="authenticity_drop",
                severity=Severity.HIGH,
                message=f"Authenticity dropped by {authenticity_drop:.1f}%",
                details={
                    'previous_authenticity': previous.authenticity_rate,
                    'current_authenticity': snapshot.authenticity_rate,
                    'drop_amount': authenticity_drop
                }
            )
        
        # Check for fake progress spike
        fake_progress_increase = snapshot.fake_progress - previous.fake_progress
        if fake_progress_increase > self.alert_thresholds['fake_progress_spike']:
            await self._create_alert(
                alert_type="fake_progress_spike",
                severity=Severity.MEDIUM,
                message=f"Fake progress increased by {fake_progress_increase:.1f}%",
                details={
                    'previous_fake': previous.fake_progress,
                    'current_fake': snapshot.fake_progress,
                    'increase_amount': fake_progress_increase
                }
            )
        
        # Check for quality degradation
        quality_drop = previous.quality_score - snapshot.quality_score
        if quality_drop > self.alert_thresholds['quality_degradation']:
            await self._create_alert(
                alert_type="quality_degradation",
                severity=Severity.MEDIUM,
                message=f"Quality score dropped by {quality_drop:.1f} points",
                details={
                    'previous_quality': previous.quality_score,
                    'current_quality': snapshot.quality_score,
                    'drop_amount': quality_drop
                }
            )
        
        # Check for stagnation
        time_diff = (snapshot.timestamp - previous.timestamp).total_seconds()
        progress_diff = abs(snapshot.real_progress - previous.real_progress)
        
        if (time_diff > self.alert_thresholds['stagnation_time'] and 
            progress_diff < 1.0):
            await self._create_alert(
                alert_type="progress_stagnation",
                severity=Severity.LOW,
                message=f"No significant progress in {time_diff:.0f} seconds",
                details={
                    'stagnation_duration': time_diff,
                    'progress_change': progress_diff
                }
            )
    
    async def _create_alert(
        self,
        alert_type: str,
        severity: Severity,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Create a new progress alert."""
        
        alert_id = f"{alert_type}_{int(time.time())}"
        
        alert = ProgressAlert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Progress alert: {message}")
        
        # Maintain alert history size
        if len(self.alerts) > 100:  # Keep last 100 alerts
            self.alerts = self.alerts[-100:]
    
    def _get_source_files(self) -> List[Path]:
        """Get list of source files in project."""
        source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', '.go', '.rs'}
        
        source_files = []
        for file_path in self.project_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in source_extensions and
                not any(part.startswith('.') for part in file_path.parts) and
                'node_modules' not in file_path.parts and
                '__pycache__' not in file_path.parts):
                source_files.append(file_path)
        
        return source_files


# Utility functions

async def track_development_session(
    project_path: PathStr,
    duration_hours: int = 8,
    validation_interval: int = 60
) -> Dict[str, Any]:
    """
    Track a complete development session.
    
    Args:
        project_path: Path to project directory
        duration_hours: How long to track (in hours)
        validation_interval: How often to validate (in seconds)
        
    Returns:
        Session tracking results
    """
    tracker = RealProgressTracker(
        project_path=project_path,
        validation_interval=validation_interval
    )
    
    await tracker.start_monitoring()
    
    try:
        # Monitor for specified duration
        await asyncio.sleep(duration_hours * 3600)
    finally:
        await tracker.stop_monitoring()
    
    return tracker.get_progress_report()


async def analyze_project_authenticity(
    project_path: PathStr,
    save_report: bool = True
) -> Dict[str, Any]:
    """
    Perform one-time authenticity analysis of a project.
    
    Args:
        project_path: Path to project directory
        save_report: Whether to save detailed report
        
    Returns:
        Authenticity analysis results
    """
    tracker = RealProgressTracker(project_path)
    snapshot = await tracker.take_snapshot()
    
    analysis = {
        'timestamp': snapshot.timestamp.isoformat(),
        'authenticity_assessment': {
            'real_progress': snapshot.real_progress,
            'fake_progress': snapshot.fake_progress,
            'authenticity_rate': snapshot.authenticity_rate,
            'quality_score': snapshot.quality_score,
            'confidence': snapshot.confidence.value
        },
        'metrics': {
            'files_analyzed': snapshot.files_analyzed,
            'issues_found': snapshot.issues_found,
            'details': snapshot.details
        },
        'recommendations': _generate_authenticity_recommendations(snapshot)
    }
    
    if save_report:
        report_path = Path(project_path) / 'authenticity_report.json'
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    return analysis


def _generate_authenticity_recommendations(snapshot: ProgressSnapshot) -> List[str]:
    """Generate recommendations based on authenticity analysis."""
    recommendations = []
    
    if snapshot.authenticity_rate < 70:
        recommendations.append("Code authenticity is concerning. Review and complete placeholder implementations.")
    
    if snapshot.fake_progress > 30:
        recommendations.append("High levels of fake progress detected. Focus on substantial implementations.")
    
    if snapshot.quality_score < 60:
        recommendations.append("Code quality is below standards. Improve documentation, testing, and structure.")
    
    if snapshot.confidence == ProgressConfidence.LOW:
        recommendations.append("Low confidence in progress assessment. Manual review recommended.")
    
    if not recommendations:
        recommendations.append("Code authenticity looks good. Continue with current development practices.")
    
    return recommendations