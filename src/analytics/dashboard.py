"""
Analytics Dashboard Components for Real-time Performance Monitoring.

Interactive dashboard system for visualizing performance metrics, trends,
bottlenecks, and optimization recommendations with real-time updates.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict
from enum import Enum

from .models import (
    PerformanceMetrics, BottleneckAnalysis, OptimizationRecommendation,
    TrendAnalysis, PerformanceAlert, PerformanceSnapshot,
    AnalyticsConfiguration
)
from .engine import PerformanceAnalyticsEngine
from .collector import MetricsCollector
from .optimizer import PerformanceOptimizer
from .predictor import PerformancePredictor


class DashboardTheme(str, Enum):
    """Dashboard themes."""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"


class ChartType(str, Enum):
    """Chart types for data visualization."""
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    PIE = "pie"


class DashboardWidget:
    """Base class for dashboard widgets."""
    
    def __init__(
        self,
        widget_id: str,
        title: str,
        widget_type: str,
        refresh_interval: int = 30
    ):
        """Initialize dashboard widget."""
        self.widget_id = widget_id
        self.title = title
        self.widget_type = widget_type
        self.refresh_interval = refresh_interval
        self.last_updated = None
        self.data = {}
        self.config = {}
    
    async def update_data(self, **kwargs) -> None:
        """Update widget data."""
        self.data.update(kwargs)
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary for serialization."""
        return {
            'id': self.widget_id,
            'title': self.title,
            'type': self.widget_type,
            'data': self.data,
            'config': self.config,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'refresh_interval': self.refresh_interval
        }


class MetricsWidget(DashboardWidget):
    """Widget for displaying performance metrics."""
    
    def __init__(
        self,
        widget_id: str,
        title: str,
        metric_names: List[str],
        chart_type: ChartType = ChartType.LINE,
        time_window: timedelta = timedelta(hours=1)
    ):
        super().__init__(widget_id, title, "metrics")
        self.metric_names = metric_names
        self.chart_type = chart_type
        self.time_window = time_window
        self.config = {
            'chart_type': chart_type.value,
            'time_window_hours': time_window.total_seconds() / 3600,
            'metrics': metric_names
        }
    
    async def update_from_metrics(
        self,
        metrics_history: List[PerformanceMetrics]
    ) -> None:
        """Update widget with metrics data."""
        # Filter by time window
        cutoff_time = datetime.utcnow() - self.time_window
        recent_metrics = [
            m for m in metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        # Prepare chart data
        chart_data = {
            'labels': [m.timestamp.isoformat() for m in recent_metrics],
            'datasets': []
        }
        
        for metric_name in self.metric_names:
            values = []
            for metric in recent_metrics:
                if hasattr(metric, metric_name):
                    values.append(getattr(metric, metric_name))
                else:
                    values.append(0)
            
            dataset = {
                'label': metric_name.replace('_', ' ').title(),
                'data': values,
                'borderColor': self._get_metric_color(metric_name),
                'backgroundColor': self._get_metric_color(metric_name, alpha=0.2)
            }
            chart_data['datasets'].append(dataset)
        
        await self.update_data(chart_data=chart_data)
    
    def _get_metric_color(self, metric_name: str, alpha: float = 1.0) -> str:
        """Get color for metric visualization."""
        color_map = {
            'cpu_percent': f'rgba(255, 99, 132, {alpha})',
            'memory_percent': f'rgba(54, 162, 235, {alpha})',
            'throughput': f'rgba(75, 192, 192, {alpha})',
            'latency_p95': f'rgba(255, 205, 86, {alpha})',
            'error_rate': f'rgba(255, 159, 64, {alpha})',
            'cache_hit_rate': f'rgba(153, 102, 255, {alpha})'
        }
        return color_map.get(metric_name, f'rgba(128, 128, 128, {alpha})')


class AlertsWidget(DashboardWidget):
    """Widget for displaying performance alerts."""
    
    def __init__(
        self,
        widget_id: str,
        title: str = "Performance Alerts",
        max_alerts: int = 10
    ):
        super().__init__(widget_id, title, "alerts")
        self.max_alerts = max_alerts
        self.config = {'max_alerts': max_alerts}
    
    async def update_from_alerts(
        self,
        alerts: List[PerformanceAlert]
    ) -> None:
        """Update widget with alerts data."""
        # Sort by severity and time
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (
                ['low', 'medium', 'high', 'critical'].index(a.severity.value),
                a.created_at
            ),
            reverse=True
        )
        
        # Take only recent alerts
        recent_alerts = sorted_alerts[:self.max_alerts]
        
        # Prepare alert data
        alert_data = []
        for alert in recent_alerts:
            alert_data.append({
                'id': str(alert.id),
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.value,
                'created_at': alert.created_at.isoformat(),
                'affected_component': alert.affected_component,
                'is_active': alert.is_active()
            })
        
        # Summary statistics
        summary = {
            'total_alerts': len(alerts),
            'active_alerts': len([a for a in alerts if a.is_active()]),
            'critical_alerts': len([a for a in alerts if a.severity.value == 'critical']),
            'by_severity': {}
        }
        
        for severity in ['low', 'medium', 'high', 'critical']:
            summary['by_severity'][severity] = len([
                a for a in alerts if a.severity.value == severity
            ])
        
        await self.update_data(
            alerts=alert_data,
            summary=summary
        )


class BottleneckWidget(DashboardWidget):
    """Widget for displaying bottleneck analysis."""
    
    def __init__(
        self,
        widget_id: str,
        title: str = "Performance Bottlenecks"
    ):
        super().__init__(widget_id, title, "bottlenecks")
    
    async def update_from_bottlenecks(
        self,
        bottlenecks: List[BottleneckAnalysis]
    ) -> None:
        """Update widget with bottleneck data."""
        # Sort by performance impact
        sorted_bottlenecks = sorted(
            bottlenecks,
            key=lambda b: b.performance_impact,
            reverse=True
        )
        
        # Prepare bottleneck data
        bottleneck_data = []
        for bottleneck in sorted_bottlenecks:
            bottleneck_data.append({
                'id': bottleneck.id,
                'type': bottleneck.bottleneck_type,
                'component': bottleneck.component,
                'severity': bottleneck.severity.value,
                'impact': bottleneck.performance_impact,
                'root_cause': bottleneck.root_cause,
                'immediate_actions': bottleneck.immediate_actions,
                'detected_at': bottleneck.detected_at.isoformat()
            })
        
        # Create impact chart data
        impact_chart = {
            'labels': [b.component for b in sorted_bottlenecks[:5]],  # Top 5
            'data': [b.performance_impact for b in sorted_bottlenecks[:5]],
            'backgroundColor': [
                'rgba(255, 99, 132, 0.8)' if b.severity.value == 'critical' else
                'rgba(255, 205, 86, 0.8)' if b.severity.value == 'high' else
                'rgba(54, 162, 235, 0.8)'
                for b in sorted_bottlenecks[:5]
            ]
        }
        
        await self.update_data(
            bottlenecks=bottleneck_data,
            impact_chart=impact_chart,
            total_bottlenecks=len(bottlenecks)
        )


class OptimizationWidget(DashboardWidget):
    """Widget for displaying optimization recommendations."""
    
    def __init__(
        self,
        widget_id: str,
        title: str = "Optimization Opportunities",
        max_recommendations: int = 5
    ):
        super().__init__(widget_id, title, "optimization")
        self.max_recommendations = max_recommendations
    
    async def update_from_recommendations(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> None:
        """Update widget with optimization data."""
        # Sort by estimated improvement and priority
        sorted_recs = sorted(
            recommendations,
            key=lambda r: (
                ['low', 'medium', 'high', 'critical'].index(r.priority.value),
                r.estimated_improvement
            ),
            reverse=True
        )
        
        # Take top recommendations
        top_recs = sorted_recs[:self.max_recommendations]
        
        # Prepare recommendation data
        rec_data = []
        for rec in top_recs:
            rec_data.append({
                'id': str(rec.id),
                'title': rec.title,
                'description': rec.description,
                'type': rec.optimization_type.value,
                'estimated_improvement': rec.estimated_improvement,
                'confidence_score': rec.confidence_score,
                'priority': rec.priority.value,
                'risk_level': rec.risk_level,
                'implementation_steps': rec.implementation_steps[:3],  # First 3 steps
                'created_at': rec.created_at.isoformat()
            })
        
        # Create improvement potential chart
        improvement_chart = {
            'labels': [r.title[:30] + '...' if len(r.title) > 30 else r.title for r in top_recs],
            'data': [r.estimated_improvement for r in top_recs],
            'backgroundColor': [
                'rgba(75, 192, 192, 0.8)' if r.confidence_score > 80 else
                'rgba(255, 205, 86, 0.8)' if r.confidence_score > 60 else
                'rgba(255, 159, 64, 0.8)'
                for r in top_recs
            ]
        }
        
        await self.update_data(
            recommendations=rec_data,
            improvement_chart=improvement_chart,
            total_potential_improvement=sum(r.estimated_improvement for r in recommendations)
        )


class TrendWidget(DashboardWidget):
    """Widget for displaying performance trends and predictions."""
    
    def __init__(
        self,
        widget_id: str,
        title: str = "Performance Trends",
        metric_name: str = "cpu_percent",
        prediction_hours: int = 6
    ):
        super().__init__(widget_id, title, "trends")
        self.metric_name = metric_name
        self.prediction_hours = prediction_hours
        self.config = {
            'metric_name': metric_name,
            'prediction_hours': prediction_hours
        }
    
    async def update_from_trend_analysis(
        self,
        trend_analysis: TrendAnalysis,
        historical_metrics: List[PerformanceMetrics],
        predictions: List[tuple] = None
    ) -> None:
        """Update widget with trend analysis data."""
        # Prepare historical data
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_metrics = [
            m for m in historical_metrics
            if m.timestamp >= cutoff_time and hasattr(m, self.metric_name)
        ]
        
        historical_data = {
            'labels': [m.timestamp.isoformat() for m in recent_metrics],
            'values': [getattr(m, self.metric_name) for m in recent_metrics]
        }
        
        # Prepare prediction data
        prediction_data = {}
        if predictions:
            prediction_data = {
                'labels': [p[0].isoformat() for p in predictions],
                'values': [p[1] for p in predictions]
            }
        
        # Trend summary
        trend_summary = {
            'direction': trend_analysis.trend_direction,
            'strength': trend_analysis.trend_strength,
            'mean_value': trend_analysis.mean_value,
            'std_deviation': trend_analysis.std_deviation,
            'data_points': trend_analysis.data_points,
            'significant_changes': len(trend_analysis.significant_changes),
            'anomalies_detected': trend_analysis.anomalies_detected
        }
        
        await self.update_data(
            historical_data=historical_data,
            prediction_data=prediction_data,
            trend_summary=trend_summary,
            metric_name=self.metric_name
        )


class PerformanceOverviewWidget(DashboardWidget):
    """Widget for displaying overall performance overview."""
    
    def __init__(
        self,
        widget_id: str,
        title: str = "Performance Overview"
    ):
        super().__init__(widget_id, title, "overview")
    
    async def update_from_snapshot(
        self,
        snapshot: PerformanceSnapshot
    ) -> None:
        """Update widget with performance snapshot."""
        # Key performance indicators
        kpis = {
            'composite_score': snapshot.composite_score,
            'cpu_usage': snapshot.metrics.cpu_percent,
            'memory_usage': snapshot.metrics.memory_percent,
            'throughput': snapshot.metrics.throughput,
            'error_rate': snapshot.metrics.error_rate * 100,  # Convert to percentage
            'response_time': snapshot.metrics.ai_response_time
        }
        
        # Status indicators
        status = {
            'overall_health': 'good' if snapshot.composite_score > 80 else 
                            'warning' if snapshot.composite_score > 60 else 'critical',
            'active_alerts': len(snapshot.active_alerts),
            'critical_issues': len(snapshot.get_critical_issues()),
            'bottlenecks': len(snapshot.bottlenecks),
            'optimization_opportunities': len(snapshot.optimization_opportunities)
        }
        
        # Resource utilization gauge data
        resource_gauges = [
            {
                'name': 'CPU',
                'value': snapshot.metrics.cpu_percent,
                'max': 100,
                'color': self._get_gauge_color(snapshot.metrics.cpu_percent, 80, 95)
            },
            {
                'name': 'Memory',
                'value': snapshot.metrics.memory_percent,
                'max': 100,
                'color': self._get_gauge_color(snapshot.metrics.memory_percent, 85, 95)
            },
            {
                'name': 'Disk',
                'value': snapshot.metrics.disk_percent,
                'max': 100,
                'color': self._get_gauge_color(snapshot.metrics.disk_percent, 90, 95)
            }
        ]
        
        await self.update_data(
            kpis=kpis,
            status=status,
            resource_gauges=resource_gauges,
            snapshot_time=snapshot.timestamp.isoformat()
        )
    
    def _get_gauge_color(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get color for gauge based on value and thresholds."""
        if value >= critical_threshold:
            return '#dc3545'  # Red
        elif value >= warning_threshold:
            return '#ffc107'  # Yellow
        else:
            return '#28a745'  # Green


class AnalyticsDashboard:
    """
    Main analytics dashboard for performance monitoring.
    
    Features:
    - Real-time performance metrics visualization
    - Interactive charts and graphs
    - Alert notifications and management
    - Bottleneck analysis displays
    - Optimization recommendation panels
    - Trend analysis and forecasting
    - Customizable dashboard layouts
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        theme: DashboardTheme = DashboardTheme.LIGHT
    ):
        """Initialize the analytics dashboard."""
        self.config = config or AnalyticsConfiguration()
        self.theme = theme
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Dashboard components
        self.widgets: Dict[str, DashboardWidget] = {}
        self.layouts: Dict[str, List[Dict[str, Any]]] = {}
        
        # Component references
        self.analytics_engine: Optional[PerformanceAnalyticsEngine] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.optimizer: Optional[PerformanceOptimizer] = None
        self.predictor: Optional[PerformancePredictor] = None
        
        # Update scheduling
        self.is_updating = False
        self.update_task: Optional[asyncio.Task] = None
        self.update_interval = 30  # seconds
        
        # Dashboard state
        self.dashboard_state = {
            'last_updated': None,
            'active_filters': {},
            'selected_time_range': timedelta(hours=1),
            'auto_refresh': True
        }
    
    async def initialize(
        self,
        analytics_engine: PerformanceAnalyticsEngine,
        metrics_collector: MetricsCollector,
        optimizer: PerformanceOptimizer,
        predictor: PerformancePredictor
    ) -> None:
        """Initialize dashboard with component references."""
        try:
            self.logger.info("Initializing Analytics Dashboard...")
            
            # Store component references
            self.analytics_engine = analytics_engine
            self.metrics_collector = metrics_collector
            self.optimizer = optimizer
            self.predictor = predictor
            
            # Create default widgets
            await self._create_default_widgets()
            
            # Create default layouts
            await self._create_default_layouts()
            
            self.logger.info("Analytics Dashboard initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {e}")
            raise
    
    async def start_updates(self) -> None:
        """Start automatic dashboard updates."""
        if self.is_updating:
            self.logger.warning("Dashboard updates already active")
            return
        
        self.logger.info("Starting dashboard updates...")
        self.is_updating = True
        
        # Start update loop
        self.update_task = asyncio.create_task(self._update_loop())
    
    async def stop_updates(self) -> None:
        """Stop automatic dashboard updates."""
        if not self.is_updating:
            return
        
        self.logger.info("Stopping dashboard updates...")
        self.is_updating = False
        
        # Cancel update task
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
    
    async def update_all_widgets(self) -> None:
        """Update all dashboard widgets with latest data."""
        try:
            if not all([self.analytics_engine, self.metrics_collector]):
                self.logger.warning("Dashboard components not fully initialized")
                return
            
            # Get latest performance snapshot
            snapshot = await self.analytics_engine.get_performance_snapshot()
            
            # Get metrics history
            metrics_history = await self.metrics_collector.get_metrics_history(
                limit=1000,
                start_time=datetime.utcnow() - self.dashboard_state['selected_time_range']
            )
            
            # Update each widget
            for widget_id, widget in self.widgets.items():
                try:
                    if isinstance(widget, PerformanceOverviewWidget):
                        await widget.update_from_snapshot(snapshot)
                    
                    elif isinstance(widget, MetricsWidget):
                        await widget.update_from_metrics(metrics_history)
                    
                    elif isinstance(widget, AlertsWidget):
                        await widget.update_from_alerts(snapshot.active_alerts)
                    
                    elif isinstance(widget, BottleneckWidget):
                        await widget.update_from_bottlenecks(snapshot.bottlenecks)
                    
                    elif isinstance(widget, OptimizationWidget):
                        await widget.update_from_recommendations(snapshot.optimization_opportunities)
                    
                    elif isinstance(widget, TrendWidget) and self.predictor:
                        # Get trend analysis for the widget's metric
                        try:
                            trend = await self.analytics_engine.analyze_trends(
                                widget.metric_name, timedelta(hours=6)
                            )
                            
                            # Get predictions
                            predictions = await self.predictor.predict_metric_trend(
                                widget.metric_name, metrics_history[-100:],
                                timedelta(hours=widget.prediction_hours)
                            )
                            
                            await widget.update_from_trend_analysis(
                                trend, metrics_history, predictions.predicted_values
                            )
                        except Exception as e:
                            self.logger.warning(f"Error updating trend widget {widget_id}: {e}")
                
                except Exception as e:
                    self.logger.error(f"Error updating widget {widget_id}: {e}")
            
            # Update dashboard state
            self.dashboard_state['last_updated'] = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard widgets: {e}")
    
    def get_dashboard_data(self, layout_name: str = "default") -> Dict[str, Any]:
        """Get complete dashboard data for rendering."""
        layout = self.layouts.get(layout_name, self.layouts.get("default", []))
        
        dashboard_data = {
            'layout': layout,
            'widgets': {
                widget_id: widget.to_dict()
                for widget_id, widget in self.widgets.items()
            },
            'theme': self.theme.value,
            'state': self.dashboard_state.copy(),
            'config': {
                'update_interval': self.update_interval,
                'auto_refresh': self.dashboard_state['auto_refresh']
            }
        }
        
        return dashboard_data
    
    def add_widget(self, widget: DashboardWidget) -> None:
        """Add a widget to the dashboard."""
        self.widgets[widget.widget_id] = widget
        self.logger.info(f"Added widget: {widget.widget_id}")
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard."""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            self.logger.info(f"Removed widget: {widget_id}")
            return True
        return False
    
    def set_theme(self, theme: DashboardTheme) -> None:
        """Set dashboard theme."""
        self.theme = theme
        self.logger.info(f"Dashboard theme set to: {theme.value}")
    
    def set_time_range(self, time_range: timedelta) -> None:
        """Set time range for data display."""
        self.dashboard_state['selected_time_range'] = time_range
        self.logger.info(f"Dashboard time range set to: {time_range}")
    
    # Private methods
    
    async def _update_loop(self) -> None:
        """Main dashboard update loop."""
        while self.is_updating:
            try:
                await self.update_all_widgets()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _create_default_widgets(self) -> None:
        """Create default dashboard widgets."""
        # Performance overview
        overview_widget = PerformanceOverviewWidget(
            "performance_overview",
            "System Performance Overview"
        )
        self.add_widget(overview_widget)
        
        # Key metrics chart
        metrics_widget = MetricsWidget(
            "key_metrics",
            "Key Performance Metrics",
            ["cpu_percent", "memory_percent", "throughput", "latency_p95"],
            ChartType.LINE,
            timedelta(hours=1)
        )
        self.add_widget(metrics_widget)
        
        # Alerts panel
        alerts_widget = AlertsWidget(
            "performance_alerts",
            "Performance Alerts",
            max_alerts=10
        )
        self.add_widget(alerts_widget)
        
        # Bottlenecks analysis
        bottleneck_widget = BottleneckWidget(
            "bottleneck_analysis",
            "Performance Bottlenecks"
        )
        self.add_widget(bottleneck_widget)
        
        # Optimization recommendations
        optimization_widget = OptimizationWidget(
            "optimization_opportunities",
            "Optimization Opportunities",
            max_recommendations=5
        )
        self.add_widget(optimization_widget)
        
        # CPU trend analysis
        cpu_trend_widget = TrendWidget(
            "cpu_trend",
            "CPU Usage Trend & Prediction",
            "cpu_percent",
            prediction_hours=6
        )
        self.add_widget(cpu_trend_widget)
        
        # Memory trend analysis
        memory_trend_widget = TrendWidget(
            "memory_trend",
            "Memory Usage Trend & Prediction",
            "memory_percent",
            prediction_hours=6
        )
        self.add_widget(memory_trend_widget)
    
    async def _create_default_layouts(self) -> None:
        """Create default dashboard layouts."""
        # Default layout - overview focused
        self.layouts["default"] = [
            {
                "widget_id": "performance_overview",
                "position": {"x": 0, "y": 0, "width": 12, "height": 4}
            },
            {
                "widget_id": "key_metrics",
                "position": {"x": 0, "y": 4, "width": 8, "height": 6}
            },
            {
                "widget_id": "performance_alerts",
                "position": {"x": 8, "y": 4, "width": 4, "height": 6}
            },
            {
                "widget_id": "bottleneck_analysis",
                "position": {"x": 0, "y": 10, "width": 6, "height": 6}
            },
            {
                "widget_id": "optimization_opportunities",
                "position": {"x": 6, "y": 10, "width": 6, "height": 6}
            }
        ]
        
        # Detailed layout - with trends
        self.layouts["detailed"] = [
            {
                "widget_id": "performance_overview",
                "position": {"x": 0, "y": 0, "width": 12, "height": 3}
            },
            {
                "widget_id": "key_metrics",
                "position": {"x": 0, "y": 3, "width": 6, "height": 5}
            },
            {
                "widget_id": "performance_alerts",
                "position": {"x": 6, "y": 3, "width": 6, "height": 5}
            },
            {
                "widget_id": "cpu_trend",
                "position": {"x": 0, "y": 8, "width": 6, "height": 5}
            },
            {
                "widget_id": "memory_trend",
                "position": {"x": 6, "y": 8, "width": 6, "height": 5}
            },
            {
                "widget_id": "bottleneck_analysis",
                "position": {"x": 0, "y": 13, "width": 6, "height": 5}
            },
            {
                "widget_id": "optimization_opportunities",
                "position": {"x": 6, "y": 13, "width": 6, "height": 5}
            }
        ]
        
        # Compact layout - minimal view
        self.layouts["compact"] = [
            {
                "widget_id": "performance_overview",
                "position": {"x": 0, "y": 0, "width": 12, "height": 6}
            },
            {
                "widget_id": "performance_alerts",
                "position": {"x": 0, "y": 6, "width": 6, "height": 4}
            },
            {
                "widget_id": "bottleneck_analysis",
                "position": {"x": 6, "y": 6, "width": 6, "height": 4}
            }
        ]