#!/usr/bin/env python3
"""
Metrics Dashboard Widget - Real-time performance monitoring and analytics
with comprehensive system health and productivity metrics.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container, Grid
from textual.widgets import Static, Label, Button, DataTable, TabbedContent, TabPane
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
# from rich.chart import Chart  # Chart not available in current Rich version
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn


class MetricType(Enum):
    """Types of metrics"""
    PERFORMANCE = "performance"
    PRODUCTIVITY = "productivity"
    QUALITY = "quality"
    SYSTEM = "system"
    USER = "user"
    BUSINESS = "business"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    unit: str
    category: MetricType
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""
    
    @property
    def alert_level(self) -> Optional[AlertLevel]:
        """Determine alert level based on thresholds"""
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            return AlertLevel.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            return AlertLevel.WARNING
        return None
    
    def get_alert_icon(self) -> str:
        """Get icon for current alert level"""
        level = self.alert_level
        if level == AlertLevel.CRITICAL:
            return "🔴"
        elif level == AlertLevel.WARNING:
            return "🟡"
        else:
            return "🟢"


@dataclass
class SystemHealth:
    """Overall system health metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    @property
    def overall_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        # Weight different metrics
        cpu_score = max(0, 100 - self.cpu_usage)
        memory_score = max(0, 100 - self.memory_usage)
        disk_score = max(0, 100 - self.disk_usage)
        latency_score = max(0, 100 - min(self.network_latency * 10, 100))
        error_score = max(0, 100 - min(self.error_rate * 20, 100))
        
        return (cpu_score * 0.3 + memory_score * 0.3 + disk_score * 0.2 + 
                latency_score * 0.1 + error_score * 0.1)
    
    def get_health_status(self) -> Tuple[str, str]:
        """Get health status and color"""
        score = self.overall_health_score
        if score >= 90:
            return "Excellent", "green"
        elif score >= 75:
            return "Good", "yellow"
        elif score >= 50:
            return "Fair", "orange"
        else:
            return "Poor", "red"


@dataclass
class ProductivityMetrics:
    """Developer/AI productivity metrics"""
    tasks_completed: int = 0
    code_lines_generated: int = 0
    bugs_fixed: int = 0
    tests_written: int = 0
    commits_made: int = 0
    ai_interactions: int = 0
    average_task_time: float = 0.0  # minutes
    success_rate: float = 0.0  # percentage
    quality_score: float = 0.0
    
    @property
    def productivity_index(self) -> float:
        """Calculate overall productivity index"""
        # Normalize and weight different metrics
        normalized_tasks = min(self.tasks_completed / 10, 1.0) * 100
        normalized_code = min(self.code_lines_generated / 1000, 1.0) * 100
        normalized_quality = self.quality_score * 10
        
        return (normalized_tasks * 0.4 + normalized_code * 0.3 + 
                normalized_quality * 0.3)


class MetricHistoryChart(Static):
    """Chart widget showing metric history over time"""
    
    def __init__(self, metric_name: str, history_hours: int = 24) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.history_hours = history_hours
        self.data_points: List[Tuple[datetime, float]] = []
        
    def add_data_point(self, timestamp: datetime, value: float) -> None:
        """Add new data point to history"""
        self.data_points.append((timestamp, value))
        
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(hours=self.history_hours)
        self.data_points = [
            (ts, val) for ts, val in self.data_points 
            if ts > cutoff_time
        ]
        
        self.refresh()
    
    def render(self) -> Panel:
        """Render metric history chart"""
        if not self.data_points:
            return Panel("No data available", title=self.metric_name)
        
        # Create ASCII chart
        chart_lines = []
        if len(self.data_points) > 1:
            values = [point[1] for point in self.data_points]
            min_val = min(values)
            max_val = max(values)
            
            if max_val > min_val:
                # Normalize values to chart height
                chart_height = 10
                normalized = [
                    int((val - min_val) / (max_val - min_val) * chart_height)
                    for val in values
                ]
                
                # Draw chart
                for row in range(chart_height, -1, -1):
                    line = ""
                    for val in normalized:
                        if val >= row:
                            line += "█"
                        else:
                            line += " "
                    chart_lines.append(line)
                
                # Add axis labels
                chart_lines.append("-" * len(normalized))
                chart_lines.append(f"Min: {min_val:.1f} | Max: {max_val:.1f} | Current: {values[-1]:.1f}")
            else:
                chart_lines.append(f"Constant value: {values[0]:.1f}")
        else:
            chart_lines.append(f"Single value: {self.data_points[0][1]:.1f}")
        
        content = "\n".join(chart_lines)
        return Panel(content, title=f"📈 {self.metric_name} ({self.history_hours}h)")


class SystemHealthWidget(Static):
    """Widget displaying system health metrics"""
    
    def __init__(self) -> None:
        super().__init__()
        self.health_data = SystemHealth()
        
    def render(self) -> Panel:
        """Render system health panel"""
        table = Table("Component", "Usage", "Status", show_header=True)
        
        # CPU
        cpu_bar = "█" * int(self.health_data.cpu_usage / 5) + "░" * (20 - int(self.health_data.cpu_usage / 5))
        cpu_status = "🔴 High" if self.health_data.cpu_usage > 80 else "🟡 Medium" if self.health_data.cpu_usage > 60 else "🟢 Normal"
        table.add_row("CPU", f"[{cpu_bar}] {self.health_data.cpu_usage:.1f}%", cpu_status)
        
        # Memory
        mem_bar = "█" * int(self.health_data.memory_usage / 5) + "░" * (20 - int(self.health_data.memory_usage / 5))
        mem_status = "🔴 High" if self.health_data.memory_usage > 85 else "🟡 Medium" if self.health_data.memory_usage > 70 else "🟢 Normal"
        table.add_row("Memory", f"[{mem_bar}] {self.health_data.memory_usage:.1f}%", mem_status)
        
        # Disk
        disk_bar = "█" * int(self.health_data.disk_usage / 5) + "░" * (20 - int(self.health_data.disk_usage / 5))
        disk_status = "🔴 High" if self.health_data.disk_usage > 90 else "🟡 Medium" if self.health_data.disk_usage > 75 else "🟢 Normal"
        table.add_row("Disk", f"[{disk_bar}] {self.health_data.disk_usage:.1f}%", disk_status)
        
        # Network
        network_status = "🔴 Slow" if self.health_data.network_latency > 100 else "🟡 Fair" if self.health_data.network_latency > 50 else "🟢 Fast"
        table.add_row("Network", f"{self.health_data.network_latency:.1f}ms", network_status)
        
        # Overall health
        health_status, health_color = self.health_data.get_health_status()
        table.add_section()
        table.add_row("Overall Health", f"{self.health_data.overall_health_score:.1f}/100", f"🎯 {health_status}")
        
        return Panel(table, title="🖥️ System Health", border_style="blue")
    
    def update_health(self, health_data: SystemHealth) -> None:
        """Update health data"""
        self.health_data = health_data
        self.refresh()


class ProductivityWidget(Static):
    """Widget displaying productivity metrics"""
    
    def __init__(self) -> None:
        super().__init__()
        self.productivity = ProductivityMetrics()
        
    def render(self) -> Panel:
        """Render productivity panel"""
        table = Table("Metric", "Value", "Trend", show_header=True)
        
        # Tasks
        table.add_row(
            "Tasks Completed", 
            str(self.productivity.tasks_completed),
            "📈" if self.productivity.tasks_completed > 0 else "➖"
        )
        
        # Code generation
        table.add_row(
            "Code Lines Generated",
            f"{self.productivity.code_lines_generated:,}",
            "📈" if self.productivity.code_lines_generated > 0 else "➖"
        )
        
        # Quality metrics
        table.add_row(
            "Quality Score",
            f"{self.productivity.quality_score:.1f}/10",
            "🎯" if self.productivity.quality_score >= 7 else "📉" if self.productivity.quality_score < 5 else "➖"
        )
        
        # Success rate
        table.add_row(
            "Success Rate",
            f"{self.productivity.success_rate:.1f}%",
            "✅" if self.productivity.success_rate >= 80 else "⚠️" if self.productivity.success_rate >= 60 else "❌"
        )
        
        # Average task time
        if self.productivity.average_task_time > 0:
            time_str = f"{self.productivity.average_task_time:.1f} min"
            time_trend = "🚀" if self.productivity.average_task_time < 30 else "⏳" if self.productivity.average_task_time < 60 else "🐌"
            table.add_row("Avg Task Time", time_str, time_trend)
        
        # AI interactions
        table.add_row(
            "AI Interactions",
            str(self.productivity.ai_interactions),
            "🤖" if self.productivity.ai_interactions > 0 else "💤"
        )
        
        # Productivity index
        table.add_section()
        index_score = self.productivity.productivity_index
        index_status = "🔥 Excellent" if index_score >= 80 else "✨ Good" if index_score >= 60 else "📈 Improving" if index_score >= 40 else "🎯 Needs Focus"
        table.add_row("Productivity Index", f"{index_score:.1f}/100", index_status)
        
        return Panel(table, title="🚀 Productivity Metrics", border_style="green")
    
    def update_productivity(self, productivity: ProductivityMetrics) -> None:
        """Update productivity data"""
        self.productivity = productivity
        self.refresh()


class AlertsWidget(Static):
    """Widget displaying current alerts and warnings"""
    
    def __init__(self) -> None:
        super().__init__()
        self.alerts: List[Tuple[AlertLevel, str, datetime]] = []
        
    def render(self) -> Panel:
        """Render alerts panel"""
        if not self.alerts:
            content = Text("No active alerts ✅", style="green")
        else:
            table = Table("Level", "Message", "Time", show_header=True)
            
            for level, message, timestamp in sorted(self.alerts, key=lambda x: x[2], reverse=True):
                level_icon = {
                    AlertLevel.CRITICAL: "🔴",
                    AlertLevel.WARNING: "🟡",
                    AlertLevel.INFO: "🔵"
                }.get(level, "❓")
                
                time_ago = datetime.now() - timestamp
                if time_ago.seconds < 60:
                    time_str = "just now"
                elif time_ago.seconds < 3600:
                    time_str = f"{time_ago.seconds // 60}m ago"
                else:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                
                table.add_row(f"{level_icon} {level.value.title()}", message, time_str)
        
        return Panel(table if self.alerts else content, title="🚨 Alerts", border_style="red" if self.alerts else "green")
    
    def add_alert(self, level: AlertLevel, message: str) -> None:
        """Add new alert"""
        self.alerts.append((level, message, datetime.now()))
        
        # Keep only recent alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            (lvl, msg, ts) for lvl, msg, ts in self.alerts 
            if ts > cutoff_time
        ]
        
        self.refresh()
    
    def clear_alerts(self) -> None:
        """Clear all alerts"""
        self.alerts.clear()
        self.refresh()


class MetricsDashboardWidget(Vertical):
    """Main metrics dashboard with multiple tabs and real-time updates"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Widgets
        self.system_health_widget: Optional[SystemHealthWidget] = None
        self.productivity_widget: Optional[ProductivityWidget] = None
        self.alerts_widget: Optional[AlertsWidget] = None
        
        # Charts
        self.cpu_chart: Optional[MetricHistoryChart] = None
        self.memory_chart: Optional[MetricHistoryChart] = None
        self.productivity_chart: Optional[MetricHistoryChart] = None
        
        # Data
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.monitoring_active = False
        
    def compose(self) -> ComposeResult:
        """Compose metrics dashboard"""
        yield Label("📊 Metrics Dashboard", classes="header")
        
        with TabbedContent():
            # System Health Tab
            with TabPane("System Health", id="system-tab"):
                with Grid(classes="metrics-grid"):
                    self.system_health_widget = SystemHealthWidget()
                    yield self.system_health_widget
                    
                    self.alerts_widget = AlertsWidget()
                    yield self.alerts_widget
                
                with Horizontal():
                    self.cpu_chart = MetricHistoryChart("CPU Usage %")
                    yield self.cpu_chart
                    
                    self.memory_chart = MetricHistoryChart("Memory Usage %")
                    yield self.memory_chart
            
            # Productivity Tab
            with TabPane("Productivity", id="productivity-tab"):
                with Grid(classes="productivity-grid"):
                    self.productivity_widget = ProductivityWidget()
                    yield self.productivity_widget
                    
                    self.productivity_chart = MetricHistoryChart("Productivity Index")
                    yield self.productivity_chart
            
            # Quality Tab
            with TabPane("Quality", id="quality-tab"):
                yield Static("Quality metrics coming soon...", classes="coming-soon")
            
            # Performance Tab
            with TabPane("Performance", id="performance-tab"):
                yield Static("Performance metrics coming soon...", classes="coming-soon")
        
        # Control buttons
        with Horizontal(classes="dashboard-controls"):
            yield Button("🔄 Refresh", id="refresh-metrics")
            yield Button("📤 Export", id="export-metrics")
            yield Button("⚙️ Settings", id="metrics-settings")
            yield Button("🚨 Clear Alerts", id="clear-alerts")
    
    def on_mount(self) -> None:
        """Start monitoring when mounted"""
        self.start_monitoring()
    
    @work(exclusive=True)
    async def start_monitoring(self) -> None:
        """Start real-time metrics monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Update system health
                await self._update_system_health()
                
                # Update productivity metrics
                await self._update_productivity_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                # Log error and continue
                await asyncio.sleep(30)
    
    async def _update_system_health(self) -> None:
        """Update system health metrics"""
        # This would integrate with actual system monitoring
        # For now, simulate data
        import random
        
        health = SystemHealth(
            cpu_usage=random.uniform(10, 80),
            memory_usage=random.uniform(30, 90),
            disk_usage=random.uniform(40, 85),
            network_latency=random.uniform(5, 50),
            error_rate=random.uniform(0, 5),
            uptime=random.uniform(95, 100)
        )
        
        if self.system_health_widget:
            self.system_health_widget.update_health(health)
        
        # Update charts
        now = datetime.now()
        if self.cpu_chart:
            self.cpu_chart.add_data_point(now, health.cpu_usage)
        if self.memory_chart:
            self.memory_chart.add_data_point(now, health.memory_usage)
    
    async def _update_productivity_metrics(self) -> None:
        """Update productivity metrics"""
        # This would integrate with actual productivity tracking
        # For now, simulate data
        import random
        
        productivity = ProductivityMetrics(
            tasks_completed=random.randint(0, 20),
            code_lines_generated=random.randint(100, 2000),
            bugs_fixed=random.randint(0, 10),
            tests_written=random.randint(0, 15),
            commits_made=random.randint(0, 8),
            ai_interactions=random.randint(5, 50),
            average_task_time=random.uniform(15, 120),
            success_rate=random.uniform(70, 95),
            quality_score=random.uniform(6, 9.5)
        )
        
        if self.productivity_widget:
            self.productivity_widget.update_productivity(productivity)
        
        # Update productivity chart
        if self.productivity_chart:
            self.productivity_chart.add_data_point(
                datetime.now(), 
                productivity.productivity_index
            )
    
    async def _check_alerts(self) -> None:
        """Check for and generate alerts"""
        if not self.alerts_widget:
            return
        
        # Check system health alerts
        if self.system_health_widget:
            health = self.system_health_widget.health_data
            
            if health.cpu_usage > 90:
                self.alerts_widget.add_alert(AlertLevel.CRITICAL, f"CPU usage extremely high: {health.cpu_usage:.1f}%")
            elif health.cpu_usage > 80:
                self.alerts_widget.add_alert(AlertLevel.WARNING, f"CPU usage high: {health.cpu_usage:.1f}%")
            
            if health.memory_usage > 95:
                self.alerts_widget.add_alert(AlertLevel.CRITICAL, f"Memory usage critical: {health.memory_usage:.1f}%")
            elif health.memory_usage > 85:
                self.alerts_widget.add_alert(AlertLevel.WARNING, f"Memory usage high: {health.memory_usage:.1f}%")
            
            if health.disk_usage > 95:
                self.alerts_widget.add_alert(AlertLevel.CRITICAL, f"Disk space critical: {health.disk_usage:.1f}%")
    
    # Event handlers
    @on(Button.Pressed, "#refresh-metrics")
    def refresh_metrics(self) -> None:
        """Refresh all metrics immediately"""
        asyncio.create_task(self._update_system_health())
        asyncio.create_task(self._update_productivity_metrics())
    
    @on(Button.Pressed, "#export-metrics")
    def export_metrics(self) -> None:
        """Export metrics data"""
        self.post_message(ExportMetricsMessage())
    
    @on(Button.Pressed, "#metrics-settings")
    def show_settings(self) -> None:
        """Show metrics settings"""
        self.post_message(ShowMetricsSettingsMessage())
    
    @on(Button.Pressed, "#clear-alerts")
    def clear_all_alerts(self) -> None:
        """Clear all alerts"""
        if self.alerts_widget:
            self.alerts_widget.clear_alerts()
    
    def stop_monitoring(self) -> None:
        """Stop metrics monitoring"""
        self.monitoring_active = False
    
    def add_custom_metric(self, metric: Metric) -> None:
        """Add custom metric to dashboard"""
        # Store metric in history
        if metric.name not in self.metrics_history:
            self.metrics_history[metric.name] = []
        
        self.metrics_history[metric.name].append((metric.timestamp, metric.value))
        
        # Check for alerts
        if metric.alert_level:
            if self.alerts_widget:
                self.alerts_widget.add_alert(
                    metric.alert_level, 
                    f"{metric.name}: {metric.value} {metric.unit}"
                )


# Message classes for dashboard events
class ExportMetricsMessage(Message):
    """Message to export metrics data"""
    pass


class ShowMetricsSettingsMessage(Message):
    """Message to show metrics settings"""
    pass