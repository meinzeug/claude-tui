#!/usr/bin/env python3
"""
Monitoring Dashboard for Claude-TUI MCP Integration
Real-time monitoring of swarm operations and performance metrics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Static, DataTable, ProgressBar,
    TabbedContent, TabPane, Log, Button, Label
)
from textual.reactive import reactive
from textual.timer import Timer
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.live import Live
from rich.console import Console

from ..mcp.server import MCPServerClient, SwarmCoordinator

@dataclass
class SwarmMetrics:
    """Data structure for swarm metrics"""
    active_agents: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MetricsCollector:
    """Collects and stores metrics from MCP server"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path.cwd() / ".swarm" / "metrics.db")
        Path(self.db_path).parent.mkdir(exist_ok=True)
        self._init_database()
        self.mcp_client: Optional[MCPServerClient] = None
    
    def _init_database(self):
        """Initialize metrics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS swarm_metrics (
                    id INTEGER PRIMARY KEY,
                    active_agents INTEGER,
                    completed_tasks INTEGER,
                    failed_tasks INTEGER,
                    avg_response_time REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY,
                    agent_name TEXT,
                    agent_type TEXT,
                    tasks_completed INTEGER,
                    avg_completion_time REAL,
                    success_rate REAL,
                    last_active TIMESTAMP,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def collect_metrics(self) -> SwarmMetrics:
        """Collect current metrics from MCP server"""
        if not self.mcp_client:
            self.mcp_client = MCPServerClient()
            await self.mcp_client.__aenter__()
        
        coordinator = SwarmCoordinator(self.mcp_client)
        
        try:
            # Get swarm status
            status = await coordinator.get_swarm_status()
            
            # Get agent metrics
            agent_metrics = await coordinator.get_agent_metrics()
            
            # Calculate aggregate metrics
            metrics = SwarmMetrics(
                active_agents=len(agent_metrics),
                completed_tasks=sum(m.get("completed_tasks", 0) for m in agent_metrics),
                failed_tasks=sum(m.get("failed_tasks", 0) for m in agent_metrics),
                avg_response_time=sum(m.get("avg_response_time", 0) for m in agent_metrics) / max(len(agent_metrics), 1),
                memory_usage=status.get("memory_usage", 0),
                cpu_usage=status.get("cpu_usage", 0)
            )
            
            # Store in database
            self.store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return SwarmMetrics()
    
    def store_metrics(self, metrics: SwarmMetrics):
        """Store metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO swarm_metrics 
                (active_agents, completed_tasks, failed_tasks, avg_response_time, memory_usage, cpu_usage)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metrics.active_agents,
                metrics.completed_tasks,
                metrics.failed_tasks,
                metrics.avg_response_time,
                metrics.memory_usage,
                metrics.cpu_usage
            ))
    
    def get_historical_metrics(self, hours: int = 24) -> List[SwarmMetrics]:
        """Get historical metrics from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT active_agents, completed_tasks, failed_tasks, 
                       avg_response_time, memory_usage, cpu_usage, timestamp
                FROM swarm_metrics 
                WHERE timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours))
            
            return [
                SwarmMetrics(
                    active_agents=row[0],
                    completed_tasks=row[1],
                    failed_tasks=row[2],
                    avg_response_time=row[3],
                    memory_usage=row[4],
                    cpu_usage=row[5],
                    timestamp=datetime.fromisoformat(row[6])
                )
                for row in cursor.fetchall()
            ]

class SwarmStatusWidget(Static):
    """Widget displaying current swarm status"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics: SwarmMetrics = SwarmMetrics()
    
    def update_metrics(self, metrics: SwarmMetrics):
        """Update displayed metrics"""
        self.metrics = metrics
        self.update(self.render_status())
    
    def render_status(self) -> Panel:
        """Render status panel"""
        table = Table(title="Swarm Status", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Active Agents", str(self.metrics.active_agents))
        table.add_row("Completed Tasks", str(self.metrics.completed_tasks))
        table.add_row("Failed Tasks", str(self.metrics.failed_tasks))
        table.add_row("Avg Response Time", f"{self.metrics.avg_response_time:.2f}s")
        table.add_row("Memory Usage", f"{self.metrics.memory_usage:.1f}%")
        table.add_row("CPU Usage", f"{self.metrics.cpu_usage:.1f}%")
        table.add_row("Last Update", self.metrics.timestamp.strftime("%H:%M:%S"))
        
        return Panel(table, title="📊 Swarm Metrics", border_style="blue")

class PerformanceChartWidget(Static):
    """Widget displaying performance charts"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history: List[SwarmMetrics] = []
    
    def update_history(self, history: List[SwarmMetrics]):
        """Update performance history"""
        self.history = history[-50:]  # Keep last 50 entries
        self.update(self.render_chart())
    
    def render_chart(self) -> Panel:
        """Render performance chart"""
        if not self.history:
            return Panel("No historical data available", title="📈 Performance Trends")
        
        # Create a simple ASCII chart
        table = Table(title="Recent Performance", show_header=True, header_style="bold green")
        table.add_column("Time", style="cyan")
        table.add_column("Agents", style="yellow")
        table.add_column("Tasks", style="green")
        table.add_column("Response", style="blue")
        
        for metrics in self.history[-10:]:  # Show last 10 entries
            table.add_row(
                metrics.timestamp.strftime("%H:%M"),
                str(metrics.active_agents),
                str(metrics.completed_tasks),
                f"{metrics.avg_response_time:.1f}s"
            )
        
        return Panel(table, title="📈 Performance History", border_style="green")

class AgentListWidget(Static):
    """Widget displaying active agents"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agents: List[Dict[str, Any]] = []
    
    async def update_agents(self, coordinator: SwarmCoordinator):
        """Update agent list"""
        try:
            metrics = await coordinator.get_agent_metrics()
            self.agents = metrics
            self.update(self.render_agents())
        except Exception as e:
            self.update(Panel(f"Error loading agents: {e}", border_style="red"))
    
    def render_agents(self) -> Panel:
        """Render agents table"""
        if not self.agents:
            return Panel("No active agents", title="🤖 Active Agents")
        
        table = Table(title="Agent Status", show_header=True, header_style="bold cyan")
        table.add_column("Agent", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Tasks", style="blue")
        
        for agent in self.agents:
            table.add_row(
                agent.get("name", "Unknown"),
                agent.get("type", "Unknown"),
                agent.get("status", "Active"),
                str(agent.get("completed_tasks", 0))
            )
        
        return Panel(table, title="🤖 Active Agents", border_style="cyan")

class MonitoringDashboard(App):
    """Main monitoring dashboard application"""
    
    CSS = """
    .container {
        height: 100vh;
    }
    
    .status-panel {
        height: 40%;
    }
    
    .chart-panel {
        height: 30%;
    }
    
    .agents-panel {
        height: 30%;
    }
    
    .control-panel {
        height: 10%;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.metrics_collector = MetricsCollector()
        self.coordinator: Optional[SwarmCoordinator] = None
        self.update_timer: Optional[Timer] = None
    
    def compose(self) -> ComposeResult:
        """Compose the dashboard UI"""
        yield Header(show_clock=True)
        
        with Container(classes="container"):
            with TabbedContent():
                with TabPane("Swarm Status", id="status"):
                    with Vertical():
                        yield SwarmStatusWidget(classes="status-panel", id="status_widget")
                        yield PerformanceChartWidget(classes="chart-panel", id="chart_widget")
                        
                        with Horizontal(classes="control-panel"):
                            yield Button("Refresh", id="refresh_btn")
                            yield Button("Reset Metrics", id="reset_btn")
                
                with TabPane("Agents", id="agents"):
                    yield AgentListWidget(classes="agents-panel", id="agents_widget")
                
                with TabPane("Logs", id="logs"):
                    yield Log(id="log_widget")
        
        yield Footer()
    
    async def on_mount(self):
        """Initialize the dashboard"""
        # Initialize MCP client
        client = MCPServerClient()
        await client.__aenter__()
        self.coordinator = SwarmCoordinator(client)
        
        # Start update timer
        self.update_timer = self.set_interval(5.0, self.update_dashboard)
        
        # Initial update
        await self.update_dashboard()
        
        self.query_one("#log_widget", Log).write("Dashboard initialized")
    
    async def update_dashboard(self):
        """Update all dashboard components"""
        try:
            # Collect current metrics
            metrics = await self.metrics_collector.collect_metrics()
            
            # Update status widget
            status_widget = self.query_one("#status_widget", SwarmStatusWidget)
            status_widget.update_metrics(metrics)
            
            # Update chart widget with historical data
            history = self.metrics_collector.get_historical_metrics(1)  # Last hour
            chart_widget = self.query_one("#chart_widget", PerformanceChartWidget)
            chart_widget.update_history(history)
            
            # Update agents widget
            agents_widget = self.query_one("#agents_widget", AgentListWidget)
            await agents_widget.update_agents(self.coordinator)
            
            # Log update
            self.query_one("#log_widget", Log).write(f"Dashboard updated - {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.query_one("#log_widget", Log).write(f"Update error: {e}")
    
    async def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses"""
        if event.button.id == "refresh_btn":
            await self.update_dashboard()
            self.query_one("#log_widget", Log).write("Manual refresh completed")
        
        elif event.button.id == "reset_btn":
            # Clear metrics database
            Path(self.metrics_collector.db_path).unlink(missing_ok=True)
            self.metrics_collector._init_database()
            await self.update_dashboard()
            self.query_one("#log_widget", Log).write("Metrics reset completed")

def run_dashboard():
    """Run the monitoring dashboard"""
    app = MonitoringDashboard()
    app.run()

if __name__ == "__main__":
    run_dashboard()