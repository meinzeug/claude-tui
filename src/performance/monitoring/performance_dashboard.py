"""
Performance Dashboard

Advanced performance monitoring dashboard with real-time metrics visualization,
alerting, and historical trend analysis.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str] = None

@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    points: List[MetricPoint]
    unit: str = ""
    description: str = ""

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    title: str
    widget_type: str  # 'line_chart', 'gauge', 'counter', 'table'
    metrics: List[str]
    config: Dict[str, Any] = None

class MetricsDatabase:
    """SQLite-based metrics storage for historical data"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    metric_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def store_metrics(self, metrics: List[MetricPoint], metric_name: str):
        """Store multiple metric points"""
        with self._get_connection() as conn:
            for point in metrics:
                tags_json = json.dumps(point.tags) if point.tags else None
                conn.execute(
                    'INSERT INTO metrics (name, timestamp, value, tags) VALUES (?, ?, ?, ?)',
                    (metric_name, point.timestamp, point.value, tags_json)
                )
    
    def query_metrics(self, metric_name: str, start_time: float, end_time: float, 
                     limit: int = 1000) -> List[MetricPoint]:
        """Query metrics within time range"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT timestamp, value, tags FROM metrics 
                WHERE name = ? AND timestamp BETWEEN ? AND ? 
                ORDER BY timestamp DESC LIMIT ?
            ''', (metric_name, start_time, end_time, limit))
            
            points = []
            for row in cursor.fetchall():
                timestamp, value, tags_json = row
                tags = json.loads(tags_json) if tags_json else None
                points.append(MetricPoint(timestamp=timestamp, value=value, tags=tags))
            
            return points
    
    def get_metric_names(self) -> List[str]:
        """Get all available metric names"""
        with self._get_connection() as conn:
            cursor = conn.execute('SELECT DISTINCT name FROM metrics ORDER BY name')
            return [row[0] for row in cursor.fetchall()]
    
    def store_alert(self, alert_id: str, metric_name: str, severity: str, 
                   message: str, timestamp: float):
        """Store alert in database"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO alerts 
                (alert_id, metric_name, severity, message, timestamp) 
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_id, metric_name, severity, message, timestamp))
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT alert_id, metric_name, severity, message, timestamp 
                FROM alerts WHERE resolved = FALSE 
                ORDER BY timestamp DESC
            ''')
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'alert_id': row[0],
                    'metric_name': row[1],
                    'severity': row[2],
                    'message': row[3],
                    'timestamp': row[4]
                })
            
            return alerts

class PerformanceDashboard:
    """Advanced performance dashboard with real-time updates"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.metrics_db = MetricsDatabase()
        self.widgets = {}
        self.metric_collectors = {}
        self.dashboard_active = False
        self.update_interval = self.config.get('update_interval', 5.0)
        
        # Real-time data buffers
        self.real_time_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize dashboard
        self._init_dashboard()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default dashboard configuration"""
        return {
            'update_interval': 5.0,
            'data_retention_hours': 24,
            'max_data_points_per_series': 1000,
            'auto_refresh': True,
            'theme': 'light',
            'alert_sound': False
        }
    
    def _init_dashboard(self):
        """Initialize dashboard widgets"""
        # System performance widgets
        self.add_widget(DashboardWidget(
            id='cpu_usage',
            title='CPU Usage',
            widget_type='line_chart',
            metrics=['system.cpu.percent'],
            config={'unit': '%', 'min': 0, 'max': 100, 'color': 'blue'}
        ))
        
        self.add_widget(DashboardWidget(
            id='memory_usage',
            title='Memory Usage',
            widget_type='line_chart',
            metrics=['system.memory.percent'],
            config={'unit': '%', 'min': 0, 'max': 100, 'color': 'green'}
        ))
        
        self.add_widget(DashboardWidget(
            id='network_io',
            title='Network I/O',
            widget_type='line_chart',
            metrics=['system.network.bytes_sent', 'system.network.bytes_recv'],
            config={'unit': 'bytes/s', 'colors': ['red', 'orange']}
        ))
        
        # Application performance widgets
        self.add_widget(DashboardWidget(
            id='response_times',
            title='Response Times',
            widget_type='line_chart',
            metrics=['app.response_time.avg', 'app.response_time.p95', 'app.response_time.p99'],
            config={'unit': 'ms', 'colors': ['blue', 'orange', 'red']}
        ))
        
        self.add_widget(DashboardWidget(
            id='throughput',
            title='Throughput',
            widget_type='gauge',
            metrics=['app.throughput'],
            config={'unit': 'req/s', 'min': 0, 'max': 1000, 'color': 'purple'}
        ))
        
        self.add_widget(DashboardWidget(
            id='error_rate',
            title='Error Rate',
            widget_type='gauge',
            metrics=['app.error_rate'],
            config={'unit': '%', 'min': 0, 'max': 10, 'color': 'red', 'threshold': 5}
        ))
        
        # Database performance widgets
        self.add_widget(DashboardWidget(
            id='db_connections',
            title='Database Connections',
            widget_type='counter',
            metrics=['db.connections.active'],
            config={'color': 'teal'}
        ))
        
        self.add_widget(DashboardWidget(
            id='db_query_time',
            title='Database Query Time',
            widget_type='line_chart',
            metrics=['db.query_time.avg'],
            config={'unit': 'ms', 'color': 'brown'}
        ))
        
        # Alert summary widget
        self.add_widget(DashboardWidget(
            id='alerts_summary',
            title='Active Alerts',
            widget_type='table',
            metrics=['alerts.active'],
            config={'columns': ['Severity', 'Metric', 'Message', 'Time']}
        ))
    
    def add_widget(self, widget: DashboardWidget):
        """Add widget to dashboard"""
        self.widgets[widget.id] = widget
        logger.info(f"Added dashboard widget: {widget.title}")
    
    def remove_widget(self, widget_id: str):
        """Remove widget from dashboard"""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            logger.info(f"Removed dashboard widget: {widget_id}")
    
    def add_metric_collector(self, metric_name: str, collector_func: Callable[[], float]):
        """Add custom metric collector function"""
        self.metric_collectors[metric_name] = collector_func
        logger.info(f"Added metric collector for: {metric_name}")
    
    async def start_dashboard(self):
        """Start dashboard data collection and updates"""
        if self.dashboard_active:
            logger.warning("Dashboard is already active")
            return
        
        self.dashboard_active = True
        logger.info("Starting performance dashboard")
        
        # Start data collection tasks
        tasks = [
            asyncio.create_task(self._collect_metrics_loop()),
            asyncio.create_task(self._update_widgets_loop()),
            asyncio.create_task(self._cleanup_old_data_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Dashboard task failed: {e}")
        finally:
            self.dashboard_active = False
    
    async def stop_dashboard(self):
        """Stop dashboard"""
        logger.info("Stopping performance dashboard")
        self.dashboard_active = False
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        import psutil
        
        while self.dashboard_active:
            try:
                current_time = time.time()
                
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                network = psutil.net_io_counters()
                
                # Store system metrics
                self._store_metric_point('system.cpu.percent', current_time, cpu_percent)
                self._store_metric_point('system.memory.percent', current_time, memory.percent)
                
                if network:
                    # Calculate rates (bytes per second)
                    if hasattr(self, '_last_network_time'):
                        time_delta = current_time - self._last_network_time
                        if time_delta > 0:
                            bytes_sent_rate = (network.bytes_sent - self._last_bytes_sent) / time_delta
                            bytes_recv_rate = (network.bytes_recv - self._last_bytes_recv) / time_delta
                            
                            self._store_metric_point('system.network.bytes_sent', current_time, bytes_sent_rate)
                            self._store_metric_point('system.network.bytes_recv', current_time, bytes_recv_rate)
                    
                    self._last_network_time = current_time
                    self._last_bytes_sent = network.bytes_sent
                    self._last_bytes_recv = network.bytes_recv
                
                # Collect application metrics (simulated)
                # In production, these would come from your application
                app_response_time = 100 + 50 * np.random.random()  # Simulate 100-150ms
                app_throughput = 200 + 100 * np.random.random()    # Simulate 200-300 req/s
                app_error_rate = 1 + 4 * np.random.random()        # Simulate 1-5% error rate
                
                self._store_metric_point('app.response_time.avg', current_time, app_response_time)
                self._store_metric_point('app.response_time.p95', current_time, app_response_time * 1.5)
                self._store_metric_point('app.response_time.p99', current_time, app_response_time * 2.0)
                self._store_metric_point('app.throughput', current_time, app_throughput)
                self._store_metric_point('app.error_rate', current_time, app_error_rate)
                
                # Collect database metrics (simulated)
                db_connections = 10 + int(15 * np.random.random())
                db_query_time = 50 + 30 * np.random.random()
                
                self._store_metric_point('db.connections.active', current_time, db_connections)
                self._store_metric_point('db.query_time.avg', current_time, db_query_time)
                
                # Collect custom metrics
                for metric_name, collector_func in self.metric_collectors.items():
                    try:
                        value = collector_func()
                        self._store_metric_point(metric_name, current_time, value)
                    except Exception as e:
                        logger.error(f"Custom metric collection failed for {metric_name}: {e}")
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    def _store_metric_point(self, metric_name: str, timestamp: float, value: float, tags: Dict[str, str] = None):
        """Store metric point in both real-time buffer and database"""
        point = MetricPoint(timestamp=timestamp, value=value, tags=tags)
        
        # Store in real-time buffer
        self.real_time_metrics[metric_name].append(point)
        
        # Store in database (batch every 10 points to reduce I/O)
        if len(self.real_time_metrics[metric_name]) % 10 == 0:
            recent_points = list(self.real_time_metrics[metric_name])[-10:]
            self.metrics_db.store_metrics(recent_points, metric_name)
    
    async def _update_widgets_loop(self):
        """Update dashboard widgets periodically"""
        while self.dashboard_active:
            try:
                await self._update_all_widgets()
            except Exception as e:
                logger.error(f"Widget update failed: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    async def _cleanup_old_data_loop(self):
        """Clean up old data periodically"""
        cleanup_interval = 3600  # 1 hour
        
        while self.dashboard_active:
            try:
                await self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Data cleanup failed: {e}")
            
            await asyncio.sleep(cleanup_interval)
    
    async def _update_all_widgets(self):
        """Update data for all widgets"""
        current_time = time.time()
        
        for widget_id, widget in self.widgets.items():
            try:
                widget_data = await self._generate_widget_data(widget, current_time)
                # In a real implementation, this would push data to frontend
                logger.debug(f"Updated widget {widget_id}: {len(widget_data.get('data', []))} data points")
            except Exception as e:
                logger.error(f"Failed to update widget {widget_id}: {e}")
    
    async def _generate_widget_data(self, widget: DashboardWidget, current_time: float) -> Dict[str, Any]:
        """Generate data for a specific widget"""
        widget_data = {
            'id': widget.id,
            'title': widget.title,
            'type': widget.widget_type,
            'config': widget.config or {},
            'data': [],
            'timestamp': current_time
        }
        
        if widget.widget_type == 'line_chart':
            # Get time series data for line chart
            time_window = 3600  # 1 hour
            start_time = current_time - time_window
            
            series_data = []
            for metric_name in widget.metrics:
                points = self._get_metric_data(metric_name, start_time, current_time)
                series_data.append({
                    'name': metric_name,
                    'data': [{'x': p.timestamp * 1000, 'y': p.value} for p in points]  # Convert to milliseconds
                })
            
            widget_data['data'] = series_data
        
        elif widget.widget_type == 'gauge':
            # Get latest value for gauge
            if widget.metrics:
                metric_name = widget.metrics[0]
                latest_point = self._get_latest_metric_value(metric_name)
                widget_data['data'] = {
                    'value': latest_point.value if latest_point else 0,
                    'timestamp': latest_point.timestamp if latest_point else current_time
                }
        
        elif widget.widget_type == 'counter':
            # Get latest value for counter
            if widget.metrics:
                metric_name = widget.metrics[0]
                latest_point = self._get_latest_metric_value(metric_name)
                widget_data['data'] = {
                    'value': int(latest_point.value) if latest_point else 0,
                    'timestamp': latest_point.timestamp if latest_point else current_time
                }
        
        elif widget.widget_type == 'table':
            # Get tabular data (e.g., for alerts)
            if widget.id == 'alerts_summary':
                active_alerts = self.metrics_db.get_active_alerts()
                widget_data['data'] = {
                    'rows': [
                        [
                            alert['severity'].upper(),
                            alert['metric_name'],
                            alert['message'][:50] + '...' if len(alert['message']) > 50 else alert['message'],
                            datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                        ]
                        for alert in active_alerts
                    ]
                }
        
        return widget_data
    
    def _get_metric_data(self, metric_name: str, start_time: float, end_time: float) -> List[MetricPoint]:
        """Get metric data from real-time buffer and database"""
        # First, get from real-time buffer
        real_time_points = [
            p for p in self.real_time_metrics[metric_name]
            if start_time <= p.timestamp <= end_time
        ]
        
        # If not enough recent data, query database
        if len(real_time_points) < 100:
            db_points = self.metrics_db.query_metrics(metric_name, start_time, end_time, 500)
            
            # Combine and deduplicate
            all_points = list(real_time_points) + db_points
            unique_points = {}
            for point in all_points:
                # Use timestamp as key to deduplicate
                timestamp_key = round(point.timestamp, 1)  # Round to 0.1 second precision
                unique_points[timestamp_key] = point
            
            return sorted(unique_points.values(), key=lambda p: p.timestamp)
        
        return sorted(real_time_points, key=lambda p: p.timestamp)
    
    def _get_latest_metric_value(self, metric_name: str) -> Optional[MetricPoint]:
        """Get the latest value for a metric"""
        if metric_name in self.real_time_metrics and self.real_time_metrics[metric_name]:
            return self.real_time_metrics[metric_name][-1]
        
        # Fallback to database
        current_time = time.time()
        points = self.metrics_db.query_metrics(metric_name, current_time - 300, current_time, 1)  # Last 5 minutes
        return points[0] if points else None
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent unbounded growth"""
        retention_hours = self.config.get('data_retention_hours', 24)
        cutoff_time = time.time() - (retention_hours * 3600)
        
        # Clean up database (this would require additional SQL in production)
        logger.info(f"Cleaning up data older than {retention_hours} hours")
        
        # Clean up real-time buffers (they're already limited by deque maxlen)
        for metric_name in list(self.real_time_metrics.keys()):
            buffer = self.real_time_metrics[metric_name]
            # Remove old points
            while buffer and buffer[0].timestamp < cutoff_time:
                buffer.popleft()
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current dashboard state"""
        return {
            'active': self.dashboard_active,
            'config': self.config,
            'widgets': [asdict(widget) for widget in self.widgets.values()],
            'metrics_count': len(self.real_time_metrics),
            'last_update': time.time()
        }
    
    def export_dashboard_config(self, filepath: str):
        """Export dashboard configuration to file"""
        config_data = {
            'config': self.config,
            'widgets': [asdict(widget) for widget in self.widgets.values()],
            'collectors': list(self.metric_collectors.keys())
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Dashboard configuration exported to {filepath}")
    
    def import_dashboard_config(self, filepath: str):
        """Import dashboard configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            self.config.update(config_data.get('config', {}))
            
            # Import widgets
            self.widgets.clear()
            for widget_data in config_data.get('widgets', []):
                widget = DashboardWidget(**widget_data)
                self.widgets[widget.id] = widget
            
            logger.info(f"Dashboard configuration imported from {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to import dashboard configuration: {e}")

# Example usage and testing
async def main():
    """Example usage of performance dashboard"""
    dashboard = PerformanceDashboard()
    
    # Add custom metric collector
    dashboard.add_metric_collector('custom.random_metric', lambda: np.random.random() * 100)
    
    try:
        # Start dashboard
        dashboard_task = asyncio.create_task(dashboard.start_dashboard())
        
        # Let it run for demo period
        await asyncio.sleep(60)  # 1 minute
        
        # Stop dashboard
        await dashboard.stop_dashboard()
        
        # Get dashboard state
        state = dashboard.get_dashboard_state()
        print(f"Dashboard collected metrics for {state['metrics_count']} series")
        
        # Export configuration
        dashboard.export_dashboard_config("dashboard_config.json")
        
    except KeyboardInterrupt:
        logger.info("Dashboard interrupted by user")
        await dashboard.stop_dashboard()

if __name__ == "__main__":
    asyncio.run(main())