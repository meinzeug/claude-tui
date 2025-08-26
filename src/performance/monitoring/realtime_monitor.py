"""
Real-time Performance Monitoring System

Implements comprehensive real-time performance monitoring with SLA tracking,
alerting, and automated optimization recommendations.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import aiohttp
from collections import deque, defaultdict
import threading
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SLAThreshold:
    """SLA threshold configuration"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'lt', 'gt', 'eq'
    severity: str  # 'critical', 'warning', 'info'
    description: str

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str
    action: str

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    timestamp: datetime
    description: str
    resolved: bool = False

class MetricsBuffer:
    """Thread-safe circular buffer for metrics"""
    
    def __init__(self, maxsize: int = 1000):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
    
    def append(self, metric: Dict[str, Any]):
        """Add metric to buffer"""
        with self.lock:
            self.buffer.append(metric)
    
    def get_recent(self, count: int = None) -> List[Dict[str, Any]]:
        """Get recent metrics"""
        with self.lock:
            if count is None:
                return list(self.buffer)
            return list(self.buffer)[-count:]
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()

class SLATracker:
    """Service Level Agreement tracking and compliance monitoring"""
    
    def __init__(self, sla_config: List[SLAThreshold]):
        self.sla_thresholds = {sla.metric_name: sla for sla in sla_config}
        self.sla_violations = []
        self.compliance_history = defaultdict(list)
        
    def check_sla_compliance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check current metrics against SLA thresholds"""
        violations = []
        compliance_status = {}
        
        for metric_name, sla in self.sla_thresholds.items():
            current_value = self._get_metric_value(metrics, metric_name)
            
            if current_value is not None:
                is_compliant = self._evaluate_threshold(current_value, sla)
                compliance_status[metric_name] = {
                    'compliant': is_compliant,
                    'current_value': current_value,
                    'threshold': sla.threshold_value,
                    'description': sla.description
                }
                
                if not is_compliant:
                    violation = {
                        'metric_name': metric_name,
                        'current_value': current_value,
                        'threshold_value': sla.threshold_value,
                        'severity': sla.severity,
                        'timestamp': datetime.utcnow(),
                        'description': sla.description
                    }
                    violations.append(violation)
                    self.sla_violations.append(violation)
                
                # Track compliance history
                self.compliance_history[metric_name].append({
                    'timestamp': datetime.utcnow(),
                    'compliant': is_compliant,
                    'value': current_value
                })
        
        return {
            'violations': violations,
            'compliance_status': compliance_status,
            'overall_compliance': len(violations) == 0
        }
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value using dot notation"""
        keys = metric_name.split('.')
        value = metrics
        
        try:
            for key in keys:
                value = value[key]
            return float(value) if value is not None else None
        except (KeyError, TypeError, ValueError):
            return None
    
    def _evaluate_threshold(self, value: float, sla: SLAThreshold) -> bool:
        """Evaluate if value meets SLA threshold"""
        if sla.comparison == 'lt':
            return value < sla.threshold_value
        elif sla.comparison == 'gt':
            return value > sla.threshold_value
        elif sla.comparison == 'eq':
            return value == sla.threshold_value
        return True
    
    def get_compliance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Generate SLA compliance summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        summary = {}
        for metric_name, history in self.compliance_history.items():
            recent_history = [
                h for h in history 
                if h['timestamp'] >= cutoff_time
            ]
            
            if recent_history:
                compliant_count = sum(1 for h in recent_history if h['compliant'])
                total_count = len(recent_history)
                compliance_rate = compliant_count / total_count if total_count > 0 else 0
                
                summary[metric_name] = {
                    'compliance_rate': compliance_rate,
                    'total_checks': total_count,
                    'violations': total_count - compliant_count,
                    'last_violation': max(
                        (h['timestamp'] for h in recent_history if not h['compliant']),
                        default=None
                    )
                }
        
        return summary

class AlertManager:
    """Advanced alerting system with rule-based triggers"""
    
    def __init__(self, alert_rules: List[AlertRule]):
        self.alert_rules = {rule.name: rule for rule in alert_rules}
        self.active_alerts = {}
        self.alert_history = []
        self.alert_callbacks: List[Callable] = []
        
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
        
    async def evaluate_alerts(self, metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Evaluate all alert rules against current metrics"""
        new_alerts = []
        current_time = datetime.utcnow()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                alert = await self._evaluate_rule(rule, metrics, current_time)
                if alert:
                    new_alerts.append(alert)
                    
                    # Trigger callbacks
                    for callback in self.alert_callbacks:
                        try:
                            await callback(alert) if asyncio.iscoroutinefunction(callback) else callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
                            
            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule_name}: {e}")
        
        return new_alerts
    
    async def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any], current_time: datetime) -> Optional[PerformanceAlert]:
        """Evaluate individual alert rule"""
        # Extract metric value
        metric_value = self._extract_metric_value(metrics, rule.condition)
        
        if metric_value is None:
            return None
        
        # Check if threshold is exceeded
        if metric_value > rule.threshold:
            alert_id = f"{rule.name}_{int(current_time.timestamp())}"
            
            # Check if this is a new alert or update to existing
            if rule.name in self.active_alerts:
                # Update existing alert
                existing_alert = self.active_alerts[rule.name]
                existing_alert.current_value = metric_value
                existing_alert.timestamp = current_time
                return None  # Don't create new alert for existing condition
            else:
                # Create new alert
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    rule_name=rule.name,
                    metric_name=rule.condition,
                    current_value=metric_value,
                    threshold_value=rule.threshold,
                    severity=rule.severity,
                    timestamp=current_time,
                    description=f"Metric {rule.condition} exceeded threshold: {metric_value} > {rule.threshold}"
                )
                
                self.active_alerts[rule.name] = alert
                self.alert_history.append(alert)
                
                return alert
        else:
            # Check if we should resolve an existing alert
            if rule.name in self.active_alerts:
                resolved_alert = self.active_alerts[rule.name]
                resolved_alert.resolved = True
                del self.active_alerts[rule.name]
                logger.info(f"Alert resolved: {rule.name}")
        
        return None
    
    def _extract_metric_value(self, metrics: Dict[str, Any], condition: str) -> Optional[float]:
        """Extract metric value from condition string"""
        # Simple implementation - can be enhanced for complex expressions
        keys = condition.split('.')
        value = metrics
        
        try:
            for key in keys:
                value = value[key]
            return float(value) if value is not None else None
        except (KeyError, TypeError, ValueError):
            return None

class PerformanceDashboard:
    """Real-time performance dashboard with metrics visualization"""
    
    def __init__(self):
        self.dashboard_data = {
            'current_metrics': {},
            'historical_trends': defaultdict(list),
            'alerts': [],
            'sla_status': {},
            'system_health': 'unknown'
        }
        
    def update_dashboard(self, 
                        current_metrics: Dict[str, Any],
                        alerts: List[PerformanceAlert],
                        sla_status: Dict[str, Any]):
        """Update dashboard with latest data"""
        self.dashboard_data['current_metrics'] = current_metrics
        self.dashboard_data['alerts'] = [asdict(alert) for alert in alerts]
        self.dashboard_data['sla_status'] = sla_status
        self.dashboard_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Update historical trends
        timestamp = time.time()
        for metric_name, value in self._flatten_metrics(current_metrics).items():
            self.dashboard_data['historical_trends'][metric_name].append({
                'timestamp': timestamp,
                'value': value
            })
            
            # Keep only last 1000 data points
            if len(self.dashboard_data['historical_trends'][metric_name]) > 1000:
                self.dashboard_data['historical_trends'][metric_name] = \
                    self.dashboard_data['historical_trends'][metric_name][-1000:]
        
        # Update system health
        self._update_system_health(alerts, sla_status)
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = '') -> Dict[str, float]:
        """Flatten nested metrics dictionary"""
        flattened = {}
        
        for key, value in metrics.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics(value, new_key))
            elif isinstance(value, (int, float)):
                flattened[new_key] = float(value)
        
        return flattened
    
    def _update_system_health(self, alerts: List[PerformanceAlert], sla_status: Dict[str, Any]):
        """Update overall system health status"""
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        warning_alerts = [a for a in alerts if a.severity == 'warning']
        
        sla_violations = not sla_status.get('overall_compliance', True)
        
        if critical_alerts or sla_violations:
            self.dashboard_data['system_health'] = 'critical'
        elif warning_alerts:
            self.dashboard_data['system_health'] = 'warning'
        else:
            self.dashboard_data['system_health'] = 'healthy'
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()
    
    async def export_dashboard(self, filepath: str):
        """Export dashboard data to file"""
        with open(filepath, 'w') as f:
            json.dump(self.dashboard_data, f, indent=2, default=str)

class RealtimePerformanceMonitor:
    """Main real-time performance monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.monitoring_active = False
        
        # Initialize components
        self.metrics_buffer = MetricsBuffer(maxsize=self.config.get('buffer_size', 1000))
        self.sla_tracker = SLATracker(self._load_sla_config())
        self.alert_manager = AlertManager(self._load_alert_rules())
        self.dashboard = PerformanceDashboard()
        
        # Monitoring settings
        self.monitoring_interval = self.config.get('monitoring_interval', 5.0)  # seconds
        self.metrics_collection_tasks = []
        
        # Setup alert callbacks
        self.alert_manager.add_alert_callback(self._handle_alert)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            'monitoring_interval': 5.0,
            'buffer_size': 1000,
            'enable_sla_tracking': True,
            'enable_alerting': True,
            'dashboard_export_interval': 60.0,
            'log_metrics': True
        }
    
    def _load_sla_config(self) -> List[SLAThreshold]:
        """Load SLA configuration"""
        return [
            SLAThreshold('cpu.percent', 80.0, 'lt', 'warning', 'CPU usage should be below 80%'),
            SLAThreshold('memory.percent', 85.0, 'lt', 'critical', 'Memory usage should be below 85%'),
            SLAThreshold('response_time.p95', 2.0, 'lt', 'warning', 'P95 response time should be below 2s'),
            SLAThreshold('error_rate', 0.05, 'lt', 'critical', 'Error rate should be below 5%'),
            SLAThreshold('throughput', 100.0, 'gt', 'warning', 'Throughput should be above 100 ops/s')
        ]
    
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rule configuration"""
        return [
            AlertRule('high_cpu', 'cpu.percent', 90.0, 30, 'critical', 'scale_up'),
            AlertRule('high_memory', 'memory.percent', 90.0, 30, 'critical', 'restart_service'),
            AlertRule('high_error_rate', 'error_rate', 0.1, 60, 'critical', 'investigate'),
            AlertRule('low_throughput', 'throughput', 50.0, 120, 'warning', 'optimize'),
            AlertRule('high_latency', 'response_time.p95', 3.0, 60, 'warning', 'investigate')
        ]
    
    async def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting real-time performance monitoring")
        
        # Start metrics collection tasks
        self.metrics_collection_tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_application_metrics()),
            asyncio.create_task(self._process_metrics()),
            asyncio.create_task(self._export_dashboard_periodically())
        ]
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.metrics_collection_tasks)
        except Exception as e:
            logger.error(f"Monitoring task failed: {e}")
        finally:
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        logger.info("Stopping performance monitoring")
        self.monitoring_active = False
        
        # Cancel all tasks
        for task in self.metrics_collection_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.metrics_collection_tasks, return_exceptions=True)
        
        # Final dashboard export
        await self.dashboard.export_dashboard(f"final_dashboard_{int(time.time())}.json")
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        while self.monitoring_active:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count()
                load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                
                # Disk metrics
                disk_usage = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                
                # Network metrics
                network_io = psutil.net_io_counters()
                
                system_metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'cpu': {
                        'percent': cpu_percent,
                        'count': cpu_count,
                        'load_avg_1': load_avg[0],
                        'load_avg_5': load_avg[1],
                        'load_avg_15': load_avg[2]
                    },
                    'memory': {
                        'percent': memory.percent,
                        'available': memory.available,
                        'used': memory.used,
                        'total': memory.total,
                        'free': memory.free
                    },
                    'disk': {
                        'usage_percent': disk_usage.percent,
                        'free': disk_usage.free,
                        'total': disk_usage.total,
                        'read_bytes': disk_io.read_bytes if disk_io else 0,
                        'write_bytes': disk_io.write_bytes if disk_io else 0,
                        'read_count': disk_io.read_count if disk_io else 0,
                        'write_count': disk_io.write_count if disk_io else 0
                    },
                    'network': {
                        'bytes_sent': network_io.bytes_sent if network_io else 0,
                        'bytes_recv': network_io.bytes_recv if network_io else 0,
                        'packets_sent': network_io.packets_sent if network_io else 0,
                        'packets_recv': network_io.packets_recv if network_io else 0
                    }
                }
                
                self.metrics_buffer.append({'type': 'system', 'data': system_metrics})
                
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_application_metrics(self):
        """Collect application-specific performance metrics"""
        while self.monitoring_active:
            try:
                # Simulate application metrics collection
                # In real implementation, this would collect from your application
                app_metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'response_time': {
                        'avg': 0.5 + 0.3 * np.random.random(),
                        'p50': 0.4 + 0.2 * np.random.random(),
                        'p95': 1.0 + 0.5 * np.random.random(),
                        'p99': 1.5 + 0.8 * np.random.random()
                    },
                    'throughput': 100 + 50 * np.random.random(),
                    'error_rate': 0.01 + 0.04 * np.random.random(),
                    'active_connections': 50 + int(25 * np.random.random()),
                    'queue_length': int(10 * np.random.random()),
                    'database': {
                        'connections_active': 10 + int(5 * np.random.random()),
                        'query_time_avg': 0.1 + 0.05 * np.random.random(),
                        'slow_queries': int(3 * np.random.random())
                    }
                }
                
                self.metrics_buffer.append({'type': 'application', 'data': app_metrics})
                
            except Exception as e:
                logger.error(f"Failed to collect application metrics: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _process_metrics(self):
        """Process collected metrics for SLA tracking and alerting"""
        while self.monitoring_active:
            try:
                # Get recent metrics
                recent_metrics = self.metrics_buffer.get_recent(10)
                
                if recent_metrics:
                    # Combine system and application metrics
                    combined_metrics = self._combine_metrics(recent_metrics)
                    
                    # SLA tracking
                    if self.config.get('enable_sla_tracking', True):
                        sla_result = self.sla_tracker.check_sla_compliance(combined_metrics)
                    else:
                        sla_result = {'violations': [], 'overall_compliance': True}
                    
                    # Alert evaluation
                    if self.config.get('enable_alerting', True):
                        alerts = await self.alert_manager.evaluate_alerts(combined_metrics)
                    else:
                        alerts = []
                    
                    # Update dashboard
                    self.dashboard.update_dashboard(combined_metrics, alerts, sla_result)
                    
                    # Log metrics if enabled
                    if self.config.get('log_metrics', False):
                        logger.info(f"Processed metrics: {len(recent_metrics)} samples")
                        if sla_result['violations']:
                            logger.warning(f"SLA violations: {len(sla_result['violations'])}")
                        if alerts:
                            logger.warning(f"Active alerts: {len(alerts)}")
                
            except Exception as e:
                logger.error(f"Failed to process metrics: {e}")
            
            await asyncio.sleep(self.monitoring_interval * 2)  # Process less frequently
    
    def _combine_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine and aggregate metrics from different sources"""
        combined = {}
        
        # Separate by type
        system_metrics = [m['data'] for m in metrics_list if m['type'] == 'system']
        app_metrics = [m['data'] for m in metrics_list if m['type'] == 'application']
        
        # Aggregate system metrics (take latest)
        if system_metrics:
            combined.update(system_metrics[-1])
        
        # Aggregate application metrics (take latest)
        if app_metrics:
            combined.update(app_metrics[-1])
        
        return combined
    
    async def _export_dashboard_periodically(self):
        """Periodically export dashboard data"""
        export_interval = self.config.get('dashboard_export_interval', 60.0)
        
        while self.monitoring_active:
            try:
                timestamp = int(time.time())
                filepath = f"dashboard_data_{timestamp}.json"
                await self.dashboard.export_dashboard(filepath)
                logger.info(f"Dashboard data exported to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export dashboard: {e}")
            
            await asyncio.sleep(export_interval)
    
    async def _handle_alert(self, alert: PerformanceAlert):
        """Handle performance alert"""
        logger.warning(f"ALERT: {alert.rule_name} - {alert.description}")
        
        # In a real implementation, this would:
        # - Send notifications (email, Slack, PagerDuty)
        # - Trigger automated remediation actions
        # - Update monitoring dashboards
        # - Log to centralized logging system
        
        alert_data = asdict(alert)
        alert_file = f"alerts/alert_{alert.alert_id}.json"
        
        try:
            Path("alerts").mkdir(exist_ok=True)
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")

# Example usage and testing
async def main():
    """Example usage of real-time performance monitoring"""
    monitor = RealtimePerformanceMonitor()
    
    try:
        # Start monitoring
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        
        # Let it run for a demo period
        await asyncio.sleep(30)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Get final dashboard data
        dashboard_data = monitor.dashboard.get_dashboard_data()
        print(f"Final system health: {dashboard_data['system_health']}")
        print(f"Total alerts: {len(dashboard_data['alerts'])}")
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())