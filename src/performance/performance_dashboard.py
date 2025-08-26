#!/usr/bin/env python3
"""
Performance Dashboard - Real-time Performance Monitoring and Visualization

Provides a comprehensive dashboard for monitoring system performance with:
- Real-time metrics visualization
- Performance trend analysis
- Alert system for threshold breaches
- Historical performance tracking
- Optimization progress monitoring
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics

try:
    from .performance_profiler import PerformanceProfiler, PerformanceSnapshot, PerformanceMetric
except ImportError:
    from performance_profiler import PerformanceProfiler, PerformanceSnapshot, PerformanceMetric


@dataclass
class DashboardConfig:
    """Configuration for performance dashboard"""
    refresh_interval_seconds: float = 5.0
    history_retention_hours: int = 24
    alert_cooldown_minutes: int = 5
    performance_targets: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_targets is None:
            self.performance_targets = {
                'memory_mb': 100.0,
                'cpu_percent': 10.0,
                'startup_time_ms': 2000.0,
                'response_time_ms': 200.0
            }


@dataclass
class PerformanceAlert:
    """Performance alert for threshold breaches"""
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # INFO, WARNING, CRITICAL
    timestamp: float
    message: str
    acknowledged: bool = False


class PerformanceDashboard:
    """
    Real-time Performance Monitoring Dashboard
    
    Features:
    - Live performance metrics
    - Historical trend analysis
    - Automated alerting
    - Performance regression detection
    - Optimization tracking
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.profiler = PerformanceProfiler()
        
        # Dashboard state
        self.running = False
        self.alerts: List[PerformanceAlert] = []
        self.last_alert_times: Dict[str, float] = {}
        
        # Performance history
        self.metrics_history: Dict[str, List[float]] = {}
        self.snapshots_history: List[PerformanceSnapshot] = []
        
        # Dashboard thread
        self.dashboard_thread: Optional[threading.Thread] = None
        
    def start_dashboard(self):
        """Start the performance dashboard monitoring"""
        if self.running:
            return
            
        self.running = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dashboard_thread.start()
        
        print("🚀 PERFORMANCE DASHBOARD STARTED")
        print(f"   Refresh interval: {self.config.refresh_interval_seconds}s")
        print(f"   History retention: {self.config.history_retention_hours}h")
        print(f"   Targets: {self.config.performance_targets}")
        
    def stop_dashboard(self):
        """Stop the performance dashboard monitoring"""
        self.running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5.0)
        print("🛑 PERFORMANCE DASHBOARD STOPPED")
        
    def _dashboard_loop(self):
        """Main dashboard monitoring loop"""
        while self.running:
            try:
                # Take performance snapshot
                snapshot = self.profiler.create_snapshot()
                self.snapshots_history.append(snapshot)
                
                # Update metrics history
                self._update_metrics_history(snapshot)
                
                # Check for alerts
                self._check_alerts(snapshot)
                
                # Clean old data
                self._cleanup_old_data()
                
                # Display dashboard (in production, this would update UI)
                self._display_dashboard_summary()
                
                time.sleep(self.config.refresh_interval_seconds)
                
            except Exception as e:
                print(f"Dashboard error: {e}")
                time.sleep(self.config.refresh_interval_seconds)
                
    def _update_metrics_history(self, snapshot: PerformanceSnapshot):
        """Update historical metrics tracking"""
        timestamp = snapshot.timestamp
        
        # Core metrics
        metrics = {
            'memory_mb': snapshot.process_memory_mb,
            'system_memory_percent': snapshot.system_memory_percent,
            'cpu_percent': snapshot.cpu_percent,
            'gc_objects': snapshot.gc_objects,
            'modules': snapshot.module_count,
            'threads': snapshot.thread_count
        }
        
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            
            self.metrics_history[name].append(value)
            
            # Limit history size (keep last 1000 points)
            if len(self.metrics_history[name]) > 1000:
                self.metrics_history[name] = self.metrics_history[name][-500:]
                
    def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts"""
        current_time = time.time()
        
        # Check memory usage
        if snapshot.process_memory_mb > 200:  # High memory threshold
            self._create_alert(
                "memory_usage",
                snapshot.process_memory_mb,
                200,
                "CRITICAL" if snapshot.process_memory_mb > 500 else "WARNING",
                f"High memory usage: {snapshot.process_memory_mb:.1f}MB"
            )
            
        # Check CPU usage
        if snapshot.cpu_percent > 80:  # High CPU threshold
            self._create_alert(
                "cpu_usage", 
                snapshot.cpu_percent,
                80,
                "CRITICAL" if snapshot.cpu_percent > 95 else "WARNING",
                f"High CPU usage: {snapshot.cpu_percent:.1f}%"
            )
            
        # Check system memory
        if snapshot.system_memory_percent > 90:
            self._create_alert(
                "system_memory",
                snapshot.system_memory_percent,
                90,
                "CRITICAL",
                f"System memory critical: {snapshot.system_memory_percent:.1f}%"
            )
            
        # Check GC pressure
        if snapshot.gc_objects > 100000:
            self._create_alert(
                "gc_pressure",
                snapshot.gc_objects,
                100000,
                "WARNING",
                f"High GC pressure: {snapshot.gc_objects:,} objects"
            )
            
    def _create_alert(self, metric_name: str, current_value: float, threshold: float, 
                     severity: str, message: str):
        """Create performance alert with cooldown"""
        current_time = time.time()
        
        # Check cooldown
        last_alert_time = self.last_alert_times.get(metric_name, 0)
        cooldown_seconds = self.config.alert_cooldown_minutes * 60
        
        if current_time - last_alert_time < cooldown_seconds:
            return  # Still in cooldown
            
        alert = PerformanceAlert(
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold,
            severity=severity,
            timestamp=current_time,
            message=message
        )
        
        self.alerts.append(alert)
        self.last_alert_times[metric_name] = current_time
        
        # Limit alert history
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
            
        print(f"🚨 PERFORMANCE ALERT [{severity}]: {message}")
        
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        cutoff_time = time.time() - (self.config.history_retention_hours * 3600)
        
        # Clean snapshots
        self.snapshots_history = [s for s in self.snapshots_history if s.timestamp > cutoff_time]
        
        # Clean alerts older than 24 hours
        alert_cutoff = time.time() - (24 * 3600)
        self.alerts = [a for a in self.alerts if a.timestamp > alert_cutoff]
        
    def _display_dashboard_summary(self):
        """Display dashboard summary (console version)"""
        if not self.snapshots_history:
            return
            
        latest = self.snapshots_history[-1]
        
        # Calculate trends (last 10 snapshots)
        recent_snapshots = self.snapshots_history[-10:] if len(self.snapshots_history) >= 10 else self.snapshots_history
        
        if len(recent_snapshots) > 1:
            memory_trend = self._calculate_trend([s.process_memory_mb for s in recent_snapshots])
            cpu_trend = self._calculate_trend([s.cpu_percent for s in recent_snapshots if s.cpu_percent > 0])
        else:
            memory_trend = 0
            cpu_trend = 0
            
        # Dashboard display (simple console version)
        print("\n" + "="*60)
        print("📊 PERFORMANCE DASHBOARD")
        print("="*60)
        print(f"🕒 Time: {datetime.fromtimestamp(latest.timestamp).strftime('%H:%M:%S')}")
        print(f"📈 Memory: {latest.process_memory_mb:.1f}MB ({self._trend_arrow(memory_trend)} {abs(memory_trend):.1f}%)")
        print(f"⚡ CPU: {latest.cpu_percent:.1f}% ({self._trend_arrow(cpu_trend)} {abs(cpu_trend):.1f}%)")
        print(f"🗑️  GC Objects: {latest.gc_objects:,}")
        print(f"📦 Modules: {latest.module_count}")
        print(f"🧵 Threads: {latest.thread_count}")
        
        # Active alerts
        active_alerts = [a for a in self.alerts[-5:] if not a.acknowledged]
        if active_alerts:
            print(f"\n🚨 Active Alerts ({len(active_alerts)}):")
            for alert in active_alerts[-3:]:  # Show last 3
                age_minutes = (time.time() - alert.timestamp) / 60
                print(f"   [{alert.severity}] {alert.message} ({age_minutes:.0f}m ago)")
        
        print("="*60)
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend percentage (positive = increasing, negative = decreasing)"""
        if len(values) < 2:
            return 0
            
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return 0
            
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        if avg_first == 0:
            return 0
            
        return ((avg_second - avg_first) / avg_first) * 100
        
    def _trend_arrow(self, trend: float) -> str:
        """Get trend arrow indicator"""
        if trend > 5:
            return "⬆️"
        elif trend < -5:
            return "⬇️"
        else:
            return "➡️"
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.snapshots_history:
            return {}
            
        recent_snapshots = self.snapshots_history[-20:]  # Last 20 snapshots
        latest = recent_snapshots[-1]
        
        # Calculate statistics
        memory_values = [s.process_memory_mb for s in recent_snapshots]
        cpu_values = [s.cpu_percent for s in recent_snapshots if s.cpu_percent > 0]
        
        summary = {
            "current_status": {
                "timestamp": latest.timestamp,
                "memory_mb": latest.process_memory_mb,
                "cpu_percent": latest.cpu_percent,
                "system_memory_percent": latest.system_memory_percent,
                "gc_objects": latest.gc_objects,
                "modules": latest.module_count,
                "threads": latest.thread_count
            },
            "trends_last_20_snapshots": {
                "memory_avg": statistics.mean(memory_values),
                "memory_max": max(memory_values),
                "memory_min": min(memory_values),
                "memory_std": statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                "cpu_avg": statistics.mean(cpu_values) if cpu_values else 0,
                "cpu_max": max(cpu_values) if cpu_values else 0
            },
            "performance_vs_targets": {
                name: {
                    "current": getattr(latest, f"process_memory_mb" if "memory" in name else f"cpu_percent", 0),
                    "target": target,
                    "within_target": getattr(latest, f"process_memory_mb" if "memory" in name else f"cpu_percent", 0) <= target
                }
                for name, target in self.config.performance_targets.items()
                if name in ["memory_mb", "cpu_percent"]
            },
            "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "total_snapshots": len(self.snapshots_history),
            "monitoring_duration_hours": (latest.timestamp - self.snapshots_history[0].timestamp) / 3600 if len(self.snapshots_history) > 1 else 0
        }
        
        return summary
        
    def export_dashboard_data(self, filepath: Optional[str] = None) -> str:
        """Export dashboard data for analysis"""
        if filepath is None:
            filepath = f"dashboard_data_{int(time.time())}.json"
            
        summary = self.get_performance_summary()
        
        dashboard_data = {
            "export_timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "performance_summary": summary,
            "recent_snapshots": [
                {
                    "timestamp": s.timestamp,
                    "memory_mb": s.process_memory_mb,
                    "cpu_percent": s.cpu_percent,
                    "system_memory_percent": s.system_memory_percent,
                    "gc_objects": s.gc_objects,
                    "modules": s.module_count,
                    "threads": s.thread_count
                }
                for s in self.snapshots_history[-100:]  # Last 100 snapshots
            ],
            "recent_alerts": [
                {
                    "metric": a.metric_name,
                    "value": a.current_value,
                    "threshold": a.threshold_value,
                    "severity": a.severity,
                    "timestamp": a.timestamp,
                    "message": a.message,
                    "acknowledged": a.acknowledged
                }
                for a in self.alerts[-20:]  # Last 20 alerts
            ],
            "metrics_history": {
                name: values[-100:]  # Last 100 values
                for name, values in self.metrics_history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
            
        return filepath
        
    def acknowledge_alert(self, metric_name: str):
        """Acknowledge alerts for specific metric"""
        for alert in self.alerts:
            if alert.metric_name == metric_name and not alert.acknowledged:
                alert.acknowledged = True
                
    def get_optimization_progress(self) -> Dict[str, Any]:
        """Track optimization progress over time"""
        if len(self.snapshots_history) < 10:
            return {"status": "insufficient_data", "message": "Need more data points"}
            
        # Compare first 10% vs last 10% of snapshots
        snapshot_count = len(self.snapshots_history)
        early_snapshots = self.snapshots_history[:snapshot_count//10] or self.snapshots_history[:5]
        recent_snapshots = self.snapshots_history[-snapshot_count//10:] or self.snapshots_history[-5:]
        
        early_memory = statistics.mean([s.process_memory_mb for s in early_snapshots])
        recent_memory = statistics.mean([s.process_memory_mb for s in recent_snapshots])
        
        early_cpu = statistics.mean([s.cpu_percent for s in early_snapshots if s.cpu_percent > 0]) or 0
        recent_cpu = statistics.mean([s.cpu_percent for s in recent_snapshots if s.cpu_percent > 0]) or 0
        
        memory_improvement = ((early_memory - recent_memory) / early_memory) * 100 if early_memory > 0 else 0
        cpu_improvement = ((early_cpu - recent_cpu) / early_cpu) * 100 if early_cpu > 0 else 0
        
        return {
            "monitoring_period_hours": (recent_snapshots[-1].timestamp - early_snapshots[0].timestamp) / 3600,
            "memory_improvement_percent": memory_improvement,
            "cpu_improvement_percent": cpu_improvement,
            "early_period": {
                "memory_mb": early_memory,
                "cpu_percent": early_cpu
            },
            "recent_period": {
                "memory_mb": recent_memory,
                "cpu_percent": recent_cpu
            },
            "optimization_status": "IMPROVING" if memory_improvement > 5 or cpu_improvement > 5 else 
                                  "DEGRADING" if memory_improvement < -10 or cpu_improvement < -10 else "STABLE"
        }


# Convenience functions
def start_performance_dashboard(config: Optional[DashboardConfig] = None) -> PerformanceDashboard:
    """Start performance dashboard with default or custom config"""
    dashboard = PerformanceDashboard(config)
    dashboard.start_dashboard()
    return dashboard


def run_dashboard_demo(duration_minutes: int = 5):
    """Run dashboard demo for specified duration"""
    print(f"🚀 Starting {duration_minutes}-minute dashboard demo...")
    
    dashboard = start_performance_dashboard()
    
    try:
        time.sleep(duration_minutes * 60)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard demo interrupted by user")
    finally:
        dashboard.stop_dashboard()
        
        # Export final data
        export_path = dashboard.export_dashboard_data()
        print(f"📄 Dashboard data exported to: {export_path}")
        
        # Show optimization progress
        progress = dashboard.get_optimization_progress()
        if progress.get("status") != "insufficient_data":
            print(f"\n📊 OPTIMIZATION PROGRESS:")
            print(f"   Status: {progress['optimization_status']}")
            print(f"   Memory: {progress['memory_improvement_percent']:+.1f}%")
            print(f"   CPU: {progress['cpu_improvement_percent']:+.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Dashboard")
    parser.add_argument("--duration", type=int, default=5, help="Demo duration in minutes")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    config = DashboardConfig(refresh_interval_seconds=args.interval)
    run_dashboard_demo(args.duration)