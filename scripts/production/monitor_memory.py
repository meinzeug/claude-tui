#!/usr/bin/env python3
"""
Production Memory Monitoring Script
Continuous monitoring of memory usage in production
"""

import os
import sys
import time
import json
import signal
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from performance import (
    MemoryProfiler,
    quick_memory_check,
    emergency_memory_rescue
)


class ProductionMemoryMonitor:
    """Production memory monitoring system"""
    
    def __init__(self, 
                 target_memory_mb: int = 200,
                 alert_threshold_mb: int = 300,
                 critical_threshold_mb: int = 500,
                 monitoring_interval: float = 5.0):
        
        self.target_memory_mb = target_memory_mb
        self.alert_threshold_mb = alert_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.monitoring_interval = monitoring_interval
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.profiler = MemoryProfiler(target_memory_mb)
        
        # Alert system
        self.alerts_sent = set()
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes
        
        # Performance tracking
        self.performance_history = []
        self.max_history_size = 1000
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_monitoring()
        sys.exit(0)
        
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            print("⚠️ Monitoring already active")
            return
            
        print(f"📊 Starting memory monitoring (interval: {self.monitoring_interval}s)")
        print(f"🎯 Target: {self.target_memory_mb}MB")
        print(f"⚠️ Alert threshold: {self.alert_threshold_mb}MB")  
        print(f"🚨 Critical threshold: {self.critical_threshold_mb}MB")
        
        self.monitoring_active = True
        self.profiler.start_monitoring(self.monitoring_interval)
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        print("✅ Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self.monitoring_active:
            return
            
        print("🛑 Stopping memory monitoring...")
        
        self.monitoring_active = False
        
        if self.profiler:
            self.profiler.stop_monitoring()
            
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            
        print("✅ Memory monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Take snapshot
                snapshot = self.profiler.take_snapshot()
                current_memory_mb = snapshot.process_memory / 1024 / 1024
                
                # Track performance
                performance_data = {
                    "timestamp": snapshot.timestamp,
                    "memory_mb": current_memory_mb,
                    "gc_objects": snapshot.gc_objects,
                    "largest_objects": snapshot.largest_objects[:5],
                    "leak_suspects": snapshot.leak_suspects
                }
                
                self.performance_history.append(performance_data)
                
                # Limit history size
                if len(self.performance_history) > self.max_history_size:
                    self.performance_history = self.performance_history[-self.max_history_size:]
                    
                # Check thresholds and trigger alerts
                self._check_thresholds(current_memory_mb, snapshot)
                
                # Log status periodically
                self._log_periodic_status(current_memory_mb)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
                
    def _check_thresholds(self, current_memory_mb: float, snapshot):
        """Check memory thresholds and trigger appropriate actions"""
        
        current_time = time.time()
        
        # Critical threshold - immediate action
        if current_memory_mb >= self.critical_threshold_mb:
            alert_key = "critical_memory"
            if self._should_send_alert(alert_key, current_time):
                self._send_critical_alert(current_memory_mb, snapshot)
                self._trigger_emergency_optimization(current_memory_mb)
                
        # Alert threshold - warning
        elif current_memory_mb >= self.alert_threshold_mb:
            alert_key = "high_memory"
            if self._should_send_alert(alert_key, current_time):
                self._send_alert(current_memory_mb, snapshot)
                
        # Memory growth detection
        elif self._detect_rapid_growth():
            alert_key = "rapid_growth"
            if self._should_send_alert(alert_key, current_time):
                self._send_growth_alert()
                
    def _should_send_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if alert should be sent (considering cooldown)"""
        
        last_alert = self.last_alert_time.get(alert_key, 0)
        return (current_time - last_alert) >= self.alert_cooldown
        
    def _send_alert(self, memory_mb: float, snapshot):
        """Send memory alert"""
        
        alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
⚠️ MEMORY ALERT - {alert_time}
Current Memory: {memory_mb:.1f}MB
Alert Threshold: {self.alert_threshold_mb}MB
Target Memory: {self.target_memory_mb}MB
Excess: {memory_mb - self.target_memory_mb:.1f}MB ({((memory_mb/self.target_memory_mb-1)*100):+.1f}%)

Objects: {snapshot.gc_objects:,}
Leak Suspects: {len(snapshot.leak_suspects)}
"""
        
        print(message)
        self._log_alert("ALERT", message)
        
        self.last_alert_time["high_memory"] = time.time()
        
    def _send_critical_alert(self, memory_mb: float, snapshot):
        """Send critical memory alert"""
        
        alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
🚨 CRITICAL MEMORY ALERT - {alert_time}
Current Memory: {memory_mb:.1f}MB
Critical Threshold: {self.critical_threshold_mb}MB
Target Memory: {self.target_memory_mb}MB
Excess: {memory_mb - self.target_memory_mb:.1f}MB ({((memory_mb/self.target_memory_mb-1)*100):+.1f}%)

Emergency optimization will be triggered automatically.
"""
        
        print(message)
        self._log_alert("CRITICAL", message)
        
        self.last_alert_time["critical_memory"] = time.time()
        
    def _trigger_emergency_optimization(self, current_memory_mb: float):
        """Trigger emergency memory optimization"""
        
        print("🚨 TRIGGERING EMERGENCY MEMORY OPTIMIZATION")
        
        try:
            # Run emergency rescue
            rescue_result = emergency_memory_rescue(self.target_memory_mb)
            
            if rescue_result.get("rescue_success", False):
                print("✅ Emergency optimization successful")
                
                # Log success
                self._log_alert("EMERGENCY_SUCCESS", 
                    f"Emergency optimization reduced memory from {current_memory_mb:.1f}MB")
                    
            else:
                print("❌ Emergency optimization failed")
                self._log_alert("EMERGENCY_FAILED", 
                    "Emergency optimization failed to achieve target")
                    
        except Exception as e:
            print(f"❌ Emergency optimization error: {e}")
            self._log_alert("EMERGENCY_ERROR", f"Emergency optimization error: {e}")
            
    def _detect_rapid_growth(self) -> bool:
        """Detect rapid memory growth"""
        
        if len(self.performance_history) < 10:
            return False
            
        # Check last 10 measurements
        recent = self.performance_history[-10:]
        
        # Calculate growth rate
        start_memory = recent[0]["memory_mb"]
        end_memory = recent[-1]["memory_mb"]
        time_span = recent[-1]["timestamp"] - recent[0]["timestamp"]
        
        if time_span <= 0:
            return False
            
        # Growth rate in MB/minute
        growth_rate = (end_memory - start_memory) / (time_span / 60)
        
        # Alert if growing > 10MB/minute
        return growth_rate > 10.0
        
    def _send_growth_alert(self):
        """Send rapid growth alert"""
        
        recent = self.performance_history[-10:]
        start_memory = recent[0]["memory_mb"]
        end_memory = recent[-1]["memory_mb"]
        growth = end_memory - start_memory
        
        message = f"""
📈 RAPID MEMORY GROWTH DETECTED
Growth: {growth:.1f}MB in last monitoring period
Current: {end_memory:.1f}MB
Rate: {growth / (self.monitoring_interval * 10 / 60):.1f}MB/min
"""
        
        print(message)
        self._log_alert("GROWTH", message)
        
        self.last_alert_time["rapid_growth"] = time.time()
        
    def _log_periodic_status(self, current_memory_mb: float):
        """Log periodic status updates"""
        
        # Log every 60 seconds
        if hasattr(self, '_last_status_log'):
            if time.time() - self._last_status_log < 60:
                return
                
        status_emoji = "✅" if current_memory_mb <= self.target_memory_mb else "⚠️"
        
        print(f"{status_emoji} Memory: {current_memory_mb:.1f}MB (target: {self.target_memory_mb}MB) - {datetime.now().strftime('%H:%M:%S')}")
        
        self._last_status_log = time.time()
        
    def _log_alert(self, level: str, message: str):
        """Log alert to file"""
        
        # Create logs directory
        logs_dir = Path("monitoring_logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Log to file
        log_file = logs_dir / f"memory_alerts_{datetime.now().strftime('%Y%m%d')}.log"
        
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from monitoring history"""
        
        if not self.performance_history:
            return {"error": "No performance data available"}
            
        # Calculate statistics from recent history
        recent_data = self.performance_history[-100:] if len(self.performance_history) > 100 else self.performance_history
        
        memories = [d["memory_mb"] for d in recent_data]
        object_counts = [d["gc_objects"] for d in recent_data]
        
        summary = {
            "monitoring_duration_minutes": (time.time() - self.performance_history[0]["timestamp"]) / 60,
            "total_measurements": len(self.performance_history),
            "recent_measurements": len(recent_data),
            "memory_stats": {
                "current_mb": memories[-1] if memories else 0,
                "average_mb": sum(memories) / len(memories) if memories else 0,
                "min_mb": min(memories) if memories else 0,
                "max_mb": max(memories) if memories else 0,
                "target_mb": self.target_memory_mb
            },
            "object_stats": {
                "current_objects": object_counts[-1] if object_counts else 0,
                "average_objects": sum(object_counts) / len(object_counts) if object_counts else 0,
                "min_objects": min(object_counts) if object_counts else 0,
                "max_objects": max(object_counts) if object_counts else 0
            },
            "alerts_sent": len(self.last_alert_time),
            "target_compliance": (sum(1 for m in memories if m <= self.target_memory_mb) / len(memories)) * 100 if memories else 0
        }
        
        return summary
        
    def save_performance_data(self):
        """Save performance data to file"""
        
        if not self.performance_history:
            return
            
        # Create data directory
        data_dir = Path("monitoring_data")
        data_dir.mkdir(exist_ok=True)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = data_dir / f"memory_performance_{timestamp}.json"
        
        performance_data = {
            "monitoring_config": {
                "target_memory_mb": self.target_memory_mb,
                "alert_threshold_mb": self.alert_threshold_mb,
                "critical_threshold_mb": self.critical_threshold_mb,
                "monitoring_interval": self.monitoring_interval
            },
            "performance_history": self.performance_history,
            "summary": self.get_performance_summary()
        }
        
        with open(data_file, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
            
        print(f"📄 Performance data saved: {data_file}")
        
    def run_monitoring(self, duration_minutes: Optional[int] = None):
        """Run monitoring for specified duration or indefinitely"""
        
        self.start_monitoring()
        
        try:
            if duration_minutes:
                print(f"⏱️ Running monitoring for {duration_minutes} minutes...")
                time.sleep(duration_minutes * 60)
                
            else:
                print("♾️ Running monitoring indefinitely (Ctrl+C to stop)...")
                
                # Keep main thread alive
                while self.monitoring_active:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n👋 Monitoring interrupted by user")
            
        finally:
            self.stop_monitoring()
            
            # Save final data
            self.save_performance_data()
            
            # Print final summary
            summary = self.get_performance_summary()
            print(f"\n📊 MONITORING SUMMARY:")
            print(f"Duration: {summary['monitoring_duration_minutes']:.1f} minutes")
            print(f"Measurements: {summary['total_measurements']}")
            print(f"Average Memory: {summary['memory_stats']['average_mb']:.1f}MB")
            print(f"Target Compliance: {summary['target_compliance']:.1f}%")
            print(f"Alerts Sent: {summary['alerts_sent']}")


def main():
    """Main monitoring function"""
    
    # Parse command line arguments
    target_memory = 200
    alert_threshold = 300
    critical_threshold = 500
    interval = 5.0
    duration = None
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--target" and i + 1 < len(sys.argv):
            target_memory = int(sys.argv[i + 1])
            i += 2
        elif arg == "--alert" and i + 1 < len(sys.argv):
            alert_threshold = int(sys.argv[i + 1])
            i += 2
        elif arg == "--critical" and i + 1 < len(sys.argv):
            critical_threshold = int(sys.argv[i + 1])
            i += 2
        elif arg == "--interval" and i + 1 < len(sys.argv):
            interval = float(sys.argv[i + 1])
            i += 2
        elif arg == "--duration" and i + 1 < len(sys.argv):
            duration = int(sys.argv[i + 1])
            i += 2
        elif arg in ["-h", "--help"]:
            print("""
Production Memory Monitor

Usage: python monitor_memory.py [options]

Options:
  --target MB       Target memory usage (default: 200)
  --alert MB        Alert threshold (default: 300)
  --critical MB     Critical threshold (default: 500)
  --interval SECS   Monitoring interval (default: 5.0)
  --duration MINS   Run for specific duration (default: indefinite)
  -h, --help        Show this help
            """)
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)
            
    # Create and run monitor
    monitor = ProductionMemoryMonitor(
        target_memory_mb=target_memory,
        alert_threshold_mb=alert_threshold,
        critical_threshold_mb=critical_threshold,
        monitoring_interval=interval
    )
    
    monitor.run_monitoring(duration)


if __name__ == "__main__":
    main()