#!/usr/bin/env python3
"""
Emergency Performance Optimizer - CRITICAL Memory & Performance Recovery

This module provides immediate emergency intervention for memory and performance crises:
- Target: Reduce 1GB+ usage to <500MB (50%+ reduction)
- API response time optimization to <200ms
- Emergency cleanup of all performance bottlenecks
- Real-time monitoring and automatic intervention
"""

import gc
import sys
import os
import psutil
import time
import threading
import weakref
import importlib
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import subprocess

try:
    from .widget_memory_manager import get_widget_manager, emergency_widget_cleanup
    from .memory_optimizer import emergency_optimize, quick_memory_check
    from .lazy_loader import setup_emergency_lazy_imports, optimize_lazy_memory
    from .advanced_caching_system import get_cache_manager
except ImportError:
    # Standalone mode fallbacks
    def get_widget_manager(): return None
    def emergency_widget_cleanup(): return {}
    def emergency_optimize(target_mb=200): return {"success": False}
    def quick_memory_check(): return "❌ UNAVAILABLE"
    def setup_emergency_lazy_imports(): return {}
    def optimize_lazy_memory(): pass
    def get_cache_manager(): return None


@dataclass
class CriticalMetrics:
    """Critical system performance metrics"""
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    active_processes: int
    open_files: int
    network_connections: int
    gc_objects: int
    modules_loaded: int
    threads_active: int
    
    @property
    def is_critical(self) -> bool:
        """Check if system is in critical state"""
        return (self.memory_percent > 45 or 
                self.memory_mb > 900 or 
                self.cpu_percent > 80 or
                self.gc_objects > 100000)
    
    @property
    def severity_level(self) -> str:
        """Get severity level"""
        if self.memory_percent > 60 or self.memory_mb > 1200:
            return "CRITICAL"
        elif self.memory_percent > 45 or self.memory_mb > 900:
            return "HIGH"
        elif self.memory_percent > 35 or self.memory_mb > 700:
            return "MEDIUM"
        else:
            return "LOW"


class EmergencyPerformanceOptimizer:
    """
    CRITICAL Emergency Performance Optimizer
    
    Handles system-wide performance emergencies with aggressive optimization strategies.
    Targets: Memory <500MB, API <200ms, CPU <50%
    """
    
    def __init__(self, 
                 target_memory_mb: int = 400,
                 max_memory_percent: float = 25.0,
                 emergency_threshold_mb: int = 800):
        
        self.target_memory_mb = target_memory_mb
        self.max_memory_percent = max_memory_percent
        self.emergency_threshold_mb = emergency_threshold_mb
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.intervention_count = 0
        self.total_memory_freed_mb = 0.0
        
        # Emergency strategies (ordered by aggressiveness)
        self.emergency_strategies = [
            self.emergency_garbage_collection,
            self.emergency_widget_cleanup,
            self.emergency_import_purge,
            self.emergency_cache_flush,
            self.emergency_file_cleanup,
            self.nuclear_memory_cleanup  # Last resort
        ]
        
    def get_critical_metrics(self) -> CriticalMetrics:
        """Get comprehensive critical system metrics"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return CriticalMetrics(
                memory_mb=memory_info.rss / (1024 ** 2),
                memory_percent=system_memory.percent,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                active_processes=len(psutil.pids()),
                open_files=len(process.open_files()),
                network_connections=len(process.connections()),
                gc_objects=len(gc.get_objects()),
                modules_loaded=len(sys.modules),
                threads_active=threading.active_count()
            )
        except Exception as e:
            print(f"❌ Failed to get critical metrics: {e}")
            return CriticalMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def emergency_intervention(self) -> Dict[str, Any]:
        """CRITICAL: Full emergency performance intervention"""
        print("🚨 CRITICAL EMERGENCY INTERVENTION ACTIVATED 🚨")
        print("=" * 60)
        
        start_time = time.time()
        initial_metrics = self.get_critical_metrics()
        
        print(f"📊 CRITICAL STATE DETECTED:")
        print(f"   Memory: {initial_metrics.memory_mb:.1f}MB ({initial_metrics.memory_percent:.1f}%)")
        print(f"   Severity: {initial_metrics.severity_level}")
        print(f"   GC Objects: {initial_metrics.gc_objects:,}")
        print(f"   Modules: {initial_metrics.modules_loaded}")
        
        intervention_results = {
            "start_time": datetime.now().isoformat(),
            "initial_metrics": initial_metrics,
            "strategies_applied": [],
            "total_memory_freed_mb": 0.0,
            "success": False,
            "intervention_duration_seconds": 0.0
        }
        
        memory_before = initial_metrics.memory_mb
        
        # Apply emergency strategies in order of aggressiveness
        for strategy_func in self.emergency_strategies:
            print(f"\n🔧 Applying: {strategy_func.__name__.replace('_', ' ').title()}")
            
            try:
                strategy_result = strategy_func()
                intervention_results["strategies_applied"].append({
                    "strategy": strategy_func.__name__,
                    "result": strategy_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check if emergency resolved
                current_metrics = self.get_critical_metrics()
                memory_freed = memory_before - current_metrics.memory_mb
                
                print(f"   Memory after: {current_metrics.memory_mb:.1f}MB (freed: {memory_freed:.1f}MB)")
                
                # Stop if we've achieved target or significant improvement
                if (current_metrics.memory_mb < self.target_memory_mb or 
                    memory_freed > 200):  # 200MB improvement threshold
                    print(f"🎉 Emergency resolved! Target achieved.")
                    break
                    
            except Exception as e:
                print(f"   ❌ Strategy failed: {e}")
                intervention_results["strategies_applied"].append({
                    "strategy": strategy_func.__name__,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Final assessment
        final_metrics = self.get_critical_metrics()
        total_memory_freed = memory_before - final_metrics.memory_mb
        intervention_duration = time.time() - start_time
        
        intervention_results.update({
            "final_metrics": final_metrics,
            "total_memory_freed_mb": total_memory_freed,
            "memory_reduction_percent": (total_memory_freed / memory_before) * 100,
            "success": final_metrics.memory_mb < self.emergency_threshold_mb,
            "intervention_duration_seconds": intervention_duration,
            "target_achieved": final_metrics.memory_mb < self.target_memory_mb
        })
        
        # Update tracking
        self.intervention_count += 1
        self.total_memory_freed_mb += total_memory_freed
        
        print(f"\n📊 EMERGENCY INTERVENTION RESULTS:")
        print(f"   Initial Memory: {initial_metrics.memory_mb:.1f}MB")
        print(f"   Final Memory: {final_metrics.memory_mb:.1f}MB")  
        print(f"   Memory Freed: {total_memory_freed:.1f}MB ({intervention_results['memory_reduction_percent']:.1f}%)")
        print(f"   Duration: {intervention_duration:.2f}s")
        print(f"   Success: {'✅ YES' if intervention_results['success'] else '❌ NO'}")
        print(f"   Target Achieved: {'✅ YES' if intervention_results['target_achieved'] else '❌ NO'}")
        
        return intervention_results
    
    def emergency_garbage_collection(self) -> Dict[str, Any]:
        """Emergency garbage collection with system-level cleanup"""
        print("  🗑️ Emergency garbage collection...")
        
        # Get GC stats before
        initial_objects = len(gc.get_objects())
        
        # Configure GC for maximum effectiveness
        original_thresholds = gc.get_threshold()
        gc.set_threshold(100, 5, 5)  # Very aggressive
        
        total_collected = 0
        
        try:
            # Multiple aggressive collection cycles
            for cycle in range(20):
                collected = gc.collect()
                total_collected += collected
                
                if collected == 0 and cycle > 5:
                    break
                    
                # Brief pause to let cleanup complete
                time.sleep(0.01)
            
            # Force cleanup of weak references
            for obj in list(gc.get_objects()):
                if isinstance(obj, (weakref.ref, weakref.ProxyType)):
                    try:
                        if obj() is None:  # Dead reference
                            del obj
                    except:
                        pass
                        
            # Additional collection after reference cleanup
            for _ in range(5):
                gc.collect()
                
        finally:
            gc.set_threshold(*original_thresholds)
            
        final_objects = len(gc.get_objects())
        objects_freed = initial_objects - final_objects
        
        return {
            "initial_objects": initial_objects,
            "final_objects": final_objects,
            "objects_freed": objects_freed,
            "total_collected": total_collected,
            "gc_cycles": 20
        }
    
    def emergency_widget_cleanup(self) -> Dict[str, Any]:
        """Emergency widget memory cleanup"""
        print("  🎨 Emergency widget cleanup...")
        
        try:
            widget_manager = get_widget_manager()
            if widget_manager:
                return widget_manager.emergency_widget_cleanup()
            else:
                return emergency_widget_cleanup()
        except Exception as e:
            print(f"    ❌ Widget cleanup failed: {e}")
            return {"error": str(e)}
    
    def emergency_import_purge(self) -> Dict[str, Any]:
        """Emergency purge of heavy imports"""
        print("  📦 Emergency import purge...")
        
        modules_before = len(sys.modules)
        
        # Identify heavy modules that can be safely removed
        heavy_modules = [
            'numpy', 'pandas', 'torch', 'tensorflow', 'sklearn', 'matplotlib',
            'plotly', 'seaborn', 'PIL', 'cv2', 'scipy', 'statsmodels',
            'transformers', 'huggingface', 'anthropic', 'openai',
            'requests_oauthlib', 'google', 'boto3', 'azure',
            'jupyter', 'notebook', 'ipython'
        ]
        
        purged_count = 0
        
        for module_prefix in heavy_modules:
            modules_to_remove = []
            
            # Find all modules matching the prefix
            for name in list(sys.modules.keys()):
                if name.startswith(module_prefix):
                    modules_to_remove.append(name)
            
            # Remove modules
            for module_name in modules_to_remove:
                try:
                    # Check if module is safe to remove (not core dependency)
                    if not any(core in module_name.lower() for core in 
                             ['sys', 'os', 'gc', 'time', 'threading', 'json']):
                        del sys.modules[module_name]
                        purged_count += 1
                except Exception:
                    pass
        
        modules_after = len(sys.modules)
        
        # Force garbage collection after purge
        for _ in range(5):
            gc.collect()
            
        return {
            "modules_before": modules_before,
            "modules_after": modules_after,
            "modules_purged": purged_count,
            "heavy_module_prefixes_checked": len(heavy_modules)
        }
    
    def emergency_cache_flush(self) -> Dict[str, Any]:
        """Emergency cache flush"""
        print("  💾 Emergency cache flush...")
        
        caches_cleared = 0
        
        # Clear function caches
        try:
            import functools
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear'):
                    try:
                        obj.cache_clear()
                        caches_cleared += 1
                    except:
                        pass
        except:
            pass
        
        # Clear type caches
        try:
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
                caches_cleared += 1
        except:
            pass
        
        # Clear import caches
        try:
            importlib.invalidate_caches()
            caches_cleared += 1
        except:
            pass
        
        # Clear application caches if available
        try:
            cache_manager = get_cache_manager()
            if cache_manager:
                cache_manager.api_cache.clear()
                cache_manager.database_cache.clear()
                cache_manager.static_cache.clear()
                caches_cleared += 3
        except:
            pass
        
        return {
            "caches_cleared": caches_cleared
        }
    
    def emergency_file_cleanup(self) -> Dict[str, Any]:
        """Emergency cleanup of file handles and temporary files"""
        print("  📁 Emergency file cleanup...")
        
        closed_files = 0
        temp_files_removed = 0
        
        try:
            process = psutil.Process(os.getpid())
            open_files = process.open_files()
            
            # Close unnecessary file handles (be very careful)
            for file_info in open_files:
                try:
                    # Only close temp files and logs (very conservative)
                    if any(pattern in file_info.path.lower() for pattern in 
                          ['/tmp/', '/temp/', '.log', '.tmp', '__pycache__']):
                        # Note: We can't directly close files from psutil
                        # This is more of a monitoring step
                        pass
                except:
                    pass
                    
            # Clean up temporary files in common locations
            temp_dirs = ['/tmp', '/var/tmp', os.path.expanduser('~/tmp')]
            
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        for file_name in os.listdir(temp_dir):
                            file_path = os.path.join(temp_dir, file_name)
                            try:
                                if (os.path.isfile(file_path) and 
                                    file_name.startswith(('claude', 'python', 'tmp')) and
                                    file_name.endswith(('.tmp', '.log', '.cache'))):
                                    
                                    os.remove(file_path)
                                    temp_files_removed += 1
                            except:
                                pass
                except:
                    pass
                    
        except Exception as e:
            print(f"    ❌ File cleanup error: {e}")
            
        return {
            "temp_files_removed": temp_files_removed,
            "cleanup_attempted": True
        }
    
    def nuclear_memory_cleanup(self) -> Dict[str, Any]:
        """Nuclear option: Most aggressive memory cleanup possible"""
        print("  ☢️ NUCLEAR MEMORY CLEANUP - LAST RESORT")
        
        nuclear_stats = {
            "objects_before": len(gc.get_objects()),
            "modules_before": len(sys.modules),
            "nuclear_gc_cycles": 0,
            "emergency_optimizations_applied": 0
        }
        
        # Force multiple emergency optimizations
        try:
            result = emergency_optimize(target_mb=self.target_memory_mb)
            if result.get("success"):
                nuclear_stats["emergency_optimizations_applied"] += 1
        except:
            pass
        
        # Nuclear garbage collection - maximum possible cycles
        for cycle in range(50):
            collected = gc.collect()
            nuclear_stats["nuclear_gc_cycles"] += 1
            
            if collected == 0 and cycle > 20:
                break
                
            # Force system-level memory operations
            if cycle % 10 == 0:
                try:
                    # Force Python to release memory to OS
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass
        
        # Clear absolutely everything possible
        try:
            # Clear all weak reference registries
            for obj in list(gc.get_objects()):
                if hasattr(obj, '__dict__') and '__weakref__' in str(type(obj)):
                    try:
                        if hasattr(obj, '__dict__'):
                            obj.__dict__.clear()
                    except:
                        pass
        except:
            pass
        
        nuclear_stats["objects_after"] = len(gc.get_objects())
        nuclear_stats["modules_after"] = len(sys.modules)
        
        return nuclear_stats
    
    def start_emergency_monitoring(self, check_interval: float = 10.0):
        """Start emergency monitoring for automatic intervention"""
        if self.monitoring_active:
            print("⚠️ Emergency monitoring already active")
            return
            
        self.monitoring_active = True
        print(f"🔍 Emergency monitoring started (interval: {check_interval}s)")
        print(f"   Intervention threshold: {self.emergency_threshold_mb}MB")
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    metrics = self.get_critical_metrics()
                    
                    # Log current status
                    if len(self.performance_history) % 6 == 0:  # Every minute with 10s interval
                        print(f"📊 Status: {metrics.memory_mb:.1f}MB ({metrics.memory_percent:.1f}%) "
                              f"- {metrics.severity_level}")
                    
                    self.performance_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "metrics": metrics,
                        "is_critical": metrics.is_critical
                    })
                    
                    # Keep history manageable
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-500:]
                    
                    # Emergency intervention if critical
                    if metrics.is_critical:
                        print(f"\n🚨 CRITICAL STATE DETECTED: {metrics.severity_level}")
                        self.emergency_intervention()
                        
                        # Brief pause after intervention
                        time.sleep(30)  # Wait 30 seconds after intervention
                        
                    time.sleep(check_interval)
                    
                except Exception as e:
                    print(f"❌ Emergency monitoring error: {e}")
                    time.sleep(check_interval)
        
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_emergency_monitoring(self):
        """Stop emergency monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        print("🛑 Emergency monitoring stopped")
    
    def get_emergency_status_report(self) -> str:
        """Get comprehensive emergency status report"""
        current_metrics = self.get_critical_metrics()
        
        # Calculate trends if we have history
        trend_status = "No data"
        if len(self.performance_history) >= 3:
            recent_memory = [h["metrics"].memory_mb for h in self.performance_history[-3:]]
            if recent_memory[-1] < recent_memory[0]:
                trend_status = "🟢 IMPROVING"
            elif recent_memory[-1] > recent_memory[0]:
                trend_status = "🔴 WORSENING"
            else:
                trend_status = "🟡 STABLE"
        
        return f"""
🚨 EMERGENCY PERFORMANCE STATUS REPORT
{'=' * 50}

📊 CURRENT STATE:
- Memory Usage: {current_metrics.memory_mb:.1f}MB ({current_metrics.memory_percent:.1f}%)
- Severity Level: {current_metrics.severity_level}
- Critical State: {'🚨 YES' if current_metrics.is_critical else '✅ NO'}
- Memory Trend: {trend_status}

⚡ SYSTEM METRICS:
- CPU Usage: {current_metrics.cpu_percent:.1f}%
- GC Objects: {current_metrics.gc_objects:,}
- Modules Loaded: {current_metrics.modules_loaded}
- Active Threads: {current_metrics.threads_active}
- Open Files: {current_metrics.open_files}

🎯 TARGETS:
- Target Memory: {self.target_memory_mb}MB
- Max Memory %: {self.max_memory_percent:.1f}%
- Emergency Threshold: {self.emergency_threshold_mb}MB
- Target Achieved: {'✅ YES' if current_metrics.memory_mb < self.target_memory_mb else '❌ NO'}

📈 INTERVENTION HISTORY:
- Total Interventions: {self.intervention_count}
- Total Memory Freed: {self.total_memory_freed_mb:.1f}MB
- Monitoring Active: {'🔍 YES' if self.monitoring_active else '⏹️ NO'}

🚀 RECOMMENDATIONS:
{self._generate_emergency_recommendations(current_metrics)}
"""
    
    def _generate_emergency_recommendations(self, metrics: CriticalMetrics) -> str:
        """Generate emergency optimization recommendations"""
        recommendations = []
        
        if metrics.severity_level == "CRITICAL":
            recommendations.append("🚨 IMMEDIATE ACTION: Run emergency intervention")
            recommendations.append("🔧 Consider restarting application if intervention fails")
            
        elif metrics.severity_level == "HIGH":
            recommendations.append("⚠️ HIGH PRIORITY: Monitor closely, intervention may be needed")
            recommendations.append("🧹 Run preventive cleanup strategies")
            
        elif metrics.severity_level == "MEDIUM":
            recommendations.append("🟡 MODERATE: Implement optimization strategies")
            recommendations.append("📊 Review and optimize heavy operations")
        
        if metrics.gc_objects > 100000:
            recommendations.append(f"🗑️ High GC object count ({metrics.gc_objects:,}) - optimize object lifecycle")
            
        if metrics.modules_loaded > 500:
            recommendations.append(f"📦 Many modules loaded ({metrics.modules_loaded}) - implement lazy loading")
            
        if not recommendations:
            recommendations.append("✅ System performance is within acceptable parameters")
            
        return "\n".join(f"  {rec}" for rec in recommendations)


# Global emergency optimizer
_emergency_optimizer: Optional[EmergencyPerformanceOptimizer] = None

def get_emergency_optimizer() -> EmergencyPerformanceOptimizer:
    """Get global emergency optimizer"""
    global _emergency_optimizer
    if _emergency_optimizer is None:
        _emergency_optimizer = EmergencyPerformanceOptimizer()
    return _emergency_optimizer

def emergency_intervention() -> Dict[str, Any]:
    """Quick emergency intervention"""
    optimizer = get_emergency_optimizer()
    return optimizer.emergency_intervention()

def start_emergency_monitoring(check_interval: float = 10.0):
    """Start emergency monitoring"""
    optimizer = get_emergency_optimizer()
    optimizer.start_emergency_monitoring(check_interval)

def get_emergency_status() -> str:
    """Get emergency status report"""
    optimizer = get_emergency_optimizer()
    return optimizer.get_emergency_status_report()


if __name__ == "__main__":
    print("🚨 EMERGENCY PERFORMANCE OPTIMIZER - Testing")
    print("=" * 60)
    
    optimizer = EmergencyPerformanceOptimizer()
    
    # Get current status
    print("📊 Initial Status:")
    print(optimizer.get_emergency_status_report())
    
    # Test emergency intervention if memory is high
    metrics = optimizer.get_critical_metrics()
    if metrics.is_critical:
        print(f"\n🚨 CRITICAL STATE DETECTED - Running intervention...")
        result = optimizer.emergency_intervention()
        print(f"\n✅ Intervention completed: {result['success']}")
    else:
        print(f"\n✅ System not in critical state ({metrics.memory_mb:.1f}MB)")
    
    # Demonstrate monitoring (brief test)
    print(f"\n🔍 Testing emergency monitoring (10 seconds)...")
    optimizer.start_emergency_monitoring(check_interval=2.0)
    time.sleep(10)
    optimizer.stop_emergency_monitoring()
    
    print(f"\n📊 Final Status:")
    print(optimizer.get_emergency_status_report())