#!/usr/bin/env python3
"""
Emergency Memory Recovery System - CRITICAL MEMORY CRISIS HANDLER
Comprehensive system to handle 77-97% memory usage situations
"""

import gc
import sys
import os
import time
import threading
import weakref
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import tracemalloc


@dataclass
class MemoryCrisisReport:
    """Report for memory crisis recovery operations"""
    initial_memory_mb: float
    final_memory_mb: float
    memory_recovered_mb: float
    recovery_percentage: float
    operations_performed: List[str]
    time_taken_seconds: float
    success: bool
    critical_threshold_reached: bool


class EmergencyMemoryRecovery:
    """
    CRITICAL: Emergency memory recovery for 77-97% memory usage
    Implements aggressive memory reduction strategies
    """
    
    def __init__(self):
        self.CRITICAL_MEMORY_THRESHOLD_MB = 1500  # 1.5GB
        self.EMERGENCY_THRESHOLD_MB = 1000        # 1GB
        self.TARGET_MEMORY_MB = 500               # 500MB target
        
        # Recovery strategy priorities
        self.recovery_strategies = [
            ("gc_optimization", self._emergency_gc_recovery),
            ("cache_clearing", self._clear_all_caches),
            ("module_unloading", self._unload_heavy_modules),
            ("object_pool_cleanup", self._cleanup_object_pools),
            ("widget_cleanup", self._cleanup_widgets),
            ("circular_reference_breaking", self._break_circular_references),
            ("memory_mapping_cleanup", self._cleanup_memory_mappings),
            ("final_aggressive_gc", self._final_aggressive_collection)
        ]
        
        self.recovery_history = []
        self.monitoring_active = False
        
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_memory_percentage(self) -> float:
        """Get memory usage percentage of system"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception:
            return 0.0
    
    def is_memory_critical(self) -> Tuple[bool, float, str]:
        """Check if memory usage is at critical levels"""
        memory_mb = self._get_memory_usage_mb()
        memory_pct = self._get_memory_percentage()
        
        if memory_mb > self.CRITICAL_MEMORY_THRESHOLD_MB or memory_pct > 95:
            return True, memory_mb, "CRITICAL"
        elif memory_mb > self.EMERGENCY_THRESHOLD_MB or memory_pct > 85:
            return True, memory_mb, "EMERGENCY"
        elif memory_mb > self.TARGET_MEMORY_MB or memory_pct > 70:
            return True, memory_mb, "WARNING"
        else:
            return False, memory_mb, "NORMAL"
    
    def execute_emergency_recovery(self) -> MemoryCrisisReport:
        """Execute complete emergency memory recovery"""
        print("🚨 EMERGENCY MEMORY RECOVERY ACTIVATED")
        
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        operations_performed = []
        
        print(f"Initial Memory: {initial_memory:.1f}MB")
        
        is_critical, current_memory, status = self.is_memory_critical()
        print(f"Memory Status: {status} ({current_memory:.1f}MB)")
        
        # Execute recovery strategies in priority order
        for strategy_name, strategy_func in self.recovery_strategies:
            print(f"\n📈 Executing: {strategy_name}")
            
            before_memory = self._get_memory_usage_mb()
            
            try:
                strategy_result = strategy_func()
                operations_performed.append(f"{strategy_name}: {strategy_result}")
                
                after_memory = self._get_memory_usage_mb()
                freed_mb = before_memory - after_memory
                
                print(f"  ✅ Freed {freed_mb:.1f}MB")
                
                # Check if we've reached target
                if after_memory <= self.TARGET_MEMORY_MB:
                    print(f"🎯 Target memory achieved: {after_memory:.1f}MB")
                    break
                    
                # Give system time to stabilize
                time.sleep(0.5)
                
            except Exception as e:
                error_msg = f"{strategy_name}: ERROR - {str(e)}"
                operations_performed.append(error_msg)
                print(f"  ❌ {error_msg}")
                continue
        
        # Final measurements
        end_time = time.time()
        final_memory = self._get_memory_usage_mb()
        memory_recovered = initial_memory - final_memory
        recovery_percentage = (memory_recovered / initial_memory * 100) if initial_memory > 0 else 0
        time_taken = end_time - start_time
        
        # Create recovery report
        report = MemoryCrisisReport(
            initial_memory_mb=initial_memory,
            final_memory_mb=final_memory,
            memory_recovered_mb=memory_recovered,
            recovery_percentage=recovery_percentage,
            operations_performed=operations_performed,
            time_taken_seconds=time_taken,
            success=final_memory < initial_memory,
            critical_threshold_reached=final_memory <= self.TARGET_MEMORY_MB
        )
        
        print(f"\n🎯 EMERGENCY RECOVERY COMPLETE")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB") 
        print(f"  Recovered: {memory_recovered:.1f}MB ({recovery_percentage:.1f}%)")
        print(f"  Time: {time_taken:.2f}s")
        print(f"  Success: {'✅' if report.success else '❌'}")
        
        self.recovery_history.append(report)
        return report
    
    def _emergency_gc_recovery(self) -> str:
        """Emergency garbage collection recovery"""
        initial_objects = len(gc.get_objects())
        
        # Configure GC for aggressive collection
        old_thresholds = gc.get_threshold()
        gc.set_threshold(50, 3, 3)  # Very aggressive
        
        total_collected = 0
        
        # Multiple collection passes
        for generation in range(3):
            for _ in range(10):  # 10 passes per generation
                collected = gc.collect(generation)
                total_collected += collected
                if collected == 0:
                    break
        
        # Final comprehensive collections
        for _ in range(10):
            total_collected += gc.collect()
        
        # Restore thresholds
        gc.set_threshold(*old_thresholds)
        
        final_objects = len(gc.get_objects())
        objects_freed = initial_objects - final_objects
        
        return f"Objects freed: {objects_freed:,}, Collected: {total_collected:,}"
    
    def _clear_all_caches(self) -> str:
        """Clear all possible caches"""
        cleared_count = 0
        
        # Clear function caches (lru_cache, etc.)
        for obj in gc.get_objects():
            if hasattr(obj, 'cache_clear') and callable(getattr(obj, 'cache_clear')):
                try:
                    obj.cache_clear()
                    cleared_count += 1
                except Exception:
                    pass
        
        # Clear type cache
        try:
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
                cleared_count += 1
        except Exception:
            pass
        
        # Clear import cache
        try:
            import importlib
            importlib.invalidate_caches()
            cleared_count += 1
        except Exception:
            pass
        
        # Clear regex cache
        try:
            import re
            re.purge()
            cleared_count += 1
        except Exception:
            pass
        
        return f"Caches cleared: {cleared_count}"
    
    def _unload_heavy_modules(self) -> str:
        """Unload heavy/unused modules"""
        heavy_modules = [
            'numpy', 'pandas', 'matplotlib', 'plotly', 'seaborn',
            'scipy', 'sklearn', 'tensorflow', 'torch', 'keras',
            'PIL', 'cv2', 'skimage', 'networkx', 'requests_cache'
        ]
        
        unloaded = 0
        for module_name in heavy_modules:
            if module_name in sys.modules:
                try:
                    # Check if module has few references (safe to unload)
                    module = sys.modules[module_name]
                    if sys.getrefcount(module) <= 5:  # Conservative threshold
                        del sys.modules[module_name]
                        unloaded += 1
                except Exception:
                    pass
        
        # Additional cleanup of module fragments
        module_fragments = []
        for name in list(sys.modules.keys()):
            for heavy in heavy_modules:
                if name.startswith(f"{heavy}.") or name.endswith(f".{heavy}"):
                    module_fragments.append(name)
        
        for fragment in module_fragments:
            try:
                if fragment in sys.modules:
                    del sys.modules[fragment]
                    unloaded += 1
            except Exception:
                pass
        
        return f"Modules unloaded: {unloaded}"
    
    def _cleanup_object_pools(self) -> str:
        """Cleanup object pools if available"""
        try:
            from .object_pool import emergency_pool_cleanup
            emergency_pool_cleanup()
            return "Object pools cleaned"
        except ImportError:
            try:
                from object_pool import emergency_pool_cleanup
                emergency_pool_cleanup()
                return "Object pools cleaned"
            except ImportError:
                return "Object pools not available"
    
    def _cleanup_widgets(self) -> str:
        """Cleanup widget memory if available"""
        try:
            from .widget_memory_manager import emergency_widget_cleanup
            stats = emergency_widget_cleanup()
            return f"Widgets cleaned: {stats}"
        except ImportError:
            return "Widget cleanup not available"
    
    def _break_circular_references(self) -> str:
        """Break circular references aggressively"""
        # Enable GC debugging to find circular references
        original_debug = gc.get_debug()
        gc.set_debug(gc.DEBUG_SAVEALL)
        
        # Force collection to populate gc.garbage
        collected = gc.collect()
        
        circular_refs_broken = 0
        if hasattr(gc, 'garbage') and gc.garbage:
            for obj in list(gc.garbage):
                try:
                    # Break common circular reference patterns
                    if hasattr(obj, '__dict__'):
                        obj.__dict__.clear()
                        circular_refs_broken += 1
                    elif isinstance(obj, (list, set)):
                        obj.clear()
                        circular_refs_broken += 1
                    elif isinstance(obj, dict):
                        obj.clear()
                        circular_refs_broken += 1
                except Exception:
                    pass
            
            gc.garbage.clear()
        
        # Restore original debug settings
        gc.set_debug(original_debug)
        
        return f"Circular references broken: {circular_refs_broken}"
    
    def _cleanup_memory_mappings(self) -> str:
        """Cleanup memory mappings and file handles"""
        cleaned = 0
        
        # Close file handles that might be holding memory
        for obj in gc.get_objects():
            try:
                if hasattr(obj, 'close') and callable(getattr(obj, 'close')):
                    if hasattr(obj, 'closed') and not obj.closed:
                        # Only close if it's a file-like object
                        if hasattr(obj, 'read') or hasattr(obj, 'write'):
                            obj.close()
                            cleaned += 1
            except Exception:
                pass
        
        return f"File handles closed: {cleaned}"
    
    def _final_aggressive_collection(self) -> str:
        """Final aggressive garbage collection"""
        # Set extremely aggressive thresholds
        gc.set_threshold(10, 1, 1)
        
        total_collected = 0
        
        # Multiple aggressive collection cycles
        for cycle in range(20):
            collected = gc.collect()
            total_collected += collected
            
            if collected == 0 and cycle > 5:
                break
            
            # Brief pause to let system stabilize
            time.sleep(0.1)
        
        # Final burst of collections
        for _ in range(10):
            total_collected += gc.collect()
        
        return f"Final aggressive collection: {total_collected:,} objects"
    
    def start_crisis_monitoring(self, check_interval: float = 10.0):
        """Start continuous monitoring for memory crisis"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        print(f"🔥 Crisis monitoring activated (checking every {check_interval}s)")
        
        def crisis_monitor():
            while self.monitoring_active:
                try:
                    is_critical, memory_mb, status = self.is_memory_critical()
                    
                    if status == "CRITICAL":
                        print(f"🚨 CRITICAL MEMORY CRISIS: {memory_mb:.1f}MB - FULL RECOVERY")
                        self.execute_emergency_recovery()
                    elif status == "EMERGENCY":
                        print(f"⚠️ Emergency memory level: {memory_mb:.1f}MB - QUICK RECOVERY")
                        # Quick but aggressive recovery
                        self._emergency_gc_recovery()
                        self._clear_all_caches()
                    elif status == "WARNING":
                        print(f"⚡ Warning memory level: {memory_mb:.1f}MB - PREVENTIVE MEASURES")
                        # Preventive measures
                        self._clear_all_caches()
                        for _ in range(5):
                            gc.collect()
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    print(f"Crisis monitoring error: {e}")
                    time.sleep(check_interval)
        
        crisis_thread = threading.Thread(target=crisis_monitor, daemon=True)
        crisis_thread.start()
    
    def stop_crisis_monitoring(self):
        """Stop crisis monitoring"""
        self.monitoring_active = False
        print("🔄 Crisis monitoring stopped")
    
    def get_crisis_report(self) -> str:
        """Get comprehensive crisis management report"""
        current_memory = self._get_memory_usage_mb()
        is_critical, _, status = self.is_memory_critical()
        
        recent_recoveries = len([r for r in self.recovery_history if r.time_taken_seconds > 0])
        total_recovered = sum(r.memory_recovered_mb for r in self.recovery_history)
        
        return f"""
🚨 MEMORY CRISIS MANAGEMENT REPORT

Current Status:
- Memory Usage: {current_memory:.1f}MB
- Status: {status}
- Crisis Level: {'🔥 CRITICAL' if is_critical else '✅ NORMAL'}

Recovery History:
- Total Recoveries: {recent_recoveries}
- Memory Recovered: {total_recovered:.1f}MB
- Monitoring: {'🔍 Active' if self.monitoring_active else '⏹️ Inactive'}

Thresholds:
- Critical: {self.CRITICAL_MEMORY_THRESHOLD_MB}MB
- Emergency: {self.EMERGENCY_THRESHOLD_MB}MB  
- Target: {self.TARGET_MEMORY_MB}MB

System Recommendations:
{'🚨 IMMEDIATE RECOVERY NEEDED' if current_memory > self.CRITICAL_MEMORY_THRESHOLD_MB else '✅ Memory usage within acceptable range'}
"""


# Global emergency recovery instance
_global_recovery_system = None

def get_emergency_recovery() -> EmergencyMemoryRecovery:
    """Get global emergency recovery system"""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = EmergencyMemoryRecovery()
    return _global_recovery_system


# Critical convenience functions
def emergency_memory_recovery() -> MemoryCrisisReport:
    """Execute emergency memory recovery immediately"""
    recovery_system = get_emergency_recovery()
    return recovery_system.execute_emergency_recovery()


def check_memory_crisis() -> Tuple[bool, float, str]:
    """Check if system is in memory crisis"""
    recovery_system = get_emergency_recovery()
    return recovery_system.is_memory_critical()


def start_crisis_monitoring():
    """Start automatic crisis monitoring"""
    recovery_system = get_emergency_recovery()
    recovery_system.start_crisis_monitoring()


if __name__ == "__main__":
    print("🚨 EMERGENCY MEMORY RECOVERY SYSTEM TEST")
    
    # Check current status
    is_critical, memory_mb, status = check_memory_crisis()
    print(f"Current status: {status} ({memory_mb:.1f}MB)")
    
    if is_critical:
        print("🔥 EXECUTING EMERGENCY RECOVERY!")
        report = emergency_memory_recovery()
        print(f"Recovery completed: {report.recovery_percentage:.1f}% memory recovered")
    else:
        print("✅ Memory usage is within acceptable range")
        # Still run a quick test recovery
        recovery_system = get_emergency_recovery()
        print("\n🧪 Running test recovery...")
        report = recovery_system.execute_emergency_recovery()
        print(f"Test recovery: {report.memory_recovered_mb:.1f}MB freed")