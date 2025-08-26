#!/usr/bin/env python3
"""
Garbage Collection Optimizer - Emergency Memory Management
Advanced GC tuning and optimization for critical memory reduction
"""

import gc
import sys
import threading
import time
import weakref
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil


@dataclass
class GCStats:
    """Garbage collection statistics"""
    collections: List[int] = field(default_factory=list)
    collected: List[int] = field(default_factory=list) 
    objects_before: int = 0
    objects_after: int = 0
    unreachable: int = 0
    duration_ms: float = 0.0
    memory_freed_bytes: int = 0


class AdvancedGCOptimizer:
    """
    Advanced garbage collection optimizer for emergency memory reduction
    Implements aggressive GC strategies while maintaining performance
    """
    
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        self.original_thresholds = gc.get_threshold()
        self.gc_history = deque(maxlen=100)
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Track object types for optimization
        self.object_type_counts = defaultdict(int)
        self.large_objects_registry = weakref.WeakSet()
        
    def optimize_gc_settings(self) -> Dict[str, Any]:
        """Optimize GC settings for memory reduction"""
        print("🗑️ Optimizing garbage collection settings...")
        
        if self.aggressive:
            # Aggressive settings for emergency memory reduction
            new_thresholds = (100, 5, 5)  # Very frequent collection
            gc.set_threshold(*new_thresholds)
            
            # Disable automatic collection to control timing
            gc.disable()
            
        else:
            # Balanced settings for production
            new_thresholds = (700, 10, 10)
            gc.set_threshold(*new_thresholds)
            
        # Enable debugging for leak detection in development
        if __debug__:
            gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_LEAK)
            
        return {
            "original_thresholds": self.original_thresholds,
            "new_thresholds": new_thresholds,
            "aggressive_mode": self.aggressive,
            "auto_gc_disabled": self.aggressive
        }
        
    def emergency_gc_cycle(self) -> GCStats:
        """Perform emergency garbage collection cycle"""
        print("🚨 Running emergency GC cycle...")
        
        start_time = time.time()
        process = psutil.Process()
        memory_before = process.memory_info().rss
        objects_before = len(gc.get_objects())
        
        stats = GCStats(objects_before=objects_before)
        
        try:
            # Multiple collection passes for thorough cleanup
            for generation in range(3):
                collected = gc.collect(generation)
                stats.collections.append(generation)
                stats.collected.append(collected)
                
                print(f"  Generation {generation}: {collected} objects collected")
                
            # Final full collection
            final_collected = 0
            for _ in range(5):  # Multiple passes
                collected = gc.collect()
                final_collected += collected
                
            stats.unreachable = final_collected
            
            # Post-collection measurements
            objects_after = len(gc.get_objects())
            memory_after = process.memory_info().rss
            duration = (time.time() - start_time) * 1000
            
            stats.objects_after = objects_after
            stats.duration_ms = duration
            stats.memory_freed_bytes = memory_before - memory_after
            
            print(f"  ✅ Emergency GC completed in {duration:.1f}ms")
            print(f"  Objects freed: {objects_before - objects_after:,}")
            print(f"  Memory freed: {stats.memory_freed_bytes / 1024 / 1024:.1f}MB")
            
            self.gc_history.append(stats)
            return stats
            
        except Exception as e:
            print(f"  ❌ Emergency GC failed: {e}")
            stats.duration_ms = (time.time() - start_time) * 1000
            return stats
            
    def analyze_object_types(self) -> Dict[str, Any]:
        """Analyze object types for optimization opportunities"""
        print("🔍 Analyzing object types...")
        
        # Count objects by type
        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)
        large_objects = []
        
        all_objects = gc.get_objects()
        
        for obj in all_objects:
            try:
                obj_type = type(obj).__name__
                type_counts[obj_type] += 1
                
                size = sys.getsizeof(obj, 0)
                type_sizes[obj_type] += size
                
                # Track large objects
                if size > 10240:  # > 10KB
                    large_objects.append((obj_type, size))
                    self.large_objects_registry.add(obj)
                    
            except (TypeError, ReferenceError):
                continue
                
        # Sort by memory usage
        sorted_types = sorted(
            type_sizes.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Find optimization opportunities
        optimization_targets = []
        for obj_type, total_size in sorted_types[:10]:
            count = type_counts[obj_type]
            avg_size = total_size / count if count > 0 else 0
            
            if total_size > 1024 * 1024:  # > 1MB total
                optimization_targets.append({
                    "type": obj_type,
                    "count": count,
                    "total_size_mb": total_size / 1024 / 1024,
                    "avg_size_bytes": avg_size
                })
                
        return {
            "total_objects": len(all_objects),
            "unique_types": len(type_counts),
            "large_objects": len(large_objects),
            "top_memory_users": optimization_targets,
            "largest_single_objects": sorted(large_objects, key=lambda x: x[1], reverse=True)[:10]
        }
        
    def find_circular_references(self) -> List[Any]:
        """Find and break circular references"""
        print("🔄 Finding circular references...")
        
        # Enable GC debugging to find uncollectable objects
        gc.set_debug(gc.DEBUG_SAVEALL)
        
        # Collect garbage to populate gc.garbage
        collected = gc.collect()
        
        # Find circular references in garbage
        circular_refs = []
        if hasattr(gc, 'garbage') and gc.garbage:
            circular_refs = list(gc.garbage)
            
            # Try to break circular references
            for obj in gc.garbage:
                try:
                    # Clear dictionaries that might contain references
                    if hasattr(obj, '__dict__'):
                        obj.__dict__.clear()
                        
                    # Clear lists that might contain references  
                    if isinstance(obj, list):
                        obj.clear()
                        
                    # Clear sets that might contain references
                    if isinstance(obj, set):
                        obj.clear()
                        
                except Exception:
                    pass
                    
            # Clear the garbage list
            gc.garbage.clear()
            
        # Reset debugging
        gc.set_debug(0)
        
        print(f"  Found {len(circular_refs)} circular references")
        return circular_refs
        
    def optimize_weak_references(self) -> Dict[str, int]:
        """Optimize weak reference usage"""
        print("🔗 Optimizing weak references...")
        
        cleared_weakrefs = 0
        cleared_callbacks = 0
        
        # Find and optimize WeakKeyDictionary and WeakValueDictionary objects
        for obj in gc.get_objects():
            try:
                if hasattr(obj, 'clear') and 'Weak' in type(obj).__name__:
                    obj.clear()
                    cleared_weakrefs += 1
                    
                # Clear callback registries
                if hasattr(obj, '_remove') and hasattr(obj, 'clear'):
                    if callable(getattr(obj, 'clear')):
                        obj.clear()
                        cleared_callbacks += 1
                        
            except Exception:
                continue
                
        return {
            "cleared_weakrefs": cleared_weakrefs,
            "cleared_callbacks": cleared_callbacks
        }
        
    def memory_pressure_gc(self, pressure_threshold_mb: float = 500) -> bool:
        """Perform GC based on memory pressure"""
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        
        if current_memory_mb > pressure_threshold_mb:
            print(f"🔥 Memory pressure detected: {current_memory_mb:.1f}MB")
            
            # Progressive GC strategy
            for intensity in range(1, 6):  # 5 levels of intensity
                before_memory = process.memory_info().rss
                
                # Increase collection intensity
                for _ in range(intensity):
                    gc.collect()
                    
                after_memory = process.memory_info().rss
                freed_mb = (before_memory - after_memory) / 1024 / 1024
                
                print(f"  Intensity {intensity}: freed {freed_mb:.1f}MB")
                
                # Check if pressure is relieved
                if after_memory / 1024 / 1024 < pressure_threshold_mb:
                    return True
                    
            return False
        return True
        
    def start_adaptive_gc_monitoring(self, interval_seconds: float = 2.0):
        """Start adaptive GC monitoring based on memory usage"""
        if self.monitoring_active:
            return
            
        print(f"📊 Starting adaptive GC monitoring (every {interval_seconds}s)")
        self.monitoring_active = True
        
        def monitoring_loop():
            process = psutil.Process()
            last_memory = process.memory_info().rss
            
            while self.monitoring_active:
                try:
                    current_memory = process.memory_info().rss
                    memory_mb = current_memory / 1024 / 1024
                    
                    # Memory growth detection
                    growth = current_memory - last_memory
                    growth_mb = growth / 1024 / 1024
                    
                    # Adaptive GC triggering
                    if memory_mb > 800:  # Critical threshold
                        self.emergency_gc_cycle()
                    elif memory_mb > 400:  # Warning threshold
                        gc.collect()
                    elif growth_mb > 50:  # Rapid growth
                        gc.collect()
                        
                    last_memory = current_memory
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"GC monitoring error: {e}")
                    time.sleep(interval_seconds)
                    
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop adaptive GC monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def get_gc_performance_report(self) -> str:
        """Generate GC performance report"""
        if not self.gc_history:
            return "No GC history available"
            
        total_objects_freed = sum(
            stats.objects_before - stats.objects_after 
            for stats in self.gc_history
        )
        
        total_memory_freed = sum(
            stats.memory_freed_bytes 
            for stats in self.gc_history
        )
        
        avg_duration = sum(
            stats.duration_ms 
            for stats in self.gc_history
        ) / len(self.gc_history)
        
        return f"""
🗑️ GARBAGE COLLECTION PERFORMANCE REPORT

Recent Activity:
- GC Cycles: {len(self.gc_history)}
- Objects Freed: {total_objects_freed:,}
- Memory Freed: {total_memory_freed / 1024 / 1024:.1f}MB
- Average Duration: {avg_duration:.1f}ms

Current Settings:
- Thresholds: {gc.get_threshold()}
- Auto GC: {'Disabled' if not gc.isenabled() else 'Enabled'}
- Large Objects Tracked: {len(self.large_objects_registry)}

Efficiency Metrics:
- Objects/ms: {total_objects_freed / max(avg_duration, 1):.1f}
- MB/s: {(total_memory_freed / 1024 / 1024) / max(avg_duration / 1000, 0.001):.1f}
"""
        
    def restore_default_settings(self):
        """Restore default GC settings"""
        gc.set_threshold(*self.original_thresholds)
        gc.enable()
        gc.set_debug(0)
        print("🔄 Restored default GC settings")


# Convenience functions for emergency use
def emergency_gc_optimization() -> Dict[str, Any]:
    """Run emergency GC optimization"""
    optimizer = AdvancedGCOptimizer(aggressive=True)
    
    # Optimize settings
    settings = optimizer.optimize_gc_settings()
    
    # Run emergency cycle
    stats = optimizer.emergency_gc_cycle()
    
    # Analyze objects
    analysis = optimizer.analyze_object_types()
    
    # Find circular references
    circular_refs = optimizer.find_circular_references()
    
    # Optimize weak references
    weak_refs = optimizer.optimize_weak_references()
    
    return {
        "settings": settings,
        "emergency_stats": {
            "objects_freed": stats.objects_before - stats.objects_after,
            "memory_freed_mb": stats.memory_freed_bytes / 1024 / 1024,
            "duration_ms": stats.duration_ms
        },
        "object_analysis": analysis,
        "circular_references": len(circular_refs),
        "weak_references": weak_refs
    }


def quick_gc_cleanup() -> int:
    """Quick GC cleanup for immediate memory relief"""
    freed_objects = 0
    for _ in range(3):
        freed_objects += gc.collect()
    return freed_objects


if __name__ == "__main__":
    # Run emergency GC optimization
    print("🚨 EMERGENCY GC OPTIMIZATION STARTING...")
    
    result = emergency_gc_optimization()
    
    print(f"✅ Emergency GC completed:")
    print(f"  Objects freed: {result['emergency_stats']['objects_freed']:,}")
    print(f"  Memory freed: {result['emergency_stats']['memory_freed_mb']:.1f}MB")
    print(f"  Duration: {result['emergency_stats']['duration_ms']:.1f}ms")