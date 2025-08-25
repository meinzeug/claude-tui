#!/usr/bin/env python3
"""
Emergency Memory Profiler - Critical Performance Optimization
Target: Reduce memory usage from 1.7GB to <200MB (8.5x reduction)
"""

import tracemalloc
import gc
import sys
import psutil
import os
import weakref
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading
import time
import json
from pathlib import Path


@dataclass
class MemorySnapshot:
    """Memory usage snapshot with detailed breakdown"""
    timestamp: float
    total_memory: int
    heap_memory: int
    process_memory: int
    gc_objects: int
    tracemalloc_current: int
    tracemalloc_peak: int
    largest_objects: List[Tuple[str, int]]
    leak_suspects: List[str]


class MemoryProfiler:
    """
    High-performance memory profiler for emergency optimization
    Designed to identify and fix critical memory issues quickly
    """
    
    def __init__(self, target_memory_mb: int = 200):
        self.target_memory_mb = target_memory_mb
        self.target_bytes = target_memory_mb * 1024 * 1024
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring_active = False
        self.leak_detector = MemoryLeakDetector()
        self.object_tracker = ObjectTracker()
        self.profiling_thread: Optional[threading.Thread] = None
        
        # Initialize tracemalloc for detailed tracking
        tracemalloc.start()
        
        # Track process for system-level memory
        self.process = psutil.Process(os.getpid())
        
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous memory monitoring in background thread"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.profiling_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.profiling_thread.start()
        
    def stop_monitoring(self) -> List[MemorySnapshot]:
        """Stop monitoring and return collected snapshots"""
        self.monitoring_active = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=5.0)
        return self.snapshots
        
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                snapshot = self.take_snapshot()
                self.snapshots.append(snapshot)
                
                # Emergency alert if memory exceeds critical threshold
                if snapshot.process_memory > self.target_bytes * 2:
                    self._emergency_cleanup()
                    
                time.sleep(interval)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                
    def take_snapshot(self) -> MemorySnapshot:
        """Take comprehensive memory snapshot"""
        # Force garbage collection for accurate measurement
        gc.collect()
        
        # System memory info
        memory_info = self.process.memory_info()
        
        # Tracemalloc statistics
        current, peak = tracemalloc.get_traced_memory()
        
        # Object counting
        gc_objects = len(gc.get_objects())
        
        # Find largest objects
        largest_objects = self._find_largest_objects()
        
        # Detect potential leaks
        leak_suspects = self.leak_detector.detect_leaks()
        
        return MemorySnapshot(
            timestamp=time.time(),
            total_memory=sys.getsizeof(gc.get_objects()),
            heap_memory=current,
            process_memory=memory_info.rss,
            gc_objects=gc_objects,
            tracemalloc_current=current,
            tracemalloc_peak=peak,
            largest_objects=largest_objects,
            leak_suspects=leak_suspects
        )
        
    def _find_largest_objects(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Find the largest objects in memory"""
        objects = []
        
        # Sample objects to avoid performance impact
        all_objects = gc.get_objects()
        sample_size = min(10000, len(all_objects))
        sample_objects = all_objects[::len(all_objects)//sample_size]
        
        for obj in sample_objects:
            try:
                size = sys.getsizeof(obj, 0)
                if size > 1024:  # Only track objects > 1KB
                    obj_type = type(obj).__name__
                    objects.append((obj_type, size))
            except (TypeError, ReferenceError):
                continue
                
        return sorted(objects, key=lambda x: x[1], reverse=True)[:top_n]
        
    def _emergency_cleanup(self):
        """Emergency memory cleanup when threshold exceeded"""
        print("🚨 EMERGENCY MEMORY CLEANUP ACTIVATED")
        
        # Force aggressive garbage collection
        for _ in range(3):
            gc.collect()
            
        # Clear weak references
        weakref.WeakSet().clear()
        
        # Try to free cached data
        try:
            import numpy as np
            np.ndarray.__del__ = lambda self: None
        except ImportError:
            pass
            
    def analyze_memory_trends(self) -> Dict[str, Any]:
        """Analyze memory usage trends and identify issues"""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots for analysis"}
            
        latest = self.snapshots[-1]
        first = self.snapshots[0]
        
        # Calculate trends
        duration = latest.timestamp - first.timestamp
        memory_growth = latest.process_memory - first.process_memory
        growth_rate = memory_growth / duration if duration > 0 else 0
        
        # Identify critical issues
        issues = []
        
        if latest.process_memory > self.target_bytes:
            excess_mb = (latest.process_memory - self.target_bytes) / 1024 / 1024
            issues.append(f"Memory exceeds target by {excess_mb:.1f}MB")
            
        if growth_rate > 1024 * 1024:  # 1MB/s growth
            issues.append(f"High memory growth rate: {growth_rate/1024/1024:.2f}MB/s")
            
        if latest.gc_objects > 100000:
            issues.append(f"High object count: {latest.gc_objects:,}")
            
        return {
            "current_memory_mb": latest.process_memory / 1024 / 1024,
            "target_memory_mb": self.target_memory_mb,
            "memory_excess_mb": (latest.process_memory - self.target_bytes) / 1024 / 1024,
            "growth_rate_mb_per_second": growth_rate / 1024 / 1024,
            "gc_objects": latest.gc_objects,
            "critical_issues": issues,
            "largest_objects": latest.largest_objects[:5],
            "leak_suspects": latest.leak_suspects,
            "optimization_urgency": self._calculate_urgency(latest)
        }
        
    def _calculate_urgency(self, snapshot: MemorySnapshot) -> str:
        """Calculate optimization urgency level"""
        memory_ratio = snapshot.process_memory / self.target_bytes
        
        if memory_ratio > 8:  # > 1.6GB when target is 200MB
            return "CRITICAL"
        elif memory_ratio > 4:  # > 800MB
            return "HIGH"
        elif memory_ratio > 2:  # > 400MB
            return "MEDIUM"
        else:
            return "LOW"
            
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        analysis = self.analyze_memory_trends()
        
        report = f"""
🚨 MEMORY OPTIMIZATION EMERGENCY REPORT 🚨

Current Status:
- Memory Usage: {analysis['current_memory_mb']:.1f}MB
- Target: {analysis['target_memory_mb']}MB
- Excess: {analysis['memory_excess_mb']:.1f}MB ({analysis['memory_excess_mb']/analysis['target_memory_mb']*100:.1f}% over target)
- Urgency: {analysis['optimization_urgency']}

Growth Analysis:
- Growth Rate: {analysis['growth_rate_mb_per_second']:.2f}MB/s
- Object Count: {analysis['gc_objects']:,}

Critical Issues:
"""
        
        for issue in analysis['critical_issues']:
            report += f"- {issue}\n"
            
        report += "\nLargest Objects:\n"
        for obj_type, size in analysis['largest_objects']:
            report += f"- {obj_type}: {size/1024:.1f}KB\n"
            
        if analysis['leak_suspects']:
            report += "\nPotential Memory Leaks:\n"
            for suspect in analysis['leak_suspects']:
                report += f"- {suspect}\n"
                
        return report
        
    def save_profile_data(self, filepath: str):
        """Save profiling data for detailed analysis"""
        data = {
            "target_memory_mb": self.target_memory_mb,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "memory_mb": s.process_memory / 1024 / 1024,
                    "gc_objects": s.gc_objects,
                    "largest_objects": s.largest_objects,
                    "leak_suspects": s.leak_suspects
                }
                for s in self.snapshots
            ],
            "analysis": self.analyze_memory_trends()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class MemoryLeakDetector:
    """Detects potential memory leaks"""
    
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.previous_counts = defaultdict(int)
        
    def detect_leaks(self) -> List[str]:
        """Detect potential memory leaks by tracking object growth"""
        suspects = []
        
        # Count current objects by type
        current_counts = defaultdict(int)
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            current_counts[obj_type] += 1
            
        # Compare with previous counts
        for obj_type, count in current_counts.items():
            prev_count = self.previous_counts.get(obj_type, 0)
            growth = count - prev_count
            
            # Flag types with significant growth
            if growth > 100 and count > 500:
                suspects.append(f"{obj_type}: {count} objects (+{growth})")
                
        self.previous_counts = current_counts.copy()
        return suspects


class ObjectTracker:
    """Tracks object lifecycle for memory optimization"""
    
    def __init__(self):
        self.tracked_objects = weakref.WeakSet()
        
    def track(self, obj):
        """Track an object for lifecycle monitoring"""
        self.tracked_objects.add(obj)
        
    def get_alive_count(self) -> int:
        """Get count of still-alive tracked objects"""
        return len(self.tracked_objects)


def profile_function_memory(func):
    """Decorator to profile memory usage of a function"""
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        
        # Take before snapshot
        before = profiler.take_snapshot()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Take after snapshot
            after = profiler.take_snapshot()
            
            memory_delta = after.process_memory - before.process_memory
            if memory_delta > 1024 * 1024:  # > 1MB
                print(f"⚠️  Function {func.__name__} used {memory_delta/1024/1024:.2f}MB")
                
    return wrapper


# Emergency memory optimization utility functions
def emergency_memory_check() -> bool:
    """Quick emergency memory check"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 1000:  # > 1GB
        print(f"🚨 EMERGENCY: Memory usage {memory_mb:.1f}MB exceeds 1GB!")
        return False
    return True


def force_cleanup():
    """Force aggressive memory cleanup"""
    # Multiple GC passes
    for _ in range(5):
        gc.collect()
        
    # Clear caches if available
    try:
        import sys
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
    except:
        pass


if __name__ == "__main__":
    # Quick emergency check
    profiler = MemoryProfiler(target_memory_mb=200)
    snapshot = profiler.take_snapshot()
    
    print(profiler.generate_optimization_report())
    
    # Save profile data
    profiler.save_profile_data("/tmp/emergency_memory_profile.json")