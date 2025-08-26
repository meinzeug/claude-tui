#!/usr/bin/env python3
"""
Advanced Memory Profiler - Comprehensive Memory Analysis and Tracking System
Provides detailed memory profiling, hotspot detection, and optimization recommendations
"""

import gc
import sys
import os
import time
import psutil
import tracemalloc
import threading
import weakref
from collections import defaultdict, deque, OrderedDict
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import linecache
import types


@dataclass
class MemorySnapshot:
    """Comprehensive memory snapshot with detailed breakdown"""
    timestamp: float
    process_memory: int
    heap_memory: int
    gc_objects: int
    tracemalloc_peak: int
    largest_objects: List[Tuple[str, int]] = field(default_factory=list)
    leak_suspects: List[Tuple[str, int, str]] = field(default_factory=list)
    module_memory: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    generation_stats: Dict[str, int] = field(default_factory=dict)
    memory_growth_rate: float = 0.0
    hotspots: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class MemoryLeak:
    """Memory leak detection result"""
    object_type: str
    object_count: int
    total_size: int
    growth_rate: float
    suspect_score: float
    stack_trace: Optional[List[str]] = None
    first_seen: float = 0.0
    last_seen: float = 0.0


class AdvancedMemoryProfiler:
    """
    Advanced Memory Profiler with comprehensive analysis capabilities
    
    Features:
    - Real-time memory tracking
    - Leak detection with AI-powered scoring
    - Hotspot identification
    - Growth rate analysis
    - Optimization recommendations
    - Memory pressure monitoring
    """
    
    def __init__(self, target_mb: int = 100):
        self.target_mb = target_mb
        self.monitoring_active = False
        self.snapshots = deque(maxlen=1000)
        self.leak_tracker = defaultdict(list)
        self.hotspot_tracker = defaultdict(float)
        self.growth_tracker = deque(maxlen=100)
        
        # Advanced tracking
        self.object_lifecycle = weakref.WeakKeyDictionary()
        self.allocation_patterns = defaultdict(list)
        self.memory_pressure_events = []
        
        # Configuration
        self.leak_detection_threshold = 1.5  # 1.5x growth = potential leak
        self.hotspot_threshold_mb = 10       # 10MB threshold for hotspots
        self.monitoring_interval = 1.0       # 1 second default
        
        # Initialize tracemalloc if not active
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames
            
        # Start baseline snapshot
        self.baseline_snapshot = self.take_snapshot()
        
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous memory monitoring with advanced analytics"""
        if self.monitoring_active:
            return
            
        self.monitoring_interval = interval_seconds
        self.monitoring_active = True
        
        print(f"🔍 Advanced memory monitoring started (interval: {interval_seconds}s)")
        
        def monitoring_loop():
            last_snapshot_time = time.time()
            
            while self.monitoring_active:
                try:
                    current_time = time.time()
                    
                    # Take detailed snapshot
                    snapshot = self.take_snapshot()
                    self.snapshots.append(snapshot)
                    
                    # Analyze growth patterns
                    if len(self.snapshots) > 1:
                        self._analyze_growth_patterns()
                        
                    # Detect memory leaks
                    if len(self.snapshots) >= 10:  # Need history for leak detection
                        leaks = self._detect_memory_leaks()
                        if leaks:
                            self._report_memory_leaks(leaks)
                    
                    # Monitor memory pressure
                    self._monitor_memory_pressure(snapshot)
                    
                    # Identify hotspots
                    if current_time - last_snapshot_time >= 5.0:  # Every 5 seconds
                        self._identify_memory_hotspots()
                        last_snapshot_time = current_time
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"Memory monitoring error: {e}")
                    time.sleep(interval_seconds)
                    
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        print("🔍 Advanced memory monitoring stopped")
        
    def take_snapshot(self) -> MemorySnapshot:
        """Take comprehensive memory snapshot"""
        timestamp = time.time()
        
        # Basic memory info
        try:
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss
            heap_memory = process.memory_info().vms
        except Exception:
            process_memory = 0
            heap_memory = 0
            
        # GC info
        gc_objects = len(gc.get_objects())
        generation_stats = {
            f"gen_{i}": stat['collections'] for i, stat in enumerate(gc.get_stats())
        }
        
        # Tracemalloc info
        if tracemalloc.is_tracing():
            tracemalloc_peak = tracemalloc.get_traced_memory()[1]
        else:
            tracemalloc_peak = 0
            
        # Analyze object distribution
        largest_objects = self._find_largest_objects()
        type_distribution = self._analyze_type_distribution()
        module_memory = self._analyze_module_memory()
        
        # Calculate growth rate
        growth_rate = self._calculate_growth_rate()
        
        # Identify current hotspots
        hotspots = self._get_current_hotspots()
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            process_memory=process_memory,
            heap_memory=heap_memory,
            gc_objects=gc_objects,
            tracemalloc_peak=tracemalloc_peak,
            largest_objects=largest_objects,
            module_memory=module_memory,
            type_distribution=type_distribution,
            generation_stats=generation_stats,
            memory_growth_rate=growth_rate,
            hotspots=hotspots
        )
        
        return snapshot
        
    def _find_largest_objects(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Find the largest objects in memory"""
        large_objects = []
        
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                if size > 1024:  # >1KB objects
                    obj_type = type(obj).__name__
                    obj_info = f"{obj_type}"
                    
                    # Add more context for complex objects
                    if hasattr(obj, '__module__'):
                        obj_info = f"{obj.__module__}.{obj_type}"
                    if hasattr(obj, '__name__'):
                        obj_info = f"{obj_info}({obj.__name__})"
                        
                    large_objects.append((obj_info, size))
                    
            except (TypeError, ReferenceError):
                continue
                
        # Sort by size, largest first
        large_objects.sort(key=lambda x: x[1], reverse=True)
        return large_objects[:limit]
        
    def _analyze_type_distribution(self) -> Dict[str, int]:
        """Analyze distribution of object types"""
        type_counts = defaultdict(int)
        
        for obj in gc.get_objects():
            try:
                obj_type = type(obj).__name__
                type_counts[obj_type] += 1
            except (TypeError, ReferenceError):
                continue
                
        return dict(type_counts)
        
    def _analyze_module_memory(self) -> Dict[str, int]:
        """Analyze memory usage by module"""
        module_memory = defaultdict(int)
        
        for obj in gc.get_objects():
            try:
                if hasattr(obj, '__module__'):
                    module_name = obj.__module__
                    if module_name:
                        size = sys.getsizeof(obj)
                        module_memory[module_name] += size
            except (TypeError, ReferenceError, AttributeError):
                continue
                
        return dict(module_memory)
        
    def _calculate_growth_rate(self) -> float:
        """Calculate memory growth rate"""
        if len(self.snapshots) < 2:
            return 0.0
            
        current = self.snapshots[-1].process_memory
        previous = self.snapshots[-2].process_memory
        time_delta = self.snapshots[-1].timestamp - self.snapshots[-2].timestamp
        
        if time_delta > 0:
            growth_rate = (current - previous) / time_delta  # bytes per second
            return growth_rate
        return 0.0
        
    def _get_current_hotspots(self) -> List[Tuple[str, int]]:
        """Get current memory hotspots"""
        hotspots = []
        
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            for stat in top_stats[:10]:
                hotspots.append((f"{stat.traceback.format()[-1]}", stat.size))
                
        return hotspots
        
    def _analyze_growth_patterns(self):
        """Analyze memory growth patterns"""
        if len(self.snapshots) < 2:
            return
            
        current = self.snapshots[-1]
        previous = self.snapshots[-2]
        
        growth = current.process_memory - previous.process_memory
        self.growth_tracker.append(growth)
        
        # Track allocation patterns
        if growth > 0:
            self.allocation_patterns['positive_growth'].append({
                'timestamp': current.timestamp,
                'growth': growth,
                'gc_objects': current.gc_objects
            })
            
    def _detect_memory_leaks(self) -> List[MemoryLeak]:
        """Advanced memory leak detection using multiple heuristics"""
        leaks = []
        
        if len(self.snapshots) < 10:
            return leaks
            
        # Analyze object count growth over time
        type_growth_rates = defaultdict(list)
        
        for snapshot in self.snapshots[-10:]:
            for obj_type, count in snapshot.type_distribution.items():
                type_growth_rates[obj_type].append((snapshot.timestamp, count))
                
        # Calculate growth rates for each type
        for obj_type, data_points in type_growth_rates.items():
            if len(data_points) < 5:
                continue
                
            # Calculate linear regression to detect consistent growth
            growth_rate = self._calculate_linear_regression(data_points)
            
            if growth_rate > self.leak_detection_threshold:
                # Potential leak detected
                current_count = data_points[-1][1]
                
                # Calculate suspect score based on multiple factors
                suspect_score = self._calculate_leak_suspect_score(
                    obj_type, growth_rate, current_count
                )
                
                if suspect_score > 0.7:  # High confidence threshold
                    leak = MemoryLeak(
                        object_type=obj_type,
                        object_count=current_count,
                        total_size=self._estimate_type_memory_usage(obj_type),
                        growth_rate=growth_rate,
                        suspect_score=suspect_score,
                        first_seen=data_points[0][0],
                        last_seen=data_points[-1][0]
                    )
                    leaks.append(leak)
                    
        return leaks
        
    def _calculate_linear_regression(self, data_points: List[Tuple[float, int]]) -> float:
        """Calculate linear regression slope for growth rate"""
        if len(data_points) < 2:
            return 0.0
            
        n = len(data_points)
        sum_x = sum(x for x, y in data_points)
        sum_y = sum(y for x, y in data_points)
        sum_xy = sum(x * y for x, y in data_points)
        sum_x2 = sum(x * x for x, y in data_points)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
        
    def _calculate_leak_suspect_score(self, obj_type: str, growth_rate: float, count: int) -> float:
        """Calculate leak suspect score using multiple heuristics"""
        score = 0.0
        
        # Growth rate factor (0-0.4)
        if growth_rate > 10:
            score += 0.4
        elif growth_rate > 5:
            score += 0.3
        elif growth_rate > 2:
            score += 0.2
        elif growth_rate > 1:
            score += 0.1
            
        # Object count factor (0-0.3)
        if count > 10000:
            score += 0.3
        elif count > 5000:
            score += 0.2
        elif count > 1000:
            score += 0.1
            
        # Object type factor (0-0.3)
        suspicious_types = ['dict', 'list', 'set', 'function', 'method']
        if obj_type in suspicious_types:
            score += 0.2
        if 'wrapper' in obj_type.lower() or 'proxy' in obj_type.lower():
            score += 0.1
            
        return min(score, 1.0)
        
    def _estimate_type_memory_usage(self, obj_type: str) -> int:
        """Estimate memory usage for objects of a specific type"""
        total_size = 0
        count = 0
        
        for obj in gc.get_objects():
            try:
                if type(obj).__name__ == obj_type:
                    total_size += sys.getsizeof(obj)
                    count += 1
                    if count > 1000:  # Limit sampling for performance
                        break
            except (TypeError, ReferenceError):
                continue
                
        return total_size
        
    def _monitor_memory_pressure(self, snapshot: MemorySnapshot):
        """Monitor memory pressure and trigger alerts"""
        memory_mb = snapshot.process_memory / 1024 / 1024
        
        pressure_level = "NORMAL"
        if memory_mb > self.target_mb * 3:
            pressure_level = "CRITICAL"
        elif memory_mb > self.target_mb * 2:
            pressure_level = "HIGH" 
        elif memory_mb > self.target_mb * 1.5:
            pressure_level = "MODERATE"
            
        if pressure_level != "NORMAL":
            event = {
                'timestamp': snapshot.timestamp,
                'memory_mb': memory_mb,
                'pressure_level': pressure_level,
                'growth_rate': snapshot.memory_growth_rate
            }
            self.memory_pressure_events.append(event)
            
            if len(self.memory_pressure_events) >= 5:
                # Keep only recent events
                self.memory_pressure_events = self.memory_pressure_events[-50:]
                
    def _identify_memory_hotspots(self):
        """Identify memory allocation hotspots"""
        if not tracemalloc.is_tracing():
            return
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('filename')
        
        for stat in top_stats[:20]:
            if stat.size > self.hotspot_threshold_mb * 1024 * 1024:
                hotspot_key = f"{stat.traceback.format()[-1]}"
                self.hotspot_tracker[hotspot_key] = stat.size
                
    def _report_memory_leaks(self, leaks: List[MemoryLeak]):
        """Report detected memory leaks"""
        if not leaks:
            return
            
        print(f"\n⚠️ MEMORY LEAKS DETECTED ({len(leaks)} suspects):")
        for leak in leaks[:5]:  # Show top 5
            print(f"  🔍 {leak.object_type}: {leak.object_count:,} objects")
            print(f"     Growth: {leak.growth_rate:.2f}/s, Score: {leak.suspect_score:.2f}")
            print(f"     Memory: {leak.total_size/1024/1024:.1f}MB")
            
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.snapshots:
            return {"error": "No snapshots available"}
            
        latest = self.snapshots[-1]
        
        # Calculate statistics
        memory_mb = latest.process_memory / 1024 / 1024
        target_reduction = max(0, memory_mb - self.target_mb)
        
        # Top memory consumers
        top_modules = sorted(
            latest.module_memory.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Growth analysis
        avg_growth = sum(self.growth_tracker) / len(self.growth_tracker) if self.growth_tracker else 0
        
        # Leak analysis
        leaks = self._detect_memory_leaks()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_memory_mb": memory_mb,
            "target_memory_mb": self.target_mb,
            "reduction_needed_mb": target_reduction,
            "memory_efficiency": (self.target_mb / memory_mb) * 100 if memory_mb > 0 else 100,
            
            "growth_analysis": {
                "average_growth_rate": avg_growth,
                "growth_trend": "increasing" if avg_growth > 0 else "decreasing" if avg_growth < 0 else "stable",
                "recent_snapshots": len(self.snapshots)
            },
            
            "object_analysis": {
                "total_objects": latest.gc_objects,
                "largest_objects": latest.largest_objects[:10],
                "type_distribution": dict(sorted(
                    latest.type_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            },
            
            "module_analysis": {
                "top_memory_modules": [(name, size/1024/1024) for name, size in top_modules],
                "total_modules": len(latest.module_memory)
            },
            
            "leak_analysis": {
                "potential_leaks": len(leaks),
                "high_confidence_leaks": len([l for l in leaks if l.suspect_score > 0.8]),
                "leak_details": [
                    {
                        "type": leak.object_type,
                        "score": leak.suspect_score,
                        "growth_rate": leak.growth_rate,
                        "memory_mb": leak.total_size / 1024 / 1024
                    } for leak in leaks[:5]
                ]
            },
            
            "hotspots": [
                {"location": location, "size_mb": size / 1024 / 1024}
                for location, size in sorted(
                    self.hotspot_tracker.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ],
            
            "optimization_recommendations": self._generate_optimization_recommendations(
                memory_mb, target_reduction, leaks
            )
        }
        
        return report
        
    def _generate_optimization_recommendations(
        self, 
        memory_mb: float, 
        target_reduction: float, 
        leaks: List[MemoryLeak]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered optimization recommendations"""
        recommendations = []
        
        # High-impact recommendations based on memory usage
        if memory_mb > 1000:  # >1GB
            recommendations.append({
                "priority": "CRITICAL",
                "strategy": "Emergency Memory Recovery",
                "impact": "High",
                "description": "Implement emergency memory recovery protocols",
                "estimated_reduction_mb": min(target_reduction * 0.6, 500),
                "actions": [
                    "Run ultimate garbage collection",
                    "Unload heavy modules", 
                    "Clear all caches",
                    "Implement lazy loading"
                ]
            })
            
        if leaks and len([l for l in leaks if l.suspect_score > 0.8]) > 0:
            total_leak_memory = sum(l.total_size for l in leaks) / 1024 / 1024
            recommendations.append({
                "priority": "HIGH",
                "strategy": "Memory Leak Mitigation",
                "impact": "High",
                "description": f"Address {len(leaks)} detected memory leaks",
                "estimated_reduction_mb": min(total_leak_memory, target_reduction * 0.4),
                "actions": [
                    f"Fix {leak.object_type} leak (score: {leak.suspect_score:.2f})"
                    for leak in leaks[:3]
                ]
            })
            
        # Module-specific recommendations
        if self.snapshots:
            latest = self.snapshots[-1]
            large_modules = [(name, size) for name, size in latest.module_memory.items() 
                           if size > 50 * 1024 * 1024]  # >50MB modules
            
            if large_modules:
                recommendations.append({
                    "priority": "MEDIUM",
                    "strategy": "Module Optimization",
                    "impact": "Medium",
                    "description": f"Optimize {len(large_modules)} large modules",
                    "estimated_reduction_mb": min(
                        sum(size for _, size in large_modules) / 1024 / 1024 * 0.3,
                        target_reduction * 0.3
                    ),
                    "actions": [
                        f"Optimize {name} ({size/1024/1024:.1f}MB)"
                        for name, size in large_modules[:5]
                    ]
                })
                
        # Object-specific recommendations
        if memory_mb > 200:  # >200MB
            recommendations.append({
                "priority": "MEDIUM", 
                "strategy": "Object Lifecycle Optimization",
                "impact": "Medium",
                "description": "Optimize object creation and destruction patterns",
                "estimated_reduction_mb": min(target_reduction * 0.2, 100),
                "actions": [
                    "Implement object pooling",
                    "Use weak references for caches",
                    "Optimize data structure choices",
                    "Implement streaming for large datasets"
                ]
            })
            
        return recommendations
        
    def export_profile_data(self, filename: Optional[str] = None) -> str:
        """Export profiling data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/tmp/memory_profile_{timestamp}.json"
            
        report = self.generate_optimization_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📊 Memory profile exported to: {filename}")
        return filename


# Convenience functions for compatibility
class MemoryProfiler(AdvancedMemoryProfiler):
    """Alias for backwards compatibility"""
    pass
    

def emergency_memory_check():
    """Quick emergency memory check"""
    profiler = AdvancedMemoryProfiler()
    snapshot = profiler.take_snapshot()
    memory_mb = snapshot.process_memory / 1024 / 1024
    
    if memory_mb > 1000:
        return f"🚨 CRITICAL: {memory_mb:.1f}MB"
    elif memory_mb > 500:
        return f"⚠️ HIGH: {memory_mb:.1f}MB"
    else:
        return f"✅ GOOD: {memory_mb:.1f}MB"


def force_cleanup():
    """Force aggressive cleanup"""
    # Multiple GC passes
    for _ in range(10):
        collected = gc.collect()
        if collected == 0:
            break
            
    # Clear caches
    try:
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
    except:
        pass
        
    try:
        import linecache
        linecache.clearcache()
    except:
        pass


if __name__ == "__main__":
    # Demo the advanced memory profiler
    profiler = AdvancedMemoryProfiler(target_mb=100)
    
    print("🔍 Advanced Memory Profiler Demo")
    print(f"Current status: {emergency_memory_check()}")
    
    # Start monitoring for 10 seconds
    profiler.start_monitoring(1.0)
    
    try:
        time.sleep(10)
    finally:
        profiler.stop_monitoring()
        
    # Generate and display report
    report = profiler.generate_optimization_report()
    print(f"\n📊 Memory Analysis Complete:")
    print(f"   Current: {report['current_memory_mb']:.1f}MB")
    print(f"   Target: {report['target_memory_mb']:.1f}MB")  
    print(f"   Reduction needed: {report['reduction_needed_mb']:.1f}MB")
    print(f"   Efficiency: {report['memory_efficiency']:.1f}%")
    
    if report['leak_analysis']['potential_leaks'] > 0:
        print(f"   ⚠️ Potential leaks: {report['leak_analysis']['potential_leaks']}")
        
    # Export detailed report
    filename = profiler.export_profile_data()
    print(f"   📄 Full report: {filename}")