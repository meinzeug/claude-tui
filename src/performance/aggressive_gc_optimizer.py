#!/usr/bin/env python3
"""
Aggressive Garbage Collection Optimizer - Hive Mind Memory Management
Implements intelligent, adaptive garbage collection strategies to maintain <1.5Gi memory target
"""

import gc
import sys
import os
import time
import threading
import psutil
import weakref
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import tracemalloc
from datetime import datetime, timedelta

# Import memory profiler with fallback
try:
    from .memory_profiler import MemoryProfiler, MemorySnapshot
except ImportError:
    class MemoryProfiler:
        def __init__(self, *args, **kwargs): pass
        def take_snapshot(self):
            class Snapshot:
                def __init__(self):
                    try:
                        process = psutil.Process(os.getpid())
                        self.process_memory = process.memory_info().rss
                        self.gc_objects = len(gc.get_objects())
                    except:
                        self.process_memory = 0
                        self.gc_objects = 0
                    self.timestamp = time.time()
            return Snapshot()


@dataclass
class GCStats:
    """Garbage collection statistics"""
    generation_0_collections: int = 0
    generation_1_collections: int = 0  
    generation_2_collections: int = 0
    total_objects_collected: int = 0
    memory_freed_mb: float = 0.0
    average_collection_time_ms: float = 0.0
    collection_efficiency: float = 0.0  # objects_collected / collection_time
    circular_references_broken: int = 0
    
    @property
    def total_collections(self) -> int:
        return self.generation_0_collections + self.generation_1_collections + self.generation_2_collections


@dataclass
class GCOptimizationStrategy:
    """Garbage collection optimization strategy configuration"""
    name: str
    description: str
    threshold_0: int  # Gen 0 threshold
    threshold_1: int  # Gen 1 threshold  
    threshold_2: int  # Gen 2 threshold
    collection_frequency_seconds: float
    memory_pressure_threshold_mb: float
    aggressive_mode: bool = False
    debug_mode: bool = False


class AggressiveGCOptimizer:
    """
    Aggressive Garbage Collection Optimizer
    Implements intelligent GC strategies to maintain memory under 1.5GB
    """
    
    def __init__(self, target_memory_gb: float = 1.5):
        self.target_memory_bytes = int(target_memory_gb * 1024 * 1024 * 1024)
        self.target_memory_mb = target_memory_gb * 1024
        
        # Memory monitoring
        self.profiler = MemoryProfiler(target_memory_mb=int(self.target_memory_mb))
        
        # GC Statistics tracking
        self.gc_stats = GCStats()
        self.collection_history: deque = deque(maxlen=1000)
        self.memory_timeline: deque = deque(maxlen=500)
        
        # Optimization strategies
        self.strategies = {
            'conservative': GCOptimizationStrategy(
                name='conservative',
                description='Conservative GC for normal operation',
                threshold_0=700,   # Default is 700
                threshold_1=10,    # Default is 10  
                threshold_2=10,    # Default is 10
                collection_frequency_seconds=30.0,
                memory_pressure_threshold_mb=1200,  # 1.2GB
                aggressive_mode=False
            ),
            
            'balanced': GCOptimizationStrategy(
                name='balanced',
                description='Balanced GC for moderate memory pressure',
                threshold_0=500,   # More frequent gen 0
                threshold_1=8,     # More frequent gen 1
                threshold_2=8,     # More frequent gen 2
                collection_frequency_seconds=15.0,
                memory_pressure_threshold_mb=1000,  # 1GB
                aggressive_mode=False
            ),
            
            'aggressive': GCOptimizationStrategy(
                name='aggressive', 
                description='Aggressive GC for high memory pressure',
                threshold_0=300,   # Very frequent gen 0
                threshold_1=5,     # Very frequent gen 1
                threshold_2=5,     # Very frequent gen 2
                collection_frequency_seconds=5.0,
                memory_pressure_threshold_mb=800,   # 800MB
                aggressive_mode=True
            ),
            
            'emergency': GCOptimizationStrategy(
                name='emergency',
                description='Emergency GC for critical memory situations',
                threshold_0=100,   # Extremely frequent gen 0
                threshold_1=3,     # Extremely frequent gen 1
                threshold_2=3,     # Extremely frequent gen 2
                collection_frequency_seconds=1.0,
                memory_pressure_threshold_mb=600,   # 600MB
                aggressive_mode=True,
                debug_mode=True
            )
        }
        
        self.current_strategy = 'conservative'
        self.auto_adaptation = True
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Circular reference tracking
        self.circular_ref_detector = CircularReferenceDetector()
        
        # Memory pressure callbacks
        self.pressure_callbacks: List[Callable[[float], None]] = []
        
        # Store original GC settings for restoration
        self.original_thresholds = gc.get_threshold()
        self.original_debug_flags = gc.get_debug()
        
        # Apply initial strategy
        self.apply_strategy(self.current_strategy)
        
    def add_pressure_callback(self, callback: Callable[[float], None]):
        """Add callback to be called when memory pressure increases"""
        self.pressure_callbacks.append(callback)
    
    def apply_strategy(self, strategy_name: str) -> bool:
        """Apply a specific GC optimization strategy"""
        if strategy_name not in self.strategies:
            print(f"⚠️ Unknown GC strategy: {strategy_name}")
            return False
        
        strategy = self.strategies[strategy_name]
        self.current_strategy = strategy_name
        
        print(f"🎯 Applying GC strategy: {strategy.name}")
        print(f"   Description: {strategy.description}")
        print(f"   Thresholds: ({strategy.threshold_0}, {strategy.threshold_1}, {strategy.threshold_2})")
        
        # Apply GC thresholds
        gc.set_threshold(strategy.threshold_0, strategy.threshold_1, strategy.threshold_2)
        
        # Configure debug mode if needed
        if strategy.debug_mode:
            gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_UNCOLLECTABLE)
        else:
            gc.set_debug(0)
        
        return True
    
    def start_adaptive_monitoring(self):
        """Start adaptive GC monitoring and optimization"""
        if self.monitoring_active:
            return
        
        print("📊 Starting adaptive GC monitoring...")
        self.monitoring_active = True
        
        def monitoring_loop():
            last_collection_time = 0
            
            while self.monitoring_active:
                try:
                    current_time = time.time()
                    
                    # Take memory snapshot
                    snapshot = self.profiler.take_snapshot()
                    current_mb = snapshot.process_memory / 1024 / 1024
                    
                    # Record memory timeline
                    self.memory_timeline.append((current_time, current_mb))
                    
                    # Check for strategy adaptation
                    if self.auto_adaptation:
                        self._adapt_strategy(current_mb, snapshot)
                    
                    # Check if scheduled collection is needed
                    strategy = self.strategies[self.current_strategy]
                    time_since_last = current_time - last_collection_time
                    
                    if time_since_last >= strategy.collection_frequency_seconds:
                        collection_result = self.intelligent_collection()
                        last_collection_time = current_time
                        
                        # Update statistics
                        self._update_statistics(collection_result)
                        
                        # Trigger pressure callbacks if needed
                        if current_mb > strategy.memory_pressure_threshold_mb:
                            for callback in self.pressure_callbacks:
                                try:
                                    callback(current_mb)
                                except Exception as e:
                                    print(f"⚠️ Pressure callback error: {e}")
                    
                    # Sleep based on current strategy
                    sleep_time = min(strategy.collection_frequency_seconds, 10.0)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"GC monitoring error: {e}")
                    time.sleep(5.0)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop adaptive monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        # Restore original GC settings
        gc.set_threshold(*self.original_thresholds)
        gc.set_debug(self.original_debug_flags)
        
        print("🛑 GC monitoring stopped, original settings restored")
    
    def _adapt_strategy(self, current_mb: float, snapshot: MemorySnapshot):
        """Adapt GC strategy based on current memory conditions"""
        
        # Determine appropriate strategy based on memory usage
        if current_mb > self.target_memory_mb:
            # Above target - use emergency strategy
            if self.current_strategy != 'emergency':
                print(f"🚨 Memory above target ({current_mb:.1f}MB), switching to emergency GC")
                self.apply_strategy('emergency')
                
        elif current_mb > self.target_memory_mb * 0.9:
            # Close to target - use aggressive strategy
            if self.current_strategy not in ['aggressive', 'emergency']:
                print(f"⚡ High memory pressure ({current_mb:.1f}MB), switching to aggressive GC")
                self.apply_strategy('aggressive')
                
        elif current_mb > self.target_memory_mb * 0.7:
            # Moderate pressure - use balanced strategy
            if self.current_strategy in ['conservative']:
                print(f"⚖️ Moderate memory pressure ({current_mb:.1f}MB), switching to balanced GC")
                self.apply_strategy('balanced')
                
        elif current_mb < self.target_memory_mb * 0.5:
            # Low memory usage - use conservative strategy
            if self.current_strategy not in ['conservative']:
                print(f"✅ Low memory usage ({current_mb:.1f}MB), switching to conservative GC")
                self.apply_strategy('conservative')
    
    def intelligent_collection(self) -> Dict[str, Any]:
        """Perform intelligent garbage collection with optimization"""
        start_time = time.time()
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        start_objects = start_snapshot.gc_objects
        
        collection_result = {
            'strategy': self.current_strategy,
            'start_memory_mb': start_mb,
            'start_objects': start_objects,
            'collections_performed': [],
            'circular_refs_broken': 0,
            'total_objects_collected': 0,
            'collection_time_ms': 0
        }
        
        strategy = self.strategies[self.current_strategy]
        
        try:
            # Pre-collection optimizations
            if strategy.aggressive_mode:
                # Break circular references first
                circular_refs = self.circular_ref_detector.detect_and_break()
                collection_result['circular_refs_broken'] = circular_refs
                self.gc_stats.circular_references_broken += circular_refs
            
            # Perform generation-specific collections
            total_collected = 0
            
            # Generation 0 (youngest objects)
            gen0_collected = gc.collect(0)
            total_collected += gen0_collected
            collection_result['collections_performed'].append(('gen0', gen0_collected))
            self.gc_stats.generation_0_collections += 1
            
            # Generation 1 (older objects) - collect if strategy allows
            if strategy.aggressive_mode or start_mb > strategy.memory_pressure_threshold_mb:
                gen1_collected = gc.collect(1)
                total_collected += gen1_collected
                collection_result['collections_performed'].append(('gen1', gen1_collected))
                self.gc_stats.generation_1_collections += 1
            
            # Generation 2 (oldest objects) - collect if really needed
            if strategy.aggressive_mode and start_mb > strategy.memory_pressure_threshold_mb * 0.9:
                gen2_collected = gc.collect(2)
                total_collected += gen2_collected
                collection_result['collections_performed'].append(('gen2', gen2_collected))
                self.gc_stats.generation_2_collections += 1
            
            # Full collection for emergency situations
            if self.current_strategy == 'emergency':
                for _ in range(3):  # Multiple passes for thorough cleanup
                    additional_collected = gc.collect()
                    total_collected += additional_collected
                    time.sleep(0.01)  # Brief pause between collections
            
            collection_result['total_objects_collected'] = total_collected
            self.gc_stats.total_objects_collected += total_collected
            
        except Exception as e:
            print(f"⚠️ Error during GC collection: {e}")
            collection_result['error'] = str(e)
        
        # Post-collection measurement
        end_time = time.time()
        end_snapshot = self.profiler.take_snapshot()
        end_mb = end_snapshot.process_memory / 1024 / 1024
        end_objects = end_snapshot.gc_objects
        
        collection_time_ms = (end_time - start_time) * 1000
        memory_freed = start_mb - end_mb
        objects_freed = start_objects - end_objects
        
        collection_result.update({
            'end_memory_mb': end_mb,
            'end_objects': end_objects,
            'memory_freed_mb': memory_freed,
            'objects_freed': objects_freed,
            'collection_time_ms': collection_time_ms,
            'efficiency': total_collected / collection_time_ms if collection_time_ms > 0 else 0
        })
        
        # Update global statistics
        self.gc_stats.memory_freed_mb += max(0, memory_freed)
        if collection_time_ms > 0:
            # Update average collection time (running average)
            if self.gc_stats.average_collection_time_ms == 0:
                self.gc_stats.average_collection_time_ms = collection_time_ms
            else:
                self.gc_stats.average_collection_time_ms = (
                    self.gc_stats.average_collection_time_ms * 0.9 + collection_time_ms * 0.1
                )
        
        # Record in collection history
        self.collection_history.append({
            'timestamp': start_time,
            'strategy': self.current_strategy,
            'memory_before_mb': start_mb,
            'memory_after_mb': end_mb,
            'memory_freed_mb': memory_freed,
            'objects_collected': total_collected,
            'collection_time_ms': collection_time_ms
        })
        
        if memory_freed > 1.0:  # Only log significant collections
            print(f"🗑️ GC ({self.current_strategy}): {memory_freed:.1f}MB freed, "
                  f"{total_collected:,} objects, {collection_time_ms:.1f}ms")
        
        return collection_result
    
    def _update_statistics(self, collection_result: Dict[str, Any]):
        """Update GC statistics from collection result"""
        if collection_result.get('total_objects_collected', 0) > 0 and collection_result.get('collection_time_ms', 0) > 0:
            efficiency = collection_result['total_objects_collected'] / collection_result['collection_time_ms']
            
            # Update running average efficiency
            if self.gc_stats.collection_efficiency == 0:
                self.gc_stats.collection_efficiency = efficiency
            else:
                self.gc_stats.collection_efficiency = (
                    self.gc_stats.collection_efficiency * 0.9 + efficiency * 0.1
                )
    
    def force_emergency_collection(self) -> Dict[str, Any]:
        """Force emergency garbage collection for critical memory situations"""
        print("🚨 EMERGENCY GARBAGE COLLECTION ACTIVATED")
        
        # Temporarily switch to emergency strategy
        original_strategy = self.current_strategy
        self.apply_strategy('emergency')
        
        try:
            # Take baseline measurement
            start_snapshot = self.profiler.take_snapshot()
            start_mb = start_snapshot.process_memory / 1024 / 1024
            
            print(f"   Memory before emergency GC: {start_mb:.1f}MB")
            
            # Perform multiple aggressive collection cycles
            total_collected = 0
            total_memory_freed = 0.0
            
            # Cycle 1: Break circular references
            circular_refs = self.circular_ref_detector.detect_and_break()
            print(f"   Broken circular references: {circular_refs}")
            
            # Cycle 2-6: Multiple full collections
            for cycle in range(5):
                cycle_start_mb = self.profiler.take_snapshot().process_memory / 1024 / 1024
                
                # Collect all generations
                collected = 0
                for generation in [0, 1, 2]:
                    gen_collected = gc.collect(generation)
                    collected += gen_collected
                
                cycle_end_mb = self.profiler.take_snapshot().process_memory / 1024 / 1024
                cycle_freed = cycle_start_mb - cycle_end_mb
                
                total_collected += collected
                total_memory_freed += cycle_freed
                
                print(f"   Cycle {cycle + 1}: {collected:,} objects, {cycle_freed:.1f}MB freed")
                
                # Brief pause between cycles
                time.sleep(0.05)
            
            # Final measurement
            end_snapshot = self.profiler.take_snapshot()
            end_mb = end_snapshot.process_memory / 1024 / 1024
            
            result = {
                'emergency_collection': True,
                'start_memory_mb': start_mb,
                'end_memory_mb': end_mb,
                'total_memory_freed_mb': start_mb - end_mb,
                'total_objects_collected': total_collected,
                'circular_references_broken': circular_refs,
                'collection_cycles': 5,
                'success': end_mb < start_mb
            }
            
            print(f"🎯 Emergency GC complete: {result['total_memory_freed_mb']:.1f}MB freed")
            print(f"   Final memory: {end_mb:.1f}MB")
            
            return result
            
        finally:
            # Restore original strategy
            if original_strategy != 'emergency':
                self.apply_strategy(original_strategy)
    
    def optimize_for_workload(self, workload_type: str) -> bool:
        """Optimize GC for specific workload types"""
        workload_strategies = {
            'interactive': 'balanced',    # UI/TUI applications
            'batch_processing': 'conservative',  # Batch jobs
            'memory_intensive': 'aggressive',    # High memory usage
            'real_time': 'conservative'         # Real-time applications
        }
        
        if workload_type in workload_strategies:
            strategy = workload_strategies[workload_type]
            print(f"🎯 Optimizing GC for {workload_type} workload -> {strategy} strategy")
            return self.apply_strategy(strategy)
        else:
            print(f"⚠️ Unknown workload type: {workload_type}")
            return False
    
    def get_performance_report(self) -> str:
        """Generate comprehensive GC performance report"""
        current_snapshot = self.profiler.take_snapshot()
        current_mb = current_snapshot.process_memory / 1024 / 1024
        
        # Calculate collection frequency
        if len(self.collection_history) > 1:
            recent_collections = list(self.collection_history)[-10:]
            if len(recent_collections) > 1:
                time_span = recent_collections[-1]['timestamp'] - recent_collections[0]['timestamp']
                collection_frequency = len(recent_collections) / time_span if time_span > 0 else 0
            else:
                collection_frequency = 0
        else:
            collection_frequency = 0
        
        # Memory trend analysis
        if len(self.memory_timeline) > 10:
            recent_memory = [mb for t, mb in list(self.memory_timeline)[-10:]]
            memory_trend = "INCREASING" if recent_memory[-1] > recent_memory[0] else "STABLE"
            memory_variance = max(recent_memory) - min(recent_memory)
        else:
            memory_trend = "UNKNOWN"
            memory_variance = 0.0
        
        report = f"""
🗑️ AGGRESSIVE GC OPTIMIZER REPORT
{'='*50}

Current Status:
• Memory Usage: {current_mb:.1f}MB (Target: {self.target_memory_mb:.1f}MB)
• Status: {'✅ OPTIMAL' if current_mb <= self.target_memory_mb else '⚠️ ABOVE TARGET'}
• Current Strategy: {self.current_strategy.upper()}
• Auto-Adaptation: {'✅ ENABLED' if self.auto_adaptation else '❌ DISABLED'}

GC Statistics:
• Total Collections: {self.gc_stats.total_collections:,}
  - Generation 0: {self.gc_stats.generation_0_collections:,}
  - Generation 1: {self.gc_stats.generation_1_collections:,}
  - Generation 2: {self.gc_stats.generation_2_collections:,}
• Objects Collected: {self.gc_stats.total_objects_collected:,}
• Memory Freed: {self.gc_stats.memory_freed_mb:.1f}MB
• Avg Collection Time: {self.gc_stats.average_collection_time_ms:.1f}ms
• Collection Efficiency: {self.gc_stats.collection_efficiency:.1f} obj/ms
• Circular Refs Broken: {self.gc_stats.circular_references_broken:,}

Performance Metrics:
• Collection Frequency: {collection_frequency:.2f} collections/second
• Memory Trend: {memory_trend}
• Memory Variance: {memory_variance:.1f}MB
• Monitoring: {'✅ ACTIVE' if self.monitoring_active else '❌ INACTIVE'}

Current GC Settings:
• Thresholds: {gc.get_threshold()}
• Debug Flags: {gc.get_debug()}

Recommendations:
"""
        
        # Generate recommendations
        if current_mb > self.target_memory_mb:
            report += f"• 🎯 Memory above target - consider emergency collection\n"
        
        if self.gc_stats.average_collection_time_ms > 100:
            report += f"• ⚡ High collection time - consider less aggressive thresholds\n"
        
        if collection_frequency < 0.1:
            report += f"• 📈 Low collection frequency - consider more aggressive strategy\n"
        
        if memory_trend == "INCREASING":
            report += f"• 📊 Memory trend increasing - monitor for leaks\n"
        
        if not self.monitoring_active:
            report += f"• 🔄 Start monitoring for adaptive optimization\n"
        
        report += f"• 🏊‍♂️ Use object pools to reduce allocation pressure\n"
        
        return report
    
    def get_collection_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Get recent collection history"""
        return list(self.collection_history)[-last_n:]
    
    def clear_statistics(self):
        """Clear all GC statistics and history"""
        self.gc_stats = GCStats()
        self.collection_history.clear()
        self.memory_timeline.clear()
        print("📊 GC statistics cleared")


class CircularReferenceDetector:
    """Detects and breaks circular references for memory optimization"""
    
    def __init__(self):
        self.known_circular_types = {
            'frame', 'traceback', 'function', 'method',
            'module', 'class', 'type'
        }
        self.detection_cache: Set[int] = set()
    
    def detect_and_break(self) -> int:
        """Detect and break circular references"""
        if not gc.isenabled():
            return 0
        
        # Get objects with circular references
        try:
            # Force a collection to identify unreachable circular references
            unreachable = gc.collect()
            
            # Get objects that are in cycles
            if hasattr(gc, 'get_referrers') and hasattr(gc, 'get_referents'):
                broken_refs = self._break_known_circular_patterns()
                return unreachable + broken_refs
            
            return unreachable
            
        except Exception as e:
            print(f"⚠️ Circular reference detection error: {e}")
            return 0
    
    def _break_known_circular_patterns(self) -> int:
        """Break known circular reference patterns"""
        broken_count = 0
        
        try:
            # Get all objects
            all_objects = gc.get_objects()
            
            # Sample objects to avoid performance impact
            sample_size = min(10000, len(all_objects))
            sample_objects = all_objects[:sample_size]
            
            for obj in sample_objects:
                try:
                    obj_id = id(obj)
                    
                    # Skip if already processed
                    if obj_id in self.detection_cache:
                        continue
                    
                    obj_type = type(obj).__name__
                    
                    # Check for problematic circular patterns
                    if obj_type in self.known_circular_types:
                        # Try to break references safely
                        if self._safe_break_references(obj):
                            broken_count += 1
                    
                    self.detection_cache.add(obj_id)
                    
                    # Limit cache size
                    if len(self.detection_cache) > 50000:
                        self.detection_cache.clear()
                    
                except (ReferenceError, AttributeError, TypeError):
                    continue
            
            return broken_count
            
        except Exception as e:
            print(f"⚠️ Circular pattern breaking error: {e}")
            return 0
    
    def _safe_break_references(self, obj: Any) -> bool:
        """Safely attempt to break circular references in an object"""
        try:
            # For frame objects, clear locals if possible
            if hasattr(obj, 'f_locals') and hasattr(obj, 'clear'):
                obj.clear()
                return True
            
            # For traceback objects, clear if possible  
            if hasattr(obj, 'tb_frame') and hasattr(obj, 'tb_next'):
                # Don't clear active tracebacks
                if not hasattr(obj, '__traceback__'):
                    try:
                        obj.tb_next = None
                        return True
                    except (AttributeError, TypeError):
                        pass
            
            # For dictionaries with circular references
            if isinstance(obj, dict) and len(obj) > 100:
                # Clear large dictionaries that might have circular refs
                circular_keys = []
                for key, value in obj.items():
                    if value is obj or (hasattr(value, '__dict__') and obj is value.__dict__):
                        circular_keys.append(key)
                
                for key in circular_keys:
                    try:
                        del obj[key]
                        return True
                    except (KeyError, TypeError):
                        pass
            
            return False
            
        except Exception:
            return False


# Convenience functions for global GC optimization
_gc_optimizer: Optional[AggressiveGCOptimizer] = None

def get_gc_optimizer(target_memory_gb: float = 1.5) -> AggressiveGCOptimizer:
    """Get global GC optimizer instance"""
    global _gc_optimizer
    
    if _gc_optimizer is None:
        _gc_optimizer = AggressiveGCOptimizer(target_memory_gb)
    
    return _gc_optimizer

def optimize_gc_for_memory_target(target_memory_gb: float = 1.5) -> Dict[str, Any]:
    """Quick GC optimization for memory target"""
    optimizer = get_gc_optimizer(target_memory_gb)
    return optimizer.intelligent_collection()

def start_adaptive_gc_monitoring():
    """Start adaptive GC monitoring"""
    optimizer = get_gc_optimizer()
    optimizer.start_adaptive_monitoring()

def emergency_gc_cleanup() -> Dict[str, Any]:
    """Emergency GC cleanup"""
    optimizer = get_gc_optimizer()
    return optimizer.force_emergency_collection()

def stop_gc_monitoring():
    """Stop GC monitoring"""
    global _gc_optimizer
    if _gc_optimizer:
        _gc_optimizer.stop_monitoring()

if __name__ == "__main__":
    # Demo aggressive GC optimization
    print("🗑️ Aggressive GC Optimizer - Demo")
    
    optimizer = AggressiveGCOptimizer(target_memory_gb=1.5)
    
    # Show initial state
    print("Initial memory state:")
    snapshot = optimizer.profiler.take_snapshot()
    print(f"Memory: {snapshot.process_memory / 1024 / 1024:.1f}MB")
    print(f"Objects: {snapshot.gc_objects:,}")
    
    # Create some objects to collect
    test_objects = []
    for i in range(10000):
        obj = {
            'id': i,
            'data': list(range(50)),
            'circular': None
        }
        obj['circular'] = obj  # Create circular reference
        test_objects.append(obj)
    
    print(f"\nAfter creating test objects:")
    snapshot = optimizer.profiler.take_snapshot()
    print(f"Memory: {snapshot.process_memory / 1024 / 1024:.1f}MB")
    print(f"Objects: {snapshot.gc_objects:,}")
    
    # Test intelligent collection
    print("\nRunning intelligent GC...")
    result = optimizer.intelligent_collection()
    print(f"Collection result: {result}")
    
    # Test emergency collection
    print("\nRunning emergency GC...")
    emergency_result = optimizer.force_emergency_collection()
    print(f"Emergency result: {emergency_result}")
    
    # Show performance report
    print("\nPerformance Report:")
    print(optimizer.get_performance_report())
    
    # Cleanup
    test_objects.clear()
    optimizer.stop_monitoring()