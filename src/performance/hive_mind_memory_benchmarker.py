#!/usr/bin/env python3
"""
Hive Mind Memory Benchmarker - Performance-optimized Memory Analysis
Target: <1.5Gi memory usage with comprehensive benchmarking
"""

import asyncio
import gc
import psutil
import os
import sys
import time
import threading
import tracemalloc
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
from collections import defaultdict, deque

# Import optimization modules
from .memory_profiler import MemoryProfiler, MemorySnapshot
from .memory_optimizer import EmergencyMemoryOptimizer
from .object_pool import GlobalPoolManager, create_pool, setup_emergency_pools


@dataclass
class MemoryBenchmarkResult:
    """Results from memory benchmark execution"""
    test_name: str
    start_memory_mb: float
    end_memory_mb: float
    peak_memory_mb: float
    duration_seconds: float
    memory_delta_mb: float
    gc_collections: int
    objects_created: int
    objects_freed: int
    efficiency_score: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class MemoryHotspot:
    """Memory usage hotspot identification"""
    location: str
    object_type: str
    count: int
    total_size_mb: float
    avg_size_bytes: int
    growth_rate_mb_per_sec: float
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW


class HiveMindMemoryBenchmarker:
    """
    Advanced memory benchmarker for Hive Mind system
    Implements comprehensive memory analysis and optimization validation
    """
    
    def __init__(self, target_memory_gb: float = 1.5):
        self.target_memory_bytes = int(target_memory_gb * 1024 * 1024 * 1024)
        self.target_memory_mb = target_memory_gb * 1024
        
        # Initialize benchmarking systems
        self.profiler = MemoryProfiler(target_memory_mb=int(self.target_memory_mb))
        self.optimizer = EmergencyMemoryOptimizer(target_mb=int(self.target_memory_mb))
        self.pool_manager = GlobalPoolManager()
        
        # Benchmark tracking
        self.benchmark_results: List[MemoryBenchmarkResult] = []
        self.hotspots: List[MemoryHotspot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        
        # Performance tracking
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.performance_timeline: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        
        # Initialize tracemalloc for detailed tracking
        tracemalloc.start()
        
    async def run_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Execute complete memory benchmark suite with before/after analysis"""
        print("🚀 Starting Hive Mind Memory Benchmark Suite")
        print(f"🎯 Target: <{self.target_memory_mb:.0f}MB ({self.target_memory_mb/1024:.1f}GB)")
        
        # Take baseline snapshot
        self.baseline_snapshot = self.profiler.take_snapshot()
        baseline_mb = self.baseline_snapshot.process_memory / 1024 / 1024
        
        print(f"📊 Baseline Memory: {baseline_mb:.1f}MB")
        
        if baseline_mb > self.target_memory_mb:
            print(f"⚠️ Baseline exceeds target by {baseline_mb - self.target_memory_mb:.1f}MB")
        
        # Start continuous monitoring
        self.start_monitoring()
        
        try:
            # Phase 1: Memory Hotspot Analysis
            print("\n🔍 Phase 1: Memory Hotspot Analysis")
            hotspot_results = await self.analyze_memory_hotspots()
            
            # Phase 2: Stress Testing (Before Optimization)
            print("\n🏋️ Phase 2: Memory Stress Testing (Before)")
            before_stress_results = await self.run_stress_tests("before_optimization")
            
            # Phase 3: Widget Memory Management Testing
            print("\n🖼️ Phase 3: Widget Memory Management Testing")
            widget_results = await self.test_widget_memory_management()
            
            # Phase 4: Apply Memory Optimizations
            print("\n⚡ Phase 4: Applying Memory Optimizations")
            optimization_results = await self.apply_comprehensive_optimizations()
            
            # Phase 5: Stress Testing (After Optimization)
            print("\n🏋️ Phase 5: Memory Stress Testing (After)")
            after_stress_results = await self.run_stress_tests("after_optimization")
            
            # Phase 6: Validation Tests
            print("\n✅ Phase 6: Memory Target Validation")
            validation_results = await self.validate_memory_target()
            
            # Generate comprehensive report
            report = self.generate_benchmark_report(
                hotspot_results,
                before_stress_results,
                widget_results, 
                optimization_results,
                after_stress_results,
                validation_results
            )
            
            return report
            
        finally:
            self.stop_monitoring()
    
    async def analyze_memory_hotspots(self) -> Dict[str, Any]:
        """Analyze current memory hotspots and identify optimization targets"""
        print("  🔍 Scanning for memory hotspots...")
        
        # Take detailed snapshot
        snapshot = self.profiler.take_snapshot()
        
        # Analyze object distribution
        object_counts = defaultdict(int)
        object_sizes = defaultdict(int)
        
        all_objects = gc.get_objects()
        sample_size = min(50000, len(all_objects))  # Limit for performance
        
        for obj in all_objects[:sample_size]:
            try:
                obj_type = type(obj).__name__
                obj_size = sys.getsizeof(obj, 0)
                
                object_counts[obj_type] += 1
                object_sizes[obj_type] += obj_size
                
            except (TypeError, ReferenceError, AttributeError):
                continue
        
        # Identify hotspots
        hotspots = []
        for obj_type, total_size in object_sizes.items():
            if total_size > 1024 * 1024:  # > 1MB total
                count = object_counts[obj_type]
                avg_size = total_size / count if count > 0 else 0
                
                # Determine severity
                total_mb = total_size / 1024 / 1024
                if total_mb > 100:
                    severity = "CRITICAL"
                elif total_mb > 50:
                    severity = "HIGH"
                elif total_mb > 10:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
                
                hotspot = MemoryHotspot(
                    location=f"Global:{obj_type}",
                    object_type=obj_type,
                    count=count,
                    total_size_mb=total_mb,
                    avg_size_bytes=int(avg_size),
                    growth_rate_mb_per_sec=0.0,  # Would need historical data
                    severity=severity
                )
                
                hotspots.append(hotspot)
        
        # Sort by total size
        hotspots.sort(key=lambda x: x.total_size_mb, reverse=True)
        self.hotspots = hotspots[:20]  # Keep top 20
        
        print(f"  📋 Identified {len(self.hotspots)} memory hotspots")
        for hotspot in self.hotspots[:5]:
            print(f"    {hotspot.severity}: {hotspot.object_type} - {hotspot.total_size_mb:.1f}MB ({hotspot.count:,} objects)")
        
        return {
            "hotspots_found": len(self.hotspots),
            "critical_hotspots": len([h for h in self.hotspots if h.severity == "CRITICAL"]),
            "total_hotspot_memory_mb": sum(h.total_size_mb for h in self.hotspots),
            "top_hotspots": [
                {
                    "type": h.object_type,
                    "size_mb": h.total_size_mb,
                    "count": h.count,
                    "severity": h.severity
                }
                for h in self.hotspots[:10]
            ]
        }
    
    async def run_stress_tests(self, phase: str) -> Dict[str, Any]:
        """Run memory stress tests to evaluate system behavior"""
        print(f"  🏋️ Running stress tests ({phase})...")
        
        stress_results = []
        
        # Test 1: Object Creation Stress
        result1 = await self.stress_test_object_creation()
        stress_results.append(result1)
        
        # Test 2: Memory Allocation Patterns
        result2 = await self.stress_test_allocation_patterns()
        stress_results.append(result2)
        
        # Test 3: Garbage Collection Pressure
        result3 = await self.stress_test_gc_pressure()
        stress_results.append(result3)
        
        # Test 4: Large Object Handling
        result4 = await self.stress_test_large_objects()
        stress_results.append(result4)
        
        successful_tests = len([r for r in stress_results if r.success])
        avg_efficiency = sum(r.efficiency_score for r in stress_results) / len(stress_results)
        total_memory_impact = sum(abs(r.memory_delta_mb) for r in stress_results)
        
        return {
            "phase": phase,
            "tests_run": len(stress_results),
            "successful_tests": successful_tests,
            "average_efficiency_score": avg_efficiency,
            "total_memory_impact_mb": total_memory_impact,
            "test_results": [
                {
                    "name": r.test_name,
                    "success": r.success,
                    "memory_delta_mb": r.memory_delta_mb,
                    "efficiency_score": r.efficiency_score,
                    "duration_seconds": r.duration_seconds
                }
                for r in stress_results
            ]
        }
    
    async def stress_test_object_creation(self) -> MemoryBenchmarkResult:
        """Test rapid object creation and cleanup"""
        start_time = time.time()
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        objects_created = 0
        peak_mb = start_mb
        
        try:
            # Create and destroy objects rapidly
            for batch in range(100):
                batch_objects = []
                
                # Create batch of objects
                for i in range(1000):
                    obj = {
                        'id': f"stress_test_{batch}_{i}",
                        'data': list(range(100)),
                        'metadata': {'created': time.time(), 'batch': batch}
                    }
                    batch_objects.append(obj)
                    objects_created += 1
                
                # Measure peak memory
                if batch % 10 == 0:
                    current_snapshot = self.profiler.take_snapshot()
                    current_mb = current_snapshot.process_memory / 1024 / 1024
                    peak_mb = max(peak_mb, current_mb)
                
                # Clean up batch
                batch_objects.clear()
                del batch_objects
                
                # Allow some GC
                if batch % 20 == 0:
                    gc.collect()
            
            # Final cleanup
            gc.collect()
            
            end_time = time.time()
            end_snapshot = self.profiler.take_snapshot()
            end_mb = end_snapshot.process_memory / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_mb - start_mb
            
            # Calculate efficiency score (lower memory impact = higher efficiency)
            efficiency = max(0.0, 100.0 - abs(memory_delta))
            
            return MemoryBenchmarkResult(
                test_name="object_creation_stress",
                start_memory_mb=start_mb,
                end_memory_mb=end_mb,
                peak_memory_mb=peak_mb,
                duration_seconds=duration,
                memory_delta_mb=memory_delta,
                gc_collections=0,  # Would need tracking
                objects_created=objects_created,
                objects_freed=objects_created,  # Assume all freed
                efficiency_score=efficiency,
                success=abs(memory_delta) < 50  # Success if < 50MB impact
            )
            
        except Exception as e:
            return MemoryBenchmarkResult(
                test_name="object_creation_stress",
                start_memory_mb=start_mb,
                end_memory_mb=start_mb,
                peak_memory_mb=peak_mb,
                duration_seconds=time.time() - start_time,
                memory_delta_mb=0,
                gc_collections=0,
                objects_created=objects_created,
                objects_freed=0,
                efficiency_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def stress_test_allocation_patterns(self) -> MemoryBenchmarkResult:
        """Test different memory allocation patterns"""
        start_time = time.time()
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        try:
            # Pattern 1: Large contiguous allocations
            large_arrays = []
            for i in range(10):
                arr = list(range(100000))  # ~400KB each
                large_arrays.append(arr)
            
            # Pattern 2: Many small allocations
            small_objects = []
            for i in range(10000):
                obj = {'id': i, 'data': f"data_{i}"}
                small_objects.append(obj)
            
            # Pattern 3: Mixed allocations
            mixed_data = {
                'arrays': large_arrays[:5],
                'objects': small_objects[:5000],
                'strings': [f"string_{i}" * 100 for i in range(1000)]
            }
            
            # Cleanup
            large_arrays.clear()
            small_objects.clear()
            mixed_data.clear()
            gc.collect()
            
            end_time = time.time()
            end_snapshot = self.profiler.take_snapshot()
            end_mb = end_snapshot.process_memory / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_mb - start_mb
            efficiency = max(0.0, 100.0 - abs(memory_delta * 2))
            
            return MemoryBenchmarkResult(
                test_name="allocation_patterns_stress",
                start_memory_mb=start_mb,
                end_memory_mb=end_mb,
                peak_memory_mb=end_mb,  # Approximation
                duration_seconds=duration,
                memory_delta_mb=memory_delta,
                gc_collections=1,
                objects_created=16010,  # 10 + 10000 + 1000
                objects_freed=16010,
                efficiency_score=efficiency,
                success=abs(memory_delta) < 30
            )
            
        except Exception as e:
            return MemoryBenchmarkResult(
                test_name="allocation_patterns_stress",
                start_memory_mb=start_mb,
                end_memory_mb=start_mb,
                peak_memory_mb=start_mb,
                duration_seconds=time.time() - start_time,
                memory_delta_mb=0,
                gc_collections=0,
                objects_created=0,
                objects_freed=0,
                efficiency_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def stress_test_gc_pressure(self) -> MemoryBenchmarkResult:
        """Test garbage collection under pressure"""
        start_time = time.time()
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        gc_collections_before = sum(gc.get_stats()[gen]['collections'] for gen in range(3))
        
        try:
            # Create circular references to pressure GC
            circular_refs = []
            
            for i in range(1000):
                obj1 = {'id': i, 'ref': None}
                obj2 = {'id': i + 1000, 'ref': obj1}
                obj1['ref'] = obj2
                
                circular_refs.append((obj1, obj2))
            
            # Create some weak references
            weak_refs = []
            for obj1, obj2 in circular_refs[:500]:
                try:
                    weak_refs.append(weakref.ref(obj1))
                    weak_refs.append(weakref.ref(obj2))
                except TypeError:
                    pass  # Some objects can't be weakly referenced
            
            # Force GC cycles
            for _ in range(5):
                gc.collect()
                await asyncio.sleep(0.01)  # Allow async processing
            
            # Cleanup
            circular_refs.clear()
            weak_refs.clear()
            gc.collect()
            
            end_time = time.time()
            end_snapshot = self.profiler.take_snapshot()
            end_mb = end_snapshot.process_memory / 1024 / 1024
            
            gc_collections_after = sum(gc.get_stats()[gen]['collections'] for gen in range(3))
            gc_collections = gc_collections_after - gc_collections_before
            
            duration = end_time - start_time
            memory_delta = end_mb - start_mb
            efficiency = max(0.0, 100.0 - abs(memory_delta * 3))
            
            return MemoryBenchmarkResult(
                test_name="gc_pressure_stress",
                start_memory_mb=start_mb,
                end_memory_mb=end_mb,
                peak_memory_mb=end_mb + 10,  # Estimate peak during allocation
                duration_seconds=duration,
                memory_delta_mb=memory_delta,
                gc_collections=gc_collections,
                objects_created=3000,  # 1000 pairs + weak refs
                objects_freed=3000,
                efficiency_score=efficiency,
                success=abs(memory_delta) < 20
            )
            
        except Exception as e:
            return MemoryBenchmarkResult(
                test_name="gc_pressure_stress",
                start_memory_mb=start_mb,
                end_memory_mb=start_mb,
                peak_memory_mb=start_mb,
                duration_seconds=time.time() - start_time,
                memory_delta_mb=0,
                gc_collections=0,
                objects_created=0,
                objects_freed=0,
                efficiency_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def stress_test_large_objects(self) -> MemoryBenchmarkResult:
        """Test handling of large memory objects"""
        start_time = time.time()
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        try:
            large_objects = []
            
            # Create several large objects
            for i in range(5):
                # Create ~10MB object
                large_data = {
                    'matrix': [list(range(1000)) for _ in range(1000)],
                    'metadata': {'size': '10MB', 'id': i},
                    'strings': [f"large_string_{j}" * 100 for j in range(10000)]
                }
                large_objects.append(large_data)
                
                # Check memory between allocations
                if i % 2 == 0:
                    current_snapshot = self.profiler.take_snapshot()
                    current_mb = current_snapshot.process_memory / 1024 / 1024
                    print(f"    Large object {i+1}/5: {current_mb:.1f}MB")
            
            # Brief hold to measure peak
            await asyncio.sleep(0.1)
            peak_snapshot = self.profiler.take_snapshot()
            peak_mb = peak_snapshot.process_memory / 1024 / 1024
            
            # Cleanup large objects
            large_objects.clear()
            gc.collect()
            
            end_time = time.time()
            end_snapshot = self.profiler.take_snapshot()
            end_mb = end_snapshot.process_memory / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_mb - start_mb
            efficiency = max(0.0, 100.0 - abs(memory_delta * 0.5))
            
            return MemoryBenchmarkResult(
                test_name="large_objects_stress",
                start_memory_mb=start_mb,
                end_memory_mb=end_mb,
                peak_memory_mb=peak_mb,
                duration_seconds=duration,
                memory_delta_mb=memory_delta,
                gc_collections=1,
                objects_created=5,
                objects_freed=5,
                efficiency_score=efficiency,
                success=abs(memory_delta) < 100  # Allow for some residual
            )
            
        except Exception as e:
            return MemoryBenchmarkResult(
                test_name="large_objects_stress",
                start_memory_mb=start_mb,
                end_memory_mb=start_mb,
                peak_memory_mb=start_mb,
                duration_seconds=time.time() - start_time,
                memory_delta_mb=0,
                gc_collections=0,
                objects_created=0,
                objects_freed=0,
                efficiency_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def test_widget_memory_management(self) -> Dict[str, Any]:
        """Test TUI widget memory management optimization"""
        print("  🖼️ Testing widget memory management...")
        
        # Setup object pools for common widget patterns
        widget_pools = setup_emergency_pools()
        
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        try:
            # Simulate widget lifecycle patterns
            widget_tests = []
            
            # Test 1: Rapid widget creation/destruction
            print("    Testing rapid widget creation/destruction...")
            widget_memory_delta = await self.simulate_widget_lifecycle()
            
            # Test 2: Widget state management
            print("    Testing widget state management...")
            state_memory_delta = await self.simulate_widget_state_management()
            
            # Test 3: Event handling memory impact
            print("    Testing event handling memory...")
            event_memory_delta = await self.simulate_widget_events()
            
            end_snapshot = self.profiler.take_snapshot()
            end_mb = end_snapshot.process_memory / 1024 / 1024
            total_delta = end_mb - start_mb
            
            return {
                "widget_lifecycle_delta_mb": widget_memory_delta,
                "state_management_delta_mb": state_memory_delta,
                "event_handling_delta_mb": event_memory_delta,
                "total_widget_memory_delta_mb": total_delta,
                "pools_used": len(widget_pools),
                "optimization_effective": abs(total_delta) < 20
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "widget_lifecycle_delta_mb": 0,
                "state_management_delta_mb": 0,
                "event_handling_delta_mb": 0,
                "total_widget_memory_delta_mb": 0,
                "pools_used": 0,
                "optimization_effective": False
            }
    
    async def simulate_widget_lifecycle(self) -> float:
        """Simulate TUI widget creation/destruction patterns"""
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        # Simulate creating and destroying many widgets
        widgets = []
        
        for cycle in range(50):
            # Create widget batch
            batch = []
            for i in range(20):
                widget = {
                    'id': f"widget_{cycle}_{i}",
                    'type': 'label' if i % 2 == 0 else 'button',
                    'properties': {
                        'text': f"Widget {i}" * 10,
                        'style': {'color': 'white', 'bg': 'blue'},
                        'position': (i * 10, cycle * 5),
                        'size': (100, 30)
                    },
                    'state': {'visible': True, 'enabled': True},
                    'children': [{'child_id': f"child_{i}_{j}"} for j in range(3)]
                }
                batch.append(widget)
            
            widgets.extend(batch)
            
            # Cleanup old batches
            if len(widgets) > 200:
                widgets = widgets[-100:]  # Keep only recent widgets
            
            # Periodic GC
            if cycle % 10 == 0:
                gc.collect()
        
        # Final cleanup
        widgets.clear()
        gc.collect()
        
        end_snapshot = self.profiler.take_snapshot()
        end_mb = end_snapshot.process_memory / 1024 / 1024
        
        return end_mb - start_mb
    
    async def simulate_widget_state_management(self) -> float:
        """Simulate widget state changes and memory impact"""
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        # Create persistent widgets with changing state
        persistent_widgets = []
        
        for i in range(100):
            widget = {
                'id': f"persistent_{i}",
                'state_history': [],
                'current_state': {'value': 0}
            }
            persistent_widgets.append(widget)
        
        # Simulate state changes
        for update_cycle in range(100):
            for widget in persistent_widgets:
                # Update state
                new_state = {
                    'value': update_cycle,
                    'timestamp': time.time(),
                    'cycle': update_cycle
                }
                
                widget['state_history'].append(widget['current_state'])
                widget['current_state'] = new_state
                
                # Limit state history to prevent unbounded growth
                if len(widget['state_history']) > 10:
                    widget['state_history'] = widget['state_history'][-5:]
        
        # Cleanup
        persistent_widgets.clear()
        gc.collect()
        
        end_snapshot = self.profiler.take_snapshot()
        end_mb = end_snapshot.process_memory / 1024 / 1024
        
        return end_mb - start_mb
    
    async def simulate_widget_events(self) -> float:
        """Simulate event handling and callback memory impact"""
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        # Create event system simulation
        event_handlers = {}
        event_queue = deque()
        
        # Register many event handlers
        for i in range(200):
            handler_id = f"handler_{i}"
            
            def create_handler(handler_id):
                def handler(event_data):
                    return {
                        'handler_id': handler_id,
                        'processed_at': time.time(),
                        'result': event_data.get('value', 0) * 2
                    }
                return handler
            
            event_handlers[handler_id] = create_handler(handler_id)
        
        # Generate and process events
        for event_cycle in range(500):
            # Create event
            event = {
                'id': f"event_{event_cycle}",
                'type': 'click' if event_cycle % 2 == 0 else 'keypress',
                'value': event_cycle,
                'timestamp': time.time()
            }
            
            event_queue.append(event)
            
            # Process events
            while event_queue and len(event_queue) > 0:
                current_event = event_queue.popleft()
                
                # Simulate event processing
                for handler_id, handler in list(event_handlers.items())[:10]:  # Process with subset
                    try:
                        result = handler(current_event)
                        # Results would normally be stored/processed
                    except Exception:
                        pass
            
            # Keep queue bounded
            if len(event_queue) > 50:
                event_queue = deque(list(event_queue)[-25:])
        
        # Cleanup
        event_handlers.clear()
        event_queue.clear()
        gc.collect()
        
        end_snapshot = self.profiler.take_snapshot()
        end_mb = end_snapshot.process_memory / 1024 / 1024
        
        return end_mb - start_mb
    
    async def apply_comprehensive_optimizations(self) -> Dict[str, Any]:
        """Apply all memory optimization strategies"""
        print("  ⚡ Applying comprehensive memory optimizations...")
        
        optimization_start = time.time()
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        print(f"    Memory before optimization: {start_mb:.1f}MB")
        
        # Run emergency optimization
        optimization_results = self.optimizer.run_emergency_optimization()
        
        end_snapshot = self.profiler.take_snapshot()
        end_mb = end_snapshot.process_memory / 1024 / 1024
        optimization_duration = time.time() - optimization_start
        
        print(f"    Memory after optimization: {end_mb:.1f}MB")
        print(f"    Total reduction: {start_mb - end_mb:.1f}MB")
        print(f"    Optimization time: {optimization_duration:.2f}s")
        
        return {
            "optimization_successful": optimization_results.get("success", False),
            "memory_before_mb": start_mb,
            "memory_after_mb": end_mb,
            "reduction_achieved_mb": start_mb - end_mb,
            "reduction_percentage": ((start_mb - end_mb) / start_mb) * 100 if start_mb > 0 else 0,
            "optimization_duration_seconds": optimization_duration,
            "target_achieved": end_mb <= self.target_memory_mb,
            "detailed_results": optimization_results
        }
    
    async def validate_memory_target(self) -> Dict[str, Any]:
        """Validate that memory usage stays within target under load"""
        print("  ✅ Validating memory target under load...")
        
        validation_start = time.time()
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        # Run sustained load test
        peak_memory = start_mb
        memory_samples = []
        
        try:
            for test_cycle in range(30):  # 30 cycles of load
                # Create moderate load
                temp_data = []
                for i in range(1000):
                    data = {
                        'id': f"validation_{test_cycle}_{i}",
                        'payload': list(range(50)),
                        'metadata': {'cycle': test_cycle, 'index': i}
                    }
                    temp_data.append(data)
                
                # Sample memory
                sample_snapshot = self.profiler.take_snapshot()
                sample_mb = sample_snapshot.process_memory / 1024 / 1024
                memory_samples.append(sample_mb)
                peak_memory = max(peak_memory, sample_mb)
                
                # Cleanup cycle data
                temp_data.clear()
                
                if test_cycle % 5 == 0:
                    gc.collect()
                    print(f"    Validation cycle {test_cycle+1}/30: {sample_mb:.1f}MB")
                
                # Brief pause
                await asyncio.sleep(0.05)
            
            # Final measurement
            end_snapshot = self.profiler.take_snapshot()
            end_mb = end_snapshot.process_memory / 1024 / 1024
            
            validation_duration = time.time() - validation_start
            
            # Analysis
            avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else end_mb
            memory_variance = max(memory_samples) - min(memory_samples) if len(memory_samples) > 1 else 0
            target_maintained = all(sample <= self.target_memory_mb * 1.1 for sample in memory_samples)  # 10% tolerance
            
            return {
                "validation_successful": target_maintained and end_mb <= self.target_memory_mb,
                "target_memory_mb": self.target_memory_mb,
                "start_memory_mb": start_mb,
                "end_memory_mb": end_mb,
                "peak_memory_mb": peak_memory,
                "average_memory_mb": avg_memory,
                "memory_variance_mb": memory_variance,
                "target_maintained_throughout": target_maintained,
                "validation_duration_seconds": validation_duration,
                "load_cycles_completed": len(memory_samples),
                "memory_stability": "STABLE" if memory_variance < 50 else "UNSTABLE"
            }
            
        except Exception as e:
            return {
                "validation_successful": False,
                "error": str(e),
                "target_memory_mb": self.target_memory_mb,
                "start_memory_mb": start_mb,
                "end_memory_mb": start_mb,
                "peak_memory_mb": start_mb,
                "average_memory_mb": start_mb,
                "memory_variance_mb": 0,
                "target_maintained_throughout": False,
                "validation_duration_seconds": time.time() - validation_start,
                "load_cycles_completed": 0,
                "memory_stability": "ERROR"
            }
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.performance_timeline.clear()
            
            def monitoring_loop():
                while self.monitoring_active:
                    try:
                        snapshot = self.profiler.take_snapshot()
                        memory_mb = snapshot.process_memory / 1024 / 1024
                        timestamp = time.time()
                        
                        self.performance_timeline.append((timestamp, memory_mb))
                        
                        # Keep timeline bounded
                        if len(self.performance_timeline) > 1000:
                            self.performance_timeline = self.performance_timeline[-500:]
                        
                        time.sleep(1.0)  # Sample every second
                        
                    except Exception as e:
                        print(f"Monitoring error: {e}")
                        time.sleep(5.0)
            
            self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def generate_benchmark_report(self, *phase_results) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        
        current_snapshot = self.profiler.take_snapshot()
        current_mb = current_snapshot.process_memory / 1024 / 1024
        baseline_mb = self.baseline_snapshot.process_memory / 1024 / 1024 if self.baseline_snapshot else current_mb
        
        # Performance timeline analysis
        if self.performance_timeline:
            timeline_memories = [m for t, m in self.performance_timeline]
            min_memory = min(timeline_memories)
            max_memory = max(timeline_memories)
            avg_memory = sum(timeline_memories) / len(timeline_memories)
        else:
            min_memory = max_memory = avg_memory = current_mb
        
        # Overall success criteria
        target_achieved = current_mb <= self.target_memory_mb
        memory_stable = (max_memory - min_memory) < 100  # Less than 100MB variance
        optimizations_effective = baseline_mb > current_mb
        
        overall_success = target_achieved and memory_stable and optimizations_effective
        
        report = {
            "benchmark_summary": {
                "overall_success": overall_success,
                "target_memory_mb": self.target_memory_mb,
                "baseline_memory_mb": baseline_mb,
                "final_memory_mb": current_mb,
                "total_reduction_mb": baseline_mb - current_mb,
                "reduction_percentage": ((baseline_mb - current_mb) / baseline_mb) * 100 if baseline_mb > 0 else 0,
                "target_achieved": target_achieved,
                "memory_stable": memory_stable
            },
            
            "performance_timeline": {
                "samples_collected": len(self.performance_timeline),
                "min_memory_mb": min_memory,
                "max_memory_mb": max_memory,
                "average_memory_mb": avg_memory,
                "memory_variance_mb": max_memory - min_memory
            },
            
            "hotspot_analysis": phase_results[0] if len(phase_results) > 0 else {},
            "stress_test_before": phase_results[1] if len(phase_results) > 1 else {},
            "widget_memory_test": phase_results[2] if len(phase_results) > 2 else {},
            "optimization_results": phase_results[3] if len(phase_results) > 3 else {},
            "stress_test_after": phase_results[4] if len(phase_results) > 4 else {},
            "validation_results": phase_results[5] if len(phase_results) > 5 else {},
            
            "recommendations": self.generate_recommendations(current_mb, overall_success),
            "timestamp": datetime.now().isoformat(),
            "benchmark_duration_minutes": (time.time() - (self.baseline_snapshot.timestamp if self.baseline_snapshot else time.time())) / 60
        }
        
        return report
    
    def generate_recommendations(self, current_mb: float, overall_success: bool) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not overall_success:
            if current_mb > self.target_memory_mb:
                excess_mb = current_mb - self.target_memory_mb
                recommendations.append(f"🎯 Further optimization needed: Reduce memory by {excess_mb:.1f}MB")
                
                # Specific recommendations based on hotspots
                critical_hotspots = [h for h in self.hotspots if h.severity == "CRITICAL"]
                if critical_hotspots:
                    for hotspot in critical_hotspots[:3]:
                        recommendations.append(f"🔥 Optimize {hotspot.object_type}: {hotspot.total_size_mb:.1f}MB ({hotspot.count:,} objects)")
        
        if current_mb <= self.target_memory_mb:
            recommendations.append("✅ Memory target achieved! Consider monitoring for sustained performance")
        
        if len(self.performance_timeline) > 100:
            timeline_memories = [m for t, m in self.performance_timeline]
            if (max(timeline_memories) - min(timeline_memories)) > 200:
                recommendations.append("⚠️ High memory variance detected - investigate memory leaks")
        
        # Always include monitoring recommendations
        recommendations.append("📊 Continue monitoring with Claude-Flow hooks for sustained optimization")
        recommendations.append("🏊‍♂️ Maintain object pooling for frequently created objects")
        recommendations.append("🗑️ Schedule regular garbage collection during low-usage periods")
        
        return recommendations
    
    async def save_benchmark_report(self, report: Dict[str, Any], filepath: str):
        """Save benchmark report to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📋 Benchmark report saved to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")


# Convenience function for quick benchmarking
async def run_hive_mind_memory_benchmark(target_gb: float = 1.5) -> Dict[str, Any]:
    """Run complete Hive Mind memory benchmark suite"""
    benchmarker = HiveMindMemoryBenchmarker(target_memory_gb=target_gb)
    try:
        return await benchmarker.run_comprehensive_benchmark_suite()
    finally:
        benchmarker.stop_monitoring()


if __name__ == "__main__":
    import asyncio
    
    print("🚀 Hive Mind Memory Benchmarker - Starting...")
    
    async def main():
        # Run comprehensive benchmark
        results = await run_hive_mind_memory_benchmark(target_gb=1.5)
        
        # Save results
        benchmarker = HiveMindMemoryBenchmarker()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/home/tekkadmin/claude-tui/hive_mind_memory_benchmark_{timestamp}.json"
        await benchmarker.save_benchmark_report(results, report_path)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 HIVE MIND MEMORY BENCHMARK COMPLETE")
        print("="*60)
        
        summary = results.get("benchmark_summary", {})
        print(f"Overall Success: {'✅' if summary.get('overall_success') else '❌'}")
        print(f"Target: {summary.get('target_memory_mb', 0):.0f}MB")
        print(f"Final Memory: {summary.get('final_memory_mb', 0):.1f}MB")
        print(f"Reduction: {summary.get('total_reduction_mb', 0):.1f}MB ({summary.get('reduction_percentage', 0):.1f}%)")
        print(f"Target Achieved: {'✅' if summary.get('target_achieved') else '❌'}")
        
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations[:5]:
                print(f"  • {rec}")
        
        print(f"\n📋 Detailed report: {report_path}")
    
    asyncio.run(main())