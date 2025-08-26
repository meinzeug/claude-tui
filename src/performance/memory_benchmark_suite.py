#!/usr/bin/env python3
"""
Memory Benchmark Suite - Comprehensive Memory Performance Testing
Tests memory optimization strategies and measures their effectiveness
"""

import gc
import sys
import os
import time
import json
import psutil
import tracemalloc
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import our optimization modules
try:
    from .memory_optimizer import UltimateMemoryOptimizer, ultimate_optimize
    from .emergency_memory_recovery import EmergencyMemoryRecovery, emergency_memory_recovery
    from .widget_memory_manager import get_widget_manager, emergency_widget_cleanup
    from .advanced_memory_profiler import AdvancedMemoryProfiler
except ImportError:
    # Fallback for direct execution
    try:
        from memory_optimizer import UltimateMemoryOptimizer, ultimate_optimize
        from emergency_memory_recovery import EmergencyMemoryRecovery, emergency_memory_recovery
        from widget_memory_manager import get_widget_manager, emergency_widget_cleanup
        from advanced_memory_profiler import AdvancedMemoryProfiler
    except ImportError:
        print("⚠️ Some optimization modules not available - using fallbacks")
        UltimateMemoryOptimizer = None
        ultimate_optimize = None
        EmergencyMemoryRecovery = None
        emergency_memory_recovery = None
        get_widget_manager = None
        emergency_widget_cleanup = None
        AdvancedMemoryProfiler = None


@dataclass
class BenchmarkResult:
    """Result of a memory benchmark test"""
    test_name: str
    initial_memory_mb: float
    final_memory_mb: float
    peak_memory_mb: float
    memory_reduction_mb: float
    reduction_percentage: float
    execution_time_seconds: float
    objects_before: int
    objects_after: int
    objects_freed: int
    gc_collections: int
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class BenchmarkSuite:
    """Complete benchmark suite results"""
    timestamp: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    total_time_seconds: float
    initial_system_memory_mb: float
    final_system_memory_mb: float
    system_memory_saved_mb: float
    results: List[BenchmarkResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MemoryBenchmarkSuite:
    """
    Comprehensive memory benchmark testing suite
    
    Tests various memory optimization strategies and measures their effectiveness
    """
    
    def __init__(self, target_memory_mb: int = 100):
        self.target_memory_mb = target_memory_mb
        self.test_results = []
        self.profiler = None
        
        # Initialize tracemalloc for detailed tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            
    def run_complete_benchmark(self) -> BenchmarkSuite:
        """Run complete memory optimization benchmark suite"""
        print("🚀 MEMORY OPTIMIZATION BENCHMARK SUITE")
        print("=" * 50)
        
        suite_start_time = time.time()
        initial_system_memory = self._get_memory_usage_mb()
        
        # Define benchmark tests
        benchmark_tests = [
            ("Baseline Memory Measurement", self._benchmark_baseline),
            ("Ultimate Memory Optimizer", self._benchmark_ultimate_optimizer),
            ("Emergency Memory Recovery", self._benchmark_emergency_recovery),
            ("Widget Memory Cleanup", self._benchmark_widget_cleanup),
            ("Garbage Collection Optimization", self._benchmark_gc_optimization),
            ("Cache Clearing Strategies", self._benchmark_cache_clearing),
            ("Module Unloading", self._benchmark_module_unloading),
            ("Object Pool Optimization", self._benchmark_object_pooling),
            ("Lazy Loading Implementation", self._benchmark_lazy_loading),
            ("Memory Profiler Analysis", self._benchmark_memory_profiling)
        ]
        
        successful_tests = 0
        failed_tests = 0
        
        for test_name, test_func in benchmark_tests:
            print(f"\n📊 Running: {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                result.test_name = test_name
                self.test_results.append(result)
                
                if result.success:
                    successful_tests += 1
                    print(f"✅ {test_name}: {result.memory_reduction_mb:.1f}MB saved ({result.reduction_percentage:.1f}%)")
                else:
                    failed_tests += 1
                    print(f"❌ {test_name}: Failed - {result.error_message}")
                    
            except Exception as e:
                failed_tests += 1
                error_result = BenchmarkResult(
                    test_name=test_name,
                    initial_memory_mb=0,
                    final_memory_mb=0,
                    peak_memory_mb=0,
                    memory_reduction_mb=0,
                    reduction_percentage=0,
                    execution_time_seconds=0,
                    objects_before=0,
                    objects_after=0,
                    objects_freed=0,
                    gc_collections=0,
                    success=False,
                    error_message=str(e)
                )
                self.test_results.append(error_result)
                print(f"💥 {test_name}: Exception - {str(e)}")
        
        suite_end_time = time.time()
        final_system_memory = self._get_memory_usage_mb()
        
        # Create benchmark suite result
        suite_result = BenchmarkSuite(
            timestamp=datetime.now().isoformat(),
            total_tests=len(benchmark_tests),
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            total_time_seconds=suite_end_time - suite_start_time,
            initial_system_memory_mb=initial_system_memory,
            final_system_memory_mb=final_system_memory,
            system_memory_saved_mb=initial_system_memory - final_system_memory,
            results=self.test_results,
            recommendations=self._generate_recommendations()
        )
        
        self._print_final_report(suite_result)
        return suite_result
        
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
            
    def _get_object_count(self) -> int:
        """Get current object count"""
        return len(gc.get_objects())
        
    def _get_gc_stats(self) -> int:
        """Get total GC collections"""
        return sum(stat['collections'] for stat in gc.get_stats())
        
    def _benchmark_baseline(self) -> BenchmarkResult:
        """Baseline memory measurement"""
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        
        # Just wait a moment to establish baseline
        time.sleep(0.1)
        
        final_memory = self._get_memory_usage_mb()
        final_objects = self._get_object_count()
        final_gc = self._get_gc_stats()
        end_time = time.time()
        
        return BenchmarkResult(
            test_name="Baseline",
            initial_memory_mb=initial_memory,
            final_memory_mb=final_memory,
            peak_memory_mb=max(initial_memory, final_memory),
            memory_reduction_mb=initial_memory - final_memory,
            reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
            execution_time_seconds=end_time - start_time,
            objects_before=initial_objects,
            objects_after=final_objects,
            objects_freed=initial_objects - final_objects,
            gc_collections=final_gc - initial_gc,
            success=True,
            additional_metrics={
                "baseline_memory": initial_memory,
                "baseline_objects": initial_objects
            }
        )
        
    def _benchmark_ultimate_optimizer(self) -> BenchmarkResult:
        """Benchmark the Ultimate Memory Optimizer"""
        if not UltimateMemoryOptimizer:
            return self._create_error_result("Ultimate Memory Optimizer not available")
            
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        peak_memory = initial_memory
        
        try:
            # Create memory load to test optimization
            memory_hogs = []
            for i in range(100):
                # Create objects that consume memory
                memory_hog = {
                    'data': [j for j in range(1000)],
                    'text': 'x' * 10000,
                    'nested': {'level1': {'level2': [k for k in range(500)]}}
                }
                memory_hogs.append(memory_hog)
                
            # Track peak memory
            peak_memory = max(peak_memory, self._get_memory_usage_mb())
            
            # Run ultimate optimizer
            optimizer = UltimateMemoryOptimizer(target_mb=self.target_memory_mb)
            optimization_result = optimizer.run_emergency_optimization()
            
            # Clear our test objects
            memory_hogs.clear()
            del memory_hogs
            
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Ultimate Memory Optimizer",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=optimization_result.get('success', False),
                additional_metrics={
                    'optimization_result': optimization_result,
                    'strategies_applied': optimization_result.get('strategies_applied', 0),
                    'target_achieved': optimization_result.get('target_achieved', False)
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Ultimate optimizer failed: {str(e)}")
            
    def _benchmark_emergency_recovery(self) -> BenchmarkResult:
        """Benchmark emergency memory recovery"""
        if not EmergencyMemoryRecovery:
            return self._create_error_result("Emergency Memory Recovery not available")
            
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        peak_memory = initial_memory
        
        try:
            # Create critical memory situation
            critical_objects = []
            for i in range(200):
                obj = {
                    'id': i,
                    'payload': bytearray(50000),  # 50KB each
                    'references': [j for j in range(100)],
                    'circular_ref': None
                }
                obj['circular_ref'] = obj  # Create circular reference
                critical_objects.append(obj)
                
            peak_memory = max(peak_memory, self._get_memory_usage_mb())
            
            # Run emergency recovery
            recovery_system = EmergencyMemoryRecovery()
            recovery_result = recovery_system.execute_emergency_recovery()
            
            # Clear our test objects
            critical_objects.clear()
            del critical_objects
            
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Emergency Memory Recovery",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=recovery_result.success,
                additional_metrics={
                    'recovery_result': {
                        'memory_recovered_mb': recovery_result.memory_recovered_mb,
                        'recovery_percentage': recovery_result.recovery_percentage,
                        'operations_performed': len(recovery_result.operations_performed)
                    }
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Emergency recovery failed: {str(e)}")
            
    def _benchmark_widget_cleanup(self) -> BenchmarkResult:
        """Benchmark widget memory cleanup"""
        if not get_widget_manager:
            return self._create_error_result("Widget Memory Manager not available")
            
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        
        try:
            # Create mock widgets
            class MockWidget:
                def __init__(self, widget_id, size=10240):  # 10KB default
                    self.id = widget_id
                    self.data = bytearray(size)
                    self.children = []
                    
                def destroy(self):
                    self.data = None
                    self.children.clear()
            
            # Create widgets
            widget_manager = get_widget_manager()
            widgets_created = 500
            
            for i in range(widgets_created):
                widget = MockWidget(f"benchmark_widget_{i}", size=5120 * (1 + i % 10))
                widget_manager.register_widget(f"benchmark_widget_{i}", widget, f"category_{i % 10}")
            
            peak_memory = self._get_memory_usage_mb()
            
            # Run widget cleanup
            cleanup_stats = emergency_widget_cleanup()
            
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Widget Memory Cleanup",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=True,
                additional_metrics={
                    'widgets_created': widgets_created,
                    'cleanup_stats': cleanup_stats
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Widget cleanup failed: {str(e)}")
            
    def _benchmark_gc_optimization(self) -> BenchmarkResult:
        """Benchmark garbage collection optimization"""
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        
        try:
            # Create garbage
            garbage = []
            for i in range(1000):
                obj = [j for j in range(100)]
                obj.append(obj)  # Circular reference
                garbage.append(obj)
                
            # Create unreferenced objects
            for i in range(500):
                temp = {'data': [k for k in range(50)]}
                
            peak_memory = self._get_memory_usage_mb()
            
            # Optimize GC
            original_thresholds = gc.get_threshold()
            gc.set_threshold(50, 3, 3)  # Aggressive
            
            collected_total = 0
            for generation in range(3):
                for _ in range(5):
                    collected = gc.collect(generation)
                    collected_total += collected
                    if collected == 0:
                        break
                        
            # Additional comprehensive collection
            for _ in range(10):
                collected_total += gc.collect()
                
            # Restore thresholds
            gc.set_threshold(*original_thresholds)
            
            # Clear our garbage
            garbage.clear()
            del garbage
            
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="GC Optimization",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=True,
                additional_metrics={
                    'objects_collected': collected_total,
                    'original_thresholds': original_thresholds
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"GC optimization failed: {str(e)}")
            
    def _benchmark_cache_clearing(self) -> BenchmarkResult:
        """Benchmark cache clearing strategies"""
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        
        try:
            # Create cached functions
            import functools
            
            @functools.lru_cache(maxsize=1000)
            def cached_function(x):
                return [i for i in range(x)]
                
            # Populate caches
            for i in range(500):
                cached_function(100 + i)
                
            peak_memory = self._get_memory_usage_mb()
            
            # Clear caches
            caches_cleared = 0
            
            # Clear function caches
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear') and callable(getattr(obj, 'cache_clear')):
                    try:
                        obj.cache_clear()
                        caches_cleared += 1
                    except:
                        pass
                        
            # System caches
            try:
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                    caches_cleared += 1
            except:
                pass
                
            try:
                import importlib
                importlib.invalidate_caches()
                caches_cleared += 1
            except:
                pass
                
            try:
                import linecache
                linecache.clearcache()
                caches_cleared += 1
            except:
                pass
                
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Cache Clearing",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=True,
                additional_metrics={
                    'caches_cleared': caches_cleared
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Cache clearing failed: {str(e)}")
            
    def _benchmark_module_unloading(self) -> BenchmarkResult:
        """Benchmark module unloading"""
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        modules_before = len(sys.modules)
        
        try:
            # Import some modules that can be safely unloaded
            test_modules = ['json', 'uuid', 'random', 'string', 'decimal']
            
            # Force import
            for module_name in test_modules:
                __import__(module_name)
                
            peak_memory = self._get_memory_usage_mb()
            
            # Unload modules with low reference counts
            modules_unloaded = 0
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('test_') or module_name in test_modules:
                    try:
                        module = sys.modules.get(module_name)
                        if module and sys.getrefcount(module) <= 3:
                            del sys.modules[module_name]
                            modules_unloaded += 1
                    except:
                        pass
                        
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            modules_after = len(sys.modules)
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Module Unloading",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=True,
                additional_metrics={
                    'modules_before': modules_before,
                    'modules_after': modules_after,
                    'modules_unloaded': modules_unloaded
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Module unloading failed: {str(e)}")
            
    def _benchmark_object_pooling(self) -> BenchmarkResult:
        """Benchmark object pooling optimization"""
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        
        try:
            # Create object pools
            from collections import deque
            
            pools = {
                'dict': deque(maxlen=100),
                'list': deque(maxlen=100), 
                'set': deque(maxlen=100),
                'tuple': deque(maxlen=100)
            }
            
            # Create objects to pool
            test_objects = []
            for i in range(1000):
                test_dict = {'id': i, 'data': [j for j in range(10)]}
                test_list = [k for k in range(20)]
                test_set = {l for l in range(5)}
                test_tuple = tuple(range(15))
                
                test_objects.extend([test_dict, test_list, test_set, test_tuple])
                
            peak_memory = self._get_memory_usage_mb()
            
            # Pool objects
            pooled_objects = 0
            for obj in test_objects:
                obj_type = type(obj).__name__
                if obj_type in pools and len(pools[obj_type]) < 100:
                    try:
                        if sys.getsizeof(obj) < 1024:  # Only pool small objects
                            pools[obj_type].append(obj)
                            pooled_objects += 1
                    except:
                        pass
                        
            # Clear original objects
            test_objects.clear()
            del test_objects
            
            # Force GC
            for _ in range(3):
                gc.collect()
                
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Object Pooling",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=True,
                additional_metrics={
                    'pools_created': len(pools),
                    'objects_pooled': pooled_objects
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Object pooling failed: {str(e)}")
            
    def _benchmark_lazy_loading(self) -> BenchmarkResult:
        """Benchmark lazy loading implementation"""
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        
        try:
            # Create lazy loading proxies
            class LazyProxy:
                def __init__(self, name, data_factory):
                    self.name = name
                    self._data_factory = data_factory
                    self._data = None
                    self._loaded = False
                    
                def __getattr__(self, name):
                    if not self._loaded:
                        self._data = self._data_factory()
                        self._loaded = True
                    return getattr(self._data, name)
                    
            # Create data factories
            def create_large_data():
                return {'data': [i for i in range(10000)], 'text': 'x' * 50000}
                
            # Create lazy proxies instead of actual data
            lazy_objects = []
            for i in range(100):
                proxy = LazyProxy(f"lazy_{i}", create_large_data)
                lazy_objects.append(proxy)
                
            # Only access a few to trigger loading
            for i in range(10):
                _ = lazy_objects[i].__dict__  # Trigger loading
                
            peak_memory = self._get_memory_usage_mb()
            
            # Clear proxies
            lazy_objects.clear()
            del lazy_objects
            
            # Force GC
            for _ in range(3):
                gc.collect()
                
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Lazy Loading",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=True,
                additional_metrics={
                    'lazy_proxies_created': 100,
                    'objects_actually_loaded': 10,
                    'memory_savings_estimate': '90% of potential objects not loaded'
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Lazy loading failed: {str(e)}")
            
    def _benchmark_memory_profiling(self) -> BenchmarkResult:
        """Benchmark memory profiling analysis"""
        if not AdvancedMemoryProfiler:
            return self._create_error_result("Advanced Memory Profiler not available")
            
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        initial_objects = self._get_object_count()
        initial_gc = self._get_gc_stats()
        
        try:
            # Create profiler
            profiler = AdvancedMemoryProfiler(target_mb=self.target_memory_mb)
            
            # Create some memory activity to profile
            memory_activity = []
            for i in range(200):
                activity = {
                    'id': i,
                    'data': [j for j in range(100)],
                    'nested': {'level': i, 'content': 'x' * 1000}
                }
                memory_activity.append(activity)
                
            # Take profiling snapshot
            snapshot = profiler.take_snapshot()
            
            peak_memory = self._get_memory_usage_mb()
            
            # Generate optimization report
            report = profiler.generate_optimization_report()
            
            # Clear activity
            memory_activity.clear()
            del memory_activity
            
            final_memory = self._get_memory_usage_mb()
            final_objects = self._get_object_count()
            final_gc = self._get_gc_stats()
            end_time = time.time()
            
            return BenchmarkResult(
                test_name="Memory Profiling",
                initial_memory_mb=initial_memory,
                final_memory_mb=final_memory,
                peak_memory_mb=peak_memory,
                memory_reduction_mb=initial_memory - final_memory,
                reduction_percentage=((initial_memory - final_memory) / initial_memory * 100) if initial_memory > 0 else 0,
                execution_time_seconds=end_time - start_time,
                objects_before=initial_objects,
                objects_after=final_objects,
                objects_freed=initial_objects - final_objects,
                gc_collections=final_gc - initial_gc,
                success=True,
                additional_metrics={
                    'profiling_report': {
                        'current_memory_mb': report.get('current_memory_mb', 0),
                        'reduction_needed_mb': report.get('reduction_needed_mb', 0),
                        'memory_efficiency': report.get('memory_efficiency', 0),
                        'potential_leaks': report.get('leak_analysis', {}).get('potential_leaks', 0)
                    },
                    'snapshot_objects': snapshot.gc_objects,
                    'largest_objects_count': len(snapshot.largest_objects)
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Memory profiling failed: {str(e)}")
            
    def _create_error_result(self, error_message: str) -> BenchmarkResult:
        """Create error result"""
        return BenchmarkResult(
            test_name="Error",
            initial_memory_mb=0,
            final_memory_mb=0,
            peak_memory_mb=0,
            memory_reduction_mb=0,
            reduction_percentage=0,
            execution_time_seconds=0,
            objects_before=0,
            objects_after=0,
            objects_freed=0,
            gc_collections=0,
            success=False,
            error_message=error_message
        )
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on benchmark results"""
        recommendations = []
        
        if not self.test_results:
            return ["No benchmark results available"]
            
        # Analyze results
        successful_results = [r for r in self.test_results if r.success]
        if not successful_results:
            return ["All benchmark tests failed - check system configuration"]
            
        # Find most effective strategies
        best_reduction = max(successful_results, key=lambda x: x.memory_reduction_mb)
        best_percentage = max(successful_results, key=lambda x: x.reduction_percentage)
        fastest = min(successful_results, key=lambda x: x.execution_time_seconds)
        
        recommendations.append(f"🏆 Best absolute reduction: {best_reduction.test_name} ({best_reduction.memory_reduction_mb:.1f}MB)")
        recommendations.append(f"🎯 Best percentage reduction: {best_percentage.test_name} ({best_percentage.reduction_percentage:.1f}%)")
        recommendations.append(f"⚡ Fastest strategy: {fastest.test_name} ({fastest.execution_time_seconds:.2f}s)")
        
        # Calculate total potential savings
        total_reduction = sum(r.memory_reduction_mb for r in successful_results)
        recommendations.append(f"💰 Total potential savings: {total_reduction:.1f}MB from all strategies combined")
        
        # Specific recommendations based on performance
        high_impact_strategies = [r for r in successful_results if r.memory_reduction_mb > 5]
        if high_impact_strategies:
            strategy_names = [r.test_name for r in high_impact_strategies]
            recommendations.append(f"🔥 High-impact strategies to prioritize: {', '.join(strategy_names)}")
        
        return recommendations
        
    def _print_final_report(self, suite_result: BenchmarkSuite):
        """Print final benchmark report"""
        print("\n" + "=" * 60)
        print("🏁 MEMORY OPTIMIZATION BENCHMARK COMPLETE")
        print("=" * 60)
        
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {suite_result.total_tests}")
        print(f"   Successful: {suite_result.successful_tests} ✅")
        print(f"   Failed: {suite_result.failed_tests} ❌")
        print(f"   Success Rate: {(suite_result.successful_tests / suite_result.total_tests * 100):.1f}%")
        print(f"   Total Time: {suite_result.total_time_seconds:.2f}s")
        
        print(f"\n💾 Memory Impact:")
        print(f"   Initial Memory: {suite_result.initial_system_memory_mb:.1f}MB")
        print(f"   Final Memory: {suite_result.final_system_memory_mb:.1f}MB")
        print(f"   Total Saved: {suite_result.system_memory_saved_mb:.1f}MB")
        print(f"   Reduction: {(suite_result.system_memory_saved_mb / suite_result.initial_system_memory_mb * 100):.1f}%")
        
        print(f"\n🎯 Target Analysis:")
        target_achieved = suite_result.final_system_memory_mb <= self.target_memory_mb
        print(f"   Target: {self.target_memory_mb}MB")
        print(f"   Achieved: {'✅ YES' if target_achieved else '❌ NO'}")
        if not target_achieved:
            remaining = suite_result.final_system_memory_mb - self.target_memory_mb
            print(f"   Remaining: {remaining:.1f}MB to target")
        
        print(f"\n📈 Top Performing Strategies:")
        successful_results = [r for r in suite_result.results if r.success]
        if successful_results:
            top_strategies = sorted(successful_results, key=lambda x: x.memory_reduction_mb, reverse=True)[:5]
            for i, result in enumerate(top_strategies, 1):
                print(f"   {i}. {result.test_name}: {result.memory_reduction_mb:.1f}MB ({result.reduction_percentage:.1f}%)")
        
        print(f"\n💡 Recommendations:")
        for rec in suite_result.recommendations:
            print(f"   {rec}")
            
    def export_results(self, filename: Optional[str] = None) -> str:
        """Export benchmark results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/tmp/memory_benchmark_{timestamp}.json"
            
        # Create comprehensive results
        export_data = {
            'benchmark_suite': {
                'timestamp': datetime.now().isoformat(),
                'target_memory_mb': self.target_memory_mb,
                'total_tests': len(self.test_results),
                'successful_tests': len([r for r in self.test_results if r.success]),
                'failed_tests': len([r for r in self.test_results if not r.success])
            },
            'results': [
                {
                    'test_name': result.test_name,
                    'success': result.success,
                    'initial_memory_mb': result.initial_memory_mb,
                    'final_memory_mb': result.final_memory_mb,
                    'peak_memory_mb': result.peak_memory_mb,
                    'memory_reduction_mb': result.memory_reduction_mb,
                    'reduction_percentage': result.reduction_percentage,
                    'execution_time_seconds': result.execution_time_seconds,
                    'objects_freed': result.objects_freed,
                    'gc_collections': result.gc_collections,
                    'error_message': result.error_message,
                    'additional_metrics': result.additional_metrics
                }
                for result in self.test_results
            ],
            'system_info': {
                'python_version': sys.version,
                'platform': os.name,
                'pid': os.getpid(),
                'initial_memory_mb': self.test_results[0].initial_memory_mb if self.test_results else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        print(f"📄 Benchmark results exported to: {filename}")
        return filename


if __name__ == "__main__":
    print("🚀 MEMORY OPTIMIZATION BENCHMARK SUITE")
    print("Testing memory reduction from target 1.7GB → <100MB\n")
    
    # Create and run benchmark suite
    benchmark = MemoryBenchmarkSuite(target_memory_mb=100)
    results = benchmark.run_complete_benchmark()
    
    # Export results
    benchmark.export_results()
    
    # Final assessment
    print(f"\n🎖️  FINAL ASSESSMENT:")
    if results.system_memory_saved_mb >= 100:
        print("🏆 EXCELLENT: Achieved significant memory reduction!")
    elif results.system_memory_saved_mb >= 50:
        print("🥈 GOOD: Substantial memory savings achieved")
    elif results.system_memory_saved_mb >= 10:
        print("🥉 MODERATE: Some memory optimization successful")
    else:
        print("📈 BASELINE: Limited memory reduction - more optimization needed")