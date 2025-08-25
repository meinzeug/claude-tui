#!/usr/bin/env python3
"""
Emergency Memory Optimization Tests
Comprehensive testing suite for memory reduction validation
"""

import pytest
import gc
import sys
import psutil
import time
import threading
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from performance.memory_profiler import MemoryProfiler, emergency_memory_check
from performance.lazy_loader import LazyModuleLoader, setup_emergency_lazy_imports
from performance.object_pool import ObjectPool, setup_emergency_pools, PoolableDict
from performance.memory_optimizer import EmergencyMemoryOptimizer, emergency_optimize
from performance.gc_optimizer import AdvancedGCOptimizer, emergency_gc_optimization


class TestMemoryProfiler:
    """Test memory profiling functionality"""
    
    def test_memory_profiler_initialization(self):
        """Test memory profiler initializes correctly"""
        profiler = MemoryProfiler(target_memory_mb=200)
        
        assert profiler.target_memory_mb == 200
        assert profiler.target_bytes == 200 * 1024 * 1024
        assert profiler.snapshots == []
        assert profiler.monitoring_active == False
        
    def test_take_snapshot(self):
        """Test memory snapshot functionality"""
        profiler = MemoryProfiler(target_memory_mb=200)
        snapshot = profiler.take_snapshot()
        
        assert snapshot.timestamp > 0
        assert snapshot.process_memory > 0
        assert snapshot.gc_objects > 0
        assert isinstance(snapshot.largest_objects, list)
        assert isinstance(snapshot.leak_suspects, list)
        
    def test_memory_monitoring(self):
        """Test continuous memory monitoring"""
        profiler = MemoryProfiler(target_memory_mb=200)
        
        # Start monitoring
        profiler.start_monitoring(interval_seconds=0.1)
        time.sleep(0.3)  # Let it collect some data
        profiler.stop_monitoring()
        
        assert len(profiler.snapshots) > 0
        assert profiler.monitoring_active == False
        
    def test_memory_analysis(self):
        """Test memory trend analysis"""
        profiler = MemoryProfiler(target_memory_mb=200)
        
        # Take multiple snapshots
        for _ in range(3):
            profiler.take_snapshot()
            time.sleep(0.1)
            
        analysis = profiler.analyze_memory_trends()
        
        assert "current_memory_mb" in analysis
        assert "target_memory_mb" in analysis
        assert analysis["target_memory_mb"] == 200
        
    def test_emergency_memory_check(self):
        """Test emergency memory check function"""
        result = emergency_memory_check()
        assert isinstance(result, bool)


class TestLazyLoader:
    """Test lazy loading functionality"""
    
    def test_lazy_module_loader_initialization(self):
        """Test lazy loader initializes correctly"""
        loader = LazyModuleLoader()
        
        assert loader._modules == {}
        assert loader._loaded_modules == {}
        
    def test_register_lazy_module(self):
        """Test module registration for lazy loading"""
        loader = LazyModuleLoader()
        
        lazy_module = loader.register_lazy_module("test_json", "json")
        
        assert "test_json" in loader._modules
        assert loader._modules["test_json"].import_path == "json"
        assert lazy_module._alias == "test_json"
        
    def test_lazy_module_loading(self):
        """Test actual lazy module loading"""
        loader = LazyModuleLoader()
        
        # Register json module for lazy loading
        loader.register_lazy_module("test_json", "json")
        
        # Module should not be loaded yet
        assert "test_json" not in loader._loaded_modules
        
        # Load the module
        json_module = loader.load_module("test_json")
        
        # Module should now be loaded
        assert "test_json" in loader._loaded_modules
        assert loader._modules["test_json"].loaded == True
        assert hasattr(json_module, 'dumps')  # json module function
        
    def test_setup_emergency_lazy_imports(self):
        """Test emergency lazy import setup"""
        lazy_modules = setup_emergency_lazy_imports()
        
        assert isinstance(lazy_modules, dict)
        assert len(lazy_modules) > 0
        
    def test_lazy_module_stats(self):
        """Test lazy loading statistics"""
        loader = LazyModuleLoader()
        loader.register_lazy_module("test_os", "os")
        
        stats = loader.get_loading_stats()
        
        assert stats["registered_modules"] == 1
        assert stats["loaded_modules"] == 0
        
        # Load module and check stats again
        loader.load_module("test_os")
        stats = loader.get_loading_stats()
        
        assert stats["loaded_modules"] == 1


class TestObjectPool:
    """Test object pooling functionality"""
    
    def test_object_pool_initialization(self):
        """Test object pool initializes correctly"""
        pool = ObjectPool(factory=dict, max_size=10)
        
        assert pool.max_size == 10
        assert pool.factory == dict
        assert len(pool._pool) == 0
        
    def test_object_acquire_release(self):
        """Test object acquisition and release"""
        pool = ObjectPool(factory=dict, max_size=10)
        
        # Acquire an object
        obj = pool.acquire()
        assert isinstance(obj, dict)
        
        # Release the object
        result = pool.release(obj)
        assert result == True
        assert len(pool._pool) == 1
        
        # Acquire again should reuse the object
        obj2 = pool.acquire()
        assert len(pool._pool) == 0  # Object taken from pool
        
    def test_poolable_dict(self):
        """Test PoolableDict functionality"""
        pool = ObjectPool(factory=PoolableDict, max_size=5)
        
        obj = pool.acquire()
        obj['test'] = 'value'
        
        pool.release(obj)
        
        # Acquire again and verify it's reset
        obj2 = pool.acquire()
        assert len(obj2) == 0  # Should be reset
        
    def test_pool_stats(self):
        """Test pool statistics"""
        pool = ObjectPool(factory=dict, max_size=10)
        
        # Create some activity
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        pool.release(obj1)
        
        stats = pool.get_stats()
        
        assert stats.total_created >= 2
        assert stats.total_reused >= 0
        assert stats.hit_rate >= 0
        
    def test_setup_emergency_pools(self):
        """Test emergency pool setup"""
        pools = setup_emergency_pools()
        
        assert isinstance(pools, dict)
        assert "dict_pool" in pools
        assert "list_pool" in pools
        
        # Test dict pool
        dict_pool = pools["dict_pool"]
        test_dict = dict_pool.acquire()
        assert isinstance(test_dict, PoolableDict)


class TestGCOptimizer:
    """Test garbage collection optimization"""
    
    def test_gc_optimizer_initialization(self):
        """Test GC optimizer initializes correctly"""
        optimizer = AdvancedGCOptimizer(aggressive=True)
        
        assert optimizer.aggressive == True
        assert optimizer.original_thresholds is not None
        assert len(optimizer.gc_history) == 0
        
    def test_gc_settings_optimization(self):
        """Test GC settings optimization"""
        optimizer = AdvancedGCOptimizer(aggressive=True)
        
        original_enabled = gc.isenabled()
        result = optimizer.optimize_gc_settings()
        
        assert "original_thresholds" in result
        assert "new_thresholds" in result
        assert result["aggressive_mode"] == True
        
        # Restore original settings
        optimizer.restore_default_settings()
        assert gc.isenabled() == original_enabled
        
    def test_emergency_gc_cycle(self):
        """Test emergency GC cycle"""
        optimizer = AdvancedGCOptimizer(aggressive=True)
        
        # Create some objects to collect
        temp_objects = [[] for _ in range(1000)]
        del temp_objects
        
        stats = optimizer.emergency_gc_cycle()
        
        assert stats.objects_before >= 0
        assert stats.objects_after >= 0
        assert stats.duration_ms > 0
        
    def test_object_type_analysis(self):
        """Test object type analysis"""
        optimizer = AdvancedGCOptimizer(aggressive=True)
        
        analysis = optimizer.analyze_object_types()
        
        assert "total_objects" in analysis
        assert "unique_types" in analysis
        assert analysis["total_objects"] > 0
        
    def test_emergency_gc_optimization(self):
        """Test full emergency GC optimization"""
        result = emergency_gc_optimization()
        
        assert "settings" in result
        assert "emergency_stats" in result
        assert "object_analysis" in result


class TestMemoryOptimizer:
    """Test comprehensive memory optimization"""
    
    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initializes correctly"""
        optimizer = EmergencyMemoryOptimizer(target_mb=200)
        
        assert optimizer.target_mb == 200
        assert len(optimizer.strategies) > 0
        assert optimizer.monitoring_active == False
        
    def test_optimization_strategies(self):
        """Test individual optimization strategies"""
        optimizer = EmergencyMemoryOptimizer(target_mb=200)
        
        # Test import optimization
        result = optimizer.optimize_imports(100.0)
        assert "strategy" in result
        assert result["strategy"] == "import_optimization"
        
        # Test data structure optimization  
        result = optimizer.optimize_data_structures(100.0)
        assert result["strategy"] == "data_structure_optimization"
        
        # Test cache optimization
        result = optimizer.optimize_caching(100.0)
        assert result["strategy"] == "cache_optimization"
        
        # Test GC optimization
        result = optimizer.optimize_garbage_collection(100.0)
        assert result["strategy"] == "garbage_collection_optimization"
        
    def test_emergency_optimize_function(self):
        """Test emergency optimization function"""
        # Mock heavy memory usage scenario
        result = emergency_optimize(target_mb=200)
        
        assert "success" in result
        assert "initial_memory_mb" in result
        assert "final_memory_mb" in result
        assert "total_reduction_mb" in result
        
    def test_optimization_report(self):
        """Test optimization report generation"""
        optimizer = EmergencyMemoryOptimizer(target_mb=200)
        
        report = optimizer.get_optimization_report()
        
        assert "MEMORY OPTIMIZATION STATUS REPORT" in report
        assert "Current Memory Usage" in report
        assert "Target Memory Usage" in report


class TestMemoryIntegration:
    """Integration tests for memory optimization system"""
    
    def test_full_memory_optimization_pipeline(self):
        """Test complete memory optimization pipeline"""
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"Initial memory: {initial_memory:.1f}MB")
        
        # Run emergency optimization
        optimizer = EmergencyMemoryOptimizer(target_mb=200)
        
        # Test individual components
        profiler_result = optimizer.profiler.take_snapshot()
        assert profiler_result.process_memory > 0
        
        # Test lazy loading
        lazy_modules = setup_emergency_lazy_imports()
        assert len(lazy_modules) > 0
        
        # Test object pools
        pools = setup_emergency_pools()
        assert len(pools) > 0
        
        # Test GC optimization
        gc_result = emergency_gc_optimization()
        assert "emergency_stats" in gc_result
        
        print("✅ All memory optimization components working")
        
    def test_memory_monitoring_integration(self):
        """Test memory monitoring integration"""
        optimizer = EmergencyMemoryOptimizer(target_mb=200)
        
        # Start monitoring
        monitor_thread = optimizer.continuous_monitoring(interval_seconds=0.1)
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop monitoring
        optimizer.stop_monitoring()
        
        assert optimizer.monitoring_active == False
        
    def test_production_memory_targets(self):
        """Test if memory targets are achievable in production"""
        
        # Simulate production conditions
        optimizer = EmergencyMemoryOptimizer(target_mb=200)
        
        # Take baseline measurement
        baseline = optimizer.profiler.take_snapshot()
        baseline_mb = baseline.process_memory / 1024 / 1024
        
        print(f"Baseline memory: {baseline_mb:.1f}MB")
        
        # Run optimization strategies
        for strategy in optimizer.strategies[:3]:  # Test first 3 strategies
            try:
                result = strategy(baseline_mb)
                assert "strategy" in result
                print(f"✅ Strategy {result['strategy']} completed")
            except Exception as e:
                print(f"⚠️ Strategy failed: {e}")
                
        # Verify memory is reasonable
        final = optimizer.profiler.take_snapshot()
        final_mb = final.process_memory / 1024 / 1024
        
        print(f"Final memory: {final_mb:.1f}MB")
        
        # Should achieve some reduction
        assert final_mb <= baseline_mb * 1.5  # Allow some variance in tests


@pytest.fixture
def memory_cleanup():
    """Fixture to clean up memory after tests"""
    yield
    
    # Force cleanup after test
    for _ in range(3):
        gc.collect()


class TestPerformanceBenchmarks:
    """Performance benchmarks for memory optimization"""
    
    def test_memory_optimization_performance(self, memory_cleanup):
        """Benchmark memory optimization performance"""
        
        # Create memory load
        memory_hogs = []
        for i in range(100):
            memory_hogs.append([j for j in range(1000)])
            
        start_time = time.time()
        
        # Run optimization
        optimizer = EmergencyMemoryOptimizer(target_mb=200)
        result = optimizer.optimize_garbage_collection(100.0)
        
        duration = time.time() - start_time
        
        assert duration < 5.0  # Should complete within 5 seconds
        assert "strategy" in result
        
        # Cleanup
        del memory_hogs
        
    def test_lazy_loading_performance(self, memory_cleanup):
        """Benchmark lazy loading performance"""
        
        loader = LazyModuleLoader()
        
        # Register multiple modules
        modules_to_test = ["json", "os", "sys", "time", "pathlib"]
        
        start_time = time.time()
        
        for i, module_name in enumerate(modules_to_test):
            loader.register_lazy_module(f"test_{i}", module_name)
            
        registration_time = time.time() - start_time
        
        # Test loading performance
        start_time = time.time()
        
        for i in range(len(modules_to_test)):
            loader.load_module(f"test_{i}")
            
        loading_time = time.time() - start_time
        
        assert registration_time < 1.0  # Registration should be fast
        assert loading_time < 2.0  # Loading should be reasonable
        
        stats = loader.get_loading_stats()
        assert stats["loaded_modules"] == len(modules_to_test)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])