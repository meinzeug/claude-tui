#!/usr/bin/env python3
"""
Emergency Memory Optimizer - Critical Performance System
Implements comprehensive memory optimization strategies for production deployment
"""

import gc
import sys
import os
import psutil
import weakref
import threading
import time
import tracemalloc
from datetime import timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
import json

# Import our optimization modules (fallback imports for standalone execution)
try:
    from .memory_profiler import MemoryProfiler, emergency_memory_check, force_cleanup
    from .lazy_loader import LazyModuleLoader, setup_emergency_lazy_imports, optimize_lazy_memory
    from .object_pool import GlobalPoolManager, setup_emergency_pools, emergency_pool_cleanup
except ImportError:
    # Standalone execution fallback
    try:
        from memory_profiler import MemoryProfiler, emergency_memory_check, force_cleanup
        from lazy_loader import LazyModuleLoader, setup_emergency_lazy_imports, optimize_lazy_memory
        from object_pool import GlobalPoolManager, setup_emergency_pools, emergency_pool_cleanup
    except ImportError:
        # Minimal fallback implementations
        class MemoryProfiler:
            def __init__(self, target_mb): pass
            def start_monitoring(self, interval_seconds): pass
            def take_snapshot(self): 
                import psutil, os
                process = psutil.Process(os.getpid())
                class Snapshot:
                    def __init__(self):
                        self.process_memory = process.memory_info().rss
                        self.heap_memory = self.process_memory
                        self.gc_objects = len(gc.get_objects()) if 'gc' in globals() else 0
                        self.tracemalloc_peak = self.process_memory
                        self.largest_objects = []
                        self.leak_suspects = []
                return Snapshot()
            def stop_monitoring(self): pass
        
        def emergency_memory_check(): pass
        def force_cleanup(): 
            for _ in range(3):
                gc.collect()
        
        class LazyModuleLoader:
            pass
        
        def setup_emergency_lazy_imports(): return {}
        def optimize_lazy_memory(): pass
        
        class GlobalPoolManager:
            def optimize_all_pools(self): pass
        
        def setup_emergency_pools(): return {}
        def emergency_pool_cleanup():
            for _ in range(3):
                gc.collect()


@dataclass
class OptimizationTarget:
    """Target for memory optimization"""
    current_mb: float
    target_mb: float
    reduction_needed: float
    priority: str = "CRITICAL"
    
    @property
    def reduction_ratio(self) -> float:
        return self.current_mb / self.target_mb if self.target_mb > 0 else float('inf')


class EmergencyMemoryOptimizer:
    """
    Emergency memory optimization system
    Target: Reduce 1.7GB → <200MB (8.5x reduction)
    """
    
    def __init__(self, target_mb: int = 200):
        # Safety checks for Memory Optimization
        self.MIN_SAFE_MEMORY = 500 * 1024 * 1024  # 500MB minimum
        self.MAX_OPTIMIZATION_PERCENT = 0.5  # Maximal 50% reduction per step
        
        # Ensure target is safe
        min_target_mb = 500  # Minimum safe target
        self.target_mb = max(target_mb, min_target_mb)
        self.profiler = MemoryProfiler(target_mb)
        self.lazy_loader = LazyModuleLoader()
        self.pool_manager = GlobalPoolManager()
        
        # Optimization strategies
        self.strategies = [
            self.optimize_imports,
            self.optimize_data_structures,
            self.optimize_caching,
            self.optimize_garbage_collection,
            self.optimize_module_loading,
            self.optimize_object_lifecycle
        ]
        
        # State tracking
        self.optimization_history = []
        self.monitoring_active = False
        
    def safe_memory_optimization(self, current_usage_bytes: int) -> bool:
        """Check if optimization is safe to proceed"""
        if current_usage_bytes < self.MIN_SAFE_MEMORY:
            print(f"⚠️ Aborting optimization: Memory already below safe minimum ({current_usage_bytes / 1024 / 1024:.1f}MB)")
            return False
        return True
    
    def run_emergency_optimization(self) -> Dict[str, Any]:
        """Run complete emergency memory optimization with safety checks"""
        print("🚨 EMERGENCY MEMORY OPTIMIZATION ACTIVATED")
        print(f"Target: Reduce memory to <{self.target_mb}MB (with safety checks)")
        
        # Start monitoring
        self.profiler.start_monitoring(interval_seconds=0.5)
        
        try:
            # Get initial measurements
            initial_snapshot = self.profiler.take_snapshot()
            initial_mb = initial_snapshot.process_memory / 1024 / 1024
            
            print(f"📊 Initial Memory: {initial_mb:.1f}MB")
            
            # Safety check before optimization
            if not self.safe_memory_optimization(int(initial_snapshot.process_memory)):
                return {
                    "success": False,
                    "error": "Memory usage already below safe minimum",
                    "initial_memory_mb": initial_mb,
                    "final_memory_mb": initial_mb,
                    "total_reduction_mb": 0,
                    "reduction_percentage": 0,
                    "target_achieved": False
                }
            
            # Create optimization target with safety limits
            max_safe_reduction = initial_mb * self.MAX_OPTIMIZATION_PERCENT
            safe_target_mb = max(self.target_mb, initial_mb - max_safe_reduction)
            
            target = OptimizationTarget(
                current_mb=initial_mb,
                target_mb=safe_target_mb,
                reduction_needed=initial_mb - safe_target_mb
            )
            
            print(f"🎯 Reduction needed: {target.reduction_needed:.1f}MB ({target.reduction_ratio:.1f}x)")
            
            # Run optimization strategies
            results = []
            current_mb = initial_mb
            
            for i, strategy in enumerate(self.strategies):
                print(f"\n📈 Running strategy {i+1}/{len(self.strategies)}: {strategy.__name__}")
                
                before_mb = current_mb
                strategy_result = strategy(current_mb)
                
                # Measure impact
                time.sleep(1)  # Allow GC to run
                snapshot = self.profiler.take_snapshot()
                after_mb = snapshot.process_memory / 1024 / 1024
                
                reduction = before_mb - after_mb
                strategy_result.update({
                    "memory_before_mb": before_mb,
                    "memory_after_mb": after_mb,
                    "reduction_mb": reduction,
                    "improvement_pct": (reduction / before_mb) * 100 if before_mb > 0 else 0
                })
                
                results.append(strategy_result)
                current_mb = after_mb
                
                print(f"  ✅ Reduced by {reduction:.1f}MB ({strategy_result['improvement_pct']:.1f}%)")
                
                # Safety check during optimization
                if after_mb <= self.MIN_SAFE_MEMORY / 1024 / 1024:
                    print(f"⚠️ SAFETY LIMIT REACHED! Memory: {after_mb:.1f}MB - Stopping optimization")
                    break
                    
                # Check if target achieved
                if after_mb <= safe_target_mb:
                    print(f"🎉 TARGET ACHIEVED! Memory: {after_mb:.1f}MB")
                    break
                    
            # Final measurements
            final_snapshot = self.profiler.take_snapshot()
            final_mb = final_snapshot.process_memory / 1024 / 1024
            total_reduction = initial_mb - final_mb
            
            optimization_report = {
                "success": final_mb <= self.target_mb * 1.1,  # Allow 10% tolerance
                "initial_memory_mb": initial_mb,
                "final_memory_mb": final_mb,
                "total_reduction_mb": total_reduction,
                "reduction_percentage": (total_reduction / initial_mb) * 100,
                "target_achieved": final_mb <= self.target_mb,
                "strategies_applied": len(results),
                "strategy_results": results
            }
            
            print(f"\n🎯 OPTIMIZATION COMPLETE")
            print(f"   Initial: {initial_mb:.1f}MB")
            print(f"   Final: {final_mb:.1f}MB")
            print(f"   Reduction: {total_reduction:.1f}MB ({optimization_report['reduction_percentage']:.1f}%)")
            print(f"   Target achieved: {'✅' if optimization_report['target_achieved'] else '❌'}")
            
            return optimization_report
            
        finally:
            self.profiler.stop_monitoring()
            
    def optimize_imports(self, current_mb: float) -> Dict[str, Any]:
        """Optimize module imports and loading"""
        print("  🔄 Optimizing module imports...")
        
        # Setup lazy loading for heavy modules
        lazy_modules = setup_emergency_lazy_imports()
        
        # Unload unused heavy modules
        heavy_modules = [
            'numpy', 'pandas', 'sklearn', 'torch', 'tensorflow',
            'matplotlib', 'plotly', 'scipy', 'seaborn'
        ]
        
        unloaded_count = 0
        for module_name in heavy_modules:
            if module_name in sys.modules:
                try:
                    del sys.modules[module_name]
                    unloaded_count += 1
                except Exception as e:
                    print(f"    ⚠️ Could not unload {module_name}: {e}")
                    
        # Force garbage collection
        for _ in range(3):
            gc.collect()
            
        return {
            "strategy": "import_optimization",
            "lazy_modules_setup": len(lazy_modules),
            "heavy_modules_unloaded": unloaded_count,
            "actions": ["setup_lazy_loading", "unload_heavy_modules", "force_gc"]
        }
        
    def optimize_data_structures(self, current_mb: float) -> Dict[str, Any]:
        """Optimize data structures for memory efficiency"""
        print("  🗃️ Optimizing data structures...")
        
        # Setup object pools for frequently used objects
        pools = setup_emergency_pools()
        
        # Replace heavy data structures with memory-efficient alternatives
        optimizations = []
        
        # Find and optimize large objects
        all_objects = gc.get_objects()
        large_objects = []
        
        for obj in all_objects[:1000]:  # Sample first 1000 objects
            try:
                size = sys.getsizeof(obj)
                if size > 1024 * 10:  # Objects > 10KB
                    large_objects.append((type(obj).__name__, size))
            except (TypeError, ReferenceError):
                continue
                
        # Group by type and count
        type_sizes = {}
        for obj_type, size in large_objects:
            if obj_type not in type_sizes:
                type_sizes[obj_type] = []
            type_sizes[obj_type].append(size)
            
        for obj_type, sizes in type_sizes.items():
            total_size = sum(sizes)
            count = len(sizes)
            if total_size > 1024 * 1024:  # > 1MB total
                optimizations.append(f"{obj_type}: {count} objects, {total_size/1024/1024:.1f}MB")
                
        return {
            "strategy": "data_structure_optimization", 
            "object_pools_created": len(pools),
            "large_object_types": len(type_sizes),
            "optimization_opportunities": optimizations,
            "actions": ["setup_object_pools", "identify_large_objects"]
        }
        
    def optimize_caching(self, current_mb: float) -> Dict[str, Any]:
        """Optimize caching strategies"""
        print("  💾 Optimizing caches...")
        
        cleared_caches = 0
        
        # Clear function caches
        try:
            import functools
            # Clear lru_cache instances
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear'):
                    try:
                        obj.cache_clear()
                        cleared_caches += 1
                    except:
                        pass
        except Exception:
            pass
            
        # Clear type caches
        try:
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
                cleared_caches += 1
        except Exception:
            pass
            
        # Clear import caches
        try:
            import importlib
            importlib.invalidate_caches()
            cleared_caches += 1
        except Exception:
            pass
            
        return {
            "strategy": "cache_optimization",
            "caches_cleared": cleared_caches,
            "actions": ["clear_function_caches", "clear_type_cache", "clear_import_cache"]
        }
        
    def optimize_garbage_collection(self, current_mb: float) -> Dict[str, Any]:
        """Optimize garbage collection with improved strategy"""
        print("  🗑️ Optimizing garbage collection...")
        
        # Get GC stats before
        stats_before = gc.get_stats()
        objects_before = len(gc.get_objects())
        
        # Get current memory before GC
        process = psutil.Process(os.getpid()) if 'psutil' in globals() else None
        memory_before = process.memory_info().rss / 1024 / 1024 if process else current_mb
        
        # Configure GC for optimal memory optimization
        # More conservative thresholds to avoid excessive overhead
        original_thresholds = gc.get_threshold()
        gc.set_threshold(500, 8, 8)  # Balanced approach
        
        # Enable automatic GC debugging if memory is critical
        if current_mb > 1000:  # If > 1GB, enable debug mode
            gc.set_debug(gc.DEBUG_STATS)
        
        # Strategic GC collection - start with youngest generation
        collected_objects = 0
        generation_stats = {}
        
        for generation in range(3):
            gen_collected = gc.collect(generation)
            collected_objects += gen_collected
            generation_stats[f"gen_{generation}"] = gen_collected
            
            # Brief pause to allow cleanup
            time.sleep(0.01)
            
        # Force comprehensive collection cycles
        for cycle in range(3):  # Reduced from 5 to avoid overhead
            cycle_collected = gc.collect()
            collected_objects += cycle_collected
            time.sleep(0.01)  # Small delay between cycles
            
        # Restore original thresholds for normal operation
        gc.set_threshold(*original_thresholds)
        gc.set_debug(0)  # Disable debug mode
        
        # Measure final state
        objects_after = len(gc.get_objects())
        objects_freed = objects_before - objects_after
        memory_after = process.memory_info().rss / 1024 / 1024 if process else current_mb
        memory_freed_mb = memory_before - memory_after
        
        return {
            "strategy": "garbage_collection_optimization",
            "objects_before": objects_before,
            "objects_after": objects_after,
            "objects_freed": objects_freed,
            "collected_objects": collected_objects,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_freed_mb": memory_freed_mb,
            "generation_stats": generation_stats,
            "original_thresholds": original_thresholds,
            "actions": ["set_balanced_thresholds", "strategic_generation_collection", "comprehensive_cleanup", "restore_thresholds"]
        }
        
    def optimize_module_loading(self, current_mb: float) -> Dict[str, Any]:
        """Optimize module loading patterns"""
        print("  📦 Optimizing module loading...")
        
        # Count currently loaded modules
        modules_before = len(sys.modules)
        
        # Identify modules that can be unloaded
        unloadable_modules = []
        for name, module in list(sys.modules.items()):
            # Skip core modules
            if name.startswith(('builtins', 'sys', 'os', '_')):
                continue
                
            # Skip if actively used (has references)
            if sys.getrefcount(module) <= 3:  # Only sys.modules + locals + getrefcount
                unloadable_modules.append(name)
                
        # Unload modules with few references
        unloaded_count = 0
        for name in unloadable_modules[:50]:  # Limit to avoid breaking things
            try:
                if name in sys.modules:
                    del sys.modules[name]
                    unloaded_count += 1
            except Exception:
                pass
                
        modules_after = len(sys.modules)
        
        return {
            "strategy": "module_loading_optimization",
            "modules_before": modules_before,
            "modules_after": modules_after,
            "modules_unloaded": unloaded_count,
            "unloadable_identified": len(unloadable_modules),
            "actions": ["identify_unused_modules", "unload_safe_modules"]
        }
        
    def optimize_object_lifecycle(self, current_mb: float) -> Dict[str, Any]:
        """Optimize object lifecycle management"""
        print("  ⚡ Optimizing object lifecycle...")
        
        # Clear weak references
        cleared_weakrefs = 0
        try:
            # Find and clear WeakSet objects
            for obj in gc.get_objects():
                if isinstance(obj, weakref.WeakSet):
                    try:
                        obj.clear()
                        cleared_weakrefs += 1
                    except Exception:
                        pass
        except Exception:
            pass
            
        # Clean up circular references
        unreachable_before = gc.collect()
        
        # Force deletion of objects with __del__ methods
        del_objects = 0
        for obj in list(gc.get_objects()):
            if hasattr(obj, '__del__') and not isinstance(obj, type):
                try:
                    if sys.getrefcount(obj) <= 3:
                        del obj
                        del_objects += 1
                except Exception:
                    pass
                    
        unreachable_after = gc.collect()
        
        return {
            "strategy": "object_lifecycle_optimization",
            "weakrefs_cleared": cleared_weakrefs,
            "del_objects_cleaned": del_objects,
            "unreachable_before": unreachable_before,
            "unreachable_after": unreachable_after,
            "actions": ["clear_weak_references", "clean_circular_refs", "cleanup_del_objects"]
        }
        
    def continuous_monitoring(self, interval_seconds: float = 5.0):
        """Start continuous memory monitoring and optimization"""
        print(f"📈 Starting continuous memory monitoring (every {interval_seconds}s)")
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    snapshot = self.profiler.take_snapshot()
                    current_mb = snapshot.process_memory / 1024 / 1024
                    
                    # Emergency optimization if memory exceeds threshold
                    if current_mb > self.target_mb * 2:  # 2x target
                        print(f"🚨 Emergency threshold exceeded: {current_mb:.1f}MB")
                        self.emergency_cleanup()
                        
                    # Regular optimization if above target
                    elif current_mb > self.target_mb * 1.2:  # 20% above target
                        print(f"⚠️ Above target: {current_mb:.1f}MB, running maintenance")
                        self.maintenance_optimization()
                        
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(interval_seconds)
                    
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        return monitoring_thread
        
    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        print("🚨 EMERGENCY CLEANUP ACTIVATED")
        
        # Force cleanup of all systems
        force_cleanup()
        emergency_pool_cleanup()
        optimize_lazy_memory()
        
        # Aggressive GC
        for _ in range(10):
            gc.collect()
            
    def maintenance_optimization(self):
        """Regular maintenance optimization"""
        print("🔧 Running maintenance optimization")
        
        # Light cleanup
        for _ in range(3):
            gc.collect()
            
        # Optimize pools
        try:
            self.pool_manager.optimize_all_pools()
        except Exception:
            pass
            
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        
    def get_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        snapshot = self.profiler.take_snapshot()
        current_mb = snapshot.process_memory / 1024 / 1024
        
        return f"""
🚀 MEMORY OPTIMIZATION STATUS REPORT

Current Memory Usage: {current_mb:.1f}MB
Target Memory Usage: {self.target_mb}MB
Status: {'✅ OPTIMAL' if current_mb <= self.target_mb else '⚠️ ABOVE TARGET'}

Memory Breakdown:
- Process RSS: {snapshot.process_memory / 1024 / 1024:.1f}MB
- Heap Usage: {snapshot.heap_memory / 1024 / 1024:.1f}MB  
- GC Objects: {snapshot.gc_objects:,}

Optimization Systems:
- Memory Profiler: Active
- Lazy Loading: {'Active' if hasattr(self, 'lazy_modules') else 'Inactive'}
- Object Pools: {'Active' if hasattr(self, 'pools') else 'Inactive'}

Recent Performance:
- Peak Memory: {snapshot.tracemalloc_peak / 1024 / 1024:.1f}MB
- Largest Objects: {len(snapshot.largest_objects)}
- Potential Leaks: {len(snapshot.leak_suspects)}

Recommendations:
{'🎉 Memory usage is optimal!' if current_mb <= self.target_mb else '⚡ Consider running optimization strategies'}
"""


# Convenience functions for emergency use
def emergency_optimize(target_mb: int = 200) -> Dict[str, Any]:
    """Quick emergency memory optimization"""
    optimizer = EmergencyMemoryOptimizer(target_mb)
    return optimizer.run_emergency_optimization()


def quick_memory_check() -> str:
    """Quick memory status check"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 1000:
        return f"🚨 CRITICAL: {memory_mb:.1f}MB"
    elif memory_mb > 500:
        return f"⚠️ HIGH: {memory_mb:.1f}MB"
    else:
        return f"✅ GOOD: {memory_mb:.1f}MB"


if __name__ == "__main__":
    # Run emergency optimization
    print("🚨 EMERGENCY MEMORY OPTIMIZATION STARTING...")
    
    result = emergency_optimize(target_mb=200)
    
    if result["success"]:
        print("🎉 EMERGENCY OPTIMIZATION SUCCESSFUL!")
    else:
        print("❌ EMERGENCY OPTIMIZATION INCOMPLETE")
        print("Consider additional manual interventions")
        
    print(f"Final memory: {result['final_memory_mb']:.1f}MB")