#!/usr/bin/env python3
"""
Performance Optimizations for Claude-TUI
Based on performance analysis results, implement optimizations for identified areas.
"""

import asyncio
import gc
import sys
import time
import weakref
from collections import deque, defaultdict
from functools import lru_cache, wraps
from typing import Dict, List, Any, Optional, Callable, Generator
import threading
from pathlib import Path


class ObjectPool:
    """Object pool for frequently created/destroyed objects"""
    
    def __init__(self, factory_func: Callable, max_size: int = 100):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.active_objects = weakref.WeakSet()
        
    def acquire(self, *args, **kwargs):
        """Acquire an object from the pool"""
        try:
            obj = self.pool.popleft()
            # Reset the object if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset(*args, **kwargs)
            else:
                # Re-initialize key attributes
                for key, value in kwargs.items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)
        except IndexError:
            # Pool is empty, create new object
            obj = self.factory_func(*args, **kwargs)
            
        self.active_objects.add(obj)
        return obj
        
    def release(self, obj):
        """Release an object back to the pool"""
        if obj in self.active_objects:
            self.active_objects.discard(obj)
            if len(self.pool) < self.max_size:
                # Clean up the object before pooling
                if hasattr(obj, 'cleanup'):
                    obj.cleanup()
                self.pool.append(obj)
                
    def clear(self):
        """Clear the pool"""
        self.pool.clear()
        
    def stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        return {
            'pool_size': len(self.pool),
            'active_objects': len(self.active_objects),
            'max_size': self.max_size
        }


class MemoryOptimizedCache:
    """Memory-efficient LRU cache with automatic cleanup"""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self._lock = threading.RLock()
        
    def get(self, key: Any, default=None):
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                # Check TTL if specified
                if self.ttl and time.time() - self.creation_times[key] > self.ttl:
                    self._remove_key(key)
                    return default
                    
                self.access_times[key] = time.time()
                return self.cache[key]
            return default
            
    def set(self, key: Any, value: Any):
        """Set item in cache"""
        with self._lock:
            current_time = time.time()
            
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.maxsize and key not in self.cache:
                self._evict_lru()
                
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
            
    def _remove_key(self, key: Any):
        """Remove a key from all cache structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
        
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
        
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            expired_count = 0
            if self.ttl:
                current_time = time.time()
                expired_count = sum(
                    1 for t in self.creation_times.values()
                    if current_time - t > self.ttl
                )
                
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hit_ratio': 0,  # Would need hit/miss tracking
                'expired_items': expired_count
            }


class VirtualScrollingOptimizer:
    """Virtual scrolling implementation for large datasets"""
    
    def __init__(self, item_height: int = 20, viewport_height: int = 400):
        self.item_height = item_height
        self.viewport_height = viewport_height
        self.items_per_page = viewport_height // item_height
        self.buffer_size = self.items_per_page // 2  # Extra items for smooth scrolling
        
    def get_visible_range(self, scroll_position: int, total_items: int) -> tuple[int, int]:
        """Calculate which items should be visible"""
        start_index = max(0, (scroll_position // self.item_height) - self.buffer_size)
        end_index = min(total_items, start_index + self.items_per_page + (2 * self.buffer_size))
        return start_index, end_index
        
    def get_virtual_height(self, total_items: int) -> int:
        """Get the total virtual height of all items"""
        return total_items * self.item_height
        
    def get_offset_for_index(self, index: int) -> int:
        """Get the pixel offset for a given item index"""
        return index * self.item_height


class BatchUpdateManager:
    """Batch multiple UI updates to reduce render frequency"""
    
    def __init__(self, batch_interval: float = 0.016):  # ~60 FPS
        self.batch_interval = batch_interval
        self.pending_updates = {}
        self.update_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        
    def schedule_update(self, widget_id: str, update_func: Callable, *args, **kwargs):
        """Schedule an update for batching"""
        with self._lock:
            # Store the latest update for each widget
            self.pending_updates[widget_id] = (update_func, args, kwargs)
            
            # Start timer if not already running
            if self.update_timer is None:
                self.update_timer = threading.Timer(self.batch_interval, self._flush_updates)
                self.update_timer.start()
                
    def _flush_updates(self):
        """Execute all pending updates"""
        with self._lock:
            updates_to_process = self.pending_updates.copy()
            self.pending_updates.clear()
            self.update_timer = None
            
        # Execute updates outside the lock
        for widget_id, (update_func, args, kwargs) in updates_to_process.items():
            try:
                update_func(*args, **kwargs)
            except Exception as e:
                print(f"Batch update error for {widget_id}: {e}")
                
    def flush_immediately(self):
        """Force immediate flush of all pending updates"""
        if self.update_timer:
            self.update_timer.cancel()
        self._flush_updates()


class LazyDataLoader:
    """Lazy loading for large datasets"""
    
    def __init__(self, data_source: Callable[[int, int], List], chunk_size: int = 100):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.cache = MemoryOptimizedCache(maxsize=50)  # Cache 50 chunks
        
    def get_items(self, start_index: int, count: int) -> List:
        """Get items with lazy loading and caching"""
        items = []
        current_index = start_index
        remaining_count = count
        
        while remaining_count > 0:
            chunk_index = current_index // self.chunk_size
            chunk_offset = current_index % self.chunk_size
            
            # Check cache first
            chunk_key = f"chunk_{chunk_index}"
            chunk_data = self.cache.get(chunk_key)
            
            if chunk_data is None:
                # Load chunk from data source
                chunk_start = chunk_index * self.chunk_size
                chunk_data = self.data_source(chunk_start, self.chunk_size)
                self.cache.set(chunk_key, chunk_data)
                
            # Extract needed items from chunk
            chunk_items_to_take = min(remaining_count, len(chunk_data) - chunk_offset)
            items.extend(chunk_data[chunk_offset:chunk_offset + chunk_items_to_take])
            
            current_index += chunk_items_to_take
            remaining_count -= chunk_items_to_take
            
        return items[:count]  # Ensure we don't return more than requested


class GCOptimizer:
    """Garbage collection optimization utilities"""
    
    @staticmethod
    def optimize_gc_thresholds():
        """Optimize garbage collection thresholds for better performance"""
        # Get current thresholds
        current = gc.get_threshold()
        
        # Increase thresholds to reduce GC frequency
        # This trades some memory for better performance
        new_thresholds = (
            current[0] * 2,    # Generation 0: young objects
            current[1] * 2,    # Generation 1: medium-lived objects  
            current[2] * 2     # Generation 2: old objects
        )
        
        gc.set_threshold(*new_thresholds)
        return current, new_thresholds
        
    @staticmethod
    def force_full_cleanup():
        """Force comprehensive garbage collection"""
        # Disable GC temporarily
        gc.disable()
        
        try:
            # Clear weak references
            import weakref
            
            # Multiple passes to ensure thorough cleanup
            for _ in range(3):
                collected = gc.collect()
                if collected == 0:
                    break
                    
        finally:
            # Re-enable GC
            gc.enable()


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, alert_threshold_mb: float = 80.0):
        self.alert_threshold_mb = alert_threshold_mb
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history = deque(maxlen=100)  # Keep last 100 measurements
        
    def start_monitoring(self, interval: float = 5.0):
        """Start background performance monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        import psutil
        process = psutil.Process()
        
        while self.monitoring_active:
            try:
                # Collect metrics
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                metric = {
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent,
                    'gc_objects': len(gc.get_objects())
                }
                
                self.metrics_history.append(metric)
                
                # Check for alerts
                if memory_mb > self.alert_threshold_mb:
                    self._trigger_memory_alert(memory_mb)
                    
                time.sleep(interval)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                
    def _trigger_memory_alert(self, memory_mb: float):
        """Trigger memory usage alert"""
        print(f"⚠️ Memory Alert: Usage {memory_mb:.1f}MB exceeds threshold {self.alert_threshold_mb}MB")
        
        # Trigger emergency cleanup
        GCOptimizer.force_full_cleanup()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.metrics_history:
            return {}
            
        latest = self.metrics_history[-1]
        
        # Calculate trends
        if len(self.metrics_history) >= 10:
            recent = list(self.metrics_history)[-10:]
            memory_trend = (recent[-1]['memory_mb'] - recent[0]['memory_mb']) / len(recent)
            cpu_avg = sum(m['cpu_percent'] for m in recent) / len(recent)
        else:
            memory_trend = 0
            cpu_avg = latest.get('cpu_percent', 0)
            
        return {
            'current_memory_mb': latest['memory_mb'],
            'current_cpu_percent': latest['cpu_percent'],
            'memory_trend_mb_per_sample': memory_trend,
            'average_cpu_percent': cpu_avg,
            'gc_objects': latest['gc_objects'],
            'alert_threshold_mb': self.alert_threshold_mb
        }


# Decorators for performance optimization
def memoize_with_ttl(ttl_seconds: float = 300):
    """Memoization decorator with TTL"""
    def decorator(func):
        cache = MemoryOptimizedCache(maxsize=100, ttl=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            result = cache.get(cache_key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(cache_key, result)
                
            return result
            
        wrapper.cache = cache  # Allow access to cache for stats
        return wrapper
    return decorator


def profile_performance(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process()
        
        # Before measurements
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # After measurements
            end_time = time.time()
            end_memory = process.memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Log if significant resource usage
            if execution_time > 0.1 or abs(memory_delta) > 1024 * 1024:  # 100ms or 1MB
                print(f"⚡ {func.__name__}: {execution_time:.3f}s, {memory_delta/1024/1024:.2f}MB")
                
    return wrapper


# Global optimization manager
class OptimizationManager:
    """Central manager for all performance optimizations"""
    
    def __init__(self):
        self.object_pools: Dict[str, ObjectPool] = {}
        self.cache = MemoryOptimizedCache(maxsize=500)
        self.batch_manager = BatchUpdateManager()
        self.performance_monitor = PerformanceMonitor()
        self.gc_optimized = False
        
    def initialize_optimizations(self):
        """Initialize all performance optimizations"""
        print("🚀 Initializing performance optimizations...")
        
        # Optimize GC thresholds
        if not self.gc_optimized:
            old_thresholds, new_thresholds = GCOptimizer.optimize_gc_thresholds()
            print(f"   GC thresholds: {old_thresholds} → {new_thresholds}")
            self.gc_optimized = True
            
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        print("   Performance monitoring started")
        
        # Pre-create common object pools
        self.create_object_pool('widget', lambda **kwargs: self._create_mock_widget(**kwargs))
        self.create_object_pool('task', lambda **kwargs: self._create_mock_task(**kwargs))
        
        print("✅ Performance optimizations initialized")
        
    def create_object_pool(self, name: str, factory_func: Callable, max_size: int = 100):
        """Create a named object pool"""
        self.object_pools[name] = ObjectPool(factory_func, max_size)
        
    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get a named object pool"""
        return self.object_pools.get(name)
        
    def _create_mock_widget(self, **kwargs):
        """Factory for mock widgets"""
        class OptimizedMockWidget:
            def __init__(self, **kwargs):
                self.widget_type = kwargs.get('widget_type', 'generic')
                self.widget_id = kwargs.get('widget_id', 0)
                self.data = kwargs.get('data', '')
                self.visible = kwargs.get('visible', True)
                
            def reset(self, **kwargs):
                self.__init__(**kwargs)
                
            def cleanup(self):
                self.data = ''
                self.visible = False
                
        return OptimizedMockWidget(**kwargs)
        
    def _create_mock_task(self, **kwargs):
        """Factory for mock tasks"""
        class OptimizedMockTask:
            def __init__(self, **kwargs):
                self.task_id = kwargs.get('task_id', 0)
                self.name = kwargs.get('name', '')
                self.status = kwargs.get('status', 'pending')
                
            def reset(self, **kwargs):
                self.__init__(**kwargs)
                
            def cleanup(self):
                self.name = ''
                self.status = 'pending'
                
        return OptimizedMockTask(**kwargs)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization performance"""
        summary = {
            'monitoring': self.performance_monitor.get_current_metrics(),
            'cache_stats': self.cache.stats(),
            'object_pools': {
                name: pool.stats() for name, pool in self.object_pools.items()
            },
            'gc_optimized': self.gc_optimized
        }
        
        return summary
        
    def cleanup(self):
        """Clean up all optimization resources"""
        print("🧹 Cleaning up performance optimizations...")
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Clear caches
        self.cache.clear()
        
        # Clear object pools
        for pool in self.object_pools.values():
            pool.clear()
            
        # Force final GC
        GCOptimizer.force_full_cleanup()
        
        print("✅ Optimization cleanup complete")


# Global instance
optimization_manager = OptimizationManager()


def initialize_performance_optimizations():
    """Initialize all performance optimizations"""
    optimization_manager.initialize_optimizations()
    return optimization_manager


if __name__ == "__main__":
    # Test the optimizations
    print("🧪 Testing performance optimizations...")
    
    # Initialize optimizations
    manager = initialize_performance_optimizations()
    
    # Test object pooling
    widget_pool = manager.get_object_pool('widget')
    if widget_pool:
        # Create and release widgets
        widgets = []
        for i in range(100):
            widget = widget_pool.acquire(widget_type='test', widget_id=i)
            widgets.append(widget)
            
        for widget in widgets:
            widget_pool.release(widget)
            
        print(f"   Widget pool stats: {widget_pool.stats()}")
    
    # Test caching
    @memoize_with_ttl(ttl_seconds=60)
    def expensive_computation(n):
        time.sleep(0.01)  # Simulate work
        return sum(range(n))
        
    # First call - cache miss
    start = time.time()
    result1 = expensive_computation(1000)
    time1 = time.time() - start
    
    # Second call - cache hit
    start = time.time()
    result2 = expensive_computation(1000)
    time2 = time.time() - start
    
    print(f"   Cache test: {time1:.3f}s vs {time2:.3f}s (speedup: {time1/time2:.1f}x)")
    
    # Test virtual scrolling
    virtual_scroller = VirtualScrollingOptimizer(item_height=25, viewport_height=500)
    visible_range = virtual_scroller.get_visible_range(scroll_position=250, total_items=10000)
    print(f"   Virtual scrolling: showing items {visible_range[0]}-{visible_range[1]} of 10,000")
    
    # Get performance summary
    time.sleep(1)  # Let monitoring collect data
    summary = manager.get_performance_summary()
    print(f"   Current memory: {summary['monitoring'].get('current_memory_mb', 0):.1f}MB")
    
    # Cleanup
    manager.cleanup()
    
    print("✅ Performance optimization testing complete")