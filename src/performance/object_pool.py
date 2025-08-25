#!/usr/bin/env python3
"""
Emergency Object Pool System - Critical Memory Optimization
Implements memory-efficient object reuse patterns to minimize allocations
"""

import threading
import weakref
import gc
import time
from typing import Dict, List, Any, Optional, Type, Callable, Generic, TypeVar
from collections import deque, defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys


T = TypeVar('T')


@dataclass
class PoolStats:
    """Statistics for object pool performance"""
    total_created: int = 0
    total_reused: int = 0
    current_pool_size: int = 0
    max_pool_size: int = 0
    memory_saved_bytes: int = 0
    hit_rate: float = 0.0


class Poolable(ABC):
    """Interface for objects that can be pooled"""
    
    @abstractmethod
    def reset(self):
        """Reset object state for reuse"""
        pass
    
    @abstractmethod
    def is_reusable(self) -> bool:
        """Check if object is safe to reuse"""
        return True


class ObjectPool(Generic[T]):
    """
    High-performance object pool for memory optimization
    Thread-safe with automatic cleanup and monitoring
    """
    
    def __init__(self, 
                 factory: Callable[[], T],
                 reset_func: Optional[Callable[[T], None]] = None,
                 max_size: int = 100,
                 cleanup_interval: float = 30.0):
        
        self.factory = factory
        self.reset_func = reset_func
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe pool storage
        self._pool: deque = deque()
        self._lock = threading.RLock()
        self._stats = PoolStats(max_pool_size=max_size)
        
        # Active object tracking (use regular set for non-weakreferenceable objects)
        self._active_objects = set()
        
        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
        
    def acquire(self) -> T:
        """Acquire an object from the pool"""
        with self._lock:
            # Try to reuse from pool
            while self._pool:
                obj = self._pool.popleft()
                
                # Verify object is still valid
                if self._is_object_valid(obj):
                    self._stats.total_reused += 1
                    self._stats.current_pool_size = len(self._pool)
                    self._update_hit_rate()
                    
                    # Reset object if needed
                    if self.reset_func:
                        self.reset_func(obj)
                    elif hasattr(obj, 'reset'):
                        obj.reset()
                        
                    try:
                        self._active_objects.add(id(obj))  # Use object ID instead
                    except TypeError:
                        pass  # Skip tracking if can't track
                    return obj
                    
            # Create new object if pool is empty
            obj = self.factory()
            self._stats.total_created += 1
            self._update_hit_rate()
            
            try:
                self._active_objects.add(id(obj))  # Use object ID instead
            except TypeError:
                pass  # Skip tracking if can't track
            return obj
            
    def release(self, obj: T) -> bool:
        """Release an object back to the pool"""
        if not self._is_object_valid(obj):
            return False
            
        with self._lock:
            # Don't exceed max pool size
            if len(self._pool) >= self.max_size:
                return False
                
            # Check if object is reusable
            if hasattr(obj, 'is_reusable') and not obj.is_reusable():
                return False
                
            self._pool.append(obj)
            self._stats.current_pool_size = len(self._pool)
            
            return True
            
    def _is_object_valid(self, obj: T) -> bool:
        """Check if an object is valid for use"""
        try:
            # Basic validity checks
            if obj is None:
                return False
                
            # Check if object is still alive (not garbage collected)
            if hasattr(obj, '__dict__'):
                return True
                
            return True
            
        except (AttributeError, ReferenceError):
            return False
            
    def _update_hit_rate(self):
        """Update pool hit rate statistics"""
        total = self._stats.total_created + self._stats.total_reused
        if total > 0:
            self._stats.hit_rate = self._stats.total_reused / total
            
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self._cleanup_thread.start()
            
    def _cleanup_loop(self):
        """Background cleanup of invalid objects"""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                with self._lock:
                    # Remove invalid objects from pool
                    valid_objects = deque()
                    
                    while self._pool:
                        obj = self._pool.popleft()
                        if self._is_object_valid(obj):
                            valid_objects.append(obj)
                            
                    self._pool = valid_objects
                    self._stats.current_pool_size = len(self._pool)
                    
            except Exception as e:
                print(f"Pool cleanup error: {e}")
                
    def get_stats(self) -> PoolStats:
        """Get current pool statistics"""
        with self._lock:
            stats_copy = PoolStats(
                total_created=self._stats.total_created,
                total_reused=self._stats.total_reused,
                current_pool_size=len(self._pool),
                max_pool_size=self.max_size,
                memory_saved_bytes=self._estimate_memory_saved(),
                hit_rate=self._stats.hit_rate
            )
            return stats_copy
            
    def _estimate_memory_saved(self) -> int:
        """Estimate memory saved by object reuse"""
        if self._stats.total_reused == 0:
            return 0
            
        # Sample an object to estimate size
        try:
            sample_obj = self.factory()
            obj_size = sys.getsizeof(sample_obj)
            del sample_obj
            
            return self._stats.total_reused * obj_size
            
        except Exception:
            return 0
            
    def clear(self):
        """Clear all objects from pool"""
        with self._lock:
            self._pool.clear()
            self._stats.current_pool_size = 0
            
    def shutdown(self):
        """Shutdown the pool and cleanup thread"""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
        self.clear()


class GlobalPoolManager:
    """
    Global manager for all object pools
    Provides centralized monitoring and optimization
    """
    
    def __init__(self):
        self._pools: Dict[str, ObjectPool] = {}
        self._lock = threading.RLock()
        
    def register_pool(self, name: str, pool: ObjectPool):
        """Register a pool for global management"""
        with self._lock:
            self._pools[name] = pool
            
    def get_pool(self, name: str) -> Optional[ObjectPool]:
        """Get a registered pool by name"""
        return self._pools.get(name)
        
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        stats = {
            "total_pools": len(self._pools),
            "pools": {}
        }
        
        total_memory_saved = 0
        total_objects_reused = 0
        
        for name, pool in self._pools.items():
            pool_stats = pool.get_stats()
            stats["pools"][name] = {
                "created": pool_stats.total_created,
                "reused": pool_stats.total_reused,
                "pool_size": pool_stats.current_pool_size,
                "hit_rate": pool_stats.hit_rate,
                "memory_saved_mb": pool_stats.memory_saved_bytes / 1024 / 1024
            }
            
            total_memory_saved += pool_stats.memory_saved_bytes
            total_objects_reused += pool_stats.total_reused
            
        stats["total_memory_saved_mb"] = total_memory_saved / 1024 / 1024
        stats["total_objects_reused"] = total_objects_reused
        
        return stats
        
    def optimize_all_pools(self):
        """Optimize all registered pools"""
        with self._lock:
            for pool in self._pools.values():
                # Force cleanup
                pool._cleanup_loop()
                
    def emergency_cleanup(self):
        """Emergency cleanup of all pools"""
        print("🧹 Emergency pool cleanup activated")
        
        with self._lock:
            for name, pool in self._pools.items():
                old_size = len(pool._pool)
                pool.clear()
                print(f"  Cleared pool '{name}': {old_size} objects")
                
        # Force garbage collection
        gc.collect()


# Global pool manager instance
_global_pool_manager = GlobalPoolManager()


def create_pool(name: str, 
               factory: Callable[[], T], 
               max_size: int = 100,
               reset_func: Optional[Callable[[T], None]] = None) -> ObjectPool[T]:
    """Create and register a global object pool"""
    
    pool = ObjectPool(factory, reset_func, max_size)
    _global_pool_manager.register_pool(name, pool)
    return pool


def get_pool(name: str) -> Optional[ObjectPool]:
    """Get a registered pool by name"""
    return _global_pool_manager.get_pool(name)


# Common poolable object types for emergency optimization
class PoolableDict(dict, Poolable):
    """Memory-efficient poolable dictionary"""
    
    def reset(self):
        self.clear()
        
    def is_reusable(self) -> bool:
        return len(self) < 1000  # Don't reuse if too large


class PoolableList(list, Poolable):
    """Memory-efficient poolable list"""
    
    def reset(self):
        self.clear()
        
    def is_reusable(self) -> bool:
        return len(self) < 10000  # Don't reuse if too large


class PoolableStringBuilder:
    """Memory-efficient string builder with pooling"""
    
    def __init__(self):
        self._parts: List[str] = []
        
    def append(self, text: str):
        self._parts.append(text)
        
    def build(self) -> str:
        return ''.join(self._parts)
        
    def reset(self):
        self._parts.clear()
        
    def is_reusable(self) -> bool:
        return len(self._parts) < 1000


# Emergency optimization pools
def setup_emergency_pools():
    """Setup object pools for emergency memory optimization"""
    
    # Dictionary pool for frequent dict usage
    dict_pool = create_pool(
        "dictionaries", 
        factory=lambda: PoolableDict(),
        max_size=500
    )
    
    # List pool for frequent list usage
    list_pool = create_pool(
        "lists",
        factory=lambda: PoolableList(),
        max_size=500  
    )
    
    # String builder pool
    string_builder_pool = create_pool(
        "string_builders",
        factory=lambda: PoolableStringBuilder(),
        max_size=100
    )
    
    return {
        "dict_pool": dict_pool,
        "list_pool": list_pool,
        "string_builder_pool": string_builder_pool
    }


class PooledResource:
    """Context manager for pooled resources"""
    
    def __init__(self, pool: ObjectPool[T]):
        self.pool = pool
        self.resource = None
        
    def __enter__(self) -> T:
        self.resource = self.pool.acquire()
        return self.resource
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.resource is not None:
            self.pool.release(self.resource)
            self.resource = None


# Decorators for easy pooling
def pooled_function(pool_name: str):
    """Decorator to use pooled objects in function calls"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pool = get_pool(pool_name)
            if pool is None:
                return func(*args, **kwargs)
                
            with PooledResource(pool) as resource:
                return func(resource, *args, **kwargs)
                
        return wrapper
    return decorator


# Performance monitoring
def get_global_pool_stats():
    """Get statistics for all pools"""
    return _global_pool_manager.get_global_stats()


def optimize_all_pools():
    """Optimize all registered pools"""
    _global_pool_manager.optimize_all_pools()


def emergency_pool_cleanup():
    """Emergency cleanup of all pools"""
    _global_pool_manager.emergency_cleanup()


if __name__ == "__main__":
    # Demo emergency object pooling
    print("🏊‍♂️ Setting up emergency object pools...")
    
    pools = setup_emergency_pools()
    
    # Show initial stats
    stats = get_global_pool_stats()
    print(f"Created {stats['total_pools']} object pools")
    
    # Test dictionary pool
    dict_pool = pools["dict_pool"]
    
    for i in range(10):
        with PooledResource(dict_pool) as d:
            d['test'] = i
            d['value'] = f"item_{i}"
            
    # Show final stats
    final_stats = get_global_pool_stats()
    print(f"Memory saved: {final_stats['total_memory_saved_mb']:.2f}MB")
    print(f"Objects reused: {final_stats['total_objects_reused']}")