#!/usr/bin/env python3
"""
Widget Memory Manager - Optimized TUI Widget Memory Management
Implements aggressive memory optimization for TUI widgets to maintain <1.5Gi target
"""

import gc
import weakref
import threading
import time
import sys
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import tracemalloc

# Import optimization modules with fallbacks
try:
    from .object_pool import ObjectPool, create_pool, get_pool, Poolable
    from .memory_profiler import MemoryProfiler
except ImportError:
    # Minimal fallbacks
    class ObjectPool:
        def __init__(self, *args, **kwargs): pass
        def acquire(self): return {}
        def release(self, obj): return True
        def get_stats(self): return {}
    
    class Poolable:
        def reset(self): pass
        def is_reusable(self): return True
    
    def create_pool(*args, **kwargs): return ObjectPool()
    def get_pool(name): return ObjectPool()
    
    class MemoryProfiler:
        def __init__(self, *args, **kwargs): pass
        def take_snapshot(self):
            import psutil, os
            class Snapshot:
                def __init__(self):
                    try:
                        process = psutil.Process(os.getpid())
                        self.process_memory = process.memory_info().rss
                    except:
                        self.process_memory = 0
            return Snapshot()


@dataclass
class WidgetMemoryStats:
    """Widget memory usage statistics"""
    widget_count: int = 0
    total_memory_mb: float = 0.0
    pooled_objects_count: int = 0
    cached_renders_count: int = 0
    memory_savings_mb: float = 0.0
    gc_collections: int = 0
    weak_refs_active: int = 0
    largest_widgets: List[Tuple[str, float]] = field(default_factory=list)


class PoolableWidgetState(dict, Poolable):
    """Memory-efficient poolable widget state"""
    
    def reset(self):
        """Reset state for reuse"""
        self.clear()
        self.update({
            'visible': True,
            'enabled': True,
            'focused': False,
            'dirty': False,
            'children': [],
            'properties': {}
        })
    
    def is_reusable(self) -> bool:
        """Check if state is safe to reuse"""
        # Don't reuse if too much data accumulated
        total_size = sum(sys.getsizeof(v, 0) for v in self.values())
        return total_size < 10240  # < 10KB


class PoolableWidgetProperties(dict, Poolable):
    """Memory-efficient poolable widget properties"""
    
    def reset(self):
        """Reset properties for reuse"""
        self.clear()
        self.update({
            'text': '',
            'style': {},
            'position': (0, 0),
            'size': (100, 30),
            'padding': (0, 0, 0, 0),
            'margin': (0, 0, 0, 0)
        })
    
    def is_reusable(self) -> bool:
        """Check if properties are safe to reuse"""
        # Check for complex nested structures
        for value in self.values():
            if isinstance(value, (list, dict)) and len(str(value)) > 1000:
                return False
        return True


class RenderCache:
    """Efficient render result caching with automatic cleanup"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 30.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (result, timestamp)
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached render result"""
        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                
                # Check TTL
                if time.time() - timestamp <= self.ttl_seconds:
                    return result
                else:
                    # Expired, remove
                    del self._cache[key]
            
            return None
    
    def put(self, key: str, result: Any):
        """Cache render result"""
        with self._lock:
            # Cleanup if at capacity
            if len(self._cache) >= self.max_size:
                self._cleanup_expired()
                
                # If still at capacity, remove oldest
                if len(self._cache) >= self.max_size:
                    oldest_key = min(self._cache.keys(), 
                                   key=lambda k: self._cache[k][1])
                    del self._cache[oldest_key]
            
            self._cache[key] = (result, time.time())
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for _, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl_seconds
            )
            
            return {
                'total_entries': len(self._cache),
                'expired_entries': expired_count,
                'active_entries': len(self._cache) - expired_count,
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hit_rate', 0.0)
            }


class WidgetMemoryManager:
    """
    Comprehensive widget memory management system
    Targets <1.5Gi memory usage through aggressive optimization
    """
    
    def __init__(self, target_memory_mb: float = 1536.0):  # 1.5GB in MB
        self.target_memory_mb = target_memory_mb
        self.profiler = MemoryProfiler(target_memory_mb=int(target_memory_mb))
        
        # Widget tracking - use simple dict instead of WeakSet for compatibility
        self.active_widgets: Set[int] = set()  # Track widget IDs instead
        self.widget_registry: Dict[str, weakref.ref] = {}
        self.widget_memory_map: Dict[str, float] = {}  # widget_id -> memory_mb
        
        # Object pools for widget components
        self.state_pool: Optional[ObjectPool] = None
        self.props_pool: Optional[ObjectPool] = None
        self.event_pool: Optional[ObjectPool] = None
        
        # Render caching
        self.render_cache = RenderCache(max_size=500, ttl_seconds=10.0)
        
        # Memory optimization settings
        self.aggressive_cleanup = True
        self.auto_gc_threshold = 100  # MB increase before triggering GC
        self.last_memory_check = 0.0
        
        # Statistics tracking
        self.stats = WidgetMemoryStats()
        
        # Background cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_active = False
        
        self._initialize_pools()
        self._start_background_cleanup()
    
    def _initialize_pools(self):
        """Initialize object pools for widget components"""
        try:
            # Widget state pool
            self.state_pool = create_pool(
                "widget_states",
                factory=lambda: PoolableWidgetState(),
                max_size=200
            )
            
            # Widget properties pool
            self.props_pool = create_pool(
                "widget_properties", 
                factory=lambda: PoolableWidgetProperties(),
                max_size=300
            )
            
            # Event data pool (for widget events)
            self.event_pool = create_pool(
                "widget_events",
                factory=lambda: {'type': '', 'data': {}, 'timestamp': 0, 'handled': False},
                max_size=100,
                reset_func=lambda e: e.update({'type': '', 'data': {}, 'timestamp': 0, 'handled': False})
            )
            
            print("🏊‍♂️ Widget object pools initialized")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize widget pools: {e}")
            # Use minimal fallback pools
            self.state_pool = ObjectPool()
            self.props_pool = ObjectPool()
            self.event_pool = ObjectPool()
    
    def register_widget(self, widget_id: str, widget_obj: Any) -> bool:
        """Register a widget for memory management"""
        try:
            # Create weak reference to avoid keeping widget alive
            widget_ref = weakref.ref(widget_obj, self._widget_cleanup_callback(widget_id))
            self.widget_registry[widget_id] = widget_ref
            
            # Track widget ID
            try:
                self.active_widgets.add(id(widget_obj))
            except TypeError:
                # Object not compatible, use string ID
                self.active_widgets.add(hash(widget_id))
            
            # Estimate memory usage
            try:
                widget_memory = self._estimate_widget_memory(widget_obj)
                self.widget_memory_map[widget_id] = widget_memory
                self.stats.total_memory_mb += widget_memory
            except Exception:
                self.widget_memory_map[widget_id] = 0.1  # Default estimate
            
            self.stats.widget_count += 1
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to register widget {widget_id}: {e}")
            return False
    
    def _widget_cleanup_callback(self, widget_id: str):
        """Create cleanup callback for when widget is garbage collected"""
        def cleanup_callback(ref):
            # Clean up widget tracking data
            if widget_id in self.widget_registry:
                del self.widget_registry[widget_id]
            
            if widget_id in self.widget_memory_map:
                memory_freed = self.widget_memory_map[widget_id]
                self.stats.total_memory_mb -= memory_freed
                self.stats.memory_savings_mb += memory_freed
                del self.widget_memory_map[widget_id]
            
            self.stats.widget_count -= 1
            
        return cleanup_callback
    
    def _estimate_widget_memory(self, widget_obj: Any) -> float:
        """Estimate memory usage of a widget object"""
        try:
            # Basic size estimation
            base_size = sys.getsizeof(widget_obj, 0)
            
            # Add estimated size of attributes
            total_size = base_size
            
            if hasattr(widget_obj, '__dict__'):
                for attr_name, attr_value in widget_obj.__dict__.items():
                    try:
                        attr_size = sys.getsizeof(attr_value, 0)
                        
                        # Special handling for collections
                        if isinstance(attr_value, (list, tuple)):
                            attr_size += sum(sys.getsizeof(item, 0) for item in attr_value[:100])  # Limit sample
                        elif isinstance(attr_value, dict):
                            attr_size += sum(sys.getsizeof(k, 0) + sys.getsizeof(v, 0) 
                                           for k, v in list(attr_value.items())[:100])
                        
                        total_size += attr_size
                        
                    except (TypeError, ReferenceError):
                        total_size += 100  # Default estimate for problematic attributes
            
            return total_size / 1024 / 1024  # Convert to MB
            
        except Exception:
            return 0.5  # Default 500KB estimate
    
    def get_pooled_state(self) -> Dict[str, Any]:
        """Get a pooled widget state object"""
        try:
            if self.state_pool:
                state = self.state_pool.acquire()
                self.stats.pooled_objects_count += 1
                return state
        except Exception:
            pass
        
        # Fallback to new object
        return PoolableWidgetState()
    
    def release_pooled_state(self, state: Dict[str, Any]) -> bool:
        """Release a widget state back to the pool"""
        try:
            if self.state_pool and isinstance(state, (PoolableWidgetState, dict)):
                return self.state_pool.release(state)
        except Exception:
            pass
        
        return False
    
    def get_pooled_properties(self) -> Dict[str, Any]:
        """Get pooled widget properties object"""
        try:
            if self.props_pool:
                props = self.props_pool.acquire()
                self.stats.pooled_objects_count += 1
                return props
        except Exception:
            pass
        
        return PoolableWidgetProperties()
    
    def release_pooled_properties(self, props: Dict[str, Any]) -> bool:
        """Release widget properties back to the pool"""
        try:
            if self.props_pool and isinstance(props, (PoolableWidgetProperties, dict)):
                return self.props_pool.release(props)
        except Exception:
            pass
        
        return False
    
    def cache_render(self, widget_id: str, render_key: str, render_result: Any):
        """Cache a widget render result"""
        cache_key = f"{widget_id}:{render_key}"
        self.render_cache.put(cache_key, render_result)
        self.stats.cached_renders_count += 1
    
    def get_cached_render(self, widget_id: str, render_key: str) -> Optional[Any]:
        """Get cached render result"""
        cache_key = f"{widget_id}:{render_key}"
        result = self.render_cache.get(cache_key)
        return result
    
    def invalidate_widget_cache(self, widget_id: str):
        """Invalidate all cached renders for a widget"""
        # For efficiency, we'll clear the entire cache
        # In production, you'd want more granular cache invalidation
        self.render_cache.clear()
    
    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """Optimize widget memory usage"""
        start_snapshot = self.profiler.take_snapshot()
        start_mb = start_snapshot.process_memory / 1024 / 1024
        
        optimization_actions = []
        
        # Check if optimization is needed
        if not force and start_mb < self.target_memory_mb * 0.8:  # Only optimize if > 80% of target
            return {
                "optimization_needed": False,
                "current_memory_mb": start_mb,
                "target_memory_mb": self.target_memory_mb,
                "actions_taken": []
            }
        
        print(f"🔧 Optimizing widget memory (current: {start_mb:.1f}MB, target: {self.target_memory_mb:.1f}MB)")
        
        # 1. Clean up dead widget references
        dead_widgets = self._cleanup_dead_widgets()
        if dead_widgets > 0:
            optimization_actions.append(f"Cleaned {dead_widgets} dead widget references")
        
        # 2. Clear render cache
        cache_stats = self.render_cache.get_stats()
        if cache_stats['total_entries'] > 100:
            self.render_cache.clear()
            optimization_actions.append(f"Cleared render cache ({cache_stats['total_entries']} entries)")
        
        # 3. Optimize object pools
        if self.state_pool:
            try:
                state_stats = self.state_pool.get_stats()
                if state_stats.current_pool_size > 50:  # Only clean if pool has many objects
                    self.state_pool.clear()
                    optimization_actions.append("Cleared state object pool")
            except Exception:
                pass
        
        # 4. Force garbage collection
        gc_before = len(gc.get_objects())
        for _ in range(3):
            collected = gc.collect()
            self.stats.gc_collections += 1
        gc_after = len(gc.get_objects())
        
        if gc_before > gc_after:
            objects_freed = gc_before - gc_after
            optimization_actions.append(f"GC freed {objects_freed:,} objects")
        
        # 5. Clean up large widgets
        large_widgets = self._identify_large_widgets()
        if large_widgets:
            freed_memory = self._cleanup_large_widgets(large_widgets)
            if freed_memory > 0:
                optimization_actions.append(f"Cleaned large widgets: {freed_memory:.1f}MB freed")
        
        # Final measurement
        end_snapshot = self.profiler.take_snapshot()
        end_mb = end_snapshot.process_memory / 1024 / 1024
        memory_reduction = start_mb - end_mb
        
        self.stats.memory_savings_mb += max(0, memory_reduction)
        
        return {
            "optimization_needed": True,
            "current_memory_mb": end_mb,
            "target_memory_mb": self.target_memory_mb,
            "memory_reduction_mb": memory_reduction,
            "target_achieved": end_mb <= self.target_memory_mb,
            "actions_taken": optimization_actions,
            "widgets_active": len(self.widget_registry),
            "memory_freed_mb": max(0, memory_reduction)
        }
    
    def _cleanup_dead_widgets(self) -> int:
        """Clean up references to dead widgets"""
        dead_count = 0
        dead_widget_ids = []
        
        for widget_id, widget_ref in list(self.widget_registry.items()):
            if widget_ref() is None:  # Widget has been garbage collected
                dead_widget_ids.append(widget_id)
        
        for widget_id in dead_widget_ids:
            if widget_id in self.widget_registry:
                del self.widget_registry[widget_id]
                dead_count += 1
            
            if widget_id in self.widget_memory_map:
                freed_memory = self.widget_memory_map[widget_id]
                self.stats.total_memory_mb -= freed_memory
                del self.widget_memory_map[widget_id]
        
        return dead_count
    
    def _identify_large_widgets(self) -> List[Tuple[str, float]]:
        """Identify widgets consuming significant memory"""
        large_widgets = []
        
        for widget_id, memory_mb in self.widget_memory_map.items():
            if memory_mb > 5.0:  # Widgets > 5MB
                large_widgets.append((widget_id, memory_mb))
        
        # Sort by memory usage
        large_widgets.sort(key=lambda x: x[1], reverse=True)
        
        return large_widgets[:10]  # Return top 10
    
    def _cleanup_large_widgets(self, large_widgets: List[Tuple[str, float]]) -> float:
        """Clean up large widgets if possible"""
        memory_freed = 0.0
        
        for widget_id, memory_mb in large_widgets:
            # Check if widget is still alive
            widget_ref = self.widget_registry.get(widget_id)
            if widget_ref and widget_ref() is not None:
                widget_obj = widget_ref()
                
                # Try to optimize the widget
                try:
                    # Clear large attributes if possible
                    if hasattr(widget_obj, '__dict__'):
                        for attr_name, attr_value in list(widget_obj.__dict__.items()):
                            if isinstance(attr_value, list) and len(attr_value) > 1000:
                                # Clear large lists
                                widget_obj.__dict__[attr_name] = []
                                memory_freed += 0.1  # Estimate
                            elif isinstance(attr_value, dict) and len(attr_value) > 500:
                                # Clear large dicts
                                widget_obj.__dict__[attr_name] = {}
                                memory_freed += 0.1  # Estimate
                except Exception:
                    pass  # Skip if we can't modify
        
        return memory_freed
    
    def _start_background_cleanup(self):
        """Start background cleanup thread"""
        if not self._cleanup_active:
            self._cleanup_active = True
            
            def cleanup_loop():
                while self._cleanup_active:
                    try:
                        # Check memory every 30 seconds
                        time.sleep(30.0)
                        
                        if not self._cleanup_active:
                            break
                        
                        # Take memory snapshot
                        snapshot = self.profiler.take_snapshot()
                        current_mb = snapshot.process_memory / 1024 / 1024
                        
                        # Auto-optimization if memory exceeds threshold
                        if current_mb > self.target_memory_mb * 0.9:  # 90% of target
                            print(f"🧹 Auto-optimization triggered: {current_mb:.1f}MB")
                            self.optimize_memory(force=False)
                        
                        # Regular maintenance
                        self._cleanup_dead_widgets()
                        self.render_cache._cleanup_expired()
                        
                    except Exception as e:
                        print(f"Background cleanup error: {e}")
                        time.sleep(60.0)  # Wait longer on error
            
            self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def get_memory_stats(self) -> WidgetMemoryStats:
        """Get current memory statistics"""
        # Update current stats
        current_snapshot = self.profiler.take_snapshot()
        current_mb = current_snapshot.process_memory / 1024 / 1024
        
        # Update largest widgets
        largest_widgets = sorted(
            self.widget_memory_map.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        self.stats.largest_widgets = [(wid, mem) for wid, mem in largest_widgets]
        self.stats.weak_refs_active = len(self.active_widgets)
        
        return self.stats
    
    def generate_memory_report(self) -> str:
        """Generate comprehensive memory report"""
        stats = self.get_memory_stats()
        current_snapshot = self.profiler.take_snapshot()
        current_mb = current_snapshot.process_memory / 1024 / 1024
        
        cache_stats = self.render_cache.get_stats()
        
        report = f"""
🖼️ WIDGET MEMORY MANAGER REPORT
{'='*50}

Current Status:
• Total Memory: {current_mb:.1f}MB
• Target Memory: {self.target_memory_mb:.1f}MB
• Status: {'✅ OPTIMAL' if current_mb <= self.target_memory_mb else '⚠️ ABOVE TARGET'}

Widget Statistics:
• Active Widgets: {stats.widget_count:,}
• Widget Memory: {stats.total_memory_mb:.1f}MB
• Memory Saved: {stats.memory_savings_mb:.1f}MB
• Pooled Objects: {stats.pooled_objects_count:,}

Optimization Systems:
• Render Cache: {cache_stats['active_entries']:,} entries
• GC Collections: {stats.gc_collections:,}
• Weak References: {stats.weak_refs_active:,}

Top Memory Consumers:
"""
        
        for i, (widget_id, memory_mb) in enumerate(stats.largest_widgets[:5]):
            report += f"  {i+1}. {widget_id}: {memory_mb:.2f}MB\n"
        
        if current_mb > self.target_memory_mb:
            report += f"\n⚡ Optimization Needed: {current_mb - self.target_memory_mb:.1f}MB reduction required\n"
        else:
            report += f"\n🎉 Memory target achieved! ({self.target_memory_mb - current_mb:.1f}MB under target)\n"
        
        return report
    
    def emergency_cleanup(self):
        """Emergency memory cleanup for critical situations"""
        print("🚨 EMERGENCY WIDGET MEMORY CLEANUP")
        
        # Aggressive cleanup
        self.render_cache.clear()
        
        # Clear all object pools
        if self.state_pool:
            self.state_pool.clear()
        if self.props_pool:
            self.props_pool.clear()
        if self.event_pool:
            self.event_pool.clear()
        
        # Clean up all widget tracking
        dead_count = self._cleanup_dead_widgets()
        
        # Force multiple GC cycles
        for _ in range(5):
            gc.collect()
            self.stats.gc_collections += 1
        
        print(f"  Cleaned {dead_count} dead widgets")
        print(f"  Cleared all caches and pools")
        print(f"  Forced garbage collection")
    
    def shutdown(self):
        """Shutdown the memory manager"""
        self._cleanup_active = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)
        
        self.emergency_cleanup()


# Global widget memory manager instance
_widget_memory_manager: Optional[WidgetMemoryManager] = None


def get_widget_memory_manager(target_memory_mb: float = 1536.0) -> WidgetMemoryManager:
    """Get the global widget memory manager instance"""
    global _widget_memory_manager
    
    if _widget_memory_manager is None:
        _widget_memory_manager = WidgetMemoryManager(target_memory_mb)
    
    return _widget_memory_manager


def optimize_widget_memory() -> Dict[str, Any]:
    """Quick widget memory optimization"""
    manager = get_widget_memory_manager()
    return manager.optimize_memory()


def emergency_widget_cleanup():
    """Emergency widget memory cleanup"""
    manager = get_widget_memory_manager()
    manager.emergency_cleanup()


# Context manager for automatic widget memory management
class ManagedWidget:
    """Context manager for automatic widget memory management"""
    
    def __init__(self, widget_id: str, widget_obj: Any):
        self.widget_id = widget_id
        self.widget_obj = widget_obj
        self.manager = get_widget_memory_manager()
        
    def __enter__(self):
        self.manager.register_widget(self.widget_id, self.widget_obj)
        return self.widget_obj
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Widget will be cleaned up automatically via weak references
        pass


if __name__ == "__main__":
    # Demo widget memory management
    print("🖼️ Widget Memory Manager - Demo")
    
    manager = WidgetMemoryManager(target_memory_mb=1536.0)
    
    # Create some mock widgets
    class MockWidget:
        def __init__(self, widget_id: str):
            self.id = widget_id
            self.properties = manager.get_pooled_properties()
            self.state = manager.get_pooled_state()
            self.render_cache = []
            
        def render(self):
            render_result = f"Rendered {self.id} at {time.time()}"
            manager.cache_render(self.id, "default", render_result)
            return render_result
    
    # Create and register widgets
    widgets = []
    for i in range(100):
        widget = MockWidget(f"widget_{i}")
        widgets.append(widget)
        manager.register_widget(widget.id, widget)
        widget.render()
    
    print(f"Created {len(widgets)} widgets")
    
    # Show stats
    stats = manager.get_memory_stats()
    print(f"Widget count: {stats.widget_count}")
    print(f"Total memory: {stats.total_memory_mb:.1f}MB")
    print(f"Pooled objects: {stats.pooled_objects_count}")
    
    # Optimize memory
    optimization_result = manager.optimize_memory(force=True)
    print(f"Optimization result: {optimization_result}")
    
    # Generate report
    print(manager.generate_memory_report())
    
    manager.shutdown()