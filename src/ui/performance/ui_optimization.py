#!/usr/bin/env python3
"""
UI Performance Optimization - Rendering optimization, virtual scrolling, caching
"""

from __future__ import annotations

import asyncio
import time
import weakref
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Set
from dataclasses import dataclass
from enum import Enum
from collections import deque
import psutil
import gc

from textual import events
from textual.app import App
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive


class PerformanceMode(Enum):
    """Performance optimization modes"""
    HIGH_PERFORMANCE = "high_performance"
    BALANCED = "balanced"
    POWER_SAVING = "power_saving"
    ACCESSIBILITY = "accessibility"


class OptimizationLevel(Enum):
    """Optimization levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    AGGRESSIVE = 4


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    render_time_ms: float = 0.0
    layout_time_ms: float = 0.0
    paint_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    frame_rate: float = 0.0
    widget_count: int = 0
    update_count: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class OptimizationSettings:
    """Performance optimization settings"""
    # Rendering optimizations
    lazy_rendering: bool = True
    virtual_scrolling: bool = True
    render_caching: bool = True
    diff_based_updates: bool = True
    
    # Memory optimizations
    widget_pooling: bool = True
    garbage_collection: bool = True
    memory_limit_mb: int = 256
    
    # Update optimizations
    batch_updates: bool = True
    throttle_updates: bool = True
    max_fps: int = 60
    
    # Quality trade-offs
    reduce_animations: bool = False
    simplify_rendering: bool = False
    disable_effects: bool = False


class VirtualScrollContainer:
    """Virtual scrolling container for large lists"""
    
    def __init__(
        self,
        item_height: int,
        visible_items: int,
        total_items: int,
        item_renderer: Callable[[int], Widget]
    ):
        self.item_height = item_height
        self.visible_items = visible_items
        self.total_items = total_items
        self.item_renderer = item_renderer
        
        self.scroll_offset = 0
        self.rendered_widgets: Dict[int, Widget] = {}
        self.widget_pool: List[Widget] = []
        
    def get_visible_range(self) -> Tuple[int, int]:
        """Get range of visible items"""
        start_index = max(0, self.scroll_offset // self.item_height)
        end_index = min(self.total_items, start_index + self.visible_items + 2)  # +2 for buffer
        return start_index, end_index
    
    def update_scroll(self, offset: int) -> List[Widget]:
        """Update scroll position and return widgets to render"""
        self.scroll_offset = offset
        start_index, end_index = self.get_visible_range()
        
        # Recycle widgets not in visible range
        for index in list(self.rendered_widgets.keys()):
            if index < start_index or index >= end_index:
                widget = self.rendered_widgets.pop(index)
                self.widget_pool.append(widget)
        
        # Create widgets for visible range
        widgets_to_render = []
        for index in range(start_index, end_index):
            if index not in self.rendered_widgets:
                # Try to reuse from pool
                if self.widget_pool:
                    widget = self.widget_pool.pop()
                    # Update widget content for new index
                    self._update_widget_content(widget, index)
                else:
                    # Create new widget
                    widget = self.item_renderer(index)
                
                self.rendered_widgets[index] = widget
            
            widgets_to_render.append(self.rendered_widgets[index])
        
        return widgets_to_render
    
    def _update_widget_content(self, widget: Widget, index: int) -> None:
        """Update widget content for new index"""
        # This would update the widget's content based on the new index
        # Implementation depends on specific widget type
        pass


class RenderCache:
    """Cache for rendered widget content"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order = deque()
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached render result"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hit_count += 1
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Cache render result"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def invalidate(self, key: str) -> None:
        """Invalidate cached entry"""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
    
    def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'usage': len(self.cache) / self.max_size
        }


class UpdateBatcher:
    """Batches widget updates to reduce redraws"""
    
    def __init__(self, batch_delay: float = 0.016):  # ~60fps
        self.batch_delay = batch_delay
        self.pending_updates: Set[Widget] = set()
        self.batch_task: Optional[asyncio.Task] = None
        
    def schedule_update(self, widget: Widget) -> None:
        """Schedule widget for batch update"""
        self.pending_updates.add(widget)
        
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._process_batch())
    
    async def _process_batch(self) -> None:
        """Process batched updates"""
        await asyncio.sleep(self.batch_delay)
        
        if self.pending_updates:
            # Update all widgets in batch
            widgets_to_update = list(self.pending_updates)
            self.pending_updates.clear()
            
            for widget in widgets_to_update:
                if widget and hasattr(widget, 'refresh'):
                    try:
                        widget.refresh()
                    except Exception as e:
                        print(f"Error updating widget: {e}")


class MemoryManager:
    """Manages memory usage and cleanup"""
    
    def __init__(self, memory_limit_mb: int = 256):
        self.memory_limit_mb = memory_limit_mb
        self.widget_references: weakref.WeakSet = weakref.WeakSet()
        self.cleanup_threshold = 0.8  # Cleanup when 80% of limit reached
        
    def register_widget(self, widget: Widget) -> None:
        """Register widget for memory monitoring"""
        self.widget_references.add(widget)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is high"""
        current_usage = self.get_memory_usage()
        return current_usage > (self.memory_limit_mb * self.cleanup_threshold)
    
    def cleanup_memory(self) -> int:
        """Perform memory cleanup and return bytes freed"""
        initial_usage = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clean up dead widget references
        dead_widgets = []
        for widget_ref in list(self.widget_references):
            try:
                widget = widget_ref()
                if widget is None:
                    dead_widgets.append(widget_ref)
            except:
                dead_widgets.append(widget_ref)
        
        for dead_ref in dead_widgets:
            self.widget_references.discard(dead_ref)
        
        final_usage = self.get_memory_usage()
        bytes_freed = max(0, initial_usage - final_usage)
        
        return int(bytes_freed * 1024 * 1024)  # Convert to bytes
    
    async def monitor_memory(self) -> None:
        """Continuously monitor memory usage"""
        while True:
            if self.check_memory_pressure():
                freed = self.cleanup_memory()
                print(f"Memory cleanup freed {freed} bytes")
            
            await asyncio.sleep(5)  # Check every 5 seconds


class PerformanceOptimizer:
    """Main performance optimization manager"""
    
    def __init__(self, app: App):
        self.app = app
        self.settings = OptimizationSettings()
        self.metrics_history: List[PerformanceMetrics] = []
        self.render_cache = RenderCache()
        self.update_batcher = UpdateBatcher()
        self.memory_manager = MemoryManager()
        
        # Performance tracking
        self.frame_times: deque = deque(maxlen=60)  # Last 60 frame times
        self.optimization_level = OptimizationLevel.MEDIUM
        
        # Setup optimizations
        self._setup_optimizations()
        
        # Start monitoring
        asyncio.create_task(self._monitor_performance())
        asyncio.create_task(self.memory_manager.monitor_memory())
    
    def _setup_optimizations(self) -> None:
        """Setup performance optimizations"""
        # Override app's refresh method for batching
        if self.settings.batch_updates:
            original_refresh = self.app.refresh
            
            def batched_refresh(*args, **kwargs):
                # Schedule for batch update instead of immediate
                self.update_batcher.schedule_update(self.app)
            
            # Would need proper method to override app refresh
        
        # Setup render caching
        if self.settings.render_caching:
            self._setup_render_caching()
        
        # Setup memory monitoring
        if self.settings.garbage_collection:
            self._setup_memory_monitoring()
    
    def _setup_render_caching(self) -> None:
        """Setup render result caching"""
        # Would override widget render methods to use caching
        pass
    
    def _setup_memory_monitoring(self) -> None:
        """Setup memory usage monitoring"""
        self.memory_manager.memory_limit_mb = self.settings.memory_limit_mb
    
    def optimize_widget(self, widget: Widget) -> None:
        """Apply optimizations to a specific widget"""
        # Register for memory monitoring
        self.memory_manager.register_widget(widget)
        
        # Apply caching if enabled
        if self.settings.render_caching:
            self._apply_render_caching(widget)
        
        # Apply virtual scrolling for lists
        if (self.settings.virtual_scrolling and 
            'list' in type(widget).__name__.lower()):
            self._apply_virtual_scrolling(widget)
        
        # Apply lazy rendering
        if self.settings.lazy_rendering:
            self._apply_lazy_rendering(widget)
    
    def _apply_render_caching(self, widget: Widget) -> None:
        """Apply render caching to widget"""
        original_render = getattr(widget, 'render', None)
        if not original_render:
            return
        
        def cached_render(*args, **kwargs):
            # Create cache key based on widget state
            cache_key = self._generate_cache_key(widget)
            
            # Check cache first
            cached_result = self.render_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Render and cache result
            result = original_render(*args, **kwargs)
            self.render_cache.put(cache_key, result)
            return result
        
        # Replace render method
        widget.render = cached_render
    
    def _generate_cache_key(self, widget: Widget) -> str:
        """Generate cache key for widget"""
        # Include widget type, id, and relevant state
        widget_type = type(widget).__name__
        widget_id = getattr(widget, 'id', 'no_id')
        
        # Include relevant state properties
        state_items = []
        for attr in ['value', 'text', 'disabled', 'visible']:
            if hasattr(widget, attr):
                state_items.append(f"{attr}:{getattr(widget, attr)}")
        
        state_str = ",".join(state_items)
        return f"{widget_type}:{widget_id}:{hash(state_str)}"
    
    def _apply_virtual_scrolling(self, widget: Widget) -> None:
        """Apply virtual scrolling optimization"""
        # This would replace list rendering with virtual scrolling
        if hasattr(widget, 'items') and len(widget.items) > 100:
            # Large list - apply virtual scrolling
            pass
    
    def _apply_lazy_rendering(self, widget: Widget) -> None:
        """Apply lazy rendering optimization"""
        # Only render widgets when they become visible
        original_render = getattr(widget, 'render', None)
        if not original_render:
            return
        
        def lazy_render(*args, **kwargs):
            # Check if widget is visible
            if not self._is_widget_visible(widget):
                return None  # Don't render invisible widgets
            
            return original_render(*args, **kwargs)
        
        widget.render = lazy_render
    
    def _is_widget_visible(self, widget: Widget) -> bool:
        """Check if widget is currently visible"""
        # This would check if widget is in viewport
        # For now, return True (always render)
        return getattr(widget, 'display', True)
    
    def record_metrics(self, render_time: float) -> None:
        """Record performance metrics"""
        metrics = PerformanceMetrics(
            render_time_ms=render_time * 1000,
            memory_usage_mb=self.memory_manager.get_memory_usage(),
            cpu_usage_percent=psutil.cpu_percent(),
            widget_count=len(self.memory_manager.widget_references),
            frame_rate=self._calculate_frame_rate()
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Record frame time
        self.frame_times.append(render_time)
        
        # Adjust optimization level if needed
        self._adjust_optimization_level(metrics)
    
    def _calculate_frame_rate(self) -> float:
        """Calculate current frame rate"""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def _adjust_optimization_level(self, metrics: PerformanceMetrics) -> None:
        """Adjust optimization level based on performance"""
        # Increase optimization if performance is poor
        if metrics.frame_rate < 30 or metrics.memory_usage_mb > self.settings.memory_limit_mb:
            if self.optimization_level.value < OptimizationLevel.AGGRESSIVE.value:
                self.optimization_level = OptimizationLevel(self.optimization_level.value + 1)
                self._apply_optimization_level()
        
        # Decrease optimization if performance is good
        elif metrics.frame_rate > 50 and metrics.memory_usage_mb < self.settings.memory_limit_mb * 0.5:
            if self.optimization_level.value > OptimizationLevel.LOW.value:
                self.optimization_level = OptimizationLevel(self.optimization_level.value - 1)
                self._apply_optimization_level()
    
    def _apply_optimization_level(self) -> None:
        """Apply current optimization level settings"""
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            # Most aggressive optimizations
            self.settings.reduce_animations = True
            self.settings.simplify_rendering = True
            self.settings.max_fps = 30
            self.render_cache.max_size = 2000
        
        elif self.optimization_level == OptimizationLevel.HIGH:
            # High optimizations
            self.settings.reduce_animations = True
            self.settings.max_fps = 45
            self.render_cache.max_size = 1500
        
        elif self.optimization_level == OptimizationLevel.MEDIUM:
            # Balanced optimizations
            self.settings.reduce_animations = False
            self.settings.max_fps = 60
            self.render_cache.max_size = 1000
        
        elif self.optimization_level == OptimizationLevel.LOW:
            # Minimal optimizations
            self.settings.simplify_rendering = False
            self.settings.max_fps = 60
            self.render_cache.max_size = 500
        
        else:  # NONE
            # No optimizations
            self.settings.reduce_animations = False
            self.settings.simplify_rendering = False
            self.settings.disable_effects = False
    
    async def _monitor_performance(self) -> None:
        """Monitor performance continuously"""
        while True:
            start_time = time.time()
            
            # Simulate frame rendering time
            await asyncio.sleep(1/60)  # 60 FPS target
            
            render_time = time.time() - start_time
            self.record_metrics(render_time)
            
            await asyncio.sleep(1)  # Check every second
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-60:]  # Last 60 records
        
        avg_render_time = sum(m.render_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_fps = sum(m.frame_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            "optimization_level": self.optimization_level.name,
            "average_render_time_ms": avg_render_time,
            "average_memory_mb": avg_memory,
            "average_fps": avg_fps,
            "cache_stats": self.render_cache.get_stats(),
            "widget_count": len(self.memory_manager.widget_references),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        latest = self.metrics_history[-1]
        
        if latest.frame_rate < 30:
            recommendations.append("Frame rate is low - consider enabling more optimizations")
        
        if latest.memory_usage_mb > self.settings.memory_limit_mb * 0.8:
            recommendations.append("Memory usage is high - consider reducing widget count or enabling virtual scrolling")
        
        cache_stats = self.render_cache.get_stats()
        if cache_stats['hit_rate'] < 0.5:
            recommendations.append("Render cache hit rate is low - consider optimizing widget state changes")
        
        if latest.widget_count > 1000:
            recommendations.append("High widget count detected - consider using virtual scrolling for lists")
        
        return recommendations
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force performance cleanup"""
        # Clear render cache
        cache_size = len(self.render_cache.cache)
        self.render_cache.clear()
        
        # Memory cleanup
        bytes_freed = self.memory_manager.cleanup_memory()
        
        # Clear old metrics
        metrics_cleared = max(0, len(self.metrics_history) - 100)
        self.metrics_history = self.metrics_history[-100:]
        
        return {
            "cache_entries_cleared": cache_size,
            "memory_freed_bytes": bytes_freed,
            "metrics_cleared": metrics_cleared,
            "optimization_level_reset": self.optimization_level.name
        }


# Message classes
class PerformanceAlert(Message):
    """Message sent when performance issues detected"""
    
    def __init__(self, alert_type: str, metrics: PerformanceMetrics) -> None:
        super().__init__()
        self.alert_type = alert_type
        self.metrics = metrics


class OptimizationLevelChanged(Message):
    """Message sent when optimization level changes"""
    
    def __init__(self, level: OptimizationLevel) -> None:
        super().__init__()
        self.level = level


# Global performance optimizer
_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer(app: App) -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None or _performance_optimizer.app != app:
        _performance_optimizer = PerformanceOptimizer(app)
    return _performance_optimizer

def setup_performance_optimization(app: App) -> PerformanceOptimizer:
    """Setup performance optimization for an app"""
    optimizer = get_performance_optimizer(app)
    
    # Apply optimizations to existing widgets
    for widget in app.query("*"):
        optimizer.optimize_widget(widget)
    
    return optimizer

def optimize_widget(widget: Widget, app: Optional[App] = None) -> None:
    """Apply optimizations to a specific widget"""
    if app:
        optimizer = get_performance_optimizer(app)
        optimizer.optimize_widget(widget)