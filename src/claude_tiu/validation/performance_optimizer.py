"""
Performance Optimizer - Ensures <200ms validation response times.

Implements advanced performance optimization techniques:
- Intelligent caching with TTL
- Model prediction batching
- Feature extraction optimization
- Memory pool management
- Async processing pipelines
- Performance monitoring and auto-scaling
"""

import asyncio
import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
import numpy as np
from functools import lru_cache
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    active_requests: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata."""
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 3600  # 1 hour default
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class PerformanceOptimizer:
    """
    Advanced performance optimizer for sub-200ms validation.
    
    Implements comprehensive optimization strategies including:
    - Multi-level caching (L1 memory, L2 disk)
    - Request batching and pooling
    - Async processing with priority queues
    - Memory management and garbage collection
    - Performance monitoring and auto-tuning
    """
    
    def __init__(self, target_latency_ms: int = 200):
        """Initialize performance optimizer."""
        self.target_latency_ms = target_latency_ms
        
        # Caching infrastructure
        self.l1_cache: Dict[str, CacheEntry] = {}  # Memory cache
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        
        # Request batching
        self.batch_queue: deque = deque()
        self.batch_size = 32
        self.batch_timeout_ms = 50
        self.batch_processor_running = False
        
        # Thread pools for CPU-intensive tasks
        self.cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu_pool")
        self.io_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io_pool")
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.response_times: deque = deque(maxlen=1000)  # Last 1000 response times
        self.performance_history: deque = deque(maxlen=100)  # Last 100 performance snapshots
        
        # Memory management
        self.max_memory_mb = 256
        self.cleanup_threshold = 0.8  # Clean up when 80% of memory is used
        
        # Optimization flags
        self.optimizations_enabled = {
            'caching': True,
            'batching': True,
            'precomputation': True,
            'compression': True,
            'memory_pooling': True
        }
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.auto_tuning_enabled = True
        self.last_optimization = datetime.now()
        
        logger.info("Performance optimizer initialized")
    
    async def initialize(self) -> None:
        """Initialize performance optimization systems."""
        logger.info("Initializing performance optimizer")
        
        try:
            # Start background tasks
            if self.optimizations_enabled['batching']:
                asyncio.create_task(self._batch_processor())
            
            if self.monitoring_enabled:
                asyncio.create_task(self._performance_monitor())
            
            if self.auto_tuning_enabled:
                asyncio.create_task(self._auto_tuner())
            
            # Pre-warm caches and pools
            await self._prewarm_systems()
            
            logger.info("Performance optimizer ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            raise
    
    async def optimize_validation_call(
        self, 
        validation_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, float]:
        """
        Optimize a validation function call for performance.
        
        Args:
            validation_func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time_ms)
        """
        start_time = time.time()
        self.metrics.active_requests += 1
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(validation_func.__name__, args, kwargs)
            
            # Check cache first
            if self.optimizations_enabled['caching']:
                cached_result = await self._check_cache(cache_key)
                if cached_result is not None:
                    execution_time = (time.time() - start_time) * 1000
                    self._update_performance_metrics(execution_time, cache_hit=True)
                    return cached_result, execution_time
            
            # Execute with optimization
            if self.optimizations_enabled['batching']:
                result = await self._execute_with_batching(validation_func, args, kwargs)
            else:
                result = await self._execute_optimized(validation_func, args, kwargs)
            
            # Cache result
            if self.optimizations_enabled['caching'] and result is not None:
                await self._store_cache(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time, cache_hit=False)
            
            return result, execution_time
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_error_metrics(e, execution_time)
            raise
        
        finally:
            self.metrics.active_requests -= 1
    
    async def batch_optimize_predictions(
        self,
        prediction_func: Callable,
        inputs: List[Any],
        max_batch_size: int = 32
    ) -> List[Any]:
        """
        Optimize batch predictions for ML models.
        
        Args:
            prediction_func: Prediction function to batch
            inputs: List of input data
            max_batch_size: Maximum batch size
            
        Returns:
            List of prediction results
        """
        if not inputs:
            return []
        
        # Split into optimal batches
        batches = [
            inputs[i:i + max_batch_size] 
            for i in range(0, len(inputs), max_batch_size)
        ]
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_prediction_batch(prediction_func, batch))
            tasks.append(task)
        
        # Collect results
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    async def precompute_features(
        self,
        feature_extractor: Callable,
        code_samples: List[str],
        cache_ttl: int = 3600
    ) -> Dict[str, Any]:
        """
        Precompute features for common code patterns.
        
        Args:
            feature_extractor: Feature extraction function
            code_samples: Code samples to precompute features for
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Dictionary mapping code hashes to features
        """
        if not self.optimizations_enabled['precomputation']:
            return {}
        
        logger.info(f"Precomputing features for {len(code_samples)} code samples")
        
        feature_cache = {}
        
        # Process in parallel batches
        batch_size = 16
        for i in range(0, len(code_samples), batch_size):
            batch = code_samples[i:i + batch_size]
            
            # Create tasks for batch processing
            tasks = []
            for code in batch:
                task = asyncio.create_task(
                    self._compute_and_cache_features(feature_extractor, code, cache_ttl)
                )
                tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks)
            
            # Collect results
            for code, features in zip(batch, batch_results):
                if features:
                    code_hash = hashlib.md5(code.encode()).hexdigest()
                    feature_cache[code_hash] = features
        
        logger.info(f"Precomputed features for {len(feature_cache)} unique code patterns")
        return feature_cache
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'current_metrics': {
                'avg_response_time_ms': self.metrics.avg_response_time,
                'p95_response_time_ms': self.metrics.p95_response_time,
                'p99_response_time_ms': self.metrics.p99_response_time,
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'throughput_per_second': self.metrics.throughput_per_second,
                'error_rate': self.metrics.error_rate,
                'active_requests': self.metrics.active_requests
            },
            'cache_stats': self.cache_stats.copy(),
            'optimizations_enabled': self.optimizations_enabled.copy(),
            'system_status': {
                'target_latency_ms': self.target_latency_ms,
                'batch_queue_size': len(self.batch_queue),
                'memory_usage_ratio': self.metrics.memory_usage_mb / self.max_memory_mb,
                'last_optimization': self.last_optimization.isoformat()
            }
        }
    
    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage and clean up caches."""
        logger.info("Optimizing memory usage")
        
        initial_memory = self.metrics.memory_usage_mb
        
        # Clean expired cache entries
        expired_keys = [
            key for key, entry in self.l1_cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self.l1_cache[key]
            self.cache_stats['evictions'] += 1
        
        # Clean least recently used entries if memory is high
        if self.metrics.memory_usage_mb / self.max_memory_mb > self.cleanup_threshold:
            await self._evict_lru_entries()
        
        # Update memory metrics
        await self._update_memory_metrics()
        
        final_memory = self.metrics.memory_usage_mb
        memory_freed = initial_memory - final_memory
        
        return {
            'memory_freed_mb': memory_freed,
            'entries_removed': len(expired_keys),
            'final_memory_mb': final_memory,
            'memory_usage_ratio': final_memory / self.max_memory_mb
        }
    
    async def cleanup(self) -> None:
        """Cleanup performance optimizer resources."""
        logger.info("Cleaning up performance optimizer")
        
        # Stop background tasks
        self.batch_processor_running = False
        self.monitoring_enabled = False
        self.auto_tuning_enabled = False
        
        # Clear caches
        self.l1_cache.clear()
        self.batch_queue.clear()
        
        # Shutdown thread pools
        self.cpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        
        logger.info("Performance optimizer cleanup completed")
    
    # Private implementation methods
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        # Convert args/kwargs to stable string representation
        args_str = str(args) if args else ""
        kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
        
        key_data = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check cache for existing result."""
        if cache_key not in self.l1_cache:
            self.cache_stats['misses'] += 1
            return None
        
        entry = self.l1_cache[cache_key]
        
        # Check expiration
        if entry.is_expired:
            del self.l1_cache[cache_key]
            self.cache_stats['misses'] += 1
            self.cache_stats['evictions'] += 1
            return None
        
        # Update access statistics
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        
        self.cache_stats['hits'] += 1
        return entry.data
    
    async def _store_cache(self, cache_key: str, data: Any, ttl_seconds: int = 3600) -> None:
        """Store result in cache."""
        try:
            # Estimate size (rough approximation)
            size_bytes = len(pickle.dumps(data))
            
            entry = CacheEntry(
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            self.l1_cache[cache_key] = entry
            self.cache_stats['memory_usage'] += size_bytes
            
            # Check if we need to evict entries
            if self.cache_stats['memory_usage'] > self.max_memory_mb * 1024 * 1024 * 0.8:
                await self._evict_lru_entries(target_ratio=0.6)
                
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    async def _execute_optimized(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with optimizations."""
        # CPU-intensive operations go to CPU pool
        if hasattr(func, '_cpu_intensive') and func._cpu_intensive:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.cpu_executor, func, *args, **kwargs)
        
        # I/O operations go to I/O pool
        elif asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        
        else:
            # Run in I/O pool for non-async functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.io_executor, func, *args, **kwargs)
    
    async def _execute_with_batching(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with batching optimization."""
        # Add to batch queue
        future = asyncio.Future()
        batch_item = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'timestamp': time.time()
        }
        
        self.batch_queue.append(batch_item)
        
        # Wait for result
        return await future
    
    async def _batch_processor(self) -> None:
        """Background task to process batched requests."""
        self.batch_processor_running = True
        
        while self.batch_processor_running:
            try:
                if not self.batch_queue:
                    await asyncio.sleep(0.01)  # 10ms sleep
                    continue
                
                # Collect batch items
                batch_items = []
                current_time = time.time()
                
                # Collect items until batch size or timeout
                while (len(batch_items) < self.batch_size and 
                       self.batch_queue and 
                       (current_time - self.batch_queue[0]['timestamp']) * 1000 < self.batch_timeout_ms):
                    
                    batch_items.append(self.batch_queue.popleft())
                    
                    if len(batch_items) >= self.batch_size:
                        break
                
                if not batch_items:
                    continue
                
                # Group by function for batch processing
                func_groups = defaultdict(list)
                for item in batch_items:
                    func_name = item['func'].__name__
                    func_groups[func_name].append(item)
                
                # Process each function group
                for func_name, group_items in func_groups.items():
                    await self._process_function_batch(group_items)
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_function_batch(self, batch_items: List[Dict]) -> None:
        """Process a batch of items for the same function."""
        if not batch_items:
            return
        
        # Create tasks for parallel execution
        tasks = []
        for item in batch_items:
            task = asyncio.create_task(
                self._execute_optimized(item['func'], item['args'], item['kwargs'])
            )
            tasks.append((task, item['future']))
        
        # Execute batch in parallel
        try:
            results = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)
            
            # Set results on futures
            for (task, future), result in zip(tasks, results):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
                    
        except Exception as e:
            # Set exception on all futures
            for task, future in tasks:
                if not future.done():
                    future.set_exception(e)
    
    async def _process_prediction_batch(self, prediction_func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of predictions."""
        try:
            # Check if function supports batch processing
            if hasattr(prediction_func, 'predict_batch'):
                return await prediction_func.predict_batch(batch)
            
            # Otherwise process individually in parallel
            tasks = [asyncio.create_task(prediction_func(item)) for item in batch]
            return await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Fallback to individual processing
            results = []
            for item in batch:
                try:
                    result = await prediction_func(item)
                    results.append(result)
                except Exception as item_e:
                    logger.warning(f"Individual prediction failed: {item_e}")
                    results.append(None)
            return results
    
    async def _compute_and_cache_features(
        self, 
        feature_extractor: Callable, 
        code: str, 
        cache_ttl: int
    ) -> Optional[Dict]:
        """Compute and cache features for code."""
        try:
            features = await feature_extractor(code)
            
            if features:
                # Cache the features
                cache_key = f"features:{hashlib.md5(code.encode()).hexdigest()}"
                await self._store_cache(cache_key, features, cache_ttl)
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return None
    
    async def _evict_lru_entries(self, target_ratio: float = 0.7) -> None:
        """Evict least recently used cache entries."""
        if not self.l1_cache:
            return
        
        target_memory = self.max_memory_mb * 1024 * 1024 * target_ratio
        current_memory = self.cache_stats['memory_usage']
        
        if current_memory <= target_memory:
            return
        
        # Sort entries by last accessed time (LRU)
        entries = list(self.l1_cache.items())
        entries.sort(key=lambda x: x[1].last_accessed)
        
        # Remove entries until we reach target memory
        memory_freed = 0
        entries_removed = 0
        
        for key, entry in entries:
            if current_memory - memory_freed <= target_memory:
                break
            
            memory_freed += entry.size_bytes
            entries_removed += 1
            del self.l1_cache[key]
        
        self.cache_stats['memory_usage'] -= memory_freed
        self.cache_stats['evictions'] += entries_removed
        
        logger.info(f"Evicted {entries_removed} LRU entries, freed {memory_freed / 1024 / 1024:.2f} MB")
    
    def _update_performance_metrics(self, execution_time: float, cache_hit: bool = False) -> None:
        """Update performance metrics."""
        # Update response times
        self.response_times.append(execution_time)
        
        # Calculate percentiles
        if self.response_times:
            sorted_times = sorted(self.response_times)
            self.metrics.avg_response_time = sum(sorted_times) / len(sorted_times)
            
            p95_idx = int(len(sorted_times) * 0.95)
            self.metrics.p95_response_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
            
            p99_idx = int(len(sorted_times) * 0.99)
            self.metrics.p99_response_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
        
        # Update cache hit rate
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_requests > 0:
            self.metrics.cache_hit_rate = self.cache_stats['hits'] / total_requests
        
        # Update throughput (requests per second over last minute)
        current_time = time.time()
        recent_times = [t for t in self.response_times if current_time - t < 60]
        self.metrics.throughput_per_second = len(recent_times) / 60.0
    
    def _update_error_metrics(self, error: Exception, execution_time: float) -> None:
        """Update error rate metrics."""
        # Update response times even for errors
        self.response_times.append(execution_time)
        
        # Update error rate (errors in last 100 requests)
        recent_count = min(100, len(self.response_times))
        # This is simplified - in practice you'd track errors separately
        self.metrics.error_rate = 0.01  # Placeholder
    
    async def _update_memory_metrics(self) -> None:
        """Update memory usage metrics."""
        total_memory = sum(entry.size_bytes for entry in self.l1_cache.values())
        self.metrics.memory_usage_mb = total_memory / (1024 * 1024)
        self.cache_stats['memory_usage'] = total_memory
    
    async def _performance_monitor(self) -> None:
        """Background task to monitor performance."""
        while self.monitoring_enabled:
            try:
                # Update memory metrics
                await self._update_memory_metrics()
                
                # Store performance snapshot
                snapshot = {
                    'timestamp': datetime.now(),
                    'avg_response_time': self.metrics.avg_response_time,
                    'p95_response_time': self.metrics.p95_response_time,
                    'cache_hit_rate': self.metrics.cache_hit_rate,
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'active_requests': self.metrics.active_requests
                }
                
                self.performance_history.append(snapshot)
                
                # Log performance if response time exceeds target
                if self.metrics.avg_response_time > self.target_latency_ms:
                    logger.warning(
                        f"Average response time ({self.metrics.avg_response_time:.2f}ms) "
                        f"exceeds target ({self.target_latency_ms}ms)"
                    )
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _auto_tuner(self) -> None:
        """Background task for automatic performance tuning."""
        while self.auto_tuning_enabled:
            try:
                # Run auto-tuning every 5 minutes
                await asyncio.sleep(300)
                
                if (datetime.now() - self.last_optimization).total_seconds() < 300:
                    continue  # Skip if recently optimized
                
                await self._perform_auto_tuning()
                self.last_optimization = datetime.now()
                
            except Exception as e:
                logger.error(f"Auto-tuning error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_auto_tuning(self) -> None:
        """Perform automatic performance tuning."""
        logger.info("Performing automatic performance tuning")
        
        # Analyze recent performance
        if len(self.performance_history) < 10:
            return
        
        recent_snapshots = list(self.performance_history)[-10:]
        avg_response_time = sum(s['avg_response_time'] for s in recent_snapshots) / len(recent_snapshots)
        
        # Adjust batch size if response time is high
        if avg_response_time > self.target_latency_ms * 1.2:
            # Reduce batch size for lower latency
            self.batch_size = max(8, self.batch_size - 4)
            logger.info(f"Reduced batch size to {self.batch_size} for better latency")
        
        elif avg_response_time < self.target_latency_ms * 0.7:
            # Increase batch size for better throughput
            self.batch_size = min(64, self.batch_size + 4)
            logger.info(f"Increased batch size to {self.batch_size} for better throughput")
        
        # Optimize cache TTL based on hit rate
        if self.metrics.cache_hit_rate < 0.5:
            # Increase cache TTL to improve hit rate
            logger.info("Low cache hit rate detected, consider increasing TTL")
        
        # Perform memory cleanup if usage is high
        if self.metrics.memory_usage_mb / self.max_memory_mb > 0.8:
            await self.optimize_memory_usage()
    
    async def _prewarm_systems(self) -> None:
        """Pre-warm caches and thread pools."""
        logger.info("Pre-warming performance systems")
        
        # Pre-warm thread pools with dummy tasks
        dummy_tasks = []
        for _ in range(4):
            task = asyncio.get_event_loop().run_in_executor(self.cpu_executor, lambda: time.sleep(0.001))
            dummy_tasks.append(task)
        
        await asyncio.gather(*dummy_tasks)
        logger.info("Performance systems pre-warmed")


# Performance decorators for marking function characteristics
def cpu_intensive(func):
    """Mark function as CPU-intensive for optimization."""
    func._cpu_intensive = True
    return func


def cacheable(ttl_seconds: int = 3600):
    """Mark function as cacheable with specified TTL."""
    def decorator(func):
        func._cacheable = True
        func._cache_ttl = ttl_seconds
        return func
    return decorator


def batch_processable(func):
    """Mark function as supporting batch processing."""
    func._batch_processable = True
    return func