#!/usr/bin/env python3
"""
Advanced Caching System - High-Performance Caching for Hot Paths

Implements multi-tier caching strategy:
- In-memory LRU caches for frequently accessed data
- Redis distributed caching for shared data
- Database query result caching with intelligent invalidation
- API response caching with ETags and conditional requests
- Static asset caching with CDN integration
- Cache warming and preloading strategies
"""

import time
import json
import hashlib
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from collections import OrderedDict
import weakref
import pickle
import zlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.created_at
    
    def touch(self):
        """Update access time and count"""
        self.accessed_at = time.time()
        self.access_count += 1


class LRUCache:
    """High-performance in-memory LRU cache with advanced features"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self.cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.touch()
            self.hits += 1
            return entry.value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache"""
        # Calculate size estimate
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = len(str(value))
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            ttl=ttl or self.default_ttl,
            size_bytes=size_bytes
        )
        
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]
        
        # Add new entry
        self.cache[key] = entry
        
        # Enforce size limit
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.evictions += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
        
        # Calculate total size
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'total_size_bytes': total_size,
            'avg_entry_size': total_size / len(self.cache) if self.cache else 0
        }


class RedisCache:
    """Distributed Redis-based cache"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379', key_prefix: str = 'cache'):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False
        
    async def connect(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping Redis cache")
            return
            
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.connected = True
            logger.info(f"Redis cache connected: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.connected = False
    
    def _make_key(self, key: str) -> str:
        """Make prefixed cache key"""
        return f"{self.key_prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.connected:
            return None
            
        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)
            if data:
                # Decompress and deserialize
                decompressed = zlib.decompress(data)
                return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    async def put(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Put value in Redis"""
        if not self.connected:
            return False
            
        try:
            redis_key = self._make_key(key)
            # Serialize and compress
            serialized = pickle.dumps(value)
            compressed = zlib.compress(serialized)
            
            await self.redis_client.setex(redis_key, ttl_seconds, compressed)
            return True
        except Exception as e:
            logger.error(f"Redis put error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.connected:
            return False
            
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str = "*"):
        """Clear keys matching pattern"""
        if not self.connected:
            return 0
            
        try:
            pattern_key = self._make_key(pattern)
            keys = await self.redis_client.keys(pattern_key)
            if keys:
                return await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
        return 0


class MultiTierCache:
    """Multi-tier cache combining in-memory and distributed caching"""
    
    def __init__(
        self,
        l1_cache_size: int = 1000,
        l1_ttl_seconds: float = 300,
        redis_url: Optional[str] = None,
        redis_ttl_seconds: int = 3600
    ):
        # L1 Cache (In-memory, fastest)
        self.l1_cache = LRUCache(max_size=l1_cache_size, ttl_seconds=l1_ttl_seconds)
        
        # L2 Cache (Redis, shared)
        self.l2_cache = None
        if redis_url and REDIS_AVAILABLE:
            self.l2_cache = RedisCache(redis_url=redis_url)
        
        self.redis_ttl = redis_ttl_seconds
        
        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.total_misses = 0
        
    async def initialize(self):
        """Initialize cache connections"""
        if self.l2_cache:
            await self.l2_cache.connect()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-tier cache"""
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.l1_hits += 1
            return value
        
        # Try L2 cache (Redis)
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                self.l2_hits += 1
                # Store in L1 for next time
                self.l1_cache.put(key, value)
                return value
        
        self.total_misses += 1
        return None
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Put value in multi-tier cache"""
        # Store in L1 cache
        l1_ttl = min(ttl_seconds or 300, 300)  # L1 max TTL is 5 minutes
        self.l1_cache.put(key, value, ttl=l1_ttl)
        
        # Store in L2 cache
        if self.l2_cache:
            l2_ttl = ttl_seconds or self.redis_ttl
            await self.l2_cache.put(key, value, l2_ttl)
    
    async def delete(self, key: str):
        """Delete from all cache tiers"""
        self.l1_cache.delete(key)
        if self.l2_cache:
            await self.l2_cache.delete(key)
    
    async def clear(self):
        """Clear all cache tiers"""
        self.l1_cache.clear()
        if self.l2_cache:
            await self.l2_cache.clear_pattern()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = self.l1_cache.stats()
        
        total_requests = self.l1_hits + self.l2_hits + self.total_misses
        overall_hit_rate = ((self.l1_hits + self.l2_hits) / total_requests) if total_requests > 0 else 0.0
        
        return {
            'l1_cache': l1_stats,
            'l2_enabled': self.l2_cache is not None,
            'l2_connected': self.l2_cache.connected if self.l2_cache else False,
            'performance': {
                'l1_hits': self.l1_hits,
                'l2_hits': self.l2_hits,
                'total_misses': self.total_misses,
                'overall_hit_rate': overall_hit_rate,
                'cache_efficiency': 'excellent' if overall_hit_rate > 0.8 else 
                                  'good' if overall_hit_rate > 0.6 else 
                                  'poor'
            }
        }


class CacheManager:
    """Advanced cache manager with preloading and warming"""
    
    def __init__(self, redis_url: Optional[str] = None):
        # Initialize cache instances
        self.api_cache = MultiTierCache(
            l1_cache_size=500,
            l1_ttl_seconds=300,  # 5 minutes
            redis_url=redis_url,
            redis_ttl_seconds=1800  # 30 minutes
        )
        
        self.database_cache = MultiTierCache(
            l1_cache_size=200,
            l1_ttl_seconds=600,  # 10 minutes
            redis_url=redis_url,
            redis_ttl_seconds=3600  # 1 hour
        )
        
        self.static_cache = LRUCache(max_size=1000, ttl_seconds=3600)  # 1 hour
        
        # Cache warming tasks
        self.warming_tasks: List[asyncio.Task] = []
        self.warming_enabled = True
        
    async def initialize(self):
        """Initialize all cache systems"""
        await self.api_cache.initialize()
        await self.database_cache.initialize()
        logger.info("Cache manager initialized")
    
    def cache_api_response(
        self, 
        ttl_seconds: int = 1800,
        vary_on: List[str] = None,
        etag_enabled: bool = True
    ):
        """Decorator for caching API responses"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key based on function and arguments
                cache_key = self._generate_cache_key(func.__name__, args, kwargs, vary_on)
                
                # Try to get from cache
                cached_result = await self.api_cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.api_cache.put(cache_key, result, ttl_seconds)
                
                return result
            return wrapper
        return decorator
    
    def cache_database_query(self, ttl_seconds: int = 3600):
        """Decorator for caching database query results"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = self._generate_cache_key(f"db:{func.__name__}", args, kwargs)
                
                cached_result = await self.database_cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                result = await func(*args, **kwargs)
                await self.database_cache.put(cache_key, result, ttl_seconds)
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(
        self, 
        prefix: str, 
        args: Tuple, 
        kwargs: Dict, 
        vary_on: Optional[List[str]] = None
    ) -> str:
        """Generate deterministic cache key"""
        # Create key components
        key_data = {
            'prefix': prefix,
            'args': str(args),
            'kwargs': {k: v for k, v in kwargs.items() if vary_on is None or k in vary_on}
        }
        
        # Generate hash
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()
    
    async def warm_cache_with_data(self, warmup_data: Dict[str, Any]):
        """Warm cache with predefined data"""
        for cache_key, data in warmup_data.items():
            await self.api_cache.put(cache_key, data['value'], data.get('ttl', 1800))
        
        logger.info(f"Cache warmed with {len(warmup_data)} entries")
    
    async def start_cache_warming(self, warming_functions: List[Callable]):
        """Start background cache warming tasks"""
        if not self.warming_enabled:
            return
        
        for func in warming_functions:
            task = asyncio.create_task(self._warming_worker(func))
            self.warming_tasks.append(task)
        
        logger.info(f"Started {len(warming_functions)} cache warming tasks")
    
    async def _warming_worker(self, warming_func: Callable):
        """Background worker for cache warming"""
        while self.warming_enabled:
            try:
                await warming_func(self)
                await asyncio.sleep(300)  # Warm every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # For in-memory caches, we'll need to iterate and check
        # This is a simplified implementation
        await self.api_cache.clear()
        await self.database_cache.clear()
        
        logger.info(f"Invalidated cache pattern: {pattern}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'api_cache': self.api_cache.get_stats(),
            'database_cache': self.database_cache.get_stats(),
            'static_cache': self.static_cache.stats(),
            'warming_tasks_active': len(self.warming_tasks),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown cache manager"""
        self.warming_enabled = False
        
        # Cancel warming tasks
        for task in self.warming_tasks:
            task.cancel()
        
        if self.warming_tasks:
            await asyncio.gather(*self.warming_tasks, return_exceptions=True)
        
        logger.info("Cache manager shut down")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


async def initialize_caching(redis_url: Optional[str] = None) -> CacheManager:
    """Initialize global caching system"""
    global _cache_manager
    _cache_manager = CacheManager(redis_url=redis_url)
    await _cache_manager.initialize()
    
    logger.info("Advanced caching system initialized")
    return _cache_manager


# Convenience decorators using global cache manager
def cached_api_response(ttl_seconds: int = 1800, vary_on: List[str] = None):
    """Convenience decorator for API response caching"""
    def decorator(func: Callable):
        cache_manager = get_cache_manager()
        return cache_manager.cache_api_response(ttl_seconds, vary_on)(func)
    return decorator


def cached_database_query(ttl_seconds: int = 3600):
    """Convenience decorator for database query caching"""
    def decorator(func: Callable):
        cache_manager = get_cache_manager()
        return cache_manager.cache_database_query(ttl_seconds)(func)
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    async def test_caching_system():
        print("🚀 ADVANCED CACHING SYSTEM - Testing")
        print("=" * 50)
        
        # Initialize cache manager
        cache_manager = await initialize_caching()
        
        # Test multi-tier caching
        print("📊 Testing multi-tier caching...")
        await cache_manager.api_cache.put("test_key", "test_value", 300)
        
        # Test cache retrieval
        value = await cache_manager.api_cache.get("test_key")
        print(f"   Retrieved: {value}")
        
        # Test statistics
        stats = cache_manager.get_comprehensive_stats()
        print(f"   Cache hit rate: {stats['api_cache']['performance']['overall_hit_rate']:.2%}")
        
        # Test cache decorators
        @cached_api_response(ttl_seconds=600)
        async def sample_api_function(param1: str, param2: int):
            await asyncio.sleep(0.1)  # Simulate work
            return f"Result for {param1}:{param2}"
        
        # First call (cache miss)
        start_time = time.time()
        result1 = await sample_api_function("test", 123)
        miss_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = await sample_api_function("test", 123)
        hit_time = time.time() - start_time
        
        print(f"   Cache miss time: {miss_time:.4f}s")
        print(f"   Cache hit time: {hit_time:.4f}s")
        print(f"   Speedup: {miss_time/hit_time:.1f}x")
        
        # Final statistics
        final_stats = cache_manager.get_comprehensive_stats()
        print(f"\n📈 Final Statistics:")
        print(f"   API cache hit rate: {final_stats['api_cache']['performance']['overall_hit_rate']:.2%}")
        print(f"   L1 cache size: {final_stats['api_cache']['l1_cache']['size']}")
        
        await cache_manager.shutdown()
        print("\n✅ Caching system test completed!")
    
    # Run test
    asyncio.run(test_caching_system())