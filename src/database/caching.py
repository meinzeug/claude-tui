"""
Database Caching Layer - Redis-based Performance Enhancement

Advanced caching system providing:
- Multi-level caching (L1: In-memory, L2: Redis, L3: Database)
- Query result caching with intelligent invalidation
- Connection pool caching and optimization
- Session-based caching for user-specific data
- Cache warming and preloading strategies
- Real-time cache performance metrics
"""

import asyncio
import json
import time
import hashlib
import pickle
from typing import Any, Dict, Optional, List, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from collections import defaultdict, OrderedDict
import logging
from functools import wraps

import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTUIException

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_time_saved: float = 0.0
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_ratio(self) -> float:
        """Calculate cache miss ratio."""
        return 1.0 - self.hit_ratio


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class LRUCache(Generic[T]):
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self.metrics = CacheMetrics()
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        async with self._lock:
            self.metrics.total_requests += 1
            
            if key not in self._cache:
                self.metrics.misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                self.metrics.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self.metrics.hits += 1
            return entry.data
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None, tags: Optional[List[str]] = None):
        """Set value in cache."""
        async with self._lock:
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Calculate approximate size
            size_bytes = len(str(value).encode('utf-8'))
            
            entry = CacheEntry(
                data=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict oldest entries if over capacity
            while len(self._cache) > self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self.metrics.evictions += 1
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'total_size_bytes': total_size,
                'hit_ratio': self.metrics.hit_ratio,
                'hits': self.metrics.hits,
                'misses': self.metrics.misses,
                'evictions': self.metrics.evictions
            }


class DatabaseCache:
    """
    Advanced database caching system with multi-level support.
    
    Features:
    - L1: In-memory LRU cache for hot data
    - L2: Redis distributed cache for shared data
    - Query result caching with smart invalidation
    - Connection pool optimization
    - Cache warming strategies
    - Real-time performance metrics
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        l1_cache_size: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        enable_query_caching: bool = True,
        enable_connection_caching: bool = True
    ):
        """
        Initialize database cache.
        
        Args:
            redis_url: Redis connection URL
            l1_cache_size: Size of L1 in-memory cache
            default_ttl: Default TTL in seconds
            enable_query_caching: Enable query result caching
            enable_connection_caching: Enable connection metadata caching
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.enable_query_caching = enable_query_caching
        self.enable_connection_caching = enable_connection_caching
        
        # L1 Cache (In-memory)
        self.l1_cache = LRUCache(max_size=l1_cache_size)
        
        # L2 Cache (Redis)
        self.redis_client: Optional[redis.Redis] = None
        self.redis_connected = False
        
        # Cache invalidation tracking
        self.table_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.cache_tags: Dict[str, List[str]] = defaultdict(list)
        
        # Performance metrics
        self.total_cache_requests = 0
        self.total_cache_hits = 0
        self.cache_time_saved = 0.0
        
        logger.info(f"Database cache initialized (L1: {l1_cache_size}, TTL: {default_ttl}s)")
    
    async def initialize(self):
        """Initialize cache components."""
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,
                    retry_on_timeout=True
                )
                await self.redis_client.ping()
                self.redis_connected = True
                logger.info("Redis cache L2 connected successfully")
            except Exception as e:
                logger.warning(f"Redis L2 cache unavailable, using L1 only: {e}")
                self.redis_client = None
                self.redis_connected = False
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key."""
        # Create deterministic hash from arguments
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """Get value from cache (L1 -> L2 -> None)."""
        start_time = time.time()
        self.total_cache_requests += 1
        
        try:
            # Try L1 cache first
            value = await self.l1_cache.get(key)
            if value is not None:
                self.total_cache_hits += 1
                self.cache_time_saved += time.time() - start_time
                return value
            
            # Try L2 cache (Redis)
            if self.redis_connected and self.redis_client:
                redis_value = await self.redis_client.get(key)
                if redis_value is not None:
                    # Deserialize from Redis
                    if deserialize:
                        try:
                            value = json.loads(redis_value)
                        except (json.JSONDecodeError, TypeError):
                            # Try pickle for complex objects
                            value = pickle.loads(redis_value.encode('latin1'))
                    else:
                        value = redis_value
                    
                    # Store in L1 for faster future access
                    await self.l1_cache.set(key, value, ttl=self.default_ttl // 2)
                    
                    self.total_cache_hits += 1
                    self.cache_time_saved += time.time() - start_time
                    return value
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        serialize: bool = True
    ):
        """Set value in cache (both L1 and L2)."""
        ttl = ttl or self.default_ttl
        
        try:
            # Store in L1 cache
            await self.l1_cache.set(key, value, ttl=ttl, tags=tags)
            
            # Store in L2 cache (Redis)
            if self.redis_connected and self.redis_client:
                if serialize:
                    try:
                        serialized_value = json.dumps(value, default=str)
                    except (TypeError, ValueError):
                        # Use pickle for complex objects
                        serialized_value = pickle.dumps(value).decode('latin1')
                else:
                    serialized_value = value
                
                await self.redis_client.setex(key, ttl, serialized_value)
            
            # Track cache tags for invalidation
            if tags:
                for tag in tags:
                    self.cache_tags[tag].append(key)
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
    
    async def delete(self, key: str):
        """Delete key from both cache levels."""
        try:
            await self.l1_cache.delete(key)
            
            if self.redis_connected and self.redis_client:
                await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
    
    async def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags."""
        keys_to_invalidate = set()
        
        for tag in tags:
            if tag in self.cache_tags:
                keys_to_invalidate.update(self.cache_tags[tag])
                del self.cache_tags[tag]
        
        # Delete all affected keys
        for key in keys_to_invalidate:
            await self.delete(key)
        
        logger.debug(f"Invalidated {len(keys_to_invalidate)} cache entries for tags: {tags}")
    
    async def invalidate_table(self, table_name: str):
        """Invalidate all cache entries related to a table."""
        await self.invalidate_by_tags([f"table:{table_name}"])
    
    def cached_query(
        self, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        key_prefix: str = "query"
    ):
        """Decorator for caching query results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.enable_query_caching:
                    return await func(*args, **kwargs)
                
                # Generate cache key
                cache_key = self._generate_cache_key(key_prefix, func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for query: {func.__name__}")
                    return cached_result
                
                # Execute query
                start_time = time.time()
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Cache the result
                cache_ttl = ttl or self.default_ttl
                cache_tags = tags or []
                
                # Add automatic tags based on function name
                if "user" in func.__name__.lower():
                    cache_tags.append("table:users")
                elif "project" in func.__name__.lower():
                    cache_tags.append("table:projects")
                elif "task" in func.__name__.lower():
                    cache_tags.append("table:tasks")
                
                await self.set(cache_key, result, ttl=cache_ttl, tags=cache_tags)
                
                logger.debug(f"Cached query result: {func.__name__} (exec: {execution_time:.3f}s)")
                return result
            
            return wrapper
        return decorator
    
    async def warm_cache(self, session: AsyncSession, queries: List[str]):
        """Warm cache with predefined queries."""
        logger.info(f"Warming cache with {len(queries)} queries...")
        
        for query in queries:
            try:
                cache_key = self._generate_cache_key("warm", query)
                
                # Check if already cached
                if await self.get(cache_key) is not None:
                    continue
                
                # Execute and cache
                result = await session.execute(text(query))
                rows = result.fetchall()
                
                # Convert to serializable format
                serializable_rows = [dict(row._mapping) for row in rows]
                
                await self.set(cache_key, serializable_rows, tags=["warmup"])
                
            except Exception as e:
                logger.error(f"Cache warming failed for query: {query[:50]}... Error: {e}")
        
        logger.info("Cache warming completed")
    
    async def preload_user_data(self, session: AsyncSession, user_id: str):
        """Preload frequently accessed user data."""
        try:
            # Common user queries to preload
            queries = [
                f"SELECT * FROM users WHERE id = '{user_id}'",
                f"SELECT * FROM projects WHERE owner_id = '{user_id}' AND is_archived = false",
                f"SELECT * FROM tasks WHERE assigned_to = '{user_id}' AND status != 'completed'",
                f"SELECT * FROM user_sessions WHERE user_id = '{user_id}' AND is_active = true"
            ]
            
            await self.warm_cache(session, queries)
            logger.debug(f"Preloaded data for user: {user_id}")
            
        except Exception as e:
            logger.error(f"User data preloading failed for {user_id}: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = await self.l1_cache.get_stats()
        
        total_hit_ratio = 0.0
        if self.total_cache_requests > 0:
            total_hit_ratio = self.total_cache_hits / self.total_cache_requests
        
        stats = {
            'total_requests': self.total_cache_requests,
            'total_hits': self.total_cache_hits,
            'hit_ratio': total_hit_ratio,
            'time_saved_seconds': self.cache_time_saved,
            'l1_cache': l1_stats,
            'l2_cache': {
                'connected': self.redis_connected,
                'url': self.redis_url if self.redis_url else 'Not configured'
            },
            'settings': {
                'default_ttl': self.default_ttl,
                'query_caching_enabled': self.enable_query_caching,
                'connection_caching_enabled': self.enable_connection_caching
            },
            'invalidation': {
                'tracked_tags': len(self.cache_tags),
                'total_tagged_keys': sum(len(keys) for keys in self.cache_tags.values())
            }
        }
        
        # Add Redis stats if available
        if self.redis_connected and self.redis_client:
            try:
                redis_info = await self.redis_client.info('memory')
                stats['l2_cache'].update({
                    'memory_used': redis_info.get('used_memory_human', 'Unknown'),
                    'memory_peak': redis_info.get('used_memory_peak_human', 'Unknown'),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0)
                })
            except Exception as e:
                logger.debug(f"Could not fetch Redis stats: {e}")
        
        return stats
    
    async def clear_all_caches(self):
        """Clear all cache levels."""
        await self.l1_cache.clear()
        
        if self.redis_connected and self.redis_client:
            await self.redis_client.flushdb()
        
        self.cache_tags.clear()
        logger.info("All caches cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        health = {
            'l1_cache': 'healthy',
            'l2_cache': 'healthy' if self.redis_connected else 'unavailable',
            'total_performance': 'good'
        }
        
        # Check L1 cache health
        try:
            await self.l1_cache.set('health_check', 'test', ttl=1)
            test_value = await self.l1_cache.get('health_check')
            if test_value != 'test':
                health['l1_cache'] = 'unhealthy'
        except Exception:
            health['l1_cache'] = 'unhealthy'
        
        # Check L2 cache health
        if self.redis_connected and self.redis_client:
            try:
                await self.redis_client.set('health_check', 'test', ex=1)
                test_value = await self.redis_client.get('health_check')
                if test_value != 'test':
                    health['l2_cache'] = 'unhealthy'
            except Exception:
                health['l2_cache'] = 'unhealthy'
                self.redis_connected = False
        
        # Overall health assessment
        if health['l1_cache'] == 'unhealthy':
            health['total_performance'] = 'degraded'
        elif health['l2_cache'] == 'unhealthy' and self.redis_url:
            health['total_performance'] = 'degraded'
        
        return health
    
    async def close(self):
        """Clean up cache resources."""
        if self.redis_client:
            await self.redis_client.close()
        
        await self.l1_cache.clear()
        self.cache_tags.clear()
        
        logger.info("Database cache closed")


# Global cache instance
_database_cache: Optional[DatabaseCache] = None


def get_database_cache() -> DatabaseCache:
    """Get global database cache instance."""
    global _database_cache
    if _database_cache is None:
        _database_cache = DatabaseCache()
    return _database_cache


async def setup_database_caching(
    redis_url: Optional[str] = None,
    l1_cache_size: int = 1000,
    default_ttl: int = 3600
) -> DatabaseCache:
    """Set up database caching system."""
    global _database_cache
    
    _database_cache = DatabaseCache(
        redis_url=redis_url,
        l1_cache_size=l1_cache_size,
        default_ttl=default_ttl
    )
    
    await _database_cache.initialize()
    
    logger.info("Database caching system initialized")
    return _database_cache


# Context manager for automatic cache invalidation
@asynccontextmanager
async def cache_invalidation_context(cache: DatabaseCache, tables: List[str]):
    """Context manager for automatic cache invalidation on table changes."""
    try:
        yield
    finally:
        # Invalidate caches for modified tables
        for table in tables:
            await cache.invalidate_table(table)