"""
Marketplace Cache - High-performance caching layer for marketplace operations.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from ...core.logger import get_logger
from ...core.config import get_settings

logger = get_logger(__name__)


class CacheKey:
    """Cache key generator with consistent naming patterns."""
    
    @staticmethod
    def template(template_id: UUID, suffix: str = "") -> str:
        """Generate cache key for template data."""
        base = f"template:{template_id}"
        return f"{base}:{suffix}" if suffix else base
    
    @staticmethod
    def template_list(category: str = "", filters: str = "", page: int = 1) -> str:
        """Generate cache key for template lists."""
        key_parts = ["templates"]
        if category:
            key_parts.append(f"cat:{category}")
        if filters:
            key_parts.append(f"filters:{filters}")
        if page > 1:
            key_parts.append(f"page:{page}")
        return ":".join(key_parts)
    
    @staticmethod
    def user_recommendations(user_id: UUID, limit: int = 20) -> str:
        """Generate cache key for user recommendations."""
        return f"user_recs:{user_id}:{limit}"
    
    @staticmethod
    def trending(period: str = "7d", category: str = "") -> str:
        """Generate cache key for trending data."""
        base = f"trending:{period}"
        return f"{base}:{category}" if category else base
    
    @staticmethod
    def search(query: str, filters: Dict[str, Any]) -> str:
        """Generate cache key for search results."""
        import hashlib
        
        # Create consistent hash of filters
        filter_str = json.dumps(filters, sort_keys=True)
        filter_hash = hashlib.md5(filter_str.encode()).hexdigest()[:8]
        
        # Clean query for key
        clean_query = query.lower().replace(" ", "_")[:50]
        
        return f"search:{clean_query}:{filter_hash}"
    
    @staticmethod
    def analytics(item_id: UUID, metric: str, period: str) -> str:
        """Generate cache key for analytics data."""
        return f"analytics:{item_id}:{metric}:{period}"
    
    @staticmethod
    def marketplace_stats(period: str = "30d") -> str:
        """Generate cache key for marketplace statistics."""
        return f"marketplace_stats:{period}"


class SerializationStrategy:
    """Pluggable serialization strategies for different data types."""
    
    @staticmethod
    def serialize_json(data: Any) -> bytes:
        """JSON serialization for simple data types."""
        return json.dumps(data, default=str, ensure_ascii=False).encode('utf-8')
    
    @staticmethod
    def deserialize_json(data: bytes) -> Any:
        """JSON deserialization."""
        return json.loads(data.decode('utf-8'))
    
    @staticmethod
    def serialize_pickle(data: Any) -> bytes:
        """Pickle serialization for complex objects."""
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize_pickle(data: bytes) -> Any:
        """Pickle deserialization."""
        return pickle.loads(data)


class MarketplaceCache:
    """
    High-performance caching layer for marketplace operations with Redis backend.
    
    Features:
    - Multiple serialization strategies
    - Automatic expiration and TTL management
    - Pattern-based cache invalidation
    - Circuit breaker for cache failures
    - Compression for large objects
    - Cache warming and prefetching
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        compression_threshold: int = 1024
    ):
        """Initialize marketplace cache."""
        self.settings = get_settings()
        self.redis_url = redis_url or self.settings.redis_url or "redis://localhost:6379"
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Redis connection
        self.pool: Optional[ConnectionPool] = None
        self.redis: Optional[redis.Redis] = None
        self._connected = False
        
        # Circuit breaker for cache failures
        self._failure_count = 0
        self._max_failures = 5
        self._failure_window = 300  # 5 minutes
        self._last_failure = None
        
        # Serialization strategies
        self._serializers = {
            'json': (SerializationStrategy.serialize_json, SerializationStrategy.deserialize_json),
            'pickle': (SerializationStrategy.serialize_pickle, SerializationStrategy.deserialize_pickle)
        }
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            self.redis = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis.ping()
            
            self._connected = True
            self.logger.info("Connected to Redis cache")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            self.redis = None
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.aclose()
        
        if self.pool:
            await self.pool.aclose()
        
        self._connected = False
        self.logger.info("Disconnected from Redis cache")
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._failure_count < self._max_failures:
            return False
        
        if self._last_failure is None:
            return False
        
        time_since_failure = (datetime.utcnow() - self._last_failure).total_seconds()
        return time_since_failure < self._failure_window
    
    def _record_failure(self) -> None:
        """Record cache operation failure."""
        self._failure_count += 1
        self._last_failure = datetime.utcnow()
        self._stats['errors'] += 1
    
    def _record_success(self) -> None:
        """Record successful cache operation."""
        if self._failure_count > 0:
            self._failure_count = max(0, self._failure_count - 1)
    
    async def get(
        self,
        key: str,
        serialization: str = "json",
        default: Any = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            serialization: Serialization strategy ('json' or 'pickle')
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        if not self._connected or self._is_circuit_open():
            self._stats['misses'] += 1
            return default
        
        try:
            if not self.redis:
                await self.connect()
            
            data = await self.redis.get(key)
            
            if data is None:
                self._stats['misses'] += 1
                return default
            
            # Deserialize data
            deserialize_func = self._serializers[serialization][1]
            
            # Handle compression if needed
            if data.startswith(b'COMPRESSED:'):
                import gzip
                data = gzip.decompress(data[11:])  # Remove 'COMPRESSED:' prefix
            
            result = deserialize_func(data)
            
            self._stats['hits'] += 1
            self._record_success()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Cache get error for key '{key}': {e}")
            self._record_failure()
            self._stats['misses'] += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialization: str = "json",
        compress: bool = False
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialization: Serialization strategy
            compress: Whether to compress large values
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or self._is_circuit_open():
            return False
        
        try:
            if not self.redis:
                await self.connect()
            
            # Serialize data
            serialize_func = self._serializers[serialization][0]
            data = serialize_func(value)
            
            # Compress if data is large and compression is enabled
            if compress or (len(data) > self.compression_threshold):
                import gzip
                compressed_data = gzip.compress(data)
                if len(compressed_data) < len(data) * 0.8:  # Only use if 20% smaller
                    data = b'COMPRESSED:' + compressed_data
            
            # Set with TTL
            ttl_value = ttl or self.default_ttl
            await self.redis.setex(key, ttl_value, data)
            
            self._stats['sets'] += 1
            self._record_success()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache set error for key '{key}': {e}")
            self._record_failure()
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or self._is_circuit_open():
            return False
        
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.delete(key)
            
            self._stats['deletes'] += 1
            self._record_success()
            
            return bool(result)
            
        except Exception as e:
            self.logger.warning(f"Cache delete error for key '{key}': {e}")
            self._record_failure()
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Key pattern (supports * wildcards)
            
        Returns:
            Number of keys deleted
        """
        if not self._connected or self._is_circuit_open():
            return 0
        
        try:
            if not self.redis:
                await self.connect()
            
            # Find matching keys
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return 0
            
            # Delete in batches to avoid blocking
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                result = await self.redis.delete(*batch)
                deleted_count += result
            
            self._stats['deletes'] += deleted_count
            self._record_success()
            
            return deleted_count
            
        except Exception as e:
            self.logger.warning(f"Cache delete pattern error for pattern '{pattern}': {e}")
            self._record_failure()
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if not self._connected or self._is_circuit_open():
            return False
        
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.exists(key)
            self._record_success()
            
            return bool(result)
            
        except Exception as e:
            self.logger.warning(f"Cache exists error for key '{key}': {e}")
            self._record_failure()
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or self._is_circuit_open():
            return False
        
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.expire(key, ttl)
            self._record_success()
            
            return bool(result)
            
        except Exception as e:
            self.logger.warning(f"Cache expire error for key '{key}': {e}")
            self._record_failure()
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment numeric value in cache.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value after increment, or None if failed
        """
        if not self._connected or self._is_circuit_open():
            return None
        
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.incrby(key, amount)
            self._record_success()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Cache increment error for key '{key}': {e}")
            self._record_failure()
            return None
    
    async def get_multiple(
        self,
        keys: List[str],
        serialization: str = "json"
    ) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            serialization: Serialization strategy
            
        Returns:
            Dictionary mapping keys to values
        """
        if not self._connected or self._is_circuit_open():
            return {}
        
        try:
            if not self.redis:
                await self.connect()
            
            # Get all values at once
            values = await self.redis.mget(keys)
            
            # Deserialize non-None values
            deserialize_func = self._serializers[serialization][1]
            result = {}
            
            for key, data in zip(keys, values):
                if data is not None:
                    try:
                        # Handle compression
                        if data.startswith(b'COMPRESSED:'):
                            import gzip
                            data = gzip.decompress(data[11:])
                        
                        result[key] = deserialize_func(data)
                        self._stats['hits'] += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to deserialize cached data for key '{key}': {e}")
                        self._stats['misses'] += 1
                else:
                    self._stats['misses'] += 1
            
            self._record_success()
            return result
            
        except Exception as e:
            self.logger.warning(f"Cache get_multiple error: {e}")
            self._record_failure()
            self._stats['misses'] += len(keys)
            return {}
    
    async def set_multiple(
        self,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        serialization: str = "json"
    ) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            serialization: Serialization strategy
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or self._is_circuit_open():
            return False
        
        try:
            if not self.redis:
                await self.connect()
            
            # Serialize all values
            serialize_func = self._serializers[serialization][0]
            serialized_data = {}
            
            for key, value in data.items():
                try:
                    serialized_value = serialize_func(value)
                    
                    # Apply compression if needed
                    if len(serialized_value) > self.compression_threshold:
                        import gzip
                        compressed_value = gzip.compress(serialized_value)
                        if len(compressed_value) < len(serialized_value) * 0.8:
                            serialized_value = b'COMPRESSED:' + compressed_value
                    
                    serialized_data[key] = serialized_value
                    
                except Exception as e:
                    self.logger.warning(f"Failed to serialize data for key '{key}': {e}")
            
            if not serialized_data:
                return False
            
            # Use pipeline for efficiency
            pipe = self.redis.pipeline()
            
            # Set all values
            pipe.mset(serialized_data)
            
            # Set expiration for all keys if TTL specified
            if ttl:
                for key in serialized_data.keys():
                    pipe.expire(key, ttl)
            
            # Execute pipeline
            await pipe.execute()
            
            self._stats['sets'] += len(serialized_data)
            self._record_success()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache set_multiple error: {e}")
            self._record_failure()
            return False
    
    async def clear_all(self) -> bool:
        """
        Clear all cache data (use with caution).
        
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or self._is_circuit_open():
            return False
        
        try:
            if not self.redis:
                await self.connect()
            
            await self.redis.flushall()
            self._record_success()
            
            self.logger.warning("All cache data has been cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache clear_all error: {e}")
            self._record_failure()
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'connected': self._connected,
            'circuit_open': self._is_circuit_open(),
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'sets': self._stats['sets'],
            'deletes': self._stats['deletes'],
            'errors': self._stats['errors'],
            'hit_rate_percent': round(hit_rate, 2),
            'failure_count': self._failure_count,
            'last_failure': self._last_failure.isoformat() if self._last_failure else None
        }
        
        # Add Redis info if connected
        if self._connected and self.redis:
            try:
                redis_info = await self.redis.info('memory')
                stats['redis_memory_usage'] = redis_info.get('used_memory_human', 'unknown')
                stats['redis_keys'] = await self.redis.dbsize()
            except Exception as e:
                self.logger.warning(f"Failed to get Redis info: {e}")
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache.
        
        Returns:
            Health check results
        """
        health = {
            'status': 'healthy',
            'connected': self._connected,
            'circuit_open': self._is_circuit_open(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            if not self._connected:
                await self.connect()
            
            # Test basic operations
            test_key = f"health_check:{datetime.utcnow().timestamp()}"
            test_value = {'test': True, 'timestamp': datetime.utcnow().isoformat()}
            
            # Test set
            set_success = await self.set(test_key, test_value, ttl=60)
            if not set_success:
                raise Exception("Failed to set test value")
            
            # Test get
            retrieved_value = await self.get(test_key)
            if retrieved_value != test_value:
                raise Exception("Retrieved value doesn't match set value")
            
            # Test delete
            delete_success = await self.delete(test_key)
            if not delete_success:
                raise Exception("Failed to delete test value")
            
            health['operations_test'] = 'passed'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            health['operations_test'] = 'failed'
        
        return health


# Global cache instance
_cache_instance: Optional[MarketplaceCache] = None


async def get_cache() -> MarketplaceCache:
    """Get global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = MarketplaceCache()
        await _cache_instance.connect()
    
    return _cache_instance


async def close_cache() -> None:
    """Close global cache instance."""
    global _cache_instance
    
    if _cache_instance:
        await _cache_instance.disconnect()
        _cache_instance = None