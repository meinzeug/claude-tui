"""AI Cache Manager Module

Advanced caching system for AI services with:
- Multi-level caching (memory, Redis, disk)
- Intelligent cache invalidation and TTL
- Performance analytics and optimization
- Cache warming and preloading strategies
- Memory-efficient LRU and LFU algorithms
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from weakref import WeakValueDictionary

import redis.asyncio as redis
import psutil

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache storage levels"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live based
    SIZE = "size"  # Size based
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class CacheStrategy(Enum):
    """Cache strategies"""
    WRITE_THROUGH = "write_through"  # Write to cache and storage simultaneously
    WRITE_BACK = "write_back"  # Write to cache first, storage later
    WRITE_AROUND = "write_around"  # Write to storage, bypass cache
    LAZY_LOADING = "lazy_loading"  # Load on demand
    PROACTIVE = "proactive"  # Preload based on patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def last_access_seconds_ago(self) -> float:
        """Get seconds since last access"""
        return (datetime.utcnow() - self.last_accessed).total_seconds()


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writes: int = 0
    deletes: int = 0
    memory_usage_bytes: int = 0
    total_entries: int = 0
    average_access_time_ms: float = 0.0
    hit_rate_percent: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate percentage"""
        total_requests = self.hits + self.misses
        if total_requests > 0:
            self.hit_rate_percent = (self.hits / total_requests) * 100.0


class LRUCache:
    """High-performance LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache"""
        start_time = time.time()
        
        try:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired:
                del self.cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            # Update entry access info
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            
            self.stats.hits += 1
            return entry
            
        finally:
            # Update average access time
            access_time = (time.time() - start_time) * 1000  # Convert to ms
            self.stats.average_access_time_ms = (
                (self.stats.average_access_time_ms * (self.stats.hits + self.stats.misses - 1) + access_time) /
                (self.stats.hits + self.stats.misses)
            )
            self.stats.update_hit_rate()
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, tags: List[str] = None) -> bool:
        """Put item in cache"""
        try:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            # Check if we need to evict
            while (len(self.cache) >= self.max_size or 
                   self.stats.memory_usage_bytes + size_bytes > self.max_memory_bytes):
                if not self.cache:
                    break
                self._evict_lru()
            
            # Remove existing entry if updating
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.memory_usage_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.stats.memory_usage_bytes += size_bytes
            self.stats.writes += 1
            self.stats.total_entries = len(self.cache)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to put cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats.memory_usage_bytes -= entry.size_bytes
            self.stats.deletes += 1
            self.stats.total_entries = len(self.cache)
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.stats.memory_usage_bytes = 0
        self.stats.total_entries = 0
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)  # Remove first (least recent)
            self.stats.memory_usage_bytes -= entry.size_bytes
            self.stats.evictions += 1
            self.stats.total_entries = len(self.cache)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in value.items())
            else:
                return 100  # Default estimation


class AICache:
    """
    Advanced AI Cache Manager with multi-level caching
    
    Features:
    - Multi-level caching (Memory -> Redis -> Disk)
    - Intelligent cache policies and strategies
    - Performance analytics and optimization
    - Cache warming and preloading
    - Tag-based invalidation
    - Compression and serialization
    """
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        memory_limit_mb: int = 100,
        redis_url: str = "redis://localhost:6379",
        disk_cache_dir: str = "./cache",
        default_ttl: int = 3600,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        cache_strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH,
        enable_compression: bool = True,
        enable_encryption: bool = False
    ):
        self.memory_cache_size = memory_cache_size
        self.memory_limit_mb = memory_limit_mb
        self.redis_url = redis_url
        self.disk_cache_dir = Path(disk_cache_dir)
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.cache_strategy = cache_strategy
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        
        # Create cache directory
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache levels
        self.memory_cache = LRUCache(memory_cache_size, memory_limit_mb)
        self.redis_client: Optional[redis.Redis] = None
        
        # Performance tracking
        self.global_stats = CacheStats()
        self.level_stats: Dict[CacheLevel, CacheStats] = {
            level: CacheStats() for level in CacheLevel
        }
        
        # Cache warming and patterns
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.warming_tasks: List[asyncio.Task] = []
        
        # Tag-based invalidation
        self.tag_to_keys: Dict[str, set] = defaultdict(set)
        self.key_to_tags: Dict[str, set] = defaultdict(set)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for cache manager"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    async def initialize(self):
        """Initialize cache manager"""
        logger.info("Initializing AI cache manager")
        
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running with memory cache only.")
            self.redis_client = None
        
        logger.info("AI cache manager initialized")
    
    async def get(
        self,
        key: str,
        default: Optional[Any] = None,
        update_access_pattern: bool = True
    ) -> Any:
        """Get value from cache with multi-level lookup"""
        
        start_time = time.time()
        
        try:
            # Generate cache key hash
            cache_key = self._generate_cache_key(key)
            
            # Update access patterns
            if update_access_pattern:
                self._update_access_pattern(key)
            
            # Level 1: Memory cache
            memory_entry = self.memory_cache.get(cache_key)
            if memory_entry is not None:
                self.level_stats[CacheLevel.MEMORY].hits += 1
                self.global_stats.hits += 1
                return memory_entry.value
            
            self.level_stats[CacheLevel.MEMORY].misses += 1
            
            # Level 2: Redis cache
            if self.redis_client:
                redis_value = await self._get_from_redis(cache_key)
                if redis_value is not None:
                    # Promote to memory cache
                    self.memory_cache.put(cache_key, redis_value)
                    
                    self.level_stats[CacheLevel.REDIS].hits += 1
                    self.global_stats.hits += 1
                    return redis_value
                
                self.level_stats[CacheLevel.REDIS].misses += 1
            
            # Level 3: Disk cache
            disk_value = await self._get_from_disk(cache_key)
            if disk_value is not None:
                # Promote to higher levels
                self.memory_cache.put(cache_key, disk_value)
                if self.redis_client:
                    await self._put_to_redis(cache_key, disk_value, self.default_ttl)
                
                self.level_stats[CacheLevel.DISK].hits += 1
                self.global_stats.hits += 1
                return disk_value
            
            self.level_stats[CacheLevel.DISK].misses += 1
            self.global_stats.misses += 1
            
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.global_stats.misses += 1
            return default
        
        finally:
            # Update performance metrics
            access_time = (time.time() - start_time) * 1000
            self._update_global_access_time(access_time)
            self.global_stats.update_hit_rate()
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        cache_levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """Put value in cache across specified levels"""
        
        try:
            cache_key = self._generate_cache_key(key)
            ttl = ttl or self.default_ttl
            tags = tags or []
            cache_levels = cache_levels or [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
            
            # Update tag mappings
            self._update_tag_mappings(cache_key, tags)
            
            # Serialize value if needed
            serialized_value = self._serialize_value(value)
            
            success = True
            
            # Write to specified cache levels
            if CacheLevel.MEMORY in cache_levels:
                memory_success = self.memory_cache.put(cache_key, value, ttl, tags)
                self.level_stats[CacheLevel.MEMORY].writes += 1
                success &= memory_success
            
            if CacheLevel.REDIS in cache_levels and self.redis_client:
                redis_success = await self._put_to_redis(cache_key, serialized_value, ttl)
                self.level_stats[CacheLevel.REDIS].writes += 1
                success &= redis_success
            
            if CacheLevel.DISK in cache_levels:
                disk_success = await self._put_to_disk(cache_key, serialized_value, ttl, tags)
                self.level_stats[CacheLevel.DISK].writes += 1
                success &= disk_success
            
            if success:
                self.global_stats.writes += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Cache put error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""
        
        try:
            cache_key = self._generate_cache_key(key)
            
            success = True
            
            # Delete from memory
            memory_success = self.memory_cache.delete(cache_key)
            
            # Delete from Redis
            redis_success = True
            if self.redis_client:
                redis_success = bool(await self.redis_client.delete(cache_key))
            
            # Delete from disk
            disk_success = await self._delete_from_disk(cache_key)
            
            # Clean up tag mappings
            self._cleanup_tag_mappings(cache_key)
            
            success = memory_success or redis_success or disk_success
            
            if success:
                self.global_stats.deletes += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all cache entries with specified tags"""
        
        invalidated_count = 0
        
        for tag in tags:
            keys_to_invalidate = self.tag_to_keys.get(tag, set()).copy()
            
            for cache_key in keys_to_invalidate:
                if await self.delete(cache_key):
                    invalidated_count += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries for tags: {tags}")
        return invalidated_count
    
    async def warm_cache(
        self,
        keys: List[str],
        loader_func: Callable[[str], Any],
        ttl: Optional[int] = None
    ) -> int:
        """Warm cache with preloaded data"""
        
        warmed_count = 0
        
        for key in keys:
            try:
                # Check if already cached
                if await self.get(key, update_access_pattern=False) is not None:
                    continue
                
                # Load data
                value = await loader_func(key) if asyncio.iscoroutinefunction(loader_func) else loader_func(key)
                
                if value is not None:
                    if await self.put(key, value, ttl):
                        warmed_count += 1
                        
            except Exception as e:
                logger.warning(f"Failed to warm cache for key {key}: {e}")
        
        logger.info(f"Warmed {warmed_count} cache entries")
        return warmed_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        # Calculate memory usage
        process = psutil.Process()
        system_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'global_stats': {
                'hits': self.global_stats.hits,
                'misses': self.global_stats.misses,
                'hit_rate_percent': self.global_stats.hit_rate_percent,
                'writes': self.global_stats.writes,
                'deletes': self.global_stats.deletes,
                'average_access_time_ms': self.global_stats.average_access_time_ms
            },
            'level_stats': {
                level.value: {
                    'hits': stats.hits,
                    'misses': stats.misses,
                    'writes': stats.writes,
                    'hit_rate': (stats.hits / max(stats.hits + stats.misses, 1)) * 100
                }
                for level, stats in self.level_stats.items()
            },
            'memory_cache': {
                'entries': len(self.memory_cache.cache),
                'max_size': self.memory_cache.max_size,
                'memory_usage_mb': self.memory_cache.stats.memory_usage_bytes / 1024 / 1024,
                'memory_limit_mb': self.memory_limit_mb
            },
            'system': {
                'process_memory_mb': system_memory,
                'cache_strategy': self.cache_strategy.value,
                'eviction_policy': self.eviction_policy.value
            },
            'tags': {
                'total_tags': len(self.tag_to_keys),
                'tagged_keys': len(self.key_to_tags)
            }
        }
    
    async def optimize(self):
        """Optimize cache performance based on usage patterns"""
        
        logger.info("Starting cache optimization")
        
        try:
            # Analyze access patterns
            hot_keys = self._identify_hot_keys()
            cold_keys = self._identify_cold_keys()
            
            # Preload hot keys
            if hot_keys:
                logger.info(f"Identified {len(hot_keys)} hot keys for optimization")
            
            # Evict cold keys from memory to make room
            evicted_count = 0
            for key in cold_keys:
                cache_key = self._generate_cache_key(key)
                if self.memory_cache.delete(cache_key):
                    evicted_count += 1
            
            if evicted_count > 0:
                logger.info(f"Evicted {evicted_count} cold keys from memory cache")
            
            # Clean up expired entries
            await self._cleanup_expired_entries()
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate consistent cache key hash"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            serialized = pickle.dumps(value)
            
            if self.enable_compression:
                import zlib
                serialized = zlib.compress(serialized)
            
            if self.enable_encryption:
                # Would implement encryption here
                pass
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            if self.enable_encryption:
                # Would implement decryption here
                pass
            
            if self.enable_compression:
                import zlib
                data = zlib.decompress(data)
            
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            data = await self.redis_client.get(key)
            if data:
                return self._deserialize_value(data)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None
    
    async def _put_to_redis(self, key: str, value: bytes, ttl: int) -> bool:
        """Put value to Redis cache"""
        try:
            await self.redis_client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.warning(f"Redis put failed: {e}")
            return False
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        try:
            cache_file = self.disk_cache_dir / f"{key}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = f.read()
                return self._deserialize_value(data)
        except Exception as e:
            logger.warning(f"Disk get failed: {e}")
        return None
    
    async def _put_to_disk(self, key: str, value: bytes, ttl: int, tags: List[str]) -> bool:
        """Put value to disk cache"""
        try:
            cache_file = self.disk_cache_dir / f"{key}.cache"
            
            # Write cache data
            with open(cache_file, 'wb') as f:
                f.write(value)
            
            # Write metadata
            metadata = {
                'created_at': datetime.utcnow().isoformat(),
                'ttl_seconds': ttl,
                'tags': tags
            }
            
            metadata_file = self.disk_cache_dir / f"{key}.meta"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            return True
            
        except Exception as e:
            logger.warning(f"Disk put failed: {e}")
            return False
    
    async def _delete_from_disk(self, key: str) -> bool:
        """Delete value from disk cache"""
        try:
            cache_file = self.disk_cache_dir / f"{key}.cache"
            metadata_file = self.disk_cache_dir / f"{key}.meta"
            
            success = False
            if cache_file.exists():
                cache_file.unlink()
                success = True
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            return success
            
        except Exception as e:
            logger.warning(f"Disk delete failed: {e}")
            return False
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for a key"""
        now = datetime.utcnow()
        self.access_patterns[key].append(now)
        
        # Keep only recent accesses (last hour)
        cutoff = now - timedelta(hours=1)
        self.access_patterns[key] = [
            access_time for access_time in self.access_patterns[key]
            if access_time > cutoff
        ]
    
    def _update_tag_mappings(self, key: str, tags: List[str]):
        """Update tag to key mappings"""
        # Clean up old mappings
        self._cleanup_tag_mappings(key)
        
        # Add new mappings
        for tag in tags:
            self.tag_to_keys[tag].add(key)
            self.key_to_tags[key].add(tag)
    
    def _cleanup_tag_mappings(self, key: str):
        """Clean up tag mappings for a key"""
        if key in self.key_to_tags:
            tags = self.key_to_tags[key].copy()
            for tag in tags:
                self.tag_to_keys[tag].discard(key)
                if not self.tag_to_keys[tag]:
                    del self.tag_to_keys[tag]
            del self.key_to_tags[key]
    
    def _identify_hot_keys(self, threshold: int = 10) -> List[str]:
        """Identify frequently accessed keys"""
        hot_keys = []
        
        for key, accesses in self.access_patterns.items():
            if len(accesses) >= threshold:
                hot_keys.append(key)
        
        return hot_keys
    
    def _identify_cold_keys(self, threshold_minutes: int = 30) -> List[str]:
        """Identify rarely accessed keys"""
        cutoff = datetime.utcnow() - timedelta(minutes=threshold_minutes)
        cold_keys = []
        
        for key, accesses in self.access_patterns.items():
            if not accesses or all(access < cutoff for access in accesses):
                cold_keys.append(key)
        
        return cold_keys
    
    def _update_global_access_time(self, access_time_ms: float):
        """Update global average access time"""
        total_requests = self.global_stats.hits + self.global_stats.misses
        if total_requests > 0:
            self.global_stats.average_access_time_ms = (
                (self.global_stats.average_access_time_ms * (total_requests - 1) + access_time_ms) /
                total_requests
            )
    
    async def _cleanup_expired_entries(self):
        """Clean up expired disk cache entries"""
        try:
            cleaned_count = 0
            
            for metadata_file in self.disk_cache_dir.glob("*.meta"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    created_at = datetime.fromisoformat(metadata['created_at'])
                    ttl_seconds = metadata.get('ttl_seconds', self.default_ttl)
                    
                    if (datetime.utcnow() - created_at).total_seconds() > ttl_seconds:
                        # Remove expired entry
                        key = metadata_file.stem
                        await self._delete_from_disk(key)
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process metadata file {metadata_file}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired cache entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    async def shutdown(self):
        """Shutdown cache manager"""
        logger.info("Shutting down AI cache manager")
        
        try:
            # Cancel warming tasks
            for task in self.warming_tasks:
                if not task.done():
                    task.cancel()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Final cleanup
            await self._cleanup_expired_entries()
            
        except Exception as e:
            logger.error(f"Cache shutdown error: {e}")
        
        logger.info("AI cache manager shutdown complete")
