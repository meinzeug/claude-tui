"""
Advanced caching layer for API Gateway.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
from fastapi import Request, Response

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    READ_THROUGH = "read_through"


@dataclass
class CacheConfig:
    """Cache configuration."""
    strategy: CacheStrategy = CacheStrategy.TTL
    default_ttl: int = 300  # 5 minutes
    max_memory: str = "100mb"
    eviction_policy: str = "allkeys-lru"
    key_prefix: str = "api_cache:"
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Compress if larger than 1KB


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    content_type: Optional[str] = None
    compressed: bool = False
    tags: List[str] = None


class CacheManager:
    """Advanced cache manager with multiple strategies."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: Optional[CacheConfig] = None
    ):
        self.redis_url = redis_url
        self.config = config or CacheConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    async def initialize(self):
        """Initialize cache manager."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Configure Redis for caching
        await self.redis_client.config_set('maxmemory', self.config.max_memory)
        await self.redis_client.config_set('maxmemory-policy', self.config.eviction_policy)
        
        logger.info(f"Cache Manager initialized with {self.config.strategy.value} strategy")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        if not self.redis_client:
            return default
        
        try:
            cache_key = f"{self.config.key_prefix}{key}"
            
            # Get value and metadata
            pipe = self.redis_client.pipeline()
            pipe.hgetall(cache_key)
            pipe.ttl(cache_key)
            result = await pipe.execute()
            
            cached_data, ttl = result
            
            if not cached_data:
                self.stats['misses'] += 1
                return default
            
            # Check if expired (additional check beyond Redis TTL)
            if ttl <= 0:
                await self.delete(key)
                self.stats['misses'] += 1
                return default
            
            # Update access statistics
            await self._update_access_stats(cache_key)
            
            # Deserialize value
            value = await self._deserialize_value(cached_data)
            self.stats['hits'] += 1
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats['misses'] += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        content_type: Optional[str] = None
    ) -> bool:
        """Set value in cache."""
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"{self.config.key_prefix}{key}"
            ttl = ttl or self.config.default_ttl
            
            # Serialize and potentially compress value
            serialized_data = await self._serialize_value(
                value, content_type or 'application/json'
            )
            
            # Prepare cache entry metadata
            now = datetime.utcnow()
            entry_data = {
                'value': serialized_data['data'],
                'created_at': now.isoformat(),
                'expires_at': (now + timedelta(seconds=ttl)).isoformat(),
                'access_count': '0',
                'content_type': content_type or 'application/json',
                'compressed': '1' if serialized_data['compressed'] else '0'
            }
            
            if tags:
                entry_data['tags'] = json.dumps(tags)
            
            # Store in Redis
            pipe = self.redis_client.pipeline()
            pipe.hmset(cache_key, entry_data)
            pipe.expire(cache_key, ttl)
            
            # Add to tag indices if tags provided
            if tags:
                for tag in tags:
                    tag_key = f"{self.config.key_prefix}tag:{tag}"
                    pipe.sadd(tag_key, key)
                    pipe.expire(tag_key, ttl)
            
            await pipe.execute()
            
            self.stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"{self.config.key_prefix}{key}"
            
            # Get tags before deletion for cleanup
            cached_data = await self.redis_client.hgetall(cache_key)
            tags = []
            if cached_data and b'tags' in cached_data:
                tags = json.loads(cached_data[b'tags'].decode())
            
            # Delete main entry
            deleted = await self.redis_client.delete(cache_key)
            
            # Clean up tag indices
            if tags:
                pipe = self.redis_client.pipeline()
                for tag in tags:
                    tag_key = f"{self.config.key_prefix}tag:{tag}"
                    pipe.srem(tag_key, key)
                await pipe.execute()
            
            if deleted:
                self.stats['deletes'] += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def delete_by_tag(self, tag: str) -> int:
        """Delete all entries with specific tag."""
        if not self.redis_client:
            return 0
        
        try:
            tag_key = f"{self.config.key_prefix}tag:{tag}"
            keys_to_delete = await self.redis_client.smembers(tag_key)
            
            if not keys_to_delete:
                return 0
            
            # Delete all keys with this tag
            cache_keys = [f"{self.config.key_prefix}{key.decode()}" for key in keys_to_delete]
            deleted_count = await self.redis_client.delete(*cache_keys)
            
            # Delete tag index
            await self.redis_client.delete(tag_key)
            
            self.stats['deletes'] += deleted_count
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache delete by tag error: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"{self.config.key_prefix}{key}"
            return await self.redis_client.exists(cache_key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            if pattern:
                search_pattern = f"{self.config.key_prefix}{pattern}"
            else:
                search_pattern = f"{self.config.key_prefix}*"
            
            keys = await self.redis_client.keys(search_pattern)
            if keys:
                deleted_count = await self.redis_client.delete(*keys)
                self.stats['deletes'] += deleted_count
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return self.stats
        
        try:
            # Get Redis memory info
            redis_info = await self.redis_client.info('memory')
            
            # Calculate hit rate
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / max(total_requests, 1)) * 100
            
            stats = {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'memory_used': redis_info.get('used_memory_human', '0B'),
                'memory_peak': redis_info.get('used_memory_peak_human', '0B'),
                'keys_count': await self.redis_client.dbsize(),
                'config': {
                    'strategy': self.config.strategy.value,
                    'default_ttl': self.config.default_ttl,
                    'max_memory': self.config.max_memory,
                    'compression_enabled': self.config.compression_enabled
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return self.stats
    
    async def _serialize_value(self, value: Any, content_type: str) -> Dict[str, Any]:
        """Serialize and optionally compress value."""
        try:
            # Serialize based on content type
            if content_type == 'application/json':
                serialized = json.dumps(value, default=str)
            else:
                serialized = str(value)
            
            # Check if compression is beneficial
            should_compress = (
                self.config.compression_enabled and
                len(serialized.encode()) > self.config.compression_threshold
            )
            
            if should_compress:
                import gzip
                compressed = gzip.compress(serialized.encode())
                return {
                    'data': compressed,
                    'compressed': True
                }
            
            return {
                'data': serialized.encode(),
                'compressed': False
            }
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return {'data': b'', 'compressed': False}
    
    async def _deserialize_value(self, cached_data: Dict) -> Any:
        """Deserialize and decompress value."""
        try:
            value_bytes = cached_data[b'value']
            compressed = cached_data.get(b'compressed', b'0') == b'1'
            content_type = cached_data.get(b'content_type', b'application/json').decode()
            
            # Decompress if needed
            if compressed:
                import gzip
                value_bytes = gzip.decompress(value_bytes)
            
            # Deserialize based on content type
            if content_type == 'application/json':
                return json.loads(value_bytes.decode())
            else:
                return value_bytes.decode()
            
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    async def _update_access_stats(self, cache_key: str):
        """Update access statistics for cache entry."""
        try:
            pipe = self.redis_client.pipeline()
            pipe.hincrby(cache_key, 'access_count', 1)
            pipe.hset(cache_key, 'last_accessed', datetime.utcnow().isoformat())
            await pipe.execute()
        except Exception as e:
            logger.error(f"Access stats update error: {e}")


class ResponseCache:
    """HTTP response caching system."""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        default_cache_control: str = "public, max-age=300"
    ):
        self.cache_manager = cache_manager
        self.default_cache_control = default_cache_control
    
    def generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        # Include method, path, query params, and relevant headers
        key_components = [
            request.method,
            request.url.path,
            request.url.query or "",
            request.headers.get('accept', ''),
            request.headers.get('accept-encoding', ''),
            # Include user-specific info for personalized content
            request.headers.get('authorization', '')[:20] if request.headers.get('authorization') else ''
        ]
        
        key_string = '|'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get_cached_response(self, request: Request) -> Optional[Response]:
        """Get cached response for request."""
        if request.method.upper() not in ['GET', 'HEAD']:
            return None
        
        cache_key = self.generate_cache_key(request)
        cached_data = await self.cache_manager.get(cache_key)
        
        if not cached_data:
            return None
        
        try:
            # Reconstruct response
            response = Response(
                content=cached_data['content'],
                status_code=cached_data['status_code'],
                headers=cached_data['headers'],
                media_type=cached_data['media_type']
            )
            
            # Add cache headers
            response.headers['X-Cache'] = 'HIT'
            response.headers['X-Cache-Key'] = cache_key[:16]
            
            return response
            
        except Exception as e:
            logger.error(f"Response reconstruction error: {e}")
            return None
    
    async def cache_response(
        self,
        request: Request,
        response: Response,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Cache response."""
        if request.method.upper() not in ['GET', 'HEAD']:
            return False
        
        if response.status_code not in [200, 203, 300, 301, 302, 404, 410]:
            return False
        
        # Check cache-control headers
        cache_control = response.headers.get('cache-control', '')
        if 'no-cache' in cache_control or 'no-store' in cache_control:
            return False
        
        try:
            cache_key = self.generate_cache_key(request)
            
            # Read response content
            content = b''
            if hasattr(response, 'body_iterator'):
                async for chunk in response.body_iterator:
                    content += chunk
            elif hasattr(response, 'body'):
                content = response.body
            
            # Prepare cache data
            cache_data = {
                'content': content.decode('utf-8', errors='ignore'),
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'media_type': response.media_type or 'application/json'
            }
            
            # Determine TTL from cache-control or use default
            if ttl is None:
                ttl = self._parse_cache_control_ttl(cache_control) or 300
            
            # Add default tags
            if tags is None:
                tags = []
            
            tags.extend([
                f"path:{request.url.path.split('/')[1:3]}",  # First two path segments
                f"status:{response.status_code}",
                f"method:{request.method}"
            ])
            
            success = await self.cache_manager.set(
                cache_key,
                cache_data,
                ttl=ttl,
                tags=tags,
                content_type='application/json'
            )
            
            if success:
                # Add cache headers to original response
                response.headers['X-Cache'] = 'MISS'
                response.headers['X-Cache-Key'] = cache_key[:16]
                response.headers['X-Cache-TTL'] = str(ttl)
            
            return success
            
        except Exception as e:
            logger.error(f"Response caching error: {e}")
            return False
    
    async def invalidate_cache(
        self,
        pattern: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """Invalidate cached responses."""
        total_deleted = 0
        
        if tags:
            for tag in tags:
                deleted = await self.cache_manager.delete_by_tag(tag)
                total_deleted += deleted
        
        if pattern:
            deleted = await self.cache_manager.clear(pattern)
            total_deleted += deleted
        
        return total_deleted
    
    def _parse_cache_control_ttl(self, cache_control: str) -> Optional[int]:
        """Parse TTL from cache-control header."""
        if not cache_control:
            return None
        
        for directive in cache_control.split(','):
            directive = directive.strip()
            if directive.startswith('max-age='):
                try:
                    return int(directive.split('=')[1])
                except (ValueError, IndexError):
                    continue
        
        return None


class SmartCacheWarming:
    """Intelligent cache warming system."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.warming_stats = {
            'warmed_keys': 0,
            'warming_errors': 0,
            'last_warming': None
        }
    
    async def warm_cache(
        self,
        popular_endpoints: List[str],
        sample_requests: List[Dict[str, Any]]
    ):
        """Warm cache with popular endpoints and sample requests."""
        logger.info(f"Starting cache warming for {len(popular_endpoints)} endpoints")
        
        for endpoint in popular_endpoints:
            try:
                # Generate sample cache entries for popular endpoints
                await self._warm_endpoint(endpoint, sample_requests)
                self.warming_stats['warmed_keys'] += 1
                
            except Exception as e:
                logger.error(f"Cache warming error for {endpoint}: {e}")
                self.warming_stats['warming_errors'] += 1
        
        self.warming_stats['last_warming'] = datetime.utcnow().isoformat()
        logger.info(f"Cache warming completed: {self.warming_stats}")
    
    async def _warm_endpoint(self, endpoint: str, sample_requests: List[Dict[str, Any]]):
        """Warm cache for specific endpoint."""
        # This would integrate with actual endpoint handlers to generate responses
        # For now, we'll create placeholder cache entries
        
        for sample in sample_requests:
            if endpoint in sample.get('path', ''):
                cache_key = f"warm_{endpoint}_{hash(str(sample))}"
                
                # Create mock response data
                mock_response = {
                    'data': {'message': f'Warmed response for {endpoint}'},
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.cache_manager.set(
                    cache_key,
                    mock_response,
                    ttl=600,  # 10 minutes
                    tags=[f"endpoint:{endpoint}", "warmed"]
                )