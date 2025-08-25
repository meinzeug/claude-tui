"""
High-Performance Caching Middleware for API Response Optimization.

Implements Redis-based caching with intelligent TTL and cache invalidation
to achieve <500ms API response times.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Union
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import gzip
import pickle

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class HighPerformanceCacheMiddleware(BaseHTTPMiddleware):
    """
    High-performance caching middleware with Redis backend.
    
    Features:
    - Intelligent cache key generation
    - Configurable TTL per endpoint
    - Response compression
    - Cache warming and preloading
    - Memory fallback when Redis unavailable
    """
    
    def __init__(self, app, redis_url: str = "redis://localhost:6379", default_ttl: int = 300):
        super().__init__(app)
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        
        # Cache configuration per endpoint
        self.endpoint_config = {
            '/api/v1/ai/code/generate': {'ttl': 3600, 'compress': True},
            '/api/v1/ai/validate': {'ttl': 1800, 'compress': False},
            '/api/v1/ai/performance': {'ttl': 60, 'compress': False},
            '/api/v1/tasks/': {'ttl': 300, 'compress': True},
            '/api/v1/projects/': {'ttl': 600, 'compress': True}
        }
        
    async def setup_redis(self):
        """Initialize Redis connection with error handling."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using memory cache fallback")
            return
            
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=False,  # Handle binary data for compression
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch with intelligent caching."""
        # Initialize Redis on first request
        if REDIS_AVAILABLE and self.redis_client is None:
            await self.setup_redis()
        
        # Skip caching for non-GET requests or specific paths
        if request.method != "GET" or self._should_skip_cache(request):
            return await call_next(request)
        
        # Generate cache key
        cache_key = await self._generate_cache_key(request)
        
        # Try to get from cache
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            self.cache_stats['hits'] += 1
            logger.debug(f"Cache HIT for {request.url.path}")
            return await self._create_cached_response(cached_response, request)
        
        # Execute request
        start_time = time.time()
        response = await call_next(request)
        execution_time = time.time() - start_time
        
        # Cache successful responses
        if response.status_code == 200:
            await self._cache_response(cache_key, response, request, execution_time)
            self.cache_stats['misses'] += 1
            logger.debug(f"Cache MISS for {request.url.path} (executed in {execution_time:.3f}s)")
        
        return response
    
    def _should_skip_cache(self, request: Request) -> bool:
        """Determine if request should skip caching."""
        skip_paths = ['/health', '/docs', '/redoc', '/openapi.json']
        
        # Skip if path in skip list
        if any(request.url.path.startswith(path) for path in skip_paths):
            return True
        
        # Skip if has auth headers (user-specific data)
        if 'authorization' in request.headers:
            return True
        
        # Skip if has cache-control: no-cache
        cache_control = request.headers.get('cache-control', '')
        if 'no-cache' in cache_control.lower():
            return True
        
        return False
    
    async def _generate_cache_key(self, request: Request) -> str:
        """Generate intelligent cache key from request."""
        # Base components
        path = request.url.path
        query_params = str(sorted(request.query_params.items()))
        
        # Include relevant headers
        relevant_headers = ['accept', 'accept-encoding', 'user-agent']
        header_parts = []
        for header in relevant_headers:
            if header in request.headers:
                header_parts.append(f"{header}:{request.headers[header]}")
        
        # Create hash
        key_data = f"{path}:{query_params}:{'|'.join(header_parts)}"
        cache_key = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        
        return f"claude_tui:api:{cache_key}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve response from cache (Redis or memory fallback)."""
        try:
            if self.redis_client:
                # Try Redis first
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)
            
            # Fallback to memory cache
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                
                # Check if expired
                if time.time() - cached_item['timestamp'] < cached_item['ttl']:
                    return cached_item['data']
                else:
                    del self.memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            self.cache_stats['errors'] += 1
            return None
    
    async def _cache_response(
        self, 
        cache_key: str, 
        response: Response, 
        request: Request,
        execution_time: float
    ):
        """Cache response with intelligent TTL and compression."""
        try:
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Get endpoint-specific config
            config = self._get_endpoint_config(request.url.path)
            ttl = config['ttl']
            should_compress = config['compress']
            
            # Prepare cache data
            cache_data = {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'body': response_body.decode('utf-8') if response_body else '',
                'content_type': response.headers.get('content-type', 'application/json'),
                'execution_time': execution_time,
                'timestamp': time.time()
            }
            
            # Compress if configured
            if should_compress and len(cache_data['body']) > 1024:
                compressed_body = gzip.compress(cache_data['body'].encode())
                cache_data['body'] = compressed_body
                cache_data['compressed'] = True
            
            # Cache in Redis
            if self.redis_client:
                serialized_data = pickle.dumps(cache_data)
                await self.redis_client.setex(cache_key, ttl, serialized_data)
            else:
                # Fallback to memory cache with size limit
                if len(self.memory_cache) > 1000:  # Limit memory cache size
                    # Remove oldest entry
                    oldest_key = min(
                        self.memory_cache.keys(), 
                        key=lambda k: self.memory_cache[k]['timestamp']
                    )
                    del self.memory_cache[oldest_key]
                
                self.memory_cache[cache_key] = {
                    'data': cache_data,
                    'ttl': ttl,
                    'timestamp': time.time()
                }
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            self.cache_stats['errors'] += 1
    
    def _get_endpoint_config(self, path: str) -> Dict[str, Any]:
        """Get caching configuration for specific endpoint."""
        for endpoint_path, config in self.endpoint_config.items():
            if path.startswith(endpoint_path):
                return config
        
        return {'ttl': self.default_ttl, 'compress': False}
    
    async def _create_cached_response(
        self, 
        cached_data: Dict[str, Any], 
        request: Request
    ) -> Response:
        """Create Response object from cached data."""
        try:
            # Handle compressed data
            body = cached_data['body']
            if cached_data.get('compressed'):
                if isinstance(body, bytes):
                    body = gzip.decompress(body).decode('utf-8')
                else:
                    body = gzip.decompress(body.encode()).decode('utf-8')
            
            # Create response
            response = Response(
                content=body,
                status_code=cached_data['status_code'],
                media_type=cached_data.get('content_type', 'application/json')
            )
            
            # Add cache headers
            response.headers['X-Cache'] = 'HIT'
            response.headers['X-Cache-Age'] = str(int(time.time() - cached_data['timestamp']))
            response.headers['X-Original-Execution-Time'] = str(cached_data.get('execution_time', 0))
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating cached response: {e}")
            self.cache_stats['errors'] += 1
            raise
    
    async def clear_cache(self, pattern: str = None):
        """Clear cache (for administrative purposes)."""
        try:
            if self.redis_client:
                if pattern:
                    # Clear matching keys
                    keys = await self.redis_client.keys(f"claude_tui:api:{pattern}*")
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    # Clear all API cache
                    keys = await self.redis_client.keys("claude_tui:api:*")
                    if keys:
                        await self.redis_client.delete(*keys)
            
            # Clear memory cache
            if pattern:
                keys_to_delete = [
                    k for k in self.memory_cache.keys() 
                    if pattern in k
                ]
                for key in keys_to_delete:
                    del self.memory_cache[key]
            else:
                self.memory_cache.clear()
            
            logger.info(f"Cache cleared: pattern={pattern}")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'errors': self.cache_stats['errors'],
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'redis_available': self.redis_client is not None,
            'memory_cache_size': len(self.memory_cache)
        }


# Global cache instance
cache_middleware = None

def get_cache_middleware() -> HighPerformanceCacheMiddleware:
    """Get global cache middleware instance."""
    global cache_middleware
    return cache_middleware

def setup_cache_middleware(app, redis_url: str = "redis://localhost:6379", default_ttl: int = 300):
    """Setup cache middleware for FastAPI app."""
    global cache_middleware
    cache_middleware = HighPerformanceCacheMiddleware(app, redis_url, default_ttl)
    app.add_middleware(HighPerformanceCacheMiddleware, redis_url=redis_url, default_ttl=default_ttl)
    return cache_middleware