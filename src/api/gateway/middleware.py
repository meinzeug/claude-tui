"""
Advanced middleware components for API Gateway.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import json
import hashlib

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    timeout: float = 60.0
    recovery_timeout: float = 30.0
    success_threshold: int = 3  # For half-open to closed transition


class CircuitBreaker:
    """Circuit breaker implementation for backend services."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_request_time = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise HTTPException(status_code=503, detail="Service unavailable - Circuit breaker OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful request."""
        self.last_request_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker transitioned to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker transitioned to OPEN after {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker transitioned back to OPEN from HALF_OPEN")


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker middleware for API Gateway."""
    
    def __init__(
        self, 
        app, 
        redis_url: str = "redis://localhost:6379",
        default_config: Optional[CircuitBreakerConfig] = None
    ):
        super().__init__(app)
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.default_config = default_config or CircuitBreakerConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply circuit breaker protection."""
        try:
            # Initialize Redis if needed
            if not self.redis_client:
                self.redis_client = redis.from_url(self.redis_url)
            
            # Get or create circuit breaker for this service
            service_key = self._get_service_key(request)
            circuit_breaker = self._get_circuit_breaker(service_key)
            
            # Execute request with circuit breaker protection
            return await circuit_breaker.call(call_next, request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Circuit breaker middleware error: {e}")
            return await call_next(request)
    
    def _get_service_key(self, request: Request) -> str:
        """Get service key for circuit breaker identification."""
        # Use path prefix as service identifier
        path_parts = request.url.path.split('/')
        if len(path_parts) >= 4:  # /api/v1/service/...
            return '/'.join(path_parts[:4])
        return request.url.path
    
    def _get_circuit_breaker(self, service_key: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_key not in self.circuit_breakers:
            self.circuit_breakers[service_key] = CircuitBreaker(self.default_config)
        return self.circuit_breakers[service_key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with Redis backend."""
    
    def __init__(
        self,
        app,
        redis_url: str = "redis://localhost:6379",
        default_limit: int = 60,
        default_window: int = 60,
        burst_limit: int = 10,
        burst_window: int = 10
    ):
        super().__init__(app)
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.default_limit = default_limit
        self.default_window = default_window
        self.burst_limit = burst_limit
        self.burst_window = burst_window
        
        # Endpoint-specific rate limits
        self.endpoint_limits = {
            '/api/v1/ai/advanced': {'limit': 30, 'window': 60},
            '/api/v1/projects': {'limit': 100, 'window': 60},
            '/api/v1/tasks': {'limit': 50, 'window': 60}
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply rate limiting."""
        try:
            # Initialize Redis if needed
            if not self.redis_client:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
            
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Get rate limit config for this endpoint
            limit_config = self._get_limit_config(request.url.path)
            
            # Check burst limit
            await self._check_burst_limit(client_id, request.url.path)
            
            # Check main rate limit
            remaining = await self._check_rate_limit(
                client_id, 
                request.url.path,
                limit_config['limit'],
                limit_config['window']
            )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers['X-RateLimit-Limit'] = str(limit_config['limit'])
            response.headers['X-RateLimit-Remaining'] = str(remaining)
            response.headers['X-RateLimit-Reset'] = str(
                int(time.time()) + limit_config['window']
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}")
            return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from JWT token
        auth_header = request.headers.get('authorization', '')
        if auth_header.startswith('Bearer '):
            # In production, decode JWT and extract user ID
            # For now, use a hash of the token
            token = auth_header[7:]
            return hashlib.md5(token.encode()).hexdigest()[:12]
        
        # Fall back to IP address
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return 'unknown'
    
    def _get_limit_config(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for path."""
        for endpoint, config in self.endpoint_limits.items():
            if path.startswith(endpoint):
                return config
        
        return {'limit': self.default_limit, 'window': self.default_window}
    
    async def _check_burst_limit(self, client_id: str, path: str):
        """Check burst rate limit."""
        burst_key = f"burst:{client_id}:{int(time.time() / self.burst_window)}"
        
        current = await self.redis_client.get(burst_key)
        if current and int(current) >= self.burst_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Burst limit exceeded. Max {self.burst_limit} requests per {self.burst_window} seconds.",
                headers={"Retry-After": str(self.burst_window)}
            )
        
        # Increment burst counter
        pipe = self.redis_client.pipeline()
        pipe.incr(burst_key)
        pipe.expire(burst_key, self.burst_window)
        await pipe.execute()
    
    async def _check_rate_limit(
        self, 
        client_id: str, 
        path: str, 
        limit: int, 
        window: int
    ) -> int:
        """Check main rate limit and return remaining requests."""
        window_key = f"rate:{client_id}:{int(time.time() / window)}"
        
        current = await self.redis_client.get(window_key)
        current_count = int(current) if current else 0
        
        if current_count >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {limit} requests per {window} seconds.",
                headers={"Retry-After": str(window)}
            )
        
        # Increment counter
        pipe = self.redis_client.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, window)
        await pipe.execute()
        
        return limit - current_count - 1


class RequestTransformMiddleware(BaseHTTPMiddleware):
    """Request/response transformation middleware."""
    
    def __init__(self, app, transformations: Optional[Dict[str, Dict]] = None):
        super().__init__(app)
        self.transformations = transformations or {}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply request/response transformations."""
        try:
            # Apply request transformation
            transformed_request = await self._transform_request(request)
            
            # Process request
            response = await call_next(transformed_request)
            
            # Apply response transformation
            transformed_response = await self._transform_response(request, response)
            
            return transformed_response
            
        except Exception as e:
            logger.error(f"Transform middleware error: {e}")
            return await call_next(request)
    
    async def _transform_request(self, request: Request) -> Request:
        """Transform incoming request."""
        path = request.url.path
        
        # Apply path-specific transformations
        for pattern, transform_config in self.transformations.items():
            if path.startswith(pattern):
                request_transform = transform_config.get('request')
                if request_transform:
                    # Apply request transformation logic here
                    # This could include header modification, body transformation, etc.
                    pass
        
        return request
    
    async def _transform_response(self, request: Request, response: Response) -> Response:
        """Transform outgoing response."""
        path = request.url.path
        
        # Apply path-specific transformations
        for pattern, transform_config in self.transformations.items():
            if path.startswith(pattern):
                response_transform = transform_config.get('response')
                if response_transform:
                    # Apply response transformation logic here
                    # This could include adding headers, modifying body, etc.
                    response.headers['X-Gateway-Transformed'] = 'true'
        
        return response


class LoadBalancerMiddleware(BaseHTTPMiddleware):
    """Load balancer middleware for distributing requests."""
    
    def __init__(
        self,
        app,
        backend_pools: Dict[str, list],
        health_check_url: str = "/health"
    ):
        super().__init__(app)
        self.backend_pools = backend_pools
        self.health_check_url = health_check_url
        self.current_backends = {}
        self.backend_health = {}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Distribute request across backend pool."""
        try:
            # Select backend for this request
            backend = await self._select_backend(request)
            if backend:
                # Add backend information to request
                request.state.selected_backend = backend
            
            return await call_next(request)
            
        except Exception as e:
            logger.error(f"Load balancer middleware error: {e}")
            return await call_next(request)
    
    async def _select_backend(self, request: Request) -> Optional[str]:
        """Select best backend for request."""
        path = request.url.path
        
        # Find matching backend pool
        pool_name = None
        for pattern in self.backend_pools:
            if path.startswith(pattern):
                pool_name = pattern
                break
        
        if not pool_name:
            return None
        
        # Get healthy backends from pool
        backends = self.backend_pools[pool_name]
        healthy_backends = [b for b in backends if self._is_backend_healthy(b)]
        
        if not healthy_backends:
            return None
        
        # Simple round-robin selection
        if pool_name not in self.current_backends:
            self.current_backends[pool_name] = 0
        
        selected = healthy_backends[self.current_backends[pool_name] % len(healthy_backends)]
        self.current_backends[pool_name] += 1
        
        return selected
    
    def _is_backend_healthy(self, backend: str) -> bool:
        """Check if backend is healthy."""
        # Simple health check - in production this would be more sophisticated
        return self.backend_health.get(backend, True)


class CachingMiddleware(BaseHTTPMiddleware):
    """Response caching middleware with Redis backend."""
    
    def __init__(
        self,
        app,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 300,
        cache_patterns: Optional[Dict[str, int]] = None
    ):
        super().__init__(app)
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = default_ttl
        self.cache_patterns = cache_patterns or {
            '/api/v1/projects': 600,
            '/health': 60
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply response caching."""
        try:
            # Only cache GET requests
            if request.method.upper() != 'GET':
                return await call_next(request)
            
            # Initialize Redis if needed
            if not self.redis_client:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
            
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Try to get from cache
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
            
            # Process request
            response = await call_next(request)
            
            # Cache successful responses
            if response.status_code == 200:
                ttl = self._get_cache_ttl(request.url.path)
                await self._cache_response(cache_key, response, ttl)
            
            return response
            
        except Exception as e:
            logger.error(f"Caching middleware error: {e}")
            return await call_next(request)
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        # Include path, query parameters, and relevant headers
        key_parts = [
            request.url.path,
            request.url.query or "",
            request.headers.get('authorization', '')[:20]  # Partial auth for user-specific caching
        ]
        
        key_string = '|'.join(key_parts)
        return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Get response from cache."""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return Response(
                    content=data['content'],
                    status_code=data['status_code'],
                    headers=data['headers'],
                    media_type=data['media_type']
                )
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Response, ttl: int):
        """Cache response."""
        try:
            # Read response content
            content = b''
            async for chunk in response.body_iterator:
                content += chunk
            
            # Prepare cache data
            cache_data = {
                'content': content.decode('utf-8', errors='ignore'),
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'media_type': response.media_type
            }
            
            # Store in cache
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(cache_data)
            )
            
            # Recreate response with content
            new_response = Response(
                content=content,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type
            )
            new_response.headers['X-Cache'] = 'MISS'
            
            return new_response
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _get_cache_ttl(self, path: str) -> int:
        """Get cache TTL for path."""
        for pattern, ttl in self.cache_patterns.items():
            if path.startswith(pattern):
                return ttl
        
        return self.default_ttl