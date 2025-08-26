"""
Core API Gateway implementation with advanced routing and load balancing.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import random

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    HASH = "hash"
    HEALTH_BASED = "health_based"


class HealthStatus(Enum):
    """Backend health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


@dataclass
class BackendServer:
    """Backend server configuration."""
    id: str
    host: str
    port: int
    weight: int = 100
    max_connections: int = 1000
    timeout: float = 30.0
    retries: int = 3
    health_check_path: str = "/health"
    health_check_interval: int = 30
    status: HealthStatus = HealthStatus.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    last_health_check: Optional[datetime] = None
    response_time_avg: float = 0.0
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_check_url(self) -> str:
        return f"{self.url}{self.health_check_path}"
    
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY
    
    def calculate_score(self) -> float:
        """Calculate server score for load balancing."""
        if not self.is_healthy():
            return 0.0
        
        # Lower response time and connections = higher score
        connection_factor = 1 - (self.current_connections / max(self.max_connections, 1))
        response_factor = 1 / max(self.response_time_avg, 0.001)
        error_factor = 1 / max(self.error_count + 1, 1)
        
        return (self.weight / 100) * connection_factor * response_factor * error_factor


@dataclass 
class RouteConfig:
    """Route configuration for API Gateway."""
    path: str
    methods: List[str]
    backend_pool: str
    auth_required: bool = True
    rate_limit: Optional[Dict[str, int]] = None
    cache_ttl: Optional[int] = None
    transform_request: Optional[str] = None
    transform_response: Optional[str] = None
    timeout: float = 30.0
    retries: int = 3
    circuit_breaker: bool = True


@dataclass
class BackendPool:
    """Backend server pool configuration."""
    name: str
    servers: List[BackendServer]
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_enabled: bool = True
    failover_enabled: bool = True
    session_affinity: bool = False
    round_robin_index: int = 0


class APIGateway:
    """Production-grade API Gateway with advanced features."""
    
    def __init__(
        self,
        app: FastAPI,
        redis_url: str = "redis://localhost:6379",
        enable_monitoring: bool = True,
        health_check_interval: int = 30
    ):
        self.app = app
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.enable_monitoring = enable_monitoring
        self.health_check_interval = health_check_interval
        
        # Gateway state
        self.backend_pools: Dict[str, BackendPool] = {}
        self.routes: Dict[str, RouteConfig] = {}
        self.session = None
        self.health_check_task = None
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'active_connections': 0
        }
    
    async def initialize(self):
        """Initialize gateway components."""
        logger.info("Initializing API Gateway...")
        
        # Initialize Redis
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        # Start health check task
        if self.health_check_interval > 0:
            self.health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
        
        logger.info("API Gateway initialized successfully")
    
    async def shutdown(self):
        """Shutdown gateway components."""
        logger.info("Shutting down API Gateway...")
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("API Gateway shutdown complete")
    
    def add_backend_pool(self, pool: BackendPool):
        """Add a backend server pool."""
        self.backend_pools[pool.name] = pool
        logger.info(f"Added backend pool: {pool.name} with {len(pool.servers)} servers")
    
    def add_route(self, route: RouteConfig):
        """Add a route configuration."""
        self.routes[route.path] = route
        logger.info(f"Added route: {route.path} -> {route.backend_pool}")
    
    async def route_request(self, request: Request) -> Response:
        """Route request through the gateway."""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        self.metrics['active_connections'] += 1
        
        try:
            # Find matching route
            route_config = await self._find_route(request)
            if not route_config:
                raise HTTPException(status_code=404, detail="Route not found")
            
            # Get backend pool
            pool = self.backend_pools.get(route_config.backend_pool)
            if not pool:
                raise HTTPException(status_code=503, detail="Backend pool not available")
            
            # Select backend server
            server = await self._select_backend(pool, request)
            if not server:
                raise HTTPException(status_code=503, detail="No healthy backends available")
            
            # Check rate limits
            if route_config.rate_limit:
                await self._check_rate_limit(request, route_config.rate_limit)
            
            # Check cache
            if route_config.cache_ttl and request.method.upper() == "GET":
                cached_response = await self._get_cached_response(request, route_config.cache_ttl)
                if cached_response:
                    return cached_response
            
            # Transform request
            if route_config.transform_request:
                request = await self._transform_request(request, route_config.transform_request)
            
            # Forward request
            response = await self._forward_request(
                request, server, route_config
            )
            
            # Transform response
            if route_config.transform_response:
                response = await self._transform_response(response, route_config.transform_response)
            
            # Cache response
            if route_config.cache_ttl and request.method.upper() == "GET" and response.status_code == 200:
                await self._cache_response(request, response, route_config.cache_ttl)
            
            # Update metrics
            response_time = time.time() - start_time
            await self._update_metrics(server, response_time, True)
            
            self.metrics['successful_requests'] += 1
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Gateway error: {e}")
            self.metrics['failed_requests'] += 1
            raise HTTPException(status_code=500, detail="Gateway error")
        finally:
            self.metrics['active_connections'] -= 1
            
            # Update average response time
            total_time = time.time() - start_time
            self.metrics['avg_response_time'] = (
                self.metrics['avg_response_time'] * 0.9 + total_time * 0.1
            )
    
    async def _find_route(self, request: Request) -> Optional[RouteConfig]:
        """Find matching route configuration."""
        path = request.url.path
        method = request.method.upper()
        
        # Exact match first
        if path in self.routes and method in self.routes[path].methods:
            return self.routes[path]
        
        # Pattern matching (simple prefix matching for now)
        for route_path, route_config in self.routes.items():
            if path.startswith(route_path.rstrip('*')) and method in route_config.methods:
                return route_config
        
        return None
    
    async def _select_backend(self, pool: BackendPool, request: Request) -> Optional[BackendServer]:
        """Select backend server based on load balancing strategy."""
        healthy_servers = [s for s in pool.servers if s.is_healthy()]
        if not healthy_servers:
            return None
        
        if pool.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            server = healthy_servers[pool.round_robin_index % len(healthy_servers)]
            pool.round_robin_index += 1
            return server
        
        elif pool.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            total_weight = sum(s.weight for s in healthy_servers)
            random_weight = random.randint(0, total_weight - 1)
            weight_sum = 0
            for server in healthy_servers:
                weight_sum += server.weight
                if weight_sum > random_weight:
                    return server
        
        elif pool.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_servers, key=lambda s: s.current_connections)
        
        elif pool.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_servers)
        
        elif pool.strategy == LoadBalancingStrategy.HASH:
            client_ip = request.client.host if request.client else "unknown"
            hash_value = hashlib.md5(client_ip.encode()).hexdigest()
            index = int(hash_value, 16) % len(healthy_servers)
            return healthy_servers[index]
        
        elif pool.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return max(healthy_servers, key=lambda s: s.calculate_score())
        
        return healthy_servers[0]
    
    async def _forward_request(
        self, 
        request: Request, 
        server: BackendServer, 
        route_config: RouteConfig
    ) -> Response:
        """Forward request to backend server."""
        if not self.session:
            raise HTTPException(status_code=503, detail="Gateway not initialized")
        
        # Build target URL
        target_url = f"{server.url}{request.url.path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers.pop('host', None)  # Remove host header
        headers['X-Forwarded-For'] = request.client.host if request.client else "unknown"
        headers['X-Forwarded-Proto'] = request.url.scheme
        headers['X-Gateway-Server'] = server.id
        
        # Get request body
        body = await request.body() if request.method.upper() in ['POST', 'PUT', 'PATCH'] else None
        
        server.current_connections += 1
        try:
            async with self.session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body,
                timeout=aiohttp.ClientTimeout(total=route_config.timeout)
            ) as response:
                # Read response content
                content = await response.read()
                
                # Create FastAPI response
                return Response(
                    content=content,
                    status_code=response.status,
                    headers=dict(response.headers),
                    media_type=response.headers.get('content-type', 'application/json')
                )
                
        except asyncio.TimeoutError:
            server.error_count += 1
            raise HTTPException(status_code=504, detail="Backend timeout")
        except Exception as e:
            server.error_count += 1
            logger.error(f"Backend request failed: {e}")
            raise HTTPException(status_code=502, detail="Backend error")
        finally:
            server.current_connections -= 1
            server.total_requests += 1
    
    async def _check_rate_limit(self, request: Request, rate_limit: Dict[str, int]):
        """Check rate limiting."""
        if not self.redis_client:
            return
        
        client_ip = request.client.host if request.client else "unknown"
        window = rate_limit.get('window', 60)
        max_requests = rate_limit.get('max_requests', 100)
        
        key = f"rate_limit:{client_ip}:{int(time.time() / window)}"
        
        current = await self.redis_client.get(key)
        if current and int(current) >= max_requests:
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded",
                headers={"Retry-After": str(window)}
            )
        
        pipe = self.redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        await pipe.execute()
    
    async def _get_cached_response(self, request: Request, ttl: int) -> Optional[Response]:
        """Get cached response."""
        if not self.redis_client:
            return None
        
        cache_key = f"cache:{hashlib.md5(str(request.url).encode()).hexdigest()}"
        cached = await self.redis_client.get(cache_key)
        
        if cached:
            data = json.loads(cached)
            return Response(
                content=data['content'],
                status_code=data['status_code'],
                headers=data['headers']
            )
        
        return None
    
    async def _cache_response(self, request: Request, response: Response, ttl: int):
        """Cache response."""
        if not self.redis_client:
            return
        
        cache_key = f"cache:{hashlib.md5(str(request.url).encode()).hexdigest()}"
        data = {
            'content': response.body.decode() if hasattr(response, 'body') else '',
            'status_code': response.status_code,
            'headers': dict(response.headers)
        }
        
        await self.redis_client.setex(cache_key, ttl, json.dumps(data))
    
    async def _transform_request(self, request: Request, transform: str) -> Request:
        """Transform request (placeholder for custom transformations)."""
        # This would implement custom request transformations
        return request
    
    async def _transform_response(self, response: Response, transform: str) -> Response:
        """Transform response (placeholder for custom transformations)."""
        # This would implement custom response transformations  
        return response
    
    async def _update_metrics(self, server: BackendServer, response_time: float, success: bool):
        """Update server metrics."""
        # Update server response time average
        server.response_time_avg = (server.response_time_avg * 0.9 + response_time * 0.1)
        
        if not success:
            server.error_count += 1
        
        # Store metrics in Redis for persistence
        if self.redis_client:
            metrics_key = f"metrics:server:{server.id}"
            await self.redis_client.hset(metrics_key, mapping={
                'response_time_avg': server.response_time_avg,
                'error_count': server.error_count,
                'total_requests': server.total_requests,
                'current_connections': server.current_connections
            })
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all backend servers."""
        if not self.session:
            return
        
        for pool in self.backend_pools.values():
            if not pool.health_check_enabled:
                continue
            
            for server in pool.servers:
                try:
                    start_time = time.time()
                    async with self.session.get(
                        server.health_check_url,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            if server.status != HealthStatus.HEALTHY:
                                logger.info(f"Server {server.id} is now healthy")
                            server.status = HealthStatus.HEALTHY
                            server.response_time_avg = response_time
                        else:
                            if server.status != HealthStatus.UNHEALTHY:
                                logger.warning(f"Server {server.id} is unhealthy (status: {response.status})")
                            server.status = HealthStatus.UNHEALTHY
                            
                except Exception as e:
                    if server.status != HealthStatus.UNHEALTHY:
                        logger.error(f"Server {server.id} health check failed: {e}")
                    server.status = HealthStatus.UNHEALTHY
                
                server.last_health_check = datetime.utcnow()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics."""
        server_metrics = {}
        for pool_name, pool in self.backend_pools.items():
            server_metrics[pool_name] = {
                'servers': [asdict(server) for server in pool.servers],
                'healthy_count': len([s for s in pool.servers if s.is_healthy()]),
                'total_count': len(pool.servers)
            }
        
        return {
            'gateway_metrics': self.metrics,
            'server_metrics': server_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }