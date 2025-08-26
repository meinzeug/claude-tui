"""
API Gateway integration with existing FastAPI application.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time

from .core import APIGateway, BackendPool, BackendServer, RouteConfig, LoadBalancingStrategy
from .middleware import (
    CircuitBreakerMiddleware,
    RateLimitMiddleware, 
    RequestTransformMiddleware,
    CachingMiddleware
)
from .auth import APIKeyManager, GatewayAuthenticator
from .monitor import GatewayMonitor
from .cache import CacheManager, ResponseCache

logger = logging.getLogger(__name__)


class GatewayMiddleware(BaseHTTPMiddleware):
    """Main gateway middleware that orchestrates all gateway features."""
    
    def __init__(
        self,
        app: ASGIApp,
        gateway: APIGateway,
        monitor: GatewayMonitor,
        response_cache: ResponseCache,
        authenticator: GatewayAuthenticator
    ):
        super().__init__(app)
        self.gateway = gateway
        self.monitor = monitor
        self.response_cache = response_cache
        self.authenticator = authenticator
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main gateway request processing."""
        start_time = time.time()
        error_message = None
        
        try:
            # Skip gateway processing for internal routes
            if self._should_skip_gateway(request):
                return await call_next(request)
            
            # Check cache first
            cached_response = await self.response_cache.get_cached_response(request)
            if cached_response:
                return cached_response
            
            # Authenticate request
            try:
                auth_info = await self.authenticator.authenticate_request(request)
                request.state.auth = auth_info
            except HTTPException as e:
                if not self._is_public_endpoint(request):
                    raise e
            
            # Route through gateway
            response = await self.gateway.route_request(request)
            
            # Cache response if appropriate
            await self.response_cache.cache_response(request, response)
            
            return response
            
        except HTTPException as e:
            error_message = e.detail
            raise e
        except Exception as e:
            error_message = str(e)
            logger.error(f"Gateway middleware error: {e}")
            raise HTTPException(status_code=500, detail="Internal gateway error")
        finally:
            # Record metrics
            response_time = time.time() - start_time
            if hasattr(request.state, 'auth'):  # Only record if we got far enough to auth
                try:
                    # Create a minimal response object for metrics if we don't have one
                    if 'response' not in locals():
                        response = Response(status_code=500)
                    
                    await self.monitor.record_request(
                        request, response, response_time, error_message
                    )
                except Exception as e:
                    logger.error(f"Metrics recording error: {e}")
    
    def _should_skip_gateway(self, request: Request) -> bool:
        """Check if request should bypass gateway processing."""
        skip_paths = [
            "/health",
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/api/gateway",  # Gateway management endpoints
            "/static"
        ]
        
        path = request.url.path
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    def _is_public_endpoint(self, request: Request) -> bool:
        """Check if endpoint is public (doesn't require authentication)."""
        public_endpoints = [
            "/api/v1/auth/token",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        
        path = request.url.path
        return any(path.startswith(endpoint) for endpoint in public_endpoints)


def setup_api_gateway(
    app: FastAPI,
    redis_url: str = "redis://localhost:6379",
    jwt_secret: str = "your-jwt-secret",
    enable_monitoring: bool = True
) -> APIGateway:
    """Setup and configure API Gateway for FastAPI application."""
    
    logger.info("Setting up API Gateway...")
    
    # Initialize gateway components
    gateway = APIGateway(app, redis_url, enable_monitoring)
    cache_manager = CacheManager(redis_url)
    response_cache = ResponseCache(cache_manager)
    api_key_manager = APIKeyManager(redis_url)
    authenticator = GatewayAuthenticator(api_key_manager, jwt_secret)
    monitor = GatewayMonitor(redis_url)
    
    # Configure backend pools
    _configure_backend_pools(gateway)
    
    # Configure routes
    _configure_routes(gateway)
    
    # Add middleware stack (order is important!)
    
    # 1. CORS (if not already configured)
    if not any(isinstance(m, CORSMiddleware) for m in app.user_middleware):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
    
    # 2. Circuit Breaker
    app.add_middleware(CircuitBreakerMiddleware, redis_url=redis_url)
    
    # 3. Rate Limiting  
    app.add_middleware(RateLimitMiddleware, redis_url=redis_url)
    
    # 4. Caching
    app.add_middleware(CachingMiddleware, redis_url=redis_url)
    
    # 5. Request/Response Transformation
    app.add_middleware(RequestTransformMiddleware)
    
    # 6. Main Gateway Middleware (should be last)
    app.add_middleware(
        GatewayMiddleware,
        gateway=gateway,
        monitor=monitor,
        response_cache=response_cache,
        authenticator=authenticator
    )
    
    # Add gateway management endpoints
    _add_gateway_endpoints(app, gateway, monitor, api_key_manager, cache_manager)
    
    # Setup startup/shutdown handlers
    @app.on_event("startup")
    async def startup_handler():
        await gateway.initialize()
        await cache_manager.initialize()
        await api_key_manager.initialize()
        await monitor.initialize()
        logger.info("API Gateway initialized successfully")
    
    @app.on_event("shutdown") 
    async def shutdown_handler():
        await gateway.shutdown()
        await monitor.shutdown()
        logger.info("API Gateway shutdown complete")
    
    logger.info("API Gateway setup complete")
    return gateway


def _configure_backend_pools(gateway: APIGateway):
    """Configure backend server pools."""
    
    # Main API backend pool
    main_pool = BackendPool(
        name="main_api",
        servers=[
            BackendServer(
                id="api_server_1",
                host="localhost",
                port=8001,
                weight=100,
                max_connections=500
            ),
            BackendServer(
                id="api_server_2", 
                host="localhost",
                port=8002,
                weight=100,
                max_connections=500
            )
        ],
        strategy=LoadBalancingStrategy.ROUND_ROBIN,
        health_check_enabled=True,
        failover_enabled=True
    )
    
    # AI services backend pool
    ai_pool = BackendPool(
        name="ai_services",
        servers=[
            BackendServer(
                id="ai_server_1",
                host="localhost", 
                port=8003,
                weight=150,
                max_connections=200,
                timeout=60.0  # AI operations can take longer
            )
        ],
        strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
        health_check_enabled=True
    )
    
    # WebSocket backend pool
    websocket_pool = BackendPool(
        name="websocket_services",
        servers=[
            BackendServer(
                id="ws_server_1",
                host="localhost",
                port=8004,
                weight=100,
                max_connections=1000
            )
        ],
        strategy=LoadBalancingStrategy.HASH  # Session affinity
    )
    
    gateway.add_backend_pool(main_pool)
    gateway.add_backend_pool(ai_pool)
    gateway.add_backend_pool(websocket_pool)


def _configure_routes(gateway: APIGateway):
    """Configure gateway routes."""
    
    # Main API routes
    main_routes = [
        "/api/v1/projects",
        "/api/v1/tasks",
        "/api/v1/workflows", 
        "/api/v1/analytics",
        "/api/v1/auth",
        "/api/v1/users"
    ]
    
    for route_path in main_routes:
        route_config = RouteConfig(
            path=route_path,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            backend_pool="main_api",
            auth_required=True,
            rate_limit={"max_requests": 100, "window": 60},
            cache_ttl=300,
            timeout=30.0,
            retries=3,
            circuit_breaker=True
        )
        gateway.add_route(route_config)
    
    # AI services routes
    ai_routes = [
        "/api/v1/ai",
        "/api/v1/ai/advanced"
    ]
    
    for route_path in ai_routes:
        route_config = RouteConfig(
            path=route_path,
            methods=["GET", "POST"],
            backend_pool="ai_services",
            auth_required=True,
            rate_limit={"max_requests": 30, "window": 60},  # More restrictive for AI
            cache_ttl=600,  # Cache AI responses longer
            timeout=60.0,   # AI operations take longer
            retries=2,
            circuit_breaker=True
        )
        gateway.add_route(route_config)
    
    # WebSocket routes
    websocket_config = RouteConfig(
        path="/api/v1/ws",
        methods=["GET"],
        backend_pool="websocket_services",
        auth_required=True,
        rate_limit={"max_requests": 10, "window": 60},
        timeout=300.0,  # Long-lived connections
        retries=1,
        circuit_breaker=False  # Don't break WebSocket connections
    )
    gateway.add_route(websocket_config)
    
    # Public routes (no auth required)
    public_config = RouteConfig(
        path="/health",
        methods=["GET"],
        backend_pool="main_api",
        auth_required=False,
        cache_ttl=60,
        timeout=5.0,
        retries=1,
        circuit_breaker=False
    )
    gateway.add_route(public_config)


def _add_gateway_endpoints(
    app: FastAPI,
    gateway: APIGateway,
    monitor: GatewayMonitor,
    api_key_manager: APIKeyManager,
    cache_manager: CacheManager
):
    """Add gateway management endpoints."""
    
    from fastapi import APIRouter
    from .auth import APIKeyScope, APIKeyStatus
    
    gateway_router = APIRouter(prefix="/api/gateway", tags=["Gateway Management"])
    
    @gateway_router.get("/status")
    async def get_gateway_status():
        """Get gateway status and metrics."""
        return {
            "status": "healthy",
            "metrics": gateway.get_metrics(),
            "timestamp": time.time()
        }
    
    @gateway_router.get("/metrics")
    async def get_gateway_metrics():
        """Get detailed gateway metrics."""
        return await monitor.get_current_metrics()
    
    @gateway_router.get("/metrics/historical")
    async def get_historical_metrics(
        hours: int = 1,
        resolution: str = "hour"
    ):
        """Get historical metrics."""
        from datetime import datetime, timedelta
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        return await monitor.get_historical_metrics(start_time, end_time, resolution)
    
    @gateway_router.get("/cache/stats")
    async def get_cache_stats():
        """Get cache statistics."""
        return await cache_manager.get_stats()
    
    @gateway_router.delete("/cache/clear")
    async def clear_cache(pattern: Optional[str] = None):
        """Clear cache entries."""
        cleared_count = await cache_manager.clear(pattern)
        return {
            "cleared": cleared_count,
            "pattern": pattern or "*"
        }
    
    @gateway_router.post("/api-keys")
    async def create_api_key(
        name: str,
        description: str,
        scopes: List[str],
        expires_in_days: Optional[int] = None
    ):
        """Create new API key."""
        scope_enums = [APIKeyScope(s) for s in scopes]
        
        api_key, key_obj = await api_key_manager.create_api_key(
            name=name,
            description=description,
            scopes=scope_enums,
            expires_in_days=expires_in_days
        )
        
        # Don't return the actual key in the response body
        return {
            "key_id": key_obj.key_id,
            "name": key_obj.name,
            "scopes": scopes,
            "created_at": key_obj.created_at.isoformat(),
            "expires_at": key_obj.expires_at.isoformat() if key_obj.expires_at else None,
            "api_key": api_key  # Only returned once!
        }
    
    @gateway_router.get("/api-keys")
    async def list_api_keys():
        """List all API keys."""
        keys = await api_key_manager.list_api_keys()
        return [key.to_dict() for key in keys]
    
    @gateway_router.delete("/api-keys/{key_id}")
    async def revoke_api_key(key_id: str):
        """Revoke API key."""
        success = await api_key_manager.revoke_api_key(key_id)
        return {"revoked": success, "key_id": key_id}
    
    @gateway_router.get("/routes")
    async def list_routes():
        """List configured routes."""
        return {
            "routes": [
                {
                    "path": path,
                    "methods": config.methods,
                    "backend_pool": config.backend_pool,
                    "auth_required": config.auth_required
                }
                for path, config in gateway.routes.items()
            ]
        }
    
    @gateway_router.get("/backend-pools")
    async def list_backend_pools():
        """List backend pools and their status."""
        return gateway.get_metrics()["server_metrics"]
    
    @gateway_router.get("/alerts")
    async def get_alerts(active_only: bool = True):
        """Get system alerts."""
        return await monitor.get_alerts(active_only)
    
    app.include_router(gateway_router)


# Helper function for easy setup
def create_gateway_app(
    base_app: Optional[FastAPI] = None,
    redis_url: str = "redis://localhost:6379",
    jwt_secret: str = "your-jwt-secret"
) -> FastAPI:
    """Create a FastAPI app with gateway pre-configured."""
    
    if base_app is None:
        base_app = FastAPI(
            title="API Gateway",
            description="Production-grade API Gateway",
            version="1.0.0"
        )
    
    # Setup gateway
    gateway = setup_api_gateway(base_app, redis_url, jwt_secret)
    
    return base_app