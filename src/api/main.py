"""
FastAPI main application for Claude TIU API.

Provides REST API endpoints for AI-powered development operations.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
from typing import Dict, Any
import logging

from api.routes import auth, users, commands, plugins, themes
from api.v1 import projects, tasks, validation, ai, community, ai_advanced, workflows, analytics, websocket
from api.middleware.security import SecurityMiddleware
from api.middleware.logging import LoggingMiddleware
from api.middleware.caching import setup_cache_middleware
from api.middleware.compression import setup_compression_middleware
from api.dependencies.database import get_database
from api.models.base import init_database


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Claude TIU API...")
    await init_database()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down Claude TIU API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Claude TIU API",
        description="AI-Powered Development Tool REST API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # High-Performance Middleware Stack (order matters!)
    
    # 1. Compression (first to compress all responses)
    setup_compression_middleware(
        app, 
        minimum_size=500,
        gzip_level=6,
        brotli_level=4,
        enable_streaming=True
    )
    
    # 2. Caching (before security to cache public responses)
    setup_cache_middleware(
        app,
        redis_url="redis://localhost:6379",
        default_ttl=300  # 5 minutes default
    )
    
    # 3. Security middleware
    app.add_middleware(SecurityMiddleware)
    
    # 4. Logging (last to log final response)
    app.add_middleware(LoggingMiddleware)

    # Include Routers
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(commands.router, prefix="/api/v1/commands", tags=["Commands"])
    app.include_router(plugins.router, prefix="/api/v1/plugins", tags=["Plugins"])
    app.include_router(themes.router, prefix="/api/v1/themes", tags=["Themes"])
    
    # Include v1 API endpoints
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects"])
    app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])
    app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["Workflows"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(websocket.router, prefix="/api/v1/websocket", tags=["WebSocket"])
    app.include_router(validation.router, prefix="/api/v1/validation", tags=["Validation"])
    app.include_router(ai.router, prefix="/api/v1/ai", tags=["AI Integration"])
    app.include_router(ai_advanced.router, prefix="/api/v1", tags=["AI Advanced Services"])
    app.include_router(community.router, prefix="/api/v1", tags=["Community"])
    
    # High-Performance Endpoints
    from api.v1 import performance
    app.include_router(performance.router, prefix="/api/v1/performance", tags=["High Performance"])

    # Health Check Endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Welcome to Claude TIU API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc"
        }

    # Custom OpenAPI Schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="Claude TIU API",
            version="1.0.0",
            description="""
AI-Powered Development Tool REST API with comprehensive endpoints for:

## Core Features
- **Project Management**: Create, manage, validate, backup, and restore AI-powered development projects
- **Task Orchestration**: Execute complex development tasks with AI assistance and dependency management
- **Workflow Engine**: Advanced workflow orchestration with multi-agent coordination and real-time monitoring
- **Code Validation**: Anti-hallucination validation and quality assurance with automated fixing
- **AI Integration**: Seamless Claude Code and Claude Flow integration with intelligent routing
- **Real-time Communication**: WebSocket support for live progress updates and notifications
- **Analytics & Monitoring**: Comprehensive performance tracking, usage analytics, and trend analysis
- **Community Platform**: Template marketplace, user profiles, ratings, reviews, and collaborative development

## Extended Capabilities  
- **Advanced Project Operations**: Duplication, dependency analysis, health monitoring, analytics
- **Workflow Management**: Multi-step workflow creation, execution, pause/resume, cancellation
- **Real-time Updates**: WebSocket subscriptions for task progress, workflow status, agent coordination
- **Analytics Dashboard**: Performance metrics, usage analytics, trend analysis, custom reports
- **Community Features**: User reputation system, leaderboards, events, discussions, content moderation
- **Backup & Recovery**: Project backup creation, listing, and restoration with compression options

## Authentication & Security
- JWT token authentication with role-based access control
- Rate limiting protection with configurable windows
- Request validation and sanitization
- Comprehensive error handling and logging
- Security middleware for input validation

## Key Capabilities
- AI-powered code generation and review with quality assurance
- Multi-agent workflow orchestration with adaptive strategies
- Project structure validation with automated health checks
- Task dependency management with intelligent scheduling
- Progress authenticity verification with anti-hallucination detection
- Performance monitoring and analytics with predictive insights
- Real-time progress tracking via WebSocket connections
- Batch operations support for high-throughput scenarios
- Community-driven template sharing and collaboration

## Technical Standards
- OpenAPI 3.0 compliant with comprehensive schemas
- RESTful design principles with consistent patterns
- WebSocket support for real-time communication
- Comprehensive documentation with examples
- Response caching and pagination support
- Health checks and monitoring endpoints
- Multi-format export capabilities (JSON, CSV, PDF)
- Async/await patterns for optimal performance

## API Organization
- **/projects**: Project management and operations
- **/tasks**: Task creation, execution, and monitoring  
- **/workflows**: Advanced workflow orchestration
- **/websocket**: Real-time communication endpoints
- **/analytics**: Performance metrics and insights
- **/community**: Social features and marketplace
- **/ai**: AI service integration and management
            """,
            routes=app.routes,
        )
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    # Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "status_code": 500}
        )

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )