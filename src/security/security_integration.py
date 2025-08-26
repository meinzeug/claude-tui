"""
Security Integration Module for Claude-TUI

This module integrates all security components and provides a unified interface
for initializing and managing security features across the application.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .redis_token_blacklist import RedisTokenBlacklist, init_token_blacklist
from .input_validation_middleware import InputValidationMiddleware, ValidationRule
from .csrf_protection import CSRFProtectionMiddleware
from ..auth.enhanced_jwt_auth import EnhancedJWTAuthenticator, init_enhanced_jwt_authenticator
from ..auth.security_config import get_security_config, SecurityConfig

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Centralized Security Manager for Claude-TUI.
    
    Manages initialization and coordination of all security components:
    - JWT Authentication with Redis blacklist
    - Input validation middleware
    - CSRF protection
    - Security headers
    - Rate limiting
    - Audit logging
    """
    
    def __init__(self):
        self.config: Optional[SecurityConfig] = None
        self.jwt_auth: Optional[EnhancedJWTAuthenticator] = None
        self.token_blacklist: Optional[RedisTokenBlacklist] = None
        self.initialized = False
        self._security_metrics = {
            "initialization_time": None,
            "components_initialized": [],
            "components_failed": [],
            "health_checks_passed": 0,
            "security_events": 0
        }
    
    async def initialize(self, config: Optional[SecurityConfig] = None) -> bool:
        """
        Initialize all security components.
        
        Args:
            config: Security configuration (will use default if not provided)
            
        Returns:
            True if initialization was successful
        """
        try:
            logger.info("🔒 Initializing Claude-TUI Security Manager...")
            start_time = asyncio.get_event_loop().time()
            
            # Load configuration
            self.config = config or get_security_config()
            self._security_metrics["components_initialized"].append("config")
            
            # Initialize Redis token blacklist
            success = await self._init_token_blacklist()
            if success:
                self._security_metrics["components_initialized"].append("token_blacklist")
            else:
                self._security_metrics["components_failed"].append("token_blacklist")
            
            # Initialize enhanced JWT authenticator
            success = await self._init_jwt_authenticator()
            if success:
                self._security_metrics["components_initialized"].append("jwt_auth")
            else:
                self._security_metrics["components_failed"].append("jwt_auth")
            
            # Mark as initialized
            self.initialized = True
            end_time = asyncio.get_event_loop().time()
            self._security_metrics["initialization_time"] = end_time - start_time
            
            # Log initialization summary
            logger.info(f"✅ Security Manager initialized in {self._security_metrics['initialization_time']:.2f}s")
            logger.info(f"✅ Components initialized: {len(self._security_metrics['components_initialized'])}")
            
            if self._security_metrics["components_failed"]:
                logger.warning(f"⚠️  Components failed: {self._security_metrics['components_failed']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Security Manager: {e}")
            return False
    
    async def _init_token_blacklist(self) -> bool:
        """Initialize Redis token blacklist."""
        try:
            redis_url = self.config.get_redis_url("blacklist")
            self.token_blacklist = await init_token_blacklist(redis_url)
            logger.info("✅ Redis token blacklist initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize token blacklist: {e}")
            # Continue without Redis blacklist (will use in-memory fallback)
            return False
    
    async def _init_jwt_authenticator(self) -> bool:
        """Initialize enhanced JWT authenticator."""
        try:
            self.jwt_auth = init_enhanced_jwt_authenticator(
                secret_key=None,  # Will use environment variable
                redis_blacklist=self.token_blacklist,
                access_token_expire_minutes=self.config.jwt.access_token_expire_minutes,
                refresh_token_expire_days=self.config.jwt.refresh_token_expire_days,
                algorithm=self.config.jwt.algorithm,
                issuer=self.config.jwt.issuer,
                audience=self.config.jwt.audience
            )
            logger.info("✅ Enhanced JWT authenticator initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize JWT authenticator: {e}")
            return False
    
    def configure_fastapi_security(self, app: FastAPI) -> None:
        """
        Configure FastAPI application with all security middleware.
        
        Args:
            app: FastAPI application instance
        """
        if not self.initialized:
            raise RuntimeError("Security Manager not initialized. Call initialize() first.")
        
        logger.info("🔧 Configuring FastAPI security middleware...")
        
        # 1. HTTPS Redirect (if in production)
        if self.config.environment == "production":
            app.add_middleware(HTTPSRedirectMiddleware)
        
        # 2. Trusted Host Middleware
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.allowed_hosts
        )
        
        # 3. CORS Middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # 4. Session Middleware (for CSRF)
        app.add_middleware(
            SessionMiddleware,
            secret_key=self.config.encryption_key,
            max_age=self.config.sessions.session_ttl_hours * 3600,
            same_site="strict",
            https_only=self.config.environment == "production"
        )
        
        # 5. CSRF Protection Middleware
        csrf_middleware = CSRFProtectionMiddleware(
            app=None,  # Will be set by FastAPI
            secret_key=self.config.encryption_key,
            trusted_origins=self.config.cors_origins,
            enforce_origin_check=self.config.environment == "production"
        )
        app.add_middleware(type(csrf_middleware), **csrf_middleware.__dict__)
        
        # 6. Input Validation Middleware
        validation_rules = self._create_validation_rules()
        input_validation_middleware = InputValidationMiddleware(
            app=None,  # Will be set by FastAPI
            validation_rules=validation_rules
        )
        app.add_middleware(type(input_validation_middleware), **input_validation_middleware.__dict__)
        
        # 7. Security Headers Middleware
        @app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            
            # Add security headers
            if self.config.headers.enable_hsts:
                response.headers["Strict-Transport-Security"] = (
                    f"max-age={self.config.headers.hsts_max_age}; "
                    f"includeSubDomains; preload"
                )
            
            if self.config.headers.enable_csp:
                response.headers["Content-Security-Policy"] = self.config.headers.csp_policy
            
            if self.config.headers.enable_xss_protection:
                response.headers["X-XSS-Protection"] = "1; mode=block"
            
            if self.config.headers.enable_content_type_nosniff:
                response.headers["X-Content-Type-Options"] = "nosniff"
            
            if self.config.headers.enable_frame_options:
                response.headers["X-Frame-Options"] = self.config.headers.frame_options
            
            if self.config.headers.enable_referrer_policy:
                response.headers["Referrer-Policy"] = self.config.headers.referrer_policy
            
            # Additional security headers
            response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
            response.headers["X-Download-Options"] = "noopen"
            response.headers["Permissions-Policy"] = (
                "geolocation=(), microphone=(), camera=(), "
                "payment=(), usb=(), screen-wake-lock=()"
            )
            
            return response
        
        logger.info("✅ FastAPI security middleware configured")
    
    def _create_validation_rules(self) -> Dict[str, Dict[str, ValidationRule]]:
        """Create input validation rules for API endpoints."""
        return {
            "/api/v1/auth/login": {
                "username": ValidationRule(
                    name="username",
                    max_length=100,
                    min_length=3,
                    pattern=r'^[a-zA-Z0-9._@-]+$',
                    sanitize=True,
                    required=True
                ),
                "password": ValidationRule(
                    name="password",
                    max_length=128,
                    min_length=8,
                    sanitize=False,
                    required=True
                )
            },
            "/api/v1/projects": {
                "name": ValidationRule(
                    name="project_name",
                    max_length=100,
                    min_length=1,
                    pattern=r'^[a-zA-Z0-9\s._-]+$',
                    sanitize=True,
                    required=True
                ),
                "description": ValidationRule(
                    name="description",
                    max_length=1000,
                    sanitize=True,
                    required=False
                )
            },
            "/api/v1/users": {
                "email": ValidationRule(
                    name="email",
                    max_length=255,
                    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    sanitize=True,
                    required=True
                ),
                "username": ValidationRule(
                    name="username",
                    max_length=50,
                    min_length=3,
                    pattern=r'^[a-zA-Z0-9._-]+$',
                    sanitize=True,
                    required=True
                )
            },
            "/api/v1/files/upload": {
                "file": ValidationRule(
                    name="file",
                    sanitize=True,
                    required=True
                )
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive security health check.
        
        Returns:
            Health check results
        """
        health_status = {
            "healthy": True,
            "components": {},
            "metrics": self._security_metrics.copy(),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        try:
            # Check JWT authenticator
            if self.jwt_auth:
                jwt_metrics = await self.jwt_auth.get_security_metrics()
                health_status["components"]["jwt_auth"] = {
                    "healthy": True,
                    "metrics": jwt_metrics
                }
            else:
                health_status["components"]["jwt_auth"] = {
                    "healthy": False,
                    "error": "Not initialized"
                }
                health_status["healthy"] = False
            
            # Check token blacklist
            if self.token_blacklist:
                blacklist_health = await self.token_blacklist.health_check()
                health_status["components"]["token_blacklist"] = blacklist_health
                if not blacklist_health["healthy"]:
                    health_status["healthy"] = False
            else:
                health_status["components"]["token_blacklist"] = {
                    "healthy": False,
                    "error": "Not initialized"
                }
            
            # Update metrics
            if health_status["healthy"]:
                self._security_metrics["health_checks_passed"] += 1
            
        except Exception as e:
            logger.error(f"Security health check failed: {e}")
            health_status["healthy"] = False
            health_status["error"] = str(e)
        
        return health_status
    
    async def rotate_secrets(self, reason: str = "Scheduled rotation") -> bool:
        """
        Rotate security secrets.
        
        Args:
            reason: Reason for rotation
            
        Returns:
            True if rotation was successful
        """
        try:
            logger.info(f"🔄 Starting secret rotation: {reason}")
            
            # Rotate JWT tokens globally
            if self.jwt_auth:
                await self.jwt_auth.rotate_tokens_globally(reason)
            
            # Additional secret rotation logic can be added here
            
            logger.info("✅ Secret rotation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Secret rotation failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up security resources."""
        try:
            logger.info("🧹 Cleaning up security resources...")
            
            if self.token_blacklist:
                await self.token_blacklist.close()
            
            logger.info("✅ Security cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Security cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return self._security_metrics.copy()


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


async def init_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Initialize global security manager."""
    global _security_manager
    
    _security_manager = SecurityManager()
    await _security_manager.initialize(config)
    
    return _security_manager


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    
    if _security_manager is None:
        raise RuntimeError("Security manager not initialized. Call init_security_manager() first.")
    
    return _security_manager


@asynccontextmanager
async def security_lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for security initialization.
    
    Usage:
        app = FastAPI(lifespan=security_lifespan)
    """
    # Startup
    logger.info("🚀 Starting Claude-TUI with security initialization...")
    
    try:
        # Initialize security manager
        security_manager = await init_security_manager()
        
        # Configure FastAPI security
        security_manager.configure_fastapi_security(app)
        
        logger.info("✅ Security initialization completed")
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Shutting down security manager...")
        
        if _security_manager:
            await _security_manager.cleanup()
        
        logger.info("✅ Security shutdown completed")


# Utility functions for common security operations
async def validate_api_token(token: str, request: Request) -> Dict[str, Any]:
    """Validate API token with enhanced security checks."""
    security_manager = get_security_manager()
    
    if not security_manager.jwt_auth:
        raise RuntimeError("JWT authenticator not initialized")
    
    # Get session repository (you'll need to implement this)
    # session_repo = get_session_repository()
    
    # For now, validate without session verification
    # In production, you should integrate with your session repository
    token_data = await security_manager.jwt_auth.validate_token(
        token=token,
        session_repo=None,  # Replace with actual session repo
        verify_session=False,  # Enable this when session repo is available
        verify_device_fingerprint=False,  # Enable for stricter security
        current_ip=request.client.host if request.client else None,
        current_user_agent=request.headers.get("user-agent")
    )
    
    return token_data.dict()


async def revoke_api_token(token: str, reason: str = "Manual revocation") -> bool:
    """Revoke API token."""
    security_manager = get_security_manager()
    
    if not security_manager.jwt_auth:
        raise RuntimeError("JWT authenticator not initialized")
    
    # Get session repository (you'll need to implement this)
    # session_repo = get_session_repository()
    
    return await security_manager.jwt_auth.revoke_token(
        token=token,
        session_repo=None,  # Replace with actual session repo
        reason=reason
    )