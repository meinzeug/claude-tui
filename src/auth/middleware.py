"""
Authentication and Authorization Middleware for FastAPI

Provides JWT token validation, user authentication, and RBAC enforcement
with comprehensive security features and audit logging.
"""

from typing import Optional, List, Callable, Any, Dict
from datetime import datetime, timezone
import logging
from functools import wraps
import asyncio

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .jwt_service import JWTService, TokenValidationResult, TokenType
from .session_service import SessionService
from .rbac import RBACService, Permission, Role
from .audit_logger import AuditLogger
from ..database.models import User
from ..core.exceptions import AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

# Security scheme for FastAPI
security = HTTPBearer(auto_error=False)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware that validates JWT tokens and manages sessions.
    
    Features:
    - JWT token validation
    - Session management
    - IP address validation
    - Rate limiting integration
    - Audit logging
    """
    
    def __init__(
        self,
        app,
        jwt_service: JWTService,
        session_service: SessionService,
        audit_logger: AuditLogger,
        excluded_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.jwt_service = jwt_service
        self.session_service = session_service
        self.audit_logger = audit_logger
        self.excluded_paths = excluded_paths or [
            "/docs", "/redoc", "/openapi.json",
            "/health", "/metrics",
            "/auth/login", "/auth/register", "/auth/oauth",
            "/auth/forgot-password", "/auth/reset-password"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware"""
        
        # Skip authentication for excluded paths
        if self._should_skip_auth(request.url.path):
            return await call_next(request)
        
        try:
            # Extract and validate token
            auth_result = await self._authenticate_request(request)
            
            if auth_result:
                # Add authentication context to request
                request.state.user = auth_result["user"]
                request.state.session = auth_result["session"]
                request.state.token_payload = auth_result["token_payload"]
                
                # Log successful authentication
                await self.audit_logger.log_authentication(
                    user_id=str(auth_result["user"].id),
                    action="token_validated",
                    ip_address=self._get_client_ip(request),
                    user_agent=request.headers.get("user-agent"),
                    success=True
                )
            
            # Process request
            response = await call_next(request)
            
            # Update session activity if authenticated
            if hasattr(request.state, "session") and request.state.session:
                # Update session activity asynchronously
                asyncio.create_task(
                    self.session_service.validate_session(
                        request.state.session.session_id,
                        self._get_client_ip(request)
                    )
                )
            
            return response
            
        except AuthenticationError as e:
            # Log authentication failure
            await self.audit_logger.log_authentication(
                action="authentication_failed",
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                success=False,
                error_message=str(e)
            )
            
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": str(e), "type": "authentication_error"}
            )
        
        except Exception as e:
            logger.error("Authentication middleware error: %s", e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Authentication system error"}
            )
    
    def _should_skip_auth(self, path: str) -> bool:
        """Check if path should skip authentication"""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    async def _authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate request and return user context"""
        
        # Extract token from header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        
        if not auth_header.startswith("Bearer "):
            raise AuthenticationError("Invalid authorization header format")
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Validate JWT token
        validation_result = self.jwt_service.verify_token(token, TokenType.ACCESS)
        
        if not validation_result.valid:
            if validation_result.expired:
                raise AuthenticationError("Token has expired")
            else:
                raise AuthenticationError(validation_result.error or "Invalid token")
        
        # Get token payload
        token_payload = validation_result.payload
        if not token_payload:
            raise AuthenticationError("Invalid token payload")
        
        # Validate session
        session_metadata = await self.session_service.validate_session(
            token_payload.jti,  # Use JWT ID as session reference
            self._get_client_ip(request)
        )
        
        if not session_metadata:
            raise AuthenticationError("Session invalid or expired")
        
        # TODO: Get user from database using token_payload.sub
        # For now, create a mock user object
        user = type('User', (), {
            'id': token_payload.sub,
            'username': 'user',  # Would come from DB
            'email': 'user@example.com',  # Would come from DB
            'is_active': True
        })()
        
        return {
            "user": user,
            "session": session_metadata,
            "token_payload": token_payload
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support"""
        # Check for forwarded headers (common with load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"


class RBACMiddleware:
    """
    Role-Based Access Control middleware for FastAPI dependency injection.
    
    Features:
    - Role and permission validation
    - Resource-based authorization
    - Audit logging
    - Flexible permission checking
    """
    
    def __init__(
        self,
        rbac_service: RBACService,
        audit_logger: AuditLogger
    ):
        self.rbac_service = rbac_service
        self.audit_logger = audit_logger
    
    def require_permission(
        self,
        permission: str,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None
    ) -> Callable:
        """
        Decorator/dependency that requires specific permission.
        
        Args:
            permission: Required permission (e.g., "users:read")
            resource: Resource type (optional)
            resource_id: Specific resource ID (optional)
        
        Returns:
            FastAPI dependency function
        """
        async def check_permission(request: Request):
            """Check if user has required permission"""
            
            # Ensure user is authenticated
            if not hasattr(request.state, "user") or not request.state.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user = request.state.user
            
            try:
                # Check permission
                has_permission = await self.rbac_service.check_user_permission(
                    user_id=str(user.id),
                    permission=permission,
                    resource=resource,
                    resource_id=resource_id
                )
                
                if not has_permission:
                    # Log authorization failure
                    await self.audit_logger.log_authorization(
                        user_id=str(user.id),
                        action="permission_denied",
                        resource_type=resource or "unknown",
                        resource_id=resource_id,
                        permission=permission,
                        success=False,
                        ip_address=self._get_request_ip(request)
                    )
                    
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission denied: {permission}"
                    )
                
                # Log successful authorization
                await self.audit_logger.log_authorization(
                    user_id=str(user.id),
                    action="permission_granted",
                    resource_type=resource or "unknown",
                    resource_id=resource_id,
                    permission=permission,
                    success=True,
                    ip_address=self._get_request_ip(request)
                )
                
                return user
                
            except Exception as e:
                logger.error("Permission check failed: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authorization system error"
                )
        
        return check_permission
    
    def require_role(self, role: str) -> Callable:
        """
        Decorator/dependency that requires specific role.
        
        Args:
            role: Required role name
        
        Returns:
            FastAPI dependency function
        """
        async def check_role(request: Request):
            """Check if user has required role"""
            
            if not hasattr(request.state, "user") or not request.state.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user = request.state.user
            
            try:
                has_role = await self.rbac_service.check_user_role(
                    user_id=str(user.id),
                    role_name=role
                )
                
                if not has_role:
                    await self.audit_logger.log_authorization(
                        user_id=str(user.id),
                        action="role_denied",
                        resource_type="role",
                        resource_id=role,
                        success=False,
                        ip_address=self._get_request_ip(request)
                    )
                    
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Role required: {role}"
                    )
                
                return user
                
            except Exception as e:
                logger.error("Role check failed: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authorization system error"
                )
        
        return check_role
    
    def require_any_permission(self, permissions: List[str]) -> Callable:
        """
        Decorator/dependency that requires any of the specified permissions.
        
        Args:
            permissions: List of acceptable permissions
        
        Returns:
            FastAPI dependency function
        """
        async def check_any_permission(request: Request):
            """Check if user has any of the required permissions"""
            
            if not hasattr(request.state, "user") or not request.state.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user = request.state.user
            
            try:
                for permission in permissions:
                    has_permission = await self.rbac_service.check_user_permission(
                        user_id=str(user.id),
                        permission=permission
                    )
                    
                    if has_permission:
                        await self.audit_logger.log_authorization(
                            user_id=str(user.id),
                            action="permission_granted",
                            resource_type="any_permission",
                            permission=permission,
                            success=True,
                            ip_address=self._get_request_ip(request)
                        )
                        return user
                
                # No permissions matched
                await self.audit_logger.log_authorization(
                    user_id=str(user.id),
                    action="permission_denied",
                    resource_type="any_permission",
                    permission=", ".join(permissions),
                    success=False,
                    ip_address=self._get_request_ip(request)
                )
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"One of these permissions required: {', '.join(permissions)}"
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Permission check failed: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authorization system error"
                )
        
        return check_any_permission
    
    def _get_request_ip(self, request: Request) -> str:
        """Get client IP from request"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"


# Convenience functions for dependency injection

async def get_current_user(request: Request) -> User:
    """Get current authenticated user from request context"""
    if not hasattr(request.state, "user") or not request.state.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return request.state.user


async def get_current_active_user(request: Request) -> User:
    """Get current authenticated and active user"""
    user = await get_current_user(request)
    if not getattr(user, 'is_active', True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return user


async def get_current_admin_user(request: Request) -> User:
    """Get current user if they are an admin"""
    user = await get_current_active_user(request)
    if not getattr(user, 'is_superuser', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


def create_permission_dependency(permission: str, resource: Optional[str] = None):
    """
    Create a FastAPI dependency that checks for a specific permission.
    
    Args:
        permission: Required permission
        resource: Resource type (optional)
    
    Returns:
        FastAPI dependency
    """
    async def check_permission(
        request: Request,
        user: User = Depends(get_current_active_user)
    ):
        # This would typically integrate with your RBAC service
        # For now, assuming user has permission info
        if not hasattr(user, 'permissions') or permission not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return user
    
    return check_permission


# Rate limiting middleware integration
class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with user-aware limits"""
    
    def __init__(
        self,
        app,
        default_limit: int = 100,  # requests per minute
        authenticated_limit: int = 1000,  # higher limit for authenticated users
        admin_limit: int = 5000  # even higher for admins
    ):
        super().__init__(app)
        self.default_limit = default_limit
        self.authenticated_limit = authenticated_limit
        self.admin_limit = admin_limit
        
        # Simple in-memory rate limiting (use Redis in production)
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on user context"""
        
        # Get identifier (IP or user ID)
        identifier = self._get_rate_limit_key(request)
        
        # Determine limit based on user type
        limit = self._get_rate_limit(request)
        
        # Check rate limit
        current_count = self.request_counts.get(identifier, 0)
        
        if current_count >= limit:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )
        
        # Update count
        self.request_counts[identifier] = current_count + 1
        
        # Process request
        response = await call_next(request)
        
        return response
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """Get rate limiting key (user ID or IP)"""
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Fallback to IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip = forwarded_for.split(",")[0].strip()
        else:
            ip = getattr(request.client, "host", "unknown")
        
        return f"ip:{ip}"
    
    def _get_rate_limit(self, request: Request) -> int:
        """Get rate limit based on user privileges"""
        if hasattr(request.state, "user") and request.state.user:
            user = request.state.user
            if getattr(user, 'is_superuser', False):
                return self.admin_limit
            return self.authenticated_limit
        
        return self.default_limit