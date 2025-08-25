"""
Comprehensive Authentication Endpoints.

Provides secure authentication endpoints with enhanced features:
- JWT access/refresh token management
- OAuth integration (GitHub/Google)
- Rate limiting and security measures
- Audit logging
- Password reset functionality
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..dependencies.database import get_database
from ..dependencies.auth_helpers import (
    get_current_user_token,
    get_jwt_service,
    get_session_manager,
    get_oauth_providers
)
from ..models.user import User, UserSession
from ..schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    TokenRefreshRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    PasswordChangeRequest,
    OAuthCallbackRequest
)
from ..schemas.user import UserResponse
from ...auth.jwt_service import JWTService, TokenData
from ...auth.session_manager import SessionManager
from ...auth.oauth import GitHubOAuthProvider, GoogleOAuthProvider, OAuthError
from ...middleware.rbac import RBACMiddleware, Permission
from ...core.exceptions import AuthenticationError, ValidationError


# Setup logging
logger = logging.getLogger(__name__)

# Setup rate limiter
limiter = Limiter(key_func=get_remote_address)

# Router
router = APIRouter(prefix="/auth", tags=["authentication"])
router.state.limiter = limiter
router.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_database),
    jwt_service: JWTService = Depends(get_jwt_service),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    User login with enhanced security features.
    
    Features:
    - Rate limiting (5 attempts per minute)
    - Session management
    - Audit logging
    - IP tracking
    """
    try:
        # Get client information
        client_ip = get_remote_address(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Authenticate user
        user = await authenticate_user_enhanced(
            db=db,
            identifier=login_data.username,
            password=login_data.password,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not user:
            # Log failed attempt
            await log_auth_event(
                event_type="login_failed",
                user_id=None,
                ip_address=client_ip,
                user_agent=user_agent,
                details={"identifier": login_data.username}
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        if not user.is_active:
            await log_auth_event(
                event_type="login_inactive_user",
                user_id=str(user.id),
                ip_address=client_ip,
                user_agent=user_agent
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Create session
        session = await session_manager.create_session(
            user_id=str(user.id),
            username=user.username,
            role=user.role,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Get user permissions
        permissions = await get_user_permissions(user)
        
        # Create token pair
        token_pair = await jwt_service.create_token_pair(
            user=user,
            session_id=session.session_id,
            permissions=permissions,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Update user last login
        user.last_login = datetime.now(timezone.utc)
        await db.commit()
        
        # Log successful login
        await log_auth_event(
            event_type="login_success",
            user_id=str(user.id),
            ip_address=client_ip,
            user_agent=user_agent,
            details={"session_id": session.session_id}
        )
        
        logger.info(f"User {user.username} logged in successfully from {client_ip}")
        
        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
            user_id=token_pair.user_id,
            permissions=token_pair.permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout")
@limiter.limit("20/minute")
async def logout(
    request: Request,
    token_data: TokenData = Depends(get_current_user_token),
    jwt_service: JWTService = Depends(get_jwt_service),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    User logout with token revocation and session cleanup.
    """
    try:
        client_ip = get_remote_address(request)
        
        # Revoke current session
        if token_data.session_id:
            await session_manager.invalidate_session(token_data.session_id)
        
        # Revoke access token
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            await jwt_service.revoke_token(token)
        
        # Log logout
        await log_auth_event(
            event_type="logout",
            user_id=token_data.user_id,
            ip_address=client_ip,
            details={"session_id": token_data.session_id}
        )
        
        logger.info(f"User {token_data.username} logged out from {client_ip}")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return {"message": "Logout completed"}  # Always succeed for security


# OAuth endpoints
@router.get("/oauth/{provider}")
async def oauth_login(
    provider: str,
    request: Request,
    oauth_providers: dict = Depends(get_oauth_providers)
):
    """
    Initiate OAuth login flow.
    """
    try:
        if provider not in oauth_providers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported OAuth provider: {provider}"
            )
        
        oauth_provider = oauth_providers[provider]
        auth_url, state = oauth_provider.get_authorization_url()
        
        return {
            "authorization_url": auth_url,
            "state": state
        }
        
    except Exception as e:
        logger.error(f"OAuth initiation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth initiation failed"
        )


# Helper functions (placeholder implementations)
async def authenticate_user_enhanced(
    db: AsyncSession,
    identifier: str,
    password: str,
    ip_address: str,
    user_agent: str
) -> Optional[User]:
    """Enhanced user authentication with security logging."""
    # Implementation would go here
    return None


async def get_user_permissions(user: User) -> list:
    """Get user permissions based on role."""
    # Implementation would go here - integrate with RBAC
    return []


async def log_auth_event(
    event_type: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[dict] = None
):
    """Log authentication event for audit purposes."""
    logger.info(
        f"AUTH_EVENT: {event_type} - User: {user_id} - IP: {ip_address} - Details: {details}"
    )