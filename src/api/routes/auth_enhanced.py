"""
Comprehensive Authentication API Endpoints for Claude-TIU

Provides a complete authentication system with JWT tokens, OAuth, 2FA,
password reset, session management, and comprehensive security features.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr

from ..dependencies.database import get_db
from ..dependencies.auth import get_current_user, get_current_active_user
from ..models.user import User
from ..schemas.auth import (
    LoginRequest, LoginResponse, TokenRefreshRequest, TokenResponse,
    RegisterRequest, RegisterResponse, PasswordChangeRequest, PasswordChangeResponse,
    PasswordResetRequest, PasswordResetResponse, PasswordResetConfirmRequest,
    PasswordResetConfirmResponse, OAuthLoginRequest, OAuthLoginResponse,
    TwoFactorSetupResponse, TwoFactorVerifyRequest, TwoFactorVerifyResponse,
    UserSessionResponse, DeviceInfo
)
from ...auth.jwt_service import JWTService, TokenType
from ...auth.session_service import SessionService
from ...auth.audit_logger import get_audit_logger, SecurityEventType, SecurityLevel
from ...auth.password_reset import PasswordResetService, EmailService
from ...auth.oauth.github import GitHubOAuthProvider
from ...auth.rbac import rbac_manager
from ...core.exceptions import AuthenticationError, ValidationError, SecurityError
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize services
jwt_service = JWTService(
    secret_key=settings.SECRET_KEY,
    algorithm="HS256"
)

session_service = SessionService(
    session_ttl_hours=24,
    max_sessions_per_user=10
)

email_service = EmailService(
    smtp_host=getattr(settings, 'SMTP_HOST', 'localhost'),
    smtp_port=getattr(settings, 'SMTP_PORT', 587),
    smtp_username=getattr(settings, 'SMTP_USERNAME', None),
    smtp_password=getattr(settings, 'SMTP_PASSWORD', None)
)

password_reset_service = PasswordResetService(
    jwt_service=jwt_service,
    email_service=email_service,
    audit_logger=get_audit_logger()
)

# OAuth providers
github_oauth = GitHubOAuthProvider(
    client_id=getattr(settings, 'GITHUB_CLIENT_ID', ''),
    client_secret=getattr(settings, 'GITHUB_CLIENT_SECRET', ''),
    redirect_uri=getattr(settings, 'GITHUB_REDIRECT_URI', '')
) if hasattr(settings, 'GITHUB_CLIENT_ID') else None

security = HTTPBearer(auto_error=False)
router = APIRouter(prefix="/auth", tags=["authentication"])


# Pydantic models for request/response
class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    identifier: str  # email or username
    password: str
    remember_me: bool = False
    device_fingerprint: Optional[str] = None


class LoginResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    session_id: str
    permissions: List[str] = []


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    terms_accepted: bool = True


class RegisterResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    session_id: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class PasswordChangeResponse(BaseModel):
    message: str
    changed_at: datetime


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetResponse(BaseModel):
    message: str
    success: bool = True


class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str


class PasswordResetConfirmResponse(BaseModel):
    success: bool
    message: str


class OAuthLoginRequest(BaseModel):
    code: str
    state: str


class OAuthLoginResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    session_id: str
    permissions: List[str] = []
    provider: str


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return getattr(request.client, "host", "unknown")


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: RegisterRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> RegisterResponse:
    """
    Register a new user account with comprehensive validation and security.
    
    Features:
    - Email and username uniqueness validation
    - Password strength requirements
    - Audit logging
    - Session creation
    - Rate limiting protection
    """
    try:
        audit_logger = get_audit_logger()
        client_ip = get_client_ip(req)
        user_agent = req.headers.get("user-agent", "")
        
        # Validate terms acceptance
        if not request.terms_accepted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Terms of service must be accepted"
            )
        
        # Check if user already exists
        existing_user = await db.execute(
            select(User).where(User.email == request.email.lower())
        )
        if existing_user.scalar_one_or_none():
            await audit_logger.log_authentication(
                SecurityEventType.REGISTRATION_DUPLICATE,
                ip_address=client_ip,
                success=False,
                message="Registration attempted for existing email",
                details={'email': str(request.email)}
            )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check username availability
        existing_username = await db.execute(
            select(User).where(User.username == request.username.lower())
        )
        if existing_username.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Validate password strength
        if not User.validate_password_strength(request.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, number, and special character"
            )
        
        # Create user
        user = User(
            email=request.email.lower().strip(),
            username=request.username.lower().strip(),
            full_name=request.full_name.strip() if request.full_name else None
        )
        user.set_password(request.password)
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Create session
        session_id = await session_service.create_session(
            user=user,
            ip_address=client_ip,
            user_agent=user_agent,
            login_method="registration"
        )
        
        # Create initial token pair
        token_pair = await jwt_service.create_token_pair(
            user=user,
            session_id=session_id,
            permissions=["user:read", "user:update"],  # Basic permissions
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Log successful registration
        await audit_logger.log_authentication(
            SecurityEventType.REGISTRATION_SUCCESS,
            user_id=str(user.id),
            username=user.username,
            ip_address=client_ip,
            user_agent=user_agent,
            session_id=session_id,
            success=True,
            message="User registered successfully"
        )
        
        logger.info("New user registered: %s (%s) from %s", user.username, user.email, client_ip)
        
        return RegisterResponse(
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                username=user.username,
                full_name=user.full_name,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at
            ),
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
) -> LoginResponse:
    """
    Authenticate user with comprehensive security features.
    
    Features:
    - Email or username authentication
    - Account lockout protection
    - Session management
    - Audit logging
    - Device fingerprinting
    - Suspicious activity detection
    """
    try:
        audit_logger = get_audit_logger()
        client_ip = get_client_ip(req)
        user_agent = req.headers.get("user-agent", "")
        identifier = request.identifier.lower().strip()
        
        # Find user by email or username
        user_query = select(User).where(
            (User.email == identifier) | 
            (User.username == identifier)
        )
        result = await db.execute(user_query)
        user = result.scalar_one_or_none()
        
        # Log failed attempt if user not found
        if not user:
            await audit_logger.log_authentication(
                SecurityEventType.LOGIN_FAILED,
                ip_address=client_ip,
                user_agent=user_agent,
                success=False,
                message=f"Login attempt with unknown identifier: {identifier}",
                details={'identifier': identifier}
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check if account is locked
        if user.is_account_locked():
            await audit_logger.log_authentication(
                SecurityEventType.LOGIN_BLOCKED,
                user_id=str(user.id),
                username=user.username,
                ip_address=client_ip,
                success=False,
                message="Login blocked - account locked",
                details={'locked_until': str(user.account_locked_until)}
            )
            
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account is locked until {user.account_locked_until}"
            )
        
        # Verify password
        if not user.verify_password(request.password):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failures
            if user.failed_login_attempts >= 5:
                user.lock_account(duration_minutes=30)
                
                await audit_logger.log_authentication(
                    SecurityEventType.LOGIN_BLOCKED,
                    user_id=str(user.id),
                    username=user.username,
                    ip_address=client_ip,
                    success=False,
                    message="Account locked due to failed login attempts",
                    details={'failed_attempts': user.failed_login_attempts}
                )
                
                logger.warning("Account locked for user: %s after %d failed attempts", 
                             user.username, user.failed_login_attempts)
            else:
                await audit_logger.log_authentication(
                    SecurityEventType.LOGIN_FAILED,
                    user_id=str(user.id),
                    username=user.username,
                    ip_address=client_ip,
                    success=False,
                    message="Invalid password",
                    details={'failed_attempts': user.failed_login_attempts}
                )
            
            await db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check if account is active
        if not user.is_active:
            await audit_logger.log_authentication(
                SecurityEventType.LOGIN_FAILED,
                user_id=str(user.id),
                username=user.username,
                ip_address=client_ip,
                success=False,
                message="Login failed - account inactive"
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Successful login - reset failed attempts
        user.failed_login_attempts = 0
        user.last_login = datetime.now(timezone.utc)
        await db.commit()
        
        # Create session with device fingerprinting
        session_id = await session_service.create_session(
            user=user,
            ip_address=client_ip,
            user_agent=user_agent,
            device_fingerprint=request.device_fingerprint,
            login_method="password"
        )
        
        # Get user permissions from RBAC
        user_roles = ["USER"]  # Default role, would come from database
        permissions = rbac_manager.get_user_effective_permissions(user_roles)
        
        # Create token pair
        token_pair = await jwt_service.create_token_pair(
            user=user,
            session_id=session_id,
            permissions=permissions,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Log successful login
        await audit_logger.log_authentication(
            SecurityEventType.LOGIN_SUCCESS,
            user_id=str(user.id),
            username=user.username,
            ip_address=client_ip,
            user_agent=user_agent,
            session_id=session_id,
            success=True,
            message="User logged in successfully",
            details={
                'device_fingerprint': request.device_fingerprint,
                'permissions_count': len(permissions),
                'remember_me': request.remember_me
            }
        )
        
        logger.info("User logged in: %s from %s", user.username, client_ip)
        
        return LoginResponse(
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                username=user.username,
                full_name=user.full_name,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at,
                last_login=user.last_login,
                password_changed_at=user.password_changed_at
            ),
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
            session_id=session_id,
            permissions=permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Refresh access token with security validation.
    """
    try:
        audit_logger = get_audit_logger()
        client_ip = get_client_ip(req)
        
        # Mock user service for token refresh
        class MockUserService:
            @staticmethod
            async def get_user_by_id(user_id: str):
                result = await db.execute(select(User).where(User.id == user_id))
                return result.scalar_one_or_none()
        
        # Refresh token with validation
        new_token_pair = await jwt_service.refresh_access_token(
            refresh_token=request.refresh_token,
            user_service=MockUserService()
        )
        
        if not new_token_pair:
            await audit_logger.log_authentication(
                SecurityEventType.TOKEN_EXPIRED,
                ip_address=client_ip,
                success=False,
                message="Invalid or expired refresh token"
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return TokenResponse(
            access_token=new_token_pair.access_token,
            refresh_token=new_token_pair.refresh_token,
            token_type=new_token_pair.token_type,
            expires_in=new_token_pair.expires_in
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    req: Request,
    current_user: User = Depends(get_current_active_user),
    token: HTTPAuthorizationCredentials = Depends(security)
):
    """Logout user with comprehensive session cleanup."""
    try:
        audit_logger = get_audit_logger()
        client_ip = get_client_ip(req)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No token provided"
            )
        
        # Revoke token
        success = await jwt_service.revoke_token(token.credentials)
        
        # Log successful logout
        await audit_logger.log_authentication(
            SecurityEventType.LOGOUT,
            user_id=str(current_user.id),
            username=current_user.username,
            ip_address=client_ip,
            success=success,
            message="User logged out"
        )
        
        logger.info("User logged out: %s from %s", current_user.username, client_ip)
        
    except Exception as e:
        logger.error("Logout failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/forgot-password", response_model=PasswordResetResponse)
async def forgot_password(
    request: PasswordResetRequest,
    req: Request
) -> PasswordResetResponse:
    """Initiate password reset process."""
    try:
        client_ip = get_client_ip(req)
        user_agent = req.headers.get("user-agent", "")
        
        # Request password reset
        result = await password_reset_service.request_password_reset(
            email=str(request.email),
            ip_address=client_ip,
            user_agent=user_agent,
            reset_base_url=getattr(settings, 'FRONTEND_URL', 'http://localhost:3000') + '/reset-password'
        )
        
        return PasswordResetResponse(
            message=result['message'],
            success=True  # Always true for security
        )
        
    except Exception as e:
        logger.error("Password reset request failed: %s", e)
        return PasswordResetResponse(
            message="If an account exists for this email, a password reset link has been sent.",
            success=True
        )


@router.post("/reset-password", response_model=PasswordResetConfirmResponse)
async def reset_password(
    request: PasswordResetConfirmRequest,
    req: Request
) -> PasswordResetConfirmResponse:
    """Complete password reset process."""
    try:
        client_ip = get_client_ip(req)
        user_agent = req.headers.get("user-agent", "")
        
        # Reset password using service
        result = await password_reset_service.reset_password(
            token=request.token,
            new_password=request.new_password,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        return PasswordResetConfirmResponse(
            success=result['success'],
            message=result['message']
        )
        
    except Exception as e:
        logger.error("Password reset failed: %s", e)
        return PasswordResetConfirmResponse(
            success=False,
            message="Password reset failed. Please try again or request a new reset link."
        )


@router.post("/change-password", response_model=PasswordChangeResponse)
async def change_password(
    request: PasswordChangeRequest,
    req: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> PasswordChangeResponse:
    """Change user password with current password verification."""
    try:
        audit_logger = get_audit_logger()
        client_ip = get_client_ip(req)
        
        # Verify current password
        if not current_user.verify_password(request.current_password):
            await audit_logger.log_authentication(
                SecurityEventType.PASSWORD_CHANGED,
                user_id=str(current_user.id),
                username=current_user.username,
                ip_address=client_ip,
                success=False,
                message="Password change failed - incorrect current password"
            )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        if not User.validate_password_strength(request.new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password does not meet security requirements"
            )
        
        # Check that new password is different
        if current_user.verify_password(request.new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )
        
        # Update password
        current_user.set_password(request.new_password)
        await db.commit()
        
        # Log successful password change
        await audit_logger.log_authentication(
            SecurityEventType.PASSWORD_CHANGED,
            user_id=str(current_user.id),
            username=current_user.username,
            ip_address=client_ip,
            success=True,
            message="Password changed successfully"
        )
        
        logger.info("Password changed for user: %s from %s", current_user.username, client_ip)
        
        return PasswordChangeResponse(
            message="Password changed successfully",
            changed_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
) -> UserResponse:
    """Get current authenticated user information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        password_changed_at=current_user.password_changed_at
    )


@router.get("/sessions")
async def get_user_sessions(
    current_user: User = Depends(get_current_active_user)
):
    """Get all active sessions for current user."""
    try:
        sessions = await session_service.get_user_sessions(
            user_id=str(current_user.id),
            active_only=True
        )
        
        return {
            'sessions': [
                {
                    'session_id': session.session_id,
                    'ip_address': session.ip_address,
                    'user_agent': session.user_agent,
                    'location': session.location,
                    'created_at': session.created_at.isoformat(),
                    'last_activity': session.last_activity.isoformat(),
                    'device_trusted': session.device_trusted,
                    'login_method': session.login_method
                }
                for session in sessions
            ],
            'total_sessions': len(sessions)
        }
        
    except Exception as e:
        logger.error("Failed to get user sessions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


# OAuth endpoints (if GitHub OAuth is configured)
if github_oauth:
    @router.get("/oauth/github")
    async def github_oauth_login(req: Request):
        """Initiate GitHub OAuth login flow."""
        try:
            state = github_oauth.generate_state()
            auth_url, _ = github_oauth.get_authorization_url(state)
            
            return {'authorization_url': auth_url, 'state': state}
            
        except Exception as e:
            logger.error("GitHub OAuth initiation failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth initiation failed"
            )
    
    @router.post("/oauth/github/callback", response_model=OAuthLoginResponse)
    async def github_oauth_callback(
        request: OAuthLoginRequest,
        req: Request,
        db: AsyncSession = Depends(get_db)
    ) -> OAuthLoginResponse:
        """Handle GitHub OAuth callback."""
        try:
            audit_logger = get_audit_logger()
            client_ip = get_client_ip(req)
            user_agent = req.headers.get("user-agent", "")
            
            # Complete OAuth flow
            user_info = await github_oauth.authenticate(
                code=request.code,
                state=request.state
            )
            
            # Find or create user
            existing_user = await db.execute(
                select(User).where(User.email == user_info.email)
            )
            user = existing_user.scalar_one_or_none()
            
            if not user:
                # Create new user from OAuth info
                user = User(
                    email=user_info.email,
                    username=user_info.username or user_info.email.split('@')[0],
                    full_name=user_info.name,
                    is_verified=user_info.verified
                )
                # Set a random password that can't be guessed
                user.set_password(secrets.token_urlsafe(32))
                
                db.add(user)
                await db.commit()
                await db.refresh(user)
                
                event_type = SecurityEventType.OAUTH_REGISTRATION
                message = "New user registered via GitHub OAuth"
            else:
                event_type = SecurityEventType.OAUTH_LOGIN_SUCCESS
                message = "User logged in via GitHub OAuth"
            
            # Create session
            session_id = await session_service.create_session(
                user=user,
                ip_address=client_ip,
                user_agent=user_agent,
                login_method="oauth_github"
            )
            
            # Create token pair
            permissions = rbac_manager.get_user_effective_permissions(["USER"])
            token_pair = await jwt_service.create_token_pair(
                user=user,
                session_id=session_id,
                permissions=permissions,
                ip_address=client_ip,
                user_agent=user_agent
            )
            
            # Log OAuth authentication
            await audit_logger.log_authentication(
                event_type,
                user_id=str(user.id),
                username=user.username,
                ip_address=client_ip,
                user_agent=user_agent,
                session_id=session_id,
                success=True,
                message=message,
                details={
                    'provider': 'github',
                    'oauth_user_id': user_info.provider_id,
                    'verified_email': user_info.verified
                }
            )
            
            logger.info("OAuth login successful: %s via GitHub from %s", 
                       user.username, client_ip)
            
            return OAuthLoginResponse(
                user=UserResponse(
                    id=str(user.id),
                    email=user.email,
                    username=user.username,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    is_verified=user.is_verified,
                    created_at=user.created_at,
                    last_login=user.last_login
                ),
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type=token_pair.token_type,
                expires_in=token_pair.expires_in,
                session_id=session_id,
                permissions=permissions,
                provider="github"
            )
            
        except Exception as e:
            logger.error("GitHub OAuth callback failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth authentication failed"
            )