"""
JWT Authentication System for claude-tui.

Implements secure JWT-based authentication with comprehensive security features:
- Token generation and validation
- Refresh token mechanism
- Role-based access control integration
- Security audit logging
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from ..core.exceptions import AuthenticationError, ValidationError
from ..database.models import User, UserSession
from ..database.repositories import UserRepository, SessionRepository
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool


class TokenData(BaseModel):
    """Token payload data model."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    session_id: Optional[str] = Field(None, description="Session ID")
    iat: int = Field(..., description="Issued at timestamp")
    exp: int = Field(..., description="Expiration timestamp")
    jti: str = Field(..., description="JWT ID for revocation")


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user_id: str = Field(..., description="User ID")
    permissions: List[str] = Field(default_factory=list, description="User permissions")


class JWTAuthenticator:
    """
    JWT Authentication and Authorization System.
    
    Provides comprehensive JWT-based authentication with security features:
    - Secure token generation and validation
    - Refresh token mechanism
    - Session management
    - Role-based permissions
    - Audit logging
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 30,
        issuer: str = "claude-tui",
        audience: str = "claude-tui-api"
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer
        self.audience = audience
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Token blacklist management with Redis persistence
        self._blacklisted_tokens = set()  # Fallback in-memory blacklist
        self._redis_blacklist_enabled = False
        self._init_redis_blacklist()
    
    def _generate_jti(self) -> str:
        """Generate unique JWT ID."""
        return secrets.token_hex(16)
    
    def _generate_refresh_token(self) -> str:
        """Generate secure refresh token."""
        return secrets.token_urlsafe(64)
    
    async def authenticate_user(
        self,
        user_repo: UserRepository,
        identifier: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate user with enhanced security.
        
        Args:
            user_repo: User repository instance
            identifier: Email or username
            password: User password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Authenticated user or None
        """
        try:
            user = await user_repo.authenticate_user(
                identifier=identifier,
                password=password,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return user
            
        except Exception as e:
            raise AuthenticationError(f"Authentication failed: {str(e)}")
    
    async def create_tokens(
        self,
        user: User,
        session_repo: SessionRepository,
        permissions: Optional[List[str]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """
        Create JWT access and refresh tokens.
        
        Args:
            user: Authenticated user
            session_repo: Session repository
            permissions: User permissions
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Token response with access and refresh tokens
        """
        try:
            # Generate unique identifiers
            jti = self._generate_jti()
            refresh_token = self._generate_refresh_token()
            
            # Calculate expiration times
            access_exp = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
            refresh_exp = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
            
            # Extract user roles
            roles = []
            if hasattr(user, 'roles') and user.roles:
                roles = [role.role.name for role in user.roles if not role.is_expired()]
            
            # Create session
            session = await session_repo.create_session(
                user_id=user.id,
                session_token=jti,
                refresh_token=refresh_token,
                expires_at=refresh_exp,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Prepare token payload
            token_payload = {
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "roles": roles,
                "permissions": permissions or [],
                "session_id": str(session.id) if session else None,
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "exp": int(access_exp.timestamp()),
                "iss": self.issuer,
                "aud": self.audience,
                "jti": jti
            }
            
            # Generate access token
            access_token = jwt.encode(
                token_payload,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire_minutes * 60,
                user_id=str(user.id),
                permissions=permissions or []
            )
            
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise AuthenticationError(f"Failed to create tokens: {str(e)}")
    
    def _init_redis_blacklist(self):
        """Initialize Redis blacklist connection."""
        try:
            # Try to connect to Redis for persistent blacklist
            import os
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_db = int(os.getenv("REDIS_BLACKLIST_DB", 2))
            
            self._redis_pool = ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                max_connections=10
            )
            self._redis_blacklist_enabled = True
            logger.info("Redis blacklist initialized successfully")
        except Exception as e:
            logger.warning(f"Redis blacklist not available, using in-memory: {e}")
            self._redis_blacklist_enabled = False
    
    async def revoke_token(self, jti: str, expiry: datetime, user_id: Optional[str] = None) -> bool:
        """Revoke a JWT token by adding it to blacklist."""
        try:
            if self._redis_blacklist_enabled:
                # Use Redis for persistent blacklist
                client = redis.Redis(connection_pool=self._redis_pool)
                ttl = max(int((expiry - datetime.now(timezone.utc)).total_seconds()), 60)
                
                # Store in Redis with expiry
                blacklist_key = f"jwt_blacklist:{jti}"
                await client.setex(
                    blacklist_key,
                    ttl,
                    json.dumps({
                        "jti": jti,
                        "user_id": user_id or "",
                        "revoked_at": datetime.now(timezone.utc).isoformat()
                    })
                )
                logger.info(f"Token {jti} revoked in Redis blacklist")
                return True
            else:
                # Fallback to in-memory blacklist
                self._blacklisted_tokens.add(jti)
                logger.info(f"Token {jti} revoked in memory blacklist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            # Fallback to in-memory on Redis failure
            self._blacklisted_tokens.add(jti)
            return True
    
    async def is_token_blacklisted(self, jti: str) -> bool:
        """Check if a token is blacklisted."""
        try:
            # Check in-memory first (fast path)
            if jti in self._blacklisted_tokens:
                return True
            
            if self._redis_blacklist_enabled:
                # Check Redis blacklist
                client = redis.Redis(connection_pool=self._redis_pool)
                blacklist_key = f"jwt_blacklist:{jti}"
                exists = await client.exists(blacklist_key)
                return bool(exists)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check blacklist: {e}")
            # On error, check in-memory only
            return jti in self._blacklisted_tokens
    
    async def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate JWT token with blacklist checking."""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer
            )
            
            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and await self.is_token_blacklisted(jti):
                logger.warning(f"Blacklisted token attempted: {jti}")
                return None
            
            # Convert to TokenData
            return TokenData(**payload)
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    async def revoke_user_tokens(self, user_id: str, session_repo: SessionRepository) -> int:
        """Revoke all tokens for a user."""
        try:
            # Get all active sessions for user
            sessions = await session_repo.get_user_sessions(user_id)
            revoked_count = 0
            
            for session in sessions:
                if session.session_token:
                    await self.revoke_token(
                        session.session_token,
                        session.expires_at,
                        user_id
                    )
                    revoked_count += 1
            
            # Invalidate sessions in database
            await session_repo.invalidate_user_sessions(user_id)
            
            logger.info(f"Revoked {revoked_count} tokens for user {user_id}")
            return revoked_count
            
        except Exception as e:
            logger.error(f"Failed to revoke user tokens: {e}")
            return 0
            )
            
        except Exception as e:
            raise AuthenticationError(f"Token creation failed: {str(e)}")
    
    async def validate_token(
        self,
        token: str,
        session_repo: SessionRepository,
        verify_session: bool = True
    ) -> TokenData:
        """
        Validate JWT token.
        
        Args:
            token: JWT token to validate
            session_repo: Session repository
            verify_session: Whether to verify session existence
            
        Returns:
            Validated token data
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Check if token is blacklisted
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if token_hash in self._blacklisted_tokens:
                raise AuthenticationError("Token has been revoked")
            
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True
                }
            )
            
            # Create token data
            token_data = TokenData(**payload)
            
            # Verify session if required
            if verify_session and token_data.session_id:
                session = await session_repo.get_session(payload["jti"])
                if not session or not session.is_active or session.is_expired():
                    raise AuthenticationError("Session is invalid or expired")
                
                # Update session activity
                session.last_activity = datetime.now(timezone.utc)
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")
    
    async def refresh_token(
        self,
        refresh_token: str,
        session_repo: SessionRepository,
        user_repo: UserRepository
    ) -> TokenResponse:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            session_repo: Session repository
            user_repo: User repository
            
        Returns:
            New token response
        """
        try:
            # Validate refresh token format
            if not refresh_token or len(refresh_token) < 32:
                raise AuthenticationError("Invalid refresh token format")
            
            # Find session by refresh token
            session = await session_repo.find_by_refresh_token(refresh_token)
            
            if not session:
                raise AuthenticationError("Invalid refresh token")
            
            # Check if session is expired or inactive
            if not session.is_active or session.is_expired():
                await session_repo.invalidate_session(session.session_token)
                raise AuthenticationError("Session expired")
            
            # Get user data
            user = await user_repo.get_user_by_id(session.user_id)
            if not user or not user.is_active:
                await session_repo.invalidate_session(session.session_token)
                raise AuthenticationError("User not found or inactive")
            
            # Generate new tokens
            new_jti = self._generate_jti()
            new_refresh_token = self._generate_refresh_token()
            
            # Calculate expiration times
            access_exp = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
            refresh_exp = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
            
            # Extract user roles
            roles = []
            if hasattr(user, 'roles') and user.roles:
                roles = [role.role.name for role in user.roles if not role.is_expired()]
            
            # Update session with new tokens
            await session_repo.update_session_tokens(
                session.id,
                session_token=new_jti,
                refresh_token=new_refresh_token,
                expires_at=refresh_exp
            )
            
            # Prepare new token payload
            token_payload = {
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "roles": roles,
                "permissions": [],  # Could be fetched from user roles
                "session_id": str(session.id),
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "exp": int(access_exp.timestamp()),
                "iss": self.issuer,
                "aud": self.audience,
                "jti": new_jti
            }
            
            # Generate new access token
            access_token = jwt.encode(
                token_payload,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            # Update session last activity
            session.last_activity = datetime.now(timezone.utc)
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=self.access_token_expire_minutes * 60,
                user_id=str(user.id),
                permissions=[]
            )
            
        except Exception as e:
            raise AuthenticationError(f"Token refresh failed: {str(e)}")
    
    async def revoke_token(
        self,
        token: str,
        session_repo: SessionRepository
    ) -> bool:
        """
        Revoke JWT token.
        
        Args:
            token: JWT token to revoke
            session_repo: Session repository
            
        Returns:
            True if token was revoked successfully
        """
        try:
            # Add token hash to blacklist
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            self._blacklisted_tokens.add(token_hash)
            
            # Try to invalidate associated session
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                    options={"verify_exp": False}  # Don't verify expiration for revocation
                )
                
                if "jti" in payload:
                    await session_repo.invalidate_session(payload["jti"])
                
            except Exception:
                # Continue even if session invalidation fails
                pass
            
            return True
            
        except Exception as e:
            raise AuthenticationError(f"Token revocation failed: {str(e)}")
    
    async def revoke_user_sessions(
        self,
        user_id: UUID,
        session_repo: SessionRepository
    ) -> int:
        """
        Revoke all sessions for a user.
        
        Args:
            user_id: User ID
            session_repo: Session repository
            
        Returns:
            Number of sessions revoked
        """
        try:
            count = await session_repo.invalidate_user_sessions(user_id)
            return count
            
        except Exception as e:
            raise AuthenticationError(f"Failed to revoke user sessions: {str(e)}")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        return self.pwd_context.hash(password)
    
    async def change_password(
        self,
        user_repo: UserRepository,
        session_repo: SessionRepository,
        user_id: UUID,
        current_password: str,
        new_password: str,
        revoke_sessions: bool = True
    ) -> bool:
        """
        Change user password with security checks.
        
        Args:
            user_repo: User repository
            session_repo: Session repository
            user_id: User ID
            current_password: Current password
            new_password: New password
            revoke_sessions: Whether to revoke all user sessions
            
        Returns:
            True if password was changed successfully
        """
        try:
            success = await user_repo.change_password(
                user_id=user_id,
                old_password=current_password,
                new_password=new_password
            )
            
            if success and revoke_sessions:
                # Revoke all existing sessions to force re-authentication
                await self.revoke_user_sessions(user_id, session_repo)
            
            return success
            
        except Exception as e:
            raise AuthenticationError(f"Password change failed: {str(e)}")


class PermissionChecker:
    """
    Permission checking utility.
    
    Provides methods to check user permissions and roles.
    """
    
    @staticmethod
    def has_permission(token_data: TokenData, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in token_data.permissions
    
    @staticmethod
    def has_role(token_data: TokenData, role: str) -> bool:
        """Check if user has specific role."""
        return role in token_data.roles
    
    @staticmethod
    def has_any_role(token_data: TokenData, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in token_data.roles for role in roles)
    
    @staticmethod
    def has_all_roles(token_data: TokenData, roles: List[str]) -> bool:
        """Check if user has all specified roles."""
        return all(role in token_data.roles for role in roles)
    
    @staticmethod
    def is_admin(token_data: TokenData) -> bool:
        """Check if user is admin."""
        return "ADMIN" in token_data.roles or "SUPERUSER" in token_data.roles
    
    @staticmethod
    def can_access_resource(
        token_data: TokenData,
        resource_type: str,
        action: str,
        resource_owner_id: Optional[str] = None
    ) -> bool:
        """
        Check if user can access resource.
        
        Args:
            token_data: Token data with user information
            resource_type: Type of resource (project, task, etc.)
            action: Action to perform (read, write, delete)
            resource_owner_id: ID of resource owner
            
        Returns:
            True if user can access resource
        """
        # Admin can access everything
        if PermissionChecker.is_admin(token_data):
            return True
        
        # Check specific permission
        permission = f"{resource_type}:{action}"
        if permission in token_data.permissions:
            return True
        
        # Owner can access their own resources
        if resource_owner_id and resource_owner_id == token_data.user_id:
            return True
        
        # Check wildcard permissions
        wildcard_permission = f"{resource_type}:*"
        if wildcard_permission in token_data.permissions:
            return True
        
        return False


# Utility functions for FastAPI integration
def create_jwt_authenticator(config: Dict[str, Any]) -> JWTAuthenticator:
    """Create JWT authenticator from configuration."""
    return JWTAuthenticator(
        secret_key=config["jwt_secret_key"],  # No default fallback
        algorithm=config.get("jwt_algorithm", "HS256"),
        access_token_expire_minutes=config.get("jwt_access_token_expire_minutes", 30),
        refresh_token_expire_days=config.get("jwt_refresh_token_expire_days", 30),
        issuer=config.get("jwt_issuer", "claude-tui"),
        audience=config.get("jwt_audience", "claude-tui-api")
    )


def get_token_from_header(authorization: str) -> str:
    """Extract token from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
        return token
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )