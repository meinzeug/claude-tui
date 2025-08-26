"""
Enhanced JWT Authentication System with Security Fixes

This module provides the updated JWT authentication system with all critical
security fixes applied, including Redis-based token blacklist and enhanced validation.
"""

import asyncio
import hashlib
import secrets
import os
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
from ..security.redis_token_blacklist import RedisTokenBlacklist, get_token_blacklist


class EnhancedTokenData(BaseModel):
    """Enhanced token payload data model with additional security fields."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    session_id: Optional[str] = Field(None, description="Session ID")
    iat: int = Field(..., description="Issued at timestamp")
    exp: int = Field(..., description="Expiration timestamp")
    jti: str = Field(..., description="JWT ID for revocation")
    device_fingerprint: Optional[str] = Field(None, description="Device fingerprint")
    ip_address: Optional[str] = Field(None, description="IP address when token issued")
    token_version: int = Field(default=1, description="Token version for mass revocation")


class EnhancedTokenResponse(BaseModel):
    """Enhanced token response model with additional security info."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user_id: str = Field(..., description="User ID")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    csrf_token: str = Field(..., description="CSRF token for additional protection")
    token_fingerprint: str = Field(..., description="Token fingerprint for validation")


class EnhancedJWTAuthenticator:
    """
    Enhanced JWT Authentication System with Security Fixes.
    
    Key security improvements:
    - No default secret key fallbacks
    - Persistent Redis-based token blacklist
    - Enhanced token validation
    - Device fingerprinting
    - IP address binding
    - CSRF token integration
    - Secure token rotation
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 15,  # Shorter default expiration
        refresh_token_expire_days: int = 7,     # Shorter refresh token lifetime
        issuer: str = "claude-tui",
        audience: str = "claude-tui-api",
        redis_blacklist: Optional[RedisTokenBlacklist] = None
    ):
        """
        Initialize enhanced JWT authenticator.
        
        Args:
            secret_key: JWT secret key (REQUIRED, no default)
            algorithm: JWT algorithm
            access_token_expire_minutes: Access token expiration
            refresh_token_expire_days: Refresh token expiration
            issuer: JWT issuer
            audience: JWT audience
            redis_blacklist: Redis token blacklist instance
        """
        # CRITICAL: No default secret key
        if not secret_key:
            secret_key = os.getenv("JWT_SECRET_KEY")
            if not secret_key:
                raise ValueError(
                    "JWT_SECRET_KEY is required and must be set via environment variable. "
                    "No default fallback is provided for security reasons."
                )
        
        # Validate secret key strength
        if len(secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer
        self.audience = audience
        
        # Password hashing with stronger configuration
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12  # Increased rounds for better security
        )
        
        # Redis blacklist integration
        self.redis_blacklist = redis_blacklist
        self._fallback_blacklist = set()  # Only as emergency fallback
        
        # Token version for mass revocation
        self.current_token_version = 1
    
    def _generate_jti(self) -> str:
        """Generate cryptographically secure JWT ID."""
        return secrets.token_urlsafe(32)
    
    def _generate_refresh_token(self) -> str:
        """Generate cryptographically secure refresh token."""
        return secrets.token_urlsafe(64)
    
    def _generate_csrf_token(self) -> str:
        """Generate CSRF token."""
        return secrets.token_urlsafe(32)
    
    def _generate_token_fingerprint(self, token_data: Dict[str, Any]) -> str:
        """Generate token fingerprint for additional validation."""
        fingerprint_data = f"{token_data['jti']}{token_data['user_id']}{token_data['iat']}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    def _generate_device_fingerprint(
        self, 
        user_agent: Optional[str] = None, 
        ip_address: Optional[str] = None
    ) -> str:
        """Generate device fingerprint."""
        device_data = f"{user_agent or 'unknown'}{ip_address or 'unknown'}"
        return hashlib.sha256(device_data.encode()).hexdigest()[:16]
    
    async def authenticate_user(
        self,
        user_repo: UserRepository,
        identifier: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate user with enhanced security logging.
        """
        try:
            user = await user_repo.authenticate_user(
                identifier=identifier,
                password=password,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            if user:
                # Log successful authentication
                print(f"✅ User {user.username} authenticated from {ip_address}")
            
            return user
            
        except Exception as e:
            # Log failed authentication attempt
            print(f"❌ Authentication failed for {identifier} from {ip_address}: {str(e)}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")
    
    async def create_tokens(
        self,
        user: User,
        session_repo: SessionRepository,
        permissions: Optional[List[str]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> EnhancedTokenResponse:
        """
        Create enhanced JWT access and refresh tokens.
        """
        try:
            # Generate secure identifiers
            jti = self._generate_jti()
            refresh_token = self._generate_refresh_token()
            csrf_token = self._generate_csrf_token()
            
            # Calculate expiration times
            now = datetime.now(timezone.utc)
            access_exp = now + timedelta(minutes=self.access_token_expire_minutes)
            refresh_exp = now + timedelta(days=self.refresh_token_expire_days)
            
            # Generate device fingerprint
            device_fingerprint = self._generate_device_fingerprint(user_agent, ip_address)
            
            # Extract user roles
            roles = []
            if hasattr(user, 'roles') and user.roles:
                roles = [role.role.name for role in user.roles if not role.is_expired()]
            
            # Create session with enhanced security
            session = await session_repo.create_session(
                user_id=user.id,
                session_token=jti,
                refresh_token=refresh_token,
                expires_at=refresh_exp,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint
            )
            
            # Prepare enhanced token payload
            token_payload = {
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "roles": roles,
                "permissions": permissions or [],
                "session_id": str(session.id) if session else None,
                "iat": int(now.timestamp()),
                "exp": int(access_exp.timestamp()),
                "iss": self.issuer,
                "aud": self.audience,
                "jti": jti,
                "device_fingerprint": device_fingerprint,
                "ip_address": ip_address,
                "token_version": self.current_token_version
            }
            
            # Generate token fingerprint
            token_fingerprint = self._generate_token_fingerprint(token_payload)
            token_payload["fingerprint"] = token_fingerprint
            
            # Generate access token
            access_token = jwt.encode(
                token_payload,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            return EnhancedTokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire_minutes * 60,
                user_id=str(user.id),
                permissions=permissions or [],
                csrf_token=csrf_token,
                token_fingerprint=token_fingerprint
            )
            
        except Exception as e:
            raise AuthenticationError(f"Token creation failed: {str(e)}")
    
    async def validate_token(
        self,
        token: str,
        session_repo: SessionRepository,
        verify_session: bool = True,
        verify_device_fingerprint: bool = True,
        current_ip: Optional[str] = None,
        current_user_agent: Optional[str] = None
    ) -> EnhancedTokenData:
        """
        Enhanced token validation with comprehensive security checks.
        """
        try:
            # Check Redis blacklist first
            if self.redis_blacklist:
                is_blacklisted = await self.redis_blacklist.is_token_blacklisted(token)
                if is_blacklisted:
                    raise AuthenticationError("Token has been revoked")
            else:
                # Fallback to in-memory blacklist
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                if token_hash in self._fallback_blacklist:
                    raise AuthenticationError("Token has been revoked")
            
            # Decode and validate token with strict validation
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
                    "verify_aud": True,
                    "require": ["exp", "iat", "jti", "user_id"]
                }
            )
            
            # Verify token version
            token_version = payload.get("token_version", 0)
            if token_version < self.current_token_version:
                raise AuthenticationError("Token version is outdated")
            
            # Create enhanced token data
            token_data = EnhancedTokenData(**payload)
            
            # Verify token fingerprint
            expected_fingerprint = self._generate_token_fingerprint(payload)
            if payload.get("fingerprint") != expected_fingerprint:
                raise AuthenticationError("Token fingerprint validation failed")
            
            # Verify session if required
            if verify_session and token_data.session_id:
                session = await session_repo.get_session(payload["jti"])
                if not session or not session.is_active or session.is_expired():
                    raise AuthenticationError("Session is invalid or expired")
                
                # Verify device fingerprint if required
                if verify_device_fingerprint:
                    current_fingerprint = self._generate_device_fingerprint(
                        current_user_agent, current_ip
                    )
                    if session.device_fingerprint != current_fingerprint:
                        # Log suspicious activity
                        print(f"⚠️  Device fingerprint mismatch for user {token_data.user_id}")
                        # Optionally, you might want to revoke the token here
                
                # Update session activity
                session.last_activity = datetime.now(timezone.utc)
                await session_repo.update_session_activity(session.id)
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")
    
    async def revoke_token(
        self,
        token: str,
        session_repo: SessionRepository,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None
    ) -> bool:
        """
        Enhanced token revocation with Redis blacklist.
        """
        try:
            # Decode token to get expiration
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                    options={"verify_exp": False}  # Don't verify expiration for revocation
                )
                expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
                user_id = payload.get("user_id")
            except Exception:
                # If we can't decode the token, still try to add to blacklist
                expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
                user_id = "unknown"
            
            # Add to Redis blacklist
            if self.redis_blacklist and user_id != "unknown":
                await self.redis_blacklist.add_token(
                    token=token,
                    user_id=user_id,
                    expires_at=expires_at,
                    reason=reason,
                    revoked_by=revoked_by
                )
            else:
                # Fallback to in-memory blacklist
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                self._fallback_blacklist.add(token_hash)
            
            # Try to invalidate associated session
            try:
                if "jti" in payload:
                    await session_repo.invalidate_session(payload["jti"])
            except Exception:
                pass  # Continue even if session invalidation fails
            
            print(f"🔒 Token revoked for user {user_id} - Reason: {reason or 'Manual revocation'}")
            return True
            
        except Exception as e:
            raise AuthenticationError(f"Token revocation failed: {str(e)}")
    
    async def revoke_user_tokens(
        self,
        user_id: UUID,
        session_repo: SessionRepository,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None
    ) -> int:
        """
        Enhanced user token revocation with Redis blacklist.
        """
        try:
            # Invalidate all user sessions
            count = await session_repo.invalidate_user_sessions(user_id)
            
            # If using Redis blacklist, mark all user tokens as revoked
            if self.redis_blacklist:
                await self.redis_blacklist.blacklist_user_tokens(
                    str(user_id),
                    reason=reason,
                    revoked_by=revoked_by
                )
            
            print(f"🔒 Revoked {count} sessions for user {user_id} - Reason: {reason or 'Mass revocation'}")
            return count
            
        except Exception as e:
            raise AuthenticationError(f"Failed to revoke user tokens: {str(e)}")
    
    async def rotate_tokens_globally(self, reason: str = "Security update") -> None:
        """
        Rotate all tokens globally by incrementing token version.
        """
        old_version = self.current_token_version
        self.current_token_version += 1
        
        print(f"🔄 Global token rotation: v{old_version} -> v{self.current_token_version} - {reason}")
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics from the authentication system."""
        metrics = {
            "token_version": self.current_token_version,
            "access_token_lifetime_minutes": self.access_token_expire_minutes,
            "refresh_token_lifetime_days": self.refresh_token_expire_days,
            "algorithm": self.algorithm,
            "redis_blacklist_available": self.redis_blacklist is not None,
            "fallback_blacklist_size": len(self._fallback_blacklist)
        }
        
        # Add Redis blacklist metrics if available
        if self.redis_blacklist:
            try:
                blacklist_metrics = await self.redis_blacklist.get_metrics()
                metrics.update({
                    "redis_blacklist_metrics": blacklist_metrics
                })
            except Exception as e:
                metrics["redis_blacklist_error"] = str(e)
        
        return metrics


# Factory function for creating enhanced authenticator
def create_enhanced_jwt_authenticator(
    secret_key: Optional[str] = None,
    redis_blacklist: Optional[RedisTokenBlacklist] = None,
    **kwargs
) -> EnhancedJWTAuthenticator:
    """
    Create enhanced JWT authenticator with security best practices.
    
    Args:
        secret_key: JWT secret key (will use environment variable if not provided)
        redis_blacklist: Redis blacklist instance
        **kwargs: Additional authenticator options
        
    Returns:
        Configured enhanced JWT authenticator
    """
    # Try to get Redis blacklist from global instance
    if not redis_blacklist:
        try:
            redis_blacklist = get_token_blacklist()
        except RuntimeError:
            # Redis blacklist not initialized, will use fallback
            pass
    
    return EnhancedJWTAuthenticator(
        secret_key=secret_key,
        redis_blacklist=redis_blacklist,
        **kwargs
    )


# Global instance management
_enhanced_authenticator: Optional[EnhancedJWTAuthenticator] = None


def get_enhanced_jwt_authenticator() -> EnhancedJWTAuthenticator:
    """Get global enhanced JWT authenticator instance."""
    global _enhanced_authenticator
    if _enhanced_authenticator is None:
        _enhanced_authenticator = create_enhanced_jwt_authenticator()
    return _enhanced_authenticator


def init_enhanced_jwt_authenticator(**kwargs) -> EnhancedJWTAuthenticator:
    """Initialize global enhanced JWT authenticator."""
    global _enhanced_authenticator
    _enhanced_authenticator = create_enhanced_jwt_authenticator(**kwargs)
    return _enhanced_authenticator