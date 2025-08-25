"""
Enhanced JWT Service with comprehensive token management.

Provides secure JWT access/refresh token generation, validation,
blacklist management, and session integration.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import redis
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from ..core.exceptions import AuthenticationError, ValidationError
from ..api.models.user import User, UserSession


class TokenPair(BaseModel):
    """Token pair response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    permissions: List[str] = Field(default_factory=list, description="User permissions")


class TokenData(BaseModel):
    """Decoded token data."""
    user_id: str
    username: str
    email: str
    role: str
    permissions: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    jti: str
    iat: int
    exp: int
    token_type: str = "access"


class JWTService:
    """
    Enhanced JWT Service with comprehensive security features.
    
    Features:
    - Access/Refresh token generation
    - Token validation and refresh logic
    - Redis-based token blacklist
    - Session management integration
    - Audit logging
    """
    
    def __init__(
        self,
        secret_key: str,
        redis_client: Optional[redis.Redis] = None,
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
        
        # Redis for token blacklist
        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=0, decode_responses=True
        )
        
        # Password context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Blacklist key prefix
        self.blacklist_prefix = "auth:blacklist:"
        self.refresh_prefix = "auth:refresh:"
    
    def _generate_jti(self) -> str:
        """Generate unique JWT ID."""
        return secrets.token_hex(16)
    
    def _generate_refresh_token(self) -> str:
        """Generate secure refresh token."""
        return secrets.token_urlsafe(64)
    
    async def create_token_pair(
        self,
        user: User,
        session_id: str,
        permissions: Optional[List[str]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenPair:
        """
        Create access and refresh token pair.
        
        Args:
            user: Authenticated user
            session_id: Session ID
            permissions: User permissions
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            TokenPair with access and refresh tokens
        """
        try:
            # Generate tokens
            access_jti = self._generate_jti()
            refresh_token = self._generate_refresh_token()
            
            # Calculate expiration
            now = datetime.now(timezone.utc)
            access_exp = now + timedelta(minutes=self.access_token_expire_minutes)
            refresh_exp = now + timedelta(days=self.refresh_token_expire_days)
            
            # Prepare access token payload
            access_payload = {
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "permissions": permissions or [],
                "session_id": session_id,
                "token_type": "access",
                "jti": access_jti,
                "iat": int(now.timestamp()),
                "exp": int(access_exp.timestamp()),
                "iss": self.issuer,
                "aud": self.audience
            }
            
            # Generate access token
            access_token = jwt.encode(
                access_payload,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            # Store refresh token in Redis with metadata
            refresh_key = f"{self.refresh_prefix}{refresh_token}"
            refresh_data = {
                "user_id": str(user.id),
                "session_id": session_id,
                "access_jti": access_jti,
                "ip_address": ip_address or "",
                "user_agent": user_agent or "",
                "created_at": now.isoformat(),
                "expires_at": refresh_exp.isoformat()
            }
            
            # Store with expiration
            await self._redis_hmset(refresh_key, refresh_data)
            await self._redis_expire(refresh_key, int(self.refresh_token_expire_days * 24 * 3600))
            
            return TokenPair(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire_minutes * 60,
                user_id=str(user.id),
                session_id=session_id,
                permissions=permissions or []
            )
            
        except Exception as e:
            raise AuthenticationError(f"Token creation failed: {str(e)}")
    
    async def validate_access_token(self, token: str) -> TokenData:
        """
        Validate access token.
        
        Args:
            token: JWT access token
            
        Returns:
            TokenData with decoded information
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Check blacklist
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            blacklist_key = f"{self.blacklist_prefix}{token_hash}"
            
            if await self._redis_exists(blacklist_key):
                raise AuthenticationError("Token has been revoked")
            
            # Decode token
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
            
            # Validate token type
            if payload.get("token_type") != "access":
                raise AuthenticationError("Invalid token type")
            
            return TokenData(**payload)
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Access token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid access token: {str(e)}")
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")
    
    async def refresh_access_token(
        self,
        refresh_token: str,
        user_service,  # To avoid circular imports
    ) -> TokenPair:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            user_service: User service instance
            
        Returns:
            New token pair
        """
        try:
            refresh_key = f"{self.refresh_prefix}{refresh_token}"
            
            # Get refresh token data
            refresh_data = await self._redis_hgetall(refresh_key)
            if not refresh_data:
                raise AuthenticationError("Invalid or expired refresh token")
            
            # Check expiration
            expires_at = datetime.fromisoformat(refresh_data["expires_at"])
            if datetime.now(timezone.utc) >= expires_at:
                # Cleanup expired token
                await self._redis_delete(refresh_key)
                raise AuthenticationError("Refresh token has expired")
            
            # Get user
            user = await user_service.get_user_by_id(refresh_data["user_id"])
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Blacklist old access token
            old_jti = refresh_data["access_jti"]
            await self.blacklist_token_by_jti(old_jti, self.access_token_expire_minutes * 60)
            
            # Create new token pair
            session_id = refresh_data["session_id"]
            permissions = []  # Get from user service if needed
            
            new_tokens = await self.create_token_pair(
                user=user,
                session_id=session_id,
                permissions=permissions,
                ip_address=refresh_data.get("ip_address"),
                user_agent=refresh_data.get("user_agent")
            )
            
            # Update refresh token data
            refresh_data["access_jti"] = jwt.decode(
                new_tokens.access_token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )["jti"]
            
            await self._redis_hmset(refresh_key, refresh_data)
            
            return new_tokens
            
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Token refresh failed: {str(e)}")
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke access token by adding to blacklist.
        
        Args:
            token: JWT token to revoke
            
        Returns:
            True if successful
        """
        try:
            # Decode to get expiration
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            # Calculate remaining TTL
            exp_timestamp = payload.get("exp", 0)
            current_timestamp = int(datetime.now(timezone.utc).timestamp())
            ttl = max(0, exp_timestamp - current_timestamp)
            
            if ttl > 0:
                # Add to blacklist
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                blacklist_key = f"{self.blacklist_prefix}{token_hash}"
                
                await self._redis_setex(blacklist_key, ttl, "revoked")
            
            return True
            
        except Exception as e:
            raise AuthenticationError(f"Token revocation failed: {str(e)}")
    
    async def blacklist_token_by_jti(self, jti: str, ttl_seconds: int) -> bool:
        """
        Blacklist token by JTI.
        
        Args:
            jti: JWT ID
            ttl_seconds: TTL in seconds
            
        Returns:
            True if successful
        """
        try:
            jti_hash = hashlib.sha256(jti.encode()).hexdigest()
            blacklist_key = f"{self.blacklist_prefix}{jti_hash}"
            
            await self._redis_setex(blacklist_key, ttl_seconds, "revoked")
            return True
            
        except Exception as e:
            raise AuthenticationError(f"JTI blacklisting failed: {str(e)}")
    
    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """
        Revoke refresh token.
        
        Args:
            refresh_token: Refresh token to revoke
            
        Returns:
            True if successful
        """
        try:
            refresh_key = f"{self.refresh_prefix}{refresh_token}"
            result = await self._redis_delete(refresh_key)
            return result > 0
            
        except Exception as e:
            raise AuthenticationError(f"Refresh token revocation failed: {str(e)}")
    
    async def revoke_all_user_tokens(self, user_id: str) -> int:
        """
        Revoke all tokens for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of tokens revoked
        """
        try:
            pattern = f"{self.refresh_prefix}*"
            keys = await self._redis_keys(pattern)
            
            revoked_count = 0
            for key in keys:
                refresh_data = await self._redis_hgetall(key)
                if refresh_data.get("user_id") == user_id:
                    await self._redis_delete(key)
                    # Also blacklist associated access token
                    access_jti = refresh_data.get("access_jti")
                    if access_jti:
                        await self.blacklist_token_by_jti(
                            access_jti,
                            self.access_token_expire_minutes * 60
                        )
                    revoked_count += 1
            
            return revoked_count
            
        except Exception as e:
            raise AuthenticationError(f"User token revocation failed: {str(e)}")
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Cleanup expired refresh tokens.
        
        Returns:
            Number of tokens cleaned up
        """
        try:
            pattern = f"{self.refresh_prefix}*"
            keys = await self._redis_keys(pattern)
            
            now = datetime.now(timezone.utc)
            cleaned_count = 0
            
            for key in keys:
                refresh_data = await self._redis_hgetall(key)
                if refresh_data:
                    expires_at = datetime.fromisoformat(refresh_data["expires_at"])
                    if now >= expires_at:
                        await self._redis_delete(key)
                        cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            raise AuthenticationError(f"Token cleanup failed: {str(e)}")
    
    # Redis helper methods (async wrappers)
    async def _redis_setex(self, key: str, ttl: int, value: str) -> bool:
        """Set key with expiration."""
        try:
            return self.redis_client.setex(key, ttl, value)
        except Exception:
            return False
    
    async def _redis_hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set hash fields."""
        try:
            return self.redis_client.hset(key, mapping=mapping)
        except Exception:
            return False
    
    async def _redis_hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields."""
        try:
            return self.redis_client.hgetall(key)
        except Exception:
            return {}
    
    async def _redis_exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return self.redis_client.exists(key) > 0
        except Exception:
            return False
    
    async def _redis_delete(self, key: str) -> int:
        """Delete key."""
        try:
            return self.redis_client.delete(key)
        except Exception:
            return 0
    
    async def _redis_expire(self, key: str, ttl: int) -> bool:
        """Set key expiration."""
        try:
            return self.redis_client.expire(key, ttl)
        except Exception:
            return False
    
    async def _redis_keys(self, pattern: str) -> List[str]:
        """Get keys by pattern."""
        try:
            return self.redis_client.keys(pattern)
        except Exception:
            return []
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
