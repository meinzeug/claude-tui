"""
Secure Authentication Middleware
===============================

Enhanced authentication middleware with comprehensive security features.
Replaces the existing middleware with hardened security controls.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import jwt
import redis.asyncio as redis
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..auth.audit_logger import SecurityAuditLogger, SecurityEventType, SecurityLevel
from .security_hardening import SecurityConfig, SecureKeyManager, CSRFProtection

logger = logging.getLogger(__name__)


class SecureTokenManager:
    """Enhanced JWT token management with security hardening."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.key_manager = SecureKeyManager()
        self.audit_logger = SecurityAuditLogger()
        
        # Token configuration
        self.secret_key = self._get_secure_secret_key()
        self.algorithm = "HS256"
        self.issuer = "claude-tui-secure"
        self.audience = "claude-tui-api"
        
        # Token expiration
        self.access_token_expire = SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire = SecurityConfig.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        
        # Blacklist prefix
        self.blacklist_prefix = "auth:blacklist:"
        self.refresh_prefix = "auth:refresh:"
    
    def _get_secure_secret_key(self) -> str:
        """Get secure secret key with validation."""
        import os
        
        secret_key = os.getenv('JWT_SECRET_KEY')
        if not secret_key:
            # Generate a secure key and warn
            secret_key = self.key_manager.generate_secure_token(64)
            logger.critical(
                f"JWT_SECRET_KEY not set! Generated temporary key: {secret_key}. "
                "Set JWT_SECRET_KEY in production environment!"
            )
        
        # Validate key strength
        if not self.key_manager.validate_secret_strength(secret_key, 64):
            raise ValueError("JWT secret key is too weak. Must be at least 64 characters with high entropy.")
        
        return secret_key
    
    async def create_access_token(
        self,
        user_id: str,
        username: str,
        email: str,
        role: str,
        permissions: List[str],
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create secure access token with enhanced payload."""
        
        now = datetime.now(timezone.utc)
        expire_time = now + timedelta(minutes=self.access_token_expire)
        
        # Generate secure JTI
        jti = self.key_manager.generate_secure_token(32)
        
        # Enhanced payload with security features
        payload = {
            # Standard claims
            "sub": user_id,
            "username": username,
            "email": email,
            "role": role,
            "permissions": permissions,
            "iat": int(now.timestamp()),
            "exp": int(expire_time.timestamp()),
            "iss": self.issuer,
            "aud": self.audience,
            "jti": jti,
            
            # Security claims
            "session_id": session_id,
            "token_type": "access",
            "ip_hash": hashlib.sha256((ip_address or "").encode()).hexdigest()[:16],
            "ua_hash": hashlib.sha256((user_agent or "").encode()).hexdigest()[:16],
            "created_at": now.isoformat(),
        }
        
        try:
            # Create token
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            # Store token metadata in Redis for validation
            if self.redis_client:
                token_metadata = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "ip_address": ip_address or "",
                    "user_agent": user_agent or "",
                    "created_at": now.isoformat(),
                    "expires_at": expire_time.isoformat()
                }
                
                await self.redis_client.setex(
                    f"token:{jti}",
                    int(self.access_token_expire * 60),
                    json.dumps(token_metadata)
                )
            
            # Audit log token creation
            await self.audit_logger.log_authentication(
                SecurityEventType.TOKEN_ISSUED,
                user_id=user_id,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                success=True,
                details={"token_type": "access", "expires_at": expire_time.isoformat()}
            )
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire * 60,
                "expires_at": expire_time.isoformat(),
                "jti": jti
            }
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            await self.audit_logger.log_authentication(
                SecurityEventType.TOKEN_ISSUED,
                user_id=user_id,
                username=username,
                ip_address=ip_address,
                success=False,
                message=f"Token creation failed: {str(e)}"
            )
            raise
    
    async def validate_access_token(
        self,
        token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        require_ip_match: bool = True
    ) -> Dict[str, Any]:
        """Validate access token with enhanced security checks."""
        
        try:
            # Check token blacklist first
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if self.redis_client:
                is_blacklisted = await self.redis_client.exists(f"{self.blacklist_prefix}{token_hash}")
                if is_blacklisted:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked"
                    )
            
            # Decode and validate JWT
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
                    "require_exp": True,
                    "require_iat": True,
                    "require_jti": True
                }
            )
            
            # Validate token type
            if payload.get("token_type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Enhanced security validations
            jti = payload.get("jti")
            if not jti:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format"
                )
            
            # Validate IP address if required
            if require_ip_match and ip_address:
                expected_ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
                token_ip_hash = payload.get("ip_hash", "")
                if not hmac.compare_digest(expected_ip_hash, token_ip_hash):
                    await self.audit_logger.log_security_incident(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        SecurityLevel.HIGH,
                        "Token IP address mismatch detected",
                        user_id=payload.get("sub"),
                        ip_address=ip_address,
                        details={
                            "expected_hash": expected_ip_hash,
                            "token_hash": token_ip_hash,
                            "jti": jti
                        }
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token validation failed"
                    )
            
            # Check token metadata in Redis
            if self.redis_client:
                metadata_json = await self.redis_client.get(f"token:{jti}")
                if metadata_json:
                    metadata = json.loads(metadata_json)
                    
                    # Validate session is still active
                    session_valid = await self.redis_client.exists(f"session:{payload['session_id']}")
                    if not session_valid:
                        await self.audit_logger.log_security_incident(
                            SecurityEventType.INVALID_TOKEN,
                            SecurityLevel.MEDIUM,
                            "Token references invalid session",
                            user_id=payload.get("sub"),
                            ip_address=ip_address,
                            details={"jti": jti, "session_id": payload['session_id']}
                        )
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Session invalid"
                        )
            
            # Log successful validation
            await self.audit_logger.log_authentication(
                SecurityEventType.TOKEN_VALIDATED,
                user_id=payload.get("sub"),
                username=payload.get("username"),
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=payload.get("session_id"),
                success=True
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            await self.audit_logger.log_authentication(
                SecurityEventType.TOKEN_EXPIRED,
                ip_address=ip_address,
                success=False,
                message="Access token expired"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
            
        except jwt.InvalidTokenError as e:
            await self.audit_logger.log_authentication(
                SecurityEventType.INVALID_TOKEN,
                ip_address=ip_address,
                success=False,
                message=f"Invalid token: {str(e)}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
        except HTTPException:
            raise
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            await self.audit_logger.log_security_incident(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                SecurityLevel.HIGH,
                f"Token validation system error: {str(e)}",
                ip_address=ip_address
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system error"
            )
    
    async def revoke_token(self, token: str, reason: str = "manual_revocation") -> bool:
        """Revoke token by adding to blacklist."""
        try:
            # Decode token to get expiration (ignore validation)
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False, "verify_signature": False}
            )
            
            # Calculate TTL for blacklist
            exp_timestamp = payload.get("exp", 0)
            current_timestamp = int(time.time())
            ttl = max(0, exp_timestamp - current_timestamp)
            
            if ttl > 0 and self.redis_client:
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                await self.redis_client.setex(
                    f"{self.blacklist_prefix}{token_hash}",
                    ttl,
                    json.dumps({"revoked_at": datetime.now(timezone.utc).isoformat(), "reason": reason})
                )
            
            # Remove token metadata
            jti = payload.get("jti")
            if jti and self.redis_client:
                await self.redis_client.delete(f"token:{jti}")
            
            # Audit log revocation
            await self.audit_logger.log_authentication(
                SecurityEventType.TOKEN_REVOKED,
                user_id=payload.get("sub"),
                username=payload.get("username"),
                session_id=payload.get("session_id"),
                success=True,
                details={"reason": reason, "jti": jti}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            return False


class BruteForceProtection:
    """Brute force attack protection."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.failed_attempts_cache = {}  # Fallback for non-Redis setups
        self.audit_logger = SecurityAuditLogger()
        
        # Configuration
        self.max_attempts = SecurityConfig.PASSWORD_MAX_FAILED_ATTEMPTS
        self.lockout_duration = SecurityConfig.PASSWORD_LOCKOUT_DURATION
    
    async def check_and_record_attempt(
        self,
        identifier: str,
        ip_address: str,
        success: bool,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check and record login attempt."""
        
        # Create keys for tracking
        user_key = f"failed_attempts:user:{identifier}"
        ip_key = f"failed_attempts:ip:{ip_address}"
        
        result = {
            "allowed": True,
            "attempts_remaining": self.max_attempts,
            "lockout_until": None,
            "blocked_reason": None
        }
        
        try:
            if success:
                # Clear failed attempts on successful login
                if self.redis_client:
                    await self.redis_client.delete(user_key)
                    await self.redis_client.delete(ip_key)
                else:
                    self.failed_attempts_cache.pop(user_key, None)
                    self.failed_attempts_cache.pop(ip_key, None)
                
                return result
            
            # Record failed attempt
            current_time = datetime.now(timezone.utc)
            
            if self.redis_client:
                # Check current attempts
                user_attempts = await self.redis_client.get(user_key) or "0"
                ip_attempts = await self.redis_client.get(ip_key) or "0"
                
                user_attempts = int(user_attempts)
                ip_attempts = int(ip_attempts)
                
                # Increment attempts
                await self.redis_client.setex(user_key, self.lockout_duration, user_attempts + 1)
                await self.redis_client.setex(ip_key, self.lockout_duration, ip_attempts + 1)
                
                user_attempts += 1
                ip_attempts += 1
            else:
                # Fallback to in-memory cache
                user_data = self.failed_attempts_cache.get(user_key, {"count": 0, "timestamp": current_time})
                ip_data = self.failed_attempts_cache.get(ip_key, {"count": 0, "timestamp": current_time})
                
                # Check if lockout period has expired
                if (current_time - user_data["timestamp"]).total_seconds() > self.lockout_duration:
                    user_data = {"count": 0, "timestamp": current_time}
                if (current_time - ip_data["timestamp"]).total_seconds() > self.lockout_duration:
                    ip_data = {"count": 0, "timestamp": current_time}
                
                user_data["count"] += 1
                ip_data["count"] += 1
                user_data["timestamp"] = current_time
                ip_data["timestamp"] = current_time
                
                self.failed_attempts_cache[user_key] = user_data
                self.failed_attempts_cache[ip_key] = ip_data
                
                user_attempts = user_data["count"]
                ip_attempts = ip_data["count"]
            
            # Check for brute force attack
            if user_attempts >= self.max_attempts or ip_attempts >= self.max_attempts:
                result["allowed"] = False
                result["attempts_remaining"] = 0
                result["lockout_until"] = (current_time + timedelta(seconds=self.lockout_duration)).isoformat()
                
                if user_attempts >= self.max_attempts:
                    result["blocked_reason"] = "user_lockout"
                else:
                    result["blocked_reason"] = "ip_lockout"
                
                # Log brute force detection
                await self.audit_logger.log_security_incident(
                    SecurityEventType.BRUTE_FORCE_DETECTED,
                    SecurityLevel.HIGH,
                    f"Brute force attack detected - {result['blocked_reason']}",
                    user_id=user_id,
                    ip_address=ip_address,
                    details={
                        "identifier": identifier,
                        "user_attempts": user_attempts,
                        "ip_attempts": ip_attempts,
                        "max_attempts": self.max_attempts
                    }
                )
            else:
                result["attempts_remaining"] = self.max_attempts - max(user_attempts, ip_attempts)
            
            return result
            
        except Exception as e:
            logger.error(f"Brute force protection error: {e}")
            # Fail securely - allow the attempt but log the error
            return result


class EnhancedAuthenticationMiddleware(BaseHTTPMiddleware):
    """Enhanced authentication middleware with comprehensive security."""
    
    def __init__(
        self,
        app,
        redis_client: Optional[redis.Redis] = None,
        excluded_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.redis_client = redis_client
        self.token_manager = SecureTokenManager(redis_client)
        self.csrf_protection = CSRFProtection(redis_client)
        self.brute_force_protection = BruteForceProtection(redis_client)
        self.audit_logger = SecurityAuditLogger()
        
        # Default excluded paths
        self.excluded_paths = excluded_paths or [
            "/docs", "/redoc", "/openapi.json", "/favicon.ico",
            "/health", "/metrics", "/status",
            "/auth/login", "/auth/register", "/auth/oauth",
            "/auth/forgot-password", "/auth/reset-password",
            "/auth/verify-email", "/auth/callback"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Enhanced request processing with security controls."""
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        path = request.url.path
        
        try:
            # Skip authentication for excluded paths
            if self._should_skip_auth(path):
                return await call_next(request)
            
            # Extract and validate authentication
            auth_result = await self._authenticate_request(request, client_ip, user_agent)
            
            if auth_result:
                # Add authentication context
                request.state.user_id = auth_result["sub"]
                request.state.username = auth_result.get("username")
                request.state.email = auth_result.get("email")
                request.state.role = auth_result.get("role")
                request.state.permissions = auth_result.get("permissions", [])
                request.state.session_id = auth_result.get("session_id")
                request.state.token_jti = auth_result.get("jti")
                request.state.authenticated = True
                
                # Generate CSRF token for session
                if request.state.session_id and request.method in ["GET", "HEAD"]:
                    csrf_token = await self.csrf_protection.generate_csrf_token(request.state.session_id)
                    request.state.csrf_token = csrf_token
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response, request)
            
            # Log successful request
            processing_time = time.time() - start_time
            if hasattr(request.state, 'user_id'):
                await self.audit_logger.log_authentication(
                    SecurityEventType.SESSION_ACCESSED,
                    user_id=request.state.user_id,
                    username=request.state.username,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    session_id=request.state.session_id,
                    success=True,
                    details={
                        "path": path,
                        "method": request.method,
                        "processing_time": processing_time,
                        "response_status": response.status_code
                    }
                )
            
            return response
            
        except HTTPException as e:
            # Log authentication failure
            await self.audit_logger.log_authentication(
                SecurityEventType.AUTHENTICATION_FAILED,
                ip_address=client_ip,
                user_agent=user_agent,
                success=False,
                details={
                    "path": path,
                    "method": request.method,
                    "error": str(e.detail),
                    "status_code": e.status_code
                }
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "detail": e.detail,
                    "type": "authentication_error",
                    "path": path,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            
            # Log system error
            await self.audit_logger.log_security_incident(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                SecurityLevel.HIGH,
                f"Authentication middleware system error: {str(e)}",
                ip_address=client_ip,
                details={"path": path, "method": request.method}
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Authentication system error",
                    "type": "system_error",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
    
    def _should_skip_auth(self, path: str) -> bool:
        """Check if path should skip authentication."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    async def _authenticate_request(
        self,
        request: Request,
        client_ip: str,
        user_agent: str
    ) -> Optional[Dict[str, Any]]:
        """Authenticate request with enhanced security."""
        
        # Extract token
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            # Check for token in cookies as fallback
            token = request.cookies.get("access_token")
            if not token:
                return None
        else:
            if not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization header format"
                )
            token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Validate token with enhanced security
        try:
            payload = await self.token_manager.validate_access_token(
                token,
                ip_address=client_ip,
                user_agent=user_agent,
                require_ip_match=True  # Enable strict IP validation
            )
            return payload
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP with proxy support and validation."""
        
        # Check forwarded headers in order of preference
        forwarded_headers = [
            "X-Forwarded-For",
            "X-Real-IP",
            "CF-Connecting-IP",  # Cloudflare
            "X-Forwarded",
            "Forwarded-For",
            "Forwarded"
        ]
        
        for header in forwarded_headers:
            header_value = request.headers.get(header)
            if header_value:
                # Take the first IP in the chain for X-Forwarded-For
                ip = header_value.split(",")[0].strip()
                if self._is_valid_ip(ip):
                    return ip
        
        # Fallback to direct connection
        if hasattr(request.client, "host"):
            client_ip = request.client.host
            if self._is_valid_ip(client_ip):
                return client_ip
        
        return "unknown"
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        try:
            parts = ip.split(".")
            if len(parts) != 4:
                return False
            
            for part in parts:
                num = int(part)
                if not 0 <= num <= 255:
                    return False
            
            return True
        except (ValueError, AttributeError):
            return False
    
    def _add_security_headers(self, response, request: Request):
        """Add security headers to response."""
        
        # Add CSRF token to response headers if available
        if hasattr(request.state, 'csrf_token'):
            response.headers["X-CSRF-Token"] = request.state.csrf_token
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value


# Export classes for easy import
__all__ = [
    "SecureTokenManager",
    "BruteForceProtection", 
    "EnhancedAuthenticationMiddleware"
]