"""
Security Hardening Module
=========================

Implements critical security fixes and hardening measures for the Claude-TUI system.
This module addresses the critical vulnerabilities identified in the security audit.
"""

import secrets
import hashlib
import hmac
import os
import re
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from functools import wraps
import asyncio
import json

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Enhanced Security Configuration with hardened defaults."""
    
    # Cryptographic settings
    MIN_SECRET_KEY_LENGTH = 32
    JWT_SECRET_KEY_MIN_LENGTH = 64
    ENCRYPTION_KEY_LENGTH = 32
    
    # Password policy
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_NUMBERS = True
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    PASSWORD_MAX_FAILED_ATTEMPTS = 5
    PASSWORD_LOCKOUT_DURATION = 1800  # 30 minutes
    
    # Token security
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Shorter for security
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7     # Shorter for security
    JWT_ALGORITHM = "HS256"
    JWT_REQUIRE_AUDIENCE = True
    JWT_REQUIRE_ISSUER = True
    
    # Session security
    SESSION_TIMEOUT_MINUTES = 60
    MAX_CONCURRENT_SESSIONS = 5
    SESSION_REGENERATE_INTERVAL = 300  # 5 minutes
    
    # Rate limiting
    LOGIN_RATE_LIMIT = 5  # attempts per minute
    API_RATE_LIMIT = 1000  # requests per hour for authenticated users
    ANONYMOUS_RATE_LIMIT = 100  # requests per hour for anonymous users
    
    # CSRF protection
    CSRF_TOKEN_LENGTH = 32
    CSRF_TOKEN_EXPIRE_MINUTES = 60
    
    # Security headers
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        ),
        'X-Frame-Options': 'DENY',
        'X-Content-Type-Options': 'nosniff',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': (
            'geolocation=(), microphone=(), camera=(), '
            'payment=(), usb=(), magnetometer=(), gyroscope=()'
        )
    }


class SecureKeyManager:
    """Secure key management and encryption utilities."""
    
    def __init__(self):
        self.encryption_key = self._get_or_generate_key()
        self.fernet = Fernet(self.encryption_key)
    
    def _get_or_generate_key(self) -> bytes:
        """Get or generate encryption key."""
        key_env = os.getenv('ENCRYPTION_KEY')
        if key_env:
            try:
                return base64.urlsafe_b64decode(key_env.encode())
            except Exception:
                logger.warning("Invalid ENCRYPTION_KEY format, generating new key")
        
        # Generate new key
        key = Fernet.generate_key()
        logger.warning(
            f"Generated new encryption key. Set ENCRYPTION_KEY={key.decode()} "
            "in your environment variables for production"
        )
        return key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token."""
        return secrets.token_hex(SecurityConfig.CSRF_TOKEN_LENGTH)
    
    @staticmethod
    def validate_secret_strength(secret: str, min_length: int = 32) -> bool:
        """Validate secret key strength."""
        if len(secret) < min_length:
            return False
        
        # Check entropy
        unique_chars = len(set(secret))
        if unique_chars < min_length // 2:
            return False
        
        return True


class EnhancedPasswordValidator:
    """Enhanced password validation with configurable policies."""
    
    def __init__(self):
        self.min_length = SecurityConfig.PASSWORD_MIN_LENGTH
        self.require_uppercase = SecurityConfig.PASSWORD_REQUIRE_UPPERCASE
        self.require_lowercase = SecurityConfig.PASSWORD_REQUIRE_LOWERCASE
        self.require_numbers = SecurityConfig.PASSWORD_REQUIRE_NUMBERS
        self.require_special = SecurityConfig.PASSWORD_REQUIRE_SPECIAL
        self.special_chars = SecurityConfig.PASSWORD_SPECIAL_CHARS
    
    def validate_password(self, password: str) -> tuple[bool, List[str]]:
        """Validate password against policy."""
        errors = []
        
        # Length check
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        
        # Character class checks
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.require_special and not re.search(f'[{re.escape(self.special_chars)}]', password):
            errors.append(f"Password must contain at least one special character: {self.special_chars}")
        
        # Common password checks
        if self._is_common_password(password):
            errors.append("Password is too common, please choose a more secure password")
        
        # Pattern checks
        if self._has_repeated_patterns(password):
            errors.append("Password contains repeated patterns")
        
        return len(errors) == 0, errors
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is commonly used."""
        common_passwords = {
            'password', '123456', 'password123', 'admin', 'letmein',
            'welcome', 'monkey', '1234567890', 'qwerty', 'abc123'
        }
        return password.lower() in common_passwords
    
    def _has_repeated_patterns(self, password: str) -> bool:
        """Check for repeated patterns in password."""
        # Check for repeated characters (3+ in a row)
        if re.search(r'(.)\1{2,}', password):
            return True
        
        # Check for sequential patterns
        sequences = ['123456', 'abcdef', 'qwerty', '654321', 'fedcba']
        password_lower = password.lower()
        for seq in sequences:
            if seq in password_lower or seq[::-1] in password_lower:
                return True
        
        return False


class CSRFProtection:
    """CSRF protection implementation."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.token_cache = {}  # Fallback for non-Redis setups
    
    async def generate_csrf_token(self, session_id: str) -> str:
        """Generate and store CSRF token for session."""
        token = SecureKeyManager.generate_csrf_token()
        
        if self.redis_client:
            await self.redis_client.setex(
                f"csrf:{session_id}",
                SecurityConfig.CSRF_TOKEN_EXPIRE_MINUTES * 60,
                token
            )
        else:
            # Fallback to in-memory cache
            self.token_cache[session_id] = {
                'token': token,
                'expires': datetime.now() + timedelta(minutes=SecurityConfig.CSRF_TOKEN_EXPIRE_MINUTES)
            }
        
        return token
    
    async def validate_csrf_token(self, session_id: str, provided_token: str) -> bool:
        """Validate CSRF token."""
        if not provided_token:
            return False
        
        stored_token = None
        
        if self.redis_client:
            stored_token = await self.redis_client.get(f"csrf:{session_id}")
            if stored_token:
                stored_token = stored_token.decode()
        else:
            # Fallback to in-memory cache
            cached = self.token_cache.get(session_id)
            if cached and cached['expires'] > datetime.now():
                stored_token = cached['token']
            elif cached:
                # Expired token
                del self.token_cache[session_id]
        
        if not stored_token:
            return False
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(stored_token, provided_token)


class SQLInjectionPrevention:
    """SQL injection prevention utilities."""
    
    # Dangerous SQL patterns
    DANGEROUS_PATTERNS = [
        r';\s*(drop|delete|truncate|insert|update)\s+',
        r'union\s+select',
        r'@@version',
        r'information_schema',
        r'sys\.',
        r'xp_cmdshell',
        r'sp_executesql',
        r'--\s*$',
        r'/\*.*\*/',
        r'char\s*\(',
        r'ascii\s*\(',
        r'waitfor\s+delay'
    ]
    
    @classmethod
    def validate_input(cls, input_value: str) -> bool:
        """Validate input for SQL injection patterns."""
        if not isinstance(input_value, str):
            return True
        
        input_lower = input_value.lower()
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                logger.warning(f"Potential SQL injection attempt blocked: {pattern}")
                return False
        
        return True
    
    @classmethod
    def sanitize_input(cls, input_value: str) -> str:
        """Sanitize input by removing dangerous characters."""
        if not isinstance(input_value, str):
            return str(input_value)
        
        # Remove null bytes
        sanitized = input_value.replace('\x00', '')
        
        # Remove SQL comments
        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        
        return sanitized


class XSSProtection:
    """XSS protection utilities."""
    
    # HTML entities mapping
    HTML_ENTITIES = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;'
    }
    
    @classmethod
    def escape_html(cls, text: str) -> str:
        """Escape HTML entities to prevent XSS."""
        if not isinstance(text, str):
            text = str(text)
        
        for char, entity in cls.HTML_ENTITIES.items():
            text = text.replace(char, entity)
        
        return text
    
    @classmethod
    def validate_javascript_context(cls, value: str) -> bool:
        """Validate value for JavaScript context."""
        dangerous_patterns = [
            r'javascript:',
            r'on\w+\s*=',
            r'<script',
            r'</script>',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\('
        ]
        
        value_lower = value.lower()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value_lower):
                return False
        
        return True


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting middleware."""
    
    def __init__(self, app: ASGIApp, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.rate_limit_cache = {}  # Fallback for non-Redis setups
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on endpoint and user context."""
        
        # Get rate limit key and limit
        rate_key, rate_limit = self._get_rate_limit_config(request)
        
        # Check rate limit
        if not await self._check_rate_limit(rate_key, rate_limit):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        response = await call_next(request)
        return response
    
    def _get_rate_limit_config(self, request: Request) -> tuple[str, int]:
        """Get rate limiting configuration for request."""
        path = request.url.path
        
        # Get user identifier
        if hasattr(request.state, 'user') and request.state.user:
            user_id = str(request.state.user.id)
            if getattr(request.state.user, 'is_admin', False):
                rate_limit = SecurityConfig.API_RATE_LIMIT * 5  # Higher limit for admins
            else:
                rate_limit = SecurityConfig.API_RATE_LIMIT
            rate_key = f"user:{user_id}"
        else:
            # Anonymous user - use IP
            client_ip = self._get_client_ip(request)
            rate_limit = SecurityConfig.ANONYMOUS_RATE_LIMIT
            rate_key = f"ip:{client_ip}"
        
        # Special handling for login endpoints
        if '/auth/login' in path:
            rate_limit = SecurityConfig.LOGIN_RATE_LIMIT
            rate_key = f"login:{self._get_client_ip(request)}"
        
        return rate_key, rate_limit
    
    async def _check_rate_limit(self, key: str, limit: int) -> bool:
        """Check if request is within rate limit."""
        current_time = datetime.now()
        window_key = f"rate_limit:{key}:{current_time.strftime('%Y-%m-%d-%H-%M')}"
        
        if self.redis_client:
            try:
                current_count = await self.redis_client.incr(window_key)
                if current_count == 1:
                    await self.redis_client.expire(window_key, 60)  # 1 minute window
                return current_count <= limit
            except Exception as e:
                logger.error(f"Redis rate limiting error: {e}")
                return True  # Fail open
        else:
            # Fallback to in-memory cache
            if window_key not in self.rate_limit_cache:
                self.rate_limit_cache[window_key] = 0
            
            self.rate_limit_cache[window_key] += 1
            
            # Cleanup old entries
            self._cleanup_rate_limit_cache()
            
            return self.rate_limit_cache[window_key] <= limit
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return getattr(request.client, "host", "unknown")
    
    def _cleanup_rate_limit_cache(self):
        """Cleanup expired rate limit entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key in self.rate_limit_cache:
            try:
                # Extract timestamp from key
                timestamp_str = key.split(":")[-1]
                key_time = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M')
                if (current_time - key_time).total_seconds() > 60:
                    expired_keys.append(key)
            except ValueError:
                continue
        
        for key in expired_keys:
            del self.rate_limit_cache[key]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Add CSRF token to response if session exists
        if hasattr(request.state, 'session') and request.state.session:
            # This would be handled by the CSRF protection class
            pass
        
        return response


class SecureExecutionEnvironment:
    """Secure code execution environment replacing eval/exec usage."""
    
    ALLOWED_BUILTINS = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'enumerate': enumerate,
            'zip': zip,
            'range': range,
        }
    }
    
    @classmethod
    def safe_evaluate(cls, expression: str, context: dict = None) -> Any:
        """Safely evaluate expressions using ast.literal_eval."""
        import ast
        
        context = context or {}
        
        try:
            # Try literal evaluation first (safest)
            return ast.literal_eval(expression)
        except (ValueError, SyntaxError):
            # If literal eval fails, reject the expression
            raise ValueError("Expression contains unsafe operations")
    
    @classmethod
    def validate_code_safety(cls, code: str) -> tuple[bool, List[str]]:
        """Validate code for safety before execution."""
        import ast
        
        errors = []
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]
        
        # Check for dangerous operations
        dangerous_nodes = {
            ast.Import: "Import statements not allowed",
            ast.ImportFrom: "Import statements not allowed",
            ast.Exec: "Exec statements not allowed",
            ast.Eval: "Eval statements not allowed",
            ast.Call: "Function calls need validation"
        }
        
        for node in ast.walk(tree):
            node_type = type(node)
            if node_type in dangerous_nodes:
                if node_type == ast.Call:
                    # Special handling for function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in cls.ALLOWED_BUILTINS['__builtins__']:
                            errors.append(f"Function '{node.func.id}' not allowed")
                else:
                    errors.append(dangerous_nodes[node_type])
        
        return len(errors) == 0, errors


# Utility functions for easy integration

def require_csrf_token(func):
    """Decorator to require CSRF token validation."""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        csrf_protection = request.app.state.csrf_protection
        session_id = getattr(request.state, 'session_id', None)
        csrf_token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
        
        if not session_id or not csrf_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token required"
            )
        
        if not await csrf_protection.validate_csrf_token(session_id, csrf_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token"
            )
        
        return await func(request, *args, **kwargs)
    
    return wrapper


def validate_input_safety(func):
    """Decorator to validate input for security issues."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Validate all string inputs
        for arg in args:
            if isinstance(arg, str):
                if not SQLInjectionPrevention.validate_input(arg):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid input detected"
                    )
        
        for key, value in kwargs.items():
            if isinstance(value, str):
                if not SQLInjectionPrevention.validate_input(value):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid input in parameter '{key}'"
                    )
        
        return await func(*args, **kwargs)
    
    return wrapper


# Security utilities for immediate use
class SecurityUtils:
    """Utility class for common security operations."""
    
    key_manager = SecureKeyManager()
    password_validator = EnhancedPasswordValidator()
    
    @classmethod
    def generate_secure_secret(cls, length: int = 32) -> str:
        """Generate a cryptographically secure secret."""
        return cls.key_manager.generate_secure_token(length)
    
    @classmethod
    def validate_password_strength(cls, password: str) -> tuple[bool, List[str]]:
        """Validate password strength."""
        return cls.password_validator.validate_password(password)
    
    @classmethod
    def sanitize_user_input(cls, input_data: str) -> str:
        """Sanitize user input for safety."""
        # SQL injection prevention
        sanitized = SQLInjectionPrevention.sanitize_input(input_data)
        # XSS prevention
        sanitized = XSSProtection.escape_html(sanitized)
        return sanitized
    
    @classmethod
    def validate_input_safety(cls, input_data: str) -> bool:
        """Validate input for security issues."""
        return (SQLInjectionPrevention.validate_input(input_data) and 
                XSSProtection.validate_javascript_context(input_data))


# Configuration validation
def validate_production_security():
    """Validate security configuration for production deployment."""
    issues = []
    warnings = []
    
    # Check JWT secret
    jwt_secret = os.getenv('JWT_SECRET_KEY')
    if not jwt_secret or len(jwt_secret) < SecurityConfig.JWT_SECRET_KEY_MIN_LENGTH:
        issues.append("JWT_SECRET_KEY must be at least 64 characters long")
    
    # Check general secret key
    secret_key = os.getenv('SECRET_KEY')
    if not secret_key or secret_key == "your-super-secret-key-change-in-production":
        issues.append("SECRET_KEY must be changed from default value")
    
    # Check encryption key
    encryption_key = os.getenv('ENCRYPTION_KEY')
    if not encryption_key:
        warnings.append("ENCRYPTION_KEY not set - will generate temporary key")
    
    # Check database URL security
    db_url = os.getenv('DATABASE_URL', '')
    if 'localhost' in db_url and os.getenv('ENVIRONMENT') == 'production':
        warnings.append("Using localhost database in production")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


if __name__ == "__main__":
    # Run security validation
    result = validate_production_security()
    print("Security Configuration Validation:")
    print(f"Valid: {result['valid']}")
    
    if result['issues']:
        print("Issues:")
        for issue in result['issues']:
            print(f"  ❌ {issue}")
    
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  ⚠️  {warning}")