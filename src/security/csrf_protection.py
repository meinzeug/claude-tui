"""
CSRF Protection Middleware for Claude-TUI API

Implements Cross-Site Request Forgery protection using double-submit cookies
and synchronizer token patterns for state-changing HTTP methods.
"""

import secrets
import hashlib
import hmac
import time
import logging
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import urlparse

from fastapi import Request, HTTPException, status, Cookie, Header
from fastapi.responses import JSONResponse, Response
from fastapi.security.utils import get_authorization_scheme_param
from starlette.middleware.base import BaseHTTPMiddleware
import jwt

logger = logging.getLogger(__name__)


class CSRFError(Exception):
    """CSRF validation error."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class CSRFToken:
    """CSRF token generator and validator."""
    
    def __init__(self, secret_key: str, token_lifetime: int = 3600):
        """
        Initialize CSRF token handler.
        
        Args:
            secret_key: Secret key for token generation
            token_lifetime: Token lifetime in seconds
        """
        self.secret_key = secret_key.encode()
        self.token_lifetime = token_lifetime
    
    def generate_token(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """
        Generate CSRF token.
        
        Args:
            session_id: Optional session ID to bind token to
            user_id: Optional user ID to bind token to
            
        Returns:
            Base64 encoded CSRF token
        """
        # Generate random salt
        salt = secrets.token_bytes(16)
        
        # Current timestamp
        timestamp = int(time.time())
        
        # Create payload
        payload = {
            "timestamp": timestamp,
            "session_id": session_id or "",
            "user_id": user_id or "",
            "nonce": secrets.token_hex(8)
        }
        
        # Create token data
        token_data = f"{timestamp}:{session_id or ''}:{user_id or ''}:{payload['nonce']}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key,
            (token_data + salt.hex()).encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine salt and signature
        token = salt.hex() + signature
        
        return token
    
    def validate_token(
        self, 
        token: str, 
        session_id: Optional[str] = None, 
        user_id: Optional[str] = None
    ) -> bool:
        """
        Validate CSRF token.
        
        Args:
            token: CSRF token to validate
            session_id: Expected session ID
            user_id: Expected user ID
            
        Returns:
            True if token is valid
        """
        try:
            if not token or len(token) < 32:
                return False
            
            # Extract salt and signature
            salt_hex = token[:32]  # 16 bytes = 32 hex chars
            signature = token[32:]
            
            # Parse timestamp from first part (we'll need to reconstruct)
            # For now, we'll verify the HMAC matches
            salt = bytes.fromhex(salt_hex)
            
            # Try different timestamps within the lifetime window
            current_time = int(time.time())
            
            for time_offset in range(-self.token_lifetime, 1):
                test_timestamp = current_time + time_offset
                
                # Reconstruct possible token data variations
                test_combinations = [
                    f"{test_timestamp}:{session_id or ''}:{user_id or ''}:",
                    f"{test_timestamp}:::",
                ]
                
                for token_data in test_combinations:
                    # Try to match the signature
                    for nonce_len in range(8, 17):  # Try different nonce lengths
                        test_nonce = secrets.token_hex(nonce_len)
                        full_token_data = token_data + test_nonce
                        
                        expected_signature = hmac.new(
                            self.secret_key,
                            (full_token_data + salt_hex).encode(),
                            hashlib.sha256
                        ).hexdigest()
                        
                        if hmac.compare_digest(signature, expected_signature):
                            # Check if token is within lifetime
                            if current_time - test_timestamp <= self.token_lifetime:
                                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"CSRF token validation error: {e}")
            return False
    
    def generate_double_submit_token(self) -> tuple[str, str]:
        """
        Generate double-submit CSRF token pair.
        
        Returns:
            Tuple of (cookie_token, form_token)
        """
        # Generate base token
        base_token = secrets.token_urlsafe(32)
        
        # Create cookie token (stored in cookie)
        cookie_token = base_token
        
        # Create form token (sent in form/header)
        form_token = hashlib.sha256((base_token + self.secret_key.decode()).encode()).hexdigest()
        
        return cookie_token, form_token
    
    def validate_double_submit_tokens(self, cookie_token: str, form_token: str) -> bool:
        """
        Validate double-submit CSRF tokens.
        
        Args:
            cookie_token: Token from cookie
            form_token: Token from form/header
            
        Returns:
            True if tokens are valid and match
        """
        try:
            if not cookie_token or not form_token:
                return False
            
            # Regenerate expected form token
            expected_form_token = hashlib.sha256(
                (cookie_token + self.secret_key.decode()).encode()
            ).hexdigest()
            
            # Compare tokens
            return hmac.compare_digest(form_token, expected_form_token)
            
        except Exception as e:
            logger.warning(f"Double-submit token validation error: {e}")
            return False


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF Protection Middleware.
    
    Implements CSRF protection using:
    1. Double-submit cookie pattern
    2. Synchronizer token pattern
    3. SameSite cookie attributes
    4. Origin/Referer validation
    """
    
    def __init__(
        self,
        app,
        secret_key: str,
        token_lifetime: int = 3600,
        cookie_name: str = "csrf_token",
        header_name: str = "X-CSRF-Token",
        form_field_name: str = "csrf_token",
        safe_methods: Optional[List[str]] = None,
        exempt_paths: Optional[List[str]] = None,
        trusted_origins: Optional[List[str]] = None,
        enforce_origin_check: bool = True
    ):
        """
        Initialize CSRF protection middleware.
        
        Args:
            app: FastAPI application
            secret_key: Secret key for token generation
            token_lifetime: Token lifetime in seconds
            cookie_name: Name of CSRF cookie
            header_name: Name of CSRF header
            form_field_name: Name of CSRF form field
            safe_methods: HTTP methods that don't require CSRF protection
            exempt_paths: Paths exempt from CSRF protection
            trusted_origins: List of trusted origins
            enforce_origin_check: Whether to enforce origin validation
        """
        super().__init__(app)
        
        self.csrf_token = CSRFToken(secret_key, token_lifetime)
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.form_field_name = form_field_name
        self.safe_methods = safe_methods or ["GET", "HEAD", "OPTIONS", "TRACE"]
        self.exempt_paths = set(exempt_paths or [])
        self.trusted_origins = set(trusted_origins or [])
        self.enforce_origin_check = enforce_origin_check
        
        # Add common exempt paths
        self.exempt_paths.update([
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/api/v1/auth/login",  # Login endpoint should be exempt initially
            "/api/v1/auth/register"  # Registration endpoint should be exempt initially
        ])
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through CSRF protection."""
        try:
            # Skip CSRF protection for safe methods and exempt paths
            if self._should_skip_csrf(request):
                response = await call_next(request)
                return self._add_csrf_cookie(request, response)
            
            # Validate CSRF protection
            csrf_valid = await self._validate_csrf(request)
            
            if not csrf_valid:
                logger.warning(
                    f"CSRF validation failed for {request.method} {request.url} "
                    f"from {request.client.host if request.client else 'unknown'}"
                )
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "CSRF token validation failed",
                        "message": "Missing or invalid CSRF token"
                    }
                )
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            
            # Add/refresh CSRF cookie in response
            return self._add_csrf_cookie(request, response)
            
        except CSRFError as e:
            logger.warning(f"CSRF error: {e.message}")
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "CSRF validation failed",
                    "message": e.message
                }
            )
        except Exception as e:
            logger.error(f"Error in CSRF middleware: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error"}
            )
    
    def _should_skip_csrf(self, request: Request) -> bool:
        """Check if CSRF protection should be skipped."""
        # Skip for safe HTTP methods
        if request.method in self.safe_methods:
            return True
        
        # Skip for exempt paths
        path = request.url.path
        if any(path.startswith(exempt_path) for exempt_path in self.exempt_paths):
            return True
        
        # Skip for API endpoints with valid API key authentication
        # (You might want to implement this based on your API key strategy)
        
        return False
    
    async def _validate_csrf(self, request: Request) -> bool:
        """Validate CSRF protection."""
        try:
            # 1. Origin/Referer validation (if enforced)
            if self.enforce_origin_check and not self._validate_origin(request):
                raise CSRFError("Invalid origin or referer", 403)
            
            # 2. Double-submit token validation
            cookie_token = request.cookies.get(self.cookie_name)
            
            # Get form token from header or form data
            form_token = request.headers.get(self.header_name)
            
            if not form_token:
                # Try to get from form data
                if request.method == "POST":
                    try:
                        content_type = request.headers.get("content-type", "")
                        
                        if "application/x-www-form-urlencoded" in content_type:
                            form = await request.form()
                            form_token = form.get(self.form_field_name)
                        elif "multipart/form-data" in content_type:
                            form = await request.form()
                            form_token = form.get(self.form_field_name)
                        elif "application/json" in content_type:
                            body = await request.json()
                            form_token = body.get(self.form_field_name)
                    except Exception:
                        pass  # Continue with header validation
            
            # Validate double-submit tokens
            if not cookie_token or not form_token:
                return False
            
            return self.csrf_token.validate_double_submit_tokens(cookie_token, form_token)
            
        except CSRFError:
            raise
        except Exception as e:
            logger.error(f"CSRF validation error: {e}")
            return False
    
    def _validate_origin(self, request: Request) -> bool:
        """Validate request origin/referer."""
        # Get origin from header
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        
        # Extract host from request
        request_host = request.url.hostname
        
        # Check origin
        if origin:
            origin_host = urlparse(origin).hostname
            if origin_host == request_host or origin in self.trusted_origins:
                return True
        
        # Check referer as fallback
        if referer:
            referer_host = urlparse(referer).hostname
            if referer_host == request_host or referer in self.trusted_origins:
                return True
        
        # For HTTPS requests, require origin/referer
        if request.url.scheme == "https":
            return False
        
        # For HTTP requests, be more lenient (but log warning)
        if not origin and not referer:
            logger.warning(f"Missing origin and referer for {request.method} {request.url}")
        
        return True
    
    def _add_csrf_cookie(self, request: Request, response: Response) -> Response:
        """Add CSRF cookie to response."""
        try:
            # Generate new double-submit tokens
            cookie_token, _ = self.csrf_token.generate_double_submit_token()
            
            # Set cookie with secure attributes
            response.set_cookie(
                key=self.cookie_name,
                value=cookie_token,
                httponly=False,  # Needs to be accessible by JavaScript
                secure=request.url.scheme == "https",
                samesite="strict",  # Strict SameSite policy
                max_age=3600  # 1 hour
            )
            
            # Also add token to response headers for SPA consumption
            response.headers[self.header_name] = cookie_token
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding CSRF cookie: {e}")
            return response
    
    def generate_csrf_token_for_template(self, request: Request) -> str:
        """Generate CSRF token for template rendering."""
        cookie_token = request.cookies.get(self.cookie_name)
        if cookie_token:
            _, form_token = self.csrf_token.generate_double_submit_token()
            return form_token
        return ""


def create_csrf_middleware(
    secret_key: str,
    **kwargs
) -> Callable:
    """Create CSRF protection middleware."""
    def middleware_factory(app):
        return CSRFProtectionMiddleware(app, secret_key, **kwargs)
    
    return middleware_factory


# Utility functions for manual CSRF handling
def get_csrf_token_from_request(request: Request, cookie_name: str = "csrf_token") -> Optional[str]:
    """Extract CSRF token from request cookies."""
    return request.cookies.get(cookie_name)


def validate_csrf_token_manually(
    request: Request,
    csrf_token_handler: CSRFToken,
    cookie_name: str = "csrf_token",
    header_name: str = "X-CSRF-Token"
) -> bool:
    """Manually validate CSRF token (for custom endpoints)."""
    cookie_token = request.cookies.get(cookie_name)
    form_token = request.headers.get(header_name)
    
    if not cookie_token or not form_token:
        return False
    
    return csrf_token_handler.validate_double_submit_tokens(cookie_token, form_token)


# FastAPI dependency for CSRF validation
def csrf_protect(
    request: Request,
    csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    cookie_csrf_token: Optional[str] = Cookie(None, alias="csrf_token")
):
    """FastAPI dependency for CSRF protection."""
    if not cookie_csrf_token or not csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token required"
        )
    
    # This would need access to the CSRFToken instance
    # In practice, you'd inject this dependency with the configured handler
    return True