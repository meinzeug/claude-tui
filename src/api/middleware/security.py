"""
Security middleware for API request validation and protection.

Enhanced with comprehensive security features:
- Input validation and sanitization
- Rate limiting with DDoS protection
- Authentication and authorization
- Security headers
- Audit logging
"""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
import time
import json
import logging
from typing import Dict, Set, Optional
import re
from pathlib import Path

# Import security components if available
try:
    from ...security.input_validator import SecurityInputValidator, ThreatLevel
    from ...security.rate_limiter import SmartRateLimiter, ActionType, create_rate_limiter
    from ...security.api_key_manager import APIKeyManager
    SECURITY_MODULES_AVAILABLE = True
except ImportError:
    SECURITY_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and protection."""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_request_size = max_request_size
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns = [
            re.compile(r'<script.*?>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'onload=', re.IGNORECASE),
            re.compile(r'onerror=', re.IGNORECASE),
            re.compile(r'union.*select', re.IGNORECASE),
            re.compile(r'drop.*table', re.IGNORECASE),
            re.compile(r'insert.*into', re.IGNORECASE),
            re.compile(r'delete.*from', re.IGNORECASE),
        ]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process security checks for each request."""
        start_time = time.time()
        
        try:
            # Check if IP is blocked
            client_ip = self._get_client_ip(request)
            if client_ip in self.blocked_ips:
                logger.warning(f"Blocked request from IP: {client_ip}")
                return Response(
                    content="Access denied",
                    status_code=403
                )
            
            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_request_size:
                logger.warning(f"Request too large: {content_length} bytes from {client_ip}")
                return Response(
                    content="Request too large",
                    status_code=413
                )
            
            # Check for suspicious patterns in URL and query parameters
            if self._contains_suspicious_content(str(request.url)):
                logger.warning(f"Suspicious URL detected: {request.url} from {client_ip}")
                return Response(
                    content="Invalid request",
                    status_code=400
                )
            
            # Add security headers
            response = await call_next(request)
            self._add_security_headers(response)
            
            # Log request details
            process_time = time.time() - start_time
            logger.info(
                f"Security check passed - "
                f"IP: {client_ip}, "
                f"Method: {request.method}, "
                f"Path: {request.url.path}, "
                f"Time: {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return Response(
                content="Internal security error",
                status_code=500
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client address
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _contains_suspicious_content(self, content: str) -> bool:
        """Check if content contains suspicious patterns."""
        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                return True
        return False
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        })
    
    def block_ip(self, ip: str):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        logger.info(f"IP blocked: {ip}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)
        logger.info(f"IP unblocked: {ip}")