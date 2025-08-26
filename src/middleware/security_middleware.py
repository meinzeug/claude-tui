"""
Security Middleware for FastAPI
Implements CORS, security headers, and other security-related middleware
"""
from typing import Dict, List, Optional, Callable, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import time
import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio
import ipaddress

from ..core.config import config

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.security_headers = config.get_security_headers()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        if self.enabled and self.security_headers:
            # Add security headers
            for header_name, header_value in self.security_headers.items():
                response.headers[header_name] = header_value
            
            # Add additional security headers based on content type
            content_type = response.headers.get("content-type", "")
            
            if "application/json" in content_type:
                # JSON-specific security headers
                response.headers["X-Content-Type-Options"] = "nosniff"
            
            # Add request ID for tracking
            request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            response.headers["X-Request-ID"] = request_id
            
            # Add timing header (for development/debugging)
            if hasattr(request.state, "start_time"):
                processing_time = time.time() - request.state.start_time
                response.headers["X-Process-Time"] = f"{processing_time:.4f}"
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with different strategies"""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        window_size: int = 60,
        enabled: bool = True,
        whitelist: List[str] = None,
        blacklist: List[str] = None
    ):
        super().__init__(app)
        self.enabled = enabled
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.window_size = window_size
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage = defaultdict(deque)
        self.burst_storage = defaultdict(int)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support"""
        # Check for forwarded headers (when behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (client IP)
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_ip_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        if not self.whitelist:
            return False
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            for whitelist_entry in self.whitelist:
                if "/" in whitelist_entry:
                    # CIDR notation
                    network = ipaddress.ip_network(whitelist_entry, strict=False)
                    if ip_obj in network:
                        return True
                else:
                    # Single IP
                    if str(ip_obj) == whitelist_entry:
                        return True
        except ValueError:
            logger.warning(f"Invalid IP address for whitelist check: {ip}")
        
        return False
    
    def _is_ip_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted"""
        if not self.blacklist:
            return False
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            for blacklist_entry in self.blacklist:
                if "/" in blacklist_entry:
                    # CIDR notation
                    network = ipaddress.ip_network(blacklist_entry, strict=False)
                    if ip_obj in network:
                        return True
                else:
                    # Single IP
                    if str(ip_obj) == blacklist_entry:
                        return True
        except ValueError:
            logger.warning(f"Invalid IP address for blacklist check: {ip}")
        
        return False
    
    def _cleanup_old_entries(self):
        """Remove old entries from rate limit storage"""
        current_time = time.time()
        
        # Only cleanup every cleanup_interval seconds
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.window_size
        
        for client_id in list(self.rate_limit_storage.keys()):
            deque_data = self.rate_limit_storage[client_id]
            
            # Remove old timestamps
            while deque_data and deque_data[0] < cutoff_time:
                deque_data.popleft()
            
            # Remove empty entries
            if not deque_data:
                del self.rate_limit_storage[client_id]
        
        self.last_cleanup = current_time
    
    def _check_rate_limit(self, client_ip: str) -> tuple[bool, Dict[str, Any]]:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Get or create deque for this client
        client_requests = self.rate_limit_storage[client_ip]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        
        # Count requests in current window
        requests_in_window = len(client_requests)
        
        # Check burst limit (immediate requests)
        burst_count = self.burst_storage[client_ip]
        
        # Reset burst count if enough time has passed
        if not client_requests or current_time - client_requests[-1] > 1:
            burst_count = 0
            self.burst_storage[client_ip] = 0
        
        # Check limits
        rate_limit_exceeded = requests_in_window >= self.requests_per_minute
        burst_limit_exceeded = burst_count >= self.burst_size
        
        # Calculate retry after time
        if rate_limit_exceeded:
            oldest_request = client_requests[0] if client_requests else current_time
            retry_after = int(oldest_request + self.window_size - current_time) + 1
        elif burst_limit_exceeded:
            retry_after = 1  # Short burst cooldown
        else:
            retry_after = 0
        
        # Rate limit info
        rate_limit_info = {
            "requests_in_window": requests_in_window,
            "requests_per_minute": self.requests_per_minute,
            "window_size": self.window_size,
            "retry_after": retry_after,
            "burst_count": burst_count,
            "burst_size": self.burst_size
        }
        
        is_limited = rate_limit_exceeded or burst_limit_exceeded
        
        # If not limited, record this request
        if not is_limited:
            client_requests.append(current_time)
            self.burst_storage[client_ip] += 1
        
        return is_limited, rate_limit_info
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        # Check blacklist first
        if self._is_ip_blacklisted(client_ip):
            logger.warning(f"Blacklisted IP blocked: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "status": "error",
                    "message": "Access denied",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Skip rate limiting for whitelisted IPs
        if self._is_ip_whitelisted(client_ip):
            return await call_next(request)
        
        # Cleanup old entries periodically
        self._cleanup_old_entries()
        
        # Check rate limit
        is_limited, rate_info = self._check_rate_limit(client_ip)
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            
            # Return rate limit error
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "message": "Rate limit exceeded",
                    "retry_after": rate_info["retry_after"],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.requests_per_minute - rate_info["requests_in_window"])
            )
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time() + rate_info["retry_after"])
            )
            response.headers["Retry-After"] = str(rate_info["retry_after"])
            
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - rate_info["requests_in_window"])
        )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and monitoring"""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Record start time
        start_time = time.time()
        request.state.start_time = start_time
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        if self.log_requests:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "unknown")
            
            logger.info(
                f"REQUEST {request_id} - {request.method} {request.url.path} "
                f"from {client_ip} - {user_agent}"
            )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            processing_time = time.time() - start_time
            
            if self.log_responses:
                logger.info(
                    f"RESPONSE {request_id} - {response.status_code} "
                    f"({processing_time:.4f}s)"
                )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"ERROR {request_id} - {str(e)} ({processing_time:.4f}s)"
            )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"


class ContentSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for content security and input validation"""
    
    def __init__(
        self,
        app,
        max_content_length: int = 10 * 1024 * 1024,  # 10MB
        allowed_content_types: List[str] = None
    ):
        super().__init__(app)
        self.max_content_length = max_content_length
        self.allowed_content_types = allowed_content_types or [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_content_length:
                    logger.warning(f"Content length too large: {length} bytes")
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "status": "error",
                            "message": "Request entity too large",
                            "max_size": self.max_content_length
                        }
                    )
            except ValueError:
                logger.warning(f"Invalid content-length header: {content_length}")
        
        # Check content type for requests with body
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").split(";")[0]
            
            if content_type and content_type not in self.allowed_content_types:
                logger.warning(f"Unsupported content type: {content_type}")
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={
                        "status": "error",
                        "message": "Unsupported media type",
                        "allowed_types": self.allowed_content_types
                    }
                )
        
        return await call_next(request)


def setup_cors_middleware(app, config_manager=None):
    """Setup CORS middleware with configuration"""
    if config_manager is None:
        config_manager = config
    
    cors_config = config_manager.get_cors_config()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"],
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ]
    )
    
    logger.info("CORS middleware configured")


def setup_security_middleware(app):
    """Setup all security middleware"""
    
    # Add security headers middleware
    app.add_middleware(
        SecurityHeadersMiddleware,
        enabled=config.security.security_headers_enabled
    )
    
    # Add rate limiting middleware
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=config.security.rate_limit_requests_per_minute,
        burst_size=config.security.rate_limit_burst_size,
        enabled=True
    )
    
    # Add request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware,
        log_requests=True,
        log_responses=config.app.debug
    )
    
    # Add content security middleware
    app.add_middleware(
        ContentSecurityMiddleware,
        max_content_length=10 * 1024 * 1024,  # 10MB
        allowed_content_types=[
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data"
        ]
    )
    
    logger.info("Security middleware setup completed")