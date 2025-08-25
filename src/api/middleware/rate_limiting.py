"""
Rate limiting middleware for API protection.
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import logging
from typing import Dict, List, Tuple
from collections import defaultdict, deque
from functools import wraps

logger = logging.getLogger(__name__)

# In-memory rate limiting storage for decorator (use Redis in production)
decorator_rate_limit_storage: Dict[str, deque] = defaultdict(deque)

def rate_limit(requests: int = 10, window: int = 60):
    """
    Rate limiting decorator for individual endpoints.
    
    Args:
        requests: Number of allowed requests
        window: Time window in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get client IP (simplified - in production use proper client identification)
            client_id = "default_client"  # This should be extracted from request context
            current_time = time.time()
            
            # Clean old requests outside the window
            window_start = current_time - window
            client_requests = decorator_rate_limit_storage[client_id]
            
            while client_requests and client_requests[0] < window_start:
                client_requests.popleft()
            
            # Check if rate limit exceeded
            if len(client_requests) >= requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Max {requests} requests per {window} seconds."
                )
            
            # Add current request
            client_requests.append(current_time)
            
            # Execute the function
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with sliding window algorithm."""
    
    def __init__(
        self, 
        app,
        requests_per_minute: int = 60,
        burst_requests: int = 10,
        window_seconds: int = 60
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_requests = burst_requests
        self.window_seconds = window_seconds
        
        # Client request tracking
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.burst_tracking: Dict[str, List[float]] = defaultdict(list)
        
        # Cleanup interval
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process rate limiting for each request."""
        try:
            client_ip = self._get_client_ip(request)
            current_time = time.time()
            
            # Periodic cleanup of old entries
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_entries(current_time)
                self.last_cleanup = current_time
            
            # Check rate limits
            if not self._is_allowed(client_ip, current_time):
                # Rate limit exceeded
                remaining_time = self._get_reset_time(client_ip, current_time)
                
                logger.warning(
                    f"Rate limit exceeded for IP: {client_ip}, "
                    f"Path: {request.url.path}, "
                    f"Reset in: {remaining_time:.0f}s"
                )
                
                # Return rate limit response
                response = Response(
                    content="Rate limit exceeded. Please try again later.",
                    status_code=429
                )
                response.headers.update({
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + remaining_time)),
                    "Retry-After": str(int(remaining_time))
                })
                return response
            
            # Record the request
            self._record_request(client_ip, current_time)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            remaining_requests = self._get_remaining_requests(client_ip, current_time)
            reset_time = self._get_reset_time(client_ip, current_time)
            
            response.headers.update({
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": str(remaining_requests),
                "X-RateLimit-Reset": str(int(current_time + reset_time))
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            return await call_next(request)
    
    def _is_allowed(self, client_ip: str, current_time: float) -> bool:
        """Check if request is allowed based on rate limits."""
        # Check burst limit (requests in last 10 seconds)
        burst_window = current_time - 10
        recent_requests = [
            req_time for req_time in self.burst_tracking[client_ip]
            if req_time > burst_window
        ]
        
        if len(recent_requests) >= self.burst_requests:
            return False
        
        # Check sliding window limit
        window_start = current_time - self.window_seconds
        client_requests = self.client_requests[client_ip]
        
        # Remove old requests from window
        while client_requests and client_requests[0] <= window_start:
            client_requests.popleft()
        
        # Check if under limit
        return len(client_requests) < self.requests_per_minute
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request for rate limiting tracking."""
        # Record for sliding window
        self.client_requests[client_ip].append(current_time)
        
        # Record for burst tracking
        self.burst_tracking[client_ip].append(current_time)
        
        # Keep only recent burst requests
        burst_window = current_time - 10
        self.burst_tracking[client_ip] = [
            req_time for req_time in self.burst_tracking[client_ip]
            if req_time > burst_window
        ]
    
    def _get_remaining_requests(self, client_ip: str, current_time: float) -> int:
        """Get remaining requests in current window."""
        window_start = current_time - self.window_seconds
        client_requests = self.client_requests[client_ip]
        
        # Count requests in current window
        current_requests = sum(1 for req_time in client_requests if req_time > window_start)
        
        return max(0, self.requests_per_minute - current_requests)
    
    def _get_reset_time(self, client_ip: str, current_time: float) -> float:
        """Get time until rate limit resets."""
        client_requests = self.client_requests[client_ip]
        
        if not client_requests:
            return 0
        
        # Time until oldest request in window expires
        oldest_request = client_requests[0] if client_requests else current_time
        reset_time = oldest_request + self.window_seconds - current_time
        
        return max(0, reset_time)
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old request entries to prevent memory leaks."""
        cutoff_time = current_time - self.window_seconds * 2
        
        # Clean up sliding window tracking
        for client_ip in list(self.client_requests.keys()):
            client_requests = self.client_requests[client_ip]
            while client_requests and client_requests[0] <= cutoff_time:
                client_requests.popleft()
            
            # Remove empty entries
            if not client_requests:
                del self.client_requests[client_ip]
        
        # Clean up burst tracking
        burst_cutoff = current_time - 60  # Keep 1 minute of burst data
        for client_ip in list(self.burst_tracking.keys()):
            self.burst_tracking[client_ip] = [
                req_time for req_time in self.burst_tracking[client_ip]
                if req_time > burst_cutoff
            ]
            
            # Remove empty entries
            if not self.burst_tracking[client_ip]:
                del self.burst_tracking[client_ip]
        
        logger.debug(f"Cleaned up rate limiting entries. Current clients: {len(self.client_requests)}")
    
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