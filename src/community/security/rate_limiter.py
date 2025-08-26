"""
Rate Limiter - Advanced rate limiting for community platform API endpoints.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.cache_service import get_cache_service
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimiterConfig:
    """Rate limiter configuration."""
    
    # Default rate limits (requests per time window)
    DEFAULT_LIMITS = {
        # General API limits
        "default": (100, 3600),  # 100 requests per hour
        "authenticated": (500, 3600),  # 500 requests per hour for authenticated users
        
        # Content creation limits
        "template_upload": (5, 3600),  # 5 template uploads per hour
        "plugin_upload": (3, 3600),   # 3 plugin uploads per hour
        "review_creation": (10, 3600), # 10 reviews per hour
        
        # Search and browsing (more lenient)
        "search": (200, 3600),         # 200 searches per hour
        "download": (50, 3600),        # 50 downloads per hour
        
        # Moderation and reporting
        "report_content": (10, 86400), # 10 reports per day
        "appeal_submission": (3, 86400), # 3 appeals per day
        
        # High-frequency actions
        "rating_vote": (50, 3600),     # 50 helpfulness votes per hour
        "view_tracking": (1000, 3600), # 1000 views per hour
        
        # Security-sensitive actions
        "login_attempts": (5, 900),    # 5 login attempts per 15 minutes
        "password_reset": (3, 3600),   # 3 password reset requests per hour
    }
    
    # Burst limits (short time windows for preventing spam)
    BURST_LIMITS = {
        "search": (30, 60),           # 30 searches per minute
        "download": (10, 60),         # 10 downloads per minute
        "api_calls": (50, 60),        # 50 API calls per minute
        "upload": (2, 60),            # 2 uploads per minute
    }
    
    # IP-based limits (per IP address)
    IP_LIMITS = {
        "default": (1000, 3600),      # 1000 requests per hour per IP
        "registration": (5, 86400),    # 5 registrations per day per IP
        "contact_form": (3, 86400),    # 3 contact form submissions per day per IP
    }


class RateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.cache = get_cache_service()
        self.config = RateLimiterConfig()
    
    def get_client_identifier(self, request: Request, user_id: Optional[UUID] = None) -> str:
        """Get client identifier for rate limiting."""
        if user_id:
            return f"user:{user_id}"
        
        # Use IP address as fallback
        ip = self.get_client_ip(request)
        return f"ip:{ip}"
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (for load balancers/proxies)
        forwarded_ip = request.headers.get("X-Forwarded-For")
        if forwarded_ip:
            return forwarded_ip.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"
    
    def check_rate_limit(
        self, 
        identifier: str, 
        action: str,
        custom_limit: Optional[Tuple[int, int]] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if action is within rate limits.
        
        Returns:
            Tuple of (is_allowed, limit_info)
        """
        # Get limit configuration
        if custom_limit:
            limit, window = custom_limit
        else:
            limit, window = self.config.DEFAULT_LIMITS.get(action, (100, 3600))
        
        # Create cache key
        cache_key = f"rate_limit:{action}:{identifier}"
        
        # Check current count and increment
        current_count, is_exceeded = self.cache.increment_rate_limit(
            cache_key, window, limit
        )
        
        # Calculate remaining requests
        remaining = max(0, limit - current_count)
        
        # Calculate reset time
        reset_time = int(datetime.now().timestamp()) + window
        
        limit_info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "window": window
        }
        
        is_allowed = not is_exceeded
        return is_allowed, limit_info
    
    def is_rate_limited(
        self, 
        request: Request, 
        action: str,
        user_id: Optional[UUID] = None,
        custom_limit: Optional[Tuple[int, int]] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Comprehensive rate limit check.
        
        Returns:
            Tuple of (is_limited, combined_limit_info)
        """
        client_identifier = self.get_client_identifier(request, user_id)
        
        # Check main rate limit
        is_allowed, limit_info = self.check_rate_limit(
            client_identifier, action, custom_limit
        )
        
        # Any limit exceeded means request is limited
        is_limited = not is_allowed
        
        return is_limited, limit_info


# Rate limiting decorators
def rate_limit(action: str, custom_limit: Optional[Tuple[int, int]] = None):
    """Decorator for rate limiting specific endpoint functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            user_id = None
            
            # Try to find request object
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Try to find user_id in kwargs
            if "current_user" in kwargs:
                user_data = kwargs["current_user"]
                if isinstance(user_data, dict) and "id" in user_data:
                    user_id = UUID(user_data["id"])
            
            if request is None:
                # If no request object, skip rate limiting
                logger.warning("Rate limiting skipped - no request object found")
                return await func(*args, **kwargs)
            
            # Check rate limits
            rate_limiter = RateLimiter()
            is_limited, limit_info = rate_limiter.is_rate_limited(
                request, action, user_id, custom_limit
            )
            
            if is_limited:
                headers = {}
                if "limit" in limit_info:
                    headers["X-RateLimit-Limit"] = str(limit_info["limit"])
                if "remaining" in limit_info:
                    headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
                if "reset" in limit_info:
                    headers["X-RateLimit-Reset"] = str(limit_info["reset"])
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded for {action}",
                    headers=headers
                )
            
            # Execute function normally
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


# Specific rate limiting decorators
def rate_limit_upload(func):
    """Rate limit for upload operations."""
    return rate_limit("upload", (5, 3600))(func)


def rate_limit_search(func):
    """Rate limit for search operations."""
    return rate_limit("search", (200, 3600))(func)


def rate_limit_download(func):
    """Rate limit for download operations."""
    return rate_limit("download", (50, 3600))(func)


def rate_limit_review(func):
    """Rate limit for review creation."""
    return rate_limit("review_creation", (10, 3600))(func)


def rate_limit_report(func):
    """Rate limit for content reporting."""
    return rate_limit("report_content", (10, 86400))(func)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter