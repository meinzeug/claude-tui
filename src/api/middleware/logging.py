"""
Logging middleware for API request/response tracking.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import logging
import json
import uuid
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for comprehensive request/response tracking."""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.sensitive_headers = {
            "authorization", "cookie", "x-api-key", 
            "x-auth-token", "x-access-token"
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process logging for each request."""
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Log incoming request
        if self.log_requests:
            await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            # Log response
            if self.log_responses:
                await self._log_response(request, response, request_id, process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed - "
                f"ID: {request_id}, "
                f"Error: {str(e)}, "
                f"Time: {process_time:.3f}s"
            )
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details."""
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")
        
        # Filter sensitive headers
        headers = self._filter_sensitive_headers(dict(request.headers))
        
        # Get query parameters
        query_params = dict(request.query_params)
        
        # Log request information
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "headers": headers,
            "query_params": query_params,
            "timestamp": time.time()
        }
        
        logger.info(f"Incoming request: {json.dumps(request_info, default=str)}")
    
    async def _log_response(
        self, 
        request: Request, 
        response: Response, 
        request_id: str, 
        process_time: float
    ):
        """Log response details."""
        # Get response headers (filter sensitive ones)
        response_headers = self._filter_sensitive_headers(dict(response.headers))
        
        # Log response information
        response_info = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "headers": response_headers,
            "process_time": process_time,
            "timestamp": time.time()
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
            message = "Server error response"
        elif response.status_code >= 400:
            log_level = logging.WARNING
            message = "Client error response"
        else:
            log_level = logging.INFO
            message = "Successful response"
        
        logger.log(log_level, f"{message}: {json.dumps(response_info, default=str)}")
    
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
    
    def _filter_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter out sensitive headers from logging."""
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        return filtered