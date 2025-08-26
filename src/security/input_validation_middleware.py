"""
Comprehensive Input Validation Middleware for Claude-TUI API

Provides centralized input validation, sanitization, and security checks
for all API endpoints to prevent injection attacks and malicious input.
"""

import re
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from urllib.parse import unquote

from fastapi import Request, HTTPException, status, Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from pydantic import BaseModel, Field, validator
import bleach
from markupsafe import escape

logger = logging.getLogger(__name__)


class ValidationRule(BaseModel):
    """Input validation rule definition."""
    name: str
    pattern: Optional[str] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    allowed_values: Optional[List[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    sanitize: bool = True
    required: bool = False


class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error in {field}: {message}")


class InputSanitizer:
    """Advanced input sanitizer with context-aware cleaning."""
    
    # XSS Prevention Patterns
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>',
        r'<link[^>]*>',
        r'<meta[^>]*>',
        r'<style[^>]*>.*?</style>',
        r'data:text/html',
        r'vbscript:',
        r'expression\s*\(',
    ]
    
    # SQL Injection Patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
        r'(\b(or|and)\s+\d+\s*=\s*\d+)',
        r'([\'";])',
        r'(\-\-)',
        r'(/\*.*?\*/)',
        r'(\b(information_schema|sys|mysql|pg_|sqlite_)\b)',
    ]
    
    # Command Injection Patterns
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$]',
        r'\b(cat|ls|pwd|id|whoami|uname|ps|netstat|ifconfig|ping|curl|wget|nc|telnet|ssh)\b',
        r'(\.\./|\.\.\\)',
        r'(/etc/passwd|/etc/shadow|cmd\.exe|powershell\.exe)',
    ]
    
    # Path Traversal Patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',
        r'\.\.\\',
        r'%2e%2e%2f',
        r'%2e%2e%5c',
        r'\.%2e/',
        r'%252e%252e%252f',
    ]
    
    def __init__(self):
        # HTML sanitizer configuration
        self.html_allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'i', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'blockquote', 'code', 'pre'
        ]
        
        self.html_allowed_attributes = {
            '*': ['class', 'id'],
            'a': ['href', 'title', 'target'],
            'img': ['src', 'alt', 'title', 'width', 'height']
        }
    
    def sanitize_string(self, value: str, context: str = "general") -> str:
        """
        Sanitize string based on context.
        
        Args:
            value: Input string to sanitize
            context: Context for sanitization (general, html, sql, path, filename)
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Basic sanitization
        value = value.strip()
        
        if context == "html":
            return self._sanitize_html(value)
        elif context == "sql":
            return self._sanitize_sql(value)
        elif context == "path":
            return self._sanitize_path(value)
        elif context == "filename":
            return self._sanitize_filename(value)
        elif context == "command":
            return self._sanitize_command(value)
        else:
            return self._sanitize_general(value)
    
    def _sanitize_html(self, value: str) -> str:
        """Sanitize HTML content."""
        # Remove XSS patterns
        for pattern in self.XSS_PATTERNS:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE | re.DOTALL)
        
        # Use bleach for additional HTML sanitization
        return bleach.clean(
            value,
            tags=self.html_allowed_tags,
            attributes=self.html_allowed_attributes,
            strip=True
        )
    
    def _sanitize_sql(self, value: str) -> str:
        """Sanitize for SQL context."""
        # Escape single quotes
        value = value.replace("'", "''")
        
        # Check for injection patterns
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential SQL injection attempt: {value[:100]}")
                # Remove potentially dangerous content
                value = re.sub(pattern, '', value, flags=re.IGNORECASE)
        
        return value
    
    def _sanitize_path(self, value: str) -> str:
        """Sanitize file path."""
        # URL decode first
        value = unquote(value)
        
        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Path traversal attempt detected: {value}")
                return ""  # Block suspicious paths entirely
        
        # Normalize path
        try:
            path = Path(value).resolve()
            return str(path)
        except Exception:
            return ""
    
    def _sanitize_filename(self, value: str) -> str:
        """Sanitize filename."""
        # Remove path separators and dangerous characters
        value = re.sub(r'[<>:"/\\|?*]', '', value)
        value = re.sub(r'[\x00-\x1f\x7f]', '', value)  # Remove control characters
        
        # Prevent reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        if value.upper() in reserved_names:
            value = f"file_{value}"
        
        return value[:255]  # Limit length
    
    def _sanitize_command(self, value: str) -> str:
        """Sanitize command input."""
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Command injection attempt detected: {value}")
                return ""  # Block entirely
        
        return value
    
    def _sanitize_general(self, value: str) -> str:
        """General sanitization."""
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove excessive whitespace
        value = re.sub(r'\s+', ' ', value)
        
        # HTML escape as fallback
        return escape(value)
    
    def validate_file_upload(self, file: UploadFile) -> Dict[str, Any]:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_filename": ""
        }
        
        # Check filename
        if not file.filename:
            result["valid"] = False
            result["errors"].append("Filename is required")
            return result
        
        # Sanitize filename
        sanitized_filename = self._sanitize_filename(file.filename)
        if not sanitized_filename:
            result["valid"] = False
            result["errors"].append("Invalid filename")
            return result
        
        result["sanitized_filename"] = sanitized_filename
        
        # Check file size
        if hasattr(file, 'size') and file.size:
            max_size = 10 * 1024 * 1024  # 10MB
            if file.size > max_size:
                result["valid"] = False
                result["errors"].append(f"File size exceeds maximum ({max_size} bytes)")
        
        # Check content type
        allowed_types = [
            'text/plain', 'text/csv', 'application/json', 'application/xml',
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'application/pdf', 'application/zip'
        ]
        
        if file.content_type not in allowed_types:
            result["valid"] = False
            result["errors"].append(f"File type not allowed: {file.content_type}")
        
        # Check file extension
        allowed_extensions = {
            '.txt', '.csv', '.json', '.xml', '.jpg', '.jpeg', '.png', 
            '.gif', '.webp', '.pdf', '.zip'
        }
        
        file_ext = Path(sanitized_filename).suffix.lower()
        if file_ext not in allowed_extensions:
            result["warnings"].append(f"Unusual file extension: {file_ext}")
        
        return result


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive input validation middleware.
    
    Validates and sanitizes all incoming requests based on configurable rules.
    """
    
    def __init__(self, app, validation_rules: Optional[Dict[str, Dict[str, ValidationRule]]] = None):
        super().__init__(app)
        self.sanitizer = InputSanitizer()
        self.validation_rules = validation_rules or {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules for common endpoints."""
        default_rules = {
            "/api/v1/auth/login": {
                "username": ValidationRule(
                    name="username",
                    max_length=100,
                    min_length=3,
                    pattern=r'^[a-zA-Z0-9._@-]+$',
                    sanitize=True,
                    required=True
                ),
                "password": ValidationRule(
                    name="password",
                    max_length=128,
                    min_length=8,
                    sanitize=False,  # Don't sanitize passwords
                    required=True
                )
            },
            "/api/v1/projects": {
                "name": ValidationRule(
                    name="project_name",
                    max_length=100,
                    min_length=1,
                    pattern=r'^[a-zA-Z0-9\s._-]+$',
                    sanitize=True,
                    required=True
                ),
                "description": ValidationRule(
                    name="description",
                    max_length=1000,
                    sanitize=True,
                    required=False
                )
            },
            "/api/v1/files/upload": {
                "file": ValidationRule(
                    name="file",
                    sanitize=True,
                    required=True
                )
            }
        }
        
        # Merge with provided rules
        for path, rules in default_rules.items():
            if path not in self.validation_rules:
                self.validation_rules[path] = rules
            else:
                self.validation_rules[path].update(rules)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process incoming request through validation pipeline."""
        try:
            # Skip validation for certain endpoints
            if self._should_skip_validation(request):
                return await call_next(request)
            
            # Validate request
            validation_result = await self._validate_request(request)
            
            if not validation_result["valid"]:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Validation failed",
                        "details": validation_result["errors"]
                    }
                )
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            
            # Log security events if needed
            if validation_result.get("security_warnings"):
                logger.warning(
                    f"Security warnings for {request.url}: "
                    f"{validation_result['security_warnings']}"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in input validation middleware: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Validation error"}
            )
    
    def _should_skip_validation(self, request: Request) -> bool:
        """Check if validation should be skipped for this request."""
        skip_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/static"
        ]
        
        return any(request.url.path.startswith(path) for path in skip_paths)
    
    async def _validate_request(self, request: Request) -> Dict[str, Any]:
        """
        Validate incoming request.
        
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "security_warnings": []
        }
        
        try:
            # Get validation rules for this path
            path_rules = self._get_path_rules(request.url.path)
            
            # Validate query parameters
            if request.query_params:
                query_validation = self._validate_query_params(
                    request.query_params,
                    path_rules.get("query", {})
                )
                result["errors"].extend(query_validation["errors"])
                result["warnings"].extend(query_validation["warnings"])
            
            # Validate request body for POST/PUT/PATCH
            if request.method in ["POST", "PUT", "PATCH"]:
                body_validation = await self._validate_request_body(request, path_rules)
                result["errors"].extend(body_validation["errors"])
                result["warnings"].extend(body_validation["warnings"])
                result["security_warnings"].extend(body_validation.get("security_warnings", []))
            
            # Validate headers
            header_validation = self._validate_headers(request.headers)
            result["warnings"].extend(header_validation["warnings"])
            
            # Check for suspicious patterns
            suspicious_checks = await self._check_suspicious_patterns(request)
            result["security_warnings"].extend(suspicious_checks)
            
            result["valid"] = len(result["errors"]) == 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            result["valid"] = False
            result["errors"].append("Validation processing error")
            return result
    
    def _get_path_rules(self, path: str) -> Dict[str, Any]:
        """Get validation rules for specific path."""
        # Exact match first
        if path in self.validation_rules:
            return self.validation_rules[path]
        
        # Pattern matching for dynamic paths
        for rule_path, rules in self.validation_rules.items():
            if self._path_matches_pattern(path, rule_path):
                return rules
        
        return {}
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a pattern with wildcards."""
        # Simple pattern matching - can be enhanced
        pattern_regex = pattern.replace("*", "[^/]*").replace("**", ".*")
        return re.match(f"^{pattern_regex}$", path) is not None
    
    def _validate_query_params(self, params: Dict[str, str], rules: Dict[str, ValidationRule]) -> Dict[str, Any]:
        """Validate query parameters."""
        result = {"errors": [], "warnings": []}
        
        for param_name, param_value in params.items():
            if param_name in rules:
                rule = rules[param_name]
                try:
                    self._validate_single_field(param_name, param_value, rule)
                except ValidationError as e:
                    result["errors"].append(e.message)
            else:
                # Sanitize unknown parameters
                sanitized = self.sanitizer.sanitize_string(param_value)
                if sanitized != param_value:
                    result["warnings"].append(f"Query parameter '{param_name}' was sanitized")
        
        return result
    
    async def _validate_request_body(self, request: Request, path_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request body."""
        result = {"errors": [], "warnings": [], "security_warnings": []}
        
        try:
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                body = await request.json()
                json_validation = self._validate_json_body(body, path_rules)
                result.update(json_validation)
                
            elif "multipart/form-data" in content_type:
                form_validation = await self._validate_form_data(request, path_rules)
                result.update(form_validation)
                
            elif "application/x-www-form-urlencoded" in content_type:
                form = await request.form()
                form_validation = self._validate_form_fields(dict(form), path_rules)
                result.update(form_validation)
            
        except Exception as e:
            result["errors"].append(f"Error parsing request body: {str(e)}")
        
        return result
    
    def _validate_json_body(self, body: Dict[str, Any], path_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON request body."""
        result = {"errors": [], "warnings": [], "security_warnings": []}
        
        body_rules = path_rules.get("body", {})
        
        for field_name, field_value in body.items():
            if field_name in body_rules:
                rule = body_rules[field_name]
                try:
                    self._validate_single_field(field_name, field_value, rule)
                except ValidationError as e:
                    result["errors"].append(e.message)
            
            # Check for suspicious content
            if isinstance(field_value, str):
                if self._contains_suspicious_patterns(field_value):
                    result["security_warnings"].append(
                        f"Suspicious content detected in field '{field_name}'"
                    )
        
        return result
    
    async def _validate_form_data(self, request: Request, path_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate multipart form data."""
        result = {"errors": [], "warnings": [], "security_warnings": []}
        
        try:
            form = await request.form()
            
            for field_name, field_value in form.items():
                if isinstance(field_value, UploadFile):
                    # Validate file upload
                    file_validation = self.sanitizer.validate_file_upload(field_value)
                    if not file_validation["valid"]:
                        result["errors"].extend(file_validation["errors"])
                    result["warnings"].extend(file_validation["warnings"])
                else:
                    # Validate form field
                    field_rules = path_rules.get("form", {})
                    if field_name in field_rules:
                        rule = field_rules[field_name]
                        try:
                            self._validate_single_field(field_name, field_value, rule)
                        except ValidationError as e:
                            result["errors"].append(e.message)
        
        except Exception as e:
            result["errors"].append(f"Error validating form data: {str(e)}")
        
        return result
    
    def _validate_form_fields(self, form_data: Dict[str, str], path_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate URL-encoded form fields."""
        result = {"errors": [], "warnings": [], "security_warnings": []}
        
        form_rules = path_rules.get("form", {})
        
        for field_name, field_value in form_data.items():
            if field_name in form_rules:
                rule = form_rules[field_name]
                try:
                    self._validate_single_field(field_name, field_value, rule)
                except ValidationError as e:
                    result["errors"].append(e.message)
        
        return result
    
    def _validate_single_field(self, field_name: str, value: Any, rule: ValidationRule) -> None:
        """Validate a single field against its rule."""
        # Check if required field is present
        if rule.required and (value is None or value == ""):
            raise ValidationError(field_name, "Field is required", value)
        
        # Skip validation for None values on optional fields
        if value is None:
            return
        
        # Convert to string for validation
        str_value = str(value)
        
        # Length validation
        if rule.min_length and len(str_value) < rule.min_length:
            raise ValidationError(field_name, f"Minimum length is {rule.min_length}", value)
        
        if rule.max_length and len(str_value) > rule.max_length:
            raise ValidationError(field_name, f"Maximum length is {rule.max_length}", value)
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, str_value):
            raise ValidationError(field_name, "Invalid format", value)
        
        # Allowed values validation
        if rule.allowed_values and str_value not in rule.allowed_values:
            raise ValidationError(field_name, f"Value must be one of: {rule.allowed_values}", value)
        
        # Forbidden patterns validation
        if rule.forbidden_patterns:
            for pattern in rule.forbidden_patterns:
                if re.search(pattern, str_value, re.IGNORECASE):
                    raise ValidationError(field_name, "Contains forbidden content", value)
        
        # Sanitization
        if rule.sanitize and isinstance(value, str):
            sanitized = self.sanitizer.sanitize_string(value)
            if sanitized != value:
                logger.info(f"Field '{field_name}' was sanitized")
    
    def _validate_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate request headers."""
        result = {"warnings": []}
        
        # Check for suspicious user agents
        user_agent = headers.get("user-agent", "").lower()
        suspicious_agents = ["sqlmap", "nikto", "nmap", "burp", "zap"]
        
        if any(agent in user_agent for agent in suspicious_agents):
            result["warnings"].append("Suspicious user agent detected")
        
        return result
    
    async def _check_suspicious_patterns(self, request: Request) -> List[str]:
        """Check for suspicious patterns in request."""
        warnings = []
        
        # Check URL for suspicious patterns
        url_str = str(request.url)
        if self._contains_suspicious_patterns(url_str):
            warnings.append("Suspicious URL patterns detected")
        
        # Check for high request rate (basic check)
        client_ip = request.client.host if request.client else "unknown"
        # In production, implement proper rate limiting here
        
        return warnings
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns."""
        all_patterns = (
            self.sanitizer.XSS_PATTERNS +
            self.sanitizer.SQL_INJECTION_PATTERNS +
            self.sanitizer.COMMAND_INJECTION_PATTERNS +
            self.sanitizer.PATH_TRAVERSAL_PATTERNS
        )
        
        for pattern in all_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


def create_validation_middleware(
    validation_rules: Optional[Dict[str, Dict[str, ValidationRule]]] = None
) -> InputValidationMiddleware:
    """Create input validation middleware with custom rules."""
    return lambda app: InputValidationMiddleware(app, validation_rules)