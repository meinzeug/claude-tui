"""
Advanced Input Sanitization and Validation
==========================================

Comprehensive input sanitization system to prevent injection attacks,
XSS, and other input-based vulnerabilities.
"""

import html
import json
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SanitizationLevel(Enum):
    """Sanitization levels for different contexts."""
    STRICT = "strict"          # Maximum sanitization
    MODERATE = "moderate"      # Balanced sanitization
    PERMISSIVE = "permissive"  # Minimal sanitization


class ValidationResult:
    """Result of input validation."""
    
    def __init__(self, is_valid: bool, sanitized_value: str, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.sanitized_value = sanitized_value
        self.errors = errors or []
        self.warnings = warnings or []


@dataclass
class SanitizationRule:
    """Individual sanitization rule."""
    pattern: str
    replacement: str
    description: str
    severity: str = "medium"


class AdvancedInputSanitizer:
    """Advanced input sanitization with context-aware rules."""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        SanitizationRule(
            pattern=r"(?i)(\bUNION\s+SELECT\b)",
            replacement="",
            description="SQL UNION SELECT injection attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(\bDROP\s+TABLE\b)",
            replacement="",
            description="SQL DROP TABLE injection attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(\bINSERT\s+INTO\b)",
            replacement="",
            description="SQL INSERT injection attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(\bDELETE\s+FROM\b)",
            replacement="",
            description="SQL DELETE injection attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(\bUPDATE\s+\w+\s+SET\b)",
            replacement="",
            description="SQL UPDATE injection attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(@@\w+)",
            replacement="",
            description="SQL system variable access attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(\binformation_schema\b)",
            replacement="",
            description="SQL information schema access attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(\bsys\.\w+)",
            replacement="",
            description="SQL system function access attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(\bxp_cmdshell\b)",
            replacement="",
            description="SQL command execution attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(\bsp_executesql\b)",
            replacement="",
            description="SQL dynamic execution attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(--\s*$)",
            replacement="",
            description="SQL comment injection",
            severity="medium"
        ),
        SanitizationRule(
            pattern=r"(/\*.*?\*/)",
            replacement="",
            description="SQL block comment injection",
            severity="medium"
        ),
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        SanitizationRule(
            pattern=r"(?i)(<script[^>]*>.*?</script>)",
            replacement="",
            description="Script tag XSS attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(<script[^>]*>)",
            replacement="",
            description="Opening script tag XSS attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(</script>)",
            replacement="",
            description="Closing script tag XSS attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(javascript:)",
            replacement="",
            description="JavaScript URL scheme XSS attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(vbscript:)",
            replacement="",
            description="VBScript URL scheme XSS attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(data:.*?script)",
            replacement="",
            description="Data URL script XSS attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(on\w+\s*=)",
            replacement="",
            description="HTML event handler XSS attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(<iframe[^>]*>)",
            replacement="",
            description="Iframe injection attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(<object[^>]*>)",
            replacement="",
            description="Object tag injection attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(<embed[^>]*>)",
            replacement="",
            description="Embed tag injection attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(<form[^>]*>)",
            replacement="",
            description="Form injection attempt",
            severity="medium"
        ),
        SanitizationRule(
            pattern=r"(?i)(expression\s*\()",
            replacement="",
            description="CSS expression XSS attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(behavior\s*:)",
            replacement="",
            description="CSS behavior XSS attempt",
            severity="high"
        ),
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        SanitizationRule(
            pattern=r"(;\s*\w+)",
            replacement="",
            description="Command chaining attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(\|\s*\w+)",
            replacement="",
            description="Command piping attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(&&\s*\w+)",
            replacement="",
            description="Command AND chaining attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(\|\|\s*\w+)",
            replacement="",
            description="Command OR chaining attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(`[^`]*`)",
            replacement="",
            description="Command substitution attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(\$\([^)]*\))",
            replacement="",
            description="Command substitution attempt",
            severity="critical"
        ),
    ]
    
    # LDAP injection patterns
    LDAP_INJECTION_PATTERNS = [
        SanitizationRule(
            pattern=r"(\*\)|\(\*)",
            replacement="",
            description="LDAP wildcard injection attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(\)\(|\(\&)",
            replacement="",
            description="LDAP logical operator injection attempt",
            severity="high"
        ),
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        SanitizationRule(
            pattern=r"(\.\.\/)",
            replacement="",
            description="Path traversal attempt (../)",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(\.\.\\)",
            replacement="",
            description="Path traversal attempt (..\\)",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(%2e%2e%2f)",
            replacement="",
            description="URL encoded path traversal attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(%2e%2e%5c)",
            replacement="",
            description="URL encoded path traversal attempt",
            severity="high"
        ),
    ]
    
    # NoSQL injection patterns
    NOSQL_INJECTION_PATTERNS = [
        SanitizationRule(
            pattern=r"(?i)(\$where)",
            replacement="",
            description="NoSQL $where injection attempt",
            severity="critical"
        ),
        SanitizationRule(
            pattern=r"(?i)(\$ne)",
            replacement="",
            description="NoSQL $ne injection attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(\$gt)",
            replacement="",
            description="NoSQL $gt injection attempt",
            severity="high"
        ),
        SanitizationRule(
            pattern=r"(?i)(\$regex)",
            replacement="",
            description="NoSQL $regex injection attempt",
            severity="high"
        ),
    ]
    
    def __init__(self, level: SanitizationLevel = SanitizationLevel.MODERATE):
        self.level = level
        self.logger = logging.getLogger(__name__)
    
    def sanitize_input(
        self,
        input_value: Any,
        context: str = "general",
        allow_html: bool = False,
        max_length: Optional[int] = None
    ) -> ValidationResult:
        """
        Comprehensive input sanitization.
        
        Args:
            input_value: Input to sanitize
            context: Context for sanitization (sql, html, json, etc.)
            allow_html: Whether to allow HTML tags
            max_length: Maximum allowed length
            
        Returns:
            ValidationResult with sanitized value and validation info
        """
        
        if input_value is None:
            return ValidationResult(True, "")
        
        # Convert to string
        if isinstance(input_value, (int, float, bool)):
            string_value = str(input_value)
        elif isinstance(input_value, bytes):
            try:
                string_value = input_value.decode('utf-8')
            except UnicodeDecodeError:
                return ValidationResult(False, "", ["Invalid byte sequence"])
        else:
            string_value = str(input_value)
        
        errors = []
        warnings = []
        sanitized = string_value
        
        # Length validation
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            warnings.append(f"Input truncated to {max_length} characters")
        
        # Remove null bytes
        if '\x00' in sanitized:
            sanitized = sanitized.replace('\x00', '')
            warnings.append("Null bytes removed")
        
        # Context-specific sanitization
        if context.lower() in ['sql', 'database']:
            sanitized, context_errors = self._sanitize_sql_context(sanitized)
            errors.extend(context_errors)
        
        elif context.lower() in ['html', 'web']:
            sanitized, context_errors = self._sanitize_html_context(sanitized, allow_html)
            errors.extend(context_errors)
        
        elif context.lower() in ['json', 'api']:
            sanitized, context_errors = self._sanitize_json_context(sanitized)
            errors.extend(context_errors)
        
        elif context.lower() in ['file', 'path']:
            sanitized, context_errors = self._sanitize_file_context(sanitized)
            errors.extend(context_errors)
        
        elif context.lower() in ['command', 'shell']:
            sanitized, context_errors = self._sanitize_command_context(sanitized)
            errors.extend(context_errors)
        
        else:
            # General sanitization
            sanitized, context_errors = self._sanitize_general_context(sanitized)
            errors.extend(context_errors)
        
        # Additional security checks
        sanitized, security_errors = self._apply_security_filters(sanitized)
        errors.extend(security_errors)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized, errors, warnings)
    
    def _sanitize_sql_context(self, value: str) -> Tuple[str, List[str]]:
        """Sanitize input for SQL context."""
        errors = []
        sanitized = value
        
        for rule in self.SQL_INJECTION_PATTERNS:
            matches = re.findall(rule.pattern, sanitized, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                errors.append(f"SQL injection attempt detected: {rule.description}")
                sanitized = re.sub(rule.pattern, rule.replacement, sanitized, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        return sanitized, errors
    
    def _sanitize_html_context(self, value: str, allow_html: bool) -> Tuple[str, List[str]]:
        """Sanitize input for HTML context."""
        errors = []
        sanitized = value
        
        # Apply XSS patterns
        for rule in self.XSS_PATTERNS:
            matches = re.findall(rule.pattern, sanitized, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                errors.append(f"XSS attempt detected: {rule.description}")
                sanitized = re.sub(rule.pattern, rule.replacement, sanitized, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        if not allow_html:
            # Escape all HTML entities
            sanitized = html.escape(sanitized, quote=True)
        else:
            # Allow specific safe tags only
            allowed_tags = {'b', 'i', 'u', 'em', 'strong', 'p', 'br', 'span'}
            sanitized = self._filter_html_tags(sanitized, allowed_tags)
        
        return sanitized, errors
    
    def _sanitize_json_context(self, value: str) -> Tuple[str, List[str]]:
        """Sanitize input for JSON context."""
        errors = []
        sanitized = value
        
        # Check for JSON injection attempts
        suspicious_patterns = [
            r'"\s*:\s*\{',  # Object injection
            r'"\s*:\s*\[',  # Array injection
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes
            r'\\u[0-9a-fA-F]{4}',  # Unicode escapes in suspicious contexts
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized):
                errors.append("Potential JSON injection detected")
                break
        
        # Escape JSON special characters
        json_escape_map = {
            '"': '\\"',
            '\\': '\\\\',
            '/': '\\/',
            '\b': '\\b',
            '\f': '\\f',
            '\n': '\\n',
            '\r': '\\r',
            '\t': '\\t'
        }
        
        for char, escape in json_escape_map.items():
            sanitized = sanitized.replace(char, escape)
        
        return sanitized, errors
    
    def _sanitize_file_context(self, value: str) -> Tuple[str, List[str]]:
        """Sanitize input for file/path context."""
        errors = []
        sanitized = value
        
        for rule in self.PATH_TRAVERSAL_PATTERNS:
            matches = re.findall(rule.pattern, sanitized, re.IGNORECASE)
            if matches:
                errors.append(f"Path traversal attempt detected: {rule.description}")
                sanitized = re.sub(rule.pattern, rule.replacement, sanitized, flags=re.IGNORECASE)
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '|', '&', ';', '`', '$']
        for char in dangerous_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '')
                errors.append(f"Dangerous character '{char}' removed")
        
        return sanitized, errors
    
    def _sanitize_command_context(self, value: str) -> Tuple[str, List[str]]:
        """Sanitize input for command/shell context."""
        errors = []
        sanitized = value
        
        for rule in self.COMMAND_INJECTION_PATTERNS:
            matches = re.findall(rule.pattern, sanitized)
            if matches:
                errors.append(f"Command injection attempt detected: {rule.description}")
                sanitized = re.sub(rule.pattern, rule.replacement, sanitized)
        
        return sanitized, errors
    
    def _sanitize_general_context(self, value: str) -> Tuple[str, List[str]]:
        """General sanitization for unknown contexts."""
        errors = []
        sanitized = value
        
        # Apply all patterns with lower severity thresholds
        all_patterns = (
            self.SQL_INJECTION_PATTERNS +
            self.XSS_PATTERNS +
            self.COMMAND_INJECTION_PATTERNS +
            self.PATH_TRAVERSAL_PATTERNS +
            self.NOSQL_INJECTION_PATTERNS +
            self.LDAP_INJECTION_PATTERNS
        )
        
        for rule in all_patterns:
            if rule.severity in ['critical', 'high']:
                matches = re.findall(rule.pattern, sanitized, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if matches:
                    errors.append(f"Security threat detected: {rule.description}")
                    sanitized = re.sub(rule.pattern, rule.replacement, sanitized, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        return sanitized, errors
    
    def _apply_security_filters(self, value: str) -> Tuple[str, List[str]]:
        """Apply additional security filters."""
        errors = []
        sanitized = value
        
        # Check for encoded attacks
        decoded_variants = [
            urllib.parse.unquote(sanitized),
            urllib.parse.unquote_plus(sanitized),
            html.unescape(sanitized)
        ]
        
        for variant in decoded_variants:
            if variant != sanitized:
                # Re-check decoded content for attacks
                temp_result = self.sanitize_input(variant, "general")
                if not temp_result.is_valid:
                    errors.append("Encoded attack detected")
                    break
        
        # Check for potential data exfiltration patterns
        exfiltration_patterns = [
            r'http://\d+\.\d+\.\d+\.\d+',  # Direct IP URLs
            r'ftp://[\w\.-]+',  # FTP URLs
            r'mailto:[\w\.-]+@[\w\.-]+',  # Email URLs
        ]
        
        for pattern in exfiltration_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                errors.append("Potential data exfiltration pattern detected")
        
        return sanitized, errors
    
    def _filter_html_tags(self, html_content: str, allowed_tags: set) -> str:
        """Filter HTML content to only allow specific tags."""
        import re
        
        # Remove all tags except allowed ones
        def replace_tag(match):
            tag_name = match.group(1).lower()
            if tag_name in allowed_tags:
                return match.group(0)
            else:
                return ''
        
        # Pattern to match HTML tags
        tag_pattern = r'<(/?)(\w+)(?:\s[^>]*)?>'
        sanitized = re.sub(tag_pattern, replace_tag, html_content, flags=re.IGNORECASE)
        
        return sanitized
    
    def validate_email(self, email: str) -> ValidationResult:
        """Validate and sanitize email address."""
        sanitized = email.strip().lower()
        errors = []
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, sanitized):
            errors.append("Invalid email format")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'[<>"\']',  # HTML/script injection
            r'javascript:',  # JavaScript injection
            r'\bscript\b',  # Script references
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                errors.append("Suspicious content in email address")
                break
        
        return ValidationResult(len(errors) == 0, sanitized, errors)
    
    def validate_url(self, url: str, allowed_schemes: List[str] = None) -> ValidationResult:
        """Validate and sanitize URL."""
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        sanitized = url.strip()
        errors = []
        
        try:
            parsed = urllib.parse.urlparse(sanitized)
            
            # Check scheme
            if parsed.scheme.lower() not in allowed_schemes:
                errors.append(f"URL scheme '{parsed.scheme}' not allowed")
            
            # Check for dangerous schemes
            dangerous_schemes = ['javascript', 'vbscript', 'data', 'file']
            if parsed.scheme.lower() in dangerous_schemes:
                errors.append("Dangerous URL scheme detected")
            
            # Check for private IP addresses in production
            import socket
            if parsed.hostname:
                try:
                    ip = socket.gethostbyname(parsed.hostname)
                    if self._is_private_ip(ip):
                        errors.append("Private IP address not allowed")
                except socket.gaierror:
                    pass  # Hostname resolution failed, but that's ok
            
        except Exception as e:
            errors.append(f"Invalid URL format: {str(e)}")
        
        return ValidationResult(len(errors) == 0, sanitized, errors)
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private/internal."""
        try:
            octets = [int(x) for x in ip.split('.')]
            
            # 10.0.0.0/8
            if octets[0] == 10:
                return True
            
            # 172.16.0.0/12
            if octets[0] == 172 and 16 <= octets[1] <= 31:
                return True
            
            # 192.168.0.0/16
            if octets[0] == 192 and octets[1] == 168:
                return True
            
            # 127.0.0.0/8 (localhost)
            if octets[0] == 127:
                return True
            
            return False
        except:
            return False
    
    def sanitize_filename(self, filename: str) -> ValidationResult:
        """Sanitize filename for safe filesystem operations."""
        sanitized = filename.strip()
        errors = []
        warnings = []
        
        # Remove path separators
        sanitized = sanitized.replace('/', '_').replace('\\', '_')
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        for char in dangerous_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '_')
                warnings.append(f"Dangerous character '{char}' replaced with '_'")
        
        # Check for reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        name_without_ext = sanitized.split('.')[0].upper()
        if name_without_ext in reserved_names:
            sanitized = f"file_{sanitized}"
            warnings.append("Reserved filename detected and prefixed")
        
        # Check length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
            warnings.append("Filename truncated to 255 characters")
        
        # Ensure not empty
        if not sanitized.strip():
            sanitized = "unnamed_file"
            errors.append("Empty filename not allowed")
        
        return ValidationResult(len(errors) == 0, sanitized, errors, warnings)


# Decorator for automatic input sanitization
def sanitize_inputs(context: str = "general", allow_html: bool = False):
    """Decorator to automatically sanitize function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            sanitizer = AdvancedInputSanitizer()
            
            # Sanitize positional arguments
            sanitized_args = []
            for arg in args:
                if isinstance(arg, str):
                    result = sanitizer.sanitize_input(arg, context, allow_html)
                    if not result.is_valid:
                        raise ValueError(f"Input validation failed: {'; '.join(result.errors)}")
                    sanitized_args.append(result.sanitized_value)
                else:
                    sanitized_args.append(arg)
            
            # Sanitize keyword arguments
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    result = sanitizer.sanitize_input(value, context, allow_html)
                    if not result.is_valid:
                        raise ValueError(f"Input validation failed for '{key}': {'; '.join(result.errors)}")
                    sanitized_kwargs[key] = result.sanitized_value
                else:
                    sanitized_kwargs[key] = value
            
            return func(*sanitized_args, **sanitized_kwargs)
        
        return wrapper
    return decorator


# Global sanitizer instance
default_sanitizer = AdvancedInputSanitizer()

# Convenience functions
def sanitize_sql_input(value: str) -> ValidationResult:
    """Quick SQL input sanitization."""
    return default_sanitizer.sanitize_input(value, "sql")

def sanitize_html_input(value: str, allow_html: bool = False) -> ValidationResult:
    """Quick HTML input sanitization."""
    return default_sanitizer.sanitize_input(value, "html", allow_html)

def sanitize_json_input(value: str) -> ValidationResult:
    """Quick JSON input sanitization."""
    return default_sanitizer.sanitize_input(value, "json")

def sanitize_general_input(value: str) -> ValidationResult:
    """Quick general input sanitization."""
    return default_sanitizer.sanitize_input(value, "general")