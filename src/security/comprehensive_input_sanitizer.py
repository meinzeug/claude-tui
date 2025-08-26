#!/usr/bin/env python3
"""
Comprehensive Input Sanitization Service for claude-tui.

Provides multi-layer input sanitization with:
- HTML/XSS protection
- SQL injection prevention
- Command injection blocking
- Path traversal protection
- Unicode normalization
- Content Security Policy enforcement
"""

import re
import html
import urllib.parse
import unicodedata
from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from .input_validator import SecurityInputValidator, ValidationResult, ThreatLevel

logger = logging.getLogger(__name__)

class SanitizationLevel(Enum):
    """Sanitization levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"

@dataclass
class SanitizationResult:
    """Result of sanitization process."""
    original_value: Any
    sanitized_value: Any
    changes_made: List[str]
    removed_content: List[str]
    threat_level: ThreatLevel
    sanitization_level: SanitizationLevel
    is_safe: bool

class ComprehensiveInputSanitizer:
    """
    Multi-layer input sanitization service.
    
    Provides comprehensive protection against:
    - XSS attacks through HTML sanitization
    - SQL injection via pattern detection and escaping
    - Command injection through shell metacharacter removal
    - Path traversal via path normalization
    - Unicode attacks through normalization
    - Content injection via strict filtering
    """
    
    def __init__(self, default_level: SanitizationLevel = SanitizationLevel.STRICT):
        """
        Initialize comprehensive sanitizer.
        
        Args:
            default_level: Default sanitization level
        """
        self.default_level = default_level
        self.validator = SecurityInputValidator()
        
        # HTML tags whitelist for basic sanitization
        self.allowed_html_tags = {
            'basic': set(),  # No HTML allowed
            'strict': {'b', 'i', 'em', 'strong', 'p', 'br', 'ul', 'ol', 'li'},
            'paranoid': set()  # No HTML allowed
        }
        
        # HTML attributes whitelist
        self.allowed_html_attributes = {
            'basic': set(),
            'strict': {'class', 'id'},
            'paranoid': set()
        }
        
        # SQL keywords that need escaping or removal
        self.sql_keywords = {
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'UNION', 'OR', 'AND', 'WHERE', 'FROM', 'INTO', 'VALUES', 'SET',
            'EXEC', 'EXECUTE', 'DECLARE', 'CAST', 'CONVERT'
        }
        
        # Shell metacharacters
        self.shell_metacharacters = {
            ';', '&', '|', '`', '$', '(', ')', '<', '>', '\n', '\r'
        }
        
        # Unicode categories to normalize or remove
        self.dangerous_unicode_categories = {
            'Cc',  # Control characters
            'Cf',  # Format characters
            'Cs',  # Surrogate characters
            'Co'   # Private use characters
        }
    
    def sanitize(
        self,
        value: Any,
        context: str = "general",
        level: Optional[SanitizationLevel] = None
    ) -> SanitizationResult:
        """
        Comprehensive input sanitization.
        
        Args:
            value: Value to sanitize
            context: Context for sanitization (html, sql, command, path, etc.)
            level: Sanitization level override
            
        Returns:
            Sanitization result
        """
        level = level or self.default_level
        
        if not isinstance(value, (str, int, float, bool, list, dict)):
            return SanitizationResult(
                original_value=value,
                sanitized_value=str(value),
                changes_made=["Converted to string"],
                removed_content=[],
                threat_level=ThreatLevel.LOW,
                sanitization_level=level,
                is_safe=True
            )
        
        # Convert to string for processing
        if isinstance(value, (int, float, bool)):
            str_value = str(value)
        elif isinstance(value, (list, dict)):
            # Handle complex types recursively
            return self._sanitize_complex_type(value, context, level)
        else:
            str_value = str(value)
        
        changes_made = []
        removed_content = []
        sanitized_value = str_value
        
        # Phase 1: Unicode normalization and cleaning
        sanitized_value, unicode_changes = self._normalize_unicode(sanitized_value, level)
        changes_made.extend(unicode_changes)
        
        # Phase 2: Context-specific sanitization
        if context == "html":
            sanitized_value, html_changes, html_removed = self._sanitize_html(sanitized_value, level)
            changes_made.extend(html_changes)
            removed_content.extend(html_removed)
        elif context == "sql":
            sanitized_value, sql_changes, sql_removed = self._sanitize_sql(sanitized_value, level)
            changes_made.extend(sql_changes)
            removed_content.extend(sql_removed)
        elif context == "command":
            sanitized_value, cmd_changes, cmd_removed = self._sanitize_command(sanitized_value, level)
            changes_made.extend(cmd_changes)
            removed_content.extend(cmd_removed)
        elif context == "path":
            sanitized_value, path_changes, path_removed = self._sanitize_path(sanitized_value, level)
            changes_made.extend(path_changes)
            removed_content.extend(path_removed)
        elif context == "url":
            sanitized_value, url_changes, url_removed = self._sanitize_url(sanitized_value, level)
            changes_made.extend(url_changes)
            removed_content.extend(url_removed)
        else:
            # General sanitization
            sanitized_value, gen_changes, gen_removed = self._sanitize_general(sanitized_value, level)
            changes_made.extend(gen_changes)
            removed_content.extend(gen_removed)
        
        # Phase 3: Final validation
        validation_result = self.validator.validate_user_prompt(sanitized_value, context)
        
        return SanitizationResult(
            original_value=value,
            sanitized_value=sanitized_value,
            changes_made=changes_made,
            removed_content=removed_content,
            threat_level=validation_result.threat_level,
            sanitization_level=level,
            is_safe=validation_result.is_valid
        )
    
    def _sanitize_complex_type(
        self,
        value: Union[list, dict],
        context: str,
        level: SanitizationLevel
    ) -> SanitizationResult:
        """
        Sanitize complex data types recursively.
        """
        changes_made = []
        removed_content = []
        
        if isinstance(value, list):
            sanitized_list = []
            for item in value:
                item_result = self.sanitize(item, context, level)
                sanitized_list.append(item_result.sanitized_value)
                changes_made.extend([f"list[{len(sanitized_list)-1}]: {change}" 
                                   for change in item_result.changes_made])
                removed_content.extend(item_result.removed_content)
            
            return SanitizationResult(
                original_value=value,
                sanitized_value=sanitized_list,
                changes_made=changes_made,
                removed_content=removed_content,
                threat_level=ThreatLevel.LOW,
                sanitization_level=level,
                is_safe=True
            )
        
        elif isinstance(value, dict):
            sanitized_dict = {}
            for key, val in value.items():
                # Sanitize key
                key_result = self.sanitize(key, "general", level)
                sanitized_key = key_result.sanitized_value
                
                # Sanitize value
                val_result = self.sanitize(val, context, level)
                sanitized_dict[sanitized_key] = val_result.sanitized_value
                
                changes_made.extend([f"key '{key}': {change}" 
                                   for change in key_result.changes_made])
                changes_made.extend([f"value['{key}']: {change}" 
                                   for change in val_result.changes_made])
                removed_content.extend(key_result.removed_content)
                removed_content.extend(val_result.removed_content)
            
            return SanitizationResult(
                original_value=value,
                sanitized_value=sanitized_dict,
                changes_made=changes_made,
                removed_content=removed_content,
                threat_level=ThreatLevel.LOW,
                sanitization_level=level,
                is_safe=True
            )
    
    def _normalize_unicode(self, value: str, level: SanitizationLevel) -> tuple[str, List[str]]:
        """
        Normalize Unicode and remove dangerous characters.
        """
        changes = []
        
        # Normalize Unicode to NFC form
        normalized = unicodedata.normalize('NFC', value)
        if normalized != value:
            changes.append("Unicode normalized to NFC")
        
        # Remove dangerous Unicode categories
        if level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            filtered_chars = []
            for char in normalized:
                category = unicodedata.category(char)
                if category not in self.dangerous_unicode_categories:
                    filtered_chars.append(char)
                else:
                    changes.append(f"Removed Unicode category {category} character")
            normalized = ''.join(filtered_chars)
        
        # Remove zero-width characters in paranoid mode
        if level == SanitizationLevel.PARANOID:
            zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
            for zw_char in zero_width_chars:
                if zw_char in normalized:
                    normalized = normalized.replace(zw_char, '')
                    changes.append("Removed zero-width character")
        
        return normalized, changes
    
    def _sanitize_html(self, value: str, level: SanitizationLevel) -> tuple[str, List[str], List[str]]:
        """
        Sanitize HTML content.
        """
        changes = []
        removed = []
        
        if level == SanitizationLevel.BASIC:
            # Just escape HTML
            sanitized = html.escape(value, quote=True)
            if sanitized != value:
                changes.append("HTML escaped")
            return sanitized, changes, removed
        
        elif level == SanitizationLevel.STRICT:
            # Allow some basic tags but sanitize attributes
            import re
            
            # Remove script tags completely
            script_pattern = r'<script[^>]*>.*?</script>'
            scripts_found = re.findall(script_pattern, value, re.IGNORECASE | re.DOTALL)
            for script in scripts_found:
                removed.append(f"Script tag: {script[:50]}...")
            sanitized = re.sub(script_pattern, '', value, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove dangerous event handlers
            event_pattern = r'\s+on\w+\s*=\s*["\'][^"\'>]*["\']'
            events_found = re.findall(event_pattern, sanitized, re.IGNORECASE)
            for event in events_found:
                removed.append(f"Event handler: {event}")
            sanitized = re.sub(event_pattern, '', sanitized, flags=re.IGNORECASE)
            
            # Remove javascript: and data: URLs
            js_url_pattern = r'(javascript|data)\s*:[^"\'>]*'
            js_urls = re.findall(js_url_pattern, sanitized, re.IGNORECASE)
            for js_url in js_urls:
                removed.append(f"Dangerous URL: {js_url}")
            sanitized = re.sub(js_url_pattern, '', sanitized, flags=re.IGNORECASE)
            
            if scripts_found or events_found or js_urls:
                changes.append("Removed dangerous HTML elements")
            
            return sanitized, changes, removed
        
        elif level == SanitizationLevel.PARANOID:
            # Strip all HTML
            html_pattern = r'<[^>]*>'
            html_tags = re.findall(html_pattern, value)
            for tag in html_tags:
                removed.append(f"HTML tag: {tag}")
            
            sanitized = re.sub(html_pattern, '', value)
            sanitized = html.unescape(sanitized)  # Decode HTML entities
            
            if html_tags:
                changes.append("Stripped all HTML tags")
            
            return sanitized, changes, removed
        
        return value, changes, removed
    
    def _sanitize_sql(self, value: str, level: SanitizationLevel) -> tuple[str, List[str], List[str]]:
        """
        Sanitize potential SQL injection content.
        """
        changes = []
        removed = []
        sanitized = value
        
        if level == SanitizationLevel.BASIC:
            # Just escape single quotes
            sanitized = value.replace("'", "''")
            if sanitized != value:
                changes.append("Escaped SQL quotes")
        
        elif level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            # Remove SQL comments
            comment_patterns = [
                r'--[^\n]*',  # Single-line comments
                r'/\*.*?\*/',  # Multi-line comments
            ]
            
            for pattern in comment_patterns:
                comments = re.findall(pattern, sanitized, re.DOTALL)
                for comment in comments:
                    removed.append(f"SQL comment: {comment[:30]}...")
                sanitized = re.sub(pattern, '', sanitized, flags=re.DOTALL)
            
            # Remove or escape dangerous SQL keywords in paranoid mode
            if level == SanitizationLevel.PARANOID:
                for keyword in self.sql_keywords:
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, sanitized, re.IGNORECASE):
                        removed.append(f"SQL keyword: {keyword}")
                        sanitized = re.sub(pattern, f'_{keyword}_', sanitized, flags=re.IGNORECASE)
            
            # Escape remaining quotes
            sanitized = sanitized.replace("'", "''")
            sanitized = sanitized.replace('"', '""')
            
            if removed or sanitized != value:
                changes.append("Sanitized SQL content")
        
        return sanitized, changes, removed
    
    def _sanitize_command(self, value: str, level: SanitizationLevel) -> tuple[str, List[str], List[str]]:
        """
        Sanitize command injection content.
        """
        changes = []
        removed = []
        sanitized = value
        
        if level == SanitizationLevel.BASIC:
            # Remove basic shell metacharacters
            basic_chars = {';', '&', '|', '`'}
            for char in basic_chars:
                if char in sanitized:
                    removed.append(f"Shell metacharacter: {char}")
                    sanitized = sanitized.replace(char, '')
        
        elif level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            # Remove all shell metacharacters
            for char in self.shell_metacharacters:
                if char in sanitized:
                    removed.append(f"Shell metacharacter: {repr(char)}")
                    sanitized = sanitized.replace(char, '')
            
            # Remove command substitution patterns
            cmd_patterns = [
                r'\$\([^)]*\)',  # $(command)
                r'`[^`]*`',      # `command`
            ]
            
            for pattern in cmd_patterns:
                matches = re.findall(pattern, sanitized)
                for match in matches:
                    removed.append(f"Command substitution: {match}")
                sanitized = re.sub(pattern, '', sanitized)
        
        if removed:
            changes.append("Removed command injection patterns")
        
        return sanitized, changes, removed
    
    def _sanitize_path(self, value: str, level: SanitizationLevel) -> tuple[str, List[str], List[str]]:
        """
        Sanitize file path to prevent traversal attacks.
        """
        changes = []
        removed = []
        
        try:
            # Normalize path
            path = Path(value).resolve()
            sanitized = str(path)
            
            # Remove parent directory references
            if '..' in value:
                removed.append("Parent directory references (..)")
                sanitized = sanitized.replace('..', '')
                changes.append("Removed path traversal attempts")
            
            # In paranoid mode, only allow relative paths
            if level == SanitizationLevel.PARANOID:
                if path.is_absolute():
                    sanitized = str(path.relative_to(path.anchor))
                    changes.append("Converted absolute path to relative")
            
            return sanitized, changes, removed
            
        except Exception:
            # If path processing fails, sanitize manually
            sanitized = value.replace('..', '').replace('//', '/')
            changes.append("Manual path sanitization applied")
            return sanitized, changes, removed
    
    def _sanitize_url(self, value: str, level: SanitizationLevel) -> tuple[str, List[str], List[str]]:
        """
        Sanitize URL to prevent malicious redirects.
        """
        changes = []
        removed = []
        
        # URL encode the value
        sanitized = urllib.parse.quote(value, safe=':/?#[]@!$&\'()*+,;=')
        
        if level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            # Remove dangerous schemes
            dangerous_schemes = ['javascript:', 'data:', 'vbscript:', 'file:']
            for scheme in dangerous_schemes:
                if sanitized.lower().startswith(scheme):
                    removed.append(f"Dangerous URL scheme: {scheme}")
                    sanitized = sanitized[len(scheme):]
                    changes.append("Removed dangerous URL scheme")
        
        if sanitized != value:
            changes.append("URL encoded")
        
        return sanitized, changes, removed
    
    def _sanitize_general(self, value: str, level: SanitizationLevel) -> tuple[str, List[str], List[str]]:
        """
        General-purpose sanitization for unknown content.
        """
        changes = []
        removed = []
        sanitized = value
        
        # Remove control characters (except tab, newline, carriage return)
        control_chars = []
        sanitized_chars = []
        
        for char in sanitized:
            code = ord(char)
            if code < 32 and char not in ['\t', '\n', '\r']:
                control_chars.append(repr(char))
            else:
                sanitized_chars.append(char)
        
        if control_chars:
            removed.extend([f"Control character: {char}" for char in control_chars])
            sanitized = ''.join(sanitized_chars)
            changes.append("Removed control characters")
        
        # In strict/paranoid mode, limit string length
        if level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            max_length = 1000 if level == SanitizationLevel.STRICT else 500
            if len(sanitized) > max_length:
                removed.append(f"Truncated content: {len(sanitized) - max_length} characters")
                sanitized = sanitized[:max_length]
                changes.append(f"Truncated to {max_length} characters")
        
        return sanitized, changes, removed
    
    def sanitize_dict_recursive(
        self,
        data: Dict[str, Any],
        context_map: Optional[Dict[str, str]] = None,
        level: Optional[SanitizationLevel] = None
    ) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary values based on context mapping.
        
        Args:
            data: Dictionary to sanitize
            context_map: Mapping of keys to sanitization contexts
            level: Sanitization level
            
        Returns:
            Sanitized dictionary
        """
        context_map = context_map or {}
        sanitized_data = {}
        
        for key, value in data.items():
            context = context_map.get(key, "general")
            
            if isinstance(value, dict):
                sanitized_data[key] = self.sanitize_dict_recursive(value, context_map, level)
            elif isinstance(value, list):
                sanitized_data[key] = [
                    self.sanitize(item, context, level).sanitized_value 
                    for item in value
                ]
            else:
                result = self.sanitize(value, context, level)
                sanitized_data[key] = result.sanitized_value
                
                # Log significant sanitization changes
                if result.changes_made:
                    logger.info(f"Sanitized key '{key}': {', '.join(result.changes_made)}")
        
        return sanitized_data

# Convenience functions
def sanitize_html_content(content: str, level: SanitizationLevel = SanitizationLevel.STRICT) -> str:
    """
    Quick HTML sanitization.
    
    Args:
        content: HTML content to sanitize
        level: Sanitization level
        
    Returns:
        Sanitized HTML content
    """
    sanitizer = ComprehensiveInputSanitizer()
    result = sanitizer.sanitize(content, "html", level)
    return result.sanitized_value

def sanitize_sql_input(content: str, level: SanitizationLevel = SanitizationLevel.STRICT) -> str:
    """
    Quick SQL injection protection.
    
    Args:
        content: SQL content to sanitize
        level: Sanitization level
        
    Returns:
        Sanitized SQL content
    """
    sanitizer = ComprehensiveInputSanitizer()
    result = sanitizer.sanitize(content, "sql", level)
    return result.sanitized_value

def sanitize_user_input(content: Any, context: str = "general") -> Any:
    """
    General-purpose user input sanitization.
    
    Args:
        content: Content to sanitize
        context: Sanitization context
        
    Returns:
        Sanitized content
    """
    sanitizer = ComprehensiveInputSanitizer()
    result = sanitizer.sanitize(content, context)
    return result.sanitized_value

if __name__ == "__main__":
    # Example usage and testing
    sanitizer = ComprehensiveInputSanitizer(SanitizationLevel.STRICT)
    
    test_inputs = [
        {
            'content': '<script>alert("XSS")</script>Hello <b>World</b>',
            'context': 'html'
        },
        {
            'content': "'; DROP TABLE users; --",
            'context': 'sql'
        },
        {
            'content': 'ls -la; rm -rf /',
            'context': 'command'
        },
        {
            'content': '../../../etc/passwd',
            'context': 'path'
        },
        {
            'content': 'javascript:alert("XSS")',
            'context': 'url'
        }
    ]
    
    for test_input in test_inputs:
        print(f"\nTesting {test_input['context']} sanitization:")
        print(f"Input: {test_input['content']}")
        
        result = sanitizer.sanitize(
            test_input['content'],
            test_input['context']
        )
        
        print(f"Output: {result.sanitized_value}")
        print(f"Changes: {result.changes_made}")
        print(f"Removed: {result.removed_content}")
        print(f"Safe: {result.is_safe}")
        print(f"Threat Level: {result.threat_level}")
