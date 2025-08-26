"""
Security Manager - Security utilities for safe AI integration.

Provides security validation, input sanitization, and safe execution
utilities for AI-generated content.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityManager:
    """Security utilities for AI integration."""
    
    def __init__(self):
        """Initialize security manager."""
        # Dangerous patterns
        self._dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.\w+',
            r'os\.system',
            r'os\.popen',
            r'os\.spawn\w+',
            r'rm\s+-rf',
            r'del\s+/',
            r'format\s+c:',
            r'fdisk',
            r'dd\s+if=',
        ]
        
        # Compile patterns
        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._dangerous_patterns]
        
        # Allowed imports (whitelist)
        self._allowed_imports = {
            'json', 'os', 'sys', 're', 'datetime', 'pathlib', 'typing',
            'dataclasses', 'enum', 'collections', 'itertools', 'functools',
            'asyncio', 'logging', 'unittest', 'pytest'
        }
    
    async def sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize AI prompt for security.
        
        Args:
            prompt: Input prompt to sanitize
            
        Returns:
            Sanitized prompt
        """
        if not prompt:
            return prompt
        
        # Remove potentially dangerous content
        sanitized = prompt
        
        # Remove or warn about dangerous patterns
        for pattern in self._compiled_patterns:
            if pattern.search(sanitized):
                logger.warning(f"Potentially dangerous pattern detected in prompt: {pattern.pattern}")
                # For now, just log - could replace or block
        
        return sanitized
    
    async def validate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize context dictionary.
        
        Args:
            context: Context dictionary to validate
            
        Returns:
            Sanitized context dictionary
        """
        if not context:
            return {}
        
        sanitized_context = {}
        
        for key, value in context.items():
            # Validate key
            if not isinstance(key, str) or not key.isalnum():
                logger.warning(f"Suspicious context key: {key}")
                continue
            
            # Validate value
            if isinstance(value, str):
                sanitized_context[key] = await self._sanitize_string_value(value)
            elif isinstance(value, (int, float, bool)):
                sanitized_context[key] = value
            elif isinstance(value, (list, dict)):
                # Recursively sanitize nested structures
                sanitized_context[key] = await self._sanitize_nested_value(value)
            else:
                logger.warning(f"Unsupported context value type: {type(value)}")
        
        return sanitized_context
    
    async def is_safe_code(self, code: str) -> bool:
        """
        Check if code is safe to execute.
        
        Args:
            code: Code to check
            
        Returns:
            True if code appears safe
        """
        if not code:
            return True
        
        # Check for dangerous patterns
        for pattern in self._compiled_patterns:
            if pattern.search(code):
                return False
        
        # Check imports
        import_matches = re.findall(r'import\s+(\w+)', code)
        from_matches = re.findall(r'from\s+(\w+)', code)
        
        all_imports = import_matches + from_matches
        
        for imp in all_imports:
            if imp not in self._allowed_imports:
                logger.warning(f"Potentially unsafe import: {imp}")
                return False
        
        return True
    
    async def _sanitize_string_value(self, value: str) -> str:
        """Sanitize string value."""
        # Remove or escape potentially dangerous content
        sanitized = value
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "... (truncated)"
            logger.warning("String value truncated for security")
        
        return sanitized
    
    async def _sanitize_nested_value(self, value: Any) -> Any:
        """Sanitize nested list/dict values."""
        if isinstance(value, dict):
            return {
                k: await self._sanitize_nested_value(v)
                for k, v in value.items()
                if isinstance(k, str) and k.isalnum()
            }
        elif isinstance(value, list):
            return [
                await self._sanitize_nested_value(item)
                for item in value[:100]  # Limit list size
            ]
        elif isinstance(value, str):
            return await self._sanitize_string_value(value)
        else:
            return value