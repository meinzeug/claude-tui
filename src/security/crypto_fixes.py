"""
Cryptographic Security Fixes for Claude-TUI

This module provides secure replacements for weak cryptographic functions
found throughout the codebase, specifically replacing MD5 with SHA-256.

Security fixes implemented:
- MD5 → SHA-256 replacement
- Secure random number generation
- Safe temporary file handling
- Cryptographically secure token generation

Author: Security Specialist - Hive Mind Team
Date: 2025-08-26
"""

import hashlib
import secrets
import tempfile
import os
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def secure_hash(data: Union[str, bytes], algorithm: str = "sha256", length: Optional[int] = None) -> str:
    """
    Secure hash function replacing MD5 usage throughout the system.
    
    Args:
        data: Data to hash (string or bytes)
        algorithm: Hash algorithm (sha256, sha384, sha512)
        length: Optional length limit for output (for compatibility)
        
    Returns:
        Hexadecimal hash string
        
    Example:
        # Replace: hashlib.md5(content.encode()).hexdigest()[:16]
        # With:    secure_hash(content, length=16)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
        
    hash_algorithms = {
        'sha256': hashlib.sha256,
        'sha384': hashlib.sha384,
        'sha512': hashlib.sha512
    }
    
    hash_func = hash_algorithms.get(algorithm.lower(), hashlib.sha256)
    hash_digest = hash_func(data).hexdigest()
    
    if length and length > 0:
        return hash_digest[:length]
    
    return hash_digest


def secure_token(length: int = 32) -> str:
    """
    Generate cryptographically secure token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        URL-safe base64 encoded token
    """
    return secrets.token_urlsafe(length)


def secure_random_int(min_val: int, max_val: int) -> int:
    """
    Generate cryptographically secure random integer.
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        
    Returns:
        Secure random integer in range
    """
    return secrets.randbelow(max_val - min_val + 1) + min_val


def create_secure_temp_file(suffix: str = "", prefix: str = "secure_", dir_path: Optional[str] = None) -> str:
    """
    Create secure temporary file with restricted permissions.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir_path: Directory path (defaults to secure temp directory)
        
    Returns:
        Path to created temporary file
    """
    if dir_path is None:
        # Create secure temp directory
        secure_temp_dir = Path(tempfile.gettempdir()) / "claude-tui-secure"
        secure_temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        dir_path = str(secure_temp_dir)
    
    fd, path = tempfile.mkstemp(
        suffix=suffix,
        prefix=prefix,
        dir=dir_path
    )
    os.close(fd)
    
    # Set restrictive permissions (owner read/write only)
    os.chmod(path, 0o600)
    
    return path


def create_secure_temp_dir(suffix: str = "", prefix: str = "secure_", base_dir: Optional[str] = None) -> str:
    """
    Create secure temporary directory with restricted permissions.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        base_dir: Base directory path
        
    Returns:
        Path to created temporary directory
    """
    if base_dir is None:
        base_dir = Path(tempfile.gettempdir()) / "claude-tui-secure"
        base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        base_dir = str(base_dir)
    
    temp_dir = tempfile.mkdtemp(
        suffix=suffix,
        prefix=prefix,
        dir=base_dir
    )
    
    # Set restrictive permissions (owner read/write/execute only)
    os.chmod(temp_dir, 0o700)
    
    return temp_dir


# Legacy compatibility functions for easy replacement
def md5_replacement(data: Union[str, bytes]) -> 'HashCompat':
    """
    Drop-in replacement for hashlib.md5() calls.
    
    Returns a hash-compatible object that uses SHA-256 internally.
    """
    return HashCompat(data, 'sha256')


class HashCompat:
    """
    Compatibility class for replacing hashlib.md5() objects.
    
    Provides the same interface as hashlib hash objects but uses secure algorithms.
    """
    
    def __init__(self, initial_data: Union[str, bytes] = b"", algorithm: str = "sha256"):
        """Initialize hash compatibility object."""
        self.algorithm = algorithm
        
        hash_algorithms = {
            'sha256': hashlib.sha256,
            'sha384': hashlib.sha384,
            'sha512': hashlib.sha512
        }
        
        self._hasher = hash_algorithms.get(algorithm, hashlib.sha256)()
        
        if initial_data:
            self.update(initial_data)
    
    def update(self, data: Union[str, bytes]) -> None:
        """Update hash with new data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        self._hasher.update(data)
    
    def digest(self) -> bytes:
        """Return binary hash digest."""
        return self._hasher.digest()
    
    def hexdigest(self) -> str:
        """Return hexadecimal hash digest."""
        return self._hasher.hexdigest()
    
    def copy(self) -> 'HashCompat':
        """Return copy of hash object."""
        new_hash = HashCompat(algorithm=self.algorithm)
        new_hash._hasher = self._hasher.copy()
        return new_hash


# Monkey-patch replacement functions (use with caution in production)
def patch_md5_usage():
    """
    Monkey-patch MD5 usage with secure alternatives.
    
    WARNING: This is a temporary fix. Code should be updated to use secure_hash() directly.
    """
    import hashlib
    
    # Store original function
    original_md5 = hashlib.md5
    
    def secure_md5_replacement(*args, **kwargs):
        """Replacement for hashlib.md5() that uses SHA-256."""
        logger.warning("MD5 usage detected and replaced with SHA-256. Please update code to use secure_hash().")
        
        initial_data = args[0] if args else b""
        return HashCompat(initial_data, 'sha256')
    
    # Replace the function
    hashlib.md5 = secure_md5_replacement
    
    logger.info("✅ MD5 usage has been patched with secure SHA-256 alternative")
    
    return original_md5


def unpatch_md5_usage(original_md5):
    """
    Restore original MD5 function (for testing purposes).
    
    Args:
        original_md5: Original MD5 function returned by patch_md5_usage()
    """
    import hashlib
    hashlib.md5 = original_md5
    logger.info("MD5 function restored to original")


# Application-specific secure hash functions for common use cases
def hash_user_id_task(user_id: str, task_name: str, features: list = None) -> str:
    """
    Secure hash for user ID and task combination.
    
    Replacement for pattern_engine.py line 594.
    """
    features_str = ",".join(sorted(features)) if features else ""
    content = f"{user_id}:{task_name}:{features_str}"
    return secure_hash(content, length=16)


def hash_client_ip(client_ip: str, server_list_size: int) -> int:
    """
    Secure hash for client IP load balancing.
    
    Replacement for gateway/core.py line 310.
    """
    hash_value = secure_hash(client_ip)
    # Convert first 8 hex chars to int for consistent load balancing
    return int(hash_value[:8], 16) % server_list_size


def hash_request_url(url: str) -> str:
    """
    Secure hash for request URL caching.
    
    Replacement for gateway/core.py lines 404, 422.
    """
    return secure_hash(str(url))


def hash_auth_token(token: str) -> str:
    """
    Secure hash for authentication token identification.
    
    Replacement for gateway/middleware.py line 220.
    """
    return secure_hash(token, length=12)


def hash_cache_key(key_parts: list) -> str:
    """
    Secure hash for cache key generation.
    
    Replacement for gateway/middleware.py line 472.
    """
    key_string = '|'.join(str(part) for part in key_parts)
    return secure_hash(key_string)


def hash_input_data(input_data: any) -> str:
    """
    Secure hash for neural trainer input data.
    
    Replacement for neural_trainer.py line 730.
    """
    return secure_hash(str(input_data), length=8)


def hash_signature_data(signature_data: dict) -> str:
    """
    Secure hash for performance signature generation.
    
    Replacement for regression_tester.py line 320.
    """
    import json
    signature_str = json.dumps(signature_data, sort_keys=True)
    return secure_hash(signature_str, length=16)


# Export secure functions for easy import
__all__ = [
    'secure_hash',
    'secure_token', 
    'secure_random_int',
    'create_secure_temp_file',
    'create_secure_temp_dir',
    'md5_replacement',
    'HashCompat',
    'patch_md5_usage',
    'unpatch_md5_usage',
    'hash_user_id_task',
    'hash_client_ip', 
    'hash_request_url',
    'hash_auth_token',
    'hash_cache_key',
    'hash_input_data',
    'hash_signature_data'
]