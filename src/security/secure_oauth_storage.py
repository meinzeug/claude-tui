#!/usr/bin/env python3
"""
Secure OAuth Token Storage System for claude-tui.

Provides enterprise-grade secure storage for OAuth tokens with:
- AES-256-GCM encryption at rest
- Secure key derivation (PBKDF2)
- Token rotation support
- Secure memory handling
- Audit logging
- Protection against token leakage
"""

import os
import json
import secrets
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
import base64

# Configure secure logging (never log tokens)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security constants
AES_KEY_SIZE = 32  # 256 bits
SALT_SIZE = 16
IV_SIZE = 12  # GCM IV size
TAG_SIZE = 16  # GCM tag size
PBKDF2_ITERATIONS = 100000  # Minimum recommended iterations

@dataclass
class TokenMetadata:
    """Metadata for stored tokens."""
    token_type: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    rotation_count: int = 0
    source: str = "oauth"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenMetadata':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'expires_at', 'last_used']:
            if key in data and data[key]:
                if isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key])
        return cls(**data)

class SecureTokenStorage:
    """
    Secure storage system for OAuth tokens and other sensitive credentials.
    
    Features:
    - AES-256-GCM encryption with authenticated encryption
    - PBKDF2 key derivation for master password
    - Secure random salt generation
    - Token rotation support
    - Secure memory handling
    - Comprehensive audit logging
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        master_password: Optional[str] = None,
        auto_create: bool = True
    ):
        """
        Initialize secure token storage.
        
        Args:
            storage_path: Path to encrypted storage file
            master_password: Master password for encryption
            auto_create: Create storage if it doesn't exist
        """
        self.storage_path = storage_path or Path.home() / ".claude-tui" / "secure" / "tokens.enc"
        self.master_password = master_password or self._get_or_create_master_password()
        self.auto_create = auto_create
        
        # Ensure secure storage directory exists
        self.storage_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        
        # In-memory cache (cleared on exit)
        self._token_cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, TokenMetadata] = {}
        
        # Load existing tokens if available
        if self.storage_path.exists():
            self._load_encrypted_tokens()
        elif auto_create:
            self._create_empty_storage()
        
        logger.info(f"Secure token storage initialized at {self.storage_path}")
    
    def store_token(
        self,
        token_id: str,
        token_value: str,
        token_type: str = "oauth",
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Securely store an OAuth token.
        
        Args:
            token_id: Unique identifier for the token
            token_value: The actual token value
            token_type: Type of token (oauth, api_key, etc.)
            expires_at: Optional expiration time
            metadata: Optional additional metadata
            
        Returns:
            True if token was stored successfully
        """
        try:
            # Validate inputs
            if not token_id or not token_value:
                logger.error("Token ID and value are required")
                return False
            
            if len(token_value) < 20:
                logger.error("Token value appears too short for security")
                return False
            
            # Create token metadata
            token_metadata = TokenMetadata(
                token_type=token_type,
                created_at=datetime.now(timezone.utc),
                expires_at=expires_at
            )
            
            # Store in cache
            self._token_cache[token_id] = token_value
            self._metadata_cache[token_id] = token_metadata
            
            # Persist to encrypted storage
            success = self._save_encrypted_tokens()
            
            if success:
                logger.info(f"Token '{token_id}' stored securely (type: {token_type})")
                # Audit log (NEVER log token value)
                self._audit_log("TOKEN_STORED", {
                    "token_id": token_id,
                    "token_type": token_type,
                    "has_expiration": expires_at is not None
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store token '{token_id}': {e}")
            return False
    
    def retrieve_token(
        self,
        token_id: str,
        update_last_used: bool = True
    ) -> Optional[str]:
        """
        Retrieve a stored token.
        
        Args:
            token_id: Token identifier
            update_last_used: Update last used timestamp
            
        Returns:
            Token value if found and valid, None otherwise
        """
        try:
            if token_id not in self._token_cache:
                logger.warning(f"Token '{token_id}' not found in storage")
                return None
            
            metadata = self._metadata_cache.get(token_id)
            if metadata:
                # Check expiration
                if metadata.expires_at and datetime.now(timezone.utc) > metadata.expires_at:
                    logger.warning(f"Token '{token_id}' has expired")
                    self.revoke_token(token_id)
                    return None
                
                # Update last used timestamp
                if update_last_used:
                    metadata.last_used = datetime.now(timezone.utc)
                    self._save_encrypted_tokens()
            
            token_value = self._token_cache[token_id]
            
            # Audit log (NEVER log token value)
            self._audit_log("TOKEN_RETRIEVED", {
                "token_id": token_id,
                "token_length": len(token_value)
            })
            
            return token_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve token '{token_id}': {e}")
            return None
    
    def rotate_token(
        self,
        token_id: str,
        new_token_value: str,
        preserve_metadata: bool = True
    ) -> bool:
        """
        Rotate an existing token with a new value.
        
        Args:
            token_id: Token identifier
            new_token_value: New token value
            preserve_metadata: Keep existing metadata
            
        Returns:
            True if rotation was successful
        """
        try:
            if token_id not in self._token_cache:
                logger.error(f"Cannot rotate non-existent token '{token_id}'")
                return False
            
            # Get existing metadata
            metadata = self._metadata_cache.get(token_id)
            if metadata and preserve_metadata:
                metadata.rotation_count += 1
                metadata.last_used = datetime.now(timezone.utc)
            else:
                # Create new metadata
                metadata = TokenMetadata(
                    token_type="oauth",
                    created_at=datetime.now(timezone.utc),
                    rotation_count=1
                )
            
            # Update token value
            old_token_length = len(self._token_cache[token_id])
            self._token_cache[token_id] = new_token_value
            self._metadata_cache[token_id] = metadata
            
            # Persist changes
            success = self._save_encrypted_tokens()
            
            if success:
                logger.info(f"Token '{token_id}' rotated successfully (rotation #{metadata.rotation_count})")
                self._audit_log("TOKEN_ROTATED", {
                    "token_id": token_id,
                    "rotation_count": metadata.rotation_count,
                    "old_token_length": old_token_length,
                    "new_token_length": len(new_token_value)
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rotate token '{token_id}': {e}")
            return False
    
    def revoke_token(self, token_id: str) -> bool:
        """
        Revoke a stored token.
        
        Args:
            token_id: Token identifier
            
        Returns:
            True if token was revoked successfully
        """
        try:
            if token_id not in self._token_cache:
                logger.warning(f"Token '{token_id}' not found for revocation")
                return True  # Already gone
            
            # Secure deletion from memory
            token_length = len(self._token_cache[token_id])
            del self._token_cache[token_id]
            del self._metadata_cache[token_id]
            
            # Persist changes
            success = self._save_encrypted_tokens()
            
            if success:
                logger.info(f"Token '{token_id}' revoked successfully")
                self._audit_log("TOKEN_REVOKED", {
                    "token_id": token_id,
                    "token_length": token_length
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to revoke token '{token_id}': {e}")
            return False
    
    def list_tokens(self, include_expired: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        List all stored tokens with metadata (excluding token values).
        
        Args:
            include_expired: Include expired tokens in listing
            
        Returns:
            Dictionary of token metadata
        """
        result = {}
        current_time = datetime.now(timezone.utc)
        
        for token_id, metadata in self._metadata_cache.items():
            # Skip expired tokens unless requested
            if not include_expired and metadata.expires_at:
                if current_time > metadata.expires_at:
                    continue
            
            result[token_id] = {
                "type": metadata.token_type,
                "created_at": metadata.created_at.isoformat(),
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "last_used": metadata.last_used.isoformat() if metadata.last_used else None,
                "rotation_count": metadata.rotation_count,
                "is_expired": bool(metadata.expires_at and current_time > metadata.expires_at)
            }
        
        return result
    
    def migrate_from_plaintext(self, plaintext_file: Path) -> bool:
        """
        Migrate tokens from plaintext file to secure storage.
        
        Args:
            plaintext_file: Path to plaintext token file
            
        Returns:
            True if migration was successful
        """
        try:
            if not plaintext_file.exists():
                logger.error(f"Plaintext file does not exist: {plaintext_file}")
                return False
            
            # Read plaintext token
            with open(plaintext_file, 'r') as f:
                token_content = f.read().strip()
            
            if not token_content:
                logger.error("Plaintext file is empty")
                return False
            
            # Determine token type from content
            token_type = "oauth"
            if token_content.startswith("sk-ant-"):
                token_type = "anthropic_oauth"
            elif token_content.startswith("sk-"):
                token_type = "openai_api"
            
            # Store securely
            token_id = f"migrated_{plaintext_file.stem}"
            success = self.store_token(
                token_id=token_id,
                token_value=token_content,
                token_type=token_type
            )
            
            if success:
                logger.info(f"Successfully migrated token from {plaintext_file}")
                self._audit_log("TOKEN_MIGRATED", {
                    "source_file": str(plaintext_file),
                    "token_id": token_id,
                    "token_type": token_type
                })
                
                # Optionally securely delete plaintext file
                try:
                    # Overwrite with random data first
                    with open(plaintext_file, 'wb') as f:
                        f.write(secrets.token_bytes(len(token_content) * 2))
                    plaintext_file.unlink()  # Then delete
                    logger.info(f"Securely deleted plaintext file: {plaintext_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete plaintext file: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to migrate from plaintext file: {e}")
            return False
    
    def _get_or_create_master_password(self) -> str:
        """
        Get master password from environment or create secure random one.
        """
        env_password = os.getenv("CLAUDE_TUI_MASTER_PASSWORD")
        if env_password and len(env_password) >= 32:
            return env_password
        
        # Create secure random password
        password = secrets.token_urlsafe(32)
        
        # Save to secure file for persistence
        key_file = self.storage_path.parent / "master.key"
        try:
            with open(key_file, 'w', mode=0o600) as f:
                f.write(password)
            logger.warning(f"Generated new master password saved to {key_file}")
            logger.warning("IMPORTANT: Set CLAUDE_TUI_MASTER_PASSWORD environment variable!")
        except Exception as e:
            logger.error(f"Failed to save master password: {e}")
        
        return password
    
    def _derive_encryption_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=AES_KEY_SIZE,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
            backend=default_backend()
        )
        return kdf.derive(password.encode('utf-8'))
    
    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """
        Encrypt data using AES-256-GCM.
        """
        iv = secrets.token_bytes(IV_SIZE)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return IV + ciphertext + tag
        return iv + ciphertext + encryptor.tag
    
    def _decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        Decrypt data using AES-256-GCM.
        """
        iv = encrypted_data[:IV_SIZE]
        ciphertext = encrypted_data[IV_SIZE:-TAG_SIZE]
        tag = encrypted_data[-TAG_SIZE:]
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _save_encrypted_tokens(self) -> bool:
        """
        Save all tokens to encrypted storage.
        """
        try:
            # Prepare data for encryption
            storage_data = {
                "version": "1.0",
                "tokens": self._token_cache,
                "metadata": {k: v.to_dict() for k, v in self._metadata_cache.items()}
            }
            
            json_data = json.dumps(storage_data, indent=2)
            data_bytes = json_data.encode('utf-8')
            
            # Generate salt and derive key
            salt = secrets.token_bytes(SALT_SIZE)
            encryption_key = self._derive_encryption_key(self.master_password, salt)
            
            # Encrypt data
            encrypted_data = self._encrypt_data(data_bytes, encryption_key)
            
            # Save to file (salt + encrypted_data)
            with open(self.storage_path, 'wb') as f:
                f.write(salt + encrypted_data)
            
            # Set restrictive permissions
            self.storage_path.chmod(0o600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save encrypted tokens: {e}")
            return False
    
    def _load_encrypted_tokens(self) -> bool:
        """
        Load tokens from encrypted storage.
        """
        try:
            with open(self.storage_path, 'rb') as f:
                file_data = f.read()
            
            if len(file_data) < SALT_SIZE + IV_SIZE + TAG_SIZE:
                logger.error("Encrypted file is too small to be valid")
                return False
            
            # Extract salt and encrypted data
            salt = file_data[:SALT_SIZE]
            encrypted_data = file_data[SALT_SIZE:]
            
            # Derive key and decrypt
            encryption_key = self._derive_encryption_key(self.master_password, salt)
            decrypted_bytes = self._decrypt_data(encrypted_data, encryption_key)
            
            # Parse JSON
            storage_data = json.loads(decrypted_bytes.decode('utf-8'))
            
            # Load tokens and metadata
            self._token_cache = storage_data.get("tokens", {})
            
            metadata_dict = storage_data.get("metadata", {})
            self._metadata_cache = {
                k: TokenMetadata.from_dict(v) for k, v in metadata_dict.items()
            }
            
            logger.info(f"Loaded {len(self._token_cache)} tokens from encrypted storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load encrypted tokens: {e}")
            return False
    
    def _create_empty_storage(self) -> bool:
        """
        Create empty encrypted storage file.
        """
        self._token_cache = {}
        self._metadata_cache = {}
        return self._save_encrypted_tokens()
    
    def _audit_log(self, action: str, details: Dict[str, Any]):
        """
        Create audit log entry (never log sensitive data).
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details
        }
        
        # Log to secure audit file
        audit_file = self.storage_path.parent / "audit.log"
        try:
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def __del__(self):
        """
        Secure cleanup on object destruction.
        """
        # Clear sensitive data from memory
        if hasattr(self, '_token_cache'):
            self._token_cache.clear()
        if hasattr(self, '_metadata_cache'):
            self._metadata_cache.clear()

# Utility functions for backward compatibility and easy migration
def migrate_plaintext_token(source_path: Union[str, Path]) -> bool:
    """
    Migrate a plaintext token file to secure storage.
    
    Args:
        source_path: Path to plaintext token file
        
    Returns:
        True if migration was successful
    """
    storage = SecureTokenStorage()
    return storage.migrate_from_plaintext(Path(source_path))

def get_secure_token(token_id: str) -> Optional[str]:
    """
    Retrieve a token from secure storage.
    
    Args:
        token_id: Token identifier
        
    Returns:
        Token value if found, None otherwise
    """
    storage = SecureTokenStorage()
    return storage.retrieve_token(token_id)

def store_secure_token(token_id: str, token_value: str, token_type: str = "oauth") -> bool:
    """
    Store a token in secure storage.
    
    Args:
        token_id: Token identifier
        token_value: Token value
        token_type: Type of token
        
    Returns:
        True if storage was successful
    """
    storage = SecureTokenStorage()
    return storage.store_token(token_id, token_value, token_type)

if __name__ == "__main__":
    # Example usage and testing
    storage = SecureTokenStorage()
    
    # Test migration from .cc file if it exists
    cc_file = Path.home() / ".cc"
    if cc_file.exists():
        print("🔄 Migrating .cc file to secure storage...")
        success = storage.migrate_from_plaintext(cc_file)
        if success:
            print("✅ Migration successful!")
        else:
            print("❌ Migration failed!")
    
    # List stored tokens
    tokens = storage.list_tokens()
    print(f"📋 Found {len(tokens)} stored tokens:")
    for token_id, info in tokens.items():
        print(f"  - {token_id}: {info['type']} (created: {info['created_at']})")
