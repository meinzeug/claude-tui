"""
API Key and Authentication management for API Gateway.
"""

import hashlib
import secrets
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
import json
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

logger = logging.getLogger(__name__)


class APIKeyScope(Enum):
    """API key scopes."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    FULL_ACCESS = "full_access"


class APIKeyStatus(Enum):
    """API key status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class APIKey:
    """API key data structure."""
    key_id: str
    key_hash: str
    name: str
    description: str
    scopes: List[APIKeyScope]
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    rate_limit_override: Optional[Dict[str, int]]
    metadata: Dict[str, Any]
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if self.status != APIKeyStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def has_scope(self, required_scope: APIKeyScope) -> bool:
        """Check if API key has required scope."""
        if APIKeyScope.FULL_ACCESS in self.scopes:
            return True
        
        return required_scope in self.scopes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key_id': self.key_id,
            'name': self.name,
            'description': self.description,
            'scopes': [s.value for s in self.scopes],
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'usage_count': self.usage_count,
            'rate_limit_override': self.rate_limit_override,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIKey':
        """Create from dictionary."""
        return cls(
            key_id=data['key_id'],
            key_hash=data.get('key_hash', ''),
            name=data['name'],
            description=data['description'],
            scopes=[APIKeyScope(s) for s in data['scopes']],
            status=APIKeyStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            last_used_at=datetime.fromisoformat(data['last_used_at']) if data.get('last_used_at') else None,
            usage_count=data.get('usage_count', 0),
            rate_limit_override=data.get('rate_limit_override'),
            metadata=data.get('metadata', {})
        )


class APIKeyManager:
    """API key management system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.bearer_security = HTTPBearer()
        self.key_prefix = "apikey:"
        self.hash_prefix = "apihash:"
    
    async def initialize(self):
        """Initialize API key manager."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        logger.info("API Key Manager initialized")
    
    async def create_api_key(
        self,
        name: str,
        description: str,
        scopes: List[APIKeyScope],
        expires_in_days: Optional[int] = None,
        rate_limit_override: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, APIKey]:
        """Create a new API key."""
        if not self.redis_client:
            raise RuntimeError("API Key Manager not initialized")
        
        # Generate key components
        key_id = self._generate_key_id()
        raw_key = self._generate_raw_key()
        key_hash = self._hash_key(raw_key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            description=description,
            scopes=scopes,
            status=APIKeyStatus.ACTIVE,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_used_at=None,
            usage_count=0,
            rate_limit_override=rate_limit_override,
            metadata=metadata or {}
        )
        
        # Store in Redis
        await self.redis_client.set(
            f"{self.key_prefix}{key_id}",
            json.dumps(api_key.to_dict()),
            ex=int((expires_at - datetime.utcnow()).total_seconds()) if expires_at else None
        )
        
        # Store hash mapping
        await self.redis_client.set(
            f"{self.hash_prefix}{key_hash}",
            key_id,
            ex=int((expires_at - datetime.utcnow()).total_seconds()) if expires_at else None
        )
        
        # Return full key (only time it's revealed)
        full_key = f"ctui_{key_id}_{raw_key}"
        
        logger.info(f"Created API key: {key_id} for {name}")
        return full_key, api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate and return API key information."""
        if not self.redis_client:
            raise RuntimeError("API Key Manager not initialized")
        
        try:
            # Parse API key format: ctui_{key_id}_{raw_key}
            if not api_key.startswith("ctui_"):
                return None
            
            parts = api_key[5:].split("_", 1)  # Remove "ctui_" prefix
            if len(parts) != 2:
                return None
            
            key_id, raw_key = parts
            key_hash = self._hash_key(raw_key)
            
            # Verify hash matches stored key ID
            stored_key_id = await self.redis_client.get(f"{self.hash_prefix}{key_hash}")
            if not stored_key_id or stored_key_id.decode() != key_id:
                return None
            
            # Get API key data
            key_data = await self.redis_client.get(f"{self.key_prefix}{key_id}")
            if not key_data:
                return None
            
            api_key_obj = APIKey.from_dict(json.loads(key_data))
            api_key_obj.key_hash = key_hash  # Restore hash
            
            # Check validity
            if not api_key_obj.is_valid():
                return None
            
            # Update usage statistics
            await self._update_usage_stats(api_key_obj)
            
            return api_key_obj
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if not self.redis_client:
            raise RuntimeError("API Key Manager not initialized")
        
        try:
            # Get current key data
            key_data = await self.redis_client.get(f"{self.key_prefix}{key_id}")
            if not key_data:
                return False
            
            api_key = APIKey.from_dict(json.loads(key_data))
            api_key.status = APIKeyStatus.REVOKED
            
            # Update in Redis
            await self.redis_client.set(
                f"{self.key_prefix}{key_id}",
                json.dumps(api_key.to_dict())
            )
            
            # Remove hash mapping
            await self.redis_client.delete(f"{self.hash_prefix}{api_key.key_hash}")
            
            logger.info(f"Revoked API key: {key_id}")
            return True
            
        except Exception as e:
            logger.error(f"API key revocation error: {e}")
            return False
    
    async def list_api_keys(self, pattern: str = "*") -> List[APIKey]:
        """List API keys matching pattern."""
        if not self.redis_client:
            raise RuntimeError("API Key Manager not initialized")
        
        try:
            keys = await self.redis_client.keys(f"{self.key_prefix}{pattern}")
            api_keys = []
            
            for key in keys:
                key_data = await self.redis_client.get(key)
                if key_data:
                    api_key = APIKey.from_dict(json.loads(key_data))
                    api_keys.append(api_key)
            
            return api_keys
            
        except Exception as e:
            logger.error(f"API key listing error: {e}")
            return []
    
    async def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        if not self.redis_client:
            raise RuntimeError("API Key Manager not initialized")
        
        try:
            key_data = await self.redis_client.get(f"{self.key_prefix}{key_id}")
            if key_data:
                return APIKey.from_dict(json.loads(key_data))
            return None
            
        except Exception as e:
            logger.error(f"API key retrieval error: {e}")
            return None
    
    async def update_api_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scopes: Optional[List[APIKeyScope]] = None,
        status: Optional[APIKeyStatus] = None,
        rate_limit_override: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[APIKey]:
        """Update API key properties."""
        if not self.redis_client:
            raise RuntimeError("API Key Manager not initialized")
        
        try:
            # Get current key
            api_key = await self.get_api_key(key_id)
            if not api_key:
                return None
            
            # Update fields
            if name is not None:
                api_key.name = name
            if description is not None:
                api_key.description = description
            if scopes is not None:
                api_key.scopes = scopes
            if status is not None:
                api_key.status = status
            if rate_limit_override is not None:
                api_key.rate_limit_override = rate_limit_override
            if metadata is not None:
                api_key.metadata = metadata
            
            # Save updated key
            await self.redis_client.set(
                f"{self.key_prefix}{key_id}",
                json.dumps(api_key.to_dict())
            )
            
            logger.info(f"Updated API key: {key_id}")
            return api_key
            
        except Exception as e:
            logger.error(f"API key update error: {e}")
            return None
    
    async def cleanup_expired_keys(self):
        """Clean up expired API keys."""
        if not self.redis_client:
            return
        
        try:
            keys = await self.redis_client.keys(f"{self.key_prefix}*")
            expired_count = 0
            
            for key in keys:
                key_data = await self.redis_client.get(key)
                if key_data:
                    api_key = APIKey.from_dict(json.loads(key_data))
                    if not api_key.is_valid() and api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                        await self.redis_client.delete(key)
                        await self.redis_client.delete(f"{self.hash_prefix}{api_key.key_hash}")
                        expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired API keys")
                
        except Exception as e:
            logger.error(f"API key cleanup error: {e}")
    
    def _generate_key_id(self) -> str:
        """Generate unique key ID."""
        return secrets.token_hex(16)
    
    def _generate_raw_key(self) -> str:
        """Generate raw key component."""
        return secrets.token_hex(32)
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash raw key for storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    async def _update_usage_stats(self, api_key: APIKey):
        """Update usage statistics."""
        if not self.redis_client:
            return
        
        try:
            api_key.usage_count += 1
            api_key.last_used_at = datetime.utcnow()
            
            # Update in Redis
            await self.redis_client.set(
                f"{self.key_prefix}{api_key.key_id}",
                json.dumps(api_key.to_dict())
            )
            
        except Exception as e:
            logger.error(f"Usage stats update error: {e}")


class GatewayAuthenticator:
    """Gateway authentication handler."""
    
    def __init__(
        self,
        api_key_manager: APIKeyManager,
        jwt_secret: str,
        jwt_algorithm: str = "HS256"
    ):
        self.api_key_manager = api_key_manager
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.bearer_security = HTTPBearer()
    
    async def authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate incoming request."""
        auth_header = request.headers.get('authorization', '')
        
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            
            # Try JWT token first
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
                return {
                    'type': 'jwt',
                    'user_id': payload.get('sub'),
                    'scopes': payload.get('scopes', []),
                    'exp': payload.get('exp')
                }
            except jwt.InvalidTokenError:
                pass
            
            # Try API key
            api_key = await self.api_key_manager.validate_api_key(token)
            if api_key:
                return {
                    'type': 'api_key',
                    'key_id': api_key.key_id,
                    'scopes': [s.value for s in api_key.scopes],
                    'rate_limit_override': api_key.rate_limit_override
                }
        
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    def require_scope(self, required_scope: str):
        """Decorator to require specific scope."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract request from args/kwargs
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                
                if not request:
                    raise HTTPException(status_code=500, detail="Request not found")
                
                # Get authentication info
                auth_info = await self.authenticate_request(request)
                
                # Check scope
                if required_scope not in auth_info.get('scopes', []) and 'full_access' not in auth_info.get('scopes', []):
                    raise HTTPException(status_code=403, detail=f"Scope '{required_scope}' required")
                
                # Add auth info to request state
                request.state.auth = auth_info
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator