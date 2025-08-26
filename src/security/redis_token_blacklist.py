"""
Redis-based Token Blacklist System for Claude-TUI

Provides persistent token revocation capabilities using Redis as the backend.
This replaces the in-memory token blacklist with a distributed solution.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Set, Dict, Any, List

import redis.asyncio as aioredis
from redis.asyncio import Redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenBlacklistEntry(BaseModel):
    """Model for blacklisted token entries."""
    token_hash: str = Field(..., description="SHA256 hash of the token")
    user_id: str = Field(..., description="User ID associated with the token")
    revoked_at: datetime = Field(..., description="When the token was revoked")
    expires_at: datetime = Field(..., description="When the token expires")
    reason: Optional[str] = Field(None, description="Reason for revocation")
    revoked_by: Optional[str] = Field(None, description="Who revoked the token")


class RedisTokenBlacklist:
    """
    Redis-based Token Blacklist System.
    
    Provides distributed token revocation capabilities with:
    - Persistent storage in Redis
    - Automatic cleanup of expired tokens
    - Batch operations for performance
    - Monitoring and metrics
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/4",
        key_prefix: str = "auth:blacklist:",
        cleanup_interval: int = 3600,  # 1 hour
        max_pool_connections: int = 10
    ):
        """
        Initialize Redis Token Blacklist.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
            cleanup_interval: Seconds between cleanup operations
            max_pool_connections: Maximum Redis pool connections
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.cleanup_interval = cleanup_interval
        self.max_pool_connections = max_pool_connections
        
        self._redis: Optional[Redis] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics = {
            "tokens_blacklisted": 0,
            "tokens_checked": 0,
            "cleanup_runs": 0,
            "tokens_expired": 0
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection and start background tasks."""
        try:
            # Create Redis connection pool
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.max_pool_connections,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self._redis.ping()
            logger.info("Redis token blacklist initialized successfully")
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis token blacklist: {e}")
            raise
    
    async def close(self) -> None:
        """Close Redis connection and stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._redis:
            await self._redis.close()
            logger.info("Redis token blacklist closed")
    
    def _get_token_hash(self, token: str) -> str:
        """Generate SHA256 hash for token."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _get_redis_key(self, token_hash: str) -> str:
        """Generate Redis key for token hash."""
        return f"{self.key_prefix}{token_hash}"
    
    async def add_token(
        self,
        token: str,
        user_id: str,
        expires_at: datetime,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None
    ) -> bool:
        """
        Add token to blacklist.
        
        Args:
            token: JWT token to blacklist
            user_id: User ID associated with the token
            expires_at: When the token expires
            reason: Reason for revocation
            revoked_by: Who revoked the token
            
        Returns:
            True if token was added successfully
        """
        if not self._redis:
            raise RuntimeError("Redis connection not initialized")
        
        try:
            token_hash = self._get_token_hash(token)
            redis_key = self._get_redis_key(token_hash)
            
            entry = TokenBlacklistEntry(
                token_hash=token_hash,
                user_id=user_id,
                revoked_at=datetime.now(timezone.utc),
                expires_at=expires_at,
                reason=reason,
                revoked_by=revoked_by
            )
            
            # Calculate TTL based on token expiration
            now = datetime.now(timezone.utc)
            if expires_at > now:
                ttl_seconds = int((expires_at - now).total_seconds()) + 3600  # Extra hour buffer
            else:
                ttl_seconds = 3600  # 1 hour for already expired tokens
            
            # Store in Redis with TTL
            await self._redis.setex(
                redis_key,
                ttl_seconds,
                entry.model_dump_json()
            )
            
            # Add to user's blacklisted tokens set
            user_key = f"{self.key_prefix}user:{user_id}"
            await self._redis.sadd(user_key, token_hash)
            await self._redis.expire(user_key, ttl_seconds)
            
            self._metrics["tokens_blacklisted"] += 1
            logger.info(f"Token blacklisted for user {user_id}: {token_hash[:16]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add token to blacklist: {e}")
            return False
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if token is blacklisted.
        
        Args:
            token: JWT token to check
            
        Returns:
            True if token is blacklisted
        """
        if not self._redis:
            raise RuntimeError("Redis connection not initialized")
        
        try:
            token_hash = self._get_token_hash(token)
            redis_key = self._get_redis_key(token_hash)
            
            result = await self._redis.exists(redis_key)
            self._metrics["tokens_checked"] += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to check token blacklist: {e}")
            # Fail secure - assume token is blacklisted if we can't check
            return True
    
    async def get_blacklist_entry(self, token: str) -> Optional[TokenBlacklistEntry]:
        """
        Get blacklist entry for token.
        
        Args:
            token: JWT token
            
        Returns:
            Blacklist entry if exists
        """
        if not self._redis:
            raise RuntimeError("Redis connection not initialized")
        
        try:
            token_hash = self._get_token_hash(token)
            redis_key = self._get_redis_key(token_hash)
            
            entry_json = await self._redis.get(redis_key)
            if entry_json:
                return TokenBlacklistEntry.model_validate_json(entry_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get blacklist entry: {e}")
            return None
    
    async def remove_token(self, token: str) -> bool:
        """
        Remove token from blacklist.
        
        Args:
            token: JWT token to remove
            
        Returns:
            True if token was removed
        """
        if not self._redis:
            raise RuntimeError("Redis connection not initialized")
        
        try:
            token_hash = self._get_token_hash(token)
            redis_key = self._get_redis_key(token_hash)
            
            # Get entry to find user_id
            entry_json = await self._redis.get(redis_key)
            if entry_json:
                entry = TokenBlacklistEntry.model_validate_json(entry_json)
                
                # Remove from user's set
                user_key = f"{self.key_prefix}user:{entry.user_id}"
                await self._redis.srem(user_key, token_hash)
            
            # Remove the main entry
            result = await self._redis.delete(redis_key)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to remove token from blacklist: {e}")
            return False
    
    async def blacklist_user_tokens(
        self,
        user_id: str,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None
    ) -> int:
        """
        Blacklist all active tokens for a user.
        
        Args:
            user_id: User ID
            reason: Reason for revocation
            revoked_by: Who revoked the tokens
            
        Returns:
            Number of tokens blacklisted
        """
        if not self._redis:
            raise RuntimeError("Redis connection not initialized")
        
        try:
            user_key = f"{self.key_prefix}user:{user_id}"
            token_hashes = await self._redis.smembers(user_key)
            
            count = 0
            for token_hash in token_hashes:
                redis_key = self._get_redis_key(token_hash)
                
                # Update existing entry or mark as revoked
                entry_json = await self._redis.get(redis_key)
                if entry_json:
                    entry = TokenBlacklistEntry.model_validate_json(entry_json)
                    entry.reason = reason or entry.reason
                    entry.revoked_by = revoked_by or entry.revoked_by
                    entry.revoked_at = datetime.now(timezone.utc)
                    
                    # Update with same TTL
                    ttl = await self._redis.ttl(redis_key)
                    if ttl > 0:
                        await self._redis.setex(
                            redis_key,
                            ttl,
                            entry.model_dump_json()
                        )
                        count += 1
            
            logger.info(f"Blacklisted {count} tokens for user {user_id}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to blacklist user tokens: {e}")
            return 0
    
    async def get_user_blacklisted_tokens(self, user_id: str) -> List[TokenBlacklistEntry]:
        """
        Get all blacklisted tokens for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of blacklisted token entries
        """
        if not self._redis:
            raise RuntimeError("Redis connection not initialized")
        
        try:
            user_key = f"{self.key_prefix}user:{user_id}"
            token_hashes = await self._redis.smembers(user_key)
            
            entries = []
            for token_hash in token_hashes:
                redis_key = self._get_redis_key(token_hash)
                entry_json = await self._redis.get(redis_key)
                
                if entry_json:
                    entry = TokenBlacklistEntry.model_validate_json(entry_json)
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to get user blacklisted tokens: {e}")
            return []
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired token entries.
        
        Returns:
            Number of tokens cleaned up
        """
        if not self._redis:
            raise RuntimeError("Redis connection not initialized")
        
        try:
            # Redis TTL handles most cleanup automatically
            # This method handles additional cleanup of user sets
            
            pattern = f"{self.key_prefix}user:*"
            cursor = "0"
            cleaned_count = 0
            
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    # Get all token hashes for this user
                    token_hashes = await self._redis.smembers(key)
                    
                    # Check which tokens still exist
                    for token_hash in token_hashes:
                        redis_key = self._get_redis_key(token_hash)
                        if not await self._redis.exists(redis_key):
                            # Token expired, remove from user set
                            await self._redis.srem(key, token_hash)
                            cleaned_count += 1
                    
                    # Remove user key if empty
                    if await self._redis.scard(key) == 0:
                        await self._redis.delete(key)
                
                if cursor == "0":
                    break
            
            self._metrics["tokens_expired"] += cleaned_count
            self._metrics["cleanup_runs"] += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired token references")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0
    
    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_tokens()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get blacklist metrics.
        
        Returns:
            Dictionary with metrics data
        """
        if not self._redis:
            return {"error": "Redis connection not initialized"}
        
        try:
            # Count active blacklisted tokens
            pattern = f"{self.key_prefix}*"
            cursor = "0"
            active_tokens = 0
            user_count = 0
            
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    if key.endswith(":user:"):
                        user_count += 1
                    else:
                        active_tokens += 1
                
                if cursor == "0":
                    break
            
            return {
                **self._metrics,
                "active_blacklisted_tokens": active_tokens,
                "users_with_blacklisted_tokens": user_count,
                "redis_connected": await self._redis.ping(),
                "cleanup_interval_seconds": self.cleanup_interval
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status information
        """
        try:
            if not self._redis:
                return {"healthy": False, "error": "Redis connection not initialized"}
            
            # Test Redis connectivity
            await self._redis.ping()
            
            # Check cleanup task
            cleanup_healthy = (
                self._cleanup_task is not None and 
                not self._cleanup_task.done()
            )
            
            return {
                "healthy": True,
                "redis_connected": True,
                "cleanup_task_running": cleanup_healthy,
                "metrics": await self.get_metrics()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "redis_connected": False
            }


# Global instance
_token_blacklist: Optional[RedisTokenBlacklist] = None


def get_token_blacklist() -> RedisTokenBlacklist:
    """Get global token blacklist instance."""
    global _token_blacklist
    if _token_blacklist is None:
        raise RuntimeError("Token blacklist not initialized. Call init_token_blacklist() first.")
    return _token_blacklist


async def init_token_blacklist(
    redis_url: str = "redis://localhost:6379/4",
    **kwargs
) -> RedisTokenBlacklist:
    """Initialize global token blacklist instance."""
    global _token_blacklist
    
    _token_blacklist = RedisTokenBlacklist(redis_url=redis_url, **kwargs)
    await _token_blacklist.initialize()
    
    return _token_blacklist


async def close_token_blacklist() -> None:
    """Close global token blacklist instance."""
    global _token_blacklist
    
    if _token_blacklist:
        await _token_blacklist.close()
        _token_blacklist = None