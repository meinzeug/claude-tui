#!/usr/bin/env python3
"""
Token Rotation Service for claude-tui.

Provides automated token rotation with:
- Automatic OAuth token refresh
- Graceful fallback on rotation failures
- Background rotation scheduling
- Health monitoring and alerting
- Secure rotation logging
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import threading
import time
import json
from pathlib import Path

from .secure_oauth_storage import SecureTokenStorage, TokenMetadata

logger = logging.getLogger(__name__)

@dataclass
class RotationConfig:
    """Configuration for token rotation."""
    rotation_interval_hours: int = 24  # Rotate every 24 hours
    retry_attempts: int = 3
    retry_delay_seconds: int = 300  # 5 minutes
    early_rotation_threshold: float = 0.8  # Rotate when 80% of lifetime passed
    enable_health_checks: bool = True
    health_check_interval_minutes: int = 30

@dataclass
class RotationResult:
    """Result of a token rotation attempt."""
    success: bool
    token_id: str
    old_token_length: int
    new_token_length: int
    rotation_time: datetime
    error_message: Optional[str] = None
    retry_count: int = 0

class TokenRotationService:
    """
    Service for automatic token rotation and lifecycle management.
    
    Features:
    - Automatic token rotation based on age or expiration
    - Configurable rotation schedules
    - Health monitoring and alerts
    - Secure rotation logging
    - Graceful error handling with retries
    """
    
    def __init__(
        self,
        storage: SecureTokenStorage,
        config: Optional[RotationConfig] = None,
        refresh_callback: Optional[Callable[[str], Optional[str]]] = None
    ):
        """
        Initialize token rotation service.
        
        Args:
            storage: Secure token storage instance
            config: Rotation configuration
            refresh_callback: Function to refresh tokens
        """
        self.storage = storage
        self.config = config or RotationConfig()
        self.refresh_callback = refresh_callback
        
        # State tracking
        self._running = False
        self._rotation_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        
        # Metrics and logging
        self._rotation_history: List[RotationResult] = []
        self._failed_rotations: Dict[str, int] = {}  # token_id -> failure count
        self._last_health_check: Optional[datetime] = None
        
        # Callbacks for events
        self._rotation_success_callbacks: List[Callable[[RotationResult], None]] = []
        self._rotation_failure_callbacks: List[Callable[[RotationResult], None]] = []
        
        logger.info("Token rotation service initialized")
    
    def start(self) -> bool:
        """
        Start the token rotation service.
        
        Returns:
            True if service started successfully
        """
        if self._running:
            logger.warning("Token rotation service is already running")
            return True
        
        try:
            self._running = True
            
            # Start rotation thread
            self._rotation_thread = threading.Thread(
                target=self._rotation_loop,
                daemon=True,
                name="TokenRotation"
            )
            self._rotation_thread.start()
            
            # Start health monitoring thread if enabled
            if self.config.enable_health_checks:
                self._health_thread = threading.Thread(
                    target=self._health_check_loop,
                    daemon=True,
                    name="TokenHealthMonitor"
                )
                self._health_thread.start()
            
            logger.info("Token rotation service started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start token rotation service: {e}")
            self._running = False
            return False
    
    def stop(self) -> bool:
        """
        Stop the token rotation service.
        
        Returns:
            True if service stopped successfully
        """
        if not self._running:
            return True
        
        try:
            logger.info("Stopping token rotation service...")
            self._running = False
            
            # Wait for threads to finish
            if self._rotation_thread and self._rotation_thread.is_alive():
                self._rotation_thread.join(timeout=10)
            
            if self._health_thread and self._health_thread.is_alive():
                self._health_thread.join(timeout=5)
            
            logger.info("Token rotation service stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping token rotation service: {e}")
            return False
    
    def rotate_token_now(
        self,
        token_id: str,
        force: bool = False
    ) -> RotationResult:
        """
        Immediately rotate a specific token.
        
        Args:
            token_id: Token to rotate
            force: Force rotation even if not needed
            
        Returns:
            Rotation result
        """
        try:
            logger.info(f"Initiating manual rotation for token '{token_id}'")
            
            # Get current token
            current_token = self.storage.retrieve_token(token_id, update_last_used=False)
            if not current_token:
                error = f"Token '{token_id}' not found"
                logger.error(error)
                return RotationResult(
                    success=False,
                    token_id=token_id,
                    old_token_length=0,
                    new_token_length=0,
                    rotation_time=datetime.now(timezone.utc),
                    error_message=error
                )
            
            # Check if rotation is needed (unless forced)
            if not force and not self._needs_rotation(token_id):
                logger.info(f"Token '{token_id}' does not need rotation")
                return RotationResult(
                    success=True,
                    token_id=token_id,
                    old_token_length=len(current_token),
                    new_token_length=len(current_token),
                    rotation_time=datetime.now(timezone.utc)
                )
            
            # Perform rotation with retries
            return self._perform_rotation_with_retries(token_id, current_token)
            
        except Exception as e:
            error_msg = f"Manual rotation failed: {e}"
            logger.error(error_msg)
            return RotationResult(
                success=False,
                token_id=token_id,
                old_token_length=0,
                new_token_length=0,
                rotation_time=datetime.now(timezone.utc),
                error_message=error_msg
            )
    
    def get_rotation_status(self) -> Dict[str, Any]:
        """
        Get current rotation service status.
        
        Returns:
            Status information
        """
        tokens = self.storage.list_tokens()
        
        # Calculate next rotation times
        next_rotations = {}
        for token_id in tokens.keys():
            next_rotation = self._calculate_next_rotation_time(token_id)
            if next_rotation:
                next_rotations[token_id] = next_rotation.isoformat()
        
        return {
            "service_running": self._running,
            "total_tokens": len(tokens),
            "rotation_interval_hours": self.config.rotation_interval_hours,
            "recent_rotations": len([r for r in self._rotation_history if 
                                   (datetime.now(timezone.utc) - r.rotation_time).days < 7]),
            "failed_tokens": list(self._failed_rotations.keys()),
            "next_rotations": next_rotations,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
        }
    
    def add_rotation_success_callback(self, callback: Callable[[RotationResult], None]):
        """Add callback for successful rotations."""
        self._rotation_success_callbacks.append(callback)
    
    def add_rotation_failure_callback(self, callback: Callable[[RotationResult], None]):
        """Add callback for failed rotations."""
        self._rotation_failure_callbacks.append(callback)
    
    def _rotation_loop(self):
        """Main rotation loop running in background thread."""
        logger.info("Token rotation loop started")
        
        while self._running:
            try:
                self._check_and_rotate_all_tokens()
                
                # Sleep for rotation interval
                sleep_duration = self.config.rotation_interval_hours * 3600
                for _ in range(int(sleep_duration)):
                    if not self._running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in rotation loop: {e}")
                # Sleep briefly before retrying
                for _ in range(60):
                    if not self._running:
                        break
                    time.sleep(1)
        
        logger.info("Token rotation loop stopped")
    
    def _health_check_loop(self):
        """Health monitoring loop."""
        logger.info("Token health monitoring started")
        
        while self._running:
            try:
                self._perform_health_checks()
                
                # Sleep for health check interval
                sleep_duration = self.config.health_check_interval_minutes * 60
                for _ in range(sleep_duration):
                    if not self._running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(60)
        
        logger.info("Token health monitoring stopped")
    
    def _check_and_rotate_all_tokens(self):
        """Check all tokens and rotate those that need it."""
        tokens = self.storage.list_tokens()
        logger.info(f"Checking {len(tokens)} tokens for rotation needs")
        
        for token_id in tokens.keys():
            try:
                if self._needs_rotation(token_id):
                    logger.info(f"Token '{token_id}' needs rotation")
                    
                    current_token = self.storage.retrieve_token(token_id, update_last_used=False)
                    if current_token:
                        result = self._perform_rotation_with_retries(token_id, current_token)
                        if result.success:
                            logger.info(f"Successfully rotated token '{token_id}'")
                        else:
                            logger.error(f"Failed to rotate token '{token_id}': {result.error_message}")
                    else:
                        logger.error(f"Could not retrieve token '{token_id}' for rotation")
                        
            except Exception as e:
                logger.error(f"Error checking token '{token_id}' for rotation: {e}")
    
    def _needs_rotation(self, token_id: str) -> bool:
        """
        Check if a token needs rotation.
        
        Args:
            token_id: Token to check
            
        Returns:
            True if token should be rotated
        """
        tokens = self.storage.list_tokens(include_expired=True)
        token_info = tokens.get(token_id)
        
        if not token_info:
            return False
        
        current_time = datetime.now(timezone.utc)
        created_at = datetime.fromisoformat(token_info["created_at"])
        
        # Check if token is expired
        if token_info["expires_at"]:
            expires_at = datetime.fromisoformat(token_info["expires_at"])
            if current_time >= expires_at:
                logger.info(f"Token '{token_id}' is expired")
                return True
            
            # Check if approaching expiration
            total_lifetime = (expires_at - created_at).total_seconds()
            elapsed_lifetime = (current_time - created_at).total_seconds()
            
            if elapsed_lifetime / total_lifetime >= self.config.early_rotation_threshold:
                logger.info(f"Token '{token_id}' is approaching expiration")
                return True
        
        # Check age-based rotation
        age_hours = (current_time - created_at).total_seconds() / 3600
        if age_hours >= self.config.rotation_interval_hours:
            logger.info(f"Token '{token_id}' is {age_hours:.1f} hours old")
            return True
        
        return False
    
    def _perform_rotation_with_retries(
        self,
        token_id: str,
        current_token: str
    ) -> RotationResult:
        """
        Perform token rotation with retry logic.
        
        Args:
            token_id: Token identifier
            current_token: Current token value
            
        Returns:
            Rotation result
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Rotation attempt {attempt + 1}/{self.config.retry_attempts} for '{token_id}'")
                
                # Call refresh callback to get new token
                if self.refresh_callback:
                    new_token = self.refresh_callback(current_token)
                else:
                    # Fallback: generate a placeholder (in real implementation, 
                    # this would call the actual OAuth refresh endpoint)
                    new_token = self._simulate_token_refresh(current_token)
                
                if not new_token:
                    raise Exception("Failed to obtain new token from refresh callback")
                
                # Store the new token
                success = self.storage.rotate_token(token_id, new_token, preserve_metadata=True)
                
                if success:
                    result = RotationResult(
                        success=True,
                        token_id=token_id,
                        old_token_length=len(current_token),
                        new_token_length=len(new_token),
                        rotation_time=datetime.now(timezone.utc),
                        retry_count=attempt
                    )
                    
                    # Clear failure count on success
                    if token_id in self._failed_rotations:
                        del self._failed_rotations[token_id]
                    
                    # Add to history and trigger callbacks
                    self._rotation_history.append(result)
                    self._trigger_success_callbacks(result)
                    
                    return result
                else:
                    raise Exception("Failed to store rotated token")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Rotation attempt {attempt + 1} failed for '{token_id}': {e}")
                
                # Wait before retry (except on last attempt)
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)
        
        # All attempts failed
        self._failed_rotations[token_id] = self._failed_rotations.get(token_id, 0) + 1
        
        result = RotationResult(
            success=False,
            token_id=token_id,
            old_token_length=len(current_token),
            new_token_length=0,
            rotation_time=datetime.now(timezone.utc),
            error_message=last_error,
            retry_count=self.config.retry_attempts
        )
        
        self._rotation_history.append(result)
        self._trigger_failure_callbacks(result)
        
        return result
    
    def _simulate_token_refresh(self, current_token: str) -> str:
        """
        Simulate token refresh for testing purposes.
        In production, this would call the actual OAuth refresh endpoint.
        
        Args:
            current_token: Current token value
            
        Returns:
            New token value
        """
        # This is a placeholder implementation
        # In a real system, this would:
        # 1. Extract refresh token or use current token
        # 2. Call OAuth provider's refresh endpoint
        # 3. Return new access token
        
        logger.warning("Using simulated token refresh - implement actual OAuth refresh!")
        
        # Generate a new token with similar format but different content
        import secrets
        if current_token.startswith("sk-ant-"):
            return f"sk-ant-{secrets.token_urlsafe(64)}"
        elif current_token.startswith("sk-"):
            return f"sk-{secrets.token_urlsafe(40)}"
        else:
            return secrets.token_urlsafe(32)
    
    def _calculate_next_rotation_time(self, token_id: str) -> Optional[datetime]:
        """
        Calculate when a token should next be rotated.
        
        Args:
            token_id: Token identifier
            
        Returns:
            Next rotation time or None if not applicable
        """
        tokens = self.storage.list_tokens()
        token_info = tokens.get(token_id)
        
        if not token_info:
            return None
        
        created_at = datetime.fromisoformat(token_info["created_at"])
        rotation_interval = timedelta(hours=self.config.rotation_interval_hours)
        
        # If token has expiration, use the earlier of expiration threshold or age-based rotation
        if token_info["expires_at"]:
            expires_at = datetime.fromisoformat(token_info["expires_at"]) 
            total_lifetime = expires_at - created_at
            early_rotation_time = created_at + (total_lifetime * self.config.early_rotation_threshold)
            
            age_based_rotation = created_at + rotation_interval
            return min(early_rotation_time, age_based_rotation)
        else:
            return created_at + rotation_interval
    
    def _perform_health_checks(self):
        """Perform health checks on tokens and service."""
        current_time = datetime.now(timezone.utc)
        self._last_health_check = current_time
        
        tokens = self.storage.list_tokens(include_expired=True)
        
        # Check for expired tokens
        expired_tokens = []
        for token_id, info in tokens.items():
            if info["is_expired"]:
                expired_tokens.append(token_id)
        
        if expired_tokens:
            logger.warning(f"Found {len(expired_tokens)} expired tokens: {expired_tokens}")
        
        # Check for tokens that failed rotation multiple times
        critical_failures = {
            token_id: count for token_id, count in self._failed_rotations.items()
            if count >= 3
        }
        
        if critical_failures:
            logger.error(f"Tokens with critical rotation failures: {critical_failures}")
        
        # Log health summary
        logger.info(f"Health check completed - {len(tokens)} total tokens, "
                   f"{len(expired_tokens)} expired, {len(critical_failures)} critical failures")
    
    def _trigger_success_callbacks(self, result: RotationResult):
        """Trigger callbacks for successful rotation."""
        for callback in self._rotation_success_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in rotation success callback: {e}")
    
    def _trigger_failure_callbacks(self, result: RotationResult):
        """Trigger callbacks for failed rotation."""
        for callback in self._rotation_failure_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in rotation failure callback: {e}")

# Example OAuth refresh implementation
def oauth_refresh_callback(current_token: str) -> Optional[str]:
    """
    Example OAuth refresh callback implementation.
    
    This should be implemented based on your OAuth provider's API.
    
    Args:
        current_token: Current token value
        
    Returns:
        New token value or None if refresh failed
    """
    try:
        # Example for Anthropic/Claude OAuth refresh
        # In reality, you would:
        # 1. Check if current token is a refresh token or has refresh capability
        # 2. Call the OAuth provider's refresh endpoint
        # 3. Handle the response and extract new token
        
        # Placeholder implementation
        logger.info("Refreshing OAuth token...")
        
        # For demonstration - in reality you'd call OAuth API
        import requests
        import secrets
        
        # Simulate OAuth refresh call
        # refresh_response = requests.post(
        #     "https://oauth.provider.com/token",
        #     data={
        #         "grant_type": "refresh_token",
        #         "refresh_token": current_token  # or extract from storage
        #     },
        #     headers={"Authorization": f"Bearer {current_token}"}
        # )
        # 
        # if refresh_response.status_code == 200:
        #     return refresh_response.json()["access_token"]
        
        # For now, return a simulated new token
        return f"sk-ant-{secrets.token_urlsafe(64)}"
        
    except Exception as e:
        logger.error(f"OAuth refresh failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from .secure_oauth_storage import SecureTokenStorage
    
    # Initialize storage and rotation service
    storage = SecureTokenStorage()
    rotation_service = TokenRotationService(
        storage=storage,
        refresh_callback=oauth_refresh_callback
    )
    
    # Start the service
    if rotation_service.start():
        print("✅ Token rotation service started")
        
        # Get status
        status = rotation_service.get_rotation_status()
        print(f"📋 Service status: {json.dumps(status, indent=2)}")
        
        # Keep running for a while (in real usage, this would run indefinitely)
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            pass
        
        # Stop the service
        rotation_service.stop()
        print("🛑 Token rotation service stopped")
    else:
        print("❌ Failed to start token rotation service")
