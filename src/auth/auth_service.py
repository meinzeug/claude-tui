"""
Comprehensive Authentication Service.

Provides unified authentication interface including:
- Traditional email/password authentication
- OAuth authentication (GitHub, Google)
- JWT token management and refresh
- Session management
- Security headers and validation
- Claude Code and Claude Flow integration
"""

import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone

from .user_service import get_user_service, UserService
from .jwt_service import JWTService
from .session_manager import SessionManager
from .oauth_manager import get_oauth_manager, OAuthManager
from .audit_logger import get_audit_logger, SecurityAuditLogger
from ..database.models import User
from ..core.exceptions import AuthenticationError, ValidationError
from ..core.logger import get_logger

logger = get_logger(__name__)


class AuthenticationService:
    """
    Comprehensive authentication service.
    
    Features:
    - Multi-provider authentication
    - JWT token management
    - Session management
    - Security auditing
    - Integration with Claude services
    """
    
    def __init__(
        self,
        secret_key: str,
        user_service: Optional[UserService] = None,
        jwt_service: Optional[JWTService] = None,
        session_manager: Optional[SessionManager] = None,
        oauth_manager: Optional[OAuthManager] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
        oauth_config: Optional[Dict[str, Dict]] = None
    ):
        """Initialize authentication service."""
        self.user_service = user_service or get_user_service()
        self.jwt_service = jwt_service or JWTService(secret_key=secret_key)
        self.session_manager = session_manager or SessionManager()
        self.oauth_manager = oauth_manager or get_oauth_manager(
            jwt_service=self.jwt_service,
            session_manager=self.session_manager,
            providers_config=oauth_config
        )
        self.audit_logger = audit_logger or get_audit_logger()
    
    async def authenticate_user(
        self,
        identifier: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate user with email/username and password.
        
        Args:
            identifier: Email or username
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Authentication result with tokens and user info
            
        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            # Authenticate user
            user = await self.user_service.authenticate_user(
                identifier=identifier,
                password=password,
                ip_address=ip_address
            )
            
            if not user:
                # Log failed attempt
                await self.audit_logger.log_authentication(
                    event_type="login_failed",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    message=f"Authentication failed for {identifier}"
                )
                raise AuthenticationError("Invalid credentials")
            
            # Create session
            session_data = await self.session_manager.create_session(
                user_id=str(user.id),
                username=user.username,
                role=getattr(user, 'role', 'user'),
                ip_address=ip_address or 'unknown',
                user_agent=user_agent or 'unknown'
            )
            
            # Get user permissions
            permissions = await self.user_service.get_user_permissions(str(user.id))
            
            # Create JWT tokens
            token_pair = await self.jwt_service.create_token_pair(
                user=user,
                session_id=session_data.session_id,
                permissions=permissions,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Log successful authentication
            await self.audit_logger.log_authentication(
                event_type="login_success",
                user_id=str(user.id),
                username=user.username,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_data.session_id,
                success=True
            )
            
            return {
                'user': {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'full_name': getattr(user, 'full_name', None),
                    'is_verified': getattr(user, 'is_verified', False),
                    'permissions': permissions
                },
                'tokens': {
                    'access_token': token_pair.access_token,
                    'refresh_token': token_pair.refresh_token,
                    'token_type': token_pair.token_type,
                    'expires_in': token_pair.expires_in
                },
                'session': {
                    'session_id': session_data.session_id,
                    'expires_at': session_data.expires_at.isoformat()
                }
            }
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Authentication service error: {e}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")
    
    async def refresh_token(
        self,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            New token pair
        """
        try:
            # Refresh tokens
            token_pair = await self.jwt_service.refresh_access_token(
                refresh_token=refresh_token,
                user_service=self.user_service
            )
            
            # Log token refresh
            await self.audit_logger.log_authentication(
                event_type="token_refreshed",
                user_id=token_pair.user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=token_pair.session_id,
                success=True
            )
            
            return {
                'tokens': {
                    'access_token': token_pair.access_token,
                    'refresh_token': token_pair.refresh_token,
                    'token_type': token_pair.token_type,
                    'expires_in': token_pair.expires_in
                }
            }
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            # Log failed refresh
            await self.audit_logger.log_authentication(
                event_type="token_refresh_failed",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                message=str(e)
            )
            raise AuthenticationError(f"Token refresh failed: {str(e)}")
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT access token.
        
        Args:
            token: JWT access token
            
        Returns:
            Token validation result with user info
        """
        try:
            # Validate token
            token_data = await self.jwt_service.validate_access_token(token)
            
            # Get user information
            user = await self.user_service.get_user_by_id(token_data.user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            if not user.is_active:
                raise AuthenticationError("User account is inactive")
            
            # Validate session
            session_data = await self.session_manager.get_session(token_data.session_id)
            if not session_data:
                raise AuthenticationError("Session not found or expired")
            
            return {
                'valid': True,
                'user': {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'permissions': token_data.permissions
                },
                'token_data': token_data.dict(),
                'session': session_data.dict()
            }
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise AuthenticationError(f"Token validation failed: {str(e)}")
    
    async def logout(
        self,
        token: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Logout user by revoking tokens and invalidating session.
        
        Args:
            token: JWT access token
            session_id: Session ID (optional)
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if logout successful
        """
        try:
            # Validate token to get user info
            token_data = await self.jwt_service.validate_access_token(token)
            
            # Revoke access token
            await self.jwt_service.revoke_token(token)
            
            # Invalidate session
            if session_id or token_data.session_id:
                session_to_invalidate = session_id or token_data.session_id
                await self.session_manager.invalidate_session(session_to_invalidate)
            
            # Log logout
            await self.audit_logger.log_authentication(
                event_type="logout",
                user_id=token_data.user_id,
                username=token_data.username,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=token_data.session_id,
                success=True
            )
            
            logger.info(f"User {token_data.username} logged out successfully")
            return True
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    async def logout_all_sessions(
        self,
        user_id: str,
        current_session_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> int:
        """
        Logout user from all sessions.
        
        Args:
            user_id: User ID
            current_session_id: Current session to exclude (optional)
            ip_address: Client IP address
            
        Returns:
            Number of sessions invalidated
        """
        try:
            # Revoke all user tokens
            revoked_tokens = await self.jwt_service.revoke_all_user_tokens(user_id)
            
            # Invalidate all user sessions
            invalidated_sessions = await self.session_manager.invalidate_user_sessions(
                user_id=user_id,
                exclude_session_id=current_session_id
            )
            
            # Log logout all
            await self.audit_logger.log_authentication(
                event_type="logout_all",
                user_id=user_id,
                ip_address=ip_address,
                success=True,
                message=f"Invalidated {invalidated_sessions} sessions"
            )
            
            logger.info(f"User {user_id} logged out from {invalidated_sessions} sessions")
            return invalidated_sessions
            
        except Exception as e:
            logger.error(f"Logout all sessions error: {e}")
            return 0
    
    async def register_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register new user account.
        
        Args:
            email: User email
            username: Username
            password: Plain text password
            full_name: User full name (optional)
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Registration result with user info
        """
        try:
            # Create user
            user = await self.user_service.create_user(
                email=email,
                username=username,
                password=password,
                full_name=full_name
            )
            
            if not user:
                raise AuthenticationError("User registration failed")
            
            # Log successful registration
            await self.audit_logger.log_authentication(
                event_type="registration_success",
                user_id=str(user.id),
                username=user.username,
                ip_address=ip_address,
                user_agent=user_agent,
                success=True
            )
            
            return {
                'user': {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'is_verified': user.is_verified
                },
                'message': 'User registered successfully'
            }
            
        except AuthenticationError:
            # Log failed registration
            await self.audit_logger.log_authentication(
                event_type="registration_failed",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                message=f"Registration failed for {username}"
            )
            raise
        except Exception as e:
            logger.error(f"User registration error: {e}")
            raise AuthenticationError(f"Registration failed: {str(e)}")
    
    def get_oauth_authorization_url(
        self,
        provider: str,
        redirect_after_auth: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get OAuth authorization URL for provider.
        
        Args:
            provider: OAuth provider name
            redirect_after_auth: URL to redirect after auth
            
        Returns:
            Authorization URL and state
        """
        try:
            auth_url, state = self.oauth_manager.generate_authorization_url(
                provider_name=provider,
                redirect_after_auth=redirect_after_auth
            )
            
            return {
                'authorization_url': auth_url,
                'state': state,
                'provider': provider
            }
            
        except Exception as e:
            logger.error(f"OAuth URL generation error: {e}")
            raise AuthenticationError(f"OAuth URL generation failed: {str(e)}")
    
    async def handle_oauth_callback(
        self,
        provider: str,
        code: str,
        state: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle OAuth callback and complete authentication.
        
        Args:
            provider: OAuth provider name
            code: Authorization code
            state: State token
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            OAuth authentication result
        """
        try:
            result = await self.oauth_manager.handle_callback(
                provider_name=provider,
                code=code,
                state=state,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Log OAuth success
            await self.audit_logger.log_authentication(
                event_type="oauth_login_success",
                user_id=result['user']['id'],
                username=result['user']['username'],
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=result['session']['session_id'],
                success=True,
                message=f"OAuth login via {provider}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            # Log OAuth failure
            await self.audit_logger.log_authentication(
                event_type="oauth_login_failed",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                message=f"OAuth login failed via {provider}: {str(e)}"
            )
            raise AuthenticationError(f"OAuth authentication failed: {str(e)}")
    
    def get_security_headers(self) -> Dict[str, str]:
        """
        Get security headers for HTTP responses.
        
        Returns:
            Dictionary of security headers
        """
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get active sessions for user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active sessions
        """
        try:
            sessions = await self.session_manager.get_user_sessions(user_id)
            return [session.dict() for session in sessions]
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    async def cleanup_expired_tokens_and_sessions(self) -> Dict[str, int]:
        """
        Cleanup expired tokens and sessions.
        
        Returns:
            Cleanup statistics
        """
        try:
            # Run cleanup tasks in parallel
            token_cleanup, session_cleanup, oauth_cleanup = await asyncio.gather(
                self.jwt_service.cleanup_expired_tokens(),
                self.session_manager.cleanup_expired_sessions(),
                asyncio.create_task(self._cleanup_oauth_states()),
                return_exceptions=True
            )
            
            return {
                'tokens_cleaned': token_cleanup if isinstance(token_cleanup, int) else 0,
                'sessions_cleaned': session_cleanup if isinstance(session_cleanup, int) else 0,
                'oauth_states_cleaned': oauth_cleanup if isinstance(oauth_cleanup, int) else 0
            }
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {'tokens_cleaned': 0, 'sessions_cleaned': 0, 'oauth_states_cleaned': 0}
    
    async def _cleanup_oauth_states(self) -> int:
        """Cleanup expired OAuth states."""
        try:
            self.oauth_manager.cleanup_expired_states()
            return 1
        except Exception:
            return 0


# Global authentication service instance
_auth_service: Optional[AuthenticationService] = None


def get_auth_service(
    secret_key: Optional[str] = None,
    oauth_config: Optional[Dict[str, Dict]] = None
) -> AuthenticationService:
    """Get global authentication service instance."""
    global _auth_service
    if _auth_service is None:
        if not secret_key:
            from ..core.config import get_settings
            settings = get_settings()
            secret_key = settings.SECRET_KEY
        
        _auth_service = AuthenticationService(
            secret_key=secret_key,
            oauth_config=oauth_config
        )
    
    return _auth_service
