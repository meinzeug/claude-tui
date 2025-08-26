"""
User Service for Authentication System.

Provides high-level user management operations including:
- User CRUD operations
- Authentication and session management
- Role and permission management
- OAuth user integration
- Password reset and security features
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from ..database.session import get_async_session
from ..database.repositories.user_repository import UserRepository
from ..database.models import User
from ..core.exceptions import AuthenticationError, ValidationError
from ..core.logger import get_logger
from .oauth.base import OAuthUserInfo

logger = get_logger(__name__)


class UserService:
    """
    User service providing comprehensive user management.
    
    Features:
    - User authentication and registration
    - OAuth user integration
    - Role and permission management
    - Password security operations
    - Account management
    """
    
    def __init__(self):
        """Initialize user service."""
        self._repository = None
    
    async def _get_repository(self) -> UserRepository:
        """Get user repository with database session."""
        if not self._repository:
            session = await get_async_session()
            self._repository = UserRepository(session)
        return self._repository
    
    async def get_user_by_id(
        self, 
        user_id: str, 
        load_relationships: bool = False
    ) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID (string)
            load_relationships: Whether to load user relationships
            
        Returns:
            User instance or None if not found
        """
        try:
            repository = await self._get_repository()
            uuid_id = uuid.UUID(user_id)
            return await repository.get_by_id(uuid_id, load_relationships)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid user ID format: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {e}")
            return None
    
    async def get_user_by_email(
        self, 
        email: str, 
        load_relationships: bool = False
    ) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: User email
            load_relationships: Whether to load user relationships
            
        Returns:
            User instance or None if not found
        """
        try:
            repository = await self._get_repository()
            return await repository.get_by_email(email, load_relationships)
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None
    
    async def get_user_by_username(
        self, 
        username: str, 
        load_relationships: bool = False
    ) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
            load_relationships: Whether to load user relationships
            
        Returns:
            User instance or None if not found
        """
        try:
            repository = await self._get_repository()
            return await repository.get_by_username(username, load_relationships)
        except Exception as e:
            logger.error(f"Error getting user by username {username}: {e}")
            return None
    
    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        **kwargs
    ) -> Optional[User]:
        """
        Create new user account.
        
        Args:
            email: User email
            username: Username
            password: Plain text password
            full_name: User full name (optional)
            **kwargs: Additional user fields
            
        Returns:
            Created user instance
            
        Raises:
            ValidationError: If user data is invalid
            AuthenticationError: If user creation fails
        """
        try:
            repository = await self._get_repository()
            user = await repository.create_user(
                email=email,
                username=username,
                password=password,
                full_name=full_name,
                **kwargs
            )
            
            if user:
                logger.info(f"Created new user: {user.username} ({user.email})")
            
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise AuthenticationError(f"User creation failed: {str(e)}")
    
    async def authenticate_user(
        self,
        identifier: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate user with email/username and password.
        
        Args:
            identifier: Email or username
            password: Plain text password
            ip_address: Client IP address for logging
            
        Returns:
            User instance if authentication successful
            
        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            repository = await self._get_repository()
            user = await repository.authenticate_user(
                identifier=identifier,
                password=password,
                ip_address=ip_address
            )
            
            if user:
                logger.info(f"User authenticated: {user.username}")
            
            return user
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Get all permissions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of permission names
        """
        try:
            repository = await self._get_repository()
            uuid_id = uuid.UUID(user_id)
            return await repository.get_user_permissions(uuid_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid user ID format: {user_id}")
            return []
        except Exception as e:
            logger.error(f"Error getting permissions for user {user_id}: {e}")
            return []
    
    async def create_oauth_user(
        self,
        oauth_info: OAuthUserInfo,
        default_password: Optional[str] = None
    ) -> Optional[User]:
        """
        Create user from OAuth information.
        
        Args:
            oauth_info: OAuth user information
            default_password: Default password (randomly generated if None)
            
        Returns:
            Created user instance
        """
        try:
            import secrets
            
            # Generate secure password if not provided
            password = default_password or secrets.token_urlsafe(32)
            
            # Try to create username from OAuth data
            username = oauth_info.username
            if not username:
                # Generate from email or name
                if oauth_info.email:
                    username = oauth_info.email.split('@')[0]
                elif oauth_info.name:
                    username = oauth_info.name.lower().replace(' ', '_')
                else:
                    username = f"{oauth_info.provider}_{oauth_info.provider_id}"
            
            # Ensure username is unique
            base_username = username
            counter = 1
            while await self.get_user_by_username(username):
                username = f"{base_username}_{counter}"
                counter += 1
            
            user = await self.create_user(
                email=oauth_info.email,
                username=username,
                password=password,
                full_name=oauth_info.name,
                is_verified=oauth_info.verified,
                # Store OAuth info in metadata (if supported)
                # oauth_provider=oauth_info.provider,
                # oauth_provider_id=oauth_info.provider_id
            )
            
            if user:
                logger.info(f"Created OAuth user: {user.username} via {oauth_info.provider}")
            
            return user
            
        except Exception as e:
            logger.error(f"Error creating OAuth user: {e}")
            raise AuthenticationError(f"OAuth user creation failed: {str(e)}")
    
    async def find_oauth_user(
        self,
        oauth_info: OAuthUserInfo
    ) -> Optional[User]:
        """
        Find existing user by OAuth information.
        
        Args:
            oauth_info: OAuth user information
            
        Returns:
            Existing user instance or None
        """
        try:
            # Try to find by email first
            user = await self.get_user_by_email(oauth_info.email)
            if user:
                logger.debug(f"Found OAuth user by email: {oauth_info.email}")
                return user
            
            # Try to find by username if available
            if oauth_info.username:
                user = await self.get_user_by_username(oauth_info.username)
                if user:
                    logger.debug(f"Found OAuth user by username: {oauth_info.username}")
                    return user
            
            # OAuth provider linking table implementation needed for production
            # Current implementation uses email/username matching as fallback
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding OAuth user: {e}")
            return None
    
    async def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful
        """
        try:
            repository = await self._get_repository()
            uuid_id = uuid.UUID(user_id)
            result = await repository.change_password(
                user_id=uuid_id,
                old_password=old_password,
                new_password=new_password
            )
            
            if result:
                logger.info(f"Password changed for user {user_id}")
            
            return result
            
        except (ValueError, TypeError):
            logger.warning(f"Invalid user ID format: {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error changing password for user {user_id}: {e}")
            raise AuthenticationError(f"Password change failed: {str(e)}")
    
    async def reset_password(
        self,
        user_id: str,
        new_password: str
    ) -> bool:
        """
        Reset user password (admin function).
        
        Args:
            user_id: User ID
            new_password: New password
            
        Returns:
            True if successful
        """
        try:
            repository = await self._get_repository()
            uuid_id = uuid.UUID(user_id)
            result = await repository.reset_password(
                user_id=uuid_id,
                new_password=new_password
            )
            
            if result:
                logger.info(f"Password reset for user {user_id}")
            
            return result
            
        except (ValueError, TypeError):
            logger.warning(f"Invalid user ID format: {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error resetting password for user {user_id}: {e}")
            raise AuthenticationError(f"Password reset failed: {str(e)}")


# Global user service instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get global user service instance."""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
