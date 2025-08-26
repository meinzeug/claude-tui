"""
User Repository Implementation

Provides user-specific database operations with:
- Authentication and security features
- Role and permission management
- Account security controls
- Password management
- Session handling
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.sql import Select

from .base import BaseRepository, RepositoryError
from ..models import User, Role, Permission, UserRole, RolePermission
from ...core.logger import get_logger

logger = get_logger(__name__)


class UserRepository(BaseRepository[User]):
    """User repository with authentication and security features."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize user repository.
        
        Args:
            session: AsyncSession instance
        """
        super().__init__(session, User)
    
    def _add_relationship_loading(self, query: Select) -> Select:
        """
        Add eager loading for user relationships.
        
        Args:
            query: SQLAlchemy query
            
        Returns:
            Query with relationship loading options
        """
        return query.options(
            selectinload(User.roles).selectinload(UserRole.role),
            selectinload(User.sessions),
            selectinload(User.projects)
        )
    
    async def get_by_email(self, email: str, load_relationships: bool = False) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: User email address
            load_relationships: Whether to eagerly load relationships
            
        Returns:
            User instance or None if not found
            
        Raises:
            RepositoryError: If database operation fails
        """
        try:
            query = select(User)
            
            if load_relationships:
                query = self._add_relationship_loading(query)
            
            query = query.where(User.email == email.lower().strip())
            
            result = await self.session.execute(query)
            user = result.scalar_one_or_none()
            
            if user:
                self.logger.debug(f"Retrieved user by email: {email}")
            else:
                self.logger.debug(f"User not found by email: {email}")
            
            return user
            
        except Exception as e:
            self.logger.error(f"Error getting user by email {email}: {e}")
            raise RepositoryError(
                "Failed to retrieve user by email",
                "GET_USER_BY_EMAIL_ERROR",
                {"email": email, "error": str(e)}
            )
    
    async def get_by_username(self, username: str, load_relationships: bool = False) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
            load_relationships: Whether to eagerly load relationships
            
        Returns:
            User instance or None if not found
            
        Raises:
            RepositoryError: If database operation fails
        """
        try:
            query = select(User)
            
            if load_relationships:
                query = self._add_relationship_loading(query)
            
            query = query.where(User.username == username.lower().strip())
            
            result = await self.session.execute(query)
            user = result.scalar_one_or_none()
            
            if user:
                self.logger.debug(f"Retrieved user by username: {username}")
            else:
                self.logger.debug(f"User not found by username: {username}")
            
            return user
            
        except Exception as e:
            self.logger.error(f"Error getting user by username {username}: {e}")
            raise RepositoryError(
                "Failed to retrieve user by username",
                "GET_USER_BY_USERNAME_ERROR",
                {"username": username, "error": str(e)}
            )
    
    async def create_user(
        self, 
        email: str, 
        username: str, 
        password: str,
        full_name: Optional[str] = None,
        **kwargs
    ) -> Optional[User]:
        """
        Create new user with password hashing and validation.
        
        Args:
            email: User email address
            username: Username
            password: Plain text password
            full_name: User full name (optional)
            **kwargs: Additional user fields
            
        Returns:
            Created user instance or None on failure
            
        Raises:
            RepositoryError: If user creation fails or user already exists
        """
        try:
            # Check if user already exists
            existing_email = await self.get_by_email(email)
            if existing_email:
                raise RepositoryError(
                    "User with this email already exists",
                    "USER_EMAIL_EXISTS",
                    {"email": email}
                )
            
            existing_username = await self.get_by_username(username)
            if existing_username:
                raise RepositoryError(
                    "User with this username already exists",
                    "USER_USERNAME_EXISTS",
                    {"username": username}
                )
            
            # Create user
            user = User(
                email=email,
                username=username,
                full_name=full_name,
                **kwargs
            )
            user.set_password(password)
            
            self.session.add(user)
            await self.session.flush()
            await self.session.refresh(user)
            
            self.logger.info(f"Created user: {user.username} ({user.email})")
            return user
            
        except RepositoryError:
            await self.session.rollback()
            raise
        except ValueError as e:
            await self.session.rollback()
            self.logger.error(f"Validation error creating user: {e}")
            raise RepositoryError(
                f"User validation failed: {str(e)}",
                "USER_VALIDATION_ERROR",
                {"email": email, "username": username, "error": str(e)}
            )
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Database error creating user: {e}")
            raise RepositoryError(
                "Failed to create user",
                "USER_CREATE_ERROR",
                {"email": email, "username": username, "error": str(e)}
            )
    
    async def authenticate_user(
        self, 
        identifier: str, 
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate user by email/username and password.
        
        Args:
            identifier: Email or username
            password: Plain text password
            ip_address: Client IP address for logging
            
        Returns:
            User instance if authentication successful, None otherwise
            
        Raises:
            RepositoryError: If authentication process fails
        """
        try:
            # Get user by email or username
            user = await self.get_by_email(identifier)
            if not user:
                user = await self.get_by_username(identifier)
            
            if not user:
                self.logger.warning(
                    f"Authentication failed: user not found",
                    extra={"identifier": identifier, "ip_address": ip_address}
                )
                return None
            
            # Check if account is locked
            if user.is_account_locked():
                self.logger.warning(
                    f"Authentication failed: account locked",
                    extra={"user_id": str(user.id), "username": user.username, "ip_address": ip_address}
                )
                return None
            
            # Check if account is active
            if not user.is_active:
                self.logger.warning(
                    f"Authentication failed: account inactive",
                    extra={"user_id": str(user.id), "username": user.username, "ip_address": ip_address}
                )
                return None
            
            # Verify password
            if not user.verify_password(password):
                # Increment failed attempts
                user.failed_login_attempts += 1
                
                # Lock account after 5 failed attempts
                if user.failed_login_attempts >= 5:
                    user.lock_account(30)  # 30 minutes
                    self.logger.warning(
                        f"Account locked due to failed login attempts",
                        extra={"user_id": str(user.id), "username": user.username, "ip_address": ip_address}
                    )
                
                await self.session.flush()
                
                self.logger.warning(
                    f"Authentication failed: invalid password",
                    extra={
                        "user_id": str(user.id), 
                        "username": user.username, 
                        "failed_attempts": user.failed_login_attempts,
                        "ip_address": ip_address
                    }
                )
                return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.last_login = datetime.now(timezone.utc)
            await self.session.flush()
            
            self.logger.info(
                f"User authenticated successfully",
                extra={"user_id": str(user.id), "username": user.username, "ip_address": ip_address}
            )
            return user
            
        except Exception as e:
            self.logger.error(f"Error authenticating user {identifier}: {e}")
            raise RepositoryError(
                "Authentication process failed",
                "USER_AUTHENTICATION_ERROR",
                {"identifier": identifier, "ip_address": ip_address, "error": str(e)}
            )
    
    async def get_user_permissions(self, user_id: uuid.UUID) -> List[str]:
        """
        Get all permissions for a user through their roles.
        
        Args:
            user_id: User ID
            
        Returns:
            List of permission names
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            result = await self.session.execute(
                select(Permission.name)
                .join(RolePermission)
                .join(Role)
                .join(UserRole)
                .where(
                    and_(
                        UserRole.user_id == user_id,
                        or_(
                            UserRole.expires_at.is_(None), 
                            UserRole.expires_at > datetime.now(timezone.utc)
                        )
                    )
                )
                .distinct()
            )
            
            permissions = [perm for perm in result.scalars().all()]
            
            self.logger.debug(f"Retrieved {len(permissions)} permissions for user {user_id}")
            return permissions
            
        except Exception as e:
            self.logger.error(f"Error getting permissions for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to retrieve user permissions",
                "GET_USER_PERMISSIONS_ERROR",
                {"user_id": str(user_id), "error": str(e)}
            )
    
    async def get_user_roles(self, user_id: uuid.UUID, active_only: bool = True) -> List[Role]:
        """
        Get all roles for a user.
        
        Args:
            user_id: User ID
            active_only: Whether to return only non-expired roles
            
        Returns:
            List of Role instances
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            query = (
                select(Role)
                .join(UserRole)
                .where(UserRole.user_id == user_id)
            )
            
            if active_only:
                query = query.where(
                    or_(
                        UserRole.expires_at.is_(None),
                        UserRole.expires_at > datetime.now(timezone.utc)
                    )
                )
            
            result = await self.session.execute(query)
            roles = result.scalars().all()
            
            self.logger.debug(f"Retrieved {len(roles)} roles for user {user_id}")
            return list(roles)
            
        except Exception as e:
            self.logger.error(f"Error getting roles for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to retrieve user roles",
                "GET_USER_ROLES_ERROR",
                {"user_id": str(user_id), "error": str(e)}
            )
    
    async def assign_role(
        self, 
        user_id: uuid.UUID, 
        role_id: uuid.UUID,
        granted_by: Optional[uuid.UUID] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Assign role to user.
        
        Args:
            user_id: User ID
            role_id: Role ID
            granted_by: ID of user granting the role
            expires_at: When the role assignment expires
            
        Returns:
            True if role assigned successfully, False if already assigned
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            # Check if role assignment already exists
            existing = await self.session.execute(
                select(UserRole).where(
                    and_(UserRole.user_id == user_id, UserRole.role_id == role_id)
                )
            )
            
            if existing.scalar_one_or_none():
                self.logger.warning(f"Role {role_id} already assigned to user {user_id}")
                return False
            
            # Create role assignment
            user_role = UserRole(
                user_id=user_id,
                role_id=role_id,
                granted_by=granted_by,
                expires_at=expires_at
            )
            
            self.session.add(user_role)
            await self.session.flush()
            
            self.logger.info(f"Assigned role {role_id} to user {user_id}")
            return True
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error assigning role {role_id} to user {user_id}: {e}")
            raise RepositoryError(
                "Failed to assign role to user",
                "ASSIGN_ROLE_ERROR",
                {"user_id": str(user_id), "role_id": str(role_id), "error": str(e)}
            )
    
    async def revoke_role(self, user_id: uuid.UUID, role_id: uuid.UUID) -> bool:
        """
        Revoke role from user.
        
        Args:
            user_id: User ID
            role_id: Role ID
            
        Returns:
            True if role revoked, False if assignment not found
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            result = await self.session.execute(
                delete(UserRole).where(
                    and_(UserRole.user_id == user_id, UserRole.role_id == role_id)
                )
            )
            
            if result.rowcount > 0:
                await self.session.flush()
                self.logger.info(f"Revoked role {role_id} from user {user_id}")
                return True
            else:
                self.logger.warning(f"No role assignment found for user {user_id} and role {role_id}")
                return False
                
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error revoking role {role_id} from user {user_id}: {e}")
            raise RepositoryError(
                "Failed to revoke role from user",
                "REVOKE_ROLE_ERROR",
                {"user_id": str(user_id), "role_id": str(role_id), "error": str(e)}
            )
    
    async def change_password(
        self, 
        user_id: uuid.UUID, 
        old_password: str, 
        new_password: str
    ) -> bool:
        """
        Change user password with validation.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully, False if old password is invalid
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            user = await self.get_by_id(user_id)
            if not user:
                raise RepositoryError(
                    "User not found",
                    "USER_NOT_FOUND",
                    {"user_id": str(user_id)}
                )
            
            # Verify old password
            if not user.verify_password(old_password):
                self.logger.warning(
                    f"Password change failed: invalid old password",
                    extra={"user_id": str(user_id), "username": user.username}
                )
                return False
            
            # Set new password
            user.set_password(new_password)
            await self.session.flush()
            
            self.logger.info(
                f"Password changed successfully",
                extra={"user_id": str(user_id), "username": user.username}
            )
            return True
            
        except RepositoryError:
            raise
        except ValueError as e:
            self.logger.error(f"Password validation error for user {user_id}: {e}")
            raise RepositoryError(
                f"Password validation failed: {str(e)}",
                "PASSWORD_VALIDATION_ERROR",
                {"user_id": str(user_id), "error": str(e)}
            )
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error changing password for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to change password",
                "CHANGE_PASSWORD_ERROR",
                {"user_id": str(user_id), "error": str(e)}
            )
    
    async def reset_password(self, user_id: uuid.UUID, new_password: str) -> bool:
        """
        Reset user password (admin function).
        
        Args:
            user_id: User ID
            new_password: New password
            
        Returns:
            True if password reset successfully
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            user = await self.get_by_id(user_id)
            if not user:
                raise RepositoryError(
                    "User not found",
                    "USER_NOT_FOUND",
                    {"user_id": str(user_id)}
                )
            
            # Set new password
            user.set_password(new_password)
            
            # Reset failed login attempts
            user.failed_login_attempts = 0
            user.unlock_account()
            
            await self.session.flush()
            
            self.logger.info(
                f"Password reset successfully",
                extra={"user_id": str(user_id), "username": user.username}
            )
            return True
            
        except RepositoryError:
            raise
        except ValueError as e:
            self.logger.error(f"Password validation error for user {user_id}: {e}")
            raise RepositoryError(
                f"Password validation failed: {str(e)}",
                "PASSWORD_VALIDATION_ERROR",
                {"user_id": str(user_id), "error": str(e)}
            )
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error resetting password for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to reset password",
                "RESET_PASSWORD_ERROR",
                {"user_id": str(user_id), "error": str(e)}
            )
    
    async def get_active_users(self, limit: int = 100) -> List[User]:
        """
        Get list of active users.
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            List of active User instances
        """
        return await self.get_all(
            limit=limit,
            filters={'is_active': True},
            order_by='last_login',
            order_desc=True
        )
    
    async def search_users(self, query: str, limit: int = 50) -> List[User]:
        """
        Search users by username, email, or full name.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching User instances
        """
        return await self.get_all(
            limit=limit,
            filters={
                'username__like': query,
                'email__like': query,
                'full_name__like': query
            }
        )
