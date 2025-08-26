"""
Role-Based Access Control (RBAC) Middleware.

Provides comprehensive RBAC implementation with decorators,
permission checking, and role management for the authentication system.
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass
from fastapi import HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..auth.jwt_service import JWTService, TokenData
from ..core.exceptions import AuthenticationError, ValidationError


class Permission(Enum):
    """System permissions."""
    # User permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"
    
    # Project permissions
    PROJECT_READ = "project:read"
    PROJECT_WRITE = "project:write"
    PROJECT_DELETE = "project:delete"
    PROJECT_ADMIN = "project:admin"
    
    # Task permissions
    TASK_READ = "task:read"
    TASK_WRITE = "task:write"
    TASK_DELETE = "task:delete"
    TASK_ADMIN = "task:admin"
    
    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    
    # Community permissions
    COMMUNITY_READ = "community:read"
    COMMUNITY_WRITE = "community:write"
    COMMUNITY_MODERATE = "community:moderate"
    COMMUNITY_ADMIN = "community:admin"
    
    # AI permissions
    AI_READ = "ai:read"
    AI_WRITE = "ai:write"
    AI_ADMIN = "ai:admin"


class Role(Enum):
    """System roles with associated permissions."""
    VIEWER = "viewer"
    DEVELOPER = "developer"
    PROJECT_MANAGER = "project_manager"
    COMMUNITY_MODERATOR = "community_moderator"
    ADMIN = "admin"
    SUPERUSER = "superuser"


@dataclass
class RoleDefinition:
    """Role definition with permissions."""
    name: str
    permissions: Set[Permission]
    description: str
    inherits_from: Optional[List[str]] = None


class RoleRegistry:
    """Registry for role definitions and permissions."""
    
    def __init__(self):
        self._roles: Dict[str, RoleDefinition] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        
        # Viewer - read-only access
        self.register_role(RoleDefinition(
            name=Role.VIEWER.value,
            permissions={
                Permission.USER_READ,
                Permission.PROJECT_READ,
                Permission.TASK_READ,
                Permission.COMMUNITY_READ,
                Permission.AI_READ
            },
            description="Read-only access to resources"
        ))
        
        # Developer - can create and manage own resources
        self.register_role(RoleDefinition(
            name=Role.DEVELOPER.value,
            permissions={
                Permission.USER_READ, Permission.USER_WRITE,
                Permission.PROJECT_READ, Permission.PROJECT_WRITE,
                Permission.TASK_READ, Permission.TASK_WRITE,
                Permission.COMMUNITY_READ, Permission.COMMUNITY_WRITE,
                Permission.AI_READ, Permission.AI_WRITE
            },
            description="Full access to own resources",
            inherits_from=[Role.VIEWER.value]
        ))
        
        # Project Manager - can manage projects and teams
        self.register_role(RoleDefinition(
            name=Role.PROJECT_MANAGER.value,
            permissions={
                Permission.PROJECT_ADMIN,
                Permission.TASK_ADMIN,
                Permission.USER_READ
            },
            description="Can manage projects and teams",
            inherits_from=[Role.DEVELOPER.value]
        ))
        
        # Community Moderator - can moderate community content
        self.register_role(RoleDefinition(
            name=Role.COMMUNITY_MODERATOR.value,
            permissions={
                Permission.COMMUNITY_MODERATE,
                Permission.COMMUNITY_ADMIN
            },
            description="Can moderate community content",
            inherits_from=[Role.DEVELOPER.value]
        ))
        
        # Admin - full system access except user management
        self.register_role(RoleDefinition(
            name=Role.ADMIN.value,
            permissions={
                Permission.SYSTEM_READ, Permission.SYSTEM_WRITE,
                Permission.USER_ADMIN,
                Permission.PROJECT_ADMIN,
                Permission.TASK_ADMIN,
                Permission.COMMUNITY_ADMIN,
                Permission.AI_ADMIN
            },
            description="Full system administration access",
            inherits_from=[Role.PROJECT_MANAGER.value, Role.COMMUNITY_MODERATOR.value]
        ))
        
        # Superuser - complete system access
        self.register_role(RoleDefinition(
            name=Role.SUPERUSER.value,
            permissions={Permission.SYSTEM_ADMIN},
            description="Complete system access",
            inherits_from=[Role.ADMIN.value]
        ))
    
    def register_role(self, role_definition: RoleDefinition):
        """Register a new role definition."""
        # Resolve inherited permissions
        all_permissions = set(role_definition.permissions)
        
        if role_definition.inherits_from:
            for parent_role in role_definition.inherits_from:
                if parent_role in self._roles:
                    all_permissions.update(self._roles[parent_role].permissions)
        
        # Update role definition with resolved permissions
        role_definition.permissions = all_permissions
        self._roles[role_definition.name] = role_definition
    
    def get_role(self, role_name: str) -> Optional[RoleDefinition]:
        """Get role definition by name."""
        return self._roles.get(role_name)
    
    def get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Get permissions for a role."""
        role = self.get_role(role_name)
        return role.permissions if role else set()
    
    def has_permission(self, role_name: str, permission: Permission) -> bool:
        """Check if role has specific permission."""
        permissions = self.get_role_permissions(role_name)
        return permission in permissions or Permission.SYSTEM_ADMIN in permissions
    
    def list_roles(self) -> List[RoleDefinition]:
        """List all registered roles."""
        return list(self._roles.values())


class RBACMiddleware:
    """RBAC middleware for request authorization."""
    
    def __init__(
        self,
        jwt_service: JWTService,
        role_registry: Optional[RoleRegistry] = None
    ):
        self.jwt_service = jwt_service
        self.role_registry = role_registry or RoleRegistry()
        self.security = HTTPBearer(auto_error=False)
    
    async def get_current_user_token(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> TokenData:
        """Extract and validate JWT token from request."""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required"
            )
        
        try:
            token_data = await self.jwt_service.validate_access_token(credentials.credentials)
            return token_data
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
    
    async def check_permission(
        self,
        token_data: TokenData,
        required_permission: Union[Permission, str],
        resource_owner_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has required permission.
        
        Args:
            token_data: Token data with user information
            required_permission: Required permission
            resource_owner_id: Resource owner ID for ownership check
            
        Returns:
            True if user has permission
        """
        # Convert string to Permission enum
        if isinstance(required_permission, str):
            try:
                required_permission = Permission(required_permission)
            except ValueError:
                return False
        
        # Superuser has all permissions
        if self.role_registry.has_permission(token_data.role, Permission.SYSTEM_ADMIN):
            return True
        
        # Check direct permission in token
        permission_str = required_permission.value
        if permission_str in token_data.permissions:
            return True
        
        # Check role-based permission
        if self.role_registry.has_permission(token_data.role, required_permission):
            return True
        
        # Check resource ownership
        if resource_owner_id and resource_owner_id == token_data.user_id:
            # Owner has read/write access to their own resources
            if required_permission.value.endswith((':read', ':write')):
                return True
        
        return False
    
    def require_permission(
        self,
        permission: Union[Permission, str],
        allow_self: bool = False
    ):
        """
        Decorator to require specific permission.
        
        Args:
            permission: Required permission
            allow_self: Allow access to own resources
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract token data from dependencies
                token_data = None
                for arg in args:
                    if isinstance(arg, TokenData):
                        token_data = arg
                        break
                
                if not token_data:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Check for resource owner ID if allow_self is enabled
                resource_owner_id = None
                if allow_self:
                    # Look for common parameter names
                    for param_name in ['user_id', 'owner_id', 'created_by']:
                        if param_name in kwargs:
                            resource_owner_id = kwargs[param_name]
                            break
                
                # Check permission
                if not await self.check_permission(
                    token_data,
                    permission,
                    resource_owner_id
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission '{permission.value if isinstance(permission, Permission) else permission}' required"
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def require_role(self, required_role: Union[Role, str]):
        """
        Decorator to require specific role.
        
        Args:
            required_role: Required role
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract token data from dependencies
                token_data = None
                for arg in args:
                    if isinstance(arg, TokenData):
                        token_data = arg
                        break
                
                if not token_data:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Convert to string
                required_role_str = required_role.value if isinstance(required_role, Role) else required_role
                
                # Check role
                if token_data.role != required_role_str:
                    # Check if user has admin privileges
                    if not self.role_registry.has_permission(token_data.role, Permission.SYSTEM_ADMIN):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Role '{required_role_str}' required"
                        )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def require_any_role(self, roles: List[Union[Role, str]]):
        """
        Decorator to require any of the specified roles.
        
        Args:
            roles: List of acceptable roles
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                token_data = None
                for arg in args:
                    if isinstance(arg, TokenData):
                        token_data = arg
                        break
                
                if not token_data:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Convert roles to strings
                role_strings = [
                    role.value if isinstance(role, Role) else role
                    for role in roles
                ]
                
                # Check if user has any of the required roles
                if token_data.role not in role_strings:
                    # Check admin privileges
                    if not self.role_registry.has_permission(token_data.role, Permission.SYSTEM_ADMIN):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"One of these roles required: {', '.join(role_strings)}"
                        )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


class PermissionChecker:
    """Utility class for permission checking."""
    
    def __init__(self, role_registry: Optional[RoleRegistry] = None):
        self.role_registry = role_registry or RoleRegistry()
    
    def can_access_resource(
        self,
        token_data: TokenData,
        resource_type: str,
        action: str,
        resource_owner_id: Optional[str] = None
    ) -> bool:
        """
        Check if user can access a resource.
        
        Args:
            token_data: Token data
            resource_type: Resource type (user, project, task, etc.)
            action: Action (read, write, delete, admin)
            resource_owner_id: Resource owner ID
            
        Returns:
            True if access is allowed
        """
        # Construct permission string
        permission_str = f"{resource_type}:{action}"
        
        try:
            permission = Permission(permission_str)
        except ValueError:
            return False
        
        # Check direct permission
        if permission_str in token_data.permissions:
            return True
        
        # Check role-based permission
        if self.role_registry.has_permission(token_data.role, permission):
            return True
        
        # Check ownership for read/write operations
        if resource_owner_id and resource_owner_id == token_data.user_id:
            if action in ['read', 'write']:
                return True
        
        return False
    
    def get_user_permissions(self, token_data: TokenData) -> Set[str]:
        """Get all permissions for a user."""
        permissions = set(token_data.permissions)
        role_permissions = self.role_registry.get_role_permissions(token_data.role)
        permissions.update(perm.value for perm in role_permissions)
        return permissions


# Convenience functions for FastAPI dependencies
def get_role_registry() -> RoleRegistry:
    """Get role registry instance."""
    return RoleRegistry()


def get_permission_checker(
    role_registry: RoleRegistry = Depends(get_role_registry)
) -> PermissionChecker:
    """Get permission checker instance."""
    return PermissionChecker(role_registry)


# Common permission decorators
require_admin = lambda: RBACMiddleware(jwt_service=None).require_role(Role.ADMIN)
require_developer = lambda: RBACMiddleware(jwt_service=None).require_role(Role.DEVELOPER)
require_project_read = lambda: RBACMiddleware(jwt_service=None).require_permission(Permission.PROJECT_READ)
require_project_write = lambda: RBACMiddleware(jwt_service=None).require_permission(Permission.PROJECT_WRITE, allow_self=True)
require_user_admin = lambda: RBACMiddleware(jwt_service=None).require_permission(Permission.USER_ADMIN)