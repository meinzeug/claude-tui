"""
Authentication and Authorization Decorators for FastAPI
Provides secure route protection with role-based access control
"""
from typing import Optional, List, Callable, Any
from functools import wraps
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from .jwt_auth import jwt_manager
from .rbac import RBACManager
from ..database.repositories import RepositoryFactory
from ..database.models import User
from ..api.dependencies.database import get_db_session

logger = logging.getLogger(__name__)

# Security scheme for FastAPI
security = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization error"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session = Depends(get_db_session)
) -> Optional[User]:
    """Dependency to get current authenticated user"""
    
    if not credentials:
        raise AuthenticationError("Missing authentication credentials")
    
    token = credentials.credentials
    if not token:
        raise AuthenticationError("Invalid token format")
    
    # Verify JWT token
    payload = jwt_manager.verify_token(token)
    if not payload:
        raise AuthenticationError("Invalid or expired token")
    
    user_id = payload.get("user_id")
    if not user_id:
        raise AuthenticationError("Invalid token payload")
    
    # Get user from database
    try:
        repo_factory = RepositoryFactory(session)
        user_repo = repo_factory.get_user_repository()
        user = await user_repo.get_by_id(user_id)
        
        if not user:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("Account is deactivated")
        
        if user.is_account_locked():
            raise AuthenticationError("Account is locked")
        
        # Log user activity
        audit_repo = repo_factory.get_audit_repository()
        await audit_repo.log_action(
            user_id=user.id,
            action="token_validation",
            resource_type="authentication",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        logger.debug(f"User {user.username} authenticated successfully")
        return user
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise AuthenticationError("Authentication failed")


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to get current active user"""
    if not current_user.is_active:
        raise AuthenticationError("Account is deactivated")
    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_active_user)) -> User:
    """Dependency to get current superuser"""
    if not current_user.is_superuser:
        raise AuthorizationError("Superuser access required")
    return current_user


def require_permissions(permissions: List[str], require_all: bool = True):
    """
    Decorator to require specific permissions
    
    Args:
        permissions: List of required permissions
        require_all: If True, user must have ALL permissions. If False, ANY permission is sufficient
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (should be injected by FastAPI)
            current_user = kwargs.get('current_user')
            if not current_user:
                raise AuthenticationError("User context not found")
            
            # Get user permissions
            session = kwargs.get('session')
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session not available"
                )
            
            repo_factory = RepositoryFactory(session)
            user_repo = repo_factory.get_user_repository()
            user_permissions = await user_repo.get_user_permissions(current_user.id)
            
            # Check permissions
            rbac = RBACManager()
            if require_all:
                has_permission = rbac.user_has_all_permissions(user_permissions, permissions)
            else:
                has_permission = rbac.user_has_any_permission(user_permissions, permissions)
            
            if not has_permission:
                logger.warning(f"User {current_user.username} denied access: missing permissions {permissions}")
                raise AuthorizationError(f"Required permissions: {', '.join(permissions)}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_roles(roles: List[str], require_all: bool = False):
    """
    Decorator to require specific roles
    
    Args:
        roles: List of required roles
        require_all: If True, user must have ALL roles. If False, ANY role is sufficient
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise AuthenticationError("User context not found")
            
            # Get user roles
            user_roles = [role.role.name for role in current_user.roles if not role.is_expired()]
            
            # Check roles
            rbac = RBACManager()
            if require_all:
                has_role = all(role in user_roles for role in roles)
            else:
                has_role = any(role in user_roles for role in roles)
            
            if not has_role:
                logger.warning(f"User {current_user.username} denied access: missing roles {roles}")
                raise AuthorizationError(f"Required roles: {', '.join(roles)}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_resource_ownership(resource_param: str = "resource_id", resource_type: str = "project"):
    """
    Decorator to require resource ownership or admin access
    
    Args:
        resource_param: Parameter name containing resource ID
        resource_type: Type of resource (project, task, etc.)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise AuthenticationError("User context not found")
            
            # Superusers bypass ownership checks
            if current_user.is_superuser:
                return await func(*args, **kwargs)
            
            resource_id = kwargs.get(resource_param)
            if not resource_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Resource ID parameter '{resource_param}' not found"
                )
            
            session = kwargs.get('session')
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session not available"
                )
            
            # Check resource ownership based on type
            repo_factory = RepositoryFactory(session)
            
            if resource_type == "project":
                project_repo = repo_factory.get_project_repository()
                has_access = await project_repo.check_project_access(resource_id, current_user.id)
            else:
                # Add other resource types as needed
                logger.warning(f"Unknown resource type for ownership check: {resource_type}")
                has_access = False
            
            if not has_access:
                logger.warning(f"User {current_user.username} denied access to {resource_type} {resource_id}")
                raise AuthorizationError("Access denied: insufficient permissions for this resource")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(max_requests: int = 100, window_minutes: int = 1):
    """
    Decorator for rate limiting (simplified implementation)
    In production, use Redis or similar for distributed rate limiting
    """
    request_counts = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                # If no request object, skip rate limiting
                return await func(*args, **kwargs)
            
            # Use IP address as key (in production, consider user ID)
            client_ip = request.client.host if request.client else "unknown"
            current_time = datetime.now()
            
            # Clean old entries
            cutoff_time = current_time - timedelta(minutes=window_minutes)
            request_counts[client_ip] = [
                req_time for req_time in request_counts.get(client_ip, [])
                if req_time > cutoff_time
            ]
            
            # Check rate limit
            if len(request_counts.get(client_ip, [])) >= max_requests:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            # Record this request
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            request_counts[client_ip].append(current_time)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def audit_action(action: str, resource_type: str):
    """
    Decorator to automatically audit API actions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            current_user = kwargs.get('current_user')
            session = kwargs.get('session')
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
                
                # Log successful action
                if session and current_user:
                    repo_factory = RepositoryFactory(session)
                    audit_repo = repo_factory.get_audit_repository()
                    await audit_repo.log_action(
                        user_id=current_user.id,
                        action=action,
                        resource_type=resource_type,
                        ip_address=request.client.host if request and request.client else None,
                        user_agent=request.headers.get("user-agent") if request else None,
                        result="success"
                    )
                
                return result
                
            except Exception as e:
                # Log failed action
                if session and current_user:
                    repo_factory = RepositoryFactory(session)
                    audit_repo = repo_factory.get_audit_repository()
                    await audit_repo.log_action(
                        user_id=current_user.id,
                        action=action,
                        resource_type=resource_type,
                        ip_address=request.client.host if request and request.client else None,
                        user_agent=request.headers.get("user-agent") if request else None,
                        result="failure",
                        error_message=str(e)
                    )
                
                raise e
        
        return wrapper
    return decorator


def require_verified_email(func: Callable) -> Callable:
    """Decorator to require verified email"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        current_user = kwargs.get('current_user')
        if not current_user:
            raise AuthenticationError("User context not found")
        
        if not current_user.is_verified:
            raise AuthorizationError("Email verification required")
        
        return await func(*args, **kwargs)
    
    return wrapper


# Convenience decorators for common permission combinations
def admin_required(func: Callable) -> Callable:
    """Require admin role"""
    return require_roles(["ADMIN"])(func)


def user_management_required(func: Callable) -> Callable:
    """Require user management permissions"""
    return require_permissions(["users.create", "users.update", "users.delete"])(func)


def project_management_required(func: Callable) -> Callable:
    """Require project management permissions"""
    return require_permissions(["projects.create", "projects.update", "projects.delete"])(func)