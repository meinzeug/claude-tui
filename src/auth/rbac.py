"""
Role-Based Access Control (RBAC) System
Implements comprehensive permission and role management
"""
from typing import List, Dict, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Enumeration of system resources"""
    USER = "user"
    PROJECT = "project"
    TASK = "task"
    ROLE = "role"
    PERMISSION = "permission"
    SYSTEM = "system"
    API = "api"


class ActionType(Enum):
    """Enumeration of possible actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"
    ADMIN = "admin"


@dataclass
class Permission:
    """Permission data class"""
    name: str
    resource: str
    action: str
    description: str = ""
    
    def __str__(self) -> str:
        return f"{self.resource}.{self.action}"
    
    def __hash__(self) -> int:
        return hash(f"{self.resource}.{self.action}")


@dataclass
class Role:
    """Role data class"""
    name: str
    description: str
    permissions: Set[Permission]
    is_system_role: bool = False
    
    def add_permission(self, permission: Permission) -> None:
        """Add permission to role"""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission) -> None:
        """Remove permission from role"""
        self.permissions.discard(permission)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in self.permissions


class RBACManager:
    """RBAC management system"""
    
    def __init__(self):
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self._initialize_default_permissions()
        self._initialize_default_roles()
    
    def _initialize_default_permissions(self) -> None:
        """Initialize default system permissions"""
        default_permissions = [
            # User management permissions
            ("users.create", ResourceType.USER, ActionType.CREATE, "Create new users"),
            ("users.read", ResourceType.USER, ActionType.READ, "View user information"),
            ("users.update", ResourceType.USER, ActionType.UPDATE, "Update user information"),
            ("users.delete", ResourceType.USER, ActionType.DELETE, "Delete users"),
            ("users.manage", ResourceType.USER, ActionType.MANAGE, "Full user management"),
            
            # Project permissions
            ("projects.create", ResourceType.PROJECT, ActionType.CREATE, "Create new projects"),
            ("projects.read", ResourceType.PROJECT, ActionType.READ, "View projects"),
            ("projects.update", ResourceType.PROJECT, ActionType.UPDATE, "Update projects"),
            ("projects.delete", ResourceType.PROJECT, ActionType.DELETE, "Delete projects"),
            ("projects.manage", ResourceType.PROJECT, ActionType.MANAGE, "Full project management"),
            
            # Task permissions
            ("tasks.create", ResourceType.TASK, ActionType.CREATE, "Create new tasks"),
            ("tasks.read", ResourceType.TASK, ActionType.READ, "View tasks"),
            ("tasks.update", ResourceType.TASK, ActionType.UPDATE, "Update tasks"),
            ("tasks.delete", ResourceType.TASK, ActionType.DELETE, "Delete tasks"),
            ("tasks.execute", ResourceType.TASK, ActionType.EXECUTE, "Execute tasks"),
            
            # Role and permission management
            ("roles.create", ResourceType.ROLE, ActionType.CREATE, "Create roles"),
            ("roles.read", ResourceType.ROLE, ActionType.READ, "View roles"),
            ("roles.update", ResourceType.ROLE, ActionType.UPDATE, "Update roles"),
            ("roles.delete", ResourceType.ROLE, ActionType.DELETE, "Delete roles"),
            ("permissions.read", ResourceType.PERMISSION, ActionType.READ, "View permissions"),
            
            # System administration
            ("system.admin", ResourceType.SYSTEM, ActionType.ADMIN, "System administration"),
            ("system.manage", ResourceType.SYSTEM, ActionType.MANAGE, "System management"),
            ("system.read", ResourceType.SYSTEM, ActionType.READ, "View system information"),
            
            # API access
            ("api.read", ResourceType.API, ActionType.READ, "API read access"),
            ("api.write", ResourceType.API, ActionType.UPDATE, "API write access"),
            ("api.admin", ResourceType.API, ActionType.ADMIN, "API administration"),
        ]
        
        for name, resource, action, description in default_permissions:
            permission = Permission(
                name=name,
                resource=resource.value,
                action=action.value,
                description=description
            )
            self.permissions[name] = permission
            logger.debug(f"Initialized permission: {name}")
    
    def _initialize_default_roles(self) -> None:
        """Initialize default system roles"""
        # Super Admin role - all permissions
        super_admin = Role(
            name="SUPER_ADMIN",
            description="Super administrator with all permissions",
            permissions=set(self.permissions.values()),
            is_system_role=True
        )
        self.roles["SUPER_ADMIN"] = super_admin
        
        # Admin role - most permissions except system admin
        admin_permissions = {
            p for p in self.permissions.values()
            if not p.name.startswith("system.")
        }
        admin = Role(
            name="ADMIN",
            description="Administrator with user and project management",
            permissions=admin_permissions,
            is_system_role=True
        )
        self.roles["ADMIN"] = admin
        
        # Project Manager role
        project_manager_permissions = {
            self.permissions["projects.create"],
            self.permissions["projects.read"],
            self.permissions["projects.update"],
            self.permissions["projects.delete"],
            self.permissions["projects.manage"],
            self.permissions["tasks.create"],
            self.permissions["tasks.read"],
            self.permissions["tasks.update"],
            self.permissions["tasks.delete"],
            self.permissions["tasks.execute"],
            self.permissions["api.read"],
            self.permissions["api.write"],
        }
        project_manager = Role(
            name="PROJECT_MANAGER",
            description="Project manager with project and task permissions",
            permissions=project_manager_permissions,
            is_system_role=True
        )
        self.roles["PROJECT_MANAGER"] = project_manager
        
        # Developer role
        developer_permissions = {
            self.permissions["projects.read"],
            self.permissions["tasks.create"],
            self.permissions["tasks.read"],
            self.permissions["tasks.update"],
            self.permissions["tasks.execute"],
            self.permissions["api.read"],
            self.permissions["api.write"],
        }
        developer = Role(
            name="DEVELOPER",
            description="Developer with task management permissions",
            permissions=developer_permissions,
            is_system_role=True
        )
        self.roles["DEVELOPER"] = developer
        
        # Viewer role - read-only access
        viewer_permissions = {
            self.permissions["projects.read"],
            self.permissions["tasks.read"],
            self.permissions["api.read"],
        }
        viewer = Role(
            name="VIEWER",
            description="Read-only access to projects and tasks",
            permissions=viewer_permissions,
            is_system_role=True
        )
        self.roles["VIEWER"] = viewer
        
        # Guest role - minimal access
        guest_permissions = {
            self.permissions["api.read"],
        }
        guest = Role(
            name="GUEST",
            description="Guest access with minimal permissions",
            permissions=guest_permissions,
            is_system_role=True
        )
        self.roles["GUEST"] = guest
        
        logger.info(f"Initialized {len(self.roles)} default roles")
    
    def create_permission(self, name: str, resource: str, action: str, description: str = "") -> Permission:
        """Create new permission"""
        if name in self.permissions:
            raise ValueError(f"Permission {name} already exists")
        
        permission = Permission(
            name=name,
            resource=resource,
            action=action,
            description=description
        )
        
        self.permissions[name] = permission
        logger.info(f"Created permission: {name}")
        return permission
    
    def create_role(self, name: str, description: str, permissions: List[str] = None) -> Role:
        """Create new role with permissions"""
        if name in self.roles:
            raise ValueError(f"Role {name} already exists")
        
        role_permissions = set()
        if permissions:
            for perm_name in permissions:
                if perm_name in self.permissions:
                    role_permissions.add(self.permissions[perm_name])
                else:
                    logger.warning(f"Permission {perm_name} not found when creating role {name}")
        
        role = Role(
            name=name,
            description=description,
            permissions=role_permissions
        )
        
        self.roles[name] = role
        logger.info(f"Created role: {name} with {len(role_permissions)} permissions")
        return role
    
    def assign_permission_to_role(self, role_name: str, permission_name: str) -> bool:
        """Assign permission to role"""
        if role_name not in self.roles:
            logger.error(f"Role {role_name} not found")
            return False
        
        if permission_name not in self.permissions:
            logger.error(f"Permission {permission_name} not found")
            return False
        
        role = self.roles[role_name]
        permission = self.permissions[permission_name]
        
        role.add_permission(permission)
        logger.info(f"Assigned permission {permission_name} to role {role_name}")
        return True
    
    def remove_permission_from_role(self, role_name: str, permission_name: str) -> bool:
        """Remove permission from role"""
        if role_name not in self.roles:
            logger.error(f"Role {role_name} not found")
            return False
        
        if permission_name not in self.permissions:
            logger.error(f"Permission {permission_name} not found")
            return False
        
        role = self.roles[role_name]
        permission = self.permissions[permission_name]
        
        role.remove_permission(permission)
        logger.info(f"Removed permission {permission_name} from role {role_name}")
        return True
    
    def user_has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has specific permission"""
        return required_permission in user_permissions
    
    def user_has_all_permissions(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """Check if user has all required permissions"""
        return all(perm in user_permissions for perm in required_permissions)
    
    def user_has_any_permission(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """Check if user has any of the required permissions"""
        return any(perm in user_permissions for perm in required_permissions)
    
    def get_role_permissions(self, role_name: str) -> List[str]:
        """Get all permissions for a role"""
        if role_name not in self.roles:
            return []
        
        role = self.roles[role_name]
        return [perm.name for perm in role.permissions]
    
    def get_user_effective_permissions(self, user_roles: List[str]) -> List[str]:
        """Get all effective permissions for user based on roles"""
        effective_permissions = set()
        
        for role_name in user_roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                effective_permissions.update(perm.name for perm in role.permissions)
        
        return list(effective_permissions)
    
    def check_resource_access(self, user_permissions: List[str], resource_type: str, 
                            action: str, resource_owner: str = None, user_id: str = None) -> bool:
        """
        Check if user can perform action on resource type
        Includes ownership and hierarchy checks
        """
        required_permission = f"{resource_type}.{action}"
        
        # Check if user has the specific permission
        if self.user_has_permission(user_permissions, required_permission):
            return True
        
        # Check if user has manage permission for the resource
        manage_permission = f"{resource_type}.manage"
        if self.user_has_permission(user_permissions, manage_permission):
            return True
        
        # Check if user has admin permissions
        if self.user_has_permission(user_permissions, "system.admin"):
            return True
        
        # Check ownership for certain actions
        if resource_owner and user_id and resource_owner == user_id:
            owner_actions = ["read", "update"]
            if action in owner_actions:
                return True
        
        return False
    
    def get_accessible_resources(self, user_permissions: List[str], resource_type: str) -> List[str]:
        """Get list of actions user can perform on resource type"""
        accessible_actions = []
        
        for action in ["create", "read", "update", "delete", "execute", "manage"]:
            if self.check_resource_access(user_permissions, resource_type, action):
                accessible_actions.append(action)
        
        return accessible_actions
    
    def validate_permission_hierarchy(self) -> Dict[str, List[str]]:
        """Validate permission hierarchy and return any issues"""
        issues = {}
        
        # Check for orphaned permissions (not assigned to any role)
        assigned_permissions = set()
        for role in self.roles.values():
            assigned_permissions.update(perm.name for perm in role.permissions)
        
        orphaned = set(self.permissions.keys()) - assigned_permissions
        if orphaned:
            issues["orphaned_permissions"] = list(orphaned)
        
        # Check for roles without permissions
        empty_roles = [name for name, role in self.roles.items() if not role.permissions]
        if empty_roles:
            issues["empty_roles"] = empty_roles
        
        # Check for permission naming consistency
        inconsistent_names = []
        for perm_name, permission in self.permissions.items():
            expected_name = f"{permission.resource}.{permission.action}"
            if perm_name != expected_name:
                inconsistent_names.append(f"{perm_name} (expected: {expected_name})")
        
        if inconsistent_names:
            issues["inconsistent_permission_names"] = inconsistent_names
        
        return issues
    
    def export_rbac_config(self) -> Dict[str, Any]:
        """Export RBAC configuration for backup/migration"""
        return {
            "permissions": {
                name: {
                    "resource": perm.resource,
                    "action": perm.action,
                    "description": perm.description
                }
                for name, perm in self.permissions.items()
            },
            "roles": {
                name: {
                    "description": role.description,
                    "permissions": [perm.name for perm in role.permissions],
                    "is_system_role": role.is_system_role
                }
                for name, role in self.roles.items()
            }
        }
    
    def import_rbac_config(self, config: Dict[str, Any]) -> None:
        """Import RBAC configuration from backup"""
        # Import permissions
        for perm_name, perm_data in config.get("permissions", {}).items():
            if perm_name not in self.permissions:
                self.create_permission(
                    name=perm_name,
                    resource=perm_data["resource"],
                    action=perm_data["action"],
                    description=perm_data.get("description", "")
                )
        
        # Import roles
        for role_name, role_data in config.get("roles", {}).items():
            if role_name not in self.roles:
                self.create_role(
                    name=role_name,
                    description=role_data["description"],
                    permissions=role_data.get("permissions", [])
                )
        
        logger.info("RBAC configuration imported successfully")


# Global RBAC manager instance
rbac_manager = RBACManager()