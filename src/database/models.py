"""
SQLAlchemy Data Models with Security Best Practices
Implements secure data models with proper validation and constraints
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID
from passlib.context import CryptContext
import uuid
import re
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TimestampMixin:
    """Mixin for adding timestamp fields to models"""
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class User(Base, TimestampMixin):
    """User model with comprehensive security features"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    
    # Security fields
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Authentication tracking
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    account_locked_until = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Security tokens
    email_verification_token = Column(String(255), nullable=True)
    password_reset_token = Column(String(255), nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)
    
    # Relationships
    roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    
    # Database constraints
    __table_args__ = (
        Index('ix_users_email_active', 'email', 'is_active'),
        Index('ix_users_username_active', 'username', 'is_active'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        if not email:
            raise ValueError("Email cannot be empty")
        
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            raise ValueError("Invalid email format")
        
        return email.lower().strip()
    
    @validates('username')
    def validate_username(self, key, username):
        """Validate username format"""
        if not username:
            raise ValueError("Username cannot be empty")
        
        if len(username) < 3 or len(username) > 50:
            raise ValueError("Username must be between 3 and 50 characters")
        
        # Allow only alphanumeric characters and underscores
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            raise ValueError("Username can only contain letters, numbers, and underscores")
        
        return username.lower().strip()
    
    def set_password(self, password: str) -> None:
        """Hash and set user password with validation"""
        if not self.validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
        
        self.hashed_password = pwd_context.hash(password)
        self.password_changed_at = datetime.now(timezone.utc)
        logger.info(f"Password updated for user {self.username}")
    
    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def validate_password_strength(password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            return False
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            return False
        
        # Check for at least one digit
        if not re.search(r'\d', password):
            return False
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False
        
        return True
    
    def is_account_locked(self) -> bool:
        """Check if account is locked due to failed login attempts"""
        if self.account_locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.account_locked_until
    
    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock account for specified duration"""
        self.account_locked_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        logger.warning(f"Account locked for user {self.username} for {duration_minutes} minutes")
    
    def unlock_account(self) -> None:
        """Unlock account and reset failed attempts"""
        self.account_locked_until = None
        self.failed_login_attempts = 0
        logger.info(f"Account unlocked for user {self.username}")


class Role(Base, TimestampMixin):
    """Role model for RBAC system"""
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    is_system_role = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    users = relationship("UserRole", back_populates="role")
    permissions = relationship("RolePermission", back_populates="role", cascade="all, delete-orphan")
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate role name"""
        if not name:
            raise ValueError("Role name cannot be empty")
        
        # Convert to uppercase for consistency
        name = name.upper().strip()
        
        # Validate format (only letters, numbers, underscores)
        if not re.match(r'^[A-Z0-9_]+$', name):
            raise ValueError("Role name can only contain uppercase letters, numbers, and underscores")
        
        return name


class Permission(Base, TimestampMixin):
    """Permission model for granular access control"""
    __tablename__ = "permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    resource = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    
    # Relationships
    roles = relationship("RolePermission", back_populates="permission")
    
    # Database constraints
    __table_args__ = (
        Index('ix_permissions_resource_action', 'resource', 'action'),
        UniqueConstraint('resource', 'action', name='uq_permission_resource_action'),
    )
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate permission name format"""
        if not name:
            raise ValueError("Permission name cannot be empty")
        return name.lower().strip()
    
    @validates('resource', 'action')
    def validate_resource_action(self, key, value):
        """Validate resource and action format"""
        if not value:
            raise ValueError(f"{key} cannot be empty")
        return value.lower().strip()


class UserRole(Base, TimestampMixin):
    """Many-to-many relationship between users and roles"""
    __tablename__ = "user_roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role_id = Column(UUID(as_uuid=True), ForeignKey("roles.id"), nullable=False)
    granted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    granted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="roles")
    role = relationship("Role", back_populates="users")
    granter = relationship("User", foreign_keys=[granted_by])
    
    # Database constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'role_id', name='uq_user_role'),
        Index('ix_user_roles_user_id', 'user_id'),
        Index('ix_user_roles_role_id', 'role_id'),
    )
    
    def is_expired(self) -> bool:
        """Check if role assignment has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class RolePermission(Base, TimestampMixin):
    """Many-to-many relationship between roles and permissions"""
    __tablename__ = "role_permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_id = Column(UUID(as_uuid=True), ForeignKey("roles.id"), nullable=False)
    permission_id = Column(UUID(as_uuid=True), ForeignKey("permissions.id"), nullable=False)
    
    # Relationships
    role = relationship("Role", back_populates="permissions")
    permission = relationship("Permission", back_populates="roles")
    
    # Database constraints
    __table_args__ = (
        UniqueConstraint('role_id', 'permission_id', name='uq_role_permission'),
    )


class UserSession(Base, TimestampMixin):
    """User session tracking for security"""
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), unique=True, nullable=True)
    
    # Session metadata
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    location = Column(String(100), nullable=True)
    
    # Session lifecycle
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    # Database constraints
    __table_args__ = (
        Index('ix_user_sessions_user_id_active', 'user_id', 'is_active'),
        Index('ix_user_sessions_expires_at', 'expires_at'),
        Index('ix_user_sessions_session_token', 'session_token'),
    )
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def invalidate(self) -> None:
        """Invalidate the session"""
        self.is_active = False
        logger.info(f"Session invalidated for user {self.user_id}")


class Project(Base, TimestampMixin):
    """Project model with security constraints"""
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Ownership and access control
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    is_archived = Column(Boolean, default=False, nullable=False)
    
    # Project metadata
    project_type = Column(String(50), nullable=False)
    status = Column(String(20), default="active", nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    
    # Database constraints
    __table_args__ = (
        Index('ix_projects_owner_id_status', 'owner_id', 'status'),
        Index('ix_projects_is_public', 'is_public'),
    )
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate project name"""
        if not name:
            raise ValueError("Project name cannot be empty")
        
        if len(name) > 100:
            raise ValueError("Project name cannot exceed 100 characters")
        
        return name.strip()
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate project status"""
        valid_statuses = ['active', 'completed', 'on_hold', 'cancelled']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return status


class Task(Base, TimestampMixin):
    """Task model with security and validation"""
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Task classification
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    assigned_to = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Task status and priority
    status = Column(String(20), default="pending", nullable=False)
    priority = Column(String(10), default="medium", nullable=False)
    
    # Task timing
    due_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="tasks")
    assignee = relationship("User", foreign_keys=[assigned_to])
    
    # Database constraints
    __table_args__ = (
        Index('ix_tasks_project_id_status', 'project_id', 'status'),
        Index('ix_tasks_assigned_to_status', 'assigned_to', 'status'),
        Index('ix_tasks_due_date', 'due_date'),
    )
    
    @validates('title')
    def validate_title(self, key, title):
        """Validate task title"""
        if not title:
            raise ValueError("Task title cannot be empty")
        
        if len(title) > 200:
            raise ValueError("Task title cannot exceed 200 characters")
        
        return title.strip()
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate task status"""
        valid_statuses = ['pending', 'in_progress', 'completed', 'cancelled']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return status
    
    @validates('priority')
    def validate_priority(self, key, priority):
        """Validate task priority"""
        valid_priorities = ['low', 'medium', 'high', 'urgent']
        if priority not in valid_priorities:
            raise ValueError(f"Priority must be one of: {', '.join(valid_priorities)}")
        return priority


class AuditLog(Base, TimestampMixin):
    """Audit log for security and compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    
    # Request metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Audit details
    old_values = Column(Text, nullable=True)  # JSON string
    new_values = Column(Text, nullable=True)  # JSON string
    result = Column(String(20), default="success", nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_audit_logs_user_id_action', 'user_id', 'action'),
        Index('ix_audit_logs_resource_type_id', 'resource_type', 'resource_id'),
        Index('ix_audit_logs_created_at', 'created_at'),
    )