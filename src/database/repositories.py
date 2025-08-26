"""
Repository Pattern Implementation with Security Best Practices
Provides secure data access layer with proper error handling and validation
"""
from typing import List, Optional, Dict, Any, Type, Generic, TypeVar
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from datetime import datetime, timezone, timedelta
import logging
import uuid

from .models import User, Role, Permission, UserRole, RolePermission, Project, Task, AuditLog, UserSession

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations and security features"""
    
    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model
    
    async def get_by_id(self, id: uuid.UUID) -> Optional[T]:
        """Get entity by ID with security checks"""
        try:
            result = await self.session.execute(
                select(self.model).where(self.model.id == id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} by ID {id}: {e}")
            return None
    
    async def get_all(self, skip: int = 0, limit: int = 100, filters: Dict[str, Any] = None) -> List[T]:
        """Get all entities with pagination and filtering"""
        try:
            query = select(self.model)
            
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key) and value is not None:
                        query = query.where(getattr(self.model, key) == value)
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            return []
    
    async def create(self, **kwargs) -> Optional[T]:
        """Create new entity with validation"""
        try:
            entity = self.model(**kwargs)
            self.session.add(entity)
            await self.session.commit()
            await self.session.refresh(entity)
            logger.info(f"Created {self.model.__name__} with ID {entity.id}")
            return entity
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            return None
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            return None
    
    async def update(self, id: uuid.UUID, **kwargs) -> Optional[T]:
        """Update entity with validation"""
        try:
            entity = await self.get_by_id(id)
            if not entity:
                return None
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            # Update timestamp if model has it
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.now(timezone.utc)
            
            await self.session.commit()
            await self.session.refresh(entity)
            logger.info(f"Updated {self.model.__name__} with ID {id}")
            return entity
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error updating {self.model.__name__} with ID {id}: {e}")
            return None
    
    async def delete(self, id: uuid.UUID) -> bool:
        """Delete entity with security checks"""
        try:
            entity = await self.get_by_id(id)
            if not entity:
                return False
            
            await self.session.delete(entity)
            await self.session.commit()
            logger.info(f"Deleted {self.model.__name__} with ID {id}")
            return True
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error deleting {self.model.__name__} with ID {id}: {e}")
            return False
    
    async def count(self, filters: Dict[str, Any] = None) -> int:
        """Count entities with optional filters"""
        try:
            query = select(func.count(self.model.id))
            
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key) and value is not None:
                        query = query.where(getattr(self.model, key) == value)
            
            result = await self.session.execute(query)
            return result.scalar() or 0
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0


class UserRepository(BaseRepository[User]):
    """User repository with authentication and security features"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        try:
            result = await self.session.execute(
                select(User)
                .options(selectinload(User.roles).selectinload(UserRole.role))
                .where(User.email == email.lower().strip())
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            result = await self.session.execute(
                select(User)
                .options(selectinload(User.roles).selectinload(UserRole.role))
                .where(User.username == username.lower().strip())
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by username {username}: {e}")
            return None
    
    async def create_user(self, email: str, username: str, password: str, 
                         full_name: Optional[str] = None, **kwargs) -> Optional[User]:
        """Create new user with password hashing and validation"""
        try:
            # Check if user already exists
            existing_email = await self.get_by_email(email)
            if existing_email:
                logger.warning(f"User with email {email} already exists")
                return None
            
            existing_username = await self.get_by_username(username)
            if existing_username:
                logger.warning(f"User with username {username} already exists")
                return None
            
            # Create user
            user = User(
                email=email,
                username=username,
                full_name=full_name,
                **kwargs
            )
            user.set_password(password)
            
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            
            logger.info(f"Created user {user.username} with email {user.email}")
            return user
            
        except ValueError as e:
            await self.session.rollback()
            logger.error(f"Validation error creating user: {e}")
            return None
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error creating user: {e}")
            return None
    
    async def authenticate_user(self, identifier: str, password: str, ip_address: str = None) -> Optional[User]:
        """Authenticate user by email/username and password"""
        try:
            # Get user by email or username
            user = await self.get_by_email(identifier)
            if not user:
                user = await self.get_by_username(identifier)
            
            if not user:
                logger.warning(f"Authentication failed: user not found for {identifier}")
                return None
            
            # Check if account is locked
            if user.is_account_locked():
                logger.warning(f"Authentication failed: account locked for {user.username}")
                return None
            
            # Check if account is active
            if not user.is_active:
                logger.warning(f"Authentication failed: account inactive for {user.username}")
                return None
            
            # Verify password
            if not user.verify_password(password):
                # Increment failed attempts
                user.failed_login_attempts += 1
                
                # Lock account after 5 failed attempts
                if user.failed_login_attempts >= 5:
                    user.lock_account(30)  # 30 minutes
                
                await self.session.commit()
                logger.warning(f"Authentication failed: invalid password for {user.username}")
                return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.last_login = datetime.now(timezone.utc)
            await self.session.commit()
            
            logger.info(f"User {user.username} authenticated successfully from {ip_address}")
            return user
            
        except SQLAlchemyError as e:
            logger.error(f"Error authenticating user {identifier}: {e}")
            return None
    
    async def get_user_permissions(self, user_id: uuid.UUID) -> List[str]:
        """Get all permissions for a user through their roles"""
        try:
            result = await self.session.execute(
                select(Permission.name)
                .join(RolePermission)
                .join(Role)
                .join(UserRole)
                .where(
                    and_(
                        UserRole.user_id == user_id,
                        or_(UserRole.expires_at.is_(None), UserRole.expires_at > datetime.now(timezone.utc))
                    )
                )
                .distinct()
            )
            return [perm for perm in result.scalars().all()]
        except SQLAlchemyError as e:
            logger.error(f"Error getting permissions for user {user_id}: {e}")
            return []
    
    async def assign_role(self, user_id: uuid.UUID, role_id: uuid.UUID, 
                         granted_by: uuid.UUID = None, expires_at: datetime = None) -> bool:
        """Assign role to user"""
        try:
            # Check if role assignment already exists
            existing = await self.session.execute(
                select(UserRole).where(
                    and_(UserRole.user_id == user_id, UserRole.role_id == role_id)
                )
            )
            if existing.scalar_one_or_none():
                logger.warning(f"Role {role_id} already assigned to user {user_id}")
                return False
            
            # Create role assignment
            user_role = UserRole(
                user_id=user_id,
                role_id=role_id,
                granted_by=granted_by,
                expires_at=expires_at
            )
            
            self.session.add(user_role)
            await self.session.commit()
            
            logger.info(f"Assigned role {role_id} to user {user_id}")
            return True
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error assigning role {role_id} to user {user_id}: {e}")
            return False
    
    async def revoke_role(self, user_id: uuid.UUID, role_id: uuid.UUID) -> bool:
        """Revoke role from user"""
        try:
            result = await self.session.execute(
                delete(UserRole).where(
                    and_(UserRole.user_id == user_id, UserRole.role_id == role_id)
                )
            )
            
            if result.rowcount > 0:
                await self.session.commit()
                logger.info(f"Revoked role {role_id} from user {user_id}")
                return True
            else:
                logger.warning(f"No role assignment found for user {user_id} and role {role_id}")
                return False
                
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error revoking role {role_id} from user {user_id}: {e}")
            return False
    
    async def change_password(self, user_id: uuid.UUID, old_password: str, new_password: str) -> bool:
        """Change user password with validation"""
        try:
            user = await self.get_by_id(user_id)
            if not user:
                return False
            
            # Verify old password
            if not user.verify_password(old_password):
                logger.warning(f"Password change failed: invalid old password for user {user.username}")
                return False
            
            # Set new password
            user.set_password(new_password)
            await self.session.commit()
            
            logger.info(f"Password changed successfully for user {user.username}")
            return True
            
        except ValueError as e:
            logger.error(f"Password validation error for user {user_id}: {e}")
            return False
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error changing password for user {user_id}: {e}")
            return False


class ProjectRepository(BaseRepository[Project]):
    """Project repository with access control"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Project)
    
    async def get_user_projects(self, user_id: uuid.UUID, include_public: bool = True) -> List[Project]:
        """Get projects accessible to user"""
        try:
            query = select(Project).options(
                selectinload(Project.owner),
                selectinload(Project.tasks)
            )
            
            if include_public:
                query = query.where(
                    or_(
                        Project.owner_id == user_id,
                        Project.is_public == True
                    )
                )
            else:
                query = query.where(Project.owner_id == user_id)
            
            query = query.where(Project.is_archived == False)
            query = query.order_by(Project.created_at.desc())
            
            result = await self.session.execute(query)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting projects for user {user_id}: {e}")
            return []
    
    async def create_project(self, owner_id: uuid.UUID, name: str, description: str = None,
                           project_type: str = "general", is_public: bool = False) -> Optional[Project]:
        """Create project with ownership validation"""
        try:
            project = Project(
                name=name,
                description=description,
                owner_id=owner_id,
                project_type=project_type,
                is_public=is_public
            )
            
            self.session.add(project)
            await self.session.commit()
            await self.session.refresh(project)
            
            logger.info(f"Created project '{name}' for user {owner_id}")
            return project
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error creating project: {e}")
            return None
    
    async def check_project_access(self, project_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Check if user has access to project"""
        try:
            result = await self.session.execute(
                select(Project).where(
                    and_(
                        Project.id == project_id,
                        or_(
                            Project.owner_id == user_id,
                            Project.is_public == True
                        )
                    )
                )
            )
            return result.scalar_one_or_none() is not None
        except SQLAlchemyError as e:
            logger.error(f"Error checking project access: {e}")
            return False


class TaskRepository(BaseRepository[Task]):
    """Task repository with project access control"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Task)
    
    async def get_project_tasks(self, project_id: uuid.UUID, user_id: uuid.UUID = None) -> List[Task]:
        """Get tasks for a project with access control"""
        try:
            # First check project access if user_id provided
            if user_id:
                project_repo = ProjectRepository(self.session)
                if not await project_repo.check_project_access(project_id, user_id):
                    logger.warning(f"User {user_id} denied access to tasks for project {project_id}")
                    return []
            
            result = await self.session.execute(
                select(Task)
                .options(
                    selectinload(Task.assignee),
                    selectinload(Task.project)
                )
                .where(Task.project_id == project_id)
                .order_by(Task.created_at.desc())
            )
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting tasks for project {project_id}: {e}")
            return []
    
    async def get_user_tasks(self, user_id: uuid.UUID, status: str = None) -> List[Task]:
        """Get tasks assigned to user"""
        try:
            query = select(Task).options(
                selectinload(Task.project),
                selectinload(Task.assignee)
            ).where(Task.assigned_to == user_id)
            
            if status:
                query = query.where(Task.status == status)
            
            query = query.order_by(Task.due_date.asc().nulls_last())
            
            result = await self.session.execute(query)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting tasks for user {user_id}: {e}")
            return []


class AuditRepository(BaseRepository[AuditLog]):
    """Audit log repository for security tracking"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AuditLog)
    
    async def log_action(self, user_id: uuid.UUID = None, action: str = None,
                        resource_type: str = None, resource_id: str = None,
                        ip_address: str = None, user_agent: str = None,
                        old_values: str = None, new_values: str = None,
                        result: str = "success", error_message: str = None) -> Optional[AuditLog]:
        """Log audit event"""
        try:
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                old_values=old_values,
                new_values=new_values,
                result=result,
                error_message=error_message
            )
            
            self.session.add(audit_log)
            await self.session.commit()
            return audit_log
            
        except SQLAlchemyError as e:
            logger.error(f"Error logging audit event: {e}")
            return None
    
    async def get_user_activity(self, user_id: uuid.UUID, days: int = 30) -> List[AuditLog]:
        """Get user activity for specified number of days"""
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            result = await self.session.execute(
                select(AuditLog)
                .where(
                    and_(
                        AuditLog.user_id == user_id,
                        AuditLog.created_at >= start_date
                    )
                )
                .order_by(AuditLog.created_at.desc())
            )
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting user activity: {e}")
            return []


class SessionRepository(BaseRepository[UserSession]):
    """User session repository for authentication tracking"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, UserSession)
    
    async def create_session(self, user_id: uuid.UUID, session_token: str,
                           ip_address: str = None, user_agent: str = None,
                           expires_at: datetime = None) -> Optional[UserSession]:
        """Create new user session"""
        try:
            if not expires_at:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
            
            session = UserSession(
                user_id=user_id,
                session_token=session_token,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_at=expires_at
            )
            
            self.session.add(session)
            await self.session.commit()
            await self.session.refresh(session)
            
            logger.info(f"Created session for user {user_id}")
            return session
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error creating session: {e}")
            return None
    
    async def get_active_session(self, session_token: str) -> Optional[UserSession]:
        """Get active session by token"""
        try:
            result = await self.session.execute(
                select(UserSession)
                .options(selectinload(UserSession.user))
                .where(
                    and_(
                        UserSession.session_token == session_token,
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.now(timezone.utc)
                    )
                )
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error getting active session: {e}")
            return None
    
    async def invalidate_user_sessions(self, user_id: uuid.UUID) -> int:
        """Invalidate all user sessions"""
        try:
            result = await self.session.execute(
                update(UserSession)
                .where(UserSession.user_id == user_id)
                .values(is_active=False)
            )
            
            await self.session.commit()
            logger.info(f"Invalidated {result.rowcount} sessions for user {user_id}")
            return result.rowcount
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error invalidating sessions for user {user_id}: {e}")
            return 0


# Repository factory for dependency injection
class RepositoryFactory:
    """Factory for creating repository instances"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._repositories = {}
    
    def get_user_repository(self) -> UserRepository:
        """Get user repository instance"""
        if 'user' not in self._repositories:
            self._repositories['user'] = UserRepository(self.session)
        return self._repositories['user']
    
    def get_project_repository(self) -> ProjectRepository:
        """Get project repository instance"""
        if 'project' not in self._repositories:
            self._repositories['project'] = ProjectRepository(self.session)
        return self._repositories['project']
    
    def get_task_repository(self) -> TaskRepository:
        """Get task repository instance"""
        if 'task' not in self._repositories:
            self._repositories['task'] = TaskRepository(self.session)
        return self._repositories['task']
    
    def get_audit_repository(self) -> AuditRepository:
        """Get audit repository instance"""
        if 'audit' not in self._repositories:
            self._repositories['audit'] = AuditRepository(self.session)
        return self._repositories['audit']
    
    def get_session_repository(self) -> SessionRepository:
        """Get session repository instance"""
        if 'session' not in self._repositories:
            self._repositories['session'] = SessionRepository(self.session)
        return self._repositories['session']