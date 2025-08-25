"""
Base Repository Pattern Implementation

Provides abstract base repository with:
- Generic CRUD operations
- Async/await support
- Error handling and validation
- Logging and monitoring
- Type safety with generics
"""

import uuid
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type, Generic, TypeVar, Union
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.sql import Select

from ...core.exceptions import ClaudeTIUException
from ...core.logger import get_logger

T = TypeVar('T')  # Generic type for model
logger = get_logger(__name__)


class RepositoryError(ClaudeTIUException):
    """Repository-specific error."""
    
    def __init__(self, message: str, error_code: str = "REPOSITORY_ERROR", details: Optional[Dict] = None):
        super().__init__(message, error_code, details)


class BaseRepository(Generic[T], ABC):
    """Abstract base repository with common CRUD operations and security features."""
    
    def __init__(self, session: AsyncSession, model: Type[T]):
        """
        Initialize base repository.
        
        Args:
            session: AsyncSession instance
            model: SQLAlchemy model class
        """
        self.session = session
        self.model = model
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    async def get_by_id(self, id: Union[uuid.UUID, str, int], load_relationships: bool = False) -> Optional[T]:
        """
        Get entity by ID with optional relationship loading.
        
        Args:
            id: Entity ID
            load_relationships: Whether to eagerly load relationships
            
        Returns:
            Entity instance or None if not found
            
        Raises:
            RepositoryError: If database operation fails
        """
        try:
            query = select(self.model)
            
            # Add relationship loading if requested
            if load_relationships:
                query = self._add_relationship_loading(query)
            
            query = query.where(self.model.id == id)
            
            result = await self.session.execute(query)
            entity = result.scalar_one_or_none()
            
            if entity:
                self.logger.debug(f"Retrieved {self.model.__name__} with ID {id}")
            else:
                self.logger.debug(f"{self.model.__name__} with ID {id} not found")
            
            return entity
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting {self.model.__name__} by ID {id}: {e}")
            raise RepositoryError(
                f"Failed to retrieve {self.model.__name__} by ID",
                "GET_BY_ID_ERROR",
                {"id": str(id), "model": self.model.__name__, "error": str(e)}
            )
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        load_relationships: bool = False
    ) -> List[T]:
        """
        Get all entities with pagination, filtering, and ordering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of field filters
            order_by: Field name to order by
            order_desc: Whether to order in descending order
            load_relationships: Whether to eagerly load relationships
            
        Returns:
            List of entity instances
            
        Raises:
            RepositoryError: If database operation fails
        """
        try:
            query = select(self.model)
            
            # Add relationship loading if requested
            if load_relationships:
                query = self._add_relationship_loading(query)
            
            # Apply filters
            query = self._apply_filters(query, filters)
            
            # Apply ordering
            if order_by and hasattr(self.model, order_by):
                order_field = getattr(self.model, order_by)
                query = query.order_by(desc(order_field) if order_desc else asc(order_field))
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = await self.session.execute(query)
            entities = result.scalars().all()
            
            self.logger.debug(
                f"Retrieved {len(entities)} {self.model.__name__} entities "
                f"(skip={skip}, limit={limit})"
            )
            
            return list(entities)
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise RepositoryError(
                f"Failed to retrieve {self.model.__name__} entities",
                "GET_ALL_ERROR",
                {
                    "model": self.model.__name__,
                    "skip": skip,
                    "limit": limit,
                    "filters": filters,
                    "error": str(e)
                }
            )
    
    async def create(self, **kwargs) -> Optional[T]:
        """
        Create new entity with validation.
        
        Args:
            **kwargs: Entity field values
            
        Returns:
            Created entity instance or None on failure
            
        Raises:
            RepositoryError: If creation fails
        """
        try:
            # Remove None values to avoid overriding defaults
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            entity = self.model(**clean_kwargs)
            self.session.add(entity)
            await self.session.flush()  # Get ID without committing
            await self.session.refresh(entity)
            
            self.logger.info(f"Created {self.model.__name__} with ID {entity.id}")
            return entity
            
        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise RepositoryError(
                f"Constraint violation creating {self.model.__name__}",
                "INTEGRITY_ERROR",
                {"model": self.model.__name__, "data": kwargs, "error": str(e)}
            )
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Error creating {self.model.__name__}: {e}")
            raise RepositoryError(
                f"Failed to create {self.model.__name__}",
                "CREATE_ERROR",
                {"model": self.model.__name__, "data": kwargs, "error": str(e)}
            )
    
    async def update(self, id: Union[uuid.UUID, str, int], **kwargs) -> Optional[T]:
        """
        Update entity with validation.
        
        Args:
            id: Entity ID
            **kwargs: Fields to update
            
        Returns:
            Updated entity instance or None if not found
            
        Raises:
            RepositoryError: If update fails
        """
        try:
            entity = await self.get_by_id(id)
            if not entity:
                self.logger.warning(f"{self.model.__name__} with ID {id} not found for update")
                return None
            
            # Update fields, skipping None values
            updated_fields = []
            for key, value in kwargs.items():
                if value is not None and hasattr(entity, key):
                    old_value = getattr(entity, key)
                    setattr(entity, key, value)
                    updated_fields.append(f"{key}: {old_value} -> {value}")
            
            # Update timestamp if model has it
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.now(timezone.utc)
            
            await self.session.flush()
            await self.session.refresh(entity)
            
            self.logger.info(
                f"Updated {self.model.__name__} with ID {id}. "
                f"Changed fields: {', '.join(updated_fields)}"
            )
            return entity
            
        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(f"Integrity error updating {self.model.__name__} with ID {id}: {e}")
            raise RepositoryError(
                f"Constraint violation updating {self.model.__name__}",
                "INTEGRITY_ERROR",
                {"model": self.model.__name__, "id": str(id), "data": kwargs, "error": str(e)}
            )
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Error updating {self.model.__name__} with ID {id}: {e}")
            raise RepositoryError(
                f"Failed to update {self.model.__name__}",
                "UPDATE_ERROR",
                {"model": self.model.__name__, "id": str(id), "data": kwargs, "error": str(e)}
            )
    
    async def delete(self, id: Union[uuid.UUID, str, int]) -> bool:
        """
        Delete entity with security checks.
        
        Args:
            id: Entity ID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            RepositoryError: If deletion fails
        """
        try:
            entity = await self.get_by_id(id)
            if not entity:
                self.logger.warning(f"{self.model.__name__} with ID {id} not found for deletion")
                return False
            
            await self.session.delete(entity)
            await self.session.flush()
            
            self.logger.info(f"Deleted {self.model.__name__} with ID {id}")
            return True
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Error deleting {self.model.__name__} with ID {id}: {e}")
            raise RepositoryError(
                f"Failed to delete {self.model.__name__}",
                "DELETE_ERROR",
                {"model": self.model.__name__, "id": str(id), "error": str(e)}
            )
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities with optional filters.
        
        Args:
            filters: Dictionary of field filters
            
        Returns:
            Number of matching entities
            
        Raises:
            RepositoryError: If count fails
        """
        try:
            query = select(func.count(self.model.id))
            query = self._apply_filters(query, filters)
            
            result = await self.session.execute(query)
            count = result.scalar() or 0
            
            self.logger.debug(f"Counted {count} {self.model.__name__} entities")
            return count
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error counting {self.model.__name__}: {e}")
            raise RepositoryError(
                f"Failed to count {self.model.__name__} entities",
                "COUNT_ERROR",
                {"model": self.model.__name__, "filters": filters, "error": str(e)}
            )
    
    async def exists(self, id: Union[uuid.UUID, str, int]) -> bool:
        """
        Check if entity exists.
        
        Args:
            id: Entity ID
            
        Returns:
            True if entity exists, False otherwise
        """
        try:
            count = await self.session.scalar(
                select(func.count(self.model.id)).where(self.model.id == id)
            )
            return count > 0
        except SQLAlchemyError as e:
            self.logger.error(f"Error checking existence of {self.model.__name__} with ID {id}: {e}")
            return False
    
    def _apply_filters(self, query: Select, filters: Optional[Dict[str, Any]]) -> Select:
        """
        Apply filters to query.
        
        Args:
            query: SQLAlchemy query
            filters: Dictionary of field filters
            
        Returns:
            Query with filters applied
        """
        if not filters:
            return query
        
        for key, value in filters.items():
            if value is None:
                continue
            
            # Handle special filter keys
            if key.endswith('__in') and isinstance(value, (list, tuple)):
                field_name = key[:-4]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    query = query.where(field.in_(value))
            elif key.endswith('__like') and isinstance(value, str):
                field_name = key[:-6]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    query = query.where(field.like(f"%{value}%"))
            elif key.endswith('__gt'):
                field_name = key[:-4]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    query = query.where(field > value)
            elif key.endswith('__lt'):
                field_name = key[:-4]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    query = query.where(field < value)
            elif key.endswith('__gte'):
                field_name = key[:-5]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    query = query.where(field >= value)
            elif key.endswith('__lte'):
                field_name = key[:-5]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    query = query.where(field <= value)
            elif hasattr(self.model, key):
                # Simple equality filter
                field = getattr(self.model, key)
                query = query.where(field == value)
        
        return query
    
    def _add_relationship_loading(self, query: Select) -> Select:
        """
        Add eager loading for relationships (override in subclasses).
        
        Args:
            query: SQLAlchemy query
            
        Returns:
            Query with relationship loading options
        """
        # Default implementation - override in subclasses to add specific relationships
        return query
    
    async def bulk_create(self, entities_data: List[Dict[str, Any]]) -> List[T]:
        """
        Create multiple entities in bulk.
        
        Args:
            entities_data: List of entity data dictionaries
            
        Returns:
            List of created entities
            
        Raises:
            RepositoryError: If bulk creation fails
        """
        if not entities_data:
            return []
        
        try:
            entities = []
            for data in entities_data:
                clean_data = {k: v for k, v in data.items() if v is not None}
                entity = self.model(**clean_data)
                entities.append(entity)
                self.session.add(entity)
            
            await self.session.flush()
            
            # Refresh all entities
            for entity in entities:
                await self.session.refresh(entity)
            
            self.logger.info(f"Bulk created {len(entities)} {self.model.__name__} entities")
            return entities
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Error in bulk create {self.model.__name__}: {e}")
            raise RepositoryError(
                f"Failed to bulk create {self.model.__name__} entities",
                "BULK_CREATE_ERROR",
                {"model": self.model.__name__, "count": len(entities_data), "error": str(e)}
            )
    
    async def bulk_update(
        self, 
        updates: List[Dict[str, Any]], 
        id_field: str = 'id'
    ) -> int:
        """
        Update multiple entities in bulk.
        
        Args:
            updates: List of update dictionaries (must include ID field)
            id_field: Name of the ID field
            
        Returns:
            Number of updated entities
            
        Raises:
            RepositoryError: If bulk update fails
        """
        if not updates:
            return 0
        
        try:
            updated_count = 0
            
            for update_data in updates:
                if id_field not in update_data:
                    continue
                
                entity_id = update_data.pop(id_field)
                
                # Update timestamp if model has it
                if hasattr(self.model, 'updated_at'):
                    update_data['updated_at'] = datetime.now(timezone.utc)
                
                result = await self.session.execute(
                    update(self.model)
                    .where(self.model.id == entity_id)
                    .values(**update_data)
                )
                updated_count += result.rowcount
            
            await self.session.flush()
            
            self.logger.info(f"Bulk updated {updated_count} {self.model.__name__} entities")
            return updated_count
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Error in bulk update {self.model.__name__}: {e}")
            raise RepositoryError(
                f"Failed to bulk update {self.model.__name__} entities",
                "BULK_UPDATE_ERROR",
                {"model": self.model.__name__, "count": len(updates), "error": str(e)}
            )
    
    async def bulk_delete(self, ids: List[Union[uuid.UUID, str, int]]) -> int:
        """
        Delete multiple entities in bulk.
        
        Args:
            ids: List of entity IDs
            
        Returns:
            Number of deleted entities
            
        Raises:
            RepositoryError: If bulk deletion fails
        """
        if not ids:
            return 0
        
        try:
            result = await self.session.execute(
                delete(self.model).where(self.model.id.in_(ids))
            )
            
            deleted_count = result.rowcount
            await self.session.flush()
            
            self.logger.info(f"Bulk deleted {deleted_count} {self.model.__name__} entities")
            return deleted_count
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Error in bulk delete {self.model.__name__}: {e}")
            raise RepositoryError(
                f"Failed to bulk delete {self.model.__name__} entities",
                "BULK_DELETE_ERROR",
                {"model": self.model.__name__, "count": len(ids), "error": str(e)}
            )
