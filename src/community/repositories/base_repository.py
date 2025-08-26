"""
Base Repository - Abstract base class for data repositories.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

T = TypeVar('T')


class BaseRepository(Generic[T], ABC):
    """Abstract base repository class."""
    
    def __init__(self, db: AsyncSession, model_class: Type[T]):
        """
        Initialize repository.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
        """
        self.db = db
        self.model_class = model_class
    
    async def create(self, **kwargs) -> T:
        """
        Create a new record.
        
        Args:
            **kwargs: Model fields
            
        Returns:
            Created model instance
        """
        instance = self.model_class(**kwargs)
        self.db.add(instance)
        await self.db.commit()
        await self.db.refresh(instance)
        return instance
    
    async def get_by_id(self, id: UUID, load_relationships: Optional[List[str]] = None) -> Optional[T]:
        """
        Get record by ID.
        
        Args:
            id: Record ID
            load_relationships: Relationships to eager load
            
        Returns:
            Model instance or None
        """
        query = select(self.model_class).where(self.model_class.id == id)
        
        if load_relationships:
            for relationship in load_relationships:
                query = query.options(selectinload(getattr(self.model_class, relationship)))
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        load_relationships: Optional[List[str]] = None
    ) -> List[T]:
        """
        Get all records with optional filtering and pagination.
        
        Args:
            limit: Maximum number of records
            offset: Number of records to skip
            filters: Filter conditions
            order_by: Order by field
            load_relationships: Relationships to eager load
            
        Returns:
            List of model instances
        """
        query = select(self.model_class)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.where(getattr(self.model_class, field) == value)
        
        # Apply ordering
        if order_by and hasattr(self.model_class, order_by):
            query = query.order_by(getattr(self.model_class, order_by))
        
        # Apply pagination
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        # Load relationships
        if load_relationships:
            for relationship in load_relationships:
                query = query.options(selectinload(getattr(self.model_class, relationship)))
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def update(self, id: UUID, **kwargs) -> Optional[T]:
        """
        Update record by ID.
        
        Args:
            id: Record ID
            **kwargs: Fields to update
            
        Returns:
            Updated model instance or None
        """
        query = update(self.model_class).where(self.model_class.id == id).values(**kwargs)
        await self.db.execute(query)
        await self.db.commit()
        
        return await self.get_by_id(id)
    
    async def delete(self, id: UUID) -> bool:
        """
        Delete record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        query = delete(self.model_class).where(self.model_class.id == id)
        result = await self.db.execute(query)
        await self.db.commit()
        
        return result.rowcount > 0
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records with optional filters.
        
        Args:
            filters: Filter conditions
            
        Returns:
            Number of matching records
        """
        query = select(self.model_class)
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.where(getattr(self.model_class, field) == value)
        
        result = await self.db.execute(query)
        return len(result.scalars().all())
    
    async def exists(self, **kwargs) -> bool:
        """
        Check if record exists.
        
        Args:
            **kwargs: Filter conditions
            
        Returns:
            True if record exists
        """
        query = select(self.model_class)
        
        for field, value in kwargs.items():
            if hasattr(self.model_class, field):
                query = query.where(getattr(self.model_class, field) == value)
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none() is not None
    
    async def bulk_create(self, records: List[Dict[str, Any]]) -> List[T]:
        """
        Create multiple records in bulk.
        
        Args:
            records: List of record data
            
        Returns:
            List of created instances
        """
        instances = [self.model_class(**record) for record in records]
        self.db.add_all(instances)
        await self.db.commit()
        
        for instance in instances:
            await self.db.refresh(instance)
        
        return instances
    
    async def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """
        Update multiple records in bulk.
        
        Args:
            updates: List of update data with 'id' field
            
        Returns:
            Number of updated records
        """
        updated_count = 0
        
        for update_data in updates:
            record_id = update_data.pop('id')
            query = update(self.model_class).where(
                self.model_class.id == record_id
            ).values(**update_data)
            
            result = await self.db.execute(query)
            updated_count += result.rowcount
        
        await self.db.commit()
        return updated_count
    
    async def get_by_field(self, field: str, value: Any) -> Optional[T]:
        """
        Get record by specific field.
        
        Args:
            field: Field name
            value: Field value
            
        Returns:
            Model instance or None
        """
        if not hasattr(self.model_class, field):
            return None
        
        query = select(self.model_class).where(getattr(self.model_class, field) == value)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_many_by_field(
        self,
        field: str,
        values: List[Any],
        load_relationships: Optional[List[str]] = None
    ) -> List[T]:
        """
        Get multiple records by field values.
        
        Args:
            field: Field name
            values: List of field values
            load_relationships: Relationships to eager load
            
        Returns:
            List of model instances
        """
        if not hasattr(self.model_class, field) or not values:
            return []
        
        query = select(self.model_class).where(getattr(self.model_class, field).in_(values))
        
        if load_relationships:
            for relationship in load_relationships:
                query = query.options(selectinload(getattr(self.model_class, relationship)))
        
        result = await self.db.execute(query)
        return result.scalars().all()


class ReadOnlyRepository(BaseRepository[T]):
    """Read-only repository base class."""
    
    async def create(self, **kwargs) -> T:
        """Not supported in read-only repository."""
        raise NotImplementedError("Create operation not supported in read-only repository")
    
    async def update(self, id: UUID, **kwargs) -> Optional[T]:
        """Not supported in read-only repository."""
        raise NotImplementedError("Update operation not supported in read-only repository")
    
    async def delete(self, id: UUID) -> bool:
        """Not supported in read-only repository."""
        raise NotImplementedError("Delete operation not supported in read-only repository")
    
    async def bulk_create(self, records: List[Dict[str, Any]]) -> List[T]:
        """Not supported in read-only repository."""
        raise NotImplementedError("Bulk create operation not supported in read-only repository")
    
    async def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """Not supported in read-only repository."""
        raise NotImplementedError("Bulk update operation not supported in read-only repository")