"""
Template Repository - Data access layer for template operations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.template import Template, TemplateVersion
from .base_repository import BaseRepository


class TemplateRepository(BaseRepository[Template]):
    """Repository for template data access."""
    
    def __init__(self, db: AsyncSession):
        """Initialize template repository."""
        super().__init__(db, Template)
    
    async def get_template_by_slug(self, slug: str) -> Optional[Template]:
        """
        Get template by slug.
        
        Args:
            slug: Template slug
            
        Returns:
            Template instance or None
        """
        return await self.get_by_field('slug', slug)
    
    async def get_public_templates(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: str = 'updated_at'
    ) -> List[Template]:
        """
        Get public templates.
        
        Args:
            limit: Maximum number of templates
            offset: Number of templates to skip
            order_by: Order by field
            
        Returns:
            List of public templates
        """
        query = select(Template).where(Template.is_public == True)
        
        if order_by and hasattr(Template, order_by):
            query = query.order_by(desc(getattr(Template, order_by)))
        
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        query = query.options(selectinload(Template.author))
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_featured_templates(self, limit: int = 10) -> List[Template]:
        """
        Get featured templates.
        
        Args:
            limit: Maximum number of templates
            
        Returns:
            List of featured templates
        """
        query = select(Template).where(
            and_(Template.is_featured == True, Template.is_public == True)
        ).order_by(desc(Template.updated_at)).limit(limit)
        
        query = query.options(selectinload(Template.author))
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_templates_by_author(
        self,
        author_id: UUID,
        include_private: bool = False,
        limit: Optional[int] = None
    ) -> List[Template]:
        """
        Get templates by author.
        
        Args:
            author_id: Author user ID
            include_private: Whether to include private templates
            limit: Maximum number of templates
            
        Returns:
            List of author's templates
        """
        query = select(Template).where(Template.author_id == author_id)
        
        if not include_private:
            query = query.where(Template.is_public == True)
        
        query = query.order_by(desc(Template.updated_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def search_templates_advanced(
        self,
        query_text: Optional[str] = None,
        template_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        frameworks: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        complexity_level: Optional[str] = None,
        is_featured: Optional[bool] = None,
        author_id: Optional[UUID] = None,
        created_after: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = 'updated_at',
        sort_order: str = 'desc'
    ) -> Tuple[List[Template], int]:
        """
        Advanced template search with multiple filters.
        
        Args:
            query_text: Text search query
            template_type: Template type filter
            categories: Category filters
            frameworks: Framework filters
            languages: Language filters
            min_rating: Minimum rating
            max_rating: Maximum rating
            complexity_level: Complexity level filter
            is_featured: Featured filter
            author_id: Author filter
            created_after: Created after date
            updated_after: Updated after date
            limit: Results limit
            offset: Results offset
            sort_by: Sort field
            sort_order: Sort order (asc/desc)
            
        Returns:
            Tuple of (templates, total_count)
        """
        # Base query for public templates
        query = select(Template).where(Template.is_public == True)
        count_query = select(func.count(Template.id)).where(Template.is_public == True)
        
        # Text search
        if query_text:
            search_terms = query_text.lower().split()
            for term in search_terms:
                text_filter = or_(
                    Template.name.ilike(f'%{term}%'),
                    Template.description.ilike(f'%{term}%'),
                    Template.short_description.ilike(f'%{term}%'),
                    func.jsonb_exists_any(Template.tags, [term]),
                    func.jsonb_exists_any(Template.categories, [term])
                )
                query = query.where(text_filter)
                count_query = count_query.where(text_filter)
        
        # Type filter
        if template_type:
            query = query.where(Template.template_type == template_type)
            count_query = count_query.where(Template.template_type == template_type)
        
        # Categories filter
        if categories:
            category_filter = or_(*[
                func.jsonb_exists(Template.categories, category) 
                for category in categories
            ])
            query = query.where(category_filter)
            count_query = count_query.where(category_filter)
        
        # Frameworks filter
        if frameworks:
            framework_filter = or_(*[
                func.jsonb_exists(Template.frameworks, framework)
                for framework in frameworks
            ])
            query = query.where(framework_filter)
            count_query = count_query.where(framework_filter)
        
        # Languages filter
        if languages:
            language_filter = or_(*[
                func.jsonb_exists(Template.languages, language)
                for language in languages
            ])
            query = query.where(language_filter)
            count_query = count_query.where(language_filter)
        
        # Rating filters
        if min_rating is not None:
            rating_filter = Template.average_rating >= min_rating
            query = query.where(rating_filter)
            count_query = count_query.where(rating_filter)
        
        if max_rating is not None:
            rating_filter = Template.average_rating <= max_rating
            query = query.where(rating_filter)
            count_query = count_query.where(rating_filter)
        
        # Complexity filter
        if complexity_level:
            complexity_filter = Template.complexity_level == complexity_level
            query = query.where(complexity_filter)
            count_query = count_query.where(complexity_filter)
        
        # Featured filter
        if is_featured is not None:
            featured_filter = Template.is_featured == is_featured
            query = query.where(featured_filter)
            count_query = count_query.where(featured_filter)
        
        # Author filter
        if author_id:
            author_filter = Template.author_id == author_id
            query = query.where(author_filter)
            count_query = count_query.where(author_filter)
        
        # Date filters
        if created_after:
            created_filter = Template.created_at >= created_after
            query = query.where(created_filter)
            count_query = count_query.where(created_filter)
        
        if updated_after:
            updated_filter = Template.updated_at >= updated_after
            query = query.where(updated_filter)
            count_query = count_query.where(updated_filter)
        
        # Get total count
        count_result = await self.db.execute(count_query)
        total_count = count_result.scalar()
        
        # Apply sorting
        if hasattr(Template, sort_by):
            sort_column = getattr(Template, sort_by)
            if sort_order.lower() == 'asc':
                query = query.order_by(sort_column)
            else:
                query = query.order_by(desc(sort_column))
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        # Load relationships
        query = query.options(selectinload(Template.author))
        
        # Execute query
        result = await self.db.execute(query)
        templates = result.scalars().all()
        
        return templates, total_count
    
    async def get_similar_templates(
        self,
        template: Template,
        limit: int = 10
    ) -> List[Template]:
        """
        Get templates similar to the given template.
        
        Args:
            template: Reference template
            limit: Maximum number of similar templates
            
        Returns:
            List of similar templates
        """
        query = select(Template).where(
            and_(
                Template.id != template.id,
                Template.is_public == True,
                or_(
                    # Similar categories
                    func.jsonb_path_exists(
                        Template.categories,
                        text(f"'$[*] ? (@ in {template.categories})'")
                    ) if template.categories else text('false'),
                    # Similar frameworks
                    func.jsonb_path_exists(
                        Template.frameworks,
                        text(f"'$[*] ? (@ in {template.frameworks})'")
                    ) if template.frameworks else text('false'),
                    # Same complexity level
                    Template.complexity_level == template.complexity_level,
                    # Same template type
                    Template.template_type == template.template_type
                )
            )
        ).order_by(
            desc(Template.average_rating),
            desc(Template.download_count)
        ).limit(limit)
        
        query = query.options(selectinload(Template.author))
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_trending_templates(
        self,
        days: int = 7,
        limit: int = 10
    ) -> List[Template]:
        """
        Get trending templates based on recent activity.
        
        Args:
            days: Number of days to consider for trending
            limit: Maximum number of templates
            
        Returns:
            List of trending templates
        """
        # This is a simplified version - in practice you'd join with
        # download/view tables for more accurate trending calculation
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(Template).where(
            and_(
                Template.is_public == True,
                Template.updated_at >= since_date
            )
        ).order_by(
            desc(Template.download_count + Template.star_count * 2),
            desc(Template.average_rating)
        ).limit(limit)
        
        query = query.options(selectinload(Template.author))
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_template_with_versions(self, template_id: UUID) -> Optional[Template]:
        """
        Get template with all its versions.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template with versions loaded
        """
        query = select(Template).where(Template.id == template_id).options(
            selectinload(Template.versions),
            selectinload(Template.author)
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_template_stats(self, template_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive template statistics.
        
        Args:
            template_id: Template ID
            
        Returns:
            Dictionary of statistics
        """
        template = await self.get_by_id(template_id)
        if not template:
            return None
        
        # Get version count
        version_count_query = select(func.count(TemplateVersion.id)).where(
            TemplateVersion.template_id == template_id
        )
        version_count_result = await self.db.execute(version_count_query)
        version_count = version_count_result.scalar()
        
        return {
            'template_id': str(template_id),
            'download_count': template.download_count,
            'usage_count': template.usage_count,
            'star_count': template.star_count,
            'fork_count': template.fork_count,
            'average_rating': template.average_rating,
            'rating_count': template.rating_count,
            'quality_score': template.quality_score,
            'version_count': version_count,
            'is_featured': template.is_featured,
            'created_at': template.created_at,
            'updated_at': template.updated_at,
            'last_used_at': template.last_used_at
        }
    
    async def update_template_metrics(
        self,
        template_id: UUID,
        download_increment: int = 0,
        usage_increment: int = 0,
        star_increment: int = 0,
        fork_increment: int = 0
    ) -> bool:
        """
        Update template metrics atomically.
        
        Args:
            template_id: Template ID
            download_increment: Download count increment
            usage_increment: Usage count increment
            star_increment: Star count increment
            fork_increment: Fork count increment
            
        Returns:
            True if updated successfully
        """
        try:
            updates = {}
            
            if download_increment != 0:
                updates['download_count'] = Template.download_count + download_increment
                updates['last_used_at'] = datetime.utcnow()
            
            if usage_increment != 0:
                updates['usage_count'] = Template.usage_count + usage_increment
                updates['last_used_at'] = datetime.utcnow()
            
            if star_increment != 0:
                updates['star_count'] = Template.star_count + star_increment
            
            if fork_increment != 0:
                updates['fork_count'] = Template.fork_count + fork_increment
            
            if updates:
                updates['updated_at'] = datetime.utcnow()
                await self.update(template_id, **updates)
            
            return True
            
        except Exception:
            return False


class TemplateVersionRepository(BaseRepository[TemplateVersion]):
    """Repository for template version data access."""
    
    def __init__(self, db: AsyncSession):
        """Initialize template version repository."""
        super().__init__(db, TemplateVersion)
    
    async def get_versions_by_template(
        self,
        template_id: UUID,
        limit: Optional[int] = None
    ) -> List[TemplateVersion]:
        """
        Get all versions for a template.
        
        Args:
            template_id: Template ID
            limit: Maximum number of versions
            
        Returns:
            List of template versions
        """
        query = select(TemplateVersion).where(
            TemplateVersion.template_id == template_id
        ).order_by(desc(TemplateVersion.created_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_latest_version(self, template_id: UUID) -> Optional[TemplateVersion]:
        """
        Get the latest version for a template.
        
        Args:
            template_id: Template ID
            
        Returns:
            Latest template version
        """
        query = select(TemplateVersion).where(
            TemplateVersion.template_id == template_id
        ).order_by(desc(TemplateVersion.created_at)).limit(1)
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_version_by_number(
        self,
        template_id: UUID,
        version: str
    ) -> Optional[TemplateVersion]:
        """
        Get specific version by version number.
        
        Args:
            template_id: Template ID
            version: Version string
            
        Returns:
            Template version or None
        """
        query = select(TemplateVersion).where(
            and_(
                TemplateVersion.template_id == template_id,
                TemplateVersion.version == version
            )
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()