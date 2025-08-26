"""
Comprehensive Test Suite for Community Platform Features.

Tests cover:
- Template Marketplace
- Plugin Management  
- Rating & Review System
- Content Moderation
- Security & Rate Limiting
- Performance & Caching
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

# Import services under test
from src.community.services.marketplace_service import MarketplaceService
from src.community.services.plugin_service import PluginService
from src.community.services.rating_service import RatingService
from src.community.services.moderation_service import ModerationService
from src.community.services.cache_service import CacheService
from src.community.security.rate_limiter import RateLimiter

# Import models
from src.community.models.template import TemplateSearchFilters
from src.community.models.plugin import PluginSearchFilters, PluginCreate
from src.community.models.rating import TemplateRatingCreate
from src.community.models.moderation import ContentType, ViolationType

# Import exceptions
from src.core.exceptions import ValidationError, NotFoundError, PermissionError


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_db():
    """Mock async database session."""
    db = Mock(spec=AsyncSession)
    db.query = Mock()
    db.add = Mock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.delete = Mock()
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.ping.return_value = True
    return redis_mock


@pytest.fixture
def sample_template_data():
    """Sample template data for testing."""
    return {
        'id': str(uuid4()),
        'slug': 'test-template',
        'name': 'Test Template',
        'description': 'A comprehensive test template',
        'short_description': 'Test template for unit tests',
        'template_type': 'web',
        'complexity_level': 'intermediate',
        'author_id': str(uuid4()),
        'categories': ['web', 'frontend'],
        'tags': ['react', 'typescript'],
        'frameworks': ['react', 'nextjs'],
        'languages': ['javascript', 'typescript', 'css'],
        'is_public': True,
        'is_featured': True,
        'is_premium': False,
        'price': 0.0,
        'average_rating': 4.3,
        'rating_count': 27,
        'download_count': 1523,
        'view_count': 5847,
        'star_count': 234
    }


@pytest.fixture
def sample_plugin_data():
    """Sample plugin data for testing."""
    return {
        'slug': 'test-plugin',
        'name': 'Test Plugin',
        'description': 'A test plugin for the community platform',
        'short_description': 'Test plugin',
        'plugin_type': 'extension',
        'version': '1.0.0',
        'categories': ['development', 'testing'],
        'tags': ['testing', 'automation'],
        'is_premium': False,
        'license_type': 'MIT'
    }


# =============================================================================
# MARKETPLACE SERVICE TESTS
# =============================================================================

class TestMarketplaceService:
    """Test cases for MarketplaceService."""
    
    def test_init(self, mock_db):
        """Test service initialization."""
        service = MarketplaceService(mock_db)
        assert service.db == mock_db
        assert hasattr(service, 'repository')
    
    @pytest.mark.asyncio
    async def test_search_templates_success(self, mock_db):
        """Test successful template search."""
        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.count = AsyncMock(return_value=5)
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query  
        mock_query.limit.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.all = AsyncMock(return_value=[])
        
        mock_db.query.return_value = mock_query
        
        service = MarketplaceService(mock_db)
        filters = TemplateSearchFilters(
            query="react",
            categories=["web"],
            min_rating=4.0,
            sort_by="popularity",
            sort_order="desc"
        )
        
        templates, total_count, metadata = await service.search_templates(
            filters, page=1, page_size=20
        )
        
        assert isinstance(templates, list)
        assert total_count == 5
        assert metadata['page'] == 1
        assert metadata['page_size'] == 20
        assert 'search_time' in metadata
    
    @pytest.mark.asyncio
    async def test_search_templates_error(self, mock_db):
        """Test error handling in template search."""
        mock_db.query.side_effect = Exception("Database error")
        
        service = MarketplaceService(mock_db)
        filters = TemplateSearchFilters(query="test")
        
        with pytest.raises(ValidationError, match="Search failed"):
            await service.search_templates(filters)
    
    @pytest.mark.asyncio
    async def test_get_featured_templates(self, mock_db):
        """Test getting featured templates."""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.all = AsyncMock(return_value=[])
        
        mock_db.query.return_value = mock_query
        
        service = MarketplaceService(mock_db)
        result = await service.get_featured_templates(limit=10)
        
        assert isinstance(result, list)
    
    @pytest.mark.asyncio 
    async def test_record_template_download(self, mock_db):
        """Test recording template downloads."""
        service = MarketplaceService(mock_db)
        service.repository = Mock()
        service.repository.get_template_by_id = AsyncMock(return_value=None)
        
        template_id = uuid4()
        user_id = uuid4()
        
        await service.record_template_download(
            template_id=template_id,
            user_id=user_id,
            download_type="api",
            project_name="Test Project"
        )
        
        mock_db.add.assert_called_once()


# =============================================================================
# PLUGIN SERVICE TESTS  
# =============================================================================

class TestPluginService:
    """Test cases for PluginService."""
    
    def test_init(self, mock_db):
        """Test plugin service initialization."""
        service = PluginService(mock_db)
        assert service.db == mock_db
        assert hasattr(service, 'storage_path')
        assert hasattr(service, 'code_sandbox')
    
    @pytest.mark.asyncio
    async def test_create_plugin_success(self, mock_db, sample_plugin_data):
        """Test successful plugin creation."""
        service = PluginService(mock_db)
        author_id = uuid4()
        
        # Mock successful creation
        with patch.object(service, '_store_plugin_file', return_value="/tmp/test.zip"), \
             patch.object(service, '_schedule_security_scan'):
            
            plugin = await service.create_plugin(
                sample_plugin_data, 
                author_id,
                plugin_file=b"fake zip content"
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_plugins(self, mock_db):
        """Test plugin search functionality."""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.count = AsyncMock(return_value=3)
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.all = AsyncMock(return_value=[])
        
        mock_db.query.return_value = mock_query
        
        service = PluginService(mock_db)
        filters = PluginSearchFilters(
            query="automation",
            plugin_type="extension",
            is_verified=True
        )
        
        plugins, total_count = await service.search_plugins(filters)
        
        assert isinstance(plugins, list)
        assert total_count == 3
    
    @pytest.mark.asyncio
    async def test_install_plugin(self, mock_db):
        """Test plugin installation."""
        # Mock plugin exists and is available
        mock_plugin = Mock()
        mock_plugin.is_public = True
        mock_plugin.status = "published"
        mock_plugin.version = "1.0.0"
        mock_plugin.install_count = 100
        mock_plugin.active_installs = 50
        
        service = PluginService(mock_db)
        service.get_plugin_by_id = AsyncMock(return_value=mock_plugin)
        
        # Mock no existing installation
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=None)
        
        plugin_id = uuid4()
        user_id = uuid4()
        
        install = await service.install_plugin(plugin_id, user_id, "manual")
        
        mock_db.add.assert_called()
        mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_security_scan(self, mock_db):
        """Test plugin security scanning."""
        service = PluginService(mock_db)
        
        # Mock plugin exists
        mock_plugin = Mock()
        mock_plugin.download_url = "/tmp/test.zip"
        mock_plugin.version = "1.0.0"
        
        service.get_plugin_by_id = AsyncMock(return_value=mock_plugin)
        
        # Mock security scan results
        with patch.object(service, '_run_security_scan') as mock_scan:
            mock_scan.return_value = {
                "overall_score": 85.0,
                "risk_level": "low",
                "vulnerabilities": [],
                "security_issues": [],
                "code_quality_issues": [],
                "scan_duration": 30,
                "files_scanned": 15
            }
            
            plugin_id = uuid4()
            scan_result = await service.perform_security_scan(plugin_id)
            
            assert scan_result.status == "completed"
            mock_db.add.assert_called()
            mock_db.commit.assert_called()


# =============================================================================
# RATING SERVICE TESTS
# =============================================================================

class TestRatingService:
    """Test cases for RatingService."""
    
    @pytest.mark.asyncio
    async def test_create_rating_success(self, mock_db):
        """Test successful rating creation."""
        service = RatingService(mock_db)
        
        # Mock no existing rating
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=None)
        
        # Mock template exists
        mock_template = Mock()
        mock_template.is_public = True
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=mock_template)
        
        # Mock download verification
        mock_db.query.return_value.filter.return_value.scalar = AsyncMock(return_value=1)
        
        rating_data = {
            "overall_rating": 4,
            "quality_rating": 4,
            "usability_rating": 5,
            "title": "Great template!",
            "content": "Very helpful for my project",
            "use_case": "Web development",
            "experience_level": "intermediate"
        }
        
        with patch.object(service, '_queue_for_moderation'), \
             patch.object(service, '_update_user_reputation'):
            
            rating = await service.create_rating(
                user_id=uuid4(),
                template_id=uuid4(),
                rating_data=rating_data
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_create_rating_duplicate(self, mock_db):
        """Test creating rating when user already rated."""
        service = RatingService(mock_db)
        
        # Mock existing rating
        existing_rating = Mock()
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=existing_rating)
        
        rating_data = {"overall_rating": 4}
        
        with pytest.raises(ValidationError, match="already rated"):
            await service.create_rating(
                user_id=uuid4(),
                template_id=uuid4(),
                rating_data=rating_data
            )
    
    @pytest.mark.asyncio
    async def test_vote_helpful(self, mock_db):
        """Test voting on review helpfulness."""
        service = RatingService(mock_db)
        
        # Mock no existing vote
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=None)
        
        with patch.object(service, '_update_helpfulness_counts'):
            result = await service.vote_helpful(
                rating_id=uuid4(),
                user_id=uuid4(),
                is_helpful=True
            )
            
            assert result == True
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_rating_summary(self, mock_db):
        """Test getting rating summary for template."""
        service = RatingService(mock_db)
        
        # Mock ratings data
        mock_ratings = [
            Mock(overall_rating=5, quality_rating=4, is_verified_usage=True, created_at=datetime.utcnow()),
            Mock(overall_rating=4, quality_rating=5, is_verified_usage=False, created_at=datetime.utcnow()),
            Mock(overall_rating=3, quality_rating=3, is_verified_usage=True, created_at=datetime.utcnow() - timedelta(days=40))
        ]
        
        mock_db.query.return_value.filter.return_value.all = AsyncMock(return_value=mock_ratings)
        
        template_id = uuid4()
        summary = await service.get_rating_summary(template_id)
        
        assert summary['total_ratings'] == 3
        assert summary['average_rating'] == 4.0  # (5+4+3)/3
        assert summary['verified_ratings'] == 2
        assert summary['recent_ratings'] == 2
        assert 'rating_distribution' in summary


# =============================================================================
# MODERATION SERVICE TESTS
# =============================================================================

class TestModerationService:
    """Test cases for ModerationService."""
    
    @pytest.mark.asyncio
    async def test_moderate_content_new(self, mock_db):
        """Test moderating new content."""
        service = ModerationService(mock_db)
        
        # Mock no existing moderation
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=None)
        
        # Mock content text extraction
        with patch.object(service, '_extract_content_text', return_value="Test content"), \
             patch.object(service, '_run_ai_analysis') as mock_ai, \
             patch.object(service, '_apply_moderation_rules') as mock_rules:
            
            mock_ai.return_value = {
                "confidence_score": 0.9,
                "spam_probability": 0.1,
                "toxicity_score": 0.05,
                "violations": [],
                "recommendation": "approve"
            }
            
            mock_rules.return_value = {"auto_action": "approve"}
            
            entry = await service.moderate_content(
                content_type="template",
                content_id=uuid4(),
                content_author_id=uuid4(),
                detection_method="ai_auto"
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_submit_appeal(self, mock_db):
        """Test submitting moderation appeal."""
        service = ModerationService(mock_db)
        
        # Mock existing moderation entry
        mock_entry = Mock()
        mock_entry.content_author_id = uuid4()
        mock_entry.appeal_count = 0
        
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=mock_entry)
        
        # Mock no existing appeal
        mock_db.query.return_value.filter.return_value.first = AsyncMock(return_value=None)
        
        appeal = await service.submit_appeal(
            moderation_id=uuid4(),
            user_id=mock_entry.content_author_id,
            reason="The content was incorrectly flagged",
            additional_context="This is educational content"
        )
        
        mock_db.add.assert_called()
        mock_db.commit.assert_called()
        assert mock_entry.appeal_count == 1
    
    @pytest.mark.asyncio
    async def test_get_moderation_stats(self, mock_db):
        """Test getting moderation statistics."""
        service = ModerationService(mock_db)
        
        # Mock various count queries
        mock_db.query.return_value.scalar = AsyncMock(side_effect=[100, 15, 5, 3, 80, 25, 10, 2])
        mock_db.query.return_value.group_by.return_value.order_by.return_value.limit.return_value.all = AsyncMock(return_value=[])
        mock_db.query.return_value.filter.return_value.scalar = AsyncMock(return_value=0.5)
        
        stats = await service.get_moderation_stats()
        
        assert 'total_items_moderated' in stats
        assert 'pending_items' in stats
        assert 'auto_moderation_percentage' in stats


# =============================================================================
# CACHE SERVICE TESTS
# =============================================================================

class TestCacheService:
    """Test cases for CacheService."""
    
    @patch('src.community.services.cache_service.redis')
    def test_init_success(self, mock_redis_module):
        """Test successful cache service initialization."""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_module.from_url.return_value = mock_redis_client
        
        cache = CacheService("redis://localhost:6379/0")
        
        assert cache._redis == mock_redis_client
        assert cache.is_available() == True
    
    @patch('src.community.services.cache_service.redis')
    def test_init_failure(self, mock_redis_module):
        """Test cache service initialization failure."""
        mock_redis_module.from_url.side_effect = Exception("Connection failed")
        
        cache = CacheService("redis://localhost:6379/0")
        
        assert cache._redis is None
        assert cache.is_available() == False
    
    def test_get_set_operations(self, mock_redis):
        """Test basic get/set operations."""
        with patch('src.community.services.cache_service.redis.from_url', return_value=mock_redis):
            cache = CacheService()
            
            # Test set
            result = cache.set("test:key", {"data": "value"}, ttl=3600)
            assert result == True
            
            # Test get
            mock_redis.get.return_value = b'{"data": "value"}'
            result = cache.get("test:key")
            assert result == {"data": "value"}
    
    def test_template_caching(self, mock_redis):
        """Test template-specific caching methods."""
        with patch('src.community.services.cache_service.redis.from_url', return_value=mock_redis):
            cache = CacheService()
            
            template_id = uuid4()
            template_data = {"name": "Test Template", "type": "web"}
            
            # Test set template
            result = cache.set_template(template_id, template_data)
            assert result == True
            
            # Test get template
            mock_redis.get.return_value = b'{"name": "Test Template", "type": "web"}'
            result = cache.get_template(template_id)
            assert result == template_data
    
    def test_search_result_caching(self, mock_redis):
        """Test search result caching."""
        with patch('src.community.services.cache_service.redis.from_url', return_value=mock_redis):
            cache = CacheService()
            
            search_hash = "abc123"
            results = {"templates": [], "total": 0}
            
            # Test set search results
            cache.set_search_results(search_hash, results, page=1, page_size=20)
            
            # Verify correct key format
            expected_key = f"search:{search_hash}:p1:s20"
            mock_redis.setex.assert_called_with(
                expected_key, 
                cache.SEARCH_TTL, 
                cache._serialize(results)
            )


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    def test_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter()
        assert hasattr(limiter, 'cache')
        assert hasattr(limiter, 'config')
    
    def test_get_client_ip(self):
        """Test client IP extraction."""
        limiter = RateLimiter()
        
        # Mock request with forwarded header
        mock_request = Mock()
        mock_request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        mock_request.client = None
        
        ip = limiter.get_client_ip(mock_request)
        assert ip == "192.168.1.1"
        
        # Mock request with real IP header
        mock_request.headers = {"X-Real-IP": "203.0.113.1"}
        ip = limiter.get_client_ip(mock_request)
        assert ip == "203.0.113.1"
        
        # Mock request with direct connection
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "198.51.100.1"
        
        ip = limiter.get_client_ip(mock_request)
        assert ip == "198.51.100.1"
    
    def test_check_rate_limit(self):
        """Test rate limit checking."""
        limiter = RateLimiter()
        
        # Mock cache service
        limiter.cache.increment_rate_limit = Mock(return_value=(5, False))
        
        is_allowed, limit_info = limiter.check_rate_limit(
            "user:123", 
            "api_calls", 
            custom_limit=(100, 3600)
        )
        
        assert is_allowed == True
        assert limit_info['limit'] == 100
        assert limit_info['remaining'] == 95
        assert 'reset' in limit_info
    
    @patch('src.community.security.rate_limiter.get_cache_service')
    def test_rate_limit_decorator(self, mock_get_cache):
        """Test rate limiting decorator."""
        from src.community.security.rate_limiter import rate_limit
        
        # Mock cache service
        mock_cache = Mock()
        mock_cache.increment_rate_limit.return_value = (1, False)
        mock_get_cache.return_value = mock_cache
        
        @rate_limit("test_action", (10, 60))
        async def test_function(request, current_user=None):
            return "success"
        
        # Mock request
        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        
        # Test function execution
        result = asyncio.run(test_function(mock_request))
        assert result == "success"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCommunityIntegration:
    """Integration tests for community platform."""
    
    @pytest.mark.asyncio
    async def test_template_upload_workflow(self, mock_db):
        """Test complete template upload workflow."""
        # This would test the full workflow:
        # 1. Template creation
        # 2. Security scanning
        # 3. Content moderation
        # 4. Cache invalidation
        # 5. Search indexing
        pass
    
    @pytest.mark.asyncio
    async def test_plugin_install_workflow(self, mock_db):
        """Test complete plugin installation workflow."""
        # This would test:
        # 1. Plugin search and discovery
        # 2. Security verification
        # 3. Installation tracking
        # 4. Analytics recording
        pass
    
    @pytest.mark.asyncio
    async def test_review_moderation_workflow(self, mock_db):
        """Test review creation and moderation workflow."""
        # This would test:
        # 1. Review creation
        # 2. AI moderation
        # 3. Human review (if needed)
        # 4. Publication
        # 5. Reputation updates
        pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for community platform."""
    
    @pytest.mark.asyncio
    async def test_search_performance(self):
        """Test search performance with large datasets."""
        # This would test search performance under load
        pass
    
    @pytest.mark.asyncio
    async def test_cache_hit_rates(self):
        """Test cache effectiveness and hit rates."""
        # This would measure cache performance
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent template/plugin operations."""
        # This would test thread safety and concurrency
        pass


# =============================================================================
# SECURITY TESTS
# =============================================================================

class TestSecurity:
    """Security tests for community platform."""
    
    @pytest.mark.asyncio
    async def test_malicious_file_upload(self):
        """Test handling of malicious file uploads."""
        # This would test security scanning effectiveness
        pass
    
    @pytest.mark.asyncio
    async def test_content_injection_protection(self):
        """Test protection against content injection attacks."""
        # This would test input sanitization
        pass
    
    @pytest.mark.asyncio
    async def test_rate_limit_bypass_attempts(self):
        """Test attempts to bypass rate limiting."""
        # This would test rate limiter robustness
        pass


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        "-v",
        "tests/community/test_community_platform.py::TestMarketplaceService",
        "tests/community/test_community_platform.py::TestPluginService",
        "tests/community/test_community_platform.py::TestRatingService",
        "tests/community/test_community_platform.py::TestModerationService",
        "tests/community/test_community_platform.py::TestCacheService",
        "tests/community/test_community_platform.py::TestRateLimiter"
    ])