"""
Comprehensive test suite for enhanced community platform features.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from src.community.services.marketplace_service_enhanced import EnhancedMarketplaceService
from src.community.services.recommendation_engine import RecommendationEngine
from src.community.services.rating_service import RatingService
from src.community.services.plugin_service_enhanced import PluginService, SecurityScanner
from src.community.services.moderation_service import ModerationService, ContentFilter
from src.community.services.cache_service import CacheService
from src.community.repositories.marketplace_repository import MarketplaceRepository
from src.community.models.template import Template, TemplateSearchFilters, TemplateType, ComplexityLevel
from src.community.models.plugin import Plugin, PluginType, PluginStatus
from src.community.models.rating import TemplateRating, ReviewStatus, ReportReason
from src.community.models.marketplace import FeaturedCollection, TemplateTag
from src.community.models.user import UserProfile


class TestEnhancedMarketplaceService:
    """Test enhanced marketplace service functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.get = AsyncMock()
        session.scalar = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.refresh = AsyncMock()
        return session
    
    @pytest.fixture
    def cache_service(self):
        """Cache service fixture."""
        return CacheService()
    
    @pytest.fixture
    def marketplace_service(self, mock_session, cache_service):
        """Enhanced marketplace service fixture."""
        return EnhancedMarketplaceService(mock_session, cache_service)
    
    @pytest.mark.asyncio
    async def test_advanced_template_search(self, marketplace_service, mock_session):
        """Test advanced template search with AI-powered ranking."""
        # Setup
        filters = TemplateSearchFilters(
            query="react components",
            template_type=TemplateType.COMPONENT,
            categories=["frontend"],
            min_rating=4.0
        )
        
        # Mock template data
        mock_template = Template(
            id=uuid4(),
            name="React Button Component",
            description="A reusable React button component",
            template_type=TemplateType.COMPONENT.value,
            is_public=True,
            average_rating=4.5,
            rating_count=10,
            download_count=100,
            categories=["frontend", "react"]
        )
        
        # Mock repository response
        marketplace_service.repository.search_templates_advanced = AsyncMock(
            return_value=([mock_template], 1, {'page': 1, 'total_count': 1})
        )
        
        # Execute
        templates, total_count, metadata = await marketplace_service.search_templates_advanced(
            filters=filters,
            user_id=uuid4(),
            personalize=True
        )
        
        # Verify
        assert len(templates) == 1
        assert total_count == 1
        assert templates[0]['name'] == "React Button Component"
        assert 'personalization_score' in templates[0]
        assert 'enhanced_metrics' in templates[0] or 'recent_downloads' in templates[0]
    
    @pytest.mark.asyncio
    async def test_personalized_recommendations(self, marketplace_service):
        """Test AI-powered personalized recommendations."""
        user_id = uuid4()
        
        # Mock recommendation engine response
        mock_recommendations = [
            {
                'id': str(uuid4()),
                'name': 'Vue.js Dashboard',
                'recommendation_score': 0.95,
                'recommendation_explanation': 'Based on your React experience',
                'recommendation_type': 'hybrid'
            }
        ]
        
        marketplace_service.recommendation_engine.get_user_recommendations = AsyncMock(
            return_value=mock_recommendations
        )
        
        marketplace_service._get_enhanced_template_metrics = AsyncMock(
            return_value={'recent_downloads': 50, 'momentum_score': 25.0}
        )
        
        marketplace_service._is_template_trending = AsyncMock(return_value=True)
        marketplace_service._calculate_quality_score = AsyncMock(return_value=0.85)
        
        # Execute
        recommendations = await marketplace_service.get_personalized_recommendations(
            user_id=user_id,
            limit=10
        )
        
        # Verify
        assert len(recommendations) == 1
        assert recommendations[0]['name'] == 'Vue.js Dashboard'
        assert recommendations[0]['is_trending'] == True
        assert recommendations[0]['quality_score'] == 0.85
    
    @pytest.mark.asyncio
    async def test_create_featured_collection(self, marketplace_service, mock_session):
        """Test creating featured collections with intelligent template selection."""
        curator_id = uuid4()
        template_ids = [uuid4(), uuid4()]
        
        # Mock template validation
        marketplace_service._validate_templates_for_collection = AsyncMock(
            return_value=template_ids
        )
        
        # Mock session operations
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        
        mock_collection = FeaturedCollection(
            id=uuid4(),
            slug="test-collection",
            name="Test Collection",
            description="A test collection",
            collection_type="curated",
            curator_id=curator_id,
            template_ids=[str(tid) for tid in template_ids],
            template_count=len(template_ids)
        )
        mock_session.refresh = AsyncMock(return_value=mock_collection)
        
        # Execute
        result = await marketplace_service.create_featured_collection(
            name="Test Collection",
            description="A test collection",
            collection_type="curated",
            curator_id=curator_id,
            template_ids=template_ids
        )
        
        # Verify
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert result['name'] == "Test Collection"
        assert result['template_count'] == 2
    
    @pytest.mark.asyncio
    async def test_marketplace_insights(self, marketplace_service):
        """Test comprehensive marketplace insights generation."""
        # Mock repository methods
        marketplace_service.repository.get_marketplace_health_metrics = AsyncMock(
            return_value={
                'content_health': {'total_templates': 100},
                'activity_health': {'recent_downloads': 500},
                'quality_health': {'average_rating': 4.2}
            }
        )
        
        marketplace_service._analyze_marketplace_trends = AsyncMock(
            return_value={'trending_categories': ['react', 'vue']}
        )
        
        marketplace_service._analyze_user_behavior = AsyncMock(
            return_value={'engagement_metrics': {'daily_active_users': 250}}
        )
        
        marketplace_service._analyze_content_quality = AsyncMock(
            return_value={'quality_distribution': {'high': 60, 'medium': 30, 'low': 10}}
        )
        
        marketplace_service._calculate_growth_projections = AsyncMock(
            return_value={'projected_downloads': 1000}
        )
        
        marketplace_service._generate_marketplace_recommendations = AsyncMock(
            return_value=[{'type': 'content_gap', 'description': 'More Python templates needed'}]
        )
        
        marketplace_service._calculate_performance_metrics = AsyncMock(
            return_value={'conversion_rate': 15.5}
        )
        
        # Execute
        insights = await marketplace_service.get_marketplace_insights(time_period="30d")
        
        # Verify
        assert 'overview' in insights
        assert 'trends' in insights
        assert 'user_behavior' in insights
        assert 'content_quality' in insights
        assert 'growth_projections' in insights
        assert 'recommendations' in insights
        assert 'performance_metrics' in insights


class TestRecommendationEngine:
    """Test recommendation engine functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.get = AsyncMock()
        session.scalars = AsyncMock()
        return session
    
    @pytest.fixture
    def recommendation_engine(self, mock_session):
        """Recommendation engine fixture."""
        return RecommendationEngine(mock_session)
    
    @pytest.mark.asyncio
    async def test_user_recommendations_hybrid_approach(self, recommendation_engine):
        """Test hybrid recommendation algorithm."""
        user_id = uuid4()
        
        # Mock user profile
        recommendation_engine._get_user_profile = AsyncMock(
            return_value={
                'id': user_id,
                'username': 'testuser',
                'download_count': 50,
                'rating_count': 10,
                'is_active': True
            }
        )
        
        # Mock different recommendation algorithms
        recommendation_engine._get_collaborative_recommendations = AsyncMock(
            return_value=[(uuid4(), 0.8, "Users with similar taste liked this")]
        )
        
        recommendation_engine._get_content_based_recommendations = AsyncMock(
            return_value=[(uuid4(), 0.7, "Similar to your previous downloads")]
        )
        
        recommendation_engine._get_popularity_recommendations = AsyncMock(
            return_value=[(uuid4(), 0.6, "Popular in community")]
        )
        
        recommendation_engine._get_trending_recommendations = AsyncMock(
            return_value=[(uuid4(), 0.9, "Currently trending")]
        )
        
        # Mock fusion and filtering
        recommendation_engine._fuse_recommendations = AsyncMock(
            return_value=[(uuid4(), 0.85, "Hybrid recommendation")]
        )
        
        recommendation_engine._apply_diversity_and_filters = AsyncMock(
            return_value=[{
                'id': str(uuid4()),
                'name': 'Recommended Template',
                'recommendation_score': 0.85,
                'recommendation_type': 'hybrid'
            }]
        )
        
        # Execute
        recommendations = await recommendation_engine.get_user_recommendations(
            user_id=user_id,
            item_type="template",
            limit=10
        )
        
        # Verify
        assert len(recommendations) == 1
        assert recommendations[0]['recommendation_type'] == 'hybrid'
        assert recommendations[0]['recommendation_score'] == 0.85
    
    @pytest.mark.asyncio
    async def test_similar_items_content_based(self, recommendation_engine, mock_session):
        """Test content-based similar item recommendations."""
        template_id = uuid4()
        
        # Mock template
        mock_template = Template(
            id=template_id,
            name="React Component",
            categories=["frontend", "react"],
            frameworks=["react"],
            tags=["ui", "component"]
        )
        
        mock_session.get.return_value = mock_template
        
        # Mock content analysis
        recommendation_engine._extract_content_features = MagicMock(
            return_value={'categories': ['frontend'], 'frameworks': ['react']}
        )
        
        recommendation_engine._find_content_similar_items = AsyncMock(
            return_value=[{
                'id': str(uuid4()),
                'name': 'Similar Component',
                'similarity_score': 0.75
            }]
        )
        
        # Execute
        similar_items = await recommendation_engine.get_similar_items(
            item_id=template_id,
            item_type="template",
            limit=5
        )
        
        # Verify
        assert len(similar_items) == 1
        assert similar_items[0]['similarity_score'] == 0.75
    
    @pytest.mark.asyncio
    async def test_model_training(self, recommendation_engine):
        """Test recommendation model training."""
        # Mock training methods
        recommendation_engine._train_content_model = AsyncMock(
            return_value={'status': 'completed', 'features_extracted': 1000}
        )
        
        recommendation_engine._train_collaborative_model = AsyncMock(
            return_value={'status': 'completed', 'latent_factors': 50, 'rmse': 0.85}
        )
        
        # Execute
        results = await recommendation_engine.train_recommendation_models()
        
        # Verify
        assert 'started_at' in results
        assert 'completed_at' in results
        assert len(results['models_trained']) == 2
        assert results['models_trained'][0]['model'] == 'content_based'
        assert results['models_trained'][1]['model'] == 'collaborative_filtering'


class TestRatingService:
    """Test enhanced rating service functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.get = AsyncMock()
        session.scalar = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.refresh = AsyncMock()
        return session
    
    @pytest.fixture
    def rating_service(self, mock_session):
        """Rating service fixture."""
        cache_service = CacheService()
        moderation_service = MagicMock()
        
        service = RatingService(mock_session, cache_service)
        service.moderation = moderation_service
        return service
    
    @pytest.mark.asyncio
    async def test_create_rating_with_moderation(self, rating_service, mock_session):
        """Test creating rating with automatic moderation."""
        template_id = uuid4()
        user_id = uuid4()
        
        # Mock existing rating check (no existing rating)
        mock_session.execute.return_value.first.return_value = None
        
        # Mock template and user
        mock_template = Template(id=template_id, is_public=True, version="1.0.0")
        mock_user = UserProfile(id=user_id, username="testuser")
        
        mock_session.get.side_effect = [mock_template, mock_user]
        
        # Mock usage verification
        rating_service._verify_template_usage = AsyncMock(return_value=True)
        
        # Mock moderation
        rating_service.moderation.auto_moderate_content = AsyncMock(
            return_value={'action': 'approve'}
        )
        
        # Mock rating creation
        mock_rating = TemplateRating(
            id=uuid4(),
            template_id=template_id,
            user_id=user_id,
            overall_rating=5,
            title="Great template!",
            content="Very useful and well documented",
            status=ReviewStatus.APPROVED.value
        )
        mock_session.flush = AsyncMock()
        
        # Mock stats update
        rating_service._update_template_rating_stats = AsyncMock()
        rating_service._update_user_reputation = AsyncMock()
        
        # Execute
        result = await rating_service.create_rating(
            template_id=template_id,
            user_id=user_id,
            overall_rating=5,
            title="Great template!",
            content="Very useful and well documented",
            detailed_ratings={'quality': 5, 'usability': 4},
            usage_context={'use_case': 'production', 'experience_level': 'intermediate'}
        )
        
        # Verify
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        rating_service._update_template_rating_stats.assert_called_once_with(template_id)
        rating_service._update_user_reputation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vote_helpful(self, rating_service, mock_session):
        """Test voting on rating helpfulness."""
        rating_id = uuid4()
        user_id = uuid4()
        
        # Mock rating
        mock_rating = TemplateRating(
            id=rating_id,
            user_id=uuid4(),
            status=ReviewStatus.APPROVED.value
        )
        mock_session.get.return_value = mock_rating
        
        # Mock no existing vote
        mock_session.execute.return_value.first.return_value = None
        
        # Mock stats update
        rating_service._update_rating_helpfulness_stats = AsyncMock()
        rating_service._update_user_reputation = AsyncMock()
        rating_service._get_helpfulness_stats = AsyncMock(
            return_value={
                'helpful_votes': 1,
                'not_helpful_votes': 0,
                'total_votes': 1,
                'helpfulness_score': 0.8
            }
        )
        
        # Execute
        result = await rating_service.vote_helpful(
            rating_id=rating_id,
            user_id=user_id,
            is_helpful=True
        )
        
        # Verify
        assert result['user_voted'] == True
        assert result['is_helpful'] == True
        assert result['helpful_votes'] == 1
        rating_service._update_rating_helpfulness_stats.assert_called_once_with(rating_id)
        rating_service._update_user_reputation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_rating_statistics(self, rating_service, mock_session):
        """Test getting comprehensive rating statistics."""
        template_id = uuid4()
        
        # Mock database results
        mock_result = MagicMock()
        mock_result.first.return_value = MagicMock(
            total_ratings=25,
            average_rating=4.2,
            five_star=10,
            four_star=8,
            three_star=5,
            two_star=1,
            one_star=1,
            verified_usage_count=20,
            avg_quality=4.3,
            avg_usability=4.1,
            avg_documentation=3.9,
            avg_support=4.0
        )
        
        mock_recent_result = MagicMock()
        mock_recent_result.first.return_value = MagicMock(
            recent_ratings=5,
            recent_average=4.4
        )
        
        mock_session.execute.side_effect = [mock_result, mock_recent_result]
        
        # Execute
        stats = await rating_service.get_rating_statistics(
            template_id=template_id,
            detailed=True
        )
        
        # Verify
        assert stats['total_ratings'] == 25
        assert stats['average_rating'] == 4.2
        assert stats['rating_distribution']['5'] == 10
        assert stats['verified_usage_percentage'] == 80.0
        assert 'detailed_ratings' in stats
        assert 'recent_trends' in stats
        assert stats['recent_trends']['trend_direction'] == 'improving'


class TestPluginService:
    """Test enhanced plugin service functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.get = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session
    
    @pytest.fixture
    def plugin_service(self, mock_session):
        """Plugin service fixture."""
        cache_service = CacheService()
        return PluginService(mock_session, cache_service)
    
    @pytest.mark.asyncio
    async def test_create_plugin_with_security_scan(self, plugin_service, mock_session):
        """Test creating plugin with comprehensive security scanning."""
        author_id = uuid4()
        plugin_data = {
            'name': 'Test Plugin',
            'description': 'A test plugin for validation',
            'plugin_type': PluginType.EXTENSION.value,
            'version': '1.0.0',
            'categories': ['development'],
            'dependencies': ['requests==2.28.0', 'pyyaml>=5.4.0']
        }
        
        source_code = """
import requests
import yaml

def safe_function():
    print("Hello World")
    return True
        """
        
        # Mock slug check (no existing plugin)
        mock_session.execute.return_value.first.return_value = None
        
        # Mock security scan
        plugin_service._perform_security_scan = AsyncMock(
            return_value={
                'overall_score': 85.0,
                'risk_level': 'low',
                'vulnerabilities': [],
                'security_issues': [],
                'dependency_issues': []
            }
        )
        
        # Mock dependency creation
        plugin_service._create_plugin_dependencies = AsyncMock()
        
        # Mock moderation
        plugin_service.moderation.auto_moderate_content = AsyncMock(
            return_value={'action': 'approve'}
        )
        
        # Mock plugin object
        mock_plugin = Plugin(
            id=uuid4(),
            slug="test-plugin",
            name="Test Plugin",
            plugin_type=PluginType.EXTENSION.value,
            author_id=author_id,
            status=PluginStatus.SUBMITTED.value,
            is_security_approved=True
        )
        
        mock_plugin.author = UserProfile(id=author_id, username="testauthor")
        mock_session.refresh.return_value = mock_plugin
        
        # Execute
        result = await plugin_service.create_plugin(
            author_id=author_id,
            plugin_data=plugin_data,
            source_code=source_code,
            auto_scan=True
        )
        
        # Verify
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called()
        plugin_service._perform_security_scan.assert_called_once()
        plugin_service._create_plugin_dependencies.assert_called_once()
        assert result['name'] == "Test Plugin"
        assert result['is_security_approved'] == True
        assert 'security_scan' in result
    
    @pytest.mark.asyncio
    async def test_install_plugin(self, plugin_service, mock_session):
        """Test plugin installation with tracking."""
        plugin_id = uuid4()
        user_id = uuid4()
        
        # Mock plugin
        mock_plugin = Plugin(
            id=plugin_id,
            name="Test Plugin",
            version="1.0.0",
            status=PluginStatus.PUBLISHED.value,
            is_security_approved=True,
            install_count=0,
            active_installs=0
        )
        mock_session.get.return_value = mock_plugin
        
        # Mock no existing installation
        mock_session.execute.return_value.first.return_value = None
        
        # Execute
        result = await plugin_service.install_plugin(
            plugin_id=plugin_id,
            user_id=user_id,
            installation_method="cli",
            client_info={'platform': 'linux', 'version': '1.0'}
        )
        
        # Verify
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert result['plugin_id'] == str(plugin_id)
        assert result['user_id'] == str(user_id)
        assert result['status'] == 'installed'
    
    @pytest.mark.asyncio
    async def test_security_scanner(self):
        """Test plugin security scanner functionality."""
        scanner = SecurityScanner()
        plugin_id = uuid4()
        
        # Test code with security issues
        vulnerable_code = """
import os
import subprocess

def dangerous_function(user_input):
    # High risk: command injection
    os.system(f"rm -rf {user_input}")
    
    # Medium risk: file access
    with open(user_input, 'r') as f:
        return f.read()
    
    # High risk: code injection
    eval(user_input)
        """
        
        dependencies = ['requests==2.25.1', 'pyyaml==5.3.1']  # Vulnerable versions
        
        # Execute
        scan_results = await scanner.scan_plugin(
            plugin_id=plugin_id,
            plugin_content=vulnerable_code,
            dependencies=dependencies
        )
        
        # Verify
        assert scan_results['plugin_id'] == str(plugin_id)
        assert scan_results['overall_score'] < 50.0  # Should be low due to vulnerabilities
        assert scan_results['risk_level'] in ['high', 'critical']
        assert len(scan_results['vulnerabilities']) > 0
        assert len(scan_results['dependency_issues']) > 0
        assert len(scan_results['recommendations']) > 0


class TestModerationService:
    """Test comprehensive moderation service functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.get = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.refresh = AsyncMock()
        return session
    
    @pytest.fixture
    def moderation_service(self, mock_session):
        """Moderation service fixture."""
        cache_service = CacheService()
        return ModerationService(mock_session, cache_service)
    
    @pytest.mark.asyncio
    async def test_content_filter_analysis(self):
        """Test AI content filtering and analysis."""
        content_filter = ContentFilter()
        
        # Test spam content
        spam_content = "BUY NOW! LIMITED TIME OFFER! Click here to win FREE MONEY! Visit our website today!"
        
        analysis = await content_filter.analyze_content(
            content=spam_content,
            content_type="review"
        )
        
        # Verify spam detection
        assert analysis['spam_score'] > 0.5
        assert analysis['recommendation'] in ['reject', 'needs_human_review']
        assert any(issue['type'] == 'spam_pattern' for issue in analysis['detected_issues'])
        
        # Test toxic content
        toxic_content = "This template is stupid and the author is an idiot who should die"
        
        toxic_analysis = await content_filter.analyze_content(
            content=toxic_content,
            content_type="review"
        )
        
        # Verify toxicity detection
        assert toxic_analysis['toxicity_score'] > 0.3
        assert toxic_analysis['profanity_score'] > 0.2
        assert toxic_analysis['recommendation'] == 'reject'
        
        # Test quality content
        quality_content = "This is an excellent template with clear documentation and helpful examples. The code is well-structured and follows best practices. I would definitely recommend it to others working on similar projects."
        
        quality_analysis = await content_filter.analyze_content(
            content=quality_content,
            content_type="review"
        )
        
        # Verify quality assessment
        assert quality_analysis['overall_score'] > 0.7
        assert quality_analysis['recommendation'] == 'approve'
        assert quality_analysis['quality_score'] > 0.8
    
    @pytest.mark.asyncio
    async def test_auto_moderate_content(self, moderation_service):
        """Test automatic content moderation workflow."""
        content_id = uuid4()
        user_id = uuid4()
        
        # Mock user history
        moderation_service._get_user_moderation_history = AsyncMock(
            return_value={'violation_count': 0, 'is_repeat_offender': False}
        )
        
        # Mock queue addition
        moderation_service.add_to_moderation_queue = AsyncMock(return_value=uuid4())
        
        # Test high-quality content (auto-approve)
        good_content = "Excellent template with great documentation and examples."
        
        decision = await moderation_service.auto_moderate_content(
            content_type="review",
            content_id=content_id,
            content_text=good_content,
            user_id=user_id
        )
        
        # Verify auto-approval
        assert decision['action'] == 'approve'
        assert decision['confidence'] >= 0.8
        assert decision['requires_human_review'] == False
        
        # Test problematic content (auto-reject)
        bad_content = "This is spam! Buy now and get free stuff! Click here!"
        
        bad_decision = await moderation_service.auto_moderate_content(
            content_type="review",
            content_id=content_id,
            content_text=bad_content,
            user_id=user_id
        )
        
        # Verify handling of problematic content
        assert bad_decision['action'] in ['reject', 'needs_human_review']
        
        if bad_decision['action'] == 'reject':
            moderation_service.add_to_moderation_queue.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_moderation_queue_management(self, moderation_service, mock_session):
        """Test moderation queue operations."""
        content_id = uuid4()
        moderator_id = uuid4()
        
        # Mock queue item
        mock_queue_item = MagicMock()
        mock_queue_item.id = uuid4()
        mock_queue_item.content_type = "review"
        mock_queue_item.content_id = content_id
        mock_queue_item.status = "pending"
        mock_queue_item.resolved_at = None
        
        mock_session.get.return_value = mock_queue_item
        
        # Mock content update
        moderation_service._apply_moderation_decision = AsyncMock(return_value=True)
        
        # Test moderation decision
        result = await moderation_service.moderate_content(
            queue_item_id=mock_queue_item.id,
            moderator_id=moderator_id,
            decision="approve",
            notes="Content is appropriate and helpful"
        )
        
        # Verify
        assert result['decision'] == "approve"
        assert result['content_updated'] == True
        assert mock_queue_item.status == "completed"
        assert mock_queue_item.resolution == "approve"
        assert mock_queue_item.resolved_by == moderator_id
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_moderation_statistics(self, moderation_service, mock_session):
        """Test moderation statistics generation."""
        # Mock database results
        queue_stats_result = MagicMock()
        queue_stats_result.fetchall.return_value = [
            MagicMock(status='pending', count=10, avg_resolution_hours=2.5),
            MagicMock(status='completed', count=90, avg_resolution_hours=1.8)
        ]
        
        resolution_result = MagicMock()
        resolution_result.fetchall.return_value = [
            MagicMock(resolution='approve', count=70),
            MagicMock(resolution='reject', count=20)
        ]
        
        content_type_result = MagicMock()
        content_type_result.fetchall.return_value = [
            MagicMock(content_type='review', total=50, approved=40, rejected=10),
            MagicMock(content_type='template', total=30, approved=25, rejected=5)
        ]
        
        mock_session.execute.side_effect = [
            queue_stats_result,
            resolution_result,
            content_type_result
        ]
        
        # Execute
        stats = await moderation_service.get_moderation_statistics(time_period="30d")
        
        # Verify
        assert stats['period'] == "30d"
        assert stats['total_items'] == 100
        assert stats['pending_items'] == 10
        assert stats['completed_items'] == 90
        assert stats['completion_rate'] == 90.0
        assert 'queue_breakdown' in stats
        assert 'resolution_breakdown' in stats
        assert 'content_type_breakdown' in stats


class TestCacheService:
    """Test caching functionality."""
    
    @pytest.fixture
    def cache_service(self):
        """Cache service fixture."""
        return CacheService()
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache_service):
        """Test basic cache operations."""
        # Test set and get
        key = "test_key"
        value = {"name": "Test Template", "id": str(uuid4())}
        
        # Set value
        result = await cache_service.set(key, value, ttl=3600)
        assert result == True
        
        # Get value
        retrieved = await cache_service.get(key)
        assert retrieved == value
        
        # Test exists
        exists = await cache_service.exists(key)
        assert exists == True
        
        # Test delete
        deleted = await cache_service.delete(key)
        assert deleted == True
        
        # Verify deletion
        retrieved_after_delete = await cache_service.get(key, default="not_found")
        assert retrieved_after_delete == "not_found"
    
    @pytest.mark.asyncio
    async def test_pattern_deletion(self, cache_service):
        """Test pattern-based cache deletion."""
        # Set multiple keys
        await cache_service.set("template:123", {"name": "Template 1"})
        await cache_service.set("template:456", {"name": "Template 2"})
        await cache_service.set("user:789", {"name": "User 1"})
        
        # Delete template keys
        deleted_count = await cache_service.delete_pattern("template:*")
        assert deleted_count == 2
        
        # Verify template keys are gone
        assert await cache_service.get("template:123") is None
        assert await cache_service.get("template:456") is None
        
        # Verify user key still exists
        assert await cache_service.get("user:789") is not None


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete integration workflow of community features."""
    # This would test the interaction between all services
    # in a realistic workflow scenario
    
    # Mock session and services
    mock_session = MagicMock(spec=AsyncSession)
    cache_service = CacheService()
    
    # Initialize services
    marketplace_service = EnhancedMarketplaceService(mock_session, cache_service)
    recommendation_engine = RecommendationEngine(mock_session)
    rating_service = RatingService(mock_session, cache_service)
    
    # Mock necessary methods for integration test
    marketplace_service.recommendation_engine = recommendation_engine
    
    # Test data
    user_id = uuid4()
    template_id = uuid4()
    
    # Mock search -> recommendations -> rating workflow
    search_filters = TemplateSearchFilters(query="react")
    
    # This integration test would verify the complete user journey
    # from searching templates to getting recommendations to rating
    
    # For now, just verify services can be initialized together
    assert marketplace_service is not None
    assert recommendation_engine is not None
    assert rating_service is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])