"""
Comprehensive test suite for AI Learning and Personalization features.

This test suite validates all components of the AI learning system including
pattern recognition, personalization, federated learning, analytics, and
privacy-preserving mechanisms.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4

from src.ai.learning.pattern_engine import (
    PatternRecognitionEngine, 
    UserInteractionPattern,
    DevelopmentSuccessPattern
)
from src.ai.learning.personalization import (
    PersonalizedAIBehavior,
    PersonalizationProfile,
    PersonalizedPromptTemplate
)
from src.ai.learning.federated import (
    FederatedLearningSystem,
    PrivacyPreservingPattern,
    FederatedLearningCoordinator
)
from src.ai.learning.analytics import (
    LearningAnalytics,
    LearningMetrics,
    PatternAnalysis
)
from src.ai.learning.privacy import (
    PrivacyPreservingLearning,
    DifferentialPrivacyEngine,
    AnonymizationEngine
)
from src.ai.learning.integration import (
    IntelligentSystemIntegration,
    EnhancedAIInterface,
    EnhancedTaskEngine
)
from src.core.types import Task, AITaskResult, ValidationResult
from src.core.ai_interface import AIContext


@pytest.fixture
def sample_user_patterns():
    """Create sample user interaction patterns for testing."""
    patterns = []
    
    for i in range(10):
        pattern = UserInteractionPattern(
            pattern_id=f"pattern_{i}",
            user_id="test_user",
            pattern_type="success" if i % 2 == 0 else "failure",
            features={
                "execution_time": 10.0 + i,
                "task_type": "code_generation",
                "success_rate": 0.8 if i % 2 == 0 else 0.4,
                "confidence": 0.7 + (i * 0.02)
            },
            success_rate=0.8 if i % 2 == 0 else 0.4,
            frequency=i + 1,
            confidence=0.7 + (i * 0.02),
            timestamp=datetime.utcnow() - timedelta(days=i)
        )
        patterns.append(pattern)
    
    return patterns


@pytest.fixture
def sample_task():
    """Create sample task for testing."""
    return Task(
        id=uuid4(),
        name="Test Task",
        description="Test task description",
        ai_prompt="Generate test code"
    )


@pytest.fixture
def sample_ai_result():
    """Create sample AI task result."""
    return AITaskResult(
        task_id=uuid4(),
        success=True,
        generated_content="print('Hello, World!')",
        files_modified=["test.py"],
        validation_score=85.0,
        execution_time=5.2
    )


@pytest.fixture
def sample_validation_result():
    """Create sample validation result."""
    return ValidationResult(
        is_authentic=True,
        authenticity_score=87.5,
        real_progress=85.0,
        fake_progress=15.0,
        issues=[],
        suggestions=["Consider adding error handling"],
        next_actions=["Run tests"]
    )


class TestPatternRecognitionEngine:
    """Test pattern recognition engine functionality."""
    
    @pytest.fixture
    def pattern_engine(self):
        """Create pattern recognition engine for testing."""
        return PatternRecognitionEngine(
            learning_rate=0.1,
            pattern_confidence_threshold=0.7,
            enable_clustering=True
        )
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self, pattern_engine, sample_task, sample_ai_result):
        """Test learning from user interaction."""
        user_id = "test_user"
        context = {"task_type": "code_generation", "framework": "python"}
        
        pattern = await pattern_engine.learn_from_interaction(
            user_id, sample_task, sample_ai_result, context
        )
        
        assert pattern is not None
        assert pattern.user_id == user_id
        assert pattern.pattern_type in ["success", "failure"]
        assert user_id in pattern_engine._user_patterns
        assert len(pattern_engine._user_patterns[user_id]) == 1
    
    @pytest.mark.asyncio
    async def test_learn_from_validation_feedback(self, pattern_engine, sample_ai_result, sample_validation_result):
        """Test learning from validation feedback."""
        user_id = "test_user"
        feedback_context = {"validation_type": "code_quality"}
        
        await pattern_engine.learn_from_validation_feedback(
            user_id, sample_validation_result, sample_ai_result, feedback_context
        )
        
        assert user_id in pattern_engine._user_patterns
        patterns = pattern_engine._user_patterns[user_id]
        assert len(patterns) == 1
        assert patterns[0].pattern_type in ["validation_success", "validation_failure"]
    
    @pytest.mark.asyncio
    async def test_identify_success_patterns(self, pattern_engine, sample_user_patterns):
        """Test success pattern identification."""
        # Add patterns to engine
        pattern_engine._user_patterns["test_user"] = sample_user_patterns
        
        success_patterns = await pattern_engine.identify_success_patterns(
            user_id="test_user",
            min_frequency=1
        )
        
        assert len(success_patterns) > 0
        for pattern in success_patterns:
            assert isinstance(pattern, DevelopmentSuccessPattern)
            assert pattern.effectiveness_score > 0
    
    @pytest.mark.asyncio
    async def test_predict_task_success_probability(self, pattern_engine, sample_user_patterns, sample_task):
        """Test task success prediction."""
        # Add patterns to engine
        pattern_engine._user_patterns["test_user"] = sample_user_patterns
        
        success_prob, factors = await pattern_engine.predict_task_success_probability(
            "test_user", sample_task, {"task_type": "code_generation"}
        )
        
        assert 0.0 <= success_prob <= 1.0
        assert isinstance(factors, list)
        assert len(factors) > 0
    
    @pytest.mark.asyncio
    async def test_generate_personalized_recommendations(self, pattern_engine, sample_user_patterns):
        """Test personalized recommendation generation."""
        # Add patterns to engine
        pattern_engine._user_patterns["test_user"] = sample_user_patterns
        
        recommendations = await pattern_engine.generate_personalized_recommendations(
            "test_user", {"task_type": "code_generation"}
        )
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert "type" in rec
            assert "message" in rec
            assert "confidence" in rec


class TestPersonalizedAIBehavior:
    """Test personalized AI behavior functionality."""
    
    @pytest.fixture
    def personalized_behavior(self):
        """Create personalized AI behavior system for testing."""
        pattern_engine = PatternRecognitionEngine()
        return PersonalizedAIBehavior(pattern_engine, adaptation_rate=0.1)
    
    @pytest.mark.asyncio
    async def test_get_or_create_profile(self, personalized_behavior):
        """Test profile creation and retrieval."""
        user_id = "test_user"
        
        # First call should create profile
        profile1 = await personalized_behavior.get_or_create_profile(user_id)
        assert profile1.user_id == user_id
        assert isinstance(profile1.creation_date, datetime)
        
        # Second call should return same profile
        profile2 = await personalized_behavior.get_or_create_profile(user_id)
        assert profile1.user_id == profile2.user_id
        assert profile1.creation_date == profile2.creation_date
    
    @pytest.mark.asyncio
    async def test_personalize_ai_request(self, personalized_behavior, sample_task):
        """Test AI request personalization."""
        from src.core.ai_interface import ClaudeCodeRequest, AIContext
        
        user_id = "test_user"
        context = AIContext()
        base_request = ClaudeCodeRequest(
            prompt="Generate code",
            context=context
        )
        
        personalized_request = await personalized_behavior.personalize_ai_request(
            user_id, base_request, sample_task, {"task_type": "code_generation"}
        )
        
        assert personalized_request is not None
        assert isinstance(personalized_request, ClaudeCodeRequest)
        # Prompt should be modified (personalized)
        assert len(personalized_request.prompt) >= len(base_request.prompt)
    
    @pytest.mark.asyncio
    async def test_adapt_validation_criteria(self, personalized_behavior):
        """Test validation criteria adaptation."""
        user_id = "test_user"
        base_criteria = {"authenticity_threshold": 0.8}
        context = {"task_type": "code_review"}
        
        adapted_criteria = await personalized_behavior.adapt_validation_criteria(
            user_id, base_criteria, context
        )
        
        assert "authenticity_threshold" in adapted_criteria
        assert isinstance(adapted_criteria["authenticity_threshold"], float)
    
    @pytest.mark.asyncio
    async def test_learn_from_user_feedback(self, personalized_behavior, sample_task, sample_ai_result, sample_validation_result):
        """Test learning from user feedback."""
        user_id = "test_user"
        user_feedback = {
            "satisfaction": 0.8,
            "type": "positive",
            "comments": "Good result, but could be more detailed"
        }
        
        await personalized_behavior.learn_from_user_feedback(
            user_id, sample_task, sample_ai_result, sample_validation_result, user_feedback
        )
        
        # Profile should be created and updated
        assert user_id in personalized_behavior._user_profiles
        profile = personalized_behavior._user_profiles[user_id]
        assert profile.last_updated is not None
    
    @pytest.mark.asyncio
    async def test_get_personalized_suggestions(self, personalized_behavior):
        """Test personalized suggestion generation."""
        user_id = "test_user"
        context = {"task_type": "code_generation", "framework": "python"}
        
        suggestions = await personalized_behavior.get_personalized_suggestions(
            user_id, context
        )
        
        assert isinstance(suggestions, list)
        # Should have at least one suggestion
        assert len(suggestions) >= 1
    
    @pytest.mark.asyncio
    async def test_measure_personalization_effectiveness(self, personalized_behavior):
        """Test personalization effectiveness measurement."""
        user_id = "test_user"
        
        # Create a profile first
        await personalized_behavior.get_or_create_profile(user_id)
        
        effectiveness = await personalized_behavior.measure_personalization_effectiveness(
            user_id, time_window_days=30
        )
        
        assert isinstance(effectiveness, dict)
        assert "profile_age_days" in effectiveness


class TestFederatedLearningSystem:
    """Test federated learning system functionality."""
    
    @pytest.fixture
    def federated_system(self):
        """Create federated learning system for testing."""
        pattern_engine = PatternRecognitionEngine()
        return FederatedLearningSystem(
            pattern_engine,
            node_id="test_node",
            organization="test_org"
        )
    
    @pytest.mark.asyncio
    async def test_enable_federated_learning(self, federated_system):
        """Test enabling federated learning."""
        config = {
            "endpoint": "localhost:8080",
            "credentials": {"api_key": "test_key"}
        }
        
        # Mock the network connection
        with patch.object(federated_system.coordinator, 'join_federation', return_value=True):
            result = await federated_system.enable_federated_learning(config)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_contribute_to_collective_learning(self, federated_system, sample_user_patterns):
        """Test contributing patterns to collective learning."""
        # Add patterns to the pattern engine
        federated_system.pattern_engine._user_patterns["test_user"] = sample_user_patterns
        federated_system._learning_active = True
        
        with patch.object(federated_system.coordinator, 'contribute_patterns') as mock_contribute:
            mock_contribute.return_value = {
                'patterns_contributed': 5,
                'patterns_rejected': 0
            }
            
            result = await federated_system.contribute_to_collective_learning(
                learning_objectives=["pattern_sharing"]
            )
            
            assert "patterns_contributed" in result
            assert result["patterns_contributed"] >= 0
    
    @pytest.mark.asyncio
    async def test_measure_federated_impact(self, federated_system):
        """Test measuring federated learning impact."""
        # Enable learning first
        federated_system._learning_active = True
        
        impact_metrics = await federated_system.measure_federated_impact(
            time_window_days=30
        )
        
        assert isinstance(impact_metrics, dict)
        assert "federation_analytics" in impact_metrics
        assert "local_vs_federated" in impact_metrics


class TestLearningAnalytics:
    """Test learning analytics functionality."""
    
    @pytest.fixture
    def learning_analytics(self):
        """Create learning analytics system for testing."""
        pattern_engine = PatternRecognitionEngine()
        personalized_behavior = PersonalizedAIBehavior(pattern_engine)
        return LearningAnalytics(pattern_engine, personalized_behavior)
    
    @pytest.mark.asyncio
    async def test_generate_user_analytics(self, learning_analytics, sample_user_patterns):
        """Test user analytics generation."""
        user_id = "test_user"
        learning_analytics.pattern_engine._user_patterns[user_id] = sample_user_patterns
        
        analytics = await learning_analytics.generate_user_analytics(
            user_id, time_range_days=30
        )
        
        assert isinstance(analytics, dict)
        assert "user_id" in analytics
        assert "core_metrics" in analytics
        assert "learning_progression" in analytics
        assert "pattern_analysis" in analytics
        assert analytics["user_id"] == user_id
    
    @pytest.mark.asyncio
    async def test_generate_team_analytics(self, learning_analytics, sample_user_patterns):
        """Test team analytics generation."""
        team_users = ["user1", "user2", "user3"]
        
        # Add patterns for each user
        for user_id in team_users:
            learning_analytics.pattern_engine._user_patterns[user_id] = sample_user_patterns[:5]
        
        team_analytics = await learning_analytics.generate_team_analytics(
            team_users, time_range_days=30
        )
        
        assert isinstance(team_analytics, dict)
        assert "team_composition" in team_analytics
        assert "aggregated_metrics" in team_analytics
        assert team_analytics["team_composition"]["user_count"] == len(team_users)
    
    @pytest.mark.asyncio
    async def test_generate_learning_insights(self, learning_analytics):
        """Test learning insights generation."""
        insights = await learning_analytics.generate_learning_insights(
            scope="all",
            focus_areas=["patterns", "trends"]
        )
        
        assert isinstance(insights, list)
        # Should generate some insights even with minimal data
        for insight in insights:
            assert hasattr(insight, 'insight_id')
            assert hasattr(insight, 'insight_type')
            assert hasattr(insight, 'description')
    
    @pytest.mark.asyncio
    async def test_create_learning_dashboard(self, learning_analytics, sample_user_patterns):
        """Test learning dashboard creation."""
        import tempfile
        from pathlib import Path
        
        user_id = "test_user"
        learning_analytics.pattern_engine._user_patterns[user_id] = sample_user_patterns
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            dashboard = await learning_analytics.create_learning_dashboard(
                user_id=user_id,
                output_path=output_path
            )
            
            assert isinstance(dashboard, dict)
            assert "dashboard_data" in dashboard
            assert "html_dashboard_path" in dashboard
            assert Path(dashboard["html_dashboard_path"]).exists()


class TestPrivacyPreservingLearning:
    """Test privacy-preserving learning functionality."""
    
    @pytest.fixture
    def privacy_system(self):
        """Create privacy-preserving learning system for testing."""
        return PrivacyPreservingLearning(
            differential_privacy_epsilon=1.0
        )
    
    @pytest.mark.asyncio
    async def test_privatize_user_patterns(self, privacy_system, sample_user_patterns):
        """Test user pattern privatization."""
        user_id = "test_user"
        
        # Mock consent check
        privacy_system._consent_records[user_id] = {"learning": True}
        
        result = await privacy_system.privatize_user_patterns(
            user_id, sample_user_patterns, purpose="learning"
        )
        
        assert isinstance(result, dict)
        assert "privatized_patterns" in result
        assert "privacy_metadata" in result
        assert "privacy_guarantees" in result
    
    @pytest.mark.asyncio
    async def test_secure_federated_aggregation(self, privacy_system):
        """Test secure federated aggregation."""
        local_metrics = {
            "success_rate": 0.8,
            "learning_velocity": 0.6,
            "pattern_count": 10
        }
        
        result = await privacy_system.secure_federated_aggregation(
            local_metrics, purpose="collaborative_learning"
        )
        
        assert isinstance(result, dict)
        assert "encrypted_metrics" in result
        assert "privacy_metadata" in result
        assert len(result["encrypted_metrics"]) == len(local_metrics)
    
    @pytest.mark.asyncio
    async def test_privacy_preserving_analytics(self, privacy_system):
        """Test privacy-preserving analytics."""
        data = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
            {"value": 40}
        ]
        
        result = await privacy_system.privacy_preserving_analytics(
            data, query_type="average", sensitivity=1.0
        )
        
        assert isinstance(result, dict)
        assert "result" in result
        assert "privacy_metadata" in result
        assert isinstance(result["result"], float)
    
    def test_differential_privacy_engine(self, privacy_system):
        """Test differential privacy engine."""
        dp_engine = privacy_system.dp_engine
        
        # Test noise addition
        original_value = 100.0
        noisy_value = dp_engine.add_laplace_noise(original_value, sensitivity=1.0)
        
        assert isinstance(noisy_value, float)
        assert noisy_value != original_value  # Should be different due to noise
        
        # Test privacy budget consumption
        initial_budget = dp_engine.get_remaining_budget()
        consumed = dp_engine.consume_privacy_budget(0.1)
        
        assert consumed is True
        assert dp_engine.get_remaining_budget() < initial_budget
    
    def test_anonymization_engine(self, privacy_system):
        """Test anonymization engine."""
        anonymizer = privacy_system.anonymization_engine
        
        # Test pattern anonymization
        from src.ai.learning.pattern_engine import UserInteractionPattern
        
        pattern = UserInteractionPattern(
            pattern_id="test",
            user_id="user123",
            pattern_type="success",
            features={"execution_time": 10.5, "task_type": "coding"},
            success_rate=0.8,
            frequency=5,
            confidence=0.7,
            timestamp=datetime.utcnow()
        )
        
        anonymized = anonymizer.apply_semantic_anonymization([pattern])
        
        assert len(anonymized) == 1
        assert "pattern_hash" in anonymized[0]
        assert "timestamp_bin" in anonymized[0]
        assert "user_id" not in str(anonymized[0])  # Should not contain original user ID


class TestIntelligentSystemIntegration:
    """Test intelligent system integration functionality."""
    
    @pytest.fixture
    def integration_system(self):
        """Create intelligent system integration for testing."""
        config = {
            "learning_rate": 0.1,
            "confidence_threshold": 0.7,
            "enable_federated_learning": False,
            "privacy_epsilon": 1.0
        }
        return IntelligentSystemIntegration(config)
    
    @pytest.mark.asyncio
    async def test_execute_intelligent_development_task(self, integration_system):
        """Test intelligent development task execution."""
        task_description = "Create a simple Python function"
        user_id = "test_user"
        project_context = {
            "project_path": "/test/project",
            "framework_info": {"language": "python"},
            "dependencies": ["pytest"]
        }
        
        # Mock the AI interface execution
        with patch.object(integration_system.ai_interface, 'execute_intelligent_task') as mock_execute:
            mock_ai_result = {
                "ai_result": sample_ai_result,
                "recommendations": [],
                "learning_metadata": {"personalized": True}
            }
            mock_execute.return_value = mock_ai_result
            
            # Mock validation
            with patch.object(integration_system.validation_service, 'validate_with_learning') as mock_validate:
                mock_validate.return_value = {"score": 0.85, "is_valid": True}
                
                result = await integration_system.execute_intelligent_development_task(
                    task_description,
                    user_id,
                    project_context,
                    learn_from_execution=True
                )
                
                assert isinstance(result, dict)
                assert "task_execution" in result
                assert "validation_result" in result
                assert "learning_insights" in result
                assert "execution_metadata" in result
    
    @pytest.mark.asyncio
    async def test_get_system_health(self, integration_system):
        """Test system health monitoring."""
        health_status = await integration_system.get_system_health()
        
        assert isinstance(health_status, dict)
        assert "overall_status" in health_status
        assert "components" in health_status
        assert "timestamp" in health_status
        
        # Check component health
        components = health_status["components"]
        assert "pattern_engine" in components
        assert "analytics" in components
        assert "validation_service" in components
    
    def test_get_learning_components(self, integration_system):
        """Test learning components access."""
        components = integration_system.get_learning_components()
        
        assert isinstance(components, dict)
        assert "pattern_engine" in components
        assert "personalized_behavior" in components
        assert "privacy_system" in components
        assert "analytics" in components
        assert "ai_interface" in components
        assert "task_engine" in components
        assert "validation_service" in components


# Integration tests
class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.fixture
    def complete_system(self):
        """Create complete system for integration testing."""
        return IntelligentSystemIntegration({
            "learning_rate": 0.1,
            "confidence_threshold": 0.7,
            "enable_federated_learning": False
        })
    
    @pytest.mark.asyncio
    async def test_complete_learning_cycle(self, complete_system):
        """Test complete learning cycle from interaction to personalization."""
        user_id = "integration_test_user"
        
        # Step 1: Execute task and learn
        task_description = "Write a sorting algorithm"
        project_context = {"framework_info": {"language": "python"}}
        
        with patch.object(complete_system.ai_interface, 'execute_intelligent_task') as mock_ai:
            mock_ai.return_value = {
                "ai_result": AITaskResult(
                    task_id=uuid4(),
                    success=True,
                    generated_content="def sort_list(arr): return sorted(arr)",
                    execution_time=3.0
                ),
                "learning_metadata": {"personalized": True}
            }
            
            with patch.object(complete_system.validation_service, 'validate_with_learning') as mock_val:
                mock_val.return_value = {"score": 0.9, "is_valid": True}
                
                # Execute task
                result1 = await complete_system.execute_intelligent_development_task(
                    task_description, user_id, project_context
                )
                
                assert result1 is not None
                assert "task_execution" in result1
        
        # Step 2: Verify learning occurred
        user_patterns = complete_system.pattern_engine._user_patterns.get(user_id, [])
        assert len(user_patterns) > 0
        
        # Step 3: Check personalization
        profile = await complete_system.personalized_behavior.get_or_create_profile(user_id)
        assert profile is not None
        assert profile.user_id == user_id
        
        # Step 4: Generate analytics
        analytics = await complete_system.analytics.generate_user_analytics(user_id)
        assert analytics is not None
        assert analytics["user_id"] == user_id
    
    @pytest.mark.asyncio
    async def test_privacy_preserved_learning(self, complete_system):
        """Test that learning respects privacy constraints."""
        user_id = "privacy_test_user"
        
        # Add some patterns
        patterns = [
            UserInteractionPattern(
                pattern_id=f"pattern_{i}",
                user_id=user_id,
                pattern_type="success",
                features={"execution_time": 5.0 + i, "task_type": "coding"},
                success_rate=0.8,
                frequency=1,
                confidence=0.7,
                timestamp=datetime.utcnow()
            )
            for i in range(5)
        ]
        
        complete_system.pattern_engine._user_patterns[user_id] = patterns
        
        # Test privacy-preserving pattern sharing
        result = await complete_system.privacy_system.privatize_user_patterns(
            user_id, patterns, purpose="learning"
        )
        
        assert "privatized_patterns" in result
        assert "privacy_guarantees" in result
        
        # Verify patterns are anonymized
        privatized_patterns = result["privatized_patterns"]
        for pattern in privatized_patterns:
            # Should not contain original user ID in any form
            pattern_str = str(pattern).lower()
            assert user_id.lower() not in pattern_str
    
    @pytest.mark.asyncio 
    async def test_federated_learning_simulation(self, complete_system):
        """Test federated learning capabilities."""
        if complete_system.federated_system is None:
            pytest.skip("Federated learning not enabled in test configuration")
        
        user_id = "federated_test_user"
        
        # Add patterns to simulate local learning
        patterns = [
            UserInteractionPattern(
                pattern_id=f"fed_pattern_{i}",
                user_id=user_id,
                pattern_type="success",
                features={"task_type": "algorithm", "complexity": i},
                success_rate=0.85,
                frequency=2,
                confidence=0.8,
                timestamp=datetime.utcnow()
            )
            for i in range(3)
        ]
        
        complete_system.pattern_engine._user_patterns[user_id] = patterns
        
        # Simulate contribution to federated learning
        with patch.object(complete_system.federated_system.coordinator, 'contribute_patterns') as mock_contribute:
            mock_contribute.return_value = {
                'patterns_contributed': len(patterns),
                'patterns_rejected': 0
            }
            
            contribution_result = await complete_system.federated_system.contribute_to_collective_learning()
            
            assert contribution_result['patterns_contributed'] > 0
        
        # Test receiving federated insights
        with patch.object(complete_system.federated_system.coordinator, 'receive_federated_insights') as mock_receive:
            from src.ai.learning.federated import CollaborativeInsight
            
            mock_insights = [
                CollaborativeInsight(
                    insight_id="test_insight",
                    insight_type="best_practice",
                    description="Test collaborative insight",
                    supporting_evidence=[],
                    confidence=0.8,
                    applicability=["algorithm_tasks"],
                    learned_from_nodes=3,
                    privacy_safe=True,
                    actionable_recommendations=["Use efficient algorithms"]
                )
            ]
            mock_receive.return_value = mock_insights
            
            insights = await complete_system.federated_system.receive_collective_insights()
            
            assert len(insights) > 0
            assert insights[0].insight_type == "best_practice"


# Performance and stress tests
class TestPerformance:
    """Test system performance under load."""
    
    @pytest.mark.asyncio
    async def test_pattern_engine_scalability(self):
        """Test pattern engine performance with large datasets."""
        pattern_engine = PatternRecognitionEngine()
        
        # Create many users with many patterns each
        num_users = 10
        patterns_per_user = 100
        
        start_time = datetime.utcnow()
        
        for user_idx in range(num_users):
            user_id = f"perf_user_{user_idx}"
            patterns = []
            
            for pattern_idx in range(patterns_per_user):
                pattern = UserInteractionPattern(
                    pattern_id=f"pattern_{user_idx}_{pattern_idx}",
                    user_id=user_id,
                    pattern_type="success" if pattern_idx % 2 == 0 else "failure",
                    features={
                        "execution_time": float(pattern_idx),
                        "task_complexity": pattern_idx % 5,
                        "success_rate": 0.7 + (pattern_idx % 30) / 100
                    },
                    success_rate=0.7 + (pattern_idx % 30) / 100,
                    frequency=pattern_idx % 10 + 1,
                    confidence=0.6 + (pattern_idx % 40) / 100,
                    timestamp=datetime.utcnow() - timedelta(days=pattern_idx)
                )
                patterns.append(pattern)
            
            pattern_engine._user_patterns[user_id] = patterns
        
        # Test analytics generation performance
        analytics = await pattern_engine.get_pattern_analytics()
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 seconds max
        assert analytics['total_patterns'] == num_users * patterns_per_user
        assert analytics['users'] == num_users
    
    @pytest.mark.asyncio
    async def test_privacy_system_performance(self):
        """Test privacy system performance with differential privacy."""
        privacy_system = PrivacyPreservingLearning(differential_privacy_epsilon=1.0)
        
        # Test many privacy operations
        num_operations = 100
        
        start_time = datetime.utcnow()
        
        for i in range(num_operations):
            data = [{"value": j} for j in range(10)]
            result = await privacy_system.privacy_preserving_analytics(
                data, query_type="average", sensitivity=1.0
            )
            assert "result" in result
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 seconds max for 100 operations
    
    @pytest.mark.asyncio
    async def test_concurrent_learning_operations(self):
        """Test concurrent learning operations."""
        integration_system = IntelligentSystemIntegration()
        
        # Create multiple concurrent tasks
        num_concurrent_tasks = 5
        tasks = []
        
        for i in range(num_concurrent_tasks):
            task_coro = integration_system.execute_intelligent_development_task(
                f"Task {i} description",
                f"concurrent_user_{i}",
                {"framework_info": {"language": "python"}},
                learn_from_execution=True
            )
            tasks.append(task_coro)
        
        # Mock the underlying AI and validation calls
        with patch.object(integration_system.ai_interface, 'execute_intelligent_task') as mock_ai:
            with patch.object(integration_system.validation_service, 'validate_with_learning') as mock_val:
                mock_ai.return_value = {
                    "ai_result": AITaskResult(task_id=uuid4(), success=True, generated_content="test", execution_time=1.0),
                    "learning_metadata": {"personalized": True}
                }
                mock_val.return_value = {"score": 0.8, "is_valid": True}
                
                start_time = datetime.utcnow()
                
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # All tasks should complete successfully
                assert len(results) == num_concurrent_tasks
                for result in results:
                    assert "task_execution" in result
                
                # Should be faster than sequential execution
                assert execution_time < 10.0  # Should complete quickly with mocking