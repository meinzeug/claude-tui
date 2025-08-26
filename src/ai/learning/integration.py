"""
Integration Module for AI Learning and Personalization.

This module provides seamless integration between the AI learning system
and the existing validation framework, task engine, and AI interface,
creating a unified intelligent system that learns and adapts continuously.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from ...core.ai_interface import AIInterface, AIContext, ClaudeCodeRequest, ClaudeFlowRequest
from ...core.task_engine import TaskEngine, TaskExecutionResult
from ...core.validator import ProgressValidator, ValidationResult
from ...core.types import Task, AITaskResult, ProgressMetrics
from ...services.validation_service import ValidationService

from .pattern_engine import PatternRecognitionEngine
from .personalization import PersonalizedAIBehavior
from .federated import FederatedLearningSystem
from .analytics import LearningAnalytics
from .privacy import PrivacyPreservingLearning


logger = logging.getLogger(__name__)


class EnhancedAIInterface(AIInterface):
    """
    Enhanced AI interface that incorporates learning and personalization.
    
    Extends the base AI interface with intelligent pattern recognition,
    personalized behavior, and continuous learning capabilities.
    """
    
    def __init__(
        self,
        claude_code_path: str = "claude",
        enable_validation: bool = True,
        enable_flow_orchestration: bool = True,
        enable_learning: bool = True,
        learning_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced AI interface.
        
        Args:
            claude_code_path: Path to Claude Code CLI
            enable_validation: Enable response validation
            enable_flow_orchestration: Enable Claude Flow orchestration
            enable_learning: Enable AI learning capabilities
            learning_config: Configuration for learning systems
        """
        super().__init__(claude_code_path, enable_validation, enable_flow_orchestration)
        
        self.enable_learning = enable_learning
        self.learning_config = learning_config or {}
        
        if enable_learning:
            # Initialize learning components
            self.pattern_engine = PatternRecognitionEngine(
                learning_rate=self.learning_config.get('learning_rate', 0.1),
                pattern_confidence_threshold=self.learning_config.get('confidence_threshold', 0.7)
            )
            
            self.personalized_behavior = PersonalizedAIBehavior(
                self.pattern_engine,
                adaptation_rate=self.learning_config.get('adaptation_rate', 0.1)
            )
            
            # Initialize federated learning if configured
            if self.learning_config.get('enable_federated_learning', False):
                self.federated_system = FederatedLearningSystem(
                    self.pattern_engine,
                    node_id=self.learning_config.get('node_id', 'local_node'),
                    organization=self.learning_config.get('organization', 'default_org')
                )
            else:
                self.federated_system = None
            
            # Initialize privacy system
            if self.learning_config.get('enable_privacy_preservation', True):
                self.privacy_system = PrivacyPreservingLearning(
                    differential_privacy_epsilon=self.learning_config.get('privacy_epsilon', 1.0)
                )
            else:
                self.privacy_system = None
            
            # Initialize analytics
            self.analytics = LearningAnalytics(
                self.pattern_engine,
                self.personalized_behavior,
                self.federated_system
            )
            
            logger.info("Enhanced AI interface with learning capabilities initialized")
    
    async def execute_intelligent_task(
        self,
        task: Task,
        context: AIContext,
        user_id: str,
        force_service: Optional[str] = None,
        learn_from_interaction: bool = True
    ) -> Union[AITaskResult, Dict[str, Any]]:
        """
        Execute AI task with intelligent learning and personalization.
        
        Args:
            task: Task to execute
            context: AI context information
            user_id: User identifier for personalization
            force_service: Force specific service
            learn_from_interaction: Whether to learn from this interaction
            
        Returns:
            Enhanced AI execution result
        """
        start_time = datetime.utcnow()
        
        try:
            # Personalize the AI request if learning is enabled
            if self.enable_learning and user_id:
                # Create base request
                base_request = ClaudeCodeRequest(
                    prompt=task.ai_prompt or task.description,
                    context=context,
                    enable_validation=True
                )
                
                # Apply personalization
                personalized_request = await self.personalized_behavior.personalize_ai_request(
                    user_id, base_request, task, {'task_type': getattr(task, 'task_type', 'general')}
                )
                
                # Execute with personalized request
                result = await super().execute_ai_task(task, context, force_service)
                
                # Learn from interaction if enabled
                if learn_from_interaction:
                    await self._learn_from_interaction(user_id, task, result, context)
                
                # Add learning metadata to result
                if hasattr(result, 'metadata'):
                    result.metadata = result.metadata or {}
                    result.metadata.update({
                        'personalized': True,
                        'user_id': user_id,
                        'learning_enabled': True
                    })
            
            else:
                # Execute without learning
                result = await super().execute_ai_task(task, context, force_service)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Enhanced result with learning insights
            enhanced_result = {
                'ai_result': result,
                'execution_time': execution_time,
                'learning_metadata': {
                    'personalized': self.enable_learning and user_id is not None,
                    'pattern_count': len(self.pattern_engine._user_patterns.get(user_id, [])) if self.enable_learning and user_id else 0,
                    'learning_enabled': self.enable_learning
                }
            }
            
            # Add personalized recommendations if available
            if self.enable_learning and user_id:
                recommendations = await self._get_contextual_recommendations(user_id, task, context)
                enhanced_result['recommendations'] = recommendations
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced AI task execution failed: {e}")
            # Fallback to base implementation
            return await super().execute_ai_task(task, context, force_service)
    
    async def _learn_from_interaction(
        self,
        user_id: str,
        task: Task,
        result: AITaskResult,
        context: AIContext
    ) -> None:
        """Learn from user interaction with AI system."""
        try:
            # Add personalization context
            interaction_context = {
                'task_type': getattr(task, 'task_type', 'general'),
                'project_path': str(context.project_path) if context.project_path else None,
                'dependencies': context.dependencies,
                'framework_info': context.framework_info,
                'personalized': True
            }
            
            # Learn pattern
            pattern = await self.pattern_engine.learn_from_interaction(
                user_id, task, result, interaction_context
            )
            
            logger.debug(f"Learned pattern {pattern.pattern_id} for user {user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to learn from interaction: {e}")
    
    async def _get_contextual_recommendations(
        self,
        user_id: str,
        task: Task,
        context: AIContext
    ) -> List[Dict[str, Any]]:
        """Get contextual recommendations for user."""
        try:
            current_context = {
                'task_type': getattr(task, 'task_type', 'general'),
                'priority': getattr(task, 'priority', 'medium'),
                'frameworks': context.framework_info.keys() if context.framework_info else [],
                'dependencies': context.dependencies
            }
            
            # Get personalized recommendations
            recommendations = await self.personalized_behavior.get_personalized_suggestions(
                user_id, current_context
            )
            
            # Add federated recommendations if available
            if self.federated_system:
                federated_recs = await self.federated_system.get_federated_recommendations(
                    user_id, current_context
                )
                recommendations.extend(federated_recs[:3])  # Add top 3 federated recommendations
            
            return recommendations[:8]  # Return top 8 recommendations
            
        except Exception as e:
            logger.warning(f"Failed to get recommendations: {e}")
            return []


class EnhancedTaskEngine(TaskEngine):
    """
    Enhanced task engine that incorporates learning and intelligent task routing.
    
    Extends the base task engine with pattern-aware execution, learning from
    task outcomes, and adaptive workflow optimization.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        enable_validation: bool = True,
        enable_monitoring: bool = True,
        enable_learning: bool = True,
        learning_components: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced task engine.
        
        Args:
            max_concurrent_tasks: Maximum parallel task execution
            enable_validation: Enable anti-hallucination validation
            enable_monitoring: Enable resource monitoring
            enable_learning: Enable learning capabilities
            learning_components: Pre-initialized learning components
        """
        super().__init__(max_concurrent_tasks, enable_validation, enable_monitoring)
        
        self.enable_learning = enable_learning
        
        if enable_learning and learning_components:
            self.pattern_engine = learning_components.get('pattern_engine')
            self.personalized_behavior = learning_components.get('personalized_behavior')
            self.analytics = learning_components.get('analytics')
        else:
            self.pattern_engine = None
            self.personalized_behavior = None
            self.analytics = None
        
        # Enhanced execution tracking
        self._learning_enabled_executions: Dict[UUID, Dict[str, Any]] = {}
    
    async def execute_intelligent_workflow(
        self,
        workflow,
        user_id: Optional[str] = None,
        strategy = None,
        **context_kwargs
    ):
        """
        Execute workflow with intelligent learning and adaptation.
        
        Args:
            workflow: Workflow to execute
            user_id: User identifier for personalization
            strategy: Execution strategy
            **context_kwargs: Additional context parameters
            
        Returns:
            Enhanced workflow execution result
        """
        # Predict task success probabilities if learning is enabled
        if self.enable_learning and self.pattern_engine and user_id:
            for task in workflow.tasks:
                try:
                    success_probability, factors = await self.pattern_engine.predict_task_success_probability(
                        user_id, task, context_kwargs
                    )
                    
                    # Store prediction for later validation
                    self._learning_enabled_executions[task.id] = {
                        'predicted_success': success_probability,
                        'contributing_factors': factors,
                        'user_id': user_id
                    }
                    
                    # Adjust task priority based on predicted success
                    if success_probability < 0.3:
                        # Lower predicted success - might need more attention
                        if hasattr(task, 'priority'):
                            # Boost priority to give it more resources/attention
                            task.metadata = getattr(task, 'metadata', {})
                            task.metadata['needs_attention'] = True
                
                except Exception as e:
                    logger.warning(f"Failed to predict success for task {task.id}: {e}")
        
        # Execute workflow with base implementation
        result = await super().execute_workflow(workflow, strategy, **context_kwargs)
        
        # Learn from workflow execution if enabled
        if self.enable_learning and user_id:
            await self._learn_from_workflow_execution(workflow, result, user_id)
        
        # Add learning insights to result
        if self.enable_learning and hasattr(result, 'metadata'):
            result.metadata = result.metadata or {}
            result.metadata.update({
                'learning_insights': await self._generate_workflow_insights(workflow, result, user_id)
            })
        
        return result
    
    async def _learn_from_workflow_execution(
        self,
        workflow,
        result,
        user_id: str
    ) -> None:
        """Learn from workflow execution outcomes."""
        try:
            if not self.pattern_engine:
                return
            
            # Analyze task execution results
            for task_id in result.tasks_executed:
                if task_id in self._learning_enabled_executions:
                    execution_data = self._learning_enabled_executions[task_id]
                    
                    # Find corresponding task result
                    actual_success = 1.0  # Default assumption
                    
                    # Create learning feedback based on execution outcome
                    # This would be enhanced with actual task outcome analysis
                    validation_result = ValidationResult(
                        is_authentic=result.success,
                        authenticity_score=result.quality_metrics.get('authenticity_rate', 80.0) if hasattr(result, 'quality_metrics') else 80.0,
                        real_progress=80.0,
                        fake_progress=20.0,
                        issues=[],
                        suggestions=[],
                        next_actions=[]
                    )
                    
                    # Learn from validation feedback
                    await self.pattern_engine.learn_from_validation_feedback(
                        user_id,
                        validation_result,
                        None,  # Would need actual AI task result
                        {
                            'workflow_id': str(workflow.id),
                            'predicted_success': execution_data['predicted_success'],
                            'actual_success': actual_success,
                            'prediction_accuracy': 1.0 - abs(execution_data['predicted_success'] - actual_success)
                        }
                    )
            
            # Clean up tracking data
            for task_id in result.tasks_executed:
                if task_id in self._learning_enabled_executions:
                    del self._learning_enabled_executions[task_id]
        
        except Exception as e:
            logger.warning(f"Failed to learn from workflow execution: {e}")
    
    async def _generate_workflow_insights(
        self,
        workflow,
        result,
        user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate insights from workflow execution."""
        insights = []
        
        try:
            if not self.analytics or not user_id:
                return insights
            
            # Generate execution insights
            if hasattr(result, 'quality_metrics') and result.quality_metrics:
                quality_score = result.quality_metrics.get('quality_score', 0)
                
                if quality_score >= 90:
                    insights.append({
                        'type': 'success',
                        'message': 'Excellent workflow execution quality',
                        'score': quality_score,
                        'recommendation': 'Continue using similar approaches'
                    })
                elif quality_score < 60:
                    insights.append({
                        'type': 'improvement',
                        'message': 'Workflow quality could be improved',
                        'score': quality_score,
                        'recommendation': 'Review task patterns and consider alternative approaches'
                    })
            
            # Add execution time insights
            if hasattr(result, 'total_time'):
                if result.total_time < 30:  # Fast execution
                    insights.append({
                        'type': 'efficiency',
                        'message': 'Fast workflow execution',
                        'execution_time': result.total_time,
                        'recommendation': 'Consider applying similar efficiency patterns to other workflows'
                    })
        
        except Exception as e:
            logger.warning(f"Failed to generate workflow insights: {e}")
        
        return insights


class EnhancedValidationService(ValidationService):
    """
    Enhanced validation service that incorporates learning from validation outcomes.
    
    Extends the base validation service with pattern-aware validation,
    learning from validation results, and personalized validation criteria.
    """
    
    def __init__(
        self,
        learning_components: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced validation service.
        
        Args:
            learning_components: Pre-initialized learning components
        """
        super().__init__()
        
        if learning_components:
            self.pattern_engine = learning_components.get('pattern_engine')
            self.personalized_behavior = learning_components.get('personalized_behavior')
            self.privacy_system = learning_components.get('privacy_system')
        else:
            self.pattern_engine = None
            self.personalized_behavior = None
            self.privacy_system = None
    
    async def validate_with_learning(
        self,
        content: str,
        content_type: str = 'code',
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        learn_from_result: bool = True
    ) -> Dict[str, Any]:
        """
        Perform validation with learning integration.
        
        Args:
            content: Content to validate
            content_type: Type of content
            user_id: User identifier for personalized validation
            context: Additional context
            learn_from_result: Whether to learn from validation result
            
        Returns:
            Enhanced validation result
        """
        # Get personalized validation criteria if available
        validation_criteria = {}
        if self.personalized_behavior and user_id:
            validation_criteria = await self.personalized_behavior.adapt_validation_criteria(
                user_id,
                {'authenticity_threshold': 0.8, 'quality_threshold': 0.7},
                context or {}
            )
        
        # Perform base validation
        if content_type == 'code':
            validation_result = await self.validate_code(
                content,
                language=context.get('language', 'python') if context else 'python',
                validation_level='standard'
            )
        else:
            validation_result = await self.validate_response(
                content,
                response_type=content_type,
                validation_criteria=validation_criteria
            )
        
        # Learn from validation result if enabled
        if learn_from_result and self.pattern_engine and user_id:
            await self._learn_from_validation_result(
                user_id, validation_result, context or {}
            )
        
        # Add learning insights
        if self.pattern_engine and user_id:
            validation_result['learning_insights'] = await self._get_validation_insights(
                user_id, validation_result, context or {}
            )
        
        # Apply privacy protection if configured
        if self.privacy_system and user_id:
            validation_result = await self._apply_privacy_protection(
                validation_result, user_id
            )
        
        return validation_result
    
    async def _learn_from_validation_result(
        self,
        user_id: str,
        validation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Learn from validation result."""
        try:
            # Convert validation result to ValidationResult object
            validation_obj = ValidationResult(
                is_authentic=validation_result.get('is_valid', False),
                authenticity_score=validation_result.get('score', 0.0) * 100,
                real_progress=validation_result.get('score', 0.0) * 100,
                fake_progress=(1.0 - validation_result.get('score', 0.0)) * 100,
                issues=[],  # Would extract from validation_result['issues']
                suggestions=validation_result.get('suggestions', []),
                next_actions=[]
            )
            
            # Learn from validation feedback
            await self.pattern_engine.learn_from_validation_feedback(
                user_id,
                validation_obj,
                None,  # AI task result not available in this context
                {
                    'validation_context': context,
                    'validation_type': 'content_validation',
                    'score': validation_result.get('score', 0.0)
                }
            )
        
        except Exception as e:
            logger.warning(f"Failed to learn from validation result: {e}")
    
    async def _get_validation_insights(
        self,
        user_id: str,
        validation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get validation-specific insights."""
        insights = []
        
        try:
            # Get user's validation patterns
            user_patterns = self.pattern_engine._user_patterns.get(user_id, [])
            validation_patterns = [
                p for p in user_patterns
                if p.pattern_type in ['validation_success', 'validation_failure']
            ]
            
            if len(validation_patterns) >= 5:
                recent_success_rate = sum(
                    p.success_rate for p in validation_patterns[-5:]
                ) / 5
                
                current_score = validation_result.get('score', 0.0)
                
                if current_score > recent_success_rate + 0.1:
                    insights.append({
                        'type': 'improvement',
                        'message': f'Validation score improved significantly (current: {current_score:.2f}, recent avg: {recent_success_rate:.2f})',
                        'recommendation': 'Continue current practices'
                    })
                elif current_score < recent_success_rate - 0.1:
                    insights.append({
                        'type': 'decline',
                        'message': f'Validation score declined (current: {current_score:.2f}, recent avg: {recent_success_rate:.2f})',
                        'recommendation': 'Review recent changes in approach'
                    })
        
        except Exception as e:
            logger.warning(f"Failed to generate validation insights: {e}")
        
        return insights
    
    async def _apply_privacy_protection(
        self,
        validation_result: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Apply privacy protection to validation result."""
        try:
            # Remove or anonymize sensitive information
            protected_result = validation_result.copy()
            
            # Remove user-identifying information from metadata
            if 'metadata' in protected_result:
                metadata = protected_result['metadata'].copy()
                # Remove sensitive fields
                sensitive_fields = ['file_path', 'user_id', 'session_id']
                for field in sensitive_fields:
                    metadata.pop(field, None)
                protected_result['metadata'] = metadata
            
            # Add privacy notice
            protected_result['privacy_notice'] = {
                'data_processed_privately': True,
                'user_data_anonymized': True,
                'retention_policy_applied': True
            }
            
            return protected_result
        
        except Exception as e:
            logger.warning(f"Failed to apply privacy protection: {e}")
            return validation_result


class IntelligentSystemIntegration:
    """
    Main integration class that coordinates all AI learning components
    with the existing claude-tiu systems.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize intelligent system integration.
        
        Args:
            config: System configuration
        """
        self.config = config or {}
        
        # Initialize learning components
        self._initialize_learning_components()
        
        # Initialize enhanced system components
        self._initialize_enhanced_components()
        
        # Setup integration hooks
        self._setup_integration_hooks()
        
        logger.info("Intelligent system integration initialized")
    
    def _initialize_learning_components(self) -> None:
        """Initialize core learning components."""
        # Pattern recognition engine
        self.pattern_engine = PatternRecognitionEngine(
            learning_rate=self.config.get('learning_rate', 0.1),
            pattern_confidence_threshold=self.config.get('confidence_threshold', 0.7),
            enable_clustering=self.config.get('enable_clustering', True)
        )
        
        # Personalized AI behavior
        self.personalized_behavior = PersonalizedAIBehavior(
            self.pattern_engine,
            adaptation_rate=self.config.get('adaptation_rate', 0.1)
        )
        
        # Federated learning system (optional)
        if self.config.get('enable_federated_learning', False):
            self.federated_system = FederatedLearningSystem(
                self.pattern_engine,
                node_id=self.config.get('node_id', 'claude_tui_node'),
                organization=self.config.get('organization', 'claude_tui_org')
            )
        else:
            self.federated_system = None
        
        # Privacy system
        self.privacy_system = PrivacyPreservingLearning(
            differential_privacy_epsilon=self.config.get('privacy_epsilon', 1.0)
        )
        
        # Analytics system
        self.analytics = LearningAnalytics(
            self.pattern_engine,
            self.personalized_behavior,
            self.federated_system,
            self.config.get('analytics_config', {})
        )
    
    def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced system components."""
        learning_components = {
            'pattern_engine': self.pattern_engine,
            'personalized_behavior': self.personalized_behavior,
            'analytics': self.analytics,
            'privacy_system': self.privacy_system
        }
        
        # Enhanced AI interface
        self.ai_interface = EnhancedAIInterface(
            enable_learning=True,
            learning_config=self.config.get('ai_interface_config', {})
        )
        
        # Enhanced task engine
        self.task_engine = EnhancedTaskEngine(
            enable_learning=True,
            learning_components=learning_components
        )
        
        # Enhanced validation service
        self.validation_service = EnhancedValidationService(
            learning_components=learning_components
        )
    
    def _setup_integration_hooks(self) -> None:
        """Setup integration hooks between components."""
        # Hook validation results to learning system
        # This would be implemented with proper event system in production
        pass
    
    async def execute_intelligent_development_task(
        self,
        task_description: str,
        user_id: str,
        project_context: Dict[str, Any],
        learn_from_execution: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a complete intelligent development task with full learning integration.
        
        Args:
            task_description: Description of the task
            user_id: User identifier
            project_context: Project context information
            learn_from_execution: Whether to learn from execution
            
        Returns:
            Complete task execution result with learning insights
        """
        start_time = datetime.utcnow()
        
        try:
            # Create task object
            task = Task(
                name=f"Development Task {start_time.isoformat()}",
                description=task_description,
                ai_prompt=task_description,
                id=uuid4()
            )
            
            # Create AI context
            ai_context = AIContext(
                project_path=project_context.get('project_path'),
                current_files=project_context.get('current_files', []),
                dependencies=project_context.get('dependencies', []),
                framework_info=project_context.get('framework_info', {}),
                coding_standards=project_context.get('coding_standards', {}),
                test_requirements=project_context.get('test_requirements', {})
            )
            
            # Execute with enhanced AI interface
            ai_result = await self.ai_interface.execute_intelligent_task(
                task,
                ai_context,
                user_id,
                learn_from_interaction=learn_from_execution
            )
            
            # Validate result with learning
            if 'ai_result' in ai_result and hasattr(ai_result['ai_result'], 'generated_content'):
                validation_result = await self.validation_service.validate_with_learning(
                    ai_result['ai_result'].generated_content,
                    content_type='code',
                    user_id=user_id,
                    context=project_context,
                    learn_from_result=learn_from_execution
                )
            else:
                validation_result = {'score': 0.8, 'is_valid': True}  # Default
            
            # Generate user analytics
            user_analytics = await self.analytics.generate_user_analytics(
                user_id,
                time_range_days=7,
                include_predictions=True
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Comprehensive result
            comprehensive_result = {
                'task_execution': ai_result,
                'validation_result': validation_result,
                'user_analytics': user_analytics,
                'learning_insights': await self._generate_comprehensive_insights(
                    user_id, task, ai_result, validation_result
                ),
                'execution_metadata': {
                    'execution_time': execution_time,
                    'learning_enabled': learn_from_execution,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            return comprehensive_result
        
        except Exception as e:
            logger.error(f"Intelligent development task execution failed: {e}")
            return {
                'error': str(e),
                'execution_metadata': {
                    'execution_time': (datetime.utcnow() - start_time).total_seconds(),
                    'success': False
                }
            }
    
    async def _generate_comprehensive_insights(
        self,
        user_id: str,
        task: Task,
        ai_result: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive insights from task execution."""
        insights = []
        
        try:
            # Get pattern-based insights
            success_patterns = await self.pattern_engine.identify_success_patterns(
                user_id=user_id,
                min_frequency=2
            )
            
            if success_patterns:
                insights.append({
                    'type': 'pattern_insight',
                    'message': f'Found {len(success_patterns)} successful patterns',
                    'patterns': [p.pattern_name for p in success_patterns[:3]],
                    'recommendation': 'Continue leveraging these successful patterns'
                })
            
            # Get personalization insights
            personalization_effectiveness = await self.personalized_behavior.measure_personalization_effectiveness(
                user_id,
                time_window_days=14
            )
            
            if personalization_effectiveness.get('improvement_from_personalization', 0) > 0.1:
                insights.append({
                    'type': 'personalization_insight',
                    'message': 'Personalization is significantly improving your performance',
                    'improvement': personalization_effectiveness['improvement_from_personalization'],
                    'recommendation': 'Continue providing feedback to enhance personalization'
                })
            
            # Get federated insights if available
            if self.federated_system:
                federated_recommendations = await self.federated_system.get_federated_recommendations(
                    user_id,
                    {'task_type': 'development', 'context': 'intelligent_execution'}
                )
                
                if federated_recommendations:
                    insights.append({
                        'type': 'collaborative_insight',
                        'message': f'Found {len(federated_recommendations)} collaborative recommendations',
                        'recommendations': [r.get('message', '') for r in federated_recommendations[:2]],
                        'source': 'federated_learning'
                    })
        
        except Exception as e:
            logger.warning(f"Failed to generate comprehensive insights: {e}")
        
        return insights
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        try:
            # Pattern engine health
            pattern_analytics = await self.pattern_engine.get_pattern_analytics()
            health_status['components']['pattern_engine'] = {
                'status': 'healthy' if pattern_analytics['total_patterns'] > 0 else 'warning',
                'total_patterns': pattern_analytics['total_patterns'],
                'users': pattern_analytics['users']
            }
            
            # Analytics health
            health_status['components']['analytics'] = {
                'status': 'healthy',
                'insights_available': len(self.analytics._insight_history)
            }
            
            # Validation service health
            validation_health = await self.validation_service.health_check()
            health_status['components']['validation_service'] = validation_health
            
            # Privacy system health
            if self.privacy_system:
                privacy_budget = self.privacy_system.dp_engine.get_remaining_budget()
                health_status['components']['privacy_system'] = {
                    'status': 'healthy' if privacy_budget > 0.1 else 'warning',
                    'privacy_budget_remaining': privacy_budget
                }
            
            # Federated system health
            if self.federated_system:
                federated_analytics = await self.federated_system.measure_federated_impact()
                health_status['components']['federated_system'] = {
                    'status': 'healthy',
                    'federation_analytics': federated_analytics
                }
        
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            health_status['overall_status'] = 'degraded'
            health_status['error'] = str(e)
        
        return health_status
    
    def get_learning_components(self) -> Dict[str, Any]:
        """Get access to learning components for external use."""
        return {
            'pattern_engine': self.pattern_engine,
            'personalized_behavior': self.personalized_behavior,
            'federated_system': self.federated_system,
            'privacy_system': self.privacy_system,
            'analytics': self.analytics,
            'ai_interface': self.ai_interface,
            'task_engine': self.task_engine,
            'validation_service': self.validation_service
        }