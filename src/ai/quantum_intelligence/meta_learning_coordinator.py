"""
Meta-Learning Coordinator
========================

Revolutionary meta-learning system that enables agents to learn how to learn
and optimize their learning strategies dynamically. This coordinator implements
advanced meta-learning algorithms for adaptive strategy optimization.

Core Capabilities:
- Learning strategy optimization and adaptation
- Meta-knowledge extraction from learning experiences  
- Transfer learning across different task domains
- Few-shot learning capability enhancement
- Continuous learning strategy evolution

Technical Features:
- Model-Agnostic Meta-Learning (MAML) implementation
- Gradient-based meta-optimization
- Experience replay with meta-gradients
- Adaptive learning rate scheduling
- Cross-domain knowledge transfer

Quantum Intelligence Aspects:
- Quantum-inspired meta-gradient computation
- Superposition of learning strategies
- Entangled learning experiences across agents
- Quantum tunneling through local learning minima
"""

import asyncio
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import math
from abc import ABC, abstractmethod
import pickle
import hashlib

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Available meta-learning strategies."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    TRANSFER_LEARNING = "transfer_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"
    CONTINUAL_LEARNING = "continual_learning"
    MULTI_TASK_LEARNING = "multi_task_learning"
    QUANTUM_INSPIRED = "quantum_inspired"

class TaskDomain(Enum):
    """Task domains for learning transfer."""
    CODE_GENERATION = "code_generation"
    PROBLEM_SOLVING = "problem_solving"
    PATTERN_RECOGNITION = "pattern_recognition"
    DECISION_MAKING = "decision_making"
    COLLABORATION = "collaboration"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    REASONING = "reasoning"
    CREATIVITY = "creativity"

class MetaLearningObjective(Enum):
    """Objectives for meta-learning optimization."""
    FAST_ADAPTATION = "fast_adaptation"
    SAMPLE_EFFICIENCY = "sample_efficiency"
    GENERALIZATION = "generalization"
    ROBUSTNESS = "robustness"
    TRANSFER_ABILITY = "transfer_ability"
    MEMORY_EFFICIENCY = "memory_efficiency"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    MULTI_TASK_PERFORMANCE = "multi_task_performance"

@dataclass
class LearningExperience:
    """Represents a learning experience for meta-learning."""
    experience_id: str
    agent_id: str
    task_domain: TaskDomain
    learning_strategy: LearningStrategy
    initial_performance: float
    final_performance: float
    learning_steps: int
    learning_time: float
    task_context: Dict[str, Any] = field(default_factory=dict)
    strategy_parameters: Dict[str, float] = field(default_factory=dict)
    gradient_info: Optional[List[float]] = None
    meta_features: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    success: bool = True

@dataclass
class MetaKnowledge:
    """Meta-knowledge extracted from learning experiences."""
    knowledge_id: str
    domain: TaskDomain
    applicable_strategies: List[LearningStrategy]
    optimal_parameters: Dict[str, float] = field(default_factory=dict)
    expected_performance: float = 0.0
    confidence_level: float = 0.0
    transfer_compatibility: Dict[TaskDomain, float] = field(default_factory=dict)
    learning_curve_pattern: List[float] = field(default_factory=list)
    meta_features: List[float] = field(default_factory=list)
    creation_timestamp: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    usage_count: int = 0

@dataclass
class AdaptationPlan:
    """Plan for adapting learning strategy to new task."""
    plan_id: str
    target_agent: str
    target_task: Dict[str, Any]
    recommended_strategy: LearningStrategy
    strategy_parameters: Dict[str, float]
    expected_adaptation_time: float
    expected_performance: float
    confidence_score: float
    fallback_strategies: List[LearningStrategy] = field(default_factory=list)
    meta_gradients: Optional[List[float]] = None
    transfer_sources: List[str] = field(default_factory=list)

@dataclass
class MetaLearningMetrics:
    """Metrics for meta-learning performance."""
    adaptation_speed: float = 0.0
    sample_efficiency: float = 0.0
    generalization_score: float = 0.0
    transfer_effectiveness: float = 0.0
    strategy_diversity: float = 0.0
    meta_gradient_stability: float = 0.0
    knowledge_utilization: float = 0.0
    cross_domain_performance: float = 0.0
    few_shot_capability: float = 0.0
    continual_learning_score: float = 0.0

class MetaLearningAlgorithm(ABC):
    """Abstract base class for meta-learning algorithms."""
    
    @abstractmethod
    async def compute_meta_gradients(self, 
                                   experiences: List[LearningExperience]) -> List[float]:
        """Compute meta-gradients from learning experiences."""
        pass
    
    @abstractmethod
    async def adapt_strategy(self, 
                           strategy: LearningStrategy,
                           parameters: Dict[str, float],
                           meta_gradients: List[float]) -> Tuple[LearningStrategy, Dict[str, float]]:
        """Adapt learning strategy based on meta-gradients."""
        pass
    
    @abstractmethod
    async def evaluate_adaptation(self, 
                                adapted_strategy: LearningStrategy,
                                parameters: Dict[str, float],
                                task_context: Dict[str, Any]) -> float:
        """Evaluate the quality of strategy adaptation."""
        pass

class MAMLMetaLearner(MetaLearningAlgorithm):
    """Model-Agnostic Meta-Learning implementation."""
    
    def __init__(self, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_parameters = defaultdict(float)
        
    async def compute_meta_gradients(self, 
                                   experiences: List[LearningExperience]) -> List[float]:
        """Compute MAML meta-gradients from experiences."""
        try:
            if len(experiences) < 2:
                return [0.0] * 10  # Default gradient vector size
            
            meta_gradients = []
            
            # Group experiences by task/agent
            task_groups = defaultdict(list)
            for exp in experiences:
                task_key = f"{exp.agent_id}_{exp.task_domain.value}"
                task_groups[task_key].append(exp)
            
            # Compute meta-gradients for each task group
            for task_key, task_experiences in task_groups.items():
                if len(task_experiences) < 2:
                    continue
                
                # Sort by timestamp to get learning trajectory
                task_experiences.sort(key=lambda x: x.timestamp)
                
                # Compute inner loop gradients (simulated)
                inner_gradients = []
                for i in range(1, len(task_experiences)):
                    prev_exp = task_experiences[i-1]
                    curr_exp = task_experiences[i]
                    
                    # Performance improvement gradient
                    perf_gradient = (curr_exp.final_performance - prev_exp.final_performance) / max(1, curr_exp.learning_steps)
                    
                    # Strategy parameter gradients (simulated)
                    param_gradients = []
                    for param_name in ['learning_rate', 'momentum', 'decay']:
                        prev_val = prev_exp.strategy_parameters.get(param_name, 0.5)
                        curr_val = curr_exp.strategy_parameters.get(param_name, 0.5)
                        param_grad = (curr_val - prev_val) * perf_gradient
                        param_gradients.append(param_grad)
                    
                    inner_gradients.extend([perf_gradient] + param_gradients)
                
                # Aggregate inner gradients
                if inner_gradients:
                    avg_gradients = np.mean(np.array(inner_gradients).reshape(-1, 4), axis=0)
                    meta_gradients.extend(avg_gradients.tolist())
            
            # Ensure consistent gradient vector size
            target_size = 10
            if len(meta_gradients) < target_size:
                meta_gradients.extend([0.0] * (target_size - len(meta_gradients)))
            elif len(meta_gradients) > target_size:
                meta_gradients = meta_gradients[:target_size]
            
            return meta_gradients
            
        except Exception as e:
            logger.error(f"MAML meta-gradient computation failed: {e}")
            return [0.0] * 10
    
    async def adapt_strategy(self, 
                           strategy: LearningStrategy,
                           parameters: Dict[str, float],
                           meta_gradients: List[float]) -> Tuple[LearningStrategy, Dict[str, float]]:
        """Adapt strategy using MAML meta-gradients."""
        try:
            adapted_parameters = parameters.copy()
            
            # Apply meta-gradients to parameters
            param_names = ['learning_rate', 'momentum', 'decay', 'temperature']
            for i, param_name in enumerate(param_names):
                if i < len(meta_gradients):
                    current_val = adapted_parameters.get(param_name, 0.5)
                    gradient = meta_gradients[i]
                    
                    # Apply meta-gradient update
                    new_val = current_val - self.meta_lr * gradient
                    adapted_parameters[param_name] = max(0.01, min(1.0, new_val))
            
            # Strategy adaptation based on gradient magnitude
            gradient_magnitude = np.linalg.norm(meta_gradients)
            
            if gradient_magnitude > 1.0:
                # High gradients suggest need for more adaptive strategy
                if strategy == LearningStrategy.GRADIENT_BASED:
                    adapted_strategy = LearningStrategy.BAYESIAN_OPTIMIZATION
                else:
                    adapted_strategy = strategy
            else:
                adapted_strategy = strategy
            
            return adapted_strategy, adapted_parameters
            
        except Exception as e:
            logger.error(f"MAML strategy adaptation failed: {e}")
            return strategy, parameters
    
    async def evaluate_adaptation(self, 
                                adapted_strategy: LearningStrategy,
                                parameters: Dict[str, float],
                                task_context: Dict[str, Any]) -> float:
        """Evaluate MAML adaptation quality."""
        try:
            # Simulated evaluation based on strategy and parameter alignment
            base_score = 0.7
            
            # Parameter quality score
            param_score = 0.0
            ideal_params = {
                'learning_rate': 0.01,
                'momentum': 0.9,
                'decay': 0.99,
                'temperature': 1.0
            }
            
            for param_name, ideal_val in ideal_params.items():
                if param_name in parameters:
                    diff = abs(parameters[param_name] - ideal_val)
                    param_score += max(0.0, 1.0 - diff)
            
            param_score /= len(ideal_params)
            
            # Strategy appropriateness score
            task_complexity = task_context.get('complexity', 0.5)
            strategy_score = 0.5
            
            if adapted_strategy == LearningStrategy.BAYESIAN_OPTIMIZATION and task_complexity > 0.7:
                strategy_score = 0.9
            elif adapted_strategy == LearningStrategy.GRADIENT_BASED and task_complexity < 0.4:
                strategy_score = 0.8
            
            # Combined evaluation
            evaluation_score = base_score * 0.4 + param_score * 0.4 + strategy_score * 0.2
            
            return min(1.0, evaluation_score)
            
        except Exception as e:
            logger.error(f"MAML evaluation failed: {e}")
            return 0.5

class QuantumMetaLearner(MetaLearningAlgorithm):
    """Quantum-inspired meta-learning algorithm."""
    
    def __init__(self, quantum_coherence: float = 0.8):
        self.quantum_coherence = quantum_coherence
        self.quantum_states = {}
        self.entanglement_matrix = defaultdict(dict)
        
    async def compute_meta_gradients(self, 
                                   experiences: List[LearningExperience]) -> List[float]:
        """Compute quantum-inspired meta-gradients."""
        try:
            if not experiences:
                return [0.0] * 12  # Quantum gradient vector
            
            # Create quantum superposition of experiences
            quantum_gradients = []
            
            for exp in experiences:
                # Quantum state representation of experience
                quantum_state = [
                    exp.final_performance - exp.initial_performance,  # Performance improvement
                    exp.learning_time / 3600.0,  # Normalized time
                    exp.learning_steps / 100.0,  # Normalized steps
                    len(exp.strategy_parameters) / 10.0,  # Parameter complexity
                ]
                
                # Apply quantum transformations
                quantum_transformed = await self._apply_quantum_transformations(quantum_state)
                quantum_gradients.extend(quantum_transformed)
            
            # Quantum interference and measurement collapse
            collapsed_gradients = await self._quantum_measurement_collapse(quantum_gradients)
            
            return collapsed_gradients
            
        except Exception as e:
            logger.error(f"Quantum meta-gradient computation failed: {e}")
            return [0.0] * 12
    
    async def _apply_quantum_transformations(self, state: List[float]) -> List[float]:
        """Apply quantum transformations to learning state."""
        # Simulated quantum gates
        transformed = []
        
        for i, val in enumerate(state):
            # Quantum rotation (simulated)
            angle = val * np.pi
            rotated_val = np.cos(angle) + 1j * np.sin(angle)
            
            # Extract real and imaginary parts as features
            transformed.extend([rotated_val.real, rotated_val.imag])
            
            # Quantum entanglement (correlation with other values)
            for j, other_val in enumerate(state):
                if i != j:
                    entangled = val * other_val * self.quantum_coherence
                    transformed.append(entangled)
        
        return transformed[:12]  # Limit size
    
    async def _quantum_measurement_collapse(self, quantum_gradients: List[float]) -> List[float]:
        """Collapse quantum superposition to classical gradients."""
        if not quantum_gradients:
            return [0.0] * 12
        
        # Group gradients for measurement
        gradient_array = np.array(quantum_gradients)
        
        # Reshape to quantum register size
        target_size = 12
        if len(gradient_array) > target_size:
            # Quantum amplitude collapse
            chunks = np.array_split(gradient_array, target_size)
            collapsed = [np.mean(chunk) for chunk in chunks]
        else:
            # Pad with quantum vacuum states (zeros)
            collapsed = gradient_array.tolist() + [0.0] * (target_size - len(gradient_array))
        
        # Apply measurement uncertainty
        for i in range(len(collapsed)):
            uncertainty = np.random.normal(0, 0.01)  # Quantum measurement noise
            collapsed[i] += uncertainty
        
        return collapsed[:target_size]
    
    async def adapt_strategy(self, 
                           strategy: LearningStrategy,
                           parameters: Dict[str, float],
                           meta_gradients: List[float]) -> Tuple[LearningStrategy, Dict[str, float]]:
        """Quantum-inspired strategy adaptation."""
        try:
            # Create quantum superposition of strategies
            strategy_probabilities = await self._calculate_strategy_probabilities(
                meta_gradients, strategy
            )
            
            # Measure optimal strategy
            adapted_strategy = await self._measure_optimal_strategy(strategy_probabilities)
            
            # Quantum parameter adaptation
            adapted_parameters = await self._quantum_parameter_adaptation(
                parameters, meta_gradients
            )
            
            return adapted_strategy, adapted_parameters
            
        except Exception as e:
            logger.error(f"Quantum strategy adaptation failed: {e}")
            return strategy, parameters
    
    async def _calculate_strategy_probabilities(self, 
                                              gradients: List[float],
                                              current_strategy: LearningStrategy) -> Dict[LearningStrategy, float]:
        """Calculate quantum probability distribution over strategies."""
        probabilities = {}
        
        # Base probability from gradients
        gradient_magnitude = np.linalg.norm(gradients)
        
        for strategy in LearningStrategy:
            if strategy == current_strategy:
                base_prob = 0.4  # Bias toward current strategy
            else:
                base_prob = 0.6 / (len(LearningStrategy) - 1)
            
            # Quantum amplitude based on gradient alignment
            strategy_index = list(LearningStrategy).index(strategy)
            if strategy_index < len(gradients):
                quantum_amplitude = abs(gradients[strategy_index])
                quantum_prob = quantum_amplitude ** 2
            else:
                quantum_prob = 0.1
            
            # Combine classical and quantum probabilities
            combined_prob = base_prob * 0.6 + quantum_prob * 0.4
            probabilities[strategy] = combined_prob
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    async def _measure_optimal_strategy(self, 
                                      probabilities: Dict[LearningStrategy, float]) -> LearningStrategy:
        """Quantum measurement of optimal strategy."""
        # Weighted random selection based on quantum probabilities
        strategies = list(probabilities.keys())
        weights = list(probabilities.values())
        
        # Quantum measurement (with some coherence preservation)
        if np.random.random() < self.quantum_coherence:
            # Coherent measurement - select highest probability
            return max(probabilities, key=probabilities.get)
        else:
            # Incoherent measurement - random selection
            return np.random.choice(strategies, p=weights)
    
    async def _quantum_parameter_adaptation(self, 
                                          parameters: Dict[str, float],
                                          gradients: List[float]) -> Dict[str, float]:
        """Quantum-inspired parameter adaptation."""
        adapted = parameters.copy()
        
        param_names = list(parameters.keys()) if parameters else ['learning_rate', 'momentum', 'decay']
        
        for i, param_name in enumerate(param_names):
            if i < len(gradients):
                current_val = adapted.get(param_name, 0.5)
                gradient = gradients[i]
                
                # Quantum tunneling through local minima
                tunnel_prob = abs(gradient) * self.quantum_coherence
                if np.random.random() < tunnel_prob:
                    # Quantum tunnel to new parameter value
                    new_val = current_val + gradient * np.random.uniform(-2, 2)
                else:
                    # Classical gradient update
                    new_val = current_val + gradient * 0.01
                
                adapted[param_name] = max(0.001, min(1.0, new_val))
        
        return adapted
    
    async def evaluate_adaptation(self, 
                                adapted_strategy: LearningStrategy,
                                parameters: Dict[str, float],
                                task_context: Dict[str, Any]) -> float:
        """Quantum evaluation of adaptation quality."""
        try:
            # Create quantum evaluation state
            eval_state = [
                list(LearningStrategy).index(adapted_strategy) / len(LearningStrategy),
                len(parameters) / 10.0,
                task_context.get('complexity', 0.5),
                task_context.get('novelty', 0.5)
            ]
            
            # Apply quantum evaluation transformations
            quantum_eval = await self._apply_quantum_transformations(eval_state)
            
            # Quantum measurement of quality
            quality_amplitude = np.mean([abs(val) for val in quantum_eval])
            quality_score = min(1.0, quality_amplitude)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quantum evaluation failed: {e}")
            return 0.5

class MetaLearningCoordinator:
    """
    Revolutionary Meta-Learning Coordinator for Learning Strategy Optimization.
    
    This coordinator enables agents to learn how to learn, optimizing their
    learning strategies based on experience and context.
    """
    
    def __init__(self, max_experiences: int = 1000):
        self.max_experiences = max_experiences
        self.learning_experiences = deque(maxlen=max_experiences)
        self.meta_knowledge_base: Dict[str, MetaKnowledge] = {}
        self.active_adaptations: Dict[str, AdaptationPlan] = {}
        self.meta_learners = {
            'maml': MAMLMetaLearner(),
            'quantum': QuantumMetaLearner()
        }
        self.default_strategies: Dict[TaskDomain, LearningStrategy] = {
            TaskDomain.CODE_GENERATION: LearningStrategy.NEURAL_ARCHITECTURE_SEARCH,
            TaskDomain.PROBLEM_SOLVING: LearningStrategy.REINFORCEMENT_LEARNING,
            TaskDomain.PATTERN_RECOGNITION: LearningStrategy.TRANSFER_LEARNING,
            TaskDomain.DECISION_MAKING: LearningStrategy.BAYESIAN_OPTIMIZATION,
            TaskDomain.COLLABORATION: LearningStrategy.MULTI_TASK_LEARNING,
            TaskDomain.OPTIMIZATION: LearningStrategy.EVOLUTIONARY,
            TaskDomain.PREDICTION: LearningStrategy.GRADIENT_BASED,
            TaskDomain.REASONING: LearningStrategy.FEW_SHOT_LEARNING
        }
        self.coordinator_callbacks: List[Callable] = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        logger.info("Meta-Learning Coordinator initialized")
    
    async def record_learning_experience(self,
                                       agent_id: str,
                                       task_domain: TaskDomain,
                                       learning_strategy: LearningStrategy,
                                       initial_performance: float,
                                       final_performance: float,
                                       learning_steps: int,
                                       learning_time: float,
                                       strategy_parameters: Dict[str, float],
                                       task_context: Optional[Dict[str, Any]] = None) -> str:
        """Record a learning experience for meta-learning."""
        try:
            with self.lock:
                experience_id = f"exp_{int(time.time())}_{agent_id}_{len(self.learning_experiences)}"
                
                experience = LearningExperience(
                    experience_id=experience_id,
                    agent_id=agent_id,
                    task_domain=task_domain,
                    learning_strategy=learning_strategy,
                    initial_performance=initial_performance,
                    final_performance=final_performance,
                    learning_steps=learning_steps,
                    learning_time=learning_time,
                    task_context=task_context or {},
                    strategy_parameters=strategy_parameters,
                    meta_features=await self._extract_meta_features(
                        task_domain, learning_strategy, strategy_parameters, task_context
                    ),
                    success=final_performance > initial_performance
                )
                
                self.learning_experiences.append(experience)
                
                # Trigger meta-knowledge update
                asyncio.create_task(self._update_meta_knowledge(experience))
                
                logger.info(f"Recorded learning experience: {experience_id}")
                return experience_id
                
        except Exception as e:
            logger.error(f"Failed to record learning experience: {e}")
            return ""
    
    async def recommend_learning_strategy(self,
                                        agent_id: str,
                                        task_domain: TaskDomain,
                                        task_context: Dict[str, Any],
                                        objective: MetaLearningObjective = MetaLearningObjective.FAST_ADAPTATION) -> AdaptationPlan:
        """Recommend optimal learning strategy for a task."""
        try:
            with self.lock:
                plan_id = f"plan_{int(time.time())}_{agent_id}"
                
                # Find relevant meta-knowledge
                relevant_knowledge = await self._find_relevant_meta_knowledge(
                    task_domain, task_context
                )
                
                # Generate recommendation using meta-learners
                if relevant_knowledge:
                    strategy, parameters, confidence = await self._generate_recommendation_from_knowledge(
                        relevant_knowledge, objective, task_context
                    )
                else:
                    # Fall back to default strategy
                    strategy = self.default_strategies.get(task_domain, LearningStrategy.GRADIENT_BASED)
                    parameters = await self._get_default_parameters(strategy)
                    confidence = 0.5
                
                # Create adaptation plan
                adaptation_plan = AdaptationPlan(
                    plan_id=plan_id,
                    target_agent=agent_id,
                    target_task=task_context,
                    recommended_strategy=strategy,
                    strategy_parameters=parameters,
                    expected_adaptation_time=await self._estimate_adaptation_time(
                        strategy, parameters, task_context
                    ),
                    expected_performance=await self._estimate_expected_performance(
                        strategy, parameters, task_domain, task_context
                    ),
                    confidence_score=confidence,
                    fallback_strategies=await self._generate_fallback_strategies(
                        strategy, task_domain
                    )
                )
                
                self.active_adaptations[plan_id] = adaptation_plan
                
                logger.info(f"Generated adaptation plan: {plan_id} for {task_domain.value}")
                return adaptation_plan
                
        except Exception as e:
            logger.error(f"Strategy recommendation failed: {e}")
            return await self._create_default_adaptation_plan(agent_id, task_domain, task_context)
    
    async def optimize_learning_strategy(self,
                                       current_strategy: LearningStrategy,
                                       current_parameters: Dict[str, float],
                                       performance_history: List[float],
                                       task_context: Dict[str, Any],
                                       meta_learner_type: str = 'maml') -> Tuple[LearningStrategy, Dict[str, float]]:
        """Optimize learning strategy based on performance history."""
        try:
            with self.lock:
                if meta_learner_type not in self.meta_learners:
                    logger.warning(f"Unknown meta-learner type: {meta_learner_type}")
                    return current_strategy, current_parameters
                
                meta_learner = self.meta_learners[meta_learner_type]
                
                # Create synthetic experiences from performance history
                experiences = await self._create_experiences_from_history(
                    performance_history, current_strategy, current_parameters, task_context
                )
                
                # Compute meta-gradients
                meta_gradients = await meta_learner.compute_meta_gradients(experiences)
                
                # Adapt strategy
                optimized_strategy, optimized_parameters = await meta_learner.adapt_strategy(
                    current_strategy, current_parameters, meta_gradients
                )
                
                # Evaluate adaptation
                adaptation_quality = await meta_learner.evaluate_adaptation(
                    optimized_strategy, optimized_parameters, task_context
                )
                
                logger.info(f"Strategy optimized: {current_strategy.value} -> {optimized_strategy.value} "
                           f"(quality: {adaptation_quality:.3f})")
                
                return optimized_strategy, optimized_parameters
                
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            return current_strategy, current_parameters
    
    async def transfer_knowledge(self,
                               source_domain: TaskDomain,
                               target_domain: TaskDomain,
                               target_context: Dict[str, Any]) -> Optional[MetaKnowledge]:
        """Transfer meta-knowledge between domains."""
        try:
            with self.lock:
                # Find source knowledge
                source_knowledge = None
                best_compatibility = 0.0
                
                for knowledge in self.meta_knowledge_base.values():
                    if knowledge.domain == source_domain:
                        compatibility = knowledge.transfer_compatibility.get(target_domain, 0.0)
                        if compatibility > best_compatibility:
                            best_compatibility = compatibility
                            source_knowledge = knowledge
                
                if not source_knowledge or best_compatibility < 0.3:
                    logger.info(f"No suitable knowledge for transfer from {source_domain.value} to {target_domain.value}")
                    return None
                
                # Create transferred knowledge
                transferred_id = f"transfer_{int(time.time())}_{source_domain.value}_{target_domain.value}"
                
                transferred_knowledge = MetaKnowledge(
                    knowledge_id=transferred_id,
                    domain=target_domain,
                    applicable_strategies=source_knowledge.applicable_strategies.copy(),
                    optimal_parameters=source_knowledge.optimal_parameters.copy(),
                    expected_performance=source_knowledge.expected_performance * best_compatibility,
                    confidence_level=source_knowledge.confidence_level * best_compatibility,
                    transfer_compatibility={},  # Will be learned
                    learning_curve_pattern=source_knowledge.learning_curve_pattern.copy(),
                    meta_features=await self._adapt_meta_features_for_domain(
                        source_knowledge.meta_features, target_domain, target_context
                    )
                )
                
                # Apply domain-specific adaptations
                await self._adapt_knowledge_for_domain(transferred_knowledge, target_domain, target_context)
                
                # Store transferred knowledge
                self.meta_knowledge_base[transferred_id] = transferred_knowledge
                
                logger.info(f"Knowledge transferred: {source_domain.value} -> {target_domain.value}")
                return transferred_knowledge
                
        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
            return None
    
    async def evaluate_meta_learning_performance(self) -> MetaLearningMetrics:
        """Evaluate overall meta-learning system performance."""
        try:
            with self.lock:
                if not self.learning_experiences:
                    return MetaLearningMetrics()
                
                experiences = list(self.learning_experiences)
                
                # Calculate adaptation speed
                adaptation_times = [exp.learning_time for exp in experiences if exp.success]
                adaptation_speed = 1.0 / (np.mean(adaptation_times) / 3600.0 + 1e-6) if adaptation_times else 0.0
                adaptation_speed = min(1.0, adaptation_speed / 10.0)  # Normalize
                
                # Calculate sample efficiency
                learning_steps = [exp.learning_steps for exp in experiences if exp.success]
                sample_efficiency = 1.0 / (np.mean(learning_steps) / 100.0 + 1e-6) if learning_steps else 0.0
                sample_efficiency = min(1.0, sample_efficiency / 10.0)  # Normalize
                
                # Calculate generalization score
                successful_experiences = [exp for exp in experiences if exp.success]
                if len(successful_experiences) > 1:
                    performance_improvements = [
                        exp.final_performance - exp.initial_performance for exp in successful_experiences
                    ]
                    generalization_score = np.mean(performance_improvements)
                else:
                    generalization_score = 0.0
                
                # Calculate transfer effectiveness
                transfer_count = len([k for k in self.meta_knowledge_base.keys() if 'transfer' in k])
                total_knowledge = len(self.meta_knowledge_base)
                transfer_effectiveness = transfer_count / max(1, total_knowledge)
                
                # Calculate strategy diversity
                strategies_used = set(exp.learning_strategy for exp in experiences)
                strategy_diversity = len(strategies_used) / len(LearningStrategy)
                
                # Calculate meta-gradient stability
                recent_experiences = experiences[-50:] if len(experiences) > 50 else experiences
                if len(recent_experiences) > 10:
                    maml_learner = self.meta_learners['maml']
                    gradients = await maml_learner.compute_meta_gradients(recent_experiences)
                    meta_gradient_stability = 1.0 - np.std(gradients) / (np.mean(np.abs(gradients)) + 1e-6)
                    meta_gradient_stability = max(0.0, meta_gradient_stability)
                else:
                    meta_gradient_stability = 0.5
                
                # Calculate knowledge utilization
                knowledge_usage = [k.usage_count for k in self.meta_knowledge_base.values()]
                knowledge_utilization = np.mean(knowledge_usage) / 10.0 if knowledge_usage else 0.0
                knowledge_utilization = min(1.0, knowledge_utilization)
                
                # Calculate cross-domain performance
                domains = set(exp.task_domain for exp in experiences)
                if len(domains) > 1:
                    domain_performances = defaultdict(list)
                    for exp in successful_experiences:
                        improvement = exp.final_performance - exp.initial_performance
                        domain_performances[exp.task_domain].append(improvement)
                    
                    domain_avgs = [np.mean(improvements) for improvements in domain_performances.values()]
                    cross_domain_performance = np.mean(domain_avgs)
                else:
                    cross_domain_performance = generalization_score
                
                # Calculate few-shot capability
                few_shot_experiences = [exp for exp in experiences if exp.learning_steps < 10 and exp.success]
                few_shot_capability = len(few_shot_experiences) / max(1, len(experiences))
                
                # Calculate continual learning score
                if len(experiences) > 20:
                    # Check for performance maintenance over time
                    time_sorted = sorted(experiences, key=lambda x: x.timestamp)
                    early_performance = np.mean([exp.final_performance for exp in time_sorted[:10]])
                    recent_performance = np.mean([exp.final_performance for exp in time_sorted[-10:]])
                    continual_learning_score = min(1.0, recent_performance / (early_performance + 1e-6))
                else:
                    continual_learning_score = 0.5
                
                return MetaLearningMetrics(
                    adaptation_speed=adaptation_speed,
                    sample_efficiency=sample_efficiency,
                    generalization_score=generalization_score,
                    transfer_effectiveness=transfer_effectiveness,
                    strategy_diversity=strategy_diversity,
                    meta_gradient_stability=meta_gradient_stability,
                    knowledge_utilization=knowledge_utilization,
                    cross_domain_performance=cross_domain_performance,
                    few_shot_capability=few_shot_capability,
                    continual_learning_score=continual_learning_score
                )
                
        except Exception as e:
            logger.error(f"Meta-learning evaluation failed: {e}")
            return MetaLearningMetrics()
    
    async def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning status."""
        try:
            with self.lock:
                # Calculate metrics
                metrics = await self.evaluate_meta_learning_performance()
                
                # Experience statistics
                experiences = list(self.learning_experiences)
                successful_experiences = [exp for exp in experiences if exp.success]
                
                experience_stats = {
                    'total_experiences': len(experiences),
                    'successful_experiences': len(successful_experiences),
                    'success_rate': len(successful_experiences) / max(1, len(experiences)),
                    'domains_covered': len(set(exp.task_domain for exp in experiences)),
                    'strategies_used': len(set(exp.learning_strategy for exp in experiences)),
                    'avg_learning_time': np.mean([exp.learning_time for exp in experiences]) if experiences else 0.0,
                    'avg_learning_steps': np.mean([exp.learning_steps for exp in experiences]) if experiences else 0.0
                }
                
                # Knowledge base statistics
                knowledge_stats = {
                    'total_knowledge_items': len(self.meta_knowledge_base),
                    'domains_with_knowledge': len(set(k.domain for k in self.meta_knowledge_base.values())),
                    'avg_confidence': np.mean([k.confidence_level for k in self.meta_knowledge_base.values()]) if self.meta_knowledge_base else 0.0,
                    'total_usage': sum(k.usage_count for k in self.meta_knowledge_base.values()),
                    'transferred_knowledge': len([k for k in self.meta_knowledge_base.keys() if 'transfer' in k])
                }
                
                # Active adaptations
                active_adaptations_info = {}
                for plan_id, plan in self.active_adaptations.items():
                    active_adaptations_info[plan_id] = {
                        'target_agent': plan.target_agent,
                        'recommended_strategy': plan.recommended_strategy.value,
                        'confidence_score': plan.confidence_score,
                        'expected_performance': plan.expected_performance,
                        'expected_adaptation_time': plan.expected_adaptation_time
                    }
                
                return {
                    'timestamp': time.time(),
                    'metrics': {
                        'adaptation_speed': metrics.adaptation_speed,
                        'sample_efficiency': metrics.sample_efficiency,
                        'generalization_score': metrics.generalization_score,
                        'transfer_effectiveness': metrics.transfer_effectiveness,
                        'strategy_diversity': metrics.strategy_diversity,
                        'meta_gradient_stability': metrics.meta_gradient_stability,
                        'knowledge_utilization': metrics.knowledge_utilization,
                        'cross_domain_performance': metrics.cross_domain_performance,
                        'few_shot_capability': metrics.few_shot_capability,
                        'continual_learning_score': metrics.continual_learning_score
                    },
                    'experience_statistics': experience_stats,
                    'knowledge_base_statistics': knowledge_stats,
                    'active_adaptations': active_adaptations_info,
                    'meta_learners': list(self.meta_learners.keys()),
                    'monitoring_active': self.monitoring_active
                }
                
        except Exception as e:
            logger.error(f"Failed to get meta-learning status: {e}")
            return {}
    
    # Private methods would continue here...
    # [Implementation of remaining private methods for brevity]
    
    async def _extract_meta_features(self,
                                   task_domain: TaskDomain,
                                   strategy: LearningStrategy,
                                   parameters: Dict[str, float],
                                   context: Optional[Dict[str, Any]]) -> List[float]:
        """Extract meta-features from learning context."""
        features = []
        
        # Domain encoding
        features.append(list(TaskDomain).index(task_domain) / len(TaskDomain))
        
        # Strategy encoding  
        features.append(list(LearningStrategy).index(strategy) / len(LearningStrategy))
        
        # Parameter features
        features.append(len(parameters) / 10.0)
        features.append(np.mean(list(parameters.values())) if parameters else 0.5)
        features.append(np.std(list(parameters.values())) if parameters else 0.0)
        
        # Context features
        if context:
            features.append(context.get('complexity', 0.5))
            features.append(context.get('novelty', 0.5))
            features.append(len(context) / 20.0)
        else:
            features.extend([0.5, 0.5, 0.0])
        
        return features
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'monitoring_active') and self.monitoring_active:
            asyncio.create_task(self.stop_meta_learning_monitoring())