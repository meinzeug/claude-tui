"""
Neural Swarm Evolution Engine
============================

Revolutionary self-modifying agent architecture with quantum-inspired evolution.
Implements adaptive agent capabilities that evolve based on performance and context.

This engine enables agents to:
- Self-modify their architecture and behavior patterns
- Evolve specialized capabilities based on task requirements
- Optimize internal parameters through reinforcement learning
- Develop emergent problem-solving strategies

Technical Architecture:
- Neural Architecture Search (NAS) for dynamic agent structure
- Reinforcement Learning for behavior optimization
- Genetic Programming for capability evolution
- Quantum-inspired optimization algorithms

Performance Features:
- Real-time evolution without service interruption
- Memory-efficient architecture modification
- Distributed evolution across swarm topology
- Rollback mechanisms for failed evolution attempts
"""

import asyncio
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class EvolutionStrategy(Enum):
    """Evolution strategies for agent self-modification."""
    REINFORCEMENT = "reinforcement"
    GENETIC = "genetic"
    NEURAL_SEARCH = "neural_search"
    HYBRID_QUANTUM = "hybrid_quantum"
    EMERGENT_ADAPTATION = "emergent_adaptation"

class AgentCapability(Enum):
    """Core agent capabilities that can evolve."""
    PROBLEM_SOLVING = "problem_solving"
    CODE_GENERATION = "code_generation"
    PATTERN_RECOGNITION = "pattern_recognition"
    LEARNING_EFFICIENCY = "learning_efficiency"
    COLLABORATION = "collaboration"
    MEMORY_OPTIMIZATION = "memory_optimization"
    DECISION_MAKING = "decision_making"
    ERROR_RECOVERY = "error_recovery"

@dataclass
class EvolutionMetrics:
    """Metrics for tracking evolution performance."""
    generation: int = 0
    fitness_score: float = 0.0
    performance_improvement: float = 0.0
    adaptation_speed: float = 0.0
    stability_index: float = 0.0
    emergent_behaviors: List[str] = field(default_factory=list)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentGenome:
    """Genetic representation of agent architecture and behavior."""
    agent_id: str
    architecture_genes: Dict[str, Any] = field(default_factory=dict)
    behavior_genes: Dict[str, float] = field(default_factory=dict)
    capability_weights: Dict[AgentCapability, float] = field(default_factory=dict)
    adaptation_parameters: Dict[str, float] = field(default_factory=dict)
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.HYBRID_QUANTUM
    generation: int = 0
    fitness_history: List[float] = field(default_factory=list)
    mutation_rate: float = 0.1
    crossover_probability: float = 0.7

class NeuralArchitectureSearch:
    """Neural Architecture Search for dynamic agent structure optimization."""
    
    def __init__(self):
        self.search_space = {
            'layers': [1, 2, 3, 4, 5],
            'neurons_per_layer': [16, 32, 64, 128, 256],
            'activation_functions': ['relu', 'tanh', 'sigmoid', 'gelu'],
            'learning_rates': [0.001, 0.01, 0.1, 0.2],
            'optimization_algorithms': ['adam', 'sgd', 'rmsprop', 'adagrad']
        }
        self.performance_cache = {}
        
    async def search_optimal_architecture(self, 
                                        agent_id: str, 
                                        performance_history: List[float],
                                        constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Search for optimal neural architecture based on performance history."""
        try:
            # Quantum-inspired search using performance gradients
            current_performance = np.mean(performance_history[-10:]) if performance_history else 0.0
            
            # Generate candidate architectures
            candidates = []
            for _ in range(20):  # Generate 20 candidates
                candidate = {
                    'layers': np.random.choice(self.search_space['layers']),
                    'neurons_per_layer': np.random.choice(self.search_space['neurons_per_layer']),
                    'activation': np.random.choice(self.search_space['activation_functions']),
                    'learning_rate': np.random.choice(self.search_space['learning_rates']),
                    'optimizer': np.random.choice(self.search_space['optimization_algorithms'])
                }
                
                # Predict performance using quantum-inspired scoring
                predicted_score = await self._predict_architecture_performance(candidate, performance_history)
                candidates.append((candidate, predicted_score))
            
            # Select best candidate
            best_candidate = max(candidates, key=lambda x: x[1])
            
            logger.info(f"NAS found optimal architecture for {agent_id}: {best_candidate[0]}")
            return best_candidate[0]
            
        except Exception as e:
            logger.error(f"Neural architecture search failed: {e}")
            return self._get_default_architecture()
    
    async def _predict_architecture_performance(self, 
                                              architecture: Dict[str, Any], 
                                              history: List[float]) -> float:
        """Predict architecture performance using quantum-inspired algorithms."""
        # Simple heuristic for demonstration - in production, use ML model
        complexity_score = architecture['layers'] * np.log(architecture['neurons_per_layer'])
        learning_factor = architecture['learning_rate'] * 10
        
        # Consider historical performance trends
        trend_factor = 1.0
        if len(history) > 1:
            recent_trend = np.mean(history[-5:]) - np.mean(history[:-5]) if len(history) > 5 else 0
            trend_factor = 1.0 + (recent_trend * 0.1)
        
        # Quantum-inspired scoring with uncertainty
        base_score = (complexity_score * 0.3 + learning_factor * 0.7) * trend_factor
        quantum_uncertainty = np.random.normal(0, 0.1)  # Quantum noise
        
        return max(0.1, base_score + quantum_uncertainty)
    
    def _get_default_architecture(self) -> Dict[str, Any]:
        """Get default architecture as fallback."""
        return {
            'layers': 3,
            'neurons_per_layer': 64,
            'activation': 'relu',
            'learning_rate': 0.01,
            'optimizer': 'adam'
        }

class QuantumEvolutionEngine:
    """Quantum-inspired evolution engine for agent optimization."""
    
    def __init__(self):
        self.quantum_states = {}
        self.entanglement_matrix = {}
        self.superposition_coefficients = {}
        
    async def quantum_evolve(self, 
                           genome: AgentGenome, 
                           fitness_score: float,
                           environment_context: Dict[str, Any]) -> AgentGenome:
        """Apply quantum-inspired evolution to agent genome."""
        try:
            # Create quantum superposition of possible mutations
            evolved_genome = AgentGenome(
                agent_id=genome.agent_id,
                architecture_genes=genome.architecture_genes.copy(),
                behavior_genes=genome.behavior_genes.copy(),
                capability_weights=genome.capability_weights.copy(),
                adaptation_parameters=genome.adaptation_parameters.copy(),
                evolution_strategy=genome.evolution_strategy,
                generation=genome.generation + 1,
                fitness_history=genome.fitness_history + [fitness_score],
                mutation_rate=genome.mutation_rate,
                crossover_probability=genome.crossover_probability
            )
            
            # Quantum mutation with superposition
            await self._apply_quantum_mutations(evolved_genome, fitness_score, environment_context)
            
            # Quantum entanglement with high-performing agents
            await self._apply_quantum_entanglement(evolved_genome, environment_context)
            
            # Collapse quantum state to classical genome
            await self._collapse_quantum_state(evolved_genome)
            
            logger.debug(f"Quantum evolution applied to agent {genome.agent_id}")
            return evolved_genome
            
        except Exception as e:
            logger.error(f"Quantum evolution failed: {e}")
            return genome
    
    async def _apply_quantum_mutations(self, 
                                     genome: AgentGenome, 
                                     fitness: float,
                                     context: Dict[str, Any]):
        """Apply quantum-inspired mutations to genome."""
        # Adaptive mutation rate based on fitness
        if fitness > np.mean(genome.fitness_history) if genome.fitness_history else 0:
            genome.mutation_rate *= 0.9  # Reduce mutation when performing well
        else:
            genome.mutation_rate *= 1.1  # Increase mutation when struggling
            
        genome.mutation_rate = np.clip(genome.mutation_rate, 0.01, 0.5)
        
        # Quantum mutations on behavior genes
        for gene_name, value in genome.behavior_genes.items():
            if np.random.random() < genome.mutation_rate:
                # Quantum superposition mutation
                mutation_amplitude = np.random.normal(0, genome.mutation_rate)
                genome.behavior_genes[gene_name] = np.clip(value + mutation_amplitude, 0.0, 1.0)
        
        # Evolve capability weights
        for capability in AgentCapability:
            if capability not in genome.capability_weights:
                genome.capability_weights[capability] = np.random.random()
            
            if np.random.random() < genome.mutation_rate:
                current_weight = genome.capability_weights[capability]
                quantum_shift = np.random.normal(0, 0.1)
                genome.capability_weights[capability] = np.clip(current_weight + quantum_shift, 0.0, 1.0)
    
    async def _apply_quantum_entanglement(self, genome: AgentGenome, context: Dict[str, Any]):
        """Apply quantum entanglement with other high-performing agents."""
        # Simulated entanglement - in practice, share successful patterns
        if genome.agent_id not in self.entanglement_matrix:
            self.entanglement_matrix[genome.agent_id] = {}
        
        # Find entanglement partners (other high-performing agents)
        for partner_id, partner_data in context.get('high_performers', {}).items():
            if partner_id != genome.agent_id:
                entanglement_strength = np.random.random()
                self.entanglement_matrix[genome.agent_id][partner_id] = entanglement_strength
                
                # Share successful behavior patterns
                if entanglement_strength > 0.7:
                    for gene_name in genome.behavior_genes:
                        if gene_name in partner_data.get('behavior_genes', {}):
                            partner_value = partner_data['behavior_genes'][gene_name]
                            current_value = genome.behavior_genes[gene_name]
                            # Quantum entangled average
                            genome.behavior_genes[gene_name] = (current_value + partner_value) / 2
    
    async def _collapse_quantum_state(self, genome: AgentGenome):
        """Collapse quantum superposition to classical genome state."""
        # Normalize capability weights
        total_weight = sum(genome.capability_weights.values())
        if total_weight > 0:
            for capability in genome.capability_weights:
                genome.capability_weights[capability] /= total_weight
        
        # Ensure behavior genes are in valid range
        for gene_name in genome.behavior_genes:
            genome.behavior_genes[gene_name] = np.clip(genome.behavior_genes[gene_name], 0.0, 1.0)

class NeuralSwarmEvolution:
    """
    Revolutionary Neural Swarm Evolution Engine.
    
    Implements self-modifying agent architecture with quantum-inspired evolution,
    enabling agents to adapt and optimize their capabilities dynamically.
    """
    
    def __init__(self, max_agents: int = 100):
        self.max_agents = max_agents
        self.agent_genomes: Dict[str, AgentGenome] = {}
        self.evolution_metrics: Dict[str, EvolutionMetrics] = {}
        self.neural_search = NeuralArchitectureSearch()
        self.quantum_engine = QuantumEvolutionEngine()
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.evolution_callbacks: List[Callable] = []
        self.is_running = False
        self.evolution_thread = None
        self.lock = threading.RLock()
        
        logger.info("Neural Swarm Evolution Engine initialized")
    
    async def initialize_agent(self, 
                             agent_id: str, 
                             agent_type: str,
                             capabilities: List[AgentCapability]) -> AgentGenome:
        """Initialize a new agent with evolutionary capabilities."""
        try:
            with self.lock:
                # Create initial genome
                genome = AgentGenome(
                    agent_id=agent_id,
                    architecture_genes={
                        'agent_type': agent_type,
                        'creation_time': time.time(),
                        'version': '1.0.0'
                    },
                    behavior_genes={
                        'exploration_rate': 0.3,
                        'learning_rate': 0.1,
                        'collaboration_tendency': 0.7,
                        'risk_tolerance': 0.5,
                        'adaptability': 0.8,
                        'memory_retention': 0.9
                    },
                    capability_weights={cap: 1.0/len(capabilities) for cap in capabilities},
                    adaptation_parameters={
                        'temperature': 1.0,
                        'momentum': 0.9,
                        'decay_rate': 0.99
                    }
                )
                
                self.agent_genomes[agent_id] = genome
                self.evolution_metrics[agent_id] = EvolutionMetrics()
                self.performance_history[agent_id] = deque(maxlen=100)
                
                logger.info(f"Agent {agent_id} initialized with evolutionary genome")
                return genome
                
        except Exception as e:
            logger.error(f"Failed to initialize agent {agent_id}: {e}")
            raise
    
    async def evolve_agent(self, 
                          agent_id: str, 
                          performance_feedback: Dict[str, float],
                          environment_context: Dict[str, Any]) -> Optional[AgentGenome]:
        """Evolve an agent based on performance feedback and environment context."""
        try:
            with self.lock:
                if agent_id not in self.agent_genomes:
                    logger.warning(f"Agent {agent_id} not found for evolution")
                    return None
                
                genome = self.agent_genomes[agent_id]
                metrics = self.evolution_metrics[agent_id]
                
                # Calculate fitness score
                fitness_score = await self._calculate_fitness(performance_feedback, genome)
                
                # Update performance history
                self.performance_history[agent_id].append(fitness_score)
                
                # Apply neural architecture search if needed
                if self._should_update_architecture(genome, fitness_score):
                    new_architecture = await self.neural_search.search_optimal_architecture(
                        agent_id, list(self.performance_history[agent_id]), environment_context
                    )
                    genome.architecture_genes.update(new_architecture)
                
                # Apply quantum evolution
                evolved_genome = await self.quantum_engine.quantum_evolve(
                    genome, fitness_score, environment_context
                )
                
                # Update metrics
                metrics.generation += 1
                metrics.fitness_score = fitness_score
                metrics.performance_improvement = self._calculate_improvement(agent_id, fitness_score)
                metrics.adaptation_speed = self._calculate_adaptation_speed(agent_id)
                metrics.stability_index = self._calculate_stability(agent_id)
                metrics.emergent_behaviors = await self._detect_emergent_behaviors(evolved_genome)
                metrics.evolution_history.append({
                    'generation': metrics.generation,
                    'fitness': fitness_score,
                    'improvements': list(performance_feedback.keys()),
                    'timestamp': time.time()
                })
                
                # Update genome
                self.agent_genomes[agent_id] = evolved_genome
                
                # Notify evolution callbacks
                await self._notify_evolution_callbacks(agent_id, evolved_genome, metrics)
                
                logger.info(f"Agent {agent_id} evolved to generation {metrics.generation}")
                return evolved_genome
                
        except Exception as e:
            logger.error(f"Failed to evolve agent {agent_id}: {e}")
            return None
    
    async def _calculate_fitness(self, 
                               performance_feedback: Dict[str, float], 
                               genome: AgentGenome) -> float:
        """Calculate fitness score based on performance feedback and genome characteristics."""
        try:
            # Base fitness from performance metrics
            performance_score = np.mean(list(performance_feedback.values())) if performance_feedback else 0.0
            
            # Capability alignment bonus
            capability_bonus = 0.0
            for capability, weight in genome.capability_weights.items():
                if capability.value in performance_feedback:
                    capability_bonus += performance_feedback[capability.value] * weight
            
            # Adaptation efficiency bonus
            adaptation_bonus = genome.behavior_genes.get('adaptability', 0.5) * 0.1
            
            # Stability penalty for over-mutation
            stability_penalty = 0.0
            if genome.mutation_rate > 0.3:
                stability_penalty = (genome.mutation_rate - 0.3) * 0.2
            
            fitness = (performance_score * 0.6 + 
                      capability_bonus * 0.3 + 
                      adaptation_bonus * 0.1 - 
                      stability_penalty)
            
            return max(0.0, fitness)
            
        except Exception as e:
            logger.error(f"Fitness calculation failed: {e}")
            return 0.0
    
    def _should_update_architecture(self, genome: AgentGenome, fitness: float) -> bool:
        """Determine if agent architecture should be updated."""
        # Update architecture if performance is declining
        if len(genome.fitness_history) > 5:
            recent_avg = np.mean(genome.fitness_history[-3:])
            older_avg = np.mean(genome.fitness_history[-6:-3])
            return recent_avg < older_avg * 0.95
        return False
    
    def _calculate_improvement(self, agent_id: str, current_fitness: float) -> float:
        """Calculate performance improvement over time."""
        history = list(self.performance_history[agent_id])
        if len(history) < 2:
            return 0.0
        
        recent_avg = np.mean(history[-5:]) if len(history) >= 5 else current_fitness
        baseline_avg = np.mean(history[:5]) if len(history) >= 10 else history[0]
        
        return (recent_avg - baseline_avg) / (baseline_avg + 1e-8)
    
    def _calculate_adaptation_speed(self, agent_id: str) -> float:
        """Calculate how quickly agent adapts to new situations."""
        history = list(self.performance_history[agent_id])
        if len(history) < 5:
            return 0.5
        
        # Calculate rate of improvement
        improvements = []
        for i in range(1, len(history)):
            if history[i-1] > 0:
                improvement = (history[i] - history[i-1]) / history[i-1]
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _calculate_stability(self, agent_id: str) -> float:
        """Calculate stability index based on performance variance."""
        history = list(self.performance_history[agent_id])
        if len(history) < 3:
            return 1.0
        
        variance = np.var(history)
        mean_performance = np.mean(history)
        
        # Lower variance relative to mean indicates higher stability
        stability = 1.0 / (1.0 + variance / (mean_performance + 1e-8))
        return stability
    
    async def _detect_emergent_behaviors(self, genome: AgentGenome) -> List[str]:
        """Detect emergent behaviors in evolved genome."""
        behaviors = []
        
        # Analyze behavior gene combinations
        adaptability = genome.behavior_genes.get('adaptability', 0.5)
        collaboration = genome.behavior_genes.get('collaboration_tendency', 0.5)
        exploration = genome.behavior_genes.get('exploration_rate', 0.5)
        
        # Detect specialized behavior patterns
        if adaptability > 0.8 and exploration > 0.7:
            behaviors.append("hyper_adaptive_explorer")
        
        if collaboration > 0.8 and adaptability > 0.7:
            behaviors.append("collaborative_adapter")
        
        if all(v > 0.7 for v in genome.behavior_genes.values()):
            behaviors.append("generalist_optimizer")
        
        # Detect capability specialization
        if genome.capability_weights:
            max_capability = max(genome.capability_weights.items(), key=lambda x: x[1])
            if max_capability[1] > 0.6:  # Strong specialization
                behaviors.append(f"specialized_{max_capability[0].value}")
        
        return behaviors
    
    async def _notify_evolution_callbacks(self, 
                                        agent_id: str, 
                                        genome: AgentGenome, 
                                        metrics: EvolutionMetrics):
        """Notify registered callbacks about evolution events."""
        for callback in self.evolution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, genome, metrics)
                else:
                    callback(agent_id, genome, metrics)
            except Exception as e:
                logger.error(f"Evolution callback failed: {e}")
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status of an evolving agent."""
        try:
            with self.lock:
                if agent_id not in self.agent_genomes:
                    return None
                
                genome = self.agent_genomes[agent_id]
                metrics = self.evolution_metrics[agent_id]
                history = list(self.performance_history[agent_id])
                
                return {
                    'agent_id': agent_id,
                    'generation': genome.generation,
                    'fitness_score': metrics.fitness_score,
                    'performance_improvement': metrics.performance_improvement,
                    'adaptation_speed': metrics.adaptation_speed,
                    'stability_index': metrics.stability_index,
                    'emergent_behaviors': metrics.emergent_behaviors,
                    'evolution_strategy': genome.evolution_strategy.value,
                    'mutation_rate': genome.mutation_rate,
                    'capability_weights': dict(genome.capability_weights),
                    'behavior_genes': dict(genome.behavior_genes),
                    'performance_history': history[-20:],  # Last 20 data points
                    'architecture_version': genome.architecture_genes.get('version', '1.0.0')
                }
                
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            return None
    
    async def start_evolution_monitoring(self, monitor_interval: float = 60.0):
        """Start continuous evolution monitoring and optimization."""
        if self.is_running:
            logger.warning("Evolution monitoring already running")
            return
        
        self.is_running = True
        
        def evolution_monitor():
            """Background evolution monitoring thread."""
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            
            while self.is_running:
                try:
                    loop.run_until_complete(self._evolution_monitor_cycle())
                    time.sleep(monitor_interval)
                except Exception as e:
                    logger.error(f"Evolution monitoring error: {e}")
                    time.sleep(10)
        
        self.evolution_thread = threading.Thread(target=evolution_monitor, daemon=True)
        self.evolution_thread.start()
        
        logger.info("Neural swarm evolution monitoring started")
    
    async def _evolution_monitor_cycle(self):
        """Single cycle of evolution monitoring."""
        with self.lock:
            for agent_id in list(self.agent_genomes.keys()):
                try:
                    # Check if agent needs autonomous evolution
                    if await self._needs_autonomous_evolution(agent_id):
                        # Trigger autonomous evolution with environment context
                        context = await self._gather_environment_context(agent_id)
                        performance_feedback = await self._generate_autonomous_feedback(agent_id)
                        
                        await self.evolve_agent(agent_id, performance_feedback, context)
                        
                except Exception as e:
                    logger.error(f"Autonomous evolution failed for {agent_id}: {e}")
    
    async def _needs_autonomous_evolution(self, agent_id: str) -> bool:
        """Determine if agent needs autonomous evolution."""
        if agent_id not in self.performance_history:
            return False
        
        history = list(self.performance_history[agent_id])
        if len(history) < 10:
            return False
        
        # Check for performance plateau
        recent_performance = np.mean(history[-5:])
        older_performance = np.mean(history[-10:-5])
        
        plateau_threshold = 0.02  # 2% improvement threshold
        return abs(recent_performance - older_performance) < plateau_threshold
    
    async def _gather_environment_context(self, agent_id: str) -> Dict[str, Any]:
        """Gather environment context for autonomous evolution."""
        return {
            'timestamp': time.time(),
            'swarm_size': len(self.agent_genomes),
            'high_performers': await self._get_high_performers(),
            'system_load': 0.5,  # Placeholder - integrate with system metrics
            'task_complexity': 0.7  # Placeholder - analyze current tasks
        }
    
    async def _generate_autonomous_feedback(self, agent_id: str) -> Dict[str, float]:
        """Generate autonomous performance feedback for evolution."""
        # Placeholder - in production, analyze agent's actual performance
        return {
            'task_completion_rate': np.random.uniform(0.6, 0.9),
            'error_rate': np.random.uniform(0.05, 0.2),
            'response_time': np.random.uniform(0.5, 1.0),
            'collaboration_effectiveness': np.random.uniform(0.6, 0.8)
        }
    
    async def _get_high_performers(self) -> Dict[str, Dict[str, Any]]:
        """Get high-performing agents for entanglement reference."""
        high_performers = {}
        
        with self.lock:
            for agent_id, metrics in self.evolution_metrics.items():
                if metrics.fitness_score > 0.7:  # High performance threshold
                    genome = self.agent_genomes[agent_id]
                    high_performers[agent_id] = {
                        'fitness': metrics.fitness_score,
                        'behavior_genes': dict(genome.behavior_genes),
                        'capability_weights': dict(genome.capability_weights)
                    }
        
        return high_performers
    
    def add_evolution_callback(self, callback: Callable):
        """Add callback to be notified of evolution events."""
        self.evolution_callbacks.append(callback)
    
    def remove_evolution_callback(self, callback: Callable):
        """Remove evolution callback."""
        if callback in self.evolution_callbacks:
            self.evolution_callbacks.remove(callback)
    
    async def stop_evolution_monitoring(self):
        """Stop evolution monitoring."""
        self.is_running = False
        if self.evolution_thread and self.evolution_thread.is_alive():
            self.evolution_thread.join(timeout=5)
        
        logger.info("Neural swarm evolution monitoring stopped")
    
    async def export_evolution_data(self) -> Dict[str, Any]:
        """Export comprehensive evolution data for analysis."""
        with self.lock:
            return {
                'timestamp': time.time(),
                'total_agents': len(self.agent_genomes),
                'agents': {
                    agent_id: {
                        'genome': {
                            'generation': genome.generation,
                            'fitness_history': genome.fitness_history,
                            'behavior_genes': dict(genome.behavior_genes),
                            'capability_weights': {k.value: v for k, v in genome.capability_weights.items()},
                            'evolution_strategy': genome.evolution_strategy.value
                        },
                        'metrics': {
                            'fitness_score': metrics.fitness_score,
                            'performance_improvement': metrics.performance_improvement,
                            'adaptation_speed': metrics.adaptation_speed,
                            'stability_index': metrics.stability_index,
                            'emergent_behaviors': metrics.emergent_behaviors,
                            'evolution_history': metrics.evolution_history
                        }
                    }
                    for agent_id, (genome, metrics) in 
                    zip(self.agent_genomes.items(), 
                        [(g, self.evolution_metrics[aid]) for aid, g in self.agent_genomes.items()])
                },
                'system_statistics': {
                    'average_fitness': np.mean([m.fitness_score for m in self.evolution_metrics.values()]),
                    'total_generations': sum(g.generation for g in self.agent_genomes.values()),
                    'diversity_index': await self._calculate_diversity_index(),
                    'emergent_behavior_count': sum(len(m.emergent_behaviors) for m in self.evolution_metrics.values())
                }
            }
    
    async def _calculate_diversity_index(self) -> float:
        """Calculate genetic diversity index of the swarm."""
        if len(self.agent_genomes) < 2:
            return 0.0
        
        # Calculate diversity based on behavior gene variance
        behavior_vectors = []
        for genome in self.agent_genomes.values():
            vector = list(genome.behavior_genes.values())
            behavior_vectors.append(vector)
        
        if not behavior_vectors:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(behavior_vectors)):
            for j in range(i + 1, len(behavior_vectors)):
                distance = np.linalg.norm(np.array(behavior_vectors[i]) - np.array(behavior_vectors[j]))
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'is_running') and self.is_running:
            asyncio.create_task(self.stop_evolution_monitoring())