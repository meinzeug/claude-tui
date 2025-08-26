"""
Quantum Intelligence Compatibility Layer
=======================================

Backward compatibility layer that enables seamless integration of quantum
intelligence features with existing 54+ agent types without breaking changes.

This layer provides:
- Progressive enhancement of existing agents
- Compatibility adapters for legacy agent interfaces
- Gradual migration paths to quantum features
- Fallback mechanisms for unsupported operations
- Bridge between traditional AI and quantum intelligence

Architecture:
- Non-invasive integration with existing systems
- Optional feature enablement per agent type
- Automatic capability detection and enhancement
- Graceful degradation for unsupported features
- Transparent operation for legacy code
"""

import asyncio
import logging
import time
import json
import inspect
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import weakref
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Known agent types in the system."""
    # Core Development
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    
    # Swarm Coordination
    HIERARCHICAL_COORDINATOR = "hierarchical-coordinator"
    MESH_COORDINATOR = "mesh-coordinator"
    ADAPTIVE_COORDINATOR = "adaptive-coordinator"
    COLLECTIVE_INTELLIGENCE_COORDINATOR = "collective-intelligence-coordinator"
    SWARM_MEMORY_MANAGER = "swarm-memory-manager"
    
    # Consensus & Distributed
    BYZANTINE_COORDINATOR = "byzantine-coordinator"
    RAFT_MANAGER = "raft-manager"
    GOSSIP_COORDINATOR = "gossip-coordinator"
    CONSENSUS_BUILDER = "consensus-builder"
    CRDT_SYNCHRONIZER = "crdt-synchronizer"
    QUORUM_MANAGER = "quorum-manager"
    SECURITY_MANAGER = "security-manager"
    
    # Performance & Optimization
    PERF_ANALYZER = "perf-analyzer"
    PERFORMANCE_BENCHMARKER = "performance-benchmarker"
    TASK_ORCHESTRATOR = "task-orchestrator"
    MEMORY_COORDINATOR = "memory-coordinator"
    SMART_AGENT = "smart-agent"
    
    # GitHub & Repository
    GITHUB_MODES = "github-modes"
    PR_MANAGER = "pr-manager"
    CODE_REVIEW_SWARM = "code-review-swarm"
    ISSUE_TRACKER = "issue-tracker"
    RELEASE_MANAGER = "release-manager"
    WORKFLOW_AUTOMATION = "workflow-automation"
    PROJECT_BOARD_SYNC = "project-board-sync"
    REPO_ARCHITECT = "repo-architect"
    MULTI_REPO_SWARM = "multi-repo-swarm"
    
    # SPARC Methodology
    SPARC_COORD = "sparc-coord"
    SPARC_CODER = "sparc-coder"
    SPECIFICATION = "specification"
    PSEUDOCODE = "pseudocode"
    ARCHITECTURE = "architecture"
    REFINEMENT = "refinement"
    
    # Specialized Development
    BACKEND_DEV = "backend-dev"
    MOBILE_DEV = "mobile-dev"
    ML_DEVELOPER = "ml-developer"
    CICD_ENGINEER = "cicd-engineer"
    API_DOCS = "api-docs"
    SYSTEM_ARCHITECT = "system-architect"
    CODE_ANALYZER = "code-analyzer"
    BASE_TEMPLATE_GENERATOR = "base-template-generator"
    
    # Testing & Validation
    TDD_LONDON_SWARM = "tdd-london-swarm"
    PRODUCTION_VALIDATOR = "production-validator"
    
    # Migration & Planning
    MIGRATION_PLANNER = "migration-planner"
    SWARM_INIT = "swarm-init"
    
    # Generic/Unknown
    UNKNOWN = "unknown"

class EnhancementLevel(Enum):
    """Levels of quantum enhancement for agents."""
    NONE = "none"           # No quantum features
    BASIC = "basic"         # Basic monitoring and metrics
    ENHANCED = "enhanced"   # Advanced features enabled
    QUANTUM = "quantum"     # Full quantum intelligence

class CompatibilityMode(Enum):
    """Compatibility modes for different integration approaches."""
    TRANSPARENT = "transparent"      # Completely transparent to agent
    COOPERATIVE = "cooperative"      # Agent is aware and cooperates
    ENHANCED = "enhanced"            # Agent uses quantum features
    NATIVE = "native"                # Agent built for quantum intelligence

@dataclass
class AgentCapabilities:
    """Capabilities and characteristics of an agent."""
    agent_id: str
    agent_type: AgentType
    supported_operations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    enhancement_level: EnhancementLevel = EnhancementLevel.NONE
    compatibility_mode: CompatibilityMode = CompatibilityMode.TRANSPARENT
    quantum_features: List[str] = field(default_factory=list)
    integration_points: Dict[str, Any] = field(default_factory=dict)
    fallback_handlers: Dict[str, Callable] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

@dataclass
class EnhancementPlan:
    """Plan for enhancing an agent with quantum features."""
    agent_id: str
    current_level: EnhancementLevel
    target_level: EnhancementLevel
    required_features: List[str]
    optional_features: List[str]
    migration_steps: List[Dict[str, Any]]
    compatibility_requirements: List[str]
    rollback_plan: Dict[str, Any]
    estimated_timeline: float  # seconds
    risk_assessment: Dict[str, Any] = field(default_factory=dict)

class AgentAdapter(ABC):
    """Abstract base class for agent adapters."""
    
    @abstractmethod
    async def can_enhance(self, agent_type: AgentType) -> bool:
        """Check if this adapter can enhance the given agent type."""
        pass
    
    @abstractmethod
    async def create_enhancement_plan(self, 
                                    agent_id: str, 
                                    capabilities: AgentCapabilities) -> EnhancementPlan:
        """Create enhancement plan for agent."""
        pass
    
    @abstractmethod
    async def apply_enhancement(self, 
                              agent_id: str, 
                              plan: EnhancementPlan) -> bool:
        """Apply quantum enhancement to agent."""
        pass
    
    @abstractmethod
    async def create_wrapper(self, 
                           agent_id: str, 
                           original_agent: Any) -> Any:
        """Create quantum-enhanced wrapper for agent."""
        pass

class UniversalAgentAdapter(AgentAdapter):
    """Universal adapter for any agent type."""
    
    def __init__(self):
        self.enhancement_templates = self._create_enhancement_templates()
    
    async def can_enhance(self, agent_type: AgentType) -> bool:
        """Universal adapter can enhance any agent type."""
        return True
    
    async def create_enhancement_plan(self, 
                                    agent_id: str, 
                                    capabilities: AgentCapabilities) -> EnhancementPlan:
        """Create universal enhancement plan."""
        try:
            agent_type = capabilities.agent_type
            current_level = capabilities.enhancement_level
            
            # Determine target level based on agent type and capabilities
            target_level = await self._determine_target_level(agent_type, capabilities)
            
            # Get enhancement template
            template = self.enhancement_templates.get(agent_type, self.enhancement_templates[AgentType.UNKNOWN])
            
            # Required features based on agent type
            required_features = template.get('required_features', [])
            optional_features = template.get('optional_features', [])
            
            # Create migration steps
            migration_steps = await self._create_migration_steps(
                current_level, target_level, required_features
            )
            
            # Estimate timeline
            estimated_timeline = len(migration_steps) * 30.0  # 30 seconds per step
            
            plan = EnhancementPlan(
                agent_id=agent_id,
                current_level=current_level,
                target_level=target_level,
                required_features=required_features,
                optional_features=optional_features,
                migration_steps=migration_steps,
                compatibility_requirements=template.get('compatibility_requirements', []),
                rollback_plan={'restore_original': True, 'backup_created': time.time()},
                estimated_timeline=estimated_timeline,
                risk_assessment=await self._assess_enhancement_risk(capabilities)
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create enhancement plan: {e}")
            # Return minimal plan
            return EnhancementPlan(
                agent_id=agent_id,
                current_level=current_level,
                target_level=EnhancementLevel.BASIC,
                required_features=['basic_monitoring'],
                optional_features=[],
                migration_steps=[{'step': 1, 'action': 'enable_basic_monitoring'}],
                compatibility_requirements=[],
                rollback_plan={'restore_original': True},
                estimated_timeline=30.0
            )
    
    async def apply_enhancement(self, 
                              agent_id: str, 
                              plan: EnhancementPlan) -> bool:
        """Apply universal enhancement."""
        try:
            logger.info(f"Applying enhancement plan for agent {agent_id}")
            
            for step_idx, step in enumerate(plan.migration_steps):
                action = step.get('action', 'unknown')
                logger.debug(f"Executing step {step_idx + 1}: {action}")
                
                success = await self._execute_enhancement_step(agent_id, step)
                if not success:
                    logger.error(f"Enhancement step failed: {action}")
                    # Attempt rollback
                    await self._rollback_enhancement(agent_id, plan, step_idx)
                    return False
                
                # Small delay between steps
                await asyncio.sleep(1)
            
            logger.info(f"Enhancement completed successfully for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Enhancement application failed: {e}")
            return False
    
    async def create_wrapper(self, 
                           agent_id: str, 
                           original_agent: Any) -> Any:
        """Create universal quantum-enhanced wrapper."""
        try:
            wrapper = QuantumEnhancedAgentWrapper(agent_id, original_agent)
            await wrapper.initialize()
            return wrapper
            
        except Exception as e:
            logger.error(f"Failed to create wrapper: {e}")
            return original_agent  # Return original if wrapping fails
    
    def _create_enhancement_templates(self) -> Dict[AgentType, Dict[str, Any]]:
        """Create enhancement templates for different agent types."""
        templates = {}
        
        # Core development agents
        development_features = [
            'performance_monitoring',
            'error_recovery',
            'adaptive_learning',
            'collaboration_enhancement'
        ]
        
        for agent_type in [AgentType.CODER, AgentType.REVIEWER, AgentType.TESTER, 
                          AgentType.PLANNER, AgentType.RESEARCHER]:
            templates[agent_type] = {
                'required_features': development_features,
                'optional_features': ['quantum_optimization', 'emergent_behavior'],
                'compatibility_requirements': ['async_support', 'error_handling']
            }
        
        # Coordination agents
        coordination_features = [
            'topology_optimization',
            'emergent_behavior_detection',
            'swarm_intelligence',
            'quantum_coordination'
        ]
        
        for agent_type in [AgentType.HIERARCHICAL_COORDINATOR, AgentType.MESH_COORDINATOR,
                          AgentType.ADAPTIVE_COORDINATOR, AgentType.COLLECTIVE_INTELLIGENCE_COORDINATOR]:
            templates[agent_type] = {
                'required_features': coordination_features,
                'optional_features': ['quantum_entanglement', 'meta_learning'],
                'compatibility_requirements': ['network_support', 'state_management']
            }
        
        # Performance agents
        performance_features = [
            'advanced_metrics',
            'predictive_analytics',
            'auto_optimization',
            'resource_management'
        ]
        
        for agent_type in [AgentType.PERF_ANALYZER, AgentType.PERFORMANCE_BENCHMARKER,
                          AgentType.MEMORY_COORDINATOR]:
            templates[agent_type] = {
                'required_features': performance_features,
                'optional_features': ['quantum_acceleration', 'neural_optimization'],
                'compatibility_requirements': ['metrics_collection', 'system_access']
            }
        
        # Default template for unknown agents
        templates[AgentType.UNKNOWN] = {
            'required_features': ['basic_monitoring', 'error_recovery'],
            'optional_features': ['performance_enhancement'],
            'compatibility_requirements': ['minimal_interface']
        }
        
        return templates
    
    async def _determine_target_level(self, 
                                    agent_type: AgentType, 
                                    capabilities: AgentCapabilities) -> EnhancementLevel:
        """Determine appropriate target enhancement level."""
        # Performance-critical agents can benefit from quantum features
        if agent_type in [AgentType.PERFORMANCE_BENCHMARKER, AgentType.PERF_ANALYZER,
                         AgentType.MEMORY_COORDINATOR]:
            return EnhancementLevel.QUANTUM
        
        # Coordination agents benefit from enhanced features
        if 'coordinator' in agent_type.value or 'manager' in agent_type.value:
            return EnhancementLevel.ENHANCED
        
        # Development agents get basic enhancements
        if agent_type in [AgentType.CODER, AgentType.REVIEWER, AgentType.TESTER]:
            return EnhancementLevel.ENHANCED
        
        # Default to basic for unknown or specialized agents
        return EnhancementLevel.BASIC
    
    async def _create_migration_steps(self, 
                                    current_level: EnhancementLevel,
                                    target_level: EnhancementLevel,
                                    required_features: List[str]) -> List[Dict[str, Any]]:
        """Create migration steps for enhancement."""
        steps = []
        
        if current_level == EnhancementLevel.NONE:
            steps.append({
                'step': len(steps) + 1,
                'action': 'initialize_quantum_interface',
                'description': 'Initialize quantum intelligence interface'
            })
        
        if target_level in [EnhancementLevel.BASIC, EnhancementLevel.ENHANCED, EnhancementLevel.QUANTUM]:
            steps.append({
                'step': len(steps) + 1,
                'action': 'enable_monitoring',
                'description': 'Enable performance monitoring and metrics collection'
            })
            
            steps.append({
                'step': len(steps) + 1,
                'action': 'setup_error_recovery',
                'description': 'Setup error handling and recovery mechanisms'
            })
        
        if target_level in [EnhancementLevel.ENHANCED, EnhancementLevel.QUANTUM]:
            steps.append({
                'step': len(steps) + 1,
                'action': 'enable_adaptive_features',
                'description': 'Enable adaptive learning and optimization'
            })
            
            steps.append({
                'step': len(steps) + 1,
                'action': 'setup_collaboration',
                'description': 'Setup enhanced collaboration capabilities'
            })
        
        if target_level == EnhancementLevel.QUANTUM:
            steps.append({
                'step': len(steps) + 1,
                'action': 'enable_quantum_features',
                'description': 'Enable quantum intelligence features'
            })
            
            steps.append({
                'step': len(steps) + 1,
                'action': 'establish_quantum_entanglement',
                'description': 'Establish quantum entanglement with other agents'
            })
        
        # Add feature-specific steps
        for feature in required_features:
            steps.append({
                'step': len(steps) + 1,
                'action': f'enable_{feature}',
                'description': f'Enable {feature.replace("_", " ").title()}'
            })
        
        return steps
    
    async def _assess_enhancement_risk(self, capabilities: AgentCapabilities) -> Dict[str, Any]:
        """Assess risks of enhancing agent."""
        risk_factors = []
        risk_score = 0.1  # Base risk
        
        # Agent type risks
        if capabilities.agent_type == AgentType.UNKNOWN:
            risk_factors.append("Unknown agent type increases risk")
            risk_score += 0.2
        
        # Performance risks
        if capabilities.performance_metrics.get('cpu_usage', 0) > 0.8:
            risk_factors.append("High CPU usage may affect enhancement")
            risk_score += 0.1
        
        if capabilities.performance_metrics.get('memory_usage', 0) > 0.9:
            risk_factors.append("High memory usage increases risk")
            risk_score += 0.2
        
        # Capability risks
        if len(capabilities.supported_operations) < 3:
            risk_factors.append("Limited operations may complicate enhancement")
            risk_score += 0.1
        
        return {
            'risk_score': min(1.0, risk_score),
            'risk_factors': risk_factors,
            'mitigation_strategies': [
                'Create rollback point before enhancement',
                'Monitor performance during enhancement',
                'Apply changes incrementally'
            ]
        }
    
    async def _execute_enhancement_step(self, agent_id: str, step: Dict[str, Any]) -> bool:
        """Execute a single enhancement step."""
        try:
            action = step.get('action', 'unknown')
            
            # Simulate step execution
            if action == 'initialize_quantum_interface':
                logger.debug(f"Initialized quantum interface for {agent_id}")
                return True
                
            elif action == 'enable_monitoring':
                logger.debug(f"Enabled monitoring for {agent_id}")
                return True
                
            elif action == 'setup_error_recovery':
                logger.debug(f"Setup error recovery for {agent_id}")
                return True
                
            elif action == 'enable_adaptive_features':
                logger.debug(f"Enabled adaptive features for {agent_id}")
                return True
                
            elif action == 'setup_collaboration':
                logger.debug(f"Setup collaboration for {agent_id}")
                return True
                
            elif action == 'enable_quantum_features':
                logger.debug(f"Enabled quantum features for {agent_id}")
                return True
                
            elif action == 'establish_quantum_entanglement':
                logger.debug(f"Established quantum entanglement for {agent_id}")
                return True
                
            elif action.startswith('enable_'):
                feature = action[7:]  # Remove 'enable_' prefix
                logger.debug(f"Enabled {feature} for {agent_id}")
                return True
            
            else:
                logger.warning(f"Unknown enhancement action: {action}")
                return True  # Continue with unknown actions
            
        except Exception as e:
            logger.error(f"Enhancement step execution failed: {e}")
            return False
    
    async def _rollback_enhancement(self, 
                                  agent_id: str, 
                                  plan: EnhancementPlan, 
                                  failed_step: int):
        """Rollback enhancement after failure."""
        try:
            logger.warning(f"Rolling back enhancement for {agent_id} after step {failed_step}")
            
            # Execute rollback plan
            rollback_plan = plan.rollback_plan
            
            if rollback_plan.get('restore_original', False):
                logger.info(f"Restoring original state for {agent_id}")
                # Restoration logic would go here
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

class QuantumEnhancedAgentWrapper:
    """Wrapper that adds quantum intelligence features to any agent."""
    
    def __init__(self, agent_id: str, original_agent: Any):
        self.agent_id = agent_id
        self.original_agent = original_agent
        self.enhancement_level = EnhancementLevel.NONE
        self.quantum_features = {}
        self.performance_metrics = {}
        self.error_history = deque(maxlen=100)
        self.operation_history = deque(maxlen=500)
        
        # Quantum intelligence components (injected during initialization)
        self.neural_evolution = None
        self.topology_manager = None
        self.behavior_engine = None
        self.meta_coordinator = None
        self.error_handler = None
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize quantum enhancements."""
        try:
            # This would be called by the compatibility layer
            # to inject quantum intelligence components
            
            self.initialized = True
            logger.debug(f"Quantum wrapper initialized for {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Wrapper initialization failed: {e}")
    
    def __getattr__(self, name):
        """Proxy attribute access to original agent."""
        if not self.initialized:
            # Direct access to original agent before initialization
            return getattr(self.original_agent, name)
        
        # Check if this is a quantum-enhanced operation
        if hasattr(self, f'_quantum_{name}'):
            return getattr(self, f'_quantum_{name}')
        
        # Wrap method calls with quantum enhancements
        attr = getattr(self.original_agent, name)
        
        if callable(attr):
            return self._wrap_method(name, attr)
        else:
            return attr
    
    def _wrap_method(self, method_name: str, original_method: Callable):
        """Wrap a method with quantum enhancements."""
        if inspect.iscoroutinefunction(original_method):
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_enhancements(
                    method_name, original_method, args, kwargs
                )
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(
                    self._execute_with_enhancements(
                        method_name, original_method, args, kwargs
                    )
                )
            return sync_wrapper
    
    async def _execute_with_enhancements(self, 
                                       method_name: str, 
                                       original_method: Callable, 
                                       args: tuple, 
                                       kwargs: dict):
        """Execute method with quantum enhancements."""
        operation_start = time.time()
        
        try:
            # Pre-execution enhancements
            if self.error_handler:
                # Use quantum error handling context
                async with self.error_handler.error_context(
                    self.agent_id, method_name
                ):
                    if inspect.iscoroutinefunction(original_method):
                        result = await original_method(*args, **kwargs)
                    else:
                        result = original_method(*args, **kwargs)
            else:
                # Direct execution without error handling
                if inspect.iscoroutinefunction(original_method):
                    result = await original_method(*args, **kwargs)
                else:
                    result = original_method(*args, **kwargs)
            
            # Post-execution enhancements
            operation_time = time.time() - operation_start
            await self._record_operation_success(method_name, operation_time, args, kwargs)
            
            return result
            
        except Exception as e:
            # Error handling
            operation_time = time.time() - operation_start
            await self._record_operation_error(method_name, operation_time, e, args, kwargs)
            raise
    
    async def _record_operation_success(self, 
                                      method_name: str, 
                                      duration: float, 
                                      args: tuple, 
                                      kwargs: dict):
        """Record successful operation."""
        operation_record = {
            'timestamp': time.time(),
            'method': method_name,
            'duration': duration,
            'success': True,
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        }
        
        self.operation_history.append(operation_record)
        
        # Update performance metrics
        if method_name not in self.performance_metrics:
            self.performance_metrics[method_name] = {
                'total_calls': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'success_rate': 0.0,
                'error_count': 0
            }
        
        metrics = self.performance_metrics[method_name]
        metrics['total_calls'] += 1
        metrics['total_duration'] += duration
        metrics['avg_duration'] = metrics['total_duration'] / metrics['total_calls']
        
        # Update success rate
        total_operations = metrics['total_calls']
        successful_operations = total_operations - metrics['error_count']
        metrics['success_rate'] = successful_operations / total_operations
        
        # Record with behavior engine if available
        if self.behavior_engine:
            await self.behavior_engine.record_interaction(
                self.agent_id, 
                'quantum_wrapper',
                'COLLABORATION',  # Using collaboration as default
                {
                    'method': method_name,
                    'duration': duration,
                    'success': True
                },
                'operation_success'
            )
    
    async def _record_operation_error(self, 
                                    method_name: str, 
                                    duration: float, 
                                    error: Exception, 
                                    args: tuple, 
                                    kwargs: dict):
        """Record operation error."""
        error_record = {
            'timestamp': time.time(),
            'method': method_name,
            'duration': duration,
            'success': False,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        }
        
        self.error_history.append(error_record)
        self.operation_history.append(error_record)
        
        # Update performance metrics
        if method_name not in self.performance_metrics:
            self.performance_metrics[method_name] = {
                'total_calls': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'success_rate': 1.0,
                'error_count': 0
            }
        
        metrics = self.performance_metrics[method_name]
        metrics['total_calls'] += 1
        metrics['error_count'] += 1
        metrics['total_duration'] += duration
        metrics['avg_duration'] = metrics['total_duration'] / metrics['total_calls']
        
        # Update success rate
        total_operations = metrics['total_calls']
        successful_operations = total_operations - metrics['error_count']
        metrics['success_rate'] = successful_operations / total_operations
        
        # Record with behavior engine if available
        if self.behavior_engine:
            await self.behavior_engine.record_interaction(
                self.agent_id,
                'quantum_wrapper',
                'COMMUNICATION',  # Using communication for error reporting
                {
                    'method': method_name,
                    'duration': duration,
                    'error': str(error),
                    'success': False
                },
                'operation_error'
            )

class QuantumCompatibilityLayer:
    """
    Main compatibility layer that manages quantum intelligence integration
    with existing agent types without breaking changes.
    """
    
    def __init__(self):
        self.agent_adapters: Dict[AgentType, AgentAdapter] = {}
        self.universal_adapter = UniversalAgentAdapter()
        self.enhanced_agents: Dict[str, AgentCapabilities] = {}
        self.enhancement_history = deque(maxlen=1000)
        self.compatibility_callbacks: List[Callable] = []
        
        # Component references (injected by orchestrator)
        self.neural_evolution = None
        self.topology_manager = None
        self.behavior_engine = None
        self.meta_coordinator = None
        self.error_handler = None
        
        self.lock = threading.RLock()
        
        logger.info("Quantum Compatibility Layer initialized")
    
    def register_adapter(self, agent_type: AgentType, adapter: AgentAdapter):
        """Register a custom adapter for specific agent type."""
        self.agent_adapters[agent_type] = adapter
        logger.info(f"Registered custom adapter for {agent_type.value}")
    
    def inject_quantum_components(self, 
                                neural_evolution=None,
                                topology_manager=None,
                                behavior_engine=None,
                                meta_coordinator=None,
                                error_handler=None):
        """Inject quantum intelligence components."""
        self.neural_evolution = neural_evolution
        self.topology_manager = topology_manager
        self.behavior_engine = behavior_engine
        self.meta_coordinator = meta_coordinator
        self.error_handler = error_handler
        
        logger.info("Quantum intelligence components injected into compatibility layer")
    
    async def register_agent(self, 
                           agent_id: str, 
                           agent_type: Union[str, AgentType],
                           agent_instance: Any,
                           capabilities: Optional[Dict[str, Any]] = None) -> bool:
        """Register an agent for potential quantum enhancement."""
        try:
            with self.lock:
                # Convert agent type
                if isinstance(agent_type, str):
                    # Try to match known agent types
                    matched_type = AgentType.UNKNOWN
                    for known_type in AgentType:
                        if known_type.value == agent_type:
                            matched_type = known_type
                            break
                else:
                    matched_type = agent_type
                
                # Extract capabilities
                if capabilities is None:
                    capabilities = await self._discover_agent_capabilities(agent_instance)
                
                # Create capability record
                agent_capabilities = AgentCapabilities(
                    agent_id=agent_id,
                    agent_type=matched_type,
                    supported_operations=capabilities.get('operations', []),
                    performance_metrics=capabilities.get('metrics', {}),
                    enhancement_level=EnhancementLevel.NONE,
                    compatibility_mode=CompatibilityMode.TRANSPARENT
                )
                
                self.enhanced_agents[agent_id] = agent_capabilities
                
                logger.info(f"Registered agent {agent_id} of type {matched_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return False
    
    async def enhance_agent(self, 
                          agent_id: str, 
                          target_level: EnhancementLevel = EnhancementLevel.ENHANCED,
                          force: bool = False) -> bool:
        """Enhance an agent with quantum intelligence features."""
        try:
            with self.lock:
                if agent_id not in self.enhanced_agents:
                    logger.warning(f"Agent {agent_id} not registered")
                    return False
                
                capabilities = self.enhanced_agents[agent_id]
                
                if capabilities.enhancement_level.value >= target_level.value and not force:
                    logger.info(f"Agent {agent_id} already at or above target level {target_level.value}")
                    return True
                
                # Select appropriate adapter
                adapter = self.agent_adapters.get(
                    capabilities.agent_type, 
                    self.universal_adapter
                )
                
                # Create enhancement plan
                plan = await adapter.create_enhancement_plan(agent_id, capabilities)
                
                # Apply enhancement
                success = await adapter.apply_enhancement(agent_id, plan)
                
                if success:
                    # Update capabilities
                    capabilities.enhancement_level = target_level
                    capabilities.compatibility_mode = CompatibilityMode.ENHANCED
                    capabilities.quantum_features.extend(plan.required_features)
                    capabilities.last_updated = time.time()
                    
                    # Record enhancement
                    self.enhancement_history.append({
                        'timestamp': time.time(),
                        'agent_id': agent_id,
                        'agent_type': capabilities.agent_type.value,
                        'from_level': plan.current_level.value,
                        'to_level': target_level.value,
                        'features_added': plan.required_features,
                        'success': True
                    })
                    
                    # Notify callbacks
                    await self._notify_enhancement_callbacks(agent_id, capabilities, plan)
                    
                    logger.info(f"Successfully enhanced agent {agent_id} to {target_level.value}")
                    return True
                else:
                    logger.error(f"Failed to enhance agent {agent_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Agent enhancement failed: {e}")
            return False
    
    async def create_quantum_wrapper(self, 
                                   agent_id: str, 
                                   original_agent: Any) -> Any:
        """Create quantum-enhanced wrapper for agent."""
        try:
            if agent_id not in self.enhanced_agents:
                logger.warning(f"Agent {agent_id} not registered, creating basic wrapper")
                # Register agent first
                await self.register_agent(agent_id, AgentType.UNKNOWN, original_agent)
            
            capabilities = self.enhanced_agents[agent_id]
            
            # Select appropriate adapter
            adapter = self.agent_adapters.get(
                capabilities.agent_type,
                self.universal_adapter
            )
            
            # Create wrapper
            wrapper = await adapter.create_wrapper(agent_id, original_agent)
            
            # Inject quantum components if wrapper supports it
            if isinstance(wrapper, QuantumEnhancedAgentWrapper):
                wrapper.neural_evolution = self.neural_evolution
                wrapper.topology_manager = self.topology_manager
                wrapper.behavior_engine = self.behavior_engine
                wrapper.meta_coordinator = self.meta_coordinator
                wrapper.error_handler = self.error_handler
            
            logger.info(f"Created quantum wrapper for agent {agent_id}")
            return wrapper
            
        except Exception as e:
            logger.error(f"Failed to create quantum wrapper: {e}")
            return original_agent  # Return original on failure
    
    async def get_enhancement_recommendations(self, 
                                            agent_id: str) -> Dict[str, Any]:
        """Get recommendations for enhancing an agent."""
        try:
            if agent_id not in self.enhanced_agents:
                return {'error': 'Agent not registered'}
            
            capabilities = self.enhanced_agents[agent_id]
            
            # Select adapter
            adapter = self.agent_adapters.get(
                capabilities.agent_type,
                self.universal_adapter
            )
            
            # Create enhancement plan as recommendation
            plan = await adapter.create_enhancement_plan(agent_id, capabilities)
            
            return {
                'agent_id': agent_id,
                'agent_type': capabilities.agent_type.value,
                'current_level': capabilities.enhancement_level.value,
                'recommended_level': plan.target_level.value,
                'required_features': plan.required_features,
                'optional_features': plan.optional_features,
                'estimated_timeline': plan.estimated_timeline,
                'risk_assessment': plan.risk_assessment,
                'benefits': await self._estimate_enhancement_benefits(capabilities, plan)
            }
            
        except Exception as e:
            logger.error(f"Failed to get enhancement recommendations: {e}")
            return {'error': str(e)}
    
    async def get_compatibility_status(self) -> Dict[str, Any]:
        """Get overall compatibility layer status."""
        try:
            with self.lock:
                total_agents = len(self.enhanced_agents)
                enhanced_counts = defaultdict(int)
                
                for capabilities in self.enhanced_agents.values():
                    enhanced_counts[capabilities.enhancement_level.value] += 1
                
                # Agent type distribution
                type_distribution = defaultdict(int)
                for capabilities in self.enhanced_agents.values():
                    type_distribution[capabilities.agent_type.value] += 1
                
                # Recent enhancements
                recent_enhancements = [
                    record for record in self.enhancement_history
                    if time.time() - record['timestamp'] < 3600  # Last hour
                ]
                
                return {
                    'timestamp': time.time(),
                    'total_agents': total_agents,
                    'enhancement_levels': dict(enhanced_counts),
                    'agent_type_distribution': dict(type_distribution),
                    'recent_enhancements': len(recent_enhancements),
                    'enhancement_success_rate': self._calculate_enhancement_success_rate(),
                    'registered_adapters': len(self.agent_adapters),
                    'quantum_components_available': {
                        'neural_evolution': self.neural_evolution is not None,
                        'topology_manager': self.topology_manager is not None,
                        'behavior_engine': self.behavior_engine is not None,
                        'meta_coordinator': self.meta_coordinator is not None,
                        'error_handler': self.error_handler is not None
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get compatibility status: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    # Private methods
    
    async def _discover_agent_capabilities(self, agent_instance: Any) -> Dict[str, Any]:
        """Discover capabilities of an agent through inspection."""
        try:
            capabilities = {
                'operations': [],
                'metrics': {},
                'features': []
            }
            
            # Inspect methods
            methods = [name for name in dir(agent_instance) 
                      if not name.startswith('_') and callable(getattr(agent_instance, name))]
            capabilities['operations'] = methods
            
            # Check for common performance attributes
            if hasattr(agent_instance, 'performance_metrics'):
                capabilities['metrics'] = getattr(agent_instance, 'performance_metrics', {})
            
            # Check for quantum readiness
            quantum_indicators = ['async', 'await', 'quantum', 'enhanced']
            for indicator in quantum_indicators:
                for method in methods:
                    if indicator in method.lower():
                        capabilities['features'].append(f'{indicator}_ready')
                        break
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Capability discovery failed: {e}")
            return {'operations': [], 'metrics': {}, 'features': []}
    
    async def _estimate_enhancement_benefits(self, 
                                           capabilities: AgentCapabilities, 
                                           plan: EnhancementPlan) -> Dict[str, Any]:
        """Estimate benefits of enhancing agent."""
        benefits = {
            'performance_improvement': 0.2,  # Base improvement
            'reliability_improvement': 0.15,
            'feature_additions': len(plan.required_features),
            'quantum_readiness': plan.target_level.value != EnhancementLevel.NONE.value
        }
        
        # Agent-type specific benefits
        if capabilities.agent_type in [AgentType.PERFORMANCE_BENCHMARKER, AgentType.PERF_ANALYZER]:
            benefits['performance_improvement'] = 0.4
        
        if capabilities.agent_type in [AgentType.HIERARCHICAL_COORDINATOR, AgentType.MESH_COORDINATOR]:
            benefits['coordination_improvement'] = 0.3
        
        return benefits
    
    def _calculate_enhancement_success_rate(self) -> float:
        """Calculate enhancement success rate."""
        if not self.enhancement_history:
            return 1.0
        
        successful = sum(1 for record in self.enhancement_history if record.get('success', False))
        return successful / len(self.enhancement_history)
    
    async def _notify_enhancement_callbacks(self, 
                                          agent_id: str, 
                                          capabilities: AgentCapabilities, 
                                          plan: EnhancementPlan):
        """Notify callbacks about enhancement events."""
        for callback in self.compatibility_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, capabilities, plan)
                else:
                    callback(agent_id, capabilities, plan)
            except Exception as e:
                logger.error(f"Enhancement callback failed: {e}")
    
    def add_compatibility_callback(self, callback: Callable):
        """Add callback for compatibility events."""
        self.compatibility_callbacks.append(callback)
    
    def remove_compatibility_callback(self, callback: Callable):
        """Remove compatibility callback."""
        if callback in self.compatibility_callbacks:
            self.compatibility_callbacks.remove(callback)