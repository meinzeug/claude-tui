"""
Consciousness-Level AI Reasoning System

Advanced AI capabilities that go beyond traditional machine learning:
- Causal Inference Engine: Understanding cause-effect relationships
- Abstract Reasoning: High-level conceptual processing
- Strategic Decision Making: Executive-level planning
- Context-Aware Reasoning: Situational awareness
"""

from .consciousness_coordinator import ConsciousnessCoordinator
from .engines.causal_inference_engine import CausalInferenceEngine
from .engines.abstract_reasoning_module import AbstractReasoningModule
from .engines.strategic_decision_maker import StrategicDecisionMaker

__version__ = "1.0.0"
__all__ = [
    "ConsciousnessCoordinator",
    "CausalInferenceEngine", 
    "AbstractReasoningModule",
    "StrategicDecisionMaker"
]