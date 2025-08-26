"""
AI Learning and Personalization Module for claude-tiu.

This package provides advanced AI learning capabilities including:
- Personalized AI behavior based on user patterns
- Federated learning infrastructure for cross-team collaboration
- Advanced pattern recognition and analysis
- Privacy-preserving knowledge sharing
- Adaptive AI behavior modification
"""

from .pattern_engine import PatternRecognitionEngine
from .personalization import PersonalizedAIBehavior
from .federated import FederatedLearningSystem
from .analytics import LearningAnalytics
from .privacy import PrivacyPreservingLearning

__all__ = [
    "PatternRecognitionEngine",
    "PersonalizedAIBehavior", 
    "FederatedLearningSystem",
    "LearningAnalytics",
    "PrivacyPreservingLearning"
]

__version__ = "1.0.0"