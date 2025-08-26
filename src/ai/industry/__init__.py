"""
Industry-Specific AI Intelligence Modules

This package provides specialized AI intelligence for various industries:
- Healthcare Intelligence (HIPAA compliance, medical device development)
- Financial Intelligence (banking regulations, blockchain, fintech)
- Aerospace Intelligence (safety-critical systems, certification standards)
- Automotive Intelligence (autonomous systems, safety standards)

Each module provides:
- Industry-specific code patterns and best practices
- Regulatory compliance checking
- Domain-specific testing strategies
- Specialized security requirements
- Industry standard validation
"""

__version__ = "1.0.0"

from .healthcare_intelligence import HealthcareIntelligence
from .financial_intelligence import FinancialIntelligence
from .aerospace_intelligence import AerospaceIntelligence
from .automotive_intelligence import AutomotiveIntelligence
from .industry_intelligence_coordinator import IndustryIntelligenceCoordinator

__all__ = [
    'HealthcareIntelligence',
    'FinancialIntelligence', 
    'AerospaceIntelligence',
    'AutomotiveIntelligence',
    'IndustryIntelligenceCoordinator'
]