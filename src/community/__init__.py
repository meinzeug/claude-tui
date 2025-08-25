"""
Community Platform for Claude-TIU Template Marketplace.

This module provides comprehensive community features including:
- Template marketplace with discovery and search
- User profiles and contribution tracking
- Rating and review system
- Template validation and quality scoring
- Version management and notifications
"""

from .models import *
from .services import *
from .api import *

__version__ = "1.0.0"
__all__ = [
    "Template",
    "TemplateMarketplace",
    "CommunityService",
    "TemplateService",
    "ReviewService",
    "UserProfileService"
]