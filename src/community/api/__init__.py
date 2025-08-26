"""
Community API - REST API endpoints for template marketplace and community features.
"""

from .marketplace_routes import marketplace_router
from .template_routes import template_router
# Commented out missing route imports to fix production deployment
# from .review_routes import review_router
# from .user_routes import user_router
# from .validation_routes import validation_router

__all__ = [
    "marketplace_router",
    "template_router",
    # "review_router",
    # "user_router",
    # "validation_router"
]