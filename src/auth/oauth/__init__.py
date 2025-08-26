"""
OAuth Integration Package.

Provides OAuth 2.0 integration for GitHub and Google authentication.
"""

from .github import GitHubOAuthProvider
from .google import GoogleOAuthProvider
from .base import OAuthProvider, OAuthUserInfo, OAuthError

__all__ = [
    'GitHubOAuthProvider',
    'GoogleOAuthProvider', 
    'OAuthProvider',
    'OAuthUserInfo',
    'OAuthError'
]