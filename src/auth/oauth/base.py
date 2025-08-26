"""
Base OAuth Provider Classes.

Defines the interface and common functionality for OAuth providers.
"""

import secrets
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode
import httpx
from pydantic import BaseModel, Field

from ...core.exceptions import AuthenticationError


class OAuthError(AuthenticationError):
    """OAuth-specific authentication error."""
    
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code


class OAuthUserInfo(BaseModel):
    """Standard OAuth user information."""
    provider: str = Field(..., description="OAuth provider name")
    provider_id: str = Field(..., description="User ID from provider")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User full name")
    username: Optional[str] = Field(None, description="Username")
    avatar_url: Optional[str] = Field(None, description="Avatar URL")
    profile_url: Optional[str] = Field(None, description="Profile URL")
    verified: bool = Field(default=False, description="Email verified status")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw provider data")


class OAuthProvider(ABC):
    """
    Base OAuth provider interface.
    
    Defines the contract for OAuth authentication providers.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: Optional[list] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes or self.get_default_scopes()
        self.http_client = httpx.AsyncClient()
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name."""
        pass
    
    @property
    @abstractmethod
    def authorization_url(self) -> str:
        """Get authorization endpoint URL."""
        pass
    
    @property
    @abstractmethod
    def token_url(self) -> str:
        """Get token endpoint URL."""
        pass
    
    @property
    @abstractmethod
    def user_info_url(self) -> str:
        """Get user info endpoint URL."""
        pass
    
    @abstractmethod
    def get_default_scopes(self) -> list:
        """Get default scopes for this provider."""
        pass
    
    @abstractmethod
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user information from provider."""
        pass
    
    def generate_state(self) -> str:
        """Generate CSRF state parameter."""
        return secrets.token_urlsafe(32)
    
    def get_authorization_url(self, state: Optional[str] = None) -> Tuple[str, str]:
        """
        Get authorization URL for OAuth flow.
        
        Args:
            state: CSRF state parameter (generated if not provided)
            
        Returns:
            Tuple of (authorization_url, state)
        """
        if state is None:
            state = self.generate_state()
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes),
            'response_type': 'code',
            'state': state
        }
        
        # Add provider-specific parameters
        params.update(self.get_additional_auth_params())
        
        url = f"{self.authorization_url}?{urlencode(params)}"
        return url, state
    
    def get_additional_auth_params(self) -> Dict[str, str]:
        """Get additional authorization parameters for this provider."""
        return {}
    
    async def exchange_code_for_token(self, code: str, state: str) -> str:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code
            state: CSRF state parameter
            
        Returns:
            Access token
            
        Raises:
            OAuthError: If token exchange fails
        """
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            headers = {'Accept': 'application/json'}
            
            response = await self.http_client.post(
                self.token_url,
                data=data,
                headers=headers
            )
            
            if response.status_code != 200:
                raise OAuthError(
                    f"Token exchange failed: {response.text}",
                    self.provider_name,
                    str(response.status_code)
                )
            
            token_data = response.json()
            
            if 'error' in token_data:
                raise OAuthError(
                    f"Token exchange error: {token_data.get('error_description', token_data['error'])}",
                    self.provider_name,
                    token_data['error']
                )
            
            access_token = token_data.get('access_token')
            if not access_token:
                raise OAuthError(
                    "No access token in response",
                    self.provider_name
                )
            
            return access_token
            
        except httpx.HTTPError as e:
            raise OAuthError(
                f"HTTP error during token exchange: {str(e)}",
                self.provider_name
            )
        except Exception as e:
            raise OAuthError(
                f"Token exchange failed: {str(e)}",
                self.provider_name
            )
    
    async def authenticate(self, code: str, state: str) -> OAuthUserInfo:
        """
        Complete OAuth authentication flow.
        
        Args:
            code: Authorization code
            state: CSRF state parameter
            
        Returns:
            OAuthUserInfo: User information from provider
        """
        # Exchange code for token
        access_token = await self.exchange_code_for_token(code, state)
        
        # Get user information
        user_info = await self.get_user_info(access_token)
        
        return user_info
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()