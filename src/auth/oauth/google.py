"""
Google OAuth Provider.

Implements OAuth 2.0 authentication with Google.
"""

import httpx
from typing import List

from .base import OAuthProvider, OAuthUserInfo, OAuthError


class GoogleOAuthProvider(OAuthProvider):
    """
    Google OAuth 2.0 provider implementation.
    
    Provides authentication via Google OAuth API.
    """
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    @property
    def authorization_url(self) -> str:
        return "https://accounts.google.com/o/oauth2/v2/auth"
    
    @property
    def token_url(self) -> str:
        return "https://oauth2.googleapis.com/token"
    
    @property
    def user_info_url(self) -> str:
        return "https://www.googleapis.com/oauth2/v2/userinfo"
    
    def get_default_scopes(self) -> List[str]:
        """Get default Google scopes."""
        return [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid"
        ]
    
    def get_additional_auth_params(self) -> dict:
        """Get additional Google-specific auth parameters."""
        return {
            'access_type': 'offline',  # Get refresh token
            'prompt': 'consent',       # Force consent screen
            'include_granted_scopes': 'true'
        }
    
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from Google API.
        
        Args:
            access_token: Google access token
            
        Returns:
            OAuthUserInfo: Standardized user information
            
        Raises:
            OAuthError: If API request fails
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            # Get user profile
            response = await self.http_client.get(
                self.user_info_url,
                headers=headers
            )
            
            if response.status_code != 200:
                raise OAuthError(
                    f"Failed to get user info: {response.text}",
                    self.provider_name,
                    str(response.status_code)
                )
            
            user_data = response.json()
            
            # Validate required fields
            if 'id' not in user_data:
                raise OAuthError(
                    "Invalid response from Google: missing user ID",
                    self.provider_name,
                    "invalid_response"
                )
            
            if 'email' not in user_data:
                raise OAuthError(
                    "No email available from Google. Please grant email permissions.",
                    self.provider_name,
                    "no_email"
                )
            
            # Extract user information
            name = user_data.get('name', '')
            if not name:
                # Fallback to given_name + family_name
                given_name = user_data.get('given_name', '')
                family_name = user_data.get('family_name', '')
                name = f"{given_name} {family_name}".strip()
            
            return OAuthUserInfo(
                provider=self.provider_name,
                provider_id=str(user_data['id']),
                email=user_data['email'],
                name=name,
                username=None,  # Google doesn't provide usernames
                avatar_url=user_data.get('picture'),
                profile_url=None,  # Google doesn't provide profile URLs in userinfo
                verified=user_data.get('verified_email', False),
                raw_data=user_data
            )
            
        except httpx.HTTPError as e:
            raise OAuthError(
                f"HTTP error getting Google user info: {str(e)}",
                self.provider_name
            )
        except KeyError as e:
            raise OAuthError(
                f"Missing required field in Google response: {str(e)}",
                self.provider_name,
                "invalid_response"
            )
        except Exception as e:
            raise OAuthError(
                f"Failed to get Google user info: {str(e)}",
                self.provider_name
            )
    
    async def get_user_profile_v3(self, access_token: str) -> dict:
        """
        Get extended user profile using Google People API v1.
        
        Args:
            access_token: Google access token
            
        Returns:
            Extended user profile data
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            # Use People API for more detailed profile
            response = await self.http_client.get(
                "https://people.googleapis.com/v1/people/me?personFields=names,emailAddresses,photos,urls",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception:
            return {}
    
    async def revoke_token(self, access_token: str) -> bool:
        """
        Revoke Google access token.
        
        Args:
            access_token: Google access token to revoke
            
        Returns:
            True if revoked successfully
        """
        try:
            response = await self.http_client.post(
                f"https://oauth2.googleapis.com/revoke?token={access_token}"
            )
            
            return response.status_code == 200
            
        except Exception:
            return False
    
    async def refresh_access_token(self, refresh_token: str) -> dict:
        """
        Refresh Google access token.
        
        Args:
            refresh_token: Google refresh token
            
        Returns:
            New token data
            
        Raises:
            OAuthError: If refresh fails
        """
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token'
            }
            
            response = await self.http_client.post(
                self.token_url,
                data=data
            )
            
            if response.status_code != 200:
                raise OAuthError(
                    f"Token refresh failed: {response.text}",
                    self.provider_name,
                    str(response.status_code)
                )
            
            token_data = response.json()
            
            if 'error' in token_data:
                raise OAuthError(
                    f"Token refresh error: {token_data.get('error_description', token_data['error'])}",
                    self.provider_name,
                    token_data['error']
                )
            
            return token_data
            
        except httpx.HTTPError as e:
            raise OAuthError(
                f"HTTP error during token refresh: {str(e)}",
                self.provider_name
            )
        except Exception as e:
            raise OAuthError(
                f"Token refresh failed: {str(e)}",
                self.provider_name
            )