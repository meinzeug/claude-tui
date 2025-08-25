"""
GitHub OAuth Provider.

Implements OAuth 2.0 authentication with GitHub.
"""

import httpx
from typing import List

from .base import OAuthProvider, OAuthUserInfo, OAuthError


class GitHubOAuthProvider(OAuthProvider):
    """
    GitHub OAuth 2.0 provider implementation.
    
    Provides authentication via GitHub OAuth API.
    """
    
    @property
    def provider_name(self) -> str:
        return "github"
    
    @property
    def authorization_url(self) -> str:
        return "https://github.com/login/oauth/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://github.com/login/oauth/access_token"
    
    @property
    def user_info_url(self) -> str:
        return "https://api.github.com/user"
    
    def get_default_scopes(self) -> List[str]:
        """Get default GitHub scopes."""
        return ["user:email", "read:user"]
    
    def get_additional_auth_params(self) -> dict:
        """Get additional GitHub-specific auth parameters."""
        return {
            'allow_signup': 'true'
        }
    
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from GitHub API.
        
        Args:
            access_token: GitHub access token
            
        Returns:
            OAuthUserInfo: Standardized user information
            
        Raises:
            OAuthError: If API request fails
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'claude-tiu-app'
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
            
            # Get user emails (primary email)
            email_response = await self.http_client.get(
                "https://api.github.com/user/emails",
                headers=headers
            )
            
            primary_email = user_data.get('email')
            email_verified = False
            
            if email_response.status_code == 200:
                emails = email_response.json()
                # Find primary email
                for email_info in emails:
                    if email_info.get('primary', False):
                        primary_email = email_info.get('email')
                        email_verified = email_info.get('verified', False)
                        break
            
            # Handle missing email
            if not primary_email:
                # Try to get public email
                primary_email = user_data.get('email')
                if not primary_email:
                    raise OAuthError(
                        "No email available from GitHub. Please make your email public or grant email permissions.",
                        self.provider_name,
                        "no_email"
                    )
            
            return OAuthUserInfo(
                provider=self.provider_name,
                provider_id=str(user_data['id']),
                email=primary_email,
                name=user_data.get('name') or user_data.get('login', ''),
                username=user_data.get('login'),
                avatar_url=user_data.get('avatar_url'),
                profile_url=user_data.get('html_url'),
                verified=email_verified,
                raw_data=user_data
            )
            
        except httpx.HTTPError as e:
            raise OAuthError(
                f"HTTP error getting GitHub user info: {str(e)}",
                self.provider_name
            )
        except KeyError as e:
            raise OAuthError(
                f"Missing required field in GitHub response: {str(e)}",
                self.provider_name,
                "invalid_response"
            )
        except Exception as e:
            raise OAuthError(
                f"Failed to get GitHub user info: {str(e)}",
                self.provider_name
            )
    
    async def get_user_organizations(self, access_token: str) -> List[dict]:
        """
        Get user's GitHub organizations.
        
        Args:
            access_token: GitHub access token
            
        Returns:
            List of organization data
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'claude-tiu-app'
            }
            
            response = await self.http_client.get(
                "https://api.github.com/user/orgs",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception:
            return []
    
    async def get_user_repositories(
        self,
        access_token: str,
        type_filter: str = "owner",
        sort: str = "updated",
        per_page: int = 30
    ) -> List[dict]:
        """
        Get user's GitHub repositories.
        
        Args:
            access_token: GitHub access token
            type_filter: Repository type (owner, collaborator, organization_member)
            sort: Sort order (created, updated, pushed, full_name)
            per_page: Number of repositories per page
            
        Returns:
            List of repository data
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'claude-tiu-app'
            }
            
            params = {
                'type': type_filter,
                'sort': sort,
                'per_page': per_page
            }
            
            response = await self.http_client.get(
                "https://api.github.com/user/repos",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception:
            return []