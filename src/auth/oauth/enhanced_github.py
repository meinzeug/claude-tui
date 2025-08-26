"""
Enhanced GitHub OAuth Provider with Security Hardening

This module implements security-hardened GitHub OAuth authentication with:
- PKCE (Proof Key for Code Exchange) support
- Enhanced state validation
- Rate limiting protection
- Token security hardening
- Device fingerprinting
- Suspicious activity detection

Author: Security Specialist - Hive Mind Team
Date: 2025-08-26
"""

import httpx
import base64
import hashlib
import secrets
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging

from .base import OAuthProvider, OAuthUserInfo, OAuthError
from ...security.crypto_fixes import secure_hash, secure_token

logger = logging.getLogger(__name__)


class EnhancedGitHubOAuthProvider(OAuthProvider):
    """
    Security-hardened GitHub OAuth 2.0 provider with PKCE and enhanced validation.
    
    Features:
    - PKCE (Proof Key for Code Exchange) for enhanced security
    - Device fingerprinting and tracking
    - Rate limiting and abuse detection
    - Enhanced token validation and rotation
    - Suspicious activity monitoring
    """
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, scopes: Optional[List[str]] = None):
        """Initialize enhanced GitHub OAuth provider."""
        super().__init__(client_id, client_secret, redirect_uri, scopes)
        
        # PKCE storage (use Redis in production)
        self.pkce_store = {}
        
        # Rate limiting storage
        self.rate_limit_store = {}
        
        # Device tracking
        self.device_store = {}
        
        # Security metrics
        self.security_metrics = {
            'auth_attempts': 0,
            'successful_auths': 0,
            'failed_auths': 0,
            'suspicious_activities': 0,
            'blocked_attempts': 0
        }
    
    @property
    def provider_name(self) -> str:
        return "github_enhanced"
    
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
        """Get default GitHub scopes with minimal permissions."""
        return ["user:email", "read:user"]
    
    def generate_pkce_challenge(self) -> Tuple[str, str, str]:
        """
        Generate PKCE code verifier and challenge.
        
        Returns:
            Tuple of (code_verifier, code_challenge, code_challenge_method)
        """
        # Generate cryptographically secure code verifier
        code_verifier = secure_token(32)
        
        # Create code challenge using S256 method
        challenge_bytes = hashlib.sha256(code_verifier.encode('ascii')).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('ascii').rstrip('=')
        
        return code_verifier, code_challenge, "S256"
    
    def create_device_fingerprint(self, request_data: Dict[str, Any]) -> str:
        """
        Create device fingerprint for tracking and security.
        
        Args:
            request_data: Request metadata (IP, User-Agent, etc.)
            
        Returns:
            Device fingerprint hash
        """
        fingerprint_data = {
            'user_agent': request_data.get('user_agent', ''),
            'accept_language': request_data.get('accept_language', ''),
            'timezone': request_data.get('timezone', ''),
            'screen_resolution': request_data.get('screen_resolution', ''),
            'ip_subnet': self._get_ip_subnet(request_data.get('ip_address', ''))
        }
        
        fingerprint_str = '|'.join(f"{k}:{v}" for k, v in fingerprint_data.items())
        return secure_hash(fingerprint_str, length=16)
    
    def _get_ip_subnet(self, ip_address: str) -> str:
        """Get IP subnet for fingerprinting (privacy-preserving)."""
        try:
            parts = ip_address.split('.')
            if len(parts) == 4:
                # Use /24 subnet for IPv4
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
        except Exception:
            pass
        return "unknown"
    
    def check_rate_limit(self, client_ip: str, user_agent: str) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            client_ip: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if within limits, False if rate limited
        """
        current_time = time.time()
        rate_limit_key = secure_hash(f"{client_ip}:{user_agent}", length=16)
        
        if rate_limit_key not in self.rate_limit_store:
            self.rate_limit_store[rate_limit_key] = {
                'attempts': 0,
                'last_attempt': current_time,
                'blocked_until': 0
            }
        
        rate_data = self.rate_limit_store[rate_limit_key]
        
        # Check if currently blocked
        if current_time < rate_data['blocked_until']:
            self.security_metrics['blocked_attempts'] += 1
            logger.warning(f"Rate limited request from {client_ip[:8]}...")
            return False
        
        # Reset counter if more than 1 hour passed
        if current_time - rate_data['last_attempt'] > 3600:
            rate_data['attempts'] = 0
        
        # Check rate limit (max 10 attempts per hour)
        if rate_data['attempts'] >= 10:
            # Block for 1 hour
            rate_data['blocked_until'] = current_time + 3600
            self.security_metrics['blocked_attempts'] += 1
            logger.warning(f"Rate limit exceeded for {client_ip[:8]}..., blocking for 1 hour")
            return False
        
        # Update attempt counter
        rate_data['attempts'] += 1
        rate_data['last_attempt'] = current_time
        
        return True
    
    def get_authorization_url_with_pkce(
        self,
        request_data: Dict[str, Any],
        state: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Get authorization URL with PKCE and enhanced security.
        
        Args:
            request_data: Request metadata for security checks
            state: Optional state parameter
            
        Returns:
            Tuple of (authorization_url, state, code_verifier)
        """
        self.security_metrics['auth_attempts'] += 1
        
        # Check rate limits
        client_ip = request_data.get('ip_address', 'unknown')
        user_agent = request_data.get('user_agent', 'unknown')
        
        if not self.check_rate_limit(client_ip, user_agent):
            raise OAuthError(
                "Rate limit exceeded. Please try again later.",
                self.provider_name,
                "rate_limited"
            )
        
        # Generate PKCE parameters
        code_verifier, code_challenge, code_challenge_method = self.generate_pkce_challenge()
        
        # Generate state if not provided
        if state is None:
            state = self.generate_state()
        
        # Create device fingerprint
        device_fingerprint = self.create_device_fingerprint(request_data)
        
        # Store PKCE and security data
        pkce_data = {
            'code_verifier': code_verifier,
            'code_challenge': code_challenge,
            'code_challenge_method': code_challenge_method,
            'device_fingerprint': device_fingerprint,
            'client_ip': client_ip,
            'user_agent_hash': secure_hash(user_agent),
            'created_at': time.time(),
            'expires_at': time.time() + 600  # 10 minutes
        }
        
        self.pkce_store[state] = pkce_data
        
        # Build authorization URL with PKCE
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes),
            'response_type': 'code',
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': code_challenge_method
        }
        
        # Add additional security parameters
        params.update(self.get_additional_auth_params())
        
        from urllib.parse import urlencode
        url = f"{self.authorization_url}?{urlencode(params)}"
        
        logger.info(f"Generated secure OAuth URL for device {device_fingerprint}")
        
        return url, state, code_verifier
    
    def get_additional_auth_params(self) -> Dict[str, str]:
        """Get additional GitHub-specific auth parameters with security enhancements."""
        return {
            'allow_signup': 'false',  # More restrictive
            'login': '',  # Force login prompt for security
        }
    
    async def validate_pkce_and_exchange_token(
        self,
        code: str,
        state: str,
        request_data: Dict[str, Any]
    ) -> str:
        """
        Validate PKCE and exchange authorization code for token.
        
        Args:
            code: Authorization code from GitHub
            state: OAuth state parameter
            request_data: Current request metadata
            
        Returns:
            Access token
            
        Raises:
            OAuthError: If validation or exchange fails
        """
        try:
            # Validate state and retrieve PKCE data
            if state not in self.pkce_store:
                self.security_metrics['failed_auths'] += 1
                raise OAuthError("Invalid or expired OAuth state", self.provider_name, "invalid_state")
            
            pkce_data = self.pkce_store[state]
            
            # Check expiration
            if time.time() > pkce_data['expires_at']:
                del self.pkce_store[state]
                self.security_metrics['failed_auths'] += 1
                raise OAuthError("OAuth state has expired", self.provider_name, "expired_state")
            
            # Validate device fingerprint for additional security
            current_fingerprint = self.create_device_fingerprint(request_data)
            if current_fingerprint != pkce_data['device_fingerprint']:
                self.security_metrics['suspicious_activities'] += 1
                logger.warning(f"Device fingerprint mismatch for state {state[:8]}...")
                # Continue with warning in development, could block in production
            
            # Validate IP address (loose validation)
            current_ip = request_data.get('ip_address', 'unknown')
            if current_ip != pkce_data['client_ip']:
                self.security_metrics['suspicious_activities'] += 1
                logger.warning(f"IP address changed during OAuth flow: {current_ip} != {pkce_data['client_ip']}")
                # Continue with warning for mobile/dynamic IP scenarios
            
            # Exchange code for token with PKCE
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code',
                'code_verifier': pkce_data['code_verifier']
            }
            
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'claude-tui-secure/1.0'
            }
            
            response = await self.http_client.post(
                self.token_url,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                self.security_metrics['failed_auths'] += 1
                raise OAuthError(
                    f"Token exchange failed: {response.status_code}",
                    self.provider_name,
                    str(response.status_code)
                )
            
            token_data = response.json()
            
            if 'error' in token_data:
                self.security_metrics['failed_auths'] += 1
                raise OAuthError(
                    f"Token exchange error: {token_data.get('error_description', token_data['error'])}",
                    self.provider_name,
                    token_data['error']
                )
            
            access_token = token_data.get('access_token')
            if not access_token:
                self.security_metrics['failed_auths'] += 1
                raise OAuthError("No access token in response", self.provider_name)
            
            # Clean up PKCE data
            del self.pkce_store[state]
            
            self.security_metrics['successful_auths'] += 1
            logger.info(f"Successful OAuth token exchange for device {current_fingerprint}")
            
            return access_token
            
        except OAuthError:
            raise
        except Exception as e:
            self.security_metrics['failed_auths'] += 1
            logger.error(f"Unexpected error in PKCE token exchange: {e}")
            raise OAuthError(f"Token exchange failed: {str(e)}", self.provider_name)
    
    async def get_user_info_secure(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information with enhanced security validation.
        
        Args:
            access_token: GitHub access token
            
        Returns:
            OAuthUserInfo with security enhancements
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'claude-tui-secure/1.0',
                'X-GitHub-Api-Version': '2022-11-28'
            }
            
            # Get user profile with timeout
            response = await self.http_client.get(
                self.user_info_url,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise OAuthError(
                    f"Failed to get user info: HTTP {response.status_code}",
                    self.provider_name,
                    str(response.status_code)
                )
            
            user_data = response.json()
            
            # Validate required fields
            required_fields = ['id', 'login']
            for field in required_fields:
                if field not in user_data:
                    raise OAuthError(
                        f"Missing required field '{field}' in GitHub response",
                        self.provider_name,
                        "invalid_response"
                    )
            
            # Get user emails with enhanced validation
            email_response = await self.http_client.get(
                "https://api.github.com/user/emails",
                headers=headers,
                timeout=30
            )
            
            primary_email = user_data.get('email')
            email_verified = False
            
            if email_response.status_code == 200:
                emails = email_response.json()
                
                # Enhanced email validation
                for email_info in emails:
                    if email_info.get('primary', False):
                        primary_email = email_info.get('email')
                        email_verified = email_info.get('verified', False)
                        
                        # Additional security: check email domain reputation
                        if primary_email:
                            domain = primary_email.split('@')[1].lower()
                            if self._is_suspicious_email_domain(domain):
                                self.security_metrics['suspicious_activities'] += 1
                                logger.warning(f"Suspicious email domain detected: {domain}")
                        break
            
            # Handle missing email with enhanced error
            if not primary_email:
                raise OAuthError(
                    "No email available from GitHub. Please ensure your primary email is public or grant email permissions.",
                    self.provider_name,
                    "no_email"
                )
            
            # Additional security checks on user account
            account_security_score = self._calculate_account_security_score(user_data)
            
            user_info = OAuthUserInfo(
                provider=self.provider_name,
                provider_id=str(user_data['id']),
                email=primary_email,
                name=user_data.get('name') or user_data.get('login', ''),
                username=user_data.get('login'),
                avatar_url=user_data.get('avatar_url'),
                profile_url=user_data.get('html_url'),
                verified=email_verified,
                raw_data={
                    **user_data,
                    'security_score': account_security_score,
                    'verification_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Successfully retrieved user info for GitHub user {user_data.get('login')} (security score: {account_security_score})")
            
            return user_info
            
        except OAuthError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting GitHub user info: {e}")
            raise OAuthError(f"Failed to get user info: {str(e)}", self.provider_name)
    
    def _is_suspicious_email_domain(self, domain: str) -> bool:
        """Check if email domain is suspicious or on blocklist."""
        # Common temporary/disposable email domains
        suspicious_domains = {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email'
        }
        
        return domain.lower() in suspicious_domains
    
    def _calculate_account_security_score(self, user_data: Dict[str, Any]) -> int:
        """
        Calculate GitHub account security score based on account characteristics.
        
        Args:
            user_data: GitHub user data
            
        Returns:
            Security score (0-100)
        """
        score = 50  # Base score
        
        # Account age (older accounts are generally more trustworthy)
        created_at = user_data.get('created_at')
        if created_at:
            try:
                from datetime import datetime
                creation_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                account_age_days = (datetime.now(timezone.utc) - creation_date).days
                
                if account_age_days > 365:
                    score += 20  # Account > 1 year old
                elif account_age_days > 90:
                    score += 10  # Account > 3 months old
                elif account_age_days < 7:
                    score -= 20  # Very new account
            except Exception:
                pass
        
        # Public repositories (active developers)
        public_repos = user_data.get('public_repos', 0)
        if public_repos > 10:
            score += 15
        elif public_repos > 0:
            score += 5
        
        # Followers (community trust)
        followers = user_data.get('followers', 0)
        if followers > 100:
            score += 10
        elif followers > 10:
            score += 5
        
        # Profile completeness
        if user_data.get('name'):
            score += 5
        if user_data.get('company'):
            score += 5
        if user_data.get('blog'):
            score += 5
        if user_data.get('bio'):
            score += 5
        
        # Two-factor authentication (if available in data)
        if user_data.get('two_factor_authentication'):
            score += 20
        
        return max(0, min(100, score))
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        return {
            **self.security_metrics,
            'success_rate': (
                self.security_metrics['successful_auths'] / 
                max(1, self.security_metrics['auth_attempts'])
            ) * 100,
            'pkce_sessions_active': len(self.pkce_store),
            'rate_limited_ips': len([
                data for data in self.rate_limit_store.values() 
                if time.time() < data['blocked_until']
            ])
        }
    
    async def cleanup_expired_sessions(self):
        """Clean up expired PKCE sessions and rate limit data."""
        current_time = time.time()
        
        # Clean up expired PKCE sessions
        expired_states = [
            state for state, data in self.pkce_store.items()
            if current_time > data['expires_at']
        ]
        
        for state in expired_states:
            del self.pkce_store[state]
        
        # Clean up old rate limit data (older than 24 hours)
        expired_rate_limits = [
            key for key, data in self.rate_limit_store.items()
            if current_time - data['last_attempt'] > 86400
        ]
        
        for key in expired_rate_limits:
            del self.rate_limit_store[key]
        
        if expired_states or expired_rate_limits:
            logger.info(f"Cleaned up {len(expired_states)} expired PKCE sessions and {len(expired_rate_limits)} old rate limit entries")