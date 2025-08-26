"""
Comprehensive Authentication System Tests.

Tests all components of the authentication system including:
- User authentication and registration
- JWT token management
- OAuth integration
- Session management
- Security features
- Claude Code/Flow integration
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import authentication components
sys.path.append('/home/tekkadmin/claude-tui/src')
from auth.auth_service import AuthenticationService
from auth.user_service import UserService
from auth.jwt_service import JWTService
from auth.session_manager import SessionManager
from auth.oauth_manager import OAuthManager
from auth.oauth.github import GitHubOAuthProvider
from auth.oauth.google import GoogleOAuthProvider
from auth.oauth.base import OAuthUserInfo
from core.exceptions import AuthenticationError, ValidationError


class TestUserService:
    """Test user service functionality."""
    
    @pytest.fixture
    def user_service(self):
        """Create user service for testing."""
        return UserService()
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for testing."""
        user = Mock()
        user.id = uuid.uuid4()
        user.username = "testuser"
        user.email = "test@example.com"
        user.full_name = "Test User"
        user.is_active = True
        user.is_verified = False
        user.verify_password = Mock(return_value=True)
        return user
    
    @pytest.mark.asyncio
    async def test_get_user_by_id_success(self, user_service, mock_user):
        """Test successful user retrieval by ID."""
        with patch.object(user_service, '_get_repository') as mock_repo:
            mock_repository = AsyncMock()
            mock_repository.get_by_id.return_value = mock_user
            mock_repo.return_value = mock_repository
            
            result = await user_service.get_user_by_id(str(mock_user.id))
            
            assert result == mock_user
            mock_repository.get_by_id.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_by_id_invalid_uuid(self, user_service):
        """Test user retrieval with invalid UUID."""
        result = await user_service.get_user_by_id("invalid-uuid")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, user_service, mock_user):
        """Test successful user authentication."""
        with patch.object(user_service, '_get_repository') as mock_repo:
            mock_repository = AsyncMock()
            mock_repository.authenticate_user.return_value = mock_user
            mock_repo.return_value = mock_repository
            
            result = await user_service.authenticate_user(
                identifier="test@example.com",
                password="password123",
                ip_address="192.168.1.1"
            )
            
            assert result == mock_user
            mock_repository.authenticate_user.assert_called_once_with(
                identifier="test@example.com",
                password="password123",
                ip_address="192.168.1.1"
            )
    
    @pytest.mark.asyncio
    async def test_create_oauth_user(self, user_service):
        """Test OAuth user creation."""
        oauth_info = OAuthUserInfo(
            provider="github",
            provider_id="12345",
            email="oauth@example.com",
            name="OAuth User",
            username="oauthuser",
            verified=True
        )
        
        mock_user = Mock()
        mock_user.username = "oauthuser"
        mock_user.email = "oauth@example.com"
        
        with patch.object(user_service, 'get_user_by_username', return_value=None):
            with patch.object(user_service, 'create_user', return_value=mock_user):
                result = await user_service.create_oauth_user(oauth_info)
                
                assert result == mock_user


class TestJWTService:
    """Test JWT service functionality."""
    
    @pytest.fixture
    def jwt_service(self):
        """Create JWT service for testing."""
        return JWTService(
            secret_key="test-secret-key",
            access_token_expire_minutes=30,
            refresh_token_expire_days=30
        )
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for JWT testing."""
        user = Mock()
        user.id = uuid.uuid4()
        user.username = "jwtuser"
        user.email = "jwt@example.com"
        user.role = "user"
        return user
    
    @pytest.mark.asyncio
    async def test_create_token_pair(self, jwt_service, mock_user):
        """Test JWT token pair creation."""
        with patch.object(jwt_service, '_redis_hmset', return_value=True):
            with patch.object(jwt_service, '_redis_expire', return_value=True):
                token_pair = await jwt_service.create_token_pair(
                    user=mock_user,
                    session_id="test-session-id",
                    permissions=["read", "write"],
                    ip_address="192.168.1.1"
                )
                
                assert token_pair.access_token is not None
                assert token_pair.refresh_token is not None
                assert token_pair.token_type == "bearer"
                assert token_pair.user_id == str(mock_user.id)
                assert token_pair.session_id == "test-session-id"
                assert token_pair.permissions == ["read", "write"]
    
    @pytest.mark.asyncio
    async def test_validate_access_token_success(self, jwt_service, mock_user):
        """Test successful access token validation."""
        with patch.object(jwt_service, '_redis_exists', return_value=False):
            with patch.object(jwt_service, '_redis_hmset', return_value=True):
                with patch.object(jwt_service, '_redis_expire', return_value=True):
                    # Create token first
                    token_pair = await jwt_service.create_token_pair(
                        user=mock_user,
                        session_id="test-session-id",
                        permissions=["read"]
                    )
                    
                    # Validate token
                    token_data = await jwt_service.validate_access_token(
                        token_pair.access_token
                    )
                    
                    assert token_data.user_id == str(mock_user.id)
                    assert token_data.username == mock_user.username
                    assert token_data.token_type == "access"
                    assert "read" in token_data.permissions
    
    @pytest.mark.asyncio
    async def test_validate_access_token_blacklisted(self, jwt_service, mock_user):
        """Test validation of blacklisted token."""
        with patch.object(jwt_service, '_redis_exists', return_value=True):
            with pytest.raises(AuthenticationError, match="Token has been revoked"):
                await jwt_service.validate_access_token("fake-token")


class TestSessionManager:
    """Test session management functionality."""
    
    @pytest.fixture
    def session_manager(self):
        """Create session manager for testing."""
        return SessionManager(session_timeout_minutes=60)
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test session creation."""
        with patch.object(session_manager, '_redis_setex', return_value=True):
            with patch.object(session_manager, '_redis_sadd', return_value=1):
                with patch.object(session_manager, '_redis_expire', return_value=True):
                    with patch.object(session_manager, '_get_user_session_count', return_value=0):
                        with patch.object(session_manager, '_update_session_stats'):
                            session_data = await session_manager.create_session(
                                user_id="test-user-id",
                                username="testuser",
                                role="user",
                                ip_address="192.168.1.1",
                                user_agent="Mozilla/5.0"
                            )
                            
                            assert session_data.user_id == "test-user-id"
                            assert session_data.username == "testuser"
                            assert session_data.role == "user"
                            assert session_data.ip_address == "192.168.1.1"
                            assert session_data.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_session_success(self, session_manager):
        """Test successful session retrieval."""
        session_json = '''{
            "session_id": "test-session",
            "user_id": "test-user",
            "username": "testuser",
            "role": "user",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "created_at": "2024-01-01T00:00:00+00:00",
            "last_activity": "2024-01-01T00:00:00+00:00",
            "expires_at": "2030-01-01T00:00:00+00:00",
            "metadata": {},
            "is_active": true
        }'''
        
        with patch.object(session_manager, '_redis_get', return_value=session_json):
            session_data = await session_manager.get_session("test-session")
            
            assert session_data is not None
            assert session_data.session_id == "test-session"
            assert session_data.user_id == "test-user"
            assert session_data.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_session_expired(self, session_manager):
        """Test retrieval of expired session."""
        session_json = '''{
            "session_id": "test-session",
            "user_id": "test-user",
            "username": "testuser",
            "role": "user",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "created_at": "2020-01-01T00:00:00+00:00",
            "last_activity": "2020-01-01T00:00:00+00:00",
            "expires_at": "2020-01-01T01:00:00+00:00",
            "metadata": {},
            "is_active": true
        }'''
        
        with patch.object(session_manager, '_redis_get', return_value=session_json):
            with patch.object(session_manager, 'invalidate_session', return_value=True):
                session_data = await session_manager.get_session("test-session")
                
                assert session_data is None


class TestOAuthProviders:
    """Test OAuth provider implementations."""
    
    @pytest.fixture
    def github_provider(self):
        """Create GitHub OAuth provider for testing."""
        return GitHubOAuthProvider(
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/auth/callback"
        )
    
    @pytest.fixture
    def google_provider(self):
        """Create Google OAuth provider for testing."""
        return GoogleOAuthProvider(
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/auth/callback"
        )
    
    def test_github_provider_properties(self, github_provider):
        """Test GitHub provider properties."""
        assert github_provider.provider_name == "github"
        assert "github.com" in github_provider.authorization_url
        assert "github.com" in github_provider.token_url
        assert "api.github.com" in github_provider.user_info_url
        assert "user:email" in github_provider.get_default_scopes()
    
    def test_google_provider_properties(self, google_provider):
        """Test Google provider properties."""
        assert google_provider.provider_name == "google"
        assert "accounts.google.com" in google_provider.authorization_url
        assert "googleapis.com" in google_provider.token_url
        assert "googleapis.com" in google_provider.user_info_url
        assert "openid" in google_provider.get_default_scopes()
    
    def test_get_authorization_url(self, github_provider):
        """Test OAuth authorization URL generation."""
        auth_url, state = github_provider.get_authorization_url()
        
        assert "github.com/login/oauth/authorize" in auth_url
        assert "client_id=" in auth_url
        assert "redirect_uri=" in auth_url
        assert "state=" in auth_url
        assert len(state) > 10  # Should be a proper random state
    
    @pytest.mark.asyncio
    async def test_github_get_user_info_success(self, github_provider):
        """Test successful GitHub user info retrieval."""
        mock_user_data = {
            "id": 12345,
            "login": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://github.com/avatar.jpg",
            "html_url": "https://github.com/testuser"
        }
        
        mock_emails_data = [
            {"email": "test@example.com", "primary": True, "verified": True}
        ]
        
        with patch.object(github_provider.http_client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = [mock_user_data, mock_emails_data]
            mock_get.return_value = mock_response
            
            user_info = await github_provider.get_user_info("test-access-token")
            
            assert user_info.provider == "github"
            assert user_info.provider_id == "12345"
            assert user_info.email == "test@example.com"
            assert user_info.name == "Test User"
            assert user_info.username == "testuser"
            assert user_info.verified is True
    
    @pytest.mark.asyncio
    async def test_google_get_user_info_success(self, google_provider):
        """Test successful Google user info retrieval."""
        mock_user_data = {
            "id": "67890",
            "email": "test@gmail.com",
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
            "picture": "https://lh3.googleusercontent.com/avatar.jpg",
            "verified_email": True
        }
        
        with patch.object(google_provider.http_client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_user_data
            mock_get.return_value = mock_response
            
            user_info = await google_provider.get_user_info("test-access-token")
            
            assert user_info.provider == "google"
            assert user_info.provider_id == "67890"
            assert user_info.email == "test@gmail.com"
            assert user_info.name == "Test User"
            assert user_info.verified is True


class TestOAuthManager:
    """Test OAuth manager functionality."""
    
    @pytest.fixture
    def oauth_manager(self):
        """Create OAuth manager for testing."""
        jwt_service = Mock(spec=JWTService)
        session_manager = Mock(spec=SessionManager)
        
        config = {
            "github": {
                "client_id": "test-github-id",
                "client_secret": "test-github-secret",
                "redirect_uri": "http://localhost:8000/auth/callback"
            }
        }
        
        return OAuthManager(
            jwt_service=jwt_service,
            session_manager=session_manager,
            providers_config=config
        )
    
    def test_initialization(self, oauth_manager):
        """Test OAuth manager initialization."""
        assert "github" in oauth_manager.get_available_providers()
        assert oauth_manager.get_provider("github") is not None
    
    def test_generate_authorization_url(self, oauth_manager):
        """Test OAuth authorization URL generation."""
        auth_url, state = oauth_manager.generate_authorization_url(
            provider_name="github",
            redirect_after_auth="/dashboard"
        )
        
        assert "github.com" in auth_url
        assert state in oauth_manager.state_cache
        assert oauth_manager.state_cache[state]['provider'] == "github"
        assert oauth_manager.state_cache[state]['redirect_after_auth'] == "/dashboard"


class TestAuthenticationService:
    """Test comprehensive authentication service."""
    
    @pytest.fixture
    def auth_service(self):
        """Create authentication service for testing."""
        return AuthenticationService(
            secret_key="test-secret-key",
            oauth_config={
                "github": {
                    "client_id": "test-github-id",
                    "client_secret": "test-github-secret",
                    "redirect_uri": "http://localhost:8000/auth/callback"
                }
            }
        )
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for testing."""
        user = Mock()
        user.id = uuid.uuid4()
        user.username = "authuser"
        user.email = "auth@example.com"
        user.full_name = "Auth User"
        user.is_active = True
        user.is_verified = True
        return user
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service, mock_user):
        """Test successful user authentication through auth service."""
        mock_session_data = Mock()
        mock_session_data.session_id = "test-session-id"
        mock_session_data.expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
        
        mock_token_pair = Mock()
        mock_token_pair.access_token = "test-access-token"
        mock_token_pair.refresh_token = "test-refresh-token"
        mock_token_pair.token_type = "bearer"
        mock_token_pair.expires_in = 1800
        
        with patch.object(auth_service.user_service, 'authenticate_user', return_value=mock_user):
            with patch.object(auth_service.session_manager, 'create_session', return_value=mock_session_data):
                with patch.object(auth_service.user_service, 'get_user_permissions', return_value=["read", "write"]):
                    with patch.object(auth_service.jwt_service, 'create_token_pair', return_value=mock_token_pair):
                        with patch.object(auth_service.audit_logger, 'log_authentication'):
                            result = await auth_service.authenticate_user(
                                identifier="auth@example.com",
                                password="password123",
                                ip_address="192.168.1.1"
                            )
                            
                            assert result['user']['username'] == "authuser"
                            assert result['tokens']['access_token'] == "test-access-token"
                            assert result['session']['session_id'] == "test-session-id"
    
    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_credentials(self, auth_service):
        """Test authentication with invalid credentials."""
        with patch.object(auth_service.user_service, 'authenticate_user', return_value=None):
            with patch.object(auth_service.audit_logger, 'log_authentication'):
                with pytest.raises(AuthenticationError, match="Invalid credentials"):
                    await auth_service.authenticate_user(
                        identifier="invalid@example.com",
                        password="wrongpassword"
                    )
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, auth_service, mock_user):
        """Test successful user registration."""
        with patch.object(auth_service.user_service, 'create_user', return_value=mock_user):
            with patch.object(auth_service.audit_logger, 'log_authentication'):
                result = await auth_service.register_user(
                    email="newuser@example.com",
                    username="newuser",
                    password="password123",
                    full_name="New User"
                )
                
                assert result['user']['username'] == "authuser"
                assert result['message'] == 'User registered successfully'
    
    def test_get_oauth_authorization_url(self, auth_service):
        """Test OAuth authorization URL generation."""
        result = auth_service.get_oauth_authorization_url(
            provider="github",
            redirect_after_auth="/dashboard"
        )
        
        assert "authorization_url" in result
        assert "state" in result
        assert result["provider"] == "github"
        assert "github.com" in result["authorization_url"]
    
    def test_get_security_headers(self, auth_service):
        """Test security headers generation."""
        headers = auth_service.get_security_headers()
        
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert headers["X-Frame-Options"] == "DENY"
        assert "nosniff" in headers["X-Content-Type-Options"]


class TestClaudeIntegration:
    """Test Claude Code and Claude Flow integration."""
    
    @pytest.mark.asyncio
    async def test_claude_code_auth_integration(self):
        """Test Claude Code authentication integration."""
        # Mock Claude Code authentication request
        auth_service = AuthenticationService(secret_key="claude-secret")
        
        mock_user = Mock()
        mock_user.id = uuid.uuid4()
        mock_user.username = "claude_user"
        mock_user.email = "claude@anthropic.com"
        mock_user.is_active = True
        
        with patch.object(auth_service.user_service, 'authenticate_user', return_value=mock_user):
            with patch.object(auth_service.session_manager, 'create_session'):
                with patch.object(auth_service.jwt_service, 'create_token_pair'):
                    with patch.object(auth_service.audit_logger, 'log_authentication'):
                        with patch.object(auth_service.user_service, 'get_user_permissions', return_value=["claude:code"]):
                            result = await auth_service.authenticate_user(
                                identifier="claude@anthropic.com",
                                password="claude-password",
                                ip_address="127.0.0.1",
                                user_agent="Claude-Code/1.0"
                            )
                            
                            assert "user" in result
                            assert "tokens" in result
                            assert result['user']['username'] == "claude_user"
    
    @pytest.mark.asyncio
    async def test_claude_flow_oauth_integration(self):
        """Test Claude Flow OAuth integration."""
        # Mock Claude Flow OAuth callback
        auth_service = AuthenticationService(
            secret_key="flow-secret",
            oauth_config={
                "github": {
                    "client_id": "flow-github-id",
                    "client_secret": "flow-github-secret",
                    "redirect_uri": "http://localhost:3000/auth/callback"
                }
            }
        )
        
        # Generate state first
        oauth_result = auth_service.get_oauth_authorization_url("github")
        state = oauth_result["state"]
        
        # Mock OAuth callback
        mock_oauth_result = {
            'user': {
                'id': str(uuid.uuid4()),
                'username': 'flow_user',
                'email': 'flow@anthropic.com',
                'permissions': ['claude:flow']
            },
            'tokens': {
                'access_token': 'flow-access-token',
                'refresh_token': 'flow-refresh-token'
            },
            'session': {
                'session_id': 'flow-session-id'
            }
        }
        
        with patch.object(auth_service.oauth_manager, 'handle_callback', return_value=mock_oauth_result):
            with patch.object(auth_service.audit_logger, 'log_authentication'):
                result = await auth_service.handle_oauth_callback(
                    provider="github",
                    code="oauth-code",
                    state=state,
                    ip_address="127.0.0.1",
                    user_agent="Claude-Flow/2.0"
                )
                
                assert result['user']['username'] == 'flow_user'
                assert 'claude:flow' in result['user']['permissions']


class TestSecurityFeatures:
    """Test security features and best practices."""
    
    @pytest.fixture
    def auth_service(self):
        """Create auth service for security testing."""
        return AuthenticationService(secret_key="security-test-key")
    
    @pytest.mark.asyncio
    async def test_token_blacklisting(self, auth_service):
        """Test JWT token blacklisting functionality."""
        mock_token_data = Mock()
        mock_token_data.user_id = "test-user-id"
        mock_token_data.username = "testuser"
        mock_token_data.session_id = "test-session-id"
        
        with patch.object(auth_service.jwt_service, 'validate_access_token', return_value=mock_token_data):
            with patch.object(auth_service.jwt_service, 'revoke_token', return_value=True):
                with patch.object(auth_service.session_manager, 'invalidate_session', return_value=True):
                    with patch.object(auth_service.audit_logger, 'log_authentication'):
                        result = await auth_service.logout(
                            token="test-token",
                            session_id="test-session-id"
                        )
                        
                        assert result is True
                        auth_service.jwt_service.revoke_token.assert_called_once_with("test-token")
    
    @pytest.mark.asyncio
    async def test_session_timeout_handling(self, auth_service):
        """Test session timeout and cleanup."""
        with patch.object(auth_service.jwt_service, 'cleanup_expired_tokens', return_value=5):
            with patch.object(auth_service.session_manager, 'cleanup_expired_sessions', return_value=3):
                result = await auth_service.cleanup_expired_tokens_and_sessions()
                
                assert result['tokens_cleaned'] == 5
                assert result['sessions_cleaned'] == 3
    
    def test_security_headers_comprehensive(self, auth_service):
        """Test comprehensive security headers."""
        headers = auth_service.get_security_headers()
        
        # Check all security headers are present
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy',
            'Referrer-Policy',
            'Permissions-Policy'
        ]
        
        for header in required_headers:
            assert header in headers
        
        # Check specific security values
        assert headers['X-Frame-Options'] == 'DENY'
        assert 'nosniff' in headers['X-Content-Type-Options']
        assert 'max-age=31536000' in headers['Strict-Transport-Security']
        assert "default-src 'self'" in headers['Content-Security-Policy']
    
    @pytest.mark.asyncio
    async def test_rate_limiting_simulation(self, auth_service):
        """Test rate limiting behavior simulation."""
        # Simulate multiple failed login attempts
        failed_attempts = 0
        
        for i in range(6):  # Exceed typical rate limit
            try:
                with patch.object(auth_service.user_service, 'authenticate_user', return_value=None):
                    with patch.object(auth_service.audit_logger, 'log_authentication'):
                        await auth_service.authenticate_user(
                            identifier="attacker@evil.com",
                            password="wrongpassword",
                            ip_address="192.168.1.100"
                        )
            except AuthenticationError:
                failed_attempts += 1
        
        assert failed_attempts == 6  # All attempts should fail


if __name__ == "__main__":
    # Run specific test categories
import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "user":
            pytest.main(["-v", "TestUserService"])
        elif test_category == "jwt":
            pytest.main(["-v", "TestJWTService"])
        elif test_category == "oauth":
            pytest.main(["-v", "TestOAuthProviders", "TestOAuthManager"])
        elif test_category == "auth":
            pytest.main(["-v", "TestAuthenticationService"])
        elif test_category == "claude":
            pytest.main(["-v", "TestClaudeIntegration"])
        elif test_category == "security":
            pytest.main(["-v", "TestSecurityFeatures"])
        else:
            print("Available test categories: user, jwt, oauth, auth, claude, security")
    else:
        # Run all tests
        pytest.main(["-v", __file__])
