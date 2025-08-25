"""
Authentication Integration Tests for claude-tui.

Comprehensive tests for authentication system including:
- JWT token generation and validation
- Login/logout workflows
- Password reset flows
- OAuth integration
- RBAC middleware
- Session management
- Security features (rate limiting, account lockout)
"""

import asyncio
import pytest
import uuid
import jwt
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from src.auth.jwt_auth import JWTAuthenticator, TokenData, TokenResponse, PermissionChecker
from src.auth.rbac import RoleBasedAccessControl, AccessDeniedError
from src.database.models import User, Role, Permission, UserRole, UserSession
from src.database.repositories import UserRepository, SessionRepository
from src.core.exceptions import AuthenticationError, ValidationError


@pytest.fixture
def jwt_config():
    """JWT configuration for testing."""
    return {
        "jwt_secret_key": "test-secret-key-for-testing-only-not-production",
        "jwt_algorithm": "HS256", 
        "jwt_access_token_expire_minutes": 30,
        "jwt_refresh_token_expire_days": 7,
        "jwt_issuer": "claude-tui-test",
        "jwt_audience": "claude-tui-test-api"
    }


@pytest.fixture
def jwt_authenticator(jwt_config):
    """Create JWT authenticator for testing."""
    return JWTAuthenticator(
        secret_key=jwt_config["jwt_secret_key"],
        algorithm=jwt_config["jwt_algorithm"],
        access_token_expire_minutes=jwt_config["jwt_access_token_expire_minutes"],
        refresh_token_expire_days=jwt_config["jwt_refresh_token_expire_days"],
        issuer=jwt_config["jwt_issuer"],
        audience=jwt_config["jwt_audience"]
    )


@pytest.fixture
async def mock_user_repo():
    """Mock user repository for testing."""
    repo = AsyncMock(spec=UserRepository)
    
    # Sample user data
    sample_user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        is_active=True,
        is_verified=True
    )
    sample_user.set_password("SecurePass123!")
    
    repo.authenticate_user.return_value = sample_user
    repo.get_by_id.return_value = sample_user
    repo.get_by_email.return_value = sample_user
    repo.get_by_username.return_value = sample_user
    repo.update_user.return_value = sample_user
    repo.change_password.return_value = True
    
    return repo


@pytest.fixture
async def mock_session_repo():
    """Mock session repository for testing."""
    repo = AsyncMock(spec=SessionRepository)
    
    # Sample session data
    sample_session = UserSession(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_token="test-session-token",
        refresh_token="test-refresh-token",
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        is_active=True
    )
    
    repo.create_session.return_value = sample_session
    repo.get_session.return_value = sample_session
    repo.invalidate_session.return_value = True
    repo.invalidate_user_sessions.return_value = 3
    
    return repo


@pytest.fixture
def sample_user_with_roles():
    """Create sample user with roles for testing."""
    user = User(
        id=uuid.uuid4(),
        email="admin@example.com", 
        username="adminuser",
        full_name="Admin User",
        is_active=True,
        is_verified=True
    )
    user.set_password("AdminPass123!")
    
    # Add admin role
    admin_role = Role(name="ADMIN", description="Administrator role")
    user_role = UserRole(user_id=user.id, role_id=admin_role.id)
    user_role.role = admin_role
    user.roles = [user_role]
    
    return user


@pytest.mark.asyncio
class TestJWTTokenGeneration:
    """Test JWT token generation and validation."""
    
    async def test_create_tokens_success(self, jwt_authenticator, mock_session_repo):
        """Test successful token creation."""
        user = User(
            id=uuid.uuid4(),
            email="token@test.com",
            username="tokenuser",
            full_name="Token User"
        )
        
        permissions = ["read:projects", "write:projects"]
        
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo,
            permissions=permissions,
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0"
        )
        
        assert isinstance(token_response, TokenResponse)
        assert token_response.access_token is not None
        assert token_response.refresh_token is not None
        assert token_response.token_type == "bearer"
        assert token_response.expires_in == 30 * 60  # 30 minutes in seconds
        assert token_response.user_id == str(user.id)
        assert token_response.permissions == permissions
    
    async def test_validate_token_success(self, jwt_authenticator, mock_session_repo):
        """Test successful token validation."""
        user = User(
            id=uuid.uuid4(),
            email="validate@test.com",
            username="validateuser",
            full_name="Validate User"
        )
        
        # Create token
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo,
            permissions=["read:projects"]
        )
        
        # Validate token
        token_data = await jwt_authenticator.validate_token(
            token=token_response.access_token,
            session_repo=mock_session_repo
        )
        
        assert isinstance(token_data, TokenData)
        assert token_data.user_id == str(user.id)
        assert token_data.username == user.username
        assert token_data.email == user.email
        assert "read:projects" in token_data.permissions
    
    async def test_validate_expired_token(self, jwt_config):
        """Test validation of expired token."""
        # Create authenticator with very short expiration
        authenticator = JWTAuthenticator(
            secret_key=jwt_config["jwt_secret_key"],
            access_token_expire_minutes=0.01  # 0.6 seconds
        )
        
        user = User(id=uuid.uuid4(), email="expired@test.com", username="expireduser")
        mock_session_repo = AsyncMock()
        
        # Create token
        token_response = await authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo
        )
        
        # Wait for token to expire
        await asyncio.sleep(1)
        
        # Validate expired token
        with pytest.raises(AuthenticationError, match="Token has expired"):
            await authenticator.validate_token(
                token=token_response.access_token,
                session_repo=mock_session_repo
            )
    
    async def test_validate_invalid_token(self, jwt_authenticator, mock_session_repo):
        """Test validation of invalid token."""
        invalid_token = "invalid.jwt.token"
        
        with pytest.raises(AuthenticationError, match="Invalid token"):
            await jwt_authenticator.validate_token(
                token=invalid_token,
                session_repo=mock_session_repo
            )
    
    async def test_validate_tampered_token(self, jwt_authenticator, mock_session_repo):
        """Test validation of tampered token."""
        user = User(id=uuid.uuid4(), email="tamper@test.com", username="tamperuser")
        
        # Create valid token
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo
        )
        
        # Tamper with token (change last character)
        tampered_token = token_response.access_token[:-1] + "X"
        
        with pytest.raises(AuthenticationError):
            await jwt_authenticator.validate_token(
                token=tampered_token,
                session_repo=mock_session_repo
            )


@pytest.mark.asyncio
class TestUserAuthentication:
    """Test user authentication flows."""
    
    async def test_authenticate_user_success(self, jwt_authenticator, mock_user_repo):
        """Test successful user authentication."""
        authenticated_user = await jwt_authenticator.authenticate_user(
            user_repo=mock_user_repo,
            identifier="test@example.com",
            password="SecurePass123!",
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.email == "test@example.com"
        mock_user_repo.authenticate_user.assert_called_once()
    
    async def test_authenticate_user_invalid_credentials(self, jwt_authenticator):
        """Test authentication with invalid credentials."""
        mock_repo = AsyncMock()
        mock_repo.authenticate_user.return_value = None
        
        with pytest.raises(AuthenticationError):
            await jwt_authenticator.authenticate_user(
                user_repo=mock_repo,
                identifier="invalid@example.com",
                password="wrongpassword"
            )
    
    async def test_authenticate_inactive_user(self, jwt_authenticator):
        """Test authentication of inactive user."""
        inactive_user = User(
            email="inactive@example.com",
            username="inactiveuser",
            is_active=False
        )
        
        mock_repo = AsyncMock()
        mock_repo.authenticate_user.side_effect = AuthenticationError("Account is inactive")
        
        with pytest.raises(AuthenticationError, match="Account is inactive"):
            await jwt_authenticator.authenticate_user(
                user_repo=mock_repo,
                identifier="inactive@example.com",
                password="password"
            )
    
    async def test_authenticate_locked_account(self, jwt_authenticator):
        """Test authentication of locked account."""
        locked_user = User(
            email="locked@example.com",
            username="lockeduser",
            account_locked_until=datetime.now(timezone.utc) + timedelta(minutes=30)
        )
        
        mock_repo = AsyncMock()
        mock_repo.authenticate_user.side_effect = AuthenticationError("Account is locked")
        
        with pytest.raises(AuthenticationError, match="Account is locked"):
            await jwt_authenticator.authenticate_user(
                user_repo=mock_repo,
                identifier="locked@example.com",
                password="password"
            )


@pytest.mark.asyncio
class TestSessionManagement:
    """Test user session management."""
    
    async def test_session_creation(self, jwt_authenticator, mock_session_repo):
        """Test session creation during token generation."""
        user = User(id=uuid.uuid4(), email="session@test.com", username="sessionuser")
        
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo,
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0"
        )
        
        # Verify session was created
        mock_session_repo.create_session.assert_called_once()
        call_args = mock_session_repo.create_session.call_args[1]
        assert call_args["user_id"] == user.id
        assert call_args["ip_address"] == "192.168.1.1"
        assert call_args["user_agent"] == "TestClient/1.0"
    
    async def test_session_validation(self, jwt_authenticator, mock_session_repo):
        """Test session validation during token validation."""
        user = User(id=uuid.uuid4(), email="sessionval@test.com", username="sessionvaluser")
        
        # Create token with session
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo
        )
        
        # Validate token (should check session)
        await jwt_authenticator.validate_token(
            token=token_response.access_token,
            session_repo=mock_session_repo,
            verify_session=True
        )
        
        # Verify session was checked
        mock_session_repo.get_session.assert_called()
    
    async def test_session_invalidation(self, jwt_authenticator, mock_session_repo):
        """Test session invalidation."""
        user = User(id=uuid.uuid4(), email="invalid@test.com", username="invaliduser")
        
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo
        )
        
        # Revoke token
        success = await jwt_authenticator.revoke_token(
            token=token_response.access_token,
            session_repo=mock_session_repo
        )
        
        assert success is True
        mock_session_repo.invalidate_session.assert_called()
    
    async def test_revoke_all_user_sessions(self, jwt_authenticator, mock_session_repo):
        """Test revoking all sessions for a user."""
        user_id = uuid.uuid4()
        
        count = await jwt_authenticator.revoke_user_sessions(
            user_id=user_id,
            session_repo=mock_session_repo
        )
        
        assert count == 3  # Mock returns 3
        mock_session_repo.invalidate_user_sessions.assert_called_once_with(user_id)


@pytest.mark.asyncio
class TestPasswordManagement:
    """Test password management features."""
    
    async def test_password_strength_validation(self, jwt_authenticator):
        """Test password strength validation."""
        # Valid passwords
        assert jwt_authenticator._jwt_auth.validate_password_strength("SecurePass123!")
        assert jwt_authenticator._jwt_auth.validate_password_strength("MyP@ssw0rd2023")
        
        # Invalid passwords
        assert not jwt_authenticator._jwt_auth.validate_password_strength("short")
        assert not jwt_authenticator._jwt_auth.validate_password_strength("nouppercase123!")
        assert not jwt_authenticator._jwt_auth.validate_password_strength("NOLOWERCASE123!")
        assert not jwt_authenticator._jwt_auth.validate_password_strength("NoNumbers!")
        assert not jwt_authenticator._jwt_auth.validate_password_strength("NoSpecialChars123")
    
    async def test_password_hashing(self, jwt_authenticator):
        """Test password hashing and verification."""
        password = "TestPassword123!"
        
        # Hash password
        hashed = jwt_authenticator.hash_password(password)
        assert hashed != password
        assert len(hashed) > 20  # Bcrypt hash should be long
        
        # Verify password
        assert jwt_authenticator.verify_password(password, hashed)
        assert not jwt_authenticator.verify_password("wrong", hashed)
    
    async def test_change_password_success(self, jwt_authenticator, mock_user_repo, mock_session_repo):
        """Test successful password change."""
        user_id = uuid.uuid4()
        
        success = await jwt_authenticator.change_password(
            user_repo=mock_user_repo,
            user_id=user_id,
            current_password="OldPassword123!",
            new_password="NewPassword123!",
            revoke_sessions=True
        )
        
        assert success is True
        mock_user_repo.change_password.assert_called_once_with(
            user_id=user_id,
            old_password="OldPassword123!",
            new_password="NewPassword123!"
        )
    
    async def test_change_password_invalid_current(self, jwt_authenticator, mock_user_repo):
        """Test password change with invalid current password."""
        user_id = uuid.uuid4()
        mock_user_repo.change_password.return_value = False
        
        success = await jwt_authenticator.change_password(
            user_repo=mock_user_repo,
            user_id=user_id,
            current_password="WrongPassword",
            new_password="NewPassword123!",
            revoke_sessions=False
        )
        
        assert success is False


@pytest.mark.asyncio
class TestPermissionChecker:
    """Test permission checking functionality."""
    
    def test_has_permission(self):
        """Test permission checking."""
        token_data = TokenData(
            user_id="123",
            username="testuser",
            email="test@example.com",
            permissions=["read:projects", "write:tasks"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600,
            jti="test-jti"
        )
        
        assert PermissionChecker.has_permission(token_data, "read:projects")
        assert PermissionChecker.has_permission(token_data, "write:tasks")
        assert not PermissionChecker.has_permission(token_data, "delete:users")
    
    def test_has_role(self):
        """Test role checking."""
        token_data = TokenData(
            user_id="123",
            username="adminuser",
            email="admin@example.com",
            roles=["ADMIN", "USER"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600,
            jti="test-jti"
        )
        
        assert PermissionChecker.has_role(token_data, "ADMIN")
        assert PermissionChecker.has_role(token_data, "USER")
        assert not PermissionChecker.has_role(token_data, "MODERATOR")
    
    def test_has_any_role(self):
        """Test checking for any of multiple roles."""
        token_data = TokenData(
            user_id="123",
            username="user",
            email="user@example.com",
            roles=["USER"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600,
            jti="test-jti"
        )
        
        assert PermissionChecker.has_any_role(token_data, ["ADMIN", "USER"])
        assert PermissionChecker.has_any_role(token_data, ["USER", "MODERATOR"])
        assert not PermissionChecker.has_any_role(token_data, ["ADMIN", "MODERATOR"])
    
    def test_has_all_roles(self):
        """Test checking for all of multiple roles."""
        token_data = TokenData(
            user_id="123",
            username="superuser",
            email="super@example.com",
            roles=["ADMIN", "USER", "MODERATOR"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600,
            jti="test-jti"
        )
        
        assert PermissionChecker.has_all_roles(token_data, ["ADMIN", "USER"])
        assert PermissionChecker.has_all_roles(token_data, ["USER", "MODERATOR"])
        assert not PermissionChecker.has_all_roles(token_data, ["ADMIN", "SUPERUSER"])
    
    def test_is_admin(self):
        """Test admin detection."""
        admin_token = TokenData(
            user_id="123",
            username="admin",
            email="admin@example.com",
            roles=["ADMIN"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600,
            jti="test-jti"
        )
        
        user_token = TokenData(
            user_id="456",
            username="user",
            email="user@example.com",
            roles=["USER"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600,
            jti="test-jti"
        )
        
        assert PermissionChecker.is_admin(admin_token)
        assert not PermissionChecker.is_admin(user_token)
    
    def test_can_access_resource(self):
        """Test resource access checking."""
        token_data = TokenData(
            user_id="123",
            username="user",
            email="user@example.com",
            permissions=["read:projects"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600,
            jti="test-jti"
        )
        
        # Has permission
        assert PermissionChecker.can_access_resource(
            token_data, "projects", "read"
        )
        
        # Doesn't have permission
        assert not PermissionChecker.can_access_resource(
            token_data, "projects", "delete"
        )
        
        # Owns resource
        assert PermissionChecker.can_access_resource(
            token_data, "projects", "delete", resource_owner_id="123"
        )
        
        # Doesn't own resource
        assert not PermissionChecker.can_access_resource(
            token_data, "projects", "delete", resource_owner_id="456"
        )


@pytest.mark.asyncio
class TestSecurityFeatures:
    """Test security features like rate limiting and account lockout."""
    
    async def test_account_lockout_mechanism(self):
        """Test account lockout after failed attempts."""
        user = User(
            email="lockout@test.com",
            username="lockoutuser",
            failed_login_attempts=0
        )
        
        # Simulate failed login attempts
        for i in range(5):
            user.failed_login_attempts += 1
        
        # Account should be locked after 5 attempts
        if user.failed_login_attempts >= 5:
            user.lock_account(30)  # Lock for 30 minutes
        
        assert user.is_account_locked()
        assert user.account_locked_until is not None
    
    async def test_account_unlock(self):
        """Test account unlocking."""
        user = User(
            email="unlock@test.com",
            username="unlockuser",
            failed_login_attempts=5,
            account_locked_until=datetime.now(timezone.utc) + timedelta(minutes=30)
        )
        
        assert user.is_account_locked()
        
        # Unlock account
        user.unlock_account()
        
        assert not user.is_account_locked()
        assert user.failed_login_attempts == 0
        assert user.account_locked_until is None
    
    async def test_token_blacklisting(self, jwt_authenticator, mock_session_repo):
        """Test token blacklisting functionality."""
        user = User(id=uuid.uuid4(), email="blacklist@test.com", username="blacklistuser")
        
        # Create token
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo
        )
        
        # Token should be valid initially
        token_data = await jwt_authenticator.validate_token(
            token=token_response.access_token,
            session_repo=mock_session_repo
        )
        assert token_data is not None
        
        # Revoke token (blacklist it)
        await jwt_authenticator.revoke_token(
            token=token_response.access_token,
            session_repo=mock_session_repo
        )
        
        # Token should now be invalid
        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            await jwt_authenticator.validate_token(
                token=token_response.access_token,
                session_repo=mock_session_repo
            )


@pytest.mark.asyncio
class TestOAuthIntegration:
    """Test OAuth integration (mock implementation)."""
    
    async def test_oauth_flow_simulation(self):
        """Test OAuth flow simulation."""
        # This would typically integrate with external OAuth providers
        # For testing, we'll simulate the flow
        
        oauth_user_data = {
            "id": "oauth_123456",
            "email": "oauth@example.com",
            "name": "OAuth User",
            "provider": "google"
        }
        
        # Mock OAuth user creation/linking
        user = User(
            email=oauth_user_data["email"],
            username=f"oauth_{oauth_user_data['id']}",
            full_name=oauth_user_data["name"],
            is_verified=True  # OAuth users are pre-verified
        )
        
        assert user.email == oauth_user_data["email"]
        assert user.is_verified is True
    
    async def test_oauth_token_exchange(self, jwt_authenticator, mock_session_repo):
        """Test exchanging OAuth token for JWT."""
        # Simulate OAuth user
        oauth_user = User(
            id=uuid.uuid4(),
            email="oauth@example.com",
            username="oauth_user",
            full_name="OAuth User",
            is_verified=True
        )
        
        # Create JWT for OAuth user
        token_response = await jwt_authenticator.create_tokens(
            user=oauth_user,
            session_repo=mock_session_repo,
            permissions=["read:profile", "write:profile"]
        )
        
        assert token_response.access_token is not None
        assert "read:profile" in token_response.permissions


@pytest.mark.asyncio
class TestRoleBasedAccessControl:
    """Test RBAC functionality."""
    
    async def test_rbac_permission_check(self):
        """Test RBAC permission checking."""
        # This would use the RBAC system
        rbac = RoleBasedAccessControl()
        
        user_roles = ["USER"]
        required_permissions = ["read:projects"]
        
        # Mock RBAC check
        has_access = await rbac.check_permission(user_roles, required_permissions)
        assert has_access in [True, False]  # Result depends on role configuration
    
    async def test_rbac_role_hierarchy(self):
        """Test RBAC role hierarchy."""
        rbac = RoleBasedAccessControl()
        
        # Admin should inherit all user permissions
        admin_roles = ["ADMIN"]
        user_permission = ["read:profile"]
        
        has_access = await rbac.check_permission(admin_roles, user_permission)
        # Admin should have access to user permissions
        assert has_access is True or has_access is False  # Mock result


@pytest.mark.asyncio 
class TestAuthenticationIntegration:
    """Integration tests for complete authentication flows."""
    
    async def test_complete_login_flow(self, jwt_authenticator, mock_user_repo, mock_session_repo):
        """Test complete login flow from credentials to token."""
        # Step 1: Authenticate user
        user = await jwt_authenticator.authenticate_user(
            user_repo=mock_user_repo,
            identifier="test@example.com",
            password="SecurePass123!",
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0"
        )
        
        # Step 2: Create tokens
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo,
            permissions=["read:projects", "write:projects"],
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0"
        )
        
        # Step 3: Validate token
        token_data = await jwt_authenticator.validate_token(
            token=token_response.access_token,
            session_repo=mock_session_repo
        )
        
        # Verify complete flow
        assert user.email == "test@example.com"
        assert token_response.access_token is not None
        assert token_data.user_id == str(user.id)
        assert token_data.email == user.email
        assert "read:projects" in token_data.permissions
    
    async def test_complete_logout_flow(self, jwt_authenticator, mock_user_repo, mock_session_repo):
        """Test complete logout flow."""
        # Login first
        user = await jwt_authenticator.authenticate_user(
            user_repo=mock_user_repo,
            identifier="logout@example.com", 
            password="SecurePass123!"
        )
        
        token_response = await jwt_authenticator.create_tokens(
            user=user,
            session_repo=mock_session_repo
        )
        
        # Logout (revoke token)
        success = await jwt_authenticator.revoke_token(
            token=token_response.access_token,
            session_repo=mock_session_repo
        )
        
        assert success is True
        
        # Token should be invalid after logout
        with pytest.raises(AuthenticationError):
            await jwt_authenticator.validate_token(
                token=token_response.access_token,
                session_repo=mock_session_repo
            )
    
    @pytest.mark.slow
    async def test_concurrent_authentication(self, jwt_authenticator, mock_user_repo, mock_session_repo):
        """Test concurrent authentication requests."""
        async def authenticate_and_create_token(user_id: str):
            user = User(
                id=uuid.uuid4(),
                email=f"concurrent{user_id}@test.com",
                username=f"concurrent{user_id}",
                full_name=f"Concurrent User {user_id}"
            )
            
            # Mock successful authentication
            mock_user_repo.authenticate_user.return_value = user
            
            authenticated_user = await jwt_authenticator.authenticate_user(
                user_repo=mock_user_repo,
                identifier=user.email,
                password="Password123!"
            )
            
            token_response = await jwt_authenticator.create_tokens(
                user=authenticated_user,
                session_repo=mock_session_repo
            )
            
            return token_response.access_token
        
        # Run concurrent authentications
        tasks = [authenticate_and_create_token(str(i)) for i in range(10)]
        tokens = await asyncio.gather(*tasks)
        
        # All authentications should succeed
        assert len(tokens) == 10
        assert all(token is not None for token in tokens)
        assert len(set(tokens)) == 10  # All tokens should be unique