"""
Comprehensive Security Tests
Tests all security components including authentication, authorization, and middleware
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import Mock, patch, AsyncMock
import uuid
import json

from src.auth.jwt_auth import JWTManager, PasswordManager
from src.auth.rbac import RBACManager
from src.middleware.security_middleware import (
    SecurityHeadersMiddleware,
    RateLimitingMiddleware,
    ContentSecurityMiddleware
)
from src.database.models import User, Role, Permission
from src.database.repositories import UserRepository, RepositoryFactory


class TestJWTAuthentication:
    """Test JWT authentication system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.jwt_manager = JWTManager()
        self.password_manager = PasswordManager()
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "TestPassword123!"
        
        # Hash password
        hashed = self.password_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        
        # Verify correct password
        assert self.password_manager.verify_password(password, hashed)
        
        # Verify wrong password
        assert not self.password_manager.verify_password("WrongPassword", hashed)
    
    def test_password_strength_validation(self):
        """Test password strength validation"""
        # Valid password
        strong_password = "StrongPass123!"
        result = self.password_manager.validate_password_strength(strong_password)
        assert result["valid"] is True
        assert result["score"] >= 4
        assert result["strength"] in ["Medium", "Strong"]
        
        # Weak passwords
        weak_passwords = [
            "123456",  # Too short, no letters
            "password",  # No numbers, no special chars
            "Password",  # No numbers, no special chars
            "Password123",  # No special chars
            "password123!",  # No uppercase
        ]
        
        for weak_pass in weak_passwords:
            result = self.password_manager.validate_password_strength(weak_pass)
            assert result["valid"] is False
            assert len(result["errors"]) > 0
    
    def test_jwt_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        user_data = {
            "sub": "test@example.com",
            "user_id": str(uuid.uuid4()),
            "email": "test@example.com",
            "username": "testuser",
            "roles": ["USER"]
        }
        
        # Create access token
        access_token = self.jwt_manager.create_access_token(user_data)
        assert access_token is not None
        assert len(access_token) > 100  # JWT tokens are long
        
        # Verify token
        payload = self.jwt_manager.verify_token(access_token)
        assert payload is not None
        assert payload["sub"] == user_data["sub"]
        assert payload["user_id"] == user_data["user_id"]
        assert payload["token_type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload
    
    def test_jwt_refresh_token(self):
        """Test JWT refresh token functionality"""
        user_data = {
            "sub": "test@example.com",
            "user_id": str(uuid.uuid4()),
            "email": "test@example.com",
            "username": "testuser"
        }
        
        # Create refresh token
        refresh_token = self.jwt_manager.create_refresh_token(user_data)
        assert refresh_token is not None
        
        # Verify refresh token
        payload = self.jwt_manager.verify_token(refresh_token, "refresh")
        assert payload is not None
        assert payload["token_type"] == "refresh"
        
        # Generate new access token from refresh token
        new_access_token = self.jwt_manager.refresh_access_token(refresh_token)
        assert new_access_token is not None
        
        # Verify new access token
        new_payload = self.jwt_manager.verify_token(new_access_token)
        assert new_payload["sub"] == user_data["sub"]
        assert new_payload["token_type"] == "access"
    
    def test_token_revocation(self):
        """Test token revocation functionality"""
        user_data = {"sub": "test@example.com"}
        
        # Create and verify token
        token = self.jwt_manager.create_access_token(user_data)
        payload = self.jwt_manager.verify_token(token)
        assert payload is not None
        
        # Revoke token
        success = self.jwt_manager.revoke_token(token)
        assert success is True
        
        # Verify revoked token fails
        revoked_payload = self.jwt_manager.verify_token(token)
        assert revoked_payload is None
    
    def test_expired_token(self):
        """Test expired token handling"""
        user_data = {"sub": "test@example.com"}
        
        # Create token with immediate expiration
        expired_token = self.jwt_manager.create_access_token(
            user_data,
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        # Verify expired token fails
        payload = self.jwt_manager.verify_token(expired_token)
        assert payload is None


class TestRBACSystem:
    """Test Role-Based Access Control system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.rbac = RBACManager()
    
    def test_default_permissions_initialization(self):
        """Test that default permissions are properly initialized"""
        assert len(self.rbac.permissions) > 0
        
        # Check for core permissions
        assert "users.create" in self.rbac.permissions
        assert "users.read" in self.rbac.permissions
        assert "users.update" in self.rbac.permissions
        assert "users.delete" in self.rbac.permissions
        assert "projects.create" in self.rbac.permissions
        assert "tasks.create" in self.rbac.permissions
    
    def test_default_roles_initialization(self):
        """Test that default roles are properly initialized"""
        assert len(self.rbac.roles) > 0
        
        # Check for core roles
        required_roles = ["SUPER_ADMIN", "ADMIN", "PROJECT_MANAGER", "DEVELOPER", "VIEWER", "GUEST"]
        for role_name in required_roles:
            assert role_name in self.rbac.roles
    
    def test_permission_creation(self):
        """Test custom permission creation"""
        permission_name = "test.custom"
        
        permission = self.rbac.create_permission(
            name=permission_name,
            resource="test",
            action="custom",
            description="Custom test permission"
        )
        
        assert permission.name == permission_name
        assert permission.resource == "test"
        assert permission.action == "custom"
        assert permission_name in self.rbac.permissions
    
    def test_role_creation(self):
        """Test custom role creation"""
        role_name = "TEST_ROLE"
        permissions = ["users.read", "projects.read"]
        
        role = self.rbac.create_role(
            name=role_name,
            description="Test role",
            permissions=permissions
        )
        
        assert role.name == role_name
        assert len(role.permissions) == 2
        assert role_name in self.rbac.roles
    
    def test_permission_assignment_to_role(self):
        """Test assigning permissions to roles"""
        role_name = "TEST_ROLE_2"
        
        # Create role
        self.rbac.create_role(role_name, "Test role 2")
        
        # Assign permission
        success = self.rbac.assign_permission_to_role(role_name, "users.read")
        assert success is True
        
        # Check role has permission
        role_permissions = self.rbac.get_role_permissions(role_name)
        assert "users.read" in role_permissions
    
    def test_user_permission_checking(self):
        """Test user permission checking"""
        user_permissions = ["users.read", "projects.create", "tasks.update"]
        
        # Test single permission
        assert self.rbac.user_has_permission(user_permissions, "users.read") is True
        assert self.rbac.user_has_permission(user_permissions, "users.delete") is False
        
        # Test all permissions
        required_all = ["users.read", "projects.create"]
        assert self.rbac.user_has_all_permissions(user_permissions, required_all) is True
        
        required_all_fail = ["users.read", "users.delete"]
        assert self.rbac.user_has_all_permissions(user_permissions, required_all_fail) is False
        
        # Test any permission
        required_any = ["users.delete", "projects.create"]
        assert self.rbac.user_has_any_permission(user_permissions, required_any) is True
        
        required_any_fail = ["users.delete", "system.admin"]
        assert self.rbac.user_has_any_permission(user_permissions, required_any_fail) is False
    
    def test_resource_access_checking(self):
        """Test resource access checking"""
        user_permissions = ["projects.read", "projects.update"]
        
        # Direct permission
        assert self.rbac.check_resource_access(
            user_permissions, "projects", "read"
        ) is True
        
        # No permission
        assert self.rbac.check_resource_access(
            user_permissions, "projects", "delete"
        ) is False
        
        # Admin override
        admin_permissions = ["system.admin"]
        assert self.rbac.check_resource_access(
            admin_permissions, "projects", "delete"
        ) is True
    
    def test_rbac_configuration_export_import(self):
        """Test RBAC configuration export and import"""
        # Export configuration
        config = self.rbac.export_rbac_config()
        
        assert "permissions" in config
        assert "roles" in config
        assert len(config["permissions"]) > 0
        assert len(config["roles"]) > 0
        
        # Create new RBAC instance and import
        new_rbac = RBACManager()
        new_rbac.import_rbac_config(config)
        
        # Verify import
        assert len(new_rbac.permissions) == len(self.rbac.permissions)
        assert len(new_rbac.roles) == len(self.rbac.roles)


@pytest.mark.asyncio
class TestSecurityMiddleware:
    """Test security middleware components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.app = FastAPI()
        self.client = TestClient(self.app)
    
    def test_security_headers_middleware(self):
        """Test security headers middleware"""
        # Add middleware
        self.app.add_middleware(SecurityHeadersMiddleware, enabled=True)
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # Make request
        response = self.client.get("/test")
        
        # Check security headers
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "X-Request-ID" in response.headers
        
        # Check header values
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_rate_limiting_middleware(self):
        """Test rate limiting middleware"""
        # Add middleware with low limits for testing
        self.app.add_middleware(
            RateLimitingMiddleware,
            requests_per_minute=2,
            burst_size=1,
            enabled=True
        )
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # First request should succeed
        response1 = self.client.get("/test")
        assert response1.status_code == 200
        assert "X-RateLimit-Limit" in response1.headers
        assert "X-RateLimit-Remaining" in response1.headers
        
        # Second request should succeed
        response2 = self.client.get("/test")
        assert response2.status_code == 200
        
        # Third request should be rate limited
        response3 = self.client.get("/test")
        assert response3.status_code == 429
        assert "Retry-After" in response3.headers
        assert response3.json()["status"] == "error"
        assert "rate limit" in response3.json()["message"].lower()
    
    def test_content_security_middleware(self):
        """Test content security middleware"""
        # Add middleware with small content limit for testing
        self.app.add_middleware(
            ContentSecurityMiddleware,
            max_content_length=100,  # 100 bytes
            allowed_content_types=["application/json"]
        )
        
        @self.app.post("/test")
        async def test_endpoint(data: dict):
            return {"received": data}
        
        # Test valid content type and size
        small_data = {"test": "data"}
        response1 = self.client.post("/test", json=small_data)
        assert response1.status_code == 200
        
        # Test invalid content type
        response2 = self.client.post(
            "/test",
            data="test data",
            headers={"Content-Type": "text/plain"}
        )
        assert response2.status_code == 415
        assert "media type" in response2.json()["message"].lower()
        
        # Test content too large
        large_data = {"large": "x" * 1000}  # Over 100 bytes
        response3 = self.client.post("/test", json=large_data)
        assert response3.status_code == 413
        assert "too large" in response3.json()["message"].lower()


@pytest.mark.asyncio
class TestUserRepository:
    """Test user repository security features"""
    
    def setup_method(self):
        """Setup test environment"""
        self.session = AsyncMock()
        self.user_repo = UserRepository(self.session)
    
    @pytest.mark.asyncio
    async def test_user_authentication_success(self):
        """Test successful user authentication"""
        # Mock user with correct password
        mock_user = Mock(spec=User)
        mock_user.id = uuid.uuid4()
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.is_active = True
        mock_user.is_account_locked.return_value = False
        mock_user.verify_password.return_value = True
        mock_user.failed_login_attempts = 0
        
        # Mock database query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        self.session.execute.return_value = mock_result
        
        # Test authentication
        authenticated_user = await self.user_repo.authenticate_user(
            "test@example.com", "correct_password", "192.168.1.1"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.username == "testuser"
        assert mock_user.verify_password.called
        assert self.session.commit.called
    
    @pytest.mark.asyncio
    async def test_user_authentication_failure(self):
        """Test failed user authentication"""
        # Mock user with wrong password
        mock_user = Mock(spec=User)
        mock_user.id = uuid.uuid4()
        mock_user.username = "testuser"
        mock_user.is_active = True
        mock_user.is_account_locked.return_value = False
        mock_user.verify_password.return_value = False
        mock_user.failed_login_attempts = 0
        
        # Mock database query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        self.session.execute.return_value = mock_result
        
        # Test authentication
        authenticated_user = await self.user_repo.authenticate_user(
            "test@example.com", "wrong_password", "192.168.1.1"
        )
        
        assert authenticated_user is None
        assert mock_user.verify_password.called
        assert mock_user.failed_login_attempts == 1
        assert self.session.commit.called
    
    @pytest.mark.asyncio
    async def test_account_lockout(self):
        """Test account lockout after failed attempts"""
        # Mock user with multiple failed attempts
        mock_user = Mock(spec=User)
        mock_user.id = uuid.uuid4()
        mock_user.username = "testuser"
        mock_user.is_active = True
        mock_user.is_account_locked.return_value = True  # Already locked
        
        # Mock database query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        self.session.execute.return_value = mock_result
        
        # Test authentication with locked account
        authenticated_user = await self.user_repo.authenticate_user(
            "test@example.com", "any_password", "192.168.1.1"
        )
        
        assert authenticated_user is None
        assert not mock_user.verify_password.called  # Should not check password if locked
    
    @pytest.mark.asyncio
    async def test_password_change_validation(self):
        """Test password change with validation"""
        # Mock user
        mock_user = Mock(spec=User)
        mock_user.id = uuid.uuid4()
        mock_user.username = "testuser"
        mock_user.verify_password.return_value = True
        mock_user.set_password = Mock()
        
        # Mock get_by_id
        self.user_repo.get_by_id = AsyncMock(return_value=mock_user)
        
        # Test password change
        success = await self.user_repo.change_password(
            mock_user.id, "old_password", "NewPassword123!"
        )
        
        assert success is True
        assert mock_user.verify_password.called
        assert mock_user.set_password.called
        assert self.session.commit.called


class TestIntegrationSecurity:
    """Integration tests for security components"""
    
    def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        jwt_manager = JWTManager()
        password_manager = PasswordManager()
        
        # Create user credentials
        email = "test@example.com"
        password = "SecurePassword123!"
        
        # Hash password (as would be done during registration)
        hashed_password = password_manager.hash_password(password)
        
        # Simulate login
        login_successful = password_manager.verify_password(password, hashed_password)
        assert login_successful
        
        # Create JWT tokens
        user_data = {
            "sub": email,
            "user_id": str(uuid.uuid4()),
            "email": email,
            "username": "testuser",
            "roles": ["USER"]
        }
        
        tokens = jwt_manager.create_token_pair(user_data)
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        
        # Verify access token
        access_payload = jwt_manager.verify_token(tokens["access_token"])
        assert access_payload["sub"] == email
        
        # Use refresh token to get new access token
        new_access_token = jwt_manager.refresh_access_token(tokens["refresh_token"])
        assert new_access_token is not None
        
        # Verify new access token
        new_payload = jwt_manager.verify_token(new_access_token)
        assert new_payload["sub"] == email
    
    def test_rbac_with_jwt_integration(self):
        """Test RBAC integration with JWT tokens"""
        jwt_manager = JWTManager()
        rbac = RBACManager()
        
        # Create user with specific roles
        user_roles = ["DEVELOPER", "PROJECT_MANAGER"]
        user_data = {
            "sub": "dev@example.com",
            "user_id": str(uuid.uuid4()),
            "roles": user_roles
        }
        
        # Create token
        token = jwt_manager.create_access_token(user_data)
        payload = jwt_manager.verify_token(token)
        
        # Get effective permissions for user roles
        effective_permissions = rbac.get_user_effective_permissions(user_roles)
        
        # Check that developer can create tasks
        assert rbac.user_has_permission(effective_permissions, "tasks.create")
        
        # Check that developer can read projects
        assert rbac.user_has_permission(effective_permissions, "projects.read")
        
        # Check that developer cannot manage system
        assert not rbac.user_has_permission(effective_permissions, "system.admin")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])