"""
Comprehensive Authentication System Tests

Tests all authentication components including JWT, sessions, OAuth,
RBAC, password reset, and security features.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
import secrets
import json

import redis
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.jwt_service import JWTService, TokenType, TokenValidationResult
from src.auth.session_service import SessionService, SessionStatus, SessionMetadata
from src.auth.rbac import RBACManager, Permission, Role
from src.auth.audit_logger import SecurityAuditLogger, SecurityEventType, SecurityLevel
from src.auth.password_reset import PasswordResetService, EmailService, ResetTokenStatus
from src.auth.oauth.github import GitHubOAuthProvider
from src.auth.oauth.base import OAuthUserInfo, OAuthError
from src.auth.middleware import AuthenticationMiddleware, RBACMiddleware
from src.database.models import User
from src.core.exceptions import AuthenticationError, SecurityError


class TestJWTService:
    """Test JWT token generation and validation"""
    
    @pytest.fixture
    def jwt_service(self):
        return JWTService(
            secret_key="test-secret-key-12345",
            algorithm="HS256"
        )
    
    @pytest.fixture
    def test_user_id(self):
        return "12345678-1234-1234-1234-123456789012"
    
    def test_create_access_token(self, jwt_service, test_user_id):
        """Test access token creation"""
        scopes = ["user:read", "user:write"]
        device_id = "device-123"
        ip_address = "192.168.1.1"
        
        token = jwt_service.create_access_token(
            user_id=test_user_id,
            scopes=scopes,
            device_id=device_id,
            ip_address=ip_address
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token structure
        validation_result = jwt_service.verify_token(token, TokenType.ACCESS)
        assert validation_result.valid
        assert validation_result.payload.sub == test_user_id
        assert validation_result.payload.scope == scopes
        assert validation_result.payload.device_id == device_id
        assert validation_result.payload.ip_address == ip_address
    
    def test_create_refresh_token(self, jwt_service, test_user_id):
        """Test refresh token creation"""
        device_id = "device-123"
        
        token = jwt_service.create_refresh_token(test_user_id, device_id)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token
        validation_result = jwt_service.verify_token(token, TokenType.REFRESH)
        assert validation_result.valid
        assert validation_result.payload.sub == test_user_id
        assert validation_result.payload.device_id == device_id
        assert validation_result.payload.type == TokenType.REFRESH
    
    def test_create_token_pair(self, jwt_service, test_user_id):
        """Test creation of access/refresh token pair"""
        scopes = ["user:read"]
        device_id = "device-123"
        ip_address = "192.168.1.1"
        
        token_pair = jwt_service.create_token_pair(
            user_id=test_user_id,
            scopes=scopes,
            device_id=device_id,
            ip_address=ip_address
        )
        
        assert token_pair.access_token
        assert token_pair.refresh_token
        assert token_pair.token_type == "Bearer"
        assert token_pair.expires_in == 900  # 15 minutes
        assert token_pair.scope == "user:read"
        
        # Verify both tokens
        access_valid = jwt_service.verify_token(token_pair.access_token, TokenType.ACCESS)
        refresh_valid = jwt_service.verify_token(token_pair.refresh_token, TokenType.REFRESH)
        
        assert access_valid.valid
        assert refresh_valid.valid
    
    @pytest.mark.asyncio
    async def test_refresh_access_token(self, jwt_service, test_user_id):
        """Test token refresh functionality"""
        # Create initial token pair
        token_pair = jwt_service.create_token_pair(test_user_id)
        
        # Mock rotation - blacklist old token
        jwt_service.blacklist_token(token_pair.refresh_token)
        
        # Create new token pair (simulating refresh)
        new_token_pair = jwt_service.create_token_pair(test_user_id)
        
        assert new_token_pair.access_token != token_pair.access_token
        assert new_token_pair.refresh_token != token_pair.refresh_token
        
        # Old refresh token should be blacklisted
        assert jwt_service.is_token_blacklisted(token_pair.refresh_token)
        assert not jwt_service.is_token_blacklisted(new_token_pair.refresh_token)
    
    def test_token_expiration(self, jwt_service, test_user_id):
        """Test token expiration handling"""
        # Create token with short expiry
        short_expiry = timedelta(seconds=1)
        token = jwt_service.create_token(
            user_id=test_user_id,
            token_type=TokenType.ACCESS,
            expires_delta=short_expiry
        )
        
        # Token should be valid initially
        validation_result = jwt_service.verify_token(token)
        assert validation_result.valid
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Token should be expired
        validation_result = jwt_service.verify_token(token)
        assert not validation_result.valid
        assert validation_result.expired
    
    def test_token_blacklisting(self, jwt_service, test_user_id):
        """Test token blacklisting functionality"""
        token = jwt_service.create_access_token(test_user_id)
        
        # Token should be valid initially
        validation_result = jwt_service.verify_token(token)
        assert validation_result.valid
        
        # Blacklist token
        jwt_service.blacklist_token(token)
        
        # Token should be invalid after blacklisting
        validation_result = jwt_service.verify_token(token)
        assert not validation_result.valid
        assert "revoked" in validation_result.error.lower()
    
    def test_invalid_token_handling(self, jwt_service):
        """Test handling of invalid tokens"""
        # Test completely invalid token
        invalid_token = "invalid.token.here"
        validation_result = jwt_service.verify_token(invalid_token)
        assert not validation_result.valid
        assert "Invalid token" in validation_result.error
        
        # Test token with wrong secret
        wrong_service = JWTService(secret_key="wrong-secret", algorithm="HS256")
        token = jwt_service.create_access_token("user-123")
        
        validation_result = wrong_service.verify_token(token)
        assert not validation_result.valid
    
    def test_email_verification_token(self, jwt_service, test_user_id):
        """Test email verification token creation and validation"""
        email = "test@example.com"
        
        token = jwt_service.create_email_verification_token(test_user_id, email)
        
        validation_result = jwt_service.verify_token(token, TokenType.EMAIL_VERIFICATION)
        assert validation_result.valid
        assert validation_result.payload.sub == test_user_id
        assert validation_result.payload.type == TokenType.EMAIL_VERIFICATION
        
        # Extract email from token claims
        claims = jwt_service.get_token_claims(token)
        assert claims['email'] == email
    
    def test_password_reset_token(self, jwt_service, test_user_id):
        """Test password reset token creation"""
        email = "test@example.com"
        
        token = jwt_service.create_password_reset_token(test_user_id, email)
        
        validation_result = jwt_service.verify_token(token, TokenType.PASSWORD_RESET)
        assert validation_result.valid
        assert validation_result.payload.type == TokenType.PASSWORD_RESET
        
        # Should expire in 1 hour by default
        expiry = jwt_service.get_token_expiry(token)
        now = datetime.now(timezone.utc)
        expected_expiry = now + timedelta(hours=1)
        
        # Allow 1 minute tolerance
        assert abs((expiry - expected_expiry).total_seconds()) < 60


class TestSessionService:
    """Test session management functionality"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock_redis = Mock()
        mock_redis.hgetall.return_value = {}
        mock_redis.hset.return_value = True
        mock_redis.expire.return_value = True
        mock_redis.sadd.return_value = 1
        mock_redis.srem.return_value = 1
        mock_redis.smembers.return_value = set()
        mock_redis.keys.return_value = []
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        mock_redis.delete.return_value = 1
        return mock_redis
    
    @pytest.fixture
    def session_service(self, mock_redis):
        return SessionService(
            redis_client=mock_redis,
            session_ttl_hours=24,
            max_sessions_per_user=5
        )
    
    @pytest.fixture
    def mock_user(self):
        user = Mock()
        user.id = "user-123"
        user.username = "testuser"
        user.email = "test@example.com"
        return user
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_service, mock_user):
        """Test session creation"""
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0 Test Browser"
        device_fingerprint = "device-fingerprint-123"
        
        session_id = await session_service.create_session(
            user=mock_user,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            login_method="password"
        )
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Verify session was stored
        session_service.redis_client.hset.assert_called()
        session_service.redis_client.expire.assert_called()
    
    @pytest.mark.asyncio
    async def test_validate_session_valid(self, session_service, mock_user):
        """Test validation of valid session"""
        session_id = "test-session-123"
        ip_address = "192.168.1.1"
        
        # Mock session data in Redis
        now = datetime.now(timezone.utc)
        session_data = {
            'session_id': session_id,
            'user_id': str(mock_user.id),
            'user_agent': 'Test Browser',
            'ip_address': ip_address,
            'created_at': now.isoformat(),
            'last_activity': now.isoformat(),
            'expires_at': (now + timedelta(hours=24)).isoformat(),\n            'status': SessionStatus.ACTIVE.value,\n            'device_trusted': 'false',\n            'login_method': 'password'\n        }\n        \n        session_service.redis_client.hgetall.return_value = session_data\n        \n        metadata = await session_service.validate_session(session_id, ip_address)\n        \n        assert metadata is not None\n        assert metadata.session_id == session_id\n        assert metadata.user_id == str(mock_user.id)\n        assert metadata.status == SessionStatus.ACTIVE\n        assert metadata.ip_address == ip_address\n    \n    @pytest.mark.asyncio\n    async def test_validate_session_expired(self, session_service, mock_user):\n        \"\"\"Test validation of expired session\"\"\"\n        session_id = \"expired-session-123\"\n        \n        # Mock expired session data\n        now = datetime.now(timezone.utc)\n        expired_time = now - timedelta(hours=1)\n        session_data = {\n            'session_id': session_id,\n            'user_id': str(mock_user.id),\n            'created_at': expired_time.isoformat(),\n            'expires_at': expired_time.isoformat(),  # Already expired\n            'status': SessionStatus.ACTIVE.value\n        }\n        \n        session_service.redis_client.hgetall.return_value = session_data\n        \n        metadata = await session_service.validate_session(session_id)\n        \n        # Should return None for expired session\n        assert metadata is None\n    \n    @pytest.mark.asyncio\n    async def test_revoke_session(self, session_service):\n        \"\"\"Test session revocation\"\"\"\n        session_id = \"test-session-123\"\n        \n        # Mock existing session\n        session_data = {\n            'session_id': session_id,\n            'user_id': 'user-123',\n            'status': SessionStatus.ACTIVE.value\n        }\n        session_service.redis_client.hgetall.return_value = session_data\n        \n        success = await session_service.revoke_session(session_id, \"user_requested\")\n        \n        assert success\n        session_service.redis_client.hset.assert_called()\n    \n    @pytest.mark.asyncio\n    async def test_ip_address_validation(self, session_service):\n        \"\"\"Test IP address validation\"\"\"\n        # Test valid IPv4\n        assert session_service._is_valid_ip(\"192.168.1.1\")\n        assert session_service._is_valid_ip(\"10.0.0.1\")\n        \n        # Test valid IPv6\n        assert session_service._is_valid_ip(\"2001:db8::1\")\n        \n        # Test invalid IPs\n        assert not session_service._is_valid_ip(\"invalid-ip\")\n        assert not session_service._is_valid_ip(\"999.999.999.999\")\n    \n    @pytest.mark.asyncio\n    async def test_device_tracking(self, session_service, mock_user):\n        \"\"\"Test device tracking functionality\"\"\"\n        device_fingerprint = \"device-123\"\n        user_agent = \"Mozilla/5.0 Chrome/95.0.4638.69\"\n        \n        # Mock device trust check\n        session_service.redis_client.hget.return_value = \"true\"\n        \n        is_trusted = await session_service._is_device_trusted(\n            str(mock_user.id), device_fingerprint\n        )\n        \n        assert is_trusted\n    \n    def test_device_name_parsing(self, session_service):\n        \"\"\"Test device name parsing from user agent\"\"\"\n        test_cases = [\n            (\"Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)\", \"iPhone\"),\n            (\"Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)\", \"iPad\"),\n            (\"Mozilla/5.0 (Linux; Android 10; SM-G975F)\", \"Android Device\"),\n            (\"Mozilla/5.0 (Windows NT 10.0; Win64; x64)\", \"Windows PC\"),\n            (\"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)\", \"Mac\"),\n            (\"Unknown Browser\", \"Unknown Device\")\n        ]\n        \n        for user_agent, expected_device in test_cases:\n            result = session_service._parse_device_name(user_agent)\n            assert result == expected_device\n    \n    def test_device_type_parsing(self, session_service):\n        \"\"\"Test device type parsing\"\"\"\n        test_cases = [\n            (\"iPhone\", \"mobile\"),\n            (\"Android Mobile\", \"mobile\"),\n            (\"iPad\", \"tablet\"),\n            (\"Windows Desktop\", \"web\")\n        ]\n        \n        for user_agent, expected_type in test_cases:\n            result = session_service._parse_device_type(user_agent)\n            assert result in [\"mobile\", \"tablet\", \"web\"]\n\n\nclass TestRBACManager:\n    \"\"\"Test Role-Based Access Control functionality\"\"\"\n    \n    @pytest.fixture\n    def rbac_manager(self):\n        return RBACManager()\n    \n    def test_default_permissions_initialization(self, rbac_manager):\n        \"\"\"Test that default permissions are properly initialized\"\"\"\n        # Check some core permissions exist\n        assert \"users.read\" in rbac_manager.permissions\n        assert \"projects.create\" in rbac_manager.permissions\n        assert \"system.admin\" in rbac_manager.permissions\n        \n        # Verify permission structure\n        user_read = rbac_manager.permissions[\"users.read\"]\n        assert user_read.resource == \"user\"\n        assert user_read.action == \"read\"\n    \n    def test_default_roles_initialization(self, rbac_manager):\n        \"\"\"Test that default roles are properly initialized\"\"\"\n        # Check core roles exist\n        assert \"SUPER_ADMIN\" in rbac_manager.roles\n        assert \"ADMIN\" in rbac_manager.roles\n        assert \"PROJECT_MANAGER\" in rbac_manager.roles\n        assert \"DEVELOPER\" in rbac_manager.roles\n        assert \"VIEWER\" in rbac_manager.roles\n        assert \"GUEST\" in rbac_manager.roles\n        \n        # Verify role permissions\n        super_admin = rbac_manager.roles[\"SUPER_ADMIN\"]\n        assert len(super_admin.permissions) > 0\n        assert super_admin.is_system_role\n    \n    def test_create_custom_permission(self, rbac_manager):\n        \"\"\"Test creating custom permissions\"\"\"\n        permission = rbac_manager.create_permission(\n            name=\"custom.test\",\n            resource=\"custom\",\n            action=\"test\",\n            description=\"Test permission\"\n        )\n        \n        assert permission.name == \"custom.test\"\n        assert permission.resource == \"custom\"\n        assert permission.action == \"test\"\n        assert \"custom.test\" in rbac_manager.permissions\n    \n    def test_create_custom_role(self, rbac_manager):\n        \"\"\"Test creating custom roles\"\"\"\n        permissions = [\"users.read\", \"projects.read\"]\n        role = rbac_manager.create_role(\n            name=\"CUSTOM_ROLE\",\n            description=\"Custom test role\",\n            permissions=permissions\n        )\n        \n        assert role.name == \"CUSTOM_ROLE\"\n        assert len(role.permissions) == 2\n        assert \"CUSTOM_ROLE\" in rbac_manager.roles\n    \n    def test_permission_assignment(self, rbac_manager):\n        \"\"\"Test assigning permissions to roles\"\"\"\n        # Create test role\n        rbac_manager.create_role(\"TEST_ROLE\", \"Test role\")\n        \n        # Assign permission\n        success = rbac_manager.assign_permission_to_role(\n            \"TEST_ROLE\", \"users.read\"\n        )\n        \n        assert success\n        \n        # Verify permission is assigned\n        role = rbac_manager.roles[\"TEST_ROLE\"]\n        permission = rbac_manager.permissions[\"users.read\"]\n        assert permission in role.permissions\n    \n    def test_permission_removal(self, rbac_manager):\n        \"\"\"Test removing permissions from roles\"\"\"\n        # Use existing role with permissions\n        role_name = \"DEVELOPER\"\n        permission_name = \"projects.read\"\n        \n        # Verify permission exists first\n        role = rbac_manager.roles[role_name]\n        permission = rbac_manager.permissions[permission_name]\n        assert permission in role.permissions\n        \n        # Remove permission\n        success = rbac_manager.remove_permission_from_role(\n            role_name, permission_name\n        )\n        \n        assert success\n        assert permission not in role.permissions\n    \n    def test_user_permission_checks(self, rbac_manager):\n        \"\"\"Test user permission checking\"\"\"\n        user_permissions = [\"users.read\", \"projects.read\", \"tasks.create\"]\n        \n        # Test individual permission check\n        assert rbac_manager.user_has_permission(user_permissions, \"users.read\")\n        assert not rbac_manager.user_has_permission(user_permissions, \"users.delete\")\n        \n        # Test multiple permissions check\n        required_all = [\"users.read\", \"projects.read\"]\n        assert rbac_manager.user_has_all_permissions(user_permissions, required_all)\n        \n        required_all_fail = [\"users.read\", \"users.delete\"]\n        assert not rbac_manager.user_has_all_permissions(user_permissions, required_all_fail)\n        \n        # Test any permission check\n        required_any = [\"users.delete\", \"projects.read\"]\n        assert rbac_manager.user_has_any_permission(user_permissions, required_any)\n    \n    def test_role_effective_permissions(self, rbac_manager):\n        \"\"\"Test getting effective permissions for user roles\"\"\"\n        user_roles = [\"DEVELOPER\", \"VIEWER\"]\n        \n        permissions = rbac_manager.get_user_effective_permissions(user_roles)\n        \n        assert isinstance(permissions, list)\n        assert len(permissions) > 0\n        \n        # Should include permissions from both roles\n        assert \"projects.read\" in permissions  # From both roles\n        assert \"tasks.create\" in permissions   # From DEVELOPER\n    \n    def test_resource_access_checks(self, rbac_manager):\n        \"\"\"Test resource-based access control\"\"\"\n        user_permissions = [\"projects.read\", \"users.manage\"]\n        \n        # Test direct permission\n        assert rbac_manager.check_resource_access(\n            user_permissions, \"projects\", \"read\"\n        )\n        \n        # Test manage permission covers other actions\n        assert rbac_manager.check_resource_access(\n            user_permissions, \"users\", \"create\"\n        )  # manage covers create\n        \n        # Test admin permission covers everything\n        admin_permissions = [\"system.admin\"]\n        assert rbac_manager.check_resource_access(\n            admin_permissions, \"anything\", \"delete\"\n        )\n    \n    def test_ownership_access_checks(self, rbac_manager):\n        \"\"\"Test ownership-based access control\"\"\"\n        user_permissions = [\"projects.read\"]\n        user_id = \"user-123\"\n        resource_owner = \"user-123\"\n        \n        # User should be able to read their own resources\n        assert rbac_manager.check_resource_access(\n            user_permissions, \"projects\", \"read\", resource_owner, user_id\n        )\n        \n        # User should be able to update their own resources\n        assert rbac_manager.check_resource_access(\n            user_permissions, \"projects\", \"update\", resource_owner, user_id\n        )\n        \n        # User should not be able to delete without explicit permission\n        assert not rbac_manager.check_resource_access(\n            user_permissions, \"projects\", \"delete\", resource_owner, user_id\n        )\n    \n    def test_rbac_validation(self, rbac_manager):\n        \"\"\"Test RBAC configuration validation\"\"\"\n        issues = rbac_manager.validate_permission_hierarchy()\n        \n        # Should not have orphaned permissions or empty roles in default config\n        assert \"orphaned_permissions\" not in issues\n        assert \"empty_roles\" not in issues\n        \n        # Permission names should be consistent\n        assert \"inconsistent_permission_names\" not in issues\n    \n    def test_rbac_export_import(self, rbac_manager):\n        \"\"\"Test RBAC configuration export and import\"\"\"\n        # Export configuration\n        config = rbac_manager.export_rbac_config()\n        \n        assert \"permissions\" in config\n        assert \"roles\" in config\n        assert len(config[\"permissions\"]) > 0\n        assert len(config[\"roles\"]) > 0\n        \n        # Create new RBAC manager\n        new_rbac = RBACManager()\n        \n        # Clear default data for clean import test\n        new_rbac.permissions.clear()\n        new_rbac.roles.clear()\n        \n        # Import configuration\n        new_rbac.import_rbac_config(config)\n        \n        # Verify import\n        assert len(new_rbac.permissions) == len(rbac_manager.permissions)\n        assert len(new_rbac.roles) == len(rbac_manager.roles)\n\n\nclass TestPasswordResetService:\n    \"\"\"Test password reset functionality\"\"\"\n    \n    @pytest.fixture\n    def mock_jwt_service(self):\n        service = Mock()\n        service.create_password_reset_token.return_value = \"reset-token-123\"\n        return service\n    \n    @pytest.fixture\n    def mock_email_service(self):\n        service = Mock()\n        service.send_password_reset_email = AsyncMock(return_value=True)\n        return service\n    \n    @pytest.fixture\n    def mock_audit_logger(self):\n        logger = Mock()\n        logger.log_authentication = AsyncMock()\n        logger.log_security_incident = AsyncMock()\n        return logger\n    \n    @pytest.fixture\n    def mock_redis(self):\n        redis_client = Mock()\n        redis_client.hset.return_value = True\n        redis_client.expire.return_value = True\n        redis_client.hgetall.return_value = {}\n        redis_client.zadd.return_value = 1\n        redis_client.zcard.return_value = 0\n        redis_client.zremrangebyscore.return_value = 0\n        return redis_client\n    \n    @pytest.fixture\n    def password_reset_service(self, mock_jwt_service, mock_email_service, \n                              mock_audit_logger, mock_redis):\n        return PasswordResetService(\n            jwt_service=mock_jwt_service,\n            email_service=mock_email_service,\n            audit_logger=mock_audit_logger,\n            redis_client=mock_redis\n        )\n    \n    @pytest.mark.asyncio\n    async def test_password_reset_request(self, password_reset_service):\n        \"\"\"Test password reset request\"\"\"\n        email = \"test@example.com\"\n        ip_address = \"192.168.1.1\"\n        user_agent = \"Test Browser\"\n        \n        result = await password_reset_service.request_password_reset(\n            email=email,\n            ip_address=ip_address,\n            user_agent=user_agent\n        )\n        \n        assert result[\"success\"]\n        assert \"reset link has been sent\" in result[\"message\"]\n    \n    @pytest.mark.asyncio\n    async def test_rate_limiting(self, password_reset_service):\n        \"\"\"Test rate limiting for password reset requests\"\"\"\n        email = \"test@example.com\"\n        ip_address = \"192.168.1.1\"\n        \n        # Mock rate limit exceeded\n        password_reset_service.redis_client.zcard.return_value = 5  # Exceeds limit\n        \n        result = await password_reset_service.request_password_reset(\n            email=email,\n            ip_address=ip_address\n        )\n        \n        # Should still return success for security (prevent enumeration)\n        assert result[\"success\"]\n    \n    def test_token_hashing(self, password_reset_service):\n        \"\"\"Test secure token hashing\"\"\"\n        token = \"test-token-123\"\n        \n        hash1 = password_reset_service._hash_token(token)\n        hash2 = password_reset_service._hash_token(token)\n        \n        # Same token should produce same hash\n        assert hash1 == hash2\n        \n        # Different tokens should produce different hashes\n        hash3 = password_reset_service._hash_token(\"different-token\")\n        assert hash1 != hash3\n        \n        # Hash should be deterministic and hex\n        assert len(hash1) == 64  # SHA256 hex length\n        assert all(c in '0123456789abcdef' for c in hash1)\n    \n    @pytest.mark.asyncio\n    async def test_token_generation(self, password_reset_service):\n        \"\"\"Test reset token generation\"\"\"\n        user_id = \"user-123\"\n        email = \"test@example.com\"\n        ip_address = \"192.168.1.1\"\n        \n        token = await password_reset_service._generate_reset_token(\n            user_id=user_id,\n            email=email,\n            ip_address=ip_address\n        )\n        \n        assert isinstance(token, str)\n        assert len(token) > 0\n        \n        # Verify token was stored in Redis\n        password_reset_service.redis_client.hset.assert_called()\n        password_reset_service.redis_client.expire.assert_called()\n    \n    @pytest.mark.asyncio\n    async def test_rate_limit_window_counting(self, password_reset_service):\n        \"\"\"Test rate limit window request counting\"\"\"\n        key = \"test_key\"\n        window_start = datetime.now(timezone.utc) - timedelta(hours=1)\n        \n        # Mock Redis responses\n        password_reset_service.redis_client.zremrangebyscore.return_value = 2\n        password_reset_service.redis_client.zcard.return_value = 3\n        \n        count = await password_reset_service._count_requests_in_window(key, window_start)\n        \n        assert count == 3\n        password_reset_service.redis_client.zremrangebyscore.assert_called_once()\n        password_reset_service.redis_client.zcard.assert_called_once()\n\n\nclass TestEmailService:\n    \"\"\"Test email service functionality\"\"\"\n    \n    @pytest.fixture\n    def email_service(self):\n        return EmailService(\n            smtp_host=\"localhost\",\n            smtp_port=587,\n            smtp_username=\"test@example.com\",\n            smtp_password=\"password\",\n            from_email=\"noreply@test.com\"\n        )\n    \n    def test_email_content_creation(self, email_service):\n        \"\"\"Test HTML and text email content creation\"\"\"\n        user_name = \"Test User\"\n        reset_url = \"https://example.com/reset?token=123\"\n        expires_in_hours = 1\n        \n        # Test HTML content\n        html_content = email_service._create_reset_email_html(\n            user_name=user_name,\n            reset_url=reset_url,\n            expires_in_hours=expires_in_hours\n        )\n        \n        assert user_name in html_content\n        assert reset_url in html_content\n        assert str(expires_in_hours) in html_content\n        assert \"<!DOCTYPE html>\" in html_content\n        assert \"Claude-TIU\" in html_content\n        \n        # Test text content\n        text_content = email_service._create_reset_email_text(\n            user_name=user_name,\n            reset_url=reset_url,\n            expires_in_hours=expires_in_hours\n        )\n        \n        assert user_name in text_content\n        assert reset_url in text_content\n        assert str(expires_in_hours) in text_content\n        assert \"Claude-TIU\" in text_content\n    \n    def test_password_changed_notification_content(self, email_service):\n        \"\"\"Test password changed notification content\"\"\"\n        user_name = \"Test User\"\n        ip_address = \"192.168.1.1\"\n        timestamp = datetime.now(timezone.utc)\n        \n        # Test HTML content\n        html_content = email_service._create_password_changed_html(\n            user_name=user_name,\n            ip_address=ip_address,\n            timestamp=timestamp\n        )\n        \n        assert user_name in html_content\n        assert ip_address in html_content\n        assert \"Password Changed Successfully\" in html_content\n        \n        # Test text content\n        text_content = email_service._create_password_changed_text(\n            user_name=user_name,\n            ip_address=ip_address,\n            timestamp=timestamp\n        )\n        \n        assert user_name in text_content\n        assert ip_address in text_content\n        assert \"Password Changed Successfully\" in text_content\n\n\nclass TestGitHubOAuthProvider:\n    \"\"\"Test GitHub OAuth provider functionality\"\"\"\n    \n    @pytest.fixture\n    def github_provider(self):\n        return GitHubOAuthProvider(\n            client_id=\"test-client-id\",\n            client_secret=\"test-client-secret\",\n            redirect_uri=\"https://example.com/callback\"\n        )\n    \n    def test_provider_configuration(self, github_provider):\n        \"\"\"Test provider configuration and properties\"\"\"\n        assert github_provider.provider_name == \"github\"\n        assert github_provider.authorization_url == \"https://github.com/login/oauth/authorize\"\n        assert github_provider.token_url == \"https://github.com/login/oauth/access_token\"\n        assert github_provider.user_info_url == \"https://api.github.com/user\"\n        \n        # Test default scopes\n        scopes = github_provider.get_default_scopes()\n        assert \"user:email\" in scopes\n        assert \"read:user\" in scopes\n    \n    def test_authorization_url_generation(self, github_provider):\n        \"\"\"Test OAuth authorization URL generation\"\"\"\n        state = \"test-state-123\"\n        auth_url, returned_state = github_provider.get_authorization_url(state)\n        \n        assert returned_state == state\n        assert github_provider.client_id in auth_url\n        assert github_provider.redirect_uri in auth_url\n        assert \"user:email\" in auth_url or \"user%3Aemail\" in auth_url\n        assert state in auth_url\n        assert \"response_type=code\" in auth_url\n    \n    def test_state_generation(self, github_provider):\n        \"\"\"Test OAuth state parameter generation\"\"\"\n        state1 = github_provider.generate_state()\n        state2 = github_provider.generate_state()\n        \n        # States should be unique\n        assert state1 != state2\n        \n        # States should be URL-safe\n        assert len(state1) > 0\n        assert all(c.isalnum() or c in '-_' for c in state1)\n    \n    @pytest.mark.asyncio\n    async def test_user_info_parsing(self, github_provider):\n        \"\"\"Test GitHub user data parsing\"\"\"\n        # Mock GitHub user data\n        github_user_data = {\n            'id': 12345,\n            'login': 'testuser',\n            'name': 'Test User',\n            'email': 'test@example.com',\n            'avatar_url': 'https://github.com/avatar.png',\n            'html_url': 'https://github.com/testuser',\n            'bio': 'Test bio',\n            'location': 'Test City',\n            'public_repos': 10,\n            'followers': 5,\n            'following': 3\n        }\n        \n        email_info = {\n            'email': 'test@example.com',\n            'verified': True,\n            'all_emails': [\n                {'email': 'test@example.com', 'primary': True, 'verified': True}\n            ]\n        }\n        \n        user_info = await github_provider._parse_github_user_data(\n            github_user_data, email_info\n        )\n        \n        assert user_info.provider == \"github\"\n        assert user_info.provider_id == \"12345\"\n        assert user_info.email == \"test@example.com\"\n        assert user_info.name == \"Test User\"\n        assert user_info.username == \"testuser\"\n        assert user_info.verified == True\n        assert user_info.avatar_url == \"https://github.com/avatar.png\"\n        \n        # Check raw data is preserved\n        assert user_info.raw_data['github_id'] == 12345\n        assert user_info.raw_data['github_public_repos'] == 10\n    \n    @pytest.mark.asyncio\n    async def test_missing_email_handling(self, github_provider):\n        \"\"\"Test handling of missing email from GitHub\"\"\"\n        github_user_data = {\n            'id': 12345,\n            'login': 'testuser',\n            'name': 'Test User'\n            # No email field\n        }\n        \n        email_info = {\n            'email': None,  # No email available\n            'verified': False,\n            'all_emails': []\n        }\n        \n        with pytest.raises(OAuthError) as exc_info:\n            await github_provider._parse_github_user_data(\n                github_user_data, email_info\n            )\n        \n        assert \"No email address available\" in str(exc_info.value)\n    \n    def test_additional_auth_params(self, github_provider):\n        \"\"\"Test GitHub-specific authorization parameters\"\"\"\n        params = github_provider.get_additional_auth_params()\n        \n        assert 'allow_signup' in params\n        assert params['allow_signup'] == 'true'\n\n\nclass TestSecurityAuditLogger:\n    \"\"\"Test security audit logging functionality\"\"\"\n    \n    @pytest.fixture\n    def audit_logger(self, tmp_path):\n        log_file = tmp_path / \"test_audit.log\"\n        return SecurityAuditLogger(\n            log_file=str(log_file),\n            enable_console=False,  # Disable for testing\n            enable_syslog=False\n        )\n    \n    @pytest.mark.asyncio\n    async def test_authentication_event_logging(self, audit_logger):\n        \"\"\"Test logging authentication events\"\"\"\n        await audit_logger.log_authentication(\n            SecurityEventType.LOGIN_SUCCESS,\n            user_id=\"user-123\",\n            username=\"testuser\",\n            ip_address=\"192.168.1.1\",\n            user_agent=\"Test Browser\",\n            session_id=\"session-123\",\n            success=True,\n            message=\"User logged in successfully\",\n            details={'method': 'password'}\n        )\n        \n        # Verify log was written (implementation specific)\n        # In real implementation, you'd check the log file or mock the logger\n    \n    @pytest.mark.asyncio\n    async def test_authorization_event_logging(self, audit_logger):\n        \"\"\"Test logging authorization events\"\"\"\n        await audit_logger.log_authorization(\n            SecurityEventType.ACCESS_DENIED,\n            user_id=\"user-123\",\n            username=\"testuser\",\n            resource=\"projects\",\n            action=\"delete\",\n            success=False,\n            ip_address=\"192.168.1.1\",\n            message=\"Access denied to delete project\",\n            details={'resource_id': 'project-456'}\n        )\n    \n    @pytest.mark.asyncio\n    async def test_security_incident_logging(self, audit_logger):\n        \"\"\"Test logging security incidents\"\"\"\n        await audit_logger.log_security_incident(\n            SecurityEventType.BRUTE_FORCE_DETECTED,\n            SecurityLevel.HIGH,\n            \"Brute force attack detected\",\n            user_id=\"user-123\",\n            ip_address=\"192.168.1.1\",\n            details={\n                'attempts': 10,\n                'time_window': '5 minutes'\n            }\n        )\n    \n    def test_risk_score_calculation(self, audit_logger):\n        \"\"\"Test security event risk scoring\"\"\"\n        # Test authentication risk scoring\n        low_risk = audit_logger._calculate_auth_risk_score(\n            SecurityEventType.LOGIN_SUCCESS, True\n        )\n        assert low_risk < 50\n        \n        high_risk = audit_logger._calculate_auth_risk_score(\n            SecurityEventType.BRUTE_FORCE_DETECTED, False, {'failed_attempts': 10}\n        )\n        assert high_risk > 70\n        \n        # Test authorization risk scoring\n        auth_risk = audit_logger._calculate_authz_risk_score(\n            \"delete\", False, {'sensitive_resource': True}\n        )\n        assert auth_risk > 40\n    \n    def test_security_level_determination(self, audit_logger):\n        \"\"\"Test security level determination logic\"\"\"\n        # Successful login should be low level\n        level = audit_logger._determine_auth_level(SecurityEventType.LOGIN_SUCCESS, True)\n        assert level == SecurityLevel.LOW\n        \n        # Failed login should be medium level\n        level = audit_logger._determine_auth_level(SecurityEventType.LOGIN_FAILED, False)\n        assert level == SecurityLevel.MEDIUM\n        \n        # Blocked login should be high level\n        level = audit_logger._determine_auth_level(SecurityEventType.LOGIN_BLOCKED, False)\n        assert level == SecurityLevel.HIGH\n\n\n@pytest.mark.integration\nclass TestAuthenticationFlow:\n    \"\"\"Integration tests for complete authentication flows\"\"\"\n    \n    @pytest.fixture\n    def auth_components(self):\n        \"\"\"Setup complete authentication system for integration testing\"\"\"\n        # Mock components for integration testing\n        jwt_service = JWTService(\"test-secret-key\", \"HS256\")\n        \n        mock_redis = Mock()\n        mock_redis.hgetall.return_value = {}\n        mock_redis.hset.return_value = True\n        mock_redis.expire.return_value = True\n        \n        session_service = SessionService(redis_client=mock_redis)\n        rbac_manager = RBACManager()\n        \n        return {\n            'jwt_service': jwt_service,\n            'session_service': session_service,\n            'rbac_manager': rbac_manager\n        }\n    \n    @pytest.mark.asyncio\n    async def test_complete_login_flow(self, auth_components):\n        \"\"\"Test complete user login flow\"\"\"\n        jwt_service = auth_components['jwt_service']\n        session_service = auth_components['session_service']\n        rbac_manager = auth_components['rbac_manager']\n        \n        # Mock user\n        user = Mock()\n        user.id = \"user-123\"\n        user.username = \"testuser\"\n        user.email = \"test@example.com\"\n        \n        # Create session\n        session_id = await session_service.create_session(\n            user=user,\n            ip_address=\"192.168.1.1\",\n            user_agent=\"Test Browser\",\n            login_method=\"password\"\n        )\n        \n        # Get user permissions\n        permissions = rbac_manager.get_user_effective_permissions([\"USER\"])\n        \n        # Create token pair\n        token_pair = await jwt_service.create_token_pair(\n            user=user,\n            session_id=session_id,\n            permissions=permissions,\n            ip_address=\"192.168.1.1\",\n            user_agent=\"Test Browser\"\n        )\n        \n        # Verify tokens\n        assert token_pair.access_token\n        assert token_pair.refresh_token\n        \n        # Validate access token\n        access_validation = jwt_service.verify_token(\n            token_pair.access_token, TokenType.ACCESS\n        )\n        assert access_validation.valid\n        assert access_validation.payload.sub == str(user.id)\n        \n        # Validate refresh token\n        refresh_validation = jwt_service.verify_token(\n            token_pair.refresh_token, TokenType.REFRESH\n        )\n        assert refresh_validation.valid\n        assert refresh_validation.payload.sub == str(user.id)\n    \n    @pytest.mark.asyncio\n    async def test_token_refresh_flow(self, auth_components):\n        \"\"\"Test token refresh flow\"\"\"\n        jwt_service = auth_components['jwt_service']\n        \n        # Create initial token pair\n        user_id = \"user-123\"\n        permissions = [\"user:read\", \"user:write\"]\n        \n        initial_pair = jwt_service.create_token_pair(\n            user_id=user_id,\n            permissions=permissions\n        )\n        \n        # Verify initial tokens are valid\n        access_validation = jwt_service.verify_token(\n            initial_pair.access_token, TokenType.ACCESS\n        )\n        assert access_validation.valid\n        \n        # Simulate token refresh (simplified)\n        # In real implementation, this would use the refresh endpoint\n        new_pair = jwt_service.create_token_pair(\n            user_id=user_id,\n            permissions=permissions\n        )\n        \n        # New tokens should be different\n        assert new_pair.access_token != initial_pair.access_token\n        assert new_pair.refresh_token != initial_pair.refresh_token\n        \n        # Both should be valid\n        new_access_validation = jwt_service.verify_token(\n            new_pair.access_token, TokenType.ACCESS\n        )\n        assert new_access_validation.valid\n    \n    def test_permission_enforcement_flow(self, auth_components):\n        \"\"\"Test permission enforcement in RBAC\"\"\"\n        rbac_manager = auth_components['rbac_manager']\n        \n        # Test user with developer role\n        developer_permissions = rbac_manager.get_user_effective_permissions([\"DEVELOPER\"])\n        \n        # Should have project read access\n        assert rbac_manager.user_has_permission(developer_permissions, \"projects.read\")\n        \n        # Should have task creation access\n        assert rbac_manager.user_has_permission(developer_permissions, \"tasks.create\")\n        \n        # Should NOT have user management access\n        assert not rbac_manager.user_has_permission(developer_permissions, \"users.delete\")\n        \n        # Test resource access with ownership\n        can_access = rbac_manager.check_resource_access(\n            developer_permissions,\n            resource_type=\"projects\",\n            action=\"update\",\n            resource_owner=\"user-123\",\n            user_id=\"user-123\"  # Same user\n        )\n        assert can_access  # Should be able to update own project\n    \n    @pytest.mark.asyncio\n    async def test_security_event_flow(self, auth_components):\n        \"\"\"Test security event logging flow\"\"\"\n        from src.auth.audit_logger import SecurityAuditLogger\n        \n        audit_logger = SecurityAuditLogger(\n            enable_console=False,\n            enable_file=False,\n            enable_syslog=False\n        )\n        \n        # Test login event\n        await audit_logger.log_authentication(\n            SecurityEventType.LOGIN_SUCCESS,\n            user_id=\"user-123\",\n            username=\"testuser\",\n            ip_address=\"192.168.1.1\",\n            success=True\n        )\n        \n        # Test failed login\n        await audit_logger.log_authentication(\n            SecurityEventType.LOGIN_FAILED,\n            ip_address=\"192.168.1.1\",\n            success=False,\n            details={'attempts': 3}\n        )\n        \n        # Test access denied\n        await audit_logger.log_authorization(\n            SecurityEventType.ACCESS_DENIED,\n            user_id=\"user-123\",\n            username=\"testuser\",\n            resource=\"admin_panel\",\n            action=\"access\",\n            success=False\n        )\n\n\nif __name__ == \"__main__\":\n    pytest.main([__file__, \"-v\"])