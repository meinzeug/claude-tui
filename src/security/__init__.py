"""
Security module for claude-tui.

This module provides comprehensive security features including:
- Input validation and sanitization
- Secure subprocess execution
- API key management with encryption
- Rate limiting and DDoS protection
- Code sandbox execution
- Security auditing and monitoring
- Command injection prevention
"""

from .input_validator import (
    SecurityInputValidator,
    ValidationResult,
    SecurityThreat,
    ThreatLevel,
    validate_user_input
)

from .secure_subprocess import (
    SecureSubprocessManager,
    CommandPolicy,
    ExecutionMode,
    ExecutionResult,
    ResourceLimits,
    create_secure_subprocess_manager,
    execute_safe_command
)

from .api_key_manager import (
    APIKeyManager,
    KeyType,
    EncryptionLevel,
    APIKeyMetadata,
    create_api_key_manager
)

from .rate_limiter import (
    SmartRateLimiter,
    RateLimit,
    ActionType,
    AttackType,
    CircuitBreaker,
    create_rate_limiter
)

from .code_sandbox import (
    SecureCodeSandbox,
    SandboxMode,
    SecurityLevel,
    CodeAnalysisResult,
    CodeAnalyzer,
    create_secure_sandbox,
    analyze_code_safety
)

__all__ = [
    # Input validation
    'SecurityInputValidator',
    'ValidationResult',
    'SecurityThreat',
    'ThreatLevel',
    'validate_user_input',
    
    # Secure subprocess
    'SecureSubprocessManager',
    'CommandPolicy',
    'ExecutionMode',
    'ExecutionResult',
    'ResourceLimits',
    'create_secure_subprocess_manager',
    'execute_safe_command',
    
    # API key management
    'APIKeyManager',
    'KeyType',
    'EncryptionLevel',
    'APIKeyMetadata',
    'create_api_key_manager',
    
    # Rate limiting
    'SmartRateLimiter',
    'RateLimit',
    'ActionType',
    'AttackType',
    'CircuitBreaker',
    'create_rate_limiter',
    
    # Code sandbox
    'SecureCodeSandbox',
    'SandboxMode',
    'SecurityLevel',
    'CodeAnalysisResult',
    'CodeAnalyzer',
    'create_secure_sandbox',
    'analyze_code_safety',
]

# Security configuration
DEFAULT_SECURITY_CONFIG = {
    'input_validation': {
        'max_prompt_length': 50000,
        'threat_detection_enabled': True,
        'auto_sanitization': True
    },
    'subprocess': {
        'execution_mode': 'restricted',
        'memory_limit_mb': 256,
        'timeout_seconds': 30
    },
    'api_keys': {
        'encryption_level': 'enhanced',
        'auto_rotation_enabled': False,
        'backup_enabled': True
    },
    'rate_limiting': {
        'ddos_protection': True,
        'adaptive_limiting': True,
        'max_requests_per_minute': 60
    },
    'code_sandbox': {
        'security_level': 'high',
        'sandbox_mode': 'docker',
        'max_execution_time': 30
    }
}

def get_security_version() -> str:
    """Get security module version."""
    return "1.0.0"

def get_security_info() -> dict:
    """Get comprehensive security module information."""
    return {
        'version': get_security_version(),
        'components': {
            'input_validator': 'Multi-layer input validation and threat detection',
            'secure_subprocess': 'Hardened subprocess execution with resource limits',
            'api_key_manager': 'Enterprise-grade encrypted API key management',
            'rate_limiter': 'Advanced rate limiting with DDoS protection',
            'code_sandbox': 'Secure code execution environment'
        },
        'security_features': [
            'Command injection prevention',
            'Path traversal protection',
            'SQL injection detection',
            'XSS prevention',
            'API key pattern detection',
            'Resource limit enforcement',
            'Network isolation',
            'File system restrictions',
            'Real-time monitoring',
            'Audit logging'
        ],
        'compliance': [
            'OWASP Top 10 protection',
            'Security by design',
            'Zero trust architecture',
            'Defense in depth'
        ]
    }