"""
Security Configuration for Claude-TIU Authentication System

Centralizes all security-related configuration settings and provides
secure defaults with environment variable overrides.
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import timedelta
import secrets


@dataclass
class JWTConfig:
    """JWT token configuration"""
    secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32)))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 30
    issuer: str = "claude-tiu"
    audience: str = "claude-tiu-api"
    
    # Token type expiration settings
    email_verification_expire_hours: int = 24
    password_reset_expire_hours: int = 1
    two_factor_expire_minutes: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if len(self.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")


@dataclass
class SessionConfig:
    """Session management configuration"""
    session_ttl_hours: int = 24
    max_sessions_per_user: int = 10
    suspicious_threshold: int = 3
    device_trust_days: int = 30
    
    # Redis configuration
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_SESSION_DB", "1"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # Session validation
    strict_ip_validation: bool = False  # Set to True for stricter security
    require_device_fingerprint: bool = False


@dataclass
class PasswordConfig:
    """Password security configuration"""
    min_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    special_chars: str = "!@#$%^&*(),.?\":{}|<>"
    
    # Password policy
    prevent_reuse_count: int = 5  # Prevent reusing last N passwords
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # Reset configuration
    reset_token_expire_hours: int = 1
    reset_max_attempts: int = 3
    reset_rate_limit_per_email: int = 3
    reset_rate_limit_window_hours: int = 1


@dataclass
class EmailConfig:
    """Email service configuration"""
    smtp_host: str = os.getenv("SMTP_HOST", "localhost")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: Optional[str] = os.getenv("SMTP_USERNAME")
    smtp_password: Optional[str] = os.getenv("SMTP_PASSWORD")
    smtp_use_tls: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    smtp_use_ssl: bool = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
    
    from_email: str = os.getenv("FROM_EMAIL", "noreply@claude-tiu.com")
    from_name: str = os.getenv("FROM_NAME", "Claude-TIU")
    
    # Email content settings
    company_name: str = "Claude-TIU"
    support_email: str = os.getenv("SUPPORT_EMAIL", "support@claude-tiu.com")
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")


@dataclass
class OAuthConfig:
    """OAuth provider configuration"""
    # GitHub OAuth
    github_client_id: Optional[str] = os.getenv("GITHUB_CLIENT_ID")
    github_client_secret: Optional[str] = os.getenv("GITHUB_CLIENT_SECRET")
    github_redirect_uri: Optional[str] = os.getenv("GITHUB_REDIRECT_URI")
    
    # Google OAuth
    google_client_id: Optional[str] = os.getenv("GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET")
    google_redirect_uri: Optional[str] = os.getenv("GOOGLE_REDIRECT_URI")
    
    # OAuth settings
    oauth_state_expire_minutes: int = 10
    oauth_timeout_seconds: int = 30


@dataclass
class RBACConfig:
    """Role-Based Access Control configuration"""
    # Default roles that should always exist
    default_roles: List[str] = field(default_factory=lambda: [
        "SUPER_ADMIN", "ADMIN", "PROJECT_MANAGER", "DEVELOPER", "VIEWER", "GUEST"
    ])
    
    # Role assignment settings
    allow_self_role_change: bool = False
    require_admin_for_role_assignment: bool = True
    
    # Permission caching
    permission_cache_ttl_seconds: int = 300  # 5 minutes


@dataclass
class AuditConfig:
    """Audit logging configuration"""
    # Logging levels
    log_level: str = os.getenv("AUDIT_LOG_LEVEL", "INFO")
    
    # File logging
    log_file: Optional[str] = os.getenv("AUDIT_LOG_FILE", "/var/log/claude-tiu/audit.log")
    max_file_size_mb: int = int(os.getenv("AUDIT_MAX_FILE_SIZE_MB", "100"))
    backup_count: int = int(os.getenv("AUDIT_BACKUP_COUNT", "10"))
    
    # Console logging
    enable_console_logging: bool = os.getenv("AUDIT_CONSOLE", "false").lower() == "true"
    
    # Syslog
    enable_syslog: bool = os.getenv("AUDIT_SYSLOG", "false").lower() == "true"
    syslog_address: str = os.getenv("SYSLOG_ADDRESS", "/dev/log")
    
    # Alerting
    alert_webhook: Optional[str] = os.getenv("SECURITY_ALERT_WEBHOOK")
    alert_high_risk_threshold: int = 70
    alert_critical_events: List[str] = field(default_factory=lambda: [
        "BRUTE_FORCE_DETECTED", "ACCOUNT_TAKEOVER_ATTEMPT", "PRIVILEGE_ESCALATION"
    ])
    
    # Retention
    retention_days: int = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))
    
    # Redis for audit events
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_AUDIT_DB", "2"))


@dataclass
class TwoFactorConfig:
    """Two-Factor Authentication configuration"""
    enabled: bool = os.getenv("TWO_FACTOR_ENABLED", "false").lower() == "true"
    
    # TOTP settings
    totp_issuer: str = "Claude-TIU"
    totp_period: int = 30  # seconds
    totp_digits: int = 6
    
    # Backup codes
    backup_codes_count: int = 10
    backup_code_length: int = 8
    
    # Recovery settings
    recovery_grace_period_hours: int = 24


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    # Authentication endpoints
    login_requests_per_minute: int = 10
    login_burst_size: int = 5
    
    register_requests_per_hour: int = 3
    password_reset_requests_per_hour: int = 3
    
    # API rate limits
    authenticated_requests_per_minute: int = 1000
    anonymous_requests_per_minute: int = 100
    admin_requests_per_minute: int = 5000
    
    # IP-based limits
    max_requests_per_ip_per_hour: int = 10000
    
    # Redis configuration for rate limiting
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_RATELIMIT_DB", "3"))


@dataclass
class SecurityHeaders:
    """Security headers configuration"""
    enable_hsts: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True
    
    enable_csp: bool = True
    csp_policy: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https:; "
        "connect-src 'self' https:; "
        "frame-ancestors 'none';"
    )
    
    enable_xss_protection: bool = True
    enable_content_type_nosniff: bool = True
    enable_frame_options: bool = True
    frame_options: str = "DENY"
    
    enable_referrer_policy: bool = True
    referrer_policy: str = "strict-origin-when-cross-origin"


@dataclass
class SecurityConfig:
    """Comprehensive security configuration"""
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Component configurations
    jwt: JWTConfig = field(default_factory=JWTConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    passwords: PasswordConfig = field(default_factory=PasswordConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    oauth: OAuthConfig = field(default_factory=OAuthConfig)
    rbac: RBACConfig = field(default_factory=RBACConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    two_factor: TwoFactorConfig = field(default_factory=TwoFactorConfig)
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    headers: SecurityHeaders = field(default_factory=SecurityHeaders)
    
    # Global security settings
    allowed_hosts: List[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "claude-tiu.com", "*.claude-tiu.com"
    ])
    
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000", "https://claude-tiu.com"
    ])
    
    trusted_proxies: List[str] = field(default_factory=lambda: [
        "127.0.0.1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"
    ])
    
    # Encryption
    encryption_key: str = field(default_factory=lambda: os.getenv("ENCRYPTION_KEY", secrets.token_urlsafe(32)))
    
    def __post_init__(self):
        """Post-initialization validation and adjustments"""
        # Adjust settings based on environment
        if self.environment == "production":
            self._apply_production_security()
        elif self.environment == "development":
            self._apply_development_settings()
        
        # Validate critical settings
        self._validate_critical_settings()
    
    def _apply_production_security(self):
        """Apply production security hardening"""
        # Stricter session security
        self.sessions.strict_ip_validation = True
        self.sessions.require_device_fingerprint = True
        
        # Shorter token expiration
        self.jwt.access_token_expire_minutes = 15
        self.jwt.refresh_token_expire_days = 7
        
        # Stricter password policy
        self.passwords.max_failed_attempts = 3
        self.passwords.lockout_duration_minutes = 60
        
        # Enable all security headers
        self.headers.enable_hsts = True
        self.headers.enable_csp = True
        
        # Stricter rate limits
        self.rate_limits.login_requests_per_minute = 5
        self.rate_limits.anonymous_requests_per_minute = 50
        
        # Disable debug logging
        self.debug = False
    
    def _apply_development_settings(self):
        """Apply development-friendly settings"""
        # Longer token expiration for development
        self.jwt.access_token_expire_minutes = 60
        self.jwt.refresh_token_expire_days = 30
        
        # More lenient rate limits
        self.rate_limits.login_requests_per_minute = 20
        self.rate_limits.anonymous_requests_per_minute = 200
        
        # Enable console logging
        self.audit.enable_console_logging = True
        
        # Relaxed CORS for development
        self.cors_origins.extend([
            "http://localhost:3000", "http://localhost:8000",
            "http://127.0.0.1:3000", "http://127.0.0.1:8000"
        ])
    
    def _validate_critical_settings(self):
        """Validate critical security settings"""
        # Ensure encryption keys are strong enough
        if len(self.jwt.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")
        
        if len(self.encryption_key) < 32:
            raise ValueError("Encryption key must be at least 32 characters")
        
        # Validate email configuration for production
        if self.environment == "production":
            if not self.email.smtp_host or self.email.smtp_host == "localhost":
                raise ValueError("Production environment requires proper SMTP configuration")
        
        # Validate OAuth configuration if enabled
        if self.oauth.github_client_id and not self.oauth.github_client_secret:
            raise ValueError("GitHub OAuth requires both client ID and secret")
        
        if self.oauth.google_client_id and not self.oauth.google_client_secret:
            raise ValueError("Google OAuth requires both client ID and secret")
    
    def get_redis_url(self, db_type: str = "session") -> str:
        """Get Redis URL for specific database type"""
        db_mapping = {
            "session": self.sessions.redis_db,
            "audit": self.audit.redis_db,
            "ratelimit": self.rate_limits.redis_db
        }
        
        db = db_mapping.get(db_type, 0)
        password_part = f":{self.sessions.redis_password}@" if self.sessions.redis_password else ""
        
        return f"redis://{password_part}{self.sessions.redis_host}:{self.sessions.redis_port}/{db}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for serialization)"""
        from dataclasses import asdict
        return asdict(self)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security configuration summary for monitoring"""
        return {
            "environment": self.environment,
            "jwt_expiration_minutes": self.jwt.access_token_expire_minutes,
            "session_ttl_hours": self.sessions.session_ttl_hours,
            "max_failed_attempts": self.passwords.max_failed_attempts,
            "two_factor_enabled": self.two_factor.enabled,
            "strict_ip_validation": self.sessions.strict_ip_validation,
            "oauth_providers": [
                provider for provider in ["github", "google"]
                if getattr(self.oauth, f"{provider}_client_id") is not None
            ],
            "security_headers_enabled": {
                "hsts": self.headers.enable_hsts,
                "csp": self.headers.enable_csp,
                "xss_protection": self.headers.enable_xss_protection
            },
            "audit_logging": {
                "file_enabled": bool(self.audit.log_file),
                "console_enabled": self.audit.enable_console_logging,
                "syslog_enabled": self.audit.enable_syslog,
                "retention_days": self.audit.retention_days
            }
        }


# Global security configuration instance
_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get global security configuration instance"""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


def init_security_config(**kwargs) -> SecurityConfig:
    """Initialize security configuration with custom settings"""
    global _security_config
    _security_config = SecurityConfig(**kwargs)
    return _security_config


def validate_security_environment():
    """Validate security configuration for current environment"""
    config = get_security_config()
    
    issues = []
    warnings = []
    
    # Production-specific validations
    if config.environment == "production":
        if config.debug:
            issues.append("Debug mode should be disabled in production")
        
        if not config.headers.enable_hsts:
            warnings.append("HSTS should be enabled in production")
        
        if config.jwt.access_token_expire_minutes > 60:
            warnings.append("Access token expiration is quite long for production")
        
        if not config.audit.log_file:
            issues.append("Audit logging to file should be enabled in production")
    
    # General security validations
    if len(config.jwt.secret_key) < 32:
        issues.append("JWT secret key is too short (minimum 32 characters)")
    
    if config.passwords.max_failed_attempts > 10:
        warnings.append("Maximum failed login attempts seems high")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "config_summary": config.get_security_summary()
    }


# Example usage and configuration templates
def get_development_config() -> SecurityConfig:
    """Get development-optimized security configuration"""
    return SecurityConfig(
        environment="development",
        debug=True
    )


def get_production_config() -> SecurityConfig:
    """Get production-hardened security configuration"""
    return SecurityConfig(
        environment="production",
        debug=False
    )


def get_testing_config() -> SecurityConfig:
    """Get testing-optimized security configuration"""
    return SecurityConfig(
        environment="testing",
        debug=True,
        jwt=JWTConfig(
            access_token_expire_minutes=5,  # Short for testing
            secret_key="test-secret-key-for-testing-only"
        ),
        audit=AuditConfig(
            enable_console_logging=False,
            log_file=None
        )
    )


if __name__ == "__main__":
    # Example: Validate current environment configuration
    validation_result = validate_security_environment()
    
    if validation_result["valid"]:
        print("✅ Security configuration is valid")
    else:
        print("❌ Security configuration has issues:")
        for issue in validation_result["issues"]:
            print(f"  - {issue}")
    
    if validation_result["warnings"]:
        print("⚠️  Security configuration warnings:")
        for warning in validation_result["warnings"]:
            print(f"  - {warning}")
    
    print("\n📊 Security Configuration Summary:")
    summary = validation_result["config_summary"]
    for key, value in summary.items():
        print(f"  {key}: {value}")