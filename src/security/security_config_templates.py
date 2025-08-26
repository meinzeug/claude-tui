"""
Security Configuration Templates
===============================

Production-ready security configuration templates for different deployment scenarios.
"""

import os
import secrets
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class SecurityTemplate:
    """Security configuration template."""
    name: str
    description: str
    environment: DeploymentEnvironment
    config: Dict[str, Any]
    required_env_vars: List[str]
    optional_env_vars: List[str]
    security_notes: List[str]


class SecurityConfigTemplates:
    """Production-ready security configuration templates."""
    
    @staticmethod
    def get_production_template() -> SecurityTemplate:
        """High-security production configuration."""
        return SecurityTemplate(
            name="Production High Security",
            description="Maximum security configuration for production deployment",
            environment=DeploymentEnvironment.PRODUCTION,
            config={
                # Authentication & JWT
                "JWT_SECRET_KEY": "REQUIRED_ENV_VAR",
                "JWT_ALGORITHM": "HS256",
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 15,
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": 7,
                "JWT_REQUIRE_AUDIENCE": True,
                "JWT_REQUIRE_ISSUER": True,
                
                # Encryption
                "ENCRYPTION_KEY": "REQUIRED_ENV_VAR",
                "SECRET_KEY": "REQUIRED_ENV_VAR",
                
                # Password Policy
                "PASSWORD_MIN_LENGTH": 12,
                "PASSWORD_REQUIRE_UPPERCASE": True,
                "PASSWORD_REQUIRE_LOWERCASE": True,
                "PASSWORD_REQUIRE_NUMBERS": True,
                "PASSWORD_REQUIRE_SPECIAL": True,
                "PASSWORD_MAX_FAILED_ATTEMPTS": 3,
                "PASSWORD_LOCKOUT_DURATION": 3600,  # 1 hour
                
                # Session Security
                "SESSION_TIMEOUT_MINUTES": 30,
                "SESSION_REGENERATE_INTERVAL": 300,  # 5 minutes
                "MAX_CONCURRENT_SESSIONS": 3,
                "STRICT_IP_VALIDATION": True,
                "REQUIRE_DEVICE_FINGERPRINT": True,
                
                # Rate Limiting
                "LOGIN_RATE_LIMIT_PER_MINUTE": 3,
                "API_RATE_LIMIT_PER_HOUR": 5000,
                "ANONYMOUS_RATE_LIMIT_PER_HOUR": 100,
                "ENABLE_RATE_LIMITING": True,
                
                # Security Features
                "CSRF_PROTECTION_ENABLED": True,
                "CSRF_TOKEN_EXPIRE_MINUTES": 30,
                "XSS_PROTECTION_ENABLED": True,
                "SQL_INJECTION_PREVENTION_ENABLED": True,
                
                # Security Headers
                "SECURITY_HEADERS_ENABLED": True,
                "HSTS_MAX_AGE": 31536000,  # 1 year
                "HSTS_INCLUDE_SUBDOMAINS": True,
                "HSTS_PRELOAD": True,
                "CSP_ENABLED": True,
                "CSP_POLICY": (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: https:; "
                    "font-src 'self' https:; "
                    "connect-src 'self' https:; "
                    "frame-ancestors 'none'; "
                    "base-uri 'self'; "
                    "form-action 'self'"
                ),
                "X_FRAME_OPTIONS": "DENY",
                "X_CONTENT_TYPE_OPTIONS": "nosniff",
                "REFERRER_POLICY": "strict-origin-when-cross-origin",
                
                # Audit & Monitoring
                "AUDIT_LOGGING_ENABLED": True,
                "AUDIT_LOG_LEVEL": "INFO",
                "SECURITY_MONITORING_ENABLED": True,
                "SUSPICIOUS_ACTIVITY_ALERTS": True,
                "REAL_TIME_THREAT_DETECTION": True,
                
                # Database Security
                "DB_CONNECTION_ENCRYPTION": True,
                "DB_QUERY_LOGGING": True,
                "DB_CONNECTION_POOLING": True,
                
                # TLS/SSL
                "FORCE_HTTPS": True,
                "TLS_VERSION": "1.3",
                "CERT_VALIDATION": True,
                
                # Environment
                "DEBUG": False,
                "ENVIRONMENT": "production",
                "LOG_SENSITIVE_DATA": False,
            },
            required_env_vars=[
                "JWT_SECRET_KEY",
                "ENCRYPTION_KEY", 
                "SECRET_KEY",
                "DATABASE_URL",
                "REDIS_URL",
                "SECURITY_ALERT_WEBHOOK"
            ],
            optional_env_vars=[
                "SMTP_HOST",
                "SMTP_USERNAME", 
                "SMTP_PASSWORD",
                "OAUTH_GITHUB_CLIENT_ID",
                "OAUTH_GITHUB_CLIENT_SECRET",
                "OAUTH_GOOGLE_CLIENT_ID",
                "OAUTH_GOOGLE_CLIENT_SECRET"
            ],
            security_notes=[
                "All secret keys must be cryptographically secure (64+ characters)",
                "Database connections must use TLS encryption",
                "Redis must be configured with password authentication",
                "All external API calls must use HTTPS",
                "Security monitoring and alerting must be active",
                "Regular security updates and patches required",
                "Backup encryption keys securely (separate from application)",
                "Implement proper key rotation procedures"
            ]
        )
    
    @staticmethod
    def get_staging_template() -> SecurityTemplate:
        """Balanced security configuration for staging environment."""
        return SecurityTemplate(
            name="Staging Environment",
            description="Balanced security configuration for staging/testing",
            environment=DeploymentEnvironment.STAGING,
            config={
                # Authentication & JWT
                "JWT_SECRET_KEY": "REQUIRED_ENV_VAR",
                "JWT_ALGORITHM": "HS256",
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 30,
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": 14,
                "JWT_REQUIRE_AUDIENCE": True,
                "JWT_REQUIRE_ISSUER": True,
                
                # Encryption
                "ENCRYPTION_KEY": "REQUIRED_ENV_VAR",
                "SECRET_KEY": "REQUIRED_ENV_VAR",
                
                # Password Policy
                "PASSWORD_MIN_LENGTH": 10,
                "PASSWORD_REQUIRE_UPPERCASE": True,
                "PASSWORD_REQUIRE_LOWERCASE": True,
                "PASSWORD_REQUIRE_NUMBERS": True,
                "PASSWORD_REQUIRE_SPECIAL": True,
                "PASSWORD_MAX_FAILED_ATTEMPTS": 5,
                "PASSWORD_LOCKOUT_DURATION": 1800,  # 30 minutes
                
                # Session Security
                "SESSION_TIMEOUT_MINUTES": 60,
                "SESSION_REGENERATE_INTERVAL": 600,  # 10 minutes
                "MAX_CONCURRENT_SESSIONS": 5,
                "STRICT_IP_VALIDATION": False,
                "REQUIRE_DEVICE_FINGERPRINT": False,
                
                # Rate Limiting
                "LOGIN_RATE_LIMIT_PER_MINUTE": 10,
                "API_RATE_LIMIT_PER_HOUR": 10000,
                "ANONYMOUS_RATE_LIMIT_PER_HOUR": 500,
                "ENABLE_RATE_LIMITING": True,
                
                # Security Features
                "CSRF_PROTECTION_ENABLED": True,
                "CSRF_TOKEN_EXPIRE_MINUTES": 60,
                "XSS_PROTECTION_ENABLED": True,
                "SQL_INJECTION_PREVENTION_ENABLED": True,
                
                # Security Headers
                "SECURITY_HEADERS_ENABLED": True,
                "HSTS_MAX_AGE": 86400,  # 1 day
                "CSP_ENABLED": True,
                
                # Audit & Monitoring
                "AUDIT_LOGGING_ENABLED": True,
                "AUDIT_LOG_LEVEL": "DEBUG",
                "SECURITY_MONITORING_ENABLED": True,
                
                # Environment
                "DEBUG": False,
                "ENVIRONMENT": "staging",
            },
            required_env_vars=[
                "JWT_SECRET_KEY",
                "ENCRYPTION_KEY",
                "SECRET_KEY",
                "DATABASE_URL"
            ],
            optional_env_vars=[
                "REDIS_URL",
                "SECURITY_ALERT_WEBHOOK"
            ],
            security_notes=[
                "Use production-like security settings",
                "Enable detailed logging for testing",
                "Test all security features thoroughly",
                "Validate security configurations before production"
            ]
        )
    
    @staticmethod
    def get_development_template() -> SecurityTemplate:
        """Development-friendly security configuration."""
        return SecurityTemplate(
            name="Development Environment",
            description="Development-friendly security configuration with debugging enabled",
            environment=DeploymentEnvironment.DEVELOPMENT,
            config={
                # Authentication & JWT
                "JWT_SECRET_KEY": secrets.token_urlsafe(64),  # Auto-generated for dev
                "JWT_ALGORITHM": "HS256",
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 60,
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": 30,
                
                # Encryption
                "ENCRYPTION_KEY": secrets.token_urlsafe(32),  # Auto-generated for dev
                "SECRET_KEY": secrets.token_urlsafe(32),  # Auto-generated for dev
                
                # Password Policy (Relaxed for development)
                "PASSWORD_MIN_LENGTH": 8,
                "PASSWORD_REQUIRE_UPPERCASE": True,
                "PASSWORD_REQUIRE_LOWERCASE": True,
                "PASSWORD_REQUIRE_NUMBERS": False,
                "PASSWORD_REQUIRE_SPECIAL": False,
                "PASSWORD_MAX_FAILED_ATTEMPTS": 10,
                "PASSWORD_LOCKOUT_DURATION": 300,  # 5 minutes
                
                # Session Security
                "SESSION_TIMEOUT_MINUTES": 120,
                "SESSION_REGENERATE_INTERVAL": 1800,  # 30 minutes
                "MAX_CONCURRENT_SESSIONS": 10,
                "STRICT_IP_VALIDATION": False,
                
                # Rate Limiting (Relaxed)
                "LOGIN_RATE_LIMIT_PER_MINUTE": 50,
                "API_RATE_LIMIT_PER_HOUR": 50000,
                "ANONYMOUS_RATE_LIMIT_PER_HOUR": 5000,
                "ENABLE_RATE_LIMITING": False,  # Disabled for development
                
                # Security Features
                "CSRF_PROTECTION_ENABLED": True,
                "XSS_PROTECTION_ENABLED": True,
                "SQL_INJECTION_PREVENTION_ENABLED": True,
                
                # Security Headers (Relaxed)
                "SECURITY_HEADERS_ENABLED": True,
                "CSP_ENABLED": False,  # Often conflicts with dev tools
                
                # Audit & Monitoring
                "AUDIT_LOGGING_ENABLED": True,
                "AUDIT_LOG_LEVEL": "DEBUG",
                "SECURITY_MONITORING_ENABLED": False,
                
                # Environment
                "DEBUG": True,
                "ENVIRONMENT": "development",
                "LOG_SENSITIVE_DATA": False,  # Still protect sensitive data
            },
            required_env_vars=[],  # All auto-generated for development
            optional_env_vars=[
                "DATABASE_URL",
                "REDIS_URL"
            ],
            security_notes=[
                "Auto-generated keys for convenience - NOT for production",
                "Relaxed security settings for development ease",
                "Still maintains core security protections",
                "Remember to use proper environment variables in production"
            ]
        )
    
    @staticmethod
    def get_testing_template() -> SecurityTemplate:
        """Testing environment security configuration."""
        return SecurityTemplate(
            name="Testing Environment", 
            description="Security configuration optimized for automated testing",
            environment=DeploymentEnvironment.TESTING,
            config={
                # Authentication & JWT (Short expiration for testing)
                "JWT_SECRET_KEY": "test-secret-key-for-testing-only-do-not-use-in-production",
                "JWT_ALGORITHM": "HS256",
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 5,
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": 1,
                
                # Encryption (Fixed keys for reproducible tests)
                "ENCRYPTION_KEY": "test-encryption-key-for-testing-only",
                "SECRET_KEY": "test-secret-key-for-testing-only",
                
                # Password Policy (Minimal for testing)
                "PASSWORD_MIN_LENGTH": 6,
                "PASSWORD_REQUIRE_UPPERCASE": False,
                "PASSWORD_REQUIRE_LOWERCASE": False,
                "PASSWORD_REQUIRE_NUMBERS": False,
                "PASSWORD_REQUIRE_SPECIAL": False,
                "PASSWORD_MAX_FAILED_ATTEMPTS": 100,
                "PASSWORD_LOCKOUT_DURATION": 1,
                
                # Session Security (Minimal)
                "SESSION_TIMEOUT_MINUTES": 60,
                "MAX_CONCURRENT_SESSIONS": 100,
                "STRICT_IP_VALIDATION": False,
                
                # Rate Limiting (Disabled for testing)
                "ENABLE_RATE_LIMITING": False,
                "LOGIN_RATE_LIMIT_PER_MINUTE": 1000,
                
                # Security Features (Enabled for testing security features)
                "CSRF_PROTECTION_ENABLED": True,
                "XSS_PROTECTION_ENABLED": True,
                "SQL_INJECTION_PREVENTION_ENABLED": True,
                
                # Audit & Monitoring (Disabled for performance)
                "AUDIT_LOGGING_ENABLED": False,
                "SECURITY_MONITORING_ENABLED": False,
                
                # Environment
                "DEBUG": True,
                "ENVIRONMENT": "testing",
                "LOG_SENSITIVE_DATA": False,
            },
            required_env_vars=[],
            optional_env_vars=[],
            security_notes=[
                "ONLY for testing - never use in production",
                "Fixed keys ensure reproducible test results",
                "Security features enabled to test security code paths",
                "Relaxed limits for test performance"
            ]
        )


class SecurityConfigValidator:
    """Validates security configurations against best practices."""
    
    @staticmethod
    def validate_production_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate production security configuration."""
        issues = []
        warnings = []
        recommendations = []
        
        # Check JWT configuration
        jwt_secret = config.get("JWT_SECRET_KEY", "")
        if len(str(jwt_secret)) < 64:
            issues.append("JWT_SECRET_KEY must be at least 64 characters long")
        
        access_token_expire = config.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 0)
        if access_token_expire > 30:
            warnings.append("JWT access token expiration should be 30 minutes or less for production")
        
        # Check password policy
        min_length = config.get("PASSWORD_MIN_LENGTH", 0)
        if min_length < 12:
            issues.append("PASSWORD_MIN_LENGTH should be at least 12 characters for production")
        
        max_attempts = config.get("PASSWORD_MAX_FAILED_ATTEMPTS", 100)
        if max_attempts > 5:
            warnings.append("PASSWORD_MAX_FAILED_ATTEMPTS should be 5 or less for production")
        
        # Check security features
        required_features = [
            "CSRF_PROTECTION_ENABLED",
            "XSS_PROTECTION_ENABLED", 
            "SQL_INJECTION_PREVENTION_ENABLED",
            "SECURITY_HEADERS_ENABLED",
            "AUDIT_LOGGING_ENABLED"
        ]
        
        for feature in required_features:
            if not config.get(feature, False):
                issues.append(f"{feature} must be enabled in production")
        
        # Check environment variables
        if config.get("DEBUG", False):
            issues.append("DEBUG must be False in production")
        
        if config.get("ENVIRONMENT") != "production":
            warnings.append("ENVIRONMENT should be set to 'production'")
        
        # Check HTTPS enforcement
        if not config.get("FORCE_HTTPS", False):
            issues.append("FORCE_HTTPS must be enabled in production")
        
        # Recommendations
        if not config.get("REAL_TIME_THREAT_DETECTION", False):
            recommendations.append("Enable real-time threat detection for enhanced security")
        
        if not config.get("SECURITY_MONITORING_ENABLED", False):
            recommendations.append("Enable security monitoring and alerting")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "score": max(0, 100 - len(issues) * 20 - len(warnings) * 5)
        }
    
    @staticmethod
    def generate_secure_keys() -> Dict[str, str]:
        """Generate cryptographically secure keys."""
        return {
            "JWT_SECRET_KEY": secrets.token_urlsafe(64),
            "ENCRYPTION_KEY": secrets.token_urlsafe(32),
            "SECRET_KEY": secrets.token_urlsafe(32),
            "CSRF_SECRET_KEY": secrets.token_urlsafe(32)
        }


class SecurityConfigGenerator:
    """Generates security configuration files."""
    
    @staticmethod
    def generate_env_file(template: SecurityTemplate, output_path: str = None) -> str:
        """Generate .env file from security template."""
        env_content = []
        env_content.append(f"# {template.name}")
        env_content.append(f"# {template.description}")
        env_content.append(f"# Environment: {template.environment.value}")
        env_content.append("")
        
        # Add security notes as comments
        if template.security_notes:
            env_content.append("# SECURITY NOTES:")
            for note in template.security_notes:
                env_content.append(f"# - {note}")
            env_content.append("")
        
        # Add required environment variables
        if template.required_env_vars:
            env_content.append("# REQUIRED ENVIRONMENT VARIABLES:")
            for var in template.required_env_vars:
                if template.environment == DeploymentEnvironment.PRODUCTION:
                    env_content.append(f"{var}=CHANGE_ME_TO_SECURE_VALUE")
                else:
                    # Use generated values for non-production
                    if var in ["JWT_SECRET_KEY", "ENCRYPTION_KEY", "SECRET_KEY"]:
                        env_content.append(f"{var}={secrets.token_urlsafe(64 if 'JWT' in var else 32)}")
                    else:
                        env_content.append(f"{var}=")
            env_content.append("")
        
        # Add configuration values
        env_content.append("# SECURITY CONFIGURATION:")
        for key, value in template.config.items():
            if value == "REQUIRED_ENV_VAR":
                continue  # Skip, already handled above
            env_content.append(f"{key}={value}")
        
        env_content.append("")
        
        # Add optional environment variables
        if template.optional_env_vars:
            env_content.append("# OPTIONAL ENVIRONMENT VARIABLES:")
            for var in template.optional_env_vars:
                env_content.append(f"# {var}=")
            env_content.append("")
        
        env_content.append("# Generated on: " + str(os.getenv('DATE', 'unknown')))
        
        content = "\n".join(env_content)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        
        return content
    
    @staticmethod
    def generate_docker_env(template: SecurityTemplate) -> str:
        """Generate Docker environment file."""
        docker_content = []
        docker_content.append(f"# Docker environment for {template.name}")
        docker_content.append("")
        
        for key, value in template.config.items():
            if value == "REQUIRED_ENV_VAR":
                docker_content.append(f"{key}=$${key}")
            else:
                docker_content.append(f"{key}={value}")
        
        return "\n".join(docker_content)
    
    @staticmethod
    def generate_security_checklist(template: SecurityTemplate) -> str:
        """Generate security deployment checklist."""
        checklist = []
        checklist.append(f"# Security Deployment Checklist - {template.name}")
        checklist.append(f"Environment: {template.environment.value}")
        checklist.append("")
        
        checklist.append("## Pre-Deployment Security Checklist")
        checklist.append("")
        
        if template.required_env_vars:
            checklist.append("### Required Environment Variables")
            for var in template.required_env_vars:
                checklist.append(f"- [ ] {var} - Set to secure value")
            checklist.append("")
        
        checklist.append("### Security Features")
        security_features = [
            "CSRF_PROTECTION_ENABLED",
            "XSS_PROTECTION_ENABLED",
            "SQL_INJECTION_PREVENTION_ENABLED",
            "SECURITY_HEADERS_ENABLED",
            "AUDIT_LOGGING_ENABLED"
        ]
        
        for feature in security_features:
            if template.config.get(feature, False):
                checklist.append(f"- [x] {feature} - Enabled")
            else:
                checklist.append(f"- [ ] {feature} - Disabled (verify intentional)")
        checklist.append("")
        
        if template.security_notes:
            checklist.append("### Security Notes")
            for note in template.security_notes:
                checklist.append(f"- {note}")
            checklist.append("")
        
        checklist.append("### Post-Deployment Verification")
        checklist.append("- [ ] All security headers present in responses")
        checklist.append("- [ ] JWT tokens properly validated")
        checklist.append("- [ ] Rate limiting functioning")
        checklist.append("- [ ] CSRF protection working")
        checklist.append("- [ ] Audit logging active")
        checklist.append("- [ ] Security monitoring alerts configured")
        checklist.append("- [ ] SSL/TLS certificates valid and properly configured")
        checklist.append("- [ ] Database connections encrypted")
        checklist.append("- [ ] Backup systems secured")
        checklist.append("")
        
        return "\n".join(checklist)


# Usage examples
if __name__ == "__main__":
    # Generate production configuration
    templates = SecurityConfigTemplates()
    
    prod_template = templates.get_production_template()
    print("Production Security Template:")
    print("=" * 50)
    
    generator = SecurityConfigGenerator()
    env_content = generator.generate_env_file(prod_template)
    print(env_content)
    
    print("\nValidation Results:")
    print("=" * 50)
    validator = SecurityConfigValidator()
    validation = validator.validate_production_config(prod_template.config)
    print(f"Valid: {validation['valid']}")
    print(f"Score: {validation['score']}/100")
    
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  ❌ {issue}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    if validation['recommendations']:
        print("Recommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")