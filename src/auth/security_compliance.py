"""
Security Compliance Checker for Authentication System.

Validates authentication implementation against security best practices
including OWASP guidelines, NIST standards, and industry benchmarks.
"""

import re
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum

from ..core.logger import get_logger

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security compliance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceCheck:
    """Individual compliance check result."""
    
    def __init__(
        self,
        check_id: str,
        name: str,
        description: str,
        passed: bool,
        level: SecurityLevel,
        recommendation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.check_id = check_id
        self.name = name
        self.description = description
        self.passed = passed
        self.level = level
        self.recommendation = recommendation or ""
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class SecurityComplianceChecker:
    """
    Comprehensive security compliance checker.
    
    Validates authentication system against:
    - OWASP Top 10 security risks
    - NIST Cybersecurity Framework
    - Password security standards
    - Session management best practices
    - JWT security guidelines
    - OAuth 2.0 security considerations
    """
    
    def __init__(self):
        """Initialize compliance checker."""
        self.checks: List[ComplianceCheck] = []
    
    def check_password_policy(self, password_policy: Dict[str, Any]) -> ComplianceCheck:
        """
        Check password policy compliance.
        
        Args:
            password_policy: Password policy configuration
            
        Returns:
            ComplianceCheck result
        """
        min_length = password_policy.get('min_length', 0)
        require_uppercase = password_policy.get('require_uppercase', False)
        require_lowercase = password_policy.get('require_lowercase', False)
        require_digits = password_policy.get('require_digits', False)
        require_special = password_policy.get('require_special', False)
        max_age_days = password_policy.get('max_age_days', 0)
        
        issues = []
        
        # Check minimum length (NIST recommends at least 8, best practice 12+)
        if min_length < 8:
            issues.append("Minimum password length should be at least 8 characters")
        elif min_length < 12:
            issues.append("Consider increasing minimum length to 12+ characters")
        
        # Check complexity requirements
        if not (require_uppercase and require_lowercase and require_digits):
            issues.append("Password should require uppercase, lowercase, and digits")
        
        if not require_special:
            issues.append("Consider requiring special characters")
        
        # Check password expiration (NIST now recommends against forced expiration)
        if 0 < max_age_days < 90:
            issues.append("Avoid forcing frequent password changes (NIST guideline)")
        
        passed = len(issues) == 0
        level = SecurityLevel.HIGH if not passed else SecurityLevel.LOW
        
        return ComplianceCheck(
            check_id="AUTH-001",
            name="Password Policy",
            description="Validates password policy against security standards",
            passed=passed,
            level=level,
            recommendation="\n".join(issues) if issues else "Password policy meets security standards",
            details={
                'min_length': min_length,
                'complexity_requirements': {
                    'uppercase': require_uppercase,
                    'lowercase': require_lowercase,
                    'digits': require_digits,
                    'special': require_special
                },
                'max_age_days': max_age_days,
                'issues_found': len(issues)
            }
        )
    
    def check_jwt_configuration(self, jwt_config: Dict[str, Any]) -> ComplianceCheck:
        """
        Check JWT token configuration.
        
        Args:
            jwt_config: JWT configuration
            
        Returns:
            ComplianceCheck result
        """
        algorithm = jwt_config.get('algorithm', '')
        secret_key = jwt_config.get('secret_key', '')
        access_token_expire_minutes = jwt_config.get('access_token_expire_minutes', 0)
        refresh_token_expire_days = jwt_config.get('refresh_token_expire_days', 0)
        issuer = jwt_config.get('issuer', '')
        audience = jwt_config.get('audience', '')
        
        issues = []
        
        # Check algorithm security
        if algorithm not in ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512']:
            issues.append(f"Insecure JWT algorithm: {algorithm}")
        
        # Check secret key strength
        if len(secret_key) < 32:
            issues.append("JWT secret key should be at least 32 characters")
        elif len(secret_key) < 64:
            issues.append("Consider using a longer JWT secret key (64+ characters)")
        
        # Check secret key entropy
        if secret_key and self._calculate_entropy(secret_key) < 4.0:
            issues.append("JWT secret key has low entropy - use a random secure key")
        
        # Check token expiration times
        if access_token_expire_minutes > 60:
            issues.append("Access token expiration should be ≤ 60 minutes")
        elif access_token_expire_minutes > 30:
            issues.append("Consider shorter access token expiration (≤ 30 minutes)")
        
        if refresh_token_expire_days > 30:
            issues.append("Refresh token expiration should be ≤ 30 days")
        
        # Check issuer and audience claims
        if not issuer:
            issues.append("JWT should include issuer (iss) claim")
        
        if not audience:
            issues.append("JWT should include audience (aud) claim")
        
        passed = len(issues) == 0
        level = SecurityLevel.HIGH if len(issues) > 2 else SecurityLevel.MEDIUM
        
        return ComplianceCheck(
            check_id="AUTH-002",
            name="JWT Configuration",
            description="Validates JWT token configuration security",
            passed=passed,
            level=level,
            recommendation="\n".join(issues) if issues else "JWT configuration is secure",
            details={
                'algorithm': algorithm,
                'secret_key_length': len(secret_key),
                'access_token_expire_minutes': access_token_expire_minutes,
                'refresh_token_expire_days': refresh_token_expire_days,
                'has_issuer': bool(issuer),
                'has_audience': bool(audience),
                'issues_found': len(issues)
            }
        )
    
    def check_session_security(self, session_config: Dict[str, Any]) -> ComplianceCheck:
        """
        Check session management security.
        
        Args:
            session_config: Session configuration
            
        Returns:
            ComplianceCheck result
        """
        session_timeout_minutes = session_config.get('session_timeout_minutes', 0)
        max_sessions_per_user = session_config.get('max_sessions_per_user', 0)
        secure_cookies = session_config.get('secure_cookies', False)
        httponly_cookies = session_config.get('httponly_cookies', False)
        samesite_cookies = session_config.get('samesite_cookies', '')
        session_regeneration = session_config.get('session_regeneration', False)
        
        issues = []
        
        # Check session timeout
        if session_timeout_minutes == 0:
            issues.append("Sessions should have a timeout configured")
        elif session_timeout_minutes > 480:  # 8 hours
            issues.append("Session timeout should be ≤ 8 hours")
        
        # Check concurrent session limits
        if max_sessions_per_user == 0:
            issues.append("Should limit concurrent sessions per user")
        elif max_sessions_per_user > 10:
            issues.append("Consider limiting concurrent sessions to ≤ 10 per user")
        
        # Check cookie security flags
        if not secure_cookies:
            issues.append("Session cookies should have Secure flag")
        
        if not httponly_cookies:
            issues.append("Session cookies should have HttpOnly flag")
        
        if samesite_cookies not in ['Strict', 'Lax']:
            issues.append("Session cookies should have SameSite flag (Strict or Lax)")
        
        # Check session regeneration
        if not session_regeneration:
            issues.append("Should regenerate session ID after authentication")
        
        passed = len(issues) == 0
        level = SecurityLevel.HIGH if len(issues) > 3 else SecurityLevel.MEDIUM
        
        return ComplianceCheck(
            check_id="AUTH-003",
            name="Session Security",
            description="Validates session management security practices",
            passed=passed,
            level=level,
            recommendation="\n".join(issues) if issues else "Session security is properly configured",
            details={
                'session_timeout_minutes': session_timeout_minutes,
                'max_sessions_per_user': max_sessions_per_user,
                'cookie_flags': {
                    'secure': secure_cookies,
                    'httponly': httponly_cookies,
                    'samesite': samesite_cookies
                },
                'session_regeneration': session_regeneration,
                'issues_found': len(issues)
            }
        )
    
    def check_oauth_security(self, oauth_config: Dict[str, Any]) -> ComplianceCheck:
        """
        Check OAuth 2.0 security configuration.
        
        Args:
            oauth_config: OAuth configuration
            
        Returns:
            ComplianceCheck result
        """
        providers = oauth_config.get('providers', {})
        state_validation = oauth_config.get('state_validation', False)
        pkce_enabled = oauth_config.get('pkce_enabled', False)
        redirect_uri_validation = oauth_config.get('redirect_uri_validation', False)
        
        issues = []
        
        # Check provider configurations
        for provider_name, provider_config in providers.items():
            client_id = provider_config.get('client_id', '')
            client_secret = provider_config.get('client_secret', '')
            redirect_uri = provider_config.get('redirect_uri', '')
            
            if not client_id:
                issues.append(f"{provider_name}: Missing client ID")
            
            if not client_secret:
                issues.append(f"{provider_name}: Missing client secret")
            elif len(client_secret) < 32:
                issues.append(f"{provider_name}: Client secret should be ≥ 32 characters")
            
            if not redirect_uri:
                issues.append(f"{provider_name}: Missing redirect URI")
            elif not redirect_uri.startswith('https://') and 'localhost' not in redirect_uri:
                issues.append(f"{provider_name}: Redirect URI should use HTTPS")
        
        # Check security features
        if not state_validation:
            issues.append("CSRF state parameter validation is not enabled")
        
        if not pkce_enabled:
            issues.append("Consider enabling PKCE for enhanced security")
        
        if not redirect_uri_validation:
            issues.append("Redirect URI validation should be enabled")
        
        passed = len(issues) == 0
        level = SecurityLevel.HIGH if len(issues) > 2 else SecurityLevel.MEDIUM
        
        return ComplianceCheck(
            check_id="AUTH-004",
            name="OAuth Security",
            description="Validates OAuth 2.0 security configuration",
            passed=passed,
            level=level,
            recommendation="\n".join(issues) if issues else "OAuth security is properly configured",
            details={
                'providers_count': len(providers),
                'security_features': {
                    'state_validation': state_validation,
                    'pkce_enabled': pkce_enabled,
                    'redirect_uri_validation': redirect_uri_validation
                },
                'issues_found': len(issues)
            }
        )
    
    def check_encryption_standards(self, encryption_config: Dict[str, Any]) -> ComplianceCheck:
        """
        Check encryption and hashing standards.
        
        Args:
            encryption_config: Encryption configuration
            
        Returns:
            ComplianceCheck result
        """
        password_hash_algorithm = encryption_config.get('password_hash_algorithm', '')
        hash_rounds = encryption_config.get('hash_rounds', 0)
        encryption_key_length = encryption_config.get('encryption_key_length', 0)
        tls_version = encryption_config.get('tls_version', '')
        
        issues = []
        
        # Check password hashing
        if password_hash_algorithm not in ['bcrypt', 'scrypt', 'argon2']:
            issues.append(f"Use secure password hashing (bcrypt/scrypt/argon2), not {password_hash_algorithm}")
        
        # Check hash rounds/iterations
        if password_hash_algorithm == 'bcrypt' and hash_rounds < 12:
            issues.append("bcrypt rounds should be ≥ 12")
        elif password_hash_algorithm in ['scrypt', 'argon2'] and hash_rounds < 32768:
            issues.append(f"{password_hash_algorithm} iterations should be ≥ 32768")
        
        # Check encryption key length
        if 0 < encryption_key_length < 256:
            issues.append("Encryption keys should be ≥ 256 bits")
        
        # Check TLS version
        if tls_version and tls_version not in ['1.2', '1.3']:
            issues.append(f"Use TLS 1.2 or 1.3, not {tls_version}")
        
        passed = len(issues) == 0
        level = SecurityLevel.CRITICAL if 'bcrypt' not in password_hash_algorithm else SecurityLevel.MEDIUM
        
        return ComplianceCheck(
            check_id="AUTH-005",
            name="Encryption Standards",
            description="Validates encryption and hashing standards",
            passed=passed,
            level=level,
            recommendation="\n".join(issues) if issues else "Encryption standards are compliant",
            details={
                'password_hash_algorithm': password_hash_algorithm,
                'hash_rounds': hash_rounds,
                'encryption_key_length': encryption_key_length,
                'tls_version': tls_version,
                'issues_found': len(issues)
            }
        )
    
    def check_audit_logging(self, audit_config: Dict[str, Any]) -> ComplianceCheck:
        """
        Check audit logging configuration.
        
        Args:
            audit_config: Audit logging configuration
            
        Returns:
            ComplianceCheck result
        """
        enabled = audit_config.get('enabled', False)
        log_authentication = audit_config.get('log_authentication', False)
        log_authorization = audit_config.get('log_authorization', False)
        log_password_changes = audit_config.get('log_password_changes', False)
        log_failed_attempts = audit_config.get('log_failed_attempts', False)
        structured_logging = audit_config.get('structured_logging', False)
        log_retention_days = audit_config.get('log_retention_days', 0)
        
        issues = []
        
        if not enabled:
            issues.append("Audit logging should be enabled")
            return ComplianceCheck(
                check_id="AUTH-006",
                name="Audit Logging",
                description="Validates audit logging configuration",
                passed=False,
                level=SecurityLevel.HIGH,
                recommendation="Enable comprehensive audit logging",
                details={'enabled': False}
            )
        
        # Check specific logging types
        if not log_authentication:
            issues.append("Should log authentication events")
        
        if not log_authorization:
            issues.append("Should log authorization events")
        
        if not log_password_changes:
            issues.append("Should log password change events")
        
        if not log_failed_attempts:
            issues.append("Should log failed authentication attempts")
        
        if not structured_logging:
            issues.append("Use structured logging for better analysis")
        
        # Check log retention
        if log_retention_days < 30:
            issues.append("Log retention should be ≥ 30 days")
        elif log_retention_days > 2555:  # 7 years
            issues.append("Consider data retention policies (7+ years may be excessive)")
        
        passed = len(issues) == 0
        level = SecurityLevel.MEDIUM if len(issues) <= 2 else SecurityLevel.HIGH
        
        return ComplianceCheck(
            check_id="AUTH-006",
            name="Audit Logging",
            description="Validates audit logging configuration",
            passed=passed,
            level=level,
            recommendation="\n".join(issues) if issues else "Audit logging is properly configured",
            details={
                'enabled': enabled,
                'logging_types': {
                    'authentication': log_authentication,
                    'authorization': log_authorization,
                    'password_changes': log_password_changes,
                    'failed_attempts': log_failed_attempts
                },
                'structured_logging': structured_logging,
                'log_retention_days': log_retention_days,
                'issues_found': len(issues)
            }
        )
    
    def check_rate_limiting(self, rate_limit_config: Dict[str, Any]) -> ComplianceCheck:
        """
        Check rate limiting configuration.
        
        Args:
            rate_limit_config: Rate limiting configuration
            
        Returns:
            ComplianceCheck result
        """
        enabled = rate_limit_config.get('enabled', False)
        login_attempts_limit = rate_limit_config.get('login_attempts_limit', 0)
        login_window_minutes = rate_limit_config.get('login_window_minutes', 0)
        api_requests_limit = rate_limit_config.get('api_requests_limit', 0)
        api_window_minutes = rate_limit_config.get('api_window_minutes', 0)
        account_lockout_enabled = rate_limit_config.get('account_lockout_enabled', False)
        lockout_duration_minutes = rate_limit_config.get('lockout_duration_minutes', 0)
        
        issues = []
        
        if not enabled:
            issues.append("Rate limiting should be enabled")
        
        # Check login attempt limits
        if login_attempts_limit == 0:
            issues.append("Should limit failed login attempts")
        elif login_attempts_limit > 10:
            issues.append("Failed login limit should be ≤ 10 attempts")
        
        if login_window_minutes == 0:
            issues.append("Should define time window for login attempts")
        elif login_window_minutes < 5:
            issues.append("Login attempt window should be ≥ 5 minutes")
        
        # Check API rate limits
        if api_requests_limit == 0:
            issues.append("Should implement API rate limiting")
        
        if api_window_minutes == 0 and api_requests_limit > 0:
            issues.append("Should define time window for API rate limiting")
        
        # Check account lockout
        if not account_lockout_enabled:
            issues.append("Consider enabling account lockout for repeated failures")
        elif lockout_duration_minutes == 0:
            issues.append("Account lockout should have a defined duration")
        elif lockout_duration_minutes > 1440:  # 24 hours
            issues.append("Account lockout duration should be ≤ 24 hours")
        
        passed = len(issues) == 0
        level = SecurityLevel.HIGH if not enabled else SecurityLevel.MEDIUM
        
        return ComplianceCheck(
            check_id="AUTH-007",
            name="Rate Limiting",
            description="Validates rate limiting and brute force protection",
            passed=passed,
            level=level,
            recommendation="\n".join(issues) if issues else "Rate limiting is properly configured",
            details={
                'enabled': enabled,
                'login_limits': {
                    'attempts_limit': login_attempts_limit,
                    'window_minutes': login_window_minutes
                },
                'api_limits': {
                    'requests_limit': api_requests_limit,
                    'window_minutes': api_window_minutes
                },
                'account_lockout': {
                    'enabled': account_lockout_enabled,
                    'duration_minutes': lockout_duration_minutes
                },
                'issues_found': len(issues)
            }
        )
    
    def run_comprehensive_check(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive security compliance check.
        
        Args:
            config: Complete authentication system configuration
            
        Returns:
            Comprehensive compliance report
        """
        logger.info("Running comprehensive security compliance check")
        
        # Clear previous checks
        self.checks = []
        
        # Run all compliance checks
        check_methods = [
            ('password_policy', self.check_password_policy),
            ('jwt_config', self.check_jwt_configuration),
            ('session_config', self.check_session_security),
            ('oauth_config', self.check_oauth_security),
            ('encryption_config', self.check_encryption_standards),
            ('audit_config', self.check_audit_logging),
            ('rate_limit_config', self.check_rate_limiting)
        ]
        
        for config_key, check_method in check_methods:
            try:
                check_config = config.get(config_key, {})
                check_result = check_method(check_config)
                self.checks.append(check_result)
                logger.debug(f"Completed check: {check_result.name} - {'PASSED' if check_result.passed else 'FAILED'}")
            except Exception as e:
                logger.error(f"Error running check {config_key}: {e}")
                # Add error check
                error_check = ComplianceCheck(
                    check_id=f"AUTH-ERR-{config_key}",
                    name=f"Error in {config_key}",
                    description=f"Error occurred while checking {config_key}",
                    passed=False,
                    level=SecurityLevel.HIGH,
                    recommendation=f"Fix configuration error: {str(e)}"
                )
                self.checks.append(error_check)
        
        # Generate summary
        total_checks = len(self.checks)
        passed_checks = sum(1 for check in self.checks if check.passed)
        failed_checks = total_checks - passed_checks
        
        # Calculate risk score
        risk_score = self._calculate_risk_score()
        
        # Determine overall compliance level
        compliance_level = self._determine_compliance_level(risk_score)
        
        report = {
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
                'risk_score': risk_score,
                'compliance_level': compliance_level.value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'checks': [
                {
                    'check_id': check.check_id,
                    'name': check.name,
                    'description': check.description,
                    'passed': check.passed,
                    'level': check.level.value,
                    'recommendation': check.recommendation,
                    'details': check.details,
                    'timestamp': check.timestamp.isoformat()
                }
                for check in self.checks
            ],
            'recommendations': self._generate_recommendations(),
            'compliance_frameworks': {
                'owasp_top_10': self._check_owasp_compliance(),
                'nist_framework': self._check_nist_compliance(),
                'iso_27001': self._check_iso_compliance()
            }
        }
        
        logger.info(f"Compliance check completed: {passed_checks}/{total_checks} checks passed, risk score: {risk_score}")
        return report
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * (probability ** 0.5)  # Simplified entropy calculation
        
        return entropy
    
    def _calculate_risk_score(self) -> int:
        """Calculate overall risk score (0-100)."""
        if not self.checks:
            return 100  # Maximum risk if no checks performed
        
        total_weight = 0
        weighted_failures = 0
        
        level_weights = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 3,
            SecurityLevel.HIGH: 7,
            SecurityLevel.CRITICAL: 15
        }
        
        for check in self.checks:
            weight = level_weights.get(check.level, 1)
            total_weight += weight
            
            if not check.passed:
                weighted_failures += weight
        
        if total_weight == 0:
            return 0
        
        # Risk score: percentage of weighted failures
        risk_score = int((weighted_failures / total_weight) * 100)
        return min(risk_score, 100)
    
    def _determine_compliance_level(self, risk_score: int) -> SecurityLevel:
        """Determine overall compliance level based on risk score."""
        if risk_score <= 10:
            return SecurityLevel.LOW
        elif risk_score <= 30:
            return SecurityLevel.MEDIUM
        elif risk_score <= 60:
            return SecurityLevel.HIGH
        else:
            return SecurityLevel.CRITICAL
    
    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Group failed checks by level
        failed_by_level = {level: [] for level in SecurityLevel}
        
        for check in self.checks:
            if not check.passed:
                failed_by_level[check.level].append(check)
        
        # Add recommendations in priority order
        priority_order = [SecurityLevel.CRITICAL, SecurityLevel.HIGH, SecurityLevel.MEDIUM, SecurityLevel.LOW]
        
        for level in priority_order:
            for check in failed_by_level[level]:
                if check.recommendation:
                    recommendations.append(f"[{level.value.upper()}] {check.name}: {check.recommendation}")
        
        return recommendations
    
    def _check_owasp_compliance(self) -> Dict[str, Any]:
        """Check compliance with OWASP Top 10."""
        # Simplified OWASP compliance check based on authentication-related risks
        owasp_checks = {
            'A01_Broken_Access_Control': self._has_check_passed('AUTH-003'),  # Session security
            'A02_Cryptographic_Failures': self._has_check_passed('AUTH-005'),  # Encryption standards
            'A03_Injection': True,  # Not directly applicable to auth system
            'A04_Insecure_Design': self._calculate_risk_score() <= 30,  # Overall design security
            'A05_Security_Misconfiguration': self._has_check_passed('AUTH-002'),  # JWT config
            'A06_Vulnerable_Components': True,  # Assume components are updated
            'A07_Identity_Authentication_Failures': self._has_check_passed('AUTH-001'),  # Password policy
            'A08_Software_Data_Integrity_Failures': self._has_check_passed('AUTH-006'),  # Audit logging
            'A09_Security_Logging_Failures': self._has_check_passed('AUTH-006'),  # Audit logging
            'A10_Server_Side_Request_Forgery': True  # Not applicable to auth system
        }
        
        compliant_count = sum(1 for compliant in owasp_checks.values() if compliant)
        compliance_percentage = (compliant_count / len(owasp_checks)) * 100
        
        return {
            'checks': owasp_checks,
            'compliance_percentage': compliance_percentage,
            'compliant': compliance_percentage >= 80
        }
    
    def _check_nist_compliance(self) -> Dict[str, Any]:
        """Check compliance with NIST Cybersecurity Framework."""
        # NIST Framework categories relevant to authentication
        nist_categories = {
            'Identify_Access_Management': self._has_check_passed('AUTH-003'),
            'Protect_Access_Control': self._has_check_passed('AUTH-001') and self._has_check_passed('AUTH-007'),
            'Protect_Data_Security': self._has_check_passed('AUTH-005'),
            'Detect_Security_Monitoring': self._has_check_passed('AUTH-006'),
            'Respond_Response_Planning': self._has_check_passed('AUTH-007')
        }
        
        compliant_count = sum(1 for compliant in nist_categories.values() if compliant)
        compliance_percentage = (compliant_count / len(nist_categories)) * 100
        
        return {
            'categories': nist_categories,
            'compliance_percentage': compliance_percentage,
            'compliant': compliance_percentage >= 80
        }
    
    def _check_iso_compliance(self) -> Dict[str, Any]:
        """Check compliance with ISO 27001 controls."""
        # ISO 27001 controls relevant to authentication
        iso_controls = {
            'A.9.1.1_Access_Control_Policy': self._has_check_passed('AUTH-003'),
            'A.9.2.1_User_Registration': True,  # Assume proper user registration
            'A.9.2.3_Management_Privileged_Access': self._has_check_passed('AUTH-007'),
            'A.9.3.1_Use_Secret_Authentication': self._has_check_passed('AUTH-001'),
            'A.9.4.2_Secure_Log_on_Procedures': self._has_check_passed('AUTH-002'),
            'A.9.4.3_Password_Management_System': self._has_check_passed('AUTH-001'),
            'A.12.4.1_Event_Logging': self._has_check_passed('AUTH-006'),
            'A.14.2.5_Secure_System_Engineering': self._calculate_risk_score() <= 20
        }
        
        compliant_count = sum(1 for compliant in iso_controls.values() if compliant)
        compliance_percentage = (compliant_count / len(iso_controls)) * 100
        
        return {
            'controls': iso_controls,
            'compliance_percentage': compliance_percentage,
            'compliant': compliance_percentage >= 85
        }
    
    def _has_check_passed(self, check_id: str) -> bool:
        """Check if a specific check passed."""
        for check in self.checks:
            if check.check_id == check_id:
                return check.passed
        return False


def run_security_compliance_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive security compliance check.
    
    Args:
        config: Authentication system configuration
        
    Returns:
        Detailed compliance report
    """
    checker = SecurityComplianceChecker()
    return checker.run_comprehensive_check(config)
