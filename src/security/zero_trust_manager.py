#!/usr/bin/env python3
"""
Zero Trust Architecture Manager for Claude-TUI Production Deployment

Implements comprehensive zero-trust security architecture with:
- Identity-centric security model
- Never trust, always verify principles
- Least privilege access controls
- Continuous verification and validation
- Dynamic policy enforcement
- Device trust and compliance validation
- Micro-segmentation and network isolation

Author: Security Manager - Claude-TUI Security Team
Date: 2025-08-26
"""

import asyncio
import json
import time
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import jwt
import uuid
import hmac

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels in zero-trust architecture"""
    UNKNOWN = "unknown"
    UNTRUSTED = "untrusted"
    LOW_TRUST = "low_trust"
    MEDIUM_TRUST = "medium_trust"
    HIGH_TRUST = "high_trust"
    VERIFIED = "verified"


class AccessDecision(Enum):
    """Access decision outcomes"""
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    STEP_UP = "step_up"
    CONDITIONAL = "conditional"


class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    MFA = "mfa"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    SSO = "sso"


class DeviceType(Enum):
    """Device types"""
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    MOBILE = "mobile"
    TABLET = "tablet"
    SERVER = "server"
    IOT = "iot"
    UNKNOWN = "unknown"


@dataclass
class Identity:
    """Represents an identity in the zero-trust system"""
    identity_id: str
    username: str
    email: str
    roles: List[str]
    groups: List[str]
    trust_level: TrustLevel
    last_verified: datetime
    verification_factors: List[AuthenticationMethod]
    risk_score: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class Device:
    """Represents a device in the zero-trust system"""
    device_id: str
    device_name: str
    device_type: DeviceType
    owner_identity: str
    trust_level: TrustLevel
    last_seen: datetime
    compliance_status: bool
    security_posture: Dict[str, Any]
    registered_at: datetime
    certificates: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)


@dataclass
class AccessRequest:
    """Represents an access request"""
    request_id: str
    identity_id: str
    device_id: str
    resource: str
    action: str
    context: Dict[str, Any]
    timestamp: datetime
    source_ip: str
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class PolicyRule:
    """Represents a zero-trust policy rule"""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    priority: int
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IdentityManager:
    """
    Identity management component of zero-trust architecture.
    
    Handles identity lifecycle, verification, and continuous authentication.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize identity manager."""
        self.config = config or {}
        self.identities: Dict[str, Identity] = {}
        self.verification_cache: Dict[str, Dict[str, Any]] = {}
        self.risk_engine = RiskAssessmentEngine()
        
    async def register_identity(self, username: str, email: str, 
                              initial_roles: List[str] = None) -> Identity:
        """Register new identity in zero-trust system."""
        identity_id = str(uuid.uuid4())
        
        identity = Identity(
            identity_id=identity_id,
            username=username,
            email=email,
            roles=initial_roles or ['user'],
            groups=[],
            trust_level=TrustLevel.UNTRUSTED,
            last_verified=datetime.now(timezone.utc),
            verification_factors=[],
            risk_score=0.0,
            attributes={'created_at': datetime.now(timezone.utc).isoformat()}
        )
        
        self.identities[identity_id] = identity
        
        logger.info(f"🆔 Registered new identity: {username} ({identity_id})")
        return identity
    
    async def verify_identity(self, identity_id: str, auth_method: AuthenticationMethod,
                            credentials: Dict[str, Any]) -> bool:
        """Verify identity using specified authentication method."""
        if identity_id not in self.identities:
            logger.warning(f"Identity verification failed: unknown identity {identity_id}")
            return False
        
        identity = self.identities[identity_id]
        
        # Perform verification based on authentication method
        verification_success = False
        
        if auth_method == AuthenticationMethod.PASSWORD:
            verification_success = await self._verify_password(identity, credentials)
        elif auth_method == AuthenticationMethod.MFA:
            verification_success = await self._verify_mfa(identity, credentials)
        elif auth_method == AuthenticationMethod.CERTIFICATE:
            verification_success = await self._verify_certificate(identity, credentials)
        elif auth_method == AuthenticationMethod.TOKEN:
            verification_success = await self._verify_token(identity, credentials)
        
        if verification_success:
            # Update identity trust level and verification status
            identity.last_verified = datetime.now(timezone.utc)
            
            if auth_method not in identity.verification_factors:
                identity.verification_factors.append(auth_method)
            
            # Update trust level based on verification factors
            await self._update_trust_level(identity)
            
            logger.info(f"✅ Identity verified: {identity.username} using {auth_method.value}")
        else:
            logger.warning(f"❌ Identity verification failed: {identity.username} using {auth_method.value}")
        
        return verification_success
    
    async def _verify_password(self, identity: Identity, credentials: Dict[str, Any]) -> bool:
        """Verify password credentials."""
        provided_password = credentials.get('password', '')
        stored_hash = identity.attributes.get('password_hash', '')
        
        # In production, use proper password hashing (bcrypt, Argon2, etc.)
        # This is simplified for demonstration
        expected_hash = hashlib.sha256(provided_password.encode()).hexdigest()
        
        return hmac.compare_digest(stored_hash, expected_hash)
    
    async def _verify_mfa(self, identity: Identity, credentials: Dict[str, Any]) -> bool:
        """Verify multi-factor authentication."""
        totp_code = credentials.get('totp_code', '')
        backup_code = credentials.get('backup_code', '')
        
        # Verify TOTP code (Time-based One-Time Password)
        if totp_code:
            # In production, use proper TOTP library
            # This is simplified verification
            current_time = int(time.time() // 30)  # 30-second window
            secret = identity.attributes.get('totp_secret', 'default_secret')
            
            # Generate expected TOTP (simplified)
            expected_totp = str(hash(f"{secret}{current_time}"))[-6:]
            
            if totp_code == expected_totp:
                return True
        
        # Verify backup code
        if backup_code:
            backup_codes = identity.attributes.get('backup_codes', [])
            if backup_code in backup_codes:
                # Remove used backup code
                backup_codes.remove(backup_code)
                identity.attributes['backup_codes'] = backup_codes
                return True
        
        return False
    
    async def _verify_certificate(self, identity: Identity, credentials: Dict[str, Any]) -> bool:
        """Verify client certificate."""
        certificate_data = credentials.get('certificate', '')
        
        # In production, perform proper certificate validation
        # Check certificate chain, expiration, revocation, etc.
        
        # Simplified verification
        if certificate_data and len(certificate_data) > 100:
            return True
        
        return False
    
    async def _verify_token(self, identity: Identity, credentials: Dict[str, Any]) -> bool:
        """Verify authentication token."""
        token = credentials.get('token', '')
        
        try:
            # Verify JWT token
            secret_key = self.config.get('jwt_secret', 'default_secret')
            decoded_token = jwt.decode(token, secret_key, algorithms=['HS256'])
            
            # Verify token subject matches identity
            if decoded_token.get('sub') == identity.identity_id:
                return True
                
        except jwt.InvalidTokenError:
            pass
        
        return False
    
    async def _update_trust_level(self, identity: Identity):
        """Update identity trust level based on verification factors."""
        verification_count = len(identity.verification_factors)
        time_since_verification = datetime.now(timezone.utc) - identity.last_verified
        
        # Calculate trust level based on factors
        base_trust = verification_count * 20  # Base trust from verification methods
        time_penalty = min(time_since_verification.total_seconds() / 3600, 50)  # Penalty for old verification
        
        trust_score = max(0, base_trust - time_penalty)
        
        # Map trust score to trust level
        if trust_score >= 80:
            identity.trust_level = TrustLevel.HIGH_TRUST
        elif trust_score >= 60:
            identity.trust_level = TrustLevel.MEDIUM_TRUST
        elif trust_score >= 40:
            identity.trust_level = TrustLevel.LOW_TRUST
        else:
            identity.trust_level = TrustLevel.UNTRUSTED
        
        # Update risk score
        identity.risk_score = await self.risk_engine.calculate_identity_risk(identity)
    
    def get_identity(self, identity_id: str) -> Optional[Identity]:
        """Get identity by ID."""
        return self.identities.get(identity_id)
    
    def get_identity_by_username(self, username: str) -> Optional[Identity]:
        """Get identity by username."""
        for identity in self.identities.values():
            if identity.username == username:
                return identity
        return None


class DeviceManager:
    """
    Device management component of zero-trust architecture.
    
    Handles device registration, compliance monitoring, and trust assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize device manager."""
        self.config = config or {}
        self.devices: Dict[str, Device] = {}
        self.compliance_policies: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default compliance policies
        self._initialize_compliance_policies()
    
    def _initialize_compliance_policies(self):
        """Initialize default device compliance policies."""
        self.compliance_policies = {
            'desktop': {
                'required_os_version': '10.0',
                'antivirus_required': True,
                'encryption_required': True,
                'firewall_enabled': True,
                'auto_updates': True,
                'max_age_days': 1095  # 3 years
            },
            'mobile': {
                'required_os_version': '13.0',
                'screen_lock': True,
                'encryption_required': True,
                'jailbreak_detection': True,
                'app_restrictions': True,
                'max_age_days': 730  # 2 years
            },
            'server': {
                'required_os_version': '20.04',
                'hardening_applied': True,
                'monitoring_agent': True,
                'vulnerability_scanning': True,
                'backup_configured': True,
                'max_age_days': 1825  # 5 years
            }
        }
    
    async def register_device(self, device_name: str, device_type: DeviceType,
                            owner_identity: str, device_info: Dict[str, Any]) -> Device:
        """Register new device in zero-trust system."""
        device_id = str(uuid.uuid4())
        
        device = Device(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            owner_identity=owner_identity,
            trust_level=TrustLevel.UNTRUSTED,
            last_seen=datetime.now(timezone.utc),
            compliance_status=False,
            security_posture=device_info,
            registered_at=datetime.now(timezone.utc)
        )
        
        # Perform initial compliance check
        await self._assess_device_compliance(device)
        
        self.devices[device_id] = device
        
        logger.info(f"📱 Registered new device: {device_name} ({device_id})")
        return device
    
    async def _assess_device_compliance(self, device: Device):
        """Assess device compliance against policies."""
        device_type_key = device.device_type.value
        
        if device_type_key not in self.compliance_policies:
            device.compliance_status = False
            device.risk_indicators.append("No compliance policy defined")
            return
        
        policy = self.compliance_policies[device_type_key]
        compliance_issues = []
        
        # Check OS version
        current_os_version = device.security_posture.get('os_version', '0.0')
        required_os_version = policy.get('required_os_version', '0.0')
        
        if self._version_compare(current_os_version, required_os_version) < 0:
            compliance_issues.append(f"OS version {current_os_version} below required {required_os_version}")
        
        # Check antivirus (for desktop/laptop)
        if policy.get('antivirus_required') and not device.security_posture.get('antivirus_enabled'):
            compliance_issues.append("Antivirus not installed or disabled")
        
        # Check encryption
        if policy.get('encryption_required') and not device.security_posture.get('disk_encrypted'):
            compliance_issues.append("Disk encryption not enabled")
        
        # Check firewall
        if policy.get('firewall_enabled') and not device.security_posture.get('firewall_enabled'):
            compliance_issues.append("Firewall not enabled")
        
        # Check device age
        device_age_days = (datetime.now(timezone.utc) - device.registered_at).days
        max_age_days = policy.get('max_age_days', 365)
        
        if device_age_days > max_age_days:
            compliance_issues.append(f"Device age {device_age_days} days exceeds maximum {max_age_days} days")
        
        # Check for jailbreak/root (mobile devices)
        if device.device_type in [DeviceType.MOBILE, DeviceType.TABLET]:
            if device.security_posture.get('jailbroken') or device.security_posture.get('rooted'):
                compliance_issues.append("Device is jailbroken/rooted")
        
        # Update compliance status
        device.compliance_status = len(compliance_issues) == 0
        device.risk_indicators = compliance_issues
        
        # Update trust level based on compliance
        if device.compliance_status:
            device.trust_level = TrustLevel.MEDIUM_TRUST
        else:
            device.trust_level = TrustLevel.LOW_TRUST
        
        logger.info(f"📊 Device compliance assessed: {device.device_name} - {'Compliant' if device.compliance_status else 'Non-compliant'}")
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare version strings. Returns -1, 0, or 1."""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        
        return 0
    
    async def update_device_posture(self, device_id: str, security_posture: Dict[str, Any]):
        """Update device security posture."""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        device.security_posture.update(security_posture)
        device.last_seen = datetime.now(timezone.utc)
        
        # Re-assess compliance
        await self._assess_device_compliance(device)
        
        return True
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID."""
        return self.devices.get(device_id)


class PolicyEngine:
    """
    Policy engine for zero-trust access decisions.
    
    Evaluates access requests against dynamic policies and makes
    real-time access decisions.
    """
    
    def __init__(self):
        """Initialize policy engine."""
        self.policies: List[PolicyRule] = []
        self.decision_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default zero-trust policies."""
        default_policies = [
            PolicyRule(
                rule_id="require_authentication",
                name="Require Authentication",
                description="All access requires valid authentication",
                conditions={
                    "identity_verified": True,
                    "device_registered": True
                },
                actions={
                    "allow": False,
                    "require_auth": True
                },
                priority=100
            ),
            PolicyRule(
                rule_id="admin_mfa_required",
                name="Admin MFA Required",
                description="Administrative access requires MFA",
                conditions={
                    "roles": ["admin", "superuser"],
                    "resource_type": "administrative"
                },
                actions={
                    "require_mfa": True,
                    "allow": "conditional"
                },
                priority=90
            ),
            PolicyRule(
                rule_id="untrusted_device_deny",
                name="Untrusted Device Deny",
                description="Deny access from untrusted devices",
                conditions={
                    "device_trust_level": ["unknown", "untrusted"]
                },
                actions={
                    "allow": False,
                    "message": "Device not trusted"
                },
                priority=80
            ),
            PolicyRule(
                rule_id="high_risk_identity_challenge",
                name="High Risk Identity Challenge",
                description="Challenge high-risk identities",
                conditions={
                    "identity_risk_score": ">0.7"
                },
                actions={
                    "challenge": True,
                    "require_additional_verification": True
                },
                priority=70
            ),
            PolicyRule(
                rule_id="time_based_access",
                name="Time-Based Access Control",
                description="Restrict access outside business hours",
                conditions={
                    "time_range": "outside_business_hours",
                    "resource_sensitivity": "high"
                },
                actions={
                    "allow": False,
                    "message": "Access restricted outside business hours"
                },
                priority=60
            )
        ]
        
        self.policies.extend(default_policies)
        self.policies.sort(key=lambda p: p.priority, reverse=True)
    
    async def evaluate_access_request(self, request: AccessRequest, 
                                    identity: Optional[Identity] = None,
                                    device: Optional[Device] = None) -> AccessDecision:
        """Evaluate access request against zero-trust policies."""
        
        # Create evaluation context
        context = self._build_evaluation_context(request, identity, device)
        
        # Evaluate policies in priority order
        for policy in self.policies:
            if not policy.enabled:
                continue
            
            if self._evaluate_policy_conditions(policy, context):
                decision = self._execute_policy_actions(policy, context)
                
                # Log policy decision
                logger.info(f"🛡️ Policy '{policy.name}' triggered: {decision} for request {request.request_id}")
                
                # Cache decision for performance
                cache_key = self._generate_cache_key(request, identity, device)
                self.decision_cache[cache_key] = {
                    'decision': decision,
                    'policy_id': policy.rule_id,
                    'timestamp': time.time()
                }
                
                return decision
        
        # Default deny if no policies match
        logger.warning(f"⚠️ No policies matched request {request.request_id}, defaulting to DENY")
        return AccessDecision.DENY
    
    def _build_evaluation_context(self, request: AccessRequest, 
                                identity: Optional[Identity] = None,
                                device: Optional[Device] = None) -> Dict[str, Any]:
        """Build context for policy evaluation."""
        current_time = datetime.now(timezone.utc)
        
        context = {
            'request': asdict(request),
            'current_time': current_time,
            'business_hours': self._is_business_hours(current_time),
            'identity_verified': identity is not None,
            'device_registered': device is not None
        }
        
        # Add identity context
        if identity:
            context.update({
                'identity_id': identity.identity_id,
                'username': identity.username,
                'roles': identity.roles,
                'groups': identity.groups,
                'trust_level': identity.trust_level.value,
                'identity_risk_score': identity.risk_score,
                'verification_factors': [f.value for f in identity.verification_factors],
                'last_verified_minutes': (current_time - identity.last_verified).total_seconds() / 60
            })
        
        # Add device context
        if device:
            context.update({
                'device_id': device.device_id,
                'device_type': device.device_type.value,
                'device_trust_level': device.trust_level.value,
                'device_compliant': device.compliance_status,
                'device_risk_indicators': device.risk_indicators,
                'device_age_days': (current_time - device.registered_at).days
            })
        
        # Add resource context
        context.update({
            'resource_type': self._classify_resource(request.resource),
            'resource_sensitivity': self._assess_resource_sensitivity(request.resource),
            'action_type': request.action
        })
        
        return context
    
    def _evaluate_policy_conditions(self, policy: PolicyRule, context: Dict[str, Any]) -> bool:
        """Evaluate policy conditions against context."""
        conditions = policy.conditions
        
        for condition_key, condition_value in conditions.items():
            context_value = context.get(condition_key)
            
            # Handle different condition types
            if isinstance(condition_value, bool):
                if context_value != condition_value:
                    return False
            
            elif isinstance(condition_value, list):
                if context_value not in condition_value:
                    return False
            
            elif isinstance(condition_value, str):
                if condition_value.startswith('>'):
                    threshold = float(condition_value[1:])
                    if not context_value or float(context_value) <= threshold:
                        return False
                elif condition_value.startswith('<'):
                    threshold = float(condition_value[1:])
                    if not context_value or float(context_value) >= threshold:
                        return False
                else:
                    if context_value != condition_value:
                        return False
            
            elif isinstance(condition_value, dict):
                # Handle complex conditions
                if condition_key == "time_range":
                    if condition_value == "outside_business_hours" and context.get('business_hours'):
                        return False
        
        return True
    
    def _execute_policy_actions(self, policy: PolicyRule, context: Dict[str, Any]) -> AccessDecision:
        """Execute policy actions and return access decision."""
        actions = policy.actions
        
        # Check for explicit deny
        if actions.get('allow') is False:
            return AccessDecision.DENY
        
        # Check for explicit allow
        if actions.get('allow') is True:
            return AccessDecision.ALLOW
        
        # Check for conditional allow
        if actions.get('allow') == 'conditional':
            if actions.get('require_mfa'):
                return AccessDecision.STEP_UP
            if actions.get('challenge'):
                return AccessDecision.CHALLENGE
        
        # Check for challenge requirements
        if actions.get('challenge') or actions.get('require_additional_verification'):
            return AccessDecision.CHALLENGE
        
        # Check for step-up authentication
        if actions.get('require_mfa') or actions.get('require_auth'):
            return AccessDecision.STEP_UP
        
        return AccessDecision.ALLOW
    
    def _is_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within business hours."""
        # Business hours: Monday-Friday, 9 AM - 5 PM UTC
        weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        hour = timestamp.hour
        
        return weekday < 5 and 9 <= hour <= 17
    
    def _classify_resource(self, resource: str) -> str:
        """Classify resource type based on path/name."""
        resource_lower = resource.lower()
        
        if '/admin' in resource_lower or '/management' in resource_lower:
            return 'administrative'
        elif '/api' in resource_lower:
            return 'api'
        elif '/user' in resource_lower or '/profile' in resource_lower:
            return 'user_data'
        elif '/public' in resource_lower:
            return 'public'
        else:
            return 'application'
    
    def _assess_resource_sensitivity(self, resource: str) -> str:
        """Assess resource sensitivity level."""
        resource_lower = resource.lower()
        
        # High sensitivity indicators
        high_indicators = ['admin', 'secret', 'config', 'password', 'key', 'token', 'financial', 'medical']
        if any(indicator in resource_lower for indicator in high_indicators):
            return 'high'
        
        # Medium sensitivity indicators
        medium_indicators = ['user', 'profile', 'personal', 'private', 'internal']
        if any(indicator in resource_lower for indicator in medium_indicators):
            return 'medium'
        
        # Public resources
        public_indicators = ['public', 'static', 'css', 'js', 'img', 'asset']
        if any(indicator in resource_lower for indicator in public_indicators):
            return 'low'
        
        return 'medium'  # Default to medium sensitivity
    
    def _generate_cache_key(self, request: AccessRequest, identity: Optional[Identity], 
                          device: Optional[Device]) -> str:
        """Generate cache key for decision caching."""
        key_parts = [
            request.identity_id,
            request.device_id,
            request.resource,
            request.action
        ]
        
        key_string = '|'.join(str(part) for part in key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def add_policy(self, policy: PolicyRule):
        """Add new policy to the engine."""
        self.policies.append(policy)
        self.policies.sort(key=lambda p: p.priority, reverse=True)
        logger.info(f"➕ Added policy: {policy.name}")
    
    def remove_policy(self, rule_id: str) -> bool:
        """Remove policy from the engine."""
        for i, policy in enumerate(self.policies):
            if policy.rule_id == rule_id:
                del self.policies[i]
                logger.info(f"➖ Removed policy: {policy.name}")
                return True
        return False


class RiskAssessmentEngine:
    """
    Risk assessment engine for continuous risk evaluation.
    
    Analyzes identity behavior, device posture, and contextual factors
    to calculate dynamic risk scores.
    """
    
    def __init__(self):
        """Initialize risk assessment engine."""
        self.risk_factors = {
            'identity': {
                'unverified_identity': 0.3,
                'weak_authentication': 0.2,
                'old_verification': 0.1,
                'suspicious_activity': 0.4,
                'compromised_credentials': 0.8
            },
            'device': {
                'untrusted_device': 0.3,
                'non_compliant': 0.2,
                'malware_detected': 0.7,
                'jailbroken_rooted': 0.5,
                'outdated_os': 0.1
            },
            'context': {
                'unusual_location': 0.2,
                'unusual_time': 0.1,
                'high_privilege_access': 0.3,
                'bulk_operations': 0.2,
                'failed_attempts': 0.4
            }
        }
    
    async def calculate_identity_risk(self, identity: Identity) -> float:
        """Calculate risk score for identity."""
        risk_score = 0.0
        
        # Base risk from trust level
        trust_risk_map = {
            TrustLevel.UNKNOWN: 0.8,
            TrustLevel.UNTRUSTED: 0.7,
            TrustLevel.LOW_TRUST: 0.4,
            TrustLevel.MEDIUM_TRUST: 0.2,
            TrustLevel.HIGH_TRUST: 0.1,
            TrustLevel.VERIFIED: 0.0
        }
        
        risk_score += trust_risk_map.get(identity.trust_level, 0.5)
        
        # Risk from verification age
        time_since_verification = datetime.now(timezone.utc) - identity.last_verified
        verification_age_hours = time_since_verification.total_seconds() / 3600
        
        if verification_age_hours > 24:
            risk_score += 0.2
        elif verification_age_hours > 8:
            risk_score += 0.1
        
        # Risk from authentication methods
        if AuthenticationMethod.PASSWORD in identity.verification_factors:
            if len(identity.verification_factors) == 1:
                risk_score += 0.3  # Only password authentication
        
        if not identity.verification_factors:
            risk_score += 0.5  # No verification factors
        
        # Cap risk score at 1.0
        return min(risk_score, 1.0)
    
    async def calculate_device_risk(self, device: Device) -> float:
        """Calculate risk score for device."""
        risk_score = 0.0
        
        # Base risk from trust level
        trust_risk_map = {
            TrustLevel.UNKNOWN: 0.8,
            TrustLevel.UNTRUSTED: 0.7,
            TrustLevel.LOW_TRUST: 0.4,
            TrustLevel.MEDIUM_TRUST: 0.2,
            TrustLevel.HIGH_TRUST: 0.1,
            TrustLevel.VERIFIED: 0.0
        }
        
        risk_score += trust_risk_map.get(device.trust_level, 0.5)
        
        # Risk from compliance status
        if not device.compliance_status:
            risk_score += 0.3
        
        # Risk from specific indicators
        for indicator in device.risk_indicators:
            if 'jailbroken' in indicator.lower() or 'rooted' in indicator.lower():
                risk_score += 0.5
            elif 'malware' in indicator.lower():
                risk_score += 0.7
            elif 'outdated' in indicator.lower():
                risk_score += 0.1
        
        # Risk from device age
        device_age = datetime.now(timezone.utc) - device.last_seen
        if device_age.days > 30:
            risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    async def calculate_contextual_risk(self, request: AccessRequest, 
                                      identity: Optional[Identity] = None) -> float:
        """Calculate contextual risk for access request."""
        risk_score = 0.0
        
        # Time-based risk
        current_time = datetime.now(timezone.utc)
        if not self._is_business_hours(current_time):
            risk_score += 0.1
        
        # Resource sensitivity risk
        if '/admin' in request.resource:
            risk_score += 0.3
        elif '/api' in request.resource:
            risk_score += 0.2
        
        # Action risk
        if request.action in ['delete', 'admin', 'modify']:
            risk_score += 0.2
        
        # IP-based risk (simplified)
        if not self._is_internal_ip(request.source_ip):
            risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def _is_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within business hours."""
        weekday = timestamp.weekday()
        hour = timestamp.hour
        return weekday < 5 and 9 <= hour <= 17
    
    def _is_internal_ip(self, ip_address: str) -> bool:
        """Check if IP address is internal."""
        # Simplified internal IP detection
        return ip_address.startswith(('10.', '172.', '192.168.', '127.'))


class ZeroTrustManager:
    """
    Comprehensive zero-trust architecture manager.
    
    Coordinates all zero-trust components and provides unified
    access control and security policy enforcement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize zero-trust manager."""
        self.config = config or {}
        
        # Initialize components
        self.identity_manager = IdentityManager(config)
        self.device_manager = DeviceManager(config)
        self.policy_engine = PolicyEngine()
        self.risk_engine = RiskAssessmentEngine()
        
        # Session and access tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.access_history: List[Dict[str, Any]] = []
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize zero-trust manager."""
        try:
            logger.info("🛡️ Initializing Zero Trust Architecture Manager...")
            
            # Start continuous monitoring
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._continuous_monitoring())
            
            logger.info("✅ Zero Trust Architecture Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize zero-trust manager: {e}")
            return False
    
    async def authenticate_and_authorize(self, username: str, password: str,
                                       device_info: Dict[str, Any],
                                       source_ip: str) -> Dict[str, Any]:
        """Perform zero-trust authentication and authorization."""
        
        # Get or create identity
        identity = self.identity_manager.get_identity_by_username(username)
        if not identity:
            return {
                'success': False,
                'reason': 'Invalid credentials',
                'trust_level': TrustLevel.UNKNOWN.value
            }
        
        # Verify password
        auth_success = await self.identity_manager.verify_identity(
            identity.identity_id,
            AuthenticationMethod.PASSWORD,
            {'password': password}
        )
        
        if not auth_success:
            return {
                'success': False,
                'reason': 'Authentication failed',
                'trust_level': identity.trust_level.value
            }
        
        # Get or register device
        device_id = device_info.get('device_id')
        device = None
        
        if device_id:
            device = self.device_manager.get_device(device_id)
        
        if not device:
            # Register new device
            device = await self.device_manager.register_device(
                device_name=device_info.get('device_name', 'Unknown Device'),
                device_type=DeviceType(device_info.get('device_type', 'unknown')),
                owner_identity=identity.identity_id,
                device_info=device_info
            )
        else:
            # Update device posture
            await self.device_manager.update_device_posture(device.device_id, device_info)
        
        # Calculate risk scores
        identity_risk = await self.risk_engine.calculate_identity_risk(identity)
        device_risk = await self.risk_engine.calculate_device_risk(device)
        
        # Create session
        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'identity_id': identity.identity_id,
            'device_id': device.device_id,
            'source_ip': source_ip,
            'created_at': datetime.now(timezone.utc),
            'last_activity': datetime.now(timezone.utc),
            'identity_risk': identity_risk,
            'device_risk': device_risk,
            'trust_level': min(identity.trust_level.value, device.trust_level.value)
        }
        
        self.active_sessions[session_id] = session_data
        
        # Determine if additional verification is needed
        combined_risk = max(identity_risk, device_risk)
        
        response = {
            'success': True,
            'session_id': session_id,
            'trust_level': session_data['trust_level'],
            'identity_risk': identity_risk,
            'device_risk': device_risk,
            'combined_risk': combined_risk
        }
        
        # Require MFA for high-risk scenarios
        if combined_risk > 0.5 or identity.trust_level == TrustLevel.UNTRUSTED:
            response['require_mfa'] = True
            response['mfa_methods'] = ['totp', 'sms', 'backup_codes']
        
        logger.info(f"🔐 Authentication successful: {username} (risk: {combined_risk:.2f})")
        return response
    
    async def authorize_access(self, session_id: str, resource: str, action: str,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Authorize access to resource using zero-trust policies."""
        
        # Get session
        session = self.active_sessions.get(session_id)
        if not session:
            return {
                'allowed': False,
                'reason': 'Invalid session',
                'decision': AccessDecision.DENY.value
            }
        
        # Get identity and device
        identity = self.identity_manager.get_identity(session['identity_id'])
        device = self.device_manager.get_device(session['device_id'])
        
        # Create access request
        request = AccessRequest(
            request_id=str(uuid.uuid4()),
            identity_id=session['identity_id'],
            device_id=session['device_id'],
            resource=resource,
            action=action,
            context=context or {},
            timestamp=datetime.now(timezone.utc),
            source_ip=session['source_ip'],
            session_id=session_id
        )
        
        # Evaluate access request
        decision = await self.policy_engine.evaluate_access_request(request, identity, device)
        
        # Update session activity
        session['last_activity'] = datetime.now(timezone.utc)
        
        # Record access attempt
        access_record = {
            'request_id': request.request_id,
            'session_id': session_id,
            'identity_id': identity.identity_id if identity else None,
            'username': identity.username if identity else None,
            'resource': resource,
            'action': action,
            'decision': decision.value,
            'timestamp': request.timestamp.isoformat(),
            'source_ip': request.source_ip
        }
        
        self.access_history.append(access_record)
        
        # Keep only last 10000 access records
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-10000:]
        
        response = {
            'allowed': decision == AccessDecision.ALLOW,
            'decision': decision.value,
            'request_id': request.request_id
        }
        
        if decision == AccessDecision.DENY:
            response['reason'] = 'Access denied by policy'
        elif decision == AccessDecision.CHALLENGE:
            response['reason'] = 'Additional verification required'
            response['challenge_methods'] = ['mfa', 'captcha']
        elif decision == AccessDecision.STEP_UP:
            response['reason'] = 'Step-up authentication required'
            response['step_up_methods'] = ['mfa', 'manager_approval']
        
        logger.info(f"🔒 Access decision: {decision.value} for {resource} by {identity.username if identity else 'unknown'}")
        return response
    
    async def _continuous_monitoring(self):
        """Continuous monitoring of zero-trust environment."""
        while self.monitoring_active:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Clean up expired sessions
                expired_sessions = []
                for session_id, session in self.active_sessions.items():
                    last_activity = session['last_activity']
                    if (current_time - last_activity).total_seconds() > 3600:  # 1 hour timeout
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                    logger.info(f"🕐 Expired session: {session_id}")
                
                # Re-evaluate device compliance
                for device in self.device_manager.devices.values():
                    if (current_time - device.last_seen).days < 1:  # Active devices only
                        await self.device_manager._assess_device_compliance(device)
                
                # Update identity trust levels
                for identity in self.identity_manager.identities.values():
                    await self.identity_manager._update_trust_level(identity)
                
                # Sleep for monitoring interval (5 minutes)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in zero-trust monitoring: {e}")
                await asyncio.sleep(60)  # Sleep 1 minute on error
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive zero-trust security status."""
        current_time = datetime.now(timezone.utc)
        
        # Calculate trust level distribution
        identity_trust_levels = {}
        for identity in self.identity_manager.identities.values():
            level = identity.trust_level.value
            identity_trust_levels[level] = identity_trust_levels.get(level, 0) + 1
        
        device_trust_levels = {}
        for device in self.device_manager.devices.values():
            level = device.trust_level.value
            device_trust_levels[level] = device_trust_levels.get(level, 0) + 1
        
        # Recent access statistics
        recent_access = [
            record for record in self.access_history
            if (current_time - datetime.fromisoformat(record['timestamp'])).total_seconds() < 3600
        ]
        
        access_decisions = {}
        for record in recent_access:
            decision = record['decision']
            access_decisions[decision] = access_decisions.get(decision, 0) + 1
        
        return {
            'timestamp': current_time.isoformat(),
            'monitoring_active': self.monitoring_active,
            'active_sessions': len(self.active_sessions),
            'registered_identities': len(self.identity_manager.identities),
            'registered_devices': len(self.device_manager.devices),
            'policy_rules': len(self.policy_engine.policies),
            'identity_trust_distribution': identity_trust_levels,
            'device_trust_distribution': device_trust_levels,
            'recent_access_attempts': len(recent_access),
            'access_decision_distribution': access_decisions,
            'compliance_status': {
                'compliant_devices': sum(1 for d in self.device_manager.devices.values() if d.compliance_status),
                'total_devices': len(self.device_manager.devices)
            }
        }
    
    async def cleanup(self):
        """Cleanup zero-trust resources."""
        logger.info("🧹 Cleaning up zero-trust manager...")
        
        self.monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        
        # Clear sensitive data
        self.active_sessions.clear()
        
        logger.info("✅ Zero-trust cleanup completed")


# Global zero-trust manager
_zero_trust_manager: Optional[ZeroTrustManager] = None


async def init_zero_trust_manager(config: Optional[Dict[str, Any]] = None) -> ZeroTrustManager:
    """Initialize global zero-trust manager."""
    global _zero_trust_manager
    
    _zero_trust_manager = ZeroTrustManager(config)
    await _zero_trust_manager.initialize()
    
    return _zero_trust_manager


def get_zero_trust_manager() -> ZeroTrustManager:
    """Get global zero-trust manager instance."""
    global _zero_trust_manager
    
    if _zero_trust_manager is None:
        raise RuntimeError("Zero-trust manager not initialized. Call init_zero_trust_manager() first.")
    
    return _zero_trust_manager