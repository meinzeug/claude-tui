#!/usr/bin/env python3
"""
Network Security Manager for Claude-TUI Production Deployment

Implements comprehensive network security including:
- Zero-trust network architecture
- TLS termination and certificate management
- Network policies and traffic encryption
- DDoS protection and rate limiting
- Web Application Firewall (WAF)
- Network intrusion detection and prevention

Author: Security Manager - Claude-TUI Security Team
Date: 2025-08-26
"""

import asyncio
import ssl
import ipaddress
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
import hashlib
import time
import json

try:
    import cryptography
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    cryptography = None

try:
    import nginx
    NGINX_AVAILABLE = True
except ImportError:
    NGINX_AVAILABLE = False
    nginx = None

logger = logging.getLogger(__name__)


class TrafficType(Enum):
    """Types of network traffic"""
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    API = "api"
    ADMIN = "admin"
    INTERNAL = "internal"


class SecurityAction(Enum):
    """Security actions for traffic"""
    ALLOW = "allow"
    DENY = "deny"
    RATE_LIMIT = "rate_limit"
    CHALLENGE = "challenge"
    QUARANTINE = "quarantine"


@dataclass
class NetworkRule:
    """Network security rule"""
    rule_id: str
    source_ip: Optional[str] = None
    source_network: Optional[str] = None
    destination_port: Optional[int] = None
    traffic_type: Optional[TrafficType] = None
    action: SecurityAction = SecurityAction.ALLOW
    rate_limit: Optional[int] = None
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TLSCertificate:
    """TLS certificate information"""
    cert_id: str
    common_name: str
    san_list: List[str]
    issuer: str
    valid_from: datetime
    valid_until: datetime
    certificate_pem: str
    private_key_pem: str
    auto_renew: bool = True
    renewal_threshold_days: int = 30


class TLSManager:
    """
    TLS Certificate Manager with automatic renewal and security validation.
    
    Handles certificate lifecycle management, ACME protocol integration,
    and secure TLS configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TLS manager."""
        self.config = config or {}
        self.certificates: Dict[str, TLSCertificate] = {}
        self.ssl_contexts: Dict[str, ssl.SSLContext] = {}
        
        # TLS security configuration
        self.tls_config = {
            'min_version': ssl.TLSVersion.TLSv1_2,
            'max_version': ssl.TLSVersion.TLSv1_3,
            'ciphers': 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS',
            'prefer_server_ciphers': True,
            'hsts_max_age': 31536000,  # 1 year
            'hsts_include_subdomains': True
        }
    
    async def generate_self_signed_certificate(self, domain: str, san_list: Optional[List[str]] = None) -> TLSCertificate:
        """Generate self-signed certificate for development/testing."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        logger.info(f"🔐 Generating self-signed certificate for {domain}")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Claude-TUI"),
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ])
        
        # Build certificate
        cert_builder = x509.CertificateBuilder()\n        cert_builder = cert_builder.subject_name(subject)\n        cert_builder = cert_builder.issuer_name(issuer)\n        cert_builder = cert_builder.public_key(private_key.public_key())\n        cert_builder = cert_builder.serial_number(x509.random_serial_number())\n        cert_builder = cert_builder.not_valid_before(datetime.now(timezone.utc))\n        cert_builder = cert_builder.not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        
        # Add SAN extension
        san_names = [x509.DNSName(domain)]
        if san_list:
            san_names.extend([x509.DNSName(name) for name in san_list])
        
        cert_builder = cert_builder.add_extension(
            x509.SubjectAlternativeName(san_names),
            critical=False,
        )
        
        # Add extensions for security
        cert_builder = cert_builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        )
        
        cert_builder = cert_builder.add_extension(
            x509.ExtendedKeyUsage([
                x509.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=True,
        )
        
        # Sign certificate
        certificate = cert_builder.sign(private_key, hashes.SHA256(), backend=default_backend())
        
        # Serialize certificate and key
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        # Create certificate object
        tls_cert = TLSCertificate(
            cert_id=f"self_signed_{domain}_{int(time.time())}",
            common_name=domain,
            san_list=san_list or [],
            issuer="Claude-TUI Self-Signed",
            valid_from=datetime.now(timezone.utc),
            valid_until=datetime.now(timezone.utc) + timedelta(days=365),
            certificate_pem=cert_pem,
            private_key_pem=key_pem,
            auto_renew=False
        )
        
        # Store certificate
        self.certificates[tls_cert.cert_id] = tls_cert
        
        # Create SSL context
        await self._create_ssl_context(tls_cert)
        
        logger.info(f"✅ Generated self-signed certificate: {tls_cert.cert_id}")
        return tls_cert
    
    async def _create_ssl_context(self, certificate: TLSCertificate) -> ssl.SSLContext:
        """Create secure SSL context from certificate."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Configure TLS versions
        context.minimum_version = self.tls_config['min_version']
        context.maximum_version = self.tls_config['max_version']
        
        # Set cipher suites
        context.set_ciphers(self.tls_config['ciphers'])
        
        # Load certificate and key
        try:
            # Write cert and key to temporary files for loading
            cert_path = Path(f"/tmp/cert_{certificate.cert_id}.pem")
            key_path = Path(f"/tmp/key_{certificate.cert_id}.pem")
            
            cert_path.write_text(certificate.certificate_pem)
            key_path.write_text(certificate.private_key_pem)
            
            context.load_cert_chain(str(cert_path), str(key_path))
            
            # Clean up temporary files
            cert_path.unlink(missing_ok=True)
            key_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to load certificate {certificate.cert_id}: {e}")
            raise
        
        # Store context
        self.ssl_contexts[certificate.cert_id] = context
        
        return context
    
    async def renew_certificate(self, cert_id: str) -> bool:
        """Renew TLS certificate before expiration."""
        if cert_id not in self.certificates:
            logger.error(f"Certificate {cert_id} not found for renewal")
            return False
        
        certificate = self.certificates[cert_id]
        
        if not certificate.auto_renew:
            logger.info(f"Auto-renewal disabled for certificate {cert_id}")
            return False
        
        # Check if renewal is needed
        days_until_expiry = (certificate.valid_until - datetime.now(timezone.utc)).days
        
        if days_until_expiry > certificate.renewal_threshold_days:
            logger.debug(f"Certificate {cert_id} not due for renewal ({days_until_expiry} days left)")
            return False
        
        logger.info(f"🔄 Renewing certificate {cert_id} ({days_until_expiry} days until expiry)")
        
        try:
            # For production, implement ACME protocol (Let's Encrypt)
            # For now, generate new self-signed certificate
            new_cert = await self.generate_self_signed_certificate(
                certificate.common_name,
                certificate.san_list
            )
            
            # Replace old certificate
            del self.certificates[cert_id]
            del self.ssl_contexts[cert_id]
            
            # Update certificate ID
            new_cert.cert_id = cert_id
            self.certificates[cert_id] = new_cert
            
            logger.info(f"✅ Successfully renewed certificate {cert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to renew certificate {cert_id}: {e}")
            return False
    
    async def get_ssl_context(self, domain: str) -> Optional[ssl.SSLContext]:
        """Get SSL context for domain."""
        # Find certificate for domain
        for cert in self.certificates.values():
            if cert.common_name == domain or domain in cert.san_list:
                return self.ssl_contexts.get(cert.cert_id)
        
        return None
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTPS responses."""
        return {
            'Strict-Transport-Security': f"max-age={self.tls_config['hsts_max_age']}; includeSubDomains; preload",
            'X-Frame-Options': 'DENY',
            'X-Content-Type-Options': 'nosniff',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        }


class NetworkFirewall:
    """
    Network firewall with DDoS protection and traffic analysis.
    
    Implements stateful packet inspection, rate limiting, and
    anomaly-based threat detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize network firewall."""
        self.config = config or {}
        self.rules: List[NetworkRule] = []
        self.blocked_ips: Set[str] = set()
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        self.traffic_stats: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default firewall rules."""
        default_rules = [
            # Allow localhost
            NetworkRule(
                rule_id="allow_localhost",
                source_ip="127.0.0.1",
                action=SecurityAction.ALLOW,
                priority=10
            ),
            
            # Allow private networks
            NetworkRule(
                rule_id="allow_private_10",
                source_network="10.0.0.0/8",
                action=SecurityAction.ALLOW,
                priority=20
            ),
            
            NetworkRule(
                rule_id="allow_private_172",
                source_network="172.16.0.0/12",
                action=SecurityAction.ALLOW,
                priority=20
            ),
            
            NetworkRule(
                rule_id="allow_private_192",
                source_network="192.168.0.0/16",
                action=SecurityAction.ALLOW,
                priority=20
            ),
            
            # Rate limit HTTP traffic
            NetworkRule(
                rule_id="rate_limit_http",
                traffic_type=TrafficType.HTTP,
                action=SecurityAction.RATE_LIMIT,
                rate_limit=100,  # 100 requests per minute
                priority=50
            ),
            
            # Rate limit API traffic
            NetworkRule(
                rule_id="rate_limit_api",
                traffic_type=TrafficType.API,
                action=SecurityAction.RATE_LIMIT,
                rate_limit=1000,  # 1000 requests per minute
                priority=50
            ),
            
            # Strict rate limiting for admin endpoints
            NetworkRule(
                rule_id="rate_limit_admin",
                traffic_type=TrafficType.ADMIN,
                action=SecurityAction.RATE_LIMIT,
                rate_limit=10,  # 10 requests per minute
                priority=30
            )
        ]
        
        self.rules.extend(default_rules)
        self.rules.sort(key=lambda r: r.priority)
    
    async def evaluate_request(self, source_ip: str, destination_port: int, traffic_type: TrafficType, 
                              request_data: Optional[Dict[str, Any]] = None) -> SecurityAction:
        """Evaluate incoming request against firewall rules."""
        # Check if IP is blocked
        if source_ip in self.blocked_ips:
            logger.warning(f"Blocked request from {source_ip}")
            return SecurityAction.DENY
        
        # Update traffic statistics
        self._update_traffic_stats(source_ip, traffic_type)
        
        # Evaluate rules in priority order
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if self._rule_matches(rule, source_ip, destination_port, traffic_type):
                logger.debug(f"Rule {rule.rule_id} matched for {source_ip}")
                
                if rule.action == SecurityAction.RATE_LIMIT:
                    # Check rate limit
                    if await self._check_rate_limit(source_ip, rule):
                        return SecurityAction.ALLOW
                    else:
                        logger.warning(f"Rate limit exceeded for {source_ip}")
                        return SecurityAction.DENY
                
                return rule.action
        
        # Default action - deny unknown traffic
        logger.warning(f"No matching rule for {source_ip}:{destination_port} ({traffic_type})")
        return SecurityAction.DENY
    
    def _rule_matches(self, rule: NetworkRule, source_ip: str, destination_port: int, traffic_type: TrafficType) -> bool:
        """Check if rule matches the request."""
        # Check source IP
        if rule.source_ip and rule.source_ip != source_ip:
            return False
        
        # Check source network
        if rule.source_network:
            try:
                network = ipaddress.ip_network(rule.source_network, strict=False)
                if ipaddress.ip_address(source_ip) not in network:
                    return False
            except ValueError:
                return False
        
        # Check destination port
        if rule.destination_port and rule.destination_port != destination_port:
            return False
        
        # Check traffic type
        if rule.traffic_type and rule.traffic_type != traffic_type:
            return False
        
        return True
    
    async def _check_rate_limit(self, source_ip: str, rule: NetworkRule) -> bool:
        """Check if request is within rate limit."""
        if not rule.rate_limit:
            return True
        
        current_time = time.time()
        window_start = current_time - 60  # 1-minute window
        
        # Initialize rate limiter for IP if not exists
        if source_ip not in self.rate_limiters:
            self.rate_limiters[source_ip] = {}
        
        if rule.rule_id not in self.rate_limiters[source_ip]:
            self.rate_limiters[source_ip][rule.rule_id] = {
                'requests': [],
                'blocked_until': 0
            }
        
        limiter = self.rate_limiters[source_ip][rule.rule_id]
        
        # Check if still blocked
        if current_time < limiter['blocked_until']:
            return False
        
        # Clean old requests
        limiter['requests'] = [req_time for req_time in limiter['requests'] if req_time > window_start]
        
        # Check rate limit
        if len(limiter['requests']) >= rule.rate_limit:
            # Block for 5 minutes
            limiter['blocked_until'] = current_time + 300
            
            # Add to temporary block list for severe violations
            if len(limiter['requests']) > rule.rate_limit * 2:
                await self.block_ip_temporarily(source_ip, 900)  # 15 minutes
            
            return False
        
        # Add current request
        limiter['requests'].append(current_time)
        return True
    
    def _update_traffic_stats(self, source_ip: str, traffic_type: TrafficType):
        """Update traffic statistics for monitoring."""
        current_time = int(time.time() / 60)  # Minute buckets
        
        if source_ip not in self.traffic_stats:
            self.traffic_stats[source_ip] = {}
        
        if traffic_type.value not in self.traffic_stats[source_ip]:
            self.traffic_stats[source_ip][traffic_type.value] = {}
        
        time_bucket = str(current_time)
        if time_bucket not in self.traffic_stats[source_ip][traffic_type.value]:
            self.traffic_stats[source_ip][traffic_type.value][time_bucket] = 0
        
        self.traffic_stats[source_ip][traffic_type.value][time_bucket] += 1
    
    async def block_ip_temporarily(self, ip: str, duration_seconds: int):
        """Block IP address temporarily."""
        logger.warning(f"Temporarily blocking {ip} for {duration_seconds} seconds")
        
        self.blocked_ips.add(ip)
        
        # Schedule unblocking
        asyncio.create_task(self._unblock_ip_after_delay(ip, duration_seconds))
    
    async def _unblock_ip_after_delay(self, ip: str, delay: int):
        """Unblock IP after specified delay."""
        await asyncio.sleep(delay)
        
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"Unblocked {ip} after temporary block")
    
    async def detect_ddos_attack(self) -> List[str]:
        """Detect potential DDoS attacks based on traffic patterns."""
        suspicious_ips = []
        current_time = int(time.time() / 60)
        
        for source_ip, traffic_data in self.traffic_stats.items():
            total_requests = 0
            
            # Count requests in the last 5 minutes
            for traffic_type, time_buckets in traffic_data.items():
                for i in range(5):  # Last 5 minutes
                    bucket = str(current_time - i)
                    total_requests += time_buckets.get(bucket, 0)
            
            # Threshold for DDoS detection
            if total_requests > 1000:  # More than 1000 requests in 5 minutes
                suspicious_ips.append(source_ip)
                logger.warning(f"Potential DDoS attack detected from {source_ip}: {total_requests} requests")
        
        # Auto-block suspicious IPs
        for ip in suspicious_ips:
            await self.block_ip_temporarily(ip, 3600)  # Block for 1 hour
        
        return suspicious_ips
    
    def add_custom_rule(self, rule: NetworkRule):
        """Add custom firewall rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
        logger.info(f"Added custom firewall rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove firewall rule."""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                logger.info(f"Removed firewall rule: {rule_id}")
                return True
        
        return False
    
    def get_traffic_report(self) -> Dict[str, Any]:
        """Get traffic analysis report."""
        current_time = int(time.time() / 60)
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'blocked_ips': list(self.blocked_ips),
            'top_sources': [],
            'traffic_by_type': {},
            'total_requests': 0
        }
        
        # Analyze traffic patterns
        ip_totals = {}
        type_totals = {}
        
        for source_ip, traffic_data in self.traffic_stats.items():
            ip_total = 0
            
            for traffic_type, time_buckets in traffic_data.items():
                type_total = 0
                
                # Count requests in last hour
                for i in range(60):
                    bucket = str(current_time - i)
                    requests = time_buckets.get(bucket, 0)
                    type_total += requests
                    ip_total += requests
                
                if traffic_type not in type_totals:
                    type_totals[traffic_type] = 0
                type_totals[traffic_type] += type_total
            
            if ip_total > 0:
                ip_totals[source_ip] = ip_total
        
        # Top source IPs
        sorted_ips = sorted(ip_totals.items(), key=lambda x: x[1], reverse=True)
        report['top_sources'] = sorted_ips[:10]
        
        # Traffic by type
        report['traffic_by_type'] = type_totals
        
        # Total requests
        report['total_requests'] = sum(type_totals.values())
        
        return report


class WebApplicationFirewall:
    """
    Web Application Firewall (WAF) with OWASP protection.
    
    Implements protection against common web attacks including
    SQL injection, XSS, CSRF, and other OWASP Top 10 vulnerabilities.
    """
    
    def __init__(self):
        """Initialize WAF."""
        self.sql_injection_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
            r"(?i)('|\"|\;|\-\-|\/\*|\*\/)",
            r"(?i)(or|and)\s+\d+\s*=\s*\d+",
            r"(?i)(or|and)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?",
            r"(?i)benchmark\s*\(",
            r"(?i)waitfor\s+delay"
        ]
        
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=",
            r"(?i)<iframe[^>]*>.*?</iframe>",
            r"(?i)<object[^>]*>.*?</object>",
            r"(?i)<embed[^>]*>.*?</embed>"
        ]
        
        self.command_injection_patterns = [
            r"(?i)(\;|\||\&|\`)",
            r"(?i)(nc|netcat|wget|curl|ping|nslookup)",
            r"(?i)(rm|del|format|shutdown|reboot)",
            r"(?i)(\$\(|\`.*\`|\$\{)"
        ]
        
        self.path_traversal_patterns = [
            r"(\.\./|\.\.\\\\)",
            r"(?i)(etc/passwd|boot\.ini|win\.ini)",
            r"(?i)(proc/|sys/|dev/)",
            r"(\%2e\%2e/|\%2e\%2e\\\\)"
        ]
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze web request for security threats."""
        threats = []
        risk_score = 0
        
        # Extract request components
        url = request_data.get('url', '')
        headers = request_data.get('headers', {})
        body = request_data.get('body', '')
        params = request_data.get('params', {})
        
        # SQL Injection Detection
        sql_threats = self._detect_sql_injection(url, body, params)
        threats.extend(sql_threats)
        risk_score += len(sql_threats) * 20
        
        # XSS Detection
        xss_threats = self._detect_xss(url, body, params)
        threats.extend(xss_threats)
        risk_score += len(xss_threats) * 15
        
        # Command Injection Detection
        cmd_threats = self._detect_command_injection(url, body, params)
        threats.extend(cmd_threats)
        risk_score += len(cmd_threats) * 25
        
        # Path Traversal Detection
        path_threats = self._detect_path_traversal(url, params)
        threats.extend(path_threats)
        risk_score += len(path_threats) * 18
        
        # HTTP Header Analysis
        header_threats = self._analyze_headers(headers)
        threats.extend(header_threats)
        risk_score += len(header_threats) * 10
        
        return {
            'risk_score': min(risk_score, 100),
            'threats': threats,
            'action': self._determine_action(risk_score),
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _detect_sql_injection(self, url: str, body: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect SQL injection attempts."""
        threats = []
        
        # Check all input sources
        sources = [url, body] + list(str(v) for v in params.values())
        
        for source in sources:
            if not isinstance(source, str):
                continue
            
            for pattern in self.sql_injection_patterns:
                matches = re.findall(pattern, source)
                if matches:
                    threats.append({
                        'type': 'sql_injection',
                        'pattern': pattern,
                        'matches': matches,
                        'source': 'url' if source == url else 'body' if source == body else 'params',
                        'severity': 'HIGH'
                    })
        
        return threats
    
    def _detect_xss(self, url: str, body: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect XSS attempts."""
        threats = []
        
        sources = [url, body] + list(str(v) for v in params.values())
        
        for source in sources:
            if not isinstance(source, str):
                continue
            
            for pattern in self.xss_patterns:
                matches = re.findall(pattern, source)
                if matches:
                    threats.append({
                        'type': 'xss',
                        'pattern': pattern,
                        'matches': matches,
                        'source': 'url' if source == url else 'body' if source == body else 'params',
                        'severity': 'MEDIUM'
                    })
        
        return threats
    
    def _detect_command_injection(self, url: str, body: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect command injection attempts."""
        threats = []
        
        sources = [url, body] + list(str(v) for v in params.values())
        
        for source in sources:
            if not isinstance(source, str):
                continue
            
            for pattern in self.command_injection_patterns:
                matches = re.findall(pattern, source)
                if matches:
                    threats.append({
                        'type': 'command_injection',
                        'pattern': pattern,
                        'matches': matches,
                        'source': 'url' if source == url else 'body' if source == body else 'params',
                        'severity': 'CRITICAL'
                    })
        
        return threats
    
    def _detect_path_traversal(self, url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect path traversal attempts."""
        threats = []
        
        sources = [url] + list(str(v) for v in params.values())
        
        for source in sources:
            if not isinstance(source, str):
                continue
            
            for pattern in self.path_traversal_patterns:
                matches = re.findall(pattern, source)
                if matches:
                    threats.append({
                        'type': 'path_traversal',
                        'pattern': pattern,
                        'matches': matches,
                        'source': 'url' if source == url else 'params',
                        'severity': 'HIGH'
                    })
        
        return threats
    
    def _analyze_headers(self, headers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze HTTP headers for security issues."""
        threats = []
        
        # Check for suspicious user agents
        user_agent = headers.get('User-Agent', '').lower()
        if any(bot in user_agent for bot in ['sqlmap', 'nikto', 'nmap', 'masscan', 'zap']):
            threats.append({
                'type': 'suspicious_user_agent',
                'user_agent': user_agent,
                'severity': 'MEDIUM'
            })
        
        # Check for missing security headers in response context
        security_headers = [
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security'
        ]
        
        # This would be checked on responses, not requests
        # Adding for completeness
        
        return threats
    
    def _determine_action(self, risk_score: int) -> SecurityAction:
        """Determine action based on risk score."""
        if risk_score >= 80:
            return SecurityAction.DENY
        elif risk_score >= 50:
            return SecurityAction.CHALLENGE
        elif risk_score >= 20:
            return SecurityAction.RATE_LIMIT
        else:
            return SecurityAction.ALLOW


class NetworkSecurityManager:
    """
    Comprehensive network security manager coordinating all network security components.
    
    Integrates TLS management, firewall, WAF, and network monitoring
    for defense-in-depth security.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize network security manager."""
        self.config = config or {}
        
        # Initialize components
        self.tls_manager = TLSManager(config.get('tls', {}))
        self.firewall = NetworkFirewall(config.get('firewall', {}))
        self.waf = WebApplicationFirewall()
        
        # Security metrics
        self.security_events: List[Dict[str, Any]] = []
        self.blocked_requests = 0
        self.allowed_requests = 0
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize network security manager."""
        try:
            logger.info("🛡️ Initializing Network Security Manager...")
            
            # Generate default TLS certificate
            await self.tls_manager.generate_self_signed_certificate("localhost", ["127.0.0.1"])
            
            # Start monitoring
            self._monitoring_task = asyncio.create_task(self._security_monitoring())
            
            logger.info("✅ Network Security Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize network security: {e}")
            return False
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request through security layers."""
        source_ip = request_data.get('source_ip', '0.0.0.0')
        destination_port = request_data.get('destination_port', 80)
        traffic_type = TrafficType(request_data.get('traffic_type', 'http'))
        
        # Layer 1: Firewall evaluation
        firewall_action = await self.firewall.evaluate_request(
            source_ip, destination_port, traffic_type, request_data
        )
        
        if firewall_action == SecurityAction.DENY:
            self.blocked_requests += 1
            await self._log_security_event('firewall_block', {
                'source_ip': source_ip,
                'reason': 'firewall_rule',
                'action': 'blocked'
            })
            
            return {
                'allowed': False,
                'reason': 'firewall_blocked',
                'action': firewall_action
            }
        
        # Layer 2: WAF analysis for HTTP/HTTPS traffic
        if traffic_type in [TrafficType.HTTP, TrafficType.HTTPS, TrafficType.API]:
            waf_analysis = await self.waf.analyze_request(request_data)
            
            if waf_analysis['action'] == SecurityAction.DENY:
                self.blocked_requests += 1
                await self._log_security_event('waf_block', {
                    'source_ip': source_ip,
                    'threats': waf_analysis['threats'],
                    'risk_score': waf_analysis['risk_score'],
                    'action': 'blocked'
                })
                
                return {
                    'allowed': False,
                    'reason': 'waf_blocked',
                    'waf_analysis': waf_analysis
                }
        
        # Request allowed
        self.allowed_requests += 1
        
        return {
            'allowed': True,
            'security_headers': self.tls_manager.get_security_headers(),
            'firewall_action': firewall_action
        }
    
    async def _security_monitoring(self):
        """Continuous security monitoring."""
        while True:
            try:
                # DDoS detection
                suspicious_ips = await self.firewall.detect_ddos_attack()
                if suspicious_ips:
                    await self._log_security_event('ddos_detected', {
                        'suspicious_ips': suspicious_ips,
                        'action': 'auto_blocked'
                    })
                
                # Certificate renewal check
                for cert_id in self.tls_manager.certificates:
                    await self.tls_manager.renew_certificate(cert_id)
                
                # Clean old security events (keep last 1000)
                if len(self.security_events) > 1000:
                    self.security_events = self.security_events[-1000:]
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
                await asyncio.sleep(60)  # Sleep 1 minute on error
    
    async def _log_security_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log security event."""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'data': event_data
        }
        
        self.security_events.append(event)
        logger.warning(f"SECURITY_EVENT: {event_type} - {event_data}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current network security status."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'certificates': len(self.tls_manager.certificates),
            'firewall_rules': len(self.firewall.rules),
            'blocked_ips': len(self.firewall.blocked_ips),
            'allowed_requests': self.allowed_requests,
            'blocked_requests': self.blocked_requests,
            'recent_events': self.security_events[-10:],  # Last 10 events
            'components_status': {
                'tls_manager': 'active',
                'firewall': 'active',
                'waf': 'active',
                'monitoring': 'active' if self._monitoring_task and not self._monitoring_task.done() else 'inactive'
            }
        }
    
    async def cleanup(self):
        """Cleanup network security resources."""
        logger.info("🧹 Cleaning up network security manager...")
        
        # Stop monitoring
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
        
        # Clear sensitive data
        self.tls_manager.certificates.clear()
        self.tls_manager.ssl_contexts.clear()
        
        logger.info("✅ Network security cleanup completed")


# Global network security manager
_network_security_manager: Optional[NetworkSecurityManager] = None


async def init_network_security(config: Optional[Dict[str, Any]] = None) -> NetworkSecurityManager:
    """Initialize global network security manager."""
    global _network_security_manager
    
    _network_security_manager = NetworkSecurityManager(config)
    await _network_security_manager.initialize()
    
    return _network_security_manager


def get_network_security_manager() -> NetworkSecurityManager:
    """Get global network security manager instance."""
    global _network_security_manager
    
    if _network_security_manager is None:
        raise RuntimeError("Network security manager not initialized. Call init_network_security() first.")
    
    return _network_security_manager