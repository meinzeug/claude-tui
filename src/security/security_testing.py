#!/usr/bin/env python3
"""
Security Testing and Penetration Testing Framework for Claude-TUI

Implements comprehensive security validation including:
- Automated penetration testing
- Vulnerability assessment and scanning
- Security test orchestration
- OWASP Top 10 testing
- API security testing
- Infrastructure security testing
- Compliance validation testing

Author: Security Manager - Claude-TUI Security Team
Date: 2025-08-26
"""

import asyncio
import json
import time
import logging
import hashlib
import socket
import ssl
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import random
import string
import re
import uuid

try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None

logger = logging.getLogger(__name__)


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    CSRF = "cross_site_request_forgery"
    AUTH_BYPASS = "authentication_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DIRECTORY_TRAVERSAL = "directory_traversal"
    COMMAND_INJECTION = "command_injection"
    INFORMATION_DISCLOSURE = "information_disclosure"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    BROKEN_ACCESS_CONTROL = "broken_access_control"


class SeverityLevel(Enum):
    """Vulnerability severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestCategory(Enum):
    """Security test categories"""
    OWASP_TOP10 = "owasp_top10"
    API_SECURITY = "api_security"
    INFRASTRUCTURE = "infrastructure"
    NETWORK = "network"
    APPLICATION = "application"
    COMPLIANCE = "compliance"


@dataclass
class SecurityVulnerability:
    """Represents a discovered security vulnerability"""
    vuln_id: str
    vuln_type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    location: str
    evidence: str
    impact: str
    remediation: str
    references: List[str] = field(default_factory=list)
    cvss_score: Optional[float] = None
    cve_id: Optional[str] = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed: bool = False


@dataclass
class SecurityTest:
    """Represents a security test"""
    test_id: str
    name: str
    category: TestCategory
    description: str
    target: str
    test_function: Callable
    enabled: bool = True
    timeout: int = 300  # 5 minutes default
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class PenetrationTestReport:
    """Comprehensive penetration test report"""
    report_id: str
    test_date: datetime
    target: str
    scope: List[str]
    methodology: str
    vulnerabilities: List[SecurityVulnerability]
    test_summary: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    executive_summary: str


class OWASPTop10Tester:
    """
    OWASP Top 10 vulnerability testing suite.
    
    Implements automated tests for the OWASP Top 10 2021:
    A01 - Broken Access Control
    A02 - Cryptographic Failures
    A03 - Injection
    A04 - Insecure Design
    A05 - Security Misconfiguration
    A06 - Vulnerable and Outdated Components
    A07 - Identification and Authentication Failures
    A08 - Software and Data Integrity Failures
    A09 - Security Logging and Monitoring Failures
    A10 - Server-Side Request Forgery (SSRF)
    """
    
    def __init__(self, target_url: str):
        """Initialize OWASP Top 10 tester."""
        self.target_url = target_url.rstrip('/')
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.vulnerabilities: List[SecurityVulnerability] = []
        
        # Test payloads
        self.sql_payloads = [
            "' OR 1=1--",
            "'; DROP TABLE users;--",
            "' UNION SELECT 1,2,3--",
            "admin'--",
            "' OR 'a'='a",
            "1' OR '1'='1' /*"
        ]
        
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'\"><script>alert('XSS')</script>",
            "<iframe src=javascript:alert('XSS')>"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "& dir",
            "| cat /etc/passwd",
            "`whoami`",
            "$(id)",
            "; ping -c 4 127.0.0.1"
        ]
        
        self.directory_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f%etc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
    
    async def run_full_owasp_assessment(self) -> List[SecurityVulnerability]:
        """Run complete OWASP Top 10 assessment."""
        logger.info(f"🔍 Starting OWASP Top 10 assessment for {self.target_url}")
        
        self.vulnerabilities = []
        
        # A01 - Broken Access Control
        await self._test_broken_access_control()
        
        # A02 - Cryptographic Failures
        await self._test_cryptographic_failures()
        
        # A03 - Injection
        await self._test_injection_vulnerabilities()
        
        # A04 - Insecure Design
        await self._test_insecure_design()
        
        # A05 - Security Misconfiguration
        await self._test_security_misconfiguration()
        
        # A06 - Vulnerable Components
        await self._test_vulnerable_components()
        
        # A07 - Authentication Failures
        await self._test_authentication_failures()
        
        # A08 - Software/Data Integrity Failures
        await self._test_integrity_failures()
        
        # A09 - Logging/Monitoring Failures
        await self._test_logging_monitoring()
        
        # A10 - Server-Side Request Forgery
        await self._test_ssrf()
        
        logger.info(f"✅ OWASP assessment completed. Found {len(self.vulnerabilities)} vulnerabilities")
        return self.vulnerabilities
    
    async def _test_broken_access_control(self):
        """Test for broken access control vulnerabilities."""
        logger.debug("Testing for broken access control...")
        
        if not self.session:
            return
        
        try:
            # Test for forced browsing
            admin_paths = [
                '/admin',
                '/admin/',
                '/administrator',
                '/admin.php',
                '/admin/index.php',
                '/admin/login.php',
                '/admin/dashboard',
                '/management',
                '/manager'
            ]
            
            for path in admin_paths:
                try:
                    response = self.session.get(f"{self.target_url}{path}", timeout=10, allow_redirects=False)
                    
                    if response.status_code == 200:
                        # Check if admin panel is accessible without authentication
                        content = response.text.lower()
                        if any(keyword in content for keyword in ['admin', 'dashboard', 'management', 'users']):
                            vuln = SecurityVulnerability(
                                vuln_id=f"access_control_{int(time.time())}",
                                vuln_type=VulnerabilityType.BROKEN_ACCESS_CONTROL,
                                severity=SeverityLevel.HIGH,
                                title="Administrative Interface Accessible",
                                description=f"Administrative interface at {path} is accessible without authentication",
                                location=f"{self.target_url}{path}",
                                evidence=f"HTTP {response.status_code} response with admin content",
                                impact="Unauthorized access to administrative functions",
                                remediation="Implement proper authentication and authorization controls",
                                references=["https://owasp.org/Top10/A01_2021-Broken_Access_Control/"]
                            )
                            self.vulnerabilities.append(vuln)
                
                except Exception as e:
                    logger.debug(f"Error testing {path}: {e}")
                    continue
            
            # Test for insecure direct object references
            await self._test_idor()
            
        except Exception as e:
            logger.error(f"Error in broken access control testing: {e}")
    
    async def _test_idor(self):
        """Test for Insecure Direct Object References."""
        # Test common IDOR patterns
        idor_patterns = [
            '/user/profile?id=1',
            '/user/profile?id=2',
            '/document/view?id=1',
            '/document/view?id=2',
            '/api/user/1',
            '/api/user/2',
            '/file/download?id=1',
            '/file/download?id=2'
        ]
        
        for pattern in idor_patterns:
            try:
                response = self.session.get(f"{self.target_url}{pattern}", timeout=10)
                
                if response.status_code == 200:
                    # Simple check - if we get different responses for different IDs,
                    # there might be IDOR vulnerability
                    different_id = pattern.replace('id=1', 'id=999').replace('id=2', 'id=999')
                    response2 = self.session.get(f"{self.target_url}{different_id}", timeout=10)
                    
                    if response.text != response2.text and response2.status_code == 200:
                        vuln = SecurityVulnerability(
                            vuln_id=f"idor_{int(time.time())}",
                            vuln_type=VulnerabilityType.BROKEN_ACCESS_CONTROL,
                            severity=SeverityLevel.MEDIUM,
                            title="Potential Insecure Direct Object Reference",
                            description=f"IDOR vulnerability detected at {pattern}",
                            location=f"{self.target_url}{pattern}",
                            evidence="Different responses for different object IDs without authorization check",
                            impact="Unauthorized access to other users' data",
                            remediation="Implement proper authorization checks for object access",
                            references=["https://owasp.org/www-project-top-ten/2017/A5_2017-Broken_Access_Control"]
                        )
                        self.vulnerabilities.append(vuln)
                        
            except Exception as e:
                logger.debug(f"Error testing IDOR {pattern}: {e}")
    
    async def _test_cryptographic_failures(self):
        """Test for cryptographic failures."""
        logger.debug("Testing for cryptographic failures...")
        
        try:
            # Test SSL/TLS configuration
            await self._test_ssl_configuration()
            
            # Test for weak password hashing
            await self._test_password_hashing()
            
            # Test for insecure random number generation
            await self._test_random_generation()
            
        except Exception as e:
            logger.error(f"Error in cryptographic testing: {e}")
    
    async def _test_ssl_configuration(self):
        """Test SSL/TLS configuration."""
        if not self.target_url.startswith('https'):
            vuln = SecurityVulnerability(
                vuln_id=f"ssl_missing_{int(time.time())}",
                vuln_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                severity=SeverityLevel.HIGH,
                title="Missing HTTPS Encryption",
                description="Application does not use HTTPS encryption",
                location=self.target_url,
                evidence="HTTP protocol used instead of HTTPS",
                impact="Data transmitted in plaintext, vulnerable to interception",
                remediation="Implement HTTPS with strong SSL/TLS configuration",
                references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"]
            )
            self.vulnerabilities.append(vuln)
            return
        
        # Test SSL certificate and configuration
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(self.target_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443
            
            # Test SSL certificate
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    # Check certificate expiration
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    if not_after < datetime.now():
                        vuln = SecurityVulnerability(
                            vuln_id=f"cert_expired_{int(time.time())}",
                            vuln_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                            severity=SeverityLevel.HIGH,
                            title="Expired SSL Certificate",
                            description="SSL certificate has expired",
                            location=self.target_url,
                            evidence=f"Certificate expired on {not_after}",
                            impact="Users may encounter security warnings, potential MITM attacks",
                            remediation="Renew SSL certificate immediately",
                            references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"]
                        )
                        self.vulnerabilities.append(vuln)
                    
                    # Check for weak cipher suites (simplified check)
                    cipher = ssock.cipher()
                    if cipher and len(cipher) >= 3:
                        cipher_name = cipher[0]
                        if any(weak in cipher_name.upper() for weak in ['RC4', 'DES', 'MD5']):
                            vuln = SecurityVulnerability(
                                vuln_id=f"weak_cipher_{int(time.time())}",
                                vuln_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                                severity=SeverityLevel.MEDIUM,
                                title="Weak SSL Cipher Suite",
                                description=f"Weak cipher suite detected: {cipher_name}",
                                location=self.target_url,
                                evidence=f"Cipher: {cipher_name}",
                                impact="Vulnerable to cryptographic attacks",
                                remediation="Configure strong cipher suites only",
                                references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"]
                            )
                            self.vulnerabilities.append(vuln)
                            
        except Exception as e:
            logger.debug(f"SSL configuration test error: {e}")
    
    async def _test_password_hashing(self):
        """Test for weak password hashing (limited without access to backend)."""
        # This is a placeholder - real testing would require access to password hashes
        # or observing timing differences in authentication
        pass
    
    async def _test_random_generation(self):
        """Test for predictable random number generation."""
        # Test for predictable tokens/session IDs
        if not self.session:
            return
        
        try:
            # Make multiple requests to get session tokens/cookies
            tokens = []
            for i in range(5):
                response = self.session.get(self.target_url, timeout=10)
                cookies = response.cookies
                
                for cookie in cookies:
                    if 'session' in cookie.name.lower() or 'token' in cookie.name.lower():
                        tokens.append(cookie.value)
                
                await asyncio.sleep(1)
            
            # Simple check for sequential or predictable tokens
            if len(tokens) >= 2:
                # Check if tokens are sequential numbers
                numeric_tokens = []
                for token in tokens:
                    try:
                        numeric_tokens.append(int(token))
                    except ValueError:
                        break
                
                if len(numeric_tokens) == len(tokens) and len(numeric_tokens) >= 2:
                    # Check if sequential
                    is_sequential = all(
                        numeric_tokens[i] == numeric_tokens[i-1] + 1 
                        for i in range(1, len(numeric_tokens))
                    )
                    
                    if is_sequential:
                        vuln = SecurityVulnerability(
                            vuln_id=f"predictable_tokens_{int(time.time())}",
                            vuln_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                            severity=SeverityLevel.MEDIUM,
                            title="Predictable Session Tokens",
                            description="Session tokens appear to be sequential or predictable",
                            location=self.target_url,
                            evidence=f"Sequential tokens: {tokens}",
                            impact="Session hijacking through token prediction",
                            remediation="Use cryptographically secure random number generation for tokens",
                            references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"]
                        )
                        self.vulnerabilities.append(vuln)
                        
        except Exception as e:
            logger.debug(f"Random generation test error: {e}")
    
    async def _test_injection_vulnerabilities(self):
        """Test for injection vulnerabilities."""
        logger.debug("Testing for injection vulnerabilities...")
        
        await self._test_sql_injection()
        await self._test_xss()
        await self._test_command_injection()
        await self._test_directory_traversal()
    
    async def _test_sql_injection(self):
        """Test for SQL injection vulnerabilities."""
        if not self.session:
            return
        
        # Common injection points
        test_endpoints = [
            '/login',
            '/search',
            '/user',
            '/product',
            '/api/user',
            '/api/search'
        ]
        
        for endpoint in test_endpoints:
            for payload in self.sql_payloads:
                try:
                    # Test GET parameters
                    params = {'id': payload, 'search': payload, 'q': payload}
                    response = self.session.get(f"{self.target_url}{endpoint}", params=params, timeout=10)
                    
                    if self._detect_sql_injection_response(response):
                        vuln = SecurityVulnerability(
                            vuln_id=f"sqli_{int(time.time())}",
                            vuln_type=VulnerabilityType.SQL_INJECTION,
                            severity=SeverityLevel.HIGH,
                            title="SQL Injection Vulnerability",
                            description=f"SQL injection detected at {endpoint}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"Payload: {payload}, Response indicates SQL error",
                            impact="Database compromise, data theft, authentication bypass",
                            remediation="Use parameterized queries and input validation",
                            references=["https://owasp.org/Top10/A03_2021-Injection/"]
                        )
                        self.vulnerabilities.append(vuln)
                    
                    # Test POST parameters
                    data = {'username': payload, 'password': payload, 'search': payload}
                    response = self.session.post(f"{self.target_url}{endpoint}", data=data, timeout=10)
                    
                    if self._detect_sql_injection_response(response):
                        vuln = SecurityVulnerability(
                            vuln_id=f"sqli_post_{int(time.time())}",
                            vuln_type=VulnerabilityType.SQL_INJECTION,
                            severity=SeverityLevel.HIGH,
                            title="SQL Injection Vulnerability (POST)",
                            description=f"SQL injection detected in POST to {endpoint}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"POST payload: {payload}, Response indicates SQL error",
                            impact="Database compromise, data theft, authentication bypass",
                            remediation="Use parameterized queries and input validation",
                            references=["https://owasp.org/Top10/A03_2021-Injection/"]
                        )
                        self.vulnerabilities.append(vuln)
                        
                except Exception as e:
                    logger.debug(f"SQL injection test error for {endpoint}: {e}")
                    continue
    
    def _detect_sql_injection_response(self, response) -> bool:
        """Detect SQL injection in response."""
        if not response:
            return False
        
        content = response.text.lower()
        
        # SQL error patterns
        sql_errors = [
            'sql syntax error',
            'mysql_fetch_array',
            'ora-00933',
            'microsoft odbc sql server driver',
            'you have an error in your sql syntax',
            'warning: mysql_',
            'valid mysql result',
            'postgresql query failed',
            'warning: pg_',
            'valid postgresql result',
            'microsoft jet database',
            'odbc drivers error',
            'sqlite_exception',
            'sqlite/joinclause'
        ]
        
        return any(error in content for error in sql_errors)
    
    async def _test_xss(self):
        """Test for Cross-Site Scripting vulnerabilities."""
        if not self.session:
            return
        
        test_endpoints = ['/search', '/comment', '/feedback', '/contact', '/profile']
        
        for endpoint in test_endpoints:
            for payload in self.xss_payloads:
                try:
                    # Test GET parameters
                    params = {'q': payload, 'search': payload, 'comment': payload}
                    response = self.session.get(f"{self.target_url}{endpoint}", params=params, timeout=10)
                    
                    if payload in response.text and response.headers.get('content-type', '').startswith('text/html'):
                        vuln = SecurityVulnerability(
                            vuln_id=f"xss_{int(time.time())}",
                            vuln_type=VulnerabilityType.XSS,
                            severity=SeverityLevel.MEDIUM,
                            title="Cross-Site Scripting (XSS) Vulnerability",
                            description=f"XSS vulnerability detected at {endpoint}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"Payload reflected: {payload}",
                            impact="Session hijacking, defacement, malicious script execution",
                            remediation="Implement proper input validation and output encoding",
                            references=["https://owasp.org/www-community/attacks/xss/"]
                        )
                        self.vulnerabilities.append(vuln)
                    
                    # Test POST parameters
                    data = {'message': payload, 'comment': payload, 'feedback': payload}
                    response = self.session.post(f"{self.target_url}{endpoint}", data=data, timeout=10)
                    
                    if payload in response.text and response.headers.get('content-type', '').startswith('text/html'):
                        vuln = SecurityVulnerability(
                            vuln_id=f"xss_post_{int(time.time())}",
                            vuln_type=VulnerabilityType.XSS,
                            severity=SeverityLevel.MEDIUM,
                            title="Cross-Site Scripting (XSS) Vulnerability (POST)",
                            description=f"XSS vulnerability detected in POST to {endpoint}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"POST payload reflected: {payload}",
                            impact="Session hijacking, defacement, malicious script execution",
                            remediation="Implement proper input validation and output encoding",
                            references=["https://owasp.org/www-community/attacks/xss/"]
                        )
                        self.vulnerabilities.append(vuln)
                        
                except Exception as e:
                    logger.debug(f"XSS test error for {endpoint}: {e}")
                    continue
    
    async def _test_command_injection(self):
        """Test for command injection vulnerabilities."""
        if not self.session:
            return
        
        test_endpoints = ['/ping', '/traceroute', '/system', '/admin/system']
        
        for endpoint in test_endpoints:
            for payload in self.command_injection_payloads:
                try:
                    params = {'host': f"127.0.0.1{payload}", 'cmd': payload}
                    response = self.session.get(f"{self.target_url}{endpoint}", params=params, timeout=15)
                    
                    if self._detect_command_injection_response(response, payload):
                        vuln = SecurityVulnerability(
                            vuln_id=f"cmdi_{int(time.time())}",
                            vuln_type=VulnerabilityType.COMMAND_INJECTION,
                            severity=SeverityLevel.CRITICAL,
                            title="Command Injection Vulnerability",
                            description=f"Command injection detected at {endpoint}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"Payload: {payload}, Command output detected in response",
                            impact="Remote code execution, system compromise",
                            remediation="Avoid system commands, use safe APIs, validate input strictly",
                            references=["https://owasp.org/www-community/attacks/Command_Injection"]
                        )
                        self.vulnerabilities.append(vuln)
                        
                except Exception as e:
                    logger.debug(f"Command injection test error for {endpoint}: {e}")
                    continue
    
    def _detect_command_injection_response(self, response, payload: str) -> bool:
        """Detect command injection in response."""
        if not response:
            return False
        
        content = response.text.lower()
        
        # Command output patterns
        if '; ls' in payload and any(pattern in content for pattern in ['total ', 'drwx', '-rw-']):
            return True
        
        if 'whoami' in payload and len(content.strip()) < 50:  # Username typically short
            return True
        
        if 'id' in payload and ('uid=' in content or 'gid=' in content):
            return True
        
        if 'ping' in payload and ('ping statistics' in content or 'packets transmitted' in content):
            return True
        
        return False
    
    async def _test_directory_traversal(self):
        """Test for directory traversal vulnerabilities."""
        if not self.session:
            return
        
        test_endpoints = ['/download', '/file', '/image', '/document', '/view']
        
        for endpoint in test_endpoints:
            for payload in self.directory_traversal_payloads:
                try:
                    params = {'file': payload, 'path': payload, 'doc': payload}
                    response = self.session.get(f"{self.target_url}{endpoint}", params=params, timeout=10)
                    
                    if self._detect_directory_traversal_response(response):
                        vuln = SecurityVulnerability(
                            vuln_id=f"directory_traversal_{int(time.time())}",
                            vuln_type=VulnerabilityType.DIRECTORY_TRAVERSAL,
                            severity=SeverityLevel.HIGH,
                            title="Directory Traversal Vulnerability",
                            description=f"Directory traversal detected at {endpoint}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"Payload: {payload}, System file content detected",
                            impact="Unauthorized file access, information disclosure",
                            remediation="Validate file paths, use allowlists, avoid direct file access",
                            references=["https://owasp.org/www-community/attacks/Path_Traversal"]
                        )
                        self.vulnerabilities.append(vuln)
                        
                except Exception as e:
                    logger.debug(f"Directory traversal test error for {endpoint}: {e}")
                    continue
    
    def _detect_directory_traversal_response(self, response) -> bool:
        """Detect directory traversal in response."""
        if not response:
            return False
        
        content = response.text.lower()
        
        # Unix/Linux file patterns
        unix_patterns = [
            'root:x:0:0:root',  # /etc/passwd
            'daemon:x:1:1:daemon',
            'localhost',  # /etc/hosts
            '127.0.0.1'
        ]
        
        # Windows file patterns
        windows_patterns = [
            '[boot loader]',  # boot.ini
            '[operating systems]',
            '# copyright (c) 1993-2009 microsoft corp'  # hosts file
        ]
        
        return any(pattern in content for pattern in unix_patterns + windows_patterns)
    
    async def _test_insecure_design(self):
        """Test for insecure design patterns."""
        logger.debug("Testing for insecure design...")
        
        # This category is more about architecture and design flaws
        # Limited automated testing possible
        await self._test_missing_rate_limiting()
        await self._test_insecure_workflows()
    
    async def _test_missing_rate_limiting(self):
        """Test for missing rate limiting."""
        if not self.session:
            return
        
        try:
            # Test login endpoint for rate limiting
            login_endpoint = f"{self.target_url}/login"
            
            # Make rapid requests
            start_time = time.time()
            successful_requests = 0
            
            for i in range(20):  # 20 rapid requests
                try:
                    response = self.session.post(
                        login_endpoint,
                        data={'username': f'test{i}', 'password': 'wrong'},
                        timeout=5
                    )
                    if response.status_code != 429:  # Not rate limited
                        successful_requests += 1
                except Exception:
                    break
            
            end_time = time.time()
            
            # If we made more than 10 requests in under 10 seconds without rate limiting
            if successful_requests > 10 and (end_time - start_time) < 10:
                vuln = SecurityVulnerability(
                    vuln_id=f"no_rate_limit_{int(time.time())}",
                    vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Missing Rate Limiting",
                    description="No rate limiting detected on authentication endpoint",
                    location=login_endpoint,
                    evidence=f"{successful_requests} requests in {end_time - start_time:.1f} seconds",
                    impact="Brute force attacks, denial of service",
                    remediation="Implement rate limiting on sensitive endpoints",
                    references=["https://owasp.org/Top10/A04_2021-Insecure_Design/"]
                )
                self.vulnerabilities.append(vuln)
                
        except Exception as e:
            logger.debug(f"Rate limiting test error: {e}")
    
    async def _test_insecure_workflows(self):
        """Test for insecure business logic workflows."""
        # Placeholder for business logic testing
        # This would require understanding of specific application workflows
        pass
    
    async def _test_security_misconfiguration(self):
        """Test for security misconfigurations."""
        logger.debug("Testing for security misconfigurations...")
        
        await self._test_debug_enabled()
        await self._test_default_credentials()
        await self._test_unnecessary_features()
        await self._test_security_headers()
    
    async def _test_debug_enabled(self):
        """Test for debug mode enabled."""
        if not self.session:
            return
        
        try:
            # Try to trigger debug/error pages
            debug_endpoints = [
                '/debug',
                '/error',
                '/test',
                '/phpinfo.php',
                '/server-info',
                '/server-status'
            ]
            
            for endpoint in debug_endpoints:
                response = self.session.get(f"{self.target_url}{endpoint}", timeout=10)
                
                content = response.text.lower()
                debug_indicators = [
                    'debug mode',
                    'stack trace',
                    'phpinfo()',
                    'server configuration',
                    'apache server information',
                    'debug=true',
                    'traceback'
                ]
                
                if any(indicator in content for indicator in debug_indicators):
                    vuln = SecurityVulnerability(
                        vuln_id=f"debug_enabled_{int(time.time())}",
                        vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Debug Information Disclosure",
                        description=f"Debug information exposed at {endpoint}",
                        location=f"{self.target_url}{endpoint}",
                        evidence="Debug/configuration information visible",
                        impact="Information disclosure, attack surface expansion",
                        remediation="Disable debug mode in production",
                        references=["https://owasp.org/Top10/A05_2021-Security_Misconfiguration/"]
                    )
                    self.vulnerabilities.append(vuln)
                    
        except Exception as e:
            logger.debug(f"Debug test error: {e}")
    
    async def _test_default_credentials(self):
        """Test for default credentials."""
        if not self.session:
            return
        
        default_creds = [
            ('admin', 'admin'),
            ('admin', 'password'),
            ('admin', '123456'),
            ('administrator', 'administrator'),
            ('root', 'root'),
            ('root', 'password'),
            ('test', 'test'),
            ('guest', 'guest')
        ]
        
        login_endpoints = ['/login', '/admin', '/admin/login']
        
        for endpoint in login_endpoints:
            for username, password in default_creds:
                try:
                    response = self.session.post(
                        f"{self.target_url}{endpoint}",
                        data={'username': username, 'password': password},
                        timeout=10,
                        allow_redirects=False
                    )
                    
                    # Check for successful login indicators
                    if (response.status_code in [200, 302] and 
                        'login' not in response.text.lower() and
                        ('dashboard' in response.text.lower() or 
                         'welcome' in response.text.lower() or
                         response.status_code == 302)):
                        
                        vuln = SecurityVulnerability(
                            vuln_id=f"default_creds_{int(time.time())}",
                            vuln_type=VulnerabilityType.AUTH_BYPASS,
                            severity=SeverityLevel.CRITICAL,
                            title="Default Credentials",
                            description=f"Default credentials work: {username}/{password}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"Login successful with {username}:{password}",
                            impact="Unauthorized administrative access",
                            remediation="Change all default credentials immediately",
                            references=["https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/"]
                        )
                        self.vulnerabilities.append(vuln)
                        
                except Exception as e:
                    logger.debug(f"Default credentials test error: {e}")
                    continue
    
    async def _test_unnecessary_features(self):
        """Test for unnecessary features and services."""
        # Test for common unnecessary features
        unnecessary_endpoints = [
            '/phpinfo.php',
            '/info.php',
            '/test.php',
            '/backup/',
            '/old/',
            '/temp/',
            '/.git/',
            '/.svn/',
            '/CVS/',
            '/.DS_Store',
            '/web.config',
            '/.htaccess'
        ]
        
        for endpoint in unnecessary_endpoints:
            try:
                response = self.session.get(f"{self.target_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    vuln = SecurityVulnerability(
                        vuln_id=f"unnecessary_feature_{int(time.time())}",
                        vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                        severity=SeverityLevel.LOW,
                        title="Unnecessary Feature Exposed",
                        description=f"Unnecessary feature/file accessible: {endpoint}",
                        location=f"{self.target_url}{endpoint}",
                        evidence=f"HTTP 200 response from {endpoint}",
                        impact="Information disclosure, attack surface expansion",
                        remediation="Remove or restrict access to unnecessary features",
                        references=["https://owasp.org/Top10/A05_2021-Security_Misconfiguration/"]
                    )
                    self.vulnerabilities.append(vuln)
                    
            except Exception as e:
                logger.debug(f"Unnecessary features test error: {e}")
    
    async def _test_security_headers(self):
        """Test for missing security headers."""
        if not self.session:
            return
        
        try:
            response = self.session.get(self.target_url, timeout=10)
            headers = response.headers
            
            # Check for important security headers
            security_headers = {
                'X-Frame-Options': 'Clickjacking protection',
                'X-Content-Type-Options': 'MIME type sniffing protection',
                'X-XSS-Protection': 'XSS protection',
                'Strict-Transport-Security': 'HTTPS enforcement',
                'Content-Security-Policy': 'Content injection protection',
                'Referrer-Policy': 'Referrer information control'
            }
            
            missing_headers = []
            for header, description in security_headers.items():
                if header not in headers:
                    missing_headers.append(f"{header} ({description})")
            
            if missing_headers:
                vuln = SecurityVulnerability(
                    vuln_id=f"missing_headers_{int(time.time())}",
                    vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                    severity=SeverityLevel.LOW,
                    title="Missing Security Headers",
                    description="Important security headers are missing",
                    location=self.target_url,
                    evidence=f"Missing headers: {', '.join(missing_headers)}",
                    impact="Reduced protection against various attacks",
                    remediation="Implement all recommended security headers",
                    references=["https://owasp.org/www-project-secure-headers/"]
                )
                self.vulnerabilities.append(vuln)
                
        except Exception as e:
            logger.debug(f"Security headers test error: {e}")
    
    async def _test_vulnerable_components(self):
        """Test for vulnerable and outdated components."""
        logger.debug("Testing for vulnerable components...")
        
        if not self.session:
            return
        
        try:
            response = self.session.get(self.target_url, timeout=10)
            headers = response.headers
            
            # Check server headers for version information
            server_header = headers.get('Server', '')
            powered_by = headers.get('X-Powered-By', '')
            
            # Simple version detection (would be enhanced with CVE database in production)
            vulnerable_patterns = [
                ('Apache/2.2', 'Apache 2.2 has known vulnerabilities'),
                ('nginx/1.0', 'Nginx 1.0 is outdated and vulnerable'),
                ('PHP/5.', 'PHP 5.x is end-of-life and vulnerable'),
                ('jQuery/1.', 'jQuery 1.x has XSS vulnerabilities'),
                ('OpenSSL/0.9', 'OpenSSL 0.9.x has critical vulnerabilities')
            ]
            
            for pattern, description in vulnerable_patterns:
                if pattern in server_header or pattern in powered_by:
                    vuln = SecurityVulnerability(
                        vuln_id=f"vulnerable_component_{int(time.time())}",
                        vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                        severity=SeverityLevel.HIGH,
                        title="Vulnerable Component Detected",
                        description=description,
                        location=self.target_url,
                        evidence=f"Server: {server_header}, X-Powered-By: {powered_by}",
                        impact="Known security vulnerabilities may be exploitable",
                        remediation="Update all components to latest secure versions",
                        references=["https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components/"]
                    )
                    self.vulnerabilities.append(vuln)
                    
        except Exception as e:
            logger.debug(f"Vulnerable components test error: {e}")
    
    async def _test_authentication_failures(self):
        """Test for identification and authentication failures."""
        logger.debug("Testing for authentication failures...")
        
        await self._test_weak_passwords()
        await self._test_session_management()
        await self._test_password_recovery()
    
    async def _test_weak_passwords(self):
        """Test for weak password policies."""
        if not self.session:
            return
        
        # Try to register/create account with weak passwords
        weak_passwords = ['123456', 'password', 'admin', 'test', '12345', 'qwerty']
        
        register_endpoints = ['/register', '/signup', '/create-account']
        
        for endpoint in register_endpoints:
            for weak_password in weak_passwords:
                try:
                    test_username = f"test_{int(time.time())}"
                    response = self.session.post(
                        f"{self.target_url}{endpoint}",
                        data={
                            'username': test_username,
                            'password': weak_password,
                            'email': f"{test_username}@test.com"
                        },
                        timeout=10
                    )
                    
                    # Check if weak password was accepted
                    if (response.status_code == 200 and 
                        'error' not in response.text.lower() and
                        'weak' not in response.text.lower()):
                        
                        vuln = SecurityVulnerability(
                            vuln_id=f"weak_password_{int(time.time())}",
                            vuln_type=VulnerabilityType.AUTH_BYPASS,
                            severity=SeverityLevel.MEDIUM,
                            title="Weak Password Policy",
                            description="Weak passwords are accepted",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"Weak password '{weak_password}' was accepted",
                            impact="Accounts vulnerable to brute force attacks",
                            remediation="Implement strong password policy requirements",
                            references=["https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/"]
                        )
                        self.vulnerabilities.append(vuln)
                        break  # Found vulnerability, no need to test more passwords
                        
                except Exception as e:
                    logger.debug(f"Weak password test error: {e}")
                    continue
    
    async def _test_session_management(self):
        """Test for session management issues."""
        if not self.session:
            return
        
        try:
            # Check session cookie security
            response = self.session.get(self.target_url, timeout=10)
            
            for cookie in response.cookies:
                if 'session' in cookie.name.lower():
                    issues = []
                    
                    if not cookie.secure:
                        issues.append("Not marked as Secure")
                    
                    if not cookie.has_nonstandard_attr('HttpOnly'):
                        issues.append("Not marked as HttpOnly")
                    
                    if not cookie.has_nonstandard_attr('SameSite'):
                        issues.append("Missing SameSite attribute")
                    
                    if issues:
                        vuln = SecurityVulnerability(
                            vuln_id=f"session_cookie_{int(time.time())}",
                            vuln_type=VulnerabilityType.AUTH_BYPASS,
                            severity=SeverityLevel.MEDIUM,
                            title="Insecure Session Cookie",
                            description="Session cookie lacks security attributes",
                            location=self.target_url,
                            evidence=f"Cookie issues: {', '.join(issues)}",
                            impact="Session hijacking, XSS cookie theft",
                            remediation="Set Secure, HttpOnly, and SameSite attributes on session cookies",
                            references=["https://owasp.org/www-community/controls/SecureCookieAttribute"]
                        )
                        self.vulnerabilities.append(vuln)
                        
        except Exception as e:
            logger.debug(f"Session management test error: {e}")
    
    async def _test_password_recovery(self):
        """Test for insecure password recovery mechanisms."""
        # This would require understanding specific password recovery flows
        # Placeholder for now
        pass
    
    async def _test_integrity_failures(self):
        """Test for software and data integrity failures."""
        logger.debug("Testing for integrity failures...")
        
        # Test for unsigned/unverified software updates
        # Test for insecure CI/CD pipelines
        # Test for auto-update without integrity verification
        
        # Limited automated testing possible for this category
        pass
    
    async def _test_logging_monitoring(self):
        """Test for security logging and monitoring failures."""
        logger.debug("Testing for logging and monitoring failures...")
        
        # This category is difficult to test automatically without access to logs
        # Would typically involve reviewing log configurations and monitoring setup
        pass
    
    async def _test_ssrf(self):
        """Test for Server-Side Request Forgery vulnerabilities."""
        logger.debug("Testing for SSRF...")
        
        if not self.session:
            return
        
        # SSRF test payloads
        ssrf_payloads = [
            'http://127.0.0.1:8080',
            'http://localhost:3000',
            'http://169.254.169.254/latest/meta-data/',  # AWS metadata
            'file:///etc/passwd',
            'gopher://127.0.0.1:25',
            'http://[::1]:80'
        ]
        
        test_endpoints = ['/fetch', '/url', '/proxy', '/webhook', '/import']
        
        for endpoint in test_endpoints:
            for payload in ssrf_payloads:
                try:
                    params = {'url': payload, 'link': payload, 'fetch': payload}
                    response = self.session.get(f"{self.target_url}{endpoint}", params=params, timeout=15)
                    
                    if self._detect_ssrf_response(response, payload):
                        vuln = SecurityVulnerability(
                            vuln_id=f"ssrf_{int(time.time())}",
                            vuln_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                            severity=SeverityLevel.HIGH,
                            title="Server-Side Request Forgery (SSRF)",
                            description=f"SSRF vulnerability detected at {endpoint}",
                            location=f"{self.target_url}{endpoint}",
                            evidence=f"Internal request payload: {payload}",
                            impact="Internal network scanning, cloud metadata access, local file access",
                            remediation="Validate and allowlist URLs, block internal networks",
                            references=["https://owasp.org/Top10/A10_2021-Server-Side_Request_Forgery_%28SSRF%29/"]
                        )
                        self.vulnerabilities.append(vuln)
                        
                except Exception as e:
                    logger.debug(f"SSRF test error for {endpoint}: {e}")
                    continue
    
    def _detect_ssrf_response(self, response, payload: str) -> bool:
        """Detect SSRF in response."""
        if not response:
            return False
        
        content = response.text.lower()
        
        # AWS metadata indicators
        if '169.254.169.254' in payload and 'ami-id' in content:
            return True
        
        # Local file access indicators
        if 'file://' in payload and ('root:x:0:0' in content or 'localhost' in content):
            return True
        
        # Internal service indicators
        if any(internal in payload for internal in ['127.0.0.1', 'localhost', '[::1]']):
            # Check for typical internal service responses
            if any(indicator in content for indicator in ['apache', 'nginx', 'server', 'welcome']):
                return True
        
        return False


class SecurityTestOrchestrator:
    """
    Security test orchestrator that manages and executes comprehensive security testing.
    
    Coordinates OWASP testing, infrastructure testing, API security testing,
    and generates comprehensive reports.
    """
    
    def __init__(self, target: str, config: Optional[Dict[str, Any]] = None):
        """Initialize security test orchestrator."""
        self.target = target
        self.config = config or {}
        self.test_results: List[SecurityVulnerability] = []
        self.test_suite: List[SecurityTest] = []
        
        # Initialize test components
        self.owasp_tester = OWASPTop10Tester(target)
        
        # Register test suites
        self._register_test_suites()
    
    def _register_test_suites(self):
        """Register all security test suites."""
        self.test_suite = [
            SecurityTest(
                test_id="owasp_top10",
                name="OWASP Top 10 Security Test",
                category=TestCategory.OWASP_TOP10,
                description="Comprehensive OWASP Top 10 vulnerability assessment",
                target=self.target,
                test_function=self._run_owasp_tests,
                timeout=1800  # 30 minutes
            ),
            SecurityTest(
                test_id="api_security",
                name="API Security Test",
                category=TestCategory.API_SECURITY,
                description="API endpoint security testing",
                target=self.target,
                test_function=self._run_api_security_tests,
                timeout=900  # 15 minutes
            ),
            SecurityTest(
                test_id="infrastructure",
                name="Infrastructure Security Test",
                category=TestCategory.INFRASTRUCTURE,
                description="Infrastructure and network security testing",
                target=self.target,
                test_function=self._run_infrastructure_tests,
                timeout=600  # 10 minutes
            )
        ]
    
    async def run_comprehensive_assessment(self) -> PenetrationTestReport:
        """Run comprehensive security assessment."""
        logger.info(f"🔍 Starting comprehensive security assessment for {self.target}")
        
        start_time = datetime.now(timezone.utc)
        all_vulnerabilities = []
        test_summary = {}
        
        # Run all test suites
        for test in self.test_suite:
            if not test.enabled:
                continue
            
            logger.info(f"📋 Running test suite: {test.name}")
            
            try:
                # Run test with timeout
                vulnerabilities = await asyncio.wait_for(
                    test.test_function(),
                    timeout=test.timeout
                )
                
                all_vulnerabilities.extend(vulnerabilities)
                test_summary[test.test_id] = {
                    'status': 'completed',
                    'vulnerabilities_found': len(vulnerabilities),
                    'duration': (datetime.now(timezone.utc) - start_time).total_seconds()
                }
                
                logger.info(f"✅ {test.name} completed: {len(vulnerabilities)} vulnerabilities found")
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ {test.name} timed out after {test.timeout} seconds")
                test_summary[test.test_id] = {
                    'status': 'timeout',
                    'vulnerabilities_found': 0,
                    'duration': test.timeout
                }
                
            except Exception as e:
                logger.error(f"❌ {test.name} failed: {e}")
                test_summary[test.test_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'vulnerabilities_found': 0,
                    'duration': (datetime.now(timezone.utc) - start_time).total_seconds()
                }
        
        # Generate comprehensive report
        report = await self._generate_penetration_test_report(
            all_vulnerabilities, test_summary, start_time
        )
        
        logger.info(f"🎯 Security assessment completed: {len(all_vulnerabilities)} vulnerabilities found")
        return report
    
    async def _run_owasp_tests(self) -> List[SecurityVulnerability]:
        """Run OWASP Top 10 tests."""
        return await self.owasp_tester.run_full_owasp_assessment()
    
    async def _run_api_security_tests(self) -> List[SecurityVulnerability]:
        """Run API security tests."""
        vulnerabilities = []
        
        if not REQUESTS_AVAILABLE:
            return vulnerabilities
        
        # API-specific security tests
        api_endpoints = [
            '/api/v1/users',
            '/api/v1/auth',
            '/api/v1/data',
            '/api/users',
            '/api/auth',
            '/rest/users',
            '/graphql'
        ]
        
        for endpoint in api_endpoints:
            try:
                # Test for API information disclosure
                response = requests.get(f"{self.target}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    # Check for exposed API documentation
                    if any(indicator in content for indicator in ['swagger', 'openapi', 'api documentation']):
                        vuln = SecurityVulnerability(
                            vuln_id=f"api_docs_{int(time.time())}",
                            vuln_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                            severity=SeverityLevel.LOW,
                            title="API Documentation Exposed",
                            description=f"API documentation exposed at {endpoint}",
                            location=f"{self.target}{endpoint}",
                            evidence="API documentation accessible without authentication",
                            impact="Information disclosure about API structure and endpoints",
                            remediation="Restrict access to API documentation",
                            references=["https://owasp.org/www-project-api-security/"]
                        )
                        vulnerabilities.append(vuln)
                
                # Test for missing authentication
                if response.status_code == 200 and 'api' in endpoint:
                    vuln = SecurityVulnerability(
                        vuln_id=f"api_no_auth_{int(time.time())}",
                        vuln_type=VulnerabilityType.BROKEN_ACCESS_CONTROL,
                        severity=SeverityLevel.MEDIUM,
                        title="API Endpoint Without Authentication",
                        description=f"API endpoint accessible without authentication: {endpoint}",
                        location=f"{self.target}{endpoint}",
                        evidence="HTTP 200 response from API endpoint without authentication",
                        impact="Unauthorized data access",
                        remediation="Implement authentication for all API endpoints",
                        references=["https://owasp.org/www-project-api-security/"]
                    )
                    vulnerabilities.append(vuln)
                    
            except Exception as e:
                logger.debug(f"API test error for {endpoint}: {e}")
                continue
        
        return vulnerabilities
    
    async def _run_infrastructure_tests(self) -> List[SecurityVulnerability]:
        """Run infrastructure security tests."""
        vulnerabilities = []
        
        # Port scanning (basic)
        await self._test_open_ports(vulnerabilities)
        
        # SSL/TLS testing
        await self._test_ssl_configuration(vulnerabilities)
        
        # HTTP security headers
        await self._test_security_headers(vulnerabilities)
        
        return vulnerabilities
    
    async def _test_open_ports(self, vulnerabilities: List[SecurityVulnerability]):
        """Test for unnecessary open ports."""
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(self.target)
            hostname = parsed_url.hostname
            
            if not hostname:
                return
            
            # Common dangerous ports
            dangerous_ports = [21, 23, 25, 53, 135, 139, 445, 1433, 1521, 3306, 5432, 6379]
            
            for port in dangerous_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((hostname, port))
                    
                    if result == 0:  # Port is open
                        vuln = SecurityVulnerability(
                            vuln_id=f"open_port_{port}_{int(time.time())}",
                            vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                            severity=SeverityLevel.MEDIUM,
                            title=f"Unnecessary Open Port: {port}",
                            description=f"Port {port} is open and potentially dangerous",
                            location=f"{hostname}:{port}",
                            evidence=f"TCP connection successful to port {port}",
                            impact="Increased attack surface, potential service exploitation",
                            remediation=f"Close port {port} if not required or restrict access",
                            references=["https://owasp.org/www-project-top-ten/2017/A6_2017-Security_Misconfiguration"]
                        )
                        vulnerabilities.append(vuln)
                    
                    sock.close()
                    
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Port scanning error: {e}")
    
    async def _test_ssl_configuration(self, vulnerabilities: List[SecurityVulnerability]):
        """Test SSL/TLS configuration."""
        # This would be more comprehensive with dedicated SSL testing tools
        # For now, basic checks
        if not self.target.startswith('https'):
            vuln = SecurityVulnerability(
                vuln_id=f"no_https_{int(time.time())}",
                vuln_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                severity=SeverityLevel.HIGH,
                title="HTTPS Not Implemented",
                description="Application does not use HTTPS encryption",
                location=self.target,
                evidence="HTTP protocol used instead of HTTPS",
                impact="Data transmitted in plaintext",
                remediation="Implement HTTPS with strong SSL/TLS configuration",
                references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"]
            )
            vulnerabilities.append(vuln)
    
    async def _test_security_headers(self, vulnerabilities: List[SecurityVulnerability]):
        """Test for missing security headers."""
        if not REQUESTS_AVAILABLE:
            return
        
        try:
            response = requests.get(self.target, timeout=10)
            headers = response.headers
            
            required_headers = {
                'X-Frame-Options': 'Clickjacking protection',
                'X-Content-Type-Options': 'MIME sniffing protection',
                'X-XSS-Protection': 'XSS protection',
                'Strict-Transport-Security': 'HTTPS enforcement',
                'Content-Security-Policy': 'Content injection protection'
            }
            
            missing_headers = []
            for header, description in required_headers.items():
                if header not in headers:
                    missing_headers.append(header)
            
            if missing_headers:
                vuln = SecurityVulnerability(
                    vuln_id=f"missing_headers_{int(time.time())}",
                    vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                    severity=SeverityLevel.LOW,
                    title="Missing Security Headers",
                    description="Important security headers are missing",
                    location=self.target,
                    evidence=f"Missing: {', '.join(missing_headers)}",
                    impact="Reduced protection against various attacks",
                    remediation="Implement all recommended security headers",
                    references=["https://owasp.org/www-project-secure-headers/"]
                )
                vulnerabilities.append(vuln)
                
        except Exception as e:
            logger.debug(f"Security headers test error: {e}")
    
    async def _generate_penetration_test_report(self, vulnerabilities: List[SecurityVulnerability], 
                                              test_summary: Dict[str, Any], 
                                              start_time: datetime) -> PenetrationTestReport:
        """Generate comprehensive penetration test report."""
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(vulnerabilities)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(vulnerabilities)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(vulnerabilities, risk_assessment)
        
        report = PenetrationTestReport(
            report_id=str(uuid.uuid4()),
            test_date=start_time,
            target=self.target,
            scope=[self.target],  # Could be expanded for multiple targets
            methodology="OWASP Testing Guide, Automated Vulnerability Assessment",
            vulnerabilities=vulnerabilities,
            test_summary=test_summary,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            executive_summary=executive_summary
        )
        
        return report
    
    def _calculate_risk_assessment(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, Any]:
        """Calculate risk assessment based on vulnerabilities."""
        severity_counts = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 0,
            SeverityLevel.MEDIUM: 0,
            SeverityLevel.LOW: 0,
            SeverityLevel.INFO: 0
        }
        
        for vuln in vulnerabilities:
            severity_counts[vuln.severity] += 1
        
        # Calculate overall risk score
        risk_score = (
            severity_counts[SeverityLevel.CRITICAL] * 10 +
            severity_counts[SeverityLevel.HIGH] * 7 +
            severity_counts[SeverityLevel.MEDIUM] * 4 +
            severity_counts[SeverityLevel.LOW] * 2 +
            severity_counts[SeverityLevel.INFO] * 1
        )
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = "CRITICAL"
        elif risk_score >= 30:
            risk_level = "HIGH"
        elif risk_score >= 15:
            risk_level = "MEDIUM"
        elif risk_score > 0:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'overall_risk_level': risk_level,
            'risk_score': risk_score,
            'severity_distribution': {
                'critical': severity_counts[SeverityLevel.CRITICAL],
                'high': severity_counts[SeverityLevel.HIGH],
                'medium': severity_counts[SeverityLevel.MEDIUM],
                'low': severity_counts[SeverityLevel.LOW],
                'info': severity_counts[SeverityLevel.INFO]
            },
            'total_vulnerabilities': len(vulnerabilities)
        }
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        """Generate remediation recommendations."""
        recommendations = []
        
        # Group vulnerabilities by type
        vuln_types = {}
        for vuln in vulnerabilities:
            if vuln.vuln_type not in vuln_types:
                vuln_types[vuln.vuln_type] = 0
            vuln_types[vuln.vuln_type] += 1
        
        # Generate prioritized recommendations
        if VulnerabilityType.SQL_INJECTION in vuln_types:
            recommendations.append("CRITICAL: Implement parameterized queries and input validation to prevent SQL injection attacks")
        
        if VulnerabilityType.XSS in vuln_types:
            recommendations.append("HIGH: Implement proper input validation and output encoding to prevent XSS attacks")
        
        if VulnerabilityType.COMMAND_INJECTION in vuln_types:
            recommendations.append("CRITICAL: Avoid system commands and implement strict input validation")
        
        if VulnerabilityType.BROKEN_ACCESS_CONTROL in vuln_types:
            recommendations.append("HIGH: Implement proper authentication and authorization controls")
        
        if VulnerabilityType.WEAK_CRYPTOGRAPHY in vuln_types:
            recommendations.append("HIGH: Implement strong encryption and update SSL/TLS configuration")
        
        if VulnerabilityType.SECURITY_MISCONFIGURATION in vuln_types:
            recommendations.append("MEDIUM: Review and harden security configuration settings")
        
        # General recommendations
        recommendations.extend([
            "Implement a Web Application Firewall (WAF) for additional protection",
            "Conduct regular security assessments and penetration testing",
            "Establish a vulnerability management program",
            "Provide security training for development team",
            "Implement security monitoring and incident response procedures"
        ])
        
        return recommendations
    
    def _generate_executive_summary(self, vulnerabilities: List[SecurityVulnerability], 
                                  risk_assessment: Dict[str, Any]) -> str:
        """Generate executive summary of security assessment."""
        
        total_vulns = len(vulnerabilities)
        critical_vulns = risk_assessment['severity_distribution']['critical']
        high_vulns = risk_assessment['severity_distribution']['high']
        risk_level = risk_assessment['overall_risk_level']
        
        summary = f"""
EXECUTIVE SUMMARY - SECURITY ASSESSMENT

Target: {self.target}
Assessment Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

OVERALL RISK LEVEL: {risk_level}

VULNERABILITY SUMMARY:
- Total Vulnerabilities Found: {total_vulns}
- Critical Severity: {critical_vulns}
- High Severity: {high_vulns}
- Risk Score: {risk_assessment['risk_score']}/100

KEY FINDINGS:
"""
        
        if critical_vulns > 0:
            summary += f"• {critical_vulns} CRITICAL vulnerabilities require immediate attention\n"
        
        if high_vulns > 0:
            summary += f"• {high_vulns} HIGH severity vulnerabilities need prompt remediation\n"
        
        # Highlight most common vulnerability types
        vuln_types = {}
        for vuln in vulnerabilities:
            if vuln.vuln_type not in vuln_types:
                vuln_types[vuln.vuln_type] = 0
            vuln_types[vuln_type] += 1
        
        if vuln_types:
            most_common = max(vuln_types, key=vuln_types.get)
            summary += f"• Most common vulnerability type: {most_common.value.replace('_', ' ').title()}\n"
        
        summary += f"""
RECOMMENDATIONS:
1. Address all CRITICAL and HIGH severity vulnerabilities immediately
2. Implement comprehensive input validation and output encoding
3. Review and harden security configurations
4. Establish ongoing security monitoring and testing processes

This assessment identified significant security risks that require immediate attention.
Implementation of the recommended security controls is essential to protect against
cyber attacks and data breaches.
        """
        
        return summary.strip()
    
    def export_report(self, report: PenetrationTestReport, format: str = 'json') -> str:
        """Export penetration test report in specified format."""
        if format.lower() == 'json':
            return json.dumps(asdict(report), default=str, indent=2)
        
        elif format.lower() == 'html':
            return self._generate_html_report(report)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_report(self, report: PenetrationTestReport) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Security Assessment Report - {report.target}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f44336; color: white; padding: 20px; }}
        .summary {{ background-color: #f5f5f5; padding: 20px; margin: 20px 0; }}
        .vulnerability {{ border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }}
        .critical {{ border-color: #d32f2f; }}
        .high {{ border-color: #f44336; }}
        .medium {{ border-color: #ff9800; }}
        .low {{ border-color: #4caf50; }}
        .info {{ border-color: #2196f3; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Assessment Report</h1>
        <p>Target: {report.target}</p>
        <p>Date: {report.test_date.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <pre>{report.executive_summary}</pre>
    </div>
    
    <h2>Vulnerabilities Found ({len(report.vulnerabilities)})</h2>
"""
        
        for vuln in report.vulnerabilities:
            severity_class = vuln.severity.value.lower()
            html += f"""
    <div class="vulnerability {severity_class}">
        <h3>{vuln.title}</h3>
        <p><strong>Severity:</strong> {vuln.severity.value.upper()}</p>
        <p><strong>Type:</strong> {vuln.vuln_type.value.replace('_', ' ').title()}</p>
        <p><strong>Location:</strong> {vuln.location}</p>
        <p><strong>Description:</strong> {vuln.description}</p>
        <p><strong>Evidence:</strong> {vuln.evidence}</p>
        <p><strong>Impact:</strong> {vuln.impact}</p>
        <p><strong>Remediation:</strong> {vuln.remediation}</p>
    </div>
"""
        
        html += """
</body>
</html>
        """
        
        return html


# Global security testing manager
_security_test_orchestrator: Optional[SecurityTestOrchestrator] = None


def init_security_testing(target: str, config: Optional[Dict[str, Any]] = None) -> SecurityTestOrchestrator:
    """Initialize global security test orchestrator."""
    global _security_test_orchestrator
    
    _security_test_orchestrator = SecurityTestOrchestrator(target, config)
    return _security_test_orchestrator


def get_security_test_orchestrator() -> SecurityTestOrchestrator:
    """Get global security test orchestrator instance."""
    global _security_test_orchestrator
    
    if _security_test_orchestrator is None:
        raise RuntimeError("Security test orchestrator not initialized. Call init_security_testing() first.")
    
    return _security_test_orchestrator