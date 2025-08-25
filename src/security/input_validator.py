"""
Comprehensive security-first input validation system for claude-tui.

This module implements enterprise-grade security validation with:
- Multi-layered threat detection
- Command injection prevention
- Path traversal protection
- API key detection and prevention
- SQL injection blocking
- XSS prevention
- Comprehensive sanitization
"""

import re
import html
import ast
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import logging
from dataclasses import dataclass
from urllib.parse import urlparse
from enum import Enum

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a security validation operation."""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_value: Optional[Any] = None
    threat_level: ThreatLevel = ThreatLevel.NONE
    threats_detected: List["SecurityThreat"] = None
    
    def __post_init__(self):
        if self.threats_detected is None:
            self.threats_detected = []

@dataclass
class SecurityThreat:
    """Detected security threat."""
    threat_type: str
    severity: ThreatLevel
    description: str
    evidence: str
    recommendation: str
    pattern_matched: Optional[str] = None
    confidence_score: float = 1.0

class SecurityInputValidator:
    """
    Enterprise-grade security input validator.
    
    Implements multiple layers of security validation:
    1. Pattern-based threat detection
    2. Context-aware validation
    3. Content sanitization
    4. Threat scoring and reporting
    """
    
    # Critical security patterns - immediate rejection
    CRITICAL_PATTERNS = [
        (r'__import__\s*\(', "Dynamic import injection", "code_injection"),
        (r'exec\s*\(', "Code execution injection", "code_injection"), 
        (r'eval\s*\(', "Code evaluation injection", "code_injection"),
        (r'subprocess\.\w+\s*\([^)]*shell\s*=\s*True', "Shell injection", "command_injection"),
        (r'os\.system\s*\(', "System command injection", "command_injection"),
        (r'<script[^>]*>.*?</script>', "Script tag injection", "xss"),
        (r'javascript\s*:', "JavaScript protocol injection", "xss"),
        (r'vbscript\s*:', "VBScript protocol injection", "xss"),
        (r'data\s*:\s*text/html', "Data URI HTML injection", "xss"),
        (r'(\x00|\x08|\x0b|\x0c|\x0e|\x1f)', "Null byte injection", "binary_injection"),
    ]
    
    # High risk patterns - flag for review
    HIGH_RISK_PATTERNS = [
        (r'rm\s+-rf', "Dangerous file deletion", "file_system"),
        (r'del\s+/[sq]', "Windows deletion command", "file_system"),
        (r'DROP\s+TABLE', "SQL table deletion", "sql_injection"),
        (r'DELETE\s+FROM', "SQL deletion", "sql_injection"),
        (r'INSERT\s+INTO', "SQL insertion", "sql_injection"),
        (r'UPDATE\s+\w+\s+SET', "SQL update", "sql_injection"),
        (r'on\w+\s*=\s*[\'"][^\'\"]*[\'"]', "Event handler injection", "xss"),
        (r'<iframe[^>]*src', "Iframe injection", "xss"),
        (r'<object[^>]*data', "Object tag injection", "xss"),
        (r'<embed[^>]*src', "Embed tag injection", "xss"),
    ]
    
    # Medium risk patterns - sanitize and warn
    MEDIUM_RISK_PATTERNS = [
        (r'\.\./', "Directory traversal", "path_traversal"),
        (r'%[0-9a-fA-F]{2}', "URL encoding", "encoding"),
        (r'\$\{.*\}', "Template injection", "template_injection"),
        (r'{{.*}}', "Template injection", "template_injection"),
        (r'curl\s+', "External HTTP request", "network"),
        (r'wget\s+', "External HTTP request", "network"),
        (r'nc\s+', "Netcat usage", "network"),
        (r'telnet\s+', "Telnet usage", "network"),
    ]
    
    # API key and secret patterns
    SECRET_PATTERNS = [
        (r'sk-[a-zA-Z0-9]{40,}', "OpenAI API key", "api_key"),
        (r'sk-ant-[a-zA-Z0-9]{40,}', "Anthropic API key", "api_key"),
        (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token", "api_key"),
        (r'gho_[a-zA-Z0-9]{36}', "GitHub OAuth token", "api_key"), 
        (r'AIza[0-9A-Za-z\\-_]{35}', "Google API key", "api_key"),
        (r'AKIA[0-9A-Z]{16}', "AWS access key", "api_key"),
        (r'password\s*[:=]\s*["\'][^"\']{8,}["\']', "Hardcoded password", "password"),
        (r'secret\s*[:=]\s*["\'][^"\']{20,}["\']', "Hardcoded secret", "secret"),
        (r'token\s*[:=]\s*["\'][^"\']{20,}["\']', "Hardcoded token", "token"),
        (r'key\s*[:=]\s*["\'][^"\']{20,}["\']', "Hardcoded key", "key"),
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        (r"('\s*(or|and)\s*')", "SQL boolean injection", "sql_injection"),
        (r'("\s*(or|and)\s*")', "SQL boolean injection", "sql_injection"),
        (r'union\s+select', "SQL union injection", "sql_injection"),
        (r';--', "SQL comment injection", "sql_injection"),
        (r'/\*.*\*/', "SQL comment injection", "sql_injection"),
        (r"';\s*(drop|delete|insert|update)", "SQL command injection", "sql_injection"),
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        (r'[;&|`$()]', "Shell metacharacters", "command_injection"),
        (r'>\s*/dev/', "Device redirection", "command_injection"),
        (r'<\s*/', "Input redirection", "command_injection"),
        (r'\|\s*(sh|bash|cmd|powershell)', "Shell pipe injection", "command_injection"),
        (r'&&|\|\|', "Command chaining", "command_injection"),
        (r'2>&1', "Error redirection", "command_injection"),
        (r'>/dev/null', "Output redirection", "command_injection"),
    ]
    
    # File extensions by security context
    SAFE_EXTENSIONS = {
        'code': {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.less'},
        'config': {'.json', '.yaml', '.yml', '.toml', '.ini', '.env'},
        'docs': {'.md', '.txt', '.rst', '.pdf'},
        'data': {'.csv', '.xml', '.sql'},
        'general': {'.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml', '.md', '.txt'},
    }
    
    # Dangerous executable extensions
    EXECUTABLE_EXTENSIONS = {'.exe', '.bat', '.cmd', '.sh', '.ps1', '.scr', '.msi', '.dll', '.so'}
    
    # Allowed commands by category
    ALLOWED_COMMANDS = {
        'git': {'status', 'add', 'commit', 'push', 'pull', 'branch', 'checkout', 'clone', 'diff', 'log', 'remote', 'init'},
        'npm': {'install', 'test', 'run', 'start', 'build', 'init', 'list', 'version'},
        'node': {'-v', '--version', '-e'},
        'python': {'-c', '-m', 'setup.py', '-V', '--version'},
        'python3': {'-c', '-m', 'setup.py', '-V', '--version'},
        'pip': {'install', 'list', 'show', 'freeze', 'uninstall'},
        'pip3': {'install', 'list', 'show', 'freeze', 'uninstall'},
        'ls': {'-l', '-la', '-lah', '-a', '-h'},
        'pwd': set(),
        'mkdir': {'-p'},
        'cd': set(),
        'cp': {'-r', '-f'},
        'mv': set(),
        'echo': set(),
        'cat': set(),
        'grep': {'-r', '-i', '-n', '-v'},
        'find': {'-name', '-type', '-exec'},
        'wc': {'-l', '-w', '-c'},
        'sort': {'-n', '-r'},
        'head': {'-n'},
        'tail': {'-n', '-f'},
    }
    
    def __init__(self):
        """Initialize the security validator."""
        self.threats_detected: List[SecurityThreat] = []
        self._validation_cache: Dict[str, ValidationResult] = {}
    
    def validate_user_prompt(self, prompt: str, context: str = "general") -> ValidationResult:
        """
        Validate user prompts with comprehensive security checks.
        
        Args:
            prompt: User input to validate
            context: Validation context for specialized rules
            
        Returns:
            ValidationResult with security assessment
        """
        if not isinstance(prompt, str):
            return ValidationResult(
                is_valid=False,
                error_message="Prompt must be a string",
                threat_level=ThreatLevel.MEDIUM
            )
        
        # Length validation
        if len(prompt) > 100000:  # 100K character limit
            return ValidationResult(
                is_valid=False,
                error_message="Prompt exceeds maximum length (100,000 characters)",
                threat_level=ThreatLevel.MEDIUM
            )
        
        if len(prompt.strip()) == 0:
            return ValidationResult(
                is_valid=False,
                error_message="Prompt cannot be empty",
                threat_level=ThreatLevel.LOW
            )
        
        # Reset threat detection
        self.threats_detected = []
        
        # Multi-layer security scanning
        max_threat_level = self._scan_for_threats(prompt)
        
        # Check for hardcoded secrets
        secret_threats = self._detect_secrets(prompt)
        if secret_threats:
            self.threats_detected.extend(secret_threats)
            max_threat_level = max(max_threat_level, ThreatLevel.CRITICAL, key=self._threat_priority)
        
        # SQL injection detection
        sql_threats = self._detect_sql_injection(prompt)
        if sql_threats:
            self.threats_detected.extend(sql_threats)
            max_threat_level = max(max_threat_level, ThreatLevel.HIGH, key=self._threat_priority)
        
        # Command injection detection
        cmd_threats = self._detect_command_injection(prompt)
        if cmd_threats:
            self.threats_detected.extend(cmd_threats)
            max_threat_level = max(max_threat_level, ThreatLevel.CRITICAL, key=self._threat_priority)
        
        # Context-specific validation
        context_result = self._validate_context_specific(prompt, context)
        if not context_result.is_valid:
            return context_result
        
        # If critical or high threats detected, reject
        if max_threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            threat_descriptions = [t.description for t in self.threats_detected]
            return ValidationResult(
                is_valid=False,
                error_message=f"Security threats detected: {'; '.join(threat_descriptions)}",
                threat_level=max_threat_level,
                threats_detected=self.threats_detected
            )
        
        # Sanitize the prompt
        sanitized_prompt = self._sanitize_prompt(prompt)
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized_prompt,
            threat_level=max_threat_level,
            threats_detected=self.threats_detected
        )
    
    def validate_file_path(self, path: str, context: str = "general") -> ValidationResult:
        """
        Validate file paths with enhanced security checks.
        
        Args:
            path: File path to validate
            context: Security context (code, config, docs, etc.)
            
        Returns:
            ValidationResult with path security assessment
        """
        if not isinstance(path, str):
            return ValidationResult(
                is_valid=False,
                error_message="Path must be a string",
                threat_level=ThreatLevel.MEDIUM
            )
        
        try:
            # Normalize and resolve path
            path_obj = Path(path).resolve()
        except (ValueError, OSError) as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid path format: {e}",
                threat_level=ThreatLevel.MEDIUM
            )
        
        # Security checks
        path_str = str(path_obj)
        threats = []
        
        # Directory traversal detection
        if '..' in path or '/..' in path_str or '\\..\\' in path_str:
            threats.append(SecurityThreat(
                threat_type="path_traversal",
                severity=ThreatLevel.HIGH,
                description="Directory traversal attempt detected",
                evidence=path,
                recommendation="Use absolute paths or validate path resolution"
            ))
        
        # System directory protection
        dangerous_paths = [
            '/etc/', '/proc/', '/sys/', '/dev/', '/root/', '/home/root/',
            'C:\\Windows\\', 'C:\\System32\\', 'C:\\Program Files\\',
            '/usr/bin/', '/bin/', '/sbin/', '/usr/sbin/'
        ]
        
        for danger_path in dangerous_paths:
            if danger_path.lower() in path_str.lower():
                threats.append(SecurityThreat(
                    threat_type="system_access",
                    severity=ThreatLevel.HIGH,
                    description=f"Attempt to access system directory: {danger_path}",
                    evidence=path,
                    recommendation="Access to system directories is not allowed"
                ))
        
        # Extension validation
        if path_obj.suffix:
            extension = path_obj.suffix.lower()
            
            # Check for executable extensions
            if extension in self.EXECUTABLE_EXTENSIONS:
                threats.append(SecurityThreat(
                    threat_type="executable_file",
                    severity=ThreatLevel.HIGH,
                    description=f"Executable file extension detected: {extension}",
                    evidence=path,
                    recommendation="Executable files are not allowed in this context"
                ))
            
            # Context-specific extension validation
            allowed_extensions = self.SAFE_EXTENSIONS.get(context, self.SAFE_EXTENSIONS['general'])
            if extension not in allowed_extensions:
                threats.append(SecurityThreat(
                    threat_type="invalid_extension",
                    severity=ThreatLevel.MEDIUM,
                    description=f"File extension not allowed in {context} context: {extension}",
                    evidence=path,
                    recommendation=f"Use allowed extensions: {', '.join(allowed_extensions)}"
                ))
        
        # Check for suspicious file names
        suspicious_names = ['passwd', 'shadow', 'hosts', 'sudoers', 'id_rsa', 'id_dsa']
        if any(sus_name in path_obj.name.lower() for sus_name in suspicious_names):
            threats.append(SecurityThreat(
                threat_type="sensitive_file",
                severity=ThreatLevel.HIGH,
                description="Attempt to access sensitive system file",
                evidence=path,
                recommendation="Access to sensitive files is denied"
            ))
        
        # Determine overall threat level
        if threats:
            max_threat = max(threats, key=lambda t: self._threat_priority(t.severity))
            if max_threat.severity in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Path security violation: {max_threat.description}",
                    threat_level=max_threat.severity,
                    threats_detected=threats
                )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=path_obj,
            threat_level=ThreatLevel.LOW if threats else ThreatLevel.NONE,
            threats_detected=threats
        )
    
    def validate_command(self, command: str, allowed_commands: Optional[Set[str]] = None) -> ValidationResult:
        """
        Validate system commands with strict security controls.
        
        Args:
            command: Command to validate
            allowed_commands: Override default allowed commands
            
        Returns:
            ValidationResult with command security assessment
        """
        if not isinstance(command, str):
            return ValidationResult(
                is_valid=False,
                error_message="Command must be a string",
                threat_level=ThreatLevel.MEDIUM
            )
        
        command = command.strip()
        if not command:
            return ValidationResult(
                is_valid=False,
                error_message="Command cannot be empty",
                threat_level=ThreatLevel.LOW
            )
        
        # Parse command parts
        command_parts = command.split()
        if not command_parts:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid command format",
                threat_level=ThreatLevel.MEDIUM
            )
        
        base_command = command_parts[0]
        threats = []
        
        # Check against allowed commands
        if allowed_commands is None:
            allowed_commands = set(self.ALLOWED_COMMANDS.keys())
        
        if base_command not in allowed_commands:
            threats.append(SecurityThreat(
                threat_type="unauthorized_command",
                severity=ThreatLevel.HIGH,
                description=f"Command not in allowed list: {base_command}",
                evidence=command,
                recommendation="Use only whitelisted commands"
            ))
        
        # Command injection detection
        cmd_threats = self._detect_command_injection(command)
        threats.extend(cmd_threats)
        
        # Argument validation for known commands
        if base_command in self.ALLOWED_COMMANDS:
            allowed_args = self.ALLOWED_COMMANDS[base_command]
            if len(command_parts) > 1 and allowed_args:
                for arg in command_parts[1:]:
                    if arg.startswith('-') and arg not in allowed_args:
                        threats.append(SecurityThreat(
                            threat_type="unauthorized_argument",
                            severity=ThreatLevel.MEDIUM,
                            description=f"Argument not allowed for {base_command}: {arg}",
                            evidence=command,
                            recommendation=f"Use allowed arguments: {', '.join(allowed_args)}"
                        ))
        
        # Specific command validation
        specific_threats = self._validate_specific_command(base_command, command)
        threats.extend(specific_threats)
        
        # Determine result
        if threats:
            max_threat = max(threats, key=lambda t: self._threat_priority(t.severity))
            if max_threat.severity in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Command security violation: {max_threat.description}",
                    threat_level=max_threat.severity,
                    threats_detected=threats
                )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=command,
            threat_level=ThreatLevel.LOW if threats else ThreatLevel.NONE,
            threats_detected=threats
        )
    
    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate URLs with security considerations.
        
        Args:
            url: URL to validate
            
        Returns:
            ValidationResult with URL security assessment
        """
        if not isinstance(url, str):
            return ValidationResult(
                is_valid=False,
                error_message="URL must be a string",
                threat_level=ThreatLevel.MEDIUM
            )
        
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid URL format: {e}",
                threat_level=ThreatLevel.MEDIUM
            )
        
        threats = []
        
        # Scheme validation
        allowed_schemes = {'https', 'http'}
        dangerous_schemes = {'javascript', 'vbscript', 'data', 'file', 'ftp'}
        
        if parsed.scheme in dangerous_schemes:
            threats.append(SecurityThreat(
                threat_type="dangerous_scheme",
                severity=ThreatLevel.CRITICAL,
                description=f"Dangerous URL scheme: {parsed.scheme}",
                evidence=url,
                recommendation="Use only HTTP/HTTPS URLs"
            ))
        elif parsed.scheme not in allowed_schemes:
            threats.append(SecurityThreat(
                threat_type="invalid_scheme",
                severity=ThreatLevel.MEDIUM,
                description=f"URL scheme not allowed: {parsed.scheme}",
                evidence=url,
                recommendation="Use HTTP or HTTPS URLs"
            ))
        
        # HTTPS requirement for external URLs
        if parsed.hostname and parsed.hostname not in ['localhost', '127.0.0.1', '::1']:
            if parsed.scheme != 'https':
                threats.append(SecurityThreat(
                    threat_type="insecure_external_url",
                    severity=ThreatLevel.HIGH,
                    description="External URLs must use HTTPS",
                    evidence=url,
                    recommendation="Use HTTPS for external URLs"
                ))
        
        # IP address detection
        if parsed.hostname:
            ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
            if re.match(ip_pattern, parsed.hostname):
                threats.append(SecurityThreat(
                    threat_type="direct_ip_access",
                    severity=ThreatLevel.MEDIUM,
                    description="Direct IP address access detected",
                    evidence=url,
                    recommendation="Use domain names instead of IP addresses"
                ))
        
        # Suspicious URL patterns
        suspicious_patterns = [
            (r'bit\.ly|tinyurl|t\.co', "URL shortener detected"),
            (r'[0-9a-f]{32,}', "Suspicious hash pattern in URL"),
            (r'%[0-9a-f]{2}', "URL encoding detected"),
        ]
        
        for pattern, description in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                threats.append(SecurityThreat(
                    threat_type="suspicious_url",
                    severity=ThreatLevel.LOW,
                    description=description,
                    evidence=url,
                    recommendation="Review URL for legitimacy"
                ))
        
        # Determine result
        if threats:
            max_threat = max(threats, key=lambda t: self._threat_priority(t.severity))
            if max_threat.severity in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"URL security violation: {max_threat.description}",
                    threat_level=max_threat.severity,
                    threats_detected=threats
                )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=url,
            threat_level=ThreatLevel.LOW if threats else ThreatLevel.NONE,
            threats_detected=threats
        )
    
    def validate_project_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate project configuration with security checks.
        
        Args:
            config: Project configuration dictionary
            
        Returns:
            ValidationResult with configuration security assessment
        """
        if not isinstance(config, dict):
            return ValidationResult(
                is_valid=False,
                error_message="Configuration must be a dictionary",
                threat_level=ThreatLevel.MEDIUM
            )
        
        threats = []
        
        # Required fields validation
        required_fields = {'name', 'type', 'language'}
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                error_message=f"Missing required fields: {', '.join(missing_fields)}",
                threat_level=ThreatLevel.LOW
            )
        
        # Validate project name
        project_name = config.get('name', '')
        if not isinstance(project_name, str):
            return ValidationResult(
                is_valid=False,
                error_message="Project name must be a string",
                threat_level=ThreatLevel.MEDIUM
            )
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
            threats.append(SecurityThreat(
                threat_type="invalid_project_name",
                severity=ThreatLevel.MEDIUM,
                description="Project name contains invalid characters",
                evidence=project_name,
                recommendation="Use only alphanumeric characters, underscores, and hyphens"
            ))
        
        if len(project_name) < 1 or len(project_name) > 100:
            return ValidationResult(
                is_valid=False,
                error_message="Project name must be between 1-100 characters",
                threat_level=ThreatLevel.LOW
            )
        
        # Validate other string fields
        for field in ['type', 'language', 'description', 'version']:
            if field in config:
                field_result = self._validate_config_field(field, config[field])
                if not field_result.is_valid:
                    return field_result
                if field_result.threats_detected:
                    threats.extend(field_result.threats_detected)
        
        # Check entire configuration for secrets
        config_str = json.dumps(config, indent=2)
        secret_threats = self._detect_secrets(config_str)
        if secret_threats:
            threats.extend(secret_threats)
        
        # Determine result
        if threats:
            max_threat = max(threats, key=lambda t: self._threat_priority(t.severity))
            if max_threat.severity in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Configuration security violation: {max_threat.description}",
                    threat_level=max_threat.severity,
                    threats_detected=threats
                )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=config,
            threat_level=ThreatLevel.LOW if threats else ThreatLevel.NONE,
            threats_detected=threats
        )
    
    def _scan_for_threats(self, text: str) -> ThreatLevel:
        """
        Comprehensive threat scanning using pattern matching.
        
        Args:
            text: Text to scan for threats
            
        Returns:
            Highest threat level detected
        """
        max_threat_level = ThreatLevel.NONE
        
        # Critical patterns
        for pattern, description, category in self.CRITICAL_PATTERNS:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                self.threats_detected.append(SecurityThreat(
                    threat_type=category,
                    severity=ThreatLevel.CRITICAL,
                    description=description,
                    evidence=match.group(0)[:50] + ("..." if len(match.group(0)) > 50 else ""),
                    recommendation="Remove or sanitize dangerous patterns",
                    pattern_matched=pattern
                ))
                max_threat_level = ThreatLevel.CRITICAL
        
        # High risk patterns
        if max_threat_level != ThreatLevel.CRITICAL:
            for pattern, description, category in self.HIGH_RISK_PATTERNS:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
                for match in matches:
                    self.threats_detected.append(SecurityThreat(
                        threat_type=category,
                        severity=ThreatLevel.HIGH,
                        description=description,
                        evidence=match.group(0)[:50] + ("..." if len(match.group(0)) > 50 else ""),
                        recommendation="Review and validate high-risk patterns",
                        pattern_matched=pattern
                    ))
                    if max_threat_level not in [ThreatLevel.CRITICAL]:
                        max_threat_level = ThreatLevel.HIGH
        
        # Medium risk patterns
        if max_threat_level not in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            for pattern, description, category in self.MEDIUM_RISK_PATTERNS:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
                for match in matches:
                    self.threats_detected.append(SecurityThreat(
                        threat_type=category,
                        severity=ThreatLevel.MEDIUM,
                        description=description,
                        evidence=match.group(0)[:50] + ("..." if len(match.group(0)) > 50 else ""),
                        recommendation="Monitor and sanitize medium-risk patterns",
                        pattern_matched=pattern
                    ))
                    if max_threat_level not in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                        max_threat_level = ThreatLevel.MEDIUM
        
        return max_threat_level
    
    def _detect_secrets(self, text: str) -> List[SecurityThreat]:
        """Detect hardcoded secrets and API keys."""
        threats = []
        
        for pattern, description, category in self.SECRET_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                evidence = match.group(0)[:10] + "***REDACTED***"  # Partial evidence only
                threats.append(SecurityThreat(
                    threat_type=category,
                    severity=ThreatLevel.CRITICAL,
                    description=f"{description} detected",
                    evidence=evidence,
                    recommendation="Use environment variables or secure vaults for secrets",
                    pattern_matched=pattern
                ))
        
        return threats
    
    def _detect_sql_injection(self, text: str) -> List[SecurityThreat]:
        """Detect SQL injection patterns."""
        threats = []
        
        for pattern, description, category in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                threats.append(SecurityThreat(
                    threat_type=category,
                    severity=ThreatLevel.HIGH,
                    description=description,
                    evidence=pattern,
                    recommendation="Use parameterized queries and input validation",
                    pattern_matched=pattern
                ))
        
        return threats
    
    def _detect_command_injection(self, text: str) -> List[SecurityThreat]:
        """Detect command injection patterns."""
        threats = []
        
        for pattern, description, category in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(SecurityThreat(
                    threat_type=category,
                    severity=ThreatLevel.CRITICAL,
                    description=description,
                    evidence=pattern,
                    recommendation="Use safe command execution methods and input validation",
                    pattern_matched=pattern
                ))
        
        return threats
    
    def _validate_context_specific(self, text: str, context: str) -> ValidationResult:
        """Apply context-specific validation rules."""
        # Context-specific rules can be added here
        # For now, return valid
        return ValidationResult(is_valid=True, threat_level=ThreatLevel.NONE)
    
    def _validate_config_field(self, field: str, value: Any) -> ValidationResult:
        """Validate individual configuration field."""
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                error_message=f"Field '{field}' must be a string",
                threat_level=ThreatLevel.MEDIUM
            )
        
        if len(value) > 1000:
            return ValidationResult(
                is_valid=False,
                error_message=f"Field '{field}' exceeds maximum length",
                threat_level=ThreatLevel.MEDIUM
            )
        
        # Scan field for threats
        field_threats = []
        field_threat_level = self._scan_for_threats(value)
        if self.threats_detected:
            # Get threats from last scan
            field_threats = [t for t in self.threats_detected if t.evidence in value]
        
        if field_threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            return ValidationResult(
                is_valid=False,
                error_message=f"Security threat in field '{field}'",
                threat_level=field_threat_level,
                threats_detected=field_threats
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=value,
            threat_level=field_threat_level,
            threats_detected=field_threats
        )
    
    def _validate_specific_command(self, base_command: str, full_command: str) -> List[SecurityThreat]:
        """Additional validation for specific commands."""
        threats = []
        
        if base_command == 'git':
            # Check for dangerous git operations
            dangerous_git_args = ['--upload-pack', '--receive-pack', '--exec']
            for arg in dangerous_git_args:
                if arg in full_command:
                    threats.append(SecurityThreat(
                        threat_type="dangerous_git_operation",
                        severity=ThreatLevel.HIGH,
                        description=f"Dangerous git argument: {arg}",
                        evidence=full_command,
                        recommendation="Avoid using dangerous git arguments"
                    ))
        
        elif base_command in ['npm', 'pip', 'pip3']:
            # Check for package installation with suspicious patterns
            if 'install' in full_command:
                if re.search(r'[^a-zA-Z0-9@_.-]', full_command.split('install', 1)[1]):
                    threats.append(SecurityThreat(
                        threat_type="suspicious_package_name",
                        severity=ThreatLevel.MEDIUM,
                        description="Suspicious package name pattern",
                        evidence=full_command,
                        recommendation="Validate package names before installation"
                    ))
        
        elif base_command in ['python', 'python3']:
            # Validate Python code execution
            if '-c' in full_command:
                code_match = re.search(r'-c\s+["\']([^"\']*)["\']', full_command)
                if code_match:
                    code = code_match.group(1)
                    try:
                        ast.parse(code)
                    except SyntaxError:
                        threats.append(SecurityThreat(
                            threat_type="invalid_python_syntax",
                            severity=ThreatLevel.MEDIUM,
                            description="Invalid Python syntax in -c argument",
                            evidence=code,
                            recommendation="Ensure valid Python syntax"
                        ))
                    
                    # Check for dangerous imports
                    dangerous_imports = ['os', 'subprocess', 'sys', '__import__', 'eval', 'exec']
                    for imp in dangerous_imports:
                        if imp in code:
                            threats.append(SecurityThreat(
                                threat_type="dangerous_python_import",
                                severity=ThreatLevel.HIGH,
                                description=f"Dangerous Python import: {imp}",
                                evidence=code,
                                recommendation="Avoid dangerous Python imports in inline code"
                            ))
        
        return threats
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt content."""
        # HTML escape
        sanitized = html.escape(prompt)
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _threat_priority(self, threat_level: ThreatLevel) -> int:
        """Get numeric priority for threat level comparison."""
        priorities = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return priorities.get(threat_level, 0)

# Utility functions for backward compatibility
def validate_user_input(user_input: str, input_type: str = "general") -> ValidationResult:
    """
    Main validation function with enhanced security.
    
    Args:
        user_input: Input to validate
        input_type: Type of input (general, prompt, file_path, command, url)
        
    Returns:
        ValidationResult with security assessment
    """
    validator = SecurityInputValidator()
    
    validation_map = {
        "general": lambda x: validator.validate_user_prompt(x),
        "prompt": lambda x: validator.validate_user_prompt(x),
        "file_path": lambda x: validator.validate_file_path(x),
        "command": lambda x: validator.validate_command(x),
        "url": lambda x: validator.validate_url(x),
        "config": lambda x: validator.validate_project_config(json.loads(x) if isinstance(x, str) else x),
    }
    
    validation_func = validation_map.get(input_type, validation_map["general"])
    return validation_func(user_input)