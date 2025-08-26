# Security Architecture & Guidelines

## 1. Security Architecture Overview

### Defense-in-Depth Strategy
The claude-tui project implements multiple layers of security to protect against various threats:

```
┌─────────────────────────────────────────┐
│            User Interface Layer         │ ← Input validation, XSS prevention
├─────────────────────────────────────────┤
│           Application Layer             │ ← Authorization, secure subprocess
├─────────────────────────────────────────┤
│          Data Processing Layer          │ ← Sandboxing, rate limiting
├─────────────────────────────────────────┤
│           Storage Layer                 │ ← Encryption at rest
├─────────────────────────────────────────┤
│          Network Layer                  │ ← TLS, API key protection
└─────────────────────────────────────────┘
```

### Core Security Principles
- **Zero Trust**: Never trust, always verify
- **Principle of Least Privilege**: Minimal necessary permissions
- **Fail Secure**: Safe failure modes
- **Defense in Depth**: Multiple security layers
- **Secure by Default**: Security-first configuration

## 2. Input Validation Strategies

### Comprehensive Input Sanitization
```python
import re
import html
from typing import Any, Dict, List
from pathlib import Path

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'open\s*\(',
        r'subprocess\.',
        r'os\.system',
        r'rm\s+-rf',
        r'del\s+/.*',
        r'DROP\s+TABLE',
        r'<script.*?>',
        r'javascript:',
        r'vbscript:',
        r'on\w+\s*=',
    ]
    
    # Safe file extensions
    SAFE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml', '.md', '.txt'}
    
    def validate_user_prompt(self, prompt: str) -> str:
        """Validate and sanitize user prompts"""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
            
        if len(prompt) > 10000:  # Max 10k characters
            raise ValueError("Prompt too long")
            
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous content detected: {pattern}")
        
        # HTML escape for display
        return html.escape(prompt)
    
    def validate_file_path(self, path: str) -> Path:
        """Validate file paths for safety"""
        try:
            path_obj = Path(path).resolve()
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid path: {e}")
        
        # Prevent directory traversal
        if '..' in str(path_obj):
            raise ValueError("Directory traversal detected")
            
        # Check extension
        if path_obj.suffix not in self.SAFE_EXTENSIONS:
            raise ValueError(f"Unsafe file extension: {path_obj.suffix}")
            
        return path_obj
    
    def validate_project_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate project configuration"""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
            
        # Required fields
        required_fields = {'name', 'type', 'language'}
        if not all(field in config for field in required_fields):
            raise ValueError("Missing required configuration fields")
            
        # Validate project name
        if not re.match(r'^[a-zA-Z0-9_-]+$', config['name']):
            raise ValueError("Invalid project name. Use only alphanumeric, underscore, hyphen")
            
        return config

# Usage in application
validator = InputValidator()

def secure_prompt_handler(user_input: str):
    try:
        validated_prompt = validator.validate_user_prompt(user_input)
        return process_ai_request(validated_prompt)
    except ValueError as e:
        log_security_event(f"Input validation failed: {e}")
        return {"error": "Invalid input detected"}
```

### Anti-Injection Protection
```python
class AntiInjectionFilter:
    """Protection against various injection attacks"""
    
    SQL_INJECTION_PATTERNS = [
        r"('\s*(or|and)\s*')",
        r'("\s*(or|and)\s*")',
        r'(union\s+select)',
        r'(drop\s+table)',
        r'(insert\s+into)',
        r'(delete\s+from)',
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$()]',
        r'>\s*/dev/',
        r'<\s*/',
        r'\|\s*sh',
        r'\|\s*bash',
    ]
    
    def scan_for_injections(self, text: str) -> List[str]:
        """Scan text for injection patterns"""
        threats = []
        
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"SQL injection pattern: {pattern}")
        
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"Command injection pattern: {pattern}")
                
        return threats
```

## 3. Sandbox Execution for Generated Code

### Secure Code Execution Environment
```python
import subprocess
import tempfile
import os
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import docker  # Optional: Docker-based sandboxing

class SecureCodeSandbox:
    """Secure sandbox for executing generated code"""
    
    def __init__(self):
        self.docker_client = None
        self.timeout_seconds = 30
        self.memory_limit = "128m"
        self.cpu_limit = "0.5"
        
        # Try to initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception:
            # Fall back to subprocess isolation
            pass
    
    def execute_python_code(self, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Python code in secure sandbox"""
        if self.docker_client:
            return self._execute_in_docker(code, "python:3.9-alpine", context)
        else:
            return self._execute_in_subprocess(code, context)
    
    def _execute_in_docker(self, code: str, image: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute code in Docker container"""
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run in Docker with restrictions
            container = self.docker_client.containers.run(
                image,
                f"python {os.path.basename(temp_file)}",
                volumes={os.path.dirname(temp_file): {'bind': '/app', 'mode': 'ro'}},
                working_dir='/app',
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=50000,  # 50% CPU
                network_disabled=True,
                remove=True,
                timeout=self.timeout_seconds,
                stdout=True,
                stderr=True,
                detach=False
            )
            
            return {
                "success": True,
                "output": container.decode('utf-8'),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _execute_in_subprocess(self, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute code in restricted subprocess"""
        try:
            # Create restricted environment
            restricted_env = {
                'PATH': '/usr/bin:/bin',
                'PYTHONPATH': '',
                'HOME': '/tmp',
            }
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Prepend security restrictions
                secure_code = self._wrap_with_restrictions(code)
                f.write(secure_code)
                temp_file = f.name
            
            # Execute with timeout and restrictions
            process = subprocess.Popen(
                ['python3', temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=restricted_env,
                preexec_fn=self._set_process_limits,
                cwd=tempfile.gettempdir()
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                return {
                    "success": process.returncode == 0,
                    "output": stdout.decode('utf-8'),
                    "error": stderr.decode('utf-8') if stderr else None
                }
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "success": False,
                    "output": None,
                    "error": "Execution timeout"
                }
                
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _wrap_with_restrictions(self, code: str) -> str:
        """Wrap code with security restrictions"""
        return f"""
import sys
import os

# Disable dangerous builtins
__builtins__.__dict__.clear()
__builtins__.__dict__.update({{
    'print': print,
    'len': len,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sum': sum,
    'max': max,
    'min': min,
    'abs': abs,
    'round': round,
}})

# Remove dangerous modules
for module in ['os', 'subprocess', 'sys', 'importlib']:
    if module in sys.modules:
        del sys.modules[module]

# User code starts here
{code}
"""
    
    def _set_process_limits(self):
        """Set process limits for subprocess"""
        try:
            # Limit memory (128MB)
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))  # 10 seconds CPU time
        except ImportError:
            pass
```

## 4. API Key Management

### Secure API Key Storage and Rotation
```python
import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import keyring
from typing import Optional, Dict

class SecureAPIKeyManager:
    """Secure management of API keys"""
    
    def __init__(self, app_name: str = "claude-tui"):
        self.app_name = app_name
        self.config_dir = Path.home() / f".{app_name}"
        self.config_dir.mkdir(exist_ok=True, mode=0o700)  # Owner only
        
        # Use system keyring if available, fallback to encrypted file
        self.use_keyring = True
        try:
            keyring.get_password("test", "test")
        except Exception:
            self.use_keyring = False
    
    def store_api_key(self, service: str, api_key: str, user: str = "default") -> None:
        """Store API key securely"""
        if self.use_keyring:
            keyring.set_password(f"{self.app_name}-{service}", user, api_key)
        else:
            self._store_encrypted_key(service, user, api_key)
        
        # Log successful storage (without key value)
        self._log_key_operation("store", service, user)
    
    def retrieve_api_key(self, service: str, user: str = "default") -> Optional[str]:
        """Retrieve API key securely"""
        try:
            if self.use_keyring:
                key = keyring.get_password(f"{self.app_name}-{service}", user)
            else:
                key = self._retrieve_encrypted_key(service, user)
            
            if key:
                self._log_key_operation("retrieve", service, user)
            
            return key
        except Exception as e:
            self._log_key_operation("retrieve_error", service, user, str(e))
            return None
    
    def delete_api_key(self, service: str, user: str = "default") -> bool:
        """Delete API key securely"""
        try:
            if self.use_keyring:
                keyring.delete_password(f"{self.app_name}-{service}", user)
            else:
                self._delete_encrypted_key(service, user)
            
            self._log_key_operation("delete", service, user)
            return True
        except Exception as e:
            self._log_key_operation("delete_error", service, user, str(e))
            return False
    
    def rotate_api_key(self, service: str, new_key: str, user: str = "default") -> bool:
        """Rotate API key with validation"""
        # Validate new key format
        if not self._validate_api_key_format(service, new_key):
            return False
        
        # Test new key if possible
        if self._test_api_key(service, new_key):
            self.store_api_key(service, new_key, user)
            self._log_key_operation("rotate", service, user)
            return True
        
        return False
    
    def _generate_encryption_key(self, password: str) -> bytes:
        """Generate encryption key from password"""
        password_bytes = password.encode()
        salt = b'claude-tui-salt'  # In production, use random salt per key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def _store_encrypted_key(self, service: str, user: str, api_key: str) -> None:
        """Store encrypted API key to file"""
        # Use machine-specific key
        machine_key = self._get_machine_key()
        fernet = Fernet(self._generate_encryption_key(machine_key))
        
        encrypted_key = fernet.encrypt(api_key.encode())
        
        key_file = self.config_dir / f"{service}_{user}.key"
        key_file.write_bytes(encrypted_key)
        key_file.chmod(0o600)  # Owner read/write only
    
    def _retrieve_encrypted_key(self, service: str, user: str) -> Optional[str]:
        """Retrieve encrypted API key from file"""
        key_file = self.config_dir / f"{service}_{user}.key"
        if not key_file.exists():
            return None
        
        machine_key = self._get_machine_key()
        fernet = Fernet(self._generate_encryption_key(machine_key))
        
        try:
            encrypted_key = key_file.read_bytes()
            decrypted_key = fernet.decrypt(encrypted_key)
            return decrypted_key.decode()
        except Exception:
            return None
    
    def _delete_encrypted_key(self, service: str, user: str) -> None:
        """Delete encrypted key file"""
        key_file = self.config_dir / f"{service}_{user}.key"
        if key_file.exists():
            key_file.unlink()
    
    def _get_machine_key(self) -> str:
        """Get machine-specific identifier for encryption"""
        import uuid
        return str(uuid.getnode())
    
    def _validate_api_key_format(self, service: str, api_key: str) -> bool:
        """Validate API key format for specific services"""
        formats = {
            'claude': r'^sk-ant-[a-zA-Z0-9]{40,}$',
            'openai': r'^sk-[a-zA-Z0-9]{40,}$',
            'github': r'^gh[pousr]_[a-zA-Z0-9]{36}$'
        }
        
        if service in formats:
            import re
            return bool(re.match(formats[service], api_key))
        
        # Generic validation: at least 20 chars, alphanumeric + common chars
        return len(api_key) >= 20 and api_key.replace('-', '').replace('_', '').isalnum()
    
    def _test_api_key(self, service: str, api_key: str) -> bool:
        """Test API key validity (implement per service)"""
        # Placeholder for API key testing
        # In real implementation, make test API calls
        return True
    
    def _log_key_operation(self, operation: str, service: str, user: str, error: str = None):
        """Log key operations for audit trail"""
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "operation": operation,
            "service": service,
            "user": user,
            "error": error
        }
        
        log_file = self.config_dir / "key_operations.log"
        with log_file.open('a') as f:
            f.write(json.dumps(log_entry) + '\n')

# Usage example
key_manager = SecureAPIKeyManager()
key_manager.store_api_key("claude", "sk-ant-api03-your-key-here")
api_key = key_manager.retrieve_api_key("claude")
```

## 5. Rate Limiting and DDoS Protection

### Smart Rate Limiting System
```python
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Tuple
import hashlib

class SmartRateLimiter:
    """Advanced rate limiting with adaptive thresholds"""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.blocked_ips = defaultdict(float)
        self.rate_limits = {
            'ai_requests': {'limit': 60, 'window': 60},      # 60 per minute
            'file_operations': {'limit': 100, 'window': 60}, # 100 per minute
            'api_calls': {'limit': 1000, 'window': 3600}     # 1000 per hour
        }
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str, operation_type: str) -> Tuple[bool, str]:
        """Check if request is allowed"""
        with self.lock:
            current_time = time.time()
            
            # Check if identifier is blocked
            if identifier in self.blocked_ips:
                if current_time < self.blocked_ips[identifier]:
                    return False, "IP temporarily blocked due to excessive requests"
                else:
                    del self.blocked_ips[identifier]
            
            # Get rate limit for operation type
            if operation_type not in self.rate_limits:
                return True, "Operation not rate limited"
            
            limit_config = self.rate_limits[operation_type]
            window = limit_config['window']
            limit = limit_config['limit']
            
            # Clean old requests
            key = f"{identifier}:{operation_type}"
            request_times = self.requests[key]
            
            while request_times and current_time - request_times[0] > window:
                request_times.popleft()
            
            # Check if limit exceeded
            if len(request_times) >= limit:
                # Block aggressive users
                if len(request_times) > limit * 2:
                    self.blocked_ips[identifier] = current_time + 300  # 5 min block
                    return False, f"Rate limit exceeded. Blocked for 5 minutes."
                
                return False, f"Rate limit exceeded: {limit} {operation_type} per {window} seconds"
            
            # Add current request
            request_times.append(current_time)
            return True, "Request allowed"
    
    def get_rate_limit_status(self, identifier: str, operation_type: str) -> Dict:
        """Get current rate limit status"""
        with self.lock:
            current_time = time.time()
            key = f"{identifier}:{operation_type}"
            
            if operation_type not in self.rate_limits:
                return {"status": "unlimited"}
            
            limit_config = self.rate_limits[operation_type]
            request_times = self.requests[key]
            
            # Clean old requests
            while request_times and current_time - request_times[0] > limit_config['window']:
                request_times.popleft()
            
            remaining = max(0, limit_config['limit'] - len(request_times))
            reset_time = current_time + limit_config['window']
            
            return {
                "status": "limited",
                "limit": limit_config['limit'],
                "remaining": remaining,
                "reset_time": reset_time,
                "window": limit_config['window']
            }

class DDoSProtection:
    """DDoS protection system"""
    
    def __init__(self):
        self.connection_counts = defaultdict(int)
        self.suspicious_patterns = defaultdict(list)
        self.blocked_ips = set()
        self.whitelist = set(['127.0.0.1', '::1'])  # Localhost
        
    def analyze_request_pattern(self, ip: str, user_agent: str, endpoint: str) -> bool:
        """Analyze request patterns for DDoS detection"""
        current_time = time.time()
        
        # Skip analysis for whitelisted IPs
        if ip in self.whitelist:
            return True
        
        # Check if already blocked
        if ip in self.blocked_ips:
            return False
        
        # Analyze patterns
        pattern_key = f"{ip}:{user_agent}"
        self.suspicious_patterns[pattern_key].append({
            'time': current_time,
            'endpoint': endpoint
        })
        
        # Clean old entries (last 5 minutes)
        self.suspicious_patterns[pattern_key] = [
            req for req in self.suspicious_patterns[pattern_key]
            if current_time - req['time'] < 300
        ]
        
        recent_requests = self.suspicious_patterns[pattern_key]
        
        # DDoS detection heuristics
        if len(recent_requests) > 500:  # More than 500 requests in 5 minutes
            self.blocked_ips.add(ip)
            return False
        
        # Check for identical requests (potential bot)
        endpoints = [req['endpoint'] for req in recent_requests[-50:]]
        if len(set(endpoints)) == 1 and len(endpoints) > 30:
            self.blocked_ips.add(ip)
            return False
        
        return True
```

## 6. Secure Subprocess Execution

### Hardened Process Execution
```python
import subprocess
import shlex
import os
import signal
import resource
from typing import List, Optional, Dict, Any
import logging

class SecureSubprocessManager:
    """Secure subprocess execution with comprehensive restrictions"""
    
    ALLOWED_COMMANDS = {
        'git': [
            'status', 'add', 'commit', 'push', 'pull', 'branch', 
            'checkout', 'clone', 'diff', 'log', 'remote'
        ],
        'npm': ['install', 'test', 'run', 'start', 'build'],
        'python': ['-c', '-m', 'setup.py'],
        'pip': ['install', 'list', 'show'],
        'node': ['-v', '--version'],
    }
    
    BLOCKED_COMMANDS = [
        'rm', 'rmdir', 'del', 'sudo', 'su', 'chmod', 'chown',
        'curl', 'wget', 'nc', 'netcat', 'telnet', 'ssh',
        'dd', 'format', 'fdisk', 'mkfs', 'mount', 'umount'
    ]
    
    def __init__(self):
        self.timeout = 60  # Default timeout
        self.max_memory = 512 * 1024 * 1024  # 512MB
        self.max_cpu_time = 30  # 30 seconds
        
    def execute_command(self, command: List[str], cwd: Optional[str] = None,
                       env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute command with security restrictions"""
        try:
            # Validate command
            if not self._validate_command(command):
                return {
                    "success": False,
                    "output": "",
                    "error": f"Command not allowed: {command[0]}",
                    "returncode": -1
                }
            
            # Prepare secure environment
            secure_env = self._create_secure_environment(env)
            
            # Execute with restrictions
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=secure_env,
                preexec_fn=self._set_security_limits,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return {
                    "success": process.returncode == 0,
                    "output": stdout,
                    "error": stderr,
                    "returncode": process.returncode
                }
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return {
                    "success": False,
                    "output": "",
                    "error": f"Command timed out after {self.timeout} seconds",
                    "returncode": -1
                }
                
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "returncode": -1
            }
    
    def _validate_command(self, command: List[str]) -> bool:
        """Validate command against whitelist/blacklist"""
        if not command:
            return False
        
        cmd_name = os.path.basename(command[0])
        
        # Check blocked commands
        if cmd_name in self.BLOCKED_COMMANDS:
            logging.warning(f"Blocked dangerous command: {cmd_name}")
            return False
        
        # Check allowed commands
        if cmd_name in self.ALLOWED_COMMANDS:
            allowed_args = self.ALLOWED_COMMANDS[cmd_name]
            if len(command) > 1 and command[1] not in allowed_args:
                logging.warning(f"Command {cmd_name} with disallowed argument: {command[1]}")
                return False
        
        # Additional security checks
        cmd_string = ' '.join(command)
        dangerous_patterns = [
            '&', '|', ';', '`', '$(',
            '>', '>>', '<', '<<',
            'rm -rf', 'del /q', 'format'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in cmd_string:
                logging.warning(f"Dangerous pattern in command: {pattern}")
                return False
        
        return True
    
    def _create_secure_environment(self, env: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Create restricted environment variables"""
        # Start with minimal environment
        secure_env = {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'HOME': '/tmp',
            'USER': 'nobody',
            'SHELL': '/bin/sh',
            'LANG': 'C',
            'LC_ALL': 'C'
        }
        
        # Allow specific additional variables
        if env:
            allowed_vars = {
                'PYTHONPATH', 'NODE_PATH', 'NPM_CONFIG_PREFIX',
                'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL'
            }
            
            for key, value in env.items():
                if key in allowed_vars:
                    secure_env[key] = value
        
        return secure_env
    
    def _set_security_limits(self):
        """Set process security limits"""
        try:
            # Set memory limit
            resource.setrlimit(resource.RLIMIT_AS, (self.max_memory, self.max_memory))
            
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time, self.max_cpu_time))
            
            # Set file size limit (100MB)
            resource.setrlimit(resource.RLIMIT_FSIZE, (100*1024*1024, 100*1024*1024))
            
            # Set number of processes limit
            resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
            
            # Prevent core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            
            # Set nice value (lower priority)
            os.nice(10)
            
        except (resource.error, OSError) as e:
            logging.warning(f"Failed to set some resource limits: {e}")

# Usage example
subprocess_manager = SecureSubprocessManager()

# Safe command execution
result = subprocess_manager.execute_command(['git', 'status'])
if result['success']:
    print(f"Git status: {result['output']}")
else:
    print(f"Command failed: {result['error']}")
```

## 7. Data Encryption at Rest and in Transit

### Comprehensive Encryption Strategy
```python
import os
import json
import hashlib
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import ssl
import requests
from typing import Any, Dict, Optional

class DataEncryption:
    """Comprehensive data encryption system"""
    
    def __init__(self, app_name: str = "claude-tui"):
        self.app_name = app_name
        self.key_dir = Path.home() / f".{app_name}" / "keys"
        self.key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Initialize or load master key
        self.master_key = self._get_or_create_master_key()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        key_file = self.key_dir / "master.key"
        
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            key_file.chmod(0o600)
            return key
    
    def encrypt_sensitive_data(self, data: Any) -> str:
        """Encrypt sensitive data for storage"""
        fernet = Fernet(self.master_key)
        
        # Serialize data
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        # Encrypt
        encrypted_data = fernet.encrypt(data_str.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Any:
        """Decrypt sensitive data"""
        fernet = Fernet(self.master_key)
        
        try:
            # Decode and decrypt
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            data_str = decrypted_bytes.decode()
            
            # Try to deserialize as JSON
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return data_str
                
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {e}")
    
    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt a file in place"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        content = file_path.read_bytes()
        
        # Encrypt content
        fernet = Fernet(self.master_key)
        encrypted_content = fernet.encrypt(content)
        
        # Write encrypted content
        encrypted_path = file_path.with_suffix(file_path.suffix + '.enc')
        encrypted_path.write_bytes(encrypted_content)
        encrypted_path.chmod(0o600)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_file_path: Path) -> Path:
        """Decrypt a file"""
        if not encrypted_file_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_path}")
        
        # Read encrypted content
        encrypted_content = encrypted_file_path.read_bytes()
        
        # Decrypt content
        fernet = Fernet(self.master_key)
        content = fernet.decrypt(encrypted_content)
        
        # Write decrypted content
        decrypted_path = encrypted_file_path.with_suffix('')
        decrypted_path.write_bytes(content)
        
        return decrypted_path

class SecureHTTPClient:
    """Secure HTTP client for API communications"""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Configure secure SSL context
        self.session.verify = True
        self.session.headers.update({
            'User-Agent': 'claude-tui/1.0.0 (Security Enhanced)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Configure SSL/TLS settings
        self.session.mount('https://', self._create_secure_adapter())
    
    def _create_secure_adapter(self):
        """Create secure HTTP adapter with strict TLS settings"""
        from requests.adapters import HTTPAdapter
        from urllib3.util.ssl_ import create_urllib3_context
        
        class SecureHTTPAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                context = create_urllib3_context()
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
                kwargs['ssl_context'] = context
                return super().init_poolmanager(*args, **kwargs)
        
        return SecureHTTPAdapter()
    
    def secure_post(self, url: str, data: Dict, api_key: str,
                   timeout: int = 30) -> requests.Response:
        """Make secure POST request with proper authentication"""
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Add request signing if needed
        headers.update(self._sign_request(data))
        
        try:
            response = self.session.post(
                url,
                json=data,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.SSLError as e:
            raise SecurityError(f"SSL verification failed: {e}")
        except requests.exceptions.Timeout:
            raise SecurityError("Request timed out")
        except requests.exceptions.RequestException as e:
            raise SecurityError(f"Request failed: {e}")
    
    def _sign_request(self, data: Dict) -> Dict[str, str]:
        """Sign request for integrity verification"""
        # Create request signature
        data_string = json.dumps(data, sort_keys=True, separators=(',', ':'))
        signature = hashlib.sha256(data_string.encode()).hexdigest()
        
        return {
            'X-Request-Signature': signature,
            'X-Timestamp': str(int(time.time()))
        }

class SecurityError(Exception):
    """Custom security exception"""
    pass
```

## 8. Security Audit Procedures

### Automated Security Auditing
```python
import os
import re
import ast
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SecurityFinding:
    """Security audit finding"""
    severity: str  # critical, high, medium, low
    category: str
    description: str
    file_path: str
    line_number: int
    evidence: str
    recommendation: str

class SecurityAuditor:
    """Comprehensive security auditing system"""
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.findings: List[SecurityFinding] = []
        
        # Security patterns to detect
        self.vulnerability_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'sk-[a-zA-Z0-9]{40,}',  # API key pattern
            ],
            'command_injection': [
                r'subprocess\.(call|run|Popen)\([^)]*shell=True',
                r'os\.system\(',
                r'os\.popen\(',
                r'eval\(',
                r'exec\(',
            ],
            'path_traversal': [
                r'\.\./|\.\.\\\',
                r'os\.path\.join\([^)]*\.\.',
                r'open\([^)]*\.\.',
            ],
            'insecure_random': [
                r'random\.random\(',
                r'random\.choice\(',
                r'random\.randint\(',
            ],
            'sql_injection': [
                r'execute\([^)]*%[sd]',
                r'cursor\.execute\([^)]*\+',
                r'SELECT.*WHERE.*%s',
            ]
        }
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        self.findings = []
        
        # Code analysis
        self._analyze_python_files()
        self._check_dependencies()
        self._audit_configurations()
        self._check_file_permissions()
        self._analyze_network_usage()
        
        # Generate report
        return self._generate_audit_report()
    
    def _analyze_python_files(self):
        """Analyze Python files for security issues"""
        python_files = list(self.project_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                self._scan_file_for_vulnerabilities(file_path, content)
                self._analyze_ast_security(file_path, content)
            except Exception as e:
                self.findings.append(SecurityFinding(
                    severity="medium",
                    category="file_analysis",
                    description=f"Failed to analyze file: {e}",
                    file_path=str(file_path),
                    line_number=0,
                    evidence="",
                    recommendation="Ensure file is readable and valid Python"
                ))
    
    def _scan_file_for_vulnerabilities(self, file_path: Path, content: str):
        """Scan file content for vulnerability patterns"""
        lines = content.split('\n')
        
        for category, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        severity = self._get_severity_for_category(category)
                        self.findings.append(SecurityFinding(
                            severity=severity,
                            category=category,
                            description=f"Potential {category.replace('_', ' ')} detected",
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=line_num,
                            evidence=line.strip(),
                            recommendation=self._get_recommendation_for_category(category)
                        ))
    
    def _analyze_ast_security(self, file_path: Path, content: str):
        """AST-based security analysis"""
        try:
            tree = ast.parse(content)
            visitor = SecurityASTVisitor(file_path, self.project_path)
            visitor.visit(tree)
            self.findings.extend(visitor.findings)
        except SyntaxError:
            pass  # File has syntax errors, skip AST analysis
    
    def _check_dependencies(self):
        """Check for vulnerable dependencies"""
        requirements_files = [
            'requirements.txt', 'requirements-dev.txt', 
            'Pipfile', 'pyproject.toml', 'setup.py'
        ]
        
        for req_file in requirements_files:
            req_path = self.project_path / req_file
            if req_path.exists():
                self._analyze_requirements_file(req_path)
    
    def _analyze_requirements_file(self, req_path: Path):
        """Analyze requirements file for vulnerable packages"""
        # Known vulnerable packages (simplified list)
        vulnerable_packages = {
            'flask': ['0.12.0', '0.12.1', '0.12.2'],
            'django': ['1.11.0', '1.11.1', '1.11.2'],
            'requests': ['2.19.0', '2.19.1'],
        }
        
        try:
            content = req_path.read_text()
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    self._check_package_vulnerability(req_path, line, line_num, vulnerable_packages)
        except Exception:
            pass
    
    def _check_package_vulnerability(self, req_path: Path, line: str, line_num: int, 
                                   vulnerable_packages: Dict[str, List[str]]):
        """Check if package version is vulnerable"""
        # Parse package specification
        match = re.match(r'^([a-zA-Z0-9_-]+)([><=!]+)?([\d.]+)?', line)
        if match:
            package_name = match.group(1).lower()
            version = match.group(3)
            
            if package_name in vulnerable_packages and version:
                if version in vulnerable_packages[package_name]:
                    self.findings.append(SecurityFinding(
                        severity="high",
                        category="vulnerable_dependency",
                        description=f"Vulnerable package version: {package_name} {version}",
                        file_path=str(req_path.relative_to(self.project_path)),
                        line_number=line_num,
                        evidence=line,
                        recommendation=f"Update {package_name} to latest version"
                    ))
    
    def _audit_configurations(self):
        """Audit configuration files for security issues"""
        config_files = [
            '*.json', '*.yaml', '*.yml', '*.ini', '*.cfg', '.env*'
        ]
        
        for pattern in config_files:
            for config_file in self.project_path.rglob(pattern):
                if config_file.is_file():
                    self._audit_config_file(config_file)
    
    def _audit_config_file(self, config_file: Path):
        """Audit individual configuration file"""
        try:
            content = config_file.read_text()
            
            # Check for hardcoded secrets
            secret_patterns = [
                r'password.*[:=]\s*["\'][^"\']{8,}["\']',
                r'secret.*[:=]\s*["\'][^"\']{20,}["\']',
                r'key.*[:=]\s*["\'][^"\']{20,}["\']',
            ]
            
            for pattern in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.findings.append(SecurityFinding(
                        severity="critical",
                        category="hardcoded_secrets",
                        description="Hardcoded secret in configuration file",
                        file_path=str(config_file.relative_to(self.project_path)),
                        line_number=line_num,
                        evidence=match.group(0),
                        recommendation="Use environment variables or secure key management"
                    ))
                    
        except Exception:
            pass
    
    def _check_file_permissions(self):
        """Check file permissions for security issues"""
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = stat.st_mode
                    
                    # Check for world-writable files
                    if mode & 0o002:
                        self.findings.append(SecurityFinding(
                            severity="medium",
                            category="file_permissions",
                            description="World-writable file detected",
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=0,
                            evidence=f"Permissions: {oct(mode)[-3:]}",
                            recommendation="Remove world write permissions"
                        ))
                    
                    # Check for executable config files
                    if file_path.suffix in ['.json', '.yaml', '.yml', '.ini'] and mode & 0o111:
                        self.findings.append(SecurityFinding(
                            severity="low",
                            category="file_permissions",
                            description="Executable configuration file",
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=0,
                            evidence=f"Permissions: {oct(mode)[-3:]}",
                            recommendation="Remove execute permissions from config files"
                        ))
                        
                except Exception:
                    pass
    
    def _analyze_network_usage(self):
        """Analyze network-related code for security issues"""
        network_patterns = [
            (r'requests\.get\([^)]*verify=False', 'SSL verification disabled'),
            (r'urllib\.request\.urlopen\([^)]*http://', 'Insecure HTTP usage'),
            (r'socket\.socket\(.*SOCK_RAW', 'Raw socket usage'),
        ]
        
        for py_file in self.project_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern, description in network_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self.findings.append(SecurityFinding(
                            severity="medium",
                            category="network_security",
                            description=description,
                            file_path=str(py_file.relative_to(self.project_path)),
                            line_number=line_num,
                            evidence=match.group(0),
                            recommendation="Use secure network communications"
                        ))
            except Exception:
                pass
    
    def _get_severity_for_category(self, category: str) -> str:
        """Get severity level for vulnerability category"""
        severity_map = {
            'hardcoded_secrets': 'critical',
            'command_injection': 'critical',
            'sql_injection': 'high',
            'path_traversal': 'high',
            'insecure_random': 'medium',
        }
        return severity_map.get(category, 'medium')
    
    def _get_recommendation_for_category(self, category: str) -> str:
        """Get recommendation for vulnerability category"""
        recommendations = {
            'hardcoded_secrets': 'Use environment variables or secure key management systems',
            'command_injection': 'Use parameterized commands and input validation',
            'sql_injection': 'Use parameterized queries and ORM',
            'path_traversal': 'Validate and sanitize file paths',
            'insecure_random': 'Use cryptographically secure random functions',
        }
        return recommendations.get(category, 'Review code for security implications')
    
    def _generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        # Categorize findings by severity
        by_severity = {'critical': [], 'high': [], 'medium': [], 'low': []}
        by_category = {}
        
        for finding in self.findings:
            by_severity[finding.severity].append(finding)
            if finding.category not in by_category:
                by_category[finding.category] = []
            by_category[finding.category].append(finding)
        
        # Calculate security score
        security_score = self._calculate_security_score(by_severity)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'project_path': str(self.project_path),
            'total_findings': len(self.findings),
            'security_score': security_score,
            'findings_by_severity': {
                severity: [
                    {
                        'category': f.category,
                        'description': f.description,
                        'file_path': f.file_path,
                        'line_number': f.line_number,
                        'evidence': f.evidence,
                        'recommendation': f.recommendation
                    }
                    for f in findings
                ]
                for severity, findings in by_severity.items()
            },
            'findings_by_category': {
                category: len(findings)
                for category, findings in by_category.items()
            },
            'summary': {
                'critical_issues': len(by_severity['critical']),
                'high_issues': len(by_severity['high']),
                'medium_issues': len(by_severity['medium']),
                'low_issues': len(by_severity['low']),
            }
        }
    
    def _calculate_security_score(self, by_severity: Dict[str, List]) -> int:
        """Calculate overall security score (0-100)"""
        base_score = 100
        
        # Deduct points based on severity
        deductions = {
            'critical': 25,
            'high': 10,
            'medium': 5,
            'low': 2
        }
        
        total_deduction = 0
        for severity, findings in by_severity.items():
            total_deduction += len(findings) * deductions[severity]
        
        return max(0, base_score - total_deduction)

class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor for security analysis"""
    
    def __init__(self, file_path: Path, project_path: Path):
        self.file_path = file_path
        self.project_path = project_path
        self.findings: List[SecurityFinding] = []
    
    def visit_Call(self, node):
        """Visit function calls for security issues"""
        # Check for dangerous function calls
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'subprocess' and node.func.attr in ['call', 'run', 'Popen']:
                    self._check_subprocess_call(node)
        
        elif isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec']:
                self.findings.append(SecurityFinding(
                    severity="critical",
                    category="code_injection",
                    description=f"Dangerous function: {node.func.id}",
                    file_path=str(self.file_path.relative_to(self.project_path)),
                    line_number=node.lineno,
                    evidence="",
                    recommendation="Avoid using eval() and exec()"
                ))
        
        self.generic_visit(node)
    
    def _check_subprocess_call(self, node):
        """Check subprocess calls for shell injection"""
        for keyword in node.keywords:
            if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    self.findings.append(SecurityFinding(
                        severity="high",
                        category="command_injection",
                        description="subprocess call with shell=True",
                        file_path=str(self.file_path.relative_to(self.project_path)),
                        line_number=node.lineno,
                        evidence="shell=True",
                        recommendation="Use shell=False and pass commands as list"
                    ))

# Usage
auditor = SecurityAuditor(Path("/path/to/project"))
audit_report = auditor.run_full_audit()
print(f"Security Score: {audit_report['security_score']}/100")
```

## 9. Vulnerability Response Plan

### Incident Response Procedures
```python
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class VulnerabilitySeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class IncidentStatus(Enum):
    NEW = "new"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class SecurityIncident:
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    affected_components: List[str]
    mitigation_steps: List[str]
    reporter: str
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None

class VulnerabilityResponseManager:
    """Comprehensive vulnerability response and incident management"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.incidents: Dict[str, SecurityIncident] = {}
        self.log_path = Path.home() / ".claude-tui" / "security_incidents.log"
        self.log_path.parent.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # SLA timelines (in hours)
        self.response_sla = {
            VulnerabilitySeverity.CRITICAL: 1,
            VulnerabilitySeverity.HIGH: 4,
            VulnerabilitySeverity.MEDIUM: 24,
            VulnerabilitySeverity.LOW: 72
        }
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration for incident response"""
        default_config = {
            "notification_emails": ["admin@example.com"],
            "escalation_emails": ["security@example.com"],
            "auto_mitigation": True,
            "backup_before_fix": True,
            "require_approval_for_critical": True
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def report_vulnerability(self, title: str, description: str, 
                           severity: VulnerabilitySeverity,
                           affected_components: List[str],
                           reporter: str = "system") -> str:
        """Report a new vulnerability"""
        incident_id = self._generate_incident_id()
        
        incident = SecurityIncident(
            id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.NEW,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            affected_components=affected_components,
            mitigation_steps=[],
            reporter=reporter
        )
        
        self.incidents[incident_id] = incident
        
        # Log the incident
        self.logger.critical(
            f"SECURITY INCIDENT {incident_id}: {title} "
            f"(Severity: {severity.value}, Components: {', '.join(affected_components)})"
        )
        
        # Send notifications
        self._send_incident_notification(incident)
        
        # Auto-initiate response if configured
        if severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]:
            self._initiate_emergency_response(incident_id)
        
        return incident_id
    
    def update_incident_status(self, incident_id: str, new_status: IncidentStatus,
                             notes: Optional[str] = None) -> bool:
        """Update incident status"""
        if incident_id not in self.incidents:
            self.logger.error(f"Incident {incident_id} not found")
            return False
        
        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = new_status
        incident.updated_at = datetime.utcnow()
        
        if notes:
            incident.mitigation_steps.append(f"{datetime.utcnow().isoformat()}: {notes}")
        
        self.logger.info(
            f"Incident {incident_id} status changed: {old_status.value} -> {new_status.value}"
        )
        
        # Send status update notification
        self._send_status_update_notification(incident, old_status)
        
        return True
    
    def _initiate_emergency_response(self, incident_id: str):
        """Initiate emergency response procedures"""
        incident = self.incidents[incident_id]
        
        self.logger.critical(f"EMERGENCY RESPONSE INITIATED for {incident_id}")
        
        # Update status
        incident.status = IncidentStatus.INVESTIGATING
        incident.updated_at = datetime.utcnow()
        
        # Auto-mitigation for known vulnerability types
        if self.config.get("auto_mitigation", True):
            mitigation_applied = self._apply_auto_mitigation(incident)
            if mitigation_applied:
                incident.mitigation_steps.append(
                    f"{datetime.utcnow().isoformat()}: Auto-mitigation applied"
                )
        
        # Create system backup if critical
        if incident.severity == VulnerabilitySeverity.CRITICAL and self.config.get("backup_before_fix", True):
            self._create_emergency_backup(incident_id)
        
        # Escalate if required
        if (incident.severity == VulnerabilitySeverity.CRITICAL and 
            self.config.get("require_approval_for_critical", True)):
            self._escalate_to_security_team(incident)
    
    def _apply_auto_mitigation(self, incident: SecurityIncident) -> bool:
        """Apply automatic mitigation based on incident type"""
        mitigation_rules = {
            "hardcoded_secret": self._mitigate_hardcoded_secret,
            "command_injection": self._mitigate_command_injection,
            "path_traversal": self._mitigate_path_traversal,
            "vulnerable_dependency": self._mitigate_vulnerable_dependency
        }
        
        # Analyze incident description to determine type
        incident_type = self._classify_incident_type(incident.description)
        
        if incident_type in mitigation_rules:
            try:
                return mitigation_rules[incident_type](incident)
            except Exception as e:
                self.logger.error(f"Auto-mitigation failed for {incident.id}: {e}")
                return False
        
        return False
    
    def _classify_incident_type(self, description: str) -> str:
        """Classify incident type based on description"""
        keywords = {
            "hardcoded_secret": ["api key", "password", "secret", "token"],
            "command_injection": ["subprocess", "os.system", "shell=True"],
            "path_traversal": ["../", "path traversal", "directory traversal"],
            "vulnerable_dependency": ["vulnerable package", "CVE-", "outdated dependency"]
        }
        
        description_lower = description.lower()
        
        for incident_type, type_keywords in keywords.items():
            if any(keyword in description_lower for keyword in type_keywords):
                return incident_type
        
        return "unknown"
    
    def _mitigate_hardcoded_secret(self, incident: SecurityIncident) -> bool:
        """Mitigate hardcoded secret vulnerabilities"""
        self.logger.info(f"Applying hardcoded secret mitigation for {incident.id}")
        
        # Create incident-specific mitigation
        mitigation_steps = [
            "1. Rotate affected API keys/secrets immediately",
            "2. Remove hardcoded secrets from code",
            "3. Implement environment variable usage",
            "4. Add secrets to .gitignore",
            "5. Audit git history for secret exposure"
        ]
        
        incident.mitigation_steps.extend([
            f"{datetime.utcnow().isoformat()}: {step}" for step in mitigation_steps
        ])
        
        return True
    
    def _mitigate_command_injection(self, incident: SecurityIncident) -> bool:
        """Mitigate command injection vulnerabilities"""
        self.logger.info(f"Applying command injection mitigation for {incident.id}")
        
        mitigation_steps = [
            "1. Disable shell=True in subprocess calls",
            "2. Use parameterized command execution",
            "3. Implement input validation and sanitization",
            "4. Apply principle of least privilege",
            "5. Monitor system for unusual process execution"
        ]
        
        incident.mitigation_steps.extend([
            f"{datetime.utcnow().isoformat()}: {step}" for step in mitigation_steps
        ])
        
        return True
    
    def _mitigate_path_traversal(self, incident: SecurityIncident) -> bool:
        """Mitigate path traversal vulnerabilities"""
        self.logger.info(f"Applying path traversal mitigation for {incident.id}")
        
        mitigation_steps = [
            "1. Implement strict path validation",
            "2. Use Path.resolve() for path normalization",
            "3. Whitelist allowed file extensions",
            "4. Restrict file system access to specific directories",
            "5. Add logging for file access attempts"
        ]
        
        incident.mitigation_steps.extend([
            f"{datetime.utcnow().isoformat()}: {step}" for step in mitigation_steps
        ])
        
        return True
    
    def _mitigate_vulnerable_dependency(self, incident: SecurityIncident) -> bool:
        """Mitigate vulnerable dependency issues"""
        self.logger.info(f"Applying vulnerable dependency mitigation for {incident.id}")
        
        mitigation_steps = [
            "1. Update vulnerable packages to latest versions",
            "2. Review changelogs for breaking changes",
            "3. Run tests to ensure compatibility",
            "4. Implement dependency scanning in CI/CD",
            "5. Set up automated security updates"
        ]
        
        incident.mitigation_steps.extend([
            f"{datetime.utcnow().isoformat()}: {step}" for step in mitigation_steps
        ])
        
        return True
    
    def _create_emergency_backup(self, incident_id: str):
        """Create emergency backup before applying fixes"""
        backup_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"emergency_backup_{incident_id}_{backup_time}"
        
        # Implementation would depend on backup strategy
        self.logger.info(f"Emergency backup created: {backup_name}")
        
        # Update incident with backup information
        self.incidents[incident_id].mitigation_steps.append(
            f"{datetime.utcnow().isoformat()}: Emergency backup created: {backup_name}"
        )
    
    def _send_incident_notification(self, incident: SecurityIncident):
        """Send incident notification to relevant stakeholders"""
        try:
            notification_emails = self.config.get("notification_emails", [])
            
            if incident.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]:
                notification_emails.extend(self.config.get("escalation_emails", []))
            
            subject = f"SECURITY INCIDENT {incident.id}: {incident.title} ({incident.severity.value.upper()})"
            
            body = f"""
Security Incident Report

Incident ID: {incident.id}
Title: {incident.title}
Severity: {incident.severity.value.upper()}
Status: {incident.status.value}
Reporter: {incident.reporter}
Created: {incident.created_at.isoformat()}

Description:
{incident.description}

Affected Components:
{', '.join(incident.affected_components)}

SLA Response Time: {self.response_sla[incident.severity]} hours

This is an automated notification from claude-tui Security Response System.
"""
            
            self._send_email_notification(notification_emails, subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send incident notification: {e}")
    
    def _send_status_update_notification(self, incident: SecurityIncident, 
                                       old_status: IncidentStatus):
        """Send status update notification"""
        try:
            notification_emails = self.config.get("notification_emails", [])
            
            subject = f"INCIDENT UPDATE {incident.id}: Status changed to {incident.status.value.upper()}"
            
            body = f"""
Security Incident Status Update

Incident ID: {incident.id}
Title: {incident.title}
Previous Status: {old_status.value}
New Status: {incident.status.value}
Updated: {incident.updated_at.isoformat()}

Recent Mitigation Steps:
{chr(10).join(incident.mitigation_steps[-5:]) if incident.mitigation_steps else 'None'}

This is an automated notification from claude-tui Security Response System.
"""
            
            self._send_email_notification(notification_emails, subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send status update notification: {e}")
    
    def _escalate_to_security_team(self, incident: SecurityIncident):
        """Escalate incident to security team"""
        escalation_emails = self.config.get("escalation_emails", [])
        
        if not escalation_emails:
            self.logger.warning(f"No escalation emails configured for incident {incident.id}")
            return
        
        subject = f"CRITICAL SECURITY ESCALATION {incident.id}: {incident.title}"
        
        body = f"""
CRITICAL SECURITY INCIDENT ESCALATION

This incident requires immediate security team attention.

Incident ID: {incident.id}
Title: {incident.title}
Severity: {incident.severity.value.upper()}
Created: {incident.created_at.isoformat()}
SLA Deadline: {(incident.created_at + timedelta(hours=self.response_sla[incident.severity])).isoformat()}

Description:
{incident.description}

Affected Components:
{', '.join(incident.affected_components)}

Automatic mitigation has been initiated where possible.
Manual intervention may be required.

This is an automated escalation from claude-tui Security Response System.
"""
        
        try:
            self._send_email_notification(escalation_emails, subject, body)
            self.logger.critical(f"Incident {incident.id} escalated to security team")
        except Exception as e:
            self.logger.error(f"Failed to escalate incident {incident.id}: {e}")
    
    def _send_email_notification(self, recipients: List[str], subject: str, body: str):
        """Send email notification"""
        # Note: This is a simplified implementation
        # In production, use a proper email service or SMTP configuration
        
        # For now, just log the email content
        self.logger.info(f"EMAIL NOTIFICATION:")
        self.logger.info(f"To: {', '.join(recipients)}")
        self.logger.info(f"Subject: {subject}")
        self.logger.info(f"Body: {body[:200]}...")
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"SEC-{timestamp}-{len(self.incidents):03d}"
    
    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get current incident status"""
        if incident_id not in self.incidents:
            return None
        
        incident = self.incidents[incident_id]
        return {
            "id": incident.id,
            "title": incident.title,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "created_at": incident.created_at.isoformat(),
            "updated_at": incident.updated_at.isoformat(),
            "affected_components": incident.affected_components,
            "mitigation_steps": incident.mitigation_steps,
            "sla_deadline": (incident.created_at + timedelta(
                hours=self.response_sla[incident.severity]
            )).isoformat()
        }
    
    def get_all_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents"""
        active_statuses = [IncidentStatus.NEW, IncidentStatus.INVESTIGATING, IncidentStatus.MITIGATING]
        
        active_incidents = []
        for incident in self.incidents.values():
            if incident.status in active_statuses:
                active_incidents.append({
                    "id": incident.id,
                    "title": incident.title,
                    "severity": incident.severity.value,
                    "status": incident.status.value,
                    "created_at": incident.created_at.isoformat(),
                    "affected_components": incident.affected_components
                })
        
        return sorted(active_incidents, key=lambda x: x["created_at"], reverse=True)
    
    def generate_incident_report(self, time_range_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive incident report"""
        cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
        
        recent_incidents = [
            incident for incident in self.incidents.values()
            if incident.created_at >= cutoff_date
        ]
        
        # Statistics
        by_severity = {severity: 0 for severity in VulnerabilitySeverity}
        by_status = {status: 0 for status in IncidentStatus}
        
        for incident in recent_incidents:
            by_severity[incident.severity] += 1
            by_status[incident.status] += 1
        
        # Calculate average resolution time
        resolved_incidents = [
            incident for incident in recent_incidents
            if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]
        ]
        
        avg_resolution_time = 0
        if resolved_incidents:
            total_time = sum(
                (incident.updated_at - incident.created_at).total_seconds()
                for incident in resolved_incidents
            )
            avg_resolution_time = total_time / len(resolved_incidents) / 3600  # Convert to hours
        
        return {
            "report_period_days": time_range_days,
            "total_incidents": len(recent_incidents),
            "by_severity": {sev.value: count for sev, count in by_severity.items()},
            "by_status": {status.value: count for status, count in by_status.items()},
            "average_resolution_time_hours": round(avg_resolution_time, 2),
            "active_incidents": len([i for i in recent_incidents if i.status in [
                IncidentStatus.NEW, IncidentStatus.INVESTIGATING, IncidentStatus.MITIGATING
            ]]),
            "sla_compliance": self._calculate_sla_compliance(recent_incidents)
        }
    
    def _calculate_sla_compliance(self, incidents: List[SecurityIncident]) -> Dict[str, float]:
        """Calculate SLA compliance rates"""
        compliance_by_severity = {}
        
        for severity in VulnerabilitySeverity:
            severity_incidents = [i for i in incidents if i.severity == severity]
            
            if not severity_incidents:
                compliance_by_severity[severity.value] = 100.0
                continue
            
            sla_hours = self.response_sla[severity]
            compliant_count = 0
            
            for incident in severity_incidents:
                if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                    resolution_time = (incident.updated_at - incident.created_at).total_seconds() / 3600
                    if resolution_time <= sla_hours:
                        compliant_count += 1
            
            compliance_rate = (compliant_count / len(severity_incidents)) * 100
            compliance_by_severity[severity.value] = round(compliance_rate, 2)
        
        return compliance_by_severity

# Usage example
response_manager = VulnerabilityResponseManager()

# Report a vulnerability
incident_id = response_manager.report_vulnerability(
    title="Hardcoded API Key Detected",
    description="Found hardcoded API key in config.py line 45",
    severity=VulnerabilitySeverity.CRITICAL,
    affected_components=["authentication", "api_client"],
    reporter="automated_scan"
)

# Update incident status
response_manager.update_incident_status(
    incident_id, 
    IncidentStatus.MITIGATING,
    "Applied temporary fix, rotating API keys"
)

# Generate report
report = response_manager.generate_incident_report(30)
print(f"Security incidents in last 30 days: {report['total_incidents']}")
```

## Security Best Practices Summary

1. **Input Validation**: Validate all user inputs with whitelisting approach
2. **Principle of Least Privilege**: Minimal necessary permissions
3. **Defense in Depth**: Multiple security layers
4. **Secure by Default**: Security-first configuration
5. **Regular Auditing**: Automated and manual security reviews
6. **Incident Response**: Documented procedures for security events
7. **Encryption**: Protect data at rest and in transit
8. **Access Control**: Strong authentication and authorization
9. **Monitoring**: Continuous security monitoring
10. **Update Management**: Keep dependencies current and secure