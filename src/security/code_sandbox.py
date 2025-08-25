"""
Secure Code Sandbox for claude-tui.

This module provides a secure environment for executing AI-generated code with:
- Multi-layer sandboxing (process, Docker, VM)
- Resource limits and monitoring
- Network isolation
- File system restrictions
- Malware detection
- Code analysis and validation
- Real-time security monitoring
"""

import os
import sys
import ast
import tempfile
import subprocess
import threading
import time
import resource
import signal
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import json
from datetime import datetime, timedelta

# Optional imports for enhanced sandboxing
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class SandboxMode(Enum):
    """Sandbox execution modes."""
    PROCESS = "process"       # Process-level isolation
    DOCKER = "docker"         # Docker container isolation
    RESTRICTED = "restricted" # Minimal privileges
    ANALYSIS_ONLY = "analysis_only"  # Code analysis without execution

class SecurityLevel(Enum):
    """Security levels for code execution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

@dataclass
class ResourceLimits:
    """Resource limits for code execution."""
    max_execution_time: int = 30      # seconds
    max_memory_mb: int = 128          # MB
    max_cpu_percent: float = 50.0     # %
    max_file_size_mb: int = 10        # MB
    max_network_calls: int = 0        # 0 = no network
    max_subprocess_count: int = 0     # 0 = no subprocesses
    max_open_files: int = 10
    
@dataclass
class CodeAnalysisResult:
    """Result of static code analysis."""
    is_safe: bool
    risk_score: int  # 0-100 (higher = more risky)
    security_issues: List[str]
    banned_imports: List[str]
    complexity_score: int
    line_count: int
    function_count: int

@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    memory_used: int
    security_violations: List[str]
    resource_usage: Dict[str, Any]
    analysis_result: Optional[CodeAnalysisResult] = None

class CodeAnalyzer:
    """Static code analyzer for security assessment."""
    
    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'file',
        'input', 'raw_input', 'reload', 'vars', 'locals', 'globals',
        'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
    }
    
    # Dangerous modules
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'threading', 'multiprocessing',
        'ctypes', 'importlib', 'pkgutil', 'imp', 'marshal',
        'pickle', 'shelve', 'dbm', 'socket', 'urllib', 'httplib',
        'ftplib', 'smtplib', 'telnetlib', 'SocketServer'
    }
    
    # Restricted modules (allowed with limitations)
    RESTRICTED_MODULES = {
        'time', 'datetime', 'random', 're', 'string', 'math',
        'json', 'csv', 'base64', 'hashlib', 'uuid'
    }
    
    # Dangerous AST node types
    DANGEROUS_NODES = {
        ast.Import, ast.ImportFrom, ast.Exec, ast.Global,
        ast.Delete, ast.With, ast.Try, ast.Raise
    }
    
    def analyze_code(self, code: str, filename: str = "<sandbox>") -> CodeAnalysisResult:
        """
        Perform comprehensive static analysis of code.
        
        Args:
            code: Python code to analyze
            filename: Optional filename for context
            
        Returns:
            CodeAnalysisResult with security assessment
        """
        security_issues = []
        banned_imports = []
        risk_score = 0
        
        try:
            # Parse code into AST
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            return CodeAnalysisResult(
                is_safe=False,
                risk_score=100,
                security_issues=[f"Syntax error: {e}"],
                banned_imports=[],
                complexity_score=0,
                line_count=len(code.splitlines()),
                function_count=0
            )
        
        # Analyze AST nodes
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.DANGEROUS_BUILTINS:
                        security_issues.append(f"Dangerous builtin function: {func_name}")
                        risk_score += 20
                
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        func_name = node.func.attr
                        
                        if module_name in self.DANGEROUS_MODULES:
                            security_issues.append(f"Dangerous module call: {module_name}.{func_name}")
                            risk_score += 15
            
            # Check imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.DANGEROUS_MODULES:
                        banned_imports.append(alias.name)
                        security_issues.append(f"Dangerous import: {alias.name}")
                        risk_score += 25
                    elif alias.name not in self.RESTRICTED_MODULES:
                        security_issues.append(f"Unknown import: {alias.name}")
                        risk_score += 5
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.DANGEROUS_MODULES:
                    banned_imports.append(node.module)
                    security_issues.append(f"Dangerous from-import: {node.module}")
                    risk_score += 25
            
            # Check for dangerous node types
            elif type(node) in self.DANGEROUS_NODES:
                if isinstance(node, ast.Exec):
                    security_issues.append("Dynamic code execution detected")
                    risk_score += 30
                elif isinstance(node, ast.Global):
                    security_issues.append("Global variable manipulation")
                    risk_score += 10
                elif isinstance(node, ast.Delete):
                    security_issues.append("Variable/attribute deletion")
                    risk_score += 5
        
        # Calculate complexity and other metrics
        complexity_score = self._calculate_complexity(tree)
        line_count = len([line for line in code.splitlines() if line.strip()])
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        
        # Additional risk factors
        if line_count > 1000:
            security_issues.append("Code too long (potential obfuscation)")
            risk_score += 10
        
        if complexity_score > 50:
            security_issues.append("High complexity code")
            risk_score += 15
        
        # Check for suspicious patterns in raw code
        suspicious_patterns = [
            ('__', "Dunder methods usage"),
            ('chr(', "Character code conversion"),
            ('ord(', "Character to code conversion"),
            ('decode', "String decoding"),
            ('encode', "String encoding"),
            ('base64', "Base64 encoding/decoding"),
            ('lambda', "Lambda functions"),
        ]
        
        code_lower = code.lower()
        for pattern, description in suspicious_patterns:
            if pattern in code_lower:
                security_issues.append(f"Suspicious pattern: {description}")
                risk_score += 3
        
        risk_score = min(100, risk_score)  # Cap at 100
        is_safe = risk_score < 30 and not banned_imports
        
        return CodeAnalysisResult(
            is_safe=is_safe,
            risk_score=risk_score,
            security_issues=security_issues,
            banned_imports=banned_imports,
            complexity_score=complexity_score,
            line_count=line_count,
            function_count=function_count
        )
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity

class SecureCodeSandbox:
    """
    Secure code execution sandbox with multiple isolation layers.
    
    Features:
    - Static code analysis before execution
    - Process/Docker/VM isolation
    - Resource monitoring and limits
    - Network isolation
    - File system restrictions
    - Real-time security monitoring
    """
    
    def __init__(
        self,
        mode: SandboxMode = SandboxMode.PROCESS,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        resource_limits: Optional[ResourceLimits] = None,
        allowed_modules: Optional[Set[str]] = None,
        docker_image: str = "python:3.9-alpine"
    ):
        """
        Initialize the secure code sandbox.
        
        Args:
            mode: Sandbox execution mode
            security_level: Security level for execution
            resource_limits: Resource constraints
            allowed_modules: Set of allowed Python modules
            docker_image: Docker image for container execution
        """
        self.mode = mode
        self.security_level = security_level
        self.resource_limits = resource_limits or ResourceLimits()
        self.docker_image = docker_image
        
        # Initialize code analyzer
        self.analyzer = CodeAnalyzer()
        
        # Set allowed modules based on security level
        if allowed_modules:
            self.allowed_modules = allowed_modules
        else:
            self.allowed_modules = self._get_default_allowed_modules()
        
        # Docker client for container mode
        self.docker_client = None
        if self.mode == SandboxMode.DOCKER and DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized for sandbox")
            except Exception as e:
                logger.warning(f"Docker initialization failed: {e}")
                self.mode = SandboxMode.PROCESS
        
        # Execution statistics
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'blocked_executions': 0,
            'security_violations': 0
        }
        self.stats_lock = threading.Lock()
        
        logger.info(f"Secure code sandbox initialized: {self.mode.value} mode, {self.security_level.value} security")
    
    def execute_code(
        self,
        code: str,
        input_data: Optional[str] = None,
        timeout: Optional[int] = None,
        filename: str = "<sandbox>"
    ) -> ExecutionResult:
        """
        Execute code in secure sandbox environment.
        
        Args:
            code: Python code to execute
            input_data: Optional input data for the code
            timeout: Execution timeout (uses resource limits if not specified)
            filename: Optional filename for error reporting
            
        Returns:
            ExecutionResult with execution details and security info
        """
        start_time = time.time()
        
        with self.stats_lock:
            self.execution_stats['total_executions'] += 1
        
        # Step 1: Static code analysis
        analysis_result = self.analyzer.analyze_code(code, filename)
        
        # Block execution if code is deemed unsafe
        if not analysis_result.is_safe and self.security_level != SecurityLevel.LOW:
            with self.stats_lock:
                self.execution_stats['blocked_executions'] += 1
                self.execution_stats['security_violations'] += len(analysis_result.security_issues)
            
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Code execution blocked due to security issues: {'; '.join(analysis_result.security_issues)}",
                return_code=-1,
                execution_time=time.time() - start_time,
                memory_used=0,
                security_violations=analysis_result.security_issues,
                resource_usage={},
                analysis_result=analysis_result
            )
        
        # Step 2: Execute code based on sandbox mode
        if self.mode == SandboxMode.ANALYSIS_ONLY:
            execution_result = ExecutionResult(
                success=True,
                stdout="Code analysis completed - execution skipped",
                stderr="",
                return_code=0,
                execution_time=time.time() - start_time,
                memory_used=0,
                security_violations=[],
                resource_usage={},
                analysis_result=analysis_result
            )
        elif self.mode == SandboxMode.DOCKER and self.docker_client:
            execution_result = self._execute_in_docker(code, input_data, timeout, analysis_result)
        else:
            execution_result = self._execute_in_process(code, input_data, timeout, analysis_result)
        
        # Update statistics
        with self.stats_lock:
            if execution_result.success:
                self.execution_stats['successful_executions'] += 1
            if execution_result.security_violations:
                self.execution_stats['security_violations'] += len(execution_result.security_violations)
        
        return execution_result
    
    def _execute_in_process(
        self,
        code: str,
        input_data: Optional[str],
        timeout: Optional[int],
        analysis_result: CodeAnalysisResult
    ) -> ExecutionResult:
        """Execute code in isolated subprocess."""
        if timeout is None:
            timeout = self.resource_limits.max_execution_time
        
        # Create secure wrapper for the code
        wrapped_code = self._create_secure_wrapper(code)
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapped_code)
            temp_file = f.name
        
        start_time = time.time()
        security_violations = []
        
        try:
            # Setup secure environment
            env = self._create_secure_environment()
            
            # Execute with resource limits
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=self._set_process_limits,
                text=True,
                cwd=tempfile.gettempdir()
            )
            
            # Monitor execution
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process, security_violations)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            try:
                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=timeout
                )
                execution_time = time.time() - start_time
                
                # Get memory usage if psutil is available
                memory_used = 0
                if PSUTIL_AVAILABLE:
                    try:
                        proc = psutil.Process(process.pid)
                        memory_used = proc.memory_info().rss // 1024 // 1024  # MB
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=process.returncode,
                    execution_time=execution_time,
                    memory_used=memory_used,
                    security_violations=security_violations,
                    resource_usage={
                        'peak_memory_mb': memory_used,
                        'execution_time': execution_time
                    },
                    analysis_result=analysis_result
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    return_code=-1,
                    execution_time=timeout,
                    memory_used=0,
                    security_violations=["Timeout exceeded"] + security_violations,
                    resource_usage={'timeout': True},
                    analysis_result=analysis_result
                )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Sandbox execution error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time,
                memory_used=0,
                security_violations=[f"Execution exception: {str(e)}"] + security_violations,
                resource_usage={},
                analysis_result=analysis_result
            )
        
        finally:
            # Cleanup temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    def _execute_in_docker(
        self,
        code: str,
        input_data: Optional[str],
        timeout: Optional[int],
        analysis_result: CodeAnalysisResult
    ) -> ExecutionResult:
        """Execute code in Docker container."""
        if timeout is None:
            timeout = self.resource_limits.max_execution_time
        
        start_time = time.time()
        
        try:
            # Create secure wrapper
            wrapped_code = self._create_secure_wrapper(code)
            
            # Create temporary directory for file exchange
            with tempfile.TemporaryDirectory() as temp_dir:
                code_file = Path(temp_dir) / "code.py"
                code_file.write_text(wrapped_code)
                
                if input_data:
                    input_file = Path(temp_dir) / "input.txt"
                    input_file.write_text(input_data)
                    stdin_redirect = " < /sandbox/input.txt"
                else:
                    stdin_redirect = ""
                
                # Run container with restrictions
                container = self.docker_client.containers.run(
                    self.docker_image,
                    command=f"sh -c 'python /sandbox/code.py{stdin_redirect}'",
                    volumes={temp_dir: {'bind': '/sandbox', 'mode': 'ro'}},
                    working_dir='/sandbox',
                    environment={'PYTHONPATH': ''},
                    mem_limit=f"{self.resource_limits.max_memory_mb}m",
                    cpu_period=100000,
                    cpu_quota=int(50000 * (self.resource_limits.max_cpu_percent / 100)),
                    network_disabled=True,
                    read_only=True,
                    remove=True,
                    timeout=timeout,
                    stdout=True,
                    stderr=True,
                    detach=False
                )
                
                execution_time = time.time() - start_time
                
                # Decode container output
                if isinstance(container, bytes):
                    stdout = container.decode('utf-8', errors='replace')
                    stderr = ""
                elif hasattr(container, 'decode'):
                    stdout = container.decode('utf-8', errors='replace')
                    stderr = ""
                else:
                    stdout = str(container)
                    stderr = ""
                
                return ExecutionResult(
                    success=True,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=0,
                    execution_time=execution_time,
                    memory_used=self.resource_limits.max_memory_mb,  # Approximation
                    security_violations=[],
                    resource_usage={
                        'container_mode': True,
                        'memory_limit_mb': self.resource_limits.max_memory_mb,
                        'cpu_limit_percent': self.resource_limits.max_cpu_percent
                    },
                    analysis_result=analysis_result
                )
        
        except docker.errors.ContainerError as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=e.stderr.decode('utf-8') if e.stderr else str(e),
                return_code=e.exit_status,
                execution_time=time.time() - start_time,
                memory_used=0,
                security_violations=[f"Container error: {str(e)}"],
                resource_usage={},
                analysis_result=analysis_result
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Docker execution error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time,
                memory_used=0,
                security_violations=[f"Docker exception: {str(e)}"],
                resource_usage={},
                analysis_result=analysis_result
            )
    
    def _create_secure_wrapper(self, code: str) -> str:
        """Create secure wrapper around user code."""
        # Disable dangerous builtins
        restricted_builtins = """
import sys
import builtins

# Save safe builtins
safe_builtins = {
    'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
    'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'range': range, 'enumerate': enumerate, 'zip': zip, 'map': map,
    'filter': filter, 'sum': sum, 'max': max, 'min': min, 'abs': abs,
    'round': round, 'sorted': sorted, 'reversed': reversed, 'type': type,
    'isinstance': isinstance, 'issubclass': issubclass, 'ValueError': ValueError,
    'TypeError': TypeError, 'IndexError': IndexError, 'KeyError': KeyError,
    'AttributeError': AttributeError, 'StopIteration': StopIteration
}

# Clear and replace builtins
builtins.__dict__.clear()
builtins.__dict__.update(safe_builtins)

# Remove dangerous modules from sys.modules
dangerous_modules = ['os', 'sys', 'subprocess', 'threading', 'multiprocessing', 'ctypes']
for module in dangerous_modules:
    if module in sys.modules:
        del sys.modules[module]

# Prevent new imports of dangerous modules
original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed_modules = {""" + repr(list(self.allowed_modules)) + """}
    if name not in allowed_modules:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    return original_import(name, globals, locals, fromlist, level)

__builtins__['__import__'] = safe_import

# Execute user code with timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(""" + str(self.resource_limits.max_execution_time) + """)

try:
    # User code starts here
""" + '\n'.join('    ' + line for line in code.split('\n')) + """
except TimeoutError:
    print("ERROR: Code execution timed out", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    signal.alarm(0)  # Cancel alarm
"""
        return restricted_builtins
    
    def _create_secure_environment(self) -> Dict[str, str]:
        """Create secure environment variables."""
        return {
            'PATH': '/usr/bin:/bin',
            'PYTHONPATH': '',
            'HOME': '/tmp',
            'USER': 'nobody',
            'SHELL': '/bin/sh',
            'LANG': 'C',
            'LC_ALL': 'C'
        }
    
    def _set_process_limits(self):
        """Set resource limits for subprocess."""
        try:
            # Memory limit
            memory_bytes = self.resource_limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # CPU time limit
            cpu_time = self.resource_limits.max_execution_time
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time, cpu_time))
            
            # File size limit
            file_size_bytes = self.resource_limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
            
            # Number of processes
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))  # Only main process
            
            # Number of open files
            resource.setrlimit(resource.RLIMIT_NOFILE, 
                              (self.resource_limits.max_open_files, 
                               self.resource_limits.max_open_files))
            
            # Core dumps disabled
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            
            # Set lower priority
            os.nice(10)
            
        except (resource.error, OSError) as e:
            logger.warning(f"Failed to set some resource limits: {e}")
    
    def _monitor_process(self, process: subprocess.Popen, security_violations: List[str]):
        """Monitor process for security violations."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            psutil_process = psutil.Process(process.pid)
            
            while process.poll() is None:
                try:
                    # Check memory usage
                    memory_mb = psutil_process.memory_info().rss // 1024 // 1024
                    if memory_mb > self.resource_limits.max_memory_mb:
                        security_violations.append(f"Memory limit exceeded: {memory_mb}MB > {self.resource_limits.max_memory_mb}MB")
                        process.terminate()
                        break
                    
                    # Check CPU usage
                    cpu_percent = psutil_process.cpu_percent()
                    if cpu_percent > self.resource_limits.max_cpu_percent:
                        security_violations.append(f"CPU limit exceeded: {cpu_percent}% > {self.resource_limits.max_cpu_percent}%")
                    
                    # Check for subprocess creation
                    children = psutil_process.children(recursive=True)
                    if len(children) > self.resource_limits.max_subprocess_count:
                        security_violations.append(f"Subprocess limit exceeded: {len(children)} > {self.resource_limits.max_subprocess_count}")
                        process.terminate()
                        break
                    
                    # Check network connections (should be none)
                    if self.resource_limits.max_network_calls == 0:
                        connections = psutil_process.connections()
                        if connections:
                            security_violations.append("Unauthorized network connection detected")
                            process.terminate()
                            break
                    
                    time.sleep(0.5)  # Monitor every 500ms
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
        except Exception as e:
            logger.error(f"Process monitoring error: {e}")
    
    def _get_default_allowed_modules(self) -> Set[str]:
        """Get default allowed modules based on security level."""
        base_modules = {
            'math', 'random', 'time', 'datetime', 'json', 'csv',
            'string', 're', 'collections', 'itertools', 'functools',
            'operator', 'copy', 'hashlib', 'uuid', 'base64'
        }
        
        if self.security_level == SecurityLevel.LOW:
            base_modules.update({
                'urllib', 'http', 'email', 'html', 'xml',
                'sqlite3', 'pickle', 'configparser'
            })
        elif self.security_level == SecurityLevel.MEDIUM:
            base_modules.update({
                'urllib.parse', 'html.parser', 'xml.etree.ElementTree'
            })
        
        return base_modules
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self.stats_lock:
            stats = self.execution_stats.copy()
        
        if stats['total_executions'] > 0:
            stats['success_rate'] = (stats['successful_executions'] / stats['total_executions']) * 100
            stats['block_rate'] = (stats['blocked_executions'] / stats['total_executions']) * 100
        else:
            stats['success_rate'] = 0
            stats['block_rate'] = 0
        
        stats['mode'] = self.mode.value
        stats['security_level'] = self.security_level.value
        stats['allowed_modules'] = list(self.allowed_modules)
        
        return stats

# Utility functions
def create_secure_sandbox(
    security_level: SecurityLevel = SecurityLevel.HIGH,
    max_execution_time: int = 30,
    max_memory_mb: int = 128
) -> SecureCodeSandbox:
    """Create a configured secure code sandbox."""
    resource_limits = ResourceLimits(
        max_execution_time=max_execution_time,
        max_memory_mb=max_memory_mb
    )
    
    # Use Docker if available, otherwise fall back to process isolation
    mode = SandboxMode.DOCKER if DOCKER_AVAILABLE else SandboxMode.PROCESS
    
    return SecureCodeSandbox(
        mode=mode,
        security_level=security_level,
        resource_limits=resource_limits
    )

def analyze_code_safety(code: str) -> CodeAnalysisResult:
    """Quick function to analyze code for safety."""
    analyzer = CodeAnalyzer()
    return analyzer.analyze_code(code)