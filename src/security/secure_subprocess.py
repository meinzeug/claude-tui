"""
Secure subprocess execution system for claude-tiu.

This module provides enterprise-grade security for subprocess execution with:
- Command whitelisting and argument validation
- Resource limits and sandboxing
- Environment variable restriction
- Process monitoring and timeout handling
- Logging and audit trails
- Docker-based isolation (optional)
"""

import os
import re
import signal
import shlex
import subprocess
import tempfile
import threading
import time
import resource
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import json
from datetime import datetime, timedelta

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Subprocess execution modes."""
    RESTRICTED = "restricted"     # Limited commands, strict validation
    SANDBOXED = "sandboxed"       # Process isolation
    DOCKER = "docker"             # Docker container isolation
    MINIMAL = "minimal"           # Absolute minimum privileges

@dataclass
class ResourceLimits:
    """Resource limits for subprocess execution."""
    memory_mb: int = 256
    cpu_time_seconds: int = 30
    wall_time_seconds: int = 60
    max_processes: int = 5
    max_file_size_mb: int = 100
    max_open_files: int = 100

@dataclass
class ExecutionResult:
    """Result of a subprocess execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    security_violations: List[str] = field(default_factory=list)
    command_hash: Optional[str] = None

@dataclass
class CommandPolicy:
    """Security policy for command execution."""
    command: str
    allowed_args: Set[str] = field(default_factory=set)
    forbidden_args: Set[str] = field(default_factory=set)
    max_execution_time: int = 30
    allowed_paths: Set[str] = field(default_factory=set)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    requires_confirmation: bool = False

class SecureSubprocessManager:
    """
    Enterprise-grade secure subprocess execution manager.
    
    Features:
    - Command validation and whitelisting
    - Resource limits enforcement
    - Environment sandboxing
    - Docker isolation (optional)
    - Comprehensive logging and monitoring
    """
    
    # Default command policies
    DEFAULT_POLICIES = {
        'git': CommandPolicy(
            command='git',
            allowed_args={
                'status', 'add', 'commit', 'push', 'pull', 'branch', 'checkout',
                'clone', 'diff', 'log', 'remote', 'init', 'config', '--version'
            },
            forbidden_args={'--upload-pack', '--receive-pack', '--exec'},
            max_execution_time=300,  # 5 minutes for git operations
            environment_vars={'GIT_TERMINAL_PROMPT': '0'}
        ),
        'npm': CommandPolicy(
            command='npm',
            allowed_args={
                'install', 'test', 'run', 'start', 'build', 'init', 'list',
                'version', 'audit', 'update', '--version'
            },
            max_execution_time=600,  # 10 minutes for npm operations
            environment_vars={'NPM_CONFIG_PROGRESS': 'false'}
        ),
        'python': CommandPolicy(
            command='python',
            allowed_args={'-c', '-m', '-V', '--version', 'setup.py'},
            forbidden_args={'-i', '--interactive'},
            max_execution_time=60
        ),
        'python3': CommandPolicy(
            command='python3',
            allowed_args={'-c', '-m', '-V', '--version', 'setup.py'},
            forbidden_args={'-i', '--interactive'},
            max_execution_time=60
        ),
        'node': CommandPolicy(
            command='node',
            allowed_args={'-v', '--version', '-e', '--eval'},
            max_execution_time=60
        ),
        'pip': CommandPolicy(
            command='pip',
            allowed_args={
                'install', 'list', 'show', 'freeze', 'uninstall', '--version'
            },
            max_execution_time=300
        ),
        'pip3': CommandPolicy(
            command='pip3',
            allowed_args={
                'install', 'list', 'show', 'freeze', 'uninstall', '--version'
            },
            max_execution_time=300
        )
    }
    
    # Completely forbidden commands
    FORBIDDEN_COMMANDS = {
        'rm', 'rmdir', 'del', 'sudo', 'su', 'chmod', 'chown', 'passwd',
        'curl', 'wget', 'nc', 'netcat', 'telnet', 'ssh', 'scp', 'rsync',
        'dd', 'format', 'fdisk', 'mkfs', 'mount', 'umount', 'systemctl',
        'service', 'crontab', 'at', 'batch', 'nohup', 'screen', 'tmux'
    }
    
    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.RESTRICTED,
        resource_limits: Optional[ResourceLimits] = None,
        custom_policies: Optional[Dict[str, CommandPolicy]] = None,
        docker_image: str = "python:3.9-alpine",
        work_directory: Optional[Path] = None
    ):
        """
        Initialize the secure subprocess manager.
        
        Args:
            execution_mode: Security mode for subprocess execution
            resource_limits: Resource constraints for processes
            custom_policies: Additional command policies
            docker_image: Docker image for container execution
            work_directory: Working directory for subprocess execution
        """
        self.execution_mode = execution_mode
        self.resource_limits = resource_limits or ResourceLimits()
        self.work_directory = work_directory or Path.cwd()
        self.docker_image = docker_image
        
        # Initialize command policies
        self.command_policies = self.DEFAULT_POLICIES.copy()
        if custom_policies:
            self.command_policies.update(custom_policies)
        
        # Initialize Docker client if available and needed
        self.docker_client = None
        if self.execution_mode == ExecutionMode.DOCKER and DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized successfully")
            except Exception as e:
                logger.warning(f"Docker initialization failed: {e}")
                self.execution_mode = ExecutionMode.SANDBOXED
        
        # Execution history for monitoring
        self.execution_history: List[ExecutionResult] = []
        self._execution_lock = threading.Lock()
        
        # Setup audit logging
        self._setup_audit_logging()
    
    def execute_command(
        self,
        command: Union[str, List[str]],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute a command with comprehensive security controls.
        
        Args:
            command: Command to execute (string or list)
            cwd: Working directory for execution
            env: Additional environment variables
            input_data: Input data to send to process
            timeout: Execution timeout in seconds
            
        Returns:
            ExecutionResult with execution details and security info
        """
        start_time = time.time()
        
        # Parse and validate command
        if isinstance(command, str):
            command_parts = shlex.split(command)
        else:
            command_parts = command
        
        if not command_parts:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="Empty command provided",
                execution_time=0.0,
                security_violations=["Empty command"]
            )
        
        base_command = os.path.basename(command_parts[0])
        command_hash = hashlib.sha256(' '.join(command_parts).encode()).hexdigest()[:16]
        
        # Security validation
        validation_result = self._validate_command(command_parts, cwd, env)
        if not validation_result[0]:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Security validation failed: {validation_result[1]}",
                execution_time=time.time() - start_time,
                security_violations=[validation_result[1]],
                command_hash=command_hash
            )
        
        # Log command execution attempt
        self._audit_log("COMMAND_EXECUTION_ATTEMPT", {
            "command": command_parts,
            "cwd": str(cwd) if cwd else None,
            "mode": self.execution_mode.value,
            "command_hash": command_hash
        })
        
        # Execute based on mode
        try:
            if self.execution_mode == ExecutionMode.DOCKER and self.docker_client:
                result = self._execute_in_docker(command_parts, cwd, env, input_data, timeout)
            elif self.execution_mode == ExecutionMode.SANDBOXED:
                result = self._execute_sandboxed(command_parts, cwd, env, input_data, timeout)
            else:  # RESTRICTED or MINIMAL
                result = self._execute_restricted(command_parts, cwd, env, input_data, timeout)
            
            result.command_hash = command_hash
            
            # Add to execution history
            with self._execution_lock:
                self.execution_history.append(result)
                # Keep only last 1000 executions
                if len(self.execution_history) > 1000:
                    self.execution_history.pop(0)
            
            # Log execution result
            self._audit_log("COMMAND_EXECUTION_RESULT", {
                "command_hash": command_hash,
                "success": result.success,
                "return_code": result.return_code,
                "execution_time": result.execution_time,
                "security_violations": result.security_violations
            })
            
            return result
            
        except Exception as e:
            error_result = ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                security_violations=[f"Execution exception: {str(e)}"],
                command_hash=command_hash
            )
            
            self._audit_log("COMMAND_EXECUTION_ERROR", {
                "command_hash": command_hash,
                "error": str(e),
                "command": command_parts
            })
            
            return error_result
    
    def _validate_command(
        self,
        command_parts: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]]
    ) -> Tuple[bool, str]:
        """
        Comprehensive command validation.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command_parts:
            return False, "Empty command"
        
        base_command = os.path.basename(command_parts[0])
        
        # Check forbidden commands
        if base_command in self.FORBIDDEN_COMMANDS:
            return False, f"Forbidden command: {base_command}"
        
        # Check if command is in policy
        if base_command not in self.command_policies:
            return False, f"Command not in whitelist: {base_command}"
        
        policy = self.command_policies[base_command]
        
        # Validate arguments
        for arg in command_parts[1:]:
            # Check forbidden arguments
            if arg in policy.forbidden_args:
                return False, f"Forbidden argument: {arg}"
            
            # Check allowed arguments (if specified)
            if policy.allowed_args and arg.startswith('-'):
                if arg not in policy.allowed_args:
                    return False, f"Argument not allowed: {arg}"
        
        # Check for command injection patterns
        command_str = ' '.join(command_parts)
        injection_patterns = [
            r'[;&|`$()]', r'>\s*/dev/', r'<\s*/', r'\|\s*(sh|bash|cmd)',
            r'&&|\|\|', r'2>&1', r'>/dev/null'
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, command_str):
                return False, f"Command injection pattern detected: {pattern}"
        
        # Validate working directory
        if cwd:
            cwd_path = Path(cwd).resolve()
            if not cwd_path.exists():
                return False, f"Working directory does not exist: {cwd}"
            
            # Check if working directory is allowed
            if policy.allowed_paths:
                if not any(str(cwd_path).startswith(str(Path(allowed).resolve())) 
                          for allowed in policy.allowed_paths):
                    return False, f"Working directory not allowed: {cwd}"
        
        # Validate environment variables
        if env:
            dangerous_env_vars = {'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'HOME'}
            for var in env:
                if var in dangerous_env_vars:
                    return False, f"Dangerous environment variable: {var}"
        
        return True, "Valid"
    
    def _execute_restricted(
        self,
        command_parts: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        input_data: Optional[str],
        timeout: Optional[int]
    ) -> ExecutionResult:
        """Execute command with restricted privileges."""
        base_command = os.path.basename(command_parts[0])
        policy = self.command_policies[base_command]
        
        # Setup secure environment
        secure_env = self._create_secure_environment(policy, env)
        
        # Use policy timeout if not specified
        if timeout is None:
            timeout = policy.max_execution_time
        
        # Setup working directory
        work_dir = cwd or self.work_directory
        
        start_time = time.time()
        
        try:
            # Create process with security restrictions
            process = subprocess.Popen(
                command_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if input_data else None,
                cwd=str(work_dir),
                env=secure_env,
                preexec_fn=self._set_process_limits,
                text=True
            )
            
            # Communicate with timeout
            try:
                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return ExecutionResult(
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds",
                    execution_time=time.time() - start_time,
                    security_violations=["Timeout exceeded"]
                )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Process execution failed: {str(e)}",
                execution_time=time.time() - start_time,
                security_violations=[f"Process exception: {str(e)}"]
            )
    
    def _execute_sandboxed(
        self,
        command_parts: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        input_data: Optional[str],
        timeout: Optional[int]
    ) -> ExecutionResult:
        """Execute command in sandboxed environment."""
        # For now, use restricted execution with enhanced isolation
        # In production, this could use additional sandboxing technologies
        return self._execute_restricted(command_parts, cwd, env, input_data, timeout)
    
    def _execute_in_docker(
        self,
        command_parts: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        input_data: Optional[str],
        timeout: Optional[int]
    ) -> ExecutionResult:
        """Execute command in Docker container."""
        if not self.docker_client:
            return self._execute_sandboxed(command_parts, cwd, env, input_data, timeout)
        
        base_command = os.path.basename(command_parts[0])
        policy = self.command_policies[base_command]
        
        if timeout is None:
            timeout = policy.max_execution_time
        
        start_time = time.time()
        
        try:
            # Create temporary directory for file exchange
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy working directory if specified
                container_work_dir = "/workspace"
                if cwd and cwd.exists():
                    import shutil
                    shutil.copytree(cwd, temp_path / "workspace", dirs_exist_ok=True)
                else:
                    (temp_path / "workspace").mkdir(exist_ok=True)
                
                # Setup environment
                container_env = self._create_secure_environment(policy, env)
                
                # Run container
                container = self.docker_client.containers.run(
                    self.docker_image,
                    command=command_parts,
                    volumes={str(temp_path): {'bind': '/tmp/workspace', 'mode': 'rw'}},
                    working_dir=container_work_dir,
                    environment=container_env,
                    mem_limit=f"{self.resource_limits.memory_mb}m",
                    cpu_period=100000,
                    cpu_quota=50000,  # 50% CPU
                    network_disabled=True,
                    remove=True,
                    timeout=timeout,
                    stdout=True,
                    stderr=True,
                    stdin_open=bool(input_data),
                    detach=False
                )
                
                # Get output
                output = container.decode('utf-8') if container else ""
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=True,
                    return_code=0,
                    stdout=output,
                    stderr="",
                    execution_time=execution_time
                )
                
        except docker.errors.ContainerError as e:
            return ExecutionResult(
                success=False,
                return_code=e.exit_status,
                stdout="",
                stderr=e.stderr.decode('utf-8') if e.stderr else str(e),
                execution_time=time.time() - start_time,
                security_violations=[f"Container error: {str(e)}"]
            )
        except docker.errors.ImageNotFound:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Docker image not found: {self.docker_image}",
                execution_time=time.time() - start_time,
                security_violations=["Docker image not available"]
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Docker execution failed: {str(e)}",
                execution_time=time.time() - start_time,
                security_violations=[f"Docker exception: {str(e)}"]
            )
    
    def _create_secure_environment(
        self,
        policy: CommandPolicy,
        additional_env: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """Create a secure environment for subprocess execution."""
        # Start with minimal environment
        secure_env = {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'HOME': str(self.work_directory),
            'USER': 'nobody',
            'SHELL': '/bin/sh',
            'LANG': 'C',
            'LC_ALL': 'C',
            'TMPDIR': '/tmp'
        }
        
        # Add policy-specific environment variables
        secure_env.update(policy.environment_vars)
        
        # Add safe additional environment variables
        if additional_env:
            safe_vars = {
                'PYTHONPATH', 'NODE_PATH', 'NPM_CONFIG_PREFIX',
                'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME',
                'GIT_COMMITTER_EMAIL'
            }
            
            for key, value in additional_env.items():
                if key in safe_vars:
                    secure_env[key] = value
        
        return secure_env
    
    def _set_process_limits(self):
        """Set resource limits for the subprocess."""
        try:
            # Set memory limit
            memory_bytes = self.resource_limits.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # Set CPU time limit
            cpu_limit = self.resource_limits.cpu_time_seconds
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
            # Set file size limit
            file_size_bytes = self.resource_limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
            
            # Set number of processes limit
            resource.setrlimit(resource.RLIMIT_NPROC, 
                              (self.resource_limits.max_processes, 
                               self.resource_limits.max_processes))
            
            # Set number of open files limit
            resource.setrlimit(resource.RLIMIT_NOFILE, 
                              (self.resource_limits.max_open_files, 
                               self.resource_limits.max_open_files))
            
            # Prevent core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            
            # Set nice value (lower priority)
            os.nice(10)
            
        except (resource.error, OSError) as e:
            logger.warning(f"Failed to set some resource limits: {e}")
    
    def _setup_audit_logging(self):
        """Setup audit logging for subprocess execution."""
        # Create audit log directory
        log_dir = Path.home() / ".claude-tiu" / "security"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup audit logger
        self.audit_logger = logging.getLogger("claude_tiu_subprocess_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        audit_handler = logging.FileHandler(log_dir / "subprocess_audit.log")
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
    
    def _audit_log(self, event_type: str, details: Dict[str, Any]):
        """Log security audit event."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "execution_mode": self.execution_mode.value
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics and security metrics."""
        with self._execution_lock:
            if not self.execution_history:
                return {"total_executions": 0}
            
            total_executions = len(self.execution_history)
            successful_executions = sum(1 for r in self.execution_history if r.success)
            failed_executions = total_executions - successful_executions
            
            security_violations = []
            for result in self.execution_history:
                security_violations.extend(result.security_violations)
            
            total_execution_time = sum(r.execution_time for r in self.execution_history)
            avg_execution_time = total_execution_time / total_executions
            
            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": (successful_executions / total_executions) * 100,
                "total_security_violations": len(security_violations),
                "unique_violations": len(set(security_violations)),
                "average_execution_time": avg_execution_time,
                "total_execution_time": total_execution_time,
                "execution_mode": self.execution_mode.value
            }
    
    def add_command_policy(self, command: str, policy: CommandPolicy):
        """Add or update a command policy."""
        self.command_policies[command] = policy
        self._audit_log("POLICY_ADDED", {
            "command": command,
            "policy": {
                "allowed_args": list(policy.allowed_args),
                "forbidden_args": list(policy.forbidden_args),
                "max_execution_time": policy.max_execution_time
            }
        })
    
    def remove_command_policy(self, command: str) -> bool:
        """Remove a command policy."""
        if command in self.command_policies:
            del self.command_policies[command]
            self._audit_log("POLICY_REMOVED", {"command": command})
            return True
        return False
    
    def list_allowed_commands(self) -> List[str]:
        """Get list of allowed commands."""
        return list(self.command_policies.keys())
    
    def is_command_allowed(self, command: str) -> bool:
        """Check if a command is allowed."""
        return command in self.command_policies

# Utility functions
def create_secure_subprocess_manager(
    execution_mode: ExecutionMode = ExecutionMode.RESTRICTED,
    memory_limit_mb: int = 256,
    cpu_time_limit: int = 30
) -> SecureSubprocessManager:
    """
    Create a configured secure subprocess manager.
    
    Args:
        execution_mode: Security execution mode
        memory_limit_mb: Memory limit in MB
        cpu_time_limit: CPU time limit in seconds
        
    Returns:
        Configured SecureSubprocessManager instance
    """
    resource_limits = ResourceLimits(
        memory_mb=memory_limit_mb,
        cpu_time_seconds=cpu_time_limit
    )
    
    return SecureSubprocessManager(
        execution_mode=execution_mode,
        resource_limits=resource_limits
    )

def execute_safe_command(
    command: Union[str, List[str]],
    cwd: Optional[Path] = None,
    timeout: int = 30
) -> ExecutionResult:
    """
    Quick function to execute a command safely.
    
    Args:
        command: Command to execute
        cwd: Working directory
        timeout: Execution timeout
        
    Returns:
        ExecutionResult
    """
    manager = create_secure_subprocess_manager()
    return manager.execute_command(command, cwd=cwd, timeout=timeout)