"""
Subprocess Manager - Safe subprocess execution utilities.

Provides secure subprocess management with timeouts, resource limits,
and output handling for AI integration.
"""

import asyncio
import logging
import subprocess
import signal
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SubprocessResult:
    """Result of subprocess execution."""
    returncode: int
    stdout: str
    stderr: str
    execution_time: float
    timed_out: bool = False
    killed: bool = False


class SubprocessManager:
    """Safe subprocess execution manager."""
    
    def __init__(self):
        """Initialize subprocess manager."""
        self._active_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._default_timeout = 30
        self._max_concurrent = 5
        
    async def execute_command(
        self,
        cmd_args: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        input_data: Optional[str] = None
    ) -> SubprocessResult:
        """
        Execute command safely with timeout and resource limits.
        
        Args:
            cmd_args: Command arguments
            cwd: Working directory
            env: Environment variables
            timeout: Execution timeout in seconds
            capture_output: Whether to capture stdout/stderr
            input_data: Input data to send to process
            
        Returns:
            SubprocessResult: Execution result
        """
        if not cmd_args:
            raise ValueError("Command arguments cannot be empty")
        
        timeout = timeout or self._default_timeout
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate command for security
            if not await self._is_safe_command(cmd_args):
                raise SecurityError(f"Potentially unsafe command: {cmd_args[0]}")
            
            # Set up process arguments
            kwargs = {
                'cwd': str(cwd) if cwd else None,
                'env': env,
            }
            
            if capture_output:
                kwargs.update({
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.PIPE
                })
            
            if input_data:
                kwargs['stdin'] = subprocess.PIPE
            
            # Create and execute process
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                **kwargs
            )
            
            process_id = str(id(process))
            self._active_processes[process_id] = process
            
            try:
                # Wait for completion with timeout
                if input_data:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input_data.encode()),
                        timeout=timeout
                    )
                else:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                return SubprocessResult(
                    returncode=process.returncode,
                    stdout=stdout.decode() if stdout else "",
                    stderr=stderr.decode() if stderr else "",
                    execution_time=execution_time,
                    timed_out=False,
                    killed=False
                )
                
            except asyncio.TimeoutError:
                # Handle timeout
                logger.warning(f"Process timed out after {timeout}s: {cmd_args[0]}")
                
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                return SubprocessResult(
                    returncode=-1,
                    stdout="",
                    stderr=f"Process timed out after {timeout} seconds",
                    execution_time=execution_time,
                    timed_out=True,
                    killed=True
                )
            
            finally:
                # Clean up process tracking
                self._active_processes.pop(process_id, None)
                
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Subprocess execution failed: {e}")
            
            return SubprocessResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                timed_out=False,
                killed=False
            )
    
    async def execute_shell_command(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None
    ) -> SubprocessResult:
        """
        Execute shell command (use with caution).
        
        Args:
            command: Shell command to execute
            cwd: Working directory
            timeout: Execution timeout
            
        Returns:
            SubprocessResult: Execution result
        """
        logger.warning(f"Executing shell command: {command}")
        
        # Basic command validation
        dangerous_commands = ['rm -rf', 'format', 'del', 'fdisk', 'dd']
        for dangerous in dangerous_commands:
            if dangerous in command.lower():
                raise SecurityError(f"Dangerous command detected: {dangerous}")
        
        return await self.execute_command(
            ['sh', '-c', command],
            cwd=cwd,
            timeout=timeout
        )
    
    async def kill_all_processes(self) -> None:
        """Kill all active processes."""
        for process_id, process in list(self._active_processes.items()):
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            except Exception as e:
                logger.warning(f"Failed to kill process {process_id}: {e}")
        
        self._active_processes.clear()
    
    async def cleanup(self) -> None:
        """Clean up subprocess manager."""
        await self.kill_all_processes()
        logger.info("Subprocess manager cleanup completed")
    
    async def _is_safe_command(self, cmd_args: List[str]) -> bool:
        """
        Check if command is safe to execute.
        
        Args:
            cmd_args: Command arguments
            
        Returns:
            True if command appears safe
        """
        if not cmd_args:
            return False
        
        command = cmd_args[0].lower()
        
        # Whitelist of safe commands
        safe_commands = {
            'python', 'python3', 'node', 'npm', 'pip', 'git',
            'ls', 'cat', 'echo', 'mkdir', 'cp', 'mv', 'touch',
            'grep', 'find', 'head', 'tail', 'wc', 'sort', 'uniq'
        }
        
        # Extract base command name
        base_command = Path(command).name
        
        if base_command not in safe_commands:
            logger.warning(f"Potentially unsafe command: {command}")
            return False
        
        # Check for dangerous flags
        dangerous_flags = ['-rf', '--force', '--delete', '/dev/']
        for arg in cmd_args:
            for flag in dangerous_flags:
                if flag in arg.lower():
                    logger.warning(f"Dangerous flag detected: {flag}")
                    return False
        
        return True


class SecurityError(Exception):
    """Security-related error in subprocess execution."""
    pass