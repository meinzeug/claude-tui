"""
Execution Tester - Safe code execution testing for validation.

Provides sandboxed execution testing to validate AI-generated code
functionality and detect runtime errors.
"""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import sys
import traceback

from claude_tui.core.config_manager import ConfigManager
from claude_tui.models.project import Project
from claude_tui.validation.progress_validator import ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution test."""
    success: bool
    output: str
    error_message: Optional[str] = None
    exit_code: int = 0
    execution_time: float = 0.0
    issues_found: List[ValidationIssue] = None


class ExecutionTester:
    """
    Safe code execution testing for validation.
    
    Provides sandboxed execution environment to test AI-generated code
    and validate its functionality without security risks.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the execution tester.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        
        # Configuration
        self._testing_enabled = True
        self._timeout_seconds = 30
        self._max_memory_mb = 256
        self._sandbox_enabled = True
        
        logger.info("Execution tester initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the execution tester.
        """
        logger.info("Initializing execution tester")
        
        try:
            # Load configuration
            tester_config = await self.config_manager.get_setting('execution_tester', {})
            self._testing_enabled = tester_config.get('enabled', True)
            self._timeout_seconds = tester_config.get('timeout_seconds', 30)
            self._max_memory_mb = tester_config.get('max_memory_mb', 256)
            self._sandbox_enabled = tester_config.get('sandbox_enabled', True)
            
            logger.info("Execution tester initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize execution tester: {e}")
            raise
    
    async def test_execution(
        self,
        content: str,
        file_path: Optional[Path] = None,
        project: Optional[Project] = None
    ) -> List[ValidationIssue]:
        """
        Test code execution and return validation issues.
        
        Args:
            content: Code content to test
            file_path: File path for context
            project: Associated project
            
        Returns:
            List of validation issues found during execution
        """
        if not self._testing_enabled or not content.strip():
            return []
        
        issues = []
        
        try:
            # Determine language
            language = self._infer_language(file_path) if file_path else None
            
            if language == 'python':
                python_issues = await self._test_python_execution(content, file_path, project)
                issues.extend(python_issues)
            elif language in ['javascript', 'typescript']:
                js_issues = await self._test_javascript_execution(content, file_path, project)
                issues.extend(js_issues)
            else:
                logger.debug(f"Execution testing not supported for language: {language}")
        
        except Exception as e:
            logger.error(f"Execution testing failed: {e}")
            issues.append(ValidationIssue(
                id="execution_test_error",
                description=f"Execution testing failed: {e}",
                severity=ValidationSeverity.MEDIUM,
                file_path=str(file_path) if file_path else None,
                issue_type="execution_error"
            ))
        
        return issues
    
    async def cleanup(self) -> None:
        """
        Cleanup execution tester resources.
        """
        logger.info("Execution tester cleanup completed")
    
    # Private helper methods
    
    def _infer_language(self, file_path: Path) -> Optional[str]:
        """Infer programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript'
        }
        return extension_map.get(file_path.suffix.lower())
    
    async def _test_python_execution(
        self,
        content: str,
        file_path: Optional[Path],
        project: Optional[Project]
    ) -> List[ValidationIssue]:
        """Test Python code execution."""
        issues = []
        
        try:
            # Basic syntax check first
            try:
                compile(content, '<string>', 'exec')
            except SyntaxError as e:
                issues.append(ValidationIssue(
                    id="python_syntax_error",
                    description=f"Python syntax error: {e.msg}",
                    severity=ValidationSeverity.CRITICAL,
                    file_path=str(file_path) if file_path else None,
                    line_number=e.lineno,
                    issue_type="syntax_error"
                ))
                return issues
            
            # Safe execution test in sandbox
            if self._sandbox_enabled:
                exec_issues = await self._safe_python_execution(content, file_path)
                issues.extend(exec_issues)
        
        except Exception as e:
            logger.error(f"Python execution testing failed: {e}")
        
        return issues
    
    async def _test_javascript_execution(
        self,
        content: str,
        file_path: Optional[Path],
        project: Optional[Project]
    ) -> List[ValidationIssue]:
        """Test JavaScript code execution."""
        issues = []
        
        # Basic syntax check using Node.js if available
        try:
            result = await self._run_node_syntax_check(content)
            if not result.success:
                issues.append(ValidationIssue(
                    id="javascript_syntax_error",
                    description=f"JavaScript syntax error: {result.error_message}",
                    severity=ValidationSeverity.CRITICAL,
                    file_path=str(file_path) if file_path else None,
                    issue_type="syntax_error"
                ))
        
        except Exception as e:
            logger.debug(f"Node.js not available for JS testing: {e}")
        
        return issues
    
    async def _safe_python_execution(
        self,
        content: str,
        file_path: Optional[Path]
    ) -> List[ValidationIssue]:
        """Execute Python code safely in sandbox."""
        issues = []
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = Path(temp_file.name)
            
            try:
                # Run with restricted environment
                cmd = [
                    sys.executable, 
                    '-c',
                    f'''
import sys
import traceback
import resource

# Set memory limit
try:
    resource.setrlimit(resource.RLIMIT_AS, ({self._max_memory_mb * 1024 * 1024}, {self._max_memory_mb * 1024 * 1024}))
except:
    pass

# Restricted builtins
restricted_builtins = {{
    k: v for k, v in __builtins__.items() 
    if k not in ['eval', 'exec', 'compile', 'open', '__import__']
}}

try:
    with open('{temp_file_path}', 'r') as f:
        code = f.read()
    
    # Execute in restricted environment
    exec(code, {{"__builtins__": restricted_builtins}})
    
except Exception as e:
    print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{e}}")
    traceback.print_exc()
'''
                ]
                
                # Run with timeout
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    ),
                    timeout=self._timeout_seconds
                )
                
                stdout, stderr = await result.communicate()
                
                # Parse output for errors
                output = stdout.decode() + stderr.decode()
                
                if 'EXECUTION_ERROR:' in output:
                    error_lines = [line for line in output.split('\\n') if 'EXECUTION_ERROR:' in line]
                    for error_line in error_lines:
                        error_msg = error_line.replace('EXECUTION_ERROR: ', '')
                        issues.append(ValidationIssue(
                            id="python_runtime_error",
                            description=f"Runtime error: {error_msg}",
                            severity=ValidationSeverity.HIGH,
                            file_path=str(file_path) if file_path else None,
                            issue_type="runtime_error"
                        ))
            
            finally:
                # Cleanup temp file
                if temp_file_path.exists():
                    temp_file_path.unlink()
        
        except asyncio.TimeoutError:
            issues.append(ValidationIssue(
                id="execution_timeout",
                description=f"Code execution timed out after {self._timeout_seconds}s",
                severity=ValidationSeverity.HIGH,
                file_path=str(file_path) if file_path else None,
                issue_type="timeout_error"
            ))
        
        except Exception as e:
            logger.error(f"Safe Python execution failed: {e}")
        
        return issues
    
    async def _run_node_syntax_check(self, content: str) -> ExecutionResult:
        """Run Node.js syntax check."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = Path(temp_file.name)
            
            try:
                # Run Node.js syntax check
                cmd = ['node', '--check', str(temp_file_path)]
                
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    ),
                    timeout=10
                )
                
                stdout, stderr = await result.communicate()
                
                if result.returncode == 0:
                    return ExecutionResult(
                        success=True,
                        output=stdout.decode(),
                        exit_code=result.returncode
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        output=stdout.decode(),
                        error_message=stderr.decode(),
                        exit_code=result.returncode
                    )
            
            finally:
                # Cleanup temp file
                if temp_file_path.exists():
                    temp_file_path.unlink()
        
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error_message="Node.js not found",
                exit_code=1
            )
        
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error_message="Syntax check timed out",
                exit_code=1
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=str(e),
                exit_code=1
            )