"""
Test suite for ClaudeCodeDirectClient

Comprehensive tests for the direct CLI integration with Claude Code,
including OAuth authentication, subprocess execution, and output parsing.
"""

import asyncio
import json
import os
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import subprocess
import pytest

# Import the client we're testing
import sys
sys.path.append('/home/tekkadmin/claude-tui/src')

from claude_tui.integrations.claude_code_direct_client import (
    ClaudeCodeDirectClient,
    CliCommandBuilder,
    ClaudeCodeCliError,
    ClaudeCodeAuthError,
    ClaudeCodeTimeoutError,
    TaskExecutionRequest,
    ValidationRequest,
    RefactorRequest
)


class TestCliCommandBuilder:
    """Test the CLI command builder functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.builder = CliCommandBuilder("claude")
    
    def test_build_task_command_basic(self):
        """Test basic task command building."""
        cmd = self.builder.build_task_command("Write a hello world function")
        
        assert cmd[0] == "claude"
        assert "Write a hello world function" in cmd
        assert len(cmd) >= 2
    
    def test_build_task_command_with_context(self):
        """Test task command with context."""
        context = {"language": "python", "style": "functional"}
        cmd = self.builder.build_task_command(
            "Write a function", 
            context=context,
            model="claude-3-opus"
        )
        
        assert "--context" in cmd
        assert "--model" in cmd
        assert "claude-3-opus" in cmd
        
        # Find context JSON in command
        context_idx = cmd.index("--context") + 1
        context_json = json.loads(cmd[context_idx])
        assert context_json["language"] == "python"
        assert context_json["style"] == "functional"
    
    def test_build_validation_command(self):
        """Test validation command building."""
        code = "def hello(): print('world')"
        rules = ["check syntax", "verify style"]
        
        cmd, temp_file = self.builder.build_validation_command(
            code_content=code,
            validation_rules=rules,
            working_dir="/tmp"
        )
        
        assert cmd[0] == "claude"
        assert "--cwd" in cmd
        assert "/tmp" in cmd
        assert temp_file.endswith('.py')
        assert os.path.exists(temp_file)
        
        # Verify file content
        with open(temp_file, 'r') as f:
            assert f.read() == code
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_build_refactor_command(self):
        """Test refactor command building."""
        code = "def old_function(): pass"
        instructions = "Rename to new_function and add docstring"
        
        cmd, temp_file = self.builder.build_refactor_command(
            code_content=code,
            instructions=instructions,
            preserve_comments=True
        )
        
        assert cmd[0] == "claude"
        assert instructions in ' '.join(cmd)
        assert "preserve existing comments" in ' '.join(cmd)
        assert os.path.exists(temp_file)
        
        # Verify file content
        with open(temp_file, 'r') as f:
            assert f.read() == code
        
        # Cleanup
        os.unlink(temp_file)


class TestClaudeCodeDirectClient:
    """Test the main Claude Code Direct Client functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create a temporary .cc file for testing
        self.temp_cc_fd, self.temp_cc_file = tempfile.mkstemp(suffix='.cc')
        with os.fdopen(self.temp_cc_fd, 'w') as f:
            f.write('sk-ant-oat01-test-token-for-testing')
        
        # Mock the CLI validation
        with patch.object(ClaudeCodeDirectClient, '_validate_claude_code_cli'):
            self.client = ClaudeCodeDirectClient(
                oauth_token_file=self.temp_cc_file,
                claude_code_path="claude"
            )
    
    def teardown_method(self):
        """Cleanup after each test method."""
        if os.path.exists(self.temp_cc_file):
            os.unlink(self.temp_cc_file)
        
        if hasattr(self, 'client'):
            self.client.cleanup_session()
    
    def test_oauth_token_loading(self):
        """Test OAuth token loading from .cc file."""
        assert self.client.oauth_token == 'sk-ant-oat01-test-token-for-testing'
        assert self.client.oauth_token_file == self.temp_cc_file
    
    def test_session_info(self):
        """Test session information retrieval."""
        session_info = self.client.get_session_info()
        
        assert 'session_id' in session_info
        assert 'session_start_time' in session_info
        assert 'execution_count' in session_info
        assert session_info['oauth_token_available'] is True
        assert session_info['oauth_token_prefix'].startswith('sk-ant-oat01-test-to')
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        # Mock subprocess execution
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'Claude Code v1.0.0\n', b'')
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            health_result = await self.client.health_check()
        
        assert health_result['healthy'] is True
        assert 'cli_version' in health_result
        assert health_result['oauth_token_available'] is True
        assert 'session_id' in health_result
    
    @pytest.mark.asyncio
    async def test_execute_task_via_cli_success(self):
        """Test successful task execution via CLI."""
        # Mock subprocess execution
        mock_result = subprocess.CompletedProcess(
            args=['claude', 'Write hello world'],
            returncode=0,
            stdout='```python\ndef hello_world():\n    print("Hello, World!")\n```\n',
            stderr=''
        )
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_result):
            result = await self.client.execute_task_via_cli(
                "Write a hello world function in Python"
            )
        
        assert result['success'] is True
        assert 'execution_id' in result
        assert 'execution_time' in result
        assert 'generated_code' in result
        assert 'def hello_world' in result['generated_code']
    
    @pytest.mark.asyncio
    async def test_execute_task_via_cli_failure(self):
        """Test failed task execution via CLI."""
        # Mock subprocess execution with error
        mock_result = subprocess.CompletedProcess(
            args=['claude', 'Invalid task'],
            returncode=1,
            stdout='',
            stderr='Error: Invalid task description'
        )
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_result):
            result = await self.client.execute_task_via_cli(
                "Invalid task description"
            )
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Invalid task description' in result['error']
    
    @pytest.mark.asyncio
    async def test_validate_code_via_cli(self):
        """Test code validation via CLI."""
        # Mock subprocess execution
        mock_result = subprocess.CompletedProcess(
            args=['claude', 'validate'],
            returncode=0,
            stdout='Code validation completed\nSuggestion: Add type hints\nIssue: Missing docstring',
            stderr=''
        )
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_result):
            result = await self.client.validate_code_via_cli(
                "def hello(): print('world')",
                validation_rules=["check style", "verify syntax"]
            )
        
        assert result['valid'] is True
        assert len(result['suggestions']) > 0
        assert len(result['issues']) > 0
        assert any('Add type hints' in suggestion for suggestion in result['suggestions'])
        assert any('Missing docstring' in issue for issue in result['issues'])
    
    @pytest.mark.asyncio
    async def test_refactor_code_via_cli(self):
        """Test code refactoring via CLI."""
        original_code = "def old_func(): pass"
        refactored_code = "def new_func():\n    \"\"\"New function with docstring.\"\"\"\n    pass"
        
        # Mock subprocess execution
        mock_result = subprocess.CompletedProcess(
            args=['claude', 'refactor'],
            returncode=0,
            stdout=f'```python\n{refactored_code}\n```\n',
            stderr=''
        )
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_result):
            result = await self.client.refactor_code_via_cli(
                original_code,
                "Rename function and add docstring"
            )
        
        assert result['success'] is True
        assert result['refactored_code'] == refactored_code
        assert result['original_code'] == original_code
        assert len(result['changes_made']) > 0
    
    def test_code_block_extraction(self):
        """Test extraction of code blocks from output."""
        output = '''
        Here's the solution:
        
        ```python
        def hello():
            print("Hello, World!")
        ```
        
        And here's another example:
        
        ```javascript
        console.log("Hello, World!");
        ```
        '''
        
        code_blocks = self.client._extract_code_blocks(output)
        assert len(code_blocks) == 2
        assert 'def hello()' in code_blocks[0]
        assert 'console.log' in code_blocks[1]
    
    def test_json_extraction_from_output(self):
        """Test extraction of JSON from CLI output."""
        output = '''
        Analysis complete.
        {"status": "success", "issues": 3, "suggestions": 5}
        Process finished.
        '''
        
        json_data = self.client._extract_json_from_output(output)
        assert json_data is not None
        assert json_data['status'] == 'success'
        assert json_data['issues'] == 3
        assert json_data['suggestions'] == 5
    
    def test_change_identification(self):
        """Test identification of changes between code versions."""
        original = "def hello():\n    print('world')"
        modified = "def hello_world():\n    \"\"\"Say hello.\"\"\"\n    print('Hello, World!')"
        
        changes = self.client._identify_code_changes(original, modified)
        assert len(changes) > 0
        assert any('Line count changed' in change for change in changes)
    
    @pytest.mark.asyncio
    async def test_cli_command_execution_timeout(self):
        """Test CLI command timeout handling."""
        # Mock a long-running process
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_process.kill = AsyncMock()
        mock_process.wait = AsyncMock()
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with pytest.raises(ClaudeCodeTimeoutError):
                await self.client._execute_cli_command(
                    ['claude', 'long task'],
                    timeout=1,
                    execution_id='test-timeout'
                )
        
        # Verify process was killed
        mock_process.kill.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test authentication error handling."""
        # Create client without OAuth token
        with patch.object(ClaudeCodeDirectClient, '_validate_claude_code_cli'):
            client_no_auth = ClaudeCodeDirectClient(
                oauth_token_file=None,
                claude_code_path="claude"
            )
        
        # Health check should still work but indicate no auth
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b'Claude Code v1.0.0\n', b'')
            mock_exec.return_value = mock_process
            
            health_result = await client_no_auth.health_check()
        
        assert health_result['oauth_token_available'] is False
        
        client_no_auth.cleanup_session()


class TestRequestModels:
    """Test the Pydantic request models."""
    
    def test_task_execution_request(self):
        """Test TaskExecutionRequest model."""
        request = TaskExecutionRequest(
            description="Write a function",
            context={"language": "python"},
            timeout=300
        )
        
        assert request.description == "Write a function"
        assert request.context["language"] == "python"
        assert request.timeout == 300
        assert request.model == "claude-3-sonnet"  # default
    
    def test_validation_request(self):
        """Test ValidationRequest model."""
        request = ValidationRequest(
            code="def hello(): pass",
            validation_rules=["syntax", "style"],
            expected_format="python"
        )
        
        assert request.code == "def hello(): pass"
        assert "syntax" in request.validation_rules
        assert request.expected_format == "python"
    
    def test_refactor_request(self):
        """Test RefactorRequest model."""
        request = RefactorRequest(
            code="old code",
            instructions="make it better",
            preserve_comments=False
        )
        
        assert request.code == "old code"
        assert request.instructions == "make it better"
        assert request.preserve_comments is False


class TestFactoryMethods:
    """Test factory methods for client creation."""
    
    def test_create_with_token_file(self):
        """Test factory method with token file."""
        # Create temporary token file
        temp_fd, temp_file = tempfile.mkstemp(suffix='.cc')
        try:
            with os.fdopen(temp_fd, 'w') as f:
                f.write('sk-ant-oat01-factory-test')
            
            with patch.object(ClaudeCodeDirectClient, '_validate_claude_code_cli'):
                client = ClaudeCodeDirectClient.create_with_token_file(
                    token_file_path=temp_file,
                    claude_code_path="test-claude",
                    working_directory="/test/dir"
                )
            
            assert client.oauth_token == 'sk-ant-oat01-factory-test'
            assert client.claude_code_path == "test-claude"
            assert client.working_directory == "/test/dir"
            
            client.cleanup_session()
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


# Integration test class
class TestIntegration:
    """Integration tests (require actual Claude Code CLI)."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not shutil.which("claude"), reason="Claude Code CLI not available")
    async def test_real_cli_integration(self):
        """Test with real Claude Code CLI (if available)."""
        # Only run if Claude Code CLI is actually installed
        client = ClaudeCodeDirectClient(claude_code_path="claude")
        
        try:
            health_result = await client.health_check()
            assert 'healthy' in health_result
            # Note: May fail if no OAuth token is configured
        
        finally:
            client.cleanup_session()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, "-v"])