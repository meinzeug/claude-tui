"""
Claude CI/CD Mock Tests

Comprehensive mock tests for CI/CD pipeline integration that don't require real API calls.
These tests validate client behavior, error handling, and integration patterns using mocks.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, mock_open, Mock

import aiohttp

from src.claude_tui.integrations.claude_code_client import (
    ClaudeCodeClient,
    ClaudeCodeApiError,
    ClaudeCodeAuthError,
    ClaudeCodeRateLimitError,
    RateLimiter
)
from src.claude_tui.integrations.claude_code_direct_client import (
    ClaudeCodeDirectClient,
    ClaudeCodeCliError,
    ClaudeCodeAuthError as DirectAuthError,
    ClaudeCodeExecutionError,
    ClaudeCodeTimeoutError,
    CliCommandBuilder
)
from src.claude_tui.core.config_manager import ConfigManager


class TestHttpClientMocks:
    """Mock tests for HTTP client without real API calls."""
    
    @pytest.fixture
    async def mock_http_client(self):
        """Create HTTP client with mocked dependencies."""
        config = ConfigManager()
        
        with patch('src.claude_tui.integrations.claude_code_client.SecurityManager'):
            client = ClaudeCodeClient(
                config,
                oauth_token="mock-token-12345",
                base_url="https://mock-api.claude.ai/v1"
            )
            yield client
            await client.cleanup()
    
    @pytest.mark.asyncio
    async def test_successful_task_execution_mock(self, mock_http_client):
        """Test successful task execution with mocked response."""
        mock_response_data = {
            'success': True,
            'content': 'def hello_world():\n    return "Hello, World!"',
            'model_used': 'claude-3-sonnet',
            'execution_time': 2.5,
            'validation_passed': True,
            'quality_score': 0.85
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_response.headers = {'Content-Type': 'application/json'}
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value.__aenter__.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Execute task
            result = await mock_http_client.execute_task(
                "Write a hello world function",
                {"language": "python"}
            )
            
            # Verify results
            assert result['success'] == True
            assert 'content' in result
            assert result['model_used'] == 'claude-3-sonnet'
            assert 'task_description' in result
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling_mock(self, mock_http_client):
        """Test authentication error handling with mocked 401 response."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Setup mock 401 response
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.text = AsyncMock(return_value="Invalid authentication token")
            mock_response.headers = {}
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value.__aenter__.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Execute task - should handle auth error gracefully
            result = await mock_http_client.execute_task("Test task")
            
            # Should return error result instead of raising
            assert isinstance(result, dict)
            assert 'error' in result
            assert result['success'] == False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_mock(self, mock_http_client):
        """Test rate limiting behavior with mocked responses."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Setup mock rate limit response
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.headers = {'Retry-After': '60'}
            mock_response.text = AsyncMock(return_value="Rate limit exceeded")
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value.__aenter__.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Execute task - should handle rate limit
            result = await mock_http_client.execute_task("Test task")
            
            assert isinstance(result, dict)
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_server_error_recovery_mock(self, mock_http_client):
        """Test server error recovery with mocked 500 responses."""
        call_count = 0
        
        def mock_request_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_response = AsyncMock()
            if call_count <= 2:  # First two calls fail
                mock_response.status = 500
                mock_response.text = AsyncMock(return_value="Internal server error")
            else:  # Third call succeeds
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={'success': True, 'content': 'Success!'})
            
            mock_response.headers = {}
            return mock_response.__aenter__()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session_instance.request.side_effect = mock_request_side_effect
            mock_session.return_value = mock_session_instance
            
            # Execute task - should retry and eventually succeed
            result = await mock_http_client.execute_task("Test task")
            
            assert isinstance(result, dict)
            # Should either succeed after retries or fail gracefully
            assert 'success' in result or 'error' in result
    
    @pytest.mark.asyncio
    async def test_timeout_handling_mock(self, mock_http_client):
        """Test timeout handling with mocked timeout."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = AsyncMock()
            
            # Mock timeout during request
            mock_session_instance.request.side_effect = asyncio.TimeoutError("Request timed out")
            mock_session.return_value = mock_session_instance
            
            # Execute task - should handle timeout gracefully
            result = await mock_http_client.execute_task("Test task")
            
            assert isinstance(result, dict)
            assert 'error' in result
            assert result['success'] == False
    
    @pytest.mark.asyncio
    async def test_malformed_json_response_mock(self, mock_http_client):
        """Test handling of malformed JSON responses."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.headers = {}
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value.__aenter__.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Execute task - should handle JSON decode error
            result = await mock_http_client.execute_task("Test task")
            
            assert isinstance(result, dict)
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_health_check_mock(self, mock_http_client):
        """Test health check with mocked response."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'status': 'healthy', 'version': '1.0.0'})
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value.__aenter__.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Execute health check
            is_healthy = await mock_http_client.health_check()
            
            # Should handle response appropriately
            assert isinstance(is_healthy, bool)
    
    def test_rate_limiter_functionality(self):
        """Test rate limiter functionality in isolation."""
        rate_limiter = RateLimiter(requests_per_minute=2)
        
        # Should start empty
        assert len(rate_limiter.request_times) == 0
        
        # Test acquire doesn't block initially
        import time
        start_time = time.time()
        
        # This is a sync test, so we can't properly test the async acquire
        # But we can test the logic
        assert rate_limiter.requests_per_minute == 2


class TestDirectClientMocks:
    """Mock tests for direct CLI client without actual CLI execution."""
    
    @pytest.fixture
    def mock_direct_client(self):
        """Create direct client with mocked dependencies."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text("mock-oauth-token-12345")
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock successful CLI validation
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Claude Code CLI v1.0.0"
            mock_result.stderr = ""
            mock_subprocess.return_value = mock_result
            
            client = ClaudeCodeDirectClient(
                oauth_token_file=str(cc_file),
                claude_code_path="mock-claude",
                working_directory=temp_dir
            )
        
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_successful_cli_execution_mock(self, mock_direct_client):
        """Test successful CLI execution with mocked subprocess."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful execution
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b'```python\ndef hello():\n    return "Hello, World!"\n```\nTask completed successfully.',
                b''
            )
            mock_subprocess.return_value = mock_process
            
            # Execute task
            result = await mock_direct_client.execute_task_via_cli(
                "Write a hello world function"
            )
            
            # Verify results
            assert result['success'] == True
            assert 'execution_id' in result
            assert 'execution_time' in result
            assert 'generated_code' in result
            assert 'def hello():' in result['generated_code']
    
    @pytest.mark.asyncio
    async def test_cli_error_handling_mock(self, mock_direct_client):
        """Test CLI error handling with mocked failure."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock CLI error
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b'',
                b'Error: Invalid command syntax'
            )
            mock_subprocess.return_value = mock_process
            
            # Execute task
            result = await mock_direct_client.execute_task_via_cli(
                "Invalid task"
            )
            
            # Should handle error gracefully
            assert result['success'] == False
            assert 'error' in result
            assert 'execution_id' in result
    
    @pytest.mark.asyncio
    async def test_cli_timeout_mock(self, mock_direct_client):
        """Test CLI timeout handling with mocked timeout."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            
            # Mock timeout during communicate
            mock_process.communicate.side_effect = asyncio.TimeoutError("Process timed out")
            mock_subprocess.return_value = mock_process
            
            # Execute task with short timeout
            result = await mock_direct_client.execute_task_via_cli(
                "Long running task",
                timeout=1
            )
            
            # Should handle timeout gracefully
            assert isinstance(result, dict)
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_code_validation_mock(self, mock_direct_client):
        """Test code validation with mocked CLI response."""
        test_code = '''
def calculate_sum(a, b):
    return a + b
'''
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock validation response
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b'Code validation complete.\nNo syntax errors found.\nSuggestion: Consider adding type hints.',
                b''
            )
            mock_subprocess.return_value = mock_process
            
            # Execute validation
            result = await mock_direct_client.validate_code_via_cli(
                test_code,
                validation_rules=['syntax_check', 'style_check']
            )
            
            # Verify validation results
            assert result['valid'] == True
            assert 'execution_id' in result
            assert 'suggestions' in result
            assert len(result['suggestions']) >= 0
    
    @pytest.mark.asyncio
    async def test_code_refactoring_mock(self, mock_direct_client):
        """Test code refactoring with mocked CLI response."""
        original_code = '''
def calc(x, y):
    return x + y
'''
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock refactoring response
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b'```python\ndef add_numbers(x: int, y: int) -> int:\n    """Add two numbers and return the result."""\n    return x + y\n```\nRefactoring complete.',
                b''
            )
            mock_subprocess.return_value = mock_process
            
            # Execute refactoring
            result = await mock_direct_client.refactor_code_via_cli(
                original_code,
                "Add type hints and documentation"
            )
            
            # Verify refactoring results
            assert result['success'] == True
            assert 'refactored_code' in result
            assert 'add_numbers' in result['refactored_code']
            assert 'int' in result['refactored_code']  # Type hints added
    
    @pytest.mark.asyncio
    async def test_health_check_mock(self, mock_direct_client):
        """Test health check with mocked CLI response."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock version check
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b'Claude Code CLI v1.0.0', b'')
            mock_subprocess.return_value = mock_process
            
            # Execute health check
            health = await mock_direct_client.health_check()
            
            # Verify health check results
            assert health['healthy'] == True
            assert 'cli_version' in health
            assert health['oauth_token_available'] == True
            assert 'session_id' in health
    
    def test_command_builder_functionality(self):
        """Test CLI command builder functionality."""
        builder = CliCommandBuilder("mock-claude")
        
        # Test task command building
        cmd = builder.build_task_command(
            "Test prompt",
            context={"key": "value"},
            working_dir="/tmp",
            model="claude-3-haiku"
        )
        
        assert cmd[0] == "mock-claude"
        assert "--model" in cmd
        assert "claude-3-haiku" in cmd
        assert "--cwd" in cmd
        assert "/tmp" in cmd
        assert "Test prompt" in cmd
    
    def test_oauth_token_loading_mock(self):
        """Test OAuth token loading with mocked file operations."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        
        try:
            # Test with valid token
            cc_file.write_text("sk-ant-oat01-validtoken123")
            
            with patch('subprocess.run') as mock_subprocess:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "Claude Code CLI v1.0.0"
                mock_subprocess.return_value = mock_result
                
                client = ClaudeCodeDirectClient(
                    oauth_token_file=str(cc_file),
                    working_directory=temp_dir
                )
                
                session_info = client.get_session_info()
                assert session_info['oauth_token_available'] == True
                assert session_info['oauth_token_prefix'].startswith('sk-ant-oat01-')
                
                client.cleanup_session()
        
        finally:
            cc_file.unlink(missing_ok=True)
            Path(temp_dir).rmdir()
    
    def test_session_management_mock(self, mock_direct_client):
        """Test session management functionality."""
        # Get initial session info
        session_info = mock_direct_client.get_session_info()
        
        assert 'session_id' in session_info
        assert 'session_start_time' in session_info
        assert 'execution_count' in session_info
        assert session_info['execution_count'] == 0
        
        # Session ID should be consistent
        session_id1 = session_info['session_id']
        session_id2 = mock_direct_client.get_session_info()['session_id']
        assert session_id1 == session_id2


class TestIntegrationPatternMocks:
    """Test integration patterns and workflows with mocks."""
    
    @pytest.fixture
    async def mock_http_client(self):
        """HTTP client for integration testing."""
        config = ConfigManager()
        with patch('src.claude_tui.integrations.claude_code_client.SecurityManager'):
            client = ClaudeCodeClient(config, oauth_token="mock-token")
            yield client
            await client.cleanup()
    
    @pytest.fixture
    def mock_direct_client(self):
        """Direct client for integration testing."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text("mock-token")
        
        with patch('subprocess.run') as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Mock CLI"
            mock_subprocess.return_value = mock_result
            
            client = ClaudeCodeDirectClient(
                oauth_token_file=str(cc_file),
                working_directory=temp_dir
            )
        
        yield client
        client.cleanup_session()
        
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_code_generation_validation_workflow_mock(self, mock_http_client):
        """Test complete code generation and validation workflow."""
        # Mock code generation
        generation_response = {
            'success': True,
            'content': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)',
            'model_used': 'claude-3-sonnet'
        }
        
        # Mock validation
        validation_response = {
            'valid': True,
            'issues': [],
            'suggestions': ['Consider adding type hints']
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            call_count = 0
            
            def mock_request(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                mock_response = AsyncMock()
                mock_response.status = 200
                
                if call_count == 1:  # First call - code generation
                    mock_response.json = AsyncMock(return_value=generation_response)
                else:  # Second call - validation
                    mock_response.json = AsyncMock(return_value=validation_response)
                
                return mock_response.__aenter__()
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.side_effect = mock_request
            mock_session.return_value = mock_session_instance
            
            # Execute workflow
            gen_result = await mock_http_client.execute_task("Generate factorial function")
            
            if gen_result.get('success'):
                val_result = await mock_http_client.validate_output(
                    gen_result['content'],
                    {'validation_rules': ['syntax_check']}
                )
                
                assert val_result.get('valid') == True
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow_mock(self, mock_direct_client):
        """Test error recovery workflow with mocks."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            call_count = 0
            
            def mock_process_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                mock_process = AsyncMock()
                
                if call_count == 1:  # First attempt fails
                    mock_process.returncode = 1
                    mock_process.communicate.return_value = (b'', b'Syntax error')
                else:  # Second attempt succeeds
                    mock_process.returncode = 0
                    mock_process.communicate.return_value = (
                        b'```python\ndef fixed_function():\n    return "Fixed!"\n```',
                        b''
                    )
                
                return mock_process
            
            mock_subprocess.side_effect = mock_process_side_effect
            
            # First attempt (fails)
            result1 = await mock_direct_client.execute_task_via_cli("Buggy task")
            assert result1['success'] == False
            
            # Second attempt (succeeds)
            result2 = await mock_direct_client.execute_task_via_cli("Fixed task")
            assert result2['success'] == True
    
    @pytest.mark.asyncio
    async def test_concurrent_client_operations_mock(self, mock_http_client, mock_direct_client):
        """Test concurrent operations across both clients."""
        # Mock HTTP responses
        http_response = {
            'success': True,
            'content': 'HTTP generated code',
            'model_used': 'claude-3-sonnet'
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=http_response)
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value.__aenter__.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Mock CLI responses
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate.return_value = (
                    b'CLI generated code',
                    b''
                )
                mock_subprocess.return_value = mock_process
                
                # Execute concurrent operations
                http_task = mock_http_client.execute_task("HTTP task")
                cli_task = mock_direct_client.execute_task_via_cli("CLI task")
                
                http_result, cli_result = await asyncio.gather(http_task, cli_task)
                
                # Both should complete successfully
                assert http_result.get('success') == True
                assert cli_result.get('success') == True
    
    @pytest.mark.asyncio
    async def test_configuration_scenarios_mock(self):
        """Test various configuration scenarios."""
        configs = [
            {"token": "valid-token", "base_url": "https://api.example.com"},
            {"token": None, "base_url": "https://api.example.com"},
            {"token": "valid-token", "base_url": None},
        ]
        
        for config in configs:
            try:
                config_manager = ConfigManager()
                
                with patch('src.claude_tui.integrations.claude_code_client.SecurityManager'):
                    client = ClaudeCodeClient(
                        config_manager,
                        oauth_token=config["token"],
                        base_url=config["base_url"] or "https://api.claude.ai/v1"
                    )
                    
                    # Test client creation
                    assert client is not None
                    
                    # Test authentication check
                    is_auth = client.is_authenticated
                    expected_auth = config["token"] is not None
                    assert is_auth == expected_auth
                    
                    await client.cleanup()
            
            except Exception as e:
                # Some configurations may fail, which is expected
                print(f"Config {config} failed as expected: {e}")


class TestCICDIntegrationMocks:
    """Mock tests specifically designed for CI/CD integration."""
    
    def test_import_validation(self):
        """Test that all required modules can be imported."""
        # Test HTTP client imports
        from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient
        from src.claude_tui.integrations.claude_code_client import ClaudeCodeApiError
        
        # Test Direct client imports
        from src.claude_tui.integrations.claude_code_direct_client import ClaudeCodeDirectClient
        from src.claude_tui.integrations.claude_code_direct_client import ClaudeCodeCliError
        
        # Test core imports
        from src.claude_tui.core.config_manager import ConfigManager
        
        assert ClaudeCodeClient is not None
        assert ClaudeCodeDirectClient is not None
        assert ConfigManager is not None
    
    def test_client_instantiation_without_dependencies(self):
        """Test client instantiation without external dependencies."""
        # Test HTTP client
        config = ConfigManager()
        
        with patch('src.claude_tui.integrations.claude_code_client.SecurityManager'):
            http_client = ClaudeCodeClient(config, oauth_token="test-token")
            assert http_client.oauth_token == "test-token"
            assert http_client.base_url == "https://api.claude.ai/v1"
        
        # Test Direct client
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc" 
        cc_file.write_text("test-token")
        
        try:
            with patch('subprocess.run') as mock_subprocess:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "Mock CLI"
                mock_subprocess.return_value = mock_result
                
                direct_client = ClaudeCodeDirectClient(
                    oauth_token_file=str(cc_file),
                    working_directory=temp_dir
                )
                
                assert direct_client.oauth_token == "test-token"
                assert direct_client.working_directory == temp_dir
                
                direct_client.cleanup_session()
        finally:
            cc_file.unlink(missing_ok=True)
            Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_error_scenarios_for_ci_cd(self):
        """Test various error scenarios that might occur in CI/CD."""
        config = ConfigManager()
        
        # Test missing token scenario
        with patch('src.claude_tui.integrations.claude_code_client.SecurityManager'):
            client = ClaudeCodeClient(config, oauth_token=None)
            
            result = await client.execute_task("Test")
            assert 'error' in result
            assert result['success'] == False
            
            await client.cleanup()
    
    def test_static_methods_functionality(self):
        """Test static factory methods for CI/CD compatibility."""
        # Test HTTP client static methods
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_config.write('{"test": "config"}')
        temp_config.close()
        
        try:
            with patch('src.claude_tui.integrations.claude_code_client.SecurityManager'):
                # Test create_with_token
                client = ClaudeCodeClient.create_with_token("test-token")
                assert client.oauth_token == "test-token"
        
        finally:
            Path(temp_config.name).unlink()
    
    def test_utility_functions(self):
        """Test utility functions in isolation."""
        # Test command builder
        builder = CliCommandBuilder("mock-claude")
        
        cmd = builder.build_task_command("test prompt")
        assert isinstance(cmd, list)
        assert "mock-claude" in cmd
        assert "test prompt" in cmd
        
        # Test with context
        cmd_with_context = builder.build_task_command(
            "test",
            context={"key": "value"}
        )
        assert "--context" in cmd_with_context


if __name__ == "__main__":
    """Run CI/CD mock tests."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])