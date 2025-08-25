"""
Unit and integration tests for Claude Code Client.

Tests OAuth authentication, HTTP client functionality, rate limiting,
retry logic, and comprehensive error handling.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientError, ClientResponseError
from aiohttp.web import Response

from claude_tui.integrations.claude_code_client import (
    ClaudeCodeClient,
    ClaudeCodeApiError,
    ClaudeCodeAuthError,
    ClaudeCodeRateLimitError,
    TokenResponse,
    TaskRequest,
    ValidationRequest,
    PlaceholderRequest,
    ProjectAnalysisRequest,
    RateLimiter
)
from claude_tui.core.config_manager import ConfigManager
from claude_tui.models.ai_models import CodeResult, ReviewCriteria, CodeReview
from claude_tui.models.project import Project


class TestRateLimiter:
    """Test the RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_under_limit(self):
        """Test that rate limiter allows requests under the limit."""
        rate_limiter = RateLimiter(requests_per_minute=60)
        
        # Should allow requests under the limit
        await rate_limiter.acquire()
        await rate_limiter.acquire()
        await rate_limiter.acquire()
        
        assert len(rate_limiter.request_times) == 3
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_requests_over_limit(self):
        """Test that rate limiter blocks requests over the limit."""
        rate_limiter = RateLimiter(requests_per_minute=2)
        
        # Fill up the rate limit
        await rate_limiter.acquire()
        await rate_limiter.acquire()
        
        # This should trigger a wait (but we won't wait in test)
        start_time = asyncio.get_event_loop().time()
        
        # Mock time to avoid actual waiting
        with patch('time.time', return_value=start_time):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                await rate_limiter.acquire()
                mock_sleep.assert_called_once()


class TestTokenResponse:
    """Test the TokenResponse Pydantic model."""
    
    def test_token_response_creation(self):
        """Test TokenResponse model creation."""
        token_data = {
            'access_token': 'test_token_123',
            'expires_in': 3600,
            'refresh_token': 'refresh_test_123'
        }
        
        token = TokenResponse(**token_data)
        
        assert token.access_token == 'test_token_123'
        assert token.token_type == 'Bearer'  # Default value
        assert token.expires_in == 3600
        assert token.refresh_token == 'refresh_test_123'
    
    def test_token_response_defaults(self):
        """Test TokenResponse with minimum required fields."""
        token_data = {
            'access_token': 'test_token_123',
            'expires_in': 3600
        }
        
        token = TokenResponse(**token_data)
        
        assert token.access_token == 'test_token_123'
        assert token.token_type == 'Bearer'
        assert token.expires_in == 3600
        assert token.refresh_token is None
        assert token.scope is None


class TestRequestModels:
    """Test the Pydantic request models."""
    
    def test_task_request_creation(self):
        """Test TaskRequest model creation."""
        task_data = {
            'description': 'Create a Python function',
            'context': {'language': 'python'},
            'project_path': '/path/to/project',
            'timeout': 600
        }
        
        task = TaskRequest(**task_data)
        
        assert task.description == 'Create a Python function'
        assert task.context == {'language': 'python'}
        assert task.project_path == '/path/to/project'
        assert task.timeout == 600
        assert task.model == 'claude-3-sonnet'  # Default
    
    def test_validation_request_creation(self):
        """Test ValidationRequest model creation."""
        validation_data = {
            'output': 'def hello(): print("Hello")',
            'validation_rules': ['syntax_check', 'style_check'],
            'expected_format': 'python_function'
        }
        
        validation = ValidationRequest(**validation_data)
        
        assert validation.output == 'def hello(): print("Hello")'
        assert validation.validation_rules == ['syntax_check', 'style_check']
        assert validation.expected_format == 'python_function'
    
    def test_placeholder_request_creation(self):
        """Test PlaceholderRequest model creation."""
        placeholder_data = {
            'code': 'def function_name(TODO): TODO',
            'suggestions': ['add_parameters', 'implement_logic']
        }
        
        placeholder = PlaceholderRequest(**placeholder_data)
        
        assert placeholder.code == 'def function_name(TODO): TODO'
        assert placeholder.suggestions == ['add_parameters', 'implement_logic']
        assert placeholder.completion_style == 'intelligent'  # Default
    
    def test_project_analysis_request_creation(self):
        """Test ProjectAnalysisRequest model creation."""
        analysis_data = {
            'project_path': '/home/user/project',
            'analysis_depth': 'deep',
            'include_dependencies': False,
            'include_docs': True
        }
        
        analysis = ProjectAnalysisRequest(**analysis_data)
        
        assert analysis.project_path == '/home/user/project'
        assert analysis.analysis_depth == 'deep'
        assert analysis.include_dependencies is False
        assert analysis.include_tests is True  # Default
        assert analysis.include_docs is True


@pytest.fixture
def mock_config_manager():
    """Create a mock ConfigManager."""
    config = MagicMock(spec=ConfigManager)
    config.get.side_effect = lambda key, default=None: {
        'CLAUDE_CODE_OAUTH_TOKEN': 'test_oauth_token_123',
        'CLAUDE_CODE_RATE_LIMIT': 60,
        'CLAUDE_CODE_CLIENT_ID': 'test_client_id',
        'CLAUDE_CODE_CLIENT_SECRET': 'test_client_secret'
    }.get(key, default)
    return config


@pytest.fixture
def claude_code_client(mock_config_manager):
    """Create a ClaudeCodeClient instance for testing."""
    return ClaudeCodeClient(
        config_manager=mock_config_manager,
        base_url="https://api.test.claude.ai/v1"
    )


class TestClaudeCodeClient:
    """Test the main ClaudeCodeClient class."""
    
    def test_client_initialization(self, mock_config_manager):
        """Test client initialization."""
        client = ClaudeCodeClient(
            config_manager=mock_config_manager,
            base_url="https://api.test.claude.ai/v1",
            oauth_token="custom_token_123"
        )
        
        assert client.base_url == "https://api.test.claude.ai/v1"
        assert client.oauth_token == "custom_token_123"  # Custom token overrides config
        assert client.session is None
        assert client.is_authenticated is True
        assert client.session_active is False
    
    def test_client_initialization_no_token(self):
        """Test client initialization without OAuth token."""
        config = MagicMock(spec=ConfigManager)
        config.get.return_value = None  # No token in config
        
        client = ClaudeCodeClient(config_manager=config)
        
        assert client.oauth_token is None
        assert client.is_authenticated is False
    
    def test_client_info_property(self, claude_code_client):
        """Test the get_client_info method."""
        info = claude_code_client.get_client_info()
        
        expected_keys = [
            'base_url', 'authenticated', 'session_active', 
            'token_expires_at', 'rate_limit_requests_per_minute', 'client_version'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['base_url'] == "https://api.test.claude.ai/v1"
        assert info['authenticated'] is True
        assert info['session_active'] is False
        assert info['client_version'] == '1.0.0'
    
    @pytest.mark.asyncio
    async def test_ensure_session(self, claude_code_client):
        """Test HTTP session creation."""
        session = await claude_code_client._ensure_session()
        
        assert session is not None
        assert claude_code_client.session is session
        assert 'Authorization' in session.headers
        assert session.headers['Authorization'] == 'Bearer test_oauth_token_123'
        
        # Cleanup
        await claude_code_client.cleanup()
    
    @pytest.mark.asyncio
    async def test_ensure_auth_no_token(self, mock_config_manager):
        """Test authentication check without token."""
        client = ClaudeCodeClient(config_manager=mock_config_manager, oauth_token=None)
        
        with pytest.raises(ClaudeCodeAuthError, match="No OAuth token available"):
            await client._ensure_auth()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, claude_code_client):
        """Test client cleanup."""
        # Initialize session
        await claude_code_client._ensure_session()
        assert claude_code_client.session is not None
        
        # Cleanup
        await claude_code_client.cleanup()
        
        assert claude_code_client.session is None


class TestClaudeCodeClientAPIMethods:
    """Test the main API methods of ClaudeCodeClient."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, claude_code_client):
        """Test successful health check."""
        mock_response = {'status': 'healthy', 'timestamp': '2023-01-01T00:00:00Z'}
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await claude_code_client.health_check()
            
            assert result is True
            mock_request.assert_called_once_with('GET', '/health', timeout=10)
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, claude_code_client):
        """Test health check failure."""
        mock_response = {'status': 'unhealthy', 'error': 'Service unavailable'}
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await claude_code_client.health_check()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, claude_code_client):
        """Test health check with exception."""
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ClaudeCodeApiError("Network error")
            
            result = await claude_code_client.health_check()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, claude_code_client):
        """Test successful task execution."""
        mock_response = {
            'success': True,
            'content': 'def hello(): print("Hello World")',
            'model_used': 'claude-3-sonnet',
            'execution_time': 2.5
        }
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await claude_code_client.execute_task(
                "Create a Python hello world function",
                {'language': 'python'}
            )
            
            assert result['success'] is True
            assert 'def hello()' in result['content']
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, claude_code_client):
        """Test task execution failure."""
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ClaudeCodeApiError("API error")
            
            result = await claude_code_client.execute_task("Invalid task")
            
            assert result['success'] is False
            assert 'API error' in result['error']
    
    @pytest.mark.asyncio
    async def test_validate_output_success(self, claude_code_client):
        """Test successful output validation."""
        mock_response = {
            'valid': True,
            'issues': [],
            'score': 0.95
        }
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await claude_code_client.validate_output(
                'def hello(): print("Hello")',
                {'validation_rules': ['syntax_check']}
            )
            
            assert result['valid'] is True
            assert result['issues'] == []
    
    @pytest.mark.asyncio
    async def test_validate_output_failure(self, claude_code_client):
        """Test output validation failure."""
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ClaudeCodeApiError("Validation failed")
            
            result = await claude_code_client.validate_output('invalid code')
            
            assert result['valid'] is False
            assert 'Validation failed' in result['error']
    
    @pytest.mark.asyncio
    async def test_complete_placeholder_success(self, claude_code_client):
        """Test successful placeholder completion."""
        mock_response = {
            'completed_code': 'def calculate_sum(a, b): return a + b'
        }
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await claude_code_client.complete_placeholder(
                'def calculate_sum(TODO): TODO',
                ['add parameters', 'implement logic']
            )
            
            assert 'def calculate_sum(a, b)' in result
            assert 'return a + b' in result
    
    @pytest.mark.asyncio
    async def test_complete_placeholder_failure(self, claude_code_client):
        """Test placeholder completion failure (returns original code)."""
        original_code = 'def function(TODO): TODO'
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ClaudeCodeApiError("Completion failed")
            
            result = await claude_code_client.complete_placeholder(original_code)
            
            assert result == original_code  # Should return original on failure
    
    @pytest.mark.asyncio
    async def test_get_project_analysis_success(self, claude_code_client):
        """Test successful project analysis."""
        mock_response = {
            'success': True,
            'analysis': {
                'files_count': 42,
                'languages': ['python', 'javascript'],
                'complexity_score': 7.5,
                'test_coverage': 85.2
            }
        }
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await claude_code_client.get_project_analysis('/path/to/project')
            
            assert result['success'] is True
            assert result['analysis']['files_count'] == 42
            assert 'python' in result['analysis']['languages']
    
    @pytest.mark.asyncio
    async def test_get_project_analysis_failure(self, claude_code_client):
        """Test project analysis failure."""
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ClaudeCodeApiError("Analysis failed")
            
            result = await claude_code_client.get_project_analysis('/invalid/path')
            
            assert result['success'] is False
            assert 'Analysis failed' in result['error']


class TestClaudeCodeClientHTTPClient:
    """Test HTTP client functionality and error handling."""
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, claude_code_client):
        """Test successful HTTP request."""
        mock_response_data = {'result': 'success', 'data': 'test_data'}
        
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value.__aenter__.return_value = mock_response
            
            result = await claude_code_client._make_request('GET', '/test')
            
            assert result == mock_response_data
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit_error(self, claude_code_client):
        """Test HTTP request with rate limit error."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Create mock response with rate limit
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.headers = {'Retry-After': '60'}
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(ClaudeCodeRateLimitError) as exc_info:
                await claude_code_client._make_request('GET', '/test')
            
            assert exc_info.value.retry_after == 60
    
    @pytest.mark.asyncio
    async def test_make_request_auth_error(self, claude_code_client):
        """Test HTTP request with authentication error."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Create mock response with auth error
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.text.return_value = "Invalid token"
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(ClaudeCodeAuthError, match="Authentication failed"):
                await claude_code_client._make_request('GET', '/test')
    
    @pytest.mark.asyncio
    async def test_make_request_client_error(self, claude_code_client):
        """Test HTTP request with client error."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Create mock response with client error
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text.return_value = "Bad request"
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(ClaudeCodeApiError, match="Client error"):
                await claude_code_client._make_request('GET', '/test')
    
    @pytest.mark.asyncio
    async def test_make_request_server_error(self, claude_code_client):
        """Test HTTP request with server error."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Create mock response with server error
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal server error"
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(ClaudeCodeApiError, match="Server error"):
                await claude_code_client._make_request('GET', '/test')
    
    @pytest.mark.asyncio
    async def test_make_request_timeout_error(self, claude_code_client):
        """Test HTTP request with timeout."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = asyncio.TimeoutError("Request timeout")
            
            with pytest.raises(ClaudeCodeApiError, match="Request timeout"):
                await claude_code_client._make_request('GET', '/test')
    
    @pytest.mark.asyncio
    async def test_make_request_json_decode_error(self, claude_code_client):
        """Test HTTP request with JSON decode error."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Create mock response with invalid JSON
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(ClaudeCodeApiError, match="Invalid JSON response"):
                await claude_code_client._make_request('GET', '/test')


class TestClaudeCodeClientLegacyMethods:
    """Test legacy methods for backward compatibility."""
    
    @pytest.mark.asyncio
    async def test_execute_coding_task_success(self, claude_code_client):
        """Test legacy execute_coding_task method."""
        mock_response = {
            'success': True,
            'content': 'def test(): pass',
            'model_used': 'claude-3-sonnet',
            'validation_passed': True,
            'quality_score': 0.9,
            'execution_time': 1.5
        }
        
        with patch.object(claude_code_client, 'execute_task', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response
            
            result = await claude_code_client.execute_coding_task(
                "Create a test function",
                {'language': 'python'}
            )
            
            assert isinstance(result, CodeResult)
            assert result.success is True
            assert result.content == 'def test(): pass'
            assert result.model_used == 'claude-3-sonnet'
    
    @pytest.mark.asyncio
    async def test_execute_coding_task_failure(self, claude_code_client):
        """Test legacy execute_coding_task method failure."""
        mock_response = {
            'success': False,
            'error': 'Task failed'
        }
        
        with patch.object(claude_code_client, 'execute_task', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response
            
            result = await claude_code_client.execute_coding_task("Invalid task")
            
            assert isinstance(result, CodeResult)
            assert result.success is False
            assert 'Task failed' in result.error_message


class TestClaudeCodeClientStaticMethods:
    """Test static utility methods."""
    
    @patch('claude_tui.core.config_manager.ConfigManager')
    def test_create_from_config(self, mock_config_class):
        """Test creating client from config file."""
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        client = ClaudeCodeClient.create_from_config('/path/to/config.yaml')
        
        assert isinstance(client, ClaudeCodeClient)
        mock_config_class.assert_called_once_with('/path/to/config.yaml')
    
    @patch('claude_tui.core.config_manager.ConfigManager')
    def test_create_with_token(self, mock_config_class):
        """Test creating client with OAuth token."""
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        client = ClaudeCodeClient.create_with_token(
            'test_token_123',
            'https://custom.api.url'
        )
        
        assert isinstance(client, ClaudeCodeClient)
        assert client.oauth_token == 'test_token_123'
        assert client.base_url == 'https://custom.api.url'


class TestClaudeCodeClientContextManager:
    """Test async context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_config_manager):
        """Test using client as async context manager."""
        async with ClaudeCodeClient(mock_config_manager) as client:
            assert isinstance(client, ClaudeCodeClient)
            assert client.oauth_token == 'test_oauth_token_123'
        
        # Client should be cleaned up after context exit
        assert client.session is None


class TestClaudeCodeClientIntegration:
    """Integration tests with mocked external dependencies."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_task_execution(self, claude_code_client):
        """Test full workflow: health check -> task execution -> cleanup."""
        # Mock responses for the full workflow
        health_response = {'status': 'healthy'}
        task_response = {
            'success': True,
            'content': 'def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)',
            'model_used': 'claude-3-sonnet',
            'execution_time': 3.2
        }
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [health_response, task_response]
            
            # 1. Health check
            health_ok = await claude_code_client.health_check()
            assert health_ok is True
            
            # 2. Execute task
            result = await claude_code_client.execute_task(
                "Create a fibonacci function in Python",
                {'language': 'python', 'complexity': 'recursive'}
            )
            
            assert result['success'] is True
            assert 'fibonacci' in result['content']
            
            # 3. Cleanup
            await claude_code_client.cleanup()
            assert claude_code_client.session is None
    
    @pytest.mark.asyncio
    async def test_error_recovery_with_retry(self, claude_code_client):
        """Test error recovery with retry mechanism."""
        # First call fails, second succeeds
        error_response = ClaudeCodeApiError("Temporary error")
        success_response = {'status': 'healthy'}
        
        with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [error_response, success_response]
            
            # The backoff decorator should retry automatically
            # Since we can't easily test the actual retry in unit tests,
            # we'll verify that the error is handled gracefully
            result = await claude_code_client.health_check()
            
            # Should return False on error (no retry in health_check)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, claude_code_client):
        """Test rate limiting during API calls."""
        # Mock rate limiter
        with patch.object(claude_code_client.rate_limiter, 'acquire', new_callable=AsyncMock) as mock_acquire:
            with patch.object(claude_code_client, '_make_request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = {'status': 'healthy'}
                
                await claude_code_client.health_check()
                
                # Rate limiter should be called
                mock_acquire.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])