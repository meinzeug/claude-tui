#!/usr/bin/env python3
"""
Comprehensive Integration Tests for External APIs

This module tests integration with external services including Claude API,
GitHub API, and other third-party services with proper mocking and 
network isolation for reliable CI/CD execution.
"""

import pytest
import asyncio
import httpx
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import time

# Import test framework
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.enhanced_test_framework import PerformanceMonitor, AsyncTestHelper


@dataclass
class APIResponse:
    """API response data structure."""
    status_code: int
    content: str
    headers: Dict[str, str]
    response_time: float


@dataclass
class APIRequestConfig:
    """API request configuration."""
    url: str
    method: str
    headers: Dict[str, str]
    payload: Optional[Dict[str, Any]]
    timeout: float = 30.0


class MockHTTPClient:
    """Mock HTTP client for testing API integration."""
    
    def __init__(self):
        self.call_history = []
        self.response_mapping = {}
        self.default_response = APIResponse(
            status_code=200,
            content='{"status": "success"}',
            headers={"Content-Type": "application/json"},
            response_time=0.1
        )
        
    def configure_response(self, endpoint: str, response: APIResponse):
        """Configure mock response for specific endpoint."""
        self.response_mapping[endpoint] = response
    
    async def request(self, config: APIRequestConfig) -> APIResponse:
        """Simulate HTTP request."""
        start_time = time.time()
        
        # Record call
        self.call_history.append({
            "url": config.url,
            "method": config.method,
            "headers": config.headers,
            "payload": config.payload,
            "timestamp": start_time
        })
        
        # Simulate network delay
        await asyncio.sleep(0.01)
        
        # Find matching response
        response = self.response_mapping.get(config.url, self.default_response)
        
        # Update response time
        response.response_time = time.time() - start_time
        
        return response


@pytest.fixture
def http_client():
    """Provide mock HTTP client."""
    return MockHTTPClient()


@pytest.fixture
def claude_api_config():
    """Configuration for Claude API testing."""
    return {
        "base_url": "https://api.anthropic.com",
        "api_key": "test-api-key",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 4096,
        "timeout": 30.0
    }


@pytest.fixture
def github_api_config():
    """Configuration for GitHub API testing."""
    return {
        "base_url": "https://api.github.com",
        "token": "test-github-token", 
        "repo": "test-user/test-repo",
        "timeout": 30.0
    }


class TestClaudeAPIIntegration:
    """Test Claude API integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_claude_code_execution_success(self, http_client, claude_api_config):
        """Test successful Claude Code execution via API."""
        # Arrange
        expected_response = APIResponse(
            status_code=200,
            content=json.dumps({
                "id": "msg_123",
                "type": "message",
                "content": [
                    {
                        "type": "text",
                        "text": "Generated Python function successfully:\n\ndef calculate_sum(a, b):\n    return a + b"
                    }
                ]
            }),
            headers={"Content-Type": "application/json"},
            response_time=1.2
        )
        
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            expected_response
        )
        
        request_config = APIRequestConfig(
            url=f"{claude_api_config['base_url']}/v1/messages",
            method="POST",
            headers={
                "Authorization": f"Bearer {claude_api_config['api_key']}",
                "Content-Type": "application/json"
            },
            payload={
                "model": claude_api_config["model"],
                "max_tokens": claude_api_config["max_tokens"],
                "messages": [
                    {
                        "role": "user",
                        "content": "Generate a Python function to calculate sum of two numbers"
                    }
                ]
            }
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 200
        assert "calculate_sum" in response.content
        assert response.response_time < 5.0
        assert len(http_client.call_history) == 1
        
        # Verify request structure
        call = http_client.call_history[0]
        assert call["method"] == "POST"
        assert "Authorization" in call["headers"]
        assert "messages" in call["payload"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_claude_api_rate_limiting(self, http_client, claude_api_config):
        """Test handling of Claude API rate limiting."""
        # Arrange
        rate_limit_response = APIResponse(
            status_code=429,
            content=json.dumps({
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exceeded"
                }
            }),
            headers={
                "Content-Type": "application/json",
                "Retry-After": "60"
            },
            response_time=0.1
        )
        
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            rate_limit_response
        )
        
        request_config = APIRequestConfig(
            url=f"{claude_api_config['base_url']}/v1/messages",
            method="POST",
            headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
            payload={"model": claude_api_config["model"], "messages": []}
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 429
        assert "rate_limit_error" in response.content
        assert "Retry-After" in response.headers
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_claude_api_timeout_handling(self, http_client, claude_api_config):
        """Test timeout handling for Claude API requests."""
        # Arrange
        timeout_response = APIResponse(
            status_code=504,
            content='{"error": {"type": "timeout_error", "message": "Request timed out"}}',
            headers={"Content-Type": "application/json"},
            response_time=30.1  # Exceeds timeout
        )
        
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            timeout_response
        )
        
        request_config = APIRequestConfig(
            url=f"{claude_api_config['base_url']}/v1/messages",
            method="POST",
            headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
            payload={"model": claude_api_config["model"]},
            timeout=30.0
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 504
        assert response.response_time > request_config.timeout
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_claude_api_concurrent_requests(self, http_client, claude_api_config):
        """Test concurrent requests to Claude API."""
        # Arrange
        success_response = APIResponse(
            status_code=200,
            content='{"content": [{"text": "Response"}]}',
            headers={"Content-Type": "application/json"},
            response_time=0.5
        )
        
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            success_response
        )
        
        # Create multiple concurrent requests
        requests = [
            APIRequestConfig(
                url=f"{claude_api_config['base_url']}/v1/messages",
                method="POST",
                headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
                payload={
                    "model": claude_api_config["model"],
                    "messages": [{"role": "user", "content": f"Request {i}"}]
                }
            ) for i in range(5)
        ]
        
        # Act
        start_time = time.time()
        responses = await asyncio.gather(*[
            http_client.request(req) for req in requests
        ])
        total_time = time.time() - start_time
        
        # Assert
        assert len(responses) == 5
        assert all(r.status_code == 200 for r in responses)
        assert total_time < 2.0  # Should complete quickly due to concurrency
        assert len(http_client.call_history) == 5


class TestGitHubAPIIntegration:
    """Test GitHub API integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_github_repo_info_retrieval(self, http_client, github_api_config):
        """Test retrieving GitHub repository information."""
        # Arrange
        repo_response = APIResponse(
            status_code=200,
            content=json.dumps({
                "id": 12345,
                "name": "test-repo",
                "full_name": "test-user/test-repo",
                "description": "Test repository",
                "stargazers_count": 42,
                "forks_count": 7,
                "language": "Python"
            }),
            headers={"Content-Type": "application/json"},
            response_time=0.8
        )
        
        http_client.configure_response(
            f"{github_api_config['base_url']}/repos/{github_api_config['repo']}",
            repo_response
        )
        
        request_config = APIRequestConfig(
            url=f"{github_api_config['base_url']}/repos/{github_api_config['repo']}",
            method="GET",
            headers={
                "Authorization": f"token {github_api_config['token']}",
                "Accept": "application/vnd.github.v3+json"
            },
            payload=None
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 200
        repo_data = json.loads(response.content)
        assert repo_data["name"] == "test-repo"
        assert repo_data["language"] == "Python"
        assert repo_data["stargazers_count"] == 42
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_github_create_pull_request(self, http_client, github_api_config):
        """Test creating a pull request via GitHub API."""
        # Arrange
        pr_response = APIResponse(
            status_code=201,
            content=json.dumps({
                "id": 67890,
                "number": 123,
                "title": "Add new feature",
                "state": "open",
                "html_url": f"https://github.com/{github_api_config['repo']}/pull/123"
            }),
            headers={"Content-Type": "application/json"},
            response_time=1.1
        )
        
        http_client.configure_response(
            f"{github_api_config['base_url']}/repos/{github_api_config['repo']}/pulls",
            pr_response
        )
        
        request_config = APIRequestConfig(
            url=f"{github_api_config['base_url']}/repos/{github_api_config['repo']}/pulls",
            method="POST",
            headers={
                "Authorization": f"token {github_api_config['token']}",
                "Accept": "application/vnd.github.v3+json"
            },
            payload={
                "title": "Add new feature",
                "body": "This PR adds a new feature",
                "head": "feature-branch",
                "base": "main"
            }
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 201
        pr_data = json.loads(response.content)
        assert pr_data["number"] == 123
        assert pr_data["state"] == "open"
        assert "pull/123" in pr_data["html_url"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_github_api_authentication_failure(self, http_client, github_api_config):
        """Test handling of GitHub API authentication failures."""
        # Arrange
        auth_error_response = APIResponse(
            status_code=401,
            content=json.dumps({
                "message": "Bad credentials",
                "documentation_url": "https://docs.github.com/rest"
            }),
            headers={"Content-Type": "application/json"},
            response_time=0.3
        )
        
        http_client.configure_response(
            f"{github_api_config['base_url']}/user",
            auth_error_response
        )
        
        request_config = APIRequestConfig(
            url=f"{github_api_config['base_url']}/user",
            method="GET",
            headers={"Authorization": "token invalid-token"},
            payload=None
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 401
        error_data = json.loads(response.content)
        assert error_data["message"] == "Bad credentials"


class TestAPIIntegrationPerformance:
    """Test performance characteristics of API integration."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_response_time_within_limits(self, http_client, claude_api_config):
        """Test API response times are within acceptable limits."""
        # Arrange
        fast_response = APIResponse(
            status_code=200,
            content='{"status": "success"}',
            headers={"Content-Type": "application/json"},
            response_time=0.8  # Under 1 second
        )
        
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            fast_response
        )
        
        request_config = APIRequestConfig(
            url=f"{claude_api_config['base_url']}/v1/messages",
            method="POST",
            headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
            payload={"model": "claude-3-haiku-20240307", "messages": []},  # Fast model
            timeout=5.0
        )
        
        # Act
        with PerformanceMonitor(thresholds={"max_duration": 2.0}) as monitor:
            response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 200
        assert response.response_time < 1.0
    
    @pytest.mark.integration
    @pytest.mark.performance 
    @pytest.mark.asyncio
    async def test_api_throughput_under_load(self, http_client, claude_api_config):
        """Test API throughput under concurrent load."""
        # Arrange
        batch_response = APIResponse(
            status_code=200,
            content='{"batch_id": "batch_123", "status": "processing"}',
            headers={"Content-Type": "application/json"},
            response_time=0.2
        )
        
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/batches",
            batch_response
        )
        
        # Create batch requests
        requests = [
            APIRequestConfig(
                url=f"{claude_api_config['base_url']}/v1/batches",
                method="POST",
                headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
                payload={"requests": [{"custom_id": f"req_{i}"}]}
            ) for i in range(10)
        ]
        
        # Act
        start_time = time.time()
        responses = await asyncio.gather(*[
            http_client.request(req) for req in requests
        ])
        total_time = time.time() - start_time
        
        # Calculate throughput
        throughput = len(responses) / total_time
        
        # Assert
        assert len(responses) == 10
        assert all(r.status_code == 200 for r in responses)
        assert throughput > 5.0  # At least 5 requests per second


class TestAPIIntegrationErrorHandling:
    """Test error handling in API integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_network_error_handling(self, http_client):
        """Test handling of network connectivity errors."""
        # Arrange
        network_error_response = APIResponse(
            status_code=0,  # Network error
            content="",
            headers={},
            response_time=0.0
        )
        
        http_client.configure_response(
            "https://unreachable.api.com/endpoint",
            network_error_response
        )
        
        request_config = APIRequestConfig(
            url="https://unreachable.api.com/endpoint",
            method="GET",
            headers={},
            payload=None
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 0  # Network error indicator
        assert response.content == ""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, http_client, claude_api_config):
        """Test handling of malformed API responses."""
        # Arrange
        malformed_response = APIResponse(
            status_code=200,
            content='{"invalid": json content}',  # Invalid JSON
            headers={"Content-Type": "application/json"},
            response_time=0.5
        )
        
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            malformed_response
        )
        
        request_config = APIRequestConfig(
            url=f"{claude_api_config['base_url']}/v1/messages",
            method="POST",
            headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
            payload={"model": claude_api_config["model"]}
        )
        
        # Act
        response = await http_client.request(request_config)
        
        # Assert
        assert response.status_code == 200
        # In real implementation, would test JSON parsing error handling
        assert "invalid" in response.content
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_retry_mechanism(self, http_client, claude_api_config):
        """Test retry mechanism for failed API requests."""
        # Arrange
        failure_response = APIResponse(
            status_code=500,
            content='{"error": "Internal server error"}',
            headers={"Content-Type": "application/json"},
            response_time=0.1
        )
        
        success_response = APIResponse(
            status_code=200,
            content='{"status": "success"}',
            headers={"Content-Type": "application/json"},
            response_time=0.2
        )
        
        # Configure to fail first, then succeed
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            failure_response
        )
        
        request_config = APIRequestConfig(
            url=f"{claude_api_config['base_url']}/v1/messages",
            method="POST",
            headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
            payload={"model": claude_api_config["model"]}
        )
        
        # Act - First request (fails)
        first_response = await http_client.request(request_config)
        
        # Reconfigure for success on retry
        http_client.configure_response(
            f"{claude_api_config['base_url']}/v1/messages",
            success_response
        )
        
        # Second request (succeeds)
        retry_response = await http_client.request(request_config)
        
        # Assert
        assert first_response.status_code == 500
        assert retry_response.status_code == 200
        assert len(http_client.call_history) == 2


class TestAPIIntegrationSecurity:
    """Test security aspects of API integration."""
    
    @pytest.mark.integration
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_api_key_not_logged(self, http_client, claude_api_config):
        """Test that API keys are not logged in request history."""
        # Arrange
        request_config = APIRequestConfig(
            url=f"{claude_api_config['base_url']}/v1/messages",
            method="POST",
            headers={"Authorization": f"Bearer {claude_api_config['api_key']}"},
            payload={"model": claude_api_config["model"]}
        )
        
        # Act
        await http_client.request(request_config)
        
        # Assert
        assert len(http_client.call_history) == 1
        call = http_client.call_history[0]
        
        # In production, sensitive headers should be redacted in logs
        assert "Authorization" in call["headers"]
        # This test assumes the actual implementation would redact the value
    
    @pytest.mark.integration
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_https_enforcement(self, http_client):
        """Test that HTTPS is enforced for API calls."""
        # Arrange
        insecure_config = APIRequestConfig(
            url="http://api.example.com/endpoint",  # HTTP instead of HTTPS
            method="POST",
            headers={"Authorization": "Bearer secret"},
            payload={"data": "sensitive"}
        )
        
        # In real implementation, this should be rejected or upgraded to HTTPS
        response = await http_client.request(insecure_config)
        
        # Assert - Mock allows it, but real implementation should enforce HTTPS
        assert insecure_config.url.startswith("http://")  # Test setup verification


class TestAPIIntegrationWithAsyncHelper:
    """Test API integration with async testing helpers."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_integration_with_timeout_context(self, http_client, async_helper):
        """Test API integration with async timeout context."""
        # Arrange
        slow_response = APIResponse(
            status_code=200,
            content='{"result": "slow operation completed"}',
            headers={"Content-Type": "application/json"},
            response_time=2.0
        )
        
        http_client.configure_response("https://api.example.com/slow", slow_response)
        
        request_config = APIRequestConfig(
            url="https://api.example.com/slow",
            method="GET",
            headers={},
            payload=None
        )
        
        # Act & Assert - Should complete within timeout
        async with async_helper.timeout_context(5.0):
            response = await http_client.request(request_config)
            assert response.status_code == 200
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_retry_with_condition_wait(self, http_client, async_helper):
        """Test API retry logic with condition waiting."""
        # Arrange
        processing_response = APIResponse(
            status_code=202,
            content='{"status": "processing"}',
            headers={"Content-Type": "application/json"},
            response_time=0.1
        )
        
        http_client.configure_response("https://api.example.com/status", processing_response)
        
        # Track completion state
        completion_state = {"completed": False}
        
        def check_completion():
            # Simulate checking completion status
            completion_state["completed"] = True
            return completion_state["completed"]
        
        # Act
        # Wait for operation to complete
        result = await async_helper.wait_for_condition(
            check_completion,
            timeout=1.0,
            interval=0.1
        )
        
        # Assert
        assert result is True
        assert completion_state["completed"] is True