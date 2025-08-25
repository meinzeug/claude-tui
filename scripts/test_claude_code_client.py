#!/usr/bin/env python3
"""
Test script for Claude Code Client - Basic functionality verification.

This script demonstrates the Claude Code Client usage and validates
that the implementation works correctly with mock data.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from claude_tui.integrations.claude_code_client import (
    ClaudeCodeClient, 
    ClaudeCodeApiError,
    RateLimiter
)
from claude_tui.core.config_manager import ConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockConfigManager:
    """Mock configuration manager for testing."""
    
    def __init__(self):
        self.config = {
            'CLAUDE_CODE_OAUTH_TOKEN': 'test_oauth_token_12345',
            'CLAUDE_CODE_RATE_LIMIT': 60,
            'CLAUDE_CODE_CLIENT_ID': 'test_client_id',
            'CLAUDE_CODE_CLIENT_SECRET': 'test_client_secret'
        }
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)


async def test_rate_limiter():
    """Test the RateLimiter functionality."""
    logger.info("Testing RateLimiter...")
    
    rate_limiter = RateLimiter(requests_per_minute=10)
    
    # Test multiple requests
    for i in range(3):
        await rate_limiter.acquire()
        logger.info(f"Request {i+1} allowed")
    
    logger.info(f"Rate limiter has {len(rate_limiter.request_times)} requests tracked")
    logger.info("✅ RateLimiter test passed")


async def test_client_initialization():
    """Test Claude Code Client initialization."""
    logger.info("Testing Claude Code Client initialization...")
    
    config_manager = MockConfigManager()
    client = ClaudeCodeClient(
        config_manager=config_manager,
        base_url="https://api.test.claude.ai/v1"
    )
    
    # Test properties
    assert client.is_authenticated is True
    assert client.base_url == "https://api.test.claude.ai/v1"
    assert client.oauth_token == "test_oauth_token_12345"
    assert client.session_active is False
    
    logger.info("✅ Client initialization test passed")
    return client


async def test_client_info():
    """Test client information retrieval."""
    logger.info("Testing client info...")
    
    config_manager = MockConfigManager()
    client = ClaudeCodeClient(config_manager=config_manager)
    
    info = client.get_client_info()
    required_keys = [
        'base_url', 'authenticated', 'session_active', 
        'token_expires_at', 'rate_limit_requests_per_minute', 'client_version'
    ]
    
    for key in required_keys:
        assert key in info, f"Missing key: {key}"
    
    assert info['authenticated'] is True
    assert info['client_version'] == '1.0.0'
    
    logger.info("Client info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("✅ Client info test passed")
    return client


async def test_session_creation():
    """Test HTTP session creation."""
    logger.info("Testing HTTP session creation...")
    
    config_manager = MockConfigManager()
    client = ClaudeCodeClient(config_manager=config_manager)
    
    # Create session
    session = await client._ensure_session()
    
    assert session is not None
    assert client.session_active is True
    assert 'Authorization' in session.headers
    assert session.headers['Authorization'] == 'Bearer test_oauth_token_12345'
    
    logger.info("✅ Session creation test passed")
    
    # Cleanup
    await client.cleanup()
    assert client.session_active is False
    
    logger.info("✅ Session cleanup test passed")


async def test_static_methods():
    """Test static utility methods."""
    logger.info("Testing static methods...")
    
    # Test create_with_token
    client = ClaudeCodeClient.create_with_token(
        oauth_token="static_test_token",
        base_url="https://static.test.api"
    )
    
    assert client.oauth_token == "static_test_token"
    assert client.base_url == "https://static.test.api"
    
    logger.info("✅ Static methods test passed")
    
    # Cleanup
    await client.cleanup()


async def test_context_manager():
    """Test async context manager functionality."""
    logger.info("Testing async context manager...")
    
    config_manager = MockConfigManager()
    
    async with ClaudeCodeClient(config_manager) as client:
        assert isinstance(client, ClaudeCodeClient)
        assert client.is_authenticated is True
        
        # Initialize session to test cleanup
        await client._ensure_session()
        assert client.session_active is True
    
    # Client should be cleaned up after context exit
    assert client.session_active is False
    
    logger.info("✅ Context manager test passed")


async def test_error_handling():
    """Test error handling for edge cases."""
    logger.info("Testing error handling...")
    
    # Test client without token
    config_manager = MockConfigManager()
    config_manager.config['CLAUDE_CODE_OAUTH_TOKEN'] = None
    
    client = ClaudeCodeClient(config_manager=config_manager)
    
    assert client.is_authenticated is False
    
    # Test auth error
    try:
        await client._ensure_auth()
        assert False, "Should have raised ClaudeCodeApiError"
    except ClaudeCodeApiError as e:
        assert "No OAuth token available" in str(e)
        logger.info(f"Correctly caught auth error: {e}")
    
    logger.info("✅ Error handling test passed")


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Claude Code Client tests...")
    
    try:
        # Run tests
        await test_rate_limiter()
        await test_client_initialization()
        await test_client_info()
        await test_session_creation()
        await test_static_methods()
        await test_context_manager()
        await test_error_handling()
        
        logger.info("🎉 All tests passed successfully!")
        logger.info("✅ Claude Code Client is working correctly")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())