#!/usr/bin/env python3
"""Test script for Claude Code Client with Anthropic API integration."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OAuth Token
OAUTH_TOKEN = "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"

async def test_health_check():
    """Test the health check functionality."""
    logger.info("Testing health check...")
    
    config_manager = ConfigManager()
    client = ClaudeCodeClient(config_manager, oauth_token=OAUTH_TOKEN)
    
    try:
        async with client:
            is_healthy = await client.health_check()
            logger.info(f"Health check result: {'PASSED' if is_healthy else 'FAILED'}")
            return is_healthy
    except Exception as e:
        logger.error(f"Health check test failed: {e}")
        return False

async def test_simple_task():
    """Test a simple task execution."""
    logger.info("Testing simple task execution...")
    
    config_manager = ConfigManager()
    client = ClaudeCodeClient(config_manager, oauth_token=OAUTH_TOKEN)
    
    try:
        async with client:
            result = await client.execute_task(
                "Write a simple Python function that returns 'Hello World'",
                context={'model': 'claude-3-haiku-20240307', 'max_tokens': 200}
            )
            
            if result.get('success'):
                logger.info("Task execution successful!")
                logger.info(f"Content preview: {result.get('content', '')[:200]}...")
                logger.info(f"Model used: {result.get('model_used')}")
                logger.info(f"Task ID: {result.get('task_id')}")
                return True
            else:
                logger.error(f"Task execution failed: {result.get('error')}")
                return False
                
    except Exception as e:
        logger.error(f"Task execution test failed: {e}")
        return False

async def test_client_info():
    """Test client information retrieval."""
    logger.info("Testing client info...")
    
    config_manager = ConfigManager()
    client = ClaudeCodeClient(config_manager, oauth_token=OAUTH_TOKEN)
    
    try:
        info = client.get_client_info()
        logger.info(f"Client info: {info}")
        return True
    except Exception as e:
        logger.error(f"Client info test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Starting Claude Code Client Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Client Info", test_client_info),
        ("Health Check", test_health_check),
        ("Simple Task", test_simple_task),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = await test_func()
            results.append((test_name, result))
            logger.info(f"{test_name} Test: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name} Test FAILED with exception: {e}")
            results.append((test_name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary:")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Integration is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    # Set OAuth token in environment for potential config loading
    os.environ['ANTHROPIC_API_KEY'] = OAUTH_TOKEN
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)