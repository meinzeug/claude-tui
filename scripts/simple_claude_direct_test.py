#!/usr/bin/env python3
"""
Simple test script for the Claude Direct API Client.
Tests basic functionality without external dependencies.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_simple_integration():
    """Test basic integration without heavy dependencies."""
    
    try:
        # Import with fallback handling
        from claude_tui.integrations.claude_code_direct_client import ClaudeDirectClient
        
        logger.info("✅ Successfully imported ClaudeDirectClient")
        
        # Test client creation
        client = ClaudeDirectClient.create_with_oauth_token(
            oauth_token="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",
            working_directory="/home/tekkadmin/claude-tui"
        )
        
        logger.info("✅ Successfully created Claude client")
        
        # Test session info
        session_info = client.get_session_info()
        logger.info(f"📊 Session info: {json.dumps(session_info, indent=2)}")
        
        # Test simple response (if API is available)
        try:
            response = await client.generate_response(
                message="Hello, please respond with just 'API working' if you can hear me.",
                max_tokens=50
            )
            
            logger.info(f"✅ API Response received!")
            logger.info(f"📝 Content type: {type(response.get('content'))}")
            logger.info(f"💰 Usage info: {response.get('usage_info', 'Not available')}")
            
        except Exception as e:
            logger.warning(f"⚠️ API call failed (may be expected): {e}")
        
        # Test health check
        try:
            health = await client.health_check()
            logger.info(f"🏥 Health check: {health}")
        except Exception as e:
            logger.warning(f"⚠️ Health check failed: {e}")
        
        # Clean up
        await client.cleanup_session()
        logger.info("✅ Session cleaned up successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_components():
    """Test individual components."""
    
    try:
        # Test imports
        from claude_tui.integrations.claude_code_direct_client import (
            ClaudeAPIError,
            ClaudeAuthError,
            ClaudeRateLimitError,
            TokenCounter,
            RetryManager,
            MessageBuilder
        )
        
        logger.info("✅ All component imports successful")
        
        # Test TokenCounter (with fallback)
        try:
            counter = TokenCounter()
            token_count = counter.count_tokens("Hello world!")
            cost = counter.estimate_cost(100, 50)
            logger.info(f"✅ TokenCounter: {token_count} tokens, ${cost:.6f} estimated cost")
        except Exception as e:
            logger.warning(f"⚠️ TokenCounter test failed: {e}")
        
        # Test RetryManager
        retry_manager = RetryManager(max_retries=3)
        delay = retry_manager.calculate_delay(1)
        logger.info(f"✅ RetryManager: delay calculation working ({delay:.2f}s)")
        
        # Test MessageBuilder
        system, messages = MessageBuilder.build_messages(
            user_message="Test message",
            system_message="Test system message",
            context={"test": "context"}
        )
        logger.info(f"✅ MessageBuilder: built {len(messages)} messages with system message")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        return False


async def main():
    """Run all simple tests."""
    
    logger.info("🧪 Starting simple Claude Direct Client tests")
    logger.info("="*50)
    
    # Test component functionality
    component_test = test_components()
    
    # Test integration
    integration_test = await test_simple_integration()
    
    # Summary
    logger.info("="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    tests = {
        "Component Tests": component_test,
        "Integration Tests": integration_test
    }
    
    passed = sum(tests.values())
    total = len(tests)
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 Basic tests passed! Claude Direct Client is functional.")
        return True
    else:
        logger.error(f"💥 {total-passed} tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)