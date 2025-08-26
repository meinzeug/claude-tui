#!/usr/bin/env python3
"""Token verification script for Anthropic API."""

import asyncio
import aiohttp
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The OAuth token provided
OAUTH_TOKEN = "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"

async def test_token_formats():
    """Test different ways to use the OAuth token with Anthropic API."""
    
    base_url = "https://api.anthropic.com"
    test_message = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    }
    
    # Test scenarios
    scenarios = [
        ("x-api-key header", {"x-api-key": OAUTH_TOKEN}),
        ("Authorization Bearer", {"Authorization": f"Bearer {OAUTH_TOKEN}"}),
        ("Authorization header direct", {"Authorization": OAUTH_TOKEN}),
    ]
    
    for scenario_name, headers in scenarios:
        logger.info(f"\n--- Testing {scenario_name} ---")
        
        full_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **headers
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/v1/messages",
                    headers=full_headers,
                    json=test_message,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    text = await response.text()
                    logger.info(f"Status: {response.status}")
                    
                    if response.status == 200:
                        logger.info("✅ SUCCESS: Token format works!")
                        data = await response.json() if response.content_type == 'application/json' else None
                        if data:
                            logger.info(f"Response: {json.dumps(data, indent=2)[:200]}...")
                    elif response.status == 401:
                        logger.warning(f"❌ Authentication failed: {text}")
                    else:
                        logger.warning(f"⚠️  Status {response.status}: {text[:200]}...")
                        
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
    
    # Special check: The token might be an OAuth token that needs exchange
    logger.info(f"\n--- Token Analysis ---")
    logger.info(f"Token prefix: {OAUTH_TOKEN[:20]}...")
    logger.info(f"Token length: {len(OAUTH_TOKEN)}")
    logger.info(f"Appears to be OAuth format (oat01): {'oat01' in OAUTH_TOKEN}")
    
    if 'oat01' in OAUTH_TOKEN:
        logger.info("This appears to be an OAuth access token, not a direct API key.")
        logger.info("For Anthropic API, you typically need an API key starting with 'sk-ant-api03-'")
        logger.info("This OAuth token might need to be used differently or exchanged.")

if __name__ == "__main__":
    asyncio.run(test_token_formats())