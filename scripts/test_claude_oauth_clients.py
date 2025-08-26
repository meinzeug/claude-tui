#!/usr/bin/env python3
"""
Test Script for Claude Code OAuth Integration

This script tests both Claude Code clients with the production OAuth token
to ensure they work correctly with the Anthropic Claude API.

Usage:
    python scripts/test_claude_oauth_clients.py
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_tui.integrations.claude_code_client import ClaudeCodeClient
from claude_tui.integrations.claude_code_direct_client import ClaudeDirectClient
from claude_tui.core.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ClaudeClientTester:
    """Comprehensive tester for Claude Code clients with OAuth token."""
    
    def __init__(self):
        """Initialize the tester with production OAuth token."""
        self.production_token = "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"
        self.config_manager = ConfigManager()
        self.test_results = []
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if details:
            logger.info(f"    Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
    
    async def test_claude_code_client(self) -> bool:
        """Test the ClaudeCodeClient with OAuth token."""
        logger.info("\n=== Testing ClaudeCodeClient ===")
        
        try:
            # Initialize client
            client = ClaudeCodeClient(
                config_manager=self.config_manager,
                oauth_token=self.production_token
            )
            
            # Test 1: Client initialization
            success = bool(client.oauth_token)
            self.log_test_result(
                "ClaudeCodeClient - Initialization", 
                success,
                f"Token loaded: {client.oauth_token[:20]}..." if success else "No token"
            )
            
            if not success:
                return False
            
            # Test 2: Health check
            try:
                health_result = await client.health_check()
                success = health_result
                self.log_test_result(
                    "ClaudeCodeClient - Health Check",
                    success,
                    f"Health status: {health_result}"
                )
            except Exception as e:
                self.log_test_result(
                    "ClaudeCodeClient - Health Check",
                    False,
                    f"Error: {e}"
                )
                success = False
            
            # Test 3: Simple task execution
            try:
                task_result = await client.execute_task(
                    "Please respond with 'Hello from Claude Code API!' in JSON format",
                    context={'test': True, 'timeout': 30}
                )
                success = task_result.get('success', False)
                self.log_test_result(
                    "ClaudeCodeClient - Task Execution",
                    success,
                    f"Response: {str(task_result.get('content', ''))[:100]}..."
                )
            except Exception as e:
                self.log_test_result(
                    "ClaudeCodeClient - Task Execution",
                    False,
                    f"Error: {e}"
                )
                success = False
            
            # Clean up
            await client.cleanup()
            
            return success
            
        except Exception as e:
            self.log_test_result(
                "ClaudeCodeClient - Overall Test",
                False,
                f"Exception: {e}"
            )
            return False
    
    async def test_claude_direct_client(self) -> bool:
        """Test the ClaudeDirectClient with OAuth token."""
        logger.info("\n=== Testing ClaudeDirectClient ===")
        
        try:
            # Initialize client
            client = ClaudeDirectClient(
                api_key=self.production_token,
                working_directory="/home/tekkadmin/claude-tui"
            )
            
            # Test 1: Client initialization
            success = bool(client.api_key)
            self.log_test_result(
                "ClaudeDirectClient - Initialization",
                success,
                f"Token loaded: {client.api_key[:20]}..." if success else "No token"
            )
            
            if not success:
                return False
            
            # Test 2: Health check
            try:
                health_result = await client.health_check()
                success = health_result.get('healthy', False)
                self.log_test_result(
                    "ClaudeDirectClient - Health Check",
                    success,
                    f"Response time: {health_result.get('response_time', 'N/A')}s"
                )
            except Exception as e:
                self.log_test_result(
                    "ClaudeDirectClient - Health Check",
                    False,
                    f"Error: {e}"
                )
                success = False
            
            # Test 3: Simple response generation
            try:
                response = await client.generate_response(
                    message="Please respond with 'Hello from Claude Direct API!' and confirm you're working.",
                    max_tokens=100
                )
                success = bool(response.get('content'))
                content = str(response.get('content', []))
                self.log_test_result(
                    "ClaudeDirectClient - Response Generation",
                    success,
                    f"Response: {content[:100]}..."
                )
            except Exception as e:
                self.log_test_result(
                    "ClaudeDirectClient - Response Generation",
                    False,
                    f"Error: {e}"
                )
                success = False
            
            # Test 4: Token counting and cost estimation
            try:
                session_info = client.get_session_info()
                success = session_info.get('session_id') is not None
                self.log_test_result(
                    "ClaudeDirectClient - Session Management",
                    success,
                    f"Requests: {session_info.get('request_count', 0)}, "
                    f"Tokens: {session_info.get('total_tokens', 0)}, "
                    f"Cost: ${session_info.get('estimated_total_cost_usd', 0):.4f}"
                )
            except Exception as e:
                self.log_test_result(
                    "ClaudeDirectClient - Session Management",
                    False,
                    f"Error: {e}"
                )
                success = False
            
            # Clean up
            await client.cleanup_session()
            
            return success
            
        except Exception as e:
            self.log_test_result(
                "ClaudeDirectClient - Overall Test",
                False,
                f"Exception: {e}"
            )
            return False
    
    async def test_oauth_token_from_file(self) -> bool:
        """Test OAuth token loading from .cc file."""
        logger.info("\n=== Testing OAuth Token Loading ===")
        
        try:
            # Test reading .cc file
            cc_file = Path("/home/tekkadmin/claude-tui/.cc")
            if cc_file.exists():
                with open(cc_file, 'r') as f:
                    token = f.read().strip()
                
                success = token == self.production_token
                self.log_test_result(
                    "OAuth Token File - Reading",
                    success,
                    f"Token matches: {success}, Length: {len(token)}"
                )
                return success
            else:
                self.log_test_result(
                    "OAuth Token File - Reading",
                    False,
                    ".cc file not found"
                )
                return False
                
        except Exception as e:
            self.log_test_result(
                "OAuth Token File - Reading",
                False,
                f"Error: {e}"
            )
            return False
    
    def generate_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "="*60)
        logger.info("CLAUDE CODE OAUTH INTEGRATION TEST REPORT")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            logger.info("\nFailed Tests:")
            for result in self.test_results:
                if not result['success']:
                    logger.error(f"  - {result['test']}: {result['details']}")
        
        logger.info("\nTest Summary:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            logger.info(f"  {status} {result['test']}")
        
        # Save detailed report
        report_data = {
            'test_run_timestamp': time.time(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests/total_tests)*100 if total_tests > 0 else 0,
            'test_results': self.test_results
        }
        
        report_file = Path("/home/tekkadmin/claude-tui/oauth_test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        return passed_tests == total_tests
    
    async def run_all_tests(self) -> bool:
        """Run all OAuth integration tests."""
        logger.info("Starting Claude Code OAuth Integration Tests...")
        logger.info(f"Production Token: {self.production_token[:20]}...")
        
        # Test OAuth token file loading
        await self.test_oauth_token_from_file()
        
        # Test both clients
        claude_code_success = await self.test_claude_code_client()
        claude_direct_success = await self.test_claude_direct_client()
        
        # Generate report
        all_passed = self.generate_report()
        
        if all_passed:
            logger.info("\n🎉 All tests PASSED! OAuth integration is working correctly.")
        else:
            logger.error("\n⚠️  Some tests FAILED. Please check the details above.")
        
        return all_passed


async def main():
    """Main test execution."""
    print("Claude Code OAuth Integration Test")
    print("=" * 50)
    
    tester = ClaudeClientTester()
    success = await tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        sys.exit(1)