#!/usr/bin/env python3
"""
Claude Code Direct Client Usage Example

This example demonstrates how to use the ClaudeCodeDirectClient for
direct integration with Claude Code CLI using OAuth tokens from .cc files.

This approach works with actual Claude Code OAuth tokens and avoids
the API endpoint restrictions.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from claude_tui.integrations.claude_code_direct_client import ClaudeCodeDirectClient
from claude_tui.core.config_manager import ConfigManager


async def demonstrate_claude_code_direct():
    """Demonstrate ClaudeCodeDirectClient usage."""
    
    print("🚀 Claude Code Direct Client Demo")
    print("=" * 50)
    
    # Method 1: Create client with automatic .cc file detection
    print("\n1. Creating client with automatic OAuth token detection...")
    try:
        client = ClaudeCodeDirectClient()
        
        # Display session information
        session_info = client.get_session_info()
        print(f"   ✅ Session ID: {session_info['session_id']}")
        print(f"   ✅ OAuth Token: {'Available' if session_info['oauth_token_available'] else 'Not found'}")
        print(f"   ✅ CLI Path: {session_info['claude_code_path']}")
        print(f"   ✅ Working Directory: {session_info['working_directory']}")
        
        # Perform health check
        print("\n2. Performing health check...")
        health_result = await client.health_check()
        
        if health_result['healthy']:
            print(f"   ✅ Claude Code CLI is healthy")
            print(f"   ✅ Response time: {health_result['response_time']:.3f}s")
            if health_result.get('cli_version'):
                print(f"   ✅ CLI Version: {health_result['cli_version']}")
        else:
            print(f"   ❌ Health check failed: {health_result.get('error', 'Unknown error')}")
            return
        
        # Example 1: Execute a simple coding task
        print("\n3. Executing coding task...")
        task_result = await client.execute_task_via_cli(
            task_description="Write a Python function that calculates the factorial of a number",
            context={
                "language": "python",
                "style": "clean and well-documented",
                "include_tests": False
            },
            timeout=60
        )
        
        if task_result['success']:
            print("   ✅ Task executed successfully")
            print(f"   ✅ Execution time: {task_result['execution_time']:.2f}s")
            if 'generated_code' in task_result:
                print("   📝 Generated code:")
                print("   " + "─" * 40)
                code_lines = task_result['generated_code'].split('\n')[:10]  # First 10 lines
                for line in code_lines:
                    print(f"   {line}")
                if len(task_result['generated_code'].split('\n')) > 10:
                    print("   ... (truncated)")
            print("   " + "─" * 40)
        else:
            print(f"   ❌ Task failed: {task_result.get('error', 'Unknown error')}")
        
        # Example 2: Validate some code
        print("\n4. Validating code...")
        sample_code = '''def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)'''
        
        validation_result = await client.validate_code_via_cli(
            code=sample_code,
            validation_rules=["check syntax", "verify best practices", "look for improvements"],
            timeout=30
        )
        
        print(f"   ✅ Validation completed: {'Valid' if validation_result['valid'] else 'Issues found'}")
        print(f"   ✅ Execution time: {validation_result['execution_time']:.2f}s")
        
        if validation_result.get('issues'):
            print(f"   📋 Issues found ({len(validation_result['issues'])}):")
            for issue in validation_result['issues'][:3]:  # First 3 issues
                print(f"      • {issue}")
        
        if validation_result.get('suggestions'):
            print(f"   💡 Suggestions ({len(validation_result['suggestions'])}):")
            for suggestion in validation_result['suggestions'][:3]:  # First 3 suggestions
                print(f"      • {suggestion}")
        
        # Example 3: Refactor code
        print("\n5. Refactoring code...")
        refactor_result = await client.refactor_code_via_cli(
            code=sample_code,
            instructions="Add type hints and a proper docstring. Make it more robust with error handling.",
            preserve_comments=True,
            timeout=45
        )
        
        if refactor_result['success']:
            print("   ✅ Refactoring completed successfully")
            print(f"   ✅ Execution time: {refactor_result['execution_time']:.2f}s")
            
            if refactor_result.get('changes_made'):
                print(f"   🔄 Changes made ({len(refactor_result['changes_made'])}):")
                for change in refactor_result['changes_made'][:3]:
                    print(f"      • {change}")
            
            print("   📝 Refactored code (first 8 lines):")
            print("   " + "─" * 40)
            refactored_lines = refactor_result['refactored_code'].split('\n')[:8]
            for line in refactored_lines:
                print(f"   {line}")
            print("   " + "─" * 40)
        else:
            print(f"   ❌ Refactoring failed: {refactor_result.get('error', 'Unknown error')}")
        
        # Display final session statistics
        print("\n6. Session statistics...")
        final_session_info = client.get_session_info()
        print(f"   ✅ Total executions: {final_session_info['execution_count']}")
        print(f"   ✅ Session uptime: {final_session_info['session_uptime_seconds']:.1f}s")
        
        # Cleanup
        client.cleanup_session()
        print("   ✅ Session cleaned up")
    
    except Exception as e:
        print(f"   ❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demonstrate_factory_methods():
    """Demonstrate different ways to create the client."""
    
    print("\n🏭 Factory Methods Demo")
    print("=" * 50)
    
    # Method 1: Create with specific token file
    print("\n1. Create with specific token file...")
    try:
        client1 = ClaudeCodeDirectClient.create_with_token_file(
            token_file_path=".cc",
            claude_code_path="claude",
            working_directory="/home/tekkadmin/claude-tui"
        )
        
        session_info = client1.get_session_info()
        print(f"   ✅ Created with session: {session_info['session_id']}")
        print(f"   ✅ Token file: {session_info['oauth_token_file']}")
        
        client1.cleanup_session()
    
    except Exception as e:
        print(f"   ⚠️  Factory method 1 failed: {e}")
    
    # Method 2: Create with ConfigManager
    print("\n2. Create with ConfigManager...")
    try:
        config_manager = ConfigManager()
        client2 = ClaudeCodeDirectClient.create_from_config(
            config_manager=config_manager,
            claude_code_path="claude"
        )
        
        session_info = client2.get_session_info()
        print(f"   ✅ Created with session: {session_info['session_id']}")
        
        client2.cleanup_session()
    
    except Exception as e:
        print(f"   ⚠️  Factory method 2 failed: {e}")


async def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    
    print("\n🛡️ Error Handling Demo")
    print("=" * 50)
    
    # Test with invalid OAuth token file
    print("\n1. Testing with invalid token file...")
    try:
        client = ClaudeCodeDirectClient(
            oauth_token_file="/nonexistent/token.cc",
            claude_code_path="claude"
        )
        
        session_info = client.get_session_info()
        print(f"   ⚠️  Client created but no OAuth token: {session_info['oauth_token_available']}")
        
        client.cleanup_session()
    
    except Exception as e:
        print(f"   ✅ Properly handled invalid token file: {type(e).__name__}")
    
    # Test with invalid CLI path
    print("\n2. Testing with invalid CLI path...")
    try:
        client = ClaudeCodeDirectClient(
            claude_code_path="/nonexistent/claude-cli"
        )
    
    except Exception as e:
        print(f"   ✅ Properly handled invalid CLI path: {type(e).__name__}: {e}")


async def main():
    """Main demonstration function."""
    
    print("Claude Code Direct Client - Comprehensive Demo")
    print("=" * 60)
    print("This demo shows how to use ClaudeCodeDirectClient for direct")
    print("integration with Claude Code CLI using OAuth tokens.")
    print("=" * 60)
    
    # Run all demonstrations
    await demonstrate_claude_code_direct()
    await demonstrate_factory_methods()
    await demonstrate_error_handling()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\nKey Benefits of ClaudeCodeDirectClient:")
    print("• Works with OAuth tokens from .cc files")
    print("• No API endpoint restrictions") 
    print("• Full access to Claude Code CLI features")
    print("• Production-ready error handling")
    print("• Comprehensive logging and session management")
    print("• Async/await support for non-blocking operations")
    print("=" * 60)


if __name__ == "__main__":
    # Run the demonstration
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with exception: {e}")
        import traceback
        traceback.print_exc()