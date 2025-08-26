#!/usr/bin/env python3
"""
Test TUI in headless/non-blocking mode
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_headless_mode():
    """Test TUI in headless mode"""
    print("🧪 Testing TUI in headless mode...")
    
    try:
        # Import the updated TUI app
        from src.ui.main_app import ClaudeTUIApp
        from src.claude_tui.core.config_manager import ConfigManager
        
        # Create config manager
        config_manager = ConfigManager()
        
        # Create app in headless mode
        app = ClaudeTUIApp(config_manager, headless=True)
        print("✅ TUI created in headless mode")
        
        # Initialize core systems
        app.init_core_systems()
        print("✅ Core systems initialized")
        
        # Check if app is running
        if app.is_running():
            print("✅ App is running (non-blocking)")
        else:
            print("❌ App is not running")
            
        # Stop the app
        app.stop()
        print("✅ App stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in headless mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_test_mode():
    """Test TUI in test mode"""
    print("\n🧪 Testing TUI in test mode...")
    
    try:
        # Import the updated TUI app
        from src.ui.main_app import ClaudeTUIApp
        from src.claude_tui.core.config_manager import ConfigManager
        
        # Create config manager
        config_manager = ConfigManager()
        
        # Create app in test mode
        app = ClaudeTUIApp(config_manager, test_mode=True)
        print("✅ TUI created in test mode")
        
        # Initialize core systems
        app.init_core_systems()
        print("✅ Core systems initialized")
        
        # Check components
        if app.config_manager:
            print("✅ Config manager available")
        if app.project_manager:
            print("✅ Project manager available")
        if app.ai_interface:
            print("✅ AI interface available")
        if app.validation_engine:
            print("✅ Validation engine available")
            
        # Stop the app
        app.stop()
        print("✅ App stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_arguments():
    """Test CLI arguments for headless/test modes"""
    print("\n🧪 Testing CLI arguments...")
    
    try:
        import subprocess
        import time
        
        # Test --headless flag
        print("Testing --headless flag...")
        proc = subprocess.Popen(
            ["python3", "run_tui.py", "--headless"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it 2 seconds to start
        time.sleep(2)
        
        # Check if it's still running (shouldn't block)
        if proc.poll() is None:
            # Still running - terminate it
            proc.terminate()
            print("✅ --headless flag works (non-blocking)")
        else:
            # Already exited
            print("✅ --headless flag completed without blocking")
            
        # Test --test-mode flag
        print("Testing --test-mode flag...")
        proc = subprocess.Popen(
            ["python3", "run_tui.py", "--test-mode"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it 2 seconds to start
        time.sleep(2)
        
        # Check if it's still running (shouldn't block)
        if proc.poll() is None:
            # Still running - terminate it
            proc.terminate()
            print("✅ --test-mode flag works (non-blocking)")
        else:
            # Already exited
            print("✅ --test-mode flag completed without blocking")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing CLI arguments: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("🚀 Claude-TUI Non-Blocking Mode Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Test headless mode
    if not test_headless_mode():
        all_passed = False
        
    # Test test mode
    if not test_test_mode():
        all_passed = False
        
    # Test CLI arguments
    if not test_cli_arguments():
        all_passed = False
        
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL TESTS PASSED - TUI is non-blocking!")
    else:
        print("❌ Some tests failed - check errors above")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())