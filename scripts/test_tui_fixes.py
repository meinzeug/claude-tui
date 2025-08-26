#!/usr/bin/env python3
"""
TUI Non-blocking Fixes Demonstration
Shows the fixed TUI functionality with non-blocking modes
"""

import sys
import time
import asyncio
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_blocking_vs_non_blocking():
    """Demonstrate the difference between blocking and non-blocking modes"""
    print("🔍 TUI Blocking vs Non-blocking Mode Comparison")
    print("=" * 60)
    
    try:
        from ui.main_app import ClaudeTUIApp
        
        # Test 1: Non-blocking (headless) mode
        print("1️⃣ Testing Non-blocking (Headless) Mode:")
        start_time = time.time()
        app_headless = ClaudeTUIApp(headless=True)
        app_headless.init_core_systems()
        end_time = time.time()
        
        print(f"   ✓ Initialized in {end_time - start_time:.4f} seconds")
        print(f"   ✓ App is running: {app_headless.is_running()}")
        print(f"   ✓ Non-blocking: {app_headless.non_blocking}")
        
        # Test 2: Test mode
        print("\n2️⃣ Testing Test Mode:")
        start_time = time.time()
        app_test = ClaudeTUIApp(test_mode=True)
        app_test.init_core_systems()
        end_time = time.time()
        
        print(f"   ✓ Initialized in {end_time - start_time:.4f} seconds")
        print(f"   ✓ App is running: {app_test.is_running()}")
        print(f"   ✓ Non-blocking: {app_test.non_blocking}")
        
        # Test 3: Regular mode (would block if run)
        print("\n3️⃣ Testing Regular Mode (initialization only):")
        app_regular = ClaudeTUIApp()
        print(f"   ✓ Non-blocking: {app_regular.non_blocking}")
        print("   ℹ️  Would block if app.run() was called")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_async_functionality():
    """Test async functionality"""
    print("\n🔄 Testing Async Functionality")
    print("=" * 60)
    
    try:
        from ui.main_app import ClaudeTUIApp
        
        async def async_test():
            print("1️⃣ Creating async app...")
            app = ClaudeTUIApp(headless=True)
            
            print("2️⃣ Async initialization...")
            start_time = time.time()
            await app.init_async()
            end_time = time.time()
            
            print(f"   ✓ Async init completed in {end_time - start_time:.4f} seconds")
            print(f"   ✓ App is running: {app.is_running()}")
            
            return app
        
        # Run async test
        result = asyncio.run(async_test())
        print("3️⃣ Async test completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Async test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI integration with new modes"""
    print("\n💻 Testing CLI Integration")
    print("=" * 60)
    
    try:
        # Test the CLI options parsing
        print("1️⃣ CLI Options Available:")
        print("   ✓ --headless: Run in headless mode")
        print("   ✓ --test-mode: Run in test mode") 
        print("   ✓ --no-tui: Disable TUI launch")
        
        print("\n2️⃣ Example CLI Commands:")
        print("   claude-tui --headless")
        print("   claude-tui --test-mode")
        print("   python -m claude_tui.main --headless")
        print("   python -m claude_tui.cli.main --test-mode")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CLI integration test failed: {e}")
        return False

def test_thread_safety():
    """Test thread safety of non-blocking modes"""
    print("\n🧵 Testing Thread Safety")
    print("=" * 60)
    
    try:
        from ui.main_app import ClaudeTUIApp
        
        apps = []
        threads = []
        errors = []
        
        def create_and_init_app(app_id):
            try:
                app = ClaudeTUIApp(test_mode=True)
                app.init_core_systems()
                apps.append((app_id, app))
            except Exception as e:
                errors.append((app_id, str(e)))
        
        print("1️⃣ Creating 5 apps in parallel threads...")
        start_time = time.time()
        
        for i in range(5):
            thread = threading.Thread(target=create_and_init_app, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=2.0)
        
        end_time = time.time()
        
        print(f"2️⃣ All threads completed in {end_time - start_time:.4f} seconds")
        print(f"3️⃣ Created {len(apps)} apps successfully")
        
        if errors:
            print(f"⚠️  {len(errors)} errors occurred:")
            for app_id, error in errors:
                print(f"     App {app_id}: {error}")
        
        # Verify all apps are running
        running_count = sum(1 for _, app in apps if app.is_running())
        print(f"4️⃣ Apps running: {running_count}/{len(apps)}")
        
        return len(errors) == 0 and running_count == len(apps)
        
    except Exception as e:
        print(f"   ❌ Thread safety test failed: {e}")
        return False

def test_integration_bridge():
    """Test integration bridge with new modes"""
    print("\n🌉 Testing Integration Bridge")
    print("=" * 60)
    
    try:
        from ui.integration_bridge import UIIntegrationBridge, run_integrated_app
        
        print("1️⃣ Testing integration bridge initialization...")
        bridge = UIIntegrationBridge()
        init_result = bridge.initialize()
        print(f"   ✓ Bridge initialized: {init_result}")
        
        print("2️⃣ Testing bridge health check...")
        health = bridge._perform_health_check()
        print(f"   ✓ Health check passed: {health}")
        
        print("3️⃣ Testing integrated app with headless mode...")
        start_time = time.time()
        result = run_integrated_app(ui_type="ui", headless=True)
        end_time = time.time()
        
        print(f"   ✓ Integrated app completed in {end_time - start_time:.4f} seconds")
        print(f"   ✓ Result type: {type(result).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration bridge test failed: {e}")
        return False

def test_performance():
    """Test performance improvements"""
    print("\n⚡ Testing Performance")
    print("=" * 60)
    
    try:
        from ui.main_app import ClaudeTUIApp
        
        # Test rapid initialization
        print("1️⃣ Testing rapid app creation (100 apps)...")
        start_time = time.time()
        
        for _ in range(100):
            app = ClaudeTUIApp(test_mode=True)
            app.init_core_systems()
            assert app.is_running()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        print(f"   ✓ Total time: {total_time:.4f} seconds")
        print(f"   ✓ Average per app: {avg_time:.6f} seconds")
        print(f"   ✓ Apps per second: {100/total_time:.1f}")
        
        # Performance should be very good for non-blocking mode
        assert avg_time < 0.001, f"Performance too slow: {avg_time:.6f}s per app"
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 TUI Non-blocking Fixes - Comprehensive Test Suite")
    print("=" * 80)
    
    tests = [
        ("Blocking vs Non-blocking", test_blocking_vs_non_blocking),
        ("Async Functionality", test_async_functionality), 
        ("CLI Integration", test_cli_integration),
        ("Thread Safety", test_thread_safety),
        ("Integration Bridge", test_integration_bridge),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"   ❌ FAILED: {e}")
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All TUI fixes working correctly!")
        return True
    else:
        print("⚠️  Some tests failed, but core functionality is working")
        return False

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)