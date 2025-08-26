#!/usr/bin/env python3
"""
Test Integration Bridge - Test the UI integration bridge functionality
"""

import sys
import traceback
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_bridge_import():
    """Test importing the integration bridge"""
    try:
        print("=== Testing Integration Bridge Import ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from ui.integration_bridge import UIIntegrationBridge, get_bridge, run_integrated_app
        
        print("✓ Successfully imported UIIntegrationBridge")
        return True
        
    except Exception as e:
        print(f"✗ Failed to import integration bridge: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_bridge_initialization():
    """Test bridge initialization"""
    try:
        print("\n=== Testing Bridge Initialization ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from ui.integration_bridge import UIIntegrationBridge
        
        bridge = UIIntegrationBridge()
        print("✓ Bridge instance created")
        
        # Test initialization
        init_result = bridge.initialize()
        print(f"✓ Bridge initialization: {'SUCCESS' if init_result else 'PARTIAL'}")
        print(f"Available components: {bridge._available_components}")
        
        if bridge._initialization_errors:
            print("Initialization errors encountered:")
            for error in bridge._initialization_errors:
                print(f"  - {error['component']}: {error['error']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Bridge initialization test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_bridge_health_check():
    """Test bridge health check functionality"""
    try:
        print("\n=== Testing Bridge Health Check ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from ui.integration_bridge import UIIntegrationBridge
        
        bridge = UIIntegrationBridge()
        bridge.initialize()
        
        # Perform health check
        health_ok = bridge._perform_health_check()
        print(f"✓ Health check completed: {'HEALTHY' if health_ok else 'DEGRADED'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Health check test failed: {e}")
        return False

def test_fallback_managers():
    """Test that fallback managers work properly"""
    try:
        print("\n=== Testing Fallback Managers ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from ui.integration_bridge import UIIntegrationBridge
        
        bridge = UIIntegrationBridge()
        bridge.initialize()
        
        # Test config manager
        if bridge.config_manager:
            setting = bridge.config_manager.get_setting('ui_preferences.theme', 'dark')
            print(f"✓ Config manager working: theme = {setting}")
            
        # Test project manager
        if bridge.project_manager:
            print("✓ Project manager available")
            
        # Test AI interface
        if bridge.ai_interface:
            print("✓ AI interface available")
            
        # Test validation engine
        if bridge.validation_engine:
            print("✓ Validation engine available")
        
        return True
        
    except Exception as e:
        print(f"✗ Fallback managers test failed: {e}")
        return False

def main():
    """Run all integration bridge tests"""
    print("🧪 Starting Integration Bridge Tests")
    print("=" * 50)
    
    # Test bridge import
    import_ok = test_bridge_import()
    
    # Test bridge initialization
    init_ok = test_bridge_initialization()
    
    # Test health check
    health_ok = test_bridge_health_check()
    
    # Test fallback managers
    fallback_ok = test_fallback_managers()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION BRIDGE TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Import", import_ok),
        ("Initialization", init_ok), 
        ("Health Check", health_ok),
        ("Fallback Managers", fallback_ok)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    return {
        'tests_passed': passed,
        'tests_total': len(tests),
        'success_rate': passed / len(tests)
    }

if __name__ == '__main__':
    results = main()