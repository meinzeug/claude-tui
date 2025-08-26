#!/usr/bin/env python3
"""
Test script for the improved integration bridge
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.integration_bridge import get_bridge, reset_bridge


def test_bridge_functionality():
    """Test the bridge's key functionality"""
    print("🔧 Integration Bridge Test Suite")
    print("=" * 50)
    
    # Reset to get fresh instance
    reset_bridge()
    bridge = get_bridge()
    
    # Test initialization
    print("\n1️⃣ Testing initialization...")
    result = bridge.initialize(force_reinit=True)
    print(f"   Initialization: {'✅ SUCCESS' if result else '❌ FAILED'}")
    print(f"   Components: {len(bridge._available_components)}/4 available")
    print(f"   Available: {', '.join(sorted(bridge._available_components))}")
    
    if bridge._initialization_errors:
        print(f"   Errors: {len(bridge._initialization_errors)} encountered")
        for error in bridge._initialization_errors:
            print(f"     - {error['component']}: {error['error']}")
    
    # Test health check
    print("\n2️⃣ Testing health check...")
    health = bridge._perform_health_check()
    print(f"   Health check: {'✅ HEALTHY' if health else '❌ UNHEALTHY'}")
    
    # Test app instance creation
    print("\n3️⃣ Testing app instance creation...")
    
    ui_types = ['auto', 'ui', 'claude_tui']
    for ui_type in ui_types:
        print(f"   Testing {ui_type}...")
        try:
            app, used_type = bridge.get_app_instance(ui_type)
            print(f"     ✅ Created {used_type} app: {type(app).__name__}")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
    
    # Test error recovery
    print("\n4️⃣ Testing error recovery...")
    try:
        # Force an error by trying invalid UI type
        bridge.get_app_instance("invalid_ui_type")
    except ValueError as e:
        print(f"   ✅ Properly handled invalid UI type: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    print("\n🎉 Test suite complete!")
    print(f"📊 Bridge Status: {'READY FOR PRODUCTION' if result and health else 'NEEDS ATTENTION'}")


if __name__ == "__main__":
    test_bridge_functionality()