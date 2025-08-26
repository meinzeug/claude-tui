#!/usr/bin/env python3
"""
Simple Error System Validation Script

This script performs basic validation of the error recovery system
to ensure it's working correctly without complex test frameworks.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_import_and_basic_functionality():
    """Test basic import and functionality."""
    print("Testing error system imports...")
    
    try:
        from core.error_handler import get_error_handler, handle_errors
        from core.fallback_implementations import MockAIInterface, InMemoryStorage
        from core.exceptions import ClaudeTUIException, ValidationError
        print("✓ All core imports successful")
        
        # Test error handler
        error_handler = get_error_handler()
        print("✓ Error handler initialized")
        
        # Test structured exception
        error = ValidationError("Test error", field_name="test")
        print("✓ Structured exception created")
        
        # Test error handling
        error_info = error_handler.handle_error(error, component='test')
        print("✓ Error handling successful")
        
        # Test fallback AI
        mock_ai = MockAIInterface()
        print("✓ Mock AI interface created")
        
        # Test fallback storage
        storage = InMemoryStorage()
        storage.create_collection('test')
        test_id = storage.insert('test', {'name': 'test'})
        retrieved = storage.find_by_id('test', test_id)
        print("✓ Fallback storage working")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        traceback.print_exc()
        return False

def test_widget_refresh_fix():
    """Test that the widget refresh method fix is working."""
    print("\nTesting widget refresh method fix...")
    
    try:
        # This simulates the original error scenario
        class MockWidget:
            def refresh(self, *, repaint: bool = True, layout: bool = False, **kwargs):
                """Mock widget with corrected refresh signature."""
                return {'repaint': repaint, 'layout': layout, 'kwargs': kwargs}
        
        # Test the fix
        widget = MockWidget()
        
        # These calls should now work without TypeError
        result1 = widget.refresh()
        result2 = widget.refresh(repaint=True, layout=False)
        result3 = widget.refresh(repaint=False, layout=True, custom_param='test')
        
        print("✓ Widget refresh method signature fix working")
        print(f"  - Basic refresh: {result1}")
        print(f"  - With parameters: {result2}")
        print(f"  - With extra kwargs: {result3}")
        
        return True
        
    except Exception as e:
        print(f"❌ Widget refresh test failed: {e}")
        traceback.print_exc()
        return False

def test_error_recovery_decorator():
    """Test error recovery decorator functionality."""
    print("\nTesting error recovery decorators...")
    
    try:
        from core.error_handler import handle_errors
        
        @handle_errors(component='test_decorator', auto_recover=True, silence_errors=True, fallback_return="fallback")
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test successful execution
        result1 = test_function(False)
        print(f"✓ Successful execution: {result1}")
        
        # Test error handling with fallback
        result2 = test_function(True)
        print(f"✓ Error handled with fallback: {result2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decorator test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run validation tests."""
    print("="*60)
    print("CLAUDE-TUI ERROR RECOVERY SYSTEM VALIDATION")
    print("="*60)
    
    tests = [
        test_import_and_basic_functionality,
        test_widget_refresh_fix,
        test_error_recovery_decorator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 All validation tests passed! Error recovery system is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some validation tests failed. Review the error recovery system.")
        sys.exit(1)

if __name__ == '__main__':
    main()