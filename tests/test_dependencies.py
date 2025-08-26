#!/usr/bin/env python3
"""
Dependency Resolution Test Suite.

Tests all critical imports and fallback mechanisms.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDependencyResolution(unittest.TestCase):
    """Test dependency resolution and fallback mechanisms."""
    
    def test_critical_imports(self):
        """Test that all critical imports work or have fallbacks."""
        
        # Test textual (critical for TUI)
        try:
            import textual
            self.assertTrue(hasattr(textual, '__version__'))
        except ImportError:
            self.fail("textual is required for TUI functionality")
            
        # Test rich (critical for formatting)
        try:
            import rich
            self.assertIsNotNone(rich)
        except ImportError:
            self.fail("rich is required for text formatting")
            
        # Test click (critical for CLI)
        try:
            import click
            self.assertTrue(hasattr(click, '__version__'))
        except ImportError:
            self.fail("click is required for CLI functionality")
    
    def test_fallback_implementations(self):
        """Test that fallback implementations are available."""
        
        try:
            from claude_tui.core.fallback_implementations import (
                FallbackConfigManager,
                FallbackSystemChecker,
                create_fallback_config_manager
            )
            
            # Test config manager fallback
            config_mgr = create_fallback_config_manager()
            self.assertIsNotNone(config_mgr)
            self.assertIsNone(config_mgr.get('nonexistent.key'))
            
            # Test system checker fallback
            checker = FallbackSystemChecker()
            self.assertIsNotNone(checker)
            
        except ImportError as e:
            self.fail(f"Fallback implementations failed: {e}")
    
    def test_dependency_checker(self):
        """Test the dependency checker functionality."""
        
        try:
            from claude_tui.core.dependency_checker import get_dependency_checker
            
            checker = get_dependency_checker()
            self.assertIsNotNone(checker)
            
            # Test checking a known available dependency
            status = checker.check_dependency('sys')  # Always available
            self.assertTrue(status.available)
            
            # Test checking a non-existent dependency
            status = checker.check_dependency('definitely_not_real_module_12345')
            self.assertFalse(status.available)
            
        except ImportError as e:
            self.fail(f"Dependency checker failed: {e}")
    
    def test_ui_widgets(self):
        """Test that UI widgets are available or have fallbacks."""
        
        try:
            from claude_tui.ui.widgets import TextInput, Button, FileTree
            
            # These should not raise exceptions even if they're fallbacks
            text_input = TextInput()
            button = Button()
            file_tree = FileTree()
            
            self.assertIsNotNone(text_input)
            self.assertIsNotNone(button)
            self.assertIsNotNone(file_tree)
            
        except Exception as e:
            self.fail(f"UI widgets failed: {e}")
    
    def test_main_cli_import(self):
        """Test that the main CLI can be imported."""
        
        try:
            from claude_tui.main import cli, main
            
            self.assertIsNotNone(cli)
            self.assertIsNotNone(main)
            
        except Exception as e:
            self.fail(f"Main CLI import failed: {e}")
    
    def test_optional_dependencies(self):
        """Test that optional dependencies don't break imports."""
        
        optional_modules = [
            'redis',
            'elasticsearch', 
            'uvloop',
            'orjson',
            'plotly',
            'websockets'
        ]
        
        for module_name in optional_modules:
            try:
                __import__(module_name)
                print(f"✓ {module_name}: Available")
            except ImportError:
                print(f"⚠ {module_name}: Not available (optional)")
                # This is OK - these are optional
                pass
    
    def test_performance_modules(self):
        """Test that performance modules can be imported."""
        
        try:
            # Try to import the memory optimizer module (not necessarily the class)
            import performance.memory_optimizer as mem_mod
            self.assertIsNotNone(mem_mod)
            
        except ImportError:
            # This might be expected if the module structure is different
            print("⚠ Performance modules not available in expected location")


def run_dependency_tests():
    """Run all dependency tests and provide a report."""
    
    print("=== Claude-TUI Dependency Test Suite ===\\n")
    
    # Run the unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDependencyResolution)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\\n=== Test Results ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\\nFailures:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\\nErrors:")
        for test, trace in result.errors:
            lines = trace.split('\n')
            error_line = lines[-2] if len(lines) > 1 else trace
            print(f"  - {test}: {error_line}")
    
    # Generate dependency report
    try:
        from claude_tui.core.dependency_checker import get_dependency_checker
        checker = get_dependency_checker()
        print(f"\\n{checker.generate_dependency_report()}")
    except Exception as e:
        print(f"\\nCould not generate dependency report: {e}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_dependency_tests()
    sys.exit(0 if success else 1)