#!/usr/bin/env python3
"""
Test Widget Imports - Comprehensive testing of all TUI widget imports
"""

import sys
import traceback
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_individual_widget_imports():
    """Test each widget module individually"""
    test_results = {}
    
    # Add src to path for imports
    sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
    
    widgets_to_test = [
        ('console_widget', 'ConsoleWidget'),
        ('notification_system', 'NotificationSystem'), 
        ('placeholder_alert', 'PlaceholderAlert'),
        ('progress_intelligence', 'ProgressIntelligence'),
        ('project_tree', 'ProjectTree'),
        ('task_dashboard', 'TaskDashboard'),
        ('workflow_visualizer', 'WorkflowVisualizerWidget'),
        ('metrics_dashboard', 'MetricsDashboardWidget'),
        ('modal_dialogs', 'ConfigurationModal'),
        ('advanced_components', None),  # Check if module exists
        ('enhanced_terminal_components', None),
        ('git_workflow_widget', None)
    ]
    
    for widget_module, main_class in widgets_to_test:
        try:
            print(f"\n=== Testing {widget_module} ===")
            
            # Try importing from ui.widgets
            module_path = f'ui.widgets.{widget_module}'
            module = __import__(module_path, fromlist=[main_class] if main_class else [''])
            
            if main_class:
                widget_class = getattr(module, main_class)
                print(f"✓ Successfully imported {main_class} from {module_path}")
                
                # Test class instantiation (if safe)
                try:
                    if hasattr(widget_class, '__init__'):
                        import inspect
                        sig = inspect.signature(widget_class.__init__)
                        params = list(sig.parameters.keys())[1:]  # Skip 'self'
                        
                        if not params or all(p.default != inspect.Parameter.empty for p in sig.parameters.values() if p.name != 'self'):
                            # Try creating instance only if no required params
                            instance = widget_class()
                            print(f"✓ Successfully instantiated {main_class}")
                        else:
                            print(f"✓ {main_class} requires parameters: {params}")
                except Exception as e:
                    print(f"! Instantiation failed for {main_class}: {e}")
            else:
                print(f"✓ Successfully imported module {module_path}")
                
            test_results[widget_module] = {'status': 'success', 'error': None}
            
        except Exception as e:
            print(f"✗ Failed to import {widget_module}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            test_results[widget_module] = {'status': 'failed', 'error': str(e)}
    
    return test_results

def test_widgets_init_import():
    """Test importing from widgets.__init__"""
    try:
        print("\n=== Testing widgets.__init__ import ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        # Test main imports from __init__.py
        from ui.widgets import (
            ConsoleWidget, NotificationSystem, PlaceholderAlert,
            ProgressIntelligence, ProjectTree, TaskDashboard
        )
        
        print("✓ Successfully imported main widgets from __init__.py")
        return True
        
    except Exception as e:
        print(f"✗ Failed to import from widgets.__init__: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_textual_dependency():
    """Test if Textual is properly installed and working"""
    try:
        print("\n=== Testing Textual Framework ===")
        
        from textual.app import App
        from textual.widget import Widget
        from rich.text import Text
        
        print("✓ Textual core classes imported successfully")
        
        # Test simple widget creation
        class TestWidget(Widget):
            def render(self):
                return Text("Test widget")
        
        widget = TestWidget()
        print("✓ Test widget created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Textual dependency test failed: {e}")
        return False

def main():
    """Run all widget tests"""
    print("🧪 Starting Widget Import Tests")
    print("=" * 50)
    
    # Test Textual first
    textual_ok = test_textual_dependency()
    
    # Test individual widgets
    widget_results = test_individual_widget_imports()
    
    # Test __init__ import
    init_ok = test_widgets_init_import()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Textual Framework: {'✓ OK' if textual_ok else '✗ FAILED'}")
    print(f"Widgets __init__.py: {'✓ OK' if init_ok else '✗ FAILED'}")
    
    print(f"\nIndividual Widget Results:")
    success_count = 0
    for widget, result in widget_results.items():
        status = "✓ OK" if result['status'] == 'success' else "✗ FAILED"
        print(f"  {widget}: {status}")
        if result['status'] == 'success':
            success_count += 1
    
    print(f"\nOverall: {success_count}/{len(widget_results)} widgets imported successfully")
    
    return {
        'textual_ok': textual_ok,
        'init_ok': init_ok,
        'widget_results': widget_results,
        'success_rate': success_count / len(widget_results) if widget_results else 0
    }

if __name__ == '__main__':
    results = main()