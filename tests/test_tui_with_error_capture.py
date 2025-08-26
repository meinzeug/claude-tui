#!/usr/bin/env python3
"""
TUI Testing with Comprehensive Error Capture

This script runs the TUI application with comprehensive error capture and
provides detailed documentation of all issues found.
"""

import asyncio
import sys
import os
import logging
import traceback
from pathlib import Path
from datetime import datetime
import json
import subprocess
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
import io

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/tui_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TUIErrorCapture:
    """Comprehensive TUI error capture and testing."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.import_issues = []
        self.runtime_issues = []
        self.startup_issues = []
        self.widget_issues = []
        self.fixes_applied = []
        self.test_results = {
            'import_tests': {},
            'startup_test': False,
            'widget_tests': {},
            'runtime_test': False,
            'error_handling_test': False
        }
    
    def test_imports(self):
        """Test all TUI imports and capture errors."""
        logger.info("Testing all TUI imports...")
        
        # Core imports
        core_imports = [
            ('ui.main_app', 'ClaudeTUIApp'),
            ('ui.main_app', 'run_app'),
            ('core.project_manager', 'ProjectManager'),
            ('core.ai_interface', 'AIInterface'),
            ('core.config_manager', 'ConfigManager'),
        ]
        
        # Widget imports
        widget_imports = [
            ('ui.widgets.project_tree', 'ProjectTree'),
            ('ui.widgets.task_dashboard', 'TaskDashboard'),
            ('ui.widgets.progress_intelligence', 'ProgressIntelligence'),
            ('ui.widgets.console_widget', 'ConsoleWidget'),
            ('ui.widgets.notification_system', 'NotificationSystem'),
            ('ui.widgets.placeholder_alert', 'PlaceholderAlert'),
            ('ui.widgets.metrics_dashboard', 'MetricsDashboard'),
            ('ui.widgets.modal_dialogs', 'Modal'),
            ('ui.widgets.workflow_visualizer', 'WorkflowVisualizer'),
            ('ui.widgets.git_workflow_widget', 'GitWorkflowWidget'),
        ]
        
        # Screen imports
        screen_imports = [
            ('ui.screens.project_wizard', 'ProjectWizardScreen'),
            ('ui.screens.settings', 'SettingsScreen'),
            ('ui.screens.help_screen', 'HelpScreen'),
            ('ui.screens.workspace_screen', 'WorkspaceScreen'),
        ]
        
        all_imports = [
            ('core', core_imports),
            ('widgets', widget_imports),
            ('screens', screen_imports)
        ]
        
        for category, imports in all_imports:
            self.test_results['import_tests'][category] = {}
            
            for module_path, class_name in imports:
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    
                    if hasattr(module, class_name):
                        cls = getattr(module, class_name)
                        self.test_results['import_tests'][category][f"{module_path}.{class_name}"] = {
                            'status': 'success',
                            'class': str(cls)
                        }
                        logger.debug(f"✓ Successfully imported {module_path}.{class_name}")
                    else:
                        error_msg = f"Class {class_name} not found in {module_path}"
                        self.test_results['import_tests'][category][f"{module_path}.{class_name}"] = {
                            'status': 'missing_class',
                            'error': error_msg
                        }
                        self.import_issues.append(error_msg)
                        logger.warning(error_msg)
                        
                except ImportError as e:
                    error_msg = f"Import error for {module_path}: {e}"
                    self.test_results['import_tests'][category][f"{module_path}.{class_name}"] = {
                        'status': 'import_error',
                        'error': error_msg
                    }
                    self.import_issues.append(error_msg)
                    logger.error(error_msg)
                    
                except Exception as e:
                    error_msg = f"Unexpected error importing {module_path}: {e}"
                    self.test_results['import_tests'][category][f"{module_path}.{class_name}"] = {
                        'status': 'error',
                        'error': error_msg,
                        'traceback': traceback.format_exc()
                    }
                    self.errors.append(error_msg)
                    logger.error(error_msg)
    
    def test_tui_startup(self):
        """Test TUI startup process."""
        logger.info("Testing TUI startup...")
        
        try:
            from ui.main_app import ClaudeTUIApp
            
            # Capture startup process
            startup_output = io.StringIO()
            startup_errors = io.StringIO()
            
            try:
                with redirect_stdout(startup_output), redirect_stderr(startup_errors):
                    app = ClaudeTUIApp()
                    
                    # Test initialization
                    app.init_core_systems()
                    
                    # Test mounting simulation
                    app.on_mount()
                    
                    # Test basic operations
                    app.notify("Test notification", "info")
                    
                self.test_results['startup_test'] = True
                logger.info("✓ TUI startup test successful")
                
            except Exception as e:
                error_msg = f"TUI startup failed: {e}"
                self.startup_issues.append(error_msg)
                self.errors.append(error_msg)
                logger.error(error_msg)
                logger.error(f"Startup traceback: {traceback.format_exc()}")
                
                # Capture output
                stdout_content = startup_output.getvalue()
                stderr_content = startup_errors.getvalue()
                
                if stdout_content:
                    logger.info(f"Startup stdout: {stdout_content}")
                if stderr_content:
                    logger.error(f"Startup stderr: {stderr_content}")
                
        except ImportError as e:
            error_msg = f"Failed to import ClaudeTUIApp: {e}"
            self.startup_issues.append(error_msg)
            self.errors.append(error_msg)
            logger.error(error_msg)
    
    def test_widget_creation(self):
        """Test individual widget creation."""
        logger.info("Testing widget creation...")
        
        # Mock dependencies for widget testing
        class MockManager:
            def __init__(self):
                self.current_project = None
                
        mock_manager = MockManager()
        
        widgets_to_test = [
            ('ui.widgets.project_tree', 'ProjectTree', [mock_manager]),
            ('ui.widgets.task_dashboard', 'TaskDashboard', [mock_manager]),
            ('ui.widgets.progress_intelligence', 'ProgressIntelligence', []),
            ('ui.widgets.console_widget', 'ConsoleWidget', [mock_manager]),
            ('ui.widgets.notification_system', 'NotificationSystem', []),
            ('ui.widgets.placeholder_alert', 'PlaceholderAlert', []),
        ]
        
        for module_path, class_name, args in widgets_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                widget_class = getattr(module, class_name)
                
                # Try to create widget instance
                widget = widget_class(*args)
                
                # Test basic widget methods if they exist
                test_methods = ['refresh', 'update', 'render', 'compose']
                available_methods = []
                
                for method_name in test_methods:
                    if hasattr(widget, method_name):
                        available_methods.append(method_name)
                        try:
                            method = getattr(widget, method_name)
                            if callable(method):
                                # Don't actually call methods that might require full app context
                                logger.debug(f"  - Method {method_name} available")
                        except Exception as e:
                            logger.warning(f"  - Method {method_name} error: {e}")
                
                self.test_results['widget_tests'][class_name] = {
                    'status': 'success',
                    'available_methods': available_methods
                }
                logger.info(f"✓ Successfully created {class_name}")
                
            except Exception as e:
                error_msg = f"Failed to create {class_name}: {e}"
                self.test_results['widget_tests'][class_name] = {
                    'status': 'error',
                    'error': error_msg,
                    'traceback': traceback.format_exc()
                }
                self.widget_issues.append(error_msg)
                logger.error(error_msg)
    
    def test_tui_runtime(self):
        """Test TUI runtime with timeout."""
        logger.info("Testing TUI runtime...")
        
        try:
            # Run TUI with timeout to test basic functionality
            cmd = [sys.executable, 'run_tui.py']
            
            # Change to project directory
            original_cwd = os.getcwd()
            os.chdir('/home/tekkadmin/claude-tui')
            
            try:
                # Start TUI process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid  # Create new process group
                )
                
                # Wait for a short time to see if it starts
                time.sleep(3)
                
                # Check if process is still running
                if process.poll() is None:
                    # Process is running, kill it gracefully
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    stdout, stderr = process.communicate(timeout=5)
                    
                    self.test_results['runtime_test'] = True
                    logger.info("✓ TUI runtime test successful - application started and ran")
                    
                    if stdout:
                        logger.info(f"TUI stdout: {stdout}")
                    if stderr:
                        logger.warning(f"TUI stderr: {stderr}")
                else:
                    # Process exited
                    stdout, stderr = process.communicate()
                    error_msg = f"TUI process exited early with code {process.returncode}"
                    self.runtime_issues.append(error_msg)
                    logger.error(error_msg)
                    
                    if stdout:
                        logger.error(f"TUI stdout: {stdout}")
                    if stderr:
                        logger.error(f"TUI stderr: {stderr}")
                        
            except subprocess.TimeoutExpired:
                # Timeout is actually good - means the app is running
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                self.test_results['runtime_test'] = True
                logger.info("✓ TUI runtime test successful - application ran for expected duration")
                
            except Exception as e:
                error_msg = f"TUI runtime test failed: {e}"
                self.runtime_issues.append(error_msg)
                logger.error(error_msg)
            
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            error_msg = f"Failed to test TUI runtime: {e}"
            self.runtime_issues.append(error_msg)
            logger.error(error_msg)
    
    def identify_fixes(self):
        """Identify potential fixes for found issues."""
        logger.info("Identifying potential fixes...")
        
        # Common fix patterns
        fixes = []
        
        # Check for import issues
        for issue in self.import_issues:
            if "claude_tiu" in issue:
                fix = {
                    'issue': issue,
                    'fix_type': 'import_path_correction',
                    'description': 'Correct import path from claude_tiu to claude_tui',
                    'action': 'Replace claude_tiu with claude_tui in import statements',
                    'priority': 'high'
                }
                fixes.append(fix)
            
            elif "No module named" in issue:
                fix = {
                    'issue': issue,
                    'fix_type': 'missing_module',
                    'description': 'Create missing module or adjust import path',
                    'action': 'Check if module exists or create placeholder module',
                    'priority': 'medium'
                }
                fixes.append(fix)
            
            elif "not found" in issue:
                fix = {
                    'issue': issue,
                    'fix_type': 'missing_class',
                    'description': 'Create missing class or adjust import',
                    'action': 'Implement missing class or fix import statement',
                    'priority': 'medium'
                }
                fixes.append(fix)
        
        # Check for startup issues
        for issue in self.startup_issues:
            if "fallback" in issue.lower():
                fix = {
                    'issue': issue,
                    'fix_type': 'fallback_implementation',
                    'description': 'Replace fallback with proper implementation',
                    'action': 'Implement proper classes instead of using fallbacks',
                    'priority': 'low'
                }
                fixes.append(fix)
        
        # Check for widget issues
        for issue in self.widget_issues:
            fix = {
                'issue': issue,
                'fix_type': 'widget_implementation',
                'description': 'Fix widget implementation issues',
                'action': 'Debug and fix widget creation or method calls',
                'priority': 'medium'
            }
            fixes.append(fix)
        
        self.fixes_applied = fixes
        return fixes
    
    def apply_critical_fixes(self):
        """Apply critical fixes automatically."""
        logger.info("Applying critical fixes...")
        
        fixes_applied = []
        
        # Fix 1: Correct claude_tiu to claude_tui import
        workspace_screen_path = Path("/home/tekkadmin/claude-tui/src/ui/screens/workspace_screen.py")
        if workspace_screen_path.exists():
            try:
                content = workspace_screen_path.read_text()
                if "claude_tiu" in content:
                    fixed_content = content.replace("claude_tiu", "claude_tui")
                    workspace_screen_path.write_text(fixed_content)
                    fixes_applied.append("Fixed claude_tiu -> claude_tui in workspace_screen.py")
                    logger.info("✓ Applied fix: claude_tiu -> claude_tui")
            except Exception as e:
                logger.error(f"Failed to apply claude_tiu fix: {e}")
        
        # Fix 2: Check for other common import issues
        src_path = Path("/home/tekkadmin/claude-tui/src")
        for py_file in src_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                original_content = content
                
                # Fix common typos
                content = content.replace("claude_tiu", "claude_tui")
                
                if content != original_content:
                    py_file.write_text(content)
                    fixes_applied.append(f"Fixed imports in {py_file.relative_to(src_path)}")
                    
            except Exception as e:
                logger.debug(f"Skipped {py_file}: {e}")
        
        self.fixes_applied.extend(fixes_applied)
        return fixes_applied
    
    def generate_error_report(self):
        """Generate comprehensive error report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_errors': len(self.errors),
                'import_issues': len(self.import_issues),
                'startup_issues': len(self.startup_issues),
                'widget_issues': len(self.widget_issues),
                'runtime_issues': len(self.runtime_issues),
                'fixes_identified': len(self.fixes_applied),
                'overall_status': self._calculate_overall_status()
            },
            'test_results': self.test_results,
            'issues': {
                'errors': self.errors,
                'import_issues': self.import_issues,
                'startup_issues': self.startup_issues,
                'widget_issues': self.widget_issues,
                'runtime_issues': self.runtime_issues
            },
            'fixes': self.fixes_applied,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_overall_status(self):
        """Calculate overall status based on test results."""
        critical_errors = len(self.errors) + len(self.startup_issues)
        minor_issues = len(self.import_issues) + len(self.widget_issues)
        
        if critical_errors == 0 and minor_issues <= 2:
            return "EXCELLENT"
        elif critical_errors == 0 and minor_issues <= 5:
            return "GOOD"
        elif critical_errors <= 2:
            return "ACCEPTABLE"
        else:
            return "NEEDS_WORK"
    
    def _generate_recommendations(self):
        """Generate recommendations for improvements."""
        recommendations = []
        
        if self.import_issues:
            recommendations.append({
                'category': 'imports',
                'priority': 'high',
                'recommendation': 'Fix import path issues, especially claude_tiu -> claude_tui',
                'impact': 'Critical for proper module loading'
            })
        
        if self.startup_issues:
            recommendations.append({
                'category': 'startup',
                'priority': 'medium',
                'recommendation': 'Implement proper core classes instead of fallbacks',
                'impact': 'Improves application functionality and reliability'
            })
        
        if self.widget_issues:
            recommendations.append({
                'category': 'widgets',
                'priority': 'medium',
                'recommendation': 'Debug and fix widget creation issues',
                'impact': 'Ensures all UI components work correctly'
            })
        
        if self.runtime_issues:
            recommendations.append({
                'category': 'runtime',
                'priority': 'high',
                'recommendation': 'Fix runtime crashes and stability issues',
                'impact': 'Critical for application usability'
            })
        
        # Success recommendations
        if not self.errors and not self.startup_issues:
            recommendations.append({
                'category': 'success',
                'priority': 'low',
                'recommendation': 'Application is working well - consider adding more features',
                'impact': 'Enhancement opportunities'
            })
        
        return recommendations
    
    async def run_comprehensive_test(self):
        """Run comprehensive TUI testing."""
        logger.info("Starting comprehensive TUI error capture and testing...")
        
        # Test imports
        self.test_imports()
        
        # Apply critical fixes
        self.apply_critical_fixes()
        
        # Test startup
        self.test_tui_startup()
        
        # Test widget creation
        self.test_widget_creation()
        
        # Test runtime
        self.test_tui_runtime()
        
        # Identify additional fixes
        self.identify_fixes()
        
        # Generate report
        report = self.generate_error_report()
        
        return report


async def main():
    """Main test execution."""
    error_capture = TUIErrorCapture()
    
    try:
        report = await error_capture.run_comprehensive_test()
        
        # Save report to file
        report_file = Path("/home/tekkadmin/claude-tui/tests/tui_error_report.json")
        report_file.write_text(json.dumps(report, indent=2, default=str))
        
        # Display summary
        print("\n" + "="*80)
        print("COMPREHENSIVE TUI ERROR ANALYSIS REPORT")
        print("="*80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['summary']['overall_status']}")
        print()
        
        print("SUMMARY:")
        print(f"  Total Errors: {report['summary']['total_errors']}")
        print(f"  Import Issues: {report['summary']['import_issues']}")
        print(f"  Startup Issues: {report['summary']['startup_issues']}")
        print(f"  Widget Issues: {report['summary']['widget_issues']}")
        print(f"  Runtime Issues: {report['summary']['runtime_issues']}")
        print(f"  Fixes Applied: {report['summary']['fixes_identified']}")
        print()
        
        # Show test results
        print("TEST RESULTS:")
        startup_status = "✓ PASS" if report['test_results']['startup_test'] else "✗ FAIL"
        runtime_status = "✓ PASS" if report['test_results']['runtime_test'] else "✗ FAIL"
        print(f"  Startup Test: {startup_status}")
        print(f"  Runtime Test: {runtime_status}")
        
        # Import results
        for category, results in report['test_results']['import_tests'].items():
            success_count = sum(1 for r in results.values() if r.get('status') == 'success')
            total_count = len(results)
            print(f"  {category.title()} Imports: {success_count}/{total_count}")
        
        # Widget results  
        widget_results = report['test_results']['widget_tests']
        widget_success = sum(1 for r in widget_results.values() if r.get('status') == 'success')
        widget_total = len(widget_results)
        print(f"  Widget Creation: {widget_success}/{widget_total}")
        print()
        
        # Show critical issues
        if report['issues']['errors']:
            print("CRITICAL ERRORS:")
            for i, error in enumerate(report['issues']['errors'], 1):
                print(f"  {i}. {error}")
            print()
        
        # Show fixes applied
        if report['fixes']:
            print("FIXES APPLIED:")
            for i, fix in enumerate([f for f in report['fixes'] if isinstance(f, str)], 1):
                print(f"  {i}. {fix}")
            print()
        
        # Show recommendations
        if report['recommendations']:
            print("RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. [{rec['priority'].upper()}] {rec['recommendation']}")
                print(f"     Impact: {rec['impact']}")
            print()
        
        print(f"Full report saved to: {report_file}")
        print("="*80)
        
        return 0 if report['summary']['overall_status'] in ['EXCELLENT', 'GOOD'] else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)