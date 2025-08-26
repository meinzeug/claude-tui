#!/usr/bin/env python3
"""
Error Recovery System Test Suite

This script tests the error handling and recovery capabilities of Claude-TUI.
It simulates various failure scenarios and validates that the system can
recover gracefully with appropriate fallbacks.

Usage:
    python scripts/test_error_recovery.py
    python scripts/test_error_recovery.py --component ai_interface
    python scripts/test_error_recovery.py --verbose
"""

import asyncio
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from core.error_handler import (
        get_error_handler, handle_errors, handle_async_errors, error_context,
        SystemHealthMonitor, emergency_recovery
    )
    from core.fallback_implementations import (
        MockAIInterface, InMemoryStorage, BasicProjectManager,
        BasicConfigManager, SafeFileOperations, BasicTaskEngine
    )
    from core.exceptions import (
        ClaudeTUIException, ValidationError, AIServiceError,
        NetworkError, FileSystemError, ConfigurationError
    )
    ERROR_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error system not available: {e}")
    ERROR_SYSTEM_AVAILABLE = False


class ErrorRecoveryTester:
    """Comprehensive test suite for error recovery system."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results = []
        self.error_handler = get_error_handler() if ERROR_SYSTEM_AVAILABLE else None
        
    def log(self, message: str, level: str = "INFO"):
        """Log test message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "FAIL"]:
            print(f"[{timestamp}] {level}: {message}")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
        """Run a single test and record results."""
        self.log(f"Running test: {test_name}")
        
        start_time = datetime.now()
        result = {
            'test_name': test_name,
            'start_time': start_time,
            'success': False,
            'error': None,
            'duration': 0,
            'details': {}
        }
        
        try:
            test_result = test_func(*args, **kwargs)
            result['success'] = True
            result['details'] = test_result if isinstance(test_result, dict) else {'result': test_result}
            self.log(f"✓ {test_name} - PASSED")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            self.log(f"✗ {test_name} - FAILED: {str(e)}", "ERROR")
            
        result['duration'] = (datetime.now() - start_time).total_seconds()
        self.test_results.append(result)
        
        return result
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped', 'reason': 'Error system not available'}
        
        error_handler = get_error_handler()
        assert error_handler is not None, "Error handler should be initialized"
        assert hasattr(error_handler, 'handle_error'), "Should have handle_error method"
        assert hasattr(error_handler, 'get_error_stats'), "Should have get_error_stats method"
        
        return {'status': 'passed', 'handler_type': type(error_handler).__name__}
    
    def test_structured_exception_handling(self):
        """Test structured exception creation and handling."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        # Test creating structured exception
        error = ValidationError(
            "Test validation error",
            field_name="test_field",
            field_value="invalid_value"
        )
        
        assert error.category.value == "validation"
        assert error.severity.value in ["low", "medium", "high", "critical"]
        assert error.error_id is not None
        assert "test_field" in str(error.metadata)
        
        return {
            'status': 'passed',
            'error_id': error.error_id,
            'category': error.category.value,
            'severity': error.severity.value
        }
    
    def test_automatic_error_recovery(self):
        """Test automatic error recovery mechanisms."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        # Simulate an error that should trigger recovery
        test_error = NetworkError("Connection failed")
        
        error_info = self.error_handler.handle_error(
            test_error,
            component='test_component',
            context={'test': True},
            auto_recover=True
        )
        
        assert error_info['error_id'] is not None
        assert error_info['component'] == 'test_component'
        assert 'recovery_attempted' in error_info
        
        return {
            'status': 'passed',
            'error_id': error_info['error_id'],
            'recovery_attempted': error_info.get('recovery_attempted', False),
            'recovery_successful': error_info.get('recovery_successful', False)
        }
    
    def test_fallback_ai_interface(self):
        """Test fallback AI interface functionality."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        mock_ai = MockAIInterface()
        
        # Test code generation
        result = asyncio.run(mock_ai.generate_code("Create a hello world function", "python"))
        assert result['success'] == True
        assert 'code' in result
        assert 'python' in result['code'].lower()
        
        # Test code review
        review_result = asyncio.run(mock_ai.review_code("def hello(): pass"))
        assert review_result['success'] == True
        assert 'issues' in review_result
        
        return {
            'status': 'passed',
            'code_generation': result['success'],
            'code_review': review_result['success'],
            'call_count': mock_ai.call_count
        }
    
    def test_fallback_storage(self):
        """Test fallback in-memory storage."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        storage = InMemoryStorage()
        
        # Test collection operations
        storage.create_collection('test_collection')
        assert 'test_collection' in storage.collections
        
        # Test record operations
        record_id = storage.insert('test_collection', {
            'name': 'test_record',
            'value': 42,
            'type': 'test'
        })
        assert record_id is not None
        
        # Test retrieval
        record = storage.find_by_id('test_collection', record_id)
        assert record is not None
        assert record['name'] == 'test_record'
        assert record['value'] == 42
        
        # Test search
        results = storage.find_by_field('test_collection', 'type', 'test')
        assert len(results) == 1
        assert results[0]['name'] == 'test_record'
        
        # Test update
        success = storage.update('test_collection', record_id, {'value': 100})
        assert success == True
        
        updated_record = storage.find_by_id('test_collection', record_id)
        assert updated_record['value'] == 100
        
        return {
            'status': 'passed',
            'collection_created': True,
            'record_created': record_id is not None,
            'record_retrieved': record is not None,
            'record_updated': success,
            'stats': storage.get_stats()
        }
    
    def test_fallback_project_manager(self):
        """Test fallback project manager."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_manager = BasicProjectManager()
            
            # Test project creation
            project_id = project_manager.create_project(
                name="Test Project",
                path=temp_dir,
                description="Test project for error recovery"
            )
            assert project_id is not None
            
            # Test project retrieval
            project = project_manager.get_project(project_id)
            assert project is not None
            assert project['name'] == "Test Project"
            assert project['path'] == temp_dir
            
            # Test project listing
            projects = project_manager.list_projects()
            assert len(projects) >= 1
            assert any(p['id'] == project_id for p in projects)
            
            # Test current project management
            success = project_manager.set_current_project(project_id)
            assert success == True
            
            current = project_manager.get_current_project()
            assert current is not None
            assert current['id'] == project_id
            
            return {
                'status': 'passed',
                'project_created': project_id,
                'project_retrieved': project is not None,
                'current_project_set': success
            }
    
    def test_fallback_config_manager(self):
        """Test fallback configuration manager."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        config = BasicConfigManager()
        
        # Test default configuration
        app_name = config.get('app.name')
        assert app_name == 'Claude-TUI'
        
        debug_mode = config.get('app.debug', False)
        assert isinstance(debug_mode, bool)
        
        # Test configuration setting
        success = config.set('test.value', 'test_data')
        assert success == True
        
        # Test configuration retrieval
        retrieved_value = config.get('test.value')
        assert retrieved_value == 'test_data'
        
        # Test section retrieval
        app_section = config.get_section('app')
        assert isinstance(app_section, dict)
        assert 'name' in app_section
        
        # Test offline mode
        offline_mode = config.is_offline_mode()
        assert isinstance(offline_mode, bool)
        
        return {
            'status': 'passed',
            'default_config_loaded': app_name is not None,
            'config_set': success,
            'config_retrieved': retrieved_value == 'test_data',
            'section_retrieved': isinstance(app_section, dict),
            'offline_mode': offline_mode
        }
    
    def test_safe_file_operations(self):
        """Test safe file operations with fallbacks."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_ops = SafeFileOperations(temp_dir)
            
            test_file = Path(temp_dir) / "test_file.txt"
            test_content = "This is test content\nWith multiple lines\nFor testing"
            
            # Test file writing
            success = file_ops.write_file(test_file, test_content)
            assert success == True
            assert test_file.exists()
            
            # Test file reading
            read_content = file_ops.read_file(test_file)
            assert read_content == test_content
            
            # Test directory listing
            files = file_ops.list_files(temp_dir, "*.txt")
            assert len(files) >= 1
            assert str(test_file) in files
            
            # Test directory creation
            new_dir = Path(temp_dir) / "subdir" / "nested"
            success = file_ops.create_directory(new_dir)
            assert success == True
            assert new_dir.exists()
            
            return {
                'status': 'passed',
                'file_written': test_file.exists(),
                'file_read': read_content == test_content,
                'files_listed': len(files),
                'directory_created': new_dir.exists()
            }
    
    def test_fallback_task_engine(self):
        """Test fallback task engine."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        task_engine = BasicTaskEngine()
        
        # Test task creation
        task_id = task_engine.create_task(
            name="Test Task",
            description="Test task for error recovery",
            priority="high"
        )
        assert task_id is not None
        
        # Test task retrieval
        task = task_engine.get_task(task_id)
        assert task is not None
        assert task['name'] == "Test Task"
        assert task['priority'] == "high"
        assert task['status'] == "pending"
        
        # Test task status update
        success = task_engine.update_task_status(task_id, "in_progress", 0.5)
        assert success == True
        
        updated_task = task_engine.get_task(task_id)
        assert updated_task['status'] == "in_progress"
        assert updated_task['progress'] == 0.5
        
        # Test task listing
        all_tasks = task_engine.list_tasks()
        assert len(all_tasks) >= 1
        assert any(t['id'] == task_id for t in all_tasks)
        
        # Test filtered task listing
        pending_tasks = task_engine.list_tasks(status="pending")
        in_progress_tasks = task_engine.list_tasks(status="in_progress")
        
        assert len(in_progress_tasks) >= 1
        
        return {
            'status': 'passed',
            'task_created': task_id,
            'task_retrieved': task is not None,
            'task_updated': success,
            'total_tasks': len(all_tasks),
            'in_progress_tasks': len(in_progress_tasks)
        }
    
    def test_error_decorators(self):
        """Test error handling decorators."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        # Test synchronous decorator
        @handle_errors(component='test_sync', auto_recover=True, silence_errors=True, fallback_return="fallback")
        def test_sync_function(should_fail: bool = False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test successful execution
        result = test_sync_function(False)
        assert result == "success"
        
        # Test error handling with fallback
        result = test_sync_function(True)
        assert result == "fallback"
        
        # Test async decorator
        @handle_async_errors(component='test_async', auto_recover=True, silence_errors=True, fallback_return="async_fallback")
        async def test_async_function(should_fail: bool = False):
            if should_fail:
                raise NetworkError("Test network error")
            return "async_success"
        
        # Test successful async execution
        result = asyncio.run(test_async_function(False))
        assert result == "async_success"
        
        # Test async error handling with fallback
        result = asyncio.run(test_async_function(True))
        assert result == "async_fallback"
        
        return {
            'status': 'passed',
            'sync_success': True,
            'sync_fallback': True,
            'async_success': True,
            'async_fallback': True
        }
    
    def test_health_monitoring(self):
        """Test system health monitoring."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        monitor = SystemHealthMonitor(self.error_handler)
        
        # Test health check
        health = monitor.check_system_health()
        
        assert 'timestamp' in health
        assert 'overall_status' in health
        assert health['overall_status'] in ['healthy', 'degraded', 'critical']
        assert 'components' in health
        assert 'error_stats' in health
        
        return {
            'status': 'passed',
            'overall_status': health['overall_status'],
            'components_checked': len(health['components']),
            'error_stats': health['error_stats']
        }
    
    def test_emergency_recovery(self):
        """Test emergency recovery procedures."""
        if not ERROR_SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        # Create some test errors first
        self.error_handler.handle_error(
            Exception("Test error 1"),
            component='test_component_1'
        )
        self.error_handler.handle_error(
            Exception("Test error 2"),
            component='test_component_2'
        )
        
        # Get stats before recovery
        stats_before = self.error_handler.get_error_stats()
        
        # Perform emergency recovery
        recovery_log = emergency_recovery()
        
        assert isinstance(recovery_log, list)
        assert len(recovery_log) > 0
        
        # Get stats after recovery
        stats_after = self.error_handler.get_error_stats()
        
        return {
            'status': 'passed',
            'recovery_steps': len(recovery_log),
            'errors_before': stats_before['total_errors'],
            'errors_after': stats_after['total_errors'],
            'recovery_log': recovery_log
        }
    
    def test_component_specific(self, component: str):
        """Test specific component error handling."""
        component_tests = {
            'ai_interface': self.test_fallback_ai_interface,
            'storage': self.test_fallback_storage,
            'project_manager': self.test_fallback_project_manager,
            'config_manager': self.test_fallback_config_manager,
            'file_operations': self.test_safe_file_operations,
            'task_engine': self.test_fallback_task_engine
        }
        
        if component in component_tests:
            return self.run_test(f"Component Test: {component}", component_tests[component])
        else:
            return {
                'status': 'error',
                'message': f"Unknown component: {component}",
                'available_components': list(component_tests.keys())
            }
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        self.log("Starting comprehensive error recovery test suite...")
        
        # Core system tests
        self.run_test("Error Handler Initialization", self.test_error_handler_initialization)
        self.run_test("Structured Exception Handling", self.test_structured_exception_handling)
        self.run_test("Automatic Error Recovery", self.test_automatic_error_recovery)
        
        # Fallback implementation tests
        self.run_test("Fallback AI Interface", self.test_fallback_ai_interface)
        self.run_test("Fallback Storage", self.test_fallback_storage)
        self.run_test("Fallback Project Manager", self.test_fallback_project_manager)
        self.run_test("Fallback Config Manager", self.test_fallback_config_manager)
        self.run_test("Safe File Operations", self.test_safe_file_operations)
        self.run_test("Fallback Task Engine", self.test_fallback_task_engine)
        
        # Advanced features
        self.run_test("Error Decorators", self.test_error_decorators)
        self.run_test("Health Monitoring", self.test_health_monitoring)
        self.run_test("Emergency Recovery", self.test_emergency_recovery)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['success']])
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r['duration'] for r in self.test_results)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration': total_duration
            },
            'test_results': self.test_results,
            'error_system_available': ERROR_SYSTEM_AVAILABLE,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report."""
        print("\n" + "="*80)
        print("ERROR RECOVERY SYSTEM TEST REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✓")
        print(f"Failed: {summary['failed']} ✗")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print(f"Error System Available: {'Yes' if report['error_system_available'] else 'No'}")
        
        if summary['failed'] > 0:
            print("\nFAILED TESTS:")
            print("-" * 40)
            for result in report['test_results']:
                if not result['success']:
                    print(f"  ✗ {result['test_name']}")
                    if result['error']:
                        print(f"    Error: {result['error']}")
        
        print("\nTEST DETAILS:")
        print("-" * 40)
        for result in report['test_results']:
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            duration = f"({result['duration']:.2f}s)"
            print(f"  {status} {result['test_name']} {duration}")
            
            if result['success'] and result['details'] and self.verbose:
                for key, value in result['details'].items():
                    if isinstance(value, dict):
                        print(f"    {key}: {len(value)} items")
                    else:
                        print(f"    {key}: {value}")
        
        print("\n" + "="*80)
        
        if summary['success_rate'] >= 0.8:
            print("🎉 Error recovery system is functioning well!")
        elif summary['success_rate'] >= 0.6:
            print("⚠️  Error recovery system has some issues that need attention.")
        else:
            print("❌ Error recovery system has significant issues that require immediate attention.")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Test Claude-TUI error recovery system')
    parser.add_argument('--component', help='Test specific component only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', help='Save report to file')
    
    args = parser.parse_args()
    
    tester = ErrorRecoveryTester(verbose=args.verbose)
    
    if args.component:
        # Test specific component
        result = tester.test_component_specific(args.component)
        report = {'test_results': [result], 'summary': {'total_tests': 1}}
    else:
        # Run all tests
        report = tester.run_all_tests()
    
    # Print report
    tester.print_report(report)
    
    # Save report if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    if report['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()