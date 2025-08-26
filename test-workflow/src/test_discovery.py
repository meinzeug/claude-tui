"""Test Discovery and File Watching - London School TDD

Supports test-driven workflow by:
1. Automatically finding test files
2. Loading and analyzing test functions
3. Supporting file watching for continuous testing (optional)
4. Filtering tests by patterns
"""

import os
import sys
import glob
import importlib
import importlib.util
import inspect
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass

# File watching is optional - gracefully handle if watchdog is not available
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None


@dataclass
class TestFunction:
    """Represents a discovered test function"""
    name: str
    fn: Callable
    file_path: str
    line_number: int
    is_async: bool = False
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        
        # Auto-detect if function is async
        self.is_async = asyncio.iscoroutinefunction(self.fn)
        
        # Extract description from docstring
        if self.fn.__doc__:
            self.description = self.fn.__doc__.strip().split('\n')[0]


class TestFileWatcher:
    """Watches test files for changes - only available if watchdog is installed"""
    
    def __init__(self, callback: Callable[[List[str]], None], patterns: List[str] = None):
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("File watching requires 'watchdog' package. Install with: pip install watchdog")
        
        self.callback = callback
        self.patterns = patterns or ['test_*.py', '*_test.py']
        self._last_triggered = 0
        self._debounce_delay = 0.5  # 500ms debounce
        
        # Create the actual event handler
        self._event_handler = self._create_handler()
    
    def _create_handler(self):
        """Create the actual file system event handler"""
        
        class Handler(FileSystemEventHandler):
            def __init__(self, watcher):
                self.watcher = watcher
            
            def on_modified(self, event):
                if event.is_directory:
                    return
                self.watcher._handle_file_event(event.src_path)
            
            def on_created(self, event):
                if event.is_directory:
                    return
                self.watcher._handle_file_event(event.src_path)
        
        return Handler(self)
    
    def _handle_file_event(self, file_path: str):
        """Handle file system events with debouncing"""
        current_time = time.time()
        
        # Debounce rapid file changes
        if current_time - self._last_triggered < self._debounce_delay:
            return
        
        # Check if file matches test patterns
        if self._is_test_file(file_path):
            self._last_triggered = current_time
            
            # Call callback with changed files
            if asyncio.iscoroutinefunction(self.callback):
                # Handle async callbacks
                loop = asyncio.get_event_loop()
                loop.create_task(self.callback([file_path]))
            else:
                self.callback([file_path])
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if file matches test patterns"""
        file_name = os.path.basename(file_path)
        
        for pattern in self.patterns:
            if self._matches_pattern(file_name, pattern):
                return True
        
        return False
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Simple glob pattern matching"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)


class TestDiscovery:
    """Discovers and loads test functions
    
    London School approach: focuses on coordinating with
    file system and module loading collaborators.
    """
    
    def __init__(self, file_system=None, watcher=None, root_path: str = "."):
        self.file_system = file_system or FileSystemInterface()
        self.watcher = watcher
        self.root_path = Path(root_path).resolve()
        self._observer = None
        self._watch_callback = None
        
        # Test discovery patterns
        self.test_file_patterns = [
            'test_*.py',
            '*_test.py',
            'tests/**/*.py'
        ]
        
        self.test_function_patterns = [
            'test_*',
            '*_test'
        ]
    
    def find_test_files(self, pattern: Optional[str] = None) -> List[str]:
        """Discover test files matching patterns
        
        Coordinates with file system to find test files.
        """
        test_files = []
        
        # Use custom pattern if provided
        patterns = [pattern] if pattern else self.test_file_patterns
        
        for pattern in patterns:
            # Use file system collaborator to find files
            matching_files = self.file_system.glob(pattern, root=self.root_path)
            
            # Filter for Python files
            python_files = [f for f in matching_files if f.endswith('.py')]
            test_files.extend(python_files)
        
        # Remove duplicates and sort
        test_files = list(set(test_files))
        test_files.sort()
        
        return test_files
    
    def load_tests(self, test_files: List[str]) -> List[Dict[str, Any]]:
        """Load test functions from files
        
        Coordinates with module loading to extract test functions.
        """
        all_tests = []
        
        for file_path in test_files:
            try:
                tests = self._load_tests_from_file(file_path)
                all_tests.extend(tests)
            except Exception as e:
                print(f"Warning: Could not load tests from {file_path}: {e}")
                continue
        
        return all_tests
    
    def _load_tests_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load test functions from a single file"""
        tests = []
        
        # Use file system collaborator to import module
        module = self.file_system.import_module(file_path)
        
        if not module:
            return tests
        
        # Find test functions in module
        for name, obj in inspect.getmembers(module):
            if self._is_test_function(name, obj):
                # Get source line number
                try:
                    line_number = inspect.getsourcelines(obj)[1]
                except (OSError, TypeError):
                    line_number = 0
                
                test_info = {
                    'name': name,
                    'fn': obj,
                    'file_path': file_path,
                    'line_number': line_number,
                    'async': asyncio.iscoroutinefunction(obj),
                    'description': self._get_function_description(obj),
                    'tags': self._extract_tags(obj)
                }
                
                tests.append(test_info)
        
        return tests
    
    def _is_test_function(self, name: str, obj: Any) -> bool:
        """Check if object is a test function"""
        if not callable(obj):
            return False
        
        if not inspect.isfunction(obj):
            return False
        
        # Check if name matches test patterns
        for pattern in self.test_function_patterns:
            if self._matches_pattern(name, pattern):
                return True
        
        return False
    
    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)
    
    def _get_function_description(self, fn: Callable) -> str:
        """Extract description from function docstring"""
        if fn.__doc__:
            return fn.__doc__.strip().split('\n')[0]
        return ""
    
    def _extract_tags(self, fn: Callable) -> List[str]:
        """Extract tags from function annotations or decorators"""
        tags = []
        
        # Check for @tag decorator or similar
        if hasattr(fn, '_test_tags'):
            tags.extend(fn._test_tags)
        
        # Check for async functions
        if asyncio.iscoroutinefunction(fn):
            tags.append('async')
        
        return tags
    
    def watch_files(self, callback: Callable[[List[str]], None]) -> None:
        """Start watching test files for changes
        
        Coordinates with file watcher to monitor changes.
        Only works if watchdog is available.
        """
        if not WATCHDOG_AVAILABLE:
            print("Warning: File watching not available. Install 'watchdog' package for file watching.")
            return
        
        if self._observer:
            self.stop_watching()
        
        self._watch_callback = callback
        
        # Create file watcher
        event_handler = TestFileWatcher(callback, self.test_file_patterns)._event_handler
        
        # Start watching
        self._observer = Observer()
        self._observer.schedule(event_handler, str(self.root_path), recursive=True)
        self._observer.start()
    
    def stop_watching(self) -> None:
        """Stop file watching"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
    
    def filter_tests(self, tests: List[Dict[str, Any]], 
                    pattern: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Filter tests by pattern or tags"""
        filtered_tests = tests
        
        # Filter by name pattern
        if pattern:
            filtered_tests = [
                test for test in filtered_tests 
                if pattern.lower() in test['name'].lower() or 
                   pattern.lower() in test['description'].lower()
            ]
        
        # Filter by tags
        if tags:
            filtered_tests = [
                test for test in filtered_tests
                if any(tag in test.get('tags', []) for tag in tags)
            ]
        
        return filtered_tests


class FileSystemInterface:
    """File system interface for test discovery
    
    Abstraction to allow mocking in tests.
    """
    
    def glob(self, pattern: str, root: Path = None) -> List[str]:
        """Find files matching glob pattern"""
        if root:
            # Change to root directory for globbing
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                matches = glob.glob(pattern, recursive=True)
                # Convert back to absolute paths
                matches = [str(root / match) for match in matches]
                return matches
            finally:
                os.chdir(old_cwd)
        else:
            return glob.glob(pattern, recursive=True)
    
    def import_module(self, file_path: str):
        """Import Python module from file path"""
        try:
            # Convert file path to module name
            file_path = Path(file_path)
            
            # Create module spec
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                return None
            
            # Import module
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to handle relative imports
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
                return module
            except Exception as e:
                # Clean up sys.modules on failure
                if module_name in sys.modules:
                    del sys.modules[module_name]
                raise e
                
        except Exception as e:
            print(f"Failed to import {file_path}: {e}")
            return None
    
    def exists(self, path: str) -> bool:
        """Check if path exists"""
        return os.path.exists(path)
    
    def is_file(self, path: str) -> bool:
        """Check if path is a file"""
        return os.path.isfile(path)
    
    def is_dir(self, path: str) -> bool:
        """Check if path is a directory"""
        return os.path.isdir(path)


# Convenience functions
def discover_tests(root_path: str = ".", pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """Quick function to discover all tests in a directory"""
    discovery = TestDiscovery(root_path=root_path)
    test_files = discovery.find_test_files(pattern)
    return discovery.load_tests(test_files)


def find_test_files(root_path: str = ".", pattern: Optional[str] = None) -> List[str]:
    """Quick function to find test files"""
    discovery = TestDiscovery(root_path=root_path)
    return discovery.find_test_files(pattern)


class TestCollector:
    """Collects and organizes discovered tests
    
    Provides higher-level organization of test discovery results.
    """
    
    def __init__(self, discovery: TestDiscovery):
        self.discovery = discovery
        self._cached_tests = None
        self._cache_time = 0
        self._cache_ttl = 30  # 30 second cache
    
    def collect_all_tests(self, use_cache: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Collect all tests organized by file"""
        current_time = time.time()
        
        if use_cache and self._cached_tests and (current_time - self._cache_time) < self._cache_ttl:
            return self._cached_tests
        
        # Discover fresh tests
        test_files = self.discovery.find_test_files()
        all_tests = self.discovery.load_tests(test_files)
        
        # Organize by file
        tests_by_file = {}
        for test in all_tests:
            file_path = test['file_path']
            if file_path not in tests_by_file:
                tests_by_file[file_path] = []
            tests_by_file[file_path].append(test)
        
        # Cache results
        self._cached_tests = tests_by_file
        self._cache_time = current_time
        
        return tests_by_file
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered tests"""
        tests_by_file = self.collect_all_tests()
        
        total_tests = sum(len(tests) for tests in tests_by_file.values())
        async_tests = 0
        files_with_tests = len(tests_by_file)
        
        for tests in tests_by_file.values():
            async_tests += sum(1 for test in tests if test['async'])
        
        return {
            'total_tests': total_tests,
            'async_tests': async_tests,
            'sync_tests': total_tests - async_tests,
            'files_with_tests': files_with_tests,
            'average_tests_per_file': total_tests / files_with_tests if files_with_tests > 0 else 0
        }
    
    def invalidate_cache(self):
        """Invalidate the test cache"""
        self._cached_tests = None
        self._cache_time = 0


# Test tagging decorator
def tag(*tags):
    """Decorator to tag test functions
    
    Usage:
        @tag('integration', 'database')
        def test_user_creation():
            pass
    """
    def decorator(fn):
        if not hasattr(fn, '_test_tags'):
            fn._test_tags = []
        fn._test_tags.extend(tags)
        return fn
    return decorator


# Skip decorator
def skip(reason=""):
    """Decorator to skip test functions
    
    Usage:
        @skip("Not implemented yet")
        def test_future_feature():
            pass
    """
    def decorator(fn):
        fn._test_skip = True
        fn._test_skip_reason = reason
        return fn
    return decorator


# Async test decorator (for explicit marking)
def async_test(fn):
    """Decorator to explicitly mark async tests
    
    Usage:
        @async_test
        async def test_async_operation():
            await some_async_call()
    """
    if not asyncio.iscoroutinefunction(fn):
        raise ValueError(f"@async_test can only be applied to async functions, got {fn.__name__}")
    
    fn._test_async = True
    return fn