"""
Test Context - Shared test context and data management for test-workflow
Integrates with test runner to provide consistent test environment
"""

from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
import json
import threading
from contextlib import contextmanager
import tempfile
import os


@dataclass
class ContextSnapshot:
    """Snapshot of context state for rollback"""
    name: str
    timestamp: float
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'data': self.data
        }


class TestContext:
    """
    Comprehensive test context that manages:
    - Shared test data
    - Test fixtures
    - Cleanup operations
    - Context inheritance
    - Thread-safe operations
    """
    
    def __init__(self, name: str):
        self.name = name
        self._data: Dict[str, Any] = {}
        self._fixtures: Dict[str, Any] = {}
        self._cleanup_operations: List[callable] = []
        self._snapshots: List[ContextSnapshot] = []
        self._lock = threading.RLock()
        self._temp_files: List[str] = []
        self._temp_dirs: List[str] = []
        self._child_contexts: List['TestContext'] = []
        self._parent_context: Optional['TestContext'] = None
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context"""
        with self._lock:
            self._data[key] = value
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context with inheritance"""
        with self._lock:
            if key in self._data:
                return self._data[key]
            elif self._parent_context:
                return self._parent_context.get(key, default)
            else:
                return default
                
    def has(self, key: str) -> bool:
        """Check if key exists in context or parent contexts"""
        with self._lock:
            if key in self._data:
                return True
            elif self._parent_context:
                return self._parent_context.has(key)
            else:
                return False
                
    def remove(self, key: str) -> Any:
        """Remove and return a value from context"""
        with self._lock:
            return self._data.pop(key, None)
            
    def update(self, data: Dict[str, Any]) -> None:
        """Update context with dictionary data"""
        with self._lock:
            self._data.update(data)
            
    def get_all(self) -> Dict[str, Any]:
        """Get all context data including inherited"""
        with self._lock:
            result = {}
            if self._parent_context:
                result.update(self._parent_context.get_all())
            result.update(self._data)
            return result
            
    def clear(self) -> None:
        """Clear local context data (not inherited)"""
        with self._lock:
            self._data.clear()
            
    # Context inheritance
    def inherit_from(self, parent: 'TestContext') -> None:
        """Inherit from parent context"""
        with self._lock:
            self._parent_context = parent
            parent._child_contexts.append(self)
            
    def create_child_context(self, name: str) -> 'TestContext':
        """Create a child context that inherits from this one"""
        child = TestContext(name)
        child.inherit_from(self)
        return child
        
    # Fixture management
    def add_fixture(self, name: str, fixture: Any) -> None:
        """Add a fixture to the context"""
        with self._lock:
            self._fixtures[name] = fixture
            
    def get_fixture(self, name: str) -> Any:
        """Get a fixture from context or parent contexts"""
        with self._lock:
            if name in self._fixtures:
                return self._fixtures[name]
            elif self._parent_context:
                return self._parent_context.get_fixture(name)
            else:
                raise KeyError(f"Fixture '{name}' not found")
                
    def has_fixture(self, name: str) -> bool:
        """Check if fixture exists"""
        with self._lock:
            if name in self._fixtures:
                return True
            elif self._parent_context:
                return self._parent_context.has_fixture(name)
            else:
                return False
                
    # Cleanup management
    def add_cleanup(self, cleanup_func: callable) -> None:
        """Add a cleanup operation"""
        with self._lock:
            self._cleanup_operations.append(cleanup_func)
            
    def cleanup(self) -> None:
        """Execute all cleanup operations"""
        with self._lock:
            errors = []
            
            # Clean up temporary files
            for temp_file in self._temp_files[:]:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    self._temp_files.remove(temp_file)
                except Exception as e:
                    errors.append(f"Failed to clean up temp file {temp_file}: {e}")
                    
            # Clean up temporary directories
            for temp_dir in self._temp_dirs[:]:
                try:
                    if os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir)
                    self._temp_dirs.remove(temp_dir)
                except Exception as e:
                    errors.append(f"Failed to clean up temp dir {temp_dir}: {e}")
                    
            # Execute custom cleanup operations
            for cleanup_op in self._cleanup_operations[:]:
                try:
                    cleanup_op()
                    self._cleanup_operations.remove(cleanup_op)
                except Exception as e:
                    errors.append(f"Cleanup operation failed: {e}")
                    
            # Clean up child contexts
            for child in self._child_contexts[:]:
                try:
                    child.cleanup()
                except Exception as e:
                    errors.append(f"Child context cleanup failed: {e}")
                    
            if errors:
                raise Exception("Cleanup errors occurred:\n" + "\n".join(errors))
                
    # Snapshot management
    def create_snapshot(self, name: str) -> str:
        """Create a snapshot of current context state"""
        import time
        
        with self._lock:
            snapshot = ContextSnapshot(
                name=name,
                timestamp=time.time(),
                data=self._data.copy()
            )
            self._snapshots.append(snapshot)
            return name
            
    def restore_snapshot(self, name: str) -> None:
        """Restore context to snapshot state"""
        with self._lock:
            snapshot = None
            for s in self._snapshots:
                if s.name == name:
                    snapshot = s
                    break
                    
            if not snapshot:
                raise ValueError(f"Snapshot '{name}' not found")
                
            self._data = snapshot.data.copy()
            
    def list_snapshots(self) -> List[str]:
        """List all snapshot names"""
        with self._lock:
            return [s.name for s in self._snapshots]
            
    # Temporary file management
    def create_temp_file(
        self,
        content: str = "",
        suffix: str = "",
        prefix: str = "test_",
        mode: str = "w"
    ) -> str:
        """Create a temporary file managed by context"""
        with tempfile.NamedTemporaryFile(
            mode=mode,
            suffix=suffix,
            prefix=prefix,
            delete=False
        ) as f:
            if content:
                f.write(content)
            temp_path = f.name
            
        with self._lock:
            self._temp_files.append(temp_path)
            
        return temp_path
        
    def create_temp_dir(self, prefix: str = "test_") -> str:
        """Create a temporary directory managed by context"""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        
        with self._lock:
            self._temp_dirs.append(temp_dir)
            
        return temp_dir
        
    # Context managers
    @contextmanager
    def temporary_set(self, key: str, value: Any):
        """Temporarily set a value, restore on exit"""
        original_value = self.get(key)
        original_exists = self.has(key)
        
        self.set(key, value)
        try:
            yield
        finally:
            if original_exists:
                self.set(key, original_value)
            else:
                self.remove(key)
                
    @contextmanager
    def snapshot_scope(self, name: str):
        """Create snapshot on enter, restore on exit"""
        self.create_snapshot(name)
        try:
            yield
        finally:
            self.restore_snapshot(name)
            
    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        with self._lock:
            return {
                'name': self.name,
                'data': self._serialize_data(self._data),
                'fixtures': list(self._fixtures.keys()),
                'cleanup_count': len(self._cleanup_operations),
                'snapshots': [s.to_dict() for s in self._snapshots],
                'temp_files_count': len(self._temp_files),
                'temp_dirs_count': len(self._temp_dirs),
                'child_contexts': [child.name for child in self._child_contexts],
                'parent_context': self._parent_context.name if self._parent_context else None
            }
            
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for JSON output"""
        result = {}
        for key, value in data.items():
            try:
                json.dumps(value)  # Test if serializable
                result[key] = value
            except (TypeError, ValueError):
                result[key] = f"<non-serializable: {type(value).__name__}>"
        return result
        
    def to_json(self) -> str:
        """Convert context to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
        
    # Helper methods for common patterns
    def setup_database(self, connection_string: str = "sqlite:///:memory:") -> Any:
        """Setup a test database"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            engine = create_engine(connection_string)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            self.add_fixture("db_engine", engine)
            self.add_fixture("db_session", session)
            self.add_cleanup(lambda: session.close())
            
            return session
        except ImportError:
            # Fallback to simple dict-based mock
            mock_db = {"data": {}}
            self.add_fixture("mock_db", mock_db)
            return mock_db
            
    def setup_http_server(self, port: int = 0) -> Dict[str, Any]:
        """Setup a test HTTP server"""
        import threading
        import http.server
        import socketserver
        from urllib.parse import urlparse
        
        # Create a simple HTTP server
        handler = http.server.SimpleHTTPRequestHandler
        
        with socketserver.TCPServer(("localhost", port), handler) as httpd:
            actual_port = httpd.server_address[1]
            
            # Start server in background thread
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            server_info = {
                "host": "localhost",
                "port": actual_port,
                "url": f"http://localhost:{actual_port}",
                "httpd": httpd,
                "thread": server_thread
            }
            
            self.add_fixture("http_server", server_info)
            self.add_cleanup(httpd.shutdown)
            
            return server_info
            
    def create_test_file_structure(self, structure: Dict[str, Union[str, dict]]) -> str:
        """Create a temporary file structure for testing"""
        base_dir = self.create_temp_dir()
        
        def create_structure(current_dir: str, items: Dict[str, Union[str, dict]]):
            for name, content in items.items():
                path = os.path.join(current_dir, name)
                
                if isinstance(content, dict):
                    os.makedirs(path, exist_ok=True)
                    create_structure(path, content)
                else:
                    with open(path, 'w') as f:
                        f.write(content)
                        
        create_structure(base_dir, structure)
        
        self.add_fixture("test_file_structure", base_dir)
        return base_dir
        
    # Integration with common testing patterns
    def mock_environment_variables(self, env_vars: Dict[str, str]) -> None:
        """Mock environment variables for the test"""
        import os
        
        original_env = {}
        
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
            
        def restore_env():
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
                    
        self.add_cleanup(restore_env)
        
    def capture_logging(self, logger_name: str = None) -> List[str]:
        """Capture logging output during test"""
        import logging
        import io
        
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()
            
        logger.addHandler(handler)
        self.add_cleanup(lambda: logger.removeHandler(handler))
        
        def get_logs():
            return log_capture.getvalue().split('\n')
            
        self.add_fixture("captured_logs", get_logs)
        return get_logs