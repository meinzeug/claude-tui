"""
Mock implementations for external dependencies.

Provides reusable mocks for testing services without external dependencies.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List, Optional
from pathlib import Path


class MockAIInterface:
    """Mock AI interface for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_request = None
    
    async def generate_code(
        self,
        prompt: str,
        language: str = 'python',
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Mock code generation."""
        self.call_count += 1
        self.last_request = {
            'prompt': prompt,
            'language': language,
            'context': context
        }
        
        # Generate different responses based on prompt keywords
        if 'fibonacci' in prompt.lower():
            code = '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
        elif 'class' in prompt.lower():
            code = '''
class GeneratedClass:
    """Auto-generated class."""
    
    def __init__(self, value=None):
        self.value = value
    
    def get_value(self):
        return self.value
'''
        else:
            code = f'''
def generated_function():
    """Generated function for: {prompt}"""
    return "{language} implementation"
'''
        
        return {
            'code': code.strip(),
            'language': language,
            'metadata': {
                'generated_at': '2025-01-01T00:00:00Z',
                'model': 'mock-ai-model',
                'prompt': prompt
            }
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Mock connection test."""
        return {'status': 'connected', 'mock': True}


class MockClaudeCodeIntegration:
    """Mock Claude Code integration."""
    
    def __init__(self):
        self.connected = True
        self.call_history = []
    
    async def test_connection(self) -> Dict[str, Any]:
        """Mock connection test."""
        return {'status': 'connected' if self.connected else 'disconnected'}
    
    async def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Mock command execution."""
        self.call_history.append({'command': command, 'kwargs': kwargs})
        
        return {
            'success': True,
            'output': f"Mock execution of: {command}",
            'metadata': {'mock': True}
        }


class MockClaudeFlowIntegration:
    """Mock Claude Flow integration."""
    
    def __init__(self):
        self.connected = True
        self.orchestration_history = []
    
    async def test_connection(self) -> Dict[str, Any]:
        """Mock connection test."""
        return {'status': 'connected' if self.connected else 'disconnected'}
    
    async def orchestrate_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock task orchestration."""
        self.orchestration_history.append(request)
        
        task_id = f"mock-task-{len(self.orchestration_history)}"
        
        # Generate mock agents based on task complexity
        agents = ['coder', 'reviewer']
        if 'complex' in request.get('task', '').lower():
            agents.extend(['architect', 'tester'])
        
        return {
            'task_id': task_id,
            'status': 'running',
            'agents': agents,
            'estimated_completion': '2025-01-01T01:00:00Z',
            'mock': True
        }


class MockProjectManager:
    """Mock project manager for testing."""
    
    def __init__(self):
        self.projects = {}
        self.call_history = []
    
    async def create_project(
        self,
        name: str,
        project_type: str,
        path: Path,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mock project creation."""
        project_id = f"mock-project-{len(self.projects)}"
        
        project_data = {
            'id': project_id,
            'name': name,
            'type': project_type,
            'path': str(path),
            'template': template,
            'created_at': '2025-01-01T00:00:00Z',
            'mock': True
        }
        
        self.projects[project_id] = project_data
        self.call_history.append(('create_project', name, project_type))
        
        return project_data
    
    async def load_project(self, path: Path) -> Dict[str, Any]:
        """Mock project loading."""
        self.call_history.append(('load_project', str(path)))
        
        return {
            'name': path.name,
            'type': 'python',  # Default type
            'path': str(path),
            'loaded_at': '2025-01-01T00:00:00Z',
            'mock': True
        }
    
    async def get_active_projects(self) -> List[Dict[str, Any]]:
        """Mock active projects retrieval."""
        return list(self.projects.values())


class MockTaskEngine:
    """Mock task engine for testing."""
    
    def __init__(self):
        self.execution_history = []
        self.should_fail = False
        self.execution_time = 0.1
    
    async def execute_task(self, task) -> Dict[str, Any]:
        """Mock task execution."""
        if self.should_fail:
            raise Exception("Mock task execution failure")
        
        # Simulate execution time
        await asyncio.sleep(self.execution_time)
        
        execution_result = {
            'result': f"Mock execution of task: {getattr(task, 'name', 'unnamed')}",
            'metadata': {
                'execution_time': self.execution_time,
                'mock': True
            }
        }
        
        self.execution_history.append(execution_result)
        return execution_result
    
    def set_failure_mode(self, should_fail: bool):
        """Configure mock to simulate failures."""
        self.should_fail = should_fail
    
    def set_execution_time(self, time_seconds: float):
        """Configure mock execution time."""
        self.execution_time = time_seconds


class MockProgressValidator:
    """Mock progress validator for testing."""
    
    def __init__(self):
        self.validation_history = []
    
    class MockProgressResult:
        """Mock progress validation result."""
        
        def __init__(
            self,
            is_valid: bool = True,
            authenticity_score: float = 0.85,
            issues: Optional[List[str]] = None,
            suggestions: Optional[List[str]] = None,
            placeholder_count: int = 0
        ):
            self.is_valid = is_valid
            self.authenticity_score = authenticity_score
            self.issues = issues or []
            self.suggestions = suggestions or []
            self.placeholder_count = placeholder_count
    
    async def validate_progress(
        self,
        file_path: Path,
        project_context: Optional[Dict[str, Any]] = None
    ) -> MockProgressResult:
        """Mock progress validation."""
        self.validation_history.append({
            'file_path': str(file_path),
            'project_context': project_context
        })
        
        # Simulate different results based on file content
        if file_path.suffix == '.py':
            try:
                content = file_path.read_text() if file_path.exists() else ""
                
                # Check for placeholder patterns
                placeholder_count = 0
                placeholder_patterns = ['TODO', 'FIXME', 'pass', '...']
                for pattern in placeholder_patterns:
                    placeholder_count += content.count(pattern)
                
                # Determine authenticity based on placeholder count
                if placeholder_count > 5:
                    return self.MockProgressResult(
                        is_valid=False,
                        authenticity_score=0.3,
                        issues=['Too many placeholders detected'],
                        placeholder_count=placeholder_count
                    )
                elif placeholder_count > 2:
                    return self.MockProgressResult(
                        is_valid=True,
                        authenticity_score=0.6,
                        suggestions=['Consider completing placeholder implementations'],
                        placeholder_count=placeholder_count
                    )
                else:
                    return self.MockProgressResult(
                        is_valid=True,
                        authenticity_score=0.9,
                        placeholder_count=placeholder_count
                    )
            except Exception:
                return self.MockProgressResult(
                    is_valid=False,
                    authenticity_score=0.0,
                    issues=['Failed to read file']
                )
        
        # Default result for non-Python files
        return self.MockProgressResult()


class MockFileSystem:
    """Mock file system operations for testing."""
    
    def __init__(self):
        self.files = {}
        self.directories = set()
        self.operations = []
    
    def create_file(self, path: str, content: str = ""):
        """Create a mock file."""
        self.files[path] = content
        self.operations.append(('create_file', path))
    
    def create_directory(self, path: str):
        """Create a mock directory."""
        self.directories.add(path)
        self.operations.append(('create_directory', path))
    
    def read_file(self, path: str) -> str:
        """Read a mock file."""
        self.operations.append(('read_file', path))
        return self.files.get(path, "")
    
    def file_exists(self, path: str) -> bool:
        """Check if mock file exists."""
        return path in self.files
    
    def directory_exists(self, path: str) -> bool:
        """Check if mock directory exists."""
        return path in self.directories
    
    def list_files(self, directory: str) -> List[str]:
        """List files in mock directory."""
        self.operations.append(('list_files', directory))
        return [f for f in self.files.keys() if f.startswith(directory)]


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.data = {}
        self.queries = []
        self.connected = False
    
    async def connect(self):
        """Mock database connection."""
        self.connected = True
    
    async def disconnect(self):
        """Mock database disconnection."""
        self.connected = False
    
    async def execute(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Mock query execution."""
        self.queries.append({'query': query, 'params': params})
        
        return {
            'affected_rows': 1,
            'result': f"Mock execution of: {query[:50]}...",
            'mock': True
        }
    
    async def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Mock data fetching."""
        self.queries.append({'query': query, 'params': params})
        
        # Return mock data based on query
        if 'users' in query.lower():
            return [
                {'id': 1, 'name': 'Test User 1', 'email': 'test1@example.com'},
                {'id': 2, 'name': 'Test User 2', 'email': 'test2@example.com'}
            ]
        elif 'projects' in query.lower():
            return [
                {'id': 1, 'name': 'Test Project 1', 'type': 'python'},
                {'id': 2, 'name': 'Test Project 2', 'type': 'javascript'}
            ]
        
        return []


class MockNetworkClient:
    """Mock network client for testing external API calls."""
    
    def __init__(self):
        self.requests = []
        self.responses = {}
        self.should_fail = False
        self.failure_message = "Mock network error"
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock GET request."""
        self.requests.append(('GET', url, kwargs))
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Return pre-configured response or default
        response = self.responses.get(url, {
            'status': 200,
            'data': {'mock': True, 'url': url}
        })
        
        return response
    
    async def post(self, url: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Mock POST request."""
        self.requests.append(('POST', url, data, kwargs))
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        return {
            'status': 201,
            'data': {'created': True, 'mock': True}
        }
    
    def set_response(self, url: str, response: Dict[str, Any]):
        """Configure mock response for specific URL."""
        self.responses[url] = response
    
    def set_failure_mode(self, should_fail: bool, message: str = "Mock network error"):
        """Configure mock to simulate network failures."""
        self.should_fail = should_fail
        self.failure_message = message


# Factory functions for creating mocks
def create_mock_ai_service() -> MockAIInterface:
    """Create mock AI interface."""
    return MockAIInterface()


def create_mock_project_manager() -> MockProjectManager:
    """Create mock project manager."""
    return MockProjectManager()


def create_mock_task_engine() -> MockTaskEngine:
    """Create mock task engine."""
    return MockTaskEngine()


def create_mock_progress_validator() -> MockProgressValidator:
    """Create mock progress validator."""
    return MockProgressValidator()


def create_mock_file_system() -> MockFileSystem:
    """Create mock file system."""
    return MockFileSystem()


def create_mock_database() -> MockDatabase:
    """Create mock database."""
    return MockDatabase()


def create_mock_network_client() -> MockNetworkClient:
    """Create mock network client."""
    return MockNetworkClient()


# Context managers for patching
class MockServiceContext:
    """Context manager for mocking all services."""
    
    def __init__(self):
        self.patches = []
        self.mocks = {}
    
    def __enter__(self):
        """Enter context and apply patches."""
        from unittest.mock import patch
        
        # Create mocks
        self.mocks['ai_interface'] = create_mock_ai_service()
        self.mocks['claude_code'] = MockClaudeCodeIntegration()
        self.mocks['claude_flow'] = MockClaudeFlowIntegration()
        self.mocks['project_manager'] = create_mock_project_manager()
        self.mocks['task_engine'] = create_mock_task_engine()
        self.mocks['progress_validator'] = create_mock_progress_validator()
        
        # Apply patches
        patches = [
            ('services.ai_service.AIInterface', self.mocks['ai_interface']),
            ('services.ai_service.ClaudeCodeIntegration', self.mocks['claude_code']),
            ('services.ai_service.ClaudeFlowIntegration', self.mocks['claude_flow']),
            ('services.project_service.ProjectManager', self.mocks['project_manager']),
            ('services.task_service.TaskEngine', self.mocks['task_engine']),
            ('services.validation_service.ProgressValidator', self.mocks['progress_validator']),
        ]
        
        for target, mock in patches:
            patcher = patch(target, return_value=mock)
            self.patches.append(patcher)
            patcher.start()
        
        return self.mocks
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and stop patches."""
        for patcher in self.patches:
            patcher.stop()
        self.patches.clear()