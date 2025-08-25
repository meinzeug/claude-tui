#!/usr/bin/env python3
"""
Mock Backend Bridge for TUI Testing

Comprehensive mock implementation of all backend services required for TUI testing.
This allows the TUI application to run in isolation without external dependencies.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class MockTUIEventType(str, Enum):
    """Mock TUI event types."""
    WIDGET_FOCUS_CHANGED = "widget_focus_changed"
    SCREEN_CHANGED = "screen_changed"
    PROJECT_LOADED = "project_loaded"
    TASK_SELECTED = "task_selected"
    UI_LAG_DETECTED = "ui_lag_detected"
    BACKEND_CONNECTION_STATUS = "backend_connection_status"
    SYNC_STATUS_CHANGED = "sync_status_changed"


@dataclass
class MockTUIState:
    """Mock TUI application state."""
    current_screen: str = "main"
    focused_widget: Optional[str] = None
    active_project: Optional[str] = None
    selected_task: Optional[str] = None
    open_files: Set[str] = field(default_factory=set)
    ui_theme: str = "dark"
    layout_mode: str = "standard"
    last_update: datetime = field(default_factory=datetime.now)
    refresh_rate: float = 60.0
    lag_threshold: float = 0.1
    backend_connected: bool = True  # Always connected for mock
    websocket_connected: bool = True
    claude_flow_connected: bool = True


@dataclass
class MockTUIEvent:
    """Mock TUI event data structure."""
    event_type: MockTUIEventType
    timestamp: datetime
    data: Dict[str, Any]
    widget_id: Optional[str] = None
    screen_name: Optional[str] = None
    user_id: Optional[str] = None


class MockBackendSyncStatus:
    """Mock backend synchronization status."""
    
    def __init__(self):
        self.connected = True
        self.last_sync = datetime.now()
        self.pending_updates = 0
        self.sync_errors = []
        self.latency_ms = 5.0  # Mock low latency


class MockServiceOrchestrator:
    """Mock service orchestrator that provides all backend services."""
    
    def __init__(self):
        self.services = {}
        self.service_status = "healthy"
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all mock services."""
        self.services = {
            'cache_service': MockCacheService(),
            'database_service': MockDatabaseService(),
            'claude_flow_service': MockClaudeFlowService(),
            'ai_service': MockAIService(),
            'project_service': MockProjectService(),
            'task_service': MockTaskService(),
            'validation_service': MockValidationService(),
            'websocket_service': MockWebSocketService()
        }
    
    def get_cache_service(self):
        """Get mock cache service."""
        return self.services['cache_service']
    
    def get_claude_flow_service(self):
        """Get mock Claude Flow service."""
        return self.services['claude_flow_service']
    
    def get_ai_service(self):
        """Get mock AI service."""
        return self.services['ai_service']
    
    def get_project_service(self):
        """Get mock project service."""
        return self.services['project_service']
    
    def get_task_service(self):
        """Get mock task service."""
        return self.services['task_service']
    
    def get_validation_service(self):
        """Get mock validation service."""
        return self.services['validation_service']
    
    async def get_service_status(self):
        """Get overall service status."""
        return {
            'overall_status': self.service_status,
            'services': {name: 'healthy' for name in self.services.keys()},
            'last_check': datetime.now().isoformat()
        }


class MockCacheService:
    """Mock cache service for testing."""
    
    def __init__(self):
        self.cache = {}
        self.operations = []
    
    async def get(self, key: str):
        """Mock cache get."""
        self.operations.append(('get', key))
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Mock cache set."""
        self.operations.append(('set', key, value, ttl))
        self.cache[key] = {
            'value': value,
            'ttl': ttl,
            'set_at': datetime.now()
        }
    
    async def delete(self, key: str):
        """Mock cache delete."""
        self.operations.append(('delete', key))
        if key in self.cache:
            del self.cache[key]


class MockDatabaseService:
    """Mock database service for testing."""
    
    def __init__(self):
        self.connected = True
        self.data = {
            'users': [
                {'id': 1, 'name': 'Test User', 'email': 'test@example.com'},
            ],
            'projects': [
                {'id': 1, 'name': 'Test Project', 'type': 'python', 'path': '/mock/project'},
            ],
            'tasks': [
                {'id': 1, 'name': 'Test Task', 'status': 'pending', 'project_id': 1},
            ]
        }
        self.queries = []
    
    async def execute_query(self, query: str, params: Optional[Dict] = None):
        """Mock query execution."""
        self.queries.append({'query': query, 'params': params})
        return {'affected_rows': 1, 'mock': True}
    
    async def fetch_all(self, query: str, params: Optional[Dict] = None):
        """Mock data fetching."""
        self.queries.append({'query': query, 'params': params})
        
        # Return mock data based on query
        if 'users' in query.lower():
            return self.data['users']
        elif 'projects' in query.lower():
            return self.data['projects']
        elif 'tasks' in query.lower():
            return self.data['tasks']
        
        return []


class MockClaudeFlowService:
    """Mock Claude Flow service for testing."""
    
    def __init__(self):
        self.connected = True
        self.orchestrations = []
        self.agents = ['coder', 'reviewer', 'tester', 'architect']
    
    async def orchestrate_task(self, request):
        """Mock task orchestration."""
        task_id = f"mock-task-{len(self.orchestrations)}"
        
        orchestration = {
            'task_id': task_id,
            'status': 'running',
            'agents_assigned': self.agents[:2],  # Assign 2 agents
            'progress': 0.0,
            'created_at': datetime.now().isoformat()
        }
        
        self.orchestrations.append(orchestration)
        return orchestration
    
    async def get_task_status(self, task_id: str):
        """Get mock task status."""
        for orchestration in self.orchestrations:
            if orchestration['task_id'] == task_id:
                # Simulate progress
                orchestration['progress'] = min(100.0, orchestration['progress'] + 10.0)
                return orchestration
        
        return None


class MockAIService:
    """Mock AI service for testing."""
    
    def __init__(self):
        self.connected = True
        self.requests = []
    
    async def generate_code(self, prompt: str, language: str = 'python', context: Optional[Dict] = None):
        """Mock code generation."""
        self.requests.append({'prompt': prompt, 'language': language, 'context': context})
        
        # Generate different mock responses based on prompt
        if 'fibonacci' in prompt.lower():
            code = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)'''
        elif 'class' in prompt.lower():
            code = '''class MockClass:
    def __init__(self):
        self.value = None
    
    def get_value(self):
        return self.value'''
        else:
            code = f'''# Generated code for: {prompt}
def mock_function():
    return "Mock {language} implementation"'''
        
        return {
            'code': code,
            'language': language,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model': 'mock-ai-model',
                'mock': True
            }
        }
    
    async def execute_task(self, task_description: str, context: Dict[str, Any]):
        """Mock AI task execution."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        result = {
            'result': f"Mock AI execution: {task_description}",
            'context': context,
            'completed_at': datetime.now().isoformat(),
            'mock': True
        }
        
        self.requests.append({'task': task_description, 'context': context, 'result': result})
        return result


class MockProjectService:
    """Mock project service for testing."""
    
    def __init__(self):
        self.projects = {}
        self.current_project = None
    
    def create_project_from_config(self, config):
        """Mock project creation."""
        project_id = f"mock-project-{len(self.projects)}"
        
        project = {
            'id': project_id,
            'name': getattr(config, 'name', 'Test Project'),
            'type': getattr(config, 'type', 'python'),
            'path': getattr(config, 'path', '/mock/project'),
            'created_at': datetime.now().isoformat(),
            'mock': True
        }
        
        self.projects[project_id] = project
        self.current_project = project
        return project
    
    def initialize(self):
        """Mock initialization."""
        pass
    
    def save_current_project(self):
        """Mock project saving."""
        if self.current_project:
            self.current_project['saved_at'] = datetime.now().isoformat()
            return True
        return False


class MockTaskService:
    """Mock task service for testing."""
    
    def __init__(self):
        self.tasks = {}
        self.execution_history = []
    
    async def create_task(self, name: str, description: str, project_id: str):
        """Mock task creation."""
        task_id = f"mock-task-{len(self.tasks)}"
        
        task = {
            'id': task_id,
            'name': name,
            'description': description,
            'project_id': project_id,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'mock': True
        }
        
        self.tasks[task_id] = task
        return task
    
    async def execute_task(self, task_id: str):
        """Mock task execution."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task['status'] = 'running'
            
            # Simulate execution
            await asyncio.sleep(0.1)
            
            task['status'] = 'completed'
            task['completed_at'] = datetime.now().isoformat()
            
            execution_result = {
                'task_id': task_id,
                'result': f"Mock execution of {task['name']}",
                'execution_time': 0.1,
                'mock': True
            }
            
            self.execution_history.append(execution_result)
            return execution_result
        
        return None


class MockValidationService:
    """Mock validation service for testing."""
    
    def __init__(self):
        self.validations = []
    
    def initialize(self):
        """Mock initialization."""
        pass
    
    async def analyze_project(self, project_path):
        """Mock project analysis."""
        await asyncio.sleep(0.05)  # Simulate analysis time
        
        # Mock progress report
        from types import SimpleNamespace
        
        progress_report = SimpleNamespace(
            real_progress=0.75,
            claimed_progress=0.85,
            fake_progress=0.10,
            quality_score=8.2,
            authenticity_score=0.82,
            placeholders_found=2,
            todos_found=3,
            analysis_timestamp=datetime.now().isoformat(),
            mock=True
        )
        
        self.validations.append({
            'project_path': project_path,
            'report': progress_report,
            'analyzed_at': datetime.now().isoformat()
        })
        
        return progress_report
    
    async def validate_ai_output(self, result, context):
        """Mock AI output validation."""
        # Mock validation result
        from types import SimpleNamespace
        
        validation = SimpleNamespace(
            is_authentic=True,
            authenticity_score=0.88,
            completion_suggestions=[
                "Consider adding error handling",
                "Add input validation",
                "Include unit tests"
            ],
            issues_found=[],
            mock=True
        )
        
        return validation


class MockWebSocketService:
    """Mock WebSocket service for testing."""
    
    def __init__(self):
        self.connected_clients = set()
        self.message_history = []
    
    async def connect_client(self, client_id: str):
        """Mock client connection."""
        self.connected_clients.add(client_id)
        return True
    
    async def disconnect_client(self, client_id: str):
        """Mock client disconnection."""
        self.connected_clients.discard(client_id)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Mock message broadcast."""
        broadcast_event = {
            'message': message,
            'recipients': list(self.connected_clients),
            'timestamp': datetime.now().isoformat(),
            'mock': True
        }
        
        self.message_history.append(broadcast_event)
        return len(self.connected_clients)


class MockTUIBackendBridge:
    """
    Mock TUI Backend Bridge for testing.
    
    Provides all functionality of the real bridge but with mocked services.
    """
    
    def __init__(self, config_manager=None):
        """Initialize mock TUI backend bridge."""
        self.config_manager = config_manager or MockConfigManager()
        self.orchestrator = MockServiceOrchestrator()
        self.tui_state = MockTUIState()
        self.event_handlers = {}
        self.event_queue = asyncio.Queue()
        self.sync_status = MockBackendSyncStatus()
        self.sync_tasks = set()
        self.performance_metrics = {}
        self.update_timestamps = []
        self.batch_events = []
        self.batch_timeout = 0.1
        self.last_batch_time = datetime.now()
        self.websocket_client = MockWebSocketClient()
        self.connection_manager = MockConnectionManager()
        
        logger.info("Mock TUI Backend Bridge initialized")
    
    async def initialize(self):
        """Initialize the mock backend bridge."""
        logger.info("Initializing Mock TUI Backend Bridge...")
        
        # Start mock background tasks
        await self._start_background_tasks()
        
        # Mock WebSocket connection
        await self._connect_websocket()
        
        # Mock Claude Flow initialization
        await self._initialize_claude_flow()
        
        self.sync_status.connected = True
        logger.info("Mock TUI Backend Bridge initialized successfully")
    
    async def _start_background_tasks(self):
        """Start mock background processing tasks."""
        # Create minimal mock tasks that don't actually run indefinitely
        task1 = asyncio.create_task(self._mock_event_processing())
        task2 = asyncio.create_task(self._mock_sync_monitoring())
        task3 = asyncio.create_task(self._mock_performance_monitoring())
        
        self.sync_tasks.update([task1, task2, task3])
        logger.info(f"Started {len(self.sync_tasks)} mock background tasks")
    
    async def _mock_event_processing(self):
        """Mock event processing loop."""
        try:
            for _ in range(3):  # Process a few mock events
                await asyncio.sleep(0.1)
            logger.debug("Mock event processing completed")
        except Exception as e:
            logger.error(f"Mock event processing error: {e}")
    
    async def _mock_sync_monitoring(self):
        """Mock sync monitoring loop."""
        try:
            await asyncio.sleep(0.1)
            self.sync_status.last_sync = datetime.now()
            logger.debug("Mock sync monitoring completed")
        except Exception as e:
            logger.error(f"Mock sync monitoring error: {e}")
    
    async def _mock_performance_monitoring(self):
        """Mock performance monitoring loop."""
        try:
            await asyncio.sleep(0.1)
            self.tui_state.refresh_rate = 60.0
            logger.debug("Mock performance monitoring completed")
        except Exception as e:
            logger.error(f"Mock performance monitoring error: {e}")
    
    async def _connect_websocket(self):
        """Mock WebSocket connection."""
        self.tui_state.websocket_connected = True
        logger.info("Mock WebSocket connection established")
    
    async def _initialize_claude_flow(self):
        """Mock Claude Flow initialization."""
        self.tui_state.claude_flow_connected = True
        logger.info("Mock Claude Flow integration initialized")
    
    async def emit_event(self, event):
        """Mock event emission."""
        event.created_at = datetime.now()
        await self.event_queue.put(event)
        logger.debug(f"Mock event emitted: {event.event_type}")
    
    def register_event_handler(self, event_type, handler):
        """Mock event handler registration."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Mock handler registered for {event_type}")
    
    async def update_tui_state(self, **kwargs):
        """Mock TUI state update."""
        for key, value in kwargs.items():
            if hasattr(self.tui_state, key):
                setattr(self.tui_state, key, value)
                logger.debug(f"Mock TUI state updated: {key} = {value}")
        
        self.tui_state.last_update = datetime.now()
    
    async def get_backend_data(self, data_type: str, **filters):
        """Mock backend data retrieval."""
        cache_service = self.orchestrator.get_cache_service()
        if cache_service:
            cache_key = f"mock_data:{data_type}"
            cached_data = await cache_service.get(cache_key)
            
            if cached_data:
                return cached_data['value']
            
            # Generate mock data
            mock_data = {
                'type': data_type,
                'filters': filters,
                'mock': True,
                'generated_at': datetime.now().isoformat()
            }
            
            await cache_service.set(cache_key, mock_data, ttl=300)
            return mock_data
        
        return None
    
    async def send_command_to_backend(self, command: str, parameters: Dict[str, Any] = None):
        """Mock command sending."""
        mock_command = {
            'command': command,
            'parameters': parameters or {},
            'timestamp': datetime.now().isoformat(),
            'tui_context': {
                'screen': self.tui_state.current_screen,
                'widget': self.tui_state.focused_widget,
                'project': self.tui_state.active_project
            },
            'mock': True
        }
        
        logger.info(f"Mock command sent: {command}")
        return True
    
    def get_sync_status(self):
        """Get mock synchronization status."""
        return self.sync_status
    
    def get_performance_metrics(self):
        """Get mock performance metrics."""
        return {
            'refresh_rate': self.tui_state.refresh_rate,
            'websocket_latency_ms': self.sync_status.latency_ms,
            'queue_size': self.event_queue.qsize(),
            'batch_size': len(self.batch_events),
            'mock': True
        }
    
    async def cleanup(self):
        """Mock cleanup."""
        logger.info("Cleaning up Mock TUI Backend Bridge...")
        
        # Cancel mock tasks
        for task in self.sync_tasks:
            task.cancel()
        
        # Reset state
        self.sync_status.connected = False
        self.tui_state.backend_connected = False
        self.tui_state.websocket_connected = False
        
        logger.info("Mock TUI Backend Bridge cleanup completed")


class MockWebSocketClient:
    """Mock WebSocket client."""
    
    def __init__(self):
        self.connected = True
        self.messages_sent = []
        self.messages_received = []
    
    async def send(self, message):
        """Mock message sending."""
        self.messages_sent.append({
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'mock': True
        })
    
    async def close(self):
        """Mock connection close."""
        self.connected = False


class MockConnectionManager:
    """Mock connection manager."""
    
    def __init__(self):
        self.active_connections = {}
    
    async def connect(self, websocket, client_id: str):
        """Mock connection."""
        self.active_connections[client_id] = websocket
        return True
    
    async def disconnect(self, client_id: str):
        """Mock disconnection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]


class MockConfigManager:
    """Mock configuration manager."""
    
    def __init__(self):
        self.config = {
            'debug': False,
            'validation_enabled': True,
            'theme': 'dark',
            'mock': True
        }
    
    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value


# Utility functions for creating mock instances
def get_mock_service_orchestrator():
    """Get mock service orchestrator instance."""
    return MockServiceOrchestrator()


def get_mock_tui_bridge(config_manager=None):
    """Get mock TUI bridge instance."""
    return MockTUIBackendBridge(config_manager)


async def initialize_mock_tui_bridge(config_manager=None):
    """Initialize mock TUI backend bridge."""
    bridge = MockTUIBackendBridge(config_manager)
    await bridge.initialize()
    return bridge


# Mock core classes for TUI components
class MockProjectManager:
    """Mock project manager for TUI testing."""
    
    def __init__(self):
        self.current_project = None
        self.projects = []
    
    def initialize(self):
        """Mock initialization."""
        pass
    
    def save_current_project(self):
        """Mock project saving."""
        if self.current_project:
            return True
        return False
    
    def create_project_from_config(self, config):
        """Mock project creation from config."""
        project = {
            'name': getattr(config, 'name', 'Test Project'),
            'type': getattr(config, 'type', 'python'),
            'path': getattr(config, 'path', Path('/mock/project')),
            'created_at': datetime.now(),
            'mock': True
        }
        
        self.current_project = project
        self.projects.append(project)
        return project


class MockAIInterface:
    """Mock AI interface for TUI testing."""
    
    def __init__(self):
        self.connected = True
        self.requests = []
    
    def initialize(self):
        """Mock initialization."""
        pass
    
    async def execute_task(self, task_description, context):
        """Mock AI task execution."""
        request = {
            'task_description': task_description,
            'context': context,
            'timestamp': datetime.now()
        }
        self.requests.append(request)
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        return f"Mock AI result for: {task_description}"
    
    async def complete_placeholder_code(self, code, suggestions):
        """Mock placeholder completion."""
        return f"Completed: {code} with suggestions: {suggestions}"


class MockValidationEngine:
    """Mock validation engine for TUI testing."""
    
    def __init__(self):
        self.validations = []
    
    def initialize(self):
        """Mock initialization."""
        pass
    
    async def analyze_project(self, project_path):
        """Mock project analysis."""
        from types import SimpleNamespace
        
        # Create mock progress report
        progress_report = SimpleNamespace(
            real_progress=0.7,
            claimed_progress=0.9,
            fake_progress=0.2,
            quality_score=7.5,
            authenticity_score=0.78,
            placeholders_found=3,
            todos_found=5,
            mock=True
        )
        
        self.validations.append({
            'project_path': project_path,
            'report': progress_report,
            'analyzed_at': datetime.now()
        })
        
        return progress_report
    
    async def validate_ai_output(self, result, context):
        """Mock AI output validation."""
        from types import SimpleNamespace
        
        validation = SimpleNamespace(
            is_authentic=True,
            authenticity_score=0.85,
            completion_suggestions=[
                "Add error handling",
                "Include documentation",
                "Add unit tests"
            ],
            mock=True
        )
        
        return validation


# Global mock instances
_mock_orchestrator = None
_mock_tui_bridge = None


def get_mock_service_orchestrator_instance():
    """Get singleton mock service orchestrator."""
    global _mock_orchestrator
    if _mock_orchestrator is None:
        _mock_orchestrator = MockServiceOrchestrator()
    return _mock_orchestrator


def get_mock_tui_bridge_instance():
    """Get singleton mock TUI bridge."""
    global _mock_tui_bridge
    if _mock_tui_bridge is None:
        _mock_tui_bridge = MockTUIBackendBridge()
    return _mock_tui_bridge


def reset_mock_instances():
    """Reset global mock instances."""
    global _mock_orchestrator, _mock_tui_bridge
    _mock_orchestrator = None
    _mock_tui_bridge = None


if __name__ == "__main__":
    # Simple test of mock backend
    async def test_mock_backend():
        """Test the mock backend functionality."""
        print("Testing Mock Backend...")
        
        # Test service orchestrator
        orchestrator = get_mock_service_orchestrator_instance()
        cache_service = orchestrator.get_cache_service()
        
        await cache_service.set("test_key", "test_value")
        value = await cache_service.get("test_key")
        print(f"Cache test: {value}")
        
        # Test TUI bridge
        bridge = get_mock_tui_bridge_instance()
        await bridge.initialize()
        
        # Test event emission
        event = MockTUIEvent(
            event_type=MockTUIEventType.SCREEN_CHANGED,
            timestamp=datetime.now(),
            data={"screen": "test"}
        )
        
        await bridge.emit_event(event)
        print(f"Event emitted: {event.event_type}")
        
        # Cleanup
        await bridge.cleanup()
        print("Mock backend test completed successfully!")
    
    # Run the test
    asyncio.run(test_mock_backend())