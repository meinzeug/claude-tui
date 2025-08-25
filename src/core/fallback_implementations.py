#!/usr/bin/env python3
"""
Fallback Implementations for Claude-TUI Components.

This module provides lightweight fallback implementations for critical components
when primary implementations fail. These fallbacks ensure the application 
continues to function even when dependencies are missing or services are down.

Key Features:
- Mock AI interfaces for development and testing
- In-memory data storage for database failures  
- Basic file operations for file system issues
- Simplified project management for recovery scenarios
- Offline mode capabilities
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from uuid import uuid4

from .error_handler import handle_errors, handle_async_errors


# Fallback AI Interface

class MockAIInterface:
    """Mock AI interface for development and fallback scenarios."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = {
            'code_generation': self._mock_code_response,
            'code_review': self._mock_review_response,
            'documentation': self._mock_docs_response,
            'task_creation': self._mock_task_response,
        }
    
    @handle_async_errors(component='mock_ai', silence_errors=True, fallback_return={'success': False})
    async def generate_code(self, prompt: str, language: str = 'python', **kwargs) -> Dict[str, Any]:
        """Generate mock code response."""
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate API delay
        
        return {
            'success': True,
            'code': f'# Generated code for: {prompt[:50]}...\n# Language: {language}\n# Mock implementation\npass',
            'explanation': f'This is a mock code generation response for development purposes.',
            'metadata': {
                'language': language,
                'tokens_used': len(prompt) // 4,
                'mock_call': True,
                'call_number': self.call_count
            }
        }
    
    @handle_async_errors(component='mock_ai', silence_errors=True, fallback_return={'success': False})
    async def review_code(self, code: str, **kwargs) -> Dict[str, Any]:
        """Generate mock code review."""
        self.call_count += 1
        await asyncio.sleep(0.1)
        
        issues = []
        if len(code) < 10:
            issues.append({
                'type': 'warning',
                'message': 'Code snippet seems very short',
                'line': 1,
                'suggestion': 'Consider adding more implementation details'
            })
        
        return {
            'success': True,
            'issues': issues,
            'overall_rating': 7.5,
            'suggestions': ['Add error handling', 'Include documentation', 'Add unit tests'],
            'metadata': {
                'lines_reviewed': code.count('\n') + 1,
                'mock_call': True,
                'call_number': self.call_count
            }
        }
    
    @handle_async_errors(component='mock_ai', silence_errors=True, fallback_return={'success': False})
    async def create_documentation(self, code: str, doc_type: str = 'api', **kwargs) -> Dict[str, Any]:
        """Generate mock documentation."""
        self.call_count += 1
        await asyncio.sleep(0.1)
        
        return {
            'success': True,
            'documentation': f'# Mock Documentation\n\n## Overview\nThis is mock documentation for the provided code.\n\n## Usage\n```python\n# Example usage\n{code[:100]}...\n```',
            'format': doc_type,
            'metadata': {
                'doc_type': doc_type,
                'mock_call': True,
                'call_number': self.call_count
            }
        }
    
    def _mock_code_response(self, prompt: str) -> str:
        return f"# Mock code for: {prompt}\npass"
    
    def _mock_review_response(self, code: str) -> str:
        return f"Mock review: Code looks acceptable (mock response)"
    
    def _mock_docs_response(self, code: str) -> str:
        return f"# Mock Documentation\n\nGenerated documentation for provided code."
    
    def _mock_task_response(self, description: str) -> str:
        return f"Mock task created: {description}"


# Fallback Data Storage

@dataclass
class InMemoryRecord:
    """In-memory record for fallback storage."""
    id: str
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class InMemoryStorage:
    """In-memory storage fallback for database failures."""
    
    def __init__(self):
        self.collections: Dict[str, Dict[str, InMemoryRecord]] = {}
        self.indexes: Dict[str, Dict[str, List[str]]] = {}
    
    @handle_errors(component='memory_storage', silence_errors=True)
    def create_collection(self, name: str) -> bool:
        """Create a new collection."""
        if name not in self.collections:
            self.collections[name] = {}
            self.indexes[name] = {}
        return True
    
    @handle_errors(component='memory_storage', silence_errors=True)
    def insert(self, collection: str, data: Dict[str, Any]) -> str:
        """Insert a record into collection."""
        if collection not in self.collections:
            self.create_collection(collection)
        
        record_id = str(uuid4())
        now = datetime.utcnow()
        
        record = InMemoryRecord(
            id=record_id,
            data=data,
            created_at=now,
            updated_at=now
        )
        
        self.collections[collection][record_id] = record
        self._update_indexes(collection, record_id, data)
        
        return record_id
    
    @handle_errors(component='memory_storage', silence_errors=True, fallback_return=None)
    def find_by_id(self, collection: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Find record by ID."""
        if collection in self.collections:
            record = self.collections[collection].get(record_id)
            if record:
                return {**record.data, 'id': record.id}
        return None
    
    @handle_errors(component='memory_storage', silence_errors=True, fallback_return=[])
    def find_by_field(self, collection: str, field: str, value: Any) -> List[Dict[str, Any]]:
        """Find records by field value."""
        results = []
        if collection in self.collections:
            for record in self.collections[collection].values():
                if record.data.get(field) == value:
                    results.append({**record.data, 'id': record.id})
        return results
    
    @handle_errors(component='memory_storage', silence_errors=True, fallback_return=[])
    def list_all(self, collection: str) -> List[Dict[str, Any]]:
        """List all records in collection."""
        results = []
        if collection in self.collections:
            for record in self.collections[collection].values():
                results.append({**record.data, 'id': record.id})
        return results
    
    @handle_errors(component='memory_storage', silence_errors=True)
    def update(self, collection: str, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record."""
        if collection in self.collections and record_id in self.collections[collection]:
            record = self.collections[collection][record_id]
            record.data.update(data)
            record.updated_at = datetime.utcnow()
            self._update_indexes(collection, record_id, record.data)
            return True
        return False
    
    @handle_errors(component='memory_storage', silence_errors=True)
    def delete(self, collection: str, record_id: str) -> bool:
        """Delete a record."""
        if collection in self.collections and record_id in self.collections[collection]:
            del self.collections[collection][record_id]
            return True
        return False
    
    def _update_indexes(self, collection: str, record_id: str, data: Dict[str, Any]):
        """Update field indexes for fast lookup."""
        # Simple indexing for common fields
        indexed_fields = ['name', 'type', 'status', 'priority']
        
        for field in indexed_fields:
            if field in data:
                if field not in self.indexes[collection]:
                    self.indexes[collection][field] = {}
                
                value = str(data[field])
                if value not in self.indexes[collection][field]:
                    self.indexes[collection][field][value] = []
                
                if record_id not in self.indexes[collection][field][value]:
                    self.indexes[collection][field][value].append(record_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'collections': len(self.collections),
            'total_records': sum(len(coll) for coll in self.collections.values()),
            'collection_stats': {}
        }
        
        for name, collection in self.collections.items():
            stats['collection_stats'][name] = {
                'record_count': len(collection),
                'indexed_fields': list(self.indexes.get(name, {}).keys())
            }
        
        return stats


# Fallback Project Manager

class BasicProjectManager:
    """Basic project manager fallback for complex project management failures."""
    
    def __init__(self, storage: Optional[InMemoryStorage] = None):
        self.storage = storage or InMemoryStorage()
        self.current_project = None
        self.projects_collection = 'projects'
        self.storage.create_collection(self.projects_collection)
    
    @handle_errors(component='basic_project_manager', silence_errors=True)
    def create_project(self, name: str, path: str, **kwargs) -> Optional[str]:
        """Create a new project."""
        project_data = {
            'name': name,
            'path': path,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active',
            'description': kwargs.get('description', ''),
            'language': kwargs.get('language', 'python'),
            'framework': kwargs.get('framework', 'generic')
        }
        
        project_id = self.storage.insert(self.projects_collection, project_data)
        return project_id
    
    @handle_errors(component='basic_project_manager', silence_errors=True, fallback_return=None)
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        return self.storage.find_by_id(self.projects_collection, project_id)
    
    @handle_errors(component='basic_project_manager', silence_errors=True, fallback_return=[])
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        return self.storage.list_all(self.projects_collection)
    
    @handle_errors(component='basic_project_manager', silence_errors=True)
    def set_current_project(self, project_id: str) -> bool:
        """Set the current active project."""
        project = self.get_project(project_id)
        if project:
            self.current_project = project
            return True
        return False
    
    @handle_errors(component='basic_project_manager', silence_errors=True, fallback_return=None)
    def get_current_project(self) -> Optional[Dict[str, Any]]:
        """Get the current active project."""
        return self.current_project
    
    @handle_errors(component='basic_project_manager', silence_errors=True, fallback_return=[])
    def get_project_files(self, project_id: str) -> List[str]:
        """Get list of files in project (basic implementation)."""
        project = self.get_project(project_id)
        if not project:
            return []
        
        try:
            project_path = Path(project['path'])
            if project_path.exists():
                # Simple file listing
                files = []
                for file_path in project_path.rglob('*'):
                    if file_path.is_file():
                        files.append(str(file_path.relative_to(project_path)))
                return files[:100]  # Limit to first 100 files
        except Exception:
            pass
        
        return []


# Fallback Configuration Manager

class BasicConfigManager:
    """Basic configuration manager fallback."""
    
    def __init__(self):
        self.config = self._load_default_config()
        self.config_file = None
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            'app': {
                'name': 'Claude-TUI',
                'version': '1.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            'ai': {
                'provider': 'mock',
                'model': 'fallback',
                'timeout': 30,
                'max_retries': 3
            },
            'ui': {
                'theme': 'dark',
                'refresh_interval': 5,
                'show_notifications': True
            },
            'storage': {
                'type': 'memory',
                'max_records': 10000
            },
            'features': {
                'auto_save': True,
                'validation': True,
                'offline_mode': True
            }
        }
    
    @handle_errors(component='basic_config', silence_errors=True, fallback_return=None)
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @handle_errors(component='basic_config', silence_errors=True)
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            return True
        except (KeyError, TypeError):
            return False
    
    @handle_errors(component='basic_config', silence_errors=True, fallback_return={})
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def is_offline_mode(self) -> bool:
        """Check if running in offline mode."""
        return self.get('features.offline_mode', True)


# Fallback File System Operations

class SafeFileOperations:
    """Safe file operations with error handling and recovery."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.backup_dir = self.temp_dir / 'claude_tui_backup'
        self.backup_dir.mkdir(exist_ok=True)
    
    @handle_errors(component='safe_file_ops', silence_errors=True, fallback_return='')
    def read_file(self, file_path: Union[str, Path]) -> str:
        """Safely read file with fallback."""
        try:
            return Path(file_path).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    return Path(file_path).read_text(encoding=encoding)
                except:
                    continue
        except Exception:
            # Check backup
            backup_path = self.backup_dir / Path(file_path).name
            if backup_path.exists():
                return backup_path.read_text(encoding='utf-8')
        
        return ''
    
    @handle_errors(component='safe_file_ops', silence_errors=True)
    def write_file(self, file_path: Union[str, Path], content: str, backup: bool = True) -> bool:
        """Safely write file with backup."""
        file_path = Path(file_path)
        
        # Create backup if file exists
        if backup and file_path.exists():
            backup_path = self.backup_dir / f"{file_path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                backup_path.write_text(file_path.read_text())
            except:
                pass  # Continue without backup
        
        # Write file
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return True
        except Exception:
            # Try writing to temp location
            temp_path = self.temp_dir / file_path.name
            try:
                temp_path.write_text(content, encoding='utf-8')
                return True
            except:
                return False
    
    @handle_errors(component='safe_file_ops', silence_errors=True, fallback_return=[])
    def list_files(self, directory: Union[str, Path], pattern: str = '*') -> List[str]:
        """Safely list files in directory."""
        try:
            dir_path = Path(directory)
            if dir_path.exists() and dir_path.is_dir():
                return [str(p) for p in dir_path.glob(pattern) if p.is_file()]
        except Exception:
            pass
        return []
    
    @handle_errors(component='safe_file_ops', silence_errors=True)
    def create_directory(self, dir_path: Union[str, Path]) -> bool:
        """Safely create directory."""
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False


# Fallback Task Engine

class BasicTaskEngine:
    """Basic task engine for fallback operations."""
    
    def __init__(self, storage: Optional[InMemoryStorage] = None):
        self.storage = storage or InMemoryStorage()
        self.tasks_collection = 'tasks'
        self.storage.create_collection(self.tasks_collection)
        self.task_counter = 0
    
    @handle_errors(component='basic_task_engine', silence_errors=True)
    def create_task(self, name: str, description: str = '', **kwargs) -> str:
        """Create a new task."""
        self.task_counter += 1
        task_data = {
            'name': name,
            'description': description,
            'status': 'pending',
            'priority': kwargs.get('priority', 'medium'),
            'created_at': datetime.utcnow().isoformat(),
            'progress': 0.0,
            'task_number': self.task_counter
        }
        
        return self.storage.insert(self.tasks_collection, task_data)
    
    @handle_errors(component='basic_task_engine', silence_errors=True, fallback_return=None)
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.storage.find_by_id(self.tasks_collection, task_id)
    
    @handle_errors(component='basic_task_engine', silence_errors=True, fallback_return=[])
    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks, optionally filtered by status."""
        if status:
            return self.storage.find_by_field(self.tasks_collection, 'status', status)
        return self.storage.list_all(self.tasks_collection)
    
    @handle_errors(component='basic_task_engine', silence_errors=True)
    def update_task_status(self, task_id: str, status: str, progress: float = None) -> bool:
        """Update task status and progress."""
        update_data = {'status': status}
        if progress is not None:
            update_data['progress'] = progress
        
        return self.storage.update(self.tasks_collection, task_id, update_data)


# Factory Functions

def create_fallback_ai_interface() -> MockAIInterface:
    """Create fallback AI interface."""
    return MockAIInterface()

def create_fallback_storage() -> InMemoryStorage:
    """Create fallback storage."""
    return InMemoryStorage()

def create_fallback_project_manager(storage: Optional[InMemoryStorage] = None) -> BasicProjectManager:
    """Create fallback project manager."""
    return BasicProjectManager(storage)

def create_fallback_config_manager() -> BasicConfigManager:
    """Create fallback configuration manager."""
    return BasicConfigManager()

def create_fallback_file_operations() -> SafeFileOperations:
    """Create fallback file operations."""
    return SafeFileOperations()

def create_fallback_task_engine(storage: Optional[InMemoryStorage] = None) -> BasicTaskEngine:
    """Create fallback task engine."""
    return BasicTaskEngine(storage)


# Export all fallback implementations
__all__ = [
    'MockAIInterface',
    'InMemoryStorage',
    'BasicProjectManager',
    'BasicConfigManager',
    'SafeFileOperations',
    'BasicTaskEngine',
    'create_fallback_ai_interface',
    'create_fallback_storage',
    'create_fallback_project_manager',
    'create_fallback_config_manager',
    'create_fallback_file_operations',
    'create_fallback_task_engine'
]