"""
Task-related test fixtures and factories.
"""

import pytest
from faker import Faker
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

fake = Faker()


class TaskType(Enum):
    """Available task types."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    BUG_FIX = "bug_fix"
    FEATURE = "feature"


class TaskStatus(Enum):
    """Task status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MockTask:
    """Mock task data structure."""
    id: str
    name: str
    description: str
    prompt: str
    type: str
    status: str
    priority: str
    project_id: str
    estimated_duration: int  # minutes
    actual_duration: Optional[int]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    dependencies: List[str]
    tags: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "type": self.type,
            "status": self.status,
            "priority": self.priority,
            "project_id": self.project_id,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "metadata": self.metadata
        }


class TaskFactory:
    """Factory for creating test task data."""
    
    @staticmethod
    def create_task_data(**overrides) -> Dict[str, Any]:
        """Create task data dictionary."""
        defaults = {
            "name": fake.catch_phrase(),
            "description": fake.text(max_nb_chars=300),
            "prompt": fake.sentence(nb_words=10),
            "type": fake.random_element(TaskType).value,
            "status": TaskStatus.PENDING.value,
            "priority": TaskPriority.MEDIUM.value,
            "estimated_duration": fake.random_int(min=15, max=240),  # 15 min to 4 hours
            "tags": [fake.word() for _ in range(fake.random_int(min=1, max=4))],
            "metadata": {
                "complexity": fake.random_element(["low", "medium", "high"]),
                "requires_review": fake.boolean(),
                "automated": fake.boolean()
            }
        }
        defaults.update(overrides)
        return defaults
    
    @staticmethod
    def create_mock_task(project_id: str = None, **overrides) -> MockTask:
        """Create mock task instance."""
        data = TaskFactory.create_task_data(**overrides)
        
        now = datetime.now()
        completed_at = None
        actual_duration = None
        
        if data["status"] == TaskStatus.COMPLETED.value:
            completed_at = now - timedelta(minutes=fake.random_int(min=1, max=60))
            actual_duration = data["estimated_duration"] + fake.random_int(min=-30, max=60)
        
        return MockTask(
            id=fake.uuid4(),
            project_id=project_id or fake.uuid4(),
            created_at=now - timedelta(days=fake.random_int(min=0, max=30)),
            updated_at=now - timedelta(hours=fake.random_int(min=0, max=24)),
            completed_at=completed_at,
            actual_duration=actual_duration,
            dependencies=[],
            **data
        )
    
    @staticmethod
    def create_task_chain(count: int = 3, project_id: str = None) -> List[MockTask]:
        """Create a chain of dependent tasks."""
        tasks = []
        previous_task_id = None
        
        for i in range(count):
            task = TaskFactory.create_mock_task(
                project_id=project_id,
                name=f"Task {i + 1} in chain",
                dependencies=[previous_task_id] if previous_task_id else []
            )
            tasks.append(task)
            previous_task_id = task.id
        
        return tasks
    
    @staticmethod
    def create_code_generation_task(**overrides) -> MockTask:
        """Create a code generation task."""
        defaults = {
            "name": "Generate user authentication module",
            "type": TaskType.CODE_GENERATION.value,
            "prompt": "Create a secure user authentication system with login, logout, and password reset functionality",
            "tags": ["authentication", "security", "backend"],
            "estimated_duration": 120,
            "metadata": {
                "complexity": "high",
                "requires_review": True,
                "automated": False,
                "expected_files": ["auth.py", "models.py", "tests/test_auth.py"]
            }
        }
        defaults.update(overrides)
        return TaskFactory.create_mock_task(**defaults)
    
    @staticmethod
    def create_testing_task(**overrides) -> MockTask:
        """Create a testing task."""
        defaults = {
            "name": "Write comprehensive tests",
            "type": TaskType.TESTING.value,
            "prompt": "Create unit tests with >90% coverage for the authentication module",
            "tags": ["testing", "coverage", "quality"],
            "estimated_duration": 90,
            "metadata": {
                "complexity": "medium",
                "requires_review": True,
                "automated": True,
                "target_coverage": 90
            }
        }
        defaults.update(overrides)
        return TaskFactory.create_mock_task(**defaults)
    
    @staticmethod
    def create_bug_fix_task(**overrides) -> MockTask:
        """Create a bug fix task."""
        defaults = {
            "name": "Fix authentication timeout issue",
            "type": TaskType.BUG_FIX.value,
            "priority": TaskPriority.HIGH.value,
            "prompt": "Investigate and fix the authentication timeout that occurs after 30 minutes of inactivity",
            "tags": ["bug", "authentication", "timeout"],
            "estimated_duration": 60,
            "metadata": {
                "complexity": "medium",
                "requires_review": True,
                "automated": False,
                "bug_severity": "high",
                "reproduction_steps": ["Login to system", "Wait 30 minutes", "Try to access protected resource"]
            }
        }
        defaults.update(overrides)
        return TaskFactory.create_mock_task(**defaults)


@pytest.fixture
def task_factory():
    """Provide task factory for tests."""
    return TaskFactory


@pytest.fixture
def sample_task_data():
    """Generate sample task data."""
    return TaskFactory.create_task_data()


@pytest.fixture
def code_generation_task():
    """Create a code generation task."""
    return TaskFactory.create_code_generation_task()


@pytest.fixture
def testing_task():
    """Create a testing task."""
    return TaskFactory.create_testing_task()


@pytest.fixture
def bug_fix_task():
    """Create a bug fix task."""
    return TaskFactory.create_bug_fix_task()


@pytest.fixture
def task_chain(sample_project_data):
    """Create a chain of dependent tasks."""
    project_id = fake.uuid4()
    return TaskFactory.create_task_chain(count=5, project_id=project_id)


@pytest.fixture
def mixed_status_tasks():
    """Create tasks with various statuses."""
    project_id = fake.uuid4()
    
    return [
        TaskFactory.create_mock_task(
            project_id=project_id,
            status=TaskStatus.COMPLETED.value,
            name="Completed task"
        ),
        TaskFactory.create_mock_task(
            project_id=project_id,
            status=TaskStatus.IN_PROGRESS.value,
            name="In progress task"
        ),
        TaskFactory.create_mock_task(
            project_id=project_id,
            status=TaskStatus.PENDING.value,
            name="Pending task"
        ),
        TaskFactory.create_mock_task(
            project_id=project_id,
            status=TaskStatus.FAILED.value,
            name="Failed task"
        )
    ]


class TaskTestHelper:
    """Helper methods for task testing."""
    
    @staticmethod
    def assert_valid_task_data(data: Dict[str, Any]):
        """Assert task data has required fields."""
        required_fields = ["name", "description", "prompt", "type", "status", "priority"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert len(data["name"]) > 0, "Task name cannot be empty"
        assert len(data["prompt"]) > 0, "Task prompt cannot be empty"
        assert data["type"] in [t.value for t in TaskType], f"Invalid task type: {data['type']}"
        assert data["status"] in [s.value for s in TaskStatus], f"Invalid status: {data['status']}"
        assert data["priority"] in [p.value for p in TaskPriority], f"Invalid priority: {data['priority']}"
        
        if "estimated_duration" in data:
            assert data["estimated_duration"] > 0, "Estimated duration must be positive"
    
    @staticmethod
    def assert_task_chain_valid(tasks: List[MockTask]):
        """Assert task chain has valid dependencies."""
        task_ids = {task.id for task in tasks}
        
        for i, task in enumerate(tasks):
            # First task should have no dependencies
            if i == 0:
                assert len(task.dependencies) == 0, "First task should have no dependencies"
            else:
                # Each subsequent task should depend on previous task
                assert len(task.dependencies) > 0, f"Task {i} should have dependencies"
                for dep_id in task.dependencies:
                    assert dep_id in task_ids, f"Dependency {dep_id} not found in task chain"
    
    @staticmethod
    def assert_task_completion_valid(task: MockTask):
        """Assert completed task has valid completion data."""
        if task.status == TaskStatus.COMPLETED.value:
            assert task.completed_at is not None, "Completed task must have completion timestamp"
            assert task.actual_duration is not None, "Completed task must have actual duration"
            assert task.actual_duration > 0, "Actual duration must be positive"
        else:
            assert task.completed_at is None, "Non-completed task should not have completion timestamp"
    
    @staticmethod
    def create_task_execution_result(task: MockTask, success: bool = True) -> Dict[str, Any]:
        """Create mock task execution result."""
        if success:
            return {
                "task_id": task.id,
                "status": "completed",
                "output": f"Successfully completed task: {task.name}",
                "files_created": [f"{task.name.lower().replace(' ', '_')}.py"],
                "files_modified": [],
                "execution_time": task.estimated_duration * 60,  # seconds
                "quality_score": fake.random.uniform(0.7, 1.0),
                "placeholders_found": fake.random_int(min=0, max=3),
                "tests_created": fake.random_int(min=1, max=5),
                "coverage": fake.random.uniform(0.8, 1.0)
            }
        else:
            return {
                "task_id": task.id,
                "status": "failed",
                "error": "Task execution failed due to invalid prompt",
                "output": "",
                "files_created": [],
                "files_modified": [],
                "execution_time": fake.random_int(min=5, max=30),
                "quality_score": 0.0,
                "placeholders_found": 0,
                "tests_created": 0,
                "coverage": 0.0
            }


@pytest.fixture
def task_helper():
    """Provide task test helper."""
    return TaskTestHelper


@pytest.fixture
def task_execution_results(mixed_status_tasks):
    """Create mock task execution results."""
    helper = TaskTestHelper()
    return [
        helper.create_task_execution_result(task, success=(task.status != TaskStatus.FAILED.value))
        for task in mixed_status_tasks
    ]