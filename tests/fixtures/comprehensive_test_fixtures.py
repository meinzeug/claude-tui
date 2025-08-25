#!/usr/bin/env python3
"""
Comprehensive Test Fixtures and Utilities.

Provides reusable test fixtures, mock objects, and utilities for:
- Project and task management testing
- AI integration testing
- Validation and performance testing
- Database and file system mocking
- Test data generation and validation
"""

import asyncio
import json
import pytest
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass, field

# Import test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.types import (
    Task, TaskStatus, Priority, ProjectState, ProgressMetrics,
    ValidationResult, Issue, IssueType, Severity, AITaskResult
)
from core.project_manager import Project
from core.config_manager import ProjectConfig


@dataclass
class TestConfig:
    """Configuration for test fixtures."""
    enable_real_integrations: bool = False
    enable_performance_tests: bool = True
    temp_dir_cleanup: bool = True
    mock_ai_responses: bool = True
    generate_realistic_data: bool = True
    seed: int = 42


class TestDataGenerator:
    """Generates realistic test data for various scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible data."""
        import random
        self.random = random.Random(seed)
        self.faker = self._setup_faker(seed)
    
    def _setup_faker(self, seed: int):
        """Setup Faker for realistic data generation."""
        try:
            from faker import Faker
            fake = Faker()
            fake.seed_instance(seed)
            return fake
        except ImportError:
            # Fallback to simple data generation
            return None
    
    def generate_project_data(self, count: int = 1) -> List[Dict[str, Any]]:
        """Generate realistic project data."""
        projects = []
        
        project_types = ['python', 'javascript', 'typescript', 'java', 'go']
        frameworks = {
            'python': ['fastapi', 'django', 'flask', 'basic'],
            'javascript': ['react', 'vue', 'angular', 'node'],
            'typescript': ['next', 'nest', 'angular', 'express'],
            'java': ['spring', 'maven', 'gradle', 'basic'],
            'go': ['gin', 'echo', 'basic', 'grpc']
        }
        
        for i in range(count):
            project_type = self.random.choice(project_types)
            framework = self.random.choice(frameworks[project_type])
            
            project_data = {
                'id': str(uuid.uuid4()),
                'name': self._generate_project_name(),
                'description': self._generate_project_description(),
                'type': project_type,
                'framework': framework,
                'created_at': self._generate_datetime(),
                'state': self.random.choice(list(ProjectState)),
                'features': self._generate_project_features(project_type),
                'complexity': self.random.choice(['low', 'medium', 'high']),
                'estimated_duration': self.random.randint(30, 480),  # 30 minutes to 8 hours
            }
            
            projects.append(project_data)
        
        return projects
    
    def generate_task_data(self, count: int = 1, project_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Generate realistic task data."""
        tasks = []
        
        task_templates = [
            {
                'name': 'Create REST API endpoint',
                'description': 'Implement {endpoint} endpoint with {auth} authentication',
                'complexity': 'medium',
                'estimated_duration': 120
            },
            {
                'name': 'Add user authentication',
                'description': 'Implement JWT-based authentication system with role management',
                'complexity': 'high',
                'estimated_duration': 240
            },
            {
                'name': 'Write unit tests',
                'description': 'Create comprehensive unit tests for {module} module',
                'complexity': 'low',
                'estimated_duration': 60
            },
            {
                'name': 'Implement database models',
                'description': 'Create database models for {entity} with relationships',
                'complexity': 'medium',
                'estimated_duration': 90
            },
            {
                'name': 'Add error handling',
                'description': 'Implement comprehensive error handling and logging',
                'complexity': 'low',
                'estimated_duration': 45
            }
        ]
        
        for i in range(count):
            template = self.random.choice(task_templates)
            
            task_data = {
                'id': str(uuid.uuid4()),
                'name': self._customize_task_name(template['name'], project_context),
                'description': self._customize_task_description(template['description'], project_context),
                'priority': self.random.choice(list(Priority)),
                'status': self.random.choice(list(TaskStatus)),
                'complexity': template['complexity'],
                'estimated_duration': template['estimated_duration'] + self.random.randint(-30, 30),
                'created_at': self._generate_datetime(),
                'dependencies': self._generate_task_dependencies(i, count),
                'metadata': self._generate_task_metadata(template['complexity'])
            }
            
            tasks.append(task_data)
        
        return tasks
    
    def generate_validation_data(self, authenticity_level: str = 'mixed') -> Dict[str, Any]:
        """Generate validation test data with controlled authenticity levels."""
        if authenticity_level == 'high':
            authenticity_score = self.random.uniform(80, 95)
            real_progress = self.random.uniform(75, 90)
            fake_progress = 100 - real_progress
            issues_count = self.random.randint(0, 3)
        elif authenticity_level == 'low':
            authenticity_score = self.random.uniform(30, 60)
            real_progress = self.random.uniform(25, 50)
            fake_progress = 100 - real_progress
            issues_count = self.random.randint(8, 20)
        else:  # mixed
            authenticity_score = self.random.uniform(60, 80)
            real_progress = self.random.uniform(55, 75)
            fake_progress = 100 - real_progress
            issues_count = self.random.randint(3, 10)
        
        issues = self._generate_validation_issues(issues_count)
        
        return {
            'is_authentic': authenticity_score >= 70,
            'authenticity_score': authenticity_score,
            'real_progress': real_progress,
            'fake_progress': fake_progress,
            'issues': issues,
            'quality_score': self.random.uniform(60, 95),
            'suggestions': self._generate_suggestions(issues),
            'next_actions': self._generate_next_actions(authenticity_score)
        }
    
    def generate_performance_data(self) -> Dict[str, Any]:
        """Generate performance test data."""
        return {
            'startup_time': self.random.uniform(0.5, 2.0),
            'response_time_ms': self.random.uniform(50, 500),
            'memory_usage_mb': self.random.uniform(50, 300),
            'cpu_usage_percent': self.random.uniform(5, 80),
            'throughput': self.random.uniform(5, 50),
            'success_rate': self.random.uniform(85, 99),
            'error_rate': self.random.uniform(0, 15)
        }
    
    def _generate_project_name(self) -> str:
        """Generate realistic project name."""
        if self.faker:
            return f"{self.faker.company().replace(' ', '')}_{self.faker.word().title()}"
        else:
            prefixes = ['Smart', 'Quick', 'Easy', 'Pro', 'Super', 'Ultra']
            suffixes = ['App', 'System', 'Tool', 'Platform', 'Service', 'Engine']
            return f"{self.random.choice(prefixes)}{self.random.choice(suffixes)}"
    
    def _generate_project_description(self) -> str:
        """Generate realistic project description."""
        descriptions = [
            "A modern web application for managing business processes",
            "Scalable microservice architecture for enterprise solutions",
            "API-first platform for data analytics and reporting",
            "Real-time collaboration tool with advanced features",
            "Machine learning powered recommendation system",
            "Secure cloud-native application with high availability"
        ]
        return self.random.choice(descriptions)
    
    def _generate_datetime(self) -> str:
        """Generate random datetime within reasonable range."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        random_date = start_date + timedelta(
            seconds=self.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        return random_date.isoformat()
    
    def _generate_project_features(self, project_type: str) -> List[str]:
        """Generate realistic project features based on type."""
        feature_sets = {
            'python': ['api', 'database', 'auth', 'testing', 'logging', 'caching'],
            'javascript': ['frontend', 'routing', 'state-management', 'testing', 'bundling'],
            'typescript': ['types', 'interfaces', 'decorators', 'testing', 'compilation'],
            'java': ['mvc', 'dependency-injection', 'orm', 'security', 'testing'],
            'go': ['http-server', 'concurrency', 'modules', 'testing', 'performance']
        }
        
        available_features = feature_sets.get(project_type, ['basic', 'testing'])
        num_features = self.random.randint(2, min(5, len(available_features)))
        return self.random.sample(available_features, num_features)
    
    def _customize_task_name(self, template_name: str, project_context: Optional[Dict]) -> str:
        """Customize task name based on project context."""
        if not project_context:
            return template_name
        
        framework = project_context.get('framework', '')
        project_type = project_context.get('type', '')
        
        # Simple customization
        if 'REST API' in template_name and framework:
            return template_name.replace('REST API', f'{framework.title()} API')
        
        return template_name
    
    def _customize_task_description(self, template_desc: str, project_context: Optional[Dict]) -> str:
        """Customize task description based on project context."""
        # Simple template variable replacement
        replacements = {
            '{endpoint}': self.random.choice(['user', 'product', 'order', 'payment']),
            '{auth}': self.random.choice(['JWT', 'OAuth2', 'Basic', 'API Key']),
            '{module}': self.random.choice(['user', 'auth', 'core', 'utils']),
            '{entity}': self.random.choice(['User', 'Product', 'Order', 'Category'])
        }
        
        result = template_desc
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result
    
    def _generate_task_dependencies(self, current_index: int, total_count: int) -> List[str]:
        """Generate realistic task dependencies."""
        if current_index == 0:
            return []  # First task has no dependencies
        
        # Higher index tasks can depend on earlier tasks
        max_deps = min(2, current_index)
        num_deps = self.random.randint(0, max_deps)
        
        if num_deps == 0:
            return []
        
        # Generate dependency IDs (in real scenario, these would be actual UUIDs)
        return [f"task-{i}" for i in self.random.sample(range(current_index), num_deps)]
    
    def _generate_task_metadata(self, complexity: str) -> Dict[str, Any]:
        """Generate task metadata based on complexity."""
        metadata = {
            'complexity': complexity,
            'tags': self._generate_task_tags(complexity),
            'files_to_modify': self.random.randint(1, 5 if complexity == 'high' else 3),
            'estimated_lines': self.random.randint(10, 200 if complexity == 'high' else 100)
        }
        
        if complexity == 'high':
            metadata.update({
                'requires_coordination': True,
                'security_critical': self.random.choice([True, False]),
                'performance_critical': self.random.choice([True, False])
            })
        
        return metadata
    
    def _generate_task_tags(self, complexity: str) -> List[str]:
        """Generate task tags based on complexity."""
        all_tags = ['backend', 'frontend', 'api', 'database', 'auth', 'testing', 'security', 'performance']
        
        if complexity == 'high':
            return self.random.sample(all_tags, self.random.randint(3, 5))
        elif complexity == 'medium':
            return self.random.sample(all_tags, self.random.randint(2, 3))
        else:
            return self.random.sample(all_tags, self.random.randint(1, 2))
    
    def _generate_validation_issues(self, count: int) -> List[Dict[str, Any]]:
        """Generate validation issues."""
        issue_types = list(IssueType)
        severities = list(Severity)
        
        issues = []
        for i in range(count):
            issue = {
                'type': self.random.choice(issue_types).value,
                'severity': self.random.choice(severities).value,
                'description': self._generate_issue_description(),
                'file_path': f"src/{self.random.choice(['main', 'auth', 'models', 'utils'])}.py",
                'line_number': self.random.randint(1, 100),
                'auto_fix_available': self.random.choice([True, False])
            }
            issues.append(issue)
        
        return issues
    
    def _generate_issue_description(self) -> str:
        """Generate realistic issue descriptions."""
        descriptions = [
            "TODO comment found - implementation needed",
            "Empty function with pass statement",
            "Placeholder value detected",
            "FIXME comment requires attention",
            "Hardcoded value should be configurable",
            "Exception handling incomplete",
            "Function lacks proper documentation"
        ]
        return self.random.choice(descriptions)
    
    def _generate_suggestions(self, issues: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        if not issues:
            return ["Code quality looks good"]
        
        suggestions = [
            f"Fix {len(issues)} detected issues",
            "Implement placeholder code",
            "Add proper error handling",
            "Include comprehensive documentation"
        ]
        
        return self.random.sample(suggestions, min(len(suggestions), self.random.randint(1, 3)))
    
    def _generate_next_actions(self, authenticity_score: float) -> List[str]:
        """Generate next actions based on authenticity score."""
        if authenticity_score >= 80:
            return ["validation-passed", "ready-for-review"]
        elif authenticity_score >= 60:
            return ["auto-fix-placeholders", "manual-review-recommended"]
        else:
            return ["manual-review-required", "significant-rework-needed"]


class MockComponents:
    """Factory for creating mock components."""
    
    @staticmethod
    def create_mock_project_manager() -> Mock:
        """Create mock ProjectManager with realistic behavior."""
        mock = Mock()
        mock.current_project = None
        mock.create_project = AsyncMock()
        mock.delete_project = AsyncMock()
        mock.list_projects = AsyncMock(return_value=[])
        mock.get_progress_report = AsyncMock()
        mock.validate_project_authenticity = AsyncMock()
        mock.orchestrate_development = AsyncMock()
        
        # Configure realistic return values
        mock.create_project.return_value = MockComponents._create_mock_project()
        mock.delete_project.return_value = True
        mock.get_progress_report.return_value = MockComponents._create_mock_progress_report()
        
        return mock
    
    @staticmethod
    def create_mock_ai_interface() -> AsyncMock:
        """Create mock AI interface with realistic responses."""
        mock = AsyncMock()
        mock.execute_task = AsyncMock()
        mock.batch_execute_tasks = AsyncMock()
        mock.get_performance_metrics = Mock()
        
        # Configure realistic return values
        mock.execute_task.return_value = AITaskResult(
            task_id=uuid.uuid4(),
            success=True,
            generated_content="Mock AI generated content",
            execution_time=1.5,
            metadata={'mock': True}
        )
        
        mock.get_performance_metrics.return_value = {
            'total_tasks_executed': 10,
            'success_rate': 95.0,
            'average_execution_time': 2.3,
            'cache_hit_rate': 15.0
        }
        
        return mock
    
    @staticmethod
    def create_mock_validator() -> AsyncMock:
        """Create mock ProgressValidator with configurable results."""
        mock = AsyncMock()
        mock.validate_codebase = AsyncMock()
        mock.validate_single_file = AsyncMock()
        mock.auto_fix_placeholders = AsyncMock()
        
        # Default validation result
        default_validation = ValidationResult(
            is_authentic=True,
            authenticity_score=85.0,
            real_progress=80.0,
            fake_progress=20.0,
            issues=[],
            suggestions=["Continue development"],
            next_actions=["validation-passed"]
        )
        
        mock.validate_codebase.return_value = default_validation
        mock.validate_single_file.return_value = default_validation
        mock.auto_fix_placeholders.return_value = {
            'fixed_count': 5,
            'failed_count': 1,
            'success_rate': 83.3
        }
        
        return mock
    
    @staticmethod
    def create_mock_task_engine() -> AsyncMock:
        """Create mock TaskEngine with workflow execution."""
        mock = AsyncMock()
        mock.execute_workflow = AsyncMock()
        mock.monitor_progress = AsyncMock()
        mock.register_task_executor = Mock()
        
        # Default workflow result
        mock.execute_workflow.return_value = Mock(
            success=True,
            workflow_id=uuid.uuid4(),
            tasks_executed=[uuid.uuid4(), uuid.uuid4()],
            files_generated=['src/main.py', 'tests/test_main.py'],
            total_time=5.2,
            quality_metrics={
                'quality_score': 88.5,
                'authenticity_rate': 92.0,
                'success_rate': 100.0
            }
        )
        
        mock.monitor_progress.return_value = {
            'workflow_id': str(uuid.uuid4()),
            'workflow_status': 'IN_PROGRESS',
            'task_statuses': {},
            'progress': 0.65
        }
        
        return mock
    
    @staticmethod
    def _create_mock_project() -> Mock:
        """Create mock project with realistic properties."""
        project = Mock()
        project.id = uuid.uuid4()
        project.name = "Mock Project"
        project.description = "A mock project for testing"
        project.state = ProjectState.ACTIVE
        project.path = Path("/tmp/mock_project")
        project.created_at = datetime.now()
        project.updated_at = datetime.now()
        project.progress = ProgressMetrics(
            real_progress=75.0,
            fake_progress=25.0,
            authenticity_rate=85.0,
            quality_score=82.5,
            tasks_completed=8,
            tasks_total=10
        )
        project.validation_history = []
        project.generated_files = set(['src/main.py', 'tests/test_main.py'])
        project.modified_files = set(['README.md'])
        
        return project
    
    @staticmethod
    def _create_mock_progress_report() -> Mock:
        """Create mock progress report."""
        report = Mock()
        report.project_id = uuid.uuid4()
        report.metrics = ProgressMetrics(
            real_progress=80.0,
            fake_progress=20.0,
            authenticity_rate=88.5,
            quality_score=85.2
        )
        report.validation = ValidationResult(
            is_authentic=True,
            authenticity_score=88.5,
            real_progress=80.0,
            fake_progress=20.0
        )
        report.recent_activities = [
            "Generated 3 files",
            "Modified 2 files",
            "Completed validation check"
        ]
        report.performance_data = {
            'total_workflows': 5,
            'completed_workflows': 4,
            'validation_checks': 12
        }
        
        return report


class TestFileSystem:
    """Utilities for creating temporary test file systems."""
    
    @staticmethod
    def create_test_project(base_path: Path, project_data: Dict[str, Any]) -> Path:
        """Create a realistic test project structure."""
        project_path = base_path / project_data['name'].lower().replace(' ', '_')
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create basic structure
        (project_path / "src").mkdir(exist_ok=True)
        (project_path / "tests").mkdir(exist_ok=True)
        (project_path / "docs").mkdir(exist_ok=True)
        
        # Create files based on project type
        project_type = project_data.get('type', 'python')
        
        if project_type == 'python':
            TestFileSystem._create_python_project_files(project_path)
        elif project_type == 'javascript':
            TestFileSystem._create_javascript_project_files(project_path)
        elif project_type == 'typescript':
            TestFileSystem._create_typescript_project_files(project_path)
        
        # Create README
        readme_content = f"# {project_data['name']}\n\n{project_data.get('description', 'A test project')}\n"
        (project_path / "README.md").write_text(readme_content)
        
        return project_path
    
    @staticmethod
    def _create_python_project_files(project_path: Path) -> None:
        """Create Python-specific files."""
        # Main module
        (project_path / "src" / "__init__.py").write_text('"""Main package."""\n')
        (project_path / "src" / "main.py").write_text(
            'def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n'
        )
        
        # Requirements
        (project_path / "requirements.txt").write_text('requests>=2.25.1\npydantic>=1.8.0\n')
        
        # Test files
        (project_path / "tests" / "__init__.py").write_text('')
        (project_path / "tests" / "test_main.py").write_text(
            'import unittest\nfrom src.main import main\n\nclass TestMain(unittest.TestCase):\n    def test_main(self):\n        # TODO: Add real tests\n        pass\n'
        )
    
    @staticmethod
    def _create_javascript_project_files(project_path: Path) -> None:
        """Create JavaScript-specific files."""
        # Package.json
        package_json = {
            "name": project_path.name,
            "version": "1.0.0",
            "description": "Test JavaScript project",
            "main": "src/index.js",
            "scripts": {
                "start": "node src/index.js",
                "test": "jest"
            },
            "dependencies": {
                "express": "^4.17.1"
            },
            "devDependencies": {
                "jest": "^27.0.0"
            }
        }
        
        (project_path / "package.json").write_text(json.dumps(package_json, indent=2))
        
        # Main file
        (project_path / "src" / "index.js").write_text(
            'console.log("Hello, World!");\n'
        )
        
        # Test file
        (project_path / "tests" / "index.test.js").write_text(
            'test("sample test", () => {\n  expect(true).toBe(true);\n});\n'
        )
    
    @staticmethod
    def _create_typescript_project_files(project_path: Path) -> None:
        """Create TypeScript-specific files."""
        # Package.json
        package_json = {
            "name": project_path.name,
            "version": "1.0.0",
            "description": "Test TypeScript project",
            "main": "dist/index.js",
            "scripts": {
                "build": "tsc",
                "start": "node dist/index.js",
                "test": "jest"
            },
            "dependencies": {
                "express": "^4.17.1"
            },
            "devDependencies": {
                "typescript": "^4.0.0",
                "@types/express": "^4.17.0",
                "jest": "^27.0.0"
            }
        }
        
        (project_path / "package.json").write_text(json.dumps(package_json, indent=2))
        
        # TypeScript config
        tsconfig = {
            "compilerOptions": {
                "target": "es2018",
                "module": "commonjs",
                "outDir": "./dist",
                "rootDir": "./src",
                "strict": True,
                "esModuleInterop": True
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules", "dist"]
        }
        
        (project_path / "tsconfig.json").write_text(json.dumps(tsconfig, indent=2))
        
        # Main TypeScript file
        (project_path / "src" / "index.ts").write_text(
            'console.log("Hello, TypeScript!");\n'
        )


class TestDatabase:
    """Utilities for database testing."""
    
    @staticmethod
    def create_test_db_session():
        """Create test database session (SQLite in-memory)."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            engine = create_engine("sqlite:///:memory:", echo=False)
            TestSession = sessionmaker(bind=engine)
            return TestSession()
        except ImportError:
            return Mock()  # Fallback mock session
    
    @staticmethod
    def populate_test_data(session, generator: TestDataGenerator):
        """Populate test database with generated data."""
        # This would populate the database with test data
        # Implementation depends on actual database models
        pass


# Pytest fixtures using the utilities above

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture
def data_generator(test_config):
    """Provide test data generator."""
    return TestDataGenerator(seed=test_config.seed)


@pytest.fixture
def mock_project_manager():
    """Provide mock project manager."""
    return MockComponents.create_mock_project_manager()


@pytest.fixture
def mock_ai_interface():
    """Provide mock AI interface."""
    return MockComponents.create_mock_ai_interface()


@pytest.fixture
def mock_validator():
    """Provide mock validator."""
    return MockComponents.create_mock_validator()


@pytest.fixture
def mock_task_engine():
    """Provide mock task engine."""
    return MockComponents.create_mock_task_engine()


@pytest.fixture
def temp_test_workspace(test_config):
    """Provide temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        yield workspace
        
        # Cleanup handled automatically by TemporaryDirectory


@pytest.fixture
def sample_project_data(data_generator):
    """Provide sample project data."""
    return data_generator.generate_project_data(count=1)[0]


@pytest.fixture
def sample_task_data(data_generator):
    """Provide sample task data."""
    return data_generator.generate_task_data(count=3)


@pytest.fixture
def sample_validation_data(data_generator):
    """Provide sample validation data."""
    return data_generator.generate_validation_data('mixed')


@pytest.fixture
def test_project_structure(temp_test_workspace, sample_project_data):
    """Create realistic test project structure."""
    project_path = TestFileSystem.create_test_project(
        temp_test_workspace, sample_project_data
    )
    return project_path


@pytest.fixture
def test_db_session():
    """Provide test database session."""
    session = TestDatabase.create_test_db_session()
    yield session
    
    # Cleanup
    if hasattr(session, 'close'):
        session.close()


# Utility functions for test scenarios

def create_realistic_test_scenario(scenario_type: str, **kwargs) -> Dict[str, Any]:
    """Create realistic test scenarios for different use cases."""
    generator = TestDataGenerator()
    
    scenarios = {
        'new_project_creation': {
            'project_data': generator.generate_project_data(1)[0],
            'expected_files': ['README.md', 'src/', 'tests/'],
            'expected_duration': 5.0,
            'success_criteria': ['project_created', 'files_generated', 'state_saved']
        },
        'complex_development_workflow': {
            'project_data': generator.generate_project_data(1)[0],
            'task_data': generator.generate_task_data(8),
            'validation_data': generator.generate_validation_data('mixed'),
            'expected_duration': 30.0,
            'success_criteria': ['tasks_completed', 'validation_passed', 'files_modified']
        },
        'validation_intensive': {
            'project_data': generator.generate_project_data(1)[0],
            'validation_data': generator.generate_validation_data('low'),
            'expected_issues': 15,
            'auto_fix_success_rate': 0.8,
            'success_criteria': ['issues_detected', 'fixes_applied', 'quality_improved']
        },
        'performance_stress': {
            'concurrent_users': kwargs.get('concurrent_users', 25),
            'task_count': kwargs.get('task_count', 100),
            'duration_limit': kwargs.get('duration_limit', 60),
            'success_criteria': ['throughput_met', 'response_time_ok', 'error_rate_low']
        }
    }
    
    return scenarios.get(scenario_type, {})


def assert_performance_requirements(metrics: Dict[str, Any], requirements: Dict[str, Any]):
    """Assert that performance metrics meet requirements."""
    for requirement_name, requirement_value in requirements.items():
        if requirement_name in metrics:
            metric_value = metrics[requirement_name]
            
            # Different assertion logic based on requirement type
            if 'time' in requirement_name or 'duration' in requirement_name:
                assert metric_value <= requirement_value, f"{requirement_name}: {metric_value} > {requirement_value}"
            elif 'rate' in requirement_name or 'percent' in requirement_name:
                assert metric_value >= requirement_value, f"{requirement_name}: {metric_value} < {requirement_value}"
            elif 'memory' in requirement_name:
                assert metric_value <= requirement_value, f"{requirement_name}: {metric_value} > {requirement_value}"
            else:
                # Generic comparison
                assert metric_value >= requirement_value, f"{requirement_name}: {metric_value} < {requirement_value}"


def validate_test_data_quality(data: Dict[str, Any], data_type: str) -> bool:
    """Validate the quality of generated test data."""
    if data_type == 'project':
        required_fields = ['id', 'name', 'description', 'type', 'state']
        return all(field in data for field in required_fields)
    
    elif data_type == 'task':
        required_fields = ['id', 'name', 'description', 'priority', 'status']
        return all(field in data for field in required_fields)
    
    elif data_type == 'validation':
        required_fields = ['authenticity_score', 'real_progress', 'fake_progress', 'issues']
        return all(field in data for field in required_fields)
    
    return False


if __name__ == "__main__":
    # Example usage and testing
    generator = TestDataGenerator()
    
    # Generate sample data
    project_data = generator.generate_project_data(3)
    task_data = generator.generate_task_data(5)
    validation_data = generator.generate_validation_data('mixed')
    
    print("Generated project data:")
    for project in project_data:
        print(f"  - {project['name']}: {project['type']} ({project['framework']})")
    
    print(f"\nGenerated {len(task_data)} tasks")
    print(f"Validation authenticity score: {validation_data['authenticity_score']:.1f}%")
    print(f"Found {len(validation_data['issues'])} issues")
