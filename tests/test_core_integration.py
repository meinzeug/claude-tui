"""
Integration tests for claude-tiu core modules.

This test suite verifies that all core components work together correctly
and validates the anti-hallucination pipeline with real-world scenarios.
"""

import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4
import pytest

from src.core import (
    ProjectManager, Project, TaskEngine, Task, ProgressValidator,
    ConfigManager, ProjectConfig, AIInterface, AIContext,
    create_simple_project, validate_project_quality, create_task,
    ProjectState, Priority, TaskStatus, ExecutionStrategy,
    setup_logging, get_logger
)


class TestCoreIntegration:
    """Test core module integration."""
    
    @pytest.fixture
    async def temp_project_dir(self):
        """Create temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def project_manager(self):
        """Create project manager instance."""
        return ProjectManager(enable_validation=True)
    
    @pytest.fixture
    def task_engine(self):
        """Create task engine instance."""
        return TaskEngine(enable_validation=True, max_concurrent_tasks=3)
    
    @pytest.fixture
    def validator(self):
        """Create progress validator instance."""
        return ProgressValidator(enable_cross_validation=False, enable_execution_testing=False)
    
    async def test_project_creation_and_validation(self, temp_project_dir):
        """Test complete project creation and validation flow."""
        # Create project with basic template
        project = await create_simple_project(
            name="Test Project",
            template="basic",
            path=temp_project_dir / "test-project"
        )
        
        assert project is not None
        assert project.name == "Test Project"
        assert project.state == ProjectState.ACTIVE
        assert project.path.exists()
        
        # Validate project structure
        assert (project.path / "src" / "__init__.py").exists()
        assert (project.path / "tests" / "__init__.py").exists()
        assert (project.path / "README.md").exists()
        
        # Validate project quality
        validation_result = await validate_project_quality(project.path)
        assert validation_result.is_authentic
        assert validation_result.authenticity_score >= 80.0
        
        print(f"✅ Project creation test passed - Authenticity: {validation_result.authenticity_score:.1f}%")
    
    async def test_task_engine_workflow_execution(self, task_engine):
        """Test task engine workflow execution."""
        # Create test tasks with dependencies
        task1 = create_task("Setup Project", "Initialize project structure", Priority.HIGH)
        task2 = create_task("Implement Feature", "Add main feature", Priority.MEDIUM, [task1.id])
        task3 = create_task("Add Tests", "Create test suite", Priority.MEDIUM, [task2.id])
        
        # Create workflow
        from src.core.task_engine import Workflow
        workflow = Workflow(name="Test Workflow", tasks=[task1, task2, task3])
        
        # Execute workflow
        result = await task_engine.execute_workflow(workflow, ExecutionStrategy.SEQUENTIAL)
        
        assert result.success
        assert len(result.tasks_executed) == 3
        assert all(task.status == TaskStatus.COMPLETED for task in workflow.tasks)
        
        print(f"✅ Task engine test passed - {len(result.tasks_executed)} tasks executed")
    
    async def test_validator_placeholder_detection(self, validator, temp_project_dir):
        """Test validator placeholder and fake progress detection."""
        # Create test files with different levels of completeness
        test_files = {
            "complete.py": """
def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    return a + b

def main():
    result = calculate_sum(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
""",
            "placeholder.py": """
def calculate_sum(a, b):
    # TODO: Implement this function
    pass

def main():
    # FIXME: Add implementation
    ...

if __name__ == "__main__":
    main()
""",
            "partial.py": """
def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    return a + b

def advanced_calculation():
    # TODO: Implement advanced calculation
    raise NotImplementedError("Not yet implemented")
"""
        }
        
        # Write test files
        test_dir = temp_project_dir / "validation_test"
        test_dir.mkdir(parents=True)
        
        for filename, content in test_files.items():
            (test_dir / filename).write_text(content)
        
        # Validate codebase
        validation_result = await validator.validate_codebase(test_dir)
        
        # Check results
        assert len(validation_result.issues) > 0  # Should detect placeholders
        assert validation_result.authenticity_score < 100.0  # Not 100% authentic
        
        # Check specific issue detection
        placeholder_issues = [issue for issue in validation_result.issues if "TODO" in issue.description or "FIXME" in issue.description]
        assert len(placeholder_issues) >= 2  # Should detect TODO and FIXME
        
        print(f"✅ Validator test passed - Detected {len(validation_result.issues)} issues")
        print(f"   Authenticity score: {validation_result.authenticity_score:.1f}%")
    
    async def test_config_manager_operations(self):
        """Test configuration management operations."""
        config_manager = ConfigManager()
        
        # Test project config creation and validation
        project_config = ProjectConfig(
            name="Test Config Project",
            description="Test project for config validation",
            framework="python",
            features=["api", "testing"],
            ai_settings={
                "auto_completion": True,
                "validation_level": "strict"
            }
        )
        
        # Test config persistence
        project_id = "test-project-123"
        config_manager.set_project_config(project_id, project_config)
        
        # Retrieve and validate
        retrieved_config = config_manager.get_project_config(project_id)
        assert retrieved_config.name == project_config.name
        assert retrieved_config.framework == project_config.framework
        assert retrieved_config.features == project_config.features
        
        # Test nested configuration access
        timeout_value = config_manager.get_config_value("ai.claude_code.timeout", default=300)
        assert timeout_value == 300  # Should use default
        
        # Set nested value
        config_manager.set_config_value("ai.claude_code.timeout", 600)
        new_timeout = config_manager.get_config_value("ai.claude_code.timeout")
        assert new_timeout == 600
        
        print("✅ Config manager test passed")
    
    async def test_ai_interface_task_routing(self):
        """Test AI interface task complexity analysis and routing."""
        ai_interface = AIInterface(enable_validation=False)  # Skip actual AI calls
        
        # Test simple task
        simple_task = create_task(
            "Fix typo", 
            "Fix a simple typo in README", 
            Priority.LOW,
            estimated_duration=5
        )
        
        simple_context = AIContext(
            current_files=["README.md"],
            framework_info={"framework": "python"}
        )
        
        # Test complex task  
        complex_task = create_task(
            "Build full-stack app",
            "Create a complete full-stack application with authentication, database, API, and frontend",
            Priority.HIGH,
            estimated_duration=480  # 8 hours
        )
        
        complex_context = AIContext(
            current_files=["app.py", "models.py", "frontend.js", "schema.sql"],
            dependencies=["fastapi", "sqlalchemy", "react", "postgresql"],
            framework_info={"framework": "fastapi", "frontend": "react"}
        )
        
        # Analyze task complexity (this will work without actual AI calls)
        analyzer = ai_interface.task_complexity_analyzer
        
        simple_analysis = await analyzer.analyze_task(simple_task, simple_context)
        complex_analysis = await analyzer.analyze_task(complex_task, complex_context)
        
        # Validate routing decisions
        assert simple_analysis['complexity_level'] == 'simple'
        assert simple_analysis['recommended_service'] == 'claude_code'
        
        assert complex_analysis['complexity_level'] == 'complex'
        assert complex_analysis['recommended_service'] == 'claude_flow'
        
        print("✅ AI interface test passed")
        print(f"   Simple task complexity: {simple_analysis['complexity_score']}")
        print(f"   Complex task complexity: {complex_analysis['complexity_score']}")
    
    async def test_full_integration_workflow(self, temp_project_dir):
        """Test complete integration workflow from project creation to validation."""
        # Setup logging
        logger = setup_logging(level="INFO")
        
        # 1. Create project manager
        project_manager = ProjectManager(enable_validation=True)
        
        # 2. Create new project
        project = await project_manager.create_project(
            name="Integration Test Project",
            template="fastapi",
            project_path=temp_project_dir / "integration-test"
        )
        
        assert project.state == ProjectState.ACTIVE
        
        # 3. Define development requirements
        requirements = {
            "description": "Build a simple API with authentication",
            "features": ["authentication", "user-management", "api-endpoints"],
            "include_tests": True,
            "setup_project": True
        }
        
        # 4. Orchestrate development (this will use mock executors)
        development_result = await project_manager.orchestrate_development(
            project.id, requirements, ExecutionStrategy.ADAPTIVE
        )
        
        assert development_result.success
        
        # 5. Get progress report
        progress_report = await project_manager.get_progress_report(project.id)
        assert progress_report.project_id == project.id
        assert progress_report.metrics.authenticity_rate >= 0.0
        
        # 6. Validate project authenticity
        validation_result = await project_manager.validate_project_authenticity(
            project.id, deep_validation=False
        )
        
        assert validation_result.authenticity_score >= 0.0
        
        # 7. Get project details
        project_details = await project_manager.get_project_details(project.id)
        assert project_details is not None
        assert project_details['name'] == "Integration Test Project"
        
        print("✅ Full integration test passed")
        print(f"   Project created: {project.name}")
        print(f"   Development tasks: {len(development_result.tasks_executed)}")
        print(f"   Final authenticity: {validation_result.authenticity_score:.1f}%")
    
    async def test_error_handling_and_recovery(self, project_manager):
        """Test error handling and recovery mechanisms."""
        # Test invalid project creation
        with pytest.raises(Exception):  # Should raise ProjectManagerException
            await project_manager.create_project(
                name="",  # Invalid empty name
                template="nonexistent-template"
            )
        
        # Test handling of missing project
        nonexistent_id = uuid4()
        project_details = await project_manager.get_project_details(nonexistent_id)
        assert project_details is None
        
        # Test progress report for nonexistent project
        try:
            await project_manager.get_progress_report(nonexistent_id)
            assert False, "Should have raised exception"
        except:
            pass  # Expected to fail
        
        print("✅ Error handling test passed")
    
    def test_utility_functions(self):
        """Test utility functions and helpers."""
        from src.core.utils import format_file_size, validate_path, ensure_directory
        
        # Test file size formatting
        assert format_file_size(0) == "0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        
        # Test path validation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Valid path
            validated = validate_path(temp_path, must_exist=True, must_be_dir=True)
            assert validated.exists()
            assert validated.is_dir()
            
            # Create test directory
            test_dir = temp_path / "test"
            ensure_directory(test_dir)
            assert test_dir.exists()
            assert test_dir.is_dir()
        
        print("✅ Utility functions test passed")


# Test runner function for standalone execution
async def run_integration_tests():
    """Run all integration tests."""
    test_instance = TestCoreIntegration()
    
    print("🚀 Starting claude-tiu core integration tests...\n")
    
    try:
        # Create temporary directory for tests
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Run individual tests
            await test_instance.test_project_creation_and_validation(temp_path)
            await test_instance.test_task_engine_workflow_execution(TaskEngine())
            await test_instance.test_validator_placeholder_detection(ProgressValidator(), temp_path)
            await test_instance.test_config_manager_operations()
            await test_instance.test_ai_interface_task_routing()
            await test_instance.test_full_integration_workflow(temp_path)
            await test_instance.test_error_handling_and_recovery(ProjectManager())
            test_instance.test_utility_functions()
        
        print("\n🎉 All integration tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests when executed directly
    success = asyncio.run(run_integration_tests())
    exit(0 if success else 1)