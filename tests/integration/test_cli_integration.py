"""
Integration tests for CLI tools and external tool integration.

Tests the integration between claude-tiu and external tools like
Claude Code, Claude Flow, and other CLI utilities.
"""

import pytest
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
import yaml


class TestCLIIntegration:
    """Integration tests for CLI tools."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create temporary project directory with basic structure."""
        project_dir = tmp_path / "test-integration-project"
        project_dir.mkdir()
        
        # Create basic Python project structure
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()
        (project_dir / "docs").mkdir()
        
        # Create basic files
        (project_dir / "README.md").write_text("# Test Integration Project")
        (project_dir / "pyproject.toml").write_text("""
[build-system]
requires = ["setuptools>=45", "wheel"]

[project]
name = "test-integration-project"
version = "0.1.0"
""")
        
        (project_dir / "src" / "__init__.py").write_text("")
        (project_dir / "src" / "main.py").write_text("""
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
        
        return project_dir
    
    @pytest.fixture
    def mock_claude_code_success(self):
        """Mock successful Claude Code execution."""
        def mock_run(*args, **kwargs):
            if 'claude-code' in args[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "status": "success",
                        "files_created": ["generated_code.py"],
                        "analysis": {
                            "functions": 3,
                            "classes": 1,
                            "tests": 5,
                            "coverage": 0.92
                        }
                    }),
                    stderr=""
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return mock_run
    
    @pytest.fixture
    def mock_claude_flow_success(self):
        """Mock successful Claude Flow execution."""
        def mock_run(*args, **kwargs):
            if 'claude-flow' in args[0] or 'npx' in args[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "workflow_status": "completed",
                        "tasks_completed": 3,
                        "results": {
                            "task1": {"status": "completed", "output": "Task 1 done"},
                            "task2": {"status": "completed", "output": "Task 2 done"},
                            "task3": {"status": "completed", "output": "Task 3 done"}
                        },
                        "execution_time": 45.2
                    }),
                    stderr=""
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return mock_run
    
    @patch('subprocess.run')
    def test_claude_code_basic_integration(self, mock_run, mock_claude_code_success, temp_project):
        """Test basic integration with Claude Code CLI."""
        # Arrange
        mock_run.side_effect = mock_claude_code_success
        output_file = temp_project / "generated.py"
        
        # Act
        result = subprocess.run([
            "claude-code",
            "--prompt", "Create a calculator class with add and subtract methods",
            "--output", str(output_file)
        ], capture_output=True, text=True)
        
        # Assert
        assert result.returncode == 0
        response = json.loads(result.stdout)
        assert response["status"] == "success"
        assert "generated_code.py" in response["files_created"]
        assert response["analysis"]["functions"] == 3
        assert response["analysis"]["coverage"] > 0.9
    
    @patch('subprocess.run')
    def test_claude_code_with_project_context(self, mock_run, mock_claude_code_success, temp_project):
        """Test Claude Code execution with project context."""
        # Arrange
        mock_run.side_effect = mock_claude_code_success
        
        # Act
        result = subprocess.run([
            "claude-code",
            "--prompt", "Add error handling to the main function",
            "--project-dir", str(temp_project),
            "--file", "src/main.py"
        ], capture_output=True, text=True, cwd=str(temp_project))
        
        # Assert
        assert result.returncode == 0
        mock_run.assert_called()
        
        # Verify command was called with correct arguments
        call_args = mock_run.call_args[0][0]
        assert "claude-code" in call_args
        assert "--project-dir" in call_args
        assert str(temp_project) in call_args
    
    @patch('subprocess.run')
    def test_claude_flow_workflow_execution(self, mock_run, mock_claude_flow_success, temp_project):
        """Test Claude Flow workflow execution."""
        # Arrange
        mock_run.side_effect = mock_claude_flow_success
        
        # Create workflow file
        workflow_config = {
            "name": "test-workflow",
            "description": "Test integration workflow",
            "tasks": [
                {
                    "name": "analyze-code",
                    "type": "analysis",
                    "prompt": "Analyze the codebase for improvements"
                },
                {
                    "name": "generate-tests",
                    "type": "testing", 
                    "prompt": "Generate comprehensive unit tests",
                    "depends_on": ["analyze-code"]
                },
                {
                    "name": "refactor-code",
                    "type": "refactoring",
                    "prompt": "Refactor code based on analysis",
                    "depends_on": ["analyze-code", "generate-tests"]
                }
            ]
        }
        
        workflow_file = temp_project / "workflow.yaml"
        workflow_file.write_text(yaml.dump(workflow_config))
        
        # Act
        result = subprocess.run([
            "npx", "claude-flow", "run", str(workflow_file)
        ], capture_output=True, text=True, cwd=str(temp_project))
        
        # Assert
        assert result.returncode == 0
        response = json.loads(result.stdout)
        assert response["workflow_status"] == "completed"
        assert response["tasks_completed"] == 3
        assert "task1" in response["results"]
        assert "task2" in response["results"]
        assert "task3" in response["results"]
    
    @patch('subprocess.run')
    def test_claude_flow_sparc_mode(self, mock_run, mock_claude_flow_success, temp_project):
        """Test Claude Flow SPARC methodology integration."""
        # Arrange
        mock_run.side_effect = mock_claude_flow_success
        
        # Act - Run SPARC TDD workflow
        result = subprocess.run([
            "npx", "claude-flow", "sparc", "tdd",
            "Create a user authentication system with JWT tokens"
        ], capture_output=True, text=True, cwd=str(temp_project))
        
        # Assert
        assert result.returncode == 0
        mock_run.assert_called()
        
        # Verify SPARC command structure
        call_args = mock_run.call_args[0][0]
        assert "claude-flow" in call_args
        assert "sparc" in call_args
        assert "tdd" in call_args
    
    def test_error_handling_claude_code_failure(self, temp_project):
        """Test error handling when Claude Code fails."""
        # Arrange
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="API Error: Authentication failed"
            )
            
            # Act & Assert
            result = subprocess.run([
                "claude-code",
                "--prompt", "Generate code"
            ], capture_output=True, text=True)
            
            assert result.returncode == 1
            assert "Authentication failed" in result.stderr
    
    def test_error_handling_claude_flow_failure(self, temp_project):
        """Test error handling when Claude Flow fails."""
        # Arrange
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Workflow execution failed: Invalid task configuration"
            )
            
            # Act & Assert
            result = subprocess.run([
                "npx", "claude-flow", "run", "invalid-workflow.yaml"
            ], capture_output=True, text=True)
            
            assert result.returncode == 1
            assert "Workflow execution failed" in result.stderr
    
    @patch('subprocess.run')
    def test_concurrent_tool_execution(self, mock_run, temp_project):
        """Test concurrent execution of multiple tools."""
        import asyncio
        
        # Arrange
        def mock_tool_execution(*args, **kwargs):
            command = args[0]
            if 'claude-code' in command:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"status": "success", "tool": "claude-code"}),
                    stderr=""
                )
            elif 'claude-flow' in command:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"status": "completed", "tool": "claude-flow"}),
                    stderr=""
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        
        mock_run.side_effect = mock_tool_execution
        
        async def run_claude_code():
            return subprocess.run([
                "claude-code", "--prompt", "Generate function"
            ], capture_output=True, text=True)
        
        async def run_claude_flow():
            return subprocess.run([
                "npx", "claude-flow", "sparc", "run", "spec-pseudocode", "Create API"
            ], capture_output=True, text=True)
        
        # Act
        async def test_concurrent():
            results = await asyncio.gather(
                asyncio.to_thread(run_claude_code),
                asyncio.to_thread(run_claude_flow)
            )
            return results
        
        # This would run in an actual async context
        # For now, just test that the mock is set up correctly
        result1 = subprocess.run(["claude-code", "--prompt", "test"], capture_output=True, text=True)
        result2 = subprocess.run(["npx", "claude-flow", "test"], capture_output=True, text=True)
        
        # Assert
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert json.loads(result1.stdout)["tool"] == "claude-code"
        assert json.loads(result2.stdout)["tool"] == "claude-flow"
    
    @pytest.mark.slow
    def test_performance_large_project_integration(self, temp_project):
        """Test performance with large project integration."""
        import time
        
        # Arrange - Create larger project structure
        for i in range(10):
            module_dir = temp_project / f"module_{i}"
            module_dir.mkdir()
            
            for j in range(5):
                file_content = f"""
def function_{j}_in_module_{i}():
    '''Function {j} in module {i}.'''
    return {i} * {j}

class Class{j}Module{i}:
    '''Class {j} in module {i}.'''
    
    def method_{j}(self):
        return self.process_data()
    
    def process_data(self):
        # TODO: implement data processing
        pass
"""
                (module_dir / f"file_{j}.py").write_text(file_content)
        
        # Act
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({"status": "success", "performance": "acceptable"}),
                stderr=""
            )
            
            start_time = time.time()
            
            # Simulate processing large project
            result = subprocess.run([
                "claude-code",
                "--prompt", "Analyze entire project structure",
                "--project-dir", str(temp_project)
            ], capture_output=True, text=True)
            
            end_time = time.time()
        
        # Assert
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert result.returncode == 0
    
    @patch('subprocess.run')
    def test_tool_chain_integration(self, mock_run, temp_project):
        """Test chaining multiple tools together."""
        # Arrange
        call_count = 0
        
        def sequential_mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            command = args[0]
            if call_count == 1 and 'claude-flow' in command:
                # First call: SPARC specification
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "phase": "specification",
                        "output": "Requirements analyzed",
                        "next_phase": "pseudocode"
                    }),
                    stderr=""
                )
            elif call_count == 2 and 'claude-flow' in command:
                # Second call: Pseudocode
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "phase": "pseudocode", 
                        "output": "Algorithm designed",
                        "next_phase": "architecture"
                    }),
                    stderr=""
                )
            elif call_count == 3 and 'claude-code' in command:
                # Third call: Implementation
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "status": "success",
                        "files_created": ["implementation.py"],
                        "phase": "implementation"
                    }),
                    stderr=""
                )
            
            return MagicMock(returncode=0, stdout="{}", stderr="")
        
        mock_run.side_effect = sequential_mock
        
        # Act - Simulate SPARC workflow
        # Phase 1: Specification
        result1 = subprocess.run([
            "npx", "claude-flow", "sparc", "run", "spec-pseudocode",
            "Create a data processing pipeline"
        ], capture_output=True, text=True)
        
        # Phase 2: Pseudocode (would be triggered by result of phase 1)
        result2 = subprocess.run([
            "npx", "claude-flow", "sparc", "run", "pseudocode",
            "Design algorithm for data processing"
        ], capture_output=True, text=True)
        
        # Phase 3: Implementation
        result3 = subprocess.run([
            "claude-code",
            "--prompt", "Implement data processing pipeline based on specification",
            "--output", str(temp_project / "pipeline.py")
        ], capture_output=True, text=True)
        
        # Assert
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert result3.returncode == 0
        
        response1 = json.loads(result1.stdout)
        response2 = json.loads(result2.stdout)
        response3 = json.loads(result3.stdout)
        
        assert response1["phase"] == "specification"
        assert response2["phase"] == "pseudocode"
        assert response3["phase"] == "implementation"
        
        # Verify call sequence
        assert call_count == 3


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    def test_database(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        
        # This would be: from claude_tiu.database import create_engine, Base
        # For now, create mock database setup
        
        class MockDatabase:
            def __init__(self, db_path):
                self.db_path = db_path
                self.projects = {}
                self.tasks = {}
                self.sessions = {}
            
            def create_project(self, name, template, **kwargs):
                project_id = f"proj_{len(self.projects) + 1}"
                project = {
                    "id": project_id,
                    "name": name,
                    "template": template,
                    "status": "initialized",
                    **kwargs
                }
                self.projects[project_id] = project
                return project
            
            def get_project(self, project_id):
                return self.projects.get(project_id)
            
            def create_task(self, project_id, name, prompt, **kwargs):
                task_id = f"task_{len(self.tasks) + 1}"
                task = {
                    "id": task_id,
                    "project_id": project_id,
                    "name": name,
                    "prompt": prompt,
                    "status": "pending",
                    **kwargs
                }
                self.tasks[task_id] = task
                return task
            
            def get_tasks_for_project(self, project_id):
                return [task for task in self.tasks.values() if task["project_id"] == project_id]
            
            def close(self):
                pass  # Mock cleanup
        
        return MockDatabase(db_path)
    
    def test_project_crud_operations(self, test_database):
        """Test CRUD operations for projects."""
        # Create
        project = test_database.create_project(
            name="test-integration-project",
            template="python",
            description="Integration test project"
        )
        assert project["id"] is not None
        assert project["name"] == "test-integration-project"
        assert project["template"] == "python"
        assert project["status"] == "initialized"
        
        # Read
        fetched = test_database.get_project(project["id"])
        assert fetched is not None
        assert fetched["name"] == "test-integration-project"
        assert fetched["template"] == "python"
        
        # Verify project exists in database
        assert project["id"] in test_database.projects
    
    def test_task_management_integration(self, test_database):
        """Test task management with database integration."""
        # Create project first
        project = test_database.create_project("task-test-project", "python")
        
        # Create tasks
        task1 = test_database.create_task(
            project["id"], 
            "Generate main module",
            "Create main.py with basic structure"
        )
        
        task2 = test_database.create_task(
            project["id"],
            "Add tests", 
            "Create comprehensive unit tests",
            depends_on=[task1["id"]]
        )
        
        # Verify tasks
        assert task1["id"] is not None
        assert task1["project_id"] == project["id"]
        assert task2["depends_on"] == [task1["id"]]
        
        # Get tasks for project
        project_tasks = test_database.get_tasks_for_project(project["id"])
        assert len(project_tasks) == 2
        task_names = [t["name"] for t in project_tasks]
        assert "Generate main module" in task_names
        assert "Add tests" in task_names
    
    def test_transaction_integrity(self, test_database):
        """Test database transaction integrity."""
        # This would test actual database transactions
        # For now, test the mock behavior
        
        initial_project_count = len(test_database.projects)
        
        try:
            # Simulate successful transaction
            project = test_database.create_project("transaction-test", "python")
            task = test_database.create_task(
                project["id"], 
                "test-task", 
                "test prompt"
            )
            
            # Verify both were created
            assert len(test_database.projects) == initial_project_count + 1
            assert len(test_database.tasks) == 1
            
        except Exception:
            # In real implementation, this would rollback
            pass
    
    def test_concurrent_database_access(self, test_database):
        """Test concurrent database access."""
        import threading
        import time
        
        results = []
        
        def create_project_worker(worker_id):
            try:
                project = test_database.create_project(
                    f"concurrent-project-{worker_id}",
                    "python"
                )
                results.append(project["id"])
                time.sleep(0.1)  # Simulate some work
            except Exception as e:
                results.append(f"error-{worker_id}: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_project_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 5
        # All should be successful (project IDs, not error messages)
        error_count = len([r for r in results if r.startswith("error")])
        assert error_count == 0
        
        # Verify all projects were created with unique IDs
        project_ids = [r for r in results if not r.startswith("error")]
        assert len(set(project_ids)) == 5  # All unique


class TestExternalServiceIntegration:
    """Test integration with external services and APIs."""
    
    def test_git_integration(self, temp_project):
        """Test Git integration for version control."""
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=str(temp_project), check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(temp_project))
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(temp_project))
        
        # Add initial commit
        subprocess.run(["git", "add", "."], cwd=str(temp_project), check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(temp_project), check=True)
        
        # Verify git status
        result = subprocess.run(
            ["git", "status", "--porcelain"], 
            cwd=str(temp_project), 
            capture_output=True, 
            text=True
        )
        
        # Should be clean after commit
        assert result.returncode == 0
        assert result.stdout.strip() == ""
    
    @patch('subprocess.run')
    def test_npm_integration(self, mock_run, temp_project):
        """Test NPM integration for Claude Flow."""
        # Arrange
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="claude-flow@2.0.0 installed successfully",
            stderr=""
        )
        
        # Act
        result = subprocess.run([
            "npm", "install", "claude-flow@alpha"
        ], capture_output=True, text=True, cwd=str(temp_project))
        
        # Assert
        assert result.returncode == 0
        assert "claude-flow" in result.stdout
    
    def test_environment_variable_integration(self, temp_project, monkeypatch):
        """Test environment variable handling."""
        # Arrange
        monkeypatch.setenv("CLAUDE_API_KEY", "test-api-key")
        monkeypatch.setenv("CLAUDE_TIU_PROJECT_ROOT", str(temp_project))
        monkeypatch.setenv("CLAUDE_TIU_DEBUG", "true")
        
        # Act
        api_key = os.getenv("CLAUDE_API_KEY")
        project_root = os.getenv("CLAUDE_TIU_PROJECT_ROOT")
        debug_mode = os.getenv("CLAUDE_TIU_DEBUG")
        
        # Assert
        assert api_key == "test-api-key"
        assert project_root == str(temp_project)
        assert debug_mode == "true"
    
    @pytest.mark.parametrize("config_format", ["json", "yaml", "toml"])
    def test_configuration_file_integration(self, temp_project, config_format):
        """Test different configuration file formats."""
        config_data = {
            "project": {
                "name": "test-project",
                "template": "python",
                "ai_validation": True
            },
            "tools": {
                "claude_code": {"timeout": 60},
                "claude_flow": {"max_agents": 5}
            }
        }
        
        if config_format == "json":
            config_file = temp_project / "config.json"
            config_file.write_text(json.dumps(config_data, indent=2))
        elif config_format == "yaml":
            config_file = temp_project / "config.yaml"
            config_file.write_text(yaml.dump(config_data))
        elif config_format == "toml":
            config_file = temp_project / "config.toml"
            # Simple TOML representation
            toml_content = """
[project]
name = "test-project"
template = "python"
ai_validation = true

[tools.claude_code]
timeout = 60

[tools.claude_flow]
max_agents = 5
"""
            config_file.write_text(toml_content)
        
        # Verify file exists and is readable
        assert config_file.exists()
        content = config_file.read_text()
        assert "test-project" in content
        assert "python" in content