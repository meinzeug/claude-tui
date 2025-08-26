"""Integration tests for service interactions in claude-tui."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.services.ai_service import AIService
from src.services.project_service import ProjectService  
from src.services.task_service import TaskService
from src.services.validation_service import ValidationService


class TestServiceIntegration:
    """Test suite for service integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_project_workflow(
        self,
        ai_service,
        project_service,
        task_service,
        validation_service,
        sample_project_data,
        temp_dir
    ):
        """Test complete project workflow from creation to validation."""
        # Step 1: Create project
        project_data = sample_project_data.copy()
        project_data["path"] = str(temp_dir / "integration_project")
        
        project = await project_service.create_project(**project_data)
        assert project["id"] is not None
        assert project["status"] == "initialized"
        
        # Step 2: Generate code using AI service
        code_prompt = "Create a simple REST API with user authentication"
        ai_result = await ai_service.generate_code(
            prompt=code_prompt,
            context={"project_id": project["id"]}
        )
        
        assert ai_result["status"] == "success"
        assert "code" in ai_result
        
        # Step 3: Create and execute task
        task_data = {
            "name": "implement-auth-api",
            "description": "Implement authentication API",
            "prompt": code_prompt,
            "project_id": project["id"]
        }
        
        task = await task_service.create_task(**task_data)
        assert task["id"] is not None
        
        task_result = await task_service.execute_task(task["id"])
        assert task_result["status"] == "completed"
        
        # Step 4: Validate generated code
        validation_result = await validation_service.validate_project(project["id"])
        
        assert "quality_score" in validation_result
        assert "placeholder_count" in validation_result
        assert validation_result["quality_score"] > 0.5
        
        # Step 5: Update project status
        final_project = await project_service.get_project(project["id"])
        assert final_project["status"] in ["active", "completed"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_validation_feedback_loop(
        self,
        ai_service,
        validation_service,
        sample_code_with_placeholders
    ):
        """Test feedback loop between AI service and validation service."""
        # Initial validation
        initial_validation = await validation_service.validate_code(sample_code_with_placeholders)
        assert initial_validation["has_placeholders"] is True
        placeholder_count = initial_validation["placeholder_count"]
        
        # AI improvement
        improvement_result = await ai_service.improve_code(
            code=sample_code_with_placeholders,
            validation_feedback=initial_validation
        )
        
        assert improvement_result["status"] == "success"
        improved_code = improvement_result["improved_code"]
        
        # Re-validation
        final_validation = await validation_service.validate_code(improved_code)
        
        # Should have fewer placeholders after improvement
        assert final_validation["placeholder_count"] < placeholder_count
        assert final_validation["quality_score"] > initial_validation["quality_score"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(
        self,
        task_service,
        ai_service,
        sample_project_data
    ):
        """Test concurrent execution of multiple tasks."""
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task_data = {
                "name": f"concurrent-task-{i}",
                "description": f"Concurrent task {i}",
                "prompt": f"Generate function number {i}",
                "priority": "medium"
            }
            task = await task_service.create_task(**task_data)
            tasks.append(task)
        
        # Execute all tasks concurrently
        execution_coroutines = [
            task_service.execute_task(task["id"])
            for task in tasks
        ]
        
        results = await asyncio.gather(*execution_coroutines)
        
        # Verify all tasks completed successfully
        assert len(results) == 5
        for result in results:
            assert result["status"] == "completed"
            assert "execution_time" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_across_services(
        self,
        project_service,
        task_service,
        validation_service
    ):
        """Test error handling and recovery across services."""
        # Try to create task for non-existent project
        invalid_task_data = {
            "name": "invalid-task",
            "description": "Task for non-existent project",
            "prompt": "Test prompt",
            "project_id": "non-existent-id"
        }
        
        with pytest.raises(ValueError, match="Project not found"):
            await task_service.create_task(**invalid_task_data)
        
        # Try to validate non-existent project
        with pytest.raises(ValueError, match="Project not found"):
            await validation_service.validate_project("non-existent-id")
        
        # Verify services remain functional after errors
        valid_project_data = {
            "name": "recovery-test",
            "description": "Test project after error",
            "template": "python"
        }
        
        project = await project_service.create_project(**valid_project_data)
        assert project["id"] is not None

    @pytest.mark.integration
    def test_database_transaction_consistency(
        self,
        project_service,
        task_service,
        db_session
    ):
        """Test database transaction consistency across services."""
        # This test would verify that database operations are properly
        # wrapped in transactions and maintain consistency
        
        # Start a transaction
        with db_session.begin():
            # Create project and task in same transaction
            project_data = {
                "name": "transaction-test",
                "description": "Test transaction consistency",
                "template": "python"
            }
            
            project = asyncio.run(project_service.create_project(**project_data))
            
            task_data = {
                "name": "transaction-task",
                "description": "Task in transaction",
                "prompt": "Test prompt",
                "project_id": project["id"]
            }
            
            task = asyncio.run(task_service.create_task(**task_data))
            
            # Verify both exist in database
            assert project["id"] is not None
            assert task["id"] is not None
            
            # Rollback transaction (simulating error)
            db_session.rollback()
        
        # Verify rollback worked (this is simplified for testing)
        # In real implementation, we'd check database state


class TestAPIIntegration:
    """Test suite for API endpoint integration."""
    
    @pytest.mark.integration
    def test_project_api_endpoints(self, test_client, sample_project_data):
        """Test project API endpoint integration."""
        # Create project
        response = test_client.post("/api/v1/projects", json=sample_project_data)
        assert response.status_code == 201
        
        project_data = response.json()
        project_id = project_data["id"]
        
        # Get project
        response = test_client.get(f"/api/v1/projects/{project_id}")
        assert response.status_code == 200
        assert response.json()["name"] == sample_project_data["name"]
        
        # List projects
        response = test_client.get("/api/v1/projects")
        assert response.status_code == 200
        projects = response.json()
        assert len(projects) >= 1
        
        # Update project
        update_data = {"description": "Updated description"}
        response = test_client.patch(f"/api/v1/projects/{project_id}", json=update_data)
        assert response.status_code == 200
        assert response.json()["description"] == "Updated description"
        
        # Delete project
        response = test_client.delete(f"/api/v1/projects/{project_id}")
        assert response.status_code == 204

    @pytest.mark.integration
    def test_task_api_endpoints(self, test_client, sample_task_data):
        """Test task API endpoint integration."""
        # Create task
        response = test_client.post("/api/v1/tasks", json=sample_task_data)
        assert response.status_code == 201
        
        task_data = response.json()
        task_id = task_data["id"]
        
        # Get task
        response = test_client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        assert response.json()["name"] == sample_task_data["name"]
        
        # Execute task
        response = test_client.post(f"/api/v1/tasks/{task_id}/execute")
        assert response.status_code == 200
        execution_result = response.json()
        assert "status" in execution_result
        
        # Get task status
        response = test_client.get(f"/api/v1/tasks/{task_id}/status")
        assert response.status_code == 200
        status_data = response.json()
        assert "status" in status_data

    @pytest.mark.integration
    def test_validation_api_endpoints(self, test_client, sample_code_with_placeholders):
        """Test validation API endpoint integration."""
        # Validate code
        validation_data = {"code": sample_code_with_placeholders}
        response = test_client.post("/api/v1/validation/code", json=validation_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "has_placeholders" in result
        assert "quality_score" in result
        assert "placeholder_count" in result
        
        # Get validation report
        response = test_client.get("/api/v1/validation/report")
        assert response.status_code == 200
        report = response.json()
        assert "total_validations" in report

    @pytest.mark.integration
    def test_ai_api_endpoints(self, test_client):
        """Test AI service API endpoint integration."""
        # Generate code
        code_request = {
            "prompt": "Create a simple Python function",
            "language": "python",
            "context": {"project_type": "web_api"}
        }
        
        response = test_client.post("/api/v1/ai/generate-code", json=code_request)
        assert response.status_code == 200
        
        result = response.json()
        assert "code" in result
        assert "status" in result
        
        # Analyze code
        analysis_request = {"code": "def hello(): print('Hello')"}
        response = test_client.post("/api/v1/ai/analyze-code", json=analysis_request)
        assert response.status_code == 200
        
        analysis = response.json()
        assert "quality_score" in analysis
        assert "suggestions" in analysis

    @pytest.mark.integration
    def test_authenticated_endpoints(self, authenticated_client, sample_project_data):
        """Test endpoints that require authentication."""
        # Test with authenticated client
        response = authenticated_client.post("/api/v1/projects", json=sample_project_data)
        assert response.status_code == 201
        
        # Test protected endpoint access
        response = authenticated_client.get("/api/v1/user/profile")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_error_responses(self, test_client):
        """Test API error response handling."""
        # Test 404 for non-existent resource
        response = test_client.get("/api/v1/projects/non-existent-id")
        assert response.status_code == 404
        assert "detail" in response.json()
        
        # Test 400 for invalid request
        invalid_project = {"name": ""}  # Empty name should be invalid
        response = test_client.post("/api/v1/projects", json=invalid_project)
        assert response.status_code == 422  # FastAPI validation error
        
        # Test 405 for unsupported method
        response = test_client.patch("/api/v1/projects")  # PATCH without ID
        assert response.status_code == 405


class TestCLIIntegration:
    """Test suite for CLI tool integration."""
    
    @pytest.mark.integration
    def test_claude_code_cli_integration(self, temp_dir):
        """Test integration with Claude Code CLI."""
        import subprocess
        
        # Create a test script that simulates Claude Code
        test_script = temp_dir / "mock_claude_code.py"
        test_script.write_text("""
import sys
import json

if len(sys.argv) > 1 and sys.argv[1] == "--prompt":
    result = {
        "status": "success",
        "result": "def hello():\\n    print('Hello, World!')"
    }
    print(json.dumps(result))
else:
    sys.exit(1)
""")
        
        # Execute the mock CLI
        result = subprocess.run([
            "python", str(test_script), 
            "--prompt", "Generate hello world function"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        response = json.loads(result.stdout)
        assert response["status"] == "success"

    @pytest.mark.integration
    def test_claude_flow_workflow_integration(self, temp_dir):
        """Test integration with Claude Flow workflows."""
        # Create a workflow configuration file
        workflow_config = {
            "name": "test-workflow",
            "version": "1.0",
            "tasks": [
                {
                    "name": "generate-code",
                    "type": "ai-generation",
                    "prompt": "Create a simple function"
                },
                {
                    "name": "validate-code",
                    "type": "validation",
                    "depends_on": ["generate-code"]
                }
            ]
        }
        
        workflow_file = temp_dir / "test-workflow.json"
        workflow_file.write_text(json.dumps(workflow_config, indent=2))
        
        # Simulate workflow execution (in real implementation, this would use actual Claude Flow)
        assert workflow_file.exists()
        config = json.loads(workflow_file.read_text())
        assert config["name"] == "test-workflow"
        assert len(config["tasks"]) == 2

    @pytest.mark.integration 
    def test_git_integration(self, temp_dir):
        """Test Git integration functionality."""
        import subprocess
        
        # Initialize a git repository
        subprocess.run(["git", "init"], cwd=temp_dir, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_dir)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir)
        
        # Create a test file
        test_file = temp_dir / "test.py"
        test_file.write_text("def test(): pass")
        
        # Add and commit
        subprocess.run(["git", "add", "test.py"], cwd=temp_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True)
        
        # Verify git operations worked
        result = subprocess.run(
            ["git", "log", "--oneline"], 
            cwd=temp_dir, 
            capture_output=True, 
            text=True
        )
        assert "Initial commit" in result.stdout


class TestPerformanceIntegration:
    """Test suite for performance-related integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_load_service_integration(
        self,
        ai_service,
        validation_service,
        performance_monitor
    ):
        """Test service integration under high load."""
        with performance_monitor as monitor:
            # Create many concurrent requests
            tasks = []
            for i in range(50):  # 50 concurrent operations
                # Alternate between AI generation and validation
                if i % 2 == 0:
                    task = ai_service.generate_code(f"Generate function {i}", {"simple": True})
                else:
                    task = validation_service.validate_code(f"def func_{i}(): pass")
                tasks.append(task)
            
            # Execute all concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify performance requirements
        assert monitor.duration < 30.0  # Should complete within 30 seconds
        assert monitor.memory_delta < 100 * 1024 * 1024  # Less than 100MB memory increase
        
        # Verify all requests succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 45  # Allow for some failures under load

    @pytest.mark.integration
    @pytest.mark.performance
    def test_database_performance_integration(self, project_service, performance_monitor):
        """Test database performance under load."""
        with performance_monitor as monitor:
            # Create many projects rapidly
            projects = []
            for i in range(100):
                project_data = {
                    "name": f"perf-test-project-{i}",
                    "description": f"Performance test project {i}",
                    "template": "python"
                }
                project = asyncio.run(project_service.create_project(**project_data))
                projects.append(project)
        
        assert monitor.duration < 10.0  # Should create 100 projects in under 10 seconds
        assert len(projects) == 100
        
        # Verify all projects have unique IDs
        project_ids = [p["id"] for p in projects]
        assert len(set(project_ids)) == 100

    @pytest.mark.integration
    @pytest.mark.performance
    def test_memory_leak_detection(self, ai_service, validation_service):
        """Test for memory leaks in service interactions."""
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform many operations that could potentially leak memory
        for i in range(200):
            # Force garbage collection periodically
            if i % 50 == 0:
                gc.collect()
            
            # Perform operations
            code = f"def function_{i}(): return {i}"
            asyncio.run(validation_service.validate_code(code))
            
            if i % 10 == 0:  # Less frequent AI calls due to potential rate limits
                asyncio.run(ai_service.generate_code(f"Simple function {i}", {"minimal": True}))
        
        # Final garbage collection
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 200 operations)
        assert memory_increase < 50 * 1024 * 1024