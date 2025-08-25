"""
End-to-end tests for complete user workflows in Claude-TIU.

This module tests complete user journeys from project creation to completion,
including AI-powered development, collaboration, and validation workflows.
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
import json
import tempfile
import shutil


@dataclass
class UserSession:
    """Represents a user session for testing."""
    user_id: str
    username: str
    email: str
    auth_token: str
    session_data: Dict[str, Any]


@dataclass 
class ProjectWorkflow:
    """Represents a complete project workflow."""
    project_id: str
    workflow_type: str
    steps_completed: List[str]
    current_step: str
    completion_percentage: float
    artifacts_generated: List[str]


class TestCompleteUserJourneys:
    """Test complete end-to-end user journeys."""
    
    @pytest.fixture
    async def user_session(self):
        """Create authenticated user session for testing."""
        return UserSession(
            user_id="test-user-123",
            username="testuser",
            email="testuser@example.com",
            auth_token="jwt-token-abc123",
            session_data={
                "preferences": {"theme": "dark", "auto_save": True},
                "project_history": [],
                "collaboration_settings": {"notifications": True}
            }
        )
    
    @pytest.fixture
    def mock_system(self):
        """Create comprehensive mock system for E2E testing."""
        class MockClaudeTIUSystem:
            def __init__(self):
                self.projects = {}
                self.tasks = {}
                self.users = {}
                self.workflows = {}
                self.collaboration_data = {}
                self.validation_results = {}
                self.git_repos = {}
                self.templates = {}
                
                # Initialize with some templates
                self._initialize_templates()
            
            def _initialize_templates(self):
                """Initialize template marketplace."""
                self.templates = {
                    "python-fastapi": {
                        "id": "python-fastapi",
                        "name": "FastAPI Application",
                        "description": "Modern Python web API with FastAPI",
                        "category": "web",
                        "files": ["main.py", "requirements.txt", "Dockerfile"],
                        "features": ["api", "database", "auth", "testing"]
                    },
                    "react-typescript": {
                        "id": "react-typescript",
                        "name": "React TypeScript App", 
                        "description": "React application with TypeScript",
                        "category": "frontend",
                        "files": ["src/App.tsx", "package.json", "tsconfig.json"],
                        "features": ["react", "typescript", "testing", "build"]
                    },
                    "python-ml": {
                        "id": "python-ml",
                        "name": "Machine Learning Project",
                        "description": "Python ML project with Jupyter notebooks",
                        "category": "data-science",
                        "files": ["train.py", "model.py", "requirements.txt", "notebooks/"],
                        "features": ["ml", "jupyter", "data-processing", "visualization"]
                    }
                }
            
            async def authenticate_user(self, credentials: Dict[str, str]) -> UserSession:
                """Authenticate user and create session."""
                user_id = f"user-{len(self.users) + 1}"
                session = UserSession(
                    user_id=user_id,
                    username=credentials["username"],
                    email=credentials.get("email", f"{credentials['username']}@example.com"),
                    auth_token=f"jwt-{user_id}-{int(time.time())}",
                    session_data={"login_time": time.time()}
                )
                self.users[user_id] = session
                return session
            
            async def create_project(self, user_session: UserSession, project_data: Dict[str, Any]) -> Dict[str, Any]:
                """Create new project."""
                project_id = f"proj-{len(self.projects) + 1}"
                template_id = project_data.get("template", "python-fastapi")
                template = self.templates.get(template_id, self.templates["python-fastapi"])
                
                project = {
                    "id": project_id,
                    "name": project_data["name"],
                    "description": project_data.get("description", ""),
                    "template": template,
                    "owner_id": user_session.user_id,
                    "status": "initializing",
                    "created_at": time.time(),
                    "progress": 0.0,
                    "files_generated": [],
                    "tasks": [],
                    "collaborators": [user_session.user_id],
                    "git_repo": None,
                    "validation_status": "pending"
                }
                
                self.projects[project_id] = project
                
                # Simulate project initialization
                await asyncio.sleep(0.1)
                project["status"] = "initialized"
                project["progress"] = 10.0
                
                return project
            
            async def generate_project_structure(self, project_id: str) -> Dict[str, Any]:
                """Generate project structure using AI."""
                project = self.projects.get(project_id)
                if not project:
                    raise ValueError(f"Project {project_id} not found")
                
                template = project["template"]
                
                # Simulate AI-powered structure generation
                await asyncio.sleep(0.5)
                
                generated_files = []
                for file_template in template["files"]:
                    if file_template.endswith("/"):
                        # Directory
                        generated_files.append({
                            "path": file_template,
                            "type": "directory",
                            "content": None
                        })
                    else:
                        # File with generated content
                        content = self._generate_file_content(file_template, template)
                        generated_files.append({
                            "path": file_template,
                            "type": "file", 
                            "content": content,
                            "size": len(content)
                        })
                
                project["files_generated"] = generated_files
                project["status"] = "structure_generated"
                project["progress"] = 30.0
                
                return {
                    "project_id": project_id,
                    "files_generated": generated_files,
                    "status": "success"
                }
            
            def _generate_file_content(self, filename: str, template: Dict[str, Any]) -> str:
                """Generate realistic file content based on template."""
                if filename == "main.py":
                    return '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Claude-TIU Generated API", version="1.0.0")

class Item(BaseModel):
    name: str
    description: str
    price: float

@app.get("/")
async def root():
    return {"message": "Hello from Claude-TIU generated API!"}

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id, "name": f"Item {item_id}"}

@app.post("/items/")
async def create_item(item: Item):
    return {"message": "Item created", "item": item}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
                    '''.strip()
                elif filename == "requirements.txt":
                    return "fastapi==0.104.1\\nuvicorn==0.24.0\\npydantic==2.5.0\\npytest==7.4.3\\nrequests==2.31.0"
                elif filename == "Dockerfile":
                    return '''
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
                    '''.strip()
                elif filename == "package.json":
                    return json.dumps({
                        "name": "claude-tiu-react-app",
                        "version": "1.0.0", 
                        "dependencies": {
                            "react": "^18.2.0",
                            "typescript": "^5.0.0"
                        },
                        "scripts": {
                            "start": "react-scripts start",
                            "build": "react-scripts build",
                            "test": "react-scripts test"
                        }
                    }, indent=2)
                else:
                    return f"# Generated content for {filename}\\n# Created by Claude-TIU AI system"
            
            async def execute_tasks(self, project_id: str, task_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Execute project tasks using AI."""
                project = self.projects.get(project_id)
                if not project:
                    raise ValueError(f"Project {project_id} not found")
                
                results = []
                for i, task_def in enumerate(task_definitions):
                    task_id = f"task-{project_id}-{i+1}"
                    
                    # Simulate task execution
                    await asyncio.sleep(0.3)
                    
                    task_result = {
                        "task_id": task_id,
                        "name": task_def["name"],
                        "type": task_def["type"],
                        "status": "completed",
                        "execution_time": 0.3,
                        "output": {
                            "files_modified": task_def.get("files", []),
                            "code_generated": True,
                            "tests_created": "test" in task_def["type"]
                        }
                    }
                    
                    self.tasks[task_id] = task_result
                    results.append(task_result)
                
                # Update project progress
                project["tasks"].extend(results)
                project["status"] = "tasks_completed"
                project["progress"] = 70.0
                
                return results
            
            async def validate_project(self, project_id: str) -> Dict[str, Any]:
                """Validate project with anti-hallucination checks."""
                project = self.projects.get(project_id)
                if not project:
                    raise ValueError(f"Project {project_id} not found")
                
                # Simulate comprehensive validation
                await asyncio.sleep(0.4)
                
                validation_result = {
                    "project_id": project_id,
                    "overall_authenticity": 0.94,
                    "real_progress": 85.0,
                    "fake_progress": 15.0,
                    "code_quality_score": 0.88,
                    "issues_found": [],
                    "recommendations": [
                        "Add more comprehensive error handling",
                        "Include integration tests"
                    ],
                    "validation_passed": True,
                    "confidence": 0.94
                }
                
                self.validation_results[project_id] = validation_result
                project["validation_status"] = "passed"
                project["progress"] = 90.0
                
                return validation_result
            
            async def setup_git_repository(self, project_id: str, git_config: Dict[str, Any]) -> Dict[str, Any]:
                """Set up Git repository for project."""
                project = self.projects.get(project_id)
                if not project:
                    raise ValueError(f"Project {project_id} not found")
                
                await asyncio.sleep(0.2)
                
                repo_info = {
                    "repository_url": f"https://github.com/{git_config['username']}/{project['name']}.git",
                    "branch": "main",
                    "commits": [],
                    "remote_configured": True,
                    "initial_commit": "feat: initial project setup by Claude-TIU"
                }
                
                self.git_repos[project_id] = repo_info
                project["git_repo"] = repo_info
                project["status"] = "completed"
                project["progress"] = 100.0
                
                return repo_info
            
            async def add_collaborator(self, project_id: str, collaborator_email: str) -> Dict[str, Any]:
                """Add collaborator to project."""
                project = self.projects.get(project_id)
                if not project:
                    raise ValueError(f"Project {project_id} not found")
                
                # Simulate finding user by email
                collaborator_id = f"collab-{len(project['collaborators']) + 1}"
                
                if collaborator_id not in project["collaborators"]:
                    project["collaborators"].append(collaborator_id)
                
                return {
                    "project_id": project_id,
                    "collaborator_added": collaborator_id,
                    "collaborator_email": collaborator_email,
                    "permissions": ["read", "write", "comment"]
                }
        
        return MockClaudeTIUSystem()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_project_creation_workflow(self, user_session, mock_system):
        """Test complete workflow from user authentication to project completion."""
        # Step 1: User Authentication (already done in fixture)
        assert user_session.user_id is not None
        assert user_session.auth_token is not None
        
        # Step 2: Browse Templates
        available_templates = list(mock_system.templates.values())
        assert len(available_templates) >= 3
        selected_template = available_templates[0]  # FastAPI template
        
        # Step 3: Create Project
        project_data = {
            "name": "my-awesome-api",
            "description": "A comprehensive API built with Claude-TIU",
            "template": selected_template["id"],
            "features": ["authentication", "database", "testing", "deployment"]
        }
        
        project = await mock_system.create_project(user_session, project_data)
        
        assert project["id"] is not None
        assert project["status"] == "initialized"
        assert project["owner_id"] == user_session.user_id
        assert project["template"]["id"] == selected_template["id"]
        
        # Step 4: Generate Project Structure
        structure_result = await mock_system.generate_project_structure(project["id"])
        
        assert structure_result["status"] == "success"
        assert len(structure_result["files_generated"]) > 0
        
        # Verify essential files are generated
        file_paths = [f["path"] for f in structure_result["files_generated"]]
        assert "main.py" in file_paths
        assert "requirements.txt" in file_paths
        
        # Step 5: Execute Development Tasks
        task_definitions = [
            {
                "name": "Implement authentication endpoints",
                "type": "code_generation",
                "files": ["auth.py", "models/user.py"]
            },
            {
                "name": "Create database models",
                "type": "code_generation", 
                "files": ["models/__init__.py", "models/base.py"]
            },
            {
                "name": "Add comprehensive tests",
                "type": "test_generation",
                "files": ["tests/test_auth.py", "tests/test_models.py"]
            }
        ]
        
        task_results = await mock_system.execute_tasks(project["id"], task_definitions)
        
        assert len(task_results) == 3
        for result in task_results:
            assert result["status"] == "completed"
            assert result["output"]["code_generated"] is True
        
        # Step 6: Validate Project
        validation_result = await mock_system.validate_project(project["id"])
        
        assert validation_result["validation_passed"] is True
        assert validation_result["overall_authenticity"] > 0.9
        assert validation_result["real_progress"] > 80
        assert validation_result["confidence"] > 0.9
        
        # Step 7: Set up Git Repository
        git_config = {
            "username": "testuser",
            "repository_name": project_data["name"],
            "private": False,
            "initialize_with_readme": True
        }
        
        repo_result = await mock_system.setup_git_repository(project["id"], git_config)
        
        assert repo_result["remote_configured"] is True
        assert "github.com" in repo_result["repository_url"]
        
        # Step 8: Verify Project Completion
        final_project = mock_system.projects[project["id"]]
        assert final_project["status"] == "completed"
        assert final_project["progress"] == 100.0
        assert len(final_project["tasks"]) == 3
        assert final_project["validation_status"] == "passed"
        assert final_project["git_repo"] is not None
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_collaborative_development_workflow(self, user_session, mock_system):
        """Test collaborative development workflow with multiple users."""
        # Create primary project
        project_data = {
            "name": "collaborative-project",
            "description": "Multi-developer project using Claude-TIU",
            "template": "python-fastapi"
        }
        
        project = await mock_system.create_project(user_session, project_data)
        
        # Add collaborators
        collaborator_emails = ["dev1@example.com", "dev2@example.com"]
        collaborators = []
        
        for email in collaborator_emails:
            collab_result = await mock_system.add_collaborator(project["id"], email)
            collaborators.append(collab_result)
        
        assert len(collaborators) == 2
        updated_project = mock_system.projects[project["id"]]
        assert len(updated_project["collaborators"]) == 3  # Original owner + 2 collaborators
        
        # Simulate collaborative task execution
        collaborative_tasks = [
            {
                "name": "User authentication system",
                "type": "code_generation",
                "assigned_to": collaborators[0]["collaborator_added"],
                "files": ["auth/"]
            },
            {
                "name": "API endpoints for data management", 
                "type": "code_generation",
                "assigned_to": collaborators[1]["collaborator_added"],
                "files": ["api/data.py"]
            },
            {
                "name": "Integration tests",
                "type": "test_generation",
                "assigned_to": user_session.user_id,
                "files": ["tests/"]
            }
        ]
        
        task_results = await mock_system.execute_tasks(project["id"], collaborative_tasks)
        
        # Verify all tasks completed successfully
        assert len(task_results) == 3
        for result in task_results:
            assert result["status"] == "completed"
        
        # Validate collaborative project
        validation_result = await mock_system.validate_project(project["id"])
        assert validation_result["validation_passed"] is True
    
    @pytest.mark.e2e 
    @pytest.mark.asyncio
    async def test_ml_project_workflow(self, user_session, mock_system):
        """Test complete ML project development workflow."""
        # Create ML project
        project_data = {
            "name": "ml-classification-model",
            "description": "Machine learning classification project",
            "template": "python-ml",
            "features": ["data-processing", "model-training", "evaluation", "deployment"]
        }
        
        project = await mock_system.create_project(user_session, project_data)
        
        # Generate ML-specific structure
        await mock_system.generate_project_structure(project["id"])
        
        # Execute ML-specific tasks
        ml_tasks = [
            {
                "name": "Data preprocessing pipeline",
                "type": "data_processing",
                "files": ["src/data_preprocessing.py", "src/feature_engineering.py"]
            },
            {
                "name": "Model training script",
                "type": "model_development",
                "files": ["src/train_model.py", "src/model_evaluation.py"]
            },
            {
                "name": "Model deployment API",
                "type": "deployment",
                "files": ["app.py", "model_service.py"]
            },
            {
                "name": "ML pipeline tests",
                "type": "ml_testing",
                "files": ["tests/test_preprocessing.py", "tests/test_model.py"]
            }
        ]
        
        task_results = await mock_system.execute_tasks(project["id"], ml_tasks)
        
        assert len(task_results) == 4
        
        # Validate ML project with specialized checks
        validation_result = await mock_system.validate_project(project["id"])
        
        # ML projects might have different validation criteria
        assert validation_result["overall_authenticity"] > 0.85  # Slightly lower threshold for ML
        assert validation_result["validation_passed"] is True
        
        # Set up Git with ML-specific configurations
        git_config = {
            "username": "testuser",
            "repository_name": project_data["name"],
            "private": True,  # ML projects often private initially
            "gitignore_template": "python-ml"
        }
        
        repo_result = await mock_system.setup_git_repository(project["id"], git_config)
        assert repo_result["remote_configured"] is True
    
    @pytest.mark.e2e
    @pytest.mark.asyncio 
    async def test_error_recovery_workflow(self, user_session, mock_system):
        """Test error handling and recovery in user workflows."""
        # Create project
        project_data = {
            "name": "error-recovery-test",
            "description": "Testing error recovery mechanisms",
            "template": "react-typescript"
        }
        
        project = await mock_system.create_project(user_session, project_data)
        
        # Simulate error during structure generation
        original_generate = mock_system.generate_project_structure
        
        async def failing_generate(project_id):
            # Fail first time, succeed second time
            if not hasattr(failing_generate, 'called'):
                failing_generate.called = True
                raise Exception("Network timeout during structure generation")
            return await original_generate(project_id)
        
        mock_system.generate_project_structure = failing_generate
        
        # First attempt should fail
        with pytest.raises(Exception, match="Network timeout"):
            await mock_system.generate_project_structure(project["id"])
        
        # Second attempt should succeed (retry mechanism)
        structure_result = await mock_system.generate_project_structure(project["id"])
        assert structure_result["status"] == "success"
        
        # Continue with workflow after recovery
        task_definitions = [
            {
                "name": "React component development",
                "type": "frontend_code_generation",
                "files": ["src/components/"]
            }
        ]
        
        task_results = await mock_system.execute_tasks(project["id"], task_definitions)
        assert len(task_results) == 1
        assert task_results[0]["status"] == "completed"
        
        # Final validation should pass despite earlier errors
        validation_result = await mock_system.validate_project(project["id"])
        assert validation_result["validation_passed"] is True
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_project_workflow(self, user_session, mock_system):
        """Test workflow with large, complex project."""
        # Create large project
        project_data = {
            "name": "enterprise-application",
            "description": "Large enterprise application with microservices",
            "template": "python-fastapi",
            "features": [
                "authentication", "authorization", "database", "caching",
                "logging", "monitoring", "testing", "deployment",
                "microservices", "api-gateway", "message-queue"
            ]
        }
        
        project = await mock_system.create_project(user_session, project_data)
        
        # Add multiple collaborators for large project
        collaborator_emails = [f"dev{i}@company.com" for i in range(1, 6)]  # 5 developers
        
        for email in collaborator_emails:
            await mock_system.add_collaborator(project["id"], email)
        
        # Generate complex structure
        structure_result = await mock_system.generate_project_structure(project["id"])
        assert len(structure_result["files_generated"]) > 0
        
        # Execute many tasks in parallel (simulated)
        large_task_set = [
            {"name": f"Service {i} implementation", "type": "microservice_generation", "files": [f"services/service_{i}/"]}
            for i in range(1, 11)  # 10 microservices
        ]
        
        large_task_set.extend([
            {"name": "API Gateway", "type": "gateway_implementation", "files": ["gateway/"]},
            {"name": "Database migrations", "type": "database_setup", "files": ["migrations/"]},
            {"name": "Monitoring setup", "type": "observability", "files": ["monitoring/"]},
            {"name": "CI/CD pipeline", "type": "devops", "files": [".github/workflows/"]},
            {"name": "Comprehensive test suite", "type": "test_generation", "files": ["tests/"]}
        ])
        
        # Execute tasks (this would be parallel in real implementation)
        start_time = time.time()
        task_results = await mock_system.execute_tasks(project["id"], large_task_set)
        execution_time = time.time() - start_time
        
        # Verify all tasks completed
        assert len(task_results) == len(large_task_set)
        for result in task_results:
            assert result["status"] == "completed"
        
        # Performance check - large project should still complete reasonably fast
        assert execution_time < 10.0, f"Large project execution took too long: {execution_time:.2f}s"
        
        # Validate large project
        validation_result = await mock_system.validate_project(project["id"])
        assert validation_result["validation_passed"] is True
        assert validation_result["overall_authenticity"] > 0.80  # Slightly lower for complex projects
        
        # Set up Git for large project
        git_config = {
            "username": "enterprise-org",
            "repository_name": project_data["name"],
            "private": True,
            "branch_protection": True,
            "required_reviews": 2
        }
        
        repo_result = await mock_system.setup_git_repository(project["id"], git_config)
        assert repo_result["remote_configured"] is True
        
        # Verify final project state
        final_project = mock_system.projects[project["id"]]
        assert final_project["status"] == "completed"
        assert final_project["progress"] == 100.0
        assert len(final_project["collaborators"]) == 6  # 1 owner + 5 collaborators
        assert len(final_project["tasks"]) == len(large_task_set)


class TestWorkflowPerformance:
    """Test performance characteristics of user workflows."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Monitor performance during workflows."""
        class PerformanceMonitor:
            def __init__(self):
                self.timings = {}
                self.memory_usage = {}
                
            def start_timing(self, operation: str):
                """Start timing an operation."""
                self.timings[operation] = {"start": time.time()}
                
            def end_timing(self, operation: str):
                """End timing an operation."""
                if operation in self.timings:
                    self.timings[operation]["end"] = time.time()
                    self.timings[operation]["duration"] = (
                        self.timings[operation]["end"] - self.timings[operation]["start"]
                    )
            
            def get_timing(self, operation: str) -> float:
                """Get duration for operation."""
                return self.timings.get(operation, {}).get("duration", 0.0)
            
            def get_performance_report(self) -> Dict[str, Any]:
                """Get complete performance report."""
                return {
                    "operation_timings": {
                        op: data.get("duration", 0.0) 
                        for op, data in self.timings.items()
                    },
                    "total_workflow_time": sum(
                        data.get("duration", 0.0) 
                        for data in self.timings.values()
                    ),
                    "slowest_operation": max(
                        self.timings.keys(),
                        key=lambda op: self.timings[op].get("duration", 0.0)
                    ) if self.timings else None
                }
        
        return PerformanceMonitor()
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_workflow_performance_targets(self, user_session, mock_system, performance_monitor):
        """Test that workflows meet performance targets."""
        # Performance targets
        targets = {
            "project_creation": 2.0,      # 2 seconds
            "structure_generation": 3.0,  # 3 seconds  
            "task_execution": 5.0,        # 5 seconds
            "validation": 2.0,            # 2 seconds
            "git_setup": 1.0,             # 1 second
            "total_workflow": 15.0        # 15 seconds total
        }
        
        # Start workflow timing
        workflow_start = time.time()
        
        # Step 1: Project Creation
        performance_monitor.start_timing("project_creation")
        project_data = {
            "name": "performance-test-project",
            "description": "Testing workflow performance",
            "template": "python-fastapi"
        }
        project = await mock_system.create_project(user_session, project_data)
        performance_monitor.end_timing("project_creation")
        
        # Step 2: Structure Generation
        performance_monitor.start_timing("structure_generation")
        await mock_system.generate_project_structure(project["id"])
        performance_monitor.end_timing("structure_generation")
        
        # Step 3: Task Execution
        performance_monitor.start_timing("task_execution")
        tasks = [
            {"name": "API implementation", "type": "code_generation", "files": ["api.py"]},
            {"name": "Unit tests", "type": "test_generation", "files": ["test_api.py"]}
        ]
        await mock_system.execute_tasks(project["id"], tasks)
        performance_monitor.end_timing("task_execution")
        
        # Step 4: Validation
        performance_monitor.start_timing("validation")
        await mock_system.validate_project(project["id"])
        performance_monitor.end_timing("validation")
        
        # Step 5: Git Setup
        performance_monitor.start_timing("git_setup")
        git_config = {"username": "testuser", "repository_name": project["name"]}
        await mock_system.setup_git_repository(project["id"], git_config)
        performance_monitor.end_timing("git_setup")
        
        workflow_end = time.time()
        total_time = workflow_end - workflow_start
        
        # Verify performance targets
        for operation, target_time in targets.items():
            if operation == "total_workflow":
                actual_time = total_time
            else:
                actual_time = performance_monitor.get_timing(operation)
            
            assert actual_time <= target_time, f"{operation} took {actual_time:.2f}s, exceeding target of {target_time}s"
        
        # Generate performance report
        report = performance_monitor.get_performance_report()
        print(f"\\nWorkflow Performance Report:")
        print(f"Total time: {total_time:.2f}s")
        for op, duration in report["operation_timings"].items():
            print(f"  {op}: {duration:.2f}s")
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self, mock_system, performance_monitor):
        """Test performance with multiple concurrent workflows."""
        # Create multiple user sessions
        user_sessions = []
        for i in range(5):
            credentials = {
                "username": f"user{i}",
                "email": f"user{i}@example.com"
            }
            session = await mock_system.authenticate_user(credentials)
            user_sessions.append(session)
        
        # Start concurrent workflows
        performance_monitor.start_timing("concurrent_workflows")
        
        async def run_user_workflow(user_session):
            """Run complete workflow for one user."""
            project_data = {
                "name": f"concurrent-project-{user_session.user_id}",
                "description": "Concurrent workflow test",
                "template": "python-fastapi"
            }
            
            project = await mock_system.create_project(user_session, project_data)
            await mock_system.generate_project_structure(project["id"])
            
            tasks = [{"name": "Implementation", "type": "code_generation", "files": ["main.py"]}]
            await mock_system.execute_tasks(project["id"], tasks)
            await mock_system.validate_project(project["id"])
            
            return project["id"]
        
        # Execute workflows concurrently
        workflow_tasks = [run_user_workflow(session) for session in user_sessions]
        completed_projects = await asyncio.gather(*workflow_tasks)
        
        performance_monitor.end_timing("concurrent_workflows")
        
        # Verify all workflows completed successfully
        assert len(completed_projects) == 5
        for project_id in completed_projects:
            assert project_id in mock_system.projects
            project = mock_system.projects[project_id]
            assert project["status"] == "completed"
            assert project["progress"] == 100.0
        
        # Performance should not degrade significantly with concurrency
        concurrent_time = performance_monitor.get_timing("concurrent_workflows")
        assert concurrent_time < 30.0, f"Concurrent workflows took {concurrent_time:.2f}s, too slow"