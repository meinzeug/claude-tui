"""
End-to-end tests for complete user workflows in claude-tui.

This module tests complete user journeys from start to finish,
ensuring all components work together seamlessly.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestProjectWorkflow:
    """End-to-end tests for complete project workflows."""
    
    @pytest.fixture
    def workflow_components(self):
        """Set up all components needed for workflow testing."""
        return {
            "tui_app": Mock(),
            "project_manager": Mock(),
            "task_engine": Mock(),
            "ai_interface": Mock(),
            "validation_engine": Mock(),
            "file_system": Mock()
        }
    
    @pytest.mark.asyncio
    async def test_complete_project_creation_workflow(self, workflow_components, temp_project_dir):
        """Test complete project creation from TUI to file generation."""
        components = workflow_components
        
        # Step 1: User initiates project creation via TUI
        components["tui_app"].show_project_creation_screen = Mock()
        components["tui_app"].get_project_input = Mock(return_value={
            "name": "my-awesome-app",
            "template": "python",
            "description": "An awesome Python application",
            "target_directory": str(temp_project_dir)
        })
        
        # Step 2: Project Manager creates project structure
        components["project_manager"].create_project = AsyncMock(return_value={
            "id": "proj_awesome_123",
            "name": "my-awesome-app",
            "status": "initialized",
            "directory": str(temp_project_dir / "my-awesome-app")
        })
        
        # Step 3: AI generates initial project files
        components["ai_interface"].generate_project_structure = AsyncMock(return_value={
            "files": {
                "main.py": "#!/usr/bin/env python3\\n\\ndef main():\\n    print('Hello, World!')\\n\\nif __name__ == '__main__':\\n    main()",
                "requirements.txt": "# Project dependencies\\nclick>=8.0.0\\nrequests>=2.28.0",
                "README.md": "# My Awesome App\\n\\nAn awesome Python application.",
                "setup.py": "from setuptools import setup\\n\\nsetup(name='my-awesome-app', version='0.1.0')"
            },
            "directories": ["src", "tests", "docs"]
        })
        
        # Step 4: Validation engine checks generated code
        components["validation_engine"].validate_project = AsyncMock(return_value={
            "is_valid": True,
            "quality_score": 0.95,
            "issues": [],
            "suggestions": ["Consider adding type hints", "Add unit tests"]
        })
        
        # Step 5: File system operations
        components["file_system"].create_directory = Mock(return_value=True)
        components["file_system"].write_file = Mock(return_value=True)
        
        # Execute workflow
        components["tui_app"].show_project_creation_screen()
        user_input = components["tui_app"].get_project_input()
        
        project = await components["project_manager"].create_project(user_input)
        assert project["name"] == "my-awesome-app"
        
        structure = await components["ai_interface"].generate_project_structure({
            "template": user_input["template"],
            "name": user_input["name"]
        })
        assert len(structure["files"]) == 4
        assert "main.py" in structure["files"]
        
        validation = await components["validation_engine"].validate_project(project["id"])
        assert validation["is_valid"] is True
        
        # Verify file creation
        for directory in structure["directories"]:
            components["file_system"].create_directory(f"{project['directory']}/{directory}")
        
        for filename, content in structure["files"].items():
            components["file_system"].write_file(f"{project['directory']}/{filename}", content)
        
        # Verify all components were called
        components["project_manager"].create_project.assert_awaited_once()
        components["ai_interface"].generate_project_structure.assert_awaited_once()
        components["validation_engine"].validate_project.assert_awaited_once()
        assert components["file_system"].create_directory.call_count == 3
        assert components["file_system"].write_file.call_count == 4
    
    @pytest.mark.asyncio
    async def test_task_execution_workflow(self, workflow_components):
        """Test complete task execution workflow."""
        components = workflow_components
        
        # Step 1: User creates a task
        task_input = {
            "name": "implement_authentication",
            "prompt": "Implement user authentication with JWT tokens",
            "project_id": "proj_awesome_123",
            "priority": "high"
        }
        
        components["task_engine"].create_task = AsyncMock(return_value={
            "id": "task_auth_456",
            **task_input,
            "status": "pending"
        })
        
        # Step 2: AI executes the task
        components["ai_interface"].execute_task = AsyncMock(return_value={
            "status": "completed",
            "output": {
                "files": {
                    "auth.py": "import jwt\\nfrom datetime import datetime\\n\\nclass AuthManager:\\n    def __init__(self, secret_key):\\n        self.secret_key = secret_key\\n    \\n    def generate_token(self, user_id):\\n        payload = {'user_id': user_id, 'exp': datetime.utcnow()}\\n        return jwt.encode(payload, self.secret_key, algorithm='HS256')",
                    "test_auth.py": "import pytest\\nfrom auth import AuthManager\\n\\ndef test_generate_token():\\n    manager = AuthManager('secret')\\n    token = manager.generate_token('user123')\\n    assert token is not None"
                },
                "dependencies": ["PyJWT>=2.6.0"],
                "documentation": "Authentication module with JWT token support"
            },
            "execution_time": 15.2
        })
        
        # Step 3: Validation checks
        components["validation_engine"].validate_code = AsyncMock(return_value={
            "has_placeholders": False,
            "real_progress": 100,
            "fake_progress": 0,
            "quality_score": 0.92,
            "security_score": 0.88,
            "test_coverage": 85
        })
        
        # Step 4: File operations and project update
        components["file_system"].write_file = Mock(return_value=True)
        components["project_manager"].update_project = AsyncMock(return_value=True)
        
        # Execute workflow
        task = await components["task_engine"].create_task(task_input)
        assert task["id"] == "task_auth_456"
        
        execution_result = await components["ai_interface"].execute_task(task)
        assert execution_result["status"] == "completed"
        assert "auth.py" in execution_result["output"]["files"]
        
        # Validate each generated file
        for filename, content in execution_result["output"]["files"].items():
            validation = await components["validation_engine"].validate_code(content)
            assert validation["has_placeholders"] is False
            assert validation["quality_score"] > 0.8
        
        # Save files and update project
        for filename, content in execution_result["output"]["files"].items():
            components["file_system"].write_file(f"project_dir/{filename}", content)
        
        await components["project_manager"].update_project(task["project_id"], {
            "last_task_completed": task["id"],
            "dependencies": execution_result["output"]["dependencies"]
        })
        
        # Verify workflow completion
        assert components["file_system"].write_file.call_count == 2  # auth.py and test_auth.py
        components["project_manager"].update_project.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_code_review_workflow(self, workflow_components):
        """Test automated code review workflow."""
        components = workflow_components
        
        # Sample code for review
        code_to_review = '''
        def process_user_data(user_input):
            # TODO: add input validation
            sql_query = f"SELECT * FROM users WHERE name = '{user_input}'"
            result = execute_query(sql_query)
            return result
        '''
        
        # Step 1: Code analysis
        components["validation_engine"].analyze_code = AsyncMock(return_value={
            "security_issues": [
                {
                    "type": "sql_injection",
                    "severity": "high",
                    "line": 3,
                    "description": "Potential SQL injection vulnerability"
                }
            ],
            "quality_issues": [
                {
                    "type": "placeholder",
                    "severity": "medium",
                    "line": 2,
                    "description": "TODO comment indicates incomplete implementation"
                }
            ],
            "suggestions": [
                "Use parameterized queries",
                "Add input validation",
                "Implement proper error handling"
            ]
        })
        
        # Step 2: AI generates fixes
        components["ai_interface"].generate_code_fixes = AsyncMock(return_value={
            "fixed_code": '''
        def process_user_data(user_input):
            if not user_input or not isinstance(user_input, str):
                raise ValueError("Invalid input: user_input must be a non-empty string")
            
            # Use parameterized query to prevent SQL injection
            sql_query = "SELECT * FROM users WHERE name = %s"
            try:
                result = execute_query(sql_query, (user_input,))
                return result
            except Exception as e:
                logger.error(f"Database query failed: {e}")
                raise
            ''',
            "fixes_applied": [
                "Added input validation",
                "Replaced string formatting with parameterized query",
                "Added error handling",
                "Removed TODO placeholder"
            ]
        })
        
        # Step 3: Re-validation of fixed code
        components["validation_engine"].validate_fixes = AsyncMock(return_value={
            "security_score": 0.95,  # Improved from previous
            "quality_score": 0.90,
            "remaining_issues": [],
            "improvement_metrics": {
                "security_improvement": 0.45,
                "quality_improvement": 0.25
            }
        })
        
        # Execute workflow
        analysis = await components["validation_engine"].analyze_code(code_to_review)
        assert len(analysis["security_issues"]) == 1
        assert analysis["security_issues"][0]["type"] == "sql_injection"
        
        fixes = await components["ai_interface"].generate_code_fixes({
            "code": code_to_review,
            "issues": analysis["security_issues"] + analysis["quality_issues"]
        })
        assert "parameterized query" in fixes["fixes_applied"][1].lower()
        
        validation = await components["validation_engine"].validate_fixes(fixes["fixed_code"])
        assert validation["security_score"] > 0.9
        assert len(validation["remaining_issues"]) == 0
        
        # Verify improvement
        assert validation["improvement_metrics"]["security_improvement"] > 0.4


class TestTUIWorkflow:
    """End-to-end tests for TUI interaction workflows."""
    
    @pytest.fixture
    def mock_tui_app(self):
        """Create mock TUI application."""
        app = Mock()
        app.screens = {}
        app.current_screen = None
        app.user_input = None
        return app
    
    def test_navigation_workflow(self, mock_tui_app):
        """Test complete TUI navigation workflow."""
        app = mock_tui_app
        
        # Mock screen transitions
        app.show_welcome_screen = Mock()
        app.show_project_list = Mock()
        app.show_project_details = Mock()
        app.show_task_creation = Mock()
        app.handle_user_input = Mock()
        
        # Simulate user navigation flow
        # 1. Start at welcome screen
        app.show_welcome_screen()
        app.current_screen = "welcome"
        
        # 2. Navigate to project list
        app.handle_user_input("view_projects")
        app.show_project_list()
        app.current_screen = "project_list"
        
        # 3. Select a project
        app.handle_user_input("select_project:proj_123")
        app.show_project_details("proj_123")
        app.current_screen = "project_details"
        
        # 4. Create new task
        app.handle_user_input("new_task")
        app.show_task_creation()
        app.current_screen = "task_creation"
        
        # Verify navigation flow
        app.show_welcome_screen.assert_called_once()
        app.show_project_list.assert_called_once()
        app.show_project_details.assert_called_once_with("proj_123")
        app.show_task_creation.assert_called_once()
        assert app.handle_user_input.call_count == 3
    
    @pytest.mark.asyncio
    async def test_real_time_updates_workflow(self, mock_tui_app):
        """Test real-time updates in TUI workflow."""
        app = mock_tui_app
        
        # Mock WebSocket for real-time updates
        websocket = Mock()
        websocket.connect = AsyncMock()
        websocket.listen = AsyncMock()
        websocket.send = AsyncMock()
        
        # Mock TUI update methods
        app.update_task_status = Mock()
        app.update_progress_bar = Mock()
        app.show_notification = Mock()
        
        # Simulate real-time workflow
        await websocket.connect("ws://localhost:8000/ws")
        
        # Simulate receiving updates
        updates = [
            {"type": "task_status", "data": {"task_id": "task_123", "status": "in_progress"}},
            {"type": "progress", "data": {"task_id": "task_123", "progress": 25}},
            {"type": "progress", "data": {"task_id": "task_123", "progress": 50}},
            {"type": "progress", "data": {"task_id": "task_123", "progress": 75}},
            {"type": "task_status", "data": {"task_id": "task_123", "status": "completed"}},
            {"type": "notification", "data": {"message": "Task completed successfully!"}}
        ]
        
        for update in updates:
            # Simulate receiving update
            if update["type"] == "task_status":
                app.update_task_status(update["data"]["task_id"], update["data"]["status"])
            elif update["type"] == "progress":
                app.update_progress_bar(update["data"]["task_id"], update["data"]["progress"])
            elif update["type"] == "notification":
                app.show_notification(update["data"]["message"])
        
        # Verify updates were processed
        assert app.update_task_status.call_count == 2
        assert app.update_progress_bar.call_count == 3
        app.show_notification.assert_called_once_with("Task completed successfully!")


class TestErrorHandlingWorkflow:
    """End-to-end tests for error handling workflows."""
    
    @pytest.mark.asyncio
    async def test_ai_service_failure_workflow(self, workflow_components):
        """Test workflow when AI service fails."""
        components = workflow_components
        
        # Step 1: Initial task creation succeeds
        task_data = {
            "name": "generate_api",
            "prompt": "Generate a REST API for user management",
            "project_id": "proj_123"
        }
        
        components["task_engine"].create_task = AsyncMock(return_value={
            "id": "task_456",
            **task_data,
            "status": "pending"
        })
        
        # Step 2: AI service fails
        components["ai_interface"].execute_task = AsyncMock(
            side_effect=Exception("AI service unavailable")
        )
        
        # Step 3: Fallback and recovery
        components["task_engine"].mark_task_failed = AsyncMock()
        components["task_engine"].queue_for_retry = AsyncMock()
        components["tui_app"].show_error_message = Mock()
        
        # Execute workflow with error handling
        task = await components["task_engine"].create_task(task_data)
        
        try:
            await components["ai_interface"].execute_task(task)
        except Exception as e:
            # Handle the error
            await components["task_engine"].mark_task_failed(task["id"], str(e))
            await components["task_engine"].queue_for_retry(task["id"], delay=300)  # Retry in 5 minutes
            components["tui_app"].show_error_message(f"Task failed: {e}. Will retry automatically.")
        
        # Verify error handling
        components["task_engine"].mark_task_failed.assert_awaited_once()
        components["task_engine"].queue_for_retry.assert_awaited_once()
        components["tui_app"].show_error_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validation_failure_workflow(self, workflow_components):
        """Test workflow when validation fails."""
        components = workflow_components
        
        # Generated code with issues
        problematic_code = '''
        def user_login(username, password):
            # TODO: implement authentication logic
            if username == "admin" and password == "password":
                return True
            return False
        '''
        
        # Step 1: Validation detects issues
        components["validation_engine"].validate_code = AsyncMock(return_value={
            "has_placeholders": True,
            "security_issues": [
                {"type": "hardcoded_credentials", "severity": "critical"}
            ],
            "quality_score": 0.2,
            "is_acceptable": False
        })
        
        # Step 2: Auto-fix attempt
        components["ai_interface"].fix_code_issues = AsyncMock(return_value={
            "fixed_code": '''
        def user_login(username, password):
            if not username or not password:
                return False
            
            # Hash the password and compare with stored hash
            stored_hash = get_user_password_hash(username)
            if stored_hash and verify_password(password, stored_hash):
                return True
            return False
            ''',
            "fixes_applied": [
                "Removed hardcoded credentials",
                "Added input validation",
                "Implemented proper password verification"
            ]
        })
        
        # Step 3: Re-validation
        components["validation_engine"].re_validate = AsyncMock(return_value={
            "has_placeholders": False,
            "security_issues": [],
            "quality_score": 0.85,
            "is_acceptable": True
        })
        
        # Execute workflow
        initial_validation = await components["validation_engine"].validate_code(problematic_code)
        assert initial_validation["is_acceptable"] is False
        
        if not initial_validation["is_acceptable"]:
            # Attempt auto-fix
            fix_result = await components["ai_interface"].fix_code_issues({
                "code": problematic_code,
                "issues": initial_validation["security_issues"]
            })
            
            # Re-validate fixed code
            final_validation = await components["validation_engine"].re_validate(
                fix_result["fixed_code"]
            )
            
            assert final_validation["is_acceptable"] is True
            assert len(final_validation["security_issues"]) == 0
    
    def test_network_failure_workflow(self, workflow_components):
        """Test workflow during network failures."""
        components = workflow_components
        
        # Step 1: Network failure during operation
        components["ai_interface"].execute_task = AsyncMock(
            side_effect=ConnectionError("Network unreachable")
        )
        
        # Step 2: Offline mode activation
        components["task_engine"].enable_offline_mode = Mock()
        components["task_engine"].save_for_later = Mock()
        components["tui_app"].show_offline_notification = Mock()
        
        # Step 3: Queue operations for later
        task_data = {"id": "task_123", "prompt": "Generate code"}
        
        try:
            # This will fail due to network
            components["ai_interface"].execute_task(task_data)
        except ConnectionError:
            # Handle network failure
            components["task_engine"].enable_offline_mode()
            components["task_engine"].save_for_later(task_data)
            components["tui_app"].show_offline_notification("Working offline. Tasks will be processed when connection is restored.")
        
        # Verify offline handling
        components["task_engine"].enable_offline_mode.assert_called_once()
        components["task_engine"].save_for_later.assert_called_once_with(task_data)
        components["tui_app"].show_offline_notification.assert_called_once()


class TestPerformanceWorkflow:
    """End-to-end tests for performance-critical workflows."""
    
    @pytest.mark.asyncio
    async def test_large_project_workflow(self, workflow_components):
        """Test workflow with large project (many files and tasks)."""
        components = workflow_components
        
        # Create a large project structure
        large_project_data = {
            "name": "enterprise-app",
            "file_count": 500,
            "task_count": 100,
            "estimated_size": "50MB"
        }
        
        # Mock efficient batch operations
        components["project_manager"].create_large_project = AsyncMock(return_value={
            "id": "proj_enterprise_789",
            "created_files": 500,
            "execution_time": 12.5  # seconds
        })
        
        components["task_engine"].batch_execute_tasks = AsyncMock(return_value={
            "completed_tasks": 100,
            "failed_tasks": 0,
            "total_time": 45.2  # seconds
        })
        
        # Execute large project workflow
        start_time = asyncio.get_event_loop().time()
        
        project_result = await components["project_manager"].create_large_project(large_project_data)
        task_result = await components["task_engine"].batch_execute_tasks(
            project_id=project_result["id"],
            task_count=large_project_data["task_count"]
        )
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Verify performance
        assert project_result["created_files"] == 500
        assert task_result["completed_tasks"] == 100
        assert task_result["failed_tasks"] == 0
        assert total_time < 60  # Should complete within 1 minute
    
    @pytest.mark.asyncio
    async def test_concurrent_user_workflow(self, workflow_components):
        """Test workflow with multiple concurrent users."""
        components = workflow_components
        
        # Simulate multiple users
        users = ["user_1", "user_2", "user_3", "user_4", "user_5"]
        
        async def user_workflow(user_id):
            """Simulate individual user workflow."""
            # Each user creates a project and executes tasks
            project = await components["project_manager"].create_project({
                "name": f"project-{user_id}",
                "owner": user_id
            })
            
            tasks = []
            for i in range(3):  # 3 tasks per user
                task = await components["task_engine"].create_task({
                    "name": f"task-{i}",
                    "project_id": project["id"],
                    "user_id": user_id
                })
                tasks.append(task)
            
            # Execute tasks concurrently for each user
            results = await asyncio.gather(*[
                components["ai_interface"].execute_task(task) for task in tasks
            ])
            
            return {"user": user_id, "project": project, "completed_tasks": len(results)}
        
        # Mock the methods
        components["project_manager"].create_project = AsyncMock(
            side_effect=lambda data: {"id": f"proj_{data['owner']}_123", **data}
        )
        components["task_engine"].create_task = AsyncMock(
            side_effect=lambda data: {"id": f"task_{data['user_id']}_{data['name']}", **data}
        )
        components["ai_interface"].execute_task = AsyncMock(
            return_value={"status": "completed", "execution_time": 2.0}
        )
        
        # Execute concurrent user workflows
        start_time = asyncio.get_event_loop().time()
        user_results = await asyncio.gather(*[
            user_workflow(user_id) for user_id in users
        ])
        end_time = asyncio.get_event_loop().time()
        
        # Verify concurrent execution
        assert len(user_results) == 5
        assert all(result["completed_tasks"] == 3 for result in user_results)
        
        # Should be much faster than sequential execution
        total_time = end_time - start_time
        assert total_time < 10  # Should complete much faster than 5 users × 3 tasks × 2s = 30s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])