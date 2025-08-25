#!/usr/bin/env python3
"""
Integration Example Usage

This module demonstrates how to use the Claude-TIU integration modules
for comprehensive project management and AI-powered development.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from claude_code import ClaudeCodeClient, ExecutionContext
from claude_flow import ClaudeFlowOrchestrator, SwarmManager, SwarmConfig, SwarmTopology
from git_manager import GitManager, BranchStrategy
from file_system import FileSystemManager, SafetyLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_integration_workflow():
    """
    Demonstrate a complete integration workflow combining all modules:
    1. Initialize project structure
    2. Set up Git repository
    3. Create development workflow with Claude Flow
    4. Execute coding tasks with Claude Code
    5. Manage files safely
    """
    
    project_path = Path("./demo-project")
    
    # Initialize all integration components
    file_manager = FileSystemManager(
        base_path=project_path,
        safety_level=SafetyLevel.STANDARD,
        enable_backups=True
    )
    
    git_manager = GitManager(
        repository_path=project_path,
        branch_strategy=BranchStrategy.GITHUB_FLOW,
        safe_mode=True
    )
    
    swarm_manager = SwarmManager(
        claude_flow_binary="npx claude-flow@alpha",
        enable_monitoring=True
    )
    
    orchestrator = ClaudeFlowOrchestrator(
        swarm_manager=swarm_manager,
        enable_caching=True
    )
    
    claude_client = ClaudeCodeClient(
        cli_path="claude",
        max_concurrent_executions=3,
        enable_streaming=True,
        cache_enabled=True
    )
    
    try:
        # Step 1: Create project structure
        logger.info("🚀 Step 1: Creating project structure")
        
        await file_manager.create_directory("src")
        await file_manager.create_directory("tests")
        await file_manager.create_directory("docs")
        await file_manager.create_directory("config")
        
        # Create initial files
        await file_manager.write_file(
            "README.md",
            "# Demo Project\n\nA demonstration of Claude-TIU integration capabilities.\n"
        )
        
        await file_manager.write_file(
            "requirements.txt",
            "fastapi==0.104.1\npydantic==2.5.0\naiofiles==23.2.1\n"
        )
        
        await file_manager.write_file(
            "src/__init__.py",
            '"""Demo project package."""\n'
        )
        
        logger.info("✅ Project structure created")
        
        # Step 2: Initialize Git repository
        logger.info("📁 Step 2: Initializing Git repository")
        
        await git_manager.initialize_repository(project_path, initial_branch="main")
        
        # Add files to Git
        await git_manager.add_files([
            "README.md", "requirements.txt", "src/__init__.py",
            "src", "tests", "docs", "config"
        ])
        
        # Create initial commit
        await git_manager.commit("Initial project setup with basic structure")
        
        logger.info("✅ Git repository initialized")
        
        # Step 3: Set up Claude Flow workflow
        logger.info("🔄 Step 3: Setting up Claude Flow orchestration")
        
        # Initialize swarm with mesh topology for complex coordination
        swarm_config = SwarmConfig(
            topology=SwarmTopology.MESH,
            max_agents=6,
            strategy="adaptive",
            enable_coordination=True,
            enable_learning=True
        )
        
        swarm_id = await swarm_manager.initialize_swarm(swarm_config)
        logger.info(f"Swarm {swarm_id} initialized")
        
        # Spawn specialized agents
        backend_agent = await swarm_manager.spawn_agent(
            swarm_id, 
            AgentType.BACKEND_DEV,
            name="FastAPI-Backend-Developer"
        )
        
        test_agent = await swarm_manager.spawn_agent(
            swarm_id,
            AgentType.TEST_ENGINEER,
            name="Python-Test-Engineer"
        )
        
        logger.info("✅ Agents spawned and ready")
        
        # Step 4: Execute development workflow
        logger.info("💻 Step 4: Executing development workflow")
        
        # Define project specification
        project_spec = {
            "name": "FastAPI Demo API",
            "description": "A simple REST API built with FastAPI for demonstration",
            "requires_backend": True,
            "requires_frontend": False,
            "requires_database": False,
            "requires_testing": True,
            "features": ["health_check", "user_endpoints", "validation"],
            "technology_stack": {
                "framework": "fastapi",
                "language": "python",
                "testing": "pytest"
            },
            "estimated_files": 8
        }
        
        # Execute orchestrated workflow
        workflow_result = await orchestrator.orchestrate_development_workflow(
            project_spec
        )
        
        if workflow_result.is_success:
            logger.info("✅ Workflow executed successfully")
        else:
            logger.error(f"❌ Workflow failed: {workflow_result.error_message}")
            return
        
        # Step 5: Execute specific coding tasks with Claude Code
        logger.info("🤖 Step 5: Executing coding tasks with Claude Code")
        
        # Create execution context
        context = ExecutionContext(
            project_path=project_path,
            files=["src/main.py", "src/models.py"],
            requirements="Create a FastAPI application with user management endpoints",
            coding_standards={
                "style": "PEP8",
                "type_hints": True,
                "docstrings": True
            }
        )
        
        # Generate main application
        main_app_result = await claude_client.execute_coding_task(
            prompt="""
            Create a FastAPI main application with the following features:
            1. Health check endpoint
            2. User CRUD endpoints (create, read, update, delete)
            3. Pydantic models for request/response validation
            4. Proper error handling and logging
            5. API documentation with OpenAPI/Swagger
            
            Structure the code professionally with proper imports, type hints, and docstrings.
            """,
            context=context,
            format_output="json",
            streaming_callback=lambda output: logger.info(f"AI: {output.strip()}")
        )
        
        if main_app_result.is_success:
            # Write generated code to file
            await file_manager.write_file(
                "src/main.py",
                main_app_result.generated_code,
                atomic=True,
                create_backup=True
            )
            logger.info("✅ Main application generated and saved")
        
        # Generate tests
        test_context = ExecutionContext(
            project_path=project_path,
            files=["src/main.py", "tests/test_main.py"],
            requirements="Create comprehensive tests for the FastAPI application"
        )
        
        test_result = await claude_client.execute_coding_task(
            prompt="""
            Create comprehensive pytest tests for the FastAPI application including:
            1. Test client setup with FastAPI TestClient
            2. Health check endpoint tests
            3. User CRUD endpoint tests with various scenarios
            4. Input validation tests
            5. Error handling tests
            6. Proper fixtures and test data
            
            Use pytest best practices with clear test names and good coverage.
            """,
            context=test_context,
            format_output="json"
        )
        
        if test_result.is_success:
            await file_manager.write_file(
                "tests/test_main.py",
                test_result.generated_code,
                atomic=True
            )
            logger.info("✅ Tests generated and saved")
        
        # Step 6: Commit changes to Git
        logger.info("📝 Step 6: Committing changes to Git")
        
        # Add all new/modified files
        await git_manager.add_files([
            "src/main.py", "tests/test_main.py"
        ])
        
        # Create feature commit
        await git_manager.commit(
            "Implement FastAPI application with user management and comprehensive tests\n\n"
            "- Added main FastAPI application with health check and user CRUD endpoints\n"
            "- Implemented Pydantic models for request/response validation\n"
            "- Added comprehensive pytest test suite with fixtures\n"
            "- Included proper error handling and API documentation\n\n"
            "🤖 Generated with Claude Code\n"
            "Co-Authored-By: Claude <noreply@anthropic.com>"
        )
        
        logger.info("✅ Changes committed to Git")
        
        # Step 7: Generate project reports and metrics
        logger.info("📊 Step 7: Generating integration reports")
        
        # File system metrics
        fs_metrics = file_manager.get_metrics()
        logger.info(f"File System Operations: {fs_metrics['total_operations']} "
                   f"({fs_metrics['success_rate']:.1%} success rate)")
        
        # Git metrics
        git_metrics = git_manager.get_metrics()
        logger.info(f"Git Operations: {git_metrics['total_operations']} "
                   f"({git_metrics['success_rate']:.1%} success rate)")
        
        # Claude Code metrics
        ai_metrics = claude_client.get_metrics()
        logger.info(f"AI Operations: {ai_metrics['total_executions']} "
                   f"({ai_metrics['success_rate']:.1%} success rate, "
                   f"{ai_metrics['cache_hit_rate']:.1%} cache hit rate)")
        
        # Claude Flow metrics
        flow_metrics = orchestrator.get_metrics()
        logger.info(f"Workflow Operations: {flow_metrics['total_workflows']} "
                   f"({flow_metrics['success_rate']:.1%} success rate)")
        
        # Generate comprehensive report
        report = {
            "project": {
                "name": project_spec["name"],
                "path": str(project_path),
                "files_created": fs_metrics['total_operations'],
                "commits": git_metrics['total_operations']
            },
            "ai_integration": {
                "executions": ai_metrics['total_executions'],
                "tokens_used": ai_metrics.get('tokens_used', 0),
                "average_response_time": ai_metrics['average_execution_time'],
                "cache_efficiency": ai_metrics['cache_hit_rate']
            },
            "workflow_performance": {
                "workflows_completed": flow_metrics['total_workflows'],
                "average_execution_time": flow_metrics['average_execution_time'],
                "success_rate": flow_metrics['success_rate']
            },
            "safety_metrics": {
                "backups_created": fs_metrics.get('backup_operations', 0),
                "validation_checks": "All operations validated",
                "rollback_points": git_metrics.get('rollback_points_available', 0)
            }
        }
        
        # Save report
        await file_manager.write_file(
            "integration_report.json",
            json.dumps(report, indent=2),
            atomic=True
        )
        
        logger.info("📋 Integration report generated")
        
        # Health checks
        logger.info("🔍 Performing health checks")
        
        fs_health = await file_manager.health_check()
        git_health = await git_manager.health_check()
        ai_health = await claude_client.health_check()
        flow_health = await orchestrator.health_check()
        
        logger.info(f"File System Health: {fs_health['status']}")
        logger.info(f"Git Health: {git_health['status']}")
        logger.info(f"Claude Code Health: {ai_health['status']}")
        logger.info(f"Claude Flow Health: {flow_health['status']}")
        
        logger.info("🎉 Integration workflow demonstration completed successfully!")
        
        return {
            "success": True,
            "project_path": str(project_path),
            "metrics": {
                "file_system": fs_metrics,
                "git": git_metrics,
                "claude_code": ai_metrics,
                "claude_flow": flow_metrics
            },
            "health_status": {
                "file_system": fs_health['status'],
                "git": git_health['status'],
                "claude_code": ai_health['status'],
                "claude_flow": flow_health['status']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Integration workflow failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def demonstrate_individual_modules():
    """Demonstrate individual module capabilities"""
    
    logger.info("🔧 Demonstrating individual module capabilities")
    
    # File System Manager Demo
    logger.info("📁 File System Manager Features:")
    
    file_manager = FileSystemManager(safety_level=SafetyLevel.STANDARD)
    
    # Safe file operations
    await file_manager.write_file("demo.txt", "Hello, World!", atomic=True)
    read_result = await file_manager.read_file("demo.txt")
    logger.info(f"  - File content: {read_result.metadata['content'][:20]}...")
    
    # File analysis
    file_info = await file_manager.get_file_info("demo.txt")
    logger.info(f"  - File type: {file_info.file_type.value}")
    logger.info(f"  - File size: {file_info.size} bytes")
    
    # Git Manager Demo
    logger.info("📝 Git Manager Features:")
    
    git_manager = GitManager(safe_mode=True)
    
    if git_manager.repo:
        status = await git_manager.get_repository_status()
        logger.info(f"  - Current branch: {status['current_branch']}")
        logger.info(f"  - Total commits: {status['statistics']['total_commits']}")
        logger.info(f"  - Is dirty: {status['is_dirty']}")
    
    # Claude Code Client Demo
    logger.info("🤖 Claude Code Client Features:")
    
    claude_client = ClaudeCodeClient(enable_streaming=True)
    
    # Simple coding task
    simple_result = await claude_client.execute_coding_task(
        prompt="Create a Python function that calculates the factorial of a number",
        format_output="text"
    )
    
    if simple_result.is_success:
        logger.info("  - Simple coding task completed successfully")
        logger.info(f"  - Execution time: {simple_result.execution_time:.2f}s")
    
    # Performance metrics
    metrics = claude_client.get_metrics()
    logger.info(f"  - Total executions: {metrics['total_executions']}")
    logger.info(f"  - Success rate: {metrics['success_rate']:.1%}")
    
    # Claude Flow Orchestrator Demo
    logger.info("🔄 Claude Flow Orchestrator Features:")
    
    swarm_manager = SwarmManager()
    orchestrator = ClaudeFlowOrchestrator(swarm_manager=swarm_manager)
    
    # Simple project specification
    simple_project = {
        "name": "Calculator Module",
        "description": "A simple calculator with basic operations",
        "requires_backend": False,
        "requires_frontend": False,
        "requires_testing": True,
        "features": ["add", "subtract", "multiply", "divide"],
        "technology_stack": {"language": "python"},
        "estimated_files": 3
    }
    
    try:
        workflow_result = await orchestrator.orchestrate_development_workflow(simple_project)
        logger.info(f"  - Workflow status: {workflow_result.status.value}")
        logger.info(f"  - Steps completed: {workflow_result.steps_completed}/{workflow_result.steps_total}")
    except Exception as e:
        logger.warning(f"  - Workflow demonstration skipped: {e}")
    
    logger.info("✅ Individual module demonstration completed")


if __name__ == "__main__":
    """
    Main execution entry point
    
    This demonstrates both the full integration workflow and individual
    module capabilities of the Claude-TIU integration system.
    """
    
    logger.info("🚀 Starting Claude-TIU Integration Demonstration")
    
    async def main():
        try:
            # Run individual module demonstrations
            await demonstrate_individual_modules()
            
            # Run full integration workflow
            result = await demonstrate_integration_workflow()
            
            if result["success"]:
                logger.info("🎉 All demonstrations completed successfully!")
                logger.info(f"📁 Project created at: {result['project_path']}")
            else:
                logger.error(f"❌ Demonstration failed: {result['error']}")
                
        except KeyboardInterrupt:
            logger.info("🛑 Demonstration interrupted by user")
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
    
    # Run the demonstration
    asyncio.run(main())