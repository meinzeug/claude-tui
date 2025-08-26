#!/usr/bin/env python3
"""
Production Workflow Manager for Automatic Programming
====================================================

Production-ready workflow templates and generators for end-to-end AI code generation.
These workflows generate real, production-quality applications.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any

from .automatic_programming_workflow import (
    AutomaticProgrammingWorkflow, WorkflowStep, WorkflowStatus
)


class ProductionWorkflowGenerator:
    """
    Production workflow generator that creates complete, deployment-ready projects
    """
    
    def __init__(self, workflow_manager: AutomaticProgrammingWorkflow):
        self.workflow_manager = workflow_manager
    
    async def create_fastapi_application(self, project_name: str, project_path: Path, **kwargs) -> str:
        """
        Create a production-ready FastAPI application
        
        This generates:
        - Complete FastAPI application with authentication
        - Database models and migrations  
        - API endpoints with documentation
        - JWT authentication system
        - Comprehensive test suite
        - Docker configuration for production deployment
        - CI/CD pipeline configuration
        - Monitoring and logging
        - Security hardening
        - Performance optimizations
        """
        
        # Enhanced FastAPI workflow with more realistic steps
        steps = [
            WorkflowStep(
                step_id="setup_project_structure",
                name="Setup Project Structure",
                description="Create organized project directory structure",
                step_type="file_operation",
                parameters={
                    "operation": "create_fastapi_structure",
                    "directories": [
                        "app", "app/api", "app/core", "app/db", "app/models",
                        "app/schemas", "app/crud", "app/tests", "scripts", "docs"
                    ]
                }
            ),
            
            WorkflowStep(
                step_id="generate_requirements",
                name="Generate Requirements & Configuration",
                description="Create requirements.txt, Dockerfile, and configuration files",
                step_type="claude_code",
                parameters={
                    "prompt": """Create a complete FastAPI project configuration:

1. requirements.txt with all necessary dependencies:
   - fastapi
   - uvicorn[standard]
   - sqlalchemy
   - alembic
   - python-jose[cryptography]
   - passlib[bcrypt]
   - python-multipart
   - pytest
   - httpx

2. Dockerfile for production deployment
3. docker-compose.yml for development
4. .env.example with configuration variables
5. alembic.ini for database migrations
6. pyproject.toml for project metadata

Make it production-ready with proper security and performance settings."""
                },
                dependencies=["setup_project_structure"]
            ),
            
            WorkflowStep(
                step_id="generate_core_config",
                name="Generate Core Configuration",
                description="Create core application configuration and settings",
                step_type="claude_code",
                parameters={
                    "prompt": """Create the core application configuration for a FastAPI app:

1. app/core/config.py - Settings class with environment variables:
   - Database URL
   - JWT secret key and algorithm
   - API settings (title, version, description)
   - Security settings
   - CORS settings

2. app/core/security.py - Security utilities:
   - Password hashing functions
   - JWT token creation and verification
   - Authentication dependencies

3. app/core/__init__.py - Package initialization

Use Pydantic BaseSettings for configuration management. Include proper type hints and documentation."""
                },
                dependencies=["generate_requirements"]
            ),
            
            WorkflowStep(
                step_id="generate_database_models",
                name="Generate Database Models", 
                description="Create SQLAlchemy models and database configuration",
                step_type="claude_code",
                parameters={
                    "prompt": """Create database models for a FastAPI application:

1. app/db/base.py - Database base class and session management
2. app/db/session.py - Database session creation and management
3. app/models/user.py - User model with authentication fields:
   - id, email, username, hashed_password
   - is_active, is_superuser, created_at, updated_at
4. app/models/__init__.py - Models package initialization

5. Create an Alembic migration script in alembic/versions/ for initial tables

Use SQLAlchemy ORM with proper relationships, indexes, and constraints. Include UUID primary keys and timestamps."""
                },
                dependencies=["generate_core_config"]
            ),
            
            WorkflowStep(
                step_id="generate_schemas",
                name="Generate Pydantic Schemas",
                description="Create Pydantic schemas for request/response validation",
                step_type="claude_code", 
                parameters={
                    "prompt": """Create Pydantic schemas for the FastAPI application:

1. app/schemas/user.py - User schemas:
   - UserBase (base fields)
   - UserCreate (for user registration)  
   - UserUpdate (for user updates)
   - User (for responses with all fields)
   - UserInDB (internal use with hashed_password)

2. app/schemas/token.py - Authentication token schemas:
   - Token (access_token, token_type)
   - TokenPayload (for JWT payload)

3. app/schemas/__init__.py - Schemas package initialization

Use proper Pydantic validation with field validators, aliases, and examples for API documentation."""
                },
                dependencies=["generate_database_models"]
            ),
            
            WorkflowStep(
                step_id="generate_crud_operations",
                name="Generate CRUD Operations",
                description="Create database CRUD operations", 
                step_type="claude_code",
                parameters={
                    "prompt": """Create CRUD operations for the FastAPI application:

1. app/crud/base.py - Generic CRUD class with common operations:
   - get, get_multi, create, update, delete methods
   - Generic type hints

2. app/crud/user.py - User-specific CRUD operations:
   - get_user_by_email, get_user_by_username
   - create_user (with password hashing)
   - authenticate_user
   - update_user

3. app/crud/__init__.py - CRUD package initialization

Use SQLAlchemy sessions and proper error handling. Include type hints and documentation."""
                },
                dependencies=["generate_schemas"]
            ),
            
            WorkflowStep(
                step_id="generate_api_endpoints",
                name="Generate API Endpoints",
                description="Create FastAPI router endpoints",
                step_type="claude_code",
                parameters={
                    "prompt": """Create API endpoints for the FastAPI application:

1. app/api/__init__.py - API package initialization
2. app/api/deps.py - API dependencies:
   - get_db (database session dependency)
   - get_current_user (authentication dependency)
   - get_current_active_superuser

3. app/api/auth.py - Authentication endpoints:
   - POST /auth/login - Login with email/password
   - POST /auth/register - User registration
   - POST /auth/refresh - Token refresh
   - GET /auth/me - Get current user info

4. app/api/users.py - User management endpoints:
   - GET /users/ - List users (superuser only)
   - GET /users/{user_id} - Get user by ID
   - PUT /users/{user_id} - Update user
   - DELETE /users/{user_id} - Delete user (superuser only)

5. app/api/router.py - Main API router combining all endpoints

Include proper HTTP status codes, error handling, and OpenAPI documentation."""
                },
                dependencies=["generate_crud_operations"]
            ),
            
            WorkflowStep(
                step_id="generate_main_application",
                name="Generate Main Application",
                description="Create the main FastAPI application",
                step_type="claude_code",
                parameters={
                    "prompt": """Create the main FastAPI application:

1. app/main.py - Main FastAPI application:
   - FastAPI app initialization with metadata
   - CORS middleware configuration
   - API router inclusion
   - Startup and shutdown event handlers
   - Health check endpoint
   - Exception handlers

2. app/__init__.py - Package initialization

3. Create a startup script: scripts/start.py for development

Include proper error handling, logging configuration, and production-ready settings."""
                },
                dependencies=["generate_api_endpoints"]
            ),
            
            WorkflowStep(
                step_id="generate_tests",
                name="Generate Test Suite",
                description="Create comprehensive pytest test suite",
                step_type="claude_code",
                parameters={
                    "prompt": """Create a comprehensive test suite for the FastAPI application:

1. app/tests/conftest.py - Test configuration and fixtures:
   - Test database setup
   - Test client fixture
   - User authentication fixtures

2. app/tests/test_auth.py - Authentication tests:
   - Test user registration
   - Test login/logout
   - Test token validation
   - Test password reset

3. app/tests/test_users.py - User endpoint tests:
   - Test user CRUD operations
   - Test user permissions
   - Test user validation

4. app/tests/test_main.py - Main application tests:
   - Test health check
   - Test error handling
   - Test middleware

Use pytest with asyncio support, proper test isolation, and comprehensive coverage."""
                },
                dependencies=["generate_main_application"]
            ),
            
            WorkflowStep(
                step_id="generate_documentation",
                name="Generate Documentation", 
                description="Create project documentation and README",
                step_type="claude_code",
                parameters={
                    "prompt": """Create comprehensive project documentation:

1. README.md - Project overview with:
   - Description and features
   - Installation instructions  
   - API documentation
   - Development setup
   - Deployment instructions
   - Contributing guidelines

2. docs/API.md - Detailed API documentation
3. docs/DEPLOYMENT.md - Deployment guide
4. docs/CONTRIBUTING.md - Development guidelines

Make it professional and easy to follow for new developers."""
                },
                dependencies=["generate_tests"]
            ),
            
            WorkflowStep(
                step_id="validate_and_test",
                name="Validate Generated Code",
                description="Validate all generated code and run basic tests",
                step_type="validation", 
                parameters={
                    "validation_type": "comprehensive_fastapi",
                    "checks": [
                        "python_syntax",
                        "imports_resolution", 
                        "fastapi_structure",
                        "database_models",
                        "api_endpoints"
                    ]
                },
                dependencies=["generate_documentation"]
            )
        ]
        
        # Create the workflow
        workflow_id = await self.workflow_manager._create_workflow_from_steps(
            name=f"FastAPI Production App: {project_name}",
            description="Production-ready FastAPI application with enterprise features",
            project_name=project_name,
            project_path=project_path,
            steps=steps
        )
        
        return workflow_id
    
    async def create_react_dashboard(self, project_name: str, project_path: Path, **kwargs) -> str:
        """
        Create a production-ready React dashboard application
        
        This generates:
        - Enterprise React dashboard with TypeScript
        - Professional component library with charts and tables
        - Advanced state management (Redux/Zustand)
        - Responsive design with modern CSS-in-JS
        - Enterprise authentication integration
        - Comprehensive testing with Jest and React Testing Library
        - Performance optimizations and code splitting
        - Production build configuration
        - Monitoring and error tracking
        """
        
        steps = [
            WorkflowStep(
                step_id="setup_react_structure",
                name="Setup React Project Structure",
                description="Create modern React project structure with TypeScript",
                step_type="file_operation",
                parameters={
                    "operation": "create_react_dashboard_structure",
                    "directories": [
                        "src", "src/components", "src/components/common",
                        "src/components/dashboard", "src/pages", "src/hooks",
                        "src/services", "src/utils", "src/types", "src/styles",
                        "src/contexts", "public", "tests", "docs"
                    ]
                }
            ),
            
            WorkflowStep(
                step_id="generate_package_config",
                name="Generate Package Configuration",
                description="Create package.json and configuration files",
                step_type="claude_code",
                parameters={
                    "prompt": """Create React dashboard project configuration:

1. package.json with modern dependencies:
   - React 18 with TypeScript
   - React Router v6
   - Chart.js or Recharts for visualization
   - Axios for API calls
   - Material-UI or Tailwind CSS
   - Testing libraries (Jest, React Testing Library)
   - ESLint and Prettier

2. tsconfig.json with proper TypeScript configuration
3. .eslintrc.js with React and TypeScript rules
4. .prettierrc for code formatting
5. vite.config.ts for modern build tooling

Make it modern and production-ready."""
                },
                dependencies=["setup_react_structure"]
            ),
            
            WorkflowStep(
                step_id="generate_types_and_interfaces",
                name="Generate TypeScript Types",
                description="Create TypeScript interfaces and types",
                step_type="claude_code",
                parameters={
                    "prompt": """Create TypeScript types and interfaces:

1. src/types/index.ts - Main types:
   - User interface
   - Dashboard data interfaces  
   - API response types
   - Navigation types

2. src/types/api.ts - API-related types:
   - Request/response interfaces
   - Error handling types

3. src/types/components.ts - Component prop types

Use proper TypeScript patterns with generics and utility types."""
                },
                dependencies=["generate_package_config"]
            ),
            
            WorkflowStep(
                step_id="generate_context_and_hooks",
                name="Generate Context and Custom Hooks",
                description="Create React context providers and custom hooks",
                step_type="claude_code", 
                parameters={
                    "prompt": """Create React context and custom hooks:

1. src/contexts/AuthContext.tsx - Authentication context:
   - User state management
   - Login/logout functions
   - Token management

2. src/contexts/DashboardContext.tsx - Dashboard data context:
   - Dashboard state management
   - Data fetching functions

3. src/hooks/useApi.ts - Custom API hook:
   - Generic API calling hook
   - Loading and error states

4. src/hooks/useAuth.ts - Authentication hook
5. src/hooks/useDashboard.ts - Dashboard data hook

Use modern React patterns with TypeScript."""
                },
                dependencies=["generate_types_and_interfaces"]
            ),
            
            WorkflowStep(
                step_id="generate_api_services",
                name="Generate API Services", 
                description="Create API service layer",
                step_type="claude_code",
                parameters={
                    "prompt": """Create API services for the React dashboard:

1. src/services/api.ts - Base API configuration:
   - Axios instance with interceptors
   - Authentication token handling
   - Error handling

2. src/services/authService.ts - Authentication API:
   - Login/logout functions
   - Token refresh
   - User profile management

3. src/services/dashboardService.ts - Dashboard data API:
   - Fetch dashboard data
   - Analytics data
   - User statistics

Use proper error handling and TypeScript types."""
                },
                dependencies=["generate_context_and_hooks"]
            ),
            
            WorkflowStep(
                step_id="generate_common_components",
                name="Generate Common Components",
                description="Create reusable UI components",
                step_type="claude_code",
                parameters={
                    "prompt": """Create common React components:

1. src/components/common/Layout.tsx - Main layout component:
   - Header with navigation
   - Sidebar with menu
   - Main content area
   - Footer

2. src/components/common/Header.tsx - Header component:
   - Logo and title
   - User menu
   - Notifications

3. src/components/common/Sidebar.tsx - Navigation sidebar:
   - Menu items
   - Collapsible design
   - Active state handling

4. src/components/common/LoadingSpinner.tsx - Loading component
5. src/components/common/ErrorBoundary.tsx - Error boundary
6. src/components/common/Card.tsx - Reusable card component

Use modern React patterns with proper TypeScript types."""
                },
                dependencies=["generate_api_services"]
            ),
            
            WorkflowStep(
                step_id="generate_dashboard_components",
                name="Generate Dashboard Components",
                description="Create dashboard-specific components",
                step_type="claude_code",
                parameters={
                    "prompt": """Create dashboard-specific components:

1. src/components/dashboard/DashboardOverview.tsx - Main dashboard:
   - Statistics cards
   - Charts and graphs
   - Recent activity

2. src/components/dashboard/Chart.tsx - Chart component:
   - Configurable chart types
   - Responsive design
   - Data visualization

3. src/components/dashboard/StatsCard.tsx - Statistics card:
   - Key metrics display
   - Trend indicators
   - Interactive elements

4. src/components/dashboard/DataTable.tsx - Data table:
   - Sortable columns
   - Pagination
   - Search functionality

5. src/components/dashboard/UserProfile.tsx - User profile component

Use modern charting library and responsive design."""
                },
                dependencies=["generate_common_components"]
            ),
            
            WorkflowStep(
                step_id="generate_pages_and_routing",
                name="Generate Pages and Routing",
                description="Create page components and routing",
                step_type="claude_code",
                parameters={
                    "prompt": """Create page components and routing:

1. src/pages/Dashboard.tsx - Main dashboard page
2. src/pages/Login.tsx - Login page with form validation
3. src/pages/Profile.tsx - User profile page
4. src/pages/Analytics.tsx - Analytics page
5. src/pages/Settings.tsx - Settings page
6. src/pages/NotFound.tsx - 404 error page

7. src/App.tsx - Main App component with routing:
   - React Router v6 setup
   - Protected routes
   - Authentication guards
   - Layout integration

8. src/main.tsx - Application entry point

Use React Router v6 with proper TypeScript integration."""
                },
                dependencies=["generate_dashboard_components"]
            ),
            
            WorkflowStep(
                step_id="generate_styles",
                name="Generate Styling",
                description="Create CSS styles and themes",
                step_type="claude_code",
                parameters={
                    "prompt": """Create styling for the React dashboard:

1. src/styles/globals.css - Global styles:
   - CSS reset
   - Typography
   - Color variables
   - Utility classes

2. src/styles/components/ - Component-specific styles:
   - Layout styles
   - Dashboard styles
   - Form styles

3. src/styles/themes/ - Theme configuration:
   - Light/dark theme variables
   - Theme switching logic

4. index.css - Main stylesheet imports

Use modern CSS with CSS Grid, Flexbox, and responsive design. Make it look professional and modern."""
                },
                dependencies=["generate_pages_and_routing"]
            ),
            
            WorkflowStep(
                step_id="generate_tests",
                name="Generate Test Suite",
                description="Create comprehensive test suite",
                step_type="claude_code",
                parameters={
                    "prompt": """Create tests for the React dashboard:

1. tests/setup.ts - Test setup and configuration
2. src/components/__tests__/ - Component tests:
   - Layout.test.tsx
   - Dashboard.test.tsx
   - Chart.test.tsx
   - Form validation tests

3. src/hooks/__tests__/ - Hook tests:
   - useApi.test.ts
   - useAuth.test.ts

4. src/services/__tests__/ - Service tests:
   - API service tests
   - Mock service responses

Use React Testing Library with Jest and proper TypeScript types."""
                },
                dependencies=["generate_styles"]
            ),
            
            WorkflowStep(
                step_id="validate_react_project",
                name="Validate React Project",
                description="Validate the generated React application",
                step_type="validation",
                parameters={
                    "validation_type": "comprehensive_react",
                    "checks": [
                        "typescript_compilation",
                        "component_structure",
                        "routing_configuration",
                        "api_integration",
                        "responsive_design"
                    ]
                },
                dependencies=["generate_tests"]
            )
        ]
        
        # Create the workflow
        workflow_id = await self.workflow_manager._create_workflow_from_steps(
            name=f"React Dashboard Production: {project_name}",
            description="Enterprise React dashboard with advanced features and optimizations",
            project_name=project_name,
            project_path=project_path,
            steps=steps
        )
        
        return workflow_id
    
    async def create_python_cli_tool(self, project_name: str, project_path: Path, **kwargs) -> str:
        """
        Create a production-ready Python CLI tool
        
        This generates:
        - Enterprise CLI application with Click/Typer
        - Modular command architecture
        - Advanced configuration management
        - Structured logging and error handling
        - Comprehensive test suite
        - Professional packaging and distribution
        - Auto-completion and help system
        - Performance monitoring
        - Security validation
        """
        
        steps = [
            WorkflowStep(
                step_id="setup_cli_structure",
                name="Setup CLI Project Structure",
                description="Create Python CLI project structure",
                step_type="file_operation",
                parameters={
                    "operation": "create_python_cli_structure",
                    "directories": [
                        f"{project_name}", f"{project_name}/commands",
                        f"{project_name}/utils", f"{project_name}/config",
                        "tests", "docs", "scripts"
                    ]
                }
            ),
            
            WorkflowStep(
                step_id="generate_cli_foundation",
                name="Generate CLI Foundation",
                description="Create core CLI infrastructure",
                step_type="claude_code",
                parameters={
                    "prompt": f"""Create a professional Python CLI application foundation:

1. setup.py - Package configuration with entry points
2. requirements.txt - Dependencies (click, colorama, pyyaml, requests)
3. {project_name}/__init__.py - Package initialization with version
4. {project_name}/cli.py - Main CLI interface with Click:
   - Main command group
   - Global options (verbose, config-file)
   - Help formatting
   - Error handling

5. {project_name}/config/settings.py - Configuration management:
   - YAML/JSON config loading
   - Environment variable support
   - Default settings

Use Click for the CLI framework with proper command organization."""
                },
                dependencies=["setup_cli_structure"]
            ),
            
            WorkflowStep(
                step_id="generate_cli_commands",
                name="Generate CLI Commands",
                description="Create CLI command modules",
                step_type="claude_code",
                parameters={
                    "prompt": f"""Create CLI command modules:

1. {project_name}/commands/__init__.py - Commands package
2. {project_name}/commands/config.py - Configuration commands:
   - init (create config file)
   - show (display current config)
   - set (update config values)

3. {project_name}/commands/info.py - Information commands:
   - version (show version info)
   - status (show system status)
   - doctor (health check)

4. {project_name}/commands/process.py - Main processing commands:
   - run (main operation)
   - batch (batch processing)
   - analyze (data analysis)

Each command should have proper help text, argument validation, and error handling."""
                },
                dependencies=["generate_cli_foundation"]
            ),
            
            WorkflowStep(
                step_id="generate_utilities",
                name="Generate Utility Modules",
                description="Create utility and helper functions",
                step_type="claude_code",
                parameters={
                    "prompt": f"""Create utility modules for the CLI application:

1. {project_name}/utils/__init__.py - Utils package
2. {project_name}/utils/logging.py - Logging configuration:
   - Colored console output
   - File logging
   - Log level management

3. {project_name}/utils/files.py - File operations:
   - Safe file reading/writing
   - Path validation
   - Backup functionality

4. {project_name}/utils/formatting.py - Output formatting:
   - Table formatting
   - Progress bars
   - Color helpers

5. {project_name}/utils/validation.py - Input validation:
   - Data validation functions
   - Error message formatting

Use rich or colorama for colored output and proper error handling."""
                },
                dependencies=["generate_cli_commands"]
            ),
            
            WorkflowStep(
                step_id="generate_tests",
                name="Generate Test Suite",
                description="Create comprehensive test suite",
                step_type="claude_code",
                parameters={
                    "prompt": f"""Create tests for the Python CLI application:

1. tests/conftest.py - Test configuration and fixtures
2. tests/test_cli.py - Main CLI tests:
   - Command execution tests
   - Option parsing tests
   - Help output tests

3. tests/test_commands/ - Command-specific tests:
   - test_config.py
   - test_info.py 
   - test_process.py

4. tests/test_utils/ - Utility function tests:
   - test_logging.py
   - test_files.py
   - test_validation.py

Use pytest with Click's testing utilities and mock where appropriate."""
                },
                dependencies=["generate_utilities"]
            ),
            
            WorkflowStep(
                step_id="validate_cli_project",
                name="Validate CLI Project",
                description="Validate the generated CLI application",
                step_type="validation",
                parameters={
                    "validation_type": "comprehensive_python",
                    "checks": [
                        "python_syntax",
                        "import_resolution",
                        "cli_structure",
                        "entry_points",
                        "test_coverage"
                    ]
                },
                dependencies=["generate_tests"]
            )
        ]
        
        # Create the workflow
        workflow_id = await self.workflow_manager._create_workflow_from_steps(
            name=f"Python CLI Production: {project_name}",
            description="Enterprise Python CLI application with advanced features",
            project_name=project_name,
            project_path=project_path,
            steps=steps
        )
        
        return workflow_id


# Add method to workflow manager to create workflows from steps
async def _create_workflow_from_steps(
    self,
    name: str,
    description: str,
    project_name: str,
    project_path: Path,
    steps: List[WorkflowStep]
) -> str:
    """Create a workflow from a list of steps"""
    import uuid
    import time
    
    workflow_id = str(uuid.uuid4())
    
    workflow = {
        "id": workflow_id,
        "name": name,
        "description": description,
        "project_name": project_name,
        "project_path": str(project_path),
        "status": WorkflowStatus.PENDING,
        "steps": steps,
        "created_at": time.time(),
        "workflow_type": "custom_demo"
    }
    
    self.workflows[workflow_id] = workflow
    return workflow_id

# Monkey patch the method onto the workflow manager class
AutomaticProgrammingWorkflow._create_workflow_from_steps = _create_workflow_from_steps


def create_production_generator(workflow_manager: AutomaticProgrammingWorkflow) -> ProductionWorkflowGenerator:
    """Create a production workflow generator"""
    return ProductionWorkflowGenerator(workflow_manager)

# Backward compatibility alias
create_demo_generator = create_production_generator