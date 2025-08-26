#!/usr/bin/env python3
"""
Generate API client SDKs for multiple programming languages.
"""

import os
import subprocess
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for client SDK generation."""
    language: str
    generator: str
    package_name: str
    output_dir: str
    additional_properties: Dict[str, str]
    post_generate_commands: List[str]


class APIClientGenerator:
    """Generate API client SDKs from OpenAPI specification."""
    
    def __init__(self, openapi_spec_path: str, output_base_dir: str):
        self.openapi_spec_path = Path(openapi_spec_path)
        self.output_base_dir = Path(output_base_dir)
        self.clients_config = self._get_client_configs()
    
    def _get_client_configs(self) -> List[ClientConfig]:
        """Get configuration for different client languages."""
        return [
            # Python Client
            ClientConfig(
                language="python",
                generator="python",
                package_name="claude_tui_client",
                output_dir="python",
                additional_properties={
                    "packageName": "claude_tui_client",
                    "projectName": "claude-tui-python-client",
                    "packageVersion": "1.0.0",
                    "packageUrl": "https://github.com/claude-tui/python-client",
                    "generateSourceCodeOnly": "true"
                },
                post_generate_commands=[
                    "cd {output_dir} && python -m pip install -e .",
                    "cd {output_dir} && python -m pytest tests/ || true"
                ]
            ),
            
            # TypeScript/JavaScript Client
            ClientConfig(
                language="typescript",
                generator="typescript-axios",
                package_name="claude-tui-client",
                output_dir="typescript",
                additional_properties={
                    "npmName": "claude-tui-client",
                    "npmVersion": "1.0.0",
                    "npmRepository": "https://github.com/claude-tui/typescript-client",
                    "supportsES6": "true",
                    "withInterfaces": "true",
                    "useSingleRequestParameter": "true"
                },
                post_generate_commands=[
                    "cd {output_dir} && npm install",
                    "cd {output_dir} && npm run build",
                    "cd {output_dir} && npm test || true"
                ]
            ),
            
            # Go Client
            ClientConfig(
                language="go",
                generator="go",
                package_name="claude-tui-go-client",
                output_dir="go",
                additional_properties={
                    "packageName": "claudetui",
                    "packageVersion": "1.0.0",
                    "packageUrl": "github.com/claude-tui/go-client",
                    "withGoCodegenComment": "true",
                    "generateInterfaces": "true"
                },
                post_generate_commands=[
                    "cd {output_dir} && go mod init github.com/claude-tui/go-client",
                    "cd {output_dir} && go mod tidy",
                    "cd {output_dir} && go test ./... || true"
                ]
            ),
            
            # Java Client
            ClientConfig(
                language="java",
                generator="java",
                package_name="com.claudetui.client",
                output_dir="java",
                additional_properties={
                    "groupId": "com.claudetui",
                    "artifactId": "claude-tui-java-client",
                    "artifactVersion": "1.0.0",
                    "artifactDescription": "Claude TUI Java Client Library",
                    "dateLibrary": "java8",
                    "java8": "true",
                    "library": "okhttp-gson"
                },
                post_generate_commands=[
                    "cd {output_dir} && mvn compile",
                    "cd {output_dir} && mvn test || true"
                ]
            ),
            
            # C# Client
            ClientConfig(
                language="csharp",
                generator="csharp",
                package_name="ClaudeTui.Client",
                output_dir="csharp",
                additional_properties={
                    "packageName": "ClaudeTui.Client",
                    "packageVersion": "1.0.0",
                    "clientPackage": "ClaudeTui.Client",
                    "packageCompany": "Claude TUI",
                    "packageAuthors": "Claude TUI Team",
                    "packageCopyright": "Copyright © Claude TUI 2023",
                    "packageDescription": "Claude TUI C# Client Library",
                    "targetFramework": "netstandard2.0"
                },
                post_generate_commands=[
                    "cd {output_dir} && dotnet build",
                    "cd {output_dir} && dotnet test || true"
                ]
            ),
            
            # Ruby Client
            ClientConfig(
                language="ruby",
                generator="ruby",
                package_name="claude_tui_client",
                output_dir="ruby",
                additional_properties={
                    "gemName": "claude_tui_client",
                    "gemVersion": "1.0.0",
                    "gemSummary": "Claude TUI Ruby Client Library",
                    "gemDescription": "Ruby client library for Claude TUI API",
                    "gemAuthor": "Claude TUI Team",
                    "gemEmail": "support@claude-tui.com",
                    "gemHomepage": "https://github.com/claude-tui/ruby-client"
                },
                post_generate_commands=[
                    "cd {output_dir} && bundle install",
                    "cd {output_dir} && rspec spec/ || true"
                ]
            ),
            
            # PHP Client
            ClientConfig(
                language="php",
                generator="php",
                package_name="claude-tui/client",
                output_dir="php",
                additional_properties={
                    "composerVendorName": "claude-tui",
                    "composerProjectName": "client",
                    "packagePath": "ClaudeTuiClient",
                    "invokerPackage": "ClaudeTuiClient",
                    "srcBasePath": "lib"
                },
                post_generate_commands=[
                    "cd {output_dir} && composer install",
                    "cd {output_dir} && ./vendor/bin/phpunit tests/ || true"
                ]
            )
        ]
    
    async def generate_all_clients(self):
        """Generate all client SDKs."""
        logger.info("Starting API client generation for all languages")
        
        # Ensure output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate OpenAPI spec
        if not await self._validate_openapi_spec():
            logger.error("OpenAPI spec validation failed")
            return
        
        # Generate clients for each language
        results = {}
        for config in self.clients_config:
            try:
                logger.info(f"Generating {config.language} client...")
                success = await self._generate_client(config)
                results[config.language] = "success" if success else "failed"
                
                if success:
                    await self._create_client_documentation(config)
                    await self._create_client_examples(config)
                
            except Exception as e:
                logger.error(f"Failed to generate {config.language} client: {e}")
                results[config.language] = f"error: {str(e)}"
        
        # Generate summary report
        await self._generate_summary_report(results)
        
        logger.info("API client generation completed")
        logger.info(f"Results: {results}")
    
    async def _validate_openapi_spec(self) -> bool:
        """Validate OpenAPI specification."""
        try:
            if not self.openapi_spec_path.exists():
                logger.error(f"OpenAPI spec file not found: {self.openapi_spec_path}")
                return False
            
            # Load and validate YAML
            with open(self.openapi_spec_path, 'r') as f:
                spec = yaml.safe_load(f)
            
            # Basic validation
            required_fields = ['openapi', 'info', 'paths']
            for field in required_fields:
                if field not in spec:
                    logger.error(f"Missing required field in OpenAPI spec: {field}")
                    return False
            
            logger.info("OpenAPI specification validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"OpenAPI validation error: {e}")
            return False
    
    async def _generate_client(self, config: ClientConfig) -> bool:
        """Generate client SDK for specific language."""
        try:
            output_dir = self.output_base_dir / config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build openapi-generator command
            cmd = [
                "npx", "@openapitools/openapi-generator-cli", "generate",
                "-i", str(self.openapi_spec_path),
                "-g", config.generator,
                "-o", str(output_dir),
                "--additional-properties"
            ]
            
            # Add additional properties
            properties = []
            for key, value in config.additional_properties.items():
                properties.append(f"{key}={value}")
            
            cmd.append(",".join(properties))
            
            # Execute generation command
            process = await self._run_command(cmd, cwd=str(self.output_base_dir))
            
            if process.returncode != 0:
                logger.error(f"Client generation failed for {config.language}")
                return False
            
            # Run post-generation commands
            for post_cmd in config.post_generate_commands:
                formatted_cmd = post_cmd.format(output_dir=str(output_dir))
                await self._run_shell_command(formatted_cmd, cwd=str(self.output_base_dir))
            
            logger.info(f"Successfully generated {config.language} client")
            return True
            
        except Exception as e:
            logger.error(f"Error generating {config.language} client: {e}")
            return False
    
    async def _create_client_documentation(self, config: ClientConfig):
        """Create documentation for the generated client."""
        output_dir = self.output_base_dir / config.output_dir
        docs_dir = output_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create README
        readme_content = self._generate_readme_content(config)
        with open(docs_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        # Create quickstart guide
        quickstart_content = self._generate_quickstart_guide(config)
        with open(docs_dir / "QUICKSTART.md", "w") as f:
            f.write(quickstart_content)
    
    async def _create_client_examples(self, config: ClientConfig):
        """Create example code for the generated client."""
        output_dir = self.output_base_dir / config.output_dir
        examples_dir = output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Generate language-specific examples
        examples = self._get_language_examples(config)
        
        for example_name, example_code in examples.items():
            file_extension = self._get_file_extension(config.language)
            example_file = examples_dir / f"{example_name}.{file_extension}"
            
            with open(example_file, "w") as f:
                f.write(example_code)
    
    def _generate_readme_content(self, config: ClientConfig) -> str:
        """Generate README content for client SDK."""
        return f"""# Claude TUI {config.language.title()} Client

Official {config.language.title()} client library for the Claude TUI API.

## Installation

### {self._get_installation_instructions(config)}

## Quick Start

```{config.language}
{self._get_quickstart_example(config)}
```

## Features

- Full API coverage for all Claude TUI endpoints
- Automatic authentication handling
- Built-in retry logic and error handling
- Type-safe request/response models
- Comprehensive documentation and examples

## Documentation

- [API Documentation](https://docs.claude-tui.com)
- [Client Documentation](./QUICKSTART.md)
- [Examples](./examples/)

## Support

- GitHub Issues: https://github.com/claude-tui/{config.language}-client/issues
- Documentation: https://docs.claude-tui.com
- Support Email: support@claude-tui.com

## License

MIT License - see LICENSE file for details.
"""
    
    def _get_installation_instructions(self, config: ClientConfig) -> str:
        """Get installation instructions for the language."""
        instructions = {
            "python": "```bash\npip install claude-tui-client\n```",
            "typescript": "```bash\nnpm install claude-tui-client\n```",
            "go": "```bash\ngo get github.com/claude-tui/go-client\n```",
            "java": "```xml\n<dependency>\n  <groupId>com.claudetui</groupId>\n  <artifactId>claude-tui-java-client</artifactId>\n  <version>1.0.0</version>\n</dependency>\n```",
            "csharp": "```bash\ndotnet add package ClaudeTui.Client\n```",
            "ruby": "```bash\ngem install claude_tui_client\n```",
            "php": "```bash\ncomposer require claude-tui/client\n```"
        }
        
        return instructions.get(config.language, "See documentation for installation instructions")
    
    def _get_quickstart_example(self, config: ClientConfig) -> str:
        """Get quickstart example for the language."""
        examples = {
            "python": '''
import claude_tui_client
from claude_tui_client.rest import ApiException

# Configure API client
configuration = claude_tui_client.Configuration(
    host = "https://api.claude-tui.com"
)

# Configure authentication
configuration.api_key['Authorization'] = 'YOUR_API_KEY'

# Create API client
with claude_tui_client.ApiClient(configuration) as api_client:
    # Create projects API instance
    projects_api = claude_tui_client.ProjectsApi(api_client)
    
    try:
        # List projects
        projects = projects_api.list_projects()
        print(f"Found {len(projects.projects)} projects")
        
        # Create new project
        new_project = projects_api.create_project(
            create_project_request={
                "name": "My Project",
                "description": "Created with Python client",
                "technology_stack": ["Python", "FastAPI"]
            }
        )
        print(f"Created project: {new_project.id}")
        
    except ApiException as e:
        print(f"API Error: {e}")
            ''',
            
            "typescript": '''
import { Configuration, ProjectsApi, CreateProjectRequest } from 'claude-tui-client';

// Configure API client
const configuration = new Configuration({
    basePath: 'https://api.claude-tui.com',
    accessToken: 'YOUR_API_KEY'
});

const projectsApi = new ProjectsApi(configuration);

async function example() {
    try {
        // List projects
        const projects = await projectsApi.listProjects();
        console.log(`Found ${projects.data.projects.length} projects`);
        
        // Create new project
        const createRequest: CreateProjectRequest = {
            name: "My Project",
            description: "Created with TypeScript client",
            technologyStack: ["TypeScript", "React"]
        };
        
        const newProject = await projectsApi.createProject(createRequest);
        console.log(`Created project: ${newProject.data.id}`);
        
    } catch (error) {
        console.error('API Error:', error);
    }
}

example();
            ''',
            
            "go": '''
package main

import (
    "context"
    "fmt"
    "log"
    
    claudetui "github.com/claude-tui/go-client"
)

func main() {
    // Configure API client
    cfg := claudetui.NewConfiguration()
    cfg.Host = "api.claude-tui.com"
    cfg.AddDefaultHeader("Authorization", "Bearer YOUR_API_KEY")
    
    client := claudetui.NewAPIClient(cfg)
    ctx := context.Background()
    
    // List projects
    projects, _, err := client.ProjectsApi.ListProjects(ctx).Execute()
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Found %d projects\\n", len(projects.Projects))
    
    // Create new project
    createReq := claudetui.CreateProjectRequest{
        Name: "My Project",
        Description: claudetui.PtrString("Created with Go client"),
        TechnologyStack: []string{"Go", "Gin"},
    }
    
    newProject, _, err := client.ProjectsApi.CreateProject(ctx).CreateProjectRequest(createReq).Execute()
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Created project: %s\\n", newProject.Id)
}
            '''
        }
        
        return examples.get(config.language, "// See documentation for usage examples")
    
    def _generate_quickstart_guide(self, config: ClientConfig) -> str:
        """Generate quickstart guide for the client."""
        return f"""# {config.language.title()} Client Quick Start Guide

## Installation

{self._get_installation_instructions(config)}

## Authentication

The Claude TUI API uses Bearer token authentication. You can authenticate using either:

1. **JWT Tokens** - Obtained via username/password authentication
2. **API Keys** - Generated in the Claude TUI dashboard

### JWT Authentication

```{config.language}
{self._get_jwt_auth_example(config)}
```

### API Key Authentication

```{config.language}
{self._get_api_key_auth_example(config)}
```

## Core Operations

### Project Management

```{config.language}
{self._get_project_management_example(config)}
```

### AI Task Execution

```{config.language}
{self._get_ai_task_example(config)}
```

### Real-time Updates

```{config.language}
{self._get_websocket_example(config)}
```

## Error Handling

```{config.language}
{self._get_error_handling_example(config)}
```

## Best Practices

1. **Rate Limiting**: The API has rate limits. Implement exponential backoff for retries.
2. **Authentication**: Store API keys securely and rotate them regularly.
3. **Error Handling**: Always handle API errors gracefully.
4. **Pagination**: Use pagination for list endpoints to avoid timeouts.
5. **WebSocket**: Use WebSocket connections for real-time updates.

## Advanced Features

- Batch operations for bulk data processing
- WebSocket subscriptions for real-time updates
- Streaming responses for large datasets
- Request/response interceptors for custom logic

For more examples and advanced usage, see the [examples](./examples/) directory.
"""
    
    def _get_language_examples(self, config: ClientConfig) -> Dict[str, str]:
        """Get example files for the language."""
        # This would return language-specific example code files
        examples = {}
        
        if config.language == "python":
            examples.update({
                "authentication": '''
# Authentication example
import claude_tui_client
from claude_tui_client.rest import ApiException

# JWT Authentication
def authenticate_with_jwt(username, password):
    configuration = claude_tui_client.Configuration(
        host="https://api.claude-tui.com"
    )
    
    with claude_tui_client.ApiClient(configuration) as api_client:
        auth_api = claude_tui_client.AuthenticationApi(api_client)
        
        try:
            token_response = auth_api.get_access_token(
                username=username,
                password=password
            )
            
            # Set token for future requests
            configuration.api_key['Authorization'] = f"Bearer {token_response.access_token}"
            
            return token_response.access_token
            
        except ApiException as e:
            print(f"Authentication failed: {e}")
            return None

# API Key Authentication
def authenticate_with_api_key(api_key):
    configuration = claude_tui_client.Configuration(
        host="https://api.claude-tui.com"
    )
    configuration.api_key['Authorization'] = f"Bearer {api_key}"
    
    return configuration
                ''',
                
                "project_operations": '''
# Project operations example
import claude_tui_client
from claude_tui_client.rest import ApiException

def project_operations_example():
    # Assume authentication is already configured
    configuration = claude_tui_client.Configuration()
    
    with claude_tui_client.ApiClient(configuration) as api_client:
        projects_api = claude_tui_client.ProjectsApi(api_client)
        
        try:
            # Create a new project
            create_request = claude_tui_client.CreateProjectRequest(
                name="AI Web App",
                description="An AI-powered web application",
                technology_stack=["Python", "FastAPI", "React", "PostgreSQL"],
                template_id="fullstack-template"
            )
            
            project = projects_api.create_project(create_request)
            print(f"Created project: {project.id}")
            
            # Get project details
            project_details = projects_api.get_project(project.id)
            print(f"Project has {len(project_details.files)} files")
            
            # Update project
            update_request = claude_tui_client.UpdateProjectRequest(
                description="Updated description",
                status="active"
            )
            
            updated_project = projects_api.update_project(project.id, update_request)
            print(f"Updated project status: {updated_project.status}")
            
            # List all projects
            projects_response = projects_api.list_projects(limit=10)
            print(f"Total projects: {len(projects_response.projects)}")
            
        except ApiException as e:
            print(f"Project operation failed: {e}")
                ''',
                
                "ai_tasks": '''
# AI task execution example
import claude_tui_client
from claude_tui_client.rest import ApiException
import time

def ai_task_example():
    configuration = claude_tui_client.Configuration()
    
    with claude_tui_client.ApiClient(configuration) as api_client:
        ai_api = claude_tui_client.AIAdvancedApi(api_client)
        
        try:
            # Initialize AI swarm
            swarm_request = claude_tui_client.SwarmInitRequest(
                project_context={
                    "language": "Python",
                    "framework": "FastAPI",
                    "complexity": "medium"
                },
                preferred_topology="mesh",
                max_agents=5,
                enable_auto_scaling=True
            )
            
            swarm_response = ai_api.initialize_swarm(swarm_request)
            print(f"Initialized swarm: {swarm_response.swarm_id}")
            
            # Execute AI task
            task_request = claude_tui_client.TaskExecutionRequest(
                description="Implement user authentication with JWT tokens",
                context_type="development",
                priority="high",
                agent_requirements=["python", "security", "database"],
                estimated_duration=1800  # 30 minutes
            )
            
            task_response = ai_api.execute_task(task_request)
            print(f"Started task: {task_response.task_id}")
            
            # Monitor task progress
            task_id = task_response.task_id
            while True:
                status = ai_api.get_task_status(task_id)
                print(f"Task status: {status.status}")
                
                if status.status in ['completed', 'failed', 'cancelled']:
                    break
                
                time.sleep(10)  # Check every 10 seconds
            
            print(f"Task completed with status: {status.status}")
            
        except ApiException as e:
            print(f"AI task failed: {e}")
                '''
            })
        
        return examples
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": "py",
            "typescript": "ts",
            "go": "go",
            "java": "java",
            "csharp": "cs",
            "ruby": "rb",
            "php": "php"
        }
        return extensions.get(language, "txt")
    
    def _get_jwt_auth_example(self, config: ClientConfig) -> str:
        """Get JWT authentication example."""
        # Return language-specific JWT auth example
        return "// JWT authentication example"
    
    def _get_api_key_auth_example(self, config: ClientConfig) -> str:
        """Get API key authentication example."""
        # Return language-specific API key auth example
        return "// API key authentication example"
    
    def _get_project_management_example(self, config: ClientConfig) -> str:
        """Get project management example."""
        return "// Project management example"
    
    def _get_ai_task_example(self, config: ClientConfig) -> str:
        """Get AI task example."""
        return "// AI task execution example"
    
    def _get_websocket_example(self, config: ClientConfig) -> str:
        """Get WebSocket example."""
        return "// WebSocket connection example"
    
    def _get_error_handling_example(self, config: ClientConfig) -> str:
        """Get error handling example."""
        return "// Error handling example"
    
    async def _run_command(self, cmd: List[str], cwd: str = None) -> subprocess.CompletedProcess:
        """Run command asynchronously."""
        process = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        
        if process.stdout:
            logger.debug(f"Command output: {process.stdout}")
        if process.stderr:
            logger.error(f"Command error: {process.stderr}")
        
        return process
    
    async def _run_shell_command(self, cmd: str, cwd: str = None) -> subprocess.CompletedProcess:
        """Run shell command asynchronously."""
        process = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        
        if process.stdout:
            logger.debug(f"Shell output: {process.stdout}")
        if process.stderr and process.returncode != 0:
            logger.error(f"Shell error: {process.stderr}")
        
        return process
    
    async def _generate_summary_report(self, results: Dict[str, str]):
        """Generate summary report of client generation."""
        report_path = self.output_base_dir / "generation_report.json"
        
        report = {
            "generated_at": "2023-01-01T00:00:00Z",
            "openapi_spec": str(self.openapi_spec_path),
            "output_directory": str(self.output_base_dir),
            "results": results,
            "successful_clients": [lang for lang, status in results.items() if status == "success"],
            "failed_clients": [lang for lang, status in results.items() if status != "success"]
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generation report saved to: {report_path}")


async def main():
    """Main function to generate all API clients."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Paths
    openapi_spec_path = project_root / "docs" / "api" / "openapi.yaml"
    output_dir = project_root / "clients"
    
    # Generate clients
    generator = APIClientGenerator(str(openapi_spec_path), str(output_dir))
    await generator.generate_all_clients()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())