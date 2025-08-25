# Claude TIU - API Specification

## Table of Contents

1. [API Overview](#api-overview)
2. [Core API Classes](#core-api-classes)
3. [Project Management API](#project-management-api)
4. [AI Integration API](#ai-integration-api)
5. [Task Engine API](#task-engine-api)
6. [Validation Engine API](#validation-engine-api)
7. [CLI API](#cli-api)
8. [Configuration API](#configuration-api)
9. [Error Handling](#error-handling)
10. [Examples](#examples)

---

## API Overview

Claude TIU provides a comprehensive Python API for building AI-powered development tools. The API is designed with async/await patterns and follows modern Python conventions.

### Core Principles

- **Async-First**: All operations are asynchronous by default
- **Type-Safe**: Full type hint support with Pydantic models
- **Modular**: Clear separation of concerns
- **Extensible**: Plugin-based architecture
- **Validated**: Built-in validation and error handling

### Installation

```python
# For API usage
pip install claude-tiu

# For development
pip install claude-tiu[dev]
```

### Basic Usage

```python
import asyncio
from claude_tiu import ClaudeTIU, ProjectConfig

async def main():
    # Initialize Claude TIU
    tiu = ClaudeTIU()
    
    # Create a new project
    config = ProjectConfig(
        name="my-project",
        template="react-typescript",
        features=["authentication", "testing"]
    )
    
    project = await tiu.create_project(config)
    print(f"Created project: {project.path}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Core API Classes

### ClaudeTIU

The main entry point for all Claude TIU operations.

```python
class ClaudeTIU:
    """Main Claude TIU interface for AI-powered development"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize Claude TIU instance
        
        Args:
            config_path: Path to configuration file
            claude_api_key: Claude API key (or set CLAUDE_API_KEY env var)
            debug: Enable debug mode
        """
    
    async def create_project(
        self, 
        config: ProjectConfig
    ) -> Project:
        """
        Create a new project with AI assistance
        
        Args:
            config: Project configuration
            
        Returns:
            Project: Created project instance
            
        Raises:
            ProjectCreationError: If project creation fails
            ValidationError: If configuration is invalid
        """
    
    async def load_project(
        self, 
        path: str
    ) -> Project:
        """
        Load existing project
        
        Args:
            path: Path to project directory
            
        Returns:
            Project: Loaded project instance
            
        Raises:
            ProjectNotFoundError: If project doesn't exist
            ProjectLoadError: If project loading fails
        """
    
    async def execute_task(
        self,
        task: DevelopmentTask
    ) -> TaskResult:
        """
        Execute a development task with AI assistance
        
        Args:
            task: Development task to execute
            
        Returns:
            TaskResult: Task execution result
            
        Raises:
            TaskExecutionError: If task execution fails
            ValidationError: If task validation fails
        """
```

### Project

Represents a software project managed by Claude TIU.

```python
class Project:
    """Software project with AI-powered development capabilities"""
    
    @property
    def path(self) -> Path:
        """Project directory path"""
    
    @property
    def config(self) -> ProjectConfig:
        """Project configuration"""
    
    @property
    def structure(self) -> ProjectStructure:
        """Project file structure"""
    
    async def add_feature(
        self, 
        feature_spec: FeatureSpec
    ) -> FeatureResult:
        """
        Add a new feature to the project
        
        Args:
            feature_spec: Feature specification
            
        Returns:
            FeatureResult: Feature implementation result
        """
    
    async def refactor(
        self, 
        refactor_spec: RefactorSpec
    ) -> RefactorResult:
        """
        Refactor existing code
        
        Args:
            refactor_spec: Refactoring specification
            
        Returns:
            RefactorResult: Refactoring result
        """
    
    async def validate_codebase(self) -> ValidationReport:
        """
        Validate entire codebase for quality and completeness
        
        Returns:
            ValidationReport: Comprehensive validation report
        """
    
    async def generate_tests(
        self, 
        test_spec: TestSpec
    ) -> TestResult:
        """
        Generate tests for specified components
        
        Args:
            test_spec: Test generation specification
            
        Returns:
            TestResult: Generated tests and metrics
        """
```

### ProjectConfig

Configuration model for project creation and management.

```python
class ProjectConfig(BaseModel):
    """Project configuration model"""
    
    name: str = Field(..., min_length=1, description="Project name")
    template: str = Field(..., description="Project template")
    description: Optional[str] = Field(None, description="Project description")
    
    # Technology stack
    language: str = Field("python", description="Primary programming language")
    framework: Optional[str] = Field(None, description="Framework to use")
    database: Optional[str] = Field(None, description="Database technology")
    
    # Features to include
    features: List[str] = Field(default_factory=list, description="Features to include")
    integrations: List[str] = Field(default_factory=list, description="Third-party integrations")
    
    # Development preferences
    testing_framework: Optional[str] = Field(None, description="Testing framework")
    code_style: str = Field("pep8", description="Code style standard")
    ci_cd: Optional[str] = Field(None, description="CI/CD platform")
    
    # AI behavior configuration
    ai_creativity: float = Field(0.7, ge=0.0, le=1.0, description="AI creativity level")
    validation_level: str = Field("strict", description="Validation strictness")
    auto_fix: bool = Field(True, description="Automatically fix detected issues")
    
    class Config:
        extra = "forbid"
        validate_assignment = True
```

---

## Project Management API

### ProjectManager

Core project management functionality.

```python
class ProjectManager:
    """Advanced project management with AI assistance"""
    
    def __init__(self, ai_interface: AIInterface):
        self.ai_interface = ai_interface
        self.state_manager = StateManager()
        self.template_manager = TemplateManager()
    
    async def create_from_template(
        self,
        template_name: str,
        project_config: ProjectConfig,
        target_path: Path
    ) -> Project:
        """
        Create project from template
        
        Args:
            template_name: Name of template to use
            project_config: Project configuration
            target_path: Where to create the project
            
        Returns:
            Project: Created project instance
        """
    
    async def analyze_project_health(
        self,
        project: Project
    ) -> ProjectHealthReport:
        """
        Analyze project health and provide recommendations
        
        Args:
            project: Project to analyze
            
        Returns:
            ProjectHealthReport: Comprehensive health analysis
        """
    
    async def suggest_improvements(
        self,
        project: Project
    ) -> List[Improvement]:
        """
        AI-powered improvement suggestions
        
        Args:
            project: Project to analyze
            
        Returns:
            List[Improvement]: Suggested improvements
        """
```

### TemplateManager

Manage project templates and scaffolding.

```python
class TemplateManager:
    """Project template management"""
    
    async def list_templates(self) -> List[Template]:
        """List available project templates"""
    
    async def get_template(self, name: str) -> Template:
        """Get specific template by name"""
    
    async def create_custom_template(
        self,
        name: str,
        source_project: Path,
        template_config: TemplateConfig
    ) -> Template:
        """Create custom template from existing project"""
    
    async def validate_template(
        self, 
        template: Template
    ) -> TemplateValidation:
        """Validate template structure and configuration"""
```

---

## AI Integration API

### AIInterface

Unified interface for all AI services.

```python
class AIInterface:
    """Unified AI service interface"""
    
    def __init__(
        self,
        claude_client: ClaudeCodeClient,
        flow_client: ClaudeFlowClient
    ):
        self.claude_client = claude_client
        self.flow_client = flow_client
        self.decision_engine = IntegrationDecisionEngine()
    
    async def execute_coding_task(
        self,
        task: CodingTask,
        context: Optional[Dict] = None
    ) -> CodeResult:
        """
        Execute coding task with optimal AI service
        
        Args:
            task: Coding task to execute
            context: Additional context for AI
            
        Returns:
            CodeResult: Code generation result
        """
    
    async def run_workflow(
        self,
        workflow: Workflow,
        project: Project
    ) -> WorkflowResult:
        """
        Execute complex workflow with Claude Flow
        
        Args:
            workflow: Workflow definition
            project: Target project
            
        Returns:
            WorkflowResult: Workflow execution result
        """
    
    async def validate_ai_output(
        self,
        output: str,
        validation_context: ValidationContext
    ) -> ValidationResult:
        """
        Validate AI-generated output
        
        Args:
            output: AI-generated content
            validation_context: Validation parameters
            
        Returns:
            ValidationResult: Validation results and recommendations
        """
```

### ClaudeCodeClient

Direct integration with Claude Code CLI.

```python
class ClaudeCodeClient:
    """Claude Code CLI integration"""
    
    async def generate_code(
        self,
        prompt: str,
        context: CodeContext,
        language: str = "python"
    ) -> CodeGenerationResult:
        """
        Generate code using Claude Code
        
        Args:
            prompt: Code generation prompt
            context: Code context and requirements
            language: Target programming language
            
        Returns:
            CodeGenerationResult: Generated code and metadata
        """
    
    async def review_code(
        self,
        code: str,
        review_criteria: ReviewCriteria
    ) -> CodeReview:
        """
        Review existing code for quality and issues
        
        Args:
            code: Code to review
            review_criteria: Review parameters
            
        Returns:
            CodeReview: Review results and suggestions
        """
    
    async def refactor_code(
        self,
        code: str,
        refactor_instructions: str,
        context: CodeContext
    ) -> RefactorResult:
        """
        Refactor code according to instructions
        
        Args:
            code: Code to refactor
            refactor_instructions: Refactoring requirements
            context: Code context
            
        Returns:
            RefactorResult: Refactored code and changes
        """
```

### ClaudeFlowClient

Claude Flow workflow orchestration.

```python
class ClaudeFlowClient:
    """Claude Flow orchestration client"""
    
    async def initialize_swarm(
        self,
        topology: SwarmTopology,
        max_agents: int = 5
    ) -> SwarmId:
        """
        Initialize AI agent swarm
        
        Args:
            topology: Swarm topology configuration
            max_agents: Maximum number of agents
            
        Returns:
            SwarmId: Swarm identifier
        """
    
    async def spawn_agent(
        self,
        agent_type: str,
        capabilities: List[str],
        swarm_id: SwarmId
    ) -> AgentId:
        """
        Spawn specialized AI agent
        
        Args:
            agent_type: Type of agent to spawn
            capabilities: Agent capabilities
            swarm_id: Target swarm
            
        Returns:
            AgentId: Agent identifier
        """
    
    async def orchestrate_task(
        self,
        task: ComplexTask,
        swarm_id: SwarmId,
        strategy: OrchestrationStrategy = "adaptive"
    ) -> OrchestrationResult:
        """
        Orchestrate complex task across agent swarm
        
        Args:
            task: Complex task to execute
            swarm_id: Target swarm
            strategy: Orchestration strategy
            
        Returns:
            OrchestrationResult: Task execution result
        """
```

---

## Task Engine API

### TaskEngine

Advanced task scheduling and execution.

```python
class TaskEngine:
    """Task orchestration and execution engine"""
    
    def __init__(self):
        self.scheduler = TaskScheduler()
        self.executor = AsyncTaskExecutor()
        self.monitor = ProgressMonitor()
        self.dependency_resolver = DependencyResolver()
    
    async def create_workflow(
        self,
        workflow_spec: WorkflowSpec
    ) -> Workflow:
        """
        Create workflow from specification
        
        Args:
            workflow_spec: Workflow specification
            
        Returns:
            Workflow: Created workflow instance
        """
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        execution_context: ExecutionContext
    ) -> WorkflowResult:
        """
        Execute workflow with dependency resolution
        
        Args:
            workflow: Workflow to execute
            execution_context: Execution parameters
            
        Returns:
            WorkflowResult: Workflow execution result
        """
    
    async def monitor_progress(
        self,
        workflow_id: str
    ) -> ProgressReport:
        """
        Get real-time progress for workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            ProgressReport: Current progress information
        """
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow execution"""
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume paused workflow"""
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution"""
```

### DevelopmentTask

Base class for all development tasks.

```python
class DevelopmentTask(BaseModel):
    """Base development task model"""
    
    id: str = Field(..., description="Task identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    type: TaskType = Field(..., description="Task type")
    priority: Priority = Field(Priority.MEDIUM, description="Task priority")
    
    # Dependencies and requirements
    dependencies: List[str] = Field(default_factory=list)
    requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution context
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    retry_count: int = Field(3, description="Maximum retry attempts")
    
    # Validation settings
    validation_required: bool = Field(True)
    auto_fix: bool = Field(True)
    
    class Config:
        use_enum_values = True
```

### CodingTask

Specialized task for code generation.

```python
class CodingTask(DevelopmentTask):
    """Code generation task"""
    
    type: Literal[TaskType.CODING] = TaskType.CODING
    
    # Code generation parameters
    language: str = Field(..., description="Programming language")
    framework: Optional[str] = Field(None, description="Framework to use")
    style_guide: str = Field("standard", description="Code style guide")
    
    # Context and requirements
    existing_code: Optional[str] = Field(None, description="Existing code context")
    tests_required: bool = Field(True, description="Generate tests")
    documentation_required: bool = Field(True, description="Generate documentation")
    
    # Quality requirements
    min_coverage: float = Field(0.8, description="Minimum test coverage")
    max_complexity: int = Field(10, description="Maximum cyclomatic complexity")
```

---

## Validation Engine API

### ProgressValidator

Anti-hallucination validation system.

```python
class ProgressValidator:
    """Advanced code validation and anti-hallucination system"""
    
    def __init__(self):
        self.placeholder_detector = PlaceholderDetector()
        self.semantic_analyzer = SemanticAnalyzer()
        self.execution_tester = ExecutionTester()
        self.cross_validator = CrossValidator()
    
    async def validate_codebase(
        self,
        project_path: Path,
        validation_config: ValidationConfig
    ) -> ValidationReport:
        """
        Comprehensive codebase validation
        
        Args:
            project_path: Path to project
            validation_config: Validation parameters
            
        Returns:
            ValidationReport: Detailed validation results
        """
    
    async def validate_generated_code(
        self,
        code: str,
        context: ValidationContext
    ) -> CodeValidationResult:
        """
        Validate freshly generated code
        
        Args:
            code: Generated code to validate
            context: Validation context
            
        Returns:
            CodeValidationResult: Validation results
        """
    
    async def auto_fix_issues(
        self,
        issues: List[ValidationIssue],
        fix_context: FixContext
    ) -> AutoFixResult:
        """
        Automatically fix detected validation issues
        
        Args:
            issues: Issues to fix
            fix_context: Fix parameters
            
        Returns:
            AutoFixResult: Fix results and updated code
        """
```

### PlaceholderDetector

Detect incomplete code implementations.

```python
class PlaceholderDetector:
    """Detect placeholders and incomplete implementations"""
    
    def __init__(self):
        self.patterns = self._load_detection_patterns()
        self.ml_model = self._load_ml_model()
    
    async def detect_placeholders(
        self,
        code: str,
        language: str
    ) -> PlaceholderReport:
        """
        Detect placeholder code patterns
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            PlaceholderReport: Detected placeholders and recommendations
        """
    
    async def analyze_completeness(
        self,
        code: str,
        requirements: List[str]
    ) -> CompletenessAnalysis:
        """
        Analyze code completeness against requirements
        
        Args:
            code: Code to analyze
            requirements: Expected functionality
            
        Returns:
            CompletenessAnalysis: Completeness assessment
        """
```

---

## CLI API

### ClaudeTIUCLI

Command-line interface implementation.

```python
@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, debug):
    """Claude TIU - AI-Powered Development Tool"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['debug'] = debug

@cli.command()
@click.option('--template', '-t', required=True, help='Project template')
@click.option('--name', '-n', required=True, help='Project name')
@click.option('--path', '-p', help='Target path')
@click.option('--features', '-f', multiple=True, help='Features to include')
@click.pass_context
async def create(ctx, template, name, path, features):
    """Create a new project"""
    # Implementation details...

@cli.command()
@click.argument('task_description')
@click.option('--workflow', '-w', help='Workflow to use')
@click.option('--agents', '-a', type=int, help='Number of agents')
@click.pass_context
async def execute(ctx, task_description, workflow, agents):
    """Execute a development task"""
    # Implementation details...

@cli.command()
@click.argument('project_path')
@click.option('--level', '-l', default='basic', help='Validation level')
@click.pass_context
async def validate(ctx, project_path, level):
    """Validate project code"""
    # Implementation details...
```

---

## Configuration API

### ConfigurationManager

Manage application configuration.

```python
class ConfigurationManager:
    """Application configuration management"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    async def save(self) -> None:
        """Save configuration to disk"""
        await self._write_config(self.config)
    
    async def reload(self) -> None:
        """Reload configuration from disk"""
        self.config = await self._load_config()
    
    def validate_config(self) -> List[ConfigValidationError]:
        """Validate current configuration"""
        # Validation implementation...
```

### Settings

Pydantic settings model for type-safe configuration.

```python
class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    claude_api_key: str = Field(..., env='CLAUDE_API_KEY')
    claude_flow_endpoint: str = Field('http://localhost:3000', env='CLAUDE_FLOW_ENDPOINT')
    
    # Application Settings
    debug: bool = Field(False, env='DEBUG')
    log_level: str = Field('INFO', env='LOG_LEVEL')
    max_concurrent_tasks: int = Field(5, env='MAX_CONCURRENT_TASKS')
    
    # AI Behavior
    default_creativity: float = Field(0.7, env='DEFAULT_CREATIVITY')
    validation_strictness: str = Field('strict', env='VALIDATION_STRICTNESS')
    auto_fix_enabled: bool = Field(True, env='AUTO_FIX_ENABLED')
    
    # Performance
    cache_enabled: bool = Field(True, env='CACHE_ENABLED')
    cache_ttl: int = Field(3600, env='CACHE_TTL')
    memory_limit_mb: int = Field(1024, env='MEMORY_LIMIT_MB')
    
    # Security
    sandbox_enabled: bool = Field(True, env='SANDBOX_ENABLED')
    code_execution_timeout: int = Field(30, env='CODE_EXECUTION_TIMEOUT')
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
```

---

## Error Handling

### Exception Hierarchy

```python
class ClaudeTIUError(Exception):
    """Base exception for Claude TIU"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ProjectError(ClaudeTIUError):
    """Project-related errors"""
    pass

class ProjectCreationError(ProjectError):
    """Project creation failed"""
    pass

class ProjectNotFoundError(ProjectError):
    """Project not found"""
    pass

class ValidationError(ClaudeTIUError):
    """Validation errors"""
    
    def __init__(self, message: str, issues: List[ValidationIssue]):
        super().__init__(message)
        self.issues = issues

class AIIntegrationError(ClaudeTIUError):
    """AI service integration errors"""
    pass

class TaskExecutionError(ClaudeTIUError):
    """Task execution errors"""
    
    def __init__(self, message: str, task_id: str, stage: str):
        super().__init__(message)
        self.task_id = task_id
        self.stage = stage

class AuthenticationError(ClaudeTIUError):
    """Authentication and authorization errors"""
    pass
```

### Error Context

```python
@dataclass
class ErrorContext:
    """Context information for errors"""
    
    timestamp: datetime
    operation: str
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    task_id: Optional[str] = None
    stack_trace: Optional[str] = None
    system_info: Optional[Dict] = None
```

---

## Examples

### Basic Project Creation

```python
import asyncio
from claude_tiu import ClaudeTIU, ProjectConfig

async def create_react_app():
    # Initialize Claude TIU
    tiu = ClaudeTIU(claude_api_key="your_api_key")
    
    # Configure project
    config = ProjectConfig(
        name="my-react-app",
        template="react-typescript",
        framework="react",
        features=["authentication", "testing", "styling"],
        testing_framework="jest",
        ai_creativity=0.8,
        auto_fix=True
    )
    
    # Create project
    try:
        project = await tiu.create_project(config)
        print(f"✅ Project created at: {project.path}")
        
        # Validate the created project
        validation_report = await project.validate_codebase()
        if validation_report.is_valid:
            print("✅ Project validation passed")
        else:
            print(f"❌ Validation issues: {len(validation_report.issues)}")
            
    except Exception as e:
        print(f"❌ Project creation failed: {e}")

# Run the example
asyncio.run(create_react_app())
```

### Advanced Feature Development

```python
from claude_tiu import ClaudeTIU, FeatureSpec, CodingTask

async def add_user_authentication():
    tiu = ClaudeTIU()
    project = await tiu.load_project("./my-react-app")
    
    # Define feature specification
    feature_spec = FeatureSpec(
        name="user_authentication",
        description="Complete user authentication system with JWT tokens",
        requirements=[
            "User registration with email/password",
            "Login/logout functionality", 
            "JWT token management",
            "Password reset via email",
            "Protected routes",
            "User profile management"
        ],
        acceptance_criteria=[
            "Users can register with valid email",
            "Users can login with correct credentials",
            "Protected routes require authentication",
            "Tokens expire after 24 hours",
            "Password reset emails are sent"
        ]
    )
    
    # Execute feature development
    try:
        result = await project.add_feature(feature_spec)
        
        print(f"✅ Feature implementation completed")
        print(f"📁 Files created: {len(result.created_files)}")
        print(f"📝 Files modified: {len(result.modified_files)}")
        print(f"🧪 Tests generated: {len(result.test_files)}")
        
        # Validate the implementation
        validation = await project.validate_codebase()
        if validation.authenticity_score > 0.9:
            print(f"✅ High authenticity score: {validation.authenticity_score:.2f}")
        else:
            print(f"⚠️ Authenticity concerns: {validation.authenticity_score:.2f}")
            
    except ValidationError as e:
        print(f"❌ Validation failed: {len(e.issues)} issues found")
        for issue in e.issues:
            print(f"  - {issue.description} ({issue.severity})")

asyncio.run(add_user_authentication())
```

### Custom Workflow Orchestration

```python
from claude_tiu import TaskEngine, WorkflowSpec, CodingTask

async def custom_development_workflow():
    engine = TaskEngine()
    
    # Define custom workflow
    workflow_spec = WorkflowSpec(
        name="full_stack_development",
        description="Complete full-stack feature development",
        tasks=[
            CodingTask(
                id="backend_api",
                name="Backend API Development",
                description="Create REST API endpoints",
                language="python",
                framework="fastapi",
                requirements={
                    "endpoints": ["GET /users", "POST /users", "PUT /users/{id}"],
                    "validation": "pydantic",
                    "database": "postgresql"
                }
            ),
            CodingTask(
                id="frontend_components",
                name="Frontend Components",
                description="Create React components",
                language="typescript",
                framework="react",
                dependencies=["backend_api"],
                requirements={
                    "components": ["UserList", "UserForm", "UserProfile"],
                    "styling": "tailwind",
                    "state_management": "zustand"
                }
            ),
            CodingTask(
                id="integration_tests",
                name="Integration Tests",
                description="End-to-end testing",
                language="typescript",
                dependencies=["backend_api", "frontend_components"],
                requirements={
                    "framework": "cypress",
                    "coverage": ">90%"
                }
            )
        ],
        execution_strategy="parallel_where_possible"
    )
    
    # Create and execute workflow
    workflow = await engine.create_workflow(workflow_spec)
    result = await engine.execute_workflow(workflow)
    
    print(f"✅ Workflow completed in {result.execution_time:.2f}s")
    print(f"📊 Success rate: {result.success_rate:.1%}")
    
    # Monitor progress in real-time
    async for progress in engine.monitor_progress(workflow.id):
        print(f"🔄 Progress: {progress.percentage:.1f}% - {progress.current_task}")
        if progress.is_complete:
            break

asyncio.run(custom_development_workflow())
```

### Anti-Hallucination Validation

```python
from claude_tiu import ProgressValidator, ValidationConfig

async def validate_ai_generated_code():
    validator = ProgressValidator()
    
    # Configure validation
    config = ValidationConfig(
        placeholder_detection=True,
        semantic_analysis=True,
        execution_testing=True,
        cross_validation=True,
        auto_fix=True,
        strictness="high"
    )
    
    # Validate project
    project_path = Path("./my-project")
    report = await validator.validate_codebase(project_path, config)
    
    print(f"📊 Validation Summary:")
    print(f"  Overall Score: {report.overall_score:.2f}/1.0")
    print(f"  Authenticity: {report.authenticity_score:.2f}/1.0")
    print(f"  Completeness: {report.completeness_score:.2f}/1.0")
    print(f"  Quality: {report.quality_score:.2f}/1.0")
    
    if report.issues:
        print(f"\n❌ Issues Found ({len(report.issues)}):")
        for issue in report.issues:
            print(f"  - {issue.file}:{issue.line} - {issue.description}")
            
        # Auto-fix issues
        if config.auto_fix:
            fix_result = await validator.auto_fix_issues(
                report.issues,
                fix_context={"project_path": project_path}
            )
            print(f"✅ Auto-fixed {fix_result.fixed_count} issues")
    else:
        print("✅ No validation issues found!")

asyncio.run(validate_ai_generated_code())
```

This comprehensive API specification provides developers with everything needed to build on top of Claude TIU's AI-powered development capabilities.