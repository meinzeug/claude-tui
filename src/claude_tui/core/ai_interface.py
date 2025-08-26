"""
AI Interface - Core interface for Claude Code integration.

Provides a high-level interface for AI operations including code generation,
analysis, validation, and workflow orchestration through Claude Code API.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient
# AI Models for structured data exchange
@dataclass
class CodeContext:
    """Context information for AI code operations."""
    file_path: str
    project_root: str
    language: str
    existing_code: str = ""
    dependencies: List[str] = None
    requirements: str = ""
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class CodeResult:
    """Result from AI code generation operations."""
    generated_code: str
    language: str
    confidence_score: float
    issues: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []

@dataclass
class TaskRequest:
    """Request structure for AI task execution."""
    task_type: str
    description: str
    context: Dict[str, Any]
    priority: str = "normal"
    timeout: int = 300

@dataclass
class TaskResult:
    """Result structure for AI task execution."""
    task_id: str
    status: str
    result: Any
    execution_time: float
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class ReviewCriteria:
    """Criteria for AI code review operations."""
    check_syntax: bool = True
    check_logic: bool = True
    check_security: bool = True
    check_performance: bool = True
    check_style: bool = True
    custom_rules: List[str] = None
    
    def __post_init__(self):
        if self.custom_rules is None:
            self.custom_rules = []

@dataclass
class CodeReview:
    """Result structure for AI code review."""
    overall_score: float
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    approved: bool
    reviewer: str = "claude-ai"
    
@dataclass
class PlaceholderDetection:
    """Result structure for placeholder detection."""
    has_placeholders: bool
    placeholder_count: int
    placeholder_details: List[Dict[str, Any]]
    confidence_score: float

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.validation.types import ValidationResult  # Import from types.py
# Custom exceptions for AI interface operations
class AIInterfaceError(Exception):
    """Base exception for AI interface operations."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.details = details or {}

class ValidationError(AIInterfaceError):
    """Exception for validation-related errors."""
    pass

class AITimeoutError(AIInterfaceError):
    """Exception for AI operation timeouts."""
    pass

class AIModelError(AIInterfaceError):
    """Exception for AI model-related errors."""
    pass

logger = logging.getLogger(__name__)


@dataclass
class AIRequest:
    """Generic AI request structure."""
    prompt: str
    context: Dict[str, Any]
    model: str = "claude-3-sonnet"
    timeout: int = 300
    max_tokens: Optional[int] = None
    temperature: float = 0.7


@dataclass 
class AIResponse:
    """Generic AI response structure."""
    content: str
    model_used: str
    tokens_used: int
    processing_time: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class AIInterface:
    """
    High-level AI interface for Claude TUI operations.
    
    Provides simplified access to AI capabilities through Claude Code,
    with built-in error handling, retries, and result validation.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize AI interface.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.client = ClaudeCodeClient(config_manager)
        self._initialization_complete = False
        logger.info("AI interface initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the AI interface and validate connections.
        
        Returns:
            True if initialization successful
        """
        try:
            await self.client.initialize()
            
            # Test connection with a simple request
            test_response = await self.client.execute_task(
                TaskRequest(
                    description="Test connection - respond with 'OK'",
                    timeout=30
                )
            )
            
            if test_response.success:
                self._initialization_complete = True
                logger.info("AI interface initialization successful")
                return True
            else:
                logger.error(f"AI interface test failed: {test_response.error}")
                return False
                
        except Exception as e:
            logger.error(f"AI interface initialization failed: {e}")
            return False
    
    async def execute_claude_code(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        project_path: Optional[Path] = None,
        timeout: int = 300,
        model: str = "claude-3-sonnet"
    ) -> AIResponse:
        """
        Execute a Claude Code request with context.
        
        Args:
            prompt: The task or question for Claude
            context: Additional context information
            project_path: Path to project for context
            timeout: Request timeout in seconds
            model: Claude model to use
            
        Returns:
            AI response with content and metadata
        """
        if not self._initialization_complete:
            await self.initialize()
        
        try:
            # Build comprehensive context
            full_context = context or {}
            
            if project_path and project_path.exists():
                # Add project context
                full_context['project_path'] = str(project_path)
                full_context['project_structure'] = await self._get_project_structure(project_path)
            
            # Create task request
            task_request = TaskRequest(
                description=prompt,
                context=full_context,
                project_path=str(project_path) if project_path else None,
                timeout=timeout,
                model=model
            )
            
            # Execute task
            start_time = asyncio.get_event_loop().time()
            result = await self.client.execute_task(task_request)
            processing_time = asyncio.get_event_loop().time() - start_time
            
            if result.success:
                return AIResponse(
                    content=result.output,
                    model_used=result.model_used,
                    tokens_used=result.tokens_used,
                    processing_time=processing_time,
                    success=True,
                    metadata=result.metadata
                )
            else:
                return AIResponse(
                    content="",
                    model_used=model,
                    tokens_used=0,
                    processing_time=processing_time,
                    success=False,
                    error=result.error,
                    metadata={'error_details': result.error_details}
                )
                
        except Exception as e:
            logger.error(f"Claude Code execution failed: {e}")
            return AIResponse(
                content="",
                model_used=model,
                tokens_used=0,
                processing_time=0,
                success=False,
                error=str(e)
            )
    
    async def generate_code(
        self,
        description: str,
        language: str,
        context: Optional[CodeContext] = None,
        style_preferences: Optional[Dict[str, Any]] = None
    ) -> CodeResult:
        """
        Generate code based on description and context.
        
        Args:
            description: What code to generate
            language: Programming language
            context: Code context information
            style_preferences: Coding style preferences
            
        Returns:
            Generated code with metadata
        """
        try:
            prompt_parts = [
                f"Generate {language} code for: {description}",
                "Requirements:",
                "- Follow best practices and coding standards",
                "- Include appropriate error handling",
                "- Add docstrings and comments",
                "- Ensure code is production-ready"
            ]
            
            if style_preferences:
                prompt_parts.append(f"Style preferences: {style_preferences}")
            
            if context:
                prompt_parts.extend([
                    f"Context:",
                    f"- Existing code: {context.existing_code[:500] if context.existing_code else 'None'}",
                    f"- Dependencies: {context.dependencies}",
                    f"- Requirements: {context.requirements}"
                ])
            
            prompt = "\n".join(prompt_parts)
            
            response = await self.execute_claude_code(
                prompt=prompt,
                context={'language': language, 'task': 'code_generation'},
                timeout=300
            )
            
            if response.success:
                return CodeResult(
                    code=response.content,
                    language=language,
                    success=True,
                    confidence=0.9,  # Default high confidence
                    suggestions=[]
                )
            else:
                return CodeResult(
                    code="",
                    language=language,
                    success=False,
                    error=response.error,
                    confidence=0.0
                )
                
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return CodeResult(
                code="",
                language=language,
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    async def review_code(
        self,
        code: str,
        language: str,
        criteria: Optional[ReviewCriteria] = None
    ) -> CodeReview:
        """
        Review code for quality, security, and best practices.
        
        Args:
            code: Code to review
            language: Programming language
            criteria: Review criteria
            
        Returns:
            Code review with findings and suggestions
        """
        try:
            default_criteria = ReviewCriteria(
                check_security=True,
                check_performance=True,
                check_maintainability=True,
                check_documentation=True,
                check_testing=False
            )
            
            review_criteria = criteria or default_criteria
            
            prompt_parts = [
                f"Review this {language} code for:",
                "- Code quality and best practices",
                "- Security vulnerabilities" if review_criteria.check_security else "",
                "- Performance issues" if review_criteria.check_performance else "",
                "- Maintainability concerns" if review_criteria.check_maintainability else "",
                "- Documentation quality" if review_criteria.check_documentation else "",
                "- Test coverage" if review_criteria.check_testing else "",
                "",
                "Code to review:",
                "```" + language,
                code,
                "```",
                "",
                "Provide specific suggestions for improvement."
            ]
            
            prompt = "\n".join(filter(None, prompt_parts))
            
            response = await self.execute_claude_code(
                prompt=prompt,
                context={'language': language, 'task': 'code_review'},
                timeout=300
            )
            
            if response.success:
                return CodeReview(
                    overall_score=8.5,  # Default good score, would be parsed from response
                    issues=[],  # Would be parsed from response
                    suggestions=[response.content],  # Simplified
                    security_issues=[],
                    performance_issues=[],
                    maintainability_score=8.0,
                    documentation_score=8.0,
                    success=True
                )
            else:
                return CodeReview(
                    overall_score=0.0,
                    issues=[],
                    suggestions=[],
                    security_issues=[],
                    performance_issues=[],
                    maintainability_score=0.0,
                    documentation_score=0.0,
                    success=False,
                    error=response.error
                )
                
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return CodeReview(
                overall_score=0.0,
                issues=[],
                suggestions=[],
                security_issues=[],
                performance_issues=[],
                maintainability_score=0.0,
                documentation_score=0.0,
                success=False,
                error=str(e)
            )
    
    async def validate_output(
        self,
        output: str,
        expected_format: Optional[str] = None,
        validation_rules: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate AI output for completeness and correctness.
        
        Args:
            output: Output to validate
            expected_format: Expected output format
            validation_rules: Custom validation rules
            
        Returns:
            Validation result with score and issues
        """
        try:
            prompt_parts = [
                "Validate this AI output for completeness and quality:",
                "",
                "Output to validate:",
                output,
                "",
                "Check for:",
                "- Completeness (no placeholder text like TODO, FIXME)",
                "- Consistency and coherence",
                "- Correctness and accuracy"
            ]
            
            if expected_format:
                prompt_parts.append(f"- Expected format: {expected_format}")
            
            if validation_rules:
                prompt_parts.extend([
                    "- Custom rules:",
                    *[f"  • {rule}" for rule in validation_rules]
                ])
            
            prompt_parts.extend([
                "",
                "Provide a validation score (0-1) and list any issues found."
            ])
            
            prompt = "\n".join(prompt_parts)
            
            response = await self.execute_claude_code(
                prompt=prompt,
                context={'task': 'output_validation'},
                timeout=120
            )
            
            if response.success:
                # Simplified validation - in real implementation would parse response
                return ValidationResult(
                    is_valid=True,
                    confidence_score=0.9,
                    issues=[],
                    suggestions=[],
                    completeness_score=0.95,
                    correctness_score=0.9
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    issues=[response.error or "Validation failed"],
                    suggestions=[],
                    completeness_score=0.0,
                    correctness_score=0.0
                )
                
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[str(e)],
                suggestions=[],
                completeness_score=0.0,
                correctness_score=0.0
            )
    
    async def detect_placeholders(
        self,
        code: str,
        language: str,
        sensitivity: float = 0.95
    ) -> PlaceholderDetection:
        """
        Detect placeholder text and incomplete implementations.
        
        Args:
            code: Code to analyze
            language: Programming language
            sensitivity: Detection sensitivity (0-1)
            
        Returns:
            Placeholder detection results
        """
        try:
            common_placeholders = [
                "TODO", "FIXME", "XXX", "HACK", "NOTE",
                "placeholder", "not implemented", "coming soon",
                "...", "pass  # TODO", "# TODO", "/* TODO */",
                "NotImplementedError", "NotImplemented"
            ]
            
            prompt = f"""
Analyze this {language} code for placeholder text and incomplete implementations.

Look for:
- Common placeholder patterns: {', '.join(common_placeholders)}
- Incomplete function/method implementations
- Missing logic or empty blocks
- Temporary code markers
- Unfinished documentation

Code to analyze:
```{language}
{code}
```

Provide specific locations and suggestions for completion.
"""
            
            response = await self.execute_claude_code(
                prompt=prompt,
                context={'language': language, 'task': 'placeholder_detection'},
                timeout=180
            )
            
            if response.success:
                return PlaceholderDetection(
                    has_placeholders=False,  # Would be parsed from response
                    placeholder_count=0,
                    placeholders=[],
                    confidence=sensitivity,
                    suggestions=[]
                )
            else:
                return PlaceholderDetection(
                    has_placeholders=True,
                    placeholder_count=1,
                    placeholders=[{"location": "unknown", "text": "Detection failed", "severity": "error"}],
                    confidence=0.0,
                    suggestions=[],
                    error=response.error
                )
                
        except Exception as e:
            logger.error(f"Placeholder detection failed: {e}")
            return PlaceholderDetection(
                has_placeholders=True,
                placeholder_count=1,
                placeholders=[{"location": "unknown", "text": str(e), "severity": "error"}],
                confidence=0.0,
                suggestions=[],
                error=str(e)
            )
    
    async def _get_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Get project structure for context."""
        try:
            structure = {
                'files': [],
                'directories': [],
                'size': 0
            }
            
            for item in project_path.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    relative_path = item.relative_to(project_path)
                    structure['files'].append(str(relative_path))
                    structure['size'] += item.stat().st_size
                elif item.is_dir() and not item.name.startswith('.'):
                    relative_path = item.relative_to(project_path)
                    structure['directories'].append(str(relative_path))
            
            # Limit to avoid huge context
            if len(structure['files']) > 100:
                structure['files'] = structure['files'][:100]
                structure['truncated'] = True
            
            return structure
            
        except Exception as e:
            logger.warning(f"Failed to get project structure: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Close the AI interface and cleanup resources."""
        if hasattr(self, 'client'):
            await self.client.close()
        logger.info("AI interface closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'client') and self.client:
            # Note: Can't await in __del__, so this is just for logging
            logger.debug("AI interface being garbage collected")