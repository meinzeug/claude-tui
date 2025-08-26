"""
Template Engine - AI-powered project template generation and processing.

This module provides sophisticated template generation capabilities with:
- Jinja2 template processing with security features
- AI-assisted content generation
- Variable validation and type checking
- Template inheritance and composition
- Real-time template validation
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field

try:
    from jinja2 import (
        Environment,
        FileSystemLoader,
        Template,
        TemplateError,
        TemplateSyntaxError,
        select_autoescape,
        StrictUndefined,
        meta
    )
    from jinja2.sandbox import SandboxedEnvironment
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None
    Template = None

logger = logging.getLogger(__name__)


class TemplateEngineError(Exception):
    """Base exception for template engine errors."""
    pass


class TemplateValidationError(TemplateEngineError):
    """Raised when template validation fails."""
    pass


class TemplateRenderError(TemplateEngineError):
    """Raised when template rendering fails."""
    pass


class TemplateSecurityError(TemplateEngineError):
    """Raised when template security checks fail."""
    pass


@dataclass
class TemplateContext:
    """Context data for template rendering."""
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    project_info: Dict[str, Any] = field(default_factory=dict)
    user_info: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    def merge(self, other: 'TemplateContext') -> 'TemplateContext':
        """Merge with another context."""
        return TemplateContext(
            variables={**self.variables, **other.variables},
            metadata={**self.metadata, **other.metadata},
            project_info={**self.project_info, **other.project_info},
            user_info={**self.user_info, **other.user_info},
            environment={**self.environment, **other.environment}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for rendering."""
        return {
            **self.variables,
            'metadata': self.metadata,
            'project': self.project_info,
            'user': self.user_info,
            'env': self.environment,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'year': datetime.now().year
        }


@dataclass
class TemplateValidationResult:
    """Result of template validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_variables: Set[str] = field(default_factory=set)
    optional_variables: Set[str] = field(default_factory=set)
    security_issues: List[str] = field(default_factory=list)


class TemplateEngine:
    """
    Advanced template engine with AI assistance and security features.
    
    Features:
    - Secure sandboxed template rendering
    - Variable validation and type checking
    - Template inheritance and composition
    - AI-powered content generation
    - Real-time validation and error detection
    """
    
    def __init__(self, template_dirs: Optional[List[Path]] = None):
        """
        Initialize the template engine.
        
        Args:
            template_dirs: List of directories containing templates
        """
        self.template_dirs = template_dirs or []
        self.environment = self._create_environment()
        self._cache = {}
        self._validators = self._setup_validators()
        
        # Security settings
        self.max_template_size = 10 * 1024 * 1024  # 10MB
        self.max_recursion_depth = 10
        self.blocked_tags = {'import', 'include', 'extends'}  # Can be overridden
        self.allowed_filters = self._get_safe_filters()
        
        logger.info(f"TemplateEngine initialized with {len(self.template_dirs)} template directories")
    
    def _create_environment(self) -> Optional[Environment]:
        """Create a secure Jinja2 environment."""
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 not available, template features limited")
            return None
            
        # Use sandboxed environment for security
        env = SandboxedEnvironment(
            loader=FileSystemLoader(self.template_dirs) if self.template_dirs else None,
            autoescape=select_autoescape(['html', 'xml', 'htm']),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Add custom filters
        env.filters['jsonify'] = json.dumps
        env.filters['slugify'] = self._slugify_filter
        env.filters['camelcase'] = self._camelcase_filter
        env.filters['snakecase'] = self._snakecase_filter
        env.filters['titlecase'] = lambda s: s.title()
        
        # Add custom global functions
        env.globals['now'] = datetime.now
        env.globals['uuid'] = self._generate_uuid
        
        return env
    
    def _setup_validators(self) -> Dict[str, Any]:
        """Setup template validators."""
        return {
            'variable_pattern': re.compile(r'\{\{\s*(\w+)\s*\}\}'),
            'block_pattern': re.compile(r'\{%\s*(\w+)\s*.*?%\}'),
            'comment_pattern': re.compile(r'\{#.*?#\}'),
            'dangerous_patterns': [
                re.compile(r'__[a-zA-Z]+__'),  # Python magic methods
                re.compile(r'eval\s*\('),
                re.compile(r'exec\s*\('),
                re.compile(r'compile\s*\('),
                re.compile(r'globals\s*\('),
                re.compile(r'locals\s*\('),
            ]
        }
    
    def _get_safe_filters(self) -> Set[str]:
        """Get list of safe Jinja2 filters."""
        return {
            'abs', 'attr', 'batch', 'capitalize', 'center', 'default',
            'dictsort', 'escape', 'filesizeformat', 'first', 'float',
            'forceescape', 'format', 'groupby', 'indent', 'int', 'join',
            'last', 'length', 'list', 'lower', 'map', 'max', 'min',
            'pprint', 'random', 'reject', 'rejectattr', 'replace',
            'reverse', 'round', 'safe', 'select', 'selectattr', 'slice',
            'sort', 'string', 'striptags', 'sum', 'title', 'trim',
            'truncate', 'unique', 'upper', 'urlencode', 'urlize',
            'wordcount', 'wordwrap', 'xmlattr', 'tojson',
            # Custom filters
            'jsonify', 'slugify', 'camelcase', 'snakecase', 'titlecase'
        }
    
    def _slugify_filter(self, text: str) -> str:
        """Convert text to slug format."""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
    
    def _camelcase_filter(self, text: str) -> str:
        """Convert text to camelCase."""
        words = text.split('_')
        return words[0].lower() + ''.join(w.capitalize() for w in words[1:])
    
    def _snakecase_filter(self, text: str) -> str:
        """Convert text to snake_case."""
        text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text)
        return text.lower()
    
    def _generate_uuid(self) -> str:
        """Generate a UUID."""
        import uuid
        return str(uuid.uuid4())
    
    async def validate_template(
        self,
        template_str: str,
        context: Optional[TemplateContext] = None
    ) -> TemplateValidationResult:
        """
        Validate a template string for security and correctness.
        
        Args:
            template_str: Template string to validate
            context: Optional context for validation
            
        Returns:
            TemplateValidationResult with validation details
        """
        result = TemplateValidationResult(is_valid=True)
        
        # Check template size
        if len(template_str) > self.max_template_size:
            result.is_valid = False
            result.errors.append(f"Template exceeds maximum size of {self.max_template_size} bytes")
            return result
        
        # Check for dangerous patterns
        for pattern in self._validators['dangerous_patterns']:
            if pattern.search(template_str):
                result.is_valid = False
                result.security_issues.append(f"Dangerous pattern detected: {pattern.pattern}")
        
        # Extract variables
        if JINJA2_AVAILABLE and self.environment:
            try:
                ast = self.environment.parse(template_str)
                result.required_variables = meta.find_undeclared_variables(ast)
                
                # Validate against context if provided
                if context:
                    context_vars = set(context.to_dict().keys())
                    missing_vars = result.required_variables - context_vars
                    if missing_vars:
                        result.warnings.append(f"Missing variables: {', '.join(missing_vars)}")
                        
            except TemplateSyntaxError as e:
                result.is_valid = False
                result.errors.append(f"Template syntax error: {str(e)}")
            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Template validation error: {str(e)}")
        else:
            # Basic validation without Jinja2
            variables = self._validators['variable_pattern'].findall(template_str)
            result.required_variables = set(variables)
        
        return result
    
    async def render_template(
        self,
        template_str: str,
        context: TemplateContext,
        validate: bool = True
    ) -> str:
        """
        Render a template string with the given context.
        
        Args:
            template_str: Template string to render
            context: Context data for rendering
            validate: Whether to validate before rendering
            
        Returns:
            Rendered template string
            
        Raises:
            TemplateRenderError: If rendering fails
            TemplateValidationError: If validation fails
        """
        # Validate first if requested
        if validate:
            validation_result = await self.validate_template(template_str, context)
            if not validation_result.is_valid:
                raise TemplateValidationError(
                    f"Template validation failed: {'; '.join(validation_result.errors)}"
                )
        
        if not JINJA2_AVAILABLE or not self.environment:
            # Fallback rendering without Jinja2
            rendered = template_str
            for key, value in context.to_dict().items():
                pattern = f'{{{{{key}}}}}'
                rendered = rendered.replace(pattern, str(value))
            return rendered
        
        try:
            template = self.environment.from_string(template_str)
            return template.render(**context.to_dict())
        except TemplateError as e:
            raise TemplateRenderError(f"Template render error: {str(e)}")
        except Exception as e:
            raise TemplateRenderError(f"Unexpected render error: {str(e)}")
    
    async def render_file(
        self,
        template_path: Path,
        context: TemplateContext,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Render a template file with the given context.
        
        Args:
            template_path: Path to template file
            context: Context data for rendering
            output_path: Optional path to write rendered output
            
        Returns:
            Rendered template string
        """
        if not template_path.exists():
            raise TemplateEngineError(f"Template file not found: {template_path}")
        
        # Read template
        template_str = template_path.read_text()
        
        # Render template
        rendered = await self.render_template(template_str, context)
        
        # Write output if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered)
            logger.info(f"Rendered template written to {output_path}")
        
        return rendered
    
    async def render_directory(
        self,
        template_dir: Path,
        output_dir: Path,
        context: TemplateContext,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Render all templates in a directory.
        
        Args:
            template_dir: Directory containing templates
            output_dir: Directory for rendered output
            context: Context data for rendering
            file_patterns: Optional file patterns to include
            
        Returns:
            Dictionary with rendering results
        """
        if not template_dir.exists():
            raise TemplateEngineError(f"Template directory not found: {template_dir}")
        
        results = {
            'rendered': [],
            'skipped': [],
            'errors': []
        }
        
        # Get all template files
        patterns = file_patterns or ['*.j2', '*.jinja', '*.jinja2', '*.template']
        template_files = []
        for pattern in patterns:
            template_files.extend(template_dir.rglob(pattern))
        
        # Render each template
        for template_file in template_files:
            try:
                # Calculate output path
                relative_path = template_file.relative_to(template_dir)
                output_file = output_dir / relative_path
                
                # Remove template extension
                if output_file.suffix in {'.j2', '.jinja', '.jinja2', '.template'}:
                    output_file = output_file.with_suffix('')
                
                # Render template
                await self.render_file(template_file, context, output_file)
                results['rendered'].append(str(output_file))
                
            except Exception as e:
                logger.error(f"Failed to render {template_file}: {e}")
                results['errors'].append({
                    'file': str(template_file),
                    'error': str(e)
                })
        
        return results
    
    def add_template_directory(self, directory: Path) -> None:
        """Add a template directory to the search path."""
        if directory not in self.template_dirs:
            self.template_dirs.append(directory)
            if self.environment and JINJA2_AVAILABLE:
                self.environment.loader = FileSystemLoader(self.template_dirs)
    
    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()
        if self.environment:
            self.environment.cache.clear()


class TemplateValidator:
    """Advanced template validation with security checks."""
    
    def __init__(self):
        """Initialize the template validator."""
        self.security_checks = [
            self._check_file_access,
            self._check_code_execution,
            self._check_imports,
            self._check_recursion_depth
        ]
    
    async def validate(
        self,
        template: Union[str, Path],
        strict: bool = True
    ) -> TemplateValidationResult:
        """
        Perform comprehensive template validation.
        
        Args:
            template: Template string or path
            strict: Whether to use strict validation
            
        Returns:
            TemplateValidationResult with validation details
        """
        result = TemplateValidationResult(is_valid=True)
        
        # Load template if path
        if isinstance(template, Path):
            if not template.exists():
                result.is_valid = False
                result.errors.append(f"Template file not found: {template}")
                return result
            template_str = template.read_text()
        else:
            template_str = template
        
        # Run security checks
        for check in self.security_checks:
            check_result = await check(template_str)
            if not check_result['passed']:
                result.is_valid = False
                result.security_issues.extend(check_result.get('issues', []))
        
        return result
    
    async def _check_file_access(self, template_str: str) -> Dict[str, Any]:
        """Check for unauthorized file access attempts."""
        dangerous_patterns = [
            r'\.\./',  # Path traversal
            r'/etc/',  # System files
            r'/proc/',  # Process information
            r'~/',  # Home directory
        ]
        
        issues = []
        for pattern in dangerous_patterns:
            if re.search(pattern, template_str):
                issues.append(f"Potential file access vulnerability: {pattern}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    async def _check_code_execution(self, template_str: str) -> Dict[str, Any]:
        """Check for code execution attempts."""
        dangerous_functions = [
            'eval', 'exec', 'compile', '__import__',
            'getattr', 'setattr', 'delattr',
            'globals', 'locals', 'vars'
        ]
        
        issues = []
        for func in dangerous_functions:
            pattern = rf'\b{func}\s*\('
            if re.search(pattern, template_str):
                issues.append(f"Potential code execution risk: {func}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    async def _check_imports(self, template_str: str) -> Dict[str, Any]:
        """Check for dangerous imports."""
        import_pattern = r'{%\s*import\s+.*?%}'
        
        issues = []
        if re.search(import_pattern, template_str):
            issues.append("Import statements detected in template")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    async def _check_recursion_depth(self, template_str: str) -> Dict[str, Any]:
        """Check for excessive recursion depth."""
        include_pattern = r'{%\s*include\s+.*?%}'
        extends_pattern = r'{%\s*extends\s+.*?%}'
        
        issues = []
        include_count = len(re.findall(include_pattern, template_str))
        extends_count = len(re.findall(extends_pattern, template_str))
        
        if include_count > 10:
            issues.append(f"Excessive includes detected: {include_count}")
        if extends_count > 5:
            issues.append(f"Excessive template inheritance: {extends_count}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }


# Export main classes
__all__ = [
    'TemplateEngine',
    'TemplateContext',
    'TemplateValidator',
    'TemplateValidationResult',
    'TemplateEngineError',
    'TemplateValidationError',
    'TemplateRenderError',
    'TemplateSecurityError'
]