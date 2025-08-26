"""
Semantic Analyzer - Advanced semantic validation for AI-generated code.

Performs deep semantic analysis including:
- Code structure validation
- Logic flow analysis  
- Dependency validation
- API usage verification
- Best practice compliance
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.models.project import Project
from src.claude_tui.models.task import DevelopmentTask
from src.claude_tui.validation.types import ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class SemanticIssueType(Enum):
    """Types of semantic issues."""
    SYNTAX_ERROR = "syntax_error"
    LOGICAL_ERROR = "logical_error"
    UNUSED_IMPORT = "unused_import"
    UNDEFINED_VARIABLE = "undefined_variable"
    UNREACHABLE_CODE = "unreachable_code"
    CIRCULAR_IMPORT = "circular_import"
    INVALID_API_USAGE = "invalid_api_usage"
    MISSING_DEPENDENCY = "missing_dependency"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_ISSUE = "security_issue"
    STYLE_VIOLATION = "style_violation"


@dataclass
class SemanticContext:
    """Context for semantic analysis."""
    file_path: Optional[Path]
    language: str
    project: Optional[Project]
    imported_modules: Set[str]
    defined_variables: Set[str]
    defined_functions: Set[str]
    defined_classes: Set[str]
    dependencies: List[str]


@dataclass
class CodeElement:
    """Represents a code element for analysis."""
    name: str
    element_type: str  # 'function', 'class', 'variable', etc.
    start_line: int
    end_line: int
    complexity: int
    dependencies: List[str]
    used: bool = False


class SemanticAnalyzer:
    """
    Advanced semantic validation for AI-generated code.
    
    Performs comprehensive semantic analysis to detect logical errors,
    API misuse, dependency issues, and code quality problems.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the semantic analyzer.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        
        # Language-specific analyzers
        self._python_analyzer = PythonSemanticAnalyzer()
        self._javascript_analyzer = JavaScriptSemanticAnalyzer()
        
        # Configuration
        self._strict_mode = True
        self._check_unused_imports = True
        self._check_performance = True
        self._check_security = True
        
        logger.info("Semantic analyzer initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the semantic analyzer.
        """
        logger.info("Initializing semantic analyzer")
        
        try:
            # Load configuration
            analyzer_config = await self.config_manager.get_setting('semantic_analyzer', {})
            self._strict_mode = analyzer_config.get('strict_mode', True)
            self._check_unused_imports = analyzer_config.get('check_unused_imports', True)
            self._check_performance = analyzer_config.get('check_performance', True)
            self._check_security = analyzer_config.get('check_security', True)
            
            # Initialize language-specific analyzers
            await self._python_analyzer.initialize(analyzer_config)
            await self._javascript_analyzer.initialize(analyzer_config)
            
            logger.info("Semantic analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic analyzer: {e}")
            raise
    
    async def analyze_content(
        self,
        content: str,
        file_path: Optional[Path] = None,
        project: Optional[Project] = None,
        language: Optional[str] = None
    ) -> List[ValidationIssue]:
        """
        Analyze content for semantic issues.
        
        Args:
            content: Content to analyze
            file_path: File path for context
            project: Associated project
            language: Programming language (inferred if None)
            
        Returns:
            List of validation issues
        """
        if not content.strip():
            return []
        
        # Infer language if not provided
        if not language and file_path:
            language = self._infer_language(file_path)
        
        if not language:
            logger.warning("Could not determine language, skipping semantic analysis")
            return []
        
        logger.debug(f"Analyzing semantic content for {language}")
        
        try:
            # Build semantic context
            context = await self._build_semantic_context(
                content, file_path, project, language
            )
            
            issues = []
            
            # Language-specific analysis
            if language == 'python':
                lang_issues = await self._python_analyzer.analyze(
                    content, context, file_path
                )
                issues.extend(lang_issues)
            
            elif language in ['javascript', 'typescript']:
                lang_issues = await self._javascript_analyzer.analyze(
                    content, context, file_path
                )
                issues.extend(lang_issues)
            
            # General semantic analysis
            general_issues = await self._general_semantic_analysis(
                content, context, language
            )
            issues.extend(general_issues)
            
            # Cross-reference analysis
            if project:
                cross_ref_issues = await self._cross_reference_analysis(
                    content, context, project
                )
                issues.extend(cross_ref_issues)
            
            logger.debug(f"Found {len(issues)} semantic issues")
            return issues
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return [ValidationIssue(
                id="semantic_analysis_error",
                description=f"Semantic analysis failed: {e}",
                severity=ValidationSeverity.MEDIUM,
                file_path=str(file_path) if file_path else None,
                issue_type="analysis_error"
            )]
    
    async def analyze_generated_content(
        self,
        content: str,
        task: Optional[DevelopmentTask] = None,
        project: Optional[Project] = None
    ) -> List[ValidationIssue]:
        """
        Analyze AI-generated content with task context.
        
        Args:
            content: Generated content to analyze
            task: Original task context
            project: Associated project
            
        Returns:
            List of validation issues
        """
        issues = await self.analyze_content(
            content=content,
            project=project
        )
        
        # Add task-specific validation
        if task:
            task_issues = await self._validate_task_semantic_requirements(
                content, task, project
            )
            issues.extend(task_issues)
        
        return issues
    
    async def cleanup(self) -> None:
        """
        Cleanup semantic analyzer resources.
        """
        logger.info("Cleaning up semantic analyzer")
        
        await self._python_analyzer.cleanup()
        await self._javascript_analyzer.cleanup()
        
        logger.info("Semantic analyzer cleanup completed")
    
    # Private helper methods
    
    def _infer_language(self, file_path: Path) -> Optional[str]:
        """
        Infer programming language from file extension.
        """
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        return extension_map.get(file_path.suffix.lower())
    
    async def _build_semantic_context(
        self,
        content: str,
        file_path: Optional[Path],
        project: Optional[Project],
        language: str
    ) -> SemanticContext:
        """
        Build semantic context for analysis.
        """
        context = SemanticContext(
            file_path=file_path,
            language=language,
            project=project,
            imported_modules=set(),
            defined_variables=set(),
            defined_functions=set(),
            defined_classes=set(),
            dependencies=[]
        )
        
        # Extract imports and definitions based on language
        if language == 'python':
            await self._extract_python_context(content, context)
        elif language in ['javascript', 'typescript']:
            await self._extract_javascript_context(content, context)
        
        return context
    
    async def _extract_python_context(self, content: str, context: SemanticContext) -> None:
        """
        Extract Python-specific context information.
        """
        try:
            # Parse AST for accurate analysis
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        context.imported_modules.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        context.imported_modules.add(node.module)
                elif isinstance(node, ast.FunctionDef):
                    context.defined_functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    context.defined_classes.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    context.defined_variables.add(node.id)
        
        except SyntaxError as e:
            logger.warning(f"Python syntax error during context extraction: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract Python context: {e}")
    
    async def _extract_javascript_context(self, content: str, context: SemanticContext) -> None:
        """
        Extract JavaScript-specific context information.
        """
        # Simple regex-based extraction for JavaScript
        # In a real implementation, would use a proper JS parser
        
        # Extract imports
        import_patterns = [
            r'import\\s+.*?\\s+from\\s+["\']([^"\']+)["\']',
            r'require\\s*\\(\\s*["\']([^"\']+)["\']\\s*\\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            context.imported_modules.update(matches)
        
        # Extract function definitions
        function_patterns = [
            r'function\\s+(\\w+)\\s*\\(',
            r'(\\w+)\\s*=\\s*function\\s*\\(',
            r'(\\w+)\\s*=\\s*\\([^)]*\\)\\s*=>',
            r'const\\s+(\\w+)\\s*=\\s*\\([^)]*\\)\\s*=>'
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, content)
            context.defined_functions.update(matches)
        
        # Extract class definitions
        class_matches = re.findall(r'class\\s+(\\w+)', content)
        context.defined_classes.update(class_matches)
    
    async def _general_semantic_analysis(
        self,
        content: str,
        context: SemanticContext,
        language: str
    ) -> List[ValidationIssue]:
        """
        General semantic analysis applicable to all languages.
        """
        issues = []
        
        # Check for unused imports
        if self._check_unused_imports:
            unused_issues = await self._check_unused_imports_analysis(
                content, context
            )
            issues.extend(unused_issues)
        
        # Check for basic logical issues
        logical_issues = await self._check_logical_issues(
            content, context
        )
        issues.extend(logical_issues)
        
        # Security checks
        if self._check_security:
            security_issues = await self._check_security_issues(
                content, context
            )
            issues.extend(security_issues)
        
        # Performance checks
        if self._check_performance:
            performance_issues = await self._check_performance_issues(
                content, context
            )
            issues.extend(performance_issues)
        
        return issues
    
    async def _check_unused_imports_analysis(
        self,
        content: str,
        context: SemanticContext
    ) -> List[ValidationIssue]:
        """
        Check for unused imports.
        """
        issues = []
        
        for imported_module in context.imported_modules:
            # Simple check - look for module usage in content
            module_name = imported_module.split('.')[-1]  # Get last part of module path
            
            # Check if module is used anywhere in the content
            if not re.search(rf'\\b{re.escape(module_name)}\\b', content):
                issues.append(ValidationIssue(
                    id=f"unused_import_{module_name}",
                    description=f"Unused import: {imported_module}",
                    severity=ValidationSeverity.LOW,
                    file_path=str(context.file_path) if context.file_path else None,
                    issue_type="unused_import",
                    auto_fixable=True,
                    suggested_fix=f"Remove unused import: {imported_module}"
                ))
        
        return issues
    
    async def _check_logical_issues(
        self,
        content: str,
        context: SemanticContext
    ) -> List[ValidationIssue]:
        """
        Check for logical issues in code.
        """
        issues = []
        
        lines = content.split('\
')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for unreachable code after return
            if 'return ' in stripped and i < len(lines):
                next_line = lines[i].strip() if i < len(lines) else ""
                if next_line and not next_line.startswith('#') and not next_line.startswith('//'):
                    # Check if it's not another return or end of function
                    if not any(keyword in next_line for keyword in ['def ', 'function ', 'class ', '}']):
                        issues.append(ValidationIssue(
                            id=f"unreachable_code_{i+1}",
                            description="Unreachable code after return statement",
                            severity=ValidationSeverity.MEDIUM,
                            file_path=str(context.file_path) if context.file_path else None,
                            line_number=i+1,
                            issue_type="unreachable_code",
                            suggested_fix="Remove unreachable code or restructure logic"
                        ))
            
            # Check for potential infinite loops
            if 'while True:' in stripped or 'for(;;)' in stripped:
                # Look for break or return in the loop
                loop_has_exit = False
                for j in range(i, min(i+10, len(lines))):
                    if 'break' in lines[j] or 'return' in lines[j]:
                        loop_has_exit = True
                        break
                
                if not loop_has_exit:
                    issues.append(ValidationIssue(
                        id=f"potential_infinite_loop_{i}",
                        description="Potential infinite loop without exit condition",
                        severity=ValidationSeverity.HIGH,
                        file_path=str(context.file_path) if context.file_path else None,
                        line_number=i,
                        issue_type="logical_error",
                        suggested_fix="Add break condition or return statement"
                    ))
        
        return issues
    
    async def _check_security_issues(
        self,
        content: str,
        context: SemanticContext
    ) -> List[ValidationIssue]:
        """
        Check for security issues in code.
        """
        issues = []
        
        # Common security anti-patterns
        security_patterns = [
            (r'eval\\s*\\(', "Use of eval() function is dangerous", ValidationSeverity.HIGH),
            (r'exec\\s*\\(', "Use of exec() function is dangerous", ValidationSeverity.HIGH),
            (r'innerHTML\\s*=', "Direct innerHTML assignment can lead to XSS", ValidationSeverity.MEDIUM),
            (r'password\\s*=\\s*["\'][^"\']+["\']', "Hardcoded password detected", ValidationSeverity.CRITICAL),
            (r'api_key\\s*=\\s*["\'][^"\']+["\']', "Hardcoded API key detected", ValidationSeverity.CRITICAL),
            (r'subprocess\\.call\\s*\\(.*shell\\s*=\\s*True', "Shell=True in subprocess can be dangerous", ValidationSeverity.MEDIUM)
        ]
        
        for pattern, description, severity in security_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_number = content[:match.start()].count('\
') + 1
                
                issues.append(ValidationIssue(
                    id=f"security_issue_{line_number}_{hash(pattern)}",
                    description=description,
                    severity=severity,
                    file_path=str(context.file_path) if context.file_path else None,
                    line_number=line_number,
                    issue_type="security_issue",
                    suggested_fix=f"Review and secure the usage: {match.group(0)[:30]}"
                ))
        
        return issues
    
    async def _check_performance_issues(
        self,
        content: str,
        context: SemanticContext
    ) -> List[ValidationIssue]:
        """
        Check for performance issues in code.
        """
        issues = []
        
        # Performance anti-patterns
        performance_patterns = [
            (r'for\\s+\\w+\\s+in\\s+range\\s*\\(\\s*len\\s*\\([^)]+\\)\\s*\\)', "Use enumerate() instead of range(len())", ValidationSeverity.LOW),
            (r'\\.append\\s*\\([^)]+\\)\\s*(?:\
\\s*)*\\}?\\s*(?:\
\\s*)*\\}?\\s*(?:for|while)', "List comprehension might be more efficient", ValidationSeverity.LOW),
            (r'\\+\\s*=.*\\+.*for\\s+', "String concatenation in loop is inefficient", ValidationSeverity.MEDIUM)
        ]
        
        for pattern, description, severity in performance_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.DOTALL))
            for match in matches:
                line_number = content[:match.start()].count('\
') + 1
                
                issues.append(ValidationIssue(
                    id=f"performance_issue_{line_number}_{hash(pattern)}",
                    description=description,
                    severity=severity,
                    file_path=str(context.file_path) if context.file_path else None,
                    line_number=line_number,
                    issue_type="performance_issue",
                    suggested_fix=f"Consider optimization: {description}"
                ))
        
        return issues
    
    async def _cross_reference_analysis(
        self,
        content: str,
        context: SemanticContext,
        project: Project
    ) -> List[ValidationIssue]:
        """
        Cross-reference analysis with project context.
        """
        issues = []
        
        # Check for undefined variables that might be defined in project
        # This would require more sophisticated project analysis
        
        # Check for missing dependencies
        for module in context.imported_modules:
            # Simple check - this would be more sophisticated in reality
            if module not in ['os', 'sys', 'json', 're', 'datetime', 'pathlib']:
                # Check if it's a local import or external dependency
                if '.' not in module and not module.startswith('.'):
                    # Might be missing from requirements
                    issues.append(ValidationIssue(
                        id=f"potential_missing_dep_{module}",
                        description=f"Potentially missing dependency: {module}",
                        severity=ValidationSeverity.LOW,
                        file_path=str(context.file_path) if context.file_path else None,
                        issue_type="missing_dependency",
                        suggested_fix=f"Ensure {module} is installed and in requirements"
                    ))
        
        return issues
    
    async def _validate_task_semantic_requirements(
        self,
        content: str,
        task: DevelopmentTask,
        project: Optional[Project]
    ) -> List[ValidationIssue]:
        """
        Validate semantic requirements specific to the task.
        """
        issues = []
        
        # Check if generated content matches task requirements
        if hasattr(task, 'semantic_requirements'):
            for requirement in task.semantic_requirements:
                if not self._check_semantic_requirement(content, requirement):
                    issues.append(ValidationIssue(
                        id=f"task_semantic_requirement_{hash(requirement)}",
                        description=f"Task semantic requirement not met: {requirement}",
                        severity=ValidationSeverity.HIGH,
                        issue_type="task_requirement",
                        suggested_fix=f"Implement semantic requirement: {requirement}"
                    ))
        
        return issues
    
    def _check_semantic_requirement(self, content: str, requirement: str) -> bool:
        """
        Check if content meets a semantic requirement.
        """
        # Simple implementation - would be more sophisticated in practice
        requirement_lower = requirement.lower()
        content_lower = content.lower()
        
        # Check for key terms in the requirement
        requirement_words = requirement_lower.split()
        return any(word in content_lower for word in requirement_words)


class PythonSemanticAnalyzer:
    """Python-specific semantic analyzer."""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Python analyzer."""
        self.check_imports = config.get('check_imports', True)
        self.check_unused_vars = config.get('check_unused_vars', True)
        self.check_complexity = config.get('check_complexity', True)
        self.max_complexity = config.get('max_complexity', 10)
        
        logger.debug("Python semantic analyzer initialized")
    
    async def analyze(
        self,
        content: str,
        context: SemanticContext,
        file_path: Optional[Path]
    ) -> List[ValidationIssue]:
        """Analyze Python content."""
        issues = []
        
        try:
            # Parse AST for syntax validation
            tree = ast.parse(content)
            
            # Analyze AST for Python-specific issues
            issues.extend(await self._analyze_ast(tree, context, file_path))
            
        except SyntaxError as e:
            issues.append(ValidationIssue(
                id="python_syntax_error",
                description=f"Python syntax error: {e.msg}",
                severity=ValidationSeverity.CRITICAL,
                file_path=str(file_path) if file_path else None,
                line_number=e.lineno,
                column_number=e.offset,
                issue_type="syntax_error",
                suggested_fix="Fix syntax error"
            ))
        
        return issues
    
    async def _analyze_ast(
        self,
        tree: ast.AST,
        context: SemanticContext,
        file_path: Optional[Path]
    ) -> List[ValidationIssue]:
        """Analyze Python AST for issues."""
        issues = []
        
        # Track variables and functions
        defined_vars = set()
        used_vars = set()
        imported_modules = set()
        unused_imports = set()
        
        # Walk AST and check for issues
        for node in ast.walk(tree):
            # Check for empty except blocks
            if isinstance(node, ast.ExceptHandler):
                if not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                    issues.append(ValidationIssue(
                        id=f"empty_except_{node.lineno}",
                        description="Empty except block - should handle exceptions properly",
                        severity=ValidationSeverity.MEDIUM,
                        file_path=str(file_path) if file_path else None,
                        line_number=node.lineno,
                        issue_type="logical_error",
                        suggested_fix="Add proper exception handling or logging"
                    ))
            
            # Track variable definitions
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if hasattr(node, 'targets'):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_vars.add(target.id)
                elif hasattr(node, 'target') and isinstance(node.target, ast.Name):
                    defined_vars.add(node.target.id)
            
            # Track variable usage
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
            
            # Track imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
                    unused_imports.add(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)
                    unused_imports.add(node.module)
                for alias in node.names:
                    imported_modules.add(alias.name)
                    unused_imports.add(alias.name)
            
            # Check for functions with only pass
            elif isinstance(node, ast.FunctionDef):
                if (len(node.body) == 1 and isinstance(node.body[0], ast.Pass) and 
                    not node.decorator_list):
                    issues.append(ValidationIssue(
                        id=f"stub_function_{node.name}_{node.lineno}",
                        description=f"Function '{node.name}' contains only pass statement",
                        severity=ValidationSeverity.HIGH,
                        file_path=str(file_path) if file_path else None,
                        line_number=node.lineno,
                        issue_type="logical_error",
                        suggested_fix=f"Implement the '{node.name}' function"
                    ))
                    
                # Check function complexity
                complexity = self._calculate_complexity(node)
                if complexity > self.max_complexity:
                    issues.append(ValidationIssue(
                        id=f"high_complexity_{node.name}_{node.lineno}",
                        description=f"Function '{node.name}' has high complexity ({complexity})",
                        severity=ValidationSeverity.MEDIUM,
                        file_path=str(file_path) if file_path else None,
                        line_number=node.lineno,
                        issue_type="performance_issue",
                        suggested_fix=f"Refactor '{node.name}' to reduce complexity"
                    ))
            
            # Check for classes with only pass
            elif isinstance(node, ast.ClassDef):
                if (len(node.body) == 1 and isinstance(node.body[0], ast.Pass) and 
                    not node.decorator_list):
                    issues.append(ValidationIssue(
                        id=f"empty_class_{node.name}_{node.lineno}",
                        description=f"Class '{node.name}' is empty",
                        severity=ValidationSeverity.HIGH,
                        file_path=str(file_path) if file_path else None,
                        line_number=node.lineno,
                        issue_type="logical_error",
                        suggested_fix=f"Implement the '{node.name}' class"
                    ))
            
            # Check for potential SQL injection
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr in ('execute', 'executemany')):
                    if (node.args and isinstance(node.args[0], ast.BinOp) and 
                        isinstance(node.args[0].op, ast.Mod)):
                        issues.append(ValidationIssue(
                            id=f"sql_injection_risk_{node.lineno}",
                            description="Potential SQL injection vulnerability",
                            severity=ValidationSeverity.HIGH,
                            file_path=str(file_path) if file_path else None,
                            line_number=node.lineno,
                            issue_type="security_issue",
                            suggested_fix="Use parameterized queries instead of string formatting"
                        ))
        
        # Check for unused variables
        if self.check_unused_vars:
            unused_vars = defined_vars - used_vars - {'self', '_'}  # Exclude common exceptions
            for var in unused_vars:
                issues.append(ValidationIssue(
                    id=f"unused_var_{var}",
                    description=f"Variable '{var}' is defined but never used",
                    severity=ValidationSeverity.LOW,
                    file_path=str(file_path) if file_path else None,
                    line_number=1,  # Would need better tracking for exact line
                    issue_type="logical_error",
                    suggested_fix=f"Remove unused variable '{var}' or use it"
                ))
        
        # Update context
        context.imported_modules.update(imported_modules)
        context.defined_variables.update(defined_vars)
        
        return issues
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    async def cleanup(self) -> None:
        """Cleanup Python analyzer."""
        pass


class JavaScriptSemanticAnalyzer:
    """JavaScript-specific semantic analyzer."""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize JavaScript analyzer."""
        self.check_console_logs = config.get('check_console_logs', True)
        self.check_unused_vars = config.get('check_unused_vars', True)
        self.check_eval_usage = config.get('check_eval_usage', True)
        
        logger.debug("JavaScript semantic analyzer initialized")
    
    async def analyze(
        self,
        content: str,
        context: SemanticContext,
        file_path: Optional[Path]
    ) -> List[ValidationIssue]:
        """Analyze JavaScript content."""
        issues = []
        
        try:
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Check for console.log statements
                if self.check_console_logs and 'console.log' in line:
                    issues.append(ValidationIssue(
                        id=f"console_log_{line_num}",
                        description="console.log statement found - should be removed in production",
                        severity=ValidationSeverity.LOW,
                        file_path=str(file_path) if file_path else None,
                        line_number=line_num,
                        issue_type="style_violation",
                        suggested_fix="Replace with proper logging or remove"
                    ))
                
                # Check for eval usage
                if self.check_eval_usage and re.search(r'\beval\s*\(', line):
                    issues.append(ValidationIssue(
                        id=f"eval_usage_{line_num}",
                        description="eval() usage detected - security risk",
                        severity=ValidationSeverity.HIGH,
                        file_path=str(file_path) if file_path else None,
                        line_number=line_num,
                        issue_type="security_issue",
                        suggested_fix="Avoid using eval() - use safer alternatives"
                    ))
                
                # Check for empty functions
                if re.match(r'\s*function\s+\w+\s*\([^)]*\)\s*\{\s*\}\s*$', line):
                    issues.append(ValidationIssue(
                        id=f"empty_function_{line_num}",
                        description="Empty function detected",
                        severity=ValidationSeverity.MEDIUM,
                        file_path=str(file_path) if file_path else None,
                        line_number=line_num,
                        issue_type="logical_error",
                        suggested_fix="Implement function body or add TODO comment"
                    ))
                
                # Check for == vs === usage (loose equality)
                if re.search(r'[^=!]==[^=]', line):
                    issues.append(ValidationIssue(
                        id=f"loose_equality_{line_num}",
                        description="Using == instead of === (loose equality)",
                        severity=ValidationSeverity.LOW,
                        file_path=str(file_path) if file_path else None,
                        line_number=line_num,
                        issue_type="style_violation",
                        suggested_fix="Use === for strict equality comparison"
                    ))
                
                # Check for var instead of let/const
                if re.match(r'\s*var\s+\w+', line):
                    issues.append(ValidationIssue(
                        id=f"var_usage_{line_num}",
                        description="Using 'var' instead of 'let' or 'const'",
                        severity=ValidationSeverity.LOW,
                        file_path=str(file_path) if file_path else None,
                        line_number=line_num,
                        issue_type="style_violation",
                        suggested_fix="Use 'let' for mutable variables or 'const' for constants"
                    ))
        
        except Exception as e:
            logger.error(f"Error analyzing JavaScript content: {e}")
        
        return issues
    
    async def cleanup(self) -> None:
        """Cleanup JavaScript analyzer."""
        pass