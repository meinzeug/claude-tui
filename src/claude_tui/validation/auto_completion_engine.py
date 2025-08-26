"""
Auto Completion Engine - Intelligent auto-fixing for validation issues.

Provides automated fixing capabilities for common validation issues including:
- Placeholder replacement
- Code completion
- Import optimization
- Error fixing
- Style corrections
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.models.project import Project
from src.claude_tui.validation.types import ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Auto-fix strategies."""
    SIMPLE_REPLACEMENT = "simple_replacement"
    TEMPLATE_BASED = "template_based"
    CONTEXT_AWARE = "context_aware"
    AI_ASSISTED = "ai_assisted"
    RULE_BASED = "rule_based"


@dataclass
class FixRule:
    """Rule for automatic fixing."""
    issue_type: str
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    strategy: FixStrategy = FixStrategy.SIMPLE_REPLACEMENT
    conditions: List[str] = None
    template: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0


@dataclass
class FixResult:
    """Result of auto-fix attempt."""
    success: bool
    original_content: str
    fixed_content: Optional[str] = None
    changes_made: List[str] = None
    confidence: float = 0.0
    strategy_used: Optional[FixStrategy] = None
    error_message: Optional[str] = None


class AutoCompletionEngine:
    """
    Intelligent auto-fixing for validation issues.
    
    Provides automated solutions for common code quality issues,
    placeholders, and validation problems using rule-based and
    context-aware fixing strategies.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the auto completion engine.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        
        # Fix rules and templates
        self._fix_rules: List[FixRule] = []
        self._code_templates: Dict[str, str] = {}
        self._language_handlers: Dict[str, 'LanguageHandler'] = {}
        
        # Configuration
        self._auto_fix_enabled = True
        self._min_confidence = 0.7
        self._max_changes_per_fix = 10
        
        logger.info("Auto completion engine initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the auto completion engine.
        """
        logger.info("Initializing auto completion engine")
        
        try:
            # Load configuration
            engine_config = await self.config_manager.get_setting('auto_completion', {})
            self._auto_fix_enabled = engine_config.get('enabled', True)
            self._min_confidence = engine_config.get('min_confidence', 0.7)
            self._max_changes_per_fix = engine_config.get('max_changes_per_fix', 10)
            
            # Load fix rules
            await self._load_fix_rules()
            
            # Load code templates
            await self._load_code_templates()
            
            # Initialize language handlers
            await self._initialize_language_handlers()
            
            logger.info(f"Auto completion engine initialized with {len(self._fix_rules)} rules")
            
        except Exception as e:
            logger.error(f"Failed to initialize auto completion engine: {e}")
            raise
    
    async def fix_issue(
        self,
        issue: ValidationIssue,
        content: Optional[str] = None,
        project: Optional[Project] = None
    ) -> Optional[str]:
        """
        Attempt to automatically fix a validation issue.
        
        Args:
            issue: Validation issue to fix
            content: Content to fix (required if file_path not in issue)
            project: Associated project for context
            
        Returns:
            Fixed content if successful, None otherwise
        """
        if not self._auto_fix_enabled:
            return None
        
        logger.debug(f"Attempting to fix issue: {issue.id}")
        
        try:
            # Get content if not provided
            if content is None:
                if issue.file_path:
                    file_path = Path(issue.file_path)
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    else:
                        logger.error(f"File not found: {issue.file_path}")
                        return None
                else:
                    logger.error("No content provided and no file path in issue")
                    return None
            
            # Find applicable fix rules
            applicable_rules = await self._find_applicable_rules(issue, content, project)
            
            if not applicable_rules:
                logger.debug(f"No applicable rules found for issue: {issue.id}")
                return None
            
            # Try fixing with each rule (ordered by confidence)
            for rule in sorted(applicable_rules, key=lambda r: r.confidence, reverse=True):
                fix_result = await self._apply_fix_rule(rule, issue, content, project)
                
                if fix_result.success and fix_result.confidence >= self._min_confidence:
                    logger.info(
                        f"Successfully fixed issue {issue.id} using {rule.strategy.value} "
                        f"(confidence: {fix_result.confidence:.2f})"
                    )
                    return fix_result.fixed_content
            
            logger.debug(f"All fix attempts failed for issue: {issue.id}")
            return None
            
        except Exception as e:
            logger.error(f"Error fixing issue {issue.id}: {e}")
            return None
    
    async def fix_multiple_issues(
        self,
        issues: List[ValidationIssue],
        content: str,
        project: Optional[Project] = None
    ) -> Optional[str]:
        """
        Fix multiple issues in content.
        
        Args:
            issues: List of validation issues to fix
            content: Content to fix
            project: Associated project for context
            
        Returns:
            Fixed content if successful, None otherwise
        """
        if not issues or not self._auto_fix_enabled:
            return None
        
        logger.info(f"Attempting to fix {len(issues)} issues")
        
        try:
            current_content = content
            changes_made = 0
            
            # Sort issues by severity and line number for stable fixing
            sorted_issues = sorted(
                issues,
                key=lambda i: (i.severity.value, i.line_number or 0)
            )
            
            for issue in sorted_issues:
                if changes_made >= self._max_changes_per_fix:
                    logger.warning(f"Reached maximum changes limit: {self._max_changes_per_fix}")
                    break
                
                fixed_content = await self.fix_issue(issue, current_content, project)
                
                if fixed_content and fixed_content != current_content:
                    current_content = fixed_content
                    changes_made += 1
                    logger.debug(f"Fixed issue {issue.id} (change {changes_made})")
            
            if changes_made > 0:
                logger.info(f"Successfully applied {changes_made} fixes")
                return current_content
            else:
                logger.debug("No fixes were applied")
                return None
                
        except Exception as e:
            logger.error(f"Error fixing multiple issues: {e}")
            return None
    
    async def suggest_completion(
        self,
        partial_code: str,
        context: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """
        Suggest code completion for partial code.
        
        Args:
            partial_code: Partial code to complete
            context: Additional context information
            language: Programming language
            
        Returns:
            List of completion suggestions
        """
        suggestions = []
        
        try:
            # Infer language if not provided
            if not language and context and 'file_path' in context:
                language = self._infer_language(Path(context['file_path']))
            
            # Get language-specific suggestions
            if language in self._language_handlers:
                handler = self._language_handlers[language]
                lang_suggestions = await handler.suggest_completion(
                    partial_code, context
                )
                suggestions.extend(lang_suggestions)
            
            # Template-based suggestions
            template_suggestions = await self._get_template_suggestions(
                partial_code, language, context
            )
            suggestions.extend(template_suggestions)
            
            # Remove duplicates and sort by relevance
            unique_suggestions = list(dict.fromkeys(suggestions))  # Preserves order
            
            return unique_suggestions[:10]  # Limit to top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating code suggestions: {e}")
            return []
    
    async def cleanup(self) -> None:
        """
        Cleanup auto completion engine resources.
        """
        logger.info("Cleaning up auto completion engine")
        
        # Cleanup language handlers
        for handler in self._language_handlers.values():
            await handler.cleanup()
        
        self._language_handlers.clear()
        
        logger.info("Auto completion engine cleanup completed")
    
    # Private helper methods
    
    async def _load_fix_rules(self) -> None:
        """
        Load automatic fix rules.
        """
        rules = [
            # Placeholder fixes
            FixRule(
                issue_type="placeholder",
                pattern=r"#\\s*(TODO|FIXME):?\\s*(.*)",
                strategy=FixStrategy.CONTEXT_AWARE,
                confidence=0.8,
                template="# Implemented: {description}"
            ),
            FixRule(
                issue_type="placeholder",
                pattern=r"pass\\s*#.*?(TODO|FIXME|IMPLEMENTATION)",
                replacement="# Implementation needed",
                strategy=FixStrategy.SIMPLE_REPLACEMENT,
                confidence=0.7
            ),
            FixRule(
                issue_type="placeholder",
                pattern=r"\\.\\.\\.\\s*$",
                replacement="pass  # Implementation placeholder",
                strategy=FixStrategy.SIMPLE_REPLACEMENT,
                confidence=0.9
            ),
            
            # Unused imports
            FixRule(
                issue_type="unused_import",
                strategy=FixStrategy.RULE_BASED,
                confidence=0.95
            ),
            
            # Empty function bodies
            FixRule(
                issue_type="incomplete_function",
                strategy=FixStrategy.TEMPLATE_BASED,
                template="pass  # Implementation needed: {function_name}",
                confidence=0.8
            ),
            
            # Security fixes
            FixRule(
                issue_type="security_issue",
                pattern=r"eval\\s*\\(",
                replacement="# SECURITY: eval() removed - use ast.literal_eval() or safer alternative",
                strategy=FixStrategy.SIMPLE_REPLACEMENT,
                confidence=0.6
            ),
        ]
        
        self._fix_rules = rules
    
    async def _load_code_templates(self) -> None:
        """
        Load code templates for completion.
        """
        templates = {
            'python_function': '''
def {function_name}({parameters}):
    """{description}"""
    {implementation}
    return {return_value}
''',
            
            'python_class': '''
class {class_name}:
    """{description}"""
    
    def __init__(self{init_parameters}):
        {init_implementation}
    
    def {method_name}(self{method_parameters}):
        """{method_description}"""
        {method_implementation}
''',
            
            'javascript_function': '''
function {function_name}({parameters}) {
    // {description}
    {implementation}
    return {return_value};
}
''',
            
            'error_handler': '''
try {
    {implementation}
} catch (error) {
    console.error('Error in {function_name}:', error);
    {error_handling}
}
'''
        }
        
        self._code_templates = templates
    
    async def _initialize_language_handlers(self) -> None:
        """
        Initialize language-specific handlers.
        """
        self._language_handlers = {
            'python': PythonCompletionHandler(),
            'javascript': JavaScriptCompletionHandler(),
            'typescript': JavaScriptCompletionHandler(),  # Reuse JS handler
        }
        
        # Initialize each handler
        for handler in self._language_handlers.values():
            await handler.initialize()
    
    async def _find_applicable_rules(
        self,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> List[FixRule]:
        """
        Find fix rules applicable to the given issue.
        """
        applicable_rules = []
        
        for rule in self._fix_rules:
            if rule.issue_type == issue.issue_type or rule.issue_type == "*":
                # Check if rule conditions are met
                if await self._check_rule_conditions(rule, issue, content, project):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _check_rule_conditions(
        self,
        rule: FixRule,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> bool:
        """
        Check if rule conditions are satisfied.
        """
        if not rule.conditions:
            return True
        
        # Check each condition
        for condition in rule.conditions:
            if not await self._evaluate_condition(condition, issue, content, project):
                return False
        
        return True
    
    async def _evaluate_condition(
        self,
        condition: str,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> bool:
        """
        Evaluate a rule condition.
        """
        # Simple condition evaluation
        # In practice, this would be more sophisticated
        
        if condition == "has_line_number":
            return issue.line_number is not None
        elif condition == "is_python":
            return issue.file_path and issue.file_path.endswith('.py')
        elif condition == "is_javascript":
            return issue.file_path and (issue.file_path.endswith('.js') or issue.file_path.endswith('.ts'))
        
        return True
    
    async def _apply_fix_rule(
        self,
        rule: FixRule,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> FixResult:
        """
        Apply a fix rule to resolve an issue.
        """
        try:
            if rule.strategy == FixStrategy.SIMPLE_REPLACEMENT:
                return await self._apply_simple_replacement(rule, issue, content)
            
            elif rule.strategy == FixStrategy.TEMPLATE_BASED:
                return await self._apply_template_based_fix(rule, issue, content, project)
            
            elif rule.strategy == FixStrategy.CONTEXT_AWARE:
                return await self._apply_context_aware_fix(rule, issue, content, project)
            
            elif rule.strategy == FixStrategy.RULE_BASED:
                return await self._apply_rule_based_fix(rule, issue, content, project)
            
            else:
                return FixResult(
                    success=False,
                    original_content=content,
                    error_message=f"Unsupported fix strategy: {rule.strategy}"
                )
                
        except Exception as e:
            return FixResult(
                success=False,
                original_content=content,
                error_message=str(e)
            )
    
    async def _apply_simple_replacement(
        self,
        rule: FixRule,
        issue: ValidationIssue,
        content: str
    ) -> FixResult:
        """
        Apply simple pattern replacement.
        """
        if not rule.pattern or not rule.replacement:
            return FixResult(
                success=False,
                original_content=content,
                error_message="Missing pattern or replacement"
            )
        
        # Perform replacement
        fixed_content = re.sub(rule.pattern, rule.replacement, content, flags=re.IGNORECASE)
        
        if fixed_content != content:
            return FixResult(
                success=True,
                original_content=content,
                fixed_content=fixed_content,
                changes_made=[f"Replaced pattern: {rule.pattern}"],
                confidence=rule.confidence,
                strategy_used=rule.strategy
            )
        else:
            return FixResult(
                success=False,
                original_content=content,
                error_message="Pattern not found"
            )
    
    async def _apply_template_based_fix(
        self,
        rule: FixRule,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> FixResult:
        """
        Apply template-based fix.
        """
        if not rule.template:
            return FixResult(
                success=False,
                original_content=content,
                error_message="No template specified"
            )
        
        # Extract variables from context
        variables = await self._extract_template_variables(issue, content, project)
        
        try:
            # Format template with variables
            replacement = rule.template.format(**variables)
            
            # Apply replacement at issue location
            fixed_content = await self._apply_replacement_at_location(
                content, replacement, issue.line_number, issue.column_number
            )
            
            return FixResult(
                success=True,
                original_content=content,
                fixed_content=fixed_content,
                changes_made=[f"Applied template: {rule.template[:50]}..."],
                confidence=rule.confidence,
                strategy_used=rule.strategy
            )
            
        except KeyError as e:
            return FixResult(
                success=False,
                original_content=content,
                error_message=f"Missing template variable: {e}"
            )
    
    async def _apply_context_aware_fix(
        self,
        rule: FixRule,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> FixResult:
        """
        Apply context-aware fix.
        """
        # Analyze context around the issue
        context_info = await self._analyze_issue_context(issue, content, project)
        
        # Generate context-appropriate fix
        if issue.issue_type == "placeholder" and any(keyword in (issue.context or {}).get('matched_text', '') for keyword in ["TODO", "FIXME"]):
            # Try to implement based on implementation comment
            todo_text = self._extract_todo_text(issue)
            if todo_text:
                implementation = await self._generate_implementation_from_todo(todo_text, context_info)
                if implementation:
                    fixed_content = await self._replace_issue_content(content, issue, implementation)
                    return FixResult(
                        success=True,
                        original_content=content,
                        fixed_content=fixed_content,
                        changes_made=[f"Implemented: {todo_text[:30]}..."],
                        confidence=rule.confidence * 0.8,  # Lower confidence for generated code
                        strategy_used=rule.strategy
                    )
        
        return FixResult(
            success=False,
            original_content=content,
            error_message="Context-aware fix not applicable"
        )
    
    async def _apply_rule_based_fix(
        self,
        rule: FixRule,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> FixResult:
        """
        Apply rule-based fix.
        """
        if issue.issue_type == "unused_import":
            # Remove unused import
            fixed_content = await self._remove_unused_import(content, issue)
            if fixed_content != content:
                return FixResult(
                    success=True,
                    original_content=content,
                    fixed_content=fixed_content,
                    changes_made=["Removed unused import"],
                    confidence=rule.confidence,
                    strategy_used=rule.strategy
                )
        
        return FixResult(
            success=False,
            original_content=content,
            error_message="Rule-based fix not implemented"
        )
    
    def _infer_language(self, file_path: Path) -> Optional[str]:
        """
        Infer programming language from file path.
        """
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript'
        }
        
        return extension_map.get(file_path.suffix.lower())
    
    async def _get_template_suggestions(
        self,
        partial_code: str,
        language: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Get template-based code suggestions.
        """
        suggestions = []
        
        # Match partial code to templates
        if language == 'python':
            if 'def ' in partial_code:
                suggestions.append(self._code_templates.get('python_function', ''))
            elif 'class ' in partial_code:
                suggestions.append(self._code_templates.get('python_class', ''))
        
        elif language == 'javascript':
            if 'function ' in partial_code:
                suggestions.append(self._code_templates.get('javascript_function', ''))
        
        return [s for s in suggestions if s]  # Filter empty strings
    
    async def _extract_template_variables(
        self,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> Dict[str, str]:
        """
        Extract variables for template formatting.
        """
        variables = {
            'function_name': 'new_function',
            'parameters': '',
            'description': 'Function description needed',
            'implementation': 'pass  # Implementation needed',
            'return_value': 'None'
        }
        
        # Try to extract actual function name if available
        if issue.line_number:
            lines = content.split('\
')
            if issue.line_number <= len(lines):
                line = lines[issue.line_number - 1]
                
                # Extract function name from line
                match = re.search(r'def\\s+(\\w+)', line)
                if match:
                    variables['function_name'] = match.group(1)
        
        return variables
    
    async def _analyze_issue_context(
        self,
        issue: ValidationIssue,
        content: str,
        project: Optional[Project]
    ) -> Dict[str, Any]:
        """
        Analyze context around an issue.
        """
        context = {
            'surrounding_lines': [],
            'indentation': '',
            'in_function': False,
            'in_class': False
        }
        
        if issue.line_number:
            lines = content.split('\
')
            start_line = max(0, issue.line_number - 3)
            end_line = min(len(lines), issue.line_number + 3)
            
            context['surrounding_lines'] = lines[start_line:end_line]
            
            if issue.line_number <= len(lines):
                current_line = lines[issue.line_number - 1]
                context['indentation'] = len(current_line) - len(current_line.lstrip())
        
        return context
    
    def _extract_todo_text(self, issue: ValidationIssue) -> Optional[str]:
        """
        Extract implementation text from issue context.
        """
        if not issue.context or 'matched_text' not in issue.context:
            return None
        
        matched_text = issue.context['matched_text']
        
        # Extract text after TODO/FIXME
        match = re.search(r'(TODO|FIXME):?\\s*(.*)', matched_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None
    
    async def _generate_implementation_from_todo(
        self,
        todo_text: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate implementation based on implementation text.
        """
        # Simple implementation generation
        # In practice, this would use AI or more sophisticated analysis
        
        todo_lower = todo_text.lower()
        
        if 'return' in todo_lower:
            return "return None  # Implementation: Add proper return value"
        elif 'print' in todo_lower or 'log' in todo_lower:
            return f'print("{todo_text}")  # Implementation: Add proper logging'
        elif 'validate' in todo_lower or 'check' in todo_lower:
            return "if not condition:  # Implementation: Define proper validation condition\
    raise ValueError('Validation failed')"
        else:
            return f'pass  # Implementation: {todo_text}'
    
    async def _apply_replacement_at_location(
        self,
        content: str,
        replacement: str,
        line_number: Optional[int],
        column_number: Optional[int]
    ) -> str:
        """
        Apply replacement at specific location.
        """
        if not line_number:
            return content + '\
' + replacement
        
        lines = content.split('\
')
        
        if line_number <= len(lines):
            # Replace the specific line
            lines[line_number - 1] = replacement
        else:
            # Append if line number is beyond content
            lines.append(replacement)
        
        return '\
'.join(lines)
    
    async def _replace_issue_content(
        self,
        content: str,
        issue: ValidationIssue,
        replacement: str
    ) -> str:
        """
        Replace content related to the issue.
        """
        if not issue.line_number:
            return content + '\
' + replacement
        
        lines = content.split('\
')
        
        if issue.line_number <= len(lines):
            lines[issue.line_number - 1] = replacement
        
        return '\
'.join(lines)
    
    async def _remove_unused_import(
        self,
        content: str,
        issue: ValidationIssue
    ) -> str:
        """
        Remove unused import from content.
        """
        if not issue.line_number:
            return content
        
        lines = content.split('\
')
        
        if issue.line_number <= len(lines):
            # Remove the import line
            lines.pop(issue.line_number - 1)
        
        return '\
'.join(lines)


class LanguageHandler:
    """Base class for language-specific completion handlers."""
    
    async def initialize(self) -> None:
        """Initialize the handler."""
        pass
    
    async def suggest_completion(
        self,
        partial_code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Suggest code completion."""
        return []
    
    async def cleanup(self) -> None:
        """Cleanup handler resources."""
        pass


class PythonCompletionHandler(LanguageHandler):
    """Python-specific completion handler."""
    
    async def suggest_completion(
        self,
        partial_code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Suggest Python code completion."""
        suggestions = []
        
        # Basic Python completions
        if partial_code.strip().endswith(':'):
            suggestions.append('pass')
            suggestions.append('# Implementation needed')
        
        if 'def ' in partial_code and '(' in partial_code and not partial_code.rstrip().endswith(':'):
            suggestions.append(partial_code.rstrip() + ':')
        
        return suggestions


class JavaScriptCompletionHandler(LanguageHandler):
    """JavaScript-specific completion handler."""
    
    async def suggest_completion(
        self,
        partial_code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Suggest JavaScript code completion."""
        suggestions = []
        
        # Basic JavaScript completions
        if 'function' in partial_code and '{' not in partial_code:
            suggestions.append(partial_code.rstrip() + ' {')
        
        if partial_code.strip().endswith('{'):
            suggestions.append('// Implementation needed')
            suggestions.append('return null;')
        
        return suggestions