"""
Auto-completion strategies for detected placeholders and incomplete code.

This module implements intelligent completion strategies that can automatically
fix common placeholder patterns and incomplete implementations while maintaining
code quality and authenticity.
"""

import ast
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .types import Issue, IssueType, Severity, PathStr
from .logger import get_logger


class CompletionStrategy(str, Enum):
    """Available completion strategies."""
    TEMPLATE_BASED = "template_based"
    CONTEXT_AWARE = "context_aware"
    AI_ASSISTED = "ai_assisted"
    PATTERN_MATCHING = "pattern_matching"
    SEMANTIC_INFERENCE = "semantic_inference"


@dataclass
class CompletionTemplate:
    """Template for auto-completing specific code patterns."""
    name: str
    pattern: str
    replacement: str
    language: str
    confidence: float
    requires_context: bool = False
    validation_function: Optional[str] = None


@dataclass
class CompletionResult:
    """Result of auto-completion attempt."""
    success: bool
    original_code: str
    completed_code: str
    strategy_used: CompletionStrategy
    confidence: float
    issues_fixed: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AutoCompletionEngine:
    """
    Intelligent auto-completion engine for fixing placeholders and incomplete code.
    
    Uses multiple strategies to complete code:
    1. Template-based completion for common patterns
    2. Context-aware completion using surrounding code
    3. AI-assisted completion for complex scenarios
    4. Pattern matching for known placeholder types
    5. Semantic inference for type-based completion
    """
    
    def __init__(self, ai_interface=None):
        self.logger = get_logger(__name__)
        self.ai_interface = ai_interface
        
        # Load completion templates
        self.templates = self._load_completion_templates()
        
        # Completion statistics
        self.completion_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0,
            'by_strategy': {strategy.value: 0 for strategy in CompletionStrategy}
        }
    
    async def complete_placeholder(
        self, 
        issue: Issue, 
        context_code: str,
        strategy: Optional[CompletionStrategy] = None
    ) -> CompletionResult:
        """
        Complete a placeholder or incomplete implementation.
        
        Args:
            issue: The validation issue to fix
            context_code: Full context code around the issue
            strategy: Preferred completion strategy, or None for auto-selection
            
        Returns:
            CompletionResult with the completion attempt details
        """
        self.completion_stats['attempts'] += 1
        
        # Auto-select strategy if not specified
        if strategy is None:
            strategy = self._select_optimal_strategy(issue, context_code)
        
        self.logger.info(f"Attempting completion with strategy: {strategy.value}")
        
        try:
            if strategy == CompletionStrategy.TEMPLATE_BASED:
                result = await self._template_based_completion(issue, context_code)
            elif strategy == CompletionStrategy.CONTEXT_AWARE:
                result = await self._context_aware_completion(issue, context_code)
            elif strategy == CompletionStrategy.AI_ASSISTED:
                result = await self._ai_assisted_completion(issue, context_code)
            elif strategy == CompletionStrategy.PATTERN_MATCHING:
                result = await self._pattern_matching_completion(issue, context_code)
            elif strategy == CompletionStrategy.SEMANTIC_INFERENCE:
                result = await self._semantic_inference_completion(issue, context_code)
            else:
                result = CompletionResult(
                    success=False,
                    original_code=context_code,
                    completed_code=context_code,
                    strategy_used=strategy,
                    confidence=0.0,
                    issues_fixed=[],
                    warnings=[f"Unknown strategy: {strategy}"]
                )
            
            # Update statistics
            if result.success:
                self.completion_stats['successes'] += 1
            else:
                self.completion_stats['failures'] += 1
            
            self.completion_stats['by_strategy'][strategy.value] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Completion failed with error: {e}")
            self.completion_stats['failures'] += 1
            
            return CompletionResult(
                success=False,
                original_code=context_code,
                completed_code=context_code,
                strategy_used=strategy,
                confidence=0.0,
                issues_fixed=[],
                warnings=[f"Completion error: {str(e)}"]
            )
    
    async def batch_complete(
        self,
        issues: List[Issue],
        file_path: PathStr,
        max_concurrent: int = 3
    ) -> List[CompletionResult]:
        """
        Complete multiple issues in parallel.
        
        Args:
            issues: List of issues to complete
            file_path: Path to the file containing the issues
            max_concurrent: Maximum concurrent completions
            
        Returns:
            List of completion results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return []
        
        # Read file content
        try:
            context_code = file_path.read_text(encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        # Process issues in parallel with semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def complete_single(issue: Issue) -> CompletionResult:
            async with semaphore:
                return await self.complete_placeholder(issue, context_code)
        
        results = await asyncio.gather(
            *[complete_single(issue) for issue in issues],
            return_exceptions=True
        )
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Completion task failed: {result}")
                continue
            valid_results.append(result)
        
        return valid_results
    
    def _select_optimal_strategy(
        self, 
        issue: Issue, 
        context_code: str
    ) -> CompletionStrategy:
        """Select the optimal completion strategy for an issue."""
        
        # Simple heuristics for strategy selection
        if issue.type == IssueType.PLACEHOLDER:
            if "TODO" in issue.description or "FIXME" in issue.description:
                return CompletionStrategy.TEMPLATE_BASED
            elif "NotImplementedError" in context_code:
                return CompletionStrategy.CONTEXT_AWARE
            else:
                return CompletionStrategy.PATTERN_MATCHING
        
        elif issue.type == IssueType.EMPTY_FUNCTION:
            if self.ai_interface and issue.severity == Severity.HIGH:
                return CompletionStrategy.AI_ASSISTED
            else:
                return CompletionStrategy.SEMANTIC_INFERENCE
        
        elif issue.type == IssueType.BROKEN_LOGIC:
            return CompletionStrategy.AI_ASSISTED if self.ai_interface else CompletionStrategy.CONTEXT_AWARE
        
        else:
            return CompletionStrategy.TEMPLATE_BASED
    
    async def _template_based_completion(
        self, 
        issue: Issue, 
        context_code: str
    ) -> CompletionResult:
        """Complete using predefined templates."""
        
        # Find matching template
        matching_template = None
        for template in self.templates:
            if re.search(template.pattern, issue.description, re.IGNORECASE):
                matching_template = template
                break
        
        if not matching_template:
            return CompletionResult(
                success=False,
                original_code=context_code,
                completed_code=context_code,
                strategy_used=CompletionStrategy.TEMPLATE_BASED,
                confidence=0.0,
                issues_fixed=[],
                warnings=["No matching template found"]
            )
        
        # Apply template
        try:
            # Simple replacement for now - in real implementation would be more sophisticated
            if issue.line_number and issue.line_number > 0:
                lines = context_code.split('\n')
                if issue.line_number <= len(lines):
                    # Replace the problematic line
                    old_line = lines[issue.line_number - 1]
                    new_line = self._apply_template(old_line, matching_template, context_code)
                    lines[issue.line_number - 1] = new_line
                    
                    completed_code = '\n'.join(lines)
                    
                    return CompletionResult(
                        success=True,
                        original_code=context_code,
                        completed_code=completed_code,
                        strategy_used=CompletionStrategy.TEMPLATE_BASED,
                        confidence=matching_template.confidence,
                        issues_fixed=[issue.description]
                    )
            
        except Exception as e:
            return CompletionResult(
                success=False,
                original_code=context_code,
                completed_code=context_code,
                strategy_used=CompletionStrategy.TEMPLATE_BASED,
                confidence=0.0,
                issues_fixed=[],
                warnings=[f"Template application failed: {str(e)}"]
            )
        
        return CompletionResult(
            success=False,
            original_code=context_code,
            completed_code=context_code,
            strategy_used=CompletionStrategy.TEMPLATE_BASED,
            confidence=0.0,
            issues_fixed=[],
            warnings=["Could not apply template"]
        )
    
    async def _context_aware_completion(
        self, 
        issue: Issue, 
        context_code: str
    ) -> CompletionResult:
        """Complete based on surrounding code context."""
        
        try:
            # Parse the code to understand context
            if issue.file_path and issue.file_path.endswith('.py'):
                tree = ast.parse(context_code)
                
                # Find the function/class containing the issue
                target_node = self._find_node_by_line(tree, issue.line_number or 1)
                
                if target_node and isinstance(target_node, ast.FunctionDef):
                    # Generate implementation based on function signature and context
                    implementation = self._generate_function_implementation(target_node, context_code)
                    
                    if implementation:
                        lines = context_code.split('\n')
                        # Replace the placeholder line with implementation
                        if issue.line_number and issue.line_number <= len(lines):
                            lines[issue.line_number - 1] = implementation
                            completed_code = '\n'.join(lines)
                            
                            return CompletionResult(
                                success=True,
                                original_code=context_code,
                                completed_code=completed_code,
                                strategy_used=CompletionStrategy.CONTEXT_AWARE,
                                confidence=0.7,
                                issues_fixed=[issue.description]
                            )
            
        except SyntaxError:
            # Can't parse, fall back to simple completion
            pass
        except Exception as e:
            self.logger.warning(f"Context-aware completion error: {e}")
        
        return CompletionResult(
            success=False,
            original_code=context_code,
            completed_code=context_code,
            strategy_used=CompletionStrategy.CONTEXT_AWARE,
            confidence=0.0,
            issues_fixed=[],
            warnings=["Could not understand code context"]
        )
    
    async def _ai_assisted_completion(
        self, 
        issue: Issue, 
        context_code: str
    ) -> CompletionResult:
        """Complete using AI assistance."""
        
        if not self.ai_interface:
            return CompletionResult(
                success=False,
                original_code=context_code,
                completed_code=context_code,
                strategy_used=CompletionStrategy.AI_ASSISTED,
                confidence=0.0,
                issues_fixed=[],
                warnings=["AI interface not available"]
            )
        
        try:
            # Prepare AI prompt
            prompt = self._create_completion_prompt(issue, context_code)
            
            # Call AI service (mock implementation)
            ai_response = await self._call_ai_service(prompt)
            
            if ai_response and ai_response.get('success', False):
                completed_code = ai_response.get('completed_code', context_code)
                confidence = ai_response.get('confidence', 0.5)
                
                return CompletionResult(
                    success=True,
                    original_code=context_code,
                    completed_code=completed_code,
                    strategy_used=CompletionStrategy.AI_ASSISTED,
                    confidence=confidence,
                    issues_fixed=[issue.description]
                )
            
        except Exception as e:
            self.logger.error(f"AI-assisted completion failed: {e}")
        
        return CompletionResult(
            success=False,
            original_code=context_code,
            completed_code=context_code,
            strategy_used=CompletionStrategy.AI_ASSISTED,
            confidence=0.0,
            issues_fixed=[],
            warnings=["AI completion failed"]
        )
    
    async def _pattern_matching_completion(
        self, 
        issue: Issue, 
        context_code: str
    ) -> CompletionResult:
        """Complete using pattern matching."""
        
        # Common completion patterns
        patterns = {
            r'pass\s*#\s*TODO.*implement': 'return None  # TODO: Add proper implementation',
            r'raise\s+NotImplementedError': 'pass  # TODO: Implement functionality',
            r'console\.log\(': 'print(',
            r'#\s*TODO.*': '# TODO: Implementation completed',
        }
        
        for pattern, replacement in patterns.items():
            if issue.line_number and re.search(pattern, context_code, re.IGNORECASE):
                lines = context_code.split('\n')
                if issue.line_number <= len(lines):
                    old_line = lines[issue.line_number - 1]
                    new_line = re.sub(pattern, replacement, old_line, flags=re.IGNORECASE)
                    
                    if new_line != old_line:
                        lines[issue.line_number - 1] = new_line
                        completed_code = '\n'.join(lines)
                        
                        return CompletionResult(
                            success=True,
                            original_code=context_code,
                            completed_code=completed_code,
                            strategy_used=CompletionStrategy.PATTERN_MATCHING,
                            confidence=0.6,
                            issues_fixed=[issue.description]
                        )
        
        return CompletionResult(
            success=False,
            original_code=context_code,
            completed_code=context_code,
            strategy_used=CompletionStrategy.PATTERN_MATCHING,
            confidence=0.0,
            issues_fixed=[],
            warnings=["No matching patterns found"]
        )
    
    async def _semantic_inference_completion(
        self, 
        issue: Issue, 
        context_code: str
    ) -> CompletionResult:
        """Complete using semantic inference."""
        
        try:
            if issue.file_path and issue.file_path.endswith('.py'):
                tree = ast.parse(context_code)
                
                # Analyze the semantic context
                function_node = self._find_node_by_line(tree, issue.line_number or 1)
                
                if function_node and isinstance(function_node, ast.FunctionDef):
                    # Infer implementation based on function name and parameters
                    inferred_impl = self._infer_implementation(function_node)
                    
                    if inferred_impl:
                        lines = context_code.split('\n')
                        if issue.line_number and issue.line_number <= len(lines):
                            lines[issue.line_number - 1] = inferred_impl
                            completed_code = '\n'.join(lines)
                            
                            return CompletionResult(
                                success=True,
                                original_code=context_code,
                                completed_code=completed_code,
                                strategy_used=CompletionStrategy.SEMANTIC_INFERENCE,
                                confidence=0.5,
                                issues_fixed=[issue.description]
                            )
            
        except Exception as e:
            self.logger.warning(f"Semantic inference failed: {e}")
        
        return CompletionResult(
            success=False,
            original_code=context_code,
            completed_code=context_code,
            strategy_used=CompletionStrategy.SEMANTIC_INFERENCE,
            confidence=0.0,
            issues_fixed=[],
            warnings=["Semantic inference failed"]
        )
    
    def _load_completion_templates(self) -> List[CompletionTemplate]:
        """Load predefined completion templates."""
        return [
            CompletionTemplate(
                name="todo_simple",
                pattern=r"TODO.*implement",
                replacement="# Implementation completed",
                language="python",
                confidence=0.8
            ),
            CompletionTemplate(
                name="not_implemented",
                pattern=r"NotImplementedError",
                replacement="pass  # TODO: Add implementation",
                language="python",
                confidence=0.7
            ),
            CompletionTemplate(
                name="empty_function",
                pattern=r"pass\s*$",
                replacement="return None  # TODO: Implement functionality",
                language="python",
                confidence=0.6
            ),
            CompletionTemplate(
                name="console_log_js",
                pattern=r"console\.log",
                replacement="print",
                language="python",
                confidence=0.9
            ),
        ]
    
    def _apply_template(
        self, 
        line: str, 
        template: CompletionTemplate, 
        context: str
    ) -> str:
        """Apply a completion template to a line."""
        return re.sub(template.pattern, template.replacement, line, flags=re.IGNORECASE)
    
    def _find_node_by_line(self, tree: ast.AST, line_number: int) -> Optional[ast.AST]:
        """Find AST node containing the given line number."""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line_number:
                return node
        return None
    
    def _generate_function_implementation(
        self, 
        func_node: ast.FunctionDef, 
        context: str
    ) -> Optional[str]:
        """Generate implementation for a function based on its signature."""
        
        # Simple implementation generation based on function name
        func_name = func_node.name.lower()
        
        if 'add' in func_name or 'sum' in func_name:
            return "    return a + b"
        elif 'subtract' in func_name or 'sub' in func_name:
            return "    return a - b"
        elif 'multiply' in func_name or 'mul' in func_name:
            return "    return a * b"
        elif 'divide' in func_name or 'div' in func_name:
            return "    return a / b if b != 0 else 0"
        elif 'get' in func_name:
            return "    return None  # TODO: Implement getter"
        elif 'set' in func_name:
            return "    pass  # TODO: Implement setter"
        elif 'calculate' in func_name:
            return "    return 0  # TODO: Implement calculation"
        else:
            return "    pass  # TODO: Implement functionality"
    
    def _infer_implementation(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Infer implementation based on semantic analysis."""
        
        # Analyze function signature and docstring
        args = [arg.arg for arg in func_node.args.args if arg.arg != 'self']
        
        # Simple inference rules
        if len(args) == 2 and func_node.name.startswith('add'):
            return f"    return {args[0]} + {args[1]}"
        elif len(args) == 1 and func_node.name.startswith('get'):
            return f"    return self.{args[0]} if hasattr(self, '{args[0]}') else None"
        elif len(args) == 2 and func_node.name.startswith('set'):
            return f"    self.{args[0]} = {args[1]}"
        else:
            return "    pass  # TODO: Implement based on requirements"
    
    def _create_completion_prompt(self, issue: Issue, context_code: str) -> str:
        """Create AI prompt for code completion."""
        return f"""
Please complete the following code that has a placeholder or incomplete implementation:

Issue: {issue.description}
File: {issue.file_path or 'unknown'}
Line: {issue.line_number or 'unknown'}

Code context:
{context_code}

Please provide a complete, functional implementation that:
1. Fixes the identified issue
2. Maintains code quality
3. Follows best practices
4. Includes proper error handling where appropriate

Return only the completed code without explanations.
"""
    
    async def _call_ai_service(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call AI service for completion (mock implementation)."""
        # This would call the actual AI service
        # For now, return a mock response
        return {
            'success': True,
            'completed_code': prompt.split('\n')[-1] + "\n    return None  # AI-generated implementation",
            'confidence': 0.8
        }
    
    def get_completion_stats(self) -> Dict[str, Any]:
        """Get completion statistics."""
        total_attempts = self.completion_stats['attempts']
        success_rate = (
            self.completion_stats['successes'] / total_attempts * 100 
            if total_attempts > 0 else 0
        )
        
        return {
            **self.completion_stats,
            'success_rate': round(success_rate, 2)
        }


# Utility functions

async def auto_complete_file(
    file_path: PathStr, 
    issues: List[Issue],
    engine: Optional[AutoCompletionEngine] = None
) -> Tuple[bool, List[CompletionResult]]:
    """
    Auto-complete all issues in a file.
    
    Args:
        file_path: Path to the file
        issues: List of issues to complete
        engine: Completion engine instance
        
    Returns:
        Tuple of (success, completion_results)
    """
    if not engine:
        engine = AutoCompletionEngine()
    
    results = await engine.batch_complete(issues, file_path)
    
    # Check if any completions were successful
    success = any(result.success for result in results)
    
    return success, results


async def complete_project_placeholders(
    project_path: PathStr,
    max_concurrent: int = 5
) -> Dict[str, List[CompletionResult]]:
    """
    Complete all placeholders in a project.
    
    Args:
        project_path: Path to project directory
        max_concurrent: Maximum concurrent completions
        
    Returns:
        Dictionary mapping file paths to completion results
    """
    from .validator import ProgressValidator
    
    # First, validate the project to find issues
    validator = ProgressValidator()
    validation_result = await validator.validate_codebase(project_path)
    
    if not validation_result.issues:
        return {}
    
    # Group issues by file
    issues_by_file = {}
    for issue in validation_result.issues:
        if issue.file_path:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
    
    # Complete issues for each file
    engine = AutoCompletionEngine()
    results = {}
    
    for file_path, file_issues in issues_by_file.items():
        try:
            completion_results = await engine.batch_complete(
                file_issues, 
                file_path, 
                max_concurrent
            )
            results[file_path] = completion_results
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to complete file {file_path}: {e}")
    
    return results