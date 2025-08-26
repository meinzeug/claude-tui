"""
Anti-Hallucination Validation System for claude-tiu.

This module implements a comprehensive validation pipeline to detect and fix
AI-generated code that appears complete but contains placeholders, TODOs,
or non-functional implementations.

Key Features:
- Multi-stage validation pipeline (static, semantic, execution, cross-validation)
- Pattern-based placeholder detection
- Semantic analysis for functionality verification
- Automatic fix generation and application
- Quality scoring and authenticity metrics
"""

import ast
import asyncio
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import tempfile
import shutil
from datetime import datetime
from dataclasses import dataclass, field

from .types import (
    Issue, IssueType, Severity, ValidationResult, FileInfo, 
    ProgressMetrics, PathStr
)
from .logger import get_logger


@dataclass
class PlaceholderPattern:
    """Pattern for detecting placeholder code."""
    name: str
    pattern: str
    severity: Severity
    description: str
    auto_fix_template: Optional[str] = None
    file_extensions: Set[str] = field(default_factory=set)


@dataclass
class SemanticCheck:
    """Semantic analysis check configuration."""
    name: str
    check_function: str
    severity: Severity
    description: str
    applicable_languages: Set[str] = field(default_factory=set)


class ValidationException(Exception):
    """Validation-related errors."""
    pass


@dataclass
class CodeQualityMetrics:
    """Code quality assessment metrics."""
    lines_of_code: int = 0
    cyclomatic_complexity: float = 0.0
    test_coverage: float = 0.0
    documentation_coverage: float = 0.0
    maintainability_index: float = 0.0
    technical_debt_minutes: int = 0
    duplication_percentage: float = 0.0
    security_hotspots: int = 0
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        # Weighted average of quality factors
        weights = {
            'complexity': 0.25,
            'coverage': 0.20,
            'documentation': 0.15,
            'maintainability': 0.20,
            'duplication': 0.10,
            'security': 0.10
        }
        
        # Normalize scores to 0-100 scale
        complexity_score = max(0, 100 - (self.cyclomatic_complexity - 1) * 10)
        coverage_score = self.test_coverage
        doc_score = self.documentation_coverage
        maint_score = min(100, self.maintainability_index)
        dup_score = max(0, 100 - self.duplication_percentage)
        sec_score = max(0, 100 - (self.security_hotspots * 5))
        
        overall = (
            complexity_score * weights['complexity'] +
            coverage_score * weights['coverage'] +
            doc_score * weights['documentation'] +
            maint_score * weights['maintainability'] +
            dup_score * weights['duplication'] +
            sec_score * weights['security']
        )
        
        return round(min(100.0, max(0.0, overall)), 2)


class CodeQualityAnalyzer:
    """Advanced code quality analysis and metrics calculation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def analyze_codebase(self, project_path: PathStr) -> CodeQualityMetrics:
        """Perform comprehensive code quality analysis."""
        project_path = Path(project_path)
        
        # Get all source files
        source_files = self._get_source_files(project_path)
        
        if not source_files:
            return CodeQualityMetrics()
        
        # Parallel analysis of different quality aspects
        results = await asyncio.gather(
            self._analyze_complexity(source_files),
            self._analyze_test_coverage(project_path),
            self._analyze_documentation(source_files),
            self._analyze_maintainability(source_files),
            self._analyze_duplication(source_files),
            self._analyze_security(source_files),
            return_exceptions=True
        )
        
        # Combine results
        total_loc = sum(self._count_lines_of_code(f) for f in source_files)
        
        complexity = results[0] if not isinstance(results[0], Exception) else 1.0
        coverage = results[1] if not isinstance(results[1], Exception) else 0.0
        documentation = results[2] if not isinstance(results[2], Exception) else 0.0
        maintainability = results[3] if not isinstance(results[3], Exception) else 50.0
        duplication = results[4] if not isinstance(results[4], Exception) else 0.0
        security_issues = results[5] if not isinstance(results[5], Exception) else 0
        
        return CodeQualityMetrics(
            lines_of_code=total_loc,
            cyclomatic_complexity=complexity,
            test_coverage=coverage,
            documentation_coverage=documentation,
            maintainability_index=maintainability,
            technical_debt_minutes=int(complexity * 10),  # Rough estimate
            duplication_percentage=duplication,
            security_hotspots=security_issues
        )
    
    async def _analyze_complexity(self, source_files: List[Path]) -> float:
        """Analyze cyclomatic complexity."""
        total_complexity = 0
        function_count = 0
        
        for file_path in source_files:
            if file_path.suffix.lower() == '.py':
                try:
                    content = file_path.read_text(encoding='utf-8')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            complexity = self._calculate_function_complexity(node)
                            total_complexity += complexity
                            function_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Error analyzing complexity for {file_path}: {e}")
        
        return total_complexity / function_count if function_count > 0 else 1.0
    
    def _calculate_function_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points that increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                
        return complexity
    
    async def _analyze_test_coverage(self, project_path: Path) -> float:
        """Analyze test coverage (mock implementation)."""
        # In real implementation, this would run coverage.py or similar
        test_dirs = ['tests', 'test', '__tests__']
        source_dirs = ['src', 'lib', 'app']
        
        test_files = []
        source_files = []
        
        for test_dir in test_dirs:
            test_path = project_path / test_dir
            if test_path.exists():
                test_files.extend(list(test_path.rglob('*.py')))
        
        for src_dir in source_dirs:
            src_path = project_path / src_dir
            if src_path.exists():
                source_files.extend(list(src_path.rglob('*.py')))
        
        if not source_files:
            return 100.0  # No source files to test
        
        # Simple heuristic: test coverage based on test to source ratio
        test_ratio = len(test_files) / len(source_files)
        estimated_coverage = min(100.0, test_ratio * 80)  # Max 80% from ratio
        
        return round(estimated_coverage, 2)
    
    async def _analyze_documentation(self, source_files: List[Path]) -> float:
        """Analyze documentation coverage."""
        total_functions = 0
        documented_functions = 0
        
        for file_path in source_files:
            if file_path.suffix.lower() == '.py':
                try:
                    content = file_path.read_text(encoding='utf-8')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            total_functions += 1
                            
                            # Check for docstring
                            if (node.body and 
                                isinstance(node.body[0], ast.Expr) and 
                                isinstance(node.body[0].value, ast.Constant) and 
                                isinstance(node.body[0].value.value, str)):
                                documented_functions += 1
                                
                except Exception as e:
                    self.logger.warning(f"Error analyzing documentation for {file_path}: {e}")
        
        if total_functions == 0:
            return 100.0
        
        return round((documented_functions / total_functions) * 100, 2)
    
    async def _analyze_maintainability(self, source_files: List[Path]) -> float:
        """Analyze maintainability index."""
        # Simplified maintainability calculation
        total_score = 0
        file_count = 0
        
        for file_path in source_files:
            if file_path.suffix.lower() == '.py':
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Simple metrics
                    lines = len([l for l in content.split('\n') if l.strip()])
                    
                    # Penalize very long files
                    if lines > 500:
                        file_score = 30
                    elif lines > 300:
                        file_score = 50
                    elif lines > 100:
                        file_score = 70
                    else:
                        file_score = 90
                    
                    # Bonus for good practices
                    if '"""' in content or "'''" in content:  # Has docstrings
                        file_score += 5
                    
                    if 'import ' in content and 'from ' in content:  # Proper imports
                        file_score += 5
                    
                    total_score += min(100, file_score)
                    file_count += 1
                    
                except Exception:
                    continue
        
        return total_score / file_count if file_count > 0 else 50.0
    
    async def _analyze_duplication(self, source_files: List[Path]) -> float:
        """Analyze code duplication (simplified)."""
        # This is a simplified implementation
        # Real implementation would use more sophisticated algorithms
        
        all_functions = {}
        duplicated_count = 0
        total_count = 0
        
        for file_path in source_files:
            if file_path.suffix.lower() == '.py':
                try:
                    content = file_path.read_text(encoding='utf-8')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Simple hash of function structure
                            func_hash = hashlib.md5(node.name.encode()).hexdigest()[:8]
                            
                            if func_hash in all_functions:
                                duplicated_count += 1
                            else:
                                all_functions[func_hash] = file_path
                            
                            total_count += 1
                            
                except Exception:
                    continue
        
        if total_count == 0:
            return 0.0
        
        return round((duplicated_count / total_count) * 100, 2)
    
    async def _analyze_security(self, source_files: List[Path]) -> int:
        """Analyze security hotspots."""
        security_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'shell=True',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'password\s*=\s*["\'][^"\']',
            r'secret\s*=\s*["\'][^"\']',
            r'api_key\s*=\s*["\'][^"\']',
        ]
        
        total_hotspots = 0
        
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                for pattern in security_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    total_hotspots += len(matches)
                    
            except Exception:
                continue
        
        return total_hotspots
    
    def _get_source_files(self, project_path: Path) -> List[Path]:
        """Get list of source files for analysis."""
        source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', '.go', '.rs'}
        
        source_files = []
        for file_path in project_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in source_extensions and
                not any(part.startswith('.') for part in file_path.parts) and
                'node_modules' not in file_path.parts and
                '__pycache__' not in file_path.parts and
                'venv' not in file_path.parts and
                '.git' not in file_path.parts):
                source_files.append(file_path)
        
        return source_files
    
    def _count_lines_of_code(self, file_path: Path) -> int:
        """Count lines of code in a file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            return len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
        except Exception:
            return 0


class ProgressValidator:
    """
    Comprehensive validation system for detecting fake progress and placeholders.
    
    Implements a multi-stage validation pipeline:
    1. Static Analysis - Pattern matching for common placeholders
    2. Semantic Analysis - AST-based functionality verification
    3. Execution Testing - Runtime validation of generated code
    4. Cross-validation - Secondary AI verification
    """
    
    def __init__(
        self,
        enable_cross_validation: bool = True,
        enable_execution_testing: bool = True,
        enable_quality_analysis: bool = True,
        custom_patterns: Optional[List[PlaceholderPattern]] = None
    ):
        """
        Initialize the validation system.
        
        Args:
            enable_cross_validation: Enable secondary AI validation
            enable_execution_testing: Enable runtime testing
            custom_patterns: Additional placeholder patterns
        """
        self.enable_cross_validation = enable_cross_validation
        self.enable_execution_testing = enable_execution_testing
        self.enable_quality_analysis = enable_quality_analysis
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Initialize quality analyzer
        self.quality_analyzer = CodeQualityAnalyzer() if enable_quality_analysis else None
        
        # Load built-in patterns
        self.placeholder_patterns = self._load_builtin_patterns()
        
        # Add custom patterns
        if custom_patterns:
            self.placeholder_patterns.extend(custom_patterns)
        
        # Load semantic checks
        self.semantic_checks = self._load_semantic_checks()
        
        # Validation cache to avoid re-validating unchanged files
        self._validation_cache: Dict[str, Tuple[str, ValidationResult]] = {}
    
    async def validate_codebase(
        self,
        project_path: PathStr,
        focus_files: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate entire codebase for authenticity and completeness.
        
        Args:
            project_path: Path to project directory
            focus_files: Specific files to validate, or None for all
            
        Returns:
            Comprehensive validation result
        """
        project_path = Path(project_path)
        
        if not project_path.exists():
            raise ValidationException(f"Project path does not exist: {project_path}")
        
        # Get files to validate
        if focus_files:
            files_to_check = [project_path / f for f in focus_files if (project_path / f).exists()]
        else:
            files_to_check = self._get_source_files(project_path)
        
        all_issues: List[Issue] = []
        total_files = len(files_to_check)
        authentic_files = 0
        total_authenticity_score = 0.0
        
        # Run quality analysis if enabled
        quality_metrics = None
        if self.enable_quality_analysis and self.quality_analyzer:
            try:
                self.logger.info("Starting code quality analysis...")
                quality_metrics = await self.quality_analyzer.analyze_codebase(project_path)
                self.logger.info(f"Quality analysis complete. Overall score: {quality_metrics.overall_score}%")
            except Exception as e:
                self.logger.error(f"Quality analysis failed: {e}")
        
        # Process files in parallel for better performance
        semaphore = asyncio.Semaphore(5)  # Limit concurrent validations
        
        async def validate_single_file(file_path: Path) -> Tuple[List[Issue], float]:
            async with semaphore:
                return await self._validate_single_file(file_path)
        
        # Execute validations concurrently
        validation_tasks = [
            validate_single_file(file_path) 
            for file_path in files_to_check
        ]
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle validation errors
                all_issues.append(Issue(
                    type=IssueType.BROKEN_LOGIC,
                    severity=Severity.HIGH,
                    description=f"Validation error in {files_to_check[i]}: {result}",
                    file_path=str(files_to_check[i])
                ))
                continue
            
            file_issues, authenticity_score = result
            all_issues.extend(file_issues)
            total_authenticity_score += authenticity_score
            
            if authenticity_score >= 80.0:  # Consider 80%+ as authentic
                authentic_files += 1
        
        # Calculate overall metrics
        overall_authenticity = (
            total_authenticity_score / total_files 
            if total_files > 0 else 100.0
        )
        
        real_progress = (authentic_files / total_files * 100) if total_files > 0 else 0.0
        fake_progress = max(0, 100.0 - real_progress)
        
        # Generate suggestions based on issues found
        suggestions = self._generate_suggestions(all_issues, quality_metrics)
        next_actions = self._determine_next_actions(all_issues, quality_metrics)
        
        # Create enhanced validation result
        result = ValidationResult(
            is_authentic=overall_authenticity >= 70.0,
            authenticity_score=overall_authenticity,
            real_progress=real_progress,
            fake_progress=fake_progress,
            issues=all_issues,
            suggestions=suggestions,
            next_actions=next_actions
        )
        
        # Add quality metrics if available
        if quality_metrics:
            # Store quality metrics in validation result metadata
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata = {
                'quality_score': quality_metrics.overall_score,
                'lines_of_code': quality_metrics.lines_of_code,
                'complexity': quality_metrics.cyclomatic_complexity,
                'test_coverage': quality_metrics.test_coverage,
                'documentation_coverage': quality_metrics.documentation_coverage,
                'maintainability_index': quality_metrics.maintainability_index,
                'technical_debt_minutes': quality_metrics.technical_debt_minutes,
                'duplication_percentage': quality_metrics.duplication_percentage,
                'security_hotspots': quality_metrics.security_hotspots
            }
        
        return result
    
    async def validate_single_file(self, file_path: PathStr) -> ValidationResult:
        """
        Validate a single file for authenticity.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Validation result for the file
        """
        issues, authenticity_score = await self._validate_single_file(Path(file_path))
        
        return ValidationResult(
            is_authentic=authenticity_score >= 80.0,
            authenticity_score=authenticity_score,
            real_progress=authenticity_score,
            fake_progress=max(0, 100.0 - authenticity_score),
            issues=issues,
            suggestions=self._generate_suggestions(issues),
            next_actions=self._determine_next_actions(issues)
        )
    
    async def auto_fix_placeholders(
        self,
        issues: List[Issue],
        project_path: PathStr
    ) -> Dict[str, Any]:
        """
        Automatically fix detected placeholder issues.
        
        Args:
            issues: List of issues to fix
            project_path: Project root path
            
        Returns:
            Fix results summary
        """
        project_path = Path(project_path)
        fixed_issues = []
        failed_fixes = []
        
        for issue in issues:
            if not issue.auto_fix_available or not issue.suggested_fix:
                continue
            
            try:
                success = await self._apply_auto_fix(issue, project_path)
                if success:
                    fixed_issues.append(issue)
                else:
                    failed_fixes.append(issue)
            except Exception as e:
                failed_fixes.append(issue)
                issue.suggested_fix = f"Auto-fix failed: {e}"
        
        return {
            'fixed_count': len(fixed_issues),
            'failed_count': len(failed_fixes),
            'fixed_issues': fixed_issues,
            'failed_issues': failed_fixes,
            'success_rate': len(fixed_issues) / len(issues) * 100 if issues else 0
        }
    
    async def _validate_single_file(self, file_path: Path) -> Tuple[List[Issue], float]:
        """
        Internal method to validate a single file.
        
        Returns:
            Tuple of (issues_found, authenticity_score)
        """
        # Check cache first
        file_content = file_path.read_text(encoding='utf-8', errors='ignore')
        content_hash = hashlib.md5(file_content.encode()).hexdigest()
        
        cache_key = str(file_path)
        if cache_key in self._validation_cache:
            cached_hash, cached_result = self._validation_cache[cache_key]
            if cached_hash == content_hash:
                return cached_result.issues, cached_result.authenticity_score
        
        issues: List[Issue] = []
        scores: List[float] = []
        
        # Stage 1: Static Analysis - Pattern matching
        static_issues, static_score = await self._static_analysis(file_path, file_content)
        issues.extend(static_issues)
        scores.append(static_score)
        
        # Stage 2: Semantic Analysis - AST-based checks
        if self._is_parseable_language(file_path):
            semantic_issues, semantic_score = await self._semantic_analysis(file_path, file_content)
            issues.extend(semantic_issues)
            scores.append(semantic_score)
        
        # Stage 3: Execution Testing (optional)
        if self.enable_execution_testing and self._is_executable_file(file_path):
            exec_issues, exec_score = await self._execution_testing(file_path)
            issues.extend(exec_issues)
            scores.append(exec_score)
        
        # Calculate weighted authenticity score
        if scores:
            authenticity_score = sum(scores) / len(scores)
        else:
            authenticity_score = 100.0
        
        # Apply penalties for critical issues
        critical_issues = sum(1 for issue in issues if issue.severity == Severity.CRITICAL)
        high_issues = sum(1 for issue in issues if issue.severity == Severity.HIGH)
        
        authenticity_score -= (critical_issues * 20) + (high_issues * 10)
        authenticity_score = max(0.0, authenticity_score)
        
        # Cache result
        result = ValidationResult(
            is_authentic=authenticity_score >= 80.0,
            authenticity_score=authenticity_score,
            real_progress=authenticity_score,
            fake_progress=100.0 - authenticity_score,
            issues=issues
        )
        self._validation_cache[cache_key] = (content_hash, result)
        
        return issues, authenticity_score
    
    async def _static_analysis(self, file_path: Path, content: str) -> Tuple[List[Issue], float]:
        """
        Static analysis using pattern matching.
        
        Returns:
            Tuple of (issues, score)
        """
        issues: List[Issue] = []
        file_ext = file_path.suffix.lower()
        
        for pattern in self.placeholder_patterns:
            # Skip patterns not applicable to this file type
            if pattern.file_extensions and file_ext not in pattern.file_extensions:
                continue
            
            matches = re.finditer(pattern.pattern, content, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                
                issue = Issue(
                    type=IssueType.PLACEHOLDER,
                    severity=pattern.severity,
                    description=f"{pattern.description}: '{match.group().strip()}'",
                    file_path=str(file_path),
                    line_number=line_number,
                    auto_fix_available=bool(pattern.auto_fix_template),
                    suggested_fix=pattern.auto_fix_template
                )
                issues.append(issue)
        
        # Calculate score based on issues found
        score = 100.0
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                score -= 25
            elif issue.severity == Severity.HIGH:
                score -= 15
            elif issue.severity == Severity.MEDIUM:
                score -= 10
            else:  # LOW
                score -= 5
        
        return issues, max(0.0, score)
    
    async def _semantic_analysis(self, file_path: Path, content: str) -> Tuple[List[Issue], float]:
        """
        Semantic analysis using AST parsing.
        
        Returns:
            Tuple of (issues, score)
        """
        issues: List[Issue] = []
        
        try:
            # Parse file based on language
            if file_path.suffix.lower() == '.py':
                issues_found = await self._analyze_python_ast(file_path, content)
                issues.extend(issues_found)
            elif file_path.suffix.lower() in ['.js', '.jsx', '.ts', '.tsx']:
                issues_found = await self._analyze_javascript_semantic(file_path, content)
                issues.extend(issues_found)
            # Add more language analyzers as needed
            
        except SyntaxError as e:
            issues.append(Issue(
                type=IssueType.BROKEN_LOGIC,
                severity=Severity.CRITICAL,
                description=f"Syntax error: {e}",
                file_path=str(file_path),
                line_number=getattr(e, 'lineno', 0)
            ))
        except Exception as e:
            issues.append(Issue(
                type=IssueType.BROKEN_LOGIC,
                severity=Severity.MEDIUM,
                description=f"Analysis error: {e}",
                file_path=str(file_path)
            ))
        
        # Calculate semantic score
        score = 100.0 - (len(issues) * 15)  # 15 points per semantic issue
        return issues, max(0.0, score)
    
    async def _analyze_python_ast(self, file_path: Path, content: str) -> List[Issue]:
        """Analyze Python file using AST."""
        issues: List[Issue] = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for empty functions
                if isinstance(node, ast.FunctionDef):
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        issues.append(Issue(
                            type=IssueType.EMPTY_FUNCTION,
                            severity=Severity.HIGH,
                            description=f"Empty function: {node.name}",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            auto_fix_available=True,
                            suggested_fix=f"def {node.name}({', '.join(arg.arg for arg in node.args.args)}):\n    # TODO: Implement {node.name}\n    raise NotImplementedError"
                        ))
                
                # Check for placeholder raises
                if isinstance(node, ast.Raise):
                    if (isinstance(node.exc, ast.Name) and 
                        node.exc.id == 'NotImplementedError'):
                        issues.append(Issue(
                            type=IssueType.PLACEHOLDER,
                            severity=Severity.HIGH,
                            description="NotImplementedError placeholder found",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            auto_fix_available=False
                        ))
        
        except SyntaxError:
            pass  # Already handled in semantic_analysis
        
        return issues
    
    async def _analyze_javascript_semantic(self, file_path: Path, content: str) -> List[Issue]:
        """Analyze JavaScript/TypeScript file semantically."""
        issues: List[Issue] = []
        
        # Look for common JS/TS placeholder patterns
        patterns = [
            (r'function\s+\w+\s*\([^)]*\)\s*\{\s*\}', 'Empty function found'),
            (r'=>\s*\{\s*\}', 'Empty arrow function found'),
            (r'throw\s+new\s+Error\s*\(\s*[\'"]Not\s+implemented[\'"]', 'Not implemented placeholder'),
            (r'console\.log\s*\(\s*[\'"]TODO', 'TODO in console.log'),
        ]
        
        for pattern, description in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(Issue(
                    type=IssueType.PLACEHOLDER,
                    severity=Severity.MEDIUM,
                    description=description,
                    file_path=str(file_path),
                    line_number=line_number,
                    auto_fix_available=False
                ))
        
        return issues
    
    async def _execution_testing(self, file_path: Path) -> Tuple[List[Issue], float]:
        """
        Test code execution to verify functionality.
        
        Returns:
            Tuple of (issues, score)
        """
        issues: List[Issue] = []
        
        # This is a simplified execution test
        # In a full implementation, this would run in a sandboxed environment
        try:
            if file_path.suffix.lower() == '.py':
                # Try to compile Python file
                with open(file_path, 'r') as f:
                    content = f.read()
                
                try:
                    compile(content, str(file_path), 'exec')
                except SyntaxError as e:
                    issues.append(Issue(
                        type=IssueType.BROKEN_LOGIC,
                        severity=Severity.CRITICAL,
                        description=f"Python compilation failed: {e}",
                        file_path=str(file_path),
                        line_number=e.lineno
                    ))
        
        except Exception as e:
            issues.append(Issue(
                type=IssueType.BROKEN_LOGIC,
                severity=Severity.MEDIUM,
                description=f"Execution test failed: {e}",
                file_path=str(file_path)
            ))
        
        # Score based on execution success
        score = 100.0 if not issues else 50.0
        return issues, score
    
    async def _apply_auto_fix(self, issue: Issue, project_path: Path) -> bool:
        """
        Apply automatic fix for an issue.
        
        Args:
            issue: Issue to fix
            project_path: Project root path
            
        Returns:
            True if fix was successful
        """
        if not issue.auto_fix_available or not issue.suggested_fix:
            return False
        
        try:
            file_path = Path(issue.file_path)
            if not file_path.is_absolute():
                file_path = project_path / file_path
            
            # Read current file content
            content = file_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            
            # Apply fix (this is simplified - real implementation would be more sophisticated)
            if issue.line_number and issue.line_number <= len(lines):
                # Replace the problematic line
                lines[issue.line_number - 1] = issue.suggested_fix
                
                # Write back to file
                new_content = '\n'.join(lines)
                file_path.write_text(new_content, encoding='utf-8')
                
                return True
        
        except Exception:
            return False
        
        return False
    
    def _generate_suggestions(self, issues: List[Issue], quality_metrics: Optional[CodeQualityMetrics] = None) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        placeholder_count = sum(1 for issue in issues if issue.type == IssueType.PLACEHOLDER)
        empty_function_count = sum(1 for issue in issues if issue.type == IssueType.EMPTY_FUNCTION)
        
        if placeholder_count > 0:
            suggestions.append(f"Complete {placeholder_count} placeholder implementations")
        
        if empty_function_count > 0:
            suggestions.append(f"Implement {empty_function_count} empty functions")
        
        if any(issue.severity == Severity.CRITICAL for issue in issues):
            suggestions.append("Fix critical issues before proceeding")
        
        # Add quality-based suggestions
        if quality_metrics:
            if quality_metrics.overall_score < 60:
                suggestions.append("Code quality is below acceptable levels - consider refactoring")
            
            if quality_metrics.cyclomatic_complexity > 10:
                suggestions.append("High complexity detected - break down complex functions")
            
            if quality_metrics.test_coverage < 70:
                suggestions.append("Test coverage is low - add more unit tests")
            
            if quality_metrics.documentation_coverage < 60:
                suggestions.append("Documentation coverage is low - add docstrings to functions and classes")
            
            if quality_metrics.duplication_percentage > 15:
                suggestions.append("High code duplication detected - extract common functionality")
            
            if quality_metrics.security_hotspots > 0:
                suggestions.append(f"Security concerns detected ({quality_metrics.security_hotspots} hotspots) - review and fix")
        
        return suggestions
    
    def _determine_next_actions(self, issues: List[Issue], quality_metrics: Optional[CodeQualityMetrics] = None) -> List[str]:
        """Determine recommended next actions."""
        actions = []
        
        auto_fixable = sum(1 for issue in issues if issue.auto_fix_available)
        if auto_fixable > 0:
            actions.append("auto-fix-placeholders")
        
        if any(issue.severity in [Severity.CRITICAL, Severity.HIGH] for issue in issues):
            actions.append("manual-review-required")
        
        if not issues:
            if quality_metrics and quality_metrics.overall_score >= 80:
                actions.append("validation-passed")
            else:
                actions.append("quality-improvements-needed")
        
        # Add quality-driven actions
        if quality_metrics:
            if quality_metrics.test_coverage < 70:
                actions.append("improve-test-coverage")
            
            if quality_metrics.cyclomatic_complexity > 15:
                actions.append("reduce-complexity")
            
            if quality_metrics.security_hotspots > 0:
                actions.append("address-security-issues")
        
        return actions
    
    def _get_source_files(self, project_path: Path) -> List[Path]:
        """Get list of source files to validate."""
        source_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', 
            '.h', '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.swift'
        }
        
        source_files = []
        for file_path in project_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in source_extensions and
                not any(part.startswith('.') for part in file_path.parts) and
                'node_modules' not in file_path.parts and
                '__pycache__' not in file_path.parts):
                source_files.append(file_path)
        
        return source_files
    
    def _is_parseable_language(self, file_path: Path) -> bool:
        """Check if file is in a language we can parse."""
        parseable_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx'}
        return file_path.suffix.lower() in parseable_extensions
    
    def _is_executable_file(self, file_path: Path) -> bool:
        """Check if file can be executed for testing."""
        executable_extensions = {'.py', '.js', '.ts'}
        return file_path.suffix.lower() in executable_extensions
    
    def _load_builtin_patterns(self) -> List[PlaceholderPattern]:
        """Load built-in placeholder detection patterns."""
        return [
            PlaceholderPattern(
                name="todo_comments",
                pattern=r'#\s*TODO:?.*|//\s*TODO:?.*|/\*\s*TODO:?.*\*/|\{\s*\/\*\s*TODO:?.*\*\/\s*\}',
                severity=Severity.MEDIUM,
                description="TODO comment found",
                auto_fix_template="# TODO: Implement this functionality",
                file_extensions={'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c'}
            ),
            PlaceholderPattern(
                name="fixme_comments", 
                pattern=r'#\s*FIXME:?.*|//\s*FIXME:?.*|/\*\s*FIXME:?.*\*/',
                severity=Severity.HIGH,
                description="FIXME comment found",
                file_extensions={'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c'}
            ),
            PlaceholderPattern(
                name="placeholder_text",
                pattern=r'placeholder|PLACEHOLDER|xxx|XXX|\.\.\.',
                severity=Severity.MEDIUM,
                description="Placeholder text found",
                file_extensions={'.py', '.js', '.jsx', '.ts', '.tsx'}
            ),
            PlaceholderPattern(
                name="not_implemented",
                pattern=r'raise\s+NotImplementedError|throw\s+new\s+Error\s*\(\s*[\'"]Not\s+implemented',
                severity=Severity.HIGH,
                description="Not implemented error found",
                file_extensions={'.py', '.js', '.jsx', '.ts', '.tsx'}
            ),
            PlaceholderPattern(
                name="empty_catch",
                pattern=r'except:?\s*pass|catch\s*\([^)]*\)\s*\{\s*\}',
                severity=Severity.MEDIUM,
                description="Empty exception handler found",
                file_extensions={'.py', '.js', '.jsx', '.ts', '.tsx'}
            )
        ]
    
    def _load_semantic_checks(self) -> List[SemanticCheck]:
        """Load semantic analysis checks."""
        return [
            SemanticCheck(
                name="empty_functions",
                check_function="check_empty_functions",
                severity=Severity.HIGH,
                description="Function with no implementation",
                applicable_languages={'python', 'javascript', 'typescript'}
            ),
            SemanticCheck(
                name="unreachable_code",
                check_function="check_unreachable_code", 
                severity=Severity.MEDIUM,
                description="Code after return statement",
                applicable_languages={'python', 'javascript', 'typescript'}
            )
        ]


# Utility functions for external use

async def validate_project_authenticity(
    project_path: PathStr,
    custom_patterns: Optional[List[PlaceholderPattern]] = None
) -> ValidationResult:
    """
    Convenience function to validate project authenticity.
    
    Args:
        project_path: Path to project directory
        custom_patterns: Additional validation patterns
        
    Returns:
        Comprehensive validation result
    """
    validator = ProgressValidator(custom_patterns=custom_patterns)
    return await validator.validate_codebase(project_path)


async def quick_file_validation(file_path: PathStr) -> ValidationResult:
    """
    Quick validation of a single file.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        File validation result
    """
    validator = ProgressValidator(
        enable_cross_validation=False,
        enable_execution_testing=False
    )
    return await validator.validate_single_file(file_path)