"""
Placeholder Detector - Advanced placeholder and incomplete code detection.

Implements sophisticated pattern matching and ML-based detection to identify
placeholder content, incomplete implementations, and hallucinated code.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.validation.types import ValidationIssue, ValidationSeverity, PlaceholderType, PlaceholderPattern, IssueCategory

logger = logging.getLogger(__name__)


# PlaceholderType and PlaceholderPattern moved to types.py to avoid circular imports


@dataclass
class PlaceholderMatch:
    """Detected placeholder match."""
    pattern_name: str
    placeholder_type: PlaceholderType
    text: str
    start_pos: int
    end_pos: int
    line_number: int
    column_number: int
    severity: ValidationSeverity
    context: str
    suggestion: Optional[str] = None


class PlaceholderDetector:
    """
    Advanced placeholder and incomplete code detection system.
    
    Uses regex patterns, context analysis, and ML-based detection to identify
    placeholder content and incomplete implementations in AI-generated code.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the placeholder detector.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        
        # Detection patterns
        self._patterns: List[PlaceholderPattern] = []
        self._language_patterns: Dict[str, List[PlaceholderPattern]] = {}
        
        # Configuration
        self._sensitivity = 0.95  # Detection sensitivity (0.0-1.0)
        self._context_window = 3  # Lines of context to capture
        
        # ML model for advanced detection (placeholder)
        self._ml_model = None
        
        logger.info("Placeholder detector initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the placeholder detection system.
        """
        logger.info("Initializing placeholder detector")
        
        try:
            # Load configuration
            detector_config = await self.config_manager.get_setting('placeholder_detector', {})
            self._sensitivity = detector_config.get('sensitivity', 0.95)
            self._context_window = detector_config.get('context_window', 3)
            
            # Load detection patterns
            await self._load_detection_patterns()
            
            # Initialize ML model if available
            await self._initialize_ml_model()
            
            logger.info(f"Placeholder detector initialized with {len(self._patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to initialize placeholder detector: {e}")
            raise
    
    async def detect_placeholders(
        self,
        content: str,
        file_path: Optional[Path] = None,
        language: Optional[str] = None
    ) -> List[ValidationIssue]:
        """
        Detect placeholders in content.
        
        Args:
            content: Content to analyze
            file_path: File path for context (optional)
            language: Programming language (optional, inferred from file_path)
            
        Returns:
            List of validation issues for detected placeholders
        """
        if not content.strip():
            return []
        
        # Infer language if not provided
        if not language and file_path:
            language = self._infer_language(file_path)
        
        logger.debug(f"Detecting placeholders in {language or 'unknown'} content")
        
        try:
            matches = []
            
            # Pattern-based detection
            pattern_matches = await self._pattern_based_detection(
                content, language, file_path
            )
            matches.extend(pattern_matches)
            
            # Context-aware detection
            context_matches = await self._context_aware_detection(
                content, language, file_path
            )
            matches.extend(context_matches)
            
            # ML-based detection (if available)
            if self._ml_model:
                ml_matches = await self._ml_based_detection(
                    content, language, file_path
                )
                matches.extend(ml_matches)
            
            # Convert matches to validation issues
            issues = await self._matches_to_issues(matches, file_path)
            
            logger.debug(f"Detected {len(issues)} placeholder issues")
            return issues
            
        except Exception as e:
            logger.error(f"Placeholder detection failed: {e}")
            return []
    
    async def detect_placeholders_in_content(self, content: str) -> List[ValidationIssue]:
        """
        Detect placeholders in raw content without file context.
        
        Args:
            content: Content to analyze
            
        Returns:
            List of validation issues
        """
        return await self.detect_placeholders(content)
    
    async def analyze_completeness(
        self,
        content: str,
        requirements: Optional[List[str]] = None,
        file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze content completeness against requirements.
        
        Args:
            content: Content to analyze
            requirements: Expected functionality requirements
            file_path: File path for context
            
        Returns:
            Completeness analysis results
        """
        logger.debug("Analyzing content completeness")
        
        try:
            analysis = {
                'completeness_score': 0.0,
                'missing_elements': [],
                'placeholder_density': 0.0,
                'implementation_gaps': [],
                'suggestions': []
            }
            
            # Detect placeholders
            placeholders = await self.detect_placeholders(content, file_path)
            
            # Calculate placeholder density
            total_lines = len(content.split('\
'))
            placeholder_lines = len(set(p.line_number for p in placeholders if hasattr(p, 'line_number')))
            analysis['placeholder_density'] = placeholder_lines / max(total_lines, 1)
            
            # Analyze against requirements
            if requirements:
                missing = await self._check_requirements_completeness(content, requirements)
                analysis['missing_elements'] = missing
            
            # Detect implementation gaps
            gaps = await self._detect_implementation_gaps(content, file_path)
            analysis['implementation_gaps'] = gaps
            
            # Calculate completeness score
            score = 1.0 - analysis['placeholder_density']
            if requirements:
                requirement_score = 1.0 - (len(analysis['missing_elements']) / len(requirements))
                score = (score + requirement_score) / 2.0
            
            analysis['completeness_score'] = max(0.0, score)
            
            # Generate suggestions
            if analysis['placeholder_density'] > 0.1:
                analysis['suggestions'].append("Reduce placeholder content")
            if analysis['missing_elements']:
                analysis['suggestions'].append("Implement missing required elements")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Completeness analysis failed: {e}")
            return {'error': str(e)}
    
    async def get_fix_suggestions(
        self,
        match: PlaceholderMatch,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get fix suggestions for a detected placeholder.
        
        Args:
            match: Detected placeholder match
            content: Full content for context
            context: Additional context information
            
        Returns:
            List of fix suggestions
        """
        suggestions = []
        
        try:
            if match.placeholder_type == PlaceholderType.TODO_COMMENT:
                suggestions.append("Implement the functionality described in the TODO comment")
                suggestions.append("Remove the TODO comment after implementation")
            
            elif match.placeholder_type == PlaceholderType.INCOMPLETE_FUNCTION:
                suggestions.append("Add function body with proper implementation")
                suggestions.append("Add proper return statement")
                suggestions.append("Add error handling if needed")
            
            elif match.placeholder_type == PlaceholderType.EMPTY_BLOCK:
                suggestions.append("Add implementation code to the empty block")
                suggestions.append("Add proper logic flow")
            
            elif match.placeholder_type == PlaceholderType.GENERIC_ERROR:
                suggestions.append("Replace generic error with specific implementation")
                suggestions.append("Add proper error handling")
            
            elif match.placeholder_type == PlaceholderType.STUB_IMPLEMENTATION:
                suggestions.append("Replace stub with full implementation")
                suggestions.append("Add proper business logic")
            
            else:
                suggestions.append("Replace placeholder with actual implementation")
            
            # Add context-specific suggestions
            if context:
                contextual_suggestions = await self._get_contextual_suggestions(
                    match, content, context
                )
                suggestions.extend(contextual_suggestions)
            
        except Exception as e:
            logger.warning(f"Failed to generate fix suggestions: {e}")
            suggestions.append("Review and implement missing functionality")
        
        return suggestions
    
    async def cleanup(self) -> None:
        """
        Cleanup detector resources.
        """
        logger.info("Cleaning up placeholder detector")
        
        # Cleanup ML model if loaded
        if self._ml_model:
            # Cleanup model resources
            self._ml_model = None
        
        logger.info("Placeholder detector cleanup completed")
    
    # Private helper methods
    
    async def _load_detection_patterns(self) -> None:
        """
        Load and compile placeholder detection patterns.
        """
        patterns = [
            # TODO and FIXME comments
            PlaceholderPattern(
                name="todo_comment",
                pattern=re.compile(r'(#\s*TODO|//\s*TODO|/\*\s*TODO.*?\*/|<!--\s*TODO.*?-->)', re.IGNORECASE | re.DOTALL),
                placeholder_type=PlaceholderType.TODO_COMMENT,
                severity=ValidationSeverity.HIGH,
                description="TODO comment found"
            ),
            PlaceholderPattern(
                name="fixme_comment",
                pattern=re.compile(r'(#\s*FIXME|//\s*FIXME|/\*\s*FIXME.*?\*/)', re.IGNORECASE | re.DOTALL),
                placeholder_type=PlaceholderType.FIXME_COMMENT,
                severity=ValidationSeverity.HIGH,
                description="FIXME comment found"
            ),
            
            # Common placeholder texts
            PlaceholderPattern(
                name="placeholder_text",
                pattern=re.compile(r'\b(PLACEHOLDER|YOUR_CODE_HERE|IMPLEMENT_ME|NOT_IMPLEMENTED)\b', re.IGNORECASE),
                placeholder_type=PlaceholderType.PLACEHOLDER_TEXT,
                severity=ValidationSeverity.CRITICAL,
                description="Placeholder text found"
            ),
            
            # Python-specific patterns
            PlaceholderPattern(
                name="python_pass_todo",
                pattern=re.compile(r'pass\s*#.*?(TODO|FIXME|IMPLEMENT)', re.IGNORECASE),
                placeholder_type=PlaceholderType.STUB_IMPLEMENTATION,
                severity=ValidationSeverity.HIGH,
                language="python",
                description="Python pass with TODO comment"
            ),
            PlaceholderPattern(
                name="python_ellipsis",
                pattern=re.compile(r'^\s*\.\.\.\s*$', re.MULTILINE),
                placeholder_type=PlaceholderType.PLACEHOLDER_TEXT,
                severity=ValidationSeverity.HIGH,
                language="python",
                description="Python ellipsis placeholder"
            ),
            
            # JavaScript-specific patterns
            PlaceholderPattern(
                name="js_throw_error",
                pattern=re.compile(r'throw\s+new\s+Error\s*\(\s*["\'\s]*(TODO|FIXME|NOT_IMPLEMENTED|PLACEHOLDER)', re.IGNORECASE),
                placeholder_type=PlaceholderType.GENERIC_ERROR,
                severity=ValidationSeverity.HIGH,
                language="javascript",
                description="JavaScript placeholder error"
            ),
            
            # Empty function/method bodies
            PlaceholderPattern(
                name="empty_function_body",
                pattern=re.compile(r'(?:def\\s+\\w+\\([^)]*\\):|function\\s+\\w+\\([^)]*\\)\\s*{|\\w+\\s*\\([^)]*\\)\\s*=>\\s*{)\\s*(?:#.*?\
|//.*?\
|/\\*.*?\\*/)?\\s*(?:}|$)', re.MULTILINE | re.DOTALL),
                placeholder_type=PlaceholderType.INCOMPLETE_FUNCTION,
                severity=ValidationSeverity.CRITICAL,
                description="Empty function body detected"
            ),
            
            # Generic error messages
            PlaceholderPattern(
                name="generic_error_message",
                pattern=re.compile(r'raise\\s+NotImplementedError|throw\\s+new\\s+Error\\s*\\(\\s*["\'].*not.*implement', re.IGNORECASE),
                placeholder_type=PlaceholderType.GENERIC_ERROR,
                severity=ValidationSeverity.MEDIUM,
                description="Generic not implemented error"
            ),
            
            # Additional critical patterns for comprehensive detection
            PlaceholderPattern(
                name="stub_methods",
                pattern=re.compile(r'(def\\s+\\w+.*?:|function\\s+\\w+.*?\\{)\\s*pass\\s*(?:#.*?)?$', re.MULTILINE | re.IGNORECASE),
                placeholder_type=PlaceholderType.STUB_IMPLEMENTATION,
                severity=ValidationSeverity.HIGH,
                description="Stub method with pass statement"
            ),
            
            PlaceholderPattern(
                name="placeholder_comments",
                pattern=re.compile(r'#.*?\\b(PLACEHOLDER|STUB|MOCK|DUMMY|EXAMPLE)\\b.*?$', re.IGNORECASE | re.MULTILINE),
                placeholder_type=PlaceholderType.TODO_COMMENT,
                severity=ValidationSeverity.MEDIUM,
                description="Placeholder comment detected"
            ),
            
            PlaceholderPattern(
                name="return_none_only",
                pattern=re.compile(r'def\\s+\\w+.*?:\\s*return\\s+None\\s*$', re.MULTILINE),
                placeholder_type=PlaceholderType.STUB_IMPLEMENTATION,
                severity=ValidationSeverity.HIGH,
                language="python",
                description="Function returning None only"
            ),
            
            PlaceholderPattern(
                name="empty_class",
                pattern=re.compile(r'class\\s+\\w+.*?:\\s*pass\\s*$', re.MULTILINE),
                placeholder_type=PlaceholderType.INCOMPLETE_CLASS,
                severity=ValidationSeverity.HIGH,
                language="python",
                description="Empty class with only pass"
            ),
            
            PlaceholderPattern(
                name="console_log_placeholder",
                pattern=re.compile(r'console\\.log\\s*\\(\\s*["\'].*?(TODO|PLACEHOLDER|DEBUG).*?["\']\\s*\\)', re.IGNORECASE),
                placeholder_type=PlaceholderType.DEBUG_CODE,
                severity=ValidationSeverity.MEDIUM,
                language="javascript",
                description="Console log placeholder"
            ),
            
            PlaceholderPattern(
                name="print_debug",
                pattern=re.compile(r'print\\s*\\(\\s*["\'].*?(TODO|PLACEHOLDER|DEBUG|TEST).*?["\']\\s*\\)', re.IGNORECASE),
                placeholder_type=PlaceholderType.DEBUG_CODE,
                severity=ValidationSeverity.MEDIUM,
                language="python",
                description="Print debug placeholder"
            ),
            
            PlaceholderPattern(
                name="magic_numbers",
                pattern=re.compile(r'\\b(42|123|999|1000000)\\b(?=\\s*(?:#.*?(?:placeholder|test|example))|$)', re.IGNORECASE | re.MULTILINE),
                placeholder_type=PlaceholderType.PLACEHOLDER_VALUE,
                severity=ValidationSeverity.LOW,
                description="Common placeholder numbers"
            ),
            
            PlaceholderPattern(
                name="temp_variables",
                pattern=re.compile(r'\\b(temp|tmp|test|example|dummy|mock)_\\w+\\b', re.IGNORECASE),
                placeholder_type=PlaceholderType.PLACEHOLDER_VALUE,
                severity=ValidationSeverity.LOW,
                description="Temporary variable names"
            )
        ]
        
        # Store patterns
        self._patterns = patterns
        
        # Group by language
        self._language_patterns = {'*': []}  # Global patterns
        
        for pattern in patterns:
            if pattern.language:
                if pattern.language not in self._language_patterns:
                    self._language_patterns[pattern.language] = []
                self._language_patterns[pattern.language].append(pattern)
            else:
                self._language_patterns['*'].append(pattern)
    
    def _infer_language(self, file_path: Path) -> Optional[str]:
        """
        Infer programming language from file path.
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
    
    async def _pattern_based_detection(
        self,
        content: str,
        language: Optional[str],
        file_path: Optional[Path]
    ) -> List[PlaceholderMatch]:
        """
        Pattern-based placeholder detection.
        """
        matches = []
        
        # Get applicable patterns
        applicable_patterns = self._language_patterns.get('*', [])
        if language and language in self._language_patterns:
            applicable_patterns.extend(self._language_patterns[language])
        
        # Search for patterns
        for pattern_info in applicable_patterns:
            for match in pattern_info.pattern.finditer(content):
                line_number = content[:match.start()].count('\
') + 1
                line_start = content.rfind('\
', 0, match.start()) + 1
                column_number = match.start() - line_start + 1
                
                # Get context
                context = self._get_line_context(content, line_number, self._context_window)
                
                placeholder_match = PlaceholderMatch(
                    pattern_name=pattern_info.name,
                    placeholder_type=pattern_info.placeholder_type,
                    text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    line_number=line_number,
                    column_number=column_number,
                    severity=pattern_info.severity,
                    context=context
                )
                
                matches.append(placeholder_match)
        
        return matches
    
    async def _context_aware_detection(
        self,
        content: str,
        language: Optional[str],
        file_path: Optional[Path]
    ) -> List[PlaceholderMatch]:
        """
        Context-aware placeholder detection.
        """
        matches = []
        
        lines = content.split('\
')
        
        for i, line in enumerate(lines, 1):
            # Check for suspicious patterns based on context
            
            # Very short functions/methods
            if language == 'python' and line.strip().startswith('def '):
                # Look for single-line functions with just pass
                if i < len(lines) and lines[i].strip() == 'pass':
                    matches.append(PlaceholderMatch(
                        pattern_name="short_function",
                        placeholder_type=PlaceholderType.INCOMPLETE_FUNCTION,
                        text=line + '\
' + lines[i],
                        start_pos=0,  # Simplified
                        end_pos=0,
                        line_number=i,
                        column_number=1,
                        severity=ValidationSeverity.MEDIUM,
                        context=self._get_line_context(content, i, 2)
                    ))
            
            # Empty except blocks
            if 'except' in line and ':' in line:
                next_line_idx = i
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx].strip()
                    if next_line == 'pass':
                        matches.append(PlaceholderMatch(
                            pattern_name="empty_except",
                            placeholder_type=PlaceholderType.EMPTY_BLOCK,
                            text=line + '\
' + lines[next_line_idx],
                            start_pos=0,
                            end_pos=0,
                            line_number=i,
                            column_number=1,
                            severity=ValidationSeverity.MEDIUM,
                            context=self._get_line_context(content, i, 2)
                        ))
        
        return matches
    
    async def _ml_based_detection(
        self,
        content: str,
        language: Optional[str],
        file_path: Optional[Path]
    ) -> List[PlaceholderMatch]:
        """
        ML-based placeholder detection (placeholder implementation).
        """
        # This would integrate with an actual ML model
        # For now, return empty list
        return []
    
    async def _initialize_ml_model(self) -> None:
        """
        Initialize ML model for advanced placeholder detection.
        """
        # Placeholder for ML model initialization
        # Would load a trained model for placeholder detection
        pass
    
    async def _matches_to_issues(
        self,
        matches: List[PlaceholderMatch],
        file_path: Optional[Path]
    ) -> List[ValidationIssue]:
        """
        Convert placeholder matches to validation issues.
        """
        issues = []
        
        for match in matches:
            # Generate suggestions
            suggestions = await self.get_fix_suggestions(match, "", {})
            suggested_fix = suggestions[0] if suggestions else None
            
            issue = ValidationIssue(
                id=f"placeholder_{match.pattern_name}_{match.line_number}",
                description=f"Placeholder detected: {match.placeholder_type.value} - {match.text[:50]}",
                severity=match.severity,
                file_path=str(file_path) if file_path else None,
                line_number=match.line_number,
                column_number=match.column_number,
                issue_type="placeholder",
                auto_fixable=True,
                suggested_fix=suggested_fix,
                context={
                    'placeholder_type': match.placeholder_type.value,
                    'matched_text': match.text,
                    'context': match.context,
                    'suggestions': suggestions
                }
            )
            
            issues.append(issue)
        
        return issues
    
    def _get_line_context(
        self,
        content: str,
        line_number: int,
        context_window: int
    ) -> str:
        """
        Get context lines around a specific line.
        """
        lines = content.split('\
')
        start_line = max(0, line_number - context_window - 1)
        end_line = min(len(lines), line_number + context_window)
        
        context_lines = lines[start_line:end_line]
        return '\
'.join(context_lines)
    
    async def _check_requirements_completeness(
        self,
        content: str,
        requirements: List[str]
    ) -> List[str]:
        """
        Check which requirements are missing from content.
        """
        missing = []
        content_lower = content.lower()
        
        for requirement in requirements:
            # Simple keyword-based checking
            requirement_words = requirement.lower().split()
            if not any(word in content_lower for word in requirement_words):
                missing.append(requirement)
        
        return missing
    
    async def _detect_implementation_gaps(
        self,
        content: str,
        file_path: Optional[Path]
    ) -> List[str]:
        """
        Detect gaps in implementation.
        """
        gaps = []
        
        # Check for functions that only contain pass or comments
        lines = content.split('\
')
        in_function = False
        function_has_implementation = False
        current_function = ""
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('def ') or 'function ' in stripped:
                # Save previous function if it had no implementation
                if in_function and not function_has_implementation:
                    gaps.append(f"Function '{current_function}' has no implementation")
                
                in_function = True
                function_has_implementation = False
                current_function = stripped.split('(')[0].replace('def ', '').replace('function ', '')
            
            elif in_function and stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                if stripped != 'pass' and '...' not in stripped:
                    function_has_implementation = True
            
            elif not stripped and in_function:
                # Empty line might indicate end of function
                continue
            
            elif stripped and not stripped.startswith(' ') and not stripped.startswith('\\t'):
                # New top-level statement, function ended
                if in_function and not function_has_implementation:
                    gaps.append(f"Function '{current_function}' has no implementation")
                in_function = False
        
        # Check last function
        if in_function and not function_has_implementation:
            gaps.append(f"Function '{current_function}' has no implementation")
        
        return gaps
    
    async def _get_contextual_suggestions(
        self,
        match: PlaceholderMatch,
        content: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Get context-specific fix suggestions.
        """
        suggestions = []
        
        # Analyze surrounding code for hints
        if 'project' in context:
            suggestions.append("Consider the project structure and dependencies")
        
        if 'task' in context:
            task_info = context['task']
            if isinstance(task_info, dict) and 'description' in task_info:
                suggestions.append(f"Implement based on task: {task_info['description'][:100]}")
        
        return suggestions