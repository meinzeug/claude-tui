"""
Validation Service for claude-tui.

Provides comprehensive validation and anti-hallucination capabilities:
- Code validation and quality checking
- Placeholder and incomplete code detection
- Semantic validation of AI-generated content
- Response authenticity verification
- Performance and security validation
"""

import ast
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.exceptions import (
    PlaceholderDetectionError, SemanticValidationError, ValidationError
)
from ..core.validator import ProgressValidator
from .base import BaseService


class ValidationLevel(str):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class ValidationCategory(str):
    """Validation categories."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    PLACEHOLDER = "placeholder"
    QUALITY = "quality"
    SECURITY = "security"
    PERFORMANCE = "performance"


class ValidationService(BaseService):
    """
    Anti-Hallucination Validation Service.
    
    Provides comprehensive validation of AI-generated content
    to prevent hallucinated or incomplete code/responses.
    """
    
    def __init__(self):
        super().__init__()
        self._progress_validator: Optional[ProgressValidator] = None
        self._validation_rules: Dict[str, Any] = {}
        self._placeholder_patterns: Dict[str, List[str]] = {}
        self._validation_cache: Dict[str, Any] = {}
        self._validation_history: List[Dict[str, Any]] = []
        
    async def _initialize_impl(self) -> None:
        """Initialize validation service."""
        try:
            # Initialize progress validator
            self._progress_validator = ProgressValidator()
            
            # Load validation rules
            self._load_validation_rules()
            
            # Load placeholder patterns
            self._load_placeholder_patterns()
            
            self.logger.info("Validation service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize validation service: {str(e)}")
            raise ValidationError(f"Validation service initialization failed: {str(e)}")
    
    def _load_validation_rules(self) -> None:
        """Load validation rules for different content types."""
        self._validation_rules = {
            'python': {
                'required_imports': ['import', 'from'],
                'forbidden_patterns': ['eval(', 'exec(', '__import__'],
                'quality_metrics': {
                    'max_line_length': 88,
                    'max_function_length': 50,
                    'max_complexity': 10
                }
            },
            'javascript': {
                'required_patterns': ['function', 'const', 'let', 'var'],
                'forbidden_patterns': ['eval(', 'new Function(', 'setTimeout("'],
                'quality_metrics': {
                    'max_line_length': 100,
                    'max_function_length': 30
                }
            },
            'text': {
                'min_length': 10,
                'max_length': 10000,
                'required_structure': ['sentences', 'paragraphs']
            },
            'general': {
                'completeness_threshold': 0.8,
                'coherence_threshold': 0.7
            }
        }
    
    def _load_placeholder_patterns(self) -> None:
        """Load placeholder detection patterns."""
        self._placeholder_patterns = {
            'python': [
                r'#\s*(TODO|FIXME|PLACEHOLDER|IMPLEMENT)',
                r'pass\s*#.*implement',
                r'raise\s+NotImplementedError',
                r'\.\.\.(?:\s*#.*)?$',
                r'def\s+\w+\s*\([^)]*\):\s*pass\s*$',
                r'class\s+\w+.*:\s*pass\s*$',
                r'return\s+None\s*#.*implement'
            ],
            'javascript': [
                r'//\s*(TODO|FIXME|PLACEHOLDER|IMPLEMENT)',
                r'/\*\s*(TODO|FIXME|PLACEHOLDER)',
                r'throw\s+new\s+Error\s*\(\s*["\']Not implemented',
                r'console\.log\s*\(\s*["\']TODO',
                r'function\s+\w+.*{\s*//.*implement',
                r'=>\s*{\s*//.*implement'
            ],
            'general': [
                r'\[.*PLACEHOLDER.*\]',
                r'\{.*TODO.*\}',
                r'<.*IMPLEMENT.*>',
                r'FILL_IN_.*',
                r'YOUR_.*_HERE',
                r'REPLACE_WITH_.*'
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check with validation-specific status."""
        base_health = await super().health_check()
        
        base_health.update({
            'progress_validator_available': self._progress_validator is not None,
            'validation_rules_loaded': len(self._validation_rules),
            'placeholder_patterns_loaded': len(self._placeholder_patterns),
            'cache_size': len(self._validation_cache),
            'validation_history_size': len(self._validation_history)
        })
        
        return base_health
    
    async def validate_code(
        self,
        code: str,
        language: str = 'python',
        file_path: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        check_placeholders: bool = True,
        check_syntax: bool = True,
        check_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive code validation.
        
        Args:
            code: Code content to validate
            language: Programming language
            file_path: Optional file path for context
            validation_level: Level of validation to perform
            check_placeholders: Whether to check for placeholders
            check_syntax: Whether to check syntax
            check_quality: Whether to check code quality
            
        Returns:
            Validation result with detailed analysis
        """
        return await self.execute_with_monitoring(
            'validate_code',
            self._validate_code_impl,
            code=code,
            language=language,
            file_path=file_path,
            validation_level=validation_level,
            check_placeholders=check_placeholders,
            check_syntax=check_syntax,
            check_quality=check_quality
        )
    
    async def _validate_code_impl(
        self,
        code: str,
        language: str,
        file_path: Optional[str],
        validation_level: ValidationLevel,
        check_placeholders: bool,
        check_syntax: bool,
        check_quality: bool
    ) -> Dict[str, Any]:
        """Internal code validation implementation."""
        validation_result = {
            'is_valid': True,
            'score': 1.0,
            'issues': [],
            'warnings': [],
            'suggestions': [],
            'metadata': {
                'language': language,
                'file_path': file_path,
                'validation_level': validation_level,
                'code_length': len(code),
                'line_count': len(code.splitlines()),
                'validated_at': datetime.utcnow().isoformat()
            },
            'categories': {}
        }
        
        if not code or code.strip() == '':
            validation_result.update({
                'is_valid': False,
                'score': 0.0,
                'issues': ['Code content is empty']
            })
            return validation_result
        
        try:
            # Placeholder detection
            if check_placeholders:
                placeholder_result = await self._detect_placeholders(code, language)
                validation_result['categories']['placeholder'] = placeholder_result
                
                if placeholder_result['count'] > 0:
                    validation_result['score'] *= (1 - placeholder_result['severity'] * 0.3)
                    validation_result['warnings'].extend(placeholder_result['warnings'])
            
            # Syntax validation
            if check_syntax:
                syntax_result = await self._validate_syntax(code, language)
                validation_result['categories']['syntax'] = syntax_result
                
                if not syntax_result['is_valid']:
                    validation_result['is_valid'] = False
                    validation_result['score'] = 0.0
                    validation_result['issues'].extend(syntax_result['errors'])
            
            # Quality validation
            if check_quality and validation_result['is_valid']:
                quality_result = await self._validate_code_quality(code, language)
                validation_result['categories']['quality'] = quality_result
                
                validation_result['score'] *= quality_result['score']
                validation_result['suggestions'].extend(quality_result['suggestions'])
                validation_result['warnings'].extend(quality_result['warnings'])
            
            # Security validation (for strict/comprehensive levels)
            if validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                security_result = await self._validate_security(code, language)
                validation_result['categories']['security'] = security_result
                
                if security_result['risk_level'] == 'high':
                    validation_result['score'] *= 0.3
                    validation_result['issues'].extend(security_result['issues'])
                elif security_result['risk_level'] == 'medium':
                    validation_result['score'] *= 0.7
                    validation_result['warnings'].extend(security_result['warnings'])
            
            # Semantic validation (for comprehensive level)
            if validation_level == ValidationLevel.COMPREHENSIVE:
                semantic_result = await self._validate_semantics(code, language, file_path)
                validation_result['categories']['semantic'] = semantic_result
                
                validation_result['score'] *= semantic_result['score']
                validation_result['suggestions'].extend(semantic_result['suggestions'])
            
            # Store validation history
            self._add_to_validation_history(validation_result, 'code')
            
        except Exception as e:
            validation_result.update({
                'is_valid': False,
                'score': 0.0,
                'issues': [f'Validation error: {str(e)}']
            })
        
        return validation_result
    
    async def _detect_placeholders(self, code: str, language: str) -> Dict[str, Any]:
        """Detect placeholder patterns in code."""
        patterns = self._placeholder_patterns.get(language, []) + self._placeholder_patterns.get('general', [])
        
        placeholder_result = {
            'count': 0,
            'patterns_found': [],
            'locations': [],
            'severity': 0.0,
            'warnings': []
        }
        
        lines = code.splitlines()
        
        for i, line in enumerate(lines, 1):
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    placeholder_result['count'] += 1
                    placeholder_result['patterns_found'].append(match.group(0))
                    placeholder_result['locations'].append({
                        'line': i,
                        'column': match.start(),
                        'text': match.group(0)
                    })
        
        if placeholder_result['count'] > 0:
            # Calculate severity based on count and types
            total_lines = len(lines)
            placeholder_ratio = placeholder_result['count'] / total_lines
            
            if placeholder_ratio > 0.3:
                placeholder_result['severity'] = 1.0  # High
            elif placeholder_ratio > 0.1:
                placeholder_result['severity'] = 0.7  # Medium
            else:
                placeholder_result['severity'] = 0.3  # Low
            
            placeholder_result['warnings'].append(
                f"Found {placeholder_result['count']} placeholder patterns "
                f"({placeholder_result['severity']:.1%} severity)"
            )
        
        return placeholder_result
    
    async def _validate_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax."""
        syntax_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if language.lower() == 'python':
                # Use AST to parse Python code
                ast.parse(code)
            elif language.lower() in ['javascript', 'js']:
                # For JavaScript, we'd need a JS parser - simplified check here
                if 'SyntaxError' in code or 'syntax error' in code.lower():
                    syntax_result['is_valid'] = False
                    syntax_result['errors'].append('Potential JavaScript syntax error detected')
            # Add more language support as needed
            
        except SyntaxError as e:
            syntax_result['is_valid'] = False
            syntax_result['errors'].append(f"Syntax error: {str(e)} at line {e.lineno}")
        except Exception as e:
            syntax_result['warnings'].append(f"Syntax validation warning: {str(e)}")
        
        return syntax_result
    
    async def _validate_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code quality metrics."""
        quality_result = {
            'score': 1.0,
            'suggestions': [],
            'warnings': [],
            'metrics': {}
        }
        
        rules = self._validation_rules.get(language, {})
        quality_metrics = rules.get('quality_metrics', {})
        
        lines = code.splitlines()
        
        # Line length check
        max_line_length = quality_metrics.get('max_line_length', 100)
        long_lines = [
            i + 1 for i, line in enumerate(lines) 
            if len(line) > max_line_length
        ]
        
        if long_lines:
            quality_result['score'] *= 0.9
            quality_result['warnings'].append(
                f"Lines exceed maximum length ({max_line_length}): {long_lines}"
            )
        
        quality_result['metrics']['line_length'] = {
            'max': max(len(line) for line in lines) if lines else 0,
            'average': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'violations': len(long_lines)
        }
        
        # Function complexity (simplified for Python)
        if language.lower() == 'python':
            function_count = len(re.findall(r'def\s+\w+', code))
            class_count = len(re.findall(r'class\s+\w+', code))
            
            quality_result['metrics']['structure'] = {
                'functions': function_count,
                'classes': class_count,
                'total_lines': len(lines)
            }
            
            if function_count == 0 and class_count == 0 and len(lines) > 20:
                quality_result['suggestions'].append(
                    "Consider organizing code into functions or classes"
                )
                quality_result['score'] *= 0.8
        
        # Documentation check
        if language.lower() == 'python':
            docstring_count = len(re.findall(r'"""[\s\S]*?"""', code))
            comment_count = len(re.findall(r'#.*', code))
            
            if docstring_count == 0 and comment_count < len(lines) * 0.1:
                quality_result['suggestions'].append(
                    "Consider adding more documentation and comments"
                )
                quality_result['score'] *= 0.9
        
        return quality_result
    
    async def _validate_security(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code for security issues."""
        security_result = {
            'risk_level': 'low',
            'issues': [],
            'warnings': [],
            'patterns_detected': []
        }
        
        rules = self._validation_rules.get(language, {})
        forbidden_patterns = rules.get('forbidden_patterns', [])
        
        for pattern in forbidden_patterns:
            if pattern in code:
                security_result['patterns_detected'].append(pattern)
                
                if pattern in ['eval(', 'exec(', '__import__']:
                    security_result['risk_level'] = 'high'
                    security_result['issues'].append(f"High-risk pattern detected: {pattern}")
                else:
                    if security_result['risk_level'] == 'low':
                        security_result['risk_level'] = 'medium'
                    security_result['warnings'].append(f"Potentially risky pattern: {pattern}")
        
        # Check for hardcoded credentials/secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                if security_result['risk_level'] == 'low':
                    security_result['risk_level'] = 'medium'
                security_result['warnings'].append(
                    "Potential hardcoded secret detected - use environment variables"
                )
        
        return security_result
    
    async def _validate_semantics(
        self, 
        code: str, 
        language: str, 
        file_path: Optional[str]
    ) -> Dict[str, Any]:
        """Validate code semantics and logic."""
        semantic_result = {
            'score': 1.0,
            'suggestions': [],
            'warnings': [],
            'coherence_metrics': {}
        }
        
        if language.lower() == 'python':
            try:
                # Parse AST for semantic analysis
                tree = ast.parse(code)
                
                # Check for unused variables (simplified)
                defined_vars = set()
                used_vars = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        if isinstance(node.ctx, ast.Store):
                            defined_vars.add(node.id)
                        elif isinstance(node.ctx, ast.Load):
                            used_vars.add(node.id)
                
                unused_vars = defined_vars - used_vars
                if unused_vars:
                    semantic_result['suggestions'].append(
                        f"Consider removing unused variables: {', '.join(unused_vars)}"
                    )
                    semantic_result['score'] *= 0.9
                
                semantic_result['coherence_metrics'] = {
                    'defined_variables': len(defined_vars),
                    'used_variables': len(used_vars),
                    'unused_variables': len(unused_vars)
                }
                
            except Exception as e:
                semantic_result['warnings'].append(f"Semantic analysis warning: {str(e)}")
        
        return semantic_result
    
    async def validate_response(
        self,
        response: Union[str, Dict[str, Any]],
        response_type: str = 'text',
        validation_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate AI response for completeness and authenticity.
        
        Args:
            response: Response content to validate
            response_type: Type of response (text, json, code, etc.)
            validation_criteria: Custom validation criteria
            
        Returns:
            Validation result
        """
        return await self.execute_with_monitoring(
            'validate_response',
            self._validate_response_impl,
            response=response,
            response_type=response_type,
            validation_criteria=validation_criteria
        )
    
    async def _validate_response_impl(
        self,
        response: Union[str, Dict[str, Any]],
        response_type: str,
        validation_criteria: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Internal response validation implementation."""
        validation_result = {
            'is_valid': True,
            'score': 1.0,
            'issues': [],
            'warnings': [],
            'metadata': {
                'response_type': response_type,
                'validation_criteria': validation_criteria,
                'validated_at': datetime.utcnow().isoformat()
            }
        }
        
        criteria = validation_criteria or {}
        
        if response_type == 'text':
            content = response if isinstance(response, str) else str(response)
            
            # Length validation
            min_length = criteria.get('min_length', 1)
            max_length = criteria.get('max_length', 100000)
            
            if len(content) < min_length:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Content too short (< {min_length} chars)")
            elif len(content) > max_length:
                validation_result['warnings'].append(f"Content very long (> {max_length} chars)")
            
            # Coherence check (simplified)
            if len(content.strip()) == 0:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Content is empty")
            
        elif response_type == 'json':
            if isinstance(response, str):
                try:
                    import json
                    json.loads(response)
                except json.JSONDecodeError as e:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"Invalid JSON: {str(e)}")
            
        elif response_type == 'code':
            content = response if isinstance(response, str) else str(response)
            language = criteria.get('language', 'python')
            
            # Use code validation
            code_validation = await self.validate_code(
                content, 
                language, 
                validation_level=ValidationLevel.STANDARD
            )
            
            validation_result.update({
                'is_valid': code_validation['is_valid'],
                'score': code_validation['score'],
                'issues': code_validation['issues'],
                'warnings': code_validation['warnings'],
                'code_validation': code_validation
            })
        
        # Add to validation history
        self._add_to_validation_history(validation_result, response_type)
        
        return validation_result
    
    async def check_progress_authenticity(
        self,
        file_path: Union[str, Path],
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check file progress for authenticity using progress validator."""
        if not self._progress_validator:
            raise ValidationError("Progress validator not initialized")
        
        return await self.execute_with_monitoring(
            'check_progress_authenticity',
            self._check_progress_authenticity_impl,
            file_path=file_path,
            project_context=project_context
        )
    
    async def _check_progress_authenticity_impl(
        self,
        file_path: Union[str, Path],
        project_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Internal progress authenticity check."""
        try:
            # Use progress validator
            progress_result = await self._progress_validator.validate_progress(
                file_path=Path(file_path),
                project_context=project_context
            )
            
            validation_result = {
                'is_authentic': progress_result.is_valid,
                'authenticity_score': progress_result.authenticity_score,
                'issues': progress_result.issues,
                'suggestions': progress_result.suggestions,
                'placeholder_count': progress_result.placeholder_count,
                'metadata': {
                    'file_path': str(file_path),
                    'validated_at': datetime.utcnow().isoformat(),
                    'validator_version': getattr(self._progress_validator, 'version', 'unknown')
                }
            }
            
            # Add to validation history
            self._add_to_validation_history(validation_result, 'progress')
            
            return validation_result
            
        except Exception as e:
            error_result = {
                'is_authentic': False,
                'authenticity_score': 0.0,
                'issues': [f'Progress validation failed: {str(e)}'],
                'suggestions': [],
                'placeholder_count': 0,
                'metadata': {
                    'file_path': str(file_path),
                    'validated_at': datetime.utcnow().isoformat(),
                    'error': str(e)
                }
            }
            
            self._add_to_validation_history(error_result, 'progress')
            return error_result
    
    def _add_to_validation_history(
        self, 
        validation_result: Dict[str, Any], 
        validation_type: str
    ) -> None:
        """Add validation result to history."""
        history_entry = {
            'type': validation_type,
            'timestamp': datetime.utcnow().isoformat(),
            'is_valid': validation_result.get('is_valid', False),
            'score': validation_result.get('score', 0.0),
            'issues_count': len(validation_result.get('issues', [])),
            'warnings_count': len(validation_result.get('warnings', [])),
            'metadata': validation_result.get('metadata', {})
        }
        
        self._validation_history.append(history_entry)
        
        # Keep history size manageable
        if len(self._validation_history) > 1000:
            self._validation_history = self._validation_history[-500:]
    
    async def get_validation_report(
        self,
        limit: Optional[int] = None,
        validation_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        history = self._validation_history.copy()
        
        # Apply filters
        if validation_type_filter:
            history = [item for item in history if item['type'] == validation_type_filter]
        
        if limit:
            history = history[-limit:]
        
        # Calculate metrics
        total_validations = len(history)
        successful_validations = sum(1 for item in history if item['is_valid'])
        success_rate = successful_validations / total_validations if total_validations > 0 else 0
        
        average_score = sum(item['score'] for item in history) / total_validations if total_validations > 0 else 0
        
        # Type breakdown
        type_breakdown = {}
        for item in history:
            item_type = item['type']
            if item_type not in type_breakdown:
                type_breakdown[item_type] = {'count': 0, 'success_count': 0, 'total_score': 0}
            
            type_breakdown[item_type]['count'] += 1
            type_breakdown[item_type]['total_score'] += item['score']
            if item['is_valid']:
                type_breakdown[item_type]['success_count'] += 1
        
        # Calculate type-specific metrics
        for type_name, stats in type_breakdown.items():
            stats['success_rate'] = stats['success_count'] / stats['count']
            stats['average_score'] = stats['total_score'] / stats['count']
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': success_rate,
            'average_score': average_score,
            'type_breakdown': type_breakdown,
            'recent_history': history[-20:] if len(history) >= 20 else history,
            'report_generated_at': datetime.utcnow().isoformat()
        }