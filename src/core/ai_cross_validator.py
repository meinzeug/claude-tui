"""
AI Cross-Validation System for Enhanced Anti-Hallucination.

This module implements cross-validation using multiple AI instances to verify
code authenticity and quality. It provides an additional layer of validation
beyond static analysis by leveraging AI understanding of code semantics.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .types import Issue, IssueType, Severity, ValidationResult, PathStr
from .logger import get_logger


class CrossValidationStrategy(str, Enum):
    """Cross-validation strategies."""
    CONSENSUS = "consensus"  # Multiple AI agreement
    ADVERSARIAL = "adversarial"  # AI vs AI validation
    HIERARCHICAL = "hierarchical"  # Expert AI review
    ENSEMBLE = "ensemble"  # Weighted voting


@dataclass
class AIValidationRequest:
    """Request for AI validation."""
    code_snippet: str
    context: str
    file_path: str
    validation_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIValidationResponse:
    """Response from AI validation."""
    is_authentic: bool
    confidence: float
    issues_detected: List[str]
    suggestions: List[str]
    quality_score: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidationResult:
    """Result from cross-validation process."""
    consensus_reached: bool
    final_authenticity: bool
    confidence_score: float
    individual_responses: List[AIValidationResponse]
    disagreements: List[str]
    resolution_method: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MockAIInterface:
    """Mock AI interface for testing and development."""
    
    def __init__(self, model_name: str = "mock-validator", bias: float = 0.0):
        self.model_name = model_name
        self.bias = bias  # Bias towards authentic (positive) or fake (negative)
        self.logger = get_logger(__name__)
    
    async def validate_code(self, request: AIValidationRequest) -> AIValidationResponse:
        """Mock AI validation of code."""
        await asyncio.sleep(0.1)  # Simulate AI processing time
        
        # Simple heuristics for mock validation
        code = request.code_snippet.lower()
        
        # Check for obvious placeholders
        placeholder_indicators = [
            'todo', 'fixme', 'xxx', 'not implemented', 'placeholder',
            'pass', 'raise notimplementederror', 'console.log'
        ]
        
        issues = []
        suggestions = []
        
        for indicator in placeholder_indicators:
            if indicator in code:
                issues.append(f"Found placeholder: {indicator}")
        
        # Mock quality assessment
        lines = len(request.code_snippet.split('\n'))
        has_docstring = '"""' in request.code_snippet or "'''" in request.code_snippet
        has_comments = '#' in request.code_snippet
        
        quality_factors = []
        if lines > 3:
            quality_factors.append(0.3)
        if has_docstring:
            quality_factors.append(0.4)
        if has_comments:
            quality_factors.append(0.2)
        if not issues:
            quality_factors.append(0.5)
        
        base_quality = sum(quality_factors)
        quality_score = max(0.0, min(100.0, base_quality * 100 + self.bias * 10))
        
        # Determine authenticity
        is_authentic = len(issues) == 0 and quality_score > 60
        
        # Apply model bias
        if self.bias > 0:  # Optimistic bias
            is_authentic = is_authentic or (quality_score > 40)
        elif self.bias < 0:  # Pessimistic bias
            is_authentic = is_authentic and (quality_score > 80)
        
        confidence = 0.8 if len(issues) == 0 or len(issues) > 2 else 0.6
        
        # Generate suggestions
        if issues:
            suggestions.append("Complete placeholder implementations")
        if not has_docstring and lines > 5:
            suggestions.append("Add documentation")
        if quality_score < 70:
            suggestions.append("Improve code quality")
        
        return AIValidationResponse(
            is_authentic=is_authentic,
            confidence=confidence,
            issues_detected=issues,
            suggestions=suggestions,
            quality_score=quality_score,
            reasoning=f"Mock validation by {self.model_name}: {'authentic' if is_authentic else 'needs improvement'}",
            metadata={
                'model': self.model_name,
                'processing_time': 0.1,
                'bias': self.bias
            }
        )


class AICrossValidator:
    """
    AI Cross-Validation system for enhanced authenticity verification.
    
    Uses multiple AI models/instances to cross-validate code authenticity,
    reducing false positives/negatives through consensus mechanisms.
    """
    
    def __init__(
        self,
        strategy: CrossValidationStrategy = CrossValidationStrategy.CONSENSUS,
        min_consensus: float = 0.7,
        timeout: float = 30.0
    ):
        self.strategy = strategy
        self.min_consensus = min_consensus
        self.timeout = timeout
        self.logger = get_logger(__name__)
        
        # Initialize AI interfaces (in real implementation, these would be different models)
        self.ai_interfaces = [
            MockAIInterface("validator-primary", bias=0.0),
            MockAIInterface("validator-secondary", bias=0.1),
            MockAIInterface("validator-critic", bias=-0.1),
        ]
        
        # Validation cache to avoid redundant API calls
        self._cache: Dict[str, CrossValidationResult] = {}
        self._cache_ttl = timedelta(hours=1)
        
        # Statistics
        self.stats = {
            'total_validations': 0,
            'consensus_reached': 0,
            'disagreements': 0,
            'cache_hits': 0,
            'avg_confidence': 0.0
        }
    
    async def cross_validate_code(
        self,
        code_snippet: str,
        context: str = "",
        file_path: str = "",
        force_refresh: bool = False
    ) -> CrossValidationResult:
        """
        Perform cross-validation of code using multiple AI models.
        
        Args:
            code_snippet: Code to validate
            context: Surrounding code context
            file_path: Path to the file (for context)
            force_refresh: Skip cache and force new validation
            
        Returns:
            CrossValidationResult with consensus information
        """
        
        # Check cache first
        cache_key = self._generate_cache_key(code_snippet, context)
        if not force_refresh and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            if datetime.utcnow() - cached_result.timestamp < self._cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_result
        
        self.stats['total_validations'] += 1
        
        # Prepare validation request
        request = AIValidationRequest(
            code_snippet=code_snippet,
            context=context,
            file_path=file_path,
            validation_type="authenticity_check",
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'strategy': self.strategy.value
            }
        )
        
        # Execute validation strategy
        if self.strategy == CrossValidationStrategy.CONSENSUS:
            result = await self._consensus_validation(request)
        elif self.strategy == CrossValidationStrategy.ADVERSARIAL:
            result = await self._adversarial_validation(request)
        elif self.strategy == CrossValidationStrategy.HIERARCHICAL:
            result = await self._hierarchical_validation(request)
        elif self.strategy == CrossValidationStrategy.ENSEMBLE:
            result = await self._ensemble_validation(request)
        else:
            result = await self._consensus_validation(request)  # Default fallback
        
        # Update statistics
        if result.consensus_reached:
            self.stats['consensus_reached'] += 1
        if result.disagreements:
            self.stats['disagreements'] += 1
        
        # Update average confidence
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (self.stats['total_validations'] - 1) + 
             result.confidence_score) / self.stats['total_validations']
        )
        
        # Cache result
        self._cache[cache_key] = result
        
        # Clean old cache entries
        self._clean_cache()
        
        return result
    
    async def batch_cross_validate(
        self,
        code_snippets: List[Tuple[str, str, str]],  # (code, context, file_path)
        max_concurrent: int = 3
    ) -> List[CrossValidationResult]:
        """
        Perform batch cross-validation of multiple code snippets.
        
        Args:
            code_snippets: List of (code, context, file_path) tuples
            max_concurrent: Maximum concurrent validations
            
        Returns:
            List of CrossValidationResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_single(snippet_data: Tuple[str, str, str]) -> CrossValidationResult:
            async with semaphore:
                code, context, file_path = snippet_data
                return await self.cross_validate_code(code, context, file_path)
        
        tasks = [validate_single(snippet) for snippet in code_snippets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Validation error: {result}")
                # Create error result
                error_result = CrossValidationResult(
                    consensus_reached=False,
                    final_authenticity=False,
                    confidence_score=0.0,
                    individual_responses=[],
                    disagreements=[f"Validation failed: {str(result)}"],
                    resolution_method="error"
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _consensus_validation(self, request: AIValidationRequest) -> CrossValidationResult:
        """Perform consensus-based validation."""
        
        # Get responses from all AI interfaces
        tasks = [ai.validate_code(request) for ai in self.ai_interfaces]
        
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            self.logger.error("Cross-validation timed out")
            return CrossValidationResult(
                consensus_reached=False,
                final_authenticity=False,
                confidence_score=0.0,
                individual_responses=[],
                disagreements=["Validation timeout"],
                resolution_method="timeout"
            )
        
        # Analyze consensus
        authentic_votes = sum(1 for r in responses if r.is_authentic)
        total_votes = len(responses)
        consensus_ratio = authentic_votes / total_votes
        
        # Determine final result
        consensus_reached = (
            consensus_ratio >= self.min_consensus or 
            consensus_ratio <= (1 - self.min_consensus)
        )
        
        final_authenticity = consensus_ratio > 0.5
        
        # Calculate weighted confidence
        confidences = [r.confidence for r in responses]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust confidence based on consensus strength
        if consensus_reached:
            confidence_multiplier = min(consensus_ratio, 1 - consensus_ratio) * 2
            confidence_score = avg_confidence * (0.5 + confidence_multiplier * 0.5)
        else:
            confidence_score = avg_confidence * 0.5  # Lower confidence for split decisions
        
        # Identify disagreements
        disagreements = []
        if not consensus_reached:
            disagreements.append(f"Split decision: {authentic_votes}/{total_votes} voted authentic")
        
        # Collect unique issues and suggestions
        all_issues = set()
        all_suggestions = set()
        for response in responses:
            all_issues.update(response.issues_detected)
            all_suggestions.update(response.suggestions)
        
        return CrossValidationResult(
            consensus_reached=consensus_reached,
            final_authenticity=final_authenticity,
            confidence_score=confidence_score,
            individual_responses=responses,
            disagreements=disagreements,
            resolution_method="consensus"
        )
    
    async def _adversarial_validation(self, request: AIValidationRequest) -> CrossValidationResult:
        """Perform adversarial validation (AI vs AI)."""
        
        if len(self.ai_interfaces) < 2:
            return await self._consensus_validation(request)
        
        # Use first AI as primary validator
        primary_response = await self.ai_interfaces[0].validate_code(request)
        
        # Use second AI as adversarial reviewer
        adversarial_request = AIValidationRequest(
            code_snippet=request.code_snippet,
            context=request.context + f"\nPrimary assessment: {primary_response.reasoning}",
            file_path=request.file_path,
            validation_type="adversarial_review",
            metadata=request.metadata
        )
        
        adversarial_response = await self.ai_interfaces[1].validate_code(adversarial_request)
        
        # Resolve disagreement
        agreement = (primary_response.is_authentic == adversarial_response.is_authentic)
        
        if agreement:
            final_authenticity = primary_response.is_authentic
            confidence_score = (primary_response.confidence + adversarial_response.confidence) / 2
            disagreements = []
            resolution_method = "adversarial_agreement"
        else:
            # Use third AI as tie-breaker if available
            if len(self.ai_interfaces) >= 3:
                tiebreaker_response = await self.ai_interfaces[2].validate_code(request)
                final_authenticity = tiebreaker_response.is_authentic
                confidence_score = tiebreaker_response.confidence * 0.7  # Lower confidence for tie-break
                resolution_method = "tiebreaker"
            else:
                # Conservative approach: err on the side of caution
                final_authenticity = False
                confidence_score = 0.3
                resolution_method = "conservative_fallback"
            
            disagreements = [f"Primary vs adversarial disagreement resolved via {resolution_method}"]
        
        return CrossValidationResult(
            consensus_reached=agreement,
            final_authenticity=final_authenticity,
            confidence_score=confidence_score,
            individual_responses=[primary_response, adversarial_response],
            disagreements=disagreements,
            resolution_method=resolution_method
        )
    
    async def _hierarchical_validation(self, request: AIValidationRequest) -> CrossValidationResult:
        """Perform hierarchical validation (expert review)."""
        
        # Start with basic validation
        basic_responses = []
        for i, ai in enumerate(self.ai_interfaces[:-1]):  # All but the last (expert)
            response = await ai.validate_code(request)
            basic_responses.append(response)
        
        # If basic validators agree, trust them
        basic_authentic = [r.is_authentic for r in basic_responses]
        if len(set(basic_authentic)) == 1:  # All agree
            final_authenticity = basic_authentic[0]
            confidence_score = sum(r.confidence for r in basic_responses) / len(basic_responses)
            
            return CrossValidationResult(
                consensus_reached=True,
                final_authenticity=final_authenticity,
                confidence_score=confidence_score,
                individual_responses=basic_responses,
                disagreements=[],
                resolution_method="basic_consensus"
            )
        
        # If basic validators disagree, escalate to expert
        if len(self.ai_interfaces) > 1:
            expert_request = AIValidationRequest(
                code_snippet=request.code_snippet,
                context=request.context + f"\nBasic assessment disagreement: {[r.reasoning for r in basic_responses]}",
                file_path=request.file_path,
                validation_type="expert_review",
                metadata=request.metadata
            )
            
            expert_response = await self.ai_interfaces[-1].validate_code(expert_request)
            all_responses = basic_responses + [expert_response]
            
            return CrossValidationResult(
                consensus_reached=True,
                final_authenticity=expert_response.is_authentic,
                confidence_score=expert_response.confidence,
                individual_responses=all_responses,
                disagreements=["Escalated to expert due to basic disagreement"],
                resolution_method="expert_override"
            )
        
        # Fallback to consensus
        return await self._consensus_validation(request)
    
    async def _ensemble_validation(self, request: AIValidationRequest) -> CrossValidationResult:
        """Perform ensemble validation with weighted voting."""
        
        # Define weights for each AI (could be based on historical accuracy)
        weights = [1.0, 0.8, 0.9][:len(self.ai_interfaces)]
        
        # Get all responses
        responses = []
        for ai in self.ai_interfaces:
            response = await ai.validate_code(request)
            responses.append(response)
        
        # Calculate weighted scores
        weighted_authenticity = 0.0
        weighted_confidence = 0.0
        total_weight = sum(weights)
        
        for i, response in enumerate(responses):
            weight = weights[i] if i < len(weights) else 1.0
            weighted_authenticity += weight * (1.0 if response.is_authentic else 0.0)
            weighted_confidence += weight * response.confidence
        
        weighted_authenticity /= total_weight
        weighted_confidence /= total_weight
        
        # Final decision
        final_authenticity = weighted_authenticity > 0.5
        confidence_score = weighted_confidence * (
            max(weighted_authenticity, 1 - weighted_authenticity) * 2
        )
        
        # Check for strong consensus
        consensus_reached = weighted_authenticity > 0.7 or weighted_authenticity < 0.3
        
        disagreements = []
        if not consensus_reached:
            disagreements.append(f"Weak ensemble consensus: {weighted_authenticity:.2f}")
        
        return CrossValidationResult(
            consensus_reached=consensus_reached,
            final_authenticity=final_authenticity,
            confidence_score=confidence_score,
            individual_responses=responses,
            disagreements=disagreements,
            resolution_method="weighted_ensemble"
        )
    
    def _generate_cache_key(self, code_snippet: str, context: str) -> str:
        """Generate cache key for validation request."""
        content = f"{code_snippet}{context}{self.strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _clean_cache(self) -> None:
        """Clean expired cache entries."""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, result in self._cache.items()
            if current_time - result.timestamp > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self.stats,
            'cache_size': len(self._cache),
            'strategy': self.strategy.value,
            'min_consensus': self.min_consensus,
            'success_rate': (
                self.stats['consensus_reached'] / self.stats['total_validations'] * 100
                if self.stats['total_validations'] > 0 else 0
            )
        }
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.stats = {
            'total_validations': 0,
            'consensus_reached': 0,
            'disagreements': 0,
            'cache_hits': 0,
            'avg_confidence': 0.0
        }
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._cache.clear()


# Integration utilities

async def enhance_validation_with_ai(
    validation_result: ValidationResult,
    cross_validator: AICrossValidator,
    code_snippets: Dict[str, str]  # file_path -> code mapping
) -> ValidationResult:
    """
    Enhance existing validation result with AI cross-validation.
    
    Args:
        validation_result: Original validation result
        cross_validator: AI cross-validator instance
        code_snippets: Mapping of file paths to code content
        
    Returns:
        Enhanced validation result with AI insights
    """
    
    # Collect suspicious code snippets
    suspicious_snippets = []
    for issue in validation_result.issues:
        if issue.file_path and issue.file_path in code_snippets:
            code = code_snippets[issue.file_path]
            # Extract relevant code section if line number is available
            if issue.line_number:
                lines = code.split('\n')
                start_line = max(0, issue.line_number - 3)
                end_line = min(len(lines), issue.line_number + 2)
                relevant_code = '\n'.join(lines[start_line:end_line])
            else:
                relevant_code = code
            
            suspicious_snippets.append((relevant_code, code, issue.file_path))
    
    # Perform AI cross-validation
    if suspicious_snippets:
        ai_results = await cross_validator.batch_cross_validate(suspicious_snippets)
        
        # Analyze AI feedback
        ai_confirmed_issues = 0
        ai_dismissed_issues = 0
        ai_suggestions = set()
        
        for i, ai_result in enumerate(ai_results):
            if not ai_result.final_authenticity:
                ai_confirmed_issues += 1
            else:
                ai_dismissed_issues += 1
            
            # Collect AI suggestions
            for response in ai_result.individual_responses:
                ai_suggestions.update(response.suggestions)
        
        # Adjust validation result based on AI feedback
        ai_confirmation_rate = ai_confirmed_issues / len(ai_results) if ai_results else 0
        
        # Adjust authenticity score
        if ai_confirmation_rate > 0.7:
            # AI confirms most issues - reduce authenticity score
            validation_result.authenticity_score *= 0.8
        elif ai_confirmation_rate < 0.3:
            # AI dismisses most issues - increase authenticity score slightly
            validation_result.authenticity_score = min(100.0, validation_result.authenticity_score * 1.1)
        
        # Add AI suggestions to validation result
        validation_result.suggestions.extend([
            f"AI Suggestion: {suggestion}" for suggestion in ai_suggestions
        ])
        
        # Add AI cross-validation metadata
        if hasattr(validation_result, 'metadata'):
            validation_result.metadata['ai_cross_validation'] = {
                'confirmed_issues': ai_confirmed_issues,
                'dismissed_issues': ai_dismissed_issues,
                'confirmation_rate': ai_confirmation_rate,
                'ai_suggestions_count': len(ai_suggestions)
            }
    
    return validation_result


async def validate_code_with_ai_cross_check(
    code: str,
    file_path: str = "",
    context: str = "",
    strategy: CrossValidationStrategy = CrossValidationStrategy.CONSENSUS
) -> Dict[str, Any]:
    """
    Convenience function for quick AI cross-validation of code.
    
    Args:
        code: Code to validate
        file_path: Path to the file
        context: Additional context
        strategy: Cross-validation strategy
        
    Returns:
        Validation result dictionary
    """
    cross_validator = AICrossValidator(strategy=strategy)
    result = await cross_validator.cross_validate_code(code, context, file_path)
    
    return {
        'is_authentic': result.final_authenticity,
        'confidence': result.confidence_score,
        'consensus_reached': result.consensus_reached,
        'disagreements': result.disagreements,
        'resolution_method': result.resolution_method,
        'individual_assessments': [
            {
                'is_authentic': r.is_authentic,
                'confidence': r.confidence,
                'issues': r.issues_detected,
                'suggestions': r.suggestions,
                'quality_score': r.quality_score
            }
            for r in result.individual_responses
        ],
        'statistics': cross_validator.get_statistics()
    }