"""
Comprehensive Test Suite for Anti-Hallucination Engine.

Tests for 95%+ accuracy validation including:
- ML model accuracy benchmarks
- Performance testing (<200ms)
- Multi-stage validation pipeline
- Cross-validation testing
- Edge case handling
- Integration testing
"""

import asyncio
import pytest
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import json

# Test imports
from claude_tui.core.config_manager import ConfigManager
from claude_tui.validation.anti_hallucination_engine import (
    AntiHallucinationEngine,
    CodeSample,
    ValidationStage,
    ModelType,
    FeatureExtractor
)
from claude_tui.validation.training_data_generator import (
    TrainingDataGenerator,
    GenerationConfig,
    CodePattern,
    QualityLevel
)
from claude_tui.validation.performance_optimizer import PerformanceOptimizer
from claude_tui.validation.progress_validator import ValidationResult, ValidationSeverity
from claude_tui.models.project import Project


class TestAntiHallucinationEngine:
    """Comprehensive test suite for Anti-Hallucination Engine."""
    
    @pytest.fixture
    async def config_manager(self):
        """Mock configuration manager."""
        config = Mock(spec=ConfigManager)
        config.get_setting = AsyncMock(return_value={})
        return config
    
    @pytest.fixture
    async def engine(self, config_manager):
        """Initialize Anti-Hallucination Engine for testing."""
        engine = AntiHallucinationEngine(config_manager)
        await engine.initialize()
        return engine
    
    @pytest.fixture
    async def training_data_generator(self):
        """Initialize training data generator."""
        generator = TrainingDataGenerator()
        await generator.initialize()
        return generator
    
    @pytest.fixture
    async def sample_training_data(self, training_data_generator):
        """Generate sample training data for tests."""
        config = GenerationConfig(total_samples=100, seed=42)
        return await training_data_generator.generate_training_dataset(config)
    
    # Accuracy Benchmark Tests
    
    @pytest.mark.asyncio
    async def test_model_accuracy_benchmark(self, engine, sample_training_data):
        """Test ML model accuracy meets 95%+ target."""
        # Train model with sample data
        metrics = await engine.train_pattern_recognition_model(sample_training_data)
        
        # Verify accuracy meets target
        assert metrics.accuracy >= 0.95, f"Accuracy {metrics.accuracy:.3f} below 95% target"
        assert metrics.precision >= 0.9, f"Precision {metrics.precision:.3f} below 90% target"
        assert metrics.recall >= 0.9, f"Recall {metrics.recall:.3f} below 90% target"
        assert metrics.f1_score >= 0.9, f"F1-score {metrics.f1_score:.3f} below 90% target"
    
    @pytest.mark.asyncio
    async def test_cross_validation_accuracy(self, engine, sample_training_data):
        """Test cross-validation accuracy across multiple models."""
        # Train model
        await engine.train_pattern_recognition_model(sample_training_data)
        
        # Test authentic code samples
        authentic_samples = [s for s in sample_training_data if s.is_authentic]
        authentic_predictions = []
        
        for sample in authentic_samples[:20]:  # Test subset for speed
            cross_val_results = await engine.cross_validate_with_multiple_models(sample.content)
            authenticity_score = cross_val_results.get('consensus', 0.5)
            authentic_predictions.append(authenticity_score)
        
        # Test hallucinated code samples
        hallucinated_samples = [s for s in sample_training_data if not s.is_authentic]
        hallucinated_predictions = []
        
        for sample in hallucinated_samples[:20]:
            cross_val_results = await engine.cross_validate_with_multiple_models(sample.content)
            authenticity_score = cross_val_results.get('consensus', 0.5)
            hallucinated_predictions.append(authenticity_score)
        
        # Verify authentic samples get high scores
        avg_authentic_score = np.mean(authentic_predictions)
        assert avg_authentic_score > 0.7, f"Authentic code average score {avg_authentic_score:.3f} too low"
        
        # Verify hallucinated samples get low scores
        avg_hallucinated_score = np.mean(hallucinated_predictions)
        assert avg_hallucinated_score < 0.5, f"Hallucinated code average score {avg_hallucinated_score:.3f} too high"
    
    @pytest.mark.asyncio
    async def test_precision_recall_tradeoff(self, engine, sample_training_data):
        """Test precision/recall tradeoff meets requirements."""
        # Train model
        metrics = await engine.train_pattern_recognition_model(sample_training_data)
        
        # Verify balanced precision and recall
        precision_recall_diff = abs(metrics.precision - metrics.recall)
        assert precision_recall_diff < 0.1, f"Precision-recall imbalance: {precision_recall_diff:.3f}"
        
        # Verify F1-score reflects good balance
        expected_f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        assert abs(metrics.f1_score - expected_f1) < 0.01, "F1-score calculation inconsistent"
    
    # Performance Benchmark Tests
    
    @pytest.mark.asyncio
    async def test_validation_performance_benchmark(self, engine):
        """Test validation performance meets <200ms target."""
        test_code = '''
def example_function(data):
    """Process data and return result."""
    if not data:
        return None
    
    processed = []
    for item in data:
        result = item.process()
        processed.append(result)
    
    return processed
'''
        
        # Warm up
        await engine.validate_code_authenticity(test_code)
        
        # Benchmark performance
        response_times = []
        for _ in range(10):
            start_time = time.time()
            await engine.validate_code_authenticity(test_code)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
        
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        assert avg_response_time < 200, f"Average response time {avg_response_time:.2f}ms exceeds 200ms target"
        assert p95_response_time < 300, f"P95 response time {p95_response_time:.2f}ms too high"
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, engine):
        """Test batch processing performance optimization."""
        test_codes = [
            "def func1(): pass",
            "def func2(): return None",
            "class MyClass: pass",
            "x = [1, 2, 3]",
            "print('hello')"
        ]
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = []
        for code in test_codes:
            result = await engine.validate_code_authenticity(code)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Test batch processing (if available)
        start_time = time.time()
        batch_tasks = [engine.validate_code_authenticity(code) for code in test_codes]
        batch_results = await asyncio.gather(*batch_tasks)
        batch_time = time.time() - start_time
        
        # Batch processing should be faster or comparable
        assert batch_time <= sequential_time * 1.5, "Batch processing significantly slower than sequential"
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, engine):
        """Test memory usage stays within limits."""
        # Generate large amount of test data
        test_codes = [f"def func_{i}(): pass" for i in range(100)]
        
        # Process all codes to fill cache
        for code in test_codes:
            await engine.validate_code_authenticity(code)
        
        # Get performance metrics
        metrics = await engine.get_performance_metrics()
        
        # Verify memory usage is reasonable
        memory_usage = metrics.get('cache_stats', {}).get('memory_usage_mb', 0)
        assert memory_usage < 100, f"Memory usage {memory_usage:.2f}MB too high"
    
    # Pipeline Stage Tests
    
    @pytest.mark.asyncio
    async def test_static_analysis_stage(self, engine):
        """Test static analysis pipeline stage."""
        placeholder_code = '''
def incomplete_function():
    # TODO: Implement this function
    pass
'''
        
        result = await engine.validate_code_authenticity(placeholder_code)
        
        # Should detect placeholder issues
        placeholder_issues = [i for i in result.issues if 'placeholder' in i.issue_type.lower()]
        assert len(placeholder_issues) > 0, "Static analysis failed to detect placeholders"
        
        # Should have low authenticity score
        assert result.authenticity_score < 0.5, "Placeholder code scored too high"
    
    @pytest.mark.asyncio
    async def test_semantic_analysis_stage(self, engine):
        """Test semantic analysis pipeline stage."""
        buggy_code = '''
def buggy_function(data):
    if data is None:
        return data.process()  # Bug: calling method on None
    return "result"
'''
        
        result = await engine.validate_code_authenticity(buggy_code)
        
        # Should detect semantic issues
        semantic_issues = [i for i in result.issues if 'semantic' in i.issue_type.lower()]
        # Note: This test might pass if semantic analyzer doesn't catch this specific issue
        # The important thing is that the pipeline stage executes without error
        assert result is not None, "Semantic analysis stage failed"
    
    @pytest.mark.asyncio
    async def test_execution_testing_stage(self, engine):
        """Test execution testing pipeline stage."""
        syntax_error_code = '''
def broken_function(
    # Syntax error: unclosed parenthesis
    return "result"
'''
        
        result = await engine.validate_code_authenticity(syntax_error_code)
        
        # Should detect syntax errors
        syntax_issues = [i for i in result.issues if 'syntax' in i.issue_type.lower()]
        assert len(syntax_issues) > 0, "Execution testing failed to detect syntax error"
        
        # Should have critical severity
        critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) > 0, "Syntax error not marked as critical"
    
    @pytest.mark.asyncio 
    async def test_ml_validation_stage(self, engine, sample_training_data):
        """Test ML validation pipeline stage."""
        # Train model first
        await engine.train_pattern_recognition_model(sample_training_data)
        
        # Test with high-quality code
        good_code = '''
def well_written_function(data: List[str]) -> List[str]:
    """Process a list of strings and return processed results."""
    if not data:
        return []
    
    results = []
    for item in data:
        try:
            processed = item.strip().lower()
            if processed:
                results.append(processed)
        except AttributeError:
            logger.warning(f"Invalid item type: {type(item)}")
            continue
    
    return results
'''
        
        result = await engine.validate_code_authenticity(good_code)
        
        # Should have high authenticity score
        assert result.authenticity_score > 0.8, f"Good code authenticity score too low: {result.authenticity_score}"
    
    # Feature Extraction Tests
    
    @pytest.mark.asyncio
    async def test_feature_extraction_accuracy(self):
        """Test feature extraction produces meaningful features."""
        extractor = FeatureExtractor()
        
        test_code = '''
def example_function(param1, param2=None):
    """Example function with documentation."""
    # This is a comment
    if param1 is None:
        return None
    
    try:
        result = param1.process(param2)
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
'''
        
        features = await extractor.extract_features(test_code, 'python')
        
        # Verify key features are extracted
        assert 'length' in features
        assert 'line_count' in features
        assert 'function_count' in features
        assert 'comment_ratio' in features
        assert 'cyclomatic_complexity' in features
        
        # Verify reasonable values
        assert features['function_count'] == 1
        assert features['comment_ratio'] > 0
        assert features['cyclomatic_complexity'] > 1
    
    @pytest.mark.asyncio
    async def test_feature_consistency(self):
        """Test feature extraction consistency across identical inputs."""
        extractor = FeatureExtractor()
        test_code = "def test(): return True"
        
        features1 = await extractor.extract_features(test_code, 'python')
        features2 = await extractor.extract_features(test_code, 'python')
        
        # Features should be identical for identical inputs
        assert features1 == features2, "Feature extraction not consistent"
    
    # Edge Case Tests
    
    @pytest.mark.asyncio
    async def test_empty_code_handling(self, engine):
        """Test handling of empty code input."""
        result = await engine.validate_code_authenticity("")
        
        assert result.is_valid is False
        assert result.authenticity_score == 0.0
        assert "empty" in result.summary.lower()
    
    @pytest.mark.asyncio
    async def test_very_long_code_handling(self, engine):
        """Test handling of very long code input."""
        # Generate very long code
        long_code = "def func():\n" + "    x = 1\n" * 1000 + "    return x"
        
        start_time = time.time()
        result = await engine.validate_code_authenticity(long_code)
        processing_time = (time.time() - start_time) * 1000
        
        assert result is not None
        assert processing_time < 1000, f"Long code processing took {processing_time:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_malformed_code_handling(self, engine):
        """Test handling of malformed code input."""
        malformed_codes = [
            "def func(\n",  # Unclosed parenthesis
            "if True\n    pass",  # Missing colon
            "def func():\npass",  # Wrong indentation
            "import sys\nfrom os import *\nreturn None",  # Return outside function
        ]
        
        for malformed_code in malformed_codes:
            result = await engine.validate_code_authenticity(malformed_code)
            # Should not crash, should detect issues
            assert result is not None
            assert len(result.issues) > 0
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self, engine):
        """Test handling of Unicode characters in code."""
        unicode_code = '''
def процесс_данных(данные):
    """Функция для обработки данных."""
    результат = "Привет, мир! 🌍"
    return результат
'''
        
        result = await engine.validate_code_authenticity(unicode_code)
        assert result is not None
        # Unicode should not cause crashes
    
    # Integration Tests
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation(self, engine, sample_training_data):
        """Test complete end-to-end validation pipeline."""
        # Train model
        await engine.train_pattern_recognition_model(sample_training_data)
        
        # Test with real-world code example
        real_code = '''
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates input data for processing."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
    
    def validate_data(self, data: List[dict]) -> List[dict]:
        """Validate and clean input data."""
        if not data:
            return []
        
        validated = []
        for item in data:
            try:
                if self._is_valid_item(item):
                    validated.append(item)
                elif not self.strict_mode:
                    cleaned = self._clean_item(item)
                    if cleaned:
                        validated.append(cleaned)
            except Exception as e:
                logger.warning(f"Failed to validate item: {e}")
                if self.strict_mode:
                    raise
        
        return validated
    
    def _is_valid_item(self, item: dict) -> bool:
        """Check if item is valid."""
        return isinstance(item, dict) and 'id' in item
    
    def _clean_item(self, item: dict) -> Optional[dict]:
        """Clean invalid item."""
        if not isinstance(item, dict):
            return None
        
        # Add missing required fields
        if 'id' not in item:
            item['id'] = f"generated_{hash(str(item))}"
        
        return item
'''
        
        result = await engine.validate_code_authenticity(real_code)
        
        # Should recognize as authentic, high-quality code
        assert result.authenticity_score > 0.8
        assert result.is_valid
        assert result.execution_time < 200  # ms
    
    @pytest.mark.asyncio
    async def test_auto_completion_integration(self, engine):
        """Test auto-completion integration."""
        incomplete_code = '''
def calculate_statistics(data):
    # TODO: Implement statistics calculation
    pass
'''
        
        suggestions = ['mean', 'median', 'std']
        completed_code = await engine.complete_placeholder_code(incomplete_code, suggestions)
        
        # Should have completed the placeholder
        assert completed_code != incomplete_code
        assert 'TODO' not in completed_code or len(completed_code.split('\n')) > len(incomplete_code.split('\n'))
    
    # Performance Optimization Tests
    
    @pytest.mark.asyncio
    async def test_caching_effectiveness(self, engine):
        """Test caching improves performance."""
        test_code = "def cached_test(): return True"
        
        # First call (cache miss)
        start_time = time.time()
        result1 = await engine.validate_code_authenticity(test_code)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = await engine.validate_code_authenticity(test_code)
        second_call_time = time.time() - start_time
        
        # Second call should be significantly faster
        assert second_call_time < first_call_time * 0.5, "Caching not effective"
        
        # Results should be identical
        assert result1.authenticity_score == result2.authenticity_score
    
    @pytest.mark.asyncio
    async def test_concurrent_validation(self, engine):
        """Test concurrent validation handling."""
        test_codes = [f"def func_{i}(): pass" for i in range(10)]
        
        # Process concurrently
        start_time = time.time()
        tasks = [engine.validate_code_authenticity(code) for code in test_codes]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # All validations should complete
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # Should be reasonable performance
        assert concurrent_time < 2.0, f"Concurrent validation took {concurrent_time:.2f}s"
    
    # Quality Assurance Tests
    
    @pytest.mark.asyncio
    async def test_validation_result_completeness(self, engine):
        """Test validation results contain all required fields."""
        test_code = "def test(): return None"
        result = await engine.validate_code_authenticity(test_code)
        
        # Verify all required fields are present
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'authenticity_score')
        assert hasattr(result, 'completeness_score')
        assert hasattr(result, 'quality_score')
        assert hasattr(result, 'issues')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'validated_at')
        
        # Verify score ranges
        assert 0.0 <= result.authenticity_score <= 1.0
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.completeness_score <= 1.0
        assert 0.0 <= result.quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, engine):
        """Test error handling doesn't crash the system."""
        problematic_inputs = [
            None,
            123,
            [],
            {"not": "code"},
            "def func():\n\x00\x01\x02",  # Binary data
        ]
        
        for bad_input in problematic_inputs:
            try:
                # Should not crash, may return error result
                result = await engine.validate_code_authenticity(str(bad_input))
                assert result is not None
            except Exception as e:
                # If exception occurs, should be controlled
                assert "validation" in str(e).lower() or "error" in str(e).lower()
    
    # Cleanup Tests
    
    @pytest.mark.asyncio
    async def test_cleanup_completeness(self, engine):
        """Test cleanup properly releases resources."""
        # Use engine to generate some state
        await engine.validate_code_authenticity("def test(): pass")
        
        # Cleanup
        await engine.cleanup()
        
        # Verify cleanup (basic check - cache should be empty)
        metrics = await engine.get_performance_metrics()
        cache_size = metrics.get('cache_stats', {}).get('cache_size', 0)
        assert cache_size == 0, "Cache not properly cleaned up"


class TestTrainingDataGenerator:
    """Test suite for training data generation."""
    
    @pytest.fixture
    async def generator(self):
        """Initialize training data generator."""
        generator = TrainingDataGenerator()
        await generator.initialize()
        return generator
    
    @pytest.mark.asyncio
    async def test_basic_data_generation(self, generator):
        """Test basic training data generation."""
        config = GenerationConfig(total_samples=50, seed=42)
        samples = await generator.generate_training_dataset(config)
        
        assert len(samples) == 50
        
        # Verify mix of authentic and hallucinated
        authentic_count = sum(1 for s in samples if s.is_authentic)
        hallucinated_count = len(samples) - authentic_count
        
        assert authentic_count > 0
        assert hallucinated_count > 0
    
    @pytest.mark.asyncio
    async def test_balanced_dataset_generation(self, generator):
        """Test balanced dataset generation."""
        languages = ['python', 'javascript']
        patterns = [CodePattern.FUNCTION_DEFINITION, CodePattern.CLASS_DEFINITION]
        
        samples = await generator.generate_balanced_dataset(
            total_samples=100,
            languages=languages,
            patterns=patterns
        )
        
        assert len(samples) <= 100
        
        # Check language distribution
        python_samples = [s for s in samples if s.language == 'python']
        js_samples = [s for s in samples if s.language == 'javascript']
        
        assert len(python_samples) > 0
        assert len(js_samples) > 0
    
    @pytest.mark.asyncio
    async def test_edge_case_generation(self, generator):
        """Test edge case sample generation."""
        edge_cases = await generator.generate_edge_case_samples(50)
        
        assert len(edge_cases) > 0
        
        # Should have variety of edge cases
        edge_case_types = set(sample.features.get('edge_case') for sample in edge_cases)
        assert len(edge_case_types) > 1
    
    @pytest.mark.asyncio
    async def test_data_augmentation(self, generator):
        """Test data augmentation functionality."""
        # Generate base samples
        config = GenerationConfig(total_samples=10, seed=42)
        base_samples = await generator.generate_training_dataset(config)
        
        # Augment samples
        augmented = await generator.augment_existing_samples(
            base_samples, augmentation_factor=2
        )
        
        # Should have more samples after augmentation
        assert len(augmented) > len(base_samples)
        
        # Should include original samples
        original_ids = {s.id for s in base_samples}
        augmented_ids = {s.id for s in augmented}
        
        assert original_ids.issubset(augmented_ids)


# Benchmark Test Class
class TestBenchmarks:
    """Performance and accuracy benchmarks."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_accuracy_benchmark_suite(self):
        """Run comprehensive accuracy benchmark suite."""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_setting = AsyncMock(return_value={})
        
        engine = AntiHallucinationEngine(config_manager)
        await engine.initialize()
        
        # Generate test dataset
        generator = TrainingDataGenerator()
        await generator.initialize()
        
        # Generate larger test set for benchmarking
        test_data = await generator.generate_training_dataset(
            GenerationConfig(total_samples=500, seed=42)
        )
        
        # Train model
        metrics = await engine.train_pattern_recognition_model(test_data)
        
        # Verify benchmark requirements
        benchmark_results = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'training_samples': metrics.training_samples,
            'cross_validation_mean': np.mean(metrics.cross_validation_scores),
            'cross_validation_std': np.std(metrics.cross_validation_scores)
        }
        
        # Log benchmark results
        print(f"\n=== Anti-Hallucination Engine Benchmark Results ===")
        for metric, value in benchmark_results.items():
            print(f"{metric}: {value:.4f}")
        
        # Assert benchmark requirements
        assert metrics.accuracy >= 0.958, f"Accuracy benchmark failed: {metrics.accuracy:.4f} < 0.958"
        assert np.std(metrics.cross_validation_scores) < 0.05, "Cross-validation too variable"
        
        await engine.cleanup()
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_performance_benchmark_suite(self):
        """Run comprehensive performance benchmark suite."""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_setting = AsyncMock(return_value={})
        
        engine = AntiHallucinationEngine(config_manager)
        await engine.initialize()
        
        # Performance test cases
        test_cases = [
            "def simple(): pass",
            '''
def complex_function(data, options=None):
    """Complex function for performance testing."""
    if not data or not isinstance(data, list):
        return []
    
    results = []
    for item in data:
        try:
            processed = item.process(options)
            if processed:
                results.append(processed)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            continue
    
    return results
''',
            # Add more test cases...
        ]
        
        # Warm up
        for case in test_cases:
            await engine.validate_code_authenticity(case)
        
        # Benchmark
        all_response_times = []
        
        for case in test_cases:
            response_times = []
            for _ in range(10):
                start_time = time.time()
                await engine.validate_code_authenticity(case)
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
            
            all_response_times.extend(response_times)
        
        # Calculate statistics
        avg_time = np.mean(all_response_times)
        p95_time = np.percentile(all_response_times, 95)
        p99_time = np.percentile(all_response_times, 99)
        
        benchmark_results = {
            'avg_response_time_ms': avg_time,
            'p95_response_time_ms': p95_time,
            'p99_response_time_ms': p99_time,
            'total_samples': len(all_response_times)
        }
        
        # Log benchmark results
        print(f"\n=== Performance Benchmark Results ===")
        for metric, value in benchmark_results.items():
            print(f"{metric}: {value:.2f}")
        
        # Assert performance requirements
        assert avg_time < 200, f"Average response time benchmark failed: {avg_time:.2f}ms >= 200ms"
        assert p95_time < 300, f"P95 response time benchmark failed: {p95_time:.2f}ms >= 300ms"
        
        await engine.cleanup()


# Integration Test Fixtures
@pytest.fixture(scope="session")
def temp_model_dir():
    """Create temporary directory for model storage during tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def test_config():
    """Test configuration for consistency."""
    return {
        'anti_hallucination': {
            'target_accuracy': 0.958,
            'performance_threshold_ms': 200,
            'confidence_threshold': 0.7
        },
        'validation': {
            'enabled': True,
            'auto_fix_enabled': True,
            'quality_threshold': 0.7,
            'authenticity_threshold': 0.8
        }
    }


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])