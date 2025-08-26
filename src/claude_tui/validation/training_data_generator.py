"""
Training Data Generator - Synthetic data generation for ML models.

Generates high-quality synthetic training data for anti-hallucination models:
- Code pattern generation with variations
- Authentic vs hallucinated code samples  
- Multi-language support (Python, JavaScript, TypeScript)
- Quality scoring and labeling
- Data augmentation techniques
- Balanced dataset creation
"""

import asyncio
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import json

from src.claude_tui.validation.anti_hallucination_engine import CodeSample

logger = logging.getLogger(__name__)


class CodePattern(Enum):
    """Types of code patterns to generate."""
    FUNCTION_DEFINITION = "function_definition"
    CLASS_DEFINITION = "class_definition"
    ERROR_HANDLING = "error_handling"
    DATA_PROCESSING = "data_processing"
    API_ENDPOINT = "api_endpoint"
    ALGORITHM_IMPLEMENTATION = "algorithm_implementation"
    DATABASE_OPERATION = "database_operation"
    CONFIGURATION_SETUP = "configuration_setup"
    TEST_CASE = "test_case"
    UTILITY_FUNCTION = "utility_function"


class QualityLevel(Enum):
    """Quality levels for generated code."""
    HIGH = "high"          # Production-ready, complete implementation
    MEDIUM = "medium"      # Functional but may have minor issues
    LOW = "low"           # Basic implementation, needs improvement
    PLACEHOLDER = "placeholder"  # Contains TODO/FIXME/placeholders


@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    total_samples: int = 1000
    language_distribution: Dict[str, float] = None
    pattern_distribution: Dict[CodePattern, float] = None
    quality_distribution: Dict[QualityLevel, float] = None
    authenticity_ratio: float = 0.6  # 60% authentic, 40% hallucinated
    variation_factor: float = 0.3    # Amount of variation to introduce
    include_edge_cases: bool = True
    seed: Optional[int] = None


class TrainingDataGenerator:
    """
    Advanced training data generator for anti-hallucination models.
    
    Generates diverse, realistic code samples with proper labeling
    for training ML models to detect hallucinations and quality issues.
    """
    
    def __init__(self):
        """Initialize the training data generator."""
        # Template libraries
        self.python_templates = {}
        self.javascript_templates = {}
        self.typescript_templates = {}
        
        # Variation patterns
        self.variation_patterns = {
            'variable_names': ['data', 'info', 'content', 'payload', 'input', 'params'],
            'function_names': ['process', 'handle', 'execute', 'run', 'perform'],
            'class_names': ['Handler', 'Processor', 'Manager', 'Controller', 'Service'],
            'error_types': ['ValueError', 'TypeError', 'RuntimeError', 'Exception'],
        }
        
        # Hallucination injection patterns
        self.hallucination_patterns = [
            'todo_placeholder',
            'incomplete_implementation', 
            'broken_logic',
            'missing_imports',
            'undefined_variables',
            'incorrect_syntax',
            'placeholder_comments',
            'empty_methods'
        ]
        
        # Quality indicators
        self.quality_indicators = {
            QualityLevel.HIGH: {
                'has_docstrings': 0.9,
                'has_error_handling': 0.8,
                'has_type_hints': 0.7,
                'proper_naming': 0.9,
                'no_placeholders': 1.0
            },
            QualityLevel.MEDIUM: {
                'has_docstrings': 0.6,
                'has_error_handling': 0.5,
                'has_type_hints': 0.4,
                'proper_naming': 0.7,
                'no_placeholders': 0.9
            },
            QualityLevel.LOW: {
                'has_docstrings': 0.2,
                'has_error_handling': 0.2,
                'has_type_hints': 0.1,
                'proper_naming': 0.5,
                'no_placeholders': 0.6
            },
            QualityLevel.PLACEHOLDER: {
                'has_docstrings': 0.1,
                'has_error_handling': 0.1,
                'has_type_hints': 0.0,
                'proper_naming': 0.3,
                'no_placeholders': 0.0
            }
        }
        
        logger.info("Training data generator initialized")
    
    async def initialize(self) -> None:
        """Initialize the generator with templates and patterns."""
        logger.info("Initializing training data generator")
        
        try:
            # Load code templates
            await self._load_code_templates()
            
            # Initialize variation engines
            await self._initialize_variation_engines()
            
            logger.info("Training data generator ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize training data generator: {e}")
            raise
    
    async def generate_training_dataset(
        self, 
        config: GenerationConfig = None
    ) -> List[CodeSample]:
        """
        Generate a comprehensive training dataset.
        
        Args:
            config: Generation configuration
            
        Returns:
            List of labeled code samples
        """
        config = config or GenerationConfig()
        
        if config.seed:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        logger.info(f"Generating training dataset with {config.total_samples} samples")
        
        # Set default distributions if not provided
        config = self._set_default_distributions(config)
        
        samples = []
        
        # Calculate samples per category
        authentic_count = int(config.total_samples * config.authenticity_ratio)
        hallucinated_count = config.total_samples - authentic_count
        
        logger.info(f"Generating {authentic_count} authentic and {hallucinated_count} hallucinated samples")
        
        # Generate authentic samples
        authentic_samples = await self._generate_authentic_samples(authentic_count, config)
        samples.extend(authentic_samples)
        
        # Generate hallucinated samples
        hallucinated_samples = await self._generate_hallucinated_samples(hallucinated_count, config)
        samples.extend(hallucinated_samples)
        
        # Shuffle samples
        random.shuffle(samples)
        
        logger.info(f"Generated {len(samples)} total training samples")
        return samples
    
    async def generate_balanced_dataset(
        self,
        total_samples: int = 1000,
        languages: List[str] = None,
        patterns: List[CodePattern] = None
    ) -> List[CodeSample]:
        """
        Generate a balanced dataset across languages and patterns.
        
        Args:
            total_samples: Total number of samples to generate
            languages: Languages to include (default: ['python', 'javascript'])
            patterns: Code patterns to include (default: all patterns)
            
        Returns:
            Balanced list of code samples
        """
        languages = languages or ['python', 'javascript']
        patterns = patterns or list(CodePattern)
        
        samples_per_combination = total_samples // (len(languages) * len(patterns) * 2)  # 2 for authentic/hallucinated
        
        samples = []
        
        for language in languages:
            for pattern in patterns:
                # Generate authentic samples
                authentic_samples = await self._generate_pattern_samples(
                    count=samples_per_combination,
                    language=language,
                    pattern=pattern,
                    is_authentic=True
                )
                samples.extend(authentic_samples)
                
                # Generate hallucinated samples
                hallucinated_samples = await self._generate_pattern_samples(
                    count=samples_per_combination,
                    language=language,
                    pattern=pattern,
                    is_authentic=False
                )
                samples.extend(hallucinated_samples)
        
        # Fill remaining slots randomly
        remaining_count = total_samples - len(samples)
        if remaining_count > 0:
            additional_samples = await self.generate_training_dataset(
                GenerationConfig(total_samples=remaining_count)
            )
            samples.extend(additional_samples)
        
        random.shuffle(samples)
        logger.info(f"Generated balanced dataset with {len(samples)} samples")
        return samples[:total_samples]
    
    async def augment_existing_samples(
        self,
        existing_samples: List[CodeSample],
        augmentation_factor: int = 3
    ) -> List[CodeSample]:
        """
        Augment existing samples with variations.
        
        Args:
            existing_samples: Existing code samples
            augmentation_factor: Number of variations per sample
            
        Returns:
            Augmented sample list
        """
        logger.info(f"Augmenting {len(existing_samples)} samples with factor {augmentation_factor}")
        
        augmented_samples = []
        
        for sample in existing_samples:
            # Add original sample
            augmented_samples.append(sample)
            
            # Generate variations
            for i in range(augmentation_factor):
                variation = await self._create_sample_variation(sample, i)
                if variation:
                    augmented_samples.append(variation)
        
        logger.info(f"Augmented to {len(augmented_samples)} total samples")
        return augmented_samples
    
    async def generate_edge_case_samples(self, count: int = 200) -> List[CodeSample]:
        """
        Generate edge case samples for robust testing.
        
        Args:
            count: Number of edge cases to generate
            
        Returns:
            List of edge case samples
        """
        logger.info(f"Generating {count} edge case samples")
        
        edge_cases = []
        
        # Edge case categories
        categories = [
            'empty_code',
            'single_line',
            'very_long_function',
            'deeply_nested',
            'unicode_characters',
            'mixed_indentation',
            'syntax_errors',
            'incomplete_strings',
            'circular_imports',
            'recursive_functions'
        ]
        
        samples_per_category = count // len(categories)
        
        for category in categories:
            category_samples = await self._generate_edge_case_category(
                category, samples_per_category
            )
            edge_cases.extend(category_samples)
        
        logger.info(f"Generated {len(edge_cases)} edge case samples")
        return edge_cases
    
    async def save_dataset(
        self,
        samples: List[CodeSample],
        output_path: Path,
        format: str = 'jsonl'
    ) -> None:
        """
        Save generated dataset to file.
        
        Args:
            samples: Code samples to save
            output_path: Output file path
            format: Output format ('jsonl', 'json', 'csv')
        """
        logger.info(f"Saving {len(samples)} samples to {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    sample_dict = {
                        'id': sample.id,
                        'content': sample.content,
                        'is_authentic': sample.is_authentic,
                        'has_placeholders': sample.has_placeholders,
                        'quality_score': sample.quality_score,
                        'language': sample.language,
                        'complexity': sample.complexity,
                        'features': sample.features,
                        'created_at': sample.created_at.isoformat()
                    }
                    f.write(json.dumps(sample_dict) + '\n')
        
        elif format == 'json':
            samples_data = []
            for sample in samples:
                samples_data.append({
                    'id': sample.id,
                    'content': sample.content,
                    'is_authentic': sample.is_authentic,
                    'has_placeholders': sample.has_placeholders,
                    'quality_score': sample.quality_score,
                    'language': sample.language,
                    'complexity': sample.complexity,
                    'features': sample.features,
                    'created_at': sample.created_at.isoformat()
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(samples_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_path}")
    
    # Private implementation methods
    
    async def _load_code_templates(self) -> None:
        """Load code templates for different languages and patterns."""
        
        # Python templates
        self.python_templates = {
            CodePattern.FUNCTION_DEFINITION: [
                '''
def {function_name}({parameters}):
    """{docstring}"""
    {implementation}
    return {return_value}
''',
                '''
async def {function_name}({parameters}):
    """{docstring}"""
    try:
        {implementation}
        return {return_value}
    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}")
        raise
'''
            ],
            
            CodePattern.CLASS_DEFINITION: [
                '''
class {class_name}:
    """{docstring}"""
    
    def __init__(self, {init_params}):
        """{init_docstring}"""
        {init_implementation}
    
    def {method_name}(self, {method_params}):
        """{method_docstring}"""
        {method_implementation}
        return {method_return}
'''
            ],
            
            CodePattern.ERROR_HANDLING: [
                '''
try:
    {main_implementation}
except {exception_type} as e:
    logger.error(f"{error_message}: {{e}}")
    {error_handling}
except Exception as e:
    logger.error(f"Unexpected error: {{e}}")
    raise
finally:
    {cleanup_code}
'''
            ],
            
            CodePattern.DATA_PROCESSING: [
                '''
def process_{data_type}(data: List[{type_hint}]) -> {return_type}:
    """{docstring}"""
    if not data:
        return {empty_return}
    
    processed = []
    for item in data:
        try:
            {processing_logic}
            processed.append(result)
        except Exception as e:
            logger.warning(f"Failed to process item {{item}}: {{e}}")
            continue
    
    return processed
'''
            ]
        }
        
        # JavaScript templates
        self.javascript_templates = {
            CodePattern.FUNCTION_DEFINITION: [
                '''
function {function_name}({parameters}) {
    // {docstring}
    {implementation}
    return {return_value};
}
''',
                '''
async function {function_name}({parameters}) {
    // {docstring}
    try {
        {implementation}
        return {return_value};
    } catch (error) {
        console.error(`Error in {function_name}:`, error);
        throw error;
    }
}
'''
            ],
            
            CodePattern.CLASS_DEFINITION: [
                '''
class {class_name} {
    // {docstring}
    
    constructor({constructor_params}) {
        {constructor_implementation}
    }
    
    {method_name}({method_params}) {
        // {method_docstring}
        {method_implementation}
        return {method_return};
    }
}
'''
            ],
            
            CodePattern.ERROR_HANDLING: [
                '''
try {
    {main_implementation}
} catch (error) {
    if (error instanceof {error_type}) {
        console.error('{error_message}:', error);
        {error_handling}
    } else {
        console.error('Unexpected error:', error);
        throw error;
    }
} finally {
    {cleanup_code}
}
'''
            ]
        }
    
    async def _initialize_variation_engines(self) -> None:
        """Initialize engines for creating code variations."""
        # This would initialize more sophisticated variation engines
        # For now, we use the simple pattern substitution approach
        pass
    
    def _set_default_distributions(self, config: GenerationConfig) -> GenerationConfig:
        """Set default distributions if not provided."""
        if config.language_distribution is None:
            config.language_distribution = {
                'python': 0.6,
                'javascript': 0.3,
                'typescript': 0.1
            }
        
        if config.pattern_distribution is None:
            config.pattern_distribution = {
                CodePattern.FUNCTION_DEFINITION: 0.3,
                CodePattern.CLASS_DEFINITION: 0.2,
                CodePattern.ERROR_HANDLING: 0.15,
                CodePattern.DATA_PROCESSING: 0.15,
                CodePattern.API_ENDPOINT: 0.1,
                CodePattern.ALGORITHM_IMPLEMENTATION: 0.1
            }
        
        if config.quality_distribution is None:
            config.quality_distribution = {
                QualityLevel.HIGH: 0.3,
                QualityLevel.MEDIUM: 0.4,
                QualityLevel.LOW: 0.2,
                QualityLevel.PLACEHOLDER: 0.1
            }
        
        return config
    
    async def _generate_authentic_samples(
        self, 
        count: int, 
        config: GenerationConfig
    ) -> List[CodeSample]:
        """Generate authentic code samples."""
        samples = []
        
        for i in range(count):
            # Select language, pattern, and quality based on distributions
            language = self._sample_from_distribution(config.language_distribution)
            pattern = self._sample_from_distribution(config.pattern_distribution)
            quality = self._sample_from_distribution(config.quality_distribution)
            
            # Generate sample
            sample = await self._generate_single_sample(
                sample_id=f"auth_{i}",
                language=language,
                pattern=pattern,
                quality=quality,
                is_authentic=True
            )
            
            if sample:
                samples.append(sample)
        
        return samples
    
    async def _generate_hallucinated_samples(
        self, 
        count: int, 
        config: GenerationConfig
    ) -> List[CodeSample]:
        """Generate hallucinated code samples."""
        samples = []
        
        for i in range(count):
            # Hallucinated samples tend to have lower quality
            language = self._sample_from_distribution(config.language_distribution)
            pattern = self._sample_from_distribution(config.pattern_distribution)
            
            # Bias toward lower quality for hallucinated samples
            quality_weights = {
                QualityLevel.HIGH: 0.05,
                QualityLevel.MEDIUM: 0.15,
                QualityLevel.LOW: 0.35,
                QualityLevel.PLACEHOLDER: 0.45
            }
            quality = self._sample_from_distribution(quality_weights)
            
            # Generate base sample
            sample = await self._generate_single_sample(
                sample_id=f"hall_{i}",
                language=language,
                pattern=pattern,
                quality=quality,
                is_authentic=False
            )
            
            if sample:
                # Inject hallucination patterns
                sample = await self._inject_hallucination_patterns(sample)
                samples.append(sample)
        
        return samples
    
    async def _generate_pattern_samples(
        self,
        count: int,
        language: str,
        pattern: CodePattern,
        is_authentic: bool
    ) -> List[CodeSample]:
        """Generate samples for specific language/pattern combination."""
        samples = []
        
        for i in range(count):
            quality = random.choice(list(QualityLevel))
            if not is_authentic:
                # Bias toward lower quality for hallucinated samples
                quality = random.choices(
                    list(QualityLevel),
                    weights=[0.1, 0.2, 0.3, 0.4]  # Favor lower quality
                )[0]
            
            sample = await self._generate_single_sample(
                sample_id=f"{pattern.value}_{language}_{i}",
                language=language,
                pattern=pattern,
                quality=quality,
                is_authentic=is_authentic
            )
            
            if sample and not is_authentic:
                sample = await self._inject_hallucination_patterns(sample)
            
            if sample:
                samples.append(sample)
        
        return samples
    
    async def _generate_single_sample(
        self,
        sample_id: str,
        language: str,
        pattern: CodePattern,
        quality: QualityLevel,
        is_authentic: bool
    ) -> Optional[CodeSample]:
        """Generate a single code sample."""
        try:
            # Get appropriate templates
            templates = self._get_templates_for_language_pattern(language, pattern)
            if not templates:
                return None
            
            # Select random template
            template = random.choice(templates)
            
            # Generate substitutions
            substitutions = self._generate_substitutions(language, pattern, quality)
            
            # Apply substitutions to template
            code = self._apply_substitutions(template, substitutions)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(code, quality)
            
            # Determine if has placeholders
            has_placeholders = quality == QualityLevel.PLACEHOLDER or 'TODO' in code or 'FIXME' in code
            
            # Calculate complexity
            complexity = self._calculate_complexity(code)
            
            sample = CodeSample(
                id=sample_id,
                content=code,
                is_authentic=is_authentic,
                has_placeholders=has_placeholders,
                quality_score=quality_score,
                language=language,
                complexity=complexity,
                features={'pattern': pattern.value, 'quality': quality.value}
            )
            
            return sample
            
        except Exception as e:
            logger.warning(f"Failed to generate sample {sample_id}: {e}")
            return None
    
    def _get_templates_for_language_pattern(
        self, 
        language: str, 
        pattern: CodePattern
    ) -> List[str]:
        """Get templates for specific language and pattern."""
        if language == 'python':
            return self.python_templates.get(pattern, [])
        elif language == 'javascript':
            return self.javascript_templates.get(pattern, [])
        elif language == 'typescript':
            # Use JavaScript templates with type annotations
            js_templates = self.javascript_templates.get(pattern, [])
            return [self._add_typescript_annotations(template) for template in js_templates]
        else:
            return []
    
    def _generate_substitutions(
        self, 
        language: str, 
        pattern: CodePattern, 
        quality: QualityLevel
    ) -> Dict[str, str]:
        """Generate substitutions for template variables."""
        substitutions = {}
        
        # Basic substitutions
        substitutions.update({
            'function_name': random.choice(['process_data', 'handle_request', 'calculate_result', 'validate_input']),
            'class_name': random.choice(['DataProcessor', 'RequestHandler', 'Calculator', 'Validator']),
            'parameters': random.choice(['data', 'data, options=None', 'request, context']),
            'return_value': random.choice(['result', 'processed_data', 'None']),
            'variable_name': random.choice(self.variation_patterns['variable_names']),
            'data_type': random.choice(['user', 'order', 'product', 'message']),
            'type_hint': random.choice(['str', 'int', 'dict', 'Any']),
            'return_type': random.choice(['List[str]', 'Dict[str, Any]', 'Optional[str]']),
        })
        
        # Quality-dependent substitutions
        quality_config = self.quality_indicators[quality]
        
        if random.random() < quality_config['has_docstrings']:
            substitutions['docstring'] = 'Process the input data and return results.'
        else:
            substitutions['docstring'] = ''
        
        if random.random() < quality_config['no_placeholders']:
            substitutions['implementation'] = 'result = data.process()\nvalidate_result(result)'
        else:
            substitutions['implementation'] = 'pass  # TODO: Implement this method'
        
        # Error handling
        if random.random() < quality_config['has_error_handling']:
            substitutions['error_handling'] = 'return None'
            substitutions['exception_type'] = 'ValueError'
        else:
            substitutions['error_handling'] = 'pass'
            substitutions['exception_type'] = 'Exception'
        
        return substitutions
    
    def _apply_substitutions(self, template: str, substitutions: Dict[str, str]) -> str:
        """Apply substitutions to template."""
        code = template
        
        for key, value in substitutions.items():
            code = code.replace(f'{{{key}}}', value)
        
        # Clean up formatting
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)  # Remove excessive blank lines
        code = code.strip()
        
        return code
    
    async def _inject_hallucination_patterns(self, sample: CodeSample) -> CodeSample:
        """Inject hallucination patterns into a sample."""
        code = sample.content
        
        # Select random hallucination patterns to apply
        patterns_to_apply = random.sample(
            self.hallucination_patterns, 
            k=random.randint(1, 3)
        )
        
        for pattern in patterns_to_apply:
            code = await self._apply_hallucination_pattern(code, pattern, sample.language)
        
        # Update sample
        sample.content = code
        sample.has_placeholders = True
        sample.quality_score *= 0.5  # Reduce quality score
        
        return sample
    
    async def _apply_hallucination_pattern(
        self, 
        code: str, 
        pattern: str, 
        language: str
    ) -> str:
        """Apply specific hallucination pattern to code."""
        
        if pattern == 'todo_placeholder':
            # Add TODO comments
            lines = code.split('\n')
            insert_pos = random.randint(1, max(1, len(lines) - 1))
            comment_prefix = '#' if language == 'python' else '//'
            lines.insert(insert_pos, f'    {comment_prefix} TODO: Implement this functionality')
            return '\n'.join(lines)
        
        elif pattern == 'incomplete_implementation':
            # Replace implementation with pass/placeholder
            if language == 'python':
                code = re.sub(r'(\s+)return\s+\w+', r'\1pass  # TODO: Return proper value', code)
            else:
                code = re.sub(r'(\s+)return\s+\w+;', r'\1// TODO: Return proper value\n\1return null;', code)
        
        elif pattern == 'broken_logic':
            # Introduce logical errors
            code = re.sub(r'if\s+(\w+):', r'if not \1:  # Inverted condition - bug!', code)
        
        elif pattern == 'missing_imports':
            # Remove import statements
            lines = code.split('\n')
            lines = [line for line in lines if not line.strip().startswith(('import ', 'from '))]
            return '\n'.join(lines)
        
        elif pattern == 'undefined_variables':
            # Use undefined variables
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'result' in line:
                    lines[i] = line.replace('result', 'undefined_variable')
                    break
            return '\n'.join(lines)
        
        elif pattern == 'placeholder_comments':
            # Add placeholder comments
            comment_prefix = '#' if language == 'python' else '//'
            code += f'\n    {comment_prefix} FIXME: This needs proper implementation'
        
        elif pattern == 'empty_methods':
            # Make methods empty
            if language == 'python':
                code = re.sub(r'(def\s+\w+\([^)]*\):)\s*[^#\n]*', r'\1\n    pass', code)
            else:
                code = re.sub(r'(\w+\([^)]*\)\s*\{)[^}]*\}', r'\1\n    // Empty method\n}', code)
        
        return code
    
    async def _create_sample_variation(self, original: CodeSample, variation_index: int) -> Optional[CodeSample]:
        """Create a variation of an existing sample."""
        try:
            code = original.content
            
            # Apply variations based on index
            if variation_index == 0:
                # Rename variables
                code = self._rename_variables(code)
            elif variation_index == 1:
                # Change function names
                code = self._rename_functions(code)
            elif variation_index == 2:
                # Add/remove comments
                code = self._modify_comments(code)
            
            variation = CodeSample(
                id=f"{original.id}_var{variation_index}",
                content=code,
                is_authentic=original.is_authentic,
                has_placeholders=original.has_placeholders,
                quality_score=original.quality_score * random.uniform(0.9, 1.1),  # Slight variation
                language=original.language,
                complexity=original.complexity,
                features=original.features.copy()
            )
            
            return variation
            
        except Exception as e:
            logger.warning(f"Failed to create variation of {original.id}: {e}")
            return None
    
    async def _generate_edge_case_category(
        self, 
        category: str, 
        count: int
    ) -> List[CodeSample]:
        """Generate edge cases for a specific category."""
        samples = []
        
        for i in range(count):
            sample = None
            
            if category == 'empty_code':
                sample = CodeSample(
                    id=f"edge_empty_{i}",
                    content="",
                    is_authentic=False,
                    has_placeholders=False,
                    quality_score=0.0,
                    language='python',
                    complexity=0.0,
                    features={'edge_case': category}
                )
            
            elif category == 'single_line':
                code = random.choice([
                    "x = 1",
                    "print('hello')",
                    "return None",
                    "pass"
                ])
                sample = CodeSample(
                    id=f"edge_single_{i}",
                    content=code,
                    is_authentic=True,
                    has_placeholders=False,
                    quality_score=0.3,
                    language='python',
                    complexity=1.0,
                    features={'edge_case': category}
                )
            
            elif category == 'very_long_function':
                # Generate a very long function
                lines = ['def very_long_function(data):']
                for j in range(50):
                    lines.append(f'    step_{j} = data.process_step_{j}()')
                lines.append('    return final_result')
                
                sample = CodeSample(
                    id=f"edge_long_{i}",
                    content='\n'.join(lines),
                    is_authentic=False,  # Unrealistically long
                    has_placeholders=False,
                    quality_score=0.2,
                    language='python',
                    complexity=25.0,
                    features={'edge_case': category}
                )
            
            # Add more edge case categories as needed
            
            if sample:
                samples.append(sample)
        
        return samples
    
    def _sample_from_distribution(self, distribution: Dict) -> Any:
        """Sample from a probability distribution."""
        choices = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(choices, weights=weights)[0]
    
    def _calculate_quality_score(self, code: str, quality: QualityLevel) -> float:
        """Calculate quality score for generated code."""
        base_scores = {
            QualityLevel.HIGH: 0.9,
            QualityLevel.MEDIUM: 0.7,
            QualityLevel.LOW: 0.4,
            QualityLevel.PLACEHOLDER: 0.1
        }
        
        base_score = base_scores[quality]
        
        # Add some random variation
        variation = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_score + variation))
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate rough complexity score for code."""
        lines = len(code.split('\n'))
        functions = len(re.findall(r'def\s+\w+|function\s+\w+', code, re.IGNORECASE))
        classes = len(re.findall(r'class\s+\w+', code, re.IGNORECASE))
        
        complexity = lines * 0.1 + functions * 2 + classes * 3
        return min(complexity, 50.0)  # Cap at 50
    
    def _add_typescript_annotations(self, js_template: str) -> str:
        """Add TypeScript type annotations to JavaScript template."""
        # Simple type annotation addition
        ts_template = js_template.replace(
            'function {function_name}({parameters})',
            'function {function_name}({parameters}): {return_type}'
        )
        ts_template = ts_template.replace('{return_type}', 'any')
        return ts_template
    
    def _rename_variables(self, code: str) -> str:
        """Rename variables in code for variation."""
        substitutions = {
            'data': random.choice(['info', 'content', 'payload']),
            'result': random.choice(['output', 'response', 'value']),
            'item': random.choice(['element', 'entry', 'record'])
        }
        
        for old_name, new_name in substitutions.items():
            code = re.sub(rf'\b{old_name}\b', new_name, code)
        
        return code
    
    def _rename_functions(self, code: str) -> str:
        """Rename functions in code for variation."""
        substitutions = {
            'process': random.choice(['handle', 'execute', 'run']),
            'calculate': random.choice(['compute', 'determine', 'evaluate']),
            'validate': random.choice(['check', 'verify', 'confirm'])
        }
        
        for old_name, new_name in substitutions.items():
            code = re.sub(rf'\b{old_name}\b', new_name, code)
        
        return code
    
    def _modify_comments(self, code: str) -> str:
        """Modify comments in code for variation."""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if '#' in line or '//' in line:
                # Randomly remove or modify comments
                if random.random() < 0.3:
                    # Remove comment
                    lines[i] = re.sub(r'\s*#.*$', '', line)
                    lines[i] = re.sub(r'\s*//.*$', '', lines[i])
        
        return '\n'.join(lines)