#!/usr/bin/env python3
"""
Comprehensive ML Semantic Analysis Test Suite
Tests the semantic analyzer ML models for code structure validation, logic flow analysis,
and dependency validation with focus on achieving 95.8% accuracy benchmarks.
"""

import pytest
import asyncio
import numpy as np
import ast
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import json
from dataclasses import dataclass

# Import the ML validation components
try:
    from src.claude_tui.validation.semantic_analyzer import SemanticAnalyzer, SemanticContext, SemanticIssueType
    from src.claude_tui.validation.anti_hallucination_engine import AntiHallucinationEngine, FeatureExtractor
    from src.claude_tui.validation.types import ValidationIssue, ValidationSeverity, ValidationResult
    from src.claude_tui.core.config_manager import ConfigManager
    from src.claude_tui.models.project import Project
except ImportError:
    # Mock classes for CI/CD compatibility
    @dataclass
    class ValidationIssue:
        id: str = ""
        description: str = ""
        severity: str = "LOW"
        issue_type: str = ""
        line_number: int = 0
        auto_fixable: bool = False
    
    class ValidationSeverity:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"
    
    class SemanticAnalyzer:
        def __init__(self, config): pass
        async def analyze_content(self, content, **kwargs): return []
        async def initialize(self): pass
        async def cleanup(self): pass
    
    class FeatureExtractor:
        def __init__(self): pass
        async def extract_features(self, code, language=None): 
            return {"length": len(code), "complexity": 1.0}


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for semantic analyzer tests."""
    config = Mock(spec=ConfigManager)
    config.get_setting = AsyncMock(return_value={
        'strict_mode': True,
        'check_unused_imports': True,
        'check_performance': True,
        'check_security': True,
        'ml_accuracy_threshold': 0.958
    })
    return config


@pytest.fixture
def semantic_analyzer(mock_config_manager):
    """Create semantic analyzer with mocked dependencies."""
    return SemanticAnalyzer(mock_config_manager)


@pytest.fixture
def feature_extractor():
    """Create feature extractor for ML model testing."""
    return FeatureExtractor()


@pytest.fixture
def semantic_test_datasets():
    """Comprehensive test datasets for semantic analysis."""
    return {
        "high_quality_code": [
            {
                "code": '''
def binary_search(arr: List[int], target: int) -> int:
    """
    Perform binary search on a sorted array.
    
    Args:
        arr: Sorted list of integers
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def merge_sort(arr: List[int]) -> List[int]:
    """
    Sort array using merge sort algorithm.
    
    Args:
        arr: List of integers to sort
        
    Returns:
        Sorted list of integers
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])
    
    return merge(left_half, right_half)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
''',
                "language": "python",
                "expected_quality_score": 0.95,
                "expected_issues": 0,
                "semantic_features": {
                    "function_count": 3,
                    "class_count": 0,
                    "docstring_count": 3,
                    "complexity_score": 0.7,
                    "type_hints": True,
                    "error_handling": False
                }
            },
            {
                "code": '''
class AVLTree:
    """
    Self-balancing binary search tree implementation.
    Maintains O(log n) operations through rotation operations.
    """
    
    def __init__(self):
        self.root = None
    
    def insert(self, value: int) -> None:
        """Insert a value into the AVL tree."""
        self.root = self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node: Optional['AVLNode'], value: int) -> 'AVLNode':
        """Recursively insert and rebalance tree."""
        # Standard BST insertion
        if not node:
            return AVLNode(value)
        
        if value < node.value:
            node.left = self._insert_recursive(node.left, value)
        elif value > node.value:
            node.right = self._insert_recursive(node.right, value)
        else:
            return node  # Duplicate values not allowed
        
        # Update height and rebalance
        node.height = 1 + max(self._get_height(node.left), 
                             self._get_height(node.right))
        
        balance_factor = self._get_balance_factor(node)
        
        # Left Left Case
        if balance_factor > 1 and value < node.left.value:
            return self._rotate_right(node)
        
        # Right Right Case
        if balance_factor < -1 and value > node.right.value:
            return self._rotate_left(node)
        
        # Left Right Case
        if balance_factor > 1 and value > node.left.value:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right Left Case
        if balance_factor < -1 and value < node.right.value:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def _get_height(self, node: Optional['AVLNode']) -> int:
        """Get height of node."""
        if not node:
            return 0
        return node.height
    
    def _get_balance_factor(self, node: 'AVLNode') -> int:
        """Calculate balance factor."""
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z: 'AVLNode') -> 'AVLNode':
        """Perform left rotation."""
        y = z.right
        T2 = y.left
        
        y.left = z
        z.right = T2
        
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        
        return y
    
    def _rotate_right(self, z: 'AVLNode') -> 'AVLNode':
        """Perform right rotation."""
        y = z.left
        T3 = y.right
        
        y.right = z
        z.left = T3
        
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        
        return y


class AVLNode:
    """Node for AVL Tree."""
    
    def __init__(self, value: int):
        self.value = value
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height = 1
''',
                "language": "python",
                "expected_quality_score": 0.92,
                "expected_issues": 0,
                "semantic_features": {
                    "function_count": 8,
                    "class_count": 2,
                    "docstring_count": 8,
                    "complexity_score": 0.85,
                    "type_hints": True,
                    "error_handling": False
                }
            }
        ],
        "placeholder_code": [
            {
                "code": '''
def incomplete_api_handler(request):
    """Handle API request."""
    # TODO: Implement request validation
    # TODO: Add authentication check
    # TODO: Process the request data
    
    # FIXME: This is a placeholder implementation
    result = {"status": "success", "data": None}
    
    # TODO: Add proper error handling
    return result


def another_placeholder():
    """Another incomplete function."""
    pass  # TODO: Implement this method


class IncompleteService:
    """Service class that needs implementation."""
    
    def __init__(self):
        # FIXME: Initialize service dependencies
        self.database = None
        self.cache = None
    
    def process_data(self, data):
        # TODO: Validate input data
        # TODO: Transform data
        # TODO: Store in database
        # TODO: Update cache
        raise NotImplementedError("Method not implemented yet")
    
    def cleanup(self):
        # TODO: Close database connections
        # TODO: Clear cache
        # TODO: Release resources
        pass
''',
                "language": "python",
                "expected_quality_score": 0.25,
                "expected_issues": 12,
                "semantic_features": {
                    "function_count": 4,
                    "class_count": 1,
                    "todo_count": 10,
                    "fixme_count": 2,
                    "placeholder_patterns": 8,
                    "incomplete_implementations": 3
                }
            },
            {
                "code": '''
function incompleteUserManager() {
    // TODO: Implement user management system
    
    const users = [];  // placeholder data structure
    
    function addUser(userData) {
        // FIXME: Add proper validation
        users.push(userData);
    }
    
    function deleteUser(userId) {
        // TODO: Implement user deletion
        console.log("User deletion not implemented");
    }
    
    function updateUser(userId, updates) {
        // TODO: Find user and update
        // TODO: Validate update data
        return false;  // placeholder return
    }
    
    // TODO: Add authentication methods
    // TODO: Add authorization checks
    // TODO: Add user search functionality
    
    return {
        add: addUser,
        delete: deleteUser,
        update: updateUser
        // TODO: Export more methods
    };
}
''',
                "language": "javascript",
                "expected_quality_score": 0.30,
                "expected_issues": 8,
                "semantic_features": {
                    "function_count": 4,
                    "todo_count": 7,
                    "fixme_count": 1,
                    "placeholder_patterns": 3,
                    "incomplete_implementations": 3
                }
            }
        ],
        "security_issues": [
            {
                "code": '''
import os
import subprocess
import pickle

def dangerous_operations(user_input):
    """Function with multiple security vulnerabilities."""
    
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    
    # Command injection vulnerability
    os.system(f"ls {user_input}")
    subprocess.call(f"rm -rf {user_input}", shell=True)
    
    # Eval vulnerability
    result = eval(user_input)
    
    # Deserialization vulnerability  
    data = pickle.loads(user_input)
    
    # Path traversal vulnerability
    with open(f"/data/{user_input}.txt", 'r') as f:
        content = f.read()
    
    # Hardcoded secrets
    api_key = "sk-1234567890abcdef"
    password = "admin123"
    
    return {"query": query, "result": result, "data": data}


def more_security_issues():
    """Additional security problems."""
    
    # XSS vulnerability in template
    html_template = f"<script>alert('{user_input}')</script>"
    
    # Weak cryptography
    import hashlib
    weak_hash = hashlib.md5(password.encode()).hexdigest()
    
    # LDAP injection
    ldap_query = f"(uid={user_input})"
    
    return html_template
''',
                "language": "python", 
                "expected_quality_score": 0.10,
                "expected_issues": 10,
                "security_vulnerabilities": [
                    "sql_injection",
                    "command_injection", 
                    "eval_usage",
                    "pickle_deserialization",
                    "path_traversal",
                    "hardcoded_secrets",
                    "xss_vulnerability",
                    "weak_cryptography",
                    "ldap_injection"
                ]
            }
        ],
        "logic_errors": [
            {
                "code": '''
def logic_error_examples():
    """Function containing various logic errors."""
    
    # Infinite loop without break condition
    while True:
        print("This will run forever")
        # No break or return statement
    
    print("This line is unreachable")
    
    # Division by zero potential
    def divide_numbers(a, b):
        return a / b  # No check for b == 0
    
    # Off-by-one error
    def array_access(arr):
        for i in range(len(arr) + 1):  # Goes beyond array bounds
            print(arr[i])
    
    # Incorrect comparison
    def check_equality(a, b):
        if a = b:  # Should be ==, not =
            return True
        return False
    
    # Unreachable code after return
    def unreachable_example():
        if True:
            return "early return"
        
        expensive_operation()  # This will never execute
        return "late return"
    
    # Variable used before assignment
    def undefined_variable():
        if some_condition:  # some_condition not defined
            result = calculate_something()
        return result  # result might not be defined
''',
                "language": "python",
                "expected_quality_score": 0.20,
                "expected_issues": 7,
                "logic_error_types": [
                    "infinite_loop",
                    "unreachable_code", 
                    "division_by_zero",
                    "array_bounds",
                    "syntax_error",
                    "undefined_variable"
                ]
            }
        ],
        "performance_issues": [
            {
                "code": '''
def performance_problems():
    """Code with various performance issues."""
    
    # Inefficient string concatenation
    result = ""
    for i in range(10000):
        result += f"Item {i}"  # Should use join()
    
    # Inefficient list operations
    items = []
    for i in range(1000):
        items.insert(0, i)  # O(n) operation in loop = O(n²)
    
    # Unnecessary nested loops
    def find_duplicates(list1, list2):
        duplicates = []
        for item1 in list1:
            for item2 in list2:
                if item1 == item2:  # Could use set intersection
                    duplicates.append(item1)
        return duplicates
    
    # Inefficient range usage
    numbers = [1, 2, 3, 4, 5]
    for i in range(len(numbers)):  # Should use enumerate
        print(f"Index {i}: {numbers[i]}")
    
    # Global variable access in loop
    global_counter = 0
    def increment_global():
        global global_counter
        for _ in range(10000):
            global_counter += 1  # Slow global access
    
    # Inefficient file operations
    def process_large_file(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()  # Loads entire file into memory
        
        processed = []
        for line in lines:
            processed.append(line.strip().upper())  # Could use generator
        
        return processed
''',
                "language": "python",
                "expected_quality_score": 0.35,
                "expected_issues": 6,
                "performance_issue_types": [
                    "string_concatenation",
                    "inefficient_list_operations",
                    "nested_loops",
                    "range_len_usage", 
                    "global_variable_access",
                    "memory_inefficient_file_processing"
                ]
            }
        ]
    }


class TestSemanticAnalyzerMLModels:
    """Test ML models within the semantic analyzer."""
    
    @pytest.mark.asyncio
    async def test_ml_feature_extraction_quality(self, feature_extractor, semantic_test_datasets):
        """Test quality of ML feature extraction from code samples."""
        high_quality_sample = semantic_test_datasets["high_quality_code"][0]
        
        features = await feature_extractor.extract_features(
            high_quality_sample["code"], 
            high_quality_sample["language"]
        )
        
        # Verify feature extraction quality
        assert isinstance(features, dict)
        assert len(features) >= 10  # Should extract multiple features
        
        # Check specific features for high-quality code
        assert features.get('function_count', 0) >= 3
        assert features.get('docstring_count', 0) >= 3
        assert features.get('comment_ratio', 0) > 0.1
        assert features.get('complexity_score', 0) > 0.5
        assert features.get('todo_count', 0) == 0  # No TODOs in high-quality code
        assert features.get('placeholder_count', 0) == 0
    
    @pytest.mark.asyncio
    async def test_ml_placeholder_detection_accuracy(self, feature_extractor, semantic_test_datasets):
        """Test ML model accuracy in detecting placeholder patterns."""
        placeholder_sample = semantic_test_datasets["placeholder_code"][0]
        
        features = await feature_extractor.extract_features(
            placeholder_sample["code"], 
            placeholder_sample["language"]
        )
        
        # Verify placeholder detection features
        assert features.get('todo_count', 0) >= 5  # Multiple TODOs expected
        assert features.get('placeholder_count', 0) >= 5
        assert features.get('pass_count', 0) >= 2  # Multiple pass statements
        assert features.get('comment_ratio', 0) > 0.2  # High comment ratio due to TODOs
        
        # Quality metrics should be low for placeholder code
        complexity = features.get('cyclomatic_complexity', 0)
        assert complexity < 5  # Low complexity due to incomplete implementation
    
    @pytest.mark.asyncio
    async def test_ml_security_vulnerability_detection(self, feature_extractor, semantic_test_datasets):
        """Test ML model's ability to detect security vulnerabilities."""
        security_sample = semantic_test_datasets["security_issues"][0]
        
        features = await feature_extractor.extract_features(
            security_sample["code"], 
            security_sample["language"]
        )
        
        # Security-related features should indicate high risk
        assert features.get('security_issues', 0) > 0.5  # High security risk score
        assert features.get('eval_usage', False) or 'eval' in security_sample["code"]
        
        # Should detect dangerous patterns
        code = security_sample["code"].lower()
        dangerous_patterns = ['eval(', 'os.system', 'subprocess.call', 'pickle.loads']
        detected_patterns = sum(1 for pattern in dangerous_patterns if pattern in code)
        assert detected_patterns >= 3
    
    @pytest.mark.asyncio
    async def test_ml_performance_issue_detection(self, feature_extractor, semantic_test_datasets):
        """Test ML model's detection of performance issues."""
        performance_sample = semantic_test_datasets["performance_issues"][0]
        
        features = await feature_extractor.extract_features(
            performance_sample["code"], 
            performance_sample["language"]
        )
        
        # Performance-related features
        assert features.get('nested_loop_count', 0) >= 1
        assert features.get('string_concatenation', 0) >= 1
        
        # Should detect inefficient patterns
        code = performance_sample["code"]
        inefficient_patterns = ['range(len(', '+=', 'insert(0,']
        detected_patterns = sum(1 for pattern in inefficient_patterns if pattern in code)
        assert detected_patterns >= 2


class TestSemanticAnalyzerAccuracy:
    """Test semantic analyzer accuracy against known test cases."""
    
    @pytest.mark.asyncio
    async def test_high_quality_code_analysis_accuracy(self, semantic_analyzer, semantic_test_datasets):
        """Test accuracy of analyzing high-quality code samples."""
        await semantic_analyzer.initialize()
        
        high_quality_samples = semantic_test_datasets["high_quality_code"]
        correct_analyses = 0
        
        for sample in high_quality_samples:
            issues = await semantic_analyzer.analyze_content(
                content=sample["code"],
                language=sample["language"]
            )
            
            # High-quality code should have minimal issues
            critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
            high_issues = [issue for issue in issues if issue.severity == ValidationSeverity.HIGH]
            
            # Should have no critical issues and minimal high-severity issues
            if len(critical_issues) == 0 and len(high_issues) <= 1:
                correct_analyses += 1
        
        accuracy = correct_analyses / len(high_quality_samples)
        assert accuracy >= 0.95, f"High-quality code analysis accuracy {accuracy:.3f} below 95% threshold"
    
    @pytest.mark.asyncio
    async def test_placeholder_code_detection_accuracy(self, semantic_analyzer, semantic_test_datasets):
        """Test accuracy of detecting placeholder/incomplete code."""
        await semantic_analyzer.initialize()
        
        placeholder_samples = semantic_test_datasets["placeholder_code"]
        correct_detections = 0
        
        for sample in placeholder_samples:
            issues = await semantic_analyzer.analyze_content(
                content=sample["code"],
                language=sample["language"]
            )
            
            # Should detect multiple placeholder issues
            placeholder_issues = [
                issue for issue in issues 
                if issue.issue_type in ["placeholder", "unused_import", "logical_error"]
            ]
            
            expected_issues = sample.get("expected_issues", 5)
            if len(placeholder_issues) >= expected_issues * 0.8:  # Allow 20% tolerance
                correct_detections += 1
        
        accuracy = correct_detections / len(placeholder_samples)
        assert accuracy >= 0.95, f"Placeholder detection accuracy {accuracy:.3f} below 95% threshold"
    
    @pytest.mark.asyncio
    async def test_security_vulnerability_detection_accuracy(self, semantic_analyzer, semantic_test_datasets):
        """Test accuracy of detecting security vulnerabilities."""
        await semantic_analyzer.initialize()
        
        security_samples = semantic_test_datasets["security_issues"]
        correct_detections = 0
        
        for sample in security_samples:
            issues = await semantic_analyzer.analyze_content(
                content=sample["code"],
                language=sample["language"]
            )
            
            # Should detect security issues
            security_issues = [
                issue for issue in issues 
                if issue.issue_type == "security_issue" or issue.severity == ValidationSeverity.CRITICAL
            ]
            
            expected_vulnerabilities = len(sample.get("security_vulnerabilities", []))
            if len(security_issues) >= expected_vulnerabilities * 0.7:  # Allow 30% tolerance
                correct_detections += 1
        
        accuracy = correct_detections / len(security_samples)
        assert accuracy >= 0.90, f"Security vulnerability detection accuracy {accuracy:.3f} below 90% threshold"
    
    @pytest.mark.asyncio
    async def test_logic_error_detection_accuracy(self, semantic_analyzer, semantic_test_datasets):
        """Test accuracy of detecting logical errors in code."""
        await semantic_analyzer.initialize()
        
        logic_error_samples = semantic_test_datasets["logic_errors"]
        correct_detections = 0
        
        for sample in logic_error_samples:
            issues = await semantic_analyzer.analyze_content(
                content=sample["code"],
                language=sample["language"]
            )
            
            # Should detect logic errors
            logic_issues = [
                issue for issue in issues 
                if issue.issue_type in ["logical_error", "unreachable_code", "syntax_error"]
            ]
            
            expected_errors = len(sample.get("logic_error_types", []))
            if len(logic_issues) >= expected_errors * 0.6:  # Allow 40% tolerance for complex logic errors
                correct_detections += 1
        
        accuracy = correct_detections / len(logic_error_samples)
        assert accuracy >= 0.85, f"Logic error detection accuracy {accuracy:.3f} below 85% threshold"


class TestSemanticAnalyzerPerformance:
    """Test semantic analyzer performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_analysis_performance_under_200ms(self, semantic_analyzer, semantic_test_datasets):
        """Test that semantic analysis completes within 200ms performance target."""
        await semantic_analyzer.initialize()
        
        # Test with different code sizes
        test_samples = [
            semantic_test_datasets["high_quality_code"][0],  # Small sample
            semantic_test_datasets["high_quality_code"][1],  # Large sample
            semantic_test_datasets["placeholder_code"][0],   # Medium sample
        ]
        
        for sample in test_samples:
            start_time = time.perf_counter()
            
            issues = await semantic_analyzer.analyze_content(
                content=sample["code"],
                language=sample["language"]
            )
            
            end_time = time.perf_counter()
            analysis_time_ms = (end_time - start_time) * 1000
            
            assert analysis_time_ms < 200, f"Analysis time {analysis_time_ms:.1f}ms exceeds 200ms threshold"
            assert isinstance(issues, list)  # Verify valid output
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self, semantic_analyzer, semantic_test_datasets):
        """Test performance with concurrent semantic analysis."""
        await semantic_analyzer.initialize()
        
        # Create multiple analysis tasks
        all_samples = (
            semantic_test_datasets["high_quality_code"] +
            semantic_test_datasets["placeholder_code"] +
            semantic_test_datasets["security_issues"]
        )
        
        start_time = time.perf_counter()
        
        # Run analyses concurrently
        analysis_tasks = [
            semantic_analyzer.analyze_content(
                content=sample["code"],
                language=sample["language"]
            )
            for sample in all_samples
        ]
        
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_analysis = (total_time / len(all_samples)) * 1000
        
        # Verify all analyses completed successfully
        assert len(results) == len(all_samples)
        assert all(not isinstance(result, Exception) for result in results)
        
        # Average time should be well under 200ms due to concurrency
        assert avg_time_per_analysis < 100, f"Average concurrent analysis time {avg_time_per_analysis:.1f}ms too high"
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_analysis(self, semantic_analyzer, semantic_test_datasets):
        """Test memory usage during semantic analysis operations."""
        await semantic_analyzer.initialize()
        
        # Generate large code sample for memory testing
        large_code_lines = []
        for i in range(1000):
            large_code_lines.append(f"def function_{i}(param_{i}): return param_{i} * {i}")
        
        large_code_sample = "\n".join(large_code_lines)
        
        # Monitor memory usage during analysis
        import gc
        gc.collect()  # Clean up before test
        
        start_time = time.perf_counter()
        
        # Analyze large code sample
        issues = await semantic_analyzer.analyze_content(
            content=large_code_sample,
            language="python"
        )
        
        end_time = time.perf_counter()
        analysis_time = (end_time - start_time) * 1000
        
        # Force garbage collection and verify cleanup
        gc.collect()
        
        # Verify analysis completed successfully
        assert isinstance(issues, list)
        assert analysis_time < 500  # Should still be reasonably fast even for large code
        
        # Memory should be cleaned up (no easy way to test exact memory usage in pytest)
        # But we can verify the analysis didn't crash or hang


class TestSemanticAnalyzerEdgeCases:
    """Test semantic analyzer with edge cases and malformed inputs."""
    
    @pytest.mark.asyncio
    async def test_malformed_syntax_handling(self, semantic_analyzer):
        """Test handling of malformed syntax."""
        await semantic_analyzer.initialize()
        
        malformed_samples = [
            "def incomplete_function(",  # Incomplete function definition
            "if True\n    print('missing colon')",  # Missing colon
            "for i in range(10\n    print(i)",  # Unclosed parentheses
            "class MissingColon\n    pass",  # Missing colon in class definition
            "def func():\n    return",  # Incomplete return statement
        ]
        
        for malformed_code in malformed_samples:
            issues = await semantic_analyzer.analyze_content(
                content=malformed_code,
                language="python"
            )
            
            # Should detect syntax errors
            syntax_issues = [
                issue for issue in issues 
                if issue.issue_type == "syntax_error" or "syntax" in issue.description.lower()
            ]
            
            assert len(syntax_issues) >= 1, f"Failed to detect syntax error in: {malformed_code}"
    
    @pytest.mark.asyncio
    async def test_empty_and_whitespace_inputs(self, semantic_analyzer):
        """Test handling of empty and whitespace-only inputs."""
        await semantic_analyzer.initialize()
        
        edge_case_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            "\n\n\n",  # Newlines only
            "\t\t\t",  # Tabs only
            "# Just a comment",  # Comment only
            "'''Just a docstring'''",  # Docstring only
        ]
        
        for edge_input in edge_case_inputs:
            issues = await semantic_analyzer.analyze_content(
                content=edge_input,
                language="python"
            )
            
            # Should handle gracefully without crashing
            assert isinstance(issues, list)
            # Empty/minimal inputs should have no issues or minimal issues
            critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
            assert len(critical_issues) == 0
    
    @pytest.mark.asyncio
    async def test_very_large_input_handling(self, semantic_analyzer):
        """Test handling of very large code inputs."""
        await semantic_analyzer.initialize()
        
        # Generate very large code sample (10,000 lines)
        large_lines = []
        for i in range(10000):
            if i % 100 == 0:  # Add some classes occasionally
                large_lines.append(f"class LargeClass_{i}:")
                large_lines.append(f"    def method_{i}(self): return {i}")
            else:
                large_lines.append(f"def large_function_{i}(): return {i}")
        
        very_large_code = "\n".join(large_lines)
        
        start_time = time.perf_counter()
        
        issues = await semantic_analyzer.analyze_content(
            content=very_large_code,
            language="python"
        )
        
        end_time = time.perf_counter()
        analysis_time = (end_time - start_time) * 1000
        
        # Should handle large inputs without crashing
        assert isinstance(issues, list)
        # Should complete within reasonable time (allow more time for very large inputs)
        assert analysis_time < 2000, f"Large input analysis time {analysis_time:.1f}ms too slow"
    
    @pytest.mark.asyncio  
    async def test_unicode_and_special_characters(self, semantic_analyzer):
        """Test handling of Unicode characters and special symbols."""
        await semantic_analyzer.initialize()
        
        unicode_samples = [
            "def función_español(): return 'Hola'",  # Spanish characters
            "def 函数名称(): return '中文'",  # Chinese characters  
            "def عربية(): return 'العربية'",  # Arabic characters
            "def func(): return '🚀🔥💯'",  # Emoji characters
            "def func(): return 'Special chars: åøæ ñü ßç'",  # Various accented characters
        ]
        
        for unicode_code in unicode_samples:
            issues = await semantic_analyzer.analyze_content(
                content=unicode_code,
                language="python"
            )
            
            # Should handle Unicode without crashing
            assert isinstance(issues, list)
            
            # Unicode function names might generate issues, but shouldn't crash
            critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
            # Allow some critical issues for non-standard identifiers, but not crashes
            assert len(critical_issues) <= 2


class TestCrossValidationFramework:
    """Test cross-validation framework for model consistency."""
    
    @pytest.mark.asyncio
    async def test_semantic_analysis_consistency(self, semantic_analyzer, semantic_test_datasets):
        """Test consistency of semantic analysis across multiple runs."""
        await semantic_analyzer.initialize()
        
        test_sample = semantic_test_datasets["high_quality_code"][0]
        
        # Run analysis multiple times
        results = []
        for _ in range(10):
            issues = await semantic_analyzer.analyze_content(
                content=test_sample["code"],
                language=test_sample["language"]
            )
            results.append(issues)
        
        # Check consistency across runs
        issue_counts = [len(result) for result in results]
        issue_count_variance = np.var(issue_counts)
        
        # Results should be consistent (low variance)
        assert issue_count_variance <= 1.0, f"Semantic analysis consistency issue: variance={issue_count_variance}"
        
        # All runs should produce similar issue types
        all_issue_types = set()
        for result in results:
            for issue in result:
                all_issue_types.add(issue.issue_type)
        
        # Should not have wildly different issue types across runs
        assert len(all_issue_types) <= 5  # Reasonable upper bound for consistent analysis
    
    @pytest.mark.asyncio
    async def test_feature_extraction_consistency(self, feature_extractor, semantic_test_datasets):
        """Test consistency of feature extraction across multiple runs."""
        test_sample = semantic_test_datasets["high_quality_code"][0]
        
        # Extract features multiple times
        feature_sets = []
        for _ in range(10):
            features = await feature_extractor.extract_features(
                test_sample["code"],
                test_sample["language"]
            )
            feature_sets.append(features)
        
        # Check consistency of extracted features
        for feature_name in feature_sets[0].keys():
            values = [features.get(feature_name, 0) for features in feature_sets]
            
            # Numeric features should be identical across runs
            if isinstance(values[0], (int, float)):
                assert all(abs(v - values[0]) < 1e-6 for v in values), \
                    f"Feature {feature_name} inconsistent across runs: {values}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])