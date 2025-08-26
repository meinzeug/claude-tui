"""
Comprehensive tests for Anti-Hallucination Validation System.

Tests cover:
- Advanced placeholder detection
- Real vs fake progress calculation
- Code quality analysis
- Auto-completion strategies
- Cross-validation with AI
- Progress tracking and alerts
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from src.core.validator import ProgressValidator, CodeQualityAnalyzer, CodeQualityMetrics
from src.core.completion_strategies import AutoCompletionEngine, CompletionStrategy
from src.core.progress_tracker import RealProgressTracker, ProgressSnapshot, ProgressConfidence
from src.core.types import Issue, IssueType, Severity, ValidationResult


class TestAdvancedPlaceholderDetection:
    """Test advanced placeholder detection capabilities."""
    
    @pytest.fixture
    def validator(self):
        """Create validator with enhanced patterns."""
        return ProgressValidator(enable_quality_analysis=True)
    
    def test_sophisticated_placeholder_patterns(self, validator):
        """Test detection of sophisticated placeholder patterns."""
        sophisticated_code = '''
def process_data(data):
    """Process data with advanced algorithms."""
    if not data:
        return {}  # TODO: Handle empty data case properly
    
    # Step 1: Validation
    validated_data = validate_input(data)  # TODO: Implement validation
    
    # Step 2: Processing  
    processed_data = []
    for item in validated_data:
        # Complex processing logic would go here
        result = process_item(item)  # FIXME: Actual implementation needed
        processed_data.append(result)
    
    # Step 3: Post-processing
    # ... additional logic needed here ...
    
    return processed_data

def validate_input(data):
    # Basic validation - needs enhancement
    pass

def process_item(item):
    # Placeholder implementation
    raise NotImplementedError("Core processing logic not implemented")

class DataProcessor:
    def __init__(self):
        self.config = None  # Configuration loading needed
        
    def load_config(self):
        # TODO: Load configuration from file
        console.log("Loading config...")  # JavaScript-style logging
        
    def save_results(self, results):
        # Temporary storage implementation
        with open("/tmp/results.json", "w") as f:
            json.dump(results, f)  # Not production ready
'''
        
        # Run validation
        issues, score = asyncio.run(
            validator._validate_single_file(Path("test.py"))
        ) if hasattr(validator, '_validate_single_file') else ([], 0)
        
        # Should detect multiple types of placeholders
        assert len([i for i in issues if i.type == IssueType.PLACEHOLDER]) >= 3
        
        # Should find TODO comments
        todo_issues = [i for i in issues if "TODO" in i.description]
        assert len(todo_issues) >= 2
        
        # Should find FIXME comments
        fixme_issues = [i for i in issues if "FIXME" in i.description]
        assert len(fixme_issues) >= 1
        
        # Should find NotImplementedError
        not_impl_issues = [i for i in issues if "NotImplementedError" in i.description]
        assert len(not_impl_issues) >= 1
        
        # Should find JavaScript-like constructs
        js_issues = [i for i in issues if "console.log" in i.description]
        assert len(js_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_context_aware_validation(self, validator):
        """Test context-aware validation that considers surrounding code."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir)
            
            # Create a file with context-dependent issues
            test_file = test_dir / "context_test.py"
            test_file.write_text('''
class UserManager:
    def __init__(self, database):
        self.db = database
        
    def create_user(self, username, email):
        """Create a new user account."""
        # Basic validation
        if not username or not email:
            raise ValueError("Username and email required")
        
        # TODO: Check if user already exists
        # TODO: Validate email format
        # TODO: Hash password if provided
        
        user_data = {
            'username': username,
            'email': email,
            'created_at': datetime.now()
        }
        
        # Database operation
        user_id = self.db.insert('users', user_data)
        return user_id
    
    def get_user(self, user_id):
        """Get user by ID."""
        pass  # TODO: Implement user retrieval
        
    def update_user(self, user_id, updates):
        """Update user information."""
        # Placeholder implementation
        return None
    
    def delete_user(self, user_id):
        """Delete user account."""
        raise NotImplementedError("User deletion not implemented")
''')
            
            # Validate the file
            result = await validator.validate_codebase(test_dir)
            
            # Should detect context-aware issues
            assert not result.is_authentic
            assert len(result.issues) >= 5
            
            # Should have varying severities based on context
            severities = {issue.severity for issue in result.issues}
            assert len(severities) >= 2  # Multiple severity levels
            
            # Should detect empty functions with different contexts
            empty_func_issues = [i for i in result.issues if i.type == IssueType.EMPTY_FUNCTION]
            assert len(empty_func_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_cross_language_detection(self, validator):
        """Test placeholder detection across multiple programming languages."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir)
            
            # Python file with issues
            (test_dir / "python_file.py").write_text('''
def calculate(a, b):
    # TODO: Implement calculation
    pass

def process():
    raise NotImplementedError()
''')
            
            # JavaScript file with issues
            (test_dir / "script.js").write_text('''
function processData(data) {
    // TODO: Process the data
    console.log("Processing:", data);
    return null;
}

function validateInput(input) {
    // FIXME: Add proper validation
    return true;
}
''')
            
            # TypeScript file with issues  
            (test_dir / "component.ts").write_text('''
interface User {
    id: string;
    name: string;
}

class UserService {
    getUser(id: string): User {
        // TODO: Implement user retrieval
        throw new Error("Not implemented");
    }
    
    createUser(userData: Partial<User>): User {
        // Placeholder implementation
        return {} as User;
    }
}
''')
            
            result = await validator.validate_codebase(test_dir)
            
            # Should detect issues across all file types
            issues_by_file = {}
            for issue in result.issues:
                if issue.file_path:
                    file_name = Path(issue.file_path).name
                    if file_name not in issues_by_file:
                        issues_by_file[file_name] = []
                    issues_by_file[file_name].append(issue)
            
            # Should have issues in multiple files
            assert len(issues_by_file) >= 2
            
            # Python file should have issues
            python_issues = issues_by_file.get("python_file.py", [])
            assert len(python_issues) >= 2
            
            # JavaScript files should have issues (if JS support is implemented)
            # This depends on the validator's language support


class TestCodeQualityAnalyzer:
    """Test the code quality analysis component."""
    
    @pytest.fixture
    def analyzer(self):
        """Create code quality analyzer."""
        return CodeQualityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_analysis(self, analyzer):
        """Test comprehensive code quality analysis."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir)
            
            # Create files with various quality issues
            (test_dir / "high_complexity.py").write_text('''
def complex_function(a, b, c, d, e):
    """Function with high cyclomatic complexity."""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        for i in range(100):
                            for j in range(100):
                                if i + j > 50:
                                    try:
                                        result = a * b * c * d * e
                                        if result > 1000:
                                            return result
                                        else:
                                            continue
                                    except Exception as ex:
                                        if str(ex).startswith("Error"):
                                            return -1
                                        elif "ValueError" in str(ex):
                                            return -2
                                        else:
                                            return -3
    return 0
''')
            
            (test_dir / "no_docs.py").write_text('''
class UndocumentedClass:
    def method_one(self):
        return "test"
        
    def method_two(self, param):
        return param * 2
        
    def method_three(self):
        pass

def undocumented_function(x, y):
    return x + y
''')
            
            (test_dir / "security_issues.py").write_text('''
import subprocess
import os

def dangerous_function(user_input):
    # Security vulnerability: shell injection
    result = subprocess.run(f"echo {user_input}", shell=True)
    
    # Hardcoded credentials
    password = "hardcoded_password"
    api_key = "abc123secret"
    
    # Use of eval
    dangerous_code = user_input
    eval(dangerous_code)
    
    return result
''')
            
            # Create test directory
            (test_dir / "tests").mkdir()
            (test_dir / "tests" / "test_basic.py").write_text('''
import pytest

def test_simple():
    assert True

def test_another():
    assert 1 + 1 == 2
''')
            
            # Analyze quality
            metrics = await analyzer.analyze_codebase(test_dir)
            
            # Verify quality metrics
            assert isinstance(metrics, CodeQualityMetrics)
            assert metrics.lines_of_code > 0
            assert metrics.cyclomatic_complexity > 1.0  # Should detect complexity
            assert 0 <= metrics.test_coverage <= 100
            assert 0 <= metrics.documentation_coverage <= 100
            assert 0 <= metrics.maintainability_index <= 100
            assert metrics.technical_debt_minutes >= 0
            assert 0 <= metrics.duplication_percentage <= 100
            assert metrics.security_hotspots >= 3  # Should detect security issues
            assert 0 <= metrics.overall_score <= 100
    
    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, analyzer):
        """Test quality score calculation logic."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir)
            
            # Create high-quality code
            (test_dir / "good_code.py").write_text('''
"""High-quality module with proper documentation."""

class WellDocumentedClass:
    """A well-documented class with clear purpose."""
    
    def __init__(self, name: str):
        """Initialize with name."""
        self.name = name
    
    def get_name(self) -> str:
        """Get the name."""
        return self.name
    
    def set_name(self, name: str) -> None:
        """Set the name."""
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name

def simple_function(x: int, y: int) -> int:
    """Add two integers."""
    return x + y
''')
            
            # Create tests
            (test_dir / "tests").mkdir()
            (test_dir / "tests" / "test_good_code.py").write_text('''
"""Tests for good_code module."""
import pytest
from good_code import WellDocumentedClass, simple_function

def test_class_creation():
    """Test class creation."""
    obj = WellDocumentedClass("test")
    assert obj.get_name() == "test"

def test_simple_function():
    """Test simple function."""
    assert simple_function(2, 3) == 5
''')
            
            metrics = await analyzer.analyze_codebase(test_dir)
            
            # High-quality code should have good scores
            assert metrics.overall_score >= 70
            assert metrics.documentation_coverage >= 80
            assert metrics.cyclomatic_complexity <= 5
            assert metrics.security_hotspots == 0


class TestAutoCompletionEngine:
    """Test the auto-completion engine."""
    
    @pytest.fixture
    def completion_engine(self):
        """Create completion engine."""
        return AutoCompletionEngine()
    
    @pytest.mark.asyncio
    async def test_template_based_completion(self, completion_engine):
        """Test template-based completion strategy."""
        # Create a TODO issue
        issue = Issue(
            type=IssueType.PLACEHOLDER,
            severity=Severity.MEDIUM,
            description="TODO comment found",
            file_path="test.py",
            line_number=5,
            auto_fix_available=True
        )
        
        context_code = '''
def calculate_total(items):
    """Calculate total price of items."""
    total = 0
    for item in items:
        # TODO: implement price calculation
        pass
    return total
'''
        
        result = await completion_engine.complete_placeholder(
            issue, context_code, CompletionStrategy.TEMPLATE_BASED
        )
        
        assert isinstance(result.success, bool)
        assert result.strategy_used == CompletionStrategy.TEMPLATE_BASED
        assert result.original_code == context_code
        if result.success:
            assert result.completed_code != context_code
            assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_context_aware_completion(self, completion_engine):
        """Test context-aware completion strategy."""
        issue = Issue(
            type=IssueType.EMPTY_FUNCTION,
            severity=Severity.HIGH,
            description="Empty function found",
            file_path="test.py",
            line_number=2,
            auto_fix_available=True
        )
        
        context_code = '''
def add_numbers(a, b):
    pass
'''
        
        result = await completion_engine.complete_placeholder(
            issue, context_code, CompletionStrategy.CONTEXT_AWARE
        )
        
        assert result.strategy_used == CompletionStrategy.CONTEXT_AWARE
        if result.success:
            assert "return" in result.completed_code or "a + b" in result.completed_code
    
    @pytest.mark.asyncio
    async def test_batch_completion(self, completion_engine):
        """Test batch completion of multiple issues."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "batch_test.py"
            test_file.write_text('''
def function_one():
    # TODO: implement this
    pass

def function_two():
    raise NotImplementedError()

def function_three():
    # FIXME: add proper logic
    return None
''')
            
            issues = [
                Issue(
                    type=IssueType.PLACEHOLDER,
                    severity=Severity.MEDIUM,
                    description="TODO comment",
                    file_path=str(test_file),
                    line_number=2
                ),
                Issue(
                    type=IssueType.PLACEHOLDER,
                    severity=Severity.HIGH,
                    description="NotImplementedError",
                    file_path=str(test_file),
                    line_number=6
                ),
                Issue(
                    type=IssueType.PLACEHOLDER,
                    severity=Severity.MEDIUM,
                    description="FIXME comment",
                    file_path=str(test_file),
                    line_number=10
                )
            ]
            
            results = await completion_engine.batch_complete(issues, test_file, max_concurrent=2)
            
            assert len(results) == len(issues)
            assert all(hasattr(result, 'success') for result in results)
            assert all(hasattr(result, 'strategy_used') for result in results)
    
    def test_completion_statistics(self, completion_engine):
        """Test completion statistics tracking."""
        initial_stats = completion_engine.get_completion_stats()
        
        assert 'attempts' in initial_stats
        assert 'successes' in initial_stats
        assert 'failures' in initial_stats
        assert 'by_strategy' in initial_stats
        assert 'success_rate' in initial_stats
        
        assert initial_stats['attempts'] == 0
        assert initial_stats['successes'] == 0
        assert initial_stats['failures'] == 0
        assert initial_stats['success_rate'] == 0


class TestRealProgressTracker:
    """Test the real vs fake progress tracking system."""
    
    @pytest.fixture
    def progress_tracker(self):
        """Create progress tracker for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = RealProgressTracker(tmp_dir, validation_interval=1)
            return tracker
    
    @pytest.mark.asyncio
    async def test_progress_snapshot(self, progress_tracker):
        """Test taking a progress snapshot."""
        snapshot = await progress_tracker.take_snapshot()
        
        assert isinstance(snapshot, ProgressSnapshot)
        assert isinstance(snapshot.timestamp, datetime)
        assert 0 <= snapshot.real_progress <= 100
        assert 0 <= snapshot.fake_progress <= 100
        assert 0 <= snapshot.total_progress <= 100
        assert 0 <= snapshot.authenticity_score <= 100
        assert 0 <= snapshot.quality_score <= 100
        assert snapshot.files_analyzed >= 0
        assert snapshot.issues_found >= 0
        assert isinstance(snapshot.confidence, ProgressConfidence)
        assert isinstance(snapshot.details, dict)
    
    @pytest.mark.asyncio
    async def test_progress_monitoring(self, progress_tracker):
        """Test continuous progress monitoring."""
        # Start monitoring
        await progress_tracker.start_monitoring()
        
        # Let it run for a short time
        await asyncio.sleep(2.5)  # Should get 2-3 snapshots with 1s interval
        
        # Stop monitoring
        await progress_tracker.stop_monitoring()
        
        # Check results
        assert len(progress_tracker.snapshots) >= 2
        assert progress_tracker.baseline_metrics is not None
        
        current = progress_tracker.get_current_progress()
        assert current is not None
        assert isinstance(current, ProgressSnapshot)
    
    def test_progress_trend_analysis(self, progress_tracker):
        """Test progress trend analysis."""
        # Add some mock snapshots
        now = datetime.utcnow()
        snapshots = [
            ProgressSnapshot(
                timestamp=now - timedelta(hours=2),
                real_progress=20.0,
                fake_progress=10.0,
                total_progress=30.0,
                authenticity_score=70.0,
                quality_score=60.0,
                files_analyzed=5,
                issues_found=3,
                confidence=ProgressConfidence.MEDIUM
            ),
            ProgressSnapshot(
                timestamp=now - timedelta(hours=1),
                real_progress=40.0,
                fake_progress=15.0,
                total_progress=55.0,
                authenticity_score=75.0,
                quality_score=65.0,
                files_analyzed=8,
                issues_found=2,
                confidence=ProgressConfidence.HIGH
            ),
            ProgressSnapshot(
                timestamp=now,
                real_progress=60.0,
                fake_progress=20.0,
                total_progress=80.0,
                authenticity_score=80.0,
                quality_score=70.0,
                files_analyzed=10,
                issues_found=1,
                confidence=ProgressConfidence.HIGH
            )
        ]
        
        progress_tracker.snapshots = snapshots
        
        trend = progress_tracker.analyze_progress_trend(hours=3)
        
        assert trend.direction in ["improving", "declining", "stable"]
        assert isinstance(trend.velocity, float)
        assert 0 <= trend.consistency <= 100
        assert trend.confidence >= 0
        if trend.predicted_completion:
            assert isinstance(trend.predicted_completion, datetime)
    
    def test_alert_generation(self, progress_tracker):
        """Test alert generation for progress anomalies."""
        # Mock snapshots with issues that should trigger alerts
        now = datetime.utcnow()
        
        # Good initial state
        snapshot1 = ProgressSnapshot(
            timestamp=now - timedelta(minutes=5),
            real_progress=80.0,
            fake_progress=10.0,
            total_progress=90.0,
            authenticity_score=90.0,
            quality_score=85.0,
            files_analyzed=10,
            issues_found=1,
            confidence=ProgressConfidence.VERY_HIGH
        )
        
        # Degraded state that should trigger alerts
        snapshot2 = ProgressSnapshot(
            timestamp=now,
            real_progress=60.0,  # Dropped
            fake_progress=35.0,  # Increased significantly  
            total_progress=95.0,
            authenticity_score=65.0,  # Dropped
            quality_score=70.0,  # Dropped
            files_analyzed=10,
            issues_found=5,
            confidence=ProgressConfidence.MEDIUM
        )
        
        progress_tracker.snapshots = [snapshot1, snapshot2]
        
        # Manually check for alerts (normally done in take_snapshot)
        asyncio.run(progress_tracker._check_for_alerts(snapshot2))
        
        alerts = progress_tracker.get_alerts()
        
        # Should have generated alerts for the degradation
        assert len(alerts) > 0
        
        alert_types = {alert.alert_type for alert in alerts}
        possible_alerts = {
            "authenticity_drop", 
            "fake_progress_spike", 
            "quality_degradation"
        }
        
        # At least one type of alert should be generated
        assert len(alert_types.intersection(possible_alerts)) > 0
    
    def test_progress_report_generation(self, progress_tracker):
        """Test comprehensive progress report generation."""
        # Add some mock data
        now = datetime.utcnow()
        snapshot = ProgressSnapshot(
            timestamp=now,
            real_progress=75.0,
            fake_progress=15.0,
            total_progress=90.0,
            authenticity_score=85.0,
            quality_score=80.0,
            files_analyzed=15,
            issues_found=2,
            confidence=ProgressConfidence.HIGH,
            details={"test": "data"}
        )
        
        progress_tracker.snapshots = [snapshot]
        
        report = progress_tracker.get_progress_report()
        
        assert report['status'] == 'active'
        assert 'current_progress' in report
        assert 'trend_analysis' in report
        assert 'statistics' in report
        assert 'alerts' in report
        assert 'history' in report
        
        current = report['current_progress']
        assert current['real_progress'] == 75.0
        assert current['fake_progress'] == 15.0
        assert current['authenticity_rate'] == snapshot.authenticity_rate


@pytest.mark.integration
class TestAntiHallucinationIntegration:
    """Integration tests for the complete anti-hallucination system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation_pipeline(self):
        """Test complete validation pipeline from detection to completion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir)
            
            # Create a realistic project with various issues
            (test_dir / "main.py").write_text('''
"""Main application module."""

class DataProcessor:
    """Process data with various methods."""
    
    def __init__(self):
        self.data = []
    
    def load_data(self, filename):
        """Load data from file."""
        # TODO: Implement file loading
        pass
    
    def process_data(self):
        """Process the loaded data."""
        if not self.data:
            return []
        
        results = []
        for item in self.data:
            # FIXME: Add proper processing logic
            result = self._process_item(item)
            results.append(result)
        
        return results
    
    def _process_item(self, item):
        """Process individual item."""
        raise NotImplementedError("Item processing not implemented")
    
    def save_results(self, results, filename):
        """Save results to file."""
        # Placeholder implementation
        console.log(f"Saving {len(results)} results")
        return True
''')
            
            (test_dir / "utils.py").write_text('''
def helper_function(x, y):
    """A complete helper function."""
    return x + y

def incomplete_function(data):
    """An incomplete function."""
    # TODO: implement this
    pass
''')
            
            # Create tests directory
            (test_dir / "tests").mkdir()
            (test_dir / "tests" / "test_main.py").write_text('''
import pytest
from main import DataProcessor

def test_data_processor_init():
    """Test DataProcessor initialization."""
    processor = DataProcessor()
    assert processor.data == []
''')
            
            # Initialize validator with all features
            validator = ProgressValidator(
                enable_cross_validation=True,
                enable_execution_testing=True,
                enable_quality_analysis=True
            )
            
            # Run complete validation
            result = await validator.validate_codebase(test_dir)
            
            # Verify comprehensive analysis
            assert isinstance(result, ValidationResult)
            assert not result.is_authentic  # Should detect issues
            assert result.authenticity_score < 80  # Low due to placeholders
            assert len(result.issues) >= 4  # Multiple issues should be found
            assert len(result.suggestions) > 0
            assert len(result.next_actions) > 0
            
            # Check issue types
            issue_types = {issue.type for issue in result.issues}
            assert IssueType.PLACEHOLDER in issue_types
            
            # Verify quality analysis was included
            if hasattr(result, 'metadata') and result.metadata:
                assert 'quality_score' in result.metadata
                assert 'lines_of_code' in result.metadata
                assert 'complexity' in result.metadata
            
            # Test auto-completion
            completion_engine = AutoCompletionEngine()
            placeholder_issues = [
                issue for issue in result.issues 
                if issue.type == IssueType.PLACEHOLDER
            ]
            
            if placeholder_issues:
                completion_results = await completion_engine.batch_complete(
                    placeholder_issues[:2],  # Test first 2 issues
                    test_dir / "main.py"
                )
                
                assert len(completion_results) <= 2
                for comp_result in completion_results:
                    assert hasattr(comp_result, 'success')
                    assert hasattr(comp_result, 'strategy_used')
    
    @pytest.mark.asyncio
    async def test_progress_tracking_with_improvements(self):
        """Test progress tracking as code quality improves."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir)
            
            # Start with poor quality code
            main_file = test_dir / "evolving.py"
            main_file.write_text('''
def bad_function():
    # TODO: implement
    pass

def another_bad_function():
    raise NotImplementedError()
''')
            
            # Initialize progress tracker
            tracker = RealProgressTracker(test_dir, validation_interval=1)
            
            # Take initial snapshot
            initial_snapshot = await tracker.take_snapshot()
            assert initial_snapshot.real_progress < 50
            
            # Improve the code
            main_file.write_text('''
def good_function():
    """A well-implemented function."""
    return "Hello, world!"

def another_good_function(x, y):
    """Add two numbers."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("Arguments must be numbers")
    return x + y
''')
            
            # Take another snapshot
            improved_snapshot = await tracker.take_snapshot()
            
            # Progress should have improved
            assert improved_snapshot.real_progress > initial_snapshot.real_progress
            assert improved_snapshot.authenticity_score > initial_snapshot.authenticity_score
            assert improved_snapshot.fake_progress < initial_snapshot.fake_progress
            
            # Trend analysis should show improvement
            trend = tracker.analyze_progress_trend(hours=1)
            assert trend.direction in ["improving", "stable"]  # Should not be declining


# Fixtures for test data

@pytest.fixture
def code_samples():
    """Sample code for testing."""
    return {
        'complete_function': '''
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
''',
        
        'function_with_todo': '''
def process_data(data):
    """Process input data."""
    # TODO: implement data processing
    pass
''',
        
        'mixed_implementation': '''
def complete_function(x, y):
    """This function is complete."""
    return x * y

def incomplete_function(data):
    """This function needs work."""
    # TODO: implement processing
    pass

def broken_function():
    """This function is broken."""
    raise NotImplementedError("Feature not ready")

def good_function():
    """This function is well implemented."""
    result = perform_calculation()
    return validate_result(result)
''',
        
        'security_issues': '''
import subprocess

def dangerous_function(user_input):
    # Security vulnerability
    subprocess.run(f"echo {user_input}", shell=True)
    
    # Hardcoded secret
    api_key = "secret123"
    
    # Dangerous eval
    eval(user_input)
'''
    }


@pytest.fixture
def mock_ai_interface():
    """Mock AI interface for testing."""
    interface = Mock()
    interface.validate_code = AsyncMock()
    return interface


@pytest.fixture
def validation_fixtures():
    """Validation test fixtures."""
    class ValidationFixtures:
        @staticmethod
        def get_placeholder_patterns():
            """Get test patterns for placeholder detection."""
            from src.core.validator import PlaceholderPattern
            from src.core.types import Severity
            
            return [
                PlaceholderPattern(
                    name="test_todo",
                    pattern=r"TODO.*test",
                    severity=Severity.MEDIUM,
                    description="Test TODO pattern",
                    auto_fix_template="# Implementation completed"
                ),
                PlaceholderPattern(
                    name="test_fixme",
                    pattern=r"FIXME.*bug",
                    severity=Severity.HIGH,
                    description="Test FIXME pattern"
                )
            ]
    
    return ValidationFixtures