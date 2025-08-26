"""
Adaptive Performance Optimization Tests

Tests for the adaptive performance optimization system including ML-based
parameter tuning, Bayesian optimization, and system configuration management.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import tempfile
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from performance.optimization.adaptive_optimizer import (
    AdaptivePerformanceOptimizer,
    BayesianOptimizer,
    PerformancePredictor,
    SystemConfigurationManager,
    OptimizationTarget,
    OptimizationParameter,
    OptimizationResult
)

class TestOptimizationParameter:
    """Test optimization parameter data structure"""
    
    def test_parameter_creation(self):
        """Test optimization parameter creation"""
        param = OptimizationParameter(
            name='db_pool_size',
            current_value=20,
            min_value=5,
            max_value=100,
            step_size=5,
            parameter_type='int',
            description='Database pool size'
        )
        
        assert param.name == 'db_pool_size'
        assert param.current_value == 20
        assert param.min_value == 5
        assert param.max_value == 100
        assert param.parameter_type == 'int'

class TestOptimizationTarget:
    """Test optimization target configuration"""
    
    def test_target_creation(self):
        """Test optimization target creation"""
        target = OptimizationTarget(
            name='throughput',
            metric_path='throughput',
            target_value=1000.0,
            optimization_direction='maximize',
            weight=0.4
        )
        
        assert target.name == 'throughput'
        assert target.target_value == 1000.0
        assert target.optimization_direction == 'maximize'
        assert target.weight == 0.4

class TestPerformancePredictor:
    """Test ML-based performance prediction"""
    
    @pytest.fixture
    def predictor(self):
        return PerformancePredictor()
    
    def test_predictor_initialization(self, predictor):
        """Test predictor initialization"""
        assert not predictor.is_trained
        assert len(predictor.training_data) == 0
        assert len(predictor.feature_names) == 0
    
    def test_add_training_data(self, predictor):
        """Test adding training data"""
        parameters = {'param1': 10, 'param2': 20.0, 'param3': True}
        performance_score = 0.85
        
        predictor.add_training_data(parameters, performance_score)
        
        assert len(predictor.training_data) == 1
        assert predictor.training_data[0]['parameters'] == parameters
        assert predictor.training_data[0]['performance_score'] == performance_score
    
    def test_feature_extraction(self, predictor):
        """Test feature extraction from parameters"""
        parameters = {
            'int_param': 10,
            'float_param': 20.5,
            'bool_param': True,
            'string_param': 'test'
        }
        
        # Set feature names first
        predictor.feature_names = sorted(parameters.keys())
        
        features = predictor._extract_features(parameters)
        
        assert len(features) == 4
        assert features[0] == 1.0  # bool_param converted to 1.0
        assert features[1] == 20.5  # float_param
        assert features[2] == 10.0  # int_param
        assert isinstance(features[3], float)  # string_param hashed
    
    @patch('performance.optimization.adaptive_optimizer.train_test_split')
    @patch('sklearn.ensemble.RandomForestRegressor.fit')
    @patch('sklearn.ensemble.RandomForestRegressor.score')
    def test_model_training(self, mock_score, mock_fit, mock_split, predictor):
        """Test model training process"""
        # Add sufficient training data
        for i in range(15):
            parameters = {'param1': i, 'param2': i * 2.0}
            score = 0.5 + (i / 30.0)  # Increasing score
            predictor.add_training_data(parameters, score)
        
        # Mock train_test_split
        X = np.random.random((15, 2))
        y = np.random.random(15)
        mock_split.return_value = (X[:12], X[12:], y[:12], y[12:])
        
        # Mock model scores
        mock_score.side_effect = [0.8, 0.7]  # train, test scores
        
        # Train model
        success = predictor.train_model()
        
        assert success
        assert predictor.is_trained
        mock_fit.assert_called_once()
    
    def test_insufficient_training_data(self, predictor):
        """Test training with insufficient data"""
        # Add only a few data points
        for i in range(5):
            predictor.add_training_data({'param': i}, 0.5)
        
        success = predictor.train_model()
        
        assert not success
        assert not predictor.is_trained
    
    def test_prediction_without_training(self, predictor):
        """Test prediction before training"""
        parameters = {'param1': 10}
        prediction = predictor.predict_performance(parameters)
        
        assert prediction == 0.0  # Should return 0 if not trained
    
    def test_model_save_load(self, predictor):
        """Test model saving and loading"""
        # Add training data and train
        for i in range(10):
            predictor.add_training_data({'param': i}, i / 10.0)
        
        predictor.feature_names = ['param']
        predictor.is_trained = True
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            predictor.save_model(model_path)
            
            # Create new predictor and load model
            new_predictor = PerformancePredictor()
            success = new_predictor.load_model(model_path)
            
            assert success
            assert new_predictor.is_trained
            assert new_predictor.feature_names == ['param']
            assert len(new_predictor.training_data) == 10
            
        finally:
            Path(model_path).unlink()

class TestBayesianOptimizer:
    """Test Bayesian optimization functionality"""
    
    @pytest.fixture
    def parameters(self):
        return [
            OptimizationParameter(
                name='int_param',
                current_value=10,
                min_value=1,
                max_value=20,
                step_size=1,
                parameter_type='int'
            ),
            OptimizationParameter(
                name='float_param',
                current_value=5.0,
                min_value=1.0,
                max_value=10.0,
                step_size=0.5,
                parameter_type='float'
            ),
            OptimizationParameter(
                name='bool_param',
                current_value=True,
                min_value=False,
                max_value=True,
                step_size=None,
                parameter_type='bool'
            )
        ]
    
    @pytest.fixture
    def optimizer(self, parameters):
        return BayesianOptimizer(parameters)
    
    def test_optimizer_initialization(self, optimizer, parameters):
        """Test optimizer initialization"""
        assert len(optimizer.parameters) == 3
        assert 'int_param' in optimizer.parameters
        assert 'float_param' in optimizer.parameters
        assert 'bool_param' in optimizer.parameters
        assert len(optimizer.exploration_history) == 0
    
    def test_random_suggestion(self, optimizer):
        """Test random parameter suggestion"""
        suggestions = optimizer.suggest_parameters(n_suggestions=5)
        
        assert len(suggestions) == 5
        
        for suggestion in suggestions:
            assert 'int_param' in suggestion
            assert 'float_param' in suggestion
            assert 'bool_param' in suggestion
            
            # Check parameter bounds
            assert 1 <= suggestion['int_param'] <= 20
            assert 1.0 <= suggestion['float_param'] <= 10.0
            assert isinstance(suggestion['bool_param'], bool)
    
    def test_update_results(self, optimizer):
        """Test updating optimization results"""
        parameters = {'int_param': 15, 'float_param': 7.5, 'bool_param': False}
        performance_score = 0.75
        
        optimizer.update_results(parameters, performance_score)
        
        assert len(optimizer.exploration_history) == 1
        history_entry = optimizer.exploration_history[0]
        assert history_entry['parameters'] == parameters
        assert history_entry['performance_score'] == performance_score
    
    def test_bayesian_suggestion(self, optimizer):
        """Test Bayesian optimization suggestion"""
        # Add some exploration history first
        for i in range(10):
            params = {
                'int_param': 5 + i,
                'float_param': 2.0 + i * 0.5,
                'bool_param': i % 2 == 0
            }
            score = 0.5 + (i / 20.0)  # Increasing score
            optimizer.update_results(params, score)
        
        # Get Bayesian suggestion
        suggestions = optimizer.suggest_parameters(n_suggestions=1)
        
        assert len(suggestions) == 1
        suggestion = suggestions[0]
        
        # Should be influenced by best previous results
        assert 'int_param' in suggestion
        assert 'float_param' in suggestion
        assert 'bool_param' in suggestion
    
    def test_get_best_parameters(self, optimizer):
        """Test getting best parameters"""
        # Initially no parameters
        best_params, best_score = optimizer.get_best_parameters()
        assert best_params == {}
        assert best_score == 0.0
        
        # Add some results
        test_data = [
            ({'param': 1}, 0.3),
            ({'param': 2}, 0.7),  # Best
            ({'param': 3}, 0.5)
        ]
        
        for params, score in test_data:
            optimizer.update_results(params, score)
        
        best_params, best_score = optimizer.get_best_parameters()
        assert best_params == {'param': 2}
        assert best_score == 0.7

class TestSystemConfigurationManager:
    """Test system configuration management"""
    
    @pytest.fixture
    def config_manager(self):
        return SystemConfigurationManager()
    
    @pytest.mark.asyncio
    async def test_apply_configuration(self, config_manager):
        """Test configuration application"""
        parameters = {
            'db_pool_size': 30,
            'cache_size_mb': 512,
            'worker_threads': 16
        }
        
        success = await config_manager.apply_configuration(parameters)
        
        assert success
        assert config_manager.applied_changes == parameters
    
    @pytest.mark.asyncio
    async def test_configuration_backup(self, config_manager):
        """Test configuration backup and rollback"""
        parameters = {'test_param': 'test_value'}
        
        # Apply configuration (should create backup)
        await config_manager.apply_configuration(parameters)
        
        # Check backup was created
        assert config_manager.config_backup is not None
        assert 'timestamp' in config_manager.config_backup
        assert 'config' in config_manager.config_backup
    
    @pytest.mark.asyncio
    async def test_parameter_application(self, config_manager):
        """Test individual parameter application"""
        # Test known parameters
        success = await config_manager._apply_parameter('db_pool_size', 25)
        assert success
        
        success = await config_manager._apply_parameter('cache_size_mb', 256)
        assert success
        
        # Test unknown parameter (should not fail)
        success = await config_manager._apply_parameter('unknown_param', 'value')
        assert success

class TestAdaptivePerformanceOptimizer:
    """Test the main adaptive optimizer"""
    
    @pytest.fixture
    def optimizer_config(self):
        return {
            'optimization_interval': 1,  # Fast for testing
            'experiment_duration': 0.1,
            'max_experiments': 5,
            'convergence_threshold': 0.1
        }
    
    @pytest.fixture
    def optimizer(self, optimizer_config):
        return AdaptivePerformanceOptimizer(optimizer_config)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert not optimizer.optimization_active
        assert isinstance(optimizer.predictor, PerformancePredictor)
        assert isinstance(optimizer.optimizer, BayesianOptimizer)
        assert isinstance(optimizer.config_manager, SystemConfigurationManager)
    
    def test_default_config(self):
        """Test default configuration"""
        optimizer = AdaptivePerformanceOptimizer()
        config = optimizer._default_config()
        
        required_keys = [
            'optimization_interval',
            'experiment_duration', 
            'max_experiments',
            'convergence_threshold'
        ]
        
        for key in required_keys:
            assert key in config
    
    def test_load_optimization_parameters(self, optimizer):
        """Test loading optimization parameters"""
        parameters = optimizer._load_optimization_parameters()
        
        assert len(parameters) > 0
        
        # Check for expected parameters
        param_names = [p.name for p in parameters]
        expected_params = ['db_pool_size', 'cache_size_mb', 'worker_threads']
        
        for expected in expected_params:
            assert expected in param_names
    
    def test_load_optimization_targets(self, optimizer):
        """Test loading optimization targets"""
        targets = optimizer._load_optimization_targets()
        
        assert len(targets) > 0
        
        # Check for expected targets
        target_names = [t.name for t in targets]
        expected_targets = ['throughput', 'latency_p95', 'error_rate']
        
        for expected in expected_targets:
            assert expected in target_names
    
    @patch('asyncio.sleep')
    @pytest.mark.asyncio
    async def test_run_experiment(self, mock_sleep, optimizer):
        """Test running optimization experiment"""
        mock_sleep.return_value = None  # Skip actual sleep
        
        parameters = {
            'db_pool_size': 20,
            'cache_size_mb': 256,
            'worker_threads': 8
        }
        
        with patch.object(optimizer.config_manager, 'apply_configuration', return_value=True), \
             patch.object(optimizer, '_collect_performance_metrics') as mock_metrics:
            
            # Mock metrics collection
            mock_metrics.return_value = {
                'throughput': 500.0,
                'response_time': {'p95': 1.5},
                'error_rate': 0.02,
                'cpu': {'percent': 60.0}
            }
            
            result = await optimizer._run_experiment(parameters, 1)
            
            assert result.success
            assert result.parameters == parameters
            assert result.performance_score > 0
            assert 'throughput' in result.metrics
    
    def test_calculate_performance_score(self, optimizer):
        """Test performance score calculation"""
        metrics = {
            'throughput': 800.0,
            'response_time': {'p95': 1.2},
            'error_rate': 0.015,
            'cpu': {'percent': 65.0}
        }
        
        score = optimizer._calculate_performance_score(metrics)
        
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should be positive for reasonable metrics
    
    def test_get_metric_value(self, optimizer):
        """Test metric value extraction"""
        metrics = {
            'response_time': {'p95': 1.5},
            'simple_metric': 42.0
        }
        
        # Test nested extraction
        value = optimizer._get_metric_value(metrics, 'response_time.p95')
        assert value == 1.5
        
        # Test simple extraction
        value = optimizer._get_metric_value(metrics, 'simple_metric')
        assert value == 42.0
        
        # Test missing path
        value = optimizer._get_metric_value(metrics, 'missing.path')
        assert value is None
    
    def test_check_convergence(self, optimizer):
        """Test convergence checking"""
        # Not enough history
        assert not optimizer._check_convergence()
        
        # Add experiment history with varying scores
        for i in range(12):
            result = OptimizationResult(
                experiment_id=f"exp_{i}",
                parameters={},
                metrics={},
                performance_score=0.5 + (i % 3) * 0.1,  # Varying scores
                timestamp=datetime.utcnow(),
                duration=1.0,
                success=True
            )
            optimizer.experiment_history.append(result)
        
        # Should not converge (too much variation)
        assert not optimizer._check_convergence()
        
        # Add more stable scores
        for i in range(5):
            result = OptimizationResult(
                experiment_id=f"exp_stable_{i}",
                parameters={},
                metrics={},
                performance_score=0.8 + i * 0.01,  # Small variation
                timestamp=datetime.utcnow(),
                duration=1.0,
                success=True
            )
            optimizer.experiment_history.append(result)
        
        # Should converge now (small variation)
        assert optimizer._check_convergence()
    
    def test_get_optimization_summary(self, optimizer):
        """Test optimization summary generation"""
        # Initially no experiments
        summary = optimizer.get_optimization_summary()
        assert summary['status'] == 'no_experiments'
        
        # Add some experiment results
        for i in range(3):
            result = OptimizationResult(
                experiment_id=f"exp_{i}",
                parameters={'param': i},
                metrics={'throughput': 100 + i * 50},
                performance_score=0.5 + i * 0.1,
                timestamp=datetime.utcnow(),
                duration=1.0,
                success=True
            )
            optimizer.experiment_history.append(result)
        
        summary = optimizer.get_optimization_summary()
        
        assert summary['status'] in ['active', 'completed']
        assert summary['total_experiments'] == 3
        assert summary['successful_experiments'] == 3
        assert summary['best_score'] == 0.7
        assert summary['improvement'] > 0

@pytest.mark.integration
class TestAdaptiveOptimizationIntegration:
    """Integration tests for adaptive optimization"""
    
    @pytest.mark.asyncio
    async def test_short_optimization_session(self):
        """Test complete optimization session (short version)"""
        config = {
            'optimization_interval': 0.1,
            'experiment_duration': 0.05,
            'max_experiments': 3,
            'convergence_threshold': 0.2
        }
        
        optimizer = AdaptivePerformanceOptimizer(config)
        
        # Mock external dependencies
        with patch.object(optimizer.config_manager, 'apply_configuration', return_value=True), \
             patch.object(optimizer, '_collect_performance_metrics') as mock_metrics, \
             patch('asyncio.sleep'):
            
            # Mock metrics to show improving performance
            metrics_sequence = [
                {'throughput': 400, 'response_time': {'p95': 2.0}, 'error_rate': 0.03, 'cpu': {'percent': 70}},
                {'throughput': 500, 'response_time': {'p95': 1.5}, 'error_rate': 0.02, 'cpu': {'percent': 65}},
                {'throughput': 600, 'response_time': {'p95': 1.2}, 'error_rate': 0.015, 'cpu': {'percent': 60}}
            ]
            
            call_count = 0
            def mock_collect_metrics():
                nonlocal call_count
                metrics = metrics_sequence[call_count % len(metrics_sequence)]
                call_count += 1
                return metrics
            
            mock_metrics.side_effect = mock_collect_metrics
            
            # Start optimization
            optimizer.optimization_active = True
            
            # Run optimization loop manually for testing
            await optimizer._optimization_loop()
            
            # Check results
            assert len(optimizer.experiment_history) > 0
            assert not optimizer.optimization_active  # Should be stopped
            
            summary = optimizer.get_optimization_summary()
            assert summary['total_experiments'] > 0
            assert summary['best_score'] > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])