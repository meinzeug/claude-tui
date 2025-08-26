"""
Adaptive Performance Optimizer

Implements intelligent performance optimization with machine learning-based
parameter tuning and automated system optimization.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationTarget:
    """Optimization target configuration"""
    name: str
    metric_path: str
    target_value: float
    optimization_direction: str  # 'minimize' or 'maximize'
    weight: float = 1.0
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None

@dataclass
class OptimizationParameter:
    """Parameter that can be optimized"""
    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step_size: Any
    parameter_type: str  # 'float', 'int', 'bool', 'enum'
    enum_values: Optional[List[Any]] = None
    description: str = ""

@dataclass
class OptimizationResult:
    """Result of an optimization experiment"""
    experiment_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    performance_score: float
    timestamp: datetime
    duration: float
    success: bool
    error_message: Optional[str] = None

class PerformancePredictor:
    """Machine learning model for performance prediction"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_data = []
        
    def add_training_data(self, parameters: Dict[str, Any], performance_score: float):
        """Add training data for the model"""
        self.training_data.append({
            'parameters': parameters,
            'performance_score': performance_score
        })
    
    def train_model(self):
        """Train the performance prediction model"""
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for model training")
            return False
        
        # Prepare training data
        X = []
        y = []
        
        for data_point in self.training_data:
            features = self._extract_features(data_point['parameters'])
            X.append(features)
            y.append(data_point['performance_score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model training completed - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        self.is_trained = True
        return True
    
    def predict_performance(self, parameters: Dict[str, Any]) -> float:
        """Predict performance score for given parameters"""
        if not self.is_trained:
            return 0.0
        
        features = self._extract_features(parameters)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.model.predict(features_scaled)[0]
        return max(0.0, min(1.0, prediction))  # Clamp to [0, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        for i, importance in enumerate(self.model.feature_importances_):
            if i < len(self.feature_names):
                importance_dict[self.feature_names[i]] = importance
        
        return importance_dict
    
    def _extract_features(self, parameters: Dict[str, Any]) -> List[float]:
        """Extract numerical features from parameters"""
        if not self.feature_names:
            self.feature_names = sorted(parameters.keys())
        
        features = []
        for feature_name in self.feature_names:
            value = parameters.get(feature_name, 0)
            
            # Convert to numerical value
            if isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Simple hash-based conversion for categorical values
                features.append(float(hash(value) % 1000) / 1000.0)
            else:
                features.append(0.0)
        
        return features
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_data': self.training_data
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.training_data = model_data['training_data']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

class BayesianOptimizer:
    """Bayesian optimization for parameter tuning"""
    
    def __init__(self, parameters: List[OptimizationParameter]):
        self.parameters = {p.name: p for p in parameters}
        self.exploration_history = []
        self.acquisition_function = 'expected_improvement'
        
    def suggest_parameters(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest parameter values for next experiments"""
        suggestions = []
        
        for _ in range(n_suggestions):
            if len(self.exploration_history) < 5:
                # Random exploration for initial experiments
                suggestion = self._random_suggestion()
            else:
                # Bayesian optimization suggestion
                suggestion = self._bayesian_suggestion()
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def update_results(self, parameters: Dict[str, Any], performance_score: float):
        """Update optimization history with new results"""
        self.exploration_history.append({
            'parameters': parameters.copy(),
            'performance_score': performance_score,
            'timestamp': datetime.utcnow()
        })
    
    def _random_suggestion(self) -> Dict[str, Any]:
        """Generate random parameter suggestion"""
        suggestion = {}
        
        for param_name, param_config in self.parameters.items():
            if param_config.parameter_type == 'float':
                suggestion[param_name] = np.random.uniform(param_config.min_value, param_config.max_value)
            elif param_config.parameter_type == 'int':
                suggestion[param_name] = np.random.randint(param_config.min_value, param_config.max_value + 1)
            elif param_config.parameter_type == 'bool':
                suggestion[param_name] = np.random.choice([True, False])
            elif param_config.parameter_type == 'enum':
                suggestion[param_name] = np.random.choice(param_config.enum_values)
            else:
                suggestion[param_name] = param_config.current_value
        
        return suggestion
    
    def _bayesian_suggestion(self) -> Dict[str, Any]:
        """Generate Bayesian optimization suggestion"""
        # Simplified Bayesian optimization
        # In production, would use libraries like scikit-optimize or GPyOpt
        
        best_score = max(h['performance_score'] for h in self.exploration_history)
        best_params = max(self.exploration_history, key=lambda x: x['performance_score'])['parameters']
        
        # Add small random perturbations to best parameters
        suggestion = {}
        for param_name, param_config in self.parameters.items():
            best_value = best_params.get(param_name, param_config.current_value)
            
            if param_config.parameter_type == 'float':
                perturbation = np.random.normal(0, (param_config.max_value - param_config.min_value) * 0.1)
                new_value = np.clip(best_value + perturbation, param_config.min_value, param_config.max_value)
                suggestion[param_name] = new_value
            elif param_config.parameter_type == 'int':
                perturbation = np.random.randint(-2, 3)  # Small integer perturbation
                new_value = np.clip(best_value + perturbation, param_config.min_value, param_config.max_value)
                suggestion[param_name] = int(new_value)
            elif param_config.parameter_type == 'bool':
                # Occasionally flip boolean values
                suggestion[param_name] = not best_value if np.random.random() < 0.3 else best_value
            elif param_config.parameter_type == 'enum':
                # Sometimes try different enum values
                if np.random.random() < 0.4:
                    suggestion[param_name] = np.random.choice(param_config.enum_values)
                else:
                    suggestion[param_name] = best_value
            else:
                suggestion[param_name] = best_value
        
        return suggestion
    
    def get_best_parameters(self) -> Tuple[Dict[str, Any], float]:
        """Get best parameters found so far"""
        if not self.exploration_history:
            return {}, 0.0
        
        best_experiment = max(self.exploration_history, key=lambda x: x['performance_score'])
        return best_experiment['parameters'], best_experiment['performance_score']

class SystemConfigurationManager:
    """Manages system configuration changes for optimization"""
    
    def __init__(self):
        self.config_backup = {}
        self.applied_changes = {}
        
    async def apply_configuration(self, parameters: Dict[str, Any]) -> bool:
        """Apply configuration parameters to the system"""
        try:
            # Backup current configuration
            await self._backup_current_config()
            
            # Apply each parameter
            for param_name, param_value in parameters.items():
                success = await self._apply_parameter(param_name, param_value)
                if not success:
                    logger.error(f"Failed to apply parameter {param_name}")
                    await self._rollback_configuration()
                    return False
                
                self.applied_changes[param_name] = param_value
            
            logger.info(f"Applied {len(parameters)} configuration parameters")
            return True
            
        except Exception as e:
            logger.error(f"Configuration application failed: {e}")
            await self._rollback_configuration()
            return False
    
    async def _apply_parameter(self, param_name: str, param_value: Any) -> bool:
        """Apply individual parameter"""
        # This would contain actual system configuration logic
        # Examples:
        # - Database connection pool size
        # - Cache size limits
        # - Thread pool configurations
        # - Timeout values
        # - Buffer sizes
        
        config_mappings = {
            'db_pool_size': self._set_db_pool_size,
            'cache_size_mb': self._set_cache_size,
            'worker_threads': self._set_worker_threads,
            'request_timeout': self._set_request_timeout,
            'batch_size': self._set_batch_size,
            'gc_threshold': self._set_gc_threshold,
            'compression_enabled': self._set_compression,
            'connection_keep_alive': self._set_keep_alive
        }
        
        if param_name in config_mappings:
            return await config_mappings[param_name](param_value)
        else:
            logger.warning(f"Unknown parameter: {param_name}")
            return True  # Don't fail for unknown parameters
    
    async def _set_db_pool_size(self, size: int) -> bool:
        """Set database connection pool size"""
        # Simulate database pool configuration
        logger.info(f"Setting database pool size to {size}")
        await asyncio.sleep(0.1)  # Simulate config change time
        return True
    
    async def _set_cache_size(self, size_mb: int) -> bool:
        """Set cache size limit"""
        logger.info(f"Setting cache size to {size_mb}MB")
        await asyncio.sleep(0.1)
        return True
    
    async def _set_worker_threads(self, threads: int) -> bool:
        """Set number of worker threads"""
        logger.info(f"Setting worker threads to {threads}")
        await asyncio.sleep(0.1)
        return True
    
    async def _set_request_timeout(self, timeout: float) -> bool:
        """Set request timeout"""
        logger.info(f"Setting request timeout to {timeout}s")
        await asyncio.sleep(0.1)
        return True
    
    async def _set_batch_size(self, size: int) -> bool:
        """Set batch processing size"""
        logger.info(f"Setting batch size to {size}")
        await asyncio.sleep(0.1)
        return True
    
    async def _set_gc_threshold(self, threshold: int) -> bool:
        """Set garbage collection threshold"""
        logger.info(f"Setting GC threshold to {threshold}")
        await asyncio.sleep(0.1)
        return True
    
    async def _set_compression(self, enabled: bool) -> bool:
        """Set compression enabled/disabled"""
        logger.info(f"Setting compression: {'enabled' if enabled else 'disabled'}")
        await asyncio.sleep(0.1)
        return True
    
    async def _set_keep_alive(self, enabled: bool) -> bool:
        """Set connection keep-alive"""
        logger.info(f"Setting connection keep-alive: {'enabled' if enabled else 'disabled'}")
        await asyncio.sleep(0.1)
        return True
    
    async def _backup_current_config(self):
        """Backup current configuration"""
        # Simulate backing up current configuration
        self.config_backup = {
            'timestamp': datetime.utcnow(),
            'config': {
                'db_pool_size': 20,
                'cache_size_mb': 256,
                'worker_threads': 8,
                'request_timeout': 30.0,
                'batch_size': 100,
                'gc_threshold': 1000,
                'compression_enabled': True,
                'connection_keep_alive': True
            }
        }
        logger.info("Configuration backed up")
    
    async def _rollback_configuration(self):
        """Rollback to previous configuration"""
        if self.config_backup:
            logger.info("Rolling back configuration changes")
            # Apply backup configuration
            for param_name, param_value in self.config_backup['config'].items():
                await self._apply_parameter(param_name, param_value)
            
            self.applied_changes.clear()

class AdaptivePerformanceOptimizer:
    """Main adaptive performance optimization system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.predictor = PerformancePredictor()
        self.optimizer = BayesianOptimizer(self._load_optimization_parameters())
        self.config_manager = SystemConfigurationManager()
        
        # Optimization state
        self.optimization_active = False
        self.current_experiment = None
        self.experiment_history = []
        self.best_configuration = {}
        
        # Performance targets
        self.optimization_targets = self._load_optimization_targets()
        
        # Try to load existing model
        model_path = self.config.get('model_path', 'performance_model.pkl')
        if Path(model_path).exists():
            self.predictor.load_model(model_path)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default optimization configuration"""
        return {
            'optimization_interval': 300,  # 5 minutes between optimizations
            'experiment_duration': 60,     # 1 minute per experiment
            'max_experiments': 50,         # Maximum experiments per session
            'convergence_threshold': 0.05, # Stop when improvement < 5%
            'safety_mode': True,           # Enable safety constraints
            'model_path': 'performance_model.pkl',
            'backup_interval': 10          # Backup every 10 experiments
        }
    
    def _load_optimization_parameters(self) -> List[OptimizationParameter]:
        """Load parameters that can be optimized"""
        return [
            OptimizationParameter(
                name='db_pool_size',
                current_value=20,
                min_value=5,
                max_value=100,
                step_size=5,
                parameter_type='int',
                description='Database connection pool size'
            ),
            OptimizationParameter(
                name='cache_size_mb',
                current_value=256,
                min_value=64,
                max_value=2048,
                step_size=64,
                parameter_type='int',
                description='Cache size in megabytes'
            ),
            OptimizationParameter(
                name='worker_threads',
                current_value=8,
                min_value=2,
                max_value=32,
                step_size=2,
                parameter_type='int',
                description='Number of worker threads'
            ),
            OptimizationParameter(
                name='request_timeout',
                current_value=30.0,
                min_value=5.0,
                max_value=120.0,
                step_size=5.0,
                parameter_type='float',
                description='Request timeout in seconds'
            ),
            OptimizationParameter(
                name='batch_size',
                current_value=100,
                min_value=10,
                max_value=1000,
                step_size=10,
                parameter_type='int',
                description='Batch processing size'
            ),
            OptimizationParameter(
                name='compression_enabled',
                current_value=True,
                min_value=False,
                max_value=True,
                step_size=None,
                parameter_type='bool',
                description='Enable response compression'
            )
        ]
    
    def _load_optimization_targets(self) -> List[OptimizationTarget]:
        """Load optimization targets"""
        return [
            OptimizationTarget(
                name='throughput',
                metric_path='throughput',
                target_value=1000.0,
                optimization_direction='maximize',
                weight=0.4
            ),
            OptimizationTarget(
                name='latency_p95',
                metric_path='response_time.p95',
                target_value=1.0,
                optimization_direction='minimize',
                weight=0.3
            ),
            OptimizationTarget(
                name='error_rate',
                metric_path='error_rate',
                target_value=0.01,
                optimization_direction='minimize',
                weight=0.2
            ),
            OptimizationTarget(
                name='cpu_usage',
                metric_path='cpu.percent',
                target_value=70.0,
                optimization_direction='minimize',
                weight=0.1,
                constraint_max=90.0
            )
        ]
    
    async def start_optimization(self):
        """Start adaptive performance optimization"""
        if self.optimization_active:
            logger.warning("Optimization is already active")
            return
        
        self.optimization_active = True
        logger.info("Starting adaptive performance optimization")
        
        try:
            await self._optimization_loop()
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
        finally:
            self.optimization_active = False
    
    async def stop_optimization(self):
        """Stop optimization"""
        logger.info("Stopping performance optimization")
        self.optimization_active = False
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        experiment_count = 0
        max_experiments = self.config.get('max_experiments', 50)
        
        while self.optimization_active and experiment_count < max_experiments:
            try:
                # Get parameter suggestions
                suggestions = self.optimizer.suggest_parameters(n_suggestions=1)
                parameters = suggestions[0]
                
                # Run experiment
                result = await self._run_experiment(parameters, experiment_count)
                
                if result.success:
                    # Update optimizer and predictor
                    self.optimizer.update_results(parameters, result.performance_score)
                    self.predictor.add_training_data(parameters, result.performance_score)
                    
                    # Store experiment result
                    self.experiment_history.append(result)
                    
                    # Update best configuration
                    if not self.best_configuration or result.performance_score > self.best_configuration.get('score', 0):
                        self.best_configuration = {
                            'parameters': parameters,
                            'score': result.performance_score,
                            'experiment_id': result.experiment_id
                        }
                        logger.info(f"New best configuration found: score={result.performance_score:.3f}")
                    
                    # Train model periodically
                    if len(self.predictor.training_data) >= 10 and experiment_count % 5 == 0:
                        self.predictor.train_model()
                    
                    # Save backup periodically
                    if experiment_count % self.config.get('backup_interval', 10) == 0:
                        await self._save_optimization_state()
                
                experiment_count += 1
                
                # Check convergence
                if await self._check_convergence():
                    logger.info("Optimization converged")
                    break
                
                # Wait before next experiment
                await asyncio.sleep(self.config.get('optimization_interval', 300))
                
            except Exception as e:
                logger.error(f"Experiment {experiment_count} failed: {e}")
                experiment_count += 1
        
        # Apply best configuration
        if self.best_configuration:
            await self._apply_best_configuration()
        
        # Save final state
        await self._save_optimization_state()
        
        logger.info(f"Optimization completed after {experiment_count} experiments")
    
    async def _run_experiment(self, parameters: Dict[str, Any], experiment_id: int) -> OptimizationResult:
        """Run single optimization experiment"""
        start_time = time.time()
        experiment_id_str = f"exp_{experiment_id}_{int(start_time)}"
        
        logger.info(f"Running experiment {experiment_id}: {parameters}")
        
        try:
            # Apply configuration
            config_success = await self.config_manager.apply_configuration(parameters)
            if not config_success:
                return OptimizationResult(
                    experiment_id=experiment_id_str,
                    parameters=parameters,
                    metrics={},
                    performance_score=0.0,
                    timestamp=datetime.utcnow(),
                    duration=time.time() - start_time,
                    success=False,
                    error_message="Configuration application failed"
                )
            
            # Wait for system to stabilize
            await asyncio.sleep(10)
            
            # Collect performance metrics
            metrics = await self._collect_performance_metrics()
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(metrics)
            
            duration = time.time() - start_time
            
            return OptimizationResult(
                experiment_id=experiment_id_str,
                parameters=parameters,
                metrics=metrics,
                performance_score=performance_score,
                timestamp=datetime.utcnow(),
                duration=duration,
                success=True
            )
            
        except Exception as e:
            return OptimizationResult(
                experiment_id=experiment_id_str,
                parameters=parameters,
                metrics={},
                performance_score=0.0,
                timestamp=datetime.utcnow(),
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        # Simulate metrics collection
        # In real implementation, this would collect from monitoring systems
        await asyncio.sleep(5)  # Simulate collection time
        
        return {
            'throughput': 500 + 300 * np.random.random(),
            'response_time': {
                'avg': 0.5 + 0.3 * np.random.random(),
                'p95': 1.0 + 0.5 * np.random.random(),
                'p99': 1.5 + 0.8 * np.random.random()
            },
            'error_rate': 0.005 + 0.02 * np.random.random(),
            'cpu': {
                'percent': 40 + 30 * np.random.random()
            },
            'memory': {
                'percent': 50 + 20 * np.random.random()
            }
        }
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite performance score"""
        total_score = 0.0
        total_weight = 0.0
        
        for target in self.optimization_targets:
            metric_value = self._get_metric_value(metrics, target.metric_path)
            
            if metric_value is not None:
                # Check constraints
                if target.constraint_min is not None and metric_value < target.constraint_min:
                    return 0.0  # Constraint violation
                if target.constraint_max is not None and metric_value > target.constraint_max:
                    return 0.0  # Constraint violation
                
                # Calculate normalized score
                if target.optimization_direction == 'maximize':
                    # Higher values are better
                    normalized_score = min(1.0, metric_value / target.target_value)
                else:  # minimize
                    # Lower values are better
                    if metric_value <= target.target_value:
                        normalized_score = 1.0
                    else:
                        normalized_score = max(0.0, target.target_value / metric_value)
                
                total_score += normalized_score * target.weight
                total_weight += target.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_metric_value(self, metrics: Dict[str, Any], path: str) -> Optional[float]:
        """Get nested metric value using dot notation"""
        keys = path.split('.')
        value = metrics
        
        try:
            for key in keys:
                value = value[key]
            return float(value) if value is not None else None
        except (KeyError, TypeError, ValueError):
            return None
    
    async def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if len(self.experiment_history) < 10:
            return False
        
        # Get recent performance scores
        recent_scores = [exp.performance_score for exp in self.experiment_history[-10:]]
        
        # Check if improvement has stagnated
        max_score = max(recent_scores)
        min_score = min(recent_scores)
        improvement_range = max_score - min_score
        
        convergence_threshold = self.config.get('convergence_threshold', 0.05)
        
        return improvement_range < convergence_threshold
    
    async def _apply_best_configuration(self):
        """Apply the best configuration found"""
        if not self.best_configuration:
            logger.warning("No best configuration to apply")
            return
        
        logger.info(f"Applying best configuration: score={self.best_configuration['score']:.3f}")
        
        await self.config_manager.apply_configuration(self.best_configuration['parameters'])
        
        logger.info("Best configuration applied successfully")
    
    async def _save_optimization_state(self):
        """Save optimization state to files"""
        # Save model
        model_path = self.config.get('model_path', 'performance_model.pkl')
        self.predictor.save_model(model_path)
        
        # Save optimization history
        history_data = {
            'experiment_history': [asdict(exp) for exp in self.experiment_history],
            'best_configuration': self.best_configuration,
            'optimization_targets': [asdict(target) for target in self.optimization_targets],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        history_file = f"optimization_history_{int(time.time())}.json"
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Optimization state saved to {history_file}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        if not self.experiment_history:
            return {'status': 'no_experiments'}
        
        scores = [exp.performance_score for exp in self.experiment_history if exp.success]
        
        if not scores:
            return {'status': 'no_successful_experiments'}
        
        return {
            'status': 'active' if self.optimization_active else 'completed',
            'total_experiments': len(self.experiment_history),
            'successful_experiments': len(scores),
            'best_score': max(scores),
            'average_score': np.mean(scores),
            'improvement': (max(scores) - scores[0]) / scores[0] if len(scores) > 1 else 0,
            'best_configuration': self.best_configuration,
            'feature_importance': self.predictor.get_feature_importance() if self.predictor.is_trained else {},
            'last_updated': datetime.utcnow().isoformat()
        }

# Example usage
async def main():
    """Example usage of adaptive performance optimizer"""
    optimizer = AdaptivePerformanceOptimizer()
    
    try:
        # Start optimization (run for demo period)
        optimization_task = asyncio.create_task(optimizer.start_optimization())
        
        # Let it run for a demo period
        await asyncio.sleep(120)  # 2 minutes
        
        # Stop optimization
        await optimizer.stop_optimization()
        
        # Get summary
        summary = optimizer.get_optimization_summary()
        print(f"Optimization completed:")
        print(f"- Total experiments: {summary.get('total_experiments', 0)}")
        print(f"- Best score: {summary.get('best_score', 0):.3f}")
        print(f"- Improvement: {summary.get('improvement', 0):.1%}")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        await optimizer.stop_optimization()

if __name__ == "__main__":
    asyncio.run(main())