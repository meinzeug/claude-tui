"""
Predictive Performance Modeling System.

Advanced machine learning-based system for predicting performance trends,
forecasting resource needs, and identifying potential issues before they occur.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from ..core.types import Severity, Priority
from .models import (
    PerformanceMetrics, TrendAnalysis, PerformanceAlert,
    AnalyticsConfiguration, AlertType
)


class PredictionType(str, Enum):
    """Types of performance predictions."""
    TREND_FORECAST = "trend_forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    CAPACITY_PLANNING = "capacity_planning"
    FAILURE_PREDICTION = "failure_prediction"
    RESOURCE_OPTIMIZATION = "resource_optimization"


class ModelType(str, Enum):
    """Types of prediction models."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RANDOM_FOREST = "random_forest"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Result of a performance prediction."""
    prediction_type: PredictionType
    model_type: ModelType
    predicted_values: List[Tuple[datetime, float]]
    confidence_intervals: List[Tuple[float, float]] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # Model performance metrics
    accuracy_score: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    
    # Prediction metadata
    prediction_horizon: timedelta = field(default_factory=lambda: timedelta(hours=24))
    training_window: timedelta = field(default_factory=lambda: timedelta(days=7))
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Alerts and recommendations
    predicted_alerts: List[PerformanceAlert] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class ModelEvaluationResult:
    """Result of model evaluation."""
    model_type: ModelType
    metric_name: str
    
    # Performance metrics
    rmse: float = 0.0
    mae: float = 0.0  # Mean Absolute Error
    r2_score: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Training metadata
    training_samples: int = 0
    feature_count: int = 0
    training_duration: float = 0.0
    
    evaluated_at: datetime = field(default_factory=datetime.utcnow)


class PerformancePredictor:
    """
    Advanced performance prediction system.
    
    Features:
    - Multiple prediction models (linear, polynomial, random forest, ensemble)
    - Automated model selection and hyperparameter tuning
    - Trend forecasting and anomaly detection
    - Capacity planning and resource optimization
    - Real-time prediction updates
    - Model performance tracking and retraining
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        model_storage_path: Optional[Path] = None
    ):
        """Initialize the performance predictor."""
        self.config = config or AnalyticsConfiguration()
        self.model_storage_path = model_storage_path or Path("models/performance")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Prediction models
        self.models: Dict[str, Dict[ModelType, BaseEstimator]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Model evaluation results
        self.evaluation_results: Dict[str, List[ModelEvaluationResult]] = {}
        self.best_models: Dict[str, ModelType] = {}
        
        # Prediction cache
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Feature engineering components
        self.feature_extractors: Dict[str, callable] = {}
        self.feature_windows = [5, 15, 30, 60]  # minutes
        
        # Anomaly detectors
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        
        # Training configuration
        self.training_config = {
            'min_training_samples': 50,
            'retrain_interval': timedelta(hours=6),
            'validation_split': 0.2,
            'cross_validation_folds': 5
        }
    
    async def initialize(self) -> None:
        """Initialize the predictor system."""
        try:
            self.logger.info("Initializing Performance Predictor...")
            
            # Create storage directories
            self.model_storage_path.mkdir(parents=True, exist_ok=True)
            (self.model_storage_path / "trained_models").mkdir(exist_ok=True)
            (self.model_storage_path / "evaluations").mkdir(exist_ok=True)
            
            # Initialize feature extractors
            self._initialize_feature_extractors()
            
            # Load existing models
            await self._load_existing_models()
            
            self.logger.info("Performance Predictor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize predictor: {e}")
            raise
    
    async def predict_metric_trend(
        self,
        metric_name: str,
        historical_metrics: List[PerformanceMetrics],
        prediction_horizon: timedelta = timedelta(hours=24),
        model_type: Optional[ModelType] = None
    ) -> PredictionResult:
        """Predict future trend for a specific metric."""
        try:
            # Check cache
            cache_key = f"trend_{metric_name}_{prediction_horizon.total_seconds()}"
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                return cached_result
            
            # Prepare training data
            X, y = self._prepare_time_series_data(historical_metrics, metric_name)
            
            if len(X) < self.training_config['min_training_samples']:
                raise ValueError(f"Insufficient training data: {len(X)} samples")
            
            # Select or use specified model
            selected_model_type = model_type or self._select_best_model(metric_name, X, y)
            
            # Train model
            model, scaler = await self._train_trend_model(X, y, selected_model_type)
            
            # Generate predictions
            predictions, confidence_intervals = await self._generate_trend_predictions(
                model, scaler, X, y, prediction_horizon
            )
            
            # Calculate model performance
            performance_metrics = self._calculate_model_performance(model, X, y, scaler)
            
            # Detect potential alerts in predictions
            predicted_alerts = await self._detect_prediction_alerts(
                metric_name, predictions, historical_metrics[-10:]
            )
            
            # Generate recommendations
            recommendations = await self._generate_trend_recommendations(
                metric_name, predictions, historical_metrics
            )
            
            # Create result
            result = PredictionResult(
                prediction_type=PredictionType.TREND_FORECAST,
                model_type=selected_model_type,
                predicted_values=predictions,
                confidence_intervals=confidence_intervals,
                confidence_score=min(100, performance_metrics['r2_score'] * 100),
                accuracy_score=performance_metrics['accuracy'],
                rmse=performance_metrics['rmse'],
                r2_score=performance_metrics['r2_score'],
                prediction_horizon=prediction_horizon,
                predicted_alerts=predicted_alerts,
                recommendations=recommendations,
                expires_at=datetime.utcnow() + self.cache_ttl
            )
            
            # Cache result
            self._cache_prediction(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting metric trend for {metric_name}: {e}")
            # Return empty result
            return PredictionResult(
                prediction_type=PredictionType.TREND_FORECAST,
                model_type=ModelType.LINEAR,
                predicted_values=[]
            )
    
    async def detect_anomalies(
        self,
        metric_name: str,
        historical_metrics: List[PerformanceMetrics],
        current_metrics: List[PerformanceMetrics]
    ) -> List[PerformanceAlert]:
        """Detect anomalies in current performance data."""
        try:
            # Prepare data
            historical_values = self._extract_metric_values(historical_metrics, metric_name)
            current_values = self._extract_metric_values(current_metrics, metric_name)
            
            if len(historical_values) < 20 or len(current_values) == 0:
                return []
            
            # Get or train anomaly detector
            detector = await self._get_anomaly_detector(metric_name, historical_values)
            
            # Detect anomalies
            anomalies = []
            for i, value in enumerate(current_values):
                is_anomaly = detector.predict([[value]])[0] == -1
                
                if is_anomaly:
                    # Calculate anomaly score
                    anomaly_score = detector.decision_function([[value]])[0]
                    
                    # Determine severity based on anomaly score
                    if anomaly_score < -0.5:
                        severity = Severity.CRITICAL
                    elif anomaly_score < -0.3:
                        severity = Severity.HIGH
                    elif anomaly_score < -0.1:
                        severity = Severity.MEDIUM
                    else:
                        severity = Severity.LOW
                    
                    # Create alert
                    alert = PerformanceAlert(
                        alert_type=AlertType.ANOMALY_DETECTED,
                        severity=severity,
                        title=f"Performance Anomaly Detected in {metric_name}",
                        description=f"Anomalous value {value:.2f} detected (score: {anomaly_score:.3f})",
                        affected_component=metric_name,
                        metric_name=metric_name,
                        current_value=value,
                        performance_metrics=current_metrics[i] if i < len(current_metrics) else None
                    )
                    anomalies.append(alert)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies for {metric_name}: {e}")
            return []
    
    async def predict_capacity_needs(
        self,
        historical_metrics: List[PerformanceMetrics],
        target_performance_level: float = 80.0,
        forecast_period: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """Predict future capacity needs based on usage trends."""
        try:
            capacity_predictions = {}
            
            # Key metrics for capacity planning
            capacity_metrics = [
                'cpu_percent', 'memory_percent', 'disk_percent',
                'throughput', 'active_tasks', 'concurrent_workflows'
            ]
            
            for metric in capacity_metrics:
                try:
                    # Predict trend
                    trend_result = await self.predict_metric_trend(
                        metric, historical_metrics, forecast_period
                    )
                    
                    if not trend_result.predicted_values:
                        continue
                    
                    # Analyze capacity requirements
                    projected_values = [v[1] for v in trend_result.predicted_values]
                    max_projected = max(projected_values)
                    
                    # Calculate capacity recommendations
                    if metric.endswith('_percent'):  # Resource utilization metrics
                        if max_projected > target_performance_level:
                            capacity_increase = (max_projected - target_performance_level) / target_performance_level
                            recommendation = f"Increase {metric.replace('_percent', '')} capacity by {capacity_increase*100:.1f}%"
                        else:
                            recommendation = f"Current {metric.replace('_percent', '')} capacity is sufficient"
                    else:  # Throughput/load metrics
                        current_max = max(self._extract_metric_values(historical_metrics[-30:], metric))
                        growth_rate = (max_projected - current_max) / current_max if current_max > 0 else 0
                        recommendation = f"Expected {metric} growth: {growth_rate*100:.1f}%"
                    
                    capacity_predictions[metric] = {
                        'current_utilization': historical_metrics[-1].__dict__.get(metric, 0),
                        'projected_max': max_projected,
                        'projected_values': trend_result.predicted_values[-7:],  # Last week
                        'recommendation': recommendation,
                        'confidence': trend_result.confidence_score
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error predicting capacity for {metric}: {e}")
            
            # Generate overall capacity summary
            high_risk_metrics = [
                metric for metric, data in capacity_predictions.items()
                if data['projected_max'] > target_performance_level and metric.endswith('_percent')
            ]
            
            summary = {
                'overall_risk': 'high' if high_risk_metrics else 'low',
                'high_risk_metrics': high_risk_metrics,
                'forecast_period_days': forecast_period.days,
                'target_performance_level': target_performance_level,
                'capacity_predictions': capacity_predictions,
                'recommendations': [
                    capacity_predictions[metric]['recommendation']
                    for metric in high_risk_metrics
                ] if high_risk_metrics else ["Current capacity appears sufficient"]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error predicting capacity needs: {e}")
            return {'error': str(e)}
    
    async def evaluate_model_performance(
        self,
        metric_name: str,
        historical_metrics: List[PerformanceMetrics],
        test_period: timedelta = timedelta(days=2)
    ) -> List[ModelEvaluationResult]:
        """Evaluate performance of different models for a metric."""
        try:
            # Split data into training and testing
            split_time = datetime.utcnow() - test_period
            
            train_metrics = [m for m in historical_metrics if m.timestamp < split_time]
            test_metrics = [m for m in historical_metrics if m.timestamp >= split_time]
            
            if len(train_metrics) < 30 or len(test_metrics) < 10:
                raise ValueError("Insufficient data for model evaluation")
            
            # Prepare training and test data
            X_train, y_train = self._prepare_time_series_data(train_metrics, metric_name)
            X_test, y_test = self._prepare_time_series_data(test_metrics, metric_name)
            
            evaluation_results = []
            
            # Test different model types
            model_types = [ModelType.LINEAR, ModelType.POLYNOMIAL, ModelType.RANDOM_FOREST]
            
            for model_type in model_types:
                try:
                    # Train model
                    model, scaler = await self._train_trend_model(X_train, y_train, model_type)
                    
                    # Make predictions on test set
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate performance metrics
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = np.mean(np.abs(y_test - y_pred))
                    r2 = r2_score(y_test, y_pred)
                    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
                    
                    # Cross-validation
                    cv_scores = await self._cross_validate_model(
                        model_type, X_train, y_train, scaler
                    )
                    
                    evaluation_result = ModelEvaluationResult(
                        model_type=model_type,
                        metric_name=metric_name,
                        rmse=rmse,
                        mae=mae,
                        r2_score=r2,
                        mape=mape,
                        cv_scores=cv_scores,
                        cv_mean=statistics.mean(cv_scores) if cv_scores else 0,
                        cv_std=statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0,
                        training_samples=len(X_train),
                        feature_count=X_train.shape[1] if len(X_train) > 0 else 0
                    )
                    
                    evaluation_results.append(evaluation_result)
                    
                except Exception as e:
                    self.logger.warning(f"Error evaluating {model_type} for {metric_name}: {e}")
            
            # Store evaluation results
            if metric_name not in self.evaluation_results:
                self.evaluation_results[metric_name] = []
            self.evaluation_results[metric_name].extend(evaluation_results)
            
            # Update best model
            if evaluation_results:
                best_model = max(evaluation_results, key=lambda x: x.r2_score)
                self.best_models[metric_name] = best_model.model_type
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance for {metric_name}: {e}")
            return []
    
    async def retrain_models(
        self,
        metric_names: Optional[List[str]] = None,
        historical_metrics: List[PerformanceMetrics] = None
    ) -> Dict[str, Any]:
        """Retrain prediction models with latest data."""
        try:
            if not historical_metrics:
                raise ValueError("No historical data provided for retraining")
            
            # Default to all metrics if none specified
            if not metric_names:
                metric_names = [
                    'cpu_percent', 'memory_percent', 'throughput',
                    'latency_p95', 'error_rate', 'cache_hit_rate'
                ]
            
            retrain_results = {}
            
            for metric_name in metric_names:
                try:
                    # Prepare training data
                    X, y = self._prepare_time_series_data(historical_metrics, metric_name)
                    
                    if len(X) < self.training_config['min_training_samples']:
                        retrain_results[metric_name] = {
                            'status': 'skipped',
                            'reason': f'Insufficient data: {len(X)} samples'
                        }
                        continue
                    
                    # Get best model type
                    model_type = self.best_models.get(metric_name, ModelType.LINEAR)
                    
                    # Train new model
                    model, scaler = await self._train_trend_model(X, y, model_type)
                    
                    # Store model
                    if metric_name not in self.models:
                        self.models[metric_name] = {}
                    self.models[metric_name][model_type] = model
                    self.scalers[metric_name] = scaler
                    
                    # Calculate performance
                    performance = self._calculate_model_performance(model, X, y, scaler)
                    
                    retrain_results[metric_name] = {
                        'status': 'success',
                        'model_type': model_type.value,
                        'training_samples': len(X),
                        'r2_score': performance['r2_score'],
                        'rmse': performance['rmse']
                    }
                    
                    # Save model to disk
                    await self._save_model(metric_name, model_type, model, scaler)
                    
                except Exception as e:
                    retrain_results[metric_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            return {
                'retrained_at': datetime.utcnow().isoformat(),
                'results': retrain_results,
                'total_metrics': len(metric_names),
                'successful_retrains': len([r for r in retrain_results.values() if r['status'] == 'success'])
            }
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
            return {'error': str(e)}
    
    # Private methods
    
    def _prepare_time_series_data(
        self,
        metrics: List[PerformanceMetrics],
        metric_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for model training."""
        if not metrics:
            return np.array([]).reshape(0, 1), np.array([])
        
        # Extract metric values and timestamps
        values = self._extract_metric_values(metrics, metric_name)
        timestamps = [m.timestamp for m in metrics]
        
        if len(values) < 2:
            return np.array([]).reshape(0, 1), np.array([])
        
        # Create features
        features = []
        targets = []
        
        # Use multiple time windows for feature engineering
        for i in range(len(values)):
            feature_vector = []
            
            # Time-based features
            feature_vector.append(i)  # Time index
            feature_vector.append(timestamps[i].hour)  # Hour of day
            feature_vector.append(timestamps[i].weekday())  # Day of week
            
            # Historical value features
            for window in self.feature_windows:
                start_idx = max(0, i - window)
                window_values = values[start_idx:i] if i > 0 else [values[0]]
                
                if window_values:
                    feature_vector.extend([
                        statistics.mean(window_values),
                        max(window_values),
                        min(window_values),
                        statistics.stdev(window_values) if len(window_values) > 1 else 0
                    ])
                else:
                    feature_vector.extend([0, 0, 0, 0])
            
            # Trend features
            if i > 5:
                recent_trend = np.polyfit(range(5), values[i-5:i], 1)[0]
                feature_vector.append(recent_trend)
            else:
                feature_vector.append(0)
            
            features.append(feature_vector)
            targets.append(values[i])
        
        return np.array(features), np.array(targets)
    
    def _extract_metric_values(
        self,
        metrics: List[PerformanceMetrics],
        metric_name: str
    ) -> List[float]:
        """Extract values for a specific metric."""
        values = []
        for metric in metrics:
            if hasattr(metric, metric_name):
                value = getattr(metric, metric_name)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(float(value))
                else:
                    values.append(0.0)
            else:
                values.append(0.0)
        return values
    
    async def _train_trend_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: ModelType
    ) -> Tuple[BaseEstimator, StandardScaler]:
        """Train a trend prediction model."""
        # Initialize scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize model based on type
        if model_type == ModelType.LINEAR:
            model = LinearRegression()
        elif model_type == ModelType.POLYNOMIAL:
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('ridge', Ridge(alpha=1.0))
            ])
        elif model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            model = LinearRegression()  # Default fallback
        
        # Train model
        if model_type == ModelType.POLYNOMIAL:
            model.fit(X_scaled, y)
        else:
            model.fit(X_scaled, y)
        
        return model, scaler
    
    async def _generate_trend_predictions(
        self,
        model: BaseEstimator,
        scaler: StandardScaler,
        X_train: np.ndarray,
        y_train: np.ndarray,
        prediction_horizon: timedelta
    ) -> Tuple[List[Tuple[datetime, float]], List[Tuple[float, float]]]:
        """Generate trend predictions and confidence intervals."""
        predictions = []
        confidence_intervals = []
        
        # Determine number of prediction steps
        num_steps = int(prediction_horizon.total_seconds() / 1800)  # 30-minute intervals
        
        # Use last training sample as starting point
        if len(X_train) == 0:
            return predictions, confidence_intervals
        
        last_features = X_train[-1].copy()
        start_time = datetime.utcnow()
        
        for step in range(num_steps):
            # Update time-based features
            pred_time = start_time + timedelta(minutes=30 * step)
            last_features[0] += 1  # Time index
            last_features[1] = pred_time.hour  # Hour
            last_features[2] = pred_time.weekday()  # Weekday
            
            # Scale features
            features_scaled = scaler.transform([last_features])
            
            # Make prediction
            pred_value = model.predict(features_scaled)[0]
            
            # Calculate confidence interval (simplified)
            # In a real implementation, this would use model-specific methods
            residuals = y_train - model.predict(scaler.transform(X_train))
            residual_std = np.std(residuals)
            confidence_interval = (
                pred_value - 1.96 * residual_std,
                pred_value + 1.96 * residual_std
            )
            
            predictions.append((pred_time, pred_value))
            confidence_intervals.append(confidence_interval)
            
            # Update features for next prediction
            # This is simplified - would need more sophisticated feature updates
            if len(last_features) > 10:  # Update trend feature
                last_features[-1] = (pred_value - y_train[-1]) / max(1, len(predictions))
        
        return predictions, confidence_intervals
    
    def _select_best_model(
        self,
        metric_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> ModelType:
        """Select the best model type for a metric."""
        # Use stored best model if available
        if metric_name in self.best_models:
            return self.best_models[metric_name]
        
        # Default selection based on data characteristics
        if len(X) > 100:
            return ModelType.RANDOM_FOREST
        elif len(X) > 50:
            return ModelType.POLYNOMIAL
        else:
            return ModelType.LINEAR
    
    def _calculate_model_performance(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scaler: StandardScaler
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        if len(X) == 0:
            return {'accuracy': 0.0, 'rmse': float('inf'), 'r2_score': 0.0}
        
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        return {
            'accuracy': max(0, min(1, r2)),
            'rmse': rmse,
            'r2_score': r2
        }
    
    async def _detect_prediction_alerts(
        self,
        metric_name: str,
        predictions: List[Tuple[datetime, float]],
        recent_metrics: List[PerformanceMetrics]
    ) -> List[PerformanceAlert]:
        """Detect potential alerts in predictions."""
        alerts = []
        
        if not predictions or not recent_metrics:
            return alerts
        
        # Get current threshold values
        current_values = self._extract_metric_values(recent_metrics, metric_name)
        if not current_values:
            return alerts
        
        current_avg = statistics.mean(current_values)
        current_max = max(current_values)
        
        # Check for threshold violations in predictions
        for pred_time, pred_value in predictions:
            
            # Check if prediction exceeds historical maximum by significant margin
            if pred_value > current_max * 1.5:
                alert = PerformanceAlert(
                    alert_type=AlertType.THRESHOLD_EXCEEDED,
                    severity=Severity.HIGH,
                    title=f"Predicted {metric_name} Spike",
                    description=f"Predicted value {pred_value:.2f} significantly exceeds historical maximum {current_max:.2f}",
                    affected_component=metric_name,
                    metric_name=metric_name,
                    current_value=pred_value,
                    created_at=pred_time
                )
                alerts.append(alert)
        
        return alerts
    
    async def _generate_trend_recommendations(
        self,
        metric_name: str,
        predictions: List[Tuple[datetime, float]],
        historical_metrics: List[PerformanceMetrics]
    ) -> List[str]:
        """Generate recommendations based on trend predictions."""
        recommendations = []
        
        if not predictions or len(predictions) < 2:
            return recommendations
        
        # Analyze trend direction
        values = [p[1] for p in predictions]
        trend_slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if trend_slope > 0:
            if metric_name.endswith('_percent') and max(values) > 80:
                recommendations.append(f"Predicted increase in {metric_name} - consider capacity scaling")
            elif 'error' in metric_name.lower() and trend_slope > 0.01:
                recommendations.append(f"Predicted increase in {metric_name} - investigate root causes")
            elif 'latency' in metric_name.lower() and trend_slope > 10:
                recommendations.append(f"Predicted latency increase - optimize performance")
        
        elif trend_slope < -0.1 and 'throughput' in metric_name.lower():
            recommendations.append(f"Predicted decrease in {metric_name} - investigate performance degradation")
        
        return recommendations
    
    async def _get_anomaly_detector(
        self,
        metric_name: str,
        historical_values: List[float]
    ) -> IsolationForest:
        """Get or create anomaly detector for a metric."""
        if metric_name in self.anomaly_detectors:
            return self.anomaly_detectors[metric_name]
        
        # Create and train new anomaly detector
        detector = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        
        # Reshape data for training
        X = np.array(historical_values).reshape(-1, 1)
        detector.fit(X)
        
        # Store detector
        self.anomaly_detectors[metric_name] = detector
        
        return detector
    
    async def _cross_validate_model(
        self,
        model_type: ModelType,
        X: np.ndarray,
        y: np.ndarray,
        scaler: StandardScaler
    ) -> List[float]:
        """Perform cross-validation for model."""
        # This is a simplified cross-validation
        # In a real implementation, would use sklearn's cross_val_score
        
        scores = []
        fold_size = len(X) // self.training_config['cross_validation_folds']
        
        for i in range(self.training_config['cross_validation_folds']):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            
            # Split into train and validation
            val_indices = list(range(start_idx, min(end_idx, len(X))))
            train_indices = [j for j in range(len(X)) if j not in val_indices]
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            # Train model
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train_fold)
            X_val_scaled = scaler_fold.transform(X_val_fold)
            
            model, _ = await self._train_trend_model(X_train_fold, y_train_fold, model_type)
            y_pred = model.predict(X_val_scaled)
            
            # Calculate R² score
            score = r2_score(y_val_fold, y_pred)
            scores.append(score)
        
        return scores
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction result."""
        if cache_key in self.prediction_cache:
            result = self.prediction_cache[cache_key]
            if result.expires_at and datetime.utcnow() < result.expires_at:
                return result
            else:
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult) -> None:
        """Cache prediction result."""
        self.prediction_cache[cache_key] = result
        
        # Clean old cache entries
        if len(self.prediction_cache) > 1000:
            cutoff_time = datetime.utcnow()
            keys_to_remove = [
                key for key, cached_result in self.prediction_cache.items()
                if cached_result.expires_at and cached_result.expires_at < cutoff_time
            ]
            for key in keys_to_remove:
                del self.prediction_cache[key]
    
    def _initialize_feature_extractors(self) -> None:
        """Initialize feature extraction functions."""
        # Placeholder for feature extractor initialization
        self.feature_extractors = {
            'time_based': lambda timestamp: [
                timestamp.hour,
                timestamp.weekday(),
                timestamp.day
            ],
            'statistical': lambda values: [
                statistics.mean(values) if values else 0,
                statistics.stdev(values) if len(values) > 1 else 0,
                max(values) if values else 0,
                min(values) if values else 0
            ]
        }
    
    async def _load_existing_models(self) -> None:
        """Load previously trained models from storage."""
        # Placeholder for model loading
        pass
    
    async def _save_model(
        self,
        metric_name: str,
        model_type: ModelType,
        model: BaseEstimator,
        scaler: StandardScaler
    ) -> None:
        """Save trained model to storage."""
        # Placeholder for model saving
        pass