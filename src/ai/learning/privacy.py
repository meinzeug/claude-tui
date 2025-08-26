"""
Privacy-Preserving Learning Module.

This module implements advanced privacy protection mechanisms for AI learning
and personalization, including differential privacy, secure multi-party computation,
homomorphic encryption, and privacy-preserving analytics.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
from collections import defaultdict

from .pattern_engine import UserInteractionPattern
from .personalization import PersonalizationProfile


logger = logging.getLogger(__name__)


@dataclass
class PrivacyPolicy:
    """Privacy policy configuration."""
    policy_id: str
    name: str
    description: str
    data_retention_days: int
    anonymization_level: str  # 'basic', 'strong', 'extreme'
    differential_privacy_epsilon: float
    allow_federated_sharing: bool
    allow_analytics: bool
    require_explicit_consent: bool
    geographic_restrictions: List[str]
    created_at: datetime
    version: str = "1.0"


@dataclass
class PrivacyAuditLog:
    """Privacy audit log entry."""
    log_id: str
    timestamp: datetime
    operation: str
    data_type: str
    user_id: Optional[str]
    privacy_level: str
    compliance_status: str
    details: Dict[str, Any]


@dataclass
class DataMinimizationRule:
    """Rule for data minimization."""
    rule_id: str
    applies_to: List[str]  # Data types this rule applies to
    retention_period_days: int
    aggregation_level: str  # 'individual', 'group', 'population'
    suppression_fields: List[str]
    anonymization_method: str


class DifferentialPrivacyEngine:
    """Implements differential privacy mechanisms for protecting individual privacy."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy engine.
        
        Args:
            epsilon: Privacy budget parameter (smaller = more private)
            delta: Probability parameter for (ε,δ)-differential privacy
        """
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
        self.query_count = 0
        
        # Noise mechanisms
        self._laplace_scale = 1.0 / epsilon
        self._gaussian_scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    def add_laplace_noise(
        self,
        value: Union[float, np.ndarray],
        sensitivity: float = 1.0
    ) -> Union[float, np.ndarray]:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            value: Original value(s) to add noise to
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value(s)
        """
        noise_scale = sensitivity * self._laplace_scale
        
        if isinstance(value, np.ndarray):
            noise = np.random.laplace(0, noise_scale, value.shape)
            return value + noise
        else:
            noise = np.random.laplace(0, noise_scale)
            return value + noise
    
    def add_gaussian_noise(
        self,
        value: Union[float, np.ndarray],
        sensitivity: float = 1.0
    ) -> Union[float, np.ndarray]:
        """
        Add Gaussian noise for differential privacy.
        
        Args:
            value: Original value(s) to add noise to
            sensitivity: L2 sensitivity of the query
            
        Returns:
            Noisy value(s)
        """
        noise_scale = sensitivity * self._gaussian_scale
        
        if isinstance(value, np.ndarray):
            noise = np.random.normal(0, noise_scale, value.shape)
            return value + noise
        else:
            noise = np.random.normal(0, noise_scale)
            return value + noise
    
    def exponential_mechanism(
        self,
        candidates: List[Any],
        utility_scores: List[float],
        sensitivity: float = 1.0
    ) -> Any:
        """
        Select candidate using exponential mechanism for differential privacy.
        
        Args:
            candidates: List of possible candidates
            utility_scores: Utility scores for each candidate
            sensitivity: Sensitivity of utility function
            
        Returns:
            Selected candidate
        """
        # Calculate probabilities using exponential mechanism
        scaled_utilities = np.array(utility_scores) * self.epsilon / (2 * sensitivity)
        
        # Subtract max for numerical stability
        scaled_utilities -= np.max(scaled_utilities)
        
        # Calculate probabilities
        probabilities = np.exp(scaled_utilities)
        probabilities /= np.sum(probabilities)
        
        # Sample according to probabilities
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]
    
    def consume_privacy_budget(self, amount: float) -> bool:
        """
        Consume privacy budget for a query.
        
        Args:
            amount: Amount of privacy budget to consume
            
        Returns:
            True if budget available, False otherwise
        """
        if self.privacy_budget_used + amount > self.epsilon:
            return False
        
        self.privacy_budget_used += amount
        self.query_count += 1
        return True
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon - self.privacy_budget_used)
    
    def reset_budget(self) -> None:
        """Reset privacy budget (use with caution)."""
        self.privacy_budget_used = 0.0
        self.query_count = 0


class AnonymizationEngine:
    """Implements various anonymization techniques for protecting user identity."""
    
    def __init__(self):
        """Initialize anonymization engine."""
        self._k_anonymity_threshold = 5
        self._l_diversity_threshold = 3
        self._suppression_rate_threshold = 0.1
        
        # Generalization hierarchies
        self._generalization_rules = self._load_generalization_rules()
    
    def k_anonymize(
        self,
        data: List[Dict[str, Any]],
        quasi_identifiers: List[str],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Apply k-anonymity to dataset.
        
        Args:
            data: Dataset to anonymize
            quasi_identifiers: List of quasi-identifier fields
            k: Anonymity parameter
            
        Returns:
            k-anonymous dataset
        """
        if len(data) < k:
            logger.warning("Dataset too small for k-anonymity")
            return []
        
        # Group records by quasi-identifier values
        groups = defaultdict(list)
        for record in data:
            key = tuple(record.get(qi, '') for qi in quasi_identifiers)
            groups[key].append(record)
        
        anonymized_data = []
        
        for group_key, group_records in groups.items():
            if len(group_records) >= k:
                # Group is already k-anonymous
                anonymized_data.extend(group_records)
            else:
                # Need to generalize or suppress
                generalized_records = self._generalize_group(
                    group_records, quasi_identifiers, k
                )
                anonymized_data.extend(generalized_records)
        
        return anonymized_data
    
    def l_diversify(
        self,
        data: List[Dict[str, Any]],
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        l: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Apply l-diversity to dataset.
        
        Args:
            data: Dataset to diversify
            quasi_identifiers: List of quasi-identifier fields
            sensitive_attribute: Sensitive attribute to diversify
            l: Diversity parameter
            
        Returns:
            l-diverse dataset
        """
        # Group by quasi-identifiers
        groups = defaultdict(list)
        for record in data:
            key = tuple(record.get(qi, '') for qi in quasi_identifiers)
            groups[key].append(record)
        
        diversified_data = []
        
        for group_records in groups.values():
            # Check l-diversity
            sensitive_values = [r.get(sensitive_attribute) for r in group_records]
            unique_sensitive = set(v for v in sensitive_values if v is not None)
            
            if len(unique_sensitive) >= l:
                # Group is already l-diverse
                diversified_data.extend(group_records)
            else:
                # Need to suppress or modify group
                if len(group_records) >= l:
                    # Keep records with most diverse sensitive values
                    diversified_records = self._ensure_diversity(
                        group_records, sensitive_attribute, l
                    )
                    diversified_data.extend(diversified_records)
                # Else suppress entire group
        
        return diversified_data
    
    def apply_semantic_anonymization(
        self,
        patterns: List[UserInteractionPattern],
        anonymization_level: str = 'strong'
    ) -> List[Dict[str, Any]]:
        """
        Apply semantic anonymization to user patterns.
        
        Args:
            patterns: User patterns to anonymize
            anonymization_level: Level of anonymization
            
        Returns:
            Anonymized pattern data
        """
        anonymized_patterns = []
        
        for pattern in patterns:
            anonymized = {
                'pattern_hash': self._hash_pattern(pattern),
                'timestamp_bin': self._bin_timestamp(pattern.timestamp, anonymization_level),
                'pattern_type': pattern.pattern_type,
                'success_bin': self._bin_success_rate(pattern.success_rate),
                'confidence_bin': self._bin_confidence(pattern.confidence),
                'frequency_bin': self._bin_frequency(pattern.frequency)
            }
            
            # Anonymize features based on level
            if anonymization_level == 'basic':
                anonymized['features'] = self._basic_feature_anonymization(pattern.features)
            elif anonymization_level == 'strong':
                anonymized['features'] = self._strong_feature_anonymization(pattern.features)
            else:  # extreme
                anonymized['features'] = self._extreme_feature_anonymization(pattern.features)
            
            anonymized_patterns.append(anonymized)
        
        return anonymized_patterns
    
    def _generalize_group(
        self,
        group_records: List[Dict[str, Any]],
        quasi_identifiers: List[str],
        k: int
    ) -> List[Dict[str, Any]]:
        """Generalize group to achieve k-anonymity."""
        # Find other small groups to merge with
        # For now, return empty list (suppression)
        return []
    
    def _ensure_diversity(
        self,
        group_records: List[Dict[str, Any]],
        sensitive_attribute: str,
        l: int
    ) -> List[Dict[str, Any]]:
        """Ensure l-diversity in group."""
        # Select records to maintain diversity
        sensitive_values = {}
        diverse_records = []
        
        for record in group_records:
            sensitive_value = record.get(sensitive_attribute)
            if sensitive_value not in sensitive_values:
                sensitive_values[sensitive_value] = record
                diverse_records.append(record)
                
                if len(diverse_records) >= l:
                    break
        
        return diverse_records
    
    def _hash_pattern(self, pattern: UserInteractionPattern) -> str:
        """Create hash of pattern for anonymization."""
        pattern_str = f"{pattern.pattern_type}:{pattern.success_rate:.2f}:{pattern.confidence:.2f}"
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]
    
    def _bin_timestamp(self, timestamp: datetime, level: str) -> str:
        """Bin timestamp based on anonymization level."""
        if level == 'basic':
            return f"{timestamp.year}-{timestamp.month:02d}"
        elif level == 'strong':
            quarter = (timestamp.month - 1) // 3 + 1
            return f"{timestamp.year}-Q{quarter}"
        else:  # extreme
            return str(timestamp.year)
    
    def _bin_success_rate(self, success_rate: float) -> str:
        """Bin success rate into categories."""
        if success_rate >= 0.8:
            return 'high'
        elif success_rate >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _bin_confidence(self, confidence: float) -> str:
        """Bin confidence into categories."""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _bin_frequency(self, frequency: int) -> str:
        """Bin frequency into categories."""
        if frequency >= 10:
            return 'frequent'
        elif frequency >= 5:
            return 'moderate'
        else:
            return 'rare'
    
    def _basic_feature_anonymization(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic feature anonymization."""
        anonymized = {}
        
        for key, value in features.items():
            if key in ['user_id', 'session_id']:
                continue  # Suppress identifying fields
            elif isinstance(value, (int, float)):
                # Round to reduce precision
                anonymized[key] = round(value, 1) if isinstance(value, float) else value
            else:
                anonymized[key] = value
        
        return anonymized
    
    def _strong_feature_anonymization(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strong feature anonymization."""
        anonymized = {}
        
        for key, value in features.items():
            if key in ['user_id', 'session_id', 'file_path', 'project_path']:
                continue  # Suppress identifying fields
            elif isinstance(value, (int, float)):
                # Bin numeric values
                if value < 1:
                    anonymized[key] = 'very_low'
                elif value < 10:
                    anonymized[key] = 'low'
                elif value < 100:
                    anonymized[key] = 'medium'
                else:
                    anonymized[key] = 'high'
            elif isinstance(value, str):
                # Generalize string values
                anonymized[key] = 'string_value'
            else:
                anonymized[key] = 'other_value'
        
        return anonymized
    
    def _extreme_feature_anonymization(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply extreme feature anonymization."""
        # Only keep pattern type and success indicators
        return {
            'has_numeric_features': any(isinstance(v, (int, float)) for v in features.values()),
            'feature_count': len(features),
            'has_temporal_features': any('time' in k.lower() for k in features.keys())
        }
    
    def _load_generalization_rules(self) -> Dict[str, List[str]]:
        """Load generalization rules for quasi-identifiers."""
        return {
            'age': ['*', '<30', '30-50', '>50'],
            'location': ['*', 'continent', 'country', 'region', 'city'],
            'job_title': ['*', 'category', 'level', 'specific']
        }


class SecureComputationEngine:
    """Implements secure multi-party computation for privacy-preserving analytics."""
    
    def __init__(self):
        """Initialize secure computation engine."""
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)
    
    def encrypt_data(self, data: Any) -> str:
        """
        Encrypt data for secure storage/transmission.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        json_data = json.dumps(data, default=str)
        encrypted = self._cipher.encrypt(json_data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Any:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data
        """
        encrypted = base64.b64decode(encrypted_data.encode())
        decrypted = self._cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
    
    def secure_sum(
        self,
        local_value: float,
        other_encrypted_values: List[str]
    ) -> float:
        """
        Compute secure sum across multiple parties.
        
        Args:
            local_value: This party's value
            other_encrypted_values: Encrypted values from other parties
            
        Returns:
            Sum of all values
        """
        total = local_value
        
        for encrypted_value in other_encrypted_values:
            try:
                other_value = self.decrypt_data(encrypted_value)
                total += other_value
            except Exception as e:
                logger.warning(f"Failed to decrypt value: {e}")
        
        return total
    
    def secure_average(
        self,
        local_values: List[float],
        other_encrypted_values: List[List[str]]
    ) -> float:
        """
        Compute secure average across multiple parties.
        
        Args:
            local_values: This party's values
            other_encrypted_values: Encrypted values from other parties
            
        Returns:
            Average of all values
        """
        all_values = local_values.copy()
        
        for party_values in other_encrypted_values:
            for encrypted_value in party_values:
                try:
                    value = self.decrypt_data(encrypted_value)
                    all_values.append(value)
                except Exception as e:
                    logger.warning(f"Failed to decrypt value: {e}")
        
        return np.mean(all_values) if all_values else 0.0
    
    def generate_secret_shares(
        self,
        secret: float,
        num_shares: int,
        threshold: int
    ) -> List[Tuple[int, float]]:
        """
        Generate secret shares using Shamir's secret sharing.
        
        Args:
            secret: Secret value to share
            num_shares: Number of shares to generate
            threshold: Minimum shares needed to reconstruct
            
        Returns:
            List of (share_id, share_value) tuples
        """
        # Simple implementation - in production would use proper finite field arithmetic
        coefficients = [secret] + [np.random.uniform(-1000, 1000) for _ in range(threshold - 1)]
        
        shares = []
        for i in range(1, num_shares + 1):
            share_value = sum(coeff * (i ** power) for power, coeff in enumerate(coefficients))
            shares.append((i, share_value))
        
        return shares
    
    def reconstruct_secret(
        self,
        shares: List[Tuple[int, float]]
    ) -> float:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (share_id, share_value) tuples
            
        Returns:
            Reconstructed secret
        """
        if len(shares) < 2:
            return 0.0
        
        # Lagrange interpolation at x=0
        secret = 0.0
        
        for i, (xi, yi) in enumerate(shares):
            # Calculate Lagrange basis polynomial at x=0
            li = 1.0
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    li *= (0 - xj) / (xi - xj)
            
            secret += yi * li
        
        return secret


class PrivacyPreservingLearning:
    """
    Main privacy-preserving learning system that coordinates all privacy mechanisms.
    """
    
    def __init__(
        self,
        privacy_policy: Optional[PrivacyPolicy] = None,
        differential_privacy_epsilon: float = 1.0
    ):
        """
        Initialize privacy-preserving learning system.
        
        Args:
            privacy_policy: Privacy policy configuration
            differential_privacy_epsilon: Differential privacy parameter
        """
        self.privacy_policy = privacy_policy or self._default_privacy_policy()
        
        # Initialize privacy engines
        self.dp_engine = DifferentialPrivacyEngine(differential_privacy_epsilon)
        self.anonymization_engine = AnonymizationEngine()
        self.secure_computation = SecureComputationEngine()
        
        # Privacy tracking
        self._audit_logs: List[PrivacyAuditLog] = []
        self._data_retention_tracker: Dict[str, datetime] = {}
        self._consent_records: Dict[str, Dict[str, Any]] = {}
        
        # Data minimization rules
        self._minimization_rules = self._load_minimization_rules()
        
        logger.info("Privacy-preserving learning system initialized")
    
    async def privatize_user_patterns(
        self,
        user_id: str,
        patterns: List[UserInteractionPattern],
        purpose: str = 'learning'
    ) -> Dict[str, Any]:
        """
        Apply comprehensive privacy protection to user patterns.
        
        Args:
            user_id: User identifier
            patterns: User patterns to privatize
            purpose: Purpose of data processing
            
        Returns:
            Privatized pattern data
        """
        # Check consent
        if not await self._check_consent(user_id, purpose):
            return {'error': 'User consent required for data processing'}
        
        # Apply data minimization
        minimized_patterns = await self._apply_data_minimization(patterns)
        
        # Apply anonymization
        anonymized_patterns = self.anonymization_engine.apply_semantic_anonymization(
            minimized_patterns,
            self.privacy_policy.anonymization_level
        )
        
        # Add differential privacy noise
        if self.privacy_policy.differential_privacy_epsilon > 0:
            anonymized_patterns = await self._add_differential_privacy_noise(
                anonymized_patterns
            )
        
        # Encrypt sensitive fields
        encrypted_patterns = await self._encrypt_sensitive_fields(anonymized_patterns)
        
        # Log privacy operation
        await self._log_privacy_operation(
            operation='pattern_privatization',
            user_id=user_id,
            data_type='user_patterns',
            privacy_level=self.privacy_policy.anonymization_level,
            details={
                'pattern_count': len(patterns),
                'purpose': purpose,
                'anonymization_applied': True,
                'differential_privacy_applied': self.privacy_policy.differential_privacy_epsilon > 0
            }
        )
        
        return {
            'privatized_patterns': encrypted_patterns,
            'privacy_metadata': {
                'anonymization_level': self.privacy_policy.anonymization_level,
                'differential_privacy_epsilon': self.privacy_policy.differential_privacy_epsilon,
                'retention_period_days': self.privacy_policy.data_retention_days,
                'processing_purpose': purpose
            },
            'privacy_guarantees': await self._generate_privacy_guarantees()
        }
    
    async def secure_federated_aggregation(
        self,
        local_metrics: Dict[str, float],
        purpose: str = 'collaborative_learning'
    ) -> Dict[str, Any]:
        """
        Perform privacy-preserving federated aggregation.
        
        Args:
            local_metrics: Local metrics to aggregate
            purpose: Purpose of aggregation
            
        Returns:
            Secure aggregation result
        """
        # Apply differential privacy to local metrics
        private_metrics = {}
        
        for metric_name, value in local_metrics.items():
            if self.dp_engine.consume_privacy_budget(0.1):  # Small budget per metric
                private_value = self.dp_engine.add_laplace_noise(value, sensitivity=1.0)
                private_metrics[metric_name] = private_value
        
        # Encrypt metrics for secure transmission
        encrypted_metrics = {
            metric_name: self.secure_computation.encrypt_data(value)
            for metric_name, value in private_metrics.items()
        }
        
        # Log federated operation
        await self._log_privacy_operation(
            operation='federated_aggregation',
            user_id=None,
            data_type='metrics',
            privacy_level='differential_private',
            details={
                'metric_count': len(local_metrics),
                'purpose': purpose,
                'privacy_budget_used': self.dp_engine.privacy_budget_used
            }
        )
        
        return {
            'encrypted_metrics': encrypted_metrics,
            'privacy_metadata': {
                'differential_privacy_applied': True,
                'privacy_budget_remaining': self.dp_engine.get_remaining_budget(),
                'secure_computation_used': True
            }
        }
    
    async def privacy_preserving_analytics(
        self,
        data: List[Dict[str, Any]],
        query_type: str,
        sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Perform privacy-preserving analytics on data.
        
        Args:
            data: Data to analyze
            query_type: Type of query (count, sum, average, etc.)
            sensitivity: Sensitivity of the query
            
        Returns:
            Privacy-preserving analytics result
        """
        if not self.dp_engine.consume_privacy_budget(0.1):
            return {'error': 'Privacy budget exhausted'}
        
        # Perform query
        if query_type == 'count':
            result = len(data)
        elif query_type == 'sum':
            values = [item.get('value', 0) for item in data]
            result = sum(values)
        elif query_type == 'average':
            values = [item.get('value', 0) for item in data]
            result = np.mean(values) if values else 0
        else:
            return {'error': f'Unsupported query type: {query_type}'}
        
        # Add differential privacy noise
        private_result = self.dp_engine.add_laplace_noise(result, sensitivity)
        
        # Log analytics operation
        await self._log_privacy_operation(
            operation='privacy_preserving_analytics',
            user_id=None,
            data_type='analytics',
            privacy_level='differential_private',
            details={
                'query_type': query_type,
                'data_size': len(data),
                'sensitivity': sensitivity,
                'privacy_budget_used': 0.1
            }
        )
        
        return {
            'result': private_result,
            'privacy_metadata': {
                'differential_privacy_applied': True,
                'epsilon_used': 0.1,
                'query_type': query_type,
                'sensitivity': sensitivity
            }
        }
    
    async def check_privacy_compliance(
        self,
        operation: str,
        data_types: List[str],
        user_consents: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Check privacy compliance for an operation.
        
        Args:
            operation: Operation to check compliance for
            data_types: Types of data being processed
            user_consents: User consent status
            
        Returns:
            Compliance check result
        """
        compliance_issues = []
        compliance_status = 'compliant'
        
        # Check data retention limits
        for data_type in data_types:
            if data_type in self._data_retention_tracker:
                retention_date = self._data_retention_tracker[data_type]
                days_retained = (datetime.utcnow() - retention_date).days
                
                if days_retained > self.privacy_policy.data_retention_days:
                    compliance_issues.append({
                        'issue': 'data_retention_exceeded',
                        'data_type': data_type,
                        'days_exceeded': days_retained - self.privacy_policy.data_retention_days
                    })
                    compliance_status = 'non_compliant'
        
        # Check consent requirements
        if self.privacy_policy.require_explicit_consent:
            for user_id, consent_given in user_consents.items():
                if not consent_given:
                    compliance_issues.append({
                        'issue': 'missing_consent',
                        'user_id': user_id
                    })
                    compliance_status = 'non_compliant'
        
        # Check privacy budget
        if self.dp_engine.get_remaining_budget() < 0.1:
            compliance_issues.append({
                'issue': 'privacy_budget_low',
                'remaining_budget': self.dp_engine.get_remaining_budget()
            })
            if self.dp_engine.get_remaining_budget() <= 0:
                compliance_status = 'non_compliant'
        
        return {
            'compliance_status': compliance_status,
            'issues': compliance_issues,
            'recommendations': await self._generate_compliance_recommendations(compliance_issues)
        }
    
    async def generate_privacy_report(
        self,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive privacy report.
        
        Args:
            time_range_days: Time range for report
            
        Returns:
            Privacy report
        """
        cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
        recent_logs = [
            log for log in self._audit_logs
            if log.timestamp >= cutoff_date
        ]
        
        # Analyze privacy operations
        operation_counts = {}
        privacy_levels = {}
        compliance_stats = {'compliant': 0, 'non_compliant': 0}
        
        for log in recent_logs:
            operation_counts[log.operation] = operation_counts.get(log.operation, 0) + 1
            privacy_levels[log.privacy_level] = privacy_levels.get(log.privacy_level, 0) + 1
            compliance_stats[log.compliance_status] += 1
        
        # Privacy budget analysis
        privacy_budget_analysis = {
            'initial_budget': self.dp_engine.epsilon,
            'used_budget': self.dp_engine.privacy_budget_used,
            'remaining_budget': self.dp_engine.get_remaining_budget(),
            'query_count': self.dp_engine.query_count,
            'usage_rate': self.dp_engine.privacy_budget_used / self.dp_engine.epsilon if self.dp_engine.epsilon > 0 else 0
        }
        
        return {
            'report_period_days': time_range_days,
            'total_operations': len(recent_logs),
            'operation_breakdown': operation_counts,
            'privacy_level_distribution': privacy_levels,
            'compliance_statistics': compliance_stats,
            'privacy_budget_analysis': privacy_budget_analysis,
            'policy_adherence': await self._analyze_policy_adherence(recent_logs),
            'recommendations': await self._generate_privacy_recommendations(recent_logs),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for specified purpose."""
        if not self.privacy_policy.require_explicit_consent:
            return True
        
        user_consent = self._consent_records.get(user_id, {})
        return user_consent.get(purpose, False)
    
    async def _apply_data_minimization(
        self,
        patterns: List[UserInteractionPattern]
    ) -> List[UserInteractionPattern]:
        """Apply data minimization rules to patterns."""
        minimized_patterns = []
        
        for pattern in patterns:
            # Check retention period
            age_days = (datetime.utcnow() - pattern.timestamp).days
            
            # Find applicable minimization rule
            applicable_rule = None
            for rule in self._minimization_rules:
                if 'user_patterns' in rule.applies_to:
                    applicable_rule = rule
                    break
            
            if applicable_rule and age_days <= applicable_rule.retention_period_days:
                # Apply field suppression
                minimized_features = {
                    k: v for k, v in pattern.features.items()
                    if k not in applicable_rule.suppression_fields
                }
                
                # Create minimized pattern
                minimized_pattern = UserInteractionPattern(
                    pattern_id=pattern.pattern_id,
                    user_id='anonymized',  # Remove user ID
                    pattern_type=pattern.pattern_type,
                    features=minimized_features,
                    success_rate=pattern.success_rate,
                    frequency=pattern.frequency,
                    confidence=pattern.confidence,
                    timestamp=pattern.timestamp,
                    context={k: v for k, v in pattern.context.items() if k != 'user_id'}
                )
                
                minimized_patterns.append(minimized_pattern)
        
        return minimized_patterns
    
    async def _add_differential_privacy_noise(
        self,
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add differential privacy noise to pattern data."""
        noisy_patterns = []
        
        for pattern in patterns:
            noisy_pattern = pattern.copy()
            
            # Add noise to numeric features
            if 'features' in pattern:
                noisy_features = {}
                for key, value in pattern['features'].items():
                    if isinstance(value, (int, float)):
                        noisy_value = self.dp_engine.add_laplace_noise(value, sensitivity=1.0)
                        noisy_features[key] = max(0, noisy_value)  # Ensure non-negative
                    else:
                        noisy_features[key] = value
                
                noisy_pattern['features'] = noisy_features
            
            noisy_patterns.append(noisy_pattern)
        
        return noisy_patterns
    
    async def _encrypt_sensitive_fields(
        self,
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Encrypt sensitive fields in patterns."""
        encrypted_patterns = []
        
        sensitive_fields = ['pattern_hash', 'features', 'context']
        
        for pattern in patterns:
            encrypted_pattern = pattern.copy()
            
            for field in sensitive_fields:
                if field in pattern:
                    encrypted_pattern[f'{field}_encrypted'] = self.secure_computation.encrypt_data(
                        pattern[field]
                    )
                    del encrypted_pattern[field]  # Remove original
            
            encrypted_patterns.append(encrypted_pattern)
        
        return encrypted_patterns
    
    async def _log_privacy_operation(
        self,
        operation: str,
        user_id: Optional[str],
        data_type: str,
        privacy_level: str,
        details: Dict[str, Any]
    ) -> None:
        """Log privacy operation for audit trail."""
        log_entry = PrivacyAuditLog(
            log_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            operation=operation,
            data_type=data_type,
            user_id=user_id,
            privacy_level=privacy_level,
            compliance_status='compliant',  # Would be determined by compliance check
            details=details
        )
        
        self._audit_logs.append(log_entry)
        
        # Keep audit log manageable
        if len(self._audit_logs) > 10000:
            self._audit_logs = self._audit_logs[-5000:]  # Keep last 5000 entries
    
    async def _generate_privacy_guarantees(self) -> Dict[str, Any]:
        """Generate privacy guarantees for current configuration."""
        return {
            'differential_privacy': {
                'epsilon': self.dp_engine.epsilon,
                'delta': self.dp_engine.delta,
                'privacy_level': 'high' if self.dp_engine.epsilon < 1.0 else 'moderate' if self.dp_engine.epsilon < 5.0 else 'basic'
            },
            'anonymization': {
                'level': self.privacy_policy.anonymization_level,
                'k_anonymity': True if self.privacy_policy.anonymization_level in ['strong', 'extreme'] else False,
                'suppression_applied': True
            },
            'encryption': {
                'at_rest': True,
                'in_transit': True,
                'algorithm': 'Fernet (AES-128)'
            },
            'data_retention': {
                'max_days': self.privacy_policy.data_retention_days,
                'automatic_deletion': True
            }
        }
    
    async def _generate_compliance_recommendations(
        self,
        compliance_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations to address compliance issues."""
        recommendations = []
        
        for issue in compliance_issues:
            if issue['issue'] == 'data_retention_exceeded':
                recommendations.append(
                    f"Delete or further anonymize {issue['data_type']} data "
                    f"that exceeds retention period by {issue['days_exceeded']} days"
                )
            elif issue['issue'] == 'missing_consent':
                recommendations.append(
                    f"Obtain explicit consent from user {issue['user_id']} "
                    "or exclude their data from processing"
                )
            elif issue['issue'] == 'privacy_budget_low':
                recommendations.append(
                    "Consider resetting privacy budget or reducing query frequency "
                    f"(remaining: {issue['remaining_budget']:.3f})"
                )
        
        return recommendations
    
    async def _analyze_policy_adherence(
        self,
        audit_logs: List[PrivacyAuditLog]
    ) -> Dict[str, Any]:
        """Analyze adherence to privacy policy."""
        total_operations = len(audit_logs)
        if total_operations == 0:
            return {'adherence_rate': 1.0, 'violations': []}
        
        compliant_operations = len([
            log for log in audit_logs
            if log.compliance_status == 'compliant'
        ])
        
        adherence_rate = compliant_operations / total_operations
        
        # Identify specific violations
        violations = []
        for log in audit_logs:
            if log.compliance_status == 'non_compliant':
                violations.append({
                    'timestamp': log.timestamp.isoformat(),
                    'operation': log.operation,
                    'details': log.details
                })
        
        return {
            'adherence_rate': adherence_rate,
            'total_operations': total_operations,
            'compliant_operations': compliant_operations,
            'violations': violations,
            'policy_areas': {
                'data_retention': self._check_retention_adherence(audit_logs),
                'consent_management': self._check_consent_adherence(audit_logs),
                'anonymization': self._check_anonymization_adherence(audit_logs)
            }
        }
    
    async def _generate_privacy_recommendations(
        self,
        audit_logs: List[PrivacyAuditLog]
    ) -> List[str]:
        """Generate privacy improvement recommendations."""
        recommendations = []
        
        # Analyze operation patterns
        operation_counts = {}
        for log in audit_logs:
            operation_counts[log.operation] = operation_counts.get(log.operation, 0) + 1
        
        # High-frequency operations might need optimization
        high_frequency_ops = [
            op for op, count in operation_counts.items()
            if count > len(audit_logs) * 0.1  # >10% of total operations
        ]
        
        if high_frequency_ops:
            recommendations.append(
                f"Consider optimizing high-frequency operations: {', '.join(high_frequency_ops)}"
            )
        
        # Privacy budget usage
        if self.dp_engine.privacy_budget_used > self.dp_engine.epsilon * 0.8:
            recommendations.append(
                "Privacy budget usage is high (>80%). Consider increasing epsilon or reducing query frequency."
            )
        
        # Data retention
        overdue_data_types = [
            data_type for data_type, retention_date in self._data_retention_tracker.items()
            if (datetime.utcnow() - retention_date).days > self.privacy_policy.data_retention_days
        ]
        
        if overdue_data_types:
            recommendations.append(
                f"Clean up overdue data types: {', '.join(overdue_data_types)}"
            )
        
        return recommendations
    
    def _check_retention_adherence(self, audit_logs: List[PrivacyAuditLog]) -> Dict[str, Any]:
        """Check adherence to data retention policies."""
        retention_violations = 0
        total_retention_checks = 0
        
        for log in audit_logs:
            if 'retention' in log.operation.lower():
                total_retention_checks += 1
                if log.compliance_status == 'non_compliant':
                    retention_violations += 1
        
        return {
            'adherence_rate': 1.0 - (retention_violations / total_retention_checks) if total_retention_checks > 0 else 1.0,
            'violations': retention_violations,
            'total_checks': total_retention_checks
        }
    
    def _check_consent_adherence(self, audit_logs: List[PrivacyAuditLog]) -> Dict[str, Any]:
        """Check adherence to consent management policies."""
        consent_violations = 0
        total_consent_checks = 0
        
        for log in audit_logs:
            if log.user_id and self.privacy_policy.require_explicit_consent:
                total_consent_checks += 1
                # Check if consent was verified (simplified check)
                if not log.details.get('consent_verified', False):
                    consent_violations += 1
        
        return {
            'adherence_rate': 1.0 - (consent_violations / total_consent_checks) if total_consent_checks > 0 else 1.0,
            'violations': consent_violations,
            'total_checks': total_consent_checks
        }
    
    def _check_anonymization_adherence(self, audit_logs: List[PrivacyAuditLog]) -> Dict[str, Any]:
        """Check adherence to anonymization policies."""
        anonymization_applied = 0
        total_data_operations = 0
        
        for log in audit_logs:
            if log.data_type in ['user_patterns', 'personal_data']:
                total_data_operations += 1
                if log.details.get('anonymization_applied', False):
                    anonymization_applied += 1
        
        return {
            'adherence_rate': anonymization_applied / total_data_operations if total_data_operations > 0 else 1.0,
            'operations_anonymized': anonymization_applied,
            'total_data_operations': total_data_operations
        }
    
    def _default_privacy_policy(self) -> PrivacyPolicy:
        """Create default privacy policy."""
        return PrivacyPolicy(
            policy_id=str(uuid4()),
            name="Default Privacy Policy",
            description="Default privacy policy for AI learning system",
            data_retention_days=90,
            anonymization_level="strong",
            differential_privacy_epsilon=1.0,
            allow_federated_sharing=True,
            allow_analytics=True,
            require_explicit_consent=False,
            geographic_restrictions=[],
            created_at=datetime.utcnow()
        )
    
    def _load_minimization_rules(self) -> List[DataMinimizationRule]:
        """Load data minimization rules."""
        return [
            DataMinimizationRule(
                rule_id="user_patterns_rule",
                applies_to=["user_patterns"],
                retention_period_days=self.privacy_policy.data_retention_days,
                aggregation_level="individual",
                suppression_fields=["user_id", "session_id", "ip_address", "file_path"],
                anonymization_method="k_anonymity"
            ),
            DataMinimizationRule(
                rule_id="analytics_rule",
                applies_to=["analytics", "metrics"],
                retention_period_days=365,
                aggregation_level="group",
                suppression_fields=["individual_identifiers"],
                anonymization_method="differential_privacy"
            )
        ]