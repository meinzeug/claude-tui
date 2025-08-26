#!/usr/bin/env python3
"""
Performance Regression Testing Suite

Automated performance regression testing with baseline comparison,
CI/CD integration, and detailed reporting.
"""

import asyncio
import argparse
import json
import logging
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

from performance.benchmarking.comprehensive_benchmarker import ComprehensivePerformanceBenchmarker
from performance.monitoring.realtime_monitor import RealtimePerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceRegression:
    """Performance regression analysis and testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.benchmarker = ComprehensivePerformanceBenchmarker()
        self.baseline_data = None
        self.regression_thresholds = self.config.get('regression_thresholds', {})
        
    def _default_config(self) -> Dict[str, Any]:
        """Default regression testing configuration"""
        return {
            'baseline_file': 'performance_baseline.json',
            'regression_thresholds': {
                'throughput_degradation_percent': 10.0,
                'latency_increase_percent': 20.0,
                'error_rate_increase_percent': 50.0,
                'memory_increase_percent': 15.0,
                'cpu_increase_percent': 25.0
            },
            'test_scenarios': [
                'light_load',
                'medium_load', 
                'heavy_load'
            ],
            'required_samples': 3,
            'confidence_level': 0.95,
            'warmup_iterations': 2,
            'max_regression_severity': 'warning',
            'auto_baseline_update': False
        }
    
    async def run_regression_tests(self, baseline_file: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive performance regression tests"""
        logger.info("Starting performance regression testing")
        
        # Load baseline
        baseline_file = baseline_file or self.config.get('baseline_file')
        if not await self._load_baseline(baseline_file):
            logger.error("Cannot run regression tests without baseline")
            return {'status': 'failed', 'error': 'No baseline available'}
        
        # Run current performance tests
        logger.info("Running current performance benchmarks")
        current_results = await self._run_performance_tests()
        
        if not current_results:
            return {'status': 'failed', 'error': 'Performance tests failed'}
        
        # Compare against baseline
        logger.info("Analyzing performance regressions")
        regression_analysis = await self._analyze_regressions(current_results)
        
        # Generate regression report
        report = await self._generate_regression_report(regression_analysis, current_results)
        
        return report
    
    async def _load_baseline(self, baseline_file: str) -> bool:
        """Load performance baseline data"""
        baseline_path = Path(baseline_file)
        
        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {baseline_file}")
            
            # Try to find most recent baseline
            baseline_dir = baseline_path.parent or Path('.')
            baseline_pattern = f"{baseline_path.stem}_*.json"
            
            baseline_files = list(baseline_dir.glob(baseline_pattern))
            if baseline_files:
                # Use most recent
                latest_baseline = max(baseline_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using latest baseline: {latest_baseline}")
                baseline_path = latest_baseline
            else:
                return False
        
        try:
            with open(baseline_path, 'r') as f:
                self.baseline_data = json.load(f)
            
            logger.info(f"Loaded baseline from {baseline_path}")
            
            # Validate baseline structure
            if not self._validate_baseline_structure(self.baseline_data):
                logger.error("Invalid baseline structure")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            return False
    
    def _validate_baseline_structure(self, baseline: Dict[str, Any]) -> bool:
        """Validate baseline data structure"""
        required_keys = ['benchmark_results', 'timestamp', 'environment']
        
        for key in required_keys:
            if key not in baseline:
                logger.error(f"Missing required baseline key: {key}")
                return False
        
        # Check for benchmark results structure
        benchmark_results = baseline['benchmark_results']
        if 'load_tests' not in benchmark_results:
            logger.error("Baseline missing load test results")
            return False
        
        return True
    
    async def _run_performance_tests(self) -> Optional[Dict[str, Any]]:
        """Run current performance tests"""
        try:
            # Run multiple samples for statistical significance
            samples = []
            required_samples = self.config.get('required_samples', 3)
            warmup_iterations = self.config.get('warmup_iterations', 2)
            
            logger.info(f"Running {warmup_iterations} warmup iterations")
            for i in range(warmup_iterations):
                logger.info(f"Warmup iteration {i+1}/{warmup_iterations}")
                await self.benchmarker.run_comprehensive_benchmark()
                await asyncio.sleep(30)  # Cool-down between tests
            
            logger.info(f"Running {required_samples} test samples")
            for i in range(required_samples):
                logger.info(f"Test sample {i+1}/{required_samples}")
                
                sample_result = await self.benchmarker.run_comprehensive_benchmark()
                if sample_result:
                    samples.append(sample_result)
                
                # Cool-down between samples
                if i < required_samples - 1:
                    await asyncio.sleep(60)
            
            if not samples:
                logger.error("No successful test samples")
                return None
            
            # Aggregate samples
            aggregated_results = self._aggregate_test_samples(samples)
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Performance test execution failed: {e}")
            return None
    
    def _aggregate_test_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test samples for statistical significance"""
        if not samples:
            return {}
        
        if len(samples) == 1:
            return samples[0]
        
        # Aggregate load test results
        aggregated = {
            'benchmark_id': f"regression_test_{int(time.time())}",
            'samples_count': len(samples),
            'benchmark_results': {}
        }
        
        # Aggregate load test results
        load_test_aggregation = {}
        
        # Get all test scenarios from samples
        all_scenarios = set()
        for sample in samples:
            if 'load_tests' in sample.get('benchmark_results', {}):
                all_scenarios.update(sample['benchmark_results']['load_tests'].keys())
        
        for scenario in all_scenarios:
            scenario_samples = []
            for sample in samples:
                load_tests = sample.get('benchmark_results', {}).get('load_tests', {})
                if scenario in load_tests:
                    scenario_samples.append(load_tests[scenario])
            
            if scenario_samples:
                # Aggregate metrics
                aggregated_scenario = self._aggregate_scenario_metrics(scenario_samples)
                load_test_aggregation[scenario] = aggregated_scenario
        
        aggregated['benchmark_results']['load_tests'] = load_test_aggregation
        
        # Aggregate quantum performance
        quantum_aggregation = {}
        all_modules = set()
        
        for sample in samples:
            quantum_results = sample.get('benchmark_results', {}).get('quantum_performance', {})
            all_modules.update(quantum_results.keys())
        
        for module in all_modules:
            module_samples = []
            for sample in samples:
                quantum_results = sample.get('benchmark_results', {}).get('quantum_performance', {})
                if module in quantum_results:
                    module_samples.append(quantum_results[module])
            
            if module_samples:
                aggregated_module = self._aggregate_quantum_metrics(module_samples)
                quantum_aggregation[module] = aggregated_module
        
        aggregated['benchmark_results']['quantum_performance'] = quantum_aggregation
        
        # Copy metadata from first sample
        first_sample = samples[0]
        for key in ['start_time', 'end_time', 'total_duration']:
            if key in first_sample:
                aggregated[key] = first_sample[key]
        
        return aggregated
    
    def _aggregate_scenario_metrics(self, scenario_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple scenario samples"""
        if not scenario_samples:
            return {}
        
        if len(scenario_samples) == 1:
            return scenario_samples[0]
        
        aggregated = {}
        
        # Extract numerical metrics
        numeric_metrics = [
            'throughput',
            'error_rate',
            'total_requests',
            'successful_requests'
        ]
        
        for metric in numeric_metrics:
            values = [s.get(metric, 0) for s in scenario_samples if metric in s]
            if values:
                aggregated[metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'samples': values
                }
        
        # Aggregate latency metrics
        if all('latency' in s for s in scenario_samples):
            latency_metrics = ['p50', 'p90', 'p95', 'p99', 'avg', 'max', 'min']
            latency_aggregation = {}
            
            for metric in latency_metrics:
                values = [s['latency'].get(metric, 0) for s in scenario_samples if metric in s.get('latency', {})]
                if values:
                    latency_aggregation[metric] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'samples': values
                    }
            
            aggregated['latency'] = latency_aggregation
        
        # Aggregate resource usage
        if all('resource_usage' in s for s in scenario_samples):
            resource_aggregation = {}
            
            for sample in scenario_samples:
                for resource_type, metrics in sample['resource_usage'].items():
                    if resource_type not in resource_aggregation:
                        resource_aggregation[resource_type] = defaultdict(list)
                    
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            resource_aggregation[resource_type][metric_name].append(value)
            
            # Calculate statistics for each resource metric
            final_resource_aggregation = {}
            for resource_type, metrics in resource_aggregation.items():
                final_resource_aggregation[resource_type] = {}
                for metric_name, values in metrics.items():
                    if values:
                        final_resource_aggregation[resource_type][metric_name] = {
                            'mean': statistics.mean(values),
                            'median': statistics.median(values),
                            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
                        }
            
            aggregated['resource_usage'] = final_resource_aggregation
        
        return aggregated
    
    def _aggregate_quantum_metrics(self, module_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quantum module performance metrics"""
        if not module_samples:
            return {}
        
        if len(module_samples) == 1:
            return module_samples[0]
        
        aggregated = {}
        
        # Aggregate scalability scores
        scalability_scores = [s.get('scalability_score', 0) for s in module_samples if 'scalability_score' in s]
        if scalability_scores:
            aggregated['scalability_score'] = {
                'mean': statistics.mean(scalability_scores),
                'median': statistics.median(scalability_scores),
                'stdev': statistics.stdev(scalability_scores) if len(scalability_scores) > 1 else 0.0,
                'samples': scalability_scores
            }
        
        # Copy other metrics from first sample
        first_sample = module_samples[0]
        for key in ['module', 'total_duration']:
            if key in first_sample:
                aggregated[key] = first_sample[key]
        
        return aggregated
    
    async def _analyze_regressions(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance regressions compared to baseline"""
        regressions = {
            'total_regressions': 0,
            'critical_regressions': 0,
            'warning_regressions': 0,
            'improvements': 0,
            'detailed_analysis': {},
            'summary': {},
            'status': 'pass'
        }
        
        baseline_results = self.baseline_data['benchmark_results']
        current_benchmark_results = current_results['benchmark_results']
        
        # Analyze load test regressions
        if 'load_tests' in both (baseline_results, current_benchmark_results):
            load_test_analysis = self._analyze_load_test_regressions(
                baseline_results['load_tests'],
                current_benchmark_results['load_tests']
            )
            regressions['detailed_analysis']['load_tests'] = load_test_analysis
            regressions['total_regressions'] += load_test_analysis['regression_count']
            regressions['critical_regressions'] += load_test_analysis['critical_count']
            regressions['warning_regressions'] += load_test_analysis['warning_count']
            regressions['improvements'] += load_test_analysis['improvement_count']
        
        # Analyze quantum performance regressions
        if 'quantum_performance' in both (baseline_results, current_benchmark_results):
            quantum_analysis = self._analyze_quantum_regressions(
                baseline_results['quantum_performance'],
                current_benchmark_results['quantum_performance']
            )
            regressions['detailed_analysis']['quantum_performance'] = quantum_analysis
            regressions['total_regressions'] += quantum_analysis['regression_count']
            regressions['critical_regressions'] += quantum_analysis['critical_count']
            regressions['warning_regressions'] += quantum_analysis['warning_count']
            regressions['improvements'] += quantum_analysis['improvement_count']
        
        # Determine overall status
        if regressions['critical_regressions'] > 0:
            regressions['status'] = 'critical'
        elif regressions['warning_regressions'] > 0:
            regressions['status'] = 'warning'
        elif regressions['total_regressions'] > 0:
            regressions['status'] = 'minor'
        else:
            regressions['status'] = 'pass'
        
        # Generate summary
        regressions['summary'] = {
            'baseline_timestamp': self.baseline_data.get('timestamp'),
            'current_timestamp': datetime.utcnow().isoformat(),
            'total_comparisons': len(regressions['detailed_analysis']),
            'regression_rate': regressions['total_regressions'] / max(1, len(regressions['detailed_analysis'])),
            'overall_status': regressions['status']
        }
        
        return regressions
    
    def _analyze_load_test_regressions(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze load test performance regressions"""
        analysis = {
            'regression_count': 0,
            'critical_count': 0,
            'warning_count': 0,
            'improvement_count': 0,
            'scenarios': {}
        }
        
        common_scenarios = set(baseline.keys()) & set(current.keys())
        
        for scenario in common_scenarios:
            baseline_scenario = baseline[scenario]
            current_scenario = current[scenario]
            
            scenario_analysis = self._compare_scenario_metrics(baseline_scenario, current_scenario)
            analysis['scenarios'][scenario] = scenario_analysis
            
            # Count regressions and improvements
            for metric_analysis in scenario_analysis['metrics'].values():
                if metric_analysis['regression']:
                    analysis['regression_count'] += 1
                    if metric_analysis['severity'] == 'critical':
                        analysis['critical_count'] += 1
                    else:
                        analysis['warning_count'] += 1
                elif metric_analysis['improvement']:
                    analysis['improvement_count'] += 1
        
        return analysis
    
    def _compare_scenario_metrics(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metrics between baseline and current scenario"""
        comparison = {
            'scenario_status': 'pass',
            'metrics': {}
        }
        
        # Compare throughput
        if 'throughput' in baseline and 'throughput' in current:
            throughput_comparison = self._compare_metric(
                'throughput',
                self._extract_metric_value(baseline, 'throughput'),
                self._extract_metric_value(current, 'throughput'),
                'higher_better',
                self.regression_thresholds.get('throughput_degradation_percent', 10.0)
            )
            comparison['metrics']['throughput'] = throughput_comparison
        
        # Compare error rate
        if 'error_rate' in baseline and 'error_rate' in current:
            error_rate_comparison = self._compare_metric(
                'error_rate',
                self._extract_metric_value(baseline, 'error_rate'),
                self._extract_metric_value(current, 'error_rate'),
                'lower_better',
                self.regression_thresholds.get('error_rate_increase_percent', 50.0)
            )
            comparison['metrics']['error_rate'] = error_rate_comparison
        
        # Compare latency metrics
        if 'latency' in baseline and 'latency' in current:
            latency_metrics = ['p50', 'p95', 'p99', 'avg']
            
            for lat_metric in latency_metrics:
                if lat_metric in baseline['latency'] and lat_metric in current['latency']:
                    latency_comparison = self._compare_metric(
                        f'latency_{lat_metric}',
                        self._extract_metric_value(baseline['latency'], lat_metric),
                        self._extract_metric_value(current['latency'], lat_metric),
                        'lower_better',
                        self.regression_thresholds.get('latency_increase_percent', 20.0)
                    )
                    comparison['metrics'][f'latency_{lat_metric}'] = latency_comparison
        
        # Determine scenario status
        critical_regressions = sum(1 for m in comparison['metrics'].values() 
                                 if m['regression'] and m['severity'] == 'critical')
        warning_regressions = sum(1 for m in comparison['metrics'].values() 
                                if m['regression'] and m['severity'] == 'warning')
        
        if critical_regressions > 0:
            comparison['scenario_status'] = 'critical'
        elif warning_regressions > 0:
            comparison['scenario_status'] = 'warning'
        
        return comparison
    
    def _compare_metric(self, metric_name: str, baseline_value: float, current_value: float,
                       direction: str, threshold_percent: float) -> Dict[str, Any]:
        """Compare individual metric values"""
        if baseline_value == 0:
            return {
                'metric_name': metric_name,
                'baseline_value': baseline_value,
                'current_value': current_value,
                'change_percent': 0.0,
                'regression': False,
                'improvement': current_value > 0,
                'severity': 'none'
            }
        
        change_percent = ((current_value - baseline_value) / baseline_value) * 100
        
        # Determine if this is a regression or improvement
        if direction == 'higher_better':
            # For metrics where higher is better (e.g., throughput)
            regression = change_percent < -threshold_percent
            improvement = change_percent > threshold_percent
        else:  # lower_better
            # For metrics where lower is better (e.g., latency, error rate)
            regression = change_percent > threshold_percent
            improvement = change_percent < -threshold_percent
        
        # Determine severity
        severity = 'none'
        if regression:
            if abs(change_percent) > threshold_percent * 2:
                severity = 'critical'
            else:
                severity = 'warning'
        
        return {
            'metric_name': metric_name,
            'baseline_value': baseline_value,
            'current_value': current_value,
            'change_percent': change_percent,
            'regression': regression,
            'improvement': improvement,
            'severity': severity,
            'threshold_percent': threshold_percent
        }
    
    def _extract_metric_value(self, data: Dict[str, Any], metric_key: str) -> float:
        """Extract metric value, handling aggregated results"""
        value = data.get(metric_key)
        
        if isinstance(value, dict):
            # Aggregated result, use mean
            return value.get('mean', 0.0)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0
    
    def _analyze_quantum_regressions(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quantum performance regressions"""
        analysis = {
            'regression_count': 0,
            'critical_count': 0,
            'warning_count': 0,
            'improvement_count': 0,
            'modules': {}
        }
        
        common_modules = set(baseline.keys()) & set(current.keys())
        
        for module in common_modules:
            baseline_module = baseline[module]
            current_module = current[module]
            
            if 'scalability_score' in baseline_module and 'scalability_score' in current_module:
                scalability_comparison = self._compare_metric(
                    'scalability_score',
                    self._extract_metric_value(baseline_module, 'scalability_score'),
                    self._extract_metric_value(current_module, 'scalability_score'),
                    'higher_better',
                    15.0  # 15% degradation threshold for quantum metrics
                )
                
                analysis['modules'][module] = {
                    'scalability_score': scalability_comparison
                }
                
                if scalability_comparison['regression']:
                    analysis['regression_count'] += 1
                    if scalability_comparison['severity'] == 'critical':
                        analysis['critical_count'] += 1
                    else:
                        analysis['warning_count'] += 1
                elif scalability_comparison['improvement']:
                    analysis['improvement_count'] += 1
        
        return analysis
    
    async def _generate_regression_report(self, regression_analysis: Dict[str, Any], 
                                        current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive regression test report"""
        report = {
            'regression_test_id': f"regression_{int(time.time())}",
            'timestamp': datetime.utcnow().isoformat(),
            'status': regression_analysis['status'],
            'baseline_info': {
                'timestamp': self.baseline_data.get('timestamp'),
                'environment': self.baseline_data.get('environment', {})
            },
            'current_info': {
                'timestamp': current_results.get('start_time'),
                'samples_count': current_results.get('samples_count', 1)
            },
            'regression_summary': regression_analysis['summary'],
            'regression_details': regression_analysis['detailed_analysis'],
            'recommendations': self._generate_regression_recommendations(regression_analysis),
            'ci_cd_status': self._determine_ci_cd_status(regression_analysis)
        }
        
        # Save report to file
        report_file = f"regression_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Regression report saved to {report_file}")
        
        return report
    
    def _generate_regression_recommendations(self, regression_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on regression analysis"""
        recommendations = []
        
        if regression_analysis['critical_regressions'] > 0:
            recommendations.append("CRITICAL: Performance regressions detected requiring immediate attention")
            recommendations.append("Block deployment until critical regressions are resolved")
            recommendations.append("Investigate recent code changes that may impact performance")
        
        if regression_analysis['warning_regressions'] > 0:
            recommendations.append("WARNING: Performance degradations detected")
            recommendations.append("Review and optimize affected components before next release")
            recommendations.append("Consider performance profiling of degraded scenarios")
        
        if regression_analysis['total_regressions'] == 0:
            recommendations.append("All performance metrics within acceptable ranges")
            if regression_analysis['improvements'] > 0:
                recommendations.append("Performance improvements detected - great work!")
        
        # Specific recommendations based on regression types
        load_test_analysis = regression_analysis.get('detailed_analysis', {}).get('load_tests', {})
        if load_test_analysis.get('regression_count', 0) > 0:
            recommendations.append("Load test regressions detected:")
            recommendations.append("- Review database query performance")
            recommendations.append("- Check for resource leaks or memory issues")
            recommendations.append("- Validate configuration changes")
        
        quantum_analysis = regression_analysis.get('detailed_analysis', {}).get('quantum_performance', {})
        if quantum_analysis.get('regression_count', 0) > 0:
            recommendations.append("Quantum intelligence performance regressions detected:")
            recommendations.append("- Review quantum algorithm optimizations")
            recommendations.append("- Check for concurrency issues in quantum modules")
            recommendations.append("- Validate quantum parameter tuning")
        
        return recommendations
    
    def _determine_ci_cd_status(self, regression_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine CI/CD pipeline status based on regression analysis"""
        status = regression_analysis['status']
        
        ci_cd_status = {
            'should_pass': True,
            'exit_code': 0,
            'message': 'Performance regression tests passed'
        }
        
        max_severity = self.config.get('max_regression_severity', 'warning')
        
        if status == 'critical':
            ci_cd_status['should_pass'] = False
            ci_cd_status['exit_code'] = 1
            ci_cd_status['message'] = 'FAIL: Critical performance regressions detected'
        elif status == 'warning' and max_severity == 'none':
            ci_cd_status['should_pass'] = False
            ci_cd_status['exit_code'] = 1
            ci_cd_status['message'] = 'FAIL: Performance warnings detected (strict mode)'
        elif status == 'minor':
            ci_cd_status['message'] = 'PASS: Minor performance changes detected'
        
        return ci_cd_status
    
    async def create_baseline(self, output_file: Optional[str] = None) -> str:
        """Create new performance baseline"""
        logger.info("Creating new performance baseline")
        
        # Run comprehensive benchmark
        baseline_results = await self.benchmarker.run_comprehensive_benchmark()
        
        if not baseline_results:
            raise RuntimeError("Failed to generate baseline results")
        
        # Create baseline data structure
        baseline_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'benchmark_id': baseline_results.get('benchmark_id'),
            'benchmark_results': baseline_results,
            'environment': {
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': os.cpu_count(),
                'git_commit': await self._get_git_commit(),
                'git_branch': await self._get_git_branch()
            },
            'config': self.config
        }
        
        # Save baseline
        output_file = output_file or f"performance_baseline_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)
        
        logger.info(f"Performance baseline created: {output_file}")
        return output_file
    
    async def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except Exception:
            return 'unknown'
    
    async def _get_git_branch(self) -> str:
        """Get current git branch"""
        try:
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except Exception:
            return 'unknown'

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Performance Regression Testing')
    parser.add_argument('command', choices=['test', 'baseline'], 
                       help='Command to execute')
    parser.add_argument('--baseline', '-b', help='Baseline file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--config', '-c', help='Configuration file')
    parser.add_argument('--strict', action='store_true', 
                       help='Strict mode (fail on warnings)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if args.strict:
        config['max_regression_severity'] = 'none'
    
    # Initialize regression tester
    regression_tester = PerformanceRegression(config)
    
    try:
        if args.command == 'baseline':
            # Create new baseline
            baseline_file = await regression_tester.create_baseline(args.output)
            print(f"Baseline created: {baseline_file}")
            return 0
            
        elif args.command == 'test':
            # Run regression tests
            report = await regression_tester.run_regression_tests(args.baseline)
            
            # Print summary
            print(f"\nRegression Test Results:")
            print(f"Status: {report['status'].upper()}")
            print(f"Critical Regressions: {report['regression_details']['critical_regressions']}")
            print(f"Warning Regressions: {report['regression_details']['warning_regressions']}")
            print(f"Improvements: {report['regression_details']['improvements']}")
            
            if report['recommendations']:
                print(f"\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  - {rec}")
            
            # Return appropriate exit code for CI/CD
            ci_cd_status = report.get('ci_cd_status', {})
            return ci_cd_status.get('exit_code', 0)
    
    except KeyboardInterrupt:
        logger.info("Regression testing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Regression testing failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)