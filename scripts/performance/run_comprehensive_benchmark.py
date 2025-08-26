#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Runner

Production-ready script to execute comprehensive performance benchmarking
with automated reporting and alerting.
"""

import asyncio
import argparse
import logging
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

from performance.benchmarking.comprehensive_benchmarker import ComprehensivePerformanceBenchmarker
from performance.monitoring.realtime_monitor import RealtimePerformanceMonitor
from performance.optimization.adaptive_optimizer import AdaptivePerformanceOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'benchmark_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Main benchmark execution coordinator"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.benchmarker = ComprehensivePerformanceBenchmarker()
        self.monitor = None
        self.optimizer = None
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load benchmark configuration"""
        default_config = {
            'benchmark_type': 'comprehensive',
            'enable_monitoring': True,
            'enable_optimization': False,
            'monitoring_duration': 300,  # 5 minutes
            'optimization_experiments': 20,
            'output_format': 'json',
            'alert_thresholds': {
                'error_rate_max': 0.05,
                'latency_p95_max': 2.0,
                'cpu_usage_max': 85.0,
                'memory_usage_max': 90.0
            },
            'load_test_scenarios': [
                {
                    'name': 'Light Load',
                    'duration': 120,
                    'concurrent_users': 10,
                    'ramp_up_time': 30
                },
                {
                    'name': 'Medium Load',
                    'duration': 180,
                    'concurrent_users': 50,
                    'ramp_up_time': 60
                },
                {
                    'name': 'Heavy Load',
                    'duration': 300,
                    'concurrent_users': 100,
                    'ramp_up_time': 120
                }
            ]
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    async def run_benchmark_suite(self) -> Dict[str, Any]:
        """Execute complete benchmark suite"""
        logger.info("Starting comprehensive performance benchmark suite")
        
        start_time = datetime.now()
        results = {
            'benchmark_run_id': f"benchmark_{int(start_time.timestamp())}",
            'start_time': start_time.isoformat(),
            'config': self.config.copy()
        }
        
        try:
            # 1. Pre-benchmark system check
            logger.info("Running pre-benchmark system checks")
            system_check = await self._run_system_check()
            results['system_check'] = system_check
            
            if not system_check['ready']:
                logger.error("System not ready for benchmarking")
                return results
            
            # 2. Start real-time monitoring if enabled
            if self.config.get('enable_monitoring', True):
                logger.info("Starting real-time performance monitoring")
                self.monitor = RealtimePerformanceMonitor()
                monitoring_task = asyncio.create_task(self.monitor.start_monitoring())
            else:
                monitoring_task = None
            
            # 3. Run core performance benchmarks
            logger.info("Executing core performance benchmarks")
            benchmark_results = await self.benchmarker.run_comprehensive_benchmark()
            results['benchmark_results'] = benchmark_results
            
            # 4. Analyze results and generate alerts
            logger.info("Analyzing results and checking thresholds")
            alerts = self._analyze_results_for_alerts(benchmark_results)
            results['alerts'] = alerts
            
            # 5. Run adaptive optimization if enabled
            if self.config.get('enable_optimization', False):
                logger.info("Running adaptive performance optimization")
                optimization_results = await self._run_optimization()
                results['optimization'] = optimization_results
            
            # 6. Stop monitoring and collect final metrics
            if monitoring_task:
                logger.info("Stopping monitoring and collecting final metrics")
                await self.monitor.stop_monitoring()
                monitoring_task.cancel()
                
                final_dashboard = self.monitor.dashboard.get_dashboard_data()
                results['monitoring_data'] = final_dashboard
            
            # 7. Generate comprehensive report
            logger.info("Generating comprehensive performance report")
            report_path = await self._generate_final_report(results)
            results['report_path'] = str(report_path)
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}", exc_info=True)
            results['error'] = str(e)
            results['status'] = 'failed'
        else:
            results['status'] = 'completed'
        finally:
            results['end_time'] = datetime.now().isoformat()
            results['total_duration'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    async def _run_system_check(self) -> Dict[str, Any]:
        """Run pre-benchmark system readiness check"""
        import psutil
        import aiohttp
        
        check_results = {
            'ready': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        check_results['checks']['cpu_available'] = cpu_percent < 80
        check_results['checks']['memory_available'] = memory.percent < 85
        check_results['checks']['disk_space'] = disk.percent < 90
        
        if cpu_percent >= 80:
            check_results['warnings'].append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory.percent >= 85:
            check_results['warnings'].append(f"High memory usage: {memory.percent:.1f}%")
        
        if disk.percent >= 90:
            check_results['errors'].append(f"Low disk space: {disk.percent:.1f}% used")
            check_results['ready'] = False
        
        # Check network connectivity to test endpoints
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                test_urls = [
                    'http://localhost:8000/api/v1/health',
                    'http://localhost:8000/health'
                ]
                
                connectivity_results = {}
                for url in test_urls:
                    try:
                        async with session.get(url) as response:
                            connectivity_results[url] = {
                                'status': response.status,
                                'reachable': True
                            }
                    except Exception as e:
                        connectivity_results[url] = {
                            'error': str(e),
                            'reachable': False
                        }
                
                check_results['checks']['network_connectivity'] = connectivity_results
                
        except Exception as e:
            check_results['warnings'].append(f"Network connectivity check failed: {e}")
        
        # Check required services/ports
        required_ports = [8000, 5432, 6379]  # API, PostgreSQL, Redis
        port_checks = {}
        
        import socket
        for port in required_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            port_checks[port] = result == 0
            sock.close()
            
            if result != 0:
                check_results['warnings'].append(f"Port {port} not accessible")
        
        check_results['checks']['ports'] = port_checks
        
        return check_results
    
    def _analyze_results_for_alerts(self, benchmark_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze benchmark results against alert thresholds"""
        alerts = []
        thresholds = self.config.get('alert_thresholds', {})
        
        # Check load test results
        if 'load_tests' in benchmark_results:
            for test_name, test_result in benchmark_results['load_tests'].items():
                if 'error_rate' in test_result:
                    max_error_rate = thresholds.get('error_rate_max', 0.05)
                    if test_result['error_rate'] > max_error_rate:
                        alerts.append({
                            'type': 'error_rate_high',
                            'test': test_name,
                            'current': test_result['error_rate'],
                            'threshold': max_error_rate,
                            'severity': 'critical' if test_result['error_rate'] > max_error_rate * 2 else 'warning'
                        })
                
                if 'latency' in test_result:
                    max_latency = thresholds.get('latency_p95_max', 2.0)
                    p95_latency = test_result['latency'].get('p95', 0)
                    if p95_latency > max_latency:
                        alerts.append({
                            'type': 'latency_high',
                            'test': test_name,
                            'current': p95_latency,
                            'threshold': max_latency,
                            'severity': 'critical' if p95_latency > max_latency * 1.5 else 'warning'
                        })
        
        # Check resource usage
        if 'resource_usage' in benchmark_results:
            resource_usage = benchmark_results['resource_usage']
            
            if 'cpu' in resource_usage:
                max_cpu = thresholds.get('cpu_usage_max', 85.0)
                avg_cpu = resource_usage['cpu'].get('avg', 0)
                if avg_cpu > max_cpu:
                    alerts.append({
                        'type': 'cpu_usage_high',
                        'current': avg_cpu,
                        'threshold': max_cpu,
                        'severity': 'critical' if avg_cpu > 95 else 'warning'
                    })
            
            if 'memory' in resource_usage:
                max_memory = thresholds.get('memory_usage_max', 90.0)
                avg_memory = resource_usage['memory'].get('avg', 0)
                if avg_memory > max_memory:
                    alerts.append({
                        'type': 'memory_usage_high',
                        'current': avg_memory,
                        'threshold': max_memory,
                        'severity': 'critical'
                    })
        
        # Check quantum performance
        if 'quantum_performance' in benchmark_results:
            for module, result in benchmark_results['quantum_performance'].items():
                scalability_score = result.get('scalability_score', 1.0)
                if scalability_score < 0.7:
                    alerts.append({
                        'type': 'quantum_scalability_poor',
                        'module': module,
                        'current': scalability_score,
                        'threshold': 0.7,
                        'severity': 'warning'
                    })
        
        return alerts
    
    async def _run_optimization(self) -> Dict[str, Any]:
        """Run adaptive performance optimization"""
        self.optimizer = AdaptivePerformanceOptimizer()
        
        # Run limited optimization for benchmarking
        optimization_task = asyncio.create_task(self.optimizer.start_optimization())
        
        # Let it run for configured duration
        max_experiments = self.config.get('optimization_experiments', 20)
        await asyncio.sleep(max_experiments * 60)  # Assume 1 minute per experiment
        
        await self.optimizer.stop_optimization()
        optimization_task.cancel()
        
        return self.optimizer.get_optimization_summary()
    
    async def _generate_final_report(self, results: Dict[str, Any]) -> Path:
        """Generate comprehensive final report"""
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"comprehensive_benchmark_report_{report_timestamp}"
        
        # Generate different report formats
        reports_dir = Path("benchmark_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # JSON report
        json_report_path = reports_dir / f"{report_filename}.json"
        with open(json_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # HTML report
        html_report_path = reports_dir / f"{report_filename}.html"
        await self._generate_html_report(results, html_report_path)
        
        # Executive summary
        summary_path = reports_dir / f"{report_filename}_summary.txt"
        await self._generate_executive_summary(results, summary_path)
        
        logger.info(f"Reports generated:")
        logger.info(f"  JSON: {json_report_path}")
        logger.info(f"  HTML: {html_report_path}")
        logger.info(f"  Summary: {summary_path}")
        
        return json_report_path
    
    async def _generate_html_report(self, results: Dict[str, Any], output_path: Path):
        """Generate HTML performance report"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Performance Benchmark Report - {benchmark_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .alert-critical {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
        .alert-warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Benchmark Report</h1>
        <p><strong>Benchmark ID:</strong> {benchmark_id}</p>
        <p><strong>Start Time:</strong> {start_time}</p>
        <p><strong>Duration:</strong> {duration:.2f} seconds</p>
        <p><strong>Status:</strong> {status}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>Total Alerts: {alert_count}</p>
        <p>System Health: {system_health}</p>
    </div>
    
    <div class="section">
        <h2>Alerts</h2>
        {alerts_html}
    </div>
    
    <div class="section">
        <h2>Load Test Results</h2>
        {load_tests_html}
    </div>
    
    <div class="section">
        <h2>Resource Usage</h2>
        {resource_usage_html}
    </div>
    
    <div class="section">
        <h2>Quantum Intelligence Performance</h2>
        {quantum_performance_html}
    </div>
</body>
</html>
        """
        
        # Generate sections
        alerts_html = self._generate_alerts_html(results.get('alerts', []))
        load_tests_html = self._generate_load_tests_html(results.get('benchmark_results', {}).get('load_tests', {}))
        resource_usage_html = self._generate_resource_usage_html(results.get('benchmark_results', {}).get('resource_usage', {}))
        quantum_performance_html = self._generate_quantum_performance_html(results.get('benchmark_results', {}).get('quantum_performance', {}))
        
        # Determine system health
        alert_count = len(results.get('alerts', []))
        critical_alerts = len([a for a in results.get('alerts', []) if a.get('severity') == 'critical'])
        
        if critical_alerts > 0:
            system_health = "CRITICAL"
        elif alert_count > 0:
            system_health = "WARNING"
        else:
            system_health = "HEALTHY"
        
        html_content = html_template.format(
            benchmark_id=results.get('benchmark_run_id', 'Unknown'),
            start_time=results.get('start_time', 'Unknown'),
            duration=results.get('total_duration', 0),
            status=results.get('status', 'Unknown'),
            alert_count=alert_count,
            system_health=system_health,
            alerts_html=alerts_html,
            load_tests_html=load_tests_html,
            resource_usage_html=resource_usage_html,
            quantum_performance_html=quantum_performance_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate HTML for alerts section"""
        if not alerts:
            return "<p>No alerts generated.</p>"
        
        html_parts = []
        for alert in alerts:
            severity_class = f"alert-{alert.get('severity', 'warning')}"
            html_parts.append(f"""
                <div class="alert {severity_class}">
                    <strong>{alert.get('type', 'Alert').replace('_', ' ').title()}</strong><br>
                    Current: {alert.get('current', 'N/A')}<br>
                    Threshold: {alert.get('threshold', 'N/A')}<br>
                    Severity: {alert.get('severity', 'Unknown').upper()}
                </div>
            """)
        
        return "".join(html_parts)
    
    def _generate_load_tests_html(self, load_tests: Dict[str, Any]) -> str:
        """Generate HTML for load test results"""
        if not load_tests:
            return "<p>No load test results available.</p>"
        
        html_parts = ["<table><tr><th>Test</th><th>Throughput</th><th>P95 Latency</th><th>Error Rate</th></tr>"]
        
        for test_name, test_result in load_tests.items():
            if isinstance(test_result, dict):
                throughput = test_result.get('throughput', 'N/A')
                p95_latency = test_result.get('latency', {}).get('p95', 'N/A')
                error_rate = test_result.get('error_rate', 'N/A')
                
                html_parts.append(f"""
                    <tr>
                        <td>{test_name}</td>
                        <td>{throughput}</td>
                        <td>{p95_latency}</td>
                        <td>{error_rate}</td>
                    </tr>
                """)
        
        html_parts.append("</table>")
        return "".join(html_parts)
    
    def _generate_resource_usage_html(self, resource_usage: Dict[str, Any]) -> str:
        """Generate HTML for resource usage"""
        if not resource_usage:
            return "<p>No resource usage data available.</p>"
        
        html_parts = []
        
        for resource_type, metrics in resource_usage.items():
            if isinstance(metrics, dict):
                html_parts.append(f"<h3>{resource_type.title()}</h3>")
                for metric_name, value in metrics.items():
                    html_parts.append(f'<div class="metric"><strong>{metric_name}:</strong> {value}</div>')
        
        return "".join(html_parts)
    
    def _generate_quantum_performance_html(self, quantum_performance: Dict[str, Any]) -> str:
        """Generate HTML for quantum performance results"""
        if not quantum_performance:
            return "<p>No quantum performance data available.</p>"
        
        html_parts = ["<table><tr><th>Module</th><th>Scalability Score</th><th>Status</th></tr>"]
        
        for module, result in quantum_performance.items():
            if isinstance(result, dict):
                scalability_score = result.get('scalability_score', 'N/A')
                status = "Good" if isinstance(scalability_score, (int, float)) and scalability_score >= 0.8 else "Needs Attention"
                
                html_parts.append(f"""
                    <tr>
                        <td>{module.replace('_', ' ').title()}</td>
                        <td>{scalability_score:.3f if isinstance(scalability_score, (int, float)) else scalability_score}</td>
                        <td>{status}</td>
                    </tr>
                """)
        
        html_parts.append("</table>")
        return "".join(html_parts)
    
    async def _generate_executive_summary(self, results: Dict[str, Any], output_path: Path):
        """Generate executive summary text report"""
        summary_lines = [
            "PERFORMANCE BENCHMARK EXECUTIVE SUMMARY",
            "=" * 50,
            "",
            f"Benchmark ID: {results.get('benchmark_run_id', 'Unknown')}",
            f"Execution Time: {results.get('start_time', 'Unknown')}",
            f"Total Duration: {results.get('total_duration', 0):.2f} seconds",
            f"Status: {results.get('status', 'Unknown').upper()}",
            "",
            "KEY FINDINGS:",
            "-" * 20
        ]
        
        # Alert summary
        alerts = results.get('alerts', [])
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        warning_alerts = [a for a in alerts if a.get('severity') == 'warning']
        
        summary_lines.extend([
            f"• Critical Issues: {len(critical_alerts)}",
            f"• Warnings: {len(warning_alerts)}",
            f"• Total Alerts: {len(alerts)}",
            ""
        ])
        
        # Load test summary
        load_tests = results.get('benchmark_results', {}).get('load_tests', {})
        if load_tests:
            summary_lines.extend([
                "LOAD TEST RESULTS:",
                "-" * 20
            ])
            
            for test_name, test_result in load_tests.items():
                if isinstance(test_result, dict):
                    throughput = test_result.get('throughput', 'N/A')
                    error_rate = test_result.get('error_rate', 'N/A')
                    summary_lines.append(f"• {test_name}: {throughput} req/s, {error_rate:.1%} error rate")
        
        # Recommendations
        if critical_alerts or warning_alerts:
            summary_lines.extend([
                "",
                "RECOMMENDATIONS:",
                "-" * 20
            ])
            
            if critical_alerts:
                summary_lines.append("• IMMEDIATE ACTION REQUIRED: Address critical performance issues")
            if warning_alerts:
                summary_lines.append("• Review and optimize components triggering warnings")
            summary_lines.append("• Implement continuous performance monitoring")
            summary_lines.append("• Schedule regular performance benchmarking")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))

async def main():
    """Main entry point for benchmark runner"""
    parser = argparse.ArgumentParser(description='Run comprehensive performance benchmarks')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--monitoring', '-m', action='store_true', help='Enable real-time monitoring')
    parser.add_argument('--optimization', '-o', action='store_true', help='Enable adaptive optimization')
    parser.add_argument('--output', '-f', choices=['json', 'html', 'both'], default='both', help='Output format')
    parser.add_argument('--duration', '-d', type=int, help='Override benchmark duration (seconds)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and configure benchmark runner
    runner = BenchmarkRunner(args.config)
    
    # Override config with command line arguments
    if args.monitoring:
        runner.config['enable_monitoring'] = True
    if args.optimization:
        runner.config['enable_optimization'] = True
    if args.duration:
        # Update all scenario durations
        for scenario in runner.config.get('load_test_scenarios', []):
            scenario['duration'] = args.duration
    
    try:
        # Execute benchmark suite
        results = await runner.run_benchmark_suite()
        
        # Print summary to console
        print("\n" + "="*60)
        print("BENCHMARK EXECUTION COMPLETED")
        print("="*60)
        print(f"Status: {results.get('status', 'Unknown').upper()}")
        print(f"Duration: {results.get('total_duration', 0):.2f} seconds")
        print(f"Alerts: {len(results.get('alerts', []))}")
        print(f"Report: {results.get('report_path', 'Not generated')}")
        
        if results.get('alerts'):
            print("\nALERTS:")
            for alert in results['alerts']:
                severity = alert.get('severity', 'unknown').upper()
                alert_type = alert.get('type', 'unknown').replace('_', ' ').title()
                print(f"  [{severity}] {alert_type}")
        
        return 0 if results.get('status') == 'completed' else 1
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)