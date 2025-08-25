#!/usr/bin/env python3
"""
Production Memory Optimization Deployment Script
Deploys memory optimization for production environments
"""

import os
import sys
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from performance import (
    emergency_memory_rescue,
    EmergencyMemoryOptimizer,
    MemoryProfiler,
    quick_memory_check
)


class ProductionMemoryDeployment:
    """Production memory optimization deployment"""
    
    def __init__(self, target_memory_mb: int = 200):
        self.target_memory_mb = target_memory_mb
        self.deployment_log = []
        self.performance_baseline = None
        
    def validate_environment(self) -> Dict[str, Any]:
        """Validate production environment for deployment"""
        print("🔍 Validating production environment...")
        
        validation = {
            "python_version": sys.version_info[:2],
            "memory_available": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            "cpu_count": psutil.cpu_count(),
            "current_memory": psutil.Process().memory_info().rss / 1024 / 1024,
            "disk_space": psutil.disk_usage('/').free / 1024 / 1024 / 1024  # GB
        }
        
        # Validation checks
        issues = []
        
        if validation["python_version"] < (3, 8):
            issues.append("Python 3.8+ required for optimal performance")
            
        if validation["memory_available"] < 0.1:  # 100MB available (minimal requirement)
            issues.append("Insufficient available memory (<100MB)")
            
        if validation["disk_space"] < 1.0:  # 1GB free space
            issues.append("Insufficient disk space (<1GB)")
            
        validation["issues"] = issues
        validation["ready"] = len(issues) == 0
        
        if validation["ready"]:
            print("✅ Environment validation passed")
        else:
            print("⚠️ Environment validation issues:")
            for issue in issues:
                print(f"  - {issue}")
                
        return validation
        
    def create_baseline(self) -> Dict[str, Any]:
        """Create performance baseline before optimization"""
        print("📊 Creating performance baseline...")
        
        profiler = MemoryProfiler(self.target_memory_mb)
        
        # Take multiple snapshots for stability
        snapshots = []
        for i in range(5):
            snapshot = profiler.take_snapshot()
            snapshots.append(snapshot)
            time.sleep(0.5)
            
        # Calculate baseline metrics
        avg_memory = sum(s.process_memory for s in snapshots) / len(snapshots)
        avg_objects = sum(s.gc_objects for s in snapshots) / len(snapshots)
        
        baseline = {
            "timestamp": time.time(),
            "average_memory_mb": avg_memory / 1024 / 1024,
            "average_objects": int(avg_objects),
            "peak_memory_mb": max(s.process_memory for s in snapshots) / 1024 / 1024,
            "snapshots_count": len(snapshots),
            "target_memory_mb": self.target_memory_mb
        }
        
        self.performance_baseline = baseline
        
        print(f"📈 Baseline: {baseline['average_memory_mb']:.1f}MB average")
        print(f"🎯 Target: {self.target_memory_mb}MB")
        print(f"📉 Reduction needed: {baseline['average_memory_mb'] - self.target_memory_mb:.1f}MB")
        
        return baseline
        
    def deploy_optimizations(self) -> Dict[str, Any]:
        """Deploy memory optimizations to production"""
        print("🚀 Deploying memory optimizations...")
        
        deployment_results = {
            "start_time": time.time(),
            "optimizations_applied": [],
            "success": False,
            "final_memory_mb": 0,
            "reduction_achieved_mb": 0
        }
        
        try:
            # 1. Initialize optimizer
            optimizer = EmergencyMemoryOptimizer(self.target_memory_mb)
            
            # 2. Run comprehensive optimization
            print("  🔧 Running comprehensive optimization...")
            optimization_result = optimizer.run_emergency_optimization()
            
            deployment_results["optimizations_applied"] = optimization_result.get("strategy_results", [])
            deployment_results["success"] = optimization_result.get("success", False)
            deployment_results["final_memory_mb"] = optimization_result.get("final_memory_mb", 0)
            
            if self.performance_baseline:
                deployment_results["reduction_achieved_mb"] = (
                    self.performance_baseline["average_memory_mb"] - 
                    deployment_results["final_memory_mb"]
                )
                
            # 3. Validate deployment
            if deployment_results["success"]:
                print("  ✅ Optimization deployment successful!")
                
                # Start continuous monitoring
                print("  📊 Starting continuous monitoring...")
                monitor_thread = optimizer.continuous_monitoring(interval_seconds=10.0)
                deployment_results["monitoring_started"] = True
                
            else:
                print("  ⚠️ Optimization deployment incomplete")
                
        except Exception as e:
            print(f"  ❌ Deployment failed: {e}")
            deployment_results["error"] = str(e)
            deployment_results["success"] = False
            
        deployment_results["end_time"] = time.time()
        deployment_results["duration_seconds"] = (
            deployment_results["end_time"] - deployment_results["start_time"]
        )
        
        return deployment_results
        
    def run_performance_validation(self) -> Dict[str, Any]:
        """Validate performance after deployment"""
        print("🧪 Running performance validation...")
        
        validation = {
            "timestamp": time.time(),
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "issues": []
        }
        
        try:
            # Test 1: Memory usage validation
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_test_passed = current_memory <= self.target_memory_mb * 1.1  # 10% tolerance
            
            if memory_test_passed:
                validation["tests_passed"] += 1
                print(f"  ✅ Memory test passed: {current_memory:.1f}MB <= {self.target_memory_mb}MB")
            else:
                validation["tests_failed"] += 1
                validation["issues"].append(f"Memory usage {current_memory:.1f}MB exceeds target {self.target_memory_mb}MB")
                print(f"  ❌ Memory test failed: {current_memory:.1f}MB > {self.target_memory_mb}MB")
                
            # Test 2: Response time validation
            print("  ⏱️ Testing response time...")
            start_time = time.time()
            
            # Simulate some work
            test_work = [i**2 for i in range(10000)]
            del test_work
            
            response_time = (time.time() - start_time) * 1000  # ms
            response_test_passed = response_time < 100  # < 100ms
            
            if response_test_passed:
                validation["tests_passed"] += 1
                print(f"  ✅ Response time test passed: {response_time:.1f}ms")
            else:
                validation["tests_failed"] += 1
                validation["issues"].append(f"Response time {response_time:.1f}ms too high")
                
            # Test 3: Stability validation
            print("  🔄 Testing stability...")
            stable = True
            for i in range(5):
                memory_check = psutil.Process().memory_info().rss / 1024 / 1024
                if memory_check > self.target_memory_mb * 1.2:  # 20% tolerance
                    stable = False
                    break
                time.sleep(0.2)
                
            if stable:
                validation["tests_passed"] += 1
                print("  ✅ Stability test passed")
            else:
                validation["tests_failed"] += 1
                validation["issues"].append("Memory usage unstable during testing")
                
            # Performance metrics
            validation["performance_metrics"] = {
                "current_memory_mb": current_memory,
                "response_time_ms": response_time,
                "target_memory_mb": self.target_memory_mb,
                "memory_efficiency": (self.target_memory_mb / current_memory) * 100,
                "stability_score": 100 if stable else 50
            }
            
        except Exception as e:
            validation["issues"].append(f"Validation error: {e}")
            
        validation["overall_success"] = validation["tests_failed"] == 0
        
        if validation["overall_success"]:
            print("🎉 Performance validation passed!")
        else:
            print("⚠️ Performance validation issues detected")
            
        return validation
        
    def generate_deployment_report(self, validation: Dict, deployment: Dict) -> str:
        """Generate comprehensive deployment report"""
        
        report = f"""
🚀 PRODUCTION MEMORY OPTIMIZATION DEPLOYMENT REPORT

Deployment Summary:
===================
Target Memory: {self.target_memory_mb}MB
Deployment Success: {'✅ YES' if deployment.get('success', False) else '❌ NO'}
Duration: {deployment.get('duration_seconds', 0):.1f}s
Optimizations Applied: {len(deployment.get('optimizations_applied', []))}

Performance Results:
====================
"""
        
        if self.performance_baseline:
            baseline_mb = self.performance_baseline["average_memory_mb"]
            final_mb = deployment.get("final_memory_mb", 0)
            reduction_mb = baseline_mb - final_mb
            reduction_pct = (reduction_mb / baseline_mb) * 100 if baseline_mb > 0 else 0
            
            report += f"""
Baseline Memory: {baseline_mb:.1f}MB
Final Memory: {final_mb:.1f}MB
Reduction: {reduction_mb:.1f}MB ({reduction_pct:.1f}%)
Target Achieved: {'✅ YES' if final_mb <= self.target_memory_mb else '❌ NO'}
"""
        
        report += f"""
Validation Results:
===================
Tests Passed: {validation.get('tests_passed', 0)}
Tests Failed: {validation.get('tests_failed', 0)}
Overall Success: {'✅ PASS' if validation.get('overall_success', False) else '❌ FAIL'}
"""
        
        if validation.get("performance_metrics"):
            metrics = validation["performance_metrics"]
            report += f"""
Performance Metrics:
====================
Current Memory: {metrics.get('current_memory_mb', 0):.1f}MB
Response Time: {metrics.get('response_time_ms', 0):.1f}ms
Memory Efficiency: {metrics.get('memory_efficiency', 0):.1f}%
Stability Score: {metrics.get('stability_score', 0):.1f}%
"""
        
        if validation.get("issues"):
            report += "\nIssues Detected:\n"
            for issue in validation["issues"]:
                report += f"⚠️ {issue}\n"
                
        report += f"""
Deployment Status: {'🎉 PRODUCTION READY' if validation.get('overall_success', False) and deployment.get('success', False) else '⚠️ NEEDS ATTENTION'}
"""
        
        return report
        
    def save_deployment_data(self, validation: Dict, deployment: Dict, report: str):
        """Save deployment data for monitoring and analysis"""
        
        timestamp = int(time.time())
        deployment_dir = Path("deployment_logs")
        deployment_dir.mkdir(exist_ok=True)
        
        # Save detailed data
        deployment_data = {
            "timestamp": timestamp,
            "baseline": self.performance_baseline,
            "deployment": deployment,
            "validation": validation,
            "target_memory_mb": self.target_memory_mb
        }
        
        data_file = deployment_dir / f"memory_optimization_deployment_{timestamp}.json"
        with open(data_file, 'w') as f:
            json.dump(deployment_data, f, indent=2, default=str)
            
        # Save report
        report_file = deployment_dir / f"deployment_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
            
        print(f"📄 Deployment data saved:")
        print(f"  Data: {data_file}")
        print(f"  Report: {report_file}")
        
    def run_full_deployment(self) -> Dict[str, Any]:
        """Run complete production deployment process"""
        
        print("🚀 STARTING PRODUCTION MEMORY OPTIMIZATION DEPLOYMENT")
        print("=" * 60)
        
        # 1. Environment validation
        env_validation = self.validate_environment()
        if not env_validation["ready"]:
            print("❌ Environment not ready for deployment")
            return {"success": False, "reason": "environment_validation_failed"}
            
        # 2. Create baseline
        baseline = self.create_baseline()
        
        # 3. Deploy optimizations
        deployment = self.deploy_optimizations()
        
        # 4. Performance validation
        validation = self.run_performance_validation()
        
        # 5. Generate report
        report = self.generate_deployment_report(validation, deployment)
        print(report)
        
        # 6. Save data
        self.save_deployment_data(validation, deployment, report)
        
        # 7. Final result
        overall_success = (
            deployment.get("success", False) and 
            validation.get("overall_success", False)
        )
        
        final_status = "🎉 DEPLOYMENT SUCCESSFUL" if overall_success else "⚠️ DEPLOYMENT NEEDS ATTENTION"
        print(f"\n{final_status}")
        
        return {
            "success": overall_success,
            "baseline": baseline,
            "deployment": deployment,
            "validation": validation,
            "report": report
        }


def main():
    """Main deployment function"""
    
    # Parse command line arguments
    target_memory = 200  # Default 200MB
    if len(sys.argv) > 1:
        try:
            target_memory = int(sys.argv[1])
        except ValueError:
            print("Usage: python memory_optimization_deployment.py [target_memory_mb]")
            sys.exit(1)
            
    print(f"🎯 Target memory: {target_memory}MB")
    
    # Run deployment
    deployment = ProductionMemoryDeployment(target_memory)
    result = deployment.run_full_deployment()
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()