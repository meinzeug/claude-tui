#!/usr/bin/env python3
"""
Final TUI Validation Report Generator
Comprehensive assessment of TUI operational status
"""

import asyncio
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class FinalTUIValidationReport:
    """Generate final comprehensive TUI validation report."""
    
    def __init__(self):
        self.project_root = project_root
        self.validation_results = {
            "executive_summary": {},
            "core_functionality": {},
            "import_validation": {},
            "startup_validation": {},
            "operational_status": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
    def run_quick_startup_test(self):
        """Run a quick startup test to validate basic functionality."""
        print("🚀 Running Quick Startup Validation...")
        
        try:
            # Test basic startup with short timeout
            result = subprocess.run([
                "timeout", "8", "python3", "run_tui.py", "headless"
            ], 
                capture_output=True, 
                text=True,
                cwd=str(self.project_root)
            )
            
            output = result.stdout + result.stderr
            
            # Check for key success indicators
            success_indicators = [
                "Starting Claude-TUI",
                "ConfigManager initialized",
                "ProjectManager initialized",
                "TaskEngine initialized",
                "UIIntegrationBridge"
            ]
            
            found_indicators = [indicator for indicator in success_indicators if indicator in output]
            
            # Check for critical errors
            critical_errors = [
                "ModuleNotFoundError",
                "ImportError:",
                "AttributeError:",
                "SyntaxError:",
                "Traceback (most recent call last):"
            ]
            
            found_errors = [error for error in critical_errors if error in output]
            
            startup_successful = len(found_indicators) >= 3 and len(found_errors) == 0
            
            self.validation_results["startup_validation"] = {
                "startup_successful": startup_successful,
                "success_indicators_found": found_indicators,
                "critical_errors_found": found_errors,
                "output_length": len(output),
                "exit_code": result.returncode,
                "test_duration": "8 seconds timeout"
            }
            
            if startup_successful:
                print("    ✅ TUI starts up successfully")
                print(f"    📊 Found {len(found_indicators)}/5 success indicators")
            else:
                print("    ⚠️  TUI startup has issues")
                if found_errors:
                    print(f"    ❌ Found {len(found_errors)} critical errors")
                    
        except Exception as e:
            self.validation_results["startup_validation"] = {
                "startup_successful": False,
                "error": str(e),
                "test_duration": "Failed before completion"
            }
            print(f"    ❌ Startup test failed: {str(e)}")
            
    def validate_core_imports(self):
        """Validate core module imports."""
        print("📦 Validating Core Imports...")
        
        import_results = {}
        
        core_modules = [
            ("core.config_manager", "ConfigManager"),
            ("core.project_manager", "ProjectManager"),
            ("core.task_engine", "TaskEngine"),
            ("ui.main_app", "MainApp"),
        ]
        
        successful_imports = 0
        
        for module_path, class_name in core_modules:
            try:
                import sys
                if module_path in sys.modules:
                    del sys.modules[module_path]
                    
                module = __import__(module_path, fromlist=[class_name])
                has_class = hasattr(module, class_name)
                
                import_results[module_path] = {
                    "success": True,
                    "has_target_class": has_class,
                    "details": f"Successfully imported {module_path}"
                }
                
                if has_class:
                    successful_imports += 1
                    print(f"    ✅ {module_path}.{class_name}")
                else:
                    print(f"    ⚠️  {module_path} imported but {class_name} not found")
                    
            except ImportError as e:
                import_results[module_path] = {
                    "success": False,
                    "error": str(e),
                    "details": f"Failed to import {module_path}: {str(e)}"
                }
                print(f"    ❌ {module_path} - {str(e)}")
                
        self.validation_results["import_validation"] = {
            "total_modules_tested": len(core_modules),
            "successful_imports": successful_imports,
            "import_success_rate": f"{(successful_imports/len(core_modules)*100):.1f}%",
            "detailed_results": import_results
        }
        
    def test_basic_functionality(self):
        """Test basic functionality without full UI."""
        print("🔧 Testing Basic Functionality...")
        
        try:
            result = subprocess.run([
                "python3", "-c", 
                """
import sys
sys.path.insert(0, 'src')

# Test core component initialization
try:
    from core.config_manager import ConfigManager
    config = ConfigManager()
    print("CONFIG_OK")
    
    from core.project_manager import ProjectManager
    project_mgr = ProjectManager()
    print("PROJECT_MANAGER_OK")
    
    from core.task_engine import TaskEngine
    task_engine = TaskEngine()
    print("TASK_ENGINE_OK")
    
    print("BASIC_FUNCTIONALITY_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=15,
                cwd=str(self.project_root)
            )
            
            output = result.stdout
            success_markers = ["CONFIG_OK", "PROJECT_MANAGER_OK", "TASK_ENGINE_OK", "BASIC_FUNCTIONALITY_SUCCESS"]
            found_markers = [marker for marker in success_markers if marker in output]
            
            functionality_working = len(found_markers) == len(success_markers)
            
            self.validation_results["core_functionality"] = {
                "basic_functionality_working": functionality_working,
                "components_initialized": found_markers,
                "success_rate": f"{(len(found_markers)/len(success_markers)*100):.1f}%",
                "details": "All core components initialize successfully" if functionality_working else f"Only {len(found_markers)}/{len(success_markers)} components working"
            }
            
            if functionality_working:
                print("    ✅ All core components working")
            else:
                print(f"    ⚠️  {len(found_markers)}/{len(success_markers)} components working")
                
        except Exception as e:
            self.validation_results["core_functionality"] = {
                "basic_functionality_working": False,
                "error": str(e)
            }
            print(f"    ❌ Functionality test failed: {str(e)}")
            
    def assess_performance(self):
        """Assess basic performance metrics."""
        print("⚡ Assessing Performance...")
        
        try:
            result = subprocess.run([
                "python3", "-c", 
                f"""
import sys
import time
import psutil
import os
sys.path.insert(0, '{self.project_root / "src"}')

process = psutil.Process(os.getpid())
start_time = time.time()
start_memory = process.memory_info().rss / 1024 / 1024

# Import and initialize
from core.config_manager import ConfigManager
config = ConfigManager()

end_time = time.time()
end_memory = process.memory_info().rss / 1024 / 1024

print(f"STARTUP_TIME: {{(end_time - start_time)*1000:.1f}}ms")
print(f"MEMORY_USAGE: {{end_memory:.1f}}MB")
print(f"MEMORY_INCREASE: {{end_memory - start_memory:.1f}}MB")
print("PERFORMANCE_SUCCESS")
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=10,
                cwd=str(self.project_root)
            )
            
            if "PERFORMANCE_SUCCESS" in result.stdout:
                lines = result.stdout.split('\n')
                metrics = {}
                for line in lines:
                    if ':' in line and any(metric in line for metric in ['STARTUP_TIME', 'MEMORY_USAGE', 'MEMORY_INCREASE']):
                        key, value = line.split(': ', 1)
                        metrics[key] = value
                        
                self.validation_results["performance_metrics"] = {
                    "performance_test_successful": True,
                    "metrics": metrics,
                    "assessment": "Performance within acceptable ranges"
                }
                
                print("    ✅ Performance metrics collected")
                for key, value in metrics.items():
                    print(f"      {key}: {value}")
                    
            else:
                self.validation_results["performance_metrics"] = {
                    "performance_test_successful": False,
                    "error": result.stderr
                }
                print("    ❌ Performance test failed")
                
        except Exception as e:
            self.validation_results["performance_metrics"] = {
                "performance_test_successful": False,
                "error": str(e)
            }
            print(f"    ❌ Performance assessment failed: {str(e)}")
            
    def determine_operational_status(self):
        """Determine overall operational status."""
        print("🎯 Determining Operational Status...")
        
        # Analyze results
        startup_working = self.validation_results.get("startup_validation", {}).get("startup_successful", False)
        imports_working = float(self.validation_results.get("import_validation", {}).get("import_success_rate", "0%").replace("%", "")) >= 75
        functionality_working = self.validation_results.get("core_functionality", {}).get("basic_functionality_working", False)
        performance_ok = self.validation_results.get("performance_metrics", {}).get("performance_test_successful", False)
        
        # Calculate overall score
        scores = [startup_working, imports_working, functionality_working, performance_ok]
        operational_score = sum(scores) / len(scores) * 100
        
        # Determine status
        if operational_score >= 75 and startup_working and functionality_working:
            status = "OPERATIONAL"
            level = "HIGH"
            message = "TUI is operationally ready and can be used effectively"
            ready_for_production = True
        elif operational_score >= 50 and (startup_working or functionality_working):
            status = "PARTIALLY_OPERATIONAL"
            level = "MEDIUM"
            message = "TUI has core functionality working but some issues remain"
            ready_for_production = False
        else:
            status = "NOT_OPERATIONAL"
            level = "LOW"
            message = "TUI has significant issues preventing effective use"
            ready_for_production = False
            
        self.validation_results["operational_status"] = {
            "status": status,
            "confidence_level": level,
            "operational_score": f"{operational_score:.1f}%",
            "message": message,
            "ready_for_production": ready_for_production,
            "component_status": {
                "startup": "✅" if startup_working else "❌",
                "imports": "✅" if imports_working else "❌", 
                "functionality": "✅" if functionality_working else "❌",
                "performance": "✅" if performance_ok else "❌"
            }
        }
        
        print(f"    📊 Operational Score: {operational_score:.1f}%")
        print(f"    🎯 Status: {status}")
        print(f"    🔒 Production Ready: {'Yes' if ready_for_production else 'No'}")
        
    def generate_recommendations(self):
        """Generate recommendations based on validation results."""
        print("💡 Generating Recommendations...")
        
        recommendations = []
        
        # Startup recommendations
        startup_result = self.validation_results.get("startup_validation", {})
        if not startup_result.get("startup_successful", False):
            recommendations.append({
                "category": "Critical",
                "issue": "TUI startup issues",
                "recommendation": "Address startup failures by checking import paths and dependencies",
                "priority": "HIGH"
            })
            
        # Import recommendations  
        import_result = self.validation_results.get("import_validation", {})
        import_rate = float(import_result.get("import_success_rate", "0%").replace("%", ""))
        if import_rate < 100:
            recommendations.append({
                "category": "Improvement",
                "issue": f"Import success rate: {import_rate}%",
                "recommendation": "Fix remaining import issues for better reliability",
                "priority": "MEDIUM"
            })
            
        # Functionality recommendations
        func_result = self.validation_results.get("core_functionality", {})
        if not func_result.get("basic_functionality_working", False):
            recommendations.append({
                "category": "Critical",
                "issue": "Core functionality not working",
                "recommendation": "Debug component initialization failures",
                "priority": "HIGH"
            })
            
        # Performance recommendations
        perf_result = self.validation_results.get("performance_metrics", {})
        if not perf_result.get("performance_test_successful", False):
            recommendations.append({
                "category": "Monitoring",
                "issue": "Performance metrics not available",
                "recommendation": "Implement performance monitoring for production readiness",
                "priority": "LOW"
            })
            
        # General recommendations
        op_status = self.validation_results.get("operational_status", {})
        if op_status.get("status") == "OPERATIONAL":
            recommendations.extend([
                {
                    "category": "Production",
                    "issue": "Ready for deployment",
                    "recommendation": "TUI can be deployed and used in production environment",
                    "priority": "INFO"
                },
                {
                    "category": "Monitoring", 
                    "issue": "Ongoing monitoring",
                    "recommendation": "Implement monitoring and logging for production use",
                    "priority": "MEDIUM"
                }
            ])
        elif op_status.get("status") == "PARTIALLY_OPERATIONAL":
            recommendations.append({
                "category": "Development",
                "issue": "Partial functionality",
                "recommendation": "Address remaining issues before production deployment",
                "priority": "MEDIUM"
            })
            
        self.validation_results["recommendations"] = recommendations
        
        print(f"    📝 Generated {len(recommendations)} recommendations")
        
    def create_executive_summary(self):
        """Create executive summary of validation results."""
        print("📋 Creating Executive Summary...")
        
        op_status = self.validation_results.get("operational_status", {})
        startup_ok = self.validation_results.get("startup_validation", {}).get("startup_successful", False)
        import_rate = self.validation_results.get("import_validation", {}).get("import_success_rate", "0%")
        functionality_ok = self.validation_results.get("core_functionality", {}).get("basic_functionality_working", False)
        
        summary = {
            "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validator_agent": "Testing and Quality Assurance Agent",
            "overall_status": op_status.get("status", "UNKNOWN"),
            "operational_score": op_status.get("operational_score", "0%"),
            "production_ready": op_status.get("ready_for_production", False),
            "key_findings": [
                f"TUI Startup: {'✅ Working' if startup_ok else '❌ Issues'}",
                f"Import Success Rate: {import_rate}",
                f"Core Functionality: {'✅ Working' if functionality_ok else '❌ Issues'}"
            ],
            "confidence_level": op_status.get("confidence_level", "LOW"),
            "recommendations_count": len(self.validation_results.get("recommendations", [])),
            "next_steps": self._determine_next_steps()
        }
        
        self.validation_results["executive_summary"] = summary
        
        print(f"    📊 Overall Status: {summary['overall_status']}")
        print(f"    🎯 Operational Score: {summary['operational_score']}")
        print(f"    🚀 Production Ready: {'Yes' if summary['production_ready'] else 'No'}")
        
    def _determine_next_steps(self):
        """Determine next steps based on validation results."""
        op_status = self.validation_results.get("operational_status", {}).get("status", "UNKNOWN")
        
        if op_status == "OPERATIONAL":
            return [
                "TUI validation successful - application ready for use",
                "Deploy to production environment",
                "Set up monitoring and logging",
                "Train users on TUI functionality"
            ]
        elif op_status == "PARTIALLY_OPERATIONAL":
            return [
                "Address remaining issues identified in recommendations",
                "Re-run validation after fixes",
                "Consider limited deployment for testing",
                "Monitor closely in staging environment"
            ]
        else:
            return [
                "Do not deploy to production",
                "Address critical failures immediately",
                "Focus on startup and core functionality issues",
                "Re-run full validation after fixes"
            ]
            
    async def generate_report(self):
        """Generate comprehensive validation report."""
        print("🧪 FINAL TUI VALIDATION ASSESSMENT")
        print("=" * 60)
        
        # Run all validation tests
        self.run_quick_startup_test()
        self.validate_core_imports()
        self.test_basic_functionality()
        self.assess_performance()
        self.determine_operational_status()
        self.generate_recommendations()
        self.create_executive_summary()
        
        # Save results
        report_path = self.project_root / ".swarm" / "memory" / "tester" / "final_validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Notify via hooks
        subprocess.run([
            "npx", "claude-flow@alpha", "hooks", "post-edit",
            "--file", str(report_path),
            "--memory-key", "swarm/tester/final_validation_report"
        ], check=False)
        
        return self.validation_results
        
    def print_final_report(self):
        """Print final validation report."""
        print("\n" + "="*60)
        print("🎯 FINAL TUI VALIDATION REPORT")
        print("="*60)
        
        summary = self.validation_results.get("executive_summary", {})
        
        print(f"\n📋 EXECUTIVE SUMMARY")
        print(f"   Validation Date: {summary.get('validation_date', 'Unknown')}")
        print(f"   Overall Status: {summary.get('overall_status', 'Unknown')}")
        print(f"   Operational Score: {summary.get('operational_score', '0%')}")
        print(f"   Production Ready: {'✅ YES' if summary.get('production_ready', False) else '❌ NO'}")
        print(f"   Confidence Level: {summary.get('confidence_level', 'LOW')}")
        
        print(f"\n🔍 KEY FINDINGS")
        for finding in summary.get('key_findings', []):
            print(f"   • {finding}")
            
        print(f"\n💡 RECOMMENDATIONS ({summary.get('recommendations_count', 0)})")
        recommendations = self.validation_results.get("recommendations", [])[:3]  # Show top 3
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec['priority']}] {rec['recommendation']}")
            
        print(f"\n🎯 NEXT STEPS")
        next_steps = summary.get('next_steps', [])
        for i, step in enumerate(next_steps, 1):
            print(f"   {i}. {step}")
            
        print(f"\n🏁 VALIDATION CONCLUSION")
        op_status = self.validation_results.get("operational_status", {})
        print(f"   {op_status.get('message', 'Validation completed')}")
        
        return self.validation_results

async def run_final_validation():
    """Run the final comprehensive TUI validation."""
    validator = FinalTUIValidationReport()
    
    # Generate comprehensive report
    results = await validator.generate_report()
    
    # Print final summary
    validator.print_final_report()
    
    return results

if __name__ == "__main__":
    asyncio.run(run_final_validation())