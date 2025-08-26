#!/usr/bin/env python3
"""
MCP Integration Validation Script
Validates the complete MCP server and claude-flow integration implementation
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationResult:
    def __init__(self):
        self.tests: Dict[str, Any] = {}
        self.start_time = time.time()
        self.total_tests = 0
        self.passed_tests = 0
    
    def add_test(self, name: str, success: bool, details: Any = None, error: str = None):
        self.tests[name] = {
            "success": success,
            "details": details,
            "error": error,
            "timestamp": time.time()
        }
        self.total_tests += 1
        if success:
            self.passed_tests += 1
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.total_tests - self.passed_tests,
            "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            "total_time": time.time() - self.start_time
        }

async def validate_file_structure(result: ValidationResult):
    """Validate that all required files were created"""
    logger.info("Validating file structure...")
    
    required_files = [
        # MCP Integration
        "src/mcp/__init__.py",
        "src/mcp/server.py", 
        "src/mcp/endpoints.py",
        
        # Monitoring
        "src/monitoring/__init__.py",
        "src/monitoring/dashboard.py",
        
        # Integration
        "src/integration/__init__.py",
        "src/integration/bridge.py",
        "src/integration/tui_connector.py",
        "src/integration/hooks_manager.py",
        "src/integration/startup_manager.py",
        "src/integration/test_integration.py",
        "src/integration/run_mcp_integration.py"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    success = len(missing_files) == 0
    
    result.add_test("file_structure", success, {
        "total_required": len(required_files),
        "existing_files": len(existing_files),
        "missing_files": missing_files,
        "existing_files_list": existing_files
    })
    
    logger.info(f"File structure validation: {'PASS' if success else 'FAIL'}")
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")

async def validate_code_syntax(result: ValidationResult):
    """Validate that all Python files have valid syntax"""
    logger.info("Validating code syntax...")
    
    python_files = [
        "src/mcp/server.py",
        "src/mcp/endpoints.py", 
        "src/monitoring/dashboard.py",
        "src/integration/bridge.py",
        "src/integration/tui_connector.py",
        "src/integration/hooks_manager.py",
        "src/integration/startup_manager.py",
        "src/integration/test_integration.py",
        "src/integration/run_mcp_integration.py"
    ]
    
    syntax_errors = []
    valid_files = []
    
    for file_path in python_files:
        path = Path(file_path)
        if not path.exists():
            continue
            
        try:
            with open(path, 'r') as f:
                code = f.read()
            
            # Try to compile the code
            compile(code, file_path, 'exec')
            valid_files.append(file_path)
            
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    success = len(syntax_errors) == 0
    
    result.add_test("code_syntax", success, {
        "valid_files": len(valid_files),
        "syntax_errors": syntax_errors,
        "checked_files": len(python_files)
    })
    
    logger.info(f"Code syntax validation: {'PASS' if success else 'FAIL'}")
    if syntax_errors:
        for error in syntax_errors:
            logger.error(f"Syntax error: {error}")

async def validate_imports(result: ValidationResult):
    """Validate that imports can be resolved"""
    logger.info("Validating imports...")
    
    import_tests = {}
    
    # Test core module imports
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        # Test MCP imports
        try:
            from src.mcp.server import MCPServerClient, SwarmCoordinator, HooksIntegration
            import_tests["mcp_server"] = True
        except Exception as e:
            import_tests["mcp_server"] = f"Error: {e}"
        
        # Test integration imports
        try:
            from src.integration.bridge import IntegrationBridge, BridgeConfig
            import_tests["integration_bridge"] = True
        except Exception as e:
            import_tests["integration_bridge"] = f"Error: {e}"
        
        # Test monitoring imports
        try:
            from src.monitoring.dashboard import MetricsCollector, SwarmMetrics
            import_tests["monitoring_dashboard"] = True
        except Exception as e:
            import_tests["monitoring_dashboard"] = f"Error: {e}"
        
        # Test hooks imports
        try:
            from src.integration.hooks_manager import DevelopmentHooksCoordinator
            import_tests["hooks_manager"] = True
        except Exception as e:
            import_tests["hooks_manager"] = f"Error: {e}"
            
    except Exception as e:
        import_tests["general_import"] = f"Error: {e}"
    
    success = all(v == True for v in import_tests.values())
    
    result.add_test("imports", success, import_tests)
    
    logger.info(f"Import validation: {'PASS' if success else 'FAIL'}")
    for module, status in import_tests.items():
        if status != True:
            logger.error(f"Import error in {module}: {status}")

async def validate_dependencies(result: ValidationResult):
    """Validate that required dependencies are available"""
    logger.info("Validating dependencies...")
    
    required_deps = [
        "asyncio", "json", "logging", "pathlib", "sqlite3", "subprocess",
        "aiohttp", "fastapi", "uvicorn", "pydantic", "textual"
    ]
    
    available_deps = []
    missing_deps = []
    
    for dep in required_deps:
        try:
            __import__(dep)
            available_deps.append(dep)
        except ImportError:
            missing_deps.append(dep)
    
    success = len(missing_deps) == 0
    
    result.add_test("dependencies", success, {
        "available_deps": available_deps,
        "missing_deps": missing_deps,
        "total_required": len(required_deps)
    })
    
    logger.info(f"Dependencies validation: {'PASS' if success else 'FAIL'}")
    if missing_deps:
        logger.warning(f"Missing dependencies: {missing_deps}")

async def validate_configuration(result: ValidationResult):
    """Validate configuration files and setup"""
    logger.info("Validating configuration...")
    
    config_checks = {}
    
    # Check if claude-flow config exists
    claude_flow_config = Path("claude-flow.config.json")
    config_checks["claude_flow_config"] = claude_flow_config.exists()
    
    # Check if .swarm directory was created
    swarm_dir = Path(".swarm")
    config_checks["swarm_directory"] = swarm_dir.exists()
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    config_checks["requirements_file"] = requirements_file.exists()
    
    success = all(config_checks.values())
    
    result.add_test("configuration", success, config_checks)
    
    logger.info(f"Configuration validation: {'PASS' if success else 'FAIL'}")

async def validate_integration_points(result: ValidationResult):
    """Validate integration points between components"""
    logger.info("Validating integration points...")
    
    integration_checks = {}
    
    try:
        # Check if MCP client can be instantiated
        sys.path.insert(0, str(Path.cwd()))
        from src.mcp.server import MCPServerClient
        
        client = MCPServerClient()
        integration_checks["mcp_client_creation"] = True
        
        # Check if bridge can be instantiated
        from src.integration.bridge import IntegrationBridge, BridgeConfig
        
        config = BridgeConfig()
        bridge = IntegrationBridge(config)
        integration_checks["bridge_creation"] = True
        
        # Check if hooks manager can be instantiated
        from src.integration.hooks_manager import DevelopmentHooksCoordinator
        
        coordinator = DevelopmentHooksCoordinator()
        integration_checks["hooks_coordinator_creation"] = True
        
    except Exception as e:
        integration_checks["error"] = str(e)
    
    success = "error" not in integration_checks
    
    result.add_test("integration_points", success, integration_checks)
    
    logger.info(f"Integration points validation: {'PASS' if success else 'FAIL'}")

async def validate_api_structure(result: ValidationResult):
    """Validate API endpoint structure"""
    logger.info("Validating API structure...")
    
    api_checks = {}
    
    try:
        # Check if FastAPI app can be created
        from src.mcp.endpoints import app
        api_checks["fastapi_app_creation"] = hasattr(app, "router")
        
        # Check if required endpoints exist
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        required_endpoints = ["/health", "/api/swarm/init", "/api/swarm/status"]
        missing_endpoints = []
        
        for endpoint in required_endpoints:
            found = any(endpoint in route for route in routes)
            if not found:
                missing_endpoints.append(endpoint)
        
        api_checks["required_endpoints"] = len(missing_endpoints) == 0
        api_checks["missing_endpoints"] = missing_endpoints
        api_checks["available_routes"] = routes
        
    except Exception as e:
        api_checks["error"] = str(e)
    
    success = api_checks.get("required_endpoints", False) and "error" not in api_checks
    
    result.add_test("api_structure", success, api_checks)
    
    logger.info(f"API structure validation: {'PASS' if success else 'FAIL'}")

async def run_validation():
    """Run complete validation suite"""
    result = ValidationResult()
    
    print("🔍 Starting MCP Integration Validation...")
    print("=" * 60)
    
    # Run all validation tests
    await validate_file_structure(result)
    await validate_code_syntax(result)
    await validate_imports(result)
    await validate_dependencies(result)
    await validate_configuration(result)
    await validate_integration_points(result)
    await validate_api_structure(result)
    
    # Generate and display report
    summary = result.get_summary()
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']} ✅")
    print(f"Failed: {summary['failed_tests']} ❌")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Time: {summary['total_time']:.2f}s")
    
    print("\n📋 DETAILED RESULTS:")
    print("-" * 40)
    
    for test_name, test_data in result.tests.items():
        status = "✅ PASS" if test_data["success"] else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        if not test_data["success"] and test_data.get("error"):
            print(f"  Error: {test_data['error']}")
        
        if test_data.get("details"):
            details = test_data["details"]
            if isinstance(details, dict):
                for key, value in details.items():
                    if key.endswith("_errors") and value:
                        print(f"  {key}: {value}")
                    elif key.endswith("_files") and isinstance(value, list) and len(value) < 10:
                        print(f"  {key}: {len(value)} items")
    
    # Save detailed report
    report_file = Path(".swarm") / "validation_report.json"
    report_file.parent.mkdir(exist_ok=True)
    
    full_report = {
        "summary": summary,
        "tests": result.tests,
        "timestamp": time.time()
    }
    
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Final recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if summary["success_rate"] == 100:
        print("🎉 Perfect! All validations passed.")
        print("Your MCP integration is ready to use.")
        print("\nTo start the integration:")
        print("  python src/integration/run_mcp_integration.py dev")
        print("\nTo run tests:")
        print("  python src/integration/run_mcp_integration.py test")
    else:
        print("⚠️ Some validations failed. Please review the errors above.")
        
        if any("missing_deps" in str(test.get("details", {})) for test in result.tests.values()):
            print("📦 Install missing dependencies:")
            print("  pip install aiohttp fastapi uvicorn pydantic textual")
        
        if any("syntax" in test_name for test_name in result.tests.keys() if not result.tests[test_name]["success"]):
            print("🐛 Fix syntax errors in the reported files")
        
        if any("import" in test_name for test_name in result.tests.keys() if not result.tests[test_name]["success"]):
            print("🔗 Check import paths and module structure")
    
    print("\n" + "=" * 60)
    
    return summary["success_rate"] == 100

if __name__ == "__main__":
    success = asyncio.run(run_validation())
    sys.exit(0 if success else 1)