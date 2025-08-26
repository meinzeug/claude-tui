#!/usr/bin/env python3
"""
MCP Integration Runner
Main entry point for running the complete MCP integration system
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .startup_manager import StartupManager, QuickStart
from .test_integration import IntegrationTester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.cwd() / ".swarm" / "mcp_integration.log")
    ]
)

logger = logging.getLogger(__name__)

async def run_development_mode():
    """Run in development mode"""
    logger.info("Starting MCP integration in development mode...")
    
    manager = await QuickStart.development_mode()
    
    if manager.startup_complete:
        logger.info("✅ Development mode started successfully!")
        logger.info(f"📡 MCP Server: http://{manager.config.mcp_host}:{manager.config.mcp_port}")
        logger.info(f"🌐 API Server: http://{manager.config.api_host}:{manager.config.api_port}")
        logger.info("📊 Monitoring dashboard available")
        logger.info("🔗 Hooks integration active")
        logger.info("\nPress Ctrl+C to stop...")
        
        try:
            while manager.startup_complete:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested...")
        finally:
            await manager.shutdown()
            logger.info("✅ Development mode stopped")
    else:
        logger.error("❌ Failed to start development mode")
        sys.exit(1)

async def run_production_mode():
    """Run in production mode"""
    logger.info("Starting MCP integration in production mode...")
    
    manager = await QuickStart.production_mode()
    
    if manager.startup_complete:
        logger.info("✅ Production mode started successfully!")
        logger.info("🚀 All services running in production configuration")
        logger.info("📊 Monitoring and metrics collection active")
        logger.info("🔗 Hooks integration with enhanced reliability")
        logger.info("\nPress Ctrl+C to stop...")
        
        try:
            while manager.startup_complete:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested...")
        finally:
            await manager.shutdown()
            logger.info("✅ Production mode stopped")
    else:
        logger.error("❌ Failed to start production mode")
        sys.exit(1)

async def run_tests():
    """Run integration tests"""
    logger.info("Running MCP integration tests...")
    
    tester = IntegrationTester()
    report = await tester.run_all_tests()
    
    # Print summary
    summary = report["test_summary"]
    if summary["passed_tests"] == summary["total_tests"]:
        logger.info(f"✅ All {summary['total_tests']} tests passed!")
    else:
        logger.warning(f"⚠️ {summary['failed_tests']}/{summary['total_tests']} tests failed")
    
    return summary["success_rate"] == 100.0

async def run_monitoring_only():
    """Run only monitoring dashboard"""
    try:
        from ..monitoring.dashboard import run_dashboard
        logger.info("Starting monitoring dashboard...")
        run_dashboard()
    except ImportError:
        logger.error("Monitoring dashboard not available")
    except KeyboardInterrupt:
        logger.info("Monitoring dashboard stopped")

async def validate_installation():
    """Validate that all required components are installed"""
    logger.info("Validating MCP integration installation...")
    
    validation_results = {}
    
    # Check Python dependencies
    try:
        import aiohttp
        import fastapi
        import uvicorn
        import textual
        validation_results["python_deps"] = True
    except ImportError as e:
        validation_results["python_deps"] = False
        validation_results["python_deps_error"] = str(e)
    
    # Check claude-flow installation
    try:
        process = await asyncio.create_subprocess_exec(
            "npx", "claude-flow@alpha", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        validation_results["claude_flow"] = process.returncode == 0
        validation_results["claude_flow_version"] = stdout.decode().strip()
    except Exception as e:
        validation_results["claude_flow"] = False
        validation_results["claude_flow_error"] = str(e)
    
    # Check directory structure
    required_dirs = [".swarm", "src/mcp", "src/monitoring", "src/integration"]
    for dir_path in required_dirs:
        path = Path.cwd() / dir_path
        validation_results[f"dir_{dir_path.replace('/', '_')}"] = path.exists()
    
    # Print validation results
    print("\n" + "="*50)
    print("MCP INTEGRATION VALIDATION REPORT")
    print("="*50)
    
    all_valid = True
    
    print(f"Python Dependencies: {'✅' if validation_results['python_deps'] else '❌'}")
    if not validation_results['python_deps']:
        print(f"  Error: {validation_results.get('python_deps_error', 'Unknown')}")
        all_valid = False
    
    print(f"Claude-Flow: {'✅' if validation_results['claude_flow'] else '❌'}")
    if validation_results['claude_flow']:
        print(f"  Version: {validation_results.get('claude_flow_version', 'Unknown')}")
    else:
        print(f"  Error: {validation_results.get('claude_flow_error', 'Unknown')}")
        all_valid = False
    
    print("Directory Structure:")
    for key, value in validation_results.items():
        if key.startswith("dir_"):
            dir_name = key.replace("dir_", "").replace("_", "/")
            print(f"  {dir_name}: {'✅' if value else '❌'}")
            if not value:
                all_valid = False
    
    print("\n" + "="*50)
    
    if all_valid:
        print("✅ Installation validation passed!")
        logger.info("All components validated successfully")
    else:
        print("❌ Installation validation failed!")
        logger.error("Some components failed validation")
        
        print("\nTo fix installation issues:")
        print("1. Install Python dependencies: pip install -r requirements.txt")
        print("2. Install claude-flow: npm install -g claude-flow@alpha")
        print("3. Run setup script: ./setup-claude-flow.sh")
    
    return all_valid

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Claude-TUI MCP Integration")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Development mode
    dev_parser = subparsers.add_parser("dev", help="Run in development mode")
    
    # Production mode  
    prod_parser = subparsers.add_parser("prod", help="Run in production mode")
    
    # Test mode
    test_parser = subparsers.add_parser("test", help="Run integration tests")
    
    # Monitoring only
    monitor_parser = subparsers.add_parser("monitor", help="Run monitoring dashboard only")
    
    # Validation
    validate_parser = subparsers.add_parser("validate", help="Validate installation")
    
    # Status
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Ensure .swarm directory exists
    (Path.cwd() / ".swarm").mkdir(exist_ok=True)
    
    try:
        if args.command == "dev":
            asyncio.run(run_development_mode())
        
        elif args.command == "prod":
            asyncio.run(run_production_mode())
        
        elif args.command == "test":
            success = asyncio.run(run_tests())
            sys.exit(0 if success else 1)
        
        elif args.command == "monitor":
            asyncio.run(run_monitoring_only())
        
        elif args.command == "validate":
            success = asyncio.run(validate_installation())
            sys.exit(0 if success else 1)
        
        elif args.command == "status":
            # Show status of running components
            print("Checking component status...")
            # This would check if MCP server is running, etc.
            print("Status check not yet implemented")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()