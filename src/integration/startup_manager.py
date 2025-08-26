#!/usr/bin/env python3
"""
Startup Manager for MCP Integration
Manages the startup sequence and initialization of all MCP and claude-flow components
"""

import asyncio
import json
import logging
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import subprocess
import signal
import os

from .bridge import IntegrationBridge, BridgeConfig
from .hooks_manager import DevelopmentHooksCoordinator
from .tui_connector import MCPConnectionManager
from ..monitoring.dashboard import MetricsCollector

logger = logging.getLogger(__name__)

class StartupManager:
    """Manages the complete startup sequence for MCP integration"""
    
    def __init__(self):
        self.config = self._load_config()
        self.components: Dict[str, Any] = {}
        self.startup_sequence = [
            ("mcp_server", self._start_mcp_server),
            ("integration_bridge", self._start_integration_bridge),
            ("hooks_coordinator", self._start_hooks_coordinator),
            ("metrics_collector", self._start_metrics_collector),
            ("api_server", self._start_api_server),
            ("monitoring_dashboard", self._start_monitoring_dashboard)
        ]
        self.shutdown_handlers = []
        self.startup_complete = False
    
    def _load_config(self) -> BridgeConfig:
        """Load configuration from file or use defaults"""
        config_file = Path.cwd() / "claude-flow.config.json"
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                
                return BridgeConfig(
                    mcp_host=config_data.get("mcp_host", "localhost"),
                    mcp_port=config_data.get("mcp_port", 3000),
                    api_host=config_data.get("api_host", "localhost"),
                    api_port=config_data.get("api_port", 8000),
                    tui_enabled=config_data.get("tui_enabled", True),
                    monitoring_enabled=config_data.get("monitoring_enabled", True),
                    hooks_enabled=config_data.get("hooks_enabled", True),
                    auto_retry=config_data.get("auto_retry", True),
                    retry_attempts=config_data.get("retry_attempts", 3),
                    retry_delay=config_data.get("retry_delay", 5)
                )
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")
        
        return BridgeConfig()
    
    async def _start_mcp_server(self) -> bool:
        """Start the MCP server daemon"""
        try:
            # Check if MCP server is already running
            if await self._check_mcp_server():
                logger.info("MCP server already running")
                return True
            
            # Start MCP server daemon
            logger.info("Starting MCP server daemon...")
            
            cmd = [
                "npx", "claude-flow@alpha", "mcp", "start",
                "--daemon",
                "--port", str(self.config.mcp_port),
                "--host", self.config.mcp_host
            ]
            
            # Start in background
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            # Wait a moment for startup
            await asyncio.sleep(3)
            
            # Check if server started successfully
            if await self._check_mcp_server():
                self.components["mcp_server"] = process
                logger.info("MCP server started successfully")
                return True
            else:
                logger.error("MCP server failed to start")
                return False
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    async def _check_mcp_server(self) -> bool:
        """Check if MCP server is responding"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{self.config.mcp_host}:{self.config.mcp_port}/health", timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def _start_integration_bridge(self) -> bool:
        """Start the integration bridge"""
        try:
            bridge = IntegrationBridge(self.config)
            success = await bridge.start()
            
            if success:
                self.components["integration_bridge"] = bridge
                logger.info("Integration bridge started successfully")
                return True
            else:
                logger.error("Integration bridge failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start integration bridge: {e}")
            return False
    
    async def _start_hooks_coordinator(self) -> bool:
        """Start the hooks coordinator"""
        try:
            if not self.config.hooks_enabled:
                logger.info("Hooks disabled, skipping coordinator")
                return True
            
            coordinator = DevelopmentHooksCoordinator()
            await coordinator.hooks_manager.start_session()
            
            self.components["hooks_coordinator"] = coordinator
            logger.info("Hooks coordinator started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start hooks coordinator: {e}")
            return False
    
    async def _start_metrics_collector(self) -> bool:
        """Start the metrics collector"""
        try:
            if not self.config.monitoring_enabled:
                logger.info("Monitoring disabled, skipping metrics collector")
                return True
            
            collector = MetricsCollector()
            self.components["metrics_collector"] = collector
            
            # Start background collection task
            asyncio.create_task(self._metrics_collection_task(collector))
            
            logger.info("Metrics collector started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start metrics collector: {e}")
            return False
    
    async def _start_api_server(self) -> bool:
        """Start the API server"""
        try:
            # Import and start FastAPI server
            from ..mcp.endpoints import app
            import uvicorn
            
            # Start server in background
            server_config = uvicorn.Config(
                app,
                host=self.config.api_host,
                port=self.config.api_port,
                log_level="info"
            )
            
            server = uvicorn.Server(server_config)
            
            # Start server task
            server_task = asyncio.create_task(server.serve())
            
            self.components["api_server"] = {
                "server": server,
                "task": server_task
            }
            
            logger.info(f"API server started on {self.config.api_host}:{self.config.api_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    async def _start_monitoring_dashboard(self) -> bool:
        """Start the monitoring dashboard"""
        try:
            if not self.config.monitoring_enabled:
                logger.info("Monitoring disabled, skipping dashboard")
                return True
            
            # Dashboard would be started separately or embedded in TUI
            self.components["monitoring_dashboard"] = "available"
            logger.info("Monitoring dashboard available")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring dashboard: {e}")
            return False
    
    async def _metrics_collection_task(self, collector: MetricsCollector):
        """Background task for metrics collection"""
        while self.startup_complete and "metrics_collector" in self.components:
            try:
                await collector.collect_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def startup(self) -> bool:
        """Execute complete startup sequence"""
        logger.info("Starting MCP integration system...")
        
        # Create necessary directories
        (Path.cwd() / ".swarm").mkdir(exist_ok=True)
        
        # Execute startup sequence
        for component_name, start_func in self.startup_sequence:
            logger.info(f"Starting {component_name}...")
            
            success = await start_func()
            
            if not success:
                if self.config.auto_retry:
                    # Retry logic
                    for attempt in range(self.config.retry_attempts):
                        logger.info(f"Retrying {component_name} (attempt {attempt + 1})")
                        await asyncio.sleep(self.config.retry_delay)
                        
                        success = await start_func()
                        if success:
                            break
                    
                    if not success:
                        logger.error(f"Failed to start {component_name} after {self.config.retry_attempts} attempts")
                        await self.shutdown()
                        return False
                else:
                    logger.error(f"Failed to start {component_name}")
                    await self.shutdown()
                    return False
        
        self.startup_complete = True
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("MCP integration system started successfully")
        
        # Log startup summary
        await self._log_startup_summary()
        
        return True
    
    async def shutdown(self):
        """Execute shutdown sequence"""
        logger.info("Shutting down MCP integration system...")
        
        self.startup_complete = False
        
        # Shutdown in reverse order
        shutdown_order = list(reversed([name for name, _ in self.startup_sequence]))
        
        for component_name in shutdown_order:
            if component_name in self.components:
                try:
                    await self._shutdown_component(component_name)
                except Exception as e:
                    logger.error(f"Error shutting down {component_name}: {e}")
        
        # Execute custom shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")
        
        logger.info("MCP integration system shut down")
    
    async def _shutdown_component(self, component_name: str):
        """Shutdown a specific component"""
        component = self.components.get(component_name)
        if not component:
            return
        
        if component_name == "integration_bridge":
            await component.stop()
        
        elif component_name == "hooks_coordinator":
            await component.finalize_session()
        
        elif component_name == "api_server":
            server_info = component
            if "server" in server_info:
                server_info["server"].should_exit = True
            if "task" in server_info and not server_info["task"].done():
                server_info["task"].cancel()
        
        elif component_name == "mcp_server":
            if hasattr(component, "terminate"):
                component.terminate()
                await component.wait()
        
        del self.components[component_name]
        logger.info(f"Component {component_name} shut down")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _log_startup_summary(self):
        """Log startup summary"""
        summary = {
            "startup_time": datetime.now().isoformat(),
            "components": list(self.components.keys()),
            "config": {
                "mcp_host": self.config.mcp_host,
                "mcp_port": self.config.mcp_port,
                "api_host": self.config.api_host,
                "api_port": self.config.api_port,
                "monitoring_enabled": self.config.monitoring_enabled,
                "hooks_enabled": self.config.hooks_enabled
            }
        }
        
        # Save summary to file
        summary_file = Path.cwd() / ".swarm" / "startup_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Log to hooks if available
        if "hooks_coordinator" in self.components:
            coordinator = self.components["hooks_coordinator"]
            await coordinator.hooks_manager.notify_hook(
                f"MCP integration system started with {len(self.components)} components",
                "success"
            )
    
    def add_shutdown_handler(self, handler: callable):
        """Add custom shutdown handler"""
        self.shutdown_handlers.append(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "startup_complete": self.startup_complete,
            "components": {
                name: "running" if name in self.components else "stopped"
                for name, _ in self.startup_sequence
            },
            "config": self.config.__dict__
        }

class QuickStart:
    """Quick start utility for common scenarios"""
    
    @staticmethod
    async def development_mode() -> StartupManager:
        """Start in development mode with all features enabled"""
        manager = StartupManager()
        success = await manager.startup()
        
        if success:
            logger.info("Development mode started successfully")
            logger.info(f"API available at: http://{manager.config.api_host}:{manager.config.api_port}")
            logger.info(f"MCP server at: http://{manager.config.mcp_host}:{manager.config.mcp_port}")
        else:
            logger.error("Failed to start development mode")
        
        return manager
    
    @staticmethod
    async def production_mode() -> StartupManager:
        """Start in production mode with optimized settings"""
        # Override config for production
        config = BridgeConfig(
            mcp_host="0.0.0.0",
            mcp_port=3000,
            api_host="0.0.0.0", 
            api_port=8000,
            monitoring_enabled=True,
            hooks_enabled=True,
            auto_retry=True,
            retry_attempts=5,
            retry_delay=10
        )
        
        manager = StartupManager()
        manager.config = config
        
        success = await manager.startup()
        
        if success:
            logger.info("Production mode started successfully")
        else:
            logger.error("Failed to start production mode")
        
        return manager

async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "production":
        manager = await QuickStart.production_mode()
    else:
        manager = await QuickStart.development_mode()
    
    if manager.startup_complete:
        try:
            # Keep running until interrupted
            while manager.startup_complete:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await manager.shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())