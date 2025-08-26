#!/usr/bin/env python3
"""
Integration Bridge for Claude-TUI and Claude-Flow
Connects all components and manages communication between systems
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3
from pathlib import Path
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from ..mcp.server import MCPServerClient, SwarmCoordinator, HooksIntegration
from ..monitoring.dashboard import MetricsCollector, SwarmMetrics

logger = logging.getLogger(__name__)

@dataclass
class BridgeConfig:
    """Configuration for integration bridge"""
    mcp_host: str = "localhost"
    mcp_port: int = 3000
    api_host: str = "localhost"  
    api_port: int = 8000
    tui_enabled: bool = True
    monitoring_enabled: bool = True
    hooks_enabled: bool = True
    auto_retry: bool = True
    retry_attempts: int = 3
    retry_delay: int = 5

@dataclass
class IntegrationEvent:
    """Event structure for integration bridge"""
    event_type: str
    source: str
    target: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False

class EventBus:
    """Event bus for inter-component communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[IntegrationEvent] = []
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def publish(self, event: IntegrationEvent):
        """Publish an event to all subscribers"""
        self.event_history.append(event)
        
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
        
        event.processed = True
    
    def get_recent_events(self, limit: int = 50) -> List[IntegrationEvent]:
        """Get recent events from history"""
        return self.event_history[-limit:]

class ComponentManager:
    """Manages lifecycle of integration components"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, str] = {}
        self.startup_order = [
            "mcp_client",
            "swarm_coordinator", 
            "hooks_integration",
            "metrics_collector",
            "api_server"
        ]
    
    async def initialize_component(self, component_name: str) -> bool:
        """Initialize a specific component"""
        try:
            if component_name == "mcp_client":
                client = MCPServerClient(self.config.mcp_host, self.config.mcp_port)
                await client.__aenter__()
                self.components[component_name] = client
                
            elif component_name == "swarm_coordinator":
                if "mcp_client" not in self.components:
                    raise ValueError("MCP client must be initialized first")
                coordinator = SwarmCoordinator(self.components["mcp_client"])
                self.components[component_name] = coordinator
                
            elif component_name == "hooks_integration":
                if "mcp_client" not in self.components:
                    raise ValueError("MCP client must be initialized first")
                hooks = HooksIntegration(self.components["mcp_client"])
                self.components[component_name] = hooks
                
            elif component_name == "metrics_collector":
                collector = MetricsCollector()
                self.components[component_name] = collector
                
            elif component_name == "api_server":
                # API server would be started separately
                self.components[component_name] = "started"
            
            self.component_status[component_name] = "running"
            logger.info(f"Component {component_name} initialized successfully")
            return True
            
        except Exception as e:
            self.component_status[component_name] = f"error: {str(e)}"
            logger.error(f"Failed to initialize {component_name}: {e}")
            return False
    
    async def start_all_components(self) -> bool:
        """Start all components in order"""
        success = True
        
        for component in self.startup_order:
            if not await self.initialize_component(component):
                success = False
                if not self.config.auto_retry:
                    break
                    
                # Retry logic
                for attempt in range(self.config.retry_attempts):
                    logger.info(f"Retrying {component} initialization (attempt {attempt + 1})")
                    await asyncio.sleep(self.config.retry_delay)
                    
                    if await self.initialize_component(component):
                        break
                else:
                    logger.error(f"Failed to initialize {component} after {self.config.retry_attempts} attempts")
                    success = False
        
        return success
    
    async def stop_all_components(self):
        """Stop all components"""
        for component_name in reversed(self.startup_order):
            if component_name in self.components:
                try:
                    component = self.components[component_name]
                    
                    if component_name == "mcp_client" and hasattr(component, "__aexit__"):
                        await component.__aexit__(None, None, None)
                    
                    del self.components[component_name]
                    self.component_status[component_name] = "stopped"
                    
                except Exception as e:
                    logger.error(f"Error stopping {component_name}: {e}")
    
    def get_component_status(self) -> Dict[str, str]:
        """Get status of all components"""
        return self.component_status.copy()

class IntegrationBridge:
    """Main integration bridge coordinating all systems"""
    
    def __init__(self, config: BridgeConfig = None):
        self.config = config or BridgeConfig()
        self.event_bus = EventBus()
        self.component_manager = ComponentManager(self.config)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.health_check_interval = 30  # seconds
        
        # Database for integration state
        self.db_path = Path.cwd() / ".swarm" / "integration.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _init_database(self):
        """Initialize integration database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS integration_events (
                    id INTEGER PRIMARY KEY,
                    event_type TEXT,
                    source TEXT,
                    target TEXT,
                    data TEXT,
                    timestamp TIMESTAMP,
                    processed BOOLEAN
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS component_health (
                    id INTEGER PRIMARY KEY,
                    component_name TEXT,
                    status TEXT,
                    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_count INTEGER DEFAULT 0
                )
            """)
    
    def _setup_event_handlers(self):
        """Setup event handlers for different event types"""
        self.event_bus.subscribe("mcp_connection_lost", self._handle_mcp_reconnect)
        self.event_bus.subscribe("swarm_status_change", self._handle_swarm_status)
        self.event_bus.subscribe("agent_error", self._handle_agent_error)
        self.event_bus.subscribe("performance_alert", self._handle_performance_alert)
    
    async def _handle_mcp_reconnect(self, event: IntegrationEvent):
        """Handle MCP connection loss"""
        logger.warning("MCP connection lost, attempting reconnect...")
        await self.component_manager.initialize_component("mcp_client")
    
    async def _handle_swarm_status(self, event: IntegrationEvent):
        """Handle swarm status changes"""
        status = event.data.get("status")
        logger.info(f"Swarm status changed to: {status}")
        
        # Store in database for monitoring
        self._store_event(event)
    
    async def _handle_agent_error(self, event: IntegrationEvent):
        """Handle agent errors"""
        agent_name = event.data.get("agent_name")
        error = event.data.get("error")
        logger.error(f"Agent {agent_name} error: {error}")
        
        # Could implement automatic agent restart logic here
    
    async def _handle_performance_alert(self, event: IntegrationEvent):
        """Handle performance alerts"""
        metric = event.data.get("metric")
        value = event.data.get("value")
        threshold = event.data.get("threshold")
        
        logger.warning(f"Performance alert: {metric} = {value} (threshold: {threshold})")
    
    def _store_event(self, event: IntegrationEvent):
        """Store event in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO integration_events 
                (event_type, source, target, data, timestamp, processed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.event_type,
                event.source,
                event.target,
                json.dumps(event.data),
                event.timestamp.isoformat(),
                event.processed
            ))
    
    async def start(self) -> bool:
        """Start the integration bridge"""
        logger.info("Starting integration bridge...")
        
        # Initialize all components
        success = await self.component_manager.start_all_components()
        
        if success:
            self.running = True
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            # Emit startup event
            await self.event_bus.publish(IntegrationEvent(
                event_type="bridge_started",
                source="integration_bridge",
                target="all",
                data={"status": "running", "components": list(self.component_manager.components.keys())}
            ))
            
            logger.info("Integration bridge started successfully")
            return True
        else:
            logger.error("Failed to start integration bridge")
            return False
    
    async def stop(self):
        """Stop the integration bridge"""
        logger.info("Stopping integration bridge...")
        
        self.running = False
        
        # Stop all components
        await self.component_manager.stop_all_components()
        
        # Cleanup executor
        self.executor.shutdown(wait=True)
        
        # Emit shutdown event
        await self.event_bus.publish(IntegrationEvent(
            event_type="bridge_stopped",
            source="integration_bridge",
            target="all",
            data={"status": "stopped"}
        ))
        
        logger.info("Integration bridge stopped")
    
    async def _health_monitor(self):
        """Monitor component health"""
        while self.running:
            try:
                status = self.component_manager.get_component_status()
                
                for component, comp_status in status.items():
                    if comp_status.startswith("error"):
                        await self.event_bus.publish(IntegrationEvent(
                            event_type="component_error",
                            source="health_monitor",
                            target=component,
                            data={"component": component, "status": comp_status}
                        ))
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    for component, comp_status in status.items():
                        conn.execute("""
                            INSERT OR REPLACE INTO component_health 
                            (component_name, status, last_check)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                        """, (component, comp_status))
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _metrics_collector(self):
        """Collect and process metrics"""
        while self.running:
            try:
                if "metrics_collector" in self.component_manager.components:
                    collector = self.component_manager.components["metrics_collector"]
                    metrics = await collector.collect_metrics()
                    
                    # Check for performance alerts
                    if metrics.cpu_usage > 80:
                        await self.event_bus.publish(IntegrationEvent(
                            event_type="performance_alert",
                            source="metrics_collector",
                            target="integration_bridge",
                            data={"metric": "cpu_usage", "value": metrics.cpu_usage, "threshold": 80}
                        ))
                    
                    if metrics.memory_usage > 80:
                        await self.event_bus.publish(IntegrationEvent(
                            event_type="performance_alert",
                            source="metrics_collector",
                            target="integration_bridge",
                            data={"metric": "memory_usage", "value": metrics.memory_usage, "threshold": 80}
                        ))
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(30)
    
    async def execute_swarm_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a swarm command through the bridge"""
        try:
            coordinator = self.component_manager.components.get("swarm_coordinator")
            if not coordinator:
                return {"error": "Swarm coordinator not available"}
            
            # Route command to appropriate method
            if command == "init":
                success = await coordinator.initialize_swarm(
                    topology=params.get("topology", "mesh"),
                    max_agents=params.get("max_agents", 5)
                )
                return {"success": success}
                
            elif command == "spawn":
                success = await coordinator.spawn_agent(
                    agent_type=params.get("agent_type"),
                    config=params.get("config", {})
                )
                return {"success": success}
                
            elif command == "orchestrate":
                task_id = await coordinator.orchestrate_task(
                    task_description=params.get("description"),
                    agents=params.get("agents", [])
                )
                return {"task_id": task_id}
                
            elif command == "status":
                status = await coordinator.get_swarm_status()
                return {"status": status}
            
            else:
                return {"error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"Error executing swarm command {command}: {e}")
            return {"error": str(e)}

async def main():
    """Main function to run the integration bridge"""
    config = BridgeConfig(
        mcp_host="localhost",
        mcp_port=3000,
        monitoring_enabled=True,
        hooks_enabled=True
    )
    
    bridge = IntegrationBridge(config)
    
    try:
        success = await bridge.start()
        if success:
            print("Integration bridge running. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while bridge.running:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        print("\nShutting down integration bridge...")
    finally:
        await bridge.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())