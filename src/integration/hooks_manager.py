#!/usr/bin/env python3
"""
Claude-Flow Hooks Integration Manager
Manages integration with claude-flow hooks system for coordinated operations
"""

import asyncio
import json
import subprocess
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class HookEvent:
    """Hook event data structure"""
    hook_type: str
    operation: str
    params: Dict[str, Any]
    timestamp: datetime
    success: Optional[bool] = None
    output: Optional[str] = None
    error: Optional[str] = None

class HooksManager:
    """Manages claude-flow hooks integration"""
    
    def __init__(self, swarm_id: str = None):
        self.swarm_id = swarm_id or f"swarm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.hooks_enabled = True
        self.session_active = False
        self.db_path = Path.cwd() / ".swarm" / "hooks.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # Hook callbacks
        self.hook_callbacks: Dict[str, List[Callable]] = {}
    
    def _init_database(self):
        """Initialize hooks database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hook_events (
                    id INTEGER PRIMARY KEY,
                    hook_type TEXT,
                    operation TEXT,
                    params TEXT,
                    success BOOLEAN,
                    output TEXT,
                    error TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT UNIQUE,
                    swarm_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    metrics TEXT
                )
            """)
    
    def add_hook_callback(self, hook_type: str, callback: Callable):
        """Add callback for hook events"""
        if hook_type not in self.hook_callbacks:
            self.hook_callbacks[hook_type] = []
        self.hook_callbacks[hook_type].append(callback)
    
    async def execute_hook(self, hook_type: str, operation: str = None, **params) -> HookEvent:
        """Execute a claude-flow hook"""
        event = HookEvent(
            hook_type=hook_type,
            operation=operation or "default",
            params=params,
            timestamp=datetime.now()
        )
        
        if not self.hooks_enabled:
            event.success = False
            event.error = "Hooks disabled"
            return event
        
        try:
            # Build hook command
            cmd = self._build_hook_command(hook_type, operation, params)
            
            # Execute hook
            result = await self._run_hook_command(cmd)
            
            event.success = result["returncode"] == 0
            event.output = result["stdout"]
            event.error = result["stderr"] if result["returncode"] != 0 else None
            
            # Store in database
            self._store_hook_event(event)
            
            # Call registered callbacks
            await self._call_hook_callbacks(hook_type, event)
            
            logger.info(f"Hook {hook_type} executed: success={event.success}")
            
        except Exception as e:
            event.success = False
            event.error = str(e)
            logger.error(f"Hook {hook_type} failed: {e}")
        
        return event
    
    def _build_hook_command(self, hook_type: str, operation: str, params: Dict[str, Any]) -> List[str]:
        """Build the npx claude-flow hooks command"""
        cmd = ["npx", "claude-flow@alpha", "hooks", hook_type]
        
        # Add operation-specific parameters
        if operation and operation != "default":
            cmd.extend(["--operation", operation])
        
        # Add parameters as command line arguments
        for key, value in params.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            elif value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    async def _run_hook_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run hook command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8") if stdout else "",
                "stderr": stderr.decode("utf-8") if stderr else ""
            }
            
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    async def _call_hook_callbacks(self, hook_type: str, event: HookEvent):
        """Call registered callbacks for hook events"""
        if hook_type in self.hook_callbacks:
            for callback in self.hook_callbacks[hook_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Hook callback error: {e}")
    
    def _store_hook_event(self, event: HookEvent):
        """Store hook event in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO hook_events 
                (hook_type, operation, params, success, output, error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.hook_type,
                event.operation,
                json.dumps(event.params),
                event.success,
                event.output,
                event.error
            ))
    
    async def start_session(self) -> bool:
        """Start a coordinated session"""
        try:
            event = await self.execute_hook(
                "session-restore",
                session_id=self.swarm_id
            )
            
            if event.success:
                self.session_active = True
                
                # Store session start
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO session_state 
                        (session_id, swarm_id, start_time, status)
                        VALUES (?, ?, ?, ?)
                    """, (self.swarm_id, self.swarm_id, datetime.now(), "active"))
                
                logger.info(f"Session {self.swarm_id} started")
                return True
            else:
                logger.error(f"Failed to start session: {event.error}")
                return False
                
        except Exception as e:
            logger.error(f"Session start error: {e}")
            return False
    
    async def end_session(self, export_metrics: bool = True) -> bool:
        """End the coordinated session"""
        try:
            event = await self.execute_hook(
                "session-end",
                export_metrics=export_metrics
            )
            
            if event.success:
                self.session_active = False
                
                # Update session end
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE session_state 
                        SET end_time = ?, status = ?, metrics = ?
                        WHERE session_id = ?
                    """, (datetime.now(), "completed", event.output, self.swarm_id))
                
                logger.info(f"Session {self.swarm_id} ended")
                return True
            else:
                logger.error(f"Failed to end session: {event.error}")
                return False
                
        except Exception as e:
            logger.error(f"Session end error: {e}")
            return False
    
    async def pre_task_hook(self, description: str, task_id: str = None) -> bool:
        """Execute pre-task hook"""
        event = await self.execute_hook(
            "pre-task",
            description=description,
            task_id=task_id
        )
        return event.success
    
    async def post_task_hook(self, task_id: str, success: bool = True) -> bool:
        """Execute post-task hook"""
        event = await self.execute_hook(
            "post-task",
            task_id=task_id,
            success=success
        )
        return event.success
    
    async def pre_edit_hook(self, file_path: str, auto_assign_agents: bool = True, load_context: bool = True) -> bool:
        """Execute pre-edit hook"""
        event = await self.execute_hook(
            "pre-edit",
            file=file_path,
            auto_assign_agents=auto_assign_agents,
            load_context=load_context
        )
        return event.success
    
    async def post_edit_hook(self, file_path: str, memory_key: str = None, format_code: bool = True, train_neural: bool = True) -> bool:
        """Execute post-edit hook"""
        params = {
            "file": file_path,
            "format": format_code,
            "train_neural": train_neural
        }
        
        if memory_key:
            params["memory_key"] = memory_key
            params["update_memory"] = True
        
        event = await self.execute_hook("post-edit", **params)
        return event.success
    
    async def notify_hook(self, message: str, level: str = "info") -> bool:
        """Send notification through hooks"""
        event = await self.execute_hook(
            "notify",
            message=message,
            level=level
        )
        return event.success

class DevelopmentHooksCoordinator:
    """Coordinates hooks for development swarm operations"""
    
    def __init__(self, swarm_id: str = None):
        self.hooks_manager = HooksManager(swarm_id)
        self.operation_stack: List[str] = []
        
        # Register callbacks
        self.hooks_manager.add_hook_callback("pre-task", self._on_pre_task)
        self.hooks_manager.add_hook_callback("post-task", self._on_post_task)
        self.hooks_manager.add_hook_callback("post-edit", self._on_post_edit)
    
    async def _on_pre_task(self, event: HookEvent):
        """Handle pre-task events"""
        task_id = event.params.get("task_id")
        if task_id:
            self.operation_stack.append(task_id)
            logger.info(f"Task {task_id} started")
    
    async def _on_post_task(self, event: HookEvent):
        """Handle post-task events"""
        task_id = event.params.get("task_id")
        if task_id in self.operation_stack:
            self.operation_stack.remove(task_id)
            logger.info(f"Task {task_id} completed")
    
    async def _on_post_edit(self, event: HookEvent):
        """Handle post-edit events"""
        file_path = event.params.get("file")
        if file_path:
            logger.info(f"File {file_path} edited and processed")
    
    async def coordinate_development_task(self, description: str, file_operations: List[Dict[str, str]] = None) -> bool:
        """Coordinate a complete development task with hooks"""
        task_id = f"dev-task-{datetime.now().timestamp()}"
        
        try:
            # Start session if not active
            if not self.hooks_manager.session_active:
                await self.hooks_manager.start_session()
            
            # Pre-task hook
            success = await self.hooks_manager.pre_task_hook(description, task_id)
            if not success:
                return False
            
            # Process file operations with hooks
            if file_operations:
                for op in file_operations:
                    file_path = op.get("file")
                    operation = op.get("operation", "edit")
                    
                    if file_path:
                        # Pre-edit hook
                        await self.hooks_manager.pre_edit_hook(file_path)
                        
                        # File operation would happen here
                        # (handled by the calling code)
                        
                        # Post-edit hook
                        memory_key = f"swarm/dev/{task_id}/{Path(file_path).stem}"
                        await self.hooks_manager.post_edit_hook(file_path, memory_key)
            
            # Post-task hook
            await self.hooks_manager.post_task_hook(task_id, True)
            
            # Notify completion
            await self.hooks_manager.notify_hook(f"Development task completed: {description}", "success")
            
            return True
            
        except Exception as e:
            logger.error(f"Development task coordination failed: {e}")
            
            # Notify failure
            await self.hooks_manager.notify_hook(f"Development task failed: {str(e)}", "error")
            
            # Post-task hook with failure
            await self.hooks_manager.post_task_hook(task_id, False)
            
            return False
    
    async def finalize_session(self):
        """Finalize the development session"""
        if self.hooks_manager.session_active:
            await self.hooks_manager.end_session(export_metrics=True)
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Get current operation status"""
        return {
            "session_active": self.hooks_manager.session_active,
            "active_operations": len(self.operation_stack),
            "operations": self.operation_stack.copy()
        }

async def test_hooks_integration():
    """Test hooks integration"""
    coordinator = DevelopmentHooksCoordinator()
    
    # Test development task coordination
    success = await coordinator.coordinate_development_task(
        "Implement MCP server integration",
        [
            {"file": "src/mcp/server.py", "operation": "create"},
            {"file": "src/mcp/endpoints.py", "operation": "create"},
            {"file": "src/integration/bridge.py", "operation": "create"}
        ]
    )
    
    print(f"Development task coordination: {'Success' if success else 'Failed'}")
    
    # Check status
    status = coordinator.get_operation_status()
    print(f"Operation status: {status}")
    
    # Finalize session
    await coordinator.finalize_session()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_hooks_integration())