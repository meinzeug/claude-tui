#!/usr/bin/env python3
"""
TUI Backend Bridge - Advanced Terminal UI Backend Integration

Provides comprehensive bridge between Terminal UI (Textual) and backend services:
- Real-time data synchronization
- WebSocket communication management
- Event-driven UI updates
- Performance monitoring integration
- Claude Flow coordination
- Hive Mind agent communication
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

# Third-party imports
from textual.message import Message
from textual.widgets import Widget
from fastapi import WebSocket, WebSocketDisconnect
import websockets
from pydantic import BaseModel

# Internal imports
from .core_services import ServiceOrchestrator, get_service_orchestrator
from ..api.v1.websocket import WebSocketEventType, ConnectionManager, WebSocketMessage
from ..claude_tiu.integrations.claude_flow_client import ClaudeFlowClient
from ..core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class TUIEventType(str, Enum):
    """TUI-specific event types."""
    # UI State events
    WIDGET_FOCUS_CHANGED = "widget_focus_changed"
    SCREEN_CHANGED = "screen_changed"
    LAYOUT_UPDATED = "layout_updated"
    
    # Data events
    PROJECT_LOADED = "project_loaded"
    TASK_SELECTED = "task_selected"
    FILE_OPENED = "file_opened"
    
    # Performance events
    UI_LAG_DETECTED = "ui_lag_detected"
    MEMORY_WARNING = "memory_warning"
    
    # Integration events
    BACKEND_CONNECTION_STATUS = "backend_connection_status"
    SYNC_STATUS_CHANGED = "sync_status_changed"


@dataclass
class TUIState:
    """Current TUI application state."""
    current_screen: str = "main"
    focused_widget: Optional[str] = None
    active_project: Optional[str] = None
    selected_task: Optional[str] = None
    open_files: Set[str] = field(default_factory=set)
    ui_theme: str = "dark"
    layout_mode: str = "standard"
    
    # Performance tracking
    last_update: datetime = field(default_factory=datetime.now)
    refresh_rate: float = 60.0
    lag_threshold: float = 0.1  # 100ms
    
    # Connection status
    backend_connected: bool = False
    websocket_connected: bool = False
    claude_flow_connected: bool = False


@dataclass
class TUIEvent:
    """TUI event data structure."""
    event_type: TUIEventType
    timestamp: datetime
    data: Dict[str, Any]
    widget_id: Optional[str] = None
    screen_name: Optional[str] = None
    user_id: Optional[str] = None


class TUIMessage(Message):
    """Custom Textual message for TUI events."""
    
    def __init__(self, event: TUIEvent):
        self.event = event
        super().__init__()


class BackendSyncStatus(BaseModel):
    """Backend synchronization status."""
    connected: bool
    last_sync: Optional[datetime] = None
    pending_updates: int = 0
    sync_errors: List[str] = []
    latency_ms: float = 0.0


class TUIBackendBridge:
    """
    Advanced bridge between TUI and backend services.
    
    Features:
    - Real-time bidirectional communication
    - Event routing and filtering
    - State synchronization
    - Performance monitoring
    - Automatic reconnection
    - Batch update optimization
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize TUI backend bridge."""
        self.config_manager = config_manager
        
        # Service orchestrator
        self.orchestrator: Optional[ServiceOrchestrator] = None
        
        # TUI state management
        self.tui_state = TUIState()
        self.event_handlers: Dict[TUIEventType, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # WebSocket management
        self.websocket_url = "ws://localhost:8000/api/v1/websocket/ws"
        self.websocket_client: Optional[websockets.WebSocketClientProtocol] = None
        self.connection_manager = ConnectionManager()
        
        # Synchronization
        self.sync_status = BackendSyncStatus(connected=False)
        self.sync_tasks: Set[asyncio.Task] = set()
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Any] = {}
        self.update_timestamps: List[datetime] = []
        
        # Event batching
        self.batch_events: List[TUIEvent] = []
        self.batch_timeout = 0.1  # 100ms batch window
        self.last_batch_time = datetime.now()
        
        logger.info("TUI Backend Bridge initialized")
    
    async def initialize(self) -> None:
        """Initialize the backend bridge."""
        logger.info("Initializing TUI Backend Bridge...")
        
        try:
            # Get service orchestrator
            self.orchestrator = get_service_orchestrator()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Connect to WebSocket
            await self._connect_websocket()
            
            # Initialize Claude Flow integration
            await self._initialize_claude_flow()
            
            self.sync_status.connected = True
            logger.info("TUI Backend Bridge initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TUI Backend Bridge: {e}")
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        tasks = [
            asyncio.create_task(self._event_processing_loop()),
            asyncio.create_task(self._sync_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._batch_processing_loop()),
            asyncio.create_task(self._websocket_heartbeat_loop())
        ]
        
        self.sync_tasks.update(tasks)
        logger.info(f"Started {len(tasks)} background tasks")
    
    async def _connect_websocket(self) -> None:
        """Connect to backend WebSocket."""
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to WebSocket (attempt {attempt + 1}/{max_retries})...")
                
                self.websocket_client = await websockets.connect(
                    self.websocket_url,
                    timeout=10,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                self.tui_state.websocket_connected = True
                logger.info("WebSocket connection established")
                
                # Start WebSocket message handler
                asyncio.create_task(self._websocket_message_handler())
                break
                
            except Exception as e:
                logger.warning(f"WebSocket connection attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Failed to connect to WebSocket after all retries")
                    self.tui_state.websocket_connected = False
    
    async def _initialize_claude_flow(self) -> None:
        """Initialize Claude Flow integration."""
        try:
            if self.orchestrator:
                claude_flow = self.orchestrator.get_claude_flow_service()
                if claude_flow:
                    self.tui_state.claude_flow_connected = True
                    logger.info("Claude Flow integration initialized")
                else:
                    logger.warning("Claude Flow service not available")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude Flow integration: {e}")
    
    async def _event_processing_loop(self) -> None:
        """Process TUI events from queue."""
        logger.info("Starting event processing loop")
        
        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                await self._process_event(event)
                
                # Mark task done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                # Timeout is normal, continue processing
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_event(self, event: TUIEvent) -> None:
        """Process individual TUI event."""
        try:
            # Update performance metrics
            self._update_performance_metrics(event)
            
            # Handle event based on type
            if event.event_type == TUIEventType.WIDGET_FOCUS_CHANGED:
                await self._handle_focus_change(event)
            elif event.event_type == TUIEventType.PROJECT_LOADED:
                await self._handle_project_loaded(event)
            elif event.event_type == TUIEventType.TASK_SELECTED:
                await self._handle_task_selected(event)
            elif event.event_type == TUIEventType.UI_LAG_DETECTED:
                await self._handle_ui_lag(event)
            
            # Call registered event handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
            
            # Forward to backend if connected
            if self.sync_status.connected:
                await self._forward_to_backend(event)
                
        except Exception as e:
            logger.error(f"Failed to process event {event.event_type}: {e}")
    
    async def _handle_focus_change(self, event: TUIEvent) -> None:
        """Handle widget focus change."""
        widget_id = event.data.get('widget_id')
        self.tui_state.focused_widget = widget_id
        
        # Update backend with focus context
        if widget_id and self.orchestrator:
            cache_service = self.orchestrator.get_cache_service()
            if cache_service:
                await cache_service.set(
                    f"tui:focus:{self.tui_state.current_screen}",
                    widget_id,
                    ttl=300
                )
    
    async def _handle_project_loaded(self, event: TUIEvent) -> None:
        """Handle project loaded event."""
        project_path = event.data.get('project_path')
        self.tui_state.active_project = project_path
        
        if project_path and self.orchestrator:
            # Cache project context
            cache_service = self.orchestrator.get_cache_service()
            if cache_service:
                project_data = {
                    'path': project_path,
                    'loaded_at': datetime.now().isoformat(),
                    'screen': self.tui_state.current_screen
                }
                await cache_service.set(
                    f"tui:project:active",
                    project_data,
                    ttl=3600
                )
    
    async def _handle_task_selected(self, event: TUIEvent) -> None:
        """Handle task selection event."""
        task_id = event.data.get('task_id')
        self.tui_state.selected_task = task_id
        
        # Trigger Claude Flow task coordination if available
        if task_id and self.tui_state.claude_flow_connected:
            await self._coordinate_task_with_claude_flow(task_id, event.data)
    
    async def _coordinate_task_with_claude_flow(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Coordinate task with Claude Flow."""
        try:
            claude_flow = self.orchestrator.get_claude_flow_service()
            if claude_flow:
                # Create task orchestration request
                from ..claude_tiu.models.ai_models import TaskOrchestrationRequest
                
                request = TaskOrchestrationRequest(
                    description=task_data.get('description', f'TUI Task {task_id}'),
                    priority='medium',
                    requirements=task_data.get('requirements', []),
                    context={'tui_task_id': task_id, 'screen': self.tui_state.current_screen}
                )
                
                # This would be implemented based on active swarms
                logger.info(f"Task {task_id} ready for Claude Flow coordination")
                
        except Exception as e:
            logger.error(f"Failed to coordinate task with Claude Flow: {e}")
    
    async def _handle_ui_lag(self, event: TUIEvent) -> None:
        """Handle UI lag detection."""
        lag_time = event.data.get('lag_time', 0)
        
        if lag_time > self.tui_state.lag_threshold:
            # Log performance issue
            logger.warning(f"UI lag detected: {lag_time:.3f}s")
            
            # Send alert to monitoring
            if self.orchestrator:
                cache_service = self.orchestrator.get_cache_service()
                if cache_service:
                    alert_data = {
                        'type': 'ui_lag',
                        'lag_time': lag_time,
                        'threshold': self.tui_state.lag_threshold,
                        'screen': self.tui_state.current_screen,
                        'timestamp': datetime.now().isoformat()
                    }
                    await cache_service.set(
                        f"alerts:ui_lag:{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        alert_data,
                        ttl=3600
                    )
    
    async def _forward_to_backend(self, event: TUIEvent) -> None:
        """Forward TUI event to backend services."""
        try:
            if self.websocket_client:
                # Convert TUI event to WebSocket message
                ws_message = {
                    'type': 'tui_event',
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'data': event.data,
                    'widget_id': event.widget_id,
                    'screen_name': event.screen_name
                }
                
                await self.websocket_client.send(json.dumps(ws_message))
                
        except Exception as e:
            logger.error(f"Failed to forward event to backend: {e}")
    
    async def _websocket_message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        logger.info("Starting WebSocket message handler")
        
        try:
            async for message in self.websocket_client:
                try:
                    data = json.loads(message)
                    await self._handle_backend_message(data)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from WebSocket: {message}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.tui_state.websocket_connected = False
            
            # Attempt reconnection
            asyncio.create_task(self._reconnect_websocket())
            
        except Exception as e:
            logger.error(f"WebSocket message handler error: {e}")
    
    async def _handle_backend_message(self, data: Dict[str, Any]) -> None:
        """Handle message from backend."""
        message_type = data.get('type')
        
        if message_type == 'task_progress':
            await self._handle_task_progress_update(data)
        elif message_type == 'workflow_status':
            await self._handle_workflow_status_update(data)
        elif message_type == 'agent_status':
            await self._handle_agent_status_update(data)
        elif message_type == 'system_notification':
            await self._handle_system_notification(data)
        elif message_type == 'pong':
            # Heartbeat response
            self._update_latency(data.get('timestamp'))
        else:
            logger.debug(f"Received unknown message type: {message_type}")
    
    async def _handle_task_progress_update(self, data: Dict[str, Any]) -> None:
        """Handle task progress update from backend."""
        task_id = data.get('task_id')
        progress = data.get('progress_percentage', 0)
        status = data.get('status')
        
        # Create TUI event for progress update
        event = TUIEvent(
            event_type=TUIEventType.BACKEND_CONNECTION_STATUS,
            timestamp=datetime.now(),
            data={
                'task_id': task_id,
                'progress': progress,
                'status': status,
                'update_type': 'task_progress'
            }
        )
        
        await self.emit_event(event)
    
    async def _handle_workflow_status_update(self, data: Dict[str, Any]) -> None:
        """Handle workflow status update from backend."""
        workflow_id = data.get('workflow_id')
        status = data.get('status')
        progress = data.get('progress_percentage', 0)
        
        # Update TUI state if this is the active workflow
        if self.tui_state.active_project:
            event = TUIEvent(
                event_type=TUIEventType.SYNC_STATUS_CHANGED,
                timestamp=datetime.now(),
                data={
                    'workflow_id': workflow_id,
                    'status': status,
                    'progress': progress,
                    'update_type': 'workflow_status'
                }
            )
            
            await self.emit_event(event)
    
    async def _handle_agent_status_update(self, data: Dict[str, Any]) -> None:
        """Handle agent status update from backend."""
        agent_id = data.get('agent_id')
        agent_type = data.get('agent_type')
        status = data.get('status')
        
        logger.info(f"Agent {agent_type} ({agent_id}) status: {status}")
    
    async def _handle_system_notification(self, data: Dict[str, Any]) -> None:
        """Handle system notification from backend."""
        notification_type = data.get('notification_type')
        message = data.get('message')
        severity = data.get('severity', 'info')
        
        # Create TUI notification event
        event = TUIEvent(
            event_type=TUIEventType.BACKEND_CONNECTION_STATUS,
            timestamp=datetime.now(),
            data={
                'notification_type': notification_type,
                'message': message,
                'severity': severity,
                'update_type': 'system_notification'
            }
        )
        
        await self.emit_event(event)
    
    def _update_latency(self, timestamp_str: Optional[str]) -> None:
        """Update WebSocket latency measurement."""
        if timestamp_str:
            try:
                sent_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                latency = (datetime.now() - sent_time.replace(tzinfo=None)).total_seconds() * 1000
                self.sync_status.latency_ms = latency
            except Exception as e:
                logger.error(f"Failed to calculate latency: {e}")
    
    async def _reconnect_websocket(self) -> None:
        """Attempt to reconnect WebSocket."""
        await asyncio.sleep(5)  # Wait before reconnecting
        await self._connect_websocket()
    
    async def _sync_monitoring_loop(self) -> None:
        """Monitor synchronization status."""
        logger.info("Starting sync monitoring loop")
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update sync status
                now = datetime.now()
                if self.sync_status.last_sync:
                    time_since_sync = (now - self.sync_status.last_sync).total_seconds()
                    if time_since_sync > 60:  # No sync for 1 minute
                        logger.warning(f"No backend sync for {time_since_sync:.0f} seconds")
                
                # Check service health
                if self.orchestrator:
                    health_status = await self.orchestrator.get_service_status()
                    overall_status = health_status.get('overall_status')
                    
                    if overall_status != 'healthy':
                        logger.warning(f"Backend service status: {overall_status}")
                
            except Exception as e:
                logger.error(f"Sync monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor TUI performance metrics."""
        logger.info("Starting performance monitoring loop")
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Calculate refresh rate
                now = datetime.now()
                self.update_timestamps.append(now)
                
                # Keep only last 60 seconds of timestamps
                cutoff = now - timedelta(seconds=60)
                self.update_timestamps = [
                    ts for ts in self.update_timestamps if ts > cutoff
                ]
                
                # Calculate current refresh rate
                if len(self.update_timestamps) > 1:
                    time_span = (self.update_timestamps[-1] - self.update_timestamps[0]).total_seconds()
                    if time_span > 0:
                        self.tui_state.refresh_rate = len(self.update_timestamps) / time_span
                
                # Check for performance issues
                if self.tui_state.refresh_rate < 30:  # Less than 30 FPS
                    await self._emit_performance_warning()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _emit_performance_warning(self) -> None:
        """Emit performance warning event."""
        event = TUIEvent(
            event_type=TUIEventType.UI_LAG_DETECTED,
            timestamp=datetime.now(),
            data={
                'refresh_rate': self.tui_state.refresh_rate,
                'expected_rate': 60.0,
                'performance_impact': 'high'
            }
        )
        
        await self.emit_event(event)
    
    async def _batch_processing_loop(self) -> None:
        """Process events in batches for efficiency."""
        logger.info("Starting batch processing loop")
        
        while True:
            try:
                await asyncio.sleep(self.batch_timeout)
                
                if self.batch_events:
                    # Process batch
                    events_to_process = self.batch_events.copy()
                    self.batch_events.clear()
                    
                    await self._process_event_batch(events_to_process)
                    self.last_batch_time = datetime.now()
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_event_batch(self, events: List[TUIEvent]) -> None:
        """Process a batch of events efficiently."""
        if not events:
            return
        
        logger.debug(f"Processing batch of {len(events)} events")
        
        # Group events by type for efficient processing
        events_by_type: Dict[TUIEventType, List[TUIEvent]] = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
        
        # Process each type
        for event_type, type_events in events_by_type.items():
            try:
                await self._process_events_of_type(event_type, type_events)
            except Exception as e:
                logger.error(f"Failed to process {event_type} events: {e}")
    
    async def _process_events_of_type(self, event_type: TUIEventType, events: List[TUIEvent]) -> None:
        """Process multiple events of the same type efficiently."""
        # For focus changes, only process the latest
        if event_type == TUIEventType.WIDGET_FOCUS_CHANGED and events:
            await self._process_event(events[-1])
        
        # For other events, process all
        else:
            for event in events:
                await self._process_event(event)
    
    async def _websocket_heartbeat_loop(self) -> None:
        """Send periodic heartbeat to WebSocket."""
        logger.info("Starting WebSocket heartbeat loop")
        
        while True:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
                if self.websocket_client and self.tui_state.websocket_connected:
                    heartbeat = {
                        'type': 'ping',
                        'timestamp': datetime.now().isoformat(),
                        'tui_state': {
                            'screen': self.tui_state.current_screen,
                            'focused_widget': self.tui_state.focused_widget,
                            'active_project': self.tui_state.active_project
                        }
                    }
                    
                    await self.websocket_client.send(json.dumps(heartbeat))
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(60)
    
    def _update_performance_metrics(self, event: TUIEvent) -> None:
        """Update performance metrics with event data."""
        now = datetime.now()
        
        # Track event processing time
        if hasattr(event, 'created_at'):
            processing_time = (now - event.created_at).total_seconds() * 1000
            
            if 'event_processing_times' not in self.performance_metrics:
                self.performance_metrics['event_processing_times'] = []
            
            self.performance_metrics['event_processing_times'].append(processing_time)
            
            # Keep only last 100 measurements
            if len(self.performance_metrics['event_processing_times']) > 100:
                self.performance_metrics['event_processing_times'] = \
                    self.performance_metrics['event_processing_times'][-100:]
    
    # Public API Methods
    
    async def emit_event(self, event: TUIEvent) -> None:
        """Emit a TUI event for processing."""
        event.created_at = datetime.now()
        
        # Add to batch if batching is enabled
        if self.batch_timeout > 0:
            self.batch_events.append(event)
        else:
            await self.event_queue.put(event)
    
    def register_event_handler(self, event_type: TUIEventType, handler: Callable[[TUIEvent], None]) -> None:
        """Register an event handler for specific event types."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    def unregister_event_handler(self, event_type: TUIEventType, handler: Callable[[TUIEvent], None]) -> None:
        """Unregister an event handler."""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def update_tui_state(self, **kwargs) -> None:
        """Update TUI state and sync with backend."""
        for key, value in kwargs.items():
            if hasattr(self.tui_state, key):
                setattr(self.tui_state, key, value)
                logger.debug(f"Updated TUI state: {key} = {value}")
        
        self.tui_state.last_update = datetime.now()
        
        # Sync state to backend
        if self.orchestrator:
            cache_service = self.orchestrator.get_cache_service()
            if cache_service:
                await cache_service.set(
                    "tui:state",
                    self.tui_state.__dict__,
                    ttl=300
                )
    
    async def get_backend_data(self, data_type: str, **filters) -> Optional[Dict[str, Any]]:
        """Retrieve data from backend services."""
        if not self.orchestrator:
            return None
        
        try:
            cache_service = self.orchestrator.get_cache_service()
            if cache_service:
                cache_key = f"data:{data_type}:{':'.join(f'{k}:{v}' for k, v in filters.items())}"
                return await cache_service.get(cache_key)
        
        except Exception as e:
            logger.error(f"Failed to get backend data {data_type}: {e}")
        
        return None
    
    async def send_command_to_backend(self, command: str, parameters: Dict[str, Any] = None) -> bool:
        """Send command to backend services."""
        try:
            if self.websocket_client and self.tui_state.websocket_connected:
                message = {
                    'type': 'command',
                    'command': command,
                    'parameters': parameters or {},
                    'timestamp': datetime.now().isoformat(),
                    'tui_context': {
                        'screen': self.tui_state.current_screen,
                        'widget': self.tui_state.focused_widget,
                        'project': self.tui_state.active_project
                    }
                }
                
                await self.websocket_client.send(json.dumps(message))
                return True
        
        except Exception as e:
            logger.error(f"Failed to send command to backend: {e}")
        
        return False
    
    def get_sync_status(self) -> BackendSyncStatus:
        """Get current synchronization status."""
        return self.sync_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics.update({
            'refresh_rate': self.tui_state.refresh_rate,
            'websocket_latency_ms': self.sync_status.latency_ms,
            'queue_size': self.event_queue.qsize(),
            'batch_size': len(self.batch_events)
        })
        return metrics
    
    async def cleanup(self) -> None:
        """Clean up bridge resources."""
        logger.info("Cleaning up TUI Backend Bridge...")
        
        # Cancel background tasks
        for task in self.sync_tasks:
            task.cancel()
        
        # Close WebSocket connection
        if self.websocket_client:
            await self.websocket_client.close()
        
        # Clear state
        self.sync_status.connected = False
        self.tui_state.backend_connected = False
        self.tui_state.websocket_connected = False
        
        logger.info("TUI Backend Bridge cleanup completed")


# Global bridge instance
tui_bridge: Optional[TUIBackendBridge] = None


def get_tui_bridge() -> Optional[TUIBackendBridge]:
    """Get the global TUI bridge instance."""
    return tui_bridge


async def initialize_tui_bridge(config_manager: ConfigManager) -> TUIBackendBridge:
    """Initialize the global TUI backend bridge."""
    global tui_bridge
    
    if tui_bridge is None:
        tui_bridge = TUIBackendBridge(config_manager)
        await tui_bridge.initialize()
    
    return tui_bridge


async def cleanup_tui_bridge() -> None:
    """Clean up the global TUI backend bridge."""
    global tui_bridge
    
    if tui_bridge is not None:
        await tui_bridge.cleanup()
        tui_bridge = None
