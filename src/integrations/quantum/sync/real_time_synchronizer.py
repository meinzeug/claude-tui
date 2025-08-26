"""
Real-Time Cross-Platform Synchronization Service
High-performance synchronization engine for unified development experience
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import weakref
import threading
from collections import defaultdict, deque
import hashlib
import zlib

import websockets
import aioredis
from pydantic import BaseModel, Field, ConfigDict


class SyncEventType(str, Enum):
    """Synchronization event types"""
    FILE_CHANGED = "file_changed"
    PROJECT_OPENED = "project_opened"
    PROJECT_CLOSED = "project_closed"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    BREAKPOINT_SET = "breakpoint_set"
    BREAKPOINT_REMOVED = "breakpoint_removed"
    VARIABLE_WATCHED = "variable_watched"
    TERMINAL_COMMAND = "terminal_command"
    BUILD_STARTED = "build_started"
    BUILD_COMPLETED = "build_completed"
    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    DEPLOYMENT_STARTED = "deployment_started"
    DEPLOYMENT_COMPLETED = "deployment_completed"
    USER_PREFERENCE_CHANGED = "user_preference_changed"
    WORKSPACE_STATE_CHANGED = "workspace_state_changed"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies"""
    LAST_WRITER_WINS = "last_writer_wins"
    MERGE = "merge"
    USER_CHOICE = "user_choice"
    AUTOMATIC = "automatic"
    IGNORE = "ignore"


class SyncPriority(str, Enum):
    """Synchronization priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class SyncEvent:
    """Synchronization event data structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SyncEventType = SyncEventType.FILE_CHANGED
    timestamp: float = field(default_factory=time.time)
    source_environment: str = ""
    target_environments: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: SyncPriority = SyncPriority.MEDIUM
    ttl: Optional[int] = None  # Time to live in seconds
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
            
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity"""
        content = json.dumps(self.data, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()[:16]
        
    def is_expired(self) -> bool:
        """Check if event has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncEvent':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SyncState:
    """State information for synchronization"""
    environment_id: str
    last_sync_time: float = field(default_factory=time.time)
    version: int = 0
    active_files: Set[str] = field(default_factory=set)
    cursor_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    selections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    breakpoints: Dict[str, List[int]] = field(default_factory=dict)
    watched_variables: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    workspace_settings: Dict[str, Any] = field(default_factory=dict)
    pending_changes: List[SyncEvent] = field(default_factory=list)
    
    def increment_version(self):
        """Increment state version"""
        self.version += 1
        self.last_sync_time = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "environment_id": self.environment_id,
            "last_sync_time": self.last_sync_time,
            "version": self.version,
            "active_files": list(self.active_files),
            "cursor_positions": self.cursor_positions,
            "selections": self.selections,
            "breakpoints": self.breakpoints,
            "watched_variables": self.watched_variables,
            "user_preferences": self.user_preferences,
            "workspace_settings": self.workspace_settings,
            "pending_changes": [event.to_dict() for event in self.pending_changes]
        }


class SyncTransport(ABC):
    """Abstract base class for sync transport mechanisms"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to transport"""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Disconnect from transport"""
        pass
        
    @abstractmethod
    async def send_event(self, event: SyncEvent, target: str) -> bool:
        """Send sync event to target"""
        pass
        
    @abstractmethod
    async def receive_events(self) -> List[SyncEvent]:
        """Receive pending sync events"""
        pass
        
    @abstractmethod
    async def broadcast_event(self, event: SyncEvent, targets: List[str]) -> Dict[str, bool]:
        """Broadcast event to multiple targets"""
        pass


class WebSocketSyncTransport(SyncTransport):
    """WebSocket-based sync transport"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._server = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        
    async def connect(self) -> bool:
        """Start WebSocket server"""
        try:
            self._server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port
            )
            self.logger.info(f"WebSocket sync server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            return False
            
    async def disconnect(self):
        """Stop WebSocket server"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            
        # Close all client connections
        for client_id, websocket in list(self._clients.items()):
            try:
                await websocket.close()
            except:
                pass
            
        self._clients.clear()
        self.logger.info("WebSocket sync server stopped")
        
    async def _handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        client_id = f"client_{int(time.time())}"
        self._clients[client_id] = websocket
        
        try:
            self.logger.info(f"Client connected: {client_id}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event = SyncEvent.from_dict(data)
                    await self._event_queue.put(event)
                    
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self._clients.pop(client_id, None)
            
    async def send_event(self, event: SyncEvent, target: str) -> bool:
        """Send event to specific client"""
        try:
            if target not in self._clients:
                return False
                
            websocket = self._clients[target]
            message = json.dumps(event.to_dict())
            await websocket.send(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send event to {target}: {e}")
            return False
            
    async def receive_events(self) -> List[SyncEvent]:
        """Receive pending events"""
        events = []
        
        try:
            while not self._event_queue.empty():
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                events.append(event)
                
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            self.logger.error(f"Error receiving events: {e}")
            
        return events
        
    async def broadcast_event(self, event: SyncEvent, targets: List[str]) -> Dict[str, bool]:
        """Broadcast event to multiple targets"""
        results = {}
        
        for target in targets:
            results[target] = await self.send_event(event, target)
            
        return results


class RedisSyncTransport(SyncTransport):
    """Redis-based sync transport for scalable synchronization"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.logger = logging.getLogger(__name__)
        self._redis: Optional[aioredis.Redis] = None
        self._subscriber: Optional[aioredis.Redis] = None
        self._pubsub = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._subscription_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            self._redis = aioredis.from_url(self.redis_url)
            self._subscriber = aioredis.from_url(self.redis_url)
            
            # Test connection
            await self._redis.ping()
            
            # Start subscription
            self._pubsub = self._subscriber.pubsub()
            await self._pubsub.subscribe("sync_events")
            
            self._subscription_task = asyncio.create_task(self._listen_for_events())
            
            self.logger.info("Connected to Redis sync transport")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._subscription_task:
            self._subscription_task.cancel()
            
        if self._pubsub:
            await self._pubsub.unsubscribe("sync_events")
            await self._pubsub.close()
            
        if self._redis:
            await self._redis.close()
            
        if self._subscriber:
            await self._subscriber.close()
            
        self.logger.info("Disconnected from Redis sync transport")
        
    async def _listen_for_events(self):
        """Listen for Redis pub/sub events"""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        event = SyncEvent.from_dict(data)
                        await self._event_queue.put(event)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing Redis message: {e}")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in Redis listener: {e}")
            
    async def send_event(self, event: SyncEvent, target: str) -> bool:
        """Send event via Redis"""
        try:
            if not self._redis:
                return False
                
            # Send to specific target channel
            channel = f"sync_events_{target}"
            message = json.dumps(event.to_dict())
            await self._redis.publish(channel, message)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Redis event to {target}: {e}")
            return False
            
    async def receive_events(self) -> List[SyncEvent]:
        """Receive pending events"""
        events = []
        
        try:
            while not self._event_queue.empty():
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                events.append(event)
                
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            self.logger.error(f"Error receiving Redis events: {e}")
            
        return events
        
    async def broadcast_event(self, event: SyncEvent, targets: List[str]) -> Dict[str, bool]:
        """Broadcast event to multiple targets"""
        try:
            if not self._redis:
                return {target: False for target in targets}
                
            # Broadcast to general channel
            message = json.dumps(event.to_dict())
            await self._redis.publish("sync_events", message)
            
            return {target: True for target in targets}
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast Redis event: {e}")
            return {target: False for target in targets}


class ConflictResolver:
    """Conflict resolution engine for sync conflicts"""
    
    def __init__(self, default_strategy: ConflictResolution = ConflictResolution.LAST_WRITER_WINS):
        self.default_strategy = default_strategy
        self.logger = logging.getLogger(__name__)
        self._resolution_strategies: Dict[ConflictResolution, Callable] = {
            ConflictResolution.LAST_WRITER_WINS: self._last_writer_wins,
            ConflictResolution.MERGE: self._merge_changes,
            ConflictResolution.USER_CHOICE: self._user_choice,
            ConflictResolution.AUTOMATIC: self._automatic_resolution,
            ConflictResolution.IGNORE: self._ignore_conflict
        }
        
    async def resolve_conflict(self, local_event: SyncEvent, remote_event: SyncEvent,
                             strategy: Optional[ConflictResolution] = None) -> SyncEvent:
        """Resolve conflict between two sync events"""
        try:
            resolution_strategy = strategy or self.default_strategy
            resolver = self._resolution_strategies.get(resolution_strategy)
            
            if not resolver:
                self.logger.warning(f"Unknown resolution strategy: {resolution_strategy}")
                resolver = self._resolution_strategies[ConflictResolution.LAST_WRITER_WINS]
                
            resolved_event = await resolver(local_event, remote_event)
            
            self.logger.info(f"Resolved conflict using {resolution_strategy.value}")
            return resolved_event
            
        except Exception as e:
            self.logger.error(f"Error resolving conflict: {e}")
            return local_event  # Fallback to local event
            
    async def _last_writer_wins(self, local_event: SyncEvent, remote_event: SyncEvent) -> SyncEvent:
        """Last writer wins resolution"""
        return remote_event if remote_event.timestamp > local_event.timestamp else local_event
        
    async def _merge_changes(self, local_event: SyncEvent, remote_event: SyncEvent) -> SyncEvent:
        """Merge changes from both events"""
        merged_data = {}
        
        # Merge data fields
        for key, value in local_event.data.items():
            merged_data[key] = value
            
        for key, value in remote_event.data.items():
            if key not in merged_data:
                merged_data[key] = value
            elif isinstance(value, dict) and isinstance(merged_data[key], dict):
                merged_data[key].update(value)
            elif isinstance(value, list) and isinstance(merged_data[key], list):
                merged_data[key] = list(set(merged_data[key] + value))
            else:
                # Use remote value for scalar conflicts
                merged_data[key] = value
                
        # Create merged event
        merged_event = SyncEvent(
            event_type=local_event.event_type,
            source_environment="merged",
            data=merged_data,
            timestamp=max(local_event.timestamp, remote_event.timestamp),
            priority=max(local_event.priority, remote_event.priority, key=lambda x: x.value)
        )
        
        return merged_event
        
    async def _user_choice(self, local_event: SyncEvent, remote_event: SyncEvent) -> SyncEvent:
        """User choice resolution (placeholder - would show UI)"""
        # This would present conflict to user for resolution
        # For now, fallback to last writer wins
        return await self._last_writer_wins(local_event, remote_event)
        
    async def _automatic_resolution(self, local_event: SyncEvent, remote_event: SyncEvent) -> SyncEvent:
        """Automatic intelligent resolution"""
        # Apply heuristics based on event type
        if local_event.event_type == SyncEventType.FILE_CHANGED:
            # For file changes, try to merge if possible
            return await self._merge_changes(local_event, remote_event)
        elif local_event.event_type in [SyncEventType.CURSOR_MOVED, SyncEventType.SELECTION_CHANGED]:
            # For UI events, use last writer wins
            return await self._last_writer_wins(local_event, remote_event)
        else:
            # Default to merge
            return await self._merge_changes(local_event, remote_event)
            
    async def _ignore_conflict(self, local_event: SyncEvent, remote_event: SyncEvent) -> SyncEvent:
        """Ignore conflict and keep local event"""
        return local_event


class SyncFilter:
    """Event filtering system for selective synchronization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._filters: List[Callable[[SyncEvent], bool]] = []
        self._environment_filters: Dict[str, List[Callable]] = defaultdict(list)
        
    def add_global_filter(self, filter_func: Callable[[SyncEvent], bool]):
        """Add global filter for all environments"""
        self._filters.append(filter_func)
        
    def add_environment_filter(self, environment_id: str, filter_func: Callable[[SyncEvent], bool]):
        """Add filter for specific environment"""
        self._environment_filters[environment_id].append(filter_func)
        
    def should_sync(self, event: SyncEvent, target_environment: str) -> bool:
        """Check if event should be synchronized to target environment"""
        try:
            # Apply global filters
            for filter_func in self._filters:
                if not filter_func(event):
                    return False
                    
            # Apply environment-specific filters
            for filter_func in self._environment_filters.get(target_environment, []):
                if not filter_func(event):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying sync filters: {e}")
            return True  # Default to sync if filter error
            
    @staticmethod
    def create_file_type_filter(allowed_extensions: Set[str]) -> Callable[[SyncEvent], bool]:
        """Create filter for specific file types"""
        def filter_func(event: SyncEvent) -> bool:
            if event.event_type != SyncEventType.FILE_CHANGED:
                return True
                
            file_path = event.data.get("file_path", "")
            if file_path:
                extension = file_path.split(".")[-1].lower()
                return extension in allowed_extensions
                
            return True
            
        return filter_func
        
    @staticmethod
    def create_priority_filter(min_priority: SyncPriority) -> Callable[[SyncEvent], bool]:
        """Create filter for minimum priority level"""
        priority_order = {
            SyncPriority.BACKGROUND: 0,
            SyncPriority.LOW: 1,
            SyncPriority.MEDIUM: 2,
            SyncPriority.HIGH: 3,
            SyncPriority.CRITICAL: 4
        }
        
        def filter_func(event: SyncEvent) -> bool:
            return priority_order.get(event.priority, 0) >= priority_order.get(min_priority, 0)
            
        return filter_func
        
    @staticmethod
    def create_event_type_filter(allowed_types: Set[SyncEventType]) -> Callable[[SyncEvent], bool]:
        """Create filter for specific event types"""
        def filter_func(event: SyncEvent) -> bool:
            return event.event_type in allowed_types
            
        return filter_func


class RealTimeSynchronizer:
    """
    Real-Time Cross-Platform Synchronization Engine
    High-performance synchronization for unified development experience
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Transport layer
        transport_type = config.get("transport", "websocket")
        if transport_type == "websocket":
            self.transport = WebSocketSyncTransport(
                host=config.get("host", "localhost"),
                port=config.get("port", 8765)
            )
        elif transport_type == "redis":
            self.transport = RedisSyncTransport(
                redis_url=config.get("redis_url", "redis://localhost:6379")
            )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")
            
        # Core components
        self.conflict_resolver = ConflictResolver(
            default_strategy=ConflictResolution(
                config.get("conflict_resolution", "last_writer_wins")
            )
        )
        self.sync_filter = SyncFilter()
        
        # State management
        self._environments: Dict[str, SyncState] = {}
        self._event_history: deque = deque(maxlen=config.get("history_size", 1000))
        self._sync_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self._metrics = {
            "events_sent": 0,
            "events_received": 0,
            "conflicts_resolved": 0,
            "sync_errors": 0,
            "average_sync_latency": 0.0
        }
        
        # Configuration
        self.sync_interval = config.get("sync_interval", 0.1)  # 100ms
        self.batch_size = config.get("batch_size", 10)
        self.max_retries = config.get("max_retries", 3)
        
        self._initialize_default_filters()
        
    def _initialize_default_filters(self):
        """Initialize default sync filters"""
        # Filter out temporary files
        self.sync_filter.add_global_filter(
            lambda event: not any(
                temp in event.data.get("file_path", "").lower()
                for temp in [".tmp", ".temp", "~", ".swp", ".DS_Store"]
            )
        )
        
        # Filter by priority
        min_priority = SyncPriority(self.config.get("min_priority", "medium"))
        self.sync_filter.add_global_filter(
            SyncFilter.create_priority_filter(min_priority)
        )
        
    async def initialize(self) -> bool:
        """Initialize synchronization engine"""
        try:
            self.logger.info("Initializing Real-Time Synchronizer")
            
            # Connect transport
            if not await self.transport.connect():
                self.logger.error("Failed to connect transport")
                return False
                
            # Start sync loop
            self._sync_tasks["main"] = asyncio.create_task(self._sync_loop())
            
            self.logger.info("Real-Time Synchronizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize synchronizer: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown synchronization engine"""
        self.logger.info("Shutting down Real-Time Synchronizer")
        
        # Cancel sync tasks
        for task in self._sync_tasks.values():
            task.cancel()
            
        # Disconnect transport
        await self.transport.disconnect()
        
        self.logger.info("Real-Time Synchronizer shutdown complete")
        
    async def register_environment(self, environment_id: str, 
                                 initial_state: Optional[Dict[str, Any]] = None) -> bool:
        """Register new environment for synchronization"""
        try:
            if environment_id in self._environments:
                self.logger.warning(f"Environment already registered: {environment_id}")
                return True
                
            sync_state = SyncState(environment_id=environment_id)
            
            if initial_state:
                # Apply initial state
                sync_state.active_files.update(initial_state.get("active_files", []))
                sync_state.cursor_positions.update(initial_state.get("cursor_positions", {}))
                sync_state.selections.update(initial_state.get("selections", {}))
                sync_state.breakpoints.update(initial_state.get("breakpoints", {}))
                sync_state.watched_variables.extend(initial_state.get("watched_variables", []))
                sync_state.user_preferences.update(initial_state.get("user_preferences", {}))
                sync_state.workspace_settings.update(initial_state.get("workspace_settings", {}))
                
            self._environments[environment_id] = sync_state
            
            # Start environment-specific sync task
            self._sync_tasks[environment_id] = asyncio.create_task(
                self._environment_sync_loop(environment_id)
            )
            
            self.logger.info(f"Registered environment: {environment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register environment {environment_id}: {e}")
            return False
            
    async def unregister_environment(self, environment_id: str) -> bool:
        """Unregister environment from synchronization"""
        try:
            if environment_id not in self._environments:
                return True
                
            # Cancel environment sync task
            if environment_id in self._sync_tasks:
                self._sync_tasks[environment_id].cancel()
                del self._sync_tasks[environment_id]
                
            # Remove environment state
            del self._environments[environment_id]
            
            self.logger.info(f"Unregistered environment: {environment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister environment {environment_id}: {e}")
            return False
            
    async def sync_event(self, event: SyncEvent) -> bool:
        """Synchronize single event across environments"""
        try:
            # Validate event
            if event.is_expired():
                self.logger.debug(f"Event expired: {event.event_id}")
                return False
                
            # Add to history
            self._event_history.append(event)
            
            # Determine target environments
            targets = event.target_environments
            if not targets:
                # Broadcast to all environments except source
                targets = [
                    env_id for env_id in self._environments.keys()
                    if env_id != event.source_environment
                ]
                
            # Apply filters
            filtered_targets = [
                target for target in targets
                if self.sync_filter.should_sync(event, target)
            ]
            
            if not filtered_targets:
                self.logger.debug(f"Event filtered out: {event.event_id}")
                return True
                
            # Send event to targets
            start_time = time.time()
            results = await self.transport.broadcast_event(event, filtered_targets)
            
            # Update metrics
            self._metrics["events_sent"] += 1
            sync_latency = time.time() - start_time
            self._update_average_latency(sync_latency)
            
            # Check results
            success_count = sum(1 for success in results.values() if success)
            if success_count < len(filtered_targets):
                self.logger.warning(f"Partial sync failure for event {event.event_id}")
                
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to sync event {event.event_id}: {e}")
            self._metrics["sync_errors"] += 1
            return False
            
    async def sync_state(self, environment_id: str, state_updates: Dict[str, Any]) -> bool:
        """Synchronize state changes for specific environment"""
        try:
            if environment_id not in self._environments:
                self.logger.error(f"Environment not registered: {environment_id}")
                return False
                
            env_state = self._environments[environment_id]
            
            # Create sync events for state changes
            events = []
            
            # File changes
            if "active_files" in state_updates:
                new_files = set(state_updates["active_files"])
                old_files = env_state.active_files
                
                # Files opened
                opened_files = new_files - old_files
                for file_path in opened_files:
                    events.append(SyncEvent(
                        event_type=SyncEventType.PROJECT_OPENED,
                        source_environment=environment_id,
                        data={"file_path": file_path},
                        priority=SyncPriority.HIGH
                    ))
                    
                # Files closed
                closed_files = old_files - new_files
                for file_path in closed_files:
                    events.append(SyncEvent(
                        event_type=SyncEventType.PROJECT_CLOSED,
                        source_environment=environment_id,
                        data={"file_path": file_path},
                        priority=SyncPriority.HIGH
                    ))
                    
                env_state.active_files = new_files
                
            # Cursor position changes
            if "cursor_positions" in state_updates:
                for file_path, position in state_updates["cursor_positions"].items():
                    if env_state.cursor_positions.get(file_path) != position:
                        events.append(SyncEvent(
                            event_type=SyncEventType.CURSOR_MOVED,
                            source_environment=environment_id,
                            data={
                                "file_path": file_path,
                                "position": position
                            },
                            priority=SyncPriority.LOW
                        ))
                        
                env_state.cursor_positions.update(state_updates["cursor_positions"])
                
            # Selection changes
            if "selections" in state_updates:
                for file_path, selection in state_updates["selections"].items():
                    if env_state.selections.get(file_path) != selection:
                        events.append(SyncEvent(
                            event_type=SyncEventType.SELECTION_CHANGED,
                            source_environment=environment_id,
                            data={
                                "file_path": file_path,
                                "selection": selection
                            },
                            priority=SyncPriority.LOW
                        ))
                        
                env_state.selections.update(state_updates["selections"])
                
            # Breakpoint changes
            if "breakpoints" in state_updates:
                for file_path, breakpoints in state_updates["breakpoints"].items():
                    old_breakpoints = set(env_state.breakpoints.get(file_path, []))
                    new_breakpoints = set(breakpoints)
                    
                    # Added breakpoints
                    added = new_breakpoints - old_breakpoints
                    for line_num in added:
                        events.append(SyncEvent(
                            event_type=SyncEventType.BREAKPOINT_SET,
                            source_environment=environment_id,
                            data={
                                "file_path": file_path,
                                "line_number": line_num
                            },
                            priority=SyncPriority.HIGH
                        ))
                        
                    # Removed breakpoints
                    removed = old_breakpoints - new_breakpoints
                    for line_num in removed:
                        events.append(SyncEvent(
                            event_type=SyncEventType.BREAKPOINT_REMOVED,
                            source_environment=environment_id,
                            data={
                                "file_path": file_path,
                                "line_number": line_num
                            },
                            priority=SyncPriority.HIGH
                        ))
                        
                env_state.breakpoints.update(state_updates["breakpoints"])
                
            # User preferences changes
            if "user_preferences" in state_updates:
                for key, value in state_updates["user_preferences"].items():
                    if env_state.user_preferences.get(key) != value:
                        events.append(SyncEvent(
                            event_type=SyncEventType.USER_PREFERENCE_CHANGED,
                            source_environment=environment_id,
                            data={
                                "preference_key": key,
                                "preference_value": value
                            },
                            priority=SyncPriority.MEDIUM
                        ))
                        
                env_state.user_preferences.update(state_updates["user_preferences"])
                
            # Update environment state
            env_state.increment_version()
            
            # Sync all events
            sync_results = []
            for event in events:
                result = await self.sync_event(event)
                sync_results.append(result)
                
            return all(sync_results)
            
        except Exception as e:
            self.logger.error(f"Failed to sync state for {environment_id}: {e}")
            return False
            
    async def get_environment_state(self, environment_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of environment"""
        if environment_id not in self._environments:
            return None
            
        return self._environments[environment_id].to_dict()
        
    async def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization performance metrics"""
        return {
            **self._metrics,
            "active_environments": len(self._environments),
            "event_history_size": len(self._event_history),
            "active_sync_tasks": len(self._sync_tasks)
        }
        
    async def _sync_loop(self):
        """Main synchronization loop"""
        while True:
            try:
                # Receive events from transport
                incoming_events = await self.transport.receive_events()
                
                for event in incoming_events:
                    await self._process_incoming_event(event)
                    self._metrics["events_received"] += 1
                    
                await asyncio.sleep(self.sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
                
    async def _environment_sync_loop(self, environment_id: str):
        """Environment-specific sync loop"""
        while True:
            try:
                if environment_id not in self._environments:
                    break
                    
                env_state = self._environments[environment_id]
                
                # Process pending changes
                if env_state.pending_changes:
                    batch = env_state.pending_changes[:self.batch_size]
                    env_state.pending_changes = env_state.pending_changes[self.batch_size:]
                    
                    for event in batch:
                        await self.sync_event(event)
                        
                await asyncio.sleep(self.sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in environment sync loop for {environment_id}: {e}")
                await asyncio.sleep(1)
                
    async def _process_incoming_event(self, event: SyncEvent):
        """Process incoming synchronization event"""
        try:
            # Check if we have a conflicting local event
            local_event = self._find_conflicting_event(event)
            
            if local_event:
                # Resolve conflict
                resolved_event = await self.conflict_resolver.resolve_conflict(
                    local_event, event
                )
                event = resolved_event
                self._metrics["conflicts_resolved"] += 1
                
            # Apply event to local state
            await self._apply_event_to_state(event)
            
        except Exception as e:
            self.logger.error(f"Error processing incoming event {event.event_id}: {e}")
            
    def _find_conflicting_event(self, incoming_event: SyncEvent) -> Optional[SyncEvent]:
        """Find conflicting local event"""
        # Look for recent events of the same type affecting the same resource
        for historical_event in reversed(self._event_history):
            if (historical_event.event_type == incoming_event.event_type and
                historical_event.data.get("file_path") == incoming_event.data.get("file_path") and
                abs(historical_event.timestamp - incoming_event.timestamp) < 1.0):  # Within 1 second
                return historical_event
                
        return None
        
    async def _apply_event_to_state(self, event: SyncEvent):
        """Apply sync event to local environment state"""
        # This would update the relevant environment state based on the event
        # For now, we'll just log the event
        self.logger.debug(f"Applying event: {event.event_type.value} from {event.source_environment}")
        
    def _update_average_latency(self, new_latency: float):
        """Update average sync latency metric"""
        current_avg = self._metrics["average_sync_latency"]
        if current_avg == 0:
            self._metrics["average_sync_latency"] = new_latency
        else:
            # Exponential moving average
            self._metrics["average_sync_latency"] = (current_avg * 0.9) + (new_latency * 0.1)


# Utility functions for creating common sync events
def create_file_change_event(environment_id: str, file_path: str, 
                           content: str, checksum: str) -> SyncEvent:
    """Create file change sync event"""
    return SyncEvent(
        event_type=SyncEventType.FILE_CHANGED,
        source_environment=environment_id,
        data={
            "file_path": file_path,
            "content": content,
            "checksum": checksum
        },
        priority=SyncPriority.HIGH
    )


def create_cursor_move_event(environment_id: str, file_path: str, 
                           line: int, column: int) -> SyncEvent:
    """Create cursor movement sync event"""
    return SyncEvent(
        event_type=SyncEventType.CURSOR_MOVED,
        source_environment=environment_id,
        data={
            "file_path": file_path,
            "position": [line, column]
        },
        priority=SyncPriority.LOW,
        ttl=5  # Cursor events expire quickly
    )


def create_build_event(environment_id: str, build_type: str, 
                      status: str, output: str = "") -> SyncEvent:
    """Create build-related sync event"""
    event_type = SyncEventType.BUILD_STARTED if status == "started" else SyncEventType.BUILD_COMPLETED
    
    return SyncEvent(
        event_type=event_type,
        source_environment=environment_id,
        data={
            "build_type": build_type,
            "status": status,
            "output": output
        },
        priority=SyncPriority.HIGH
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "transport": "websocket",
            "host": "localhost",
            "port": 8765,
            "sync_interval": 0.1,
            "conflict_resolution": "merge",
            "min_priority": "medium"
        }
        
        synchronizer = RealTimeSynchronizer(config)
        
        try:
            await synchronizer.initialize()
            
            # Register environments
            await synchronizer.register_environment("vscode", {
                "active_files": ["main.py", "utils.py"],
                "cursor_positions": {"main.py": [10, 5]}
            })
            
            await synchronizer.register_environment("intellij", {
                "active_files": ["main.py"],
                "cursor_positions": {"main.py": [10, 5]}
            })
            
            # Sync a file change event
            file_event = create_file_change_event(
                "vscode",
                "main.py",
                "def hello():\n    print('Hello World')",
                "abc123"
            )
            
            await synchronizer.sync_event(file_event)
            
            # Get metrics
            metrics = await synchronizer.get_sync_metrics()
            print(f"Sync metrics: {metrics}")
            
            # Keep running
            await asyncio.sleep(10)
            
        finally:
            await synchronizer.shutdown()
            
    # Run example
    # asyncio.run(main())