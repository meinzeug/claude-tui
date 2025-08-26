"""
Real-time Synchronization Engine
WebSocket-based real-time collaboration with state synchronization
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from .models import (
    Workspace, WorkspaceMember, CollaborationSession,
    ConflictResolution, ConflictType, ConflictStatus,
    ActivityFeed, ActivityType
)

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"
    
    # File operations
    FILE_OPEN = "file_open"
    FILE_CLOSE = "file_close"
    FILE_EDIT = "file_edit"
    FILE_SAVE = "file_save"
    FILE_LOCK = "file_lock"
    FILE_UNLOCK = "file_unlock"
    
    # Presence updates
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    USER_PRESENCE = "user_presence"
    
    # Collaboration events
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    STATE_SYNC = "state_sync"
    
    # Notifications
    NOTIFICATION = "notification"
    ACTIVITY_UPDATE = "activity_update"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message"""
    type: str
    workspace_id: str
    user_id: str
    data: Dict[str, Any]
    timestamp: str
    message_id: str = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class FileEditOperation:
    """File editing operation for synchronization"""
    file_path: str
    operation_type: str  # insert, delete, replace
    position: Dict[str, int]  # line, column
    content: str
    length: int = 0
    user_id: str = None
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class UserPresence:
    """User presence information"""
    user_id: str
    username: str
    current_file: Optional[str]
    cursor_position: Dict[str, int]
    selection_range: Dict[str, Any]
    is_editing: bool
    last_seen: str
    color: str = "#666666"  # UI color for user


class SynchronizationEngine:
    """
    Real-time synchronization engine for collaborative editing.
    Manages WebSocket connections, state synchronization, and conflict resolution.
    """
    
    def __init__(self, db_session: Session, host: str = "localhost", port: int = 8765):
        """
        Initialize synchronization engine.
        
        Args:
            db_session: Database session
            host: WebSocket server host
            port: WebSocket server port
        """
        self.db = db_session
        self.host = host
        self.port = port
        
        # Connection management
        self._connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._user_connections: Dict[UUID, Set[str]] = {}  # user_id -> connection_ids
        self._workspace_connections: Dict[UUID, Set[str]] = {}  # workspace_id -> connection_ids
        
        # State synchronization
        self._file_states: Dict[str, Dict[str, Any]] = {}  # file_path -> state
        self._file_locks: Dict[str, Dict[str, Any]] = {}  # file_path -> lock_info
        self._operation_queue: Dict[str, List[FileEditOperation]] = {}  # file_path -> operations
        
        # Presence tracking
        self._user_presence: Dict[UUID, UserPresence] = {}
        
        # Message handlers
        self._message_handlers: Dict[str, Callable] = {
            MessageType.CONNECT.value: self._handle_connect,
            MessageType.DISCONNECT.value: self._handle_disconnect,
            MessageType.HEARTBEAT.value: self._handle_heartbeat,
            MessageType.FILE_OPEN.value: self._handle_file_open,
            MessageType.FILE_CLOSE.value: self._handle_file_close,
            MessageType.FILE_EDIT.value: self._handle_file_edit,
            MessageType.FILE_SAVE.value: self._handle_file_save,
            MessageType.CURSOR_MOVE.value: self._handle_cursor_move,
            MessageType.SELECTION_CHANGE.value: self._handle_selection_change,
            MessageType.STATE_SYNC.value: self._handle_state_sync
        }
        
        # Server instance
        self._server = None
        self._running = False
        
        logger.info(f"Synchronization engine initialized on {host}:{port}")
    
    async def start_server(self) -> None:
        """Start WebSocket server"""
        if self._running:
            return
        
        logger.info(f"Starting synchronization server on {self.host}:{self.port}")
        
        try:
            self._server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            self._running = True
            
            # Start background tasks
            asyncio.create_task(self._cleanup_inactive_sessions())
            asyncio.create_task(self._process_operation_queue())
            
            logger.info("Synchronization server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start synchronization server: {e}")
            raise
    
    async def stop_server(self) -> None:
        """Stop WebSocket server"""
        if not self._running or not self._server:
            return
        
        logger.info("Stopping synchronization server")
        
        try:
            # Close all connections
            for connection_id in list(self._connections.keys()):
                await self._close_connection(connection_id)
            
            # Stop server
            self._server.close()
            await self._server.wait_closed()
            self._running = False
            
            logger.info("Synchronization server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
    
    async def _handle_connection(self, websocket, path: str) -> None:
        """Handle new WebSocket connection"""
        connection_id = str(uuid4())
        self._connections[connection_id] = websocket
        
        logger.debug(f"New WebSocket connection: {connection_id}")
        
        try:
            async for message in websocket:
                await self._process_message(connection_id, message)
                
        except ConnectionClosed:
            logger.debug(f"Connection {connection_id} closed normally")
        except WebSocketException as e:
            logger.warning(f"WebSocket error on connection {connection_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on connection {connection_id}: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def _process_message(self, connection_id: str, raw_message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            message = WebSocketMessage.from_json(raw_message)
            handler = self._message_handlers.get(message.type)
            
            if handler:
                await handler(connection_id, message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message from {connection_id}: {e}")
        except Exception as e:
            logger.error(f"Error processing message from {connection_id}: {e}")
    
    async def _handle_connect(self, connection_id: str, message: WebSocketMessage) -> None:
        """Handle connection establishment"""
        workspace_id = UUID(message.workspace_id)
        user_id = UUID(message.user_id)
        
        # Validate workspace membership
        member = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.is_active == True
            )
        ).first()
        
        if not member:
            await self._send_error(connection_id, "Unauthorized: Not a workspace member")
            await self._close_connection(connection_id)
            return
        
        # Create or update collaboration session
        session = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.user_id == user_id,
                CollaborationSession.is_active == True
            )
        ).first()
        
        if not session:
            session = CollaborationSession(
                workspace_id=workspace_id,
                user_id=user_id,
                session_token=connection_id,
                ip_address=message.data.get("ip_address"),
                user_agent=message.data.get("user_agent")
            )
            self.db.add(session)
        else:
            session.session_token = connection_id
            session.last_heartbeat = datetime.now(timezone.utc)
        
        self.db.commit()
        
        # Update connection mappings
        if user_id not in self._user_connections:
            self._user_connections[user_id] = set()
        self._user_connections[user_id].add(connection_id)
        
        if workspace_id not in self._workspace_connections:
            self._workspace_connections[workspace_id] = set()
        self._workspace_connections[workspace_id].add(connection_id)
        
        # Initialize user presence
        user = self.db.query(member.user).first()
        self._user_presence[user_id] = UserPresence(
            user_id=str(user_id),
            username=user.username,
            current_file=None,
            cursor_position={"line": 0, "column": 0},
            selection_range={},
            is_editing=False,
            last_seen=datetime.now(timezone.utc).isoformat(),
            color=message.data.get("color", f"#{hash(str(user_id)) % 0xFFFFFF:06x}")
        )
        
        # Send connection confirmation
        response = WebSocketMessage(
            type="connected",
            workspace_id=message.workspace_id,
            user_id=message.user_id,
            data={
                "connection_id": connection_id,
                "session_id": str(session.id),
                "workspace_state": await self._get_workspace_state(workspace_id)
            }
        )
        await self._send_message(connection_id, response)
        
        # Notify other users
        await self._broadcast_presence_update(workspace_id, user_id)
        
        logger.info(f"User {user_id} connected to workspace {workspace_id}")
    
    async def _handle_file_open(self, connection_id: str, message: WebSocketMessage) -> None:
        """Handle file open event"""
        user_id = UUID(message.user_id)
        workspace_id = UUID(message.workspace_id)
        file_path = message.data["file_path"]
        
        # Update user presence
        if user_id in self._user_presence:
            self._user_presence[user_id].current_file = file_path
            self._user_presence[user_id].last_seen = datetime.now(timezone.utc).isoformat()
        
        # Initialize file state if needed
        if file_path not in self._file_states:
            self._file_states[file_path] = {
                "content": message.data.get("content", ""),
                "version": 1,
                "last_modified": datetime.now(timezone.utc).isoformat(),
                "active_users": set()
            }
        
        self._file_states[file_path]["active_users"].add(str(user_id))
        
        # Send file state to user
        response = WebSocketMessage(
            type="file_opened",
            workspace_id=message.workspace_id,
            user_id=message.user_id,
            data={
                "file_path": file_path,
                "file_state": self._file_states[file_path].copy(),
                "active_users": list(self._file_states[file_path]["active_users"])
            }
        )
        # Convert set to list for JSON serialization
        response.data["file_state"]["active_users"] = list(response.data["file_state"]["active_users"])
        
        await self._send_message(connection_id, response)
        
        # Notify other users about file access
        await self._broadcast_to_workspace(workspace_id, WebSocketMessage(
            type="user_file_access",
            workspace_id=message.workspace_id,
            user_id=message.user_id,
            data={
                "file_path": file_path,
                "action": "opened"
            }
        ), exclude_user=user_id)
    
    async def _handle_file_edit(self, connection_id: str, message: WebSocketMessage) -> None:
        """Handle file edit operation with conflict detection"""
        user_id = UUID(message.user_id)
        workspace_id = UUID(message.workspace_id)
        file_path = message.data["file_path"]
        
        # Create edit operation
        operation = FileEditOperation(
            file_path=file_path,
            operation_type=message.data["operation_type"],
            position=message.data["position"],
            content=message.data["content"],
            length=message.data.get("length", 0),
            user_id=str(user_id)
        )
        
        # Add to operation queue
        if file_path not in self._operation_queue:
            self._operation_queue[file_path] = []
        self._operation_queue[file_path].append(operation)
        
        # Check for conflicts
        conflict_detected = await self._detect_edit_conflicts(file_path, operation)
        
        if conflict_detected:
            # Create conflict resolution record
            conflict = ConflictResolution(
                workspace_id=workspace_id,
                conflict_type=ConflictType.CONCURRENT_EDIT.value,
                affected_users=[str(user_id)],
                affected_files=[file_path],
                conflict_data={
                    "operation": asdict(operation),
                    "conflicting_operations": [
                        asdict(op) for op in self._operation_queue[file_path][-5:]
                    ]
                }
            )
            self.db.add(conflict)
            self.db.commit()
            
            # Notify about conflict
            conflict_message = WebSocketMessage(
                type=MessageType.CONFLICT_DETECTED.value,
                workspace_id=message.workspace_id,
                user_id=message.user_id,
                data={
                    "conflict_id": str(conflict.id),
                    "file_path": file_path,
                    "conflict_type": "concurrent_edit",
                    "operation": asdict(operation)
                }
            )
            
            await self._broadcast_to_file_users(file_path, conflict_message)
        else:
            # Apply operation and broadcast
            await self._apply_edit_operation(file_path, operation)
            
            # Broadcast to other users
            edit_message = WebSocketMessage(
                type="file_edited",
                workspace_id=message.workspace_id,
                user_id=message.user_id,
                data={
                    "file_path": file_path,
                    "operation": asdict(operation),
                    "file_version": self._file_states[file_path]["version"]
                }
            )
            
            await self._broadcast_to_file_users(file_path, edit_message, exclude_user=user_id)
    
    async def _handle_cursor_move(self, connection_id: str, message: WebSocketMessage) -> None:
        """Handle cursor position update"""
        user_id = UUID(message.user_id)
        workspace_id = UUID(message.workspace_id)
        
        # Update presence
        if user_id in self._user_presence:
            self._user_presence[user_id].cursor_position = message.data["position"]
            self._user_presence[user_id].current_file = message.data.get("file_path")
            self._user_presence[user_id].last_seen = datetime.now(timezone.utc).isoformat()
        
        # Update database session
        session = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.user_id == user_id,
                CollaborationSession.is_active == True
            )
        ).first()
        
        if session:
            session.cursor_position = message.data["position"]
            session.current_file = message.data.get("file_path")
            session.last_heartbeat = datetime.now(timezone.utc)
            self.db.commit()
        
        # Broadcast presence update
        await self._broadcast_presence_update(workspace_id, user_id)
    
    async def _handle_selection_change(self, connection_id: str, message: WebSocketMessage) -> None:
        """Handle text selection change"""
        user_id = UUID(message.user_id)
        workspace_id = UUID(message.workspace_id)
        
        # Update presence
        if user_id in self._user_presence:
            self._user_presence[user_id].selection_range = message.data["selection"]
            self._user_presence[user_id].is_editing = message.data.get("is_editing", False)
            self._user_presence[user_id].last_seen = datetime.now(timezone.utc).isoformat()
        
        # Broadcast to workspace
        await self._broadcast_presence_update(workspace_id, user_id)
    
    async def _handle_heartbeat(self, connection_id: str, message: WebSocketMessage) -> None:
        """Handle heartbeat message"""
        user_id = UUID(message.user_id)
        workspace_id = UUID(message.workspace_id)
        
        # Update session
        session = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.user_id == user_id,
                CollaborationSession.is_active == True
            )
        ).first()
        
        if session:
            session.last_heartbeat = datetime.now(timezone.utc)
            self.db.commit()
        
        # Send heartbeat response
        response = WebSocketMessage(
            type="heartbeat_ack",
            workspace_id=message.workspace_id,
            user_id=message.user_id,
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        )
        await self._send_message(connection_id, response)
    
    async def _detect_edit_conflicts(
        self,
        file_path: str,
        new_operation: FileEditOperation
    ) -> bool:
        """Detect if edit operation conflicts with recent operations"""
        if file_path not in self._operation_queue:
            return False
        
        recent_operations = self._operation_queue[file_path][-10:]  # Check last 10 operations
        
        for op in recent_operations:
            # Skip operations from same user
            if op.user_id == new_operation.user_id:
                continue
            
            # Check if operations overlap
            if self._operations_overlap(op, new_operation):
                # Check if operations happened within conflict window (30 seconds)
                op_time = datetime.fromisoformat(op.timestamp)
                new_time = datetime.fromisoformat(new_operation.timestamp)
                
                if abs((new_time - op_time).total_seconds()) < 30:
                    return True
        
        return False
    
    def _operations_overlap(self, op1: FileEditOperation, op2: FileEditOperation) -> bool:
        """Check if two operations overlap in the same area"""
        # Simple overlap detection based on line numbers
        line1 = op1.position.get("line", 0)
        line2 = op2.position.get("line", 0)
        
        # Consider operations on same line or adjacent lines as overlapping
        return abs(line1 - line2) <= 1
    
    async def _apply_edit_operation(self, file_path: str, operation: FileEditOperation) -> None:
        """Apply edit operation to file state"""
        if file_path not in self._file_states:
            return
        
        file_state = self._file_states[file_path]
        
        # Apply operation based on type
        if operation.operation_type == "insert":
            # Insert content at position
            lines = file_state["content"].split("\n")
            line_num = operation.position.get("line", 0)
            col_num = operation.position.get("column", 0)
            
            if line_num < len(lines):
                line = lines[line_num]
                new_line = line[:col_num] + operation.content + line[col_num:]
                lines[line_num] = new_line
                file_state["content"] = "\n".join(lines)
        
        elif operation.operation_type == "delete":
            # Delete content from position
            lines = file_state["content"].split("\n")
            line_num = operation.position.get("line", 0)
            col_num = operation.position.get("column", 0)
            
            if line_num < len(lines):
                line = lines[line_num]
                end_col = min(col_num + operation.length, len(line))
                new_line = line[:col_num] + line[end_col:]
                lines[line_num] = new_line
                file_state["content"] = "\n".join(lines)
        
        elif operation.operation_type == "replace":
            # Replace content at position
            lines = file_state["content"].split("\n")
            line_num = operation.position.get("line", 0)
            col_num = operation.position.get("column", 0)
            
            if line_num < len(lines):
                line = lines[line_num]
                end_col = min(col_num + operation.length, len(line))
                new_line = line[:col_num] + operation.content + line[end_col:]
                lines[line_num] = new_line
                file_state["content"] = "\n".join(lines)
        
        # Update version and timestamp
        file_state["version"] += 1
        file_state["last_modified"] = datetime.now(timezone.utc).isoformat()
    
    async def _broadcast_presence_update(self, workspace_id: UUID, user_id: UUID) -> None:
        """Broadcast user presence update to workspace"""
        if user_id not in self._user_presence:
            return
        
        presence = self._user_presence[user_id]
        message = WebSocketMessage(
            type=MessageType.USER_PRESENCE.value,
            workspace_id=str(workspace_id),
            user_id=str(user_id),
            data={
                "presence": asdict(presence)
            }
        )
        
        await self._broadcast_to_workspace(workspace_id, message, exclude_user=user_id)
    
    async def _broadcast_to_workspace(
        self,
        workspace_id: UUID,
        message: WebSocketMessage,
        exclude_user: UUID = None
    ) -> None:
        """Broadcast message to all connections in workspace"""
        if workspace_id not in self._workspace_connections:
            return
        
        connections = self._workspace_connections[workspace_id].copy()
        
        # Filter out excluded user's connections
        if exclude_user and exclude_user in self._user_connections:
            excluded_connections = self._user_connections[exclude_user]
            connections -= excluded_connections
        
        # Send to all remaining connections
        for connection_id in connections:
            await self._send_message(connection_id, message)
    
    async def _broadcast_to_file_users(
        self,
        file_path: str,
        message: WebSocketMessage,
        exclude_user: UUID = None
    ) -> None:
        """Broadcast message to users working on specific file"""
        if file_path not in self._file_states:
            return
        
        active_users = self._file_states[file_path]["active_users"]
        
        for user_id_str in active_users:
            user_id = UUID(user_id_str)
            
            if exclude_user and user_id == exclude_user:
                continue
            
            if user_id in self._user_connections:
                for connection_id in self._user_connections[user_id]:
                    await self._send_message(connection_id, message)
    
    async def _send_message(self, connection_id: str, message: WebSocketMessage) -> None:
        """Send message to specific connection"""
        if connection_id not in self._connections:
            return
        
        websocket = self._connections[connection_id]
        
        try:
            await websocket.send(message.to_json())
        except (ConnectionClosed, WebSocketException):
            # Connection lost, clean up
            await self._cleanup_connection(connection_id)
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
    
    async def _send_error(self, connection_id: str, error_message: str) -> None:
        """Send error message to connection"""
        error_msg = WebSocketMessage(
            type="error",
            workspace_id="",
            user_id="",
            data={"error": error_message}
        )
        await self._send_message(connection_id, error_msg)
    
    async def _close_connection(self, connection_id: str) -> None:
        """Close specific connection"""
        if connection_id in self._connections:
            try:
                websocket = self._connections[connection_id]
                await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing connection {connection_id}: {e}")
        
        await self._cleanup_connection(connection_id)
    
    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up connection and associated data"""
        # Remove from connections
        if connection_id in self._connections:
            del self._connections[connection_id]
        
        # Find and clean up user/workspace mappings
        user_to_remove = None
        for user_id, connection_set in self._user_connections.items():
            if connection_id in connection_set:
                connection_set.remove(connection_id)
                if not connection_set:
                    user_to_remove = user_id
                break
        
        if user_to_remove:
            del self._user_connections[user_to_remove]
            # Clean up presence
            if user_to_remove in self._user_presence:
                del self._user_presence[user_to_remove]
        
        # Clean up workspace mappings
        for workspace_id, connection_set in self._workspace_connections.items():
            connection_set.discard(connection_id)
        
        # Update database session
        try:
            session = self.db.query(CollaborationSession).filter(
                CollaborationSession.session_token == connection_id
            ).first()
            
            if session:
                session.is_active = False
                session.ended_at = datetime.now(timezone.utc)
                self.db.commit()
        except Exception as e:
            logger.error(f"Error updating session on cleanup: {e}")
    
    async def _get_workspace_state(self, workspace_id: UUID) -> Dict[str, Any]:
        """Get current workspace state for new connections"""
        # Get active members and their presence
        members = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.is_active == True
            )
        ).all()
        
        active_users = []
        for member in members:
            if member.user_id in self._user_presence:
                presence = self._user_presence[member.user_id]
                active_users.append(asdict(presence))
        
        return {
            "active_users": active_users,
            "member_count": len(members),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Background task to clean up inactive sessions"""
        while self._running:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=10)
                
                # Find inactive sessions
                inactive_sessions = self.db.query(CollaborationSession).filter(
                    and_(
                        CollaborationSession.is_active == True,
                        CollaborationSession.last_heartbeat < cutoff_time
                    )
                ).all()
                
                for session in inactive_sessions:
                    session.is_active = False
                    session.ended_at = datetime.now(timezone.utc)
                    
                    # Clean up connection if still exists
                    if session.session_token in self._connections:
                        await self._cleanup_connection(session.session_token)
                
                if inactive_sessions:
                    self.db.commit()
                    logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
            
            await asyncio.sleep(60)  # Run every minute
    
    async def _process_operation_queue(self) -> None:
        """Background task to process queued operations"""
        while self._running:
            try:
                # Process operations for each file
                for file_path in list(self._operation_queue.keys()):
                    operations = self._operation_queue[file_path]
                    
                    if operations:
                        # Keep only recent operations (last hour)
                        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
                        recent_ops = [
                            op for op in operations
                            if datetime.fromisoformat(op.timestamp) > cutoff_time
                        ]
                        
                        self._operation_queue[file_path] = recent_ops[-100:]  # Keep last 100 operations
                
                # Clean up empty queues
                empty_files = [f for f, ops in self._operation_queue.items() if not ops]
                for file_path in empty_files:
                    del self._operation_queue[file_path]
                
            except Exception as e:
                logger.error(f"Error in operation queue processing: {e}")
            
            await asyncio.sleep(30)  # Run every 30 seconds
    
    # Public API methods
    
    async def get_workspace_presence(self, workspace_id: UUID) -> List[Dict[str, Any]]:
        """Get presence information for all users in workspace"""
        presence_list = []
        
        if workspace_id in self._workspace_connections:
            for user_id, presence in self._user_presence.items():
                # Check if user is in this workspace
                member = self.db.query(WorkspaceMember).filter(
                    and_(
                        WorkspaceMember.workspace_id == workspace_id,
                        WorkspaceMember.user_id == user_id,
                        WorkspaceMember.is_active == True
                    )
                ).first()
                
                if member:
                    presence_list.append(asdict(presence))
        
        return presence_list
    
    async def send_notification_to_workspace(
        self,
        workspace_id: UUID,
        notification_data: Dict[str, Any]
    ) -> None:
        """Send notification to all workspace members"""
        message = WebSocketMessage(
            type=MessageType.NOTIFICATION.value,
            workspace_id=str(workspace_id),
            user_id="system",
            data=notification_data
        )
        
        await self._broadcast_to_workspace(workspace_id, message)
    
    async def force_sync_workspace(self, workspace_id: UUID) -> None:
        """Force synchronization of entire workspace state"""
        state = await self._get_workspace_state(workspace_id)
        
        sync_message = WebSocketMessage(
            type=MessageType.STATE_SYNC.value,
            workspace_id=str(workspace_id),
            user_id="system",
            data={"workspace_state": state}
        )
        
        await self._broadcast_to_workspace(workspace_id, sync_message)