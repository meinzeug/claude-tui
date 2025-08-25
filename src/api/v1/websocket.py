"""
WebSocket REST API Endpoints for Real-time Updates.

Provides real-time communication capabilities:
- Task progress monitoring
- Workflow execution updates
- Project status changes
- System notifications
- Agent coordination status
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum

from ..dependencies.auth import get_current_user_websocket
from ...core.exceptions import AuthenticationError, ValidationError

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Enums
class WebSocketEventType(str, Enum):
    """WebSocket event types."""
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PROGRESS = "workflow_progress"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    AGENT_SPAWNED = "agent_spawned"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    PROJECT_STATUS_CHANGED = "project_status_changed"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR_OCCURRED = "error_occurred"

class SubscriptionType(str, Enum):
    """Subscription types for filtering events."""
    ALL = "all"
    TASKS = "tasks"
    WORKFLOWS = "workflows"
    PROJECTS = "projects"
    AGENTS = "agents"
    SYSTEM = "system"

# Pydantic Models
class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    event_type: WebSocketEventType
    timestamp: str
    data: Dict
    subscription_id: Optional[str] = None
    user_id: Optional[str] = None

class SubscriptionRequest(BaseModel):
    """WebSocket subscription request."""
    subscription_type: SubscriptionType
    filters: Optional[Dict] = Field(default_factory=dict, description="Subscription filters")
    subscription_id: Optional[str] = Field(None, description="Custom subscription ID")

class TaskProgressData(BaseModel):
    """Task progress update data."""
    task_id: str
    progress_percentage: float
    current_step: Optional[str] = None
    estimated_remaining_seconds: Optional[int] = None
    status: str
    metrics: Optional[Dict] = None

class WorkflowProgressData(BaseModel):
    """Workflow progress update data."""
    workflow_id: str
    execution_id: str
    progress_percentage: float
    current_step: Optional[str] = None
    steps_completed: int
    steps_total: int
    agents_active: int
    status: str
    estimated_completion: Optional[str] = None

class AgentStatusData(BaseModel):
    """Agent status update data."""
    agent_id: str
    agent_type: str
    status: str
    current_task: Optional[str] = None
    performance_metrics: Optional[Dict] = None

class SystemNotificationData(BaseModel):
    """System notification data."""
    notification_type: str
    severity: str
    title: str
    message: str
    action_required: bool = False
    metadata: Optional[Dict] = None

# Connection Manager
class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self.subscriptions: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept connection and register user."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connection established: {connection_id} for user {user_id}")
    
    def disconnect(self, connection_id: str, user_id: str):
        """Remove connection and cleanup."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove related subscriptions
        to_remove = [sub_id for sub_id, sub_data in self.subscriptions.items() 
                    if sub_data.get('connection_id') == connection_id]
        for sub_id in to_remove:
            del self.subscriptions[sub_id]
        
        logger.info(f"WebSocket connection closed: {connection_id} for user {user_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                # Connection might be closed, remove it
                if connection_id in self.active_connections:
                    del self.active_connections[connection_id]
    
    async def send_to_user(self, message: str, user_id: str):
        """Send message to all connections of a user."""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_subscribers(self, event_type: WebSocketEventType, data: Dict, filters: Optional[Dict] = None):
        """Broadcast message to relevant subscribers."""
        message = WebSocketMessage(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            data=data
        )
        
        message_json = message.json()
        
        for sub_id, subscription in self.subscriptions.items():
            # Check if subscription matches event type
            if self._matches_subscription(event_type, subscription, data, filters):
                connection_id = subscription.get('connection_id')
                if connection_id:
                    # Add subscription ID to message
                    message_copy = message.dict()
                    message_copy['subscription_id'] = sub_id
                    await self.send_personal_message(json.dumps(message_copy), connection_id)
    
    def _matches_subscription(self, event_type: WebSocketEventType, subscription: Dict, 
                            data: Dict, filters: Optional[Dict] = None) -> bool:
        """Check if event matches subscription criteria."""
        sub_type = subscription.get('subscription_type')
        sub_filters = subscription.get('filters', {})
        
        # Check subscription type
        if sub_type == SubscriptionType.ALL:
            pass  # All events match
        elif sub_type == SubscriptionType.TASKS and not event_type.value.startswith('task_'):
            return False
        elif sub_type == SubscriptionType.WORKFLOWS and not event_type.value.startswith('workflow_'):
            return False
        elif sub_type == SubscriptionType.PROJECTS and not event_type.value.startswith('project_'):
            return False
        elif sub_type == SubscriptionType.AGENTS and not event_type.value.startswith('agent_'):
            return False
        elif sub_type == SubscriptionType.SYSTEM and event_type != WebSocketEventType.SYSTEM_NOTIFICATION:
            return False
        
        # Check additional filters
        if sub_filters:
            for key, value in sub_filters.items():
                if key in data and data[key] != value:
                    return False
        
        return True
    
    def add_subscription(self, subscription_id: str, connection_id: str, user_id: str, 
                        subscription: SubscriptionRequest):
        """Add new subscription."""
        self.subscriptions[subscription_id] = {
            'connection_id': connection_id,
            'user_id': user_id,
            'subscription_type': subscription.subscription_type.value,
            'filters': subscription.filters,
            'created_at': datetime.utcnow().isoformat()
        }
    
    def remove_subscription(self, subscription_id: str):
        """Remove subscription."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]

# Global connection manager
manager = ConnectionManager()

# WebSocket Endpoints
@router.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """
    Main WebSocket endpoint for real-time updates.
    
    Provides real-time communication for task progress,
    workflow updates, and system notifications.
    """
    user_id = None
    try:
        # Authenticate user (this would be implemented based on your auth system)
        user_id = await get_current_user_websocket(websocket)
        
        # Connect to manager
        await manager.connect(websocket, connection_id, user_id)
        
        # Send welcome message
        welcome_message = WebSocketMessage(
            event_type=WebSocketEventType.SYSTEM_NOTIFICATION,
            timestamp=datetime.utcnow().isoformat(),
            data={
                "message": "WebSocket connection established",
                "connection_id": connection_id,
                "features": [
                    "real_time_task_updates",
                    "workflow_progress_monitoring",
                    "agent_status_updates",
                    "system_notifications"
                ]
            }
        )
        await websocket.send_text(welcome_message.json())
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle subscription requests
                if message_data.get('type') == 'subscribe':
                    await handle_subscription(connection_id, user_id, message_data, websocket)
                elif message_data.get('type') == 'unsubscribe':
                    await handle_unsubscription(message_data, websocket)
                elif message_data.get('type') == 'ping':
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}))
                else:
                    # Echo back unknown message types
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Unknown message type",
                        "received": message_data
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Message processing error: {str(e)}"
                }))
                
    except AuthenticationError:
        await websocket.close(code=4001, reason="Authentication failed")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}")
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        if user_id:
            manager.disconnect(connection_id, user_id)

async def handle_subscription(connection_id: str, user_id: str, message_data: Dict, websocket: WebSocket):
    """Handle subscription request."""
    try:
        subscription_request = SubscriptionRequest(**message_data.get('data', {}))
        subscription_id = subscription_request.subscription_id or str(uuid4())
        
        manager.add_subscription(subscription_id, connection_id, user_id, subscription_request)
        
        response = {
            "type": "subscription_confirmed",
            "subscription_id": subscription_id,
            "subscription_type": subscription_request.subscription_type.value,
            "filters": subscription_request.filters
        }
        await websocket.send_text(json.dumps(response))
        
    except Exception as e:
        error_response = {
            "type": "subscription_error",
            "message": f"Failed to create subscription: {str(e)}"
        }
        await websocket.send_text(json.dumps(error_response))

async def handle_unsubscription(message_data: Dict, websocket: WebSocket):
    """Handle unsubscription request."""
    try:
        subscription_id = message_data.get('subscription_id')
        if subscription_id:
            manager.remove_subscription(subscription_id)
            
            response = {
                "type": "unsubscription_confirmed",
                "subscription_id": subscription_id
            }
            await websocket.send_text(json.dumps(response))
        else:
            raise ValueError("subscription_id is required")
            
    except Exception as e:
        error_response = {
            "type": "unsubscription_error",
            "message": f"Failed to remove subscription: {str(e)}"
        }
        await websocket.send_text(json.dumps(error_response))

# HTTP Endpoints for WebSocket Management
@router.get("/connections")
async def get_active_connections(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get information about active WebSocket connections.
    
    Returns statistics about active connections and subscriptions.
    """
    try:
        user_id = current_user["id"]
        user_connections = manager.user_connections.get(user_id, set())
        user_subscriptions = [
            {
                "subscription_id": sub_id,
                "subscription_type": sub_data.get("subscription_type"),
                "filters": sub_data.get("filters", {}),
                "created_at": sub_data.get("created_at")
            }
            for sub_id, sub_data in manager.subscriptions.items()
            if sub_data.get("user_id") == user_id
        ]
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "active_connections": len(user_connections),
                "connection_ids": list(user_connections),
                "subscriptions": user_subscriptions,
                "total_system_connections": len(manager.active_connections),
                "total_system_subscriptions": len(manager.subscriptions)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get connection info: {str(e)}"
        )

@router.post("/broadcast")
async def broadcast_message(
    event_type: WebSocketEventType,
    data: Dict,
    filters: Optional[Dict] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Broadcast message to WebSocket subscribers.
    
    Sends a message to all relevant WebSocket connections
    based on subscription filters.
    """
    try:
        await manager.broadcast_to_subscribers(event_type, data, filters)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Broadcast sent successfully",
                "event_type": event_type.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to broadcast message: {str(e)}"
        )

@router.post("/send-to-user/{user_id}")
async def send_message_to_user(
    user_id: str,
    event_type: WebSocketEventType,
    data: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """
    Send message to specific user's WebSocket connections.
    
    Sends a direct message to all active connections of a specific user.
    """
    try:
        message = WebSocketMessage(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            user_id=user_id
        )
        
        await manager.send_to_user(message.json(), user_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Message sent to user successfully",
                "target_user_id": user_id,
                "event_type": event_type.value,
                "timestamp": message.timestamp
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message to user: {str(e)}"
        )

# Utility Functions for Broadcasting Events
async def broadcast_task_progress(task_id: str, progress_data: TaskProgressData):
    """Broadcast task progress update."""
    await manager.broadcast_to_subscribers(
        WebSocketEventType.TASK_PROGRESS,
        progress_data.dict(),
        {"task_id": task_id}
    )

async def broadcast_workflow_progress(workflow_id: str, progress_data: WorkflowProgressData):
    """Broadcast workflow progress update."""
    await manager.broadcast_to_subscribers(
        WebSocketEventType.WORKFLOW_PROGRESS,
        progress_data.dict(),
        {"workflow_id": workflow_id}
    )

async def broadcast_agent_status(agent_id: str, status_data: AgentStatusData):
    """Broadcast agent status update."""
    await manager.broadcast_to_subscribers(
        WebSocketEventType.AGENT_STATUS_CHANGED,
        status_data.dict(),
        {"agent_id": agent_id}
    )

async def broadcast_system_notification(notification_data: SystemNotificationData):
    """Broadcast system notification."""
    await manager.broadcast_to_subscribers(
        WebSocketEventType.SYSTEM_NOTIFICATION,
        notification_data.dict()
    )

# Health Check
@router.get("/health")
async def websocket_health():
    """WebSocket service health check."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": "websocket",
            "active_connections": len(manager.active_connections),
            "active_users": len(manager.user_connections),
            "active_subscriptions": len(manager.subscriptions),
            "supported_events": [event.value for event in WebSocketEventType],
            "subscription_types": [sub_type.value for sub_type in SubscriptionType],
            "checked_at": datetime.utcnow().isoformat()
        }
    )