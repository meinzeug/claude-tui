"""
Communication Hub - Integrated team communication and notification system
Handles comments, discussions, notifications, and team messaging
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from .models import (
    Workspace, WorkspaceMember, Comment, Notification,
    ActivityFeed, ActivityType
)
from ..database.models import User
from .presence_manager import PresenceManager
from .sync_engine import SynchronizationEngine

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationType(Enum):
    """Types of notifications"""
    MENTION = "mention"
    TASK_ASSIGNMENT = "task_assignment"
    COMMENT_REPLY = "comment_reply"
    WORKSPACE_INVITATION = "workspace_invitation"
    SYSTEM_ALERT = "system_alert"
    DEADLINE_REMINDER = "deadline_reminder"
    CONFLICT_ALERT = "conflict_alert"
    APPROVAL_REQUEST = "approval_request"


class MessageType(Enum):
    """Message types for communication"""
    TEXT = "text"
    CODE = "code"
    FILE_REFERENCE = "file_reference"
    TASK_REFERENCE = "task_reference"
    EMOJI_REACTION = "emoji_reaction"


@dataclass
class Message:
    """Communication message"""
    id: str
    sender_id: str
    content: str
    message_type: str
    timestamp: str
    thread_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    mentions: Optional[List[str]] = None
    attachments: Optional[List[Dict[str, Any]]] = None


@dataclass
class Thread:
    """Message thread for discussions"""
    id: str
    title: str
    workspace_id: str
    creator_id: str
    created_at: str
    participants: List[str]
    message_count: int
    last_activity: str
    is_resolved: bool = False
    tags: Optional[List[str]] = None


class CommunicationHub:
    """
    Integrated communication hub for team collaboration.
    Handles comments, notifications, discussions, and real-time messaging.
    """
    
    def __init__(
        self,
        db_session: Session,
        presence_manager: PresenceManager,
        sync_engine: Optional[SynchronizationEngine] = None
    ):
        """
        Initialize communication hub.
        
        Args:
            db_session: Database session
            presence_manager: Presence tracking system
            sync_engine: Real-time synchronization engine
        """
        self.db = db_session
        self.presence_manager = presence_manager
        self.sync_engine = sync_engine
        
        # Message processing
        self._message_processors: Dict[str, Callable] = {
            MessageType.TEXT.value: self._process_text_message,
            MessageType.CODE.value: self._process_code_message,
            MessageType.FILE_REFERENCE.value: self._process_file_reference,
            MessageType.TASK_REFERENCE.value: self._process_task_reference,
            MessageType.EMOJI_REACTION.value: self._process_emoji_reaction
        }
        
        # Notification handlers
        self._notification_handlers: Dict[str, Callable] = {
            NotificationType.MENTION.value: self._handle_mention_notification,
            NotificationType.TASK_ASSIGNMENT.value: self._handle_task_assignment_notification,
            NotificationType.COMMENT_REPLY.value: self._handle_comment_reply_notification,
            NotificationType.DEADLINE_REMINDER.value: self._handle_deadline_notification,
            NotificationType.CONFLICT_ALERT.value: self._handle_conflict_notification
        }
        
        # Active message threads
        self._active_threads: Dict[UUID, Thread] = {}
        
        # Notification queues
        self._notification_queues: Dict[UUID, List[Notification]] = {}
        
        logger.info("Communication hub initialized")
    
    async def create_comment(
        self,
        workspace_id: UUID,
        user_id: UUID,
        content: str,
        target_type: str,
        target_id: Optional[str] = None,
        parent_comment_id: Optional[UUID] = None,
        line_number: Optional[int] = None
    ) -> Comment:
        """
        Create new comment or reply.
        
        Args:
            workspace_id: Workspace ID
            user_id: User creating comment
            content: Comment content
            target_type: Type of target (file, task, general, etc.)
            target_id: Target resource ID
            parent_comment_id: Parent comment for replies
            line_number: Line number for file comments
            
        Returns:
            Created comment
        """
        logger.info(f"Creating comment in workspace {workspace_id}")
        
        # Validate user is workspace member
        member = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.is_active == True
            )
        ).first()
        
        if not member:
            raise ValueError("User is not a member of this workspace")
        
        # Create comment
        comment = Comment(
            workspace_id=workspace_id,
            user_id=user_id,
            content=content,
            target_type=target_type,
            target_id=target_id,
            parent_id=parent_comment_id,
            line_number=line_number
        )
        
        # Set thread root
        if parent_comment_id:
            parent = self.db.query(Comment).filter(Comment.id == parent_comment_id).first()
            if parent:
                comment.thread_root_id = parent.thread_root_id or parent.id
        else:
            comment.thread_root_id = comment.id
        
        self.db.add(comment)
        self.db.flush()  # Get comment ID
        
        # Process mentions in content
        mentions = await self._extract_mentions(content)
        if mentions:
            await self._send_mention_notifications(
                workspace_id, user_id, comment.id, mentions
            )
        
        # Create activity
        user = self.db.query(User).filter(User.id == user_id).first()
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=user_id,
            activity_type=ActivityType.COMMENT_ADDED.value,
            description=f"{user.username} added a comment",
            target_resource=str(comment.id),
            metadata={
                'comment_id': str(comment.id),
                'target_type': target_type,
                'target_id': target_id,
                'is_reply': parent_comment_id is not None
            },
            is_public=True
        )
        self.db.add(activity)
        
        # Send notifications for replies
        if parent_comment_id:
            await self._send_reply_notifications(workspace_id, user_id, comment)
        
        self.db.commit()
        
        # Broadcast real-time update
        if self.sync_engine:
            await self.sync_engine.send_notification_to_workspace(
                workspace_id,
                {
                    'type': 'comment_added',
                    'comment_id': str(comment.id),
                    'user_id': str(user_id),
                    'content': content,
                    'target_type': target_type,
                    'target_id': target_id
                }
            )
        
        logger.info(f"Comment {comment.id} created successfully")
        return comment
    
    async def create_discussion_thread(
        self,
        workspace_id: UUID,
        creator_id: UUID,
        title: str,
        initial_message: str,
        tags: Optional[List[str]] = None
    ) -> Thread:
        """
        Create new discussion thread.
        
        Args:
            workspace_id: Workspace ID
            creator_id: User creating thread
            title: Thread title
            initial_message: Initial message content
            tags: Optional tags for categorization
            
        Returns:
            Created thread
        """
        logger.info(f"Creating discussion thread '{title}' in workspace {workspace_id}")
        
        # Create thread
        thread_id = str(uuid4())
        thread = Thread(
            id=thread_id,
            title=title,
            workspace_id=str(workspace_id),
            creator_id=str(creator_id),
            created_at=datetime.now(timezone.utc).isoformat(),
            participants=[str(creator_id)],
            message_count=1,
            last_activity=datetime.now(timezone.utc).isoformat(),
            tags=tags or []
        )
        
        # Store thread
        self._active_threads[UUID(thread_id)] = thread
        
        # Create initial comment as thread starter
        initial_comment = await self.create_comment(
            workspace_id=workspace_id,
            user_id=creator_id,
            content=initial_message,
            target_type='discussion',
            target_id=thread_id
        )
        
        # Create activity
        user = self.db.query(User).filter(User.id == creator_id).first()
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=creator_id,
            activity_type='discussion_created',
            description=f"{user.username} started discussion: {title}",
            metadata={
                'thread_id': thread_id,
                'title': title,
                'tags': tags
            },
            is_public=True,
            notify_team=True
        )
        self.db.add(activity)
        self.db.commit()
        
        logger.info(f"Discussion thread {thread_id} created successfully")
        return thread
    
    async def send_notification(
        self,
        workspace_id: UUID,
        recipient_id: UUID,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        action_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        Send notification to user.
        
        Args:
            workspace_id: Workspace ID
            recipient_id: Recipient user ID
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            priority: Notification priority
            action_url: Optional action URL
            metadata: Additional metadata
            
        Returns:
            Created notification
        """
        logger.debug(f"Sending {notification_type.value} notification to user {recipient_id}")
        
        # Create notification
        notification = Notification(
            workspace_id=workspace_id,
            recipient_id=recipient_id,
            title=title,
            message=message,
            notification_type=notification_type.value,
            priority=priority.value,
            action_url=action_url,
            metadata=metadata or {}
        )
        
        self.db.add(notification)
        
        # Add to queue for batch processing
        if recipient_id not in self._notification_queues:
            self._notification_queues[recipient_id] = []
        self._notification_queues[recipient_id].append(notification)
        
        # Handle notification based on type
        handler = self._notification_handlers.get(notification_type.value)
        if handler:
            await handler(workspace_id, notification)
        
        self.db.commit()
        
        # Send real-time notification
        await self._send_real_time_notification(workspace_id, recipient_id, notification)
        
        return notification
    
    async def send_message(
        self,
        workspace_id: UUID,
        sender_id: UUID,
        recipients: List[UUID],
        content: str,
        message_type: MessageType = MessageType.TEXT,
        thread_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Send message to users or thread.
        
        Args:
            workspace_id: Workspace ID
            sender_id: Sender user ID
            recipients: List of recipient user IDs
            content: Message content
            message_type: Type of message
            thread_id: Optional thread ID
            metadata: Additional metadata
            
        Returns:
            Created message
        """
        logger.info(f"Sending {message_type.value} message from user {sender_id}")
        
        # Create message
        message = Message(
            id=str(uuid4()),
            sender_id=str(sender_id),
            content=content,
            message_type=message_type.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            thread_id=str(thread_id) if thread_id else None,
            metadata=metadata,
            mentions=await self._extract_mentions(content)
        )
        
        # Process message based on type
        processor = self._message_processors.get(message_type.value)
        if processor:
            message = await processor(workspace_id, message)
        
        # Update thread if applicable
        if thread_id and thread_id in self._active_threads:
            thread = self._active_threads[thread_id]
            thread.message_count += 1
            thread.last_activity = message.timestamp
            
            # Add sender to participants if not already
            if str(sender_id) not in thread.participants:
                thread.participants.append(str(sender_id))
        
        # Send notifications to recipients
        for recipient_id in recipients:
            if recipient_id != sender_id:  # Don't notify sender
                await self.send_notification(
                    workspace_id=workspace_id,
                    recipient_id=recipient_id,
                    notification_type=NotificationType.MENTION,
                    title="New Message",
                    message=f"You have a new message: {content[:100]}...",
                    priority=NotificationPriority.MEDIUM,
                    metadata={
                        'message_id': message.id,
                        'sender_id': str(sender_id)
                    }
                )
        
        # Broadcast to workspace
        if self.sync_engine:
            await self.sync_engine.send_notification_to_workspace(
                workspace_id,
                {
                    'type': 'message_sent',
                    'message': {
                        'id': message.id,
                        'sender_id': message.sender_id,
                        'content': message.content,
                        'message_type': message.message_type,
                        'timestamp': message.timestamp,
                        'thread_id': message.thread_id
                    }
                }
            )
        
        logger.info(f"Message {message.id} sent successfully")
        return message
    
    async def get_comments(
        self,
        workspace_id: UUID,
        requester_id: UUID,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        include_resolved: bool = True,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get comments for workspace or specific target.
        
        Args:
            workspace_id: Workspace ID
            requester_id: User requesting comments
            target_type: Optional filter by target type
            target_id: Optional filter by target ID
            include_resolved: Include resolved comments
            limit: Maximum number of comments
            
        Returns:
            List of comment data
        """
        # Build query
        query = self.db.query(Comment, User).join(
            User, Comment.user_id == User.id
        ).filter(Comment.workspace_id == workspace_id)
        
        if target_type:
            query = query.filter(Comment.target_type == target_type)
        
        if target_id:
            query = query.filter(Comment.target_id == target_id)
        
        if not include_resolved:
            query = query.filter(Comment.is_resolved == False)
        
        comments = query.order_by(desc(Comment.created_at)).limit(limit).all()
        
        # Format response
        comment_list = []
        for comment, user in comments:
            comment_data = {
                'id': str(comment.id),
                'content': comment.content,
                'target_type': comment.target_type,
                'target_id': comment.target_id,
                'line_number': comment.line_number,
                'is_resolved': comment.is_resolved,
                'created_at': comment.created_at.isoformat(),
                'updated_at': comment.updated_at.isoformat(),
                'user': {
                    'id': str(user.id),
                    'username': user.username,
                    'full_name': user.full_name
                },
                'thread_info': {
                    'parent_id': str(comment.parent_id) if comment.parent_id else None,
                    'thread_root_id': str(comment.thread_root_id) if comment.thread_root_id else None
                }
            }
            
            # Add resolver info if resolved
            if comment.is_resolved and comment.resolved_by:
                resolver = self.db.query(User).filter(User.id == comment.resolved_by).first()
                if resolver:
                    comment_data['resolved_by'] = {
                        'id': str(resolver.id),
                        'username': resolver.username
                    }
                    comment_data['resolved_at'] = comment.resolved_at.isoformat()
            
            comment_list.append(comment_data)
        
        return comment_list
    
    async def get_notifications(
        self,
        workspace_id: UUID,
        user_id: UUID,
        unread_only: bool = False,
        notification_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get notifications for user.
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
            unread_only: Only return unread notifications
            notification_types: Filter by notification types
            limit: Maximum number of notifications
            
        Returns:
            List of notification data
        """
        query = self.db.query(Notification).filter(
            and_(
                Notification.workspace_id == workspace_id,
                Notification.recipient_id == user_id
            )
        )
        
        if unread_only:
            query = query.filter(Notification.is_read == False)
        
        if notification_types:
            query = query.filter(Notification.notification_type.in_(notification_types))
        
        notifications = query.order_by(desc(Notification.created_at)).limit(limit).all()
        
        return [
            {
                'id': str(notification.id),
                'title': notification.title,
                'message': notification.message,
                'notification_type': notification.notification_type,
                'priority': notification.priority,
                'is_read': notification.is_read,
                'is_dismissed': notification.is_dismissed,
                'action_url': notification.action_url,
                'metadata': notification.metadata,
                'created_at': notification.created_at.isoformat(),
                'read_at': notification.read_at.isoformat() if notification.read_at else None
            }
            for notification in notifications
        ]
    
    async def mark_notification_read(
        self,
        workspace_id: UUID,
        user_id: UUID,
        notification_id: UUID
    ) -> bool:
        """
        Mark notification as read.
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
            notification_id: Notification ID
            
        Returns:
            True if marked successfully
        """
        notification = self.db.query(Notification).filter(
            and_(
                Notification.id == notification_id,
                Notification.workspace_id == workspace_id,
                Notification.recipient_id == user_id
            )
        ).first()
        
        if not notification:
            return False
        
        notification.is_read = True
        notification.read_at = datetime.now(timezone.utc)
        self.db.commit()
        
        return True
    
    async def resolve_comment(
        self,
        workspace_id: UUID,
        comment_id: UUID,
        resolver_id: UUID
    ) -> bool:
        """
        Mark comment as resolved.
        
        Args:
            workspace_id: Workspace ID
            comment_id: Comment ID
            resolver_id: User resolving the comment
            
        Returns:
            True if resolved successfully
        """
        comment = self.db.query(Comment).filter(
            and_(
                Comment.id == comment_id,
                Comment.workspace_id == workspace_id
            )
        ).first()
        
        if not comment:
            return False
        
        comment.is_resolved = True
        comment.resolved_by = resolver_id
        comment.resolved_at = datetime.now(timezone.utc)
        
        # Create activity
        resolver = self.db.query(User).filter(User.id == resolver_id).first()
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=resolver_id,
            activity_type='comment_resolved',
            description=f"{resolver.username} resolved a comment",
            target_resource=str(comment_id),
            metadata={'comment_id': str(comment_id)},
            is_public=True
        )
        self.db.add(activity)
        
        self.db.commit()
        
        # Notify comment author if different from resolver
        if comment.user_id != resolver_id:
            await self.send_notification(
                workspace_id=workspace_id,
                recipient_id=comment.user_id,
                notification_type=NotificationType.SYSTEM_ALERT,
                title="Comment Resolved",
                message=f"Your comment was resolved by {resolver.username}",
                priority=NotificationPriority.LOW,
                metadata={'comment_id': str(comment_id)}
            )
        
        return True
    
    async def get_discussion_threads(
        self,
        workspace_id: UUID,
        requester_id: UUID,
        tags: Optional[List[str]] = None,
        resolved_filter: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get discussion threads for workspace"""
        threads = []
        
        for thread_id, thread in self._active_threads.items():
            if thread.workspace_id != str(workspace_id):
                continue
            
            # Apply filters
            if tags and not any(tag in thread.tags for tag in tags):
                continue
            
            if resolved_filter is not None and thread.is_resolved != resolved_filter:
                continue
            
            threads.append({
                'id': thread.id,
                'title': thread.title,
                'creator_id': thread.creator_id,
                'created_at': thread.created_at,
                'participants': thread.participants,
                'message_count': thread.message_count,
                'last_activity': thread.last_activity,
                'is_resolved': thread.is_resolved,
                'tags': thread.tags
            })
        
        return sorted(threads, key=lambda t: t['last_activity'], reverse=True)
    
    # Message processors
    
    async def _process_text_message(
        self,
        workspace_id: UUID,
        message: Message
    ) -> Message:
        """Process text message"""
        # Extract mentions and format
        message.mentions = await self._extract_mentions(message.content)
        return message
    
    async def _process_code_message(
        self,
        workspace_id: UUID,
        message: Message
    ) -> Message:
        """Process code message"""
        # Add syntax highlighting metadata
        if not message.metadata:
            message.metadata = {}
        message.metadata['language'] = 'auto-detect'
        return message
    
    async def _process_file_reference(
        self,
        workspace_id: UUID,
        message: Message
    ) -> Message:
        """Process file reference message"""
        # Validate file exists and add metadata
        if not message.metadata:
            message.metadata = {}
        message.metadata['reference_type'] = 'file'
        return message
    
    async def _process_task_reference(
        self,
        workspace_id: UUID,
        message: Message
    ) -> Message:
        """Process task reference message"""
        # Validate task exists and add metadata
        if not message.metadata:
            message.metadata = {}
        message.metadata['reference_type'] = 'task'
        return message
    
    async def _process_emoji_reaction(
        self,
        workspace_id: UUID,
        message: Message
    ) -> Message:
        """Process emoji reaction"""
        # Validate emoji and add reaction metadata
        if not message.metadata:
            message.metadata = {}
        message.metadata['reaction_type'] = 'emoji'
        return message
    
    # Notification handlers
    
    async def _handle_mention_notification(
        self,
        workspace_id: UUID,
        notification: Notification
    ) -> None:
        """Handle mention notification"""
        # Could integrate with external notification services
        pass
    
    async def _handle_task_assignment_notification(
        self,
        workspace_id: UUID,
        notification: Notification
    ) -> None:
        """Handle task assignment notification"""
        pass
    
    async def _handle_comment_reply_notification(
        self,
        workspace_id: UUID,
        notification: Notification
    ) -> None:
        """Handle comment reply notification"""
        pass
    
    async def _handle_deadline_notification(
        self,
        workspace_id: UUID,
        notification: Notification
    ) -> None:
        """Handle deadline notification"""
        pass
    
    async def _handle_conflict_notification(
        self,
        workspace_id: UUID,
        notification: Notification
    ) -> None:
        """Handle conflict notification"""
        pass
    
    # Helper methods
    
    async def _extract_mentions(self, content: str) -> List[str]:
        """Extract @mentions from content"""
        import re
        
        # Simple mention extraction - @username pattern
        mention_pattern = r'@(\w+)'
        mentions = re.findall(mention_pattern, content)
        
        # Validate mentions are actual users (would need user lookup)
        return mentions
    
    async def _send_mention_notifications(
        self,
        workspace_id: UUID,
        sender_id: UUID,
        comment_id: UUID,
        mentions: List[str]
    ) -> None:
        """Send notifications for mentions"""
        sender = self.db.query(User).filter(User.id == sender_id).first()
        
        for mention in mentions:
            # Find mentioned user
            mentioned_user = self.db.query(User).filter(User.username == mention).first()
            if mentioned_user:
                await self.send_notification(
                    workspace_id=workspace_id,
                    recipient_id=mentioned_user.id,
                    notification_type=NotificationType.MENTION,
                    title=f"Mentioned by {sender.username}",
                    message=f"You were mentioned in a comment",
                    priority=NotificationPriority.MEDIUM,
                    metadata={
                        'comment_id': str(comment_id),
                        'sender_id': str(sender_id)
                    }
                )
    
    async def _send_reply_notifications(
        self,
        workspace_id: UUID,
        sender_id: UUID,
        comment: Comment
    ) -> None:
        """Send notifications for comment replies"""
        if not comment.parent_id:
            return
        
        parent_comment = self.db.query(Comment).filter(Comment.id == comment.parent_id).first()
        if parent_comment and parent_comment.user_id != sender_id:
            sender = self.db.query(User).filter(User.id == sender_id).first()
            
            await self.send_notification(
                workspace_id=workspace_id,
                recipient_id=parent_comment.user_id,
                notification_type=NotificationType.COMMENT_REPLY,
                title=f"Reply from {sender.username}",
                message=f"{sender.username} replied to your comment",
                priority=NotificationPriority.MEDIUM,
                metadata={
                    'comment_id': str(comment.id),
                    'parent_comment_id': str(comment.parent_id),
                    'sender_id': str(sender_id)
                }
            )
    
    async def _send_real_time_notification(
        self,
        workspace_id: UUID,
        recipient_id: UUID,
        notification: Notification
    ) -> None:
        """Send real-time notification via websocket"""
        if self.sync_engine:
            await self.sync_engine.send_notification_to_workspace(
                workspace_id,
                {
                    'type': 'notification',
                    'recipient_id': str(recipient_id),
                    'notification': {
                        'id': str(notification.id),
                        'title': notification.title,
                        'message': notification.message,
                        'notification_type': notification.notification_type,
                        'priority': notification.priority,
                        'created_at': notification.created_at.isoformat()
                    }
                }
            )
    
    async def get_communication_analytics(
        self,
        workspace_id: UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get communication analytics for workspace"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Comment statistics
        comments = self.db.query(Comment).filter(
            and_(
                Comment.workspace_id == workspace_id,
                Comment.created_at >= cutoff_date
            )
        ).all()
        
        # Notification statistics
        notifications = self.db.query(Notification).filter(
            and_(
                Notification.workspace_id == workspace_id,
                Notification.created_at >= cutoff_date
            )
        ).all()
        
        # Calculate metrics
        total_comments = len(comments)
        total_notifications = len(notifications)
        
        # Comment distribution by type
        comment_types = {}
        for comment in comments:
            comment_types[comment.target_type] = comment_types.get(comment.target_type, 0) + 1
        
        # Notification distribution
        notification_types = {}
        for notification in notifications:
            notification_types[notification.notification_type] = notification_types.get(notification.notification_type, 0) + 1
        
        # User participation
        comment_users = set(comment.user_id for comment in comments)
        participation_rate = len(comment_users)
        
        return {
            'period_days': days,
            'total_comments': total_comments,
            'total_notifications': total_notifications,
            'comment_distribution': comment_types,
            'notification_distribution': notification_types,
            'user_participation': participation_rate,
            'avg_comments_per_day': round(total_comments / days, 2),
            'resolved_comments': len([c for c in comments if c.is_resolved]),
            'active_threads': len(self._active_threads)
        }