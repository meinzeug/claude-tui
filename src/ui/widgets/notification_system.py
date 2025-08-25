#!/usr/bin/env python3
"""
Notification System Widget - Real-time notifications with different severity levels,
auto-dismiss functionality, and notification history.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from textual import work, on
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Label, Button, ListView, ListItem
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from rich.text import Text
from rich.panel import Panel
from rich.console import Console


class NotificationLevel(Enum):
    """Notification severity levels"""
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationCategory(Enum):
    """Notification categories"""
    SYSTEM = "system"
    AI_TASK = "ai_task"
    PROJECT = "project"
    VALIDATION = "validation"
    WORKFLOW = "workflow"
    USER_ACTION = "user_action"


@dataclass
class Notification:
    """Notification data structure"""
    id: str
    title: str
    message: str
    level: NotificationLevel
    category: NotificationCategory
    timestamp: datetime
    duration: Optional[int] = None  # Auto-dismiss duration in seconds
    persistent: bool = False  # Don't auto-dismiss
    actions: Optional[List[Dict[str, Any]]] = None  # Action buttons
    metadata: Optional[Dict[str, Any]] = None
    dismissed: bool = False
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.metadata is None:
            self.metadata = {}
        
        # Set default duration based on level
        if self.duration is None and not self.persistent:
            duration_map = {
                NotificationLevel.DEBUG: 3,
                NotificationLevel.INFO: 5,
                NotificationLevel.SUCCESS: 4,
                NotificationLevel.WARNING: 8,
                NotificationLevel.ERROR: 12,
                NotificationLevel.CRITICAL: 0  # Don't auto-dismiss
            }
            self.duration = duration_map.get(self.level, 5)
    
    def get_level_icon(self) -> str:
        """Get icon for notification level"""
        icons = {
            NotificationLevel.DEBUG: "🔍",
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨"
        }
        return icons.get(self.level, "🔔")
    
    def get_category_icon(self) -> str:
        """Get icon for notification category"""
        icons = {
            NotificationCategory.SYSTEM: "⚙️",
            NotificationCategory.AI_TASK: "🤖",
            NotificationCategory.PROJECT: "📁",
            NotificationCategory.VALIDATION: "🔍",
            NotificationCategory.WORKFLOW: "🔄",
            NotificationCategory.USER_ACTION: "👤"
        }
        return icons.get(self.category, "📄")
    
    def get_level_color(self) -> str:
        """Get color for notification level"""
        colors = {
            NotificationLevel.DEBUG: "dim",
            NotificationLevel.INFO: "blue",
            NotificationLevel.SUCCESS: "green",
            NotificationLevel.WARNING: "yellow",
            NotificationLevel.ERROR: "red",
            NotificationLevel.CRITICAL: "bold red"
        }
        return colors.get(self.level, "white")


class NotificationWidget(Container):
    """Individual notification display widget"""
    
    def __init__(self, notification: Notification, dismiss_callback: Optional[Callable] = None) -> None:
        super().__init__()
        self.notification = notification
        self.dismiss_callback = dismiss_callback
        self.auto_dismiss_timer: Optional[Timer] = None
        
    def compose(self):
        """Compose notification widget"""
        with Horizontal(classes="notification-container"):
            # Icon and content
            with Vertical(classes="notification-content"):
                # Header with icons and timestamp
                header_text = Text()
                header_text.append(
                    f"{self.notification.get_level_icon()} {self.notification.get_category_icon()} ",
                    style=self.notification.get_level_color()
                )
                header_text.append(self.notification.title, style="bold")
                header_text.append(
                    f" ({self.notification.timestamp.strftime('%H:%M:%S')})",
                    style="dim"
                )
                yield Static(header_text, classes="notification-header")
                
                # Message content
                yield Label(self.notification.message, classes="notification-message")
                
                # Action buttons if any
                if self.notification.actions:
                    with Horizontal(classes="notification-actions"):
                        for action in self.notification.actions:
                            yield Button(
                                action.get('label', 'Action'),
                                id=f"action-{action.get('id', 'unknown')}",
                                variant=action.get('variant', 'default')
                            )
            
            # Dismiss button
            yield Button("❌", id="dismiss", classes="dismiss-button")
    
    def on_mount(self) -> None:
        """Start auto-dismiss timer if needed"""
        if self.notification.duration and self.notification.duration > 0:
            self.auto_dismiss_timer = self.set_timer(
                self.notification.duration,
                self._auto_dismiss
            )
    
    def _auto_dismiss(self) -> None:
        """Auto-dismiss notification"""
        self.dismiss_notification()
    
    @on(Button.Pressed, "#dismiss")
    def dismiss_button_pressed(self) -> None:
        """Handle dismiss button press"""
        self.dismiss_notification()
    
    def dismiss_notification(self) -> None:
        """Dismiss this notification"""
        if self.auto_dismiss_timer:
            self.auto_dismiss_timer.stop()
        
        self.notification.dismissed = True
        
        if self.dismiss_callback:
            self.dismiss_callback(self.notification)
        
        self.remove()


class NotificationToast(Container):
    """Toast-style notification that appears at the top"""
    
    def __init__(self, notification: Notification, dismiss_callback: Optional[Callable] = None) -> None:
        super().__init__(classes="notification-toast")
        self.notification = notification
        self.dismiss_callback = dismiss_callback
        
    def compose(self):
        """Compose toast notification"""
        # Create panel with notification content
        content = Text()
        content.append(
            f"{self.notification.get_level_icon()} {self.notification.title}: ",
            style=f"bold {self.notification.get_level_color()}"
        )
        content.append(self.notification.message)
        
        panel = Panel(
            content,
            border_style=self.notification.get_level_color(),
            title=f"{self.notification.get_category_icon()} {self.notification.category.value.title()}"
        )
        
        yield Static(panel)
    
    def on_mount(self) -> None:
        """Start auto-dismiss for toast"""
        if self.notification.duration and self.notification.duration > 0:
            self.set_timer(self.notification.duration, self._auto_dismiss)
    
    def _auto_dismiss(self) -> None:
        """Auto-dismiss toast"""
        if self.dismiss_callback:
            self.dismiss_callback(self.notification)
        self.remove()


class NotificationHistory(ListView):
    """Notification history viewer"""
    
    def __init__(self, notifications: List[Notification]) -> None:
        super().__init__()
        self.notifications = notifications
        
    def populate_history(self) -> None:
        """Populate history with notifications"""
        self.clear()
        
        # Sort notifications by timestamp (newest first)
        sorted_notifications = sorted(
            self.notifications,
            key=lambda n: n.timestamp,
            reverse=True
        )
        
        for notification in sorted_notifications[:50]:  # Show last 50
            history_item = self._create_history_item(notification)
            self.append(ListItem(history_item))
    
    def _create_history_item(self, notification: Notification) -> Static:
        """Create history item widget"""
        content = Text()
        
        # Timestamp
        content.append(
            f"[{notification.timestamp.strftime('%H:%M:%S')}] ",
            style="dim"
        )
        
        # Level and category icons
        content.append(
            f"{notification.get_level_icon()}{notification.get_category_icon()} ",
            style=notification.get_level_color()
        )
        
        # Title and message
        content.append(f"{notification.title}: ", style="bold")
        content.append(notification.message[:100])  # Truncate long messages
        
        if len(notification.message) > 100:
            content.append("...", style="dim")
        
        return Static(content)
    
    def update_history(self, notifications: List[Notification]) -> None:
        """Update history with new notifications"""
        self.notifications = notifications
        self.populate_history()


class NotificationSystem(Container):
    """Main notification system widget"""
    
    notifications: reactive[List[Notification]] = reactive([])
    show_history: reactive[bool] = reactive(False)
    max_notifications: reactive[int] = reactive(5)
    
    def __init__(self) -> None:
        super().__init__(id="notification-system")
        self.notification_history: List[Notification] = []
        self.notification_counter = 0
        self.toast_container: Optional[Container] = None
        self.history_widget: Optional[NotificationHistory] = None
        
    def compose(self):
        """Compose notification system"""
        # Toast notifications container (top overlay)
        self.toast_container = Container(id="toast-container", classes="toast-overlay")
        yield self.toast_container
        
        # History panel (hidden by default)
        with Container(id="history-panel", classes="history-panel"):
            yield Label("📋 Notification History", classes="history-header")
            
            self.history_widget = NotificationHistory(self.notification_history)
            yield self.history_widget
            
            with Horizontal(classes="history-controls"):
                yield Button("🗺️ Clear History", id="clear-history")
                yield Button("💾 Export", id="export-history")
                yield Button("❌ Close", id="close-history")
    
    def on_mount(self) -> None:
        """Initialize notification system"""
        self.add_notification(
            "Notification system initialized",
            "info",
            category="system"
        )
    
    def watch_show_history(self, show: bool) -> None:
        """Toggle history panel visibility"""
        history_panel = self.query_one("#history-panel")
        if show:
            history_panel.display = True
            if self.history_widget:
                self.history_widget.populate_history()
        else:
            history_panel.display = False
    
    def add_notification(
        self,
        message: str,
        level: str = "info",
        title: Optional[str] = None,
        category: str = "system",
        duration: Optional[int] = None,
        persistent: bool = False,
        actions: Optional[List[Dict[str, Any]]] = None,
        show_toast: bool = True
    ) -> str:
        """Add new notification"""
        
        self.notification_counter += 1
        notification_id = f"notification_{self.notification_counter}"
        
        # Create notification
        notification = Notification(
            id=notification_id,
            title=title or level.title(),
            message=message,
            level=NotificationLevel(level.lower()),
            category=NotificationCategory(category.lower()),
            timestamp=datetime.now(),
            duration=duration,
            persistent=persistent,
            actions=actions or []
        )
        
        # Add to history
        self.notification_history.append(notification)
        
        # Show as toast if requested
        if show_toast:
            self._show_toast(notification)
        
        # Update reactive property
        current = list(self.notifications)
        current.append(notification)
        
        # Limit active notifications
        if len(current) > self.max_notifications:
            # Remove oldest non-persistent notifications
            current = [n for n in current if n.persistent] + current[-self.max_notifications:]
        
        self.notifications = current
        
        return notification_id
    
    def _show_toast(self, notification: Notification) -> None:
        """Show toast notification"""
        if self.toast_container:
            toast = NotificationToast(
                notification,
                dismiss_callback=self._dismiss_toast
            )
            self.toast_container.mount(toast)
    
    def _dismiss_toast(self, notification: Notification) -> None:
        """Dismiss toast notification"""
        # Remove from active notifications if present
        current = list(self.notifications)
        self.notifications = [n for n in current if n.id != notification.id]
    
    def dismiss_notification(self, notification_id: str) -> None:
        """Dismiss notification by ID"""
        current = list(self.notifications)
        self.notifications = [n for n in current if n.id != notification_id]
    
    def clear_all_notifications(self) -> None:
        """Clear all active notifications"""
        self.notifications = []
        
        # Clear toast container
        if self.toast_container:
            self.toast_container.remove_children()
    
    def get_notifications_by_level(self, level: NotificationLevel) -> List[Notification]:
        """Get notifications by level"""
        return [n for n in self.notification_history if n.level == level]
    
    def get_notifications_by_category(self, category: NotificationCategory) -> List[Notification]:
        """Get notifications by category"""
        return [n for n in self.notification_history if n.category == category]
    
    def show_notification_history(self) -> None:
        """Show notification history panel"""
        self.show_history = True
    
    def hide_notification_history(self) -> None:
        """Hide notification history panel"""
        self.show_history = False
    
    # Convenience methods for different notification types
    def notify_info(self, message: str, title: str = "Info", **kwargs) -> str:
        """Add info notification"""
        return self.add_notification(message, "info", title, **kwargs)
    
    def notify_success(self, message: str, title: str = "Success", **kwargs) -> str:
        """Add success notification"""
        return self.add_notification(message, "success", title, **kwargs)
    
    def notify_warning(self, message: str, title: str = "Warning", **kwargs) -> str:
        """Add warning notification"""
        return self.add_notification(message, "warning", title, **kwargs)
    
    def notify_error(self, message: str, title: str = "Error", **kwargs) -> str:
        """Add error notification"""
        return self.add_notification(message, "error", title, persistent=True, **kwargs)
    
    def notify_critical(self, message: str, title: str = "Critical", **kwargs) -> str:
        """Add critical notification"""
        return self.add_notification(message, "critical", title, persistent=True, **kwargs)
    
    # Event handlers
    @on(Button.Pressed, "#clear-history")
    def clear_history(self) -> None:
        """Clear notification history"""
        self.notification_history.clear()
        if self.history_widget:
            self.history_widget.update_history([])
    
    @on(Button.Pressed, "#export-history")
    def export_history(self) -> None:
        """Export notification history"""
        self.post_message(ExportHistoryMessage(self.notification_history))
    
    @on(Button.Pressed, "#close-history")
    def close_history(self) -> None:
        """Close history panel"""
        self.hide_notification_history()
    
    def export_history_to_text(self) -> str:
        """Export history as text"""
        lines = ["Claude-TIU Notification History", "=" * 50, ""]
        
        for notification in sorted(self.notification_history, key=lambda n: n.timestamp):
            lines.extend([
                f"[{notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]",
                f"Level: {notification.level.value.upper()}",
                f"Category: {notification.category.value.title()}",
                f"Title: {notification.title}",
                f"Message: {notification.message}",
                "-" * 30,
                ""
            ])
        
        return "\n".join(lines)


class ExportHistoryMessage(Message):
    """Message to export notification history"""
    
    def __init__(self, history: List[Notification]) -> None:
        super().__init__()
        self.history = history