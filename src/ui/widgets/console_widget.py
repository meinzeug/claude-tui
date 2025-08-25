#!/usr/bin/env python3
"""
Console Widget - AI interaction interface with rich logging,
command history, and real-time AI communication.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from textual import on, work, events
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Input, RichLog, Button, Static, Label
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.console import Console
from rich.table import Table


class MessageType(Enum):
    """Types of console messages"""
    USER = "user"
    AI = "ai"
    SYSTEM = "system"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    DEBUG = "debug"


class AITaskStatus(Enum):
    """Status of AI tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConsoleMessage:
    """Console message data structure"""
    content: str
    message_type: MessageType
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AITask:
    """AI task tracking"""
    id: str
    prompt: str
    status: AITaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class ConsoleInput(Input):
    """Enhanced input widget with command history and autocomplete"""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(placeholder="Enter AI command or task...", **kwargs)
        self.command_history: List[str] = []
        self.history_index = -1
        self.common_commands = [
            "create project",
            "implement feature",
            "fix bugs",
            "add tests",
            "review code",
            "optimize performance",
            "update documentation",
            "refactor code",
            "deploy application",
            "analyze codebase"
        ]
    
    def on_key(self, event: events.Key) -> None:
        """Handle key events for history navigation"""
        if event.key == "up":
            self._navigate_history(-1)
            event.prevent_default()
        elif event.key == "down":
            self._navigate_history(1)
            event.prevent_default()
        elif event.key == "tab":
            self._autocomplete()
            event.prevent_default()
        else:
            super().on_key(event)
    
    def _navigate_history(self, direction: int) -> None:
        """Navigate command history"""
        if not self.command_history:
            return
        
        self.history_index = max(-1, min(
            len(self.command_history) - 1,
            self.history_index + direction
        ))
        
        if self.history_index >= 0:
            self.value = self.command_history[self.history_index]
        else:
            self.value = ""
    
    def _autocomplete(self) -> None:
        """Simple autocomplete for common commands"""
        current_text = self.value.lower()
        if not current_text:
            return
        
        # Find matching commands
        matches = [
            cmd for cmd in self.common_commands 
            if cmd.startswith(current_text)
        ]
        
        if matches:
            self.value = matches[0]
    
    def add_to_history(self, command: str) -> None:
        """Add command to history"""
        if command and command != self.command_history[-1:][0] if self.command_history else True:
            self.command_history.append(command)
            self.history_index = -1
            
            # Keep history size manageable
            if len(self.command_history) > 100:
                self.command_history = self.command_history[-100:]


class AITaskTracker(Static):
    """Widget to track active AI tasks"""
    
    def __init__(self) -> None:
        super().__init__()
        self.active_tasks: Dict[str, AITask] = {}
    
    def render(self) -> Panel:
        """Render active tasks"""
        if not self.active_tasks:
            content = Text("No active AI tasks", style="dim")
        else:
            table = Table("Task", "Status", "Time", show_header=True, header_style="bold blue")
            
            for task in self.active_tasks.values():
                # Format time
                elapsed = datetime.now() - task.created_at
                time_str = f"{elapsed.seconds}s"
                if elapsed.seconds > 60:
                    time_str = f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"
                
                # Status with icon
                status_icons = {
                    AITaskStatus.PENDING: "⏳",
                    AITaskStatus.IN_PROGRESS: "🔄",
                    AITaskStatus.COMPLETED: "✅",
                    AITaskStatus.FAILED: "❌",
                    AITaskStatus.CANCELLED: "🚫"
                }
                
                status_text = f"{status_icons.get(task.status, '❓')} {task.status.value}"
                
                # Truncate long prompts
                prompt = task.prompt[:30] + "..." if len(task.prompt) > 30 else task.prompt
                
                table.add_row(prompt, status_text, time_str)
            
            content = table
        
        return Panel(content, title="🤖 Active AI Tasks", border_style="green")
    
    def add_task(self, task: AITask) -> None:
        """Add new task to tracker"""
        self.active_tasks[task.id] = task
        self.refresh()
    
    def update_task_status(self, task_id: str, status: AITaskStatus, result: Optional[str] = None, error: Optional[str] = None) -> None:
        """Update task status"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = status
            
            if status == AITaskStatus.IN_PROGRESS and not task.started_at:
                task.started_at = datetime.now()
            elif status in [AITaskStatus.COMPLETED, AITaskStatus.FAILED, AITaskStatus.CANCELLED]:
                task.completed_at = datetime.now()
                
                if result:
                    task.result = result
                if error:
                    task.error = error
                    
                # Remove completed/failed tasks after a delay
                asyncio.create_task(self._remove_task_after_delay(task_id, 30))
            
            self.refresh()
    
    async def _remove_task_after_delay(self, task_id: str, delay_seconds: int) -> None:
        """Remove task after delay"""
        await asyncio.sleep(delay_seconds)
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            self.refresh()


class ConsoleWidget(Vertical):
    """Main console widget for AI interaction"""
    
    is_ai_busy: reactive[bool] = reactive(False)
    
    def __init__(self, ai_interface) -> None:
        super().__init__()
        self.ai_interface = ai_interface
        self.console_log: Optional[RichLog] = None
        self.input_widget: Optional[ConsoleInput] = None
        self.task_tracker: Optional[AITaskTracker] = None
        self.status_label: Optional[Label] = None
        self.message_history: List[ConsoleMessage] = []
        self.current_task_id: Optional[str] = None
        
    def compose(self) -> ComposeResult:
        """Compose console widget"""
        yield Label("💬 AI Console", classes="header")
        
        # Task tracker
        self.task_tracker = AITaskTracker()
        yield self.task_tracker
        
        # Console log with rich formatting
        self.console_log = RichLog(
            highlight=True,
            markup=True,
            auto_scroll=True,
            max_lines=1000
        )
        yield self.console_log
        
        # Status bar
        self.status_label = Label("Ready for AI commands", classes="status-bar")
        yield self.status_label
        
        # Input area
        with Horizontal(classes="input-area"):
            self.input_widget = ConsoleInput()
            yield self.input_widget
            
            yield Button("🚀 Execute", id="execute-ai-command")
            yield Button("📝 Templates", id="command-templates")
            yield Button("🗺️ Clear", id="clear-console")
    
    def on_mount(self) -> None:
        """Initialize console after mounting"""
        self.add_system_message("AI Console initialized. Enter commands or tasks for AI execution.")
        
        # Set input focus
        if self.input_widget:
            self.input_widget.focus()
    
    def watch_is_ai_busy(self, busy: bool) -> None:
        """React to AI busy status changes"""
        if self.status_label:
            if busy:
                self.status_label.update("🔄 AI is working...")
            else:
                self.status_label.update("Ready for AI commands")
        
        # Disable/enable input
        if self.input_widget:
            self.input_widget.disabled = busy
    
    @on(Input.Submitted)
    def handle_input_submit(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        command = event.value.strip()
        if command:
            self.execute_ai_command(command)
    
    @on(Button.Pressed, "#execute-ai-command")
    def execute_button_pressed(self) -> None:
        """Handle execute button press"""
        if self.input_widget:
            command = self.input_widget.value.strip()
            if command:
                self.execute_ai_command(command)
    
    @on(Button.Pressed, "#command-templates")
    def show_command_templates(self) -> None:
        """Show command templates"""
        self.post_message(ShowCommandTemplatesMessage())
    
    @on(Button.Pressed, "#clear-console")
    def clear_console(self) -> None:
        """Clear console log"""
        if self.console_log:
            self.console_log.clear()
        self.message_history.clear()
        self.add_system_message("Console cleared")
    
    def execute_ai_command(self, command: str) -> None:
        """Execute AI command"""
        if self.is_ai_busy:
            self.add_error_message("AI is currently busy. Please wait for the current task to complete.")
            return
        
        # Add to history
        if self.input_widget:
            self.input_widget.add_to_history(command)
            self.input_widget.value = ""
        
        # Log user command
        self.add_user_message(command)
        
        # Execute command asynchronously
        self.execute_ai_task_async(command)
    
    @work(exclusive=True)
    async def execute_ai_task_async(self, command: str) -> None:
        """Execute AI task asynchronously"""
        task_id = f"task_{datetime.now().timestamp()}"
        self.current_task_id = task_id
        
        # Create task
        task = AITask(
            id=task_id,
            prompt=command,
            status=AITaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            # Update UI state
            self.is_ai_busy = True
            
            if self.task_tracker:
                self.task_tracker.add_task(task)
            
            self.add_system_message(f"Starting AI task: {command}")
            
            # Update task to in progress
            if self.task_tracker:
                self.task_tracker.update_task_status(task_id, AITaskStatus.IN_PROGRESS)
            
            # Execute AI command
            result = await self.ai_interface.execute_command(
                command=command,
                context={
                    'task_id': task_id,
                    'timestamp': datetime.now().isoformat(),
                    'console_history': len(self.message_history)
                }
            )
            
            # Handle successful result
            if result:
                self.add_ai_message(result, task_id=task_id)
                
                if self.task_tracker:
                    self.task_tracker.update_task_status(
                        task_id, AITaskStatus.COMPLETED, result=result
                    )
                
                self.add_success_message(f"Task completed successfully: {task_id}")
            else:
                self.add_warning_message("AI command completed but returned no result")
                
                if self.task_tracker:
                    self.task_tracker.update_task_status(
                        task_id, AITaskStatus.COMPLETED
                    )
        
        except Exception as e:
            error_msg = f"AI task failed: {str(e)}"
            self.add_error_message(error_msg)
            
            if self.task_tracker:
                self.task_tracker.update_task_status(
                    task_id, AITaskStatus.FAILED, error=error_msg
                )
        
        finally:
            self.is_ai_busy = False
            self.current_task_id = None
    
    # Message helper methods
    def add_message(self, content: str, message_type: MessageType, task_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message to console"""
        message = ConsoleMessage(
            content=content,
            message_type=message_type,
            timestamp=datetime.now(),
            metadata=metadata or {},
            task_id=task_id
        )
        
        self.message_history.append(message)
        
        # Format message for display
        formatted_message = self._format_message(message)
        
        if self.console_log:
            self.console_log.write(formatted_message)
    
    def add_user_message(self, content: str) -> None:
        """Add user message"""
        self.add_message(content, MessageType.USER)
    
    def add_ai_message(self, content: str, task_id: Optional[str] = None) -> None:
        """Add AI response message"""
        self.add_message(content, MessageType.AI, task_id=task_id)
    
    def add_system_message(self, content: str) -> None:
        """Add system message"""
        self.add_message(content, MessageType.SYSTEM)
    
    def add_error_message(self, content: str) -> None:
        """Add error message"""
        self.add_message(content, MessageType.ERROR)
    
    def add_success_message(self, content: str) -> None:
        """Add success message"""
        self.add_message(content, MessageType.SUCCESS)
    
    def add_warning_message(self, content: str) -> None:
        """Add warning message"""
        self.add_message(content, MessageType.WARNING)
    
    def add_debug_message(self, content: str) -> None:
        """Add debug message"""
        self.add_message(content, MessageType.DEBUG)
    
    def _format_message(self, message: ConsoleMessage) -> Text:
        """Format message for display"""
        timestamp = message.timestamp.strftime("%H:%M:%S")
        
        # Message type styling
        type_styles = {
            MessageType.USER: "bold cyan",
            MessageType.AI: "bold green",
            MessageType.SYSTEM: "bold blue",
            MessageType.ERROR: "bold red",
            MessageType.SUCCESS: "bold green",
            MessageType.WARNING: "bold yellow",
            MessageType.DEBUG: "dim white"
        }
        
        type_icons = {
            MessageType.USER: "👤",
            MessageType.AI: "🤖",
            MessageType.SYSTEM: "⚙️",
            MessageType.ERROR: "❌",
            MessageType.SUCCESS: "✅",
            MessageType.WARNING: "⚠️",
            MessageType.DEBUG: "🔍"
        }
        
        style = type_styles.get(message.message_type, "white")
        icon = type_icons.get(message.message_type, "❓")
        
        # Build formatted text
        text = Text()
        text.append(f"[{timestamp}] ", style="dim")
        text.append(f"{icon} {message.message_type.value.upper()}: ", style=style)
        text.append(message.content)
        
        # Add task ID if present
        if message.task_id:
            text.append(f" (Task: {message.task_id})", style="dim")
        
        return text
    
    def get_conversation_history(self) -> List[ConsoleMessage]:
        """Get conversation history"""
        return self.message_history.copy()
    
    def export_conversation(self) -> str:
        """Export conversation as text"""
        lines = []
        for msg in self.message_history:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{timestamp}] {msg.message_type.value.upper()}: {msg.content}")
        return "\n".join(lines)


class ShowCommandTemplatesMessage(Message):
    """Message to show command templates dialog"""
    pass