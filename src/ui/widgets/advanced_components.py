#!/usr/bin/env python3
"""
Advanced Terminal Widgets - Modern UI components with rich interactions
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

from textual import on, work, events
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container, Grid, ScrollableContainer
from textual.widgets import (
    Static, Label, Button, Input, Select, Checkbox, 
    ProgressBar, ListView, ListItem, Tree,
    RichLog, DataTable, TabbedContent, TabPane
)
from textual.message import Message
from textual.reactive import reactive
from textual.geometry import Size
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.tree import Tree as RichTree
from rich.align import Align
from rich.columns import Columns


class ComponentState(Enum):
    """Component states for styling"""
    DEFAULT = "default"
    HOVER = "hover"
    ACTIVE = "active"
    DISABLED = "disabled"
    LOADING = "loading"
    ERROR = "error"
    SUCCESS = "success"


class AnimationType(Enum):
    """Animation types"""
    FADE_IN = "fade_in"
    SLIDE_IN = "slide_in"
    BOUNCE = "bounce"
    PULSE = "pulse"
    SHAKE = "shake"


@dataclass
class ComponentTheme:
    """Component-specific theme settings"""
    primary_color: str = "#0ea5e9"
    secondary_color: str = "#8b5cf6"
    success_color: str = "#10b981"
    warning_color: str = "#f59e0b"
    error_color: str = "#ef4444"
    background_color: str = "#1e293b"
    surface_color: str = "#334155"
    text_color: str = "#f8fafc"
    border_color: str = "#475569"


class EnhancedButton(Static):
    """Enhanced button with hover effects, icons, and loading states"""
    
    DEFAULT_CSS = """
    EnhancedButton {
        width: auto;
        height: 3;
        padding: 0 2;
        margin: 0 1;
        border: round $border;
        background: $surface;
        color: $text;
        text-align: center;
        content-align: center middle;
    }
    
    EnhancedButton:hover {
        background: $primary-light;
        border: round $primary;
    }
    
    EnhancedButton:focus {
        border: thick $primary;
    }
    
    EnhancedButton.primary {
        background: $primary;
        color: white;
    }
    
    EnhancedButton.success {
        background: $success;
        color: white;
    }
    
    EnhancedButton.warning {
        background: $warning;
        color: white;
    }
    
    EnhancedButton.error {
        background: $error;
        color: white;
    }
    
    EnhancedButton.disabled {
        background: $disabled;
        color: $text-muted;
        opacity: 0.6;
    }
    
    EnhancedButton.loading {
        opacity: 0.8;
    }
    """
    
    state: reactive[ComponentState] = reactive(ComponentState.DEFAULT)
    
    def __init__(
        self,
        label: str,
        *,
        icon: Optional[str] = None,
        variant: str = "default",
        size: str = "medium",
        loading: bool = False,
        disabled: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label = label
        self.icon = icon
        self.variant = variant
        self.size = size
        self.loading = loading
        self.disabled = disabled
        self._click_handlers: List[Callable] = []
        
        # Apply variant class
        if variant != "default":
            self.add_class(variant)
        if disabled:
            self.add_class("disabled")
        if loading:
            self.add_class("loading")
    
    def render(self) -> Text:
        """Render button content"""
        content = Text()
        
        # Add loading spinner
        if self.loading:
            content.append("🔄 ", style="dim")
        # Add icon
        elif self.icon:
            content.append(f"{self.icon} ")
        
        # Add label
        content.append(self.label)
        
        return content
    
    def on_click(self, event: events.Click) -> None:
        """Handle click events"""
        if not self.disabled and not self.loading:
            self.post_message(ButtonPressed(self))
            
            # Call registered handlers
            for handler in self._click_handlers:
                handler()
    
    def on_enter(self, event: events.Enter) -> None:
        """Handle mouse enter"""
        if not self.disabled:
            self.state = ComponentState.HOVER
    
    def on_leave(self, event: events.Leave) -> None:
        """Handle mouse leave"""
        if not self.disabled:
            self.state = ComponentState.DEFAULT
    
    def on_key(self, event: events.Key) -> None:
        """Handle keyboard events"""
        if event.key == "enter" and self.has_focus:
            if not self.disabled and not self.loading:
                self.on_click(events.Click())
    
    def set_loading(self, loading: bool) -> None:
        """Set loading state"""
        self.loading = loading
        if loading:
            self.add_class("loading")
        else:
            self.remove_class("loading")
        self.refresh()
    
    def set_disabled(self, disabled: bool) -> None:
        """Set disabled state"""
        self.disabled = disabled
        if disabled:
            self.add_class("disabled")
            self.state = ComponentState.DISABLED
        else:
            self.remove_class("disabled")
            self.state = ComponentState.DEFAULT
    
    def add_click_handler(self, handler: Callable) -> None:
        """Add click handler"""
        self._click_handlers.append(handler)


class StatusCard(Container):
    """Status card with icon, title, value, and trend"""
    
    DEFAULT_CSS = """
    StatusCard {
        width: 100%;
        height: 8;
        border: round $border;
        background: $surface;
        padding: 1;
        margin: 1;
    }
    
    StatusCard:hover {
        border: round $primary;
        background: $surface-light;
    }
    
    .status-icon {
        width: 3;
        text-align: center;
        text-style: bold;
    }
    
    .status-content {
        width: 1fr;
        padding: 0 1;
    }
    
    .status-title {
        color: $text-muted;
        height: 1;
    }
    
    .status-value {
        color: $text;
        text-style: bold;
        height: 2;
    }
    
    .status-trend {
        color: $success;
        height: 1;
    }
    
    .status-trend-down {
        color: $error;
    }
    """
    
    def __init__(
        self,
        title: str,
        value: str,
        *,
        icon: str = "📊",
        trend: Optional[str] = None,
        trend_direction: str = "up",
        color: str = "primary",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.value = value
        self.icon = icon
        self.trend = trend
        self.trend_direction = trend_direction
        self.color = color
    
    def compose(self) -> ComposeResult:
        """Compose status card"""
        with Horizontal():
            # Icon
            yield Static(self.icon, classes="status-icon")
            
            # Content
            with Vertical(classes="status-content"):
                yield Static(self.title, classes="status-title")
                yield Static(self.value, classes="status-value")
                
                if self.trend:
                    trend_class = "status-trend"
                    if self.trend_direction == "down":
                        trend_class += " status-trend-down"
                        trend_icon = "📉"
                    else:
                        trend_icon = "📈"
                    
                    yield Static(f"{trend_icon} {self.trend}", classes=trend_class)
    
    def update_value(self, value: str, trend: Optional[str] = None) -> None:
        """Update card value and trend"""
        self.value = value
        if trend is not None:
            self.trend = trend
        self.refresh()


class InteractiveChart(Static):
    """Interactive ASCII chart widget"""
    
    DEFAULT_CSS = """
    InteractiveChart {
        width: 100%;
        height: 15;
        border: round $border;
        background: $surface;
        padding: 1;
    }
    """
    
    def __init__(
        self,
        title: str,
        *,
        chart_type: str = "line",
        data_points: List[Tuple[str, float]] = None,
        max_points: int = 50,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.chart_type = chart_type
        self.data_points = data_points or []
        self.max_points = max_points
        self._hover_index: Optional[int] = None
    
    def render(self) -> Panel:
        """Render interactive chart"""
        if not self.data_points:
            return Panel("No data available", title=self.title)
        
        chart_content = []
        
        if self.chart_type == "line":
            chart_content = self._render_line_chart()
        elif self.chart_type == "bar":
            chart_content = self._render_bar_chart()
        elif self.chart_type == "sparkline":
            chart_content = self._render_sparkline()
        
        content = "\n".join(chart_content)
        
        # Add hover information
        if self._hover_index is not None and self._hover_index < len(self.data_points):
            label, value = self.data_points[self._hover_index]
            content += f"\n\nHover: {label} = {value}"
        
        return Panel(content, title=f"📈 {self.title}")
    
    def _render_line_chart(self) -> List[str]:
        """Render line chart"""
        if len(self.data_points) < 2:
            return ["Insufficient data for line chart"]
        
        values = [point[1] for point in self.data_points]
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            return [f"Constant value: {min_val}"]
        
        # Normalize values to chart height
        chart_height = 10
        normalized = [
            int((val - min_val) / (max_val - min_val) * chart_height)
            for val in values
        ]
        
        # Create chart lines
        lines = []
        for row in range(chart_height, -1, -1):
            line = ""
            for i, val in enumerate(normalized):
                if val >= row:
                    # Check if this point is being hovered
                    if i == self._hover_index:
                        line += "●"
                    else:
                        line += "█"
                else:
                    line += " "
            lines.append(line)
        
        # Add axis
        lines.append("─" * len(normalized))
        lines.append(f"Min: {min_val:.1f} │ Max: {max_val:.1f} │ Current: {values[-1]:.1f}")
        
        return lines
    
    def _render_bar_chart(self) -> List[str]:
        """Render bar chart"""
        if not self.data_points:
            return ["No data"]
        
        values = [point[1] for point in self.data_points]
        labels = [point[0] for point in self.data_points]
        max_val = max(values) if values else 1
        
        lines = []
        bar_width = 20  # Character width for bars
        
        for i, (label, value) in enumerate(self.data_points[-10:]):  # Show last 10
            # Calculate bar length
            bar_length = int((value / max_val) * bar_width) if max_val > 0 else 0
            bar = "█" * bar_length + "░" * (bar_width - bar_length)
            
            # Highlight if hovered
            if i == self._hover_index:
                lines.append(f"► {label[:8]:8} [{bar}] {value:.1f}")
            else:
                lines.append(f"  {label[:8]:8} [{bar}] {value:.1f}")
        
        return lines
    
    def _render_sparkline(self) -> List[str]:
        """Render compact sparkline"""
        if not self.data_points:
            return ["No data"]
        
        values = [point[1] for point in self.data_points]
        
        # Use sparkline characters
        sparkline_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            sparkline = sparkline_chars[0] * len(values)
        else:
            sparkline = ""
            for value in values:
                normalized = (value - min_val) / (max_val - min_val)
                char_index = min(int(normalized * (len(sparkline_chars) - 1)), len(sparkline_chars) - 1)
                sparkline += sparkline_chars[char_index]
        
        return [
            sparkline,
            f"Range: {min_val:.1f} - {max_val:.1f}",
            f"Latest: {values[-1]:.1f} ({len(values)} points)"
        ]
    
    def add_data_point(self, label: str, value: float) -> None:
        """Add new data point"""
        self.data_points.append((label, value))
        
        # Keep within max points limit
        if len(self.data_points) > self.max_points:
            self.data_points = self.data_points[-self.max_points:]
        
        self.refresh()
    
    def clear_data(self) -> None:
        """Clear all data points"""
        self.data_points.clear()
        self.refresh()


class SmartDataTable(Container):
    """Enhanced data table with sorting, filtering, and pagination"""
    
    DEFAULT_CSS = """
    SmartDataTable {
        width: 100%;
        height: 100%;
    }
    
    .table-header {
        height: 3;
        background: $surface-variant;
        padding: 0 1;
    }
    
    .table-controls {
        height: 3;
        background: $surface;
        padding: 0 1;
    }
    
    .table-data {
        height: 1fr;
    }
    
    .table-footer {
        height: 3;
        background: $surface;
        padding: 0 1;
    }
    """
    
    def __init__(
        self,
        columns: List[str],
        *,
        sortable: bool = True,
        filterable: bool = True,
        paginated: bool = True,
        page_size: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.sortable = sortable
        self.filterable = filterable
        self.paginated = paginated
        self.page_size = page_size
        
        self.data: List[List[Any]] = []
        self.filtered_data: List[List[Any]] = []
        self.sort_column: Optional[int] = None
        self.sort_reverse: bool = False
        self.current_page: int = 0
        self.filter_text: str = ""
        
        self.data_table: Optional[DataTable] = None
        self.filter_input: Optional[Input] = None
        self.page_label: Optional[Label] = None
    
    def compose(self) -> ComposeResult:
        """Compose smart data table"""
        # Header
        yield Label(f"📊 Data Table ({len(self.columns)} columns)", classes="table-header")
        
        # Controls
        if self.filterable or self.sortable:
            with Horizontal(classes="table-controls"):
                if self.filterable:
                    self.filter_input = Input(placeholder="Filter data...", id="table-filter")
                    yield self.filter_input
                
                if self.sortable:
                    yield Button("🔄 Sort", id="sort-menu")
                    yield Button("📋 Export", id="export-data")
        
        # Data table
        self.data_table = DataTable()
        self.data_table.add_columns(*self.columns)
        yield self.data_table
        
        # Footer with pagination
        if self.paginated:
            with Horizontal(classes="table-footer"):
                yield Button("⬅️ Prev", id="prev-page")
                self.page_label = Label("Page 1 of 1")
                yield self.page_label
                yield Button("➡️ Next", id="next-page")
    
    def set_data(self, data: List[List[Any]]) -> None:
        """Set table data"""
        self.data = data
        self.filtered_data = data.copy()
        self._update_display()
    
    def add_row(self, row: List[Any]) -> None:
        """Add single row"""
        self.data.append(row)
        if self._matches_filter(row):
            self.filtered_data.append(row)
        self._update_display()
    
    def _matches_filter(self, row: List[Any]) -> bool:
        """Check if row matches current filter"""
        if not self.filter_text:
            return True
        
        search_text = self.filter_text.lower()
        return any(search_text in str(cell).lower() for cell in row)
    
    def _apply_filter(self) -> None:
        """Apply current filter to data"""
        if not self.filter_text:
            self.filtered_data = self.data.copy()
        else:
            self.filtered_data = [row for row in self.data if self._matches_filter(row)]
        
        self.current_page = 0
        self._update_display()
    
    def _apply_sort(self) -> None:
        """Apply current sort to filtered data"""
        if self.sort_column is not None:
            self.filtered_data.sort(
                key=lambda row: row[self.sort_column] if self.sort_column < len(row) else "",
                reverse=self.sort_reverse
            )
        self._update_display()
    
    def _update_display(self) -> None:
        """Update table display"""
        if not self.data_table:
            return
        
        # Clear existing rows
        self.data_table.clear()
        
        # Calculate pagination
        if self.paginated:
            start_idx = self.current_page * self.page_size
            end_idx = start_idx + self.page_size
            display_data = self.filtered_data[start_idx:end_idx]
            
            # Update page label
            total_pages = max(1, (len(self.filtered_data) + self.page_size - 1) // self.page_size)
            if self.page_label:
                self.page_label.update(f"Page {self.current_page + 1} of {total_pages}")
        else:
            display_data = self.filtered_data
        
        # Add rows to table
        for row in display_data:
            self.data_table.add_row(*[str(cell) for cell in row])
    
    @on(Input.Changed, "#table-filter")
    def filter_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes"""
        self.filter_text = event.value
        self._apply_filter()
    
    @on(Button.Pressed, "#prev-page")
    def prev_page(self) -> None:
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_display()
    
    @on(Button.Pressed, "#next-page")
    def next_page(self) -> None:
        """Go to next page"""
        total_pages = max(1, (len(self.filtered_data) + self.page_size - 1) // self.page_size)
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._update_display()


class CommandPalette(Container):
    """VS Code-style command palette"""
    
    DEFAULT_CSS = """
    CommandPalette {
        dock: top;
        width: 80%;
        height: 20;
        margin: 2 10%;
        border: thick $primary;
        background: $surface;
        display: none;
    }
    
    CommandPalette.visible {
        display: block;
    }
    
    .palette-input {
        height: 3;
        background: $background;
        border-bottom: wide $border;
        padding: 0 1;
    }
    
    .palette-results {
        height: 1fr;
    }
    """
    
    visible: reactive[bool] = reactive(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands: Dict[str, Callable] = {}
        self.filtered_commands: List[Tuple[str, str]] = []
        self.selected_index: int = 0
        
        self.search_input: Optional[Input] = None
        self.results_list: Optional[ListView] = None
    
    def compose(self) -> ComposeResult:
        """Compose command palette"""
        # Search input
        with Horizontal(classes="palette-input"):
            yield Label("🔍")
            self.search_input = Input(placeholder="Type command name...")
            yield self.search_input
        
        # Results list
        self.results_list = ListView(classes="palette-results")
        yield self.results_list
    
    def watch_visible(self, visible: bool) -> None:
        """React to visibility changes"""
        if visible:
            self.add_class("visible")
            if self.search_input:
                self.search_input.focus()
        else:
            self.remove_class("visible")
    
    def toggle(self) -> None:
        """Toggle palette visibility"""
        self.visible = not self.visible
    
    def register_command(self, name: str, description: str, handler: Callable) -> None:
        """Register a new command"""
        self.commands[name] = handler
        self._update_results()
    
    def unregister_command(self, name: str) -> None:
        """Unregister a command"""
        if name in self.commands:
            del self.commands[name]
            self._update_results()
    
    def _update_results(self, search_term: str = "") -> None:
        """Update command results based on search"""
        if not self.results_list:
            return
        
        # Filter commands
        if search_term:
            self.filtered_commands = [
                (name, f"Execute: {name}")
                for name in self.commands.keys()
                if search_term.lower() in name.lower()
            ]
        else:
            self.filtered_commands = [
                (name, f"Execute: {name}")
                for name in self.commands.keys()
            ]
        
        # Update list
        self.results_list.clear()
        for i, (name, description) in enumerate(self.filtered_commands):
            item_text = f"{'►' if i == self.selected_index else ' '} {name}"
            if description:
                item_text += f"\n  {description}"
            
            list_item = ListItem(Static(item_text))
            if i == self.selected_index:
                list_item.add_class("selected")
            
            self.results_list.append(list_item)
    
    def on_key(self, event: events.Key) -> None:
        """Handle keyboard navigation"""
        if not self.visible:
            return
        
        if event.key == "escape":
            self.visible = False
        elif event.key == "up":
            self.selected_index = max(0, self.selected_index - 1)
            self._update_results(self.search_input.value if self.search_input else "")
        elif event.key == "down":
            self.selected_index = min(len(self.filtered_commands) - 1, self.selected_index + 1)
            self._update_results(self.search_input.value if self.search_input else "")
        elif event.key == "enter":
            if self.filtered_commands and self.selected_index < len(self.filtered_commands):
                command_name = self.filtered_commands[self.selected_index][0]
                if command_name in self.commands:
                    self.commands[command_name]()
                    self.visible = False


class NotificationToast(Container):
    """Modern notification toast"""
    
    DEFAULT_CSS = """
    NotificationToast {
        dock: top;
        width: 50;
        height: auto;
        margin: 1 5;
        border: round $border;
        background: $surface;
        display: none;
        offset-x: 100;
    }
    
    NotificationToast.visible {
        display: block;
        animate: offset-x 300ms ease-out;
        offset-x: 0;
    }
    
    NotificationToast.success {
        border: round $success;
        background: $success-alpha-20;
    }
    
    NotificationToast.error {
        border: round $error;
        background: $error-alpha-20;
    }
    
    NotificationToast.warning {
        border: round $warning;
        background: $warning-alpha-20;
    }
    
    NotificationToast.info {
        border: round $info;
        background: $info-alpha-20;
    }
    """
    
    def __init__(
        self,
        message: str,
        *,
        notification_type: str = "info",
        duration: int = 5,
        actions: Optional[List[Tuple[str, Callable]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.message = message
        self.notification_type = notification_type
        self.duration = duration
        self.actions = actions or []
        
        self.add_class(notification_type)
        self._auto_dismiss_task: Optional[asyncio.Task] = None
    
    def compose(self) -> ComposeResult:
        """Compose notification toast"""
        with Horizontal():
            # Icon
            icons = {
                "success": "✅",
                "error": "❌", 
                "warning": "⚠️",
                "info": "ℹ️"
            }
            yield Static(icons.get(self.notification_type, "ℹ️"), classes="toast-icon")
            
            # Content
            with Vertical(classes="toast-content"):
                yield Static(self.message)
                
                # Actions
                if self.actions:
                    with Horizontal(classes="toast-actions"):
                        for action_text, action_handler in self.actions:
                            button = EnhancedButton(action_text, size="small")
                            button.add_click_handler(action_handler)
                            yield button
            
            # Dismiss button
            dismiss_btn = EnhancedButton("×", size="small")
            dismiss_btn.add_click_handler(self.dismiss)
            yield dismiss_btn
    
    def show(self) -> None:
        """Show the notification"""
        self.add_class("visible")
        
        # Auto-dismiss after duration
        if self.duration > 0:
            self._auto_dismiss_task = asyncio.create_task(self._auto_dismiss())
    
    async def _auto_dismiss(self) -> None:
        """Auto-dismiss after duration"""
        await asyncio.sleep(self.duration)
        self.dismiss()
    
    def dismiss(self) -> None:
        """Dismiss the notification"""
        if self._auto_dismiss_task:
            self._auto_dismiss_task.cancel()
        
        self.remove_class("visible")
        # Remove from DOM after animation
        asyncio.create_task(self._remove_after_delay())
    
    async def _remove_after_delay(self) -> None:
        """Remove from DOM after animation"""
        await asyncio.sleep(0.3)  # Animation duration
        self.remove()


# Message classes
class ButtonPressed(Message):
    """Enhanced button pressed message"""
    
    def __init__(self, button: EnhancedButton) -> None:
        super().__init__()
        self.button = button