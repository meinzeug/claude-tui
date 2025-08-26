#!/usr/bin/env python3
"""
Enhanced Terminal Components - Advanced TUI components with modern interactions
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math

from textual import on, work, events
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container, Grid, ScrollableContainer
from textual.widgets import Static, Label, Button, Input, ProgressBar, ListView, ListItem
from textual.message import Message
from textual.reactive import reactive
from textual.geometry import Size
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.bar import Bar
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.align import Align
from rich.columns import Columns


class TreeView(Container):
    """Enhanced tree view with lazy loading and custom icons"""
    
    DEFAULT_CSS = """
    TreeView {
        background: $surface;
        border: round $border;
        padding: 1;
        scrollbar-background: $surface-variant;
        scrollbar-color: $primary;
    }
    
    .tree-node {
        padding: 0 1;
        height: 1;
    }
    
    .tree-node:hover {
        background: $surface-light;
    }
    
    .tree-node-selected {
        background: $primary;
        color: $on-primary;
    }
    
    .tree-node-expanded {
        color: $primary;
    }
    
    .tree-node-collapsed {
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        *,
        lazy_loading: bool = True,
        show_icons: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data = data or {}
        self.lazy_loading = lazy_loading
        self.show_icons = show_icons
        self.expanded_nodes: set = set()
        self.selected_node: Optional[str] = None
        self.node_widgets: Dict[str, Static] = {}
        
    def compose(self) -> ComposeResult:
        """Compose tree view"""
        with ScrollableContainer():
            yield from self._render_nodes(self.data, level=0)
    
    def _render_nodes(self, nodes: Dict[str, Any], level: int = 0, parent_path: str = "") -> List[Static]:
        """Render tree nodes recursively"""
        widgets = []
        
        for key, value in nodes.items():
            node_path = f"{parent_path}/{key}" if parent_path else key
            indent = "  " * level
            
            # Determine if this is a branch or leaf
            is_branch = isinstance(value, dict) and value
            is_expanded = node_path in self.expanded_nodes
            
            # Choose icon
            if self.show_icons:
                if is_branch:
                    icon = "📂" if is_expanded else "📁"
                else:
                    icon = self._get_file_icon(key)
                prefix = f"{indent}{icon} {key}"
            else:
                if is_branch:
                    symbol = "▼" if is_expanded else "▶"
                    prefix = f"{indent}{symbol} {key}"
                else:
                    prefix = f"{indent}  {key}"
            
            # Create node widget
            node_widget = Static(prefix, classes="tree-node")
            if node_path == self.selected_node:
                node_widget.add_class("tree-node-selected")
            
            node_widget.node_path = node_path
            node_widget.is_branch = is_branch
            self.node_widgets[node_path] = node_widget
            widgets.append(node_widget)
            
            # Render children if expanded
            if is_branch and is_expanded:
                child_widgets = self._render_nodes(value, level + 1, node_path)
                widgets.extend(child_widgets)
        
        return widgets
    
    def _get_file_icon(self, filename: str) -> str:
        """Get icon based on file extension"""
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        icon_map = {
            'py': '🐍',
            'js': '📜',
            'ts': '📘',
            'html': '🌐',
            'css': '🎨',
            'json': '📋',
            'md': '📖',
            'txt': '📄',
            'pdf': '📕',
            'img': '🖼️',
            'zip': '📦',
            'exe': '⚡',
        }
        
        return icon_map.get(ext, '📄')
    
    def on_click(self, event: events.Click) -> None:
        """Handle node clicks"""
        # Find clicked node
        widget = event.widget
        if hasattr(widget, 'node_path'):
            node_path = widget.node_path
            
            # Update selection
            if self.selected_node in self.node_widgets:
                self.node_widgets[self.selected_node].remove_class("tree-node-selected")
            
            self.selected_node = node_path
            widget.add_class("tree-node-selected")
            
            # Toggle expansion for branches
            if hasattr(widget, 'is_branch') and widget.is_branch:
                if node_path in self.expanded_nodes:
                    self.expanded_nodes.remove(node_path)
                else:
                    self.expanded_nodes.add(node_path)
                
                # Refresh tree
                self.refresh()
                self.post_message(TreeNodeToggled(node_path, node_path in self.expanded_nodes))
            
            # Emit selection event
            self.post_message(TreeNodeSelected(node_path))
    
    def expand_node(self, node_path: str) -> None:
        """Programmatically expand a node"""
        if node_path not in self.expanded_nodes:
            self.expanded_nodes.add(node_path)
            self.refresh()
    
    def collapse_node(self, node_path: str) -> None:
        """Programmatically collapse a node"""
        if node_path in self.expanded_nodes:
            self.expanded_nodes.remove(node_path)
            self.refresh()
    
    def select_node(self, node_path: str) -> None:
        """Programmatically select a node"""
        if self.selected_node in self.node_widgets:
            self.node_widgets[self.selected_node].remove_class("tree-node-selected")
        
        self.selected_node = node_path
        if node_path in self.node_widgets:
            self.node_widgets[node_path].add_class("tree-node-selected")


class SplitPane(Container):
    """Resizable split pane container"""
    
    DEFAULT_CSS = """
    SplitPane {
        layout: horizontal;
    }
    
    .split-pane-left {
        background: $surface;
        border-right: wide $border;
    }
    
    .split-pane-right {
        background: $background;
    }
    
    .split-pane-divider {
        width: 1;
        background: $border;
        color: $text-muted;
        text-align: center;
    }
    
    .split-pane-divider:hover {
        background: $primary;
        color: $on-primary;
    }
    """
    
    split_ratio: reactive[float] = reactive(0.5)
    
    def __init__(
        self,
        left_widget,
        right_widget,
        *,
        orientation: str = "horizontal",
        initial_ratio: float = 0.5,
        min_ratio: float = 0.1,
        max_ratio: float = 0.9,
        resizable: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.left_widget = left_widget
        self.right_widget = right_widget
        self.orientation = orientation
        self.split_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.resizable = resizable
        self.dragging = False
        
    def compose(self) -> ComposeResult:
        """Compose split pane"""
        if self.orientation == "horizontal":
            with Horizontal():
                # Left pane
                left_container = Container(classes="split-pane-left")
                left_container.styles.width = f"{self.split_ratio * 100}%"
                yield left_container
                
                # Divider
                if self.resizable:
                    divider = Static("┃", classes="split-pane-divider")
                    yield divider
                
                # Right pane  
                right_container = Container(classes="split-pane-right")
                right_container.styles.width = f"{(1 - self.split_ratio) * 100}%"
                yield right_container
        else:  # vertical
            with Vertical():
                # Top pane
                top_container = Container(classes="split-pane-left")
                top_container.styles.height = f"{self.split_ratio * 100}%"
                yield top_container
                
                # Divider
                if self.resizable:
                    divider = Static("━", classes="split-pane-divider")
                    yield divider
                
                # Bottom pane
                bottom_container = Container(classes="split-pane-right")
                bottom_container.styles.height = f"{(1 - self.split_ratio) * 100}%"
                yield bottom_container
    
    def watch_split_ratio(self, ratio: float) -> None:
        """Update pane sizes when ratio changes"""
        ratio = max(self.min_ratio, min(self.max_ratio, ratio))
        self.split_ratio = ratio
        
        # Update widget sizes
        left_pane = self.query_one(".split-pane-left")
        right_pane = self.query_one(".split-pane-right")
        
        if self.orientation == "horizontal":
            left_pane.styles.width = f"{ratio * 100}%"
            right_pane.styles.width = f"{(1 - ratio) * 100}%"
        else:
            left_pane.styles.height = f"{ratio * 100}%"
            right_pane.styles.height = f"{(1 - ratio) * 100}%"
    
    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Start dragging divider"""
        if self.resizable:
            # Check if click is on divider
            divider = self.query_one(".split-pane-divider", Static)
            if event.widget == divider:
                self.dragging = True
                event.capture_mouse()
    
    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Handle divider dragging"""
        if self.dragging:
            if self.orientation == "horizontal":
                new_ratio = event.x / self.size.width
            else:
                new_ratio = event.y / self.size.height
            
            self.split_ratio = max(self.min_ratio, min(self.max_ratio, new_ratio))
    
    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Stop dragging"""
        if self.dragging:
            self.dragging = False
            event.release_mouse()


class TabContainer(Container):
    """Enhanced tab container with closeable tabs"""
    
    DEFAULT_CSS = """
    TabContainer {
        background: $background;
    }
    
    .tab-bar {
        height: 3;
        background: $surface;
        border-bottom: wide $border;
        padding: 0;
    }
    
    .tab {
        background: $surface-variant;
        color: $text-muted;
        padding: 0 2;
        margin: 0 1;
        border-top: round $border;
        height: 3;
        min-width: 10;
        text-align: center;
    }
    
    .tab:hover {
        background: $surface-light;
        color: $text;
    }
    
    .tab-active {
        background: $background;
        color: $primary;
        border-top: round $primary;
        text-style: bold;
    }
    
    .tab-close {
        color: $text-muted;
        text-style: bold;
        width: 2;
    }
    
    .tab-close:hover {
        color: $error;
        background: $error-alpha-20;
    }
    
    .tab-content {
        height: 1fr;
        background: $background;
        padding: 1;
    }
    """
    
    active_tab: reactive[int] = reactive(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tabs: List[Dict[str, Any]] = []
        self.tab_widgets: List[Static] = []
        self.content_area: Optional[Container] = None
        
    def compose(self) -> ComposeResult:
        """Compose tab container"""
        # Tab bar
        with Horizontal(classes="tab-bar"):
            yield from self._render_tab_buttons()
        
        # Content area
        self.content_area = Container(classes="tab-content")
        yield self.content_area
    
    def _render_tab_buttons(self) -> List[Static]:
        """Render tab buttons"""
        widgets = []
        self.tab_widgets.clear()
        
        for i, tab in enumerate(self.tabs):
            with Horizontal(classes="tab" + (" tab-active" if i == self.active_tab else "")):
                # Tab title
                title_widget = Static(tab['title'])
                title_widget.tab_index = i
                widgets.append(title_widget)
                self.tab_widgets.append(title_widget)
                
                # Close button (if closeable)
                if tab.get('closeable', True):
                    close_widget = Static("×", classes="tab-close")
                    close_widget.tab_index = i
                    close_widget.is_close_button = True
                    widgets.append(close_widget)
        
        return widgets
    
    def add_tab(self, title: str, content, *, closeable: bool = True) -> int:
        """Add new tab"""
        tab_data = {
            'title': title,
            'content': content,
            'closeable': closeable
        }
        
        self.tabs.append(tab_data)
        tab_index = len(self.tabs) - 1
        
        # Refresh tab bar
        self.refresh()
        
        # Switch to new tab
        self.active_tab = tab_index
        
        return tab_index
    
    def close_tab(self, index: int) -> bool:
        """Close tab by index"""
        if 0 <= index < len(self.tabs) and self.tabs[index].get('closeable', True):
            # Remove tab
            del self.tabs[index]
            
            # Adjust active tab
            if index <= self.active_tab and self.active_tab > 0:
                self.active_tab -= 1
            elif len(self.tabs) == 0:
                self.active_tab = -1
            elif self.active_tab >= len(self.tabs):
                self.active_tab = len(self.tabs) - 1
            
            # Refresh
            self.refresh()
            self.post_message(TabClosed(index))
            return True
        
        return False
    
    def switch_to_tab(self, index: int) -> bool:
        """Switch to tab by index"""
        if 0 <= index < len(self.tabs):
            self.active_tab = index
            return True
        return False
    
    def watch_active_tab(self, index: int) -> None:
        """Update content when active tab changes"""
        if 0 <= index < len(self.tabs) and self.content_area:
            # Clear current content
            self.content_area.remove_children()
            
            # Add new content
            content = self.tabs[index]['content']
            self.content_area.mount(content)
            
            # Emit event
            self.post_message(TabChanged(index))
    
    def on_click(self, event: events.Click) -> None:
        """Handle tab clicks"""
        widget = event.widget
        
        if hasattr(widget, 'tab_index'):
            if hasattr(widget, 'is_close_button'):
                # Close button clicked
                self.close_tab(widget.tab_index)
            else:
                # Tab title clicked
                self.switch_to_tab(widget.tab_index)


class GraphWidget(Static):
    """Real-time graph widget with multiple data series"""
    
    DEFAULT_CSS = """
    GraphWidget {
        background: $surface;
        border: round $border;
        padding: 1;
        height: 15;
    }
    """
    
    def __init__(
        self,
        title: str = "",
        *,
        max_points: int = 100,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        show_legend: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.max_points = max_points
        self.y_min = y_min
        self.y_max = y_max
        self.show_legend = show_legend
        
        self.data_series: Dict[str, List[Tuple[float, float]]] = {}
        self.series_colors: Dict[str, str] = {}
        self.series_styles: Dict[str, str] = {}
        
    def add_series(self, name: str, color: str = "blue", style: str = "line") -> None:
        """Add a new data series"""
        self.data_series[name] = []
        self.series_colors[name] = color
        self.series_styles[name] = style
    
    def add_data_point(self, series_name: str, x: float, y: float) -> None:
        """Add data point to series"""
        if series_name not in self.data_series:
            self.add_series(series_name)
        
        self.data_series[series_name].append((x, y))
        
        # Limit points
        if len(self.data_series[series_name]) > self.max_points:
            self.data_series[series_name] = self.data_series[series_name][-self.max_points:]
        
        self.refresh()
    
    def clear_series(self, series_name: Optional[str] = None) -> None:
        """Clear data from series or all series"""
        if series_name:
            if series_name in self.data_series:
                self.data_series[series_name].clear()
        else:
            for series in self.data_series.values():
                series.clear()
        self.refresh()
    
    def render(self) -> Panel:
        """Render graph as ASCII art"""
        if not self.data_series or not any(self.data_series.values()):
            return Panel("No data available", title=self.title)
        
        # Collect all data points
        all_points = []
        for series_data in self.data_series.values():
            all_points.extend(series_data)
        
        if not all_points:
            return Panel("No data points", title=self.title)
        
        # Determine bounds
        x_values = [p[0] for p in all_points]
        y_values = [p[1] for p in all_points]
        
        x_min, x_max = min(x_values), max(x_values)
        y_min = self.y_min if self.y_min is not None else min(y_values)
        y_max = self.y_max if self.y_max is not None else max(y_values)
        
        if y_max == y_min:
            y_max = y_min + 1
        
        # Graph dimensions
        graph_width = 60
        graph_height = 10
        
        # Create graph grid
        graph_lines = []
        
        # Render each series
        for series_name, points in self.data_series.items():
            if not points:
                continue
            
            line = [' '] * graph_width
            
            for x, y in points:
                # Map to graph coordinates
                if x_max > x_min:
                    graph_x = int((x - x_min) / (x_max - x_min) * (graph_width - 1))
                else:
                    graph_x = 0
                
                graph_y = int((y - y_min) / (y_max - y_min) * (graph_height - 1))
                
                # Draw point
                if 0 <= graph_x < graph_width:
                    char = self._get_series_char(series_name)
                    line[graph_x] = char
            
            # Add series line to graph
            if not graph_lines:
                graph_lines = [list(line) for _ in range(graph_height)]
            else:
                for i, char in enumerate(line):
                    if char != ' ':
                        graph_lines[graph_height - 1][i] = char
        
        # Convert to string lines
        if not graph_lines:
            return Panel("No renderable data", title=self.title)
        
        content_lines = []
        for row in reversed(graph_lines):
            content_lines.append(''.join(row))
        
        # Add axes
        content_lines.append('─' * graph_width)
        content_lines.append(f"{y_min:.1f} │ {y_max:.1f}")
        
        # Add legend
        if self.show_legend and len(self.data_series) > 1:
            legend_items = []
            for name in self.data_series.keys():
                char = self._get_series_char(name)
                legend_items.append(f"{char} {name}")
            content_lines.append(" | ".join(legend_items))
        
        content = "\n".join(content_lines)
        return Panel(content, title=f"📊 {self.title}")
    
    def _get_series_char(self, series_name: str) -> str:
        """Get character representation for series"""
        chars = ['█', '▓', '▒', '░', '●', '◐', '○', '▲', '△', '▼', '▽', '◆', '◇']
        index = hash(series_name) % len(chars)
        return chars[index]


class Timeline(Container):
    """Timeline widget for showing chronological events"""
    
    DEFAULT_CSS = """
    Timeline {
        background: $surface;
        border: round $border;
        padding: 1;
        height: 20;
        scrollbar-background: $surface-variant;
        scrollbar-color: $primary;
    }
    
    .timeline-event {
        height: auto;
        margin: 1 0;
        padding: 1;
        background: $background;
        border: round $border;
    }
    
    .timeline-event:hover {
        background: $surface-light;
        border: round $primary;
    }
    
    .timeline-time {
        color: $text-muted;
        text-style: italic;
    }
    
    .timeline-title {
        color: $primary;
        text-style: bold;
        margin: 0 0 1 0;
    }
    
    .timeline-description {
        color: $text;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events: List[Dict[str, Any]] = []
        
    def compose(self) -> ComposeResult:
        """Compose timeline"""
        with ScrollableContainer():
            yield from self._render_events()
    
    def _render_events(self) -> List[Container]:
        """Render timeline events"""
        widgets = []
        
        # Sort events by timestamp
        sorted_events = sorted(self.events, key=lambda e: e.get('timestamp', datetime.min))
        
        for event in sorted_events:
            event_container = Container(classes="timeline-event")
            
            with event_container:
                # Time
                timestamp = event.get('timestamp', datetime.now())
                time_str = timestamp.strftime("%H:%M:%S")
                yield Static(time_str, classes="timeline-time")
                
                # Title
                title = event.get('title', 'Untitled Event')
                yield Static(title, classes="timeline-title")
                
                # Description
                description = event.get('description', '')
                if description:
                    yield Static(description, classes="timeline-description")
            
            widgets.append(event_container)
        
        return widgets
    
    def add_event(
        self,
        title: str,
        description: str = "",
        timestamp: Optional[datetime] = None,
        event_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add new event to timeline"""
        event = {
            'title': title,
            'description': description,
            'timestamp': timestamp or datetime.now(),
            'type': event_type,
            'metadata': metadata or {}
        }
        
        self.events.append(event)
        
        # Refresh timeline
        self.refresh()
        
        # Scroll to bottom to show latest event
        self.scroll_end()
    
    def clear_events(self) -> None:
        """Clear all events"""
        self.events.clear()
        self.refresh()
    
    def filter_events(self, event_type: Optional[str] = None, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Filter events by type and/or time"""
        filtered = self.events.copy()
        
        if event_type:
            filtered = [e for e in filtered if e.get('type') == event_type]
        
        if since:
            filtered = [e for e in filtered if e.get('timestamp', datetime.min) >= since]
        
        return filtered


# Message classes
class TreeNodeSelected(Message):
    """Tree node selection message"""
    
    def __init__(self, node_path: str) -> None:
        super().__init__()
        self.node_path = node_path


class TreeNodeToggled(Message):
    """Tree node toggle message"""
    
    def __init__(self, node_path: str, expanded: bool) -> None:
        super().__init__()
        self.node_path = node_path
        self.expanded = expanded


class TabChanged(Message):
    """Tab changed message"""
    
    def __init__(self, tab_index: int) -> None:
        super().__init__()
        self.tab_index = tab_index


class TabClosed(Message):
    """Tab closed message"""
    
    def __init__(self, tab_index: int) -> None:
        super().__init__()
        self.tab_index = tab_index