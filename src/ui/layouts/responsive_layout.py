#!/usr/bin/env python3
"""
Responsive Layout System - Adaptive layouts for different terminal sizes
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum

from textual import events
from textual.app import App
from textual.widget import Widget
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.geometry import Size
from textual.reactive import reactive
from textual.message import Message


class ScreenSize(Enum):
    """Screen size categories"""
    EXTRA_SMALL = "xs"  # < 40 cols
    SMALL = "sm"        # 40-79 cols
    MEDIUM = "md"       # 80-119 cols
    LARGE = "lg"        # 120-159 cols
    EXTRA_LARGE = "xl"  # >= 160 cols


class Breakpoint(Enum):
    """Layout breakpoints"""
    XS = 40
    SM = 80
    MD = 120
    LG = 160


@dataclass
class LayoutRule:
    """Layout rule for responsive behavior"""
    min_width: int
    max_width: Optional[int] = None
    widget_classes: List[str] = None
    hide_widgets: List[str] = None
    show_widgets: List[str] = None
    layout_type: str = "default"
    custom_properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.widget_classes is None:
            self.widget_classes = []
        if self.hide_widgets is None:
            self.hide_widgets = []
        if self.show_widgets is None:
            self.show_widgets = []
        if self.custom_properties is None:
            self.custom_properties = {}
    
    def applies_to_width(self, width: int) -> bool:
        """Check if this rule applies to given width"""
        return (width >= self.min_width and 
                (self.max_width is None or width < self.max_width))


class ResponsiveContainer(Container):
    """Container that adapts its layout based on screen size"""
    
    DEFAULT_CSS = """
    ResponsiveContainer {
        layout: vertical;
    }
    
    ResponsiveContainer.horizontal {
        layout: horizontal;
    }
    
    ResponsiveContainer.grid {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }
    
    ResponsiveContainer.stack {
        layout: vertical;
    }
    
    /* Size-specific classes */
    ResponsiveContainer.xs .hide-xs {
        display: none;
    }
    
    ResponsiveContainer.sm .hide-sm {
        display: none;
    }
    
    ResponsiveContainer.md .hide-md {
        display: none;
    }
    
    ResponsiveContainer.lg .hide-lg {
        display: none;
    }
    
    ResponsiveContainer.xl .hide-xl {
        display: none;
    }
    
    ResponsiveContainer.xs .show-xs {
        display: block;
    }
    
    ResponsiveContainer.sm .show-sm {
        display: block;
    }
    
    ResponsiveContainer.md .show-md {
        display: block;
    }
    
    ResponsiveContainer.lg .show-lg {
        display: block;
    }
    
    ResponsiveContainer.xl .show-xl {
        display: block;
    }
    """
    
    current_size: reactive[ScreenSize] = reactive(ScreenSize.MEDIUM)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout_rules: List[LayoutRule] = []
        self.responsive_widgets: Dict[str, Widget] = {}
        self.size_observers: List[Callable[[ScreenSize], None]] = []
        
        # Default responsive behavior
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default responsive layout rules"""
        self.layout_rules = [
            # Extra small screens - stack everything
            LayoutRule(
                min_width=0,
                max_width=Breakpoint.XS.value,
                widget_classes=["xs", "compact", "stack"],
                hide_widgets=["sidebar", "right-panel"],
                layout_type="vertical"
            ),
            
            # Small screens - limited sidebar
            LayoutRule(
                min_width=Breakpoint.XS.value,
                max_width=Breakpoint.SM.value,
                widget_classes=["sm", "compact"],
                hide_widgets=["right-panel"],
                show_widgets=["sidebar"],
                layout_type="horizontal"
            ),
            
            # Medium screens - standard layout
            LayoutRule(
                min_width=Breakpoint.SM.value,
                max_width=Breakpoint.MD.value,
                widget_classes=["md", "standard"],
                show_widgets=["sidebar", "main-content"],
                layout_type="horizontal"
            ),
            
            # Large screens - full layout
            LayoutRule(
                min_width=Breakpoint.MD.value,
                max_width=Breakpoint.LG.value,
                widget_classes=["lg", "expanded"],
                show_widgets=["sidebar", "main-content", "right-panel"],
                layout_type="horizontal"
            ),
            
            # Extra large screens - wide layout
            LayoutRule(
                min_width=Breakpoint.LG.value,
                widget_classes=["xl", "wide"],
                show_widgets=["sidebar", "main-content", "right-panel", "bottom-panel"],
                layout_type="grid"
            )
        ]
    
    def add_layout_rule(self, rule: LayoutRule) -> None:
        """Add custom layout rule"""
        self.layout_rules.append(rule)
        self.layout_rules.sort(key=lambda r: r.min_width)
    
    def register_responsive_widget(self, name: str, widget: Widget) -> None:
        """Register widget for responsive behavior"""
        self.responsive_widgets[name] = widget
    
    def add_size_observer(self, callback: Callable[[ScreenSize], None]) -> None:
        """Add observer for size changes"""
        self.size_observers.append(callback)
    
    def on_resize(self, event: events.Resize) -> None:
        """Handle container resize"""
        new_size = self._determine_screen_size(event.size.width)
        if new_size != self.current_size:
            self.current_size = new_size
            self._apply_responsive_layout()
    
    def watch_current_size(self, size: ScreenSize) -> None:
        """React to size changes"""
        # Update CSS classes
        self.remove_class("xs", "sm", "md", "lg", "xl")
        self.add_class(size.value)
        
        # Notify observers
        for observer in self.size_observers:
            observer(size)
        
        # Post message
        self.post_message(ResponsiveLayoutChanged(size))
    
    def _determine_screen_size(self, width: int) -> ScreenSize:
        """Determine screen size category from width"""
        if width < Breakpoint.XS.value:
            return ScreenSize.EXTRA_SMALL
        elif width < Breakpoint.SM.value:
            return ScreenSize.SMALL
        elif width < Breakpoint.MD.value:
            return ScreenSize.MEDIUM
        elif width < Breakpoint.LG.value:
            return ScreenSize.LARGE
        else:
            return ScreenSize.EXTRA_LARGE
    
    def _apply_responsive_layout(self) -> None:
        """Apply responsive layout based on current size"""
        width = self.size.width
        
        # Find applicable rule
        applicable_rule = None
        for rule in self.layout_rules:
            if rule.applies_to_width(width):
                applicable_rule = rule
        
        if not applicable_rule:
            return
        
        # Apply widget classes
        for class_name in applicable_rule.widget_classes:
            self.add_class(class_name)
        
        # Hide/show widgets
        for widget_name in applicable_rule.hide_widgets:
            widget = self.responsive_widgets.get(widget_name)
            if widget:
                widget.display = False
        
        for widget_name in applicable_rule.show_widgets:
            widget = self.responsive_widgets.get(widget_name)
            if widget:
                widget.display = True
        
        # Change layout if needed
        if applicable_rule.layout_type == "vertical":
            self.styles.layout = "vertical"
        elif applicable_rule.layout_type == "horizontal":
            self.styles.layout = "horizontal"
        elif applicable_rule.layout_type == "grid":
            self.styles.layout = "grid"
        
        # Apply custom properties
        for prop, value in applicable_rule.custom_properties.items():
            setattr(self.styles, prop, value)


class FlexContainer(Container):
    """Container with flexbox-like behavior for terminal UIs"""
    
    DEFAULT_CSS = """
    FlexContainer {
        layout: horizontal;
    }
    
    FlexContainer.column {
        layout: vertical;
    }
    
    .flex-item {
        width: auto;
        height: auto;
    }
    
    .flex-grow {
        width: 1fr;
    }
    
    .flex-shrink {
        width: auto;
        min-width: 0;
    }
    
    .flex-no-shrink {
        min-width: inherit;
    }
    """
    
    def __init__(
        self,
        direction: str = "row",
        justify_content: str = "start",
        align_items: str = "stretch",
        gap: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.direction = direction
        self.justify_content = justify_content
        self.align_items = align_items
        self.gap = gap
        
        # Set initial layout
        if direction == "column":
            self.add_class("column")
        
        # Apply gap if specified
        if gap > 0:
            self.styles.margin = gap
    
    def add_flex_item(
        self,
        widget: Widget,
        *,
        grow: int = 0,
        shrink: int = 1,
        basis: Optional[str] = None,
        align_self: Optional[str] = None
    ) -> None:
        """Add flex item with flex properties"""
        widget.add_class("flex-item")
        
        if grow > 0:
            widget.add_class("flex-grow")
            if self.direction == "row":
                widget.styles.width = f"{grow}fr"
            else:
                widget.styles.height = f"{grow}fr"
        
        if shrink == 0:
            widget.add_class("flex-no-shrink")
        else:
            widget.add_class("flex-shrink")
        
        if basis:
            if self.direction == "row":
                widget.styles.width = basis
            else:
                widget.styles.height = basis
        
        self.mount(widget)


class AdaptiveGrid(Grid):
    """Grid that adapts column count based on screen size"""
    
    DEFAULT_CSS = """
    AdaptiveGrid {
        layout: grid;
        grid-gutter: 1;
    }
    
    AdaptiveGrid.xs {
        grid-size: 1;
    }
    
    AdaptiveGrid.sm {
        grid-size: 2 1;
    }
    
    AdaptiveGrid.md {
        grid-size: 2 2;
    }
    
    AdaptiveGrid.lg {
        grid-size: 3 2;
    }
    
    AdaptiveGrid.xl {
        grid-size: 4 2;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_configs = {
            ScreenSize.EXTRA_SMALL: (1, 1),
            ScreenSize.SMALL: (2, 1),
            ScreenSize.MEDIUM: (2, 2),
            ScreenSize.LARGE: (3, 2),
            ScreenSize.EXTRA_LARGE: (4, 3),
        }
    
    def on_resize(self, event: events.Resize) -> None:
        """Handle resize events"""
        width = event.size.width
        screen_size = self._determine_screen_size(width)
        
        # Update grid configuration
        cols, rows = self.size_configs.get(screen_size, (2, 2))
        self.styles.grid_size_columns = cols
        self.styles.grid_size_rows = rows
        
        # Update CSS class
        self.remove_class("xs", "sm", "md", "lg", "xl")
        self.add_class(screen_size.value)
    
    def _determine_screen_size(self, width: int) -> ScreenSize:
        """Determine screen size from width"""
        if width < 40:
            return ScreenSize.EXTRA_SMALL
        elif width < 80:
            return ScreenSize.SMALL
        elif width < 120:
            return ScreenSize.MEDIUM
        elif width < 160:
            return ScreenSize.LARGE
        else:
            return ScreenSize.EXTRA_LARGE


class CollapsiblePanel(Container):
    """Panel that can collapse/expand based on space constraints"""
    
    DEFAULT_CSS = """
    CollapsiblePanel {
        background: $surface;
        border: round $border;
        padding: 1;
        transition: width 300ms ease-out;
    }
    
    CollapsiblePanel.collapsed {
        width: 3;
        padding: 0 1;
        overflow: hidden;
    }
    
    CollapsiblePanel.expanded {
        width: auto;
        padding: 1;
    }
    
    .collapse-toggle {
        dock: top-right;
        width: 3;
        height: 1;
        background: transparent;
        color: $primary;
    }
    
    .collapse-toggle:hover {
        background: $primary-alpha-20;
    }
    
    .panel-content {
        height: 1fr;
    }
    
    .panel-content.collapsed {
        display: none;
    }
    """
    
    collapsed: reactive[bool] = reactive(False)
    
    def __init__(
        self,
        title: str,
        *,
        collapsible: bool = True,
        auto_collapse: bool = True,
        min_width_to_show: int = 60,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.collapsible = collapsible
        self.auto_collapse = auto_collapse
        self.min_width_to_show = min_width_to_show
        self.content_container: Optional[Container] = None
        
    def compose(self):
        """Compose collapsible panel"""
        # Toggle button
        if self.collapsible:
            from ..widgets.advanced_components import EnhancedButton
            toggle_btn = EnhancedButton(
                "◀" if not self.collapsed else "▶",
                classes="collapse-toggle"
            )
            toggle_btn.add_click_handler(self.toggle_collapse)
            yield toggle_btn
        
        # Content container
        self.content_container = Container(classes="panel-content")
        yield self.content_container
    
    def watch_collapsed(self, collapsed: bool) -> None:
        """React to collapse state changes"""
        if collapsed:
            self.add_class("collapsed")
            if self.content_container:
                self.content_container.add_class("collapsed")
        else:
            self.remove_class("collapsed")
            if self.content_container:
                self.content_container.remove_class("collapsed")
        
        # Update toggle button
        toggle_btn = self.query_one(".collapse-toggle", EnhancedButton)
        if toggle_btn:
            toggle_btn.label = "▶" if collapsed else "◀"
            toggle_btn.refresh()
    
    def toggle_collapse(self) -> None:
        """Toggle collapse state"""
        self.collapsed = not self.collapsed
        self.post_message(PanelCollapsed(self.collapsed))
    
    def on_resize(self, event: events.Resize) -> None:
        """Handle resize - auto-collapse if needed"""
        if self.auto_collapse:
            if event.size.width < self.min_width_to_show and not self.collapsed:
                self.collapsed = True
            elif event.size.width >= self.min_width_to_show and self.collapsed:
                self.collapsed = False
    
    def add_content(self, widget: Widget) -> None:
        """Add widget to panel content"""
        if self.content_container:
            self.content_container.mount(widget)


class ViewportManager:
    """Manages responsive behavior across the entire application"""
    
    def __init__(self, app: App):
        self.app = app
        self.current_size: ScreenSize = ScreenSize.MEDIUM
        self.responsive_containers: List[ResponsiveContainer] = []
        self.size_observers: List[Callable[[ScreenSize], None]] = []
        
        # Monitor app resize
        self.app.bind("resize", self.on_app_resize)
    
    def register_container(self, container: ResponsiveContainer) -> None:
        """Register responsive container"""
        self.responsive_containers.append(container)
        container.add_size_observer(self._on_container_size_change)
    
    def add_size_observer(self, callback: Callable[[ScreenSize], None]) -> None:
        """Add global size observer"""
        self.size_observers.append(callback)
    
    def on_app_resize(self, event: events.Resize) -> None:
        """Handle application resize"""
        new_size = self._determine_screen_size(event.size.width)
        if new_size != self.current_size:
            self.current_size = new_size
            self._notify_size_change(new_size)
    
    def _determine_screen_size(self, width: int) -> ScreenSize:
        """Determine screen size from width"""
        if width < Breakpoint.XS.value:
            return ScreenSize.EXTRA_SMALL
        elif width < Breakpoint.SM.value:
            return ScreenSize.SMALL
        elif width < Breakpoint.MD.value:
            return ScreenSize.MEDIUM
        elif width < Breakpoint.LG.value:
            return ScreenSize.LARGE
        else:
            return ScreenSize.EXTRA_LARGE
    
    def _notify_size_change(self, size: ScreenSize) -> None:
        """Notify all observers of size change"""
        for observer in self.size_observers:
            observer(size)
    
    def _on_container_size_change(self, size: ScreenSize) -> None:
        """Handle container size change"""
        # Could implement container-specific logic here
        pass
    
    def get_current_breakpoint(self) -> Breakpoint:
        """Get current breakpoint"""
        size_to_breakpoint = {
            ScreenSize.EXTRA_SMALL: Breakpoint.XS,
            ScreenSize.SMALL: Breakpoint.SM,
            ScreenSize.MEDIUM: Breakpoint.MD,
            ScreenSize.LARGE: Breakpoint.LG,
            ScreenSize.EXTRA_LARGE: Breakpoint.LG,
        }
        return size_to_breakpoint.get(self.current_size, Breakpoint.MD)
    
    def is_mobile_size(self) -> bool:
        """Check if current size is mobile-like"""
        return self.current_size in [ScreenSize.EXTRA_SMALL, ScreenSize.SMALL]
    
    def is_desktop_size(self) -> bool:
        """Check if current size is desktop-like"""
        return self.current_size in [ScreenSize.LARGE, ScreenSize.EXTRA_LARGE]


# Message classes
class ResponsiveLayoutChanged(Message):
    """Message sent when responsive layout changes"""
    
    def __init__(self, size: ScreenSize) -> None:
        super().__init__()
        self.size = size


class PanelCollapsed(Message):
    """Message sent when panel collapses/expands"""
    
    def __init__(self, collapsed: bool) -> None:
        super().__init__()
        self.collapsed = collapsed


# Global viewport manager
_viewport_manager: Optional[ViewportManager] = None

def get_viewport_manager(app: App) -> ViewportManager:
    """Get global viewport manager instance"""
    global _viewport_manager
    if _viewport_manager is None or _viewport_manager.app != app:
        _viewport_manager = ViewportManager(app)
    return _viewport_manager

def setup_responsive_app(app: App) -> ViewportManager:
    """Setup responsive behavior for an app"""
    return get_viewport_manager(app)