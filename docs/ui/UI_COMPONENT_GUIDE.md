# Claude-TUI UI Component Guide

## Overview

This guide provides comprehensive documentation for the modernized Claude-TUI UI component system. The enhanced UI framework includes advanced terminal widgets, responsive layouts, accessibility features, and performance optimizations.

## Architecture

### Component Hierarchy

```
src/ui/
├── themes/                    # Theme management system
│   ├── theme_manager.py      # Modern theme system with dark/light modes
│   └── styles/
│       ├── themes/
│       │   ├── dark_modern.tcss
│       │   └── light_modern.tcss
├── widgets/                   # Enhanced widget library
│   ├── advanced_components.py      # Modern UI components
│   ├── enhanced_terminal_components.py  # Advanced terminal widgets
│   ├── metrics_dashboard.py        # Real-time metrics display
│   ├── progress_intelligence.py    # Progress validation widgets
│   └── task_dashboard.py          # Task management widgets
├── layouts/                   # Responsive layout system
│   └── responsive_layout.py  # Adaptive layouts for different sizes
├── accessibility/             # Accessibility features
│   └── accessibility_features.py  # Screen reader support, high contrast
├── performance/               # Performance optimization
│   └── ui_optimization.py    # Rendering optimization, caching
├── keyboard_navigation.py    # Advanced keyboard navigation
└── main_app.py               # Enhanced main application
```

## Theme System

### Theme Manager

The theme system provides comprehensive theming with multiple built-in themes and support for custom themes.

```python
from src.ui.themes.theme_manager import get_theme_manager, ThemeMode

# Get theme manager
theme_manager = get_theme_manager()

# Available themes
themes = theme_manager.get_available_themes()
# ['dark_modern', 'light_modern', 'high_contrast', 'cyberpunk', 'matrix', 'retro']

# Set theme
theme_manager.set_theme('dark_modern')

# Create custom theme
from src.ui.themes.theme_manager import Theme, ColorPalette, Typography

custom_theme = Theme(
    name="my_custom_theme",
    mode=ThemeMode.DARK,
    colors=ColorPalette(
        primary="#ff6b35",
        secondary="#4ecdc4",
        background="#2e2e2e",
        # ... other colors
    ),
    typography=Typography(),
    spacing=Spacing(),
    animations=AnimationSettings()
)

theme_manager.create_custom_theme(custom_theme)
```

### Built-in Themes

1. **Dark Modern** - Professional dark theme with blue accents
2. **Light Modern** - Clean light theme with blue accents
3. **High Contrast** - Accessibility-focused high contrast theme
4. **Cyberpunk** - Neon aesthetic with pink/cyan colors
5. **Matrix** - Green-on-black matrix digital rain theme
6. **Retro** - Warm retro color scheme

## Advanced Components

### Enhanced Button

Modern button component with hover effects, icons, and loading states.

```python
from src.ui.widgets.advanced_components import EnhancedButton

# Basic button
button = EnhancedButton("Click Me")

# Button with icon and variant
button = EnhancedButton(
    label="Save File",
    icon="💾",
    variant="primary",
    size="medium"
)

# Loading state
button.set_loading(True)

# Disabled state
button.set_disabled(True)

# Add click handler
def on_click():
    print("Button clicked!")

button.add_click_handler(on_click)
```

### Status Card

Displays status information with icon, value, and trend.

```python
from src.ui.widgets.advanced_components import StatusCard

card = StatusCard(
    title="CPU Usage",
    value="45%",
    icon="💻",
    trend="↗️ +2%",
    trend_direction="up",
    color="primary"
)

# Update value
card.update_value("50%", "↗️ +5%")
```

### Interactive Chart

Real-time ASCII chart with multiple data series.

```python
from src.ui.widgets.advanced_components import InteractiveChart

chart = InteractiveChart(
    title="Performance Metrics",
    chart_type="line",
    max_points=100,
    show_legend=True
)

# Add data series
chart.add_series("CPU", color="red", style="line")
chart.add_series("Memory", color="blue", style="line")

# Add data points
chart.add_data_point("CPU", time.time(), 45.2)
chart.add_data_point("Memory", time.time(), 67.8)
```

### Smart Data Table

Enhanced data table with sorting, filtering, and pagination.

```python
from src.ui.widgets.advanced_components import SmartDataTable

table = SmartDataTable(
    columns=["Name", "Status", "Progress", "Time"],
    sortable=True,
    filterable=True,
    paginated=True,
    page_size=20
)

# Set data
data = [
    ["Task 1", "Running", "75%", "2m 30s"],
    ["Task 2", "Completed", "100%", "1m 15s"],
    # ... more rows
]
table.set_data(data)

# Add single row
table.add_row(["Task 3", "Pending", "0%", "0s"])
```

### Command Palette

VS Code-style command palette for quick actions.

```python
from src.ui.widgets.advanced_components import CommandPalette

palette = CommandPalette()

# Register commands
palette.register_command(
    name="New Project",
    description="Create a new project",
    handler=lambda: create_new_project()
)

palette.register_command(
    name="Open Settings",
    description="Open application settings",
    handler=lambda: open_settings()
)

# Toggle visibility
palette.toggle()
```

## Enhanced Terminal Components

### Tree View

Advanced tree view with lazy loading and custom icons.

```python
from src.ui.widgets.enhanced_terminal_components import TreeView

# Tree data structure
tree_data = {
    "src": {
        "ui": {
            "widgets": {
                "button.py": None,
                "chart.py": None
            },
            "themes": {
                "dark.tcss": None,
                "light.tcss": None
            }
        },
        "core": {
            "app.py": None
        }
    }
}

tree = TreeView(
    data=tree_data,
    lazy_loading=True,
    show_icons=True
)

# Handle node selection
@tree.on(TreeNodeSelected)
def node_selected(event):
    print(f"Selected: {event.node_path}")

# Expand/collapse nodes
tree.expand_node("src/ui")
tree.collapse_node("src/core")
```

### Split Pane

Resizable split pane container for complex layouts.

```python
from src.ui.widgets.enhanced_terminal_components import SplitPane

left_widget = MyLeftPanel()
right_widget = MyRightPanel()

split_pane = SplitPane(
    left_widget=left_widget,
    right_widget=right_widget,
    orientation="horizontal",
    initial_ratio=0.3,
    resizable=True
)
```

### Tab Container

Enhanced tab container with closeable tabs.

```python
from src.ui.widgets.enhanced_terminal_components import TabContainer

tab_container = TabContainer()

# Add tabs
tab_container.add_tab("Editor", EditorWidget(), closeable=True)
tab_container.add_tab("Terminal", TerminalWidget(), closeable=True)
tab_container.add_tab("Debug", DebugWidget(), closeable=False)

# Handle tab events
@tab_container.on(TabChanged)
def tab_changed(event):
    print(f"Switched to tab {event.tab_index}")
```

### Graph Widget

Real-time graph with multiple data series.

```python
from src.ui.widgets.enhanced_terminal_components import GraphWidget

graph = GraphWidget(
    title="System Metrics",
    max_points=200,
    y_min=0,
    y_max=100,
    show_legend=True
)

# Add data series
graph.add_series("CPU", color="red", style="line")
graph.add_series("Memory", color="blue", style="line")

# Add real-time data
import time
graph.add_data_point("CPU", time.time(), cpu_usage)
graph.add_data_point("Memory", time.time(), memory_usage)
```

## Responsive Layouts

### Responsive Container

Container that adapts layout based on terminal size.

```python
from src.ui.layouts.responsive_layout import ResponsiveContainer, LayoutRule

container = ResponsiveContainer()

# Register widgets for responsive behavior
container.register_responsive_widget("sidebar", sidebar_widget)
container.register_responsive_widget("main-content", main_widget)
container.register_responsive_widget("right-panel", right_panel_widget)

# Add custom layout rule
custom_rule = LayoutRule(
    min_width=120,
    max_width=160,
    widget_classes=["large", "expanded"],
    show_widgets=["sidebar", "main-content", "right-panel"],
    layout_type="horizontal"
)
container.add_layout_rule(custom_rule)

# Listen for size changes
@container.on(ResponsiveLayoutChanged)
def layout_changed(event):
    print(f"Layout changed to: {event.size}")
```

### Flex Container

Flexbox-like behavior for terminal UIs.

```python
from src.ui.layouts.responsive_layout import FlexContainer

flex_container = FlexContainer(
    direction="row",
    justify_content="space-between",
    align_items="center",
    gap=1
)

# Add flex items
flex_container.add_flex_item(widget1, grow=1, shrink=0)
flex_container.add_flex_item(widget2, grow=2, shrink=1)
flex_container.add_flex_item(widget3, basis="20%")
```

### Adaptive Grid

Grid that adapts column count based on screen size.

```python
from src.ui.layouts.responsive_layout import AdaptiveGrid

grid = AdaptiveGrid()

# Grid automatically adjusts columns:
# XS: 1 column
# SM: 2 columns
# MD: 2x2 grid
# LG: 3x2 grid
# XL: 4x3 grid

# Add items
grid.mount(widget1)
grid.mount(widget2)
grid.mount(widget3)
```

## Keyboard Navigation

### Navigation Manager

Advanced keyboard navigation with Vim-style shortcuts.

```python
from src.ui.keyboard_navigation import setup_app_navigation, NavigationMode

# Setup navigation for app
nav_manager = setup_app_navigation(app, NavigationMode.VIM)

# Register custom key binding
from src.ui.keyboard_navigation import KeyBinding, KeyContext

custom_binding = KeyBinding(
    key="ctrl+shift+p",
    action="command_palette",
    description="Open command palette",
    context=KeyContext.GLOBAL,
    handler=lambda widget, key, count: open_command_palette()
)

nav_manager.register_binding(custom_binding)

# Create focus groups
nav_manager.register_focus_group("sidebar", [tree_widget, file_list])
nav_manager.register_focus_group("main", [editor, console])
```

### Built-in Key Bindings

#### Global Navigation
- `Tab` / `Shift+Tab` - Focus next/previous widget
- `Ctrl+Tab` / `Ctrl+Shift+Tab` - Focus next/previous group
- `F1` - Help
- `Ctrl+,` - Settings
- `Ctrl+Q` - Quit

#### Vim Mode
- `h/j/k/l` - Move left/down/up/right
- `w/b/e` - Word navigation
- `0/$` - Line start/end
- `gg/G` - Go to top/bottom
- `:/` - Search
- `ctrl+w` + `h/j/k/l` - Window navigation

## Accessibility Features

### Accessibility Manager

Comprehensive accessibility support with screen reader compatibility.

```python
from src.ui.accessibility.accessibility_features import setup_accessibility

# Setup accessibility
accessibility_manager = setup_accessibility(app)

# Enable high contrast
accessibility_manager.enable_high_contrast()

# Enable screen reader support
accessibility_manager.toggle_screen_reader_mode()

# Make announcements
accessibility_manager.announce("Task completed successfully", priority="assertive")

# Register ARIA live region
accessibility_manager.register_aria_live_region("status", status_widget)
accessibility_manager.update_live_region("status", "Processing 50% complete")
```

### Accessible Widget Mixin

Add accessibility features to custom widgets.

```python
from src.ui.accessibility.accessibility_features import AccessibilityWidget

class MyCustomWidget(AccessibilityWidget, Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set accessibility properties
        self.set_accessibility_role("button")
        self.set_accessibility_label("Save Document")
        self.set_accessibility_description("Saves the current document to disk")
        self.set_accessibility_state("disabled", False)
```

## Performance Optimization

### Performance Optimizer

Automatic performance optimization with caching and virtual scrolling.

```python
from src.ui.performance.ui_optimization import setup_performance_optimization

# Setup performance optimization
optimizer = setup_performance_optimization(app)

# Get performance summary
summary = optimizer.get_performance_summary()
print(f"Average FPS: {summary['average_fps']:.1f}")
print(f"Cache hit rate: {summary['cache_stats']['hit_rate']:.2%}")

# Force cleanup
cleanup_results = optimizer.force_cleanup()
print(f"Freed {cleanup_results['memory_freed_bytes']} bytes")

# Optimize specific widget
optimizer.optimize_widget(my_large_list_widget)
```

### Virtual Scrolling

For large lists, virtual scrolling is automatically applied:

```python
from src.ui.performance.ui_optimization import VirtualScrollContainer

# Create virtual scroll container for large dataset
def render_item(index: int) -> Widget:
    return ListItem(f"Item {index}")

virtual_list = VirtualScrollContainer(
    item_height=3,
    visible_items=20,
    total_items=10000,
    item_renderer=render_item
)
```

## Usage Examples

### Complete Application Example

```python
from textual.app import App
from src.ui.themes.theme_manager import get_theme_manager
from src.ui.keyboard_navigation import setup_app_navigation, NavigationMode
from src.ui.accessibility.accessibility_features import setup_accessibility
from src.ui.performance.ui_optimization import setup_performance_optimization
from src.ui.layouts.responsive_layout import ResponsiveContainer
from src.ui.widgets.advanced_components import *

class ModernClaudeTUIApp(App):
    def __init__(self):
        super().__init__()
        
        # Setup theme
        theme_manager = get_theme_manager()
        theme_manager.set_theme('dark_modern')
        
        # Setup navigation
        self.nav_manager = setup_app_navigation(self, NavigationMode.VIM)
        
        # Setup accessibility
        self.accessibility = setup_accessibility(self)
        
        # Setup performance optimization
        self.optimizer = setup_performance_optimization(self)
    
    def compose(self):
        # Main responsive container
        main_container = ResponsiveContainer()
        
        # Sidebar with tree view
        sidebar = TreeView(data=self.get_project_structure())
        main_container.register_responsive_widget("sidebar", sidebar)
        
        # Main content area with tabs
        tab_container = TabContainer()
        tab_container.add_tab("Editor", self.create_editor())
        tab_container.add_tab("Console", self.create_console())
        main_container.register_responsive_widget("main-content", tab_container)
        
        # Right panel with metrics
        metrics_panel = StatusCard("System Status", "Online", icon="🟢")
        main_container.register_responsive_widget("right-panel", metrics_panel)
        
        yield main_container
        
        # Command palette (hidden by default)
        self.command_palette = CommandPalette()
        self._register_commands()
        yield self.command_palette
    
    def _register_commands(self):
        self.command_palette.register_command(
            "toggle_theme",
            "Toggle between dark and light theme",
            self.toggle_theme
        )
        
        self.command_palette.register_command(
            "show_metrics",
            "Show performance metrics",
            self.show_performance_metrics
        )
    
    def on_key(self, event):
        # Custom key handling
        if event.key == "ctrl+shift+p":
            self.command_palette.toggle()
        else:
            super().on_key(event)
    
    def toggle_theme(self):
        theme_manager = get_theme_manager()
        current = theme_manager.get_current_theme()
        new_theme = "light_modern" if current.name == "dark_modern" else "dark_modern"
        theme_manager.set_theme(new_theme)
        self.accessibility.announce(f"Theme changed to {new_theme}")
    
    def show_performance_metrics(self):
        summary = self.optimizer.get_performance_summary()
        # Display metrics in a modal or status area
        pass

if __name__ == "__main__":
    app = ModernClaudeTUIApp()
    app.run()
```

### Custom Widget Example

```python
from textual.widgets import Widget
from src.ui.accessibility.accessibility_features import AccessibilityWidget
from src.ui.themes.theme_manager import get_theme_manager

class CustomStatusWidget(AccessibilityWidget, Widget):
    DEFAULT_CSS = """
    CustomStatusWidget {
        background: $surface;
        border: round $border;
        padding: 1;
        height: 5;
    }
    
    CustomStatusWidget:hover {
        border: round $primary;
        background: $surface-light;
    }
    
    CustomStatusWidget.error {
        border: round $error;
        background: $error-alpha-20;
    }
    """
    
    def __init__(self, status: str, **kwargs):
        super().__init__(**kwargs)
        self.status = status
        
        # Setup accessibility
        self.set_accessibility_role("status")
        self.set_accessibility_label(f"System status: {status}")
        self.set_accessibility_state("live", True)
        
    def render(self):
        # Get current theme colors
        theme_manager = get_theme_manager()
        theme = theme_manager.get_current_theme()
        
        # Render based on status
        if self.status == "error":
            self.add_class("error")
            icon = "❌"
        elif self.status == "success":
            icon = "✅"
        else:
            icon = "ℹ️"
        
        return Panel(f"{icon} Status: {self.status}", title="System Status")
    
    def update_status(self, new_status: str):
        old_status = self.status
        self.status = new_status
        
        # Update accessibility label
        self.set_accessibility_label(f"System status: {new_status}")
        
        # Announce change
        from src.ui.accessibility.accessibility_features import get_accessibility_manager
        accessibility = get_accessibility_manager(self.app)
        accessibility.announce(f"Status changed from {old_status} to {new_status}")
        
        self.refresh()
```

## Best Practices

### Theme Development

1. **Color Consistency**: Use theme variables consistently across components
2. **Accessibility**: Ensure sufficient contrast ratios (4.5:1 for AA compliance)
3. **Dark/Light Variants**: Provide both dark and light theme variants
4. **State Variations**: Include hover, focus, and disabled states

### Performance Optimization

1. **Large Lists**: Use virtual scrolling for lists with >100 items
2. **Render Caching**: Enable caching for complex rendering operations
3. **Memory Management**: Register widgets with memory manager for monitoring
4. **Batch Updates**: Use update batching for frequent UI changes

### Accessibility Guidelines

1. **Keyboard Navigation**: Ensure all interactive elements are keyboard accessible
2. **Screen Readers**: Provide meaningful accessibility labels and descriptions
3. **Focus Management**: Implement proper focus trapping in modals
4. **Color**: Don't rely solely on color to convey information

### Responsive Design

1. **Mobile First**: Design for smallest screens first, then enhance for larger
2. **Breakpoints**: Use consistent breakpoints (40, 80, 120, 160 columns)
3. **Progressive Enhancement**: Add features progressively as screen size increases
4. **Content Priority**: Hide less important content on smaller screens

## API Reference

### Theme Manager
- `get_theme_manager()` - Get global theme manager
- `set_theme(name)` - Set active theme
- `create_custom_theme(theme)` - Create custom theme
- `get_available_themes()` - List available themes

### Keyboard Navigation
- `setup_app_navigation(app, mode)` - Setup navigation
- `register_binding(binding)` - Register key binding
- `set_navigation_mode(mode)` - Change navigation mode

### Accessibility
- `setup_accessibility(app)` - Setup accessibility
- `announce(text, priority)` - Make announcement
- `enable_high_contrast()` - Enable high contrast
- `toggle_screen_reader_mode()` - Toggle screen reader

### Performance
- `setup_performance_optimization(app)` - Setup optimization
- `optimize_widget(widget)` - Optimize specific widget
- `get_performance_summary()` - Get performance metrics
- `force_cleanup()` - Force memory cleanup

## Troubleshooting

### Common Issues

1. **Theme not applying**: Ensure CSS variables are properly defined
2. **Keyboard shortcuts not working**: Check key binding registration and context
3. **Performance issues**: Enable virtual scrolling for large lists
4. **Accessibility problems**: Validate with accessibility audit tools

### Debug Mode

Enable debug mode to see performance metrics and accessibility warnings:

```python
app = MyApp()
app.debug = True  # Enable debug mode
app.run()
```

### Performance Monitoring

Monitor performance in real-time:

```python
optimizer = get_performance_optimizer(app)
summary = optimizer.get_performance_summary()

# Check metrics
if summary['average_fps'] < 30:
    print("Low frame rate detected!")
    
if summary['cache_stats']['hit_rate'] < 0.5:
    print("Low cache hit rate!")
```

## Contributing

When contributing to the UI system:

1. Follow the established component patterns
2. Include accessibility features in new widgets
3. Add responsive behavior where appropriate
4. Write comprehensive documentation
5. Include usage examples
6. Test with different themes and screen sizes

For detailed implementation guides and advanced topics, see the individual component documentation files.