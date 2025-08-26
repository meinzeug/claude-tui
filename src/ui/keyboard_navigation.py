#!/usr/bin/env python3
"""
Advanced Keyboard Navigation System - Vim-style navigation and shortcuts
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum

from textual import events
from textual.app import App
from textual.widget import Widget
from textual.binding import Binding
from textual.message import Message


class NavigationMode(Enum):
    """Navigation modes"""
    NORMAL = "normal"
    VIM = "vim" 
    EMACS = "emacs"
    CUSTOM = "custom"


class KeyContext(Enum):
    """Key binding contexts"""
    GLOBAL = "global"
    MODAL = "modal"
    INPUT = "input"
    LIST = "list"
    TREE = "tree"
    TABLE = "table"
    EDITOR = "editor"


@dataclass
class KeyBinding:
    """Key binding definition"""
    key: str
    action: str
    description: str
    context: KeyContext = KeyContext.GLOBAL
    mode: NavigationMode = NavigationMode.NORMAL
    handler: Optional[Callable] = None
    condition: Optional[Callable] = None


@dataclass
class KeySequence:
    """Multi-key sequence state"""
    keys: List[str]
    timestamp: float
    timeout: float = 1.0


class KeyboardNavigationManager:
    """Advanced keyboard navigation and shortcut system"""
    
    def __init__(self, app: App):
        self.app = app
        self.current_mode = NavigationMode.NORMAL
        self.bindings: Dict[Tuple[str, KeyContext, NavigationMode], KeyBinding] = {}
        self.key_sequences: Dict[str, KeySequence] = {}
        self.current_sequence: Optional[str] = None
        
        # Focus management
        self.focus_stack: List[Widget] = []
        self.focus_groups: Dict[str, List[Widget]] = {}
        
        # Setup default bindings
        self._setup_default_bindings()
        
        # Vim mode state
        self.vim_command_buffer = ""
        self.vim_repeat_count = ""
        
    def _setup_default_bindings(self):
        """Setup default key bindings for different modes"""
        
        # Global navigation bindings
        global_bindings = [
            # Application-level
            KeyBinding("ctrl+q", "quit", "Quit application"),
            KeyBinding("ctrl+c", "cancel", "Cancel current operation"),
            KeyBinding("escape", "cancel", "Cancel/Back"),
            KeyBinding("f1", "help", "Show help"),
            KeyBinding("ctrl+comma", "settings", "Open settings"),
            
            # Navigation
            KeyBinding("tab", "focus_next", "Focus next widget"),
            KeyBinding("shift+tab", "focus_previous", "Focus previous widget"),
            KeyBinding("ctrl+tab", "focus_next_group", "Focus next group"),
            KeyBinding("ctrl+shift+tab", "focus_previous_group", "Focus previous group"),
            
            # Window management
            KeyBinding("f11", "toggle_fullscreen", "Toggle fullscreen"),
            KeyBinding("ctrl+shift+i", "toggle_debug", "Toggle debug mode"),
        ]
        
        # Vim-style navigation
        vim_bindings = [
            # Movement (normal mode)
            KeyBinding("h", "move_left", "Move left", mode=NavigationMode.VIM),
            KeyBinding("j", "move_down", "Move down", mode=NavigationMode.VIM),
            KeyBinding("k", "move_up", "Move up", mode=NavigationMode.VIM),
            KeyBinding("l", "move_right", "Move right", mode=NavigationMode.VIM),
            
            # Word movement
            KeyBinding("w", "word_forward", "Word forward", mode=NavigationMode.VIM),
            KeyBinding("b", "word_backward", "Word backward", mode=NavigationMode.VIM),
            KeyBinding("e", "word_end", "Word end", mode=NavigationMode.VIM),
            
            # Line movement
            KeyBinding("0", "line_start", "Line start", mode=NavigationMode.VIM),
            KeyBinding("$", "line_end", "Line end", mode=NavigationMode.VIM),
            KeyBinding("^", "line_first_char", "First non-blank", mode=NavigationMode.VIM),
            
            # Page movement
            KeyBinding("ctrl+f", "page_down", "Page down", mode=NavigationMode.VIM),
            KeyBinding("ctrl+b", "page_up", "Page up", mode=NavigationMode.VIM),
            KeyBinding("ctrl+d", "half_page_down", "Half page down", mode=NavigationMode.VIM),
            KeyBinding("ctrl+u", "half_page_up", "Half page up", mode=NavigationMode.VIM),
            
            # Jump commands
            KeyBinding("g,g", "goto_top", "Go to top", mode=NavigationMode.VIM),
            KeyBinding("G", "goto_bottom", "Go to bottom", mode=NavigationMode.VIM),
            KeyBinding(":", "command_mode", "Command mode", mode=NavigationMode.VIM),
            KeyBinding("/", "search_forward", "Search forward", mode=NavigationMode.VIM),
            KeyBinding("?", "search_backward", "Search backward", mode=NavigationMode.VIM),
            KeyBinding("n", "search_next", "Next search result", mode=NavigationMode.VIM),
            KeyBinding("N", "search_previous", "Previous search result", mode=NavigationMode.VIM),
            
            # Edit commands
            KeyBinding("i", "insert_mode", "Insert mode", mode=NavigationMode.VIM),
            KeyBinding("a", "append_mode", "Append mode", mode=NavigationMode.VIM),
            KeyBinding("o", "open_line_below", "Open line below", mode=NavigationMode.VIM),
            KeyBinding("O", "open_line_above", "Open line above", mode=NavigationMode.VIM),
            KeyBinding("x", "delete_char", "Delete character", mode=NavigationMode.VIM),
            KeyBinding("d,d", "delete_line", "Delete line", mode=NavigationMode.VIM),
            KeyBinding("y,y", "yank_line", "Yank line", mode=NavigationMode.VIM),
            KeyBinding("p", "paste_after", "Paste after", mode=NavigationMode.VIM),
            KeyBinding("P", "paste_before", "Paste before", mode=NavigationMode.VIM),
            
            # Visual mode
            KeyBinding("v", "visual_mode", "Visual mode", mode=NavigationMode.VIM),
            KeyBinding("V", "visual_line_mode", "Visual line mode", mode=NavigationMode.VIM),
            KeyBinding("ctrl+v", "visual_block_mode", "Visual block mode", mode=NavigationMode.VIM),
            
            # Windows/panels
            KeyBinding("ctrl+w,h", "window_left", "Window left", mode=NavigationMode.VIM),
            KeyBinding("ctrl+w,j", "window_down", "Window down", mode=NavigationMode.VIM),
            KeyBinding("ctrl+w,k", "window_up", "Window up", mode=NavigationMode.VIM),
            KeyBinding("ctrl+w,l", "window_right", "Window right", mode=NavigationMode.VIM),
            KeyBinding("ctrl+w,s", "window_split", "Split window", mode=NavigationMode.VIM),
            KeyBinding("ctrl+w,v", "window_vsplit", "Vertical split", mode=NavigationMode.VIM),
            KeyBinding("ctrl+w,c", "window_close", "Close window", mode=NavigationMode.VIM),
        ]
        
        # List/Table specific bindings
        list_bindings = [
            KeyBinding("up", "list_up", "Move up", context=KeyContext.LIST),
            KeyBinding("down", "list_down", "Move down", context=KeyContext.LIST),
            KeyBinding("page_up", "list_page_up", "Page up", context=KeyContext.LIST),
            KeyBinding("page_down", "list_page_down", "Page down", context=KeyContext.LIST),
            KeyBinding("home", "list_home", "Go to top", context=KeyContext.LIST),
            KeyBinding("end", "list_end", "Go to bottom", context=KeyContext.LIST),
            KeyBinding("enter", "list_select", "Select item", context=KeyContext.LIST),
            KeyBinding("space", "list_toggle", "Toggle item", context=KeyContext.LIST),
        ]
        
        # Tree specific bindings
        tree_bindings = [
            KeyBinding("right", "tree_expand", "Expand node", context=KeyContext.TREE),
            KeyBinding("left", "tree_collapse", "Collapse node", context=KeyContext.TREE),
            KeyBinding("space", "tree_toggle", "Toggle expansion", context=KeyContext.TREE),
            KeyBinding("*", "tree_expand_all", "Expand all", context=KeyContext.TREE),
        ]
        
        # Editor bindings
        editor_bindings = [
            KeyBinding("ctrl+z", "undo", "Undo", context=KeyContext.EDITOR),
            KeyBinding("ctrl+y", "redo", "Redo", context=KeyContext.EDITOR),
            KeyBinding("ctrl+x", "cut", "Cut", context=KeyContext.EDITOR),
            KeyBinding("ctrl+c", "copy", "Copy", context=KeyContext.EDITOR),
            KeyBinding("ctrl+v", "paste", "Paste", context=KeyContext.EDITOR),
            KeyBinding("ctrl+a", "select_all", "Select all", context=KeyContext.EDITOR),
            KeyBinding("ctrl+f", "find", "Find", context=KeyContext.EDITOR),
            KeyBinding("ctrl+h", "replace", "Replace", context=KeyContext.EDITOR),
            KeyBinding("ctrl+s", "save", "Save", context=KeyContext.EDITOR),
            KeyBinding("ctrl+n", "new", "New", context=KeyContext.EDITOR),
            KeyBinding("ctrl+o", "open", "Open", context=KeyContext.EDITOR),
        ]
        
        # Register all bindings
        for binding_list in [global_bindings, vim_bindings, list_bindings, tree_bindings, editor_bindings]:
            for binding in binding_list:
                self.register_binding(binding)
    
    def register_binding(self, binding: KeyBinding) -> None:
        """Register a key binding"""
        key = (binding.key, binding.context, binding.mode)
        self.bindings[key] = binding
    
    def unregister_binding(self, key: str, context: KeyContext = KeyContext.GLOBAL, mode: NavigationMode = NavigationMode.NORMAL) -> None:
        """Unregister a key binding"""
        binding_key = (key, context, mode)
        if binding_key in self.bindings:
            del self.bindings[binding_key]
    
    def set_navigation_mode(self, mode: NavigationMode) -> None:
        """Set the current navigation mode"""
        self.current_mode = mode
        self.app.post_message(NavigationModeChanged(mode))
    
    def handle_key_event(self, widget: Widget, event: events.Key) -> bool:
        """Handle key event with advanced navigation"""
        key = event.key
        context = self._get_widget_context(widget)
        
        # Handle vim numeric prefixes
        if self.current_mode == NavigationMode.VIM:
            if key.isdigit() and not self.vim_command_buffer:
                self.vim_repeat_count += key
                return True
        
        # Check for multi-key sequences
        if self._handle_key_sequence(key, context):
            return True
        
        # Look for direct binding
        binding_key = (key, context, self.current_mode)
        if binding_key in self.bindings:
            binding = self.bindings[binding_key]
            
            # Check condition if present
            if binding.condition and not binding.condition(widget):
                return False
            
            # Execute action
            return self._execute_action(widget, binding, key)
        
        # Try global context if not found
        if context != KeyContext.GLOBAL:
            global_key = (key, KeyContext.GLOBAL, self.current_mode)
            if global_key in self.bindings:
                binding = self.bindings[global_key]
                return self._execute_action(widget, binding, key)
        
        # Try normal mode if in vim mode
        if self.current_mode == NavigationMode.VIM:
            normal_key = (key, context, NavigationMode.NORMAL)
            if normal_key in self.bindings:
                binding = self.bindings[normal_key]
                return self._execute_action(widget, binding, key)
        
        return False
    
    def _handle_key_sequence(self, key: str, context: KeyContext) -> bool:
        """Handle multi-key sequences"""
        import time
        
        current_time = time.time()
        
        # Clean up expired sequences
        expired = []
        for seq_key, sequence in self.key_sequences.items():
            if current_time - sequence.timestamp > sequence.timeout:
                expired.append(seq_key)
        
        for seq_key in expired:
            del self.key_sequences[seq_key]
        
        # Handle current sequence
        if self.current_sequence:
            sequence = self.key_sequences.get(self.current_sequence)
            if sequence:
                sequence.keys.append(key)
                sequence.timestamp = current_time
                
                # Check for complete sequence
                sequence_str = ",".join(sequence.keys)
                binding_key = (sequence_str, context, self.current_mode)
                
                if binding_key in self.bindings:
                    # Execute sequence
                    binding = self.bindings[binding_key]
                    self._execute_action(None, binding, sequence_str)
                    
                    # Clean up
                    del self.key_sequences[self.current_sequence]
                    self.current_sequence = None
                    return True
                
                # Check if this could be start of longer sequence
                possible_sequences = [k for k in self.bindings.keys() 
                                    if k[0].startswith(sequence_str + ",")]
                
                if not possible_sequences:
                    # No matching sequences, abort
                    del self.key_sequences[self.current_sequence]
                    self.current_sequence = None
                    return False
                
                return True
        else:
            # Check if this key starts a sequence
            sequence_keys = [k for k in self.bindings.keys() 
                           if "," in k[0] and k[0].startswith(key + ",")]
            
            if sequence_keys:
                # Start new sequence
                sequence_id = f"seq_{current_time}"
                self.key_sequences[sequence_id] = KeySequence([key], current_time)
                self.current_sequence = sequence_id
                return True
        
        return False
    
    def _get_widget_context(self, widget: Widget) -> KeyContext:
        """Determine the context for a widget"""
        widget_type = type(widget).__name__.lower()
        
        if "input" in widget_type or "textarea" in widget_type:
            return KeyContext.INPUT
        elif "list" in widget_type or "listview" in widget_type:
            return KeyContext.LIST
        elif "tree" in widget_type:
            return KeyContext.TREE
        elif "table" in widget_type or "datatable" in widget_type:
            return KeyContext.TABLE
        elif "modal" in widget_type:
            return KeyContext.MODAL
        else:
            return KeyContext.GLOBAL
    
    def _execute_action(self, widget: Optional[Widget], binding: KeyBinding, key: str) -> bool:
        """Execute a key binding action"""
        # Get repeat count for vim mode
        repeat_count = 1
        if self.current_mode == NavigationMode.VIM and self.vim_repeat_count:
            try:
                repeat_count = int(self.vim_repeat_count)
            except ValueError:
                repeat_count = 1
            self.vim_repeat_count = ""
        
        # Execute custom handler if present
        if binding.handler:
            try:
                return binding.handler(widget, key, repeat_count)
            except Exception as e:
                print(f"Error in key handler: {e}")
                return False
        
        # Execute built-in actions
        return self._execute_builtin_action(widget, binding.action, repeat_count)
    
    def _execute_builtin_action(self, widget: Optional[Widget], action: str, repeat_count: int = 1) -> bool:
        """Execute built-in navigation actions"""
        try:
            # Application-level actions
            if action == "quit":
                self.app.exit()
                return True
            elif action == "cancel":
                if hasattr(self.app, 'action_cancel'):
                    self.app.action_cancel()
                else:
                    # Default cancel behavior
                    if len(self.focus_stack) > 1:
                        self.pop_focus()
                return True
            elif action == "help":
                self.show_help_overlay()
                return True
            elif action == "settings":
                if hasattr(self.app, 'action_settings'):
                    self.app.action_settings()
                return True
            
            # Focus management
            elif action == "focus_next":
                self.focus_next()
                return True
            elif action == "focus_previous":
                self.focus_previous()
                return True
            elif action == "focus_next_group":
                self.focus_next_group()
                return True
            elif action == "focus_previous_group":
                self.focus_previous_group()
                return True
            
            # Movement actions (vim-style)
            elif action in ["move_left", "move_right", "move_up", "move_down"]:
                return self._handle_movement_action(widget, action, repeat_count)
            
            # List/Table actions
            elif action.startswith("list_") or action.startswith("table_"):
                return self._handle_list_action(widget, action, repeat_count)
            
            # Tree actions
            elif action.startswith("tree_"):
                return self._handle_tree_action(widget, action, repeat_count)
            
            # Editor actions
            elif action in ["undo", "redo", "cut", "copy", "paste", "select_all"]:
                return self._handle_editor_action(widget, action, repeat_count)
            
            return False
            
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return False
    
    def _handle_movement_action(self, widget: Optional[Widget], action: str, repeat_count: int) -> bool:
        """Handle movement actions"""
        if not widget:
            return False
        
        for _ in range(repeat_count):
            if action == "move_left":
                if hasattr(widget, 'cursor_position') and hasattr(widget, 'move_cursor'):
                    widget.move_cursor(-1, 0)
                else:
                    self.focus_previous()
            elif action == "move_right":
                if hasattr(widget, 'cursor_position') and hasattr(widget, 'move_cursor'):
                    widget.move_cursor(1, 0)
                else:
                    self.focus_next()
            elif action == "move_up":
                if hasattr(widget, 'scroll_up'):
                    widget.scroll_up()
                elif hasattr(widget, 'cursor_position') and hasattr(widget, 'move_cursor'):
                    widget.move_cursor(0, -1)
            elif action == "move_down":
                if hasattr(widget, 'scroll_down'):
                    widget.scroll_down()
                elif hasattr(widget, 'cursor_position') and hasattr(widget, 'move_cursor'):
                    widget.move_cursor(0, 1)
        
        return True
    
    def _handle_list_action(self, widget: Optional[Widget], action: str, repeat_count: int) -> bool:
        """Handle list/table specific actions"""
        if not widget:
            return False
        
        # Implement list navigation
        return True
    
    def _handle_tree_action(self, widget: Optional[Widget], action: str, repeat_count: int) -> bool:
        """Handle tree specific actions"""
        if not widget:
            return False
        
        # Implement tree navigation
        return True
    
    def _handle_editor_action(self, widget: Optional[Widget], action: str, repeat_count: int) -> bool:
        """Handle editor specific actions"""
        if not widget:
            return False
        
        # Implement editor actions
        return True
    
    def focus_next(self) -> None:
        """Focus next focusable widget"""
        self.app.screen.focus_next()
    
    def focus_previous(self) -> None:
        """Focus previous focusable widget"""
        self.app.screen.focus_previous()
    
    def focus_next_group(self) -> None:
        """Focus next widget group"""
        # Implementation depends on group definitions
        pass
    
    def focus_previous_group(self) -> None:
        """Focus previous widget group"""
        # Implementation depends on group definitions
        pass
    
    def push_focus(self, widget: Widget) -> None:
        """Push widget onto focus stack"""
        self.focus_stack.append(widget)
        widget.focus()
    
    def pop_focus(self) -> Optional[Widget]:
        """Pop widget from focus stack"""
        if len(self.focus_stack) > 1:
            current = self.focus_stack.pop()
            previous = self.focus_stack[-1]
            previous.focus()
            return current
        return None
    
    def register_focus_group(self, name: str, widgets: List[Widget]) -> None:
        """Register a group of related widgets"""
        self.focus_groups[name] = widgets
    
    def show_help_overlay(self) -> None:
        """Show keyboard shortcuts help overlay"""
        help_content = self._generate_help_content()
        # Show help modal/overlay
        pass
    
    def _generate_help_content(self) -> str:
        """Generate help content for current mode and context"""
        lines = [f"Keyboard Shortcuts - {self.current_mode.value.title()} Mode\n"]
        
        # Group bindings by context
        contexts = {}
        for (key, context, mode), binding in self.bindings.items():
            if mode == self.current_mode:
                if context not in contexts:
                    contexts[context] = []
                contexts[context].append((key, binding.description))
        
        # Format help text
        for context, bindings in contexts.items():
            lines.append(f"\n{context.value.title()} Context:")
            for key, desc in sorted(bindings):
                lines.append(f"  {key:15} {desc}")
        
        return "\n".join(lines)
    
    def get_active_bindings(self, context: Optional[KeyContext] = None) -> List[KeyBinding]:
        """Get currently active key bindings"""
        if context is None:
            # Get focused widget context
            focused = self.app.screen.focused
            context = self._get_widget_context(focused) if focused else KeyContext.GLOBAL
        
        active_bindings = []
        for (key, bind_context, mode), binding in self.bindings.items():
            if (mode == self.current_mode or mode == NavigationMode.NORMAL) and \
               (bind_context == context or bind_context == KeyContext.GLOBAL):
                active_bindings.append(binding)
        
        return active_bindings


class NavigationModeChanged(Message):
    """Message sent when navigation mode changes"""
    
    def __init__(self, mode: NavigationMode) -> None:
        super().__init__()
        self.mode = mode


# Global navigation manager instance
_navigation_manager: Optional[KeyboardNavigationManager] = None

def get_navigation_manager(app: App) -> KeyboardNavigationManager:
    """Get global navigation manager instance"""
    global _navigation_manager
    if _navigation_manager is None or _navigation_manager.app != app:
        _navigation_manager = KeyboardNavigationManager(app)
    return _navigation_manager

def setup_app_navigation(app: App, mode: NavigationMode = NavigationMode.NORMAL) -> KeyboardNavigationManager:
    """Setup navigation for an app"""
    nav_manager = get_navigation_manager(app)
    nav_manager.set_navigation_mode(mode)
    
    # Override app's key handling
    original_on_key = app.on_key
    
    def enhanced_on_key(event: events.Key) -> None:
        focused = app.screen.focused
        if focused and nav_manager.handle_key_event(focused, event):
            return  # Event was handled by navigation manager
        
        # Fall back to original handler
        original_on_key(event)
    
    app.on_key = enhanced_on_key
    return nav_manager