"""Command Palette Widget for Claude-TUI."""

import logging
from typing import Dict, List, Any, Callable, Optional
import re

try:
    from textual.widgets import Input, ListView, ListItem, Label
    from textual.containers import Vertical
    from textual import events, on
    from textual.message import Message
    from textual.reactive import reactive
    
    from ...interfaces.ui_interfaces import PaletteInterface
    
    logger = logging.getLogger(__name__)
    
    class CommandPalette(Vertical):
        """Enhanced command palette for quick actions."""
        
        class CommandSelected(Message):
            """Message sent when a command is selected."""
            
            def __init__(self, sender: "CommandPalette", command: str) -> None:
                self.command = command
                super().__init__(sender)
        
        # Reactive for showing/hiding the palette
        visible = reactive(False)
        
        def __init__(self, commands=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._commands: Dict[str, Dict[str, Any]] = {}
            self._filtered_commands: List[Dict[str, Any]] = []
            self._input_widget: Optional[Input] = None
            self._list_widget: Optional[ListView] = None
            self._initialized = False
            
            # Add default commands if provided
            if commands:
                for cmd in commands:
                    if isinstance(cmd, dict):
                        self.add_command(cmd['name'], cmd.get('description', ''), cmd.get('callback'))
            
        def initialize(self) -> None:
            """Initialize the command palette."""
            if self._initialized:
                return
                
            try:
                # Set up default visibility
                self.visible = False
                
                # Initialize with empty filter
                self._update_filtered_commands("")
                
                self._initialized = True
                logger.debug("Command palette initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize command palette: {e}")
        
        def cleanup(self) -> None:
            """Cleanup palette resources."""
            self._commands.clear()
            self._filtered_commands.clear()
            logger.debug("Command palette cleanup completed")
        
        def set_focus(self) -> None:
            """Set focus to the palette input."""
            if self._input_widget and hasattr(self._input_widget, 'focus'):
                self._input_widget.focus()
        
        def compose(self):
            """Compose the command palette UI."""
            self._input_widget = Input(placeholder="Type command...", id="command_input")
            self._list_widget = ListView(id="command_list")
            
            yield self._input_widget
            yield self._list_widget
            
        def add_command(self, name: str, description: str, callback: Optional[Callable] = None) -> None:
            """Add a command to the palette."""
            self._commands[name] = {
                'name': name,
                'description': description or name,
                'callback': callback,
                'keywords': self._extract_keywords(name, description)
            }
            
            # Update filtered list if palette is visible
            if self.visible and self._input_widget:
                current_query = self._input_widget.value if hasattr(self._input_widget, 'value') else ""
                self._update_filtered_commands(current_query)
            
            logger.debug(f"Added command: {name}")
        
        def remove_command(self, name: str) -> None:
            """Remove a command from the palette."""
            if name in self._commands:
                del self._commands[name]
                
                # Update filtered list if palette is visible
                if self.visible and self._input_widget:
                    current_query = self._input_widget.value if hasattr(self._input_widget, 'value') else ""
                    self._update_filtered_commands(current_query)
                
                logger.debug(f"Removed command: {name}")
        
        def show(self) -> None:
            """Show the command palette."""
            try:
                self.visible = True
                self.display = True
                
                # Clear input and focus
                if self._input_widget:
                    if hasattr(self._input_widget, 'value'):
                        self._input_widget.value = ""
                    self.set_focus()
                
                # Update command list
                self._update_filtered_commands("")
                
                logger.debug("Command palette shown")
                
            except Exception as e:
                logger.error(f"Failed to show command palette: {e}")
        
        def hide(self) -> None:
            """Hide the command palette."""
            try:
                self.visible = False
                self.display = False
                logger.debug("Command palette hidden")
            except Exception as e:
                logger.error(f"Failed to hide command palette: {e}")
        
        def filter_commands(self, query: str) -> List[Dict[str, Any]]:
            """Filter commands by query string."""
            if not query.strip():
                return list(self._commands.values())
            
            query_lower = query.lower()
            filtered = []
            
            for cmd_data in self._commands.values():
                # Check name match
                if query_lower in cmd_data['name'].lower():
                    filtered.append(cmd_data)
                    continue
                
                # Check description match
                if query_lower in cmd_data['description'].lower():
                    filtered.append(cmd_data)
                    continue
                
                # Check keyword match
                if any(query_lower in keyword.lower() for keyword in cmd_data.get('keywords', [])):
                    filtered.append(cmd_data)
                    continue
            
            # Sort by relevance (name matches first, then description, then keywords)
            def sort_key(cmd):
                name_match = query_lower in cmd['name'].lower()
                desc_match = query_lower in cmd['description'].lower()
                if name_match:
                    return 0
                elif desc_match:
                    return 1
                else:
                    return 2
            
            filtered.sort(key=sort_key)
            return filtered
        
        def execute_command(self, name: str) -> None:
            """Execute a command by name."""
            if name in self._commands:
                cmd_data = self._commands[name]
                callback = cmd_data.get('callback')
                
                if callback:
                    try:
                        callback()
                        logger.debug(f"Executed command: {name}")
                    except Exception as e:
                        logger.error(f"Failed to execute command {name}: {e}")
                else:
                    logger.warning(f"No callback for command: {name}")
                
                # Send message
                self.post_message(self.CommandSelected(self, name))
                
                # Hide palette after execution
                self.hide()
            else:
                logger.warning(f"Command not found: {name}")
        
        @on(Input.Changed)
        def on_input_changed(self, event: Input.Changed) -> None:
            """Handle input changes to filter commands."""
            if event.input.id == "command_input":
                self._update_filtered_commands(event.value)
        
        @on(Input.Submitted)
        def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle input submission to execute first filtered command."""
            if event.input.id == "command_input" and self._filtered_commands:
                # Execute the first command in the filtered list
                first_cmd = self._filtered_commands[0]
                self.execute_command(first_cmd['name'])
        
        @on(ListView.Selected)
        def on_list_selected(self, event: ListView.Selected) -> None:
            """Handle list selection to execute command."""
            if event.list_view.id == "command_list" and event.item:
                # Extract command name from the selected item
                if hasattr(event.item, 'label'):
                    cmd_name = event.item.label.split(' - ')[0].strip()
                    self.execute_command(cmd_name)
        
        def _update_filtered_commands(self, query: str) -> None:
            """Update the filtered commands list."""
            try:
                self._filtered_commands = self.filter_commands(query)
                
                if self._list_widget:
                    # Clear current items
                    self._list_widget.clear()
                    
                    # Add filtered items
                    for cmd_data in self._filtered_commands:
                        name = cmd_data['name']
                        desc = cmd_data['description']
                        display_text = f"{name} - {desc}" if desc != name else name
                        
                        # Create list item
                        item = ListItem(Label(display_text))
                        item.label = display_text  # Store for selection handling
                        self._list_widget.append(item)
                
            except Exception as e:
                logger.error(f"Failed to update filtered commands: {e}")
        
        def _extract_keywords(self, name: str, description: str) -> List[str]:
            """Extract keywords from command name and description."""
            keywords = []
            
            # Split name and description into words
            text = f"{name} {description}".lower()
            words = re.findall(r'\w+', text)
            
            # Remove common words and duplicates
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in set(words) if word not in common_words and len(word) > 2]
            
            return keywords
        
        def get_all_commands(self) -> Dict[str, Dict[str, Any]]:
            """Get all registered commands."""
            return self._commands.copy()
        
        def get_command_count(self) -> int:
            """Get total number of registered commands."""
            return len(self._commands)
                
except ImportError as e:
    # Enhanced fallback implementation
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import Textual widgets: {e}. Using enhanced fallback.")
    
    from ...interfaces.ui_interfaces import PaletteInterface
    
    class CommandPalette(PaletteInterface):
        """Enhanced fallback command palette widget."""
        
        def __init__(self, commands=None, *args, **kwargs):
            self._commands: Dict[str, Dict[str, Any]] = {}
            self._visible = False
            self._initialized = False
            
            # Add default commands if provided
            if commands:
                for cmd in commands:
                    if isinstance(cmd, dict):
                        self.add_command(cmd['name'], cmd.get('description', ''), cmd.get('callback'))
        
        def initialize(self) -> None:
            """Initialize the fallback palette."""
            self._initialized = True
            logger.info("Fallback command palette initialized")
        
        def cleanup(self) -> None:
            """Cleanup palette resources."""
            self._commands.clear()
        
        def set_focus(self) -> None:
            """Set focus (no-op in fallback)."""
            pass
        
        def add_command(self, name: str, description: str, callback: Optional[Callable] = None) -> None:
            """Add a command to the palette."""
            self._commands[name] = {
                'name': name,
                'description': description or name,
                'callback': callback,
                'keywords': self._extract_keywords(name, description)
            }
            logger.info(f"Added command: {name}")
        
        def remove_command(self, name: str) -> None:
            """Remove a command from the palette."""
            if name in self._commands:
                del self._commands[name]
                logger.info(f"Removed command: {name}")
        
        def show(self) -> None:
            """Show the command palette (logged in fallback)."""
            self._visible = True
            logger.info("Would show command palette")
        
        def hide(self) -> None:
            """Hide the command palette (logged in fallback)."""
            self._visible = False
            logger.info("Would hide command palette")
        
        def filter_commands(self, query: str) -> List[Dict[str, Any]]:
            """Filter commands by query string."""
            if not query.strip():
                return list(self._commands.values())
            
            query_lower = query.lower()
            filtered = []
            
            for cmd_data in self._commands.values():
                if (query_lower in cmd_data['name'].lower() or 
                    query_lower in cmd_data['description'].lower()):
                    filtered.append(cmd_data)
            
            return filtered
        
        def execute_command(self, name: str) -> None:
            """Execute a command by name."""
            if name in self._commands:
                cmd_data = self._commands[name]
                callback = cmd_data.get('callback')
                
                if callback:
                    try:
                        callback()
                        logger.info(f"Executed command: {name}")
                    except Exception as e:
                        logger.error(f"Failed to execute command {name}: {e}")
                else:
                    logger.info(f"Would execute command: {name}")
            else:
                logger.warning(f"Command not found: {name}")
        
        def _extract_keywords(self, name: str, description: str) -> List[str]:
            """Extract keywords from command name and description."""
            text = f"{name} {description}".lower()
            words = re.findall(r'\w+', text)
            
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in set(words) if word not in common_words and len(word) > 2]
            
            return keywords
        
        def get_all_commands(self) -> Dict[str, Dict[str, Any]]:
            """Get all registered commands."""
            return self._commands.copy()
        
        def get_command_count(self) -> int:
            """Get total number of registered commands."""
            return len(self._commands)