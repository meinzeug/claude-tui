"""UI Component Interface Definitions."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class UIComponentInterface(ABC):
    """Base interface for all UI components."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""
        self._initialized = True
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup component resources."""
        self._initialized = False
        logger.debug(f"Cleaned up {self.__class__.__name__}")
    
    @abstractmethod
    def set_focus(self) -> None:
        """Set focus to this component."""
        if hasattr(self, '_widget'):
            self._widget.focus()
        logger.debug(f"Set focus to {self.__class__.__name__}")


class EditorInterface(UIComponentInterface):
    """Interface for code editor components."""
    
    @abstractmethod
    def set_text(self, text: str) -> None:
        """Set editor text content."""
        if hasattr(self, '_content'):
            self._content = text
            self._notify_text_changed(text)
        logger.debug(f"Set text content ({len(text)} chars)")
    
    @abstractmethod
    def get_text(self) -> str:
        """Get current editor text content."""
        return getattr(self, '_content', '')
    
    @abstractmethod
    def set_language(self, language: str) -> None:
        """Set syntax highlighting language."""
        self._language = language
        if hasattr(self, '_highlighter'):
            self._highlighter.set_language(language)
        logger.debug(f"Set language to {language}")
    
    @abstractmethod
    def goto_line(self, line_number: int) -> None:
        """Go to specific line number."""
        if hasattr(self, '_cursor'):
            self._cursor.line = max(0, line_number - 1)  # Convert to 0-based
            self._cursor.column = 0
        logger.debug(f"Moved to line {line_number}")
    
    @abstractmethod
    def highlight_line(self, line_number: int, style: str = "error") -> None:
        """Highlight a specific line."""
        if not hasattr(self, '_highlights'):
            self._highlights = {}
        self._highlights[line_number] = style
        logger.debug(f"Highlighted line {line_number} with style {style}")
    
    @abstractmethod
    def set_readonly(self, readonly: bool) -> None:
        """Set editor to readonly mode."""
        self._readonly = readonly
        logger.debug(f"Set readonly mode to {readonly}")
    
    @abstractmethod
    def on_text_changed(self, callback: Callable[[str], None]) -> None:
        """Register callback for text changes."""
        if not hasattr(self, '_text_callbacks'):
            self._text_callbacks = []
        self._text_callbacks.append(callback)
        
    def _notify_text_changed(self, text: str) -> None:
        """Notify all text change callbacks."""
        for callback in getattr(self, '_text_callbacks', []):
            try:
                callback(text)
            except Exception as e:
                logger.error(f"Text change callback error: {e}")


class TreeInterface(UIComponentInterface):
    """Interface for file/directory tree components."""
    
    @abstractmethod
    def set_root_path(self, path: Path) -> None:
        """Set root directory path."""
        self._root_path = path
        self._refresh_tree()
        logger.debug(f"Set root path to {path}")
    
    @abstractmethod
    def refresh(self) -> None:
        """Refresh tree contents."""
        self._refresh_tree()
        logger.debug("Refreshed tree contents")
        
    def _refresh_tree(self) -> None:
        """Internal tree refresh implementation."""
        if hasattr(self, '_root_path') and self._root_path:
            # Clear existing nodes
            if hasattr(self, '_nodes'):
                self._nodes.clear()
            # Rebuild tree from root path
            self._build_tree_nodes(self._root_path)
    
    @abstractmethod
    def expand_node(self, path: Path) -> None:
        """Expand tree node at path."""
        if not hasattr(self, '_expanded_nodes'):
            self._expanded_nodes = set()
        self._expanded_nodes.add(str(path))
        logger.debug(f"Expanded node at {path}")
    
    @abstractmethod
    def collapse_node(self, path: Path) -> None:
        """Collapse tree node at path."""
        if hasattr(self, '_expanded_nodes'):
            self._expanded_nodes.discard(str(path))
        logger.debug(f"Collapsed node at {path}")
    
    @abstractmethod
    def select_node(self, path: Path) -> None:
        """Select tree node at path."""
        self._selected_path = path
        self._notify_file_selected(path)
        logger.debug(f"Selected node at {path}")
    
    @abstractmethod
    def get_selected_path(self) -> Optional[Path]:
        """Get currently selected path."""
        return getattr(self, '_selected_path', None)
    
    @abstractmethod
    def on_file_selected(self, callback: Callable[[Path], None]) -> None:
        """Register callback for file selection."""
        if not hasattr(self, '_file_callbacks'):
            self._file_callbacks = []
        self._file_callbacks.append(callback)
        
    def _notify_file_selected(self, path: Path) -> None:
        """Notify all file selection callbacks."""
        for callback in getattr(self, '_file_callbacks', []):
            try:
                callback(path)
            except Exception as e:
                logger.error(f"File selection callback error: {e}")
                
    def _build_tree_nodes(self, root: Path) -> None:
        """Build tree structure from filesystem."""
        if not hasattr(self, '_nodes'):
            self._nodes = {}
        try:
            if root.is_dir():
                for item in sorted(root.iterdir()):
                    if not item.name.startswith('.'):
                        self._nodes[str(item)] = {
                            'path': item,
                            'is_dir': item.is_dir(),
                            'expanded': str(item) in getattr(self, '_expanded_nodes', set())
                        }
        except PermissionError:
            logger.warning(f"Permission denied accessing {root}")


class PaletteInterface(UIComponentInterface):
    """Interface for command palette components."""
    
    @abstractmethod
    def add_command(self, name: str, description: str, callback: Callable) -> None:
        """Add a command to the palette."""
        if not hasattr(self, '_commands'):
            self._commands = {}
        self._commands[name] = {
            'description': description,
            'callback': callback
        }
        logger.debug(f"Added command: {name}")
    
    @abstractmethod
    def remove_command(self, name: str) -> None:
        """Remove a command from the palette."""
        if hasattr(self, '_commands') and name in self._commands:
            del self._commands[name]
            logger.debug(f"Removed command: {name}")
    
    @abstractmethod
    def show(self) -> None:
        """Show the command palette."""
        self._visible = True
        if hasattr(self, '_widget'):
            self._widget.show()
        logger.debug("Showed command palette")
    
    @abstractmethod
    def hide(self) -> None:
        """Hide the command palette."""
        self._visible = False
        if hasattr(self, '_widget'):
            self._widget.hide()
        logger.debug("Hid command palette")
    
    @abstractmethod
    def filter_commands(self, query: str) -> List[Dict[str, Any]]:
        """Filter commands by query string."""
        if not hasattr(self, '_commands'):
            return []
        
        filtered = []
        query_lower = query.lower()
        
        for name, cmd_info in self._commands.items():
            if (query_lower in name.lower() or 
                query_lower in cmd_info['description'].lower()):
                filtered.append({
                    'name': name,
                    'description': cmd_info['description'],
                    'callback': cmd_info['callback']
                })
        
        return sorted(filtered, key=lambda x: x['name'])
    
    @abstractmethod
    def execute_command(self, name: str) -> None:
        """Execute a command by name."""
        if hasattr(self, '_commands') and name in self._commands:
            try:
                callback = self._commands[name]['callback']
                callback()
                logger.debug(f"Executed command: {name}")
            except Exception as e:
                logger.error(f"Command execution error for {name}: {e}")
        else:
            logger.warning(f"Command not found: {name}")


class StatusInterface(UIComponentInterface):
    """Interface for status bar components."""
    
    @abstractmethod
    def set_text(self, text: str) -> None:
        """Set main status text."""
        self._status_text = text
        self._update_display()
        logger.debug(f"Set status text: {text}")
    
    @abstractmethod
    def set_progress(self, progress: float, total: float = 100.0) -> None:
        """Set progress indicator (0-100)."""
        self._progress = min(max(progress / total * 100, 0), 100)
        self._update_display()
        logger.debug(f"Set progress: {self._progress:.1f}%")
    
    @abstractmethod
    def add_indicator(self, name: str, text: str, style: str = "default") -> None:
        """Add a status indicator."""
        if not hasattr(self, '_indicators'):
            self._indicators = {}
        self._indicators[name] = {'text': text, 'style': style}
        self._update_display()
        logger.debug(f"Added indicator {name}: {text}")
    
    @abstractmethod
    def remove_indicator(self, name: str) -> None:
        """Remove a status indicator."""
        if hasattr(self, '_indicators') and name in self._indicators:
            del self._indicators[name]
            self._update_display()
            logger.debug(f"Removed indicator: {name}")
    
    @abstractmethod
    def show_notification(self, message: str, duration: int = 3000) -> None:
        """Show temporary notification."""
        import asyncio
        self._notification = message
        self._update_display()
        logger.info(f"Notification: {message}")
        
        # Auto-hide notification after duration
        async def hide_notification():
            await asyncio.sleep(duration / 1000)
            if hasattr(self, '_notification'):
                self._notification = None
                self._update_display()
        
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(hide_notification())
        except RuntimeError:
            pass  # No event loop running
    
    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """Set current mode (e.g., 'INSERT', 'NORMAL')."""
        self._mode = mode
        self._update_display()
        logger.debug(f"Set mode to {mode}")
        
    def _update_display(self) -> None:
        """Update the status display with current state."""
        # Subclasses should implement actual display logic
        if hasattr(self, '_widget'):
            # Update widget display
            pass


class InputInterface(UIComponentInterface):
    """Interface for text input components."""
    
    @abstractmethod
    def set_placeholder(self, text: str) -> None:
        """Set placeholder text."""
        self._placeholder = text
        if hasattr(self, '_widget'):
            self._widget.placeholder = text
        logger.debug(f"Set placeholder: {text}")
    
    @abstractmethod
    def set_value(self, value: str) -> None:
        """Set input value."""
        self._value = value
        self._notify_value_changed(value)
        if hasattr(self, '_widget'):
            self._widget.value = value
        logger.debug(f"Set value: {value}")
    
    @abstractmethod
    def get_value(self) -> str:
        """Get current input value."""
        return getattr(self, '_value', '')
    
    @abstractmethod
    def clear(self) -> None:
        """Clear input value."""
        self.set_value('')
        logger.debug("Cleared input value")
    
    @abstractmethod
    def set_validator(self, validator: Callable[[str], bool]) -> None:
        """Set input validator function."""
        self._validator = validator
        # Re-validate current value
        if hasattr(self, '_value'):
            is_valid = validator(self._value)
            self._is_valid = is_valid
        logger.debug("Set input validator")
    
    @abstractmethod
    def on_value_changed(self, callback: Callable[[str], None]) -> None:
        """Register callback for value changes."""
        if not hasattr(self, '_value_callbacks'):
            self._value_callbacks = []
        self._value_callbacks.append(callback)
        
    def _notify_value_changed(self, value: str) -> None:
        """Notify all value change callbacks."""
        # Validate if validator exists
        if hasattr(self, '_validator'):
            self._is_valid = self._validator(value)
            
        for callback in getattr(self, '_value_callbacks', []):
            try:
                callback(value)
            except Exception as e:
                logger.error(f"Value change callback error: {e}")
    
    @abstractmethod
    def on_submit(self, callback: Callable[[str], None]) -> None:
        """Register callback for submit (Enter key)."""
        if not hasattr(self, '_submit_callbacks'):
            self._submit_callbacks = []
        self._submit_callbacks.append(callback)
        
    def _handle_submit(self) -> None:
        """Handle submit action (called by Enter key handler)."""
        value = self.get_value()
        
        # Only submit if value is valid
        if getattr(self, '_is_valid', True):
            for callback in getattr(self, '_submit_callbacks', []):
                try:
                    callback(value)
                except Exception as e:
                    logger.error(f"Submit callback error: {e}")
        else:
            logger.debug("Submit blocked - invalid input")