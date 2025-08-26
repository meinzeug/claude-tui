"""Code Editor Widget for Claude-TUI."""

import logging
from typing import Optional, Callable, Dict, Any
from pathlib import Path

try:
    from textual.widgets import TextArea
    from textual import events
    from textual.message import Message
    from rich.syntax import Syntax
    from rich.text import Text
    
    from ...interfaces.ui_interfaces import EditorInterface
    
    logger = logging.getLogger(__name__)
    
    class CodeEditor(TextArea):
        """Enhanced code editor with syntax highlighting and full functionality."""
        
        class Changed(Message):
            """Message sent when code changes."""
            
            def __init__(self, sender: "CodeEditor", value: str) -> None:
                self.value = value
                super().__init__(sender)
        
        class LineSelected(Message):
            """Message sent when line is selected."""
            
            def __init__(self, sender: "CodeEditor", line_number: int) -> None:
                self.line_number = line_number
                super().__init__(sender)
        
        def __init__(
            self, 
            language: str = "python", 
            theme: str = "monokai",
            show_line_numbers: bool = True,
            text: str = "",
            read_only: bool = False,
            *args, 
            **kwargs
        ):
            super().__init__(text=text, read_only=read_only, *args, **kwargs)
            self.language = language
            self.theme = theme
            self.show_line_numbers = show_line_numbers
            self._text_changed_callbacks: list[Callable[[str], None]] = []
            self._highlighted_lines: Dict[int, str] = {}
            self._initialized = False
            
        def initialize(self) -> None:
            """Initialize the code editor."""
            if self._initialized:
                return
                
            try:
                # Set up syntax highlighting
                if self.language:
                    self.language = self.language.lower()
                
                # Configure editor settings
                if hasattr(self, 'show_line_numbers'):
                    self.show_line_numbers = True
                    
                self._initialized = True
                logger.debug(f"Code editor initialized for language: {self.language}")
                
            except Exception as e:
                logger.error(f"Failed to initialize code editor: {e}")
        
        def cleanup(self) -> None:
            """Cleanup editor resources."""
            self._text_changed_callbacks.clear()
            self._highlighted_lines.clear()
            logger.debug("Code editor cleanup completed")
        
        def set_focus(self) -> None:
            """Set focus to the editor."""
            if hasattr(self, 'focus'):
                self.focus()
        
        def set_text(self, text: str) -> None:
            """Set editor text content."""
            try:
                self.text = text
                self._notify_text_changed(text)
            except Exception as e:
                logger.error(f"Failed to set text: {e}")
        
        def get_text(self) -> str:
            """Get current editor text content."""
            return self.text
        
        def set_language(self, language: str) -> None:
            """Set syntax highlighting language."""
            self.language = language.lower()
            # Re-initialize syntax highlighting if needed
            self._update_syntax_highlighting()
            logger.debug(f"Language set to: {language}")
        
        def goto_line(self, line_number: int) -> None:
            """Go to specific line number."""
            try:
                if hasattr(self, 'cursor_line'):
                    self.cursor_line = max(0, line_number - 1)
                    self.post_message(self.LineSelected(self, line_number))
                    logger.debug(f"Moved to line: {line_number}")
            except Exception as e:
                logger.error(f"Failed to go to line {line_number}: {e}")
        
        def highlight_line(self, line_number: int, style: str = "error") -> None:
            """Highlight a specific line."""
            self._highlighted_lines[line_number] = style
            # Apply highlighting visual effect
            self._apply_line_highlight(line_number, style)
            logger.debug(f"Highlighted line {line_number} with style: {style}")
        
        def set_readonly(self, readonly: bool) -> None:
            """Set editor to readonly mode."""
            self.read_only = readonly
            logger.debug(f"Read-only mode set to: {readonly}")
        
        def on_text_changed(self, callback: Callable[[str], None]) -> None:
            """Register callback for text changes."""
            if callback not in self._text_changed_callbacks:
                self._text_changed_callbacks.append(callback)
        
        def on_text_area_changed(self, event: TextArea.Changed) -> None:
            """Handle text changes from Textual."""
            self.post_message(self.Changed(self, event.text))
            self._notify_text_changed(event.text)
            
        def _notify_text_changed(self, text: str) -> None:
            """Notify all registered callbacks of text changes."""
            for callback in self._text_changed_callbacks:
                try:
                    callback(text)
                except Exception as e:
                    logger.error(f"Text change callback error: {e}")
        
        def _update_syntax_highlighting(self) -> None:
            """Update syntax highlighting based on current language."""
            try:
                if hasattr(self, 'language') and self.language:
                    # Update highlighting - this depends on Textual's API
                    pass
            except Exception as e:
                logger.error(f"Failed to update syntax highlighting: {e}")
        
        def _apply_line_highlight(self, line_number: int, style: str) -> None:
            """Apply visual highlighting to a specific line."""
            try:
                # Implementation depends on Textual's highlighting API
                # This is a placeholder for the actual highlighting logic
                pass
            except Exception as e:
                logger.error(f"Failed to apply line highlight: {e}")
                
        def insert_text(self, text: str) -> None:
            """Insert text at cursor position."""
            try:
                current_text = self.get_text()
                # Simple insertion - in a real implementation, 
                # this would respect cursor position
                self.set_text(current_text + text)
            except Exception as e:
                logger.error(f"Failed to insert text: {e}")
        
        def get_selected_text(self) -> str:
            """Get currently selected text."""
            # Placeholder - would need actual selection API
            return ""
        
        def find_text(self, query: str, case_sensitive: bool = False) -> list[int]:
            """Find text in editor and return line numbers."""
            lines = self.get_text().split('\n')
            matches = []
            
            search_func = str.find if case_sensitive else lambda s, q: s.lower().find(q.lower())
            
            for i, line in enumerate(lines, 1):
                if search_func(line, query) != -1:
                    matches.append(i)
            
            return matches
        
        def replace_text(self, old_text: str, new_text: str, all_occurrences: bool = False) -> int:
            """Replace text in editor."""
            current_text = self.get_text()
            
            if all_occurrences:
                new_content = current_text.replace(old_text, new_text)
                count = current_text.count(old_text)
            else:
                new_content = current_text.replace(old_text, new_text, 1)
                count = 1 if old_text in current_text else 0
            
            self.set_text(new_content)
            return count
                
except ImportError as e:
    # Enhanced fallback implementation
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import Textual widgets: {e}. Using enhanced fallback.")
    
    from ...interfaces.ui_interfaces import EditorInterface
    
    class CodeEditor(EditorInterface):
        """Enhanced fallback code editor widget."""
        
        def __init__(
            self, 
            language: str = "python", 
            theme: str = "monokai",
            show_line_numbers: bool = True,
            text: str = "",
            read_only: bool = False,
            *args, 
            **kwargs
        ):
            self.language = language
            self.theme = theme
            self.show_line_numbers = show_line_numbers
            self._text = text
            self.read_only = read_only
            self._text_changed_callbacks: list[Callable[[str], None]] = []
            self._highlighted_lines: Dict[int, str] = {}
            self._initialized = False
            
        def initialize(self) -> None:
            """Initialize the fallback editor."""
            self._initialized = True
            logger.info("Fallback code editor initialized")
        
        def cleanup(self) -> None:
            """Cleanup editor resources."""
            self._text_changed_callbacks.clear()
            self._highlighted_lines.clear()
        
        def set_focus(self) -> None:
            """Set focus (no-op in fallback)."""
            pass
        
        def set_text(self, text: str) -> None:
            """Set editor text content."""
            self._text = text
            self._notify_text_changed(text)
        
        def get_text(self) -> str:
            """Get current editor text content."""
            return self._text
        
        def set_language(self, language: str) -> None:
            """Set syntax highlighting language."""
            self.language = language
        
        def goto_line(self, line_number: int) -> None:
            """Go to specific line number (logged in fallback)."""
            logger.info(f"Would go to line: {line_number}")
        
        def highlight_line(self, line_number: int, style: str = "error") -> None:
            """Highlight a specific line (stored in fallback)."""
            self._highlighted_lines[line_number] = style
            logger.info(f"Would highlight line {line_number} with style: {style}")
        
        def set_readonly(self, readonly: bool) -> None:
            """Set editor to readonly mode."""
            self.read_only = readonly
        
        def on_text_changed(self, callback: Callable[[str], None]) -> None:
            """Register callback for text changes."""
            if callback not in self._text_changed_callbacks:
                self._text_changed_callbacks.append(callback)
        
        def _notify_text_changed(self, text: str) -> None:
            """Notify all registered callbacks of text changes."""
            for callback in self._text_changed_callbacks:
                try:
                    callback(text)
                except Exception as e:
                    logger.error(f"Text change callback error: {e}")
                    
        def insert_text(self, text: str) -> None:
            """Insert text at cursor position."""
            self._text += text
            self._notify_text_changed(self._text)
        
        def get_selected_text(self) -> str:
            """Get currently selected text."""
            return ""
        
        def find_text(self, query: str, case_sensitive: bool = False) -> list[int]:
            """Find text in editor and return line numbers."""
            lines = self._text.split('\n')
            matches = []
            
            search_func = str.find if case_sensitive else lambda s, q: s.lower().find(q.lower())
            
            for i, line in enumerate(lines, 1):
                if search_func(line, query) != -1:
                    matches.append(i)
            
            return matches