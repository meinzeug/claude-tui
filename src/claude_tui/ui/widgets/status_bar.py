"""Status Bar Widget for Claude-TUI."""

try:
    from textual.widgets import Static
    from textual.containers import Horizontal
    from rich.text import Text
    
    class StatusBar(Horizontal):
        """Status bar showing application state."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._status_text = "Ready"
            
        def compose(self):
            yield Static(self._status_text, id="status-text")
            yield Static("", id="right-status")
            
        def update_status(self, text: str):
            """Update the status text."""
            self._status_text = text
            status_widget = self.query_one("#status-text")
            if status_widget:
                status_widget.update(text)
                
        def update_right_status(self, text: str):
            """Update the right status text."""
            right_status = self.query_one("#right-status")
            if right_status:
                right_status.update(text)
            
except ImportError:
    # Fallback implementation
    class StatusBar:
        """Fallback status bar widget."""
        
        def __init__(self, *args, **kwargs):
            self._status_text = "Ready"
            
        def update_status(self, text: str):
            self._status_text = text
            
        def update_right_status(self, text: str):
            pass