"""Enhanced Text Input Widget for Claude-TUI."""

try:
    from textual.widgets import Input
    from textual import events
    from textual.message import Message
    
    class TextInput(Input):
        """Enhanced text input with additional features."""
        
        class Changed(Message):
            """Message sent when text changes."""
            
            def __init__(self, sender: "TextInput", value: str) -> None:
                self.value = value
                super().__init__(sender)
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def on_input_changed(self, event: Input.Changed) -> None:
            """Handle input changes."""
            self.post_message(self.Changed(self, event.value))
            
except ImportError:
    # Fallback implementation
    class TextInput:
        """Fallback text input widget."""
        
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get('value', '')
            
        def on_input_changed(self, event):
            pass