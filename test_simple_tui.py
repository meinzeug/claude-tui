#!/usr/bin/env python3
"""Simple TUI test to verify basic functionality"""

from textual.app import App
from textual.widgets import Header, Footer, Static
from textual.containers import Container

class SimpleTUIApp(App):
    """Simple test app"""
    
    TITLE = "Claude-TUI Test"
    CSS = """
    Screen {
        background: #1a1a1a;
    }
    """
    
    def compose(self):
        yield Header()
        yield Container(
            Static("✅ Claude-TUI is working!\n\nPress Ctrl+C to exit.")
        )
        yield Footer()

if __name__ == "__main__":
    app = SimpleTUIApp()
    app.run()