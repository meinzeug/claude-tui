#!/usr/bin/env python3
"""
Claude-TUI Application Runner
Entry point for the Textual-based Terminal User Interface
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the correct module path
try:
    from ui.main_app import ClaudeTUIApp, run_app
except ImportError:
    # Fallback to claude_tui structure
    from claude_tui.ui.application import ClaudeTUIApp
    from claude_tui.ui.main_app import run_app


def main():
    """Main entry point for Claude-TUI"""
    try:
        # Initialize and run the TUI application
        print("🚀 Starting Claude-TUI...")
        print("   Intelligent AI-powered Terminal User Interface")
        print("   with Progress Intelligence and Anti-Hallucination")
        print()
        
        # Run the application - handle both implementations
        if 'run_app' in globals():
            run_app()
        else:
            # Direct app instantiation
            from claude_tui.core.config_manager import ConfigManager
            config_manager = ConfigManager()
            app = ClaudeTUIApp(config_manager, debug=False)
            app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Claude-TUI shutdown by user")
    except Exception as e:
        print(f"❌ Error starting Claude-TUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()