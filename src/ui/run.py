#!/usr/bin/env python3
"""
Claude-TIU Application Runner
Entry point for the Textual-based Terminal User Interface
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.main_app import ClaudeTIUApp, run_app


def main():
    """Main entry point for Claude-TIU"""
    try:
        # Initialize and run the TUI application
        print("🚀 Starting Claude-TIU...")
        print("   Intelligent AI-powered Terminal User Interface")
        print("   with Progress Intelligence and Anti-Hallucination")
        print()
        
        # Run the application
        run_app()
        
    except KeyboardInterrupt:
        print("\n👋 Claude-TIU shutdown by user")
    except Exception as e:
        print(f"❌ Error starting Claude-TIU: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()