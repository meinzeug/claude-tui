#!/usr/bin/env python3
"""
Simple TUI Runner
================

A simplified wrapper around the robust launcher for common use cases.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from launch_tui import TUILauncher, LauncherConfig

def run_interactive():
    """Run the TUI in full interactive mode"""
    print("🚀 Starting Claude-TUI in interactive mode...")
    config = LauncherConfig(
        interactive=True,
        headless=False,
        debug=False,
        log_level="INFO"
    )
    launcher = TUILauncher(config)
    success, app = launcher.launch()
    
    if not success:
        print("❌ Failed to start Claude-TUI")
        sys.exit(1)
    
    return app

def run_headless():
    """Run the TUI in headless mode"""
    print("🔧 Starting Claude-TUI in headless mode...")
    config = LauncherConfig(
        interactive=False,
        headless=True,
        debug=False,
        log_level="WARNING"
    )
    launcher = TUILauncher(config)
    success, app = launcher.launch()
    launcher.cleanup(app)
    return app if success else None

def run_debug():
    """Run the TUI with debug logging"""
    print("🐛 Starting Claude-TUI in debug mode...")
    config = LauncherConfig(
        interactive=True,
        headless=False,
        debug=True,
        log_level="DEBUG"
    )
    launcher = TUILauncher(config)
    success, app = launcher.launch()
    
    if not success:
        print("❌ Failed to start Claude-TUI")
        sys.exit(1)
    
    return app

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ["headless", "h"]:
            run_headless()
        elif mode in ["debug", "d"]:
            run_debug()
        elif mode in ["interactive", "i"]:
            run_interactive()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: interactive, headless, debug")
            sys.exit(1)
    else:
        run_interactive()