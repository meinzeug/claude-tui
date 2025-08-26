#!/usr/bin/env python3
"""
Claude-TUI Robust Launcher
==========================

A comprehensive launcher for the Claude-TUI application that handles all initialization
errors gracefully and provides both interactive and non-interactive modes.

Features:
- Comprehensive error handling and recovery
- Multiple UI fallback implementations
- Interactive and non-interactive modes
- Health checks and dependency validation
- Detailed logging and error reporting
- Graceful shutdown handling
"""

from __future__ import annotations

import sys
import os
import logging
import asyncio
import threading
import time
import signal
import argparse
import traceback
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field

# Add src to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

@dataclass
class LauncherConfig:
    """Configuration for the TUI launcher"""
    interactive: bool = True
    debug: bool = False
    headless: bool = False
    test_mode: bool = False
    ui_type: str = "auto"  # auto, ui, claude_tui
    log_level: str = "INFO"
    log_file: Optional[str] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    fallback_mode: bool = True
    health_check: bool = True
    background_init: bool = False

@dataclass
class LauncherState:
    """State tracking for the launcher"""
    initialized: bool = False
    app_instance: Optional[Any] = None
    ui_type_used: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    initialization_time: Optional[float] = None
    bridge_available: bool = False
    components_status: Dict[str, bool] = field(default_factory=dict)

class TUILauncher:
    """
    Robust TUI launcher with comprehensive error handling and multiple operation modes.
    """
    
    def __init__(self, config: LauncherConfig):
        self.config = config
        self.state = LauncherState()
        self.logger = self._setup_logging()
        self.shutdown_event = threading.Event()
        self._setup_signal_handlers()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        logger = logging.getLogger("TUILauncher")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            try:
                file_handler = logging.FileHandler(self.config.log_file, mode='a')
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_format)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")
        
        return logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (OSError, ValueError):
            # On Windows or in some environments, these signals might not be available
            self.logger.debug("Some signal handlers not available on this platform")
    
    @contextmanager
    def error_recovery_context(self, operation: str):
        """Context manager for error recovery during operations"""
        try:
            self.logger.debug(f"Starting operation: {operation}")
            yield
            self.logger.debug(f"Operation completed successfully: {operation}")
        except Exception as e:
            error_msg = f"Error in {operation}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Traceback for {operation}: {traceback.format_exc()}")
            self.state.errors.append(error_msg)
            raise
    
    def check_dependencies(self) -> bool:
        """Check for required dependencies and system requirements"""
        self.logger.info("Checking dependencies and system requirements...")
        
        missing_deps = []
        optional_missing = []
        
        # Critical dependencies
        critical_deps = [
            ("textual", "textual"),
            ("rich", "rich"),
            ("asyncio", None),  # Built-in
        ]
        
        # Optional dependencies that enable additional features
        optional_deps = [
            ("aiohttp", "aiohttp"),
            ("pydantic", "pydantic"),
            ("watchdog", "watchdog"),
        ]
        
        def check_import(name, import_name=None):
            try:
                if import_name is None:
                    import_name = name
                __import__(import_name)
                return True
            except ImportError:
                return False
        
        # Check critical dependencies
        for dep_display, dep_import in critical_deps:
            if not check_import(dep_display, dep_import):
                missing_deps.append(dep_display)
                self.logger.error(f"Missing critical dependency: {dep_display}")
        
        # Check optional dependencies
        for dep_display, dep_import in optional_deps:
            if not check_import(dep_display, dep_import):
                optional_missing.append(dep_display)
                self.logger.warning(f"Missing optional dependency: {dep_display}")
        
        # Check Python version
        if sys.version_info < (3, 8):
            error_msg = f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}"
            self.logger.error(error_msg)
            self.state.errors.append(error_msg)
            return False
        
        if missing_deps:
            error_msg = f"Missing critical dependencies: {', '.join(missing_deps)}"
            self.logger.error(error_msg)
            self.logger.error("Install with: pip install -r requirements.txt")
            self.state.errors.append(error_msg)
            return False
        
        if optional_missing:
            warning_msg = f"Missing optional dependencies: {', '.join(optional_missing)}"
            self.logger.warning(warning_msg)
            self.state.warnings.append(warning_msg)
        
        self.logger.info("✓ Dependency check passed")
        return True
    
    def initialize_bridge(self) -> Tuple[bool, Optional[Any]]:
        """Initialize the integration bridge with comprehensive error handling"""
        self.logger.info("Initializing integration bridge...")
        
        bridge = None
        bridge_available = False
        
        # Try to import and initialize the bridge
        try:
            from ui.integration_bridge import get_bridge, reset_bridge
            
            # Reset bridge to ensure clean state
            reset_bridge()
            bridge = get_bridge()
            
            # Initialize bridge with retry logic
            for attempt in range(self.config.retry_attempts):
                try:
                    self.logger.info(f"Bridge initialization attempt {attempt + 1}/{self.config.retry_attempts}")
                    if bridge.initialize(force_reinit=True):
                        bridge_available = True
                        self.logger.info("✓ Integration bridge initialized successfully")
                        break
                    else:
                        self.logger.warning(f"Bridge initialization attempt {attempt + 1} failed")
                except Exception as e:
                    self.logger.warning(f"Bridge initialization attempt {attempt + 1} error: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(1)  # Brief delay before retry
            
            if bridge_available:
                # Store component status
                if hasattr(bridge, '_available_components'):
                    for component in ['config_manager', 'project_manager', 'ai_interface', 'validation_engine']:
                        self.state.components_status[component] = component in bridge._available_components
                
                # Log bridge health status
                self.logger.info(f"Bridge components available: {list(bridge._available_components)}")
                if hasattr(bridge, '_initialization_errors') and bridge._initialization_errors:
                    self.logger.warning(f"Bridge initialization warnings: {len(bridge._initialization_errors)}")
            
        except ImportError as e:
            self.logger.warning(f"Integration bridge not available: {e}")
            self.logger.info("Will attempt direct application initialization")
        except Exception as e:
            self.logger.error(f"Unexpected error initializing bridge: {e}")
            self.logger.debug(traceback.format_exc())
        
        self.state.bridge_available = bridge_available
        return bridge_available, bridge
    
    def get_app_instance(self, bridge: Optional[Any] = None) -> Tuple[Optional[Any], Optional[str]]:
        """Get application instance with comprehensive fallback strategy"""
        self.logger.info(f"Getting app instance (ui_type: {self.config.ui_type})")
        
        app_instance = None
        ui_type_used = None
        
        # Strategy 1: Use bridge if available
        if bridge and self.state.bridge_available:
            try:
                self.logger.info("Attempting to get app instance via bridge")
                app_instance, ui_type_used = bridge.get_app_instance(
                    self.config.ui_type, 
                    self.config.headless, 
                    self.config.test_mode
                )
                self.logger.info(f"✓ Got app instance via bridge (type: {ui_type_used})")
                return app_instance, ui_type_used
            except Exception as e:
                self.logger.warning(f"Bridge app instance failed: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Strategy 2: Direct import attempts
        import_strategies = [
            # src/ui/main_app.py
            ("ui.main_app", "ClaudeTUIApp", "ui"),
            # src/claude_tui/ui/application.py
            ("claude_tui.ui.application", "ClaudeTUIApp", "claude_tui"),
            # Fallback patterns
            ("main_app", "ClaudeTUIApp", "fallback"),
            ("application", "ClaudeTUIApp", "fallback"),
        ]
        
        for module_path, class_name, ui_type in import_strategies:
            if self.config.ui_type != "auto" and ui_type != self.config.ui_type and ui_type != "fallback":
                continue
                
            try:
                self.logger.debug(f"Trying direct import: {module_path}.{class_name}")
                module = __import__(module_path, fromlist=[class_name])
                app_class = getattr(module, class_name)
                
                # Create instance with appropriate parameters
                kwargs = {}
                init_signature = app_class.__init__.__code__.co_varnames
                
                if 'headless' in init_signature:
                    kwargs['headless'] = self.config.headless
                if 'test_mode' in init_signature:
                    kwargs['test_mode'] = self.config.test_mode
                if 'debug' in init_signature:
                    kwargs['debug'] = self.config.debug
                
                app_instance = app_class(**kwargs)
                ui_type_used = ui_type
                
                self.logger.info(f"✓ Direct import successful: {module_path} (type: {ui_type})")
                return app_instance, ui_type_used
                
            except (ImportError, AttributeError, TypeError) as e:
                self.logger.debug(f"Direct import failed for {module_path}: {e}")
                continue
        
        # Strategy 3: Create minimal fallback app
        if self.config.fallback_mode:
            self.logger.warning("Creating minimal fallback application")
            try:
                app_instance = self._create_fallback_app()
                ui_type_used = "fallback"
                self.logger.info("✓ Fallback application created")
                return app_instance, ui_type_used
            except Exception as e:
                self.logger.error(f"Fallback application creation failed: {e}")
        
        return None, None
    
    def _create_fallback_app(self):
        """Create a minimal fallback application for emergency situations"""
        self.logger.info("Creating minimal fallback application")
        
        logger = self.logger  # Reference for inner class
        
        class MinimalFallbackApp:
            def __init__(self):
                self.initialized = False
                self.running = False
                
            def init_core_systems(self):
                logger.info("Fallback app: Core systems initialized (minimal)")
                self.initialized = True
                
            async def init_async(self):
                self.init_core_systems()
                self.running = True
                
            def run(self):
                logger.info("Fallback app: Running in console mode")
                self.init_core_systems()
                self.running = True
                
                print("Claude-TUI Fallback Mode")
                print("=" * 40)
                print("The application is running in minimal fallback mode.")
                print("Some features may not be available.")
                print("Press Ctrl+C to exit.")
                
                try:
                    while self.running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    self.running = False
                    
            def stop(self):
                self.running = False
                
            def cleanup(self):
                logger.debug("Fallback app cleanup")
                self.running = False
                
            def is_running(self):
                return self.running
        
        return MinimalFallbackApp()
    
    def initialize_application(self, app_instance: Any) -> bool:
        """Initialize the application with comprehensive error handling"""
        self.logger.info("Initializing application...")
        
        try:
            # Initialize core systems
            if hasattr(app_instance, 'init_core_systems'):
                self.logger.debug("Calling init_core_systems")
                app_instance.init_core_systems()
                
            # Mark as initialized in state
            if hasattr(app_instance, '_initialized'):
                app_instance._initialized = True
            if hasattr(app_instance, '_running'):
                app_instance._running = True
                
            self.logger.info("✓ Application initialization completed")
            return True
            
        except Exception as e:
            error_msg = f"Application initialization failed: {e}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            self.state.errors.append(error_msg)
            return False
    
    def run_application(self, app_instance: Any) -> bool:
        """Run the application in the appropriate mode"""
        self.logger.info(f"Running application (interactive: {self.config.interactive}, headless: {self.config.headless})")
        
        try:
            if self.config.headless or self.config.test_mode:
                return self._run_non_interactive(app_instance)
            else:
                return self._run_interactive(app_instance)
                
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            return True
        except Exception as e:
            error_msg = f"Application runtime error: {e}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            self.state.errors.append(error_msg)
            return False
    
    def _run_interactive(self, app_instance: Any) -> bool:
        """Run application in interactive mode"""
        self.logger.info("Starting interactive mode")
        
        try:
            if hasattr(app_instance, 'run'):
                if self.config.background_init:
                    # Run in background thread for non-blocking operation
                    def run_thread():
                        try:
                            app_instance.run()
                        except Exception as e:
                            self.logger.error(f"Background app error: {e}")
                    
                    thread = threading.Thread(target=run_thread, daemon=True)
                    thread.start()
                    self.logger.info("Application started in background thread")
                    return True
                else:
                    # Normal blocking run
                    app_instance.run()
                    return True
            else:
                self.logger.error("Application instance has no 'run' method")
                return False
                
        except Exception as e:
            self.logger.error(f"Interactive mode error: {e}")
            return False
    
    def _run_non_interactive(self, app_instance: Any) -> bool:
        """Run application in non-interactive/headless mode"""
        self.logger.info("Starting non-interactive mode")
        
        try:
            # Try async initialization first
            if hasattr(app_instance, 'init_async'):
                self.logger.debug("Using async initialization")
                asyncio.run(app_instance.init_async())
            elif hasattr(app_instance, 'run_async'):
                self.logger.debug("Using async run")
                asyncio.run(app_instance.run_async())
            elif hasattr(app_instance, 'init_core_systems'):
                self.logger.debug("Using sync initialization")
                app_instance.init_core_systems()
            else:
                self.logger.warning("No initialization method found, app may not be fully functional")
            
            self.logger.info("✓ Non-interactive mode initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Non-interactive mode error: {e}")
            return False
    
    def perform_health_check(self) -> bool:
        """Perform comprehensive health check"""
        if not self.config.health_check:
            return True
            
        self.logger.info("Performing system health check...")
        
        health_issues = []
        
        # Check system resources
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_issues.append(f"High memory usage: {memory.percent}%")
        except ImportError:
            self.logger.debug("psutil not available for memory check")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(project_root)
            free_percent = (free / total) * 100
            if free_percent < 10:
                health_issues.append(f"Low disk space: {free_percent:.1f}% free")
        except Exception:
            self.logger.debug("Could not check disk space")
        
        # Check Python environment
        if len(sys.path) > 50:
            health_issues.append("Python path is very long, may indicate environment issues")
        
        # Log health status
        if health_issues:
            self.logger.warning("Health check found issues:")
            for issue in health_issues:
                self.logger.warning(f"  - {issue}")
                self.state.warnings.append(issue)
        else:
            self.logger.info("✓ Health check passed")
        
        return len(health_issues) == 0
    
    def cleanup(self, app_instance: Optional[Any] = None):
        """Perform cleanup operations"""
        self.logger.info("Performing cleanup...")
        
        try:
            if app_instance:
                if hasattr(app_instance, 'cleanup'):
                    if asyncio.iscoroutinefunction(app_instance.cleanup):
                        asyncio.run(app_instance.cleanup())
                    else:
                        app_instance.cleanup()
                elif hasattr(app_instance, 'stop'):
                    app_instance.stop()
                    
            self.logger.info("✓ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def print_launch_summary(self):
        """Print a summary of the launch process"""
        print("\n" + "=" * 60)
        print("Claude-TUI Launch Summary")
        print("=" * 60)
        
        # Timing information
        if self.state.initialization_time:
            print(f"Initialization time: {self.state.initialization_time:.2f}s")
        
        total_time = time.time() - self.state.start_time
        print(f"Total launch time: {total_time:.2f}s")
        
        # Status information
        print(f"Status: {'SUCCESS' if self.state.initialized else 'FAILED'}")
        if self.state.ui_type_used:
            print(f"UI Implementation: {self.state.ui_type_used}")
        print(f"Bridge available: {'YES' if self.state.bridge_available else 'NO'}")
        
        # Component status
        if self.state.components_status:
            print(f"Components status:")
            for component, status in self.state.components_status.items():
                print(f"  {component}: {'OK' if status else 'FAILED'}")
        
        # Errors and warnings
        if self.state.errors:
            print(f"Errors ({len(self.state.errors)}):")
            for error in self.state.errors:
                print(f"  - {error}")
        
        if self.state.warnings:
            print(f"Warnings ({len(self.state.warnings)}):")
            for warning in self.state.warnings:
                print(f"  - {warning}")
        
        print("=" * 60)
    
    def launch(self) -> Tuple[bool, Optional[Any]]:
        """
        Main launch method that orchestrates the entire startup process.
        
        Returns:
            Tuple of (success: bool, app_instance: Optional[Any])
        """
        self.logger.info("Starting Claude-TUI launch sequence...")
        
        try:
            # Step 1: Check dependencies
            with self.error_recovery_context("dependency check"):
                if not self.check_dependencies():
                    return False, None
            
            # Step 2: Health check
            with self.error_recovery_context("health check"):
                self.perform_health_check()
            
            # Step 3: Initialize bridge
            bridge = None
            with self.error_recovery_context("bridge initialization"):
                bridge_available, bridge = self.initialize_bridge()
            
            # Step 4: Get application instance
            app_instance = None
            with self.error_recovery_context("app instance creation"):
                app_instance, ui_type_used = self.get_app_instance(bridge)
                if not app_instance:
                    self.state.errors.append("Failed to create application instance")
                    return False, None
                self.state.app_instance = app_instance
                self.state.ui_type_used = ui_type_used
            
            # Step 5: Initialize application
            with self.error_recovery_context("application initialization"):
                init_start_time = time.time()
                if not self.initialize_application(app_instance):
                    return False, app_instance
                self.state.initialization_time = time.time() - init_start_time
                self.state.initialized = True
            
            # Step 6: Run application
            if self.config.interactive and not self.config.headless:
                with self.error_recovery_context("application execution"):
                    success = self.run_application(app_instance)
                    if not success:
                        return False, app_instance
            
            self.logger.info("✓ Claude-TUI launch sequence completed successfully")
            return True, app_instance
            
        except Exception as e:
            error_msg = f"Launch sequence failed: {e}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            self.state.errors.append(error_msg)
            return False, self.state.app_instance
        
        finally:
            if not self.config.headless and not self.config.test_mode:
                self.print_launch_summary()


def create_launcher_config_from_args(args) -> LauncherConfig:
    """Create launcher configuration from command line arguments"""
    return LauncherConfig(
        interactive=args.interactive,
        debug=args.debug,
        headless=args.headless,
        test_mode=args.test_mode,
        ui_type=args.ui_type,
        log_level=args.log_level,
        log_file=args.log_file,
        timeout=args.timeout,
        retry_attempts=args.retry_attempts,
        fallback_mode=args.fallback_mode,
        health_check=args.health_check,
        background_init=args.background_init,
    )

def main():
    """Main entry point for the launcher"""
    parser = argparse.ArgumentParser(
        description="Claude-TUI Robust Launcher - Launch the Claude TUI application with comprehensive error handling"
    )
    
    # Mode options
    parser.add_argument(
        '--non-interactive', 
        action='store_false', 
        dest='interactive',
        help='Run in non-interactive mode'
    )
    parser.add_argument(
        '--headless', 
        action='store_true',
        help='Run in headless mode (no UI)'
    )
    parser.add_argument(
        '--test-mode', 
        action='store_true',
        help='Run in test mode (non-blocking initialization)'
    )
    parser.add_argument(
        '--background-init',
        action='store_true',
        help='Initialize application in background thread'
    )
    
    # UI options
    parser.add_argument(
        '--ui-type',
        choices=['auto', 'ui', 'claude_tui'],
        default='auto',
        help='Specify UI implementation to use'
    )
    
    # Debugging options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    parser.add_argument(
        '--log-file',
        help='Path to log file (optional)'
    )
    
    # Recovery options
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Timeout for initialization operations (seconds)'
    )
    parser.add_argument(
        '--retry-attempts',
        type=int,
        default=3,
        help='Number of retry attempts for failed operations'
    )
    parser.add_argument(
        '--no-fallback',
        action='store_false',
        dest='fallback_mode',
        help='Disable fallback mode'
    )
    parser.add_argument(
        '--no-health-check',
        action='store_false',
        dest='health_check',
        help='Skip health check'
    )
    
    args = parser.parse_args()
    
    # Create launcher with configuration
    config = create_launcher_config_from_args(args)
    launcher = TUILauncher(config)
    
    try:
        success, app_instance = launcher.launch()
        
        if success:
            launcher.logger.info("🎉 Claude-TUI launched successfully!")
            
            if config.test_mode or config.headless:
                # For non-interactive modes, return the app instance
                return app_instance
            else:
                # For interactive mode, wait for shutdown signal
                try:
                    while not launcher.shutdown_event.is_set():
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    launcher.logger.info("Shutdown requested by user")
                
        else:
            launcher.logger.error("❌ Claude-TUI launch failed!")
            launcher.logger.info("Check the logs above for details")
            sys.exit(1)
            
    except Exception as e:
        launcher.logger.error(f"Unexpected error during launch: {e}")
        launcher.logger.debug(traceback.format_exc())
        sys.exit(1)
        
    finally:
        launcher.cleanup(launcher.state.app_instance)

# Entry points for different use cases
def launch_interactive():
    """Launch in interactive mode"""
    config = LauncherConfig(interactive=True, headless=False)
    launcher = TUILauncher(config)
    success, app = launcher.launch()
    launcher.cleanup(app)
    return app if success else None

def launch_headless():
    """Launch in headless mode"""
    config = LauncherConfig(interactive=False, headless=True)
    launcher = TUILauncher(config)
    success, app = launcher.launch()
    return app if success else None

def launch_test_mode():
    """Launch in test mode"""
    config = LauncherConfig(interactive=False, test_mode=True, headless=True)
    launcher = TUILauncher(config)
    success, app = launcher.launch()
    return app if success else None

def launch_with_config(config: LauncherConfig):
    """Launch with custom configuration"""
    launcher = TUILauncher(config)
    success, app = launcher.launch()
    launcher.cleanup(app)
    return app if success else None

if __name__ == "__main__":
    main()