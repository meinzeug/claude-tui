"""
Dependency Resolution and Import Safety System.

This module provides comprehensive dependency checking and safe import mechanisms
for Claude-TUI to handle missing or optional dependencies gracefully.
"""

import sys
import importlib
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class DependencyStatus:
    """Status of a dependency check."""
    name: str
    available: bool
    version: Optional[str] = None
    error_message: Optional[str] = None
    import_path: Optional[str] = None
    fallback_available: bool = False


class DependencyChecker:
    """Comprehensive dependency checker and fallback manager."""
    
    def __init__(self):
        self.checked_dependencies: Dict[str, DependencyStatus] = {}
        self.fallback_registry: Dict[str, Callable] = {}
        
    def register_fallback(self, module_name: str, fallback_factory: Callable):
        """Register a fallback implementation for a module."""
        self.fallback_registry[module_name] = fallback_factory
        
    def check_dependency(self, 
                        module_name: str, 
                        import_path: Optional[str] = None, 
                        version_attr: str = '__version__') -> DependencyStatus:
        """
        Check if a dependency is available and get its status.
        
        Args:
            module_name: Name of the module to check
            import_path: Optional specific import path
            version_attr: Attribute name for version (default: '__version__')
        
        Returns:
            DependencyStatus object with check results
        """
        if module_name in self.checked_dependencies:
            return self.checked_dependencies[module_name]
            
        status = DependencyStatus(
            name=module_name,
            available=False,
            import_path=import_path or module_name
        )
        
        try:
            # Try to import the module
            module = importlib.import_module(import_path or module_name)
            status.available = True
            
            # Try to get version
            if hasattr(module, version_attr):
                status.version = getattr(module, version_attr)
                
        except ImportError as e:
            status.error_message = str(e)
            status.fallback_available = module_name in self.fallback_registry
            logger.debug(f"Module {module_name} not available: {e}")
            
        except Exception as e:
            status.error_message = f"Unexpected error: {e}"
            logger.warning(f"Error checking {module_name}: {e}")
            
        self.checked_dependencies[module_name] = status
        return status
        
    def safe_import(self, module_name: str, 
                   import_path: Optional[str] = None,
                   fallback_factory: Optional[Callable] = None) -> Any:
        """
        Safely import a module with fallback support.
        
        Args:
            module_name: Name of the module
            import_path: Optional specific import path
            fallback_factory: Optional fallback factory function
            
        Returns:
            Module or fallback implementation
        """
        status = self.check_dependency(module_name, import_path)
        
        if status.available:
            return importlib.import_module(import_path or module_name)
            
        # Try registered fallback
        if module_name in self.fallback_registry:
            logger.info(f"Using registered fallback for {module_name}")
            return self.fallback_registry[module_name]()
            
        # Try provided fallback
        if fallback_factory:
            logger.info(f"Using provided fallback for {module_name}")
            return fallback_factory()
            
        # Return a minimal stub
        logger.warning(f"No fallback available for {module_name}, returning stub")
        return create_module_stub(module_name)
        
    def check_all_dependencies(self) -> Dict[str, DependencyStatus]:
        """Check all critical and optional dependencies."""
        
        # Critical dependencies
        critical_deps = [
            'textual',
            'rich', 
            'click',
            'pydantic',
            'aiohttp',
            'yaml',
            'watchdog',
            'git',
            'jinja2',
            'psutil'
        ]
        
        # Optional dependencies
        optional_deps = [
            'tkinter',
            'redis',
            'elasticsearch', 
            'uvloop',
            'orjson',
            'msgpack',
            'lz4',
            'numpy',
            'matplotlib',
            'plotly',
            'websockets'
        ]
        
        results = {}
        
        for dep in critical_deps + optional_deps:
            status = self.check_dependency(dep)
            results[dep] = status
            
        return results
        
    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing critical dependencies."""
        missing = []
        for name, status in self.checked_dependencies.items():
            if not status.available and not status.fallback_available:
                missing.append(name)
        return missing
        
    def install_missing_dependencies(self, dependencies: List[str]) -> bool:
        """
        Attempt to install missing dependencies.
        
        Args:
            dependencies: List of dependency names to install
            
        Returns:
            True if all installations succeeded
        """
        import subprocess
        
        success = True
        for dep in dependencies:
            try:
                logger.info(f"Installing {dep}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep
                ], capture_output=True, text=True, check=True)
                
                logger.info(f"Successfully installed {dep}")
                
                # Re-check the dependency
                if dep in self.checked_dependencies:
                    del self.checked_dependencies[dep]
                self.check_dependency(dep)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {dep}: {e}")
                success = False
            except Exception as e:
                logger.error(f"Error installing {dep}: {e}")
                success = False
                
        return success
        
    def generate_dependency_report(self) -> str:
        """Generate a detailed dependency status report."""
        if not self.checked_dependencies:
            self.check_all_dependencies()
            
        report = []
        report.append("=== Dependency Status Report ===\n")
        
        # Separate into categories
        available = []
        missing = []
        with_fallbacks = []
        
        for name, status in self.checked_dependencies.items():
            if status.available:
                available.append(status)
            elif status.fallback_available:
                with_fallbacks.append(status)
            else:
                missing.append(status)
                
        # Available dependencies
        if available:
            report.append("✅ Available Dependencies:")
            for status in available:
                version_info = f" (v{status.version})" if status.version else ""
                report.append(f"  • {status.name}{version_info}")
            report.append("")
            
        # Dependencies with fallbacks
        if with_fallbacks:
            report.append("⚠️  Dependencies with Fallbacks:")
            for status in with_fallbacks:
                report.append(f"  • {status.name} - {status.error_message}")
            report.append("")
            
        # Missing dependencies
        if missing:
            report.append("❌ Missing Dependencies:")
            for status in missing:
                report.append(f"  • {status.name} - {status.error_message}")
            report.append("")
            
        report.append(f"Total: {len(available)} available, "
                     f"{len(with_fallbacks)} with fallbacks, "
                     f"{len(missing)} missing")
                     
        return "\n".join(report)


def create_module_stub(module_name: str) -> Any:
    """Create a minimal stub for a missing module."""
    from types import ModuleType
    
    class ModuleStub(ModuleType):
        """A stub module that provides minimal functionality."""
        
        def __init__(self, name: str):
            super().__init__(name)
            self.__name__ = name
            
        def __getattr__(self, name: str):
            # Return a callable stub for any attribute
            def stub(*args, **kwargs):
                logger.debug(f"Stub call: {self.__name__}.{name}(*{args}, **{kwargs})")
                return None
            return stub
            
    return ModuleStub(module_name)


def setup_safe_imports():
    """Setup safe import mechanisms for Claude-TUI."""
    checker = DependencyChecker()
    
    # Register fallbacks for critical modules
    from .fallback_implementations import (
        FallbackConfigManager,
        FallbackSystemChecker, 
        FallbackClaudeTUIApp,
        setup_fallback_logging
    )
    
    checker.register_fallback('claude_tui.core.config_manager', 
                             lambda: type('ConfigManager', (), 
                                        {'__new__': lambda cls, *a, **k: FallbackConfigManager(*a, **k)}))
    checker.register_fallback('claude_tui.utils.system_check',
                             lambda: type('SystemChecker', (),
                                        {'__new__': lambda cls, *a, **k: FallbackSystemChecker(*a, **k)}))
    checker.register_fallback('claude_tui.ui.main_app',
                             lambda: type('ClaudeTUIApp', (),
                                        {'__new__': lambda cls, *a, **k: FallbackClaudeTUIApp(*a, **k)}))
    checker.register_fallback('claude_tui.core.logger',
                             lambda: type('setup_logging', (),
                                        {'__call__': setup_fallback_logging}))
    
    return checker


# Global dependency checker instance
_dependency_checker: Optional[DependencyChecker] = None


def get_dependency_checker() -> DependencyChecker:
    """Get the global dependency checker instance."""
    global _dependency_checker
    if _dependency_checker is None:
        _dependency_checker = setup_safe_imports()
    return _dependency_checker


def check_and_install_dependencies() -> bool:
    """Check dependencies and attempt to install missing ones."""
    checker = get_dependency_checker()
    checker.check_all_dependencies()
    
    missing = checker.get_missing_dependencies()
    if not missing:
        logger.info("All dependencies are available")
        return True
        
    logger.info(f"Missing dependencies: {missing}")
    
    # Attempt to install missing dependencies
    success = checker.install_missing_dependencies(missing)
    
    if success:
        logger.info("All missing dependencies installed successfully")
    else:
        logger.warning("Some dependencies could not be installed, using fallbacks")
        
    return success


if __name__ == "__main__":
    # Run dependency check when called directly
    checker = get_dependency_checker()
    print(checker.generate_dependency_report())