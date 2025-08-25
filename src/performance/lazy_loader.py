#!/usr/bin/env python3
"""
Emergency Lazy Loading System - Critical Memory Optimization
Implements delayed module loading to reduce initial memory footprint
"""

import sys
import importlib
import threading
import weakref
from typing import Any, Dict, Optional, Callable, Type, Union
from functools import wraps
from dataclasses import dataclass
import time


@dataclass
class LazyModuleInfo:
    """Information about a lazy-loaded module"""
    module_name: str
    import_path: str
    loaded: bool = False
    load_time: Optional[float] = None
    memory_impact: Optional[int] = None
    usage_count: int = 0


class LazyModuleLoader:
    """
    High-performance lazy module loader for emergency memory optimization
    Delays heavy module imports until actually needed
    """
    
    def __init__(self):
        self._modules: Dict[str, LazyModuleInfo] = {}
        self._loaded_modules: Dict[str, Any] = {}
        self._loading_lock = threading.RLock()
        self._access_stats = {}
        
    def register_lazy_module(self, 
                           alias: str, 
                           import_path: str, 
                           min_usage_threshold: int = 1) -> 'LazyModule':
        """Register a module for lazy loading"""
        
        info = LazyModuleInfo(
            module_name=alias,
            import_path=import_path
        )
        self._modules[alias] = info
        
        return LazyModule(self, alias, min_usage_threshold)
    
    def load_module(self, alias: str) -> Any:
        """Load a module on-demand"""
        if alias in self._loaded_modules:
            self._modules[alias].usage_count += 1
            return self._loaded_modules[alias]
            
        with self._loading_lock:
            # Double-check after acquiring lock
            if alias in self._loaded_modules:
                return self._loaded_modules[alias]
                
            info = self._modules.get(alias)
            if not info:
                raise ValueError(f"Module {alias} not registered for lazy loading")
            
            print(f"🔄 Lazy loading {info.import_path}...")
            start_time = time.time()
            
            try:
                # Import the module
                module = importlib.import_module(info.import_path)
                
                # Track loading performance
                load_time = time.time() - start_time
                info.loaded = True
                info.load_time = load_time
                info.usage_count += 1
                
                self._loaded_modules[alias] = module
                
                print(f"✅ Loaded {info.import_path} in {load_time:.3f}s")
                
                return module
                
            except Exception as e:
                print(f"❌ Failed to load {info.import_path}: {e}")
                raise
    
    def unload_module(self, alias: str):
        """Unload a module to free memory"""
        with self._loading_lock:
            if alias in self._loaded_modules:
                del self._loaded_modules[alias]
                
            if alias in self._modules:
                self._modules[alias].loaded = False
                self._modules[alias].load_time = None
                
            # Remove from sys.modules cache
            info = self._modules.get(alias)
            if info and info.import_path in sys.modules:
                del sys.modules[info.import_path]
                
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get statistics about lazy loading performance"""
        stats = {
            "registered_modules": len(self._modules),
            "loaded_modules": len(self._loaded_modules),
            "total_usage": sum(info.usage_count for info in self._modules.values()),
            "modules": {}
        }
        
        for alias, info in self._modules.items():
            stats["modules"][alias] = {
                "loaded": info.loaded,
                "usage_count": info.usage_count,
                "load_time": info.load_time,
                "import_path": info.import_path
            }
            
        return stats
    
    def optimize_memory(self):
        """Optimize memory by unloading unused modules"""
        with self._loading_lock:
            # Find modules that haven't been used recently
            unused_modules = [
                alias for alias, info in self._modules.items()
                if info.loaded and info.usage_count < 5
            ]
            
            for alias in unused_modules:
                print(f"🧹 Unloading unused module: {alias}")
                self.unload_module(alias)


class LazyModule:
    """Proxy object for lazy-loaded modules"""
    
    def __init__(self, loader: LazyModuleLoader, alias: str, min_usage: int = 1):
        self._loader = loader
        self._alias = alias
        self._min_usage = min_usage
        self._access_count = 0
        
    def __getattr__(self, name: str) -> Any:
        self._access_count += 1
        
        # Only load if usage threshold is met
        if self._access_count >= self._min_usage:
            module = self._loader.load_module(self._alias)
            return getattr(module, name)
        else:
            # Return a placeholder that will load on actual use
            return LazyAttribute(self._loader, self._alias, name)
            
    def __call__(self, *args, **kwargs):
        module = self._loader.load_module(self._alias)
        return module(*args, **kwargs)
        
    def __repr__(self):
        return f"<LazyModule: {self._alias} (accessed {self._access_count} times)>"


class LazyAttribute:
    """Lazy attribute that loads the module when accessed"""
    
    def __init__(self, loader: LazyModuleLoader, alias: str, attr_name: str):
        self._loader = loader
        self._alias = alias
        self._attr_name = attr_name
        
    def __call__(self, *args, **kwargs):
        module = self._loader.load_module(self._alias)
        attr = getattr(module, self._attr_name)
        return attr(*args, **kwargs)
        
    def __getattr__(self, name: str):
        module = self._loader.load_module(self._alias)
        attr = getattr(module, self._attr_name)
        return getattr(attr, name)


# Global lazy loader instance
_global_loader = LazyModuleLoader()


def lazy_import(import_path: str, alias: Optional[str] = None, min_usage: int = 1):
    """Decorator/function for lazy importing"""
    if alias is None:
        alias = import_path.split('.')[-1]
    
    return _global_loader.register_lazy_module(alias, import_path, min_usage)


def lazy_class_import(import_path: str, class_name: str):
    """Lazy import for specific classes"""
    full_path = f"{import_path}.{class_name}"
    
    class LazyClass:
        def __new__(cls, *args, **kwargs):
            module = importlib.import_module(import_path)
            actual_class = getattr(module, class_name)
            return actual_class(*args, **kwargs)
            
        def __getattr__(cls, name):
            module = importlib.import_module(import_path)
            actual_class = getattr(module, class_name)
            return getattr(actual_class, name)
    
    return LazyClass


# Emergency memory optimization - lazy imports for heavy modules
def setup_emergency_lazy_imports():
    """Setup lazy imports for memory-heavy modules"""
    
    # Heavy ML/Data Science modules
    numpy = lazy_import('numpy', 'np', min_usage=2)
    pandas = lazy_import('pandas', 'pd', min_usage=2)
    
    # Optional heavy modules
    torch = lazy_import('torch', min_usage=3)
    tensorflow = lazy_import('tensorflow', 'tf', min_usage=3)
    
    # Heavy validation modules
    validation_engine = lazy_import('claude_tui.validation.anti_hallucination_engine', min_usage=2)
    
    # Heavy AI modules  
    neural_trainer = lazy_import('ai.neural_trainer', min_usage=2)
    swarm_manager = lazy_import('ai.swarm_manager', min_usage=2)
    
    return {
        'numpy': numpy,
        'pandas': pandas,
        'torch': torch,
        'tensorflow': tensorflow,
        'validation_engine': validation_engine,
        'neural_trainer': neural_trainer,
        'swarm_manager': swarm_manager
    }


class LazyClassDecorator:
    """Decorator to make class loading lazy"""
    
    def __init__(self, min_usage: int = 1):
        self.min_usage = min_usage
        self.access_count = 0
        self.original_class = None
        
    def __call__(self, cls):
        self.original_class = cls
        
        class LazyWrapper:
            def __new__(wrapper_cls, *args, **kwargs):
                nonlocal self
                self.access_count += 1
                
                if self.access_count >= self.min_usage:
                    return self.original_class(*args, **kwargs)
                else:
                    # Return a lightweight proxy
                    return LazyInstanceProxy(self.original_class, args, kwargs)
                    
        return LazyWrapper


class LazyInstanceProxy:
    """Proxy for lazy class instances"""
    
    def __init__(self, cls, args, kwargs):
        self._cls = cls
        self._args = args
        self._kwargs = kwargs
        self._instance = None
        
    def _get_instance(self):
        if self._instance is None:
            self._instance = self._cls(*self._args, **self._kwargs)
        return self._instance
        
    def __getattr__(self, name):
        return getattr(self._get_instance(), name)
        
    def __call__(self, *args, **kwargs):
        return self._get_instance()(*args, **kwargs)


def memory_efficient_import(module_path: str):
    """Memory-efficient import that only loads when needed"""
    
    class MemoryEfficientModule:
        def __init__(self, path):
            self._path = path
            self._module = None
            
        def __getattr__(self, name):
            if self._module is None:
                print(f"📦 Loading {self._path} on first access")
                self._module = importlib.import_module(self._path)
            return getattr(self._module, name)
            
    return MemoryEfficientModule(module_path)


# Performance monitoring for lazy loading
class LazyLoadingProfiler:
    """Profile lazy loading performance"""
    
    def __init__(self):
        self.load_times = {}
        self.memory_savings = 0
        
    def record_load(self, module_name: str, load_time: float, memory_delta: int):
        """Record a lazy load event"""
        self.load_times[module_name] = load_time
        if memory_delta > 0:
            self.memory_savings += memory_delta
            
    def get_report(self) -> str:
        """Get lazy loading performance report"""
        total_modules = len(self.load_times)
        avg_load_time = sum(self.load_times.values()) / max(1, total_modules)
        
        return f"""
Lazy Loading Performance Report:
- Modules lazy loaded: {total_modules}
- Average load time: {avg_load_time:.3f}s
- Estimated memory savings: {self.memory_savings / 1024 / 1024:.1f}MB
- Fastest load: {min(self.load_times.values()):.3f}s
- Slowest load: {max(self.load_times.values()):.3f}s
"""


# Convenience functions
def get_lazy_loading_stats():
    """Get global lazy loading statistics"""
    return _global_loader.get_loading_stats()


def optimize_lazy_memory():
    """Optimize memory usage of lazy loaded modules"""
    _global_loader.optimize_memory()


if __name__ == "__main__":
    # Demo of emergency lazy loading
    print("🚀 Setting up emergency lazy loading...")
    
    lazy_modules = setup_emergency_lazy_imports()
    
    # Show initial stats
    stats = get_lazy_loading_stats()
    print(f"Registered {stats['registered_modules']} modules for lazy loading")
    print(f"Currently loaded: {stats['loaded_modules']} modules")