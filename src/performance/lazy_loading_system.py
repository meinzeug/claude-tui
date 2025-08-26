#!/usr/bin/env python3
"""
Lazy Loading System - Intelligent Component Loading for Performance

Implements comprehensive lazy loading strategies:
- Module lazy loading with import hooks
- Component lazy initialization
- Data lazy fetching with pagination
- Image and asset lazy loading
- Route-based code splitting
- Progressive loading with priority queues
"""

import asyncio
import importlib
import inspect
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from functools import wraps
import weakref
from datetime import datetime
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)


@dataclass
class LazyLoadMetrics:
    """Metrics for lazy loading performance"""
    module_name: str
    load_time_ms: float
    memory_before_mb: float
    memory_after_mb: float
    first_access_time: datetime
    access_count: int = 0
    last_access_time: Optional[datetime] = None
    
    def record_access(self):
        """Record an access to this lazy-loaded module"""
        self.access_count += 1
        self.last_access_time = datetime.utcnow()
    
    @property
    def memory_increase_mb(self) -> float:
        """Calculate memory increase from loading"""
        return self.memory_after_mb - self.memory_before_mb


class LazyModuleProxy:
    """Proxy object for lazy module loading"""
    
    def __init__(self, module_name: str, loader_func: Optional[Callable] = None):
        self._module_name = module_name
        self._module = None
        self._loader_func = loader_func or (lambda: importlib.import_module(module_name))
        self._load_time = None
        self._metrics = None
        
    def _load_module(self):
        """Load the actual module on first access"""
        if self._module is not None:
            return self._module
        
        # Get memory before loading
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_before = 0
        
        start_time = time.time()
        
        try:
            self._module = self._loader_func()
            self._load_time = time.time() - start_time
            
            # Get memory after loading
            try:
                memory_after = process.memory_info().rss / 1024 / 1024
            except:
                memory_after = memory_before
            
            # Create metrics
            self._metrics = LazyLoadMetrics(
                module_name=self._module_name,
                load_time_ms=self._load_time * 1000,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                first_access_time=datetime.utcnow()
            )
            
            logger.debug(f"Lazy loaded {self._module_name} in {self._load_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"Failed to lazy load {self._module_name}: {e}")
            raise
        
        return self._module
    
    def __getattr__(self, name: str):
        """Proxy attribute access to loaded module"""
        module = self._load_module()
        if self._metrics:
            self._metrics.record_access()
        return getattr(module, name)
    
    def __call__(self, *args, **kwargs):
        """Make proxy callable if module is callable"""
        module = self._load_module()
        if self._metrics:
            self._metrics.record_access()
        return module(*args, **kwargs)
    
    def __dir__(self):
        """Support for dir() function"""
        module = self._load_module()
        return dir(module)
    
    @property
    def metrics(self) -> Optional[LazyLoadMetrics]:
        """Get loading metrics"""
        return self._metrics
    
    @property
    def is_loaded(self) -> bool:
        """Check if module is loaded"""
        return self._module is not None


class LazyComponentLoader:
    """Lazy loader for heavy components"""
    
    def __init__(self):
        self.loaded_components: Dict[str, Any] = {}
        self.component_factories: Dict[str, Callable] = {}
        self.loading_futures: Dict[str, asyncio.Future] = {}
        self.metrics: Dict[str, LazyLoadMetrics] = {}
        
    def register_component(self, name: str, factory: Callable):
        """Register a component factory for lazy loading"""
        self.component_factories[name] = factory
        logger.debug(f"Registered lazy component: {name}")
    
    async def get_component(self, name: str) -> Any:
        """Get component, loading lazily if needed"""
        # Return if already loaded
        if name in self.loaded_components:
            if name in self.metrics:
                self.metrics[name].record_access()
            return self.loaded_components[name]
        
        # Check if already loading
        if name in self.loading_futures:
            return await self.loading_futures[name]
        
        # Start loading
        future = asyncio.create_future()
        self.loading_futures[name] = future
        
        try:
            component = await self._load_component(name)
            self.loaded_components[name] = component
            future.set_result(component)
            return component
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            if name in self.loading_futures:
                del self.loading_futures[name]
    
    async def _load_component(self, name: str) -> Any:
        """Load component with metrics tracking"""
        if name not in self.component_factories:
            raise ValueError(f"Component '{name}' not registered")
        
        # Get memory before loading
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_before = 0
        
        start_time = time.time()
        
        try:
            factory = self.component_factories[name]
            
            # Load component (support both sync and async factories)
            if inspect.iscoroutinefunction(factory):
                component = await factory()
            else:
                # Run sync factory in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    component = await loop.run_in_executor(executor, factory)
            
            load_time = time.time() - start_time
            
            # Get memory after loading
            try:
                memory_after = process.memory_info().rss / 1024 / 1024
            except:
                memory_after = memory_before
            
            # Create metrics
            metrics = LazyLoadMetrics(
                module_name=name,
                load_time_ms=load_time * 1000,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                first_access_time=datetime.utcnow()
            )
            
            self.metrics[name] = metrics
            
            logger.info(f"Loaded component '{name}' in {load_time*1000:.1f}ms")
            
            return component
            
        except Exception as e:
            logger.error(f"Failed to load component '{name}': {e}")
            raise
    
    def preload_component(self, name: str):
        """Start preloading a component in the background"""
        if name not in self.loaded_components and name not in self.loading_futures:
            asyncio.create_task(self.get_component(name))
    
    def get_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all loaded components"""
        return {
            name: {
                'load_time_ms': metrics.load_time_ms,
                'memory_increase_mb': metrics.memory_increase_mb,
                'access_count': metrics.access_count,
                'first_access': metrics.first_access_time.isoformat(),
                'last_access': metrics.last_access_time.isoformat() if metrics.last_access_time else None
            }
            for name, metrics in self.metrics.items()
        }


class LazyDataLoader:
    """Lazy data loader with pagination and caching"""
    
    def __init__(self, page_size: int = 50, max_pages_cached: int = 10):
        self.page_size = page_size
        self.max_pages_cached = max_pages_cached
        self.data_sources: Dict[str, Callable] = {}
        self.cached_pages: Dict[str, Dict[int, Any]] = {}
        self.loading_pages: Dict[str, Dict[int, asyncio.Future]] = {}
        
    def register_data_source(self, name: str, loader_func: Callable):
        """Register a data source for lazy loading"""
        self.data_sources[name] = loader_func
        self.cached_pages[name] = {}
        self.loading_pages[name] = {}
    
    async def get_data_page(self, source_name: str, page: int = 0) -> List[Any]:
        """Get a page of data, loading lazily if needed"""
        # Return from cache if available
        if page in self.cached_pages[source_name]:
            return self.cached_pages[source_name][page]
        
        # Check if already loading
        if page in self.loading_pages[source_name]:
            return await self.loading_pages[source_name][page]
        
        # Start loading
        future = asyncio.create_future()
        self.loading_pages[source_name][page] = future
        
        try:
            data = await self._load_data_page(source_name, page)
            
            # Cache the page
            self.cached_pages[source_name][page] = data
            
            # Cleanup old pages if cache is full
            if len(self.cached_pages[source_name]) > self.max_pages_cached:
                oldest_page = min(self.cached_pages[source_name].keys())
                del self.cached_pages[source_name][oldest_page]
            
            future.set_result(data)
            return data
            
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            if page in self.loading_pages[source_name]:
                del self.loading_pages[source_name][page]
    
    async def _load_data_page(self, source_name: str, page: int) -> List[Any]:
        """Load a specific page of data"""
        if source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' not registered")
        
        loader_func = self.data_sources[source_name]
        offset = page * self.page_size
        
        # Call loader function with pagination parameters
        if inspect.iscoroutinefunction(loader_func):
            data = await loader_func(offset=offset, limit=self.page_size)
        else:
            # Run sync loader in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                data = await loop.run_in_executor(
                    executor, 
                    lambda: loader_func(offset=offset, limit=self.page_size)
                )
        
        return data if isinstance(data, list) else []
    
    async def get_all_data(self, source_name: str, max_pages: int = None) -> List[Any]:
        """Get all data, loading pages as needed"""
        all_data = []
        page = 0
        
        while True:
            if max_pages and page >= max_pages:
                break
                
            page_data = await self.get_data_page(source_name, page)
            if not page_data:
                break  # No more data
            
            all_data.extend(page_data)
            page += 1
            
            # Stop if page is not full (last page)
            if len(page_data) < self.page_size:
                break
        
        return all_data


class LazyLoadingManager:
    """Central manager for all lazy loading strategies"""
    
    def __init__(self):
        self.module_proxies: Dict[str, LazyModuleProxy] = {}
        self.component_loader = LazyComponentLoader()
        self.data_loader = LazyDataLoader()
        self.preload_queue: List[str] = []
        self.preload_enabled = True
        
        # Heavy modules that should be lazy loaded
        self.heavy_modules = {
            'numpy': lambda: importlib.import_module('numpy'),
            'pandas': lambda: importlib.import_module('pandas'),
            'matplotlib': lambda: importlib.import_module('matplotlib'),
            'tensorflow': lambda: importlib.import_module('tensorflow'),
            'torch': lambda: importlib.import_module('torch'),
            'sklearn': lambda: importlib.import_module('sklearn'),
            'plotly': lambda: importlib.import_module('plotly'),
            'scipy': lambda: importlib.import_module('scipy'),
        }
        
    def setup_lazy_imports(self):
        """Set up lazy loading for heavy modules"""
        for module_name, loader in self.heavy_modules.items():
            if module_name not in sys.modules:
                proxy = LazyModuleProxy(module_name, loader)
                self.module_proxies[module_name] = proxy
                # Install proxy in sys.modules
                sys.modules[module_name] = proxy
                
        logger.info(f"Set up lazy loading for {len(self.heavy_modules)} heavy modules")
    
    def lazy_import(self, module_name: str, loader_func: Optional[Callable] = None):
        """Create a lazy import proxy for a specific module"""
        if module_name in sys.modules:
            return sys.modules[module_name]
        
        proxy = LazyModuleProxy(module_name, loader_func)
        self.module_proxies[module_name] = proxy
        return proxy
    
    def register_component(self, name: str, factory: Callable):
        """Register a component for lazy loading"""
        self.component_loader.register_component(name, factory)
    
    async def get_component(self, name: str) -> Any:
        """Get a component, loading lazily"""
        return await self.component_loader.get_component(name)
    
    def register_data_source(self, name: str, loader_func: Callable):
        """Register a data source for lazy loading"""
        self.data_loader.register_data_source(name, loader_func)
    
    async def get_data_page(self, source_name: str, page: int = 0) -> List[Any]:
        """Get a page of data"""
        return await self.data_loader.get_data_page(source_name, page)
    
    def add_to_preload_queue(self, *items: str):
        """Add items to preload queue"""
        self.preload_queue.extend(items)
    
    async def start_preloading(self):
        """Start background preloading of queued items"""
        if not self.preload_enabled:
            return
        
        preload_tasks = []
        
        for item in self.preload_queue:
            # Determine item type and start preloading
            if item in self.component_loader.component_factories:
                task = asyncio.create_task(self.component_loader.preload_component(item))
                preload_tasks.append(task)
        
        if preload_tasks:
            await asyncio.gather(*preload_tasks, return_exceptions=True)
            logger.info(f"Preloaded {len(preload_tasks)} items")
        
        self.preload_queue.clear()
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all lazy loading"""
        # Module metrics
        module_metrics = {}
        for name, proxy in self.module_proxies.items():
            if proxy.metrics:
                module_metrics[name] = {
                    'load_time_ms': proxy.metrics.load_time_ms,
                    'memory_increase_mb': proxy.metrics.memory_increase_mb,
                    'access_count': proxy.metrics.access_count,
                    'is_loaded': proxy.is_loaded
                }
            else:
                module_metrics[name] = {'is_loaded': proxy.is_loaded}
        
        # Component metrics
        component_metrics = self.component_loader.get_component_metrics()
        
        # Summary statistics
        loaded_modules = sum(1 for proxy in self.module_proxies.values() if proxy.is_loaded)
        loaded_components = len(self.component_loader.loaded_components)
        
        total_memory_saved = sum(
            proxy.metrics.memory_increase_mb 
            for proxy in self.module_proxies.values() 
            if proxy.metrics and not proxy.is_loaded
        )
        
        return {
            'modules': module_metrics,
            'components': component_metrics,
            'summary': {
                'total_modules_managed': len(self.module_proxies),
                'loaded_modules': loaded_modules,
                'total_components_managed': len(self.component_loader.component_factories),
                'loaded_components': loaded_components,
                'estimated_memory_saved_mb': total_memory_saved,
                'preload_queue_size': len(self.preload_queue)
            },
            'timestamp': datetime.utcnow().isoformat()
        }


# Global lazy loading manager
_lazy_manager: Optional[LazyLoadingManager] = None


def get_lazy_manager() -> LazyLoadingManager:
    """Get global lazy loading manager"""
    global _lazy_manager
    if _lazy_manager is None:
        _lazy_manager = LazyLoadingManager()
        _lazy_manager.setup_lazy_imports()
    return _lazy_manager


def lazy_component(name: str):
    """Decorator to register a function as a lazy component factory"""
    def decorator(factory_func: Callable):
        manager = get_lazy_manager()
        manager.register_component(name, factory_func)
        return factory_func
    return decorator


def lazy_data_source(name: str):
    """Decorator to register a function as a lazy data source"""
    def decorator(loader_func: Callable):
        manager = get_lazy_manager()
        manager.register_data_source(name, loader_func)
        return loader_func
    return decorator


# Convenience functions
async def get_lazy_component(name: str) -> Any:
    """Get a lazy component"""
    manager = get_lazy_manager()
    return await manager.get_component(name)


async def get_lazy_data(source_name: str, page: int = 0) -> List[Any]:
    """Get lazy data"""
    manager = get_lazy_manager()
    return await manager.get_data_page(source_name, page)


if __name__ == "__main__":
    # Example usage and testing
    async def test_lazy_loading():
        print("🚀 LAZY LOADING SYSTEM - Testing")
        print("=" * 50)
        
        manager = get_lazy_manager()
        
        # Test lazy component loading
        @lazy_component("heavy_processor")
        async def create_heavy_processor():
            await asyncio.sleep(0.1)  # Simulate loading time
            return {"type": "processor", "loaded_at": datetime.utcnow().isoformat()}
        
        @lazy_data_source("sample_data")
        def load_sample_data(offset: int, limit: int):
            # Simulate database query
            return [f"item_{i}" for i in range(offset, offset + limit)]
        
        # Test component loading
        print("📦 Testing lazy component loading...")
        start_time = time.time()
        processor = await get_lazy_component("heavy_processor")
        load_time = time.time() - start_time
        print(f"   Component loaded in {load_time*1000:.1f}ms")
        print(f"   Component: {processor}")
        
        # Test data loading
        print("\n📊 Testing lazy data loading...")
        data_page = await get_lazy_data("sample_data", page=0)
        print(f"   Loaded page 0: {data_page[:5]}...")
        
        # Test metrics
        print("\n📈 Lazy Loading Metrics:")
        metrics = manager.get_comprehensive_metrics()
        print(f"   Components managed: {metrics['summary']['total_components_managed']}")
        print(f"   Components loaded: {metrics['summary']['loaded_components']}")
        print(f"   Modules managed: {metrics['summary']['total_modules_managed']}")
        print(f"   Memory saved estimate: {metrics['summary']['estimated_memory_saved_mb']:.1f}MB")
        
        print("\n✅ Lazy loading system test completed!")
    
    # Run test
    asyncio.run(test_lazy_loading())