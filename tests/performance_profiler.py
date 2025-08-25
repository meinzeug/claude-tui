#!/usr/bin/env python3
"""
Performance Profiler for Claude-TUI Application
Comprehensive memory, CPU, and I/O profiling with optimization recommendations.
"""

import asyncio
import cProfile
import gc
import io
import json
import os
import pstats
import psutil
import resource
import sys
import time
import threading
import traceback
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import Mock, patch

import memory_profiler
from memory_profiler import profile


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    rss: int  # Resident Set Size (physical memory)
    vms: int  # Virtual Memory Size
    percent: float  # Memory percentage
    available: int  # Available memory
    peak_rss: int  # Peak RSS usage
    objects_count: int  # Number of objects
    gc_stats: Dict[str, int]  # Garbage collector stats


@dataclass
class CPUMetrics:
    """CPU usage metrics"""
    timestamp: float
    cpu_percent: float
    cpu_times: Dict[str, float]
    context_switches: int
    threads: int


@dataclass
class IOMetrics:
    """I/O metrics"""
    timestamp: float
    read_count: int
    write_count: int
    read_bytes: int
    write_bytes: int


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    startup_time: float
    memory_snapshots: List[MemorySnapshot]
    cpu_metrics: List[CPUMetrics]
    io_metrics: List[IOMetrics]
    peak_memory_mb: float
    average_memory_mb: float
    peak_cpu_percent: float
    average_cpu_percent: float
    memory_leaks: List[str]
    bottlenecks: List[str]
    optimization_recommendations: List[str]
    test_results: Dict[str, Any]


class MemoryProfiler:
    """Memory profiling and leak detection"""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.start_time: Optional[float] = None
        self.baseline_objects: Optional[int] = None
        self.process = psutil.Process()
        
    def start(self):
        """Start memory profiling"""
        self.start_time = time.time()
        tracemalloc.start()
        gc.collect()  # Clear initial garbage
        self.baseline_objects = len(gc.get_objects())
        self._take_snapshot()
        
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        if not self.start_time:
            raise RuntimeError("Profiler not started")
            
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        virtual_memory = psutil.virtual_memory()
        
        # Get garbage collector stats
        gc_stats = {f"gen{i}": len(gc.get_objects(i)) for i in range(3)}
        gc_stats["total_objects"] = len(gc.get_objects())
        
        snapshot = MemorySnapshot(
            timestamp=time.time() - self.start_time,
            rss=memory_info.rss,
            vms=memory_info.vms,
            percent=memory_percent,
            available=virtual_memory.available,
            peak_rss=getattr(memory_info, 'peak_rss', memory_info.rss),
            objects_count=gc_stats["total_objects"],
            gc_stats=gc_stats
        )
        
        self.snapshots.append(snapshot)
        return snapshot
        
    def monitor(self, interval: float = 1.0) -> None:
        """Continuous memory monitoring"""
        def monitor_loop():
            while True:
                try:
                    self._take_snapshot()
                    time.sleep(interval)
                except Exception as e:
                    print(f"Memory monitoring error: {e}")
                    break
                    
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
    def detect_leaks(self) -> List[str]:
        """Detect potential memory leaks"""
        leaks = []
        
        if len(self.snapshots) < 3:
            return leaks
            
        # Check for consistently growing memory
        recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
        if len(recent_snapshots) >= 3:
            memory_trend = []
            for i in range(1, len(recent_snapshots)):
                diff = recent_snapshots[i].rss - recent_snapshots[i-1].rss
                memory_trend.append(diff)
                
            # If memory consistently grows
            if all(diff > 0 for diff in memory_trend[-3:]):
                leaks.append(f"Potential memory leak: consistent growth over {len(memory_trend)} samples")
                
        # Check object count growth
        if self.baseline_objects:
            current_objects = self.snapshots[-1].objects_count
            growth = current_objects - self.baseline_objects
            if growth > 1000:  # More than 1000 objects created
                leaks.append(f"Object count grew by {growth} objects")
                
        # Check for large memory jumps
        for i in range(1, len(self.snapshots)):
            prev_mb = self.snapshots[i-1].rss / 1024 / 1024
            curr_mb = self.snapshots[i].rss / 1024 / 1024
            if curr_mb - prev_mb > 50:  # 50MB jump
                leaks.append(f"Large memory jump of {curr_mb - prev_mb:.1f}MB at {self.snapshots[i].timestamp:.1f}s")
                
        return leaks
        
    def stop(self):
        """Stop memory profiling"""
        if tracemalloc.is_tracing():
            tracemalloc.stop()


class CPUProfiler:
    """CPU profiling and performance analysis"""
    
    def __init__(self):
        self.metrics: List[CPUMetrics] = []
        self.start_time: Optional[float] = None
        self.process = psutil.Process()
        self.profiler: Optional[cProfile.Profile] = None
        
    def start(self):
        """Start CPU profiling"""
        self.start_time = time.time()
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self._take_measurement()
        
    def _take_measurement(self) -> CPUMetrics:
        """Take a CPU measurement"""
        if not self.start_time:
            raise RuntimeError("Profiler not started")
            
        cpu_times = self.process.cpu_times()._asdict()
        
        try:
            context_switches = self.process.num_ctx_switches().voluntary
        except AttributeError:
            context_switches = 0
            
        metric = CPUMetrics(
            timestamp=time.time() - self.start_time,
            cpu_percent=self.process.cpu_percent(),
            cpu_times=cpu_times,
            context_switches=context_switches,
            threads=self.process.num_threads()
        )
        
        self.metrics.append(metric)
        return metric
        
    def monitor(self, interval: float = 1.0):
        """Continuous CPU monitoring"""
        def monitor_loop():
            while True:
                try:
                    self._take_measurement()
                    time.sleep(interval)
                except Exception as e:
                    print(f"CPU monitoring error: {e}")
                    break
                    
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
    def get_profile_stats(self) -> str:
        """Get detailed CPU profile statistics"""
        if not self.profiler:
            return "No profiling data available"
            
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return stats_stream.getvalue()
        
    def stop(self):
        """Stop CPU profiling"""
        if self.profiler:
            self.profiler.disable()


class IOProfiler:
    """I/O profiling and analysis"""
    
    def __init__(self):
        self.metrics: List[IOMetrics] = []
        self.start_time: Optional[float] = None
        self.process = psutil.Process()
        
    def start(self):
        """Start I/O profiling"""
        self.start_time = time.time()
        self._take_measurement()
        
    def _take_measurement(self) -> IOMetrics:
        """Take an I/O measurement"""
        if not self.start_time:
            raise RuntimeError("Profiler not started")
            
        try:
            io_counters = self.process.io_counters()
            metric = IOMetrics(
                timestamp=time.time() - self.start_time,
                read_count=io_counters.read_count,
                write_count=io_counters.write_count,
                read_bytes=io_counters.read_bytes,
                write_bytes=io_counters.write_bytes
            )
        except AttributeError:
            # I/O counters not available on this platform
            metric = IOMetrics(
                timestamp=time.time() - self.start_time,
                read_count=0,
                write_count=0,
                read_bytes=0,
                write_bytes=0
            )
        
        self.metrics.append(metric)
        return metric
        
    def monitor(self, interval: float = 1.0):
        """Continuous I/O monitoring"""
        def monitor_loop():
            while True:
                try:
                    self._take_measurement()
                    time.sleep(interval)
                except Exception as e:
                    print(f"I/O monitoring error: {e}")
                    break
                    
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()


class TUIPerformanceProfiler:
    """Comprehensive TUI application performance profiler"""
    
    def __init__(self, app_module: str = "src.ui.main_app"):
        self.app_module = app_module
        self.memory_profiler = MemoryProfiler()
        self.cpu_profiler = CPUProfiler()
        self.io_profiler = IOProfiler()
        self.startup_time: float = 0.0
        self.test_results: Dict[str, Any] = {}
        
    def profile_startup(self) -> float:
        """Profile application startup time"""
        print("📊 Profiling application startup...")
        
        start_time = time.time()
        
        try:
            # Mock TUI to avoid GUI dependencies during testing
            with patch('textual.app.App.run_async'):
                from src.ui.main_app import ClaudeTUIApp
                
                app = ClaudeTUIApp()
                # Initialize core components
                app.init_core_systems()
                
        except Exception as e:
            print(f"Startup profiling error: {e}")
            
        self.startup_time = time.time() - start_time
        print(f"✓ Startup time: {self.startup_time:.3f}s")
        
        return self.startup_time
    
    @contextmanager
    def profile_context(self):
        """Context manager for comprehensive profiling"""
        print("🔍 Starting comprehensive profiling...")
        
        # Start all profilers
        self.memory_profiler.start()
        self.cpu_profiler.start()  
        self.io_profiler.start()
        
        # Start monitoring threads
        self.memory_profiler.monitor(0.5)
        self.cpu_profiler.monitor(0.5)
        self.io_profiler.monitor(0.5)
        
        try:
            yield self
        finally:
            # Stop profilers
            self.memory_profiler.stop()
            self.cpu_profiler.stop()
            
            print("✓ Profiling complete")
    
    def test_large_dataset_performance(self) -> Dict[str, Any]:
        """Test performance with large datasets"""
        print("📈 Testing large dataset performance...")
        
        results = {
            "tasks_1000": self._test_task_performance(1000),
            "files_1000": self._test_file_performance(1000),
            "widgets_100": self._test_widget_performance(100)
        }
        
        self.test_results["large_dataset"] = results
        return results
        
    def _test_task_performance(self, task_count: int) -> Dict[str, float]:
        """Test performance with many tasks"""
        start_time = time.time()
        
        # Simulate creating many tasks
        tasks = []
        for i in range(task_count):
            task = {
                "id": i,
                "name": f"Task {i}",
                "description": f"Description for task {i}",
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            tasks.append(task)
            
        creation_time = time.time() - start_time
        
        # Test task processing
        start_time = time.time()
        processed = 0
        for task in tasks:
            # Simulate task processing
            task["processed"] = True
            processed += 1
            
        processing_time = time.time() - start_time
        
        return {
            "creation_time": creation_time,
            "processing_time": processing_time,
            "tasks_per_second": task_count / (creation_time + processing_time)
        }
        
    def _test_file_performance(self, file_count: int) -> Dict[str, float]:
        """Test performance with many files"""
        start_time = time.time()
        
        # Simulate file tree with many files
        file_tree = {}
        for i in range(file_count):
            path = f"src/module_{i % 10}/file_{i}.py"
            file_tree[path] = {
                "size": 1024 + (i * 100),
                "modified": datetime.now().timestamp(),
                "lines": 50 + (i % 100)
            }
            
        tree_creation_time = time.time() - start_time
        
        # Test tree traversal
        start_time = time.time()
        total_size = sum(f["size"] for f in file_tree.values())
        total_lines = sum(f["lines"] for f in file_tree.values())
        
        traversal_time = time.time() - start_time
        
        return {
            "tree_creation_time": tree_creation_time,
            "traversal_time": traversal_time,
            "files_per_second": file_count / (tree_creation_time + traversal_time),
            "total_size": total_size,
            "total_lines": total_lines
        }
        
    def _test_widget_performance(self, widget_count: int) -> Dict[str, float]:
        """Test widget rendering performance"""
        start_time = time.time()
        
        # Simulate widget creation and updates
        widgets = []
        for i in range(widget_count):
            widget = {
                "id": f"widget_{i}",
                "type": ["button", "label", "input", "list"][i % 4],
                "data": f"Widget data {i}",
                "visible": True,
                "updated": 0
            }
            widgets.append(widget)
            
        creation_time = time.time() - start_time
        
        # Simulate widget updates (like re-rendering)
        start_time = time.time()
        for widget in widgets:
            widget["updated"] += 1
            widget["data"] = f"Updated {widget['data']}"
            
        update_time = time.time() - start_time
        
        return {
            "creation_time": creation_time,
            "update_time": update_time,
            "widgets_per_second": widget_count / (creation_time + update_time)
        }
        
    def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze startup time
        if self.startup_time > 3.0:
            bottlenecks.append(f"Slow startup time: {self.startup_time:.3f}s (>3s)")
            
        # Analyze memory usage
        if self.memory_profiler.snapshots:
            peak_mb = max(s.rss for s in self.memory_profiler.snapshots) / 1024 / 1024
            if peak_mb > 100:
                bottlenecks.append(f"High memory usage: {peak_mb:.1f}MB (>100MB)")
                
        # Analyze CPU usage
        if self.cpu_profiler.metrics:
            avg_cpu = sum(m.cpu_percent for m in self.cpu_profiler.metrics) / len(self.cpu_profiler.metrics)
            peak_cpu = max(m.cpu_percent for m in self.cpu_profiler.metrics)
            
            if avg_cpu > 50:
                bottlenecks.append(f"High average CPU usage: {avg_cpu:.1f}% (>50%)")
            if peak_cpu > 80:
                bottlenecks.append(f"High peak CPU usage: {peak_cpu:.1f}% (>80%)")
                
        # Analyze test results
        if "large_dataset" in self.test_results:
            tasks_result = self.test_results["large_dataset"]["tasks_1000"]
            if tasks_result["tasks_per_second"] < 100:
                bottlenecks.append(f"Slow task processing: {tasks_result['tasks_per_second']:.1f} tasks/s (<100)")
                
        return bottlenecks
        
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Memory optimizations
        if self.memory_profiler.snapshots:
            peak_mb = max(s.rss for s in self.memory_profiler.snapshots) / 1024 / 1024
            if peak_mb > 50:
                recommendations.append("Consider implementing lazy loading for large datasets")
                recommendations.append("Use object pooling for frequently created/destroyed objects")
                recommendations.append("Implement memory-efficient data structures")
                
        # CPU optimizations
        if self.cpu_profiler.metrics:
            avg_cpu = sum(m.cpu_percent for m in self.cpu_profiler.metrics) / len(self.cpu_profiler.metrics)
            if avg_cpu > 30:
                recommendations.append("Consider moving heavy computations to background threads")
                recommendations.append("Implement caching for expensive operations")
                recommendations.append("Optimize widget update frequency")
                
        # Startup optimizations
        if self.startup_time > 2.0:
            recommendations.append("Delay non-critical component initialization")
            recommendations.append("Use asynchronous initialization where possible")
            recommendations.append("Reduce import overhead with lazy imports")
            
        # General recommendations
        recommendations.extend([
            "Implement virtual scrolling for large lists",
            "Use batch updates for multiple UI changes",
            "Cache frequently accessed data",
            "Profile individual widget render times",
            "Consider using C extensions for performance-critical code"
        ])
        
        return recommendations
        
    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        memory_leaks = self.memory_profiler.detect_leaks()
        bottlenecks = self.identify_bottlenecks()
        recommendations = self.generate_optimization_recommendations()
        
        # Calculate statistics
        peak_memory_mb = 0.0
        average_memory_mb = 0.0
        if self.memory_profiler.snapshots:
            memory_values = [s.rss / 1024 / 1024 for s in self.memory_profiler.snapshots]
            peak_memory_mb = max(memory_values)
            average_memory_mb = sum(memory_values) / len(memory_values)
            
        peak_cpu_percent = 0.0
        average_cpu_percent = 0.0
        if self.cpu_profiler.metrics:
            cpu_values = [m.cpu_percent for m in self.cpu_profiler.metrics]
            peak_cpu_percent = max(cpu_values)
            average_cpu_percent = sum(cpu_values) / len(cpu_values)
        
        return PerformanceReport(
            startup_time=self.startup_time,
            memory_snapshots=self.memory_profiler.snapshots,
            cpu_metrics=self.cpu_profiler.metrics,
            io_metrics=self.io_profiler.metrics,
            peak_memory_mb=peak_memory_mb,
            average_memory_mb=average_memory_mb,
            peak_cpu_percent=peak_cpu_percent,
            average_cpu_percent=average_cpu_percent,
            memory_leaks=memory_leaks,
            bottlenecks=bottlenecks,
            optimization_recommendations=recommendations,
            test_results=self.test_results
        )


def run_comprehensive_performance_analysis():
    """Run comprehensive performance analysis"""
    print("🚀 Starting Claude-TUI Performance Analysis")
    print("=" * 50)
    
    profiler = TUIPerformanceProfiler()
    
    # Test startup performance
    profiler.profile_startup()
    
    # Run comprehensive profiling
    with profiler.profile_context():
        # Wait for baseline measurements
        time.sleep(2)
        
        # Test large dataset performance  
        profiler.test_large_dataset_performance()
        
        # Simulate some work
        time.sleep(3)
        
    # Generate report
    report = profiler.generate_report()
    
    # Display results
    print("\n📊 PERFORMANCE ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"\n⏱️  Startup Time: {report.startup_time:.3f}s")
    
    print(f"\n💾 Memory Usage:")
    print(f"   Peak: {report.peak_memory_mb:.1f}MB")
    print(f"   Average: {report.average_memory_mb:.1f}MB")
    print(f"   Target: <100MB {'✓' if report.peak_memory_mb < 100 else '✗'}")
    
    print(f"\n🖥️  CPU Usage:")
    print(f"   Peak: {report.peak_cpu_percent:.1f}%")
    print(f"   Average: {report.average_cpu_percent:.1f}%")
    
    if report.memory_leaks:
        print(f"\n⚠️  Memory Leaks Detected:")
        for leak in report.memory_leaks:
            print(f"   • {leak}")
    else:
        print(f"\n✓ No memory leaks detected")
        
    if report.bottlenecks:
        print(f"\n🐛 Performance Bottlenecks:")
        for bottleneck in report.bottlenecks:
            print(f"   • {bottleneck}")
    else:
        print(f"\n✓ No major bottlenecks detected")
    
    print(f"\n💡 Optimization Recommendations:")
    for rec in report.optimization_recommendations[:5]:  # Top 5
        print(f"   • {rec}")
    
    # Large dataset test results
    if "large_dataset" in report.test_results:
        print(f"\n📈 Large Dataset Performance:")
        results = report.test_results["large_dataset"]
        print(f"   Tasks (1000): {results['tasks_1000']['tasks_per_second']:.1f} tasks/s")
        print(f"   Files (1000): {results['files_1000']['files_per_second']:.1f} files/s")
        print(f"   Widgets (100): {results['widgets_100']['widgets_per_second']:.1f} widgets/s")
    
    # CPU profiling details
    if hasattr(profiler.cpu_profiler, 'get_profile_stats'):
        cpu_stats = profiler.cpu_profiler.get_profile_stats()
        if cpu_stats and cpu_stats.strip():
            print(f"\n🔍 Top CPU Functions:")
            # Show first few lines of CPU stats
            stats_lines = cpu_stats.split('\n')[:10]
            for line in stats_lines:
                if line.strip():
                    print(f"   {line}")
    
    return report


if __name__ == "__main__":
    # Ensure required dependencies
    try:
        import psutil
        import memory_profiler
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install psutil memory-profiler")
        sys.exit(1)
        
    # Run the analysis
    report = run_comprehensive_performance_analysis()
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"performance_analysis_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {report_file}")