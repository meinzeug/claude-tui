#!/usr/bin/env python3
"""
API Performance Profiler - Deep performance analysis tool.

Uses cProfile, line_profiler, and memory_profiler to identify
performance bottlenecks in API endpoints.
"""

import cProfile
import pstats
import io
import sys
import asyncio
import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
from functools import wraps
import tracemalloc
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIProfiler:
    """
    Comprehensive API performance profiler.
    
    Features:
    - Function-level profiling with cProfile
    - Memory usage tracking
    - Async operation analysis
    - Hot path identification
    - Performance bottleneck detection
    """
    
    def __init__(self, output_dir: str = "/tmp/profiling"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Profile storage
        self.profiles = {}
        self.memory_snapshots = []
        self.performance_data = {}
        
        # Start memory tracking
        tracemalloc.start()
    
    @contextmanager
    def profile_endpoint(self, endpoint_name: str):
        """Context manager for profiling an endpoint."""
        logger.info(f"Starting profiling for endpoint: {endpoint_name}")
        
        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Memory snapshot before
        snapshot_before = tracemalloc.take_snapshot()
        process = psutil.Process()
        memory_before = process.memory_info().rss
        cpu_before = process.cpu_percent()
        
        start_time = time.time()
        
        try:
            yield profiler
        finally:
            # Stop profiling
            execution_time = time.time() - start_time
            profiler.disable()
            
            # Memory snapshot after
            snapshot_after = tracemalloc.take_snapshot()
            memory_after = process.memory_info().rss
            cpu_after = process.cpu_percent()
            
            # Store results
            self.profiles[endpoint_name] = {
                'profiler': profiler,
                'execution_time': execution_time,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before,
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'snapshot_before': snapshot_before,
                'snapshot_after': snapshot_after
            }
            
            logger.info(
                f"Profiling completed for {endpoint_name}: "
                f"{execution_time:.3f}s, {(memory_after-memory_before)/1024/1024:.2f}MB"
            )
    
    def analyze_profiles(self) -> Dict[str, Any]:
        """Analyze all collected profiles."""
        logger.info("Analyzing performance profiles...")
        
        analysis = {}
        
        for endpoint_name, profile_data in self.profiles.items():
            logger.info(f"Analyzing {endpoint_name}...")
            
            # Profile statistics
            stats = pstats.Stats(profile_data['profiler'])
            stats.sort_stats('cumulative')
            
            # Capture profile output
            stats_output = io.StringIO()
            stats.print_stats(20)  # Top 20 functions
            stats_output.seek(0)
            profile_text = stats_output.getvalue()
            
            # Memory analysis
            top_stats = profile_data['snapshot_after'].compare_to(
                profile_data['snapshot_before'], 'lineno'
            )[:10]
            
            memory_hotspots = []
            for stat in top_stats:
                memory_hotspots.append({
                    'filename': stat.traceback.format()[-1].strip(),
                    'size_diff': stat.size_diff,
                    'count_diff': stat.count_diff
                })
            
            # Performance analysis
            analysis[endpoint_name] = {
                'execution_time': profile_data['execution_time'],
                'memory_usage': {
                    'before_mb': profile_data['memory_before'] / 1024 / 1024,
                    'after_mb': profile_data['memory_after'] / 1024 / 1024,
                    'delta_mb': profile_data['memory_delta'] / 1024 / 1024
                },
                'cpu_usage': {
                    'before_percent': profile_data['cpu_before'],
                    'after_percent': profile_data['cpu_after']
                },
                'profile_stats': profile_text,
                'memory_hotspots': memory_hotspots,
                'performance_rating': self._rate_performance(profile_data),
                'recommendations': self._generate_recommendations(profile_data, stats)
            }
        
        return analysis
    
    def _rate_performance(self, profile_data: Dict[str, Any]) -> str:
        """Rate endpoint performance."""
        execution_time = profile_data['execution_time']
        memory_delta_mb = profile_data['memory_delta'] / 1024 / 1024
        
        if execution_time < 0.1 and memory_delta_mb < 10:
            return "EXCELLENT"
        elif execution_time < 0.5 and memory_delta_mb < 50:
            return "GOOD"
        elif execution_time < 2.0 and memory_delta_mb < 100:
            return "ACCEPTABLE"
        else:
            return "NEEDS_OPTIMIZATION"
    
    def _generate_recommendations(
        self, 
        profile_data: Dict[str, Any], 
        stats: pstats.Stats
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        execution_time = profile_data['execution_time']
        memory_delta_mb = profile_data['memory_delta'] / 1024 / 1024
        
        # Time-based recommendations
        if execution_time > 1.0:
            recommendations.append("Consider async optimization for long-running operations")
        
        if execution_time > 0.5:
            recommendations.append("Implement response caching")
            recommendations.append("Add request batching for bulk operations")
        
        # Memory-based recommendations
        if memory_delta_mb > 50:
            recommendations.append("Investigate memory leaks")
            recommendations.append("Implement object pooling")
            recommendations.append("Add memory-efficient data structures")
        
        # Function-level recommendations
        stats.sort_stats('cumulative')
        top_functions = stats.get_stats()
        
        # Find slow functions
        for func_key, (cc, nc, tt, ct, callers) in list(top_functions.items())[:5]:
            if ct > execution_time * 0.3:  # Function takes >30% of total time
                func_name = func_key[2] if len(func_key) > 2 else str(func_key)
                recommendations.append(f"Optimize function: {func_name}")
        
        return recommendations
    
    def save_reports(self, analysis: Dict[str, Any]) -> str:
        """Save detailed profiling reports."""
        timestamp = int(time.time())
        report_file = f"{self.output_dir}/profiling_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("API PERFORMANCE PROFILING REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {time.ctime()}\n")
            f.write(f"Total Endpoints Profiled: {len(analysis)}\n\n")
            
            for endpoint_name, endpoint_analysis in analysis.items():
                f.write(f"ENDPOINT: {endpoint_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Execution Time: {endpoint_analysis['execution_time']:.4f}s\n")
                f.write(f"Memory Usage: {endpoint_analysis['memory_usage']['delta_mb']:.2f}MB\n")
                f.write(f"Performance Rating: {endpoint_analysis['performance_rating']}\n")
                f.write("\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS:\n")
                for rec in endpoint_analysis['recommendations']:
                    f.write(f"  • {rec}\n")
                f.write("\n")
                
                # Memory hotspots
                f.write("MEMORY HOTSPOTS:\n")
                for hotspot in endpoint_analysis['memory_hotspots'][:5]:
                    f.write(f"  • {hotspot['filename']}: {hotspot['size_diff']} bytes\n")
                f.write("\n")
                
                # Profile stats (truncated)
                f.write("TOP FUNCTIONS (by cumulative time):\n")
                profile_lines = endpoint_analysis['profile_stats'].split('\n')[:25]
                f.write('\n'.join(profile_lines))
                f.write("\n\n" + "=" * 50 + "\n\n")
        
        logger.info(f"Profiling report saved to: {report_file}")
        return report_file
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile individual functions."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            with self.profile_endpoint(func_name):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            with self.profile_endpoint(func_name):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Global profiler instance
_profiler_instance: Optional[APIProfiler] = None


def get_profiler() -> APIProfiler:
    """Get global profiler instance."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = APIProfiler()
    return _profiler_instance


def profile_endpoint(endpoint_name: str):
    """Decorator for profiling endpoints."""
    def decorator(func: Callable) -> Callable:
        profiler = get_profiler()
        return profiler.profile_function(func)
    
    return decorator


async def profile_api_server():
    """Profile running API server by making requests."""
    import aiohttp
    
    profiler = get_profiler()
    base_url = "http://localhost:8000"
    
    endpoints_to_test = [
        ("/health", "GET"),
        ("/api/v1/ai/performance", "GET"),
        ("/api/v1/ai/code/generate", "POST", {
            "prompt": "def fibonacci(n): pass",
            "language": "python"
        })
    ]
    
    logger.info("Starting API server profiling...")
    
    async with aiohttp.ClientSession() as session:
        for endpoint_path, method, *payload in endpoints_to_test:
            endpoint_name = f"{method}_{endpoint_path.replace('/', '_')}"
            
            with profiler.profile_endpoint(endpoint_name):
                try:
                    if method == "GET":
                        async with session.get(f"{base_url}{endpoint_path}") as response:
                            await response.text()
                    elif method == "POST":
                        data = payload[0] if payload else {}
                        async with session.post(f"{base_url}{endpoint_path}", json=data) as response:
                            await response.text()
                    
                    logger.info(f"Profiled: {endpoint_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to profile {endpoint_name}: {e}")
                    continue
                
                # Brief pause between requests
                await asyncio.sleep(0.1)
    
    # Analyze and save results
    analysis = profiler.analyze_profiles()
    report_file = profiler.save_reports(analysis)
    
    return analysis, report_file


def main():
    """Main profiling function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Performance Profiler")
    parser.add_argument('--server', action='store_true', 
                       help='Profile running API server')
    parser.add_argument('--output-dir', default='/tmp/profiling',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    if args.server:
        # Profile running server
        analysis, report_file = asyncio.run(profile_api_server())
        
        # Print summary
        print("\nPROFILING SUMMARY")
        print("=" * 50)
        
        for endpoint_name, endpoint_analysis in analysis.items():
            rating = endpoint_analysis['performance_rating']
            time_ms = endpoint_analysis['execution_time'] * 1000
            memory_mb = endpoint_analysis['memory_usage']['delta_mb']
            
            status_icon = {
                'EXCELLENT': '✅',
                'GOOD': '👍',
                'ACCEPTABLE': '⚠️',
                'NEEDS_OPTIMIZATION': '❌'
            }.get(rating, '❓')
            
            print(f"{status_icon} {endpoint_name}: {time_ms:.1f}ms, {memory_mb:.1f}MB - {rating}")
        
        print(f"\nDetailed report: {report_file}")
        
        # Check for optimization needs
        needs_optimization = [
            name for name, analysis in analysis.items()
            if analysis['performance_rating'] == 'NEEDS_OPTIMIZATION'
        ]
        
        if needs_optimization:
            print(f"\n❌ Endpoints needing optimization: {', '.join(needs_optimization)}")
            exit(1)
        else:
            print("\n✅ All endpoints performing well!")
            exit(0)
    
    else:
        print("Use --server to profile running API server")
        exit(1)


if __name__ == "__main__":
    main()