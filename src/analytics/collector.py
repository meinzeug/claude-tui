"""
Metrics Collection and Aggregation System.

Advanced metrics collector that integrates with existing systems and provides
comprehensive data collection, aggregation, and storage capabilities.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..core.types import SystemMetrics, ProgressMetrics
from .models import (
    PerformanceMetrics, AnalyticsData, AnalyticsConfiguration,
    MetricType, AnalyticsStatus
)


class MetricsAggregator:
    """Handles aggregation of metrics over different time windows."""
    
    def __init__(self, config: AnalyticsConfiguration):
        self.config = config
        self.aggregation_windows = {
            interval.total_seconds(): deque(maxlen=int(86400 / interval.total_seconds()))  # 24h of data
            for interval in config.aggregation_intervals
        }
        self.last_aggregation = {
            interval.total_seconds(): datetime.utcnow() - interval
            for interval in config.aggregation_intervals
        }
    
    async def aggregate_metrics(
        self,
        raw_metrics: List[PerformanceMetrics],
        window_seconds: float
    ) -> PerformanceMetrics:
        """Aggregate metrics over a time window."""
        if not raw_metrics:
            return PerformanceMetrics()
        
        # Group numeric fields for aggregation
        numeric_fields = [
            'cpu_percent', 'memory_percent', 'memory_used', 'disk_percent',
            'active_tasks', 'cache_hit_rate', 'ai_response_time',
            'throughput', 'latency_p50', 'latency_p95', 'latency_p99',
            'error_rate', 'tokens_per_second', 'model_accuracy',
            'context_window_usage', 'hallucination_rate',
            'workflow_completion_rate', 'task_success_rate',
            'average_task_duration', 'concurrent_workflows',
            'network_io_bytes', 'disk_io_bytes',
            'gpu_utilization', 'gpu_memory_usage',
            'code_quality_score', 'validation_pass_rate',
            'placeholder_detection_rate'
        ]
        
        # Calculate aggregated values
        aggregated = PerformanceMetrics()
        aggregated.timestamp = raw_metrics[-1].timestamp
        aggregated.session_id = raw_metrics[-1].session_id
        aggregated.metric_type = MetricType.APPLICATION
        
        for field in numeric_fields:
            values = [getattr(m, field, 0) for m in raw_metrics if hasattr(m, field)]
            if values:
                if field in ['latency_p95', 'latency_p99']:
                    # Use 95th/99th percentile for latency metrics
                    percentile = 95 if 'p95' in field else 99
                    setattr(aggregated, field, np.percentile(values, percentile))
                elif field in ['throughput', 'tokens_per_second']:
                    # Use sum for rate metrics
                    setattr(aggregated, field, sum(values))
                elif field in ['network_io_bytes', 'disk_io_bytes', 'active_tasks']:
                    # Use sum for cumulative metrics
                    setattr(aggregated, field, sum(values))
                else:
                    # Use mean for most metrics
                    setattr(aggregated, field, statistics.mean(values))
        
        # Copy categorical fields from the latest metric
        latest = raw_metrics[-1]
        aggregated.environment = latest.environment
        aggregated.version = latest.version
        aggregated.user_id = latest.user_id
        aggregated.project_id = latest.project_id
        
        return aggregated
    
    async def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add metrics to aggregation windows."""
        current_time = datetime.utcnow()
        
        for window_seconds in self.aggregation_windows.keys():
            last_agg = self.last_aggregation[window_seconds]
            
            if (current_time - last_agg).total_seconds() >= window_seconds:
                # Time for new aggregation
                window_start = current_time - timedelta(seconds=window_seconds)
                
                # Get metrics within window (this is simplified - real implementation
                # would maintain proper sliding windows)
                self.aggregation_windows[window_seconds].append(metrics)
                self.last_aggregation[window_seconds] = current_time


class MetricsCollector:
    """
    Advanced metrics collection system.
    
    Features:
    - Multiple data source integration
    - Configurable collection intervals
    - Automatic aggregation and storage
    - Plugin-based extensibility
    - Error handling and resilience
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        storage_path: Optional[Path] = None
    ):
        """Initialize the metrics collector."""
        self.config = config or AnalyticsConfiguration()
        self.storage_path = storage_path or Path("data/metrics")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize aggregator
        self.aggregator = MetricsAggregator(self.config)
        
        # Data sources and collectors
        self.data_sources: Dict[str, Callable] = {}
        self.custom_collectors: Dict[str, Callable] = {}
        
        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'last_collection_time': None,
            'average_collection_duration': 0.0
        }
        
        # Data storage
        self.raw_metrics_buffer: deque = deque(maxlen=10000)
        self.aggregated_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Callbacks for real-time processing
        self.metrics_callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
    
    async def initialize(self) -> None:
        """Initialize the metrics collector."""
        try:
            self.logger.info("Initializing Metrics Collector...")
            
            # Create storage directories
            self.storage_path.mkdir(parents=True, exist_ok=True)
            (self.storage_path / "raw").mkdir(exist_ok=True)
            (self.storage_path / "aggregated").mkdir(exist_ok=True)
            (self.storage_path / "archives").mkdir(exist_ok=True)
            
            # Register built-in data sources
            await self._register_builtin_sources()
            
            # Load historical data
            await self._load_historical_metrics()
            
            self.logger.info(f"Metrics Collector initialized with {len(self.data_sources)} data sources")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics collector: {e}")
            raise
    
    async def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_collecting:
            self.logger.warning("Collection is already active")
            return
        
        self.logger.info("Starting metrics collection...")
        self.is_collecting = True
        
        # Start collection loop
        self.collection_task = asyncio.create_task(self._collection_loop())
    
    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self.is_collecting:
            return
        
        self.logger.info("Stopping metrics collection...")
        self.is_collecting = False
        
        # Cancel collection task
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        # Save remaining data
        await self._save_buffered_metrics()
        
        # Close thread pool
        self.thread_pool.shutdown(wait=True)
    
    def register_data_source(
        self,
        name: str,
        collector_func: Callable[[], Union[Dict[str, Any], SystemMetrics, PerformanceMetrics]]
    ) -> None:
        """Register a custom data source."""
        self.data_sources[name] = collector_func
        self.logger.info(f"Registered data source: {name}")
    
    def register_custom_collector(
        self,
        name: str,
        collector_func: Callable[[PerformanceMetrics], Dict[str, Any]]
    ) -> None:
        """Register a custom metrics collector."""
        self.custom_collectors[name] = collector_func
        self.logger.info(f"Registered custom collector: {name}")
    
    def add_metrics_callback(
        self,
        callback: Callable[[PerformanceMetrics], None]
    ) -> None:
        """Add a callback for real-time metrics processing."""
        self.metrics_callbacks.append(callback)
    
    async def collect_single_sample(
        self,
        source_name: Optional[str] = None
    ) -> Optional[PerformanceMetrics]:
        """Collect a single metrics sample."""
        try:
            start_time = time.time()
            
            # Collect from all sources or specific source
            sources_to_collect = (
                {source_name: self.data_sources[source_name]}
                if source_name and source_name in self.data_sources
                else self.data_sources
            )
            
            collected_data = {}
            base_metrics = None
            
            # Collect from each data source
            for name, collector_func in sources_to_collect.items():
                try:
                    if asyncio.iscoroutinefunction(collector_func):
                        result = await collector_func()
                    else:
                        # Run in thread pool for blocking operations
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, collector_func
                        )
                    
                    if isinstance(result, (SystemMetrics, PerformanceMetrics)):
                        if base_metrics is None:
                            if isinstance(result, SystemMetrics):
                                base_metrics = PerformanceMetrics.from_system_metrics(result)
                            else:
                                base_metrics = result
                    elif isinstance(result, dict):
                        collected_data.update(result)
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting from {name}: {e}")
                    self._record_collection_error(name, str(e))
            
            # Create comprehensive metrics
            if base_metrics is None:
                base_metrics = PerformanceMetrics()
            
            # Apply collected data
            for key, value in collected_data.items():
                if hasattr(base_metrics, key):
                    setattr(base_metrics, key, value)
            
            # Run custom collectors
            for name, collector_func in self.custom_collectors.items():
                try:
                    custom_data = await self._run_custom_collector(collector_func, base_metrics)
                    for key, value in custom_data.items():
                        if hasattr(base_metrics, key):
                            setattr(base_metrics, key, value)
                except Exception as e:
                    self.logger.warning(f"Error in custom collector {name}: {e}")
            
            # Update collection stats
            collection_duration = time.time() - start_time
            self._update_collection_stats(True, collection_duration)
            
            # Store metrics
            self.raw_metrics_buffer.append(base_metrics)
            await self.aggregator.add_metrics(base_metrics)
            
            # Trigger callbacks
            await self._trigger_metrics_callbacks(base_metrics)
            
            return base_metrics
            
        except Exception as e:
            self.logger.error(f"Error during metrics collection: {e}")
            self._update_collection_stats(False, 0.0)
            self._record_collection_error("general", str(e))
            return None
    
    async def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation_window: Optional[timedelta] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """Get historical metrics data."""
        try:
            # Default time range
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                start_time = end_time - timedelta(hours=24)
            
            # Get metrics from buffer
            metrics = [
                m for m in self.raw_metrics_buffer
                if start_time <= m.timestamp <= end_time
            ]
            
            # Apply aggregation if requested
            if aggregation_window:
                metrics = await self._aggregate_metrics_by_window(
                    metrics, aggregation_window
                )
            
            # Apply limit
            if limit:
                metrics = metrics[-limit:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error retrieving metrics history: {e}")
            return []
    
    async def get_collection_statistics(self) -> Dict[str, Any]:
        """Get collection statistics and health information."""
        return {
            'collection_stats': self.collection_stats.copy(),
            'data_sources': list(self.data_sources.keys()),
            'custom_collectors': list(self.custom_collectors.keys()),
            'buffer_size': len(self.raw_metrics_buffer),
            'buffer_capacity': self.raw_metrics_buffer.maxlen,
            'aggregation_windows': list(self.aggregator.aggregation_windows.keys()),
            'error_count': len(self.error_history),
            'is_collecting': self.is_collecting,
            'recent_errors': list(self.error_history)[-10:] if self.error_history else []
        }
    
    async def export_metrics(
        self,
        export_path: Path,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """Export metrics data to file."""
        try:
            # Get metrics data
            metrics = await self.get_metrics_history(start_time, end_time)
            
            # Prepare export data
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'start_time': start_time.isoformat() if start_time else None,
                    'end_time': end_time.isoformat() if end_time else None,
                    'record_count': len(metrics),
                    'format_version': "1.0"
                },
                'metrics': [asdict(m) for m in metrics]
            }
            
            # Export based on format
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            elif format.lower() == "csv":
                import pandas as pd
                
                # Convert to DataFrame
                df = pd.DataFrame([asdict(m) for m in metrics])
                df.to_csv(export_path, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(metrics)} metrics records to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            raise
    
    # Private methods
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.is_collecting:
            try:
                # Collect metrics sample
                await self.collect_single_sample()
                
                # Periodic data persistence
                if self.collection_stats['total_collections'] % 100 == 0:
                    await self._save_buffered_metrics()
                
                # Periodic cleanup
                if self.collection_stats['total_collections'] % 1000 == 0:
                    await self._cleanup_old_data()
                
                # Wait for next collection interval
                await asyncio.sleep(self.config.collection_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _register_builtin_sources(self) -> None:
        """Register built-in data sources."""
        # System metrics collector
        self.register_data_source("system", self._collect_system_metrics)
        
        # Process metrics collector
        self.register_data_source("process", self._collect_process_metrics)
        
        # Network metrics collector
        self.register_data_source("network", self._collect_network_metrics)
        
        # Application metrics collector
        self.register_data_source("application", self._collect_application_metrics)
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'memory_available': memory.available,
                'swap_percent': swap.percent,
                'disk_percent': disk_usage.percent,
                'disk_used': disk_usage.used,
                'disk_free': disk_usage.free,
                'disk_io_bytes': (disk_io.read_bytes + disk_io.write_bytes) if disk_io else 0,
                'network_io_bytes': (network_io.bytes_sent + network_io.bytes_recv) if network_io else 0
            }
            
        except ImportError:
            self.logger.warning("psutil not available, using basic system metrics")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_percent': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def _collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process-level metrics."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Process CPU and memory
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Process I/O
            io_counters = process.io_counters()
            
            # Open files and connections
            open_files = len(process.open_files())
            connections = len(process.connections())
            
            return {
                'process_cpu_percent': cpu_percent,
                'process_memory_rss': memory_info.rss,
                'process_memory_vms': memory_info.vms,
                'process_memory_percent': memory_percent,
                'process_read_bytes': io_counters.read_bytes if io_counters else 0,
                'process_write_bytes': io_counters.write_bytes if io_counters else 0,
                'process_open_files': open_files,
                'process_connections': connections
            }
            
        except Exception as e:
            self.logger.warning(f"Error collecting process metrics: {e}")
            return {}
    
    async def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network-related metrics."""
        try:
            import psutil
            
            network_io = psutil.net_io_counters()
            network_connections = len(psutil.net_connections())
            
            return {
                'network_bytes_sent': network_io.bytes_sent if network_io else 0,
                'network_bytes_recv': network_io.bytes_recv if network_io else 0,
                'network_packets_sent': network_io.packets_sent if network_io else 0,
                'network_packets_recv': network_io.packets_recv if network_io else 0,
                'network_connections': network_connections
            }
            
        except Exception as e:
            self.logger.warning(f"Error collecting network metrics: {e}")
            return {}
    
    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        # This would integrate with the application's internal metrics
        # For now, return basic placeholder metrics
        return {
            'active_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'queue_size': 0,
            'cache_hit_rate': 0.95,
            'response_time': 100.0
        }
    
    async def _run_custom_collector(
        self,
        collector_func: Callable,
        base_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Run a custom collector function."""
        if asyncio.iscoroutinefunction(collector_func):
            return await collector_func(base_metrics)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, collector_func, base_metrics
            )
    
    async def _trigger_metrics_callbacks(self, metrics: PerformanceMetrics) -> None:
        """Trigger all registered metrics callbacks."""
        for callback in self.metrics_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool, callback, metrics
                    )
            except Exception as e:
                self.logger.warning(f"Error in metrics callback: {e}")
    
    def _update_collection_stats(self, success: bool, duration: float) -> None:
        """Update collection statistics."""
        self.collection_stats['total_collections'] += 1
        self.collection_stats['last_collection_time'] = datetime.utcnow()
        
        if success:
            self.collection_stats['successful_collections'] += 1
        else:
            self.collection_stats['failed_collections'] += 1
        
        # Update average duration
        total_successful = self.collection_stats['successful_collections']
        if total_successful > 1:
            current_avg = self.collection_stats['average_collection_duration']
            self.collection_stats['average_collection_duration'] = (
                (current_avg * (total_successful - 1) + duration) / total_successful
            )
        else:
            self.collection_stats['average_collection_duration'] = duration
    
    def _record_collection_error(self, source: str, error: str) -> None:
        """Record a collection error."""
        error_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'source': source,
            'error': error
        }
        self.error_history.append(error_record)
    
    async def _aggregate_metrics_by_window(
        self,
        metrics: List[PerformanceMetrics],
        window: timedelta
    ) -> List[PerformanceMetrics]:
        """Aggregate metrics by time window."""
        if not metrics:
            return []
        
        aggregated = []
        window_seconds = window.total_seconds()
        
        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Group by time windows
        current_window_start = sorted_metrics[0].timestamp
        current_window_metrics = []
        
        for metric in sorted_metrics:
            if (metric.timestamp - current_window_start).total_seconds() < window_seconds:
                current_window_metrics.append(metric)
            else:
                # Aggregate current window
                if current_window_metrics:
                    aggregated_metric = await self.aggregator.aggregate_metrics(
                        current_window_metrics, window_seconds
                    )
                    aggregated.append(aggregated_metric)
                
                # Start new window
                current_window_start = metric.timestamp
                current_window_metrics = [metric]
        
        # Handle final window
        if current_window_metrics:
            aggregated_metric = await self.aggregator.aggregate_metrics(
                current_window_metrics, window_seconds
            )
            aggregated.append(aggregated_metric)
        
        return aggregated
    
    async def _save_buffered_metrics(self) -> None:
        """Save buffered metrics to storage."""
        if not self.raw_metrics_buffer:
            return
        
        try:
            # Create timestamped file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_path = self.storage_path / "raw" / f"metrics_{timestamp}.json"
            
            # Prepare data for saving
            metrics_data = [asdict(m) for m in list(self.raw_metrics_buffer)]
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.utcnow().isoformat(),
                    'count': len(metrics_data),
                    'metrics': metrics_data
                }, f, indent=2, default=str)
            
            self.logger.debug(f"Saved {len(metrics_data)} metrics to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving buffered metrics: {e}")
    
    async def _load_historical_metrics(self) -> None:
        """Load historical metrics from storage."""
        try:
            raw_path = self.storage_path / "raw"
            if not raw_path.exists():
                return
            
            # Load recent files (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            for file_path in raw_path.glob("metrics_*.json"):
                try:
                    # Parse timestamp from filename
                    timestamp_str = file_path.stem.replace("metrics_", "")
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if file_time >= cutoff_time:
                        # Load metrics from file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        for metric_data in data.get('metrics', []):
                            # Convert back to PerformanceMetrics
                            metric = PerformanceMetrics(**metric_data)
                            self.raw_metrics_buffer.append(metric)
                
                except Exception as e:
                    self.logger.warning(f"Error loading metrics from {file_path}: {e}")
            
            self.logger.info(f"Loaded {len(self.raw_metrics_buffer)} historical metrics")
            
        except Exception as e:
            self.logger.error(f"Error loading historical metrics: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data files."""
        try:
            cutoff_time = datetime.utcnow() - self.config.retention_period
            
            for storage_type in ["raw", "aggregated"]:
                storage_dir = self.storage_path / storage_type
                if not storage_dir.exists():
                    continue
                
                for file_path in storage_dir.glob("*.json"):
                    try:
                        file_stat = file_path.stat()
                        file_time = datetime.fromtimestamp(file_stat.st_mtime)
                        
                        if file_time < cutoff_time:
                            # Archive old file
                            archive_path = self.storage_path / "archives" / file_path.name
                            archive_path.parent.mkdir(exist_ok=True)
                            file_path.rename(archive_path)
                            
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up {file_path}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


class StreamingMetricsCollector:
    """
    Streaming metrics collector for real-time processing.
    
    Optimized for high-frequency data collection with minimal overhead.
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_stream = asyncio.Queue(maxsize=buffer_size)
        self.processors: List[Callable] = []
        
        self.is_streaming = False
        self.stream_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_processor(self, processor: Callable[[PerformanceMetrics], None]) -> None:
        """Add a real-time metrics processor."""
        self.processors.append(processor)
    
    async def start_streaming(self) -> None:
        """Start streaming metrics processing."""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stream_task = asyncio.create_task(self._stream_processing_loop())
    
    async def stop_streaming(self) -> None:
        """Stop streaming metrics processing."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
    
    async def push_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Push metrics to the streaming queue."""
        try:
            await self.metrics_stream.put(metrics)
            return True
        except asyncio.QueueFull:
            self.logger.warning("Metrics stream queue is full, dropping metrics")
            return False
    
    async def _stream_processing_loop(self) -> None:
        """Process metrics stream in real-time."""
        while self.is_streaming:
            try:
                # Get metrics from stream (with timeout to allow checking is_streaming)
                try:
                    metrics = await asyncio.wait_for(
                        self.metrics_stream.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process with all registered processors
                for processor in self.processors:
                    try:
                        if asyncio.iscoroutinefunction(processor):
                            await processor(metrics)
                        else:
                            processor(metrics)
                    except Exception as e:
                        self.logger.warning(f"Error in stream processor: {e}")
                
                # Mark task as done
                self.metrics_stream.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in stream processing loop: {e}")
                await asyncio.sleep(0.1)