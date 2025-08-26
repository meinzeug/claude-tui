"""
Memory Leak Detection System

Advanced memory leak detection with pattern analysis, heap profiling,
and automated leak prevention mechanisms.
"""

import asyncio
import gc
import logging
import time
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import psutil
import threading
from collections import defaultdict, deque
import weakref
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    process_memory: Dict[str, int]
    heap_size: int
    object_counts: Dict[str, int]
    gc_stats: Dict[str, Any]
    tracemalloc_stats: Optional[Dict[str, Any]] = None

@dataclass
class LeakSuspect:
    """Suspected memory leak information"""
    object_type: str
    count_growth: int
    memory_growth: int
    growth_rate: float
    first_detected: float
    confidence_score: float
    stack_traces: List[str] = None

@dataclass
class MemoryAlert:
    """Memory-related alert"""
    alert_type: str
    severity: str
    message: str
    timestamp: float
    memory_usage: float
    threshold: float
    recommendation: str

class ObjectTracker:
    """Tracks object creation and destruction patterns"""
    
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.object_references = defaultdict(lambda: weakref.WeakSet())
        self.creation_times = defaultdict(list)
        self.destruction_times = defaultdict(list)
        self.tracking_enabled = False
        
    def start_tracking(self):
        """Start tracking object lifecycle"""
        self.tracking_enabled = True
        logger.info("Object lifecycle tracking started")
    
    def stop_tracking(self):
        """Stop tracking object lifecycle"""
        self.tracking_enabled = False
        logger.info("Object lifecycle tracking stopped")
    
    def track_object_creation(self, obj: Any):
        """Track object creation"""
        if not self.tracking_enabled:
            return
            
        obj_type = type(obj).__name__
        self.object_counts[obj_type] += 1
        
        try:
            self.object_references[obj_type].add(obj)
        except TypeError:
            # Object is not weakly referenceable
            pass
        
        self.creation_times[obj_type].append(time.time())
    
    def track_object_destruction(self, obj_type: str):
        """Track object destruction"""
        if not self.tracking_enabled:
            return
            
        if self.object_counts[obj_type] > 0:
            self.object_counts[obj_type] -= 1
        
        self.destruction_times[obj_type].append(time.time())
    
    def get_object_statistics(self) -> Dict[str, Any]:
        """Get object lifecycle statistics"""
        stats = {}
        
        for obj_type in self.object_counts:
            active_count = len(self.object_references[obj_type])
            total_created = len(self.creation_times[obj_type])
            total_destroyed = len(self.destruction_times[obj_type])
            
            stats[obj_type] = {
                'active_count': active_count,
                'total_created': total_created,
                'total_destroyed': total_destroyed,
                'net_growth': total_created - total_destroyed,
                'creation_rate': self._calculate_rate(self.creation_times[obj_type]),
                'destruction_rate': self._calculate_rate(self.destruction_times[obj_type])
            }
        
        return stats
    
    def _calculate_rate(self, timestamps: List[float], window_seconds: int = 300) -> float:
        """Calculate rate of events per second over time window"""
        if not timestamps:
            return 0.0
        
        current_time = time.time()
        recent_timestamps = [t for t in timestamps if current_time - t <= window_seconds]
        
        return len(recent_timestamps) / window_seconds if recent_timestamps else 0.0

class HeapProfiler:
    """Advanced heap profiling and analysis"""
    
    def __init__(self):
        self.profiling_enabled = False
        self.snapshots = deque(maxlen=100)
        self.baseline_snapshot = None
        
    def start_profiling(self):
        """Start heap profiling"""
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Keep top 10 stack frames
        
        self.profiling_enabled = True
        self.baseline_snapshot = self._take_heap_snapshot()
        logger.info("Heap profiling started")
    
    def stop_profiling(self):
        """Stop heap profiling"""
        self.profiling_enabled = False
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        logger.info("Heap profiling stopped")
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take memory snapshot"""
        if not self.profiling_enabled:
            return None
        
        snapshot = self._take_heap_snapshot()
        self.snapshots.append(snapshot)
        
        return snapshot
    
    def _take_heap_snapshot(self) -> MemorySnapshot:
        """Take detailed heap snapshot"""
        timestamp = time.time()
        
        # Process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        process_memory = {
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available
        }
        
        # Heap size estimation
        heap_size = sys.getsizeof(gc.get_objects())
        
        # Object counts by type
        object_counts = defaultdict(int)
        for obj in gc.get_objects():
            object_counts[type(obj).__name__] += 1
        
        # GC statistics
        gc_stats = {
            'collections': gc.get_stats(),
            'threshold': gc.get_threshold(),
            'counts': gc.get_count()
        }
        
        # Tracemalloc statistics
        tracemalloc_stats = None
        if tracemalloc.is_tracing():
            current_trace = tracemalloc.take_snapshot()
            top_stats = current_trace.statistics('lineno')[:10]
            
            tracemalloc_stats = {
                'total_size': sum(stat.size for stat in top_stats),
                'total_count': sum(stat.count for stat in top_stats),
                'top_allocations': [
                    {
                        'filename': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                        'size': stat.size,
                        'count': stat.count
                    }
                    for stat in top_stats
                ]
            }
        
        return MemorySnapshot(
            timestamp=timestamp,
            process_memory=process_memory,
            heap_size=heap_size,
            object_counts=dict(object_counts),
            gc_stats=gc_stats,
            tracemalloc_stats=tracemalloc_stats
        )
    
    def analyze_heap_growth(self, window_minutes: int = 30) -> List[LeakSuspect]:
        """Analyze heap growth patterns to detect leaks"""
        if len(self.snapshots) < 2:
            return []
        
        # Get snapshots within window
        cutoff_time = time.time() - (window_minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return recent_snapshots
        
        # Analyze growth patterns
        suspects = []
        baseline = recent_snapshots[0]
        latest = recent_snapshots[-1]
        
        time_delta = latest.timestamp - baseline.timestamp
        
        # Check object count growth
        for obj_type in latest.object_counts:
            baseline_count = baseline.object_counts.get(obj_type, 0)
            latest_count = latest.object_counts[obj_type]
            count_growth = latest_count - baseline_count
            
            if count_growth > 100:  # Significant growth threshold
                growth_rate = count_growth / time_delta if time_delta > 0 else 0
                
                # Estimate memory impact
                avg_obj_size = 64  # Rough estimate
                memory_growth = count_growth * avg_obj_size
                
                # Calculate confidence score
                confidence = min(1.0, count_growth / 1000.0)  # Higher growth = higher confidence
                
                suspect = LeakSuspect(
                    object_type=obj_type,
                    count_growth=count_growth,
                    memory_growth=memory_growth,
                    growth_rate=growth_rate,
                    first_detected=baseline.timestamp,
                    confidence_score=confidence
                )
                
                suspects.append(suspect)
        
        # Sort by confidence score
        return sorted(suspects, key=lambda s: s.confidence_score, reverse=True)

class MemoryLeakDetector:
    """Main memory leak detection system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Components
        self.object_tracker = ObjectTracker()
        self.heap_profiler = HeapProfiler()
        
        # State
        self.detection_active = False
        self.memory_alerts = []
        self.leak_history = deque(maxlen=1000)
        
        # Thresholds
        self.memory_growth_threshold = self.config.get('memory_growth_threshold_mb', 50)
        self.object_growth_threshold = self.config.get('object_growth_threshold', 1000)
        self.detection_interval = self.config.get('detection_interval', 60)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'memory_growth_threshold_mb': 50,
            'object_growth_threshold': 1000,
            'detection_interval': 60,
            'enable_object_tracking': True,
            'enable_heap_profiling': True,
            'alert_on_growth': True,
            'auto_gc_on_leak': True,
            'max_snapshots': 100,
            'analysis_window_minutes': 30
        }
    
    async def start_detection(self):
        """Start memory leak detection"""
        if self.detection_active:
            logger.warning("Memory leak detection already active")
            return
        
        self.detection_active = True
        logger.info("Starting memory leak detection")
        
        # Start components
        if self.config.get('enable_object_tracking', True):
            self.object_tracker.start_tracking()
        
        if self.config.get('enable_heap_profiling', True):
            self.heap_profiler.start_profiling()
        
        # Start detection loop
        detection_task = asyncio.create_task(self._detection_loop())
        
        try:
            await detection_task
        except Exception as e:
            logger.error(f"Memory leak detection failed: {e}")
        finally:
            await self.stop_detection()
    
    async def stop_detection(self):
        """Stop memory leak detection"""
        logger.info("Stopping memory leak detection")
        self.detection_active = False
        
        # Stop components
        self.object_tracker.stop_tracking()
        self.heap_profiler.stop_profiling()
    
    async def _detection_loop(self):
        """Main detection loop"""
        while self.detection_active:
            try:
                # Take memory snapshot
                snapshot = self.heap_profiler.take_snapshot()
                
                if snapshot:
                    # Analyze for leaks
                    suspects = await self._analyze_for_leaks(snapshot)
                    
                    # Check memory thresholds
                    alerts = self._check_memory_thresholds(snapshot)
                    
                    # Process findings
                    if suspects:
                        await self._handle_leak_suspects(suspects)
                    
                    if alerts:
                        await self._handle_memory_alerts(alerts)
                
                # Trigger garbage collection if configured
                if self.config.get('auto_gc_on_leak', False) and (suspects or alerts):
                    collected = gc.collect()
                    logger.info(f"Triggered garbage collection, collected {collected} objects")
                
            except Exception as e:
                logger.error(f"Detection loop iteration failed: {e}")
            
            await asyncio.sleep(self.detection_interval)
    
    async def _analyze_for_leaks(self, snapshot: MemorySnapshot) -> List[LeakSuspect]:
        """Analyze snapshot for potential memory leaks"""
        # Heap growth analysis
        heap_suspects = self.heap_profiler.analyze_heap_growth(
            self.config.get('analysis_window_minutes', 30)
        )
        
        # Object lifecycle analysis
        object_stats = self.object_tracker.get_object_statistics()
        object_suspects = []
        
        for obj_type, stats in object_stats.items():
            if stats['net_growth'] > self.object_growth_threshold:
                suspect = LeakSuspect(
                    object_type=obj_type,
                    count_growth=stats['net_growth'],
                    memory_growth=stats['net_growth'] * 64,  # Estimate
                    growth_rate=stats['creation_rate'] - stats['destruction_rate'],
                    first_detected=time.time() - 300,  # Estimate
                    confidence_score=min(1.0, stats['net_growth'] / (self.object_growth_threshold * 2))
                )
                object_suspects.append(suspect)
        
        # Combine and deduplicate suspects
        all_suspects = heap_suspects + object_suspects
        unique_suspects = {}
        
        for suspect in all_suspects:
            if suspect.object_type not in unique_suspects:
                unique_suspects[suspect.object_type] = suspect
            else:
                # Keep the one with higher confidence
                existing = unique_suspects[suspect.object_type]
                if suspect.confidence_score > existing.confidence_score:
                    unique_suspects[suspect.object_type] = suspect
        
        return list(unique_suspects.values())
    
    def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> List[MemoryAlert]:
        """Check memory usage against thresholds"""
        alerts = []
        
        # Check RSS memory growth
        current_rss_mb = snapshot.process_memory['rss'] / (1024 * 1024)
        memory_percent = snapshot.process_memory['percent']
        
        # High memory usage alert
        if memory_percent > 85:
            alerts.append(MemoryAlert(
                alert_type='high_memory_usage',
                severity='critical' if memory_percent > 95 else 'warning',
                message=f"High memory usage: {memory_percent:.1f}%",
                timestamp=snapshot.timestamp,
                memory_usage=memory_percent,
                threshold=85.0,
                recommendation="Investigate memory usage patterns and consider scaling"
            ))
        
        # Memory growth rate alert
        if len(self.heap_profiler.snapshots) >= 2:
            prev_snapshot = self.heap_profiler.snapshots[-2]
            growth_mb = (current_rss_mb - prev_snapshot.process_memory['rss'] / (1024 * 1024))
            
            if growth_mb > self.memory_growth_threshold:
                alerts.append(MemoryAlert(
                    alert_type='memory_growth',
                    severity='warning',
                    message=f"High memory growth: {growth_mb:.1f}MB",
                    timestamp=snapshot.timestamp,
                    memory_usage=current_rss_mb,
                    threshold=self.memory_growth_threshold,
                    recommendation="Check for memory leaks and optimize memory usage"
                ))
        
        # Available memory alert
        available_gb = snapshot.process_memory['available'] / (1024 * 1024 * 1024)
        if available_gb < 1.0:  # Less than 1GB available
            alerts.append(MemoryAlert(
                alert_type='low_available_memory',
                severity='critical',
                message=f"Low available memory: {available_gb:.2f}GB",
                timestamp=snapshot.timestamp,
                memory_usage=available_gb,
                threshold=1.0,
                recommendation="Free memory immediately or add more system memory"
            ))
        
        return alerts
    
    async def _handle_leak_suspects(self, suspects: List[LeakSuspect]):
        """Handle detected leak suspects"""
        for suspect in suspects:
            logger.warning(f"Memory leak suspect detected: {suspect.object_type}")
            logger.warning(f"  Growth: {suspect.count_growth} objects ({suspect.memory_growth} bytes)")
            logger.warning(f"  Rate: {suspect.growth_rate:.2f} objects/s")
            logger.warning(f"  Confidence: {suspect.confidence_score:.2f}")
            
            # Store in history
            self.leak_history.append({
                'timestamp': time.time(),
                'suspect': asdict(suspect)
            })
            
            # Generate recommendations
            recommendations = self._generate_leak_recommendations(suspect)
            for rec in recommendations:
                logger.info(f"  Recommendation: {rec}")
    
    async def _handle_memory_alerts(self, alerts: List[MemoryAlert]):
        """Handle memory alerts"""
        for alert in alerts:
            logger.warning(f"Memory alert: {alert.message}")
            logger.warning(f"  Severity: {alert.severity}")
            logger.warning(f"  Recommendation: {alert.recommendation}")
            
            self.memory_alerts.append(alert)
            
            # Trigger emergency actions for critical alerts
            if alert.severity == 'critical':
                await self._handle_critical_memory_alert(alert)
    
    async def _handle_critical_memory_alert(self, alert: MemoryAlert):
        """Handle critical memory alerts with emergency actions"""
        logger.critical(f"CRITICAL MEMORY ALERT: {alert.message}")
        
        if alert.alert_type == 'high_memory_usage':
            # Force garbage collection
            before_gc = psutil.Process().memory_info().rss
            collected = gc.collect()
            after_gc = psutil.Process().memory_info().rss
            freed_mb = (before_gc - after_gc) / (1024 * 1024)
            
            logger.info(f"Emergency GC freed {freed_mb:.1f}MB, collected {collected} objects")
            
            # Clear caches if still critical
            if after_gc / before_gc > 0.95:  # Less than 5% freed
                logger.warning("GC had minimal effect, clearing caches")
                await self._clear_caches()
        
        elif alert.alert_type == 'low_available_memory':
            # Emergency memory cleanup
            await self._emergency_memory_cleanup()
    
    async def _clear_caches(self):
        """Clear application caches to free memory"""
        # This would be implemented based on your application's cache systems
        logger.info("Clearing application caches (placeholder)")
        
        # Example cache clearing operations:
        # - Clear Redis cache
        # - Clear in-memory caches
        # - Clear temporary files
        # - Reduce connection pools
    
    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures"""
        logger.critical("Executing emergency memory cleanup")
        
        # Force multiple GC cycles
        for i in range(3):
            collected = gc.collect()
            logger.info(f"GC cycle {i+1}: collected {collected} objects")
        
        # Clear weak references
        gc.collect()
        
        # Log memory usage after cleanup
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory after cleanup: RSS={memory_info.rss/(1024*1024):.1f}MB")
    
    def _generate_leak_recommendations(self, suspect: LeakSuspect) -> List[str]:
        """Generate recommendations for suspected memory leaks"""
        recommendations = []
        
        # Object-specific recommendations
        if suspect.object_type in ['dict', 'list', 'tuple']:
            recommendations.append("Review data structure usage, consider using generators or iterators")
            recommendations.append("Check for circular references in nested structures")
        
        elif suspect.object_type in ['function', 'method']:
            recommendations.append("Check for closure leaks, especially in event handlers")
            recommendations.append("Review lambda functions and decorators")
        
        elif 'Connection' in suspect.object_type:
            recommendations.append("Ensure database/network connections are properly closed")
            recommendations.append("Review connection pool configuration")
        
        elif 'Thread' in suspect.object_type or 'Process' in suspect.object_type:
            recommendations.append("Check for proper thread/process cleanup")
            recommendations.append("Review concurrent task management")
        
        # General recommendations based on growth rate
        if suspect.growth_rate > 10:  # High growth rate
            recommendations.append("High growth rate detected - investigate recent code changes")
            recommendations.append("Consider implementing object pooling")
        
        if suspect.confidence_score > 0.8:  # High confidence leak
            recommendations.append("High confidence leak - immediate investigation recommended")
            recommendations.append("Enable detailed stack trace collection for this object type")
        
        return recommendations
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get detection summary and statistics"""
        current_snapshot = None
        if self.heap_profiler.snapshots:
            current_snapshot = self.heap_profiler.snapshots[-1]
        
        return {
            'detection_active': self.detection_active,
            'total_snapshots': len(self.heap_profiler.snapshots),
            'total_alerts': len(self.memory_alerts),
            'total_suspects': len(self.leak_history),
            'current_memory_mb': current_snapshot.process_memory['rss'] / (1024 * 1024) if current_snapshot else 0,
            'current_memory_percent': current_snapshot.process_memory['percent'] if current_snapshot else 0,
            'recent_suspects': [entry['suspect']['object_type'] for entry in list(self.leak_history)[-5:]],
            'config': self.config
        }
    
    def export_detection_report(self, filepath: str):
        """Export detection report to file"""
        report_data = {
            'summary': self.get_detection_summary(),
            'snapshots': [asdict(s) for s in list(self.heap_profiler.snapshots)[-10:]],
            'alerts': [asdict(a) for a in self.memory_alerts[-20:]],
            'leak_history': list(self.leak_history)[-50:],
            'object_statistics': self.object_tracker.get_object_statistics(),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Memory leak detection report exported to {filepath}")

# Example usage and testing
async def main():
    """Example usage of memory leak detector"""
    detector = MemoryLeakDetector({
        'detection_interval': 5,  # Fast interval for demo
        'memory_growth_threshold_mb': 10
    })
    
    try:
        # Start detection
        detection_task = asyncio.create_task(detector.start_detection())
        
        # Simulate memory usage
        memory_hog = []
        for i in range(1000):
            # Simulate memory leak
            memory_hog.append([0] * 1000)  # Add 1000 integers
            
            if i % 100 == 0:
                await asyncio.sleep(1)
        
        # Let detector run for a bit
        await asyncio.sleep(10)
        
        # Stop detection
        await detector.stop_detection()
        detection_task.cancel()
        
        # Export report
        detector.export_detection_report("memory_leak_report.json")
        
        # Print summary
        summary = detector.get_detection_summary()
        print(f"Detection completed:")
        print(f"  Snapshots: {summary['total_snapshots']}")
        print(f"  Alerts: {summary['total_alerts']}")
        print(f"  Suspects: {summary['total_suspects']}")
        print(f"  Current Memory: {summary['current_memory_mb']:.1f}MB")
        
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        await detector.stop_detection()

if __name__ == "__main__":
    asyncio.run(main())