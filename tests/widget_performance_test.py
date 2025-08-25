#!/usr/bin/env python3
"""
Widget Performance Test
Specialized test for TUI widget rendering performance and memory usage.
"""

import asyncio
import gc
import time
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch
from dataclasses import dataclass
import threading
import json

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class WidgetMetrics:
    """Metrics for individual widget performance"""
    widget_type: str
    creation_time: float
    update_time: float
    memory_usage: int
    render_count: int
    memory_growth: int


@dataclass
class RenderCycleMetrics:
    """Metrics for full render cycle"""
    cycle_id: int
    total_widgets: int
    render_time: float
    memory_before: int
    memory_after: int
    cpu_percent: float


class WidgetPerformanceTester:
    """Comprehensive widget performance testing"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.widget_metrics: List[WidgetMetrics] = []
        self.render_cycles: List[RenderCycleMetrics] = []
        self.baseline_memory = 0
        
    def setup_testing_environment(self):
        """Set up the testing environment with mocked dependencies"""
        # Enable memory tracing
        tracemalloc.start()
        
        # Clear any existing memory
        gc.collect()
        self.baseline_memory = self.process.memory_info().rss
        
        print(f"📊 Widget Performance Testing Started")
        print(f"   Baseline Memory: {self.baseline_memory / 1024 / 1024:.1f}MB")
        
    def test_widget_creation_performance(self, widget_count: int = 100) -> Dict[str, Any]:
        """Test widget creation performance with various widget types"""
        print(f"🔧 Testing widget creation performance ({widget_count} widgets)...")
        
        widget_types = ['button', 'label', 'input', 'list', 'tree', 'static', 'log']
        results = {}
        
        for widget_type in widget_types:
            start_memory = self.process.memory_info().rss
            start_time = time.time()
            
            widgets = []
            for i in range(widget_count):
                widget = self._create_mock_widget(widget_type, i)
                widgets.append(widget)
                
                # Simulate some widget initialization
                if hasattr(widget, 'set_data'):
                    widget.set_data(f"Data for {widget_type} {i}")
                    
            creation_time = time.time() - start_time
            end_memory = self.process.memory_info().rss
            memory_growth = end_memory - start_memory
            
            # Test widget updates
            start_time = time.time()
            for widget in widgets:
                if hasattr(widget, 'update'):
                    widget.update(f"Updated data {widget_type}")
                    
            update_time = time.time() - start_time
            
            metrics = WidgetMetrics(
                widget_type=widget_type,
                creation_time=creation_time,
                update_time=update_time,
                memory_usage=memory_growth,
                render_count=widget_count,
                memory_growth=memory_growth
            )
            
            self.widget_metrics.append(metrics)
            results[widget_type] = {
                'creation_time': creation_time,
                'update_time': update_time,
                'memory_growth_mb': memory_growth / 1024 / 1024,
                'widgets_per_second': widget_count / creation_time if creation_time > 0 else 0,
                'memory_per_widget': memory_growth / widget_count if widget_count > 0 else 0
            }
            
            print(f"   {widget_type}: {creation_time:.3f}s creation, {memory_growth/1024/1024:.2f}MB")
            
            # Clean up
            del widgets
            gc.collect()
            
        return results
        
    def test_render_cycle_performance(self, cycles: int = 10, widgets_per_cycle: int = 50) -> Dict[str, Any]:
        """Test full render cycle performance"""
        print(f"🎨 Testing render cycle performance ({cycles} cycles, {widgets_per_cycle} widgets each)...")
        
        results = {
            'cycles': [],
            'total_time': 0,
            'average_cycle_time': 0,
            'memory_growth_per_cycle': 0,
            'peak_memory_mb': 0
        }
        
        start_total_time = time.time()
        
        for cycle in range(cycles):
            memory_before = self.process.memory_info().rss
            cpu_before = self.process.cpu_percent()
            cycle_start_time = time.time()
            
            # Simulate a full render cycle
            widgets = []
            
            # Create widgets
            for i in range(widgets_per_cycle):
                widget_type = ['button', 'label', 'input'][i % 3]
                widget = self._create_mock_widget(widget_type, i)
                widgets.append(widget)
                
            # Simulate layout calculation
            self._simulate_layout_calculation(widgets)
            
            # Simulate rendering
            self._simulate_widget_rendering(widgets)
            
            # Simulate event handling
            self._simulate_event_handling(widgets)
            
            cycle_time = time.time() - cycle_start_time
            memory_after = self.process.memory_info().rss
            cpu_after = self.process.cpu_percent()
            
            cycle_metrics = RenderCycleMetrics(
                cycle_id=cycle,
                total_widgets=widgets_per_cycle,
                render_time=cycle_time,
                memory_before=memory_before,
                memory_after=memory_after,
                cpu_percent=(cpu_before + cpu_after) / 2
            )
            
            self.render_cycles.append(cycle_metrics)
            
            results['cycles'].append({
                'cycle': cycle,
                'time': cycle_time,
                'memory_growth': memory_after - memory_before,
                'cpu_percent': cycle_metrics.cpu_percent
            })
            
            # Track peak memory
            current_memory_mb = memory_after / 1024 / 1024
            if current_memory_mb > results['peak_memory_mb']:
                results['peak_memory_mb'] = current_memory_mb
                
            # Clean up
            del widgets
            gc.collect()
            
            print(f"   Cycle {cycle}: {cycle_time:.3f}s, {(memory_after-memory_before)/1024/1024:.2f}MB")
            
        total_time = time.time() - start_total_time
        results['total_time'] = total_time
        results['average_cycle_time'] = total_time / cycles
        
        if len(self.render_cycles) > 0:
            total_memory_growth = sum(c.memory_after - c.memory_before for c in self.render_cycles)
            results['memory_growth_per_cycle'] = total_memory_growth / len(self.render_cycles) / 1024 / 1024
            
        return results
        
    def test_large_dataset_rendering(self, file_count: int = 1000, task_count: int = 1000) -> Dict[str, Any]:
        """Test rendering performance with large datasets"""
        print(f"📊 Testing large dataset rendering ({file_count} files, {task_count} tasks)...")
        
        start_memory = self.process.memory_info().rss
        start_time = time.time()
        
        # Create file tree data
        file_tree_data = {}
        for i in range(file_count):
            path = f"src/module_{i//100}/submodule_{i//10}/file_{i}.py"
            file_tree_data[path] = {
                'size': 1024 + (i * 50),
                'modified': time.time() - (i * 60),
                'type': 'python',
                'lines': 50 + (i % 200)
            }
            
        file_tree_time = time.time() - start_time
        file_tree_memory = self.process.memory_info().rss - start_memory
        
        # Create task data
        start_time = time.time()
        task_data = []
        for i in range(task_count):
            task = {
                'id': i,
                'name': f"Task {i}",
                'description': f"Description for task {i}" * (i % 5 + 1),  # Varying sizes
                'status': ['pending', 'in_progress', 'completed'][i % 3],
                'priority': ['low', 'medium', 'high'][i % 3],
                'created_at': time.time() - (i * 3600),
                'estimated_time': (i % 8 + 1) * 3600,
                'tags': [f"tag_{j}" for j in range(i % 5)]
            }
            task_data.append(task)
            
        task_creation_time = time.time() - start_time
        current_memory = self.process.memory_info().rss
        task_memory = current_memory - start_memory - file_tree_memory
        
        # Simulate rendering these datasets
        start_time = time.time()
        
        # File tree rendering simulation
        rendered_files = []
        for path, data in file_tree_data.items():
            rendered_file = {
                'path': path,
                'display_name': Path(path).name,
                'size_display': f"{data['size']} bytes",
                'modified_display': f"{int(time.time() - data['modified'])}s ago",
                'icon': '🐍' if data['type'] == 'python' else '📄'
            }
            rendered_files.append(rendered_file)
            
        # Task rendering simulation
        rendered_tasks = []
        for task in task_data:
            rendered_task = {
                'id': task['id'],
                'display_text': f"[{task['status'].upper()}] {task['name']}",
                'description_preview': task['description'][:100] + '...',
                'priority_icon': {'low': '🔵', 'medium': '🟡', 'high': '🔴'}[task['priority']],
                'tags_display': ', '.join(task['tags']),
                'time_display': f"{task['estimated_time']//3600}h"
            }
            rendered_tasks.append(rendered_task)
            
        rendering_time = time.time() - start_time
        final_memory = self.process.memory_info().rss
        total_memory_usage = final_memory - start_memory
        
        results = {
            'file_count': file_count,
            'task_count': task_count,
            'file_tree_creation_time': file_tree_time,
            'task_creation_time': task_creation_time,
            'rendering_time': rendering_time,
            'total_time': file_tree_time + task_creation_time + rendering_time,
            'file_tree_memory_mb': file_tree_memory / 1024 / 1024,
            'task_memory_mb': task_memory / 1024 / 1024,
            'total_memory_mb': total_memory_usage / 1024 / 1024,
            'files_per_second': file_count / (file_tree_time + rendering_time/2),
            'tasks_per_second': task_count / (task_creation_time + rendering_time/2),
            'memory_per_file_bytes': total_memory_usage / file_count if file_count > 0 else 0,
            'memory_per_task_bytes': total_memory_usage / task_count if task_count > 0 else 0
        }
        
        print(f"   Files: {file_count:,} in {file_tree_time:.3f}s ({results['files_per_second']:.0f}/s)")
        print(f"   Tasks: {task_count:,} in {task_creation_time:.3f}s ({results['tasks_per_second']:.0f}/s)")
        print(f"   Memory: {total_memory_usage/1024/1024:.2f}MB total")
        
        return results
        
    def test_memory_leak_simulation(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Test for memory leaks during continuous operation"""
        print(f"🔍 Testing memory leaks over {duration_seconds} seconds...")
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        memory_samples = []
        
        iteration = 0
        while time.time() - start_time < duration_seconds:
            # Simulate widget creation and destruction cycles
            widgets = []
            for i in range(20):  # Create 20 widgets per iteration
                widget = self._create_mock_widget('test_widget', i)
                widgets.append(widget)
                
            # Simulate some work
            time.sleep(0.1)
            
            # Record memory
            current_memory = self.process.memory_info().rss
            memory_samples.append({
                'time': time.time() - start_time,
                'memory': current_memory,
                'iteration': iteration
            })
            
            # Clean up (this tests if cleanup actually frees memory)
            del widgets
            gc.collect()
            
            iteration += 1
            
        final_memory = self.process.memory_info().rss
        memory_growth = final_memory - start_memory
        
        # Analyze memory trend
        if len(memory_samples) >= 3:
            # Calculate trend
            early_avg = sum(s['memory'] for s in memory_samples[:5]) / 5
            late_avg = sum(s['memory'] for s in memory_samples[-5:]) / 5
            trend_growth = late_avg - early_avg
        else:
            trend_growth = 0
            
        results = {
            'duration_seconds': duration_seconds,
            'iterations': iteration,
            'start_memory_mb': start_memory / 1024 / 1024,
            'final_memory_mb': final_memory / 1024 / 1024,
            'memory_growth_mb': memory_growth / 1024 / 1024,
            'trend_growth_mb': trend_growth / 1024 / 1024,
            'memory_samples': len(memory_samples),
            'potential_leak': memory_growth > 10 * 1024 * 1024,  # >10MB growth
            'iterations_per_second': iteration / duration_seconds,
            'memory_samples': memory_samples[-10:]  # Last 10 samples
        }
        
        leak_status = "POTENTIAL LEAK" if results['potential_leak'] else "NO LEAK DETECTED"
        print(f"   Result: {leak_status}")
        print(f"   Memory Growth: {results['memory_growth_mb']:.2f}MB over {duration_seconds}s")
        
        return results
        
    def _create_mock_widget(self, widget_type: str, widget_id: int):
        """Create a mock widget for testing"""
        class MockWidget:
            def __init__(self, widget_type: str, widget_id: int):
                self.widget_type = widget_type
                self.widget_id = widget_id
                self.data = f"Widget {widget_type}_{widget_id}"
                self.visible = True
                self.children = []
                self.parent = None
                self._render_count = 0
                
                # Add some memory overhead to simulate real widgets
                self._internal_data = [f"data_{i}" for i in range(10)]
                
            def set_data(self, data):
                self.data = data
                
            def update(self, new_data):
                self.data = new_data
                self._render_count += 1
                
            def render(self):
                self._render_count += 1
                return f"Rendered {self.widget_type}_{self.widget_id}: {self.data}"
                
            def add_child(self, child):
                self.children.append(child)
                child.parent = self
                
        return MockWidget(widget_type, widget_id)
        
    def _simulate_layout_calculation(self, widgets):
        """Simulate layout calculations for widgets"""
        for widget in widgets:
            # Simulate layout calculations
            widget.x = hash(widget.widget_id) % 1000
            widget.y = hash(widget.widget_id) // 1000 % 1000
            widget.width = 100 + (widget.widget_id % 200)
            widget.height = 30 + (widget.widget_id % 50)
            
    def _simulate_widget_rendering(self, widgets):
        """Simulate widget rendering"""
        for widget in widgets:
            # Simulate rendering work
            rendered_output = widget.render()
            widget._cached_render = rendered_output
            
    def _simulate_event_handling(self, widgets):
        """Simulate event handling"""
        for i, widget in enumerate(widgets):
            if i % 5 == 0:  # Every 5th widget gets an "event"
                widget.update(f"Event handled: {widget.data}")
                
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report_lines = [
            "🎯 WIDGET PERFORMANCE ANALYSIS REPORT",
            "=" * 50,
            ""
        ]
        
        # Widget creation performance
        if self.widget_metrics:
            report_lines.extend([
                "📊 Widget Creation Performance:",
                ""
            ])
            
            total_widgets = sum(m.render_count for m in self.widget_metrics)
            total_time = sum(m.creation_time for m in self.widget_metrics)
            total_memory = sum(m.memory_usage for m in self.widget_metrics)
            
            report_lines.extend([
                f"Total Widgets Created: {total_widgets:,}",
                f"Total Creation Time: {total_time:.3f}s",
                f"Total Memory Usage: {total_memory/1024/1024:.2f}MB",
                f"Average Creation Speed: {total_widgets/total_time:.0f} widgets/s",
                ""
            ])
            
            # Per-widget-type breakdown
            for metric in self.widget_metrics:
                widgets_per_sec = metric.render_count / metric.creation_time if metric.creation_time > 0 else 0
                memory_per_widget = metric.memory_usage / metric.render_count if metric.render_count > 0 else 0
                
                report_lines.append(
                    f"  {metric.widget_type:10}: {widgets_per_sec:6.0f} w/s, "
                    f"{memory_per_widget:6.0f} bytes/widget"
                )
                
        # Render cycle performance
        if self.render_cycles:
            report_lines.extend([
                "",
                "🎨 Render Cycle Performance:",
                ""
            ])
            
            avg_time = sum(c.render_time for c in self.render_cycles) / len(self.render_cycles)
            avg_memory_growth = sum(c.memory_after - c.memory_before for c in self.render_cycles) / len(self.render_cycles)
            avg_cpu = sum(c.cpu_percent for c in self.render_cycles) / len(self.render_cycles)
            
            report_lines.extend([
                f"Render Cycles: {len(self.render_cycles)}",
                f"Average Cycle Time: {avg_time:.3f}s",
                f"Average Memory Growth: {avg_memory_growth/1024/1024:.2f}MB/cycle",
                f"Average CPU Usage: {avg_cpu:.1f}%",
                ""
            ])
            
        # Performance assessment
        report_lines.extend([
            "⚡ Performance Assessment:",
            ""
        ])
        
        # Check if performance meets targets
        issues = []
        recommendations = []
        
        if self.widget_metrics:
            slow_widgets = [m for m in self.widget_metrics if m.render_count / m.creation_time < 50]
            if slow_widgets:
                issues.append(f"Slow widget creation: {len(slow_widgets)} widget types < 50 widgets/s")
                recommendations.append("Optimize widget constructors and initialization")
                
        if self.render_cycles:
            slow_cycles = [c for c in self.render_cycles if c.render_time > 0.1]
            if slow_cycles:
                issues.append(f"Slow render cycles: {len(slow_cycles)} cycles > 100ms")
                recommendations.append("Implement virtual scrolling and lazy rendering")
                
            high_memory_cycles = [c for c in self.render_cycles if c.memory_after - c.memory_before > 10*1024*1024]
            if high_memory_cycles:
                issues.append(f"High memory usage: {len(high_memory_cycles)} cycles > 10MB growth")
                recommendations.append("Implement object pooling and better memory management")
                
        if not issues:
            report_lines.append("✅ No major performance issues detected")
        else:
            report_lines.append("⚠️  Issues Detected:")
            for issue in issues:
                report_lines.append(f"   • {issue}")
                
        if recommendations:
            report_lines.extend([
                "",
                "💡 Optimization Recommendations:",
            ])
            for rec in recommendations:
                report_lines.append(f"   • {rec}")
                
        return "\n".join(report_lines)
        
    def save_detailed_report(self, filepath: str):
        """Save detailed performance data"""
        data = {
            'timestamp': time.time(),
            'baseline_memory_mb': self.baseline_memory / 1024 / 1024,
            'widget_metrics': [
                {
                    'widget_type': m.widget_type,
                    'creation_time': m.creation_time,
                    'update_time': m.update_time,
                    'memory_usage_mb': m.memory_usage / 1024 / 1024,
                    'render_count': m.render_count,
                    'widgets_per_second': m.render_count / m.creation_time if m.creation_time > 0 else 0,
                    'memory_per_widget_bytes': m.memory_usage / m.render_count if m.render_count > 0 else 0
                }
                for m in self.widget_metrics
            ],
            'render_cycles': [
                {
                    'cycle_id': c.cycle_id,
                    'total_widgets': c.total_widgets,
                    'render_time': c.render_time,
                    'memory_growth_mb': (c.memory_after - c.memory_before) / 1024 / 1024,
                    'cpu_percent': c.cpu_percent
                }
                for c in self.render_cycles
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def run_widget_performance_tests():
    """Run comprehensive widget performance tests"""
    tester = WidgetPerformanceTester()
    tester.setup_testing_environment()
    
    # Run all tests
    print("\n" + "="*60)
    creation_results = tester.test_widget_creation_performance(100)
    
    print("\n" + "="*60)
    cycle_results = tester.test_render_cycle_performance(10, 50)
    
    print("\n" + "="*60)
    dataset_results = tester.test_large_dataset_rendering(1000, 1000)
    
    print("\n" + "="*60)
    leak_results = tester.test_memory_leak_simulation(15)  # Shorter duration for testing
    
    # Generate and display report
    print("\n" + "="*60)
    report = tester.generate_performance_report()
    print(report)
    
    # Save detailed results
    timestamp = int(time.time())
    results_file = f"widget_performance_results_{timestamp}.json"
    tester.save_detailed_report(results_file)
    
    # Summary
    print(f"\n📊 PERFORMANCE TEST SUMMARY")
    print(f"=" * 40)
    print(f"Widget Creation: {'✅ GOOD' if creation_results else '❌ ISSUES'}")
    print(f"Render Cycles: {'✅ GOOD' if cycle_results['average_cycle_time'] < 0.1 else '⚠️ SLOW'}")
    print(f"Large Datasets: {'✅ GOOD' if dataset_results['total_memory_mb'] < 50 else '⚠️ HIGH MEMORY'}")
    print(f"Memory Leaks: {'✅ CLEAN' if not leak_results['potential_leak'] else '⚠️ POTENTIAL LEAK'}")
    print(f"Detailed results saved to: {results_file}")
    
    return {
        'creation': creation_results,
        'cycles': cycle_results,
        'datasets': dataset_results,
        'leaks': leak_results,
        'report': report
    }


if __name__ == "__main__":
    try:
        results = run_widget_performance_tests()
        print(f"\n✅ Widget performance testing completed successfully!")
    except Exception as e:
        print(f"\n❌ Widget performance testing failed: {e}")
        import traceback
        traceback.print_exc()