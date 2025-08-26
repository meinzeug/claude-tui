#!/usr/bin/env python3
"""
Monitoring Module
Real-time monitoring and metrics collection for Claude-TUI
"""

from .dashboard import MonitoringDashboard, MetricsCollector, SwarmMetrics, run_dashboard

__all__ = [
    "MonitoringDashboard",
    "MetricsCollector", 
    "SwarmMetrics",
    "run_dashboard"
]

__version__ = "1.0.0"