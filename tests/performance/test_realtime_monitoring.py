"""
Real-time Performance Monitoring Tests

Tests for the real-time performance monitoring system including SLA tracking,
alerting, and dashboard functionality.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from performance.monitoring.realtime_monitor import (
    RealtimePerformanceMonitor,
    SLATracker,
    SLAThreshold,
    AlertManager,
    AlertRule,
    PerformanceAlert,
    MetricsBuffer,
    PerformanceDashboard
)

class TestMetricsBuffer:
    """Test metrics buffer functionality"""
    
    def test_buffer_initialization(self):
        """Test buffer initialization with custom size"""
        buffer = MetricsBuffer(maxsize=100)
        assert len(buffer.buffer) == 0
        assert buffer.buffer.maxlen == 100
    
    def test_buffer_append_and_get(self):
        """Test adding and retrieving metrics"""
        buffer = MetricsBuffer(maxsize=5)
        
        # Add metrics
        for i in range(3):
            buffer.append({'metric': f'test_{i}', 'value': i})
        
        # Get all metrics
        metrics = buffer.get_recent()
        assert len(metrics) == 3
        assert metrics[0]['metric'] == 'test_0'
        assert metrics[2]['metric'] == 'test_2'
        
        # Get recent count
        recent = buffer.get_recent(count=2)
        assert len(recent) == 2
        assert recent[0]['metric'] == 'test_1'  # Last 2 items
        assert recent[1]['metric'] == 'test_2'
    
    def test_buffer_maxsize_behavior(self):
        """Test buffer behavior when exceeding max size"""
        buffer = MetricsBuffer(maxsize=3)
        
        # Add more items than max size
        for i in range(5):
            buffer.append({'metric': f'test_{i}', 'value': i})
        
        # Should only contain last 3 items
        metrics = buffer.get_recent()
        assert len(metrics) == 3
        assert metrics[0]['metric'] == 'test_2'  # Oldest retained
        assert metrics[2]['metric'] == 'test_4'  # Newest
    
    def test_buffer_clear(self):
        """Test buffer clearing"""
        buffer = MetricsBuffer()
        
        # Add some metrics
        buffer.append({'test': 1})
        buffer.append({'test': 2})
        
        assert len(buffer.get_recent()) == 2
        
        # Clear buffer
        buffer.clear()
        assert len(buffer.get_recent()) == 0

class TestSLATracker:
    """Test SLA tracking functionality"""
    
    @pytest.fixture
    def sla_thresholds(self):
        return [
            SLAThreshold('cpu.percent', 80.0, 'lt', 'warning', 'CPU usage should be below 80%'),
            SLAThreshold('response_time', 2.0, 'lt', 'critical', 'Response time should be below 2s'),
            SLAThreshold('throughput', 100.0, 'gt', 'warning', 'Throughput should be above 100 ops/s')
        ]
    
    @pytest.fixture
    def sla_tracker(self, sla_thresholds):
        return SLATracker(sla_thresholds)
    
    def test_sla_threshold_initialization(self, sla_tracker):
        """Test SLA threshold initialization"""
        assert len(sla_tracker.sla_thresholds) == 3
        assert 'cpu.percent' in sla_tracker.sla_thresholds
        assert sla_tracker.sla_thresholds['cpu.percent'].threshold_value == 80.0
    
    def test_sla_compliance_check_pass(self, sla_tracker):
        """Test SLA compliance when all thresholds are met"""
        metrics = {
            'cpu': {'percent': 70.0},  # Below 80 (good)
            'response_time': 1.5,      # Below 2.0 (good)
            'throughput': 150.0        # Above 100 (good)
        }
        
        result = sla_tracker.check_sla_compliance(metrics)
        
        assert result['overall_compliance'] is True
        assert len(result['violations']) == 0
        assert len(result['compliance_status']) == 3
        
        # Check individual compliance
        assert result['compliance_status']['cpu.percent']['compliant'] is True
        assert result['compliance_status']['response_time']['compliant'] is True
        assert result['compliance_status']['throughput']['compliant'] is True
    
    def test_sla_compliance_check_violations(self, sla_tracker):
        """Test SLA compliance when thresholds are violated"""
        metrics = {
            'cpu': {'percent': 90.0},  # Above 80 (violation)
            'response_time': 3.0,      # Above 2.0 (violation)
            'throughput': 50.0         # Below 100 (violation)
        }
        
        result = sla_tracker.check_sla_compliance(metrics)
        
        assert result['overall_compliance'] is False
        assert len(result['violations']) == 3
        
        # Check violation details
        cpu_violation = next(v for v in result['violations'] if v['metric_name'] == 'cpu.percent')
        assert cpu_violation['current_value'] == 90.0
        assert cpu_violation['threshold_value'] == 80.0
        assert cpu_violation['severity'] == 'warning'
    
    def test_get_metric_value(self, sla_tracker):
        """Test nested metric value extraction"""
        metrics = {
            'level1': {
                'level2': {
                    'value': 42.5
                }
            },
            'simple_value': 10.0
        }
        
        # Test nested path
        value = sla_tracker._get_metric_value(metrics, 'level1.level2.value')
        assert value == 42.5
        
        # Test simple path
        value = sla_tracker._get_metric_value(metrics, 'simple_value')
        assert value == 10.0
        
        # Test missing path
        value = sla_tracker._get_metric_value(metrics, 'missing.path')
        assert value is None
    
    def test_threshold_evaluation(self, sla_tracker):
        """Test threshold evaluation logic"""
        threshold_lt = SLAThreshold('test', 50.0, 'lt', 'warning', 'Test less than')
        threshold_gt = SLAThreshold('test', 50.0, 'gt', 'warning', 'Test greater than')
        threshold_eq = SLAThreshold('test', 50.0, 'eq', 'warning', 'Test equal')
        
        # Test less than
        assert sla_tracker._evaluate_threshold(30.0, threshold_lt) is True  # 30 < 50
        assert sla_tracker._evaluate_threshold(60.0, threshold_lt) is False  # 60 > 50
        
        # Test greater than
        assert sla_tracker._evaluate_threshold(60.0, threshold_gt) is True  # 60 > 50
        assert sla_tracker._evaluate_threshold(30.0, threshold_gt) is False  # 30 < 50
        
        # Test equal
        assert sla_tracker._evaluate_threshold(50.0, threshold_eq) is True  # 50 == 50
        assert sla_tracker._evaluate_threshold(30.0, threshold_eq) is False  # 30 != 50
    
    def test_compliance_summary(self, sla_tracker):
        """Test compliance summary generation"""
        # Add some test compliance history
        current_time = datetime.utcnow()
        
        sla_tracker.compliance_history['cpu.percent'] = [
            {'timestamp': current_time - timedelta(hours=1), 'compliant': True, 'value': 70.0},
            {'timestamp': current_time - timedelta(minutes=30), 'compliant': False, 'value': 85.0},
            {'timestamp': current_time - timedelta(minutes=15), 'compliant': True, 'value': 75.0},
            {'timestamp': current_time, 'compliant': True, 'value': 65.0}
        ]
        
        summary = sla_tracker.get_compliance_summary(hours=2)
        
        assert 'cpu.percent' in summary
        cpu_summary = summary['cpu.percent']
        assert cpu_summary['total_checks'] == 4
        assert cpu_summary['violations'] == 1
        assert cpu_summary['compliance_rate'] == 0.75  # 3 out of 4 compliant

class TestAlertManager:
    """Test alert management functionality"""
    
    @pytest.fixture
    def alert_rules(self):
        return [
            AlertRule('high_cpu', 'cpu.percent', 90.0, 30, 'critical', 'scale_up'),
            AlertRule('high_latency', 'response_time', 2.0, 60, 'warning', 'investigate')
        ]
    
    @pytest.fixture
    def alert_manager(self, alert_rules):
        return AlertManager(alert_rules)
    
    def test_alert_rule_initialization(self, alert_manager):
        """Test alert rule initialization"""
        assert len(alert_manager.alert_rules) == 2
        assert 'high_cpu' in alert_manager.alert_rules
        assert alert_manager.alert_rules['high_cpu'].threshold == 90.0
    
    @pytest.mark.asyncio
    async def test_alert_evaluation_no_trigger(self, alert_manager):
        """Test alert evaluation when thresholds are not exceeded"""
        metrics = {
            'cpu': {'percent': 70.0},  # Below 90 threshold
            'response_time': 1.5       # Below 2.0 threshold
        }
        
        alerts = await alert_manager.evaluate_alerts(metrics)
        assert len(alerts) == 0
        assert len(alert_manager.active_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_alert_evaluation_trigger(self, alert_manager):
        """Test alert evaluation when thresholds are exceeded"""
        metrics = {
            'cpu': {'percent': 95.0},  # Above 90 threshold
            'response_time': 2.5       # Above 2.0 threshold
        }
        
        alerts = await alert_manager.evaluate_alerts(metrics)
        assert len(alerts) == 2
        assert len(alert_manager.active_alerts) == 2
        
        # Check alert details
        cpu_alert = next(a for a in alerts if a.rule_name == 'high_cpu')
        assert cpu_alert.severity == 'critical'
        assert cpu_alert.current_value == 95.0
        assert cpu_alert.threshold_value == 90.0
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self, alert_manager):
        """Test alert resolution when values return to normal"""
        # First, trigger an alert
        metrics_high = {'cpu': {'percent': 95.0}}
        alerts = await alert_manager.evaluate_alerts(metrics_high)
        assert len(alerts) == 1
        assert len(alert_manager.active_alerts) == 1
        
        # Then, resolve the alert
        metrics_normal = {'cpu': {'percent': 70.0}}
        alerts = await alert_manager.evaluate_alerts(metrics_normal)
        assert len(alerts) == 0
        assert len(alert_manager.active_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_alert_callback(self, alert_manager):
        """Test alert callback functionality"""
        callback_called = False
        alert_received = None
        
        def test_callback(alert):
            nonlocal callback_called, alert_received
            callback_called = True
            alert_received = alert
        
        alert_manager.add_alert_callback(test_callback)
        
        # Trigger alert
        metrics = {'cpu': {'percent': 95.0}}
        alerts = await alert_manager.evaluate_alerts(metrics)
        
        assert callback_called
        assert alert_received is not None
        assert alert_received.rule_name == 'high_cpu'
    
    def test_extract_metric_value(self, alert_manager):
        """Test metric value extraction from condition string"""
        metrics = {
            'cpu': {'percent': 85.0},
            'memory': {'usage': 75.0}
        }
        
        # Test simple extraction
        value = alert_manager._extract_metric_value(metrics, 'cpu.percent')
        assert value == 85.0
        
        # Test missing path
        value = alert_manager._extract_metric_value(metrics, 'disk.usage')
        assert value is None

class TestPerformanceDashboard:
    """Test performance dashboard functionality"""
    
    @pytest.fixture
    def dashboard(self):
        return PerformanceDashboard()
    
    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initialization"""
        assert not dashboard.monitoring_active
        assert len(dashboard.dashboard_data) > 0
        assert 'current_metrics' in dashboard.dashboard_data
        assert 'alerts' in dashboard.dashboard_data
        assert 'sla_status' in dashboard.dashboard_data
    
    def test_update_dashboard(self, dashboard):
        """Test dashboard data update"""
        current_metrics = {
            'cpu': {'percent': 75.0},
            'memory': {'percent': 60.0}
        }
        
        alerts = [
            PerformanceAlert(
                alert_id='test_alert',
                rule_name='test_rule',
                metric_name='cpu.percent',
                current_value=75.0,
                threshold_value=70.0,
                severity='warning',
                timestamp=datetime.utcnow(),
                description='Test alert'
            )
        ]
        
        sla_status = {
            'overall_compliance': False,
            'violations': [{'metric': 'cpu.percent'}]
        }
        
        dashboard.update_dashboard(current_metrics, alerts, sla_status)
        
        # Verify update
        assert dashboard.dashboard_data['current_metrics'] == current_metrics
        assert len(dashboard.dashboard_data['alerts']) == 1
        assert dashboard.dashboard_data['sla_status'] == sla_status
        assert 'last_updated' in dashboard.dashboard_data
    
    def test_flatten_metrics(self, dashboard):
        """Test metrics flattening functionality"""
        nested_metrics = {
            'cpu': {
                'percent': 75.0,
                'load': {
                    'avg_1': 1.5,
                    'avg_5': 2.0
                }
            },
            'memory': {
                'percent': 60.0
            },
            'simple_value': 42.0
        }
        
        flattened = dashboard._flatten_metrics(nested_metrics)
        
        expected_keys = {
            'cpu.percent',
            'cpu.load.avg_1',
            'cpu.load.avg_5',
            'memory.percent',
            'simple_value'
        }
        
        assert set(flattened.keys()) == expected_keys
        assert flattened['cpu.percent'] == 75.0
        assert flattened['cpu.load.avg_1'] == 1.5
        assert flattened['simple_value'] == 42.0
    
    def test_system_health_calculation(self, dashboard):
        """Test system health status calculation"""
        # Test healthy state
        dashboard._update_system_health([], {'overall_compliance': True})
        assert dashboard.dashboard_data['system_health'] == 'healthy'
        
        # Test warning state
        warning_alert = PerformanceAlert(
            alert_id='warning_alert',
            rule_name='test',
            metric_name='test',
            current_value=1.0,
            threshold_value=0.5,
            severity='warning',
            timestamp=datetime.utcnow(),
            description='Warning alert'
        )
        
        dashboard._update_system_health([warning_alert], {'overall_compliance': True})
        assert dashboard.dashboard_data['system_health'] == 'warning'
        
        # Test critical state
        critical_alert = PerformanceAlert(
            alert_id='critical_alert',
            rule_name='test',
            metric_name='test',
            current_value=1.0,
            threshold_value=0.5,
            severity='critical',
            timestamp=datetime.utcnow(),
            description='Critical alert'
        )
        
        dashboard._update_system_health([critical_alert], {'overall_compliance': True})
        assert dashboard.dashboard_data['system_health'] == 'critical'
        
        # Test critical state due to SLA violation
        dashboard._update_system_health([], {'overall_compliance': False})
        assert dashboard.dashboard_data['system_health'] == 'critical'
    
    @pytest.mark.asyncio
    async def test_dashboard_export(self, dashboard):
        """Test dashboard data export"""
        # Update dashboard with some data
        dashboard.update_dashboard({'test': 1}, [], {'overall_compliance': True})
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_filepath = tmp_file.name
        
        try:
            await dashboard.export_dashboard(temp_filepath)
            
            # Verify file was created and contains data
            assert os.path.exists(temp_filepath)
            
            with open(temp_filepath, 'r') as f:
                exported_data = json.load(f)
            
            assert 'current_metrics' in exported_data
            assert exported_data['current_metrics']['test'] == 1
            
        finally:
            # Cleanup
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)

class TestRealtimePerformanceMonitor:
    """Test the main real-time performance monitor"""
    
    @pytest.fixture
    def monitor(self):
        config = {
            'monitoring_interval': 0.1,  # Fast interval for testing
            'buffer_size': 100,
            'enable_sla_tracking': True,
            'enable_alerting': True
        }
        return RealtimePerformanceMonitor(config)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert not monitor.monitoring_active
        assert monitor.monitoring_interval == 0.1
        assert isinstance(monitor.sla_tracker, SLATracker)
        assert isinstance(monitor.alert_manager, AlertManager)
        assert isinstance(monitor.dashboard, PerformanceDashboard)
    
    def test_default_config(self, monitor):
        """Test default configuration loading"""
        default_monitor = RealtimePerformanceMonitor()
        default_config = default_monitor._default_config()
        
        assert 'monitoring_interval' in default_config
        assert 'buffer_size' in default_config
        assert 'enable_sla_tracking' in default_config
        assert 'enable_alerting' in default_config
    
    def test_sla_config_loading(self, monitor):
        """Test SLA configuration loading"""
        sla_config = monitor._load_sla_config()
        
        assert len(sla_config) > 0
        
        # Check for expected SLA thresholds
        sla_names = [sla.metric_name for sla in sla_config]
        assert 'cpu.percent' in sla_names
        assert 'memory.percent' in sla_names
        assert 'response_time.p95' in sla_names
        assert 'error_rate' in sla_names
    
    def test_alert_rules_loading(self, monitor):
        """Test alert rules loading"""
        alert_rules = monitor._load_alert_rules()
        
        assert len(alert_rules) > 0
        
        # Check for expected alert rules
        rule_names = [rule.name for rule in alert_rules]
        assert 'high_cpu' in rule_names
        assert 'high_memory' in rule_names
        assert 'high_error_rate' in rule_names
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring"""
        assert not monitor.monitoring_active
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        
        # Give it a moment to start
        await asyncio.sleep(0.05)
        assert monitor.monitoring_active
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Give it a moment to stop
        await asyncio.sleep(0.05)
        assert not monitor.monitoring_active
        
        # Cancel the task
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, mock_net, mock_disk, mock_memory, mock_cpu, monitor):
        """Test system metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 75.0
        mock_memory.return_value = Mock(percent=60.0, available=4000000000, used=6000000000, total=10000000000, free=3000000000)
        mock_disk.return_value = Mock(percent=45.0, free=500000000000, total=1000000000000)
        mock_net.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000, packets_sent=1000, packets_recv=2000)
        
        # Collect metrics
        metrics = await monitor._collect_metrics()
        
        # Verify metrics structure and values
        assert 'timestamp' in metrics
        assert 'cpu' in metrics
        assert 'memory' in metrics
        assert 'disk' in metrics
        assert 'network' in metrics
        
        assert metrics['cpu']['percent'] == 75.0
        assert metrics['memory']['percent'] == 60.0
        assert metrics['disk']['usage_percent'] == 45.0
    
    def test_combine_metrics(self, monitor):
        """Test metrics combination from different sources"""
        metrics_list = [
            {
                'type': 'system',
                'data': {
                    'cpu': {'percent': 75.0},
                    'memory': {'percent': 60.0}
                }
            },
            {
                'type': 'application',
                'data': {
                    'response_time': {'avg': 150},
                    'throughput': 250
                }
            }
        ]
        
        combined = monitor._combine_metrics(metrics_list)
        
        # Should contain both system and application metrics
        assert 'cpu' in combined
        assert 'response_time' in combined
        assert combined['cpu']['percent'] == 75.0
        assert combined['response_time']['avg'] == 150

@pytest.mark.integration
class TestRealtimeMonitoringIntegration:
    """Integration tests for real-time monitoring"""
    
    @pytest.mark.asyncio
    async def test_short_monitoring_session(self):
        """Test a complete short monitoring session"""
        config = {
            'monitoring_interval': 0.1,
            'dashboard_export_interval': 1.0,
            'log_metrics': True
        }
        
        monitor = RealtimePerformanceMonitor(config)
        
        # Mock system calls to avoid actual system dependency
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net:
            
            mock_memory.return_value = Mock(percent=40.0, available=6000000000, used=4000000000, total=10000000000, free=5000000000)
            mock_disk.return_value = Mock(percent=30.0, free=700000000000, total=1000000000000)
            mock_net.return_value = Mock(bytes_sent=500000, bytes_recv=1000000, packets_sent=500, packets_recv=1000)
            
            # Start monitoring
            monitoring_task = asyncio.create_task(monitor.start_monitoring())
            
            # Let it collect some data
            await asyncio.sleep(0.5)
            
            # Stop monitoring
            await monitor.stop_monitoring()
            
            # Cancel monitoring task
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Verify data was collected
            dashboard_data = monitor.dashboard.get_dashboard_data()
            assert dashboard_data['last_updated'] is not None
            assert len(monitor.metrics_buffer.get_recent()) > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])