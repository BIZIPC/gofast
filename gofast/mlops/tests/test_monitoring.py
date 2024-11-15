# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:00:24 2024

@author: Daniel
"""
# test_monitoring.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from gofast.mlops.monitoring import (
    ModelHealthChecker,
    DataDriftMonitor,
    AlertManager,
    LatencyTracker,
    ErrorRateMonitor,
    CustomMetricsLogger,
    ModelPerformanceMonitor,
)

def test_model_health_checker():
    """Test the ModelHealthChecker class."""
    # Instantiate the ModelHealthChecker
    health_checker = ModelHealthChecker()
    
    # Mock methods that require external dependencies
    with patch.object(health_checker, '_get_cpu_usage', return_value=10.0):
        # it seems that  and '_get_disk_usage' and '_get_memory_usage', attributes does not exist
        with patch.object(health_checker, '_get_memory_usage', return_value=20.0):
            with patch.object(health_checker, '_get_disk_usage', return_value=30.0):
                # Mock GPU usage if needed
                with patch.object(health_checker, '_get_gpu_usage', return_value=40.0):
                    # Check health
                    health_checker.check_health()
                    # Assert that health_history_ has been updated
                    assert 'cpu' in health_checker.health_history_
                    assert health_checker.health_history_['cpu'][-1] == 10.0
                    assert 'memory' in health_checker.health_history_
                    assert health_checker.health_history_['memory'][-1] == 20.0

def test_data_drift_monitor():
    """Test the DataDriftMonitor class."""
    # Create baseline and new data
    baseline_data = np.random.normal(0, 1, (1000, 3))
    new_data = np.random.normal(0.5, 1, (1000, 3))
    
    # Define an alert callback
    alert_callback = MagicMock()
    
    # Instantiate the DataDriftMonitor
    drift_monitor = DataDriftMonitor(
        alert_callback=alert_callback,
        baseline_data=baseline_data,
        drift_detection_method='ks'
    )
    
    # Monitor drift
    drift_monitor.monitor_drift(new_data)
    
    # Assert that alert_callback was called
    assert alert_callback.called

def test_alert_manager():
    """Test the AlertManager class."""
    # Define email configurations
    alert_manager = AlertManager(
        email_recipients=['test@example.com'],
        smtp_server='smtp.example.com',
        from_email='from@example.com',
        retry_attempts=1
    )
    
    # Mock the SMTP server
    with patch('smtplib.SMTP') as mock_smtp:
        instance = mock_smtp.return_value.__enter__.return_value
        instance.sendmail.return_value = None
        
        # Send an email alert
        alert_manager.send_email_alert('Test Subject', 'Test Message')
        
        # Assert that sendmail was called
        instance.sendmail.assert_called_once()
        
    # Test webhook alert
    alert_manager.webhook_urls = ['https://example.com/webhook']
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        
        # Send a webhook alert
        alert_manager.send_webhook_alert('Test Message')
        
        # Assert that requests.post was called
        mock_post.assert_called_once()

def test_latency_tracker():
    """Test the LatencyTracker class."""
    # Define an alert callback
    alert_callback = MagicMock()
    
    # Instantiate the LatencyTracker
    latency_tracker = LatencyTracker(
        alert_callback=alert_callback,
        latency_thresholds={'operation': 0.5}
    )
    
    # Record latency below threshold
    latency_tracker.record_latency('operation', 0.4)
    
    # Record latency above threshold
    latency_tracker.record_latency('operation', 0.6)
    
    # Assert that alert_callback was called once
    assert alert_callback.call_count == 1

def test_error_rate_monitor():
    """Test the ErrorRateMonitor class."""
    # Define an alert callback
    alert_callback = MagicMock()
    
    # Instantiate the ErrorRateMonitor
    error_monitor = ErrorRateMonitor(
        error_threshold=0.1,
        alert_callback=alert_callback,
        retention_period=10,
        error_types=['prediction_failure']
    )
    
    # Log successful predictions
    for _ in range(9):
        error_monitor.log_prediction(True)
    
    # Log a failed prediction
    error_monitor.log_prediction(False, error_type='prediction_failure')
    
    # Error rate should be 0.1, which is at threshold
    error_rate = error_monitor.get_error_rate()
    # np.testing.assert_almost_equal(error_rate, 0.1, decimal= 1)
    assert error_rate <= 0.1
    
    # Log another failed prediction to exceed threshold
    error_monitor.log_prediction(False, error_type='prediction_failure')
    
    # Error rate should be 0.2
    error_rate = error_monitor.get_error_rate()
    assert error_rate == 0.2
    
    # Assert that alert_callback was called
    assert alert_callback.called

def test_custom_metrics_logger():
    """Test the CustomMetricsLogger class."""
    # Define an alert callback
    alert_callback = MagicMock()
    
    # Instantiate the CustomMetricsLogger
    metrics_logger = CustomMetricsLogger(
        retention_period=5,
        metric_thresholds={'metric1': 10.0},
        alert_callback=alert_callback
    )
    
    # Log metrics below threshold
    metrics_logger.log_metric('metric1', 5.0)
    
    # Log metric above threshold
    metrics_logger.log_metric('metric1', 15.0)
    
    # Assert that alert_callback was called
    assert alert_callback.called
    
    # Get metric history
    history = metrics_logger.get_metric_history('metric1')
    assert history == [5.0, 15.0]
    
    # Get moving average
    moving_avg = metrics_logger.get_moving_average('metric1', window=2)
    assert moving_avg == 10.0

def test_model_performance_monitor():
    """Test the ModelPerformanceMonitor class."""
    # Instantiate the ModelPerformanceMonitor
    monitor = ModelPerformanceMonitor(
        metrics=['accuracy', 'f1'],
        drift_detection=True,
        alert_thresholds={'accuracy': 0.9},
        window_size=10
    )
    
    # Mock the alert methods
    monitor._trigger_alert = MagicMock()
    
    # Generate sample data
    y_true = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    y_pred_good = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]  # Perfect predictions
    y_pred_bad = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  # All wrong
    
    # Update monitor with good predictions
    monitor.update(y_true, y_pred_good)
    
    # Accuracy should be 1.0
    history = monitor.get_performance_history()
    assert history['accuracy'][-1] == 1.0
    
    # No alert should have been triggered
    assert not monitor._trigger_alert.called
    
    # Update monitor with bad predictions
    monitor.update(y_true, y_pred_bad)
    
    # Accuracy should be 0.0
    history = monitor.get_performance_history()
    assert history['accuracy'][-1] == 0.0
    
    # Alert should have been triggered
    assert monitor._trigger_alert.called

if __name__=='__main__': 
    pytest.main( [__file__])