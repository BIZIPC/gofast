# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Monitor model performance in production, track key metrics, and set alerts 
for performance degradation.
"""
import time
import pickle 
import threading
import smtplib
from numbers import Real,  Integral
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

from sklearn.utils._param_validation import StrOptions
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score
)

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params, Interval 
from ..tools.depsutils import ensure_pkg 
from .._gofastlog import gofastlog 

logger=gofastlog.get_gofast_logger(__name__)

__all__=[
    "ModelPerformanceMonitor", "ModelHealthChecker", "DataDriftMonitor",
    "AlertManager", "LatencyTracker", "ErrorRateMonitor",
    "CustomMetricsLogger",
    ]

class ModelPerformanceMonitor(BaseClass):
    """
    Monitors model performance in production, tracks key metrics, and sets
    alerts for performance degradation.

    The `ModelPerformanceMonitor` class provides methods to monitor the
    performance of machine learning models in production environments.
    It calculates various performance metrics over a sliding window of
    recent predictions to evaluate model performance over time. It also
    supports drift detection and integration with external monitoring
    tools like Prometheus.

    Parameters
    ----------
    metrics : list of str, default=['accuracy']
        List of performance metrics to monitor. Supported metrics include
        `'accuracy'`, `'precision'`, `'recall'`, `'f1'`, etc. These metrics
        will be calculated over a sliding window of recent predictions to
        evaluate model performance over time.

    drift_detection : bool, default=True
        Whether to enable drift detection for input data and model outputs.
        When set to `True`, the class will perform statistical tests to
        identify significant changes in data distribution or model behavior.

    alert_thresholds : dict of str to float or None, default=None
        Custom thresholds for triggering alerts when performance metrics
        degrade. The keys of the dictionary are metric names (e.g.,
        `'accuracy'`, `'f1'`), and the values are the threshold values.
        If a monitored metric falls below its threshold, an alert will be
        triggered.

    monitoring_tools : list of str or None, default=None
        List of monitoring tools to integrate with, such as `'prometheus'`.
        Integration allows metrics to be exported to external systems for
        visualization and alerting. Currently supported tools are
        `'prometheus'`.

    window_size : int, default=100
        The number of recent samples to consider for computing the
        performance metrics. This sliding window approach helps in tracking
        the model's performance in real-time by focusing on the most
        recent data. Must be a positive integer.

    Attributes
    ----------
    performance_history_ : dict of str to list of float
        Historical performance metrics tracked over time.

    drift_status_ : dict
        Current status of drift detection for data and model.

    Methods
    -------
    update(y_true, y_pred)
        Update the monitoring metrics with a new batch of data.

    get_performance_history()
        Get the historical performance metrics.

    get_drift_status()
        Get the current drift detection status.

    reset_monitor()
        Reset the monitoring state.

    set_thresholds(alert_thresholds)
        Set or update custom thresholds for performance alerts.

    save_state(filepath)
        Save the current monitoring state to a file.

    load_state(filepath)
        Load monitoring state from a file.

    Notes
    -----
    The performance metrics are calculated as follows:

    - **Accuracy**:

      .. math::

         \\text{Accuracy} = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}(y_i = \\hat{y}_i)

      where :math:`n` is the number of samples, :math:`y_i` is the true label,
      :math:`\\hat{y}_i` is the predicted label, and :math:`\\mathbb{1}` is
      the indicator function.

    Concept drift refers to the change in the statistical properties of the
    target variable over time, which can affect the performance of machine
    learning models in production environments [1]_.

    Examples
    --------
    >>> from gofast.mlops.monitoring import ModelPerformanceMonitor
    >>> monitor = ModelPerformanceMonitor(metrics=['accuracy', 'f1'], window_size=50)
    >>> for batch in data_stream:
    ...     predictions = model.predict(batch['features'])
    ...     monitor.update(batch['labels'], predictions)

    See Also
    --------
    DataDriftMonitor : Class for detecting data drift.
    ErrorRateMonitor : Monitors the error rate of model predictions.

    References
    ----------
    .. [1] Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014).
           "A survey on concept drift adaptation." *ACM Computing Surveys (CSUR)*, 46(4), 1-37.

    """

    @validate_params(
        {
            'metrics': [list],
            'drift_detection': [bool],
            'alert_thresholds': [dict, None],
            'monitoring_tools': [list, None],
            'window_size': [Interval(Integral, 1, None, closed='left')],
        }
    )
    def __init__(
        self,
        metrics: List[str] = ['accuracy'],
        drift_detection: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        monitoring_tools: Optional[List[str]] = None,
        window_size: int = 100,
    ):
        self.metrics = metrics
        self.drift_detection = drift_detection
        self.alert_thresholds = alert_thresholds or {}
        self.monitoring_tools = monitoring_tools or []
        self.window_size = window_size

        self.performance_history_ = {}
        self.drift_status_ = {}
        self._labels_window = []
        self._preds_window = []

        self._initialize_monitoring_tools()
        self._init_performance_metrics()
        self._init_alerting()

        if self.drift_detection:
            self._init_drift_detection()

    def _initialize_monitoring_tools(self):
        """Initialize connections to external monitoring tools."""
        for tool in self.monitoring_tools:
            if tool.lower() == 'prometheus':
                self._init_prometheus_client()
            else:
                raise ValueError(f"Unsupported monitoring tool: {tool}")

    def _init_prometheus_client(self):
        """Initialize Prometheus client."""
        @ensure_pkg(
            "prometheus_client",
            extra="To use Prometheus integration, please install 'prometheus_client'.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA
        )
        def init_client():
            import prometheus_client
            self.prometheus_metrics_ = {}
            for metric in self.metrics:
                self.prometheus_metrics_[metric] = prometheus_client.Gauge(
                    f'model_{metric}', f'Model {metric} over time'
                )
        init_client()


    def _init_performance_metrics(self):
        """Initialize performance metric functions."""
        self.metric_functions_ = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'log_loss': log_loss,
        }
        self.selected_metrics_ = {}
        for metric in self.metrics:
            if metric in self.metric_functions_:
                self.selected_metrics_[metric] = self.metric_functions_[metric]
            else:
                raise ValueError(f"Unsupported metric: {metric}")

    def _init_alerting(self):
        """Initialize alerting mechanisms."""
        # Initialize alert configurations
        self.alert_configs_ = {}

        # Email configuration
        self.email_config = getattr(self, 'email_config', None)
        if self.email_config and self.email_config.get('enabled', False):
            self.alert_configs_['email'] = self.email_config
            logger.info("Email alerting enabled.")
        else:
            logger.info("Email alerting not enabled or not configured.")

        # Slack configuration
        self.slack_config = getattr(self, 'slack_config', None)
        if self.slack_config and self.slack_config.get('enabled', False):
            self.alert_configs_['slack'] = self.slack_config
            logger.info("Slack alerting enabled.")
        else:
            logger.info("Slack alerting not enabled or not configured.")

        # SMS configuration
        self.sms_config = getattr(self, 'sms_config', None)
        if self.sms_config and self.sms_config.get('enabled', False):
            self.alert_configs_['sms'] = self.sms_config
            logger.info("SMS alerting enabled.")
        else:
            logger.info("SMS alerting not enabled or not configured.")

    def _init_drift_detection(self):
        """Initialize drift detection mechanisms."""

        def init_drift():
            self.ks_test = ks_2samp
        init_drift()

    def update(self, y_true: List[Any], y_pred: List[Any]):
        """
        Update the monitoring metrics with a new batch of data.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            Predicted labels.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        The method updates the internal sliding windows with the new data,
        computes the specified performance metrics, checks for alerts, and
        updates external monitoring tools if integrated.

        Examples
        --------
        >>> monitor.update([0, 1, 1], [0, 0, 1])

        """
        # Update the windowed data
        self._labels_window.extend(y_true)
        self._preds_window.extend(y_pred)
        # Keep only the recent window_size elements
        self._labels_window = self._labels_window[-self.window_size:]
        self._preds_window = self._preds_window[-self.window_size:]

        # Calculate metrics
        current_metrics = {}
        for metric_name, metric_func in self.selected_metrics_.items():
            try:
                if metric_name in ['precision', 'recall', 'f1']:
                    value = metric_func(self._labels_window, self._preds_window, average='weighted')
                else:
                    value = metric_func(self._labels_window, self._preds_window)
            except ValueError as e:
                logger.error(f"Error calculating {metric_name}: {e}")
                value = float('nan')

            current_metrics[metric_name] = value

            # Update performance history
            self.performance_history_.setdefault(metric_name, []).append(value)

            # Check for alerts
            threshold = self.alert_thresholds.get(metric_name)
            if threshold is not None and value < threshold:
                self._trigger_alert(metric_name, value)

            # Update monitoring tools
            if 'prometheus' in self.monitoring_tools:
                self.prometheus_metrics_[metric_name].set(value)

        # Perform drift detection if enabled
        if self.drift_detection:
            self._detect_drift()

        return self

    def _detect_drift(self):
        """
        Detect data and model drift using statistical tests.

        Notes
        -----
        Performs the Kolmogorov-Smirnov test between the distributions of
        true labels and predicted labels. If the p-value is below a
        significance level (e.g., 0.05), drift is considered to be detected.

        """
        if len(self._labels_window) < 2 or len(self._preds_window) < 2:
            # Not enough data to perform drift detection
            return

        y_true = self._labels_window
        y_pred = self._preds_window

        # Perform KS test between true labels and predicted labels
        stat, p_value = self.ks_test(y_true, y_pred)

        # Update drift status
        self.drift_status_['ks_statistic'] = stat
        self.drift_status_['p_value'] = p_value

        # Check if p_value is below the significance level (e.g., 0.05)
        if p_value < 0.05:
            self.drift_status_['drift_detected'] = True
            logger.warning(f"Drift detected: KS test p-value = {p_value:.4f}")
        else:
            self.drift_status_['drift_detected'] = False

    def get_performance_history(self) -> Dict[str, List[float]]:
        """
        Get the historical performance metrics.

        Returns
        -------
        performance_history_ : dict of str to list of float
            Historical performance metrics tracked over time.

        Examples
        --------
        >>> history = monitor.get_performance_history()
        >>> print(history)

        """
        return self.performance_history_

    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get the current drift detection status.

        Returns
        -------
        drift_status_ : dict
            Current status of drift detection for data and model.

        Examples
        --------
        >>> drift_status = monitor.get_drift_status()
        >>> print(drift_status)

        """
        return self.drift_status_

    def reset_monitor(self):
        """
        Reset the monitoring state.

        Notes
        -----
        Clears the performance history, drift status, and internal data
        windows.

        Examples
        --------
        >>> monitor.reset_monitor()

        """
        self.performance_history_.clear()
        self.drift_status_.clear()
        self._labels_window.clear()
        self._preds_window.clear()
        logger.info("Monitoring state has been reset.")

    def set_thresholds(self, alert_thresholds: Dict[str, float]):
        """
        Set or update custom thresholds for performance alerts.

        Parameters
        ----------
        alert_thresholds : dict of str to float
            Custom thresholds for triggering alerts when performance metrics
            degrade.

        Notes
        -----
        Updates the `alert_thresholds` with the provided values.

        Examples
        --------
        >>> monitor.set_thresholds({'accuracy': 0.9})

        """
        self.alert_thresholds.update(alert_thresholds)
        logger.info("Alert thresholds have been updated.")

    def save_state(self, filepath: str):
        """
        Save the current monitoring state to a file.

        Parameters
        ----------
        filepath : str
            Path to the file where the state will be saved.

        Notes
        -----
        Uses the `pickle` module to serialize the monitoring state.

        Examples
        --------
        >>> monitor.save_state('monitor_state.pkl')

        """
        def save():
     
            state = {
                'performance_history_': self.performance_history_,
                'drift_status_': self.drift_status_,
                '_labels_window': self._labels_window,
                '_preds_window': self._preds_window
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Monitoring state saved to {filepath}.")
        save()

    def load_state(self, filepath: str):
        """
        Load monitoring state from a file.

        Parameters
        ----------
        filepath : str
            Path to the file from which the state will be loaded.

        Notes
        -----
        Uses the `pickle` module to deserialize the monitoring state.

        Examples
        --------
        >>> monitor.load_state('monitor_state.pkl')

        """

        def load():
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.performance_history_ = state.get('performance_history_', {})
            self.drift_status_ = state.get('drift_status_', {})
            self._labels_window = state.get('_labels_window', [])
            self._preds_window = state.get('_preds_window', [])
            logger.info(f"Monitoring state loaded from {filepath}.")
        load()

    def _trigger_alert(self, metric_name: str, value: float):
        """
        Trigger an alert for performance degradation.

        Parameters
        ----------
        metric_name : str
            The name of the metric that triggered the alert.

        value : float
            The current value of the metric.

        Notes
        -----
        Logs a warning message and sends alerts via configured channels.

        """
        alert_message = (f"Performance alert: {metric_name} has dropped "
                         f"below threshold: {value:.4f}")
        logger.warning(alert_message)

        # Send email alert if email configuration is provided
        if 'email' in self.alert_configs_:
            self._send_email_alert(alert_message, self.alert_configs_['email'])

        # Additional alerting mechanisms can be added here

    def _send_email_alert(self, alert_message: str, email_config: Dict[str, Any]):
        """
        Send an email alert with the specified message.

        Parameters
        ----------
        alert_message : str
            The alert message to be sent via email.

        email_config : dict
            Configuration dictionary containing email settings.

        Notes
        -----
        Requires the `smtplib` and `email` modules.

        """
        def send_email():
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            smtp_username = email_config.get('smtp_username')
            smtp_password = email_config.get('smtp_password')
            sender_email = email_config.get('sender_email')
            receiver_email = email_config.get('receiver_email')

            if not all([smtp_server, smtp_username, smtp_password, sender_email, receiver_email]):
                logger.error("Incomplete email configuration provided.")
                return

            # Create the email
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = "Model Performance Alert"
            message.attach(MIMEText(alert_message, 'plain'))

            # Send the email
            try:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.sendmail(sender_email, receiver_email, message.as_string())
                server.quit()
                logger.info("Email alert sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")

        send_email()


class ModelHealthChecker(BaseClass):
    """
    Monitors the health of the system running the machine learning model,
    including CPU usage, memory usage, disk space, GPU usage, network bandwidth,
    and inference latency. Can alert on system resource issues.

    The health checker periodically monitors system resources and triggers alerts
    when usage exceeds defined thresholds. It records historical data for analysis
    and can be customized with user-defined alert callbacks.

    Parameters
    ----------
    alert_callback : callable or None, default=None
        A function that is called when a health issue is detected. It should
        accept two parameters: the issue type (`str`) and detailed metrics
        (`dict`). If `None`, no callback is executed when an alert is triggered.

    cpu_threshold : float, default=80.0
        The CPU usage percentage threshold for triggering an alert. Must be
        between 0 and 100 inclusive.

    memory_threshold : float, default=80.0
        The memory usage percentage threshold for triggering an alert. Must be
        between 0 and 100 inclusive.

    disk_threshold : float, default=80.0
        The disk usage percentage threshold for triggering an alert. Must be
        between 0 and 100 inclusive.

    gpu_threshold : float, default=80.0
        The GPU usage percentage threshold for triggering an alert. Must be
        between 0 and 100 inclusive.

    network_threshold : float, default=80.0
        The network bandwidth usage threshold (in Mbps) for triggering an alert.
        Must be a non-negative value.

    latency_threshold : float, default=2.0
        The latency threshold (in seconds) for triggering an alert. Must be a
        non-negative value.

    alert_messages : dict of str to str, default=None
        A dictionary of custom alert messages for different alert types
        (e.g., ``{'cpu': 'High CPU usage detected!'}``). If `None`, default
        messages are used.

    health_retention_period : int, default=100
        Number of recent health records to retain for analysis. Must be a
        positive integer.

    monitor_interval : int, default=60
        Time in seconds between health checks. Must be a positive integer.

    Attributes
    ----------
    latency_history_ : list of float
        Historical latency measurements.

    health_history_ : dict of str to list of float
        Historical health metrics for each monitored resource.

    Methods
    -------
    check_health()
        Checks system health metrics and triggers alerts if any are above
        thresholds.

    record_latency(latency)
        Records the latency of an inference operation.

    get_health_history(metric)
        Returns the historical values of a specific health metric.

    Notes
    -----
    The `ModelHealthChecker` class provides a way to monitor system resources
    critical to the performance of machine learning models in production. It
    can alert when resources are constrained, potentially affecting model
    performance or availability.

    Examples
    --------
    >>> from gofast.mlops.monitoring import ModelHealthChecker
    >>> def alert_callback(metric, info):
    ...     print(f"Alert! {metric}: {info['message']}")
    >>> health_checker = ModelHealthChecker(
    ...     alert_callback=alert_callback,
    ...     cpu_threshold=75.0,
    ...     memory_threshold=80.0,
    ...     gpu_threshold=90.0,
    ...     network_threshold=100.0,
    ...     latency_threshold=1.5
    ... )
    >>> health_checker.check_health()
    >>> health_checker.record_latency(1.6)

    See Also
    --------
    psutil : A cross-platform library for retrieving information on running
        processes and system utilization.
    GPUtil : A Python module for getting the GPU status from NVIDIA GPUs.

    References
    ----------
    .. [1] "psutil Documentation", https://psutil.readthedocs.io/
    .. [2] "GPUtil Documentation", https://github.com/anderskm/gputil

    """

    @validate_params({
        'alert_callback': [callable, None],
        'cpu_threshold': [Interval(Real, 0, 100, closed='both')],
        'memory_threshold': [Interval(Real, 0, 100, closed='both')],
        'disk_threshold': [Interval(Real, 0, 100, closed='both')],
        'gpu_threshold': [Interval(Real, 0, 100, closed='both')],
        'network_threshold': [Interval(Real, 0, None, closed='left')],
        'latency_threshold': [Interval(Real, 0, None, closed='left')],
        'alert_messages': [dict, None],
        'health_retention_period': [Interval(Integral, 1, None, closed='left')],
        'monitor_interval': [Interval(Integral, 1, None, closed='left')],
    })
    def __init__(
        self,
        alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 80.0,
        gpu_threshold: float = 80.0,
        network_threshold: float = 80.0,
        latency_threshold: float = 2.0,
        alert_messages: Optional[Dict[str, str]] = None,
        health_retention_period: int = 100,
        monitor_interval: int = 60,
    ):
        self.alert_callback = alert_callback
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.gpu_threshold = gpu_threshold
        self.network_threshold = network_threshold
        self.latency_threshold = latency_threshold
        self.alert_messages = alert_messages or {}
        self.health_retention_period = health_retention_period
        self.monitor_interval = monitor_interval
        self.latency_history_ = []
        self.health_history_ = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'gpu': [],
            'network': [],
        }
        
    @ensure_pkg(
        "psutil",
        extra="The 'psutil' package is required for system monitoring, "
              "including CPU, memory, and process management.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def check_health(self):
        """
        Monitors the system's CPU, memory, disk, GPU, and network usage, and
        triggers alerts if usage exceeds the defined thresholds. Records the
        health metrics in history.

        The method gathers the current usage statistics for each monitored
        resource and checks them against their respective thresholds. If any
        resource usage exceeds its threshold, an alert is triggered.

        Notes
        -----
        - CPU, memory, and disk usage are monitored using the `psutil` library.
        - GPU usage is monitored using the `GPUtil` library.
        - Network bandwidth usage is calculated based on bytes sent and received
          over a short interval.

        Examples
        --------
        >>> health_checker.check_health()

        """
        import psutil 
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        self._log_health_metric('cpu', cpu_usage)
        self._check_and_alert('cpu', cpu_usage, self.cpu_threshold)

        # Memory usage
        memory_usage = self._get_memory_usage()
        self._log_health_metric('memory', memory_usage)
        self._check_and_alert('memory', memory_usage, self.memory_threshold)

        # Disk usage
        disk_usage = self._get_disk_usage()
        self._log_health_metric('disk', disk_usage)
        self._check_and_alert('disk', disk_usage, self.disk_threshold)

        # GPU usage
        gpu_usage = self._get_gpu_usage()
        self._log_health_metric('gpu', gpu_usage)
        self._check_and_alert('gpu', gpu_usage, self.gpu_threshold)

        # Network usage
        network_usage = self._get_network_usage()
        self._log_health_metric('network', network_usage)
        self._check_and_alert('network', network_usage, self.network_threshold)

    def record_latency(self, latency: float):
        """
        Records the latency of a model inference operation and triggers an alert
        if the latency exceeds the threshold.

        Parameters
        ----------
        latency : float
            The time taken for an inference operation (in seconds). Must be a
            non-negative value.

        Notes
        -----
        The method appends the latency value to the `latency_history_` and
        checks if it exceeds the `latency_threshold`. If it does, an alert is
        triggered.

        Examples
        --------
        >>> health_checker.record_latency(1.6)

        """
        if latency < 0:
            raise ValueError("Latency must be a non-negative value.")

        self.latency_history_.append(latency)
        if len(self.latency_history_) > self.health_retention_period:
            self.latency_history_.pop(0)

        if latency > self.latency_threshold:
            message = self.alert_messages.get(
                'latency',
                f"Inference latency too high: {latency:.2f}s"
            )
            self._trigger_alert('latency', latency, message)

    def get_health_history(self, metric: str) -> List[float]:
        """
        Returns the historical values of a specific health metric.

        Parameters
        ----------
        metric : {'cpu', 'memory', 'disk', 'gpu', 'network'}
            The name of the health metric to retrieve.

        Returns
        -------
        history : list of float
            The history of values for the specified metric.

        Raises
        ------
        ValueError
            If the metric name is not recognized.

        Examples
        --------
        >>> cpu_history = health_checker.get_health_history('cpu')
        >>> print(cpu_history)
        [45.0, 50.0, 60.0, ...]

        """
        if metric not in self.health_history_:
            raise ValueError(f"Unknown metric '{metric}'.")
        return self.health_history_[metric]

    def _get_memory_usage(self) -> float:
        """
        Retrieves the current memory usage percentage.

        Returns
        -------
        memory_usage : float
            The memory usage percentage.

        Notes
        -----
        Uses the `psutil` library to get the system's memory usage.

        """
        import psutil

        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        return memory_usage


    def _get_disk_usage(self) -> float:
        """
        Retrieves the current disk usage percentage.

        Returns
        -------
        disk_usage : float
            The disk usage percentage.

        Notes
        -----
        Uses the `psutil` library to get the disk usage of the root directory (`'/'`).

        """
        import psutil
        disk_info = psutil.disk_usage('/')
        disk_usage = disk_info.percent
        return disk_usage

    @ensure_pkg(
        "GPUtil",
        extra="The 'GPUtil' package is required for GPU monitoring.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _get_gpu_usage(self) -> float:
        """
        Retrieves the current GPU usage percentage.

        Returns
        -------
        gpu_usage : float
            The GPU usage percentage. If no GPU is detected, returns 0.0.

        Notes
        -----
        Uses the `GPUtil` library to get the GPU load.

        """
        import GPUtil
        gpus = GPUtil.getGPUs()
        if not gpus:
            gpu_usage = 0.0
        else:
            # Return the highest GPU usage among all GPUs
            gpu_usage = max([gpu.load * 100 for gpu in gpus])
        return gpu_usage

    def _get_network_usage(self) -> float:
        """
        Calculates the current network bandwidth usage in Mbps.

        Returns
        -------
        network_usage : float
            The network bandwidth usage in Mbps.

        Notes
        -----
        The method calculates the bandwidth as the total bytes sent and received
        over a one-second interval, converted to megabits per second (Mbps).

        .. math::

            \\text{Bandwidth (Mbps)} = \\frac{(\\Delta \\text{Bytes} \\times 8)}{1 \\times 10^6}

        """
        import psutil 
        net_io_1 = psutil.net_io_counters()
        bytes_sent_1 = net_io_1.bytes_sent
        bytes_recv_1 = net_io_1.bytes_recv
        time.sleep(1)
        net_io_2 = psutil.net_io_counters()
        bytes_sent_2 = net_io_2.bytes_sent
        bytes_recv_2 = net_io_2.bytes_recv
        total_bytes = (bytes_sent_2 - bytes_sent_1) + (bytes_recv_2 - bytes_recv_1)
        total_bits = total_bytes * 8
        bandwidth_mbps = total_bits / 1e6  # Convert to Mbps
        return bandwidth_mbps

    def _check_and_alert(self, metric: str, usage: float, threshold: float):
        """
        Checks if the usage of a specific resource exceeds the threshold and
        triggers an alert if necessary.

        Parameters
        ----------
        metric : str
            The type of metric being checked (e.g., 'cpu', 'memory').

        usage : float
            The current usage percentage or value of the resource.

        threshold : float
            The threshold percentage or value for the resource.

        Notes
        -----
        If the usage exceeds the threshold, an alert is triggered using the
        `_trigger_alert` method.

        """
        if usage > threshold:
            default_message = f"{metric.capitalize()} usage too high: {usage:.2f}%"
            message = self.alert_messages.get(metric, default_message)
            self._trigger_alert(metric, usage, message)

    def _trigger_alert(self, metric: str, value: float, message: str):
        """
        Triggers an alert by calling the `alert_callback` function with the
        specific details.

        Parameters
        ----------
        metric : str
            The type of resource for which the alert is triggered
            (e.g., 'cpu', 'memory').

        value : float
            The current value of the metric that caused the alert.

        message : str
            The alert message to be logged or sent.

        Notes
        -----
        If an `alert_callback` is provided, it is called with the metric and
        a dictionary containing the value and message.

        """
        logger.warning(message)
        if self.alert_callback:
            self.alert_callback(metric, {'value': value, 'message': message})

    def _log_health_metric(self, metric: str, value: float):
        """
        Logs the health metric value and maintains the retention period for
        the history.

        Parameters
        ----------
        metric : str
            The name of the metric being logged (e.g., 'cpu', 'memory').

        value : float
            The value of the metric.

        Notes
        -----
        The method appends the value to the history list for the metric and
        ensures that the history does not exceed the `health_retention_period`.

        """
        self.health_history_[metric].append(value)
        if len(self.health_history_[metric]) > self.health_retention_period:
            self.health_history_[metric].pop(0)


class DataDriftMonitor(BaseClass):
    """
    Monitors data drift by comparing distributions of input features over time.
    Alerts when a significant drift is detected in the input data distribution.

    The class compares incoming data with baseline data using statistical tests
    and alerts when significant differences are detected. It supports multiple
    drift detection methods and handles missing data according to user preference.

    Parameters
    ----------
    alert_callback : callable or None, default=None
        A function that is called when data drift is detected. It should accept
        two parameters: the issue type (`str`) and detailed metrics (`dict`).

    drift_thresholds : dict of str to float or None, default=None
        A dictionary of per-feature drift thresholds (e.g., ``{'feature_0': 0.05}``).
        If not provided, a global `drift_threshold` is used for all features.

    drift_threshold : float, default=0.05
        The global p-value threshold for detecting significant drift. Must be
        between 0 and 1 inclusive.

    baseline_data : np.ndarray or None, default=None
        Baseline data distribution to compare incoming data against. It should
        be a 2D array of shape (n_samples, n_features).

    drift_detection_method : {'ks', 'chi2', 'jsd'}, default='ks'
        The statistical method to use for drift detection:

        - ``'ks'``: Kolmogorov-Smirnov test.
        - ``'chi2'``: Chi-squared test.
        - ``'jsd'``: Jensen-Shannon Divergence.

    handle_missing : {'skip', 'impute'}, default='skip'
        How to handle missing data:

        - ``'skip'``: Ignore missing values in calculations.
        - ``'impute'``: Impute missing values with the mean of the feature.

    alert_messages : dict of str to str or None, default=None
        Custom alert messages for different drift scenarios.

    Attributes
    ----------
    drift_history_ : list of dict
        Keeps track of detected drift results over time.

    Methods
    -------
    monitor_drift(new_data)
        Compares the incoming data to the baseline and checks for drift.

    set_baseline_data(baseline_data)
        Sets or updates the baseline data for drift detection.

    get_drift_history()
        Returns the history of drift detection results.

    Notes
    -----
    The drift detection methods are based on statistical tests:

    - **Kolmogorov-Smirnov test (KS test)** compares the cumulative distributions
      of two samples.

      .. math::
          D = \\sup_x | F_1(x) - F_2(x) |

    - **Chi-squared test** evaluates whether distributions of categorical variables
      differ from each other.

    - **Jensen-Shannon Divergence (JSD)** is a symmetric measure of the difference
      between two probability distributions.

      .. math::
          JSD(P || Q) = \\frac{1}{2} D_{KL}(P || M) + \\frac{1}{2} D_{KL}(Q || M)

      where :math:`M = \\frac{1}{2}(P + Q)` and :math:`D_{KL}` is 
      the Kullback-Leibler divergence.

    Examples
    --------
    >>> from gofast.mlops.monitoring import DataDriftMonitor
    >>> import numpy as np
    >>> baseline_data = np.random.normal(0, 1, (1000, 3))
    >>> new_data = np.random.normal(0.5, 1, (1000, 3))
    >>> def alert_callback(issue_type, details):
    ...     print(f"Alert: {issue_type}, Details: {details}")
    >>> drift_monitor = DataDriftMonitor(
    ...     alert_callback=alert_callback,
    ...     baseline_data=baseline_data,
    ...     drift_detection_method='ks'
    ... )
    >>> drift_monitor.monitor_drift(new_data)
    Alert: data_drift, Details: {'feature_0': 0.0, 'feature_1': 0.0, 'feature_2': 0.0}

    See Also
    --------
    ModelHealthChecker : Monitors system health metrics such as CPU and memory usage.

    References
    ----------
    .. [1] "Scipy Statistical Functions",
            https://docs.scipy.org/doc/scipy/reference/stats.html
    .. [2] "Jensen-Shannon Divergence", 
           https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    """

    @validate_params({
        'alert_callback': [callable, None],
        'drift_thresholds': [dict, None],
        'drift_threshold': [Interval(Real, 0, 1, closed='both')],
        'baseline_data': [np.ndarray, None],
        'drift_detection_method': [StrOptions({'ks', 'chi2', 'jsd'})],
        'handle_missing': [StrOptions({'skip', 'impute'})],
        'alert_messages': [dict, None],
    })
    def __init__(
        self,
        alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        drift_thresholds: Optional[Dict[str, float]] = None,
        drift_threshold: float = 0.05,
        baseline_data: Optional[np.ndarray] = None,
        drift_detection_method: str = 'ks',
        handle_missing: str = 'skip',
        alert_messages: Optional[Dict[str, str]] = None
    ):
        self.alert_callback = alert_callback
        self.drift_thresholds = drift_thresholds or {}
        self.drift_threshold = drift_threshold
        self.baseline_data = baseline_data
        self.drift_detection_method = drift_detection_method
        self.handle_missing = handle_missing
        self.alert_messages = alert_messages or {}
        self.drift_history_ = []

    def set_baseline_data(self, baseline_data: np.ndarray):
        """
        Sets or updates the baseline data distribution for drift monitoring.

        Parameters
        ----------
        baseline_data : np.ndarray of shape (n_samples, n_features)
            The baseline data distribution.

        Raises
        ------
        ValueError
            If `baseline_data` is not a 2D array.
        """
        if baseline_data.ndim != 2:
            raise ValueError("`baseline_data` must be a 2D array.")
        self.baseline_data = baseline_data

    def monitor_drift(self, new_data: np.ndarray):
        """
        Compares the new data distribution to the baseline and triggers an alert
        if drift is detected.

        Parameters
        ----------
        new_data : np.ndarray of shape (n_samples, n_features)
            The incoming data to compare to the baseline.

        Raises
        ------
        ValueError
            If `baseline_data` is not set or if the number of features does 
            not match.
        """
        if self.baseline_data is None:
            raise ValueError(
                "Baseline data is not set. Use `set_baseline_data()`"
                " to provide baseline data.")

        if self.baseline_data.shape[1] != new_data.shape[1]:
            raise ValueError("The number of features in new data"
                             " must match the baseline data.")


        p_values = {}
        drift_detected = False

        n_features = self.baseline_data.shape[1]

        for i in range(n_features):
            feature_name = f"feature_{i}"

            # Handle missing data
            baseline_col = self._handle_missing_data(self.baseline_data[:, i])
            new_col = self._handle_missing_data(new_data[:, i])

            # Perform the drift test
            if self.drift_detection_method == 'ks':
                stat, p_value = stats.ks_2samp(baseline_col, new_col)
            elif self.drift_detection_method == 'chi2':
                # Bin the data to create a contingency table
                bins = 'auto'
                baseline_hist, bin_edges = np.histogram(baseline_col, bins=bins)
                new_hist, _ = np.histogram(new_col, bins=bin_edges)
                contingency_table = np.array([baseline_hist, new_hist])
                stat, p_value = stats.chi2_contingency(contingency_table)[:2]
            elif self.drift_detection_method == 'jsd':
                p_value = self._jensen_shannon_divergence(baseline_col, new_col)
            else:
                raise ValueError(f"Unknown drift detection method: {self.drift_detection_method}")

            # Use per-feature threshold if available, otherwise use global threshold
            threshold = self.drift_thresholds.get(feature_name, self.drift_threshold)
            p_values[feature_name] = p_value

            # For 'ks' and 'chi2', lower p-value indicates significant difference
            # For 'jsd', higher value indicates more divergence
            if self.drift_detection_method in {'ks', 'chi2'}:
                drift = p_value < threshold
            elif self.drift_detection_method == 'jsd':
                drift = p_value > threshold
            else:
                drift = False

            if drift:
                drift_detected = True

        # Store the drift detection result in history
        self.drift_history_.append({"p_values": p_values, "drift_detected": drift_detected})

        # Trigger alert if drift is detected
        if drift_detected:
            message = self.alert_messages.get('drift', f"Data drift detected: {p_values}")
            if self.alert_callback:
                self.alert_callback('data_drift', p_values)
            logger.warning(message)

    def _handle_missing_data(self, data: np.ndarray) -> np.ndarray:
        """
        Handles missing data according to the specified strategy ('skip' or 'impute').

        Parameters
        ----------
        data : np.ndarray
            The data to handle.

        Returns
        -------
        data : np.ndarray
            The data with missing values handled.

        Raises
        ------
        ValueError
            If an unknown missing data handling strategy is specified.
        """
        if self.handle_missing == 'skip':
            return data[~np.isnan(data)]
        elif self.handle_missing == 'impute':
            mean_value = np.nanmean(data)
            return np.nan_to_num(data, nan=mean_value)
        else:
            raise ValueError(f"Unknown missing data handling strategy: {self.handle_missing}")

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Jensen-Shannon Divergence between two distributions.

        Parameters
        ----------
        p : np.ndarray
            The first sample data.

        q : np.ndarray
            The second sample data.

        Returns
        -------
        jsd : float
            The Jensen-Shannon Divergence between the two distributions.

        Notes
        -----
        The samples are converted into probability distributions by creating histograms.

        """

        # Create histograms to estimate the probability distributions
        bins = 'auto'
        p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

        # Add small value to avoid zero probabilities
        epsilon = 1e-10
        p_hist += epsilon
        q_hist += epsilon

        # Normalize histograms to sum to 1
        p_hist /= np.sum(p_hist)
        q_hist /= np.sum(q_hist)

        # Compute Jensen-Shannon Divergence
        jsd = jensenshannon(p_hist, q_hist, base=2.0)
        return jsd

    def get_drift_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of drift detection results.

        Returns
        -------
        drift_history : list of dict
            The history of drift detection results.

        Examples
        --------
        >>> history = drift_monitor.get_drift_history()
        >>> print(history)
        [{'p_values': {'feature_0': 0.0, 'feature_1': 0.01}, 'drift_detected': True}, ...]
        """
        return self.drift_history_


class LatencyTracker(BaseClass):
    """
    Tracks and monitors the latency of model inference operations,
    providing detailed insights into average latency, tail latencies,
    and distribution analysis.

    The `LatencyTracker` class collects latency data for specified
    operations, enabling analysis of latency performance over time.
    It can trigger alerts when latency thresholds are exceeded and
    provides methods to retrieve statistical summaries.

    Parameters
    ----------
    alert_callback : callable or None, default=None
        A function to call when the latency exceeds a defined threshold.
        The callback receives three arguments: the operation name (`str`),
        the exceeded latency (`float`), and a custom message (`str`).

    latency_thresholds : dict of str to float or None, default=None
        A dictionary where keys are operation names (e.g., `'inference'`,
        `'preprocessing'`) and values are latency thresholds (in seconds)
        for triggering an alert. If `None`, the `global_latency_threshold`
        is used for all operations.

    global_latency_threshold : float, default=2.0
        A global maximum allowable latency (in seconds) before triggering
        an alert for any operation that does not have a specific threshold
        set in `latency_thresholds`. Must be a non-negative value.

    retention_period : int, default=100
        The number of recent latency values to retain for detailed analysis.
        Must be a positive integer.

    percentiles_to_track : list of float or None, default=None
        A list of percentiles to monitor. If `None`, defaults to
        tracking `[90.0, 95.0, 99.0]`. Each percentile must be between
        0 and 100 inclusive.

    alert_messages : dict of str to str or None, default=None
        Custom alert messages for different operations (e.g.,
        ``{'inference': 'Inference latency too high'}``). If `None`,
        default messages are used.

    Attributes
    ----------
    latencies_ : dict of str to list of float
        Stores latency values for each operation being tracked.

    Methods
    -------
    record_latency(operation, latency)
        Records the latency of a specific operation.

    get_average_latency(operation)
        Returns the average latency for a specific operation.

    get_tail_latency(operation, percentile=95.0)
        Returns the tail latency (e.g., 95th percentile) for a specific
        operation.

    get_latency_distribution(operation)
        Returns the latency distribution for a specific operation.

    set_latency_threshold(operation, threshold)
        Dynamically adjusts the latency threshold for a specific operation.

    get_percentile_summary(operation)
        Returns a summary of latency percentiles for a specific operation.

    Notes
    -----
    The latency values are stored per operation, and the retention period
    limits the number of stored values to prevent unbounded memory growth.

    Examples
    --------
    >>> from gofast.mlops.monitoring import LatencyTracker
    >>> def alert_callback(operation, latency, message):
    ...     print(f"Alert: {operation} - {message}")
    >>> latency_tracker = LatencyTracker(
    ...     alert_callback=alert_callback,
    ...     latency_thresholds={'inference': 1.0}
    ... )
    >>> latency_tracker.record_latency('inference', 1.2)
    Alert: inference - Inference latency too high: 1.2s
    >>> avg_latency = latency_tracker.get_average_latency('inference')
    >>> print(f"Average Latency: {avg_latency:.2f}s")
    Average Latency: 1.20s

    See Also
    --------
    ModelHealthChecker : Monitors system health metrics.
    DataDriftMonitor : Monitors data drift in input features.

    References
    ----------
    .. [1] "Percentile", Wikipedia,
       https://en.wikipedia.org/wiki/Percentile

    """

    @validate_params({
        'alert_callback': [callable, None],
        'latency_thresholds': [dict, None],
        'global_latency_threshold': [Interval(Real, 0, None, closed='left')],
        'retention_period': [Interval(Integral, 1, None, closed='left')],
        'percentiles_to_track': [list, None],
        'alert_messages': [dict, None],
    })
    def __init__(
        self,
        alert_callback: Optional[Callable[[str, float, str], None]] = None,
        latency_thresholds: Optional[Dict[str, float]] = None,
        global_latency_threshold: float = 2.0,
        retention_period: int = 100,
        percentiles_to_track: Optional[List[float]] = None,
        alert_messages: Optional[Dict[str, str]] = None,
    ):
        self.alert_callback = alert_callback
        self.latency_thresholds = latency_thresholds or {}
        self.global_latency_threshold = global_latency_threshold
        self.retention_period = retention_period
        self.percentiles_to_track = percentiles_to_track or [90.0, 95.0, 99.0]
        self.alert_messages = alert_messages or {}
        self.latencies_ = {}

    @validate_params({
        'operation': [str],
        'latency': [Interval(Real, 0, None, closed='left')],
    })
    def record_latency(self, operation: str, latency: float):
        """
        Records the latency of a specific operation.

        Parameters
        ----------
        operation : str
            The name of the operation (e.g., `'inference'`, `'preprocessing'`).

        latency : float
            The time taken for the operation (in seconds). Must be a
            non-negative value.

        Notes
        -----
        If the latency exceeds the threshold for the operation, an alert is
        triggered via the `alert_callback`, if provided. The method also ensures
        that only the most recent `retention_period` latency values are stored.

        """
        if operation not in self.latencies_:
            self.latencies_[operation] = []

        self.latencies_[operation].append(latency)
        if len(self.latencies_[operation]) > self.retention_period:
            self.latencies_[operation].pop(0)

        # Check if latency exceeds the threshold for the operation
        threshold = self.latency_thresholds.get(
            operation, self.global_latency_threshold)
        if latency > threshold:
            message = self.alert_messages.get(
                operation, f"{operation.capitalize()} latency too high: {latency}s")
            logger.warning(message)
            if self.alert_callback:
                self.alert_callback(operation, latency, message)

    @validate_params({
        'operation': [str],
    })
    def get_average_latency(self, operation: str) -> float:
        """
        Returns the average latency for a specific operation.

        Parameters
        ----------
        operation : str
            The name of the operation.

        Returns
        -------
        average_latency : float
            The average latency.

        Notes
        -----
        If no latency data is available for the operation, returns `nan`.

        Examples
        --------
        >>> avg_latency = latency_tracker.get_average_latency('inference')
        >>> print(f"Average Latency: {avg_latency:.2f}s")

        """
        latencies = self.latencies_.get(operation, [])
        if not latencies:
            return float('nan')
        return np.mean(latencies)

    @validate_params({
        'operation': [str],
        'percentile': [Interval(Real, 0, 100, closed='both')],
    })
    def get_tail_latency(self, operation: str, percentile: float = 95.0) -> float:
        """
        Returns the tail latency (e.g., 95th percentile) for a specific operation.

        Parameters
        ----------
        operation : str
            The name of the operation.

        percentile : float, default=95.0
            The percentile to compute. Must be between 0 and 100 inclusive.

        Returns
        -------
        tail_latency : float
            The latency at the specified percentile.

        Notes
        -----
        The tail latency provides insight into the higher end of latency
        distribution, which is critical for understanding worst-case performance.

        Examples
        --------
        >>> tail_latency = latency_tracker.get_tail_latency('inference', 99.0)
        >>> print(f"99th Percentile Latency: {tail_latency:.2f}s")

        """
        latencies = self.latencies_.get(operation, [])
        if not latencies:
            return float('nan')
        return np.percentile(latencies, percentile)

    @validate_params({
        'operation': [str],
    })
    def get_latency_distribution(self, operation: str) -> np.ndarray:
        """
        Returns the latency distribution for a specific operation.

        Parameters
        ----------
        operation : str
            The name of the operation.

        Returns
        -------
        latency_distribution : ndarray of shape (n_samples,)
            The array of recorded latencies.

        Notes
        -----
        The method returns the collected latency values for the operation,
        which can be used for further analysis or visualization.

        Examples
        --------
        >>> latency_distribution = latency_tracker.get_latency_distribution('inference')
        >>> import matplotlib.pyplot as plt
        >>> plt.hist(latency_distribution)
        >>> plt.show()

        """
        return np.array(self.latencies_.get(operation, []))

    @validate_params({
        'operation': [str],
        'threshold': [Interval(Real, 0, None, closed='left')],
    })
    def set_latency_threshold(self, operation: str, threshold: float):
        """
        Dynamically adjusts the latency threshold for a specific operation.

        Parameters
        ----------
        operation : str
            The name of the operation.

        threshold : float
            The new latency threshold in seconds. Must be a non-negative value.

        Notes
        -----
        This method allows updating the latency threshold for an operation at
        runtime, enabling dynamic tuning of alerting behavior.

        Examples
        --------
        >>> latency_tracker.set_latency_threshold('inference', 1.5)

        """
        self.latency_thresholds[operation] = threshold
        logger.info(f"Set latency threshold for {operation} to {threshold}s")

    @validate_params({
        'operation': [str],
    })
    def get_percentile_summary(self, operation: str) -> Dict[str, float]:
        """
        Returns a summary of latency percentiles for a specific operation.

        Returns
        -------
        percentile_summary : dict of str to float
            A dictionary mapping percentile names to latency values.

        Notes
        -----
        The percentiles are defined in `percentiles_to_track`. The method
        provides a quick overview of latency performance at different
        percentile levels.

        Examples
        --------
        >>> summary = latency_tracker.get_percentile_summary('inference')
        >>> print(summary)
        {'90.0th_percentile': 0.8, '95.0th_percentile': 1.0, '99.0th_percentile': 1.2}

        """
        latencies = self.latencies_.get(operation, [])
        if not latencies:
            return {}

        return {
            f"{percentile}th_percentile": np.percentile(latencies, percentile)
            for percentile in self.percentiles_to_track
        }


class AlertManager(BaseClass):
    """
    Manages different types of alerts (performance, health, latency, drift)
    and provides options to send notifications via email, webhooks (Slack,
    Microsoft Teams), and other channels. Supports retries and batching for
    robustness.

    The `AlertManager` class centralizes alert management for machine learning
    systems, providing mechanisms to send alerts through various channels and
    handle failures with retries and batching. It can be integrated with
    monitoring tools to notify stakeholders of significant events.

    Parameters
    ----------
    email_recipients : list of str or None, default=None
        List of email addresses to notify in case of an alert. If `None`, no
        email alerts are sent.

    webhook_urls : list of str or None, default=None
        List of URLs for sending alerts to webhooks (e.g., Slack, Microsoft
        Teams). If `None`, no webhook alerts are sent.

    smtp_server : str or None, default=None
        SMTP server to use for sending email alerts. Required if email alerts
        are used.

    from_email : str or None, default=None
        Email address from which alerts will be sent. Required if email alerts
        are used.

    retry_attempts : int, default=3
        The number of retry attempts if an alert fails to send. Must be a
        non-negative integer.

    batch_alerts : bool, default=False
        Whether to batch multiple alerts together and send them in one go.

    batch_interval : int, default=60
        The time interval (in seconds) for batching alerts. Must be a positive
        integer. Only applicable if `batch_alerts` is `True`.

    Attributes
    ----------
    batched_alerts_ : list of tuple
        Internal storage for batched alerts. Each element is a tuple of
        (`alert_type`, `details`).

    Methods
    -------
    send_email_alert(subject, message, retries=0)
        Sends an email alert to the configured recipients.

    send_webhook_alert(message, retries=0)
        Sends an alert message to configured webhook URLs.

    add_email_recipient(email)
        Adds an email recipient dynamically.

    remove_email_recipient(email)
        Removes an email recipient dynamically.

    log_alert(alert_type, details)
        Logs the alert for future reference and review.

    Notes
    -----
    The `AlertManager` can be customized to handle different alerting
    strategies, such as batching alerts to reduce notification frequency or
    implementing exponential backoff for retries. It supports integration with
    common communication platforms via webhooks.

    Examples
    --------
    >>> from gofast.mlops.monitoring import AlertManager
    >>> alert_manager = AlertManager(
    ...     email_recipients=['admin@example.com'],
    ...     webhook_urls=['https://hooks.slack.com/services/...'],
    ...     smtp_server='smtp.example.com',
    ...     from_email='alerts@example.com'
    ... )
    >>> alert_manager.send_email_alert(
    ...     'Model Performance Alert',
    ...     'Model accuracy dropped below threshold.'
    ... )
    >>> alert_manager.send_webhook_alert(
    ...     'Data drift detected in feature XYZ.'
    ... )

    See Also
    --------
    DataDriftMonitor : Monitors data drift in input features.
    ModelHealthChecker : Monitors system health metrics.

    References
    ----------
    .. [1] "Sending Emails with Python", Python Documentation,
       https://docs.python.org/3/library/email.examples.html
    .. [2] "Slack Webhooks", Slack API Documentation,
       https://api.slack.com/messaging/webhooks

    """

    @validate_params({
        'email_recipients': [list, None],
        'webhook_urls': [list, None],
        'smtp_server': [str, None],
        'from_email': [str, None],
        'retry_attempts': [Interval(Integral, 0, None, closed='left')],
        'batch_alerts': [bool],
        'batch_interval': [Interval(Integral, 1, None, closed='left')],
    })
    def __init__(
        self,
        email_recipients: Optional[List[str]] = None,
        webhook_urls: Optional[List[str]] = None,
        smtp_server: Optional[str] = None,
        from_email: Optional[str] = None,
        retry_attempts: int = 3,
        batch_alerts: bool = False,
        batch_interval: int = 60
    ):
        self.email_recipients = email_recipients or []
        self.webhook_urls = webhook_urls or []
        self.smtp_server = smtp_server
        self.from_email = from_email
        self.retry_attempts = retry_attempts
        self.batch_alerts = batch_alerts
        self.batch_interval = batch_interval
        self.batched_alerts_ = []
        if batch_alerts:
            self._start_batching()

    def send_email_alert(self, subject: str, message: str, retries: int = 0):
        """
        Sends an email alert to the configured recipients with retry logic.

        Parameters
        ----------
        subject : str
            The subject of the email alert.

        message : str
            The body of the email alert.

        retries : int, default=0
            The current retry attempt. Used internally for recursive retry
            calls. Users should not need to specify this parameter.

        Raises
        ------
        ValueError
            If SMTP server or sender email is not configured.

        Notes
        -----
        The method attempts to send an email using the configured SMTP server.
        If sending fails, it retries up to `retry_attempts` times. Logging is
        used to report failures and retries.

        Examples
        --------
        >>> alert_manager.send_email_alert(
        ...     'Performance Alert',
        ...     'Model accuracy dropped.'
        ... )

        """
        if not self.smtp_server or not self.from_email:
            raise ValueError(
                "SMTP server and sender email must be configured for email alerts."
            )

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = ", ".join(self.email_recipients)

        try:
            with smtplib.SMTP(self.smtp_server) as server:
                server.sendmail(
                    self.from_email,
                    self.email_recipients,
                    msg.as_string()
                )
            logger.info(f"Email alert sent: {subject}")
        except Exception as e:
            if retries < self.retry_attempts:
                logger.error(
                    f"Failed to send email alert. Retrying... "
                    f"{retries + 1}/{self.retry_attempts}"
                )
                self.send_email_alert(subject, message, retries + 1)
            else:
                logger.error(
                    f"Failed to send email alert after {self.retry_attempts} "
                    f"attempts: {e}"
                )

    @ensure_pkg(
        "requests",
        extra="The 'requests' package is required for sending webhook alerts.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def send_webhook_alert(self, message: str, retries: int = 0):
        """
        Sends an alert message to configured webhook URLs with retry logic.

        Parameters
        ----------
        message : str
            The alert message to send.

        retries : int, default=0
            The current retry attempt. Used internally for recursive retry
            calls. Users should not need to specify this parameter.

        Notes
        -----
        The method sends a POST request to each configured webhook URL with the
        alert message. If sending fails, it retries up to `retry_attempts`
        times. Logging is used to report failures and retries.

        Examples
        --------
        >>> alert_manager.send_webhook_alert(
        ...     'Data drift detected in feature XYZ.'
        ... )

        """
        import requests  

        for url in self.webhook_urls:
            try:
                response = requests.post(url, json={'text': message})
                if response.status_code == 200:
                    logger.info(f"Webhook alert sent to {url}")
                else:
                    raise ValueError(
                        f"Webhook failed with status code {response.status_code}"
                    )
            except Exception as e:
                if retries < self.retry_attempts:
                    logger.error(
                        f"Failed to send webhook alert to {url}. Retrying... "
                        f"{retries + 1}/{self.retry_attempts}"
                    )
                    self.send_webhook_alert(message, retries + 1)
                else:
                    logger.error(
                        f"Failed to send webhook alert to {url} after "
                        f"{self.retry_attempts} attempts: {e}"
                    )

    def add_email_recipient(self, email: str):
        """
        Adds an email recipient dynamically.

        Parameters
        ----------
        email : str
            The email address to add.

        Notes
        -----
        This method allows adding email recipients after the `AlertManager` has
        been initialized.

        Examples
        --------
        >>> alert_manager.add_email_recipient('new_user@example.com')

        """
        if email not in self.email_recipients:
            self.email_recipients.append(email)
            logger.info(f"Added email recipient: {email}")

    def remove_email_recipient(self, email: str):
        """
        Removes an email recipient dynamically.

        Parameters
        ----------
        email : str
            The email address to remove.

        Notes
        -----
        This method allows removing email recipients after the `AlertManager` has
        been initialized.

        Examples
        --------
        >>> alert_manager.remove_email_recipient('old_user@example.com')

        """
        if email in self.email_recipients:
            self.email_recipients.remove(email)
            logger.info(f"Removed email recipient: {email}")

    def log_alert(self, alert_type: str, details: Dict[str, Any]):
        """
        Logs the alert for future reference and review.

        Parameters
        ----------
        alert_type : str
            The type of alert (e.g., 'performance', 'health', 'latency',
            'drift').

        details : dict
            Additional details about the alert.

        Notes
        -----
        If `batch_alerts` is `True`, the alert is added to the batch to be sent
        after the `batch_interval`. Otherwise, the alert is logged immediately.

        Examples
        --------
        >>> alert_manager.log_alert('performance', {'accuracy': 0.85})

        """
        logger.info(f"Alert triggered: {alert_type} - {details}")
        if self.batch_alerts:
            self.batched_alerts_.append((alert_type, details))

    def _start_batching(self):
        """
        Starts batching alerts for sending them together at regular intervals.

        This method initializes a background thread that periodically checks for
        batched alerts and sends them via the configured channels. It is called
        automatically if `batch_alerts` is `True` during initialization.

        """
        def batch_sender():
            while True:
                if self.batched_alerts_:
                    for alert_type, details in self.batched_alerts_:
                        subject = f"Batched {alert_type.capitalize()} Alert"
                        message = str(details)
                        self.send_email_alert(subject, message)
                        self.send_webhook_alert(message)
                    self.batched_alerts_.clear()
                time.sleep(self.batch_interval)

        
        threading.Thread(target=batch_sender, daemon=True).start()

class ErrorRateMonitor(BaseClass):
    """
    Monitors the error rate of model predictions over time and triggers alerts
    when the error rate exceeds a defined threshold. Supports detailed error
    logging and per-type error tracking.

    The `ErrorRateMonitor` class helps in tracking the error rate of model
    predictions, enabling early detection of performance degradation. It can
    trigger alerts when the error rate exceeds specified thresholds and provides
    mechanisms to log and analyze different types of errors.

    Parameters
    ----------
    error_threshold : float
        The maximum allowable error rate before triggering an alert.
        Must be between 0 and 1 inclusive (e.g., 0.05 for 5% error rate).

    alert_callback : callable or None, default=None
        A function that is called when the error rate exceeds the threshold.
        It should accept two parameters: the current error rate (`float`)
        and a dictionary of error details (`dict`).

    retention_period : int, default=100
        The number of recent prediction outcomes to retain for tracking.
        Must be a positive integer.

    error_types : list of str or None, default=None
        A list of error types to track separately (e.g.,
        ``['prediction_failure', 'timeout']``). If `None`, error types are
        not tracked separately.

    Attributes
    ----------
    outcomes_ : list of bool
        Stores the most recent prediction outcomes, where `True` indicates
        a successful prediction and `False` indicates a failure.

    error_log_ : dict of str to int
        Keeps count of occurrences of each error type over the retained
        period.

    Methods
    -------
    log_prediction(outcome, error_type=None)
        Logs a prediction outcome, with optional error type.

    get_error_rate()
        Returns the current error rate over the retained period.

    get_error_type_count(error_type)
        Returns the count of a specific type of error over the retained period.

    Notes
    -----
    The error rate is calculated as:

    .. math::

        \\text{Error Rate} = 1 - \\frac{\\text{Number of Successes}}{\\text{Total Predictions}}

    The class maintains a sliding window of the most recent predictions,
    determined by the `retention_period`.

    Examples
    --------
    >>> from gofast.mlops.monitoring import ErrorRateMonitor
    >>> def alert_callback(error_rate, details):
    ...     print(f"Alert! Error rate exceeded: {error_rate:.2%}")
    >>> monitor = ErrorRateMonitor(
    ...     error_threshold=0.05,
    ...     alert_callback=alert_callback,
    ...     retention_period=100,
    ...     error_types=['prediction_failure', 'timeout']
    ... )
    >>> monitor.log_prediction(False, error_type='prediction_failure')
    >>> current_error_rate = monitor.get_error_rate()
    >>> print(f"Current Error Rate: {current_error_rate:.2%}")

    See Also
    --------
    LatencyTracker : Tracks and monitors the latency of operations.
    DataDriftMonitor : Monitors data drift in input features.

    References
    ----------
    .. [1] "Statistical Process Control", Wikipedia,
       https://en.wikipedia.org/wiki/Statistical_process_control

    """

    @validate_params({
        'error_threshold': [Interval(Real, 0, 1, closed='both')],
        'alert_callback': [callable, None],
        'retention_period': [Interval(Integral, 1, None, closed='left')],
        'error_types': [list, None],
    })
    def __init__(
        self,
        error_threshold: float,
        alert_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None,
        retention_period: int = 100,
        error_types: Optional[List[str]] = None
    ):
        self.error_threshold = error_threshold
        self.alert_callback = alert_callback
        self.retention_period = retention_period
        self.error_types = error_types or []
        self.outcomes_ = []
        self.error_log_ = {error_type: 0 for error_type in self.error_types}

    @validate_params({
        'outcome': [bool],
        'error_type': [str, None],
    })
    def log_prediction(self, outcome: bool, error_type: Optional[str] = None):
        """
        Logs a prediction outcome, with optional error type.

        Parameters
        ----------
        outcome : bool
            Whether the prediction was successful (`True`) or failed (`False`).

        error_type : str or None, default=None
            The type of error that occurred (if `outcome` is `False`).
            Must be one of the error types specified in `error_types`.

        Notes
        -----
        If the error rate exceeds the `error_threshold` after logging
        this prediction, the `alert_callback` is invoked with the current
        error rate and error details.

        """
        self.outcomes_.append(outcome)
        if len(self.outcomes_) > self.retention_period:
            self.outcomes_.pop(0)

        # Track specific error types
        if not outcome and error_type in self.error_log_:
            self.error_log_[error_type] += 1

        # Calculate error rate
        error_rate = 1 - sum(self.outcomes_) / len(self.outcomes_)

        if error_rate > self.error_threshold:
            error_details = {
                "error_rate": error_rate,
                "total_errors": len(self.outcomes_) - sum(self.outcomes_),
                "error_log": self.error_log_
            }
            logger.warning(f"Error rate exceeds threshold: {error_rate:.2%}")
            if self.alert_callback:
                self.alert_callback(error_rate, error_details)

    def get_error_rate(self) -> float:
        """
        Returns the current error rate over the retained period.

        Returns
        -------
        error_rate : float
            The current error rate. Returns `nan` if no outcomes are recorded.

        Notes
        -----
        The error rate is computed as:

        .. math::

            \\text{Error Rate} = 1 - \\frac{\\text{Number of Successes}}{\\text{Total Predictions}}

        Examples
        --------
        >>> error_rate = monitor.get_error_rate()
        >>> print(f"Error Rate: {error_rate:.2%}")

        """
        if len(self.outcomes_) == 0:
            return float('nan')
        return 1 - sum(self.outcomes_) / len(self.outcomes_)

    @validate_params({
        'error_type': [str],
    })
    def get_error_type_count(self, error_type: str) -> int:
        """
        Returns the count of a specific type of error over the retained period.

        Parameters
        ----------
        error_type : str
            The type of error to retrieve the count for.

        Returns
        -------
        count : int
            The count of the specified error type.

        Notes
        -----
        If the specified `error_type` was not initialized in `error_types`,
        returns zero.

        Examples
        --------
        >>> count = monitor.get_error_type_count('prediction_failure')
        >>> print(f"Prediction Failure Count: {count}")

        """
        return self.error_log_.get(error_type, 0)


class CustomMetricsLogger(BaseClass):
    """
    Logs and tracks custom user-defined metrics during model inference.
    Supports thresholds for metrics and triggers alerts when thresholds
    are exceeded.

    The `CustomMetricsLogger` class allows users to log custom metrics
    associated with model inference or other operations. It maintains a
    history of metric values and can trigger alerts when specified
    thresholds are exceeded. The class provides methods to retrieve
    metric histories and calculate moving averages.

    Parameters
    ----------
    retention_period : int, default=100
        The number of recent metric values to retain for each custom
        metric. Must be a positive integer.

    metric_thresholds : dict of str to float or None, default=None
        Thresholds for each metric, with an alert triggered if exceeded.
        The keys are metric names, and the values are threshold values.

    alert_callback : callable or None, default=None
        A function that is called when a metric exceeds its threshold.
        It should accept two parameters: the metric name (`str`) and the
        metric value (`float`).

    Attributes
    ----------
    metrics_history_ : dict of str to list of float
        Stores the history of values for each custom metric.

    Methods
    -------
    log_metric(metric_name, value)
        Logs a custom metric value.

    get_metric_history(metric_name)
        Returns the history of values for a specific custom metric.

    get_moving_average(metric_name, window)
        Returns the moving average for a specific metric over a given
        window.

    Notes
    -----
    The class maintains a sliding window of metric values for each
    custom metric, determined by the `retention_period`. When a new
    metric value is logged, if it exceeds the specified threshold for
    that metric, the `alert_callback` is invoked.

    Examples
    --------
    >>> from gofast.mlops.monitoring import CustomMetricsLogger
    >>> def alert_callback(metric_name, value):
    ...     print(f"Alert: {metric_name} exceeds threshold with value {value}")
    >>> metrics_logger = CustomMetricsLogger(
    ...     retention_period=100,
    ...     metric_thresholds={'throughput': 500.0},
    ...     alert_callback=alert_callback
    ... )
    >>> metrics_logger.log_metric('throughput', 550.0)
    Alert: throughput exceeds threshold with value 550.0
    >>> history = metrics_logger.get_metric_history('throughput')
    >>> print(history)
    [550.0]
    >>> moving_avg = metrics_logger.get_moving_average('throughput', window=1)
    >>> print(f"Moving Average: {moving_avg}")

    See Also
    --------
    LatencyTracker : Tracks and monitors the latency of operations.
    ErrorRateMonitor : Monitors the error rate of model predictions.

    References
    ----------
    .. [1] "Moving Average", Wikipedia,
       https://en.wikipedia.org/wiki/Moving_average

    """

    @validate_params({
        'retention_period': [Interval(Integral, 1, None, closed='left')],
        'metric_thresholds': [dict, None],
        'alert_callback': [callable, None],
    })
    def __init__(
        self,
        retention_period: int = 100,
        metric_thresholds: Optional[Dict[str, float]] = None,
        alert_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.retention_period = retention_period
        self.metric_thresholds = metric_thresholds or {}
        self.alert_callback = alert_callback
        self.metrics_history_ = {}

    @validate_params({
        'metric_name': [str],
        'value': [Real],
    })
    def log_metric(self, metric_name: str, value: float):
        """
        Logs a custom metric value.

        Parameters
        ----------
        metric_name : str
            The name of the custom metric.

        value : float
            The value of the metric.

        Notes
        -----
        If the metric value exceeds the threshold specified in
        `metric_thresholds`, an alert is triggered via the
        `alert_callback`, if provided. The method also ensures that only
        the most recent `retention_period` metric values are stored.

        Examples
        --------
        >>> metrics_logger.log_metric('throughput', 550.0)
        Alert: throughput exceeds threshold with value 550.0

        """
        if metric_name not in self.metrics_history_:
            self.metrics_history_[metric_name] = []

        self.metrics_history_[metric_name].append(value)
        if len(self.metrics_history_[metric_name]) > self.retention_period:
            self.metrics_history_[metric_name].pop(0)

        # Trigger alert if value exceeds threshold
        threshold = self.metric_thresholds.get(metric_name)
        if threshold is not None and value > threshold:
            message = f"Metric '{metric_name}' exceeds threshold: {value}"
            logger.warning(message)
            if self.alert_callback:
                self.alert_callback(metric_name, value)

    @validate_params({
        'metric_name': [str],
    })
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Returns the history of values for a specific custom metric.

        Parameters
        ----------
        metric_name : str
            The name of the custom metric.

        Returns
        -------
        history : list of float
            The history of values for the custom metric.

        Notes
        -----
        If no history is available for the specified metric, returns an
        empty list.

        Examples
        --------
        >>> history = metrics_logger.get_metric_history('throughput')
        >>> print(history)
        [550.0, 540.0, 530.0]

        """
        return self.metrics_history_.get(metric_name, [])


    @validate_params({
        'metric_name': [str],
        'window': [Interval(Integral, 1, None, closed='left')],
    })
    def get_moving_average(self, metric_name: str, window: int) -> float:
        """
        Returns the moving average for a specific metric over a given
        window.

        Parameters
        ----------
        metric_name : str
            The name of the custom metric.

        window : int
            The window size for calculating the moving average. Must be
            a positive integer.

        Returns
        -------
        moving_average : float
            The moving average of the custom metric over the specified
            window. Returns `nan` if no data is available.

        Notes
        -----
        The moving average is calculated using the most recent `window`
        number of metric values. If fewer values are available than the
        specified window, all available values are used.

        .. math::

            \\text{Moving Average} = \\frac{1}{N} \\sum_{i=1}^{N} x_i

        where :math:`N` is the number of values (up to the specified
        window size) and :math:`x_i` are the metric values.

        Examples
        --------
        >>> moving_avg = metrics_logger.get_moving_average('throughput', window=3)
        >>> print(f"Moving Average: {moving_avg}")

        """
 
        history = self.metrics_history_.get(metric_name, [])
        if not history:
            return float('nan')

        window = min(window, len(history))
        recent_values = history[-window:]
        return np.mean(recent_values)
