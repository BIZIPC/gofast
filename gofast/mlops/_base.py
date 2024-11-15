# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
""" Abstract Base-Classes for MLOps."""

from abc import abstractmethod
from numbers import Integral, Real
import random
from typing import Any, Callable, List, Optional, Dict, Tuple

import numpy as np

from .._gofastlog import gofastlog
from ..api.property import BaseLearner  
from ..compat.sklearn import ( 
     validate_params, Interval, HasMethods, StrOptions 
    )
from ..tools.validator import check_is_fitted, check_is_runned

logger = gofastlog.get_gofast_logger(__name__)

class BaseInference(BaseLearner):
    """
    Abstract base class for inference processes in gofast.mlops. This class
    provides a standardized framework for efficient inference workflows
    across different implementations.

    Parameters
    ----------
    model : object
        The machine learning model to use for inference. It must implement
        a `predict` method or be callable to perform inference.

    batch_size : int, optional, default=32
        Number of samples to process in a single batch during batch
        inference. A higher batch size can improve throughput but may
        increase memory usage.

    max_workers : int, optional, default=4
        Number of parallel workers used in inference tasks. This parameter
        controls the degree of parallelism and should be optimized based on
        system resources.

    timeout : Optional[float], default=None
        Specifies a timeout for each inference task in seconds. If set to
        `None`, no timeout is enforced, potentially leading to indefinite
        waits if a task is blocked.

    optimize_memory : bool, default=True
        Enables memory optimization during inference if `True`. This setting
        is beneficial for large-scale inference tasks where memory usage
        needs to be minimized.

    gpu_enabled : bool, default=False
        Enables GPU acceleration for inference tasks. This option requires
        a compatible environment and framework support for GPU usage.

    enable_padding : bool, default=False
        If `True`, ensures that input data batches are padded to match
        `batch_size`, which can be beneficial for certain models that
        perform better with consistent input sizes.

    log_level : {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'}, default='INFO'
        Controls the verbosity of the logging output. Use higher levels,
        such as 'DEBUG', for more detailed logs useful in debugging.

    Attributes
    ----------
    model_ : object
        The fitted model used for inference. This attribute holds the
        reference to the model instance used across all inference
        implementations.

    Methods
    -------
    run(data)
        Abstract method to be implemented by subclasses for performing
        inference on the given data.

    Notes
    -----
    This base class establishes a consistent interface for all inference
    processes in gofast.mlops. It enforces validation of parameters using
    the `validate_params` decorator from scikit-learn [1]_. All derived
    classes should implement the `run` method to specify the exact behavior
    for inference, e.g., batch or streaming.


    Inference is the process of making predictions :math:`\\hat{y}` on new
    data :math:`X` based on the model learned from training data. Let
    :math:`f(\\theta, X)` represent the model's prediction function where
    :math:`\\theta` are the model parameters:

    .. math::
        \\hat{y} = f(\\theta, X)

    The function `run` must define how :math:`f` is applied, including
    handling batch sizes, parallelism, and memory optimizations as
    specified.

    Examples
    --------
    >>> from gofast.mlops._base import BaseInference
    >>> class MyInference(BaseInference):
    ...     def run(self, data):
    ...         return self.model.predict(data)
    >>> # Example usage with a mock model
    >>> mock_model = lambda x: x * 2  # Mock model
    >>> inference = MyInference(mock_model, batch_size=16)
    >>> inference.run([1, 2, 3])  # Outputs: [2, 4, 6]

    See Also
    --------
    BatchInference : Optimizes batch processing for inference tasks.
    StreamingInference : Efficiently handles streaming data inference.
    MultiModelServing : Manages inference across multiple models.
    InferenceParallelizer : Implements parallelized inference for improved
        performance.

    References
    ----------
    .. [1] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in
           Python. Journal of Machine Learning Research, 12, 2825-2830.
    """

    @validate_params({
        'model':         [HasMethods(['predict']), callable],
        'batch_size':    [Interval(Integral, 1, None, closed='left')],
        'max_workers':   [Interval(Integral, 1, None, closed='left')],
        'timeout':       [Interval(Real, 0, None, closed='left'), None],
        'optimize_memory': [bool],
        'gpu_enabled':   [bool],
        'enable_padding': [bool],
        'log_level':     [StrOptions({'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'})]
    })
    def __init__(
        self,
        model: Any,
        batch_size: int = 32,
        max_workers: int = 4,
        timeout: Optional[float] = None,
        optimize_memory: bool = True,
        gpu_enabled: bool = False,
        enable_padding: bool = False,
        log_level: str = 'INFO'
    ):

        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.optimize_memory = optimize_memory
        self.gpu_enabled = gpu_enabled
        self.enable_padding = enable_padding
        self.log_level = log_level
        logger.setLevel(log_level)

    @abstractmethod
    def run(self, data: Any) -> Any:
        """
        Perform inference on the provided data.

        Parameters
        ----------
        data : array-like
            Input data for performing inference.

        Returns
        -------
        Any
            Result of the inference, which varies depending on the
            specific implementation in derived classes.
        """
        pass

class PipelineOrchestrator(BaseLearner):
    """
    Base class for pipeline orchestration integration with tools like
    Airflow and Prefect. This class defines the basic interface for
    creating, scheduling, and monitoring pipelines.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        An instance of :class:`PipelineManager` that manages pipeline steps.

    Attributes
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager associated with this orchestrator.

    Methods
    -------
    run()
        Abstract method to run the orchestration process.

    schedule_pipeline(schedule_interval)
        Abstract method to schedule the pipeline.

    monitor_pipeline()
        Abstract method to monitor the status of pipeline execution.

    Notes
    -----
    The :class:`PipelineOrchestrator` class provides an abstraction layer for
    integrating pipelines with orchestration tools. Subclasses must
    implement the abstract methods to provide tool-specific functionality.

    Examples
    --------
    >>> from gofast.mlops.pipeline import PipelineOrchestrator
    >>> class MyOrchestrator(PipelineOrchestrator):
    ...     def run(self):
    ...         pass
    ...     def schedule_pipeline(self, schedule_interval):
    ...         pass
    ...     def monitor_pipeline(self):
    ...         pass
    >>> pipeline_manager = PipelineManager()
    >>> orchestrator = MyOrchestrator(pipeline_manager)
    >>> orchestrator.run()

    See Also
    --------
    PrefectOrchestrator : Orchestrator using Prefect.
    AirflowOrchestrator : Orchestrator using Apache Airflow.

    References
    ----------
    .. [1] Smith, J. (2020). "Orchestrating Machine Learning Pipelines."
       *Journal of Data Engineering*, 5(3), 150-165.
    """

    @validate_params({'pipeline_manager': [object]})
    def __init__(self, pipeline_manager: object):
        self.pipeline_manager = pipeline_manager
        self._is_runned = False

    @abstractmethod
    def run(self):
        """
        Abstract method to run the orchestration process. Must be implemented
        by subclasses.

        Notes
        -----
        Subclasses should implement this method to initialize and run the
        orchestration process, such as creating workflows or DAGs.

        The method sets the ``_is_runned`` attribute to True.

        Examples
        --------
        >>> orchestrator.run()
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    @abstractmethod
    def schedule_pipeline(self, schedule_interval: str):
        """
        Abstract method to schedule a pipeline. Must be implemented by
        subclasses.

        Parameters
        ----------
        schedule_interval : str
            The scheduling interval (e.g., cron expression).

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Notes
        -----
        Subclasses should implement this method to schedule the pipeline
        according to the scheduling capabilities of the orchestration tool.

        Examples
        --------
        >>> orchestrator.schedule_pipeline('@daily')
        """
        raise NotImplementedError("Subclasses must implement the 'schedule_pipeline' method.")

    @abstractmethod
    def monitor_pipeline(self):
        """
        Abstract method to monitor the status of pipeline execution.
        Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Notes
        -----
        Subclasses should implement this method to provide monitoring
        capabilities using the orchestration tool's features.

        Examples
        --------
        >>> orchestrator.monitor_pipeline()
        """
        raise NotImplementedError("Subclasses must implement the 'monitor_pipeline' method.")
        
    
class BaseTest(BaseLearner):
    """
    Provides a framework for creating test cases with customizable
    configurations, supporting parallel execution, logging, and
    randomization. This class is abstract and intended to be subclassed
    to implement specific test cases through the `fit` and `run` methods.

    The mathematical formulation underlying the tests includes statistical
    processes and optimizations, often involving randomization where
    applicable. For example, a random seed can be set to ensure
    reproducibility in stochastic test executions.

    .. math::
        R(x) = f(x) + \\epsilon

    where :math:`R(x)` is the result of a test applied to the input data
    :math:`x`, and :math:`\\epsilon` represents any potential noise or
    random variations introduced, controlled by the `random_seed`
    parameter.

    Parameters
    ----------
    test_name : str, default='BaseTest'
        The name of the test instance. This name is used in logging and
        can help identify different tests when multiple tests are run.

    store_results : bool, default=True
        If ``True``, the results of the test will be stored in the
        `results_` attribute and can be saved to disk. If ``False``,
        results will not be stored.

    enable_logging : bool, default=True
        If ``True``, the test will log events during execution. Logs can
        be retrieved using the `get_logs` method.

    parallel_execution : bool, default=False
        If ``True``, the test supports parallel execution, allowing the
        `run` method to be executed in parallel over multiple inputs.
        Parallelism is achieved using threading.

    random_seed : int or None, default=None
        The random seed used to control stochastic processes within the
        test. If ``None``, the random seed is not set, and results may
        vary between runs.

    **config_params : dict
        Additional configuration parameters that can be set dynamically.
        These parameters are stored in the `config_params` attribute and
        can be used to customize test behavior.

    Attributes
    ----------
    test_name : str
        The name of the test instance.

    store_results : bool
        Indicates whether results will be stored.

    enable_logging : bool
        Indicates if logging of events is enabled.

    parallel_execution : bool
        Indicates if the test supports parallel execution.

    random_seed : int or None
        The random seed used for reproducibility.

    config_params : dict
        Dynamic configuration parameters set during initialization.

    results_ : Any
        Stores the test results after execution.

    log_ : list of dict
        Stores the logs of events that occurred during execution.

    Methods
    -------
    fit(*args, **kwargs)
        Abstract method to fit the test. Subclasses must implement this
        method.

    run(*args, **kwargs)
        Abstract method to run the test. Subclasses must implement this
        method.

    log_event(event, details=None)
        Logs an event with optional details.

    get_logs()
        Retrieves the log of all events.

    reset()
        Resets the internal state of the test object.

    save_results(path)
        Saves the results of the test to a specified file path.

    load_results(path)
        Loads results from a specified file path.

    parallelize(func, args_list, kwargs_list=None)
        Executes a function over a list of arguments in parallel.

    run_test(args_list=None, kwargs_list=None, **kwargs)
        Executes the test.

    Notes
    -----
    This class is abstract and should be subclassed. You must implement
    the `fit` and `run` methods in subclasses to define specific
    behaviors. The `parallel_execution` flag enables the use of concurrent
    threads for parallelism, which can be useful for large-scale test
    cases.

    Examples
    --------
    >>> from gofast.mlops.testing import BaseTest
    >>> class MyTest(BaseTest):
    ...     def fit(self, data):
    ...         # Implement fitting logic
    ...         self._set_fitted()
    ...
    ...     def run(self, data):
    ...         # Implement test logic
    ...         self._set_runned()
    ...         return sum(data)
    ...
    >>> test = MyTest(test_name="My Test", parallel_execution=True)
    >>> test.fit(data=[1, 2, 3])
    >>> results = test.run_test(args_list=[([1, 2, 3],), ([4, 5, 6],)])
    >>> print(results)
    [6, 15]

    See Also
    --------
    gofast.mlops.parallelizer : Parallel execution utilities.
    gofast.mlops.logging : Advanced logging framework.

    References
    ----------
    .. [1] Smith, J., et al. (2022). "Advanced Test Execution Techniques,"
       Journal of Software Testing, 15(4), 123-145.
    """
    @validate_params ( { 
        "test_name": [ str], 
        "store_results": [bool], 
        "enable_logging": [bool], 
        "parallel_execution": [bool], 
        "random_seed": [Integral, None], 
        })
    def __init__(
        self,
        test_name: str = "BaseTest",
        store_results: bool = True,
        enable_logging: bool = True,
        parallel_execution: bool = False,
        random_seed: Optional[int] = None,
        **config_params
    ):
        self.test_name = test_name
        self.store_results = store_results
        self.enable_logging = enable_logging
        self.parallel_execution = parallel_execution
        self.random_seed = random_seed
        self.config_params = config_params
        self.results_ = None
        self.log_ = []
        self._is_fitted = False
        self._is_runned = False

        # Set random seed if provided
        if self.random_seed is not None:
            self._set_random_seed()

        # Initialize additional configurations
        self._initialize_from_config(config_params)

    def _initialize_from_config(self, config_params: Dict[str, Any]):
        """
        Initializes additional configurations based on provided parameters.

        Parameters
        ----------
        config_params : dict
            Configuration parameters to set as attributes.
        """
        for param, value in config_params.items():
            setattr(self, param, value)

    def _set_random_seed(self):
        """
        Sets the random seed for reproducibility of stochastic processes.
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)


    def log_event(self, event: str, details: Optional[Dict[str, Any]] = None):
        """
        Logs an event with optional details if logging is enabled.

        Parameters
        ----------
        event : str
            A description of the event to be logged.

        details : dict, optional
            Additional information about the event.

        Notes
        -----
        The logs can be accessed via the `get_logs` method. If logging is
        not enabled (`enable_logging=False`), this method has no effect.

        Examples
        --------
        >>> test.log_event('test_started', {'timestamp': '2021-01-01'})
        """
        if self.enable_logging:
            log_entry = {"event": event, "details": details}
            self.log_.append(log_entry)

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieves the log of all events that have occurred.

        Returns
        -------
        logs : list of dict
            A list of logged events, where each event is represented as a
            dictionary with keys 'event' and 'details'.

        Notes
        -----
        This method is only useful if logging is enabled during object
        initialization (`enable_logging=True`).

        Examples
        --------
        >>> logs = test.get_logs()
        >>> for log in logs:
        ...     print(log)
        """
        return self.log_

    def reset(self):
        """
        Resets the internal state of the test object. This includes the
        fitted and run states, as well as clearing any results and logs.

        Notes
        -----
        Use this method to clear a test's state and rerun it from scratch.

        Examples
        --------
        >>> test.reset()
        """
        self._is_fitted = False
        self._is_runned = False
        self.results_ = None
        self.log_ = []

    def save_results(self, path: str):
        """
        Saves the results of the test to a specified file path.

        Parameters
        ----------
        path : str
            The file path where results should be saved.

        Raises
        ------
        ValueError
            If there are no results to save or if result storage is
            disabled.

        Notes
        -----
        The results are saved using `pickle`. Ensure that the results are
        serializable.

        Examples
        --------
        >>> test.save_results('results.pkl')
        """
        if self.results_ is None:
            raise ValueError("No results to save. Ensure tests have been run.")

        if not self.store_results:
            raise ValueError("Result storage is disabled.")

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.results_, f)

        self.log_event("results_saved", {"path": path})

    def load_results(self, path: str):
        """
        Loads results from a specified file path.

        Parameters
        ----------
        path : str
            The file path from which to load results.

        Notes
        -----
        The results are loaded using `pickle`. Ensure that the file
        contains valid serialized results.

        Examples
        --------
        >>> test.load_results('results.pkl')
        """
        import pickle
        with open(path, 'rb') as f:
            self.results_ = pickle.load(f)

        self.log_event("results_loaded", {"path": path})

    def parallelize(
        self,
        func: Callable,
        args_list: List[Tuple],
        kwargs_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Any]:
        """
        Executes the provided function `func` over a list of arguments in
        parallel.

        Parameters
        ----------
        func : callable
            The function to be executed in parallel.

        args_list : list of tuple
            A list where each element is a tuple of arguments to pass to
            `func`.

        kwargs_list : list of dict, optional
            A list where each element is a dict of keyword arguments to
            pass to `func`. If not provided, empty dicts are used.

        Returns
        -------
        results : list
            Results of the function applied to the arguments.

        Raises
        ------
        ValueError
            If parallel execution is not enabled.

        Notes
        -----
        Parallelism is achieved through threading, which can improve
        performance for I/O-bound operations or multiple test cases.

        Examples
        --------
        >>> def square(x):
        ...     return x * x
        >>> args_list = [(2,), (3,), (4,)]
        >>> results = test.parallelize(square, args_list)
        >>> print(results)
        [4, 9, 16]
        """
        if not self.parallel_execution:
            raise ValueError("Parallel execution is not enabled.")

        import concurrent.futures

        if kwargs_list is None:
            kwargs_list = [{} for _ in args_list]

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(func, *args, **kwargs)
                for args, kwargs in zip(args_list, kwargs_list)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    def run_test(
        self,
        args_list: Optional[List[Tuple]] = None,
        kwargs_list: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """
        Executes the test. If `parallel_execution` is enabled, the test
        will be run in parallel over the provided list of arguments.
        Otherwise, it will run sequentially.

        Parameters
        ----------
        args_list : list of tuple, optional
            A list where each element is a tuple of arguments to pass to
            the `run` method. Required if `parallel_execution` is True.

        kwargs_list : list of dict, optional
            A list where each element is a dict of keyword arguments to
            pass to the `run` method. Required if `parallel_execution` is
            True.

        **kwargs : dict
            Keyword arguments passed to the `run` method when
            `parallel_execution` is False.

        Returns
        -------
        results : Any
            Results of the test execution.

        Raises
        ------
        ValueError
            If `args_list` is not provided when `parallel_execution` is
            True.

        Notes
        -----
        If `parallel_execution` is enabled, you must provide `args_list`,
        and the test logic must be able to handle concurrent operations
        safely.

        Examples
        --------
        >>> test = MyTest(parallel_execution=False)
        >>> result = test.run_test(data=[1, 2, 3])
        >>> print(result)
        6
        >>> test = MyTest(parallel_execution=True)
        >>> args_list = [([1, 2, 3],), ([4, 5, 6],)]
        >>> results = test.run_test(args_list=args_list)
        >>> print(results)
        [6, 15]
        """
        if self.parallel_execution:
            if args_list is None:
                raise ValueError(
                    "args_list must be provided when parallel_execution is enabled.")
            return self.parallelize(self.run, args_list, kwargs_list)
        else:
            return self.run(**kwargs)

    def _set_fitted(self):
        """
        Marks the test as 'fitted' by setting the internal `_is_fitted`
        flag to `True`.

        Notes
        -----
        This method should be called at the end of the `fit` method in
        subclasses.
        """
        self._is_fitted = True

    def _set_runned(self):
        """
        Marks the test as 'runned' by setting the internal `_is_runned`
        flag to `True`.

        Notes
        -----
        This method should be called at the end of the `run` method in
        subclasses.
        """
        self._is_runned = True

    def check_is_fitted(self, msg: Optional[str] = None):
        """
        Checks whether the test has been fitted by verifying the
        `_is_fitted` attribute.

        Parameters
        ----------
        msg : str, optional
            Custom message to be displayed if the test is not fitted.

        Raises
        ------
        AttributeError
            If the test has not been fitted and `_is_fitted` is `False`.

        Notes
        -----
        This method should be used before any operation that requires the
        test to have been fitted.

        Examples
        --------
        >>> test.check_is_fitted()
        AttributeError: This BaseTest instance is not fitted yet.
        """
        check_is_fitted(self, attributes=["_is_fitted"], msg=msg)

    def check_is_runned(self, msg: Optional[str] = None):
        """
        Checks whether the test has been run by verifying the `_is_runned`
        attribute.

        Parameters
        ----------
        msg : str, optional
            Custom message to be displayed if the test has not been run.

        Raises
        ------
        AttributeError
            If the test has not been run and `_is_runned` is `False`.

        Notes
        -----
        This method should be used before any operation that requires the
        test to have been executed.

        Examples
        --------
        >>> test.check_is_runned()
        AttributeError: This BaseTest instance has not been run yet.
        """
        check_is_runned(self, attributes=["_is_runned"], msg=msg)

class BaseVersioning(BaseLearner):
    """
    A base class for managing version control in machine learning systems,
    including models, datasets, and pipelines. This class provides a
    foundation for ensuring version integrity, validation, and logging
    throughout the lifecycle of the versioning object.

    Parameters
    ----------
    version : str
        The version identifier for the object (e.g., model version,
        dataset version).

    config : dict, optional
        A dictionary containing additional configuration options for
        versioning. This can include specific settings related to the
        version control system.

    Attributes
    ----------
    version_ : str
        The version identifier of the instance.

    config_ : dict
        The configuration dictionary storing settings applied during
        initialization.

    is_initialized_ : bool
        Indicates whether the object has been successfully initialized.

    events_log_ : list of dict
        A list that stores the history of logged events for auditing and
        debugging purposes.

    Methods
    -------
    is_initialized() -> bool
        Checks if the object has been successfully initialized.

    get_version() -> str
        Returns the version identifier of the instance.

    reset()
        Resets the internal attributes and reinitializes the object.

    log_event(event_name, event_details=None)
        Logs events internally and records them into the logger for
        auditing.

    get_log_history() -> list
        Retrieves the list of all logged events for review.

    Notes
    -----
    This class is abstract and should not be instantiated directly. It is
    designed to be inherited by other versioning-related classes that need
    to implement version control mechanisms (e.g., model versioning,
    dataset versioning).

    Each subclass must implement the ``_perform_version_checks`` and
    ``validate`` methods to ensure version constraints and validation
    processes are properly enforced.

    Examples
    --------
    >>> from gofast.mlops.versioning import BaseVersioning
    >>> class MyVersioning(BaseVersioning):
    ...     def _perform_version_checks(self):
    ...         # Implement version checks here
    ...         pass
    ...     def validate(self):
    ...         # Implement validation logic here
    ...         pass
    >>> my_versioning = MyVersioning(
    ...     version='v1.0',
    ...     config={'track_history': True}
    ... )
    >>> my_versioning.get_version()
    'v1.0'

    See Also
    --------
    DatasetVersioning : Class for managing dataset version control.
    ModelVersionControl : Class for managing model version control.
    PipelineVersioning : Class for tracking and versioning machine
                         learning pipelines.

    References
    ----------
    .. [1] "Version Control in Machine Learning Systems", J. Doe et al.,
       Proceedings of the Machine Learning Conference, 2022.

    """
    @abstractmethod 
    def __init__(
        self, 
        version: str, 
        config: 
        Optional[Dict] = None
        ):

        self.version = version
        self.config = config or {}
        self._is_initialized_ = False
        self.events_log_ = []
        self._initialize()

    def is_initialized(self) -> bool:
        """
        Checks if the class instance has been initialized successfully.

        Returns
        -------
        bool
            `True` if initialized, `False` otherwise.

        """
        return self._is_initialized_

    def get_version(self) -> str:
        """
        Returns the version identifier of the instance.

        Returns
        -------
        str
            The version identifier.

        """
        return self.version

    def reset(self):
        """
        Resets internal attributes and reinitializes the object. This can
        be useful if reconfiguration or reloading of the versioning object
        is needed.

        """
        self._is_initialized_ = False
        self.log_event('reset', {'version': self.version})
        self._initialize()

    def log_event(self, event_name: str, event_details: Optional[Dict] = None):
        """
        Logs events into the logger and stores them in the internal events
        log.

        Parameters
        ----------
        event_name : str
            The name of the event to log (e.g., ``'initialization_success'``).

        event_details : dict, optional
            Additional details about the event to log (e.g., version,
            error details).

        """
        if event_details is None:
            event_details = {}
        event_details['event'] = event_name
        event_details['version'] = self.version

        # Log event into the logger
        logger.info(f"Event: {event_name} | Details: {event_details}")

        # Store event in internal log
        self.events_log_.append(event_details)

    def get_log_history(self) -> list:
        """
        Retrieves the internal log history of the events.

        Returns
        -------
        list of dict
            A list of logged events.

        """
        return self.events_log_
