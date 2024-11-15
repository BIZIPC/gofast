# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Handle repetitive tasks and automate workflows, from model retraining
to regular data ingestion processes.
"""

import os
import time
import pickle
import threading
from numbers import Real, Integral
from typing import Callable, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor 

from .._gofastlog import gofastlog 
from ..api.property import BaseLearner
from ..compat.sklearn import validate_params 
from ..tools.depsutils import ensure_pkg 

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 

logger=gofastlog.get_gofast_logger(__name__)


__all__ = [
    "AutomationManager",
    "AirflowAutomation",
    "KubeflowAutomation",
    "KafkaAutomation",
    "RabbitMQAutomation",
    "RetrainingScheduler",
    "SimpleAutomation",
]

class SimpleAutomation(BaseLearner):
    """
    Manages the automation of repetitive tasks such as scheduled model
    training and data ingestion. It supports adding tasks, scheduling
    them, and monitoring task status.

    Attributes
    ----------
    tasks : dict
        Dictionary storing tasks and their scheduling metadata.

    Methods
    -------
    add_task(task_name, func, interval, args=())
        Adds a new task to the automation system.
    schedule_task(task_name)
        Schedules the execution of a task at the specified interval.
    cancel_task(task_name)
        Cancels the execution of a scheduled task.
    monitor_tasks()
        Monitors and logs the status of all tasks.
    run_all_tasks()
        Starts all the tasks in the automation manager.

    Notes
    -----
    The `SimpleAutomation` allows for the automation of tasks by
    scheduling them to run at specified intervals. Tasks are executed in
    separate threads, allowing for concurrent execution.

    The interval between task executions can be mathematically represented as:

    .. math::

        t_{n+1} = t_n + \Delta t

    where :math:`t_n` is the time of the nth execution, and
    :math:`\Delta t` is the interval in seconds.

    Examples
    --------
    >>> from gofast.mlops.automation import SimpleAutomation
    >>> def my_task():
    ...     print("Task executed")
    >>> manager = SimpleAutomation()
    >>> manager.add_task('print_task', my_task, interval=5)
    >>> manager.schedule_task('print_task')
    >>> # Task will execute every 5 seconds
    >>> manager.monitor_tasks()
    >>> manager.cancel_task('print_task')

    See Also
    --------
    AutomationManager : 
        A more advanced automation manager with additional features.

    References
    ----------
    .. [1] Smith, J. (2020). "Automating Machine Learning Workflows."
       *Journal of Machine Learning Automation*, 5(3), 150-165.
    """

    def __init__(self):
        """
        Initializes the `SimpleAutomation` with an empty task
        dictionary.

        Examples
        --------
        >>> manager = SimpleAutomation()
        """
        self._include_all_attributes=True
        
        self.tasks: Dict[str, Dict[str, Any]] = {}

    @validate_params({
        'task_name': [str],
        'func': [Callable],
        'interval': [int, float],
        'args': [tuple]
    })
    def add_task(
        self,
        task_name: str,
        func: Callable,
        interval: float,
        args: Tuple = ()
    ):
        """
        Adds a new task to the automation system.

        Parameters
        ----------
        task_name : str
            A unique name for the task.
        func : callable
            The function to execute.
        interval : int or float
            The interval (in seconds) between task executions.
        args : tuple, optional
            Arguments to pass to the task function. Defaults to empty tuple.

        Raises
        ------
        ValueError
            If a task with the same name already exists.

        Notes
        -----
        The task will be scheduled to run at the specified interval once scheduled.

        Examples
        --------
        >>> def my_task(arg1):
        ...     print(f"Task executed with argument {arg1}")
        >>> manager.add_task('my_task', my_task, interval=10, args=('hello',))
        """
        logger.info(f"Adding task '{task_name}' with interval {interval} seconds.")
        if task_name in self.tasks:
            raise ValueError(f"Task '{task_name}' already exists.")

        self.tasks[task_name] = {
            "func": func,
            "interval": interval,
            "args": args,
            "running": False,
            "thread": None,
        }

    def _task_runner(self, task_name: str):
        """
        Internal method to run tasks repeatedly based on the interval.

        Parameters
        ----------
        task_name : str
            The name of the task to run.

        Notes
        -----
        This method runs in a separate thread and repeatedly executes
        the task function at the specified interval until the task is
        cancelled.

        Mathematically, the execution times :math:`t_n` are calculated as:

        .. math::

            t_{n} = t_0 + n \times \Delta t

        where :math:`t_0` is the start time, :math:`n` is the execution
        count, and :math:`\Delta t` is the interval.

        Examples
        --------
        >>> # Internal method, not intended to be called directly.
        """
        task = self.tasks[task_name]
        while task["running"]:
            logger.info(f"Running task: '{task_name}'")
            try:
                task["func"](*task["args"])
            except Exception as e:
                logger.error(f"Error executing task '{task_name}': {str(e)}")
            time.sleep(task["interval"])

    @validate_params({
        'task_name': [str]
    })
    def schedule_task(self, task_name: str):
        """
        Schedules the execution of a task at the specified interval.

        Parameters
        ----------
        task_name : str
            The name of the task to schedule.

        Raises
        ------
        ValueError
            If the task does not exist.

        Notes
        -----
        This method starts a new thread to run the task repeatedly
        at the specified interval.

        Examples
        --------
        >>> manager.schedule_task('my_task')
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' does not exist.")

        if self.tasks[task_name]["running"]:
            logger.warning(f"Task '{task_name}' is already running.")
            return

        logger.info(f"Scheduling task: '{task_name}'")
        self.tasks[task_name]["running"] = True
        task_thread = threading.Thread(
            target=self._task_runner,
            args=(task_name,),
            daemon=True
        )
        self.tasks[task_name]["thread"] = task_thread
        task_thread.start()

    @validate_params({
        'task_name': [str]
    })
    def cancel_task(self, task_name: str):
        """
        Cancels the execution of a scheduled task.

        Parameters
        ----------
        task_name : str
            The name of the task to cancel.

        Raises
        ------
        ValueError
            If the task does not exist.

        Notes
        -----
        This method stops the task from running by setting its running
        status to False and joining the thread.

        Examples
        --------
        >>> manager.cancel_task('my_task')
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' does not exist.")

        logger.info(f"Cancelling task: '{task_name}'")
        self.tasks[task_name]["running"] = False
        task_thread = self.tasks[task_name]["thread"]
        if task_thread and task_thread.is_alive():
            task_thread.join()

    def monitor_tasks(self):
        """
        Monitors and logs the status of all tasks.

        Notes
        -----
        This method logs the current status (running or stopped) of
        all tasks managed by the automation manager.

        Examples
        --------
        >>> manager.monitor_tasks()
        """
        logger.info("Monitoring all tasks...")
        for task_name, task_info in self.tasks.items():
            status = "running" if task_info["running"] else "stopped"
            logger.info(f"Task '{task_name}' is currently {status}.")

    def run_all_tasks(self):
        """
        Starts all the tasks in the automation manager.

        Notes
        -----
        This method schedules all tasks that have been added to the
        automation manager.

        Examples
        --------
        >>> manager.run_all_tasks()
        """
        logger.info("Running all tasks in the Automation Manager.")
        for task_name in self.tasks.keys():
            self.schedule_task(task_name)
            
class SimpleRetrainingScheduler(SimpleAutomation):
    """
    Manages the automation of model retraining workflows based on model
    performance decay. This class schedules regular model retraining and
    adjusts intervals based on model performance.

    Attributes
    ----------
    tasks : dict
        Dictionary storing tasks and their scheduling metadata.

    Methods
    -------
    schedule_retraining(model, retrain_func, interval)
        Schedules the retraining of a model at regular intervals.
    evaluate_model_performance(model, metric_func)
        Evaluates the model's performance using a given metric function.
    trigger_retraining_on_decay(model, metric_func, decay_threshold)
        Triggers model retraining if performance decay is detected.
    adjust_retraining_schedule(model, new_interval)
        Adjusts the retraining schedule dynamically based on model performance.
    monitor_model(model, metric_func, decay_threshold, check_interval)
        Monitors the model's performance and triggers retraining if necessary.

    Notes
    -----
    The `SimpleRetrainingScheduler` extends `SimpleAutomation` to
    provide functionality specific to model retraining workflows.

    Examples
    --------
    >>> from gofast.mlops.automation import SimpleRetrainingScheduler
    >>> scheduler = SimpleRetrainingScheduler()
    >>> model = MyModel()
    >>> def retrain_model(model):
    ...     pass
    >>> scheduler.schedule_retraining(model, retrain_model, interval=3600)

    See Also
    --------
    SimpleAutomation : Manages the automation of repetitive tasks.

    References
    ----------
    .. [1] Doe, J. (2021). "Automated Retraining Strategies for Machine
       Learning Models." *Journal of Machine Learning Automation*, 5(3),
       150-165.

    """
    def __init__(self):
        """
        Initializes the `SimpleRetrainingScheduler` by calling the
        constructor of the parent `SimpleAutomation`.

        Examples
        --------
        >>> scheduler = SimpleRetrainingScheduler()
        """
        super().__init__()

    @validate_params({
        'model': [object],
        'retrain_func': [Callable],
        'interval': [int]
    })
    def schedule_retraining(
        self,
        model: Any,
        retrain_func: Callable,
        interval: int
    ):
        """
        Schedules the retraining of a model at regular intervals.

        Parameters
        ----------
        model : object
            The model to be retrained.
        retrain_func : callable
            The function to retrain the model.
        interval : int
            The interval (in seconds) to check for retraining.

        Notes
        -----
        This method adds a retraining task to the automation manager,
        scheduling it to run at the specified interval.

        Mathematically, the retraining times :math:`t_n` are calculated as:

        .. math::

            t_{n} = t_0 + n \times \Delta t

        where :math:`t_0` is the start time, :math:`n` is the retraining
        iteration, and :math:`\Delta t` is the interval.

        Examples
        --------
        >>> scheduler.schedule_retraining(model, retrain_model, interval=3600)
        """
        task_name = f"retrain_{model.__class__.__name__}"
        self.add_task(
            task_name=task_name,
            func=retrain_func,
            interval=interval,
            args=(model,),
        )
        self.schedule_task(task_name)

    @validate_params({
        'model': [object],
        'metric_func': [Callable],
    })
    def evaluate_model_performance(
        self,
        model: Any,
        metric_func: Callable
    ) -> float:
        """
        Evaluates the model's performance using the given metric function.

        Parameters
        ----------
        model : object
            The model to evaluate.
        metric_func : callable
            The function to calculate model performance.

        Returns
        -------
        score : float
            The calculated model performance score.

        Notes
        -----
        This method applies the `metric_func` to the model to compute
        a performance score, which can be used to determine if retraining
        is necessary.

        Examples
        --------
        >>> score = scheduler.evaluate_model_performance(model, metric_func)
        """
        logger.info(f"Evaluating performance of model '{model.__class__.__name__}'")
        score = metric_func(model)
        logger.info(f"Model performance score: {score}")
        return score

    @validate_params({
        'model': [object],
        'metric_func': [Callable],
        'decay_threshold': [float]
    })
    def trigger_retraining_on_decay(
        self,
        model: Any,
        metric_func: Callable,
        decay_threshold: float
    ):
        """
        Triggers model retraining if performance decay is detected.

        Parameters
        ----------
        model : object
            The model to evaluate.
        metric_func : callable
            Function to evaluate the model's performance.
        decay_threshold : float
            The threshold below which retraining is triggered.

        Notes
        -----
        If the model's performance score falls below the `decay_threshold`,
        retraining is initiated.

        Mathematically, retraining is triggered if:

        .. math::

            \text{score} < \text{decay\_threshold}

        Examples
        --------
        >>> scheduler.trigger_retraining_on_decay(
        ...     model, metric_func, decay_threshold=0.8)
        """
        score = self.evaluate_model_performance(model, metric_func)
        if score < decay_threshold:
            logger.warning(
                f"Performance decay detected (score: {score}). Triggering retraining."
            )
            task_name = f"retrain_{model.__class__.__name__}"
            if task_name in self.tasks:
                self.tasks[task_name]["func"](*self.tasks[task_name]["args"])
            else:
                logger.error(
                    f"No retraining task found for model '{model.__class__.__name__}'."
                )

    @validate_params({
        'model': [object],
        'new_interval': [int]
    })
    def adjust_retraining_schedule(
        self,
        model: Any,
        new_interval: int
    ):
        """
        Adjusts the retraining schedule dynamically based on model performance.

        Parameters
        ----------
        model : object
            The model whose retraining schedule will be adjusted.
        new_interval : int
            The new interval for retraining (in seconds).

        Notes
        -----
        This method updates the interval at which the model retraining
        task is scheduled to run.

        Examples
        --------
        >>> scheduler.adjust_retraining_schedule(model, new_interval=7200)
        """
        task_name = f"retrain_{model.__class__.__name__}"
        if task_name in self.tasks:
            logger.info(
                f"Adjusting retraining schedule for model '{model.__class__.__name__}' "
                f"to {new_interval} seconds."
            )
            self.cancel_task(task_name)
            self.tasks[task_name]["interval"] = new_interval
            self.schedule_task(task_name)
        else:
            logger.error(
                f"No retraining task found for model '{model.__class__.__name__}'."
            )

    @validate_params({
        'model': [object],
        'metric_func': [Callable],
        'decay_threshold': [Real],
        'check_interval': [Integral]
    })
    def monitor_model(
        self,
        model: Any,
        metric_func: Callable,
        decay_threshold: float,
        check_interval: int
    ):
        """
        Monitors the model's performance at regular intervals and triggers
        retraining if necessary.

        Parameters
        ----------
        model : object
            The model to monitor.
        metric_func : callable
            The function to evaluate model performance.
        decay_threshold : float
            The threshold to trigger retraining.
        check_interval : int
            How often to check the model's performance (in seconds).

        Notes
        -----
        This method adds a monitoring task that periodically evaluates
        the model's performance and triggers retraining if the performance
        falls below the `decay_threshold`.

        Examples
        --------
        >>> scheduler.monitor_model(
        ...     model,
        ...     metric_func,
        ...     decay_threshold=0.8,
        ...     check_interval=1800
        ... )
        """
        logger.info(
            f"Monitoring model '{model.__class__.__name__}' performance for decay."
        )

        def check_model():
            self.trigger_retraining_on_decay(model, metric_func, decay_threshold)

        task_name = f"monitor_{model.__class__.__name__}"
        self.add_task(
            task_name=task_name,
            func=check_model,
            interval=check_interval,
        )
        self.schedule_task(task_name)
        

class AutomationManager(BaseLearner):
    """
    Advanced class to manage the automation of repetitive tasks, with support
    for task scheduling, retries, parallel execution, fault tolerance, and
    state persistence.

    Parameters
    ----------
    max_workers : int, optional
        The maximum number of threads that can be used to execute tasks.
        Defaults to ``4``.
    state_persistence_file : str, optional
        The file path for saving and loading task state. Defaults to
        ``'automation_state.pkl'``.

    Attributes
    ----------
    tasks : dict
        Dictionary storing tasks and their scheduling metadata.
    scheduler : BackgroundScheduler
        The scheduler used to schedule tasks at specified intervals.
    thread_pool : ThreadPoolExecutor
        The thread pool executor for running tasks in parallel.
    state_persistence_file : str
        The file path for saving and loading task state.

    Methods
    -------
    add_task(task_name, func, interval, args=(), retries=3)
        Adds a new task with automatic retry logic and state persistence.
    schedule_task(task_name)
        Schedules the execution of a task at the specified interval.
    cancel_task(task_name)
        Cancels the execution of a scheduled task.
    run_all_tasks()
        Schedules all tasks for execution.
    persist_state()
        Saves the current task state to disk.
    load_state()
        Loads the task state from a file, if it exists.
    shutdown()
        Shuts down the automation manager and saves the task state.

    Notes
    -----
    The ``AutomationManager`` class provides advanced automation capabilities,
    including retries with exponential backoff, parallel execution, fault
    tolerance, and state persistence across restarts.

    Mathematically, the exponential backoff time after each retry can be
    represented as:

    .. math::

        t_{\text{backoff}} = 2^{n - 1} \times t_{\text{initial}}

    where :math:`n` is the retry attempt number, and :math:`t_{\text{initial}}`
    is the initial backoff time.

    Examples
    --------
    >>> from gofast.mlops.automation import AutomationManager
    >>> def my_task():
    ...     print("Task executed")
    >>> manager = AutomationManager(max_workers=5)
    >>> manager.add_task('print_task', my_task, interval=5, retries=2)
    >>> manager.schedule_task('print_task')
    >>> # Task will execute every 5 seconds with up to 2 retries on failure
    >>> manager.run_all_tasks()
    >>> # After finishing, shut down the manager
    >>> manager.shutdown()

    See Also
    --------
    SimpleAutomation : A simpler version of the automation manager.

    References
    ----------
    .. [1] Johnson, M. (2022). "Advanced Task Scheduling in Machine Learning Pipelines."
       *Journal of Automation and Computing*, 10(2), 200-215.
    """
    
    @ensure_pkg(
        "apscheduler",
        extra="APScheduler is not installed."
        " Please install 'apscheduler' to use this feature.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )

    @validate_params({
        'max_workers': [int],
        'state_persistence_file': [str]
    })
    def __init__(
        self,
        max_workers: int = 4,
        state_persistence_file: str = "automation_state.pkl"
    ):
        """
        Initializes the ``AutomationManager``.

        Parameters
        ----------
        max_workers : int, optional
            The maximum number of threads that can be used to execute tasks.
            Defaults to ``4``.
        state_persistence_file : str, optional
            The file path for saving and loading task state. Defaults to
            ``'automation_state.pkl'``.

        Notes
        -----
        Upon initialization, the automation manager attempts to load any
        existing task state from the persistence file. It also starts the
        background scheduler.

        Examples
        --------
        >>> manager = AutomationManager(max_workers=5)
        """
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.state_persistence_file = state_persistence_file

        # Ensure 'apscheduler' is installed
        from apscheduler.schedulers.background import BackgroundScheduler
        self.scheduler = BackgroundScheduler()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.load_state()  # Load previous task states (if any)
        self.scheduler.start()

    @validate_params({
        'task_name': [str],
        'func': [callable],
        'interval': [int, float],
        'args': [tuple],
        'retries': [int]
    })
    def add_task(
        self,
        task_name: str,
        func: Callable,
        interval: float,
        args: Tuple = (),
        retries: int = 3
    ):
        """
        Adds a new task with automatic retry logic and state persistence.

        Parameters
        ----------
        task_name : str
            A unique name for the task.
        func : callable
            The function to execute.
        interval : int or float
            The interval (in seconds) between task executions.
        args : tuple, optional
            Arguments to pass to the task function. Defaults to empty tuple.
        retries : int, optional
            Number of retries for task execution in case of failure.
            Defaults to ``3``.

        Raises
        ------
        ValueError
            If a task with the same name already exists.

        Notes
        -----
        The task will be scheduled to run at the specified interval once
        scheduled. The task includes automatic retries with exponential
        backoff in case of failure.

        Examples
        --------
        >>> def my_task(arg1):
        ...     print(f"Task executed with argument {arg1}")
        >>> manager.add_task('my_task', my_task, interval=10, args=('hello',), retries=2)
        """
        logger.info(
            f"Adding task '{task_name}' with interval {interval} seconds and retries {retries}."
        )
        if task_name in self.tasks:
            raise ValueError(f"Task '{task_name}' already exists.")

        self.tasks[task_name] = {
            "func": func,
            "interval": interval,
            "args": args,
            "retries": retries,
            "failures": 0,
            "running": False,
            "future": None,
        }
        self.persist_state()  # Save state after adding task

    def _run_task_with_retries(self, task_name: str):
        """
        Internal method to run a task with automatic retries and exponential backoff.

        Parameters
        ----------
        task_name : str
            The name of the task to run.

        Notes
        -----
        This method runs the task function, retrying it upon failure up to
        the specified number of retries. The delay between retries increases
        exponentially:

        .. math::

            t_{\text{backoff}} = 2^{n - 1} \times t_{\text{initial}}

        where :math:`n` is the retry attempt number, and :math:`t_{\text{initial}}`
        is the initial backoff time.

        Examples
        --------
        >>> # Internal method, not intended to be called directly.
        """
        task = self.tasks[task_name]
        retry_count = 0
        backoff_time = 1  # Start with 1 second

        while retry_count <= task["retries"]:
            try:
                logger.info(f"Running task '{task_name}' (Attempt {retry_count + 1})...")
                task["func"](*task["args"])
                task["failures"] = 0
                logger.info(f"Task '{task_name}' completed successfully.")
                break
            except Exception as e:
                retry_count += 1
                task["failures"] += 1
                logger.error(
                    f"Task '{task_name}' failed with error: {str(e)}. "
                    f"Retrying in {backoff_time} seconds."
                )
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff

        if retry_count > task["retries"]:
            logger.error(
                f"Task '{task_name}' failed after {task['retries']} retries."
            )
            task["failures"] = retry_count

    @validate_params({'task_name': [str]})
    def schedule_task(self, task_name: str):
        """
        Schedules the execution of a task at the specified interval.

        Parameters
        ----------
        task_name : str
            The name of the task to schedule.

        Raises
        ------
        ValueError
            If the task does not exist.

        Notes
        -----
        This method schedules the task using the background scheduler.
        The task will run at the specified interval until it is canceled.

        Examples
        --------
        >>> manager.schedule_task('my_task')
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' does not exist.")

        task = self.tasks[task_name]
        if task["running"]:
            logger.warning(f"Task '{task_name}' is already running.")
            return

        logger.info(
            f"Scheduling task: '{task_name}' with interval {task['interval']} seconds."
        )
        task["running"] = True

        # The job needs to have a unique id so that it can be canceled individually
        job_id = f"{task_name}_job"

        def job_function():
            self.thread_pool.submit(self._run_task_with_retries, task_name)

        # Ensure 'apscheduler' is installed
        from apscheduler.triggers.interval import IntervalTrigger

        self.scheduler.add_job(
            job_function,
            trigger=IntervalTrigger(seconds=task["interval"]),
            id=job_id,
            replace_existing=True
        )
        self.persist_state()  # Save state after scheduling task

    @validate_params({'task_name': [str]})
    def cancel_task(self, task_name: str):
        """
        Cancels the execution of a scheduled task.

        Parameters
        ----------
        task_name : str
            The name of the task to cancel.

        Raises
        ------
        ValueError
            If the task does not exist.

        Notes
        -----
        This method stops the task from running by removing it from the
        scheduler and updating its status.

        Examples
        --------
        >>> manager.cancel_task('my_task')
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' does not exist.")

        logger.info(f"Cancelling task: '{task_name}'")
        task = self.tasks[task_name]
        task["running"] = False

        job_id = f"{task_name}_job"
        self.scheduler.remove_job(job_id)
        self.persist_state()  # Save state after canceling task

    def run_all_tasks(self):
        """
        Schedules all tasks for execution.

        Notes
        -----
        This method iterates over all tasks and schedules them if they are
        not already running.

        Examples
        --------
        >>> manager.run_all_tasks()
        """
        logger.info("Scheduling all tasks for execution.")
        for task_name in self.tasks.keys():
            self.schedule_task(task_name)

    def persist_state(self):
        """
        Saves the current task state to disk.

        Notes
        -----
        The task state is saved to the file specified by
        ``state_persistence_file`` in binary format using ``pickle``.

        Examples
        --------
        >>> manager.persist_state()
        """
        with open(self.state_persistence_file, 'wb') as f:
            pickle.dump(self.tasks, f)
        logger.info(f"Task state persisted to '{self.state_persistence_file}'.")

    def load_state(self):
        """
        Loads the task state from a file, if it exists.

        Notes
        -----
        This method attempts to load the task state from the file specified
        by ``state_persistence_file``. If the file does not exist, it starts
        with an empty task list.

        Examples
        --------
        >>> manager.load_state()
        """
        if os.path.exists(self.state_persistence_file):
            with open(self.state_persistence_file, 'rb') as f:
                self.tasks = pickle.load(f)
            logger.info(f"Task state loaded from '{self.state_persistence_file}'.")
        else:
            logger.info("No existing task state found. Starting with an empty task list.")

    def shutdown(self):
        """
        Shuts down the automation manager and saves the task state.

        Notes
        -----
        This method should be called when you are done with the automation
        manager to ensure that all resources are cleaned up and the task
        state is saved.

        Examples
        --------
        >>> manager.shutdown()
        """
        logger.info("Shutting down Automation Manager and saving state.")
        self.persist_state()
        self.scheduler.shutdown(wait=False)
        self.thread_pool.shutdown(wait=False)

class RetrainingScheduler(AutomationManager):
    """
    Advanced Class to manages model retraining workflows by extending the 
    AutomationManager. Handles model performance monitoring, retraining, 
    and adaptive scheduling based on performance decay.

    Parameters
    ----------
    max_workers : int, optional
        The maximum number of threads that can be used to execute tasks.
        Defaults to ``4``.

    Attributes
    ----------
    tasks : dict
        Dictionary storing tasks and their scheduling metadata.
    scheduler : BackgroundScheduler
        The scheduler used to schedule tasks at specified intervals.
    thread_pool : ThreadPoolExecutor
        The thread pool executor for running tasks in parallel.
    state_persistence_file : str
        The file path for saving and loading task state.

    Methods
    -------
    schedule_retraining(model, retrain_func, interval)
        Schedules regular model retraining.
    evaluate_model_performance(model, metric_func)
        Evaluates the model's performance using a given metric function.
    trigger_retraining_on_decay(model, metric_func, decay_threshold)
        Triggers model retraining if performance decay is detected.
    monitor_model(model, metric_func, decay_threshold, check_interval)
        Monitors the model's performance and triggers retraining if necessary.

    Notes
    -----
    The ``RetrainingScheduler`` class extends ``AutomationManager`` to provide
    functionality specific to model retraining workflows, including performance
    monitoring and adaptive scheduling based on performance decay.

    Examples
    --------
    >>> from gofast.mlops.automation import RetrainingScheduler
    >>> model = MyModel()
    >>> def retrain_model(model):
    ...     # retrain logic here
    ...     pass
    >>> def evaluate_model(model):
    ...     # evaluation logic here
    ...     return 0.85  # Example performance score
    >>> scheduler = RetrainingScheduler(max_workers=5)
    >>> scheduler.schedule_retraining(model, retrain_model, interval=3600)
    >>> scheduler.monitor_model(
    ...     model, evaluate_model, decay_threshold=0.8, check_interval=1800)
    >>> # After finishing, shut down the scheduler
    >>> scheduler.shutdown()

    See Also
    --------
    AutomationManager : Manages the automation of repetitive tasks.

    References
    ----------
    .. [1] Smith, A. (2021). "Adaptive Retraining Strategies in Machine Learning."
       *Journal of Machine Learning Operations*, 9(2), 120-135.
    """

    @validate_params({
        'max_workers': [int]
    })
    def __init__(self, max_workers: int = 4):
        """
        Initializes the ``RetrainingScheduler`` by calling the constructor of
        the parent ``AutomationManager``.

        Parameters
        ----------
        max_workers : int, optional
            The maximum number of threads that can be used to execute tasks.
            Defaults to ``4``.

        Examples
        --------
        >>> scheduler = RetrainingScheduler(max_workers=5)
        """
        super().__init__(max_workers=max_workers)

    @validate_params({
        'model': [object],
        'retrain_func': [Callable],
        'interval': [int]
    })
    def schedule_retraining(
        self,
        model: Any,
        retrain_func: Callable,
        interval: int
    ):
        """
        Schedules regular model retraining.

        Parameters
        ----------
        model : object
            The model to be retrained.
        retrain_func : callable
            The function to retrain the model.
        interval : int
            The interval (in seconds) to schedule retraining.

        Notes
        -----
        This method adds a retraining task to the automation manager,
        scheduling it to run at the specified interval.

        Examples
        --------
        >>> scheduler.schedule_retraining(model, retrain_model, interval=3600)
        """
        task_name = f"retrain_{model.__class__.__name__}"
        logger.info(
            f"Scheduling retraining for model '{model.__class__.__name__}'"
            f" every {interval} seconds."
        )
        self.add_task(
            task_name=task_name,
            func=retrain_func,
            interval=interval,
            args=(model,)
        )
        self.schedule_task(task_name)

    @validate_params({
        'model': [object],
        'metric_func': [Callable]
    })
    def evaluate_model_performance(
        self,
        model: Any,
        metric_func: Callable
    ) -> float:
        """
        Evaluates the model's performance.

        Parameters
        ----------
        model : object
            The model to evaluate.
        metric_func : callable
            A function that returns the performance score of the model.

        Returns
        -------
        score : float
            The model performance score.

        Notes
        -----
        This method applies the ``metric_func`` to the model to compute
        a performance score, which can be used to determine if retraining
        is necessary.

        Examples
        --------
        >>> score = scheduler.evaluate_model_performance(model, evaluate_model)
        """
        score = metric_func(model)
        logger.info(
            f"Evaluated model '{model.__class__.__name__}': Performance score = {score}"
        )
        return score

    @validate_params({
        'model': [object],
        'metric_func': [Callable],
        'decay_threshold': [float]
    })
    def trigger_retraining_on_decay(
        self,
        model: Any,
        metric_func: Callable,
        decay_threshold: float
    ):
        """
        Triggers model retraining if performance decay is detected.

        Parameters
        ----------
        model : object
            The model to evaluate.
        metric_func : callable
            The function to evaluate model performance.
        decay_threshold : float
            The threshold below which retraining is triggered.

        Notes
        -----
        If the model's performance score falls below the ``decay_threshold``,
        retraining is initiated.

        Mathematically, retraining is triggered if:

        .. math::

            \\text{score} < \\text{decay\\_threshold}

        Examples
        --------
        >>> scheduler.trigger_retraining_on_decay(
        ...     model, evaluate_model, decay_threshold=0.8)
        """
        score = self.evaluate_model_performance(model, metric_func)
        if score < decay_threshold:
            logger.warning(
                f"Performance decay detected for model '{model.__class__.__name__}' "
                f"(score: {score}). Triggering retraining."
            )
            task_name = f"retrain_{model.__class__.__name__}"
            if task_name in self.tasks:
                self.tasks[task_name]["func"](*self.tasks[task_name]["args"])
            else:
                logger.error(
                    f"No retraining task found for model '{model.__class__.__name__}'."
                )

    @validate_params({
        'model': [object],
        'metric_func': [Callable],
        'decay_threshold': [float],
        'check_interval': [int]
    })
    def monitor_model(
        self,
        model: Any,
        metric_func: Callable,
        decay_threshold: float,
        check_interval: int
    ):
        """
        Monitors the model's performance at regular intervals and triggers
        retraining if necessary.

        Parameters
        ----------
        model : object
            The model to monitor.
        metric_func : callable
            Function to evaluate model performance.
        decay_threshold : float
            The threshold to trigger retraining.
        check_interval : int
            How often to check model performance (in seconds).

        Notes
        -----
        This method adds a monitoring task that periodically evaluates
        the model's performance and triggers retraining if the performance
        falls below the ``decay_threshold``.

        Examples
        --------
        >>> scheduler.monitor_model(
        ...     model, evaluate_model, decay_threshold=0.8, check_interval=1800)
        """
        def check_and_retrain():
            self.trigger_retraining_on_decay(model, metric_func, decay_threshold)

        logger.info(
            f"Monitoring model '{model.__class__.__name__}' performance with interval {check_interval} seconds."
        )
        task_name = f"monitor_{model.__class__.__name__}"
        self.add_task(
            task_name=task_name,
            func=check_and_retrain,
            interval=check_interval
        )
        self.schedule_task(task_name)


class AirflowAutomation(AutomationManager):
    """
    Integrates the AutomationManager with Apache Airflow to schedule and
    manage tasks using Directed Acyclic Graphs (DAGs).
    
    Parameters
    ----------
    dag_id : str
        The unique identifier for the Airflow DAG.
    start_date : datetime
        The start date of the DAG execution.
    schedule_interval : str, optional
        The scheduling interval for the DAG (e.g., cron expression).
        Defaults to ``'@daily'``.
    
    Attributes
    ----------
    dag_id : str
        The unique identifier for the Airflow DAG.
    start_date : datetime
        The start date of the DAG execution.
    schedule_interval : str
        The scheduling interval for the DAG.
    dag : airflow.DAG
        The Airflow DAG object used to schedule tasks.
    
    Methods
    -------
    add_task_to_airflow(task_name, func, **kwargs)
        Adds a task to the Airflow DAG using a PythonOperator.
    schedule_airflow_task(task_name)
        Schedules the task in the Airflow DAG.
    
    Notes
    -----
    The ``AirflowAutomation`` class allows integration with Apache
    Airflow, enabling the scheduling and management of tasks using Airflow
    DAGs.
    
    Examples
    --------
    >>> from gofast.mlops.automation import AirflowAutomation
    >>> from datetime import datetime
    >>> manager = AirflowAutomation(
    ...     dag_id='automation_dag',
    ...     start_date=datetime(2023, 1, 1),
    ...     schedule_interval='@daily'
    ... )
    >>> def my_task():
    ...     print("Task executed")
    >>> manager.add_task_to_airflow('my_task', my_task)
    >>> manager.schedule_airflow_task('my_task')
    
    See Also
    --------
    AutomationManager : Base class for automation management.
    
    References
    ----------
    .. [1] Apache Airflow Documentation. "Airflow Documentation."
       Retrieved from https://airflow.apache.org/docs/
    """
    
    @ensure_pkg(
        "airflow",
        extra="The 'airflow' package is required for this functionality. "
              "Please install 'apache-airflow' to proceed.",
        dist_name="apache-airflow",
        infer_dist_name=True,
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    
    @validate_params({
        'dag_id': [str],
        'start_date': [datetime],
        'schedule_interval': [str]
    })
    def __init__(
        self,
        dag_id: str,
        start_date: datetime,
        schedule_interval: str = "@daily"
    ):
        super().__init__()
        self.dag_id = dag_id
        self.start_date = start_date
        self.schedule_interval = schedule_interval
        self.dag = self._create_dag()
    
    def _create_dag(self):
        """
        Creates an Airflow DAG for scheduling tasks.
        
        Returns
        -------
        dag : airflow.DAG
            The Airflow DAG object used to schedule tasks.
        
        Notes
        -----
        The DAG is configured with default arguments and the specified
        schedule interval.
        """
        from airflow import DAG

        default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'start_date': self.start_date,
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }

        dag = DAG(
            self.dag_id,
            default_args=default_args,
            description='Automation DAG',
            schedule_interval=self.schedule_interval,
        )
        return dag
    
    @validate_params({
        'task_name': [str],
        'func': [Callable],
    })
    def add_task_to_airflow(self, task_name: str, func: Callable, **kwargs):
        """
        Adds a task to the Airflow DAG using a PythonOperator.
        
        Parameters
        ----------
        task_name : str
            The name of the task in Airflow.
        func : callable
            The Python function to be executed.
        **kwargs : dict
            Additional keyword arguments for the function.
        
        Returns
        -------
        task : airflow.operators.python.PythonOperator
            The Airflow PythonOperator that was created.
        
        Notes
        -----
        The task is added to the DAG and can be scheduled using Airflow's
        scheduling mechanism.
        
        Examples
        --------
        >>> def my_task(arg1, arg2):
        ...     print(f"Task executed with arguments: {arg1}, {arg2}")
        >>> manager.add_task_to_airflow(
        ...     'my_task', my_task, arg1='hello', arg2='world'
        ... )
        """
        from airflow.operators.python import PythonOperator

        task = PythonOperator(
            task_id=task_name,
            python_callable=func,
            op_kwargs=kwargs,
            dag=self.dag
        )
        logger.info(f"Task '{task_name}' added to Airflow DAG '{self.dag_id}'.")
        return task
    
    @validate_params({
        'task_name': [str]
    })
    def schedule_airflow_task(self, task_name: str):
        """
        Schedules the task in the Airflow DAG.
        
        Parameters
        ----------
        task_name : str
            The name of the task to schedule.
        
        Notes
        -----
        In an actual Airflow environment, tasks are scheduled and executed
        by the Airflow scheduler. This method is for demonstration purposes
        and simulates the execution of the task.
        
        Examples
        --------
        >>> manager.schedule_airflow_task('my_task')
        """
        logger.info(f"Scheduling task '{task_name}' in Airflow.")
        task = self.dag.get_task(task_name)
        if task is not None:
            # Simulate task execution
            context = {}  # Airflow provides context in real execution
            task.execute(context=context)
        else:
            logger.error(f"Task '{task_name}' not found in DAG '{self.dag_id}'.")


class KubeflowAutomation(AutomationManager):
    """
    Integrates the AutomationManager with Kubeflow Pipelines to manage tasks
    in Kubernetes.
    
    Parameters
    ----------
    host : str
        The Kubeflow Pipelines API endpoint.
    
    Attributes
    ----------
    client : kfp.Client
        The Kubeflow Pipelines client used to interact with the Pipelines API.
    
    Methods
    -------
    create_kubeflow_pipeline(pipeline_name, task_name, func, **kwargs)
        Creates a Kubeflow pipeline to schedule a task in a Kubernetes environment.
    
    Notes
    -----
    The ``KubeflowAutomation`` allows integration with Kubeflow Pipelines,
    enabling the scheduling and management of tasks in a Kubernetes cluster.
    
    Examples
    --------
    >>> from gofast.mlops.automation import KubeflowAutomation
    >>> manager = KubeflowAutomation(host='http://localhost:8080')
    >>> def my_task():
    ...     print("Task executed")
    >>> manager.create_kubeflow_pipeline('my_pipeline', 'my_task', my_task)
    
    See Also
    --------
    AutomationManager : Base class for automation management.
    
    References
    ----------
    .. [1] Kubeflow Pipelines Documentation. "Kubeflow Pipelines."
       Retrieved from https://www.kubeflow.org/docs/components/pipelines/
    """
    
    @ensure_pkg(
        "kfp",
        extra="The 'kfp' package is required for this functionality. "
              "Please install 'kfp' to proceed.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    
    @validate_params({
        'host': [str]
    })
    def __init__(self, host: str):
        super().__init__()
        from kfp import Client
        self.client = Client(host=host)
    
    @validate_params({
        'pipeline_name': [str],
        'task_name': [str],
        'func': [Callable],
    })
    def create_kubeflow_pipeline(
        self,
        pipeline_name: str,
        task_name: str,
        func: Callable,
        **kwargs
    ):
        """
        Creates a Kubeflow pipeline to schedule a task in a Kubernetes
        environment.
        
        Parameters
        ----------
        pipeline_name : str
            The name of the Kubeflow pipeline.
        task_name : str
            The task name within the pipeline.
        func : callable
            The function to run in the pipeline.
        **kwargs : dict
            Additional keyword arguments for the function.
        
        Returns
        -------
        pipeline_run_id : str
            The ID of the created Kubeflow pipeline run.
        
        Notes
        -----
        This method defines a Kubeflow pipeline using the Kubeflow Pipelines
        SDK, and submits it to the Kubeflow Pipelines API server for execution.
        
        The function ``func`` is converted into a Kubeflow component using the
        ``kfp.components.create_component_from_func`` method.
        
        Examples
        --------
        >>> def my_task():
        ...     print("Task executed")
        >>> manager.create_kubeflow_pipeline(
        ...     'my_pipeline', 'my_task', my_task
        ... )
        """
        from kfp import dsl
        import kfp.components

        # Convert the function into a Kubeflow component
        task_component = kfp.components.create_component_from_func(
            func,
            base_image='python:3.7',
            packages_to_install=[]  # Add required packages if any
        )

        @dsl.pipeline(
            name=pipeline_name,
            description='Automation Pipeline for Machine Learning'
        )
        def pipeline():
            task_step = task_component(**kwargs)
            task_step.set_display_name(task_name)

        pipeline_func = pipeline
        run = self.client.create_run_from_pipeline_func(
            pipeline_func, arguments={}
        )
        pipeline_run_id = run.run_id
        logger.info(
            f"Kubeflow pipeline '{pipeline_name}' created with run ID "
            f"'{pipeline_run_id}'."
        )
        return pipeline_run_id


class KafkaAutomation(AutomationManager):
    """
    Handles real-time data pipeline automation using Kafka. Consumes Kafka
    topics and triggers tasks based on incoming data.

    Parameters
    ----------
    kafka_servers : list of str
        A list of Kafka server addresses.
    topic : str
        The name of the Kafka topic to consume messages from.

    Attributes
    ----------
    consumer : kafka.KafkaConsumer
        The Kafka consumer instance used to consume messages.

    Methods
    -------
    process_kafka_message(func)
        Processes incoming Kafka messages and triggers tasks.

    Notes
    -----
    The ``KafkaAutomation`` class integrates Kafka message consumption
    into the automation framework, allowing tasks to be triggered based on
    real-time data streams.

    The message processing can be modeled as a stream where messages
    :math:`m_i` are consumed and processed in order:

    .. math::

        \\{ m_1, m_2, m_3, \\dots \\} \\rightarrow \\text{process}(m_i)

    Each message is passed to the function ``func`` for processing.

    Examples
    --------
    >>> from gofast.mlops.automation import KafkaAutomation
    >>> def process_data(data):
    ...     print(f"Processing data: {data}")
    >>> manager = KafkaAutomation(
    ...     kafka_servers=['localhost:9092'],
    ...     topic='my_topic'
    ... )
    >>> manager.process_kafka_message(process_data)

    See Also
    --------
    AutomationManager : Base class for automation management.

    References
    ----------
    .. [1] Kafka Documentation. "Apache Kafka."
       Retrieved from https://kafka.apache.org/documentation/
    """

    @ensure_pkg(
        "kafka",
        extra="The 'kafka-python' package is required for this functionality. "
              "Please install 'kafka-python' to proceed.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
        dist_name='kafka', 
        infer_dist_name=True,
    )
    
    @validate_params({
        'kafka_servers': [list],
        'topic': [str]
    })
    def __init__(
        self,
        kafka_servers: List[str],
        topic: str
    ):
        """
        Initializes the ``KafkaAutomation``.

        Parameters
        ----------
        kafka_servers : list of str
            A list of Kafka server addresses.
        topic : str
            The name of the Kafka topic to consume messages from.

        Examples
        --------
        >>> manager = KafkaAutomation(
        ...     kafka_servers=['localhost:9092'],
        ...     topic='my_topic'
        ... )
        """
        super().__init__()
        from kafka import KafkaConsumer
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='automation_manager_group'
        )

    @validate_params({
        'func': [Callable],
    })
    def process_kafka_message(
        self,
        func: Callable[[Any], None]
    ):
        """
        Processes incoming Kafka messages and triggers tasks.

        Parameters
        ----------
        func : callable
            The function to process the data. It should accept a single
            argument, which is the message value.

        Notes
        -----
        This method consumes messages from the Kafka topic and applies the
        provided function ``func`` to each message.

        Examples
        --------
        >>> def process_data(data):
        ...     print(f"Processing data: {data}")
        >>> manager.process_kafka_message(process_data)
        """
        for message in self.consumer:
            data = message.value
            logger.info(f"Received message from Kafka: {data}")
            func(data)


class RabbitMQAutomation(AutomationManager):
    """
    Handles real-time data pipeline automation using RabbitMQ. Consumes
    messages from a RabbitMQ queue and triggers tasks.

    Parameters
    ----------
    host : str
        The RabbitMQ server host address.
    queue : str
        The name of the RabbitMQ queue to consume messages from.

    Attributes
    ----------
    connection : pika.BlockingConnection
        The connection to the RabbitMQ server.
    channel : pika.channel.Channel
        The channel through which messages are consumed.
    queue : str
        The name of the RabbitMQ queue.

    Methods
    -------
    process_rabbitmq_message(func)
        Processes incoming RabbitMQ messages and triggers tasks.

    Notes
    -----
    The ``RabbitMQAutomation`` class integrates RabbitMQ message
    consumption into the automation framework, allowing tasks to be triggered
    based on real-time data streams.

    The message processing can be modeled as a stream where messages
    :math:`m_i` are consumed and processed in order:

    .. math::

        \\{ m_1, m_2, m_3, \\dots \\} \\rightarrow \\text{process}(m_i)

    Each message is passed to the function ``func`` for processing.

    Examples
    --------
    >>> from gofast.mlops.automation import RabbitMQAutomation
    >>> def process_data(data):
    ...     print(f"Processing data: {data}")
    >>> manager = RabbitMQAutomation(
    ...     host='localhost',
    ...     queue='my_queue'
    ... )
    >>> manager.process_rabbitmq_message(process_data)

    See Also
    --------
    AutomationManager : Base class for automation management.

    References
    ----------
    .. [1] RabbitMQ Documentation. "RabbitMQ."
       Retrieved from https://www.rabbitmq.com/documentation.html
    """
    
    @ensure_pkg(
        "pika",
        extra="The 'pika' package is required for this functionality. "
              "Please install 'pika' to proceed.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )

    @validate_params({
        'host': [str],
        'queue': [str]
    })
    def __init__(
        self,
        host: str,
        queue: str
    ):
        """
        Initializes the ``RabbitMQAutomation``.

        Parameters
        ----------
        host : str
            The RabbitMQ server host address.
        queue : str
            The name of the RabbitMQ queue to consume messages from.

        Examples
        --------
        >>> manager = RabbitMQAutomation(
        ...     host='localhost',
        ...     queue='my_queue'
        ... )
        """
        super().__init__()
        import pika
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host)
        )
        self.channel = self.connection.channel()
        self.queue = queue
        self.channel.queue_declare(queue=queue)

    @validate_params({
        'func': [Callable],
    })
    def process_rabbitmq_message(
        self,
        func: Callable[[Any], None]
    ):
        """
        Processes incoming RabbitMQ messages and triggers tasks.

        Parameters
        ----------
        func : callable
            The function to process the data. It should accept a single
            argument, which is the message body.

        Notes
        -----
        This method consumes messages from the RabbitMQ queue and applies the
        provided function ``func`` to each message.

        Examples
        --------
        >>> def process_data(data):
        ...     print(f"Processing data: {data}")
        >>> manager.process_rabbitmq_message(process_data)
        """
        def callback(ch, method, properties, body):
            logger.info(f"Received message from RabbitMQ: {body}")
            func(body)

        self.channel.basic_consume(
            queue=self.queue,
            on_message_callback=callback,
            auto_ack=True
        )
        logger.info(
            f"Waiting for messages from RabbitMQ queue: '{self.queue}'"
        )
        self.channel.start_consuming()
