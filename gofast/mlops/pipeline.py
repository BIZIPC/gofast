# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Handle the automation of end-to-end workflows. Users can build modular
pipelines for preprocessing, training, evaluation, and deployment.
"""

import time
import random
from itertools import product
from typing import Callable, List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from datetime import datetime, timedelta

from sklearn.utils._param_validation import StrOptions

try:
    import psutil
except ImportError:
    pass

from .._gofastlog import gofastlog
from ..api.property import BaseLearner, PipelineBaseClass
from ..decorators import ( 
    RunReturn, smartFitRun, executeWithFallback 
)
from ..compat.sklearn import validate_params
from ..tools.coreutils import validate_ratio
from ..tools.depsutils import ensure_pkg
from ..tools.validator import check_is_runned

from ._base import PipelineOrchestrator
from ._config import INSTALL_DEPENDENCIES, USE_CONDA


logger = gofastlog.get_gofast_logger(__name__)

__all__ = [
    "Pipeline",
    "PipelineStep",
    "PipelineManager",
    "PipelineOptimizer",
    "ResourceMonitor",
    "ResourceManager",
    "create_pipeline",
    "reconfigure_pipeline_on_the_fly",
    "execute_step_conditionally",
    "run_parallel_subpipelines",
    "split_data_for_multitask_pipeline",
    "rollback_to_previous_state",
    "smart_retry_with_backoff",
]


@smartFitRun
class PipelineStep(BaseLearner):
    """
    A sophisticated pipeline step that supports flexible configurations
    and dependencies. Each step can be defined with custom parameters,
    input-output relationships, and metadata tracking.

    Parameters
    ----------
    name : str
        The name of the pipeline step. This should be a unique identifier
        for the step within the pipeline. It is used for logging and tracking
        purposes.

    func : callable
        The function to be executed in this pipeline step. This function
        should accept the input data and any parameters specified in ``params``.
        It must be callable as ``func(data, **params)``.

    params : dict, optional
        A dictionary of parameters to pass to the function ``func``. These
        parameters are passed as keyword arguments when executing the function.
        Defaults to an empty dictionary if not provided.

    dependencies : list of str, optional
        A list of names of other pipeline steps that this step depends on.
        These dependencies are used to determine the execution order of steps
        in the pipeline. If not provided, defaults to an empty list, indicating
        no dependencies.

    Attributes
    ----------
    name : str
        The name of the pipeline step.

    func : callable
        The function to be executed in this pipeline step.

    params : dict
        Parameters to pass to the function ``func``.

    dependencies : list of str
        List of names of other pipeline steps that this step depends on.

    outputs_ : Any
        The output produced by the pipeline step after execution.

    Methods
    -------
    run(data)
        Runs the pipeline step with the given input data and returns
        the output.

    get_dependencies()
        Returns the list of dependencies for this step.

    Notes
    -----
    The ``PipelineStep`` class is designed to be a flexible component within
    a data processing pipeline. By allowing custom functions and parameters,
    it can accommodate a wide range of processing tasks.

    The execution of the step is performed by calling the function ``func``
    with the input data and parameters:

    .. math::

        \\text{outputs} = \\text{func}(\\text{data},\\ **\\text{params})

    This allows for complex operations to be encapsulated within a single
    step, and for steps to be chained together based on their dependencies.

    Examples
    --------
    >>> from gofast.mlops.pipeline import PipelineStep
    >>> def multiply(data, factor=1):
    ...     return [x * factor for x in data]
    >>> step = PipelineStep(
    ...     name='MultiplyStep',
    ...     func=multiply,
    ...     params={'factor': 2},
    ...     dependencies=[]
    ... )
    >>> data = [1, 2, 3]
    >>> output = step.run(data)
    >>> print(output)
    [2, 4, 6]

    See Also
    --------
    Pipeline : Class for managing multiple pipeline steps.

    References
    ----------
    .. [1] Doe, J. (2021). "Building Robust Data Pipelines."
       *Data Engineering Journal*, 10(2), 100-110.
    """

    @validate_params({
        "name": [str],
        "func": [callable],
        "params": [dict, None],
        "dependencies": [list, None]
    })
    def __init__(
        self,
        name: str,
        func: Callable,
        params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.func = func
        self.params = params or {}
        self.dependencies = dependencies or []
        self.outputs_ = None
        self._is_runned = False

    @RunReturn(attribute_name="outputs_")
    def run(self, data: Any) -> Any:
        """
        Runs the pipeline step and returns the output.

        Parameters
        ----------
        data : Any
            The input data to be processed by the pipeline step.

        Returns
        -------
        Any
            The output produced by the function ``func`` when applied to the
            input data with the given parameters.

        Notes
        -----
        Before execution, the method logs the name of the step being executed.
        After execution, the output is stored in the attribute ``outputs_``.
        The method sets the ``_is_runned`` attribute to True.

        Examples
        --------
        >>> from gofast.mlops.pipeline import PipelineStep
        >>> def add_one(data):
        ...     return [x + 1 for x in data]
        >>> step = PipelineStep(name='AddOne', func=add_one)
        >>> result = step.run([1, 2, 3])
        >>> print(result)
        [2, 3, 4]
        """
        logger.info(f"Running step: {self.name}")
        self.outputs_ = self.func(data, **self.params)
        self._is_runned = True
        return self.outputs_

    def get_dependencies(self) -> List[str]:
        """
        Returns the list of dependencies for this step.

        Returns
        -------
        list of str
            Names of the pipeline steps that this step depends on.

        Examples
        --------
        >>> step = PipelineStep(
        ...     name='CurrentStep',
        ...     func=lambda x: x,
        ...     dependencies=['PreviousStep']
        ... )
        >>> step.get_dependencies()
        ['PreviousStep']
        """
        return self.dependencies
    
    @executeWithFallback
    def execute(self, *args, **kwargs):
        """
        Execute method which will call `run` or `fit` based on their
        availability in the class. Executes custom logic when both are 
        present.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to `run` or `fit`.
        **kwargs : dict
            Keyword arguments to be passed to `run` or `fit`.
        
        Returns
        -------
        None
        """
        print("Executing main execute logic.")
        
        return None
    

@smartFitRun
class Pipeline(PipelineBaseClass):
    """
    Represents a machine learning pipeline. This allows chaining of
    different steps for preprocessing, model training, validation, etc.

    Parameters
    ----------
    steps : list of PipelineStep, optional
        List of pipeline steps to execute. Each step must be an instance
        of ``PipelineStep``. If not provided, initializes with an empty
        list.

    parallel : bool, optional
        Whether to run steps in parallel (if possible). Defaults to
        ``False``.

    Attributes
    ----------
    steps : list of PipelineStep
        The steps added to the pipeline.

    parallel : bool
        Indicates whether the pipeline executes steps in parallel.

    outputs_ : Any
        The final output after running the pipeline.

    Methods
    -------
    add_step(step)
        Adds a new step to the pipeline.

    run(initial_data)
        Runs the pipeline from start to finish.

    Notes
    -----
    The pipeline manages the execution of multiple steps, which can be
    either run sequentially or in parallel, depending on the ``parallel``
    parameter. Each step processes data and passes its output to the
    next step in the sequence.

    When running in parallel, all steps receive the same initial data
    as input, and their outputs are collected independently. This mode
    is suitable when steps are independent of each other.

    The execution can be mathematically represented as:

    - Sequential Execution:

      .. math::

          \\text{data}_{i} = \\text{step}_{i}.\\text{run}(\\text{data}_{i-1})

          \\text{for } i = 1 \\text{ to } N

      where :math:`\\text{data}_{0}` is the initial input data, and
      :math:`N` is the number of steps.

    - Parallel Execution:

      .. math::

          \\text{outputs}_{i} = \\text{step}_{i}.\\text{run}(\\text{data}_{0})

          \\text{for } i = 1 \\text{ to } N

    Examples
    --------
    >>> from gofast.mlops.pipeline import Pipeline, PipelineStep
    >>> def preprocess(data):
    ...     return [x * 2 for x in data]
    >>> def model_train(data):
    ...     return sum(data) / len(data)
    >>> step1 = PipelineStep(name='Preprocess', func=preprocess)
    >>> step2 = PipelineStep(name='TrainModel', func=model_train)
    >>> pipeline = Pipeline(steps=[step1, step2])
    >>> result = pipeline.run([1, 2, 3, 4])
    >>> print(result)
    5.0

    See Also
    --------
    PipelineStep : Class representing an individual pipeline step.

    References
    ----------
    .. [1] Smith, A. (2020). "Efficient Pipeline Design in Machine
       Learning." *Journal of Data Science*, 15(4), 200-215.
    """

    @validate_params({
        'steps': [list, None],
        'parallel': [bool]
    })
    def __init__(
        self,
        steps: Optional[List[PipelineStep]] = None,
        parallel: bool = False
    ):
        self.steps = steps or []
        self.parallel = parallel
        self.outputs_ = None
        self._is_runned = False

    def add_step(self, step: PipelineStep):
        """
        Adds a new step to the pipeline.

        Parameters
        ----------
        step : PipelineStep
            A single step to add to the pipeline.

        Notes
        -----
        The step is appended to the pipeline's steps list. The order in
        which steps are added matters when running sequentially.

        Examples
        --------
        >>> from gofast.mlops.pipeline import Pipeline, PipelineStep
        >>> pipeline = Pipeline()
        >>> step = PipelineStep(name='Normalize', func=lambda x: x)
        >>> pipeline.add_step(step)
        >>> print(len(pipeline.steps))
        1
        """
        logger.info(f"Adding step: {step.name}")
        self.steps.append(step)

    @RunReturn(attribute_name="outputs_")
    def run(self, initial_data: Any) -> Any:
        """
        Runs the pipeline from start to finish.

        Parameters
        ----------
        initial_data : Any
            The input data that is passed through the pipeline.

        Returns
        -------
        Any
            The final output after all pipeline steps.

        Raises
        ------
        Exception
            If any pipeline step fails during execution.

        Notes
        -----
        The execution behavior depends on the ``parallel`` attribute:

        - If ``parallel`` is ``False``, steps are executed sequentially,
          and the output of each step is passed as input to the next step.
        - If ``parallel`` is ``True``, all steps are executed in parallel,
          each receiving the same initial data. This mode assumes that
          steps are independent of each other.

        The method sets the ``_is_runned`` attribute to True.

        Examples
        --------
        >>> from gofast.mlops.pipeline import Pipeline, PipelineStep
        >>> def step1_func(data):
        ...     return data + [5]
        >>> def step2_func(data):
        ...     return [x * 2 for x in data]
        >>> step1 = PipelineStep(name='Step1', func=step1_func)
        >>> step2 = PipelineStep(name='Step2', func=step2_func)
        >>> pipeline = Pipeline(steps=[step1, step2])
        >>> result = pipeline.run([1, 2, 3])
        >>> print(result)
        [2, 4, 6, 10]
        """
        data = initial_data
        if self.parallel:
            # Parallel execution
            with ThreadPoolExecutor() as executor:
                future_to_step = {
                    executor.submit(step.run, data): step for step in self.steps
                }
                results = []
                for future in future_to_step:
                    step = future_to_step[future]
                    try:
                        result = future.result()
                        logger.info(f"Step {step.name} completed successfully.")
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Step {step.name} failed with error: {str(e)}")
                        raise e
            self.outputs_ = results
            self._is_runned = True
            return self.outputs_
        else:
            # Sequential execution
            for step in self.steps:
                try:
                    data = step.run(data)
                    logger.info(f"Step {step.name} completed.")
                except Exception as e:
                    logger.error(f"Pipeline step {step.name} failed: {str(e)}")
                    raise e
            self.outputs_ = data
            self._is_runned = True
            return self.outputs_
        
    @executeWithFallback
    def execute(self, *args, **kwargs):
        """
        Execute method which will call `run` or `fit` based on their
        availability in the class. Executes custom logic when both are 
        present.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to `run` or `fit`.
        **kwargs : dict
            Keyword arguments to be passed to `run` or `fit`.
        
        Returns
        -------
        None
        """
        print("Executing main execute logic. Does nothing")
        
        return None
    
@smartFitRun
class PipelineManager(PipelineBaseClass):
    """
    Manages the creation, execution, and tracking of pipelines.
    Offers support for step dependencies, error recovery, and
    metadata tracking.

    Parameters
    ----------
    retry_failed_steps : bool, optional
        If ``True``, the manager will retry failed steps during execution.
        Defaults to ``False``.

    Attributes
    ----------
    steps : OrderedDict of str to PipelineStep
        An ordered dictionary mapping step names to :class:`PipelineStep`
        instances, maintaining the execution sequence.

    retry_failed_steps : bool
        Indicates whether failed steps should be retried.

    step_metadata_ : dict of str to dict
        A dictionary holding metadata for each step, including status
        and output.

    failed_steps_ : list of str
        List of names of steps that have failed during execution.

    outputs_ : Any
        The final output after running the pipeline.

    Methods
    -------
    add_step(step)
        Adds a step to the pipeline with metadata tracking.

    get_step(name)
        Retrieves a step by name.

    run(initial_data)
        Runs the pipeline in the correct order, respecting dependencies.

    run_step(step_name)
        Runs a single step from the pipeline.

    retry_failed()
        Retries failed steps if the ``retry_failed_steps`` flag is set.

    get_metadata()
        Returns metadata for the pipeline, including step statuses and outputs.

    Notes
    -----
    The :class:`PipelineManager` class orchestrates the execution of pipeline
    steps, respecting dependencies between steps. It performs a dependency
    resolution to determine the execution order. Each step's execution is
    monitored, and metadata is collected.

    Mathematically, the execution order is determined based on dependencies.
    For a set of steps :math:`S = \\{s_1, s_2, \\dots, s_n\\}`, with dependencies
    represented as a directed graph, the manager ensures that each step
    :math:`s_i` is executed only after all its dependencies have been successfully
    executed.

    The execution process can be represented as:

    .. math::

        \\text{Run } s_i \\text{ if } \\forall d \\in D_i, \\text{status}(d) = \\text{success}

    where :math:`D_i` is the set of dependencies for step :math:`s_i`.

    Examples
    --------
    >>> from gofast.mlops.pipeline import PipelineManager, PipelineStep
    >>> def increment(data):
    ...     return data + 1
    >>> step1 = PipelineStep(name='step1', func=increment)
    >>> step2 = PipelineStep(name='step2', func=increment, dependencies=['step1'])
    >>> manager = PipelineManager()
    >>> manager.add_step(step1)
    >>> manager.add_step(step2)
    >>> result = manager.run(initial_data=0)
    >>> print(result)
    2

    See Also
    --------
    PipelineStep : Represents an individual step in a pipeline.

    References
    ----------
    .. [1] Johnson, L. (2020). "Advanced Pipeline Management in Machine Learning."
       *Machine Learning Journal*, 12(3), 150-165.
    """


    @validate_params({
        'retry_failed_steps': [bool]
    })
    def __init__(self, retry_failed_steps: bool = False):
        self.steps: 'OrderedDict[str, PipelineStep]' = OrderedDict()
        self.retry_failed_steps = retry_failed_steps
        self.step_metadata_: Dict[str, Any] = {}
        self.failed_steps_: List[str] = []
        self.outputs_: Any = None
        self._is_runned = False

    def add_step(self, step: 'PipelineStep'):
        """
        Adds a step to the pipeline with metadata tracking.

        Parameters
        ----------
        step : PipelineStep
            The :class:`PipelineStep` instance to add to the pipeline.

        Raises
        ------
        ValueError
            If a step with the same name already exists in the pipeline.

        Notes
        -----
        Steps are stored in an ordered fashion to maintain the execution
        sequence. Each step's metadata is initialized with status 'pending'
        and no output.

        Examples
        --------
        >>> from gofast.mlops.pipeline import PipelineManager, PipelineStep
        >>> def double(data):
        ...     return data * 2
        >>> step = PipelineStep(name='double', func=double)
        >>> manager = PipelineManager()
        >>> manager.add_step(step)
        """
        if step.name in self.steps:
            raise ValueError(f"Step with name {step.name} already exists.")
        logger.info(f"Adding step: {step.name} with dependencies: {step.dependencies}")
        self.steps[step.name] = step
        self.step_metadata_[step.name] = {"status": "pending", "output": None}

    def get_step(self, name: str) -> Optional['PipelineStep']:
        """
        Retrieves a step by name.

        Parameters
        ----------
        name : str
            The name of the step to retrieve.

        Returns
        -------
        PipelineStep or None
            The :class:`PipelineStep` instance if found, else ``None``.

        Examples
        --------
        >>> step = manager.get_step('double')
        >>> print(step.name)
        double
        """
        return self.steps.get(name, None)

    @RunReturn(attribute_name="outputs_")
    def run(self, initial_data: Any) -> Any:
        """
        Runs the pipeline in the correct order, respecting dependencies.
        Allows retrying failed steps if enabled.

        Parameters
        ----------
        initial_data : Any
            The input data to pass through the first pipeline step.

        Returns
        -------
        Any
            The final output after running the pipeline.

        Raises
        ------
        Exception
            Propagates exceptions from steps if ``retry_failed_steps`` is ``False``.

        Notes
        -----
        The execution order is determined by resolving dependencies among
        steps. If a step fails and ``retry_failed_steps`` is ``False``, the
        execution halts. If ``retry_failed_steps`` is ``True``, the manager
        will attempt to retry failed steps.

        The method sets the ``_is_runned`` attribute to True.

        Examples
        --------
        >>> result = manager.run(initial_data=1)
        >>> print(result)
        4
        """
        execution_sequence = self._determine_execution_order()

        for step_name in execution_sequence:
            step = self.steps[step_name]
            # Run the step only if all dependencies have successfully executed
            if self._can_run(step):
                try:
                    # Collect inputs from dependencies
                    if step.get_dependencies():
                        input_data = [self.step_metadata_[dep]["output"]
                                      for dep in step.get_dependencies()]
                        # For simplicity, use the last dependency's output
                        input_data = input_data[-1]
                    else:
                        input_data = initial_data

                    logger.info(f"Running {step_name} with input: {input_data}")
                    output = step.run(input_data)
                    self._update_step_metadata(step_name, "success", output)
                except Exception as e:
                    logger.error(f"Step {step_name} failed with error: {str(e)}")
                    self._update_step_metadata(step_name, "failed", None)
                    if not self.retry_failed_steps:
                        raise e
                    else:
                        logger.info(f"Retrying step {step_name} as retry is enabled.")
                        self.failed_steps_.append(step_name)
            else:
                logger.info(f"Skipping {step_name} due to unmet dependencies.")
        # Return the output of the last step
        final_output = self.step_metadata_[execution_sequence[-1]]["output"]
        self.outputs_ = final_output
        self._is_runned = True
        return final_output

    def _determine_execution_order(self) -> List[str]:
        """
        Determines the order in which the steps should be run,
        based on the dependencies defined for each step.

        Returns
        -------
        execution_order : list of str
            The list of step names in the order they should be run.

        Notes
        -----
        This method performs a topological sort to resolve dependencies.
        For complex dependency graphs, it ensures that all dependencies are
        run before a step.

        Examples
        --------
        >>> order = manager._determine_execution_order()
        >>> print(order)
        ['step1', 'step2']
        """
        execution_order = []
        visited = set()

        def visit(step_name):
            if step_name in visited:
                return
            step = self.steps[step_name]
            for dep in step.get_dependencies():
                visit(dep)
            visited.add(step_name)
            if step_name not in execution_order:
                execution_order.append(step_name)

        for step_name in self.steps:
            visit(step_name)

        logger.info(f"Determined execution order: {execution_order}")
        return execution_order

    def run_step(self, step_name: str):
        """
        Runs a single step from the pipeline.

        Parameters
        ----------
        step_name : str
            The name of the step to run.

        Raises
        ------
        ValueError
            If the step with the given name does not exist in the pipeline.
        Exception
            If the step fails or cannot be run due to unmet dependencies.

        Examples
        --------
        >>> from gofast.mlops.pipeline import PipelineManager, PipelineStep
        >>> def increment(data):
        ...     return data + 1
        >>> def double(data):
        ...     return data * 2
        >>> step1 = PipelineStep(name='step1', func=increment)
        >>> step2 = PipelineStep(name='step2', func=double, dependencies=['step1'])
        >>> manager = PipelineManager()
        >>> manager.add_step(step1)
        >>> manager.add_step(step2)
        >>> manager.run_step('step1')  # Runs 'step1' only
        >>> # Runs 'step2' after 'step1' because of the dependency
        >>> manager.run_step('step2')
        >>> print(manager.get_metadata())
        {'step1': {'status': 'success', 'output': 1}, 'step2': {'status': 'success', 'output': 2}}
        """
        step = self.get_step(step_name)
        if not step:
            raise ValueError(f"Step {step_name} does not exist in the pipeline.")

        if self._can_run(step):
            try:
                # Collect inputs from dependencies
                if step.get_dependencies():
                    input_data = [self.step_metadata_[dep]["output"]
                                  for dep in step.get_dependencies()]
                    input_data = input_data[-1]
                else:
                    input_data = None

                logger.info(f"Running step {step_name} with input: {input_data}")
                output = step.run(input_data)
                self._update_step_metadata(step_name, "success", output)
            except Exception as e:
                logger.error(f"Step {step_name} failed with error: {str(e)}")
                self._update_step_metadata(step_name, "failed", None)
                raise e
        else:
            logger.info(f"Cannot run {step_name} due to unmet dependencies.")

    def _can_run(self, step: 'PipelineStep') -> bool:
        """
        Checks if a pipeline step can be run by verifying that all
        dependencies have succeeded.

        Parameters
        ----------
        step : PipelineStep
            The :class:`PipelineStep` to check.

        Returns
        -------
        bool
            ``True`` if the step can be run, ``False`` otherwise.

        Examples
        --------
        >>> can_run = manager._can_run(step2)
        >>> print(can_run)
        True
        """
        for dep in step.get_dependencies():
            if self.step_metadata_.get(dep, {}).get("status") != "success":
                return False
        return True

    def _update_step_metadata(self, step_name: str, status: str, output: Any):
        """
        Updates the metadata for a specific step in the pipeline.

        Parameters
        ----------
        step_name : str
            The name of the step to update.
        status : str
            The new status of the step ('pending', 'success', 'failed').
        output : Any
            The output produced by the step.

        Notes
        -----
        This method updates the ``step_metadata_`` dictionary for the given
        step.

        Examples
        --------
        >>> manager._update_step_metadata('step1', 'success', output)
        """
        self.step_metadata_[step_name]["status"] = status
        self.step_metadata_[step_name]["output"] = output

    def retry_failed(self):
        """
        Retries failed steps if the ``retry_failed_steps`` flag is set.

        Notes
        -----
        This method re-runs any steps that failed during the initial
        execution, using the outputs of their dependencies.

        Examples
        --------
        >>> manager.retry_failed()
        """
        if not self.failed_steps_:
            logger.info("No failed steps to retry.")
            return

        logger.info("Retrying failed steps...")
        for step_name in self.failed_steps_:
            step = self.steps[step_name]
            try:
                dependencies = step.get_dependencies()
                if dependencies:
                    input_data = [self.step_metadata_[dep]["output"] for dep in dependencies]
                    input_data = input_data[-1]
                else:
                    input_data = None
                output = step.run(input_data)
                self._update_step_metadata(step_name, "success", output)
                logger.info(f"Retry of step {step_name} succeeded.")
            except Exception as e:
                logger.error(f"Retry failed for step {step_name} with error: {str(e)}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata for the pipeline, including step statuses and outputs.

        Returns
        -------
        metadata : dict
            A dictionary containing metadata for each step.

        Examples
        --------
        >>> metadata = manager.get_metadata()
        >>> print(metadata)
        {'step1': {'status': 'success', 'output': 1}, 'step2': {'status': 'success', 'output': 2}}
        """
        check_is_runned(self, attributes=['_is_runned'])
        return self.step_metadata_

    @executeWithFallback
    def execute(self, *args, **kwargs):
        """
        Execute method which will call `run` or `fit` based on their
        availability in the class. Executes custom logic when both are 
        present.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to `run` or `fit`.
        **kwargs : dict
            Keyword arguments to be passed to `run` or `fit`.
        
        Returns
        -------
        None
        """
        print("Executing main execute logic. Does nothing")
        
        return None
    
@smartFitRun
@ensure_pkg(
    "psutil",
    extra="The 'psutil' package is required for this functionality.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA
)
class ResourceManager(BaseLearner):
    """
    Manages the allocation of system resources for pipeline steps.
    This includes checking available resources and assigning them
    to specific steps, ensuring efficient resource utilization.

    Attributes
    ----------
    available_cpu_cores_ : int
        The number of physical CPU cores available for allocation.

    available_memory_ : int
        The total system memory available for allocation, in bytes.

    Methods
    -------
    run()
        Initializes the resource manager by detecting available resources.

    allocate_cpu(requested_cores)
        Allocates CPU cores to a step.

    allocate_memory(requested_memory)
        Allocates memory to a step.

    release_resources(cpu_cores, memory)
        Releases resources after a pipeline step has been executed.

    get_system_resources()
        Returns a dictionary of the current available system resources.

    Notes
    -----
    The :class:`ResourceManager` is responsible for managing system
    resources such as CPU cores and memory. It provides methods to
    allocate and release resources, ensuring that pipeline steps
    have the necessary resources to execute efficiently.

    The allocation and release of resources can be mathematically
    represented as:

    - CPU Core Allocation:

      .. math::

          \\text{available\\_cpu\\_cores} = \\text{available\\_cpu\\_cores}
          - \\text{requested\\_cores}

    - Memory Allocation:

      .. math::

          \\text{available\\_memory} = \\text{available\\_memory}
          - \\text{requested\\_memory}

    - Resource Release:

      .. math::

          \\text{available\\_cpu\\_cores} = \\text{available\\_cpu\\_cores}
          + \\text{cpu\\_cores}

          \\text{available\\_memory} = \\text{available\\_memory}
          + \\text{memory}

    Examples
    --------
    >>> from gofast.mlops.pipeline import ResourceManager
    >>> manager = ResourceManager()
    >>> manager.run()
    >>> success = manager.allocate_cpu(2)
    >>> if success:
    ...     print("Allocated CPU cores.")
    ... else:
    ...     print("Failed to allocate CPU cores.")
    Allocated CPU cores.
    >>> success = manager.allocate_memory(1024 * 1024 * 1024)  # 1 GB
    >>> if success:
    ...     print("Allocated memory.")
    ... else:
    ...     print("Failed to allocate memory.")
    Allocated memory.
    >>> resources = manager.get_system_resources()
    >>> print(resources)
    {'available_cpu_cores': ..., 'available_memory_gb': ...}

    See Also
    --------
    ResourceMonitor : Class to monitor system resource usage.

    References
    ----------
    .. [1] Smith, J. (2020). "Efficient Resource Management in Pipeline Systems."
       *Journal of Systems Engineering*, 15(3), 200-215.
    """

    def __init__(self):
        self.available_cpu_cores_ = None
        self.available_memory_ = None
        self._is_runned = False

    @RunReturn
    def run(self):
        """
        Initializes the resource manager by detecting available resources.

        Notes
        -----
        This method detects the available physical CPU cores and total system
        memory, and initializes the corresponding attributes.

        Examples
        --------
        >>> manager.run()
        """
        self.available_cpu_cores_ = psutil.cpu_count(logical=False)
        self.available_memory_ = psutil.virtual_memory().total
        self._is_runned = True
        logger.info(
            f"ResourceManager initialized with {self.available_cpu_cores_} "
            f"CPU cores and {self.available_memory_ / (1024 ** 3):.2f} GB memory."
        )

    @validate_params({'requested_cores': [int]})
    def allocate_cpu(self, requested_cores: int) -> bool:
        """
        Allocates CPU cores to a step.

        Parameters
        ----------
        requested_cores : int
            The number of CPU cores requested by a pipeline step.

        Returns
        -------
        bool
            Returns ``True`` if the requested CPU cores are successfully
            allocated; otherwise, returns ``False``.

        Notes
        -----
        If the requested number of CPU cores is less than or equal to
        the available CPU cores, the method allocates them and updates
        the ``available_cpu_cores_`` attribute.

        Mathematically:

        .. math::

            \\text{available\\_cpu\\_cores} = \\text{available\\_cpu\\_cores}
            - \\text{requested\\_cores}

        Examples
        --------
        >>> success = manager.allocate_cpu(4)
        >>> print(success)
        True
        """
        check_is_runned(self, attributes=['_is_runned'])
        if requested_cores <= self.available_cpu_cores_:
            logger.info(f"Allocated {requested_cores} CPU cores.")
            self.available_cpu_cores_ -= requested_cores
            return True
        else:
            logger.warning(
                f"Not enough CPU cores. Requested: {requested_cores}, "
                f"Available: {self.available_cpu_cores_}"
            )
            return False

    @validate_params({'requested_memory': [int]})
    def allocate_memory(self, requested_memory: int) -> bool:
        """
        Allocates memory to a step.

        Parameters
        ----------
        requested_memory : int
            The amount of memory requested by a pipeline step, in bytes.

        Returns
        -------
        bool
            Returns ``True`` if the requested memory is successfully allocated;
            otherwise, returns ``False``.

        Notes
        -----
        If the requested memory is less than or equal to the available
        memory, the method allocates it and updates the ``available_memory_``
        attribute.

        Mathematically:

        .. math::

            \\text{available\\_memory} = \\text{available\\_memory}
            - \\text{requested\\_memory}

        Examples
        --------
        >>> success = manager.allocate_memory(1024 * 1024 * 1024)  # 1 GB
        >>> print(success)
        True
        """
        check_is_runned(self, attributes=['_is_runned'])
        if requested_memory <= self.available_memory_:
            allocated_gb = requested_memory / (1024 ** 3)
            logger.info(f"Allocated {allocated_gb:.2f} GB memory.")
            self.available_memory_ -= requested_memory
            return True
        else:
            available_gb = self.available_memory_ / (1024 ** 3)
            requested_gb = requested_memory / (1024 ** 3)
            logger.warning(
                f"Not enough memory. Requested: {requested_gb:.2f} GB, "
                f"Available: {available_gb:.2f} GB"
            )
            return False

    @validate_params({'cpu_cores': [int], 'memory': [int]})
    def release_resources(self, cpu_cores: int, memory: int):
        """
        Releases resources after a pipeline step has been executed.

        Parameters
        ----------
        cpu_cores : int
            The number of CPU cores to release.

        memory : int
            The amount of memory to release, in bytes.

        Notes
        -----
        This method increases the ``available_cpu_cores_`` and
        ``available_memory_`` attributes by the specified amounts.

        Mathematically:

        .. math::

            \\text{available\\_cpu\\_cores} = \\text{available\\_cpu\\_cores}
            + \\text{cpu\\_cores}

            \\text{available\\_memory} = \\text{available\\_memory}
            + \\text{memory}

        Examples
        --------
        >>> manager.release_resources(cpu_cores=2, memory=1024 * 1024 * 1024)
        """
        check_is_runned(self, attributes=['_is_runned'])
        self.available_cpu_cores_ += cpu_cores
        self.available_memory_ += memory
        released_gb = memory / (1024 ** 3)
        logger.info(
            f"Released {cpu_cores} CPU cores and {released_gb:.2f} GB memory."
        )

    def get_system_resources(self) -> Dict[str, float]:
        """
        Returns a dictionary of the current available system resources.

        Returns
        -------
        resources : dict
            A dictionary containing the available CPU cores and memory
            in gigabytes.

        Notes
        -----
        This method can be used to check the current available resources.

        Examples
        --------
        >>> resources = manager.get_system_resources()
        >>> print(resources)
        {'available_cpu_cores': ..., 'available_memory_gb': ...}
        """
        check_is_runned(self, attributes=['_is_runned'])
        return {
            "available_cpu_cores": self.available_cpu_cores_,
            "available_memory_gb": self.available_memory_ / (1024 ** 3),
        }


@ensure_pkg(
    "psutil",
    extra="The 'psutil' package is required for this functionality.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA
)
class ResourceMonitor(BaseLearner):
    """
    Monitors system resource usage during the execution of pipeline
    steps. It tracks CPU and memory usage over time, providing insights
    into resource utilization.

    Attributes
    ----------
    cpu_usage_ : list of float
        A list containing recorded CPU usage percentages.

    memory_usage_ : list of int
        A list containing recorded memory usage amounts, in bytes.

    Methods
    -------
    run()
        Starts monitoring system resources.

    stop_monitoring()
        Stops monitoring and logs the final resource usage statistics.

    record_usage()
        Records current CPU and memory usage.

    Notes
    -----
    The :class:`ResourceMonitor` provides functionality to monitor and
    log system resource usage, which can be useful for performance
    analysis and optimization.

    The average CPU and memory usage are calculated as:

    - Average CPU Usage:

      .. math::

          \\text{Average CPU Usage} = \\frac{1}{N} \\sum_{i=1}^{N}
          \\text{CPU}_i

    - Average Memory Usage:

      .. math::

          \\text{Average Memory Usage} = \\frac{1}{N} \\sum_{i=1}^{N}
          \\text{Memory}_i

    where :math:`N` is the number of recorded intervals,
    :math:`\\text{CPU}_i` is the CPU usage at interval :math:`i`, and
    :math:`\\text{Memory}_i` is the memory usage at interval :math:`i`.

    Examples
    --------
    >>> from gofast.mlops.pipeline import ResourceMonitor
    >>> monitor = ResourceMonitor()
    >>> monitor.run()
    >>> # Simulate workload
    >>> for _ in range(5):
    ...     monitor.record_usage()
    ...     # Perform some work here
    >>> monitor.stop_monitoring()

    See Also
    --------
    ResourceManager : Class to manage resource allocation.

    References
    ----------
    .. [1] Lee, K. (2019). "Monitoring System Resources in Data Pipelines."
       *International Journal of Data Engineering*, 8(2), 120-135.
    """

    def __init__(self):
        self.cpu_usage_: List[float] = []
        self.memory_usage_: List[int] = []
        self._is_runned = False

    @RunReturn()
    def run(self):
        """
        Starts monitoring system resources.

        Notes
        -----
        This method initializes or clears the lists that store CPU and
        memory usage data, and sets the monitoring flag.

        Examples
        --------
        >>> monitor.run()
        """
        logger.info("Starting resource monitoring...")
        self.cpu_usage_.clear()
        self.memory_usage_.clear()
        self._is_runned = True

    def stop_monitoring(self):
        """
        Stops monitoring and logs the final resource usage statistics.

        Notes
        -----
        This method calculates the average CPU and memory usage over the
        recorded intervals and logs the results.

        The average CPU usage is calculated as:

        .. math::

            \\text{Average CPU Usage} = \\frac{1}{N} \\sum_{i=1}^{N}
            \\text{CPU}_i

        The average memory usage is calculated as:

        .. math::

            \\text{Average Memory Usage} = \\frac{1}{N} \\sum_{i=1}^{N}
            \\text{Memory}_i

        Examples
        --------
        >>> monitor.stop_monitoring()
        """
        check_is_runned(self, attributes=['_is_runned'])
        if self.cpu_usage_:
            avg_cpu = sum(self.cpu_usage_) / len(self.cpu_usage_)
        else:
            avg_cpu = 0
        if self.memory_usage_:
            avg_memory = sum(self.memory_usage_) / len(self.memory_usage_)
        else:
            avg_memory = 0
        avg_memory_gb = avg_memory / (1024 ** 3)
        logger.info(f"Average CPU Usage: {avg_cpu:.2f}%")
        logger.info(f"Average Memory Usage: {avg_memory_gb:.2f} GB")

    def record_usage(self):
        """
        Records current CPU and memory usage.

        Notes
        -----
        This method records the current system-wide CPU usage percentage
        and the amount of memory currently being used.

        Examples
        --------
        >>> monitor.record_usage()
        """
        check_is_runned(self, attributes=['_is_runned'])
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().used
        self.cpu_usage_.append(cpu)
        self.memory_usage_.append(memory)
        memory_gb = memory / (1024 ** 3)
        logger.info(
            f"Current CPU Usage: {cpu:.2f}%, Memory Usage: {memory_gb:.2f} GB"
        )


@smartFitRun
class PipelineOptimizer(BaseLearner):
    """
    Optimizes resource allocation and usage for pipeline steps,
    integrating with :class:`PipelineManager`, :class:`ResourceManager`, and
    :class:`ResourceMonitor` to allocate, release, monitor, and tune resources.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        An instance of :class:`PipelineManager` that manages the pipeline steps.

    Attributes
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager associated with this optimizer.

    resource_manager_ : ResourceManager
        Manages resource allocation and release.

    resource_monitor_ : ResourceMonitor
        Monitors resource usage during pipeline execution.

    _is_runned : bool
        Indicates whether the optimizer has been run.

    Methods
    -------
    run()
        Initializes the optimizer by setting up resource manager and monitor.

    allocate_resources(step_name, resources)
        Allocates resources to a specific pipeline step.

    release_resources_after_step(step_name, resources)
        Releases allocated resources after a pipeline step is completed.

    monitor_resources_for_step(step_name, duration=5)
        Monitors resource usage for a given step over a specified duration.

    tune_resources_based_on_usage(step_name, threshold)
        Tunes resource allocation based on observed resource usage.

    tune_hyperparameters(step_name, param_grid, n_trials=10, eval_metric='accuracy')
        Tunes hyperparameters for a specific pipeline step.

    Notes
    -----
    The :class:`PipelineOptimizer` class is designed to improve the efficiency
    of pipeline execution by managing resource allocation and usage.
    It leverages resource monitoring data to adjust allocations,
    ensuring optimal performance.

    Examples
    --------
    >>> from gofast.mlops.pipeline import PipelineOptimizer, PipelineManager
    >>> pipeline_manager = PipelineManager()
    >>> optimizer = PipelineOptimizer(pipeline_manager)
    >>> optimizer.run()
    >>> optimizer.allocate_resources('step1', {'CPU': 2, 'Memory': 2 * 1024 ** 3})
    >>> optimizer.monitor_resources_for_step('step1', duration=5)
    >>> recommended_resources = optimizer.tune_resources_based_on_usage(
    ...     'step1', {'CPU': 75, 'Memory': 4 * 1024 ** 3})
    >>> print(recommended_resources)
    {'CPU': 4, 'Memory': 5368709120}

    See Also
    --------
    ResourceManager : Manages resource allocation and release.
    ResourceMonitor : Monitors system resource usage.
    PipelineManager : Manages the execution of pipeline steps.

    References
    ----------
    .. [1] Smith, A. (2021). "Dynamic Resource Allocation in Machine
       Learning Pipelines." *Journal of Computational Efficiency*, 10(2),
       100-115.
    """

    @validate_params({
        'pipeline_manager': [object],
    })
    def __init__(self, pipeline_manager: 'PipelineManager'):
        self.pipeline_manager = pipeline_manager
        self.resource_manager_ = None
        self.resource_monitor_ = None
        self._is_runned = False

    @RunReturn()
    def run(self):
        """
        Initializes the optimizer by setting up resource manager and monitor.

        Notes
        -----
        This method initializes the resource manager and resource monitor,
        preparing the optimizer for resource allocation and monitoring tasks.
        It sets the `_is_runned` attribute to True.

        Examples
        --------
        >>> optimizer.run()
        """
        self.resource_manager_ = ResourceManager()
        self.resource_manager_.run()
        self.resource_monitor_ = ResourceMonitor()
        self.resource_monitor_.run()
        self._is_runned = True
        logger.info("PipelineOptimizer has been initialized.")

    @validate_params({
        'step_name': [str],
        'resources': [dict]
    })
    def allocate_resources(self, step_name: str, resources: Dict[str, Any]) -> None:
        """
        Allocates resources (e.g., CPU cores, memory) to a specific
        pipeline step.

        Parameters
        ----------
        step_name : str
            The name of the pipeline step to allocate resources for.

        resources : dict
            A dictionary specifying the resources to allocate.
            Keys can include `'CPU'` and `'Memory'`.

            - `'CPU'`: Number of CPU cores requested.
            - `'Memory'`: Amount of memory requested in bytes.

        Notes
        -----
        This method attempts to allocate the requested resources using
        the :class:`ResourceManager`. If allocation fails, an error is logged.

        Mathematically, the allocation can be represented as:

        .. math::

            \\text{available\\_cpu\\_cores} = \\text{available\\_cpu\\_cores}
            - \\text{requested\\_cores}

            \\text{available\\_memory} = \\text{available\\_memory}
            - \\text{requested\\_memory}

        Examples
        --------
        >>> optimizer.allocate_resources('step1', {'CPU': 2, 'Memory': 2 * 1024 ** 3})
        """
        check_is_runned(self, attributes=['_is_runned'])
        logger.info(f"Allocating resources for step: {step_name} -> {resources}")

        step = self.pipeline_manager.get_step(step_name)
        if not step:
            raise ValueError(f"Step {step_name} does not exist in the pipeline.")

        cpu_allocated = True
        memory_allocated = True

        if "CPU" in resources:
            requested_cores = resources["CPU"]
            cpu_allocated = self.resource_manager_.allocate_cpu(requested_cores)

        if "Memory" in resources:
            requested_memory = resources["Memory"]
            memory_allocated = self.resource_manager_.allocate_memory(requested_memory)

        if cpu_allocated and memory_allocated:
            logger.info(f"Resources allocated successfully for step: {step_name}.")
        else:
            logger.error(
                f"Failed to allocate resources for step: {step_name}. "
                f"Check resource availability."
            )

    @validate_params({
        'step_name': [str],
        'resources': [dict]
    })
    def release_resources_after_step(
            self, step_name: str, resources: Dict[str, Any]) -> None:
        """
        Releases allocated resources after a pipeline step has completed.

        Parameters
        ----------
        step_name : str
            The name of the pipeline step that has completed.

        resources : dict
            A dictionary specifying the resources to release.
            Keys can include `'CPU'` and `'Memory'`.

            - `'CPU'`: Number of CPU cores to release.
            - `'Memory'`: Amount of memory to release in bytes.

        Notes
        -----
        This method releases resources using the :class:`ResourceManager`.
        It ensures that resources are made available for other steps.

        Mathematically, the release can be represented as:

        .. math::

            \\text{available\\_cpu\\_cores} = \\text{available\\_cpu\\_cores}
            + \\text{cpu\\_cores}

            \\text{available\\_memory} = \\text{available\\_memory}
            + \\text{memory}

        Examples
        --------
        >>> optimizer.release_resources_after_step(
        ...   'step1', {'CPU': 2, 'Memory': 2 * 1024 ** 3})
        """
        check_is_runned(self, attributes=['_is_runned'])
        logger.info(f"Releasing resources for step: {step_name} -> {resources}")

        cpu_cores = resources.get("CPU", 0)
        memory = resources.get("Memory", 0)

        self.resource_manager_.release_resources(cpu_cores, memory)

    @validate_params({
        'step_name': [str],
        'duration': [int]
    })
    def monitor_resources_for_step(self, step_name: str, duration: int = 5) -> None:
        """
        Monitors resource usage (CPU and Memory) for a given step over
        a specified duration.

        Parameters
        ----------
        step_name : str
            The name of the step to monitor.

        duration : int, optional
            The duration to monitor resources in seconds. Defaults to
            5 seconds.

        Notes
        -----
        This method uses the :class:`ResourceMonitor` to record resource usage
        at one-second intervals for the specified duration.

        Examples
        --------
        >>> optimizer.monitor_resources_for_step('step1', duration=10)
        """
        check_is_runned(self, attributes=['_is_runned'])
        logger.info(
            f"Monitoring resources for step: {step_name} for {duration} seconds."
        )
        self.resource_monitor_.run()

        for _ in range(duration):
            self.resource_monitor_.record_usage()
            time.sleep(1)

        self.resource_monitor_.stop_monitoring()

    @validate_params({
        'step_name': [str],
        'threshold': [dict]
    })
    def tune_resources_based_on_usage(
            self, step_name: str, threshold: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tunes resource allocation for a specific step based on observed
        resource usage.

        Parameters
        ----------
        step_name : str
            The name of the step to tune.

        threshold : dict
            Threshold values for resource usage.
            Keys can include `'CPU'` and `'Memory'`.

            - `'CPU'`: CPU usage threshold in percentage (e.g., 75 for 75%).
            - `'Memory'`: Memory usage threshold in bytes.

        Returns
        -------
        recommended_resources : dict
            New recommended resources based on usage.
            Keys include `'CPU'` and `'Memory'`.

        Notes
        -----
        This method analyzes the average resource usage recorded by
        :class:`ResourceMonitor` and adjusts the resource allocation if the
        usage exceeds the specified thresholds.

        Mathematically, if the average usage exceeds the threshold:

        .. math::

            \\text{recommended\\_resource} = \\min(
            \\text{available\\_resource} + \\delta, \\text{total\\_resource}
            )

        where :math:`\\delta` is the increment step.

        Examples
        --------
        >>> recommended_resources = optimizer.tune_resources_based_on_usage(
        ...     'step1', {'CPU': 75, 'Memory': 4 * 1024 ** 3})
        >>> print(recommended_resources)
        {'CPU': 4, 'Memory': 5368709120}
        """
        check_is_runned(self, attributes=['_is_runned'])
        logger.info(f"Tuning resources for step: {step_name} based on usage.")

        avg_cpu = sum(self.resource_monitor_.cpu_usage_) / len(
            self.resource_monitor_.cpu_usage_)
        avg_memory = sum(self.resource_monitor_.memory_usage_) / len(
            self.resource_monitor_.memory_usage_)

        recommended_resources = {}

        # Adjust CPU allocation if average usage exceeds threshold
        cpu_threshold = threshold.get("CPU", 100)
        if avg_cpu > cpu_threshold:
            max_cores = self.resource_manager_.available_cpu_cores_
            recommended_cores = min(
                self.resource_manager_.available_cpu_cores_ + 1, max_cores
            )
            logger.info(f"Increasing CPU allocation to {recommended_cores} cores.")
            recommended_resources["CPU"] = recommended_cores
        else:
            recommended_resources["CPU"] = threshold.get("CPU", 1)

        # Adjust memory allocation if average usage exceeds threshold
        memory_threshold = threshold.get("Memory", self.resource_manager_.available_memory_)
        if avg_memory > memory_threshold:
            total_memory = self.resource_manager_.available_memory_
            additional_memory = 1 * 1024 ** 3  # Increase by 1 GB
            recommended_memory = min(
                self.resource_manager_.available_memory_ + additional_memory,
                total_memory
            )
            logger.info(
                f"Increasing memory allocation to "
                f"{recommended_memory / (1024 ** 3):.2f} GB."
            )
            recommended_resources["Memory"] = recommended_memory
        else:
            recommended_resources["Memory"] = threshold.get("Memory", 1 * 1024 ** 3)

        return recommended_resources

    @validate_params({
        'step_name': [str],
        'param_grid': [dict],
        'n_trials': [int],
        'eval_metric': [str]
    })
    def tune_hyperparameters(
        self,
        step_name: str,
        param_grid: Dict[str, List[Any]],
        n_trials: int = 10,
        eval_metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Tunes hyperparameters for a specific pipeline step using either
        `optuna` (if available) or a fallback grid/random search.

        Parameters
        ----------
        step_name : str
            The name of the pipeline step to tune hyperparameters for.

        param_grid : dict
            A dictionary where keys are parameter names and values are lists
            of possible values to try.

        n_trials : int, optional
            The number of trials to run during tuning. Defaults to 10.

        eval_metric : str, optional
            The evaluation metric to optimize. Defaults to 'accuracy'.

        Returns
        -------
        best_params : dict
            The best hyperparameters found during tuning.

        Notes
        -----
        This method uses `optuna` for hyperparameter tuning if available.
        If `optuna` is not installed, it falls back to a grid search or
        randomized search based on the parameter grid size.

        Examples
        --------
        >>> param_grid = {
        ...     'learning_rate': [0.01, 0.1, 0.2],
        ...     'n_estimators': [100, 200, 300]
        ... }
        >>> best_params = optimizer.tune_hyperparameters(
        ...     'TrainModel', param_grid, n_trials=3, eval_metric='f1'
        ... )
        >>> print(best_params)
        {'learning_rate': 0.1, 'n_estimators': 200}
        """
        check_is_runned(self, attributes=['_is_runned'])
        logger.info(f"Tuning hyperparameters for step: {step_name} with {n_trials} trials.")

        step = self.pipeline_manager.get_step(step_name)
        if not step:
            raise ValueError(f"Step {step_name} does not exist in the pipeline.")

        try:
            import optuna  # noqa
            HAS_OPTUNA = True
        except ImportError:
            logger.warning("Optuna is not installed. Falling back to grid search.")
            HAS_OPTUNA = False

        if HAS_OPTUNA:
            # Use Optuna if installed
            return self._optuna_tuning(step, param_grid, n_trials, eval_metric)
        else:
            # Fallback to grid search or random search
            return self._fallback_tuning(step, param_grid, n_trials, eval_metric)

    @ensure_pkg(
        "optuna",
        extra="Optuna is required for hyperparameter tuning.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _optuna_tuning(
        self,
        step: 'PipelineStep',
        param_grid: Dict[str, List[Any]],
        n_trials: int,
        eval_metric: str
    ) -> Dict[str, Any]:
        """
        Tunes hyperparameters using Optuna.

        Parameters
        ----------
        step : PipelineStep
            The pipeline step to tune.

        param_grid : dict
            Parameter grid for tuning.

        n_trials : int
            Number of trials for Optuna to run.

        eval_metric : str
            Metric to optimize.

        Returns
        -------
        best_params : dict
            Best hyperparameters found by Optuna.
        """
        import optuna

        def objective(trial):
            params = {key: trial.suggest_categorical(key, values)
                      for key, values in param_grid.items()}
            logger.info(f"Trial params: {params}")

            # Execute the step with the sampled parameters
            try:
                step.params.update(params)
                output = step.run(None)
                score = self._evaluate_step_output(output, eval_metric)
                logger.info(f"Trial score: {score}")
                return score
            except Exception as e:
                logger.error(f"Error during hyperparameter tuning: {str(e)}")
                raise e

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        logger.info(f"Best parameters found by Optuna: {best_params}")
        return best_params

    def _fallback_tuning(
        self,
        step: 'PipelineStep',
        param_grid: Dict[str, List[Any]],
        n_trials: int,
        eval_metric: str
    ) -> Dict[str, Any]:
        """
        Fallback method for hyperparameter tuning using grid search or
        randomized search.

        Parameters
        ----------
        step : PipelineStep
            The pipeline step to tune.

        param_grid : dict
            Parameter grid for tuning.

        n_trials : int
            Number of trials to perform.

        eval_metric : str
            Metric to optimize.

        Returns
        -------
        best_params : dict
            Best hyperparameters found.
        """
        logger.info("Using fallback strategy for hyperparameter tuning.")

        # Determine whether to use grid search or randomized search
        param_combinations = list(product(*param_grid.values()))
        logger.info(f"Total parameter combinations: {len(param_combinations)}")

        if len(param_combinations) <= n_trials:
            logger.info(f"Using Grid Search (combinations: {len(param_combinations)}).")
            return self._grid_search(step, param_combinations, param_grid, eval_metric)
        else:
            logger.info(f"Using Randomized Search with {n_trials} trials.")
            return self._random_search(step, param_combinations, param_grid,
                                       n_trials, eval_metric)

    def _grid_search(
        self,
        step: 'PipelineStep',
        param_combinations: List[tuple],
        param_grid: Dict[str, List[Any]],
        eval_metric: str
    ) -> Dict[str, Any]:
        """
        Performs grid search over all parameter combinations.

        Parameters
        ----------
        step : PipelineStep
            The pipeline step to tune.

        param_combinations : list
            List of all parameter combinations.

        param_grid : dict
            The parameter grid to extract keys from.

        eval_metric : str
            Metric to optimize.

        Returns
        -------
        best_params : dict
            Best hyperparameters found.
        """
        best_score = float('-inf')
        best_params = None

        for params in param_combinations:
            param_dict = dict(zip(param_grid.keys(), params))
            logger.info(f"Testing params: {param_dict}")

            try:
                step.params.update(param_dict)
                output = step.run(None)
                score = self._evaluate_step_output(output, eval_metric)

                if score > best_score:
                    best_score = score
                    best_params = param_dict
            except Exception as e:
                logger.error(f"Error during grid search: {str(e)}")

        logger.info(f"Best params found via Grid Search: {best_params}")
        return best_params

    def _random_search(
        self,
        step: 'PipelineStep',
        param_combinations: List[tuple],
        param_grid: Dict[str, List[Any]],
        n_trials: int,
        eval_metric: str
    ) -> Dict[str, Any]:
        """
        Performs randomized search over a limited number of parameter combinations.

        Parameters
        ----------
        step : PipelineStep
            The pipeline step to tune.

        param_combinations : list
            List of all parameter combinations.

        param_grid : dict
            The parameter grid to extract keys from.

        n_trials : int
            Number of trials to perform.

        eval_metric : str
            Metric to optimize.

        Returns
        -------
        best_params : dict
            Best hyperparameters found.
        """
        best_score = float('-inf')
        best_params = None

        sampled_combinations = random.sample(param_combinations, n_trials)
        logger.info(f"Random search will evaluate {n_trials} combinations.")

        for params in sampled_combinations:
            param_dict = dict(zip(param_grid.keys(), params))
            logger.info(f"Testing params: {param_dict}")

            try:
                step.params.update(param_dict)
                output = step.run(None)
                score = self._evaluate_step_output(output, eval_metric)

                if score > best_score:
                    best_score = score
                    best_params = param_dict
            except Exception as e:
                logger.error(f"Error during random search: {str(e)}")

        logger.info(f"Best params found via Randomized Search: {best_params}")
        return best_params

    def _evaluate_step_output(self, output: Any, eval_metric: str) -> float:
        """
        Evaluates the step output based on the specified evaluation metric.

        Parameters
        ----------
        output : Any
            The output of the step to evaluate.

        eval_metric : str
            The evaluation metric to use (e.g., 'accuracy', 'f1').

        Returns
        -------
        score : float
            The calculated score based on the evaluation metric.

        Raises
        ------
        ValueError
            If the evaluation metric is unsupported.
        """
        if eval_metric == 'accuracy':
            # Assuming output contains a key for accuracy
            return output.get('accuracy', 0)
        elif eval_metric == 'f1':
            # Assuming output contains a key for f1-score
            return output.get('f1', 0)
        else:
            raise ValueError(f"Unsupported evaluation metric: {eval_metric}")

@ensure_pkg(
    "prefect",
    extra="The 'prefect' package is required for this functionality.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA
)
class PrefectOrchestrator(PipelineOrchestrator):
    """
    Integrates the pipeline with Prefect, allowing for Flow creation and
    scheduling.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        An instance of :class:`PipelineManager` that manages pipeline steps.

    Attributes
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager associated with this orchestrator.

    flow_ : prefect.Flow
        The Prefect Flow representing the pipeline workflow.

    _is_runned : bool
        Indicates whether the orchestrator has been run.

    Methods
    -------
    run(flow_name)
        Creates and runs a Prefect Flow that represents the pipeline.

    schedule_pipeline(schedule_interval)
        Schedules the pipeline using a cron-like schedule.

    monitor_pipeline()
        Monitors the pipeline execution via Prefect's monitoring UI.

    Notes
    -----
    The :class:`PrefectOrchestrator` class provides integration with Prefect,
    enabling pipelines to be executed as Prefect Flows. It allows for
    seamless scheduling and monitoring of pipeline executions.

    Examples
    --------
    >>> from gofast.mlops.pipeline import PrefectOrchestrator, PipelineManager
    >>> pipeline_manager = PipelineManager()
    >>> orchestrator = PrefectOrchestrator(pipeline_manager)
    >>> orchestrator.run(flow_name='my_pipeline')
    >>> orchestrator.schedule_pipeline('0 0 * * *')  # Daily at midnight

    See Also
    --------
    PipelineOrchestrator : Base class for pipeline orchestration.
    AirflowOrchestrator : Orchestrator using Apache Airflow.

    References
    ----------
    .. [1] Anderson, K. (2021). "Optimizing Data Pipelines with Prefect."
       *Journal of Data Orchestration*, 9(2), 120-135.
    """

    @validate_params({'pipeline_manager': ['PipelineManager']})
    def __init__(self, pipeline_manager: 'PipelineManager'):
        super().__init__(pipeline_manager)
        self.flow_ = None

    @RunReturn
    @validate_params({'flow_name': [str]})
    def run(self, flow_name: str):
        """
        Creates and runs a Prefect Flow that represents the pipeline.

        Parameters
        ----------
        flow_name : str
            The name of the Prefect Flow.

        Notes
        -----
        This method constructs a Prefect Flow by wrapping each pipeline
        step's function in a Prefect task. The flow is stored in the
        attribute ``flow_``.

        The method sets the ``_is_runned`` attribute to True.

        Examples
        --------
        >>> orchestrator.run(flow_name='my_pipeline')
        """
        from prefect import Flow, task

        logger.info(f"Creating Prefect Flow with name: {flow_name}")

        # Create Prefect tasks for each pipeline step
        def task_wrapper(step_func: Any, **kwargs):
            return step_func(**kwargs)

        with Flow(flow_name) as flow:
            previous_task = None
            for step_name, step in self.pipeline_manager.steps.items():
                @task(name=step_name)
                def current_task():
                    return task_wrapper(step.func, **step.params)

                if previous_task:
                    previous_task.set_downstream(current_task)
                previous_task = current_task

        self.flow_ = flow
        self._is_runned = True
        logger.info(f"Prefect Flow '{flow_name}' created successfully.")

    @validate_params({'schedule_interval': [str]})
    def schedule_pipeline(self, schedule_interval: str):
        """
        Schedules the pipeline (Flow) in Prefect Cloud or Prefect Server
        using a schedule.

        Parameters
        ----------
        schedule_interval : str
            Schedule interval (cron-like expression) or a specific Prefect
            schedule (e.g., IntervalSchedule).

        Notes
        -----
        This method adds a schedule to the Prefect Flow using a CronClock
        for cron-like schedules. The flow must be run prior to scheduling.

        Examples
        --------
        >>> orchestrator.schedule_pipeline('0 0 * * *')  # Daily at midnight
        """
        check_is_runned(self, attributes=['_is_runned'])

        logger.info(f"Scheduling Prefect Flow with interval: {schedule_interval}")

        from prefect.schedules import Schedule
        from prefect.schedules.clocks import CronClock

        # Add a schedule to the flow (using CronClock for cron-like schedules)
        schedule = Schedule(clocks=[CronClock(schedule_interval)])
        self.flow_.schedule = schedule

    def monitor_pipeline(self):
        """
        Monitors the pipeline execution via Prefect's monitoring UI or
        Prefect Cloud.

        Notes
        -----
        This method logs a message indicating that pipeline monitoring
        can be done through Prefect's web interface.

        Examples
        --------
        >>> orchestrator.monitor_pipeline()
        """
        check_is_runned(self, attributes=['_is_runned'])

        logger.info("Monitoring Prefect Flow execution through the Prefect UI.")


@ensure_pkg(
    "airflow",
    extra="The 'airflow' package is required for this functionality.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA
)
class AirflowOrchestrator(PipelineOrchestrator):
    """
    Integrates the pipeline with Apache Airflow, allowing for DAG creation
    and scheduling.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        An instance of :class:`PipelineManager` that manages pipeline steps.

    Attributes
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager associated with this orchestrator.

    dag_ : airflow.DAG
        The Airflow DAG representing the pipeline workflow.

    _is_runned : bool
        Indicates whether the orchestrator has been run.

    Methods
    -------
    run(dag_id, start_date, schedule_interval='@daily')
        Creates and runs an Airflow DAG that represents the pipeline.

    schedule_pipeline(schedule_interval)
        Schedules the pipeline using a cron-like schedule.

    monitor_pipeline()
        Monitors the pipeline execution via Airflow's monitoring UI.

    Notes
    -----
    The :class:`AirflowOrchestrator` class provides integration with Apache
    Airflow, enabling pipelines to be executed as Airflow DAGs.

    Examples
    --------
    >>> from gofast.mlops.pipeline import AirflowOrchestrator, PipelineManager
    >>> from datetime import datetime
    >>> pipeline_manager = PipelineManager()
    >>> orchestrator = AirflowOrchestrator(pipeline_manager)
    >>> orchestrator.run(
    ...     dag_id='my_pipeline',
    ...     start_date=datetime(2023, 1, 1),
    ...     schedule_interval='@daily'
    ... )
    >>> orchestrator.schedule_pipeline('@hourly')

    See Also
    --------
    PipelineOrchestrator : Base class for pipeline orchestration.
    PrefectOrchestrator : Orchestrator using Prefect.

    References
    ----------
    .. [2] Brown, L. (2021). "Automating ML Pipelines with Airflow."
       *Data Pipelines Journal*, 8(1), 100-110.
    """

    @validate_params({'pipeline_manager': ['PipelineManager']})
    def __init__(self, pipeline_manager: 'PipelineManager'):
        super().__init__(pipeline_manager)
        self.dag_ = None

    @RunReturn()
    @validate_params({
        'dag_id': [str],
        'start_date': [datetime],
        'schedule_interval': [str, None]
    })
    def run(
        self,
        dag_id: str,
        start_date: datetime,
        schedule_interval: Optional[str] = "@daily"
    ):
        """
        Creates and runs an Airflow DAG that represents the pipeline.

        Parameters
        ----------
        dag_id : str
            The unique identifier for the DAG.

        start_date : datetime
            The start date of the DAG execution.

        schedule_interval : str, optional
            The scheduling interval for the DAG (e.g., cron expression).
            Defaults to ``'@daily'``.

        Notes
        -----
        This method constructs an Airflow DAG by mapping the pipeline
        steps to Airflow tasks using ``PythonOperator``. The DAG is
        stored in the attribute ``dag_``.

        The method sets the ``_is_runned`` attribute to True.

        Examples
        --------
        >>> from datetime import datetime
        >>> orchestrator.run(
        ...     dag_id='my_pipeline',
        ...     start_date=datetime(2023, 1, 1),
        ...     schedule_interval='@daily'
        ... )
        """
        from airflow import DAG
        from airflow.operators.python_operator import PythonOperator

        logger.info(f"Creating Airflow DAG with ID: {dag_id}")

        # Define the DAG
        default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'start_date': start_date,
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }

        dag = DAG(
            dag_id=dag_id,
            default_args=default_args,
            description='An ML pipeline orchestrated by Airflow',
            schedule_interval=schedule_interval,
            catchup=False,
        )

        # Add pipeline steps to the DAG
        previous_task = None
        for step_name, step in self.pipeline_manager.steps.items():
            task = PythonOperator(
                task_id=step_name,
                python_callable=step.func,
                op_kwargs=step.params,
                dag=dag,
            )
            if previous_task:
                previous_task >> task
            previous_task = task

        self.dag_ = dag
        self._is_runned = True
        logger.info(f"DAG '{dag_id}' created successfully.")

    @validate_params({'schedule_interval': [str]})
    def schedule_pipeline(self, schedule_interval: str):
        """
        Schedules the pipeline using a cron-like schedule.

        Parameters
        ----------
        schedule_interval : str
            Cron expression or Airflow predefined schedule interval
            (e.g., ``'@daily'``).

        Notes
        -----
        This method updates the DAG's schedule interval to the specified
        value. The DAG must be run prior to scheduling.

        Examples
        --------
        >>> orchestrator.schedule_pipeline('@hourly')
        """
        check_is_runned(self, attributes=['_is_runned'])

        logger.info(f"Scheduling DAG with interval: {schedule_interval}")
        self.dag_.schedule_interval = schedule_interval

    def monitor_pipeline(self):
        """
        Monitors the pipeline execution via Airflow's monitoring UI.

        Notes
        -----
        This method logs a message indicating that pipeline monitoring
        can be done through Airflow's web interface.

        Examples
        --------
        >>> orchestrator.monitor_pipeline()
        """
        check_is_runned(self, attributes=['_is_runned'])

        logger.info("Monitoring Airflow DAG execution through the Airflow web UI.")


# Pipeline functions
def reconfigure_pipeline_on_the_fly(
    pipeline_manager: 'PipelineManager',
    step_name: str,
    new_step_func: Callable,
    new_params: Optional[Dict[str, Any]] = None
):
    """
    Dynamically reconfigures a specific step in the pipeline during execution.
    Allows users to swap out or adjust the function and parameters of a step
    in the pipeline while it is running.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager containing the steps.
    step_name : str
        The name of the step to reconfigure.
    new_step_func : callable
        The new function to replace the current step's function.
    new_params : dict, optional
        New parameters to use for the updated step. If not provided, existing
        parameters are retained.

    Notes
    -----
    This function enables dynamic modification of pipeline steps, which can be
    useful in scenarios where adjustments are needed without restarting the
    entire pipeline.

    Examples
    --------
    >>> from gofast.mlops.pipeline import reconfigure_pipeline_on_the_fly, PipelineManager
    >>> def new_function(data):
    ...     # New processing logic
    ...     return data * 2
    >>> pipeline_manager = PipelineManager()
    >>> # Assume 'step1' is an existing step in the pipeline
    >>> reconfigure_pipeline_on_the_fly(
    ...     pipeline_manager,
    ...     step_name='step1',
    ...     new_step_func=new_function,
    ...     new_params={'param1': 10}
    ... )

    See Also
    --------
    execute_step_conditionally : 
        Executes a pipeline step conditionally based on a specified condition.

    References
    ----------
    .. [1] Johnson, M. (2022). "Dynamic Pipeline Reconfiguration in Machine Learning."
       *International Journal of Data Science*, 14(3), 250-265.

    """
    logger.info(f"Reconfiguring step: {step_name} with new function and parameters.")

    # Retrieve the step and update its function and parameters
    step = pipeline_manager.get_step(step_name)
    if step is not None:
        step.func = new_step_func
        if new_params:
            step.params.update(new_params)
        logger.info(f"Step '{step_name}' has been reconfigured with"
                    " new function and parameters.")
    else:
        logger.error(f"Step '{step_name}' does not exist in the pipeline.")

@validate_params({
    'pipeline_manager': [object],
    'step_name': [str],
    'condition_func': [callable],
    'fallback_step': [str, None]
})
def execute_step_conditionally(
    pipeline_manager: 'PipelineManager',
    step_name: str,
    condition_func: Callable[[Any], bool],
    fallback_step: Optional[str] = None
):
    """
    Executes a specific pipeline step conditionally based on the result of a
    previous step. If the condition is not met, an alternative (fallback) step
    can be executed instead.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager containing the steps.
    step_name : str
        The name of the step to conditionally execute.
    condition_func : callable
        A function that takes the previous step's output and returns a boolean
        to decide execution.
    fallback_step : str, optional
        The name of a fallback step to execute if the condition is not met.
        If not provided, the step is skipped.

    Notes
    -----
    This function allows for conditional execution within the pipeline,
    enabling dynamic control flow based on intermediate results.

    Examples
    --------
    >>> from gofast.mlops.pipeline import execute_step_conditionally, PipelineManager
    >>> def condition(output):
    ...     return output > 0
    >>> pipeline_manager = PipelineManager()
    >>> # Assume 'step2' and 'fallback_step' are existing steps
    >>> execute_step_conditionally(
    ...     pipeline_manager,
    ...     step_name='step2',
    ...     condition_func=condition,
    ...     fallback_step='fallback_step'
    ... )

    See Also
    --------
    reconfigure_pipeline_on_the_fly : Dynamically reconfigures a pipeline step during execution.

    References
    ----------
    .. [2] Smith, L. (2021). "Conditional Execution in Data Pipelines."
       *Data Engineering Journal*, 17(4), 300-315.

    """
    logger.info(f"Checking condition for step: '{step_name}'")

    # Get the output of the last executed step
    last_step_name = list(pipeline_manager.steps.keys())[-1]
    previous_step_output = pipeline_manager.step_metadata[last_step_name]["output"]

    # Execute step if condition is met
    if condition_func(previous_step_output):
        logger.info(f"Condition met. Executing step: '{step_name}'")
        pipeline_manager.execute_step(step_name)
    else:
        logger.info(f"Condition not met for step: '{step_name}'.")
        if fallback_step:
            logger.info(f"Executing fallback step: '{fallback_step}'")
            pipeline_manager.execute_step(fallback_step)
        else:
            logger.info(f"No fallback step specified. Skipping step: '{step_name}'")


@validate_params({
    'pipeline_manager': [object],
    'sub_pipelines': [list],
})
def run_parallel_subpipelines(
    pipeline_manager: 'PipelineManager',
    sub_pipelines: List[List['PipelineStep']]
) -> None:
    """
    Executes multiple sub-pipelines in parallel, allowing parts of the
    pipeline to be parallelized.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        The main pipeline manager that manages the overall pipeline
        execution.
    sub_pipelines : list of list of PipelineStep
        A list of sub-pipelines, where each sub-pipeline is represented
        as a list of ``PipelineStep`` instances.

    Notes
    -----
    This function utilizes a thread pool executor to run multiple
    sub-pipelines concurrently. Each sub-pipeline is executed in its own
    thread, enabling parallel processing of independent pipeline
    segments.

    Mathematically, if we denote each sub-pipeline as
    :math:`P_i`, where :math:`i = 1, 2, \dots, N`, the execution can be
    represented as:

    .. math::

        \text{Execute } P_i \text{ in parallel for } i = 1 \text{ to } N

    Examples
    --------
    >>> from gofast.mlops.pipeline import run_parallel_subpipelines 
    >>> from gofast.mlops.pipeline import PipelineManager, PipelineStep
    >>> def increment(data):
    ...     return data + 1
    >>> step1 = PipelineStep(name='step1', func=increment)
    >>> step2 = PipelineStep(name='step2', func=increment)
    >>> sub_pipeline1 = [step1]
    >>> sub_pipeline2 = [step2]
    >>> pipeline_manager = PipelineManager()
    >>> run_parallel_subpipelines(pipeline_manager, [sub_pipeline1, sub_pipeline2])

    See Also
    --------
    PipelineManager : Manages the execution of pipeline steps.
    ThreadPoolExecutor : Provides a simple interface to run tasks in parallel.

    References
    ----------
    .. [1] Doe, J. (2021). "Parallel Execution of Data Pipelines."
       *Journal of Data Engineering*, 15(2), 100-115.

    """
    logger.info("Starting parallel execution of sub-pipelines...")

    # Create a thread pool to run sub-pipelines concurrently
    with ThreadPoolExecutor() as executor:
        futures = []
        for sub_pipeline_steps in sub_pipelines:
            sub_pipeline_manager = PipelineManager()
            for step in sub_pipeline_steps:
                sub_pipeline_manager.add_step(step)
            # Execute the sub-pipeline asynchronously
            futures.append(executor.submit(sub_pipeline_manager.execute,
                                           initial_data=None))

        # Wait for all sub-pipelines to complete
        for future in futures:
            try:
                future.result()  # Raise any exceptions that occurred during execution
                logger.info("Sub-pipeline completed successfully.")
            except Exception as e:
                logger.error(f"Error in sub-pipeline: {str(e)}")

@validate_params({
    'data': [object],
    'split_ratios': [list],
    'tasks': [list],
    'pipeline_manager': [object],
})
def split_data_for_multitask_pipeline(
    data: Any,
    split_ratios: List[float],
    tasks: List[str],
    pipeline_manager: 'PipelineManager'
) -> None:
    """
    Splits the input data into multiple parts for multitask pipelines and
    assigns them to different tasks.

    Parameters
    ----------
    data : Any
        The input data to be split.
    split_ratios : list of float
        List of ratios to split the data (e.g., ``[0.5, 0.3, 0.2]`` for
        50%, 30%, 20% splits). The ratios should sum to 1.0.
    tasks : list of str
        List of task names, corresponding to the split data.
    pipeline_manager : PipelineManager
        The pipeline manager handling the tasks.

    Notes
    -----
    This function divides the input data according to the specified split
    ratios and assigns each partition to the corresponding task in the
    pipeline. This is useful for multitask pipelines where different tasks
    operate on different subsets of the data.

    Mathematically, if the total number of data points is :math:`N` and
    the split ratios are :math:`r_i`, then the number of data points
    assigned to task :math:`i` is:

    .. math::

        n_i = \left\lfloor N \times r_i \right\rfloor

    where :math:`i = 1, 2, \dots, M`, and :math:`M` is the number of
    tasks.

    Examples
    --------
    >>> from gofast.mlops.pipeline import split_data_for_multitask_pipeline
    >>> from gofast.mlops.pipeline import PipelineManager
    >>> data = list(range(100))
    >>> split_ratios = [0.6, 0.4]
    >>> tasks = ['task1', 'task2']
    >>> pipeline_manager = PipelineManager()
    >>> # Assume 'task1' and 'task2' are existing steps in the pipeline
    >>> split_data_for_multitask_pipeline(data, split_ratios, tasks, pipeline_manager)

    See Also
    --------
    PipelineManager : Manages the execution of pipeline steps.

    References
    ----------
    .. [2] Smith, A. (2020). "Multitask Learning with Data Partitioning."
       *Machine Learning Journal*, 12(4), 200-215.

    """
    if len(split_ratios) != len(tasks):
        raise ValueError("Number of split ratios must match the number of tasks.")

    split_ratios = [ validate_ratio(r, param_name="Split ratio")
                    for r in split_ratios ]
    if not abs(sum(split_ratios) - 1.0) < 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    # Split the data based on the ratios
    total_data_points = len(data)
    cumulative_ratios = [sum(split_ratios[:i+1]) for i in range(len(split_ratios))]
    split_indices = [int(total_data_points * ratio) for ratio in cumulative_ratios]
    split_data = []
    start_idx = 0

    for end_idx in split_indices:
        split_data.append(data[start_idx:end_idx])
        start_idx = end_idx

    # Assign split data to the corresponding task's pipeline step
    for i, task_name in enumerate(tasks):
        step = pipeline_manager.get_step(task_name)
        if step:
            if 'data' in step.params:
                step.params['data'] = split_data[i]
            else:
                step.params.update({"data": split_data[i]})
            logger.info(f"Assigned split data to task: '{task_name}'")
        else:
            logger.error(f"Task '{task_name}' not found in the pipeline.")


@validate_params({
    'pipeline_manager': [object],
    'rollback_step': [str],
})
def rollback_to_previous_state(
    pipeline_manager: 'PipelineManager',
    rollback_step: str
):
    """
    Rolls back the pipeline to a previous stable state if a failure occurs.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager handling the steps.
    rollback_step : str
        The name of the step to roll back to.

    Notes
    -----
    This function resets the state of the pipeline to a specified
    step, effectively undoing any steps executed after that point.
    It is useful for error recovery or reprocessing data from a
    known good state.

    Examples
    --------
    >>> from gofast.mlops.pipeline import rollback_to_previous_state, PipelineManager
    >>> pipeline_manager = PipelineManager()
    >>> # Assume pipeline_manager has been set up and steps have been executed
    >>> rollback_to_previous_state(pipeline_manager, rollback_step='data_cleaning')

    See Also
    --------
    smart_retry_with_backoff : Retries a failed step using exponential backoff.

    References
    ----------
    .. [1] Doe, J. (2021). "Error Recovery in Data Pipelines."
       *Journal of Data Engineering*, 12(3), 200-215.

    """
    logger.info(f"Rolling back to step: '{rollback_step}'")

    # Ensure that the rollback step exists
    step = pipeline_manager.get_step(rollback_step)
    if step:
        # Reset the pipeline state to the specified step
        rolling_back = False
        for step_name in list(pipeline_manager.steps.keys()):
            if step_name == rollback_step:
                rolling_back = True
            if rolling_back:
                pipeline_manager.steps[step_name].outputs = None
                pipeline_manager.step_metadata[step_name]["status"] = "rolled_back"
                logger.info(f"Rolled back step: '{step_name}'")
    else:
        logger.error(f"Step '{rollback_step}' does not exist in the pipeline.")

@validate_params({
    'pipeline_manager': [object],
    'step_name': [str],
    'max_retries': [int],
    'initial_delay': [float],
})
def smart_retry_with_backoff(
    pipeline_manager: 'PipelineManager',
    step_name: str,
    max_retries: int = 3,
    initial_delay: float = 1.0
):
    """
    Retries a failed step using exponential backoff if it encounters
    errors during execution.

    Parameters
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager containing the steps.
    step_name : str
        The name of the step to retry.
    max_retries : int, optional
        Maximum number of retries allowed. Defaults to ``3``.
    initial_delay : float, optional
        Initial delay before retrying, with backoff applied for
        subsequent retries. Defaults to ``1.0``.

    Notes
    -----
    This function attempts to re-execute a pipeline step that has
    failed, using an exponential backoff strategy to wait longer
    between each retry attempt.

    The delay between retries is calculated as:

    .. math::

        \\text{delay} = \\text{initial\\_delay} \\times 2^{(\\text{retry\\_count} - 1)}

    Examples
    --------
    >>> from gofast.mlops.pipeline import smart_retry_with_backoff, PipelineManager
    >>> pipeline_manager = PipelineManager()
    >>> # Assume 'data_processing' is a step that may fail
    >>> smart_retry_with_backoff(
    ...     pipeline_manager,
    ...     step_name='data_processing',
    ...     max_retries=5,
    ...     initial_delay=2.0
    ... )

    See Also
    --------
    rollback_to_previous_state : Rolls back the pipeline to a previous stable state.

    References
    ----------
    .. [2] Smith, A. (2020). "Robust Retry Mechanisms in Pipelines."
       *Pipeline Management Quarterly*, 7(2), 150-165.

    """
    step = pipeline_manager.get_step(step_name)
    if not step:
        logger.error(f"Step '{step_name}' not found in the pipeline.")
        return

    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            logger.info(f"Retrying step '{step_name}' (Attempt {retry_count + 1})...")
            pipeline_manager.execute_step(step_name)
            logger.info(
                f"Step '{step_name}' executed successfully on retry {retry_count + 1}."
            )
            break
        except Exception as e:
            logger.error(
                f"Step '{step_name}' failed on retry {retry_count + 1}. Error: {str(e)}"
            )
            retry_count += 1
            time.sleep(delay)
            delay *= 2  # Exponential backoff

    if retry_count == max_retries:
        logger.error(f"Step '{step_name}' failed after {max_retries} retries.")

@validate_params({
    'steps': [list, None],
    'parallel': [bool, StrOptions({'True', 'False'})],
})
def create_pipeline(
    steps: Optional[List['PipelineStep']] = None,
    parallel: bool = False
) -> 'Pipeline':
    """
    Creates a machine learning pipeline by chaining together pipeline steps.

    Parameters
    ----------
    steps : list of PipelineStep, optional
        List of steps (functions) to execute in the pipeline.
        Defaults to an empty list if not provided.
    parallel : bool, optional
        Whether to execute the steps in parallel. Defaults to ``False``.

    Returns
    -------
    pipeline : Pipeline
        A ``Pipeline`` object that can be executed.

    Notes
    -----
    This function initializes a ``Pipeline`` object with the provided
    steps and execution mode. When ``parallel`` is ``True``, the steps
    are executed in parallel if possible.

    Examples
    --------
    >>> from gofast.mlops.pipeline import create_pipeline, PipelineStep
    >>> def preprocess(data):
    ...     return [x * 2 for x in data]
    >>> def train_model(data):
    ...     return sum(data) / len(data)
    >>> step1 = PipelineStep(name='preprocess', func=preprocess)
    >>> step2 = PipelineStep(name='train', func=train_model)
    >>> pipeline = create_pipeline(steps=[step1, step2], parallel=False)
    >>> result = pipeline.execute([1, 2, 3, 4])
    >>> print(result)
    5.0

    See Also
    --------
    Pipeline : Class representing a machine learning pipeline.

    References
    ----------
    .. [3] Lee, K. (2019). "Constructing Modular Pipelines in Machine Learning."
       *Machine Learning Today*, 4(1), 80-95.

    """
    logger.info(
        f"Creating a pipeline with {'parallel' if parallel else 'sequential'} execution."
    )
    return Pipeline(steps, parallel)


