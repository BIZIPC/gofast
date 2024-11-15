# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides the configuration settings for the `gofast` package, 
allowing users to customize various aspects of its behavior, including 
automatic dependency installation, backend selection, logging verbosity, 
parallel processing, memory management, and more.

Features
--------

- **Backend Selection**:
  The module allows for dynamic switching between different computational 
  backends such as NumPy, SciPy, Dask, and CuPy. Each backend is designed to 
  handle different types of workloads, including large-scale parallel processing.

- **Auto-Dependency Installation**:
  The `Configure` class allows automatic installation of missing dependencies 
  when enabled, simplifying the setup process for users.

- **Logging and Verbosity**:
  Provides fine-grained control over logging levels, from no logging to full 
  debug-level verbosity. The package logs detailed information about the 
  computation pipeline, making it easier to debug and optimize workflows.

- **Random Seed and Reproducibility**:
  Users can set a global random seed for consistent results in experiments 
  involving stochastic processes. This is critical for reproducibility in 
  scientific computing.

- **Parallel Processing and Thread Management**:
  Enables or disables parallel computation and sets thread limits for multi-core 
  processing, making it easier to control resource usage during computation-heavy 
  operations.

- **Memory Limiting**:
  On Unix-like systems, the module provides functionality to set memory usage 
  limits to prevent excessive memory consumption during large computations.

Backend Options
---------------
Available backends for computation:
- `numpy`: Standard backend using NumPy.
- `scipy`: Backend using SciPy for scientific computations.
- `dask`: Backend using Dask for parallel and distributed computations.
- `cupy`: Backend using CuPy for GPU-accelerated computations.

How to Use
----------
The `Configure` class offers methods for setting global configurations for the 
entire `gofast` package. Users can toggle various settings such as automatic 
installation of dependencies, backend selection, verbosity levels, and more.

Example:

>>> from gofast.config import Configure
>>> config = Configure(
...     auto_install_dependencies=True, 
...     verbosity=3, 
...     parallel_processing=True, 
...     random_seed=42
... )
>>> config.set_backend('dask')  # Switch to Dask backend for parallel computation

Module Attributes
-----------------
_current_backend : str
    Holds the current backend name used for computations. Default is 'numpy'.

_backend_classes : dict
    Maps backend names (e.g., 'numpy', 'scipy') to their corresponding backend 
    classes, allowing dynamic backend selection.

References
----------
.. _NumPy: https://numpy.org/
.. _SciPy: https://scipy.org/
.. _Dask: https://dask.org/
.. _CuPy: https://cupy.dev/
"""
import subprocess
import os
import logging
import warnings
import random
import numpy as np
import resource  # For Unix-like systems (Linux/macOS)
from numbers import Integral 
from typing import Optional, Union 

from .api.property import BaseClass
from .backends.numpy import NumpyBackend
from .backends.scipy import ScipyBackend 
from .backends.dask import DaskBackend 
from .backends.cupy import CuPyBackend 
from .compat.sklearn import validate_params, StrOptions

from ._gofastlog import gofastlog
logger = gofastlog.get_gofast_logger(__name__)

# Global variable to hold the name of the current backend
_current_backend = 'numpy'  # Default to NumPy

# Dictionary mapping backend names to their corresponding classes
_backend_classes = {
    'numpy': NumpyBackend,
    'scipy': ScipyBackend,
    'dask': DaskBackend,
    'cupy': CuPyBackend,
}


__all__ =["Configure", "set_backend", "get_backend"]


class Configure(BaseClass):
    """
    A class for managing and customizing the behavior of the `gofast` package.
    Provides flexibility to control features like dependency installation,
    logging, formatting, random seeds, parallel processing, and more.

    Parameters
    ----------
    auto_install_dependencies : bool, optional
        If True, the package will automatically install required dependencies.
        Default is False.
    verbosity : int, optional
        Controls the level of logging detail.
        0 = No logging,
        1 = Errors only,
        2 = Warnings,
        3 = Info,
        4 = Debug.
        Default is 3 (Info).
    auto_format : bool, optional
        If True, all output will automatically be formatted according to the
        package's standards (e.g., data output format). Default is True.
    parallel_processing : bool, optional
        If True, enables parallel processing features where supported in the
        package. Default is False.
    cache_results : bool, optional
        If True, caches intermediate results in computation-heavy processes.
        Default is True.
    random_seed : int or None, optional
        Sets a global random seed for reproducibility. Default is None.
    thread_limit : int or None, optional
        The maximum number of threads to use in parallel processing.
        Default is None (no limit).
    warnings_enabled : bool, optional
        If True, warnings will be displayed. Default is True.
    memory_limit : int or None, optional
        Sets a memory limit for large computations, in megabytes (MB).
        Default is None (no limit).
    custom_install_command : str, optional
        Custom command for installing dependencies (e.g., using `conda` or
        `pip`). Default is "pip install".
    install_on_import : bool, optional
        If True, attempts to install missing dependencies on package import.
        Default is False.
    env_file : str or None, optional
        Path to a file (e.g., .env or .ini) for loading environment variables
        to customize package behavior. Default is None.
    run_report: str or None,optional 
       Set the run return behavior if the `run_return` configuration is 
       provided by the user. This ensures that the global behavior for 
       returning values (either 'self', 'attribute', or None(both)) is applied 
       based on the configuration.

    Notes
    -----
    The `Configure` class allows users to set up and modify various settings
    within the `gofast` package, enhancing flexibility and control over
    package behavior. It provides methods to adjust logging verbosity,
    manage dependency installation, set random seeds for reproducibility,
    control parallel processing, and more.

    Examples
    --------
    >>> from gofast.config import Configure
    >>> config = Configure(
    ...     auto_install_dependencies=True,
    ...     verbosity=4,
    ...     auto_format=True,
    ...     parallel_processing=True,
    ...     install_on_import=True,
    ...     random_seed=42,
    ...     thread_limit=8,
    ...     memory_limit=2048,
    ...     env_file='.env'
    ... )
    >>> config.set_verbosity(2)
    >>> config.toggle_auto_install(False)

    See Also
    --------
    logging : Logging facility for Python.
    threading : Higher-level threading interface.
    numpy.random.seed : Seed the generator.

    References
    ----------
    .. [1] "Python Logging Module," Python Software Foundation.
           https://docs.python.org/3/library/logging.html
    .. [2] "NumPy Random Seed," NumPy Documentation.
           https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    """
    
    @validate_params(
        {
            'auto_install_dependencies': [bool],
            'verbosity': [Integral, bool], 
            'auto_format': [bool],
            'parallel_processing': [bool],
            'cache_results': [bool],
            'random_seed': [Integral, None], 
            'thread_limit': [Integral, None],  
            'warnings_enabled': [bool],
            'memory_limit': [Integral, None],  
            'custom_install_command': [str],  
            'install_on_import': [bool],
            'env_file': [str, None], 
            'run_report':[StrOptions({"self", "attribute"}), None]
        }
    )
    def __init__(
        self,
        auto_install_dependencies: bool = False,
        verbosity: Optional[Union [int, bool]] = 3,
        auto_format: bool = True,
        parallel_processing: bool = False,
        cache_results: bool = True,
        random_seed: Optional[int] = None,
        thread_limit: Optional[int] = None,
        warnings_enabled: bool = True,
        memory_limit: Optional[int] = None,
        custom_install_command: str = "pip install",
        install_on_import: bool = False,
        env_file: Optional[str] = None,
        run_return: Optional[str]=None
    ):
        self.auto_install_dependencies = auto_install_dependencies
        self.verbosity = verbosity
        self.auto_format = auto_format
        self.parallel_processing = parallel_processing
        self.cache_results = cache_results
        self.random_seed = random_seed
        self.thread_limit = thread_limit
        self.warnings_enabled = warnings_enabled
        self.memory_limit = memory_limit
        self.custom_install_command = custom_install_command
        self.install_on_import = install_on_import
        self.env_file = env_file
        self.run_return= run_return 
    
        # Set up logging
        self._setup_logging()

        # Set random seed
        if self.random_seed is not None:
            self._set_random_seed(self.random_seed)

        # Manage warnings
        self._configure_warnings()

        # Load environment variables from the file, if provided
        if self.env_file:
            self._load_env_file(self.env_file)

        # If install_on_import is True, check for missing dependencies
        if self.install_on_import:
            self._install_missing_dependencies()

        # Set thread limits for parallel processing
        if self.thread_limit is not None:
            self._set_thread_limit(self.thread_limit)

        # Set memory limit if provided
        if self.memory_limit is not None:
            self._set_memory_limit(self.memory_limit)
            
        # Set the run return behavior if the `run_return` configuration 
        # is provided by the user.
        # Apply the `run_return` configuration globally.
        if self.run_return is not None:
            self._set_run_return()  

    def toggle_auto_install(self, enable: bool):
        """
        Toggle the automatic installation of dependencies.

        Parameters
        ----------
        enable : bool
            If True, automatic installation of dependencies is enabled.
            If False, it is disabled.

        Returns
        -------
        None
        """
        self.auto_install_dependencies = enable
        logger.info("Auto-install dependencies set to %s", enable)

    def set_verbosity(self, level: int):
        """
        Set the verbosity level for logging.

        Parameters
        ----------
        level : int
            Verbosity level:
            0 = No logging,
            1 = Errors only,
            2 = Warnings,
            3 = Info,
            4 = Debug.

        Returns
        -------
        None
        """
        self.verbosity = level
        self._setup_logging()
        logger.info("Verbosity level set to %d", level)

    def set_parallel_processing(self, enable: bool):
        """
        Enable or disable parallel processing.

        Parameters
        ----------
        enable : bool
            If True, parallel processing is enabled.
            If False, it is disabled.

        Returns
        -------
        None
        """
        self.parallel_processing = enable
        logger.info("Parallel processing set to %s", enable)

    def set_auto_format(self, enable: bool):
        """
        Enable or disable automatic formatting.

        Parameters
        ----------
        enable : bool
            If True, automatic formatting is enabled.
            If False, it is disabled.

        Returns
        -------
        None
        """
        self.auto_format = enable
        logger.info("Auto-formatting set to %s", enable)

    def set_cache_results(self, enable: bool):
        """
        Enable or disable caching of results.

        Parameters
        ----------
        enable : bool
            If True, caching of results is enabled.
            If False, it is disabled.

        Returns
        -------
        None
        """
        self.cache_results = enable
        logger.info("Cache results set to %s", enable)

    def set_warnings_enabled(self, enable: bool):
        """
        Enable or disable warnings.

        Parameters
        ----------
        enable : bool
            If True, warnings are enabled.
            If False, warnings are suppressed.

        Returns
        -------
        None
        """
        self.warnings_enabled = enable
        self._configure_warnings()
        logger.info("Warnings enabled set to %s", enable)

    def set_thread_limit(self, limit: int):
        """
        Set the maximum number of threads.

        Parameters
        ----------
        limit : int
            The maximum number of threads to use for parallel processing.

        Returns
        -------
        None
        """
        self.thread_limit = limit
        self._set_thread_limit(limit)
        logger.info("Thread limit set to %d", limit)

    def set_memory_limit(self, limit: int):
        """
        Set a memory limit for large computations.

        Parameters
        ----------
        limit : int
            Memory limit in megabytes (MB).

        Returns
        -------
        None
        """
        self.memory_limit = limit
        self._set_memory_limit(limit)
        logger.info("Memory limit set to %d MB", limit)

    def set_random_seed(self, seed: int):
        """
        Set the global random seed for reproducibility.

        Parameters
        ----------
        seed : int
            The seed value to use for random number generation.

        Returns
        -------
        None
        """
        self.random_seed = seed
        self._set_random_seed(seed)
        logger.info("Random seed set to %d", seed)
        
    def set_run_return(self, return_type: str = 'self'):
        """
        Set the global behavior for the `run_return` function, which controls 
        whether methods return `self`, an attribute, or both.

        Parameters
        ----------
        return_type : str
            Specifies the return type. Options:
            'self' : Always return self.
            'attribute' : Return the attribute if it exists.
            'both' : Return a tuple of (self, attribute).

        Returns
        -------
        None
        """
        valid_options = ['self', 'attribute', None]
        if return_type not in valid_options:
            logger.error("Invalid run_return value '%s'. Expected one of %s",
                         return_type, valid_options)
            raise ValueError(f"Invalid return_type '{return_type}'."
                             " Must be one of {valid_options}.")
        
        self.run_return = return_type 
        logger.info("run_return behavior set to '%s'. This will control"
                    " return behavior globally.", return_type)

    # Private methods
    def _set_run_return(self): 
        """
        Configure run_return behavior based on the value of run_return.
        
        This private method ensures that the configured `run_return` behavior 
        is valid and applies it globally if needed.
        """
        valid_options = ['self', 'attribute', None]
        
        if self.run_return not in valid_options:
            logger.error("Invalid run_return setting: '%s'. Must be one of %s.", 
                         self.run_return, valid_options)
            raise ValueError(f"Invalid run_return '{self.run_return}'."
                             " Must be one of {valid_options}.")
    
        # Apply configuration to global behavior
        logger.info("run_return globally configured as '%s'.", self.run_return)

    def _setup_logging(self):
        """Configure logging based on verbosity level."""
        log_levels = {
            0: logging.NOTSET,
            1: logging.ERROR,
            2: logging.WARNING,
            3: logging.INFO,
            4: logging.DEBUG
        }
        logger.setLevel(log_levels.get(self.verbosity, logging.INFO))
        logger.info("Logging initialized. Current verbosity level: %d",
                    self.verbosity)

    def _set_random_seed(self, seed: int):
        """Set the global random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        logger.info("Random seed set to %d", seed)

    def _configure_warnings(self):
        """Enable or disable warnings based on user configuration."""
        if not self.warnings_enabled:
            warnings.filterwarnings('ignore')
            logger.info("Warnings are disabled.")
        else:
            warnings.resetwarnings()
            logger.info("Warnings are enabled.")

    def _load_env_file(self, env_file: str):
        """Load environment variables from a file."""
        logger.info("Loading environment variables from %s", env_file)
        if not os.path.exists(env_file):
            logger.error("Env file %s not found", env_file)
            return

        with open(env_file) as file:
            for line in file:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                    logger.debug("Environment variable %s set to %s", key, value)

    def _install_missing_dependencies(self):
        """Automatically install missing dependencies if enabled."""
        logger.info("Checking for missing dependencies...")
        required_dependencies = ["numpy", "pandas", "scikit-learn"]  # Example dependencies

        for dep in required_dependencies:
            if not self._check_dependency(dep):
                logger.warning("Missing dependency: %s", dep)
                if self.auto_install_dependencies:
                    logger.info("Auto-installing dependency: %s", dep)
                    self._install_dependency(dep)
                else:
                    logger.error("Dependency %s is missing. Please install it.", dep)

    def _check_dependency(self, package_name: str) -> bool:
        """Check if a package is installed."""
        try:
            __import__(package_name)
            logger.debug("Dependency %s is already installed", package_name)
            return True
        except ImportError:
            return False

    def _install_dependency(self, package_name: str):
        """Install a dependency using the specified command."""
        try:
            command = f"{self.custom_install_command} {package_name}"
            subprocess.check_call(command, shell=True)
            logger.info("Successfully installed %s", package_name)
        except subprocess.CalledProcessError:
            logger.error("Failed to install %s using command: %s", package_name, command)

    def _set_thread_limit(self, limit: int):
        """Set the maximum number of threads for parallel processing."""
        # Note: Limiting threads in Python is non-trivial and depends on the
        # specific libraries being used. This is a placeholder implementation.
        os.environ["OMP_NUM_THREADS"] = str(limit)
        os.environ["OPENBLAS_NUM_THREADS"] = str(limit)
        os.environ["MKL_NUM_THREADS"] = str(limit)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(limit)
        os.environ["NUMEXPR_NUM_THREADS"] = str(limit)
        logger.info("Thread limit set to %d", limit)


    def _set_memory_limit(self, limit: int):
        """
        Set a memory limit for large computations in megabytes (MB).
    
        Parameters
        ----------
        limit : int
            Memory limit in MB.
    
        Notes
        -----
        On Unix-like systems (Linux/macOS), this method uses the `resource` module 
        to set memory usage limits. On Windows, where the `resource` module is not 
        available, the implementation may require third-party libraries or system-specific 
        APIs, which are not provided here. A fallback mechanism is also included.
    
        Raises
        ------
        NotImplementedError
            If the platform does not support memory limiting via the `resource` module.
        """
        
        logger.info("Setting memory limit to %d MB", limit)
    
        # Convert limit from MB to bytes
        memory_limit_in_bytes = limit * 1024 * 1024
    
        try:
            # Unix-like system (Linux/macOS)
            if os.name == 'posix':
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                logger.debug("Current memory limits (soft: %d, hard: %d)", soft, hard)
    
                # Set both soft and hard limits to the desired memory limit
                resource.setrlimit(resource.RLIMIT_AS, (
                    memory_limit_in_bytes, memory_limit_in_bytes))
                logger.info("Memory limit set to %d MB", limit)
    
            else:
                # Windows or unsupported system
                raise NotImplementedError(
                    "Memory limiting is not natively supported on this platform.")
    
        except Exception as e:
            logger.error("Failed to set memory limit: %s", str(e))
            raise


def set_backend(backend_name: str):
    """
    Sets the active computational backend for gofast.

    Parameters
    ----------
    backend_name : str
        The name of the backend to set. Supported options are:
        'numpy', 'scipy', 'dask', and 'cupy'. The backend determines the
        underlying computational library used by gofast for numerical
        operations.

    Raises
    ------
    ValueError
        If the backend name provided is not one of the supported backends.
        The supported backends are 'numpy', 'scipy', 'dask', and 'cupy'.

    Notes
    -----
    The backend determines which computational library is used. For example:
    
    - 'numpy': Standard CPU-based numerical computations.
    - 'scipy': Advanced scientific computing on the CPU.
    - 'dask': Parallel computing, which can scale across multiple CPUs or clusters.
    - 'cupy': GPU-accelerated computing, leveraging CUDA for faster computations.

    Examples
    --------
    >>> from gofast.config import set_backend
    >>> set_backend('dask')
    Active backend set to: Dask
    
    See Also
    --------
    get_backend : Retrieves the current active backend class.
    
    References
    ----------
    .. _NumPy: https://numpy.org/
    .. _SciPy: https://scipy.org/
    .. _Dask: https://dask.org/
    .. _CuPy: https://cupy.dev/
    """
    global _current_backend
    if backend_name in _backend_classes:
        _current_backend = backend_name
        print(f"Active backend set to: {backend_name.capitalize()}")
    else:
        raise ValueError(
            f"Unsupported backend: {backend_name}. Supported backends: "
            f"{list(_backend_classes.keys())}"
        )


def get_backend():
    """
    Returns the active computational backend class for gofast.

    Returns
    -------
    backend_class : object
        The class of the currently active backend used for computations.
        The returned backend corresponds to the one set by `set_backend`.
    
    Notes
    -----
    The backend returned by this function can be one of the following:
    
    - `NumpyBackend`: For CPU-based numerical operations using NumPy.
    - `ScipyBackend`: For advanced scientific computations on the CPU using SciPy.
    - `DaskBackend`: For parallel and distributed computing using Dask.
    - `CuPyBackend`: For GPU-accelerated operations using CUDA.

    This function retrieves the backend as set by `set_backend`. If no backend 
    has been set explicitly, the default is 'numpy'.

    Examples
    --------
    >>> from gofast.config import get_backend
    >>> backend = get_backend()
    >>> print(backend)
    <class 'gofast.backends.numpy.NumpyBackend'>
    
    See Also
    --------
    set_backend : Sets the current active backend.
    
    References
    ----------
    .. _NumPy: https://numpy.org/
    .. _SciPy: https://scipy.org/
    .. _Dask: https://dask.org/
    .. _CuPy: https://cupy.dev/
    """
    backend_class = _backend_classes[_current_backend]
    return backend_class()


if __name__ == '__main__':

    # Set the active backend to NumPy
    # from gofast.config import set_backend, get_backend 
    
    set_backend('numpy')
    
    # Retrieve the active backend and use it
    backend = get_backend()
    print(type(backend))
    # # Example Usage
    # from gofast.config import set_backend, get_backend
    
    # Set the active backend to NumPy
    set_backend('numpy')
    
    # Get the current active backend
    backend = get_backend()
    
    # Perform operations using the active backend
    a = backend.array([1, 2, 3])
    b = backend.array([4, 5, 6])
    dot_product = backend.dot(a, b)
    
    print(dot_product)  
    # Output will depend on the active backend's implementation
