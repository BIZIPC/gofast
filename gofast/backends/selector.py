# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
Provides an interface to select different computation backends within 
the `gofast` package, ensuring flexibility  between various numerical and 
machine learning frameworks. It includes support for popular backends such 
as NumPy, SciPy, Dask, CuPy, and neural network (NN) libraries.

Classes
-------
BackendSelector
    A class for selecting the appropriate computational backend dynamically 
    based on the user's environment and needs. This class leverages the 
    `EnsureMethod` decorator to prioritize `select_backend` execution.

Dependencies
------------
- `NumpyBackend`: Supports standard numerical operations with NumPy.
- `ScipyBackend`: Adds functionality for scientific and advanced 
  mathematical operations with SciPy.
- `DaskBackend`: Provides parallel computation and scalable data 
  processing using Dask.
- `CuPyBackend`: Allows GPU-accelerated computing with CuPy, compatible 
  with CUDA-enabled devices.
- `NNBackend`: Enables machine learning and deep learning functionality 
  through neural network libraries.
- `EnsureMethod`: A decorator used to ensure that the `select_backend` 
  method is prioritized for execution.

Usage
-----
The `BackendSelector` class is designed to help users dynamically choose 
the backend that best fits their computational environment. It is useful 
in workflows that require seamless transitions between CPU and GPU 
resources or between different data processing libraries.

Examples
--------
>>> from gofast.selector import BackendSelector
>>> backend_selector = BackendSelector()
>>> backend = backend_selector.select_backend("numpy")

Notes
-----
The `select_backend` method, decorated with `EnsureMethod`, operates in 
soft mode with warnings, enabling flexible backend selection and handling 
potentially missing methods gracefully.

__all__
-------
- `BackendSelector`: Exposes the `BackendSelector` class for backend 
  management.

"""

from .numpy import NumpyBackend
from .scipy import ScipyBackend
from .dask import DaskBackend
from .cupy import CuPyBackend
from .nn import NNBackend 

from ..decorators import EnsureMethod 
from ..api.property import BaseClass 


__all__= ["BackendSelector"]

@EnsureMethod("select_backend", error='warn', mode='soft'  )
class BackendSelector(BaseClass):
    """
    A flexible class that manages and selects the most suitable computational 
    backend for `gofast` tasks. The `BackendSelector` allows users to specify 
    their preferred backend for both general computations (e.g., using 
    NumPy, SciPy) and neural network computations (e.g., using TensorFlow, 
    PyTorch), or to enable automatic selection based on system capabilities 
    and installed libraries.

    Parameters
    ----------
    preferred_backend : str, optional
        The preferred backend for general computations. Supported options 
        include `"numpy"`, `"scipy"`, `"dask"`, and `"cupy"`. If `None`, 
        automatic selection is applied.
    preferred_nn_backend : str, optional
        The preferred backend for neural network computations. Supported 
        options are `"tensorflow"` and `"pytorch"`. If `None`, automatic 
        selection is applied.
    verbose : int, optional
        Sets the verbosity level of the class output. `0` means no output, 
        while higher values increase the verbosity of feedback messages 
        regarding backend selection.

    Attributes
    ----------
    backends : dict
        Maps general backend names to their corresponding class implementations.
    nn_backends : dict
        Maps neural network backend names to `NNBackend`, which selects 
        between TensorFlow and PyTorch.
    selected_backend : object
        An instance of the selected backend class for general computations.
    selected_nn_backend : object
        An instance of the selected neural network backend class for deep 
        learning computations.

    Methods
    -------
    select_backend(preferred_backend)
        Selects the general computation backend based on user preference or 
        an automatic fallback if no valid preference is provided.
    select_nn_backend(preferred_nn_backend)
        Selects the neural network backend based on user preference or an 
        automatic fallback if no valid preference is provided.
    is_gpu_available()
        Checks if a CUDA-compatible GPU is available for GPU-accelerated 
        computations.
    is_dask_available()
        Checks if Dask is installed for distributed computations.
    is_scipy_available()
        Checks if SciPy is available for advanced scientific computations.
    auto_select_backend()
        Automatically selects the most suitable general backend based on 
        system resources.
    auto_select_nn_backend()
        Automatically selects the most suitable neural network backend 
        (TensorFlow or PyTorch).
    get_backend()
        Returns the currently selected general backend instance.
    get_nn_backend()
        Returns the currently selected neural network backend instance.

    Examples
    --------
    To select a general computation backend explicitly:

    >>> from gofast.backends.selector import BackendSelector
    >>> backend_selector = BackendSelector(preferred_backend='scipy')
    >>> backend = backend_selector.get_backend()
    >>> array = backend.array([1, 2, 3])
    
    If no backend is specified, an automatic selection is made:

    >>> backend_selector = BackendSelector()
    >>> backend = backend_selector.get_backend()
    # Selected backend based on available resources

    To select a neural network backend:

    >>> backend_selector = BackendSelector(preferred_nn_backend='tensorflow')
    >>> nn_backend = backend_selector.get_nn_backend()

    Notes
    -----
    The `BackendSelector` prioritizes GPU-accelerated backends for general 
    computation tasks by checking for `CuPy` support. If unavailable, it 
    defaults to `Dask`, `SciPy`, or `NumPy` in that order. Neural network 
    backend selection prioritizes TensorFlow if both it and PyTorch are 
    available, providing flexibility based on user configuration and system 
    capabilities.

    See Also
    --------
    numpy_backend, scipy_backend, dask_backend, cupy_backend, nn_backend

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).
           "Array programming with NumPy." Nature, 585(7825), 357–362.
    .. [2] Abadi, M., Agarwal, A., Barham, P., et al. (2016). "TensorFlow:
           Large-Scale Machine Learning on Heterogeneous Distributed Systems."
           arXiv preprint arXiv:1603.04467.
    .. [3] Paszke, A., Gross, S., Massa, F., et al. (2019). "PyTorch: An 
           Imperative Style, High-Performance Deep Learning Library."
           Advances in Neural Information Processing Systems, 32.
    """

    def __init__(
        self, 
        preferred_backend=None, 
        preferred_nn_backend=None, 
        verbose=0
        ):
        
        self.backends = {
            "numpy": NumpyBackend,
            "scipy": ScipyBackend,
            "dask": DaskBackend,
            "cupy": CuPyBackend
        }
        self.nn_backends = {
            "tensorflow": NNBackend(backend="tensorflow", init=False),
            "pytorch": NNBackend(backend="pytorch", init= False)
        }
        self.verbose = verbose
        self.alias_map = {
            "np": "numpy",
            "sp": "scipy",
            "da": "dask",
            "dk": "dask",
            "cp": "cupy",
            "cy": "cupy",
            "nn": "nn",
            "neural": "nn"
        }
        self.selected_backend = self.select_backend(preferred_backend)
        if preferred_nn_backend is not None:
            self.selected_nn_backend = self.select_nn_backend(preferred_nn_backend)

    def select_backend(self, preferred_backend):
        """
        Selects the general computation backend based on user preference 
        or defaults to automatic selection if the specified preference 
        is unavailable.

        Parameters
        ----------
        preferred_backend : str
            User-specified backend for general computations. Can be one of 
            `"numpy"`, `"scipy"`, `"dask"`, or `"cupy"`. Shortened forms, 
            such as `"np"` for `"numpy"` and `"cp"` for `"cupy"`, are 
            also supported.

        Returns
        -------
        backend_instance : object
            Instance of the selected backend class.

        Notes
        -----
        If `preferred_backend` is not available or is not provided, the 
        `auto_select_backend` method is called to automatically determine 
        the most suitable backend.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_backend="dask")
        >>> backend = backend_selector.select_backend("dask")
        DaskBackend selected by user preference.
        """

        normalized_backend = self.alias_map.get(preferred_backend, preferred_backend)
        if normalized_backend in self.backends:
            backend_instance = self.backends[normalized_backend]()
            if self.verbose > 0:
                print(f"{normalized_backend.capitalize()}Backend selected by user preference.")
            return backend_instance
        return self.auto_select_backend()

    def select_nn_backend(self, preferred_nn_backend):
        """
        Selects the neural network backend based on user preference or 
        defaults to automatic selection if the specified preference 
        is unavailable.

        Parameters
        ----------
        preferred_nn_backend : str
            User-specified neural network backend for computations, 
            either `"tensorflow"` or `"pytorch"`.

        Returns
        -------
        nn_backend_instance : object
            Instance of the selected neural network backend class.

        Notes
        -----
        If `preferred_nn_backend` is not available or is not provided, the 
        `auto_select_nn_backend` method is called to determine the optimal 
        backend, prioritizing `"tensorflow"`.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_nn_backend="pytorch")
        >>> nn_backend = backend_selector.select_nn_backend("pytorch")
        PyTorch selected for neural network computations.
        """

        if preferred_nn_backend in self.nn_backends:
            if self.verbose > 0:
                print(f"{preferred_nn_backend.capitalize()}"
                      " selected for neural network computations.")
            return self.nn_backends[preferred_nn_backend]._initialize_backend(
                preferred_nn_backend)
        return self.auto_select_nn_backend()

    def is_gpu_available(self):
        """
        Checks whether a CUDA-compatible GPU is accessible on the system.

        Returns
        -------
        bool
            `True` if a CUDA-compatible GPU is available, `False` otherwise.

        Notes
        -----
        This method attempts a simple GPU operation using the `cupy` library. 
        If `cupy` is installed and a GPU is accessible, the function returns 
        `True`.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend_selector.is_gpu_available()
        True
        """

        try:
            import cupy
            cupy.array([1])
            return True
        except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
            return False

    def is_dask_available(self):
        """
        Verifies the availability of the Dask library for distributed computing.

        Returns
        -------
        bool
            `True` if Dask is installed, `False` otherwise.

        Notes
        -----
        Dask is used for large-scale computations on distributed systems. 
        This method confirms its installation without initializing any 
        Dask components.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend_selector.is_dask_available()
        True
        """

        try:
            import dask #noqa
            return True
        except ImportError:
            return False

    def is_scipy_available(self):
        """
        Checks whether the SciPy library is available for scientific 
        computations.

        Returns
        -------
        bool
            `True` if SciPy is installed, `False` otherwise.

        Notes
        -----
        SciPy provides advanced scientific and numerical computing functions. 
        This method ensures SciPy is available for computations requiring 
        high precision and functionality beyond what is offered by NumPy.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend_selector.is_scipy_available()
        True
        """

        try:
            import scipy # # noqa
            return True
        except ImportError:
            return False

    def auto_select_backend(self):
        """
        Automatically selects the most suitable general backend based on 
        system capabilities, prioritizing GPU-accelerated computations 
        where available.

        Returns
        -------
        backend_instance : object
            Instance of the automatically selected general backend class.

        Notes
        -----
        This method follows a priority order:
            - `"cupy"` if a GPU is available,
            - `"dask"` if Dask is installed,
            - `"scipy"` if SciPy is installed,
            - `"numpy"` as the default.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend = backend_selector.auto_select_backend()
        CuPyBackend selected for general computations.
        """

        if self.is_gpu_available():
            selected_backend = "cupy"
        elif self.is_dask_available():
            selected_backend = "dask"
        elif self.is_scipy_available():
            selected_backend = "scipy"
        else:
            selected_backend = "numpy"
        if self.verbose > 0:
            print(f"{selected_backend.capitalize()}Backend"
                  " selected for general computations.")
        return self.backends[selected_backend]()

    def auto_select_nn_backend(self):
        """
        Automatically selects the most suitable neural network backend, 
        prioritizing TensorFlow if both TensorFlow and PyTorch are available.

        Returns
        -------
        nn_backend_instance : object
            Instance of the automatically selected neural network backend class.

        Raises
        ------
        ImportError
            If neither TensorFlow nor PyTorch is available.

        Notes
        -----
        This method attempts to import TensorFlow and PyTorch in order. 
        If TensorFlow is found, it is selected as the default neural network 
        backend. If TensorFlow is unavailable, PyTorch is chosen if available.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> nn_backend = backend_selector.auto_select_nn_backend()
        TensorFlow selected for neural network computations.
        """

        try:
            import tensorflow as tf # noqa
            if self.verbose > 0:
                print("TensorFlow selected for neural network computations.")
            return self.nn_backends["tensorflow"]
        except ImportError:
            try:
                import torch # noqa
                if self.verbose > 0:
                    print("PyTorch selected for neural network computations.")
                return self.nn_backends["pytorch"]
            except ImportError:
                raise ImportError(
                    "Neither TensorFlow nor PyTorch is"
                    " available for neural network computations.")

    def get_backend(self):
        """
        Retrieves the selected general backend instance for performing 
        computational tasks.

        Returns
        -------
        backend_instance : object
            The instance of the selected general backend class.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_backend="scipy")
        >>> backend = backend_selector.get_backend()
        """

        return self.selected_backend

    def get_nn_backend(self):
        """
        Retrieves the selected neural network backend instance for 
        performing deep learning tasks.

        Returns
        -------
        nn_backend_instance : object
            The instance of the selected neural network backend class.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_nn_backend="tensorflow")
        >>> nn_backend = backend_selector.get_nn_backend()
        """

        return self.selected_nn_backend



