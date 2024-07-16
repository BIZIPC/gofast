# -*- coding: utf-8 -*-

from .numpy import NumpyBackend
from .scipy import ScipyBackend
from .dask import DaskBackend
from .cupy import CuPyBackend

class BackendSelector:
    """
    Manages and selects the most suitable computational backend for gofast tasks,
    considering user preferences, available hardware, and installed libraries.
    Supports both automatic detection of the optimal backend based on the system's
    capabilities and explicit selection by the user.

    Parameters
    ----------
    preferred_backend : str, optional
        The name of the preferred backend. Valid options include "numpy", "scipy",
        "dask", and "cupy". If not provided, the selector will automatically
        choose the most suitable backend based on the current environment and
        available resources.
    verbose : int, optional
        Verbosity level of the output, where 0 is silent and higher numbers increase
        the verbosity.

    Attributes
    ----------
    backends : dict
        A mapping from backend names to their corresponding class implementations.
    selected_backend : object
        The backend class instance that has been selected based on the preference
        or automatic detection.

    Methods
    -------
    select_backend(preferred_backend)
        Determines and initializes the backend based on the user's preference or
        an automatic selection process.
    is_gpu_available()
        Checks if a CUDA-compatible GPU is available for GPU-accelerated computations.
    is_dask_available()
        Verifies if Dask is installed for distributed computing capabilities.
    is_scipy_available()
        Confirms the availability of SciPy for advanced scientific computations.
    auto_select_backend()
        Automatically identifies and selects the most suitable backend.
    get_backend()
        Returns the currently selected backend instance.

    Examples
    --------
    Explicitly selecting a backend:

    >>> from gofast.backends.selector import BackendSelector
    >>> backend_selector = BackendSelector(preferred_backend='cupy')
    >>> backend = backend_selector.get_backend()
    CuPyBackend selected by user preference.

    Automatically selecting the best available backend:

    >>> backend_selector = BackendSelector()
    >>> backend = backend_selector.get_backend()
    # This will print the selected backend based on available resources

    Using the selected backend for computations:

    >>> array = backend.array([1, 2, 3])
    # Perform operations using the selected backend

    Notes
    -----
    The automatic backend selection process prioritizes GPU acceleration with CuPy
    when available, falling back to distributed computing with Dask, advanced
    computations with SciPy, or general-purpose computations with NumPy as the
    default option.
    """
    def __init__(self, preferred_backend=None, verbose=0):
        self.backends = {
            "numpy": NumpyBackend,
            "scipy": ScipyBackend,
            "dask": DaskBackend,
            "cupy": CuPyBackend
        }
        self.verbose = verbose
        self.alias_map = {
            "np": "numpy",
            "sp": "scipy",
            "da": "dask",
            "dk": "dask",
            "cp": "cupy",
            "cy": "cupy"
        }
        self.selected_backend = self.select_backend(preferred_backend)

    def select_backend(self, preferred_backend):
        """
        Selects the backend based on user preference or environment. Supports
        shorthand aliases for backend names to enhance user flexibility.

        Parameters
        ----------
        preferred_backend : str, optional
            A string indicating the preferred backend, which can include shorthand
            aliases like 'np' for NumPy or 'cp' for CuPy.

        Returns
        -------
        object
            An instance of the selected backend.
        """
        # Normalize the backend name using the alias map
        normalized_backend = self.alias_map.get(preferred_backend, preferred_backend)
        if normalized_backend in self.backends:
            backend_instance = self.backends[normalized_backend]()
            if self.verbose > 0:
                print(f"{normalized_backend.capitalize()}Backend "
                      "selected by user preference.")
            return backend_instance
        
        # Fallback to automatic selection if no preference or invalid preference is provided
        return self.auto_select_backend()

    def is_gpu_available(self):
        """
        Checks if a CUDA-compatible GPU is available by attempting to perform a 
        simple operation using CuPy.
    
        Returns
        -------
        bool
            True if a CUDA-compatible GPU is available, False otherwise.
        """
        try:
            import cupy 
            cupy.array([1])  # Simple operation to ensure CuPy can use the GPU
            return True
        except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
            return False

    def is_dask_available(self):
        """
        Checks if the Dask library is installed, which is necessary for 
        distributed computing.
    
        Returns
        -------
        bool
            True if Dask is installed, False otherwise.
        """
        try:
            import dask # noqa
            return True
        except ImportError:
            return False
    
    def is_scipy_available(self):
        """
        Verifies the availability of SciPy for advanced scientific computations.
    
        Returns
        -------
        bool
            True if SciPy is installed, False otherwise.
        """
        try:
            import scipy # noqa
            return True
        except ImportError:
            return False
    
    def auto_select_backend(self):
        """
        Automatically identifies and selects the most suitable backend based 
        on available system resources. The selection process prioritizes GPU 
        acceleration, followed by distributed computing capabilities, and 
        finally advanced computations.
    
        Returns
        -------
        object
            An instance of the selected backend.
        """
        if self.is_gpu_available():
            selected_backend = "cupy"
        elif self.is_dask_available():
            selected_backend = "dask"
        elif self.is_scipy_available():
            selected_backend = "scipy"
        else:
            selected_backend = "numpy"
    
        if self.verbose:
            print(f"{selected_backend.capitalize()}Backend selected based"
                  " on system capabilities.")
        return self.backends[selected_backend]()
    
    def get_backend(self):
        """
        Returns the currently selected backend instance, allowing it to be 
        used for computational tasks.
    
        Returns
        -------
        object
            The backend class instance that has been selected based on the preference
            or automatic detection.
        """
        return self.selected_backend
