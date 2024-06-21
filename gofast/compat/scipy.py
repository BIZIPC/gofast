# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
from packaging.version import Version, parse
import warnings
# import logging
import numpy as np

import scipy
from scipy import special
from scipy import stats
from .._gofastlog import gofastlog 
# Setup logging
_logger = gofastlog().get_gofast_logger(__name__)

__all__ = [
    "integrate_quad",
    "optimize_minimize",
    "special_jn",
    "linalg_inv",
    "linalg_solve",
    "linalg_det",
    "sparse_csr_matrix",
    "ensure_scipy_compatibility", 
    "calculate_statistics", 
    "is_sparse_matrix", 
    "solve_linear_system",
    "check_scipy_interpolate",
    "get_scipy_function", 
    "SP_LT_1_6",
    "SP_LT_1_5",
    "SP_LT_1_7", 
    "SP_LT_0_15", 
]
# Version checks
scipy_version = parse(scipy.__version__)
SP_LT_1_7 = scipy_version < Version("1.6.99")
SP_LT_1_6 = scipy_version < Version("1.5.99")
SP_LT_1_5 = scipy_version < Version("1.4.99")
SP_LT_0_15 = scipy_version < Version("0.14.0")

# Utilize function-based approach for direct imports and version-based configurations
def get_scipy_function(module_path, function_name, version_requirement=None):
    """
    Dynamically imports a function from scipy based on the version requirement.
    
    Parameters
    ----------
    module_path : str
        The module path within scipy, e.g., 'integrate'.
    function_name : str
        The name of the function to import.
    version_requirement : Version, optional
        The minimum version of scipy required for this function.
        
    Returns
    -------
    function or None
        The scipy function if available and meets version requirements, else None.
    """
    if version_requirement is None: 
        version_requirement = Version("0.14.0")
    if version_requirement and scipy_version < version_requirement:
        _logger.warning(f"{function_name} requires scipy version {version_requirement} or higher.")
        return None
    
    try:
        module = __import__(f"scipy.{module_path}", fromlist=[function_name])
        return getattr(module, function_name)
    except ImportError as e:
        _logger.error(f"Failed to import {function_name} from scipy.{module_path}: {e}")
        return None

# Define functionalities using the utility function with appropriate
# version checks where necessary
integrate_quad = get_scipy_function("integrate", "quad")
# Optimization
optimize_minimize = get_scipy_function("optimize", "minimize")
# Linear algebra
special_jn = get_scipy_function("special", "jn")
linalg_inv = get_scipy_function("linalg", "inv")
linalg_solve = get_scipy_function("linalg", "solve")
linalg_det = get_scipy_function("linalg", "det")
linalg_eigh = get_scipy_function("linalg", "eigh")
# Sparse matrices
sparse_csr_matrix = get_scipy_function("sparse", "csr_matrix")
sparse_csc_matrix = get_scipy_function("sparse", "csc_matrix")

# Stats
if SP_LT_1_7:
    # Use older stats functions or define a fallback
    def stats_norm_pdf(x):
        return "norm.pdf replacement for older scipy versions"
else:
    stats_norm_pdf = stats.norm.pdf

# Define a function that uses version-dependent functionality
def calculate_statistics(data):
    """
    Calculate statistics on data using scipy's stats module,
    with handling for different versions of scipy.
    """
    if SP_LT_1_7:
        # Use an alternative approach or older function if needed
        mean = np.mean(data)
        median = np.median(data)
        pdf = "PDF calculation not available in this scipy version"
    else:
        mean = stats.tmean(data)
        median = stats.median(data)
        pdf = stats_norm_pdf(data)
    
    return mean, median, pdf

def is_sparse_matrix(matrix) -> bool:
    """
    Check if a matrix is a scipy sparse matrix.

    Parameters
    ----------
    matrix : Any
        Matrix to check.

    Returns
    -------
    bool
        True if the matrix is a scipy sparse matrix.
    """
    return scipy.sparse.issparse(matrix)


def solve_linear_system(A, b):
    """
    Solve a linear system Ax = b using scipy's linalg.solve function.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Coefficient matrix.
    b : ndarray
        Ordinate or dependent variable values.

    Returns
    -------
    x : ndarray
        Solution to the system Ax = b.
    """
    if is_sparse_matrix(A):
        A = A.toarray()  # Convert to dense array if A is sparse for compatibility
    return linalg_solve(A, b)


# Special functions
if hasattr(special, 'erf'):
    special_erf = special.erf
else:
    # Define a fallback for erf if it's not present in the current scipy version
    def special_erf(x):
        # Approximation or use an alternative approach
        return "erf function not available in this scipy version"

# Messages
_msg = ''.join([
    'Note: need scipy version 0.14.0 or higher for interpolation,',
    ' might not work.']
)
_msg0 = ''.join([
    'Could not find scipy.interpolate, cannot use method interpolate. ',
    'Check your installation. You can get scipy from scipy.org.']
)

def ensure_scipy_compatibility():
    """
    Ensures that the scipy version is compatible and required modules are available.
    Logs warnings if conditions are not met.
    """
    global interp_import
    try:
        scipy_version = [int(ss) for ss in scipy.__version__.split('.')]
        if scipy_version[0] == 0 and scipy_version[1] < 14:
            warnings.warn(_msg, ImportWarning)
            _logger.warning(_msg)

        # Attempt to import required modules
        import scipy.interpolate as spi # noqa 
        from scipy.spatial import distance # noqa 

        interp_import = True
       # _logger.info("scipy.interpolate and scipy.spatial.distance imported successfully.")

    except ImportError as e: # noqa
        warnings.warn(_msg0, ImportWarning)
       # _logger.warning(_msg0)
        interp_import = False
        #_logger.error(f"ImportError: {e}")

    return interp_import


def check_scipy_interpolate():
    """
    Checks for scipy.interpolate compatibility and imports required modules.
    
    Returns
    -------
    module or None
        The scipy.interpolate module if available, otherwise None.
    """
    try:
        import scipy

        # Checking for the minimum version requirement
        if scipy_version < Version("0.14.0"):
            _logger.warning('Scipy version 0.14.0 or higher is required for'
                            ' interpolation. Might not work.')
            return None
        
        from scipy import interpolate # noqa 
        return scipy.interpolate

    except ImportError:
        _logger.error('Could not find scipy.interpolate, cannot use method'
                      ' interpolate. Check your scipy installation.') 
        return None


