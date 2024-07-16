# -*- coding: utf-8 -*-

from .backends.numpy import NumpyBackend
from .backends.scipy import ScipyBackend 
from .backends.dask import DaskBackend 
from .backends.cupy import CuPyBackend 

# Global variable to hold the name of the current backend
_current_backend = 'numpy'  # Default to NumPy

# Dictionary mapping backend names to their corresponding classes
_backend_classes = {
    'numpy': NumpyBackend,
    'scipy': ScipyBackend,
    'dask': DaskBackend,
    'cupy': CuPyBackend,
}

def set_backend(backend_name):
    """
    Sets the active computational backend for gofast.
    
    Parameters:
    - backend_name: str, name of the backend ('numpy', 'scipy', 'dask', 'cupy')
    
    Raises:
    - ValueError if the backend name is not supported.
    """
    global _current_backend
    if backend_name in _backend_classes:
        _current_backend = backend_name
        print(f"Active backend set to: {backend_name.capitalize()}")
    else:
        raise ValueError(
            f"Unsupported backend: {backend_name}. Supported"
            f" backends: {list(_backend_classes.keys())}"
        )

def get_backend():
    """
    Returns the active computational backend class for gofast.
    
    Returns:
    - An instance of the active backend class.
    """
    backend_class = _backend_classes[_current_backend]
    return backend_class()

# Example usage
if __name__ == '__main__':
    from gofast.backends.numpy import NumpyBackend
    from gofast.backends.scipy  import ScipyBackend
    from gofast.backends.dask import DaskBackend
    from gofast.backends.cupy import CuPyBackend

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
