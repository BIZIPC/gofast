# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utilities for managing compatibility between TensorFlow's Keras and 
standalone Keras. It includes functions and classes for dynamically importing 
Keras dependencies and checking the availability of TensorFlow or Keras.
"""

import warnings 
import importlib

__all__= ['KerasDependencies', 'check_keras_backend', 
          'import_keras_dependencies', 'import_keras_function',
 ]

class KerasDependencies:
    """
    Lazy-loads Keras dependencies from `tensorflow.keras` or `keras`.

    Parameters
    ----------
    extra_msg : str, optional
        Additional message to display in the warning if TensorFlow is 
        not found. This message will be appended to the default warning 
        message. Default is `None`.
    error : str, optional
        Defines the error handling strategy. If `'warn'` (default), a 
        warning message will be displayed if TensorFlow is not found. 
        If `'raise'`, an `ImportError` will be raised. Default is 
        `'warn'`.

    Notes
    -----
    This class attempts to dynamically import necessary Keras 
    dependencies from either `tensorflow.keras` or `keras` on demand.

    Mathematically, the import of each function or class can be 
    represented as:

    .. math::

        f = 
        \begin{cases} 
        f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
        f_{\text{keras}} & \text{if only Keras is installed} \\
        \end{cases}

    where :math:`f` is the function or class to be imported, 
    :math:`f_{\text{tf.keras}}` is the function or class from 
    `tensorflow.keras`, and :math:`f_{\text{keras}}` is the function or 
    class from `keras`.

    Examples
    --------
    >>> from gofast.compat.tf import KerasDependencies
    >>> deps = KerasDependencies(extra_msg="Custom warning message")
    >>> Adam = deps.Adam  # Only imports Adam optimizer when accessed
    >>> Sequential = deps.Sequential  # Only imports Sequential model when accessed

    See Also
    --------
    importlib.import_module : Import a module programmatically.
    keras.models.Model : The `Model` class from `keras.models`.
    keras.models.Sequential : The `Sequential` class from `keras.models`.
    tensorflow.keras.models.Model : The `Model` class from `tensorflow.keras.models`.
    tensorflow.keras.models.Sequential : The `Sequential` class from 
                                         `tensorflow.keras.models`.

    References
    ----------
    .. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
    .. [2] Abadi, Martín et al. *TensorFlow: Large-Scale Machine Learning 
           on Heterogeneous Systems*. 2015.
    """

    def __init__(self, extra_msg=None, error='warn'):
        self.extra_msg = extra_msg
        self.error = error
        self._dependencies = {}

        self._check_tensorflow()

    def _check_tensorflow(self):
        try:
            import_optional_dependency('tensorflow', extra=self.extra_msg)
            import tensorflow as tf  # noqa
            if self.error == 'warn':
                warnings.warn("TensorFlow is installed.", UserWarning)
        except ImportError as e:
            warnings.warn(f"{self.extra_msg}: {e}")

    def __getattr__(self, name):
        if name not in self._dependencies:
            self._dependencies[name] = self._import_dependency(name)
        return self._dependencies[name]

    def _import_dependency(self, name):
        mapping = {
            'Model': ('models', 'Model'),
            'Sequential': ('models', 'Sequential'),
            'Dense': ('layers', 'Dense'),
            'Dropout': ('layers', 'Dropout'),
            'BatchNormalization': ('layers', 'BatchNormalization'),
            'LSTM': ('layers', 'LSTM'),
            'Input': ('layers', 'Input'),
            'Conv2D': ('layers', 'Conv2D'),
            'MaxPooling2D': ('layers', 'MaxPooling2D'),
            'Flatten': ('layers', 'Flatten'),
            'Attention': ('layers', 'Attention'),
            'Concatenate': ('layers', 'Concatenate'),
            'Adam': ('optimizers', 'Adam'),
            'SGD': ('optimizers', 'SGD'),
            'RMSprop': ('optimizers', 'RMSprop'),
            'mnist': ('datasets', 'mnist'), 
            'cifar10': ('datasets', 'cifar10'), 
            'Loss': ('losses', 'Loss'),
            'mean_squared_error': ('losses', 'mean_squared_error'),
            'categorical_crossentropy': ('losses', 'categorical_crossentropy'),  
            'binary_crossentropy': ('losses', 'binary_crossentropy'),  
            'reduce_mean': ('backend', 'reduce_mean'), 
            'reduce_sum': ('backend', 'reduce_sum'),  
            'sum': ('backend', 'sum'),  
            'ones': ('backend', 'ones'), 
            'zeros': ('backend', 'zeros'),  
            'constant': ('backend', 'constant'),  
            'random_normal': ('initializers', 'random_normal'),  
            'glorot_uniform': ('initializers', 'glorot_uniform'),  
            'EarlyStopping': ('callbacks', 'EarlyStopping'),  
            'ModelCheckpoint': ('callbacks', 'ModelCheckpoint'),  
            'TensorBoard': ('callbacks', 'TensorBoard'), 
            'LearningRateScheduler': ('callbacks', 'LearningRateScheduler'), 
            'GradientTape': ('gradients', 'GradientTape'),
            'square': ('math', 'square'), 
            'clone_model': ('models', 'clone_model'), 
            'Dataset': ('data', 'Dataset'),
            'load_model': ('models', 'load_model')
            
        }

        if name in mapping:
            module_name, function_name = mapping[name]
            return import_keras_function(module_name, function_name, error=self.error)
        raise AttributeError(f"'KerasDependencies' object has no attribute '{name}'")

def import_optional_dependency(package_name, error='warn', extra=None):
    """
    Checks if a package is installed and optionally imports it.

    Parameters
    ----------
    package_name : str
        The name of the package to check and optionally import.
    error : str, optional
        Defines the error handling strategy. If `'warn'` (default), a warning 
        message will be displayed if the package is not found. If `'raise'`,
        an ImportError will be raised.
    extra : str, optional
        Additional message to display in the warning or error if the package 
        is not found.

    Returns
    -------
    module or None
        The imported module if available, otherwise None if `error='warn'` is used.

    Raises
    ------
    ImportError
        If the package is not installed and `error='raise'` is used.

    Notes
    -----
    This function attempts to check for the presence of a specified package and 
    optionally import it. 
    If the package is not found, the behavior depends on the `error` parameter.

    Examples
    --------
    >>> import_optional_dependency('numpy')
    Using numpy
    >>> import_optional_dependency('nonexistent_package',
                                   extra="Custom warning message", error='warn')
    Custom warning message: No module named 'nonexistent_package'
    >>> import_optional_dependency('nonexistent_package', error='raise')
    Traceback (most recent call last):
      ...
    ImportError: nonexistent_package is not installed. Use pip or conda to install it.

    See Also
    --------
    importlib.import_module : Import a module programmatically.

    References
    ----------
    .. [1] Python documentation on importlib: https://docs.python.org/3/library/importlib.html
    """
    try:
        module = importlib.import_module(package_name)
        return module
    except ImportError as e:
        message = f"{package_name} is not installed"
        if extra:
            message = f"{extra}: {e}"
        else:
            message = f"{message}: {e}. Use pip or conda to install it."
        
        if error == 'warn':
            warnings.warn(message)
            return None
        elif error == 'raise':
            raise ImportError(message)
        else:
            raise ValueError("Parameter 'error' must be either 'warn' or 'raise'.")

def import_keras_function(module_name, function_name, error='warn'):
    """
    Tries to import a function from `tensorflow.keras`, falling back to 
    `keras` if necessary.

    Parameters
    ----------
    module_name : str
        The name of the module from which to import the function. 
        This could be a submodule within `tensorflow.keras` or `keras`. 
        For example, `'optimizers'` or `'models'`.
    function_name : str
        The name of the function to import from the specified module. 
        For example, `'Adam'` or `'load_model'`.
    error : str, optional
        Defines the error handling strategy. If `'warn'` (default), 
        a warning message will be displayed when the function is 
        imported. If `'ignore'`, no warnings will be shown.

    Returns
    -------
    function
        The imported function from either `tensorflow.keras` or `keras`, 
        depending on what is available in the environment.

    Raises
    ------
    ImportError
        If neither `tensorflow.keras` nor `keras` is installed, or if 
        the specified function cannot be found in the respective modules.

    Notes
    -----
    This function attempts to dynamically import a specified function 
    from either `tensorflow.keras` or `keras`. It first tries to import 
    from `tensorflow.keras`, and if that fails (due to `tensorflow` not 
    being installed), it falls back to `keras`.

    Mathematically, the import can be considered as a selection 
    operation:

    .. math::

        f = 
        \begin{cases} 
        f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
        f_{\text{keras}} & \text{if only Keras is installed} \\
        \end{cases}

    where :math:`f` is the function to be imported, :math:`f_{\text{tf.keras}}` 
    is the function from `tensorflow.keras`, and :math:`f_{\text{keras}}` 
    is the function from `keras`.

    Examples
    --------
    >>> from gofast.compat.nn import import_keras_function
    >>> Adam = import_keras_function('optimizers', 'Adam')
    >>> load_model = import_keras_function('models', 'load_model')

    See Also
    --------
    keras.optimizers.Adam
    keras.models.load_model
    tensorflow.keras.optimizers.Adam
    tensorflow.keras.models.load_model

    References
    ----------
    .. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
    .. [2] Abadi, Martn et al. *TensorFlow: Large-Scale Machine Learning 
           on Heterogeneous Systems*. 2015.
    """
    try:
        tf_keras_module = __import__('tensorflow.keras.' + module_name, 
                                     fromlist=[function_name])
        function = getattr(tf_keras_module, function_name)
        if error == 'warn': 
            warnings.warn(f"Using TensorFlow Keras for {function_name} from {module_name}")
    except ImportError:
        try:
            keras_module = __import__('keras.' + module_name, 
                                      fromlist=[function_name])
            function = getattr(keras_module, function_name)
            if error == 'warn': 
                warnings.warn(f"Using Keras for {function_name} from {module_name}")
        except ImportError:
            raise ImportError(f"Neither TensorFlow nor Keras is installed. "
                              f"Cannot import {function_name} from {module_name}.")

    return function


def import_keras_dependencies(extra_msg=None, error='warn'):
    """
    Create a `KerasDependencies` instance for lazy-loading Keras dependencies.

    Parameters
    ----------
    extra_msg : str, optional
        Additional message to display in the warning if TensorFlow is 
        not found. This message will be appended to the default warning 
        message. Default is `None`.
    error : str, optional
        Defines the error handling strategy. If `'warn'` (default), a 
        warning message will be displayed if TensorFlow is not found. 
        If `'raise'`, an `ImportError` will be raised. Default is 
        `'warn'`.

    Returns
    -------
    KerasDependencies
        An instance of the `KerasDependencies` class for lazy-loading 
        Keras dependencies.

    Notes
    -----
    This function returns an instance of the `KerasDependencies` class, 
    which attempts to dynamically import necessary Keras dependencies 
    from either `tensorflow.keras` or `keras` on demand.

    Mathematically, the import of each function or class can be 
    represented as:

    .. math::

        f = 
        \begin{cases} 
        f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
        f_{\text{keras}} & \text{if only Keras is installed} \\
        \end{cases}

    where :math:`f` is the function or class to be imported, 
    :math:`f_{\text{tf.keras}}` is the function or class from 
    `tensorflow.keras`, and :math:`f_{\text{keras}}` is the function or 
    class from `keras`.

    Examples
    --------
    >>> from gofast.compat.tf import import_keras_dependencies
    >>> deps = import_keras_dependencies(extra_msg="Custom warning message")
    >>> Adam = deps.Adam  # Only imports Adam optimizer when accessed
    >>> Sequential = deps.Sequential  # Only imports Sequential model when accessed

    See Also
    --------
    importlib.import_module : Import a module programmatically.
    keras.models.Model : The `Model` class from `keras.models`.
    keras.models.Sequential : The `Sequential` class from `keras.models`.
    tensorflow.keras.models.Model : The `Model` class from 
                                    `tensorflow.keras.models`.
    tensorflow.keras.models.Sequential : The `Sequential` class from 
                                         `tensorflow.keras.models`.

    References
    ----------
    .. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
    .. [2] Abadi, Martín et al. *TensorFlow: Large-Scale Machine Learning 
           on Heterogeneous Systems*. 2015.
    """
    return KerasDependencies(extra_msg, error)

def check_keras_backend(error='warn'):
    """
    Check if `tensorflow` or `keras` is installed.

    Parameters
    ----------
    error : str, optional
        Defines the error handling strategy. If `'warn'` (default), a 
        warning message will be displayed if neither `tensorflow` nor 
        `keras` is installed. If `'raise'`, an `ImportError` will be 
        raised. If `'ignore'`, no action will be taken. Default is `'warn'`.

    Returns
    -------
    str or None
        Returns 'tensorflow' if TensorFlow is installed, 'keras' if Keras 
        is installed, or None if neither is installed.

    Examples
    --------
    >>> backend = check_keras_backend()
    >>> if backend:
    >>>     print(f"{backend} is installed.")
    >>> else:
    >>>     print("Neither tensorflow nor keras is installed.")

    Notes
    -----
    This function first checks for TensorFlow installation and then for 
    Keras. The behavior on missing packages depends on the `error` 
    parameter.
    """
    try:
        importlib.import_module('tensorflow')
        return 'tensorflow'
    except ImportError:
        try:
            importlib.import_module('keras')
            return 'keras'
        except ImportError as e:
            message = "Neither tensorflow nor keras is installed."
            if error == 'warn':
                warnings.warn(message)
            elif error == 'raise':
                raise ImportError(message) from e
            return None
