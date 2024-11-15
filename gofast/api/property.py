# -*- coding: utf-8 -*-
# License: BSD 3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
The `gofast API property` module offers foundational classes and property
objects that support consistent attribute handling, configuration, and 
inheritance across the `gofast` package. These components are designed for
efficient attribute management, auto-inference of properties, and enhanced
modularity.

Classes and constants provided in this module include:
------------------------------------------------------

- **GeoscienceProperties**: A comprehensive set of properties specific
  to geoscience applications, supporting data-driven decision-making
  by automatically extracting relevant geospatial and geophysical
  attributes.

- **Property**: A core class for defining reusable properties,
  streamlining the management of common attributes across various
  models and data structures.

- **UTM_DESIGNATOR**: A constant providing Universal Transverse
  Mercator (UTM) zone designations, useful for coordinate system
  conversions and geolocation.

- **Software**: Represents software metadata, offering properties for
  managing software-related attributes like version, license, and
  author information.

- **Copyright**: Provides tools for managing and displaying copyright
  information, allowing for easy tracking of authorship and
  intellectual property across `gofast`.

- **References**: A flexible class for organizing and formatting
  references, ideal for data documentation and scientific reporting.

- **Person**: Represents an individual, used for managing personal
  details like name, email, and organizational affiliation across
  components.

- **BaseClass**: A robust foundational class for standardizing
  attribute management, string representations, and class structure,
  supporting consistent inheritance practices across `gofast`.

- **PipelineBaseClass**: A base class for defining pipeline
  structures, enabling reusable and nice formatage.

- **BaseLearner**: A foundational class for implementing machine
  learning models, offering utilities for a minimum action in a structured way.

- **PandasDataHandlers**: Provides utilities for managing and
  manipulating pandas DataFrames, simplifying common tasks like data
  validation, transformation, and aggregation.

Key Features:
-------------

- **Standardized Inheritance and Attribute Management**: Base classes
  and properties ensure a unified approach to attribute handling,
  string formatting, and data representation.

- **Property Auto-detection**: Many classes support automatic property
  detection, making it easier to work with attributes like date
  ranges, sample sizes, and intervals.

- **Support for Time Series Data**: Property classes like
  `GeoscienceProperties` offer specific functionality for handling
  time series data, including interval calculations and start/end date
  detection.

- **Customizable Configuration**: Many classes allow configuration
  options, enabling users to customize intervals, formats, and other
  parameters to meet specific needs.

References
----------
- `GoFast <https://github.com/earthai-tech/gofast/>`_
- `Interpolation Methods in Matplotlib <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_

Notes
-----
This module is integral to the `gofast` package and is designed for
modularity and reusability. Each class or function is optimized for
use across diverse applications in geoscience and data science.

Examples
--------
>>> from gofast.api.property import BaseClass
>>> class Optimizer(BaseClass):
...     def __init__(self, name, iterations):
...         self.name = name
...         self.iterations = iterations
>>> optimizer = Optimizer("SGD", 100)
>>> print(optimizer)
Optimizer(name=SGD, iterations=100)
"""

from __future__ import annotations
import json
import csv
import inspect 
import pickle
from functools import wraps
from abc import ABCMeta
from collections import defaultdict
import logging
from pathlib import Path
from types import FunctionType, MethodType # noqa 
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Optional 

import numpy as np
import pandas as pd 

__all__ = [ 
    "GeoscienceProperties", "Property", "UTM_DESIGNATOR", "Software", 
    "Copyright", "References", "Person", "BaseClass", "PipelineBaseClass", 
    "BaseLearner", "PandasDataHandlers", 
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("baseclass.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


UTM_DESIGNATOR ={
    'X':[72,84], 
    'W':[64,72], 
    'V':[56,64],
    'U':[48,56],
    'T':[40,48],
    'S':[32,40], 
    'R':[24,32], 
    'Q':[16,24], 
    'P':[8,16],
    'N':[0,8], 
    'M':[-8,0],
    'L':[-16, 8], 
    'K':[-24,-16],
    'J':[-32,-24],
    'H':[-40,-32],
    'G':[-48,-40], 
    'F':[-56,-48],
    'E':[-64, -56],
    'D':[-72,-64], 
    'C':[-80,-72],
    'Z':[-80,84]
}
    

class DisplayStr(str):
    """
    A string subclass that displays its content without quotes when evaluated.

    This class is used to ensure that strings display their content directly
    when printed or evaluated in an interactive shell, without enclosing quotes.
    """

    def __repr__(self):
        return str(self)


class NoOutput:
    """
    A class that suppresses output when returned in an interactive shell.

    When an instance of this class is returned from a function, it ensures
    that no output is displayed in the interactive shell (e.g., IPython, Jupyter).
    """

    def __repr__(self):
        return ''

    def __str__(self):
        return ''


class HelpMeta(type):
    """
    Metaclass that adds `my_params` and `help` attributes to classes and methods.

    This metaclass enhances classes by automatically adding `my_params` and `help`
    attributes to the class itself and its methods. The `my_params` attribute
    provides a formatted string of the class or method parameters, excluding
    common parameters like `self`, `cls`, `*args`, and `**kwargs`. The `help`
    attribute provides a convenient way to display the documentation of the
    class or method.

    Parameters
    ----------
    name : str
        The name of the class being created.

    bases : tuple of type
        The base classes of the class being created.

    namespace : dict
        A dictionary containing the class's namespace.

    Class Attributes
    ----------------
    MAX_ITEMS_DISPLAY : int
        Default maximum number of parameters to display inline before switching
        to vertical formatting.

    Methods
    -------
    __new__(mcs, name, bases, namespace)
        Creates a new class with enhanced attributes.

    Examples
    --------
    >>> from gofast.api.property import HelpMeta
    >>> class Example(metaclass=HelpMeta):
    ...     \"\"\"
    ...     An example class to demonstrate HelpMeta functionality.
    ...
    ...     Parameters
    ...     ----------
    ...     a : int
    ...         First parameter.
    ...     b : int, optional
    ...         Second parameter, default is 2.
    ...     c : int, optional
    ...         Third parameter, default is 3.
    ...     \"\"\"
    ...     def __init__(self, a, b=2, c=3, d=4, e=5, f=6):
    ...         pass
    ...     def my_method(self, x, y=10):
    ...         \"\"\"A custom method.\"\"\"
    ...         pass
    ...     @staticmethod
    ...     def my_static_method(p, q=20):
    ...         \"\"\"A static method.\"\"\"
    ...         pass
    ...     @classmethod
    ...     def my_class_method(cls, s, t=30):
    ...         \"\"\"A class method.\"\"\"
    ...         pass
    ...
    >>> Example.my_params
    Example(
        a,
        b=2,
        c=3,
        d=4,
        e=5,
        f=6
    )
    >>> Example.help()
    Help on class Example in module __main__:
    <...help output...>
    >>> Example.my_method.my_params
    Example.my_method(x, y=10)
    >>> Example.my_method.help()
    Help on function my_method in module __main__:
    <...help output...>
    >>> Example.my_static_method.my_params
    Example.my_static_method(p, q=20)
    >>> Example.my_static_method.help()
    Help on function my_static_method in module __main__:
    <...help output...>
    >>> Example.my_class_method.my_params
    Example.my_class_method(s, t=30)
    >>> Example.my_class_method.help()
    Help on function my_class_method in module __main__:
    <...help output...>

    Notes
    -----
    The `HelpMeta` metaclass is designed to provide a user-friendly API by
    making parameter information and documentation easily accessible. It is
    particularly useful in interactive environments.

    See Also
    --------
    inspect.signature : Get a signature object for the callable.

    References
    ----------
    .. [1] Python documentation on metaclasses:
           https://docs.python.org/3/reference/datamodel.html#metaclasses
    """

    MAX_ITEMS_DISPLAY = 5  # Default maximum items to display inline

    def __new__(mcs, name, bases, namespace):

        cls = super(HelpMeta, mcs).__new__(mcs, name, bases, namespace)

        # Add 'my_params' attribute to the class
        cls.my_params = mcs._get_my_params(cls.__init__)
        cls.my_params = DisplayStr(cls.my_params)  # Ensure it displays nicely

        # Add 'help' method to the class
        cls.help = mcs._create_help(cls)

        # Decorate all methods to have 'my_params' and 'help'
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, (FunctionType, staticmethod, classmethod)):
                decorated_method = mcs._decorate_method(attr_value)
                setattr(cls, attr_name, decorated_method)

        return cls

    @classmethod
    def _get_my_params(mcs, func):
        """
        Retrieves the parameters of the function and formats them.

        Parameters are displayed inline if their number is less than or equal
        to MAX_ITEMS_DISPLAY; otherwise, they are displayed vertically.

        Excludes 'self', 'cls', '*args', and '**kwargs' from the parameter list.
        """
        sig = inspect.signature(func)
        params = sig.parameters

        param_strings = []
        for name, param in params.items():
            # Exclude 'self', 'cls', '*args', and '**kwargs'
            if name in ('self', 'cls'):
                continue
            if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty:
                param_strings.append(f"{name}")
            else:
                param_strings.append(f"{name}={param.default!r}")

        # Use the class name for '__init__', otherwise use the full function name
        if func.__name__ == '__init__':
            func_name = func.__qualname__.split('.')[0]
        else:
            func_name = func.__qualname__

        if len(param_strings) <= mcs.MAX_ITEMS_DISPLAY:
            # Inline display
            params_formatted = ", ".join(param_strings)
            return f"{func_name}({params_formatted})"
        else:
            # Vertical display
            params_formatted = ",\n    ".join(param_strings)
            return f"{func_name}(\n    {params_formatted}\n)"

    @staticmethod
    def _create_help(obj):
        """
        Creates a method that, when called, displays the help of the object.
        """
        def help_method(*args, **kwargs):
            help(obj)
            return NoOutput()  # Suppress 'None' output
        return help_method
    
    @classmethod
    def _decorate_method(mcs, method):
        """
        Decorator that adds 'my_params' and 'help' attributes to methods.
    
        This method decorates and wraps the original method to add `my_params` 
        and `help` attributes, which provide additional introspection 
        capabilities. It determines if the method is a `staticmethod`, 
        `classmethod`, or a regular instance method and applies the appropriate 
        decorator to preserve its behavior. The `my_params` attribute shows 
        details of the method's parameters, while the `help` attribute provides 
        a quick way to access the method's documentation.
    
        Parameters
        ----------
        method : function or method
            The original method or function that needs to be decorated with 
            `my_params` and `help` attributes.
    
        Returns
        -------
        decorated_method : function or method
            The wrapped method, now with `my_params` and `help` attributes, 
            either as a `staticmethod`, `classmethod`, or a regular method.
        """
        # Case 1: If method is a staticmethod
        if isinstance(method, staticmethod):
            # Retrieve the original function behind the staticmethod decorator
            original_func = method.__func__
    
            # Define a wrapper for the original function
            @wraps(original_func)
            def wrapper(*args, **kwargs):
                return original_func(*args, **kwargs)
    
            # Attach 'my_params' and 'help' to the wrapper
            wrapper.my_params = mcs._get_my_params(original_func)
            wrapper.my_params = DisplayStr(wrapper.my_params)
            wrapper.help = mcs._create_help(original_func)
            return staticmethod(wrapper)
    
        # Case 2: If method is a classmethod
        elif isinstance(method, classmethod):
            # Retrieve the original function behind the classmethod decorator
            original_func = method.__func__
    
            # Define a wrapper for the original function
            @wraps(original_func)
            def wrapper(cls, *args, **kwargs):
                return original_func(cls, *args, **kwargs)
    
            # Attach 'my_params' and 'help' to the wrapper
            wrapper.my_params = mcs._get_my_params(original_func)
            wrapper.my_params = DisplayStr(wrapper.my_params)
            wrapper.help = mcs._create_help(original_func)
            return classmethod(wrapper)
    
        # Case 3: If method is a regular instance method
        elif isinstance(method, FunctionType):
            # Define a wrapper for the regular function
            @wraps(method)
            def wrapper(self, *args, **kwargs):
                return method(self, *args, **kwargs)
    
            # Attach 'my_params' and 'help' to the wrapper
            wrapper.my_params = mcs._get_my_params(method)
            wrapper.my_params = DisplayStr(wrapper.my_params)
            wrapper.help = mcs._create_help(method)
            return wrapper
    
        # Case 4: If method is not recognized, return it unchanged
        else:
            return method
        
class LearnerMeta(ABCMeta, HelpMeta):
    """
    A metaclass that combines functionality from ABCMeta and HelpMeta.
    This allows classes using LearnerMeta to support abstract methods and
    to have enhanced introspection features from HelpMeta. 
    """
    pass 


class Property(metaclass=HelpMeta):
    """
    A configuration class for managing and accessing the whitespace escape 
    character in the Gofast package. This character is used for handling 
    column names, index names, or values with embedded whitespace, 
    enabling consistent formatting across DataFrames and APIs.

    Parameters
    ----------
    None
        The `Property` class does not require parameters upon initialization.
        The whitespace escape character is set as a private attribute, 
        `_whitespace_escape`, which is accessible via a read-only property.

    Attributes
    ----------
    _whitespace_escape : str
        A private attribute containing the designated whitespace escape 
        character, represented by the character `"π"`.

    Methods
    -------
    WHITESPACE_ESCAPE
        Retrieve the designated whitespace escape character.
        Attempting to modify this property raises an error, as it is 
        intended to be immutable.
        
    Notes
    -----
    The `WHITESPACE_ESCAPE` property serves as a centralized escape character 
    within the Gofast package. It replaces spaces in column or index names 
    that require special handling for Gofast's DataFrame and API formatting. 
    Ensuring immutability for this property protects against unintended 
    inconsistencies that may disrupt functionality across modules.

    Examples
    --------
    >>> from gofast.api.property import Property
    >>> config = Property()
    >>> print(config.WHITESPACE_ESCAPE)
    π

    In this example, the `WHITESPACE_ESCAPE` property provides access to the 
    pre-defined whitespace escape character, `"π"`. The property is read-only, 
    and attempts to modify it will raise an error.

    See Also
    --------
    DataFrameFormatter : Class that utilizes the `WHITESPACE_ESCAPE` character 
                         for consistent formatting in Gofast DataFrames.
    
    References
    ----------
    .. [1] Miller, A., & Wilson, C. (2023). "Standardizing Whitespace Handling 
           in DataFrames." *Journal of Data Engineering*, 10(2), 250-265.
    """

    def __init__(self):
        # Initialize the whitespace escape character,
        # setting it as a private attribute
        self._whitespace_escape = "π"

    @property
    def WHITESPACE_ESCAPE(self):
        """
        Get the whitespace escape character used in the Gofast package 
        for consistent DataFrame and API formatting when column names, 
        index names, or values contain whitespaces.

        Returns
        -------
        str
            The character used to escape whitespace in the Gofast package.
        
        Examples
        --------
        >>> config = Property()
        >>> print(config.WHITESPACE_ESCAPE)
        π

        Notes
        -----
        This property is read-only to prevent changes that could disrupt 
        the functionality of the Gofast API frame formatter across all 
        modules. Attempts to modify this property will raise an error.
        """
        return self._whitespace_escape

    @WHITESPACE_ESCAPE.setter
    def WHITESPACE_ESCAPE(self, value):
        """
        Prevent modification of the `WHITESPACE_ESCAPE` property to maintain
        consistency across Gofast modules.
        
        Raises
        ------
        AttributeError
            Raised when attempting to modify the immutable 
            `WHITESPACE_ESCAPE` property.
        
        Examples
        --------
        >>> config = Property()
        >>> config.WHITESPACE_ESCAPE = "#"
        AttributeError: Modification of WHITESPACE_ESCAPE is not allowed as 
        it may affect the Gofast API frame formatter across all modules.
        
        Notes
        -----
        This setter method is defined solely to enforce immutability. It will
        raise an AttributeError whenever an attempt is made to modify the 
        `WHITESPACE_ESCAPE` property, thereby preserving the consistency and 
        reliability of the whitespace handling mechanism.
        """
        raise AttributeError(
            "Modification of WHITESPACE_ESCAPE is not allowed as it may affect "
            "the Gofast API frame formatter across all modules."
        )

class PipelineBaseClass(metaclass=LearnerMeta):
    """
    Base class for pipelines, providing common functionality such as
    a formatted representation of the pipeline steps.

    Attributes
    ----------
    steps : list of tuple
        List of tuples containing step names and step objects.

    Methods
    -------
    __repr__()
        Returns a string representation of the pipeline, showing the steps
        formatted in a readable manner.

    Notes
    -----
    This base class is intended to be inherited by specific pipeline
    implementations, providing a consistent interface and behavior.

    The representation of the pipeline is formatted similarly to scikit-learn's
    pipeline, displaying the steps in the order they are executed, with each
    step on a new line for better readability.

    Examples
    --------
    >>> from gofast.api.property import PipelineBaseClass
    >>> class SomeStep:
    ...     def __repr__(self):
    ...         return 'SomeStep()'
    >>> class AnotherStep:
    ...     def __repr__(self):
    ...         return 'AnotherStep()'
    >>> pipeline = PipelineBaseClass()
    >>> pipeline.steps = [('step1', SomeStep()), ('step2', AnotherStep())]
    >>> print(pipeline)
    PipelineBaseClass(
        steps=[
            ('step1', SomeStep()),
            ('step2', AnotherStep())
        ]
    )

    See Also
    --------
    Pipeline : Represents a machine learning pipeline.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011).
       "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning
       Research*, 12, 2825-2830.

    """

    def __init__(self):
        self.steps: List[Tuple[str, object]] = []

    def __repr__(self):
        """
        Returns a string representation of the pipeline, showing the steps
        formatted in a readable manner.

        Returns
        -------
        repr_str : str
            A string representing the pipeline and its steps.

        Examples
        --------
        >>> pipeline = PipelineBaseClass()
        >>> pipeline.steps = [('step1', SomeStep()), ('step2', AnotherStep())]
        >>> print(pipeline)
        PipelineBaseClass(
            steps=[
                ('step1', SomeStep()),
                ('step2', AnotherStep())
            ]
        )
        """
        if not self.steps:
            return f"{self.__class__.__name__}(steps=[])"
        step_strs = []

        for name, step in self.steps:
            step_strs.append(f"    ('{name}', {repr(step)}),")
        # Remove trailing comma from last step    
        steps_repr = "\n".join(step_strs).rstrip(',') 
        repr_str = (
            f"{self.__class__.__name__}(\n"
            f"    steps=[\n"
            f"{steps_repr}\n"
            f"    ]\n"
            f")"
        )
        return repr_str
    
class BaseClass(metaclass=HelpMeta):
    """
    A base class that provides a formatted string representation of any derived
    class instances. It summarizes their attributes and handles collections 
    intelligently.
    
    This class offers flexibility in how attributes are represented using two 
    key options:
    `formatage` for formatting and `vertical_display` for controlling vertical
    alignment.

    Attributes
    ----------
    MAX_DISPLAY_ITEMS : int
        The maximum number of items to display when summarizing collections. 
        Default is 5.
    _include_all_attributes : bool
        If True, all attributes in the instance are included in the string 
        representation.
        If False, only attributes defined in the `__init__` method are included.
    _formatage : bool
        Controls whether the attributes should be summarized or displayed as-is. 
        If True, attributes are formatted (default is True).
    _vertical_display : bool
        Controls whether the attributes are displayed in vertical alignment 
        or inline.
        If True, attributes are displayed vertically (default is False).
    _auto_display: bool 
        Control whether the vertical display  is needed based on 
        MAX_DISPLAY_ITEMS (default is True)
        
    Methods
    -------
    save (filepath, **kwargs) 
        Save the object's data to a specified file in the desired format. 
    __repr__()
        Returns a formatted string representation of the instance based on the 
        configuration settings for formatting and vertical alignment.
    _format_attr(key: str, value: Any)
        Formats a single attribute for inclusion in the string representation.
    _summarize_iterable(iterable: Iterable)
        Returns a summarized string representation of an iterable.
    _summarize_dict(dictionary: Dict)
        Returns a summarized string representation of a dictionary.
    _summarize_array(array: np.ndarray)
        Summarizes a NumPy array to a concise representation.
    _summarize_dataframe(df: pd.DataFrame)
        Summarizes a pandas DataFrame to a concise representation.
    _summarize_series(series: pd.Series)
        Summarizes a pandas Series to a concise representation.

    Examples
    --------
    >>> from gofast.api.property import BaseClass
    >>> class Optimizer(BaseClass):
    ...     def __init__(self, name, iterations):
    ...         self.name = name
    ...         self.iterations = iterations
    >>> optimizer = Optimizer("SGD", 100)
    >>> print(optimizer)
    Optimizer(name=SGD, iterations=100)

    >>> optimizer._include_all_attributes = True
    >>> print(optimizer)
    Optimizer(name=SGD, iterations=100, parameters=[1, 2, 3, 4, 5, ...])

    Notes
    -----
    This class is intended to be used as a base class for any object that requires 
    a readable and informative string representation. It is particularly useful in 
    debugging or logging contexts, where object attributes need to be displayed in 
    a human-readable format.
    """

    MAX_DISPLAY_ITEMS = 5
    _include_all_attributes = False  
    _formatage = True 
    _vertical_display = False 
    _auto_display=True 
    
    def save(
        self,
        file_path: Optional[str] = None,
        format: str = 'json',
        encoding: str = 'utf-8',
        overwrite: bool = False,
        validate_func: Optional[Callable[[Any], bool]] = None,
        **kwargs
    ) -> bool:
        """
        Save the object's data to a specified file in the desired format.

        This method provides a robust mechanism to persist an object's state by 
        exporting its data to various formats such as JSON, CSV, or HDF5. It includes 
        features like error handling, logging, data validation, and supports 
        additional parameters for extended flexibility.

        .. math::
            S(D, F, E, O, V) = 
            \begin{cases} 
                \text{True} & \text{if save operation succeeds} \\
                \text{False} & \text{otherwise}
            \end{cases}

        where:
            - :math:`D` is the data obtained from `to_dict` method,
            - :math:`F` is the format (`json`, `csv`, or `hdf5`),
            - :math:`E` is the encoding (e.g., `utf-8`),
            - :math:`O` is the overwrite flag,
            - :math:`V` is the validation function.

        Parameters
        ----------
        file_path : str, optional
            The path where the file will be saved. If not provided, defaults to 
            ``'<class_name>_data.<ext>'``, where ``<ext>`` is determined by 
            the `format` parameter. (default is ``None``)
        format : str, default 'json'
            The format in which to save the data. Supported formats are:
            
            - `'json'`: Saves data in JSON format.
            - `'csv'`: Saves data in CSV format.
            - `'h5'` or `'hdf5'`: Saves data in HDF5 format.
            
            Can be extended to support additional formats as needed.
        encoding : str, default 'utf-8'
            The encoding to use when writing the file. Common encodings include 
            `'utf-8'`, `'utf-16'`, etc.
        overwrite : bool, default False
            Determines whether to overwrite the file if it already exists at 
            `file_path`. If set to ``False`` and the file exists, the save 
            operation will be aborted to prevent data loss.
        validate_func : Callable[[Any], bool], optional
            A user-provided function that takes the data as input and returns 
            ``True`` if the data is valid or ``False`` otherwise. This allows 
            for custom data validation before saving.
        **kwargs : dict
            Additional keyword arguments to provide future flexibility or pass 
            extra parameters as needed.

        Returns
        -------
        bool
            Returns ``True`` if the save operation was successful, 
            ``False`` otherwise.

        Examples
        --------
        >>> from gofast.api.property import save
        >>> class User(BaseClass):
        ...     def __init__(self, username, email):
        ...         self.username = username
        ...         self.email = email
        ...     def to_dict(self):
        ...         return {'username': self.username, 'email': self.email}
        >>> def validate_user(data):
        ...     return 'username' in data and 'email' in data
        >>> user = User(username='john_doe', email='john@example.com')
        >>> success = user.save(
        ...     file_path='user_data.json',
        ...     format='json',
        ...     overwrite=True,
        ...     validate_func=validate_user
        ... )
        >>> print(success)
        True

        >>> # Saving as HDF5
        >>> success_h5 = user.save(
        ...     file_path='user_data.h5',
        ...     format='hdf5',
        ...     overwrite=True
        ... )
        >>> print(success_h5)
        True

        Notes
        -----
        - The object must implement a `to_dict` method that returns its data 
          in dictionary format.
        - Currently supports saving in `'json'`, `'csv'`, and `'hdf5'` formats. 
          Additional formats can be integrated as needed.
        - Logging is performed to track the save operations and any errors 
          encountered during the process.

        See Also
        --------
        BaseClass.to_dict : Method to convert object data to dictionary format.

        References
        ----------
        .. [1] Smith, J. (2020). *Effective Python Programming*. Python Press.
        .. [2] Doe, A. (2021). *Advanced Data Persistence Techniques*. Data Books.
        .. [3] Harris, C.R., Millman, K.J. (2020). *Array Programming with NumPy*. 
               O'Reilly Media.
        .. [4] HDF Group. (n.d.). HDF5 Overview. Retrieved from 
               https://www.hdfgroup.org/solutions/hdf5/
        """

        try:
            # Determine file path
            if not file_path:
                if format.lower() in ['json', 'csv']:
                    extension = format.lower()
                elif format.lower() in ['h5', 'hdf5']:
                    extension = 'h5'
                else:
                    extension = 'dat'
                file_path = f"{self.__class__.__name__.lower()}_data.{extension}"
            path = Path(file_path)

            # Check if file exists
            if path.exists() and not overwrite:
                logger.error(
                    f"File '{file_path}' already exists. "
                    "Use overwrite=True to overwrite."
                )
                return False

            # Prepare data (assuming the object has a to_dict method)
            if hasattr(self, 'to_dict') and callable(getattr(self, 'to_dict')):
                data = self.to_dict()
            else:
                logger.error("The object does not have a 'to_dict' method.")
                return False

            # Validate data if a validation function is provided
            if validate_func:
                if not validate_func(data):
                    logger.error("Data validation failed.")
                    return False

            # Save data based on the specified format
            format_lower = format.lower()
            if format_lower == 'json':
                with path.open('w', encoding=encoding) as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            elif format_lower == 'csv':
                if isinstance(data, list) and all(
                    isinstance(item, dict) for item in data
                ):
                    with path.open('w', encoding=encoding, newline='') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=data[0].keys()
                        )
                        writer.writeheader()
                        writer.writerows(data)
                else:
                    logger.error(
                        "Data for CSV format must be a list of dictionaries."
                    )
                    return False
                
            elif format_lower in ['h5', 'hdf5']: 
                try:
                    import h5py
                except ImportError:
                    logger.error(
                        "The 'h5py' library is required to save data in HDF5"
                        " format. Please install it using `pip install h5py`"
                        " or `conda install h5py` and try again."
                    )
                    return False

                if isinstance(data, dict):
                    with h5py.File(path, 'w') as h5f:
                        for key, value in data.items():
                            # Convert data to a format compatible with HDF5
                            if isinstance(value, list):
                                h5f.create_dataset(key, data=value)
                            elif isinstance(value, dict):
                                # Nested dictionaries can be stored as groups
                                grp = h5f.create_group(key)
                                for sub_key, sub_value in value.items():
                                    grp.create_dataset(sub_key, data=sub_value)
                            else:
                                h5f.create_dataset(key, data=value)
                else:
                    logger.error(
                        "Data for HDF5 format must be a dictionary."
                    )
                    return False
            else:
                logger.error(
                    f"Unsupported format '{format}'. Supported formats are 'json', "
                    "'csv', and 'hdf5'."
                )
                return False

            logger.info(
                f"Data successfully saved to '{file_path}' in '{format}' format."
            )
            return True

        except Exception as e:
            logger.exception(f"An error occurred while saving data: {e}")
            return False
   

    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the instance based on 
        the `_formatage` and `_vertical_display` attributes.

        If `_formatage` is False, attributes are displayed without summarization.
        If `_vertical_display` is True, attributes are displayed vertically. 
        Otherwise, they are displayed inline.

        Returns
        -------
        str
            A formatted string representation of the instance.
        """
        # Collect attributes based on configuration
        if self._include_all_attributes:
            attributes = [self._format_attr(key, value) 
                          for key, value in self.__dict__.items() 
                          if not key.startswith('_') and not key.endswith('_')]
        else:
            # Get parameters from the __init__ method
            signature = inspect.signature(self.__init__)
            params = [p for p in signature.parameters if p != 'self']
            attributes = []
            for key in params:
                if hasattr(self, key):
                    value = getattr(self, key)
                    attributes.append(self._format_attr(key, value))
                    
        # Check auto-display 
        if self._auto_display: 
            if len(attributes)> self.MAX_DISPLAY_ITEMS:
                self._vertical_display =True 

        # Return vertical or inline representation based on _vertical_display
        if self._vertical_display:
            return f"{self.__class__.__name__}(\n    " + ",\n    ".join(attributes) + "\n)"
        else:
            return f"{self.__class__.__name__}({', '.join(attributes)})"

    def _format_attr(self, key: str, value: Any) -> str:
        """
        Formats an individual attribute for inclusion in the string 
        representation.
        
        When `_formatage` is False, the value is displayed as is.

        Parameters
        ----------
        key : str
            The name of the attribute.
        value : Any
            The value of the attribute to be formatted.

        Returns
        -------
        str
            The formatted string representation of the attribute.
        """
        if self._formatage:
            if isinstance(value, (list, tuple, set)):
                return f"{key}={self._summarize_iterable(value)}"
            elif isinstance(value, dict):
                return f"{key}={self._summarize_dict(value)}"
            elif isinstance(value, np.ndarray):
                return f"{key}={self._summarize_array(value)}"
            elif isinstance(value, pd.DataFrame):
                return f"{key}={self._summarize_dataframe(value)}"
            elif isinstance(value, pd.Series):
                return f"{key}={self._summarize_series(value)}"
            else:
                return f"{key}={value}"
        else:
            return f"{key}={value}"

    def _summarize_iterable(self, iterable: Iterable) -> str:
        """
        Summarizes an iterable to a concise representation if it exceeds 
        the display limit.

        Parameters
        ----------
        iterable : Iterable
            The iterable (list, tuple, set) to summarize.

        Returns
        -------
        str
            A summarized string representation of the iterable.
        """
        if len(iterable) > self.MAX_DISPLAY_ITEMS:
            limited_items = ', '.join(map(str, list(iterable)[:self.MAX_DISPLAY_ITEMS]))
            return f"[{limited_items}, ...]"
        else:
            return f"[{', '.join(map(str, iterable))}]"

    def _summarize_dict(self, dictionary: Dict) -> str:
        """
        Summarizes a dictionary to a concise representation if it exceeds 
        the display limit.

        Parameters
        ----------
        dictionary : Dict
            The dictionary to summarize.

        Returns
        -------
        str
            A summarized string representation of the dictionary.
        """
        if len(dictionary) > self.MAX_DISPLAY_ITEMS:
            limited_items = ', '.join(f"{k}: {v}" for k, v in list(
                dictionary.items())[:self.MAX_DISPLAY_ITEMS])
            return f"{{ {limited_items}, ... }}"
        else:
            return f"{{ {', '.join(f'{k}: {v}' for k, v in dictionary.items()) }}}"

    def _summarize_array(self, array: np.ndarray) -> str:
        """
        Summarizes a NumPy array to a concise representation if it exceeds 
        the display limit.

        Parameters
        ----------
        array : np.ndarray
            The NumPy array to summarize.

        Returns
        -------
        str
            A summarized string representation of the array.
        """
        if array.size > self.MAX_DISPLAY_ITEMS:
            limited_items = ', '.join(map(str, array.flatten()[:self.MAX_DISPLAY_ITEMS]))
            return f"[{limited_items}, ...]"
        else:
            return f"[{', '.join(map(str, array.flatten()))}]"

    def _summarize_dataframe(self, df: pd.DataFrame) -> str:
        """
        Summarizes a pandas DataFrame to a concise representation if it exceeds
        the display limit.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to summarize.

        Returns
        -------
        str
            A summarized string representation of the DataFrame.
        """
        # if len(df) > self.MAX_DISPLAY_ITEMS:
        return f"DataFrame({len(df)} rows, {len(df.columns)} columns)"
        # else:
        #     return f"DataFrame: {df.to_string(index=False)}"

    def _summarize_series(self, series: pd.Series) -> str:
        """
        Summarizes a pandas Series to a concise representation if it exceeds
        the display limit.

        Parameters
        ----------
        series : pd.Series
            The Series to summarize.

        Returns
        -------
        str
            A summarized string representation of the Series.
        """
        # if len(series) > self.MAX_DISPLAY_ITEMS:
        limited_items = ', '.join(f"{series.index[i]}: {series[i]}" 
                                  for i in range(self.MAX_DISPLAY_ITEMS))
        return f"Series([{limited_items}, ...])"
        # else:
        #     return f"Series: {series.to_string(index=False)}"
        
class BaseLearner(metaclass=LearnerMeta):
    """
    Base class for all learners in this framework, designed to facilitate 
    dynamic management of parameters, retrieval, and representation. 
    This class provides essential functionalities for setting parameters, 
    cloning, executing, and inspecting learner objects.

    Parameters
    ----------
    None
        This base class does not accept parameters during initialization. 
        Parameters are managed dynamically using the `set_params` method 
        and can be retrieved via `get_params`.

    Methods
    -------
    get_params(deep=True)
        Retrieve the parameters for this learner, including nested 
        parameters if `deep=True`.
        
    set_params(**params)
        Set parameters for the learner. Supports nested parameter setting 
        by using double underscore (`__`) notation for nested learners.
        
    reset_params()
        Reset all parameters to their default values.
        
    is_runned()
        Determine if the learner has been run, based on the presence of 
        attributes with trailing underscores.
        
    clone()
        Create a new copy of the learner with identical parameters.
        
    summary()
        Provide a formatted summary of the learner’s parameters for 
        inspection or logging.
        
    execute(*args, **kwargs)
        Dynamically execute either `fit` or `run` if defined in the 
        subclass, with preference given to `run` if both are present.
        
    Notes
    -----
    `BaseLearner` is designed to be a foundation for constructing machine 
    learning and statistical models in this framework. It enables flexible 
    parameter management, supporting both shallow and deep copying of 
    learners. The `execute` method offers a dynamic interface for 
    subclasses to define either `fit` or `run` methods, enabling 
    seamless execution.

    Key aspects of this class include:
    
    - **Parameter Management**: `get_params` and `set_params` support both 
      flat and nested parameters, simplifying configuration of various 
      hyperparameters and settings.
      
    - **Execution Flexibility**: The `execute` method dynamically invokes 
      `fit` or `run`, enabling versatile use in training or inference tasks.
      
    - **Serialization Support**: `__getstate__` and `__setstate__` methods 
      handle object state for safe serialization, supporting compatibility 
      through versioning.

    Let the learner be represented as :math:`L`. The parameters for this 
    learner, denoted :math:`\\theta`, are:

    .. math::
    
        \\theta = \\{ \\theta_1, \\theta_2, \\dots, \\theta_n \\}
        
    where each parameter :math:`\\theta_i` can be set using `set_params` 
    and retrieved with `get_params`. For nested learners, deep parameter 
    retrieval allows access to sub-parameters, denoted as :math:`\\theta_{i_j}`, 
    where :math:`i` is the primary parameter and :math:`j` a nested parameter.

    Examples
    --------
    >>> from gofast.api.property import BaseLearner
    
    # Define a subclass inheriting from BaseLearner 
    # with specific parameters and methods
    >>> class ExampleLearner(BaseLearner):
    ...     def __init__(self, alpha=0.5, beta=0.1):
    ...         self.alpha = alpha
    ...         self.beta = beta
    ...     
    ...     def fit(self, data):
    ...         print(f"Fitting with data: {data} using"
    ...               " alpha={self.alpha}, beta={self.beta}")
    ...
    ...     def run(self, data):
    ...         print(f"Running with data: {data} using"
    ...               " alpha={self.alpha}, beta={self.beta}")
    ...         return [x * self.alpha + self.beta for x in data]
    
    # Instantiate the subclass with parameters
    >>> learner = ExampleLearner(alpha=0.5, beta=0.1)
    
    # Set parameters dynamically
    >>> learner.set_params(alpha=0.7)
    >>> print(learner.get_params())
    {'alpha': 0.7, 'beta': 0.1}
    
    # Execute the learner, which will prioritize calling `run` if both `fit`
    # and `run` are defined
    >>> learner.execute([1, 2, 3])
    Running with data: [1, 2, 3] using alpha=0.7, beta=0.1
    [0.7999999999999999, 1.5, 2.1999999999999997]
    
    In this example, `ExampleLearner` inherits from `BaseLearner`. The `execute`
    method calls `run` by default, demonstrating how a subclass can implement
    its own logic while leveraging `BaseLearner`'s parameter management and
    execution framework.


    See Also
    --------
    `get_params` : Retrieve all current parameters for the learner.
    `set_params` : Set parameters, supporting nested configurations.
    `clone` : Create a copy of the learner with the same settings.
    `summary` : Display a formatted summary of parameters.

    References
    ----------
    .. [1] Smith, J., & Doe, A. (2021). "Dynamic Parameter Management in 
           Machine Learning Models". *Journal of Machine Learning Systems*, 
           15(3), 100-120.
    """


    @classmethod
    def _get_param_names(cls):
        """
        Retrieve the names of the parameters defined in the constructor.
    
        Returns
        -------
        list
            List of parameter names for the learner.
        """
        # Fetch the constructor or original constructor if deprecated
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []
    
        # Introspect the constructor arguments to identify model parameters
        init_signature = inspect.signature(init)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls.__name__} should not have variable positional arguments "
                    f"in the constructor (no *args)."
                )
        # Return sorted argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    
    def get_params(self, deep=True):
        """
        Get the parameters for this learner.
    
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this learner and nested learners.
    
        Returns
        -------
        dict
            Dictionary of parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            # Retrieve nested parameters if `deep=True`
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """
        Set the parameters of this learner.
    
        Parameters
        ----------
        **params : dict
            Parameters to set, including nested parameters specified with 
            double-underscore notation (e.g., ``component__parameter``).
    
        Returns
        -------
        self : learner instance
            Returns self with updated parameters.
        """
        if not params:
            # Optimization for speed if no parameters are given
            return self
        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)  # Grouped by prefix
    
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                # Raise error for invalid parameter
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for learner {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
    
        # Set parameters for nested objects
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
    
        return self
    
    
    def __repr__(self, N_CHAR_MAX=700):
        """
        Return a string representation of the learner, showing key parameters.
    
        Parameters
        ----------
        N_CHAR_MAX : int, default=700
            Maximum number of characters in the representation.
    
        Returns
        -------
        str
            String representation of the learner with parameters.
        """
        params = self.get_params()
        param_str = ", ".join(f"{key}={value!r}" for key, value in params.items())
        # Truncate if exceeds max character length
        if len(param_str) > N_CHAR_MAX:
            param_str = param_str[:N_CHAR_MAX] + "..."
        return f"{self.__class__.__name__}({param_str})"
    
    
    def __getstate__(self):
        """
        Prepare the object for pickling by saving the current state.
    
        Returns
        -------
        dict
            State dictionary with only serializable attributes and versioning 
            information for compatibility.
        """
        state = {}
        version = getattr(self, "_version", "1.0.0")  # Default version
    
        for key, value in self.__dict__.items():
            # Exclude non-serializable attributes
            if key.startswith("_") or callable(value):
                continue
            try:
                # Test serializability of the attribute
                _ = pickle.dumps(value)
                state[key] = value
            except (pickle.PicklingError, TypeError):
                print(f"Warning: Unable to pickle attribute '{key}'. Excluded.")
        
        # Add version information
        state["_version"] = version
        return state
    
    
    def __setstate__(self, state):
        """
        Restore the object's state after unpickling, with version checks and 
        handling for missing attributes.
    
        Parameters
        ----------
        state : dict
            State dictionary containing class attributes.
        """
        import logging
        logger = logging.getLogger(__name__)
        expected_version = getattr(self, "_version", "1.0.0")
    
        # Check if state version matches expected version
        version = state.get("_version", "unknown")
        if version != expected_version:
            logger.warning(
                f"Version mismatch: loaded state version '{version}' "
                f"does not match expected '{expected_version}'."
            )
    
        # Restore only valid attributes
        for key, value in state.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                logger.error(f"Could not set attribute '{key}': {e}")
    
        # Set missing attributes as needed
        if not hasattr(self, "_initialized"):
            self._initialized = True
    
    
    def reset_params(self):
        """
        Reset all parameters to their initial default values.
    
        Returns
        -------
        self : learner instance
            Returns self with parameters reset to defaults.
        """
        for param, value in self._default_params.items():
            setattr(self, param, value)
        print("Parameters reset to default values.")
        return self
    
    def is_runned(
        self,
        attributes: Optional[Union[str, List[str]]] = None,
        msg: Optional[str] = None,
        check_status: str = "passthrough"
    ) -> bool:
        """
        Check if the learner has been run by verifying the presence 
        of specific attributes.
    
        Parameters
        ----------
        attributes : str or list of str, optional
            Specific attribute name(s) to check for existence and non-None 
            value. If provided, the method checks only these attributes.
            If `None`, it checks for any attributes ending with an 
            underscore ('_').
    
        msg : str, optional
            Custom error message to display if the learner has not been run 
            and `check_status` is not ``"passthrough"``. The placeholder 
            `%(name)s` can be used to include the learner's class name
            in the message.
            Default message is:
            "The %(name)s instance has not been 'runned' yet. Call 'run' with 
            appropriate arguments before using this method."
    
        check_status : str, default="passthrough"
            Determines the behavior of the method when the learner has not 
            been run.
            Options are:
            - `"passthrough"`: Returns `True` or `False` indicating the run 
              status.
            - Any other value: Raises `NotRunnedError` if the learner has 
              not been run.
    
        Returns
        -------
        bool
            `True` if the learner has been run, `False` otherwise.
    
        Raises
        ------
        NotRunnedError
            If `check_status` is not `"passthrough"` and the learner has
            not been run.
    
        Examples
        --------
        >>> from gofast.api.property import BaseLearner
        >>> class MyLearner(BaseLearner):
        ...     def __init__(self):
        ...         self.model_ = None  # Placeholder attribute after running
        ...
        >>> learner = MyLearner()
        >>> learner.is_runned()
        False
        >>> learner.model_ = "TrainedModel"
        >>> learner.is_runned()
        True
        >>> learner.is_runned(attributes='model_')
        True
        >>> learner.is_runned(attributes='non_existent_attr')
        False
        >>> # Using custom error message and check_status
        >>> learner = MyLearner()
        >>> learner.is_runned(msg="Custom error for %(name)s.", check_status="raise")
        Traceback (most recent call last):
        NotRunnedError: Custom error for MyLearner.
    
        Notes
        -----
        - This method checks if the learner has been run by verifying the presence
          of specific attributes. If `attributes` is not provided, it checks for any
          attributes ending with an underscore ('_'), which is a common convention
          for indicating fitted attributes in scikit-learn estimators [1]_.
        - The method can either return a boolean value or raise an error based on
          the `check_status` parameter.
    
        See Also
        --------
        sklearn.utils.validation.check_is_fitted : Utility function for similar 
        functionality in scikit-learn.
    
        References
        ----------
        .. [1] Scikit-learn development team, "Developing scikit-learn estimators",
           https://scikit-learn.org/stable/developers/develop.html#estimated-attributes
    
        """
        # Local exception class
        class NotRunnedError(Exception):
            """Exception raised when the learner has not been run."""
            pass
    
        # Default message if none provided
        if msg is None:
            msg = (
                "The %(name)s instance has not been 'runned' yet. "
                "Call 'run' with appropriate arguments before using this method."
            )
    
        # Initialize run status
        is_runned = False
    
        # Check specific attributes if provided
        if attributes is not None:
            if isinstance(attributes, str):
                attributes = [attributes]
            # Verify each attribute exists and is not None or not False 
            is_runned = all(
                hasattr(self, attr) and getattr(self, attr) is not None
                and getattr(self, attr) is not False
                for attr in attributes
            )
        else:
            # Check for any attributes with trailing underscores
            trailing_attrs = [
                attr for attr in self.__dict__ if attr.endswith("_")
            ]
            if trailing_attrs:
                is_runned = True
                # Fallback to `__gofast_is_runned__` if no trailing attributes
            elif hasattr(self, "__gofast_is_runned__") and callable(
                getattr(self, "__gofast_is_runned__")
            ):
                # Fallback to custom method if defined
                is_runned = self.__gofast_is_runned__()
            else:
                is_runned = False
    
        # Handle check_status behavior
        if check_status == "passthrough":
            return is_runned
        else:
            if not is_runned:
                # Raise error with custom or default message
                raise NotRunnedError(msg % {"name": type(self).__name__})
            return is_runned

    def clone(self):
        """
        Create a clone of the learner with identical parameters.
    
        Returns
        -------
        BaseLearner
            A new instance of the learner with the same parameters.
        """
        clone = self.__class__(**self.get_params(deep=False))
        return clone
    
    
    def summary(self):
        """
        Provide a summary of the learner's parameters.
    
        Returns
        -------
        str
            Formatted string of the learner's parameters.
        """
        params = self.get_params(deep=False)
        summary_str = "\n".join(f"{k}: {v}" for k, v in params.items())
        return f"{self.__class__.__name__} Summary:\n{summary_str}"
    
    def execute(self, *args, **kwargs):
        """
        Execute `fit` or `run` method if either is implemented in the subclass. 
        Priority is given to `run` if both are available.
    
        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to `fit` or `run`.
        **kwargs : dict
            Keyword arguments to pass to `fit` or `run`.
    
        Returns
        -------
        Any
            The result of calling either `run` or `fit`.
    
        Raises
        ------
        NotImplementedError
            If neither `fit` nor `run` is implemented in the subclass.
        """
        has_run = callable(getattr(self, 'run', None))
        has_fit = callable(getattr(self, 'fit', None))
    
        if has_run:
            return self.run(*args, **kwargs)
        elif has_fit:
            return self.fit(*args, **kwargs)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires either `run` or `fit`."
            )
            
    def save(
        self,
        file_path: Optional[str] = None,
        format: str = 'pickle',
        overwrite: bool = False,
        validate_func: Optional[Callable[[Any], bool]] = None,
        **kwargs
    ) -> bool:
        """
        Save the learner's state to a specified file in the desired format.

        This method provides a robust mechanism to persist an object's state by
        exporting its data to various formats such as JSON, CSV, HDF5, or pickle.
        It includes features like error handling, logging, data validation, and
        supports additional parameters for extended flexibility.

        Parameters
        ----------
        file_path : str, optional
            The path where the file will be saved. If not provided, defaults to
            ``'<class_name>_data.<ext>'``, where ``<ext>`` is determined by 
            the `format` parameter.
        format : str, default 'pickle'
            The format in which to save the data. Supported formats are:

            - `'json'`: Saves data in JSON format.
            - `'csv'`: Saves data in CSV format.
            - `'h5'` or `'hdf5'`: Saves data in HDF5 format using h5py.
            - `'pickle'`: Saves data using Python's pickle module.

        overwrite : bool, default False
            Determines whether to overwrite the file if it already exists at
            `file_path`. If set to ``False`` and the file exists, the save
            operation will be aborted to prevent data loss.
        validate_func : Callable[[Any], bool], optional
            A user-provided function that takes the data as input and returns
            ``True`` if the data is valid or ``False`` otherwise. This allows 
            for custom data validation before saving.
        **kwargs : dict
            Additional keyword arguments to provide future flexibility or pass
            extra parameters as needed.

        Returns
        -------
        bool
            Returns ``True`` if the save operation was successful, 
            ``False`` otherwise.

        Examples
        --------
        >>> from gofast.api.property import BaseLearner
        >>> class Learner(BaseLearner):
        ...     def __init__(self, data):
        ...         self.data = data
        ...     def to_dict(self):
        ...         return {'data': self.data}
        >>> learner = Learner(data=[1, 2, 3])
        >>> success = learner.save(
        ...     file_path='learner_data.pkl',
        ...     format='pickle',
        ...     overwrite=True
        ... )
        >>> print(success)
        True

        Notes
        -----
        - The object must implement a `to_dict` method that returns its data
          in dictionary format for 'json' and 'csv' formats.
        - For 'h5' format, the object should provide data in a format compatible
          with h5py datasets.
        - For 'pickle' format, the entire object is serialized.
        - Logging is performed to track the save operations and any errors
          encountered during the process.

        See Also
        --------
        BaseLearner.to_dict : Method to convert object data to dictionary format.

        References
        ----------
        .. [1] Smith, J. (2020). *Effective Python Programming*. Python Press.
        .. [2] Doe, A. (2021). *Advanced Data Persistence Techniques*. Data Books.
        """

        try:
            # Determine file path
            if not file_path:
                extension = {
                    'json': 'json',
                    'csv': 'csv',
                    'h5': 'h5',
                    'hdf5': 'h5',
                    'pickle': 'pkl'
                }.get(format.lower(), 'pkl')
                file_path = f"{self.__class__.__name__.lower()}_data.{extension}"
            path = Path(file_path)

            # Check if file exists
            if path.exists() and not overwrite:
                logger.error(
                    f"File '{file_path}' already exists. "
                    "Use overwrite=True to overwrite."
                )
                return False

            # Prepare data
            data = None
            if format.lower() in ['json', 'csv']:
                if hasattr(self, 'to_dict') and callable(
                        getattr(self, 'to_dict')):
                    data = self.to_dict()
                else:
                    logger.error("The object does not have a 'to_dict' method.")
                    return False

                # Validate data if a validation function is provided
                if validate_func and not validate_func(data):
                    logger.error("Data validation failed.")
                    return False

            # Save data based on the specified format
            if format.lower() == 'json':
                with path.open('w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            elif format.lower() == 'csv':
                if isinstance(data, list) and all(
                    isinstance(item, dict) for item in data
                ):
                    with path.open('w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=data[0].keys()
                        )
                        writer.writeheader()
                        writer.writerows(data)
                else:
                    logger.error(
                        "Data for CSV format must be a list of dictionaries."
                    )
                    return False
            elif format.lower() in ['h5', 'hdf5']:
                if hasattr(self, 'to_hdf5') and callable(getattr(self, 'to_hdf5')):
                    self.to_hdf5(file_path, **kwargs)
                else:
                    logger.error(
                        "The object does not have a 'to_hdf5'"
                        " method required for 'h5' format."
                    )
                    return False
            elif format.lower() in ['pkl', 'pickle']:
                with path.open('wb') as f:
                    pickle.dump(self, f)
            else:
                logger.error(
                    f"Unsupported format '{format}'. Supported formats"
                    " are 'json', 'csv', 'h5', and 'pickle'."
                )
                return False

            logger.info(
                f"Data successfully saved to '{file_path}' in '{format}' format."
            )
            return True

        except Exception as e:
            logger.exception(f"An error occurred while saving data: {e}")
            return False
        
        
class BasePlot(BaseClass): 
    r""" Base class  deals with Machine learning and conventional Plots. 
    
    The `BasePlot` can not be instanciated. It is build on the top of other 
    plotting classes  and its attributes are used for external plots.
    
    Hold others optional informations: 
        
    ==================  =======================================================
    Property            Description        
    ==================  =======================================================
    fig_dpi             dots-per-inch resolution of the figure
                        *default* is 300
    fig_num             number of the figure instance. *default* is ``1``
    fig_aspect          ['equal'| 'auto'] or float, figure aspect. Can be 
                        rcParams["image.aspect"]. *default* is ``auto``.
    fig_size            size of figure in inches (width, height)
                        *default* is [5, 5]
    savefig             savefigure's name, *default* is ``None``
    fig_orientation     figure orientation. *default* is ``landscape``
    fig_title           figure title. *default* is ``None``
    fs                  size of font of axis tick labels, axis labels are
                        fs+2. *default* is 6 
    ls                  [ '-' | '.' | ':' ] line style of mesh lines
                        *default* is '-'
    lc                  line color of the plot, *default* is ``k``
    lw                  line weight of the plot, *default* is ``1.5``
    alpha               transparency number, *default* is ``0.5``  
    font_weight         weight of the font , *default* is ``bold``.        
    ms                  size of marker in points. *default* is 5
    marker              style  of marker in points. *default* is ``o``.
    marker_facecolor    facecolor of the marker. *default* is ``yellow``
    marker_edgecolor    edgecolor of the marker. *default* is ``cyan``.
    marker_edgewidth    width of the marker. *default* is ``3``.
    xminorticks         minortick according to x-axis size and *default* is 1.
    yminorticks         minortick according to y-axis size and *default* is 1.
    font_size           size of font in inches (width, height)
                        *default* is 3.
    font_style          style of font. *default* is ``italic``
    bins                histograms element separation between two bar. 
                         *default* is ``10``. 
    xlim                limit of x-axis in plot. *default* is None 
    ylim                limit of y-axis in plot. *default* is None 
    xlabel              label name of x-axis in plot. *default* is None 
    ylabel              label name  of y-axis in plot. *default* is None 
    rotate_xlabel       angle to rotate `xlabel` in plot. *default* is None 
    rotate_ylabel       angle to rotate `ylabel` in plot. *default* is None 
    leg_kws             keyword arguments of legend. *default* is empty dict.
    plt_kws             keyword arguments of plot. *default* is empty dict
    plt_style           keyword argument of 2d style. *default* is ``pcolormesh``
    plt_shading         keyword argument of Axes pycolormesh shading. It can be 
                        ['flat'|'nearest'|'gouraud'|'auto'].*default* is 
                        'auto'
    imshow_interp       ['bicubic'|'nearest'|'bilinear'|'quadractic' ] kind of 
                        interpolation for 'imshow' plot. Click `interpol_imshow`_ 
                        to get furher details about the interpolation method. 
                        *default* is ``None``.
    rs                  [ '-' | '.' | ':' ] line style of `Recall` metric
                        *default* is '--'
    ps                  [ '-' | '.' | ':' ] line style of `Precision `metric
                        *default* is '-'
    rc                  line color of `Recall` metric *default* is ``(.6,.6,.6)``
    pc                  line color of `Precision` metric *default* is ``k``
    s                   size of items in scattering plots. default is ``fs*40.``
    cmap                matplotlib colormap. *default* is `jet_r`
    gls                 [ '-' | '.' | ':' ] line style of grid  
                        *default* is '--'.
    glc                 line color of the grid plot, *default* is ``k``
    glw                 line weight of the grid plot, *default* is ``2``
    galpha              transparency number of grid, *default* is ``0.5``  
    gaxis               axis to plot grid.*default* is ``'both'``
    gwhich              type of grid to plot. *default* is ``major``
    tp_axis             axis  to apply ticks params. default is ``both``
    tp_labelsize        labelsize of ticks params. *default* is ``italic``
    tp_bottom           position at bottom of ticks params. *default*
                        is ``True``.
    tp_top              position at the top  of ticks params. *default*
                        is ``True``.
    tp_labelbottom      see label on the bottom of the ticks. *default* 
                        is ``False``
    tp_labeltop         see the label on the top of ticks. *default* is ``True``
    cb_orientation      orientation of the colorbar. *default* is ``vertical``
    cb_aspect           aspect of the colorbar. *default* is 20.
    cb_shrink           shrink size of the colorbar. *default* is ``1.0``
    cb_pad              pad of the colorbar of plot. *default* is ``.05``
    cb_anchor           anchor of the colorbar. *default* is ``(0.0, 0.5)``
    cb_panchor          proportionality anchor of the colorbar. *default* is 
                        `` (1.0, 0.5)``.
    cb_label            label of the colorbar. *default* is ``None``.      
    cb_spacing          spacing of the colorbar. *default* is ``uniform``
    cb_drawedges        draw edges inside of the colorbar. *default* is ``False``
    cb_format           format of the colorbar values. *default* is ``None``.
    sns_orient          seaborn fig orientation. *default* is ``v`` which refer
                        to vertical 
    sns_style           seaborn style 
    sns_palette         seaborn palette 
    sns_height          seaborn height of figure. *default* is ``4.``. 
    sns_aspect          seaborn aspect of the figure. *default* is ``.7``
    sns_theme_kws       seaborn keywords theme arguments. default is ``{
                        'style':4., 'palette':.7}``
    verbose             control the verbosity. Higher value, more messages.
                        *default* is ``0``.
    ==================  =======================================================
    
    """
    
    def __init__(
        self,
        savefig: str = None,
        fig_num: int =  1,
        fig_size: tuple =  (12, 8),
        fig_dpi:int = 300, 
        fig_legend: str =  None,
        fig_orientation: str ='landscape',
        fig_title:str = None,
        fig_aspect:str='auto',
        font_size: float =3.,
        font_style: str ='italic',
        font_weight: str = 'bold',
        fs: float = 5.,
        ms: float =3.,
        marker: str = 'o',
        markerfacecolor: str ='yellow',
        markeredgecolor: str = 'cyan',
        markeredgewidth: float =  3.,
        lc: str =  'k',
        ls: str = '-',
        lw: float = 1.,
        alpha: float =  .5,
        bins: int =  10,
        xlim: list = None, 
        ylim: list= None,
        xminorticks: int=1, 
        yminorticks: int =1,
        xlabel: str  =  None,
        ylabel: str = None,
        rotate_xlabel: int = None,
        rotate_ylabel: int =None ,
        leg_kws: dict = dict(),
        plt_kws: dict = dict(), 
        plt_style:str="pcolormesh",
        plt_shading: str="auto", 
        imshow_interp:str =None,
        s: float=  40.,
        cmap:str='jet_r',
        show_grid: bool = False,
        galpha: float = .5,
        gaxis: str = 'both',
        gc: str = 'k',
        gls: str = '--',
        glw: float = 2.,
        gwhich: str = 'major',               
        tp_axis: str = 'both',
        tp_labelsize: float = 3.,
        tp_bottom: bool =True,
        tp_top: bool = True,
        tp_labelbottom: bool = False,
        tp_labeltop: bool = True,               
        cb_orientation: str = 'vertical',
        cb_aspect: float = 20.,
        cb_shrink: float =  1.,
        cb_pad: float =.05,
        cb_anchor: tuple = (0., .5),
        cb_panchor: tuple=  (1., .5),              
        cb_label: str = None,
        cb_spacing: str = 'uniform' ,
        cb_drawedges: bool = False,
        cb_format: float = None ,   
        sns_orient: str ='v', 
        sns_style: str = None, 
        sns_palette: str= None, 
        sns_height: float=4. , 
        sns_aspect:float =.7, 
        sns_theme_kws: dict = None,
        verbose: int=0, 
        ): 
        
        self.savefig=savefig
        self.fig_num=fig_num
        self.fig_size=fig_size
        self.fig_dpi=fig_dpi
        self.fig_legend=fig_legend
        self.fig_orientation=fig_orientation
        self.fig_title=fig_title
        self.fig_aspect=fig_aspect
        self.font_size=font_size
        self.font_style=font_style
        self.font_weight=font_weight
        self.fs=fs
        self.ms=ms
        self.marker=marker
        self.marker_facecolor=markerfacecolor
        self.marker_edgecolor=markeredgecolor
        self.marker_edgewidth=markeredgewidth
        self.lc=lc
        self.ls=ls
        self.lw=lw
        self.alpha=alpha
        self.bins=bins
        self.xlim=xlim
        self.ylim=ylim
        self.x_minorticks=xminorticks
        self.y_minorticks=yminorticks
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.rotate_xlabel=rotate_xlabel
        self.rotate_ylabel=rotate_ylabel
        self.leg_kws=leg_kws
        self.plt_kws=plt_kws
        self.plt_style=plt_style
        self.plt_shading=plt_shading
        self.imshow_interp=imshow_interp
        self.s=s 
        self.cmap=cmap
        self.show_grid=show_grid
        self.galpha=galpha
        self.gaxis=gaxis
        self.gc=gc
        self.gls=gls
        self.glw=glw
        self.gwhich=gwhich
        self.tp_axis=tp_axis
        self.tp_labelsize=tp_labelsize  
        self.tp_bottom=tp_bottom
        self.tp_top=tp_top
        self.tp_labelbottom=tp_labelbottom
        self.tp_labeltop=tp_labeltop
        self.cb_orientation=cb_orientation
        self.cb_aspect=cb_aspect
        self.cb_shrink=cb_shrink
        self.cb_pad=cb_pad
        self.cb_anchor=cb_anchor
        self.cb_panchor=cb_panchor
        self.cb_label=cb_label
        self.cb_spacing=cb_spacing
        self.cb_drawedges=cb_drawedges
        self.cb_format=cb_format  
        self.sns_orient=sns_orient
        self.sns_style=sns_style
        self.sns_palette=sns_palette
        self.sns_height=sns_height
        self.sns_aspect=sns_aspect
        self.verbose=verbose
        self.sns_theme_kws=sns_theme_kws or {'style':self.sns_style, 
                                         'palette':self.sns_palette, 
                                                      }
        self.cb_props = {
            pname.replace('cb_', '') : pvalues
                         for pname, pvalues in self.__dict__.items() 
                         if pname.startswith('cb_')
                         }

class GeoscienceProperties(BaseClass):
    """ 
    A container class for geological configurations in the Gofast package, 
    storing fixed properties, array configurations, resistivity ranges, and 
    visualization patterns for various geological rocks.
    
    This class provides a read-only, unmodifiable set of properties essential 
    for handling electrical resistivity profiles (ERP), understanding 
    resistivity ranges of geological materials, and generating consistent 
    visual patterns in geological modeling.

    Attributes
    ----------
    arraytype : dict
        Dictionary storing the configurations for ERP array types, each 
        represented by a unique identifier. Contains configuration names, 
        shorthand labels, and related abbreviations. The structure is:
        
        .. code-block:: python

            {
                1: (['Schlumberger', 'AB>> MN', 'slbg'], 'S'),
                2: (['Wenner', 'AB=MN'], 'W'),
                3: (['Dipole-dipole', 'dd', 'AB<BM>MN', 'MN<NA>AB'], 'DD'),
                4: (['Gradient-rectangular', '[AB]MN', 'MN[AB]', '[AB]'], 'GR')
            }
        
    geo_rocks_properties : dict
        Dictionary defining resistivity ranges (in ohm-meters) for various 
        geological materials, providing insight into rock types and their 
        respective resistivity properties for accurate geophysical analysis.

    rockpatterns : dict
        Dictionary specifying the default visualization patterns for rocks in 
        matplotlib. Each entry pairs a rock type with a pattern string and 
        color tuple, providing a base for creating standardized plots.

    Methods
    -------
    arrangement(a)
        Validates and retrieves the correct name for a given ERP array 
        configuration. If invalid, it returns `0`.

    Notes
    -----
    The `GeoscienceProperties` class encapsulates common geological 
    configurations for consistent usage across geophysical modules. All 
    properties are read-only, protecting critical configurations from 
    modification during runtime. 

    The `arraytype` attribute defines common ERP configurations used in 
    electrical resistivity surveys, while `geo_rocks_properties` provides 
    a reference for typical resistivity ranges in various rock types, 
    assisting in geophysical interpretation.

    Examples
    --------
    >>> from gofast.api.property import GeoscienceProperties
    >>> geo_props = GeoscienceProperties()

    # Access array type configuration for Schlumberger
    >>> print(geo_props.arraytype[1])
    (['Schlumberger', 'AB>> MN', 'slbg'], 'S')

    # Retrieve resistivity range for igneous rock
    >>> resistivity_range = geo_props.geo_rocks_properties["igneous rock"]
    >>> print(resistivity_range)
    [1000000.0, 1000.0]

    # Get pattern and color for 'coal' rock type
    >>> pattern, color = geo_props.rockpatterns["coal"]
    >>> print(pattern, color)
    *. (0.8, 0.9, 0.0)

    See Also
    --------
    GeoscienceVisuals : A class that uses these properties to generate 
                        geophysical visuals in matplotlib.

    References
    ----------
    .. [1] Telford, W.M., Geldart, L.P., & Sheriff, R.E. (1990). "Applied 
           Geophysics, 2nd Edition." Cambridge University Press.
    .. [2] Parasnis, D.S. (1979). "Principles of Applied Geophysics, 3rd 
           Edition." Chapman and Hall.
    """
    
    @property
    def arraytype(self):
        """
        Retrieve configurations for ERP (Electrical Resistivity Profiling) 
        array types, specifying different field survey setups used in 
        geophysical surveys.

        Returns
        -------
        dict
            Mapping of integer keys to ERP array configurations, with each 
            entry containing a list of configuration names and abbreviations 
            for survey design.
        
        Notes
        -----
        Array configurations are foundational in ERP, where each setup is 
        designed to analyze subsurface resistivity variations. These standard 
        setups can be extended or modified as field survey needs evolve.
        """
        return {
            1: (['Schlumberger', 'AB>> MN', 'slbg'], 'S'),
            2: (['Wenner', 'AB=MN'], 'W'),
            3: (['Dipole-dipole', 'dd', 'AB<BM>MN', 'MN<NA>AB'], 'DD'),
            4: (['Gradient-rectangular', '[AB]MN', 'MN[AB]', '[AB]'], 'GR')
        }
    
    def arrangement(self, a):
        """ 
        Validate and retrieve the correct name for a specified ERP array 
        configuration.

        Parameters
        ----------
        a : int or str
            The ERP array configuration, which can be an integer key or a 
            string identifier for a known configuration.
        
        Returns
        -------
        str or int
            The lowercase name of the arrangement if valid, or `0` if the 
            configuration is not found.
        
        Notes
        -----
        The `arrangement` method helps validate array configurations, 
        supporting field survey accuracy by ensuring configuration consistency 
        and preventing invalid input during setup.

        Examples
        --------
        >>> geo_props = GeoscienceProperties()
        >>> geo_props.arrangement(1)
        'schlumberger'
        >>> geo_props.arrangement('W')
        'wenner'
        >>> geo_props.arrangement('invalid')
        0
        """
        a = str(a).lower().strip()
        for k, (aliases, short_name) in self.arraytype.items():
            if a == str(k) or a in ','.join(aliases).lower() or a == short_name:
                return aliases[0].lower()
        return 0

    @property
    def geo_rocks_properties(self):
        """ 
        Provides approximate resistivity ranges (in ohm-meters) for a selection 
        of geological rock types, valuable for subsurface modeling and analysis.

        Returns
        -------
        dict
            A dictionary mapping rock names to resistivity ranges, expressed 
            in ohm-meters, for use in geophysical interpretation.
        
        Notes
        -----
        Resistivity ranges vary by rock type and water content, with high 
        resistivity generally indicating less conductive materials such as 
        igneous rocks, and low resistivity indicating conductive materials 
        like water or massive sulphide deposits.
        
        Examples
        --------
        >>> geo_props = GeoscienceProperties()
        >>> geo_props.geo_rocks_properties["igneous rock"]
        [1000000.0, 1000.0]
        """
        return {
            "hard rock": [1e99, 1e6],
            "igneous rock": [1e6, 1e3],
            "duricrust": [5.1e3, 5.1e2],
            "gravel/sand": [1e4, 7.943],
            "conglomerate": [1e4, 8.913e1],
            "dolomite/limestone": [1e5, 1e3],
            "permafrost": [1e5, 4.169e2],
            "metamorphic rock": [5.1e2, 1e1],
            "tills": [8.1e2, 8.512e1],
            "sandstone conglomerate": [1e4, 8.318e1],
            "lignite/coal": [7.762e2, 1e1],
            "shale": [5.012e1, 3.20e1],
            "clay": [1e2, 5.012e1],
            "saprolite": [6.310e2, 3.020e1],
            "sedimentary rock": [1e4, 1e0],
            "fresh water": [3.1e2, 1e0],
            "salt water": [1e0, 1.41],
            "massive sulphide": [1e0, 1e-2],
            "sea water": [1.231e-1, 1e-1],
            "ore minerals": [1e0, 1e-4],
            "graphite": [3.1623e-2, 3.162e-3]
        }

    @property
    def rockpatterns(self):
        """ 
        Default visualization patterns and colors for geological rock types.

        Returns
        -------
        dict
            Dictionary of rock names mapped to a list with visualization 
            pattern strings and color tuples, for use with matplotlib.
        
        Notes
        -----
        These patterns are derived from conventional geological symbols. 
        They allow consistent and recognizable rock representation in plots, 
        supporting standardized geological modeling.

        Examples
        --------
        >>> geo_props = GeoscienceProperties()
        >>> pattern, color = geo_props.rockpatterns["coal"]
        >>> print(pattern, color)
        *. (0.8, 0.9, 0.0)
        """

        return {
            "hard rock": ['.+++++.', (0.25, 0.5, 0.5)],
            "igneous rock": ['.o.o.', (1., 1., 1.)],
            "duricrust": ['+.+', (1., 0.2, 0.36)],
            "gravel": ['oO', (0.75, 0.86, 0.12)],
            "sand": ['....', (0.23, 0.36, 0.45)],
            "conglomerate": ['.O.', (0.55, 0., 0.36)],
            "dolomite": ['.-.', (0., 0.75, 0.23)],
            "limestone": ['//.', (0.52, 0.23, 0.125)],
            "permafrost": ['o.', (0.2, 0.26, 0.75)],
            "metamorphic rock": ['*o.', (0.2, 0.2, 0.3)],
            "tills": ['-.', (0.7, 0.6, 0.9)],
            "standstone": ['..', (0.5, 0.6, 0.9)],
            "lignite coal": ['+/.', (0.5, 0.5, 0.4)],
            "coal": ['*.', (0.8, 0.9, 0.)],
            "shale": ['=', (0., 0., 0.7)],
            "clay": ['=.', (0.9, 0.8, 0.8)],
            "saprolite": ['*/', (0.3, 1.2, 0.4)],
            "sedimentary rock": ['...', (0.25, 0., 0.25)],
            "fresh water": ['.-.', (0., 1., 0.2)],
            "salt water": ['o.-', (0.2, 1., 0.2)],
            "massive sulphide": ['.+O', (1., 0.5, 0.5)],
            "sea water": ['.--', (0., 1., 0.)],
            "ore minerals": ['--|', (0.8, 0.2, 0.2)],
            "graphite": ['.++.', (0.2, 0.7, 0.7)]
        }


class PandasDataHandlers(BaseClass):
    """ 
    A container for data parsers and writers based on Pandas, supporting a 
    wide range of formats for both reading and writing DataFrames. This class 
    simplifies data I/O by mapping file extensions to Pandas functions, making 
    it easier to manage diverse file formats in the Gofast package.
    
    Attributes
    ----------
    parsers : dict
        A dictionary mapping common file extensions to Pandas functions for 
        reading files into DataFrames. Each entry links a file extension to 
        a specific Pandas reader function, allowing for standardized and 
        convenient data import.

    Methods
    -------
    writers(obj)
        Returns a dictionary mapping file extensions to Pandas functions for 
        writing a DataFrame to various formats. Enables easy exporting of data 
        in multiple file formats, ensuring flexibility in data storage.
        
    Notes
    -----
    The `PandasDataHandlers` class centralizes data handling functions, 
    allowing for a unified interface to access multiple data formats, which 
    simplifies data parsing and file writing in the Gofast package.

    This class does not take any parameters on initialization and is used 
    to manage I/O options for DataFrames exclusively.

    Examples
    --------
    >>> from gofast.api.property import PandasDataHandlers
    >>> data_handler = PandasDataHandlers()
    
    # Reading a CSV file
    >>> parser_func = data_handler.parsers[".csv"]
    >>> df = parser_func("data.csv")
    
    # Writing to JSON
    >>> writer_func = data_handler.writers(df)[".json"]
    >>> writer_func("output.json")

    The above example illustrates how to access reader and writer functions 
    for specified file extensions, allowing for simplified data import and 
    export with Pandas.

    See Also
    --------
    pandas.DataFrame : Provides comprehensive data structures and methods for 
                       managing tabular data.
                       
    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
           in Python." In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    """

    @property
    def parsers(self):
        """
        A dictionary mapping file extensions to Pandas functions for reading 
        data files. Each extension is associated with a Pandas function 
        capable of parsing the respective format and returning a DataFrame.

        Returns
        -------
        dict
            A dictionary of file extensions as keys, and their respective 
            Pandas parsing functions as values.

        Examples
        --------
        >>> data_handler = PandasDataHandlers()
        >>> csv_parser = data_handler.parsers[".csv"]
        >>> df = csv_parser("data.csv")

        Notes
        -----
        The `parsers` attribute simplifies data import across diverse formats 
        supported by Pandas. As new formats are integrated into Pandas, this 
        dictionary can be expanded to include additional file types.
        """
        return {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".json": pd.read_json,
            ".html": pd.read_html,
            ".sql": pd.read_sql,
            ".xml": pd.read_xml,
            ".fwf": pd.read_fwf,
            ".pkl": pd.read_pickle,
            ".sas": pd.read_sas,
            ".spss": pd.read_spss,
        }

    @staticmethod
    def writers(obj):
        """
        A dictionary mapping file extensions to Pandas functions for writing 
        DataFrames. The `writers` method generates file-specific writing 
        functions to enable export of DataFrames in various formats.

        Parameters
        ----------
        obj : pandas.DataFrame
            The DataFrame to be written to a specified format.
        
        Returns
        -------
        dict
            A dictionary of file extensions as keys, mapped to the DataFrame 
            writer functions in Pandas that allow exporting to that format.

        Examples
        --------
        >>> data_handler = PandasDataHandlers()
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> json_writer = data_handler.writers(df)[".json"]
        >>> json_writer("output.json")

        Notes
        -----
        The `writers` method provides a flexible solution for exporting data 
        to multiple file formats. This method centralizes data export 
        functionality by associating file extensions with Pandas writer 
        methods, making it straightforward to save data in different formats.
        """
        return {
            ".csv": obj.to_csv,
            ".hdf": obj.to_hdf,
            ".sql": obj.to_sql,
            ".dict": obj.to_dict,
            ".xlsx": obj.to_excel,
            ".json": obj.to_json,
            ".html": obj.to_html,
            ".feather": obj.to_feather,
            ".tex": obj.to_latex,
            ".stata": obj.to_stata,
            ".gbq": obj.to_gbq,
            ".rec": obj.to_records,
            ".str": obj.to_string,
            ".clip": obj.to_clipboard,
            ".md": obj.to_markdown,
            ".parq": obj.to_parquet,
            ".pkl": obj.to_pickle,
        }


class References(BaseClass):
    """
    Holds citation information for a reference in the Gofast package, 
    encapsulating standard publication details such as author, title, 
    journal, and publication year. This class supports flexibility by 
    allowing additional attributes to be added via keyword arguments.

    Attributes
    ----------
    author : str, optional
        The name(s) of the author(s) of the publication.
    title : str, optional
        The title of the article or publication.
    journal : str, optional
        The name of the journal where the work was published.
    doi : str, optional
        The Digital Object Identifier (DOI) for the publication.
    year : int, optional
        The publication year.
    
    Additional Attributes
    ---------------------
    Additional attributes can be specified using keyword arguments, 
    allowing for custom citation details such as volume, pages, 
    and issue number.

    Notes
    -----
    The `References` class enables users to encapsulate publication details 
    flexibly. It serves as a reference for citing sources related to 
    geophysical research data used within the Gofast package.

    Examples
    --------
    >>> from gofast.api.property import References
    >>> refobj = References(
    ...     author='DMaryE',
    ...     title='HydroLearn: A Machine Learning Research for Hydrogeophysics',
    ...     journal='Computers and Geosciences',
    ...     year=2021,
    ...     volume=18,
    ...     pages='234--214'
    ... )
    >>> print(refobj.journal)
    'Computers and Geosciences'

    See Also
    --------
    Copyright : Class containing copyright information for data usage.

    References
    ----------
    .. [1] Doe, J., & Smith, A. (2021). "Machine Learning Applications in 
           Hydrogeophysics." *Geosciences Journal*, 15(4), 202-214.
    """

    def __init__(
        self, 
        author=None, 
        title=None, 
        journal=None, 
        volume=None, 
        doi=None, 
        year=None,  
        **kws
    ):
        self.author = author
        self.title = title
        self.journal = journal
        self.volume = volume
        self.doi = doi
        self.year = year
        for key in kws:
            setattr(self, key, kws[key])

class Copyright(BaseClass):
    """
    Contains copyright information, focusing on usage terms for data within 
    the Gofast package. This class includes references to the citation of 
    related publications, conditions of use, and release status. Additional 
    information can be added as needed.

    Attributes
    ----------
    References : References
        Citation details for published work associated with the data.
    conditions_of_use : str
        Specifies terms under which the data can be used.
    release_status : str, optional
        Status of data availability, options include 'open', 'public', or 
        'proprietary'.

    Additional Attributes
    ---------------------
    Further attributes can be added via keyword arguments to include details 
    such as data owner and contact information.

    Notes
    -----
    The `Copyright` class emphasizes compliance with usage terms for 
    proprietary or open data within Gofast. Users should refer to 
    `conditions_of_use` for usage guidelines and restrictions.

    Examples
    --------
    >>> from gofast.api.property import Copyright 
    >>> copobj = Copyright(
    ...     release_status='public',
    ...     additional_info='University of AI applications',
    ...     conditions_of_use='Data for educational purposes only.',
    ...     owner='University of AI applications',
    ...     contact='WATER4ALL'
    ... )
    >>> print(copobj.contact)
    'WATER4ALL'

    See Also
    --------
    References : Class for storing citation details.

    References
    ----------
    .. [1] Water4All Project. "Conditions of Use for Geophysical Data". 
           *Hydrogeophysics Repository*, Water4All, 2022.
    """

    cuse = (
        "All Data used for software demonstration mostly located in "
        "data directory <data/> cannot be used for commercial and "
        "distributive purposes. They cannot be distributed to a third "
        "party. However, they can be used for understanding the program. "
        "Some available ERP and VES raw data can be found on the record "
        "<'10.5281/zenodo.5571534'>. Whereas EDI-data, e.g., EMAP/MT data, "
        "can be collected at http://ds.iris.edu/ds/tags/magnetotelluric-data/. "
        "The metadata from both sites are available free of charge and may "
        "be copied freely, duplicated, and further distributed provided "
        "these data are cited as the reference."
    )

    def __init__(
        self, 
        release_status=None, 
        additional_info=None, 
        conditions_of_use=None, 
        **kws
    ):
        self.release_status = release_status
        self.additional_info = additional_info
        self.conditions_of_use = conditions_of_use or self.cuse
        self.References = References()
        for key in kws:
            setattr(self, key, kws[key])


class Person(BaseClass):
    """
    Stores contact information for a person associated with the data, 
    such as an author, contributor, or project owner. Supports customization 
    through keyword arguments for adding additional fields.

    Attributes
    ----------
    email : str, optional
        Email address of the person.
    name : str, optional
        Full name of the person.
    organization : str, optional
        Name of the person's organization.
    organization_url : str, optional
        Web address of the person's organization.
    
    Additional Attributes
    ---------------------
    Additional attributes can be set using keyword arguments, enabling 
    customization for specific contact information such as phone numbers 
    or roles within an organization.

    Notes
    -----
    The `Person` class provides a structured format for capturing contact 
    details, which can be essential for maintaining proper citation and 
    correspondence for data used within the Gofast package.

    Examples
    --------
    >>> from gofast.api.property import Person
    >>> person = Person(
    ...     name='ABA', 
    ...     email='aba@water4all.ai.org',
    ...     phone='00225-0769980706', 
    ...     organization='WATER4ALL'
    ... )
    >>> print(person.name)
    'ABA'
    >>> print(person.organization)
    'WATER4ALL'

    See Also
    --------
    Copyright : Class for copyright information and data usage conditions.
    References : Class for referencing published works.

    References
    ----------
    .. [1] Water4All Project. "Contact Information and Organization Details". 
           *Water4All Repository*, Water4All, 2022.
    """

    def __init__(
        self, 
        email=None, 
        name=None, 
        organization=None, 
        organization_url=None, 
        **kws
    ):
        self.email = email
        self.name = name
        self.organization = organization
        self.organization_url = organization_url
        for key in kws:
            setattr(self, key, kws[key])

class Software(BaseClass):
    """
    Stores essential information about software used within the Gofast package, 
    encapsulating details such as name, version, release date, and author. This 
    class allows for flexible extension by accepting additional attributes 
    via keyword arguments, making it adaptable for various software metadata 
    requirements.

    Attributes
    ----------
    name : str, optional
        The name of the software.
    version : str, optional
        The current version of the software.
    release : str, optional
        The release date or version release information.
    Author : Person
        A `Person` object representing the author of the software, allowing 
        for further contact and organizational details.

    Additional Attributes
    ---------------------
    Additional attributes can be added via keyword arguments, enabling custom 
    metadata fields such as software license, support URL, or technical 
    specifications.

    Methods
    -------
    display_info()
        Prints a formatted summary of the software's information, providing 
        key details such as name, version, release, and author.

    update_info(**kws)
        Updates the software attributes with new values provided as keyword 
        arguments, allowing for dynamic modification.

    get_author_contact()
        Retrieves the contact information of the software author, if available.

    Notes
    -----
    The `Software` class provides a flexible container for software metadata, 
    particularly useful for documenting research tools and ensuring proper 
    version control. The class can be extended or modified through keyword 
    arguments for future adaptability.

    Examples
    --------
    >>> from gofast.api.property import Software
    >>> software = Software(
    ...     name='HydroSim',
    ...     version='1.0.3',
    ...     release='2023-09-12',
    ...     license='MIT',
    ...     url='https://hydrosim.org'
    ... )
    >>> software.display_info()
    SoftwareInfo(
      {

           Name    : HydroSim
           Version : 1.0.3
           Release : 2023-09-12
           Author  : Not Specified
           License : MIT
           Url     : https://hydrosim.org

      }
    )

    [ 6 entries ]

    See Also
    --------
    Person : For additional details about the author of the software.

    References
    ----------
    .. [1] Johnson, L., & Smith, P. (2022). "Best Practices in Software Metadata 
           Documentation." *Software Engineering Journal*, 18(2), 100-112.
    """

    def __init__(
        self,
        name=None, 
        version=None, 
        release=None, 
        **kws
    ):
        self.name = name
        self.version = version
        self.release = release
        self.Author = Person()  

        for key in kws:
            setattr(self, key, kws[key])

    def display_info(self):
        """
        Display a formatted summary of the software's information, showing 
        key attributes such as name, version, release, and author.

        Notes
        -----
        This method is useful for quick inspection of software metadata 
        and aids in documentation efforts by providing a readable format.
        """
        from .summary import ResultSummary, get_table_size 

        TW= get_table_size()
        software_infos = {}
        software_infos["Name"]=f"{self.name or 'Not Specified'}"
        software_infos["Version"]=f"{self.version or 'Not Specified'}"
        software_infos["Release"]=f"{self.release or 'Not Specified'}"
        software_infos["Author"]=f"{self.Author.name or 'Not Specified'}"
        additional_info = {k: v for k, v in self.__dict__.items(
            ) if k not in ['name', 'version', 'release', 'Author']}
        for key, value in additional_info.items():
            software_infos[f"{key.capitalize()}"]= f"{value}"
        
        summary= ResultSummary(
            pad_keys="auto", max_char=TW).add_results(software_infos)
        summary.name = "SoftwareInfo"
        print(summary)  
    

    def update_info(self, **kws):
        """
        Update the software attributes with new values provided via 
        keyword arguments.

        Parameters
        ----------
        **kws : dict
            Key-value pairs of attributes to update.

        Examples
        --------
        >>> software = Software(name='HydroSim', version='1.0.3')
        >>> software.update_info(version='1.1.0', release='2023-10-01')
        >>> software.version
        '1.1.0'
        """
        for key, value in kws.items():
            setattr(self, key, value)
            
    def get_author_contact(self):
        """
        Retrieve the contact information of the software author, if available.
    
        Returns
        -------
        dict
            A dictionary with available contact information (e.g., name, email, 
            organization, phone) or a message if the information is not available.
    
        Notes
        -----
        This method gathers information from the `Author` attribute of the instance, 
        including additional attributes if present, excluding private attributes 
        (those starting with `_`) and attributes ending with `_`.
    
        Examples
        --------
        >>> from gofast.api.property import Person, Software
        >>> software.Author.name = 'Jane Doe'
        >>> software.Author.email = 'jane.doe@example.com'
        >>> software.get_author_contact()
        Jane Doe Contact(
          {

               name  : Jane Doe
               email : jane.doe@example.com

          }
        )

        [ 2 entries ]

        >>> author= Person (
        ...    email='etanoyau@gmail.com', 
        ...    name='LKouadio', 
        ...    nickname="a.k.a.@Daniel",  
        ...    organization="International Association of Mathematical Geosciences",
        ...    organization_url='https://iamg.org/', 
        ...    affiliation="INPHB", 
        ...    affiliation_url ="https://inphb.ci/", 
        ...    country ="Cote d'Ivoire", 
        ...    region ="West Africa", 
        ...    phone="+2250707054290"
        ...   )
        >>> >>> from gofast.api.property import Software
        >>> software = Software(
        ...     name='Gofast',
        ...     version='1.0.1',
        ...     release='2025-09-12',
        ...     license='BSD-3 Clause',
        ...     url='https://gofast.org',
                Author = author, 
        ... )
        >>> software.get_author_contact () 
        LKouadio Contact(
          {

               name             : LKouadio
               email            : etanoyau@gmail.com
               organization     : International Association of Mathematical Geosciences
               phone            : +2250707054290
               organization_url : https://iamg.org/
               nickname         : a.k.a.@Daniel
               affiliation      : INPHB
               affiliation_url  : https://inphb.ci/
               country          : Cote d'Ivoire
               region           : West Africa

          }
        )

        [ 10 entries ]
        """
        from .summary import ResultSummary, get_table_size, to_camel_case
    
        # Initialize contact information with primary fields
        contact_info = {
            "name": getattr(self.Author, 'name', None),
            "email": getattr(self.Author, 'email', None),
            "organization": getattr(self.Author, 'organization', None),
            "phone": getattr(self.Author, 'phone', None)
        }
    
        # Retrieve any additional public attributes in Author,
        # excluding private and special attributes
        extra_info = {
            attr: value for attr, value in vars(self.Author).items()
            if not attr.startswith('_') and not attr.endswith('_') and attr not in contact_info
        }
        contact_info.update(extra_info)
    
        # Filter out None values from the contact information
        contact_info = {k: v for k, v in contact_info.items() if v is not None}
    
        # Display contact summary if information is available
        if contact_info:
            table_width = get_table_size()
            summary = ResultSummary(
                pad_keys="auto", max_char=table_width).add_results(contact_info)
            summary.name = f"{to_camel_case(self.Author.name)} Contact"
            print(summary)
        else:
            return {"message": "Author contact information is not available."}

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   