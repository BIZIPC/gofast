# -*- coding: utf-8 -*-
"""
`funcutils` is a utilities package providing various functionalities for 
functional programming tasks.
"""
import sys 
import time
import functools
import inspect
import logging
import warnings
import subprocess
import threading
from datetime import datetime
 
import numpy as np
import pandas as pd

from .._gofastlog import gofastlog 
from ..api.types import _T, Dict, Any, Callable, List, Type 
from ..api.types import  Optional, Tuple , Union  
from ..api.types import Series, DataFrame, ArrayLike, Array1D, LambdaType
from ._dependency import import_optional_dependency
from .coreutils import to_numeric_dtypes, is_iterable
from .coreutils import get_installation_name, is_module_installed 

# Configure  logging
_logger=gofastlog.get_gofast_logger(__name__)

__all__=[
    'apply_transform',
    'batch_processor',
    'cast_numeric',
    'compose',
    'conditional_apply',
    'conditional_decorator',
    'context_checker',
    'convert_and_format_data',
    'convert_to_pandas',
    'curry',
    'drop_nan_if',
    'ensure_pkg',
    'flatten_data_if',
    'flatten_list',
    'install_package',
    'is_valid_if',
    'make_data_dynamic',
    'memoize',
    'merge_dicts',
    'preserve_input_type',
    'retry_operation',
    'series_naming',
    'to_pandas',
    'update_dataframe_index',
    'update_index',
    'update_series_index',
    'validate_years'
    ]

def curry(check_types=False, strict=False, allow_extra_args=False):
    """
    Decorator for currying a function, with options for type checking, 
    strict argument completion, and allowing extra arguments.

    Parameters
    ----------
    check_types : bool, optional
        If True, enables type checking against the annotated types of the 
        curried function's arguments. Defaults to False.
    strict : bool, optional
        If True, requires all arguments to be provided for the curried 
        function to execute. Defaults to False.
    allow_extra_args : bool, optional
        If True, the curried function accepts extra arguments beyond its 
        defined parameters. Defaults to False.

    Returns
    -------
    callable
        A curried version of the original function.

    Examples
    --------
    >>> from gofast.tools.funcutils import curry

    >>> @curry(check_types=True)
    >>> def add(x: int, y: int) -> int:
    ...     return x + y

    >>> add_five = add(5)
    >>> print(add_five(3))
    8

    >>> @curry(strict=True)
    >>> def greet(greeting, name):
    ...     return f"{greeting}, {name}!"

    >>> greet_hello = greet("Hello")
    >>> print(greet_hello("World"))
    Hello, World!
    """
    def decorator(func):
        @functools.wraps(func)
        def curried(*args, **kwargs):
            if _should_check_types(check_types):
                _check_arg_types(func, args)
            if _is_complete(func, args, kwargs, strict, allow_extra_args):
                return func(*args, **kwargs)
            else:
                return functools.partial(curried, *args, **kwargs)
        return curried
    return decorator

def _should_check_types(check_types):
    """Determine if type checking is enabled."""
    return check_types

def _check_arg_types(func, args):
    """Check argument types against function annotations."""
    for a, v in zip(inspect.signature(func).parameters.values(), args):
        if a.annotation is not inspect.Parameter.empty and not isinstance(v, a.annotation):
            raise TypeError(f"Argument {a.name} must be of type {a.annotation.__name__}")

def _is_complete(func, args, kwargs, strict, allow_extra_args):
    """Check if the argument list completes the function signature."""
    arg_count = len(args) + len(kwargs)
    required_arg_count = func.__code__.co_argcount
    if allow_extra_args:
        return arg_count >= required_arg_count
    if strict:
        return arg_count == required_arg_count
    return arg_count >= required_arg_count and not strict

def compose(*functions, reverse_order=True, type_check=False):
    """
    A hybrid function that can be used to compose multiple functions together or 
    as a decorator to enhance a single function. 
    
    When used to compose multiple functions, it returns a new function that is
    the composition of those functions. When used as a decorator, it enhances 
    the decorated function based on the provided parameters.

    Parameters
    ----------
    *functions : callable
        The functions to be composed. When used as a decorator, this contains 
        only the decorated function.
    reverse_order : bool, optional
        Controls the order of function application when composing multiple
        functions. 
        Ignored when used as a decorator. Defaults to True.
    type_check : bool, optional
        Enables type checking based on annotations in the composed functions. 
        Defaults to False.

    Returns
    -------
    callable
        When composing functions, returns a new function representing the 
        composition.  When used as a decorator, returns the enhanced decorated
        function.

    Examples
    --------
    
    # Used as a decorator
    >>> from gofast.tools.funcutils import compose 
    >>> @compose(type_check=True)
    ... def add_one(x: int) -> int:
    ...     return x + 1
    >>> print(add_one(4))
    5

    # Used to compose functions
    >>> def double(x):
    ...     return x * 2
    >>> increment_and_double = compose(lambda x: x + 1, double)
    >>> print(increment_and_double(3))
    8
    """
    if len(functions) == 1 and callable(functions[0]):
        return _enhance_function(functions[0], type_check=type_check)
    else:
        if reverse_order:
            functions = reversed(functions)
        def composed_function(*args, **kwargs):
            # Start with the initial argument
            result = args
            for func in functions:
                if type_check:
                    _check_comp_arg_types(func, *result)
                # Pass the result of the last function as the argument to the next
                result = (func(*result),) if isinstance(result, tuple) else func(result)
            # Return the final result after all functions have been applied
            return result[0] if isinstance(result, tuple) else result

        return composed_function

def _enhance_function(func, type_check=False):
    """Enhances a single function with optional type checking."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if type_check:
            _check_comp_arg_types(func, *args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def _compose_functions(*functions, reverse_order, type_check):
    """Composes multiple functions with optional type checking and error handling."""
    def composed(*args, **kwargs):
        result = args[0] if args else kwargs.get('result', None)
        funcs = functions if reverse_order else reversed(functions)
        for func in funcs:
            if type_check:
                _check_comp_arg_types(func, result)
            result = func(result)
        return result
    return composed

def _check_comp_arg_types(func, *args, **kwargs):
    """Check argument types against function annotations."""
    sig = inspect.signature(func)
    bound_values = sig.bind(*args, **kwargs).arguments
    for name, value in bound_values.items():
        expected_type = sig.parameters[name].annotation
        if expected_type is not inspect.Parameter.empty and not isinstance(
                value, expected_type):
            raise TypeError(
                f"Argument {name} must be of type {expected_type.__name__},"
                f" got {type(value).__name__}")

def memoize(
    func=None, *, 
    cache_limit=None, 
    eviction_policy='LRU', 
    thread_safe=False
    ):
    """
    A hybrid decorator for memoizing a function with options for cache size limit, 
    eviction policy, and thread safety.
    
    Functions caches results of function calls, optimizing performance for 
    expensive operations when called with repeated arguments.

    Parameters
    ----------
    func : callable, optional
        The function to be memoized. If not provided, returns a decorator that 
        takes a function and memoizes it.
    cache_limit : int, optional
        The maximum number of results to cache. If None (default), caches all results.
    eviction_policy : str, optional
        The cache eviction policy to use when the cache is full. Supported policies 
        are 'LRU' (Least Recently Used, default) and 'FIFO' (First In, First Out).
    thread_safe : bool, optional
        If True, makes the memoization thread-safe using a lock. Defaults to False.

    Returns
    -------
    callable
        The memoized function.

    Examples
    --------
    >>> from gofast.tools.funcutils import memoize

    >>> @memoize(cache_limit=100, eviction_policy='LRU', thread_safe=True)
    >>> def fibonacci(n):
    ...      if n < 2:
    ...          return n
    ...   return fibonacci(n - 1) + fibonacci(n - 2)

    >>> print(fibonacci(10))
    55
    """
    def decorator(func):
        memo = {}
        cache_keys = []
        lock = threading.Lock() if thread_safe else None

        @functools.wraps(func)
        def memoized(*args, **kwargs):
            key = args + tuple(kwargs.items())
            if lock:
                with lock:
                    result = check_cache(key,  *args, **kwargs)
            else:
                result = check_cache(key, *args, **kwargs)
            return result

        def check_cache(key, *args, **kwargs):
            if key in memo:
                if eviction_policy == 'LRU':
                    cache_keys.append(cache_keys.pop(cache_keys.index(key)))
                return memo[key]
            result = func(*args, **kwargs)
            handle_eviction(key, result)
            return result

        def handle_eviction(key, result):
            if cache_limit is not None and len(cache_keys) >= cache_limit:
                oldest_key = _apply_eviction_policy(eviction_policy, cache_keys)
                memo.pop(oldest_key, None)
            memo[key] = result
            cache_keys.append(key)

        return memoized

    if func:
        return decorator(func)
    return decorator


def _apply_eviction_policy(eviction_policy, cache_keys):
    """ Added support for customizable eviction policies (LRU and FIFO). 
    The helper function  manages which cache entry to evict based on the
    selected policy."""
    if eviction_policy == 'LRU':
        return cache_keys.pop(0)  # Remove the least recently used item
    elif eviction_policy == 'FIFO':
        return cache_keys.pop(0)  # Remove the first inserted item
    else:
        raise ValueError(
            "Unsupported eviction policy. Expected 'LRU' or 'FIFO'.")

def merge_dicts(
    *dicts: Dict[Any, Any], deep_merge: bool = False,
     list_merge: Union[bool, Callable] = False) -> Dict[Any, Any]:
    """
    Merges multiple dictionaries into a single dictionary. Allows for deep
    merging and custom handling of list concatenation.

    Parameters
    ----------
    *dicts : Dict[Any, Any]
        Variable number of dictionary objects to be merged.
    deep_merge : bool, optional
        Enables deep merging of nested dictionaries. Defaults to False.
    list_merge : bool or Callable, optional
        Determines how list values are handled during merge. If True, lists
        are concatenated. If a Callable is provided, it is used to merge lists.
        Defaults to False, which replaces lists from earlier dicts with those
        from later dicts.

    Returns
    -------
    Dict[Any, Any]
        The resulting dictionary from merging all input dictionaries.

    Examples
    --------
    >>> from gofast.tools.funcutils import merge_dicts
    >>> dict_a = {'a': 1, 'b': [2], 'c': {'d': 4}}
    >>> dict_b = {'b': [3], 'c': {'e': 5}}
    >>> print(merge_dicts(dict_a, dict_b))
    {'a': 1, 'b': [3], 'c': {'e': 5}}

    Deep merge with list concatenation:

    >>> print(merge_dicts(dict_a, dict_b, deep_merge=True, list_merge=True))
    {'a': 1, 'b': [2, 3], 'c': {'d': 4, 'e': 5}}

    Deep merge with custom list merge function (taking the max of each list):

    >>> print(merge_dicts(dict_a, dict_b, deep_merge=True,
    ... list_merge=lambda x, y: [max(x + y)]))
    {'a': 1, 'b': [3], 'c': {'d': 4, 'e': 5}}
    """
    def deep_merge_dicts(target, source):
        for key, value in source.items():
            if deep_merge and isinstance(value, dict) and key in target and isinstance(
                    target[key], dict):
                deep_merge_dicts(target[key], value)
            elif isinstance(value, list) and key in target and isinstance(
                    target[key], list):
                if list_merge is True:
                    target[key].extend(value)  # Changed from += to extend for clarity
                elif callable(list_merge):
                    target[key] = list_merge(target[key], value)
                else:
                    target[key] = value
            else:
                target[key] = value
    
    result = {}
    for dictionary in dicts:
        if deep_merge:
            deep_merge_dicts(result, dictionary)
        else:
            for key, value in dictionary.items():
                if key in result and isinstance(result[key], list) and isinstance(
                        value, list):
                    if list_merge is True:
                        result[key].extend(value)
                    elif callable(list_merge):
                        result[key] = list_merge(result[key], value)
                    else:
                        result[key] = value
                else:
                    result.update({key: value})
    return result

def retry_operation(
    func: Callable, 
    retries: int = 3, 
    delay: float = 1.0, 
    catch_exceptions: tuple = (Exception,), 
    backoff_factor: float = 1.0, 
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    pre_process_args: Optional[Callable[[int, Tuple[Any, ...], dict],
                                        Tuple[Tuple[Any, ...], dict]]] = None
):
    """
    Retries a specified function upon failure, for a defined number of retries
    and with a delay between attempts. Enhancements include the ability to 
    pre-process function arguments before each retry attempt.

    Parameters
    ----------
    func : Callable
        The function to be executed and possibly retried upon failure.
    retries : int, optional
        The number of retries to attempt upon failure. Default is 3.
    delay : float, optional
        The initial delay between retries in seconds. Default is 1.0.
    catch_exceptions : tuple, optional
        A tuple of exception classes to catch and retry. Default is (Exception,).
    backoff_factor : float, optional
        The factor by which the delay increases after each retry. Default is 1.0.
    on_retry : Optional[Callable[[int, Exception], None]], optional
        A callback function that is called after a failed attempt.
    pre_process_args : Optional[Callable[
        [int, Tuple[Any, ...], dict], Tuple[Tuple[Any, ...], dict]]], optional
        A function to pre-process the arguments to `func` before each retry.
        It receives the current attempt number, the arguments, and keyword
        arguments to `func` and returns a tuple of processed arguments and
        keyword arguments.

    Returns
    -------
    Any
        The return value of the function if successful.

    Raises
    ------
    Exception
        The exception from the last attempt if all retries fail.

    Examples
    --------
    >>> from gofast.tools.funcutils import retry_operation
    >>> def test_func(x):
    ...     print(f"Trying with x={x}...")
    ...     raise ValueError("Fail")
    >>> def pre_process(attempt, args, kwargs):
    ...     # Increase argument x by 1 on each retry
    ...     return ((args[0] + 1,), kwargs)
    >>> try:
    ...     retry_operation(test_func, retries=2, delay=0.5,
    ...                        pre_process_args=pre_process,args=(1,))
    ... except ValueError as e:
    ...     print(e)
    """
    args, kwargs = (), {}
    for attempt in range(1, retries + 1):
        try:
            if pre_process_args:
                args, kwargs = pre_process_args(attempt, args, kwargs)
            return func(*args, **kwargs)
        except catch_exceptions as e:
            if attempt < retries:
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise e

def flatten_list(
    nested_list: List[Any], 
    depth: int = -1, 
    process_item: Optional[Callable[[Any], Any]] = None
    ) -> List[Any]:
    """
    Flattens a nested list into a single list of values, with optional depth 
    and item processing parameters.

    Parameters
    ----------
    nested_list : List[Any]
        The list to flatten, which may contain nested lists of any depth.
    depth : int, optional
        The maximum depth of list nesting to flatten. A depth of -1 (default)
        means fully flatten, a depth of 0 means no flattening, a depth of 1
        means flatten one level of nesting, and so on.
    process_item : Optional[Callable[[Any], Any]], optional
        A callable that processes each item during flattening. It takes a single
        item and returns the processed item. Default is None, which means no processing.

    Returns
    -------
    List[Any]
        A single, flat list containing all values from the nested list to the
        specified depth, with each item processed if a callable is provided.

    Examples
    --------
    >>> from gofast.tools.funcutils import flatten_list
    >>> nested = [1, [2, 3], [4, [5, 6]], 7]
    >>> flat = flatten_list(nested, depth=1)
    >>> print(flat)
    [1, 2, 3, 4, [5, 6], 7]

    With item processing to square numbers:
    >>> flat_processed = flatten_list(nested, process_item=(lambda x: x**2 if
    ...                                  isinstance(x, int) else x))
    >>> print(flat_processed)
    [1, 4, 9, 16, [25, 36], 49]
    """
    def flatten(current_list: List[Any], current_depth: int) -> List[Any]:
        result = []
        for item in current_list:
            if isinstance(item, list) and current_depth != 0:
                # Decrement depth unless it's infinite (-1)
                new_depth = current_depth - 1 if current_depth > 0 else -1
                result.extend(flatten(item, new_depth))
            else:
                # Apply processing if available
                processed_item = process_item(item) if process_item else item
                result.append(processed_item)
        return result
    
    return flatten(nested_list, depth)

def timeit_decorator(
    logger: Optional[logging.Logger] = None, 
    level: int = logging.INFO
    ):
    """
    A decorator that measures the execution time of a function and optionally
    logs it.

    Parameters
    ----------
    logger : Optional[logging.Logger], optional
        A logger object to log the execution time message. If None (default),
        the message is printed using the print function.
    level : int, optional
        The logging level at which to log the execution time message. Default
        is logging.INFO.

    Returns
    -------
    Callable
        A wrapped version of the input function that, when called, logs or prints
        the execution time.

    Examples
    --------
    >>> from gofast.tools.funcutils import timeit_decorator
    >>> import logging
    >>> logger = logging.getLogger('MyLogger')
    >>> @timeit_decorator(logger=logger)
    ... def example_function(delay: float):
    ...     '''Example function that sleeps for a given delay.'''
    ...     time.sleep(delay)
    >>> example_function(1)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            message = f"'{func.__name__}' executed in {end_time - start_time:.2f}s"
            if logger:
                logger.log(level, message)
            else:
                print(message)
            return result
        return wrapper
    return decorator

def conditional_decorator(
    predicate: Callable[[Callable], bool], 
    decorator: Callable[[Callable], Callable], 
    else_decorator: Optional[Callable[[Callable], Callable]] = None
    ) -> Callable:
    """
    Applies a decorator to a function conditionally based on a predicate. 
    
    Optionally, applies an alternative decorator if the condition is not met.

    Parameters
    ----------
    predicate : Callable[[Callable], bool]
        A function that takes a function as input and returns True if the primary
        decorator should be applied, False otherwise.
    decorator : Callable[[Callable], Callable]
        The primary decorator to apply if the predicate returns True.
    else_decorator : Optional[Callable[[Callable], Callable]], optional
        An alternative decorator to apply if the predicate returns False. If None,
        the function is returned unchanged in case of False. Default is None.

    Returns
    -------
    Callable
        A new decorator that conditionally applies the given decorators based on
        the predicate.

    Examples
    --------
    >>> from gofast.tools.funcutils import conditional_decorator
    >>> def my_decorator(func):
    ...     def wrapper(*args, **kwargs):
    ...         print("Decorated")
    ...         return func(*args, **kwargs)
    ...     return wrapper
    >>> def is_even(func):
    ...     return func(2) % 2 == 0
    >>> @conditional_decorator(predicate=is_even, decorator=my_decorator)
    ... def add_one(x):
    ...     return x + 1
    >>> add_one(2)
    'Decorated'
    3
    """
    def new_decorator(func: Callable) -> Callable:
        if predicate(func):
            return decorator(func)
        elif else_decorator:
            return else_decorator(func)
        return func
    return new_decorator

def batch_processor(
    func: Callable, 
    on_error: Optional[Callable[[Exception, Any], Any]] = None,
    on_success: Optional[Callable[[Any, Any], None]] = None) -> Callable:
    """
    Process a batch of inputs, with optional error handling and success 
    callbacks.

    Parameters
    ----------
    func : Callable
        The original function to be applied to each element of the input batch.
    on_error : Optional[Callable[[Exception, Any], Any]], optional
        A callback function to be called if an exception occurs while processing
        an element. It should take two parameters: the exception and the input
        that caused it, and return a value to be included in the results list.
        If None (default), exceptions are raised immediately, stopping the batch.
    on_success : Optional[Callable[[Any, Any], None]], optional
        A callback function to be called after successfully processing an element.
        It should take two parameters: the result of the function call and the
        input that led to it. This can be used for logging or further processing.

    Returns
    -------
    Callable
        A function that takes a list of inputs and returns a list of results,
        applying `func` to each input element, handling exceptions and successes
        as specified.

    Examples
    --------
    >>> from gofast.tools.funcutils import batch_processor
    >>> def safe_divide(x):
    ...     return 10 / x
    >>> def handle_error(e, x):
    ...     print(f"Error {e} occurred with input {x}")
    ...     return None  # Default value in case of error
    >>> batch_divide = batch_processor(safe_divide, on_error=handle_error)
    >>> print(batch_divide([2, 1, 0]))  # 0 will cause a division by zero error
    Error division by zero occurred with input 0
    [5.0, 10.0, None]
    """
    def process_batch(inputs: List[Any]) -> List[Any]:
        results = []
        for input in inputs:
            try:
                result = func(input)
                if on_success:
                    on_success(result, input)
                results.append(result)
            except Exception as e:
                if on_error:
                    error_result = on_error(e, input)
                    results.append(error_result)
                else:
                    raise e
        return results
    return process_batch

def is_valid_if(
    *expected_types: Tuple[type],
    kwarg_types: Optional[Dict[str, type]] = None,
    custom_error: Optional[str] = None,
    skip_check: Optional[Callable[[Any, Any], bool]] = None
):
    """
    A decorator to verify the datatype of positional and keyword parameters 
    of a function.
    
    Allows specifying expected types for both positional and keyword arguments.
    A custom error message and a condition to skip checks can also be provided.

    Parameters
    ----------
    *expected_types : Tuple[type]
        The expected types of the positional parameters.
    kwarg_types : Optional[Dict[str, type]], optional
        A dictionary mapping keyword argument names to their expected types.
    custom_error : Optional[str], optional
        A custom error message to raise in case of a type mismatch. Should
        contain placeholders for positional formatting: {arg_index}, {func_name},
        {expected_type}, and {got_type}.
    skip_check : Optional[Callable[[Any, Any], bool]], optional
        A function that takes the argument list and keyword argument dictionary,
        and returns True if type checks should be skipped.

    Returns
    -------
    Callable
        The decorated function with type verification.

    Raises
    ------
    TypeError
        If the types of the actual parameters do not match the expected types
        and skip_check is False or not provided.

    Examples
    --------
    >>> from gofast.tools.funcutils import is_valid_if
    >>> @is_valid_if(int, float, kwarg_types={'c': str})
    ... def add(a, b, c="default"):
    ...     return a + b
    >>> add(1, 2.5, c="text")
    3.5
    >>> add(1, "2.5", c=100)  # This will raise a TypeError
    TypeError: Argument 2 of 'add' requires <class 'float'> but got <class 'str'>.
    """
    def _construct_error_msg(arg_name, func_name, expected_type, got_type):
        """Constructs a custom or default error message based on provided parameters."""
        if custom_error:
            return custom_error.format(arg_name=arg_name, func_name=func_name,
                                       expected_type=expected_type, got_type=got_type)
        else:
            return (f"Argument '{arg_name}' of '{func_name}' requires {expected_type} "
                    f"but got {got_type}.")

    def _check_arg_types(args, kwargs, func_name):
        """Checks types of positional and keyword arguments."""
        for i, (arg, expected_type) in enumerate(zip(args, expected_types), 1):
            if not isinstance(arg, expected_type):
                error_msg = _construct_error_msg(arg_name=i, func_name=func_name,
                                                 expected_type=expected_type,
                                                 got_type=type(arg))
                raise TypeError(error_msg)

        if kwarg_types:
            for kwarg, expected_type in kwarg_types.items():
                if kwarg in kwargs and not isinstance(kwargs[kwarg], expected_type):
                    error_msg = _construct_error_msg(arg_name=kwarg, func_name=func_name,
                                                     expected_type=expected_type, 
                                                     got_type=type(kwargs[kwarg]))
                    raise TypeError(error_msg)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if skip_check and skip_check(args, kwargs):
                return func(*args, **kwargs)

            _check_arg_types(args, kwargs, func.__name__)

            return func(*args, **kwargs)
        return wrapper
    return decorator

def install_package(
    name: str, 
    dist_name: Optional[str]=None,
    infer_dist_name: bool=False, 
    extra: str = '', 
    use_conda: bool = False, 
    verbose: bool = True
    ) -> None:
    """
    Install a Python package using either conda or pip, with an option to 
    display installation progress and fallback mechanism.

    This function dynamically chooses between conda and pip for installing 
    Python packages, based on user preference and system configuration. It 
    supports a verbose mode for detailed operation logging and utilizes a 
    progress bar for pip installations.

    Parameters
    ----------
    name : str
        Name of the package to install. Version specification can be included.
    dist_name : str, optional
        The distribution name of the package. Useful for packages where
        the import name differs from the distribution name.
    infer_dist_name : bool, optional
        If True, attempt to infer the distribution name for pip installation,
        defaults to False.
    extra : str, optional
        Additional options or version specifier for the package, by default ''.
    use_conda : bool, optional
        Prefer conda over pip for installation, by default False.
    verbose : bool, optional
        Enable detailed output during the installation process, by default True.

    Raises
    ------
    RuntimeError
        If installation fails via both conda and pip, or if the specified installer
        is not available.

    Examples
    --------
    Install a package using pip without version specification:

        >>> install_package('requests', verbose=True)

    Install a specific version of a package using conda:

        >>> install_package('pandas', extra='==1.2.0', use_conda=True, verbose=True)
    
    Notes
    -----
    Conda installations do not display a progress bar due to limitations in capturing
    conda command line output. Pip installations will show a progress bar indicating
    the number of processed output lines from the installation command.
    """
    def execute_command(command: list, progress_bar: bool = False) -> None:
        """
        Execute a system command with optional progress bar for output lines.

        Parameters
        ----------
        command : list
            Command and arguments to execute as a list.
        progress_bar : bool, optional
            Enable a progress bar that tracks the command's output lines, 
            by default False.
        """
        from tqdm import tqdm
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1) as process, \
             tqdm(desc="Installing", unit="line", disable=not progress_bar) as pbar:
            for line in process.stdout:
                if verbose:
                    print(line, end='')
                pbar.update(1)
            if process.wait() != 0:  # Non-zero exit code indicates failure
                raise RuntimeError(f"Installation failed for package '{name}{extra}'.")
    
    # If the module is installed don't install again.
    if is_module_installed(name, distribution_name= dist_name ): 
        if verbose:
           print(f"{name} is already installed.")
           
        return True
    # If the distribution to pkg name if the pkg name 
    # is different to distribution name .
    if infer_dist_name: 
        name = get_installation_name(name, dist_name)  
        
    conda_available = _check_conda_installed()
    try:
        if use_conda and conda_available:
            if verbose:
                print(f"Attempting to install '{name}{extra}' using conda...")
            execute_command(['conda', 'install', f"{name}{extra}", '-y'], 
                            progress_bar=False)
        elif use_conda and not conda_available:
            if verbose:
                print("Conda is not available. Falling back to pip...")
            execute_command([sys.executable, "-m", "pip", "install", f"{name}{extra}"],
                            progress_bar=True)
        else:
            if verbose:
                print(f"Attempting to install '{name}{extra}' using pip...")
            execute_command([sys.executable, "-m", "pip", "install", f"{name}{extra}"],
                            progress_bar=True)
        if verbose:
            print(f"Package '{name}{extra}' was successfully installed.")
    except Exception as e:
        raise RuntimeError(f"Failed to install '{name}{extra}': {e}") from e

def _check_conda_installed() -> bool:
    """
    Check if conda is installed and available in the system's PATH.

    Returns
    -------
    bool
        True if conda is found, False otherwise.
    """
    try:
        subprocess.check_call(['conda', '--version'], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def ensure_pkg(
    name: str, 
    extra: str = "",
    errors: str = "raise",
    min_version: str | None = None,
    exception: Exception = None, 
    dist_name: Optional[str]=None, 
    infer_dist_name: bool=False, 
    auto_install: bool = False,
    use_conda: bool = False, 
    partial_check: bool = False,
    condition: Any = None, 
    verbose: bool = False
) -> Callable[[_T], _T]:
    """
    Decorator to ensure a Python package is installed before function execution.

    If the specified package is not installed, or if its installed version does
    not meet the minimum version requirement, this decorator can optionally 
    install or upgrade the package automatically using either pip or conda.

    Parameters
    ----------
    name : str
        The name of the package.
    extra : str, optional
        Additional specification for the package, such as version or extras.
    errors : str, optional
        Error handling strategy if the package is missing: 'raise', 'ignore',
        or 'warn'.
    min_version : str or None, optional
        The minimum required version of the package. If not met, triggers 
        installation.
    exception : Exception, optional
        A custom exception to raise if the package is missing and `errors`
        is 'raise'.
    dist_name : str, optional
        The distribution name of the package as known by package managers (e.g., pip).
        If provided and the module import fails, an additional check based on the
        distribution name is performed. This parameter is useful for packages where
        the distribution name differs from the importable module name.
    infer_dist_name : bool, optional
        If True, attempt to infer the distribution name for pip installation,
        defaults to False.
    auto_install : bool, optional
        Whether to automatically install the package if missing. 
        Defaults to False.
    use_conda : bool, optional
        Prefer conda over pip for automatic installation. Defaults to False.
    partial_check : bool, optional
        If True, checks the existence of the package only if the `condition` 
        is met. This allows for conditional package checking based on the 
        function's arguments or other criteria. If `False`, the check is always
        performed. Defaults to False.
    condition : Any, optional
        A condition that determines whether to check for the package's existence. 
        This can be a callable that takes the same arguments as the decorated function 
        and returns a boolean, a specific argument name to check for truthiness, or 
        any other value that will be evaluated as a boolean. If `None`, the package 
        check is performed unconditionally unless `partial_check` is False.
    verbose : bool, optional
        Enable verbose output during the installation process. Defaults to False.

    Returns
    -------
    Callable
        A decorator that wraps functions to ensure the specified package 
        is installed.

    Examples
    --------
    >>> from gofast.tools.funcutils import ensure_pkg
    >>> @ensure_pkg("numpy", auto_install=True)
    ... def use_numpy():
    ...     import numpy as np
    ...     return np.array([1, 2, 3])

    >>> @ensure_pkg("pandas", min_version="1.1.0", errors="warn", use_conda=True)
    ... def use_pandas():
    ...     import pandas as pd
    ...     return pd.DataFrame([[1, 2], [3, 4]])

    >>> @ensure_pkg("matplotlib", partial_check=True, condition=lambda x, y: x > 0)
    ... def plot_data(x, y):
    ...     import matplotlib.pyplot as plt
    ...     plt.plot(x, y)
    ...     plt.show()
    
    >>> @ensure_pkg("skimage", partial_check=True, condition=(
    ...     lambda *args, **kwargs: 'method' in kwargs and kwargs['method'] == 'hog')
    ...     )
    >>> def check_package_installed(data, method='hog', **kwargs):
    ...     extractor_function = None
    ...     if method == 'hog':
    ...         from skimage.feature import hog
    ...         extractor_function = lambda image: hog(image, **kwargs)
    ...     return extractor_function
    """
    def decorator(func: _T) -> _T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if this is a method or a function based on the first argument
            bound_method = hasattr(args[0], func.__name__) if args else False # 
            
            # If partial_check is True, check condition before performing actions
            if not partial_check or  _should_check_condition(
                    condition, *args, **kwargs):
                try:
                    # Attempt to import the package, handling installation 
                    # if necessary and permitted
                    import_optional_dependency(
                        name, extra=extra, errors=errors, 
                        min_version=min_version, exception=exception
                    )
                except (ModuleNotFoundError, ImportError):
                    if auto_install:
                        # Install the package if auto-install is enabled
                        install_package(
                            name, dist_name=dist_name, 
                            infer_dist_name=infer_dist_name, 
                            extra=extra, use_conda=use_conda, verbose=verbose
                        )
                    elif exception is not None:
                        raise exception
                    else:
                        raise
                    
            # If the function is a bound method, call it with 'self' or 'cls'
            if bound_method:
                return func(args[0], *args[1:], **kwargs)
            else:
                return func(*args, **kwargs) # 
        
        return wrapper
    
    return decorator

def _should_check_condition(condition: Any, *args, **kwargs) -> bool:
    """
    Determines whether the condition(s) for checking a package's existence are met, 
    based on the provided arguments and keyword arguments of a decorated function.

    This function offers enhanced flexibility by allowing conditions to be specified 
    as callable functions, tuples for positional argument checks, strings for keyword 
    argument checks, or a list combining any of these types for multiple conditions.

    Parameters
    ----------
    condition : Any
        The condition(s) that determine whether to perform the package check. Can be:
        - A callable that takes `*args` and `**kwargs` and returns a boolean.
        - A string specifying a keyword argument name that should be truthy.
        - A tuple `(index, value)` for checking a specific value of a positional argument.
        - A list of any combination of the above to specify multiple conditions.
    *args : tuple
        Positional arguments passed to the decorated function.
    **kwargs : dict
        Keyword arguments passed to the decorated function.

    Returns
    -------
    bool
        `True` if the package check should be performed based on the evaluation of 
        `condition`, `False` otherwise.

    Examples
    --------
    Checking with a single callable condition for partial_check is ``True``:

    >>> _should_check_condition(lambda x, y: x > y, 5, 3)
    True

    Checking with a string condition (keyword argument name):

    >>> _should_check_condition('method', method='hog')
    True

    Checking with a tuple for positional argument value:

    >>> _should_check_condition((0, 'data'), 'data', method='hog')
    True

    Checking with multiple conditions:

    >>> conditions = [(1, 'hog'), lambda *args, **kwargs: kwargs.get('filter', False)]
    >>> _should_check_condition( conditions, 'data', 'hog', filter=True)
    True

    In the last example, the package check is performed because both conditions are met:
    the second positional argument equals 'hog', and the 'filter' keyword argument is `True`.
    """

    def eval_condition(cond):
        # Callable condition with direct application
        if callable(cond):
            return cond(*args, **kwargs)
        # String condition indicating a key in kwargs
        elif isinstance(cond, str) and cond in kwargs:
            return bool(kwargs[cond])
        # Tuple condition indicating positional argument check
        elif isinstance(cond, tuple) and len(cond) == 2:
            index, value = cond
            return index < len(args) and args[index] == value
        return False
    
    # Support for list of conditions: all must be True
    if isinstance(condition, list):
        return all(eval_condition(cond) for cond in condition)
    else:
        return eval_condition(condition)

def drop_nan_if(thresh: float, meth: str = 'drop_cols'):
    """
    A decorator that preprocesses the first positional argument 'data' of the
    decorated function to ensure it is a DataFrame and then drops rows or columns 
    based on a missing value threshold. If the data is an array or Series, it is
    converted to a DataFrame. After dropping, the data is cast back to its original
    data types.

    Parameters
    ----------
    thresh : float
        The threshold for missing values. Rows or columns with missing values
        exceeding this threshold will be dropped.
    meth : str, optional
        The method to apply: 'drop_rows' for dropping rows, 'drop_cols' for
        dropping columns. Default is 'drop_cols'.

    Returns
    -------
    Callable
        The decorated function with data preprocessing.

    Examples
    --------
    >>> @drop_nan_if(thresh=1, meth='drop_rows')
    ... def process_data(data):
    ...     print(data.dtypes)
    >>> data = pd.DataFrame({
    ...     'A': [1, np.nan, 3],
    ...     'B': ['x', 'y', np.nan],
    ...     'C': pd.to_datetime(['2021-01-01', np.nan, '2021-01-03'])
    ... })
    >>> process_data(data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            data = args[0]
            original_dtypes = None
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            elif isinstance(data, pd.Series):
                data = data.to_frame()
            elif not isinstance(data, pd.DataFrame):
                raise TypeError("The first positional argument must be a "
                                "pandas DataFrame, a numpy array, or a pandas"
                                " Series.")
            else:
                # Store original data types for later restoration
                original_dtypes = data.dtypes
            
            if meth == 'drop_rows':
                processed_data = data.dropna(axis=0, thresh=len(data.columns) * thresh)
            elif meth == 'drop_cols':
                processed_data = data.dropna(axis=1, thresh=len(data) * thresh)
            else:
                raise ValueError("Method argument 'meth' must be either 'drop_rows' or 'drop_cols'.")

            # Restore original data types
            if original_dtypes is not None:
                for col, dtype in original_dtypes.items():
                    if col in processed_data.columns:
                        processed_data[col] = processed_data[col].astype(dtype)

            new_args = (processed_data,) + args[1:]
            return func(*new_args, **kwargs)
        return wrapper
    return decorator

def conditional_apply(
        predicate: Callable[[Any], bool], 
        default_value: Any = None
        ) -> Callable:
    """
    Decorator that conditionally applies the decorated function based 
    on a predicate.
    
    If the predicate returns False, a default value is returned instead.

    Parameters
    ----------
    predicate : Callable[[Any], bool]
        A function that takes the same arguments as the decorated function and
        returns True if the function should be applied, False otherwise.
    default_value : Any, optional
        The value to return if the predicate evaluates to False. Default is None.

    Returns
    -------
    Callable
        The decorated function with conditional application logic.

    Examples
    --------
    >>> from gofast.tools.funcutils import conditional_apply
    >>> @conditional_apply(predicate=lambda x: x > 0, default_value=0)
    ... def reciprocal(x):
    ...     return 1 / x
    >>> print(reciprocal(2))
    0.5
    >>> print(reciprocal(-1))
    0
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if predicate(*args, **kwargs):
                return func(*args, **kwargs)
            else:
                return default_value
        return wrapper
    return decorator


def apply_transform(
        transformations: List[Callable]) -> Callable:
    """
    Creates a callable that applies a sequence of transformation functions 
    to its input.

    Parameters
    ----------
    transformations : List[Callable]
        A list of functions that will be applied sequentially to the input data.

    Returns
    -------
    Callable
        A new function that takes input data and applies each transformation 
        in sequence.

    Examples
    --------
    >>> from gofast.tools.funcutils import apply_transform
    >>> def square(x):
    ...     return x ** 2
    >>> def increment(x):
    ...     return x + 1
    >>> transform = apply_transform([square, increment])
    >>> print(transform(4))
    17
    """
    def transformed_callable(data):
        for transform in transformations:
            data = transform(data)
        return data
    return transformed_callable

def make_data_dynamic(
    expected_type: str = 'numeric', 
    capture_columns: bool = False, 
    drop_na: bool = False, 
    na_thresh: Optional[float] = None, 
    na_meth: str = 'drop_rows', 
    reset_index: bool = False, 
    dynamize: bool=True, 
    force_df: bool=False, 
) -> Callable:
    """
    A decorator for preprocessing data before passing it to a function, 
    with options for data type filtering, column selection, missing value 
    handling, and index resetting.

    Parameters
    ----------
    expected_type : str, optional
        Specifies the type of data the function expects: 'numeric' for numeric 
        data, 'categorical' for categorical data, or 'both' for no filtering. 
        Defaults to 'numeric'.
    capture_columns : bool, optional
        If True, uses the 'columns' keyword argument from the decorated function to 
        filter the DataFrame columns. Defaults to False.
    drop_na : bool, optional
        If True, applies missing value handling according to `meth` and `thresh`. 
        Defaults to False.
    na_thresh : Optional[float], optional
        Threshold for dropping rows or columns based on the proportion of missing 
        values. Used only if `drop_na` is True. Defaults to None, meaning no threshold.
    na_meth : str, optional
        Method for dropping missing values: 'drop_rows' to drop rows with missing 
        values, 'drop_cols' to drop columns with missing values, or any other value 
        to drop all missing values without thresholding. Defaults to 'drop_rows'.
    reset_index : bool, optional
        If True, resets the DataFrame index (dropping the current index) before 
        passing it to the function. Defaults to False.
    dynamize: bool,
        If True, the preprocessing function is attached as a method to the 
        pandas DataFrame class, allowing it to be called directly on DataFrame
        instances. This enhances the integration and usability of the function
        within pandas workflows, making it more accessible as part of 
        DataFrame's method chain. Defaults to True.
    force_df: bool  
       Determines the output format for single-column data frames. 
       If set to `True`, the function will always return a DataFrame, even if 
       it consists of a single column. If set to `False`, single-column 
       DataFrames may be converted to a Series, providing a more streamlined 
       representation. This option allows for flexibility in handling the 
       output format, catering to different preferences or requirements 
       for subsequent data processing steps. Default is False.

       

    Examples
    --------
    >>> from gofast.tools.funcutils import make_data_dynamic 
    >>> @make_data_dynamic(expected_type='numeric', capture_columns=True, 
    ... drop_na=True, thresh=0.5, meth='drop_rows', reset_index=True)
    ... def calculate_mean(data: Union[pd.DataFrame, np.ndarray]):
    ...     return data.mean()
    
    >>> data = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 5, 6]})
    >>> print(calculate_mean(data))
    # This will calculate the mean after preprocessing the input data 
    # according to the specified rules.

    The decorated function seamlessly preprocesses input data, ensuring 
    that it meets the specified criteria for data type, column selection, 
    missing value handling, and index state.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                raise ValueError("Function requires at least one argument.")
            
            data = _check_and_convert_input(args[0])
            data = _preprocess_data(
                data, capture_columns, expected_type, drop_na, na_thresh,
                na_meth, reset_index, **kwargs)
            # Infer dataframe to series for single columns.
            data= to_pandas(data, convert_single_column= force_df)
            new_args = (data,) + args[1:]
            return func(*new_args, **kwargs)
        
        if dynamize: 
            _add_dynamic_method (wrapper )

        return wrapper
    return decorator

def _check_and_convert_input(input_data):
    """
    Check the type of the input data and convert it to a suitable 
    format for processing.
    
    Parameters
    ----------
    input_data : any
        The first argument passed to the decorated function, expected 
        to be either  a pd.DataFrame, dict, np.ndarray, or an iterable that can be 
        converted to an np.ndarray.
    
    Returns
    -------
    pd.DataFrame
        The input data converted to a pandas DataFrame for further processing.
        
    Raises
    ------
    ValueError
        If the input data is not a pd.DataFrame, dict, np.ndarray, or 
        convertible iterable.
    """
    if isinstance(input_data, (pd.DataFrame, np.ndarray)):
        return to_numeric_dtypes (pd.DataFrame(input_data))
    elif isinstance(input_data, dict):
        return to_numeric_dtypes (pd.DataFrame(input_data))
    elif hasattr(input_data, '__iter__'):  # Check if it's iterable
        try:
            return to_numeric_dtypes (pd.DataFrame(np.array(input_data))) 
        except Exception:
            raise TypeError(
                "Expect the first argument to be a non-string iterable object"
                 " with minimum samples equal to two.")
    else:
        raise ValueError("First argument must be a pd.DataFrame, dict,"
                         " np.ndarray, or an iterable object.")
        
def _add_dynamic_method(func):
    """
    Dynamically adds a given function as a method to pandas DataFrame and
    Series objects. Validates that `func` is callable and logs actions and
    errors, enhancing robustness and user feedback.

    Parameters
    ----------
    func : callable
        The function to be added as a method. The function's name
        (`func.__name__`) is used as the method name.

    Examples
    --------
    >>> from gofast.tools.funcutils import _add_dynamic_method
    >>> def example_method(self, multiplier=2):
    ...     return self * multiplier
    >>> _add_dynamic_method(example_method)
    >>> import pandas as pd
    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    >>> df.example_method(3)
       A   B
    0  3   6
    1  9  12
    >>> s = pd.Series([1, 2, 3])
    >>> s.example_method()
    0    2
    1    4
    2    6
    dtype: int64
    """
    if not callable(func):
        _logger.error(f"Provided object {func} is not callable and cannot "
                     "be added as a method.")
        return

    for pandas_class in [pd.DataFrame, pd.Series]:
        method_name = "go_" + func.__name__
        if hasattr(pandas_class, method_name):
            continue
        try:
            setattr(pandas_class, method_name, func)
        except Exception as error: #noqa
            pass

def _preprocess_data(
        data, capture_columns, expected_type, drop_na, na_thresh, 
        na_meth, reset_index, **kwargs):
    """
    Apply preprocessing steps to the input data based on the decorator's 
    parameters.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input data to be preprocessed.
    capture_columns : bool
        Whether to filter the DataFrame columns using the 'columns' 
        keyword argument.
    expected_type : str
        Specifies the expected data type: 'numeric', 'categorical', or 'both'.
    drop_na : bool
        Whether to apply missing value handling.
    na_thresh : Optional[float]
        Threshold for dropping rows or columns based on the proportion of 
        missing values.
    na_meth : str
        Method for dropping missing values.
    reset_index : bool
        Whether to reset the DataFrame index before passing it to the function.
    kwargs : dict
        Additional keyword arguments passed to the decorated function.
    
    Returns
    -------
    pd.DataFrame
        The preprocessed data.
    """
    if capture_columns and 'columns' in kwargs:
        columns = kwargs.pop('columns')
        columns = is_iterable(columns, exclude_string= True, transform=True )
        try:
            data = data[columns]
        except KeyError:
            print("Specified columns do not match, ignoring columns.")

    if expected_type == 'numeric':
        data = data.select_dtypes([np.number])
    elif expected_type == 'categorical':
        data = data.select_dtypes(None, [np.number])

    if drop_na:
        if na_meth == 'drop_rows':
            data = data.dropna(axis=0, thresh=(na_thresh * len(
                data.columns) if na_thresh is not None else None))
        elif na_meth == 'drop_cols':
            data = data.dropna(axis=1, thresh=(na_thresh * len(
                data) if na_thresh is not None else None))
        else:
            data = data.dropna()

    if reset_index:
        data = data.reset_index(drop=True)

    return data

def preserve_input_type(
    keep_columns_intact: bool = False,
    custom_convert: Optional[Callable[[Any, Type, Any], Any]] = None,
    fallback_on_error: bool = True,
    specific_type: Optional[Type] = None  
) -> Callable:
    """
    A decorator that preserves the input data type of the first positional
    argument of the decorated function. If the function's output type is
    different from its input, this decorator attempts to convert the output
    back to the input's original type.

    Parameters
    ----------
    keep_columns_intact : bool, optional
        When True and the input is a pandas DataFrame, attempts to preserve
        the original DataFrame's columns in the returned DataFrame. This is
        only applicable if the result can logically map to the original
        columns. Defaults to False.
    custom_convert : Optional[Callable[[Any, Type, Any], Any]], optional
        A custom conversion function provided by the user that takes three
        arguments: the result of the decorated function, the original type
        of the first positional argument, and the original columns (if the
        input was a pandas DataFrame). This function should return the
        converted result. If None, the decorator uses default conversion
        logic. Defaults to None.
    fallback_on_error : bool, optional
        If True, the decorator will return the unconverted result when a
        conversion attempt raises an error or when the custom_convert
        function fails. If False, the error is propagated. Defaults to True.
    
    specific_type : Type, optional
        Specifies the type to preserve. If None, the type of the first positional
        argument or a specified keyword argument is used. Defaults to None.
        
    Returns
    -------
    Callable
        A wrapped version of the original function that ensures the output
        type matches the input type of the first positional argument.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.tools.funcutils import preserve_input_type
    >>> @preserve_input_type()
    ... def add_one(data):
    ...     return data + 1
    ...
    >>> add_one(np.array([1, 2, 3]))
    array([2, 3, 4])
    >>> add_one(pd.Series([1, 2, 3]))
    0    2
    1    3
    2    4
    dtype: int64
    >>> add_one([1, 2, 3])
    [2, 3, 4]

    Using custom conversion logic:
    >>> @preserve_input_type(custom_convert=lambda res, orig_type, _: orig_type(
    ...                      [x * 2 for x in res]))
    ... def multiply_by_two(data):
    ...     return data * 2
    ...
    >>> multiply_by_two(np.array([1, 2, 3]))
    array([2, 4, 6])
    >>> multiply_by_two(pd.Series([1, 2, 3]))
    0    2
    1    4
    2    6
    dtype: int64
    >>> multiply_by_two([1, 2, 3])
    [2, 4, 6]

    Note
    ----
    This decorator is particularly useful for functions that may return a
    type different from their input type, but where maintaining the input
    type is desired for consistency in the calling code.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Determine the original input and type to preserve.
            original_input = specific_type or (
                args[0] if args else next(iter(kwargs.values()), None))
            
            if original_input is None:
                return func(*args, **kwargs)
            
            original_type = specific_type or type(original_input)
            original_columns = getattr(original_input, 'columns', None) if isinstance(
                original_input, pd.DataFrame) else None
            
            try:
                result = func(*args, **kwargs)
                if custom_convert:
                    # Use custom conversion logic if provided.
                    return custom_convert(result, original_type, original_columns)
                else:
                    # Convert to original type using default logic.
                    return _convert_to_original_type(
                        result, original_type, original_columns,
                        keep_columns_intact)
            except Exception as e:
                if fallback_on_error:
                    # Return unconverted result if conversion fails.
                    return result
                else:
                    raise e
        return wrapper
    
    return decorator

def _convert_to_original_type(
        result: Any, original_type: Type, original_columns: Any,
        keep_columns_intact: bool) -> Any:
    """
    Converts the result to the original input type, using specific conversion functions
    based on the original type.
    """
    if original_type is pd.DataFrame:
        return _convert_to_dataframe(result, original_columns, keep_columns_intact)
    elif original_type is pd.Series:
        return _convert_to_series(result, original_columns)
    elif original_type is np.ndarray:
        return _convert_to_ndarray(result)
    elif original_type is list:
        return _convert_to_list(result)
    return result

def _convert_to_dataframe(
        result: Any, original_columns: Any,
        keep_columns_intact: bool) -> pd.DataFrame:
    """
    Converts the result to a DataFrame, optionally preserving original columns.
    """
    if isinstance(result, (pd.Series, np.ndarray, list)):
        if keep_columns_intact and original_columns is not None:
            return pd.DataFrame([result], columns=original_columns)
        else:
            return pd.DataFrame(result)
    return result

def _convert_to_series(
        result: Any, original_columns: Any) -> pd.Series:
    """
    Converts the result to a Series, preserving original columns as index if applicable.
    """
    if isinstance(result, (np.ndarray, list)):
        return pd.Series(
            result, index=original_columns if original_columns is not None else None)
    return result

def _convert_to_ndarray(result: Any) -> np.ndarray:
    """
    Converts the result to a NumPy ndarray, suitable for array-like results.
    """
    if isinstance(result, (pd.Series, pd.DataFrame, list)):
        return np.array(result)
    return result

def _convert_to_list(result: Any) -> list:
    """
    Converts the result to a list, suitable for list-like results.
    """
    if isinstance(result, (pd.Series, pd.DataFrame, np.ndarray)):
        return result.tolist()
    return result

def to_pandas(
    data: Any,
    prefer: str = 'auto',
    convert_single_column: bool = False,
    transform: Callable[[Any], Any] = None
) -> Union[DataFrame, Series]:
    """
    Attempts to convert input data into a Pandas DataFrame or Series. 
    
    Intelligently handling various input types and applying an optional 
    transformation. Converts single-column DataFrames to Series if specified.

    Parameters
    ----------
    data : Any
        Input data to convert. Supports lists, arrays, dictionaries, and more.
    prefer : str, optional
        Preference for ambiguous conversion ('auto', 'series', 'dataframe').
        Defaults to 'auto'.
    convert_single_column : bool, optional
        Converts single-column DataFrames to Series if True. Defaults to False.
    transform : Callable[[Any], Any], optional
        A function to apply to the data before conversion. The function should
        accept the input data as its argument and return the transformed data.
        Defaults to None.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        The converted and potentially transformed Pandas object.

    Examples
    --------
    >>> from gofast.tools.funcutils import to_pandas 
    >>> to_pandas([1, 2, 3], prefer='series')
    0    1
    1    2
    2    3
    dtype: int64

    >>> to_pandas([[1], [2], [3]], convert_single_column=True)
    0    1
    1    2
    2    3
    Name: 0, dtype: int64

    >>> to_pandas({'A': [1, 2], 'B': [3, 4]})
       A  B
    0  1  3
    1  2  4

    >>> def custom_transform(x):
    ...     return x * 2
    >>> to_pandas([1, 2, 3], transform=custom_transform)
    0    2
    1    4
    2    6
    dtype: int64
    """
    # Apply transformation if a callable is provided
    if transform and callable(transform):
        data = transform(data)

    direct_result = _handle_direct_conversion(data, convert_single_column)
    if direct_result is not None:
        return direct_result  # Return if already a Series or DataFrame
    
    try:
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            if data.ndim == 1 or (
                    data.ndim == 2 and data.shape[1] == 1 and prefer != 'dataframe'):
                return _convert_2_series(data)
            else:
                return _convert_2_dataframe(data, convert_single_column)
        elif isinstance(data, dict):
            return _convert_to_dataframe(data, convert_single_column)
        else:
            raise TypeError("Unsupported data type for conversion.")
    except Exception as e:
        print(f"Conversion failed: {e}")
        return data  # Return original data if conversion is not feasible


def _convert_2_series(data):
    """
    Converts the input data to a Pandas Series, ensuring flat structures are
    appropriately handled.
    """
    return pd.Series(np.array(data).flatten())

def _convert_2_dataframe(data, convert_single_column):
    """
    Converts the input data to a Pandas DataFrame. If `convert_single_column`
    is True and the DataFrame contains only one column, converts it to a Series.
    """
    df = pd.DataFrame(data)
    if convert_single_column and df.shape[1] == 1:
        return df.iloc[:, 0].rename(df.columns[0])
    return df

def _handle_direct_conversion(data, convert_single_column):
    """
    Directly handles conversion if the input is already a Pandas Series or DataFrame,
    applying single-column DataFrame to Series conversion if necessary.
    """
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        if convert_single_column and data.shape[1] == 1:
            return data.iloc[:, 0].rename(data.columns[0])
        return data
    return None  # Indicates that direct conversion did not occur

def flatten_data_if(
    data: Union[ArrayLike, list],
    apply_transform: Callable[[ArrayLike], ArrayLike] = None,
    return_series: bool = False,
    series_name: Optional[str] = None,
    squeeze: bool=False, 
) -> Union[Array1D, Series]:
    """
    Checks the input data's structure and flattens it if necessary to ensure
    compatibility with functions expecting one-dimensional inputs. 
    
    Optionally, applies a transformation to the flattened data and squeezes 
    single-element arrays to scalars if `squeeze` is True.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, list]
        The input data to be checked and potentially flattened.
    apply_transform : Callable[[np.ndarray], np.ndarray], optional
        A function to apply to the data after flattening. The function should
        accept a one-dimensional NumPy array and return a NumPy array.
    return_series : bool, optional
        If True, returns a Pandas Series instead of a NumPy array. 
        Defaults to False.
    series_name : str, optional
        Name of the returned Series. Used only if `return_series` is True. 
        Defaults to 'flattened_data'.
    squeeze: bool, optional
        If True, attempts to squeeze the data, removing single-dimensional entries
        from the shape of the array. For DataFrames, it attempts to return a Series
        if only a single column exists. For arrays, it converts single-element arrays
        to scalars.
        
    Returns
    -------
    Union[np.ndarray, pd.Series]
        The flattened (and potentially transformed) data as a NumPy array 
        or Pandas Series.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> flatten_data_if(df)
    array([1, 2, 3])

    >>> series = pd.Series([4, 5, 6], name='B')
    >>> flatten_data_if(series, return_series=True)
    0    4
    1    5
    2    6
    Name: flattened_data, dtype: int64

    >>> def square(x):
    ...     return x ** 2
    >>> flatten_data_if([1, 2, 3], apply_transform=square, return_series=True)
    0    1
    1    4
    2    9
    Name: flattened_data, dtype: int64
    
    >>> def square(x):
    ...     return x ** 2
    >>> flatten_data_if([1], apply_transform=square, squeeze=True)
    1
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            flattened = data.iloc[:, 0].values
            if squeeze and flattened.size == 1:
                flattened = flattened.item()
            series_name = data.columns[0] if series_name is None else series_name
            
    elif isinstance(data, pd.Series):
        flattened = data.values
        if squeeze and flattened.size == 1:
            flattened = flattened.item()
        series_name = data.name if series_name is None else series_name

    elif isinstance(data, (np.ndarray, list, tuple)):
        flattened = np.array(data).flatten()
        if squeeze and flattened.size == 1:
            flattened = flattened.item()

    else:
        raise TypeError(
            "Unsupported data type. Expected DataFrame, Series, ndarray, or list.")

    if apply_transform is not None:
        flattened = apply_transform(flattened)

    if return_series:
        series_name = series_name or 'flattened_data'
        return pd.Series(flattened, name=series_name)

    return flattened

def update_series_index(
    series: Series, 
    new_indexes: Optional[Union[list, str]] = None, 
    allow_replace: bool = False, 
    return_series: bool = False, 
    on_error: str = 'ignore', 
    transform: Optional[Callable] = None, 
    condition: Optional[LambdaType|Callable[[Series], bool]] = None
):
    """
    Updates the index of a pandas Series with new values under certain conditions.

    Parameters
    ----------
    series : pd.Series
        The Series whose index is to be updated.
    new_indexes : list of str or str, optional
        New values to set as the Series index. If a single string is provided, 
        it's converted to a list with one element. If None, no update is performed.
    allow_replace : bool, default False
        If True, allows the replacement of the existing index even if it's not 
        of integer type.
    return_series : bool, default False
        If True, the updated Series is returned. Otherwise, the new index 
        list is returned.
    on_error : str, default 'ignore'
        Determines the error handling strategy. If 'raise', an error is thrown 
        on failure conditions such as index length mismatch or unsatisfied 
        conditions.
    transform : Callable, optional
        A function to apply to each element of `new_indexes` before updating 
        the index. It must accept a single value and return the transformed value.
    condition : Callable[[pd.Series], bool], optional
        A function that takes the series as input and returns True if the 
        index update should proceed. If False, the update is skipped. This 
        function is checked before any updates.

    Returns
    -------
    pd.Series or list
        Depending on `return_series`, either the updated Series or the new 
        index list is returned.

    Raises
    ------
    ValueError
        If `on_error` is set to 'raise' and any precondition fails (
            e.g., index length mismatch, condition check fails).

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.funcutils import update_series_index
    >>> series = pd.Series([1, 2, 3])

    # Update index without conditions
    >>> update_series_index(series, new_indexes=['a', 'b', 'c'], return_series=True)
    a    1
    b    2
    c    3
    dtype: int64

    # Update index with a condition function
    >>> condition_func = lambda s: pd.api.types.is_integer_dtype(s.index.dtype)
    >>> update_series_index(series, new_indexes=['x', 'y', 'z'],
                            condition=condition_func, return_series=True)
    0    1
    1    2
    2    3
    dtype: int64

    # Example with transform function
    >>> transform_func = lambda x: f'index_{x}'
    >>> update_series_index(series, new_indexes=[1, 2, 3],
                            transform=transform_func, return_series=True)
    index_1    1
    index_2    2
    index_3    3
    dtype: int64

    Note: In the second example, the condition function checks if the series 
    dtype is integer, which is not met, hence the index is not updated.
    Adjusted the condition function according to specific requirements.
    """

    if not isinstance(series, pd.Series):
        msg = ( "Expected input to be a pandas Series,"
               f" got type '{type(series).__name__}' instead."
               )
        if on_error == 'raise':
            raise ValueError(msg)
        return new_indexes if not return_series else series
    
    # Check condition if provided
    if condition is not None and not condition(series):
        if on_error == 'raise':
            raise ValueError("Condition for updating index is not satisfied.")
        return series if return_series else list(series.index)

    if new_indexes is None:
        return series if return_series else list(series.index)
    
    if isinstance(new_indexes, str):
        new_indexes = [new_indexes]
    
    if transform and callable(transform):
        new_indexes = [transform(idx) for idx in new_indexes]
    
    if len(series.index) != len(new_indexes):
        msg = f"Index length mismatch: expected {len(series.index)}, got {len(new_indexes)}."
        if on_error == 'raise':
            raise ValueError(msg)
        return series.index if not return_series else series
    
    # Check if replacement is allowed based on index types and allow_replace flag
    if allow_replace and return_series: 
        series.index = new_indexes 
        
    # if (all(isinstance(v, int) for v in series.index) and allow_replace) or return_series:
    #     series.index = new_indexes
    return series if return_series else new_indexes

def update_dataframe_index(
    df: pd.DataFrame, 
    new_indexes: Optional[Union[list, str]] = None, 
    axis: int = 0,
    allow_replace: bool = False, 
    return_df: bool = False, 
    on_error: str = 'ignore', 
    transform: Optional[Callable] = None, 
    condition: Optional[LambdaType|Callable[[DataFrame], bool]] = None
):
    """
    Updates the index (axis=0) or columns (axis=1) of a pandas DataFrame with 
    new values under certain conditions.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose index or columns are to be updated.
    new_indexes : list of str or str, optional
        New values to set as the DataFrame index or columns. If a single string 
        is provided, it's converted to a list with one element. If None, no 
        update is performed.
    axis : int, default 0
        The axis along which to update the index or columns. 0 for index, 1 for columns.
    allow_replace : bool, default False
        If True, allows the replacement of the existing index or columns even 
        if they're not of integer type.
    return_dataframe : bool, default False
        If True, the updated DataFrame is returned. Otherwise, the new index 
        or column list is returned.
    on_error : str, default 'ignore'
        Determines the error handling strategy. If 'raise', an error is thrown 
        on failure conditions such as length mismatch or unsatisfied conditions.
    transform : Callable, optional
        A function to apply to each element of `new_indexes` before updating 
        the index or columns. It must accept a single value and return the 
        transformed value.
    condition : Callable[[pd.DataFrame], bool], optional
        A function that takes the DataFrame as input and returns True if the 
        index or column update should proceed. If False, the update is skipped. 
        This function is checked before any updates.

    Returns
    -------
    pd.DataFrame or list
        Depending on `return_dataframe`, either the updated DataFrame or the 
        new index or column list is returned.

    Raises
    ------
    ValueError
        If `on_error` is set to 'raise' and any precondition fails (e.g., 
        index/column length mismatch, condition check fails).

    Examples
    --------
    >>> import pandas as pd 
    >>> import gofast.tools.funcutils import update_dataframe_index
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> new_cols = ['X', 'Y']
    >>> update_dataframe_index(df, new_indexes=new_cols, axis=1, return_dataframe=True)
       X  Y
    0  1  4
    1  2  5
    2  3  6

    >>> condition_func = lambda df: pd.api.types.is_integer_dtype(df.index.dtype)
    >>> update_dataframe_index(df, new_indexes=['x', 'y', 'z'],
                               condition=condition_func, return_dataframe=True)
       A  B
    0  1  4
    1  2  5
    2  3  6  # Index not updated due to condition
    """
    if not isinstance(df, pd.DataFrame):
        msg = ("Expected input to be a pandas DataFrame,"
               f" got type '{type(df).__name__}' instead.")
        if on_error == 'raise':
            raise ValueError(msg)
        return new_indexes if not return_df else df
    
    # Check condition if provided
    if condition is not None and not condition(df):
        if on_error == 'raise':
            raise ValueError("Condition for updating index/columns is not satisfied.")
        return df if return_df else (
            df.index if axis == 0 else df.columns).tolist()

    if new_indexes is None:
        return df if return_df else (
            df.index if axis == 0 else df.columns).tolist()
    
    if isinstance(new_indexes, str):
        new_indexes = [new_indexes]
    
    if transform and callable(transform):
        new_indexes = [transform(idx) for idx in new_indexes]
    
    target = df.index if axis == 0 else df.columns
    if len(target) != len(new_indexes):
        msg = f"Length mismatch: expected {len(target)}, got {len(new_indexes)}."
        if on_error == 'raise':
            raise ValueError(msg)
        return target.tolist() if not return_df else df
    
    if allow_replace or return_df: 
        if axis == 0:
            df.index = new_indexes
        else:
            df.columns = new_indexes
    
    return df if return_df else new_indexes

def convert_to_pandas(
        data: ArrayLike | List, 
        error: str='raise', 
        custom_convert: Callable=None
        ):
    """
    Automatically converts input data to a pandas DataFrame or Series 
    based on its structure. 
    
    Function uses a custom conversion function for unsupported data types.
    If the data type is not supported for conversion and error is 'raise', 
    raises a ValueError. If error is 'ignore', returns the data as is.

    Parameters
    ----------
    data : list, dict, array-like, pd.Series, pd.DataFrame
        The input data to be converted.
    error : str, optional, default 'raise'
        Controls error handling for unsupported data types. If 'raise', an 
        error is thrown.If 'ignore', the original data is returned as is.
    custom_convert : callable, optional
        A custom function that takes the original data as input and returns
        a pandas Series or  DataFrame. This function is used if the data type
        is not directly supported for conversion.

    Returns
    -------
    pd.Series, pd.DataFrame, or original data
        The converted pandas Series or DataFrame, or the original data 
        if conversion is not possible and error is set to 'ignore'.

    Raises
    ------
    ValueError
        If the data type cannot be automatically converted, no custom 
        conversion function is provided, and error is set to 'raise'.

    Examples
    --------
    >>> data_list = [1, 2, 3, 4, 5]
    >>> print(convert_to_pandas(data_list))
    0    1
    1    2
    2    3
    3    4
    4    5
    dtype: int64

    >>> data_dict = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> print(convert_to_pandas(data_dict))
       A  B
    0  1  4
    1  2  5
    2  3  6

    # Using a custom conversion function for a nested dictionary
    >>> data_nested_dict = {'A': {'a': 1, 'b': 2}, 'B': {'a': 3, 'b': 4}}
    >>> custom_func = lambda x: pd.DataFrame.from_dict(x)
    >>> print(convert_to_pandas(data_nested_dict, custom_convert=custom_func))
       A  B
    a  1  3
    b  2  4
    """
    try:
        if isinstance(data, dict) or (isinstance(data, list) and all(
                isinstance(elem, list) for elem in data)):
            return pd.DataFrame(data)
        elif isinstance(data, (list, np.ndarray)) and (
                not data or not isinstance(data[0], list)):
            return pd.Series(data)
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        else:
            if callable(custom_convert):
                return custom_convert(data)
            else:
                raise ValueError(
                    f"Cannot auto-convert data of type {type(data).__name__}.")
    except ValueError as e:
        if error == 'raise':
            raise e
        else:
            # Return the original data if error handling is set to 'ignore'
            return data
        
def update_index(
    data: Union[ArrayLike, list, dict], 
    new_indexes: Optional[Union[list, str]] = None, 
    axis: int = 0,
    allow_replace: bool = False, 
    return_data: bool = False, 
    on_error: str = 'ignore', 
    transform: Optional[Callable] = None, 
    condition: Optional[LambdaType|Callable[[Union[Series, DataFrame]], bool]] = None,
    convert_to: Optional[str] = None
):
    """
    Updates the index or columns of a pandas Series or DataFrame with new 
    values, with the option to automatically convert input data to a Series 
    or DataFrame before the update. 
    
    This function provides flexibility in managing data indices, including 
    conditional updates and transformations.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame, list, dict, np.ndarray]
        The data whose index or columns are to be updated. Can be a pandas 
        Series, DataFrame, or a structure convertible to them (list, dict, 
        or ndarray).
    new_indexes : Optional[Union[list, str]], default None
        The new index or column values to apply. If a single string is provided, 
        it is treated as a list with one element.
    axis : int, default 0
        The axis along which to update the index or columns. 0 for index (rows), 
        1 for columns.
    allow_replace : bool, default False
        Whether to allow replacement of the index or columns if they already exist.
    return_data : bool, default False
        If True, returns the updated data structure (Series or DataFrame). If False, 
        returns the new index or column list.
    on_error : str, {'ignore', 'raise'}, default 'ignore'
        Error handling strategy when an update precondition is not met. 'raise' 
        throws a ValueError, while 'ignore' proceeds without throwing.
    transform : Optional[Callable], default None
        An optional function to apply to each of the `new_indexes` before the 
        update. The function must accept a single value and return a transformed value.
    condition : Optional[Callable[[Union[pd.Series, pd.DataFrame]], bool]], default None
        An optional predicate function to determine whether the update should 
        proceed. The function must accept the data structure (Series or DataFrame) 
        and return a boolean.
    convert_to : Optional[str], {'series', 'dataframe', 'auto'}, default None
        Controls automatic conversion of input data to a Series or DataFrame 
        before the update. If 'auto', the function decides the most appropriate 
        type based on the input data structure.

    Returns
    -------
    Union[pd.Series, pd.DataFrame, list]
        Depending on `return_data`, either the updated Series or DataFrame 
        or the new index or column list.

    Raises
    ------
    ValueError
        If `on_error` is 'raise' and any precondition fails (e.g., unsupported 
        data type for conversion, index length mismatch, or condition check fails).

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.funcutils import update_index
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> update_index(df, new_indexes=['X', 'Y'], axis=1, return_data=True)
       X  Y
    0  1  4
    1  2  5
    2  3  6

    >>> update_index([10, 20, 30], new_indexes=['a', 'b', 'c'],
                     convert_to='series', return_data=True)
    a    10
    b    20
    c    30
    dtype: int64

    >>> update_index({'one': [1, 2], 'two': [3, 4]}, 
                     new_indexes=['first', 'second'], axis=0, 
                     convert_to='dataframe', return_data=True)
         one  two
    first   1    3
    second  2    4

    Note
    ----
    The `convert_to` parameter with 'auto' can intelligently convert input data 
    to the most suitable pandas structure for manipulation, enhancing the function's 
    usability across different data formats.
    
    See Also 
    ----------
    gofast.tools.funcutils.to_pandas: 
        onvert input data into a Pandas DataFrame or Series
    """

    # Convert input data to Series or DataFrame if specified
    convert_to = str(convert_to).lower() 
    if convert_to == 'auto':
        data = convert_to_pandas(data, error=on_error)
    elif convert_to == 'series':
        data = pd.Series(data)
    elif convert_to == 'dataframe':
        data = pd.DataFrame(data)

    # Determine if input data is Series or DataFrame and update accordingly
    if isinstance(data, pd.Series):
        return update_series_index(
            data, 
            new_indexes, 
            allow_replace=allow_replace, 
            return_series=return_data, 
            on_error =on_error, 
            transform=transform, 
            condition=condition 
            )
    elif isinstance(data, pd.DataFrame):
        return update_dataframe_index(
            data, 
            new_indexes,
            axis=axis, 
            allow_replace=allow_replace, 
            return_df=return_data, 
            on_error=on_error, 
            transform=transform,
            condition=condition
            )
    else:
        raise ValueError("Input data must be a pandas Series or DataFrame,"
                         f" got {type(data).__name__}.")

def convert_and_format_data(
    data: Any,
    return_df: bool = False,
    series_name: Optional[str] = None,
    allow_series_conversion: bool = True,
    force_array_output: bool = False,
    condense: bool = False,
    custom_conversion: Optional[Callable[[Any], Union[DataFrame, Series]]] = None,
    condition: Optional[LambdaType | Callable[[Any], dict]] = None, 
    where: str = 'before'
):
    """
    Converts the input data into a pandas DataFrame or Series, then formats it
    according to specified parameters and an optional condition for dynamic
    parameter adjustments. Optionally applies a custom conversion function.

    Parameters
    ----------
    data : Any
        The input data to be converted. Can be any type that is supported by
        the custom conversion function or can be automatically handled by pandas.
    return_df : bool, default False
        If True, ensures the output is a DataFrame. If False and the data can
        be represented as a Series (or an array if `force_array_output` is True),
        the output will not be a DataFrame.
    series_name : Optional[str], default None
        Specifies a new name for the Series if the output is a Series. This parameter
        is ignored if `return_df` is True or if the data cannot be converted
        to a Series.
    allow_series_conversion : bool, default True
        Allows converting a single-column DataFrame into a Series if `return_df`
        is False. If False, single-column DataFrames will not be converted to Series.
    force_array_output : bool, default False
        If True, converts the output to a numpy array, reducing its dimension 
        if possible. This parameter is considered only if `return_df` is False.
    condense : bool, default False
        If True, simplifies the output in the following ways when 
        `force_array_output` is True:
        - Converts a single-value array or Series to a scalar (float or int).
        - Reduces a two-dimensional array of shape (n, 1) or (1, n) to a 
          one-dimensional array of shape (n,).
        This simplification is considered only if `force_array_output` is 
        True and `return_df` is False.

    custom_conversion : Optional[Callable[[Any], pd.DataFrame]], default None
        A custom function that takes the input data and returns a pandas DataFrame.
        This function is used for conversion if provided.
    condition : Optional[Callable[[Any], dict]], default None
        A condition function that takes the input data as is and returns a dictionary
        of parameter adjustments. For example, to modify `return_df` based on
        data properties, return `{'return_df': True}` if the condition is met.
    where : str, default 'before'
        Determines when the condition is applied: 'before' applies the condition
        before any conversion, and 'after' applies it after the initial conversion
        but before any formatting.

    Returns
    -------
    output : pd.DataFrame, pd.Series, or np.ndarray
        The converted and formatted data. The specific type depends on the input
        parameters and the nature of `data`.

    Examples
    --------
    Convert a list to a pandas DataFrame:
    
    >>> from gofast.tools.funcutils import convert_and_format_data
    >>> convert_and_format_data([1, 2, 3], output_as_frame=True)
    pd.DataFrame(data=[1, 2, 3])

    Convert a single-column DataFrame to a Series named 'my_series':

    >>> convert_and_format_data(pd.DataFrame({'A': [1, 2, 3]}), 
                                series_name='my_series')
    pd.Series(data=[1, 2, 3], name='my_series')

    Convert data using a custom conversion function and output as numpy array:

    >>> def my_conversion(data):
    ...     return pd.DataFrame(data)
    >>> convert_and_format_data({'A': [1, 2], 'B': [3, 4]}, 
                                custom_conversion=my_conversion, 
                                force_array_output=True)
    np.array([[1, 3], [2, 4]])
    
    >>> convert_and_format_data([1, 2, 3], output_as_frame=True)
    pd.DataFrame(data=[1, 2, 3])

    Convert a single-column DataFrame to a Series named 'my_series', then to an array:

    >>> condition = lambda x: isinstance(x, pd.Series)
    >>> convert_and_format_data(pd.DataFrame({'A': [1, 2, 3]}),
                                series_name='my_series',
                                force_array_output=True,
                                condition=condition)
    np.array([1, 2, 3])

    Use a custom conversion function and force output as numpy array if data is a Series:

    >>> def my_conversion(data):
    ...     return pd.DataFrame(data)
    >>> convert_and_format_data({'A': [1, 2], 'B': [3, 4]},
                                custom_conversion=my_conversion,
                                force_array_output=True,
                                condition=lambda x: isinstance(x, pd.Series))
    np.array([[1, 3], [2, 4]])

    >>> convert_and_format_data(pd.DataFrame({'A': [1, 2, 3]}), series_name='my_series')
    pd.Series(data=[1, 2, 3], name='my_series')

    >>> condition = lambda x: {'return_df': True} if isinstance(x, list) else {}
    >>> convert_and_format_data([1, 2, 3], condition=condition, where='before')
    pd.DataFrame(data=[1, 2, 3])
    

    Convert a list to a pandas DataFrame:
    
    >>> convert_and_format_data([1, 2, 3], return_df=True)
    pd.DataFrame(data=[1, 2, 3])
    
    Convert a single-column DataFrame to a Series named 'my_series',
    then simplify to a one-dimensional array:
    
    >>> convert_and_format_data(pd.DataFrame({'A': [1, 2, 3]}), 
    ...                            series_name='my_series',
    ...                            force_array_output=True,
    ...                            condense=True)
    array([1, 2, 3])
    
    Simplify a two-dimensional array to a one-dimensional array:
    
    >>> convert_and_format_data(np.array([[1], [2], [3]]), 
    ...                            force_array_output=True,
    ...                            condense=True)
    array([1, 2, 3])
    
    Convert a single-value Series to a scalar:
    
    >>> convert_and_format_data(pd.Series([100]), 
                                force_array_output=True, condense=True)
    100

    """
    adjustments = {}

    # Apply condition based on 'where' parameter
    if where == 'before':
        if condition:
            adjustments = condition(data)

    if custom_conversion:
        data = custom_conversion(data)
    else:
        data = to_pandas(data, convert_single_column=allow_series_conversion)
    
    # Apply condition after initial conversion if 'where' is 'after'
    if where == 'after':
        if condition:
            adjustments = condition(data)

    # Apply adjustments
    return_df = adjustments.get('return_df', return_df)
    series_name = adjustments.get('series_name', series_name)
    allow_series_conversion = adjustments.get(
        'allow_series_conversion', allow_series_conversion)
    force_array_output = adjustments.get(
        'force_array_output', force_array_output)
    
    # Proceed with data formatting based on adjusted parameters
    if isinstance(data, pd.Series):
        if series_name is not None:
            data.name = series_name
        if return_df:
            data = pd.DataFrame(data)
        elif force_array_output:
            data = data.to_numpy()
    elif isinstance(data, pd.DataFrame) and not return_df:
        if allow_series_conversion and data.shape[1] == 1:
            data = data.squeeze()
            if series_name:
                data.name = series_name
        if force_array_output:
            data = data.to_numpy()
            
    # Simplify output based on requested dimensions and content.
    if condense: # and if force_array_outpout
        # Use a single process to handle both DataFrame and Series to avoid repetition.
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Directly extract the scalar value if the data contains only one element.
            data = data.squeeze().item() if data.size == 1 else data.squeeze()
        elif isinstance(data, np.ndarray):
            # For numpy arrays, `.squeeze()` removes axes of length one, 
            # which simplifies the data to a lower dimension while `.item()`
            # converts a single-value array to a scalar.
            # Use `.squeeze()` first to handle both (n, 1) and (1, n) cases,
            # then `.item()` if the result is a single value.
            data = data.squeeze()
            if data.size == 1:
                data = data.item()  # Converts to a scalar
    return data

def cast_numeric(
    value, error='ignore',
    on_success=None, 
    on_failure=None
    ):
    """
    Checks whether a given value is convertible to a numeric type (int or float).

    Parameters
    ----------
    value : Any
        The value to check for numeric convertibility.
    error : str, optional
        Specifies the error handling strategy. If 'raise', raises a ValueError
        when `value` cannot be converted to a numeric type. If 'ignore', the function
        returns False without raising an error. Default is 'ignore'.
    on_success : Callable[[Union[int, float]], Any], optional
        A callable that is executed with the converted numeric value if the conversion
        is successful. The callable should accept a single numeric argument (int or float)
        and return any value. If None, this step is skipped. Default is None.
    on_failure : Callable[[Any], Any], optional
        A callable that is executed if the conversion fails. The callable should
        accept the original `value` as an argument and return any value. This allows
        for custom handling or logging of failures. If None, this step is skipped.
        Default is None.

    Returns
    -------
    bool or Any
        Returns True if the value is convertible to a numeric type, False otherwise.
        If `on_success` is provided and the conversion is successful, returns
        the result of `on_success` instead.

    Raises
    ------
    ValueError
        Raised if `error` is set to 'raise' and the value cannot be converted to
        a numeric type.

    Examples
    --------
    >>> from gofast.tools.funcutils import cast_numeric
    >>> cast_numeric("100.5")
    True

    >>> cast_numeric("abc", error='raise')
    ValueError: Value 'abc' is not convertible to numeric.

    >>> cast_numeric("123", on_success=lambda x: x*2)
    246

    >>> cast_numeric("abc", on_failure=lambda x: f"Not convertible: {x}")
    'Not convertible: abc'
    """
    try:
        # First, try converting directly to an integer or float, without assuming it's a string
        if isinstance(value, (int, float, np.integer, np.floating)):
            return True  # Already numeric, no conversion needed
        converted_value = float(value) if '.' in value or 'e' in value.lower() else int(value)
        if on_success:
            return on_success(converted_value)
        return True
    except (ValueError, TypeError):
        if error == 'raise':
            raise ValueError(f"Value '{value}' is not convertible to numeric.")
        if on_failure:
            return on_failure(value)
        return False

def series_naming(name, data=None, error='ignore'):
    """
    Generates a condition function for renaming a pandas Series or single-column
    DataFrame and optionally transforms and directly renames the provided data.

    If `data` is provided, attempts to convert it to a pandas Series (if it's an
    array-like structure) or rename it (if it's a single-column DataFrame) with
    the specified name. If `data` is None, returns a condition function that can
    be used to dynamically rename Series or single-column DataFrame based on
    runtime evaluation.

    Parameters
    ----------
    name : str
        The new name to assign to the Series or single-column DataFrame.
        This name is applied only if the current Series name or DataFrame 
        column name is numeric or not explicitly set.
    data : array-like, pd.DataFrame, optional
        The data to be converted to a Series or whose single column is to be
        renamed. Supported types include list, tuple, np.ndarray, and pd.DataFrame.
        If None, no direct conversion or renaming is performed.
    error : str, {'ignore', 'raise'}, default 'ignore'
        Error handling strategy. If 'raise', errors during data conversion or
        renaming will raise an exception. If 'ignore', errors are suppressed.

    Returns
    -------
    function or pd.Series or pd.DataFrame
        If `data` is None, returns a lambda function that applies the new name
        based on conditions. If `data` is provided, attempts the conversion or
        renaming operation and returns the transformed data.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.funcutils import series_naming
    
    >>> df = pd.DataFrame({'1': [1, 2, 3]})
    >>> condition = serie_naming('NewName')
    >>> result = condition(df)
    >>> print(result)
    {'series_name': 'NewName'}

    >>> series = pd.Series([1, 2, 3], name='2')
    >>> condition = series_naming('SeriesName')
    >>> result = condition(series)
    >>> print(result)
    {'series_name': 'SeriesName'}

    >>> series = pd.Series([1, 2, 3], name='ExistingName')
    >>> condition = series_naming('NewSeriesName')
    >>> result = condition(series)
    >>> print(result)
    {}
    >>> serie_naming('NewName', data=[1, 2, 3])
    0    1
    1    2
    2    3
    Name: NewName, dtype: int64

    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> series_naming('SingleCol', data=df)
    0    1
    1    2
    2    3
    Name: SingleCol, dtype: int64

    >>> condition = series_naming('DynamicName')
    >>> series = pd.Series([1, 2, 3], name='1')
    >>> condition(series)
    {'series_name': 'DynamicName'}

    Note
    ----
    This function supports both static and dynamic renaming of Series or DataFrame
    columns. It facilitates maintaining consistent naming conventions across data
    transformation workflows, especially when dealing with dynamically generated
    or transformed data.
    """
    if data is not None:
        try:
            if isinstance(data, (list, tuple, np.ndarray)):
                data = pd.Series(data, name=name)
            elif isinstance(data, pd.DataFrame) and data.shape[1] == 1:
                data = data.iloc[:, 0].rename(name)
            elif error == 'raise':
                raise ValueError("Data must be convertible to a Series.")
        except Exception as e:
            if error == 'raise':
                raise e
            warnings.warn(f"Warning: {e}")
        return data

    # Return a condition function for dynamic renaming if data is not directly provided
    return lambda s: {"series_name": name} if (
        isinstance(s, pd.Series) and (cast_numeric(s.name, error='ignore') or s.name is None) or
        isinstance(s, pd.DataFrame) and len(s.columns) == 1 and (
            cast_numeric(s.columns[0], error='ignore') or s.columns[0] is None)
    ) else {}


def validate_years(
    start_year: Optional[Union[int, str]] = None, 
    end_year: Optional[Union[int, str]] = None, *, 
    check_range: bool = True,
    range_validator: Optional[Callable[[int], bool]] = None
) -> Union[Callable, tuple]:
    """
    Validate the years provided to a function or use as a decorator.
    
    This function checks if the `start_year` and `end_year` are valid integer 
    years and whether `start_year` is before `end_year`. Optionally, it 
    validates if the years are within a reasonable range defined by the 
    `range_validator`. The function can be used as a decorator to enforce 
    this validation on function arguments named `start_year` and `end_year`, 
    or as a standalone function that validates and returns the years.
    
    Parameters
    ----------
    start_year : int, str, or None, optional
        The starting year as an integer, a string representing a year, or 
        the name of the  parameter representing the starting year in the 
        decorated function. If `None`, it will be treated as a decorator, 
        and the parameter will be dynamically determined.
        
    end_year : int, str, or None, optional
        The ending year as an integer, a string representing a year, or the 
        name of the parameter representing the ending year in the decorated 
        function. If `None`, it will be treated as a decorator, and the 
        parameter will be dynamically determined.
        
    check_range : bool, default=True
        If `True`, enables the range check of the years using the 
        `range_validator`. If `False`, this check is bypassed.
        
    range_validator : Callable[[int], bool], optional
        A function that accepts an integer year and returns `True` if the year 
        is within an acceptable range. If not provided, the default validator 
        checks if the year is between 1900 and the current year.
        
    Returns
    -------
    Callable or tuple
        If used as a decorator, it returns the wrapped function. If used as 
        a standalone function, it returns a tuple of validated 
        `(start_year, end_year)` as integers.
        
    Raises
    ------
    ValueError
        If the years cannot be parsed as integers, `start_year` is not less 
        than `end_year`, or the years are not within the valid range as 
        determined by `range_validator`.
        
    Examples
    --------
    As a decorator, using parameter names of the decorated function:
    
    >>> @validate_years(start_year='start', end_year='end')
    ... def some_function(start: int, end: int):
    ...     return f"Range: {start} to {end}"
    
    Using a custom `range_validator` to enforce that years must be in the
    21st century:

    >>> def is_21st_century(year: int) -> bool:
    ...     return 2000 <= year <= 2100
    >>> start, end = validate_years('2001', '2020', range_validator=is_21st_century)
    >>> print(f"Validated range: {start} to {end}")
    Validated range: 2001 to 2020
    
    As a standalone function:
    
    >>> start, end = validate_years('2001', '2020')
    >>> print(f"Validated range: {start} to {end}")
    Validated range: 2001 to 2020
    """
    if range_validator is None:
        range_validator = lambda x: (1900 <= x <= datetime.now().year, (1900, datetime.now().year))
    
    # Decorator function
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal start_year, end_year
            # If start_year and end_year are string, try getting the value from args or kwargs
            if isinstance(start_year, str) or isinstance(end_year, str):
                sy = _get_param_value(start_year, args, kwargs, func)
                ey = _get_param_value(end_year, args, kwargs, func)
            else:
                # Otherwise, parse them directly
                sy = _parse_year(start_year, 'start_year')
                ey = _parse_year(end_year, 'end_year')
            
            # Validation logic remains unchanged
            _check_year_order(sy, ey)
            if check_range:
                _check_year_range(sy, range_validator)
                _check_year_range(ey, range_validator)
                
            return func(*args, **kwargs)
        return wrapper

    # Improved condition to determine function usage based on argument types
    if ((isinstance(start_year, (int, str)) or start_year is None) and 
        (isinstance(end_year, (int, str)) or end_year is None)):
        # Direct function usage, assuming start_year and end_year are not both callables
        if not callable(start_year) and not callable(end_year):
            sy = _parse_year(start_year, 'start_year') if start_year is not None else None
            ey = _parse_year(end_year, 'end_year') if end_year is not None else None
            if sy is not None and ey is not None:
                _check_year_order(sy, ey)
                if check_range:
                    _check_year_range(sy, range_validator)
                    _check_year_range(ey, range_validator)
                return sy, ey
        return decorator
    else:
        # Otherwise, assume decorator usage with function to decorate
        # being passed later
        return decorator

def _parse_year(year: Union[int, str], name: str) -> int:
    """
    Parse the year from a string or integer to an integer.
    
    Parameters
    ----------
    year : int or str
        The year to parse.
    name : str
        The name of the parameter (start_year or end_year) for error messaging.
        
    Returns
    -------
    int
        The parsed year as an integer.
    
    Raises
    ------
    ValueError
        If the year cannot be converted to an integer.
    """
    try:
        return int(year)
    except ValueError:
        raise ValueError(f"'{name}' must be convertible to an integer.")


def _check_year_order(start_year: int, end_year: int) -> None:
    """
    Check if start_year is less than end_year.
    
    Parameters
    ----------
    start_year : int
        The starting year.
    end_year : int
        The ending year.
    
    Raises
    ------
    ValueError
        If start_year is not less than end_year.
    """
    if start_year >= end_year:
        raise ValueError(f"{start_year} must be less than {end_year}.")


def _check_year_range(
        year: int, 
        range_validator: Callable[[int], Union[bool, Tuple[bool, Tuple[int, int]]]]
        ) -> None:
    """
    Validates if a given year is within an acceptable range as determined by
    the range_validator. The range_validator function can return either a 
    boolean indicating the validity of the year, or a tuple comprising a 
    boolean and another tuple detailing the valid range (start, end).

    Parameters
    ----------
    year : int
        The year to be validated.
    range_validator : Callable[[int], Union[bool, Tuple[bool, Tuple[int, int]]]]
        A function that returns either a boolean indicating if the year is valid,
        or a tuple containing a boolean and the valid range. If the year is
        not valid and a range is provided, a ValueError will be raised including
        the valid range in its message.

    Raises
    ------
    ValueError
        If the year is not within the valid range. The exception message will
        include the valid range if it is provided by the range_validator.

    Examples
    --------
    Using a range_validator that returns a boolean:
    
    >>> from gofast.tools.funcutils import _check_year_range
    >>> def simple_validator(year: int) -> bool:
    ...     return 2000 <= year <= 2100
    >>> try:
    ...     _check_year_range(1999, simple_validator)
    ... except ValueError as e:
    ...     print(e)
    The year 1999 is out of the valid range.

    Using a range_validator that returns a tuple with the valid range:
    
    >>> def detailed_validator(year: int) -> Tuple[bool, Tuple[int, int]]:
    ...     return 2000 <= year <= 2100, (2000, 2100)
    >>> try:
    ...     _check_year_range(1999, detailed_validator)
    ... except ValueError as e:
    ...     print(e)
    The year 1999 is out of the valid range. Valid range is 2000 to 2100.
    """
    result = range_validator(year)
    is_valid = result if isinstance(result, bool) else result[0]
    valid_range = None if isinstance(result, bool) else result[1]

    if not is_valid:
        error_message = f"The year {year} is out of the valid range."
        if valid_range:
            error_message += f" Valid range is {valid_range[0]} to {valid_range[1]}."
        raise ValueError(error_message)

def _get_param_value(param_name: Union[int, str], args, kwargs, func) -> int:
    """
    Get the value of a parameter by name from args or kwargs of a function.
    
    Parameters
    ----------
    param_name : int, str
        The name of the parameter to retrieve the value for, or its actual value.
    args : tuple
        The positional arguments passed to the function.
    kwargs : dict
        The keyword arguments passed to the function.
    func : Callable
        The function from which to retrieve the parameter value.
    
    Returns
    -------
    int
        The value of the parameter as an integer.
    """
    if isinstance(param_name, int):
        return param_name
    elif param_name in kwargs:
        return kwargs[param_name]
    else:
        param_index = func.__code__.co_varnames.index(param_name)
        if param_index < len(args):
            return args[param_index]
        else:
            raise ValueError(f"The parameter {param_name} was not provided to the function.")

def context_checker(
    return_context_as_bool: bool = False,  
    custom_logic: Optional[Callable[[Callable], Callable]] = None,
    verbose: bool = False
    ) -> Callable:
    """
    A versatile decorator to introspect and modify the behavior of functions
    based on their context of usage. It can determine whether a function is
    being used as a decorator for another function or called directly. This
    decorator also supports executing with custom or default logic.
    
    Parameters
    ----------
    return_context_as_bool : bool, optional
        If True, the decorator returns a boolean value indicating the context
        instead of modifying or executing the target function. ``True`` means 
        the function is used as a decorator, and ``False`` indicates it is called
        directly. Defaults to ``False``.
    
    custom_logic : Optional[Callable[[Callable], Callable]], optional
        A function that defines custom logic to apply to the decorated function.
        If provided, this logic supersedes the default decoration logic.
        If ``None``, default logic is applied.
        
    verbose : bool, optional
        If ``True``, prints messages indicating the detected usage context. 
        Useful for debugging purposes. Defaults to ``False``.
        
    Returns
    -------
    Callable
        Depending on `return_context_as_bool`, the function either returns a
        boolean indicating the usage context or modifies the target function
        based on the specified logic (`custom_logic` or default).

    Examples
    --------
    Direct call returning context as boolean:
    
    >>> import functools 
    >>> from gofast.tools.funcutils import context_checker
    >>> @context_checker(return_context_as_bool=True)
    ... def my_function():
    ...     pass
    >>> print(my_function())
    Detected direct call or dual-use scenario.
    True
    
    Used as a decorator with custom logic, printing before and after:
    
    >>> def custom_logic(func):
    ...     @functools.wraps(func)
    ...     def wrapper(*args, **kwargs):
    ...         print("Before function call")
    ...         result = func(*args, **kwargs)
    ...         print("After function call")
    ...         return result
    ...     return wrapper
    >>> @context_checker(custom_logic=custom_logic)
    ... def another_function():
    ...     print("Function logic here.")
    >>> another_function()
    Detected dual-use or direct call scenario.
    Before function call
    Function logic here.
    After function call
    
    Using as a dual-use decorator without providing function (useful for
    decorators that can operate without arguments):
    
    >>> @context_checker(verbose=True)
    ... def dual_use_decorator(func=None):
    ...     if func is not None:
    ...         return context_checker(func)
    ...     def inner_decorator(f):
    ...         @functools.wraps(f)
    ...         def wrapper(*args, **kwargs):
    ...             print("Dual-use logic applied.")
    ...             return f(*args, **kwargs)
    ...         return wrapper
    ...     return inner_decorator
    >>> @dual_use_decorator()
    ... def decorated_function():
    ...     print("Decorated function executed.")
    >>> decorated_function()
    Detected dual-use or direct call scenario.
    Dual-use logic applied.
    Decorated function executed.
    """
    def wrapper(target: Optional[Callable] = None) -> Any:
        def apply_logic(func: Callable) -> Callable:
            logic_to_apply = custom_logic if custom_logic else _apply_default_logic
            return logic_to_apply(func)
        
        if target and callable(target):
            if return_context_as_bool:
                if verbose:
                    print("Detected as a decorator.")
                return False
            return apply_logic(target)
        
        if return_context_as_bool:
            if verbose:
                print("Detected direct call or dual-use scenario.")
            return True
        
        def dual_use_decorator(func: Optional[Callable] = None) -> Any:
            if callable(func):
                if verbose:
                    print("Detected dual-use or direct call scenario.")
                return apply_logic(func)
            else:
                if verbose:
                    print("Detected function definition missing, applying logic directly.")
                # Handle the case where the decorator is used without ()
                return apply_logic(dual_use_decorator)
        
        return dual_use_decorator if not target else apply_logic(target)
    
    def _apply_default_logic(func: Callable) -> Callable:
        @functools.wraps(func)
        def decorated(*args, **kwargs) -> Any:
            if verbose:
                print("Default decoration logic applied.")
            return func(*args, **kwargs)
        return decorated
    
    return wrapper if not return_context_as_bool else wrapper(None)























