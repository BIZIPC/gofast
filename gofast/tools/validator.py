# -*- coding: utf-8 -*-
# BSD-3-Clause License
# Copyright (c) 2024 gofast developers.
# All rights reserved.

from functools import wraps
from typing import Any, Callable, Optional, Union
import re
import inspect 
import types 
import warnings
import numbers
import operator
import joblib
from datetime import datetime
from contextlib import suppress

import numpy as np
import pandas as pd
from numpy.core.numeric import ComplexWarning  
import scipy.sparse as sp
from inspect import signature, Parameter, isclass 

from ._array_api import get_namespace, _asarray_with_order
FLOAT_DTYPES = (np.float64, np.float32, np.float16)


def filter_valid_kwargs(callable_obj, kwargs):
    """
    Filter and return only the valid keyword arguments for a given callable object.

    This function checks if the arguments in `kwargs` are valid for the 
    provided callable object (function, lambda function, method, or class). 
    If any argument is not valid, it is removed from `kwargs`. The function 
    returns only the valid `kwargs`.

    Parameters
    ----------
    callable_obj : callable
        The callable object (function, lambda function, method, or class) for 
        which the keyword arguments need to be validated.
    
    kwargs : dict
        Dictionary of keyword arguments to be validated against the callable object.

    Returns
    -------
    valid_kwargs : dict
        Dictionary containing only the valid keyword arguments for the callable object.

    Examples
    --------
    >>> def example_func(a, b, c=3):
    ...     pass
    >>> kwargs = {'a': 1, 'b': 2, 'd': 4}
    >>> filter_valid_kwargs(example_func, kwargs)
    {'a': 1, 'b': 2}
    
    >>> class ExampleClass:
    ...     def __init__(self, x, y, z=10):
    ...         pass
    >>> kwargs = {'x': 1, 'y': 2, 'a': 3}
    >>> filter_valid_kwargs(ExampleClass, kwargs)
    {'x': 1, 'y': 2}
    >>> filter_valid_kwargs(ExampleClass(), kwargs)
    {'x': 1, 'y': 2}

    Notes
    -----
    This function uses the `inspect` module to retrieve the signature of 
    the given callable object and validate the keyword arguments.
    """
    # If the callable_obj is an instance, get its class
    if not inspect.isclass(callable_obj) and not callable(callable_obj):
        callable_obj = callable_obj.__class__

    # Get the function signature
    signature = inspect.signature(callable_obj)
    
    # Extract parameter names from the function signature
    valid_params = set(signature.parameters.keys())

    # Filter kwargs to retain only valid parameters
    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return valid_kwargs

def validate_scores(
    scores, true_labels=None, 
    mode="strict", 
    accept_multi_output=False
    ):
    """
    Validates that the scores represent valid probability distributions and 
    checks consistency between scores and true labels in multi-output scenarios.

    Parameters
    ----------
    scores : list or np.ndarray
        A list of np.ndarrays for multi-output probabilities, or a single np.ndarray
        for single-output probabilities. Each ndarray should contain probability
        distributions where each row sums to approximately 1 and has 
        non-negative values.
    true_labels : list or np.ndarray, optional
        The true labels corresponding to the scores. This parameter must 
        be provided in multi-output scenarios to check the alignment of labels
        and scores. Each element or row in true_labels should correspond to 
        the equivalent in scores.
    mode : str, optional (default "strict")
       Specifies the validation mode for checking probability distributions:
       - 'strict': Each set of scores must sum exactly to 1, within a numerical
         tolerance.
       - 'soft': Scores must not exceed a total of 1, and all individual 
         scores must be non-negative.
       - 'passthrough': Only checks that each score is between 0 and 1 
         inclusive, without summing them.    
    accept_multi_output : bool, default False
        Flag indicating whether scores with multiple outputs are accepted. 
        If False and scores are provided as a list, a ValueError will be 
        raised.

    Returns
    -------
    np.ndarray
        The validated scores as a NumPy array.

    Raises
    ------
    ValueError
        If multi-output scores are provided and not accepted.
        If there is a mismatch in the number of outputs between scores and 
        true_labels.
        If scores or any subset of scores do not form valid probability 
        distributions.
        If there is a mismatch in format expectations between scores and 
        true_labels in terms of multi-output handling.

    Notes
    -----
    The function is designed to handle both single and multi-output probability
    distributions. For multi-output scenarios, both scores and true_labels 
    should be lists of np.ndarrays.
    This function is particularly useful in scenarios involving machine learning
    models where output probabilities need to be validated before further
    processing or metrics calculations.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.validator import validate_scores
    >>> scores_single = np.array([[0.1, 0.9], [0.8, 0.2]])
    >>> print(validate_scores(scores_single))
    [[0.1, 0.9]
     [0.8, 0.2]]

    >>> scores_multi = [np.array([[0.1, 0.9]]), np.array([[0.8, 0.2]])]
    >>> true_labels_multi = [np.array([1]), np.array([0])]
    >>> print(validate_scores(scores_multi, true_labels_multi, accept_multi_output=True))
    [array([[0.1, 0.9]]), array([[0.8, 0.2]])]
    """
    # Check if scores are in a list for multi-output handling
    if isinstance(scores, list):
        if not accept_multi_output:
            raise ValueError("Multi-output scores provided but not accepted.")
        if true_labels is not None and len(scores) != len(true_labels):
            raise ValueError("Mismatch in the number of outputs between"
                             " scores and true_labels.")
        if any(not _is_probability_distribution(
                score, mode=mode) for score in scores):
            raise ValueError("Each set of scores must be a valid"
                             " probability distribution.")
    else:
        if not _is_probability_distribution(scores, mode=mode):
            raise ValueError("Scores must be a valid probability distribution.")
        if true_labels is not None:
            if accept_multi_output and not isinstance(true_labels, list):
                raise ValueError("Expected multi-output for true_labels"
                                 " but got a single output.")
            if not accept_multi_output and isinstance(true_labels, list):
                raise ValueError("Non-multi-output scores with multi-output"
                                 " true_labels.")
    # Return scores as numpy array
    return np.asarray(scores)

def _is_probability_distribution(y, mode='strict'):
    """
    Checks if `y` is a probability distribution across the last axis according 
    to the specified mode.

    Parameters
    ----------
    y : np.ndarray
        Array containing score values which need to be validated as probability
        distributions.
    mode : str, optional
        Validation mode to be used. Available modes are:
        - 'strict': Requires that the sum of scores exactly equals 1 
        (within a tolerance).
        - 'soft': Requires that the sum of scores does not exceed 1 and all
        scores are non-negative.
        - 'passthrough': Only checks that all scores are non-negative and do 
          not exceed 1, without summing them.

    Returns
    -------
    bool
        True if `y` satisfies the conditions of the specified mode, False 
        otherwise.

    Raises
    ------
    ValueError
        If an invalid mode is specified.

    Examples
    --------
    >>> y = np.array([0.3, 0.7])
    >>> print(is_probability_distribution(y, mode='strict'))
    True

    >>> y = np.array([0.5, 0.5, 0.2])
    >>> print(is_probability_distribution(y, mode='soft'))
    False

    >>> y = np.array([0.2, 0.3, 0.4])
    >>> print(is_probability_distribution(y, mode='passthrough'))
    True
    """
  
    if mode == 'strict':
        return np.all(np.isclose(np.sum(y, axis=-1), 1)) and np.all(y >= 0)
    elif mode == 'soft':
        return np.all(np.sum(y, axis=-1) <= 1) and np.all(y >= 0)
    elif mode == 'passthrough':
        return np.all(np.asarray(y) <= 1) and np.all(np.asarray(y) >= 0)
    else:
        raise ValueError(f"Invalid validation mode: '{mode}'. Valid modes"
                         " are 'strict', 'soft', or 'passthrough'.")

def validate_square_matrix(data, /, align=False, align_mode="auto", message=''):
    """
    Validate that the input data forms a square matrix and optionally aligns its 
    indices and columns if specified.

    Parameters:
    -----------
    data : DataFrame or array-like
        The input data to validate as a square matrix.
    align : bool, default False
        Whether to align the DataFrame's index with its columns.
    align_mode : str, default 'auto'
        Alignment mode if indices and columns do not match. Options are 'auto', 
        'index_to_columns', and 'columns_to_index'.
    message : str, default ''
        Additional message to append to the error if validation fails.

    Returns:
    --------
    data
        The validated or aligned square matrix.

    Raises:
    -------
    ValueError
        If the input is not a square matrix.

    Examples:
    ---------
    >>> from gofast.tools.validator import validate_square_matrix
    >>> validate_square(np.array([[1, 2], [3, 4]]))
    array([[1, 2],
           [3, 4]])

    >>> validate_square(pd.DataFrame([[1, 2], [3, 4, 5]]))
    ValueError: Input must be a square matrix.

    Notes:
    ------
    A square matrix is defined as having equal number of rows and columns. 
    This function checks the dimensionality of the data and optionally aligns 
    the index and columns if `align` is set to True.
    """
    if not is_square_matrix(data):
        raise ValueError(f"Input must be a square matrix. {message}")
    if align: 
        data = validate_comparison_data(data, alignment=align_mode)
    return data

def is_square_matrix(data, data_type=None):
    """
    Determine whether the input, either a DataFrame or an array-like 
    structure, forms a square matrix.
    
    Automatically detects the data type unless specified. Supports data inputs
    that can be converted to a NumPy array.
    
    Parameters:
    -----------
    data : DataFrame, array-like, or any object convertible to a numpy array
        The input data to check.
    data_type : str, optional
        The expected type of the input data. Valid options are 'array' or 
        'dataframe'.
        If not specified, the data type is inferred. Default interpretation 
        is as an 'array'.

    Returns:
    --------
    bool
        Returns True if the data is a square matrix, otherwise False.
        
    Raises:
    ------
    ValueError
        If `data_type` is neither 'array' nor 'dataframe'.
    TypeError
        If the input `data` does not match the expected format or 
        cannot be processed.

    Examples:
    ---------
    >>> is_square_matrix(np.array([[1, 2], [3, 4]]))
    True

    >>> is_square_matrix(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))
    False

    >>> is_square_matrix([[1, 2], [3, 4]], data_type='array')
    True

    Notes:
    ------
    A square matrix has an equal number of rows and columns. This function 
    checks the dimensionality and shape of the data to confirm if it meets 
    this criterion.
    """
    # Determine the type based on the data provided
    if data_type is None:
        if isinstance(data, np.ndarray):
            data_type = 'array'
        elif isinstance(data, pd.DataFrame):
            data_type = 'dataframe'
        else:
            data = np.array(data)  # Attempt to convert to a numpy array
            data_type = 'array'

    if data_type not in ['array', 'dataframe']:
        raise ValueError("data_type must be either 'array' or 'dataframe'")

    # Check if the data is a square matrix
    if data_type == 'array':
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            return False
    elif data_type == 'dataframe':
        if data.shape[0] != data.shape[1]:
            return False
    else:
        raise TypeError(f"Unsupported or mismatched data type: {data_type}")

    return True

def validate_multiclass_target(
        y, accept_multioutput=False, return_classes=False):
    """
    Validates that the target data is suitable for multiclass classification.
    Optionally accepts multi-output targets and can return the unique classes.

    Parameters
    ----------
    y : array-like
        The target data to be validated, expected to contain class labels for
        multiclass classification. Can be a multi-output array if accept_multioutput
        is set to True.
    accept_multioutput : bool, optional
        Allows the target array to be multi-dimensional (default is False).
    return_classes : bool, optional
        If True, returns the unique classes instead of a validation boolean.

    Returns
    -------
    bool or array
        If return_classes is False, returns True if the target data is valid for
        multiclass classification, otherwise raises a ValueError.
        If return_classes is True, returns the unique classes in the target data.

    Raises
    ------
    ValueError
        If any of the following conditions are not met:
        - If accept_multioutput is False, the target data must be one-dimensional.
        - All elements in the target array must be non-negative integers.
        - The target array must contain at least two distinct classes.

    Examples
    --------
    >>> from gofast.tools.validator import validate_multiclass_target
    >>> validate_multiclass_target([0, 1, 2, 1, 0])
    array([0, 1, 2, 1, 0])
    >>> validate_multiclass_target([0, 0, 0])
    ValueError: Target array must contain at least two distinct classes.
    >>> validate_multiclass_target([0.5, 1.2, 2.3])
    ValueError: All elements in the target array must be non-negative integers.
    >>> validate_multiclass_target([[1, 2], [2, 3]], accept_multioutput=True, 
    ...                              return_classes=True)
    (array([1, 2, 2, 3]), 3)
    True
    """
    # Convert input to a numpy array and create a copy if modifying data structure
    y = np.asarray(y)
    y_eval = y.copy() if accept_multioutput else y

    # Ensure the array is one-dimensional if multi-output is not accepted
    if not accept_multioutput and y.ndim > 1:
        raise ValueError("Target array must be one-dimensional unless"
                         " multi-output is accepted.")

    # Validate that all elements are non-negative integers
    if not (np.issubdtype(y_eval.dtype, np.integer) and np.all(y_eval >= 0)):
        raise ValueError("All elements in the target array must be non-negative integers.")

    # Flatten the array for unique class check if multi-output is accepted
    if accept_multioutput:
        y_eval = y_eval.flatten()

    # Ensure there are at least two distinct classes
    unique_classes = np.unique(y_eval)
    if unique_classes.size < 2:
        raise ValueError("Target array must contain at least two distinct classes.")

    # Return the original array and the number of unique classes if requested
    if return_classes:
        return y, unique_classes.size

    return y

def validate_sample_weights(weights, y, normalize =False):
    """
    Validates that the sample weights are suitable for use in calculations.

    This function checks that the sample weights are non-negative and match
    the length of the target array `y`. It raises an error if any conditions
    are not met. If a single number is provided as weights, it will be
    converted into an array with repeated values matching the length of `y`.

    Parameters
    ----------
    weights : array-like or number
        The sample weights to be validated. Each weight must be non-negative.
        A single number will be converted to an array with repeated values.
    y : array-like
        The target array that the weights should correspond to. The length
        of `weights` must match the length of `y`.
    normalize : bool, optional
        If True, weights will be normalized to sum to 1. Default is False.
        
    Returns
    -------
    numpy.ndarray
        The validated sample weights as a numpy array.

    Raises
    ------
    ValueError
        If `weights` are not one-dimensional, if any weight is negative,
        or if the length of `weights` does not match the length of `y`.

    Examples
    --------
    >>> frpm gofast.tools.validator import validate_sample_weights
    >>> y = [0, 1, 2, 3]
    >>> weights = [0.1, 0.2, 0.3, 0.4]
    >>> validate_sample_weights(weights, y)
    array([0.1, 0.2, 0.3, 0.4])

    >>> weights = [-0.1, 0.2, 0.3, 0.4]
    >>> validate_sample_weights(weights, y)
    ValueError: Sample weights must be non-negative.

    >>> weights = [0.1, 0.2, 0.3]
    >>> validate_sample_weights(weights, y)
    ValueError: Length of sample weights must match length of y.
    """
    if isinstance(weights, (int, float, np.integer, np.floating)): 
        weights = np.full_like(y, fill_value=weights, dtype=np.float)

    weights = np.asarray(weights)
    y = np.asarray(y)

    # Check if weights are one-dimensional
    if weights.ndim != 1:
        raise ValueError("Sample weights must be one-dimensional.")

    # Check if any weights are negative
    if np.any(weights < 0):
        raise ValueError("Sample weights must be non-negative.")

    # Check if the length of weights matches the length of y
    if weights.size != y.size:
        raise ValueError("Length of sample weights must match length of y.")
     
    weights = normalize_array(weights, normalize=normalize, method="sum") 
   
    return weights  # Return the validated weights as a numpy array


def validate_weights(
        weights, min_value=None, max_value=None, normalize=False,
        allowed_dims=1):
    """
    Validates and optionally normalizes the given weights array to ensure all elements 
    meet specified criteria and the structure is suitable for computations.

    Parameters:
    ----------
    weights : array-like
        Weights to be validated. Can be a list, tuple, or numpy array.
    min_value : float, optional
        Minimum allowable value for weights (inclusive). If None, weights are 
        expected to be non-negative. Explicitly set to a negative value if 
        negative weights are allowed.
    max_value : float or None, optional
        Maximum allowable value for weights (inclusive). If None, no upper 
        limit is enforced.
    normalize : bool, optional
        If True, weights will be normalized to sum to 1. Default is False.
    allowed_dims : int or tuple, optional
        Specifies the allowed dimensions of the weights array. Default is 1 
        (one-dimensional). If a tuple is provided, weights must match one of 
        the dimensions specified in the tuple.

    Returns:
    -------
    np.ndarray
        A numpy array of the validated and optionally normalized weights.

    Raises:
    ------
    ValueError
        If weights contain values outside the specified range, or if the 
        format or dimensions are not suitable.

    Examples:
    --------
    >>> from gofast.tools.validator import validate_weights
    
    >>> validate_weights([0.25, 0.75, 0.5], normalize=True)
    array([0.2, 0.6, 0.4])

    >>> validate_weights([-0.1, 0.9], min_value=0)
    ValueError: Weights must be non-negative.

    >>> validate_weights([0.1, 0.2, 0.7], max_value=0.5)
    ValueError: Weights must not exceed 0.5.

    >>> validate_weights([1, 2, 3], allowed_dims=(1, 2))
    ValueError: Weights dimensions not allowed.
    """
    try:
        weights_array = np.asarray(weights, dtype=float)
    except Exception as e:
        raise ValueError("Weights must be provided in a format that can be"
                         " converted to a numpy array.") from e

    if isinstance(allowed_dims, int):
        allowed_dims = (allowed_dims,)
    if weights_array.ndim not in allowed_dims:
        raise ValueError(f"Weights must have dimensions in {allowed_dims}.")

    # Check if min_value is None and enforce non-negative weights by default
    if min_value is None:
        if np.any(weights_array < 0):
            raise ValueError("Weights must be non-negative unless 'min_value'"
                             " is explicitly set to allow negative values.")
        min_value=0.
    if np.any(weights_array < min_value) or (max_value is not None and np.any(
            weights_array > max_value)):
        raise ValueError(f"Weights must be between {min_value} and"
                         f" {max_value if max_value is not None else '∞'}.")
 
    if normalize:
        if np.sum(weights_array) == 0:
            raise ValueError("Cannot normalize weights because their sum is zero.")
        
        if not is_normalized(weights_array, method ='sum'):
            weights_array /= np.sum(weights_array)

    return weights_array

def is_normalized(arr, method='sum'):
    """
    Checks if the provided array is normalized according to the specified method.

    Parameters:
    ----------
    arr : array-like
        The array to check for normalization.
    method : str, optional
        The method of normalization to check against:
        - '01': Checks if values are between 0 and 1 and if min is 0 and max is 1.
        - 'zscore': Checks if the mean is 0 and the standard deviation is 1.
        - 'sum': Checks if the sum of the array elements is 1.
        Default is 'sum'.

    Returns:
    -------
    bool
        Returns True if the array is normalized according to the specified method,
        False otherwise.

    Examples:
    --------
    >>> arr = np.array([0.25, 0.25, 0.25, 0.25])
    >>> is_normalized(arr, method='sum')
    True

    >>> arr = np.array([0, 0.5, 1])
    >>> is_normalized(arr, method='01')
    True

    >>> arr = np.array([1, -1, 1, -1])
    >>> is_normalized(arr, method='zscore')
    True
    """
    arr = np.asarray(arr, dtype=float)
    method =parameter_validator(
        "method", target_strs={"01", "zscore", "sum"}) ( method)

    if method == '01':
        # Check if all elements are within [0, 1] and max is 1, min is 0
        return np.all((arr >= 0) & (arr <= 1)) and np.isclose(
            np.min(arr), 0) and np.isclose(np.max(arr), 1)
    elif method == 'zscore':
        # Check if mean is approximately 0 and std is approximately 1
        mean = np.mean(arr)
        std = np.std(arr)
        return np.isclose(mean, 0) and np.isclose(std, 1)
    elif method == 'sum':
        # Check if the sum of the elements is approximately 1
        return np.isclose(np.sum(arr), 1)
 
def normalize_array(arr, normalize="auto", method='01'):
    """
    Checks if an array is normalized according to the specified method and 
    normalizes it if required based on the 'normalize' parameter.

    Parameters:
    ----------
    arr : array-like
        The input array to check and potentially normalize.

    normalize : str, optional
        Determines whether to normalize the array:
        - 'auto': Normalize only if the array is not already normalized 
          according to the specified method.
        - True: Always normalize the array regardless of its current state.
        - False: Do not normalize the array, return as is.
        Default is 'auto'.

    method : str, optional
        The normalization method to apply:
        - '01': Normalize the array to have values between 0 and 1.
        - 'zscore': Standardize the array to have a mean of 0 and a standard
          deviation of 1.
        - 'sum': Normalize the array so that the sum of its elements equals 1.
        Default is '01'.

    Returns:
    -------
    np.ndarray
        The normalized array, or the original array if no normalization was applied.

    Raises:
    ------
    ValueError
        If an unknown normalization method is specified or if normalization 
        cannot be performed due to data characteristics (e.g., zero variance).
        
    Examples:
    --------
    >>> import numpy as np 
    >>> from gofast.tools.validator import normalize_array 

    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalized_data = normalize_array(data, normalize=True, method='01')
    >>> print("Normalized between 0 and 1:", normalized_data)
    Normalized between 0 and 1: [0.   0.25 0.5  0.75 1.  ]
    
    >>> zscore_data = normalize_array(data, normalize=True, method='zscore')
    >>> print("Standardized (Z-score):", zscore_data)
    Standardized (Z-score): [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
    
    >>> sum_data = normalize_array(data, normalize=True, method='sum')
    >>> print("Normalized by sum:", sum_data)
    Normalized by sum: [0.06666667 0.13333333 0.2        0.26666667 0.33333333]
    """
    arr = np.asarray(arr, dtype=float)
    is_normed = is_normalized(arr, method=method)
    
    normalize = parameter_validator(
        "normalize", target_strs={True, False, "auto"})( normalize)
    
    if normalize == 'auto':
        normalize = not is_normed

    if normalize:
        if method == '01':
            min_val = np.min(arr)
            max_val = np.max(arr)
            if min_val == max_val:
                raise ValueError("Normalization impossible with zero variance.")
            arr = (arr - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                raise ValueError("Standardization impossible with zero variance.")
            arr = (arr - mean) / std
        elif method == 'sum':
            total = np.sum(arr)
            if total == 0:
                raise ValueError("Normalization by sum impossible with zero sum.")
            arr = arr / total
  
    # If normalization is not required, return the original array
    return arr

def is_binary_class(y, accept_multioutput=False):
    """
    Check whether the target array represents binary classification. Optionally,
    handle multi-output arrays if each output is binary.

    Parameters:
    ----------
    y : array-like
        The target array to be checked. This can be a 1D array for single output
        or a 2D array for multiple outputs if `accept_multioutput` is True.
    accept_multioutput : bool, default False
        If True, the function checks if each column in a multi-dimensional array
        is binary. If False, the function checks if the entire array is binary.

    Returns:
    -------
    bool
        Returns True if `y` is binary (or each output is binary if multi-output
        is accepted), False otherwise.

    Examples:
    --------
    >>> from gofast.tools.validator import is_binary_class 
    >>> is_binary_class([0, 1, 1, 0])
    True
    >>> is_binary_class([[0, 1], [1, 0], [0, 1], [1, 0]], accept_multioutput=True)
    True
    >>> is_binary_class([0, 1, 2, 3])
    False
    """
    y = np.asarray(y)
    y = check_y( y, multi_output= True, y_numeric =True )
    
    if not accept_multioutput:
        # Check if the entire array is binary
        unique_values = np.unique(y)
        return len(unique_values) == 2 and np.all(np.isin(unique_values, [0, 1]))
    
    if y.ndim == 1:
        # If the array is 1D and multioutput is expected, treat it as a single column
        y = y.reshape(-1, 1)

    if y.ndim > 1:
        # Check each column independently
        for column in y.T:
            unique_values = np.unique(column)
            if not (len(unique_values) == 2 and np.all(np.isin(unique_values, [0, 1]))):
                return False
        return True

    return False

def handle_zero_division(
    y_true, 
    zero_division='warn', 
    metric_name='metric computation', 
    epsilon=1e-15,
    replace_with=None
):
    """
    Preprocess input arrays to handle cases where zero could cause division errors
    in subsequent metric computations.

    Parameters
    ----------
    y_true : array-like
        The input data array where zeros might cause division errors.
    zero_division : {'warn', 'raise', 'ignore'}, default 'warn'
        Determines the action to perform when a zero is encountered:
        - 'warn': Issues a warning and replaces zeros with `replace_with` or `epsilon`.
        - 'raise': Raises an error if a zero is found in the input data.
        - 'ignore': Leaves the zeros as they are, useful when the metric calculation
          can handle zeros natively.
    metric_name : str, optional
        Name of the metric for which this preprocessing is being done, to be included
        in warnings or error messages for better context.
    epsilon : float, optional
        Small value to use as default replacement if `replace_with` is None,
        default is 1e-15.
    replace_with : float or None, optional
        A specific value to replace zeros with, if None, `epsilon` is used.

    Returns
    -------
    numpy.ndarray
        The processed array with modifications based on the zero_division strategy.
    
    Raises
    ------
    ValueError
        If `zero_division` is 'raise' and zero is found in `y_true`.

    Notes
    -----
    Using `replace_with` allows for custom behavior when handling zeros, which can
    be tailored to the specific requirements of different metric computations.
    
    Examples 
    ---------
    >>> from gofast.tools.validator import handle_zero_division 
    >>> y_true = [0, 1, 2, 3, 0]
    >>> processed_y_true = handle_zero_division(
        y_true, replace_with=0.001, zero_division='warn')
    >>> print(processed_y_true)

    """
    y_true_processed = np.asarray(y_true, dtype=float)
    zero_division = parameter_validator(
        "zero_division", target_strs=["warn", "raise", "ignore"]) (
            zero_division)
            
    zeros_mask = y_true_processed == 0
    if np.any(zeros_mask):
        if zero_division == 'warn':
            warnings.warn(f"Encountered zero in y_true, which may lead to"
                          f" infinite values or NaNs in {metric_name}.",
                          RuntimeWarning)
            replacement_value = replace_with if replace_with is not None else epsilon
            y_true_processed[zeros_mask] = replacement_value
        elif zero_division == 'raise':
            raise ValueError(f"Encountered zero in y_true, leading to division"
                             f" by zero in {metric_name} computation.")
        elif zero_division == 'ignore':
            pass  # Do nothing, let the calling function handle zeros natively.

    return y_true_processed

def validate_comparison_data(df, /,  alignment="auto"):
    """
    Validates a DataFrame to ensure it is a square matrix and that the index 
    and column names match. Optionally aligns the index names to the column 
    names or vice versa based on the alignment parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to validate.
    alignment : str, default 'auto'
        Controls how the DataFrame's index and columns are aligned if they d
        o not match.
        Options are 'auto', 'index_to_columns', and 'columns_to_index'.
    
    Returns
    -------
    pandas.DataFrame
        The validated and potentially modified DataFrame.
    
    Raises
    ------
    ValueError
        If the DataFrame is not square or if index and column names do not match
        and no suitable alignment option is specified.

    Examples
    --------
    >>> from gofast.tools.validator import validate_comparison_data
    >>> data = pd.DataFrame({
    ...     'A': [1, 0.9, 0.8],
    ...     'B': [0.9, 1, 0.85],
    ...     'C': [0.8, 0.85, 1]
    ... }, index=['A', 'B', 'X'])
    >>> print(validate_comparison_data(data, alignment='index_to_columns'))
    
    >>> data = pd.DataFrame({
    ...     1: [1, 0.9, 0.8],
    ...     2: [0.9, 1, 0.85],
    ...     3: [0.8, 0.85, 1]
    ... }, index=[1, 2, 'X'])
    >>> print(validate_comparison_data(data, alignment='auto'))
    """
    if not isinstance ( df, pd.DataFrame): 
        raise TypeError(f"Performance data expects a DataFrame; got {type(df).__name__!r}")
    # Check if DataFrame is square
    if df.shape[0] != df.shape[1]:
        raise ValueError("DataFrame must be square (equal number of rows and columns).")

    # Check if indices and columns match
    if not df.index.equals(df.columns):
        if alignment == 'index_to_columns':
            df.index = df.columns
        elif alignment == 'columns_to_index':
            df.columns = df.index
        elif alignment == 'auto':
            # Automatically decide which one to use based on data types
            if df.index.dtype == 'object' and df.columns.dtype == 'int64':
                df.index = df.columns
            elif df.columns.dtype == 'object' and df.index.dtype == 'int64':
                df.columns = df.index
            else:
                raise ValueError(
                    "Automatic alignment failed. Index and column names do not match "
                    "and are of the same type. Please specify alignment explicitly."
                )
        else:
            raise ValueError(
                "Invalid alignment option provided. Please choose from 'index_to_columns', "
                "'columns_to_index', or 'auto'."
            )

    return df

def validate_data_types(
    data, expected_type='numeric', 
    nan_policy='omit', 
    return_data=False, 
    error='raise'
    ):
    """
    Checks for mixed data types in a pandas Series or DataFrame and handles
    according to the specified policies. This function is designed to ensure 
    data consistency by verifying that data matches expected type criteria,
    offering options to manage and report any discrepancies.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The data to be checked. This can be a pandas Series or DataFrame.
    expected_type : {'numeric', 'categoric', 'both'}, default 'numeric'
        Specifies the type of data expected:
        
        - 'numeric': All data should be of numeric types (int, float).
        - 'categoric': All data should be categorical, typically strings
          or pandas Categorical datatype.
        - 'both': Any mix of numeric and categorical data is considered valid.
        
    nan_policy : {'raise', 'omit', 'propagate'}, default 'omit'
        Determines how NaN values are handled:
        
        - 'raise': Raises an error if NaN values are found.
        - 'warn': Issues a warning if NaN values are found but proceeds.
        - 'propagate': Continues execution without addressing NaNs.
        
    return_data : bool, default False
        If True, returns a DataFrame or Series (depending on the input) that 
        only includes data rows that conform to the expected_type. If False,
        returns None.
        
    error : {'raise', 'warn'}, default 'raise'
        Configures the error handling behavior when data types do not conform 
        to the expected_type:
        
        - 'raise': Raises a TypeError if mixed types are detected.
        - 'warn': Emits a warning but attempts to continue by filtering 
          non-conforming data if `return_data` is True.

    Returns
    -------
    pd.Series or pd.DataFrame or None
        Depending on `return_data`, this function may return a filtered version
        of `data` that conforms to the `expected_type` or None if `return_data` 
        is False.

    Raises
    ------
    ValueError
        If NaN values are present and `nan_policy` is set to 'error'.
    TypeError
        If data types do not conform to `expected_type` and `error` is set to 'raise'.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.validator import validate_data_types 
    >>> df = pd.DataFrame({'A': [1, 2, 'a', 3.5, np.nan], 'B': ['x', 'y', 'z', None, 't']})
    >>> validate_data_types(df, expected_type='numeric', nan_policy='warn', 
    ...                  return_data=True, error='warn')
    UserWarning: NaN values found in the data, but processing will continue.
    UserWarning: Expected numeric types but found mixed types. 
    Non-numeric data will be ignored.
       A
    0  1.0
    1  2.0
    3  3.5

    Notes
    -----
    The `check_data_types` function is useful in data preprocessing steps,
    particularly when you need to ensure that data fed into a machine learning
    algorithm meets certain type requirements. Handling mixed data types early
    on can prevent issues in model training and evaluation.
    """
    expected_type= parameter_validator(
        "expected_type", target_strs={"numeric", "categoric", "both"}, 
        )(expected_type)
    
    if not isinstance ( data, (pd.Series, pd.DataFrame)): 
        data = build_data_if(data, raise_exception=True, force=True, 
                             input_name="feature")
        
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Handle NaN values according to the nan_policy
    nan_policy= is_valid_policies(nan_policy)
    if nan_policy == 'raise' and data.isnull().any().any():
        raise ValueError("NaN values found in the data.")
    elif nan_policy == 'propagate' and data.isnull().any().any():
        warnings.warn("NaN values found in the data, but processing will continue.")

    def _handle_numeric(data, return_data):
        is_numeric = pd.to_numeric(data, errors='coerce').notna()
        if not is_numeric.all():
            if error == 'raise':
                raise TypeError(
                    "Mixed types detected. Please encode categorical variables first.")
            elif error == 'warn':
                warnings.warn(
                    "Expected numeric types but found mixed types."
                    " Non-numeric data will be ignored.")
                if return_data:
                    return data.loc[is_numeric]
        return data[is_numeric] if return_data else None

    def _handle_categoric(data, return_data):
        is_categoric = data.apply(lambda x: isinstance(x, (str, pd.CategoricalDtype)))
        if not is_categoric.all():
            if error == 'raise':
                raise TypeError("Mixed types detected with unexpected numeric data.")
            elif error == 'warn':
                warnings.warn("Expected categoric types but found numeric data.")
                if return_data:
                    return data[is_categoric]
        return data[is_categoric] if return_data else None

    results = pd.DataFrame()

    for column in data.columns:
        col_data = data[column]
        if expected_type == 'numeric':
            result = _handle_numeric(col_data, return_data)
        elif expected_type == 'categoric':
            result = _handle_categoric(col_data, return_data)
        elif expected_type == 'both':
            if error == 'warn':
                warnings.warn(
                    "Mixed data types found. Be cautious of unintended data type issues.")
            result = col_data if return_data else None
        else:
            raise ValueError("Unsupported expected_type provided. Choose"
                             " 'numeric', 'categoric', or 'both'.")

        if return_data and result is not None:
            results[column] = result

    return results if not results.empty else None

def ensure_2d(X, output_format="auto"):
    """
    Ensure that the input X is converted to a 2-dimensional structure.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        The input data to convert. Can be a list, numpy array, or DataFrame.
    output_format : str, optional
        The format of the returned object. Options are "auto", "array", or "frame".
        "auto" returns a DataFrame if X is a DataFrame, otherwise a numpy array.
        "array" always returns a numpy array.
        "frame" always returns a pandas DataFrame.
        
    Returns
    -------
    ndarray or DataFrame
        The converted 2-dimensional structure, either as a numpy array or DataFrame.
    
    Raises
    ------
    ValueError
        If the `output_format` is not one of the allowed values.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.validator import ensure_2d
    >>> X = np.array([1, 2, 3])
    >>> ensure_2d(X, output_format="array")
    array([[1],
           [2],
           [3]])
    >>> df = pd.DataFrame([1, 2, 3])
    >>> ensure_2d(df, output_format="frame")
       0
    0  1
    1  2
    2  3
    """
    # Check for allowed output_format values
    output_format= parameter_validator(
        "output_format", target_strs=["auto", "array", "frame"]
        ) (output_format)

    # Detect if the input is a DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    
    # Ensure X is at least 2-dimensional
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X[:, np.newaxis]
    elif isinstance(X, pd.DataFrame):
        if X.shape[1] == 0:  # Implies an empty DataFrame or misshapen
            X = X.values.reshape(-1, 1)  # reshape and handle as array
            is_dataframe = False
    else:
        X = np.array(X)  # Convert other types like lists to np.array
        if X.ndim == 1:
            X = X[:, np.newaxis]

    # Decide on return type based on output_format
    if output_format == "array":
        return X if isinstance(X, np.ndarray) else X.values
    elif output_format == "frame":
        return pd.DataFrame(X) if not is_dataframe else X
    else:  # auto handling
        if is_dataframe:
            return X
        return pd.DataFrame(X) if is_dataframe else X

def is_categorical(data, column, strict=False, error='raise'):
    """
    Checks if a specified column in a DataFrame or Series is of 
    a categorical type.
    
    Parameters
    ----------
    data : DataFrame or Series
        The DataFrame or Series to check.
    column : str
        The name of the column to check.
    strict : bool, optional
        If True, only considers pandas CategoricalDtype as categorical. If False,
        also considers object dtype that often represents categorical data.
        Default is False.
    error : str, optional
        Specifies how to handle situations when the column does not exist.
        Options are 'raise', 'warn', or 'ignore'. Default is 'raise'.

    Returns
    -------
    bool
        True if the column is categorical, otherwise False.

    Raises
    ------
    ValueError
        If the column does not exist and error is set to 'raise'.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.validator import is_categorical
    >>> df = pd.DataFrame({
    ...     'fruit': ['Apple', 'Banana', 'Cherry'],
    ...     'count': [10, 20, 15]
    ... })
    >>> df['fruit'] = df['fruit'].astype('category')
    >>> print(is_categorical(df, 'fruit'))
    True
    >>> print(is_categorical(df, 'count'))
    False
    >>> print(is_categorical(df, 'non_existent', error='warn'))
    Warning: Column 'non_existent' not found in the dataframe.
    False
    """
    if column not in data.columns:
        message = f"Column '{column}' not found in the dataframe."
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
        return False  # Return False if error is 'ignore' or 'warn' and column is not found

    col_type = data[column].dtype
    if strict:
        return pd.api.types.is_categorical_dtype(col_type)
    else:
        return pd.api.types.is_categorical_dtype(col_type) or pd.api.types.is_object_dtype(col_type)

def parameter_validator(
        param_name, target_strs, match_method='contains',
        raise_exception=True, **kws):
    """
    Creates a validator function for ensuring a parameter's value matches one 
    of the allowed target strings, optionally applying normalization.

    This higher-order function returns a validator that can be used to check 
    if a given parameter value matches allowed criteria, optionally raising 
    an exception or normalizing the input.

    Parameters
    ----------
    param_name : str
        Name of the parameter to be validated. Used in error messages to 
        indicate which parameter failed validation.
    target_strs : list of str
        A list of acceptable string values for the parameter.
    match_method : str, optional
        The method used to match the input string against the target strings. 
        The default method is 'contains', which checks if the input string 
        contains any of the target strings.
    raise_exception : bool, optional
        Specifies whether an exception should be raised if validation fails. 
        Defaults to True, raising an exception on failure.
    **kws: dict, 
       Keyword arguments passed to :func:`gofast.tools.coreutils.normalize_string`. 
    Returns
    -------
    function
        A closure that takes a single string argument (the parameter value) 
        and returns a normalized version of it if the parameter matches the 
        target criteria. If the parameter does not match and `raise_exception` 
        is True, it raises an exception; otherwise, it returns the original value.

    Examples
    --------
    >>> from gofast.tools.validator import parameter_validator
    >>> validate_outlier_method = parameter_validator(
    ...  'outlier_method', ['z_score', 'iqr'])
    >>> outlier_method = "z_score"
    >>> print(validate_outlier_method(outlier_method))
    'z_score'

    >>> validate_fill_missing = parameter_validator(
    ...  'fill_missing', ['median', 'mean', 'mode'], raise_exception=False)
    >>> fill_missing = "average"  # This does not match but won't raise an exception.
    >>> print(validate_fill_missing(fill_missing))
    'average'

    Notes
    -----
    - The function leverages a custom utility function `normalize_string` 
      from a module named `.coreutils`. This utility is assumed to handle 
      string normalization and matching based on the provided `match_method`.
    - If `raise_exception` is set to False and the input does not match any 
      target string, the input string is returned unchanged. This behavior 
      allows for optional enforcement of the validation rules.
    - The primary use case for this function is to validate and optionally 
      normalize parameters for configuration settings or function arguments 
      where only specific values are allowed.
    """
    from .coreutils import normalize_string 

    def validator(param_value):
        """Validate param value from :func:`~normalize_string`"""
        if param_value:
            return normalize_string(
                param_value, target_strs=target_strs,
                return_target_only=True,
                match_method=match_method, raise_exception=raise_exception, 
                **kws
            )
        return param_value  # Return the original value if it's None or empty

    return validator

def validate_distribution(distribution, elements=None):
    """
    Validates or generates distributions for given elements ensuring the 
    sum equals 1.

    Parameters:
    ----------
    distribution : str, tuple, list
        The distribution to be validated or generated. If 'auto',
        generates a random distribution for the specified number of elements. 
        Can also be a tuple or list representing an explicit distribution.
    elements : int, list of str, optional
        Defines how many elements the distribution should be generated for 
        when 'auto' is used. If a list of strings is provided, its length 
        is used to determine the number of elements.

    Returns:
    -------
    tuple
        A tuple representing the validated or generated distribution.

    Raises:
    ------
    ValueError
        If the provided distribution does not sum to 1 or contains invalid values.
        
    Examples 
    ---------
    >>> from gofast.tools.validator import validate_distribution
    >>> validate_distribution ("auto", elements= [ 'positive', 'neutral', 'negative'])
    (0.1450318690603951, 0.5660028611331361, 0.2889652698064687)
    """
    # Determine the number of elements if a list is provided
    if isinstance(elements, list):
        distributed_elements = len(elements)
    elif isinstance(elements, ( float, int, np.integer, np.floating)):
        distributed_elements = int (elements)
    else:
        raise ValueError("'elements' must be an integer or a list of strings.")

    if str(distribution).lower() == 'auto':
        if distributed_elements is None:
            raise ValueError("'distributed_elements' must be specified when"
                             " using 'auto' distribution.")
        # Generate a random distribution
        random_values = np.random.rand(distributed_elements)
        distribution = tuple(random_values / np.sum(random_values))
    else:
        if not hasattr(distribution, '__iter__') or isinstance(distribution, str):
            # If distribution is not iterable (or a single string), raise an error
            raise ValueError(
                "distribution must be 'auto', a tuple, or a list of distributions")
        
        distribution = tuple(distribution)
        
        if distributed_elements is not None and len(distribution) != distributed_elements:
            raise ValueError(
                f"The distribution must have exactly {distributed_elements} elements")
        
        validated_distribution = []
        for value in distribution:
            if not isinstance(value, (int, float)):
                raise ValueError("All distribution values must be numeric")
            validated_distribution.append(float(value))
        
        if not np.isclose(sum(validated_distribution), 1):
            raise ValueError("The sum of the distribution values must be equal to 1")
        
        distribution = tuple(validated_distribution)
    
    return distribution

def validate_length_range(length_range, sorted_values=True, param_name=None):
    """
    Validates the review length range ensuring it's a tuple with two integers 
    where the first value is less than the second.

    Parameters:
    ----------
    length_range : tuple
        A tuple containing two integers that represent the minimum and maximum
        lengths of reviews.
    sorted_values: bool, default=True 
        If True, the function expects the input length range to be sorted in 
        ascending order and will automatically sort it if not. If False, the 
        input length range is not expected to be sorted, and it will remain 
        as provided.
    param_name : str, optional
        The name of the parameter being validated. If None, the default name 
        'length_range' will be used in error messages.
        
    Returns
    -------
    tuple
        The validated length range.

    Raise
    ------
    ValueError
        If the length range does not meet the requirements.
        
    Examples 
    --------
    >>> from gofast.tools.validator import validate_length_range
    >>> validate_length_range ( (202, 25) )
    (25, 202)
    >>> validate_length_range ( (202,) )
    ValueError: length_range must be a tuple with two elements.
    """
    param_name = param_name or "length_range" 
    if not isinstance(length_range, ( list, tuple) ) or len(length_range) != 2:
        raise ValueError(f"{param_name} must be a tuple with two elements.")

    min_length, max_length = length_range

    if not all(isinstance(x, ( float, int, np.integer, np.floating)
                          ) for x in length_range):
        raise ValueError(f"Both elements in {param_name} must be integers.")
    
    if sorted_values: 
        length_range  = tuple  (sorted ( [min_length, max_length] )) 
        if length_range[0] >= length_range[1]:
            raise ValueError(
                f"The first element in {param_name} must be less than the second.")
    else : 
        length_range = tuple ([min_length, max_length] )
  
    return length_range 
    
def contains_nested_objects(lst, strict=False, allowed_types=None):
    """
    Determines whether a list contains nested objects.

    Parameters
    ----------
    lst : list
        The list to be checked for nested objects.
    strict : bool, optional
        If True, all items in the list must be nested objects. If False, the function
        returns True if any item is a nested object. Default is False.
    allowed_types : tuple of types, optional
        A tuple of types to consider as nested objects. If None, common nested types
        like list, set, dict, and tuple are checked. Default is None.

    Returns
    -------
    bool
        True if the list contains nested objects according to the given parameters,
        otherwise False.

    Notes
    -----
    A nested object is defined as any item within the list that is not a primitive
    data type (e.g., int, float, str) or is a complex structure like lists, sets,
    dictionaries, etc. The function can be customized to check for specific types
    using the `allowed_types` parameter.

    Examples
    --------
    >>> from gofast.tools.validator import contains_nested_objects
    >>> example_list1 = [{1, 2}, [3, 4], {'key': 'value'}]
    >>> example_list2 = [1, 2, 3, [4]]
    >>> example_list3 = [1, 2, 3, 4]
    >>> contains_nested_objects(example_list1)
    True  # non-strict, contains nested objects
    >>> contains_nested_objects(example_list1, strict=True)
    True  # strict, all are nested objects
    >>> contains_nested_objects(example_list2)
    True  # non-strict, contains at least one nested object
    >>> contains_nested_objects(example_list2, strict=True)
    False  # strict, not all are nested objects
    >>> contains_nested_objects(example_list3)
    False  # non-strict, no nested objects
    >>> contains_nested_objects(example_list3, strict=True)
    False  # strict, no nested objects
    """
    if allowed_types is None:
        allowed_types = (list, set, dict, tuple)  # Default nested types
    
    # Function to check if an item is a nested type
    def is_nested(item):
        return isinstance(item, allowed_types)
    
    if strict:
        # Check if all items are nested objects
        return all(is_nested(item) for item in lst)
    else:
        # Check if any item is a nested object
        return any(is_nested(item) for item in lst)
    
def validate_nan_policy(nan_policy, *arrays, sample_weights=None):
    """
    Validates and applies a specified nan_policy to input arrays and
    optionally to sample weights. This utility is essential for pre-processing
    data prior to statistical analyses or model training, where appropriate
    handling of NaN values is critical to ensure accurate and reliable outcomes.

    Parameters
    ----------
    nan_policy : {'propagate', 'raise', 'omit'}
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        'omit' removes rows with NaNs across all input arrays and sample weights.
    *arrays : array-like
        Variable number of input arrays to be validated and adjusted based on
        the specified nan_policy.
    sample_weights : array-like, optional
        Sample weights array to be validated and adjusted in tandem with the
        input arrays according to nan_policy. Defaults to None.

    Returns
    -------
    arrays : tuple of np.ndarray
        Adjusted input arrays, with modifications applied based on nan_policy.
        The order of arrays in the tuple corresponds to the order of input.
    sample_weights : np.ndarray or None
        Adjusted sample weights, modified according to nan_policy if provided.
        Returns None if no sample_weights were provided.

    Raises
    ------
    ValueError
        If `nan_policy` is not among the valid options ('propagate', 'raise',
        'omit') or if NaNs are detected when `nan_policy` is set to 'raise'.

    Notes
    -----
    Handling NaN values is a critical step in data preprocessing, especially
    in datasets with missing values. The choice of nan_policy can significantly
    impact subsequent statistical analysis or predictive modeling by either
    including, excluding, or signaling errors for observations with missing
    values. This function ensures consistent application of the chosen policy
    across multiple datasets, facilitating robust and error-free analyses.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.validator import validate_nan_policy
    >>> y_true = np.array([1, np.nan, 3])
    >>> y_pred = np.array([1, 2, 3])
    >>> sample_weights = np.array([0.5, 0.5, 1.0])
    >>> arrays, sw = validate_nan_policy('omit', y_true, y_pred, 
    ...                                  sample_weights=sample_weights)
    >>> arrays
    (array([1., 3.]), array([1., 3.]))
    >>> sw
    array([0.5, 1. ])
    """
    nan_policy= str(nan_policy).lower() 
    valid_policies = ['propagate', 'raise', 'omit']
    if nan_policy not in valid_policies:
        raise ValueError(
            f"Invalid nan_policy: {nan_policy}. Valid options are {valid_policies}.")

    if nan_policy == 'omit':
        # Find indices that are not NaN in all arrays
        not_nan_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1)
        if sample_weights is not None:
            not_nan_mask &= ~np.isnan(sample_weights)
        
        # Filter out NaNs from all arrays and sample_weights
        arrays = tuple(array[not_nan_mask] for array in arrays)
        if sample_weights is not None:
            sample_weights = sample_weights[not_nan_mask]

    elif nan_policy == 'raise':
        # Check for NaNs in any of the arrays or sample_weights
        if any(np.isnan(array).any() for array in arrays) or (
                sample_weights is not None and np.isnan(sample_weights).any()):
            raise ValueError("Input values contain NaNs and nan_policy is 'raise'.")

    # Return adjusted arrays and sample_weights
    if sample_weights is not None:
        return (*arrays, sample_weights)

    return arrays 

def validate_fit_weights(y, sample_weight=None, weighted_y=False):
    """
    Validate and compute sample weights for fitting.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    weighted_y : bool, default=False
        If True, compute the weighted target values.

    Returns
    -------
    sample_weight : array-like of shape (n_samples,)
        Validated sample weights.

    weighted_y_values : array-like of shape (n_samples,), optional
        Weighted target values if `weighted_y` is True.

    Raises
    ------
    ValueError
        If `sample_weight` is not None and its length does not match the length of `y`.
        If any value in `sample_weight` is negative.

    Notes
    -----
    This function checks the input sample weights, ensuring they are consistent with
    the target values `y`. If `sample_weight` is None, it returns an array of ones
    indicating equal weighting. Otherwise, it validates and returns the given 
    sample weights. If `weighted_y` is True, it also computes and returns the 
    weighted target values.

    Examples
    --------
    >>> import numpy as np 
    >>> y = np.array([0, 1, 1, 0, 1])
    >>> validate_fit_weights(y)
    array([1., 1., 1., 1., 1.])

    >>> sample_weight = np.array([1, 0.5, 1, 1.5, 1])
    >>> validate_fit_weights(y, sample_weight)
    array([1. , 0.5, 1. , 1.5, 1. ])

    >>> validate_fit_weights(y, sample_weight, weighted_y=True)
    (array([1. , 0.5, 1. , 1.5, 1. ]), array([0. , 0.5, 1. , 0. , 1. ]))

    >>> validate_fit_weights(y, weighted_y=True)
    (array([1., 1., 1., 1., 1.]), array([0., 1., 1., 0., 1.]))
    """
    y = check_array(y, ensure_2d=False)
    
    if sample_weight is None:
        sample_weight = np.ones_like(y, dtype=float)
    else:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y, sample_weight)

        if not np.all(sample_weight >= 0):
            raise ValueError("Sample weights must be non-negative")
    
    if weighted_y:
        weighted_y_values = y * sample_weight
        return sample_weight, weighted_y_values
    
    return sample_weight

def is_valid_policies(nan_policy, /, allowed_policies=None):
    """
    Validates the `nan_policy` or any policy argument to ensure it is one
    of the acceptable options (`allowed_policies`). 
    
    Function is used to enforce conformity to predefined NaN handling
    strategies in data processing tasks.

    Parameters
    ----------
    nan_policy : str
        The NaN handling policy to validate. Acceptable values are:
        'propagate' - NaN values are propagated, i.e., no action is taken.
        'omit' - NaN values are omitted before proceeding with the operation.
        'raise' - Raises an error if NaN values are present.

    allowed_policies : list of str, optional
        A list of allowable policy options. If None, 
        defaults to ['propagate', 'omit', 'raise'].

    Raises
    ------
    ValueError
        If `nan_policy` is not one of the valid options in `allowed_policies`.

    Returns
    -------
    str
        The verified `nan_policy` value, confirming it is within 
        allowed parameters.        

    Examples
    --------
    >>> from gofast.tools.validator import is_valid_policies
    >>> is_valid_policies('omit')  # This should pass without an error.
    >>> is_valid_policies('ignore')  # This should raise a ValueError.
      
    """
    # Set default policies if none provided
    if allowed_policies is None:
        allowed_policies = ['propagate', 'omit', 'raise']

    # Ensure allowed_policies is a list even if a single string was provided
    if isinstance(allowed_policies, str):
        allowed_policies = [allowed_policies]

    # Normalize the input policy for comparison
    nan_policy = str(nan_policy).lower().strip()

    # Check if the provided nan_policy is in the list of allowed policies
    if nan_policy not in allowed_policies:
        raise ValueError(
            f"Invalid nan_policy {nan_policy!r}. Choose from {allowed_policies}.")
    
    return nan_policy

def validate_multioutput(value, extra=''):
    """
    Validate the `multioutput` parameter value and handle special cases.

    This function checks if the provided `multioutput` value is one of the
    accepted strings ('raw_values', 'uniform_average', 'raise', 'warn'). It
    warns or raises an error based on the value if it's applicable.

    Parameters
    ----------
    value : str
        The value of the `multioutput` parameter to be validated. Accepted
        values are 'raw_values', 'uniform_average', 'raise', 'warn'.
    extra : str, optional
        Additional text to include in the warning or error message if
        `multioutput` is not applicable.

    Returns
    -------
    str
        The validated `multioutput` value in lowercase if it's one of the
        accepted values. If the value is 'warn' or 'raise', the function
        handles the case accordingly without returning a value.

    Raises
    ------
    ValueError
        If `value` is not one of the accepted strings and is not 'raise'.

    Examples
    --------
    >>> from gofast.tools.validator import validate_multioutput
    >>> validate_multioutput('raw_values')
    'raw_values'

    >>> validate_multioutput('warn', extra=' for Dice Similarity Coefficient')
    # This will warn that multioutput parameter is not applicable for Dice
    # Similarity Coefficient.

    >>> validate_multioutput('raise', extra=' for Gini Coefficient')
    # This will raise a ValueError indicating that multioutput parameter
    # is not applicable for Gini Coefficient.

    >>> validate_multioutput('average')
    # This will raise a ValueError indicating 'average' is an invalid value
    # for multioutput parameter.

    Note
    ----
    The function is designed to ensure API consistency across various metrics
    functions by providing a standard way to handle `multioutput` parameter
    values, especially in contexts where multiple outputs are not applicable.
    """
    valid_values = ['raw_values', 'uniform_average']
    value_lower = str(value).lower()
    if value_lower=="average_uniform": value_lower ="uniform_average"
    if value_lower in ['raise', 'warn']:
        warn_msg = ("The `multioutput` parameter is not applicable" + extra +
                    " as it inherently combines outputs into a single score.")
        if value_lower == 'warn':
            warnings.warn(warn_msg, UserWarning)
        elif value_lower == 'raise':
            raise ValueError(warn_msg)
    elif value_lower not in valid_values:
        raise ValueError(
            "Invalid value for multioutput parameter. Expect 'raw_values' or "
            f"'uniform_average'. Got '{value}'.")

    return value_lower

def ensure_non_negative(*arrays, err_msg=None):
    """
    Ensure that provided arrays contain only non-negative values.

    This function checks each provided array for non-negativity. If any negative
    values are found in any array, it raises a ValueError. This check is crucial
    for computations or algorithms where negative values are not permissible, such
    as logarithmic transformations.

    Parameters
    ----------
    *arrays : array-like
        One or more array-like structures (e.g., lists, numpy arrays). Each array
        is checked for non-negativity.
    err_msg: str, optional 
        Specify a custom error message if negative values are found.
        
    Raises
    ------
    ValueError
        If any array contains negative values, a ValueError is raised with a message
        indicating that only non-negative values are expected.

    Examples
    --------
    >>> y_true = [0, 1, 2, 3]
    >>> y_pred = [0.5, 2.1, 3.5, -0.1]
    >>> ensure_non_negative(y_true, y_pred)
    ValueError: Negative value found. Expect only non-negative values.

    Note
    ----
    The function uses a variable number of arguments, allowing flexibility in the number
    of arrays checked in a single call.
    """
    for i, array in enumerate(arrays, start=1):
        if np.any(np.asarray(array) < 0):
            err_msg = err_msg or ( 
                f"Array at index {i} contains negative values."
                " Expect only non-negative values.")
            raise ValueError(err_msg)
   
def check_epsilon(
    eps, 
    y_true=None, 
    y_pred=None, 
    base_epsilon=1e-10, 
    scale_factor=1e-5
):
    """
    Dynamically determine or validate an epsilon value for numerical computations.

    This function either validates a provided epsilon if it is a numeric value, or 
    calculates an appropriate epsilon dynamically based on the input data. The dynamic
    calculation aims to adjust epsilon based on the scale of the input data, providing
    flexibility and adaptability in algorithms where numerical stability is critical.

    Parameters
    ----------
    eps : {'auto', float}
        The epsilon value to use. If 'auto', the function dynamically determines an
        appropriate epsilon based on `y_true` and `y_pred`. If a float, it validates
        this as the epsilon value.
    y_true : array-like, optional
        True values array. Used in conjunction with `y_pred` to dynamically determine
        epsilon if `eps` is 'auto'. If `None`, this input is ignored.
    y_pred : array-like, optional
        Predicted values array. Used alongside `y_true` for epsilon determination.
        If `None`, this input is ignored.
    base_epsilon : float, optional
        Base epsilon value used as a starting point in dynamic determination. This
        value is adjusted based on the `scale_factor` and the input data to compute
        the final epsilon.
    scale_factor : float, optional
        Scaling factor applied to adjust the base epsilon in relation to the scale
        of the input data. Helps tailor the epsilon to the problem's numerical scale.

    Returns
    -------
    float
        The determined or validated epsilon value. Ensures numerical operations
        are conducted with an appropriate epsilon to avoid division by zero or
        other numerical instabilities.

    Examples
    --------
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1.1, 1.9, 3.05]
    >>> check_epsilon('auto', y_true, y_pred)
    0.00001  # Example output, actual value depends on `determine_epsilon` implementation.

    >>> check_epsilon(1e-8)
    1e-8

    Notes
    -----
    Using 'auto' for `eps` allows algorithms to adapt to different scales of data,
    enhancing numerical stability without manually tuning the epsilon value.
    """
    from .mathex import determine_epsilon 
    # Initialize a list to hold arrays for dynamic epsilon determination
    y_arrays = []
    
    # Convert inputs to numpy arrays and add to y_arrays if they are not None
    if y_true is not None:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_arrays.append(y_true)
    if y_pred is not None:
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_arrays.append(y_pred)

    # Ensure y_true and y_pred have consistent lengths if both are provided
    if y_true is not None and y_pred is not None:
        check_consistent_length(y_true, y_pred)

    # If both arrays are provided, concatenate them for epsilon determination
    if len(y_arrays) == 2:
        y_arrays = [np.concatenate(y_arrays)]
    
    # Dynamically determine epsilon if 'auto', else ensure it's a float
    if str(eps).lower() == 'auto' and y_arrays:
        eps = determine_epsilon(y_arrays[0], base_epsilon=base_epsilon,
                                scale_factor=scale_factor)
    else:
        try: 
            eps = float(eps)
        except ValueError: 
            raise ValueError(f"Epsilon must be 'auto' or convertible to float. Got '{eps}'")
    
    return eps

def _ensure_y_is_valid(y_true, y_pred, **kwargs):
    """
    Validates that the true and predicted target arrays are suitable for further
    processing. This involves ensuring that both arrays are non-empty, of the
    same length, and meet any additional criteria specified by keyword arguments.

    Parameters
    ----------
    y_true : array-like
        The true target values.
    y_pred : array-like
        The predicted target values.
    **kwargs : dict
        Additional keyword arguments to pass to the check_y function for any
        extra validation criteria.

    Returns
    -------
    y_true : array-like
        Validated true target values.
    y_pred : array-like
        Validated predicted target values.

    Raises
    ------
    ValueError
        If the validation checks fail, indicating that the input arrays do not
        meet the required criteria for processing.

    Examples
    --------
    Suppose `check_y` validates that the input is a non-empty numpy array and
    `check_consistent_length` ensures the arrays have the same number of elements.
    Then, usage could be as follows:

    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 2.1, 3.1])
    >>> y_true_valid, y_pred_valid = _ensure_y_is_valid(y_true, y_pred)
    >>> print(y_true_valid, y_pred_valid)
    [1 2 3] [1.1 2.1 3.1]
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Ensure individual array validity
    y_true = check_y(y_true, **kwargs)
    y_pred = check_y(y_pred, **kwargs)

    # Check if the arrays have consistent lengths
    check_consistent_length(y_true, y_pred)

    return y_true, y_pred

def check_classification_targets(
    *y, 
    target_type='numeric', 
    strategy='auto',
    verbose=False
    ):
    """
    Validate that the target arrays are suitable for classification tasks. 
    
    This function is designed to ensure that target arrays (`y`) contain only 
    finite, categorical values, and it raises a ValueError if the targets do 
    not meet the criteria necessary for classification tasks, such as the 
    presence of continuous values, NaNs, or infinite values.
    
    This validation is crucial for preprocessing steps in machine learning 
    pipelines to ensure that the data is appropriate for classification 
    algorithms.

    Parameters
    ----------
    *y : array-like
        One or more target arrays to be validated. The input can be in the 
        form of lists, numpy arrays, or pandas series. Each array is checked 
        individually to ensure it  meets the criteria for classification targets.
        
    target_type : str, optional
        The expected data type of the target arrays. Supported values are 
        'numeric' and 'object'. If 'numeric', the function attempts to 
        convert the target arrays to integers, raising an error if conversion 
        is not possible due to non-numeric values. If 'object', the target 
        arrays are left as numpy arrays of dtype `object`, suitable for 
        categorical classification without conversion. Default is 'numeric'.
        
    strategy : str, optional
        Defines the approach for evaluating if the target arrays are suitable 
        for classification based on their unique values and data types. The 
        'auto' strategy uses heuristic or automatic detection to decide whether 
        target data should be treated as categorical, which is useful for most 
        cases. Custom strategies can be defined to enforce specific validation 
        rules or preprocessing steps based on the nature of the target data 
        (e.g., 'continuous', 'multilabel-indicator', 'unknown'). These custom 
        strategies should align with the outcomes of a predefined 
        `type_of_target` function, allowing for nuanced handling of different 
        target data scenarios. The default value is ``'auto'``, which applies 
        general rules for categorization and numeric conversion where applicable.
        
        If a strategy other than ``'auto'`` is specified, it directly influences 
        how the data is validated and potentially converted, based on the 
        expected or detected type of target data:

        - If 'continuous', the function checks if the data can be used for 
          regression tasks and raises an error for classification use without 
          explicit binning.
        - If 'multilabel-indicator', it validates the data for multilabel 
          classification tasks and ensures appropriate format.
        - If 'unknown', it attempts to validate the data with generic checks, 
          raising errors for any unclear or unsupported data formats.

    verbose : bool, optional
        If set to True, the function prints a message for each target array 
        checked, confirming that it is suitable for classification. This 
        is helpful for debugging and when validating multiple target arrays 
        simultaneously.

    Raises
    ------
    ValueError
        If any of the target arrays contain values unsuitable for classification. This
        includes arrays with continuous values, NaNs, infinite values, or arrays that do
        not represent categorical data properly.

    Examples
    --------
    Using the function with a single array of integer labels:
    
    >>> from gofast.tools.validator import check_classification_targets
    >>> y = [1, 2, 3, 2, 1]
    >>> check_classification_targets(y)
    [array([1, 2, 3, 2, 1], dtype=object)]

    Using the function with multiple arrays, including a mix of integer and 
    string labels:

    >>> y1 = [0, 1, 0, 1]
    >>> y2 = ["spam", "ham", "spam", "ham"]
    >>> check_classification_targets(y1, y2, verbose=True)
    Targets are suitable for classification.
    Targets are suitable for classification.
    [array([0, 1, 0, 1], dtype=object), array(['spam', 'ham', 'spam', 'ham'], dtype=object)]

    Attempting to use the function with an array containing NaN values:

    >>> y_with_nan = [1, np.nan, 2, 1]
    >>> check_classification_targets(y_with_nan)
    ValueError: Target values contain NaN or infinite numbers, which are not 
    suitable for classification.

    Attempting to use the function with a continuous target array:

    >>> y_continuous = np.linspace(0, 1, 10)
    >>> check_classification_targets(y_continuous)
    ValueError: The number of unique values is too high for a classification task.
    Validating and converting a mixed-type target array to numeric:

    >>> y_mixed = [1, '2', 3.0, '4', 5]
    >>> check_classification_targets(y_mixed, target_type='numeric')
    ValueError: Target array at index 0 contains non-numeric values, which 
    cannot be converted to integers: ['2', '4']...

    Validating object target arrays without attempting conversion:

    >>> y_str = ["apple", "banana", "cherry"]
    >>> check_classification_targets(y_str, target_type='object')
    [array(['apple', 'banana', 'cherry'], dtype=object)]
    
    """
    validated_targets = [_check_y(target, strategy=strategy ) for target in y]

    if target_type == 'numeric':
        # Try to convert validated targets to numeric (integer), if possible
        for i, target in enumerate(validated_targets):
            if all(isinstance(item, (int, float, np.integer, np.floating)) for item in target):
                try:
                    # Attempt conversion to integer
                    validated_targets[i] = target.astype(np.int64)
                except ValueError as e:
                    raise ValueError(f"Error converting target array at index {i} to integers. " 
                                     "Ensure all values are numeric and representable as integers. " 
                                     f"Original error: {e}")
            else:
                non_numeric = [item for item in target if not isinstance(
                    item, (int, float, np.integer, np.floating))]
                raise ValueError(f"Target array at index {i} contains non-numeric values, " 
                                 f"which cannot be converted to integers: {non_numeric[:5]}...")
    elif target_type == 'object':
        # If target_type is 'object', no conversion is needed
        # The function ensures they are numpy arrays, which might already suffice
        pass
    else:
        # In case an unsupported target_type is provided
        raise ValueError(f"Unsupported target_type '{target_type}'. Use 'numeric' or 'object'.")

    if verbose:
        print("Targets are suitable for classification.")

    return validated_targets

def _check_y(y, strategy='auto'):
    """
    Validates the target array `y`, ensuring it is suitable for classification 
    or regression tasks based on its content and the specified strategy.

    Parameters:
    - y: array-like, the target array to be validated.
    - strategy: str, specifies how to determine if `y` is categorical or continuous.
      'auto' for automatic detection based on unique values or explicitly using
      `type_of_target` for more nuanced determination.
    """
    from .coreutils import type_of_target 
    # Convert y to a numpy array of objects to handle mixed types
    y = np.array(y, dtype=object)
    
    # Check for NaN or infinite values in numeric data
    numeric_types = 'biufc'  # Numeric types
    if y.dtype.kind in numeric_types:  
        numeric_y = y.astype(float, casting='safe')  # Safely cast numeric types to float
        if not np.all(np.isfinite(numeric_y)):
            raise ValueError("Numeric target values contain NaN or infinite numbers,"
                             " not suitable for classification.")
    else:
        # For non-numeric data, ensure no elements are None or equivalent to np.nan
        if any(el is None or el is np.nan for el in y):
            raise ValueError("Non-numeric target values contain None or NaN,"
                             " not suitable for classification.")
    unique_values = np.unique(y)
    # Apply specific strategy for determining categorization
    if strategy != 'auto':
        # Implement custom logic based on `type_of_target` outcomes
        target_type = type_of_target(y)
        if target_type == 'continuous':
            raise ValueError("Continuous data not suitable for classification"
                             " without explicit binning.")
        elif target_type == "multilabel-indicator":
            raise ValueError("Multilabel-indicator format detected,"
                             " requiring different handling.")
        elif target_type == 'unknown':
            raise ValueError("Unable to determine the target type,"
                             " please check the input data.")
    else:
        # Auto detection based on unique values count
        if unique_values.shape[0] > np.sqrt(len(y)):
            raise ValueError("Automatic strategy detected too many unique values"
                             " for a classification task.")
    
    # Check for non-numeric data convertibility to categorical if not already checked
    if y.dtype.kind not in numeric_types:
        if not all(isinstance(val, (str, bool, int)) for val in unique_values):
            raise ValueError("Target values must be categorical, numeric,"
                             " or convertible to categories.")
    
    return y

def validate_yy(
    y_true, y_pred, 
    expected_type=None, *, 
    validation_mode='strict', 
    flatten=False
    ):
    """
    Validates the shapes and types of actual and predicted target arrays, 
    ensuring they are compatible for further analysis or metrics calculation.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    expected_type : str, optional
        The expected sklearn type of the target ('binary', 'multiclass', etc.).
    validation_mode : str, optional
        Validation strictness. Currently, only 'strict' is implemented,
        which requires y_true and y_pred to have the same shape and match the 
        expected_type.
    flatten : bool, optional
        If True, both y_true and y_pred are flattened to one-dimensional arrays.

    Raises
    ------
    ValueError
        If y_true and y_pred do not meet the validation criteria.

    Returns
    -------
    tuple
        The validated y_true and y_pred arrays, potentially flattened.
    """
    from .coreutils import type_of_target
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if flatten:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    if y_true.ndim != 1 or y_pred.ndim != 1:
        msg = "Both y_true and y_pred must be one-dimensional arrays after optional flattening."
        raise ValueError(msg)

    check_consistent_length(y_true, y_pred)

    if expected_type is not None:
        actual_type_y_true = type_of_target(y_true)
        actual_type_y_pred = type_of_target(y_pred)
        if validation_mode == 'strict' and (
                actual_type_y_true != expected_type or actual_type_y_pred != expected_type
                ):
            msg = (f"Validation failed in strict mode. Expected type '{expected_type}'"
                   f" for both y_true and y_pred, but got '{actual_type_y_true}'"
                  f" and '{actual_type_y_pred}' respectively.")
            raise ValueError(msg)

    return y_true, y_pred

def check_mixed_data_types(data, /) -> bool:
    """
    Checks if the given data (DataFrame or numpy array) contains both numerical 
    and categorical columns.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The data to check. Can be a pandas DataFrame or a numpy array. If `data`
        is a numpy array, it is temporarily converted to a DataFrame for type 
        checking.

    Returns
    -------
    bool
        True if the data contains both numerical and categorical columns, False
        otherwise.

    Examples
    --------
    Using with a pandas DataFrame:
        
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.validator import check_mixed_data_types
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    >>> print(check_mixed_data_types(df))
    True

    Using with a numpy array:

    >>> array = np.array([[1, 'a'], [2, 'b'], [3, 'c']])
    >>> print(check_mixed_data_types(array))
    True

    With data containing only numerical values:

    >>> df_numeric_only = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> print(check_mixed_data_types(df_numeric_only))
    False

    With data containing only categorical values:

    >>> df_categorical_only = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f']})
    >>> print(check_mixed_data_types(df_categorical_only))
    False
    """
    # Convert numpy array to DataFrame if necessary
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Check for the presence of numerical and categorical data types
    has_numerical = any(data.dtypes.apply(lambda dtype: np.issubdtype(dtype, np.number)))
    has_categorical = any(data.dtypes.apply(
        lambda dtype: dtype == 'object' or dtype.name == 'category' or dtype == 'bool'))
    
    return has_numerical and has_categorical

def is_keras_model(model: Any) -> bool:
    """
    Determine whether the provided object is an instance of a Keras model.

    Parameters
    ----------
    model : Any
        The object to be checked.

    Returns
    -------
    bool
        True if the object is an instance of `tf.keras.models.Model` or 
        `tf.keras.Sequential`,False otherwise.
    """
    from ._dependency import import_optional_dependency 
    import_optional_dependency("tensorflow")
    import tensorflow as tf
    return isinstance(model, (tf.keras.models.Model, tf.keras.Sequential))

def has_required_attributes(model: Any, attributes: list[str]) -> bool:
    """
    Check if the model has all required Keras-specific attributes.

    This function is part of the deep validation process to ensure that the
    model not only inherits from Keras model classes but also implements 
    essential methods.

    Parameters
    ----------
    model : Any
        The model object to inspect.
    attributes : list of str
        A list of strings representing the names of the attributes to check for
        in the model.

    Returns
    -------
    bool
        True if the model contains all specified attributes, False otherwise.
    """
    return all(hasattr(model, attr) for attr in attributes)

def validate_dates(
        start_date, end_date, return_as_date_str=False, date_format="%Y-%m-%d"):
    """
    Validates and parses start and end years/dates, with options for output formatting.

    This function ensures the validity of provided start and end years or dates, checks
    if they fall within a reasonable range, and allows the option to return the validated
    years or dates in a specified string format.

    Parameters
    ----------
    start_date : int, float, or str
        The starting year or date. Can be an integer, float (converted to integer),
        or string in "YYYY" or "YYYY-MM-DD" format.
    end_date : int, float, or str
        The ending year or date, with the same format options as `start_date`.
    return_as_date_str : bool, optional
        If True, returns the start and end dates as strings in the specified format.
        Default is False, returning years as integers.
    date_format : str, optional
        The format string for output dates if `return_as_date_str` is True.
        Default format is "%Y-%m-%d".

    Returns
    -------
    tuple
        A tuple of two elements, either integers (years) or strings (formatted dates),
        representing the validated start and end years or dates.

    Raises
    ------
    ValueError
        If the input years or dates are invalid, out of the acceptable range,
        or if the start year/date does not precede the end year/date.

    Examples
    --------
    >>> from gofast.tools.validator import validate_dates
    >>> validate_dates(1999, 2001)
    (1999, 2001)

    >>> validate_dates("1999/01/01", "2001/12/31", return_as_date_str=True)
    ('1999-01-01', '2001-12-31')

    >>> validate_dates("1999", "1998")
    ValueError: The start date/time must precede the end date/time.

    >>> validate_years("1899", "2001")
    ValueError: Years must be within the valid range: 1900 to [current year].

    Notes
    -----
    The function supports flexible input formats for years and dates, including
    handling both slash "/" and dash "-" separators in date strings. It enforces
    logical and chronological order between start and end inputs and allows
    customization of the output format for date strings.
    """
    def parse_year_input(year_input):
        if isinstance(year_input, (int, float)):
            return datetime(int(year_input), 1, 1)
        elif isinstance(year_input, str):
            year_input = year_input.replace("/", "-")
            try:
                return  datetime.strptime(year_input, date_format)
            except ValueError:
                try: 
                    # Fallback to parsing as year only
                    return datetime(int(year_input), 1, 1)
                except TypeError as type_err: 
                    raise TypeError (
                        "Expected int, float, or str for"
                        f" year, got {type(year_input)}."
                        ) from type_err 
                except ValueError as value_err : 
                    raise ValueError (
                        "Check your date data. For datetime value, set `date_format`"
                        " to '%Y-%m-%d %H:%M:%S'") from value_err
        raise TypeError(f"Invalid input '{year_input}'."
                        " Expected format: YYYY or YYYY-MM-DD.")

    start_date, end_date = map(parse_year_input, [start_date, end_date])

    if start_date >= end_date:
        raise ValueError("Start date/time must be earlier than end date/time.")

    if return_as_date_str:
        return start_date.strftime(date_format), end_date.strftime(date_format)

    current_year = datetime.now().year
    for year in (start_date.year, end_date.year):
        if not 1900 <= year <= current_year:
            raise ValueError(f"Year {year} is out of the valid"
                             f" range: 1900 to {current_year}.")

    # Additional validation for non-string return format
    if ( 
        start_date.year == end_date.year 
        and start_date != end_date 
        and not return_as_date_str
        ):
        raise ValueError(
            "Start and end dates are within the same year but not the same date. "
            "Consider using return_as_date_str=True or providing specific dates.")

    return start_date.year, end_date.year




def validate_positive_integer(value, variable_name, include_zero=False, round_float=None):
    """
    Validates whether the given value is a positive integer or zero based 
    on the parameter and rounds float values according to the specified method.

    Parameters:
    ----------
    value : int or float
        The value to validate.
    variable_name : str
        The name of the variable for error message purposes.
    include_zero : bool, optional
        If True, zero is considered a valid value. Default is False.
    round_float : str, optional
        If "ceil", rounds up float values; if "floor", rounds down float values;
        if None, truncates float values to the nearest whole number towards zero.

    Returns:
    -------
    int
        The validated value converted to an integer.

    Raises:
    ------
    ValueError
        If the value is not a positive integer or zero (based on `include_zero`),
        or if the `round_float` parameter is improperly specified.
    """
    import math
    
    # Determine the minimum acceptable value
    min_value = 0 if include_zero else 1

    # Check for proper type and round if necessary
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{variable_name} must be an integer or float.")

    if isinstance(value, float):
        if round_float == "ceil":
            value = math.ceil(value)
        elif round_float == "floor":
            value = math.floor(value)
        elif round_float is None:
            value = int(value)
        else:
            raise ValueError(f"Invalid rounding method '{round_float}'."
                             " Choose 'ceil', 'floor', or None.")
    # if isinstance(value, float) and not value.is_integer():
    #     raise ValueError(f"{variable_name} must be a whole number, got {value}.")
    if value < min_value:
        condition = "a non-negative integer" if include_zero else "a positive integer"
        raise ValueError(f"{variable_name} must be {condition}, got {value}.")

    return int(value)

def validate_and_adjust_ranges(**kwargs):
    """
    Validates and adjusts the provided range tuples to ensure each is
    composed of two numerical values and is sorted in ascending order.

    This function takes multiple range specifications as keyword arguments,
    each expected to be a tuple of two numerical values (min, max). It validates
    the format and contents of each range, adjusting them if necessary to ensure
    that each tuple is ordered as (min, max).

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments where each key is the name of a range (e.g., 'lat_range')
        and its corresponding value is a tuple of two numerical values representing
        the minimum and maximum of that range.

    Returns
    -------
    dict
        A dictionary with the same keys as the input, but with each tuple value
        adjusted to ensure it is in the format (min, max).

    Raises
    ------
    ValueError
        If any provided range tuple does not contain exactly two values, contains
        non-numerical values, or if the min value is not less than the max value.

    Examples
    --------
    >>> from gofast.tools.validator import validate_and_adjust_ranges
    >>> validate_and_adjust_ranges(lat_range=(34.00, 36.00), lon_range=(-118.50, -117.00))
    {'lat_range': (34.00, 36.00), 'lon_range': (-118.50, -117.00)}

    >>> validate_and_adjust_ranges(time_range=(10.0, 0.01))
    {'time_range': (0.01, 10.0)}

    >>> validate_and_adjust_ranges(invalid_range=(1, 'a'))
    ValueError: invalid_range must contain numerical values.

    Notes
    -----
    This function is particularly useful for preprocessing input ranges for
    various analyses, ensuring consistency and correctness of range specifications.
    It automates the adjustment of provided ranges, simplifying the setup process
    for further data processing or modeling tasks.
    """
    adjusted_ranges = {}

    for range_name, range_tuple in kwargs.items():
        if not isinstance(range_tuple, tuple) or len(range_tuple) != 2:
            raise ValueError(f"{range_name} must be a tuple of two values.")

        if not all(isinstance(value, (int, float)) for value in range_tuple):
            raise ValueError(f"{range_name} must contain numerical values.")

        # Ensure the range is in (min, max) format
        min_value, max_value = sorted(range_tuple)
        adjusted_ranges[range_name] = (min_value, max_value)

    return adjusted_ranges

def recheck_data_types(
    data: Union[pd.DataFrame, pd.Series, list, dict], 
    coerce_numeric: bool = True, 
    coerce_datetime: bool = True,
    column_prefix: str = "col", 
    return_as_numpy: Union[bool, str] = "auto"
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Rechecks and coerces column data types in a DataFrame to the most appropriate
    numeric or datetime types if initially identified as objects. It can also handle
    non-DataFrame inputs by attempting to construct a DataFrame before processing.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, list, or dict
        The data to process. If not a DataFrame, an attempt will be made to convert it.
    coerce_numeric : bool, default=True
        If True, tries to convert object columns to numeric data types.
    coerce_datetime : bool, default=True
        If True, tries to convert object columns to datetime data types.
    column_prefix : str, default="col"
        Prefix for column names when constructing a DataFrame from non-DataFrame input.
    return_as_numpy : bool or str, default="auto"
        If True or "auto", converts the DataFrame to a NumPy array upon returning.
        If "auto", the output type matches the input type.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        The processed data, either as a DataFrame or a NumPy array.

    Examples
    --------
    >>> data = {'a': ['1', '2', '3'], 'b': ['2021-01-01', '2021-02-01', 'not a date'], 
                'c': ['1.1', '2.2', '3.3']}
    >>> df = pd.DataFrame(data)
    >>> df = recheck_data_types(df)
    >>> print(df.dtypes)
    a             int64
    b            object  # remains object due to mixed valid and invalid dates
    c           float64
    """
    return_as_numpy= parameter_validator(
        "return_as_numpy", target_strs={"auto", True, False})(return_as_numpy)
    is_frame = True
    if not isinstance(data, pd.DataFrame):
        is_frame = False
        try:
            data = pd.DataFrame(data, columns=[
                column_prefix + str(i) for i in range(len(data))])
        except Exception as e:
            raise ValueError(
                "Failed to construct a DataFrame from the provided data. "
                "Ensure that your input data is structured correctly, such as "
                "a list of lists or a dictionary with equal-length lists. "
                "Alternatively, provide a DataFrame directly.") from e
            
    for column in data.columns:
        if data[column].dtype == 'object':
            if coerce_datetime:
                try:
                    data[column] = pd.to_datetime(data[column])
                    continue  # Skip further processing if datetime conversion is successful
                except (TypeError, ValueError):
                    pass  # Continue if datetime conversion fails

            if coerce_numeric:
                try:
                    data[column] = pd.to_numeric(data[column])
                except ValueError:
                    pass  # Keep as object if conversion fails

    if return_as_numpy == "auto" and not is_frame:
        return_as_numpy = True  # Automatically determine if output should be a NumPy array

    if return_as_numpy is True: # Explicitly set to True since "auto" is True
        return data.to_numpy()

    return data


def validate_keras_model(
        model: Any, custom_check: Optional[Callable[[Any], bool]] = None,
        deep_check: bool = False, raise_exception =False ) -> bool:
    """
    Validates whether a given object is a Keras model and optionally performs 
    additional checks.

    This function provides a mechanism to ensure that an object not only is an 
    instance of a Keras model but also conforms to additional, user-defined 
    criteria if specified. It offers an optional deep check that inspects the 
    model for key Keras methods, enhancing the validation
    process.

    Parameters
    ----------
    model : Any
        The object to validate as a Keras model.
    custom_check : Callable[[Any], bool], optional
        An optional callback function that takes the model as input and returns
        a boolean indicating whether the model passes custom validation criteria. 
        If `None`, no custom validation is performed.
    deep_check : bool, optional
        If True, performs a deep inspection of the model's attributes to ensure
        it supports essential Keras functionality (default is False).
        
    raise_exception : bool, optional
        If True, raises a TypeError when the model fails the validation
        checks, instead of returning False.
    Returns
    -------
    bool
        True if the object is validated as a Keras model and satisfies any 
        specified custom validation criteria. False otherwise.

    Raises
    ------
    ValueError
        If the custom check is provided and raises an exception, indicating 
        failure of the custom validation logic.

    Examples
    --------
    >>> from tensorflow.keras.layers import Dense
    >>> from tensorflow.keras.models import Sequential
    >>> from gofast.tools.validator import  validate_keras_model
    >>> model = Sequential([Dense(2)])

    Validate a simple Keras model without additional checks:
    >>> validate_keras_model(model)
    True

    Validate with a custom check (e.g., model must have more than 1 layer):
    >>> custom_layer_check = lambda m: len(m.layers) > 1
    >>> validate_keras_model(model, custom_check=custom_layer_check)
    False

    Validate with deep inspection:
    >>> validate_keras_model(model, deep_check=True)
    True
    """
    if not is_keras_model(model):
        if raise_exception: 
            raise TypeError("Provided object is not a Keras model.")
        return False 

    if deep_check and not has_required_attributes(
            model, ['fit', 'predict', 'compile', 'summary']):
        if raise_exception: 
            raise TypeError("Model does not support essential Keras functionalities.")
        return False

    if custom_check:
        try:
            return custom_check(model)
        except Exception as e:
            raise ValueError(f"Custom check failed: {e}")
   
    return True

def is_installed(module: str ) -> bool:
    """
    Checks if TensorFlow is installed.

    This function attempts to find the TensorFlow package specification without
    importing the package. It's a lightweight method to verify the presence of
    TensorFlow in the environment.

    Returns
    -------
    bool
        True if TensorFlow is installed, False otherwise.

    Examples
    --------
    >>> from gofast.tools.validator import is_installed 
    >>> print(is_installed("tensorflow"))
    True  # Output will be True if TensorFlow is installed, False otherwise.
    """
    import importlib.util
    module_spec = importlib.util.find_spec(module)
    return module_spec is not None

def is_time_series(data, /, time_col, check_time_interval=False ):
    """
    Check if the provided DataFrame is time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to be checked.
    time_col : str
        The name of the column in `df` expected to represent time.

    Returns
    -------
    bool
        True if `df` is a time series, False otherwise.
        
    Example
    -------
    >>> import pandas as pd 
    >>> df = pd.DataFrame({
        'Date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
        'Value': [1, 2, 3, 4, 5]
    })
    >>> # Should return True if Date column 
    >>> # can be converted to datetime
    >>> print(is_time_series(df, 'Date'))   
 
    """
    if time_col not in data.columns:
        print(f"Time column '{time_col}' not found in DataFrame.")
        return False

    # Check if the column is datetime type or can be converted to datetime
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        try:
            pd.to_datetime(data[time_col])
        except ValueError:
            print(f"Column '{time_col}' does not contain datetime objects.")
            return False

    if check_time_interval: 
        # Optional: Check for regular intervals (commented out by default)
        intervals = pd.to_datetime(data[time_col]).diff().dropna()
        if not intervals.nunique() == 1:
            print("Time intervals are not regular.")
            return False

    return True

def check_is_fitted2(estimator, attributes, *, msg=None):
    """
    Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by looking for attributes set during fitting.
    Typically, these attributes end with an underscore ('_').

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of a scikit-learn estimator.

    attributes : str or list of str
        The attributes to check for. These are typically set in the 'fit' method.

    msg : str, optional
        The message to raise in the NotFittedError. If not provided, a default
        message is used.

    Raises
    ------
    NotFittedError
        If the given attributes are not found in the estimator.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier()
    >>> check_is_fitted(clf, ['feature_importances_'])
    NotFittedError: This RandomForestClassifier instance is not fitted yet.
    """
    from ..exceptions import NotFittedError 
    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    fitted = all([hasattr(estimator, attr) for attr in attributes])

    if not fitted:
        if msg is None:
            cls_name = estimator.__class__.__name__
            msg = ("This %s instance is not fitted yet. Call 'fit' with appropriate "
                   "arguments before using this estimator." % cls_name)

        raise NotFittedError(msg)

def assert_xy_in (
    x, 
    y, *, 
    data=None,
    asarray=True, 
    to_frame=False, 
    columns= None, 
    xy_numeric=False,
    ignore=None, 
    **kws  
    ): 
    """
    Assert the name of x and y in the given data. 
    
    Check whether string arguments passed to x and y are valid in the data, 
    then retrieve the x and y array values. 
    
    Parameters 
    -----------
    x, y : Arraylike 1d or str, str  
       One dimensional arrays. In principle if data is supplied, they must 
       constitute series.  If `x` and `y` are given as string values, the 
       `data` must be supplied. x and y names must be included in the  
       dataframe otherwise an error raises. 
       
    data: pd.DataFrame, 
       Data containing x and y names. Need to be supplied when x and y 
       are given as string names. 
    asarray: bool, default =True 
       Returns x and y as array rather than series. 
    to_frame: bool, default=False, 
       Convert data to a dataframe using either the columns names or 
       the input_names when the keyword parameter ``force=True``.
    columns: list of str, Optional 
       Name of columns to transform the array ( ``data``) to a dataframe. 
    xy_numeric:bool, default=False
       Convert x and y to numeric values. 
    ignore: str, optional 
       It should be 'x' or 'y'. If set the array is ignored and not asserted. 
       
    kws: dict, 
       Keyword arguments passed to :func:`~.array_to_frame`.
       
       
    Returns 
    --------
    x, y : Arraylike 
       One dimensional array or pd.Series 
      
    Examples 
    ---------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.validator import assert_xy_in 
    >>> x, y = np.random.rand(7 ), np.arange (7 ) 
    >>> data = pd.DataFrame ({'x': x, 'y':y} ) 
    >>> assert_xy_in (x='x', y='y', data = data ) 
    (array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
            0.15599452, 0.05808361]),
     array([0, 1, 2, 3, 4, 5, 6]))
    >>> assert_xy_in (x=x, y=y) 
    (array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
            0.15599452, 0.05808361]),
     array([0, 1, 2, 3, 4, 5, 6]))
    >>> assert_xy_in (x=x, y=data.y) # y is a series 
    (array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
            0.15599452, 0.05808361]),
     array([0, 1, 2, 3, 4, 5, 6]))
    >>> assert_xy_in (x=x, y=data.y, asarray =False ) # return y like it was
    (array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
            0.15599452, 0.05808361]),
    0    0
    1    1
    2    2
    3    3
    4    4
    5    5
    6    6
    Name: y, dtype: int32)
    """
    from .coreutils import exist_features
    if to_frame : 
        data = array_to_frame(data , to_frame = True ,  input_name ='Data', 
                              columns =columns , **kws)
    if data is not None: 
        if not hasattr (data, '__array__') and not hasattr(data, 'columns'): 
            raise TypeError(f"Expect a dataframe. Got {type (data).__name__!r}")
            
    if  ( 
            ( isinstance (x, str) or isinstance (y, str))  
            and data is None) : 
        raise TypeError("Data cannot be None when x and y have string"
                        " arguments.")
    if  ( 
            (x is None or y is None) 
            and data is None): 
        raise TypeError ( "Missing x and y. NoneType not supported.") 
        
    if isinstance (x, str): 
        exist_features(data , x ) ; x = data [x ]
    if isinstance (y, str): 
        exist_features(data, y) ; y = data [y]
        
    if hasattr (x, '__len__') and not hasattr(x, '__array__'): 
        x = np.array(x )
    if hasattr (y, '__len__') and not hasattr(y, '__array__'): 
        y = np.array(y )
    
    _validate_input(ignore, x, y, _is_arraylike_1d)

    check_consistent_length(x, y )
    
    if xy_numeric: 
        if ( 
                not _is_numeric_dtype(x, to_array =True ) 
                or not _is_numeric_dtype(y, to_array=True )
                ): 
            raise ValueError ("x and y must be a numeric array.")
            
        x = x.astype (np.float64) 
        y = y.astype (np.float64)
        
    return ( np.array(x), np.array (y) ) if asarray else (x, y )  

def _validate_input(ignore: str, x, y, _is_arraylike_1d):
    """
    Validates that x and y are one-dimensional array-like structures based
    on the ignore parameter.

    Parameters
    ----------
    ignore : str
        Specifies which variable ('x' or 'y') to ignore during validation.
    x, y : array-like
        The variables to be validated.
    _is_arraylike_1d : function
        Function to check if the input is array-like and one-dimensional.

    Raises
    ------
    ValueError
        If the non-ignored variable(s) are not one-dimensional array-like structures.
    """
    validation_checks = {
        'x': lambda: _is_arraylike_1d(y),
        'y': lambda: _is_arraylike_1d(x),
        'both': lambda: _is_arraylike_1d(x) and _is_arraylike_1d(y)
    }

    check = validation_checks.get(ignore, validation_checks['both'])
    if not check():
        if ignore in ['x', 'y']:
            raise ValueError(f"Expected '{'y' if ignore == 'x' else 'x'}' to be"
                             " a one-dimensional array-like structure.")
        else:
            raise ValueError("Expected both 'x' and 'y' to be one-dimensional "
                             "array-like structures.")

def _is_numeric_dtype (o, / , to_array =False ): 
    """ Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype. 
    
    :param o: object, arraylike 
        Object presumed to be an array 
    :param to_array: bool, default=False 
        If `o` is passed as non-array like list or tuple or other iterable 
        object. Setting `to_array` to ``True`` will convert `o` to array. 
    :return: bool, 
        ``True`` if `o` has a numeric dtype and ``False`` otherwise. 
    """ 
    _NUMERIC_KINDS = set('buifc')
    if not hasattr (o, '__iter__'): 
        raise TypeError ("'o' is expected to be an iterable object."
                         f" got: {type(o).__name__!r}")
    if to_array : 
        o = np.array (o )
    if not hasattr(o, '__array__'): 
        raise ValueError (f"Expect type array, got: {type (o).__name__!r}")
    # use NUMERICKIND rather than # pd.api.types.is_numeric_dtype(arr) 
    # for series and dataframes
    return ( o.values.dtype.kind   
            if ( hasattr(o, 'columns') or hasattr (o, 'name'))
            else o.dtype.kind ) in _NUMERIC_KINDS 
        
def _check_consistency_size (ar1, ar2 , /  , error ='raise') :
    """ Check consistency of two arrays and raises error if both sizes 
    are differents. 
    Returns 'False' if sizes are not consistent and error is set to 'ignore'.
    """
    if error =='raise': 
        msg =("Array sizes must be consistent: '{}' and '{}' were given.")
        assert len(ar1)==len(ar2), msg.format(len(ar1), len(ar2))
        
    return len(ar1)==len(ar2) 

def check_consistency_size ( *arrays ): 
    """ Check consistency of array and raises error otherwise."""
    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )
        
def _is_buildin (o, /, mode ='soft'): 
    """ Returns 'True' wether the module is a Python buidling function. 
    
    If  `mode` is ``strict`` only assert the specific predifined-functions 
    like 'str', 'len' etc, otherwise check in the whole predifined functions
    including the object with type equals to 'module'
    
    :param o: object
        Any object for verification 
    :param mode: str , default='soft' 
        mode for asserting object. Can also be 'strict' for the specific 
        predifined build-in functions. 
    :param module: 
    """
    assert mode in {'strict', 'soft'}, f"Unsupports mode {mode!r}, "\
        "expects 'strict'or 'soft'"
    
    return  (isinstance(o, types.BuiltinFunctionType) and inspect.isbuiltin (o)
             ) if mode=='strict' else type (o).__module__== 'builtins' 


def get_estimator_name (estimator , /): 
    """ Get the estimator name whatever it is an instanciated object or not  
    
    :param estimator: callable or instanciated object,
        callable or instance object that has a fit method. 
    
    :return: str, 
        name of the estimator. 
    """
    name =' '
    if hasattr (estimator, '__qualname__') and hasattr(
            estimator, '__name__'): 
        name = estimator.__name__ 
    elif hasattr(estimator, '__class__') and not hasattr (
            estimator, '__name__'): 
        name = estimator.__class__.__name__ 
    return name 

def _is_cross_validated (estimator ): 
    """ Check whether the estimator has already passed the cross validation
     procedure. 
     
    We assume it has the attributes 'best_params_' and 'best_estimator_' 
    already populated.
    
    :param estimator: callable or instanciated object, that has a fit method. 
    :return: bool, 
        estimator has already passed the cross-validation procedure. 
    
    """
    return hasattr(estimator, 'best_estimator_') and hasattr (
        estimator , 'best_params_')


def _check_array_in(obj, /, arr_name):
    """Returns the array from the array name attribute. Note that the singleton 
    array is not admitted. 
    
    This helper function tries to return array from object attribute  where 
    object attribute is the array name if exists. Otherwise raises an error. 
    
    Parameters
    ----------
    obj : object 
       Object that is expected to contain the array attribute.
    Returns
    -------
    X : array
       Array fetched from its name in `obj`. 
    """
    
    type_ = type(obj)
    try : 
        type_name = f"{obj.__module__}.{obj.__qualname__}"
        o_= f" in {obj.__name__!r}"
    except AttributeError:
        type_name = type_.__qualname__
        o_=''
        
    message = (f"Unable to find the name {arr_name!r}"
               f"{o_} from {type_name!r}") 
    
    if not hasattr (obj , arr_name ): 
        raise TypeError (message )
    
    X = getattr ( obj , f"{arr_name}") 

    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            warnings.warn ( 
                "A singleton array %r cannot be considered a valid collection."% X)
            message += f" with shape {X.shape}"
            raise TypeError(message)
        
    return X 

        
def _deprecate_positional_args(func=None, *, version="1.3"):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.
    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default="1.3"
        The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = [
                "{}={}".format(name, arg)
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warnings.warn(
                f"Pass {args_msg} as keyword args. From version "
                f"{version} passing these as positional arguments "
                "will result in an error",
                FutureWarning,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args

def to_dtype_str (arr, /, return_values = False ): 
    """ Convert numeric or object dtype to string dtype. 
    
    This will avoid a particular TypeError when an array is filled by np.nan 
    and at the same time contains string values. 
    Converting the array to dtype str rather than keeping to 'object'
    will pass this error. 
    
    :param arr: array-like
        array with all numpy datatype or pandas dtypes
    :param return_values: bool, default=False 
        returns array values in string dtype. This might be usefull when a 
        series with dtype equals to object or numeric is passed. 
    :returns: array-like 
        array-like with dtype str 
        Note that if the dataframe or serie is passed, the object datatype 
        will change only if `return_values` is set to ``True``, otherwise 
        returns the same object. 
    
    """
    if not hasattr (arr, '__array__'): 
        raise TypeError (f"Expects an array, got: {type(arr).__name__!r}")
    if return_values : 
        if (hasattr(arr, 'name') or hasattr (arr,'columns')):
            arr = arr.values 
    return arr.astype (str ) 

def _is_arraylike_1d (x) :
    """ Returns whether the input is arraylike one dimensional and not a scalar"""
    if not hasattr (x, '__array__'): 
        raise TypeError ("Expects a one-dimensional array, "
                         f"got: {type(x).__name__!r}")
    _is_arraylike_not_scalar(x)
    return _is_arraylike_not_scalar(x) and  (  len(x.shape )< 2 or ( 
        len(x.shape ) ==2 and x.shape [1]==1 )) 

def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)

def _num_features(X):
    """Return the number of features in an array-like X.
    This helper function tries hard to avoid to materialize an array version
    of X unless necessary. For instance, if X is a list of lists,
    this function will return the length of the first element, assuming
    that subsequent elements are all lists of the same length without
    checking.
    Parameters
    ----------
    X : array-like
        array-like to get the number of features.
    Returns
    -------
    features : int
        Number of features
    """
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = f"Unable to find the number of features from X of type {type_name}"
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[1]

    first_sample = X[0]

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__qualname__}"
        raise TypeError(message)

    try:
        # If X is a list of lists, for instance, we assume that all nested
        # lists have the same length without checking or converting to
        # a numpy array to keep this function call as cheap as possible.
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def check_memory(memory):
    """Check that ``memory`` is joblib.Memory-like.
    joblib.Memory-like means that ``memory`` can be converted into a
    joblib.Memory instance (typically a str denoting the ``location``)
    or has the same interface (has a ``cache`` method).
    Parameters
    ----------
    memory : None, str or object with the joblib.Memory interface
        - If string, the location where to create the `joblib.Memory` interface.
        - If None, no caching is done and the Memory object is completely transparent.
    Returns
    -------
    memory : object with the joblib.Memory interface
        A correct joblib.Memory object.
    Raises
    ------
    ValueError
        If ``memory`` is not joblib.Memory-like.
    """
    if memory is None or isinstance(memory, str):
        memory = joblib.Memory(location=memory, verbose=0)
    elif not hasattr(memory, "cache"):
        raise ValueError(
            "'memory' should be None, a string or have the same"
            " interface as joblib.Memory."
            " Got memory='{}' instead.".format(memory)
        )
    return memory


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def has_fit_parameter(estimator, parameter):
    """Check whether the estimator's fit method supports the given parameter.
    Parameters
    ----------
    estimator : object
        An estimator to inspect.
    parameter : str
        The searched parameter.
    Returns
    -------
    is_parameter : bool
        Whether the parameter was found to be a named parameter of the
        estimator's fit method.
    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.tools.validation import has_fit_parameter
    >>> has_fit_parameter(SVC(), "sample_weight")
    True
    """
    return parameter in signature(estimator.fit).parameters


def check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False):
    """Make sure that array is 2D, square and symmetric.
    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.
    Parameters
    ----------
    array : {ndarray, sparse matrix}
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float, default=1e-10
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.
    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.
    Returns
    -------
    array_sym : {ndarray, sparse matrix}
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError(
            "array must be 2-dimensional and square. shape = {0}".format(array.shape)
        )

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ["csr", "csc", "coo"]:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn(
                "Array is not symmetric, and will be converted "
                "to symmetric by average with its transpose.",
                stacklevel=2,
            )
        if sp.issparse(array):
            conversion = "to" + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array

def check_scalar(
    x,
    name,
    target_type,
    *,
    min_val=None,
    max_val=None,
    include_boundaries="both",
):
    """Validate scalar parameters type and value.
    Parameters
    ----------
    x : object
        The scalar parameter to validate.
    name : str
        The name of the parameter to be printed in error messages.
    target_type : type or tuple
        Acceptable data types for the parameter.
    min_val : float or int, default=None
        The minimum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have a lower bound.
    max_val : float or int, default=None
        The maximum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have an upper bound.
    include_boundaries : {"left", "right", "both", "neither"}, default="both"
        Whether the interval defined by `min_val` and `max_val` should include
        the boundaries. Possible choices are:
        - `"left"`: only `min_val` is included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val )`.
        - `"right"`: only `max_val` is included in the valid interval.
          It is equivalent to the interval `( min_val, max_val ]`.
        - `"both"`: `min_val` and `max_val` are included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val ]`.
        - `"neither"`: neither `min_val` nor `max_val` are included in the
          valid interval. It is equivalent to the interval `( min_val, max_val )`.
    Returns
    -------
    x : numbers.Number
        The validated number.
    Raises
    ------
    TypeError
        If the parameter's type does not match the desired type.
    ValueError
        If the parameter's value violates the given bounds.
        If `min_val`, `max_val` and `include_boundaries` are inconsistent.
    """

    def type_name(t):
        """Convert type into humman readable string."""
        module = t.__module__
        qualname = t.__qualname__
        if module == "builtins":
            return qualname
        elif t == numbers.Real:
            return "float"
        elif t == numbers.Integral:
            return "int"
        return f"{module}.{qualname}"

    if not isinstance(x, target_type):
        if isinstance(target_type, tuple):
            types_str = ", ".join(type_name(t) for t in target_type)
            target_type_str = f"{{{types_str}}}"
        else:
            target_type_str = type_name(target_type)

        raise TypeError(
            f"{name} must be an instance of {target_type_str}, not"
            f" {type(x).__qualname__}."
        )

    expected_include_boundaries = ("left", "right", "both", "neither")
    if include_boundaries not in expected_include_boundaries:
        raise ValueError(
            f"Unknown value for `include_boundaries`: {repr(include_boundaries)}. "
            f"Possible values are: {expected_include_boundaries}."
        )

    if max_val is None and include_boundaries == "right":
        raise ValueError(
            "`include_boundaries`='right' without specifying explicitly `max_val` "
            "is inconsistent."
        )

    if min_val is None and include_boundaries == "left":
        raise ValueError(
            "`include_boundaries`='left' without specifying explicitly `min_val` "
            "is inconsistent."
        )

    comparison_operator = (
        operator.lt if include_boundaries in ("left", "both") else operator.le
    )
    if min_val is not None and comparison_operator(x, min_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'>=' if include_boundaries in ('left', 'both') else '>'} {min_val}."
        )

    comparison_operator = (
        operator.gt if include_boundaries in ("right", "both") else operator.ge
    )
    if max_val is not None and comparison_operator(x, max_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'<=' if include_boundaries in ('right', 'both') else '<'} {max_val}."
        )

    return x


def _get_feature_names(X):
    """Get feature names from X.
    Support for other array containers should place its implementation here.
    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.
        - pandas dataframe : The columns will be considered to be feature
          names. If the dataframe contains non-string feature names, `None` is
          returned.
        - All other array containers will return `None`.
    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None

    # extract feature names for support array containers
    if hasattr(X, "columns"):
        feature_names = np.asarray(X.columns, dtype=object)

    if feature_names is None or len(feature_names) == 0:
        return

    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

    # mixed type of string and non-string is not supported
    if len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names only support names that are all strings. "
            f"Got feature names with dtypes: {types}."
        )

    # Only feature names of all strings are supported
    if len(types) == 1 and types[0] == "str":
        return feature_names

def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify if the
    estimator is fitted or not.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance

    NotFittedError
        If the attributes are not found.
    """
    from ..exceptions import NotFittedError 
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})
        
def _check_feature_names_in(estimator, input_features=None, *, generate_names=True):
    """Check `input_features` and generate names if needed.
    Commonly used in :term:`get_feature_names_out`.
    Parameters
    ----------
    input_features : array-like of str or None, default=None
        Input features.
        - If `input_features` is `None`, then `feature_names_in_` is
          used as feature names in. If `feature_names_in_` is not defined,
          then the following input feature names are generated:
          `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
        - If `input_features` is an array-like, then `input_features` must
          match `feature_names_in_` if `feature_names_in_` is defined.
    generate_names : bool, default=True
        Whether to generate names when `input_features` is `None` and
        `estimator.feature_names_in_` is not defined. This is useful for transformers
        that validates `input_features` but do not require them in
        :term:`get_feature_names_out` e.g. `PCA`.
    Returns
    -------
    feature_names_in : ndarray of str or `None`
        Feature names in.
    """

    feature_names_in_ = getattr(estimator, "feature_names_in_", None)
    n_features_in_ = getattr(estimator, "n_features_in_", None)

    if input_features is not None:
        input_features = np.asarray(input_features, dtype=object)
        if feature_names_in_ is not None and not np.array_equal(
            feature_names_in_, input_features
        ):
            raise ValueError("input_features is not equal to feature_names_in_")

        if n_features_in_ is not None and len(input_features) != n_features_in_:
            raise ValueError(
                "input_features should have length equal to number of "
                f"features ({n_features_in_}), got {len(input_features)}"
            )
        return input_features

    if feature_names_in_ is not None:
        return feature_names_in_

    if not generate_names:
        return

    # Generates feature names if `n_features_in_` is defined
    if n_features_in_ is None:
        raise ValueError("Unable to generate feature names without n_features_in_")

    return np.asarray([f"x{i}" for i in range(n_features_in_)], dtype=object)

def _pandas_dtype_needs_early_conversion(pd_dtype):
    """Return True if pandas extension pd_dtype need to be converted early."""
    # Check these early for pandas versions without extension dtypes
    from pandas.api.types import (
        is_bool_dtype,
        # is_sparse,
        is_float_dtype,
        is_integer_dtype,
    )

    if is_bool_dtype(pd_dtype):
        # bool and extension booleans need early converstion because __array__
        # converts mixed dtype dataframes into object dtypes
        return True

    if  isinstance(pd_dtype, pd.SparseDtype ):
        # Sparse arrays will be converted later in `check_array`
        return False

    try:
        from pandas.api.types import is_extension_array_dtype
    except ImportError:
        return False

    # if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype): # deprecated 
    if isinstance(pd_dtype, pd.SparseDtype ) or not is_extension_array_dtype(pd_dtype):
        # Sparse arrays will be converted later in `check_array`
        # Only handle extension arrays for integer and floats
        return False
    elif is_float_dtype(pd_dtype):
        # Float ndarrays can normally support nans. They need to be converted
        # first to map pd.NA to np.nan
        return True
    elif is_integer_dtype(pd_dtype):
        # XXX: Warn when converting from a high integer to a float
        return True

    return False

def _ensure_no_complex_data(array):
    if (
        hasattr(array, "dtype")
        and array.dtype is not None
        and hasattr(array.dtype, "kind")
        and array.dtype.kind == "c"
    ):
        raise ValueError("Complex data not supported\n{}\n".format(array)) 
 
    
def _check_estimator_name(estimator):
    if estimator is not None:
        if isinstance(estimator, str):
            return estimator
        else:
            return estimator.__class__.__name__
    return None

def set_array_back (X, *,  to_frame=False, columns = None, input_name ='X'): 
    """ Set array back to frame, reconvert the Numpy array to pandas series 
    or dataframe. 
    
    Parameters 
    ----------
    X: Array-like 
        Array to convert to frame. 
    columns: str or list of str 
        Series name or columns names for pandas.Series and DataFrame. 
        
    to_frame: str, default=False
        If ``True`` , reconvert the array to frame using the columns ortherwise 
        no-action is performed and return the same array.
    input_name : str, default=""
        The data name used to construct the error message. 
    force: bool, default=False, 
        Force columns creating using the combination ``input_name`` and 
        columns range if `columns` is not supplied. 
    Returns 
    -------
    X, columns : Array-like 
        columns if `X` is dataframe and  name if Series. Otherwwise returns None.  
        
    """
    
    # set_back =('out', 'back','reconvert', 'to_frame', 
    #            'export', 'step back')
    type_col_name = type (columns).__name__
    
    if not  (hasattr (X, '__array__') or sp.issparse (X)): 
        raise TypeError (f"{input_name + ' o' if input_name!='' else 'O'}nly "
                        f"supports array, got: {type (X).__name__!r}")
         
    if hasattr (X, 'columns'): 
        # keep the columns 
        columns = X.columns 
    elif hasattr (X, 'name') :
        # keep the name of series 
        columns = X.name

    if (to_frame 
        and not sp.issparse (X)
        ): 
        if columns is None : 
            raise ValueError ("Name or columns must be supplied for"
                              " frame conversion.")
        # if not string is given as name 
        # check whether the columns contains only one 
        # value and use it as name to skip 
        # TypeError: Series.name must be a hashable type 
        if _is_arraylike_1d(X) : 
            if not isinstance (columns, str ) and hasattr (columns, '__len__') : 
                if len(columns ) > 1: 
                    raise ValueError (
                        f"{input_name} is 1d-array, only pandas.Series "
                        "conversion can be performed while name must be a"
                         f" hashable type: got {type_col_name!r}")
                columns = columns [0]
                
            X= pd.Series (X, name =columns )
        else: 
            # columns is str , reconvert to a list 
            # and check whether the columns match 
            # the shape [1]
            if isinstance (columns, str ): 
                columns = [columns ]
            if not hasattr (columns, '__len__'):
                raise TypeError (" Columns for {input_name!r} expects "
                                  f"a list or tuple. Got {type_col_name!r}")
            if X.shape [1] != len(columns):
                raise ValueError (
                    f"Shape of passed values for {input_name} is"
                    f" {X.shape}. Columns indices imply {X.shape[1]},"
                    f" got {len(columns)}"
                                  ) 
                
            X= pd.DataFrame (X, columns = columns )
        
    return X, columns 

def convert_array_to_pandas(X, *, to_frame=False, columns=None, input_name='X'):
    """
    Converts an array-like object to a pandas DataFrame or Series, applying
    provided column names or series name.

    Parameters
    ----------
    X : array-like
        The array to convert to a DataFrame or Series.
    to_frame : bool, default=False
        If True, converts the array to a DataFrame. Otherwise, returns the array unchanged.
    columns : str or list of str, optional
        Name(s) for the columns of the resulting DataFrame or the name of the Series.
    input_name : str, default='X'
        The name of the input variable; used in constructing error messages.

    Returns
    -------
    pd.DataFrame or pd.Series
        The converted DataFrame or Series. If `to_frame` is False, returns `X` unchanged.
    columns : str or list of str
        The column names of the DataFrame or the name of the Series, if applicable.

    Raises
    ------
    TypeError
        If `X` is not array-like or if `columns` is neither a string nor a list of strings.
    ValueError
        If the conversion to DataFrame is requested but `columns` is not provided,
        or if the length of `columns` does not match the number of columns in `X`.
    """
    # Check if the input is string, which is a common mistake
    if isinstance(X, str):
        raise TypeError(f"The parameter '{input_name}' should be an array-like"
                        " or sparse matrix, but a string was passed.")
    
    # Validate the type of X
    if not (hasattr(X, '__array__') or isinstance(
            X, (np.ndarray, pd.Series, list)) or sp.issparse(X)):
        raise TypeError(f"The parameter '{input_name}' should be array-like"
                        f" or a sparse matrix. Received: {type(X).__name__!r}")
    
    # Preserve existing DataFrame or Series column names
    if hasattr(X, 'columns'):
        columns = X.columns
    elif hasattr(X, 'name'):
        columns = X.name

    if to_frame and not sp.issparse(X):
        if columns is None:
            raise ValueError("Columns must be provided for DataFrame conversion.")

        # Ensure columns is list-like for DataFrame conversion, single string for Series
        if isinstance(columns, str):
            columns = [columns]

        if not hasattr(columns, '__len__') or isinstance(columns, str):
            raise TypeError(f"Columns for {input_name} must be a list or a single string.")

        # Convert to Series or DataFrame based on dimensionality
        if X.ndim == 1 or len(X) == len(columns) == 1:  # 1D array or single-column DataFrame
            X = pd.Series(X, name=columns[0])
        elif X.ndim == 2:  # 2D array to DataFrame
            if X.shape[1] != len(columns):
                raise ValueError(f"Shape of passed values is {X.shape},"
                                 f" but columns implied {len(columns)}")
            X = pd.DataFrame(X, columns=columns)
        else:
            raise ValueError(f"{input_name} cannot be converted to DataFrame with given columns.")

    return X, columns
 
def is_frame (arr, /, df_only =False, raise_exception: bool=False,
              objname=None  ): 
    """ Return bool wether array is a frame ( pd.Series or pd.DataFrame )
    
    To verify whether `arr` is typically a dataframe, set ``df_only =True``. 
    Isolated part of :func:`~.array_to_frame` dedicated to X and y frame
    reconversion validation.
    """
    isf= ( hasattr (arr, '__array__') and (
                (hasattr ( arr, 'name') or hasattr (arr, 'columns'))
                ) if not df_only else ( 
                hasattr (arr, '__array__') and hasattr(arr, 'columns'))
            )
    if not isf and raise_exception : 
        # then check only 
        objname='Expect' if not objname else f'{objname} expects'
        raise TypeError(
            f"{objname} a {'DataFrame' if df_only else 'data frame or series'}."
              f" Got {type(arr).__name__!r}")
    return isf 


def check_array(
    array,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    accept_sparse=False, 
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="",
    to_frame=True,
):

    """Input validation on an array, list, or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.
        
    accept_sparse : str, bool or list/tuple of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.
    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:
        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
          ``force_all_finite`` accepts the string ``'allow-nan'``.
           Accepts `pd.NA` and converts it into `np.nan`
    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.
    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.
    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
        
    to_frame: bool, default=False
        Reconvert array back to pd.Series or pd.DataFrame if 
        the original array is pd.Series or pd.DataFrame.
        
    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    if isinstance(array, np.matrix):
        raise TypeError(
            "np.matrix is not supported. Please convert to a numpy array with "
            "np.asarray. For more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
        )
    xp, is_array_api = get_namespace(array)

    # collect the name or series if 
    # data is pandas series or dataframe.
    # and reconvert by to series or dataframe 
    # array is series or dataframe. 
    array, column_orig = convert_array_to_pandas(array, input_name=input_name)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    pandas_requires_conversion = False
    
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        # throw warning if columns are sparse. If all columns are sparse, then
        # array.sparse exists and sparsity will be preserved (later).
        with suppress(ImportError):
            # from pandas.api.types import is_sparse

            if not hasattr(array, "sparse") and isinstance(array, pd.SparseDtype ):
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        dtypes_orig = list(array.dtypes)
        pandas_requires_conversion = any(
            _pandas_dtype_needs_early_conversion(i) for i in dtypes_orig
        )
        if all(isinstance(dtype_iter, np.dtype) for dtype_iter in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = xp.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if pandas_requires_conversion:
        # pandas dataframe requires conversion earlier to handle extension dtypes with
        # nans
        # Use the original dtype for conversion if dtype is None
        new_dtype = dtype_orig if dtype is None else dtype
        array = array.astype(new_dtype)
        # Since we converted here, we do not need to convert again later
        dtype = None

    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(
                force_all_finite
            )
        )
    estimator_name = _check_estimator_name(estimator)
    #context = " by %s" % estimator_name if estimator is not None else ""
    
    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(
           array,
           accept_sparse=accept_sparse,
           dtype=dtype,
           copy=copy,
           force_all_finite=force_all_finite,
           accept_large_sparse=accept_large_sparse,
           estimator_name=estimator_name,
           input_name=input_name,
       )
       
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter("error", ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in "iu":
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = _asarray_with_order(array, order=order, xp=xp)
                    if array.dtype.kind == "f":
                        _assert_all_finite(
                            array,
                            allow_nan=False,
                            msg_dtype=dtype,
                            estimator_name=estimator_name,
                            input_name=input_name,
                        )
                    array = xp.astype(array, dtype, copy=False)
                else:
                    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            except ComplexWarning as complex_warning:
                raise ValueError(
                    "Complex data not supported\n{}\n".format(array)
                ) from complex_warning
    
        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)
    
        if len(array) ==0: 
           raise ValueError (
               "Found array with 0 length while a minimum of 1 is required." )
        if ensure_2d:
            # If input is scalar raise error
            if  array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array)
                )
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead. "
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample."
                )
    
        if  ( dtype_numeric 
             and ( array.values.dtype.kind if hasattr(array, 'columns') 
                  else array.dtype.kind) 
             in "USV"
             ):
            raise ValueError(
                "dtype='numeric' is not compatible with arrays of bytes/strings."
                "Convert your data to numeric values explicitly instead."
            )
        if not allow_nd and array.ndim >= 3:
            raise ValueError(
                "Found array with dim %d. %s expected <= 2."
                % (array.ndim, estimator_name)
            )
        if force_all_finite:
            _assert_all_finite(
                array,
                input_name=input_name,
                estimator_name=estimator_name,
                allow_nan= force_all_finite == "allow-nan",
            )
        
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required."
                % (n_samples, array.shape, ensure_min_samples)
            )

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required."
                % (n_features, array.shape, ensure_min_features)
            )
              
    
    if copy:
        if xp.__name__ in {"numpy", "numpy.array_api"}:
            # only make a copy if `array` and `array_orig` may share memory`
            if np.may_share_memory(array, array_orig):
                array = _asarray_with_order(
                    array, dtype=dtype, order=order, copy=True, xp=xp
                )
        else:
            # always make a copy for non-numpy arrays
            array = _asarray_with_order(
                array, dtype=dtype, order=order, copy=True, xp=xp
            )
            
    if to_frame:
        array= array_to_frame(
                array,
                to_frame =to_frame , 
                columns = column_orig, 
                input_name= input_name, 
                raise_warning="silence", 
            ) 
    
    return array 

def check_X_y(
    X,
    y,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
    estimator=None,
    to_frame= False, 
):
    """Input validation for standard estimators.
    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.
    Parameters
    ----------
    X : {ndarray, list, sparse matrix}
        Input data.
    y : {ndarray, list, sparse matrix}
        Labels.
    accept_sparse : str, bool or list of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.
    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : {'F', 'C'}, default=None
        Whether an array will be forced to be fortran or c-style.
    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
        does not influence whether y can have np.inf, np.nan, pd.NA values.
        The possibilities are:
        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
          be infinite.
        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.
        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`
    ensure_2d : bool, default=True
        Whether to raise a value error if X is not 2D.
    allow_nd : bool, default=False
        Whether to allow X.ndim > 2.
    multi_output : bool, default=False
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.
    ensure_min_samples : int, default=1
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).
    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.
    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.
    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.
        
    Returns
    -------
    X_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.
    """
    if y is None:
        if estimator is None:
            estimator_name = "estimator"
        else:
            estimator_name = _check_estimator_name(estimator)
        raise ValueError(
            f"{estimator_name} requires y to be passed, but the target y is None"
        )

    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
        input_name="X",
        to_frame=to_frame 
    )

    y = check_y(
        y, 
        multi_output=multi_output, 
        y_numeric=y_numeric, 
        estimator=estimator
        )

    check_consistent_length(X, y)

    return X, y


def check_y(y, 
    multi_output=False, 
    y_numeric=False, 
    input_name ="y", 
    estimator=None, 
    to_frame=False,
    allow_nan= False, 
    ):
    """
    
    Parameters 
    -----------
    multi_output : bool, default=False
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.
    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.
    input_name : str, default="y"
       The data name used to construct the error message. In particular
       if `input_name` is "y".    
    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.
    allow_nan : bool, default=False
       If True, do not throw error when `y` contains NaN.
    to_frame:bool, default=False, 
        reconvert array to its initial type if it is given as pd.Series or
        pd.DataFrame. 
    Returns
    --------
    y: array-like, 
    y_converted : object
        The converted and validated y.
        
    """
    y, column_orig = convert_array_to_pandas(y, input_name= input_name ) 
    if multi_output:
        y = check_array(
            y,
            accept_sparse="csr",
            force_all_finite= True if not allow_nan else "allow-nan",
            ensure_2d=False,
            dtype=None,
            input_name=input_name,
            estimator=estimator,
        )
    else:
        estimator_name = _check_estimator_name(estimator)
        y = _check_y_1d(y, warn=True, input_name=input_name)
        _assert_all_finite(y, input_name=input_name, 
                           estimator_name=estimator_name, 
                           allow_nan=allow_nan , 
                           )
        _ensure_no_complex_data(y)
    if y_numeric and y.dtype.kind == "O":
        y = y.astype(np.float64)
        
    if to_frame: 
        y = array_to_frame (
            y, to_frame =to_frame , 
            columns = column_orig,
            input_name=input_name,
            raise_warning="mute", 
            )
       
    return y

def validate_dtype_selector(dtype_selector: str) -> str:
    """
    Validates and categorizes the dtype_selector using regex, including handling 
    cases where 'only' is specifically included.
    
    Parameters:
    - dtype_selector (str): The input dtype selector string.

    Returns:
    - str: A categorized dtype_selector based on predefined patterns. 
          If 'only' is included,
           the returned category will reflect this to enable specific data 
           type handling.

    Raises:
    - ValueError: If the input dtype_selector does not match any predefined
      category.
    """
    types = [
        "numeric", "numeric_only", "categoric", "categoric_only", 
        "biselect","biselector", "datetime"]
    # Regex patterns for matching dtype_selector categories with an optional 'only'
    numeric_pattern = r"numeric(_only)?"
    categoric_pattern = r"categoric(al|_only)?|categorical"
    datetime_pattern = r"dt|datetime"
    biselect_pattern = r"bi[-_]?selector|biselect|biselector"

    # Check if 'only' is included and modify the category accordingly
    suffix = "_only" if "only" in str(dtype_selector).lower() else ""

    if re.match(numeric_pattern, dtype_selector, re.IGNORECASE):
        return f"numeric{suffix}"
    elif re.match(categoric_pattern, dtype_selector, re.IGNORECASE):
        return f"categoric{suffix}"
    elif re.match(datetime_pattern, dtype_selector, re.IGNORECASE):
        return "datetime"
    elif re.match(biselect_pattern, dtype_selector, re.IGNORECASE):
        return "biselect"

    raise ValueError(
        f"Invalid dtype_selector provided. Valid options are :{types}")
    
def build_data_if(
    data, 
    columns=None, 
    to_frame=True, 
    input_name='data', 
    force=False, 
    raise_warning=True,
    raise_exception=False, 
    coerce_datetime=False, 
):
    """
    Converts input data into a pandas DataFrame if necessary and requested,
    applying specified columns names or generating them if the `force` parameter
    is set.

    Parameters
    ----------
    data : dict, list, tuple, np.ndarray, pd.DataFrame
        The data to potentially convert to a DataFrame. Can be a dictionary,
        list, tuple, NumPy array, or already a pandas DataFrame.
    columns : str or list of str, optional
        The names for the resulting DataFrame columns or the Series name.
    to_frame : bool, default=True
        If True, converts `data` to a DataFrame if it isn't already one.
    input_name : str, default='data'
        The name of the input variable, used for constructing error messages.
    force : bool, default=False
        Forces the conversion of `data` to a DataFrame by generating column names
        based on `input_name` if `columns` are not provided.
    raise_warning : bool, default=True
        If True, raises a warning when conversion requirements are not met.
    raise_exception : bool, default=False
        If True, raises an exception instead of a warning when conversion
        requirements are not met.
    coerce_datetime : bool, default=False
        If True, tries to convert object columns to datetime data types.
    Returns
    -------
    pd.DataFrame
        The converted DataFrame.

    Raises
    ------
    TypeError
        If `data` cannot be converted to a DataFrame based on the provided
        parameters and conditions.
    """
    if isinstance(data, dict):
        data = pd.DataFrame(data)
        columns = list(data.columns)
    elif isinstance(data, (list, tuple)):
        data = np.array(data)
    elif isinstance ( data, pd.Series): 
        data =data.to_frame () 
        
    # Ensure data is two dimensional 
    data = ensure_2d(data)
    # Check if data needs to be converted to a DataFrame
    if to_frame and not isinstance(data, pd.DataFrame):
        if columns is None and not force:
            msg = (f"Conversion of {input_name} to DataFrame requires column names. "
                   "Provide `columns` or set `force=True` to generate them automatically.")
            if raise_exception:
                raise TypeError(msg)
            if raise_warning:
                warnings.warn(msg, UserWarning)
        # Generate column names if forced and not provided
        if force and columns is None:
            columns = [f"{input_name}_{i}" for i in range(data.shape[1])]
            
        data = pd.DataFrame(data, columns=columns)
       
    data =array_to_frame(
        data, columns = columns, 
        to_frame =to_frame, 
        input_name=input_name,
        force =force, 
        )
    if isinstance (data, pd.DataFrame): 
        # re_check data_types 
        data = recheck_data_types(
            data, coerce_datetime=coerce_datetime,
            coerce_numeric=True, return_as_numpy=False
            )
    return data  # Return original data if conditions are not met

def build_data_if2 (
    data: dict|np.ndarray| pd.DataFrame, /, 
    columns =None,  
    to_frame=True,  
    input_name ='data', 
    force=False, 
    **kws
    ): 
    """ Contruct data from dict or array if necessary informations are given
    
    Paramaters 
    -------------
    data: dict, Array-like 
        Array to convert to frame. 
    columns: str or list of str 
        Series name or columns names for pandas.Series and DataFrame. 
        
    to_frame: str, default=False
        If ``True`` , reconvert the array to frame using the a naive columns
        name built from the `input_name` ortherwise no-action is performed 
        and return the same array.
        
    input_name : str, default="Data"
        The data name used to construct the error message. 
        
    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.
        If ``ignore``, silence mode is triggered.
        
    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.
        
    force:bool, default=False
        Force conversion array to a frame is columns is not supplied.
        Use the combinaison, `input_name` and `X.shape[1]` range.

    Return 
    --------
    dataframe constructed. 
    
    """
    if isinstance ( data, dict ) : 
        data = pd.DataFrame ( data)
        columns = list( data.columns)  
    elif isinstance ( data, ( list, tuple)): 
        data = np.array(data )
    
    if not is_frame ( data, df_only=True ): 
        if not to_frame: 
            raise TypeError("Expect a dataframe while columns is missing.")
        if to_frame and not force: 
            raise TypeError(
                "Expect columns to build the data frame or set"
                " `force` to ``True`` to create a temporary frame:"
               f" Got {type(data).__name__!r}.")

    return array_to_frame(
        data, columns = columns, 
        to_frame =to_frame, 
        input_name=input_name,
        force =force, 
        **kws
        )

def array_to_frame(
    X, 
    *, 
    to_frame=False, 
    columns=None, 
    raise_exception=False, 
    raise_warning=True, 
    input_name='', 
    force=False
):
    """
    Validates and optionally converts an array-like object to a pandas DataFrame,
    applying specified column names if provided or generating them if the `force`
    parameter is set.

    Parameters
    ----------
    X : array-like
        The array to potentially convert to a DataFrame.
    columns : str or list of str, optional
        The names for the resulting DataFrame columns or the Series name.
    to_frame : bool, default=False
        If True, converts `X` to a DataFrame if it isn't already one.
    input_name : str, default=''
        The name of the input variable, used for error and warning messages.
    raise_warning : bool, default=True
        If True and `to_frame` is True but `columns` are not provided,
        a warning is issued unless `force` is True.
    raise_exception : bool, default=False
        If True, raises an exception when `to_frame` is True but columns
        are not provided and `force` is False.
    force : bool, default=False
        Forces the conversion of `X` to a DataFrame by generating column names
        based on `input_name` if `columns` are not provided.

    Returns
    -------
    pd.DataFrame or pd.Series
        The potentially converted DataFrame or Series, or `X` unchanged.

    Examples
    --------
    >>> from gofast.tools.validator import array_to_frame
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> X = data.data
    >>> array_to_frame(X, to_frame=True, columns=['sepal_length', 'sepal_width',
                                                  'petal_length', 'petal_width'])
    """
    # Determine if conversion to frame is needed
    if to_frame and not isinstance(X, (pd.DataFrame, pd.Series)):
        # Handle force conversion without provided column names
        if columns is None and force:
            columns = [f"{input_name}_{i}" for i in range(X.shape[1])]
        elif columns is None:
            msg = (
                f"Array '{input_name}' requires column names for conversion to a DataFrame. "
                 "Provide `columns` or set `force=True` to auto-generate column names."
            )
            if raise_exception:
                raise ValueError(msg)
            if raise_warning and raise_warning not in ("silence", "ignore", "mute"):
                warnings.warn(msg)
            return X  # Early return if no columns and not forcing
        
        # Proceed with conversion using the provided or generated column names
        X,_ = convert_array_to_pandas(X, to_frame=True, columns=columns,
                                      input_name=input_name)
    
    return X

def array_to_frame2(
    X, 
    *, 
    to_frame = False, 
    columns = None, 
    raise_exception =False, 
    raise_warning =True, 
    input_name ='', 
    force:bool=False, 
  ): 
    """Added part of `is_frame` dedicated to X and y frame reconversion 
    validation.
    
    Parameters 
    ------------
    X: Array-like 
        Array to convert to frame. 
    columns: str or list of str 
        Series name or columns names for pandas.Series and DataFrame. 
        
    to_frame: str, default=False
        If ``True`` , reconvert the array to frame using the columns orthewise 
        no-action is performed and return the same array.
    input_name : str, default=""
        The data name used to construct the error message. 
        
    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.
        If ``ignore``, warnings silence mode is triggered.
    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.
        
    force:bool, default=False
        Force conversion array to a frame is columns is not supplied.
        Use the combinaison, `input_name` and `X.shape[1]` range.
        
    Returns
    --------
    X: converted array 
    
    Example
    ---------
    >>> from gofast.datasets import fetch_data  
    >>> from gofast.tools.validator import array_to_frame 
    >>> data = fetch_data ('hlogs').frame 
    >>> array_to_frame (data.k.values , 
                        to_frame= True, columns =None, input_name= 'y',
                        raise_warning="silence"
                                ) 
    ... array([nan, nan, nan, ..., nan, nan, nan]) # mute 
    
    """
    
    isf = to_frame ; isf = is_frame( X) 
    
    if ( to_frame 
        and not isf 
        and columns is None 
        ): 
        if force:
            columns =[f"{input_name + str(i)}" for i in range(X.shape[1])]
            isf =True 
        else:
            msg = (f"Array {input_name} is originally not a frame. Frame "
                   "conversion cannot be performed with no column names."
                   ) 
            if raise_exception: 
                raise ValueError (msg)
            if  ( raise_warning 
                 and raise_warning not in ("silence","ignore", "mute")
                 ): 
                warnings.warn(msg )
                
            isf=False 

    elif ( to_frame 
          and columns is not None
          ): 
        isf =True
        
    X, _= convert_array_to_pandas(
        X, 
        to_frame=isf, 
        columns =columns, 
        input_name=input_name
        )
                
    return X  
    
def _check_y_1d(y, *, warn=False, input_name ='y'):
    """Ravel column or 1d numpy array, else raises an error.
    and Isolated part of check_X_y dedicated to y validation
    Parameters
    ----------
    y : array-like
       Input data.
    warn : bool, default=False
       To control display of warnings.
    Returns
    -------
    y : ndarray
       Output data.
    Raises
    ------
    ValueError
        If `y` is not a 1D array or a 2D array with a single row or column.
    """
    xp, _ = get_namespace(y)
    y = xp.asarray(y)
    shape = y.shape
    if len(shape) == 1:
        return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
    
    raise ValueError(f"{input_name} should be a 1d array, got"
                     f" an array of shape {shape} instead.")

def _check_large_sparse(X, accept_large_sparse=False):
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False"""
    if not accept_large_sparse:
        supported_indices = ["int32"]
        if X.getformat() == "coo":
            index_keys = ["col", "row"]
        elif X.getformat() in ["csr", "csc", "bsr"]:
            index_keys = ["indices", "indptr"]
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if indices_datatype not in supported_indices:
                raise ValueError(
                    "Only sparse matrices with 32-bit integer"
                    " indices are accepted. Got %s indices." % indices_datatype
                )
def _ensure_sparse_format(
    spmatrix,
    accept_sparse,
    dtype,
    copy,
    force_all_finite,
    accept_large_sparse,
    estimator_name=None,
    input_name="",
):
    """Convert a sparse matrix to a given format.
    Checks the sparse format of spmatrix and converts if necessary.
    Parameters
    ----------
    spmatrix : sparse matrix
        Input to validate and convert.
    accept_sparse : str, bool or list/tuple of str
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.
    dtype : str, type or None
        Data type of result. If None, the dtype of the input is preserved.
    copy : bool
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan'
        Whether to raise an error on np.inf, np.nan, pd.NA in X. The
        possibilities are:
        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan and pd.NA values in X. Values cannot
          be infinite.
        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.
        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`
    estimator_name : str, default=None
        The estimator name, used to construct the error message.
    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
    Returns
    -------
    spmatrix_converted : sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    _check_large_sparse(spmatrix, accept_large_sparse)

    if accept_sparse is False:
        raise TypeError(
            "A sparse matrix was passed, but dense "
            "data is required. Use X.toarray() to "
            "convert to a dense numpy array."
        )
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError(
                "When providing 'accept_sparse' "
                "as a tuple or list, it must contain at "
                "least one string value."
            )
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError(
            "Parameter 'accept_sparse' should be a string, "
            "boolean or list of strings. You provided "
            "'accept_sparse={}'.".format(accept_sparse)
        )

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn(
                "Can't check %s sparse matrix for nan or inf." % spmatrix.format,
                stacklevel=2,
            )
        else:
            _assert_all_finite(
                spmatrix.data,
                allow_nan=force_all_finite == "allow-nan",
                estimator_name=estimator_name,
                input_name=input_name,
            )
        
    return spmatrix

def _object_dtype_isnan(X):
    return X != X

def _assert_all_finite(
    X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=""
):
    """Like assert_all_finite, but only for ndarray."""

    err_msg=(
        f"{input_name} does not accept missing values encoded as NaN"
        " natively. Alternatively, it is possible to preprocess the data,"
        " for instance by using the imputer transformer like the ufunc"
        " 'soft_imputer' in 'gofast.tools.mlutils.soft_imputer'."
        )
    
    xp, _ = get_namespace(X)

    # if _get_config()["assume_finite"]:
    #     return
    X = xp.asarray(X)

    # for object dtype data, we only check for NaNs (GH-13254)
    if X.dtype == np.dtype("object") and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN. " + err_msg)

    # We need only consider float arrays, hence can early return for all else.
    if X.dtype.kind not in "fc":
        return

    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space `np.isinf/isnan` or custom
    # Cython implementation to prevent false positives and provide a detailed
    # error message.
    with np.errstate(over="ignore"):
        first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return
    # Cython implementation doesn't support FP16 or complex numbers
    # use_cython = (
    #     xp is np and X.data.contiguous and X.dtype.type in {np.float32, np.float64}
    # )
    # if use_cython:
    #     out = cy_isfinite(X.reshape(-1), allow_nan=allow_nan)
    #     has_nan_error = False if allow_nan else out == FiniteStatus.has_nan
    #     has_inf = out == FiniteStatus.has_infinite
    # else:
    has_inf = np.isinf(X).any()
    has_nan_error = False if allow_nan else xp.isnan(X).any()
    if has_inf or has_nan_error:
        if has_nan_error:
            type_err = "NaN"
        else:
            msg_dtype = msg_dtype if msg_dtype is not None else X.dtype
            type_err = f"infinity or a value too large for {msg_dtype!r}"
        padded_input_name = input_name + " " if input_name else ""
        msg_err = f"Input {padded_input_name}contains {type_err}."
        if estimator_name and input_name == "X" and has_nan_error:
            # Improve the error message on how to handle missing values in
            # scikit-learn.
            msg_err += (
                f"\n{estimator_name} does not accept missing values"
                " encoded as NaN natively. For supervised learning, you might want"
                " to consider sklearn.ensemble.HistGradientBoostingClassifier and"
                " Regressor which accept missing values encoded as NaNs natively."
                " Alternatively, it is possible to preprocess the data, for"
                " instance by using an imputer transformer in a pipeline or drop"
                " samples with missing values. See"
                " https://scikit-learn.org/stable/modules/impute.html"
                " You can find a list of all estimators that handle NaN values"
                " at the following page:"
                " https://scikit-learn.org/stable/modules/impute.html"
                "#estimators-that-handle-nan-values"
            )
        elif estimator_name is None and has_nan_error: 
            msg_err += f"\n{err_msg}"
            
        raise ValueError(msg_err)
        
def assert_all_finite(
    X,
    *,
    allow_nan=False,
    estimator_name=None,
    input_name="",
):
    """Throw a ValueError if X contains NaN or infinity.
    Parameters
    ----------
    X : {ndarray, sparse matrix}
        The input data.
    allow_nan : bool, default=False
        If True, do not throw error when `X` contains NaN.
    estimator_name : str, default=None
        The estimator name, used to construct the error message.
    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
    """
    _assert_all_finite(
        X.data if sp.issparse(X) else X,
        allow_nan=allow_nan,
        estimator_name=estimator_name,
        input_name=input_name,
    )

def _generate_get_feature_names_out(estimator, n_features_out, input_features=None):
    """Generate feature names out for estimator using the estimator name as the prefix.
    The input_feature names are validated but not used. This function is useful
    for estimators that generate their own names based on `n_features_out`, i.e. PCA.
    Parameters
    ----------
    estimator : estimator instance
        Estimator producing output feature names.
    n_feature_out : int
        Number of feature names out.
    input_features : array-like of str or None, default=None
        Only used to validate feature names with `estimator.feature_names_in_`.
    Returns
    -------
    feature_names_in : ndarray of str or `None`
        Feature names in.
    """
    _check_feature_names_in(estimator, input_features, generate_names=False)
    estimator_name = estimator.__class__.__name__.lower()
    return np.asarray(
        [f"{estimator_name}{i}" for i in range(n_features_out)], dtype=object
    )

class PositiveSpectrumWarning(UserWarning):
    """Warning raised when the eigenvalues of a PSD matrix have issues
    This warning is typically raised by ``_check_psd_eigenvalues`` when the
    eigenvalues of a positive semidefinite (PSD) matrix such as a gram matrix
    (kernel) present significant negative eigenvalues, or bad conditioning i.e.
    very small non-zero eigenvalues compared to the largest eigenvalue.
    .. versionadded:: 0.22
    """
class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.
    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.
    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.
    .. versionchanged:: 0.18
       Moved from sklearn.tools.validation.
    """