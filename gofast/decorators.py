# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio~@Daniel03 <etanoyau@gmail.com>
"""
Provides a collection of decorators designed to enhance and simplify 
common programming tasks in Python. These decorators offer functionality ranging 
from suppressing output and sanitizing docstrings to appending documentation and 
managing feature importance plots. Each decorator is crafted to be reusable and 
easy to integrate into various projects.

Decorators included in this module:
- `SuppressOutput`: Context manager and decorator for suppressing stdout and 
   stderr messages.
- `SanitizeDocstring`: Cleans and restructures a function's or class's docstring 
   to adhere to the Numpy docstring standard.
- `AppendDocFrom`: Appends a specific section of a function's or class's 
   docstring to another.
- `PlotFeatureImportance`: Decorator for plotting permutation feature importance (PFI) 
   diagrams or dendrogram figures.
- `RedirectToNew`: Redirects calls from deprecated functions or classes to their 
   new implementations.
- `SanitizeDocstring`: Sanitizes and restructures docstrings to fit the Numpy
   docstring format.
- More ...

Each decorator is designed with specific use cases in mind, ranging from 
improving code documentation and readability to controlling the output of 
scripts for cleaner execution logs. Users are encouraged to explore the 
functionalities provided by each decorator to enhance their codebase.

Examples:
    >>> from gofast.decorators import SuppressOutput, SanitizeDocstring, AppendDocFrom

Note:
    While each decorator is designed to be as versatile as possible, users should
    consider their specific needs and test decorators in their environment to ensure
    compatibility and desired outcomes.

Contributions:
    - Various contributors and examples from online resources have inspired 
      these decorators.
"""

from __future__ import print_function 
import os
import re
import sys
import inspect
import warnings
import functools
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from typing import Union, Optional, Callable
from ._gofastlog import gofastlog
_logger = gofastlog.get_gofast_logger(__name__)

__docformat__='restructuredtext'

__all__= [
    'AppendDocFrom',
    'AppendDocReferences',
    'AppendDocSection',
    'CheckGDALData',
    'DataTransformer',
    'Dataify',
    'Deprecated',
    'DynamicMethod',
    'ExportData',
    'Extract1dArrayOrSeries',
    'NumpyDocstring',
    'NumpyDocstringFormatter',
    'PlotFeatureImportance',
    'PlotPrediction',
    'RedirectToNew',
    'SignalFutureChange',
    'SmartProcessor',
    'SuppressOutput',
    'Temp2D',
    'available_if',
    'example_function',
    'isdf',
    'sanitize_docstring',
    'EnsureFileExists',
  ]

class SmartProcessor:
    """
    A decorator class for data processing which selectively excludes specified 
    columns from the processing step and reintegrates them afterward. This is 
    useful for data preprocessing steps like scaling or imputing, where certain
    columns (e.g., identifiers or target variables) should be omitted from 
    the processing.

    Parameters
    ----------
    func : callable, optional
        The function to decorate. If not provided at initialization, it must 
        be provided later as the first positional argument in the call to the 
        decorator instance.
    param_name : str, optional
        The name of the keyword argument in the decorated function that 
        specifies which columns to exclude from processing. 
        Defaults to 'column_to_skip' if not provided.
    fail_silently : bool or 'warn', optional
        Controls the error handling behavior. If `False` (default), errors 
        raise exceptions. If `True`, the original data is returned on error. 
        If set to 'warn', a warning is issued, and the original data is returned.
    to_dataframe : bool, optional
        If `True`, converts the output to a pandas DataFrame, regardless of the
        input type. This is useful when working with NumPy arrays but needing 
        a DataFrame for the result. Defaults to `False`.

    Notes
    -----
    The decorator dynamically adjusts to the data type of the input, supporting
    both pandas DataFrames and NumPy arrays. When applied, it ensures that the
    columns specified for exclusion are not modified by the processing function,
    preserving their original values and positions in the output.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.decorators import SmartProcessor
    >>> @SmartProcessor(to_dataframe=True)
    ... def scale_data(data):
    ...     return (data - data.mean()) / data.std()
    ...
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> print(scale_data(df, column_to_skip=['C']))
           A         B  C
    0 -1.224745 -1.224745  7
    1  0.000000  0.000000  8
    2  1.224745  1.224745  9

    Using the decorator with NumPy arrays while skipping specific indices:
    >>> import numpy as np
    >>> arr = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> result = scale_data(arr, column_to_skip=[2])
    >>> print(result)
    [[-1.22474487 -1.22474487  7.        ]
     [ 0.          0.          8.        ]
     [ 1.22474487  1.22474487  9.        ]]
    """
    def __init__(self, func=None, *, param_name=None, fail_silently=False, 
                 to_dataframe=False):
        self.func = func
        self.param_name = param_name or 'column_to_skip'
        self.fail_silently = fail_silently
        self.to_dataframe = to_dataframe
        if func:
            functools.update_wrapper(self, func)
            
    def __call__(self, *args, **kwargs):
        """
        Call method that makes `SmartProcessor` a callable object which can 
        act as a decorator.
    
        When `SmartProcessor` is used to decorate a function without previously
        being instantiated with a function, it receives the function as the 
        first positional argument and returns a new instance of `SmartProcessor`
        as the decorator. If it was already instantiated with a function, it 
        processes the input data and handles the exclusion and reintegration of 
        specified columns.
    
        Parameters
        ----------
        *args : tuple
            The positional arguments passed to the function. If called on an 
            undecorated function, `args[0]` is expected to be the function to 
            decorate.
        **kwargs : dict
            The keyword arguments passed to the function.
    
        Returns
        -------
        callable or object
            If called without a function, returns a new instance of 
            `SmartProcessor` with the function to decorate. If called with a 
            function, returns the wrapper function that processes the data.
    
        Notes
        -----
        This method handles two scenarios:
        1. Initialization of the decorator with a function to decorate.
        2. Application of the decorator to process data by wrapping the 
        decorated function and optionally excluding specified columns from 
        being processed.
        
        The actual data processing includes error handling according to the 
        'fail_silently' attribute, allowing for warnings or silent failures 
        as configured.
    
        Examples
        --------
        >>> @SmartProcessor(to_dataframe=True)
        ... def scale_data(data):
        ...     return (data - data.mean()) / data.std()
        ...
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3],
        ...     'B': [4, 5, 6],
        ...     'C': [7, 8, 9]
        ... })
        >>> print(scale_data(df, column_to_skip=['C']))
               A         B  C
        0 -1.224745 -1.224745  7
        1  0.000000  0.000000  8
        2  1.224745  1.224745  9
        """
        if not self.func:
            # If the instance is called with a function to decorate, 
            # return a new decorated instance
            return self.__class__(
                args[0], param_name=self.param_name,
                fail_silently=self.fail_silently, 
                to_dataframe=self.to_dataframe
            )
    
        def wrapper(data, *args, **kwargs):
            columns_to_skip = kwargs.get(self.param_name, None)
            if isinstance(columns_to_skip, ( str, int)): 
                columns_to_skip = [columns_to_skip]
            try:
                if columns_to_skip is not None:
                    if isinstance(data, pd.DataFrame):
                        self._check_columns_exist(data, columns_to_skip)
                        data_to_process, skipped_data = data.drop(
                            columns=columns_to_skip), data[columns_to_skip]
                    elif isinstance(data, np.ndarray):
                        self._check_indices_valid(data, columns_to_skip)
                        data_to_process = np.delete(data, columns_to_skip, axis=1)
                        skipped_data = data[:, columns_to_skip]
                    else:
                        raise TypeError(
                            "Data must be a pandas DataFrame or a NumPy array."
                            f" Got {type(data).__name_!r}")
    
                    processed_data = self.func(data_to_process, *args, **kwargs)
    
                    if isinstance(data, pd.DataFrame):
                        result = pd.concat([processed_data, skipped_data], axis=1)
                    elif isinstance(data, np.ndarray):
                        result = self._reintegrate_skipped_numpy(
                            data, processed_data, skipped_data, columns_to_skip)
                    
                    return result if not self.to_dataframe or not isinstance(
                        result, pd.DataFrame
                        ) else self.restore_original_column_order (result)
                  
                else:
                    return self.func(data, *args, **kwargs)
    
            except Exception as e:
                if self.fail_silently == 'warn':
                    warnings.warn(str(e))
                    return data
                elif not self.fail_silently or self.fail_silently=='raise':
                    raise
    
        return wrapper(*args, **kwargs)
    
    
    def _check_columns_exist(self, dataframe, columns):
        """
        Check if the specified columns exist in the dataframe.
    
        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to check for column existence.
        columns : list of str
            List of column names to check in the DataFrame.
    
        Raises
        ------
        ValueError
            If any of the specified columns do not exist in the DataFrame, a 
            ValueError is raised with an appropriate message.
    
        Notes
        -----
        This method is used internally by the SmartProcessor class to ensure 
        that the columns specified to be skipped during processing actually 
        exist in the input DataFrame. This is crucial for preventing runtime 
        errors during data manipulation.
        """
        self._original_columns= dataframe.columns.tolist() 
        if any(col not in dataframe.columns for col in columns):
            raise ValueError("Some columns to skip do not exist in the DataFrame")
    
    def _check_indices_valid(self, array, indices):
        """
        Check if the specified indices are valid for the given NumPy array.
    
        Parameters
        ----------
        array : np.ndarray
            The NumPy array to check indices against.
        indices : list of int
            List of column indices to check in the NumPy array.
    
        Raises
        ------
        ValueError
            If any of the indices are out of the range of the array's second dimension,
            a ValueError is raised.
    
        Notes
        -----
        This method ensures that the indices specified for skipping are within the valid
        range of columns of the NumPy array. It prevents index errors during operations
        that involve slicing or accessing array elements by index.
        """
        if any(index >= array.shape[1] for index in indices):
            raise ValueError("Column index out of range. Expect indexes"
                             f" ranged between [0, {array.shape[1]}).")
    
    def _reintegrate_skipped_numpy(
            self, original_data, processed_data, skipped_data, columns_to_skip):
        """
        Reintegrate skipped data back into the processed NumPy array.
    
        Parameters
        ----------
        original_data : np.ndarray
            The original data from which columns were skipped.
        processed_data : np.ndarray
            The data after processing, missing the skipped columns.
        skipped_data : np.ndarray
            The columns that were skipped during the processing.
        columns_to_skip : list of int
            Indices of the columns that were skipped.
    
        Returns
        -------
        np.ndarray
            A new NumPy array that combines both the processed and skipped data 
            in their original column order.
    
        Notes
        -----
        This method handles the reintegration of skipped columns back into the
        NumPy array after the main processing has been completed. It ensures 
        that the final output maintains the same structure and order as the 
        original input array, which is essential for consistency in data 
        processing pipelines.
        """
        full_data = np.empty_like(original_data)
        j = 0  # Index for processed data columns
        for i in range(original_data.shape[1]):
            if i in columns_to_skip:
                # Place skipped data back in its original position
                full_data[:, i] = skipped_data[:, columns_to_skip.index(i)]
            else:
                # Insert processed data in the remaining positions
                full_data[:, i] = processed_data[:, j]
                j += 1
        return full_data
    
    def reoder_dataframe_columns (self , result): 
        # reoder dataframe columns like the original positions after concatena
        # tion 
        if not self.dataframe or isinstance ( result, pd.DataFrame): 
            return result 
        else: 
            result =pd.DataFrame(result) 
        
        if hasattr (self, '_original_columns'): 
            try :
                # try to place the original columns in order after concatenation. 
                result = result[ self._original_columns]
            except : pass # do nothing 
        
        return  result 
    
    def restore_original_column_order(self, result):
        """
        Restore the column order of a DataFrame to match the original column order
        stored in `_original_columns`. If `result` is not a DataFrame, it attempts to
        convert it into one.
    
        Parameters
        ----------
        result : pd.DataFrame or convertible to pd.DataFrame
            The result DataFrame whose columns need to be reordered.
    
        Returns
        -------
        pd.DataFrame
            DataFrame with columns reordered to match the original order, if possible.
    
        Notes
        -----
        This method relies on the presence of an attribute `_original_columns` which 
        is expected to be a list of column names in their original order. The method 
        only modifies the column order if `result` is a DataFrame and `_original_columns`
        is set.
    
        If the reordering process fails (e.g., due to missing columns), the method 
        fails silently and returns the DataFrame as is without reordering.
    
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'B': [4, 5, 6],
        ...     'A': [1, 2, 3],
        ...     'C': [7, 8, 9]
        ... })
        >>> self._original_columns = ['A', 'B', 'C']
        >>> restored_df = self.restore_original_column_order(df)
        >>> print(restored_df.columns)
        Index(['A', 'B', 'C'], dtype='object')
        """
        
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)
        
        if hasattr(self, '_original_columns'):
            try:
                # Try to reorder columns according to the original order
                result = result[self._original_columns]
            except KeyError:
                # Fails silently if reordering is not possible due to missing columns
                pass
    
        return result
        
            
class DataTransformer:
    """
    A decorator class for transforming the output of functions that return
    pandas DataFrames or Series. It can adjust the return value based on
    specified parameters, including renaming columns, resetting indexes, or
    setting a specific index. This class is useful for ensuring that the
    output of data processing functions conforms to a specific structure or
    naming convention.

    Parameters
    ----------
    name : str, optional
        The name of the keyword argument in the decorated function that 
        contains the data to be transformed. If not specified, the first 
        positional argument is used.
    data_index : int, optional
        The index of the data within the return value if the return is a 
        tuple or list. This parameter is only used if the return value is 
        not a single DataFrame or Series. Default is None, which implies 
        that the first item in the return tuple/list is used in 'lazy' mode.
    reset_index : bool, optional
        If True, the index of the DataFrame or Series is reset. Default is False.
    mode : {'lazy', 'hardworker'}, optional
        The mode of operation. In 'lazy' mode, minimal changes are made to the 
        return value. In 'hardworker' mode, the decorator attempts more 
        extensive transformations. Default is 'lazy'.
    verbose : bool, optional
        If True, the decorator will print information about the transformations
        it performs and any errors or warnings. Default is False.
    set_index : bool, optional
        If True and `original_attrs` has an 'index', the decorator will set 
        this index to the return value. Default is False.
    rename_columns : bool, optional
        If True and `original_attrs` has 'columns', the decorator will rename 
        the columns of the return value. This is only applicable if the return 
        value is a DataFrame. Default is False.

    Examples
    --------
    Use as a decorator to automatically convert the return value of a function 
    to a DataFrame and rename columns based on a predefined structure:

    >>> import pandas as pd
    >>> from gofast.decorators import DataTransformer
    >>> @DataTransformer(rename_columns=True, verbose=True)
    ... def process_data():
    ...     return pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    ...
    >>> df = process_data()
    DataTransformer: Finished processing the result.
    
    The `process_data` function will return a DataFrame with columns renamed 
    according to `original_attrs`, if they were collected and `rename_columns`
    was set to True.

    """
    def __init__(
        self, 
        name=None, 
        data_index=None, 
        reset_index=False, 
        mode='lazy', 
        verbose=False, 
        set_index=False, 
        rename_columns=False
    ):
        self.name = name
        self.data_index = data_index
        self.reset_index = reset_index
        self.mode = mode
        self.verbose = verbose
        self.set_index = set_index
        self.rename_columns = rename_columns
        self.original_attrs = {}
         
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-function execution: Data collection and attribute setup
            if args:
                self._collect_data_attributes(*args, **kwargs)
            result = func(*args, **kwargs)
            # Post-function execution: Data re-construction and manipulation
            result = self._reconstruct_data(result)
            if self.verbose: 
                print( "DataTransformer: Finished processing the result.")
            return result
        return wrapper
    
    def _collect_data_attributes(self, *args, **kwargs):
        if self.name:
            data = kwargs.get(self.name)
        else:
            data = args[0]  # Use the first positional argument if name is not specified
        # Use tolist() to ensure JSON serializability if needed
        if isinstance(data, pd.DataFrame):
            self.original_attrs['columns'] = data.columns.tolist()  
            self.original_attrs['index'] = data.index.tolist()
        elif isinstance(data, pd.Series):
            self.original_attrs['name'] = data.name
            self.original_attrs['index'] = data.index.tolist()

    def _reconstruct_data(self, result):
        is_tuple_result = isinstance(result, (tuple, list))
        data_index = self.data_index if self.data_index is not None else 0
        
        if is_tuple_result:
            if data_index >= len(result):
                if self.verbose:
                    print(f"DataTransformer: Data position {data_index} is out"
                          f" of range for the result size {len(result)}.")
                return result
            data = result[data_index]
        else:
            data = result
        
        # In lazy mode and data is already in the correct format, no need to reconstruct
        if self.mode == 'lazy' and isinstance(data, (pd.DataFrame, pd.Series)):
            return result

        data = self._convert_and_adjust_data(data)
        
        # Re-insert transformed data into the original result structure if needed
        if is_tuple_result:
            result = list(result)  # Convert to list for mutability
            result[data_index] = data
            result = tuple(result)  # Convert back if originally a tuple
        else:
            result = data
        
        return result

    def _convert_and_adjust_data(self, data):
        if isinstance(data, np.ndarray):
            data = self._convert_ndarray_to_pandas(data)
        # Add additional conversion logic here for other data 
        # types like lists or dictionaries

        if isinstance(data, pd.DataFrame) and self.rename_columns:
            data = self._rename_columns(data)

        if (isinstance(data, (pd.DataFrame, pd.Series)) and self.set_index) or self.reset_index:
            data = self._apply_index_and_name_settings(data)

        return data

    def _convert_ndarray_to_pandas(self, ndarray):
        try:
            if ndarray.ndim == 1:
                return pd.Series(ndarray, name=self.original_attrs.get('name'))
            elif ndarray.ndim > 1:
                return pd.DataFrame(ndarray, columns=self.original_attrs.get('columns'))
        except Exception as e:
            if self.verbose:
                print(f"DataTransformer: Error converting numpy array to DataFrame/Series - {e}")
        return ndarray

    def _rename_columns(self, dataframe):
        try:
            dataframe.columns = self.original_attrs['columns']
        except Exception as e:
            if self.verbose:
                print(f"DataTransformer: Error renaming columns - {e}")
        return dataframe

    def _apply_index_and_name_settings(self, data):
        if self.reset_index:
            data.reset_index(drop=True, inplace=True)
        elif self.set_index:
            try:
                data.index = self.original_attrs['index']
            except Exception as e:
                if self.verbose:
                    print(f"DataTransformer: Error setting index - {e}")
        return data
    
class Extract1dArrayOrSeries:
    """
    A decorator and callable that preprocesses input data to ensure it is 
    provided to the decorated/called function as a one-dimensional NumPy 
    array or Pandas Series.

    This utility is designed to facilitate data extraction and conversion 
    from various input formats (lists, dictionaries, Pandas DataFrames, 
    and NumPy ndarrays) into a one-dimensional array or series. It is 
    particularly useful for functions expecting standardized input data 
    formats. The class supports dynamic parameter updates for `column`, 
    `index`, `axis`, and `verbose` when used as a decorator.

    Parameters
    ----------
    func : callable, optional
        The function to be decorated. If None, the instance acts as a 
        factory for partials with preset parameters.
    column : str or int, optional
        Specifies which column to extract from the input if it is a DataFrame
        or multidimensional ndarray. Use an integer for index selection or a 
        string for a DataFrame column name.
    index : int, optional
        Specifies which row to extract from the input if it is a DataFrame 
        or ndarray.
    axis : int, optional
        Specifies the axis along which to extract a one-dimensional array 
        from an ndarray. Valid values are 0 or 1.
    method : {'strict', 'soft'}, default 'strict'
        Specifies the behavior when a specified column or index is not found,
        or when no specification is provided. 'soft' uses the first column 
        if the specified one is missing without raising an error.
    as_series : bool, default False
        Determines whether the output should be a Pandas Series. If False, 
        the output is a NumPy array.
    verbose : int, default 0
        Controls the verbosity of the process. A value greater than 0 
        activates verbose output.
    squeeze_arr : bool, default True
        Determines whether to squeeze the input array to one dimension. 
        Applicable to ndarrays only.

    Returns
    -------
    The decorator returns the modified function with input data processed 
    according to the specified parameters.

    Raises
    ------
    TypeError
        If the input is not a list, dictionary, Pandas DataFrame, or 
        NumPy ndarray.
    ValueError
        If the specified conditions (e.g., column/index out of range, 
        incorrect axis specification) are not met.

    Examples
    --------
    Using as a decorator to ensure input data is a one-dimensional array:

    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.decorators import Extract1dArrayOrSeries
    >>> @Extract1dArrayOrSeries(column=0, as_series=True, verbose=1)
    >>> def compute_average(data):
    ...     return data.mean()

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> print(compute_average(df))

    Dynamically overriding decorator parameters in function call:

    >>> @Extract1dArrayOrSeries(column='A', as_series=True)
    >>> def summarize_data(data):
    ...     return {'mean': data.mean(), 'std': data.std()}

    >>> new_df = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
    >>> # Overriding the 'column' parameter dynamically
    >>> print(summarize_data(new_df, column='B'))
    
    Using as a callable:

    >>> def process_data(data):
    ...     # Expect data to be a one-dimensional NumPy array or Pandas Series
    ...     return np.mean(data)

    >>> decorated_function = Extract1dArrayOrSeries(process_data, column=0, as_series=False)
    >>> ndarray = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print(decorated_function(ndarray))
    """
    def __init__(
        self, 
        func=None, *, 
        column=None, 
        index=None, 
        axis=None, 
        method='strict',
        as_series=False, 
        verbose=0, 
        squeeze_arr=True
        ):
        self.func = func
        self.column = column
        self.index = index
        self.axis = axis
        self.method = method
        self.as_series = as_series
        self.verbose = verbose
        self.squeeze_arr = squeeze_arr
        
        if func is not None:
            functools.wraps(func)(self)

    def __call__(self, *args, **kwargs):
        # Dynamically update only specific 
        # parameters if they are explicitly passed
        dynamic_params = ['column', 'index', 'axis', 'verbose']
        for param in dynamic_params:
            if param in kwargs:
                setattr(self, param, kwargs[param])
        if self.func:
            # Proceed with the possibly updated instance attributes
            return self._wrapper(*args, **kwargs)
        else:
            return self._partial(*args, **kwargs)

    def _partial(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return self.__class__(
                func, column=self.column, 
                index=self.index, 
                axis=self.axis,
                method=self.method, 
                as_series=self.as_series,
                verbose=self.verbose, 
                squeeze_arr=self.squeeze_arr
            )(*args, **kwargs)
        return wrapped
    
    def _wrapper(self, arr, *args, **kwargs):
        arr = self._convert_input(arr)
        
        if isinstance (arr, pd.Series): 
            result= arr.copy() 
        elif isinstance(arr, pd.DataFrame):
            result = self._extract_from_dataframe(arr)
        elif isinstance(arr, np.ndarray):
            result = self._extract_from_ndarray(arr)
        else:
            raise TypeError("The input must be either a Pandas DataFrame or a NumPy ndarray.")
        
        if self.as_series and isinstance(result, np.ndarray):
            result = pd.Series(result)
        elif not self.as_series and isinstance(result, pd.Series):
            result = result.to_numpy()
        
        return self.func(result, *args, **kwargs)
    
    def __get__(self, instance, owner):
        return self.__class__(
            self.func.__get__(instance, owner), column=self.column, 
            index=self.index, axis=self.axis,
            method=self.method, as_series=self.as_series, verbose=self.verbose, 
            squeeze_arr=self.squeeze_arr)

    def _convert_input(self, arr):
        """Convert input data to numpy array or pandas DataFrame if
        it's a list or dictionary, respectively."""
        if isinstance(arr, list):
            arr = np.array(arr)
        elif isinstance(arr, dict):
            arr = pd.DataFrame(arr)
        return arr
        
    def _extract_from_dataframe(self, arr):
        """Extract data from a pandas DataFrame based on the 
        specified column or index."""
        if self.column is not None:
            if isinstance(self.column, int):  # Column by integer index
                self._validate_column_index(arr, self.column)
                result = arr.iloc[:, self.column]
            elif isinstance(self.column, str):  # Column by name
                self._validate_column_name(arr, self.column)
                result = arr[self.column]
        elif self.index is not None:
            result = arr.iloc[self.index]
        else:
            result = self._default_dataframe_extraction(arr)
        return result
    
    def _extract_from_ndarray(self, arr):
        """Extract a specific slice from a numpy ndarray based on 
        the provided parameters."""
        if arr.ndim > 1:
            result = self._handle_multidimensional_array(arr)
        else:
            result = np.squeeze(arr) if self.squeeze_arr else arr
        return result
    
    def _validate_column_index(self, arr, index):
        """Validate if the provided column index is within the valid range."""
        if index < 0 or index >= arr.shape[1]:
            raise ValueError("The specified column index is out of range."
                             " Please provide a valid index.")
            
    def _validate_column_name(self, arr, name):
        """Validate if the provided column name exists in the DataFrame."""
        if name not in arr.columns:
            raise ValueError(f"The specified column name '{name}' does"
                             " not exist in the DataFrame.")
            
    def _default_dataframe_extraction(self, arr):
        """Extract the first column from a DataFrame by default when no 
        specific column or index is provided."""
        if self.method == 'soft':
            if self.verbose:
                print("No specific column or index provided; extracting"
                      " the first column by default.")
            return arr.iloc[:, 0]
        else:
            raise ValueError("No specific column or index was provided while "
                             "a DataFrame was passed, and 'soft' method is not"
                             " enabled.")
    
    def _handle_multidimensional_array(self, arr):
        """
        Handle the extraction from a multidimensional numpy array
        based on axis and other parameters.
        
        This method takes into account the specified axis, column, index, and
        the extraction method to retrieve a one-dimensional array from a 
        multidimensional numpy array.
        """
        if self.axis is not None:
            if self.axis == 0:
                if self.index is not None:
                    # Extract specific row
                    result = arr[self.index, :]
                else:
                    if self.method == "soft":
                        if self.column is not None and isinstance(self.column, int):
                            self._validate_column_index(arr, self.column)
                            result = arr[:, self.column]
                            if self.verbose:
                                print("Column specified, extracting based on "
                                      "column index in 'soft' mode.")
                        else:
                            raise ValueError("Column must be an integer for "
                                             "ndarray when 'soft' method is "
                                             "used and axis=0.")
                    else:
                        raise ValueError("Column cannot be used for axis=0 "
                                         "unless method is set to 'soft'.")
            elif self.axis == 1:
                if self.column is not None and isinstance(self.column, int):
                    # Extract specific column
                    self._validate_column_index(arr, self.column)
                    result = arr[:, self.column]
                elif self.index is not None:
                    # Index behaves dually; here it acts 
                    # as column extraction for axis=1
                    if self.verbose:
                        print("Index is treated as column for numpy array when axis=1.")
                    self._validate_column_index(arr, self.index)
                    result = arr[:, self.index]
                else:
                    raise ValueError("Either column or index needs to be "
                                     "specified when axis is 1.")
            else:
                raise ValueError("Axis must be 0 or 1.")
        else:
            if self.squeeze_arr:
                result = np.squeeze(arr)
                if result.ndim >= 2:
                    # Squeeze failed to convert to 1D; likely 
                    # a matrix without a specified axis
                    raise ValueError("Unable to automatically convert 2D array"
                                     " to 1D without axis specification.")
            else:
                # No squeezing; ensure arr is already 1D or has a 
                # single dimension to be treated as 1D
                if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
                    # Treat as 1D if one of the dimensions is 1,
                    # regardless of orientation
                    result = arr.flatten()
                else:
                    raise ValueError("Array is multidimensional and requires"
                                     " axis specification for extraction.")
        
        return result

class DynamicMethod:
    """
    A class-based decorator designed to preprocess data before it's passed to 
    a function or method. This preprocessing includes filtering data by type, 
    selecting specific columns, handling missing values, applying transformations,
    and executing based on custom conditions. It offers advanced options like 
    treating integer columns as categorical and encoding categorical columns.

    Parameters
    ----------
    expected_type : str, optional
        Specifies the expected data type for processing. The options are:
        - 'numeric': Only numeric columns are considered.
        - 'categorical': Only categorical columns are considered.
        - 'both': Both numeric and categorical columns are considered.
        Defaults to ``'numeric'``.

    capture_columns : bool, optional
        If set to True, the decorator filters the DataFrame columns to those 
        specified in the 'columns' keyword argument passed to the decorated 
        function. 
        Defaults to ``False``.

    treat_int_as_categorical : bool, optional
        When True, integer columns in the DataFrame are treated as categorical 
        data, which can be particularly useful for statistical operations that
        distinguish between numeric and categorical data types, such as ANOVA 
        tests. 
        Defaults to ``False``.

    encode_categories : bool, optional
        If True, categorical columns are encoded into integer values. This is
        especially useful for models that require numerical input for 
        categorical data.
        Defaults to ``False``.

    drop_na : bool, optional
        Determines whether rows or columns with missing values should be dropped.
        The specific rows or columns to drop are dictated by `na_axis` and
        `na_thresh`.
        Defaults to ``False``.

    na_axis : Union[int, str], optional
        Specifies the axis along which to drop missing values. Acceptable 
        values are:
        - 0 or 'row': Drop rows with missing values.
        - 1 or 'col': Drop columns with missing values.
        Defaults to ``0``.

    na_thresh : Optional[float], optional
        Sets a threshold for dropping rows or columns with missing values. 
        This can be specified as an absolute number of non-NA values or a
        proportion (0 < value <= 1) of the total number of values in a row
        or column.
        Defaults to ``None``.

    transform_func : Optional[Callable], optional
        A custom function to apply to the DataFrame before passing it to the
        decorated function. This allows for flexible data transformations 
        as needed.
        Defaults to ``None``.

    condition : Optional[Callable[[pd.DataFrame], bool]], optional
        A condition function that takes the DataFrame as an argument and 
        returns ``True`` if the decorated function should be executed. This
        enables conditional processing based on the data.
        Defaults to ``None``.

    reset_index : bool, optional
        If True, the DataFrame index is reset before processing. This is useful
        after filtering rows to ensure the index is continuous.
        Defaults to ``False``.
        
    prefixer : str or None, optional
        A string to prefix the function name with when adding it as a method 
        to DataFrame and Series objects. If set to "exclude" or 'false' 
        (case-insensitive), the prefix is omitted, and the original function 
        name is used. If None or not provided, a default prefix of 'go' is 
        used to denote the method's origin from the gofast package.
        
    verbose : bool, optional
        Controls the verbosity of the decoration process. If True, detailed 
        information about the preprocessing steps is printed.
        Defaults to ``False``.

    Raises
    ------
    ValueError
        If the first argument to the decorated function is not a pandas DataFrame,
        dictionary, or NumPy ndarray.

    Examples
    --------
    >>> from gofast.decorators import DynamicMethod
    >>> @DynamicMethod(expected_type='numeric', capture_columns=True, 
    ... verbose=True, drop_na=True, na_axis='row', na_thresh=0.5, reset_index=True)
    ... def calculate_variance(data):
    ...     return data.var(ddof=0).mean()
    
    >>> data = pd.DataFrame({"A": [1, 2, np.nan], "B": [4, np.nan, 6]})
    >>> print(calculate_variance(data))
    # The above example demonstrates preprocessing a DataFrame by dropping rows with
    # more than 50% missing values and calculating the variance of the remaining data.

    Notes
    -----
    - The `treat_int_as_categorical` and `encode_categories` parameters offer flexibility
      in handling integer and categorical data, which can be critical for certain types
      of analysis or modeling.
    - The `transform_func` and `condition` parameters allow for custom data transformations
      and conditional execution, adding a layer of customization to the preprocessing steps.
    """
    def __init__(
        self, 
        expected_type: str = 'numeric', 
        capture_columns: bool = False,
        treat_int_as_categorical: bool = False, 
        encode_categories: bool = False, 
        verbose: bool = False, 
        drop_na: bool = False, 
        na_axis: Union[int, str] = 0, 
        na_thresh: Optional[float] = None, 
        transform_func: Optional[Callable] = None, 
        condition: Optional[Callable[[pd.DataFrame], bool]] = None, 
        reset_index: bool = False,
        prefixer:Optional[str]=None, 
        ):
        self.expected_type = expected_type
        self.capture_columns = capture_columns
        self.treat_int_as_categorical = treat_int_as_categorical
        self.encode_categories = encode_categories
        self.drop_na = drop_na
        self.na_axis = na_axis
        self.na_thresh = na_thresh
        self.transform_func = transform_func
        self.condition = condition
        self.reset_index = reset_index
        self.prefixer = prefixer
        self.verbose = verbose

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.verbose:
                print(f"Preprocessing data for {func.__name__}...")

            data = self._validate_and_prepare_data(args[0], **kwargs)
            if data is None:
                return func(*args, **kwargs)  # Early exit if data validation fails

            data = self._process_data(data, **kwargs)
            return func(data, *args[1:], **kwargs)
        
        self._add_method_to_pandas (
            wrapper, prefixer= self.prefixer)
        
        return wrapper

    def _validate_and_prepare_data(self, data, **kwargs):
        """
        Validates the input data and converts it to a pandas DataFrame if
        necessary.
    
        This method checks if the first argument is one of the supported types 
        (pd.DataFrame, dict, np.ndarray, or iterable object) and converts it to 
        a pandas DataFrame if it's not already one. If `columns` are specified in 
        kwargs and the data is an np.ndarray or a converted iterable, it attempts 
        to use these columns when creating the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame, dict, or np.ndarray
            The input data to be validated and possibly converted.
        **kwargs : dict
            Additional keyword arguments, including 'columns' which may be used 
            if the data is an np.ndarray to specify DataFrame column names.
    
        Returns
        -------
        pd.DataFrame
            The validated and prepared pandas DataFrame.
    
        Raises
        ------
        ValueError
            If the input data is not one of the supported types.
        """
        if isinstance(data, dict):
            data= pd.DataFrame(data)
        # Convert iterable (not DataFrame or np.ndarray) to DataFrame
        elif  hasattr(data, '__iter__') and not isinstance(
                data, ( pd.DataFrame, np.ndarray, pd.Series)):  
            try:
                data =  np.array(data)
            except Exception:
                raise ValueError(
                    "Expect the first argument to be an iterable object"
                    " with minimum samples equal 2.")
        if isinstance(data, np.ndarray):
            columns = kwargs.get('columns')
            if isinstance (columns, str): 
                columns =[columns]
                
            data = pd.DataFrame(data, columns=(
                columns if columns and len(columns) == data.shape[1] else None))
            
        elif isinstance ( data, pd.Series): 
            data = pd.DataFrame ( data)
            
        # Finally validate  whether dataFrame if constructed.
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Input data must be a pd.DataFrame, dict, np.ndarray,"
                " or iterable.")
        return data
    
    def _process_data(self, data: pd.DataFrame, **kwargs):
        """
        Applies various preprocessing steps to the data based on the decorator's parameters.
    
        This method sequentially processes the data through specified steps: capturing 
        specified columns, filtering data by type, dropping missing values, applying 
        a transformation function, checking a condition for execution, and resetting 
        the index if required.
    
        Parameters
        ----------
        data : pd.DataFrame
            The data to be processed.
        **kwargs : dict
            Additional keyword arguments passed through from the decorator.
    
        Returns
        -------
        None
            The function modifies the data in place and does not return any value.
        """
        if self.capture_columns:
            data = self._capture_columns(data, **kwargs)
        if self.expected_type in ['numeric', 'categorical']:
            data = self._filter_data_type(data)
        if self.drop_na:
            data = self._drop_na(data)
        if self.transform_func:
            data = self.transform_func(data)
        if self.condition and not self.condition(data):
            if self.verbose:
                print("Condition for execution not met, skipping function call.")
            return
        if self.reset_index:
            data = data.reset_index(drop=True)

        return data 
        
    def _capture_columns(self, data: pd.DataFrame, **kwargs):
        """
        Filters the columns of the DataFrame based on the specified 'columns' in kwargs.
    
        If the 'columns' keyword argument is provided, this method attempts to filter 
        the DataFrame to include only those columns. If any specified columns do not 
        exist in the DataFrame, a warning is printed if verbose output is enabled.

        """
        columns = kwargs.pop('columns', None)
        if columns is not None:
            try:
                data = data[columns]
            except KeyError:
                if self.verbose:
                    print("Specified columns do not match, ignoring columns.")
        return data 
    
    def _filter_data_type(self, data: pd.DataFrame):
        """
        Filters the data based on the expected type ('numeric' or 'categorical').
    
        This method filters the DataFrame to include only numeric or categorical 
        columns based on the `expected_type` parameter. If 'categorical' is specified, 
        it further processes categorical data as per the class parameters.
        """
        if self.expected_type == 'numeric':
            data = data.select_dtypes(include=[np.number])
        elif self.expected_type == 'categorical':
            data =self._handle_categorical_data(data)
        
        return data 
    
    def _handle_categorical_data(self, data: pd.DataFrame):
        """
        Handles categorical data by treating integer columns as categorical
        if specified, and encoding categorical columns if required.
    
        This method processes integer columns as categorical if
        `treat_int_as_categorical` 
        is True, and encodes categorical columns into integers if 
        `encode_categories` is True.
    
        """
        if self.treat_int_as_categorical:
            int_columns = data.select_dtypes(include=[int]).columns.tolist()
            data[int_columns] = data[int_columns].astype('category')
        if self.encode_categories:
            data = self._encode_categorical_columns(data)
        
        return data 
    
    def _encode_categorical_columns(self, data: pd.DataFrame):
        """
        Encodes categorical columns in the DataFrame into integer values.
    
        This method applies Label Encoding to columns in the DataFrame that are 
        identified as categorical (either 'category' or 'object' dtype).

        """
        from sklearn.preprocessing import LabelEncoder
        cat_columns = data.select_dtypes(include=['category', 'object']).columns
        for col in cat_columns:
            data[col] = LabelEncoder().fit_transform(data[col])
            
        return data 

    def _drop_na(self, data: pd.DataFrame):
        """
        Drops rows or columns from the DataFrame based on missing values criteria.
    
        This method drops rows or columns with missing values based on the specified
        `na_axis` and `na_thresh` parameters. `na_axis` determines the axis along which
        to drop (rows or columns), and `na_thresh` specifies the threshold for dropping
        either as an absolute number of non-NA values required to keep a row/column or
        as a proportion of the total number of values.
    
        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame from which rows or columns will be dropped based 
            on missing values.
    
        Modifies
        --------
        data : pd.DataFrame
            The input DataFrame is modified in place by dropping specified 
            rows or columns.
    
        """
        # Convert na_axis from string to integer if necessary
        na_axis = 0 if self.na_axis in [0, 'row'] else 1 if self.na_axis in [
            1, 'col'] else self.na_axis
        
        # Calculate the threshold for dropping based on the proportion 
        # if specified as a float less than 1
        if self.na_thresh is None: 
            thresh= self.na_thresh
        elif 0 < self.na_thresh <= 1:
            total_elements = len(data.columns) if na_axis == 0 else len(data)
            thresh = int(total_elements * self.na_thresh)
        else:
            thresh = self.na_thresh
        # Drop missing values based on the specified axis and threshold
        return data.dropna(axis=na_axis, thresh=thresh)
                                  
    def _add_method_to_pandas(self, func, prefixer=None):
        """
        Dynamically adds a custom method to pandas DataFrame and Series classes. 
        This enhancement allows for extending pandas objects with additional 
        functionality in a flexible manner. The method can be optionally prefixed 
        to denote its origin or purpose, enhancing readability and avoiding 
        namespace collisions.
    
        The method checks if a function, optionally prefixed, already exists 
        as a method within the pandas DataFrame and Series classes. If the 
        method does not exist, it is added, making it accessible directly 
        from DataFrame and Series instances.
    
        Parameters
        ----------
        func : function
            The function to be added as a method. This function should accept a 
            DataFrame or Series as its first argument, followed by any additional 
            arguments or keyword arguments the function requires.
        prefixer : str or None, optional
            A string to prefix the function name with when adding it as a method 
            to DataFrame and Series objects. If set to "exclude" or 'false' 
            (case-insensitive), the prefix is omitted, and the original function 
            name is used. If None or not provided, a default prefix of 'go' is 
            used to denote the method's origin from the gofast package.
    
        Examples
        --------
        Suppose `custom_func` is a function intended to be added to DataFrame and 
        Series objects, and we want to prefix it with 'go_':
    
            >>> def custom_func(df, *args, **kwargs):
            ...     # Implementation here
            ...     pass
            >>> gofast_instance._add_method_to_pandas(custom_func)
    
        Now, `custom_func` can be called on DataFrame and Series objects like so:
    
            >>> df.go_custom_func(*args, **kwargs)
    
        If `dynamic_prefixer` is set to "exclude", the 'go_' prefix is omitted:
    
            >>> gofast_instance._add_method_to_pandas(custom_func, dynamic_prefixer="exclude")
            >>> df.custom_func(*args, **kwargs)
    
        Raises
        ------
        Exception
            If an error occurs while adding the method, it is caught and a message 
            is printed. This behavior can be modified to log the error or handle 
            it as needed.
        """
        # Determine whether to use a prefix based on `prefixer`
        method_name = func.__name__ if prefixer in (
            "exclude", 'false') else "go_" + func.__name__
    
        # Attempt to add the method to both DataFrame and Series classes
        for cls in [pd.DataFrame, pd.Series]:
            if not hasattr(cls, method_name):
                try:
                    setattr(cls, method_name, func)
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to add method {method_name}: {e}")
                    # Optionally log or handle the error as needed
                    
class ExportData:
    """
    A decorator for exporting data into various formats post-function execution. 
    It supports exporting pandas DataFrames or other data types to specified 
    file formats with additional customization through keyword arguments.

    Parameters
    ----------
    export_type : str, optional
        The type of data to export, which can be 'frame' for pandas DataFrames or 'text' 
        for text files. Defaults to 'frame'.
    encoding : str, optional
        The encoding to use for text files. Defaults to 'utf-8'. This parameter is not 
        applicable when exporting DataFrames.
    **kwargs : dict
        Additional keyword arguments to be passed to the pandas export function or 
        the file writing process.

    Examples
    --------
    >>> from gofast.decorators import ExportData
    >>> @ExportData(export_type='frame', file_format='csv')
    ... def data_processing_function():
    ...     df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    ...     return df, 'output_filename', 'csv', './savepath', 'Data', {}
    ...
    >>> data_processing_function()
    # This will save the DataFrame returned by data_processing_function to a CSV file
    # named 'output_filename.csv' in the './savepath' directory.
    """
    
    def __init__(self, export_type='frame', encoding='utf8', **kwargs):
        self.export_type = export_type
        self.encoding = encoding
        self.kwargs = kwargs

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_func(*args, **func_kwargs):
            # Extracting data and parameters from the decorated function
            dfs, fname, file_format, savepath, nameof, extra_kwargs = func(
                *args, **func_kwargs)
            
            if self.kwargs.get("file_format") and file_format: 
                # The priority is given to user format. 
                _= self.kwargs.pop("file_format", None)
            
            # Merge decorator's kwargs with function's extra_kwargs,
            # giving precedence to extra_kwargs
            export_kwargs = {**self.kwargs, **extra_kwargs}

            # Ensure the savepath exists
            os.makedirs(savepath, exist_ok=True)

            # Setting file format based on extension or provided format
            _, ext = os.path.splitext(fname)
            file_format = ext.lower() if ext else f".{file_format}"

            # Validate file format
            if self.export_type.lower() == 'frame' and file_format not in ['.csv', '.xlsx']:
                raise ValueError(f"Unsupported file format for DataFrame export: {file_format}")

            # Choose the writer function based on the export type
            if self.export_type.lower() == 'frame':
                fnames = self._export_frame(dfs, fname, file_format, savepath, nameof, **export_kwargs)
            else:
                fnames = self._export_others(dfs, fname, file_format, savepath, nameof, **export_kwargs)

            # Optionally move files to a designated output directory
            # Assuming move_cfile function exists and is imported correctly
            for fname in fnames:
                from .tools.coreutils import move_cfile 
                move_cfile(fname, savepath, dpath='_out')
                
            # Optionally return the filenames of the exported files
            return fnames
        return wrapper_func
        
    def _export_frame(self, dfs, fname, file_format, savepath, nameof=None, **kwargs):
        """
        Handles exporting pandas DataFrame(s) to specified file formats.
        """
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else dfs
        fnames = []
        for i, df in enumerate(dfs):
            if not isinstance(df, pd.DataFrame):
                continue
            name_suffix = f"_{nameof[i]}" if nameof and i < len(nameof) else ""
            output_fname = f"{fname}{name_suffix}{file_format}"
            full_path = os.path.join(savepath, output_fname)
            
            if file_format == '.xlsx':
                with pd.ExcelWriter(full_path) as writer:
                    df.to_excel(writer, sheet_name=nameof[i] if nameof and i < len(nameof) else 'Sheet1', **kwargs)
            else:
                df.to_csv(full_path, **kwargs)
            fnames.append(full_path)
        
        return fnames

    def _export_others(self, data, fname, file_format, savepath, nameof=None,
                       **kwargs):
        """
        Handles exporting non-DataFrame data to files, primarily text files.
        """
        output_fname = f"{fname}{file_format}"
        full_path = os.path.join(savepath, output_fname)
        
        with open(full_path, mode='w', encoding=self.encoding) as f:
            for item in data:
                f.write(f"{item}\n")
                
        return [full_path]
  
class Temp2D:
    """
    A decorator for creating two-dimensional plots from the outputs of 
    decorated functions. It integrates seamlessly with matplotlib for 
    plotting and supports customization through various parameters.

    Parameters
    ----------
    reason : str, optional
        The purpose of the plot. This parameter is for documentation 
        purposes and does not affect the plot's appearance or behavior.
    **kwargs : dict
        Additional keyword arguments for plot customization. These 
        arguments are expected to align with the parameters used by 
        matplotlib and related plotting utilities.

    Notes
    -----
    The decorator uses the last return value of the decorated function as a 
    dictionary of plotting arguments, which should include keys and values 
    compatible with `matplotlib.pyplot` functions. If these plotting 
    arguments are not provided, an AttributeError will be raised.

    Examples
    --------
    >>> from gofast.decorators import Temp2D
    >>> @Temp2D(reason="Show an example")
    ... def generate_data():
    ...     x = np.linspace(0, 10, 100)
    ...     y = np.sin(x)
    ...     return x, y, {'xlabel': 'X Axis', 'ylabel': 'Y Axis'}
    ...
    >>> generate_data()
    # This will plot a sine wave with the specified x and y labels.
    """

    def __init__(self, reason=None, **kwargs):
        self.reason = reason
        self.plot_kwargs = kwargs

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the decorated function and expect a tuple where
            # the last item is a dictionary for plot customization
            *plot_data, plot_customization = func(*args, **kwargs)

            # Update the plot customization with any additional kwargs
            # provided during the decorator initialization
            plot_customization.update(self.plot_kwargs)

            # Call the plot creation method
            return self.plot2d(*plot_data, **plot_customization)

        return wrapper

    def plot2d(self, x, y, **kwargs):
        """
        Generates a 2D plot based on the provided x and y data along with 
        customizable plotting arguments.

        Parameters
        ----------
        x : array-like
            X-coordinates for the plot.
        y : array-like
            Y-coordinates for the plot.
        **kwargs : dict
            Additional keyword arguments for customizing the plot, such as 
            'xlabel', 'ylabel', and any other matplotlib.axes.Axes method 
            arguments.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib Axes object with the plot.

        Example
        -------
        >>> Temp2D().plot2d(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)),
        ...                set_xlabel="X Axis", set_ylabel="Y Axis", set_title="Sine Wave")
        # This will create and display a 2D plot of a sine wave.
        """
        fig, ax = plt.subplots()
        ax.plot(x, y)

        # Apply customizations from kwargs
        for key, value in kwargs.items():
            if key in ["ylabel", 'xlabel', 'title']: 
                key = f"set_{key}" # for Axes 
            if hasattr(ax, key) and callable(getattr(ax, key)):
                getattr(ax, key)(value)
            else:
                print(f"Warning: {key} is not a valid Axes method")

        plt.show()
        return ax

    def __getattr__(self, name):
        # Custom error message for missing attributes
        msg = (f"{self.__class__.__name__!r} has no attribute {name!r}. "
               "Ensure plot arguments are supplied as the last return value "
               "of the decorated function.")
        raise AttributeError(msg)

class SignalFutureChange:
    """
    A decorator that signals an upcoming change to a function or class, such 
    as deprecation or a recommendation to use a more robust alternative. It 
    allows the function or class to execute normally while optionally logging 
    a warning message.

    Parameters
    ----------
    message : str, optional
        A message to be displayed to indicate the reason for the future change. 
        This could inform about deprecation or suggest using an alternative.

    Examples
    --------
    >>> from gofast.decorators import SignalFutureChange
    >>> @SignalFutureChange(message="This function will be deprecated in future "
    ...                        "releases. Consider using `new_function` instead.")
    ... def old_function():
    ...     print("This is an old function.")
    ...
    >>> old_function()
    # Executes old_function, logging a message about future deprecation or 
    # recommending an alternative, based on the provided message.
    """
    
    def __init__(self, message=None):
        self.message = message

    def __call__(self, cls_or_func):
        if self.message:
            # Log the warning message at the time of decoration, not at call time
            warnings.warn(self.message, FutureWarning, stacklevel=2)
        
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            # Directly return the result of the original function or class call
            return cls_or_func(*args, **kwargs)
        return wrapper

class AppendDocReferences:
    """
    A decorator for appending reStructuredText references to the docstring 
    of the decorated function or class, enhancing Sphinx documentation by 
    auto-retrieving and replacing values from specified references.

    This allows for dynamic insertion of common documentation elements, 
    such as glossary terms or external documentation links, into the 
    docstrings of multiple functions or classes.

    Parameters
    ----------
    docref : str, optional
        The documentation reference string to be appended to the function's 
        or class's docstring. This should be in reStructuredText format.

    Examples
    --------
    >>> from gofast.decorators import AppendDocReferences
    >>> @AppendDocReferences(docref=".. |VES| replace:: Vertical Electrical"
    ...                         " Sounding\\n.. |ERP| replace:: Electrical Resistivity Profiling")
    ... def example_function():
    ...     '''This function demonstrates appending doc references.
    ...
    ...     See more details about |VES| and |ERP|.
    ...     '''
    ...     pass
    ...
    >>> print(example_function.__doc__)
    # The docstring of example_function will now include the replaced 
    # references to VES and ERP along with their definitions.
    """

    def __init__(self, docref=None):
        self.docref = "\n" + docref if docref else ""

    def __call__(self, cls_or_func):
        
        original_doc = cls_or_func.__doc__ if cls_or_func.__doc__ else ''
        # Append the doc reference to the original docstring
        cls_or_func.__doc__ = original_doc + self.docref
        
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            # Directly return the result of the original function or class call
            return cls_or_func(*args, **kwargs)
        
        return wrapper
    
class Deprecated:
    """
    A decorator for marking functions, methods, and classes as deprecated. 
    It emits a deprecation warning when the decorated item is called or 
    instantiated.

    Parameters
    ----------
    reason : str
        The reason why the function, method, or class is deprecated.

    Examples
    --------
    >>> from gofast.decorators import Deprecated
    >>> @Deprecated(reason="Use `new_function` instead.")
    ... def old_function():
    ...     print("This function is deprecated.")
    ...
    >>> old_function()
    # Outputs a deprecation warning and prints: "This function is deprecated."

    Note
    ----
    The warning will point to the location where the deprecated item is 
    used, making it easier to identify and replace deprecated usage in 
    codebases.
    """
    
    def __init__(self, reason):
        if not reason:
            raise ValueError("A reason for deprecation must be supplied.")
        self.reason = reason

    def __call__(self, cls_or_func):
        if not inspect.isfunction(cls_or_func) and not inspect.isclass(cls_or_func):
            raise TypeError("Deprecated decorator can only be applied to functions or classes.")

        fmt = "Call to deprecated {item} {name} ({reason})."
        item_type = "class" if inspect.isclass(cls_or_func) else "function or method"
        msg = fmt.format(item=item_type, name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return cls_or_func(*args, **kwargs)

        return new_func

class CheckGDALData:
    """
    A decorator to ensure the availability of GDAL data for functions requiring GDAL. 
    It checks if the GDAL_DATA environment variable is correctly set and points to an 
    existing path. Optionally, it can raise an ImportError if the GDAL data is not 
    configured correctly.

    Parameters
    ----------
    raise_error : bool, optional
        If True, raises an ImportError when GDAL data is not found. Defaults to False.
    verbose : int, optional
        Verbosity level. A higher number indicates more verbose output. Defaults to 0.

    Examples
    --------
    >>> from gofast.decorators import CheckGDALData
    >>> @CheckGDALData(raise_error=True, verbose=1)
    ... def my_gdal_function():
    ...     print("This function uses GDAL.")
    ...
    >>> my_gdal_function()
    # This will either print "This function uses GDAL." if GDAL data is correctly set,
    # or raise an ImportError with instructions on how to configure GDAL data.

    Notes
    -----
    This decorator is particularly useful in environments where GDAL is required but 
    might not be fully configured, such as in some virtual environments or custom 
    installations.
    """

    _has_checked = False
    _gdal_data_found = False

    def __init__(self, raise_error=False, verbose=0):
        self.raise_error = raise_error
        self.verbose = verbose

    def __call__(self, func):
        if not self._has_checked:
            self._check_gdal_data()
            self._has_checked = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._gdal_data_found and self.raise_error:
                raise ImportError(
                    "GDAL is NOT installed correctly. "
                    f"GDAL wheel can be downloaded from {self._gdal_wheel_resources}. "
                    f"See the installation guide: {self._gdal_installation_guide}."
                )
            return func(*args, **kwargs)

        return wrapper

    def _check_gdal_data(self):
        from subprocess import Popen, PIPE 
        if 'GDAL_DATA' in os.environ and os.path.exists(os.environ['GDAL_DATA']):
            if self.verbose:
                _logger.info(f"GDAL_DATA is set to: {os.environ['GDAL_DATA']}")
            self._gdal_data_found = True
        else:
            if self.verbose:
                _logger.warning(
                    "GDAL_DATA environment variable is not set. "
                    f"Please see {self._gdal_data_variable_resources}"
                )
            # Attempt to locate GDAL data using gdal-config
            try:
                if self.verbose:
                    _logger.info("Trying to find gdal-data path ...")
                process = Popen(['gdal-config', '--datadir'], stdout=PIPE, stderr=PIPE)
                output, err = process.communicate()
                if process.returncode == 0 and os.path.exists(output.strip()):
                    os.environ['GDAL_DATA'] = output.strip().decode()
                    if self.verbose:
                        _logger.info(f"Found gdal-data path: {os.environ['GDAL_DATA']}")
                    self._gdal_data_found = True
            except Exception as e:
                if self.verbose:
                    _logger.error(f"Failed to find gdal-data path. Error: {e}")
                self._gdal_data_found = False

    # Class variable declarations for resources and installation guide
    _gdal_data_variable_resources = 'https://trac.osgeo.org/gdal/wiki/FAQInstallationAndBuilding#HowtosetGDAL_DATAvariable'
    _gdal_wheel_resources = 'https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal'
    _gdal_installation_guide = 'https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/'

class RedirectToNew:
    """
    A decorator to redirect calls from deprecated functions or classes to new ones,
    issuing a deprecation warning and guiding users towards the updated implementation.

    This decorator simplifies the process of transitioning codebases to use new
    functions or classes without breaking existing implementations that rely on
    deprecated ones.

    Parameters
    ----------
    new_target : callable or class
        The new function or class to which calls should be redirected.
    reason : str
        Explanation why the redirection is occurring, typically including deprecation
        information and guidance on using the new target.

    Examples
    --------
    >>> from gofast.decorators import RedirectToNew
    >>> @RedirectToNew(new_function, "Use `new_function` instead of `old_function`.")
    ... def old_function():
    ...     pass
    ...
    >>> old_function()
    # This call will be redirected to `new_function`, with a warning issued about the deprecation.

    """

    def __init__(self, new_target, reason):
        if not callable(new_target):
            raise TypeError("The new target must be a callable or a class.")
        if not isinstance(reason, str):
            raise TypeError("Redirect reason must be supplied as a string.")

        self.new_target = new_target
        self.reason = reason

    def __call__(self, cls_or_func):
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            _logger.warning(f"DEPRECATION WARNING: {self.reason}")
            return self.new_target(*args, **kwargs)

        return wrapper
  
class PlotPrediction:
    """
    A decorator for plotting predictions and observations using matplotlib. 
    This decorator enhances functions that return prediction and observation 
    data by optionally generating a scatter plot for visual comparison.

    Parameters
    ----------
    turn : str, optional
        Controls whether plotting is enabled ('on') or disabled ('off'). Defaults to 'off'.
    **kwargs : dict
        Customization options for the plot, supporting matplotlib.pyplot keywords.

    Attributes
    ----------
    fig_size : tuple
        Figure size for the plot.
    y_pred_kws : dict
        Styling options for the predicted values scatter plot.
    y_obs_kws : dict
        Styling options for the observed values scatter plot.
    tick_params : dict
        Parameters for configuring axis ticks.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    obs_line : tuple
        Controls the observation line plotting ('on', 'off') and its type ('Obs', 'Pred').
    l_kws : dict
        Line properties for the observation line.
    savefig : str or dict
        Path or options for saving the figure.

    Examples
    --------
    >>> from gofast.decorators import PlotPrediction
    >>> @PlotPrediction(turn='on', fig_size=(10, 6))
    ... def my_prediction_function():
    ...     # prediction function logic here
    ...     return y_true, y_pred, 'on'
    ...
    >>> my_prediction_function()
    # This will generate a scatter plot for the predicted and observed values.

    """

    def __init__(self, turn='off', **kwargs):
        self.turn = turn
        self.fig_size = kwargs.pop('fig_size', (16, 8))
        self.y_pred_kws = kwargs.pop('y_pred_kws', {'c': 'r', 's': 200, 'alpha': 1,
                                                     'label': 'Predicted flow:y_pred'})
        self.y_obs_kws = kwargs.pop('y_obs_kws', {'c': 'blue', 's': 100, 'alpha': 0.8,
                                                   'label': 'Observed flow:y_true'})
        self.tick_params = kwargs.pop('tick_params', {'axis': 'x', 'labelsize': 10,
                                                      'rotation': 90})
        self.xlabel = kwargs.pop('xlabel', 'Boreholes tested')
        self.ylabel = kwargs.pop('ylabel', 'Flow rates(FR) classes')
        self.obs_line = kwargs.pop('obs_line', ('off', 'Obs'))
        self.l_kws = kwargs.pop('l_kws', {'c': 'blue', 'ls': '--', 'lw': 1, 'alpha': 0.5})
        self.savefig = kwargs.pop('savefig', None)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            y_true, y_pred, switch = func(*args, **kwargs)
            turn = switch if switch is not None else self.turn

            if turn == 'on':
                self._plot(y_true, y_pred)

            return y_true, y_pred, switch

        return wrapper

    def _plot(self, y_true, y_pred):
        plt.figure(figsize=self.fig_size)
        plt.scatter(y_true.index, y_true, **self.y_obs_kws)
        plt.scatter(y_pred.index, y_pred, **self.y_pred_kws)

        if self.obs_line[0] == 'on':
            data = y_true if 'true' in self.obs_line[1].lower(
                ) or 'obs' in self.obs_line[1].lower() else y_pred
            plt.plot(data.index, data, **self.l_kws)

        plt.tick_params(**self.tick_params)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()

        if self.savefig:
            if isinstance(self.savefig, str):
                plt.savefig(self.savefig)
            else:
                plt.savefig(**self.savefig)

        plt.show()

class PlotFeatureImportance:
    """
    A decorator to plot permutation feature importance (PFI) diagrams or dendrogram 
    figures for feature correlation analysis. It utilizes matplotlib for plotting 
    and can be customized with various keyword arguments.

    Parameters
    ----------
    kind : str, optional
        Specifies the type of plot to generate. Options are:
        - 'pfi' for permutation feature importance before and after shuffling trees.
        - 'dendro' for a dendrogram plot showing feature correlations.
        Defaults to 'pfi'.
    turn : str, optional
        Controls whether to plot ('on') or not ('off'). Defaults to 'off'.
    **kwargs : dict
        Keyword arguments for matplotlib plotting functions and additional customization.

    Examples
    --------
    >>> from gofast.decorators import PlotFeatureImportance
    >>> @PlotFeatureImportance(kind='pfi', turn='on', fig_size=(10, 6))
    ... def my_model_analysis_function():
    ...     # Function logic here
    ...     return X, y_pred, y_true, model, feature_names, 'on'
    ...
    >>> my_model_analysis_function()
    # This will plot the specified PFI diagram if turn is 'on'.

    Note
    ----
    Ensure matplotlib is installed in your environment to use this decorator.
    """
    
    def __init__(self, kind='pfi', turn='off', **kwargs):
        self.kind = kind
        self.turn = turn
        self.fig_size = kwargs.pop('fig_size', (9, 3))
        self.savefig = kwargs.pop('savefig', None)
        # Default keyword arguments for various plots
        self.barh_kws = kwargs.pop('barh_kws', {'color': 'blue', 'edgecolor': 'k', 'linewidth': 2})
        self.box_kws = kwargs.pop('box_kws', {'vert': False, 'patch_artist': True})
        self.dendro_kws = kwargs.pop('dendro_kws', {'leaf_rotation': 90})
        self.plot_kwargs = kwargs  # Remaining kwargs for further customization

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the decorated function
            results = func(*args, **kwargs)
            # Unpack results based on expected structure
            X, y_pred, y_true, model, feature_names, switch = results
            
            # Update settings based on function return values if provided
            self.turn = switch if switch is not None else self.turn
            
            # Proceed to plot if enabled
            if self.turn.lower() == 'on':
                self._plot_results(X, y_pred, y_true, model, feature_names)
            
            return results

        return wrapper

    def _plot_results(self, X, y_pred, y_true, model, feature_names):
        if self.kind == 'pfi':
            self._plot_pfi(X, model, feature_names)
        elif self.kind == 'dendro':
            self._plot_dendrogram(X, feature_names)
        else:
            warnings.warn(f"Unknown kind '{self.kind}'. No plot will be generated.")

        if self.savefig:
            plt.savefig(self.savefig, **self.plot_kwargs)

    def _plot_pfi(self, X, model, feature_names):
        # Example PFI plotting.
        plt.figure(figsize=self.fig_size)
        plt.barh(range(len(feature_names)), model.feature_importances_, **self.barh_kws)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.show()

    def _plot_dendrogram(self, X, feature_names):
        # Example dendrogram plotting.
        from scipy.cluster.hierarchy import dendrogram, linkage
        Z = linkage(X, 'ward')
        plt.figure(figsize=self.fig_size)
        dendrogram(Z, labels=feature_names, **self.dendro_kws)
        plt.show()

class AppendDocSection:
    """
    A decorator to append a specific section of a function's or class's docstring
    to another. This is particularly useful for avoiding redundancy when documenting
    shared parameters or information across multiple functions or classes.

    Parameters
    ----------
    source_func : callable
        The source function or class whose docstring part will be appended.
    start : str, optional
        The start marker from where to begin appending the docstring.
    end : str, optional
        The end marker where to stop appending the docstring. If not provided, 
        everything from the `start` to the end of the source's docstring is appended.

    Examples
    --------
    >>> from gofast.decorators import AppendDocSection
    >>> @AppendDocSection(source_func=writedf, start='param reason', end='param to_')
    ... def new_function():
    ...     '''Function-specific docstring.'''
    ...     pass
    ...
    >>> print(new_function.__doc__)
    # This will include the section of `writedf`'s docstring from 'param reason' 
    # to 'param to_' appended to 'new_function' docstring.

    """
    
    def __init__(self, source_func, start=None, end=None):
        if not callable(source_func):
            raise TypeError("`source_func` must be a callable.")
        self.source_func = source_func
        self.start = start
        self.end = end

    def __call__(self, target_func):
        source_doc = inspect.getdoc(self.source_func) or ''
        target_doc = inspect.getdoc(target_func) or ''
        
        # Find the start index
        start_ix = source_doc.find(self.start) if self.start else 0
        end_ix = source_doc.find(self.end, start_ix) if self.end else len(source_doc)
        
        # Handle cases where start or end markers are not found
        if self.start and start_ix == -1:
            warnings.warn(f"Start marker '{self.start}' not found in"
                          f" `{self.source_func.__name__}` docstring.")
            start_ix = 0
        if self.end and end_ix == -1:
            warnings.warn(f"End marker '{self.end}' not found in"
                          f" `{self.source_func.__name__}` docstring.")
            end_ix = len(source_doc)

        # Extract the desired docstring section
        doc_section = source_doc[start_ix:end_ix]

        # Append the extracted section to the target's docstring
        target_func.__doc__ = (target_doc + "\n\n" + doc_section).strip()

        return target_func

class AppendDocFrom:
    """
    A decorator for appending a specific section of a function's or class's docstring
    to another. This is useful for avoiding redundancy in documentation, especially
    for shared parameters or descriptions.

    Parameters
    ----------
    source : callable
        The source function or class from which to extract the docstring section.
    from_ : str
        The start marker for the docstring section to be extracted.
    to : str, optional
        The end marker for the docstring section. If not provided, everything
        from `from_` to the end of the source's docstring is used.
    insert_at : str
        The marker in the target's docstring where the extracted section should
        be inserted. If not found, the section is appended at the end.

    Examples
    --------
    >>> from gofast.decorators import AppendDocFrom
    >>> @AppendDocFrom(source=func0, from_='Parameters', to='Returns', insert_at='Parameters')
    ... def new_func():
    ...     '''New function docstring.'''
    ...     pass
    ...
    >>> print(new_func.__doc__)
    # This will print 'new_func' docstring with the 'func0' docstring section 
    # from 'Parameters' to 'Returns' appended at the 'Parameters' marker.

    Note
    ----
    It's recommended to use docstrings with consistent formatting to ensure
    proper insertion and readability.

    """
    def __init__(self, source, from_, to=None, insert_at='Parameters'):
        self.source = source
        self.from_ = from_
        self.to = to
        self.insert_at = insert_at.lower()
        
    def __call__(self, target):
        self._append_doc(target)
        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            return target(*args, **kwargs)

        # self._append_doc(target)
        return wrapper
    
    def _append_doc(self, target):
        source_doc = inspect.getdoc(self.source) or ''
        target_doc = inspect.getdoc(target) or ''
        
        start_index = source_doc.find(self.from_)
        end_index = source_doc.find(self.to, start_index) if self.to else len(source_doc)
        
        if start_index == -1 or (self.to and end_index == -1):
            warnings.warn(f"Cannot find specified docstring section in `{self.source.__name__}`.")
            return
        
        doc_section = source_doc[start_index:end_index]
        
        insert_index = target_doc.lower().find(self.insert_at)
        if insert_index == -1:
            target_doc += "\n\n" + doc_section
        else:
            part1 = target_doc[:insert_index]
            part2 = target_doc[insert_index:]
            target_doc = part1 + doc_section + "\n\n" + part2
        
        target.__doc__ = target_doc

class NumpyDocstring:
    """
    A class decorator designed to automatically parse and reformat the docstring of
    the decorated function into a structured NumPy-style docstring. This decorator
    enhances readability and standardization of documentation, making it more useful
    for developers and users, especially in the context of Sphinx-generated docs.

    Parameters
    ----------
    func : function, optional
        The function to be decorated. If not provided at initialization, it must be
        set later by calling the instance as a decorator.
    enforce_strict : bool, optional
        If True, enforces strict NumPy docstring formatting rules. This may include
        checking for specific sections and their order. Defaults to False, allowing
        for more flexibility in the original docstring format.
    custom_sections : dict, optional
        A dictionary where keys are section titles (e.g., "Custom Section") and values
        are the content for those sections. This allows for the addition of custom
        sections not typically found in NumPy docstrings.

    Examples
    --------
    Using as a decorator directly on a function:

    @NumpyDocstring
    def my_function(x, y):
        \"\"\"Function docstring.\"\"\"
        return x + y

    Adding custom sections and enforcing strict formatting:

    @NumpyDocstring(enforce_strict=True, custom_sections={
        'Custom Section': 'Details here.'})
    def another_function(x):
        \"\"\"Another function docstring.\"\"\"
        return x * 2
    """

    def __init__(self, func=None, *, enforce_strict=False, custom_sections=None):
        self.func = func
        self.enforce_strict = enforce_strict
        self.custom_sections = custom_sections or {}
        if func is not None:
            functools.update_wrapper(self, func)
            self._update_docstring()

    def __call__(self, *args, **kwargs):
        if self.func:
            return self.func(*args, **kwargs)
        else:
            def wrapper(func):
                self.func = func
                functools.update_wrapper(self, func)
                self._update_docstring()
                return self
            return wrapper

    def __get__(self, instance, owner):
        return self if instance is None else functools.partial(self.__call__, instance)

    def _parse_docstring(self, docstring):
        """
        Advanced parsing of the original docstring to identify and reformat sections.
        """
        # Define the sections and their possible headings in docstrings
        section_headings = {
            'Parameters': ['parameters', 'args', 'arguments', ':param'],
            'Returns': ['returns', ':return', ':returns'],
            'Raises': ['raises', ':raise', ':raises'],
            'Examples': ['examples', ':example'],
            'Warnings': ['warnings', ':warning'],
            'See Also': ['see also', 'references'],
            'Notes': ['notes']
        }

        # Initialize sections dictionary
        sections = {key: '' for key in section_headings}

        # Regular expression to detect section headings
        section_regex = re.compile(
            r'^\s*(?P<section>' + '|'.join(
                [f"(?:{'|'.join(headings)})" for headings in section_headings.values()]
                ) + r')\s*$', re.IGNORECASE)

        current_section = None
        for line in docstring.split('\n'):
            match = section_regex.match(line.strip())
            if match:
                # Find which section it belongs to
                for section, headings in section_headings.items():
                    if match.group('section').lower() in headings:
                        current_section = section
                        break
            elif current_section:
                sections[current_section] += line + '\n'

        # Apply custom sections if any
        for section, content in self.custom_sections.items():
            sections[section] = content

        return sections

    def _format_section(self, title, content):
        """
        Format a single section with the given title and content.
        """
        if not content.strip():
            return ''
        return f"{title}\n{'-' * len(title)}\n{content.strip()}\n"

    def _update_docstring(self):
        """
        Update the function's docstring with parsed and formatted content.
        """
        if not self.func.__doc__:
            return

        sections = self._parse_docstring(self.func.__doc__)
        formatted_docstring = "\n".join(self._format_section(
            title, content) for title, content in sections.items() if content)

        self.func.__doc__ = formatted_docstring

    def __set_name__(self, owner, name):
        self._update_docstring()
        setattr(owner, name, self)
        
def sanitize_docstring(enforce_strict=False, custom_sections=None):
    """
    Decorator factory function that returns an instance of NumpyDocstring.
    This function simplifies the application of the decorator with additional parameters
    like enforcing strict formatting and adding custom sections to the docstring.

    Parameters
    ----------
    enforce_strict : bool, optional
        If set to True, the decorator enforces strict adherence to the NumPy docstring
        format, potentially raising errors for non-compliance. Defaults to False.
    custom_sections : dict, optional
        Allows for the specification of custom sections in the decorated function's
        docstring. Keys are the titles of the sections, and values are the content.

    Returns
    -------
    decorator : NumpyDocstring
        An instance of AdvancedNumpyDocDecorator configured with the provided parameters.

    Examples
    --------
    Decorating a function with custom sections and without strict enforcement:

    @sanitize_docstring(custom_sections={'Custom Usage': 'This is how you use this function.'})
    def sample_function(param1, param2):
        \"\"\"This function does something interesting.\"\"\"
        pass
    
    """
    def decorator(func):
        return NumpyDocstring(func, enforce_strict=enforce_strict,
                              custom_sections=custom_sections)
    return decorator

class SuppressOutput:
    """
    A context manager for suppressing stdout and stderr messages. It can be
    useful when interacting with APIs or third-party libraries that output
    messages to the console, and you want to prevent those messages from
    cluttering your output.

    Parameters
    ----------
    suppress_stdout : bool, optional
        Whether to suppress stdout messages. Default is True.
    suppress_stderr : bool, optional
        Whether to suppress stderr messages. Default is True.

    Examples
    --------
    >>> from gofast.decorators import SuppressOutput
    >>> with SuppressOutput():
    ...     print("This will not be printed to stdout.")
    ...     raise ValueError("This error message will not be printed to stderr.")
    
    Note
    ----
    This class is particularly useful in scenarios where controlling external
    library output is necessary to maintain clean and readable application logs.

    See Also
    --------
    contextlib.redirect_stdout, contextlib.redirect_stderr : For more granular control
    over output redirection in specific parts of your code.
    """
    
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None
        self._devnull = None

    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = self._devnull
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout and self._stdout is not None:
            sys.stdout = self._stdout
        if self.suppress_stderr and self._stderr is not None:
            sys.stderr = self._stderr
        if self._devnull is not None:
            self._devnull.close()

class _M:
    def _m(self): pass
MethodType = type(_M()._m)

class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        functools.update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        attr_err = AttributeError(
            f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
        )
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            if not self.check(obj):
                raise attr_err
            out = MethodType(self.fn, obj)

        else:
            # This makes it possible to use the decorated method as an unbound method,
            # for instance when monkeypatching.
            @functools.wraps(self.fn)
            def out(*args, **kwargs):
                if not self.check(args[0]):
                    raise attr_err
                return self.fn(*args, **kwargs)

        return out

def available_if(check):
    """An attribute that is available only if check returns a truthy value

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Examples
    --------
    >>> from sklearn.tools.metaestimators import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)

def isdf(func):
    """
    Advanced decorator that ensures the first positional argument 
    (after `self` for methods) passed to the decorated callable is a pandas 
    DataFrame. If it's not, attempts to convert it to a DataFrame using an 
    optional `columns` keyword argument. 
    
    Function is designed to  be flexible and efficient, suitable for 
    both functions and methods.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if we're decorating a method or a function
        args_list = list(args)
        if args and hasattr(args_list[0], func.__name__):
            # If the first argument has an attribute with the same name as `func`,
            # it's likely an instance method where `self` or `cls` is the first argument.
            self_or_cls, data_arg_index = args_list[0], 1
        else:
            self_or_cls, data_arg_index = None, 0

        # Retrieve the data argument and `columns` keyword argument if provided
        data = args_list[data_arg_index]
        columns = kwargs.get('columns', None)
        if isinstance(columns, str):
            columns = [columns]

        # Proceed with conversion if necessary
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data, columns=columns)
                if columns and len(columns) != data.shape[1]:
                    data = pd.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Unable to convert to DataFrame: {e}")
            # Update the data argument in the arguments list
            args_list[data_arg_index] = data
            
            # Reconstruct args from the potentially modified args_list
            args = tuple(args_list)

        # Call the original function or method, passing `self` or 
        # `cls` explicitly if necessary
        if self_or_cls is not None:
            return func(self_or_cls, *args[1:], **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper

def isdf0(func: Callable) -> Callable:
    """
    A decorator that ensures the first positional argument passed to the 
    decorated function is a pandas DataFrame.
    
    If the argument is not a DataFrame, the decorator attempts to convert it 
    into one using an optional 'columns' keyword argument.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function with data conversion logic.

    Notes
    -----
    The decorated function must accept its first positional argument as data
    and may optionally accept a 'columns' keyword argument to specify column names
    for the DataFrame conversion.

    Examples
    --------
    >>> from gofast.decorators import isdf
    >>> @isdf
    ... def my_function(data, /, columns=None, **kwargs):
    ...     print(data)
    ...     print("Columns:", columns)
    >>> import numpy as np
    >>> my_function(np.array([[1, 2], [3, 4]]), columns=['A', 'B'])
       A  B
    0  1  2
    1  3  4
    Columns: ['A', 'B']
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data = args[0]
        columns = kwargs.get('columns', None)
        if isinstance ( columns, str): 
            columns =[columns]
        # Check if the first positional argument is not a DataFrame
        if not isinstance(data, pd.DataFrame):
            # Attempt to convert it into a DataFrame
            try:
                data = pd.DataFrame(data, columns=columns)
                # If columns are provided but do not match data dimensions,
                # ignore them
                if columns and len(columns) != data.shape[1]:
                    data = pd.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Error converting data to DataFrame: {e}")
            # Call the decorated function with the new DataFrame 
            # as the first argument
            return func(data, *args[1:], **kwargs)
        else:
            # If the first argument is already a DataFrame, 
            # proceed as normal
            return func(*args, **kwargs)
    return wrapper

class NumpyDocstringFormatter:
    """
    A decorator class for reformatting function docstrings to adhere to the
    NumPy documentation standard.

    This class provides a flexible way to ensure that the docstrings of 
    decorated functions follow a consistent format, making them more readable
    and compatible with tools like Sphinx for generating documentation. It can
    automatically extract and reformat specified sections of a docstring, and
    optionally validate the result using Sphinx.

    Parameters
    ----------
    include_sections : list of str, optional
        A list of section names to include in the reformatted docstring. 
        If None (the default), all recognized sections are included. This 
        allows for selective inclusion of sections like "Parameters", "Returns",
        "Examples", etc., based on user preference or requirements.
        
        Example: ['Parameters', 'Returns', 'Examples']

    validate_with_sphinx : bool, default False
        Indicates whether the reformatted docstring should be validated using
        Sphinx. This can be useful for ensuring that the docstring is not only
        correctly formatted but also compatible with Sphinx documentation
        generation. Note that actual implementation of Sphinx validation is 
        not provided in this example and would require integration with Sphinx's
        documentation building process.

    custom_formatting : callable, optional
        A custom function that applies additional formatting to each section
        of the docstring. This function should accept two arguments: 
            `section_name` (a string indicating the name of the section) and 
            `section_content` (the content of the section as a string), and 
            return the formatted content as a string. This allows for
        further customization of the docstring formatting beyond the standard
        reformatting performed by this class::
        
        Example function:
            def custom_formatter(section_name, section_content):
                # Custom formatting logic here
                return formatted_content

    Examples
    --------
    Using the decorator with default settings to reformat all sections:

    >>> from gofast.decorators import NumpyDocstringFormatter
    >>> @NumpyDocstringFormatter()
    ... def example_function(param1, param2=None):
    ...     '''
    ...     This is an example function with parameters.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : int
    ...         The first parameter.
    ...     param2 : int, optional
    ...         The second parameter (default is None).
    ...     '''
    ...     return True

    Specifying sections to include and enabling Sphinx validation:

    >>> @NumpyDocstringFormatter(include_sections=['Parameters', 'Returns'], 
    ...                             validate_with_sphinx=True)
    ... def another_function(param1):
    ...     '''
    ...     Another example function demonstrating selective section inclusion
    ...     and Sphinx validation.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : str
    ...         A string parameter.
    ...
    ...     Returns
    ...     -------
    ...     bool
    ...         Always returns True.
    ...     '''
    ...     return True

    Applying custom formatting to docstring sections:

    >>> def uppercase_formatter(section_name, section_content):
    ...     # Example custom formatting function that uppercases section content
    ...     return section_content.upper()
    ...
    >>> @NumpyDocstringFormatter(custom_formatting=uppercase_formatter)
    ... def custom_formatted_function(param1):
    ...     '''
    ...     Function demonstrating custom formatting of docstring sections.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : str
    ...         A string parameter to be uppercased in the documentation.
    ...     '''
    ...     return param1.upper()
    """

    def __init__(self, include_sections=None, validate_with_sphinx=False, 
                 custom_formatting=None, verbose=0):

        self.include_sections = include_sections
        self.validate_with_sphinx = validate_with_sphinx
        self.custom_formatting = custom_formatting
        self.verbose=verbose 

    def __call__(self, func):
        """
        Decorator method to apply the docstring formatting.

        Parameters
        ----------
        func : function
            The function to decorate.

        Returns
        -------
        function
            The decorated function with a reformatted docstring.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = self.format_docstring(func.__doc__)
        if self.validate_with_sphinx:
            self.sphinx_validation(wrapper.__doc__)
        
        return wrapper

    def format_docstring(self, docstring):
        """
        

        Parameters
        ----------
        docstring : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        """
        Format the original docstring according to the NumPy standard.

        Parameters
        ----------
        docstring : str
            The original docstring to format.

        Returns
        -------
        str
            The formatted docstring.
        """
        if not docstring:
            return ''

        sections_order = ['Parameters', 'Returns', 'Raises', 'Examples', 
                          'Warnings', 'See Also', 'Notes']
        if self.include_sections is not None:
            sections_order = [s for s in sections_order if s in self.include_sections]

        sections = self.extract_sections(docstring, sections_order)
        formatted_docstring = self.reconstruct_docstring(sections, sections_order)

        return formatted_docstring

    def extract_sections(self, docstring, sections_order):
        """
        Extract and return the docstring sections found in the given order.

        Parameters
        ----------
        docstring : str
            The original docstring from which to extract sections.
        sections_order : list of str
            The ordered list of section names to extract.

        Returns
        -------
        dict
            A dictionary with section names as keys and extracted content as values.
        """
        sections = {section: '' for section in sections_order}
        # Simplified patterns
        section_patterns = {
            'Parameters': re.compile(r'parameters\s*[\n\r]+', re.IGNORECASE),
            'Returns': re.compile(r'returns\s*[\n\r]+', re.IGNORECASE),
            'Raises': re.compile(r'raise[s]?\s*[\n\r]+', re.IGNORECASE),
            'Examples': re.compile(r':examples:\s*[\n\r]+', re.IGNORECASE),
            'Warnings': re.compile(r'warnings\s*[\n\r]+', re.IGNORECASE),
            'See Also': re.compile(r'see also\s*[\n\r]+', re.IGNORECASE),
            'Notes': re.compile(r'notes\s*[\n\r]+', re.IGNORECASE),
        }

        for section in sections_order:
            if section in section_patterns:
                match = section_patterns[section].search(docstring)
                if match:
                    content = match.group(0).strip()
                    if self.custom_formatting:
                        content = self.custom_formatting(section, content)
                    sections[section] = content

        return sections

    def reconstruct_docstring(self, sections, sections_order):
        """
        Reconstruct the docstring from the extracted sections in the given order.

        Parameters
        ----------
        sections : dict
            The sections extracted from the original docstring.
        sections_order : list of str
            The ordered list of section names to include in the reconstructed docstring.

        Returns
        -------
        str
            The reconstructed docstring.
        """
        reconstructed_docstring = ""
        for section in sections_order:
            if sections[section]:
                reconstructed_docstring += f"{section}\n{'-' * len(section)}\n{sections[section]}\n\n"
        return reconstructed_docstring
    
    def sphinx_validation(self, docstring):
        """
        Validates the given docstring using Sphinx and docutils to ensure 
        it adheres to standards acceptable by Sphinx for documentation generation.
    
        Parameters
        ----------
        docstring : str
            The docstring to validate.
    
        Note
        ----
        This method provides a conceptual approach and requires a Sphinx 
        environment to be properly implemented.
        """
        from .tools._dependency import import_optional_dependency
        
        try: 
            import_optional_dependency ("docutils")
        except: 
            from .tools.coreutils import is_module_installed 
            from .tools.funcutils import install_package
            if not is_module_installed("docutils"): 
                install_package('docutils', infer_dist_name=True)
            
        from docutils import nodes
        from docutils.core import publish_doctree
        
        try:
            # Create a new document for parsing
            settings_overrides = {'report_level': 2, 'warning_stream': False}
            document = publish_doctree(docstring, settings_overrides=settings_overrides)
            
            # Check for any errors or warnings in the parsed document
            warnings_or_errors = document.traverse(condition=lambda node: isinstance(
                node, (nodes.warning, nodes.error)))
            if next(warnings_or_errors, None):
                if self.verbose:
                    _logger.warning(
                        "Docstring validation failed with warnings or errors.") 
            else:
                if self.verbose:
                    _logger.info("Docstring passed Sphinx validation.")
        except Exception as e:
            if self.verbose:
                _logger.error(
                    f"Docstring validation failed due to an exception: {e}") 

class Dataify:
    """
    A class decorator that ensures the first positional argument passed 
    to the decorated function is a pandas DataFrame, offering flexibility 
    through additional parameters for various data handling scenarios.

    Parameters
    ----------
    enforce_df : bool, optional
        Whether to enforce the conversion of the first positional argument 
        to a pandas DataFrame. Defaults to True.
    auto_columns : bool, optional
        Automatically generates column names if `columns` is not provided.
        Defaults to False.
    prefix : str, optional
        The prefix for auto-generated column names, used only if `auto_columns`
        is True. Defaults to 'col_'.
    columns : list of str, optional
        Specifies the column names for DataFrame conversion. If not provided, 
        and data conversion is necessary, default integer column names are used. 
        This parameter is considered only if `enforce_df` is True.
    ignore_mismatch : bool, optional
        If True, ignores the `columns` parameter if its length does not match 
        the data dimensions, using default integer column names instead. 
        Defaults to False.
    fail_silently : bool, optional
        If True, the decorator will not raise an exception if the conversion 
        fails, and will instead pass the original data to the function. 
        Defaults to False.

    Examples
    --------
    >>> from gofast.decorators import Dataify
    >>> @Dataify(enforce_df=True, columns=['A', 'B'], ignore_mismatch=True)
    ... def process_data(data):
    ...     print(data)

    >>> import numpy as np
    >>> process_data(np.array([[1, 2], [3, 4]]))
       A  B
    0  1  2
    1  3  4
    
    Automatically generate column names for conversion:

    >>> @Dataify(enforce_df=True, auto_columns=True, prefix='feature_')
    ... def process_data(data):
    ...     print(data)

    >>> process_data([[1, 2], [3, 4]])
       feature_0  feature_1
    0          1          2
    1          3          4

    Specify column names and handle mismatches silently:

    >>> @Dataify(enforce_df=True, columns=['A'], ignore_mismatch=True)
    ... def summarize_data(data):
    ...     print(data.describe())

    >>> summarize_data([[1, 2, 3], [4, 5, 6]])
           col_0
    count    2.0
    mean     2.5
    std      2.5
    min      1.0
    25%      1.75
    50%      2.5
    75%      3.25
    max      4.0

    Notes
    -----
    - The decorated function must accept its first positional argument as data.
    - This class is beneficial for functions expected to work with data in 
      pandas DataFrame format, automating input data conformity checks.
    """

    def __init__(
        self, enforce_df=True, 
        auto_columns=False,
        prefix='col_', 
        columns=None, 
        ignore_mismatch=False, 
        fail_silently=False
        ):
        self.enforce_df = enforce_df
        self.auto_columns = auto_columns
        self.prefix = prefix
        self.columns = columns
        self.ignore_mismatch = ignore_mismatch
        self.fail_silently = fail_silently

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enforce_df or not args:
                return func(*args, **kwargs)

            data = args[0]
            if not isinstance(data, pd.DataFrame):
                try:
                    data = self._attempt_dataframe_conversion(data, **kwargs)
                except ValueError as e:
                    if self.fail_silently:
                        warnings.warn(f"Dataify Warning: {e}")
                        return func(*args, **kwargs)
                    else:
                        raise
                        
            return func(data, *args[1:], **kwargs)
        return wrapper

    def _attempt_dataframe_conversion(self, data, **kwargs):
        """
        Attempts to convert the input data to a pandas DataFrame using 
        the specified columns if applicable, handling dimension mismatches.

        Parameters
        ----------
        data : array-like, Iterable, dict, or DataFrame
            The data to convert to a DataFrame.
        **kwargs : dict
            Additional keyword arguments passed to the decorated function, 
            potentially including 'columns' for specifying DataFrame column 
            names.

        Returns
        -------
        pd.DataFrame
            The data converted to a pandas DataFrame.

        Raises
        ------
        ValueError
            If the conversion fails due to incompatible data or column 
            specifications, unless `fail_silently` is True.
            
        Examples
        --------
        >>> @Dataify(auto_columns=True, prefix='feature_')
        ... def process_data(data):
        ...     print(data.head())
        
        >>> process_data(np.random.rand(5, 3))
           feature_0  feature_1  feature_2
        0   0.123456   0.654321   0.789012
        1   0.234567   0.765432   0.890123
        2   0.345678   0.876543   0.901234
        3   0.456789   0.987654   0.012345
        4   0.567890   0.098765   0.123456
        Notes
        -----
        This method is a private helper intended for internal use by the 
        Dataify decorator to manage DataFrame conversion.
        """
        # implement the new parameters here
        columns = kwargs.get('columns', self.columns)
        if isinstance (columns, str): 
            columns =[columns]
        # Automatically generate column names if required
        if self.auto_columns and columns is None:
            num_cols = np.shape(data)[1] if np.ndim(data) > 1 else 1
            columns = [f"{self.prefix}{i}" for i in range(num_cols)]

        try:
            # Construct DataFrame, auto-generating column names if needed
            df = pd.DataFrame(data, columns=columns)
        except Exception as e:
            if self.ignore_mismatch:
                # Re-try without columns if ignoring mismatches
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Error converting data to DataFrame: {e}")
        
        return df  
    
class EnsureFileExists:
    """
    Class decorator to ensure a file or URL exists before calling the 
    decorated function.

    This decorator checks if the specified file or URL exists before executing  
    the decorated function. If the file does not exist, it raises a
    FileNotFoundError. If the URL does not exist, it raises a ConnectionError. 
    The decorator can be configured to print verbose messages during the check.
    It also handles other data types based on the specified action.

    Parameters
    ----------
    file_param : int or str, optional
        The index of the parameter that specifies the file path or URL or 
        the name of the keyword argument (default is 0). If an integer is 
        provided, it refers to the position of the argument in the function 
        call. If a string is provided, it refers to the keyword argument name.
    verbose : bool, optional
        If True, prints messages indicating the file or URL check status 
        (default is False).
    action : str, optional
        Action to take if the parameter is not a file or URL. Options are 
        'ignore', 'warn', or 'raise' (default is 'raise').

    Examples
    --------
    Basic usage with verbose output:
    
    >>> from gofast.decorators import EnsureFileExists
    >>> @EnsureFileExists(verbose=True)
    ... def process_data(file_path: str):
    ...     print(f"Processing data from {file_path}")
    >>> process_data("example_file.txt")

    Basic usage without parentheses:
    
    >>> from gofast.decorators import EnsureFileExists
    >>> @EnsureFileExists
    ... def process_data(file_path: str):
    ...     print(f"Processing data from {file_path}")
    >>> process_data("example_file.txt")

    Checking URL existence:
    
    >>> from gofast.decorators import EnsureFileExists
    >>> @EnsureFileExists(file_param='url', verbose=True)
    ... def fetch_data(url: str):
    ...     print(f"Fetching data from {url}")
    >>> fetch_data("https://example.com/data.csv")
    
    Notes
    -----
    This decorator is particularly useful for functions that require a file path 
    or URL as an argument and need to ensure the file or URL exists before 
    proceeding with further operations. It helps in avoiding runtime errors 
    due to missing files or unreachable URLs.
    
    See Also
    --------
    os.path.isfile : Checks if a given path is an existing regular file.
    requests.head : Sends a HEAD request to a URL to check its existence.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in Python. 
           Proceedings of the 9th Python in Science Conference, 51-56.
    
    """
    def __init__(
            self, file_param: Union[int, str] = 0, 
            verbose: bool = False, 
            action: str = 'raise'
            ):
        self.file_param = file_param
        self.verbose = verbose
        self.action = action

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> any:
            # Determine the file path or URL from args or kwargs
            file_path_or_url = None
            if isinstance(self.file_param, int):
                if len(args) > self.file_param:
                    file_path_or_url = args[self.file_param]
            elif isinstance(self.file_param, str):
                file_path_or_url = kwargs.get(self.file_param)
    
            if self.verbose:
                print(f"Checking if path or URL exists: {file_path_or_url}")
    
            # Check if the file path or URL exists
            if file_path_or_url is None:
                self.handle_action(f"File or URL not specified: {file_path_or_url}")
            elif isinstance(file_path_or_url, str):
                if file_path_or_url.startswith(('http://', 'https://')):
                    if not self.url_exists(file_path_or_url):
                        self.handle_action(f"URL not reachable: {file_path_or_url}")
                    elif self.verbose:
                        print(f"URL exists: {file_path_or_url}")
                else:
                    if not os.path.isfile(file_path_or_url):
                        self.handle_action(f"File not found: {file_path_or_url}")
                    elif self.verbose:
                        print(f"File exists: {file_path_or_url}")
            else:
                if self.action == 'ignore':
                    if self.verbose:
                        print(f"Ignoring non-file, non-URL argument: {file_path_or_url}")
                elif self.action == 'warn':
                    warnings.warn(f"Non-file, non-URL argument provided: {file_path_or_url}")
                else:
                    raise TypeError(f"Invalid file or URL argument: {file_path_or_url}")
    
            return func(*args, **kwargs)
    
        return wrapper

    def handle_action(self, message: str):
        """
        Handle the action based on the specified action parameter.

        Parameters
        ----------
        message : str
            The message to display or include in the raised exception.
        """
        if self.action == 'ignore':
            if self.verbose:
                print(f"Ignoring: {message}")
        elif self.action == 'warn':
            warnings.warn(message)
        elif self.action == 'raise':
            raise FileNotFoundError(message)
        else:
            raise ValueError(f"Invalid action: {self.action}")

    @staticmethod
    def url_exists(url: str) -> bool:
        """
        Check if a URL exists.

        Parameters
        ----------
        url : str
            The URL to check.

        Returns
        -------
        bool
            True if the URL exists, False otherwise.
        """
        import requests
        try:
            response = requests.head(url, allow_redirects=True)
            return response.status_code == 200
        except requests.RequestException:
            return False

    @classmethod
    def ensure_file_exists(
        cls, func: Optional[Callable] = None, *, 
        file_param: Union[int, str] = 0, 
        verbose: bool = False, 
        action: str = 'raise'):
        """
        Class method to allow the decorator to be used without parentheses.

        This method enables the decorator to be applied directly without 
        parentheses, by using the first positional argument as the file or URL 
        to check. It also allows setting the `file_param`, `verbose`, and `action`
        parameters when called with parentheses.

        Parameters
        ----------
        func : Callable, optional
            The function to be decorated.
        file_param : int or str, optional
            The index of the parameter that specifies the file path or URL 
            or the name of the keyword argument (default is 0).
        verbose : bool, optional
            If True, prints messages indicating the file or URL check status 
            (default is False).
        action : str, optional
            Action to take if the parameter is not a file or URL. 
            Options are 'ignore', 'warn', or 'raise' (default is 'raise').

        Returns
        -------
        Callable
            The decorated function with file or URL existence check.

        Examples
        --------
        >>> from gofast.decorators import EnsureFileExists
        >>> @EnsureFileExists(verbose=True)
        ... def process_data(file_path: str):
        ...     print(f"Processing data from {file_path}")
        >>> process_data("example_file.txt")

        >>> from gofast.decorators import EnsureFileExists
        >>> @EnsureFileExists
        ... def process_data(file_path: str):
        ...     print(f"Processing data from {file_path}")
        >>> process_data("example_file.txt")
        """
        if func is not None:
            return cls(file_param, verbose, action)(func)
        return cls(file_param, verbose, action)

# Allow decorator to be used without parentheses
EnsureFileExists = EnsureFileExists.ensure_file_exists

@NumpyDocstringFormatter(include_sections=['Parameters', 'Returns'], validate_with_sphinx=True)
def example_function(param1, param2=None):
    """
    This is an example function that demonstrates the usage of the NumpyDocstringFormatter.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : int, optional
        The second parameter (default is None).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    return True
    
