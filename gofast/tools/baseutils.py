# -*- coding: utf-8 -*-
from __future__ import annotations 
import os 
import copy 
import time
import shutil
import inspect
import pathlib
import warnings
import functools 
import threading 
import subprocess 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
from scipy.signal import argrelextrema 
from scipy.interpolate import interp1d, griddata
from matplotlib.ticker import FixedLocator
from sklearn.utils import all_estimators 
from collections.abc import Iterable as IterableInstance

from ..api.property import  Config
from ..api.types import Union, List, Optional, Tuple, Iterable, Any, Set 
from ..api.types import _T, _F, DataFrame, ArrayLike, Series, NDArray
from ..compat.scipy import check_scipy_interpolate
from ..decorators import Dataify 
from ..exceptions import FileHandlingError
from ._dependency import import_optional_dependency
from .coreutils import is_iterable , ellipsis2false, smart_format  
from .coreutils import to_numeric_dtypes, validate_feature
from .coreutils import _assert_all_types, exist_features, reshape
from .validator import check_consistent_length, get_estimator_name
from .validator import _is_arraylike_1d, array_to_frame, build_data_if
from .validator import _is_numeric_dtype, check_y, check_consistency_size 
from .validator import is_categorical, is_valid_policies, contains_nested_objects 
from .validator import parameter_validator, normalize_array

__all__= [ 
    'array2hdf5',
    'binning_statistic',
    'categorize_target',
    'category_count',
    'denormalizer',
    'extract_target',
    'fancier_downloader',
    'fillNaN',
    'get_target',
    'interpolate_grid',
    'interpolate_data', 
    'labels_validator',
    'normalizer',
    'remove_outliers',
    'remove_target_from_array',
    'rename_labels_in',
    'save_or_load',
    'scale_y',
    'select_features',
    'select_features',
    'smooth1d',
    'smoothing',
    'soft_bin_stat',
    'speed_rowwise_process'
    ]

def remove_outliers(
    ar: ArrayLike|DataFrame, /, 
    method: str = 'IQR',
    threshold: float = 3.0,
    fill_value: Optional[float] = None,
    axis: int = 1,
    interpolate: bool = False,
    kind: str = 'linear'
) -> ArrayLike|DataFrame:
    """
    Efficient strategy to remove outliers in the data. 
    
    An outlier is a data point in a sample, observation, or distribution 
    that lies outside the overall pattern. A commonly used rule is to 
    consider a data point an outlier if it is more than 1.5 * IQR below 
    the first quartile or above the third quartile.
    
    Two approaches are used to identify and remove outliers:

    - Inter Quartile Range (``IQR``)
      IQR is the most commonly used and most trusted approach in 
      research. Outliers are defined as points lying below Q1 - 1.5 * IQR 
      or above Q3 + 1.5 * IQR. The quartiles and IQR are calculated as follows:
      
      .. math::
          
        Q1 = \frac{1}{4}(n + 1),\\
        Q3 = \frac{3}{4}(n + 1),\\
        IQR = Q3 - Q1,\\
        \text{Upper} = Q3 + 1.5 \times IQR,\\
        \text{Lower} = Q1 - 1.5 \times IQR.
    
    - Z-score 
      Also known as a standard score, this value helps to understand how 
      far a data point is from the mean. After setting a threshold, data 
      points with Z-scores beyond this threshold are considered outliers.
      
      .. math::
          
          \text{Zscore} = \frac{(\text{data\_point} - \text{mean})}{\text{std\_deviation}}
      
    A threshold value of generally 3.0 is chosen as 99.7% of data points 
    lie within +/- 3 standard deviations in a Gaussian Distribution.

    Parameters
    ----------
    ar : Union[np.ndarray, pd.DataFrame]
        The input data from which to remove outliers, either a numpy array 
        or pandas DataFrame.
    method : str, default 'IQR'
        Method to detect and remove outliers:
        - ``'IQR'`` uses the Inter Quartile Range to define outliers.
        - ``'Z-score'`` identifies outliers based on standard deviation.
        See detailed explanation on how each method works in the function's 
        description.
    threshold : float, default 3.0
        For ``'Z-score'``, this is the number of standard deviations a data point
        must be from the mean to be considered an outlier. For 'IQR', 
        this multiplies the IQR range.
    fill_value : float, optional
        Value used to replace outliers. If None, outliers are removed from 
        the dataset.
    axis : int, default 1
        Specifies the axis along which to remove outliers, applicable to 
        multi-dimensional data.
    interpolate : bool, default False
        Enables interpolation to estimate and replace outliers, only applicable
        if `fill_value` is NaN.
    kind : str, default 'linear'
        Type of interpolation used if interpolation is True. Options include
        ``'nearest'``, ``'linear'``, ``'cubic'``.
    
    Returns
    -------
    Union[np.ndarray, DataFrame]
        Array or DataFrame with outliers removed or replaced.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.baseutils import remove_outliers 
    >>> np.random.seed(42)
    >>> data = np.random.randn(7, 3)
    >>> data_r = remove_outliers(data)
    >>> data.shape, data_r.shape
    (7, 3), (5, 3)
    >>> remove_outliers(data, fill_value=np.nan)
    array([[ 0.49671415, -0.1382643 ,  0.64768854],
           [ 1.52302986, -0.23415337, -0.23413696],
           [ 1.57921282,  0.76743473, -0.46947439],
           [ 0.54256004, -0.46341769, -0.46572975],
           [ 0.24196227,         nan,         nan],
           [-0.56228753, -1.01283112,  0.31424733],
           [-0.90802408,         nan,  1.46564877]])
    >>> remove_outliers(data[:, 0], fill_value=np.nan, interpolate=True)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(np.arange(len(data)), data, 'r-')
    """
    method = str(method).lower()
    
    if isinstance (ar, pd.DataFrame):
        return _remove_outliers( ar, n_std= threshold )
    
    is_series = isinstance ( ar, pd.Series)
    arr =np.array (ar, dtype = float)

    if method =='iqr': 
        Q1 = np.percentile(arr[~np.isnan(arr)], 25,) 
        Q3 = np.percentile(arr[~np.isnan(arr)], 75)
        IQR = Q3 - Q1
        
        upper = Q3 + 1.5 * IQR  
        
        upper_arr = np.array (arr >= upper) 
        lower = Q3 - 1.5 * IQR 
        lower_arr =  np.array ( arr <= lower )
        # replace the oulier by nan 
        arr [upper_arr]= fill_value if fill_value else np.nan  
        arr[ lower_arr]= fill_value if fill_value else np.nan 
        
    if method =='z-score': 
        from scipy import stats
        z = np.abs(stats.zscore(arr[~np.isnan(arr)]))
        zmask  = np.array ( z > threshold )
        arr [zmask]= fill_value if fill_value else np.nan
        
    if fill_value is None: 
        # delete nan if fill value is not provided 
        arr = arr[ ~np.isnan (arr ).any(axis =1)
                  ]  if np.ndim (arr) > 1 else arr [~np.isnan(arr)]

    if interpolate: 
        arr = interpolate_grid (arr, method = kind)
        
    if is_series: 
        arr =pd.Series (arr.squeeze(), name =ar.name )
        
    return arr 

def _remove_outliers(data, n_std=3):
    """Remove outliers from a dataframe."""
    # separate cat feature and numeric features 
    # if exists 
    df, numf, catf = to_numeric_dtypes(
        data , return_feature_types= True,  drop_nan_columns =True )
    # get on;y the numeric 
    df = df[numf]
    for col in df.columns:
        # print('Working on column: {}'.format(col))
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean+(n_std*sd))]
    # get the index and select only the index 
    index = df.index 
    # get the data by index then 
    # concatename 
    df_cat = data [catf].iloc [ index ]
    df = pd.concat ( [df_cat, df ], axis = 1 )
    
    return df

def interpolate_grid (
    arr, / , 
    method ='cubic', 
    fill_value='auto', 
    view = False,
    ): 
    """
    Interpolate data containing missing values. 

    Parameters 
    -----------
    arr: ArrayLike2D 
       Two dimensional array for interpolation 
    method: str, default='cubic'
      kind of interpolation. It could be ['nearest'|'linear'|'cubic']. 
     
    fill_value: float, str, default='auto' 
       Fill the interpolated grid at the egdes or surrounding NaN with 
       a filled value. The ``auto`` uses the forward and backward 
       fill strategy. 
       
    view: bool, default=False, 
       Quick visualize the interpolated grid. 
 
    Returns 
    ---------
    arri: ArrayLike2d 
       Interpolated 2D grid. 
       
    See also 
    ---------
    spi.griddata: 
        Scipy interpolate Grid data 
    fillNaN: 
        Fill missing data strategy. 
        
    Examples
    ---------
    >>> import numpy as np
    >>> from gofast.tools.coreutils import interpolate_grid 
    >>> x = [28, np.nan, 50, 60] ; y = [np.nan, 1000, 2000, 3000]
    >>> xy = np.vstack ((x, y))._T
    >>> xyi = interpolate_grid (xy, view=True ) 
    >>> xyi 
    array([[  28.        ,   28.        ],
           [  22.78880663, 1000.        ],
           [  50.        , 2000.        ],
           [  60.        , 3000.        ]])

    """
    spi = check_scipy_interpolate()
    if spi is None:
        return None
    
    is2d = True 
    if not hasattr(arr, '__array__'): 
        arr = np.array (arr) 
    
    if arr.ndim==1: 
        #convert to two dimension array
        arr = np.vstack ((arr, arr ))
        is2d =False 
        # raise TypeError(
        #     "Expect two dimensional array for grid interpolation.")
        
    # make x, y array for mapping 
    x = np.arange(0, arr.shape[1])
    y = np.arange(0, arr.shape[0])
    #mask invalid values
    arr= np.ma.masked_invalid(arr) 
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~arr.mask]
    y1 = yy[~arr.mask]
    newarr = arr[~arr.mask]
    
    arri = spi.griddata(
        (x1, y1),
        newarr.ravel(),
        (xx, yy), 
        method=method
        )
    
    if fill_value =='auto': 
        arri = fillNaN(arri, method ='both ')
    else:
        arri [np.isnan(arri)] = float( _assert_all_types(
            fill_value, float, int, objname ="'fill_value'" )
            ) 

    if view : 
        fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True, )
        ax[0].imshow(arr ,interpolation='nearest', label ='Raw Grid')
        ax[1].imshow (arri, interpolation ='nearest', 
                      label = 'Interpolate Grid')
        
        ax[0].set_title ('Raw Grid') 
        ax[1].set_title ('Interpolate Grid') 
        
        plt.show () 
        
    if not is2d: 
        arri = arri[0, :]
        
    return arri 

def fillNaN(
    arr: Union[ArrayLike, Series, DataFrame], /, 
    method: str = 'ff'
    ) -> Union[ArrayLike, Series, DataFrame]:
    """
    Fill NaN values in a numpy array, pandas Series, or pandas DataFrame 
    using specified methods for forward filling, backward filling, or both.

    Parameters
    ----------
    arr : Union[np.ndarray, pd.Series, pd.DataFrame]
        The input data containing NaN values to be filled. This can be a numpy
        array, pandas Series,  or DataFrame expected to contain numeric data
        types.
    method : str, optional
        The method used for filling NaN values. Valid options are:
        - 'ff': forward fill (default)
        - 'bf': backward fill
        - 'both': applies both forward and backward fill sequentially

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        The array with NaN values filled according to the specified method. 
        The return type matches the input type (numpy array, Series, or DataFrame).

    Raises
    ------
    ValueError
        If an unsupported filling method is specified.

    Notes
    -----
    The function is designed to handle scenarios where NaN values are 
    framed between valid numbersand at the ends of the dataset. Forward fill 
    (``ff``) is preferred when NaNs are at the end of the data, while backward 
    fill (``bf``) is better suited for NaNs at the beginning. The `both` method
    combines both fills but at a higher computation cost.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import fillNaN 
    >>> arr2d = np.random.randn(7, 3)
    >>> arr2d[[0, 2, 3, 3], [0, 2, 1, 2]] = np.nan
    >>> print(arr2d)
    [[       nan -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895        nan]
     [-0.6839212         nan        nan]]
    >>> print(fillNaN(arr2d))
    [[       nan -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895 -2.15184895]
     [-0.6839212 -0.6839212 -0.6839212]]
    >>> print(fillNaN(arr2d, 'bf'))
    [[-0.74636104 -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895        nan]
     [-0.6839212         nan        nan]]
    >>> print(fillNaN(arr2d, 'both'))
    [[-0.74636104 -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895 -2.15184895]
     [-0.6839212 -0.6839212 -0.6839212]]

    References
    ----------
    Further details can be found at:
    https://pyquestions.com/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    name_or_columns=None 
    if not hasattr(arr, '__array__'): 
        arr = np.array(arr)
        
    arr = _handle_non_numeric(arr, action ='fill missing values NaN')
    
    if isinstance (arr, (pd.Series, pd.DataFrame)): 
        name_or_columns = arr.name if isinstance (arr, pd.Series) else arr.columns 
        arr = arr.to_numpy() # get numpy array 
        
    def ffill (arr): 
        """ Forward fill."""
        idx = np.where (~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate (idx, axis =1 , out =idx )
        return arr[np.arange(idx.shape[0])[:, None], idx ]
    
    def bfill (arr): 
        """ Backward fill """
        idx = np.where (~mask, np.arange(mask.shape[1]) , mask.shape[1]-1)
        idx = np.minimum.accumulate(idx[:, ::-1], axis =1)[:, ::-1]
        return arr [np.arange(idx.shape [0])[:, None], idx ]
    
    method= str(method).lower().strip() 
    
    if arr.ndim ==1: 
        arr = reshape(arr, axis=1)  
        
    if method  in ('backward', 'bf',  'bwd'):
        method = 'bf' 
    elif method in ('forward', 'ff', 'fwd'): 
        method= 'ff' 
    elif method in ('both', 'ffbf', 'fbwf', 'bff', 'full'): 
        method ='both'
    if method not in ('bf', 'ff', 'both'): 
        raise ValueError ("Expect a backward <'bf'>, forward <'ff'> fill "
                          f" or both <'bff'> not {method!r}")
    mask = np.isnan (arr )  
    if method =='both': 
        arr = ffill(arr) ;
        #mask = np.isnan (arr)  
        arr = bfill(arr) 
    if method in ('bf', 'ff'): 
        arr = ffill(arr) if method =='ff' else bfill(arr)
        
    if name_or_columns is not None: 
        arr = pd.Series ( arr.squeeze(), name = name_or_columns) if isinstance (
            name_or_columns, str ) else pd.DataFrame(arr, columns = name_or_columns)
        
    return arr 

def convert_array_dimensions(
        *arrays, target_dim=1, new_shape=None, orient='row'):
    """
    Convert arrays between 1D, 2D, and higher dimensions. 
    
    Function dynamically adjusts the dimensions of the input arrays based on 
    the target dimension specified and can reshape arrays according to a new 
    shape.

    Parameters:
    -----------
    *arrays : tuple of array-like
        Variable number of array-like structures (lists, tuples, np.ndarray).
    target_dim : int, optional
        The target dimension to which the arrays should be converted. Options 
        are primarily 1 or 2.
        Default is 1, which flattens arrays to 1D.
    new_shape : tuple of ints, optional
        The new shape for the array when converting to 2D or reshaping a
        higher-dimensional array.
        If None, defaults are used (e.g., flattening to 1D or reshaping to 
                                    one row per array in 2D).
    orient : str or int, optional
        Specifies the orientation for reshaping the array into 2D when no 
        new_shape is provided. 
        Accepts 'row' or 0 to reshape the array into a single row 
        (default behavior), and 'column' or 1 
        to reshape the array into a single column. This parameter determines 
        the structure of the 2D array:
        - 'row' or 0: The array is reshaped to have one row with multiple 
          columns.
        - 'column' or 1: The array is reshaped to have one column with 
           multiple rows.
        If an invalid option is provided, a ValueError is raised.

    Returns:
    --------
    list
        A list of arrays converted to the specified target dimension.

    Raises:
    -------
    ValueError
        If the target_dim is not supported or if conversion is not feasible 
        for the given dimensions.
        
    Examples
    --------
    >>> from gofast.tools.baseutils import convert_array_dimensions
    >>> import numpy as np

    # Example 1: Convert a 1D array to a 2D array with a specific shape
    >>> array_1d = np.array([1, 2, 3, 4, 5])
    >>> convert_array_dimensions(array_1d, target_dim=2, new_shape=(5, 1))
    [array([[1],
            [2],
            [3],
            [4],
            [5]])]

    # Example 2: Flatten a 2D array to 1D
    >>> array_2d = np.array([[1, 2, 3], [4, 5, 6]])
    >>> convert_array_dimensions(array_2d, target_dim=1)
    [array([1, 2, 3, 4, 5, 6])]

    # Example 3: Convert a 1D array to a default 2D array (one row)
    >>> array_1d = np.array([7, 8, 9, 10])
    >>> convert_array_dimensions(array_1d, target_dim=2)
    [array([[ 7,  8,  9, 10]])]

    # Example 4: Attempt to reshape a 1D array into an 
    # incompatible 2D shape (should raise an error)
    >>> array_1d = np.array([1, 2, 3, 4, 5, 6])
    >>> convert_array_dimensions(array_1d, target_dim=2, new_shape=(2, 4))
    Traceback (most recent call last):
      ...
    ValueError: Cannot reshape array of size 6 into shape (2, 4)
    """
    converted_arrays = []
    for arr in arrays:
        # Ensure input is converted to a NumPy array for manipulation
        array = np.array(arr) 

        if target_dim == 1:
            # Flatten the array to 1D
            converted_arrays.append(array.ravel())
        elif target_dim == 2:
            if new_shape is not None:
                # Reshape according to the provided new_shape
                try:
                    converted_arrays.append(array.reshape(new_shape))
                except ValueError as e:
                    raise ValueError(f"Cannot reshape array of size {array.size}"
                                     f" into shape {new_shape}") from e
            else:
                # Skip reshaping if the array is already 2D and no new shape is provided
                if array.ndim == 2 and array.shape == (1, -1) or array.shape == (-1, 1):
                    converted_arrays.append(array)
                else:
                    # Default 2D shape based on orientation
                    if orient == 'row' or orient == 0:
                        converted_arrays.append(array.reshape(1, -1))
                    elif orient == 'column' or orient == 1:
                        converted_arrays.append(array.reshape(-1, 1))
                    else:
                        raise ValueError("orient must be 'row', 'column', 0, or 1.")
        else:
            raise ValueError(
                f"Invalid target dimension {target_dim}. Only 1 or 2 are supported.")

    return converted_arrays

def filter_nan_entries(
    nan_policy, *listof, 
    sample_weights=None, 
    mode="strict", 
    trim_weights=False
    ):
    """
    Filters out NaN values from multiple lists of lists, or arrays, 
    based on the specified NaN handling policy ('omit', 'propagate', 'raise'), 
    and adjusts the sample weights accordingly if provided.

    This function is particularly useful when preprocessing data for
    machine learning algorithms that do not support NaN values or when NaN values
    signify missing data that should be excluded from analysis.

    Parameters
    ----------
    nan_policy : {'omit', 'propagate', 'raise'}
        The policy for handling NaN values.
        - 'omit': Exclude NaN values from all lists in `listof`. 
        - 'propagate': Keep NaN values, which may result in NaN values in output.
        - 'raise': If NaN values are detected, raise a ValueError.
    *listof : array-like sequences
        Variable number of list-like sequences from which NaN values are to be 
        filtered out.  Each sequence in `listof` must have the same length.
    sample_weights : array-like, optional
        Weights corresponding to the elements in each sequence in `listof`.
        Must have the same length as the sequences. If `nan_policy` is 'omit',
        weights are adjusted to match the filtered sequences.
    mode : str, optional
        Specifies the mode of NaN filtering:
        - 'strict': Indices are retained only if all corresponding elements 
          across sequences are non-NaN.
        - 'soft': Indices are retained if any corresponding element across 
          sequences is non-NaN.
    trim_weights : bool, optional
        If True and `sample_weights` is provided, trims the sample_weights to
        match the length of the filtered data when the 'soft' mode results in
        fewer data points than there are weights.
        
    Returns
    -------
    tuple of lists
        A tuple containing the filtered list-like sequences as per the specified
        `nan_policy`.
        If `nan_policy` is 'omit', sequences with NaN values removed are 
        returned.
    np.ndarray or None
        The adjusted sample weights matching the filtered sequences if 
        `nan_policy` is 'omit'.
        If `sample_weights` is not provided, None is returned.

    Raises
    ------
    ValueError
        If `nan_policy` is 'raise' and NaN values are present in any input 
        sequence.

    Examples
    --------
    >>> from gofast.tools.baseutils import filter_nan_entries
    >>> list1 = [1, 2, np.nan, 4]
    >>> list2 = [np.nan, 2, 3, 4]
    >>> weights = [0.5, 1.0, 1.5, 2.0]
    >>> filter_nan_entries('omit', list1, list2,sample_weights=weights,
                           mode="soft" ,)
    ([1.0, 2.0, 4.0], [2.0, 3.0, 4.0], array([0.5, 1. , 1.5, 2. ]))

    >>> filter_nan_entries('omit', list1, list2,sample_weights=weights, )
    ([2.0, 4.0], [2.0, 4.0], array([1., 2.]))
    >>> filter_nan_entries(
    ...     'omit', list1, list2,sample_weights=weights, mode="soft" ,
    ... trim_weights=True)
    ([1.0, 2.0, 4.0], [2.0, 3.0, 4.0], array([0.5, 1. , 1.5])) 
    >>> filter_nan_entries('raise', list1, list2)
    ValueError: NaN values present and nan_policy is 'raise'.

    Notes
    -----
    This function is designed to work with numerical data where NaN values
    may indicate missing data. It allows for flexible preprocessing by supporting
    multiple NaN handling strategies, making it suitable for various machine learning
    and data analysis workflows.

    When using 'omit' in 'soft' mode, it's important to ensure that all sequences
    in `listof`and the corresponding `sample_weights` (if provided) are 
    correctly aligned so that filtering does not introduce misalignments in the data.
    """
    # Validate nan_policy and check if any list contains nested objects
    nan_policy = is_valid_policies(nan_policy)
    
    for d in listof: 
        if contains_nested_objects(d): 
            # write a professionnal error message 
            raise ValueError ("filter_nan_entries does not support nested items.")
        
    # Prepare the data arrays
    arrays = [np.array(lst, dtype=float) for lst in listof]

    # Apply NaN filtering based on the selected policy
    filtered_arrays, non_nan_mask = _filter_nan_policies(arrays, nan_policy, mode)

    # Adjust sample weights if necessary
    if sample_weights is not None:
        sample_weights = _adjust_weights(
            sample_weights, non_nan_mask, mode, trim_weights, filtered_arrays
            )

    # Prepare the output to be returned
    filtered_listof = [arr.tolist() for arr in filtered_arrays]
    return (*filtered_listof, sample_weights
            ) if sample_weights is not None else tuple(filtered_listof)


def _filter_nan_policies(arrays, policy, mode):
    """Apply the NaN policy to filter the data arrays."""
    mode = parameter_validator("mode", target_strs={"soft", "strict"})(mode)
    if policy == 'omit':
        if mode == "strict":
            non_nan_mask = np.logical_and.reduce(~np.isnan(arrays))
            # Filter arrays using the computed mask
            filtered_arrays = [arr[non_nan_mask] for arr in arrays]
        elif mode == "soft":
           non_nan_mask = np.logical_or.reduce(~np.isnan(arrays))
           # Ensure that elements are NaN only if they are NaN across all arrays
           filtered_arrays =[arr[~np.isnan(arr)] for arr in  arrays]
    
    elif policy == 'raise':
        # Raise an error if any NaN values are detected
        if any(np.isnan(arr).any() for arr in arrays):
            raise ValueError("NaN values present and nan_policy is 'raise'.")
    else:
        # If 'propagate' is selected, return the original arrays
        filtered_arrays = arrays
    return filtered_arrays, non_nan_mask


def _adjust_weights(weights, mask, mode, adjusted_weights, arrays):
    """Adjust sample weights according to the filtered data."""
    if mode == 'soft' and adjusted_weights:
        # Determine the minimum length among the filtered arrays 
        # to ensure weights consistency
        min_length = min(len(arr) for arr in arrays)
    
        # Validate and adjust the non_nan_mask to ensure it aligns 
        # with the smallest array length
        if len(weights) > min_length:
            # Generate a warning if the original weights exceed the size
            # of available data
            warnings.warn(f"Adjusting sample weights from {len(weights)}"
                          f" to {min_length} due to mismatched data lengths.")
            
            # Resize the non_nan_mask to match the minimum length of the filtered data
            #       Indices where non_nan_mask is True
            valid_indices = np.where(mask)[0]  
            if valid_indices.size > min_length:
                # Truncate valid_indices to match the size of the shortest
                # filtered array
                valid_indices = valid_indices[:min_length]
            
            # Create a new mask that only includes the valid indices 
            # adjusted to min_length
            adjusted_non_nan_mask = np.zeros_like(mask, dtype=bool)
            adjusted_non_nan_mask[valid_indices] = True
            mask = adjusted_non_nan_mask
    
    # Update sample weights using the final adjusted non_nan_mask
    return np.asarray(weights)[mask]

def _flatten(items):
    """Helper function to flatten complex nested structures into a flat list."""
    for x in items:
        if isinstance(x, IterableInstance) and not isinstance(x, (str, bytes)):
            for sub_x in _flatten(x):
                yield sub_x
        else:
            yield x
            
def filter_nan_values(
        nan_policy, *data_lists, sample_weights=None, error="raise",
        flatten=False, preserve_type=False):
    """
    Filters out NaN values from provided lists based on a specified policy,
    adjusts sample weights if necessary, and can optionally flatten the input lists.

    Parameters
    ----------
    nan_policy : {'omit', 'propagate', 'raise'}
        Determines how NaN values are handled:
        - 'omit': Removes NaN values from the lists.
        - 'propagate': Keeps NaN values in the lists.
        - 'raise': Raises an error if NaN values are found.
    data_lists : list of lists or arrays
        Variable number of list-like or array-like sequences from which NaN
        values are to be filtered based on the `nan_policy`.
    sample_weights : array-like, optional
        Weights corresponding to each entry in `data_lists`. If provided,
        they are adjusted to match the filtering operation.
    error : {'raise', 'warn'}, default 'raise'
        Error handling strategy if sample weights do not match the number
        of entries after filtering.
    flatten : bool, default False
        If True, flattens each list in `data_lists` before applying the filter.
    preserve_type : bool, default False
        If True, preserves the original type of nested structures within `data_lists`.

    Returns
    -------
    tuple
        A tuple containing the filtered lists. If `sample_weights` is provided,
        it is included as the last element of the tuple.

    Raises
    ------
    ValueError
        If `nan_policy` is 'raise' and NaN values are present.

    Examples
    --------
    >>> from gofast.tools.baseutils import filter_nan_values
    >>> list1 = [{2, 3}, {1, 2, np.nan}]
    >>> list2 = [{1, 2, 3}, {1, 2, 3, np.nan}]
    >>> weights = [0.5, 1.0, 1.5, 2.0]
    >>> print(filter_nan_values('omit', list1, list2, sample_weights=weights,
    ...             error="warn", flatten=True, preserve_type=True))
    ({1, 2, 3}, {1, 2, 3}, [0.5, 1.0, 1.5, 2.0])

    >>> filter_nan_values('raise', data1, error='warn')
    ValueError: NaN values present and nan_policy is 'raise'.

    Notes
    -----
    This function is useful for pre-processing data for algorithms that do not
    support NaN values or require array-like input with optional weighting. The
    `flatten` option is particularly useful for handling nested lists or arrays.
    """
    # Validate nan_policy
    if nan_policy not in ['omit', 'propagate', 'raise']:
        raise ValueError(f"Invalid nan_policy: {nan_policy}. Must be one of"
                         " 'omit', 'propagate', 'raise'.")
    
    for listof in data_lists: 
        if not contains_nested_objects(listof, strict=True):
            raise ValueError(
                "filter_nan_values expects each item in the data_lists"
                " to be a nested structure (e.g., list, set, or dict)."
                " Please ensure all elements are nested."
                )

    # Prepare arrays by flattening if requested
    arrays = _prepare_arrays(data_lists, flatten)

    # Apply NaN policy to arrays
    filtered_arrays, non_nan_mask = _apply_nan_policy(
        arrays, nan_policy, flatten)

    # Adjust sample weights according to the filtered data
    if sample_weights is not None:
        sample_weights = _adjust_sample_weights(
            sample_weights, non_nan_mask, error)

    # Prepare the output list, preserving types if required
    filtered_listof = _prepare_output(
        filtered_arrays, preserve_type, data_lists, flatten)

    # Return the filtered data, including sample weights if provided
    return (*filtered_listof, sample_weights) if sample_weights is not None else tuple(
        filtered_listof)

def adjust_weights(
    data_lengths, 
    original_weights, 
    mode="auto", 
    fill_value=None,
    normalize=False, 
    normalize_method="01"
    ):
    """
    Adjusts sample weights to match the lengths of filtered or transformed datasets.
    This function can handle scenarios where the filtered data is shorter than the
    original weights array, either by truncating or by filling the remaining weights.

    Normalization can be applied to adjust the scale of the weights after 
    adjusting their length.
    
    Parameters
    ----------
    data_lengths : list or array-like
        A list of integers representing the lengths of the filtered datasets.
        This is used to determine how to adjust the sample weights array.
    original_weights : array-like
        The original weights array that needs adjustment to match the filtered data.
    mode : str, optional
        Specifies the method for adjusting the weights:
        - 'auto': Automatically decides based on the data_lengths. If any length in
          data_lengths is less than the length of original_weights, it trims the weights;
          otherwise, it fills or repeats weights if fill_value is not None.
        - 'trim': Trims the weights to match the shortest length in data_lengths.
        - 'fill': Extends the weights to match the longest length in data_lengths,
          using fill_value for the new elements.
    fill_value : float or None, optional
        The value used to fill the weights array if 'fill' mode is active and the
        weights need to be extended. If None and 'fill' is required, an error will
        be raised.
    normalize : bool or str, optional
        If True normalizes the weights after adjusting their length.
        If False, no normalization is applied.
    normalize_method : str, optional
        Normalization method ('01' for 0-1 scaling, 'zscore' for Z-score, or 'sum'
        to scale by the sum of weights).
    Returns
    -------
    np.ndarray
        The adjusted weights array that matches the specified requirements based on the
        filtered data lengths.

    Raises
    ------
    ValueError
        If 'fill' mode is selected but fill_value is None and the weights need
        extending.

    Examples
    --------
    >>> from gofast.tools.baseutils import adjust_weights
    >>> weights = [1, 2, 3, 4, 5]
    >>> data_lengths = [3, 4]
    >>> adjust_weights(data_lengths, weights, match_mode='trim')
    array([1, 2, 3])

    >>> adjust_weights(data_lengths, weights, match_mode='fill', fill_value=0)
    array([1, 2, 3, 4, 5, 0])

    >>> adjust_weights(data_lengths, weights, mode='auto', fill_value=0, 
    ...                normalize=True, normalize_method='01')
    array([0.2, 0.4, 0.6, 0.8, 1. , 0. ])
    
    Notes
    -----
    Normalization is applied after adjusting the weights to ensure that the 
    processed weights are ready for use in weighted statistical analyses or models.
    The choice between trimming and filling depends on the analysis needs and 
    the nature of the data. Care should be taken with 'fill' mode as it 
    introduces artificial values.
    """
    # Validate input types and values
    if not isinstance(data_lengths, (list, tuple, np.ndarray)):
        raise TypeError("data_lengths must be a list, tuple, or numpy array.")
    if not isinstance(original_weights, (list, np.ndarray)):
        raise TypeError("original_weights must be a list or numpy array.")
    
    # Convert inputs to numpy arrays for uniformity
    data_lengths = np.array(data_lengths)
    original_weights = np.array(original_weights)
    
    # Determine the required length based on the match mode
    if mode == 'auto':
        max_length = max(data_lengths)
        min_length = min(data_lengths)
        required_length = max_length if len(original_weights) < max_length else min_length
    elif mode == 'trim':
        required_length = min(data_lengths)
    elif mode == 'fill':
        required_length = max(data_lengths)
    else:
        raise ValueError("Invalid match_mode. Choose from 'auto', 'trim', or 'fill'.")

    # Adjust weights according to the required length
    if len(original_weights) > required_length:
        adjusted_weights = original_weights[:required_length]
    elif len(original_weights) < required_length and mode == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided when extending"
                             " weights in 'fill' mode.")
        extra_length = required_length - len(original_weights)
        adjusted_weights = np.concatenate([original_weights, np.full(
            extra_length, fill_value)])
    else:
        adjusted_weights = original_weights

    # Normalize weights if requested
    if normalize:
        adjusted_weights = normalize_array(adjusted_weights, method=normalize_method)
    
    return adjusted_weights


def _prepare_output(filtered_arrays, preserve_type, original_lists, flatten):
    """
    Prepares the output list from filtered arrays, optionally preserving the 
    original data types of the nested structures.

    Parameters
    ----------
    filtered_arrays : list of arrays
        Arrays that have been filtered based on NaN policies.
    preserve_type : bool
        Whether to preserve the original types of nested structures in the output.
    original_lists : list of lists
        The original list of lists provided by the user, used for type reference
        if `preserve_type` is True.
   flatten : bool, default False
       If True, flattens each list in `data_lists` before applying the filter.
    Returns
    -------
    list
        The list of filtered arrays, with types preserved if `preserve_type` is True.

    Notes
    -----
    This function aids in maintaining the integrity of the data types in nested 
    structures when required by the user. If `preserve_type` is False, all nested
    structures are converted to lists. If True, the original data structure type 
    (e.g., set, list) is preserved based on the type of the first element in each
    corresponding original list.
    """
    filtered_listof = []
    for arr, original in zip(filtered_arrays, original_lists):
        original_type = type(original[0]) if original else list
        if preserve_type and not flatten:
            # Determine the type of the original nested structure
            filtered_sublistof = [original_type(subarr) for subarr in arr]
        else:
            # Convert all sub-arrays to lists
            filtered_sublistof = original_type(arr)
        filtered_listof.append(filtered_sublistof)
    return filtered_listof

def _prepare_arrays(listof, flatten_lst):
    """Prepares and converts lists to numpy arrays, handling flattening if specified."""
    arrays = []
    for lst in listof:
        if flatten_lst:
            # Flatten and then convert to numpy array
            flattened_list = list(_flatten(lst))
            arrays.append(np.array(flattened_list, dtype=object))
        else:
            # Apply filtering to each nested element directly
            arrays.append(np.array([np.array(list(item), dtype=object) if isinstance(
                item, (list, set)) else item for item in lst], dtype=object))
    
    return arrays

def _apply_nan_policy(arrays, nan_policy, flatten_lst):
    """Applies the specified NaN policy to the arrays."""
    filtered_arrays = []
    all_non_nan_mask = []
    for arr in arrays:
        if nan_policy == 'omit':
            if flatten_lst: 
                # Create a mask for non-NaN entries
                if arr.dtype == object:
                    non_nan_mask = np.array([not isinstance(
                        x, float) or not np.isnan(x) for x in arr])
                else:
                    non_nan_mask = ~np.isnan(arr)
                filtered_array = arr[non_nan_mask]
                filtered_arrays.append(filtered_array)
                
            else: 
               filtered_subarrays = []
               for subarr in arr:
                   if isinstance(subarr, np.ndarray):
                       # Apply the non-NaN mask to each sub-array
                       non_nan_mask = ~np.isnan(subarr.astype(float))
                       filtered_subarrays.append(subarr[non_nan_mask])
                   else:
                       # If it's not an array, just append as is
                       filtered_subarrays.append(subarr)
               filtered_arrays.append(filtered_subarrays) 

            all_non_nan_mask.append(non_nan_mask)
        elif nan_policy == 'raise' and np.isnan(arr.astype(float)).any():
            raise ValueError("NaN values present and nan_policy is 'raise'.")
        else:
            filtered_arrays.append(arr)
            
    return filtered_arrays, all_non_nan_mask

def _adjust_sample_weights(sample_weights, non_nan_mask, error):
    """Adjusts sample weights to match the filtered data's length."""
    try: 
        total_entries = sum(non_nan_mask) # if flatten_lst 
    except: 
        total_entries = sum(len(arr) for arr in non_nan_mask )
    if len(sample_weights) == total_entries:
        return np.asarray(sample_weights)[non_nan_mask]
    elif error == 'warn':
        # If the length of the sample weights does not match but is greater,
        # reduce the sample weights array to match the filtered data length.
        # Note: This may not always be the desired behavior, as it can lead to
        #       unexpected results if data misalignment occurs.
        #       Use with caution or consider whether this should raise an error instead.
        warnings.warn("Length of sample_weights does not match the number"
                      " of entries after NaN filtering.")
    elif error == 'raise':
        raise ValueError("Length of sample_weights must match the number of entries in listof.")
    return sample_weights


def filter_nan_from( *listof, sample_weights=None):
    """
    Filters out NaN values from multiple lists of lists, adjusting 
    sample_weights accordingly.

    Parameters
    ----------
    *listof : tuple of list of lists
        Variable number of list of lists from which NaN values need to be 
        filtered out.
    sample_weights : list or np.ndarray, optional
        Sample weights corresponding to each sublist in the input list of lists.
        Must have the same outer length as each list in *listof.

    Returns
    -------
    filtered_listof : tuple of list of lists
        The input list of lists with NaN values removed.
    adjusted_sample_weights : np.ndarray or None
        The sample weights adjusted to match the filtered data. Same length as
        the filtered list of lists.

    Examples
    --------
    >>> from gofast.tools.baseutils import filter_nan_from
    >>> list1 = [[1, 2, np.nan], [4, np.nan, 6]]
    >>> list2 = [[np.nan, 8, 9], [10, 11, np.nan]]
    >>> weights = [0.5, 1.0]
    >>> filtered_lists, adjusted_weights = filter_nan_from(
        list1, list2, sample_weights=weights)
    >>> print(filtered_lists)
    ([[1, 2], [4, 6]], [[8, 9], [10, 11]])
    >>> print(adjusted_weights)
    [0.5 1. ]

    Notes
    -----
    This function assumes that all lists in *listof and sample_weights have 
    compatible shapes. Each sublist is expected to correspond to a set of 
    sample_weights, which are adjusted based on the presence of NaNs in 
    the sublist.
    """
    import math 
    if sample_weights is not None and len(sample_weights) != len(listof[0]):
        raise ValueError(
            "sample_weights length must match the number of sublists in listof.")

    # Convert sample_weights to a numpy array for easier manipulation
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    filtered_listof = []
    valid_indices = set(range(len(listof[0])))  # Initialize with all indices as valid

    # Identify indices with NaNs across all lists
    for lst in listof:
        for idx, sublist in enumerate(lst):
            if any(math.isnan(item) if isinstance(item, (
                    float, np.floating)) else False for item in sublist):
                valid_indices.discard(idx)

    # Filter lists based on valid indices
    for lst in listof:
        filtered_list = [lst[idx] for idx in sorted(valid_indices)]
        filtered_listof.append(filtered_list)

    # Adjust sample_weights based on valid indices
    adjusted_sample_weights = sample_weights[
        sorted(valid_indices)] if sample_weights is not None else None

    return tuple(filtered_listof), adjusted_sample_weights

def standardize_input(*arrays):
    """
    Standardizes input formats for comparison metrics, converting input data
    into a uniform format of lists of sets. This function can handle a variety
    of input formats, including 1D and 2D numpy arrays, lists of lists, and
    tuples, making it versatile for tasks that involve comparing lists of items
    like ranking or recommendation systems.

    Parameters
    ----------
    *arrays : variable number of array-like or list of lists
        Each array-like argument represents a set of labels or items, such as 
        `y_true` and `y_pred`. The function is designed to handle:
        
        - 1D arrays where each element represents a single item.
        - 2D arrays where rows represent samples and columns represent items
          (for multi-output scenarios).
        - Lists of lists or tuples, where each inner list or tuple represents 
          a set of items for a sample.

    Returns
    -------
    standardized : list of lists of set
        A list containing the standardized inputs as lists of sets. Each outer
        list corresponds to one of the input arrays, and each inner list 
        corresponds to a sample within that array.

    Raises
    ------
    ValueError
        If the lengths of the input arrays are inconsistent.
    TypeError
        If the inputs are not array-like, lists of lists, or lists of tuples,
        or if an ndarray has more than 2 dimensions.

    Examples
    --------
    >>> from numpy import array
    >>> from gofast.tools.baseutils import standardize_input
    >>> y_true = [[1, 2], [3]]
    >>> y_pred = array([[2, 1], [3]])
    >>> standardized_inputs = standardize_input(y_true, y_pred)
    >>> for standardized in standardized_inputs:
    ...     print([list(s) for s in standardized])
    [[1, 2], [3]]
    [[2, 1], [3]]

    >>> y_true_1d = array([1, 2, 3])
    >>> y_pred_1d = [4, 5, 6]
    >>> standardized_inputs = standardize_input(y_true_1d, y_pred_1d)
    >>> for standardized in standardized_inputs:
    ...     print([list(s) for s in standardized])
    [[1], [2], [3]]
    [[4], [5], [6]]

    Notes
    -----
    The function is particularly useful for preprocessing inputs to metrics 
    that require comparison of sets of items across samples, such as precision
    at K, recall at K, or NDCG. By standardizing the inputs to lists of sets,
    the function facilitates consistent handling of these computations
    regardless of the original format of the input data. This standardization
    is critical when working with real-world data, which can vary widely in
    format and structure.
    """
    standardized = []
    for data in arrays:
        # Transform ndarray based on its dimensions
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                standardized.append([set([item]) for item in data])
            elif data.ndim == 2:
                standardized.append([set(row) for row in data])
            else:
                raise TypeError("Unsupported ndarray shape. Must be 1D or 2D.")
        # Transform lists or tuples
        elif isinstance(data, (list, tuple)):
            if all(isinstance(item, (list, tuple, np.ndarray)) for item in data):
                standardized.append([set(item) for item in data])
            else:
                standardized.append([set([item]) for item in data])
        else:
            raise TypeError(
                "Inputs must be array-like, lists of lists, or lists of tuples.")
    
    # Check consistent length across all transformed inputs
    if any(len(standardized[0]) != len(arr) for arr in standardized[1:]):
        raise ValueError("All inputs must have the same length.")
    
    return standardized

def smart_rotation(ax):
    """
    Automatically adjusts the rotation of x-axis tick labels on a matplotlib
    axis object based on the overlap of labels. This function assesses the
    overlap by comparing the horizontal extents of adjacent tick labels. If
    any overlap is detected, it rotates the labels by 45 degrees to reduce
    or eliminate the overlap. If no overlap is detected, labels remain
    horizontal.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object for which to adjust the tick label rotation.

    Examples
    --------
    # Example of creating a simple time series plot with date overlap handling
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.dates import DateFormatter
    >>> from gofast.tools.baseutils import smart_rotation

    # Generate a date range and some random data
    >>> dates = pd.date_range(start="2020-01-01", periods=100, freq='D')
    >>> values = np.random.rand(100)

    # Create a DataFrame
    >>> df = pd.DataFrame({'Date': dates, 'Value': values})

    # Create a plot
    >>> fig, ax = plt.subplots()
    >>> ax.plot(df['Date'], df['Value'])
    >>> ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    # Apply smart rotation to adjust tick labels dynamically
    >>> smart_rotation(ax)

    # Show the plot
    >>> plt.show()

    Notes
    -----
    This function needs to be used in conjunction with matplotlib plots where
    the axis ('ax') is already set up with tick labels. It is especially useful
    in time series and other plots where the x-axis labels are dates or other
    large strings that may easily overlap. Drawing the canvas (plt.gcf().canvas.draw())
    is necessary to render the labels and calculate their positions, which may
    impact performance for very large plots or in tight loops.
    
    """
    
    # Draw the canvas to get the labels rendered, which is necessary for calculating overlap
    plt.gcf().canvas.draw()

    # Retrieve the x-axis tick labels and their extents
    labels = [label.get_text() for label in ax.get_xticklabels()]
    tick_locs = ax.get_xticks()  # get the locations of the current ticks
    label_extents = [label.get_window_extent() for label in ax.get_xticklabels()]

    # Check for overlap by examining the extents
    overlap = False
    num_labels = len(label_extents)
    for i in range(num_labels - 1):
        if label_extents[i].xmax > label_extents[i + 1].xmin:
            overlap = True
            break

    # Apply rotation if overlap is detected
    rotation = 45 if overlap else 0

    # Set the locator before setting labels
    ax.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax.set_xticklabels(labels, rotation=rotation)
    
def select_features(
    data: DataFrame,
    features: Optional[List[str]] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    coerce: bool = False,
    columns: Optional[List[str]] = None,
    verify_integrity: bool = False,
    parse_features: bool = False,
    **kwd
) -> DataFrame:
    """
    Selects features from a DataFrame according to specified criteria,
    returning a new DataFrame.

    Parameters
    ----------
    data : DataFrame
        The DataFrame from which to select features.
    features : List[str], optional
        Specific feature names to select. An error is raised if any
        feature is not present in `data`.
    include : str or List[str], optional
        The data type(s) to include in the selection. Possible values
        are the same as for the pandas `include` parameter in `select_dtypes`.
    exclude : str or List[str], optional
        The data type(s) to exclude from the selection. Possible values
        are the same as for the pandas `exclude` parameter in `select_dtypes`.
    coerce : bool, default False
        If True, numeric columns are coerced to the appropriate types without
        selection, ignoring `features`, `include`, and `exclude` parameters.
    columns : List[str], optional
        Columns to construct a DataFrame if `data` is passed as a Numpy array.
    verify_integrity : bool, default False
        Verifies the data type integrity and converts data to the correct
        types if necessary.
    parse_features : bool, default False
        Parses string features and converts them to an iterable object.
    **kwd : dict
        Additional keyword arguments for `pandas.DataFrame.astype`.

    Returns
    -------
    DataFrame
        A new DataFrame with the selected features.

    Examples
    --------
    >>> from gofast.tools.baseutils import select_features
    >>> data = {"Color": ['Blue', 'Red', 'Green'],
                "Name": ['Mary', "Daniel", "Augustine"],
                "Price ($)": ['200', "300", "100"]}
    >>> select_features(data, include='number')
    Empty DataFrame
    Columns: []
    Index: [0, 1, 2]

    >>> select_features(data, include='number', verify_integrity=True)
        Price ($)
    0   200.0
    1   300.0
    2   100.0

    >>> select_features(data, features=['Color', 'Price ($)'])
       Color Price ($)
    0  Blue  200
    1  Red   300
    2  Green 100

    See Also
    --------
    pandas.DataFrame.select_dtypes : For more information on how to use
                                      `include` and `exclude`.

    Reference
    ---------
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    
    """
    coerce, verify_integrity, parse_features= ellipsis2false( 
        coerce, verify_integrity, parse_features)
    
    data = build_data_if(data, columns = columns, )
  
    if verify_integrity: 
        data = to_numeric_dtypes(data )
        
    if features is not None: 
        features= list(is_iterable (
            features, exclude_string=True, transform=True, 
            parse_string = parse_features)
            )
        validate_feature(data, features, verbose ='raise')
    # change the dataype 
    data = data.astype (float, errors ='ignore', **kwd) 
    # assert whether the features are in the data columns
    if features is not None: 
        return data [features] 
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return data if coerce else data.select_dtypes (include, exclude) 

def speed_rowwise_process(
    data, /, 
    func, 
    n_jobs=-1
    ):
    """
    Processes a large dataset by applying a complex function to each row. 
    
    Function utilizes parallel processing to optimize for speed.

    Parameters
    ----------
    data : pd.DataFrames
        The large dataset to be processed. Assumes the 
        dataset is a Pandas DataFrame.

    func : function
        A complex function to apply to each row of the dataset. 
        This function should take a row of the DataFrame as 
        input and return a processed row.

    n_jobs : int, optional
        The number of jobs to run in parallel. -1 means using 
        all processors. Default is -1.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    Example
    -------
    >>> def complex_calculation(row):
    >>>     # Example of a complex row-wise calculation
    >>>     return row * 2  # This is a simple placeholder for demonstration.
    >>>
    >>> large_data = pd.DataFrame(np.random.rand(10000, 10))
    >>> processed_data = speed_rowwise_process(large_data, complex_calculation)

    """
    # Function to apply `func` to each row in parallel
    def process_row(row):
        return func(row)

    # Using Joblib's Parallel and delayed to apply the function in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_row)(row) 
                                      for row in data.itertuples(index=False))

    # Converting results back to DataFrame
    processed_data = pd.DataFrame(results, columns=data.columns)
    return processed_data
    
def run_shell_command(command, progress_bar_duration=30, pkg=None):
    """
    Run a shell command with an indeterminate progress bar.

    This function will display a progress bar for a predefined duration while 
    the package installation command runs in a separate thread. The progress 
    bar is purely for visual effect and does not reflect the actual 
    progress of the installation.

    Keep in mind:
    
    This function assumes that you have tqdm installed (pip install tqdm).
    The actual progress of the installation isn't tracked; the progress bar 
    is merely for aesthetics.
    The function assumes the command is a blocking one 
    (like most pip install commands) and waits for it to complete.
    Adjust progress_bar_duration based on how long you expect the installation
    to take. If the installation finishes before the progress bar, the bar
    will stop early. If the installation takes longer, the bar will complete, 
    but the function will continue to wait until the installation is done.
    
    Parameters:
    -----------
    command : list
        The command to run, provided as a list of strings.

    progress_bar_duration : int
        The maximum duration to display the progress bar for, in seconds.
        Defaults to 30 seconds.
    pkg: str, optional 
        The name of package to install for customizing bar description. 

    Returns:
    --------
    None
    
    Example 
    -------
    >>> from gofast.tools.baseutils import run_shell_command 
    >>> run_shell_command(["pip", "install", "gofast"])
    """
    def run_command(command):
        subprocess.run(command, check=True)

    def show_progress_bar(duration):
        with tqdm(total=duration, desc="Installing{}".format( 
                '' if pkg is None else f" {str(pkg)}"), 
                  bar_format="{l_bar}{bar}", ncols=77, ascii=True)  as pbar:
            for i in range(duration):
                time.sleep(1)
                pbar.update(1)

    # Start running the command
    thread = threading.Thread(target=run_command, args=(command,))
    thread.start()

    # Start the progress bar
    show_progress_bar(progress_bar_duration)

    # Wait for the command to finish
    thread.join()


def download_file(url, local_filename , dstpath =None ):
    """download a remote file. 
    
    Parameters 
    -----------
    url: str, 
      Url to where the file is stored. 
    loadl_filename: str,
      Name of the local file 
      
    dstpath: Optional 
      The destination path to save the downloaded file. 
      
    Return 
    --------
    None, local_filename
       None if the `dstpath` is supplied and `local_filename` otherwise. 
       
    Example 
    ---------
    >>> from gofast.tools.baseutils import download_file
    >>> url = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/gofast/datasets/data/h.h5'
    >>> local_filename = 'h.h5'
    >>> download_file(url, local_filename, test_directory)    
    
    """
    import_optional_dependency("requests")
    import requests 
    print("{:-^70}".format(f" Please, Wait while {os.path.basename(local_filename)}"
                          " is downloading. ")) 
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    local_filename = os.path.join( os.getcwd(), local_filename) 
    
    if dstpath: 
         move_file ( local_filename,  dstpath)
         
    print("{:-^70}".format(" ok! "))
    
    return None if dstpath else local_filename


def fancier_downloader(url, local_filename, dstpath =None ):
    """ Download remote file with a bar progression. 
    
    Parameters 
    -----------
    url: str, 
      Url to where the file is stored. 
    loadl_filename: str,
      Name of the local file 
      
    dstpath: Optional 
      The destination path to save the downloaded file. 
      
    Return 
    --------
    None, local_filename
       None if the `dstpath` is supplied and `local_filename` otherwise. 
    Example
    --------
    >>> from gofast.tools.baseutils import fancier_downloader
    >>> url = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/gofast/datasets/data/h.h5'
    >>> local_filename = 'h.h5'
    >>> download_file(url, local_filename)

    """
    import_optional_dependency("requests")
    import requests 
    try : 
        from tqdm import tqdm
    except: 
        # if tqm is not install 
        return download_file (url, local_filename, dstpath  )
        
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Get the total file size from header
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', 
                            unit_scale=True, ncols=77, ascii=True)
        with open(local_filename, 'wb') as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        
    local_filename = os.path.join( os.getcwd(), local_filename) 
    
    if dstpath: 
         move_file ( local_filename,  dstpath)
         
    return local_filename


def move_file(file_path, directory):
    """ Move file to a directory. 
    
    Create a directory if not exists. 
    
    Parameters 
    -----------
    file_path: str, 
       Path to the local file 
    directory: str, 
       Path to locate the directory.
    
    Example 
    ---------
    >>> from gofast.tools.baseutils import move_file
    >>> file_path = 'path/to/your/file.txt'  # Replace with your file's path
    >>> directory = 'path/to/your/directory'  # Replace with your directory's path
    >>> move_file(file_path, directory)
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Move the file to the directory
    shutil.move(file_path, os.path.join(directory, os.path.basename(file_path)))

def check_file_exists(package, resource):
    """
    Check if a file exists in a package's directory with 
    importlib.resources.

    :param package: The package containing the resource.
    :param resource: The resource (file) to check.
    :return: Boolean indicating if the resource exists.
    
    :example: 
        >>> from gofast.tools.baseutils import check_file_exists
        >>> package_name = 'gofast.datasets.data'  # Replace with your package name
        >>> file_name = 'h.h5'    # Replace with your file name

        >>> file_exists = check_file_exists(package_name, file_name)
        >>> print(f"File exists: {file_exists}")
    """

    import importlib.resources as pkg_resources
    return pkg_resources.is_resource(package, resource)

def is_readable (
        f:str, 
        *, 
        as_frame:bool=False, 
        columns:List[str]=None,
        input_name='f', 
        **kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a pandas frame.  
    
    Parameters 
    -----------
    f: Path-like object -Should be a readable files or url  
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
        
    kws: dict, 
        Pandas readableformats additional keywords arguments. 
    Returns
    ---------
    f: pandas dataframe 
         A dataframe with head contents... 
    
    """
    def _check_readable_file (f): 
        """ Return file name from path objects """
        msg =(f"Expects a Path-like object or URL. Please, check your"
              f" file: {os.path.basename(f)!r}")
        if not os.path.isfile (f): # force pandas read html etc 
            if not ('http://'  in f or 'https://' in f ):  
                raise TypeError (msg)
        elif not isinstance (f,  (str , pathlib.PurePath)): 
             raise TypeError (msg)
        if isinstance(f, str): f =f.strip() # for consistency 
        return f 
    
    if hasattr (f, '__array__' ) : 
        f = array_to_frame(
            f, 
            to_frame= True , 
            columns =columns, 
            input_name=input_name , 
            raise_exception= True, 
            force= True, 
            )
        return f 

    cpObj= Config().parsers 
    
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smart_format(cpObj.keys(), 'or')} "
                        f" files not {ex!r}.")
    try : 
        f = cpObj[ex](f, **kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except: 
        raise FileHandlingError (
            f" Can not parse the file : {os.path.basename (f)!r}")

    return f 

def lowertify(
    *values,
    strip: bool = True, 
    return_origin: bool = False, 
    unpack: bool = False
    ) -> Union[Tuple[str, ...], Tuple[Tuple[str, Any], ...], Tuple[Any, ...]]:
    """
    Convert all input values to lowercase strings, optionally stripping 
    whitespace, and optionally return the original values alongside the 
    lowercased versions.
    
    Can also unpack the tuples of lowercased and original values into a single
    flat tuple.

    Parameters
    ----------
    *values : Any
        Arbitrary number of values to be converted to lowercase. Non-string 
        values will be converted to strings before processing.
    strip : bool, optional
        If True (default), leading and trailing whitespace will be removed 
        from the strings.
    return_origin : bool, optional
        If True, each lowercased string is returned as a tuple with its 
        original value; otherwise, only the lowercased strings are returned.
    unpack : bool, optional
        If True, and `return_origin` is also True, the function returns a 
        single flat tuple containing all lowercased and original values 
        alternatively. This parameter is ignored if `return_origin` is False.

    Returns
    -------
    Union[Tuple[str, ...], Tuple[Tuple[str, Any], ...], Tuple[Any, ...]]
        Depending on `return_origin` and `unpack` flags, returns either:
        - A tuple of lowercased (and optionally stripped) strings.
        - A tuple of tuples, each containing the lowercased string and its 
          original value.
        - A single flat tuple containing all lowercased and original values 
          alternatively (if `unpack` is True).

    Examples
    --------
    >>> from gofast.tools.baseutils import lowertify
    >>> lowertify('KIND')
    ('kind',)
    
    >>> lowertify("KIND", return_origin=True)
    (('kind', 'KIND'),)
    
    >>> lowertify("args1", 120, 'ArG3')
    ('args1', '120', 'arg3')
    
    >>> lowertify("args1", 120, 'ArG3', return_origin=True)
    (('args1', 'args1'), ('120', 120), ('arg3', 'ArG3'))
    
    >>> lowertify("KIND", "task ", return_origin=True, unpack=True)
    ('kind', 'KIND', 'task', 'task ')
    """
    processed_values = [(str(val).strip().lower() if strip 
                         else str(val).lower(), val) for val in values]
    if return_origin:
        if unpack:
            # Flatten the list of tuples into a single tuple for unpacking
            return tuple(item for pair in processed_values for item in pair)
        else:
            return tuple(processed_values)
    else:
        return tuple(lowered for lowered, _ in processed_values)


def save_or_load(
    fname:str, /,
    arr: NDArray=None,  
    task: str='save', 
    format: str='.txt', 
    compressed: bool=...,  
    comments: str="#",
    delimiter: str=None, 
    **kws 
): 
    """Save or load Numpy array. 
    
    Parameters 
    -----------
    fname: file, str, or pathlib.Path
       File or filename to which the data is saved. 
       - >.npy , .npz: If file is a file-object, then the filename is unchanged. 
       If file is a string or Path, a .npy extension will be appended to the 
       filename if it does not already have one. 
       - >.txt: If the filename ends in .gz, the file is automatically saved in 
       compressed gzip format. loadtxt understands gzipped files transparently.
       
    arr: 1D or 2D array_like
      Data to be saved to a text, npy or npz file.
      
    task: str {"load", "save"}
      Action to perform. "Save" for storing file into the format 
      ".txt", "npy", ".npz". "load" for loading the data from storing files. 
      
    format: str {".txt", ".npy", ".npz"}
       The kind of format to save and load.  Note that when loading the 
       compressed data saved into `npz` format, it does not return 
       systematically the array rather than `np.lib.npyio.NpzFile` files. 
       Use either `files` attributes to get the list of registered files 
       or `f` attribute dot the data name to get the loaded data set. 

    compressed: bool, default=False 
       Compressed the file especially when file format is set to `.npz`. 

    comments: str or sequence of str or None, default='#'
       The characters or list of characters used to indicate the start 
       of a comment. None implies no comments. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. This is useful when `fname`
       is in `txt` format. 
      
     delimiter: str,  optional
        The character used to separate the values. For backwards compatibility, 
        byte strings will be decoded as 'latin1'. The default is whitespace.
        
    kws: np.save ,np.savetext,  np.load , np.loadtxt 
       Additional keywords arguments for saving and loading data. 
       
    Return 
    ------
    None| data: ArrayLike 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import save_or_load 
    >>> data = np.random.randn (2, 7)
    >>> # save to txt 
    >>> save_or_load ( "test.txt" , data)
    >>> save_or_load ( "test",  data, format='.npy')
    >>> save_or_load ( "test",  data, format='.npz')
    >>> save_or_load ( "test_compressed",  data, format='.npz', compressed=True )
    >>> # load files 
    >>> save_or_load ( "test.txt", task ='load')
    Out[36]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test.npy", task ='load')
    Out[39]: array([-2.34162127,  0.55093818])
    >>> save_or_load ( "test.npz", task ='load')
    <numpy.lib.npyio.NpzFile at 0x1b0821870a0>
    >>> npzo = save_or_load ( "test.npz", task ='load')
    >>> npzo.files
    Out[44]: ['arr_0']
    >>> npzo.f.arr_0
    Out[45]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test_compressed.npz", task ='load')
    ...
    """
    r_formats = {"npy", "txt", "npz"}
   
    (kind, kind0), ( task, task0 ) = lowertify(
        format, task, return_origin =True )
    
    assert  kind.replace ('.', '') in r_formats, (
        f"File format expects {smart_format(r_formats, 'or')}. Got {kind0!r}")
    kind = '.' + kind.replace ('.', '')
    assert task in {'save', 'load'}, ( 
        "Wrong task {task0!r}. Valid tasks are 'save' or 'load'") 
    
    save= {'.txt': np.savetxt, '.npy':np.save,  
           ".npz": np.savez_compressed if ellipsis2false(
               compressed)[0] else np.savez 
           }
    if task =='save': 
        arr = np.array (is_iterable( arr, exclude_string= True, 
                                    transform =True ))
        save.get(kind) (fname, arr, **kws )
        
    elif task =='load': 
         ext = os.path.splitext(fname)[1].lower() 
         if ext not in (".txt", '.npy', '.npz', '.gz'): 
             raise ValueError ("Unrecognized file format {ext!r}."
                               " Expect '.txt', '.npy', '.gz' or '.npz'")
         if ext in ('.txt', '.gz'): 
            arr = np.loadtxt ( fname , comments= comments, 
                              delimiter= delimiter,   **kws ) 
         else : 
            arr = np.load(fname,**kws )
         
    return arr if task=='load' else None 

def array2hdf5 (
    filename: str, /, 
    arr: NDArray=None , 
    dataname: str='data',  
    task: str='store', 
    as_frame: bool =..., 
    columns: List[str]=None, 
)-> NDArray | DataFrame: 
    """ Load or write array to hdf5
    
    Parameters 
    -----------
    arr: Arraylike ( m_samples, n_features) 
      Data to load or write 
    filename: str, 
      Hdf5 disk file name whether to write or to load 
    task: str, {"store", "load", "save", default='store'}
       Action to perform. user can use ['write'|'store'] interchnageably. Both 
       does the same task. 
    as_frame: bool, default=False 
       Concert loaded array to data frame. `Columns` can be supplied 
       to construct the datafame. 
    columns: List, Optional 
       Columns used to construct the dataframe. When its given, it must be 
       consistent with the shape of the `arr` along axis 1 
       
    Returns 
    ---------
    None| data: ArrayLike or pd.DataFrame 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import array2hdf5
    >>> data = np.random.randn (100, 27 ) 
    >>> array2hdf5 ('test.h5', data   )
    >>> load_data = array2hdf5 ( 'test.h5', data, task ='load')
    >>> load_data.shape 
    Out[177]: (100, 27)
    """
    import_optional_dependency("h5py")
    import h5py 
    
    arr = is_iterable( arr, exclude_string =True, transform =True )
    act = copy.deepcopy(task)
    task = str(task).lower().strip() 
    
    if task in ("write", "store", "save"): 
        task ='store'
    assert task in {"store", "load"}, ("Expects ['store'|'load'] as task."
                                         f" Got {act!r}")
    # for consistency 
    arr = np.array ( arr )
    h5fname = str(filename).replace ('.h5', '')
    if task =='store': 
        if arr is None: 
            raise TypeError ("Array cannot be None when the task"
                             " consists to write a file.")
        with h5py.File(h5fname + '.h5', 'w') as hf:
            hf.create_dataset(dataname,  data=arr)
            
    elif task=='load': 
        with h5py.File(h5fname +".h5", 'r') as hf:
            data = hf[dataname][:]
            
        if  ellipsis2false( as_frame )[0]: 
            data = pd.DataFrame ( data , columns = columns )
            
    return data if task=='load' else None 

def remove_target_from_array(arr,/,  target_indices):
    """
    Remove specified columns from a 2D array based on target indices.

    This function extracts columns at specified indices from a 2D array, 
    returning the modified array without these columns and a separate array 
    containing the extracted columns. It raises an error if any of the indices
    are out of bounds.

    Parameters
    ----------
    arr : ndarray
        A 2D numpy array from which columns are to be removed.
    target_indices : list or ndarray
        Indices of the columns in `arr` that need to be extracted and removed.

    Returns
    -------
    modified_arr : ndarray
        The array obtained after removing the specified columns.
    target_arr : ndarray
        An array consisting of the columns extracted from `arr`.

    Raises
    ------
    ValueError
        If any of the target indices are out of the range of the array dimensions.

    Examples
    --------
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> target_indices = [1, 2]
    >>> modified_arr, target_arr = remove_target_from_array(arr, target_indices)
    >>> modified_arr
    array([[1],
           [4],
           [7]])
    >>> target_arr
    array([[2, 3],
           [5, 6],
           [7, 8]])
    """
    if any(idx >= arr.shape[1] for idx in target_indices):
        raise ValueError("One or more indices are out of the array's bounds.")

    target_arr = arr[:, target_indices]
    modified_arr = np.delete(arr, target_indices, axis=1)
    return modified_arr, target_arr

def extract_target(
    data: Union[ArrayLike, DataFrame],/, 
    target_names: Union[str, int, List[Union[str, int]]],
    drop: bool = True,
    columns: Optional[List[str]] = None,
) -> Tuple[Union[ArrayLike, Series, DataFrame], Union[ArrayLike, DataFrame]]:
    """
    Extracts specified target column(s) from a multidimensional numpy array
    or pandas DataFrame. 
    
    with options to rename columns in a DataFrame and control over whether the 
    extracted columns are dropped from the original data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        The input data from which target columns are to be extracted. Can be a 
        NumPy array or a pandas DataFrame.
    target_names : Union[str, int, List[Union[str, int]]]
        The name(s) or integer index/indices of the column(s) to extract. 
        If `data` is a DataFrame, this can be a mix of column names and indices. 
        If `data` is a NumPy array, only integer indices are allowed.
    drop : bool, default True
        If True, the extracted columns are removed from the original `data`. 
        If False, the original `data` remains unchanged.
    columns : Optional[List[str]], default None
        If provided and `data` is a DataFrame, specifies new names for the 
        columns in `data`. The length of `columns` must match the number of 
        columns in `data`. This parameter is ignored if `data` is a NumPy array.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]
        A tuple containing two elements:
        - The extracted column(s) as a NumPy array or pandas Series/DataFrame.
        - The original data with the extracted columns optionally removed, as a
          NumPy array or pandas DataFrame.

    Raises
    ------
    ValueError
        If `columns` is provided and its length does not match the number of 
        columns in `data`.
        If any of the specified `target_names` do not exist in `data`.
        If `target_names` includes a mix of strings and integers for a NumPy 
        array input.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import extract_target
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> target, remaining = extract_target(df, 'B', drop=True)
    >>> print(target)
    0    4
    1    5
    2    6
    Name: B, dtype: int64
    >>> print(remaining)
       A  C
    0  1  7
    1  2  8
    2  3  9
    >>> arr = np.random.rand(5, 3)
    >>> target, modified_arr = extract_target(arr, 2, )
    >>> print(target)
    >>> print(modified_arr)
    """
    if isinstance (data, pd.Series): 
        data = data.to_frame() 
    if _is_arraylike_1d(data): 
        # convert to 2d array 
        data = data.reshape (-1, 1)
    
    is_frame = isinstance(data, pd.DataFrame)
    
    if is_frame and columns is not None:
        if len(columns) != data.shape[1]:
            raise ValueError("`columns` must match the number of columns in"
                             f" `data`. Expected {data.shape[1]}, got {len(columns)}.")
        data.columns = columns

    if isinstance(target_names, (int, str)):
        target_names = [target_names]

    if all(isinstance(name, int) for name in target_names):
        if max(target_names, default=-1) >= data.shape[1]:
            raise ValueError("All integer indices must be within the"
                             " column range of the data.")
    elif any(isinstance(name, int) for name in target_names) and is_frame:
        target_names = [data.columns[name] if isinstance(name, int) 
                        else name for name in target_names]

    if is_frame:
        missing_cols = [name for name in target_names 
                        if name not in data.columns]
        if missing_cols:
            raise ValueError(f"Column names {missing_cols} do not match "
                             "any column in the DataFrame.")
        target = data.loc[:, target_names]
        if drop:
            data = data.drop(columns=target_names)
    else:
        if any(isinstance(name, str) for name in target_names):
            raise ValueError("String names are not allowed for target names"
                             " when data is a NumPy array.")
        target = data[:, target_names]
        if drop:
            data = np.delete(data, target_names, axis=1)
            
    if  isinstance (target, np.ndarray): # squeeze the array 
        target = np.squeeze (target)
        
    return target, data

def _extract_target(
        X, target: Union[ArrayLike, int, str, List[Union[int, str]]]):
    """
    Extracts and validates the target variable(s) from the dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The dataset from which to extract the target variable(s).
    target : ArrayLike, int, str, or list of int/str
        The target variable(s) to be used. If an array-like or DataFrame, 
        it's directly used as `y`. If an int or str (or list of them), it 
        indicates the column(s) in `X` to be used as `y`.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        The dataset without the target column(s).
    y : pd.Series, np.ndarray, pd.DataFrame
        The target variable(s).
    target_names : list of str
        The names of the target variable(s) for labeling purposes.
    """
    target_names = []

    if isinstance(target, (list, pd.DataFrame)) or (
            isinstance(target, pd.Series) and not isinstance(X, np.ndarray)):
        if isinstance(target, list):  # List of column names or indexes
            if all(isinstance(t, str) for t in target):
                y = X[target]
                target_names = target
            elif all(isinstance(t, int) for t in target):
                y = X.iloc[:, target]
                target_names = [X.columns[i] for i in target]
            X = X.drop(columns=target_names)
        elif isinstance(target, pd.DataFrame):
            y = target
            target_names = target.columns.tolist()
            # Assuming target DataFrame is not part of X
        elif isinstance(target, pd.Series):
            y = target
            target_names = [target.name] if target.name else ["target"]
            if target.name and target.name in X.columns:
                X = X.drop(columns=target.name)
                
    elif isinstance(target, (int, str)):
        if isinstance(target, str):
            y = X.pop(target)
            target_names = [target]
        elif isinstance(target, int):
            y = X.iloc[:, target]
            target_names = [X.columns[target]]
            X = X.drop(columns=X.columns[target])
    elif isinstance(target, np.ndarray) or (
            isinstance(target, pd.Series) and isinstance(X, np.ndarray)):
        y = np.array(target)
        target_names = ["target"]
    else:
        raise ValueError("Unsupported target type or target does not match X dimensions.")
    
    check_consistent_length(X, y)
    
    return X, y, target_names

def categorize_target(
    arr : Union [ArrayLike , Series] , /, 
    func: _F = None,  
    labels: Union [int, List[int]] = None, 
    rename_labels: Optional[str] = None, 
    coerce:bool=False,
    order:str='strict',
    ): 
    """ Categorize array to hold the given identifier labels. 
    
    Classifier numerical values according to the given label values. Labels 
    are a list of integers where each integer is a group of unique identifier  
    of a sample in the dataset. 
    
    Parameters 
    -----------
    arr: array-like |pandas.Series 
        array or series containing numerical values. If a non-numerical values 
        is given , an errors will raises. 
    func: Callable, 
        Function to categorize the target y.  
    labels: int, list of int, 
        if an integer value is given, it should be considered as the number 
        of category to split 'y'. For instance ``label=3`` applied on 
        the first ten number, the labels values should be ``[0, 1, 2]``. 
        If labels are given as a list, items must be self-contain in the 
        target 'y'.
    rename_labels: list of str; 
        list of string or values to replace the label integer identifier. 
    coerce: bool, default =False, 
        force the new label names passed to `rename_labels` to appear in the 
        target including or not some integer identifier class label. If 
        `coerce` is ``True``, the target array holds the dtype of new_array. 

    Return
    --------
    arr: Arraylike |pandas.Series
        The category array with unique identifer labels 
        
    Examples 
    --------

    >>> from gofast.tools.baseutils import categorize_target 
    >>> def binfunc(v): 
            if v < 3 : return 0 
            else : return 1 
    >>> arr = np.arange (10 )
    >>> arr 
    ... array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> target = categorize_target(arr, func =binfunc)
    ... array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    >>> categorize_target(arr, labels =3 )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    >>> categorize_target(arr, labels =3 , order =None )
    ... array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> categorize_target(arr[::-1], labels =3 , order =None )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    >>> categorize_target(arr, labels =[0 , 2,  4]  )
    ... array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])

    """
    arr = _assert_all_types(arr, np.ndarray, pd.Series) 
    is_arr =False 
    if isinstance (arr, np.ndarray ) :
        arr = pd.Series (arr  , name = 'none') 
        is_arr =True 
        
    if func is not None: 
        if not  inspect.isfunction (func): 
            raise TypeError (
                f'Expect a function but got {type(func).__name__!r}')
            
        arr= arr.apply (func )
        
        return  arr.values  if is_arr else arr   
    
    name = arr.name 
    arr = arr.values 

    if labels is not None: 
        arr = _cattarget (arr , labels, order =order)
        if rename_labels is not None: 
            arr = rename_labels_in( arr , rename_labels , coerce =coerce ) 

    return arr  if is_arr else pd.Series (arr, name =name  )

def rename_labels_in (
        arr, new_names, coerce = False): 
    """ Rename label by a new names 
    
    :param arr: arr: array-like |pandas.Series 
         array or series containing numerical values. If a non-numerical values 
         is given , an errors will raises. 
    :param new_names: list of str; 
        list of string or values to replace the label integer identifier. 
    :param coerce: bool, default =False, 
        force the 'new_names' to appear in the target including or not some 
        integer identifier class label. `coerce` is ``True``, the target array 
        hold the dtype of new_array; coercing the label names will not yield 
        error. Consequently can introduce an unexpected results.
    :return: array-like, 
        An array-like with full new label names. 
    """
    
    if not is_iterable(new_names): 
        new_names= [new_names]
    true_labels = np.unique (arr) 
    
    if labels_validator(arr, new_names, return_bool= True): 
        return arr 

    if len(true_labels) != len(new_names):
        if not coerce: 
            raise ValueError(
                "Can't rename labels; the new names and unique label" 
                " identifiers size must be consistent; expect {}, got " 
                "{} label(s).".format(len(true_labels), len(new_names))
                             )
        if len(true_labels) < len(new_names) : 
            new_names = new_names [: len(new_names)]
        else: 
            new_names = list(new_names)  + list(
                true_labels)[len(new_names):]
            warnings.warn("Number of the given labels '{}' and values '{}'"
                          " are not consistent. Be aware that this could "
                          "yield an expected results.".format(
                              len(new_names), len(true_labels)))
            
    new_names = np.array(new_names)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # hold the type of arr to operate the 
    # element wise comparaison if not a 
    # ValueError:' invalid literal for int() with base 10' 
    # will appear. 
    if not np.issubdtype(np.array(new_names).dtype, np.number): 
        arr= arr.astype (np.array(new_names).dtype)
        true_labels = true_labels.astype (np.array(new_names).dtype)

    for el , nel in zip (true_labels, new_names ): 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # element comparison throws a future warning here 
        # because of a disagreement between Numpy and native python 
        # Numpy version ='1.22.4' while python version = 3.9.12
        # this code is brittle and requires these versions above. 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # suppress element wise comparison warning locally 
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            arr [arr == el ] = nel 
            
    return arr 

    
def _cattarget (ar , labels , order=None): 
    """ A shadow function of :func:`gofast.tools.baseutils.cattarget`. 
    
    :param ar: array-like of numerical values 
    :param labels: int or list of int, 
        the number of category to split 'ar'into. 
    :param order: str, optional, 
        the order of label to be categorized. If None or any other values, 
        the categorization of labels considers only the length of array. 
        For instance a reverse array and non-reverse array yield the same 
        categorization samples. When order is set to ``strict``, the 
        categorization  strictly considers the value of each element. 
        
    :return: array-like of int , array of categorized values.  
    """
    # assert labels
    if is_iterable (labels):
        labels =[int (_assert_all_types(lab, int, float)) 
                 for lab in labels ]
        labels = np.array (labels , dtype = np.int32 ) 
        cc = labels 
        # assert whether element is on the array 
        s = set (ar).intersection(labels) 
        if len(s) != len(labels): 
            mv = set(labels).difference (s) 
            
            fmt = [f"{'s' if len(mv) >1 else''} ", mv,
                   f"{'is' if len(mv) <=1 else'are'}"]
            warnings.warn("Label values must be array self-contain item. "
                           "Label{0} {1} {2} missing in the array.".format(
                               *fmt)
                          )
            raise ValueError (
                "label value{0} {1} {2} missing in the array.".format(*fmt))
    else : 
        labels = int (_assert_all_types(labels , int, float))
        labels = np.linspace ( min(ar), max (ar), labels + 1 ) #+ .00000001 
        #array([ 0.,  6., 12., 18.])
        # split arr and get the range of with max bound 
        cc = np.arange (len(labels)) #[0, 1, 3]
        # we expect three classes [ 0, 1, 3 ] while maximum 
        # value is 18 . we want the value value to be >= 12 which 
        # include 18 , so remove the 18 in the list 
        labels = labels [:-1] # remove the last items a
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 classes 
        #  array([ 0.        ,  3.33333333,  6.66666667, 10. ]) + 
    # to avoid the index bound error 
    # append nan value to lengthen arr 
    r = np.append (labels , np.nan ) 
    new_arr = np.zeros_like(ar) 
    # print(labels)
    ar = ar.astype (np.float32)

    if order =='strict': 
        for i in range (len(r)):
            if i == len(r) -2 : 
                ix = np.argwhere ( (ar >= r[i]) & (ar != np.inf ))
                new_arr[ix ]= cc[i]
                break 
            
            if i ==0 : 
                ix = np.argwhere (ar < r[i +1])
                new_arr [ix] == cc[i] 
                ar [ix ] = np.inf # replace by a big number than it was 
                # rather than delete it 
            else :
                ix = np.argwhere( (r[i] <= ar) & (ar < r[i +1]) )
                new_arr [ix ]= cc[i] 
                ar [ix ] = np.inf 
    else: 
        l= list() 
        for i in range (len(r)): 
            if i == len(r) -2 : 
                l.append (np.repeat ( cc[i], len(ar))) 
                
                break
            ix = np.argwhere ( (ar < r [ i + 1 ] ))
            l.append (np.repeat (cc[i], len (ar[ix ])))  
            # remove the value ready for i label 
            # categorization 
            ar = np.delete (ar, ix  )
            
        new_arr= np.hstack (l).astype (np.int32)  
        
    return new_arr.astype (np.int32)    
   

def labels_validator(
    target: ArrayLike, 
    labels: Union[int, str, List[Union[int, str]]], 
    return_bool: bool = False
    ) -> Union[bool, List[Union[int, str]]]:
    """
    Validates if specified labels are present in the target array and 
    optionally returns a boolean indicating the presence of all labels or 
    the list of labels themselves.
    
    Parameters
    ----------
    target : np.ndarray
        The target array expected to contain the labels.
    labels : int, str, or list of int or str
        The label(s) supposed to be in the target array.
    return_bool : bool, default=False
        If True, returns a boolean indicating whether all specified 
        labels are present. If False, returns the list of labels.

    Returns
    -------
    bool or List[Union[int, str]]
        If `return_bool` is True, returns True if all labels are present, 
        False otherwise.
        If `return_bool` is False, returns the list of labels if all are present.
    
    Raises
    ------
    ValueError
        If any of the specified labels are missing in the target array and 
        `return_bool` is False.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import labels_validator
    >>> target = np.array([1, 2, 3, 4, 5])
    >>> labels_validator(target, [1, 2, 3])
    [1, 2, 3]
    >>> labels_validator(target, [0, 1], return_bool=True)
    False
    >>> labels_validator(target, 1)
    [1]
    >>> labels_validator(target, [6], return_bool=True)
    False
    """
    if isinstance(labels, (int, str)):
        labels = [labels]
    
    labels_present = np.unique([label for label in labels if label in target])
    missing_labels = [label for label in labels if label not in labels_present]

    if missing_labels:
        if return_bool:
            return False
        raise ValueError(f"Label{'s' if len(missing_labels) > 1 else ''}"
                        f" {', '.join(map(str, missing_labels))}"
                        f" {'are' if len(missing_labels) > 1 else 'is'}"
                        " missing in the target."
                    )

    return True if return_bool else labels

def generate_placeholders(
        iterable_obj: Iterable[_T]) -> List[str]:
    """
    Generates a list of string placeholders for each item in the input
    iterable. This can be useful for creating formatted string
    representations where each item's index is used within braces.

    :param iterable_obj: An iterable object (e.g., list, set, or any
        iterable collection) whose length determines the number of
        placeholders generated.
    :return: A list of strings, each representing a placeholder in
        the format "{n}", where n is the index of the placeholder.
        
    :Example:
        >>> from gofast.tools.baseutils import generate_placeholders
        >>> generate_placeholders_for_iterable({'ohmS', 'lwi', 'power', 'id', 
        ...                                     'sfi', 'magnitude'})
        ['{0}', '{1}', '{2}', '{3}', '{4}', '{5}']
    """
    return [f"{{{index}}}" for index in range(len(iterable_obj))]


def compute_set_operation( 
    iterable1: Iterable[Any],
    iterable2: Iterable[Any],
    operation: str = "intersection"
) -> Set[Any]:
    """
    Computes the intersection or difference between two iterable objects,
    returning the result as a set. This function is flexible and works
    with any iterable types, including lists, sets, and dictionaries.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object from which to compute the operation.
    iterable2 : Iterable[Any]
        The second iterable object.
    operation : str, optional
        The operation to perform, either 'intersection' or 'difference'.
        Defaults to 'intersection'.

    Returns
    -------
    Set[Any]
        A set of either common elements (intersection) or unique elements
        (difference) from the two iterables.

    Examples
    --------
    Intersection example:
    >>> compute_set_operation(
    ...     ['a', 'b', 'c'], 
    ...     {'b', 'c', 'd'}
    ... )
    {'b', 'c'}

    Difference example:
    >>> compute_set_operation(
    ...     ['a', 'b', 'c'], 
    ...     {'b', 'c', 'd'},
    ...     operation='difference'
    ... )
    {'a', 'd'}

    Notes
    -----
    The function supports only 'intersection' and 'difference' operations.
    It ensures the result is always returned as a set, regardless of the
    input iterable types.
    """
    
    set1 = set(iterable1)
    set2 = set(iterable2)

    if operation == "intersection":
        return set1 & set2  # Using & for intersection
    elif operation == "difference":
        # Returning symmetric difference
        return set1 ^ set2  # Using ^ for symmetric difference
    else:
        raise ValueError("Invalid operation specified. Choose either"
                         " 'intersection' or 'difference'.")

def find_intersection(
    iterable1: Iterable[Any],
    iterable2: Iterable[Any]
) -> Set[Any]:
    """
    Computes the intersection of two iterable objects, returning a set
    of elements common to both. This function is designed to work with
    various iterable types, including lists, sets, and dictionaries.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object.
    iterable2 : Iterable[Any]
        The second iterable object.

    Returns
    -------
    Set[Any]
        A set of elements common to both `iterable1` and `iterable2`.

    Example
    -------
    >>> from gofast.tools.baseutils import find_intersection_between_generics
    >>> compute_intersection(
    ...     ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
    ...     {'ohmS', 'lwi', 'power'}
    ... )
    {'ohmS', 'lwi', 'power'}

    Notes
    -----
    The result is always a set, regardless of the input types, ensuring
    that each element is unique and present in both iterables.
    """

    # Utilize set intersection operation (&) for clarity and conciseness
    return set(iterable1) & set(iterable2)

def find_unique_elements(
    iterable1: Iterable[Any],
    iterable2: Iterable[Any]
) -> Optional[Set[Any]]:
    """
    Computes the difference between two iterable objects, returning a set
    containing elements unique to the iterable with more unique elements.
    If both iterables contain an equal number of unique elements, the function
    returns None.

    This function is designed to work with various iterable types, including
    lists, sets, and dictionaries. The focus is on the count of unique elements
    rather than the total length, which allows for more consistent results
    across different types of iterables.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object.
    iterable2 : Iterable[Any]
        The second iterable object.

    Returns
    -------
    Optional[Set[Any]]
        A set of elements unique to the iterable with more unique elements,
        or None if both have an equal number of unique elements.

    Example
    -------
    >>> find_unique_elements(
    ...     ['a', 'b', 'c', 'c'],
    ...     {'a', 'b'}
    ... )
    {'c'}

    Notes
    -----
    The comparison is based on the number of unique elements, not the
    iterable size. This approach ensures a more meaningful comparison
    when the iterables are of different types or when duplicates are present.
    """

    set1 = set(iterable1)
    set2 = set(iterable2)

    # Adjust the logic to focus on the uniqueness rather than size
    if len(set1) == len(set2):
        return None
    elif len(set1) > len(set2):
        return set1 - set2
    else:
        return set2 - set1

def validate_feature_existence(supervised_features: Iterable[_T], 
                               features: Iterable[_T]) -> None:
    """
    Validates the existence of supervised features within a list of all features.
    This is typically used to ensure that certain expected features (columns) are
    present in a pandas DataFrame.

    Parameters
    ----------
    supervised_features : Iterable[_T]
        An iterable of features presumed to be controlled or supervised.
        
    features : Iterable[_T]
        An iterable of all features, such as pd.DataFrame.columns.

    Raises
    ------
    ValueError
        If `supervised_features` are not found within `features`.
    """
    # Ensure input is in list format if strings are passed
    if isinstance(supervised_features, str):
        supervised_features = [supervised_features]
    if isinstance(features, str):
        features = [features]
    
    # Check for feature existence
    if not cfexist(features_to=supervised_features, features=list(features)):
        raise ValueError(f"Features {supervised_features} not found in {list(features)}")

def cfexist(features_to: List[Any], features: List[Any]) -> bool:
    """
    Checks if all elements of one list (features_to) exist within another list (features).

    Parameters
    ----------
    features_to : List[Any]
        List or array to be checked for existence within `features`.
        
    features : List[Any]
        List of whole features, e.g., as in pd.DataFrame.columns.

    Returns
    -------
    bool
        True if all elements in `features_to` exist in `features`, False otherwise.
    """
    # Normalize input to lists, handle string inputs
    if isinstance(features_to, str):
        features_to = [features_to]
    if isinstance(features, str):
        features = [features]

    # Check for existence
    return set(features_to).issubset(features)

def control_existing_estimator(
    estimator_name: str, 
    predefined_estimators=None, 
    raise_error: bool = False
) -> Union[Tuple[str, str], None]:
    """
    Validates and retrieves the corresponding prefix for a given estimator name.

    This function checks if the provided estimator name exists in a predefined
    list of estimators or in scikit-learn. If found, it returns the corresponding
    prefix and full name. Otherwise, it either raises an error or returns None,
    based on the 'raise_error' flag.

    Parameters
    ----------
    estimator_name : str
        The name of the estimator to check.
    predefined_estimators : dict, default _predefined_estimators
        A dictionary of predefined estimators.
    raise_error : bool, default False
        If True, raises an error when the estimator is not found. Otherwise, 
        emits a warning.

    Returns
    -------
    Tuple[str, str] or None
        A tuple containing the prefix and full name of the estimator, or 
        None if not found.

    Example
    -------
    >>> from gofast.tools.baseutils import control_existing_estimator
    >>> test_est = control_existing_estimator('svm')
    >>> print(test_est)
    ('svc', 'SupportVectorClassifier')
    """
    
    from ..exceptions import EstimatorError 
    # Define a dictionary of predefined estimators
    _predefined_estimators ={
            'dtc': ['DecisionTreeClassifier', 'dtc', 'dec', 'dt'],
            'svc': ['SupportVectorClassifier', 'svc', 'sup', 'svm'],
            'sdg': ['SGDClassifier','sdg', 'sd', 'sdg'],
            'knn': ['KNeighborsClassifier','knn', 'kne', 'knr'],
            'rdf': ['RandomForestClassifier', 'rdf', 'rf', 'rfc',],
            'ada': ['AdaBoostClassifier','ada', 'adc', 'adboost'],
            'vtc': ['VotingClassifier','vtc', 'vot', 'voting'],
            'bag': ['BaggingClassifier', 'bag', 'bag', 'bagg'],
            'stc': ['StackingClassifier','stc', 'sta', 'stack'],
            'xgb': ['ExtremeGradientBoosting', 'xgboost', 'gboost', 'gbdm', 'xgb'], 
          'logit': ['LogisticRegression', 'logit', 'lr', 'logreg'], 
          'extree': ['ExtraTreesClassifier', 'extree', 'xtree', 'xtr']
            }
    predefined_estimators = predefined_estimators or _predefined_estimators
    
    estimator_name= estimator_name.lower().strip() if isinstance (
        estimator_name, str) else get_estimator_name(estimator_name)
    
    # Check if the estimator is in the predefined list
    for prefix, names in predefined_estimators.items():
        lower_names = [name.lower() for name in names]
        
        if estimator_name in lower_names:
            return prefix, names[0]

    # If not found in predefined list, check if it's a valid scikit-learn estimator
    if estimator_name in _get_sklearn_estimator_names():
        return estimator_name, estimator_name

    # If XGBoost is installed, check if it's an XGBoost estimator
    if 'xgb' in predefined_estimators and estimator_name.startswith('xgb'):
        return 'xgb', estimator_name

    # If raise_error is True, raise an error; otherwise, emit a warning
    if raise_error:
        valid_names = [name for names in predefined_estimators.values() for name in names]
        raise EstimatorError(f'Unsupported estimator {estimator_name!r}. '
                             f'Expected one of {valid_names}.')
    else:
        available_estimators = _get_available_estimators(predefined_estimators)
        warning_msg = (f"Estimator {estimator_name!r} not found. "
                       f"Expected one of: {available_estimators}.")
        warnings.warn(warning_msg)

    return None

def _get_sklearn_estimator_names():
    
    # Retrieve all scikit-learn estimator names using all_estimators
    sklearn_estimators = [name for name, _ in all_estimators(type_filter='classifier')]
    sklearn_estimators += [name for name, _ in all_estimators(type_filter='regressor')]
    return sklearn_estimators

def _get_available_estimators(predefined_estimators):
    # Combine scikit-learn and predefined estimators
    sklearn_estimators = _get_sklearn_estimator_names()
    xgboost_estimators = ['xgb' + name for name in predefined_estimators['xgb']]
    
    available_estimators = sklearn_estimators + xgboost_estimators
    return available_estimators

def get_target(df, tname, inplace=True):
    """
    Extracts one or more target columns from a DataFrame and optionally
    modifies the original DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to extract the target(s).
    tname : str or list of str
        The name(s) of the target column(s) to extract. These must be present
        in the DataFrame.
    inplace : bool, optional
        If True, the DataFrame is modified in place by removing the target
        column(s). Defaults to True.

    Returns
    -------
    tuple
        A tuple containing:
        - pd.Series or pd.DataFrame: The extracted target column(s).
        - pd.DataFrame: The modified or unmodified DataFrame depending on the
          `inplace` parameter.

    Raises
    ------
    ValueError
        If any of the specified target names are not in the DataFrame columns.
    TypeError
        If `df` is not a pandas DataFrame.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gofast.baseutils import get_target 
    >>> data = load_iris(as_frame=True).frame
    >>> targets, modified_df = get_target(data, 'target', inplace=False)
    >>> print(targets.head())
    >>> print(modified_df.columns)

    Notes
    -----
    This function is particularly useful when preparing data for machine
    learning models, where separating features from labels is a common task.

    See Also
    --------
    extract_target : Similar function with enhanced capabilities for handling
                     more complex scenarios.
    
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if isinstance(tname, str):
        tname = [tname]  # Convert string to list for uniform processing

    missing_columns = [name for name in tname if name not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Target name(s) not found in DataFrame columns: {missing_columns}")

    target_data = df[tname]
    if inplace:
        df.drop(tname, axis=1, inplace=True)

    return target_data, df

@Dataify(auto_columns=True )
def binning_statistic(
    data, categorical_column, 
    value_column, 
    statistic='mean'
    ):
    """
    Compute a statistic for each category in a categorical column of a dataset.

    This function categorizes the data into bins based on a categorical variable and then
    applies a statistical function to the values of another column for each category.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    value_column : str
        Name of the column in `data` from which the statistic will be calculated.
    statistic : str, optional
        The statistic to compute (default is 'mean'). Other options include 
        'sum', 'count','median', 'min', 'max', etc.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding computed statistic.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import binning_statistic
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Value': [1, 2, 3, 4, 5, 6, 7]
    ... })
    >>> binning_statistic(df, 'Category', 'Value', statistic='mean')
       Category  Mean_Value
    0        A         3.33
    1        B         3.50
    2        C         5.50
    """
    if statistic not in ('mean', 'sum', 'count', 'median', 'min',
                         'max', 'proportion'):
        raise ValueError(
            "Unsupported statistic. Please choose from 'mean',"
            " 'sum', 'count', 'median', 'min', 'max', 'proportion'.")

    is_categorical(data, categorical_column)
    exist_features(data, features =value_column, name ="value_column")
    grouped_data = data.groupby(categorical_column)[value_column]
    
    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{value_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{value_column}')
    elif statistic == 'count':
        result = grouped_data.count().reset_index(name=f'Count_{value_column}')
    elif statistic == 'median':
        result = grouped_data.median().reset_index(name=f'Median_{value_column}')
    elif statistic == 'min':
        result = grouped_data.min().reset_index(name=f'Min_{value_column}')
    elif statistic == 'max':
        result = grouped_data.max().reset_index(name=f'Max_{value_column}')
    elif statistic == 'proportion':
        total_count = data[value_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{value_column}')
        
    return result

@Dataify(auto_columns=True)
def category_count(data, /, *categorical_columns, error='raise'):
    """
    Count occurrences of each category in one or more categorical columns 
    of a dataset.
    
    This function computes the frequency of each unique category in the specified
    categorical columns of a pandas DataFrame and handles different ways of error
    reporting including raising an error, warning, or ignoring the error when a
    specified column is not found.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    *categorical_columns : str
        One or multiple names of the columns in `data` which contain the 
        categorical variables.
    error : str, optional
        Error handling strategy - 'raise' (default), 'warn', or 'ignore' which
        dictates the action when a categorical column is not found.

    Returns
    -------
    counts : DataFrame
        A DataFrame with each category and the corresponding count from each
        categorical column. If multiple columns are provided, columns are named as
        'Category_i' and 'Count_i'.

    Raises
    ------
    ValueError
        If any categorical column is not found in the DataFrame and error is 'raise'.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import category_count
    >>> df = pd.DataFrame({
    ...     'Fruit': ['Apple', 'Banana', 'Apple', 'Cherry', 'Banana', 'Apple'],
    ...     'Color': ['Red', 'Yellow', 'Green', 'Red', 'Yellow', 'Green']
    ... })
    >>> category_count(df, 'Fruit', 'Color')
       Category_1  Count_1 Category_2  Count_2
    0      Apple        3        Red        2
    1     Banana        2     Yellow        2
    2     Cherry        1      Green        2
    >>> category_count(df, 'NonExistentColumn', error='warn')
    Warning: Column 'NonExistentColumn' not found in the dataframe.
    Empty DataFrame
    Columns: []
    Index: []
    """
    results = []
    for i, column in enumerate(categorical_columns, 1):
        if column not in data.columns:
            message = f"Column '{column}' not found in the dataframe."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                continue
            elif error == 'ignore':
                continue

        count = data[column].value_counts().reset_index()
        count.columns = [f'Category_{i}', f'Count_{i}']
        results.append(count)

    if not results:
        return pd.DataFrame()

    # Merge all results into a single DataFrame
    final_df = functools.reduce(lambda left, right: pd.merge(
        left, right, left_index=True, right_index=True, how='outer'), results)
    final_df.fillna(value=np.nan, inplace=True)
    
    if len( results)==1: 
        final_df.columns =['Category', 'Count']
    return final_df

@Dataify(auto_columns=True) 
def soft_bin_stat(
    data, /, categorical_column, 
    target_column, 
    statistic='mean', 
    update=False, 
    ):
    """
    Compute a statistic for each category in a categorical 
    column based on a binary target.

    This function calculates statistics like mean, sum, or proportion 
    for a binary target variable, grouped by categories in a 
    specified column.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    target_column : str
        Name of the column in `data` which contains the binary target variable.
    statistic : str, optional
        The statistic to compute for the binary target (default is 'mean').
        Other options include 'sum' and 'proportion'.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding 
        computed statistic.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import soft_bin_stat
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Target': [1, 0, 1, 0, 1, 0, 1]
    ... })
    >>> soft_bin_stat(df, 'Category', 'Target', statistic='mean')
       Category  Mean_Target
    0        A     0.666667
    1        B     0.500000
    2        C     0.500000

    >>> soft_bin_stat(df.values, 'col_0', 'col_1', statistic='mean')
      col_0  Mean_col_1
    0     A    0.666667
    1     B    0.500000
    2     C    0.500000
    """
    if statistic not in ['mean', 'sum', 'proportion']:
        raise ValueError("Unsupported statistic. Please choose from "
                         "'mean', 'sum', 'proportion'.")
    
    is_categorical(data, categorical_column)
    exist_features(data, features= target_column, name ='Target')
    grouped_data = data.groupby(categorical_column)[target_column]
    
    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{target_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{target_column}')
    elif statistic == 'proportion':
        total_count = data[target_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{target_column}')

    return result

def reshape_to_dataframe(flattened_array, columns, error ='raise'):
    """
    Reshapes a flattened array into a pandas DataFrame or Series based on the
    provided column names. If the number of columns does not allow reshaping
    to match the array length, it raises an error.

    Parameters
    ----------
    flattened_array : array-like
        The flattened array to reshape.
    columns : list of str
        The list of column names for the DataFrame. If a single name is provided,
        a Series is returned.
        
    error : {'raise', 'warn', 'ignore'}, default 'raise'
        Specifies how to handle the situation when the number of elements in the
        flattened array is not compatible with the number of columns required for
        reshaping. Options are:
        
        - 'raise': Raises a ValueError. This is the default behavior.
        - 'warn': Emits a warning, but still returns the original flattened array.
        - 'ignore': Does nothing about the error, just returns the original
          flattened array.
        
    Returns
    -------
    pandas.DataFrame or pandas.Series
        A DataFrame or Series reshaped according to the specified columns.

    Raises
    ------
    ValueError
        If the total number of elements in the flattened array does not match
        the required number for a complete reshaping.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import reshape_to_dataframe
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> print(reshape_to_dataframe(data, ['A', 'B', 'C']))  # DataFrame with 2 rows and 3 columns
    >>> print(reshape_to_dataframe(data, 'A'))  # Series with 6 elements
    >>> print(reshape_to_dataframe(data, ['A']))  # DataFrame with 6 rows and 1 column
    """
    # Check if the reshaping is possible
    is_string = isinstance ( columns, str )
    # Convert single string column name to list
    if isinstance(columns, str):
        columns = [columns]
        
    num_elements = len(flattened_array)
    num_columns = len(columns)
    if num_elements % num_columns != 0:
        message = ("The number of elements in the flattened array is not"
                   " compatible with the number of columns.")
        if error =="raise": 
            raise ValueError(message)
        elif error =='warn': 
            warnings.warn(message, UserWarning)
        return flattened_array
    # Calculate the number of rows that will be needed
    num_rows = num_elements // num_columns

    # Reshape the array
    reshaped_array = np.reshape(flattened_array, (num_rows, num_columns))

    # Check if we need to return a DataFrame or a Series
    if num_columns == 1 and is_string:
        return pd.Series(reshaped_array[:, 0], name=columns[0])
    else:
        return pd.DataFrame(reshaped_array, columns=columns)

def save_figure(fig, filename=None, dpi=300, close=True, ax=None, 
                tight_layout=False, bbox_inches='tight'):
    """
    Saves a matplotlib figure to a file and optionally closes it. 
    Automatically generates a unique filename if not provided.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str, optional
        The name of the file to save the figure to. If None, a unique name
        is generated based on the current date-time.
    dpi : int, optional
        The resolution of the output file in dots per inch.
    close : bool, optional
        Whether to close the figure after saving.
    ax : matplotlib.axes.Axes or array-like of Axes, optional
        Axes object(s) to perform operations on before saving.
    tight_layout : bool, optional
        Whether to adjust subplot parameters to give specified padding.
    bbox_inches : str, optional
        Bounding box in inches: 'tight' or a specific value.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from gofast.tools.baseutils import save_figure
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> ax.plot(x, y)
    >>> save_figure(fig, close=True, ax=ax)

    Notes
    -----
    If the filename is not specified, this function generates a filename that
    is unique to the second, using the pattern 'figure_YYYYMMDD_HHMMSS.png'.
    If two figures are saved within the same second, it appends microseconds
    to ensure uniqueness.
    """
    # Generate a unique filename if not provided
    if filename is None:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figure_{date_time}.png"
        while os.path.exists(filename):
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"figure_{date_time}.png"

    # Adjust layout if requested
    if tight_layout:
        fig.tight_layout()

    # Optionally adjust axis properties
    if ax is not None:
        if isinstance(ax, (list, tuple, np.ndarray)):
            for a in ax:
                a.grid(True)
        else:
            ax.grid(True)
    
    # Save the figure
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved as '{filename}' with dpi={dpi}")

    # Optionally close the figure
    if close:
        plt.close(fig)
        print("Figure closed.")

def _handle_non_numeric(data, action='normalize'):
    """Process input data (Series, DataFrame, or ndarray) to ensure 
    it contains only numeric data.
    
    Parameters:
    data (pandas.Series, pandas.DataFrame, numpy.ndarray): Input data to process.
    
    Returns:
    numpy.ndarray: An array containing only numeric data.
    
    Raises:
    ValueError: If the processed data is empty after removing non-numeric types.
    TypeError: If the input is not a Series, DataFrame, or ndarray.
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame to use select_dtypes
            data = data.to_frame()
            # Convert back to Series if needed
            numeric_data = data.select_dtypes(include=[np.number]).squeeze()  
        elif isinstance(data, pd.DataFrame):
            # For DataFrame, use select_dtypes to filter numeric data.
            numeric_data = data.select_dtypes(include=[np.number])
        # For pandas data structures, select only numeric data types.
        if numeric_data.empty:
            raise ValueError(f"No numeric data to {action}.")
    elif isinstance(data, np.ndarray):
        # For numpy arrays, ensure the dtype is numeric.
        if not np.issubdtype(data.dtype, np.number):
            # Attempt to convert non-numeric numpy array to a numeric one by coercion
            try:
                numeric_data = data.astype(np.float64)
            except ValueError:
                raise ValueError("Array contains non-numeric data that cannot"
                                 " be converted to numeric type.")
        else:
            numeric_data = data
    else:
        raise TypeError("Input must be a pandas Series, DataFrame, or a numpy array.")
    
    # Check if resulting numeric data is empty
    if numeric_data.size == 0:
        raise ValueError("No numeric data available after processing.")
    
    return numeric_data


def _nan_checker(arr, allow_nan=False):
    """Check and handle NaN values in a numpy array, pandas Series, 
    or pandas DataFrame.

    Parameters:
    arr (numpy.ndarray, pandas.Series, pandas.DataFrame): The data to check
    for NaNs.
    allow_nan (bool): If False, raises an error if NaNs are found. If True, 
    replaces NaNs with zero.

    Returns:
    numpy.ndarray, pandas.Series, pandas.DataFrame: Data with NaNs handled 
    according to allow_nan.

    Raises:
    ValueError: If NaNs are found and allow_nan is False.
    """
    # Check for NaNs across different types
    if not allow_nan:
        if isinstance(arr, (np.ndarray, pd.Series, pd.DataFrame)): 
            contain_nans = np.isnan(arr).any() if isinstance (
                arr, np.ndarray)  else pd.isnull(arr).values.any()
            if contain_nans:
                raise ValueError("NaN values found, set allow_nan=True to handle them.")
    if allow_nan:
        if isinstance(arr, np.ndarray):
            arr = np.nan_to_mode(arr)  # Replace NaNs with zero for numpy arrays
        elif isinstance(arr, (pd.Series, pd.DataFrame)):
            arr = arr.fillna(0)  # Replace NaNs with zero for pandas Series or DataFrame
    
    return arr

def normalizer(
    *arrays: tuple[np.ndarray], 
    method: str = '01', 
    scaler: str = 'naive', 
    allow_nan: bool = False, 
    axis: Optional[int] = None) -> List[np.ndarray]:
    """
    Normalize given arrays using a specified method and scaler, optionally 
    along a specified axis. 
    
    Handles non-numeric data and NaNs according to the parameters.

    Parameters
    ----------
    arrays : tuple of np.ndarray
        A tuple containing one or more arrays (either 1D or 2D) to be normalized.
    method : str, default '01'
        Specifies the normalization method to apply. Options include:
        - '01' : Normalizes the data to the range [0, 1].
        - 'zscore' : Uses Z-score normalization (standardization).
        - 'sum' : Scales the data so that the sum is 1.
        Note that the 'sum' method is not compatible with the 'sklearn' scaler.
    scaler : str, default 'naive'
        Specifies the type of scaling technique to use. Options include:
        - 'naive' : Simple mathematical operations based on the method.
        - 'sklearn' : Utilizes scikit-learn's MinMaxScaler or StandardScaler,
          depending on the `method` specified.
    allow_nan : bool, default False
        Determines how NaN values should be handled. If False, the function 
        will raise an error if NaN values are present. If True, NaNs will be 
        replaced with zero or handled by an imputer,
        depending on the context.
    axis : int, optional
        The axis along which to normalize the data. By default (None), the 
        data is normalized based on all elements. If specified, normalization 
        is done along the axis for 2D arrays:
        - axis=0 : Normalize each column.
        - axis=1 : Normalize each row.

    Returns
    -------
    list of np.ndarray
        A list of normalized arrays. If only one array is provided, a single 
        normalized array is returned.

    Raises
    ------
    ValueError
        If `allow_nan` is False and NaNs are detected.
        If an invalid normalization method or combination of scaler and method 
        is specified.

    Notes
    -----
    - The function internally converts pandas DataFrames and Series to numpy 
      arrays for processing.
    - Non-numeric data types within arrays are filtered out before normalization.
    - It's important to consider the scale of values and distribution of data
      when choosing a normalization method,
      as each method can significantly affect the outcome and interpretation 
      of results.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import normalizer 
    >>> arr = np.array([10, 20, 30, 40, 50])
    >>> normalizer(arr, method='01', scaler='naive')
    array([0. , 0.25, 0.5 , 0.75, 1. ])
    
    >>> arr2d = np.array([[1, 2], [3, 4]])
    >>> normalizer(arr2d, method='zscore', axis=0)
    array([[-1., -1.], [ 1.,  1.]])
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    def _normalize_array(arr, method, scaler):
        arr = _nan_checker(arr, allow_nan = allow_nan )
        if scaler in ['sklearn', 'scikit-learn']:
            is_array_1d =False 
            if method == 'sum':
                raise ValueError("`sum` method is not valid with `scaler`='sklearn'.")
            scaler_object = MinMaxScaler() if method == '01' else StandardScaler()
            
            if arr.ndim ==1: 
                arr = ( 
                    np.asarray(arr).reshape(-1, 0) if axis ==0 
                    else np.asarray(arr).reshape(1, -1)  
                    )
                is_array_1d =True 
            scaled = scaler_object.fit_transform(
                arr if axis == 0 else arr.T).T if axis == 0 else\
                scaler_object.fit_transform(arr.T).T
            if is_array_1d: 
                scaled = scaled.flatten() 
        else:  # naive scaling
            arr, name_or_columns, index  = pandas_manager(arr )
            if axis is None:
                arr_min = np.min(arr)
                arr_max = np.max(arr)
            else:
                arr_min = np.min(arr, axis=axis, keepdims=True)
                arr_max = np.max(arr, axis=axis, keepdims=True)

            if method == '01':
                
                scaled = (arr - arr_min) / (arr_max - arr_min)
            elif method == 'zscore':
                mean = np.mean(arr, axis=axis, keepdims=True)
                std = np.std(arr, axis=axis, keepdims=True)
                scaled = (arr - mean) / std
            elif method == 'sum':
                sum_val = np.sum(arr, axis=axis, keepdims=True)
                scaled = arr / sum_val
            else:
                raise ValueError(f"Unknown method '{method}'. Valid methods"
                                 " are '01', 'zscore', and 'sum'.")
                
        # revert back to series or dataframe is given as it 
        if name_or_columns is not None: 
            scaled = pandas_manager(
                scaled, todo='set', name_or_columns=name_or_columns, 
                index= index 
                )

        return scaled
    # Normalize each array
    normalized_arrays = []
    for arr in arrays:
        if not hasattr(arr, '__array__'):
            arr = np.asarray(arr)
        arr = _handle_non_numeric(arr)
        normalized = _normalize_array(arr, method, scaler)
        normalized_arrays.append(normalized)

    return normalized_arrays[0] if len(normalized_arrays) == 1 else normalized_arrays

def smooth1d(
    ar, /, 
    drop_outliers:bool=True, 
    ma:bool=True, 
    absolute:bool=False,
    interpolate:bool=False, 
    view:bool=False , 
    x: ArrayLike=None, 
    xlabel:str =None, 
    ylabel:str =None, 
    fig_size:tuple = ( 10, 5) 
    )-> ArrayLike[float]: 
    """ Smooth one-dimensional array. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d 
       Array of one-dimensional 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value. This seems more 
       realistic.
       
    interpolate: bool, default=False 
       Interpolate value to fit the original data size after NaN filling. 
       
    absolute: bool, default=False, 
       keep postive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    view: bool, default =False 
       Display curves 
    x: ArrayLike, optional 
       Abscissa array for visualization. If given, it must be consistent 
       with the given array `ar`. Raises error otherwise. 
    xlabel: str, optional 
       Label of x 
    ylabel:str, optional 
       label of y  
    fig_size: tuple , default=(10, 5)
       Matplotlib figure size
       
    Returns 
    --------
    yc: ArrayLike 
       Smoothed array value. 
       
    Examples 
    ---------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils  import smooth1d 
    >>> # add Guassian Noise 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20 ) * 20 + np.random.normal ( 20 )
    >>> ar [:7 ]
    array([6.42891445e+00, 3.75072493e-02, 1.82905357e+01, 2.92957265e+01,
           6.20589038e+01, 2.26399535e+01, 1.12596434e+01])
    >>> arc = smooth1d (ar, view =True , ma =False )
    >>> arc [:7 ]
    array([12.08603102, 15.29819907, 18.017749  , 20.27968322, 22.11900412,
           23.5707141 , 24.66981557])
    >>> arc = smooth1d (ar, view =True )# ma=True by default 
    array([ 5.0071604 ,  5.90839339,  9.6264018 , 13.94679804, 17.67369252,
           20.34922943, 22.00836725])
    """
    from .mathex import moving_average 
    # convert data into an iterable object 
    ar = np.array(
        is_iterable(ar, exclude_string = True , transform =True )) 
    
    if not _is_arraylike_1d(ar): 
        raise TypeError("Expect one-dimensional array. Use `gofast.smoothing`"
                        " for handling two-dimensional array.")
    if not _is_numeric_dtype(ar): 
        raise ValueError (f"{ar.dtype.name!r} is not allowed. Expect a numeric"
                          " array")
        
    arr = ar.copy() 
    if drop_outliers: 
        arr = remove_outliers( 
            arr, fill_value = np.nan , interpolate = interpolate )
    # Nan is not allow so fill NaN if exists in array 
    # is arraylike 1d 
    if not interpolate:
        # fill NaN 
        arr = reshape ( fillNaN( arr , method ='both') ) 
    if ma: 
        arr = moving_average(arr, method ='sma')
    # if extrapolation give negative  values
    # whether to keep as it was or convert to positive values. 
    # note that converting to positive values is 
    arr, *_  = scale_y ( arr ) 
    # if extrapolation gives negative values
    # convert to positive values or keep it intact. 
    # note that converting to positive values is 
    # can be used as the last option when array 
    # data must be positive.
    if absolute: 
        arr = np.abs (arr )
    if view: 
        x = np.arange ( len(ar )) if x is None else np.array (x )

        check_consistency_size( x, ar )
            
        fig,  ax = plt.subplots (1, 1, figsize = fig_size)
        ax.plot (x, 
                 ar , 
                 'ok-', 
                 label ='raw curve'
                 )
        ax.plot (x, 
                 arr, 
                 c='#0A4CEE',
                 marker = 'o', 
                 label ='smooth curve'
                 ) 
        
        ax.legend ( ) 
        ax.set_xlabel (xlabel or '')
        ax.set_ylabel ( ylabel or '') 
        
    return arr 

def smoothing (
    ar, /, 
    drop_outliers = True ,
    ma=True,
    absolute =False,
    interpolate=False, 
    axis = 0, 
    view = False, 
    fig_size =(7, 7), 
    xlabel =None, 
    ylabel =None , 
    cmap ='binary'
    ): 
    """ Smooth data along axis. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d or 2d 
       One dimensional or two dimensional array. 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing along the given axis 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value along axis. This seems 
       more realistic rather than using only the scaling method. 
       
    absolute: bool, default=False, 
       keep positive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    axis: int, default=0 
       Axis along with the data must be smoothed. The default is the along  
       the row. 
       
    view: bool, default =False 
       Visualize the two dimensional raw and smoothing grid. 
       
    xlabel: str, optional 
       Label of x 
       
    ylabel:str, optional 
    
       label of y  
    fig_size: tuple , default=(7, 5)
       Matplotlib figure size 
       
    cmap: str, default='binary'
       Matplotlib.colormap to manage the `view` color 
      
    Return 
    --------
    arr0: ArrayLike 
       Smoothed array value. 
    
    Examples 
    ---------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils  import smoothing
    >>> # add Guassian Noises 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20, 7 ) * 20 + np.random.normal ( 20, 7 )
    >>> ar [:3, :3 ]
    array([[ 31.5265026 ,  18.82693352,  34.5459903 ],
           [ 36.94091413,  12.20273182,  32.44342041],
           [-12.90613711,  10.34646896,   1.33559714]])
    >>> arc = smoothing (ar, view =True , ma =False )
    >>> arc [:3, :3 ]
    array([[32.20356863, 17.18624398, 41.22258603],
           [33.46353806, 15.56839464, 19.20963317],
           [23.22466498, 13.8985316 ,  5.04748584]])
    >>> arcma = smoothing (ar, view =True )# ma=True by default
    >>> arcma [:3, :3 ]
    array([[23.96547827,  8.48064226, 31.81490918],
           [26.21374675, 13.33233065, 12.29345026],
           [22.60143346, 16.77242118,  2.07931194]])
    >>> arcma_1 = smoothing (ar, view =True, axis =1 )
    >>> arcma_1 [:3, :3 ]
    array([[18.74017857, 26.91532187, 32.02914421],
           [18.4056216 , 21.81293014, 21.98535213],
           [-1.44359989,  3.49228057,  7.51734762]])
    """
    ar = np.array ( 
        is_iterable(ar, exclude_string = True , transform =True )
        ) 
    if ( 
            str (axis).lower().find('1')>=0 
            or str(axis).lower().find('column')>=0
            ): 
        axis = 1 
    else : axis =0 
    
    if _is_arraylike_1d(ar): 
        ar = reshape ( ar, axis = 0 ) 
    # make a copy
    arr = ar.copy() 
    along_axis = arr.shape [1] if axis == 0 else len(ar) 
    arr0 = np.zeros_like (arr)
    for ix in range (along_axis): 
        value = arr [:, ix ] if axis ==0 else arr[ix , :]
        yc = smooth1d(value, drop_outliers = drop_outliers , 
                      ma= ma, view =False , absolute =absolute , 
                      interpolate= interpolate, 
                      ) 
        if axis ==0: 
            arr0[:, ix ] = yc 
        else : arr0[ix, :] = yc 
        
    if view: 
        fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True,
                                 figsize = fig_size )
        ax[0].imshow(arr ,interpolation='nearest', label ='Raw Grid', 
                     cmap = cmap )
        ax[1].imshow (arr0, interpolation ='nearest', label = 'Smooth Grid', 
                      cmap =cmap  )
        
        ax[0].set_title ('Raw Grid') 
        ax[0].set_xlabel (xlabel or '')
        ax[0].set_ylabel ( ylabel or '')
        ax[1].set_title ('Smooth Grid') 
        ax[1].set_xlabel (xlabel or '')
        ax[1].set_ylabel ( ylabel or '')
        plt.legend
        plt.show () 
        
    if 1 in ar.shape: 
        arr0 = reshape (arr0 )
        
    return arr0 
    
def _count_local_minima(arr: ArrayLike, method: str = 'robust') -> int:
    if method == 'base':
        return sum(
            1 for i in range(1, len(arr) - 1) if arr[i] < arr[i - 1] 
            and arr[i] < arr[i + 1]
            )
    else:
        return len(argrelextrema(np.array(arr), np.less)[0])

def scale_y(
    y: ArrayLike, 
    x: ArrayLike = None, 
    deg: int = "auto", 
    func: _F = None, 
    return_xf: bool = False, 
    view: bool = False
) -> Tuple[ArrayLike, ArrayLike, _F]:
    """
    Scaling value using a fitting curve.

    Create polyfit function from specific data points `x` to correct `y` 
    values.

    Parameters
    ----------
    y : ArrayLike
        Array-like of y-axis. This is the array of values to be scaled.
    x : ArrayLike, optional
        Array-like of x-axis. If `x` is given, it should be the same length 
        as `y`, otherwise an error will occur. Default is ``None``.
    deg : int, optional
        Polynomial degree. If value is ``auto`` or ``None``, it will be computed 
        using the length of extrema (local and/or global) values.
    func : _F, optional
        Callable - The model function, ``f(x, ...)``. It must take the 
        independent variable as the first argument and the parameters to 
        fit as separate remaining arguments. `func` can be a ``linear`` 
        function i.e. for ``f(x)= ax + b`` where `a` is slope and `b` is 
        the intercept value. It is recommended to set up a custom function 
        according to the `y` value distribution for better fitting. If 
        `func` is given, `deg` is not needed.
    return_xf : bool, optional
        If True, returns the new x-axis and the fitting function. Default 
        is ``False``.
    view : bool, optional
        If True, visualizes the original and scaled data. Default is 
        ``False``.

    Returns
    -------
    yc : ArrayLike
        Array of scaled/projected sample values obtained from `f`.
    x : ArrayLike
        New x-axis generated from the samples (if `return_xf` is True).
    f : _F
        Linear or polynomial function `f` (if `return_xf` is True).

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(42)
    >>> x0 = 10 * np.random.rand(50)
    >>> y = 2 * x0 + np.random.randn(50) - 1
    >>> plt.scatter(x0, y)
    >>> yc, x, f = scale_y(y, return_xf=True, view=True)
    >>> plt.plot(x, y, label='Original Data')
    >>> plt.plot(x, yc, label='Scaled Data')
    >>> plt.legend()
    >>> plt.show()

    Notes
    -----
    - The function checks if `x` and `y` are of the same length.
    - If `func` is provided, `deg` is ignored.
    - The function raises a `TypeError` if `func` is not callable.
    - The degree of the polynomial is determined by the number of local 
      minima plus one.
    - In case of fitting errors, a `ValueError` is raised with a suggestion 
      to check the polynomial degree.
      
    References 
    -----------
    Wikipedia, Curve fitting, https://en.wikipedia.org/wiki/Curve_fitting
    Wikipedia, Polynomial interpolation, https://en.wikipedia.org/wiki/Polynomial_interpolation
    """
    _, name_or_columns, index = pandas_manager(y )
    
    y = check_y(y, y_numeric =True )
    if func is not None and (not callable(func) or not hasattr(func, '__call__')):
        raise TypeError(f"`func` argument is not a callable; got {type(func).__name__!r}")

    degree = _count_local_minima(y) + 1
    if x is None:
        x = np.arange(len(y))
        
    x = check_y(x, input_name="x")
    
    if len(x) != len(y):
        raise ValueError("`x` and `y` arrays must have the same length."
                         f" Got lengths {len(x)} and {len(y)}.")
    
    deg= None if deg =='auto' else deg 
    
    try: 
        coeff = np.polyfit(x, y, int(deg) if deg is not None else degree)
        f = np.poly1d(coeff) if func is None else func
        yc = f(x)
    except np.linalg.LinAlgError:
        raise ValueError("Check the number of degrees. SVD did not converge"
                         " in Linear Least Squares.")

    if view: 
        plt.plot(x, y, "-ok", label='Original Data')
        plt.plot(x, yc, "-or", label='Scaled Data')
        plt.xlabel ("x") ; plt.ylabel("y")
        plt.legend()
        plt.show()
    
    yc = pandas_manager(yc, todo="set", name_or_columns=name_or_columns,
                        index =index ) 
    
    return (yc, x, f) if return_xf else yc

def _visualize_interpolation(
    original_data: Union[np.ndarray, pd.Series, pd.DataFrame], 
    interpolated_data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
    """
    Helper function to visualize original and interpolated data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if isinstance(original_data, (pd.Series, np.ndarray)
                  ) and original_data.ndim == 1:
        axes[0].plot(original_data, label='Original Data', marker='o')
        axes[1].plot(interpolated_data, label='Interpolated Data', marker='o')
    elif isinstance(original_data, (pd.DataFrame, np.ndarray)
                    ) and original_data.ndim == 2:
        axes[0].imshow(original_data, aspect='auto', interpolation='none')
        axes[1].imshow(interpolated_data, aspect='auto', interpolation='none')
    
    axes[0].set_title('Original Data')
    axes[1].set_title('Interpolated Data')
    plt.show()

def interpolate_data(
    data: Union[ArrayLike, Series, DataFrame], 
    method: str = 'slinear', 
    order: Optional[int] = None, 
    fill_value: str = 'extrapolate', 
    axis: int = 0, 
    drop_outliers: bool = False, 
    outlier_method: str = "IQR", 
    view: bool = False, 
    **kwargs
) -> Union[ArrayLike, Series, DataFrame]:
    """
    Interpolates 1D or 2D data, allowing for NaN values, and visualizes 
    the result if requested.

    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, pd.DataFrame]
        Input data to be interpolated. Can be a 1D or 2D numpy array, 
        pandas Series, or pandas DataFrame.
    method : str, optional
        Method of interpolation. Options include 'linear', 'nearest', 
        'zero', 'slinear', 'quadratic', and 'cubic'. Default is 'slinear'.
    order : Optional[int], optional
        The order of the spline interpolation. Only applicable for some 
        methods. Default is None.
    fill_value : str, optional
        Specifies the fill value for points outside the interpolation 
        domain. Default is 'extrapolate'.
    axis : int, optional
        Axis along which to interpolate, if data is a DataFrame. Default 
        is 0.
    drop_outliers : bool, optional
        If True, outliers will be removed before interpolation. Default 
        is False.
    outlier_method : str, optional
        Method for outlier detection if drop_outliers is True. Options 
        are 'IQR' and 'z-score'. Default is 'IQR'.
    view : bool, optional
        If True, visualizes the original and interpolated data in two 
        panels. Default is False.
    **kwargs
        Additional keyword arguments to pass to the interpolation 
        function.

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Interpolated data, returned in the same type as the input data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import interpolate_data
    >>> s = pd.Series([1, np.nan, 3, 4, np.nan, 6])
    >>> interpolate_data(s, view=True)
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    5    6.0
    dtype: float64

    >>> df = pd.DataFrame([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]])
    >>> interpolate_data(df, view=True)
         0    1    2
    0  1.0  6.5  3.0
    1  4.0  5.0  6.0
    2  4.0  8.0  9.0

    Notes
    -----
    - The 'slinear' method is not available for 2D interpolation. It 
      defaults to 'linear' for 2D data.
    - The 'extrapolate' fill_value is not available for 2D interpolation 
      and defaults to 0.
    - If drop_outliers is True, the specified outlier detection method 
      will be applied before interpolation.
    """
    valid_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    method = str(method).lower() 
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method. Expected one of {valid_methods}, got {method}.")
    
    
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.asarray(data)
    
    data = _handle_non_numeric(data, action="interpolate")
    data, name_or_columns, index = pandas_manager(data)
    if drop_outliers: 
        data = remove_outliers(data, method=outlier_method)
        
    if isinstance(data, pd.Series):
        x = np.arange(len(data))
        y = data.values
        mask = ~np.isnan(y)
        f = interp1d(x[mask], y[mask], kind=method, fill_value=fill_value,
                     bounds_error=False, **kwargs)
        data_interp = f(x)
        result = pd.Series(data_interp, index=data.index)
    
    elif isinstance(data, pd.DataFrame):
        data_interp = data.apply(lambda col: interpolate_data(
            col, method=method, 
            order=order, fill_value=fill_value, axis=axis, 
            **kwargs), axis=axis)
        result = data_interp
    
    elif data.ndim == 1:
        x = np.arange(len(data))
        y = data
        mask = ~np.isnan(y)
        f = interp1d(x[mask], y[mask], kind=method, fill_value=fill_value,
                     bounds_error=False, **kwargs)
        data_interp = f(x)
        result = data_interp
    
    elif data.ndim == 2:
        # slinear not available for two dimensional interpolation 
        method = 'linear' if method.find("linear")>=0 else method 
        
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        x_grid, y_grid = np.meshgrid(x, y)
        mask = ~np.isnan(data)
        points = np.array((y_grid[mask], x_grid[mask])).T
        values = data[mask]
        x_new, y_new = np.meshgrid(x, y)
        # 'extrapolate fill_value' not available.
        fill_value = 0. if str(fill_value).find("extra")>=0 else fill_value
        data_interp = griddata(points, values, (
            y_new, x_new), method=method, fill_value=fill_value, **kwargs)
        result = data_interp
 
    if view:
        _visualize_interpolation(data, result)
    
    if name_or_columns is not None: 
        result = pandas_manager(
            result, 
            name_or_columns = name_or_columns,
            index = index, 
            todo='set'
        )
    
    return result

def denormalizer(
    data: Union[np.ndarray, pd.Series, pd.DataFrame], 
    min_value: float, max_value: float, 
    method: str = '01', 
    std_dev_factor: float = 3
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Denormalizes data from a normalized scale back to its original scale.

    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, pd.DataFrame]
        The data to be denormalized, can be a NumPy array, 
        pandas Series, or pandas DataFrame.
    min_value : float
        The minimum value of the original scale before normalization.
    max_value : float
        The maximum value of the original scale before normalization.
    method : str, optional
        The normalization method used. Supported methods are:
        - '01' : Min-Max normalization to range [0, 1].
        - 'zscore' : Standard score normalization (zero mean, unit variance).
        - 'sum' : Normalization by sum of elements.
        Default is '01'.
    std_dev_factor : float, optional
        The factor determining the range for standard deviation. 
        This is used only for 'zscore' method. Default is 3.

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        The denormalized data, converted back to its original scale.

    Raises
    ------
    ValueError
        If an unsupported normalization method is provided.

    Notes
    -----
    The denormalization process depends on the normalization method:
    
    - For Min-Max normalization ('01'):
     .. math:: 
         
         `denormalized\_data = data \cdot (max\_value - min\_value) + min\_value`
    
    - For z-score normalization ('zscore'):
      Assuming the original data follows a normal distribution:
          
      .. math:: 
          `denormalized\_data = data \cdot std\_dev + mean`
      where 
      
      .. math::
          `mean = \frac{min\_value + max\_value}{2}`
      and 
      
      .. math::
          `std\_dev = \frac{max\_value - min\_value}{2 \cdot std\_dev\_factor}`
    
    - For sum normalization ('sum'):
        
      .. math::
          
          `denormalized\_data = data \cdot \frac{max\_value - min\_value}{\sum(data)} + min\_value`

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.tools.baseutils import denormalizer 
    
    >>> normalized_data = np.array([0, 0.5, 1])
    >>> min_value = 10
    >>> max_value = 20
    >>> denormalized_data = denormalizer(normalized_data, min_value, max_value)
    >>> print(denormalized_data)
    [10. 15. 20.]

    >>> normalized_series = pd.Series([0, 0.5, 1])
    >>> denormalized_series = denormalizer(normalized_series, min_value, max_value)
    >>> print(denormalized_series)
    0    10.0
    1    15.0
    2    20.0
    dtype: float64

    >>> normalized_df = pd.DataFrame([[0, 0.5], [1, 0.2]])
    >>> denormalized_df = denormalizer(normalized_df, min_value, max_value)
    >>> print(denormalized_df)
         0     1
    0  10.0  15.0
    1  20.0  12.0
    """

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data_values = data.to_numpy()
        is_pandas = True
    else:
        data_values = np.asarray(data)
        is_pandas = False

    if method == '01':
        denormalized_data = data_values * (max_value - min_value) + min_value
    elif method == 'zscore':
        mean = (min_value + max_value) / 2
        # Adjusting for specified standard deviation factor
        std_dev = (max_value - min_value) /  (2 * std_dev_factor)  
        denormalized_data = data_values * std_dev + mean
    elif method == 'sum':
        total = np.sum(data_values)
        denormalized_data = data_values * (max_value - min_value) / total + min_value
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    if is_pandas:
        if isinstance(data, pd.Series):
            return pd.Series(denormalized_data, index=data.index, name=data.name)
        elif isinstance(data, pd.DataFrame):
            return pd.DataFrame(denormalized_data, index=data.index, columns=data.columns)
    else:
        return denormalized_data

def _handle_get_action(
        data: Any, action: str, error: str) -> Union[bool, Tuple[np.ndarray, Any]]:
    is_pandas = isinstance(data, (pd.Series, pd.DataFrame))
    if action == 'check_only':
        return is_pandas

    arr = data.to_numpy() if is_pandas else data
    index = data.index if is_pandas else None  
    if error == 'raise' and not is_pandas:
        raise TypeError("Expected a pandas Series or DataFrame, but got a non-pandas type.")
    elif error == 'warn' and not is_pandas:
        warnings.warn("Expected a pandas Series or DataFrame, but got a non-pandas type.")

    name_or_columns = data.name if isinstance(data, pd.Series) else\
        data.columns if is_pandas else None
        
    if action=='keep_frame': 
        arr= data 
    return arr, name_or_columns, index 

def _handle_set_action(data: Any, name_or_columns: Any, action: str, error: str, 
                       index=None, 
                       ) -> Union[pd.Series, pd.DataFrame]:
    if name_or_columns is None and action != 'default':
        raise ValueError(
            "The 'name_or_columns' parameter cannot be None when setting data. "
            "Provide a valid 'name_or_columns' or set the 'action' to 'default'."
        )
    is_pandas = isinstance(data, (pd.Series, pd.DataFrame))
    
    if is_pandas and name_or_columns is None:
        # Not set anymore when is already a series or dataframe.
        name_or_columns =  data.name if isinstance(data, pd.Series) else data.columns 
        
    data = np.asarray(data).squeeze()
    if data.ndim == 1:
        if isinstance(name_or_columns, (list, tuple)):
            if error == 'raise':
                raise ValueError("1D array provided; 'name_or_columns' should be a single string.")
            name_or_columns = list(name_or_columns)[0]
        return pd.Series(data, name=name_or_columns, index =index  )
    else:
        if isinstance(name_or_columns, str):
            name_or_columns = [name_or_columns]
        return pd.DataFrame(data, columns=name_or_columns, index =index )

def pandas_manager(
    data: Any, todo: str = 'get', 
    name_or_columns: Any = None, 
    action: str = 'as_array', 
    error: str = 'passthrough', 
    index: bool = None
) -> Union[np.ndarray, Series, DataFrame, Tuple[np.ndarray, Any], bool]:
    """
    Manages pandas objects by getting or setting data, with error handling.

    Parameters
    ----------
    data : Any
        The input data which can be a numpy array, pandas Series, or DataFrame.
    todo : str, optional
        Specifies the action to perform: 'get' or 'set'. Default is 'get'.
    name_or_columns : Any, optional
        Name for Series or columns for DataFrame when setting data.
    action : str, optional
        Specifies the sub-action for 'get': 'as_array' or 'check_only' or 'keep_frame'. 
        - `as_array` converts data to Numpy array 
        - `check_only` checks only whether the data is passed as pandas Series or 
          DataFrame
        - `keep_frame` keeps the dataframe but returns additionals frame attributes 
          like columns/name  and index. 
        Default is 'as_array'.
    error : str, optional
        Error handling strategy: 'passthrough', 'raise', or 'warn'. 
        Default is 'passthrough'.
    index : bool, optional
        Whether to include the index when getting or setting data. Default is None.

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame, Tuple[np.ndarray, Any], bool]
        Depending on the action, returns different types of results:
        - If `todo` is 'get' and `action` is 'check_only': returns a boolean 
          indicating if the data is a pandas object.
        - If `todo` is 'get' and `action` is 'as_array': returns the data as 
          a NumPy array along with the original pandas attributes.
        - If `todo` is 'set': returns the data as a pandas Series or DataFrame.

    Raises
    ------
    ValueError
        If the 'todo' parameter is not 'get' or 'set'.
        If 'name_or_columns' is None when setting data and action is not 'default'.
    TypeError
        If the data is not a pandas Series or DataFrame when expected.

 
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.tools.baseutils import pandas_manager

    Get action with pandas DataFrame:
    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    >>> array, columns, index = pandas_manager(df, todo='get', action='as_array')
    >>> print(array)
    [[1 2]
     [3 4]]
    >>> print(columns)
    Index(['A', 'B'], dtype='object')

    Set action with 1D numpy array:
    >>> arr = np.array([1, 2, 3])
    >>> series = pandas_manager(arr, todo='set', name_or_columns='Value')
    >>> print(series)
    0    1
    1    2
    2    3
    Name: Value, dtype: int64

    Set action with 2D numpy array:
    >>> arr_2d = np.array([[1, 2], [3, 4]])
    >>> df_2d = pandas_manager(arr_2d, todo='set', name_or_columns=['A', 'B'])
    >>> print(df_2d)
       A  B
    0  1  2
    1  3  4
    """
    todo = todo.lower()
    if todo not in ('get', 'set'):
        raise ValueError("Invalid 'todo' parameter. Must be 'get' or 'set'.")

    if todo == 'get':
        return _handle_get_action(data, action, error)
    elif todo == 'set':
        return _handle_set_action(
            data, name_or_columns, action , error, index)




        
    
        


        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    