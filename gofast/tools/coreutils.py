# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
`coreutils` module provides a diverse set of utility functions and tools for
data manipulation, validation, formatting, and processing. 
"""

from __future__ import print_function 
import os 
import re 
import sys
import csv 
import copy  
import json
import h5py
import yaml
import scipy
import joblib
import pickle
import shutil
import numbers 
import random
import inspect
import hashlib 
import datetime  
import warnings
import itertools
import subprocess 
import multiprocessing
from zipfile import ZipFile
from six.moves import urllib 
from collections import defaultdict 
from collections.abc import Sequence
from concurrent.futures import as_completed 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .._gofastlog import gofastlog
from ..api.types import Union, Series,Tuple,Dict,Optional,Iterable, Any, Set
from ..api.types import _T,_Sub, _F, ArrayLike,List, DataFrame, NDArray, Text 
from ._dependency import import_optional_dependency
from ..compat.scipy import ensure_scipy_compatibility 
from ..compat.scipy import check_scipy_interpolate, optimize_minimize

_logger = gofastlog.get_gofast_logger(__name__)

__all__=[
     'add_noises_to',
     'adjust_to_samples',
     'assert_ratio',
     'check_dimensionality',
     'check_uniform_type',
     'cleaner',
     'colors_to_names',
     'cpath',
     'decompose_colormap',
     'denormalize',
     'display_infos',
     'download_progress_hook',
     'exist_features',
     'extract_coordinates',
     'features_in',
     'fetch_json_data_from_url',
     'fill_nan_in',
     'find_by_regex',
     'find_close_position',
     'find_features_in',
     'format_to_datetime',
     'generate_alpha_values',
     'generate_mpl_styles',
     'generic_getattr',
     'get_colors_and_alphas',
     'get_confidence_ratio',
     'get_installation_name',
     'get_params',
     'get_valid_key',
     'get_valid_kwargs',
     'get_xy_coordinates',
     'hex_to_rgb',
     'interpol_scipy',
     'is_classification_task',
     'is_depth_in',
     'is_in_if',
     'is_installing',
     'is_iterable',
     'is_module_installed',
     'ismissing',
     'load_serialized_data',
     'make_ids',
     'move_cfile',
     'nan_to_na', 
     'normalize_string',
     'numstr2dms',
     'pair_data',
     'parallelize_jobs',
     'parse_attrs',
     'parse_csv',
     'parse_json',
     'parse_md_data',
     'parse_yaml',
     'process_and_extract_data',
     'projection_validator',
     'random_sampling',
     'random_selector',
     'random_state_validator',
     'read_from_excelsheets',
     'read_main',
     'read_worksheets',
     'rename_files',
#     'repeat_item_insertion',
     'replace_data',
     'resample_data',
     'reshape',
     'sanitize_frame_cols',
     'sanitize_unicode_string',
     'save_job',
     'savepath_',
     'serialize_data',
     'smart_label_classifier',
     'split_train_test',
     'split_train_test_by_id',
     'squeeze_specific_dim',
     'store_or_write_hdf5',
     'str2columns',
     'test_set_check_id',
     'to_hdf5',
     'to_numeric_dtypes',
     'to_series_if',
     'type_of_target',
     'unpack_list_of_dicts',
     'url_checker',
     'validate_feature',
     'validate_ratio',
     'validate_url',
     'validate_url_by_validators',
     'wrap_infos',
     'zip_extractor'
 ] 

def format_to_datetime(data, date_col, verbose=0, **dt_kws):
    """
    Reformats a specified column in a DataFrame to Pandas datetime format.

    This function attempts to convert the values in the specified column of a 
    DataFrame to Pandas datetime objects. If the conversion is successful, 
    the DataFrame with the updated column is returned. If the conversion fails, 
    a message describing the error is printed, and the original 
    DataFrame is returned.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the column to be reformatted.
    date_col : str
        The name of the column to be converted to datetime format.
    verbose : int, optional
        Verbosity mode; 0 or 1. If 1, prints messages about the conversion 
        process.Default is 0 (silent mode).
    **dt_kws : dict, optional
        Additional keyword arguments to pass to `pd.to_datetime` function.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the specified column in datetime format. If conversion
        fails, the original DataFrame is returned.

    Raises
    ------
    ValueError
        If the specified column is not found in the DataFrame.

    Examples
    --------
    >>> from gofast.tools.coreutils import format_to_datetime
    >>> df = pd.DataFrame({
    ...     'Date': ['2021-01-01', '01/02/2021', '03-Jan-2021', '2021.04.01',
                     '05 May 2021'],
    ...     'Value': [1, 2, 3, 4, 5]
    ... })
    >>> df = format_to_datetime(df, 'Date')
    >>> print(df.dtypes)
    Date     datetime64[ns]
    Value             int64
    dtype: object
    """
    if date_col not in data.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")
    
    try:
        data[date_col] = pd.to_datetime(data[date_col], **dt_kws)
        if verbose: 
            print(f"Column '{date_col}' successfully converted to datetime format.")
    except Exception as e:
        print(f"Error converting '{date_col}' to datetime format: {e}")
        return data

    return data

def adjust_to_samples(n_samples, *values, initial_guess=None, error='warn'):
    """
    Adjusts the given values to match a total number of samples, aiming to distribute
    the samples evenly across the dimensions represented by the values. The function
    can adjust even if only one value is given.

    Parameters
    ----------
    n_samples : int
        The desired total number of samples.
    *values : int
        Variable length argument list representing the dimensions to adjust.
    initial_guess : float or None, optional
        An initial guess for the adjustment factor. If None, an automatic guess is made.
    error : str, optional
        Error handling strategy ('warn', 'ignore', 'raise'). This parameter is considered
        only when no values or one value is provided.

    Returns
    -------
    adjusted_values : tuple
        A tuple of adjusted values, aiming to distribute the total samples evenly.
        If only one value is given, the function tries to adjust it based on the
        total number of samples and the initial guess.

    Raises
    ------
    ValueError
        Raised if error is set to 'raise' and no values are provided.

    Examples
    --------
    >>> from gofast.tools.coreutils import adjust_to_samples
    >>> adjust_to_samples(1000, 10, 20, initial_guess=5)
    (50, 20)

    >>> adjust_to_samples(1000, 10, initial_guess=2)
    (2,)

    Notes
    -----
    The function aims to adjust the values to match the desired total number of samples
    as closely as possible. When only one value is given, the function uses the initial
    guess to make an adjustment, respecting the total number of samples.
    """
    if len(values) == 0:
        message = "No values provided for adjustment."
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
        return ()

    if len(values) == 1:
        # If only one value is given, adjust it based on initial guess and n_samples
        single_value = values[0]
        adjusted_value = n_samples // single_value if initial_guess is None else initial_guess
        return (adjusted_value,)

    if initial_guess is None:
        initial_guess = np.mean(values)

    # Function to minimize: difference between product of adjusted values and n_samples
    def objective(factors):
        prod = np.prod(np.array(values) * factors)
        return abs(prod - n_samples)

    # Start with initial guesses for factors
    factors_initial = [initial_guess / value for value in values]
    result = optimize_minimize(objective, factors_initial, bounds=[(0, None) for _ in values])

    if result.success:
        adjusted_values = ( 
            tuple(max(1, int(round(value * factor))) 
                  for value, factor in zip(values, result.x))
            )
    else:
        adjusted_values = values  # Fallback to original values if optimization fails

    return adjusted_values

def unpack_list_of_dicts(list_of_dicts):
    """
    Unpacks a list of dictionaries into a single dictionary,
    merging all keys and values.

    Parameters:
    ----------
    list_of_dicts : list of dicts
        A list where each element is a dictionary with a single key-value pair, 
        the value being a list.

    Returns:
    -------
    dict
        A single dictionary with all keys from the original list of dictionaries, 
        each associated with its combined list of values from all occurrences 
        of the key.

    Example:
    --------
    >>> from gofast.tools.coreutils import unpack_list_of_dicts
    >>> list_of_dicts = [
            {'key1': ['value10', 'value11']},
            {'key2': ['value20', 'value21']},
            {'key1': ['value12']},
            {'key2': ['value22']}
        ]
    >>> unpacked_dict = unpack_list_of_dicts(list_of_dicts)
    >>> print(unpacked_dict)
    {'key1': ['value10', 'value11', 'value12'], 'key2': ['value20', 'value21', 'value22']}
    """
    unpacked_dict = defaultdict(list)
    for single_dict in list_of_dicts:
        for key, values in single_dict.items():
            unpacked_dict[key].extend(values)
    return dict(unpacked_dict)  # Convert defaultdict back to dict if required

def get_params (obj: object ) -> dict: 
    """
    Get object parameters. 
    
    Object can be callable or instances 
    
    :param obj: object , can be callable or instance 
    
    :return: dict of parameters values 
    
    :examples: 
    >>> from sklearn.svm import SVC 
    >>> from gofast.tools.coreutils import get_params 
    >>> sigmoid= SVC (
        **{
            'C': 512.0,
            'coef0': 0,
            'degree': 1,
            'gamma': 0.001953125,
            'kernel': 'sigmoid',
            'tol': 1.0 
            }
        )
    >>> pvalues = get_params( sigmoid)
    >>> {'decision_function_shape': 'ovr',
         'break_ties': False,
         'kernel': 'sigmoid',
         'degree': 1,
         'gamma': 0.001953125,
         'coef0': 0,
         'tol': 1.0,
         'C': 512.0,
         'nu': 0.0,
         'epsilon': 0.0,
         'shrinking': True,
         'probability': False,
         'cache_size': 200,
         'class_weight': None,
         'verbose': False,
         'max_iter': -1,
         'random_state': None
     }
    """
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not (k.endswith('_') or k.startswith('_'))}
    
    return PARAMS_VALUES

def is_classification_task(
    *y, max_unique_values=10
    ):
    """
    Check whether the given arrays are for a classification task.

    This function assumes that if all values in the provided arrays are 
    integers and the number of unique values is within the specified
    threshold, it is a classification task.

    Parameters
    ----------
    *y : list or numpy.array
        A variable number of arrays representing actual values, 
        predicted values, etc.
    max_unique_values : int, optional
        The maximum number of unique values to consider the task 
        as classification. 
        Default is 10.

    Returns
    -------
    bool
        True if the provided arrays are for a classification task, 
        False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import is_classification_task 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> is_classification_task(y_true, y_pred)
    True
    """
    max_unique_values = int (
        _assert_all_types(max_unique_values, 
                          int, float, objname="Max Unique values")
                             )
    # Combine all arrays for analysis
    combined = np.concatenate(y)

    # Check if all elements are integers
    if ( 
            not all(isinstance(x, int) for x in combined) 
            and not combined.dtype.kind in 'iu'
            ):
        return False

    # Check the number of unique elements
    unique_values = np.unique(combined)
    # check Arbitrary threshold for number of classes
    if len(unique_values) > max_unique_values:
        return False

    return True

def fancy_printer(result, report_name='Data Quality Check Report'):
    """ 
    This _fancy_print function within the check_data_quality function 
    iterates over the results dictionary and prints each category 
    (like missing data, outliers, etc.) in a formatted manner. It only 
    displays categories with findings, making the output more concise and 
    focused on the areas that need attention. The use of .title() 
    and .replace('_', ' ') methods enhances the readability of the 
    category names.

    Parameters 
    -----------
    result: dict,
       the result to print. Must contain a dictionnary. 
    report_name: str, 
       A report to fancy printer. 
       
    """
    if not isinstance ( result, dict): 
        raise TypeError("fancy_printer accepts only a dictionnary type."
                        f" Got {type(result).__name__!r}")
        
    print(f"\n{report_name}:\n")

    for key, value in result.items():
        if value:  # Only display categories with findings
            print(f"--- {key.replace('_', ' ').title()} ---")
            print("Column            | Details")
            print("-" * 40)  # Table header separator

            try : 
                
                for sub_key, sub_value in value.items():
                    # Ensuring column name and details fit into the table format
                    formatted_key = (sub_key[:15] + '..') if len(
                        sub_key) > 17 else sub_key
                    formatted_value = str(sub_value)[:20] + (
                        '..' if len(str(sub_value)) > 22 else '')
                    print(f"{formatted_key:<17} | {formatted_value}")
            except : 
                formatted_key = (key[:15] + '..') if len(key) > 17 else key
                formatted_value = f"{value:.2f}"
                print(f"{formatted_key:<17} | {formatted_value}")

            print("\n")
        else:
            print(f"--- No {key.replace('_', ' ').title()} Found ---\n")

def to_numeric_dtypes(
    arr: Union[NDArray, DataFrame], *, 
    columns: Optional[List[str]] = None, 
    return_feature_types: bool = ..., 
    missing_values: float = np.nan, 
    pop_cat_features: bool = ..., 
    sanitize_columns: bool = ..., 
    regex: Optional[re.Pattern] = None, 
    fill_pattern: str = '_', 
    drop_nan_columns: bool = True, 
    how: str = 'all', 
    reset_index: bool = ..., 
    drop_index: bool = True, 
    verbose: bool = ...
) -> Union[DataFrame, Tuple[DataFrame, List[str], List[str]]]:
    """
    Converts an array to a DataFrame and coerces values to appropriate 
    data types.

    This function is designed to process data arrays or DataFrames, ensuring
    numeric and categorical features are correctly identified and formatted. 
    It provides options to manipulate the data, including column sanitization, 
    handling of missing values, and dropping NaN-filled columns.

    Parameters
    ----------
    arr : NDArray or DataFrame
        The data to be processed, either as an array or a DataFrame.
    
    columns : list of str, optional
        Column names for creating a DataFrame from an array. 
        Length should match the number of columns in `arr`.
    
    return_feature_types : bool, default=False
        If True, returns a tuple with the DataFrame, numeric, and categorical 
        features.
    
    missing_values : float, default=np.nan
        Value used to replace missing or empty strings in the DataFrame.
    
    pop_cat_features : bool, default=False
        If True, removes categorical features from the DataFrame.
    
    sanitize_columns : bool, default=False
        If True, cleans the DataFrame columns using the specified `regex` 
        pattern.
    
    regex : re.Pattern or str, optional
        Regular expression pattern for column sanitization. the default is:: 
        
        >>> import re 
        >>> re.compile (r'[_#&.)(*@!_,;\s-]\s*', flags=re.IGNORECASE)
    
    fill_pattern : str, default='_'
        String pattern used to replace non-alphanumeric characters in 
        column names.
    
    drop_nan_columns : bool, default=True
        If True, drops columns filled entirely with NaN values.
    
    how : str, default='all'
        Determines row dropping strategy based on NaN values.
    
    reset_index : bool, default=False
        If True, resets the index of the DataFrame after processing.
    
    drop_index : bool, default=True
        If True, drops the original index when resetting the DataFrame index.
    
    verbose : bool, default=False
        If True, prints additional information during processing.

    Returns
    -------
    DataFrame or tuple of DataFrame, List[str], List[str]
        The processed DataFrame. If `return_feature_types` is True, returns a 
        tuple with the DataFrame, list of numeric feature names (`nf`), 
        and list of categorical feature names (`cf`).

    Examples
    --------
    >>> from gofast.datasets.dload import load_bagoue
    >>> from gofast.tools.coreutils import to_numeric_dtypes
    >>> X= load_bagoue(as_frame=True)
    >>> X0 = X[['shape', 'power', 'magnitude']]
    >>> df, nf, cf = to_numeric_dtypes(X0, return_feature_types=True)
    >>> print(df.dtypes, nf, cf)
    >>> X0.dtypes 
    ... shape        object
        power        object
        magnitude    object
        dtype: object
    >>> df = to_numeric_dtypes(X0)
    >>> df.dtypes 
    ... shape         object
        power        float64
        magnitude    float64
        dtype: object
    """

    from .validator import _is_numeric_dtype
    # pass ellipsis argument to False 
    ( sanitize_columns, reset_index, 
     verbose,return_feature_types, 
     pop_cat_features, 
        ) = ellipsis2false(
            sanitize_columns, 
            reset_index, 
            verbose,
            return_feature_types, 
            pop_cat_features
    )
   
    if not is_iterable (arr, exclude_string=True): 
        raise TypeError(f"Expect array. Got {type (arr).__name__!r}")

    if hasattr ( arr, '__array__') and hasattr ( arr, 'columns'): 
        df = arr.copy()
        if columns is not None: 
            if verbose: 
                print("Dataframe is passed. Columns should be replaced.")
            df =pd.DataFrame ( np.array ( arr), columns =columns )
            
    else: df = pd.DataFrame (arr, columns =columns  ) 
        
    # sanitize columns 
    if sanitize_columns: 
        # Pass in the case columns are all integer values. 
        if not _is_numeric_dtype(df.columns , to_array=True): 
           # for consistency reconvert to str 
           df.columns = df.columns.astype(str) 
           df = sanitize_frame_cols(
               df, regex=regex, fill_pattern=fill_pattern ) 

    #replace empty string by Nan if NaN exist in dataframe  
    df= df.replace(r'^\s*$', missing_values, regex=True)
    
    # check the possibililty to cast all 
    # the numerical data 
    for serie in df.columns: 
        try: 
            df= df.astype(
                {serie:np.float64})
        except:continue
    
    # drop nan  columns if exists 
    if drop_nan_columns: 
        if verbose: 
            nan_columns = df.columns [ df.isna().all()].tolist() 
            print("No NaN column found.") if len(
                nan_columns)==0 else listing_items_format (nan_columns, 
                    "NaN columns found in the data",
                    " ", inline =True, lstyle='.')                               
        # drop rows and columns with NaN values everywhere.                                                   
        df.dropna ( axis=1, how='all', inplace =True)
        if str(how).lower()=='all': 
            df.dropna ( axis=0, how='all', inplace =True)
    
    # reset_index of the dataframe
    # This is useful after droping rows
    if reset_index: 
        df.reset_index (inplace =True, drop = drop_index )
    # collect numeric and non-numeric data 
    nf, cf =[], []    
    for serie in df.columns: 
        if _is_numeric_dtype(df[serie], to_array =True ): 
            nf.append(serie)
        else: cf.append(serie)

    if pop_cat_features: 
        [ df.pop(item) for item in cf ] 
        if verbose: 
            msg ="Dataframe does not contain any categorial features."
            b= f"Feature{'s' if len(cf)>1 else ''}"
            e = (f"{'have' if len(cf) >1 else 'has'} been dropped"
                 " from the dataframe.")
            print(msg) if len(cf)==0 else listing_items_format (
                cf , b, e ,lstyle ='.', inline=True)
            
        return df 
    
    return (df, nf, cf) if return_feature_types else df 

def listing_items_format ( 
        lst,  begintext ='', endtext='' , bullet='-', 
        enum =True , lstyle=None , space =3 , inline =False, verbose=True
        ): 
    """ Format list by enumerate them successively with carriage return
    
    :param lst: list,
        object for listening 
    :param begintext: str, 
        Text to display at the beginning of listing the items in `lst`. 
    :param endtext: str, 
        Text to display at the end of the listing items in `lst`. 
    :param enum:bool, default=True, 
        Count the number of items in `lst` and display it 
    :param lstyle: str, default =None 
        listing marker. 
    :param bullet:str, default='-'
        symbol that is used to introduce item if `enum` is set to False. 
    :param space: int, 
        number of space to keep before each outputted item in `lst`
    :param inline: bool, default=False, 
        Display all element inline rather than carriage return every times. 
    :param verbose: bool, 
        Always True for print. If set to False, return list of string 
        litteral text. 
    :returns: None or str 
        None or string litteral if verbose is set to ``False``.
    Examples
    ---------
    >>> from gofast.tools.coreutils import listing_items_format 
    >>> litems = ['hole_number', 'depth_top', 'depth_bottom', 'strata_name', 
                'rock_name','thickness', 'resistivity', 'gamma_gamma', 
                'natural_gamma', 'sp','short_distance_gamma', 'well_diameter']
    >>> listing_items_format (litems , 'Features' , 
                               'have been successfully drop.' , 
                              lstyle ='.', space=3) 
    """
    out =''
    if not is_iterable(lst): 
        lst=[lst]
   
    if hasattr (lst, '__array__'): 
        if lst.ndim !=1: 
            raise ValueError (" Can not print multidimensional array."
                              " Expect one dimensional array.")
    lst = list(lst)
    begintext = str(begintext); endtext=str(endtext)
    lstyle=  lstyle or bullet  
    lstyle = str(lstyle)
    b= f"{begintext +':' } "   
    if verbose :
        print(b, end=' ') if inline else (
            print(b)  if  begintext!='' else None)
    out += b +  ('\n' if not inline else ' ') 
    for k, item in enumerate (lst): 
        sp = ' ' * space 
        if ( not enum and inline ): lstyle =''
        o = f"{sp}{str(k+1) if enum else bullet+ ' ' }{lstyle} {item}"
        if verbose:
            print (o , end=' ') if inline else print(o)
        out += o + ('\n' if not inline else ' ') 
       
    en= ' ' + endtext if inline else endtext
    if verbose: 
        print(en) if endtext !='' else None 
    out +=en 
    
    return None if verbose else out 
    
    
def parse_attrs (attr,  regex=None ): 
    """ Parse attributes using the regular expression.
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    
    attr: str, text litteral containing the attributes 
        names 
        
    regex: `re` object, default is 
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'per|mod|times|add|sub|[_#&*@!_,;\s-]\s*', 
                                flags=re.IGNORECASE) 
    Returns
    -------
    attr: List of attributes 
    
    Example
    ---------
    >>> from gofast.tools.coreutils import parse_attrs 
    >>> parse_attrs('lwi_sub_ohmSmulmagnitude')
    ... ['lwi', 'ohmS', 'magnitude']
    
    
    """
    regex = regex or re.compile (r'per|mod|times|add|sub|[_#&*@!_,;\s-]\s*', 
                        flags=re.IGNORECASE) 
    attr= list(filter (None, regex.split(attr)))
    return attr 
    
def url_checker (url: str , install:bool = False, 
                 raises:str ='ignore')-> bool : 
    """
    check whether the URL is reachable or not. 
    
    function uses the requests library. If not install, set the `install`  
    parameter to ``True`` to subprocess install it. 
    
    Parameters 
    ------------
    url: str, 
        link to the url for checker whether it is reachable 
    install: bool, 
        Action to install the 'requests' module if module is not install yet.
    raises: str 
        raise errors when url is not recheable rather than returning ``0``.
        if `raises` is ``ignore``, and module 'requests' is not installed, it 
        will use the django url validator. However, the latter only assert 
        whether url is right but not validate its reachability. 
              
    Returns
    --------
        ``True``{1} for reacheable and ``False``{0} otherwise. 
        
    Example
    ----------
    >>> from gofast.tools.coreutils import url_checker 
    >>> url_checker ("http://www.example.com")
    ...  0 # not reacheable 
    >>> url_checker ("https://gofast.readthedocs.io/en/latest/api/gofast.html")
    ... 1 
    
    """
    isr =0 ; success = False 
    
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        #domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
    
    try : 
        import requests 
    except ImportError: 
        if install: 
            success  = is_installing('requests', DEVNULL=True) 
        if not success: 
            if raises=='raises': 
                raise ModuleNotFoundError(
                    "auto-installation of 'requests' failed."
                    " Install it mannually.")
                
    else : success=True  
    
    if success: 
        try:
            get = requests.get(url) #Get Url
            if get.status_code == 200: # if the request succeeds 
                isr =1 # (f"{url}: is reachable")
                
            else:
                warnings.warn(
                    f"{url}: is not reachable, status_code: {get.status_code}")
                isr =0 
        
        except requests.exceptions.RequestException as e:
            if raises=='raises': 
                raise SystemExit(f"{url}: is not reachable \nErr: {e}")
            else: isr =0 
            
    if not success : 
        # use django url validation regex
        # https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45
        isr = 1 if re.match(regex, url) is not None else 0 
        
    return isr 

def shrunkformat(
    text: Union[str, Iterable[Any]], 
    chunksize: int = 7,
    insert_at: Optional[str] = None, 
    sep: Optional[str] = None, 
) -> None:
    """ Format class and add ellipsis when classes are greater than maxview 
    
    :param text: str - a text to shrunk and format. Can also be an iterable
        object. 
    :param chunksize: int, the size limit to keep in the formatage text. *default* 
        is ``7``.
    :param insert_at: str, the place to insert the ellipsis. If ``None``,  
        shrunk the text and put the ellipsis, between the text beginning and 
        the text endpoint. Can be ``beginning``, or ``end``. 
    :param sep: str if the text is delimited by a kind of character, the `sep` 
        parameters could be usefull so it would become a starting point for 
        word counting. *default*  is `None` which means word is counting from 
        the space. 
        
    :example: 
        
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import shrunkformat
    >>> text=" I'm a long text and I will be shrunked and replaced by ellipsis."
    >>> shrunkformat (text)
    ... 'Im a long ... and replaced by ellipsis.'
    >>> shrunkformat (text, insert_at ='end')
    ...'Im a long ... '
    >>> arr = np.arange(30)
    >>> shrunkformat (arr, chunksize=10 )
    ... '0 1 2 3 4  ...  25 26 27 28 29'
    >>> shrunkformat (arr, insert_at ='begin')
    ... ' ...  26 27 28 29'
    
    """
    is_str = False 
    chunksize = int (_assert_all_types(chunksize, float, int))
                   
    regex = re.compile (r"(begin|start|beg)|(end|close|last)")
    insert_at = str(insert_at).lower().strip() 
    gp = regex.search (insert_at) 
    if gp is not None: 
        if gp.group (1) is not None:  
            insert_at ='begin'
        elif gp.group(2) is not None: 
            insert_at ='end'
        if insert_at is None: 
            warnings.warn(f"Expect ['begining'|'end'], got {insert_at!r}"
                          " Default value is used instead.")
    if isinstance(text , str): 
        textsplt = text.strip().split(sep) # put text on list 
        is_str =True 
        
    elif hasattr (text , '__iter__'): 
        textsplt = list(text )
        
    if len(textsplt) < chunksize : 
        return  text 
    
    if is_str : 
        rl = textsplt [:len(textsplt)//2][: chunksize//2]
        ll= textsplt [len(textsplt)//2:][-chunksize//2:]
        
        if sep is None: sep =' '
        spllst = [f'{sep}'.join ( rl), f'{sep}'.join ( ll)]
        
    else : spllst = [
        textsplt[: chunksize//2 ] ,textsplt[-chunksize//2:]
        ]
    if insert_at =='begin': 
        spllst.insert(0, ' ... ') ; spllst.pop(1)
    elif insert_at =='end': 
        spllst.pop(-1) ; spllst.extend ([' ... '])
        
    else : 
        spllst.insert (1, ' ... ')
    
    spllst = spllst if is_str else str(spllst)
    
    return re.sub(r"[\[,'\]]", '', ''.join(spllst), 
                  flags=re.IGNORECASE 
                  ) 
    
def is_installing (
    module: str , 
    upgrade: bool=True , 
    action: bool=True, 
    DEVNULL: bool=False,
    verbose: int=0,
    **subpkws
    )-> bool: 
    """ Install or uninstall a module/package using the subprocess 
    under the hood.
    
    Parameters 
    ------------
    module: str,
        the module or library name to install using Python Index Package `PIP`
    
    upgrade: bool,
        install the lastest version of the package. *default* is ``True``.   
        
    DEVNULL:bool, 
        decline the stdoutput the message in the console 
    
    action: str,bool 
        Action to perform. 'install' or 'uninstall' a package. *default* is 
        ``True`` which means 'intall'. 
        
    verbose: int, Optional
        Control the verbosity i.e output a message. High level 
        means more messages. *default* is ``0``.
         
    subpkws: dict, 
        additional subprocess keywords arguments 
    Returns 
    ---------
    success: bool 
        whether the package is sucessfully installed or not. 
        
    Example
    --------
    >>> from gofast import is_installing
    >>> is_installing(
        'tqdm', action ='install', DEVNULL=True, verbose =1)
    >>> is_installing(
        'tqdm', action ='uninstall', verbose =1)
    """
    #implement pip as subprocess 
    # refer to https://pythongeeks.org/subprocess-in-python/
    if not action: 
        if verbose > 0 :
            print("---> No action `install`or `uninstall`"
                  f" of the module {module!r} performed.")
        return action  # DO NOTHING 
    
    success=False 

    action_msg ='uninstallation' if action =='uninstall' else 'installation' 

    if action in ('install', 'uninstall', True) and verbose > 0:
        print(f'---> Module {module!r} {action_msg} will take a while,'
              ' please be patient...')
        
    cmdg =f'<pip install {module}> | <python -m pip install {module}>'\
        if action in (True, 'install') else ''.join([
            f'<pip uninstall {module} -y> or <pip3 uninstall {module} -y ',
            f'or <python -m pip uninstall {module} -y>.'])
        
    upgrade ='--upgrade' if upgrade else '' 
    
    if action == 'uninstall':
        upgrade= '-y' # Don't ask for confirmation of uninstall deletions.
    elif action in ('install', True):
        action = 'install'

    cmd = ['-m', 'pip', f'{action}', f'{module}', f'{upgrade}']

    try: 
        STDOUT = subprocess.DEVNULL if DEVNULL else None 
        STDERR= subprocess.STDOUT if DEVNULL else None 
    
        subprocess.check_call(
            [sys.executable] + cmd, stdout= STDOUT, stderr=STDERR,
                              **subpkws)
        if action in (True, 'install'):
            # freeze the dependancies
            reqs = subprocess.check_output(
                [sys.executable,'-m', 'pip','freeze'])
            [r.decode().split('==')[0] for r in reqs.split()]

        success=True
        
    except: 

        if verbose > 0 : 
            print(f'---> Module {module!r} {action_msg} failed. Please use'
                f' the following command: {cmdg} to manually do it.')
    else : 
        if verbose > 0: 
            print(f"{action_msg.capitalize()} of `{module}` "
                      "and dependancies was successfully done!") 
        
    return success 

def smart_strobj_recognition(
        name: str  ,
        container: Union [List , Tuple , Dict[Any, Any ]],
        stripitems: Union [str , List , Tuple] = '_', 
        deep: bool = False,  
) -> str : 
    """ Find the likelihood word in the whole containers and 
    returns the value.
    
    :param name: str - Value of to search. I can not match the exact word in 
    the `container`
    :param container: list, tuple, dict- container of the many string words. 
    :param stripitems: str - 'str' items values to sanitize the  content 
        element of the dummy containers. if different items are provided, they 
        can be separated by ``:``, ``,`` and ``;``. The items separators 
        aforementioned can not  be used as a component in the `name`. For 
        isntance:: 
            
            name= 'dipole_'; stripitems='_' -> means remove the '_'
            under the ``dipole_``
            name= '+dipole__'; stripitems ='+;__'-> means remove the '+' and
            '__' under the value `name`. 
        
    :param deep: bool - Kind of research. Go deeper by looping each items 
         for find the initials that can fit the name. Note that, if given, 
         the first occurence should be consider as the best name... 
         
    :return: Likelihood object from `container`  or Nonetype if none object is
        detected.
        
    :Example:
        >>> from gofast.tools.coreutils import smart_strobj_recognition
        >>> from gofast.methods import ResistivityProfiling 
        >>> rObj = ResistivityProfiling(AB= 200, MN= 20,)
        >>> smart_strobj_recognition ('dip', robj.__dict__))
        ... None 
        >>> smart_strobj_recognition ('dipole_', robj.__dict__))
        ... dipole 
        >>> smart_strobj_recognition ('dip', robj.__dict__,deep=True )
        ... dipole 
        >>> smart_strobj_recognition (
            '+_dipole___', robj.__dict__,deep=True , stripitems ='+;_')
        ... 'dipole'
        
    """

    stripitems =_assert_all_types(stripitems , str, list, tuple) 
    container = _assert_all_types(container, list, tuple, dict)
    ix , rv = None , None 
    
    if isinstance (stripitems , str): 
        for sep in (':', ",", ";"): # when strip ='a,b,c' seperated object
            if sep in stripitems:
                stripitems = stripitems.strip().split(sep) ; break
        if isinstance(stripitems, str): 
            stripitems =[stripitems]
            
    # sanitize the name. 
    for s in stripitems :
        name = name.strip(s)     
        
    if isinstance(container, dict) : 
        #get only the key values and lower them 
        container_ = list(map (lambda x :x.lower(), container.keys())) 
    else :
        # for consistency put on list if values are in tuple. 
        container_ = list(container)
        
    # sanitize our dummny container item ... 
    #container_ = [it.strip(s) for it in container_ for s in stripitems ]
    if name.lower() in container_: 
        try:
            ix = container_.index (name)
        except ValueError: 
            raise AttributeError(f"{name!r} attribute is not defined")
        
    if deep and ix is None:
        # go deeper in the search... 
        for ii, n in enumerate (container_) : 
            if n.find(name.lower())>=0 : 
                ix =ii ; break 
    
    if ix is not None: 
        if isinstance(container, dict): 
            rv= list(container.keys())[ix] 
        else : rv= container[ix] 

    return  rv 

def repr_callable_obj(obj: _F  , skip = None ): 
    """ Represent callable objects. 
    
    Format class, function and instances objects. 
    
    :param obj: class, func or instances
        object to format. 
    :param skip: str , 
        attribute name that is not end with '_' and whom it needs to be 
        skipped. 
        
    :Raises: TypeError - If object is not a callable or instanciated. 
    
    :Examples: 
        
    >>> from gofast.tools.coreutils import repr_callable_obj
    >>> from gofast.methods.electrical import  ResistivityProfiling
    >>> repr_callable_obj(ResistivityProfiling)
    ... 'ResistivityProfiling(station= None, dipole= 10.0, 
            auto_station= False, kws= None)'
    >>> robj= ResistivityProfiling (AB=200, MN=20, station ='S07')
    >>> repr_callable_obj(robj)
    ... 'ResistivityProfiling(AB= 200, MN= 20, arrangememt= schlumberger, ... ,
        dipole= 10.0, station= S07, auto= False)'
    >>> repr_callable_obj(robj.fit)
    ... 'fit(data= None, kws= None)'
    
    """
    regex = re.compile (r"[{'}]")
    
    # inspect.formatargspec(*inspect.getfullargspec(cls_or_func))
    if not hasattr (obj, '__call__') and not hasattr(obj, '__dict__'): 
        raise TypeError (
            f'Format only callabe objects: Got {type (obj).__name__!r}')
        
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        objname = obj.__name__
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        objname=obj.__class__.__name__
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not ((k.endswith('_') or k.startswith('_') 
                                  # remove the dict objects
                                  or k.endswith('_kws') or k.endswith('_props'))
                                 )
                         }
    if skip is not None : 
        # skip some inner params 
        # remove them as the main function or class params 
        if isinstance(skip, (tuple, list, np.ndarray)): 
            skip = list(map(str, skip ))
            exs = [key for key in PARAMS_VALUES.keys() if key in skip]
        else:
            skip =str(skip).strip() 
            exs = [key for key in PARAMS_VALUES.keys() if key.find(skip)>=0]
 
        for d in exs: 
            PARAMS_VALUES.pop(d, None) 
            
    # use ellipsis as internal to stdout more than seven params items 
    if len(PARAMS_VALUES) >= 7 : 
        f = {k:PARAMS_VALUES.get(k) for k in list(PARAMS_VALUES.keys())[:3]}
        e = {k:PARAMS_VALUES.get(k) for k in list(PARAMS_VALUES.keys())[-3:]}
        
        PARAMS_VALUES= str(f) + ' ... ' + str(e )

    return str(objname) + '(' + regex.sub('', str (PARAMS_VALUES)
                                          ).replace(':', '=') +')'


def accept_types (
        *objtypes: list , 
        format: bool = False
        ) -> Union [List[str] , str] : 
    """ List the type format that can be accepted by a function. 
    
    :param objtypes: List of object types.
    :param format: bool - format the list of the name of objects.
    :return: list of object type names or str of object names. 
    
    :Example: 
        >>> import numpy as np; import pandas as pd 
        >>> from gofast.tools.coreutils import accept_types
        >>> accept_types (pd.Series, pd.DataFrame, tuple, list, str)
        ... "'Series','DataFrame','tuple','list' and 'str'"
        >>> atypes= accept_types (
            pd.Series, pd.DataFrame,np.ndarray, format=True )
        ..."'Series','DataFrame' and 'ndarray'"
    """
    return smart_format(
        [f'{o.__name__}' for o in objtypes]
        ) if format else [f'{o.__name__}' for o in objtypes] 

def read_from_excelsheets(erp_file: str = None ) -> List[DataFrame]: 
    
    """ Read all Excelsheets and build a list of dataframe of all sheets.
   
    :param erp_file:
        Excell workbooks containing `erp` profile data.
        
    :return: A list composed of the name of `erp_file` at index =0 and the 
      datataframes.
      
    """
    
    allfls:Dict [str, Dict [_T, List[_T]] ] = pd.read_excel(
        erp_file, sheet_name=None)
    
    list_of_df =[os.path.basename(os.path.splitext(erp_file)[0])]
    for sheets , values in allfls.items(): 
        list_of_df.append(pd.DataFrame(values))

    return list_of_df 

def check_dimensionality(obj, data, z, x):
    """ Check dimensionality of data and fix it.
    
    :param obj: Object, can be a class logged or else.
    :param data: 2D grid data of ndarray (z, x) dimensions.
    :param z: array-like should be reduced along the row axis.
    :param x: arraylike should be reduced along the columns axis.
    
    """
    def reduce_shape(Xshape, x, axis_name=None): 
        """ Reduce shape to keep the same shape"""
        mess ="`{0}` shape({1}) {2} than the data shape `{0}` = ({3})."
        ox = len(x) 
        dsh = Xshape 
        if len(x) > Xshape : 
            x = x[: int (Xshape)]
            obj._logging.debug(''.join([
                f"Resize {axis_name!r}={ox!r} to {Xshape!r}.", 
                mess.format(axis_name, len(x),'more',Xshape)])) 
                                    
        elif len(x) < Xshape: 
            Xshape = len(x)
            obj._logging.debug(''.join([
                f"Resize {axis_name!r}={dsh!r} to {Xshape!r}.",
                mess.format(axis_name, len(x),'less', Xshape)]))
        return int(Xshape), x 
    
    sz0, z = reduce_shape(data.shape[0], 
                          x=z, axis_name ='Z')
    sx0, x =reduce_shape (data.shape[1],
                          x=x, axis_name ='X')
    data = data [:sz0, :sx0]
    
    return data , z, x 

def smart_format(iter_obj, choice ='and'): 
    """ Smart format iterable object.
    
    :param iter_obj: iterable obj 
    :param choice: can be 'and' or 'or' for optional.
    
    :Example: 
        >>> from gofast.tools.coreutils import smart_format
        >>> smart_format(['model', 'iter', 'mesh', 'data'])
        ... 'model','iter','mesh' and 'data'
    """
    str_litteral =''
    try: 
        iter(iter_obj) 
    except:  return f"{iter_obj}"
    
    iter_obj = [str(obj) for obj in iter_obj]
    if len(iter_obj) ==1: 
        str_litteral= ','.join([f"{i!r}" for i in iter_obj ])
    elif len(iter_obj)>1: 
        str_litteral = ','.join([f"{i!r}" for i in iter_obj[:-1]])
        str_litteral += f" {choice} {iter_obj[-1]!r}"
    return str_litteral

def make_introspection(Obj: object , subObj: _Sub[object])->None: 
    """ Make introspection by using the attributes of instance created to 
    populate the new classes created.
    
    :param Obj: callable 
        New object to fully inherits of `subObject` attributes.
        
    :param subObj: Callable 
        Instance created.
    """
    # make introspection and set the all  attributes to self object.
    # if Obj attribute has the same name with subObj attribute, then 
    # Obj attributes get the priority.
    for key, value in  subObj.__dict__.items(): 
        if not hasattr(Obj, key) and key  != ''.join(['__', str(key), '__']):
            setattr(Obj, key, value)
  
def cpath(savepath: str = None, dpath: str = '_default_path_') -> str:
    """
    Ensure a directory exists for saving files. If the specified savepath 
    does not exist, it will be created. If no savepath is specified, a default
    directory is used.

    Parameters:
    - savepath (str, optional): The target directory to validate or create. 
      If None, dpath is used.
    - dpath (str): The default directory to use if savepath is None. Created 
      in the current working directory.

    Returns:
    - str: The absolute path to the validated or created directory.

    Example:
    ```
    # Using the default path
    default_path = cpath()
    print(f"Files will be saved to: {default_path}")

    # Specifying a custom path
    custom_path = cpath('/path/to/save')
    print(f"Files will be saved to: {custom_path}")
    ```
    """
    from pathlib import Path
    if savepath is None:
        savepath = Path.cwd() / dpath
    else:
        savepath = Path(savepath)

    try:
        savepath.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {savepath}: {e}")
        # Optionally, handle errors more gracefully or raise for critical issues
    return str(savepath.resolve())

  
def sPath (name_of_path:str):
    """ Savepath func. Create a path  with `name_of_path` if path not exists.
    
    :param name_of_path: str, Path-like object. If path does not exist,
        `name_of_path` should be created.
    """
    
    try :
        savepath = os.path.join(os.getcwd(), name_of_path)
        if not os.path.isdir(savepath):
            os.mkdir(name_of_path)#  mode =0o666)
    except :
        warnings.warn("The path seems to be existed!")
        return
    return savepath 


def format_notes(text:str , cover_str: str ='~', inline=70, **kws): 
    """ Format note 
    :param text: Text to be formated.
    
    :param cover_str: type of ``str`` to surround the text.
    
    :param inline: Nomber of character before going in liine.
    
    :param margin_space: Must be <1 and expressed in %. The empty distance 
        between the first index to the inline text 
    :Example: 
        
        >>> from gofast.tools import funcutils as func 
        >>> text ='Automatic Option is set to ``True``.'\
            ' Composite estimator building is triggered.' 
        >>>  func.format_notes(text= text ,
        ...                       inline = 70, margin_space = 0.05)
    
    """
    
    headnotes =kws.pop('headernotes', 'notes')
    margin_ratio = kws.pop('margin_space', 0.2 )
    margin = int(margin_ratio * inline)
    init_=0 
    new_textList= []
    if len(text) <= (inline - margin): 
        new_textList = text 
    else : 
        for kk, char in enumerate (text): 
            if kk % (inline - margin)==0 and kk !=0: 
                new_textList.append(text[init_:kk])
                init_ =kk 
            if kk ==  len(text)-1: 
                new_textList.append(text[init_:])
  
    print('!', headnotes.upper(), ':')
    print('{}'.format(cover_str * inline)) 
    for k in new_textList:
        fmtin_str ='{'+ '0:>{}'.format(margin) +'}'
        print('{0}{1:>2}{2:<51}'.format(fmtin_str.format(cover_str), '', k))
        
    print('{0}{1:>51}'.format(' '* (margin -1), cover_str * (inline -margin+1 ))) 
    

def interpol_scipy(
        x_value,
        y_value,
        x_new,
        kind="linear",
        plot=False,
        fill_value="extrapolate"
):
    """
    Function to interpolate data using scipy's interp1d if available.
    
    Parameters 
    ------------
    * x_value : np.ndarray 
        Original abscissa values.
                
    * y_value : np.ndarray 
        Original ordinate values (slope).
                
    * x_new : np.ndarray 
        New abscissa values for which you want to interpolate data.
                
    * kind : str 
        Type of interpolation, e.g., "linear", "cubic".
                
    * fill_value : str 
        Extrapolation method. If None, scipy's interp1d will use constrained 
        interpolation. 
        Can be "extrapolate" to use fill_value.
        
    * plot : bool 
        Set to True to plot a graph of the original and interpolated data.

    Returns 
    --------
    np.ndarray 
        Interpolated ordinate values for 'x_new'.
    """

    spi = check_scipy_interpolate()
    if spi is None:
        return None
    
    try:
        func_ = spi.interp1d(x_value, y_value, kind=kind, fill_value=fill_value)
        y_new = func_(x_new)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(x_value, y_value, "o", x_new, y_new, "--")
            plt.legend(["Data", kind.capitalize()], loc="best")
            plt.title(f"Interpolation: {kind.capitalize()}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()

        return y_new

    except Exception as e:
        _logger.error(f"An unexpected error occurred during interpolation: {e}")
        return None
    
def _remove_str_word (ch, word_to_remove, deep_remove=False):
    """
    Small funnction to remove a word present on  astring character 
    whatever the number of times it will repeated.
    
    Parameters
    ----------
        * ch : str
                may the the str phrases or sentences . main items.
        * word_to_remove : str
                specific word to remove.
        * deep_remove : bool, optional
                use the lower case to remove the word even the word is uppercased 
                of capitalized. The default is False.

    Returns
    -------
        str ; char , new_char without the removed word .
        
    Examples
    ---------
    >>> from gofast.tools import funcutils as func
    >>> ch ='AMTAVG 7.76: "K1.fld", Dated 99-01-01,AMTAVG, 
    ...    Processed 11 Jul 17 AMTAVG'
    >>> ss=func._remove_str_word(char=ch, word_to_remove='AMTAVG', 
    ...                             deep_remove=False)
    >>> print(ss)
    
    """
    if type(ch) is not str : char =str(ch)
    if type(word_to_remove) is not str : word_to_remove=str(word_to_remove)
    
    if deep_remove == True :
        word_to_remove, char =word_to_remove.lower(),char.lower()

    if word_to_remove not in char :
        return char

    while word_to_remove in char : 
        if word_to_remove not in char : 
            break 
        index_wr = char.find(word_to_remove)
        remain_len=index_wr+len(word_to_remove)
        char=char[:index_wr]+char[remain_len:]

    return char

def stn_check_split_type(data_lines): 
    """
    Read data_line and check for data line the presence of 
    split_type < ',' or ' ', or any other marks.>
    Threshold is assume to be third of total data length.
    
    :params data_lines: list of data to parse . 
    :type data_lines: list 
 
    :returns: The split _type
    :rtype: str
    
    :Example: 
        >>> from gofast.tools  import funcutils as func
        >>> path =  data/ K6.stn
        >>> with open (path, 'r', encoding='utf8') as f : 
        ...                     data= f.readlines()
        >>>  print(func.stn_check_split_type(data_lines=data))
        
    """

    split_type =[',', ':',' ',';' ]
    data_to_read =[]
    # change the data if data is not dtype string elements.
    if isinstance(data_lines, np.ndarray): 
        if data_lines.dtype in ['float', 'int', 'complex']: 
            data_lines=data_lines.astype('<U12')
        data_lines= data_lines.tolist()
        
    if isinstance(data_lines, list):
        for ii, item in enumerate(data_lines[:int(len(data_lines)/3)]):
             data_to_read.append(item)
             # be sure the list is str item . 
             data_to_read=[''.join([str(item) for item in data_to_read])] 

    elif isinstance(data_lines, str): data_to_read=[str(data_lines)]
    
    for jj, sep  in enumerate(split_type) :
        if data_to_read[0].find(sep) > 0 :
            if data_to_read[0].count(sep) >= 2 * len(data_lines)/3:
                if sep == ' ': return  None  # use None more conventional 
                else : return sep 


def round_dipole_length(value, round_value =5.): 
    """ 
    small function to graduate dipole length 5 to 5. Goes to be reality and 
    simple computation .
    
    :param value: value of dipole length 
    :type value: float 
    
    :returns: value of dipole length rounded 5 to 5 
    :rtype: float
    """ 
    mm = value % round_value 
    if mm < 3 :return np.around(value - mm)
    elif mm >= 3 and mm < 7 :return np.around(value -mm +round_value) 
    else:return np.around(value - mm +10.)
    
def display_infos(infos, **kws):
    """ Display unique element on list of array infos
    
    :param infos: Iterable object to display. 
    :param header: Change the `header` to other names. 
    
    :Example: 
    >>> from gofast.tools.coreutils import display_infos
    >>> ipts= ['river water', 'fracture zone', 'granite', 'gravel',
         'sedimentary rocks', 'massive sulphide', 'igneous rocks', 
         'gravel', 'sedimentary rocks']
    >>> display_infos('infos= ipts,header='TestAutoRocks', 
                      size =77, inline='~')
    """

    inline =kws.pop('inline', '-')
    size =kws.pop('size', 70)
    header =kws.pop('header', 'Automatic rocks')

    if isinstance(infos, str ): 
        infos =[infos]
        
    infos = list(set(infos))
    print(inline * size )
    mes= '{0}({1:02})'.format(header.capitalize(),
                                  len(infos))
    mes = '{0:^70}'.format(mes)
    print(mes)
    print(inline * size )
    am=''
    for ii in range(len(infos)): 
        if (ii+1) %2 ==0: 
            am = am + '{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            print(am)
            am=''
        else: 
            am ='{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            if ii ==len(infos)-1: 
                print(am)
    print(inline * size )

def fr_en_parser (f, delimiter =':'): 
    """ Parse the translated data file. 
    
    :param f: translation file to parse.
    
    :param delimiter: str, delimiter.
    
    :return: generator obj, composed of a list of 
        french  and english Input translation. 
    
    :Example:
        >>> file_to_parse = 'pme.parserf.md'
        >>> path_pme_data = r'C:/Users\Administrator\Desktop\__elodata
        >>> data =list(BS.fr_en_parser(
            os.path.join(path_pme_data, file_to_parse)))
    """
    
    is_file = os.path.isfile (f)
    if not is_file: 
        raise IOError(f'Input {f} is not a file. Please check your file.')
    
    with open(f, 'r', encoding ='utf8') as ft: 
        data = ft.readlines()
        for row in data :
            if row in ( '\n', ' '):
                continue 
            fr, en = row.strip().split(delimiter)
            yield([fr, en])

def convert_csvdata_from_fr_to_en(csv_fn, pf, destfile = 'pme.en.csv',
                                  savepath =None, delimiter =':'): 
    """ Translate variable data from french csv data  to english with 
    parser file. 
    
    :param csv_fn: data collected in csv format.
    
    :param pf: parser file. 
    
    :param destfile: str,  Destination file, outputfile.
    
    :param savepath: Path-Like object, save data to a path. 
                      
    :Example: 
        # to execute this script, we need to import the two modules below
        >>> import os 
        >>> import csv 
        >>> from gofast.tools.coreutils import convert_csvdata_from_fr_to_en
        >>> path_pme_data = r'C:/Users\Administrator\Desktop\__elodata
        >>> datalist=convert_csvdata_from_fr_to_en(
            os.path.join( path_pme_data, _enuv2.csv') , 
            os.path.join(path_pme_data, pme.parserf.md')
                         savefile = 'pme.en.cv')
    """
    # read the parser file and separed english from french 
    parser_data = list(fr_en_parser (pf,delimiter) )
    
    with open (csv_fn, 'r', encoding ='utf8') as csv_f : 
        csv_reader = csv.reader(csv_f) 
        csv_data =[ row for row in csv_reader]
    # get the index of the last substring row 
    ix = csv_data [0].index ('Industry_type') 
    # separateblock from two 
    csv_1b = [row [:ix +1] for row in csv_data] 
    csv_2b =[row [ix+1:] for row in csv_data ]
    # make a copy of csv_1b
    csv_1bb= copy.deepcopy(csv_1b)
   
    for ii, rowline in enumerate( csv_1bb[3:]) : # skip the first two rows 
        for jj , row in enumerate(rowline): 
            for (fr_v, en_v) in  parser_data: 
                # remove the space from french parser part
                # this could reduce the mistyping error 
                fr_v= fr_v.replace(
                    ' ', '').replace('(', '').replace(
                        ')', '').replace('\\', '').lower()
                 # go  for reading the half of the sentence
                row = row.lower().replace(
                    ' ', '').replace('(', '').replace(
                        ')', '').replace('\\', '')
                if row.find(fr_v[: int(len(fr_v)/2)]) >=0: 
                    csv_1bb[3:][ii][jj] = en_v 
    
    # once translation is done, concatenate list 
    new_csv_list = [r1 + r2 for r1, r2 in zip(csv_1bb,csv_2b )]
    # now write the new scv file 
    if destfile is None: 
        destfile = f'{os.path.basename(csv_fn)}_to.en'
        
    destfile.replace('.csv', '')
    
    with open(f'{destfile}.csv', 'w', newline ='',encoding ='utf8') as csvf: 
        csv_writer = csv.writer(csvf, delimiter=',')
        csv_writer.writerows(new_csv_list)
        # for row in  new_csv_list: 
        #     csv_writer.writerow(row)
    savepath = cpath(savepath , '__pme')
    try :
        shutil.move (f'{destfile}.csv', savepath)
    except:pass 
    
    return new_csv_list
    
def parse_md_data (pf , delimiter =':'): 
    
    if not os.path.isfile (pf): 
        raise IOError( " Unable to detect the parser file. "
                      "Need a Path-like object ")
    
    with open(pf, 'r', encoding ='utf8') as f: 
        pdata = f.readlines () 
    for row in pdata : 
        if row in ('\n', ' '): 
            continue 
        fr, en = row.strip().split(delimiter)
        fr = sanitize_unicode_string(fr)
        en = en.strip()
        # if capilize, the "I" inside the 
        #text should be in lowercase 
        # it is better to upper the first 
        # character after striping the whole 
        # string
        en = list(en)
        en[0] = en[0].upper() 
        en = "".join(en)

        yield fr, en 
        
def sanitize_unicode_string (str_) : 
    """ Replace all spaces and remove all french accents characters.
    
    :Example:
    >>> from gofast.tools.coreutils import sanitize_unicode_string 
    >>> sentence ='Nos clients sont extrêmement satisfaits '
        'de la qualité du service fourni. En outre Nos clients '
            'rachètent frequemment nos "services".'
    >>> sanitize_unicode_string  (sentence)
    ... 'nosclientssontextrmementsatisfaitsdelaqualitduservice'
        'fournienoutrenosclientsrachtentfrequemmentnosservices'
    """
    sp_re = re.compile (r"[.'()-\\/’]")
    e_re = re.compile(r'[éèê]')
    a_re= re.compile(r'[àâ]')

    str_= re.sub('\s+', '', str_.strip().lower())
    
    for cobj , repl  in zip ( (sp_re, e_re, a_re), 
                             ("", 'e', 'a')): 
        str_ = cobj.sub(repl, str_)
    
    return str_             
                  
def read_main (csv_fn , pf , delimiter =':',
               destfile ='pme.en.csv') : 
    
    parser_data = list(parse_md_data(pf, delimiter) )
    parser_dict =dict(parser_data)
    
    with open (csv_fn, 'r', encoding ='utf8') as csv_f : 
        csv_reader = csv.reader(csv_f) 
        csv_data =[ row for row in csv_reader]
        
    # get the index of the last substring row 
    # and separate block into two from "Industry_type"
    ix = csv_data [0].index ('Industry_type') 
    
    csv_1b = [row [:ix +1] for row in csv_data] 
    csv_2b =[row [ix+1:] for row in csv_data ]
    # make a copy of csv_1b
    csv_1bb= copy.deepcopy(csv_1b)
    copyd = copy.deepcopy(csv_1bb); is_missing =list()
    
    # skip the first two rows 
    for ii, rowline in enumerate( csv_1bb[3:]) : 
        for jj , row in enumerate(rowline):
            row = row.strip()
            row = sanitize_unicode_string(row )
            csv_1bb[3:][ii][jj] = row 
            
    #collect the missing values 
    for ii, rowline in enumerate( csv_1bb[3:]) : 
        for jj , row in enumerate(rowline): 
            if row not in parser_dict.keys():
                is_missing.append(copyd[3:][ii][jj])
    is_missing = list(set(is_missing))       
    
    # merge the prior two blocks and build the dataframe
    new_csv_list = [r1 + r2 for r1, r2 in zip(csv_1bb, csv_2b )]
    df = pd.DataFrame (
        np.array(new_csv_list [1:]),
        columns =new_csv_list [0] 
                       )
    for key, value in parser_dict.items(): 
        # perform operation in place and return None 
        df.replace (key, value, inplace =True )
    

    df.to_csv (destfile)
    return  df , is_missing 
    

def _isin (
        arr: Union [ArrayLike, List [float]] ,
        subarr:Union[ _Sub [ArrayLike] , _Sub[List[float]] ,float], 
        return_mask:bool=False, 
) -> bool : 
    """ Check whether the subset array `subcz` is in  `cz` array. 
    
    :param arr: Array-like - Array of item elements 
    :param subarr: Array-like, float - Subset array containing a subset items.
    :param return_mask: bool, return the mask where the element is in `arr`.
    
    :return: True if items in  test array `subarr` are in array `arr`. 
    
    """
    arr = np.array (arr );  subarr = np.array(subarr )

    return (True if True in np.isin (arr, subarr) else False
            ) if not return_mask else np.isin (arr, subarr) 

def _assert_all_types (
    obj: object , 
    *expected_objtype: type, 
    objname:str=None, 
 ) -> object: 
    """ Quick assertion of object type. Raises a `TypeError` if wrong type 
    is passed as an argument. For polishing the error message, one can add  
    the object name `objname` for specifying the object that raises errors  
    for letting the users to be aware of the reason of failure."""
    # if np.issubdtype(a1.dtype, np.integer): 
    if not isinstance( obj, expected_objtype): 
        n=str(objname) + ' expects' if objname is not None else 'Expects'
        raise TypeError (
            f"{n} type{'s' if len(expected_objtype)>1 else ''} "
            f"{smart_format(tuple (o.__name__ for o in expected_objtype))}"
            f" but {type(obj).__name__!r} is given.")
            
    return obj 

  
def savepath_ (nameOfPath): 
    """
    Shortcut to create a folder 
    :param nameOfPath: Path name to save file
    :type nameOfPath: str 
    
    :return: 
        New folder created. If the `nameOfPath` exists, will return ``None``
    :rtype:str 
        
    """
 
    try :
        savepath = os.path.join(os.getcwd(), nameOfPath)
        if not os.path.isdir(savepath):
            os.mkdir(nameOfPath)#  mode =0o666)
    except :
        warnings.warn("The path seems to be existed !")
        return
    return savepath 
     

def drawn_boundaries(erp_data, appRes, index):
    """
    Function to drawn anomaly boundary 
    and return the anomaly with its boundaries
    
    :param erp_data: erp profile 
    :type erp_data: array_like or list 
    
    :param appRes: resistivity value of minimum pk anomaly 
    :type appRes: float 
    
    :param index: index of minimum pk anomaly 
    :type index: int 
    
    :return: anomaly boundary 
    :rtype: list of array_like 

    """
    f = 0 # flag to mention which part must be calculated 
    if index ==0 : 
        f = 1 # compute only right part 
    elif appRes ==erp_data[-1]: 
        f=2 # compute left part 
    
    def loop_sideBound(term):
        """
        loop side bar from anomaly and find the term side 
        
        :param term: is array of left or right side of anomaly.
        :type term: array 
        
        :return: side bar 
        :type: array_like 
        """
        tem_drawn =[]
        maxT=0 

        for ii, tem_rho in enumerate(term) : 

            diffRes_betw_2pts= tem_rho - appRes 
            if diffRes_betw_2pts > maxT : 
                maxT = diffRes_betw_2pts
                tem_drawn.append(tem_rho)
            elif diffRes_betw_2pts < maxT : 
                # rho_limit = tem_rho 
                break 
        return np.array(tem_drawn)
    # first broke erp profile from the anomalies 
    if f ==0 or f==2 : 
        left_term = erp_data[:index][::-1] # flip left term  for looping
        # flip again to keep the order 
        left_limit = loop_sideBound(term=left_term)[::-1] 

    if f==0 or f ==1 : 
        right_term= erp_data[index :]
        right_limit=loop_sideBound(right_term)
    # concat right and left to get the complete anomaly 
    if f==2: 
        anomalyBounds = np.append(left_limit,appRes)
                                   
    elif f ==1 : 
        anomalyBounds = np.array([appRes]+ right_limit.tolist())
    else: 
        left_limit = np.append(left_limit, appRes)
        anomalyBounds = np.concatenate((left_limit, right_limit))
    
    return appRes, index, anomalyBounds 

def serialize_data(
        data, 
        filename=None, 
        force=True, 
        savepath=None,
        verbose:int =0
     ): 
    """ Store a data into a binary file 
    
    :param data: Object
        Object to store into a binary file. 
    :param filename: str
        Name of file to serialize. If 'None', should create automatically. 
    :param savepath: str, PathLike object
         Directory to save file. If not exists should automaticallycreate.
    :param force: bool
        If ``True``, remove the old file if it exists, otherwise will 
        create a new incremenmted file.
    :param verbose: int, get more message.
    :return: dumped or serialized filename.
        
    :Example:
        
        >>> import numpy as np
        >>> import gofast.tools.coreutils import serialize_data
        >>> data = np.arange(15)
        >>> file = serialize_data(data, filename=None,  force=True, 
        ...                          savepath =None, verbose =3)
        >>> file
    """
    
    def _cif(filename, force): 
        """ Control the file. If `force` is ``True`` then remove the old file, 
        Otherwise create a new file with datetime infos."""
        f = copy.deepcopy(filename)
        if force : 
            os.remove(filename)
            if verbose >2: print(f" File {os.path.basename(filename)!r} "
                      "has been removed. ")
            return None   
        else :
            # that change the name in the realpath 
            f= os.path.basename(f).replace('.pkl','') + \
                f'{datetime.datetime.now()}'.replace(':', '_')+'.pkl' 
            return f

    if filename is not None: 
        file_exist =  os.path.isfile(filename)
        if file_exist: 
            filename = _cif (filename, force)
    if filename is None: 
        filename ='__mymemoryfile.{}__'.format(datetime.datetime.now())
        filename =filename.replace(' ', '_').replace(':', '-')
    if not isinstance(filename, str): 
        raise TypeError(f"Filename needs to be a string not {type(filename)}")
    if filename.endswith('.pkl'): 
        filename = filename.replace('.pkl', '')
 
    _logger.info (
        f"Save data to {'memory' if filename.find('memo')>=0 else filename}.")    
    try : 
        joblib.dump(data, f'{filename}.pkl')
        filename +='.pkl'
        if verbose > 2:
            print(f'Data dumped in `{filename} using to `~.externals.joblib`!')
    except : 
        # Now try to pickle data Serializing data 
        with open(filename, 'wb') as wfile: 
            pickle.dump( data, wfile)
        if verbose >2:
            print( 'Data are well serialized using Python pickle module.`')
    # take the real path of the filename
    filename = os.path.realpath(filename)

    if savepath is  None:
        dirname ='_memory_'
        try : savepath = sPath(dirname)
        except :
            # for consistency
            savepath = os.getcwd() 
    if savepath is not None: 
        try:
            shutil.move(filename, savepath)
        except :
            file = _cif (os.path.join(savepath,
                                      os.path.basename(filename)), force)
            if not force: 
                os.rename(filename, os.path.join(savepath, file) )
            if file is None: 
                #take the file  in current word 
                file = os.path.join(os.getcwd(), filename)
                shutil.move(filename, savepath)
            filename = os.path.join(savepath, file)
                
    if verbose > 0: 
            print(f"Data are well stored in {savepath!r} directory.")
            
    return os.path.join(savepath, filename) 
    
def load_serialized_data (filename, verbose=0): 
    """
    Load data from dumped file.
    
    :param filename: str or path-like object 
        Name of dumped data file.
    :return: Data reloaded from dumped file.

    :Example:
        
        >>> from gofast.tools.functils import load_serialized_data
        >>> data = load_serialized_data(
        ...    filename = '_memory_/__mymemoryfile.2021-10-29_14-49-35.647295__.pkl', 
        ...    verbose =3)

    """
    if not isinstance(filename, str): 
        raise TypeError(f'filename should be a <str> not <{type(filename)}>')
        
    if not os.path.isfile(filename): 
        raise FileExistsError(f"File {filename!r} does not exist.")

    _filename = os.path.basename(filename)
    _logger.info(
        f"Loading data from {'memory' if _filename.find('memo')>=0 else _filename}.")
   
    data =None 
    try : 
        data= joblib.load(filename)
        if verbose >2:
            (f"Data from {_filename !r} are sucessfully"
             " reloaded using ~.externals.joblib`!")
    except : 
        if verbose >2:
            print(f"Nothing to reload. It's seems data from {_filename!r}" 
                      " are not dumped using ~external.joblib module!")
        
        with open(filename, 'rb') as tod: 
            data= pickle.load (tod)
            
        if verbose >2: print(f"Data from `{_filename!r} are well"
                      " deserialized using Python pickle module.`!")
        
    is_none = data is None
    if verbose > 0:
        if is_none :
            print("Unable to deserialize data. Please check your file.")
        else : print(f"Data from {_filename} have been sucessfully reloaded.")
    
    return data

def save_job(
    job , 
    savefile ,* ,  
    protocol =None,  
    append_versions=True, 
    append_date=True, 
    fix_imports= True, 
    buffer_callback = None,   
    **job_kws
    ): 
    """ Quick save your job using 'joblib' or persistent Python pickle module.
    
    Parameters 
    -----------
    job: Any 
        Anything to save, preferabaly a models in dict 
        
    savefile: str, or path-like object 
         name of file to store the model
         The *file* argument must have a write() method that accepts a
         single bytes argument. It can thus be a file object opened for
         binary writing, an io.BytesIO instance, or any other custom
         object that meets this interface.
         
    append_versions: bool, default =True 
        Append the version of Joblib module or Python Pickle module following 
        by the scikit-learn, numpy and also pandas versions. This is useful 
        to have idea about previous versions for loading file when system or 
        modules have been upgraded. This could avoid bottleneck when data 
        have been stored for long times and user has forgotten the date and 
        versions at the time the file was saved. 
        
    append_date: bool, default=True, 
       Append the date  of the day to the filename. 
       
    protocol: int, optional 
        The optional *protocol* argument tells the pickler to use the
        given protocol; supported protocols are 0, 1, 2, 3, 4 and 5.
        The default protocol is 4. It was introduced in Python 3.4, and
        is incompatible with previous versions.
    
        Specifying a negative protocol version selects the highest
        protocol version supported.  The higher the protocol used, the
        more recent the version of Python needed to read the pickle
        produced.
        
    fix_imports: bool, default=True, 
        If *fix_imports* is True and *protocol* is less than 3, pickle
        will try to map the new Python 3 names to the old module names
        used in Python 2, so that the pickle data stream is readable
        with Python 2.
        
    buffer_call_back: int, optional 
        If *buffer_callback* is None (the default), buffer views are
        serialized into *file* as part of the pickle stream.
    
        If *buffer_callback* is not None, then it can be called any number
        of times with a buffer view.  If the callback returns a false value
        (such as None), the given buffer is out-of-band; otherwise the
        buffer is serialized in-band, i.e. inside the pickle stream.
    
        It is an error if *buffer_callback* is not None and *protocol*
        is None or smaller than 5.
        
    job_kws: dict, 
        Additional keywords arguments passed to :func:`joblib.dump`. 
        
    Returns
    --------
    savefile: str, 
        returns the filename
    """
    def remove_extension(fn, ex): 
        """Remove extension either joblib or pickle """
        return fn.replace (ex, '')
    
    import sklearn 
    
    versions = 'sklearn_v{0}.numpy_v{1}.pandas_v{2}'.format( 
        sklearn.__version__, np.__version__, pd.__version__) 
    date = datetime.datetime.now() 
    
    savefile =str(savefile) 
    if ( 
            '.joblib' in savefile or '.pkl' in savefile
            ): 
        ex = '.joblib' if savefile.find('.joblib')>=0 else '.pkl'
        savefile = remove_extension(savefile ,  ex )
        
    if append_date: 
        savefile +=".{}".format(date) 
        
    if append_versions : 
        savefile += ".{}"+ versions 
    try : 
        if append_versions: 
            savefile += ".joblib_v{}.".format(joblib.__version__)
            
        joblib.dump(job, f'{savefile}.joblib', **job_kws)
        
    except : 
        if append_versions: 
            savefile +=".pickle_v{}.pkl".format(pickle.__version__)
            
        with open(savefile, 'wb') as wfile: 
            pickle.dump( job, wfile, protocol= protocol, 
                        fix_imports=fix_imports , 
                        buffer_callback=buffer_callback )

    return savefile 

def fmt_text(
        anFeatures=None, 
        title = None,
        **kwargs) :
    """
    Function format text from anomaly features 
    
    :param anFeatures: Anomaly features 
    :type anFeatures: list or dict
    
    :param title: head lines 
    :type title: list
    
    :Example: 
        
        >>> from gofast.tools.coreutils import fmt_text
        >>> fmt_text(anFeatures =[1,130, 93,(146,145, 125)])
    
    """
    if title is None: 
        title = ['Ranking', 'rho(Ω.m)', 'position pk(m)', 'rho range(Ω.m)']
    inline =kwargs.pop('inline', '-')
    mlabel =kwargs.pop('mlabels', 100)
    line = inline * int(mlabel)
    
    #--------------------header ----------------------------------------
    print(line)
    tem_head ='|'.join(['{:^15}'.format(i) for i in title[:-1]])
    tem_head +='|{:^45}'.format(title[-1])
    print(tem_head)
    print(line)
    #-----------------------end header----------------------------------
    newF =[]
    if isinstance(anFeatures, dict):
        for keys, items in anFeatures.items(): 
            rrpos=keys.replace('_pk', '')
            rank=rrpos[0]
            pos =rrpos[1:]
            newF.append([rank, min(items), pos, items])
            
    elif isinstance(anFeatures, list): 
        newF =[anFeatures]
    
    
    for anFeatures in newF: 
        strfeatures ='|'.join(['{:^15}'.format(str(i)) \
                               for i in anFeatures[:-1]])
        try : 
            iter(anFeatures[-1])
        except : 
            strfeatures +='|{:^45}'.format(str(anFeatures[-1]))
        else : 
            strfeatures += '|{:^45}'.format(
                ''.join(['{} '.format(str(i)) for i in anFeatures[-1]]))
            
        print(strfeatures)
        print(line)
    

def wrap_infos (
        phrase ,
        value ='',
        underline ='-',
        unit ='',
        site_number= '',
        **kws) : 
    """Display info from anomaly details."""
    
    repeat =kws.pop('repeat', 77)
    intermediate =kws.pop('inter+', '')
    begin_phrase_mark= kws.pop('begin_phrase', '--|>')
    on = kws.pop('on', False)
    if not on: return ''
    else : 
        print(underline * repeat)
        print('{0} {1:<50}'.format(begin_phrase_mark, phrase), 
              '{0:<10} {1}'.format(value, unit), 
              '{0}'.format(intermediate), "{}".format(site_number))
        print(underline * repeat )
    

def reshape(arr , axis = None) :
    """ Detect the array shape and reshape it accordingly, back to the given axis. 
    
    :param array: array_like with number of dimension equals to 1 or 2 
    :param axis: axis to reshape back array. If 'axis' is None and 
        the number of dimension is greater than 1, it reshapes back array 
        to array-like 
    
    :returns: New reshaped array 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.tools.coreutils import reshape 
        >>> array = np.random.randn(50 )
        >>> array.shape
        ... (50,)
        >>> ar1 = reshape(array, 1) 
        >>> ar1.shape 
        ... (1, 50)
        >>> ar2 =reshape(ar1 , 0) 
        >>> ar2.shape 
        ... (50, 1)
        >>> ar3 = reshape(ar2, axis = None)
        >>> ar3.shape # goes back to the original array  
        >>> ar3.shape 
        ... (50,)
        
    """
    arr = np.array(arr)
    if arr.ndim > 2 : 
        raise ValueError('Expect an array with max dimension equals to 2' 
                         f' but {str(arr.ndim)!r} were given.')
        
    if axis  not in (0 , 1, -1, None): 
        raise ValueError(f'Wrong axis value: {str(axis)!r}')
        
    if axis ==-1:
        axis =None 
    if arr.ndim ==1 : 
        # ie , axis is None , array is an array-like object
        s0, s1= arr.shape [0], None 
    else : 
        s0, s1 = arr.shape 
    if s1 is None: 
        return  arr.reshape ((1, s0)) if axis == 1 else (arr.reshape (
            (s0, 1)) if axis ==0 else arr )
    try : 
        arr = arr.reshape ((s0 if s1==1 else s1, )) if axis is None else (
            arr.reshape ((1, s0)) if axis==1  else arr.reshape ((s1, 1 ))
            )
    except ValueError: 
        # error raises when user mistakes to input the right axis. 
        # (ValueError: cannot reshape array of size 54 into shape (1,1)) 
        # then return to him the original array 
        pass 

    return arr   
    
    
def ismissing(refarr, arr, fill_value = np.nan, return_index =False): 
    """ Get the missing values in array-like and fill it  to match the length
    of the reference array. 
    
    The function makes sense especially for frequency interpollation in the 
    'attenuation band' when using the audio-frequency magnetotelluric methods. 
    
    :param arr: array-like- Array to be extended with fill value. It should be  
        shorter than the `refarr`. Otherwise it returns the same array `arr` 
    :param refarr: array-like- the reference array. It should have a greater 
        length than the array 
    :param fill_value: float - Value to fill the `arr` to match the length of 
        the `refarr`. 
    :param return_index: bool or str - array-like, index of the elements element 
        in `arr`. Default is ``False``. Any other value should returns the 
        mask of existing element in reference array
        
    :returns: array and values missings or indexes in reference array. 
    
    :Example: 
        
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import ismissing
    >>> refreq = np.linspace(7e7, 1e0, 20) # 20 frequencies as reference
    >>> # remove the value between index 7 to 12 and stack again
    >>> freq = np.hstack ((refreq.copy()[:7], refreq.copy()[12:] ))  
    >>> f, m  = ismissing (refreq, freq)
    >>> f, m  
    ...array([7.00000000e+07, 6.63157895e+07, 6.26315791e+07, 5.89473686e+07,
           5.52631581e+07, 5.15789476e+07, 4.78947372e+07,            nan,
                      nan,            nan,            nan,            nan,
           2.57894743e+07, 2.21052638e+07, 1.84210534e+07, 1.47368429e+07,
           1.10526324e+07, 7.36842195e+06, 3.68421147e+06, 1.00000000e+00])
    >>> m # missing values 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368])
    >>>  _, m_ix  = ismissing (refreq, freq, return_index =True)
    >>> m_ix 
    ... array([ 7,  8,  9, 10, 11], dtype=int64)
    >>> # assert the missing values from reference values 
    >>> refreq[m_ix ] # is equal to m 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368]) 
        
    """
    return_index = str(return_index).lower() 
    fill_value = _assert_all_types(fill_value, float, int)
    if return_index in ('false', 'value', 'val') :
        return_index ='values' 
    elif return_index  in ('true', 'index', 'ix') :
        return_index = 'index' 
    else : 
        return_index = 'mask'
    
    ref = refarr.copy() ; mask = np.isin(ref, arr)
    miss_values = ref [~np.isin(ref, arr)] 
    miss_val_or_ix  = (ref [:, None] == miss_values).argmax(axis=0
                         ) if return_index =='index' else ref [~np.isin(ref, arr)] 
    
    miss_val_or_ix = mask if return_index =='mask' else miss_val_or_ix 
    # if return_missing_values: 
    ref [~np.isin(ref, arr)] = fill_value 
    #arr= np.hstack ((arr , np.repeat(fill_value, 0 if m <=0 else m  ))) 
    #refarr[refarr ==arr] if return_index else arr 
    return  ref , miss_val_or_ix   

def make_arr_consistent (
        refarr, arr, fill_value = np.nan, return_index = False, 
        method='naive'): 
    """
    Make `arr` to be consistent with the reference array `refarr`. Fill the 
    missing value with param `fill_value`. 
    
    Note that it does care of the position of the value in the array. Use 
    Numpy digitize to compute the bins. The array caveat here is the bins 
    must be monotonically decreasing or increasing.
    
    If the values in `arr` are present in `refarr`, the position of `arr` 
    in new consistent array should be located decreasing or increasing order. 
    
    Parameters 
    ------------
    arr: array-like 1d, 
        Array to extended with fill value. It should be  shorter than the 
        `refarr`.
        
    refarr: array-like- the reference array. It should have a greater 
        length than the array `arr`.  
    fill_value: float, 
        Value to fill the `arr` to match the length of the `refarr`. 
    return_index: bool or str, default=True 
         index of the position of the  elements in `refarr`.
         Default is ``False``. If ``mask`` should  return the 
        mask of existing element in reference array
    method: str, default="naive"
        Is the method used to find the right position of items in `arr`
        based on the reference array. 
        - ``naive``, considers the length of ``arr`` must fit the number of 
            items that should be visible in the consistent array. This method 
            erases the remaining bins values out of length of `arr`. 
        - ``strict` did the same but rather than considering the length, 
            it considers the maximum values in the `arr`. It assumes that `arr`
            is sorted in ascending order. This methods is usefull for plotting 
            a specific stations since the station loactions are sorted in 
            ascending order. 
        
    Returns 
    ---------
    non_zero_index , mask or t  
        index: indices of the position of `arr` items in ``refarr``. 
        mask: bool of the position `arr` items in ``refarr``
        t: new consistent array with the same length as ``refarr``
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import make_arr_consistent
    >>> refarr = np.arange (12) 
    >>> arr = np.arange (7, 10) 
    >>> make_arr_consistent (refarr, arr ) 
    Out[84]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., nan, nan])
    >>> make_arr_consistent (refarr, arr , return_index =True )
    Out[104]: array([7, 8, 9], dtype=int64)
    >>> make_arr_consistent (refarr, arr , return_index ="mask" )
    Out[105]: 
    array([False, False, False, False, False, False, False,  True,  True,
            True, False, False])
    >>> a = np.arange ( 12 ); b = np.linspace (7, 10 , 7) 
    >>> make_arr_consistent (a, b ) 
    Out[112]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., 10., 11.])
    >>> make_arr_consistent (a, b ,method='strict') 
    Out[114]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., 10., nan])
    """
    try : 
        refarr = reshape( refarr).shape[1] 
        arr= reshape( arr).shape[1] 
    except :pass 
    else: raise TypeError ("Expects one-dimensional arrays for both arrays.")

    t = np.full_like( refarr, fill_value = np.nan, dtype =float )
    temp_arr = np.digitize( refarr, arr) 
    non_zero_index = reshape (np.argwhere (temp_arr!=0 ) ) 
    t[non_zero_index] = refarr [non_zero_index] 
    # force value to keep only 
    # value in array 
    if method=='strict':
        index = reshape ( np.argwhere (  (max( arr)  - t) < 0 ) ) 
        t [index ]= np.nan 
    else: 
        if len (t[~np.isnan (t)]) > len(arr): 
            t [ - (len(t[~np.isnan (t)])-len(arr)):]= np.nan 
    # update the non_zeros index 
    non_zero_index= reshape ( np.argwhere (~np.isnan (t)))
    # now replace all NaN value by filled value 
    t [np.isnan(t)] = fill_value 

    return  refarr == t  if return_index =='mask' else (
        non_zero_index if return_index else t )

def find_close_position (refarr, arr): 
    """ Get the close item from `arr` in the reference array `refarr`. 
    
    :param arr: array-like 1d, 
        Array to extended with fill value. It should be  shorter than the 
        `refarr`.
        
    :param refarr: array-like- 
        the reference array. It should have a greater length than the
        array `arr`.  
    :return: generator of index of the closest position in  `refarr`.  
    """
    for item in arr : 
        ix = np.argmin (np.abs (refarr - item)) 
        yield ix 
    

def fit_ll(ediObjs, by ='index', method ='strict', distance='cartesian' ): 
    """ Fit EDI by location and reorganize EDI according to the site  
    longitude and latitude coordinates. 
    
    EDIs data are mostly reading in an alphabetically order, so the reoganization  

    according to the location(longitude and latitude) is usefull for distance 
    betwen site computing with a right position at each site.  
    
    :param ediObjs: list of EDI object, composed of a collection of 
        gofast.edi.Edi or pycsamt.core.edi.Edi or mtpy.core.edi objects 
    :type ediObjs: gofast.edi.Edi_Collection 
  
    :param by: ['name'|'ll'|'distance'|'index'|'name'|'dataid'] 
       The kind to sorting EDI files. Default uses the position number 
       included in the EDI-files name.
    :type by: str 
    
    :param method:  ['strict|'naive']. Kind of method to sort the 
        EDI file from longitude, latitude. Default is ``strict``. 
    :type method: str 
    
    :param distance: ['cartesian'|'harvesine']. Use the distance between 
       coordinates points to sort EDI files. Default is ``cartesian`` distance.
    :type distance: str 
    
    :returns: array splitted into ediObjs and Edifiles basenames 
    :rtyple: tuple 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.methods.em import EM
        >>> from gofast.tools.coreutils import fit_ll
        >>> edipath ='data/edi_ss' 
        >>> cediObjs = EM().fit (edipath) 
        >>> ediObjs = np.random.permutation(cediObjs.ediObjs) # shuffle the  
        ... # the collection of ediObjs 
        >>> ediObjs, ediObjbname = fit_by_ll(ediObjs) 
        ...

    """
    method= 'strict' if str(method).lower() =='strict' else "naive"
    if method=='strict': 
        return _fit_ll(ediObjs, by = by, distance = distance )
    
    #get the ediObjs+ names in ndarray(len(ediObjs), 2) 
    objnames = np.c_[ediObjs, np.array(
        list(map(lambda obj: os.path.basename(obj.edifile), ediObjs)))]
    lataddlon = np.array (list(map(lambda obj: obj.lat + obj.lon , ediObjs)))
    if len(np.unique ( lataddlon)) < len(ediObjs)//2: 
        # then ignore reorganization and used the 
        # station names. 
        pass 
    else:
        sort_ix = np.argsort(lataddlon) 
        objnames = objnames[sort_ix ]
        
    #ediObjs , objbnames = np.hsplit(objnames, 2) 
    return objnames[:, 0], objnames[:, -1]
   
def _fit_ll(ediObjs, distance='cartes', by = 'index'): 
    """ Fit ediObjs using the `strict method`. 
    
    An isolated part of :func:`gofast.tools.coreutils.fit_by_ll`. 
    """
    # get one obj randomnly and compute distance 
    obj_init = ediObjs[0]
    ref_lat = 34.0522  # Latitude of Los Angeles
    ref_lon = -118.2437 # Longitude of Los Angeles
    
    if str(distance).find ('harves')>=0: 
        distance='harves'
    else: distance='cartes'
    
    # create stations list.
    stations = [ 
        {"name": os.path.basename(obj.edifile), 
         "longitude": obj.lon, 
         "latitude": obj.lat, 
         "obj": obj, 
         "dataid": obj.dataid,  
         # compute distance using cartesian or harversine 
         "distance": _compute_haversine_d (
            ref_lat, ref_lon, obj.lat, obj.lon
            ) if distance =='harves' else np.sqrt (
                ( obj_init.lon -obj.lon)**2 + (obj_init.lat -obj.lat)**2), 
         # check wether there is a position number in the data.
         "index": re.search ('\d+', str(os.path.basename(obj.edifile)),
                            flags=re.IGNORECASE).group() if bool(
                                re.search(r'\d', os.path.basename(obj.edifile)))
                                else float(ii) ,
        } 
        for ii, obj in enumerate (ediObjs) 
        ]
                  
    ll=( 'longitude', 'latitude') 
    
    by = 'index' or str(by ).lower() 
    if ( by.find ('ll')>=0 or by.find ('lonlat')>=0): 
        by ='ll'
    elif  by.find ('latlon')>=0: 
        ll =ll[::-1] # reverse 
    
    # sorted from key
    sorted_stations = sorted (
        stations , key = lambda o: (o[ll[0]], [ll[-1]])  
        if (by =='ll' or by=='latlon')
        else o[by]
             )

    objnames = np.array( list(
        map ( lambda o : o['name'], sorted_stations))) 
    ediObjs = np.array ( list(
        map ( lambda o: o['obj'], sorted_stations)), 
                        dtype =object ) 
    
    return ediObjs, objnames 

def _compute_haversine_d(lat1, lon1, lat2, lon2): 
    """ Sort coordinates using Haversine distance calculus. 
    An isolated part of :func:`gofast.tools.coreutils._fit_by_ll"""
    # get reference_lat and reference lon 
    # get one obj randomnly and compute distance 
    # obj_init = np.random.choice (ediObjs) 
    import math 
    # Define a function to calculate the distance 
    # between two points in kilometers
    # def distance(lat1, lon1, lat2, lon2):
        # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Apply the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(
        lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Earth's radius in kilometers
    
    return c * r
    

def make_ids(arr, prefix =None, how ='py', skip=False): 
    """ Generate auto Id according to the number of given sites. 
    
    :param arr: Iterable object to generate an id site . For instance it can be 
        the array-like or list of EDI object that composed a collection of 
        gofast.edi.Edi object. 
    :type ediObjs: array-like, list or tuple 

    :param prefix: string value to add as prefix of given id. Prefix can be 
        the site name.
    :type prefix: str 
    
    :param how: Mode to index the station. Default is 'Python indexing' i.e. 
        the counting starts by 0. Any other mode will start the counting by 1.
    :type cmode: str 
    
    :param skip: skip the long formatage. the formatage acccording to the 
        number of collected file. 
    :type skip: bool 
    :return: ID number formated 
    :rtype: list 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.tools.func_utils import make_ids 
        >>> values = ['edi1', 'edi2', 'edi3'] 
        >>> make_ids (values, 'ix')
        ... ['ix0', 'ix1', 'ix2']
        >>> data = np.random.randn(20)
        >>>  make_ids (data, prefix ='line', how=None)
        ... ['line01','line02','line03', ... , line20] 
        >>> make_ids (data, prefix ='line', how=None, skip =True)
        ... ['line1','line2','line3',..., line20] 
        
    """ 
    fm='{:0' + ('1' if skip else '{}'.format(int(np.log10(len(arr))) + 1)) +'}'
    id_ =[str(prefix) + fm.format(i if how=='py'else i+ 1 ) if prefix is not 
          None else fm.format(i if how=='py'else i+ 1) 
          for i in range(len(arr))] 
    return id_    
    
def show_stats(nedic , nedir, fmtl='~', lenl=77, obj='EDI'): 
    """ Estimate the file successfully read reading over the unread files

    :param nedic: number of input or collected files 
    :param nedir: number of files read sucessfully 
    :param fmt: str to format the stats line 
    :param lenl: length of line denileation."""
    
    def get_obj_len (value):
        """ Control if obj is iterable then take its length """
        try : 
            iter(value)
        except :pass 
        else : value =len(value)
        return value 
    nedic = get_obj_len(nedic)
    nedir = get_obj_len(nedir)
    
    print(fmtl * lenl )
    mesg ='|'.join( ['|{0:<15}{1:^2} {2:<7}',
                     '{3:<15}{4:^2} {5:<7}',
                     '{6:<9}{7:^2} {8:<7}%|'])
    print(mesg.format('Data collected','=',  nedic, f'{obj} success. read',
                      '=', nedir, 'Rate','=', round ((nedir/nedic) *100, 2),
                      2))
    print(fmtl * lenl ) 
    
def concat_array_from_list (list_of_array , concat_axis = 0) :
    """ Concat array from list and set the None value in the list as NaN.
    
    :param list_of_array: List of array elements 
    :type list of array: list 
    
    :param concat_axis: axis for concatenation ``0`` or ``1``
    :type concat_axis: int 
    
    :returns: Concatenated array with shape np.ndaarry(
        len(list_of_array[0]), len(list_of_array))
    :rtype: np.ndarray 
    
    :Example: 
        
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import concat_array_from_list 
    >>> np.random.seed(0)
    >>> ass=np.random.randn(10)
    >>> ass = ass2=np.linspace(0,15,10)
    >>> concat_array_from_list ([ass, ass]) 
    
    """
    concat_axis =int(_assert_all_types(concat_axis, int, float))
    if concat_axis not in (0 , 1): 
        raise ValueError(f'Unable to understand axis: {str(concat_axis)!r}')
    
    list_of_array = list(map(lambda e: np.array([np.nan])
                             if e is None else np.array(e), list_of_array))
    # if the list is composed of one element of array, keep it outside
    # reshape accordingly 
    if len(list_of_array)==1:
        ar = (list_of_array[0].reshape ((1,len(list_of_array[0]))
                 ) if concat_axis==0 else list_of_array[0].reshape(
                        (len(list_of_array[0]), 1)
                 )
             ) if list_of_array[0].ndim ==1 else list_of_array[0]
                     
        return ar 

    #if concat_axis ==1: 
    list_of_array = list(map(
            lambda e:e.reshape(e.shape[0], 1) if e.ndim ==1 else e ,
            list_of_array)
        ) if concat_axis ==1 else list(map(
            lambda e:e.reshape(1, e.shape[0]) if e.ndim ==1 else e ,
            list_of_array))
                
    return np.concatenate(list_of_array, axis = concat_axis)
    
def station_id (id_, is_index= 'index', how=None, **kws): 
    """ 
    From id get the station  name as input  and return index `id`. 
    Index starts at 0.
    
    :param id_: str, of list of the name of the station or indexes . 
    
    :param is_index: bool 
        considered the given station as a index. so it remove all the letter and
        keep digit as index of each stations. 
        
    :param how: Mode to index the station. Default is 
        'Python indexing' i.e.the counting starts by 0. Any other mode will 
        start the counting by 1. Note that if `is_index` is ``True`` and the 
        param `how` is set to it default value ``py``, the station index should 
        be downgraded to 1. 
        
    :param kws: additionnal keywords arguments from :func:`~.make_ids`.
    
    :return: station index. If the list `id_` is given will return the tuple.
    
    :Example:
        
    >>> from gofast.tools.coreutils import station_id 
    >>> dat1 = ['S13', 's02', 's85', 'pk20', 'posix1256']
    >>> station_id (dat1)
    ... (13, 2, 85, 20, 1256)
    >>> station_id (dat1, how='py')
    ... (12, 1, 84, 19, 1255)
    >>> station_id (dat1, is_index= None, prefix ='site')
    ... ('site1', 'site2', 'site3', 'site4', 'site5')
    >>> dat2 = 1 
    >>> station_id (dat2) # return index like it is
    ... 1
    >>> station_id (dat2, how='py') # considering the index starts from 0
    ... 0
    
    """
    is_iterable =False 
    is_index = str(is_index).lower().strip() 
    isix=True if  is_index in ('true', 'index', 'yes', 'ix') else False 
    
    regex = re.compile(r'\d+', flags=re.IGNORECASE)
    try : 
        iter (id_)
    except : 
        id_= [id_]
    else : is_iterable=True 
    
    #remove all the letter 
    id_= list(map( lambda o: regex.findall(o), list(map(str, id_))))
    # merge the sequences list and for consistency remove emty list or str 
    id_=tuple(filter (None, list(itertools.chain(*id_)))) 
    
    # if considering as Python index return value -1 other wise return index 
    
    id_ = tuple (map(int, np.array(id_, dtype = np.int32)-1)
                 ) if how =='py' else tuple ( map(int, id_)) 
    
    if (np.array(id_) < 0).any(): 
        warnings.warn('Index contains negative values. Be aware that you are'
                      " using a Python indexing. Otherwise turn 'how' argumennt"
                      " to 'None'.")
    if not isix : 
        id_= tuple(make_ids(id_, how= how,  **kws))
        
    if not is_iterable : 
        try: id_ = id_[0]
        except : warnings.warn("The station id is given as a non iterable "
                          "object, but can keep the same format in return.")
        if id_==-1: id_= 0 if how=='py' else id_ + 2 

    return id_

def assert_doi(doi): 
    """
     assert the depth of investigation Depth of investigation converter 

    :param doi: depth of investigation in meters.  If value is given as string 
        following by yhe index suffix of kilometers 'km', value should be 
        converted instead. 
    :type doi: str|float 
    
    :returns doi:value in meter
    :rtype: float
           
    """
    if isinstance (doi, str):
        if doi.find('km')>=0 : 
            try: doi= float(doi.replace('km', '000')) 
            except :TypeError (" Unrecognized value. Expect value in 'km' "
                           f"or 'm' not: {doi!r}")
    try: doi = float(doi)
    except: TypeError ("Depth of investigation must be a float number "
                       "not: {str(type(doi).__name__!r)}")
    return doi
    
def strip_item(item_to_clean, item=None, multi_space=12):
    """
    Function to strip item around string values.  if the item to clean is None or 
    item-to clean is "''", function will return None value

    Parameters
    ----------
        * item_to_clean : list or np.ndarray of string 
                 List to strip item.
        * cleaner : str , optional
                item to clean , it may change according the use. The default is ''.
        * multi_space : int, optional
                degree of repetition may find around the item. The default is 12.
    Returns
    -------
        list or ndarray
            item_to_clean , cleaned item 
            
    :Example: 
        
     >>> import numpy as np
     >>> new_data=_strip_item (item_to_clean=np.array(['      ss_data','    pati   ']))
     >>>  print(np.array(['      ss_data','    pati   ']))
     ... print(new_data)

    """
    if item==None :
        item = ' '
    
    cleaner =[(''+ ii*'{0}'.format(item)) for ii in range(multi_space)]
    
    if isinstance (item_to_clean, str) : 
        item_to_clean=[item_to_clean] 
        
    # if type(item_to_clean ) != list :#or type(item_to_clean ) !=np.ndarray:
    #     if type(item_to_clean ) !=np.ndarray:
    #         item_to_clean=[item_to_clean]
    if item_to_clean in cleaner or item_to_clean ==['']:
        #warnings.warn ('No data found for sanitization; returns None.')
        return None 
    try : 
        multi_space=int(multi_space)
    except : 
        raise TypeError('argument <multplier> must be an integer'
                        'not {0}'.format(type(multi_space)))
    
    for jj, ss in enumerate(item_to_clean) : 
        for space in cleaner:
            if space in ss :
                new_ss=ss.strip(space)
                item_to_clean[jj]=new_ss
    
    return item_to_clean  
 
def parse_json(json_fn =None,
               data=None, 
               todo='load',
               savepath=None,
               verbose:int =0,
               **jsonkws):
    """ Parse Java Script Object Notation file and collect data from JSON
    config file. 
    
    :param json_fn: Json filename, URL or output JSON name if `data` is 
        given and `todo` is set to ``dump``.Otherwise the JSON output filename 
        should be the `data` or the given variable name.
    :param data: Data in Python obj to serialize. 
    :param todo: Action to perform with JSON: 
        - load: Load data from the JSON file 
        - dump: serialize data from the Python object and create a JSON file
    :param savepath: If ``default``  should save the `json_fn` 
        If path does not exist, should save to the <'_savejson_'>
        default path .
    :param verbose: int, control the verbosity. Output messages
    
    .. see also:: Read more about JSON doc
            https://docs.python.org/3/library/json.html
         or https://www.w3schools.com/python/python_json.asp 
         or https://www.geeksforgeeks.org/json-load-in-python/
         ...
 
    :Example: 
        >>> PATH = 'data/model'
        >>> k_ =['model', 'iter', 'mesh', 'data']
        >>> try : 
            INVERS_KWS = {
                s +'_fn':os.path.join(PATH, file) 
                for file in os.listdir(PATH) 
                          for s in k_ if file.lower().find(s)>=0
                          }
        except :
            INVERS=dict()
        >>> TRES=[10, 66,  70, 100, 1000, 3000]# 7000]     
        >>> LNS =['river water','fracture zone', 'MWG', 'LWG', 
              'granite', 'igneous rocks', 'basement rocks']
        >>> import gofast.tools.coreutils as FU
        >>> geo_kws ={'oc2d': INVERS_KWS, 
                      'TRES':TRES, 'LN':LNS}
        # serialize json data and save to  'jsontest.json' file
        >>> FU.parse_json(json_fn = 'jsontest.json', 
                          data=geo_kws, todo='dump', indent=3,
                          savepath ='data/saveJSON', sort_keys=True)
        # Load data from 'jsontest.json' file.
        >>> FU.parse_json(json_fn='data/saveJSON/jsontest.json', todo ='load')
    
    """
    todo, domsg =return_ctask(todo)
    # read urls by default json_fn can hold a url 
    try :
        if json_fn.find('http') >=0 : 
            todo, json_fn, data = fetch_json_data_from_url(json_fn, todo)
    except:
        #'NoneType' object has no attribute 'find' if data is not given
        pass 

    if todo.find('dump')>=0:
        json_fn = get_config_fname_from_varname(
            data, config_fname= json_fn, config='.json')
        
    JSON = dict(load=json.load,# use loads rather than load  
                loads=json.loads, 
                dump= json.dump, 
                dumps= json.dumps)
    try :
        if todo=='load': # read JSON files 
            with open(json_fn) as fj: 
                data =  JSON[todo](fj)  
        elif todo=='loads': # can be JSON string format 
            data = JSON[todo](json_fn) 
        elif todo =='dump': # store data in JSON file.
            with open(f'{json_fn}.json', 'w') as fw: 
                data = JSON[todo](data, fw, **jsonkws)
        elif todo=='dumps': # store data in JSON format not output file.
            data = JSON[todo](data, **jsonkws)

    except json.JSONDecodeError: 
        raise json.JSONDecodeError(f"Unable {domsg} JSON {json_fn!r} file. "
                              "Please check your file.", f'{json_fn!r}', 1)
    except: 
        msg =''.join([
        f"{'Unrecognizable file' if todo.find('load')>=0 else'Unable to serialize'}"
        ])
        
        raise TypeError(f'{msg} {json_fn!r}. Please check your'
                        f" {'file' if todo.find('load')>=0 else 'data'}.")
        
    cparser_manager(f'{json_fn}.json',savepath, todo=todo, dpath='_savejson_', 
                    verbose=verbose , config='JSON' )

    return data 
 
def fetch_json_data_from_url (url:str , todo:str ='load'): 
    """ Retrieve JSON data from url 
    :param url: Universal Resource Locator .
    :param todo:  Action to perform with JSON:
        - load: Load data from the JSON file 
        - dump: serialize data from the Python object and create a JSON file
    """
    with urllib.request.urlopen(url) as jresponse :
        source = jresponse.read()
    data = json.loads(source)
    if todo .find('load')>=0:
        todo , json_fn  ='loads', source 
        
    if todo.find('dump')>=0:  # then collect the data and dump it
        # set json default filename 
        todo, json_fn = 'dumps',  '_urlsourcejsonf.json'  
        
    return todo, json_fn, data 
    
def parse_csv(
        csv_fn:str =None,
        data=None, 
        todo='reader', 
        fieldnames=None, 
        savepath=None,
        header: bool=False, 
        verbose:int=0,
        **csvkws
   ) : 
    """ Parse comma separated file or collect data from CSV.
    
    :param csv_fn: csv filename,or output CSV name if `data` is 
        given and `todo` is set to ``write|dictwriter``.Otherwise the CSV 
        output filename should be the `c.data` or the given variable name.
    :param data: Sequence Data in Python obj to write. 
    :param todo: Action to perform with JSON: 
        - reader|DictReader: Load data from the JSON file 
        - writer|DictWriter: Write data from the Python object 
        and create a CSV file
    :param savepath: If ``default``  should save the `csv_fn` 
        If path does not exist, should save to the <'_savecsv_'>
        default path.
    :param fieldnames: is a sequence of keys that identify the order
        in which values in the dictionary passed to the `writerow()`
            method are written `csv_fn` file.
    :param savepath: If ``default``  should save the `csv_fn` 
        If path does not exist, should save to the <'_savecsv_'>
        default path .
    :param verbose: int, control the verbosity. Output messages
    :param csvkws: additional keywords csv class arguments 
    
    .. see also:: Read more about CSV module in:
        https://docs.python.org/3/library/csv.html or find some examples
        here https://www.programcreek.com/python/example/3190/csv.DictWriter 
        or find some FAQS here: 
    https://stackoverflow.com/questions/10373247/how-do-i-write-a-python-dictionary-to-a-csv-file
        ...
    :Example:
        >>> import gofast.tools.coreutils as FU
        >>> PATH = 'data/model'
        >>> k_ =['model', 'iter', 'mesh', 'data']
        >>> try : 
            INVERS_KWS = {
                s +'_fn':os.path.join(PATH, file) 
                for file in os.listdir(PATH) 
                          for s in k_ if file.lower().find(s)>=0
                          }
        except :
            INVERS=dict()
        >>> TRES=[10, 66,  70, 100, 1000, 3000]# 7000]     
        >>> LNS =['river water','fracture zone', 'MWG', 'LWG', 
              'granite', 'igneous rocks', 'basement rocks']
        >>> geo_kws ={'oc2d': INVERS_KWS, 
                      'TRES':TRES, 'LN':LNS}
        >>> # write data and save to  'csvtest.csv' file 
        >>> # here the `data` is a sequence of dictionary geo_kws
        >>> FU.parse_csv(csv_fn = 'csvtest.csv',data = [geo_kws], 
                         fieldnames = geo_kws.keys(),todo= 'dictwriter',
                         savepath = 'data/saveCSV')
        # collect csv data from the 'csvtest.csv' file 
        >>> FU.parse_csv(csv_fn ='data/saveCSV/csvtest.csv',
                         todo='dictreader',fieldnames = geo_kws.keys()
                         )
    
    """
    todo, domsg =return_ctask(todo) 
    
    if todo.find('write')>=0:
        csv_fn = get_config_fname_from_varname(
            data, config_fname= csv_fn, config='.csv')
    try : 
        if todo =='reader': 
            with open (csv_fn, 'r') as csv_f : 
                csv_reader = csv.reader(csv_f) # iterator 
                data =[ row for row in csv_reader]
                
        elif todo=='writer': 
            # write without a blank line, --> new_line =''
            with open(f'{csv_fn}.csv', 'w', newline ='',
                      encoding ='utf8') as new_csvf:
                csv_writer = csv.writer(new_csvf, **csvkws)
                csv_writer.writerows(data) if len(
                    data ) > 1 else csv_writer.writerow(data)  
                # for row in data:
                #     csv_writer.writerow(row) 
        elif todo=='dictreader':
            with open (csv_fn, 'r', encoding ='utf8') as csv_f : 
                # generate an iterator obj 
                csv_reader= csv.DictReader (csv_f, fieldnames= fieldnames) 
                # return csvobj as a list of dicts
                data = list(csv_reader) 
        
        elif todo=='dictwriter':
            with open(f'{csv_fn}.csv', 'w') as new_csvf:
                csv_writer = csv.DictWriter(new_csvf, **csvkws)
                if header:
                    csv_writer.writeheader()
                # DictWriter.writerows()expect a list of dicts,
                # while DictWriter.writerow() expect a single row of dict.
                csv_writer.writerow(data) if isinstance(
                    data , dict) else csv_writer.writerows(data)  
                
    except csv.Error: 
        raise csv.Error(f"Unable {domsg} CSV {csv_fn!r} file. "
                      "Please check your file.")
    except: 

        msg =''.join([
        f"{'Unrecognizable file' if todo.find('read')>=0 else'Unable to write'}"
        ])
        
        raise TypeError(f'{msg} {csv_fn!r}. Please check your'
                        f" {'file' if todo.find('read')>=0 else 'data'}.")
    cparser_manager(f'{csv_fn}.csv',savepath, todo=todo, dpath='_savecsv_', 
                    verbose=verbose , config='CSV' )
    
    return data  
   
def return_ctask (todo:Optional[str]=None) -> Tuple [str, str]: 
    """ Get the convenient task to do if users misinput the `todo` action.
    
    :param todo: Action to perform: 
        - load: Load data from the config [YAML|CSV|JSON] file
        - dump: serialize data from the Python object and 
            create a config [YAML|CSV|JSON] file."""
            
    def p_csv(v, cond='dict', base='reader'):
        """ Read csv instead. 
        :param v: str, value to do 
        :param cond: str, condition if  found in the value `v`. 
        :param base: str, base task to do if condition `cond` is not met. 
        
        :Example: 
            
        >>> todo = 'readingbook' 
        >>> p_csv(todo) <=> 'dictreader' if todo.find('dict')>=0 else 'reader' 
        """
        return  f'{cond}{base}' if v.find(cond) >=0 else base   
    
    ltags = ('load', 'recover', True, 'fetch')
    dtags = ('serialized', 'dump', 'save', 'write','serialize')
    if todo is None: 
        raise ValueError('NoneType action can not be perform. Please '
                         'specify your action: `load` or `dump`?' )
    
    todo =str(todo).lower() 
    ltags = list(ltags) + [todo] if  todo=='loads' else ltags
    dtags= list(dtags) +[todo] if  todo=='dumps' else dtags 

    if todo in ltags: 
        todo = 'loads' if todo=='loads' else 'load'
        domsg= 'to parse'
    elif todo in dtags: 
        todo = 'dumps' if todo=='dumps' else 'dump'
        domsg  ='to serialize'
    elif todo.find('read')>=0:
        todo = p_csv(todo)
        domsg= 'to read'
    elif todo.find('write')>=0: 
        todo = p_csv(todo, base ='writer')
        domsg =' to write'
        
    else :
        raise ValueError(f'Wrong action {todo!r}. Please select'
                         f' the right action to perform: `load` or `dump`?'
                        ' for [JSON|YAML] and `read` or `write`? '
                        'for [CSV].')
    return todo, domsg  

def parse_yaml (yml_fn:str =None, data=None,
                todo='load', savepath=None,
                verbose:int =0, **ymlkws) : 
    """ Parse yml file and collect data from YAML config file. 
    
    :param yml_fn: yaml filename and can be the output YAML name if `data` is 
        given and `todo` is set to ``dump``.Otherwise the YAML output filename 
        should be the `data` or the given variable name.
    :param data: Data in Python obj to serialize. 
    :param todo: Action to perform with YAML: 
        - load: Load data from the YAML file 
        - dump: serialize data from the Python object and create a YAML file
    :param savepath: If ``default``  should save the `yml_fn` 
        to the default path otherwise should store to the convenient path.
        If path does not exist, should set to the default path.
    :param verbose: int, control the verbosity. Output messages
    
    .. see also:: Read more about YAML file https://pynative.com/python-yaml/
         or https://python.land/data-processing/python-yaml and download YAML 
         at https://pypi.org/project/PyYAML/
         ...

    """ 
    
    todo, domsg =return_ctask(todo)
    #in the case user use dumps or loads with 's'at the end 
    if todo.find('dump')>= 0: 
        todo='dump'
    if todo.find('load')>=0:
        todo='load'
    if todo=='dump':
        yml_fn = get_config_fname_from_varname(data, yml_fn)
    try :
        if todo=='load':
            with open(yml_fn) as fy: 
                data =  yaml.load(fy, Loader=yaml.SafeLoader)  
                # args =yaml.safe_load(fy)
        elif todo =='dump':
        
            with open(f'{yml_fn}.yml', 'w') as fw: 
                data = yaml.dump(data, fw, **ymlkws)
    except yaml.YAMLError: 
        raise yaml.YAMLError(f"Unable {domsg} YAML {yml_fn!r} file. "
                             'Please check your file.')
    except: 
        msg =''.join([
        f"{'Unrecognizable file' if todo=='load'else'Unable to serialize'}"
        ])
        
        raise TypeError(f'{msg} {yml_fn!r}. Please check your'
                        f" {'file' if todo=='load' else 'data'}.")
        
    cparser_manager(f'{yml_fn}.yml',savepath, todo=todo, dpath='_saveyaml_', 
                    verbose=verbose , config='YAML' )

    return data 
 
def cparser_manager (
    cfile,
    savepath =None, 
    todo:str ='load', 
    dpath=None,
    verbose =0, 
    **pkws): 
    """ Save and output message according to the action. 
    
    :param cfile: name of the configuration file
    :param savepath: Path-like object 
    :param dpath: default path 
    :param todo: Action to perform with config file. Can ve 
        ``load`` or ``dump``
    :param config: Type of configuration file. Can be ['YAML|CSV|JSON]
    :param verbose: int, control the verbosity. Output messages
    
    """
    if savepath is not None:
        if savepath =='default': 
            savepath = None 
        yml_fn,_= move_cfile(cfile, savepath, dpath=dpath)
    if verbose > 0: 
        print_cmsg(yml_fn, todo, **pkws)
        
    
def get_config_fname_from_varname(data,
                                  config_fname=None,
                                  config='.yml') -> str: 
    """ use the variable name given to data as the config file name.
    
    :param data: Given data to retrieve the variable name 
    :param config_fname: Configurate variable filename. If ``None`` , use 
        the name of the given varibale data 
    :param config: Type of file for configuration. Can be ``json``, ``yml`` 
        or ``csv`` file. default is ``yml``.
    :return: str, the configuration data.
    
    """
    try:
        if '.' in config: 
            config =config.replace('.','')
    except:pass # in the case None is given
    
    if config_fname is None: # get the varname 
        # try : 
        #     from varname.helpers import Wrapper 
        # except ImportError: 
        #     import_varname=False 
        #     import_varname = FU.subprocess_module_installation('varname')
        #     if import_varname: 
        #         from varname.helpers import Wrapper 
        # else : import_varname=True 
        try : 
            for c, n in zip(['yml', 'yaml', 'json', 'csv'],
                            ['cy.data', 'cy.data', 'cj.data',
                             'c.data']):
                if config ==c:
                    config_fname= n
                    break 
            if config_fname is None:
                raise # and go to except  
        except: 
            #using fstring 
            config_fname= f'{data}'.split('=')[0]
            
    elif config_fname is not None: 
        config_fname= config_fname.replace(
            f'.{config}', '').replace(f'.{config}', '').replace('.yaml', '')
    
    return config_fname

def pretty_printer(
        clfs: List[_F],  
        clf_score:List[float]=None, 
        scoring: Optional[str] =None,
        **kws
 )->None: 
    """ Format and pretty print messages after gridSearch using multiples
    estimators.
    
    Display for each estimator, its name, it best params with higher score 
    and the mean scores. 
    
    Parameters
    ----------
    clfs:Callables 
        classifiers or estimators 
    
    clf_scores: array-like
        for single classifier, usefull to provided the 
        cross validation score.
    
    scoring: str 
        Scoring used for grid search.
    """
    empty =kws.pop('empty', ' ')
    e_pad =kws.pop('e_pad', 2)
    p=list()

    if not isinstance(clfs, (list,tuple)): 
        clfs =(clfs, clf_score)

    for ii, (clf, clf_be, clf_bp, clf_sc) in enumerate(clfs): 
        s_=[e_pad* empty + '{:<20}:'.format(
            clf.__class__.__name__) + '{:<20}:'.format(
                'Best-estimator <{}>'.format(ii+1)) +'{}'.format(clf_be),
         e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'Best paramaters') + '{}'.format(clf_bp),
         e_pad* empty  +'{:<20}:'.format(' ') + '{:<20}:'.format(
            'scores<`{}`>'.format(scoring)) +'{}'.format(clf_sc)]
        try : 
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ '{}'.format(clf_sc.mean())]
        except AttributeError:
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ 'None']
            s_ +=s0
        else :
            s_ +=s0

        p.extend(s_)
    
    for i in p: 
        print(i)
 
def move_cfile(
    cfile: str, 
    savepath: Optional[str] = None, 
    **ckws
) -> Tuple[str, str]:
    """Move file to its savepath and output message.

    If the savepath does not exist, it will be created. If the file cannot
     be moved, it will be copied then the original will be deleted.

    Parameters:
    - cfile (str): Name of the file to move.
    - savepath (str, optional): Destination path for the file. If not provided 
      or determined, a default provided by `cpath` is used.
    
    Returns:
    Tuple[str, str]: Updated file path and success message.
    """
    # Assuming `cpath` function adjusts `savepath` based on `ckws`, 
    # including default path handling
    savepath = cpath(savepath, **ckws)

    # Ensure the savepath directory exists
    os.makedirs(savepath, exist_ok=True)

    destination_file_path = os.path.join(savepath, os.path.basename(cfile))
    
    try:
        shutil.move(cfile, destination_file_path)
    except shutil.Error as e: 
        # If the move is unsuccessful, copy then delete the original file
        shutil.copy2(cfile, destination_file_path)
        os.remove(cfile)
        message = (f"Warning: Could not move '{cfile}'. It was copied and the"
                   f" original was deleted. {e}")
        _logger.warning(message) 

    msg = (f"--> '{os.path.basename(destination_file_path)}' data was successfully"
           f" saved to '{os.path.realpath(destination_file_path)}'.")
 
    return destination_file_path, msg


def print_cmsg(cfile:str, todo:str='load', config:str='YAML') -> str: 
    """ Output configuration message. 
    
    :param cfile: name of the configuration file
    :param todo: Action to perform with config file. Can be 
        ``load`` or ``dump``
    :param config: Type of configuration file. Can be [YAML|CSV|JSON]
    """
    if todo=='load': 
        msg = ''.join([
        f'--> Data was successfully stored to {os.path.basename(cfile)!r}', 
            f' and saved to {os.path.realpath(cfile)!r}.']
            )
    elif todo=='dump': 
        msg =''.join([ f"--> {config.upper()} {os.path.basename(cfile)!r}", 
                      " data was sucessfully loaded."])
    return msg 


def random_state_validator(seed):
    """Turn seed into a Numpy-Random-RandomState instance.
    
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

def is_iterable (
        y, exclude_string= False, transform = False , parse_string =False, 
)->Union [bool , list]: 
    """ Asserts iterable object and returns boolean or transform object into
     an iterable.
    
    Function can also transform a non-iterable object to an iterable if 
    `transform` is set to ``True``.
    
    :param y: any, object to be asserted 
    :param exclude_string: bool, does not consider string as an iterable 
        object if `y` is passed as a string object. 
    :param transform: bool, transform  `y` to an iterable objects. But default 
        puts `y` in a list object. 
    :param parse_string: bool, parse string and convert the list of string 
        into iterable object is the `y` is a string object and containg the 
        word separator character '[#&.*@!_,;\s-]'. Refer to the function 
        :func:`~gofast.tools.coreutils.str2columns` documentation.
        
    :returns: 
        - bool, or iterable object if `transform` is set to ``True``. 
        
    .. note:: 
        Parameter `parse_string` expects `transform` to be ``True``, otherwise 
        a ValueError will raise. Note :func:`.is_iterable` is not dedicated 
        for string parsing. It parses string using the default behaviour of 
        :func:`.str2columns`. Use the latter for string parsing instead. 
        
    :Examples: 
    >>> from gofast.coreutils.is_iterable 
    >>> is_iterable ('iterable', exclude_string= True ) 
    Out[28]: False
    >>> is_iterable ('iterable', exclude_string= True , transform =True)
    Out[29]: ['iterable']
    >>> is_iterable ('iterable', transform =True)
    Out[30]: 'iterable'
    >>> is_iterable ('iterable', transform =True, parse_string=True)
    Out[31]: ['iterable']
    >>> is_iterable ('iterable', transform =True, exclude_string =True, 
                     parse_string=True)
    Out[32]: ['iterable']
    >>> is_iterable ('parse iterable object', parse_string=True, 
                     transform =True)
    Out[40]: ['parse', 'iterable', 'object']
    """
    if (parse_string and not transform) and isinstance (y, str): 
        raise ValueError ("Cannot parse the given string. Set 'transform' to"
                          " ``True`` otherwise use the 'str2columns' utils"
                          " from 'gofast.tools.coreutils' instead.")
    y = str2columns(y) if isinstance(y, str) and parse_string else y 
    
    isiter = False  if exclude_string and isinstance (
        y, str) else hasattr (y, '__iter__')
    
    return ( y if isiter else [ y ] )  if transform else isiter 

    
def str2columns (text,  regex=None , pattern = None): 
    """Split text from the non-alphanumeric markers using regular expression. 
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    text: str, 
        text litteral containing the columns the names to retrieve
        
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[#&*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    Returns
    -------
    attr: List of attributes 
    
    Examples
    ---------
    >>> from gofast.tools.coreutils import str2columns 
    >>> text = ('this.is the text to split. It is an: example of; splitting str - to text.')
    >>> str2columns (text )  
    ... ['this',
         'is',
         'the',
         'text',
         'to',
         'split',
         'It',
         'is',
         'an:',
         'example',
         'of',
         'splitting',
         'str',
         'to',
         'text']

    """
    pattern = pattern or  r'[#&.*@!_,;\s-]\s*'
    regex = regex or re.compile (pattern, flags=re.IGNORECASE) 
    text= list(filter (None, regex.split(str(text))))
    return text 
       
def sanitize_frame_cols(
        d,  func:_F = None , regex=None, pattern:str = None, 
        fill_pattern:str =None, inplace:bool =False 
        ):
    """ Remove an indesirable characters to the dataframe and returns 
    new columns. 
    
    Use regular expression for columns sanitizing 
    
    Parameters 
    -----------
    
    d: list, columns, 
        columns to sanitize. It might contain a list of items to 
        to polish. If dataframe or series are given, the dataframe columns  
        and the name respectively will be polished and returns the same 
        dataframe.
        
    func: _F, callable 
       Universal function used to clean the columns 
       
    regex: `re` object,
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&.)(*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[_#&.)(*@!_,;\s-]\s*'
        The base pattern to sanitize the text in each column names. 
        
    fill_pattern: str, default='' 
        pattern to replace the non-alphabetic character in each item of 
        columns. 
    inplace: bool, default=False, 
        transform the dataframe of series in place. 

    Returns
    -------
    columns | pd.Series | dataframe. 
        return Serie or dataframe if one is given, otherwise it returns a 
        sanitized columns. 
        
    Examples 
    ---------
    >>> from gofast.tools.coreutils import sanitize_frame_cols 
    >>> from gofast.tools.coreutils import read_data 
    >>> h502= read_data ('data/boreholes/H502.xlsx') 
    >>> h502 = sanitize_frame_cols (h502, fill_pattern ='_' ) 
    >>> h502.columns[:3]
    ... Index(['depth_top', 'depth_bottom', 'strata_name'], dtype='object') 
    >>> f = lambda r : r.replace ('_', "'s ") 
    >>> h502_f= sanitize_frame_cols( h502, func =f )
    >>> h502_f.columns [:3]
    ... Index(['depth's top', 'depth's bottom', 'strata's name'], dtype='object')
               
    """
    isf , iss= False , False 
    pattern = pattern or r'[_#&.)(*@!_,;\s-]\s*'
    fill_pattern = fill_pattern or '' 
    fill_pattern = str(fill_pattern)
    
    regex = regex or re.compile (pattern, flags=re.IGNORECASE)
    
    if isinstance(d, pd.Series): 
        c = [d.name]  
        iss =True 
    elif isinstance (d, pd.DataFrame ) :
        c = list(d.columns) 
        isf = True
        
    else : 
        if not is_iterable(d) : c = [d] 
        else : c = d 
        
    if inspect.isfunction(func): 
        c = list( map (func , c ) ) 
    
    else : c =list(map ( 
        lambda r : regex.sub(fill_pattern, r.strip() ), c ))
        
    if isf : 
        if inplace : d.columns = c
        else : d =pd.DataFrame(d.values, columns =c )
        
    elif iss:
        if inplace: d.name = c[0]
        else : d= pd.Series (data =d.values, name =c[0] )
        
    else : d = c 

    return d 

def to_hdf5(d, fn, objname =None, close =True,  **hdf5_kws): 
    """
    Store a frame data in hierachical data format 5 (HDF5) 
    
    Note that is `d` is a dataframe, make sure that the dependency 'pytables'
    is already installed, otherwise and error raises. 
    
    Parameters 
    -----------
    d: ndarray, 
        data to store in HDF5 format 
    fn: str, 
        File path to HDF5 file.
    objname: str, 
        name of the data to store 
    close: bool, default =True 
        when data is given as an array, data can still be added if 
        close is set to ``False``, otherwise, users need to open again in 
        read mode 'r' before pursuing the process of adding. 
    hdf5_kws: dict of :class:`pandas.pd.HDFStore`  
        Additional keywords arguments passed to pd.HDFStore. they could be:
        *  mode : {'a', 'w', 'r', 'r+'}, default 'a'
    
             ``'r'``
                 Read-only; no data can be modified.
             ``'w'``
                 Write; a new file is created (an existing file with the same
                 name would be deleted).
             ``'a'``
                 Append; an existing file is opened for reading and writing,
                 and if the file does not exist it is created.
             ``'r+'``
                 It is similar to ``'a'``, but the file must already exist.
         * complevel : int, 0-9, default None
             Specifies a compression level for data.
             A value of 0 or None disables compression.
         * complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
             Specifies the compression library to be used.
             As of v0.20.2 these additional compressors for Blosc are supported
             (default if no compressor specified: 'blosc:blosclz'):
             {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
              'blosc:zlib', 'blosc:zstd'}.
             Specifying a compression library which is not available issues
             a ValueError.
         * fletcher32 : bool, default False
             If applying compression use the fletcher32 checksum.
    Returns
    ------- 
    store : Dict-like IO interface for storing pandas objects.
    
    Examples 
    ------------
    >>> import os 
    >>> from gofast.tools.coreutils import sanitize_frame_cols, to_hdf5 
    >>> from gofast.tools import read_data 
    >>> data = read_data('data/boreholes/H502.xlsx') 
    >>> sanitize_frame_cols (data, fill_pattern='_', inplace =True ) 
    >>> store_path = os.path.join('gofast/datasets/data', 'h') # 'h' is the name of the data 
    >>> store = to_hdf5 (data, fn =store_path , objname ='h502' ) 
    >>> store 
    ... 
    >>> # fetch the data 
    >>> h502 = store ['h502'] 
    >>> h502.columns[:3] 
    ... Index(['hole_number', 'depth_top', 'depth_bottom'], dtype='object')

    """
    store =None 
    if ( 
        not hasattr (d, '__array__') 
        or not hasattr (d, 'columns')
            ) : 
        raise TypeError ("Expect an array or dataframe,"
                         f" not {type (d).__name__!r}")
        
    if hasattr (d, '__array__') and hasattr (d, "columns"): 
        # assert whether pytables is installed 
        import_optional_dependency ('tables') 
        # remove extension if exist.
        fn = str(fn).replace ('.h5', "").replace(".hdf5", "")
        # then store. 
        store = pd.HDFStore(fn +'.h5' ,  **hdf5_kws)
        objname = objname or 'data'
        store[ str(objname) ] = d 

    
    elif not hasattr(d, '__array__'): 
        d = np.asarray(d) 
 
        store= h5py.File(f"{fn}.hdf5", "w") 
        store.create_dataset("dataset_01", store.shape, 
                             dtype=store.dtype,
                             data=store
                             )
        
    if close : store.close () 

    return store 
    

def find_by_regex (o , pattern,  func = re.match, **kws ):
    """ Find pattern in object whatever an "iterable" or not. 
    
    when we talk about iterable, a string value is not included.
    
    Parameters 
    -------------
    o: str or iterable,  
        text litteral or an iterable object containing or not the specific 
        object to match. 
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
    
    func: re callable , default=re.match
        regular expression search function. Can be
        [re.match, re.findall, re.search ],or any other regular expression 
        function. 
        
        * ``re.match()``:  function  searches the regular expression pattern and 
            return the first occurrence. The Python RegEx Match method checks 
            for a match only at the beginning of the string. So, if a match is 
            found in the first line, it returns the match object. But if a match 
            is found in some other line, the Python RegEx Match function returns 
            null.
        * ``re.search()``: function will search the regular expression pattern 
            and return the first occurrence. Unlike Python re.match(), it will 
            check all lines of the input string. The Python re.search() function 
            returns a match object when the pattern is found and “null” if 
            the pattern is not found
        * ``re.findall()`` module is used to search for 'all' occurrences that 
            match a given pattern. In contrast, search() module will only 
            return the first occurrence that matches the specified pattern. 
            findall() will iterate over all the lines of the file and will 
            return all non-overlapping matches of pattern in a single step.
    kws: dict, 
        Additional keywords arguments passed to functions :func:`re.match` or 
        :func:`re.search` or :func:`re.findall`. 
        
    Returns 
    -------
    om: list 
        matched object put is the list 
        
    Example
    --------
    >>> from gofast.tools.coreutils import find_by_regex
    >>> from gofast.datasets import load_hlogs 
    >>> X0, _= load_hlogs (as_frame =True )
    >>> columns = X0.columns 
    >>> str_columns =','.join (columns) 
    >>> find_by_regex (str_columns , pattern='depth', func=re.search)
    ... ['depth']
    >>> find_by_regex(columns, pattern ='depth', func=re.search)
    ... ['depth_top', 'depth_bottom']
    
    """
    om = [] 
    if isinstance (o, str): 
        om = func ( pattern=pattern , string = o, **kws)
        if om: 
            om= om.group() 
        om =[om]
    elif is_iterable(o): 
        o = list(o) 
        for s in o : 
            z = func (pattern =pattern , string = s, **kws)
            if z : 
                om.append (s) 
                
    if func.__name__=='findall': 
        om = list(itertools.chain (*om )) 
    # keep None is nothing 
    # fit the corresponding pattern 
    if len(om) ==0 or om[0] is None: 
        om = None 
    return  om 
    
def is_in_if (o: iter,  items: Union [str , iter], error = 'raise', 
               return_diff =False, return_intersect = False): 
    """ Raise error if item is not  found in the iterable object 'o' 
    
    :param o: unhashable type, iterable object,  
        object for checkin. It assumes to be an iterable from which 'items' 
        is premused to be in. 
    :param items: str or list, 
        Items to assert whether it is in `o` or not. 
    :param error: str, default='raise'
        raise or ignore error when none item is found in `o`. 
    :param return_diff: bool, 
        returns the difference items which is/are not included in 'items' 
        if `return_diff` is ``True``, will put error to ``ignore`` 
        systematically.
    :param return_intersect:bool,default=False
        returns items as the intersection between `o` and `items`.
    :raise: ValueError 
        raise ValueError if `items` not in `o`. 
    :return: list,  
        `s` : object found in ``o` or the difference object i.e the object 
        that is not in `items` provided that `error` is set to ``ignore``.
        Note that if None object is found  and `error` is ``ignore`` , it 
        will return ``None``, otherwise, a `ValueError` raises. 
        
    :example: 
        >>> from gofast.datasets import load_hlogs 
        >>> from gofast.tools.coreutils import is_in_if 
        >>> X0, _= load_hlogs (as_frame =True )
        >>> is_in_if  (X0 , items= ['depth_top', 'top']) 
        ... ValueError: Item 'top' is missing in the object 
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore') 
        ... ['depth_top']
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore',
                       return_diff= True) 
        ... ['sp',
         'well_diameter',
         'layer_thickness',
         'natural_gamma',
         'short_distance_gamma',
         'strata_name',
         'gamma_gamma',
         'depth_bottom',
         'rock_name',
         'resistivity',
         'hole_id']
    """
    
    if isinstance (items, str): 
        items =[items]
    elif not is_iterable(o): 
        raise TypeError (f"Expect an iterable object, not {type(o).__name__!r}")
    # find intersect object 
    s= set (o).intersection (items) 
    
    miss_items = list(s.difference (o)) if len(s) > len(
        items) else list(set(items).difference (s)) 

    if return_diff or return_intersect: 
        error ='ignore'
    
    if len(miss_items)!=0 :
        if error =='raise': 
            v= smart_format(miss_items)
            verb = f"{ ' '+ v +' is' if len(miss_items)<2 else  's '+ v + 'are'}"
            raise ValueError (
                f"Item{verb} missing in the {type(o).__name__.lower()} {o}.")
            
       
    if return_diff : 
        # get difference 
        s = list(set(o).difference (s))  if len(o) > len( 
            s) else list(set(items).difference (s)) 
        # s = set(o).difference (s)  
    elif return_intersect: 
        s = list(set(o).intersection(s))  if len(o) > len( 
            items) else list(set(items).intersection (s))     
    
    s = None if len(s)==0 else list (s) 
    
    return s  
  
def map_specific_columns ( 
        X: DataFrame, 
        ufunc:_F , 
        columns_to_skip:List[str]=None,   
        pattern:str=None, 
        inplace:bool= False, 
        **kws
        ): 
    """ Apply function to a specific columns is the dataframe. 
    
    It is possible to skip some columns that we want operation to not be 
    performed.
    
    Parameters 
    -----------
    X: dataframe, 
        pandas dataframe with valid columns 
    ufunc: callable, 
        Universal function that can be applying to the dataframe. 
    columns_to_skip: list or str , 
        List of columns to skip. If given as string and separed by the default
        pattern items, it should be converted to a list and make sure the 
        columns name exist in the dataframe. Otherwise an error with 
        raise.
        
    pattern: str, default = '[#&*@!,;\s]\s*'
        The base pattern to split the text in `column2skip` into a columns
        For instance, the following string coulb be splitted to:: 
            
            'depth_top, thickness, sp, gamma_gamma' -> 
            ['depth_top', 'thickness', 'sp', 'gamma_gamma']
        
        Refer to :func:`~.str2columns` for further details. 
    inplace: bool, default=True 
        Modified dataframe in place and return None, otherwise return a 
        new dataframe 
    kws: dict, 
        Keywords argument passed to :func: `pandas.DataFrame.apply` function 
        
    Returns 
    ---------
    X: Dataframe or None 
        Dataframe modified inplace with values computed using the given 
        `func`except the skipped columns, or ``None`` if `inplace` is ``True``. 
        
    Examples 
    ---------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.plotutils import map_specific_columns 
    >>> X0, _= load_hlogs (as_frame =True ) 
    >>> # let visualize the  first3 values of `sp` and `resistivity` keys 
    >>> X0['sp'][:3] , X0['resistivity'][:3]  
    ... (0   -1.580000
         1   -1.580000
         2   -1.922632
         Name: sp, dtype: float64,
         0    15.919130
         1    16.000000
         2    24.422316
         Name: resistivity, dtype: float64)
    >>> column2skip = ['hole_id','depth_top', 'depth_bottom', 
                      'strata_name', 'rock_name', 'well_diameter', 'sp']
    >>> map_specific_columns (X0, ufunc = np.log10, column2skip)
    >>> # now let visualize the same keys values 
    >>> X0['sp'][:3] , X0['resistivity'][:3]
    ... (0   -1.580000
         1   -1.580000
         2   -1.922632
         Name: sp, dtype: float64,
         0    1.201919
         1    1.204120
         2    1.387787
         Name: resistivity, dtype: float64)
    >>> # it is obvious the `resistiviy` values is log10 
    >>> # while `sp` stil remains the same 
      
    """
    X = _assert_all_types(X, pd.DataFrame)
    if not callable(ufunc): 
        raise TypeError ("Expect a function for `ufunc`; "
                         f"got {type(ufunc).__name__!r}")
        
    pattern = pattern or r'[#&*@!,;\s]\s*'
    if not is_iterable( columns_to_skip): 
        raise TypeError ("Columns  to skip expect an iterable object;"
                         f" got {type(columns_to_skip).__name__!r}")
        
    if isinstance(columns_to_skip, str):
        columns_to_skip = str2columns (columns_to_skip, pattern=pattern  )
    #assert whether column to skip is in 
    if columns_to_skip:
        cskip = copy.deepcopy(columns_to_skip)
        columns_to_skip = is_in_if(X.columns, columns_to_skip, return_diff= True)
        if len(columns_to_skip) ==len (X.columns): 
            warnings.warn("Value(s) to skip are not detected.")
    elif columns_to_skip is None: 
        columns_to_skip = list(X.columns) 
        
    if inplace : 
        X[columns_to_skip] = X[columns_to_skip].apply (
            ufunc , **kws)
        X.drop (columns = cskip , inplace =True )
        return 
    if not inplace: 
        X0 = X.copy() 
        X0[columns_to_skip] = X0[columns_to_skip].apply (
            ufunc , **kws)
    
        return  X0   
    
def is_depth_in (X, name, columns = None, error= 'ignore'): 
    """ Assert wether depth exists in the data from column attributes.  
    
    If name is an integer value, it assumes to be the index in the columns 
    of the dataframe if not exist , a warming will be show to user. 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
        
    :param columns: list,
        New labels to replace the columns in the dataframe. If given , it 
        should fit the number of colums of `X`. 
        
    :param name: str, int  
        depth name in the dataframe or index to retreive the name of the depth 
        in dataframe 
    :param error: str , default='ignore'
        Raise or ignore when depth is not found in the dataframe. Whe error is 
        set to ``ignore``, a pseudo-depth is created using the lenght of the 
        the dataframe, otherwise a valueError raises.
        
    :return: X, depth 
        Dataframe without the depth columns and depth values.
    """
    
    X= _assert_all_types( X, pd.DataFrame )
    if columns is not None: 
        columns = list(columns)
        if not is_iterable(columns): 
            raise TypeError("columns expects an iterable object."
                            f" got {type (columns).__name__!r}")
        if len(columns ) != len(X.columns): 
            warnings.warn("Cannot rename columns with new labels. Expect "
                          "a size to be consistent with the columns X."
                          f" {len(columns)} and {len(X.columns)} are given."
                          )
        else : 
            X.columns = columns # rename columns
        
    else:  columns = list(X.columns) 
    
    _assert_all_types(name,str, int, float )
    
    # if name is given as indices 
    # collect the name at that index 
    if isinstance (name, (int, float) )  :     
        name = int (name )
        if name > len(columns): 
            warnings.warn ("Name index {name} is out of the columns range."
                           f" Max index of columns is {len(columns)}")
            name = None 
        else : 
            name = columns.pop (name)
    
    elif isinstance (name, str): 
        # find in columns whether a name can be 
        # found. Note that all name does not need 
        # to be written completely 
        # for instance name =depth can retrieved 
        # ['depth_top, 'depth_bottom'] , in that case 
        # the first occurence is selected i.e. 'depth_top'
        n = find_by_regex( 
            columns, pattern=fr'{name}', func=re.search)

        if n is not None:
            name = n[0]
            
        # for consistency , recheck all and let 
        # a warning to user 
        if name not in columns :
            msg = f"Name {name!r} does not match any column names."
            if error =='raise': 
                raise ValueError (msg)

            warnings.warn(msg)
            name =None  
            
    # now create a pseudo-depth 
    # as a range of len X 
    if name is None: 
        if error =='raise':
            raise ValueError ("Depth column not found in dataframe."
                              )
        depth = pd.Series ( np.arange ( len(X)), name ='depth (m)') 
    else : 
        # if depth name exists, 
        # remove it from X  
        depth = X.pop (name ) 
        
    return  X , depth     
    
def smart_label_classifier (
        arr: ArrayLike,  values: Union [float, List[float]]= None ,
        labels: Union [int, str, List[str]] =None, 
        order ='soft', func: _F=None, raise_warn=True): 
    """ map smartly the numeric array into a class labels from a map function 
    or a given fixed values. 
    
    New classes created from the fixed values can be renamed if `labels` 
    are supplied. 
    
    Parameters 
    -------------
    arr: Arraylike 1d, 
        array-like whose items are expected to be categorized. 
        
    values: float, list of float, 
        The threshold item values from which the default categorization must 
        be fixed. 
    labels: int |str| or List of [str, int], 
        The labels values that might be correspond to the fixed values. Note  
        that the number of `fixed_labels` might be consistent with the fixed 
        `values` plus one, otherwise a ValueError shall raise if `order` is 
        set to ``strict``. 
        
    order: str, ['soft'|'strict'], default='soft', 
        If order is ``strict``, the argument passed to `values` must be self 
        contain as item in the `arr`, and raise warning otherwise. 
        
    func: callable, optional 
        Function to map the given array. If given, values dont need to be  
        supply. 
        
    raise_warn: bool, default='True'
        Raise warning message if `order=soft` and the fixed `values` are not 
        found in the `arr`. Also raise warnings, if `labels` arguments does 
        not match the number of class from fixed `values`. 
        
    Returns 
    ----------
    arr: array-like 1d 
        categorized array with the same length as the raw 
        
    Examples
    ----------
    >>> import numpy as np
    >>> from gofast.tools.coreutils import smart_label_classifier
    >>> sc = np.arange (0, 7, .5 ) 
    >>> smart_label_classifier (sc, values = [1, 3.2 ]) 
    array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=int64)
    >>> # rename labels <=1 : 'l1', ]1; 3.2]: 'l2' and >3.2 :'l3'
    >>> smart_label_classifier (sc, values = [1, 3.2 ], labels =['l1', 'l2', 'l3'])
    >>> array(['l1', 'l1', 'l1', 'l2', 'l2', 'l2', 'l2', 'l3', 'l3', 'l3', 'l3',
           'l3', 'l3', 'l3'], dtype=object)
    >>> def f (v): 
            if v <=1: return 'l1'
            elif 1< v<=3.2: return "l2" 
            else : return "l3"
    >>> smart_label_classifier (sc, func= f )
    array(['l1', 'l1', 'l1', 'l2', 'l2', 'l2', 'l2', 'l3', 'l3', 'l3', 'l3',
           'l3', 'l3', 'l3'], dtype=object)
    >>> smart_label_classifier (sc, values = 1.)
    array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    >>> smart_label_classifier (sc, values = 1., labels='l1')  
    array(['l1', 'l1', 'l1', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=object)
    
    """
    name =None 
    from .validator import _is_arraylike_1d 
    if hasattr(arr, "name") and isinstance (arr, pd.Series): 
        name = arr.name 
        
    arr= np.array (arr)  
    
    if not _is_arraylike_1d(arr): 
        raise TypeError ("Expects a one-dimensional array, got array with"
                         f" shape {arr.shape }")
    
    if isinstance (values, str): 
        values = str2columns(values )
    if values is not None: 
        values = is_iterable(values, parse_string =True, transform = True )
    # if (values is not None 
    #     and not is_iterable( values)): 
    #     values =[values ]
        
    if values is not None:
        approx_vs=list()
        values_ =np.zeros ((len(values), ), dtype =float )
        for i, v in enumerate (values ) : 
            try : v= float (v)
            except TypeError as type_error : 
                raise TypeError (
                    f"Value {v} must be a valid number." + str(type_error))
            diff_v = np.abs (arr[~np.isnan(arr)] - v ) 
            
            ix_v = np.argmin (diff_v)
            if order =='strict' and diff_v [ix_v]!=0. :
                raise ValueError (
                    f" Value {v} is missing the array. {v} must be an item"
                    " existing in the array or turn order to 'soft' for"
                    " approximate values selectors. ") 
                
            # skip NaN in the case array contains NaN values 
            values_[i] = arr[~np.isnan(arr)][ix_v] 
            
            if diff_v [ix_v]!=0.: 
                approx_vs.append ((v, arr[~np.isnan(arr)][ix_v]))
          
        if len(approx_vs) !=0 and raise_warn: 
            vv, aa = zip (*approx_vs)
            verb ="are" if len(vv)>1 else "is"
            warnings.warn(f"Values {vv} are missing in the array. {aa} {verb}"
                          f" approximatively used for substituting the {vv}.")
    arr_ = arr.copy () 
    
    #### 
    if (func is None and values is None ): 
        raise TypeError ("'ufunc' cannot be None when the values are not given") 
    
    dfunc =None 

    if values is not None: 
        dfunc = lambda k : _smart_mapper (k, kr = values_ )
    func = func or  dfunc 

    # func_vectorized  =np.vectorize(func ) 
    # arr_ = func_vectorized( arr ) 
    arr_ = pd.Series (arr_, name ='temp').map (func).values 
    
    d={} 
    if labels is not None: 
        labels = is_iterable(labels, parse_string=True, transform =True )
        # if isinstance (labels, str): 
        #     labels = str2columns(labels )
        labels, d = _assert_labels_from_values (
            arr_, values_ , labels , d, raise_warn= raise_warn , order =order 
            )

    arr_ = arr_ if labels is None else ( 
        pd.Series (arr_, name = name or 'tname').map(d))
    
    # if name is None: # for consisteny if labels is None 
    arr_= (arr_.values if labels is not None else arr_
           ) if name is None else pd.Series (arr_, name = name )

    return arr_ 

def _assert_labels_from_values (ar, values , labels , d={}, 
                                raise_warn= True , order ='soft'): 
    """ Isolated part of the :func:`~.smart_label_classifier`"""
    from .validator import _check_consistency_size 

    nlabels = list(np.unique (ar))
    if not is_iterable(labels): 
        labels =[labels]
    if not _check_consistency_size(nlabels, labels, error ='ignore'): 
        if order=='strict':
            verb= "were" if len (labels) > 1 else "was"
            raise TypeError (
                "Expect {len(nlabels)} labels for the {len(values)} values"
                f" renaming. {len(labels)} {verb} given.")
 
        verb ="s are" if len(values)>1 else " is"
        msg = (f"{len(values)} value{verb} passed. Labels for"
                " renaming values expect to be composed of"
                f" {len(values)+1} items i.e. 'number of values"
                " + 1' for pure categorization.")
        ur_classes = nlabels [len(labels):] 
        labels = list(labels ) + ur_classes 
        labels = labels [:len(nlabels)] 
        msg += (f" Class{'es' if len(ur_classes)>1 else ''}"
                f" {smart_format(ur_classes)} cannot be renamed." ) 
        
        if raise_warn: 
            warnings.warn (msg )
        
    d = dict (zip (nlabels , labels ))
    
    return labels, d 

def _smart_mapper (k,   kr , return_dict_map =False ) :
    """ Default  mapping using dict to validate the continue  value 'k' 
    :param k: float, 
        continue value to be framed between `kr`
    :param kr: Tuple, 
        range of fixed values  to categorize  
    :return: int - new categorical class 
    
    :Example: 
    >>> from gofast.tools.coreutils import _smart_mapper 
    >>> _smart_mapper (10000 , ( 500, 1500, 2000, 3500) )
    Out[158]: 4
    >>> _smart_mapper (10000 , ( 500, 1500, 2000, 3500) , return_dict_map=True)
    Out[159]: {0: False, 1: False, 2: False, 3: False, 4: True}
    
    """
    import math 
    if len(kr )==1 : 
        d = {0:  k <=kr[0], 1: k > kr[0]}
    elif len(kr)==2: 
        d = {0: k <=kr[0], 1: kr[0] < k <= kr[1],  2: k > kr[1]} 
    else : 
        d= dict()
        for ii  in range (len(kr) + 1  ): 
            if ii ==0: 
                d[ii]= k <= kr[ii] 
            elif ii == len(kr):
                d[ii] = k > kr [-1] 
            else : 
                d[ii] = kr[ii-1] < k <= kr[ii]

    if return_dict_map: 
        return d 
    
    for v, value in d.items () :
        if value: return v if not math.isnan (v) else np.nan 
        
def hex_to_rgb (c): 
    """ Convert colors Hexadecimal to RGB """
    c=c.lstrip('#')
    return tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) 

def zip_extractor(
    zip_file ,
    samples ='*', 
    ftype=None,  
    savepath = None,
    pwd=None,  
): 
    """ Extract  ZIP archive objects. 
    
    Can extract all or a sample objects when the number of object is passed 
    to the parameter ``samples``. 
    
    .. versionadded:: 0.1.5
    
    Parameters 
    -----------
    zip_file: str
        Full Path to archive Zip file. 
    samples: int, str, default ='*'
       Number of data to retrieve from archive files. This is useful when 
       the archive file contains many data. ``*`` means extract all. 
    savepath: str, optional 
       Path to store the decompressed archived files.
    ftype: str, 
<<<<<<< HEAD
       Is the extension of a specific file to decompress. Indeed, if the 
=======
       Is the extension of a the specific file to decompress. Indeed, if the 
>>>>>>> 10707dcecd7d0da55b83bcf73ae48c1e6659f2f8
       archived files contains many different data formats, specifying the 
       data type would retrieve this specific files from the whole 
       files archieved. 
    pwd: int, optional
      Password to pass if the zip file is encrypted.
      
    Return 
    --------
    objnames: list, 
     List of decompressed objects. 
     
    Examples 
    ----------
    >>> from gofast.tools.coreutils import zip_extractor 
    >>> zip_extractor ('gofast/datasets/data/edis/e.E.zip')
    
    """
    def raise_msg_when ( objn, ft): 
        """ Raise message when None file is detected when the type of 
        of file is supplied. Otherwise return the object collected 
        from this kind of data-types
        """
        objn = [ o for o  in objn if o.endswith (ft)]
        if len(objn)  ==0:
            get_extension = [s.split('.')[-1] for s in objn if '.'  in s ]
            if len(get_extension)==0 : get_extension=['']
            msg = ( "The available file types are {smart_format(get_extension)}"
                   if len(get_extension)!=0 else ''
                   ) 
            raise ValueError (f"None objects in the zip collection of matches"
                              f"the {ft!r}. Available file types are {msg}")
        return objn 
    
    if not os.path.isfile (zip_file ): 
        raise FileExistsError( f"File {os.path.basename(zip_file)!r} does"
                              " not exist. Expect a Path-like object,"
                              f" got {type(zip_file).__name__!r}")
        
    if not os.path.basename(zip_file ).lower().endswith ('.zip'): 
        raise FileNotFoundError("Unrecognized zip-file.")
        
    samples = str(samples) 
    if samples !='*': 
        try :samples = int (samples )
        except: 
            raise ValueError ("samples must be an integer value"
                              f" or '*' not {samples}")

    with ZipFile (zip_file, 'r', ) as zip_obj : 
        objnames = zip_obj.namelist() 
        if samples =='*':
                samples = len(objnames )
            
        if ftype is not None: 
            objnames = raise_msg_when(objn=objnames, ft= ftype) 

        if ( samples >= len(objnames) 
            and ftype is None
            ) :
            zip_obj.extractall( path = savepath , pwd=pwd) 
        else: 
            for zf in objnames [:samples ]: 
                zip_obj.extract ( zf, path = savepath, pwd = pwd)        
    
    return objnames 


def _validate_name_in (name, defaults = '', expect_name= None, 
                         exception = None , deep=False ): 
    """ Assert name in multiples given default names. 
    
    Parameters 
    -----------
    name: str, 
      given name to assert 
    default: list, str, default =''
      default values used for assertion 
    expect_name: str, optional 
      name to return in case assertion is verified ( as ``True``)
    deep: bool, default=False 
      Find item in a litteral default string. If set  to ``True``, 
      `defaults` are joined and check whether an occurence of `name` is in the 
      defaults 
      
    exception: Exception 
      Error to raise if name is not found in the default values. 
      
    Returns
    -------
    name: str, 
      Verified name or boolean if expect name if ``None``. 
      
    Examples 
    -------
    >>> from gofast.tools.coreutils import _validate_name_in 
    >>> dnames = ('NAME', 'FIST NAME', 'SUrname')
    >>> _validate_name_in ('name', defaults=dnames )
    False 
    >>> _validate_name_in ('name', defaults= dnames, deep =True )
    True
    >>> _validate_name_in ('name', defaults=dnames , expect_name ='NAM')
    False 
    >>> _validate_name_in ('name', defaults=dnames , expect_name ='NAM', deep=True)
    'NAM'
    """
    
    name = str(name).lower().strip() 
    defaults = is_iterable(defaults, 
            exclude_string= True, parse_string= True, transform=True )
    if deep : 
        defaults = ''.join([ str(i) for i in defaults] ) 
        
    # if name in defaults: 
    name = ( True if expect_name is None  else expect_name 
            ) if name in defaults else False 
    
    #name = True if name in defaults else ( expect_name if expect_name else False )
    
    if not name and exception: 
        raise exception 
        
    return name 

def get_confidence_ratio (
        ar, 
        axis = 0, 
        invalid = 'NaN',
        mean=False, 
        ):
    
    """ Get ratio of confidence in array by counting the number of 
    invalid values. 
    
    Parameters 
    ------------
    ar: arraylike 1D or 2D  
      array for checking the ratio of confidence 
      
    axis: int, default=0, 
       Compute the ratio of confidence alongside the rows by defaults. 
       
    invalid: int, foat, default='NaN'
      The value to consider as invalid in the data might be listed if 
      applicable. The default is ``NaN``. 
      
    mean: bool, default=False, 
      Get the mean ratio. Average the percentage of each axis. 
      
      .. versionadded:: 0.2.8 
         Average the ratio of confidence of each axis. 
      
    Returns 
    ---------
    ratio: arraylike 1D 
      The ratio of confidence array alongside the ``axis``. 

    Examples 
    ----------
    >>> import numpy as np 
    >>> np.random.seed (0) 
    >>> test = np.random.randint (1, 20 , 10 ).reshape (5, 2 ) 
    >>> test
    array([[13, 16],
           [ 1,  4],
           [ 4,  8],
           [10, 19],
           [ 5,  7]])
    >>> from gofast.tools.coreutils import get_confidence_ratio 
    >>> get_confidence_ratio (test)
    >>> array([1., 1.])
    >>> get_confidence_ratio (test, invalid= ( 13, 19) )
    array([0.8, 0.8])
    >>> get_confidence_ratio (test, invalid= ( 13, 19, 4) )
    array([0.6, 0.6])
    >>> get_confidence_ratio (test, invalid= ( 13, 19, 4), axis =1 )
    array([0.5, 0.5, 0.5, 0.5, 1. ])
    
    """
    def gfc ( ar, inv):
        """ Get ratio in each column or row in the array. """
        inv = is_iterable(inv, exclude_string=True , transform =True, 
                              )
        # if inv!='NaN': 
        for iv in inv: 
            if iv in ('NAN', np.nan, 'NaN', 'nan', None): 
                iv=np.nan  
            ar [ar ==iv] = np.nan 
                
        return len( ar [ ~np.isnan (ar)])  / len(ar )
    
    # validate input axis name 
    axis = _validate_name_in (axis , ('1', 'rows', 'sites', 'stations') ,
                              expect_name=1 )
    if not axis:
        axis =0 
    
    ar = np.array(ar).astype ( np.float64) # for consistency
    ratio = np.zeros(( (ar.shape[0] if axis ==1 else ar.shape [1] )
                      if ar.ndim ==2 else 1, ), dtype= np.float64) 
    
    for i in range (len(ratio)): 
        ratio[i] = gfc ( (ar [:, i] if axis ==0 else ar [i, :])
                        if ar.ndim !=1 else ar , inv= invalid 
                        )
    if mean: 
        ratio = np.array (ratio).mean() 
    return ratio 
    
def assert_ratio(
    v,  bounds: List[float] = None , 
    exclude_value:float= None, 
    in_percent:bool =False , 
    name:str ='rate' 
    ): 
    """ Assert rate value between a specific range. 
    
    Parameters 
    -----------
    v: float, 
       ratio value to assert 
    bounds: list ( lower, upper) 
       The range that value must  be included
    exclude_value: float 
       A value that ``v`` must not taken. Exclude it from the ``bounds``. 
       Raise error otherwise. Note that  any other value will use the 
       lower bound in `bounds` as exlusion. 
       
    in_percent: bool, default=False, 
       Convert the value into a percentage.
       
    name: str, default='rate' 
       the name of the value for assertion. 
       
    Returns
    --------
    v: float 
       Asserted value. 
       
    Examples
    ---------
    >>> from gofast.tools.coreutils import assert_ratio
    >>> assert_ratio('2')
    2.0
    >>> assert_ratio(2 , bounds =(2, 8))
    2.0
    >>> assert_ratio(2 , bounds =(4, 8))
    ValueError:...
    >>> assert_ratio(2 , bounds =(1, 8), exclude_value =2 )
    ValueError: ...
    >>> assert_ratio(2 , bounds =(1, 8), exclude_value ='use bounds' )
    2.0
    >>> assert_ratio(2 , bounds =(0, 1) , in_percent =True )
    0.02
    >>> assert_ratio(2 , bounds =(0, 1) )
    ValueError:
    >>> assert_ratio(2 , bounds =(0, 1), exclude_value ='use lower bound',
                         name ='tolerance', in_percent =True )
    0.02
    """ 
    msg =("greater than {} and less than {}" )
    
    
    if isinstance (v, str): 
        if "%" in v: in_percent=True 
        v = v.replace('%', '')
    try : 
        v = float (v)
    except TypeError : 
        raise TypeError (f"Unable to convert {type(v).__name__!r} "
                         f"to float: {v}")
    except ValueError: 
        raise ValueError(f"Expects 'float' not {type(v).__name__!r}: "
                         f"{(v)!r}")
    # put value in percentage 
    # if greater than 1. 
    if in_percent: 
        if 1 < v <=100: 
            v /= 100. 
          
    bounds = bounds or []
    low, up, *_ = list(bounds) + [ None, None]
    e=("Expects a {} value {}, got: {}".format(
            name , msg.format(low, up), v)) 
    err = ValueError (e)

    if len(bounds)!=0:
        if ( 
            low is not None  # use is not None since 0. is
            and up is not None # consider as False value
            and  (v < low or v > up)
            ) :
                raise err 
        
    if exclude_value is not None: 
        try : 
            low = float (str(exclude_value))
        except : # use bounds
            pass 
        if low is None:
            warnings.warn("Cannot exclude the lower value in the interval"
                          " while `bounds` argument is not given.")
        else:  
            if v ==low: 
                raise ValueError (e.replace (", got:", ' excluding') + ".")
            
    if in_percent and v > 100: 
         raise ValueError ("{} value should be {}, got: {}".
                           format(name.title(), msg.format(low, up), v  ))
    return v 

def validate_ratio(
    value: float, 
    bounds: Optional[Tuple[float, float]] = None, 
    exclude: Optional[float] = None, 
    to_percent: bool = False, 
    param_name: str = 'value'
) -> float:
    """Validates and optionally converts a value to a percentage within 
    specified bounds, excluding specific values.

    Parameters:
    -----------
    value : float or str
        The value to validate and convert. If a string with a '%' sign, 
        conversion to percentage is attempted.
    bounds : tuple of float, optional
        A tuple specifying the lower and upper bounds (inclusive) for the value. 
        If None, no bounds are enforced.
    exclude : float, optional
        A specific value to exclude from the valid range. If the value matches 
        'exclude', a ValueError is raised.
    to_percent : bool, default=False
        If True, the value is converted to a percentage 
        (assumed to be in the range [0, 100]).
    param_name : str, default='value'
        The parameter name to use in error messages.

    Returns:
    --------
    float
        The validated (and possibly converted) value.

    Raises:
    ------
    ValueError
        If the value is outside the specified bounds, matches the 'exclude' 
        value, or cannot be converted as specified.
    """
    if isinstance(value, str) and '%' in value:
        to_percent = True
        value = value.replace('%', '')
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Expected a float, got {type(value).__name__}: {value}")

    if to_percent and 0 < value <= 100:
        value /= 100

    if bounds:
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{param_name} must be between {bounds[0]}"
                             f" and {bounds[1]}, got: {value}")
    
    if exclude is not None and value == exclude:
        raise ValueError(f"{param_name} cannot be {exclude}")

    if to_percent and value > 1:
        raise ValueError(f"{param_name} converted to percent must"
                         f" not exceed 1, got: {value}")

    return value

def exist_features (df, features, error='raise', name="Feature"): 
    """Control whether the features exist or not.  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param error: str - raise if the features don't exist in the dataframe. 
        *default* is ``raise`` and ``ignore`` otherwise. 
        
    :return: bool 
        assert whether the features exists 
    """
    isf = False  
    
    error= 'raise' if error.lower().strip().find('raise')>= 0  else 'ignore' 

    if isinstance(features, str): 
        features =[features]
        
    features = _assert_all_types(features, list, tuple, np.ndarray)
    set_f =  set (features).intersection (set(df.columns))
    if len(set_f)!= len(features): 
        nfeat= len(features) 
        msg = f"{name}{'s' if nfeat >1 else ''}"
        if len(set_f)==0:
            if error =='raise':
                raise ValueError (f"{msg} {smart_format(features)} "
                                  f"{'does not' if nfeat <2 else 'dont'}"
                                  " exist in the dataframe")
            isf = False 
        # get the difference 
        diff = set (features).difference(set_f) if len(
            features)> len(set_f) else set_f.difference (set(features))
        nfeat= len(diff)
        if error =='raise':
            raise ValueError(f"{msg} {smart_format(diff)} not found in"
                             " the dataframe.")
        isf = False  
    else : isf = True 
    
    return isf    
    


def random_selector (
        arr:ArrayLike, value: Union [float, ArrayLike], 
        seed: int = None, shuffle =False ): 
    """Randomly select the number of values in array. 
    
    Parameters
    ------------
    arr: ArrayLike 
       Array of values 
    value: float, arraylike 
        If ``float`` value is passed, it indicates the number of values to 
        select among the length of `arr`. If array (``value``) is passed, it
        should be self contain in the given ``arr`. However if ``string`` is 
        given and contain the ``%``, it calculates the ratio of 
        number to randomly select. 
    seed: int, Optional 
       Allow retrieving the identical value randomly selected in the given 
       array. 
       
    suffle: bool, False 
       If  ``True`` , shuffle the selected values. 
       
    Returns 
    --------
    arr: Array containing the selected values 
     
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import random_selector 
    >>> dat= np.arange (42 ) 
    >>> random_selector (dat , 7, seed = 42 ) 
    array([0, 1, 2, 3, 4, 5, 6])
    >>> random_selector ( dat, ( 23, 13 , 7))
    array([ 7, 13, 23])
    >>> random_selector ( dat , "7%", seed =42 )
    array([0, 1])
    >>> random_selector ( dat , "70%", seed =42 , shuffle =True )
    array([ 0,  5, 20, 25, 13,  7, 22, 10, 12, 27, 23, 21, 16,  3,  1, 17,  8,
            6,  4,  2, 19, 11, 18, 24, 14, 15,  9, 28, 26])
    """
    
    msg = "Non-numerical is not allowed. Got {!r}."
    
    if seed: 
        seed = _assert_all_types(seed , int, float, objname ='Seed')
        np.random.seed (seed ) 
       
    v = copy.deepcopy(value )
    
    if not is_iterable( value, exclude_string= True ):
        
        value = str(value )
        
        if '%' in  value: 
            try: 
               value = float( value.replace ('%', '')) /100 
            except : 
                raise TypeError(msg.format(v))
            # get the number 
            value *= len(arr )
                
        
        try : 
            value = int(value )
            
        except :
            raise TypeError (msg.format(v))
    
        if value > len(arr): 
            raise ValueError(f"Number {value} is out of the range."
                             f" Expect value less than {len(arr)}.")
            
        value = np.random.permutation(value ) 
        
    arr = np.array ( 
        is_iterable( arr, exclude_string=True, transform =True )) 
    
    arr = arr.ravel() if arr.ndim !=1 else arr 

    mask = _isin (arr, value , return_mask= True )
    arr = arr [mask ] 
    
    if shuffle : np.random.shuffle (arr )

    return arr

def cleaner(
    data: Union[DataFrame, NDArray],
    columns: List[str] = None,
    inplace: bool = False,
    labels: List[Union[int, str]] = None,
    func: _F = None,
    mode: str = 'clean',
    **kws
) -> Union[DataFrame, NDArray, None]:
    """ Sanitize data or columns by dropping specified labels 
    from rows or columns. 
    
    If data is not a pandas dataframe, should be converted to 
    dataframe and uses index to drop the labels. 
    
    Parameters 
    -----------
    data: pd.Dataframe or arraylike2D. 
       Dataframe pandas or Numpy two dimensional arrays. If 2D array is 
       passed, it should prior be converted to a daframe by default and 
       drop row index from index parameters 
       
    columns: single label or list-like
        Alternative to specifying axis (
            labels, axis=1 is equivalent to columns=labels).

    labels: single label or list-like
      Index or column labels to drop. A tuple will be used as a single 
      label and not treated as a list-like.

    func: _F, callable 
        Universal function used to clean the columns. If performs only when 
        `mode` is on ``clean`` option. 
        
    inplace: bool, default False
        If False, return a copy. Otherwise, do operation 
        inplace and return None.
       
    mode: str, default='clean' 
       Options or mode of operation to do on the data. It could 
       be ['clean'|'drop']. If ``drop``, it behaves like ``dataframe.drop`` 
       of pandas. 
       
    Returns
    --------
    DataFrame, array2D  or None
            DataFrame cleaned or without the removed index or column labels 
            or None if inplace=True or array is data is passed as an array. 
            
    """
    mode = _validate_name_in(mode , defaults =("drop", 'remove' ), 
                      expect_name ='drop')
    if not mode: 
        return sanitize_frame_cols(
            data, 
            inplace = inplace, 
            func = func 
            ) 
 
    objtype ='ar'
    if not hasattr (data , '__array__'): 
        data = np.array (data ) 
        
    if hasattr(data , "columns"): 
        objtype = "pd" 
    
    if objtype =='ar': 
        data = pd.DataFrame(data ) 
        # block inplace to False and 
        # return numpy ar 
        inplace = False 
    # if isinstance(columns , str): 
    #     columns = str2columns(columns ) 
    if columns is not None: 
        columns = is_iterable(
            columns, exclude_string=True ,
            parse_string= True, 
            transform =True )
        
    data = data.drop (labels = labels, 
                      columns = columns, 
                      inplace =inplace,  
                       **kws 
                       ) 
    # re-verify integrity 
    # for consistency
    data = to_numeric_dtypes(data )
    return np.array ( data ) if objtype =='ar' else data 
 
def rename_files(
    src_files: Union[str, List[str]], 
    dst_files: Union[str, List[str]], 
    basename: Optional[str] = None, 
    extension: Optional[str] = None, 
    how: str = 'py', 
    prefix: bool = True, 
    keep_copy: bool = True, 
    trailer: str = '_', 
    sortby: Union[re.Pattern, _F] = None, 
    **kws
) -> None:
    """Rename files in directory.

    Parameters 
    -----------
    src_files: str, Path-like object 
       Source files to rename 
      
    dst_files: str of PathLike object 
       Destination files renamed. 
       
    extension: str, optional 
       If a path is given in `src_files`, specifying the `extension` will just 
       collect only files with this typical extensions. 
       
    basename: str, optional 
       If `dst_files` is passed as Path-object, name should be needed 
       for a change, otherwise, the number is incremented using the Python 
       index counting defined by the parameter ``how=py` 
        
    how: str, default='py' 
       The way to increment files when `dst_files` is given as a Path object. 
       For instance, for a  ``name=E_survey`` and ``prefix==True``, the first 
       file should be ``E_survey_00`` if ``how='py'`` otherwise it should be 
       ``E_survey_01``.
     
    prefix: bool, default=True
      Prefix is used to position the name before the number incrementation. 
      If ``False`` and `name` is given, the number is positionning before the 
      name. If ``True`` and not `prefix` for a ``name=E_survey``, it should be 
      ``00_E_survey`` and ``01_E_survey``. 

    keep_copy: bool, default=True 
       Keep a copy of the source files. 
       
    trailer: str, default='_', 
       Item used to separate the basename for counter. 
       
    sortby: Regex or Callable, 
       Key to sort the collection of the items when `src_files` is passed as 
       a path-like object.  This is usefull to keep order as the origin files 
       especially  when files includes a specific character.  Furthermore 
       [int| float |'num'|'digit'] sorted the files according to the
       number included in the filename if exists. 

    kws: dict 
       keyword arguments passed to `os.rename`. 

    """ 
    dest_dir =None ; trailer = str(trailer)
    extension = str(extension).lower()
    
    if os.path.isfile (src_files ): 
        src_files = [src_files ] 
        
    elif os.path.isdir (src_files): 
        src_path = src_files
        ldir = os.listdir(src_path) 

        src_files = ldir if extension =='none' else [
             f for f in ldir  if  f.endswith (extension) ]
    
        if sortby: 
            if sortby in ( int, float, 'num', 'number', 'digit'): 
                src_files = sorted(ldir, key=lambda s:int( re.search(
                    '\d+', s).group()) if re.search('\d+', s) else 0 )
            else: 
                src_files = sorted(ldir, key=sortby)

        src_files = [  os.path.join(src_path, f )   for f in src_files  ] 
        # get only the files 
        src_files = [ f for f in src_files if os.path.isfile (f ) ]

    else : raise FileNotFoundError(f"{src_files!r} not found.") 
    
    # Create the directory if it doesn't exist
    if ( dst_files is not None 
        and not os.path.exists (dst_files)
        ): 
        os.makedirs(dst_files)
        
    if os.path.isdir(dst_files): 
        dest_dir = dst_files 

    dst_files = is_iterable(dst_files , exclude_string= True, transform =True ) 
    # get_extension of the source_files 
    _, ex = os.path.splitext (src_files[0]) 
    
    if dest_dir: 
        if basename is None: 
            warnings.warn(
                "Missing basename for renaming file. Should use `None` instead.")
            basename =''; trailer =''
            
        basename= str(basename)
        if prefix: 
            dst_files =[ f"{str(basename)}{trailer}" + (
                f"{i:03}" if how=='py' else f"{i+1:03}") + f"{ex}"
                        for i in range (len(src_files))]
        elif not prefix: 
            dst_files =[ (f"{i:03}" if how=='py' else f"{i+1:03}"
                        ) +f"{trailer}{str(basename)}" +f"{ex}"
                        for i in range (len(src_files))]
        
        dst_files = [os.path.join(dest_dir , f) for f in dst_files ] 
    
    for f, nf in zip (src_files , dst_files): 
        try: 
           if keep_copy : shutil.copy (f, nf , **kws )
           else : os.rename (f, nf , **kws )
        except FileExistsError: 
            os.remove(nf)
            if keep_copy : shutil.copy (f, nf , **kws )
            else : os.rename (f, nf , **kws )
            
            
def get_xy_coordinates (d, as_frame = False, drop_xy = False, 
                        raise_exception = True, verbose=0 ): 
    """Check whether the coordinate values x, y exist in the data.
    
    Parameters 
    ------------
    d: Dataframe 
       Frame that is expected to contain the longitude/latitude or 
       easting/northing coordinates.  Note if all types of coordinates are
       included in the data frame, the longitude/latitude takes the 
       priority. 
    as_frame: bool, default= False, 
       Returns the coordinates values if included in the data as a frame rather 
       than computing the middle points of the line 
    drop_xy: bool, default=False, 
       Drop the coordinates in the data and return the data transformed inplace 
       
    raise_exception: bool, default=True 
       raise error message if data is not a dataframe. If set to ``False``, 
       exception is converted to a warning instead. To mute the warning set 
       `raise_exception` to ``mute``
       
    verbose: int, default=0 
      Send message whether coordinates are detected. 
         
    returns 
    --------
    xy, d, xynames: Tuple 
      xy : tuple of float ( longitude, latitude) or (easting/northing ) 
         if `as_frame` is set to ``True``. 
      d: Dataframe transformed (coordinated removed )  or not
      xynames: str, the name of coordinates detected. 
      
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from gofast.tools.coreutils import get_xy_coordinates 
    >>> testdata = gf.make_erp ( n_stations =7, seed =42 ).frame 
    >>> xy, d, xynames = get_xy_coordinates ( testdata,  )
    >>> xy , xynames 
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
        longitude   latitude        easting      northing
    0  110.485833  26.051389  448565.380621  2.881476e+06
    1  110.485982  26.051577  448580.339199  2.881497e+06
    >>> # remove longitude and  lat in data 
    >>> testdata = testdata.drop (columns =['longitude', 'latitude']) 
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
             easting      northing
    0  448565.380621  2.881476e+06
    1  448580.339199  2.881497e+06
    >>> # note testdata should be transformed inplace when drop_xy is set to True
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    ((448610.25612032827, 2881538.4380570543), ('easting', 'northing'))
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> testdata.head(2) # coordinates are henceforth been dropped 
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    (None, ())
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5

    """   
    
    def get_value_in ( val,  col , default): 
        """ Get the value in the frame columns if `val` exists in """
        x = list( filter ( lambda x: x.find (val)>=0 , col)
                   )
        if len(x) !=0: 
            # now rename col  
            d.rename (columns = {x[0]: str(default) }, inplace = True ) 
            
        return d

    if not (
            hasattr ( d, 'columns') and hasattr ( d, '__array__') 
            ) : 
        emsg = ("Expect dataframe containing coordinates longitude/latitude"
                f" or easting/northing. Got {type (d).__name__!r}")
        
        raise_exception = str(raise_exception).lower().strip() 
        if raise_exception=='true': 
            raise TypeError ( emsg )
        
        if raise_exception  not in ('mute', 'silence'):  
            warnings.warn( emsg )
       
        return d 
    
    # check whether coordinates exists in the data columns 
    for name, tname in zip ( ('lat', 'lon', 'east', 'north'), 
                     ( 'latitude', 'longitude', 'easting', 'northing')
                     ) : 
        d = get_value_in(name, col = d.columns , default = tname )
       
    # get the exist coodinates 
    coord_columns  = []
    for x, y in zip ( ( 'longitude', 'easting' ), ( 'latitude', 'northing')):
        if ( x  in d.columns and y in d.columns ): 
            coord_columns.extend  ( [x, y] )

    xy  = d[ coord_columns] if len(coord_columns)!=0 else None 

    if ( not as_frame 
        and xy is not None ) : 
        # take the middle of the line and if both types of 
        # coordinates are supplied , take longitude and latitude 
        # and drop easting and northing  
        xy = tuple ( np.nanmean ( np.array ( xy ) , axis =0 )) [:2]

    xynames = tuple ( coord_columns)[:2]
    if ( 
            drop_xy  and len( coord_columns) !=0
            ): 
        # modifie the data inplace 
        d.drop ( columns=coord_columns, inplace = True  )

    if verbose: 
        print("###", "No" if len(xynames)==0 else ( 
            tuple (xy.columns) if as_frame else xy), "coordinates found.")
        
    return  xy , d , xynames 
       

def pair_data(
    *d: Union[DataFrame, List[DataFrame]],  
    on: Union[str, List[str]] = None, 
    parse_on: bool = False, 
    mode: str = 'strict', 
    coerce: bool = False, 
    force: bool = False, 
    decimals: int = 7, 
    raise_warn: bool = True 
) -> DataFrame:
    """ Find indentical object in all data and concatenate them using merge 
     intersection (`cross`) strategy.
    
    Parameters 
    ---------- 
    d: List of DataFrames 
       List of pandas DataFrames 
    on: str, label or list 
       Column or index level names to join on. These must be found in 
       all DataFrames. If `on` is ``None`` and not merging on indexes then 
       a concatenation along columns axis is performed in all DataFrames. 
       Note that `on` works with `parse_on` if its argument is  a list of 
       columns names passed into single litteral string. For instance:: 
           
        on ='longitude latitude' --[parse_on=True]-> ['longitude' , 'latitude'] 
        
    parse_on: bool, default=False 
       Parse `on` arguments if given as string and return_iterable objects. 
       
    mode: str, default='strict' 
      Mode to the data. Can be ['soft'|'strict']. In ``strict`` mode, all the 
      data passed must be a DataFrame, otherwise an error raises. in ``soft``
      mode, ignore the non-DataFrame. Note that any other values should be 
      in ``strict`` mode. 
      
    coerce: bool, default=False 
       Truncate all DataFrame size to much the shorter one before performing 
       the ``merge``. 
        
    force: bool, default=False, 
       Force `on` items to be in the all DataFrames, This could be possible 
       at least, `on` items should be in one DataFrame. If missing in all 
       data, an error occurs.  
 
    decimals: int, default=5 
       Decimal is used for comparison between numeric labels in `on` columns 
       items. If set, it rounds values of `on` items in all data before 
       performing the merge. 
       
     raise_warn: bool, default=False 
        Warn user to concatenate data along column axis if `on` is ``None``. 

    Returns 
    --------
    data: DataFrames 
      A DataFrame of the merged objects.
      
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from gofast.tools.coreutils import pair_data 
    >>> data = gf.make_erp (seed =42 , n_stations =12, as_frame =True ) 
    >>> table1 = gf.DCProfiling ().fit(data).summary()
    >>> table1 
           dipole   longitude  latitude  ...  shape  type       sfi
    line1      10  110.486111  26.05174  ...      C    EC  1.141844
    >>> data_no_xy = gf.make_ves ( seed=0 , as_frame =True) 
    >>> data_no_xy.head(2) 
        AB   MN  resistivity
    0  1.0  0.4   448.860148
    1  2.0  0.4   449.060335
    >>> data_xy = gf.make_ves ( seed =0 , as_frame =True , add_xy =True ) 
    >>> data_xy.head(2) 
        AB   MN  resistivity   longitude  latitude
    0  1.0  0.4   448.860148  109.332931  28.41193
    1  2.0  0.4   449.060335  109.332931  28.41193
    >>> table = gf.methods.VerticalSounding (
        xycoords = (110.486111,   26.05174)).fit(data_no_xy).summary() 
    >>> table.table_
             AB    MN   arrangememt  ... nareas   longitude  latitude
    area                             ...                             
    None  200.0  20.0  schlumberger  ...      1  110.486111  26.05174
    >>> pair_data (table1, table.table_,  ) 
           dipole   longitude  latitude  ...  nareas   longitude  latitude
    line1    10.0  110.486111  26.05174  ...     NaN         NaN       NaN
    None      NaN         NaN       NaN  ...     1.0  110.486111  26.05174
    >>> pair_data (table1, table.table_, on =['longitude', 'latitude'] ) 
    Empty DataFrame 
    >>> # comments: Empty dataframe appears because, decimal is too large 
    >>> # then it considers values longitude and latitude differents 
    >>> pair_data (table1, table.table_, on =['longitude', 'latitude'], decimals =5 ) 
        dipole  longitude  latitude  ...  max_depth  ohmic_area  nareas
    0      10  110.48611  26.05174  ...      109.0  690.063003       1
    >>> # Now is able to find existing dataframe with identical closer coordinates. 
    
    """
    from .validator import _is_numeric_dtype  

    if str(mode).lower()=='soft': 
        d = [ o for  o in d if hasattr (o, '__array__') and hasattr (o, 'columns') ]
    
    is_same = set ( [ hasattr (o, '__array__') 
                     and hasattr (o, 'columns') for o in d ] ) 
    
    if len(is_same)!=1 or not list (is_same) [0]: 
        types = [ type(o).__name__ for o in d ]
        raise TypeError (
            f"Expect DataFrame. Got {smart_format(types)}")
    
    same_len = [len(o) for o in d] 
    
    if len( set(same_len)) !=1 or not list(set(same_len))[0]: 
        if not coerce: 
            raise ValueError(
                f"Data must be a consistent size. Got {smart_format(same_len)}"
                " respectively. Set ``coerce=True`` to truncate the data"
                " to match the shorter data length.")
        # get the shorthest len 
        min_index = min (same_len) 
        d = [ o.iloc [:min_index, :  ]  for o in d ] 

    if on is None:
        if raise_warn: 
            warnings.warn("'twin_items' are missing in the data. A simple merge"
                          " along the columns axis should be performed.") 
        
        return pd.concat ( d, axis = 1 )
    
    # parse string 
    on= is_iterable(on, exclude_string= True , 
                    transform =True, parse_string= parse_on  
                    )
    
    feature_exist = [
        exist_features(o, on, error = 'ignore'
                       ) for o in d ]  

    if ( len( set (feature_exist) )!=1 
        or not list(set(feature_exist)) [0]
            ): 
        if not force: 
            raise ValueError(
                f"Unable to fit the data. Items {smart_format(on)} are"
                f" missing in the data columns. {smart_format(on)} must"
                 " include in the data columns. Please check your data.")

        # seek the value twin_items in the data and use if for all  
        dtems =[] ;  repl_twd= None 
        for o in d: 
            # select one valid data that contain the 
            # twin items
            try : 
                exist_features ( o, on ) 
            except : 
                pass 
            else:
                repl_twd = o [ on ]
                break 
            
        if repl_twd is None: 
            raise ValueError("To force data that have not consistent items,"
                             f" at least {smart_format(on)} items must"
                             " be included in one DataFrame.")
        # make add twin_data if 
        for o in d : 
            try : exist_features(o, on) 
            except : 
                a = o.copy()  
                a [ on ] = repl_twd 
            else : a = o.copy () 
            
            dtems.append(a)
        # reinitialize d
        d = dtems  
    
    # check whether values to merge are numerics 
    # if True, use decimals to round values to consider 
    # that identic 
    # round value if value before performed merges 
    # test single data with on 
    is_num = _is_numeric_dtype (d[0][on] ) 
    if is_num: 
        decimals = int (_assert_all_types(
            decimals, int, float, objname ='Decimals'))
        d_ = []
        for o in d : 
            a = o.copy()  
            a[on ] = np.around (o[ on ].values, decimals ) 
            d_.append (a )
        # not a numerick values so stop     
        d =d_

    # select both two  
    data = pd.merge (* d[:2], on= on ) 
    
    if len(d[2:]) !=0: 
        for ii, o in enumerate ( d[2:]) : 
            data = pd.merge ( *[data, o] , on = on, suffixes= (
                f"_x{ii+1}", f"_y{ii+1}")) 
            
    return data 

def read_worksheets(*data): 
    """ Read sheets and returns a list of DataFrames and sheet names. 
    
    Parameters 
    -----------
    data: list of str 
      A collection of excel sheets files. Read only `.xlsx` files. Any other 
      files raises an errors.  
    
    Return
    ------
    data, sheet_names: Tuple of DataFrames and sheet_names 
       A collection of DataFrame and sheets names. 
       
    Examples 
    -----------
    >>> import os 
    >>> from gofast.tools.coreutils import read_worksheets 
    >>> sheet_file= r'_F:\repositories\gofast\data\erp\sheets\gbalo.xlsx'
    >>> data, snames =  read_worksheets (sheet_file )
    >>> snames 
    ['l11', 'l10', 'l02'] 
    >>> data, snames =  read_worksheets (os.path.dirname (sheet_file))
    >>> snames 
    ['l11', 'l10', 'l02', 'l12', 'l13']
    
    """
    dtem = []
    data = [o for o in data if isinstance ( o, str )]
    
    for o in data: 
        if os.path.isdir (o): 
            dlist = os.listdir (o)
            # collect only the excell sheets 
            p = [ os.path.join(o, f) for f in dlist if f.endswith ('.xlsx') ]
            dtem .extend(p)
        elif os.path.isfile (o):
            _, ex = os.path.splitext( o)
            if ex == '.xlsx': 
                dtem.append(o)
            
    data = copy.deepcopy(dtem)
    # if no excel sheets is found return None 
    if len(data) ==0: 
        return None, None 
    
    # make d dict to collect data 
    ddict = dict() 
    regex = re.compile (r'[$& #@%^!]', flags=re.IGNORECASE)
    
    for d in data : 
        try: 
            ddict.update ( **pd.read_excel (d , sheet_name =None))
        except : pass 

    #collect stations names
    if len(ddict)==0 : 
        raise TypeError("Can'find the data to read.")

    sheet_names = list(map(
        lambda o: regex.sub('_', o).lower(), ddict.keys()))

    data = list(ddict.values ()) 

    return data, sheet_names      
 
def key_checker (
    keys: str ,   
    valid_keys:List[str], 
    regex:re = None, 
    pattern:str = None , 
    deep_search:bool =...
    ): 
    """check whether a give key exists in valid_keys and return a list if 
    many keys are found.
    
    Parameters 
    -----------
    keys: str, list of str 
       Key value to find in the valid_keys 
       
    valid_keys: list 
       List of valid keys by default. 
       
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&*@!_,;\s-]\s*', flags=re.IGNORECASE)
            
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    deep_search: bool, default=False 
       If deep-search, the key finder is no sensistive to lower/upper case 
       or whether a numeric data is included. 
 
       
    Returns 
    --------
    keys: str, list , 
      List of keys that exists in the `valid_keys`. 
      
    Examples
    --------
    
    >>> from gofast.tools.coreutils import key_checker
    >>> key_checker('h502', valid_keys= ['h502', 'h253','h2601'])  
    Out[68]: 'h502'
    >>> key_checker('h502+h2601', valid_keys= ['h502', 'h253','h2601'])
    Out[69]: ['h502', 'h2601']
    >>> key_checker('h502 h2601', valid_keys= ['h502', 'h253','h2601'])
    Out[70]: ['h502', 'h2601']
    >>> key_checker(['h502',  'h2601'], valid_keys= ['h502', 'h253','h2601'])
    Out[73]: ['h502', 'h2601']
    >>> key_checker(['h502',  'h2602'], valid_keys= ['h502', 'h253','h2601'])
    UserWarning: key 'h2602' is missing in ['h502', 'h2602']
    Out[82]: 'h502'
    >>> key_checker(['502',  'H2601'], valid_keys= ['h502', 'h253','h2601'], 
                    deep_search=True )
    Out[57]: ['h502', 'h2601']
    
    """
    deep_search, =ellipsis2false(deep_search)

    _keys = copy.deepcopy(keys)
    valid_keys = is_iterable(valid_keys , exclude_string =True, transform =True )
    if isinstance ( keys, str): 
        pattern = pattern or '[_#&@!_+,;\s-]\s*'
        keys = str2columns (keys, regex = regex , pattern=pattern )
    # If iterbale object , save obj 
    # to improve error 
    kkeys = copy.deepcopy(keys)
    if deep_search: 
        keys = key_search(
            keys, 
            default_keys= valid_keys,
            deep=True, 
            raise_exception= True,
            regex =regex, 
            pattern=pattern 
            )
        return keys[0] if len(keys)==1 else keys 
    # for consistency 
    keys = [ k for k in keys if ''.join(
        [ str(i) for i in valid_keys] ).find(k)>=0 ]
    # assertion error if key does not exist. 
    if len(keys)==0: 
        verb1, verb2 = ('', 'es') if len(kkeys)==1 else ('s', '') 
        msg = (f"key{verb1} {_keys!r} do{verb2} not exist."
               f" Expect {smart_format(valid_keys, 'or')}")
        raise KeyError ( msg )
        
    if len(keys) != len(kkeys):
        miss_keys = is_in_if ( kkeys, keys , return_diff= True , error ='ignore')
        miss_keys, verb = (miss_keys[0], 'is') if len( miss_keys) ==1 else ( 
            miss_keys, 'are')
        warnings.warn(f"key{'' if verb=='is' else 's'} {miss_keys!r} {verb}"
                      f" missing in {_keys}")
    keys = keys[0] if len(keys)==1 else keys 
    
    return keys 

def random_sampling (
    d,  
    samples:int = None  , 
    replace:bool = False , 
    random_state:int = None, 
    shuffle=True, 
    ): 
    """ Sampling data. 
    
    Parameters 
    ----------
    d: {array-like, sparse matrix} of shape (n_samples, n_features)
      Data for sampling, where `n_samples` is the number of samples and
      `n_features` is the number of features.
    samples: int,optional 
       Ratio or number of items from axis to return. 
       Default = 1 if `samples` is ``None``.
       
    replace: bool, default=False
       Allow or disallow sampling of the same row more than once. 
       
    random_state: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    shuffle:bool, default=True 
       Shuffle the data before sampling 
      
    Returns 
    ----------
    d: {array-like, sparse matrix} of shape (n_samples, n_features)
    samples data based on the given samples. 
    
    Examples
    ---------
    >>> from gofast.tools.coreutils import random_sampling 
    >>> from gofast.datasets import load_hlogs 
    >>> data= load_hlogs().frame
    >>> random_sampling( data, samples = 7 ).shape 
    (7, 27)
    """

    n= None ; is_percent = False
    orig= copy.deepcopy(samples )
    if not hasattr(d, "__iter__"): 
        d = is_iterable(d, exclude_string= True, transform =True )
    
    if ( 
            samples is None 
            or str(samples) in ('1', '*')
            ): 
        samples =1. 
        
    if "%" in str(samples): 
        samples = samples.replace ("%", '')
        is_percent=True 
    # assert value for consistency. 
    try: 
        samples = float( samples)
    except: 
        raise TypeError("Wrong value for 'samples'. Expect an integer."
                        f" Got {type (orig).__name__!r}")

    if samples <=1 or is_percent: 
        samples  = assert_ratio(
            samples , bounds = (0, 1), exclude_value= 'use lower bound', 
            in_percent= True )
        
        n = int ( samples * ( d.shape[0] if scipy.sparse.issparse(d)
                 else len(d))) 
    else: 
        # data frame 
        n= int(samples)
        
    # reset samples and use number of samples instead    
    samples =None  
    # get the total length of d
    dlen = ( d.shape[0] if scipy.sparse.issparse(d) else len(d))
    # if number is greater than the length 
    # block to the length so to retrieve all 
    # value no matter the arrangement. 
    if n > dlen: 
        n = dlen
    if hasattr (d, 'columns') or hasattr (d, 'name'): 
        # data frame 
        return d.sample ( n= n , frac=samples , replace = replace ,
                         random_state = random_state  
                     ) if shuffle else d.iloc [ :n , ::] 
        
    np.random.seed ( random_state)
    if scipy.sparse.issparse(d) : 
        if scipy.sparse.isspmatrix_coo(d): 
            warnings.warn("coo_matrix does not support indexing. Conversion"
                          " should be performed in CSR matrix")
            d = d.tocsr() 
            
        return d [ np.random.choice(
            np.arange(d.shape[0]), n, replace=replace )] if shuffle else d [
                [ i for i in range (n)]]
                
        #d = d[idx ]
        
    # manage the data 
    if not hasattr(d, '__array__'): 
        d = np.array (d ) 
        
    idx = np.random.randint( len(d), size = n ) if shuffle else [ i for i in range(n)]
    if len(d.shape )==1: d =d[idx ]
    else: d = d[idx , :]
        
    return d 


def make_obj_consistent_if ( 
        item= ... , default = ..., size =None, from_index: bool =True ): 
    """Combine default values to item to create default consistent iterable 
    objects. 
    
    This is valid if  the size of item does not fit the number of 
    expected iterable objects.     
    
    Parameters 
    ------------
    item : Any 
       Object to construct it default values 
       
    default: Any 
       Value to hold in the case the items does not match the size of given items 
       
    size: int, Optional 
      Number of items to return. 
      
    from_index: bool, default=True 
       make an item size to match the exact size of given items 
       
    Returns 
    -------
       item: Iterable object that contain default values. 
       
    Examples 
    ----------
    >>> from gofast.tools.coreutils import make_obj_consistent_if
    >>> from gofast.exlib import SVC, LogisticRegression, XGBClassifier 
    >>> classifiers = ["SVC", "LogisticRegression", "XGBClassifier"] 
    >>> classifier_names = ['SVC', 'LR'] 
    >>> make_obj_consistent_if (classifiers, default = classifier_names ) 
    ['SVC', 'LogisticRegression', 'XGBClassifier']
    >>> make_obj_consistent_if (classifier_names, from_index =False  )
    ['SVC', 'LR']
    >>> >>> make_obj_consistent_if ( classifier_names, 
                                     default= classifiers, size =3 , 
                                     from_index =False  )
    ['SVC', 'LR', 'SVC']
    
    """
    if default==... or None : default =[]
    # for consistency 
    default = list( is_iterable (default, exclude_string =True,
                                 transform =True ) ) 
    
    if item not in ( ...,  None) : 
         item = list( is_iterable( item , exclude_string =True ,
                                  transform = True ) ) 
    else: item = [] 
    
    item += default[len(item):] if from_index else default 
    
    if size is not None: 
        size = int (_assert_all_types(size, int, float,
                                      objname = "Item 'size'") )
        item = item [:size]
        
    return item
    
def replace_data(
    X:Union [ArrayLike, DataFrame], 
    y: Union [ArrayLike, Series] = None, 
    n: int = 1, 
    axis: int = 0, 
    reset_index: bool = False,
    include_original: bool = False,
    random_sample: bool = False,
    shuffle: bool = False
) -> Union [ ArrayLike, DataFrame , Tuple[ArrayLike , DataFrame, ArrayLike, Series]]:
    """
    Duplicates the data `n` times along a specified axis and applies various 
    optional transformations to augment the data suitability for further 
    processing or analysis.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        The input data to process. Sparse matrices are not supported.
    y : Optional[Union[np.ndarray, pd.Series]], optional
        Additional target data to process alongside `X`. Default is None.
    n : int, optional
        The number of times to replicate the data. Default is 1.
    axis : int, optional
        The axis along which to concatenate the data. Default is 0.
    reset_index : bool, optional
        If True and `X` is a DataFrame, resets the index without adding
        the old index as a column. Default is False.
    include_original : bool, optional
        If True, the original data is included in the output alongside
        the replicated data. Default is False.
    random_sample : bool, optional
        If True, samples from `X` randomly with replacement. Default is False.
    shuffle : bool, optional
        If True, shuffles the concatenated data. Default is False.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame, Tuple[Union[np.ndarray, pd.DataFrame], 
                                          Union[np.ndarray, pd.Series]]]
        The augmented data, either as a single array or DataFrame, or as a tuple
        of arrays/DataFrames if `y` is provided.

    Notes
    -----
    The replacement is mathematically formulated as follows:
    Let :math:`X` be a dataset with :math:`m` elements. The function replicates 
    :math:`X` `n` times, resulting in a new dataset :math:`X'` of :math:`m * n` 
    elements if `include_original` is False. If `include_original` is True,
    :math:`X'` will have :math:`m * (n + 1)` elements.

    Examples
    --------
    
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import replace_data
    >>> X, y = np.random.randn ( 7, 2 ), np.arange(7)
    >>> X.shape, y.shape 
    ((7, 2), (7,))
    >>> X_new, y_new = replace_data (X, y, n=10 )
    >>> X_new.shape , y_new.shape
    ((70, 2), (70,))
    >>> X = np.array([[1, 2], [3, 4]])
    >>> replace_data(X, n=2, axis=0)
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> replace_data(df, n=1, include_original=True, reset_index=True)
       A  B
    0  1  3
    1  2  4
    2  1  3
    3  2  4
    """
    def concat_data(ar: Union[ArrayLike, DataFrame]) -> Union[ArrayLike, DataFrame]:
        repeated_data = [ar] * (n + 1) if include_original else [ar] * n
        
        if random_sample:
            random_indices = np.random.choice(
                ar.shape[0], size=ar.shape[0], replace=True)
            repeated_data = [ar[random_indices] for _ in repeated_data]

        concatenated = pd.concat(repeated_data, axis=axis) if isinstance(
            ar, pd.DataFrame) else np.concatenate(repeated_data, axis=axis)
        
        if shuffle:
            shuffled_indices = np.random.permutation(concatenated.shape[0])
            concatenated = concatenated[shuffled_indices] if isinstance(
                ar, pd.DataFrame) else concatenated.iloc[shuffled_indices]

        if reset_index and isinstance(concatenated, pd.DataFrame):
            concatenated.reset_index(drop=True, inplace=True)
        
        return concatenated

    X = np.array(X) if not isinstance(X, (np.ndarray, pd.DataFrame)) else X
    y = np.array(y) if y is not None and not isinstance(y, (np.ndarray, pd.Series)) else y

    if y is not None:
        return concat_data(X), concat_data(y)
    return concat_data(X)

def convert_value_in (v, unit ='m'): 
    """Convert value based on the reference unit.
    
    Parameters 
    ------------
    v: str, float, int, 
      value to convert 
    unit: str, default='m'
      Reference unit to convert value in. Default is 'meters'. Could be 
      'kg' or else. 
      
    Returns
    -------
    v: float, 
       Value converted. 
       
    Examples 
    ---------
    >>> from gofast.tools.coreutils import convert_value_in 
    >>> convert_value_in (20) 
    20.0
    >>> convert_value_in ('20mm') 
    0.02
    >>> convert_value_in ('20kg', unit='g') 
    20000.0
    >>> convert_value_in ('20') 
    20.0
    >>> convert_value_in ('20m', unit='g')
    ValueError: Unknwon unit 'm'...
    """
    c= { 'k':1e3 , 
        'h':1e2 , 
        'dc':1e1 , 
        '':1e0 , 
        'd':1e-1, 
        'c':1e-2 , 
        'm':1e-3  
        }
    c = {k +str(unit).lower(): v for k, v in c.items() }

    v = str(v).lower()  

    regex = re.findall(r'[a-zA-Z]', v) 
    
    if len(regex) !=0: 
        unit = ''.join( regex ) 
        v = v.replace (unit, '')

    if unit not in c.keys(): 
        raise ValueError (
            f"Unknwon unit {unit!r}. Expect {smart_format(c.keys(), 'or' )}."
            f" Or rename the `unit` parameter maybe to {unit[-1]!r}.")
    
    return float ( v) * (c.get(unit) or 1e0) 

def split_list(lst:List[Any],  val:int, fill_value:Optional[Any]=None ):
    """Module to extract a slice of elements from the list 
    
    Parameters 
    ------------
    lst: list, 
      List composed of item elements 
    val: int, 
      Number of item to group by default. 
      
    Returns 
    --------
    group with slide items 
    
    Examples
    --------
    >>> from gofast.tools.coreutils import split_list
    >>> lst = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> val = 3
    >>> print(split_list(lst, val))
    [[1, 2, 3], [4, 5, 6], [7, 8]]
 
    """

    lst = is_iterable(lst , exclude_string =True , transform =True ) 
    val = int ( _assert_all_types(val, int, float )) 
    try: 
        sl= [list(group) for key, group in itertools.groupby(
                lst, lambda x: (x-1)//val)]
    except: 
        # when string is given 
        sl= list(itertools.zip_longest(
            *(iter(lst),)*val,fillvalue =fill_value),)
    return sl 

def key_search (
    keys: str,  
    default_keys: Union [Text , List[str]], 
    parse_keys: bool=True, 
    regex :re=None, 
    pattern :str=None, 
    deep: bool =...,
    raise_exception:bool=..., 
    ): 
    """Find key in a list of default keys and select the best match. 
    
    Parameters 
    -----------
    keys: str or list 
       The string or a list of key. When multiple keys is passed as a string, 
       use the space for key separating. 
       
    default_keys: str or list 
       The likehood key to find. Can be a litteral text. When a litteral text 
       is passed, it is better to provide the regex in order to skip some 
       character to parse the text properly. 
       
    parse_keys: bool, default=True 
       Parse litteral string using default `pattern` and `regex`. 
       
       .. versionadded:: 0.2.7 
        
    regex: `re` object,  
        Regular expresion object. Regex is important to specify the kind
        of data to parse. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&*@!_,;\s-]\s*', flags=re.IGNORECASE)
            
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns. Pattern is 
        important especially when some character are considers as a part of 
        word but they are not a separator. For example a data columns with 
        a name `'DH_Azimuth'`, if a pattern is not explicitely provided, 
        the default pattern will parse as two separated word which is far 
        from the expected results. 
        
    deep: bool, default=False 
       Not sensistive to uppercase. 
       
    raise_exception: bool, default=False 
       raise error when key is not find. 
       
    Return 
    -------
    list: list of valid keys or None if not find ( default) 

    Examples
    ---------
    >>> from gofast.tools.coreutils import key_search 
    >>> key_search('h502-hh2601', default_keys= ['h502', 'h253','HH2601'])
    Out[44]: ['h502']
    >>> key_search('h502-hh2601', default_keys= ['h502', 'h253','HH2601'], 
                   deep=True)
    Out[46]: ['h502', 'HH2601']
    >>> key_search('253', default_keys= ("I m here to find key among h502,
                                             h253 and HH2601"))
    Out[53]: ['h253'] 
    >>> key_search ('east', default_keys= ['DH_East', 'DH_North']  , deep =True,)
    Out[37]: ['East']
    key_search ('east', default_keys= ['DH_East', 'DH_North'], 
                deep =True,parse_keys= False)
    Out[39]: ['DH_East']
    """
    deep, raise_exception, parse_keys = ellipsis2false(
        deep, raise_exception, parse_keys)
    # make a copy of original keys 
    
    kinit = copy.deepcopy(keys)
    if parse_keys: 
        if is_iterable(keys , exclude_string= True ): 
            keys = ' '.join ( [str(k) for k in keys ]) 
             # for consisteny checker 
        pattern = pattern or '[#&@!_+,;\s-]\s*'
        keys = str2columns ( keys , regex = regex , pattern = pattern ) 
            
        if is_iterable ( default_keys , exclude_string=True ): 
            default_keys = ' '. join ( [ str(k) for k in default_keys ])
            # make a copy
        default_keys =  str2columns(
            default_keys, regex =regex , pattern = pattern )
    else : 
        keys = is_iterable(
        keys, exclude_string = True, transform =True )
        default_keys = is_iterable ( 
            default_keys, exclude_string=True, transform =True )
        
    dk_init = copy.deepcopy(default_keys )
    # if deep convert all keys to lower 
    if deep: 
        keys= [str(it).lower() for it in keys  ]
        default_keys = [str(it).lower() for it in default_keys  ]

    valid_keys =[] 
    for key in keys : 
        for ii, dkey in enumerate (default_keys) : 
            vk = re.findall(rf'\w*{key}\w*', dkey)
            # rather than rf'\b\w*{key}\w*\b'
            # if deep take the real values in defaults keys.
            if len(vk) !=0: 
                if deep: valid_keys.append( dk_init[ii] )
                else:valid_keys.extend( vk)
                break     
    if ( raise_exception 
        and len(valid_keys)==0
        ): 
        kverb ='s' if len(kinit)> 1 else ''
        raise KeyError (f"key{kverb} {kinit!r} not found."
                       f" Expect {smart_format(dk_init, 'or')}")
    return None if len(valid_keys)==0 else valid_keys 

# def repeat_item_insertion(text: str, pos: Union[int, float], item: Optional[str] = None,  fill_value: Optional[Any] = None) -> str: 
#     """ Insert character in text according to its position. 
#     
#     Parameters
#     ----------
#     text: str
#        Text 
#     pos: Union[int, float]
#       Position where the item must be inserted. 
#     item: Optional[str], default None
#       Item to insert at each position. 
#    fill_value: Optional[Any], default None
#       Does nothing special; fill the last position. 
#     Returns
#     --------
#     text: str
#       New construct object. 
#       
#     Examples
#     ----------
#     >>> repeat_item_insertion('0125356.45', pos=2, item=':')
#     '01:25:35:6.45'
#     >>> repeat_item_insertion('Function inserts car in text.', pos=10, item='TK')
#     'Function iTKnserts carTK in text.'
#     """
#     if item is None:
#         item = ''
#    # For consistency
#    lst = list(str(text))
#    # Check whether there is a decimal then remove it 
#    dec_part = []
#    ent_part = lst
#    for i, it in enumerate(lst):
#        if it == '.': 
#            ent_part, dec_part = lst[:i], lst[i:]
#            break
#    # Now split list
#    if fill_value is None:
#        fill_value = ''
#    
#    value = split_list(ent_part, val=pos, fill_value=fill_value)
#    # Join with mark
#    join_lst = [''.join(s) for s in value]
#    # Use empty string instead of None in the join operation
#    result = str(item).join(join_lst) + ''.join(dec_part)
#    return result


def numstr2dms(
    sdigit: str,  
    sanitize: bool = True, 
    func: callable = lambda x, *args, **kws: x, 
    args: tuple = (),  
    regex: re.Pattern = re.compile(r'[_#&@!+,;:"\'\s-]\s*', flags=re.IGNORECASE),   
    pattern: str = '[_#&@!+,;:"\'\s-]\s*', 
    return_values: bool = False, 
    **kws
) -> Union[str, Tuple[float, float, float]]: 
    """ Convert numerical digit string to DD:MM:SS
    
    Note that any string digit for Minutes and seconds must be composed
    of two values i.e., the function accepts at least six digits, otherwise an 
    error occurs. For instance, the value between [0-9] must be prefixed by 0 
    beforehand. Here is an example for designating 1 degree-1 min-1 seconds::
        
        sdigit= 1'1'1" --> 01'01'01 or 010101
        
    where ``010101`` is the right arguments for ``111``. 
    
    Parameters
    -----------
    sdigit: str, 
      Digit string composing of unique values. 
    func: Callable, 
      Function uses to parse digit. Function must return string values. 
      Any other values should be converted to str.
      
    args: tuple
      Function `func` positional arguments 
      
    regex: `re` object,  
        Regular expression object. Regex is important to specify the kind
        of data to parse. The default is:: 
            
            >>> import re 
            >>> re.compile(r'[_#&@!+,;:"\'\s-]\s*', flags=re.IGNORECASE) 
            
    pattern: str, default = '[_#&@!+,;:"\'\s-]\s*'
      Specific pattern for sanitizing sdigit. For instance, remove undesirable 
      non-character. 
      
    sanitize: bool, default=True 
       Remove undesirable characters using the default argument of `pattern`
       parameter. 
       
    return_values: bool, default=False, 
       Return the DD:MM:SS into a tuple of (DD, MM, SS).
    
    Returns 
    -------
    sdigit/tuple: str, tuple 
      DD:MM:SS or tuple of (DD, MM, SS)
      
    Examples
    --------
    >>> numstr2dms("1134132.08")
    '113:41:32.08'
    >>> numstr2dms("13'41'32.08")
    '13:41:32.08'
    >>> numstr2dms("11:34:13:2.08", return_values=True)
    (113.0, 41.0, 32.08)
    """
    # Remove any character from the string digit
    sdigit = str(sdigit)
    
    if sanitize: 
        sdigit = re.sub(pattern, "", sdigit, flags=re.IGNORECASE)
        
    try:
        float(sdigit)
    except ValueError:
        raise ValueError(f"Wrong value. Expects a string-digit or digit. Got {sdigit!r}")

    if callable(func): 
        sdigit = func(sdigit, *args, **kws)
        
    # In the case there is'
    decimal = '0'
    # Remove decimal
    sdigit_list = sdigit.split(".")
    
    if len(sdigit_list) == 2: 
        sdigit, decimal = sdigit_list
        
    if len(sdigit) < 6: 
        raise ValueError(f"DMS expects at least six digits (DD:MM:SS). Got {sdigit!r}")
        
    sec, sdigit = sdigit[-2:], sdigit[:-2]
    mm, sdigit = sdigit[-2:], sdigit[:-2]
    deg = sdigit  # The remaining part
    # Concatenate second decimal 
    sec += f".{decimal}" 
    
    return tuple(map(float, [deg, mm, sec])) if return_values \
        else ':'.join([deg, mm, sec])


def store_or_write_hdf5 (
    d,  
    key:str= None, 
    mode:str='a',  
    kind: str=None, 
    path_or_buf:str= None, 
    encoding:str="utf8", 
    csv_sep: str=",",
    index: bool=..., 
    columns:Union [str, List[Any]]=None, 
    sanitize_columns:bool=...,  
    func: _F= None, 
    args: tuple=(), 
    applyto: Union [str, List[Any]]=None, 
    **func_kwds, 
    )->Union [None, DataFrame]: 
    """ Store data to hdf5 or write data to csv file. 
    
    Note that by default, the data is not store nor write and 
    return data if frame or transform the Path-Like object to data frame. 

    Parameters 
    -----------
    d: Dataframe, shape (m_samples, n_features)
        data to store or write or sanitize.
    key:str
       Identifier for the group in the store.
       
    mode: {'a', 'w', 'r+'}, default 'a'
       Mode to open file:
    
       - 'w': write, a new file is created (an existing file with the 
                                          same name would be deleted).
       - 'a': append, an existing file is opened for reading and writing, 
         and if the file does not exist it is created.
       - 'r+': similar to 'a', but the file must already exist.
       
    kind: str, {'store', 'write', None} , default=None 
       Type of task to perform: 
           
       - 'store': Store data to hdf5
       - 'write': export data to csv file.
       - None: construct a dataframe if array is passed or sanitize it. 

    path_or_buf: str or pandas.HDFStore, or str, path object, file-like \
        object, or None, default=None 
       File path or HDFStore object. String, path object
       (implementing os.PathLike[str]), or file-like object implementing 
       a write() function. If ``write=True`` and  None, the result is returned 
       as a string. If a non-binary file object is passed, it should be 
       opened with newline=" ", disabling universal newlines. If a binary 
       file object is passed, mode might need to contain a 'b'.
      
    encoding: str, default='utf8'
       A string representing the encoding to use in the output file, 
       Encoding is not supported if path_or_buf is a non-binary file object. 

    csv_sep: str, default=',', 
       String of length 1. Field delimiter for the output file.
       
    index: bool, index =False, 
       Write data to csv with index or not. 
       
    columns: list of str, optional 
        Usefull to create a dataframe when array is passed. Be aware to fit 
        the number of array columns (shape[1])
        
    sanitize_columns: bool, default=False, 
       remove undesirable character in the data columns using the default
       argument of `regex` parameters and fill pattern to underscore '_'. 
       The default regex implementation is:: 
           
           >>> import re 
           >>> re.compile (r'[_#&.)(*@!,;\s-]\s*', flags=re.IGNORECASE)
           
    func: callable, Optional 
       A custom sanitizing function and apply to each columns of the dataframe.
       If provide, the expected columns must be listed to `applyto` parameter.
       
    args: tuple, optional 
       Positional arguments of the sanitizing columns 
       
    applyto: str or list of str, Optional 
       The list of columns to apply the function ``func``. To apply the 
       function to all columns, use the ``*`` instead. 
       
    func_kwds: dict, 
       Keywords arguments of the sanitizing function ``func``. 
       
    Return 
    -------
    None or d: None of dataframe. 
      returns None if `kind` is set to ``write`` or ``store`` otherwise 
      return the dataframe. 
  
    Examples
    --------
    >>> from gofast.tools.coreutils import store_or_write_hdf5
    >>> from gofast.datasets import load_bagoue 
    >>> data = load_bagoue().frame 
    >>> data.geol[:5]
    0    VOLCANO-SEDIM. SCHISTS
    1                  GRANITES
    2                  GRANITES
    3                  GRANITES
    4          GEOSYN. GRANITES
    Name: geol, dtype: object
    >>> data = store_or_write_hdf5 ( data, sanitize_columns = True)
    >>> data[['type', 'geol', 'shape']] # put all to lowercase
      type                    geol shape
    0   cp  volcano-sedim. schists     w
    1   ec                granites     v
    2   ec                granites     v
    >>> # compute using func 
    >>> def test_func ( a, times  , to_percent=False ): 
            return ( a * times / 100)   if to_percent else ( a *times )
    >>> data.sfi[:5]
    0    0.388909
    1    1.340127
    2    0.446594
    3    0.763676
    4    0.068501
    Name: sfi, dtype: float64
    >>> d = store_or_write_hdf5 ( data,  func = test_func, args =(7,), applyto='sfi')
    >>> d.sfi[:5] 
    0    2.722360
    1    9.380889
    2    3.126156
    3    5.345733
    4    0.479507
    Name: sfi, dtype: float64
    >>> store_or_write_hdf5 ( data,  func = test_func, args =(7,),
                          applyto='sfi', to_percent=True).sfi[:5]
    0    0.027224
    1    0.093809
    2    0.031262
    3    0.053457
    4    0.004795
    Name: sfi, dtype: float64
    >>> # write data to hdf5 and outputs to current directory 
    >>> store_or_write_hdf5 ( d, key='test0', path_or_buf= 'test_data.h5', 
                          kind ='store')
    >>> # export data to csv 
    >>> store_or_write_hdf5 ( d, key='test0', path_or_buf= 'test_data', 
                          kind ='export')
    """
    kind= key_search (str(kind), default_keys=(
        "none", "store", "write", "export", "tocsv"), 
        raise_exception=True , deep=True)[0]
    
    kind = "export" if kind in ('write', 'tocsv') else kind 
    
    if sanitize_columns is ...: 
        sanitize_columns=False 
    d = to_numeric_dtypes(d, columns=columns,sanitize_columns=sanitize_columns, 
                          fill_pattern='_')
   
    # get categorical variables 
    if ( sanitize_columns 
        or func is not None
        ): 
        d, _, cf = to_numeric_dtypes(d, return_feature_types= True )
        #( strip then pass to lower case all non-numerical data) 
        # for minimum sanitization  
        for cat in cf : 
            d[cat]= d[cat].str.lower()
            d[cat]= d[cat].str.strip()
            
    if func is not None: 
        if not callable(func): 
            raise TypeError(
                f"Expect a callable for `func`. Got {type(func).__name__!r}")

        if applyto is None:
            raise ValueError("Need to specify the data column to apply"
                             f"{func.__name__!r} to.")
        
        applyto = is_iterable( applyto, exclude_string=True,
                               transform =True ) if applyto !="*" else d.columns 
        # check whether the applyto columns are in data columns 
        exist_features(d, applyto)
        
        # map each colum 
        for col in applyto: 
            d [col]=d[col].apply( func, args=args, **func_kwds )

    # store in h5 file. 
    if kind=='store':
        if path_or_buf is None: 
            print("Destination file is missing. Use 'data.h5' instead outputs"
                  f" in the current directory {os.getcwd()}")
            path_or_buf= 'data.h5'
 
        d.to_hdf ( path_or_buf , key =key, mode =mode )
    # export to csv file
    if kind=="export": 
        d.to_csv(path_or_buf, encoding = encoding  , sep=csv_sep , 
                 index =False if index is ... else index   )
        
    return d if kind not in ("store", "export") else None 

def ellipsis2false( *parameters , default_value: Any=False ): 
    """ Turn all parameter arguments to False if ellipsis.
    
    Note that the output arguments must be in the same order like the 
    positional arguments. 
 
    :param parameters: tuple 
       List of parameters 
    :param default_value: Any, 
       Value by default that might be take the ellipsis. 
    :return: tuple, same list of parameters passed ellipsis to 
       ``default_value``. By default, it returns ``False``. For a single 
       parameters, uses the trailing comma for collecting the parameters 
       
    :example: 
        >>> from gofast.tools.coreutils import ellipsis2false 
        >>> var, = ellipsis2false (...)
        >>> var 
        False
        >>> data, sep , verbose = ellipsis2false ([2,3, 4], ',', ...)
        >>> verbose 
        False 
    """
    return tuple ( ( default_value  if param is  ... else param  
                    for param in parameters) )  

def type_of_target(y):
    """
    Determine the type of data indicated by the target variable.

    Parameters
    ----------
    y : array-like
        Target values. 

    Returns
    -------
    target_type : string
        Type of target data, such as 'binary', 'multiclass', 'continuous', etc.

    Examples
    --------
    >>> type_of_target([0, 1, 1, 0])
    'binary'
    >>> type_of_target([0.5, 1.5, 2.5])
    'continuous'
    >>> type_of_target([[1, 0], [0, 1]])
    'multilabel-indicator'
    """
    # Check if y is an array-like
    if not isinstance(y, (np.ndarray, list, pd.Series, Sequence, pd.DataFrame)):
        raise ValueError("Expected array-like (array or list), got %s" % type(y))

    # Check for valid number type
    if not all(isinstance(i, (int, float, np.integer, np.floating)) 
               for i in np.array(y).flatten()):
        raise ValueError("Input must be a numeric array-like")

    # Continuous data
    if any(isinstance(i, float) for i in np.array(y).flatten()):
        return 'continuous'

    # Binary or multiclass
    unique_values = np.unique(y)
    if len(unique_values) == 2:
        return 'binary'
    elif len(unique_values) > 2 and np.ndim(y) == 1:
        return 'multiclass'

    # Multilabel indicator
    if isinstance(y[0], (np.ndarray, list, Sequence)) and len(y[0]) > 1:
        return 'multilabel-indicator'

    return 'unknown'


def add_noises_to(
    data,  
    noise=0.1, 
    seed=None, 
    gaussian_noise=False,
    cat_missing_value=pd.NA
    ):
    """
    Adds NaN or specified missing values to a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to which NaN values or specified missing 
        values will be added.

    noise : float, default=0.1
        The percentage of values to be replaced with NaN or the 
        specified missing value in each column. This must be a 
        number between 0 and 1. Default is 0.1 (10%).

        .. math:: \text{noise} = \frac{\text{number of replaced values}}{\text{total values in column}}

    seed : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        Seed for random number generator to ensure reproducibility. 
        If `seed` is an int, array-like, or BitGenerator, it will be 
        used to seed the random number generator. If `seed` is a 
        np.random.RandomState or np.random.Generator, it will be used 
        as given.

    gaussian_noise : bool, default=False
        If `True`, adds Gaussian noise to the data. Otherwise, replaces 
        values with NaN or the specified missing value.

    cat_missing_value : scalar, default=pd.NA
        The value to use for missing data in categorical columns. By 
        default, `pd.NA` is used.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with NaN or specified missing values added.

    Notes
    -----
    The function modifies the DataFrame by either adding Gaussian noise 
    to numerical columns or replacing a percentage of values in each 
    column with NaN or a specified missing value.

    The Gaussian noise is added according to the formula:

    .. math:: \text{new_value} = \text{original_value} + \mathcal{N}(0, \text{noise})

    where :math:`\mathcal{N}(0, \text{noise})` represents a normal 
    distribution with mean 0 and standard deviation equal to `noise`.

    Examples
    --------
    >>> from gofast.tools.coreutils import add_noises_to
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    >>> new_df = add_noises_to(df, noise=0.2)
    >>> new_df
         A     B
    0  1.0  <NA>
    1  NaN     y
    2  3.0  <NA>

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> new_df = add_noises_to(df, noise=0.1, gaussian_noise=True)
    >>> new_df
              A         B
    0  1.063292  3.986400
    1  2.103962  4.984292
    2  2.856601  6.017380

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable, potentially 
        heterogeneous tabular data.
    numpy.random.normal : Draw random samples from a normal 
        (Gaussian) distribution.

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 
           (2020). Array programming with NumPy. Nature, 585(7825), 
           357-362.
    """
    
    is_frame = isinstance (data, pd.DataFrame ) 
    if not is_frame: 
        data = pd.DataFrame(data ) 
        
    np.random.seed(seed)
    if noise is None: 
        return data 
    noise, gaussian_noise  = _parse_gaussian_noise (noise )

    if gaussian_noise:
        # Add Gaussian noise to numerical columns only
        def add_gaussian_noise(column):
            if pd.api.types.is_numeric_dtype(column):
                return column + np.random.normal(0, noise, size=column.shape)
            return column
        
        noise_data = data.apply(add_gaussian_noise)
        
        if not is_frame: 
            noise_data = np.asarray(noise_data)
        return noise_data
    else:
        # Replace values with NaN or specified missing value
        df_with_nan = data.copy()
        nan_count_per_column = int(noise * len(df_with_nan))

        for column in df_with_nan.columns:
            nan_indices = random.sample(range(len(df_with_nan)), nan_count_per_column)
            if pd.api.types.is_numeric_dtype(df_with_nan[column]):
                df_with_nan.loc[nan_indices, column] = np.nan
            else:
                df_with_nan.loc[nan_indices, column] = cat_missing_value
                
        if not is_frame: 
            df_with_nan = df_with_nan.values 
            
        return df_with_nan

def _parse_gaussian_noise(noise):
    """
    Parses the noise parameter to determine if Gaussian noise should be used
    and extracts the noise level if specified.

    Parameters
    ----------
    noise : str, float, or None
        The noise parameter to be parsed. Can be a string specifying Gaussian
        noise with an optional noise level, a float, or None.

    Returns
    -------
    tuple
        A tuple containing:
        - float: The noise level.
        - bool: Whether Gaussian noise should be used.

    Examples
    --------
    >>> from gofast.tools.coreutils import _parse_gaussian_noise
    >>> _parse_gaussian_noise('0.1gaussian')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian_0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian10%')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian 10 %')
    (0.1, True)
    >>> _parse_gaussian_noise(0.05)
    (0.05, False)
    >>> _parse_gaussian_noise(None)
    (0.1, False)
    >>> _parse_gaussian_noise('invalid')
    Traceback (most recent call last):
        ...
    ValueError: Invalid noise value: invalid
    """
    gaussian_noise = False
    default_noise = 0.1

    if isinstance(noise, str):
        orig_noise = noise 
        noise = noise.lower()
        gaussian_keywords = ["gaussian", "gauss"]

        if any(keyword in noise for keyword in gaussian_keywords):
            gaussian_noise = True
            noise = re.sub(r'[^\d.%]', '', noise)  # Remove non-numeric and non-'%' characters
            noise = re.sub(r'%', '', noise)  # Remove '%' if present

            try:
                noise_level = float(noise) / 100 if '%' in orig_noise else float(noise)
                noise = noise_level if noise_level else default_noise
            except ValueError:
                noise = default_noise

        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError(f"Invalid noise value: {noise}")
    elif noise is None:
        noise = default_noise
    
    noise = validate_noise (noise ) 
    
    return noise, gaussian_noise


def nan_to_na(
    data, 
    cat_missing_value=pd.NA, 
    nan_spec=np.nan
    ):
    """
    Converts specified NaN values in categorical columns of a pandas 
    DataFrame or Series to `pd.NA` or another specified missing value.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The input DataFrame or Series in which specified NaN values in 
        categorical columns will be converted.
        
    cat_missing_value : scalar, default=pd.NA
        The value to use for missing data in categorical columns. By 
        default, `pd.NA` is used. This ensures that categorical columns 
        do not contain `np.nan` values, which can cause type 
        inconsistencies.

    nan_spec : scalar, default=np.nan
        The value that is treated as NaN in the input data. By default, 
        `np.nan` is used. This allows flexibility in specifying what is 
        considered as NaN.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The DataFrame or Series with specified NaN values in categorical 
        columns converted to the specified missing value.

    Notes
    -----
    This function ensures consistency in the representation of missing 
    values in categorical columns, avoiding issues that arise from the 
    presence of specified NaN values in such columns.

    The conversion follows the logic:
    
    .. math:: 
        \text{If column is categorical and contains `nan_spec`} 
        \rightarrow \text{Replace `nan_spec` with `cat_missing_value`}

    Examples
    --------
    >>> from gofast.tools.coreutils import nan_to_na
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1.0, 2.0, np.nan], 'B': ['x', np.nan, 'z']})
    >>> df['B'] = df['B'].astype('category')
    >>> df = nan_to_na(df)
    >>> df
         A     B
    0  1.0     x
    1  2.0  <NA>
    2  NaN     z

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable, potentially 
        heterogeneous tabular data.
    pandas.Series : One-dimensional ndarray with axis labels.
    numpy.nan : IEEE 754 floating point representation of Not a Number 
        (NaN).

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 
           (2020). Array programming with NumPy. Nature, 585(7825), 
           357-362.

    """
    if not isinstance (data, (pd.Series, pd.DataFrame)): 
        raise ValueError("Input must be a pandas DataFrame or Series."
                         f" Got {type(data).__name__!r} instead.")
        
    def has_nan_values(series, nan_spec):
        """Check if nan_spec exists in the series."""
        return series.isin([nan_spec]).any()
    
    if isinstance(data, pd.Series):
        if has_nan_values(data, nan_spec):
            if pd.api.types.is_categorical_dtype(data):
                return data.replace({nan_spec: cat_missing_value})
        return data
    
    elif isinstance(data, pd.DataFrame):
        df_copy = data.copy()
        for column in df_copy.columns:
            if has_nan_values(df_copy[column], nan_spec):
                if pd.api.types.is_categorical_dtype(df_copy[column]):
                    df_copy[column] = df_copy[column].replace({nan_spec: cat_missing_value})
        return df_copy

def validate_noise(noise):
    """
    Validates the `noise` parameter and returns either the noise value
    as a float or the string 'gaussian'.

    Parameters
    ----------
    noise : str or float or None
        The noise parameter to be validated. It can be the string
        'gaussian', a float value, or None.

    Returns
    -------
    float or str
        The validated noise value as a float or the string 'gaussian'.

    Raises
    ------
    ValueError
        If the `noise` parameter is a string other than 'gaussian' or
        cannot be converted to a float.

    Examples
    --------
    >>> validate_noise('gaussian')
    'gaussian'
    >>> validate_noise(0.1)
    0.1
    >>> validate_noise(None)
    None
    >>> validate_noise('0.2')
    0.2

    """
    if isinstance(noise, str):
        if noise.lower() == 'gaussian':
            return 'gaussian'
        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError("The `noise` parameter accepts the string"
                                 " 'gaussian' or a float value.")
    elif noise is not None:
        noise = validate_ratio(noise, bounds=(0, 1), param_name='noise' )
        # try:
        # except ValueError:
        #     raise ValueError("The `noise` parameter must be convertible to a float.")
    return noise

def fancier_repr_formatter(obj, max_attrs=7):
    """
    Generates a formatted string representation for any class object.

    Parameters:
    ----------
    obj : object
        The object for which the string representation is generated.

    max_attrs : int, optional
        Maximum number of attributes to display in the representation.

    Returns:
    -------
    str
        A string representation of the object.

    Examples:
    --------
    >>> from gofast.tools.coreutils import fancier_repr_formatter
    >>> class MyClass:
    >>>     def __init__(self, a, b, c):
    >>>         self.a = a
    >>>         self.b = b
    >>>         self.c = c
    >>> obj = MyClass(1, [1, 2, 3], 'hello')
    >>> print(fancier_repr_formatter(obj))
    MyClass(a=1, c='hello', ...)
    """
    attrs = [(name, getattr(obj, name)) for name in dir(obj)
             if not name.startswith('_') and
             (isinstance(getattr(obj, name), str) or
              not hasattr(getattr(obj, name), '__iter__'))]

    displayed_attrs = attrs[:min(len(attrs), max_attrs)]
    attr_str = ', '.join([f'{name}={value!r}' for name, value in displayed_attrs])

    # Add ellipsis if there are more attributes than max_attrs
    if len(attrs) > max_attrs:
        attr_str += ', ...'

    return f'{obj.__class__.__name__}({attr_str})'

def generic_getattr(obj, name, default_value=None):
    """
    A generic attribute accessor for any class instance.

    This function attempts to retrieve an attribute from the given object.
    If the attribute is not found, it provides a meaningful error message.

    Parameters:
    ----------
    obj : object
        The object from which to retrieve the attribute.

    name : str
        The name of the attribute to retrieve.

    default_value : any, optional
        A default value to return if the attribute is not found. If None,
        an AttributeError will be raised.

    Returns:
    -------
    any
        The value of the retrieved attribute or the default value.

    Raises:
    ------
    AttributeError
        If the attribute is not found and no default value is provided.

    Examples:
    --------
    >>> from gofast.tools.coreutils import generic_getattr
    >>> class MyClass:
    >>>     def __init__(self, a, b):
    >>>         self.a = a
    >>>         self.b = b
    >>> obj = MyClass(1, 2)
    >>> print(generic_getattr(obj, 'a'))  # Prints: 1
    >>> print(generic_getattr(obj, 'c', 'default'))  # Prints: 'default'
    """
    if hasattr(obj, name):
        return getattr(obj, name)
    
    if default_value is not None:
        return default_value

    # Attempt to find a similar attribute name for a more informative error
    similar_attr = _find_similar_attribute(obj, name)
    suggestion = f". Did you mean '{similar_attr}'?" if similar_attr else ""

    raise AttributeError(f"'{obj.__class__.__name__}' object has no "
                         f"attribute '{name}'{suggestion}")

def _find_similar_attribute(obj, name):
    """
    Attempts to find a similar attribute name in the object's dictionary.

    Parameters
    ----------
    obj : object
        The object whose attributes are being checked.
    name : str
        The name of the attribute to find a similar match for.

    Returns
    -------
    str or None
        A similar attribute name if found, otherwise None.
    """
    rv = smart_strobj_recognition(name, obj.__dict__, deep =True)
    return rv 
 

def validate_url(url: str) -> bool:
    """
    Check if the provided string is a valid URL.

    Parameters
    ----------
    url : str
        The string to be checked as a URL.

    Raises
    ------
    ValueError
        If the provided string is not a valid URL.

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.

    Examples
    --------
    >>> validate_url("https://www.example.com")
    True
    >>> validate_url("not_a_url")
    ValueError: The provided string is not a valid URL.
    """
    from urllib.parse import urlparse
    
    if is_module_installed("validators"): 
        return validate_url_by_validators (url)
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("The provided string is not a valid URL.")
    return True

def validate_url_by_validators(url: str):
    """
    Check if the provided string is a valid URL using `validators` packages.

    Parameters
    ----------
    url : str
        The string to be checked as a URL.

    Raises
    ------
    ValueError
        If the provided string is not a valid URL.

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.

    Examples
    --------
    >>> validate_url("https://www.example.com")
    True
    >>> validate_url("not_a_url")
    ValueError: The provided string is not a valid URL.
    """
    import validators
    if not validators.url(url):
        raise ValueError("The provided string is not a valid URL.")
    return True

def is_module_installed(module_name: str, distribution_name: str = None) -> bool:
    """
    Check if a Python module is installed by attempting to import it.
    Optionally, a distribution name can be provided if it differs from the module name.

    Parameters
    ----------
    module_name : str
        The import name of the module to check.
    distribution_name : str, optional
        The distribution name of the package as known by package managers (e.g., pip).
        If provided and the module import fails, an additional check based on the
        distribution name is performed. This parameter is useful for packages where
        the distribution name differs from the importable module name.

    Returns
    -------
    bool
        True if the module can be imported or the distribution package is installed,
        False otherwise.

    Examples
    --------
    >>> is_module_installed("sklearn")
    True
    >>> is_module_installed("scikit-learn", "scikit-learn")
    True
    >>> is_module_installed("some_nonexistent_module")
    False
    """
    if _try_import_module(module_name):
        return True
    if distribution_name and _check_distribution_installed(distribution_name):
        return True
    return False

def _try_import_module(module_name: str) -> bool:
    """
    Attempt to import a module by its name.

    Parameters
    ----------
    module_name : str
        The import name of the module.

    Returns
    -------
    bool
        True if the module can be imported, False otherwise.
    """
    # import importlib.util
    # module_spec = importlib.util.find_spec(module_name)
    # return module_spec is not None
    import importlib
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False 
    
def _check_distribution_installed(distribution_name: str) -> bool:
    """
    Check if a distribution package is installed by its name.

    Parameters
    ----------
    distribution_name : str
        The distribution name of the package.

    Returns
    -------
    bool
        True if the distribution package is installed, False otherwise.
    """
    try:
        # Prefer importlib.metadata for Python 3.8 and newer
        from importlib.metadata import distribution
        distribution(distribution_name)
        return True
    except ImportError:
        # Fallback to pkg_resources for older Python versions
        try:
            from pkg_resources import get_distribution, DistributionNotFound
            get_distribution(distribution_name)
            return True
        except DistributionNotFound:
            return False
    except Exception:
        return False
    
def get_installation_name(
        module_name: str, distribution_name: Optional[str] = None, 
        return_bool: bool = False) -> Union[str, bool]:
    """
    Determines the appropriate name for installing a package, considering potential
    discrepancies between the distribution name and the module import name. Optionally,
    returns a boolean indicating if the distribution name matches the import name.

    Parameters
    ----------
    module_name : str
        The import name of the module.
    distribution_name : str, optional
        The distribution name of the package. If None, the function attempts to infer
        the distribution name from the module name.
    return_bool : bool, optional
        If True, returns a boolean indicating whether the distribution name matches
        the module import name. Otherwise, returns the name recommended for installation.

    Returns
    -------
    Union[str, bool]
        Depending on `return_bool`, returns either a boolean indicating if the distribution
        name matches the module name, or the name (distribution or module) recommended for
        installation.
    """
    inferred_name = _infer_distribution_name(module_name)

    # If a distribution name is provided, check if it matches the inferred name
    if distribution_name:
        if return_bool:
            return distribution_name.lower() == inferred_name.lower()
        return distribution_name

    # If no distribution name is provided, return the inferred name or module name
    if return_bool:
        return inferred_name.lower() == module_name.lower()

    return inferred_name or module_name

def _infer_distribution_name(module_name: str) -> str:
    """
    Attempts to infer the distribution name of a package from its module name
    by querying the metadata of installed packages.

    Parameters
    ----------
    module_name : str
        The import name of the module.

    Returns
    -------
    str
        The inferred distribution name. If no specific inference is made, returns
        the module name.
    """
    try:
        # Use importlib.metadata for Python 3.8+; use importlib_metadata for older versions
        from importlib.metadata import distributions
    except ImportError:
        from importlib_metadata import distributions
    #  Loop through all installed distributions
    for distribution in distributions():
        # Check if the module name matches the distribution name directly
        if module_name == distribution.metadata.get('Name').replace('-', '_'):
            return distribution.metadata['Name']

        # Safely attempt to read and split 'top_level.txt'
        top_level_txt = distribution.read_text('top_level.txt')
        if top_level_txt:
            top_level_packages = top_level_txt.split()
            if any(module_name == pkg.split('.')[0] for pkg in top_level_packages):
                return distribution.metadata['Name']

    return module_name

def normalize_string(
    input_str: str, 
    target_strs: Optional[List[str]] = None, 
    num_chars_check: Optional[int] = None, 
    deep: bool = False, 
    return_target_str: bool = False,
    return_target_only: bool=False, 
    raise_exception: bool = False,
    ignore_case: bool = True,
    match_method: str = 'exact',
    error_msg: str=None, 
) -> Union[str, Tuple[str, Optional[str]]]:
    """
    Normalizes a string by applying various transformations and optionally checks 
    against a list of target strings based on different matching methods.

    Function normalizes a string by stripping leading/trailing whitespace, 
    converting to lowercase,and optionally checks against a list of target  
    strings. If specified, returns the target string that matches the 
    conditions. Raise an exception if the string is not found.
    
    Parameters
    ----------
    input_str : str
        The string to be normalized.
    target_strs : List[str], optional
        A list of target strings for comparison.
    num_chars_check : int, optional
        The number of characters at the start of the string to check 
        against each target string.
    deep : bool, optional
        If True, performs a deep substring check within each target string.
    return_target_str : bool, optional
        If True and a target string matches, returns the matched target string 
        along with the normalized string.
    return_target_only: bool, optional 
       If True and a target string  matches, returns only the matched string
       target. 
    raise_exception : bool, optional
        If True and the input string is not found in the target strings, 
        raises an exception.
    ignore_case : bool, optional
        If True, ignores case in string comparisons. Default is True.
    match_method : str, optional
        The string matching method: 'exact', 'contains', or 'startswith'.
        Default is 'exact'.
    error_msg: str, optional, 
       Message to raise if `raise_exception` is ``True``. 
       
    Returns
    -------
    Union[str, Tuple[str, Optional[str]]]
        The normalized string. If return_target_str is True and a target 
        string matches, returns a tuple of the normalized string and the 
        matched target string.

    Raises
    ------
    ValueError
        If raise_exception is True and the input string is not found in 
        the target strings.

    Examples
    --------
    >>> from gofast.tools.coreutils import normalize_string
    >>> normalize_string("Hello World", target_strs=["hello", "world"], ignore_case=True)
    'hello world'
    >>> normalize_string("Goodbye World", target_strs=["hello", "goodbye"], 
                         num_chars_check=7, return_target_str=True)
    ('goodbye world', 'goodbye')
    >>> normalize_string("Hello Universe", target_strs=["hello", "world"],
                         raise_exception=True)
    ValueError: Input string not found in target strings.
    """
    normalized_str = str(input_str).lower() if ignore_case else input_str

    if not target_strs:
        return normalized_str
    target_strs = is_iterable(target_strs, exclude_string=True, transform =True)
    normalized_targets = [str(t).lower() for t in target_strs] if ignore_case else target_strs
    matched_target = None

    for target in normalized_targets:
        if num_chars_check is not None:
            condition = (normalized_str[:num_chars_check] == target[:num_chars_check])
        elif deep:
            condition = (normalized_str in target)
        elif match_method == 'contains':
            condition = (target in normalized_str)
        elif match_method == 'startswith':
            condition = normalized_str.startswith(target)
        else:  # Exact match
            condition = (normalized_str == target)

        if condition:
            matched_target = target
            break

    if matched_target is not None:
        if return_target_only: 
            return matched_target 
        return (normalized_str, matched_target) if return_target_str else normalized_str

    if raise_exception:
        error_msg = error_msg or ( 
            f"Invalid input. Expect {smart_format(target_strs, 'or')}."
            f" Got {input_str!r}."
            )
        raise ValueError(error_msg)
    
    if return_target_only: 
        return matched_target 
    
    return ('', None) if return_target_str else ''

def format_and_print_dict(data_dict, front_space=4):
    """
    Formats and prints the contents of a dictionary in a structured way.

    Each key-value pair in the dictionary is printed with the key followed by 
    its associated values. 
    The values are expected to be dictionaries themselves, allowing for a nested 
    representation.
    The inner dictionary's keys are sorted in descending order before printing.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key contains a dictionary of items to be printed. 
        The key represents a category
        or label, and the value is another dictionary where each key-value pair 
        represents an option or description.
        
    front_space : int, optional
        The number of spaces used for indentation in front of each line (default is 4).


    Returns
    -------
    None
        This function does not return any value. It prints the formatted contents 
        of the provided dictionary.

    Examples
    --------
    >>> from gofast.tools.coreutils import format_and_print_dict
    >>> sample_dict = {
            'gender': {1: 'Male', 0: 'Female'},
            'age': {1: '35-60', 0: '16-35', 2: '>60'}
        }
    >>> format_and_print_dict(sample_dict)
    gender:
        1: Male
        0: Female
    age:
        2: >60
        1: 35-60
        0: 16-35
    """
    if not isinstance(data_dict, dict):
        raise TypeError("The input data must be a dictionary.")

    indent = ' ' * front_space
    for label, options in data_dict.items():
        print(f"{label}:")
        options= is_iterable(options, exclude_string=True, transform=True )
  
        if isinstance(options, (tuple, list)):
            for option in options:
                print(f"{indent}{option}")
        elif isinstance(options, dict):
            for key in sorted(options.keys(), reverse=True):
                print(f"{indent}{key}: {options[key]}")
        print()  # Adds an empty line for better readability between categories


def fill_nan_in(
        data: DataFrame,  method: str = 'constant', 
        value: Optional[Union[int, float, str]] = 0) -> DataFrame:
    """
    Fills NaN values in a Pandas DataFrame using various methods.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be checked and modified.
    method : str, optional
        The method to use for filling NaN values. Options include 'constant',
        'ffill', 'bfill', 'mean', 'median', 'mode'. Default is 'constant'.
    value : int, float, string, optional
        The value used when method is 'constant'. Ignored for other methods.
        Default is 0.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with NaN values filled.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.tools.coreutils import fill_nan_in
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 2, 3]})
    >>> df = fill_nan_in(df, method='median')
    >>> print(df)
       A    B
    0  1.0  2.5
    1  2.0  2.0
    2  1.5  3.0
    """
    # Check for NaN values in the DataFrame and apply the specified fill method
    if not data.isna().any().any(): 
        return data 

    fill_methods = {
        'constant': lambda: data.fillna(value, inplace=True),
        'ffill': lambda: data.fillna(method='ffill', inplace=True),
        'bfill': lambda: data.fillna(method='bfill', inplace=True),
        'mean': lambda: data.fillna(data.mean(), inplace=True),
        'median': lambda: data.fillna(data.median(), inplace=True),
        'mode': lambda: data.apply(lambda col: col.fillna(col.mode()[0], inplace=True))
    }
    
    fill_action = fill_methods.get(method)
    if fill_action:
        fill_action()
    else:
        raise ValueError(f"Method '{method}' not recognized for filling NaN values.")
        
    return data 

def get_valid_kwargs(obj_or_func, raise_warning=False, **kwargs):
    """
    Filters keyword arguments (`kwargs`) to retain only those that are valid
    for the initializer of a given object or function.

    Parameters
    ----------
    obj_or_func : object or function
        The object or function to inspect for valid keyword arguments. If it's
        callable, its `__init__` method's valid keyword arguments are considered.
    raise_warning : bool, optional
        If True, raises a warning for any keyword arguments provided that are not
        valid for `obj_or_func`. The default is False.
    **kwargs : dict
        Arbitrary keyword arguments to filter based on `obj_or_func`'s
        valid parameters.

    Returns
    -------
    dict
        A dictionary containing only the keyword arguments that are valid for the
        `obj_or_func`'s initializer.

    Raises
    ------
    Warning
        If `raise_warning` is True and there are keyword arguments that are not
        valid for `obj_or_func`, a warning is raised.

    Notes
    -----
    This function checks whether the provided keyword arguments are valid for the given
    class, method, or function. It filters out any invalid keyword arguments and returns
    a dictionary containing only the valid ones.

    If the provided object is a class, it inspects the __init__ method to determine the
    valid keyword arguments. If it is a method or function, it inspects the argument names.

    It issues a warning for any invalid keyword arguments if `raise_warning`
    is ``True`` but it does not raise an error.
    
    Examples
    --------
    >>> from gofast.tools.coreutils import get_valid_kwargs
    >>> class MyClass:
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> valid_kwargs = get_valid_kwargs(MyClass, a=1, b=2, c=3)
    >>> print(valid_kwargs)
    {'a': 1, 'b': 2}
    >>> valid_kwargs = get_valid_kwargs(MyClass, raise_warning=True,  **kwargs)
    Warning: 'arg3' is not a valid keyword argument for 'MyClass'.
    >>> print(valid_kwargs)
    {'arg1': 1, 'arg2': 2}

    >>> def my_function(a, b, c):
    ...     return a + b + c
    ...
    >>> kwargs = {'a': 1, 'b': 2, 'd': 3}
    >>> valid_kwargs = get_valid_kwargs(my_function, raise_warning=True, **kwargs)
    Warning: 'd' is not a valid keyword argument for 'my_function'.
    >>> print(valid_kwargs)
    {'a': 1, 'b': 2}
    """
    valid_kwargs = {}
    not_valid_keys = []

    # Determine whether obj_or_func is callable and get its valid arguments
    obj = obj_or_func() if callable(obj_or_func) else obj_or_func
    valid_args = obj.__init__.__code__.co_varnames if hasattr(
        obj, '__init__') else obj.__code__.co_varnames

    # Filter kwargs to separate valid from invalid ones
    for key, value in kwargs.items():
        if key in valid_args:
            valid_kwargs[key] = value
        else:
            not_valid_keys.append(key)

    # Raise a warning for invalid kwargs, if required
    if raise_warning and not_valid_keys:
        warning_msg = (f"Warning: '{', '.join(not_valid_keys)}' "
                       f"{'is' if len(not_valid_keys) == 1 else 'are'} "
                       "not a valid keyword argument "
                       f"for '{obj_or_func.__name__}'.")
        warnings.warn(warning_msg)

    return valid_kwargs
 
def projection_validator (X, Xt=None, columns =None ):
    """ Retrieve x, y coordinates of a datraframe ( X, Xt ) from columns 
    names or indexes. 
    
    If X or Xt are given as arrays, `columns` may hold integers from 
    selecting the the coordinates 'x' and 'y'. 
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to consider as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
      
    Returns 
    -------
    ( x, y, xt, yt ), (xname, yname, xtname, ytname), Tuple of coordinate 
        arrays and coordinate labels 
 
    """
    # initialize arrays and names 
    init_none = [None for i in range (4)]
    x,y, xt, yt = init_none
    xname,yname, xtname, ytname = init_none 
    
    m="{0} must be an iterable object, not {1!r}"
    ms= ("{!r} is given while columns are not supplied. set the list of "
        " feature names or indexes to fetch 'x' and 'y' coordinate arrays." )
    
    # validate X if X is np.array or dataframe 
    X =_assert_all_types(X, np.ndarray, pd.DataFrame ) 
    
    if Xt is not None: 
        # validate Xt if Xt is np.array or dataframe 
        Xt = _assert_all_types(Xt, np.ndarray, pd.DataFrame)
        
    if columns is not None: 
        if isinstance (columns, str): 
            columns = str2columns(columns )
        
        if not is_iterable(columns): 
            raise ValueError(m.format('columns', type(columns).__name__))
        
        columns = list(columns) + [ None for i in range (5)]
        xname , yname, xtname, ytname , *_= columns 

    if isinstance(X, pd.DataFrame):
        x, xname, y, yname = _validate_columns(X, [xname, yname])
        
    elif isinstance(X, np.ndarray):
        x, y = _is_valid_coordinate_arrays (X, xname, yname )    
        
        
    if isinstance (Xt, pd.DataFrame) :
        # the test set holds the same feature names
        # as the train set 
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, xtname, yt, ytname = _validate_columns(Xt, [xname, yname])

    elif isinstance(Xt, np.ndarray):
        
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, yt = _is_valid_coordinate_arrays (Xt, xtname, ytname , 'test')
        
    if (x is None) or (y is None): 
        raise ValueError (ms.format('X'))
    if Xt is not None: 
        if (xt is None) or (yt is None): 
            warnings.warn (ms.format('Xt'))

    return  (x, y , xt, yt ) , (
        xname, yname, xtname, ytname ) 
    
def _validate_columns0 (df, xni, yni ): 
    """ Validate the feature name  in the dataframe using either the 
    string litteral name of the index position in the columns.
    
    :param df: pandas.DataFrame- Dataframe with feature names as columns. 
    :param xni: str, int- feature name  or position index in the columns for 
        x-coordinate 
    :param yni: str, int- feature name  or position index in the columns for 
        y-coordinate 
    
    :returns: (x, ni) Tuple of (pandas.Series, and names) for x and y 
        coordinates respectively.
    
    """
    def _r (ni): 
        if isinstance(ni, str): # feature name
            exist_features(df, ni ) 
            s = df[ni]  
        elif isinstance (ni, (int, float)):# feature index
            s= df.iloc[:, int(ni)] 
            ni = s.name 
        return s, ni 
        
    xs , ys = [None, None ]
    if df.ndim ==1: 
        raise ValueError ("Expect a dataframe of two dimensions, got '1'")
        
    elif df.shape[1]==2: 
       warnings.warn("columns are not specify while array has dimension"
                     "equals to 2. Expect indexes 0 and 1 for (x, y)"
                     "coordinates respectively.")
       xni= df.iloc[:, 0].name 
       yni= df.iloc[:, 1].name 
    else: 
        ms = ("The matrix of features is greater than 2. Need column names or"
              " indexes to  retrieve the 'x' and 'y' coordinate arrays." ) 
        e =' Only {!r} is given.' 
        me=''
        if xni is not None: 
            me =e.format(xni)
        if yni is not None: 
            me=e.format(yni)
           
        if (xni is None) or (yni is None ): 
            raise ValueError (ms + me)
            
    xs, xni = _r (xni) ;  ys, yni = _r (yni)
  
    return xs, xni , ys, yni 

def _validate_array_indexer (arr, index): 
    """ Select the appropriate coordinates (x,y) arrays from indexes.  
    
    Index is used  to retrieve the array of (x, y) coordinates if dimension 
    of `arr` is greater than 2. Since we expect x, y coordinate for projecting 
    coordinates, 1-d  array `X` is not acceptable. 
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
    :param index: int, index to fetch x, and y coordinates in multi-dimension
        arrays. 
    :returns: arr- x or y coordinates arrays. 

    """
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
    if not isinstance (index, (float, int)): 
        raise ValueError("index is needed to coordinate array with "
                         "dimension greater than 2.")
        
    return arr[:, int (index) ]

def _is_valid_coordinate_arrays (arr, xind, yind, ptype ='train'): 
    """ Check whether array is suitable for projecting i.e. whether 
    x and y (both coordinates) can be retrived from `arr`.
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
        
    :param xind: int, index to fetch x-coordinate in multi-dimension
        arrays. 
    :param yind: int, index to fetch y-coordinate in multi-dimension
        arrays
    :param ptype: str, default='train', specify whether the array passed is 
        training or test sets. 
    :returns: (x, y)- array-like of x and y coordinates. 
    
    """
    xn, yn =('x', 'y') if ptype =='train' else ('xt', 'yt') 
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
        
    elif arr.shape[1] ==2 : 
        x, y = arr[:, 0], arr[:, 1]
        
    else :
        msg=("The matrix of features is greater than 2; Need index to  "
             " retrieve the {!r} coordinate array in param 'column'.")
        
        if xind is None: 
            raise ValueError(msg.format(xn))
        else : x = _validate_array_indexer(arr, xind)
        if yind is None : 
            raise ValueError(msg.format(yn))
        else : y = _validate_array_indexer(arr, yind)
        
    return x, y         

def extract_coordinates(X, Xt=None, columns=None):
    """
    Extracts 'x' and 'y' coordinate arrays from training (X) and optionally
    test (Xt) datasets. 
    
    Supports input as NumPy arrays or pandas DataFrames. When dealing
    with DataFrames, `columns` can specify which columns to use for coordinates.

    Parameters
    ----------
    X : ndarray or DataFrame
        Training dataset with shape (M, N) where M is the number of samples and
        N is the number of features. It represents the observed data used as
        independent variables in learning.
    Xt : ndarray or DataFrame, optional
        Test dataset with shape (M, N) where M is the number of samples and
        N is the number of features. It represents the data observed at testing
        and prediction time, used as independent variables in learning.
    columns : list of str or int, optional
        Specifies the columns to use for 'x' and 'y' coordinates. Necessary when
        X or Xt are DataFrames with more than 2 dimensions or when selecting specific
        features from NumPy arrays.

    Returns
    -------
    tuple of arrays
        A tuple containing the 'x' and 'y' coordinates from the training set and, 
        if provided, the test set. Formatted as (x, y, xt, yt).
    tuple of str or None
        A tuple containing the names or indices of the 'x' and 'y' columns 
        for the training and test sets. Formatted as (xname, yname, xtname, ytname).
        Values are None if not applicable or not provided.

    Raises
    ------
    ValueError
        If `columns` is not iterable, not provided for DataFrames with more 
        than 2 dimensions, or if X or Xt cannot be validated as coordinate arrays.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import extract_coordinates
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Xt = np.array([[5, 6], [7, 8]])
    >>> extract_coordinates(X, Xt )
    ((array([1, 3]), array([2, 4]), array([5, 7]), array([6, 8])), (0, 1, 0, 1))
    """
    if columns is None: 
        if not isinstance ( X, pd.DataFrame) and X.shape[1]!=2: 
            raise ValueError("Columns cannot be None when array is passed.")
        if isinstance(X, np.ndarray) and X.shape[1]==2: 
            columns =[0, 1] 
    
    columns = columns or ( list(X.columns) if isinstance (
        X, pd.DataFrame ) else columns )
    
    if columns is None :
        raise ValueError("Columns parameter is required to specify"
                         " 'x' and 'y' coordinates.")
    
    if not isinstance(columns, (list, tuple)) or len(columns) != 2:
        raise ValueError("Columns parameter must be a list or tuple with "
                         "exactly two elements for 'x' and 'y' coordinates.")
    
    # Process training dataset
    x, y, xname, yname = _process_dataset(X, columns)
    
    # Process test dataset, if provided
    if Xt is not None:
        xt, yt, xtname, ytname = _process_dataset(Xt, columns)
    else:
        xt, yt, xtname, ytname = None, None, None, None

    return (x, y, xt, yt), (xname, yname, xtname, ytname)
       
def _validate_columns(df, columns):
    """
    Validates and extracts x, y coordinates from a DataFrame based on column 
    names or indices.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to extract coordinate columns.
    columns : list of str or int
        The names or indices of the columns to extract as coordinates.
    
    Returns
    -------
    x, xname, y, yname : (pandas.Series, str/int, pandas.Series, str/int)
        The extracted x and y coordinate Series along with their column
        names or indices.
    
    Raises
    ------
    ValueError
        If the specified columns are not found in the DataFrame or if the 
        columns list is not correctly specified.
    """
    if not isinstance(columns, (list, tuple)) or len(columns) < 2:
        raise ValueError("Columns parameter must be a list or tuple with at"
                         " least two elements.")
    
    try:
        xname, yname = columns[0], columns[1]
        x = df[xname] if isinstance(xname, str) else df.iloc[:, xname]
        y = df[yname] if isinstance(yname, str) else df.iloc[:, yname]
    except Exception as e:
        raise ValueError(f"Error extracting columns: {e}")
    
    return x, xname, y, yname

def _process_dataset(dataset, columns):
    """
    Processes the dataset (X or Xt) to extract 'x' and 'y' coordinates based 
    on provided column names or indices.
    
    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        The dataset from which to extract 'x' and 'y' coordinates.
    columns : list of str or int
        The names or indices of the columns to extract as coordinates. 
        For ndarray, integers are expected.
    
    Returns
    -------
    x, y, xname, yname : (numpy.array or pandas.Series, numpy.array or 
                          pandas.Series, str/int, str/int)
        The extracted 'x' and 'y' coordinates, along with their column names 
        or indices.
    
    Raises
    ------
    ValueError
        If the dataset or columns are not properly specified.
    """
    if isinstance(dataset, pd.DataFrame):
        x, xname, y, yname = _validate_columns(dataset, columns)
        return x.to_numpy(), y.to_numpy(), xname, yname
    elif isinstance(dataset, np.ndarray):
        if not isinstance(columns, (list, tuple)) or len(columns) < 2:
            raise ValueError("For ndarray, columns must be a list or tuple "
                             "with at least two indices.")
        xindex, yindex = columns[0], columns[1]
        x, y = dataset[:, xindex], dataset[:, yindex]
        return x, y, xindex, yindex
    else:
        raise ValueError("Dataset must be a pandas.DataFrame or numpy.ndarray.")

def validate_feature(data: Union[DataFrame, Series],  features: List[str],
                     verbose: str = 'raise') -> bool:
    """
    Validate the existence of specified features in a DataFrame or Series.

    Parameters
    ----------
    data : DataFrame or Series
        The DataFrame or Series to validate feature existence.
    features : list of str
        List of features to check for existence in the data.
    verbose : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'raise' (default) will raise
        a ValueError if any feature is missing, while 'ignore' will return a
        boolean indicating whether all features exist.

    Returns
    -------
    bool
        True if all specified features exist in the data, False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import validate_feature
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> result = validate_feature(data, ['A', 'C'], verbose='raise')
    >>> print(result)  # This will raise a ValueError
    """
    if isinstance(data, pd.Series):
        data = data.to_frame().T  # Convert Series to DataFrame
    features= is_iterable(features, exclude_string= True, transform =True )
    present_features = set(features).intersection(data.columns)

    if len(present_features) != len(features):
        missing_features = set(features).difference(present_features)
        if verbose == 'raise':
            raise ValueError("The following features are missing in the "
                             f"data: {smart_format(missing_features)}.")
        return False

    return True

def features_in(
    *data: Union[pd.DataFrame, pd.Series], features: List[str],
    error: str = 'ignore') -> List[bool]:
    """
    Control whether the specified features exist in multiple datasets.

    Parameters
    ----------
    *data : DataFrame or Series arguments
        Multiple DataFrames or Series to check for feature existence.
    features : list of str
        List of features to check for existence in the datasets.
    error : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'ignore' (default) will ignore
        a ValueError for each dataset with missing features, while 'ignore' will
        return a list of booleans indicating whether all features exist in each dataset.

    Returns
    -------
    list of bool
        A list of booleans indicating whether the specified features exist in each dataset.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.coreutils import features_in
    >>> data1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> data2 = pd.Series([5, 6], name='C')
    >>> data3 = pd.DataFrame({'X': [7, 8]})
    >>> features = ['A', 'C']
    >>> results1 = features_in(data1, data2, features, error='raise')
    >>> print(results1)  # This will raise a ValueError for the first dataset
    >>> results2 = features_in(data1, data3, features, error='ignore')
    >>> print(results2)  # This will return [True, False]
    """
    results = []

    for dataset in data:
        results.append(validate_feature(dataset, features, verbose=error))

    return results

def find_features_in(
    data: DataFrame = None,
    features: List[str] = None,
    parse_features: bool = False,
    return_frames: bool = False,
) -> Tuple[Union[List[str], DataFrame], Union[List[str], DataFrame]]:
    """
    Retrieve the categorical or numerical features from the dataset.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame with columns representing the features.
    features : list of str, optional
        List of column names. If provided, the DataFrame will be restricted
        to only include the specified features before searching for numerical
        and categorical features. An error will be raised if any specified
        feature is missing in the DataFrame.
    return_frames : bool, optional
        If True, it returns two separate DataFrames (cat & num). Otherwise, it
        returns only the column names of categorical and numerical features.
    parse_features : bool, default False
        Use default parsers to parse string items into an iterable object.

    Returns
    -------
    Tuple : List[str] or DataFrame
        The names or DataFrames of categorical and numerical features.

    Examples
    --------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.tools.mlutils import find_features_in
    >>> data = fetch_data('bagoue').frame 
    >>> cat, num = find_features_in(data)
    >>> cat, num
    ... (['type', 'geol', 'shape', 'name', 'flow'],
    ...  ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    >>> cat, num = find_features_in(data, features=['geol', 'ohmS', 'sfi'])
    >>> cat, num
    ... (['geol'], ['ohmS', 'sfi'])
    """
    if not isinstance (data, pd.DataFrame):
        raise TypeError(f"Expect a DataFrame. Got {type(data).__name__!r}")

    if features is not None:
        features = list(
            is_iterable(
                features,
                exclude_string=True,
                transform=True,
                parse_string=parse_features,
            )
        )

    if features is None:
        features = list(data.columns)

    validate_feature(data, list(features))
    data = data[features].copy()

    # Get numerical features
    data, numnames, catnames = to_numeric_dtypes(data, return_feature_types=True )

    if catnames is None:
        catnames = []

    return (data[catnames], data[numnames]) if return_frames else (
        list(catnames), list(numnames)
    )
  
def split_train_test(
        data: DataFrame, test_ratio: float = 0.2
        ) -> Tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into train and test sets based on a given ratio.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the features.
    test_ratio : float, optional
        The ratio of the test set, ranging from 0 to 1. Default is 0.2 (20%).

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        A tuple of the train set and test set DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> from gofast.tools.coreutils import split_train_test
    >>> data = load_iris(as_frame=True)['data']
    >>> train_set, test_set = split_train_test(data, test_ratio=0.2)
    >>> len(train_set), len(test_set)
    ... (120, 30)
    """

    test_ratio = assert_ratio(test_ratio)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check_id(
        identifier: int, test_ratio: float, hash: _F[_T]) -> bool:
    """
    Check if an instance should be in the test set based on its unique identifier.

    Parameters
    ----------
    identifier : int
        A unique identifier for the instance.
    test_ratio : float, optional
        The ratio of instances to put in the test set. Default is 0.2 (20%).
    hash : callable
        A hash function to generate a hash from the identifier.
        Secure hashes and message digests algorithm. Can be 
        SHA1, SHA224, SHA256, SHA384, and SHA512 (defined in FIPS 180-2) 
        as well as RSA’s MD5 algorithm (defined in Internet RFC 1321). 
        
        Please refer to :ref:`<https://docs.python.org/3/library/hashlib.html>` 
        for futher details.

    Returns
    -------
    bool
        True if the instance should be in the test set, False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import test_set_check_id
    >>> test_set_check_id(42, test_ratio=0.2, hash=hashlib.md5)
    ... False
    """
    # def test_set_check_id(identifier: str, ratio: float, hash_function: _F) -> bool:
    #     """Determines if an identifier belongs to the test set using the hash value."""
    #     # Convert identifier to string and hash
    #     hash_val = int(hash_function(str(identifier).encode()).hexdigest(), 16)
    #     # Use the hash value to decide test set membership
    #     return hash_val % 10000 / 10000.0 < ratio
    
    #     hashed_id = hash_function(identifier.encode('utf-8')).digest()
    #     return np.frombuffer(hashed_id, dtype=np.uint8).sum() < 256 * test_ratio
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(
    data: DataFrame, test_ratio: float, id_column: Optional[List[str]] = None,
    keep_colindex: bool = True, hash: _F = hashlib.md5
) -> Tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into train and test sets while ensuring data consistency
    by using specified id columns or the DataFrame's index as unique identifiers.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the features.
    test_ratio : float
        The ratio of instances to include in the test set.
    id_column : list of str, optional
        Column names to use as unique identifiers. If None, the DataFrame's index
        is used as the identifier.
    keep_colindex : bool, optional
        Determines whether to keep or drop the index column after resetting.
        This parameter is only applicable if id_column is None and the DataFrame's
        index is reset. Default is True.
    hash : callable
        A hash function to generate a hash from the identifier.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        A tuple containing the train and test set DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.coreutils import split_train_test_by_id
    >>> data = pd.DataFrame({'ID': [1, 2, 3, 4, 5], 'Value': [10, 20, 30, 40, 50]})
    >>> train_set, test_set = split_train_test_by_id(data, test_ratio=0.2, id_column=['ID'])
    >>> len(train_set), len(test_set)
    (4, 1)
    """
    drop_tmp_index=False
    if id_column is None:
        # Check if the index is integer-based; if not, create a temporary integer index.
        if not data.index.is_integer():
            data['_tmp_hash_index'] = np.arange(len(data))
            ids = data['_tmp_hash_index']
            drop_tmp_index = True
        else:
            ids = data.index.to_series()
            drop_tmp_index = False
    else:
        # Use specified id columns as unique identifiers, combining them if necessary.
        ids = data[id_column].astype(str).apply(
            lambda row: '_'.join(row), axis=1) if isinstance(
                id_column, list) else data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check_id(id_, test_ratio, hash))

    train_set = data.loc[~in_test_set].copy()
    test_set = data.loc[in_test_set].copy()

    if drop_tmp_index or (id_column is None and not keep_colindex):
        # Remove the temporary index or reset the index as needed
        train_set.drop(columns=['_tmp_hash_index'], errors='ignore', inplace=True)
        test_set.drop(columns=['_tmp_hash_index'], errors='ignore', inplace=True)
        # for consistency if '_tmp_has_index' 
        if '_tmp_hash_index' in data.columns: 
            data.drop (columns='_tmp_hash_index', inplace =True)
    elif id_column is None and keep_colindex:
        # If keeping the original index and it was integer-based, no action needed
        pass

    return train_set, test_set

def parallelize_jobs(
    function: _F,
    tasks: Sequence[Dict[str, Any]] = (),
    n_jobs: Optional[int] = None,
    executor_type: str = 'process') -> list:
    """
    Parallelize the execution of a callable across multiple processors, 
    supporting both positional and keyword arguments.

    Parameters
    ----------
    function : Callable[..., Any]
        The function to execute in parallel. This function must be picklable 
        if using `executor_type='process'`.
    tasks : Sequence[Dict[str, Any]], optional
        A sequence of dictionaries, where each dictionary contains 
        two keys: 'args' (a tuple) for positional arguments,
        and 'kwargs' (a dict) for keyword arguments, for one execution of
        `function`. Defaults to an empty sequence.
    n_jobs : Optional[int], optional
        The number of jobs to run in parallel. `None` or `1` uses a single 
        processor, any positive integer specifies the
        exact number of processors to use, `-1` uses all available processors. 
        Default is None (1 processor).
    executor_type : str, optional
        The type of executor to use. Can be 'process' for CPU-bound tasks or
        'thread' for I/O-bound tasks. Default is 'process'.

    Returns
    -------
    list
        A list of results from the function executions.

    Raises
    ------
    ValueError
        If `function` is not picklable when using 'process' as `executor_type`.

    Examples
    --------
    >>> from gofast.tools.coreutils import parallelize_jobs
    >>> def greet(name, greeting='Hello'):
    ...     return f"{greeting}, {name}!"
    >>> tasks = [
    ...     {'args': ('John',), 'kwargs': {'greeting': 'Hi'}},
    ...     {'args': ('Jane',), 'kwargs': {}}
    ... ]
    >>> results = parallelize_jobs(greet, tasks, n_jobs=2)
    >>> print(results)
    ['Hi, John!', 'Hello, Jane!']
    """
    if executor_type == 'process':
        import_optional_dependency("cloudpickle")
        import cloudpickle
        try:
            cloudpickle.dumps(function)
        except cloudpickle.PicklingError:
            raise ValueError("The function to be parallelized must be "
                             "picklable when using 'process' executor.")

    num_workers = multiprocessing.cpu_count() if n_jobs == -1 else (
        1 if n_jobs is None else n_jobs)
    
    ExecutorClass = ProcessPoolExecutor if executor_type == 'process' \
        else ThreadPoolExecutor
    
    results = []
    with ExecutorClass(max_workers=num_workers) as executor:
        futures = [executor.submit(function, *task.get('args', ()),
                                   **task.get('kwargs', {})) for task in tasks]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results
 
def denormalize(
    data: ArrayLike, min_value: float, max_value: float
    ) -> ArrayLike:
    """
    Denormalizes data from a normalized scale back to its original scale.

    This function is useful when data has been normalized to a different 
    scale (e.g., [0, 1]) and needs to be converted back to its original scale 
    for interpretation or further processing.

    Parameters
    ----------
    data : np.ndarray
        The data to be denormalized, assumed to be a NumPy array.
    min_value : float
        The minimum value of the original scale before normalization.
    max_value : float
        The maximum value of the original scale before normalization.

    Returns
    -------
    np.ndarray
        The denormalized data, converted back to its original scale.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.coreutils import denormalize
    >>> normalized_data = np.array([0, 0.5, 1])
    >>> min_value = 10
    >>> max_value = 20
    >>> denormalized_data = denormalize(normalized_data, min_value, max_value)
    >>> print(denormalized_data)
    [10. 15. 20.]

    Note
    ----
    The denormalization process is the inverse of normalization and is applied
    to data that was previously normalized according to the formula:
        `data_norm = (data - min_value) / (max_value - min_value)`
    The denormalize function uses the inverse of this formula to restore the data.
    """
    if not isinstance (data, (pd.Series, pd.DataFrame)): 
        data = np.asarray( data )
        
    return data * (max_value - min_value) + min_value
   
def download_progress_hook(t):
    """
    Hook to update the tqdm progress bar during a download.

    Parameters
    ----------
    t : tqdm
        An instance of tqdm to update as the download progresses.
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        Updates the progress bar.

        Parameters
        ----------
        b : int
            Number of blocks transferred so far [default: 1].
        bsize : int
            Size of each block (in tqdm-compatible units) [default: 1].
        tsize : int, optional
            Total size (in tqdm-compatible units). If None, remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to 

def squeeze_specific_dim(
    arr: np.ndarray, axis: Optional[int] = -1
    ) -> np.ndarray:
    """
    Squeeze specific dimensions of a NumPy array based on the axis parameter.
    
    This function provides a flexible way to remove single-dimensional entries
    from the shape of an array. By default, it targets the last dimension,
    but can be configured to squeeze any specified dimension or all single-dimension
    axes if `axis` is set to None.

    Parameters
    ----------
    arr : np.ndarray
        The input array to potentially squeeze.
    axis : Optional[int], default -1
        The specific axis to squeeze. If the size of this axis is 1, it will be
        removed from the array. If `axis` is None, all single-dimension axes are
        squeezed. If `axis` is set to a specific dimension (0, 1, ..., arr.ndim-1),
        only that dimension will be squeezed if its size is 1.

    Returns
    -------
    np.ndarray
        The array with the specified dimension squeezed if its size was 1,
        otherwise the original array. If `axis` is None, all single-dimension
        axes are squeezed.

    Examples
    --------
    Squeeze the last dimension:

    >>> from gofast.tools.coreutils import squeeze_specific_dim
    >>> arr = np.array([[1], [2], [3]])
    >>> print(squeeze_specific_dim(arr).shape)
    (3,)

    Squeeze all single-dimension axes:

    >>> arr = np.array([[[1], [2], [3]]])
    >>> print(squeeze_specific_dim(arr, None).shape)
    (3,)

    Squeeze a specific dimension (e.g., first dimension of a 3D array):

    >>> arr = np.array([[[1, 2, 3]]])
    >>> print(squeeze_specific_dim(arr, 0).shape)
    ([[1, 2, 3]])

    Not squeezing if the specified axis does not have size 1:

    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print(squeeze_specific_dim(arr, 0).shape)
    [[1, 2, 3], [4, 5, 6]]
    """
    if axis is None:
        # Squeeze all single-dimension axes
        return np.squeeze(arr)
    else:
        # Check if the specified axis is a single-dimension axis and squeeze it
        try:
            return np.squeeze(arr, axis=axis)
        except ValueError:
            # Return the array unchanged if squeezing is not applicable
            return arr

def contains_delimiter(s: str, delimiters: Union[str, list, set]) -> bool:
    """
    Checks if the given string contains any of the specified delimiters.

    Parameters
    ----------
    s : str
        The string to check.
    delimiters : str, list, or set
        Delimiters to check for in the string. Can be specified as a single
        string (for a single delimiter), a list of strings, or a set of strings.

    Returns
    -------
    bool
        True if the string contains any of the delimiters, False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import contains_delimiter
    >>> contains_delimiter("example__string", "__")
    True

    >>> contains_delimiter("example--string", ["__", "--", "&", "@", "!"])
    True

    >>> contains_delimiter("example&string", {"__", "--", "&", "@", "!"})
    True

    >>> contains_delimiter("example@string", "__--&@!")
    True

    >>> contains_delimiter("example_string", {"__", "--", "&", "@", "!"})
    False

    >>> contains_delimiter("example#string", "#$%")
    True

    >>> contains_delimiter("example$string", ["#", "$", "%"])
    True

    >>> contains_delimiter("example%string", "#$%")
    True

    >>> contains_delimiter("example^string", ["#", "$", "%"])
    False
    """
    # for consistency
    s = str(s) 
    # Convert delimiters to a set if it's not already a set
    if not isinstance(delimiters, set):
        if isinstance(delimiters, str):
            delimiters = set(delimiters)
        else:  # Assuming it's a list or similar iterable
            delimiters = set(delimiters)
    
    return any(delimiter in s for delimiter in delimiters)    
    
def convert_to_structured_format(
        *arrays: Any, as_frame: bool = True, 
        skip_sparse: bool =True, 
        ) -> List[Union[ArrayLike, DataFrame, Series]]:
    """
    Converts input objects to structured numpy arrays or pandas DataFrame/Series
    based on their shapes and the `as_frame` flag. If conversion to a structured
    format fails, the original objects are returned. When `as_frame` is False,
    attempts are made to convert inputs to numpy arrays.
    
    Parameters
    ----------
    *arrays : Any
        A variable number of objects to potentially convert. These can be lists,
        tuples, or numpy arrays.
    as_frame : bool, default=True
        If True, attempts to convert arrays to DataFrame or Series; otherwise,
        attempts to standardize as numpy arrays.
    skip_sparse: bool, default=True 
        Dont convert any sparse matrix and keept it as is. 
    
    Returns
    -------
    List[Union[np.ndarray, pd.DataFrame, pd.Series]]
        A list containing the original objects, numpy arrays, DataFrames, or
        Series, depending on each object's structure and the `as_frame` flag.
    
    Examples
    --------
    Converting to pandas DataFrame/Series:
    >>> from gofast.tools.coreutils import convert_to_structured_format
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> features= {"feature_1": range (7), "feature_2":['1', 2, 9, 35, "0", "76", 'r']}
    >>> target= pd.Series(data=range(10), name="target")
    >>> convert_to_structured_format( features, target, as_frame=True)
    >>> arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> arr2 = np.array([7, 8, 9])
    >>> convert_to_structured_format(arr1, arr2, as_frame=True)
    [   DataFrame:
            0  1  2
        0   1  2  3
        1   4  5  6,
        Series:
        0    7
        1    8
        2    9
    ]

    Standardizing as numpy arrays:
    >>> list1 = [10, 11, 12]
    >>> tuple1 = (13, 14, 15)
    >>> convert_to_structured_format(list1, tuple1, as_frame=False)
    [   array([10, 11, 12]),
        array([13, 14, 15])
    ]
    """

    def attempt_conversion_to_numpy(arr: Any) -> np.ndarray:
        """Attempts to convert an object to a numpy array."""
        try:
            return np.array(arr)
        except Exception:
            return arr

    def attempt_conversion_to_pandas(
            arr: np.ndarray) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """Attempts to convert an array to a DataFrame or Series based on shape."""
        from scipy.sparse import issparse
        try:
            if issparse(arr) and skip_sparse: 
                raise # dont perform any convertion 
            if hasattr(arr, '__array__'): 
                if arr.ndim == 1:
                    return pd.Series(arr)
                elif arr.ndim == 2:
                    if arr.shape[1] == 1:
                        return pd.Series(arr.squeeze())
                    else:
                        return pd.DataFrame(arr)
            else: 
                return pd.DataFrame(arr)
        except Exception:
            pass
        return arr

    if as_frame:
        return [attempt_conversion_to_pandas(arr) for arr in arrays]
    else:
        # Try to convert everything to numpy arrays, return as is if it fails
        return [attempt_conversion_to_numpy(attempt_conversion_to_pandas(arr)
                                            ) for arr in arrays]

def resample_data(*data: Any, samples: Union[int, float, str] = 1,
                  replace: bool = False, random_state: int = None,
                  shuffle: bool = True) -> List[Any]:
    """
    Resample multiple data structures (arrays, sparse matrices, Series, DataFrames)
    based on specified sample size or ratio.

    Parameters
    ----------
    *data_structures : Any
        Variable number of array-like, sparse matrix, pandas Series, or DataFrame
        objects to be resampled.
    samples : int, float, or str, optional
        Number of items to sample or a ratio of items to sample (if < 1 or
        expressed as a percentage string, e.g., "50%"). Defaults to 1, meaning
        no resampling unless specified.
    replace : bool, default=False
        Whether sampling of the same row more than once is allowed.
    random_state : int, optional
        Seed for the random number generator for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle data before sampling.

    Returns
    -------
    List[Any]
        A list of resampled data structures, matching the input order. Each
        structure is resampled according to the `samples` parameter.

    Examples
    --------
    >>> from gofast.tools.coreutils import resample_data
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris(as_frame=True)
    >>> data, target = iris.data, iris.target
    >>> resampled_data, resampled_target = resample_data(data, target, samples=0.5,
    ...                                                  random_state=42)
    >>> print(resampled_data.shape, resampled_target.shape)
    """
    resampled_structures = []
    
    for d in data:
        # Correct way to access the shape of the sparse matrix 
        # encapsulated in a numpy array
        try: 
            if d.dtype == object and scipy.sparse.issparse(d.item()):
                d = d.item()  # Access the sparse matrix
            # Now you can safely access the number of rows
        except:
            # Fallback for regular numpy arrays/data or 
            # directly accessible sparse matrices
           pass 
        n_samples = _determine_sample_size(d, samples, is_percent="%" in str(samples))
        sampled_d = _perform_sampling(d, n_samples, replace, random_state, shuffle)
        resampled_structures.append(sampled_d)

    return resampled_structures

def _determine_sample_size(d: Any, samples: Union[int, float, str], 
                           is_percent: bool) -> int:
    """
    Determine the number of samples to draw based on the input size or ratio.
    """
    if isinstance(samples, str) and is_percent:
        samples = samples.replace("%", "")
    try:
        samples = float(samples)
    except ValueError:
        raise TypeError(f"Invalid type for 'samples': {type(samples).__name__}."
                        " Expected int, float, or percentage string.")
   
    d_length = d.shape[0] if hasattr(d, 'shape') else len(d)
    if samples < 1 or is_percent:
        return max(1, int(samples * d_length))
    return int(samples)

def _perform_sampling(d: Any, n_samples: int, replace: bool, 
                      random_state: int, shuffle: bool) -> Any:
    """
    Perform the actual sampling operation on the data structure.
    """
    if isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.sample(n=n_samples, replace=replace, random_state=random_state
                        ) if shuffle else d.iloc[:n_samples]
    elif scipy.sparse.issparse(d):
        if scipy.sparse.isspmatrix_coo(d):
            warnings.warn("coo_matrix does not support indexing. Conversion"
                          " to CSR matrix is recommended.")
            d = d.tocsr()
        indices = np.random.choice(d.shape[0], n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d[indices]
    else:
        d_array = np.array(d) if not hasattr(d, '__array__') else d
        indices = np.random.choice(len(d_array), n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d_array[indices] if d_array.ndim == 1 else d_array[indices, :]


def get_valid_key(input_key, default_key, substitute_key_dict=None,
                  regex_pattern = "[#&*@!,;\s]\s*", deep_search=True):
    """
    Validates an input key and substitutes it with a valid key if necessary,
    based on a mapping of valid keys to their possible substitutes. If the input
    key is not provided or is invalid, a default key is used.

    Parameters
    ----------
    input_key : str
        The key to validate and possibly substitute.
    default_key : str
        The default key to use if input_key is None, empty, or not found in 
        the substitute mapping.
    substitute_key_dict : dict, optional
        A mapping of valid keys to lists of their possible substitutes. This
        allows for flexible key substitution and validation.
    regex_pattern: str, default = '[#&*@!,;\s-]\s*'
        The base pattern to split the text into a columns
    deep_search: bool, default=False 
       If deep-search, the key finder is no sensistive to lower/upper case 
       or whether a numeric data is included. 
    Returns
    -------
    str
        A valid key, which is either the original input_key if valid, a substituted
        key if the original was found in the substitute mappings, or the default_key.

    Notes
    -----
    This function also leverages an external validation through `key_checker` for
    a deep search validation, ensuring the returned key is within the set of valid keys.
    
    Example
    -------
    >>> from gofast.tools.coreutils import get_valid_key
    >>> substitute_key_dict = {'valid_key1': ['vk1', 'key1'], 'valid_key2': ['vk2', 'key2']}
    >>> get_valid_key('vk1', 'default_key', substitute_key_dict)
    'valid_key1'
    >>> get_valid_key('unknown_key', 'default_key', substitute_key_dict)
    'KeyError...'
  
    """
    # Ensure substitute_mapping is a dictionary if not provided
    substitute_key_dict = substitute_key_dict or {}

    # Fallback to default_key if input_key is None or empty
    input_key = input_key or default_key

    # Attempt to find a valid substitute for the input_key
    for valid_key, substitutes in substitute_key_dict.items():
        # Case-insensitive comparison for substitutes
        normalized_substitutes = [str(sub).lower() for sub in substitutes]
        
        if str(input_key).lower() in normalized_substitutes:
            input_key = valid_key
            break
    
    regex = re.compile (fr'{regex_pattern}', flags=re.IGNORECASE)
    # use valid keys  only if substitute_key_dict not provided. 
    valid_keys = substitute_key_dict.keys() if substitute_key_dict else is_iterable(
            default_key, exclude_string=True, transform=True)
    valid_keys = set (list(valid_keys) + [default_key])
    # Further validate the (possibly substituted) input_key
    input_key = key_checker(input_key, valid_keys=valid_keys,
                            deep_search=deep_search,regex = regex  )
    
    return input_key

def process_and_extract_data(
    *args: ArrayLike, 
    columns: Optional[List[Union[str, int]]] = None,
    enforce_extraction: bool = True, 
    allow_split: bool = False, 
    search_multiple: bool = False,
    ensure_uniform_length: bool = False, 
    to_array: bool = False,
    on_error: str = 'raise',
) -> List[np.ndarray]:
    """
    Extracts and processes data from various input types, focusing on column 
    extraction from pandas DataFrames and conversion of inputs to numpy 
    arrays or pandas Series.

    Parameters
    ----------
    *args : ArrayLike
        A variable number of inputs, each can be a list, numpy array, pandas 
        Series,dictionary, or pandas DataFrame.
    columns : List[Union[str, int]], optional
        Specific columns to extract from pandas DataFrames. If not provided, 
        the function behaves differently based on `allow_split`.
    enforce_extraction : bool, default=True
        Forces the function to try extracting `columns` from DataFrames. 
        If False, DataFrames are returned without column extraction unless 
        `allow_split` is True.
        Removing non-conforming elements if True.
    allow_split : bool, default=False
        If True and a DataFrame is provided without `columns`, splits the 
        DataFrame into its constituent columns.
    search_multiple : bool, default=False
        Allows searching for `columns` across multiple DataFrame inputs. Once 
        a column is found, it is not searched for in subsequent DataFrames.
    ensure_uniform_length : bool, default=False
        Checks that all extracted arrays have the same length. Raises an error
        if they don't.
    to_array : bool, default=False
        Converts all extracted pandas Series to numpy arrays.
    on_error : str, {'raise', 'ignore'}, default='raise'
        Determines how to handle errors during column extraction or when 
        enforcing uniform length. 'raise' will raise an error, 'ignore' will 
        skip the problematic input.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays or pandas Series extracted based 
        on the specified conditions.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.coreutils import process_and_extract_data
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> process_and_extract_data(data, columns=['A'], to_array=True)
    [array([1, 2, 3])]

    Splitting DataFrame into individual arrays:

    >>> process_and_extract_data(data, allow_split=True, to_array=True)
    [array([1, 2, 3]), array([4, 5, 6])]

    Extracting columns from multiple DataFrames:

    >>> data2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
    >>> process_and_extract_data(data, data2, columns=['A', 'C'], 
                                  search_multiple=True, to_array=True)
    [array([1, 2, 3]), array([7, 8, 9])]

    Handling mixed data types:

    >>> process_and_extract_data([1, 2, 3], {'E': [13, 14, 15]}, to_array=True)
    [array([1, 2, 3]), array([13, 14, 15])]
    
    Extracting columns from multiple DataFrames and enforcing uniform length:
    >>> data2 = pd.DataFrame({'C': [7, 8, 9, 10], 'D': [11, 12, 13, 14]})
    >>> result = process_and_extract_data(
        data, data2, columns=['A', 'C'],search_multiple=True,
        ensure_uniform_length=True, to_array=True)
    ValueError: Extracted data arrays do not have uniform length.
    """
    extracted_data = []
    columns_found: Set[Union[str, int]] = set()

    def _process_input(
            input_data: ArrayLike,
            target_columns: Optional[List[Union[str, int]]], 
            to_array: bool) -> Optional[np.ndarray]:
        """
        Processes each input based on its type, extracting specified columns 
        if necessary, and converting to numpy array if specified.
        """
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data)
            return input_data if len(input_data.shape
                                     ) == 1 or not enforce_extraction else None

        elif isinstance(input_data, dict):
            input_data = pd.DataFrame(input_data)

        if isinstance(input_data, pd.DataFrame):
            if target_columns:
                for col in target_columns:
                    if col in input_data.columns and (
                            search_multiple or col not in columns_found):
                        data_to_add = input_data[col].to_numpy(
                            ) if to_array else input_data[col]
                        extracted_data.append(data_to_add)
                        columns_found.add(col)
                    elif on_error == 'raise':
                        raise ValueError(f"Column {col} not found in DataFrame.")
            elif allow_split:
                for col in input_data.columns:
                    data_to_add = input_data[col].to_numpy(
                        ) if to_array else input_data[col]
                    extracted_data.append(data_to_add)
            return None

        if isinstance(input_data, np.ndarray):
            if input_data.ndim > 1 and allow_split:
                input_data = np.hsplit(input_data, input_data.shape[1])
                for arr in input_data:
                    extracted_data.append(arr.squeeze())
                return None
            elif input_data.ndim > 1 and enforce_extraction and on_error == 'raise':
                raise ValueError("Multidimensional array found while "
                                 "`enforce_extraction` is True.")
            return input_data if to_array else np.squeeze(input_data)

        return input_data.to_numpy() if to_array and isinstance(
            input_data, pd.Series) else input_data

    for arg in args:
        result = _process_input(arg, columns, to_array)
        if result is not None:
            extracted_data.append(result)

    if ensure_uniform_length and not all(len(x) == len(
            extracted_data[0]) for x in extracted_data):
        if on_error == 'raise':
            raise ValueError("Extracted data arrays do not have uniform length.")
        else:
            return []

    return extracted_data

def to_series_if(
    *values: Any, 
    value_names: Optional[List[str]] = None, 
    name: Optional[str] = None,
    error: str = 'ignore',
    **kws
) -> Series:
    """
    Constructs a pandas Series from given values, optionally naming the series
    and its index.

    Parameters
    ----------
    *values : Any
        A variable number of inputs, each can be a scalar, float, int, or array-like object.
    value_names : Optional[List[str]]
        Names to be used for the index of the series. If not provided or if its length
        doesn't match the number of values, default numeric index is used.
    name : Optional[str]
        Name of the series.
    error : str, default 'ignore'
        Error handling strategy ('ignore' or 'raise'). If 'raise', errors during series
        construction lead to an exception.
    **kws : dict
        Additional keyword arguments passed to `pd.Series` constructor.

    Returns
    -------
    pd.Series or original values
        A pandas Series constructed from the inputs if successful, otherwise, the original
        values if the series construction is not applicable.

    Examples
    --------
    >>> from gofast.tools.coreutils import to_series_if
    >>> series = to_series_if(0.5, 8, np.array(
        [6.3]), [5], 2, value_names=['a', 'b', 'c', 'd', 'e'])
    >>> print(series)
    a    0.5
    b    8.0
    c    6.3
    d    5.0
    e    2.0
    dtype: float64
    >>> series = to_series_if(0.5, 8, np.array([6.3, 7]), [5], 2,
                              value_names=['a', 'b', 'c', 'd', 'e'], error='raise')
    ValueError: Failed to construct series, input types vary.
    """
    # Validate input lengths and types
    if value_names and len(value_names) != len(values):
        if error == 'raise':
            raise ValueError("Length of `value_names` does not match the number of values.")
        value_names = None  # Reset to default indexing
    # Attempt to construct series
    try:
        # Flatten array-like inputs to avoid creating Series of lists/arrays
        flattened_values = [val[0] if isinstance(
            val, (list,tuple,  np.ndarray, pd.Series)) and len(val) == 1 else val for val in values]
        series = pd.Series(flattened_values, index=value_names, name=name, **kws)
    except Exception as e:
        if error == 'raise':
            raise ValueError(f"Failed to construct series due to: {e}")
        return values  # Return the original values if series construction fails

    return series

def ensure_visualization_compatibility(
        result, as_frame=False, view=False, func_name=None,
        verbose=0, allow_singleton_view=False
        ):
    """
    Evaluates and prepares the result for visualization, adjusting its format
    if necessary and determining whether visualization is feasible based on
    given parameters. If the conditions for visualization are not met, 
    especially for singleton values, it can modify the view flag accordingly.

    Parameters
    ----------
    result : iterable or any
        The result to be checked and potentially modified for visualization.
    as_frame : bool, optional
        If True, the result is intended for frame-based visualization, which 
        may prevent conversion of singleton iterables to a float. Defaults to False.
    view : bool, optional
        Flag indicating whether visualization is intended. This function may 
        modify it to False if visualization conditions aren't met. Defaults to False.
    func_name : callable or str, optional
        The name of the function or a callable from which the name can be derived, 
        used in generating verbose messages. Defaults to None.
    verbose : int, optional
        Controls verbosity level. A value greater than 0 enables verbose messages. 
        Defaults to 0.
    allow_singleton_view : bool, optional
        Allows visualization of singleton values if set to True. If False and a 
        singleton value is encountered, `view` is set to False. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the potentially modified result and the updated view flag.
        The result is modified if it's a singleton iterable and conditions require it.
        The view flag is updated based on the allowability of visualization.

    Examples
    --------
    >>> from gofast.tools.coreutils import ensure_visualization_compatibility
    >>> result = [100.0]
    >>> modified_result, can_view = ensure_visualization_compatibility(
    ...     result, as_frame=False, view=True, verbose=1, allow_singleton_view=False)
    Visualization is not allowed for singleton value.
    >>> print(modified_result, can_view)
    100.0 False

    >>> result = [[100.0]]
    >>> modified_result, can_view = ensure_visualization_compatibility(
    ...     result, as_frame=True, verbose=1)
    >>> print(modified_result, can_view)
    [[100.0]] True
    """
    if hasattr(result, '__iter__') and len(
            result) == 1 and not allow_singleton_view:
        if not as_frame:
            # Attempt to convert to float value
            try:
                result = float(result[0])
            except ValueError:
                pass  # Keep the result as is if conversion fails

        if view: 
            if verbose > 0:
                # Construct a user-friendly verbose message
                func_name_str = f"{func_name.__name__} visualization" if callable(
                    func_name) else "Visualization"
                # Ensure the first letter is capitalized
                message_start = func_name_str[0].upper() + func_name_str[1:]  
                print(f"{message_start} is not allowed for singleton value.")
            view =False 
    return result, view 

def generate_mpl_styles(n, prop='color'):
    """
    Generates a list of matplotlib property items (colors, markers, or line styles)
    to accommodate a specified number of samples.

    Parameters
    ----------
    n : int
        Number of property items needed. It generates a list of property items.
    prop : str, optional
        Name of the property to retrieve. Accepts 'color', 'marker', or 'line'.
        Defaults to 'color'.

    Returns
    -------
    list
        A list of property items with size equal to `n`.

    Raises
    ------
    ValueError
        If the `prop` argument is not one of the accepted property names.

    Examples
    --------
    Generate 10 color properties:

    >>> from gofast.tools.coreutils import generate_mpl_styles
    >>> generate_mpl_styles(10, prop='color')
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan', 'magenta']

    Generate 5 marker properties:

    >>> generate_mpl_styles(5, prop='marker')
    ['o', '^', 's', '*', '+']

    Generate 3 line style properties:

    >>> generate_mpl_styles(3, prop='line')
    ['-', '--', '-.']
    """
    import matplotlib as mpl

    D_COLORS = ["g", "gray", "y", "blue", "orange", "purple", "lime",
                "k", "cyan", "magenta"]
    D_MARKERS = ["o", "^", "s", "*", "+", "x", "D", "H"]
    D_STYLES = ["-", "--", "-.", ":"]
    
    n = int(n)  # Ensure n is an integer
    prop = prop.lower().strip().replace('s', '')  # Normalize the prop string
    if prop not in ('color', 'marker', 'line'):
        raise ValueError(f"Property '{prop}' is not available."
                         " Expect 'color', 'marker', or 'line'.")

    # Mapping property types to their corresponding lists
    properties_map = {
        'color': D_COLORS,
        'marker': D_MARKERS + list(mpl.lines.Line2D.markers.keys()),
        'line': D_STYLES
    }

    # Retrieve the specific list of properties based on the prop parameter
    properties_list = properties_map[prop]

    # Generate the required number of properties, repeating the list if necessary
    repeated_properties = list(itertools.chain(*itertools.repeat(properties_list, (
        n + len(properties_list) - 1) // len(properties_list))))[:n]

    return repeated_properties

def generate_alpha_values(n, increase=True, start=0.1, end=1.0, epsilon=1e-10):
    """
    Generates a list of alpha (transparency) values that either increase or 
    decrease gradually to fit the number of property items.
    
    Incorporates an epsilon to safeguard against division by zero.
    
    Parameters
    ----------
    n : int
        The number of alpha values to generate.
    increase : bool, optional
        If True, the alpha values will increase; if False, they will decrease.
        Defaults to True.
    start : float, optional
        The starting alpha value. Defaults to 0.1.
    end : float, optional
        The ending alpha value. Defaults to 1.0.
    epsilon : float, optional
        Small value to avert division by zero. Defaults to 1e-10.
        
    Returns
    -------
    list
        A list of alpha values of length `n`.
    
    Examples
    --------
    >>> from gofast.tools.coreutils import generate_alpha_values
    >>> generate_alpha_values(5, increase=True)
    [0.1, 0.325, 0.55, 0.775, 1.0]
    
    >>> generate_alpha_values(5, increase=False)
    [1.0, 0.775, 0.55, 0.325, 0.1]
    """
    if not 0 <= start <= 1 or not 0 <= end <= 1:
        raise ValueError("Alpha values must be between 0 and 1.")

    # Calculate the alpha values, utilizing epsilon in the denominator 
    # to prevent division by zero
    alphas = [start + (end - start) * i / max(n - 1, epsilon) for i in range(n)]
    
    if not increase:
        alphas.reverse() # or alphas[::-1] creates new list
    
    return alphas

def decompose_colormap(cmap_name, n_colors=5):
    """
    Decomposes a colormap into a list of individual colors.

    Parameters
    ----------
    cmap_name : str
        The name of the colormap to decompose.
    n_colors : int, default=5
        The number of colors to extract from the colormap.

    Returns
    -------
    list
        A list of RGBA color values from the colormap.

    Examples
    --------
    >>> colors = decompose_colormap('viridis', 5)
    >>> print(colors)
    [(0.267004, 0.004874, 0.329415, 1.0), ..., (0.993248, 0.906157, 0.143936, 1.0)]
    """
    cmap = plt.cm.get_cmap(cmap_name, n_colors)
    colors = [cmap(i) for i in range(cmap.N)]
    return colors

def get_colors_and_alphas(
    count, 
    cmap=None, 
    alpha_direction='decrease', 
    start_alpha=0.1,
    end_alpha=1.0, 
    convert_to_named_color=True, 
    single_color_as_string=False,
    consider_alpha=False, 
    ignore_color_names=False, 
    color_space='rgb', 
    error="ignore"
):
    """
    Generates a sequence of color codes and alpha (transparency) values. 
    
    Colors can be sourced from a specified Matplotlib colormap or generated 
    using predefined styles. Alpha values can be arranged in ascending or 
    descending order to create a gradient effect.

    The function also supports converting color tuples to named colors and 
    allows for customizing the transparency gradient. Additionally, if only 
    one color is generated, it can return that color directly as a string
    rather than wrapped in a list, for convenience in functions that expect a
    single color string.

    Parameters
    ----------
    count : int or iterable
        Specifies the number of colors and alpha values to generate. If an iterable 
        is provided, its length determines the number of colors and alphas.
    cmap : str, optional
        The name of a Matplotlib colormap to generate colors. If None, colors are
        generated using predefined styles. Defaults to ``None``.
    alpha_direction : str, optional
        Direction to arrange alpha values for creating a gradient effect. ``increase``
        for ascending order, ``decrease`` for descending. Defaults to ``decrease``.
    start_alpha : float, optional
        The starting alpha value (transparency) in the gradient, between 0 (fully
        transparent) and 1 (fully opaque). Defaults to ``0.1``.
    end_alpha : float, optional
        The ending alpha value in the gradient, between 0 and 1. 
        Defaults to ``1.0``.
    convert_to_named_color : bool, optional
        Converts color tuples to the nearest Matplotlib named color. This 
        conversion applies when exactly one color is generated. 
        Defaults to ``True``.
    single_color_as_string : bool, optional
        If True and only one color is generated, returns the color as a string 
        instead of a list. Useful for functions expecting a single color string.
        Defaults to ``False``.
    consider_alpha : bool, optional
        Includes the alpha channel in the conversion process to named colors.
        Applicable only when `convert_to_named_color` is True. This is helpful
        when a human-readable color name is preferred over RGB values.
        Defaults to ``False``.
    ignore_color_names : bool, optional
        When True, any input color names (str) are ignored during conversion 
        to named colors. Useful to exclude specific colors from conversion. 
        Defaults to ``False``.
    color_space : str, optional
        The color space used for computing the closeness of colors. Can be 
        ``rgb`` for RGB color space or ``lab`` for LAB color space, which is more 
        perceptually uniform. Defaults to ``rgb``.
    error : str, optional
        Controls the error handling strategy when an invalid color is 
        encountered during the conversion process. ``raise`` will throw an error,
        while ``ignore`` will proceed without error. Defaults to ``ignore``.

    Returns
    -------
    tuple
        A tuple containing either a list of color codes (RGBA or named color strings) 
        and a corresponding list of alpha values, or a single color code and alpha 
        value if `single_color_as_string` is True and only one color is generated.

    Examples
    --------
    Generate 3 random colors with decreasing alpha values:

    >>> get_colors_and_alphas(3)
    (['#1f77b4', '#ff7f0e', '#2ca02c'], [1.0, 0.55, 0.1])

    Generate 4 colors from the 'viridis' colormap with increasing alpha values:

    >>> get_colors_and_alphas(4, cmap='viridis', alpha_direction='increase')
    (['#440154', '#3b528b', '#21918c', '#5ec962'], [0.1, 0.4, 0.7, 1.0])

    Convert a single generated color to a named color:

    >>> get_colors_and_alphas(1, convert_to_named_color=True)
    ('rebeccapurple', [1.0])

    Get a single color as a string instead of a list:

    >>> get_colors_and_alphas(1, single_color_as_string=True)
    ('#1f77b4', [1.0])
    """
    
    if hasattr(count, '__iter__'):
        count = len(count)
    colors =[]
    if cmap is not None and cmap not in plt.colormaps(): 
        cmap=None 
        colors =[cmap] # add it to generate map
    # Generate colors
    if cmap is not None:
        colors = decompose_colormap(cmap, n_colors=count)
    else:
        colors += generate_mpl_styles(count, prop='color')

    # Generate alphas
    increase = alpha_direction == 'increase'
    alphas = generate_alpha_values(count, increase=increase,
                                   start=start_alpha, end=end_alpha)
    
    # Convert tuple colors to named colors if applicable
    if convert_to_named_color: 
        colors = colors_to_names(
            *colors, consider_alpha= consider_alpha,
            ignore_color_names=ignore_color_names,  
            color_space= color_space, 
            error= error,
            )
    # If a single color is requested as a string, return it directly
    if single_color_as_string and len(colors) == 1:
        if not convert_to_named_color: 
            colors = [closest_color(colors[0], consider_alpha= consider_alpha, 
                                color_space =color_space )]
        colors = colors[0]

    return colors, alphas


def colors_to_names(*colors, consider_alpha=False, ignore_color_names=False, 
                    color_space='rgb', error='ignore'):
    """
    Converts a sequence of RGB or RGBA colors to their closest named color 
    strings. 
    
    Optionally ignores input color names and handles colors in specified 
    color spaces.
    
    Parameters
    ----------
    *colors : tuple
        A variable number of RGB(A) color tuples or color name strings.
    consider_alpha : bool, optional
        If True, the alpha channel in RGBA colors is considered in the conversion
        process. Defaults to False.
    ignore_color_names : bool, optional
        If True, input strings that are already color names are ignored. 
        Defaults to False.
    color_space : str, optional
        Specifies the color space ('rgb' or 'lab') used for color comparison. 
        Defaults to 'rgb'.
    error : str, optional
        Error handling strategy when encountering invalid colors. If 'raise', 
        errors are raised. Otherwise, errors are ignored. Defaults to 'ignore'.
    
    Returns
    -------
    list
        A list of color name strings corresponding to the input colors.

    Examples
    --------
    >>> from gofast.tools.coreutils import colors_to_names
    >>> colors_to_names((0.267004, 0.004874, 0.329415, 1.0), 
                        (0.127568, 0.566949, 0.550556, 1.0), 
                        consider_alpha=True)
    ['rebeccapurple', 'mediumseagreen']
    
    >>> colors_to_names('rebeccapurple', ignore_color_names=True)
    []
    
    >>> colors_to_names((123, 234, 45), color_space='lab', error='raise')
    ['limegreen']
    """
    color_names = []
    for color in colors:
        if isinstance(color, str):
            if ignore_color_names:
                continue
            else:
                color_names.append(color)  # String color name is found
        else:
            try:
                color_name = closest_color(color, consider_alpha=consider_alpha,
                                           color_space=color_space)
                color_names.append(color_name)
            except Exception as e:
                if error == 'raise':
                    raise e
                
    return color_names

def closest_color(rgb_color, consider_alpha=False, color_space='rgb'):
    """
    Finds the closest named CSS4 color to the given RGB(A) color in the specified
    color space, optionally considering the alpha channel.

    Parameters
    ----------
    rgb_color : tuple
        A tuple representing the RGB(A) color.
    consider_alpha : bool, optional
        Whether to include the alpha channel in the color closeness calculation.
        Defaults to False.
    color_space : str, optional
        The color space to use when computing color closeness. Can be 'rgb' or 'lab'.
        Defaults to 'rgb'.

    Returns
    -------
    str
        The name of the closest CSS4 color.

    Raises
    ------
    ValueError
        If an invalid color space is specified.

    Examples
    --------
    Find the closest named color to a given RGB color:

    >>> from gofast.tools.coreutils import closest_color
    >>> closest_color((123, 234, 45))
    'forestgreen'

    Find the closest named color to a given RGBA color, considering the alpha:

    >>> closest_color((123, 234, 45, 0.5), consider_alpha=True)
    'forestgreen'

    Find the closest named color in LAB color space (more perceptually uniform):

    >>> closest_color((123, 234, 45), color_space='lab')
    'limegreen'
    """
    if color_space not in ['rgb', 'lab']:
        raise ValueError(f"Invalid color space '{color_space}'. Choose 'rgb' or 'lab'.")

    if ensure_scipy_compatibility(): 
        from scipy.spatial import distance 
    # Adjust input color based on consider_alpha flag
    
    # Include alpha channel if consider_alpha is True
    input_color = rgb_color[:3 + consider_alpha]  

    # Convert the color to the chosen color space if needed
    if color_space == 'lab':
        # LAB conversion ignores alpha
        input_color = mcolors.rgb_to_lab(input_color[:3])  
        color_comparator = lambda color: distance.euclidean(
            mcolors.rgb_to_lab(color[:3]), input_color)
    else:  # RGB or RGBA
        color_comparator = lambda color: distance.euclidean(
            color[:len(input_color)], input_color)

    # Compute the closeness of each named color to the given color
    closest_name = None
    min_dist = float('inf')
    for name, hex_color in mcolors.CSS4_COLORS.items():
        # Adjust based on input_color length
        named_color = mcolors.to_rgba(hex_color)[:len(input_color)]  
        dist = color_comparator(named_color)
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name

def check_uniform_type(
    values: Union[Iterable[Any], Any],
    items_to_compare: Union[Iterable[Any], Any] = None,
    raise_exception: bool = True,
    convert_values: bool = False,
    return_types: bool = False,
    target_type: type = None,
    allow_mismatch: bool = True,
    infer_types: bool = False,
    comparison_method: str = 'intersection',
    custom_conversion_func: _F[Any] = None,
    return_func: bool = False
) -> Union[bool, List[type], Tuple[Iterable[Any], List[type]], _F]:
    """
    Checks whether elements in `values` are of uniform type. 
    
    Optionally comparing them against another set of items or converting all 
    values to a target type. Can return a callable for deferred execution of 
    the specified logic.Function is useful for validating data uniformity, 
    especially before performing operations that assume homogeneity of the 
    input types.
    

    Parameters
    ----------
    values : Iterable[Any] or Any
        An iterable containing items to check. If a non-iterable item is provided,
        it is treated as a single-element iterable.
    items_to_compare : Iterable[Any] or Any, optional
        An iterable of items to compare against `values`. If specified, the
        `comparison_method` is used to perform the comparison.
    raise_exception : bool, default True
        If True, raises an exception when a uniform type is not found or other
        constraints are not met. Otherwise, issues a warning.
    convert_values : bool, default False
        If True, tries to convert all `values` to `target_type`. Requires
        `target_type` to be specified.
    return_types : bool, default False
        If True, returns the types of the items in `values`.
    target_type : type, optional
        The target type to which `values` should be converted if `convert_values`
        is True.
    allow_mismatch : bool, default True
        If False, requires all values to be of identical types; otherwise,
        allows type mismatch.
    infer_types : bool, default False
        If True and different types are found, returns the types of each item
        in `values` in order.
    comparison_method : str, default 'intersection'
        The method used to compare `values` against `items_to_compare`. Must
        be one of the set comparison methods ('difference', 'intersection', etc.).
    custom_conversion_func : Callable[[Any], Any], optional
        A custom function for converting items in `values` to another type.
    return_func : bool, default False
        If True, returns a callable that encapsulates the logic based on the 
        other parameters.

    Returns
    -------
    Union[bool, List[type], Tuple[Iterable[Any], List[type]], Callable]
        The result based on the specified parameters. This can be: 
        - A boolean indicating whether all values are of the same type.
        - The common type of all values if `return_types` is True.
        - A tuple containing the converted values and their types if `convert_values`
          and `return_types` are both True.
        - a callable encapsulating the specified logic for deferred execution.
        
    Examples
    --------
    >>> from gofast.tools.coreutils import check_uniform_type
    >>> check_uniform_type([1, 2, 3])
    True

    >>> check_uniform_type([1, '2', 3], allow_mismatch=False, raise_exception=False)
    False

    >>> deferred_check = check_uniform_type([1, 2, '3'], convert_values=True, 
    ...                                        target_type=int, return_func=True)
    >>> deferred_check()
    [1, 2, 3]

    Notes
    -----
    The function is designed to be flexible, supporting immediate or deferred execution,
    with options for type conversion and detailed type information retrieval.
    """
    def operation():
        # Convert values and items_to_compare to lists if 
        # they're not already iterable
        if isinstance(values, Iterable) and not isinstance(values, str):
            val_list = list(values)
        else:
            val_list = [values]

        if items_to_compare is not None:
            if isinstance(items_to_compare, Iterable) and not isinstance(
                    items_to_compare, str):
                comp_list = list(items_to_compare)
            else:
                comp_list = [items_to_compare]
        else:
            comp_list = []

        # Extract types
        val_types = set(type(v) for v in val_list)
        comp_types = set(type(c) for c in comp_list) if comp_list else set()

        # Compare types
        if comparison_method == 'intersection':
            common_types = val_types.intersection(comp_types) if comp_types else val_types
        elif comparison_method == 'difference':
            common_types = val_types.difference(comp_types)
        else:
            if raise_exception:
                raise ValueError(f"Invalid comparison method: {comparison_method}")
            return False

        # Check for type uniformity
        if not allow_mismatch and len(common_types) > 1:
            if raise_exception:
                raise ValueError("Not all values are the same type.")
            return False

        # Conversion
        if convert_values:
            if not target_type and not custom_conversion_func:
                if raise_exception:
                    raise ValueError("Target type or custom conversion "
                                     "function must be specified for conversion.")
                return False
            try:
                if custom_conversion_func:
                    converted_values = [custom_conversion_func(v) for v in val_list]
                else:
                    converted_values = [target_type(v) for v in val_list]
            except Exception as e:
                if raise_exception:
                    raise ValueError(f"Conversion failed: {e}")
                return False
            if return_types:
                return converted_values, [type(v) for v in converted_values]
            return converted_values

        # Return types
        if return_types:
            if infer_types or len(common_types) > 1:
                return [type(v) for v in val_list]
            return list(common_types)

        return True

    return operation if return_func else operation()


