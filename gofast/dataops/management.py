# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""Deals with data storage, retrieval, and dataset handling."""

from __future__ import annotations, print_function 
import os
import h5py
import shutil 
import pathlib
import warnings 

from six.moves import urllib 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm

from ..api.property import  Config
from ..api.types import Any,  List,  DataFrame, Optional, Dict, Union
from ..api.types import BeautifulSoupTag , Tuple, ArrayLike, Callable
from ..api.util import get_table_size 
from ..decorators import Deprecated, Dataify, EnsureFileExists 
from ..exceptions import FileHandlingError 
from ..tools.baseutils import save_or_load
from ..tools.coreutils import is_iterable, ellipsis2false,smart_format, validate_url 
from ..tools.coreutils import to_numeric_dtypes
from ..tools.funcutils import ensure_pkg
from ..tools.validator import  parameter_validator  


TW = get_table_size() 

__all__=[
    "fetch_remote_data",
    "handle_datasets_with_hdfstore",
    "handle_unique_identifiers",
    "handle_datasets_in_h5", 
    "read_data",
    "request_data",
    "store_or_retrieve_data", 
    "scrape_web_data", 
    ]

@Dataify(prefix="col_")
def handle_unique_identifiers(
    data: DataFrame,
    threshold: float = 0.95, 
    action: str = 'drop', 
    transform_func: Optional[Callable[[any], any]] = None,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 8)
    ) -> DataFrame:
    """
    Examines columns in the DataFrame and handles columns with a high 
    proportion of unique values. These columns can be either dropped or 
    transformed based on specified criteria, facilitating better data 
    analysis and modeling performance by reducing the number of 
    effectively useless features.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to process for unique identifier columns.

    threshold : float, optional
        The proportion threshold above which a column is considered to have 
        too many unique values (default is 0.95). If the proportion of unique 
        values in a column exceeds this threshold, an action is taken based 
        on the `action` parameter.

    action : str, optional
        The action to perform on columns exceeding the unique value threshold:
        - 'drop': Removes the column from the DataFrame.
        - 'transform': Applies a function specified by `transform_func` to 
          the column.
        Default is 'drop'.

    transform_func : Callable[[any], any], optional
        A function to apply to columns where the `action` is 'transform'. This 
        function should take a single value and return a transformed value. 
        If `action` is 'transform' and `transform_func` is None, no 
        transformation is applied.

    view : bool, optional
        If True, visualizes the distribution of unique values before and after 
        modification (default is False).

    cmap : str, optional
        The colormap to use for visualization when `view` is True (default is 
        'viridis').

    fig_size : tuple of int, optional
        The size of the figure to use for visualization when `view` is True 
        (default is (12, 8)).

    Returns
    -------
    pandas.DataFrame
        The DataFrame with columns modified according to the specified action 
        and threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.management import handle_unique_identifiers
    >>> data = pd.DataFrame({
    ...     'ID': range(1000),
    ...     'Age': [25, 30, 35] * 333 + [40],
    ...     'Salary': [50000, 60000, 75000, 90000] * 250
    ... })
    >>> processed_data = handle_unique_identifiers(data, action='drop')
    >>> print(processed_data.columns)
    >>> processed_data = handle_unique_identifiers(data, action='drop', view=True)

    >>> def cap_values(val):
    ...     return min(val, 100)  # Cap values at 100
    >>> processed_data = handle_unique_identifiers(data, action='transform', 
    ...                                            transform_func=cap_values)
    >>> print(processed_data.head())

    Notes
    -----
    Handling columns with a high proportion of unique values is essential in 
    data preprocessing, especially when preparing data for machine learning 
    models. High-cardinality features may lead to overfitting and generally 
    provide little predictive power unless they can be meaningfully 
    transformed.

    See Also
    --------
    pandas.DataFrame.nunique : Count distinct observations over requested axis.
    pandas.DataFrame.apply : Apply a function along an axis of the DataFrame.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
    action = parameter_validator('action', ["drop", "transform"])(action)
    if view:
        # Capture the initial state for visualization before any modification
        unique_counts_before = {col: data[col].nunique() for col in data.columns}

    # Handle the action for high unique value proportions
    columns_to_drop = []
    # Iterate over columns in the DataFrame
    for column in data.columns:
        # Calculate the proportion of unique values
        unique_proportion = data[column].nunique() / len(data)
        # If the proportion of unique values is above the threshold
        if unique_proportion > threshold:
            if action == 'drop':
                columns_to_drop.append(column)
                # Drop the column from the DataFrame
            elif action == 'transform' and transform_func is not None:
                # Apply the transformation function if provided
                data[column] = data[column].apply(transform_func)
                
    # Drop columns if necessary
    if action == 'drop':
        data = data.drop(columns=columns_to_drop, axis=1)
    
    if view:
        # Capture the state after modification
        unique_counts_after = {col: data[col].nunique() for col in data.columns}
        # Visualize the distribution of unique values before and after modification
        _visualize_unique_changes(unique_counts_before, unique_counts_after, fig_size)
        
    # Return the modified DataFrame
    return data

def _visualize_unique_changes(unique_counts_before, unique_counts_after, 
                              fig_size: Tuple[int, int]):
    """
    Visualize changes in unique value counts before and after modification.
    """
    # Create 1 row, 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)  
    
    # Before modification
    ax1.bar(unique_counts_before.keys(), unique_counts_before.values(), color='skyblue')
    ax1.set_title('Unique Values Before Modification')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Unique Values')
    ax1.tick_params(axis='x', rotation=45)

    # After modification
    ax2.bar(unique_counts_after.keys(), unique_counts_after.values(), color='lightgreen')
    ax2.set_title('Unique Values After Modification')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Unique Values')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    
@EnsureFileExists(action ='ignore')
def read_data(
    f: str | pathlib.PurePath, 
    sanitize: bool = ..., 
    reset_index: bool = ..., 
    comments: str = "#", 
    delimiter: str = None, 
    columns: List[str] = None,
    npz_objkey: str = None, 
    verbose: bool = ..., 
    **read_kws
) -> DataFrame:
    """
    Read all specific files and URLs allowed by the package.

    Readable files are systematically converted to a DataFrame.

    Parameters
    ----------
    f : str, Path-like object
        File path or Pathlib object. Must contain a valid file name and 
        should be a readable file or URL.

    sanitize : bool, default=False
        Push a minimum sanitization of the data such as:
        - Replace non-alphabetic column items with a pattern '_'
        - Cast data values to numeric if applicable
        - Drop full NaN columns and rows in the data

    reset_index : bool, default=False
        Reset index if full NaN columns are dropped after sanitization. 
        Apply minimum data sanitization after reading data.

    comments : str or sequence of str or None, default='#'
        The characters or list of characters used to indicate the start 
        of a comment. None implies no comments. For backwards compatibility, 
        byte strings will be decoded as 'latin1'.

    delimiter : str, optional
        The character used to separate the values. For backwards 
        compatibility, byte strings will be decoded as 'latin1'. The default 
        is whitespace.

    columns : list of str, optional
        List of column names to use. If the file has a header row, then 
        you should explicitly pass ``header=0`` to override the column 
        names.

    npz_objkey : str, optional
        Dataset key to identify array in multiple array storages in '.npz' 
        format. If key is not set during 'npz' storage, ``arr_0`` should 
        be used. Capable of reading text and numpy formats ('.npy' and 
        '.npz') data. Note that when data is stored in compressed ".npz" 
        format, provide the '.npz' object key as an argument of parameter 
        `npz_objkey`. If None, only the first array should be read and 
        ``npz_objkey='arr_0'``.

    verbose : bool, default=0
        Outputs message for user guide.

    read_kws : dict
        Additional keyword arguments passed to pandas readable file keywords.

    Returns
    -------
    DataFrame
        A dataframe with head contents by default.

    Notes
    -----
    This function reads various file formats and converts them into a 
    pandas DataFrame. It supports sanitization of the data which includes 
    replacing non-alphabetic column names, casting data to numeric types 
    where applicable, and removing fully NaN columns and rows. The function 
    also supports reading numpy arrays from '.npy' and '.npz' files.

    Examples
    --------
    >>> from gofast.dataops.management import read_data
    >>> df = read_data('data.csv', sanitize=True, reset_index=True)
    >>> print(df.head())

    See Also
    --------
    np.loadtxt : Load text file.
    np.load : Load uncompressed or compressed numpy `.npy` and `.npz` formats.
    gofast.dataops.management.save_or_load : Save or load numpy arrays.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """

    def min_sanitizer ( d, /):
        """ Apply a minimum sanitization to the data `d`."""
        return to_numeric_dtypes(
            d, sanitize_columns= True, 
            drop_nan_columns= True, 
            reset_index=reset_index, 
            verbose = verbose , 
            fill_pattern='_', 
            drop_index = True
            )
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
    
    sanitize, reset_index, verbose = ellipsis2false (
        sanitize, reset_index, verbose )
    if ( isinstance ( f, str ) 
            and str(os.path.splitext(f)[1]).lower()in (
                '.txt', '.npy', '.npz')
            ): 
        f = save_or_load(f, task = 'load', comments=comments, 
                         delimiter=delimiter )
        # if extension is .npz
        if isinstance(f, np.lib.npyio.NpzFile):
            npz_objkey = npz_objkey or "arr_0"
            f = f[npz_objkey] 

        if columns is not None: 
            columns = is_iterable(columns, exclude_string= True, 
                                  transform =True, parse_string =True 
                                  )
            if len( columns )!= f.shape [1]: 
                warnings.warn(f"Columns expect {f.shape[1]} attributes."
                              f" Got {len(columns)}")
            
        f = pd.DataFrame(f, columns=columns )
        
    if isinstance (f, pd.DataFrame): 
        if sanitize: 
            f = min_sanitizer (f)
        return  f 
    
    cpObj= Config().parsers 
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smart_format(cpObj.keys(), 'or')} files"
                        )
    try : 
        f = cpObj[ex](f, **read_kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except BaseException as e : 
        raise FileHandlingError (
            f"Cannot parse the file : {os.path.basename (f)!r}. "+  str(e))
    if sanitize: 
        f = min_sanitizer (f)
        
    return f 

@EnsureFileExists
@ensure_pkg("requests")
def request_data(
    url: str, 
    method: str = 'get',
    data: Optional[Any] = None, 
    as_json: bool = ..., 
    as_text: bool = ..., 
    stream: bool = ..., 
    raise_status: bool = ..., 
    save_to_file: bool = ..., 
    filename: Optional[str] = None, 
    show_progress: bool = ...,
    **kwargs
) -> Union[str, dict, ...]:
    """
    Perform an HTTP request to a specified URL and process the response, with 
    optional progress bar visualization.

    Parameters
    ----------
    url : str
        The URL to which the HTTP request is sent.
    method : str, optional
        The HTTP method to use for the request. Supported values are 'get' 
        and 'post'. Default is 'get'.
    data : Any, optional
        The data to send in the body of the request, used with 'post' method.
    as_json : bool, optional
        If True, parses the response as JSON. Default is False.
    as_text : bool, optional
        If True, returns the response as a string. Default is False.
    stream : bool, optional
        If True, streams the response. Useful for large file downloads.
        Default is False.
    raise_status : bool, optional
        If True, raises an HTTPError for bad HTTP responses. 
        Default is False.
    save_to_file : bool, optional
        If True, saves the response content to a file. Default is False.
    filename : str, optional
        File path for saving response content. Required if 
        `save_to_file` is True.
    show_progress : bool, optional
        If True, displays a progress bar during file download. 
        Default is False.
    **kwargs
        Additional keyword arguments passed to the requests method
        (e.g., headers, cookies).

    Returns
    -------
    Union[str, dict, requests.Response]
        The server's response. Depending on the flags, this can be a string,
        a dictionary, or a raw Response object.

    Raises
    ------
    ValueError
        If `save_to_file` is True but no `filename` is provided.
        If an invalid HTTP method is specified.

    Examples
    --------
    >>> from gofast.dataops.management import request_data
    >>> response = request_data('https://api.github.com/user',
                                auth=('user', 'pass'), as_json=True)
    >>> print(response)
    """

    import requests 
    
    (as_text, as_json, stream, raise_status, save_to_file,
     show_progress) = ellipsis2false(
        as_text, as_json,  stream, raise_status , save_to_file,
        show_progress)
    
    if save_to_file and not filename:
        raise ValueError("A filename must be provided when "
                         "'save_to_file' is True.")

    request_method = getattr(requests, method.lower(), None)
    if not request_method:
        raise ValueError(f"Invalid HTTP method: {method}")

    response = request_method(url, data=data, stream=stream, **kwargs)

    if save_to_file:
        with open(filename, 'wb') as file:
            if show_progress:
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='iB',ascii=True,
                                    unit_scale=True, ncols=97)
            for chunk in response.iter_content(chunk_size=1024):
                if show_progress:
                    progress_bar.update(len(chunk))
                file.write(chunk)
            if show_progress:
                progress_bar.close()

    if raise_status:
        response.raise_for_status()

    return response.text if as_text else ( 
        response.json () if as_json else response )


@EnsureFileExists
@Deprecated("Deprecated function. Should be removed next release. "
            "Use `gofast.tools.fetch_remote_data` instead.")
def get_remote_data(
    remote_file: str, 
    save_path: Optional[str] = None, 
    raise_exception: bool = True
) -> bool:
    """
    Retrieve data from a remote location and optionally save it to a 
    specified path.

    Parameters
    ----------
    remote_file : str
        The full path URL to the remote file to be downloaded.
    
    save_path : str, optional
        The local file system path where the downloaded file should be saved.
        If None, the file is saved in the current directory. Default is None.
    
    raise_exception : bool, default=True
        If True, raises a ConnectionRefusedError when the connection fails.
        Otherwise, prints the error message.

    Returns
    -------
    bool
        True if the file was successfully downloaded; False otherwise.

    Raises
    ------
    ConnectionRefusedError
        If the connection fails and `raise_exception` is True.

    Examples
    --------
    >>> from gofast.dataops.management import get_remote_data
    >>> status = get_remote_data('https://example.com/file.csv', save_path='/local/path')
    >>> print(status)
    
    Notes
    -----
    This function attempts to download a file from a given URL. If the 
    download is successful, the file can optionally be saved to a specified 
    local path. If the `raise_exception` parameter is set to True, a 
    `ConnectionRefusedError` is raised if the connection fails after three 
    attempts. Otherwise, the error message is printed.

    See Also
    --------
    gofast.tools.fetch_remote_data : Recommended replacement for this function.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
    connect_reason = (
        "ConnectionRefusedError: Failed to connect to the remote server. "
        "Possible reasons include:\n"
        "1. The server is not running, thus not listening to the port.\n"
        "2. The server is running, but the port is blocked by a firewall.\n"
        "3. A security program on the PC is blocking several ports."
    )
    validate_url(remote_file)
    print(f"---> Fetching {remote_file!r}...")

    try:
        # Setting up the progress bar
        with tqdm(total=3, ascii=True, desc=f'Fetching {os.path.basename(remote_file)}', 
                  ncols=97) as pbar:
            _ , rfile = os.path.dirname(remote_file), os.path.basename(remote_file)
            status = False

            for k in range(3):
                try:
                    response = urllib.request.urlopen(remote_file)
                    data = response.read() # a `bytes` object

                    # Save the data to file
                    with open(rfile, 'wb') as out_file:
                        out_file.write(data)
                    status = True
                    break
                except TimeoutError:
                    if k == 2:
                        print("---> Connection timed out.")
                except Exception as e:
                    print(f"---> An error occurred: {e}")
                finally:
                    pbar.update(1)

            if status:
                # Move the file to the specified save_path
                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    shutil.move(os.path.realpath(rfile),
                                os.path.join(save_path, rfile))
            else:
                print(f"\n---> Failed to download {remote_file!r}.")
                if raise_exception:
                    raise ConnectionRefusedError(connect_reason)

            return status

    except Exception as e:
        print(f"An error occurred during the download: {e}")
        if raise_exception:
            raise e
        return False

@EnsureFileExists
def handle_datasets_with_hdfstore(
    file_path: str, 
    datasets: Optional[Dict[str, DataFrame]] = None, 
    operation: str = 'store'
) -> Union[None, Dict[str, DataFrame]]:
    """
    Handles storing or retrieving multiple Pandas DataFrames in an HDF5 
    file using `pd.HDFStore`.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file where datasets will be stored or from which 
        datasets will be retrieved.
    
    datasets : dict, optional
        A dictionary where keys are dataset names and values are the datasets
        (Pandas DataFrames). This parameter is required if `operation` is 
        'store'. Default is None.
    
    operation : str
        The operation to perform. Must be one of:
        - 'store': Store datasets in the HDF5 file.
        - 'retrieve': Retrieve datasets from the HDF5 file.

    Returns
    -------
    dict or None
        - If `operation` is 'retrieve', returns a dictionary where keys are 
          dataset names and values are the datasets (Pandas DataFrames).
        - If `operation` is 'store', returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified or if `datasets` is None when 
        `operation` is 'store'.
    
    OSError
        If the file cannot be opened or created.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.management import handle_datasets_with_hdfstore
    
    Storing datasets:
    >>> df1 = pd.DataFrame(np.random.rand(100, 10), 
    ...                    columns=[f'col_{i}' for i in range(10)])
    >>> df2 = pd.DataFrame(np.random.randint(0, 100, size=(200, 5)), 
    ...                    columns=['A', 'B', 'C', 'D', 'E'])
    >>> handle_datasets_with_hdfstore(
    ...    'my_datasets.h5', {'df1': df1, 'df2': df2}, operation='store')

    Retrieving datasets:
    >>> datasets = handle_datasets_with_hdfstore(
    ...     'my_datasets.h5', operation='retrieve')
    >>> print(datasets.keys())

    Notes
    -----
    This function utilizes `pd.HDFStore` to manage HDF5 files, allowing for 
    efficient storage and retrieval of large datasets. The HDF5 format is 
    particularly useful for storing heterogeneous data and provides a high 
    level of compression and performance.

    See Also
    --------
    pandas.HDFStore : Store hierarchical datasets in a file format.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
    if operation not in ['store', 'retrieve']:
        raise ValueError("Invalid operation. Please choose 'store' or 'retrieve'.")

    if operation == 'store':
        if datasets is None:
            raise ValueError("Datasets parameter is required for storing data.")

        with pd.HDFStore(file_path, 'w') as store:
            for name, df in datasets.items():
                store.put(name, df)

    elif operation == 'retrieve':
        datasets_retrieved = {}
        with pd.HDFStore(file_path, 'r') as store:
            for name in store.keys():
                datasets_retrieved[name.strip('/')] = store[name]
        return datasets_retrieved


@EnsureFileExists
def store_or_retrieve_data(
    file_path: str,
    datasets: Optional[Dict[str, Union[ArrayLike, DataFrame]]] = None,
    operation: str = 'store'
) -> Optional[Dict[str, Union[ArrayLike, DataFrame]]]:
    """
    Handles storing or retrieving multiple datasets (numpy arrays or Pandas
    DataFrames) in an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file for storing or retrieving datasets.

    datasets : dict, optional
        A dictionary with dataset names as keys and datasets 
        (numpy arrays or Pandas DataFrames) as values. Required if 
        operation is 'store'. Default is None.
        
    operation : str
        The operation to perform - 'store' for storing datasets, 'retrieve' 
        for retrieving datasets.

    Returns
    -------
    Optional[Dict[str, Union[np.ndarray, pd.DataFrame]]]
        If operation is 'retrieve', returns a dictionary with dataset names 
        as keys and datasets as values. If operation is 'store', returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified or required parameters are missing.
        
    TypeError
        If provided datasets are not in supported formats 
        (numpy arrays or pandas DataFrames).

    Examples
    --------
    >>> import pandas as pd 
    >>> import numpy as np
    >>> from gofast.dataops.management import store_or_retrieve_data
    
    Storing datasets:
    >>> df1 = pd.DataFrame(np.random.rand(100, 10), columns=[f'col_{i}' for i in range(10)])
    >>> arr1 = np.random.rand(100, 10)
    >>> store_or_retrieve_data('my_datasets.h5', {'df1': df1, 'arr1': arr1}, operation='store')

    Retrieving datasets:
    >>> datasets = store_or_retrieve_data('my_datasets.h5', operation='retrieve')
    >>> print(datasets.keys())

    Notes
    -----
    This function leverages `pd.HDFStore` to manage HDF5 files, enabling 
    efficient storage and retrieval of large datasets. The HDF5 format 
    is highly efficient for storing heterogeneous data and provides 
    excellent compression and performance characteristics.

    See Also
    --------
    pandas.HDFStore : Store hierarchical datasets in a file format.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
    valid_operations = {'store', 'retrieve'}
    if operation not in valid_operations:
        raise ValueError(f"Invalid operation '{operation}'. "
                         f"Choose from {valid_operations}.")

    with pd.HDFStore(file_path, mode='a' if operation == 'store' else 'r') as store:
        if operation == 'store':
            if not datasets:
                raise ValueError("Datasets are required for the 'store' operation.")

            for name, data in datasets.items():
                if not isinstance(data, (pd.DataFrame, np.ndarray)):
                    raise TypeError("Unsupported data type. Only numpy arrays "
                                    "and pandas DataFrames are supported.")
                
                store[name] = pd.DataFrame(data) if isinstance(data, np.ndarray) else data

        elif operation == 'retrieve':
            return {name.replace ("/", ""): store[name] for name in store.keys()}
        
@EnsureFileExists
def base_storage(
    file_path: str,
    datasets: Optional[Dict[str, Union[ArrayLike, DataFrame]]] = None, 
    operation: str = 'store'
) -> Union[None, Dict[str, Union[ArrayLike, DataFrame]]]:
    """
    Handles storing or retrieving multiple datasets (numpy arrays or Pandas 
    DataFrames) in an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file where datasets will be stored or from which 
        datasets will be retrieved.
        
    datasets : dict, optional
        A dictionary where keys are dataset names and values are the 
        datasets (numpy arrays or Pandas DataFrames).
        Required if `operation` is 'store'. 
     
    operation : str
        The operation to perform - 'store' for storing datasets, 'retrieve' 
        for retrieving datasets.

    Returns
    -------
    dict or None
        If `operation` is 'retrieve', returns a dictionary where keys are 
        dataset names and values are the datasets (numpy arrays or Pandas 
        DataFrames). If `operation` is 'store', returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified.
    OSError
        If the file cannot be opened or created.

    Examples
    --------
    Storing datasets:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.dataops.management import base_storage
    >>> data1 = np.random.rand(100, 10)
    >>> df1 = pd.DataFrame(np.random.randint(0, 100, size=(200, 5)),
                           columns=['A', 'B', 'C', 'D', 'E'])
    >>> base_storage('my_datasets.h5', {'dataset1': data1, 'df1': df1},
                     operation='store')

    Retrieving datasets:
    >>> datasets = base_storage('my_datasets.h5', operation='retrieve')
    >>> print(datasets.keys())

    Notes
    -----
    This function leverages HDF5 files for efficient storage and retrieval 
    of large datasets. The HDF5 format supports both numpy arrays and pandas 
    DataFrames, providing excellent performance and compression.

    See Also
    --------
    pandas.HDFStore : Store hierarchical datasets in a file format.
    h5py.File : Interface to HDF5 files.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
    if operation not in ['store', 'retrieve']:
        raise ValueError("Invalid operation. Please choose 'store' or 'retrieve'.")

    if operation == 'store':
        if datasets is None:
            raise ValueError("Datasets parameter is required for storing data.")

        with h5py.File(file_path, 'w') as h5file:
            for name, data in datasets.items():
                if isinstance(data, pd.DataFrame):
                    data.to_hdf(file_path, key=name, mode='a')
                elif isinstance(data, np.ndarray):
                    h5file.create_dataset(name, data=data)
                else:
                    raise TypeError("Unsupported data type. Only numpy arrays "
                                    "and pandas DataFrames are supported.")

    elif operation == 'retrieve':
        datasets_retrieved = {}
        with h5py.File(file_path, 'r') as h5file:
            for name in h5file.keys():
                try:
                    datasets_retrieved[name] = pd.read_hdf(file_path, key=name)
                except (KeyError, TypeError):
                    datasets_retrieved[name] = h5file[name][...]

        return datasets_retrieved

@EnsureFileExists
def fetch_remote_data(
    file_url: str, /,  
    save_path: Optional[str] = None, 
    raise_exception: bool = True
) -> bool:
    """
    Download a file from a remote URL and optionally save it to a specified location.

    This function attempts to download a file from the given URL. If `save_path` is 
    provided, it saves the file to that location; otherwise, it saves it in the 
    current working directory. If the download fails, it can optionally raise an 
    exception or return False.

    Parameters
    ----------
    remote_file_url : str
        The URL of the remote file to be downloaded.
    save_path : str, optional
        The local directory path where the downloaded file should be saved. 
        If None, the file is saved in the current directory. Default is None.
    raise_exception : bool, default True
        If True, raises an exception upon failure. Otherwise, returns False.

    Returns
    -------
    bool
        True if the file was successfully downloaded, False otherwise.

    Raises
    ------
    ConnectionRefusedError
        If the download fails and `raise_exception` is True.

    Examples
    --------
    >>> from gofast.dataops.management import fetch_remote_data
    >>> status = fetch_remote_data('https://example.com/file.csv', save_path='/local/path')
    >>> print(status)

    Notes
    -----
    This function uses `urllib.request` for downloading the file and `tqdm` for
    displaying a progress bar. It handles errors gracefully by either raising an
    exception or printing an error message based on the `raise_exception` parameter.

    See Also
    --------
    urllib.request.urlopen : Open a network object denoted by a URL for reading.
    tqdm : A fast, extensible progress bar for Python.

    References
    ----------
    .. [1] Python Software Foundation. Python Language Reference, version 3.8. Available at 
           https://www.python.org
    .. [2] Urllib documentation. Available at 
           https://docs.python.org/3/library/urllib.request.html
    .. [3] Tqdm documentation. Available at 
           https://tqdm.github.io/
    """

    def handle_download_error(e: Exception, message: str) -> None:
        """
        Handle download errors, either by raising an exception or printing 
        an error message.

        Parameters
        ----------
        e : Exception
            The exception that was raised.
        message : str
            The error message to be printed or included in the raised exception.

        Raises
        ------
        Exception
            The original exception, if `raise_exception` is True.
        """
        print(message)
        if raise_exception:
            raise e

    def move_file_to_save_path(file_name: str) -> None:
        """
        Move the downloaded file to the specified save path.

        Parameters
        ----------
        file_name : str
            The name of the file to be moved.
        """
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            shutil.move(os.path.realpath(file_name), os.path.join(
                save_path, file_name))

    try:
        file_name = os.path.basename(file_url)
        print(f"---> Fetching '{file_url}'...")

        with tqdm(total=3, ascii=True, desc=f'Fetching {file_name}',
                  ncols=97) as progress_bar:
            for attempt in range(3):
                try:
                    response = urllib.request.urlopen(file_url)
                    data = response.read()

                    with open(file_name, 'wb') as file:
                        file.write(data)

                    move_file_to_save_path(file_name)
                    return True

                except TimeoutError:
                    if attempt == 2:
                        handle_download_error(
                            TimeoutError(), "Connection timed out while"
                            f" downloading '{file_url}'.")
                except Exception as e:
                    handle_download_error(
                        e, f"An error occurred while downloading '{file_url}': {e}")
                finally:
                    progress_bar.update(1)

            # If all attempts fail
            return False

    except Exception as e:
        handle_download_error(
            e, f"An unexpected error occurred during the download: {e}")
        return False


@EnsureFileExists
@ensure_pkg("bs4", " Needs `BeautifulSoup` from `bs4` package" )
@ensure_pkg("requests")
def scrape_web_data(
    url: str, element: str, 
    class_name: Optional[str] = None, 
    attributes: Optional[dict] = None, 
    parser: str = 'html.parser'
) -> List[BeautifulSoupTag[str]]:
    """
    Scrape data from a web page using BeautifulSoup.

    This function fetches the content of a web page and uses BeautifulSoup to 
    parse the HTML and find all instances of a specified HTML element. The 
    search can be further refined using class names and additional attributes.

    Parameters
    ----------
    url : str
        The URL of the web page to scrape.
    element : str
        The HTML element to search for.
    class_name : str, optional
        The class attribute of the HTML element to narrow down the search. 
        Default is None.
    attributes : dict, optional
        Additional attributes of the HTML element to narrow down the search. 
        Default is None.
    parser : str, optional
        The parser used by BeautifulSoup. Default is 'html.parser'.

    Returns
    -------
    list of bs4.element.Tag
        A list of BeautifulSoup Tag objects that match the search query.

    Raises
    ------
    requests.exceptions.HTTPError
        If the HTTP request to the URL fails.

    Examples
    --------
    >>> from gofast.dataops.management import scrape_web_data
    >>> url = 'https://example.com'
    >>> element = 'div'
    >>> class_name = 'content'
    >>> data = scrape_web_data(url, element, class_name)
    >>> for item in data:
    ...     print(item.text)

    >>> url = 'https://example.com/articles'
    >>> element = 'h1'
    >>> data = scrape_web_data(url, element)
    >>> for header in data:
    ...    print(header.text)  # prints the text of each <h1> tag

    >>> url = 'https://example.com/products'
    >>> element = 'section'
    >>> attributes = {'id': 'featured-products'}
    >>> data = scrape_web_data(url, element, attributes=attributes)
    >>> # prints the text of each section with id 'featured-products'
    >>> for product in data:
    ...     print(product.text)  

    Notes
    -----
    Web scraping involves fetching and parsing content from web pages. This 
    function simplifies the process by providing an interface to search for 
    specific HTML elements based on their tag, class, and other attributes. 
    Ensure that web scraping is performed in accordance with the website's 
    terms of service and robots.txt file.

    See Also
    --------
    requests.get : Sends a GET request to the specified URL.
    bs4.BeautifulSoup : Parses HTML content and provides methods for 
                        searching the parse tree.

    References
    ----------
    .. [1] BeautifulSoup Documentation. Available at 
           https://www.crummy.com/software/BeautifulSoup/bs4/doc/
    .. [2] Requests Documentation. Available at 
           https://docs.python-requests.org/en/master/
    """
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, parser)
        if class_name:
            elements = soup.find_all(element, class_=class_name)
        elif attributes:
            elements = soup.find_all(element, **attributes)
        else:
            elements = soup.find_all(element)
        return elements
    else:
        response.raise_for_status()

@EnsureFileExists
def handle_datasets_in_h5(
    file_path: str,
    datasets: Optional[Dict[str, ArrayLike]] = None, 
    operation: str = 'store'
) -> Union[None, Dict[str, ArrayLike]]:
    """
    Handles storing or retrieving multiple datasets in an HDF5 file.

    This function facilitates the storage and retrieval of multiple datasets,
    specifically numpy arrays, in an HDF5 file. The operation can either 
    store provided datasets into the specified HDF5 file or retrieve datasets 
    from it.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file where datasets will be stored or from which 
        datasets will be retrieved.
    datasets : dict, optional
        A dictionary where keys are dataset names and values are the 
        datasets (numpy arrays). Required if operation is 'store'.
    operation : str
        The operation to perform - 'store' for storing datasets, 'retrieve' 
        for retrieving datasets.

    Returns
    -------
    dict or None
        If operation is 'retrieve', returns a dictionary where keys are dataset
        names and values are the datasets (numpy arrays). If operation is 'store', 
        returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified.
    OSError
        If the file cannot be opened or created.

    Examples
    --------
    Storing datasets:
    >>> import numpy as np
    >>> from gofast.dataops.management import handle_datasets_in_h5
    >>> data1 = np.random.rand(100, 10)
    >>> data2 = np.random.rand(200, 5)
    >>> handle_datasets_in_h5('my_datasets.h5', 
                              {'dataset1': data1, 'dataset2': data2}, operation='store')

    Retrieving datasets:
    >>> datasets = handle_datasets_in_h5('my_datasets.h5', operation='retrieve')
    >>> print(datasets.keys())

    Notes
    -----
    The HDF5 format is particularly suited for handling large datasets. This 
    function simplifies the process of managing multiple datasets within a single 
    HDF5 file. When storing data, each dataset is stored under its corresponding 
    key. When retrieving data, the function returns a dictionary of datasets where 
    keys are dataset names and values are numpy arrays.

    See Also
    --------
    h5py.File : Open an HDF5 file.
    numpy.ndarray : N-dimensional array object.

    References
    ----------
    .. [1] Collette, A. (2013). Python and HDF5. O'Reilly Media, Inc.
    """
    if operation not in ['store', 'retrieve']:
        raise ValueError("Invalid operation. Please choose 'store' or 'retrieve'.")

    if operation == 'store':
        if datasets is None:
            raise ValueError("Datasets parameter is required for storing data.")

        with h5py.File(file_path, 'w') as h5file:
            for name, data in datasets.items():
                h5file.create_dataset(name, data=data)

    elif operation == 'retrieve':
        datasets_retrieved = {}
        with h5py.File(file_path, 'r') as h5file:
            for name in h5file.keys():
                datasets_retrieved[name] = h5file[name][...]

        return datasets_retrieved

if __name__=="__main__": 
    # Create a sample DataFrame
    data = pd.DataFrame({
        'ID': range(100),  # Unique identifier
        'Age': [20] * 25 + [30] * 25 + [40] * 25 + [50] * 25,
        'Salary': [50000 + x for x in range(100)],
    })

    # Define a simple transformation function to demonstrate the transform option
    def example_transform(x):
        return f"ID_{x}"

    # Handling unique identifiers by dropping them
    result_drop = handle_unique_identifiers(data, threshold=0.9, action='drop')
    print("DataFrame after dropping high-uniqueness columns:\n", result_drop.head())

    # Handling unique identifiers by transforming them
    result_transform = handle_unique_identifiers(
        data, threshold=0.9, action='transform', transform_func=example_transform)
    print("DataFrame after transforming high-uniqueness columns:\n", result_transform.head())

















