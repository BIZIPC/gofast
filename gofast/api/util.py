# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
import os 
import re
import shutil
import warnings
from collections.abc import Iterable
import numpy as np 
import pandas as pd
from .property import GofastConfig 

# Attempting to modify the property will raise an error
GOFAST_ESCAPE= GofastConfig().WHITESPACE_ESCAPE 

def escape_dataframe_elements(
    df, /, 
    escape_columns=True, 
    escape_index=True,
    escape_all=False, 
    escape_char=None, 
    item_to_escape=' '
    ):
    """
    Escapes specific characters in a DataFrame's column names, index, or all
    string values within its cells, depending on the arguments provided.
    This function allows for customization of the escape character and the
    character to be escaped.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose elements are to be escaped.
    escape_columns : bool, optional
        If True, escapes characters in column names unless they are numeric.
        Default is True.
    escape_index : bool, optional
        If True, escapes characters in the DataFrame's index unless they are
        numeric. Default is True.
    escape_all : bool, optional
        If True, applies escaping to all string values within the DataFrame's
        cells. Default is False.
    escape_char : str, optional
        The character to use for escaping. If None, defaults to the Gofast
        package's standard escape character ('π').
    item_to_escape : str, optional
        The character to be replaced by the escape character. Default is ' ' (space).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with escaped elements as specified.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
        'Name': ['Alice Smith', 'Bob O\'Connor', 'Charlie Brown'],
        'Age': [25, 32, 37],
        'Occupation': ['Data Scientist', 'AI Specialist', 'ML Engineer']
    })
    >>> escaped_df = escape_dataframe_elements(df, escape_all=True)
    >>> print(escaped_df)
          Name           Age  Occupation
    0  Aliceπ'Smith       25  Dataπ'Scientist
    1  Bobπ'O\'Connor     32  AIπ'Specialist
    2  Charlieπ'Brown     37  MLπ'Engineer

    Note
    ----
    The function does not modify the original DataFrame but returns a new DataFrame
    with the modifications. This is to ensure that the integrity of the original data
    is maintained.
    """
    escape_char = escape_char or GOFAST_ESCAPE #  which is 'π'
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Copy the DataFrame to avoid modifying the original
    escaped_df = df.copy()

    # Escape column names if they are not numeric and escaping is enabled
    if escape_columns and not is_numeric_type(df, target="columns"):
        escaped_df.columns = [
            str(col).replace(item_to_escape, escape_char) for col in df.columns
        ]

    # Escape index if it is not numeric and escaping is enabled
    if escape_index and not is_numeric_type(df, target="index"):
        escaped_df.index = [
            str(idx).replace(item_to_escape, escape_char) for idx in df.index
        ]

    # Optionally escape all string values within the DataFrame
    if escape_all:
        escaped_df = escaped_df.applymap(
            lambda x: str(x).replace(item_to_escape, escape_char)
            if isinstance(x, str) else x
        )

    return escaped_df

def distribute_column_widths(*dfs,  **kws):
    """
    Distributes column widths among multiple DataFrames to ensure consistency
    in column display widths. This is based on the maximum width required by
    any column's content across all frames, adjusted for each column uniformly
    across the provided DataFrames.

    Parameters
    ----------
    *dfs : unpacked tuple of pd.DataFrame
        Multiple DataFrame objects whose column widths need to be synchronized.
        Each DataFrame should have the same structure (same number of columns)
        for the function to work correctly.
    insert_ellispsis: bool, 
       Insert ellipsis columns for a temporary columns construction. 
       If ``True``, ellipsis is inserted with its value set to 3. 
    Returns
    -------
    tuple
        A tuple containing:
        - index_width (int): The width of the index column.
        - adjusted_widths (dict): A dictionary with column names as keys and
          the adjusted maximum widths as values.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.util import distribute_column_widths
    >>> df1 = pd.DataFrame({
    ...     'A': ['long text here', 'short'], 
    ...     'B': [123456, 123]
    ... })
    >>> df2 = pd.DataFrame({
    ...     'A': ['short', 'tiny'], 
    ...     'B': [654321, 654321]
    ... })
    >>> index_width, adjusted_column_widths = distribute_column_widths(df1, df2)
    >>> adjusted_column_widths
    {'A': 14, 'B': 6}  # Assuming the implementation of get_column_widths_in 
    and related functions correctly calculates widths.

    Notes
    -----
    This function assumes that all input DataFrames should have the same number
    of columns, and it adjusts the widths based on the maximum content width 
    found in any of the DataFrames for each column. This is particularly useful
    when preparing DataFrames for a consistent visual display in reports or 
    across UI elements where column width uniformity is crucial.
    """
    
    index_width, column_widths = get_column_widths_in(
        *dfs, include_index=True, **kws)
    block_columns = _assemble_column_blocks(dfs)
    
    if block_columns is None:
        # Return original widths if column lengths mismatch
        return index_width, column_widths  

    max_widths = _compute_maximum_widths(block_columns, column_widths)
    adjusted_widths = _apply_maximum_widths(max_widths, block_columns)
    if '...' in column_widths.keys(): 
        # then reinsert ellipsis 
        adjusted_widths["..."] = 3 
    return index_width, adjusted_widths

def _assemble_column_blocks(dfs):
    """Compile columns from all dataframes into a structured array for 
    uniform processing."""
    if any(len(df.columns) != len(dfs[0].columns) for df in dfs):
        return None
    return np.array([df.columns.tolist() for df in dfs], dtype=object)

def _compute_maximum_widths(block_columns, column_widths):
    """Calculate the maximum width for each column block across all
    dataframes."""
    if block_columns is None:
        return []
    max_widths = []
    for columns in block_columns.T:  # Transpose to iterate column-wise
        widths = [column_widths.get(col, 0) for col in columns]
        max_widths.append(max(widths))
    return max_widths

def _apply_maximum_widths(max_widths, block_columns):
    """Apply the maximum widths back to the original column names."""
    updated_widths = {}
    for width, columns in zip(max_widths, block_columns.T):
        for col in columns:
            updated_widths[col] = width
    return updated_widths

def check_dataframe_columns(*dfs, error='raise', return_dfs=False):
    """
    Checks whether all provided dataframes have identical columns.

    This function validates that each input is a pandas DataFrame and 
    checks if all DataFrames have the same column names. It handles errors 
    according to the specified error mode and optionally returns the list of 
    valid DataFrames.

    Parameters
    ----------
    *dfs : unpacked tuple of pd.DataFrame
        Variable number of pandas DataFrame objects to be checked for column 
        consistency.
    error : str, optional
        Error handling mode: 'raise' to raise a TypeError when a non-DataFrame
        is passed or 'warn' to issue a warning and exclude the invalid item from
        checks. Default is 'raise'.
    return_dfs : bool, optional
        If True, returns a tuple containing a boolean indicating column 
        consistency and  a list of valid DataFrames. If False, returns only 
        the boolean. Default is False.

    Returns
    -------
    bool or tuple
        If `return_dfs` is False, returns True if all DataFrames have the same 
        columns, otherwise False.
        If `return_dfs` is True, returns a tuple containing the above boolean
        and the list of valid DataFrames.

    Raises
    ------
    TypeError
        If any input is not a DataFrame and `error` is set to 'raise'.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    >>> df3 = pd.DataFrame({'A': [9, 10], 'C': [11, 12]})
    >>> check_dataframe_columns(df1, df2, error='raise')
    True
    >>> check_dataframe_columns(df1, df2, df3, error='warn')
    False
    >>> check_dataframe_columns(df1, df2, "not a dataframe", error='warn',
                                return_dfs=True)
    (True, [df1, df2])
    """
    valid_dfs = []
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            message = f"Expected a DataFrame, but got {type(df).__name__}."
            if error == 'raise':
                raise TypeError(message)
            elif error == 'warn':
                warnings.warn(message)
                continue
        valid_dfs.append(df)

    if not valid_dfs:
        return (False, []) if return_dfs else False

    # Check if all dataframes have the same columns
    first_df_columns = valid_dfs[0].columns
    all_match = all(df.columns.equals(first_df_columns) for df in valid_dfs)
    return (all_match, valid_dfs) if return_dfs else all_match

def find_max_widths(listofdicts):
    """
    Finds the maximum value for each key across a list of dictionaries,
    handling keys that are not present in every dictionary and varying value
    types (iterable or non-iterable).

    Parameters
    ----------
    listofdicts : list of dict
        A list containing dictionaries which may or may not have the same keys.

    Returns
    -------
    dict
        A dictionary with keys from all dictionaries and their maximum values.
        If the value for a key is an iterable, the maximum count of elements 
        in the iterable across all dictionaries is used. If the value is 
        non-iterable, the maximum value is used.

    Examples
    --------
    >>> C = [
    ...     {'location_id': 11, 'location_name': 'Name1', 'dates': [2022, 2023]},
    ...     {'location_id': 12, 'location_name': 'Name2', 'dates': [2021, 2022, 2023, 2024]}
    ... ]
    >>> find_max_widths(C)
    {'location_id': 12, 'location_name': 5, 'dates': 4}
    >>> C = [
    ...     {'location_id': 11, 'location_name': 24, 'date': 19, 
             'total_capacity': 18, 'current_volume': 18, 'rainfall': 20, 
             'evaporation': 21, 'inflow': 19, 'outflow': 20, 'usage': 20, 
             'percentage_full': 18},
    ...     {'location_id': 11, 'location_name': 24, 'date': 19, 
             'total_capacity': 89, 'current_volume': 12, 'rainfall': 89, 
             'evaporation': 22, 'inflow': 7, 'outflow': 11, 'usage': 20, 
             'percentage_full': 16}
    ... ]
    >>> find_max_widths(C)
    {'location_id': 11, 'location_name': 24, 'date': 19, 'total_capacity': 89,
     'current_volume': 18, 'rainfall': 89, 'evaporation': 22, 'inflow': 19, 
     'outflow': 20, 'usage': 20, 'percentage_full': 18}
    """
    max_dict = {}
    for d in listofdicts:
        for key, value in d.items():
            # Determine if the value is iterable
            if isinstance(value, Iterable):
                count = len(value)
            else:
                count = value

            if key in max_dict:
                max_dict[key] = max(max_dict[key], count)
            else:
                max_dict[key] = count

    return max_dict

def _consolidate_column_widths(columns_widths, all_columns_match):
    """Helper function to consolidate column widths."""
    consolidated_widths = {}
    if all_columns_match:
        # consolidated_widths= find_max_widths(columns_widths) # faster 
        for cw in columns_widths:
            for col, width in cw.items():
                consolidated_widths[col] = max(consolidated_widths.get(col, 0), width)
    else:
        # Flatten all widths and get max for each column regardless of dataframe alignment
        for cw in columns_widths:
            for col, width in cw.items():
                consolidated_widths[col] = max(consolidated_widths.get(col, 0), width)
    return consolidated_widths

def _filter_columns_by_list(consolidated_widths, columns, error):
    """Filter consolidated column widths by a specific list of column names."""
    if isinstance(columns, str):
        columns = [columns]
    filtered_widths = {}
    missing_columns = []
    for col in columns:
        if col in consolidated_widths:
            filtered_widths[col] = consolidated_widths[col]
        else:
            missing_columns.append(col)

    if missing_columns:
        message = f"Columns {', '.join(missing_columns)} not found in DataFrames."
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message + " Skipping these columns.")
    return filtered_widths

def get_column_widths_in(
    *dfs, max_text_length=50, 
    include_index=False,
    columns=None, error='raise', 
    return_widths_only=False, 
    insert_ellipsis=False 
    ):
    """
    Calculates the maximum column widths from multiple pandas DataFrames,
    optionally filtering by specified columns and including the maximum index
    width.

    Parameters
    ----------
    *dfs : unpacked tuple of pd.DataFrame
        Variable number of DataFrame objects to analyze. Each DataFrame
        should have a consistent structure if `all_columns_match` is assumed.
    max_text_length : int, optional
        Maximum length of text to consider for each column width. Defaults
        to 50.
    include_index : bool, optional
        Whether to include the maximum width of the DataFrame index in the
        returned values. Defaults to False.
    columns : list of str or str, optional
        Specific columns to include in the width calculation. If None, all
        columns are considered. Defaults to None.
    error : {'raise', 'warn'}, optional
        Error handling strategy when a specified column is not found:
        'raise' to throw a ValueError, 'warn' to issue a warning and exclude
        the column from the results. Defaults to 'raise'.
    return_widths_only : bool, optional
        Determines the format of the returned data. If True, only the widths
        are returned. If False and `include_index` is True, the maximum index
        width is included as the first element in the returned list.
        Defaults to False.
    insert_ellispsis: bool, 
       Insert ellipsis columns for a temporary columns construction. 
       If ``True``, ellipsis is inserted with its value set to 3. 
    Returns
    -------
    list or dict
        If `return_widths_only` is False and `include_index` is True, returns
        a list starting with the maximum index width followed by the widths of
        the specified columns. Otherwise, returns a dictionary of column
        widths or a list of widths, depending on `columns`.

    Raises
    ------
    ValueError
        If `error` is set to 'raise' and any specified column is not found in
        all provided DataFrames.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import get_column_widths_in
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    >>> df2 = pd.DataFrame({'A': [3, 4], 'B': ['w', 'z']})
    >>> get_column_widths_in(df1, df2, max_text_length=10, include_index=True,
    ...                      columns=['A'], error='raise', return_widths_only=True)
    [1, 1]  
    """
    # Check if all dataframes have the same columns
    all_columns_match = check_dataframe_columns(*dfs, error=error)

    # Calculate widths for each DataFrame
    columns_widths = []
    max_index_length = 0
    for df in dfs:
        cw, idx_length = calculate_widths(df, max_text_length=max_text_length)
        columns_widths.append(cw)
        max_index_length = max(max_index_length, idx_length)
    # Consolidate column widths
    consolidated_widths = _consolidate_column_widths(
        columns_widths, all_columns_match)

    # Filter by specified columns, if applicable
    if columns:
        consolidated_widths = _filter_columns_by_list(
            consolidated_widths, columns, error)

    if return_widths_only:
        consolidated_widths = list(consolidated_widths.values())
     
    if insert_ellipsis: 
        # Fake ellipsis columns for dataframe construction
        consolidated_widths["..."]=3 
    # Include index width if requested
    if include_index:
        return ( [max_index_length] + consolidated_widths ) if  isinstance (
            consolidated_widths, list) else (max_index_length, consolidated_widths)

    return consolidated_widths

def get_display_dimensions(
    *dfs, index=True, 
    header=True, 
    max_rows=11, 
    max_cols=7, 
    return_min_dimensions=True
    ):
    """
    Determines the maximum display dimensions for a series of DataFrames by
    considering the minimum number of rows and columns that can be displayed 
    across all provided DataFrames.

    Parameters
    ----------
    *dfs : tuple of pd.DataFrame
        An arbitrary number of pandas DataFrame objects.
    index : bool, optional
        Whether to consider DataFrame indices in calculations, by default True.
    header : bool, optional
        Whether to consider DataFrame headers in calculations, by default True.
    max_rows : int, optional
        The maximum number of rows to potentially display, by default 11.
    max_cols : int, optional
        The maximum number of columns to potentially display, by default 7.
    return_min_dimensions : bool, optional
        If True, returns the minimum of the maximum dimensions (rows, columns) 
        across all  provided DataFrames. If False, returns a tuple of lists 
        containing the maximum dimensions for each DataFrame separately.

    Returns
    -------
    tuple
        If return_min_dimensions is True, returns a tuple (min_rows, min_cols) 
        where min_rows is the minimum of the maximum rows and min_cols is the
        minimum of the maximum columns across all provided DataFrames.
        If return_min_dimensions is False, returns a tuple of two lists 
        (list_of_max_rows, list_of_max_cols), where each list contains the 
        maximum rows and columns for each DataFrame respectively.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.util import get_display_dimensions
    >>> df1 = pd.DataFrame({"A": range(10), "B": range(10, 20)})
    >>> df2 = pd.DataFrame({"C": range(5), "D": range(5, 10)})
    >>> get_display_dimensions(df1, df2, max_rows=5, max_cols=1, 
                               return_min_dimensions=True)
    (5, 1)
    """
    def get_single_df_metrics(df):
        # Use external functions to get recommended dimensions
        auto_rows, auto_cols = auto_adjust_dataframe_display(
            df, index=index, header=header)
        
        # Adjust these dimensions based on input limits
        adjusted_rows = _adjust_value(max_rows, auto_rows)
        adjusted_cols = _adjust_value(max_cols, auto_cols)
        
        return adjusted_rows, adjusted_cols

    # Generate metrics for all DataFrames
    metrics = [get_single_df_metrics(df) for df in dfs]
    df_max_rows, df_max_cols = zip(*metrics)
    
    if return_min_dimensions:
        # Return the minimum of the calculated max 
        # dimensions across all DataFrames
        return min(df_max_rows), min(df_max_cols)
    
    # Return the detailed metrics for each DataFrame 
    # if not returning the minimum across all
    return df_max_rows, df_max_cols

def is_numeric_index(df):
    """
    Checks if the index of a DataFrame is numeric.

    Parameters:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        bool: True if the index is numeric, False otherwise.
    """
    # Check if the index data type is a subtype of numpy number
    return pd.api.types.is_numeric_dtype(df.index.dtype)
    

def is_numeric_type(df, target="index"):
    """
    Checks if the index or columns of a DataFrame are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose index or columns are to be checked.
    target : str, optional
        Specifies whether to check the 'index' or 'columns' for numeric data type.
        Default is 'index'.

    Returns
    -------
    bool
        True if the specified target (index or columns) is numeric, False otherwise.

    Raises
    ------
    ValueError
        If the specified target is not 'index' or 'columns'.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import is_numeric_type
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> print(is_numeric_type(df, 'index'))
    False
    >>> print(is_numeric_type(df, 'columns'))
    False

    >>> df.set_index('A', inplace=True)
    >>> print(is_numeric_type(df, 'index'))
    True
    """
    if target not in ['index', 'columns']:
        raise ValueError("target must be 'index' or 'columns'")

    # Validate if the DataFrame is correctly passed, might 
    # consider adding a specific validation function
    df = validate_data(df )

    # Check if the target attributes are of numeric data type
    data_attribute = df.index if target == 'index' else df.columns
    
    # Handle both regular and MultiIndex cases
    if hasattr(data_attribute, 'levels'):  # MultiIndex case
        return all(pd.api.types.is_numeric_dtype(level.dtype) for level in data_attribute.levels)
    else:  # Single level index or columns
        return pd.api.types.is_numeric_dtype(data_attribute.dtype)


def extract_truncate_df(df, include_truncate=False, max_rows=100, max_cols=7, 
                        return_indices_cols=False):
    """
    Extracts a subset of rows and columns from a dataframe based on its string 
    representation. Optionally includes truncated indices and returns them 
    along with the column names if specified.

    Parameters
    ----------
    df : DataFrame
        The dataframe from which to extract data.
    include_truncate : bool, optional
        Whether to include all indices in a continuous range if truncated, 
        by default False.
    max_rows : int, optional
        Maximum number of rows to display in the string representation of 
        the dataframe, by default 100.
    max_cols : int, optional
        Maximum number of columns to display in the string representation of 
        the dataframe, by default 7.
    return_indices_cols : bool, optional
        If True, returns the indices, column names, and subset dataframe,
        by default False.

    Returns
    -------
    DataFrame or tuple
        If return_indices_cols is False, returns a subset of the dataframe.
        If True, returns a tuple with list of indices, list of column names,
        and the subset dataframe.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import extract_truncate_df
    >>> df = pd.DataFrame({
    ...        "A": range(150), 
    ...        "B": range(150)
    ...    })
    >>> result = extract_truncate_df(df, include_truncate=True, 
    ...                                 max_rows=10, max_cols=1,
    ...                                 return_indices_cols=True
    ...                                 )
    >>> result 
    >>> print(result[0])  # indices
    >>> print(result[1])  # columns ['A']
    >>> print(result[2])  # dataframe subset
           A
    0      0
    1      1
    2      2
    3      3
    4      4
    ..   ...
    145  145
    146  146
    147  147
    148  148
    149  149
    """

    # check indices whether it is numeric, if Numeric keepit otherwise reset it 
    truncated_df = df.copy() 
    name_indexes = [] 
    if not is_numeric_index( truncated_df):
        name_indexes = truncated_df.index.tolist() 
        truncated_df.reset_index (drop=True, inplace =True)
    columns = truncated_df.columns.tolist() 
    
    max_rows, max_cols = find_best_display_params2(df)(
        max_rows=max_rows, max_cols=max_rows)
    
    df_string= truncated_df.to_string(
        index =True, header=True, max_rows=max_rows, max_cols=max_cols )
    # Find all index occurrences at the start of each line
    indices = re.findall(r'^\s*(\d+)', df_string, flags=re.MULTILINE)
    lines = df_string.split('\n')
    
    # Convert found indices to integers
    indices = list(map(int, indices))
    if include_truncate and indices:
        # Assume the indices are continuous and fill in the missing range
        indices = list(range(indices[0], indices[-1] + 1))
    
    if name_indexes:
        indices = [name_indexes [i] for i in indices]
        # reset df.indexes 
        truncated_df.index= name_indexes 
        
    # extract _columns from df_strings columns is expected to be the first 0 
    header_lines = lines[0].strip() 
    # Filter columns based on whether their names exist in header_parts
    columns =extract_matching_columns(header_lines, columns )
    subset_df = get_dataframe_subset(truncated_df, indices, columns )

    if return_indices_cols: 
        return indices, columns, subset_df
    
    return subset_df 

def extract_matching_columns(header_line, data_columns):
    """
    Extracts matching column names from a header line based on the columns 
    present in the dataframe.

    Parameters
    ----------
    header_line : str
        The line of text that includes the names of the columns possibly 
        separated by spaces or special characters.
    data_columns : list of str
        List of actual column names from the dataframe.

    Returns
    -------
    list of str
        A list of column names that match between the header line and the 
        dataframe columns.

    Examples
    --------
    >>> from gofast.api.util import extract_matching_columns
    >>> header_line = "name age city  ... income"
    >>> data_columns = ["name", "age", "city", "state", "income"]
    >>> matching_columns = extract_matching_columns(header_line, data_columns)
    >>> print(matching_columns)
    ['name', 'age', 'city', 'income']
    """
    # Normalize spaces in the header line and split by spaces 
    # considering that header parts are well-separated
    header_parts = header_line.replace('...', ' ').split()
    # Join split parts into complete column names based on data_columns
    normalized_header_parts = []
    temp_part = []
    for part in header_parts:
        temp_part.append(part)
        constructed_name = ' '.join(temp_part)
        if constructed_name in data_columns:
            normalized_header_parts.append(constructed_name)
            temp_part = []

    # Initialize a list to hold columns that match
    matching_columns = []
    # Loop through each column in the data_columns
    for column in data_columns:
        # Check if the column is exactly in the normalized header parts
        if column in normalized_header_parts:
            matching_columns.append(column)
    
    
    return matching_columns

def insert_ellipsis_to_df(sub_df, full_df=None, include_index=True):
    """
    Insert ellipsis into a DataFrame to simulate truncation in display, 
    mimicking the appearance of a large DataFrame that cannot be fully displayed.

    Parameters:
    sub_df : pd.DataFrame
        The DataFrame into which ellipsis will be inserted.
    full_df : pd.DataFrame, optional
        The full DataFrame from which sub_df is derived. If provided, used to 
        determine
        whether ellipsis are needed for rows or columns.
    include_index : bool, default True
        Whether to include ellipsis in the index if rows are truncated.

    Returns:
    pd.DataFrame
        The modified DataFrame with ellipsis inserted if needed.
        
    Example
    ```python    
    from gofast.api.util import insert_ellipsis_to_df
    data = {
        'location_id': [1.0, 1.0, 1.0, 100.0, 100.0],
        'location_name': ['Griffin Grove', 'Griffin Grove', 'Griffin Grove', 
                          'Galactic Gate', 'Galactic Gate'],
        'usage': [107.366256, 204.633431, 188.087255, 208.627374, 133.555798],
        'percentage_full': [87.608753, 42.492289, 78.357623, 25.639027, 89.816833]
    }
    df = pd.DataFrame(data)

    full_data = pd.concat([df]*3)  # Simulate a larger DataFrame
    example_df = insert_ellipsis_to_df(df, full_df=full_data)
    print(example_df)
    ``` 
    
    """
    modified_df = sub_df.copy()
    mid_col_index = len(modified_df.columns) // 2
    mid_row_index = len(modified_df) // 2
    
    # Determine if ellipsis should be inserted for columns
    if full_df is not None and len(sub_df.columns) < len(full_df.columns):
        modified_df.insert(mid_col_index, '...', ["..."] * len(modified_df))
        
    # Determine if ellipsis should be inserted for rows
    if full_df is not None and len(sub_df) < len(full_df) and include_index:
        # Insert a row of ellipsis at the middle index
        ellipsis_row = pd.DataFrame([["..."] * len(modified_df.columns)],
                                    columns=modified_df.columns)
        top_half = modified_df.iloc[:mid_row_index]
        bottom_half = modified_df.iloc[mid_row_index:]
        modified_df = pd.concat([top_half, ellipsis_row, bottom_half], 
                                ignore_index=True)

        # Adjust the index to include ellipsis if required
        new_index = list(sub_df.index[:mid_row_index]) + ['...'] + list(
            sub_df.index[mid_row_index:])
        modified_df.index = new_index

    return modified_df

def get_dataframe_subset(df, indices=None, columns=None):
    """
    Extracts a subset of a DataFrame based on specified indices and columns.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame from which to extract the subset.
        indices (list, optional): List of indices to include in the subset. 
        If None, all indices are included.
        columns (list, optional): List of column names to include in the subset. 
        If None, all columns are included.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only the specified 
        indices and columns.
    """
    # If indices are specified, filter the DataFrame by these indices
    if indices is not None:
        df = df.loc[indices]

    # If columns are specified, filter the DataFrame by these columns
    if columns is not None:
        df = df[columns]

    return df

def flex_df_formatter(
    df, 
    title=None, 
    max_rows='auto', 
    max_cols='auto', 
    float_format='{:.4f}', 
    index=True, 
    header=True, 
    header_line="=", 
    sub_line='-', 
    table_width='auto',
    output_format='string', 
    style="auto", 
    column_widths=None,
    max_index_length=None,
    ):
    """
    Formats and prints a DataFrame with dynamic sizing options, custom number 
    formatting, and structured headers and footers. This function allows for
    detailed customization of the DataFrame's presentation in a terminal or a 
    text-based output format, handling large DataFrames by adjusting column 
    and row visibility according to specified parameters.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to format and display. This is the primary data input 
        for formatting.
    title : str, optional
        Title to display above the DataFrame. If provided, this is centered 
        above the DataFrame
        output.
    max_rows : int, str, optional
        The maximum number of rows to display. If set to 'auto', the number 
        of rows displayed will be determined based on available terminal space
        or internal calculations.
    max_cols : int, str, optional
        The maximum number of columns to display. Similar to max_rows, if set 
        to 'auto', the number of columns is determined dynamically.
    float_format : str, optional
        A format string for floating-point numbers, e.g., '{:.4f}' for 4 
        decimal places.
    index : bool, optional
        Indicates whether the index of the DataFrame should be included in 
        the output.
    header : bool, optional
        Indicates whether the header (column names) should be displayed.
    header_line : str, optional
        A string of characters used to create a separation line below the 
        title and at the bottom of the DataFrame. Typically set to '='.
    sub_line : str, optional
        A string of characters used to create a separation line directly 
        below the header (column names).
    table_width : int, str, optional
        The overall width of the table. If set to 'auto', it will adjust
        based on the content
        and the terminal width.
    output_format : str, optional
        The format of the output. Supports 'string' for plain text output or 
        'html' for HTML formatted output.
    column_widths : [dict of col, int ] or list of int or None, optional
        Specifies the widths for each column within the DataFrame. This list 
        should correspond to the actual columns in the DataFrame and can 
        include the index width at the beginning if `index` is set to True. 
        If `None`, the widths are automatically calculated by the 
        `calculate_column_widths` function based on the content of the
        DataFrame and a specified maximum text length, ensuring optimal 
        formatting for display or printing.
    max_index_length : int or None, optional
        The maximum width allocated for formatting the index of the DataFrame.
        If `None`, the width is automatically calculated based on the contents
        of the index, ensuring that the index is visually consistent and 
        properly aligned with the data columns. This automatic calculation is 
        designed to adapt to the varying lengths of index entries, providing a 
        dynamically adjusted display that optimally uses available space.
        
    Returns
    -------
    str
        The formatted string representation of the DataFrame, including 
        headers, footers,  and any specified customizations.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.api.util import flex_df_formatter
    >>> data = {
    ...    'age': range(30),
    ...    'tenure_months': range(30),
    ...    'monthly_charges': np.random.rand(30) * 100
    ... }
    >>> df = pd.DataFrame(data)
    >>> formatter =flex_df_formatter(
    ...    df, title="Sample DataFrame", max_rows=10, max_cols=3,
    ...    table_width='auto', output_format='string')
    >>> print(formatter)
                Sample DataFrame           
    =======================================
        age  tenure_months  monthly_charges
    ---------------------------------------
    0     0              0          78.6572
    1     1              1          71.3732
    2     2              2          94.3879
    3     3              3          26.6395
    4     4              4          10.3135
    ..  ...            ...              ...
    25   25             25          79.0574
    26   26             26          68.2199
    27   27             27          96.5632
    28   28             28          31.6600
    29   29             29          23.9156
    =======================================
    
    Notes
    -----
    The function dynamically adjusts to terminal size if running in a script 
    executed in a terminal window. This is particularly useful for large 
    datasets where display space is limited and readability is a concern. The 
    optional parameters allow extensive customization to suit different output 
    needs and contexts.
    """
    df = validate_data(df )
    max_rows, max_cols = resolve_auto_settings( max_rows, max_cols )
    
    if max_rows =="auto" or max_cols =="auto": 
        max_rows, max_cols = find_best_display_params2(
            *[df],index =index, header=header)(
                max_rows=max_rows, max_cols=max_cols, )

    # Apply float formatting to the DataFrame
    if output_format == 'html': 
        # Use render for HTML output
        formatted_df = df.style.format(float_format).render()  
    else:
        df= make_format_df(df, GOFAST_ESCAPE, apply_to_column= True)
        # Convert DataFrame to string with the specified format options
        formatted_df = df.to_string(
            index=index, header=header, max_rows=max_rows, max_cols=max_cols, 
            float_format=lambda x: float_format.format(x) if isinstance(
                x, (float, np.float64)) else x
            )
        
    style= select_df_styles(style, df )
    if style =='advanced': 
        formatted_output = df_advanced_style(
            formatted_df, 
            table_width, 
            title= title,
            index= index, 
            header= header, 
            header_line= header_line, 
            sub_line=sub_line, 
            df=df,
            column_widths=column_widths, 
            max_index_length= max_index_length, 
          )
    else: 
        formatted_output=df_base_style(
            formatted_df, title=title, 
            table_width=table_width, 
            header_line= header_line, 
            sub_line= sub_line , 
            max_index_length= max_index_length, 
            column_widths= column_widths, 
            df=df 
            )
    # Remove the whitespace_sub GOFAST_ESCAPE π 
    formatted_output = formatted_output.replace (GOFAST_ESCAPE, ' ')
    return formatted_output

def resolve_auto_settings(*settings):
    """
    Resolves each provided setting to "auto" if it is None or not specified,
    ensuring all settings default to an "auto" mode when undefined.

    Parameters
    ----------
    *settings : tuple
        Variable number of setting values, each can be any type but typically
        intended to be None or a specific setting value.

    Returns
    -------
    list
        A list where each element is the original setting if defined,
        otherwise "auto".

    Examples
    --------
    >>> from gofast.api.util import resolve_auto_settings
    >>> max_rows = None
    >>> max_cols = None
    >>> display_settings = resolve_auto_settings(max_rows, max_cols)
    >>> print(display_settings)
    ['auto', 'auto']

    Notes
    -----
    This function is particularly useful when defaulting display or configuration
    parameters that have not been explicitly set by the user.
    """
    resolved_settings = []
    for setting in settings:
        # Append the setting if not None, otherwise append "auto"
        resolved_settings.append(setting or "auto")

    return resolved_settings

def select_df_styles(style, df, **kwargs):
    """
    Determines the appropriate style for formatting a DataFrame based on the
    given style preference and DataFrame characteristics.

    Parameters
    ----------
    style : str
        The style preference which can be specific names or categories like 'auto',
        'simple', 'basic', etc.
    df : DataFrame
        The DataFrame for which the style is being selected, used especially when
        'auto' style is requested.
    **kwargs : dict
        Additional keyword arguments passed to the style determination functions.

    Returns
    -------
    str
        The resolved style name, either 'advanced' or 'base'.

    Raises
    ------
    ValueError
        If the provided style name is not recognized or not supported.

    Examples
    --------
    >>> from gofast.api.util import select_df_styles
    >>> data = {'Col1': range(150), 'Col2': range(150)}
    >>> df = pd.DataFrame(data)
    >>> select_df_styles('auto', df)
    'advanced'
    >>> select_df_styles('robust', df)
    'advanced'
    >>> select_df_styles('none', df)
    'base'
    >>> select_df_styles('unknown', df)  # Doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Invalid style specified. Choose from: robust, corrtable, 2, ...
    """
    # Mapping styles to categories using regex for flexible matching
    style_map = {
        "advanced": r"(robust|corrtable|2|advanced)",
        "base": r"(simple|basic|1|none|base)"
    }

    # Normalize input style string
    style = str(style).lower()

    # Automatic style determination based on DataFrame size
    if style == "auto":
        style = "advanced" if is_dataframe_long(
            df,  return_rows_cols_size=False, **kwargs) else "base"
    else:
        # Resolve the style based on known categories
        for key, regex in style_map.items():
            if re.match(regex, style):
                style = key
                break
        else:
            # If the style does not match any known category, raise an error
            valid_styles = ", ".join([regex for patterns in style_map.values() 
                                      for regex in patterns.split('|')])
            raise ValueError(f"Invalid style specified. Choose from: {valid_styles}")

    return style

def is_dataframe_long(
        df, max_rows=100, max_cols=7, return_rows_cols_size=False):
    """
    Determines whether a DataFrame is considered 'long' based on the 
    number of rows and columns.

    Parameters:
    ----------
    df : DataFrame
        The DataFrame to evaluate.
    max_rows : int, str, optional
        The maximum number of rows a DataFrame can have to still be considered 
        'short'. If set to 'auto', adjusts based on terminal size or 
        other dynamic measures.
    max_cols : int, str, optional
        The maximum number of columns a DataFrame can have to still be 
        considered 'short'. If set to 'auto', adjusts based on terminal size 
        or other dynamic measures.
    return_expected_rows_cols : bool, optional
        If True, returns the calculated maximum rows and columns based on 
        internal adjustments or external utilities.

    Returns
    -------
    bool
        Returns True if the DataFrame is considered 'long' based on the criteria
        of either 
        exceeding the max_rows or max_cols limits; otherwise, returns False.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import is_dataframe_long
    >>> data = {'Col1': range(50), 'Col2': range(50)}
    >>> df = pd.DataFrame(data)
    >>> is_dataframe_long(df, max_rows=100, max_cols=10)
    False

    >>> data = {'Col1': range(120), 'Col2': range(120)}
    >>> df = pd.DataFrame(data)
    >>> is_dataframe_long(df, max_rows=100, max_cols=10)
    True

    >>> data = {'Col1': range(50), 'Col2': range(50), 'Col3': range(50)}
    >>> df = pd.DataFrame(data)
    >>> is_dataframe_long(df, max_rows=50, max_cols=2)
    True

    Notes
    -----
    This function is particularly useful for handling or transforming data based on its
    size. It helps in deciding whether certain operations, like specific visualizations
    or data aggregations, should be applied.
    """
    df = validate_data(df )
    rows, columns = df.shape  
    
    # Get terminal size
    # terminal_size = shutil.get_terminal_size()
    # terminal_cols = terminal_size.columns
    # terminal_rows = terminal_size.lines
    _, auto_rows = get_terminal_size()
    auto_cols= get_displayable_columns(
        df, buffer_space=3, min_col_width="auto")
    if max_rows == "auto": 
        max_rows = auto_rows 
        # to terminal row sizes 
    if max_cols =="auto":
        # compared to terminal size. 
        max_cols= auto_cols
        # max_cols = terminal_cols 
    if return_rows_cols_size: 
        # auto_rows, auto_cols = auto_adjust_dataframe_display( df)
        # select the appropriate rows and columns 
        max_rows = _adjust_value(max_rows, auto_rows)
        max_cols = _adjust_value(max_cols, auto_cols)
        return max_rows, max_cols 
    
    # Check if the DataFrame exceeds the specified row or column limits
    return rows > max_rows or columns > max_cols

def df_base_style(
    formatted_df, 
    title=None, 
    table_width="auto", 
    header_line="=", 
    sub_line="-", 
    df=None, 
    column_widths=None, 
    max_index_length=None,
    header=True, 
    index=True
    ):
    """
    Formats a given DataFrame string into a styled textual representation
    that includes a title, headers, sub-headers, and line separations. This
    function ensures that the representation fits within the specified width
    or automatically adjusts to terminal width.

    Parameters
    ----------
    formatted_df : str
        String representation of the DataFrame to be formatted.
    title : str, optional
        Title to be displayed at the top of the formatted table. If None,
        no title is displayed.
    table_width : int or 'auto', optional
        Overall width of the table. If 'auto', adjusts based on content up to
        the maximum terminal width.
    header_line : str, optional
        Character used to create the top and bottom borders of the table.
    sub_line : str, optional
        Character used to create the line below the header row.
    column_widths : dict, optional
        Manually specified widths for each column.
    max_index_length : int, optional
        Maximum length of the index column, used in formatting.
    header : bool, optional
        Whether to include headers in the output.
    index : bool, optional
        Whether to include the index column in the output.
    df : DataFrame, optional
        The original DataFrame, used for directly calculating column widths
        if needed.
    Returns
    -------
    str
        Fully formatted string of the DataFrame, ready for display, including
        the title and table borders.

    Examples
    --------
    >>> formatted_df = "index    A    B    C\\n0       1    2    3\\n1       4    5    6"
    >>> print(df_base_style(formatted_df, title="Demo Table", header_line="="))
    Demo Table
    ====================
    index    A    B    C
    --------------------
    0       1    2    3
    1       4    5    6
    ====================
    """
    # Check if the string representation of the DataFrame 
    # contains truncated parts indicated by '...'
    if '...' in formatted_df:
        # If truncation is detected, use a more robust display 
        # function capable of handling such cases
        return _robust_df_display(
            formatted_df, 
            title=title, 
            column_widths=column_widths, 
            max_index_length=max_index_length, 
            header_line=header_line, 
            header=header, 
            index=index, 
            sub_line=sub_line,
            df=df, 
        )
    
    # Determine the maximum width of any line 
    # in the string representation of the DataFrame
    max_line_width = max(len(line) for line in formatted_df.split('\n'))
    
    # Check if table width is set to 'auto' which indicates 
    # that the width should adapt to the content
    if table_width == 'auto':
        # Retrieve the width of the terminal to 
        # potentially constrain the table width
        terminal_width = shutil.get_terminal_size().columns
        # Ensure that the table width does not exceed the terminal width
        if max_line_width < terminal_width:
            # Use the maximum line width if it is less than the terminal width
            table_width = max_line_width  
        else:
            # Otherwise, use the larger of terminal width or line width
            table_width = max(terminal_width, max_line_width)  
    
    # Format the title and create separators based 
    # on the calculated or specified table width
    
      # Center the title within the table width
    header = f"{title}".center(table_width) if title else ""  
      # Create a header separator line using the specified character
    header_separator = header_line * table_width 
      # Create a sub-header separator line using the specified character
    sub_separator = sub_line * table_width  
    
    # Split the formatted DataFrame into individual lines 
    # to facilitate inserting the sub-header line
    lines = formatted_df.split('\n')
    
    # Construct the formatted output by concatenating 
    # the header, separators, and data lines
    formatted_output = f"{header}\n{header_separator}\n{lines[0]}\n{sub_separator}\n"\
        + "\n".join(lines[1:]) + f"\n{header_separator}"
    
    # Return the complete, formatted string ready for display
    return formatted_output

def _robust_df_display(
    formatted_df, 
    column_widths=None, 
    max_index_length=None, 
    header=True, 
    index=True, 
    sub_line='-', 
    header_line="=", 
    title=None, 
    df=None
    ):
    """
    Formats and displays a DataFrame as a neatly aligned string based on
    specified parameters for column widths, index length, and more.

    Parameters
    ----------
    formatted_df : str
        String representation of the DataFrame to be formatted.
    column_widths : dict, optional
        Specifies custom widths for each column. If not provided,
        widths are calculated automatically based on the content.
    max_index_length : int, optional
        Specifies the maximum length of the index column. If not
        provided, it is calculated automatically.
    header : bool, optional
        If True, includes headers in the output.
    index : bool, optional
        If True, includes index in the output.
    sub_line : str, optional
        Character used to create a line below the header.
    header_line : str, optional
        Character used to create the header separation line.
    title : str, optional
        Title of the DataFrame to be displayed at the top.
    df : DataFrame, optional
        The original DataFrame for which `formatted_df` was created.
        This is used for more accurate width calculations.

    Returns
    -------
    str
        The formatted string of the DataFrame including headers,
        indices, and specified formatting.

    Examples
    --------
    >>> df = pd.DataFrame({
            "A": range(5),
            "B": ['one', 'two', 'three', 'four', 'five']
        })
    >>> formatted_str = df.to_string()
    >>> print(_df_display(
            formatted_str, header=True, index=True, title="Sample DataFrame"))
    |                         Sample DataFrame                          |
    ===================================================================
     A    B
    ----------------------------------------
     0  one
     1  two
     2  three
     3  four
     4  five
    ===================================================================
    """
    
    # Split the input string (formatted DataFrame) into separate lines
    lines = formatted_df.split('\n')
    new_lines = []

    # Calculate widths for index and columns with a limit 
    # of 50 characters for each text element
    auto_max_index_length, *auto_column_widths = calculate_column_widths(
        lines, include_index=True, include_column_width=True, df=df,
        max_text_length=50
    )
  
    # Apply user-defined column widths if provided
    if column_widths:
        headers = lines[0].split()
        column_widths = {col.replace(' ', GOFAST_ESCAPE): val 
                         for col, val in column_widths.items()}
        column_widths = [column_widths[col] 
                         for col in headers if col in column_widths]

    # Use automatically calculated widths if no widths are provided
    if column_widths is None:
        column_widths = auto_column_widths
    
    
    # Use the automatically determined index length if not specified
    if max_index_length is None:
        max_index_length = auto_max_index_length
    max_index_length += 3  # Add extra spaces for alignment
 
    # Format the header line with adjusted widths
    header_parts = lines[0].split()
    header_line_formatted = " " * max_index_length + "  ".join(
        header_part.rjust(column_widths[idx]) 
        for idx, header_part in enumerate(header_parts)
    )
    table_width = len(header_line_formatted)
    # Process each line to align with the header formatting
    for i, line in enumerate(lines):
        if i == 0 and header:
            new_lines.append(header_line_formatted)
            continue
        elif i == 1 and header:
            new_lines.append(sub_line * table_width)
            continue
        
        # Split line into index and data parts if index is included
        if index:
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                index_part, data_part = parts
        else:
            parts = line

        # Format the data part of each line
        if len(parts) > 1:
            data_part.split()
            formatted_data_part = "  ".join(
                data.rjust(column_widths[idx]) for idx, data in enumerate(
                    data_part.split())
            )
            if index:
                new_line = f"{index_part.ljust(max_index_length - 2)}  {formatted_data_part}"
            else:
                new_line = f"{formatted_data_part}"
        else:
            new_line = " " * (max_index_length - 2) + line
        
        new_lines.append(new_line)
    
    # Adjust the overall table width based on content
    max_line_width = max(len(line) for line in new_lines)
    table_width = max_line_width if table_width == 'auto' else max(
        min(table_width, max_line_width), len(header_line_formatted))

    # Add a title and a header separator if a title is provided
    header = f"{title}".center(table_width) if title else ""
    header_separator = header_line * table_width

    # Compile the formatted output
    formatted_output = f"{header}\n{header_separator}\n" + "\n".join(
        new_lines) + f"\n{header_separator}"
    
    return formatted_output

def make_format_df(subset_df, whitespace_sub="%g%o#f#", apply_to_column=False):
    """
    Creates a new DataFrame where each string value of each column that 
    contains a whitespace is replaced by '%g%o#f#'. This is useful to fix the 
    issue with multiple whitespaces in all string values of the DataFrame. Optionally,
    replaces whitespaces in column names as well.

    Parameters:
        subset_df (pd.DataFrame): The input DataFrame to be formatted.
        whitespace_sub (str): The substitution string for whitespaces.
        apply_to_column (bool): If True, also replace whitespaces in column names.

    Returns:
        pd.DataFrame: A new DataFrame with formatted string values.
    """
    # Create a copy of the DataFrame to avoid modifying the original one
    formatted_df = subset_df.copy()
    
    # Optionally replace whitespaces in column names
    if apply_to_column:
        formatted_df.columns = [col.replace(' ', whitespace_sub) 
                                for col in formatted_df.columns]

    # Loop through each column in the DataFrame
    for col in formatted_df.columns:
        # Check if the column type is object (typically used for strings)
        if formatted_df[col].dtype == object: 
            # Replace whitespaces in string values with '%g%o#f#'
            formatted_df[col] = formatted_df[col].replace(
                r'\s+', whitespace_sub, regex=True)
    
    return formatted_df

def df_advanced_style(
    formatted_df, 
    table_width="auto", 
    index=True, 
    header=True,
    title=None, 
    header_line="=", 
    sub_line="-", 
    df=None,
    column_widths=None, 
    max_index_length=None, 
    ):
    """
    Applies advanced styling to a DataFrame string by formatting it with 
    headers, sub-headers, indexed rows, and aligning columns properly.

    Parameters
    ----------
    formatted_df : str
        The string representation of the DataFrame to format, typically 
        generated by pandas.DataFrame.to_string().
    table_width : int or str, optional
        The total width of the table. If 'auto', it adjusts based on the 
        content width and terminal size.
    index : bool, optional
        Whether to consider the index of DataFrame rows in formatting.
    header : bool, optional
        Whether to include column headers in the formatted output.
    title : str, optional
        Title text to be displayed above the table. If None, no title is shown.
    header_line : str, optional
        The character used for the main separator lines in the table.
    sub_line : str, optional
        The character used for sub-separators within the table, typically 
        under headers.
    df : DataFrame, optional
        A pandas DataFrame from which column widths can be calculated directly, 
        used as a fallback if complex column names are detected in the 
        'lines' input.
    column_widths : [dict of col, int ] or list of int or None, optional
        Specifies the widths for each column within the DataFrame. This list 
        should correspond to the actual columns in the DataFrame and can 
        include the index width at the beginning if `index` is set to True. 
        If `None`, the widths are automatically calculated by the 
        `calculate_column_widths` function based on the content of the
        DataFrame and a specified maximum text length, ensuring optimal 
        formatting for display or printing.
    max_index_length : int or None, optional
        The maximum width allocated for formatting the index of the DataFrame.
        If `None`, the width is automatically calculated based on the contents
        of the index, ensuring that the index is visually consistent and 
        properly aligned with the data columns. This automatic calculation is 
        designed to adapt to the varying lengths of index entries, providing a 
        dynamically adjusted display that optimally uses available space.

    Returns
    -------
    str
        The fully formatted and styled table as a single string, ready for display.

    Examples
    --------
    >>> from gofast.api.util import df_advanced_style
    >>> formatted_df = "index    A    B    C\\n0       1    2    3\\n1       4    5    6"
    >>> print(df_advanced_style(formatted_df, title="Advanced Table", index=True))
        Advanced Table  
     ===================
          index  A  B  C
        ----------------
     1  |     4  5  6
     ===================
    """

    # Split the formatted DataFrame string into individual lines.
    lines = formatted_df.split('\n')
    new_lines = []
    #print(index)
    # Calculate the automatic widths for the index and columns based on the content,
    # specifying the DataFrame, and limiting text length to 50 characters.
    auto_max_index_length, *auto_column_widths = calculate_column_widths(
        lines, include_index=index, include_column_width=True,
        df=df, max_text_length=50
    )
    # If explicit column widths are provided, use them.
    if column_widths:
        # Extract headers from the first line (assumes they are space-separated).
        headers = lines[0].split()
        # Modify the column names to include GOFAST_ESCAPE 'π' which helps 
        # to distinguish between
        # similar names that differ only in spacing when parsing widths.
        column_widths = {col.replace(' ', GOFAST_ESCAPE): val 
                         for col, val in column_widths.items()}
        # Filter the provided column widths to include only those that match the headers.
        # This prevents applying widths for columns that aren't present in the headers.
        column_widths = [column_widths[col] for col in headers if col in column_widths]
    
    # If no column widths are provided, fall back to automatically calculated widths.
    if column_widths is None:
        column_widths = auto_column_widths

    # If no maximum index length is provided, use the automatically determined length.
    if max_index_length is None:
        max_index_length = auto_max_index_length
        
    max_index_length +=3  # for extra spaces 

    for i, line in enumerate(lines):
        if i == 0 and header:  # Adjust header line to include vertical bar
            header_parts = line.split()
            header_line_formatted = " " *  max_index_length + "  ".join(
                header_part.rjust(
                    column_widths[idx]) for idx, header_part in enumerate(header_parts)
                )
            new_lines.append(header_line_formatted)
            continue 
        elif i == 1 and header:  # Insert sub-line after headers ( reduce extraspace -1)
            add_move_space = " " * (max_index_length -1)
            new_lines.append(
                add_move_space   + sub_line * ( # remove extra space 
                    len(header_line_formatted) - len (add_move_space))) 
            continue

        parts = line.split(maxsplit=1)
        if len(parts) > 1:
            index_part, data_part = parts
            data_part.split()
            formatted_data_part = "  ".join(
                data.rjust(column_widths[idx]) for idx, data in enumerate(data_part.split())
                )
            new_line = f"{index_part.ljust(max_index_length -2)} |{formatted_data_part}"
        else:
            new_line = " " * max_index_length  + line
            
        new_lines.append(new_line)

    
    max_line_width = max(len(line) for line in new_lines)
    table_width = max_line_width if table_width == 'auto' else max(
        min(table_width, max_line_width), len(header))

    header = f"{title}".center(table_width) if title else ""
    header_separator = header_line * table_width

    formatted_output = f"{header}\n{header_separator}\n" + "\n".join(
        new_lines) + f"\n{header_separator}"

    return formatted_output

def calculate_column_widths(
    lines, 
    include_index=True,
    include_column_width=True, 
    df=None, 
    max_text_length=50
):
    """
    Calculates the maximum width for each column based on the content of the 
    DataFrame's rows and optionally considers column headers for width calculation. 
    
    If complex multi-word columns are detected and a DataFrame is provided, 
    widths are calculated directly from the DataFrame.

    Parameters
    ----------
    lines : list of str
        List of strings representing rows in a DataFrame, where the first
        line is expected to be the header and subsequent lines the data rows.
    include_index : bool, optional
        Determines whether the index column's width should be included in the 
        width calculations. Default is True.
    include_column_width : bool, optional
        If True, column header widths are considered in calculating the maximum 
        width for each column. Default is True.
    df : DataFrame, optional
        A pandas DataFrame from which column widths can be calculated directly, 
        used as a fallback if complex column names are detected in the 'lines' input.
    max_text_length : int, optional
        The maximum allowed length for any cell content, used when calculating widths
        from the DataFrame.

    Returns
    -------
    list of int
        A list containing the maximum width for each column. If `include_index`
        is True, the first element represents the index column's width.

    Raises
    ------
    TypeError
        If an IndexError occurs due to improper splitting of 'lines' and no DataFrame
        is provided to fall back on for width calculations.

    Examples
    --------
    >>> from gofast.api.util import calculate_column_widths
    >>> lines = [
    ...     '    age  tenure_months  monthly_charges',
    ...     '0     0              0          89.0012',
    ...     '1     1              1          94.0247',
    ...     '2     2              2          71.6051',
    ...     '3     3              3          67.5316',
    ...     '4     4              4          86.3517',
    ...     '..  ...            ...              ...',
    ...     '25   25             25          22.3356',
    ...     '26   26             26          73.1798',
    ...     '27   27             27          52.7984',
    ...     '28   28             28          83.3604',
    ...     '29   29             29          88.6392'
    ... ]
    >>> calculate_column_widths(lines, include_index=True, include_column_width=True)
    [2, 3, 13, 15]
    >>> lines = [
    ...     'age  tenure_months  monthly_charges',
    ...     '0     0              0          89.0012',
    ...     '1     1              1          94.0247',
    ...     '2     2              2          71.6051'
    ... ]
    >>> calculate_column_widths(lines, include_index=True, include_column_width=True)
    [2, 3, 14, 10]

    Notes
    -----
    This function is particularly useful for formatting data tables in text-based
    outputs where column alignment is important for readability. The widths can
    be used to format tables with proper spacing and alignment across different
    data entries.
    """
    max_widths = []

    # Split the header to get the number of columns 
    # and optionally calculate header widths
    header_parts = lines[0].strip().split()
    num_columns = len(header_parts)

    # Initialize max widths list
    if include_index:
        max_widths = [0] * (num_columns + 1)
    else:
        max_widths = [0] * num_columns

    # Include column names in the width if required
    if include_column_width:
        for i, header in enumerate(header_parts):
            if include_index:
                max_widths[i+1] = max(max_widths[i+1], len(header))
            else:
                max_widths[i] = max(max_widths[i], len(header))

    try: 
        for line in lines[1:]:
            parts = line.strip().split()
            if include_index:
                for i, part in enumerate(parts):
                    max_widths[i] = max(max_widths[i], len(part))
            else:
                for i, part in enumerate(parts[1:], start=1):
                    max_widths[i] = max(max_widths[i], len(part))
    except IndexError as e:
        if df is None:
            raise ValueError(
                "An error occurred while splitting line data. Multi-word column"
                " values may be causing this issue. Please provide a DataFrame"
                " for more accurate parsing."
            ) from e
        
        # If a DataFrame is provided, calculate widths directly from the DataFrame
        column_widths, max_index_length = calculate_widths(
            df, max_text_length= max_text_length)

        for ii, header in enumerate(header_parts):
            # Handle ellipsis or unspecified header placeholders
            if header == '...':
                max_widths[ii] = 3  # Width for '...'
                continue
            if header in column_widths:
                max_widths[ii] = column_widths[header]
                
        if include_index:
            max_widths.insert(0, max_index_length)

    return max_widths

def _adjust_value(max_value, auto_value):
    """
    Adjusts the user-specified maximum number of rows or columns based on 
    an automatically determined maximum. This helps to ensure that the number 
    of rows or columns displayed does not exceed what can be practically or 
    visually accommodated on the screen.

    Parameters
    ----------
    max_value : int, float, or 'auto'
        The user-specified maximum number of rows or columns. This can be an 
        integer, a float, or 'auto', which indicates that the maximum should 
        be determined automatically.
        - If an integer or float is provided, it will be compared to `auto_value`.
        - If 'auto' is specified, `auto_value` will be used as the maximum.

    auto_value : int
        The maximum number of rows or columns determined based on the terminal 
        size or DataFrame dimensions. This value is used as a fallback or 
        comparative value when `max_value` is numeric.

    Returns
    -------
    int
        The adjusted maximum number of rows or columns to display. This value 
        is determined by comparing `max_value` and `auto_value` and choosing 
        the smaller of the two if `max_value` is numeric, or returning 
        `auto_value` directly if `max_value` is 'auto'.

    Examples
    --------
    >>> adjust_value(50, 30)
    30

    >>> adjust_value('auto', 25)
    25

    >>> adjust_value(20, 40)
    20

    Notes
    -----
    This function is intended to be used within larger functions that manage 
    the display of DataFrame objects where terminal or screen size constraints
    might limit the practical number of rows or columns that can be displayed 
    at one time.
    """
    if isinstance(max_value, (int, float)):
        return min(max_value, auto_value)
    return auto_value

def auto_adjust_dataframe_display(df, header=True, index=True, sample_size=100):
    """
    Automatically adjusts the number of rows and columns to display based on the
    terminal size and the contents of the DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to display.
    header : bool, optional
        Whether to include the header in the display, by default True.
    index : bool, optional
        Whether to include the index column in the display, by default True.
    sample_size : int, optional
        Number of entries to sample for calculating the average entry width, by
        default 100.

    Returns
    -------
    tuple
        A tuple (max_rows, max_cols) representing the maximum number of rows
        and columns to display based on the terminal dimensions and data content.

    Examples
    --------
    >>> from gofast.api.util import auto_adjust_dataframe_display
    >>> df = pd.DataFrame(np.random.randn(100, 10), columns=[f"col_{i}" for i in range(10)])
    >>> max_rows, max_cols = auto_adjust_dataframe_display(df)
    >>> print(f"Max Rows: {max_rows}, Max Cols: {max_cols}")
    >>> print(df.to_string(max_rows=max_rows, max_cols=max_cols))
    """
    validate_data(df)
    # Get terminal size
    
    terminal_size = shutil.get_terminal_size()
    screen_width = terminal_size.columns
    screen_height = terminal_size.lines

    # Estimate the average width of data entries
    sample = df.sample(n=min(sample_size, len(df)), random_state=1)
    sample_flat = pd.Series(dtype=str)
    if index:
        series_to_concat = [sample.index.to_series().astype(str)]
    else:
        series_to_concat = []
    for column in sample.columns:
        series_to_concat.append(sample[column].astype(str))
    sample_flat = pd.concat(series_to_concat, ignore_index=True)
    avg_entry_width = int(sample_flat.str.len().mean()) + 1  # Plus one for spacing

    # Determine the width used by the index
    index_width = max(len(str(idx)) for idx in df.index) if index else 0

    # Calculate the available width for data columns
    # Adjust for spacing between columns
    available_width = screen_width - index_width - 3  

    # Estimate the number of columns that can fit
    max_cols = available_width // avg_entry_width
    
    # Adjust for header if present
    header_height = 1 if header else 0
    
    # Calculate the number of rows that can fit
    # Subtract for header and to avoid clutter
    max_rows = screen_height - header_height - 3  

    # Ensure max_cols does not exceed number of DataFrame columns
    max_cols = min(max_cols, len(df.columns))

    # Ensure max_rows does not exceed number of DataFrame rows
    max_rows = min(max_rows, len(df))

    return max_rows, max_cols

def find_best_display_params(
        *dfs, max_rows=None, max_cols=None, header=True,
         index=True, sample_size=100):
    """
    Determines optimal display settings for a collection of DataFrames
    based on the desired maximum number of rows and columns to display.

    Parameters
    ----------
    *dfs : tuple of DataFrame
        Variable number of DataFrame objects to analyze.
    max_rows : int or str, optional
        Desired maximum number of rows to display. If 'auto' or None,
        the maximum across all DataFrames is used.
    max_cols : int or str, optional
        Desired maximum number of columns to display. If 'auto' or None,
        the maximum across all DataFrames is used.
    header : bool, optional
        Whether to include headers in the display settings.
    index : bool, optional
        Whether to include indices in the display settings.
    sample_size : int, optional
        Number of rows to sample from each DataFrame for determining
        display settings.

    Returns
    -------
    tuple
        A tuple containing the optimal number of rows and columns
        (max_rows, max_cols) to display.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd 
    >>> from gofast.api.util import find_best_display_params
    >>> df1 = pd.DataFrame(np.random.randn(100, 5))
    >>> df2 = pd.DataFrame(np.random.randn(200, 10))
    >>> max_rows, max_cols = find_best_display_params(
        df1, df2, max_rows='auto',max_cols='auto')
    >>> print(max_rows, max_cols)

    Notes
    -----
    The 'auto' option for max_rows and max_cols calculates the maximum
    number of rows and columns needed across all provided DataFrames,
    ensuring that all content is optimally visible without truncation.
    """
    max_rows_list = []
    max_cols_list = []

    # Collect maximum row and column counts needed for each DataFrame
    for df in dfs:
        optimal_rows, optimal_cols = auto_adjust_dataframe_display(
            df, header=header, index=index, sample_size=sample_size
        )
        max_rows_list.append(optimal_rows)
        max_cols_list.append(optimal_cols)

    # Determine the global maxima for rows and columns
    optimal_max_rows = max(max_rows_list)
    optimal_max_cols = max(max_cols_list)

    # Apply 'auto' or None logic to rows and columns
    max_rows = max_rows or "auto"
    max_cols = max_cols or "auto"
    if max_rows =="auto":
        max_rows = optimal_max_rows
    if max_cols =="auto":
        max_cols = optimal_max_cols

    return max_rows, max_cols

def find_best_display_params2(*dfs, index=True, header=True, sample_size=100):
    """
    Returns a function that determines the best display settings for the 
    given DataFrames based on their content. The settings include the optimal
    number of rows and columns to display.

    Parameters
    ----------
    *dfs : tuple of DataFrame
        Variable number of DataFrame objects to analyze.
    index : bool, optional
        Whether to include the index in the display settings calculations.
    header : bool, optional
        Whether to include headers in the display settings calculations.
    sample_size : int, optional
        The number of rows from each DataFrame to sample for setting determination.

    Returns
    -------
    function
        A function that takes max_rows and max_cols as parameters and returns
        the optimal number of rows and columns based on pre-computed values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd 
    >>> from gofast.api.util import find_best_display_params2
    >>> df1 = pd.DataFrame(np.random.randn(100, 5))
    >>> df2 = pd.DataFrame(np.random.randn(200, 10))
    >>> param_resolver = find_best_display_params2(df1, df2, index=True,
    ...                                         header=True, sample_size=100)
    >>> max_rows, max_cols = param_resolver(max_rows='auto', max_cols='auto')
    >>> print(max_rows, max_cols)
    26 4
    """
    max_rows_list = []
    max_cols_list = []

    # Collect maximum row and column counts needed for each DataFrame
    for df in dfs:
        optimal_rows, optimal_cols = auto_adjust_dataframe_display(
            df, header=header, index=index, sample_size=sample_size
        )
        max_rows_list.append(optimal_rows)
        max_cols_list.append(optimal_cols)

    # Define a function to resolve 'auto' settings
    def resolve_auto(max_rows='auto', max_cols='auto'):
        # Determine the global maxima for rows and columns
        optimal_max_rows = max(max_rows_list)
        optimal_max_cols = max(max_cols_list)

        # Apply 'auto' or None logic to rows and columns
        final_max_rows = optimal_max_rows if max_rows in (None, 'auto') else max_rows
        final_max_cols = optimal_max_cols if max_cols in (None, 'auto') else max_cols

        return final_max_rows, final_max_cols

    return resolve_auto


def parse_component_kind(pc_list, kind):
    """
    Extracts specific principal component's feature names and their importance
    values from a list based on a given component identifier.

    Parameters
    ----------
    pc_list : list of tuples
        A list where each tuple contains ('pc{i}', feature_names, 
                                          sorted_component_values),
        corresponding to each principal component. 'pc{i}' is a string label 
        like 'pc1', 'feature_names' is an array of feature names sorted by 
        their importance, and 'sorted_component_values' are the corresponding 
        sorted values of component loadings.
    kind : str
        A string that identifies the principal component number to extract, 
        e.g., 'pc1'. The string should contain a numeric part that corresponds
        to the component index in `pc_list`.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - An array of feature names for the specified principal component.
        - An array of sorted component values for the specified principal 
          component.

    Raises
    ------
    ValueError
        If the `kind` parameter does not contain a valid component number or if the
        specified component number is out of the range of available components
        in `pc_list`.

    Examples
    --------
    >>> from gofast.api.extension import parse_component_kind
    >>> pc_list = [
    ...     ('pc1', ['feature1', 'feature2', 'feature3'], [0.8, 0.5, 0.3]),
    ...     ('pc2', ['feature1', 'feature2', 'feature3'], [0.6, 0.4, 0.2])
    ... ]
    >>> feature_names, importances = parse_component_kind(pc_list, 'pc1')
    >>> print(feature_names)
    ['feature1', 'feature2', 'feature3']
    >>> print(importances)
    [0.8, 0.5, 0.3]

    Notes
    -----
    The function requires that the `kind` parameter include a numeric value 
    that accurately represents a valid index in `pc_list`. The index is derived
    from the numeric part of the `kind` string and is expected to be 1-based. 
    If no valid index is found or if the index is out of range, the function 
    raises a `ValueError`.
    """
    match = re.search(r'\d+', str(kind))
    if match:
        # Convert 1-based index from `kind` to 0-based index for list access
        index = int(match.group()) - 1  
        if index < len(pc_list) and index >= 0:
            return pc_list[index][1], pc_list[index][2]
        else:
            raise ValueError(f"Component index {index + 1} is out of the"
                             " range of available components.")
    else:
        raise ValueError("The 'kind' parameter must include an integer"
                         " indicating the desired principal component.")

def find_maximum_table_width(summary_contents, header_marker='='):
    """
    Calculates the maximum width of tables in a summary string based on header lines.

    This function parses a multi-table summary string, identifying lines that represent
    the top or bottom borders of tables (header lines). It determines the maximum width
    of these tables by measuring the length of these header lines. The function assumes
    that the header lines consist of repeated instances of a specific marker character.

    Parameters
    ----------
    summary_contents : str
        A string containing the summarized representation of one or more tables.
        This string should include header lines made up of repeated header markers
        that denote the start and end of each table's border.
    header_marker : str, optional
        The character used to construct the header lines in the summary_contents.
        Defaults to '=', the common character for denoting table borders in ASCII
        table representations.

    Returns
    -------
    int
        The maximum width of the tables found in summary_contents, measured as the
        length of the longest header line. If no header lines are found, returns 0.

    Examples
    --------
    >>> from gofast.api.util import find_maximum_table_width
    >>> summary = '''Model Performance
    ... ===============
    ... Estimator : SVC
    ... Accuracy  : 0.9500
    ... Precision : 0.8900
    ... Recall    : 0.9300
    ... ===============
    ... Model Performance
    ... =================
    ... Estimator : RandomForest
    ... Accuracy  : 0.9500
    ... Precision : 0.8900
    ... Recall    : 0.9300
    ... ================='''
    >>> find_maximum_table_width(summary)
    18

    This example shows how the function can be used to find the maximum table width
    in a string containing summaries of model performances, where '=' is used as
    the header marker.
    """
    # Split the input string into lines
    lines = summary_contents.split('\n')
    # Filter out lines that consist only of the header 
    # marker, and measure their lengths
    header_line_lengths = [len(line) for line in lines if line.strip(
        header_marker) == '']
    # Return the maximum of these lengths, or 0 if the list is empty
    return max(header_line_lengths, default=0)

def format_text(
        text, key=None, key_length=15, max_char_text=50, 
        add_frame_lines =False, border_line='=' ):
    """
    Formats a block of text to fit within a specified maximum character width,
    optionally prefixing it with a key. If the text exceeds the maximum width,
    it wraps to a new line, aligning with the key or the specified indentation.

    Parameters
    ----------
    text : str
        The text to be formatted.
    key : str, optional
        An optional key to prefix the text. Defaults to None.
    key_length : int, optional
        The length reserved for the key, including following spaces.
        If `key` is provided but `key_length` is None, the length of the
        `key` plus one space is used. Defaults to 15.
    max_char_text : int, optional
        The maximum number of characters for the text width, including the key
        if present. Defaults to 50.
    add_frame_lines: bool, False 
       If True, frame the text with '=' line (top and bottom)
    border_line: str, optional 
      The border line to frame the text.  Default is '='
      
    Returns
    -------
    str
        The formatted text with line breaks added to ensure that no line exceeds
        `max_char_text` characters. If a `key` is provided, it is included only
        on the first line, with subsequent lines aligned accordingly.

    Examples
    --------
    >>> from gofast.api.util import format_text
    >>> text_example = ("This is an example text that is supposed to wrap" 
                      "around after a certain number of characters.")
    >>> print(format_text(text_example, key="Note"))
    Note           : This is an example text that is supposed to
                      wrap around after a certain number of
                      characters.

    Notes
    -----
    - The function dynamically adjusts the text to fit within `max_char_text`,
      taking into account the length of `key` if provided.
    - Text that exceeds the `max_char_text` limit is wrapped to new lines, with
      proper alignment to match the initial line's formatting.
    """
    
    if key is not None:
        # If key_length is None, use the length of the key + 1 
        # for the space after the key
        if key_length is None:
            key_length = len(key) + 1
        key_str = f"{key.ljust(key_length)} : "
    elif key_length is not None:
        # If key is None but key_length is specified, use spaces
        key_str = " " * key_length + " : "
    else:
        # If both key and key_length are None, there's no key part
        key_str = ""
    
    # Adjust max_char_text based on the length of the key part
    effective_max_char_text = (max_char_text - len(key_str) + 2 if key_str else max_char_text)
    formatted_text = ""
    text=str(text)
    while text:
        # If the remaining text is shorter than the effective
        # max length, or if there's no key part, add it as is
        if len(text) <= effective_max_char_text - 4 or not key_str: # -4 for extraspace 
            formatted_text += key_str + text
            break
        else:
            # Find the space to break the line, ensuring it doesn't
            # exceed effective_max_char_text
            break_point = text.rfind(' ', 0, effective_max_char_text-4)
            
            if break_point == -1:  # No spaces found, force break
                break_point = effective_max_char_text -4 
            # Add the line to formatted_text
            formatted_text += key_str + text[:break_point].rstrip() + "\n"
            # Remove the added part from text
            text = text[break_point:].lstrip()
   
            # After the first line, the key part is just spaces
            key_str = " " * len(key_str)

    if add_frame_lines: 
        frame_lines = border_line * (effective_max_char_text + 1 )
        formatted_text = frame_lines +'\n' + formatted_text +'\n' + frame_lines

    return formatted_text

def format_value(value):
    """
    Format a numeric value to a string, rounding floats to four decimal
    places and converting integers directly to strings.
    
    Parameters
    ----------
    value : int, float, np.integer, np.floating
        The numeric value to be formatted.

    Returns
    -------
    str
        A formatted string representing the value.
    
    Examples
    --------
    >>> from gofast.api.util import format_value
    >>> format_value(123)
    '123'
    >>> format_value(123.45678)
    '123.4568'
    """
    value_str =str(value)
    if isinstance(value, (int, float, np.integer, np.floating)): 
        value_str = f"{value}" if isinstance ( 
            value, int) else  f"{float(value):.4f}" 
    return value_str 

def get_frame_chars(frame_char):
    """
    Retrieve framing characters based on the input frame indicator.
    
    Parameters
    ----------
    frame_char : str
        A single character that indicates the desired framing style.

    Returns
    -------
    tuple
        A tuple containing the close character and the open-close pair
        for framing index values.

    Examples
    --------
    >>> from gofast.api.util import get_frame_chars
    >>> get_frame_chars('[')
    (']', '[', ']')
    >>> get_frame_chars('{')
    ('}', '{', '}')
    """
    pairs = {
        '[': (']', '[', ']'),
        '{': ('}', '{', '}'),
        '(': (')', '(', ')'),
        '<': ('>', '<', '>')
    }
    return pairs.get(frame_char, ('.', '.', '.'))

def df_to_custom_dict(df, key_prefix='Row', frame_char='['):
    """
    Convert a DataFrame to a dictionary with custom formatting for keys
    and values, applying specified framing characters for non-unique 
    numeric indices.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to convert.
    key_prefix : str, optional
        The prefix for keys when the index is numeric and non-unique,
        default is 'Row'.
    frame_char : str, optional
        The character to determine framing for keys, default is '['.
    
    Returns
    -------
    dict
        A dictionary with custom formatted keys and values.

    Examples
    --------
    >>> from gofast.api.util import df_to_custom_dict
    >>> df = pd.DataFrame({'col0': [10, 20], 'col1': [30, 40]}, 
                          index=['a', 'b'])
    >>> dataframe_to_custom_dict(df)
    {'a': 'col0 <10> col1 <30>', 'b': 'col0 <20> col1 <40>'}
    """
    frame_open, frame_close = get_frame_chars(frame_char)[1:]
    key_format = (f"{key_prefix}{{}}" if df.index.is_unique 
                  and df.index.inferred_type == 'integer' else "{}")
    
    return {key_format.format(f"{frame_open}{index}{frame_close}" 
                              if key_format.startswith(key_prefix) else index):
            ' '.join(f"{col} <{format_value(val)}>" for col, val in row.items())
            for index, row in df.iterrows()}


def format_cell(x, max_text_length, max_width =None ):
    """
    Truncates a string to the maximum specified length and appends '...' 
    if needed, and right-aligns it.

    Parameters:
    x (str): The string to format.
    max_width (int): The width to which the string should be aligned.
    max_text_length (int): The maximum allowed length of the string before truncation.

    Returns:
    str: The formatted and aligned string.
    """
    x = str(x)
    if len(x) > max_text_length:
        x = x[:max_text_length - 3] + '...'
    return x.rjust(max_width) if max_width else x 

def calculate_widths(df, max_text_length=50):
    """
    Calculates the maximum widths for each column in a DataFrame based on the 
    content, with a cap on the maximum text length for any cell.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame for which to calculate column widths.
    max_text_length : int, optional 
        The maximum allowed length for any cell content. If the content exceeds 
        this length, it will be truncated. Default is 50 

    Returns
    -------
    tuple
        A tuple containing two elements:
        - A dictionary with column names as keys and their calculated maximum
          widths as values.
        - The maximum width of the index column.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import calculate_widths
    >>> df = pd.DataFrame({
    ...     'usage': [12345, 123],
    ...     'time': ['long text here that will be cut', 'short']
    ... })
    >>> calculate_widths(df['usage'].to_frame(), 50)
    ({'usage': 5}, 3)
    
    The example shows how to call `calculate_widths` on a DataFrame created 
    from a series and specifies a maximum text length of 50 characters. The 
    output indicates the maximum width required for the 'usage' column and 
    the index.
    """

    formatted_cells = df.applymap(lambda x: str(format_value(x))
                                  [:max_text_length] + '...' if len(
        str(x)) > max_text_length else str(format_value(x)))
    max_col_widths = {col: max(len(col), max(len(x) for x in formatted_cells[col]))
                      for col in df.columns}
    max_index_width = max(len(str(index)) for index in df.index)
    max_col_widths = {col: min(width, max_text_length) 
                      for col, width in max_col_widths.items()}
    return max_col_widths, max_index_width

def format_df(
    df, 
    max_text_length=50, 
    title=None, 
    autofit=False, 
    ):
    """
    Formats a pandas DataFrame for pretty-printing in a console or
    text-based interface. This function provides a visually-appealing
    tabular representation with aligned columns and a fixed maximum
    column width.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be formatted.
        
    max_text_length : int, optional
        The maximum length of text within each cell of the DataFrame.
        Default is 50 characters. Text exceeding this length will be
        truncated.
        
    title : str, optional
        An optional title for the formatted correlation matrix. If provided, it
        is centered above the matrix. Default is None.
        
    autofit : bool, optional
        If True, adjusts the column widths and the number of visible rows
        based on the DataFrame's content and available display size. Default
        is False.

    Returns
    -------
    str
        A formatted string representation of the DataFrame with columns
        and rows aligned, headers centered, and cells truncated according
        to `max_text_length`.

    Notes
    -----
    This function depends on helper functions `calculate_widths` to 
    determine the maximum widths for DataFrame columns based on the 
    `max_text_length`, and `format_cell` to appropriately format and
    truncate cell content. It handles both the DataFrame's index and
    columns to ensure a clean and clear display.

    Examples
    --------
    Consider a DataFrame `df` created as follows:
    
    >>> import pandas as pd 
    >>> from gofast.api.util import format_df 
    >>> data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Occupation': ['Engineer', 'Doctor', 'Artist'],
        'Age': [25, 30, 35]
    }
    >>> df = pd.DataFrame(data)

    Formatting `df` with `format_df`:

    >>> print(format_df(df, max_text_length=10))
    =============================
           Name   Occupation  Age
      ---------------------------
    0 |    Alice    Engineer   25
    1 |      Bob      Doctor   30
    2 |  Charlie      Artist   35
    =============================
    
    >>> from gofast.datasets.simulate import simulate_medical_diagnosis 
    >>> med_data = medical_diagnosis(as_frame=True ) 
    >>> print(format_correlations (med_data, autofit= True ) ) 

    Output shows adjusted row and column display based on the DataFrame's content,
    ensuring all fields are visible without exceeding the provided max text length.
    
    """

    # Set the title or default to an empty string
    title = str(title or '').title()
   
    if autofit: 
        # If autofit is True, use custom logic to adjust DataFrame display settings
        # dynamically based on content and available space.
        # Here, the function extract_truncate_df limits the display to 11 rows
        # and 7 columns for larger data sets to fit the console or output window.
        df_escaped = extract_truncate_df(df, max_rows="auto", max_cols="auto")
        # insert_ellipsis_to_df might add visual cues (ellipsis) to indicate truncated parts.
        df = insert_ellipsis_to_df(df_escaped, df) 

    # Use helper functions to format cells and calculate widths
    max_col_widths, max_index_width = calculate_widths(df, max_text_length)

    # Formatting the header
    header = " " * (max_index_width + 4) + "  ".join(col.center(
        max_col_widths[col]) for col in df.columns)
    separator = " " * (max_index_width + 1) + "-" * (len(header) - (max_index_width + 1))

    # Formatting the rows
    data_rows = [
        f"{str(index).ljust(max_index_width)} |  " + 
        "  ".join(format_cell(row[col], max_text_length, max_col_widths[col]).ljust(
            max_col_widths[col]) for col in df.columns)
        for index, row in df.iterrows()
    ]

    # Creating top and bottom borders
    full_width = len(header)
    top_border = "=" * full_width
    bottom_border = "=" * full_width

    # Full formatted table
    formatted_string = f"{top_border}\n{header}\n{separator}\n" + "\n".join(
        data_rows) + "\n" + bottom_border
    
    if title:
        max_width = find_maximum_table_width(formatted_string)
        title = title.center(max_width) + "\n"
    return title + formatted_string

def validate_data(data, columns=None, error_mode='raise'):
    """
    Validates and converts input data into a pandas DataFrame, handling
    various data types such as DataFrame, ndarray, dictionary, and Series.

    Parameters
    ----------
    data : DataFrame, ndarray, dict, Series
        The data to be validated and converted into a DataFrame.
    columns : list, str, optional
        Column names for the DataFrame. If provided, they should match the
        data dimensions. If not provided, default names will be generated.
    error_mode : str, {'raise', 'warn'}, default 'raise'
        Error handling behavior: 'raise' to raise errors, 'warn' to issue
        warnings and use default settings.

    Returns
    -------
    DataFrame
        A pandas DataFrame constructed from the input data.

    Raises
    ------
    ValueError
        If the number of provided columns does not match the data dimensions
        and error_mode is 'raise'.
    TypeError
        If the input data type is not supported.

    Notes
    -----
    This function is designed to simplify the process of converting various
    data types into a well-formed pandas DataFrame, especially when dealing
    with raw data from different sources. The function is robust against
    common data input errors and provides flexible error handling through
    the `error_mode` parameter.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.api.util import validate_data
    >>> data = np.array([[1, 2], [3, 4]])
    >>> validate_data(data)
       feature_0  feature_1
    0          1          2
    1          3          4

    >>> data = {'col1': [1, 2], 'col2': [3, 4]}
    >>> validate_data(data, columns=['column1', 'column2'])
       column1  column2
    0        1        3
    1        2        4

    >>> data = pd.Series([1, 2, 3])
    >>> validate_data(data, error_mode='warn')
       feature_0
    0          1
    1          2
    2          3
    """
    def validate_columns(data_columns, expected_columns):
        if expected_columns is None:
            return [f'feature_{i}' for i in range(data_columns)]
        
        if isinstance(expected_columns, (str, float, int)):
            expected_columns = [expected_columns]
        
        if len(expected_columns) != data_columns:
            message = "Number of provided column names does not match data dimensions."
            if error_mode == 'raise':
                raise ValueError(message)
            elif error_mode == 'warn':
                warnings.warn(f"{message} Default columns will be used.", UserWarning)
                return [f'feature_{i}' for i in range(data_columns)]
        return expected_columns

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim == 2:
            columns = validate_columns(data.shape[1], columns)
            df = pd.DataFrame(data, columns=columns)
        else:
            raise ValueError("Array with more than two dimensions is not supported.")
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        raise TypeError(
            "Unsupported data type. Data must be a DataFrame, array, dict, or Series."
            f" Got {type(data).__name__!r}")

    return df

def format_correlations(
    data, 
    min_corr=0.5, 
    high_corr=0.8,
    method='pearson', 
    min_periods=1, 
    use_symbols=False, 
    no_corr_placeholder='...', 
    hide_diag=True, 
    title=None, 
    error_mode='warn', 
    precomputed=False,
    legend_markers=None, 
    autofit=False, 
    ):
    """
    Computes and formats the correlation matrix for a DataFrame's numeric columns, 
    allowing for visual customization and conditional display based on specified
    correlation thresholds.

    Parameters
    ----------
    data : DataFrame
        The input data from which to compute correlations. Must contain at least
        one numeric column.
    min_corr : float, optional
        The minimum correlation coefficient to display explicitly. Correlation
        values below this threshold will be replaced by `no_corr_placeholder`.
        Default is 0.5.
    high_corr : float, optional
        The threshold above which correlations are considered high, which affects
        their representation when `use_symbols` is True. Default is 0.8.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Method of correlation:
        - 'pearson' : standard correlation coefficient
        - 'kendall' : Kendall Tau correlation coefficient
        - 'spearman' : Spearman rank correlation
        Default is 'pearson'.
    min_periods : int, optional
        Minimum number of observations required per pair of columns to have a 
        valid result. Default is 1.
    use_symbols : bool, optional
        If True, uses symbolic representation ('++', '--', '-+') for correlation
        values instead of numeric. Default is False.
    no_corr_placeholder : str, optional
        Text to display for correlation values below `min_corr`. Default is '...'.
    hide_diag : bool, optional
        If True, the diagonal elements of the correlation matrix (always 1) are
        not displayed. Default is True.
    title : str, optional
        An optional title for the formatted correlation matrix. If provided, it
        is centered above the matrix. Default is None.
    error_mode : str, optional
        Determines how to handle errors related to data validation: 'warn' (default),
        'raise', or 'ignore'. This affects behavior when the DataFrame has insufficient
        data or non-numeric columns.
    precomputed: bool, optional 
       Consider data as already correlated data. No need to recomputed the
       the correlation. Default is 'False'
    legend_markers: str, optional 
       A dictionary mapping correlation symbols to their descriptions. If provided,
       it overrides the default markers. Default is None.
    autofit : bool, optional
        If True, adjusts the column widths and the number of visible rows
        based on the DataFrame's content and available display size. Default
        is False.
       
    Returns
    -------
    str
        A formatted string representation of the correlation matrix that includes
        any specified title, the matrix itself, and potentially a legend if
        `use_symbols` is enabled.

    Notes
    -----
    The function relies on pandas for data manipulation and correlation computation. 
    It customizes the display of the correlation matrix based on user preferences 
    for minimum correlation, high correlation, and whether to use symbolic 
    representations.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import format_correlations
    >>> data = pd.DataFrame({
    ...     'A': np.random.randn(100),
    ...     'B': np.random.randn(100),
    ...     'C': np.random.randn(100) * 10
    ... })
    >>> print(format_correlations(data, min_corr=0.3, high_corr=0.7,
    ...                            use_symbols=True, title="Correlation Matrix"))
    Correlation Matrix
    ==================
          A    B    C 
      ----------------
    A |       ...  ...
    B |  ...       ...
    C |  ...  ...     
    ==================

    ..................
    Legend : ...:
             Non-correlate
             d++: Strong
             positive,
             --: Strong
             negative,
             +-: Moderate
    ..................
    """

    title = str(title or '').title()
    df = validate_data(data)
    if len(df.columns) == 1:
        if error_mode == 'warn':
            warnings.warn("Cannot compute correlations for a single column.")
        elif error_mode == 'raise':
            raise ValueError("Cannot compute correlations for a single column.")
        return '' if error_mode == 'ignore' else 'No correlations to display.'
    
    if precomputed: 
        corr_matrix= data.copy() 
    else:     
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            if error_mode == 'warn':
                warnings.warn("No numeric data found in the DataFrame.")
            elif error_mode == 'raise':
                raise ValueError("No numeric data found in the DataFrame.")
                # Return an empty string if no numeric data
            return '' if error_mode == 'ignore' else 'No numeric data available.'  
    
        corr_matrix = numeric_df.corr(method=method, min_periods= min_periods )

    if hide_diag:
        np.fill_diagonal(corr_matrix.values, np.nan)  # Set diagonal to NaN

    def format_value(value):
        if pd.isna(value):  # Handle NaN for diagonals
            return '' if hide_diag else ( 'o' if use_symbols else pd.isna(value))
        if abs(value) < min_corr:
            return str(no_corr_placeholder).ljust(4) if use_symbols else f"{value:.4f}"
        if use_symbols:
            if value >= high_corr:
                return '++'.ljust(4)
            elif value <= -high_corr:
                return '--'.ljust(4)
            else:
                return '-+'.ljust(4)
        else:
            return f"{value:.4f}"
    
    if autofit: 
        # remove ... to avoid confusion with no correlated symbol 
        no_corr_placeholder=''
        
    formatted_corr = corr_matrix.applymap(format_value)
    formatted_df = format_df(formatted_corr, autofit= autofit)
    max_width = find_maximum_table_width(formatted_df)
    legend = ""
    if use_symbols:
        legend = generate_legend(
            legend_markers, no_corr_placeholder, hide_diag,  max_width)
    if title:
        title = title.center(max_width) + "\n"

    return title + formatted_df + legend

def generate_legend(
    custom_markers=None, 
    no_corr_placeholder='...', 
    hide_diag=True,
    max_width=50, 
    add_frame_lines=True, 
    border_line='.'
    ):
    """
    Generates a legend for a table (dataframe) matrix visualization, formatted 
    according to specified parameters.

    This function supports both numeric and symbolic representations of table
    values. Symbolic representations, which are used primarily for visual clarity,
    include the following symbols:

    - ``'++'``: Represents a strong positive relationship.
    - ``'--'``: Represents a strong negative relationship.
    - ``'-+'``: Represents a moderate relationship.
    - ``'o'``: Used exclusively for diagonal elements, typically representing
      a perfect relationship in correlation matrices (value of 1.0).
         
    Parameters
    ----------
    custom_markers : dict, optional
        A dictionary mapping table symbols to their descriptions. If provided,
        it overrides the default markers. Default is None.
    no_corr_placeholder : str, optional
        Placeholder text for table values that do not meet the minimum threshold.
        Default is '...'.
    hide_diag : bool, optional
        If True, omits the diagonal entries from the legend. These are typically
        frame of a variable with itself (1.0). Default is True.
    max_width : int, optional
        The maximum width of the formatted legend text, influences the centering
        of the title. Default is 50.
    add_frame_lines : bool, optional
        If True, adds a frame around the legend using the specified `border_line`.
        Default is True.
    border_line : str, optional
        The character used to create the border of the frame if `add_frame_lines`
        is True. Default is '.'.

    Returns
    -------
    str
        The formatted legend text, potentially framed, centered according to the
        specified width, and including custom or default descriptions of correlation
        values.

    Examples
    --------
    >>> from gofast.api.util import generate_legend
    >>> custom_markers = {"++": "High Positive", "--": "High Negative"}
    >>> print(generate_legend(custom_markers=custom_markers, max_width=60))
    ............................................................
    Legend : ...: Non-correlated, ++: High Positive, --: High
             Negative, -+: Moderate
    ............................................................

    >>> print(generate_legend(hide_diag=False, max_width=70))
    ......................................................................
    Legend : ...: Non-correlated, ++: Strong positive, --: Strong negative,
             -+: Moderate, o: Diagonal
    ......................................................................
    >>> custom_markers = {"++": "Highly positive", "--": "Highly negative"}
    >>> legend = generate_legend(custom_markers=custom_markers,
    ...                          no_corr_placeholder='N/A', hide_diag=False,
    ...                          border_line ='=')

    >>> print(legend) 

    ==================================================
    Legend : N/A: Non-correlated, ++: Highly positive,
             --: Highly negative, -+: Moderate, o:
             Diagonal
    ==================================================
    """
    # Default markers and their descriptions
    default_markers = {
        no_corr_placeholder: "Non-correlated",
        "++": "Strong positive",
        "--": "Strong negative",
        "-+": "Moderate",
        "o": "Diagonal"  # only used if hide_diag is False
    }
    if ( custom_markers is not None 
        and not isinstance(custom_markers, dict)
    ):
        raise TypeError("The 'custom_markers' parameter must be a dictionary."
                        " Received type: {0}. Please provide a dictionary"
                        " where keys are the legend symbols and values"
                        " are their descriptions.".format(
                        type(custom_markers).__name__))

    # Update default markers with any custom markers provided
    markers = {**default_markers, **(custom_markers or {})}
    # If no correlation placeholder, then remove it from the markers.
    if not no_corr_placeholder: 
        markers.pop (no_corr_placeholder)
    # Create legend entries
    legend_entries = [f"{key}: {value}" for key, value in markers.items() if not (
        key == 'o' and hide_diag)]

    # Join entries with commas and format the legend text
    legend_text = ", ".join(legend_entries)
    legend = "\n\n" + format_text(
        legend_text, 
        key='Legend', 
        key_length=len('Legend'), 
        max_char_text=max_width + len('Legend'), 
        add_frame_lines=add_frame_lines,
        border_line=border_line
        )
    return legend

def to_snake_case(name):
    """
    Converts a string to snake_case using regex.

    Parameters
    ----------
    name : str
        The string to convert to snake_case.

    Returns
    -------
    str
        The snake_case version of the input string.
    """
    name = str(name)
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()  # CamelCase to snake_case
    name = re.sub(r'\W+', '_', name)  # Replace non-word characters with '_'
    name = re.sub(r'_+', '_', name)  # Replace multiple '_' with single '_'
    return name.strip('_')

def generate_column_name_mapping(columns):
    """
    Generates a mapping from snake_case column names to their original names.

    Parameters
    ----------
    columns : List[str]
        The list of column names to convert and map.

    Returns
    -------
    dict
        A dictionary mapping snake_case column names to their original names.
    """
    return {to_snake_case(col): col for col in columns}

def series_to_dataframe(series):
    """
    Transforms a pandas Series into a DataFrame where the columns are the index
    of the Series. If the Series' index is numeric, the index values are converted
    to strings and used as column names.

    Parameters
    ----------
    series : pandas.Series
        The Series to be transformed into a DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each column represents a value from the Series,
        with column names corresponding to the Series' index values.
        
    Raises
    ------
    TypeError
        If the input is not a pandas Series.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.util import series_to_dataframe
    >>> series = pd.Series(data=[1, 2, 3], index=['a', 'b', 'c'])
    >>> df = series_to_dataframe(series)
    >>> print(df)
       a  b  c
    0  1  2  3
    
    >>> series_numeric_index = pd.Series(data=[4, 5, 6], index=[10, 20, 30])
    >>> df_numeric = series_to_dataframe(series_numeric_index)
    >>> print(df_numeric)
      10 20 30
    0  4  5  6
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    # Convert index to string if it's numeric
    if series.index.dtype.kind in 'iufc':  # Checks for int, unsigned int, float, complex
        index_as_str = series.index.astype(str)
    else:
        index_as_str = series.index

    # Create a DataFrame with a single row populated with the Series' values
    # and columns named after the Series' index.
    df = pd.DataFrame([series.values], columns=index_as_str)
    
    return df  

def get_table_size(width="auto", error="warn", return_height=False):
    """
    Determines the appropriate width (and optionally height) for table display
    based on terminal size, with options for manual width adjustment.

    Parameters
    ----------
    width : int or str, optional
        The desired width for the table. If set to 'auto', the terminal width
        is used. If an integer is provided, it will be used as the width, 
        default is 'auto'.
    error : str, optional
        Error handling strategy when specified width exceeds terminal 
        width: 'warn' or 'ignore'.
        Default is 'warn'.
    return_height : bool, optional
        If True, the function also returns the height of the table. 
        Default is False.

    Returns
    -------
    int or tuple
        The width of the table as an integer, or a tuple of (width, height) 
        if return_height is True.

    Examples
    --------
    >>> table_width = get_table_size()
    >>> print("Table width:", table_width)
    >>> table_width, table_height = get_table_size(return_height=True)
    >>> print("Table width:", table_width, "Table height:", table_height)
    """
    auto_width, auto_height = get_terminal_size()
    if width == "auto":
        width = auto_width
    else:
        try:
            width = int(width)
            if width > auto_width:
                if error == "warn":
                    warnings.warn(
                        f"Specified width {width} exceeds terminal width {auto_width}. "
                        "This may cause display issues."
                    )
        except ValueError:
            raise ValueError(
                "Width must be 'auto' or an integer; got {type(width).__name__!r}")

    if return_height:
        return (width, auto_height)
    return width

def get_terminal_size():
    """
    Retrieves the current terminal size (width and height) to help dynamically 
    set the maximum width for displaying data columns.

    Returns
    -------
    tuple
        A tuple containing two integers:
        - The width of the terminal in characters.
        - The height of the terminal in lines.

    Examples
    --------
    >>> from gofast.api.util import get_terminal_size
    >>> terminal_width, terminal_height = get_terminal_size()
    >>> print("Terminal Width:", terminal_width)
    >>> print("Terminal Height:", terminal_height)
    """
    # Use shutil.get_terminal_size if available (Python 3.3+)
    # This provides a fallback of (80, 24) which is a common default size
    if hasattr(shutil, "get_terminal_size"):
        size = shutil.get_terminal_size(fallback=(80, 24))
    else:
        # Fallback for Python versions before 3.3
        try:
            # UNIX-based systems
            size = os.popen('stty size', 'r').read().split()
            return int(size[1]), int(size[0])
        except Exception:
            # Default fallback size
            size = (80, 24)
    return size.columns, size.lines

def optimize_col_width (max_cols=4, df=None, min_col_width=10):
    """
    Determines the optimal width for each column based on the terminal size to
    ensure that data is displayed properly without exceeding the terminal width.
    If necessary, reduces the number of columns displayed to maintain a minimum
    column width.

    Parameters
    ----------
    max_cols : int
        The desired maximum number of columns to display. This value may be
        adjusted downward to prevent column widths from falling below the
        minimum width.
    df : pandas.DataFrame, optional
        A DataFrame for which column width needs adjustment. If provided,
        `max_cols` will be considered as the upper limit, with the actual 
        number displayed possibly fewer to fit the terminal width.
    min_col_width : int, default 10
        The minimum allowable width for any column. It defaults to 10 characters,
        which is a balance between readability and space efficiency.

    Returns
    -------
    int
        The maximum width in characters that each column should have, or the
        minimum column width if space is insufficient.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import optimize_col_width
    >>> df = pd.DataFrame({'A': range(100), 'B': range(100), 'C': range(100)})
    >>> max_width = optimize_col_width(max_cols=3, df=df)
    >>> print("Maximum column width for display:", max_width)
    """
    terminal_width, _ = get_terminal_size()
    
    if df is not None:
        df = validate_data (df)
        if isinstance(min_col_width, str) and min_col_width == "auto":
            min_col_width = min(len(str(col)) for col in df.columns)
        max_cols = min(max_cols, len(df.columns))
    
    if str(min_col_width) == 'auto':
        min_col_width = 10 # In the case df is not provided while "auto" is set
    # Calculate total required width including buffer space between columns
    total_required_width = max_cols * (min_col_width + 4)  # +4 for space between columns

    if total_required_width > terminal_width:
        # Adjust max_cols if the total required width exceeds the terminal width
        max_cols = (terminal_width // (min_col_width + 4))

    # Calculate the maximum width per column
    if max_cols > 0:
        buffer_width = 4 * max_cols
        max_col_width = (terminal_width - buffer_width) // max_cols
        return max(max_col_width, min_col_width)  # Ensure at least min_col_width
    else:
        return min_col_width  # Return min_col_width if no columns fit

def get_displayable_columns(cols_or_df, /, buffer_space=4, min_col_width=10):
    """
    Computes the number of columns that can be displayed in the terminal based
    on the maximum column width, considering a buffer space between columns.

    Parameters
    ----------
    cols_or_df : list, pandas.DataFrame, or str
        A list of column names, a pandas DataFrame from which to extract column
        names, or a single column name as a string. If a DataFrame is provided,
        the column names are extracted. If a string is provided, it is treated
        as a list with a single column name.
    buffer_space : int, default 4
        The number of characters to consider as spacing between columns.
    min_col_width : int, default 10
        The minimum width in characters for each column.

    Returns
    -------
    int
        The number of columns that can be optimally displayed in the terminal
        without exceeding the terminal's width.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.util import get_displayable_columns
    >>> df = pd.DataFrame({'A': range(100), 'B': range(100), 'C': range(100)})
    >>> num_cols = get_displayable_columns(df)
    >>> print(f"Number of displayable columns: {num_cols}")
    """
    # Extract column names based on the type of `cols_or_df`
    if isinstance(cols_or_df, pd.DataFrame):
        cols = cols_or_df.columns.tolist()  # Extract columns from DataFrame
    elif isinstance(cols_or_df, list):
        cols = cols_or_df  # Use the list directly if provided
    elif isinstance(cols_or_df, str):
        cols = [cols_or_df]  # Treat a single string as a list with one element
    else:
        # Raise an error if the input type is unexpected
        raise TypeError("cols_or_df must be either a list, or a pandas.DataFrame.")

    # Determine the number of columns to evaluate
    num_cols = len(cols)
    # Create a temporary DataFrame to facilitate column width calculation
    df = pd.DataFrame(columns=cols)
    # Calculate the optimal column width
    max_col_width = optimize_col_width(max_cols=num_cols, df=df,
                                       min_col_width=min_col_width)
    # Retrieve the current terminal width
    terminal_width, _ = get_terminal_size()

    # Calculate the total space available per column including buffer
    available_space_per_column = max_col_width + buffer_space
    # Determine the maximum number of columns that can fit in the terminal
    max_displayable_cols = terminal_width // available_space_per_column

    return min(num_cols, max_displayable_cols)

def to_camel_case(text, delimiter=None, use_regex=False):
    """
    Converts a given string to CamelCase. The function handles strings with or
    without delimiters, and can optionally use regex for splitting based on
    non-alphanumeric characters.

    Parameters
    ----------
    text : str
        The string to convert to CamelCase.
    delimiter : str, optional
        A character or string used as a delimiter to split the input string. 
        Common delimiters include underscores ('_') or spaces (' '). If None 
        and use_regex is ``False``, the function tries to automatically detect 
        common delimiters like spaces or underscores. 
        If `use_regex` is ``True``, it splits the string at any non-alphabetic 
        character.
    use_regex : bool, optional
        Specifies whether to use regex for splitting the string on non-alphabetic
        characters.
        Defaults to ``False``.

    Returns
    -------
    str
        The CamelCase version of the input string.

    Examples
    --------
    >>> from gofast.api.util import to_camel_case
    >>> to_camel_case('outlier_results', '_')
    'OutlierResults'

    >>> to_camel_case('outlier results', ' ')
    'OutlierResults'

    >>> to_camel_case('outlierresults')
    'Outlierresults'

    >>> to_camel_case('data science rocks')
    'DataScienceRocks'

    >>> to_camel_case('data_science_rocks')
    'DataScienceRocks'

    >>> to_camel_case('multi@var_analysis', use_regex=True)
    'MultiVarAnalysis'
    
    >>> to_camel_case('OutlierResults')
    'OutlierResults'

    >>> to_camel_case('BoxFormatter')
    'BoxFormatter'

    >>> to_camel_case('MultiFrameFormatter')
    'MultiFrameFormatter'
    """
    # Remove any leading/trailing whitespace
    text = str(text).strip()

    # Check if text is already in CamelCase and return it as is
    if text and text[0].isupper() and text[1:].islower() == False:
        return text

    if use_regex:
        # Split text using any non-alphabetic character as a delimiter
        words = re.split('[^a-zA-Z]', text)
    elif delimiter is None:
        if ' ' in text and '_' in text:
            # Both space and underscore are present, replace '_' with ' ' then split
            text = text.replace('_', ' ')
            words = text.split()
        elif ' ' in text:
            words = text.split(' ')
        elif '_' in text:
            words = text.split('_')
        else:
            # No common delimiter found, handle as a single word
            words = [text]
    else:
        # Use the specified delimiter
        words = text.split(delimiter)

    # Capitalize the first letter of each word and join them without spaces
       # Ensure empty strings from split are ignored
    return ''.join(word.capitalize() for word in words if word)  

def check_index_column_types(df):
    """
    Checks if the data types of the index and columns of a DataFrame 
    are the same.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check.

    Returns
    -------
    bool
        True if the index and columns have the same data type, False otherwise.

    Examples
    --------
    >>> from gofast.api.util import check_index_column_types
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... }, index=[1, 2, 3])
    >>> print(check_index_column_types(data))
    False

    >>> data = pd.DataFrame({
    ...     1: [1, 2, 3],
    ...     2: [4, 5, 6],
    ...     3: [7, 8, 9]
    ... }, index=['one', 'two', 'three'])
    >>> print(check_index_column_types(data))
    False
    
    >>> data = pd.DataFrame({
    ...     "model_A": [1, 2, 3],
    ...     "model_B": [4, 5, 6],
    ...     "model_C": [7, 8, 9]
    ... }, index=["model_A", "model_B", "model_C"])
    >>> print(check_index_column_types(data))
    True
    """
    return df.index.dtype == df.columns.dtype

def beautify_dict(d, space=4, key=None, max_char=None):
    """
    Format a dictionary with custom indentation and aligned keys and values.
    
    Parameters:
    ----------
    d : dict
        The dictionary to format.
    space : int, optional
        The number of spaces to indent the dictionary entries.
    key : str, optional
        An optional key to nest the dictionary under.
    max_char : int, optional
        Maximum characters a value can have before being truncated.
    
    Returns:
    -------
    str
        A string representation of the dictionary with custom formatting.

    Examples:
    --------
    >>> from gofast.api.util import beautify_dict
    >>> dictionary = {
    ...     3: 'Home & Garden',
    ...     2: 'Health & Beauty',
    ...     4: 'Sports',
    ...     0: 'Electronics',
    ...     1: 'Fashion'
    ... }
    >>> print(beautify_dict(dictionary, space=4))
    """
    
    if not isinstance(d, dict):
        raise TypeError("Expected input to be a 'dict',"
                        f" received '{type(d).__name__}' instead.")
    
    if max_char is None: 
        # get it automatically 
        max_char, _ =get_terminal_size()
    # Determine the longest key for alignment
    if len(d)==0: 
        max_key_length=0 
    else:
        max_key_length = max(len(str(k)) for k in d.keys())
    
    # Create a list of formatted rows
    formatted_rows = []
    for dkey, value in sorted(d.items()):
        value_str = str(value)
        if max_char is not None and len(value_str) > max_char:
            value_str = value_str[:max_char] + "..."
        # Ensure all keys are right-aligned to the longest key length
        formatted_row = f"{str(dkey):>{max_key_length}}: '{value_str}'"
        formatted_rows.append(formatted_row)

    # Join all rows into a single string with custom indentation
    indent = ' ' * space
    inner_join = ',\n' + indent
    formatted_dict = '{\n' + indent + inner_join.join(formatted_rows) + '\n}'

    if key:
        # Prepare outer key indentation and format
           # Slightly less than the main indent
        outer_indent = ' ' * (space - 2 + len(key) + max_key_length)  
        # Construct a new header with the key
        formatted_dict = f"{key} : {formatted_dict}"
        # Split lines and indent properly to align with the key
        lines = formatted_dict.split('\n')
        for i in range(1, len(lines)):
            lines[i] = outer_indent + lines[i]
            if max_char is not None and len(lines[i]) > max_char:
                lines[i] = lines[i][:max_char] + "..."
        # format lins -1 
        #lines [-1] = outer_indent + lines [-1]
        formatted_dict = '\n'.join(lines)
        
    return formatted_dict

def remove_extra_spaces(text):
    """
    Removes extra spaces from the input text, leaving only one 
    space between words.

    Parameters
    ----------
    text : str
        The string from which extra spaces need to be removed.

    Returns
    -------
    str
        The string with only one space between each word.

    Example
    -------
    >>> from gofast.api.util import remove_extra_spaces
    >>> text = "this is      text that    have   extra          space."
    >>> remove_extra_spaces(text)
    'this is text that have extra space.'
    """
    # Use regular expression to replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def format_iterable(attr):
    """
    Formats an iterable with a string representation that includes
    statistical or structural information depending on the iterable's type.
    """
    def _numeric_stats(iterable):
        return {
            'min': round(np.min(iterable), 4),
            'max': round(np.max(iterable), 4),
            'mean': round(np.mean(iterable), 4),
            'len': len(iterable)
        }
    
    def _format_numeric_iterable(iterable):
        stats = _numeric_stats(iterable)
        return ( 
            f"{type(iterable).__name__} (min={stats['min']},"
            f" max={stats['max']}, mean={stats['mean']}, len={stats['len']})"
            )

    def _format_ndarray(array):
        stats = _numeric_stats(array.flat) if np.issubdtype(array.dtype, np.number) else {}
        details = ", ".join([f"{key}={value}" for key, value in stats.items()])
        return f"ndarray ({details}, shape={array.shape}, dtype={array.dtype})"
    
    def _format_pandas_object(obj):
        if isinstance(obj, pd.Series):
            stats = _numeric_stats(obj) if obj.dtype != 'object' else {}
            details = ", ".join([f"{key}={value}" for key, value in stats.items()])
            if details: 
                details +=', '
            return f"Series ({details}len={obj.size}, dtype={obj.dtype})"
        elif isinstance(obj, pd.DataFrame):
            numeric_cols = obj.select_dtypes(include=np.number).columns
            stats = _numeric_stats(obj[numeric_cols].values.flat) if not numeric_cols.empty else {}
            details = ", ".join([f"{key}={value}" for key, value in stats.items()])
            if details: 
                details +=', '
            return ( 
                f"DataFrame ({details}n_rows={obj.shape[0]},"
                f" n_cols={obj.shape[1]}, dtypes={obj.dtypes.unique()})"
                )
    
    if isinstance(attr, (list, tuple, set)) and all(
            isinstance(item, (int, float)) for item in attr):
        return _format_numeric_iterable(attr)
    elif isinstance(attr, np.ndarray):
        return _format_ndarray(attr)
    elif isinstance(attr, (pd.Series, pd.DataFrame)):
        return _format_pandas_object(attr)
    
    return str(attr)

def format_dict_result(
    dictionary, dict_name='Container', 
    max_char=50, 
    include_message=False):
    """
    Formats a dictionary into a string with specified formatting rules.

    Parameters
    ----------
    dictionary : dict
        The dictionary to format.
    dict_name : str, optional
        The name of the dictionary, by default 'Container'.
    max_char : int, optional
        The maximum number of characters for each value before truncating, 
        by default 50.
    include_message : bool, optional
        Whether to include a remainder message at the end, by default False.

    Returns
    -------
    str
        The formatted string representation of the dictionary.

    Examples
    --------
    >>> example_dict = {
    ...     'key1': 'short value',
    ...     'key2': 'a much longer value that should be truncated for readability purposes',
    ...     'key3': 'another short value',
    ...     'key4': 'value'
    ... }
    >>> print(format_dict_result(example_dict, dict_name='ExampleDict', max_char=30))
    ExampleDict({
        key1: short value,
        key2: a much longer value that s...,
        key3: another short value,
        key4: value,
    })

    Notes
    -----
    The function calculates the required indentation based on the length of the 
    dictionary name and the maximum key length. If a value exceeds the specified 
    maximum length, it truncates the value and appends an ellipsis ("...").
    """
    max_key_length = max(len(str(key)) for key in dictionary.keys())
    formatted_lines = [f"{dict_name}({{"]
    
    for key, value in dictionary.items():
        if (
                isinstance(value, value.__class__)
                and not hasattr(value, '__array__')
                and not isinstance(value, (str, list, tuple))
        ):
            try:
                formatted_value = value.__class__.__name__
            except:
                formatted_value = value.__name__
        else:
            formatted_value = format_iterable(value)
            
        if len(formatted_value) > max_char:
            formatted_value = formatted_value[:max_char - 3] + "..."
        formatted_lines.append(
            f"{' ' * (len(dict_name) + 2)}{key:{max_key_length}}: {formatted_value},")
    
    formatted_lines.append(" " * (len(dict_name) + 1) + "})")
    
    remainder = f"[Use <{dict_name}.key> to get the full value ...]"  
    
    return ( "\n".join(formatted_lines) + f"\n\n{remainder}" 
            if include_message else "\n".join(formatted_lines)
            )

if __name__=='__main__': 
    # Example usage:
    data = {
        'col0': [1, 2, 3, 4],
        'col1': [4, 3, 2, 1],
        'col2': [10, 20, 30, 40],
        'col3': [40, 30, 20, 10],
        'col4': [5, 6, 7, 8]
    }
    df = pd.DataFrame(data)

    # Calling the function
    result = format_correlations(df, 0.8, 0.9, False, hide_diag= True)
    print(result)

    # Example usage
    data = {
        'col0': [1, 2, 3, 4],
        'col1': [4, 3, 2, 1],
        'col2': [10, 20, 30, 40],
        'col3': [40, 30, 20, 10],
        'col4': [5, 6, 7, 8]
    }

    
