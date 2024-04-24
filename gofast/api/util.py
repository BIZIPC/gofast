# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import re
import warnings
import numpy as np 
import pandas as pd
from shutil import get_terminal_size


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
    
def extract_truncate_df(
        df, include_truncate=False, max_rows= 100, max_cols= 7, 
        return_indices_cols=False ):
    """
    Extracts indices from a dataframe string representation and decides 
    whether to include truncated indices.
    
    Parameters:
        df_string (str): String representation of the dataframe.
        include_truncate (bool): If True, includes all possible indices 
        assuming continuous range where truncated.

    Returns:
        list: A list of indices extracted from the dataframe string.
    """
    # check indices whether it is numeric, if Numeric keepit otherwise reset it 
    name_indexes = [] 
    if not is_numeric_index( df):
        #print('True')
        name_indexes = df.index.tolist() 
        df.reset_index (drop=True, inplace =True)
    columns = df.columns.tolist() 
    
    df_string= df.to_string(
        index =True, header=True, max_rows=max_rows, max_cols=max_cols )
    # Find all index occurrences at the start of each line
    indices = re.findall(r'^\s*(\d+)', df_string, flags=re.MULTILINE)
    lines = df_string.split('\n')
    #print(indices)
   
    # Convert found indices to integers
    indices = list(map(int, indices))
    if include_truncate and indices:
        # Assume the indices are continuous and fill in the missing range
        indices = list(range(indices[0], indices[-1] + 1))
    
    if name_indexes: 
        indices = [ name_indexes [i] for i in indices]
        
    # extract _columns from df_strings 
    # columns is expected to be the first 0 
    header_parts = lines[0].strip() 
    columns = [ col for col in columns if col in header_parts ]
    subset_df = get_dataframe_subset(df, indices, columns )
    
    if return_indices_cols: 
        return indices, columns, subset_df
    
    return subset_df 

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
        df = df.iloc[indices]

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
    style="auto"
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
    auto_rows, auto_cols = auto_adjust_dataframe_display(
        df, index=index, header=header
        )
    # Example usage within the function:
    max_rows = _adjust_value(max_rows, auto_rows)
    max_cols = _adjust_value(max_cols, auto_cols)
    
    # Apply float formatting to the DataFrame
    if output_format == 'html': 
        # Use render for HTML output
        formatted_df = df.style.format(float_format).render()  
    else:
        df= make_format_df(df, "%%", apply_to_column= True)
        # Convert DataFrame to string with the specified format options
        formatted_df = df.to_string(
            index=index, header=header, max_rows=max_rows, max_cols=max_cols, 
            float_format=lambda x: float_format.format(x) if isinstance(
                x, (float, np.float64)) else x
    )

    style= select_df_styles(style, df )
    if style =='advanced': 
        formatted_output = df_advanced_style(
            formatted_df, table_width, 
            title= title,
            index= index, 
            header= header, 
            header_line= header_line, 
            sub_line=sub_line, 
            df=df, 
          )
    else: 
        formatted_output=df_base_style(
            formatted_df, title=title, 
            table_width=table_width, 
            header_line= header_line, 
            sub_line= sub_line , 
            df=df 
            )
    # Remove the whitespace_sub %% 
    formatted_output = formatted_output.replace ("%%", '  ')
    return formatted_output

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
    terminal_size = get_terminal_size()
    terminal_cols = terminal_size.columns
    terminal_rows = terminal_size.lines
    if max_rows == "auto": 
        max_rows = terminal_rows 
        # to terminal row sizes 
    if max_cols =="auto":
        # compared to terminal size. 
        max_cols = terminal_cols 
    if return_rows_cols_size: 
        auto_rows, auto_cols = auto_adjust_dataframe_display( df)
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
    ):
    """
    Formats a given DataFrame string into a styled textual representation 
    with headers, sub-headers,and appropriate line separations.

    Parameters
    ----------
    formatted_df : str
        The string representation of the DataFrame to be formatted.
    title : str, optional
        The title to be displayed at the top of the formatted output. 
        If None, no title is displayed.
    table_width : int or str, optional
        The overall width of the table. If set to 'auto', the width will 
        adapt based on the content and terminal size, but not exceeding the 
        terminal width.
    header_line : str, optional
        The character used to create the top and bottom border of the table.
    sub_line : str, optional
        The character used to create the line below the header row of the table.
    df : DataFrame, optional
        A pandas DataFrame from which column widths can be calculated directly, 
        used as a fallback if complex column names are detected in the 
        'lines' input.
    Returns
    -------
    str
        The fully formatted table as a single string, ready for display.

    Examples
    --------
    >>> from gofast.api.util import df_base_style
    >>> formatted_df = "index    A    B    C\\n0       1    2    3\\n1       4    5    6"
    >>> print(df_base_style(formatted_df, title="Demo Table"))
    Demo Table
    ====================
    index    A    B    C
    --------------------
    0       1    2    3
    1       4    5    6
    ====================
    """
    # Determine the maximum line width in the formatted DataFrame
    max_line_width = max(len(line) for line in formatted_df.split('\n')) 

    # Determine the terminal size for table width if auto
    if table_width == 'auto':
        terminal_width = get_terminal_size().columns
        # make all possible to not exceed the terminal width 
        if max_line_width < terminal_width: 
            table_width= max_line_width
        else : 
            table_width = max(terminal_width, max_line_width)

    # Formatting title and separators
    header = f"{title}".center(table_width) if title else ""
    header_separator = header_line * table_width
    sub_separator = sub_line * table_width

    # Split the formatted DataFrame to insert the sub_separator after the headers
    lines = formatted_df.split('\n')

    formatted_output = f"{header}\n{header_separator}\n{lines[0]}\n{sub_separator}\n"\
        + "\n".join(lines[1:]) + f"\n{header_separator}"

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
    index=True, header=True,
    title=None, 
    header_line="=", 
    sub_line="-", 
    df=None, 
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

    lines = formatted_df.split('\n')
    new_lines = []

    max_index_length, *column_widths = calculate_column_widths(
        lines, include_index= index, include_column_width= True,
        df = df, max_text_length= 50 )
    # max_index_length = max(len(line.split()[0]) for line in lines[1:]) + 3  
    max_index_length +=3  # for extra spaces 
    for i, line in enumerate(lines):
        if i == 0 and header:  # Adjust header line to include vertical bar
            header_parts = line.split()
            header_line_formatted = " " * max_index_length + "  ".join(
                header_part.rjust(
                    column_widths[idx]) for idx, header_part in enumerate(header_parts))
            new_lines.append( " " + header_line_formatted)
            continue 
        elif i == 1 and header:  # Insert sub-line after headers
            new_lines.append(" " * (max_index_length-1) + sub_line * (
                len(header_line_formatted) - max_index_length + 2 ))
            continue

        parts = line.split(maxsplit=1)
        if len(parts) > 1:
            index_part, data_part = parts
            data_part.split()
            formatted_data_part = "  ".join(
                data.rjust(column_widths[idx]) for idx, data in enumerate(data_part.split()))
            new_line = f"{index_part.ljust(max_index_length - 2)} | {formatted_data_part}"

        else:
            new_line = " " * (max_index_length - 2) + line
            
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

    # Get terminal size
    terminal_size = get_terminal_size()
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
    available_width = screen_width - index_width - 3  # Adjust for spacing between columns

    # Estimate the number of columns that can fit
    max_cols = available_width // avg_entry_width
    
    # Adjust for header if present
    header_height = 1 if header else 0
    
    # Calculate the number of rows that can fit
    max_rows = screen_height - header_height - 3  # Subtract for header and to avoid clutter

    # Ensure max_cols does not exceed number of DataFrame columns
    max_cols = min(max_cols, len(df.columns))

    # Ensure max_rows does not exceed number of DataFrame rows
    max_rows = min(max_rows, len(df))

    return max_rows, max_cols


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

def calculate_widths(df, max_text_length):
    """
    Calculates the maximum widths for each column based on the content.

    Parameters:
    df (pandas.DataFrame): The DataFrame to calculate widths for.
    max_text_length (int): The maximum allowed length for any cell content.

    Returns:
    tuple: A dictionary with maximum column widths and the maximum width of the index.
    """
    formatted_cells = df.applymap(lambda x: str(x)[:max_text_length] + '...' if len(
        str(x)) > max_text_length else str(x))
    max_col_widths = {col: max(len(col), max(len(x) for x in formatted_cells[col]))
                      for col in df.columns}
    max_index_width = max(len(str(index)) for index in df.index)
    max_col_widths = {col: min(width, max_text_length) for col, width in max_col_widths.items()}
    return max_col_widths, max_index_width

def format_df(df, max_text_length=50, title=None):
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
    
    Here, the table respects a `max_text_length` of 10, ensuring that all
    cell contents do not exceed this length, and the output is well-aligned
    for easy reading.
    """
    title = str(title or '').title()
    
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
            "Unsupported data type. Data must be a DataFrame, array, dict, or Series.")

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
    legend_markers=None
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

    formatted_corr = corr_matrix.applymap(format_value)
    formatted_df = format_df(formatted_corr)
    
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

    
