# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Pertains to specific transformations and formatting of data."""

from __future__ import annotations, print_function 
import warnings 
import numpy as np 
import pandas as pd 

from ..api.types import List, DataFrame, Optional, Dict
from ..api.types import Union,TypeGuard, Tuple, Any
from ..decorators import isdf
from ..tools.coreutils import to_numeric_dtypes
from ..tools.validator import  build_data_if,  parameter_validator  
from ..tools.validator import validate_dtype_selector


__all__= [
    "format_long_column_names", 
    "summarize_text_columns", 
    "sanitize", 
    "split_data", 
    "smart_group", 
    "group_and_filter", 
    ]

def summarize_text_columns(
    data: DataFrame, /, 
    text_columns: List[str], 
    stop_words: str = 'english', 
    encode: bool = False, 
    drop_original: bool = False, 
    compression_method: Optional[str] = None,
    force: bool=False, 
    ) -> TypeGuard[DataFrame]:
    """
    Applies extractive summarization to specified text columns in a pandas
    DataFrame. 
    
    Each text entry in the specified columns is summarized to its most 
    representative sentence based on TF-IDF scores, considering the provided 
    stop words. The DataFrame is then updated with these summaries. If the 
    text entry is too short for summarization, it remains unchanged. Optionally,
    encodes and compresses the summaries.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the text data to be summarized.
    column_names : List[str]
        The list of column names in the DataFrame that contains the text to be 
        summarized.
    stop_words : str or list, optional
        Either a string denoting the language for which to use the pre-built 
        stop words list ('english', 'french', etc.), or a list of stop words 
        to use for filtering during the TF-IDF vectorization (default is 'english').
    encode : bool, default False
        If True, adds a new column for each text column containing the TF-IDF 
        encoding of the summary.
    drop_original : bool, default False
        If True, drops the original text columns after summarization and 
        encoding.
    compression_method : str, optional
       Method to compress the encoded vector into a single numeric value.
       Options include 'sum', 'mean', 'norm'. If None, the full vector is returned.
    force: bool, default=False 
       Construct a temporary dataFrame if the numpy array is passed instead.
    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified columns' text summarized and 
        optionally encoded. Original text columns are dropped if drop_original 
        is True.

    Raises
    ------
    ValueError
        If any of the specified column_names do not exist in the DataFrame.
    TypeError
        If the input df is not a pandas DataFrame or if any of the specified 
        columns are not of type str.

    Examples
    --------
    >>> from gofast.dataops.transformation import summarize_text_columns
    >>> data = {
    ...    'id': [1, 2],
    ...    'column1': [
    ...        "Sentence one. Sentence two. Sentence three.",
    ...        "Another sentence one. Another sentence two. Another sentence three."
    ...    ],
    ...    'column2': [
    ...        "More text here. Even more text here.",
    ...        "Second example here. Another example here."
    ...    ]
    ... }
    >>> df = pd.DataFrame(data)
    >>> summarized_df = summarize_text_columns(df, ['column1', 'column2'], 
    ...                stop_words='english', encode=True, drop_original=True, 
    ...                compression_method='mean')
    >>> print(summarized_df.columns)
        id  column1_encoded  column2_encoded
     0   1              1.0         1.000000
     1   2              1.0         0.697271
     Column 'column1' does not exist in the DataFrame.
    >>> # Make sure the column name is exactly "column1"
    >>> if "column1" in df.columns:
    ...     summarized_df = summarize_text_columns(
    ...      df, ['column1'], stop_words='english', encode=True, drop_original=True)
    ... else:
    ...    print("Column 'column1' does not exist in the DataFrame.")
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def _summarize_and_encode(text):
        sentences = text.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence]
        if len(sentences) <= 1:
            return text, None
    
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # The similarity matrix is square, so we need to avoid comparing a 
        # sentence to itself
        np.fill_diagonal(similarity_matrix, 0)
        # Sum the similarities of each sentence to all others
        sentence_scores = similarity_matrix.sum(axis=1)
        # Identify the index of the most important sentence
        most_important_sentence_index = np.argmax(sentence_scores)
        summary = sentences[most_important_sentence_index]
        encoding = tfidf_matrix[most_important_sentence_index].todense() if encode else None
        if encoding is not None and compression_method:
            if compression_method == 'sum':
                encoding = np.sum(encoding)
            elif compression_method == 'mean':
                encoding = np.mean(encoding)
            elif compression_method == 'norm':
                encoding = np.linalg.norm(encoding)
            else: 
                raise ValueError(
                    f"Unsupported compression method: {compression_method}")
        return summary, encoding
        
    def _summarize_and_encode0(text):
        sentences = text.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence]
        if len(sentences) <= 1:
            return text, None
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix).flatten()
        most_important_sentence_index = np.argmax(similarity_matrix[:-1])
        summary = sentences[most_important_sentence_index]
        encoding = tfidf_matrix[most_important_sentence_index].todense() if encode else None

        if encoding is not None and compression_method:
            if compression_method == 'sum':
                encoding = np.sum(encoding)
            elif compression_method == 'mean':
                encoding = np.mean(encoding)
            elif compression_method == 'norm':
                encoding = np.linalg.norm(encoding)
            else: raise ValueError(
                f"Unsupported compression method: {compression_method}")
        return summary, encoding

    if not isinstance(data, pd.DataFrame):
        if force: 
            data = build_data_if ( data, force=force, to_frame=True, 
                                  input_name="sum_col_", raise_warning="mute")
        else:
            raise TypeError("The input data must be a pandas DataFrame."
                            " Or set force to 'True' to build a temporary"
                            " dataFrame.")
    
    for column_name in text_columns:
        if column_name not in data.columns:
            raise ValueError(f"The column {column_name} does not exist in the DataFrame.")
        if not pd.api.types.is_string_dtype(data[column_name]):
            raise TypeError(f"The column {column_name} must be of type str.")

        summarized_and_encoded = data[column_name].apply(
            lambda text: _summarize_and_encode(text) if isinstance(
                text, str) and text else (text, None))
        data[column_name] = summarized_and_encoded.apply(lambda x: x[0])
        
        if encode:
            encoded_column_name = f"{column_name}_encoded"
            data[encoded_column_name] = summarized_and_encoded.apply(
                lambda x: x[1] if x[1] is not None else None)

    if drop_original:
        data.drop(columns=text_columns, inplace=True)
        data.columns = [ c.replace ("_encoded", '') for c in data.columns]

    return data

def split_data(
    data: DataFrame, /, 
    dtype_selector: str = "biselect", 
    columns: Optional[Union[list, tuple]] = None, 
    datetime_is_numeric=False, 
    ) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
    """
    Splits a DataFrame into sub-DataFrames based on data types of the columns,
    with an option to treat datetime columns as numeric.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to be processed.
    dtype_selector : str, optional
        Selector to determine the type of DataFrame to return. This 
        parameter controls the filtering and categorization of DataFrame 
        columns based on their data types. Each option has a specific 
        purpose:
        - ``numeric``: Returns a DataFrame containing only numeric columns.
        - ``numeric_only``: Similar to ``numeric``, but strictly excludes any 
          datetime columns even if they are considered numeric by pandas.
        - ``categorical``: Returns a DataFrame containing only categorical 
          columns, including string and categorical data types.
        - ``categoric_only``: Similar to ``categorical``, but strictly excludes 
          datetime columns even if they can be categorized.
        - ``dt``, ``datetime``: Returns a DataFrame containing only datetime 
          columns.
        - ``biselect``: Returns two DataFrames; one with numeric columns 
          (including datetime if not excluded by 'numeric_only') and one 
          with categorical columns.
        - ``bi-selector``, ``biselector``, ``bi_selector``: These are synonyms 
          for "biselect" and behave in the same way, offering flexibility 
          in parameter specification.
        Default is "biselect", which combines the functionality of 
        numeric and categorical selection, providing two separate 
        DataFrames.

    columns : list or tuple, optional
        Specific columns to consider in the DataFrame. If not provided, all
        columns are used.
        
    datetime_is_numeric : bool, optional
        If True, datetime columns are treated as numeric. This affects the 
        categorization and manipulation of the DataFrame based on dtype_selector.
        Default is False.
    
    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        Depending on the dtype_selector, returns either a single DataFrame or
        a tuple of DataFrames.
    
    Raises
    ------
    ValueError
        If an invalid dtype_selector is provided.
    
    Notes
    -----
    The function categorizes columns into numeric, categorical, and datetime
    types. It can exclude datetime types from the numeric and categorical
    results if 'only' is included in the dtype_selector. If specific columns
    are provided, the function will attempt to filter based on these columns;
    however, if any provided column names do not exist in the DataFrame,
    they will be ignored.
    
    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.transformation import split_data
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    ...     'C': ['apple', 'banana', 'cherry']
    ... })
    >>> numeric_df, categoric_df = split_data(df, 'biselect')
    >>> print(numeric_df)
       A
    0  1
    1  2
    2  3
    >>> print(categoric_df)
            C
    0   apple
    1  banana
    2  cherry
    """
    # validate selector 
    dtype_selector = validate_dtype_selector(dtype_selector)
    # build data frame is not.
    data = build_data_if(data, to_frame=True, input_name='col', force=True, 
                         raise_exception =True ) 
    # Initialize frame
    numeric_df = pd.DataFrame()
    categoric_df = pd.DataFrame()
    datetime_df = pd.DataFrame()

    def exclude_datetime_if_only(
            d: pd.DataFrame, selector: str) -> pd.DataFrame:
        """Exclude datetime columns from DataFrame if 'only' 
        is present in the selector."""
        if "only" in selector: 
            datetime_is_numeric=False 
        if not datetime_is_numeric:
            return d.select_dtypes(exclude=['datetime'])
        return d

    # Filter by provided columns if applicable
    if columns is not None:
        data = ( 
            data[columns] if set(columns).issubset(data.columns) 
            else data.drop(columns, axis=1, errors='ignore')
            )
    # Identify columns by their data types
    for col in data.columns:
        if ( 
                pd.api.types.is_numeric_dtype(data[col]) 
                or pd.api.types.is_datetime64_any_dtype(data[col])
                ):
            numeric_df[col] = data[col]
        if ( 
                pd.api.types.is_categorical_dtype(data[col]) 
                or pd.api.types.is_string_dtype(data[col]) 
                or pd.api.types.is_datetime64_any_dtype(data[col])
                ):
            categoric_df[col] = data[col]
            
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            datetime_df[col] = data[col]
    
    # Return based on dtype_selector
    if 'numeric' in dtype_selector: 
        if numeric_df.empty:
            warnings.warn("No numeric columns found.")
        return exclude_datetime_if_only(numeric_df, dtype_selector)
    
    elif 'categoric' in dtype_selector: 
        if categoric_df.empty:
            warnings.warn("No categorical columns found.")
        return exclude_datetime_if_only(categoric_df, dtype_selector)
    
    elif dtype_selector in ["dt", "datetime"]:
        if datetime_df.empty:
            warnings.warn("No datetime columns found.")
        return datetime_df
       
    elif dtype_selector =="biselect":
        # Use 'join' to combine dataframes to avoid duplicating 
        # columns that appear in both
        if numeric_df.empty: 
            combined_df = categoric_df.join(
                numeric_df.loc[:, numeric_df.columns.difference(categoric_df.columns)]
                )
        else: 
            combined_df = numeric_df.join(
                categoric_df.loc[:, categoric_df.columns.difference(numeric_df.columns)]
                )

        return ( 
            combined_df.loc[:, numeric_df.columns], 
            combined_df.loc[:, categoric_df.columns.difference(numeric_df.columns)]
            )

@isdf 
def sanitize(
    data:DataFrame, /, 
    fill_missing:Optional[str]=None, 
    remove_duplicates:bool=True, 
    outlier_method:Optional[str]=None, 
    consistency_transform:Optional[str]=None, 
    threshold:float|int=3
    )->TypeGuard[DataFrame]:
    """
    Perform data cleaning on a DataFrame with many options. 
    
    Options consists for handling missing values, removing duplicates, 
    detecting and removing outliers, and transforming string data for 
    consistency.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be cleaned.
    fill_missing : {'median', 'mean', 'mode', None}, optional
        Method to fill missing values. If None, no filling is performed.
        - 'median': Fill missing values with the median of each column.
        - 'mean': Fill missing values with the mean of each column.
        - 'mode': Fill missing values with the mode of each column.
        Suitable for datasets where missing values are present and a simple 
        imputation is required.
    remove_duplicates : bool, optional
        If True, removes duplicate rows from the DataFrame. Useful in scenarios
        where duplicate entries 
        do not provide additional information and might skew the analysis.
    outlier_method : {'z_score', 'iqr', None}, optional
        Method for outlier detection and removal. If None, no outlier 
        processing is performed.
        - 'z_score': Identifies and removes outliers using Z-score.
        - 'iqr': Identifies and removes outliers using the Interquartile Range.
        Choose based on the nature of the data and the requirement of the analysis.
        Default is ``iqr``. 
    consistency_transform : {'lower', 'upper', None}, optional
        Transformation to apply to string columns for consistency. If None,
        no transformation is applied.
        - 'lower': Converts strings to lowercase.
        - 'upper': Converts strings to uppercase.
        Useful for categorical data where case consistency is important.
    threshold : float, optional
        The threshold value used for outlier detection methods. Default is 3.
        For 'z_score', it represents
        the number of standard deviations from the mean. For 'iqr', it is the 
        multiplier for the IQR.

    Returns
    -------
    pd.DataFrame
        The cleaned and processed DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.transformation import sanitize
    >>> data = {'A': [1, 2, None, 4], 'B': ['X', 'Y', 'Y', None], 'C': [1, 1, 2, 2]}
    >>> df = pd.DataFrame(data)
    >>> cleaned_df = sanitize(df, fill_missing='median', remove_duplicates=True,
                                outlier_method='z_score', 
                                consistency_transform='lower', threshold=3)
    >>> print(cleaned_df)
    """
    # validate input parameters 
    outlier_method = parameter_validator(
        'outlier_method', ['z_score', 'iqr'])(outlier_method)
    fill_missing = parameter_validator(
        'fill_missing', ['median', 'mean', 'mode'])(fill_missing)
    consistency_transform = parameter_validator(
        'consistency_transform', ['lower', 'upper'])(consistency_transform)
 
    data = build_data_if(data, to_frame=True, force=True, input_name="feature_", 
                         raise_warning='mute')
    data = to_numeric_dtypes( data ) # verify integrity 
    df_cleaned = data.copy()
    if fill_missing:
        fill_methods = {
            'median': data.median(numeric_only=True),
            'mean': data.mean(numeric_only=True),
            'mode': data.mode().iloc[0]
        }
        df_cleaned.fillna(fill_methods.get(fill_missing, None), inplace=True)

    if remove_duplicates:
        df_cleaned.drop_duplicates(inplace=True)
    
    if outlier_method:
        if outlier_method == 'z_score':
            for col in df_cleaned.select_dtypes(include=[np.number]):
                df_cleaned = df_cleaned[(np.abs(df_cleaned[col] - df_cleaned[col].mean()
                                                ) / df_cleaned[col].std()) < threshold]
        elif outlier_method == 'iqr':
            for col in df_cleaned.select_dtypes(include=[np.number]):
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                df_cleaned = df_cleaned[~((df_cleaned[col] < (
                    Q1 - threshold * IQR)) | (df_cleaned[col] > (Q3 + threshold * IQR)))]

    if consistency_transform:
        transform_methods = {
            'lower': lambda x: x.lower(),
            'upper': lambda x: x.upper()
        }
        for col in df_cleaned.select_dtypes(include=[object]):
            df_cleaned[col] = df_cleaned[col].astype(str).map(
                transform_methods.get(consistency_transform, lambda x: x))

    return df_cleaned

@isdf 
def format_long_column_names(
    data:DataFrame, /,  
    max_length:int=10, 
    return_mapping:bool=False, 
    name_case:str='none'
    )->TypeGuard[DataFrame]:
    """
    Modifies long column names in a DataFrame to a more concise format
    
    Function changes the case as specified, and optionally returns a mapping 
    of new column names to original names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with potentially long column names.
    max_length : int, optional
        Maximum length of the formatted column names. Default is 10.
    return_mapping : bool, optional
        If True, returns a mapping of new column names to original names.
    name_case : {'capitalize', 'lowercase', 'none'}, optional
        Case transformation to apply to the new column names. Default is 'none'.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        The DataFrame with modified column names. If return_mapping is True,
        also returns a dictionary mapping new column names to original names.

    Example
    -------
    >>> from gofast.dataops.transformation import format_long_column_names
    >>> data = {'VeryLongColumnNameIndeed': [1, 2, 3], 'AnotherLongColumnName': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> new_df, mapping = format_long_column_names(
        df, max_length=10, return_mapping=True, name_case='capitalize')
    >>> print(new_df.columns)
    >>> print(mapping)
    """
    if not isinstance ( data, pd.DataFrame): 
        raise TypeError ("Input data is not a pandas DataFrame.")
    new_names = {}
    for col in data.columns:
        # Create a formatted name
        new_name = col[:max_length].rstrip('_')

        # Apply case transformation
        if name_case == 'capitalize':
            new_name = new_name.capitalize()
        elif name_case == 'lowercase':
            new_name = new_name.lower()

        new_names[new_name] = col
        data.rename(columns={col: new_name}, inplace=True)

    if return_mapping:
        return data, new_names

    return data

def group_and_filter(
    data: DataFrame, /, 
    group_column: str, 
    groups: List[Any], 
    sort: bool = False, 
    sort_by: Optional[Union[str, List[str]]] = None, 
    mode: str = 'strict', 
    handle_na: bool = False, 
    na_fill_value: Any = None, 
    conditional_filters: Optional[Dict[str, tuple]] = None
    ):

    """
    Filter a DataFrame based on specified group values in a column and apply
    additional conditional filters. Optionally sort the resulting DataFrame.

    Parameters
    ----------
    data : DataFrame
        The pandas DataFrame to process.
    group_column : str
        The name of the column to filter groups from.
    groups : list
        List of groups to filter in the group_column.
    sort : bool, optional
        Whether to sort the resulting DataFrame; uses `sort_by` if provided.
    sort_by : str or list of str, optional
        Columns to sort by; defaults to `group_column` if None.
    mode : str, optional
        Operational mode, 'strict' raises errors on unique identifiers, 'soft' 
        bypasses this check.
    handle_na : bool, optional
        Specifies whether to handle NA values in `group_column`.
    na_fill_value : any, optional
        Value to fill NA in `group_column` if `handle_na` is True.
    conditional_filters : dict, optional
        Additional column-specific conditions in the form {column: (operator, value)}.

    Returns
    -------
    DataFrame
        Filtered (and possibly sorted) DataFrame.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.transformation import group_and_filter
    >>> df = pd.DataFrame({
    ...     'A': ['a', 'b', 'a', 'd'],
    ...     'B': [1, 2, 3, 4]
    ... })
    >>> group_and_filter(df, 'A', ['a', 'd'], sort=True)
       A  B
    0  a  1
    2  a  3
    3  d  4

    >>> group_and_filter(df, 'A', ['a'], handle_na=True, na_fill_value='missing')
       A  B
    0  a  1
    2  a  3

    >>> group_and_filter(df, 'A', ['a'], conditional_filters={'B': ('>', 2)}, 
    ...                  mode='soft')
       A  B
    2  a  3

    Notes
    -----
    The function raises a ValueError if the `group_column` does not exist in the
    DataFrame or if it is used in 'strict' mode on a column with unique identifiers 
    when not allowed. The use of `query` in conditional filters supports complex 
    filtering expressions efficiently.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected 'data' to be a pandas DataFrame,"
                        f" but got {type(data).__name__}.")

    if group_column not in data.columns:
        raise ValueError(f"Column '{group_column}' does not exist in the DataFrame.")

    if handle_na:
        data[group_column].fillna(na_fill_value, inplace=True)

    if conditional_filters:
        for col, (operator, val) in conditional_filters.items():
            if col in data.columns:
                data = data.query(f"{col} {operator} @val")

    filtered_data = data[data[group_column].isin(groups)]

    if mode == 'strict' and filtered_data[group_column].is_unique:
        raise ValueError("Cannot perform grouping on a unique identifier"
                         " column in 'strict' mode.")

    if sort:
        sort_columns = sort_by if sort_by else group_column
        filtered_data = filtered_data.sort_values(by=sort_columns)

    return filtered_data


def validate_list_of_strings(value, param_name):
    """Helper function to validate that a parameter is a list of strings."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"'{param_name}' should be a list of strings")

def validate_dict(value, param_name):
    """Helper function to validate that a parameter is a dictionary."""
    if not isinstance(value, dict):
        raise ValueError(f"'{param_name}' should be a dictionary")

def smart_group(
    data: DataFrame, 
    group_by: Union[str, List[str]], 
    aggregations: Optional[Dict[str, Union[str, List[str], Dict[str, Any]]]] = None,
    having: Optional[Dict[str, Any]] = None, 
    fill_na: Optional[Dict[str, Any]] = None, 
    sort_by: Optional[Union[str, List[str]]] = None, 
    ascending: bool = True,
    drop_na: bool = False,
    keep_columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Groups a DataFrame by specified column(s) and applies various optional
    transformations including aggregation, conditional filtering, and sorting.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be processed.
    group_by : str or List[str]
        The column(s) to group the DataFrame by.
    aggregations : dict, optional
        Specifies the aggregation operations to apply after grouping.
        The dictionary should map column names to aggregation functions
        or lists of functions.
    having : dict, optional
        Allows filtering on the aggregated results. Each key-value pair 
        specifies the column and condition to filter by.
    fill_na : dict, optional
        Specifies values to replace NA/NaN values in the DataFrame before
        grouping. The dictionary maps column names to fill values.
    sort_by : str or List[str], optional
        Specifies one or more columns to sort the resulting DataFrame by.
    ascending : bool, optional
        Determines if the sorting should be ascending (True) or descending (False).
    drop_na : bool, optional
        If True, rows with NA values in the `group_by` columns will be dropped
        before grouping.
    keep_columns : List[str], optional
        List of non-aggregated columns to retain in the resulting DataFrame.
        
    Returns
    -------
    pd.DataFrame
        The grouped (and potentially aggregated, filtered, and sorted) DataFrame.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.dataops.transformation import smart_group 
    
    # Example of grouping by a single column with aggregation:
    >>> df = pd.DataFrame({
    ...     'A': ['foo', 'bar', 'foo', 'bar'],
    ...     'B': [1, 2, 3, 4],
    ...     'C': [2, 4, 6, 8]
    ... })
    >>> smart_group(df, group_by='A', aggregations={'B': 'sum', 'C': 'mean'})
         B    C
    A          
    bar  6  6.0
    foo  4  4.0
    
    # Example of using the 'having' filter to further filter aggregated results:
    >>> smart_group(df, group_by='A', aggregations={'B': 'sum', 'C': 'mean'},
    ...             having={'B': '>4'})
         B    C
    A          
    bar  6  6.0

    # Example of using 'fill_na' to handle missing values before grouping:
    >>> df.loc[2, 'B'] = None
    >>> smart_group(df, group_by='A', fill_na={'B': 0}, aggregations={'B': 'sum'})
           B
    A       
    bar  6.0
    foo  1.0
    
    # Example of sorting the results:
    >>> smart_group(df, group_by='A', sort_by='C', ascending=False,
    ...             keep_columns=['C'])
        index    A  count  C
     0      0  bar      2  4
     1      1  foo      2  2
    Notes
    -----
    - The 'aggregations' parameter should be used carefully to ensure that
      specified aggregation functions are compatible with the data types of
      the columns being aggregated.
    - The 'having' parameter allows complex filtering but requires understanding
      of how aggregated column names are formed.
    """
    # Assertions for input validation
    assert isinstance(data, pd.DataFrame), "Expected 'data' to be a pandas DataFrame"
    if isinstance(group_by, list):
        validate_list_of_strings(group_by, 'group_by')
    elif not isinstance(group_by, str):
        raise ValueError("'group_by' should be either a string or a list of strings")

    assert isinstance(ascending, bool), "'ascending' should be a boolean"
    assert isinstance(drop_na, bool), "'drop_na' should be a boolean"

    if aggregations:
        validate_dict(aggregations, 'aggregations')
    if having:
        validate_dict(having, 'having')
    if fill_na:
        validate_dict(fill_na, 'fill_na')
    if keep_columns:
        validate_list_of_strings(keep_columns, 'keep_columns')
        
    if sort_by:
        if isinstance(sort_by, list):
            validate_list_of_strings(sort_by, 'sort_by')
        elif not isinstance(sort_by, str):
            raise ValueError("'sort_by' should be either a string or a list of strings")
            
    # Ensure group_by is a list
    if isinstance(group_by, str):
        group_by = [group_by]  # Convert string to list for consistent handling
    
    # Combine group_by and keep_columns for selection
    if keep_columns:
        columns_to_select = group_by + keep_columns
        data = data[columns_to_select]
    
    # Handle missing values
    if fill_na:
        data = data.fillna(fill_na)
    elif drop_na:
        data = data.dropna(subset=group_by)
    
    # Grouping and aggregation
    grouped = data.groupby(group_by)
    if aggregations:
        result = grouped.agg(aggregations)
    else:
        result = grouped.size().reset_index(name='count')
    
    # Reintroduce non-aggregated columns that were kept
    if keep_columns:
        # Reset index for result to allow proper merging
        result.reset_index(inplace=True)
        
        for col in keep_columns:
            if col not in result.columns:
                # Perform a merge operation to reintroduce the column from the original data
                additional_data = data[[*group_by, col]].drop_duplicates(subset=group_by)
                result = result.merge(additional_data, on=group_by, how='left')
    
    # Apply conditions specified in 'having' if any
    if having:
        query = " & ".join([f"{key}{value}" for key, value in having.items()])
        result = result.query(query)
    
    # Sorting the final result if specified
    if sort_by:
        result = result.sort_values(by=sort_by, ascending=ascending)
    
    return result



















