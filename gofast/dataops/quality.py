# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Concentrates on assessing and improving the quality of the data."""
from __future__ import annotations, print_function 
import re
import datetime
import warnings 
from scipy import stats
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from ..api.extension import isinstance_ 
from ..api.formatter import MultiFrameFormatter, format_iterable
from ..api.summary import ReportFactory, Summary
from ..api.summary import ResultSummary, assemble_reports
from ..api.types import Any, List, DataFrame, Optional, Series
from ..api.types import Dict, Union, Tuple, ArrayLike, Callable
from ..api.util import get_table_size , to_snake_case
from ..decorators import isdf, Dataify
from ..decorators import Extract1dArrayOrSeries 
from ..tools.baseutils import reshape_to_dataframe
from ..tools.coreutils import ellipsis2false, smart_format
from ..tools.coreutils import assert_ratio, validate_ratio 
from ..tools.validator import is_frame, parameter_validator, validate_numeric  
from ..tools.validator import _is_numeric_dtype, filter_valid_kwargs

TW = get_table_size() 

__all__= [
     'analyze_data_corr',
     'assess_outlier_impact',
     'audit_data',
     'check_correlated_features',
     'check_missing_data',
     'check_unique_values',
     'convert_date_features',
     'correlation_ops',
     'data_assistant',
     'drop_correlated_features',
     'handle_categorical_features',
     'handle_duplicates',
     'handle_missing_data',
     'handle_outliers_in',
     'handle_skew',
     'merge_frames_on_index',
     'quality_control',
     'scale_data',
 ]

def audit_data(
    data: DataFrame,/,  
    dropna_threshold: float = 0.5, 
    categorical_threshold: int = 10, 
    handle_outliers: bool = False,
    handle_missing: bool = True, 
    handle_scaling: bool = False, 
    handle_date_features: bool = False, 
    handle_categorical: bool = False, 
    replace_with: str = 'median', 
    lower_quantile: float = 0.01, 
    upper_quantile: float = 0.99,
    fill_value: Optional[Any] = None,
    scale_method: str = "minmax",
    missing_method: str = 'drop_cols', 
    outliers_method: str = "clip", 
    date_features: Optional[List[str]] = None,
    day_of_week: bool = False, 
    quarter: bool = False, 
    format_date: Optional[str] = None, 
    return_report: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Audits and preprocesses a DataFrame for analytical consistency. 
    
    This function streamlines the data cleaning process by handling various 
    aspects of data quality, such as outliers, missing values, and data scaling. 
    It provides flexibility to choose specific preprocessing steps according 
    to the needs of the analysis or modeling.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be audited and preprocessed. It should be a pandas 
        DataFrame containing the data to be cleaned.

    dropna_threshold : float, optional
        Specifies the threshold for dropping columns or rows with missing 
        values. It determines the proportion of missing values above which 
        a column or row will be dropped from the DataFrame. 
        The default value is 0.5 (50%).

    categorical_threshold : int, optional
        Defines the maximum number of unique values a column can have to be 
        considered as a categorical variable. Columns with unique values 
        less than or equal to this threshold will be converted to categorical
        type.

    handle_outliers : bool, optional
        Determines whether to apply outlier handling on numerical columns. 
        If set to True, outliers in the data will be addressed according 
        to the specified method.

    handle_missing : bool, optional
        If True, the function will handle missing data in the DataFrame based 
        on the specified missing data handling method.

    handle_scaling : bool, optional
        Indicates whether to scale numerical columns using the specified 
        scaling method. Scaling is essential for certain analyses and modeling 
        techniques, especially when variables are on different scales.

    handle_date_features : bool, optional
        If True, columns specified as date features will be converted to 
        datetime format, and additional date-related features 
        (like day of the week, quarter) will be extracted.

    handle_categorical : bool, optional
        Enables the handling of categorical features. If set to True, 
        numerical columns with a number of unique values below the categorical
        threshold will be treated as categorical.

    replace_with : str, optional
        For outlier handling, specifies the method of replacement 
        ('mean' or 'median') for the 'replace' outlier method. It determines 
        how outliers will be replaced in the dataset.

    lower_quantile : float, optional
        The lower quantile value used for clipping outliers. It sets the 
        lower boundary for outlier detection and handling.

    upper_quantile : float, optional
        The upper quantile value for clipping outliers. It sets the upper 
        boundary for outlier detection and handling.

    fill_value : Any, optional
        Specifies the value to be used for filling missing data when the 
        missing data handling method is set to 'fill_value'.

    scale_method : str, optional
        Determines the method for scaling numerical data. Options include 
        'minmax' (scales data to a range of [0, 1]) and 'standard' 
        (scales data to have zero mean and unit variance).

    missing_method : str, optional
        The method used to handle missing data in the DataFrame. Options 
        include 'drop_cols' (drop columns with missing data) and other 
        methods based on specified criteria such as 'drop_rows', 'fill_mean',
        'fill_median', 'fill_value'. 

    outliers_method : str, optional
        The method used for handling outliers in the dataset. Options 
        include 'clip' (limits the extreme values to specified quantiles) and 
        other outlier handling methods such as 'remove' and 'replace'.

    date_features : List[str], optional
        A list of column names in the DataFrame to be treated as date features. 
        These columns will be converted to datetime and additional date-related
        features will be extracted.

    day_of_week : bool, optional
        If True, adds a column representing the day of the week for each 
        date feature column.

    quarter : bool, optional
        If True, adds a column representing the quarter of the year for 
        each date feature column.

    format_date : str, optional
        Specifies the format of the date columns if they are not in standard 
        datetime format.

    return_report : bool, optional
        If True, the function returns a detailed report summarizing the 
        preprocessing steps performed on the DataFrame.

    view : bool, optional
        Enables visualization of the data's state before and after 
        preprocessing. If True, displays comparative heatmaps for each step.

    cmap : str, optional
        The colormap for the heatmap visualizations, enhancing the clarity 
        and aesthetics of the plots.

    fig_size : Tuple[int, int], optional
        Determines the size of the figure for the visualizations, allowing 
        customization of the plot dimensions.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        The audited and preprocessed DataFrame. If return_report is True, 
        also returns a comprehensive report detailing the transformations 
        applied.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.dataops.quality import audit_data
    >>> data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    >>> audited_data, report = audit_data(data, handle_outliers=True, return_report=True)
    """
    is_frame (data, df_only=True, raise_exception=True, 
              objname="Data for auditing" )
    report = {}
    data_copy = data.copy()

    def update_report(new_data, step_report):
        nonlocal data, report
        if return_report:
            data, step_report = new_data
            if isinstance_( step_report, ReportFactory): 
                step_report= step_report.report 
            report = {**report, **step_report}
        else:
            data = new_data

    # Handling outliers
    if handle_outliers:
        update_report(handle_outliers_in(
            data, method=outliers_method, replace_with=replace_with,
            lower_quantile=assert_ratio(lower_quantile),
            upper_quantile=assert_ratio(upper_quantile),
            return_report=return_report),
            {})

    # Handling missing data
    if handle_missing:
        update_report(handle_missing_data(
            data, method=missing_method, dropna_threshold=assert_ratio(
                dropna_threshold), fill_value=fill_value, 
            return_report=return_report), {})

    # Handling date features
    if handle_date_features and date_features:
        update_report(convert_date_features(
            data, date_features, day_of_week=day_of_week, quarter=quarter, 
            format=format_date, return_report=return_report), {})

    # Scaling data
    if handle_scaling:
        update_report(scale_data(
            data, method=scale_method, return_report=return_report), {})

    # Handling categorical features
    if handle_categorical:
        update_report(handle_categorical_features(
            data, categorical_threshold=categorical_threshold, 
            return_report=return_report), {})

    # Compare initial and final data if view is enabled
    if view:
        plt.figure(figsize=(fig_size[0], fig_size[1] * 2))
        plt.subplot(2, 1, 1)
        sns.heatmap(data_copy.isnull(), yticklabels=False,
                    cbar=False, cmap=cmap)
        plt.title('Data Before Auditing')

        plt.subplot(2, 1, 2)
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False,
                    cmap=cmap)
        plt.title('Data After Auditing')
        plt.show()
    
    # make a report obj 
    if return_report: 
        report_obj= ReportFactory(title ="Data Audition")
        report_obj.add_mixed_types(report, table_width= TW)
    
    return (data, report_obj) if return_report else data

def handle_categorical_features(
    data: DataFrame, /, 
    categorical_threshold: int = 10,
    return_report: bool = False,
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Converts numerical columns with a limited number of unique values 
    to categorical columns in the DataFrame and optionally visualizes the 
    data distribution before and after the conversion.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to process.
    categorical_threshold : int, optional
        Maximum number of unique values in a column for it to be considered 
        categorical.
    return_report : bool, optional
        If True, returns a report summarizing the categorical feature handling.
    view : bool, optional
        If True, displays a heatmap of the data distribution before and after 
        handling.
    cmap : str, optional
        The colormap for the heatmap visualization.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        DataFrame with categorical features handled and optionally a report.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.dataops.quality import handle_categorical_features
    >>> data = pd.DataFrame({'A': [1, 2, 1, 3], 'B': range(4)})
    >>> updated_data, report = handle_categorical_features(
        data, categorical_threshold=3, return_report=True, view=True)
    >>> report.converted_columns
    ['A']
    """
    is_frame (data, df_only=True, raise_exception=True)
    original_data = data.copy()
    report = {'converted_columns': []}
    numeric_cols = data.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        if data[col].nunique() <= categorical_threshold:
            data[col] = data[col].astype('category')
            report['converted_columns'].append(col)

    # Visualization of data distribution before and after handling
    if view:
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data[numeric_cols].nunique().to_frame().T, 
                    annot=True, cbar=False, cmap=cmap)
        plt.title('Unique Values Before Categorization')

        plt.subplot(1, 2, 2)
        sns.heatmap(data[numeric_cols].nunique().to_frame().T, annot=True,
                    cbar=False, cmap=cmap)
        plt.title('Unique Values After Categorization')
        plt.show()
    # make a report obj 
    report_obj= ReportFactory(title ="Categorical Features Handling", **report )
    report_obj.add_mixed_types(report, table_width= TW  )
    
    return (data, report_obj) if return_report else data

def convert_date_features(
    data: DataFrame, /, 
    date_features: List[str], 
    day_of_week: bool = False, 
    quarter: bool = False,
    format: Optional[str] = None,
    return_report: bool = False,
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Converts specified columns in the DataFrame to datetime and extracts 
    relevant features. 
    
    Optionally Function returns a report of the transformations and 
    visualizing the data distribution  before and after conversion.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the date columns.
    date_features : List[str]
        List of column names to be converted into datetime and to extract 
        features from.
    day_of_week : bool, optional
        If True, adds a column representing the day of the week. Default is False.
    quarter : bool, optional
        If True, adds a column representing the quarter of the year.
        Default is False.
    format : str, optional
        The specific format of the date columns if they are not in a standard
        datetime format.
    return_report : bool, optional
        If True, returns a report summarizing the date feature transformations.
    view : bool, optional
        If True, displays a comparative heatmap of the data distribution 
        before and after the conversion.
    cmap : str, optional
        The colormap for the heatmap visualization.
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap.
    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        DataFrame with additional date-related features and optionally a report.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.dataops.quality import convert_date_features
    >>> data = pd.DataFrame({'date': ['2021-01-01', '2021-01-02']})
    >>> updated_data, report = convert_date_features(
        data, ['date'], day_of_week=True, quarter=True, return_report=True, view=True)
    >>> report.converted_columns
    ['date']
    >>> report.added_features
    ['date_year', 'date_month', 'date_day', 'date_dayofweek', 'date_quarter']
    """
    is_frame (data, df_only=True, raise_exception=True)
    original_data = data.copy()
    report = {'converted_columns': date_features, 'added_features': []}

    for feature in date_features:
        data[feature] = pd.to_datetime(data[feature], format=format)
        year_col = f'{feature}_year'
        month_col = f'{feature}_month'
        day_col = f'{feature}_day'
        data[year_col] = data[feature].dt.year
        data[month_col] = data[feature].dt.month
        data[day_col] = data[feature].dt.day
        report['added_features'].extend([year_col, month_col, day_col])

        if day_of_week:
            dow_col = f'{feature}_dayofweek'
            data[dow_col] = data[feature].dt.dayofweek
            report['added_features'].append(dow_col)

        if quarter:
            quarter_col = f'{feature}_quarter'
            data[quarter_col] = data[feature].dt.quarter
            report['added_features'].append(quarter_col)

    # Visualization of data distribution before and after conversion
    if view:
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data[date_features].nunique().to_frame().T,
                    annot=True, cbar=False, cmap=cmap)
        plt.title('Unique Values Before Conversion')

        plt.subplot(1, 2, 2)
        sns.heatmap(data[date_features + report['added_features']
                         ].nunique().to_frame().T, annot=True, cbar=False,
                    cmap=cmap)
        plt.title('Unique Values After Conversion')
        plt.show()

    report_obj= ReportFactory(title ="Date Features Conversion", **report )
    report_obj.add_mixed_types(report, table_width= TW)
    return (data, report_obj) if return_report else data

@isdf 
def scale_data(
    data: DataFrame, /, 
    method: str = 'norm',
    return_report: bool = False,
    use_sklearn: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Scales numerical columns in the DataFrame using the specified scaling 
    method. 
    
    Optionally returns a report on the scaling process along with 
    visualization.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be scaled.
    method : str
        Scaling method - 'minmax', 'norm', or 'standard'.
    return_report : bool, optional
        If True, returns a report summarizing the scaling process.
    use_sklearn: bool, optional 
        If True and scikit-learn is installed, use its scaling utilities.
    view : bool, optional
        If True, displays a heatmap of the data distribution before and 
        after scaling.
    cmap : str, optional
        The colormap for the heatmap visualization.
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        The scaled DataFrame and optionally a report.

    Raises
    ------
    ValueError
        If an invalid scaling method is provided.
        
    Note 
    -----
    Scaling method - 'minmax' or 'standard'.
    'minmax' scales data to the [0, 1] range using the formula:
        
    .. math:: 
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        
    'standard' scales data to zero mean and unit variance using the formula:
        
    .. math:: 
        X_scaled = (X - X.mean()) / X.std()
        
    Example
    -------
    >>> from gofast.dataops.quality import scale_data
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> scaled_data, report = scale_data(data, 'minmax', return_report=True, view=True)
    >>> print(report) 
    >>> report.method_used
    'minmax'
    >>> report.columns_scaled
    ['A', 'B']
    
    """
    is_frame (data, df_only=True, raise_exception=True, 
              objname="Exceptionnaly, scaling data")
    numeric_cols = data.select_dtypes(include=['number']).columns
    report = {'method_used': method, 'columns_scaled': list(numeric_cols)}
    
    original_data = data.copy()
    # Determine which scaling method to use
    method = parameter_validator (
        "method", target_strs=('minmax', "standard", "norm")) (method)

    if use_sklearn:
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        except ImportError:
            use_sklearn = False

    if not use_sklearn:
        minmax_scale = lambda col: (col - col.min()) / (col.max() - col.min())
        standard_scale = lambda col: (col - col.mean()) / col.std()
        scaling_func = minmax_scale if method == 'minmax' else standard_scale
        data[numeric_cols] = data[numeric_cols].apply(scaling_func)

    # Visualization of data distribution before and after scaling
    if view:
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data[numeric_cols], annot=True, cbar=False, 
                    cmap=cmap)
        plt.title('Before Scaling')

        plt.subplot(1, 2, 2)
        sns.heatmap(data[numeric_cols], annot=True, cbar=False, cmap=cmap)
        plt.title('After Scaling')
        plt.show()
        
    report['used_scikit_method']=use_sklearn
    report_obj= ReportFactory(title ="Data Scaling", **report )
    report_obj.add_mixed_types(report, table_width= TW)
    return (data, report_obj) if return_report else data

def handle_outliers_in(
    data: DataFrame, /, 
    method: str = 'clip', 
    replace_with: str = 'median', 
    lower_quantile: float = 0.01, 
    upper_quantile: float = 0.99,
    return_report: bool = False, 
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> DataFrame:
    """
    Handles outliers in numerical columns of the DataFrame using various 
    methods. 
    
    Optionally, function displays a comparative plot showing the data 
    distribution before and after outlier handling.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame with potential outliers.
    method : str, optional
        Method to handle outliers ('clip', 'remove', 'replace'). 
        Default is 'clip'.
    replace_with : str, optional
        Specifies replacement method ('mean' or 'median') for 'replace'.
        Default is 'median'.
    lower_quantile : float, optional
        Lower quantile for clipping outliers. Default is 0.01.
    upper_quantile : float, optional
        Upper quantile for clipping outliers. Default is 0.99.
    return_report : bool, optional
        If True, returns a report summarizing the outlier handling process.
    view : bool, optional
        If True, displays a comparative plot of the data distribution before 
        and after handling outliers.
    cmap : str, optional
        The colormap for the heatmap visualization. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled and optionally a report dictionary.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.dataops.quality import handle_outliers_in_data
    >>> data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    >>> data, report = handle_outliers_in_data(data, method='clip', view=True, 
                                               cmap='plasma', return_report=True)
    >>> print(report) 
    >>> report.lower_quantile
    0.01
    """
    is_frame (data, df_only=True, raise_exception=True)
    numeric_cols = data.select_dtypes(include=['number']).columns
    data_before = data.copy()  # Copy of the original data for comparison
    report = {}

    # Handling outliers
    if method == 'clip':
        lower = data[numeric_cols].quantile(lower_quantile)
        upper = data[numeric_cols].quantile(upper_quantile)
        data[numeric_cols] = data[numeric_cols].clip(lower, upper, axis=1)
        report['method'] = 'clip'
    elif method == 'remove':
        # Removing outliers based on quantiles
        lower = data[numeric_cols].quantile(lower_quantile)
        upper = data[numeric_cols].quantile(upper_quantile)
        data = data[(data[numeric_cols] >= lower) & (data[numeric_cols] <= upper)]
        report['method'] = 'remove'
    elif method == 'replace':
        if replace_with not in ['mean', 'median']:
            raise ValueError("Invalid replace_with option. Choose 'mean' or 'median'.")
        replace_func = ( data[numeric_cols].mean if replace_with == 'mean' 
                        else data[numeric_cols].median)
        data[numeric_cols] = data[numeric_cols].apply(lambda col: col.where(
            col.between(col.quantile(lower_quantile), col.quantile(upper_quantile)),
            replace_func(), axis=0))
        report['method'] = 'replace'
    else:
        raise ValueError("Invalid method for handling outliers.")
    
    report['lower_quantile'] = lower_quantile
    report['upper_quantile'] = upper_quantile

    if view:
        # Visualize data distribution before and after handling outliers
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
        
        sns.heatmap(data_before[numeric_cols].isnull(), yticklabels=False, 
                    cbar=False, cmap=cmap, ax=axes[0])
        axes[0].set_title('Before Outlier Handling')

        sns.heatmap(data[numeric_cols].isnull(), yticklabels=False,
                    cbar=False, cmap=cmap, ax=axes[1])
        axes[1].set_title('After Outlier Handling')

        plt.suptitle('Comparative Missing Value Heatmap')
        plt.show()
        
    report_obj= ReportFactory(title ="Outliers Handling", **report )
    report_obj.add_mixed_types(report, table_width= int(TW/2))
    return (data, report_obj) if return_report else data

@isdf 
def handle_missing_data(
    data: DataFrame, /, 
    method: Optional[str] = None,  
    fill_value: Optional[Any] = None,
    dropna_threshold: float = 0.5, 
    return_report: bool = False,
    view: bool = False, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Analyzes and handles patterns of missing data in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to analyze and handle missing data.
        
    method : str, optional
        Method to handle missing data. Options are:
        - 'drop_rows': Drop rows with missing data based on `dropna_threshold`.
        - 'drop_cols': Drop columns with missing data based on `dropna_threshold`.
        - 'fill_mean': Fill missing numerical data with the mean of the column.
        - 'fill_median': Fill missing numerical data with the median of the column.
        - 'fill_value': Fill missing data with the specified `fill_value`.
        - 'ffill': Forward fill to propagate the next values.
        - 'bfill': Backward fill to propagate the previous values.
        If `None`, 'ffill' (forward fill) is used as the default method.
        
    fill_value : Any, optional
        Value to use when filling missing data for the 'fill_value' method. 
        This parameter is required if `method` is 'fill_value'.
        
    dropna_threshold : float, optional
        Threshold for dropping rows/columns with missing data, expressed as 
        a proportion. Only used with 'drop_rows' or 'drop_cols' method. 
        Default is 0.5.
        
    return_report : bool, optional
        If `True`, returns a tuple of the DataFrame and a report dictionary.
        Default is `False`.
        
    view : bool, optional
        If `True`, displays a heatmap of missing data before and after handling.
        Default is `False`.
        
    cmap : str, optional
        The colormap for the heatmap visualization. Default is 'viridis'.
        
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap. Default is (12, 5).

    Returns
    -------
    Union[pandas.DataFrame, Tuple[pandas.DataFrame, dict]]
        DataFrame after handling missing data and optionally a report dictionary.

    Notes
    -----
    This function ensures the appropriate handling of missing data based on 
    the specified method. If no method is provided, forward fill ('ffill') 
    is used by default.

    The drop thresholds for rows or columns are calculated as:
    
    .. math::
        \text{Threshold} = \text{dropna_threshold} \times \text{number of columns/rows}
    
    Examples
    --------
    >>> from gofast.dataops.quality import handle_missing_data
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
    >>> updated_data, report = handle_missing_data(
    ...     data, view=True, method='fill_mean', return_report=True)
    >>> print(report)
    >>> report['stats']
    {'method_used': 'fill_mean', 'fill_value': None, 'dropna_threshold': None}

    See Also
    --------
    pandas.DataFrame.dropna : Remove missing values.
    pandas.DataFrame.fillna : Fill missing values.
    pandas.DataFrame.ffill : Forward fill.
    pandas.DataFrame.bfill : Backward fill.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in Python. 
           Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
 
    is_frame(data, df_only=True, raise_exception=True)
    
    # Analyze missing data
    original_data = data.copy()
    missing_data = pd.DataFrame(data.isnull().sum(), columns=['missing_count'])
    missing_data['missing_percentage'] = (missing_data['missing_count'] / len(data)) * 100

    # Handling missing data based on method
    if method is None:
        method = 'ffill'  

    handling_methods = {
        'drop_rows': lambda d: d.dropna(thresh=int(dropna_threshold * len(d.columns))),
        'drop_cols': lambda d: d.dropna(axis=1, thresh=int(dropna_threshold * len(d))),
        'fill_mean': lambda d: d.fillna(d.mean(numeric_only=True)),
        'fill_median': lambda d: d.fillna(d.median(numeric_only=True)),
        'fill_value': lambda d: d.fillna(fill_value),
        'ffill': lambda d: d.ffill(),
        'bfill': lambda d: d.bfill()
    }

    if method in handling_methods:
        if method == 'fill_value' and fill_value is None:
            raise ValueError("fill_value must be specified for 'fill_value' method.")
        data = handling_methods[method](data)

    else:
        raise ValueError(f"Invalid method specified: {method}")

    # Visualization of missing data before and after handling
    if view:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data.isnull(), yticklabels=False, cbar=False, cmap=cmap)
        plt.title('Before Handling Missing Data')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap=cmap)
        plt.title('After Handling Missing Data')
        plt.show()

    # Data report
    data_report = {
        "missing_data_before": original_data.isnull().sum(),
        "missing_data_after": data.isnull().sum(),
        "stats": {
            "method_used": method,
            "fill_value": fill_value if method == 'fill_value' else None,
            "dropna_threshold": dropna_threshold if method in [
                'drop_rows', 'drop_cols'] else None
        },
        "describe%% Basic statistics": missing_data.describe().round(4)
    }
    report_obj = ReportFactory(title="Missing Handling", **data_report)
    report_obj.add_mixed_types(data_report, table_width=TW)
    
    return (data, report_obj) if return_report else data

@isdf
def assess_outlier_impact(
    data: ArrayLike, /,
    outlier_threshold: int=3, 
    handle_na='ignore', 
    view=False, 
    fig_size=(14, 6)
    ):
    """
    Assess the impact of outliers on dataset statistics and optionally 
    visualize the results.

    This function calculates basic statistics with and without outliers, 
    allows handling of NaN values, and can visualize data points and outliers
    using a box plot and scatter plot.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, or numpy.ndarray
        Input data, which can be a DataFrame, Series, or NumPy array. If a
        DataFrame is provided, it should contain only one column.
    outlier_threshold : float, optional
        The z-score threshold to consider a data point an outlier, default is 3.
    handle_na : str, optional
        How to handle NaN values. Options are 'ignore', 'drop', or 'fill'. 'fill'
        replaces NaNs with the mean, default is 'ignore'.
    view : bool, optional
        If True, generates plots to visualize the outliers and their impact, 
        default is False.
    fig_size : tuple, optional
        The size of the figure for the plots if `view` is True, default is (14, 6).

    Returns
    -------
    dict
        A dictionary containing the mean and standard deviation with and without 
        outliers, and the number of outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.dataops.quality import assess_outlier_impact
    >>> data = np.random.normal(0, 1, 100)
    >>> data[::10] += np.random.normal(0, 10, 10)  # Add some outliers
    >>> results = assess_outlier_impact(data, view=True)
    >>> print(results)
    OutlierResults(
      {

           mean_with_outliers    : -0.06689646829390013
           std_with_outliers     : 2.662779727986304
           mean_without_outliers : -0.16663267681872657
           std_without_outliers  : 1.6752562593670428
           num_outliers          : 3

      }
    )
    >>> result.data 

               0
    0  -7.391298
    1  -1.173935
    2  -0.529047
    3   1.042972
    4  -0.509738
    ..       ...
    95  1.015017
    96 -1.723945
    97  0.001104
    98  1.333836
    99 -0.327440

    [100 rows x 1 columns]
    """
    columns=None  
    # Ensure the data is a numeric numpy array
    if isinstance(data, pd.Series):
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Non-numeric data types are not supported in the Series.")
        columns = data.name 
        data = data.values
    
    elif isinstance(data, pd.DataFrame):
        data = data.select_dtypes( include = [np.number])
        if data.empty: 
            raise ValueError("DataFrame must be of numeric types.")
        columns = data.columns.tolist()
        data = data.values.flatten()
    
    elif isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("The numpy array must contain only numeric data types.")
    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or a numpy array.")
        
    # Convert data to a NumPy array if it's not already
    data = np.asarray(data)
    
    outlier_threshold = validate_numeric(
        outlier_threshold, allow_negative=False )
    # Handle missing values according to the specified method
    handle_na = parameter_validator(
        "handle_na", target_strs={"ignore", "fill", "drop"}) (handle_na)
    if handle_na == 'drop':
        data = data[~np.isnan(data)]
    elif handle_na == 'fill':
        mean_data = np.nanmean(data)
        data = np.where(np.isnan(data), mean_data, data)

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Identify outliers
    outliers_mask = np.abs(data - mean) > outlier_threshold * std
    non_outliers_mask = ~outliers_mask
    
    # Calculate metrics without outliers
    mean_without_outliers = np.mean(data[non_outliers_mask])
    std_without_outliers = np.std(data[non_outliers_mask])
    
    # Number of outliers
    num_outliers = np.sum(outliers_mask)

    # Output results
    results = {
        'mean_with_outliers': mean,
        'std_with_outliers': std,
        'mean_without_outliers': mean_without_outliers,
        'std_without_outliers': std_without_outliers,
        'num_outliers': num_outliers
    }

    if view: 
        _visualize_outlier_impact(data, outliers_mask, non_outliers_mask, fig_size)
        
    if columns is not None: 
        data = reshape_to_dataframe(data, columns, error ="ignore")
        
    result_summary = ResultSummary (
        'outlier results', pad_keys="auto").add_results(results )
    result_summary.data = data 
    
    return result_summary

def _visualize_outlier_impact(
        data, outliers_mask, 
        non_outliers_mask, 
        fig_size=(14, 6)):
    outliers = data[outliers_mask]
    non_outliers = data[non_outliers_mask]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    # Box plot
    sns.boxplot(x=data, ax=axs[0])
    axs[0].set_title('Box Plot')
    axs[0].set_xlabel('Data Points')

    # Scatter plot
    axs[1].scatter(range(len(non_outliers)), non_outliers, color='blue', 
                   label='Non-outliers')
    axs[1].scatter(range(len(non_outliers), len(non_outliers) + len(outliers)
                         ), outliers, color='red', label='Outliers')
    axs[1].legend()
    axs[1].set_title('Scatter Plot of Data Points')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Data Points')

    # Show the plots
    plt.tight_layout()
    plt.show()

def merge_frames_on_index(
    *data: DataFrame, 
    index_col: str, 
    join_type: str = 'outer', 
    axis: int = 1, 
    ignore_index: bool = False, 
    sort: bool = False
    ) -> DataFrame:
    """
    Merges multiple DataFrames based on a specified column set as the index.

    Parameters
    ----------
    *data : pd.DataFrame
        Variable number of pandas DataFrames to merge.
    index_col : str
        The name of the column to set as the index in each DataFrame before merging.
    join_type : str, optional
        The type of join to perform. One of 'outer', 'inner', 'left', or 'right'.
        Defaults to 'outer'.
    axis : int, optional
        The axis to concatenate along. {0/'index', 1/'columns'}, default 1/'columns'.
    ignore_index : bool, optional
        If True, the resulting axis will be labeled 0, 1, …, n - 1. Default False.
    sort : bool, optional
        Sort non-concatenation axis if it is not already aligned. Default False.

    Returns
    -------
    pd.DataFrame
        A single DataFrame resulting from merging the input DataFrames based 
        on the specified index.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.quality import merge_frames_on_index
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'Key': ['K0', 'K1', 'K2']})
    >>> df2 = pd.DataFrame({'B': [4, 5, 6], 'Key': ['K0', 'K1', 'K2']})
    >>> merged_df = merge_frames_on_index(df1, df2, index_col='Key')
    >>> print(merged_df)

    Note: This function sets the specified column as the index for each 
    DataFrame if it is not already.
    If the column specified is not present in any of the DataFrames,
    a KeyError will be raised.
    """
    # Ensure all provided data are pandas DataFrames
    if not all(isinstance(df, pd.DataFrame) for df in data):
        raise TypeError("All data provided must be pandas DataFrames.")
        
    # Check if the specified index column exists in all DataFrames
    for df in data:
        if index_col not in df.columns:
            raise KeyError(f"The column '{index_col}' was not found in DataFrame.")

    # Set the specified column as the index for each DataFrame
    indexed_dfs = [df.set_index(index_col) for df in data]

    # Concatenate the DataFrames based on the provided parameters
    merged_df = pd.concat(indexed_dfs, axis=axis, join=join_type,
                          ignore_index=ignore_index, sort=sort)

    return merged_df

@isdf 
def check_missing_data(
    data: DataFrame, /, 
    view: bool = False,
    explode: Optional[Union[Tuple[float, ...], str]] = None,
    shadow: bool = True,
    startangle: int = 90,
    cmap: str = 'viridis',
    autopct: str = '%1.1f%%',
    verbose: int = 0
) -> DataFrame:
    """
    Check for missing data in a DataFrame and optionally visualize the 
    distribution of missing data with a pie chart.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to check for missing data.
    view : bool, optional
        If True, displays a pie chart visualization of the missing data 
        distribution.
    explode : tuple of float, or 'auto', optional
        - If a tuple, it should have a length matching the number of columns 
          with missing data, indicating how far from the center the slice 
          for that column will be.
        - If 'auto', the slice with the highest percentage of missing data will 
          be exploded. If the length does not match, an error is raised.
    shadow : bool, optional
        If True, draws a shadow beneath the pie chart.
    startangle : int, optional
        The starting angle of the pie chart. If greater than 360 degrees, 
        the value is adjusted using modulo operation.
    cmap : str, optional
        The colormap to use for the pie chart.
    autopct : str, optional
        String format for the percentage of each slice in the pie chart. 
    verbose : int, optional
        If set, prints messages about automatic adjustments.

    Returns
    -------
    missing_stats : pandas.DataFrame
        A DataFrame containing the count and percentage of missing data in 
        each column that has missing data.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import check_missing_data
    >>> # Create a sample DataFrame with missing values
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': [None, 2, 3, 4],
    ...     'C': [1, 2, 3, 4]
    ... })
    >>> missing_stats = check_missing_data(data, view=True, explode='auto',
                                           shadow=True, startangle=270, verbose=1)
    """
    def validate_autopct_format(autopct: str) -> bool:
        """
        Validates the autopct format string for matplotlib pie chart.

        Parameters
        ----------
        autopct : str
            The format string to validate.

        Returns
        -------
        bool
            True if the format string is valid, False otherwise.
        """
        # A regex pattern that matches strings like '%1.1f%%', '%1.2f%%', etc.
        # This pattern checks for the start of the string (%), optional flags,
        # optional width, a period, precision, the 'f' specifier, and ends with '%%'
        pattern = r'^%[0-9]*\.?[0-9]+f%%$'
        return bool(re.match(pattern, autopct))

    is_frame( data, df_only= True, raise_exception= True )
    missing_count = data.isnull().sum()
    missing_count = missing_count[missing_count > 0]
    missing_percentage = ((missing_count / len(data)) * 100).round(4)

    missing_stats = pd.DataFrame({'Count': missing_count,
                     'Percentage': missing_percentage})

    if view and not missing_count.empty:
        labels = missing_stats.index.tolist()
        sizes = missing_stats['Percentage'].values.tolist()
        if explode == 'auto':
            # Dynamically create explode data 
            explode = [0.1 if i == sizes.index(max(sizes)) else 0 
                       for i in range(len(sizes))]
        elif explode is not None:
            if len(explode) != len(sizes):
                raise ValueError(
                    f"The length of 'explode' ({len(explode)}) does not match "
                    f"the number of columns with missing data ({len(sizes)})."
                    " Set 'explode' to 'auto' to avoid this error.")
        
        if startangle > 360:
            startangle %= 360
            print(
            "Start angle is greater than 180 degrees. Using modulo "
            f"to adjust: startangle={startangle}") 
        if not validate_autopct_format(autopct):
            raise ValueError("`autopct` format is not valid. It should be a"
                             "  format string like '%1.1f%%'.")
        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, autopct=autopct,
               shadow=shadow, startangle=startangle, colors=plt.get_cmap(cmap)(
                   np.linspace(0, 1, len(labels))))
        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        ax.set_title('Missing Data Distribution')
        plt.show()
        
    if verbose: 
        if missing_stats.empty: 
            print("No missing data detected in the DataFrame."
                  " All columns are complete.")
          
        else:
            summary= ResultSummary(
                "MissingCheckResults", pad_keys="auto").add_results(
                missing_stats.to_dict(orient='index'))
            print(summary)

    return missing_stats

@Dataify(auto_columns=True)
def data_assistant(data: DataFrame, view: bool=False):
    """
    Performs an in-depth analysis of a pandas DataFrame, providing insights,
    identifying data quality issues, and suggesting corrective actions to
    optimize the data for statistical modeling and analysis. The function 
    generates a report that includes recommendations and possible actions 
    based on the analysis.

    The function performs a comprehensive analysis on the DataFrame, including
    checks for data integrity, descriptive statistics, correlation analysis, 
    and more, `data_assistant` provides recommendations for each identified 
    issue along with suggestions for modules and functions that could be 
    used to resolve these issues. In otherwords, function is meant to assist 
    users in understanding their data better and preparing it for further 
    modeling steps.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to be analyzed. This function expects a pandas DataFrame
        populated with data that can be numerically or categorically analyzed.

    view : bool, optional
        A boolean flag that, when set to True, enables the visualization of
        certain data aspects such as distribution plots and correlation matrices.
        Defaults to False, which means no visualizations are displayed.

    Returns
    -------
    None
        Function does not return any value; instead, it prints a detailed
        report directly to the console. The report includes sections on data
        integrity, descriptive statistics, and potential issues along with
        suggestions for data preprocessing steps.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import data_assistant
    >>> df = pd.DataFrame({
    ...     'Age': [25, 30, 35, 40, None],
    ...     'Salary': [50000, 60000, 70000, 80000, 90000],
    ...     'City': ['New York', 'Los Angeles', 'San Francisco', 'Houston', 'Seattle']
    ... })
    >>> data_assistant(df)

    Notes
    -----
    The `data_assistant` function is designed to assist in the preliminary analysis
    phase of data processing, offering a diagnostic view of the data's quality and
    structure. It is particularly useful for identifying and addressing common
    data issues before proceeding to more complex data modeling or analysis tasks.

    It is recommended to review the insights and apply the suggested transformations
    or corrections to ensure the data is optimally prepared for further analysis
    or machine learning modeling.

    See Also
    --------
    gofast.dataops.quality.inspect_data : A related function that provides a deeper
    dive into data inspection with more customizable features.
    
    """
    # Get the current datetime
    current_datetime = datetime.datetime.now()
    # Format the datetime object into a string in the specified format
    formatted_datetime = current_datetime.strftime("%d %B %Y %H:%M:%S")
    # Creating a dictionary with the formatted datetime string
    texts = {"Starting assistance...": formatted_datetime}
    
    # Initialize dictionaries to store sections of the report
    recommendations = {}
    helper_funcs = {} 
    
    # Checking for missing values
    texts ["1. Checking for missing values"]="Passed"
    if data.isnull().sum().sum() > 0:
        texts ["1. Checking for missing values"]="Failed"
        texts["   #  Found missing values in the dataset?"]='Yes'
        #print("Found missing values in the dataset.")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        texts[ "   #  Columns with missing values and counts"]= smart_format(
            [missing_data.name] if isinstance (
                missing_data, pd.Series ) else missing_data.columns ) 
        
        recommendations["1. Missing values "]= ( 
            "Missing data can lead to biased or invalid conclusions"
            " if not handled properly, as many statistical methods assume"
            " complete data. Consider imputing or dropping the missing values."
            " See helper functions for handling missing values."
            ) 
        helper_funcs["1. Missing values "]= ( 
            "Use: pandas.DataFrame.fillna(), sklearn.impute.SimpleImputer"
            " ~.tools.soft_imputer, ~.tools.one_click_prep, ~.dataops.check_missing_data"
            " ~.dataops.handle_missing_data, ~.transformers.MissingValueImputer and more..."
            )
    # Descriptive statistics
    texts ["2. Descriptive Statistics"]="Done"
    texts["   #  summary"]= format_iterable(data.describe())
    # Check for zero variance features
    texts ["3. Checking zero variance features"]="Passed"
    zero_var_cols = data.columns[data.nunique() == 1]
    if len(zero_var_cols) > 0:
        texts ["3. Checking zero variance features"]="Failed"
        texts ["   #  Found zero variance columns?"]="Yes"
        texts["   #  Zeros variances columns"]=( 
            f"{smart_format(zero_var_cols.tolist())}"
            )
        
        recommendations["3. Zero variances features"]=(
            "Zero variance features offer no useful information for modeling"
            " because they do not vary across observations. This means they"
            " cannot help in predicting the target variable or in distinguishing"
            " between different instances. Consider dropping them as they do not "
            " provide any information, redundant computation, model complexity..."
            )
        helper_funcs["3. Zero variances features"]= ( 
            "Use: pandas.DataFrame.drop(columns =<zero_var_cols>)")
        
    # Data types analysis
    texts["4. Data types summary"]="Done"
    if (data.dtypes == 'object').any():
        texts["   #  Summary types"]="Include string or mixed types"
        texts["   #  Non-numeric data types found?"]="Yes"
        texts["   #  Non-numeric features"]= smart_format( 
            data.select_dtypes( exclude=[np.number]).columns.tolist())
        
        recommendations ["4. Non-numeric data"]= (
            "Improper handling of non-numeric data can lead to misleading"
            " results or poor model performance. For instance, ordinal "
            " encoding of inherently nominal data can imply a nonexistent"
            " ordinal relationship, potentially misleading the model."  
            " Consider transforming into a numeric format through encoding"
            " techniques (like one-hot encoding, label encoding, or embedding)"
            " to be used in these models.")
        helper_funcs ["4. Non-numeric data"]=( 
            "Use: pandas.get_dummies(), sklearn.preprocessing.LabelEncoder"
            " ~.tools.soft_encoder, ~.transformers.CategoricalEncoder2"
            " ~.dataops.handle_categorical_features and more ..."
            ) 
        
    # Correlation analysis
    texts["5. Correlation analysis"]="Done"
    numeric_data = data.select_dtypes(include=[np.number])
    texts["   #  Numeric corr-feasible features"]=format_iterable(numeric_data)
    if numeric_data.empty: 
        texts["   #  Numeric corr-feasible features"]="No numeric features found."
    elif numeric_data.ndim > 1:
        exist_correlated = check_correlated_features(numeric_data)
        texts["   #  Correlation matrix review"]="Done"
        texts["   #  Are correlated features found?"] = "Yes" if exist_correlated else "No"

        if view: 
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.show()
            texts["   #  corr-matrix view"]="See displayed figure..."
        
        if exist_correlated: 
            recommendations["5. Correlated features"]= (
                "Highly correlated features can lead to multicollinearity in"
                " regression models, where it becomes difficult to estimate the"
                " relationship of each independent variable with the dependent"
                " variable due to redundancy. Review highly correlated variables"
                " as they might affect model performance due to multicollinearity."
                )
            helper_funcs ["5. Correlated features"]= ( 
                "Use: pandas.DataFrame.go_corr, ~.dataops.analyze_data_corr,"
                " ~.dataops.correlation_ops, ~.dataops.drop_correlated_features,"
                " ~.stats.descriptive.corr` and more ...")
    
    # Distribution analysis
    texts["6. Checking for potential outliers"]="Passed"
    skew_cols =[]
    for col in numeric_data.columns:
        if view: 
            plt.figure()
            sns.distplot(data[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.show()
           
        if data[col].skew() > 1.5 or data[col].skew() < -1.5:
            skew_value = data[col].skew()
            skew_cols.append ( (col, skew_value) ) 
            
    if view: 
        texts["   #  Distribution analysis view"]="See displayed figure..."
        
    if skew_cols : 
        texts["6. Checking for potential outliers"]="Failed"
        
        texts["   #  Outliers found?"]="Yes"
        texts["   #  Skewness columns"]=', '.join(
            [ f"{skew}-{val:.4f}" for skew, val in  skew_cols ]) 
        recommendations["6. Distribution ~ skewness"]= (
            "Skewness can distort the mean and standard deviation of the data."
            " Measures of skewed data do not accurately represent the center"
            " and variability of the data, potentially leading to misleading"
            " interpretations, poor model performance and unreliable predictions."
            " Consider transforming this data using logarithmic, square root,"
            " or box-cox transformations")
        helper_funcs ["6. Distribution ~ skewness" ]= ( 
            "Use: scipy.stats.boxcox, sklearn.preprocessing.PowerTransformer"
            " ~.dataops.handle_skew, ~.dataops.assess_outlier_impact"
            " pandas.DataFrame.go_skew, ~.stats.descriptive.skew and more ..."
            )
    # Duplicate rows
    texts["7. Duplicate analysis"]="Passed"
    if data.duplicated().sum() > 0:
        texts["7. Duplicate analysis"]="Failed"
        texts["   #  Found duplicate rows in the dataset?"]="Yes"
        texts["   #  Duplicated indices"]=smart_format(
            handle_duplicates(data, return_indices=True )
            ) 
        recommendations["7. Duplicate analysis"]=(
            "Duplicate entries can skew statistical calculations such as means,"
            " variances, and correlations, leading to biased or incorrect estimates"
            " Duplicate rows can lead to overfitting, particularly if the"
            " duplicates are only present in the training dataset and not in"
            " unseen data. Models might learn to overly rely on these repeated"
            " patterns, performing well on training data but poorly on new,"
            " unseen data. Consider reviewing and possibly removing duplicates."
            )
        helper_funcs ["7. Duplicate analysis"]=(
            "Use: pandas.DataFrame.drop_duplicates(),"
            " ~.tools.handle_duplicates and more ...")
        
    # Unique value check
    texts["8. Unique value check with threshold=10"]="Passed"
    
    unique_cols =check_unique_values(
        data, return_unique_cols=True, unique_threshold=10)
    if unique_cols: 
        texts["8. Unique value check with threshold=10"]="Failed"
        texts["   #  Found unique value column?"]="Yes"
        texts["   #  Uniques columns"]=smart_format(unique_cols)
        
        recommendations ["8. Unique identifiers"]= ( 
            "Unique identifiers typically do not possess any intrinsic"
            " predictive power because they are unique to each record."
            " Including these in predictive modeling can be misleading"
            " for algorithms, leading them to overfit the data without"
            " gaining any generalizable insights. Check if these columns"
            " should be treated as a categorical variables"
            )
        helper_funcs["8. Unique identifiers"]= ( 
            "Use: ~.dataops.handle_unique_identifiers,"
            " ~.transformers.BaseCategoricalEncoder,"
            " ~.transformers.CategoricalEncoder,"
            " ~.transformers.CategorizeFeatures', and more ..."
            )
    
    report_txt= {**texts}

    # Get the Note line from TW terminal width
    note_line = TW - (max(len(key) for key in report_txt.keys()) + 3)
    
    # In case both sections are empty, append a general note
    if len(recommendations) == 0 and len(helper_funcs) == 0:
        # Adding a general keynote to the report to address the absence
        # of recommendations and helper functions
        report_txt["KEYNOTE"] = "-" * note_line
        
        # Add a "TO DO" section with a general review note
        report_txt["TO DO"] = (
            "Review the provided insights. No further immediate actions or "
            "recommendations are required at this stage. For a more in-depth "
            "inspection or specific queries, consider using detailed analysis "
            "tools such as `gofast.dataops.inspect_data`. These tools offer "
            "advanced capabilities for attribute queries and can help you gain "
            "deeper insights into your data. Additionally, exploring tools like "
            " `gofast.dataops.verify_data_integrity`, `gofast.dataops.correlation_ops`"
            " and `gofast.dataops.handle_categorical_features` can further"
            " enhance your data analysis and preprocessing workflows."
        )
    else:
        # Provide actionable steps when there are recommendations and/or helper functions
        report_txt["NOTE"] = "-" * note_line
        # Add a "TO DO" section with specific guidance
        report_txt["TO DO"] = (
            "Please review the insights and recommendations provided to"
            " effectively prepare your data for modeling. For a more"
            " comprehensive analysis, make use of the tools available in the"
            " `gofast.dataops` module. Consider utilizing `gofast.datops.verify_data_integrity`"
            " to ensure accuracy and consistency, and `gofast.dataops.audit_data`"
            " for auditing your data. Additionally, you can explore further"
            " capabilities by using `gofast.explore('dataops.<module>')`."
            " Make sure to set `gofast.config.PUBLIC=True` first."
        )

    assistance_reports =[]
    assistance_reports.append( ReportFactory(
        "Data Assistant Report").add_mixed_types(
            report_txt, table_width= TW  )
        )
    if len(recommendations) > 1: 
        recommendations_report = ReportFactory("Recommendations").add_mixed_types(
            recommendations, table_width= TW  )
        assistance_reports.append(recommendations_report)
    if len(helper_funcs) > 1:
        helper_tools_report = ReportFactory("Helper tools [~:gofast]").add_mixed_types(
            helper_funcs, table_width= TW  )
        assistance_reports.append(helper_tools_report)

    assemble_reports( *assistance_reports, display=True)
    
    
@Dataify(auto_columns= True, ignore_mismatch=True)  
def check_unique_values(
    data: DataFrame, 
    columns: Optional[List[str]] = None, 
    unique_threshold: Union[int, str] = 1,
    only_unique: bool = False,
    include_nan: bool = False,
    return_unique_cols: bool = False,
    verbose: bool = False
) -> Union[DataFrame,Series, List[str]]:
    """
    Checks for unique values in a pandas DataFrame and provides detailed 
    analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to analyze for unique values.
        
    columns : list of str, optional
        List of column names to check. If `None`, all columns are checked.
        
    unique_threshold : int or str, default 1
        Threshold for defining unique values. If an integer, it specifies 
        the minimum count of unique values for consideration. If 'auto', 
        the function automatically determines uniqueness:
        - Integer columns: Count repetitive values.
        - Float columns with integer-like values (e.g., 1.0, 2.0): Count 
          unique values.
        - Float columns with non-integer-like values: Ignore.
        - Categorical columns: Apply standard uniqueness check.
        
    only_unique : bool, default False
        If `True`, returns only the columns with unique values.
        
    include_nan : bool, default False
        If `True`, includes NaN values in the uniqueness check.
        
    return_unique_cols : bool, default False
        If `True`, returns a list of columns that contain unique values.
        
    verbose : bool, default False
        If `True`, prints detailed output including counts of unique 
        values per column and columns above the threshold.

    Returns
    -------
    Union[pandas.DataFrame, pd.Series, list of str]
        A DataFrame or Series with the count of unique values per column, 
        or a list of columns with unique values, depending on the 
        parameters.

    Notes
    -----
    This function analyzes the uniqueness of values in a DataFrame. The 
    threshold for uniqueness can be set manually or determined 
    automatically based on data types.

    The uniqueness check for 'auto' threshold is performed as follows:
    
    .. math::
        \text{if column is integer: count repetitive values}
    
    .. math::
        \text{if column is float and all values are integer-like: count unique values}
    
    .. math::
        \text{if column is float and not integer-like: ignore}
    
    .. math::
        \text{if column is categorical: apply standard uniqueness check}

    Examples
    --------
    >>> from gofast.dataops.quality import check_unique_values
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['x', 'y', 'y', 'z'], 'C': [1.0, 2.0, 2.0, 3.0]})
    >>> check_unique_values(data, verbose=True)
    Unique value counts per column:
    A    3
    B    3
    C    3
    dtype: int64
    >>> check_unique_values(data, unique_threshold='auto', verbose=True)
    Unique value counts per column:
    A    3
    B    3
    C    3
    dtype: int64
    >>> check_unique_values(data, only_unique=True, unique_threshold='auto')
       A  B  C
    0  1  x  1.0
    1  2  y  2.0
    2  2  y  2.0
    3  3  z  3.0
    >>> check_unique_values(data, return_unique_cols=True, unique_threshold='auto')
    ['A', 'B', 'C']

    See Also
    --------
    pandas.DataFrame.nunique : Count unique values in DataFrame columns.
    pandas.Series.nunique : Count unique values in Series.
    pandas.Series.value_counts : Count occurrences of unique values in Series.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in Python. 
           Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
    unique_counter = {}
    summary = ResultSummary("UniqueValues", pad_keys="auto", flatten_nested_dicts=False)
    
    if columns is None:
        columns = data.columns.tolist()

    unique_counts = pd.Series(index=columns, dtype=int)
    
    for col in columns:
        if pd.api.types.is_integer_dtype(data[col]):
            unique_counts[col] = data[col].value_counts().lt(2).sum()
        elif pd.api.types.is_float_dtype(data[col]):
            float_zero_remainder = all(data[col].dropna().mod(1) == 0)
            if float_zero_remainder:
                unique_counts[col] = data[col].value_counts().lt(2).sum()
            else:
                unique_counts[col] = 0
        elif pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            unique_counts[col] = data[col].nunique(dropna=not include_nan)
        else:
            unique_counts[col] = 0
    
    if verbose:
        unique_counter["counts_per_columns"] = unique_counts.to_dict()

    if unique_threshold == 'auto':
        unique_columns = unique_counts[unique_counts > 0].index
    elif isinstance(unique_threshold, (int, float)):
        unique_columns = unique_counts[unique_counts >= unique_threshold].index
    else:
        raise ValueError("unique_threshold must be 'auto' or a numeric value.")
    
    if only_unique:
        unique_data = data[unique_columns]
        if verbose:
            unique_counter["columns_above_threshold"] = unique_columns.tolist()
            summary.add_results(unique_counter)
            print(summary)
        return unique_data

    if return_unique_cols:
        if verbose:
            unique_counter["columns"] = unique_columns.tolist()
            summary.add_results(unique_counter)
            print(summary)
        return unique_columns.tolist()
    
    return unique_counts

@Dataify(auto_columns= True, ignore_mismatch=True, prefix="feature_")   
def check_correlated_features(
    data, /, threshold: float=0.8, 
    method: str | callable ='pearson',
    return_correlated_pairs: bool=False,
    min_periods: int=1,
    view: bool=False, 
    annot: bool=True, 
    cmap: str="viridis", 
    fig_size: tuple =(12, 10)
    ):
    """
    Check for correlated features in a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the features to check for correlation.
    threshold : float, optional, default=0.8
        The correlation coefficient threshold above which features 
        are considered correlated. Values range between -1 and 1.
    method : str or callable, optional, default='pearson'
        The correlation method to use. One of {'pearson', 'spearman', 
        'kendall'} or a callable. This determines the type of 
        correlation calculation to use.
    return_correlated_pairs : bool, optional, default=False
        If True, returns a list of correlated pairs of features.
    min_periods : int, optional, default=1
        Minimum number of observations required per pair of columns to 
        have a valid result. Only used if the method is 'pearson' or 
        'spearman'.
    view : bool, optional, default=False
        If True, plots a heatmap of the correlation matrix.
    annot : bool, optional, default=True
        If True, annotates the heatmap with the correlation coefficients.
    cmap : str, optional, default='viridis'
        Colormap to use for the heatmap.
    fig_size : tuple, optional, default=(12, 10)
        Size of the figure for the heatmap.

    Returns
    -------
    correlated_pairs : list of tuple, optional
        A list of tuples containing pairs of correlated features and 
        their correlation coefficient. Returned only if 
        `return_correlated_pairs` is ``True``.

    Raises
    ------
    ValueError
        If the input data is not a pandas DataFrame.

    Notes
    -----
    The correlation coefficient, :math:`r`, measures the strength and 
    direction of a linear relationship between two features. It is 
    defined as:

    .. math::
        r = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}
        {\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}
        \\sqrt{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}}

    where :math:`x_i` and :math:`y_i` are individual sample points, and 
    :math:`\\bar{x}` and :math:`\\bar{y}` are the mean values of the 
    samples.

    Highly correlated features (with an absolute correlation coefficient 
    greater than the specified `threshold`) can introduce multicollinearity 
    into machine learning models, leading to unreliable model coefficients.

    Examples
    --------
    Generate example data and check for correlated features:

    >>> from gofast.dataops.quality import check_correlated_features
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.rand(100, 5), 
    ...                     columns=['A', 'B', 'C', 'D', 'E'])
    >>> correlated_features = check_correlated_features(data, 
    ...                                                 threshold=0.75, 
    ...                                                 view=True)
    >>> print("Correlated Features:", correlated_features)

    See Also
    --------
    gofast.dataops.analyze_data_corr :
        Computes the correlation matrix for specified columns in a pandas 
        DataFrame and optionally visualizes it using a heatmap.
    gofast.dataops.correlation_ops :
        Performs correlation analysis on a given DataFrame and classifies 
        the correlations into specified categories.
    gofast.dataops.drop_correlated_features :
        Analyzes and removes highly correlated features from a DataFrame 
        to reduce multicollinearity, improving the reliability and performance 
        of subsequent statistical models.

    References
    ----------
    .. [1] Pearson, K. (1895). "Note on Regression and Inheritance in the 
           Case of Two Parents". Proceedings of the Royal Society of London. 
           58: 240–242.
    .. [2] Spearman, C. (1904). "The Proof and Measurement of Association 
           between Two Things". American Journal of Psychology. 15 (1): 72–101.
    .. [3] Kendall, M. G. (1938). "A New Measure of Rank Correlation". 
           Biometrika. 30 (1/2): 81–89.
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr(method=method, min_periods=min_periods)

    # Find pairs of correlated features
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j],
                        round(corr_matrix.iloc[i, j], 6))
                correlated_pairs.append(pair)

    if view:
        # Plot a heatmap of the correlation matrix
        plt.figure(figsize=fig_size)
        sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap=cmap, 
                    vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix ({method} method)')
        plt.show()

    if return_correlated_pairs:
        return correlated_pairs
    
    return bool(correlated_pairs)

@Dataify(auto_columns= True , ignore_mismatch=True, prefix="var_")
def analyze_data_corr(
    data: DataFrame, 
    columns: Optional[ List[str]]=None, 
    method: str | Callable [[ArrayLike, ArrayLike], float]='pearson', 
    min_periods: int=1, 
    min_corr: float =0.5, 
    high_corr: float=0.8, 
    interpret: bool =False, 
    hide_diag: bool =True,
    no_corr_placeholder: str='...',  
    autofit: bool=True, 
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (8, 8)
    ):
    """
    Computes the correlation matrix for specified columns in a pandas DataFrame
    and optionally visualizes it using a heatmap. 
    
    This function can also symbolically represent correlation values and 
    selectively hide diagonal elements in the visualization and interpretability.

    Parameters
    ----------
    data : DataFrame
        The DataFrame from which to compute the correlation matrix.
    columns : Optional[List[str]], optional
        Specific columns to consider for the correlation calculation. If None, all
        numeric columns are used. Default is None.
    method : str | Callable[[ArrayLike, ArrayLike], float], optional
        Method to use for computing the correlation:
        - 'pearson' : Pearson correlation coefficient
        - 'kendall' : Kendall Tau correlation coefficient
        - 'spearman' : Spearman rank correlation
        - Custom function : a callable with input of two 1d ndarrays and 
        returning a float.
        Default is 'pearson'.
    min_periods : int, optional
        Minimum number of observations required per pair of columns to have 
        a valid result. 
        Default is 1.
    min_corr : float, optional
        The minimum threshold for correlations to be noted if using symbols. 
        Default is 0.5.
    high_corr : float, optional
        Threshold above which correlations are considered high, relevant if
        `use_symbols` is True. Default is 0.8.
    interpret : bool, optional
        Whether to use symbolic representation ('++', '--', '+-') for interpretation 
        instead of numeric values where: 
        - ``'++'``: Represents a strong positive relationship.
        - ``'--'``: Represents a strong negative relationship.
        - ``'-+'``: Represents a moderate relationship.
        - ``'o'``: Used exclusively for diagonal elements, typically representing
          a perfect relationship in correlation matrices (value of 1.0).
        Default is False.
    hide_diag : bool, optional
        If True, diagonal values in the correlation matrix visualization are hidden.
        Default is True.
    no_corr_placeholder : str, optional
        Text to display for correlation values below `min_corr`. Default is '...'.
    autofit : bool, optional
        If True, adjusts the column widths and the number of visible rows
        based on the DataFrame's content and available display size. 
        When `autofit` is ``True``,`no_corr_placeholder` takes the empty value 
        for non-correlated items.  Default is True. 
    view : bool, optional
        If True, displays a heatmap of the correlation matrix using matplotlib and
        seaborn. Default is False.
    cmap : str, optional
        The colormap for the heatmap visualization. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Dimensions of the figure that displays the heatmap. Default is (8, 8).

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the correlation matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import analyze_data_corr
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [2, 3, 4, 5, 6]
    ... })
    >>> corr_summary = analyze_data_corr(data, view=True)
    >>> corr_summary
    <Summary: Populated. Use print() to see the contents.>

    >>> print(corr_summary) 
          Correlation Table       
    ==============================
            A        B        C   
      ----------------------------
    A |           -1.0000   1.0000
    B |  -1.0000           -1.0000
    C |   1.0000  -1.0000         
    ==============================
    
    >>> corr_summary.corr_matrix
         A    B    C
    A  1.0 -1.0  1.0
    B -1.0  1.0 -1.0
    C  1.0 -1.0  1.0
    
    >>> corr_summary = analyze_data_corr(data, view=False, interpret=True)
    >>> print(corr_summary) 
      Correlation Table  
    =====================
          A     B     C  
      -------------------
    A |        --    ++  
    B |  --          --  
    C |  ++    --        
    =====================

    .....................
    Legend : ++: Strong
             positive,
             --: Strong
             negative,
             -+: Moderate
    .....................
    
    Notes
    -----
    If `view` is True, the function requires a matplotlib backend that supports
    interactivity, typically within a Jupyter notebook or a Python environment
    configured for GUI operations.
    """
    numeric_df = data.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric data found in the DataFrame.")
        
    # Compute the correlation matrix
    method = method.lower() if isinstance (method, str) else method  
    correlation_matrix = numeric_df.corr(method = method, min_periods = min_periods)
 
    # Check if the user wants to view the heatmap
    if view:
        plt.figure(figsize=fig_size)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=.5)
        plt.title('Heatmap of Correlation Matrix')
        plt.show()
        
    summary = Summary(corr_matrix=correlation_matrix, descriptor="CorrSummary" )
    summary.add_data_corr(
        correlation_matrix, 
        min_corr=assert_ratio( min_corr or .5 , bounds=(0, 1)),
        high_corr=assert_ratio(high_corr, bounds=(0, 1)), 
        use_symbols= interpret, 
        hide_diag= hide_diag,
        precomputed=True,
        no_corr_placeholder=str(no_corr_placeholder), 
        autofit=autofit, 
        )
    return summary

def correlation_ops(
    data: DataFrame, 
    corr_type:str='all', 
    min_corr: float=0.5, 
    high_corr: float=0.8, 
    method: str| Callable[[ArrayLike, ArrayLike], float]='pearson', 
    min_periods: int=1, 
    display_corrtable: bool=False, 
    **corr_kws
    ):
    """
    Performs correlation analysis on a given DataFrame and classifies the 
    correlations into specified categories. Depending on the `correlation_type`,
    this function can categorize correlations as strong positive, strong negative,
    or moderate. It can also display the correlation matrix and returns a 
    formatted report of the findings.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame on which to perform the correlation analysis.
    corr_type : str, optional
        The type of correlations to consider in the analysis. Valid options
        are 'all', 'strong only', 'strong positive', 'strong negative', and
        'moderate'. Defaults to 'all'.
    min_corr : float, optional
        The minimum correlation value to consider for moderate correlations.
        Defaults to 0.5.
    high_corr : float, optional
        The threshold above which correlations are considered strong. Defaults
        to 0.8.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Method of correlation:
        - 'pearson' : standard correlation coefficient
        - 'kendall' : Kendall Tau correlation coefficient
        - 'spearman' : Spearman rank correlation
        Defaults to 'pearson'.
    min_periods : int, optional
        Minimum number of observations required per pair of columns to have a
        valid result. Defaults to 1.
    display_corrtable : bool, optional
        If True, prints the correlation matrix to the console. Defaults to False.
    **corr_kws : dict
        Additional keyword arguments to be passed to the correlation function
        :func:`analyze_data_corr`. 

    Returns
    -------
    MultiFrameFormatter or ReportFactory
        Depending on the analysis results, returns either a formatted report of 
        correlation pairs or a message about the absence of significant 
        correlations.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import correlation_ops
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 2, 3, 4, 4],
    ...     'C': [5, 3, 2, 1, 1]
    ... })
    >>> result = correlation_ops(data, correlation_type='strong positive')
    >>> print(result)

    Notes
    -----
    The correlation threshold parameters (`min_corr` and `high_corr`) help in 
    fine-tuning which correlations are reported based on their strength. This 
    is particularly useful in datasets with many variables, where focusing on 
    highly correlated pairs is often more insightful.

    See Also
    --------
    pandas.DataFrame.corr : Compute pairwise correlation of columns.
    MultiFrameFormatter : A custom formatter for displaying correlated pairs.
    analyze_data_corr : A predefined function for computing and summarizing 
                      correlations.
    """
    # Compute the correlation matrix using a predefined analysis function
    corr_kws = filter_valid_kwargs(analyze_data_corr, corr_kws)
    corr_summary = analyze_data_corr(
        data, method=method, min_periods=min_periods, **corr_kws)
    
    if display_corrtable:
        print(corr_summary)
    
    corr_matrix = corr_summary.corr_matrix
    # validate correlation_type parameter 
    corr_type = parameter_validator('correlation_type',
        ["all","strong only", "strong positive", "strong negative", "moderate" ]
        )(corr_type)
    # Storage for correlation pairs
    strong_positives, strong_negatives, moderates = [], [], []

    # Evaluate each cell in the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if ( 
                    corr_type in ['all', 'strong only', 'strong positive'] 
                    and corr_value >= high_corr ) :
                strong_positives.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_value)
                    )
            if ( 
                    corr_type in ['all', 'strong only', 'strong negative'] 
                    and corr_value <= -high_corr ):
                strong_negatives.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_value)
                    )
            if ( 
                    corr_type in ['all', 'moderate'] 
                    and min_corr <= abs(corr_value) < high_corr ) :
                moderates.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                  corr_value))

    # Prepare DataFrames for each category
    
    dfs = {}
    if strong_positives:
        dfs['Strong Positives'] = pd.DataFrame(
            strong_positives, columns=['Feature 1', 'Feature 2', 'Correlation'])
    if strong_negatives:
        dfs['Strong Negatives'] = pd.DataFrame(
            strong_negatives, columns=['Feature 1', 'Feature 2', 'Correlation'])
    if moderates:
        dfs['Moderates'] = pd.DataFrame(
            moderates, columns=['Feature 1', 'Feature 2', 'Correlation'])

    # Formatting the output with MultiFrameFormatter if needed
    if dfs:
        new_dfs = {to_snake_case (k): v for k, v in dfs.items()} 
        formatted_report = MultiFrameFormatter(
            list(dfs.keys()), 
            descriptor="CorrelationOps", 
            max_cols=5, max_rows ='auto', 
            ).add_dfs(*new_dfs.values())
        formatted_report.correlation_types = list(new_dfs.keys()) 
        formatted_report.correlated_pairs = _make_correlation_pairs(new_dfs)
        for key, value in new_dfs.items(): 
            setattr (formatted_report, key, value)
        
        return formatted_report
    else:
        insights=ReportFactory(title=f"Correlation Type: {corr_type}",
            descriptor="CorrelationOps" ).add_recommendations(
            (
            "No significant correlations detected in the provided dataset. "
            "This may indicate that data variables act independently of each other "
            "within the specified thresholds. Consider adjusting the correlation "
            "thresholds or analyzing the data for other patterns."
            ), 
            keys= 'Actionable Insight', max_char_text= TW
            )
        return insights
    
def _make_correlation_pairs(dict_of_dfs):
    """
    Converts a dictionary of dataframes containing correlation data into
    a dictionary of tuples.
    
    Parameters
    ----------
    dict_of_dfs : dict
        A dictionary where keys are categories (e.g., 'strong_positives') and values are 
        pandas DataFrames with columns ['Feature 1', 'Feature 2', 'Correlation'].
    
    Returns
    -------
    dict
        A dictionary where keys are the same as input dictionary and values 
        are lists of tuples (feature1, feature2, correlation_value).
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import _make_correlation_pairs
    >>> dict_of_dfs = {
    ...     'strong_positives': pd.DataFrame({
    ...         'Feature 1': ['A'],
    ...         'Feature 2': ['B'],
    ...         'Correlation': [0.948683]
    ...     }),
    ...     'strong_negatives': pd.DataFrame({
    ...         'Feature 1': ['A', 'B'],
    ...         'Feature 2': ['C', 'C'],
    ...         'Correlation': [-0.944911, -0.896421]
    ...     })
    ... }
    >>> result = _make_correlation_pairs(dict_of_dfs)
    >>> print(result)
    {'strong_positives': [('A', 'B', 0.948683)], 
     'strong_negatives': [('A', 'C', -0.944911), ('B', 'C', -0.896421)]}
    """
    result = {}
    
    for key, df in dict_of_dfs.items():
        # Convert dataframe to dictionary with 'tight' format
        df_dict = df.to_dict('tight')
        # Create list of tuples from the 'data' key of the tight dictionary
        correlation_pairs = [(row[0], row[1], row[2]) for row in df_dict['data']]
        
        result[key] = correlation_pairs
    
    return result

@Dataify (auto_columns=True)  
def drop_correlated_features(
    data: DataFrame, 
    method: str | Callable[[ArrayLike, ArrayLike], float] = 'pearson', 
    threshold: float = 0.8, 
    display_corrtable: bool = False, 
    strategy: Optional[str] = None, 
    corr_type: str = 'all', 
    min_corr: Optional[float] = None, 
    use_default: bool = False, 
    **corr_kws
    ):
    """
    Analyzes and removes highly correlated features from a DataFrame to reduce 
    multicollinearity, improving the reliability and performance of subsequent 
    statistical models. This function allows for customization of the correlation 
    computation method and the threshold for feature removal.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame from which correlated features are to be removed.
    method : {'pearson', 'kendall', 'spearman'} or callable, optional
        Method of correlation to be used. The method can be one of 'pearson' 
        (default), 'kendall', 'spearman', or a callable with the signature 
        Callable[[ArrayLike, ArrayLike], float] providing a custom correlation 
        computation. The default is 'pearson', which assesses linear 
        relationships.
    threshold : float, optional
        The correlation coefficient threshold above which one of the features 
        in a pair will be removed. Defaults to 0.8, where features with a 
        correlation coefficient higher than this value are considered highly 
        correlated.
    display_corrtable : bool, optional
        If set to True, the correlation matrix is printed before removing 
        correlated features. This can be useful for visualization and manual 
        review of the correlation values. Defaults to False.
    strategy : {'both', 'first', 'last'}, optional
        Strategy for dropping correlated features. 'both' drops all correlated 
        features, 'first' drops the first feature in each correlated pair, and 
        'last' drops the last feature in each correlated pair.
        Default is 'both'.
    corr_type : {'negative', 'positive', 'all'}, optional
        Type of correlation to consider when dropping features. 'negative' drops 
        only negatively correlated features, 'positive' drops only positively 
        correlated features, and 'both' drops features with absolute correlation 
        values above the threshold regardless of sign. Default is 'both'.
    min_corr : float, optional
        The minimum correlation value for identifying moderate correlations. 
        This parameter is used when `corr_type` is set to 'moderate'. It 
        specifies the lower bound of the correlation range considered 
        "moderate". The value must be between 0 and 1.

        .. math::
            \text{min_corr} \in [0, 1]

    use_default : bool, default=False
        If True, the function uses default values for defining moderate 
        correlations. The default range is from 0.5 to the specified `threshold`.

        When this parameter is set to True:
    
        .. math::
            \text{min_corr} = 0.5
    
        .. math::
            \text{max_corr} = \text{threshold}
    
        This means moderate correlations are considered to be within the range 
        of 0.5 to the provided `threshold` value. If `use_default` is False, 
        both `min_corr` and `max_corr` (through `threshold`) must be provided 
        and validated to ensure they lie within the range of [0, 1].
    **corr_kws : dict
        Additional keyword arguments to be passed to the 
        :func:`analyze_data_corr` correlation function.

    Returns
    -------
    pandas.DataFrame
        Returns a DataFrame with the highly correlated features removed based 
        on the specified threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import drop_correlated_features
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 2, 3, 4, 4],
    ...     'C': [5, 3, 2, 1, 1]
    ... })
    >>> print(data.corr())
    >>> reduced_data = drop_correlated_features(
    ...     data, threshold=0.8, strategy='first')
    >>> print(reduced_data)
    
    If you want to use custom moderate correlation values, you can specify 
    both `min_corr` and `threshold`:

    >>> drop_correlated_features(data, corr_type='moderate', min_corr=0.3,
    ...                          threshold=0.7)

    If you prefer to use default moderate correlation values, set 
    `use_default` to True:

    >>> drop_correlated_features(data, corr_type='moderate', use_default=True)

    Notes
    -----
    Removing correlated features is a common preprocessing step to avoid 
    multicollinearity, which can distort the estimates of a model's parameters 
    and affect the interpretability of variable importance. This function 
    is particularly useful in the preprocessing steps for statistical modeling 
    and machine learning.

    The choice of correlation method, threshold, and strategy should be guided 
    by specific analytical needs and the nature of the dataset. Lower thresholds 
    increase the number of features removed, potentially simplifying the model but 
    at the risk of losing important information.

    See Also
    --------
    pandas.DataFrame.corr : Method to compute pairwise correlation of columns.
    gofast.dataops.quality.analyze_data_corr: Function to analyze correlations 
    with more detailed options and outputs.
    """
    # Validate parameters
    strategy = strategy or 'both'
    strategy = parameter_validator(
        "strategy", target_strs={"both", "first", "last"})(strategy)
    corr_type = parameter_validator(
        "correlation_type `corr_type`",
        target_strs={"negative", "positive", "moderate", "all"})(corr_type)
  
    # Compute the correlation matrix using a predefined analysis function
    corr_kws = filter_valid_kwargs(analyze_data_corr, corr_kws)
    corr_summary = analyze_data_corr(
        data, method=method, high_corr=threshold, 
        min_corr=min_corr, **corr_kws
    )
    if display_corrtable:
        print(corr_summary)
        
    # Compute the correlation matrix based on kind
    if corr_type == 'negative':
        corr_matrix = corr_summary.corr_matrix.where(corr_summary.corr_matrix < 0)
    elif corr_type == 'positive':
        corr_matrix = corr_summary.corr_matrix.where(corr_summary.corr_matrix > 0)
    elif corr_type == 'moderate': 
        min_corr, threshold = _check_moderate_correlation(
            min_corr=min_corr, max_corr=threshold, use_default=use_default)
        corr_matrix = corr_summary.corr_matrix.where(
            (corr_summary.corr_matrix >= min_corr) & (corr_summary.corr_matrix <= threshold)
        )
    else:  # 'both'
        corr_matrix = corr_summary.corr_matrix

    corr_matrix = corr_matrix.abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop based on the threshold and strategy 
    to_drop = _drop_correlated_features(upper, threshold, strategy)

    # Drop the identified columns and return the reduced DataFrame
    df_reduced = data.drop(to_drop, axis=1)
    
    if df_reduced.empty: # mean that data has feature correlated 
        c=f"{corr_type.title()} c" if corr_type in {
            "positive", 'negative'} else "C"
        info=ReportFactory(
            title=f"{c}orrelation Drop Strategy: {strategy}",
            descriptor="CorrDropOps" ).add_recommendations(
            (
           "All features are highly correlated and have been dropped. "
            "Consider adjusting the strategy to 'first' or 'last' to retain "
            "some features or re-evaluate the correlation threshold."
            ), 
            keys= 'Recommendation', max_char_text= TW
            )
        print(info)

    return df_reduced


def _check_moderate_correlation(
    min_corr=None, max_corr=None, use_default=False
):
    """
    Checks and returns the correlation range for moderate correlations.

    Parameters
    ----------
    min_corr : float, optional
        The minimum correlation value. Must be between 0 and 1.
        
    max_corr : float, optional
        The maximum correlation value (threshold). Must be between 0 and 1.
        
    use_default : bool, default=False
        If True, returns the default moderate correlation range (0.5, 0.8).

    Returns
    -------
    tuple
        A tuple containing:
        - float: The minimum correlation value.
        - float: The maximum correlation value.

    Raises
    ------
    ValueError
        If `min_corr` or `max_corr` are not provided when `use_default` is False.
        If `min_corr` or `max_corr` are out of bounds.
        If `min_corr` is not less than `max_corr`.

    Examples
    --------
    >>> check_moderate_correlation(use_default=True)
    (0.5, 0.8)
    >>> check_moderate_correlation(min_corr=0.3, max_corr=0.7)
    (0.3, 0.7)
    >>> check_moderate_correlation(min_corr=0.9, max_corr=0.8)
    Traceback (most recent call last):
        ...
    ValueError: min_corr must be less than max_corr.
    """

    if use_default:
        return 0.5, 0.8

    if min_corr is None or max_corr is None:
        raise ValueError(
            "When 'use_default' is False, both 'min_corr' and 'max_corr'"
            " (through `threshold`) must be provided."
        )

    # Validate min_corr and max_corr are within the range [0, 1]
    min_corr = validate_ratio(min_corr, bounds=(0, 1), param_name="min_corr")
    max_corr = validate_ratio(max_corr, bounds=(0, 1), 
                              param_name="max_corr (threshold)")

    # Ensure min_corr is less than max_corr
    if min_corr >= max_corr:
        raise ValueError(f"min_corr {min_corr} must be less than"
                         f" max_corr (threshold){max_corr}.")

    return min_corr, max_corr


def _rearrange_corr_features(
        corr_matrix: pd.DataFrame, col: str, threshold: float) -> tuple:
    """
    Rearrange the correlated features based on their position in the 
    correlation matrix.
    
    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        The correlation matrix.
    col : str
        The column name to rearrange.
    threshold : float
        The correlation coefficient threshold above which one of the features 
        in a pair will be removed.
    
    Returns
    -------
    tuple
        A tuple of correlated feature names arranged by their positions in
        the correlation matrix.
    """
    col_index = list(corr_matrix.columns).index(col)
    pair_feature = list(corr_matrix[col][corr_matrix[col].abs() > threshold].index)
    pair_index = list(corr_matrix.columns).index(pair_feature[0])
    
    if col_index > pair_index:
        return (pair_feature[0], col)
    return (col, pair_feature[0])

def _drop_correlated_features(
    corr_matrix: pd.DataFrame, threshold: float, strategy: str
) -> set:
    """
    Identify columns to drop based on the correlation matrix, threshold, and strategy.

    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        The correlation matrix.
    threshold : float
        The correlation coefficient threshold above which one of the features
        in a pair will be removed.
    strategy : {'both', 'first', 'last'}
        Strategy for dropping correlated features. 'both' drops all correlated
        features, 'first' drops the first feature in each correlated pair, and 
        'last' drops the last feature in each correlated pair.

    Returns
    -------
    set
        A set of column names to drop.
    """
    correlations_pairs = []
    for column in corr_matrix.columns:
        if any(corr_matrix[column].abs() > threshold):
            pair_features = _rearrange_corr_features(corr_matrix, column, threshold)
            correlations_pairs.append(pair_features)

    if strategy == 'both':
        to_drop = set([feature for pair in correlations_pairs for feature in pair])
    elif strategy == 'first':
        to_drop = set([a for a, _ in correlations_pairs])
    elif strategy == 'last':
        to_drop = set([b for _, b in correlations_pairs])
    
    return to_drop

@Dataify (auto_columns=True)
def handle_skew(
    data: DataFrame,
    method: str = 'log', 
    view: bool = False,
    fig_size: Tuple[int, int] = (12, 8)
    ):
    """
    Applies a specified transformation to numeric columns in the DataFrame 
    to correct for skewness. This function supports logarithmic, square root,
    and Box-Cox transformations, helping to normalize data distributions and 
    improve the performance of many statistical models and machine learning 
    algorithms.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing numeric data that may exhibit skewness.
    method : {'log', 'sqrt', 'box-cox'}, optional
        The method of transformation to apply:
        - 'log' : Applies the natural logarithm transformation. Suitable for data
                  that is positively skewed. Cannot be applied to zero or 
                  negative values.
        - 'sqrt': Applies the square root transformation. Suitable for reducing 
                  moderate skewness. Cannot be applied to negative values.
        - 'box-cox': Applies the Box-Cox transformation which can handle broader
                     ranges and types of skewness but requires all data points to be
                     positive. If any data points are not positive, a generalized
                     Yeo-Johnson transformation (handled internally) is applied
                     instead.
        Default is 'log'.
        
    view : bool, optional
        If True, visualizes the distribution of the original and transformed
        data using both box plots and violin plots. The visualization helps
        to compare the effects of the transformation on data skewness directly.
        The top row of subplots displays box plots of the original data,
        while the bottom row shows violin plots of the transformed data,
        providing a comprehensive view of the distribution changes.
        Default is False.
    
    fig_size : tuple of int, optional
        Specifies the dimensions of the figure that displays the plots, given
        as a tuple (width, height). This parameter allows customization of the
        plot size to ensure that the visualizations are clear and appropriately
        scaled to the user's display settings or presentation needs.
        The default size is set to (12, 8), which offers a balanced display
        for typical datasets but can be adjusted to accommodate larger datasets
        or different display resolutions.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with transformed data to address skewness in numeric 
        columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import handle_skew
    >>> data = pd.DataFrame({
    ...     'A': [0.1, 1.5, 3.0, 4.5, 10.0],
    ...     'B': [-1, 2, 5, 7, 9]
    ... })
    >>> transformed_data = handle_skew(data, method='log')
    >>> print(transformed_data)

    Notes
    -----
    Skewness in a dataset can lead to biases in machine learning and statistical 
    models, especially those that assume normality of the data distribution. By 
    transforming skewed data, this function helps mitigate such issues, enhancing 
    model accuracy and robustness.

    It is important to understand the nature of your data and the requirements of
    your specific models when choosing a transformation method. Some methods, like
    'log', cannot handle zero or negative values without adjustments.

    See Also
    --------
    scipy.stats.boxcox : For more details on the Box-Cox transformation.
    sklearn.preprocessing.PowerTransformer : Implements both the Box-Cox transformation
                                             and the Yeo-Johnson transformation.
    """
    from sklearn.preprocessing import PowerTransformer

    if data.select_dtypes(include=['int64', 'float64']).empty:
        warnings.warn(
            "No numeric data detected. The function will proceed without "
            "modifying the DataFrame, as transformations applicable to "
            "skew correction require numeric data types."
        )
        return data
    else: 
        original_data = data.select_dtypes(include=['int64', 'float64']).copy()
        
    # Validate and apply the chosen method to each numeric column
    method = parameter_validator('method', ["log", "sqrt", "box-cox"])(method)
    # validate skew method 
    [validate_skew_method(original_data[col], method) 
     for col in original_data.columns ]
    for column in original_data.columns:
        # Adjust for non-positive values where necessary
        if original_data[column].min() <= 0 and method in ['log', 'box-cox']:  
            data[column] += (-original_data[column].min() + 1)

        if method == 'log':
            data[column] = np.log(data[column])
        elif method == 'sqrt':
            data[column] = np.sqrt(data[column])
        elif method == 'box-cox':
            # Apply Box-Cox transformation only if all values are positive
            if data[column].min() > 0:
                data[column], _ = stats.boxcox(data[column])
            else:
                # Using a generalized Yeo-Johnson transformation 
                # if Box-Cox is not possible
                pt = PowerTransformer(method='yeo-johnson')
                data[column] = pt.fit_transform(data[[column]]).flatten()
    if view:
        _visualize_skew(original_data, data, fig_size)
        
    return data

def _visualize_skew(original_data: DataFrame, transformed_data: DataFrame,
                    fig_size: tuple):
    """
    Visualizes the original and transformed data distributions.
    """
    num_columns = len(original_data.columns)
    fig, axes = plt.subplots(nrows=2, ncols=num_columns, figsize=fig_size)

    for i, column in enumerate(original_data.columns):
        sns.boxplot(x=original_data[column], ax=axes[0, i], color='skyblue')
        axes[0, i].set_title(f'Original {column}')
        sns.violinplot(x=transformed_data[column], ax=axes[1, i], color='lightgreen')
        axes[1, i].set_title(f'Transformed {column}')

    plt.tight_layout()
    plt.show()

@Extract1dArrayOrSeries(as_series= True, axis=1, method="soft")
def validate_skew_method(data: Series, method: str):
    """
    Validates the appropriateness of a skewness correction method based on the
    characteristics of the data provided. It ensures that the chosen method can
    be applied given the nature of the data's distribution, such as the presence
    of non-positive values which may affect certain transformations.

    Parameters
    ----------
    data : pandas.Series
        A Series containing the data to be checked for skewness correction.
    method : str
        The method of transformation intended to correct skewness:
        - 'log' : Natural logarithm, requires all positive values.
        - 'sqrt': Square root, requires non-negative values.
        - 'box-cox': Box-Cox transformation, requires all positive values.
          Falls back to Yeo-Johnson if non-positive values are found.

    Raises
    ------
    ValueError
        If the selected method is not suitable for the data based on its values.

    Returns
    -------
    str
        A message confirming the method's suitability or suggesting an
        alternative.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.dataops.quality import validate_skew_method
    >>> data = pd.Series([0.1, 1.5, 3.0, 4.5, 10.0])
    >>> print(validate_skew_method(data, 'log'))
    The log transformation is appropriate for this data.
    >>> data_with_zeros = pd.Series([0, 1, 2, 3, 4])
    >>> print(validate_skew_method(data_with_zeros, 'log'))
    ValueError: Log transformation requires all data ...
    """
    if not isinstance(data, pd.Series):
        raise TypeError(f"Expected a pandas Series, but got"
                        f" {type(data).__name__!r} instead.")
    _is_numeric_dtype(data)
    min_value = data.min()
    if method == 'log':
        if min_value <= 0:
            raise ValueError(
                "Log transformation requires all data points to be positive. "
                "Consider using 'sqrt' or 'box-cox' method instead.")
    elif method == 'sqrt':
        if min_value < 0:
            raise ValueError(
                "Square root transformation requires all data points"
                " to be non-negative. Consider using 'box-cox' or a specific"
                " transformation that handles negative values.")
    elif method == 'box-cox':
        if min_value <= 0:
            return ("Box-Cox transformation requires positive values, but"
                    " non-positive values are present. Applying Yeo-Johnson"
                    " transformation instead as a fallback.")
    else:
        raise ValueError("Unsupported method provided. Choose"
                         " from 'log', 'sqrt', or 'box-cox'.")

    return f"The {method} transformation is appropriate for this data."

@isdf
def check_skew_methods_applicability(
    data: DataFrame, return_report: bool = False, 
    return_best_method: bool = False) -> Union[Dict[str, List[str]], str]:
    """
    Evaluates each numeric column in a DataFrame to determine which skew
    correction methods are applicable based on the data's characteristics.
    Utilizes the `validate_skew_method` function to check the applicability
    of 'log', 'sqrt', and 'box-cox' transformations for each column.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame whose columns are to be evaluated for skew correction
        applicability.
    return_report : bool, optional
        If True, returns a detailed report on the skew correction methods
        applicability using the ReportFactory.
    return_best_method : bool, optional
        If True, returns the best skew correction method applicable to the
        most skewed column based on skewness measure.

    Returns
    -------
    dict or str
        A dictionary where keys are column names and values are lists of 
        applicable skew correction methods, or the best method as a string
        if return_best_method is True.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import check_skew_methods_applicability
    >>> df = pd.DataFrame({
    ...     'A': [0.1, 1.5, 3.0, 4.5, 10.0],
    ...     'B': [1, 2, 3, 4, 5],
    ...     'C': [-1, -2, -3, -4, -5],
    ...     'D': [0, 0, 1, 2, 3]
    ... })
    >>> applicable_methods = check_skew_methods_applicability(df)
    >>> print(applicable_methods)
    >>> applicable_methods = check_skew_methods_applicability(
    ...     df, return_best_method=True)
    >>> print(applicable_methods)
    log
    """
    applicable_methods = {}
    best_method = ''
    highest_skew = 0

    for column in data.select_dtypes(include=[np.number]):
        methods = []
        column_skew = data[column].skew()
        for method in ['log', 'sqrt', 'box-cox']:
            try:
                validate_skew_method(data[column], method)
                methods.append(method)
            except ValueError as e:
                if return_report:
                    applicable_methods [f'{column}_err_msg']= f"{str(e)}"

        applicable_methods[column] = methods
        
        # Checking for the best method if required
        if return_best_method and column_skew > highest_skew:
            highest_skew = column_skew
            best_method = methods[0] if methods else 'No valid method'

    if return_report: 
        return ReportFactory("Skew Methods Feasability", 
                             descriptor="SkewChecker", **applicable_methods 
                ).add_contents(applicable_methods, max_width=90)

    if return_best_method:
        if not best_method:
            warnings.warn(
                "No valid skew correction method found across all columns."
                " Return all applicabale methods instead.")
        else: 
            return best_method
        
    return applicable_methods

@Dataify(auto_columns=True)
def handle_duplicates(
    data: DataFrame, 
    return_duplicate_rows: bool=False, 
    return_indices: bool=False, 
    operation: str='drop', 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 8)
    )-> DataFrame | list | None:
    """
    Handles duplicate rows in a DataFrame based on user-specified options.
    
    This function can return a DataFrame containing duplicate rows, the indice
    s
    of these rows, or remove duplicates based on the specified operation.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame in which to handle duplicates.
    return_duplicate_rows : bool, optional
        If True, returns a DataFrame containing all duplicate rows. 
        This parameter takes precedence over `return_indices` and `operation`
        if set to True.
        Defaults to False.
    return_indices : bool, optional
        If True, returns a list of indices of duplicate rows. This will only
        take effect if `return_duplicate_rows` is False and takes precedence
        over `operation`.
        Defaults to False.
    operation : {'drop', 'none'}, optional
        Specifies the operation to perform on duplicate rows:
        - 'drop': Removes all duplicate rows, keeping the first occurrence.
        - 'none': No operation on duplicates; the original DataFrame is returned.
        Defaults to 'drop'.
    view : bool, optional
        If True, visualizes the DataFrame before and after handling duplicates,
        as well as any intermediate states depending on other parameter values.
        Default is False.
    cmap : str, optional
        The colormap to use for visualizing the DataFrame.
        Default is 'viridis'.
    fig_size : tuple, optional
        The size of the figure for visualization.
        Default is (12, 5).
        
    Returns
    -------
    pandas.DataFrame or list
        Depending on the parameters provided, this function may return:
        - A DataFrame of duplicates if `return_duplicate_rows` is True.
        - A list of indices of duplicate rows if `return_indices` is True.
        - A DataFrame with duplicates removed if `operation` is 'drop'.
        - The original DataFrame if `operation` is 'none'.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.dataops.quality import handle_duplicates
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 2, 4, 5, 1],
    ...     'B': [1, 2, 2, 4, 5, 1]
    ... })
    >>> print(handle_duplicates(data, return_duplicate_rows=True))
    >>> print(handle_duplicates(data, return_indices=True))
    >>> print(handle_duplicates(data, operation='drop'))

    Notes
    -----
    The function is designed to handle duplicates flexibly, allowing users to
    explore, identify, or clean duplicates based on their specific requirements. 
    It is useful in data cleaning processes where duplicates might skew the results
    or affect the performance of data analysis and predictive modeling.

    See Also
    --------
    pandas.DataFrame.duplicated : Check for duplicate rows in a DataFrame.
    pandas.DataFrame.drop_duplicates : Remove duplicate rows from a DataFrame.
    """
    operation = parameter_validator('operation', ["drop", "none"])(operation)
    # Identify all duplicates based on all columns
    duplicates = data.duplicated(keep=False)
    
    # Remove duplicate rows if operation is 'drop'
    cleaned_data = data.drop_duplicates(keep='first'
                                        ) if operation == 'drop' else data
    
    if view:
        _visualize_data(data, duplicates, cleaned_data, cmap, fig_size)
        
    # Return DataFrame of duplicate rows if requested
    if return_duplicate_rows:
        return data[duplicates]

    # Return indices of duplicate rows if requested
    if return_indices:
        return data[duplicates].index.tolist()

    # Return the original DataFrame if no operation
    # is specified or understood
    return cleaned_data

def _visualize_data(original_data: DataFrame, duplicates_mask: Series, 
                    cleaned_data: DataFrame, cmap: str, fig_size: tuple):
    """
    Visualizes the original DataFrame with duplicates highlighted and the 
    cleaned DataFrame.
    """
    # Create 1 row, 2 columns of subplots
    fig, axs = plt.subplots(1, 2, figsize=fig_size)  
    
    # Plot original data with duplicates highlighted
    axs[0].set_title('Original Data with Duplicates Highlighted')
    im0 = axs[0].imshow(original_data.mask(~duplicates_mask, other=np.nan),
                        aspect='auto', cmap=cmap, interpolation='none')
    axs[0].set_xlabel('Columns')
    axs[0].set_ylabel('Rows')
    axs[0].set_xticks(range(len(original_data.columns)))
    axs[0].set_xticklabels(original_data.columns, rotation=45)
    
    # Plot data after duplicates are removed
    axs[1].set_title('Data After Removing Duplicates')
    im1 = axs[1].imshow(cleaned_data, aspect='auto', cmap=cmap, interpolation='none')
    axs[1].set_xlabel('Columns')
    axs[1].set_ylabel('Rows')
    axs[1].set_xticks(range(len(cleaned_data.columns)))
    axs[1].set_xticklabels(cleaned_data.columns, rotation=45)
    
    plt.tight_layout()
    # Creating colorbars for each subplot
    plt.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.show()
    
@Dataify(auto_columns=True)
def quality_control(
    data, /, 
    missing_threshold=0.05, 
    outlier_method='IQR', 
    value_ranges=None,
    unique_value_columns=None, 
    string_patterns=None, 
    include_data_types=False, 
    verbose=False, 
    polish=False, 
    columns=None, 
    **kwargs
    ):
    """
    Perform comprehensive data quality checks on a pandas DataFrame and,
    if specified, cleans and sanitizes the DataFrame based on identified 
    issues using the `polish=True` parameter. This function is designed to 
    enhance data integrity before further processing or analysis.

    The function conducts the following operations:
    - Drops columns with a high percentage of missing data based on a 
      specified threshold.
    - Removes outliers using either the Interquartile Range (IQR) or Z-score 
      methods.
    - Applies constraints such as value ranges, checks for unique values, and 
      pattern matching using regular expressions, with cleaning steps tailored 
      to the specific needs of the dataset.

    Parameters
    ----------
    data : DataFrame
        The DataFrame on which to perform data quality checks.
    missing_threshold : float, optional
        Threshold as a fraction to determine excessive missing data in a 
        column (default is 0.05, i.e., 5%).
    outlier_method : str, optional
        Method to detect outliers. 'IQR' for Interquartile Range (default) 
        and 'Z-score' are supported.
    value_ranges : dict of tuple, optional
        Mapping of column names to tuples specifying the acceptable 
        (min, max) range for values.
    unique_value_columns : list of str, optional
        List of columns expected to have unique values throughout.
    string_patterns : dict of str, optional
        Patterns that values in specified columns should match, given as 
        regular expressions.
    include_data_types : bool, optional
        Whether to include data types of each column in the results 
        (default is False).
    verbose : bool, optional
        Enables printing of messages about the operations being performed 
        (default is True).
    polish : bool, optional
        If True, cleans the DataFrame based on the checks performed 
        (default is False).
    columns : list of str, optional
        Specific subset of columns to perform checks on. If None, checks 
        are performed on all columns.
    **kwargs : dict
        Additional keyword arguments for extensions or underlying functions.

    Returns
    -------
    QualityControl
        An instance containing detailed results of the checks performed 
        and the cleaned DataFrame if `polish` is True.

    Examples
    --------
    >>> from gofast.dataops.quality import quality_control
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, None, 5],
    ...     'B': [1, 2, 100, 3, 4],
    ...     'C': ['abc', 'def', '123', 'xyz', 'ab']
    ... })
    >>> qc = quality_control(data,
    ...                      value_ranges={'A': (0, 10)},
    ...                      unique_value_columns=['B'],
    ...                      string_patterns={'C': r'^[a-zA-Z]+$'},
    ...                      polish=True)
    >>> qc
    <QualityControl: 3 checks performed, data polished. Use print() ...>
    >>> print(qc)
                 Quality Control              
    ==========================================
    missing_data    : {'A': '20.0 %'}
    outliers        : {'B': [100]}
    string_pattern_violations : {'C': ['123']}
    ==========================================

    >>> print(qc.results_)
    {
         'missing_data': {'A': '20.0 %'}, 'outliers': {'B': [100]},
         'string_pattern_violations': {'C': ['123']}
     }
    >>> print(qc.results)
    Result(
      {

           missing_data              : {'A': '20.0 %'}
           outliers                  : {'B': [100]}
           string_pattern_violations : {'C': ['123']}

      }
    )
    >>> qc.data_polished
       B    C
    0  1  abc
    1  2  def
    3  3  xyz
    4  4   ab
    
    Note
    ----
    Data cleaning and sanitization can significantly alter your dataset. It is 
    essential to understand the implications of each step and adjust the 
    thresholds and methods according to your data analysis goals.
    """
    # Initialize result dictionary
    results = {
        'missing_data': {}, 'outliers': {}, 'data_types': {},
        'value_range_violations': {},'unique_value_violations': {},
        'string_pattern_violations': {}
    }

    polish, verbose = ellipsis2false(polish, verbose)
    cleaned_data = data.copy()
    # Handle missing data
    for col in data.columns:
        missing_percentage = data[col].isna().mean()
        if missing_percentage > missing_threshold:
            message =(  f"Column '{col}' exceeds missing threshold "
                      f"with {missing_percentage:.2%} missing." ) 
            results['missing_data'][col] = str(missing_percentage *100)+ " %"
            if polish:
                cleaned_data.drop(col, axis=1, inplace=True)
                if verbose:
                    print(message)
    # Handle outliers
    for col in data.select_dtypes(include=[np.number]).columns:
        if outlier_method == 'IQR':
            # Calculate IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif outlier_method == 'Z-score':
            # Calculate mean and standard deviation for Z-score
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
    
        outliers = data[col][(data[col] < lower_bound) | (data[col] > upper_bound)]
        if not outliers.empty:
            results['outliers'][col] = outliers.tolist()
        if polish and not outliers.empty:
            # Filter out outliers from the data
            try:
                cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (
                    cleaned_data[col] <= upper_bound)]
            except : pass 
    # Value range checks
    if value_ranges:
        for col, (min_val, max_val) in value_ranges.items():
            invalid_values = data[col][(data[col] < min_val) | (data[col] > max_val)]
            if not invalid_values.empty: 
                results['value_range_violations'][col] = invalid_values.tolist()
                if polish:
                    try: 
                        # Keep only values within the specified range
                        cleaned_data = cleaned_data[(
                            cleaned_data[col] >= min_val) & (cleaned_data[col] <= max_val)]
                    except: 
                        pass 
    # Unique value checks
    if unique_value_columns:
        for col in unique_value_columns:
            if col in data.columns and data[col].duplicated().any():
                duplicates = data[col][data[col].duplicated(keep=False)]
                results['unique_value_violations'][col] = duplicates.tolist()
                if polish:
                    try: 
                        # Keep only the first occurrence of each value
                        cleaned_data = cleaned_data.drop_duplicates(
                            subset=[col], keep='first')
                    except: 
                        pass 
                    
        # complete cleaned data here is polish 
    # String pattern checks
    if string_patterns:
        for col, pattern in string_patterns.items():
            if col in data.columns and data[col].dtype == 'object':
                mismatches = data[col][~data[col].astype(str).str.match(pattern)]
                results['string_pattern_violations'][col] = mismatches.tolist()
                if polish:
                    try: 
                        valid_mask = data[col].astype(str).str.match(pattern)
                        # Using `loc` to address potential index misalignment 
                        # issues by filtering on the index
                        cleaned_data = cleaned_data.loc[valid_mask[valid_mask].index]
                    except: 
                        pass 

    # Data type information
    for col in data.columns:
        results['data_types'][col] = data[col].dtype
    
    # Data type information
    if include_data_types:
        for col in data.columns:
            results['data_types'][col] = data[col].dtype
    else:
        results.pop('data_types', None)  # Remove data_types key if not required
        
    # Remove any keys from results with empty values
    results = {key: value for key, value in results.items() if value}
    
    # Create QualityControl instance with results
    qc_summary = _QualityControl(data, results)
    qc_summary.add_recommendations ( results, max_char_text = TW )
    
    if polish:
        # Store polished data within the QualityControl object
        qc_summary.data_polished = cleaned_data  

    return qc_summary

class _QualityControl(ReportFactory):
    """
    Initializes the QualityControl class, which inherits from ReportFactory,
    to manage and present data quality check results.
    
    Parameters:
    - data (DataFrame): The DataFrame on which data quality checks were 
      performed.
    - results (dict): A dictionary containing the results of the data 
      quality checks.
    - title (str, optional): The title of the report.
      Defaults to "Quality Control".
    - **kwargs: Additional keyword arguments for further customization.
    """    
    def __init__(self, data, results, title="Quality Control", **kwargs):
        super().__init__(title=title, **kwargs)
        self.data = data
        self.results_ = results
        self.results=ResultSummary(pad_keys="auto" ).add_results(results)

    def __str__(self):
        """
        Provides a string representation of the QualityControl instance, showing a
        summary report if results exist, otherwise indicating no issues were detected.

        Returns:
        - str: Summary report string or a message indicating no issues detected.
        """
        if not self.results_:
            return "<QualityControl: No issues detected>"
        #  ReportFactory has a __str__ that handles report details.
        return super().__str__()  

    def __repr__(self):
        """
        Provides a formal string representation of the QualityControl instance,
        suitable for developers, including the status of checks and data cleaning.

        Returns:
        - str: Developer-friendly representation including the number of 
            checks and data cleaning status.
        """
        checks_count = len(self.results_)
        message = f"{checks_count} checks performed"
        if hasattr(self, "data_polished"):
            message += ", data polished"
        extra =( 
            "Use print() to see detailed contents" 
            if self.results_ else "No issues detected"
            )
        return f"<QualityControl: {message}. {extra}>"
                

if __name__ == "__main__":
    # Example usage of the function

    data_positive = pd.Series([0.1, 1.5, 3.0, 4.5, 10.0])
    try:
        print(validate_skew_method(data_positive, 'log'))
    except ValueError as e:
        print(e)

    data_with_negatives = pd.Series([-1, 2, 5, 7, 9])
    try:
        print(validate_skew_method(data_with_negatives, 'sqrt'))
    except ValueError as e:
        print(e)

    # Create a sample DataFrame with duplicate rows
    data = pd.DataFrame({
        'A': [1, 2, 2, 4, 5, 1],
        'B': [1, 2, 2, 4, 5, 1],
        'C': [1, 2, 2, 4, 5, 1]
    })

    # Finding and returning duplicate rows
    duplicates_df = handle_duplicates(data, return_duplicate_rows=True)
    print("Duplicate Rows:\n", duplicates_df)

    # Returning the indices of duplicate rows
    duplicate_indices = handle_duplicates(data, return_indices=True)
    print("Indices of Duplicate Rows:\n", duplicate_indices)

    # Dropping duplicates and returning the cleaned DataFrame
    cleaned_data = handle_duplicates(data, operation='drop')
    print("Data without Duplicates:\n", cleaned_data)


    # Create a sample DataFrame with skew
    data = pd.DataFrame({
        'A': np.random.normal(0, 2, 100),
        'B': np.random.chisquare(2, 100),
        'C': np.random.beta(2, 5, 100) * 60,  # positive and skewed
        'D': np.random.uniform(-1, 0, 100)  # contains negative values
    })

    # Apply transformation
    result_log = handle_skew(data.copy(), method='log')
    result_sqrt = handle_skew(data.copy(), method='sqrt')
    result_boxcox = handle_skew(data.copy(), method='box-cox')

    print("Log-transformed:\n", result_log.head())
    print("Square root-transformed:\n", result_sqrt.head())
    print("Box-Cox transformed:\n", result_boxcox.head())


    # Create a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5],
        'C': [5, 4, 3, 2, 1],
        'D': [2, 3, 2, 3, 2]
    })

    # Call the function to drop correlated features
    result = drop_correlated_features(data, threshold=0.8)
    print(result)


    # Create a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5],
        'C': [5, 4, 3, 2, 1],
        'D': [2, 3, 2, 3, 2]
    })

    # Call the function
    result = correlation_ops(data, correlation_type='strong positive')
    print("Strong Positive Correlations:", result)

    import pandas as pd
    #from gofast.dataops.quality import data_assistant 
    # Create a sample DataFrame
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, None],
        'Salary': [50000, 60000, 70000, 80000, 90000],
        'City': ['New York', 'Los Angeles', 'San Francisco', 'Houston', 'Seattle']
    })
    # Call the assistant function
    data_assistant(df)








