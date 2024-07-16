# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides utility functions to support statistical analyses, including 
dataset preprocessing for repeated measures ANOVA."""


from itertools import product
import numpy as np
import pandas as pd

from ..api.types import Optional, List, Union, Tuple, Callable, Any
from ..api.types import DataFrame, ArrayLike
from ..tools.coreutils import normalize_string 
from ..tools.coreutils import smart_format


__all__=["fix_rm_anova_dataset" ]

def prepare_stats_plot(
    values: ArrayLike,
    data: Optional[ArrayLike] = None,
    axis: Optional[int] = None,
    transform: Optional[Callable[[Any], Any]] = None
) -> Tuple[Any, Any, Optional[Union[pd.Index, range]]]:
    """
    Prepares data and its labels for plotting, handling different data structures
    and orientations based on the specified axis.

    Parameters
    ----------
    values : Union[np.ndarray, pd.Series, pd.DataFrame]
        The data values to be plotted, which can be the result of statistical
        operations like mean, median, etc.
    data : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], optional
        Additional data related to `values`, used for plotting. Defaults to None.
    axis : Optional[int], optional
        The axis along which to organize the plot data. Can be None, 0, or 1.
        Defaults to None.
    transform : Optional[Callable[[Any], Any]], optional
        A function to apply to the data before plotting. Defaults to None.

    Returns
    -------
    Tuple[Any, Any, Optional[Union[pd.Index, range]]]
        A tuple containing the transformed `values`, `data`, and the labels
        for plotting.

    Examples
    --------
    >>> import pandas as pd 
    >>> import numpy as np 
    >>> from gofast.stats.utils import prepare_stats_plot
    >>> values = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    >>> prepare_stats_plot(values)
    (Series([1, 2, 3], index=['a', 'b', 'c']), None, Index(['a', 'b', 'c']))

    >>> values = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
    >>> prepare_stats_plot(values, axis=1)
    (Transposed DataFrame, Transposed DataFrame, Index(['x', 'y']))

    >>> values = np.array([1, 2, 3])
    >>> prepare_stats_plot(values, transform=np.square)
    (array([1, 4, 9]), None, range(0, 3))
    """
    if transform is not None and callable(transform):
        values = transform(values)
        if data is not None:
            data = transform(data)

    if axis is None:
        return values, data, None

    if isinstance(values, pd.Series):
        data= data.T if axis ==1 else data  
        return values.T, data if data is not None else None, values.index

    elif isinstance(values, pd.DataFrame):
        if axis == 0:
            return values, data, values.columns
        elif axis == 1:
            return values.T, data.T if data is not None else None, values.index

    elif isinstance(values, np.ndarray):
        values = np.squeeze(values)
        if values.ndim == 1:
            return values, data, range(len(values))
        if axis == 0:
            return values, data, range(values.shape[1])
        elif axis == 1:
            return values.T, data.T if data is not None else None, range(values.shape[0])

    return values, data, None

def validate_stats_plot_type(
        type_: str,
        target_strs: List[str],
        match_method: str = 'contains',
        raise_exception: bool = False,
        **kwargs) -> Optional[str]:
    """
    Validates the plot type against a list of acceptable types and returns
    the normalized matching string.

    This function checks if the given plot type matches any of the target strings
    based on the specified match method. If a match is found, the function returns
    the normalized string from the target list. It can optionally raise an exception
    if no match is found.

    Parameters
    ----------
    type_ : str
        The plot type to validate.
    target_strs : List[str]
        A list of acceptable plot type strings to match against.
    match_method : str, default 'contains'
        The method used to match the plot type with the target strings. Options include:
        - 'contains': Checks if `type_` is contained within any of the target strings.
        - 'exact': Checks for an exact match between `type_` and the target strings.
        - 'startswith': Checks if `type_` starts with any of the target strings.
    raise_exception : bool, default False
        If True, raises a ValueError when no match is found. Otherwise, returns None.
    **kwargs : dict
        Additional keyword arguments to be passed to the `normalize_string` function.

    Returns
    -------
    Optional[str]
        The normalized string from `target_strs` that matches `type_`, or None if
        no match is found and `raise_exception` is False.

    Raises
    ------
    ValueError
        If `raise_exception` is True and no match is found.

    Examples
    --------
    >>> from gofast.stats.utils import validate_stats_plot_type
    >>> validate_stats_plot_type('box', ['boxplot', 'histogram', 'density'],
    ...                        match_method='startswith')
    'boxplot'

    >>> validate_stats_plot_type('exact_plot', ['boxplot', 'histogram', 'density'],
    ...                        match_method='exact')
    None

    >>> validate_stats_plot_type('hist', ['boxplot', 'histogram', 'density'], 
    ...                        match_method='contains')
    'histogram'

    >>> validate_stats_plot_type('unknown', ['boxplot', 'histogram', 'density'],
    ...                        raise_exception=True)
    ValueError: Plot type 'unknown' is not supported.

    Note
    ----
    This utility function is designed to help in functions or methods where plot type
    validation is necessary, improving error handling and user feedback for plotting
    functionalities.
    """
    matched_type = normalize_string(
        type_, target_strs=target_strs,
        return_target_str=False, match_method=match_method,
        return_target_only=True, **kwargs)

    if matched_type is None and raise_exception:
        raise ValueError(
            f"Unsupported type '{type_}'. Expect {smart_format(target_strs)}.")

    return matched_type

def fix_rm_anova_dataset(
    data: DataFrame, 
    depvar: str, 
    subject: str, 
    within: List[str], 
    strategy: str = "mean", 
    fill_value: Optional[Union[str, float, int]] = None
) -> DataFrame:
    """
    Generate all possible combinations of within-subject factors and fill missing 
    depvar values based on the specified strategy.

    Parameters
    ----------
    data : DataFrame
        The dataset to be processed.
    depvar : str
        The dependent variable whose missing values need to be filled.
    subject : str
        The subject column in the dataset.
    within : List[str]
        A list of columns representing within-subject factors.
    strategy : str, optional
        The strategy to use for filling missing depvar values. Options are "mean",
        "median", or None. Default is "mean".
    fill_value : Optional[Union[str, float, int]], optional
        A specific value to fill missing depvar values if the strategy is None.
        Default is None, which leaves missing values as None.

    Returns
    -------
    DataFrame
        The modified dataset with missing combinations filled.
    """
    if not isinstance (data, pd.DataFrame): 
        raise TypeError("First positionnal argument `data` expects a DataFrame. Got"
                        f" {type(data).__name__!r}")
    all_combinations = list(product(*[data[factor].unique() for factor in within]))
    fixed_data = []

    for subj in data[subject].unique():
        subj_data = data[data[subject] == subj].copy()
        if strategy == "mean":
            fill = subj_data[depvar].mean()
        elif strategy == "median":
            fill = subj_data[depvar].median()
        elif fill_value is not None:
            fill = fill_value if strategy is None else None
        else:
            fill = None
        for combination in all_combinations:
            if combination not in list(zip(*[subj_data[factor].values for factor in within])):
                new_row = {subject: subj, depvar: fill}
                new_row.update(dict(zip(within, combination)))
                subj_data = pd.concat([subj_data, pd.DataFrame([new_row])], ignore_index=True)

        fixed_data.append(subj_data)

    return pd.concat(fixed_data, ignore_index=True)