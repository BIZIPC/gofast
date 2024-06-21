# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from ..api.types import Optional, List,  Union, Tuple
from ..api.types import DataFrame, ArrayLike, Array1D, Series
from ..decorators import DynamicMethod
from ..tools.validator import check_consistent_length 
from ..tools.coreutils import ensure_visualization_compatibility
from ..tools.coreutils import to_series_if, ellipsis2false 
from ..tools.coreutils import get_colors_and_alphas
from ..tools.funcutils import make_data_dynamic
from ..tools.funcutils import flatten_data_if, update_series_index 
from ..tools.funcutils import update_index, convert_and_format_data
from ..tools.funcutils import series_naming 
from .utils import validate_stats_plot_type, prepare_stats_plot

__all__= [ 
    "describe",
    "get_range",
    "hmean",
    "iqr",
    "mean",
    "median",
    "mode",
    "quantile",
    "quartiles",
    "std",
    "var",
    "wmedian",
    "skew",
    "kurtosis",
    "gini_coeffs",
    "z_scores",
]

@make_data_dynamic(capture_columns=True, dynamize=False)
def gini_coeffs(
    data: Union[DataFrame, np.ndarray],
    columns: Optional[Union[str, List[str]]] = None,
    as_frame: bool = False,
    view: bool = False,
    fig_size: Tuple[int, int] = (10, 6)
):
    r"""
    Calculate the Gini coefficient of a dataset and optionally visualize
    the Lorenz curve.

    The Gini coefficient is a measure of statistical dispersion intended to 
    represent the income or wealth distribution of a nation's residents, 
    and is the most commonly used measurement of inequality. It is defined 
    mathematically based on the Lorenz curve, which plots the proportion of 
    the total income of a population that is cumulatively earned by the 
    bottom x% of the population.

    The Gini coefficient (G) can be calculated using the formula:

    .. math:: G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_{i}}{n \sum_{i=1}^{n} x_{i}}

    where \( n \) is the number of values, \( x_{i} \) is the value after 
    sorting the data in increasing order, and \( i \) is the rank of values 
    in ascending order.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input data for which to calculate the Gini coefficient. Can be a 
        pandas DataFrame or a numpy ndarray.
    columns : Optional[Union[str, List[str]]], optional
        If provided, specifies the column(s) to use when `data` is a DataFrame.
        If a single string is provided, it will select a single column.
        If a list of strings is provided, it will select multiple columns, 
        and the Gini coefficient will be calculated for each column separately.
    as_frame : bool, optional
        If True, the result will be returned as a pandas DataFrame.
        Default is False.
    view : bool, optional
        If True, displays the Lorenz curve plot. Default is False.
    fig_size : Tuple[int, int], optional
        Size of the figure to display if `view` is True. Default is (10, 6).

    Returns
    -------
    Union[float, pd.DataFrame]
        The Gini coefficient of the data. If `as_frame` is True, returns a 
        pandas DataFrame with the Gini coefficient.

    See Also
    --------
    plot_lorenz_curve : A function to plot the Lorenz curve.

    Notes
    -----
    The Gini coefficient is a widely used measure of inequality. A Gini 
    coefficient of zero expresses perfect equality where all values are the 
    same. A Gini coefficient of one (or 100%) expresses maximal inequality 
    among values (for example, where only one person has all the income).

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.descriptive import gini_coeffs
    >>> gini_coeffs(np.array([1, 2, 3, 4, 5]))
    0.26666666666666666

    >>> df = pd.DataFrame({'income': [1, 2, 3, 4, 5]})
    >>> gini_coeffs(df, columns='income', view=True)
    # This will calculate the Gini coefficient for the 'income' column and 
    # display the Lorenz curve.

    >>> df = pd.DataFrame({'income': [1, 2, 3, 4, 5], 'wealth': [5, 4, 3, 2, 1]})
    >>> gini_coeffs(df, columns=['income', 'wealth'], as_frame=True)
    # This will calculate the Gini coefficient for both 'income' and 'wealth' 
    # columns and return the results in a DataFrame.
    """
    # Ensure data is a 1D numpy array
    data = np.ravel(data)
    
    # Sort data
    data = np.sort(data)
    
    # Calculate Gini coefficient
    n = len(data)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * data) - (n + 1) * np.sum(data)) / (n * np.sum(data))
    
    # Visualize Lorenz curve if requested
    if view:
        plot_lorenz_curve(data, fig_size)

    if as_frame:
        return pd.DataFrame({'Gini-coefficients': [gini]}, index=['gini_coeffs'])

    return gini

def plot_lorenz_curve(data: np.ndarray, fig_size: Tuple[int, int]):
    """
    Plot the Lorenz curve for the given data.

    Parameters
    ----------
    data : np.ndarray
        Sorted 1D data array for which to plot the Lorenz curve.
    fig_size : Tuple[int, int]
        Size of the figure for the Lorenz curve plot.
    """
    lorenz_curve = np.cumsum(data) / np.sum(data)
    lorenz_curve = np.insert(lorenz_curve, 0, 0)  # Start at 0
    
    plt.figure(figsize=fig_size)
    plt.plot(np.linspace(0.0, 1.0, len(lorenz_curve)), lorenz_curve,
             label='Lorenz Curve', color='blue')
    plt.plot([0, 1], [0, 1], label='Line of Equality', 
             linestyle='--', color='red')
    plt.title('Lorenz Curve')
    plt.xlabel('Cumulative share of population')
    plt.ylabel('Cumulative share of wealth')
    plt.legend()
    plt.grid(True)
    plt.show()
    

@DynamicMethod ( expected_type="both", capture_columns=True)
def corr(
    data: Union[ArrayLike, DataFrame], /,  
    columns: List[str] = None,
    method: str='pearson', 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
    ):
    """
    Calculates the correlation matrix for the given dataset. If the dataset is 
    a DataFrame and columns are specified, the correlation is calculated only 
    for those columns.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the correlation matrix. Can be 
        a list, numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names for which to calculate the correlation if `data` 
        is a DataFrame. If not provided, the correlation is calculated for all 
        numeric columns in the DataFrame.
    
    **kws : dict
        Additional keyword arguments for the pandas `corr()` method, allowing 
        customization of the correlation calculation, such as specifying the 
        method (e.g., 'pearson', 'kendall', 'spearman').

    Returns
    -------
    correlation_matrix : DataFrame
        The correlation matrix of the provided data. If `data` is ArrayLike, 
        the function first converts it to a DataFrame.

    Examples
    --------
    >>> from gofast.stats.descriptive import corr
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [4, 3, 2, 1],
    ...     'C': [1, 2, 3, 4]
    ... })
    >>> corr(data)
           A         B    C
    A  1.000000 -1.000000  1.000000
    B -1.000000  1.000000 -1.000000
    C  1.000000 -1.000000  1.000000

    >>> corr2(data, columns=['A', 'C'])
           A    C
    A  1.000000  1.000000
    C  1.000000  1.000000
    >>> >>> corr(data, view=True)

    >>> corr(data, columns=['A', 'C'], view=True, plot_type='heatmap',
    ...        cmap='coolwarm')
    Note
    ----
    The function utilizes pandas' `corr()` method to compute the correlation 
    matrix, offering flexibility through `**kws` to use different correlation 
    computation methods. For non-DataFrame inputs, the data is first converted 
    to a DataFrame, ensuring uniform processing.
    """
    correlation_matrix = data.corr(method= method, **kws)
    if view:
        plt.figure(figsize=fig_size)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
    
    return correlation_matrix

@make_data_dynamic(capture_columns=True)
def skew(
    data: Union[ArrayLike, pd.DataFrame],
    columns: Optional[List[str]] = None,
    axis: int =0, 
    as_frame: bool = True,
    view: bool = False, 
    plot_type: Optional[str] = 'density', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
):
    """
    Calculates the skewness of the data distribution, providing insight into
    the asymmetry of the distribution around its mean.

    Skewness is a measure of the symmetry, or lack thereof, of a distribution.
    A skewness value > 0 indicates a distribution that is skewed to the right,
    a value < 0 indicates a distribution that is skewed to the left, and a
    value of 0 indicates a symmetrical distribution.

    Parameters
    ----------
    data : ArrayLike or pd.DataFrame
        The input data for which skewness is to be calculated. Can be a Pandas 
        DataFrame or any array-like structure containing numerical data.
        
    columns : List[str], optional
        Specific columns to include in the analysis if `data` is a DataFrame.
        If None, all columns are included. Default is None.
        
    axis: optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0. 
         
    as_frame : bool, default=True
        If True and `data` is ArrayLike, converts `data` to a Pandas DataFrame
        before calculating skewness. Useful for structured array-like inputs.
        
    view : bool, optional
        If True, visualizes the data distribution and its skewness using the
        specified `plot_type`. Defaults to False.
        
    plot_type : Optional[str], optional
        Type of plot for visualizing the data distribution. Supported values
        are 'density' for a density plot and 'hist' for a histogram. Ignored
        if `view` is False. Defaults to 'density'.
        
    cmap : str, optional
        Colormap for the plot. Defaults to 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the plot. Defaults to (10, 6).
    **kwargs : dict
        Additional keyword arguments passed to `pd.Series.skew`
        or `pd.DataFrame.skew`.


    Returns
    -------
    float or pd.Series
        The skewness value(s) of the data. Returns a single float value if `data`
        is a single column or ArrayLike without `as_frame` set to True. Returns
        a pd.Series with skewness values for each column if `data` is a DataFrame
        or `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.descriptive import skew
    >>> import numpy as np
    >>> data = np.random.normal(loc=0, scale=1, size=1000)
    >>> calculate_skewness(data)
    0.1205147559272991
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(loc=0, scale=1, size=1000),
    ...     'B': np.random.normal(loc=1, scale=2, size=1000),
    ...     'C': np.random.lognormal(mean=0, sigma=1, size=1000)
    ... })
    >>> skew(df)
    array([-0.07039989, -0.04687687,  7.20119035])
    
    >>> data = [1, 2, 2, 3, 4, 7, 9]
    >>> skew(data)
    0.9876939667076702

    >>> df = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 1000),
    ...     'right_skewed': np.random.exponential(1, 1000)
    ... })
    >>> skew(df, view=True, plot_type='density')

    Note
    ----
    This function is useful for identifying the symmetry of data distributions,
    which can be critical for certain statistical analyses and modeling. Skewness
    can indicate the need for data transformation or the use of non-parametric
    statistical methods.
    """
    # Ensuring numeric data type for calculation # for consisteny 
    data_numeric = data.apply(pd.to_numeric, errors='coerce')
    skewness_value = data_numeric.skew(axis=0, **kwargs) 

    if view:
        colors, alphas = get_colors_and_alphas( data_numeric.columns, cmap)
        plot_type = validate_stats_plot_type(
            plot_type, target_strs= ['hist', 'box', 'density'],
            raise_exception= True)
        if plot_type == 'density':
            for column in data_numeric.columns:
                sns.kdeplot(data_numeric[column],
                            label=f'Skewness: {skewness_value[column]:.2f}', 
                            fill=True)
            plt.title('Density Plot with Skewness')
        elif plot_type == 'hist':
            for column in data_numeric.columns:
                plt.hist(
                data_numeric[column], alpha=0.5, 
                label=f'{column} (Skewness: {skewness_value[column]:.2f})',
                bins=30)
            plt.title('Histogram with Skewness')
        else: 
            raise ValueError(
                f"Unsupported type '{plot_type}'. Expect 'density' and 'hist'")
        plt.xlabel("Value")
        plt.ylabel ("Frequency")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.show()
        
    skewness_value= convert_and_format_data(
        skewness_value, as_frame, 
        force_array_output= False if as_frame else True, 
        condense=False if as_frame else True,
        condition=series_naming ("skewness"),
        )    
    
    return skewness_value

@make_data_dynamic(
    capture_columns=True)
def kurtosis(
    data: Union[ArrayLike, DataFrame],
    columns: List[str] = None,
    axis: int= 0, 
    as_frame: bool = True,
    view: bool = False, 
    plot_type: str = 'density', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
    ):
    """
    Calculates the kurtosis of the data distribution, offering insights into
    the tail heaviness compared to a normal distribution.

    Kurtosis is a measure of whether the data are heavy-tailed or light-tailed
    relative to a normal distribution. Positive kurtosis indicates a distribution
    with heavier tails, while negative kurtosis indicates a distribution with
    lighter tails.

    The formula for kurtosis is defined as:

    .. math::

        Kurtosis = \\frac{N\\sum_{i=1}^{N}(X_i - \\bar{X})^4}{(\\sum_{i=1}^{N}(X_i - \\bar{X})^2)^2} - 3

    where:
    
    - :math:`N` is the number of observations,
    - :math:`X_i` is each individual observation,
    - :math:`\\bar{X}` is the mean of the observations.

    The subtraction by 3 at the end adjusts the result to fit the standard 
    definition of kurtosis, where a normal distribution has a kurtosis of 0.

    Parameters
    ----------
    data : ArrayLike or pd.DataFrame
        The input data for which kurtosis is to be calculated. Can be a Pandas
        DataFrame or any array-like structure containing numerical data.
    columns : List[str], optional
        Specific columns to include in the analysis if `data` is a DataFrame.
        If None, all columns are included. Default is None.
    axis : int, default=0
        Axis along which the kurtosis is calculated.
    as_frame : bool, default=False
        If True and `data` is ArrayLike, converts `data` to a Pandas DataFrame
        before calculating kurtosis. Useful for structured array-like inputs.
    view : bool, optional
        If True, visualizes the data distribution along with its kurtosis
        using the specified `plot_type`. Defaults to False.
    plot_type : str, optional
        Type of plot to visualize the data distribution. Supported values are
        'density' for a density plot and 'hist' for a histogram. Defaults to 'density'.
    cmap : str, optional
        Colormap for the plot. Defaults to 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the plot. Defaults to (10, 6).
    **kwargs : dict
        Additional keyword arguments passed to `scipy.stats.kurtosis`.

    Returns
    -------
    float or pd.Series
        The kurtosis value(s) of the data. Returns a single float value if `data`
        is a single column or ArrayLike without `as_frame` set to True. Returns
        a pd.Series with kurtosis values for each column if `data` is a DataFrame
        or `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.descriptive import kurtosis
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, 1000)
    >>> print(kurtosis(data))
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': np.random.normal(0, 1, size=1000),
    ...                    'B': np.random.standard_t(10, size=1000)})
    >>> print(kurtosis(df, as_frame=True))
    
    >>> data = np.random.normal(0, 1, 1000)
    >>> print(kurtosis(data))
    
    >>> df = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 1000),
    ...     'leptokurtic': np.random.normal(0, 1, 1000) ** 3,
    ... })
    >>> print(kurtosis(df, as_frame=True))

    Note
    ----
    Kurtosis is useful for understanding the extremity of outliers or the
    propensity of data to produce outliers. A higher kurtosis can indicate
    a higher risk or potential for outlier values in the dataset.
    """
    kurtosis_value = data.kurtosis(axis=axis, **kwargs)
    if view:
        plot_type= validate_stats_plot_type(
            plot_type, target_strs= ['density', 'hist'],
            raise_exception =True)
        colors, alphas = get_colors_and_alphas(
            len(kurtosis_value), cmap)
        kvalue, data, cols = prepare_stats_plot(kurtosis_value, data, axis = axis )
        if plot_type == 'density':
            for ii, col in enumerate (cols) :
                sns.kdeplot(data[col], label=f'{col} kurtosis={kvalue[ii]:.2f}',
                            color=colors [ii])
        elif plot_type == 'hist':
            data.hist(bins=30, alpha=0.5, color=colors[0], figsize=fig_size)
        plt.title("Data Distribution and Kurtosis")

        plt.xlabel("Value")
        plt.ylabel ("Frequency")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.show()

    kurtosis_value= convert_and_format_data(
        kurtosis_value, as_frame, 
        force_array_output= False if as_frame else True, 
        condense=False if as_frame else True,
        condition=series_naming ("kurtosis"),
        )    
    return kurtosis_value

@make_data_dynamic(capture_columns=True)
def wmedian(
    data: ArrayLike,
    weights:Union[str, int, Array1D],
    columns: Union[str, list] = None,
    as_frame: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (6, 4)
) -> float | Series:
    """
    Compute the weighted median of a dataset, optionally visualizing the 
    distribution of data points and their weights.

    The weighted median is defined as the value separating the higher half 
    from the lower half of a data sample, a population, or a probability 
    distribution, where each value has a corresponding weight. It is determined 
    such that the sum of weights is equal on both sides of the median in the 
    sorted list of data points.
    
    .. math::
        WM = \text{{value}} \; x \; \text{{such that}} \; \sum_{x_i < WM} w_i < \frac{1}{2} \sum w_i 
        \; \text{{and}} \; \sum_{x_i > WM} w_i \leq \frac{1}{2} \sum w_i

    Where :math:`WM` is the weighted median, :math:`x_i` are the data points, 
    and :math:`w_i` are the weights associated with each data point.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data for which the weighted median is desired. If a DataFrame 
        and `columns` is specified, the calculation uses only the 
        selected column(s).
    weights : str, int, np.ndarray, or list
        Weights corresponding to each element in `data`. If a string or int, it 
        specifies the column in the `data` DataFrame containing the weights.
    columns : str or list, optional
        Specific column(s) to use for the calculation if `data` is a DataFrame.
    as_frame : bool, optional
        If True, the function returns the result as a pandas Series. 
        Default is False.
    view : bool, optional
        If True, displays a scatter plot of the data points with sizes 
        proportional to their weights. Default is False.
    cmap : str, optional
        Colormap for the scatter plot. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the scatter plot. Default is (8, 8).

    Returns
    -------
    float or pd.Series
        The weighted median of the data set. Returns a pandas Series 
        if `as_frame` is True.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.stats.descriptive import wmean 
    >>> data = np.array([1, 2, 3])
    >>> weights = np.array([3, 1, 2])
    >>> wmedian(data, weights)
    2.0

    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [1, 2, 3], 'weights': [3, 1, 2]})
    >>> wmedian(df, 'weights', columns='values', as_frame=True)
    W-median    2.0
    dtype: float64

    Notes
    -----
    The weighted median is particularly useful in datasets where some values 
    are inherently more important than others and should be more heavily 
    considered in the central tendency measure.

    See Also
    --------
    np.median : Compute the median along the specified axis.
    pd.Series.median : Return the median of the values for the requested axis.
    scipy.stats.weighted_median : Compute the weighted median of a data 
      sample in SciPy.
    """
    # Ensure 'weights' is a column name in the DataFrame.
    if isinstance(weights, str) and weights not in data.columns:
        raise ValueError("The 'weights' parameter, when passed as a string, "
                        "must match the name of a column in the DataFrame.")

    # Extract the relevant columns and convert weights to a numpy array 
    # if specified as a column name
    if isinstance(weights, str):
        weights = data.pop(weights).values

    # re-convert weights to a numpy array for consistency 
    # and check that all weights are positive
    weights= np.asarray(weights)
    if np.any(weights <= 0):
        raise ValueError("All weights must be greater than 0.")
    
    # Convert data to numpy array and flatten if necessary
    data_values = data.to_numpy().flatten() if data.ndim > 1 else data.values
    weights = np.asarray(weights).flatten()

    try:
        check_consistent_length(data_values, weights)
    except ValueError as e:
        raise ValueError(
            "Data and weights must be of the same length. Received lengths:"
            f" data_values={len(data_values)}, weights={len(weights)}") from e

    # Sort data and weights
    sorted_indices = np.argsort(data_values)
    sorted_data = data_values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate cumulative weights and find the median index
    cumulative_weights = np.cumsum(sorted_weights)
    half_weight = cumulative_weights[-1] / 2
    median_idx = np.searchsorted(cumulative_weights, half_weight)

    # Handle edge case where searchsorted returns an index equal 
    # to the size of data
    if median_idx == len(sorted_data):
        median_idx -= 1

    w_median = sorted_data[median_idx]

    if view:
        visualize_weighted_median(sorted_data, sorted_weights, w_median, cmap, fig_size)

    return to_series_if (w_median, value_names= ['W-median'], name='weighted_median'
                          )  if as_frame else w_median

def visualize_weighted_median(data, weights, w_median, cmap, fig_size):
    plt.figure(figsize=fig_size)
    plt.scatter(range(len(data)), [0] * len(data), 
                s=weights * 100, c=data, 
                cmap=cmap, alpha=0.6)
    plt.colorbar(label='Data Value')
    plt.axvline(w_median, color='red', linestyle='dashed', 
                linewidth=2, label=f'Weighted Median: {w_median}')
    plt.title('Weighted Median Visualization')
    plt.xlabel('Sorted Data Points')
    plt.yticks([])
    plt.legend()
    plt.show()

@make_data_dynamic(capture_columns=True)
def mode(
    data: Union[ArrayLike, DataFrame], 
    columns: List[str] = None, 
    axis: Optional[int]=None, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    """
    Calculates the mode(s) of numeric data provided either as an array-like 
    structure or within a pandas DataFrame.

    Utilizes the `make_data_dynamic` decorator to preprocess the input data
    to ensure it conforms to the expected numeric type and optionally focuses
    on specified columns when dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the mode. Can be a list, 
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all 
        numeric columns are considered.
    axis: optional, {index (0), columns {1}}
       Axis for the function to be applied on.  Default is 0.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding. This parameter is particularly relevant when input
        data is not already a DataFrame.
    view : bool, default False
        If True, visualizes the mode(s) of the dataset along with its distribution.
    cmap : str, default 'viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization. Only applicable if `view` is True.
    **kws : dict
        Additional keyword arguments for further data processing or passed to 
        the underlying statistical functions.
    **kws : dict
        Additional keyword arguments for further data processing or passed to 
        underlying statistical functions.

    Returns
    -------
    mode_value : np.ndarray or pd.Series
        The mode(s) of the dataset. Returns a numpy array if the input is 
        ArrayLike or if `as_frame` is False.
        Returns a pandas Series if the input is a DataFrame and 
        `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.descriptive import mode
    >>> data_array = [1, 2, 2, 3, 4]
    >>> mode(data_array)
    2

    >>> data_df = pd.DataFrame({'A': [1, 2, 2, 3], 'B': [4, 4, 5, 5]})
    >>> mode(data_df, as_frame=True)
    A    2
    B    4
    dtype: int64

    Note
    ----
    This function is enhanced by the `make_data_dynamic` decorator, which 
    preprocesses the input data to align with the expected numeric type and 
    selectively processes columns if specified. This preprocessing
    ensures that statistical calculations are performed accurately and 
    efficiently on DataFrame inputs with mixed data types.
    """
    as_frame,  = ellipsis2false(as_frame)
    if isinstance(data, pd.DataFrame):
        axis = axis or 0
        mode_result = data.mode(axis=axis, **kws)
    else:
        data = np.asarray(data)
        mode_result = stats.mode(data, axis=axis,  **kws).mode[0]
        mode_result = pd.Series(mode_result) if as_frame else mode_result
    
    mode_result, view = ensure_visualization_compatibility(
        mode_result, as_frame, view, mode )
    if view:
        _visualize_mode(data, mode_result, cmap=cmap, fig_size=fig_size, axis=axis )
    
    return mode_result

def _visualize_mode(data, mode_result, cmap='viridis', fig_size=None, axis=0):
    """
    Visualizes the data distribution and highlights the mode values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= prepare_stats_plot(mode_result, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate (cols) :
            plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                     label=f'{col} Distribution', color=colors[ii])
            for mode in np.ravel([mode_result[col]]):
                plt.axvline(x=mode, color='red', linestyle='dashed',
                            linewidth=2, label=f'{col} Mode')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution')
        for mode in np.ravel([mode_result]):
            plt.axvline(x=mode, color='red', linestyle='dashed', 
                        linewidth=2, label='Mode')
    plt.title('Data Distribution and Mode(s)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
@make_data_dynamic(capture_columns=True, reset_index=True)
def var(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: Optional[int] = None, 
    ddof: int = 1, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    r"""
    Calculates the variance of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. This function is designed to be 
    flexible, allowing for the calculation of variance across entire datasets 
    or within specified columns of a DataFrame.

    Utilizing preprocessing parameters such as `expected_type` and 
    `capture_columns`, the input data is first sanitized to ensure it contains 
    only numeric values. This preprocessing step is essential for accurate 
    statistical analysis.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the variance. Can be a list, 
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    axis : optional, {index (0), columns (1)}
        Axis for the function to be applied on. Default is 0.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in the calculation is 
        N - ddof, where N represents the number of elements. By default, ddof 
        is set to 1, which computes the sample variance. If ddof is set to 0, 
        the population variance is calculated. 
        
        .. math:: \text{Variance} = \frac{\sum (x_i - \bar{x})^2}{N - ddof}
 
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. This parameter is particularly relevant 
        when input data is not already a DataFrame.
    view : bool, default False
        If True, visualizes the variance(s) of the dataset along with its distribution.
    cmap : str, default 'viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization. Only applicable if `view` is True.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy variance 
        function, allowing customization of the variance calculation, such as 
        specifying the axis or whether to use Bessel's correction.
    Returns
    -------
    variance_value : float or np.ndarray or pd.Series
        The calculated variance of the provided data. Returns a single float 
        if the data is one-dimensional, a numpy array if an axis is specified 
        in `kws`, or a pandas Series if the input is a DataFrame and `as_frame`
        is True.

    Examples
    --------
    >>> from gofast.stats.descriptive import var
    >>> data_array = [1, 2, 3, 4, 5]
    >>> var(data_array)
    2.0

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> var(data_df, columns=['A'], ddof=1)
    1.0

    >>> var(data_df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64
    
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> var(df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> var(df, view=True, fig_size=(8, 4))
    Note
    ----
    The preprocessing steps, controlled by the `make_data_dynamic` decorator,
    ensure that the input data is suitable for variance calculation. This
    includes converting the data to numeric types and filtering based on 
    specified columns, enhancing the function's flexibility and applicability 
    across various data forms.
    """
    if isinstance(data, pd.DataFrame):
        axis = axis or 0 # Pandas default ddof=1
        variance_result = data.var(ddof=ddof, axis=axis, **kws)  
    else:
        data = np.asarray(data)
        # Ensure consistency with pandas
        variance_result = np.var(data, ddof=ddof, axis=axis, **kws)  

    if as_frame and not isinstance(data, pd.DataFrame):
        variance_result = pd.Series(
            variance_result, index=columns if columns else ['Variance'])

    variance_result, view = ensure_visualization_compatibility(
        variance_result, as_frame, view, var )
    
    if view:
        _visualize_variance(data, variance_result, columns=columns,
                            cmap=cmap, fig_size=fig_size, axis=axis )
    return variance_result

def _visualize_variance(data, variance_result, columns=None, cmap='viridis',
                        fig_size=None, axis=0 ):
    """
    Visualizes the distribution of the data and highlights the variance values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= prepare_stats_plot(variance_result, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate(cols):
            plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                     label=f'{col} Distribution', color=colors[ii])
        for col, var in variance_result.items():
            plt.axvline(x=np.sqrt(var), color='red', 
                        linestyle='dashed', linewidth=2, 
                        label=f'{col} StdDev')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution', 
                 )
        plt.axvline(x=np.sqrt(variance_result), color='red',
                    linestyle='dashed', linewidth=2, label='StdDev')
    plt.title('Data Distribution and Standard Deviation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

@make_data_dynamic(
    capture_columns=True, 
    reset_index=True
    )    
def std(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: Optional[int] = None, 
    ddof: int = 1, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    r"""
    Computes the standard deviation of numeric data provided either as an 
    array-like structure or within a pandas DataFrame. This function is 
    designed to be flexible, allowing for the calculation of standard deviation 
    across entire datasets or within specified columns of a DataFrame.

    Utilizing preprocessing parameters such as `expected_type` and 
    `capture_columns`, the input data is first sanitized to ensure it contains 
    only numeric values. This preprocessing step is essential for accurate 
    statistical analysis.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the standard deviation. Can be 
        a list, numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    axis : optional, {index (0), columns (1)}
        Axis for the function to be applied on. Default is 0.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in the calculation is 
        N - ddof, where N represents the number of elements. By default, ddof 
        is set to 1, which computes the sample standard deviation. If ddof is 
        set to 0, the population standard deviation is calculated. The standard 
        deviation is the square root of variance:
            
        .. math:: 
            
           \text{Standard Deviation} = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N - ddof}}
       
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. This parameter is relevant when 
        input data is not already a DataFrame.
    view : bool, default False
        If True, visualizes the data distribution and highlights the 
        standard deviation. Applicable only if `view` is True.
    cmap : str, default 'viridis'
        Colormap for the visualization. Applicable only if `view` is True.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for visualization. Applicable only if `view` is True.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy standard 
        deviation function, allowing customization of the calculation, such 
        as specifying the axis or whether to use Bessel's correction.

    Returns
    -------
    std_dev_value : float or np.ndarray or pd.Series
        The calculated standard deviation of the provided data. Returns a single 
        float if the data is one-dimensional, an np.ndarray if an axis is specified 
        in `kws`, or a pandas Series if the input is a DataFrame and `as_frame` 
        is True.

    Examples
    --------
    >>> from gofast.stats.descriptive import std
    >>> data_array = [1, 2, 3, 4, 5]
    >>> std(data_array)
    1.4142135623730951

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> std(data_df, columns=['A'], ddof=1)
    1.0

    >>> std(data_df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> std(df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> std(df, view=True, fig_size=(8, 4))
    Note
    ----
    The preprocessing steps, implied by the `make_data_dynamic` decorator 
    (not shown here but assumed to be applied outside this code snippet), 
    ensure that the input data is suitable for standard deviation calculation.
    These steps convert the data to numeric types and filter based on specified 
    columns, thus enhancing the function's utility and applicability across 
    diverse data forms.
    """
    # Handling DataFrame input with captured columns
    if isinstance(data, pd.DataFrame):
        axis = axis or 0
        # Pandas defaults ddof=1
        std_dev_result = data.std(ddof=ddof, axis=axis,  **kws)  
    else:
        # Convert ArrayLike to np.ndarray for consistent processing
        # In the case frame conversion failed, ensure consistency with pandas 
        data = np.array(data)
        std_dev_result = np.std(data, ddof=ddof, axis=axis **kws)  
    
    # Visualization of results if requested
    std_dev_result, view = ensure_visualization_compatibility(
        std_dev_result, as_frame, view, std )
    if view:
        _visualize_std_dev(data, std_dev_result, cmap=cmap, fig_size=fig_size, 
                           axis=axis )
    # Convert result to Series if as_frame=True
    if as_frame and not isinstance(std_dev_result, pd.DataFrame):
        if len(std_dev_result)==1: 
            std_dev_result = to_series_if(
                std_dev_result, value_names =['Standard Deviation']) 
        else:
            std_dev_result = pd.Series(
                np.array(std_dev_result), index=columns )

    return std_dev_result

def _visualize_std_dev(data, std_dev_result, cmap='viridis', fig_size=None,
                       axis=0 ):
    """
    Visualizes the distribution of the data and highlights the standard deviation.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= prepare_stats_plot(std_dev_result, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate(cols):
            plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                     label=f'{col} Distribution', color=colors[ii])
            plt.axvline(x=data[col].mean(), color='red', linestyle='dashed', 
                        linewidth=2, label=f'{col} Mean')
            plt.axvline(x=data[col].mean() + data[col].std(), color='green',
                        linestyle='dashed', linewidth=2, label=f'{col} + Std Dev')
            plt.axvline(x=data[col].mean() - data[col].std(), color='green', 
                        linestyle='dashed', linewidth=2, label=f'{col} - Std Dev')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution',)
        mean = np.mean(data)
        std_dev = np.std(data)
        plt.axvline(x=mean, color='red', linestyle='dashed', linewidth=2, 
                    label='Mean')
        plt.axvline(x=mean + std_dev, color='green', linestyle='dashed', 
                    linewidth=2, label='+ Std Dev')
        plt.axvline(x=mean - std_dev, color='red',
                    linestyle='dashed', linewidth=2, label='- Std Dev', 
                    )
    plt.title('Data Distribution and Standard Deviation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
        
@make_data_dynamic(capture_columns=True)  
def quartiles(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: int =0, 
    as_frame: bool = True,
    view: bool = False, 
    plot_type: str = 'box',  
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    """
    Calculates the quartiles (25th, 50th, and 75th percentiles) of numeric data 
    provided either as an array-like structure or within a pandas DataFrame.
    
    This function is versatile, allowing calculations across entire datasets or 
    within specified columns of a DataFrame.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate quartiles. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    axis : optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    view : bool, default False
        If True, visualizes the quartiles of the dataset along with 
        its distribution.
    plot_type : str, default 'box'
        Type of plot for visualization. Options are 'box' for box plots and 
        'hist' for histograms. Only applicable if `view` is True.
    cmap : str, default 'viridis'
        Colormap for the visualization.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization.
        
    **kws : dict
        Additional keyword arguments passed to the numpy percentile function, 
        allowing customization of the quartile calculation, such as specifying 
        the interpolation method.

    Returns
    -------
    quartile_values : np.ndarray or pd.DataFrame
        The calculated quartiles of the provided data. Returns a numpy array 
        with the quartiles if the data is one-dimensional or a pandas DataFrame 
        if the input is a DataFrame and `as_frame` is True, showing the quartiles 
        for each specified column.

    Examples
    --------
    >>> from gofast.stats.descriptive import quartiles
    >>> data_array = [1, 2, 3, 4, 5]
    >>> quartiles(data_array)
    array([2., 3., 4.])

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quartiles(data_df, columns=['A'])
    array([4.25, 6. , 7.75])

    >>> quartiles(data_df, as_frame=True)
        25%   50%   75%
    A  4.25  6.0  7.75
    B  2.75  5.0  7.25
    
    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> print(quartiles(data_df, columns=['A'], as_frame=True))
          A
    25%  4.25
    50%  6.00
    75%  7.75

    >>> quartiles(data_df, view=True, fig_size=(8, 4))
    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quartiles(data_df, view=True, plot_type='box', fig_size=(8, 4))
    >>> quartiles(data_df, view=True, plot_type='hist', fig_size=(8, 4))
    Note
    ----
    Quartiles are a fundamental statistical measure used to understand the spread 
    and center of a dataset. By offering the flexibility to compute quartiles for 
    specific columns or entire datasets, this function aids in comprehensive data 
    analysis, particularly in exploratory data analysis (EDA) and data visualization.
    """

    if isinstance(data, pd.DataFrame):
        data_selected = data.copy() 
        quartiles_result = data_selected.quantile(
            [0.25, 0.5, 0.75], axis = axis,  **kws)
    else:
        data_selected = np.asarray(data)
        quartiles_result = np.percentile(
            data_selected, [25, 50, 75], axis =axis, **kws)

    quartiles_result, view  = ensure_visualization_compatibility(
        quartiles_result, as_frame, view , func_name=quartiles )
    if view:
        _visualize_quartiles(quartiles_result, data, plot_type=plot_type,
                             cmap=cmap, fig_size=fig_size, axis=axis )
    if as_frame: 
        quartiles_result=update_index(
            quartiles_result, 
            new_indexes = ['25%', '50%', '75%'], 
            return_data= True, 
            allow_replace=True
            ).T 
    else: 
        quartiles_result = np.squeeze (np.asarray (quartiles_result)).T

    return quartiles_result 

def _visualize_quartiles(
        quartiles_result, data, plot_type, cmap, fig_size, axis=0):
    """
    Visualizes quartiles using the specified plot type.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    quartiles_result, data, cols = prepare_stats_plot(
        quartiles_result, data , axis=axis )

    colors, alphas = get_colors_and_alphas(
        cols, cmap, convert_to_named_color=True)
    if plot_type == 'box':
        if isinstance(quartiles_result, pd.DataFrame):
            quartiles_result.boxplot(grid=True, color=colors[0])
        else:
            plt.boxplot(quartiles_result, patch_artist=True, 
                        boxprops=dict(facecolor=colors[0]))
    elif plot_type == 'hist':
        if isinstance(quartiles_result, pd.DataFrame):
            for ii, col in enumerate (cols):
                plt.hist(quartiles_result[col], bins=30, alpha=alphas[ii],
                         label=f'{col} Distribution', color=colors [ii])
        else:
            plt.hist(quartiles_result, bins=30, alpha=0.5, color=cmap)
            
    plt.title('Data Distribution')
    plt.ylabel('Frequency' if plot_type == 'hist' else 'Value')
    plt.legend() if plot_type == 'hist' and isinstance(
        data, pd.DataFrame) else None
    plt.show()

@DynamicMethod (capture_columns=True)
def quantile(
    data: Union[ArrayLike, DataFrame], 
    q: Union[float, List[float]], 
    columns: Optional[List[str]] = None, 
    axis: Optional [int] = None, 
    as_frame: bool = True, 
    view: bool = False, 
    plot_type: Optional[str] = 'box', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    """
    Computes specified quantiles of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. 
    
    Function is designed to offer flexibility in calculating quantiles across 
    entire datasets or within specified columns of a DataFrame.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to compute quantiles. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    q : float or list of float
        Quantile or sequence of quantiles to compute, which must be between 0 
        and 1 inclusive.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    axis : optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
    as_frame : bool, default=True
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    view : bool, default False
        If True, visualizes the quartiles of the dataset along with 
        its distribution.
    plot_type : str, default 'box'
        Type of plot for visualization. Options are 'box' for box plots and 
        'hist' for histograms. Only applicable if `view` is True.
    cmap : str, default 'viridis'
        Colormap for the visualization.
        
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization.
    **kws : dict
        Additional keyword arguments passed to the numpy quantile function, 
        allowing customization of the quantile calculation, such as specifying 
        the interpolation method.

    Returns
    -------
    quantile_values : float, np.ndarray, or pd.DataFrame
        The computed quantile(s) of the provided data. Returns a single float if 
        `q` is a single quantile value and the data is one-dimensional, an 
        np.ndarray if `q` is a list and the data is one-dimensional, or a pandas 
        DataFrame if the input is a DataFrame and `as_frame` is True.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.stats.descriptive import quantile
    >>> data_array = [1, 2, 3, 4, 5]
    >>> quantile(data_array, q=0.5)
    3.0

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quantile(data_df, q=[0.25, 0.75], columns=['A'])
    array([3.75, 7.25])

    >>> quantile(data_df, q=0.5, as_frame=True)
       50%
    A  6.0
    B  5.0
    >>> quantile(data_array, q=0.5, view=True, plot_type='hist')

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quantile(data_df, q=[0.25, 0.75], as_frame=True, view=True, plot_type='box')
    
    Note
    ----
    The function provides a convenient way to compute quantiles, a critical 
    statistical measure for understanding the distribution of data. The 
    flexibility to compute quantiles for specific columns or entire datasets 
    enhances its utility in exploratory data analysis and data preprocessing.
    """
    axis = axis or 0
    data_selected = data.copy() 
    # Compute the quantile or quantiles
    quantiles_result = data_selected.quantile(q, axis=axis, **kws)
  
    if view: # visualize box/hist quantiles
        _visualize_quantiles(
            data, q, quantiles_result,
            plot_type=plot_type, cmap=cmap, 
            fig_size=fig_size, axis= axis
        )
    # update data frame indexes with string quantiles
    new_indexes= [f'{int(q*100)}%' for q in np.atleast_1d(q)]

    quantiles_result=update_index( 
        quantiles_result, 
        new_indexes = new_indexes, 
        return_data= True, 
        allow_replace=True
        ).T 
    # convert to series if applicable 
    quantiles_result = convert_and_format_data(
        quantiles_result,
        as_frame, 
        allow_series_conversion=False if as_frame else True,
        series_name=new_indexes[0], 
        condense=True, 
        condition =lambda x:  {
            "force_array_output": True}  if not as_frame else {} 
        )
    return quantiles_result

def _visualize_quantiles(
    data, q, 
    quantiles_result, 
    plot_type, 
    cmap, 
    fig_size,
    axis=0 
    ):
    """
    Visualizes the data and highlights the quantiles, based on the 
    specified plot type.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The dataset from which quantiles were computed.
    q : float or list of float
        The quantile(s) that were computed.
    quantiles_result : pd.DataFrame or np.ndarray
        The computed quantiles.
    plot_type : str
        The type of plot to generate ('box' or 'hist').
    cmap : str
        The colormap name to use for coloring the plot.
    fig_size : tuple of int
        The size of the figure to create.
    axis : int, default 0
        The axis along which the quantiles were computed.

    Examples
    --------
    >>> data = pd.DataFrame({'A': np.random.normal(
        size=100), 'B': np.random.normal(size=100)})
    >>> q = [0.25, 0.5, 0.75]
    >>> quantiles_result = data.quantile(q)
    >>> _visualize_quantiles(data, q, quantiles_result, 'box', 'viridis', (10, 6))
    """
    plt.figure(figsize=fig_size if fig_size else (4, 4))
    
    q_list = np.atleast_1d(q)
    _, data, cols = prepare_stats_plot(
        quantiles_result, data , axis=axis )

    colors, alphas = get_colors_and_alphas(
        cols, cmap, convert_to_named_color=True)
    
    if plot_type == 'box':
        try: 
            data.boxplot( grid=True, color=colors[0])
        except: 
            plt.boxplot(data, patch_artist=True, 
                        boxprops=dict(facecolor=colors[0]))
        plt.title('Boxplot and Quantiles')
        
    elif plot_type == 'hist':
        try: 
            plt.hist(data, bins=30, alpha=0.5, color=colors[: len(cols)])
            for quantile in q_list:
                plt.axvline(np.quantile(data, quantile), color='r',
                            linestyle='--', label=f'{quantile*100}% Quantile')
        except: 
            for ii, col in enumerate (cols):
                plt.hist(data[col], 
                          bins=30,
                          alpha=alphas[ii],
                          label=f'{col} Distribution',
                          color=colors[ii])
                for quantile in q_list:
                    plt.axvline(data[col].quantile(quantile),
                                color='r', linestyle='--',
                                label=f'{quantile*100}% Quantile')
          
        plt.title('Histogram and Quantiles')
        
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Only add legend if labels are present
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    plt.show()
    
@make_data_dynamic(capture_columns=True)
def median(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
    axis: Optional[int]=None, 
    view: bool = False, 
    cmap: str = 'viridis', 
    as_frame: bool = False, 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
):
    """
    Calculates the median of numeric data provided either as an array-like 
    structure or within a pandas DataFrame.

    Leveraging the `make_data_dynamic` decorator, the function preprocesses 
    the input data to ensure it conforms to the expected numeric type and 
    optionally focuses on specified columns when dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the median. Can be a list,
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a
        DataFrame. If not provided, all numeric columns are considered.
    axis: optional, {index (0), columns {1}}
       Axis for the function to be applied on.  Default is 0.
    view : bool, default=False
        If True, visualizes the distribution of the data and highlights the 
        calculated median values.
    cmap : str, default='viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    as_frame : bool, default=False
        If True, the result is returned as a pandas DataFrame or Series, depending
        on the dimensionality of the calculated median. Otherwise, the result is
        returned in its native format (float or np.ndarray).
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the visualization. Only applicable if `view` is True.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy median 
        function, allowing customization of the median calculation, such as 
        specifying the axis or whether to ignore NaN values.

    Returns
    -------
    median_value : float or np.ndarray
        median_value : float, np.ndarray, pd.Series, or pd.DataFrame
            The calculated median of the provided data. The format of the return 
            value depends on the input data dimensionality and the
            `as_frame` parameter.

    Examples
    --------
    >>> from gofast.stats.descriptive import median
    >>> data_array = [3, 1, 4, 1, 5]
    >>> median(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [2, 4, 7], 'B': [1, 6, 5]})
    >>> median(data_df, columns=['A'])
    4.0

    >>> median(data_df, axis=0)
    array([4., 5.])
    
    Calculating median from a DataFrame and converting to Series:

    >>> df = pd.DataFrame({'A': [2, 4, 7], 'B': [1, 6, 5]})
    >>> print(median(df, columns=['A'], as_frame=True))
    A    4.0
    dtype: float64

    Visualizing data distribution and median from DataFrame:

    >>> median(df, view=True)
    
    Note
    ----
    This function is enhanced by the `make_data_dynamic` decorator, which 
    preprocesses the input data to align with the expected numeric type and 
    selectively processes columns if specified. This preprocessing
    is particularly useful for ensuring compatibility with statistical 
    calculations and optimizing performance for DataFrame inputs with mixed
    data types.
    """
    # Preprocess data based on input type and selected columns
    # if isinstance(data, pd.DataFrame) and columns is not None:
    #     data = data[columns]
    # Calculate the median
    if isinstance(data, pd.DataFrame):
        axis = axis or 0
        median_values = data.median(axis=axis, **kws)
    else:
        data = np.array(data)
        median_values = np.median(data, axis=axis, **kws)
    
    # Convert the result to DataFrame or Series if requested
    if as_frame:
        if isinstance(median_values, np.ndarray) and median_values.ndim > 1:
            median_values = pd.DataFrame(median_values, columns=columns)
        else:
            median_values = pd.Series(
            median_values,
            index=columns if columns else None, 
            name='Median'
        )

    # Visualization of data distribution and median
    median_values, view = ensure_visualization_compatibility(
        median_values, as_frame, view, median )
    if view:
        _visualize_median(
            data, median_values, cmap=cmap, fig_size=fig_size, axis= axis )

    return median_values

def _visualize_median(
        data, median_values, cmap='viridis', fig_size=None, axis = 0 ):
    """
    Visualizes the distribution of the data and highlights the median values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= prepare_stats_plot(median_values, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        # Generate distinct colors for each column from cmap 
        colors, alphas = get_colors_and_alphas( cols, cmap) 
        for ii, col in enumerate (cols) :
            plt.hist(data[col], bins='auto', alpha=alphas[ii],
                     label=f'{col} Distribution', color=colors [ii])
            plt.axvline(data[col].median(), color='red', linestyle='dashed', 
                        linewidth=2, label=f'{col} Median')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution',
                 )
        plt.axvline(np.median(data), color='red', linestyle='dashed',
                    linewidth=2, label='Median')
    plt.title('Data Distribution and Median')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

@make_data_dynamic(capture_columns=True)
def mean(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
    axis: int =None, 
    view: bool = False, 
    cmap: str = 'viridis', 
    as_frame: bool = False, 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
):
    """
    Calculates the mean of numeric data provided as an array-like structure 
    or within a pandas DataFrame.
    
    Optionally, the result can be returned as a pandas DataFrame or Series, 
    and the data distribution along with the mean can be visualized.

    The function dynamically preprocesses the input data based on the 
    `expected_type` and `capture_columns` parameters specified by the 
    `make_data_dynamic` decorator. For DataFrames, if `columns` are specified,
    only those columns are considered for calculating the mean.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the mean. Can be a list, 
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider in the calculation if `data` is a
        DataFrame. If not provided, all numeric columns are considered.
    axis: optional, {index (0), columns {1}}
       Axis for the function to be applied on.  Default is 0.
    view : bool, default=False
        If True, visualizes the distribution of the data and highlights the
        calculated mean values.
    cmap : str, default='viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    as_frame : bool, default=False
        If True, the result is returned as a pandas DataFrame or Series, 
        depending on the dimensionality of the calculated mean. Otherwise, 
        the result is returned in its native format (float or np.ndarray).
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the visualization. Only applicable if `view` is True.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy mean 
        function, allowing customization of the mean calculation, 
        such as specifying the axis.
         
    **kws : dict
        Additional keyword arguments passed to the underlying numpy mean 
        function, allowing customization of the mean calculation, such as 
        specifying the axis.

    Returns
    -------
    mean_value : float, np.ndarray, pd.Series, or pd.DataFrame
        The calculated mean of the provided data. The format of the return 
        value depends on the input data dimensionality and the 
        `as_frame` parameter.

    Examples
    --------
    >>> from gofast.stats.descriptive import mean
    >>> data_array = [1, 2, 3, 4, 5]
    >>> mean(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> mean(data_df, columns=['A'])
    2.0

    >>> mean(data_df, axis=0)
    array([2., 5.])
    
    Calculating mean from a list:

    >>> mean([1, 2, 3, 4, 5])
    3.0

    Calculating mean from a DataFrame and converting to Series:

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> print(mean(df, as_frame=True))
    A    2.0
    B    5.0
    dtype: float64

    Visualizing data distribution and mean from DataFrame:

    >>> mean(df, view=True)
    Note
    ----
    The function is wrapped by the `make_data_dynamic` decorator, which 
    preprocesses the input data based on the specified `expected_type` and 
    `capture_columns`. This preprocessing step is crucial for handling pandas 
    DataFrames with mixed data types and for selecting specific columns for
    the calculation.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        axis = axis or 0
        mean_values = data.mean(axis =axis, **kws)
    else:
        data = np.array(data)
        mean_values = np.mean(data, axis =axis,  **kws)
    
    if as_frame:
        if isinstance(mean_values, np.ndarray) and mean_values.ndim > 1:
            mean_values = pd.DataFrame(mean_values, columns=columns)
        else:
            mean_values = pd.Series(
                mean_values, index=columns if columns else None, name='Mean')
    
    mean_values, view = ensure_visualization_compatibility(
        mean_values, as_frame, view, mean )
    
    if view:
        _visualize_mean(
            data, mean_values, cmap=cmap, fig_size=fig_size, axis=axis)
        
   
    return mean_values

def _visualize_mean(
        data, mean_values, cmap='viridis', fig_size=None, axis =0 ):
    """
    Visualizes the distribution of the data and highlights the mean values 
    with distinct colors.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data containing the columns to be visualized.
    mean_values : pd.Series or np.ndarray
        The mean values for the columns in the data.
    cmap : str, optional
        The colormap name to be used if `data` is not a DataFrame.
    fig_size : tuple, optional
        The size of the figure to be created.

    Returns
    -------
    None
        This function does not return any value. It shows a plot.
    
    Examples
    --------
    >>> df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100) + 1})
    >>> _visualize_mean(df, df.mean())
    """
    
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    mean_values, data, cols= prepare_stats_plot(mean_values, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        # Generate distinct colors for each column from cmap 
        colors, alphas = get_colors_and_alphas( cols, cmap) 
        for ii, col in enumerate(cols):
            # Get the histogram data
            n, bins, patches = plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                                        label=f'{col} Distribution', color=colors[ii])
            # Highlight the mean
            mean_val = mean_values[col]
            plt.axvline(mean_val, color=colors[ii], linestyle='dashed',
                        linewidth=2, label=f'{col} Mean')
            
            # Calculate the height of the mean line for placing the annotation
            mean_bin_index = np.digitize(mean_val, bins) - 1
            mean_height = n[mean_bin_index] if 0 <= mean_bin_index < len(n) else 0
            
            # Add annotation for the mean value
            plt.text(mean_val, mean_height, f'{mean_val:.2f}',
                     color=colors[ii], ha='center', va='bottom')
    else:
        data =flatten_data_if(data, squeeze= True)
        # If `data` is not a DataFrame, use a single color from the colormap
        n, bins, patches = plt.hist(
            data, bins='auto', alpha=0.7, label='Data Distribution')
        # Ensure mean_val is a scalar for annotation
        mean_val =np.mean(data)
        if isinstance(mean_val, np.ndarray):
            mean_val = mean_val.item()
        elif isinstance(mean_val, pd.Series):
            mean_val = mean_val.iloc[0]
        plt.axvline(mean_val, color='red', linestyle='dashed', 
                    linewidth=2, label='Mean')
        # Add annotation for the mean value
        mean_bin_index = np.digitize(mean_val, bins) - 1
        mean_height = n[mean_bin_index] if 0 <= mean_bin_index < len(n) else 0
        plt.text(mean_val, mean_height, f'{mean_val:.2f}',
                 color='red', ha='center', va='bottom')
    
    plt.title('Data Distribution and Mean')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

@DynamicMethod ( 
    expected_type="both", 
    capture_columns=True, 
    drop_na=True
   )
def iqr(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None,
    axis: Optional[int] =None, 
    as_frame: bool = True,
    view: bool = False, 
    plot_type: Optional[str] = 'boxplot', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (4, 4), 
    orient: Optional [str] =None, 
    **kws):
    """
    Computes the interquartile range (IQR) for numeric data and optionally 
    visualizes it.

    Parameters
    ----------
    data: Union[ArrayLike, pd.DataFrame]
        The dataset from which the IQR is calculated. This can be directly 
        provided as a list, numpy array, or a pandas DataFrame. When provided 
        as a list or numpy array, the data is internally converted to a pandas 
        DataFrame for calculation.
    
    columns : Optional[List[str]] (default=None)
        Relevant when `data` is a pandas DataFrame. Specifies the columns within 
        the DataFrame for which the IQR is to be calculated. If None, the IQR 
        is calculated for all numeric columns in the DataFrame.
        
    axis: optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
         
    as_frame : bool (default=False)
        Determines the format of the return value. When True, the function returns 
        a pandas Series object representing the IQR for each column specified or 
        for all numeric columns by default. When False, the function returns a 
        single float value representing the IQR of the data or the first column 
        if `data` is a DataFrame.

    view : bool (default=False)
        If True, the function generates a plot visualizing the distribution of 
        the data along with the IQR. The type of plot generated is determined 
        by the `plot_type` parameter.

    plot_type : Optional[str] (default='boxplot')
        Specifies the type of plot to use for visualization when `view` is True. 
        Currently supports 'boxplot', which displays a box and whisker plot 
        showing the quartiles of the dataset along with outliers. Additional 
        plot types can be supported by extending the function.

    cmap : str (default='viridis')
        The colormap to use for the visualization when `view` is True. This parameter 
        is passed directly to the plotting function to customize the color scheme 
        of the plot.

    fig_size : Optional[Tuple[int, int]] (default=(8, 6))
        Specifies the size of the figure for the visualization plot when `view` is 
        True. Provided as a tuple (width, height) in inches.

    orient: Optional [str], optional 
       Orientation of box plot either vertical ( default) or horizontal 'h'. 
       
    **kws : dict
        Additional keyword arguments passed directly to the numpy percentile 
        function when calculating the 25th and 75th percentiles. This allows 
        for customization of the percentile calculation, such as specifying 
        the interpolation method.

    Returns
    -------
    iqr_value : float or pd.Series
        The calculated IQR of the provided data. Returns a single float if 
        the data is one-dimensional or if `as_frame` is False. Returns a pandas 
        Series if the input is a DataFrame and `as_frame` is True.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.descriptive import iqr
    >>> data_array = [1, 2, 3, 4, 5]
    >>> iqr(data_array)
    2.0

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> iqr(data_df, columns=['A'])
    3.5

    >>> iqr(data_df, as_frame=True)
    A    3.5
    B    4.0
    dtype: float64
    >>> data_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> print("IQR for array:", iqr(data_array, view=True, plot_type='boxplot'))

    >>> data_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(1, 2, 100),
        'C': np.random.normal(2, 3, 100)
    })
    >>> print("IQR for DataFrame:", iqr(data_df, as_frame=True))
    >>> iqr(data_df, view=True, plot_type='boxplot')
    
    Note
    ----
    The IQR is a robust measure of spread that is less influenced by outliers 
    than the range. This function simplifies the process of calculating the IQR, 
    especially useful in exploratory data analysis and for identifying potential 
    outliers.
    """
    # Calculate IQR
    Q1 = data.quantile(0.25, axis = axis or 0 , **kws)
    Q3 = data.quantile(0.75, axis = axis or 0 , **kws)
    iqr_values = Q3 - Q1
 
    # Visualization
    if view:
        _, data, cols = prepare_stats_plot(
            iqr_values, data, axis=axis )
        plt.figure(figsize=fig_size)
        if isinstance(data, pd.DataFrame):
            sns.boxplot(data=data, orient=orient, palette=cmap)
        else:
            sns.boxplot(data=pd.DataFrame(data), orient=orient, palette=cmap)
        plt.title('IQR Visualization with Boxplot')
        plt.show()
        
    iqr_values = convert_and_format_data(
        iqr_values, as_frame, 
        force_array_output= True if not as_frame else False, 
        condense= True, 
        condition= series_naming("IQR")
        )
    return iqr_values

@make_data_dynamic('numeric', capture_columns=True)
def hmean(
    data: ArrayLike,
    columns: List[str] = None,
    as_frame: bool = True,
    axis=None, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (8, 4)
):
    """
    Calculate the harmonic mean of a data set and optionally visualize the 
    data distribution through a histogram.

    The harmonic mean, useful for rates and ratios, is less influenced by
    large outliers compared to the arithmetic mean. It is defined as the 
    reciprocal of the arithmetic mean of the reciprocals of the data points.

    .. math::
        H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}

    Where :math:`H` is the harmonic mean, :math:`n` is the number of observations, 
    and :math:`x_i` are the data points.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        The data for which the harmonic mean is desired. Can be a pandas DataFrame 
        or a numpy array. If a DataFrame and `columns` is specified, only the 
        selected columns are used for the calculation.
    columns : List[str], optional
        Specific columns to use for the calculation if `data` is a DataFrame.
    as_frame : bool, optional
        If True and `data` is ArrayLike, converts `data` into a DataFrame
        using `columns` as column names. Ignored if `data` is already a DataFrame.
    axis : {None, 0, 1}, optional
        Axis along which to compute the harmonic mean:
        - `None`: Harmonic mean is calculated using all values in the input 
          array, flattening it if necessary. This is the default behavior.
        - `0`: Compute the harmonic mean along columns for a DataFrame or a 
          2D numpy array. This results in a harmonic mean value for each column.
        - `1`: Compute the harmonic mean along rows for a DataFrame or a 2D 
          numpy array. This results in a harmonic mean value for each row.
        If `data` is a DataFrame and specific `columns` are selected, the 
        computation will consider only the selected columns. If `as_frame` 
        is True, the result will be returned as a pandas DataFrame; otherwise, 
        it will be returned in the form of a numpy array or a scalar value, 
        depending on the dimensionality of the input.
    view : bool, optional
        If True, displays a histogram of the data set to visualize its
        distribution. Default is False.
    cmap : str, optional
        Colormap for the histogram. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the histogram. Default is (10, 6).

    Returns
    -------
    h_mean : float
        The harmonic mean of the data set.

    Raises
    ------
    ValueError
        If any data point is less than or equal to zero.

    Examples
    --------
    >>> from gofast.stats.descriptive import hmean
    >>> hmean([1, 2, 4])
    1.7142857142857142

    >>> hmean(np.array([2.5, 3.0, 10.0]))
    3.5294117647058822
    
    >>> df = pd.DataFrame({'A': [2.5, 3.0, 10.0], 'B': [1.5, 2.0, 8.0]})
    >>> hmean(df, columns=['A'])
    3.5999999999999996
   
    See Also
    --------
    scipy.stats.hmean : Harmonic mean function in SciPy for one-dimensional arrays.
    gofast.stats.mean : Arithmetic mean function.
    """
    #select only numeric features 
    data = data.select_dtypes (include = [np.number])
    if np.any(data <= 0):
        raise ValueError("Data points must be greater than 0 for harmonic"
                         " mean calculation.")
    
    if axis is None:
        data_values = data.values.flatten()
        h_mean = stats.hmean(data_values)
    elif axis in [0, 1]:
        h_mean = data.apply(lambda x: stats.hmean(x.dropna()), axis=axis)
        # h_mean = h_mean.to_frame(
        #     name='harmonic_mean').T if axis == 0 else h_mean.to_frame(
        #         name='harmonic_mean')
    if view:
        # Handling visualization
        visualize_data_distribution(data, h_mean, cmap, fig_size)
    
    if as_frame: 
        if axis is None: 
            return to_series_if (h_mean, value_names= ['H-mean'],
                                 name='harmonic_mean') 

        h_mean= convert_and_format_data(
            h_mean, as_frame, 
            force_array_output= False if as_frame else True, 
            condense=False if as_frame else True,
            condition=series_naming ("harmonic_mean"),
            )    
    return h_mean

def visualize_data_distribution(data, h_mean, cmap, fig_size):
    plt.figure(figsize=fig_size)
    
    def plot_mean_line(mean_value, label):
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
        plt.text(mean_value, plt.ylim()[1] * 0.9, f'{label}: {mean_value:.2f}',
                 rotation=90, verticalalignment='center')

    # Check and handle if data is a DataFrame and multiple columns are provided
    if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
        data = data.values.flatten()  # Flatten the array if it's multi-dimensional

    num_datasets = 1 if data.ndim == 1 else data.shape[1]
    colors = [plt.get_cmap(cmap)(i/num_datasets) for i in range(num_datasets)]
    
    plt.hist(data, bins='auto', color=colors, alpha=0.7, rwidth=0.85)
    plt.title('Data Distribution')
    plt.xlabel('Data Points')
    plt.ylabel('Frequency')

    # Handling different structures of h_mean (Series or DataFrame)
    if isinstance(h_mean, pd.DataFrame):
        for col in h_mean:
            mean_values = h_mean[col].dropna()
            for mean_value in mean_values:
                plot_mean_line(mean_value, col)
    elif isinstance(h_mean, pd.Series):
        for ii, mean_value in enumerate (h_mean.dropna()):
            plot_mean_line(mean_value, h_mean.index[ii])
    else:  # it's a scalar
        plot_mean_line(h_mean, 'Harmonic Mean')
    
    plt.show()

@make_data_dynamic(
    capture_columns=True, 
    reset_index=True, 
    dynamize=False, 
    )   
def get_range(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: Optional[int]=None, 
    as_frame: bool = False,
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
    ):
    """
    Calculates the range of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. 
    
    This function computes the difference between the maximum and minimum 
    values in the dataset, optionally focusing on specified columns when 
    dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the range. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    axis : optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    view : bool, default=False
        If True, visualizes the statistical analysis results or the data 
        distribution.
    cmap : str, default='viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    as_frame : bool, default=False
        If True, the result is returned as a pandas DataFrame or Series, depending
        on the dimensionality of the output. Otherwise, the result is returned in
        its native format (e.g., float, np.ndarray).
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the visualization. Only applicable if `view` 
        is True.
    **kws : dict
        Additional keyword arguments for data preprocessing, not directly 
        used in range calculation but may be used for data sanitization.

    Returns
    -------
    range_value : float or pd.Series
        The calculated range of the provided data. Returns a single float if 
        the data is one-dimensional or a pandas Series if the input is a DataFrame 
        and `as_frame` is True, showing the range for each specified column.

    Examples
    --------
    >>> from gofast.stats.descriptive import get_range
    >>> data_array = [1, 2, 3, 4, 5]
    >>> get_range(data_array)
    4

    >>> data_df = pd.DataFrame({'A': [2, 5, 8], 'B': [1, 4, 7]})
    >>> get_range(data_df, columns=['A'])
    6

    >>> get_range(data_df, as_frame=True)
    A    6
    B    6
    dtype: int64
    
    >>> data_array = [1, 2, 3, 4, 5]
    >>> print(get_range(data_array, view=True))

    >>> data_df = pd.DataFrame({'A': [2, 5, 8], 'B': [1, 4, 7]})
    >>> print(get_range(data_df, as_frame=True, view=True, fig_size=(8, 4)))

    Note
    ----
    The function aims to provide a straightforward way to compute the range 
    of a dataset, enhancing data analysis tasks. For DataFrame inputs, the 
    `as_frame` parameter allows users to maintain DataFrame structure in the 
    output, facilitating further analysis or integration with pandas-based 
    workflows.
    """

    data_selected = data.copy() if isinstance(data, pd.DataFrame) else data 
    
    # Compute the range for DataFrame or ArrayLike
    if isinstance(data_selected, pd.DataFrame):
        axis = axis or 0 
        range_values = data_selected.max(axis=axis, **kws) - data_selected.min(
            axis=axis, **kws)
    else:
        range_values = np.max(data_selected, axis=axis,  **kws) - np.min(
            data_selected, axis=axis, **kws)

    range_values, view  = ensure_visualization_compatibility(
        range_values, as_frame, view, get_range )
    
    # Visualization
    if view:
        _visualize_range(data_selected, range_values, columns=columns,
                         cmap=cmap, fig_size=fig_size, axis= axis )
        
    # For DataFrame output
    if as_frame and not isinstance (range_values, pd.DataFrame): 
        if len(range_values) ==1: 
            range_values = to_series_if(
                range_values, 
                value_names=update_series_index (
                    pd.Series (range_values) ,
                    new_indexes =["Range Values"], 
                    allow_replace=True,  
                    condition=lambda s: pd.api.types.is_integer_dtype(s.index.dtype))
                )
        else: 
            range_values = pd.Series(
                range_values, index=columns)
            
    return range_values

def _visualize_range(data, range_values, columns=None, 
                     cmap='viridis', fig_size=None, axis =0):
    """
    Visualizes the data distribution and highlights the range values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols = prepare_stats_plot(range_values, data, axis=axis )
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate(cols):
            values = data[col]
            plt.hist(values, bins=30, alpha=alphas[ii], 
                     label=f'{col} Distribution',
                     color=colors[ii])
            min_val, max_val = values.min(), values.max()
            plt.axvline(x=min_val, color='red', linestyle='dashed', 
                        linewidth=1, label=f'{col} Min')
            plt.axvline(x=max_val, color='green', linestyle='dashed',
                        linewidth=1, label=f'{col} Max')
    else:
        plt.hist(data, bins=30, alpha=0.5, label='Data Distribution',)
        min_val, max_val = np.min(data), np.max(data)
        plt.axvline(x=min_val, color='red', linestyle='dashed', 
                    linewidth=1, label='Min')
        plt.axvline(x=max_val, color='green', linestyle='dashed',
                    linewidth=1, label='Max')
    plt.title('Data Distribution and Range')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

@make_data_dynamic(capture_columns=True)
def describe(
    data: DataFrame,
    columns: Optional[List[str]] = None,
    include: Union[str, List[str]] = 'all',
    exclude: Union[str, List[str]] = None,
    as_frame: bool = True,
    view: bool = False, 
    orient: Optional[str]=None,
    plot_type: Optional[str] = 'box',   
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
):
    """
    Generates a comprehensive summary of descriptive statistics for a 
    given dataset.
    
    This function provides an easy way to get an overview of basic descriptive 
    statistics including mean, median, mode, min, max, etc., for numerical data 
    in a DataFrame or an array-like structure. It's particularly useful for 
    preliminary data analysis and understanding the distribution of data points.
    
    Parameters
    ----------
    data : ArrayLike or pd.DataFrame
        The input data for which descriptive statistics are to be calculated.
        Can be a Pandas DataFrame or any array-like structure containing 
        numerical data.
    columns : List[str], optional
        Specific columns to include in the analysis if `data` is a DataFrame.
        If None, all columns are included. Default is None.
    include : 'all', list-like of dtypes or None (default='all'), optional
        A white list of data types to include in the result. 
        Ignored for ArrayLike input.
        - 'all' : All columns of the input will be included in the output.
        - A list-like of dtypes : Limits the data to the provided data types.
          For example, [np.number, 'datetime'] will include only numeric and 
          datetime data.
    exclude : list-like of dtypes or None (default=None), optional
        A black list of data types to exclude from the result. 
        Ignored for ArrayLike input.
    as_frame : bool, default=True
        If True, the input array-like structure is converted to a Pandas DataFrame
        before calculating the descriptive statistics. This is useful if the input
        is a structured array or a sequence of sequences and you want the output
        in the form of a DataFrame.
    view : bool, optional
        If True, generates a visualization of the descriptive statistics 
        based on `plot_type`.  Default is False.
    orient: str, {None,'h'}
       Orientation of the boxplot. Use 'h' for horizontal orientation otherwise 
       the default is vertical (``None``). 
       
    plot_type : Optional[str], optional
        Specifies the type of plot for visualization: 'box', 'hist', or 
        'density'. Ignored if `view` is False. Default is 'box'.
    cmap : str, optional
        The colormap used for plotting. Default is 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
        The size of the figure for the plot. Default is (10, 6).
    **kwargs : dict
        Additional keyword arguments to be passed to the
        `pd.DataFrame.describe` method.
    
    Returns
    -------
    pd.DataFrame or pd.Series
        The descriptive statistics of the data. Returns a DataFrame if `as_frame` is True
        or if `data` is a DataFrame; otherwise, returns a Series.
    
    Examples
    --------
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.stats.descriptive import describe
    >>> data = np.random.rand(100, 4)
    >>> describe(data, as_frame=True)
    
    >>> import pandas as pd
    >>> df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    >>> describe(df, columns=['A', 'B'])
    
    >>> df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    >>> describe(df, columns=['A', 'B'], view=True, plot_type='hist')
    Note
    ----
    This function is a convenient wrapper around `pd.DataFrame.describe`,
    allowing for additional flexibility and ease of use. It's designed to
    work seamlessly with both array-like structures and Pandas DataFrames,
    making it a versatile tool for initial data exploration.
    """
    # Convert array-like input to DataFrame if necessary
    stats_result = data.describe(
        include=include, exclude=exclude, **kwargs)
    
    # Visualization
    if view:
        plot_type= validate_stats_plot_type(
            plot_type, target_strs= ['hist', 'box', 'density'],
            raise_exception= True)
        colors, alphas = get_colors_and_alphas(
            data.columns, cmap, convert_to_named_color= True )
        if plot_type == 'box':
            sns.boxplot(data=data, orient=orient, palette=cmap)
            plt.title('Box Plot of Descriptive Statistics')
        elif plot_type == 'hist':
            data.plot(kind='hist', bins=30, alpha=0.6, colormap=cmap)
            plt.title('Histogram of Descriptive Statistics')
        elif plot_type == 'density':
            data.plot(kind='density', colormap=cmap)
            plt.title('Density Plot of Descriptive Statistics')
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}."
                "Choose from 'box', 'hist', or 'density'.")
        plt.show()
        
    stats_result= convert_and_format_data(
        stats_result, as_frame, 
        force_array_output= True if not as_frame else False, 
        condense=True,
        condition=series_naming ("descriptive_stats"), 
        )
    return stats_result

@make_data_dynamic(capture_columns=True)
def z_scores(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
    as_frame: bool = True,
    view: bool = False, 
    plot_type: Optional[str] = 'hist', 
    cmap: str = 'viridis', 
    orient: Optional [str]=None, 
    fig_size: Optional[Tuple[int, int]] = (8, 6), 
    **kws):
    """
    Computes the Z-scores for each data point in a dataset. 
    
    Optionally visualizes the distribution.
    
    Z-scores, or standard scores, indicate how many standard deviations an 
    element is from the mean. This function can standardize a single array of
    values or specific columns within a pandas DataFrame.
    
    Parameters
    ----------
    data : Union[ArrayLike, pd.DataFrame]
        The dataset for which to calculate Z-scores. Can be a list, 
        numpy array, or pandas DataFrame.
        
    columns : Optional[List[str]] (default=None)
        Specifies the columns within a DataFrame for which to calculate Z-scores. 
        Applicable only when `data` is a DataFrame. If None, Z-scores for 
        all columns are calculated.
        
    as_frame : bool (default=True)
        When True, returns the result as a pandas DataFrame or Series 
        (depending on the input). Useful for maintaining DataFrame structure
        in the output.
        
    view : bool (default=False)
        If True, generates a visualization of the Z-scores distribution. 
        The type of visualization is determined by the `plot_type` parameter.
        
    plot_type : Optional[str] (default='hist')
        Specifies the type of plot for visualization:
        - 'hist': Histogram of the Z-scores distribution.
        - 'boxplot': Box plot showing the quartiles and outliers.
        - 'density': Density plot for a smooth distribution curve.
    
    cmap : str (default='viridis')
        The colormap for the visualization plot. Applicable only if `view` is True.
        
    orient: str, {None,'h'}
       Orientation of the boxplot. Use 'h' for horizontal orientation otherwise 
       the default is vertical (``None``). 
  
    fig_size : Optional[Tuple[int, int]] (default=(8, 6))
        Size of the figure for the visualization plot. Applicable only if `view` is True.
        
    **kws : dict
        Additional keyword arguments for customization, not directly used
        in Z-score calculation.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        The Z-scores of the provided data. Returns a numpy array if the data 
        is one-dimensional or if `as_frame` is False. Returns a pandas DataFrame
        if the input is a DataFrame and `as_frame` is True.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.descriptive import z_scores
    
    Calculating Z-scores for an array:
    >>> data_array = [1, 2, 3, 4, 5]
    >>> print(z_scores(data_array))
    [-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356]
    
    Calculating and visualizing Z-scores for DataFrame columns:
    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> z_scores(data_df, as_frame=True, view=True, plot_type='boxt')
    
    >>> data_array = [1, 2, 3, 4, 5]
    >>> z_scores(data_array)
    [-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356]

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> z_scores(data_df, as_frame=True)
             A         B
    0 -1.224745 -1.224745
    1  0.000000  0.000000
    2  1.224745  1.224745
    
    Note
    ----
    The Z-score is a powerful tool for data standardization, making it easier
    to compare measurements across different scales. Visualization options 
    provide quick insights into the distribution and variance of standardized data.
    """

    mean = data.mean()
    std_dev = data.std()
    result = (data - mean) / std_dev 

    # Visualization
    if view:
        plot_type= validate_stats_plot_type(
            plot_type, target_strs= ['hist', 'box', 'density'],
            raise_exception= True)
        colors,alphas = get_colors_and_alphas(result.columns)

        if plot_type == 'hist':
            if isinstance(result, pd.DataFrame):
                result.plot(kind='hist', alpha=0.5, bins=20, colormap=cmap,
                            legend=True)
            else:
                plt.hist(result, bins=20, alpha=0.5, color=cmap)
            plt.title('Z-Scores Distribution')
            plt.xlabel('Z-Score')
            plt.ylabel('Frequency')
        elif plot_type == 'box':
            if isinstance(result, pd.DataFrame):
                sns.boxplot(data=result, orient=orient, palette=cmap)
            else:
                sns.boxplot(
                    data=pd.DataFrame(result), orient=orient, color=cmap)
            plt.title('Z-Scores Box Plot')
        elif plot_type == 'density':
            if isinstance(result, pd.DataFrame):
                result.plot(kind='density', colormap=cmap, legend=True)
            else:
                sns.kdeplot(result, color=cmap)
            plt.title('Z-Scores Density Plot')
            plt.xlabel('Z-Score')

        plt.show()
    result = convert_and_format_data(
        result, as_frame, 
        force_array_output= True if not as_frame else False, 
        condense= True, 
        condition=series_naming("z_scores")
        )
    
    return result












