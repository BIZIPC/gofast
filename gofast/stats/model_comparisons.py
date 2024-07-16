# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides tools for statistical comparison and visualization of model 
performance, including tests like the Friedman and Wilcoxon tests, and 
functions for plotting comparison diagrams."""

import warnings
import itertools
import numpy as np 
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon, chi2
from scipy.stats import rankdata
from scipy.stats import t as statst, norm as statsnorm 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from ..api.types import Callable, LambdaType, DataFrame, Series, Array1D
from ..api.types import Dict, List, Union, Optional, Any, Tuple 
from ..api.extension import isinstance_  
from ..api.formatter import DataFrameFormatter, MultiFrameFormatter
from ..api.formatter import formatter_validator   
from ..decorators import isdf 
from ..tools.coreutils import validate_ratio 
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import _is_arraylike_1d, validate_comparison_data
from ..tools.validator import parameter_validator 

__all__=[
    "perform_friedman_test", 
    "perform_nemenyi_posthoc_test", 
    "generate_model_pairs", 
    "plot_nemenyi_cd_diagram", 
    "perform_wilcoxon_test", 
    "perform_friedman_test2", 
    "perform_nemenyi_posthoc_test2", 
    "compute_model_ranks", 
    "perform_wilcoxon_test2", 
    "perform_nemenyi_test2", 
    "plot_cd_diagram", 
    "plot_model_rankings", 
    "perform_posthoc_test", 
    "perform_posthoc_test2", 
    "visualize_model_performance", 
    "visualize_wilcoxon_test", 
    "compute_model_summary", 
    "perform_posthoc_analysis", 
    "calculate_pairwise_mean_differences", 
    "compute_stats_comparisons"
    ]

@isdf
def perform_friedman_test(
    model_performance_data: DataFrame, 
    alpha: float = 0.05, 
    score_preference: str = 'higher is better', 
    perform_posthoc: bool = False,
    posthoc_test: Optional[Union[Callable, LambdaType]] = None
):
    """
    Performs the Friedman test on the performance data of multiple classification
    models across multiple datasets, checking for significant differences in their
    performance.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame where each column represents a different model and each row
        represents the performance metric (e.g., accuracy) of the model on a 
        different dataset.
    alpha : float, optional
        The significance level for the hypothesis test. A typical value is 0.05.
    score_preference : str, optional
        Specifies whether higher values indicate better performance ('higher is 
        better') or lower values do ('lower is better'). Can accept snake_case and 
        is case-insensitive.
    perform_posthoc : bool, optional
        If True, executes a posthoc test provided by `posthoc_procedure` when the 
        Friedman test finds a significant difference. Defaults to False.
    posthoc_test : callable, optional
        A function that performs posthoc testing on the `performance_data`. This 
        function should accept a pandas DataFrame and return a pandas DataFrame, 
        or a formatted result in `DataFrameFormatter` object. 
        Required if `perform_posthoc` is True.

    Returns
    -------
    formatted_result : str
        A formatted string representation of the Friedman test results and, if 
        applicable, the posthoc test results.

    Notes
    -----
    The Friedman test is a non-parametric statistical test used to detect differences
    in treatments across multiple test attempts. It ranks each model's performance for
    each dataset, calculating a Friedman statistic that approximates a Chi-square 
    distribution.

    The null hypothesis (H0) of the Friedman test states that all models perform 
    equally well across all datasets. A significant result (p-value < alpha) rejects
    this hypothesis, suggesting at least one model's performance significantly 
    differs.

    The Friedman statistic is computed as:

    .. math::

        Q = \left( \frac{12}{N \cdot k \cdot (k + 1)} \right) \sum_{j=1}^{k} R_{j}^2 - 3N(k+1)


    Where:
    - :math:`N` is the number of datasets,
    - :math:`k` is the number of models,
    - :math:`R_{j}` is the sum of ranks for the :math:`j`-th model.

    The test is useful in scenarios where multiple models' performances are 
    evaluated across multiple datasets. It is robust against non-normal 
    distributions and unequal variances among groups.
    
    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_friedman_test
    >>> performance_data = pd.DataFrame({
    ...     'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...     'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...     'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> result = perform_friedman_test(performance_data)
    >>> print(result)
    
    If the result indicates a significant difference and post-hoc 
    testing is desired:

    >>> from gofast.stats.model_comparisons import perform_nemenyi_posthoc_test
    >>> result_post_hoc = perform_friedman_test(
    ...        performance_data, perform_posthoc=True, 
    ...        posthoc_test=perform_nemenyi_posthoc_test)
    >>> print(result_post_hoc)
    
    This will output a summary of the Friedman test results, indicating the Friedman
    test statistic, the degrees of freedom, the p-value, and whether there is a 
    significant difference at the specified alpha level. If `execute_posthoc` is 
    set to True and the test is significant, posthoc results will also be displayed.

    Interpretation
    ---------------
    A significant Friedman test suggests that not all models perform equally across
    the datasets. This may prompt further analysis with posthoc tests to identify
    specific differences between models.
    """
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("model_performance_data must be a pandas DataFrame.")

    if model_performance_data.empty:
        raise ValueError("model_performance_data DataFrame cannot be empty.")

    # Checking for any NaN values in DataFrame
    if model_performance_data.isnull().values.any():
        raise ValueError("model_performance_data DataFrame contains"
                         " NaN values. Please clean your data.")
        
    alpha = validate_ratio(alpha , bounds=(0, 1), exclude =0 )
    # Validate and normalize score preference
    score_preference = normalize_preference(score_preference)
    # Adjust ranks based on scoring preference
    if score_preference == 'higher is better':
        ranks = np.array([rankdata(-1 * row) for row in model_performance_data.values]) 
    else:  # 'lower is better'
        ranks = np.array([rankdata(row) for row in model_performance_data.values])
    
    n_datasets, n_models = ranks.shape
    rank_sums = np.sum(ranks, axis=0)
    # Corrected statistic calculation
    statistic = ((12 / (n_datasets * n_models * (n_models + 1))) * np.sum(
        rank_sums**2)) - (3 * n_datasets * (n_models + 1))
    
    dof = n_models - 1
    p_value = 1 - chi2.cdf(statistic, dof)

    significant_difference = p_value < alpha
    result = {
        "Statistic": statistic,
        "Degrees of Freedom": dof,
        "p-value": p_value,
        "Significant Difference": significant_difference
    }
    # Formatting the result DataFrame for presentation
    results, titles  =[ pd.DataFrame([result])], ["Friedman Test Results"]
    # Optionally perform post-hoc analysis if significant differences are detected
    if perform_posthoc and significant_difference and posthoc_test is not None:
        posthoc_result = posthoc_test(model_performance_data)
        if isinstance(posthoc_result, pd.DataFrame): 
            results.append (posthoc_result)
            titles.append ('Posthoc Results')
        elif isinstance_(posthoc_result, DataFrameFormatter): 
            results.append (posthoc_result.df)
            titles.append ('Posthoc Results')
        elif isinstance_(posthoc_result, MultiFrameFormatter) :
            results.extend (posthoc_result.dfs)
            titles.extend(posthoc_result.titles )
        if not isinstance_(
                posthoc_result, ( pd.DataFrame, DataFrameFormatter,
                                 MultiFrameFormatter ) ):
            warnings.warn(
                "Posthoc test should return a pandas DataFrame or"
                " `DataFrameFormatter` for effective pandas dataframe."
                )

    formatted_result= MultiFrameFormatter( titles=titles,
       keywords=['friedman_result', 'post_hoc_result'], 
       descriptor="FriedmanTest").add_dfs(*results)
    
    return formatted_result
 
def normalize_preference(pref: str) -> str:
    """
    Normalize the scoring preference to a standard format regardless of case
    or snake_case input.

    Parameters
    ----------
    pref : str
        The scoring preference input by the user, potentially in snake_case
        and case-insensitive.

    Returns
    -------
    str
        A normalized string in the format "lower is better" or "higher is better".
    """
    pref = pref.replace('_', ' ').lower()  # Convert to space-separated, lowercase
    if 'lower' in pref:
        return 'lower is better'
    elif 'higher' in pref:
        return 'higher is better'
    else:
        raise ValueError(
            "Invalid score_preference. Choose 'higher is better' or 'lower is better'.")

@isdf 
@ensure_pkg("scikit_posthocs", dist_name ='scikit-posthocs', infer_dist_name= True, 
            extra="Nemenyi expects 'scikit-posthocs' to be installed.")
def perform_nemenyi_posthoc_test(
    model_performance_data: DataFrame, 
    significance_level:float=0.05, 
    rank_method:str='average'
    ):
    """
    Conducts the Nemenyi post-hoc test for pairwise comparisons of multiple
    classification models across various datasets. This test is applied after
    the Friedman test indicates significant differences in model performances.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame where rows represent datasets and columns represent
        classification models. The cell values are performance metrics
        (e.g., accuracy).
    significance_level : float, optional
        The significance level for detecting differences, by default 0.05.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to rank the models in case of ties, by default 'average'.

    Returns
    -------
    dict
        A dictionary containing three key-value pairs:
        'p_values' (pd.DataFrame): Pairwise p-values from the Nemenyi test.
        'significant_differences' (pd.DataFrame): A boolean DataFrame indicating
        significant differences between model pairs.
        'average_ranks' (pd.Series): The average ranks of the models.

    Notes
    -----
    The Nemenyi test is a non-parametric pairwise comparison test that is used
    when the Friedman test has rejected the null hypothesis of equal
    performances. It compares all pairs of models to find which pairs have
    significantly different performances. The test statistic for comparing two
    models i and j is given by:

    .. math::
        q = \frac{|R_i - R_j|}{\sqrt{\frac{k(k+1)}{6N}}}

    where :math:`R_i` and :math:`R_j` are the average ranks of models i and j,
    k is the number of models, and N is the number of datasets. The critical
    value for q can be found in the studentized range distribution table for
    infinite degrees of freedom.

    The Nemenyi test is useful when you want to understand which specific models
    perform differently from each other, providing insights beyond the overall
    comparison of the Friedman test.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.stats.model_comparisons import perform_nemenyi_posthoc_test
    >>> model_performance = pd.DataFrame({
    ...     'Model A': [0.85, 0.80, 0.82],
    ...     'Model B': [0.76, 0.78, 0.75],
    ...     'Model C': [0.89, 0.85, 0.86]
    ... })
    >>> nemeyi_results = perform_nemenyi_posthoc_test(model_performance)
    >>> print(nemeyi_results) 
    >>> nemeyi_results.significant_differences
    >>> nemeyi_results.average_ranks
    """

    import scikit_posthocs as sp
    import pandas as pd
    
    significance_level= validate_ratio(
        significance_level, bounds=(0, 1), exclude=0 )
    # Validate input
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Calculate ranks if not already provided
    ranks = model_performance_data.rank(axis=1, method=rank_method)
    avg_ranks = ranks.mean()

    # Perform the Nemenyi post-hoc test
    pairwise_p_values = sp.posthoc_nemenyi_friedman(ranks)

    # Identify significant differences
    significant_diffs = pairwise_p_values < significance_level

    # Prepare output
    results = {
        'p_values': pairwise_p_values,
        'significant_differences': significant_diffs,
        'average_ranks': avg_ranks
    }
    titles = [
        "P-values", f"Significance (|P| <{significance_level:.2f})", 
        "Model Ranks"
        ] 
    keywords = ['p_values', 'significant_differences', 'average_ranks']
    results= MultiFrameFormatter(titles = titles, keywords= keywords, 
                                 descriptor="NemenyiPosthocTest").add_dfs(
        pairwise_p_values, significant_diffs,  avg_ranks
    )
    
    return results

def plot_nemenyi_cd_diagram(
    model_performance_data: Optional[DataFrame] = None, 
    model_ranks: Optional[Union[Series, Array1D]] = None, 
    score_preference: str = 'higher is better', 
    cd: float = 0.5, 
    alpha: float = 0.05,  
    ax: Optional[plt.Axes] = None, 
    figsize: Union[tuple, list] = None,
) -> None:
    """
    Produces a Critical Difference (CD) diagram to visualize the Nemenyi
    post-hoc test's pairwise comparison results. This graphical representation 
    assists in determining if the performance differences between pairs of 
    machine learning models are statistically significant.

    Parameters
    ----------
    model_performance_data : Optional[pd.DataFrame], default=None
        A DataFrame containing the performance metrics of multiple models across
        different datasets. Each row corresponds to a dataset, and each column
        corresponds to a model's metric.
        
    model_ranks : Optional[Union[pd.Series, np.ndarray]], default=None
        Either a pandas Series or a numpy array with precomputed average ranks
        of models. If not provided and `model_performance_data` is given, the
        ranks will be computed from it. The index or array should contain model 
        names.
        
    score_preference : str, default='higher is better'
        A string indicating whether higher numerical values indicate better 
        performance ('higher is better') or lower values do ('lower is better').
        This will affect the direction of the ranks computation.
        
    cd : float, default=0.5
        The critical difference value, which determines the threshold for 
        statistical significance between model ranks. Models with a rank difference
        less than this value are not considered significantly different.
        
    alpha : float, default=0.05
        The significance level used in the Nemenyi test. It determines the 
        confidence interval for the CD calculation.
        
    ax : Optional[plt.Axes], default=None
        A Matplotlib Axes object to plot the diagram on. If None, a new figure 
        and axes will be created.
        
    figsize : Union[tuple, list], default=(10, 5)
        The size of the figure, specified as a tuple or list like (width, height).
        This parameter is ignored if an `ax` object is provided.
        
    Raises
    ------
    ValueError
        If both 'model_performance_data' and 'model_ranks' are None or if 'cd'
        is not positive.
    TypeError
        If 'model_performance_data' is not a pandas DataFrame or 'model_ranks' 
        is neither a Series nor an ndarray.
        
    Examples
    --------
    >>> from gofast.stats.model_comparisons import plot_nemenyi_cd_diagram
    >>> model_performance_data = pd.DataFrame({
    ...     'Model A': [0.9, 0.85, 0.8],
    ...     'Model B': [0.84, 0.82, 0.83],
    ...     'Model C': [0.78, 0.79, 0.81],
    ... })
    >>> plot_nemenyi_cd_diagram(model_performance_data=model_performance_data, 
    ...                         score_preference='higher is better', cd=0.5)

    This will plot a CD diagram showing that if the average rank difference between
    any two models is less than 0.5, their performance is not significantly 
    different.

    Notes
    -----
    - The Critical Difference (CD) is derived from the Nemenyi post-hoc test and 
      represents the minimum required difference between model ranks for the
      performance difference to be considered significant.
    - The function will plot models on the y-axis and their average ranks on 
      the x-axis. Models within the CD of each other indicate no significant 
      difference in performance.
    - If model ranks are not provided, they will be computed as the average
      rank position across the datasets for each model based on the provided 
      performance data.
    - The figure will be displayed only if no `ax` parameter is given. This allows
      for integration of the plot within larger figure layouts or subplots.
    """
    score_preference = normalize_preference(score_preference)
    
    if model_performance_data is not None:
        if not isinstance(model_performance_data, pd.DataFrame):
            raise TypeError("model_performance_data must be a pandas DataFrame.")
        ascending = score_preference == 'lower is better'
        model_ranks = model_performance_data.rank(ascending=ascending).mean(axis=0)
    elif model_ranks is not None:
        if isinstance(model_ranks, np.ndarray):
            model_ranks = pd.Series(model_ranks)
        if not isinstance(model_ranks, pd.Series):
            raise TypeError("model_ranks must be a pandas Series or a numpy ndarray.")
    else:
        raise ValueError("Either model_performance_data or model_ranks must be provided.")
    
    if not cd > 0:
        raise ValueError("Critical difference (cd) must be a positive value.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 5) )
    sorted_ranks = model_ranks.sort_values(
        ascending=(score_preference == 'lower is better'))
    y_positions = np.arange(len(sorted_ranks)) + 1

    ax.scatter(sorted_ranks, y_positions, color='blue')
    for y, (model, rank) in zip(y_positions, sorted_ranks.items()):
        ax.plot([rank - cd, rank + cd], [y, y], color='red', linestyle='-', 
                marker='|', markersize=10)
        ax.text(rank, y, f'{model} ({rank:.2f})', ha='center', va='bottom')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_ranks.index)
    ax.invert_yaxis()  # Invert y-axis so the best (smallest rank) is on top
    ax.set_xlabel('Average Rank')
    ax.set_title(f'Nemenyi Post-Hoc Test (alpha={alpha})')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    if ax is None:  # Only show the plot if we created the figure within this function
        plt.show()

@isdf 
def perform_wilcoxon_test(
    model_performance_data: DataFrame, 
    alpha: float=0.05, 
    mask_non_significant: bool=False
    ):
    """
    Applies the Wilcoxon signed-rank test to assess the statistical significance
    of the difference between the performance metrics of pairs of models across
    multiple datasets.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A pandas DataFrame where each row represents a dataset and
        each column a different model. The cell values are the performance
        metrics (e.g., accuracy, F1 score) of the models on the datasets.
        This data is used to perform pairwise comparisons between models
        to determine if there are statistically significant differences
        in their performances.
        
    alpha : float, optional
        The significance level used to determine the threshold for
        statistically significant differences between model performances.
        Defaults to 0.05. A lower alpha value requires stronger evidence
        of a difference between model performances to reject the null
        hypothesis (that there is no difference).
        
    mask_non_significant : bool, optional
        If set to True, p-values greater than the significance level (alpha)
        are masked or replaced with NaN, highlighting only the statistically
        significant comparisons. Defaults to False, which displays all
        p-values, allowing users to see the full results of the comparisons.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each cell in row i and column j contains the p-value
        resulting from the Wilcoxon signed-rank test comparing the performance of
        model i and model j across all datasets. A p-value less than `alpha`
        indicates a statistically significant difference in performance between
        the two models.

    Notes
    -----
    The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test
    used when comparing two related samples or repeated measurements on a single
    sample to assess whether their population mean ranks differ. It is a paired
    difference test that is used as an alternative to the paired Student's t-test
    when the population cannot be assumed to be normally distributed.

    The test involves ranking each pair of observations, ignoring the signs and
    zeroes, and then summing the ranks for the observations with positive differences
    and those with negative differences. The test statistic is the smaller of these
    two sums. The null hypothesis states that the median difference between the pairs
    of observations is zero.

    Mathematically, the test statistic \(W\) can be calculated as follows:

    .. math::
        W = \min(W^+, W^-)

    where \(W^+\) is the sum of the ranks for the observations where the second
    measurement is greater than the first, and \(W^-\) is the sum of the ranks
    for the observations where the first measurement is greater than the second.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_wilcoxon_test
    >>> model_performance = pd.DataFrame({
            'Model_A': [0.90, 0.85, 0.88],
            'Model_B': [0.92, 0.83, 0.89],
            'Model_C': [0.91, 0.84, 0.87]})
    >>> result = perform_wilcoxon_test(model_performance)
    >>> print(result)
            Model_A  Model_B  Model_C
    Model_A      NaN    0.317    0.180
    Model_B    0.317      NaN    0.317
    Model_C    0.180    0.317      NaN

    In this example, we compare the performance of three models (Model_A, Model_B,
    and Model_C) across three datasets. The resulting DataFrame contains the p-values
    for each pair of models. A p-value less than 0.05 would indicate a statistically
    significant difference in performance between the two models.
    """
    n_models = model_performance_data.shape[1]
    p_values_matrix = np.full((n_models, n_models), np.nan)
    alpha = validate_ratio( alpha, bounds=(0, 1), exclude=0 )
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Extract the performance scores for the two models
            model_i_scores = model_performance_data.iloc[:, i]
            model_j_scores = model_performance_data.iloc[:, j]
            
            # Perform the Wilcoxon signed-rank test
            stat, p_value = wilcoxon(model_i_scores, model_j_scores)
            
            # Populate the symmetric matrix with p-values
            p_values_matrix[i, j] = p_value
            p_values_matrix[j, i] = p_value
    
    # Create a DataFrame from the matrix and set the model names as index/column labels
    p_values_df = pd.DataFrame(p_values_matrix, index=model_performance_data.columns,
                               columns=model_performance_data.columns)
    
    if mask_non_significant:
        # Apply significance threshold to mask non-significant comparisons
        significant_diffs = p_values_df < alpha
        p_values_df[~significant_diffs] = np.nan
        
    p_values_df= DataFrameFormatter(
        "Wilcoxon Test Results", descriptor="WilcoxonTest").add_df (p_values_df)
    return p_values_df

@isdf 
def perform_friedman_test2(
        model_performance_data: Union[Dict[str, List[float]], DataFrame], 
        alpha: float = 0.05
    ):
    """
    Performs the Friedman test on the performance data of multiple classification 
    models.
    
    This non-parametric statistical test compares the performance of three or more
    models across multiple datasets to determine if their performance is significantly
    different. It is akin to a repeated measures ANOVA but doesn't assume a normal 
    distribution of the residuals.

    The test ranks each model's performance for each dataset, calculating a 
    Friedman statistic that approximates a Chi-square distribution. A low p-value 
    indicates that at least one model's performance is significantly different.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics (e.g., accuracy) of the models.
        Each row represents a different dataset, and each column represents a different model.
    alpha : float, optional
        The significance level for determining if the observed differences are
        statistically significant. Defaults to 0.05.

    Returns
    -------
    formatted_result : DataFrameFormatter
        A formatted presentation of the Friedman test results, including the
        test statistic, p-value, and an indication of whether the differences
        in model performance are statistically significant at the specified alpha level.
        
    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_friedman_test2
    >>> df = pd.DataFrame({
    ...    'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...    'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...    'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> perform_friedman_test2(df)
    This will output a DataFrameFormatter object containing the Friedman test
    statistic, p-value, and significance result. For example:
    ```
    Friedman Test Results
    ----------------------
    Friedman Test Statistic: 5.143
    p-value: 0.076
    Significant Difference: False
    ```

    Interpretation
    ---------------
    The Friedman test statistic is calculated based on the ranked performance
    of each model across datasets. A p-value below the alpha level (e.g., 0.05)
    would indicate a significant difference in performance among the models.
    In the example output above, the p-value of 0.076 suggests that, at the 5%
    significance level, there is not enough evidence to claim a significant
    difference in model performance across the datasets.
    """
    # Check if model_performance_data is indeed a DataFrame
    if not isinstance(model_performance_data, pd.DataFrame):
        model_performance_data = pd.DataFrame(model_performance_data)

    # Perform the Friedman test
    statistic, p_value = friedmanchisquare(*model_performance_data.values.T)
    
    # Determine if the result is significant
    significance = p_value < alpha
    
    # Create a DataFrame for the results
    result = pd.DataFrame({
        "Friedman Test Statistic": [statistic],
        "p-value": [p_value],
        "Significant Difference": [significance]
    })

    # Formatting the result DataFrame for presentation
    formatted_result = DataFrameFormatter(
        title="Friedman Test Results", descriptor="FriedmanTest").add_df(result)

    return formatted_result

@isdf 
@ensure_pkg(
    "scikit_posthocs", 
    dist_name ='scikit_posthocs', 
    infer_dist_name= True 
 )
def perform_nemenyi_posthoc_test2(
       model_performance_data: Union[Dict[str, List[float]], DataFrame]
    ):
    """
    Performs the Nemenyi post-hoc test on model performance data.
    
    Parameters:
    -----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics of the models.
        Rows represent different datasets and columns represent different models.
    
    Returns:
    --------
    pairwise_comparisons : pd.DataFrame
        A DataFrame containing the p-values of pairwise comparisons between models.
        
    Note 
    -----
    This function uses the posthoc_nemenyi_friedman method from scikit-posthocs
    to perform pairwise comparisons between all models. The output is a 
    DataFrame where each element represents the p-value of the comparison 
    between two models. Lower p-values (typically ≤ 0.05) suggest a significant
    difference between the performance of those two models.

    Keep in mind that the Nemenyi test may not be as powerful for detecting 
    differences as some other post-hoc tests, but it's widely used for its 
    simplicity and non-parametric nature.
    """
    import scikit_posthocs as sp
    # Ensure the input is a DataFrame
    if not isinstance(model_performance_data, pd.DataFrame):
        raise ValueError("model_performance_data must be a pandas DataFrame.")
    
    # Perform the Nemenyi post-hoc test using the scikit-posthocs library
    pairwise_comparisons = sp.posthoc_nemenyi_friedman(model_performance_data.values)
    
    # Adjust the index and columns of the result to match the input DataFrame
    pairwise_comparisons.index = model_performance_data.columns
    pairwise_comparisons.columns = model_performance_data.columns
    
    pairwise_comparisons= DataFrameFormatter(
        "PairWise Results", descriptor="NemenyiTest").add_df (
        pairwise_comparisons)
    
    return pairwise_comparisons

@isdf 
def compute_model_ranks(
        model_performance_data: Union[Dict[str, List[float]], DataFrame]
        ):
    """
    Computes the ranks of models based on their performance.

    Parameters:
    -----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics of the models.
        Rows represent different datasets and columns represent different models.

    Returns:
    --------
    ranks : pd.DataFrame
        A DataFrame containing the ranks of each model.
    """
    
    # Ensure the input is a DataFrame
    if not isinstance(model_performance_data, pd.DataFrame):
        raise ValueError("model_performance_data must be a pandas DataFrame.")
    
    # Rank the models based on their performance
    ranks = model_performance_data.rank(axis=1, ascending=False)
    ranks= DataFrameFormatter(
        "Ranks Results", descriptor="ModelRanks").add_df (ranks)
    return ranks

@isdf
@ensure_pkg("scikit_posthocs", extra= ( 
    " nemenyi_tests needs 'scikit-posthocs' package to be installed." 
    )     
  )
def perform_nemenyi_test2(
     model_performance_data: Union[Dict[str, List[float]], DataFrame], 
     ranks:Union[Series,Array1D ] 
     ):
    """
    Performs the Nemenyi post-hoc test using ranks of model performance.

    Parameters:
    -----------
    ranks : pd.DataFrame
        A DataFrame containing the ranks of each model.
    
    Returns:
    --------
    pairwise_comparisons : pd.DataFrame
        A DataFrame containing the p-values of pairwise comparisons between models.
    """
    import scikit_posthocs as sp
    # Convert ranks to a format suitable for the scikit-posthocs library
    if isinstance ( ranks,  pd.Series): 
        ranks_array = ranks.values
    
    if not _is_arraylike_1d(ranks): 
        raise TypeError (
            f"Expect arrayLike one dimension. Got {type(ranks).__name__!r} ")
    
    # Perform the Nemenyi post-hoc test
    pairwise_comparisons = sp.posthoc_nemenyi_friedman(ranks_array)
    
    # Adjust the index and columns of the result to match the input DataFrame
    pairwise_comparisons.index = ranks.columns
    pairwise_comparisons.columns = ranks.columns
    pairwise_comparisons= DataFrameFormatter(
        "Nemenyi Results", descriptor="NemenyiTest", 
        keyword='result').add_df (
        pairwise_comparisons)
    return pairwise_comparisons

@isdf 
def perform_wilcoxon_test2(
      model_performance_data: Union[Dict[str, List[float]], DataFrame]
      ):
    """
    Executes the Wilcoxon signed-rank test pairwise across multiple models
    to statistically compare their performance metrics. This non-parametric
    test assesses whether the differences in pairs of model performances are
    symmetrically distributed around zero, indicating no significant difference.

    Parameters
    ----------
    model_performance_data : Union[Dict[str, List[float]], pd.DataFrame]
        A dictionary or DataFrame with model names as keys (columns) and
        lists (rows) of performance metrics as values. Each row should
        correspond to the performance metric (e.g., accuracy) of the model
        on a different dataset.

    Returns
    -------
    results : pd.DataFrame
        A DataFrame where each row and column header corresponds to a model
        and each cell contains the p-value resulting from the Wilcoxon signed-rank
        test between the pair of models denoted by the respective row and column.

    Notes
    -----
    The Wilcoxon signed-rank test is a non-parametric alternative to the paired
    T-test when the differences between pairs do not necessarily follow a normal
    distribution. In the context of model comparison, it can be used to determine
    whether two models have different performances on a set of datasets. The test
    assumes that the differences between paired observations are independent and
    identically distributed.
    
    This function performs the test on each unique pair of models. The resulting
    p-values are stored in a square DataFrame, with models represented both in rows
    and columns. A p-value close to 0 suggests a significant difference in performance
    between the pair of models. Diagonal cells are not computed as they would compare
    a model to itself.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_wilcoxon_test2
    >>> performance_data = pd.DataFrame({
    ...     'Model_X': [0.85, 0.80, 0.78],
    ...     'Model_Y': [0.75, 0.82, 0.79],
    ...     'Model_Z': [0.80, 0.77, 0.74]
    ... })
    >>> results = perform_wilcoxon_test2(performance_data)
    >>> print(results)
    
    The resulting DataFrame will contain NaNs on the diagonal and p-values
    below the diagonal, reflecting the symmetric nature of the test.
    """
    columns = model_performance_data.columns
    n = len(columns)
    results = pd.DataFrame(index=columns, columns=columns)

    for i in range(n):
        for j in range(i + 1, n):
            model_i = model_performance_data.iloc[:, i]
            model_j = model_performance_data.iloc[:, j]
            
            # Perform the Wilcoxon signed-rank test
            stat, p = wilcoxon(model_i, model_j)
            results.iloc[i, j] = p
            results.iloc[j, i] = p
    
    # Fill diagonal with NaNs since we don't compare a model with itself
    # pd.fill_diagonal(results.values, pd.NA)
    np.fill_diagonal(results.values, pd.NA)
    results= DataFrameFormatter(
        "Wilcoxon Results", descriptor="WilcoxonTest", 
        keyword='result').add_df (results)
    return results

def plot_cd_diagram(
    model_performance_data: Optional[DataFrame] = None, 
    model_ranks: Optional[Union[Series, Array1D]] = None, 
    score_preference: str = 'higher is better', 
    cd: float = 0.5, 
    ax=None, 
    title: Optional[str] = None
):
    """
    Generates a Critical Difference (CD) diagram, a visual tool for comparing the 
    performance rankings of different machine learning models across multiple 
    datasets. The diagram delineates which models' performances are significantly 
    different from each other based on their average ranks and the CD value.

    A CD diagram is especially useful for presenting the results of statistical 
    tests like the Friedman test and subsequent post-hoc analysis, providing a 
    clear visual interpretation of these results.

    Parameters
    ----------
    model_performance_data : pd.DataFrame, optional
        The performance data of various models. Each row corresponds to a dataset 
        and each column to a model's performance metric, like accuracy or F1 score.
    model_ranks : pd.Series or np.ndarray, optional
        The average ranks of the models. If not provided, will be computed from 
        `model_performance_data`. The index should contain model names and the 
        values should represent the average ranks of the models.
    score_preference : str, optional
        Indicates whether higher values ("higher is better") or lower values 
        ("lower is better") correspond to better performance. The ranking of models 
        will be adjusted accordingly.
    cd : float, optional
        The Critical Difference (CD) for the significance level, usually derived 
        from a post-hoc test. It is the minimum difference in rank required for 
        two models to be considered significantly different.
    ax : matplotlib.axes.Axes, optional
        An existing matplotlib Axes object to plot on. If None, a new figure and 
        axes will be generated.
    title : str, optional
        The title for the CD diagram. If None, defaults to "Critical Difference Diagram".

    Raises
    ------
    ValueError
        If both `model_performance_data` and `model_ranks` are None, indicating 
        that no input data was provided.
    TypeError
        If the provided `model_performance_data` is not a DataFrame or if 
        `model_ranks` is not a Series or 1D numpy array.

    Examples
    --------
    >>> from gofast.stats.model_comparisons import plot_cd_diagram
    >>> model_performance = pd.DataFrame({
    ...     'Model_A': [0.90, 0.85, 0.88],
    ...     'Model_B': [0.92, 0.83, 0.89],
    ...     'Model_C': [0.91, 0.84, 0.87]
    ... })
    >>> plot_cd_diagram(model_performance_data=model_performance, cd=0.5)

    The above example assumes the availability of a function `plot_cd_diagram` 
    and would result in a plot showing the models on the y-axis and their average 
    ranks on the x-axis, with lines or whiskers denoting the CD interval for each 
    model. Models whose intervals overlap are not significantly different from 
    each other in terms of performance.

    Notes
    -----
    The CD value must be determined based on the results of a Friedman test and 
    post-hoc analysis. This typically involves using the average ranks of models 
    obtained from multiple datasets and calculating the CD using critical values 
    from the appropriate statistical distribution. The significance of the CD 
    value depends on the number of models, the number of datasets, and the chosen 
    alpha level for the statistical tests.
    """

    score_preference = normalize_preference(score_preference)
    
    if model_performance_data is not None:
        if not isinstance(model_performance_data, pd.DataFrame):
            raise TypeError("model_performance_data must be a pandas DataFrame.")
        ascending = score_preference == 'lower is better'
        model_ranks = model_performance_data.rank(ascending=ascending).mean(axis=0)
    
    elif model_ranks is not None:
        if isinstance(model_ranks, np.ndarray):
            model_ranks = pd.Series(model_ranks)
        if not isinstance(model_ranks, pd.Series):
            raise TypeError("model_ranks must be a pandas Series or a numpy ndarray.")
    else:
        raise ValueError("Either model_performance_data or model_ranks must be provided.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if title is None:
        title = 'Critical Difference Diagram'
    
    # Sort the model ranks for plotting
    model_ranks_sorted = model_ranks.sort_values(ascending=False)  # Best models on top
    y_positions = np.arange(len(model_ranks_sorted)) + 1

    # Extend the x-axis limits to provide more space for the CD intervals
    min_rank = model_ranks_sorted.min() - cd
    max_rank = model_ranks_sorted.max() + cd
    ax.set_xlim(min_rank - 1, max_rank + 1)
    
    # Plot the model ranks and CD interval
    ax.hlines(y_positions, model_ranks_sorted - cd / 2, model_ranks_sorted + cd / 2,
              color='black')
    # ax.plot(model_ranks_sorted, y_positions, 'o', markersize=10, label='Models')
    
    # Adding model names and ranks as text next to each marker
    for y, (model, rank) in zip(y_positions, model_ranks_sorted.items()):
        ax.plot([rank - cd / 2, rank + cd / 2], [y, y], marker='o', 
                        markersize=10, color='black',
                        label=model if y == 1 else "")  # Adjust marker size
        # Annotate the actual average rank next to each model
        # Adjust text position
        ax.text(rank, y - 0.1, f' {model} ({rank:.2f})', va='bottom', ha='center')

    # Set the plot aesthetics
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_ranks_sorted.index)
    ax.invert_yaxis()  # Best models on top
    ax.set_xlabel('Average Rank')
    ax.set_title(title)
    ax.axvline(x=model_ranks_sorted.min() - cd / 2, linestyle='--', 
               color='gray', alpha=0.5)
    ax.axvline(x=model_ranks_sorted.max() + cd / 2, linestyle='--',
               color='gray', alpha=0.5)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

@isdf 
def plot_model_rankings(
    model_performance_data: Union[Dict[str, List[float]], DataFrame],
    score_preference: str = 'higher is better',
    fig_size: Union[tuple, list, None] = None
):
    """
    Generates a bar chart visualizing the average ranking of models based on 
    performance metrics. Each model's ranking is calculated across multiple 
    datasets, and they are ordered from best (lowest average rank) to worst 
    (highest average rank) in the chart.

    Parameters
    ----------
    model_performance_data : Union[Dict[str, List[float]], pd.DataFrame]
        A dictionary with model names as keys and lists of performance 
        metrics as values, or a pandas DataFrame with models as columns 
        and performance metrics as rows. Each cell should contain a 
        performance score of the model on a specific dataset.
    score_preference : str, optional
        Indicates whether higher values ("higher is better") or lower values 
        ("lower is better") correspond to better performance. The ranking of models 
        will be adjusted accordingly.
    fig_size : Union[tuple, list, None], optional
        The figure size as a tuple or list of two values: (width, height). 
        If None, the default figure size is used. This parameter allows 
        customization of the output plot size, enhancing readability or 
        fitting specific layout requirements.


    Returns
    -------
    None
        Displays a bar chart ranking the models by their average performance 
        but does not return any value.

    Raises
    ------
    TypeError
        If `model_performance_data` is not a dictionary or a pandas DataFrame.
    ValueError
        If `model_performance_data` is empty or does not contain numerical values.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import plot_model_rankings
    >>> performance_data = {
    ...    'Model_X': [0.85, 0.80, 0.78],
    ...    'Model_Y': [0.75, 0.82, 0.79],
    ...    'Model_Z': [0.80, 0.77, 0.74]
    ... }
    >>> plot_model_rankings(performance_data)

    Or using a DataFrame:

    >>> import pandas as pd
    >>> df_performance = pd.DataFrame(performance_data)
    >>> plot_model_rankings(df_performance)

    Notes
    -----
    The function computes the average rank of each model's performance scores 
    over all datasets. Models with consistently higher performance across 
    datasets will have a lower average rank. The bar chart provides an intuitive 
    visualization of model performance, making it easier to identify the best 
    and worst-performing models. Models are displayed on the x-axis with their 
    corresponding average ranks on the y-axis, and ranks are shown on top of 
    their respective bars for clarity.
    """
    # Convert dictionary to DataFrame if necessary
    if isinstance(model_performance_data, dict):
        model_performance_data = pd.DataFrame(model_performance_data)
    
    # Validate input is a DataFrame
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("The model_performance_data must be a"
                        " pandas DataFrame or a dictionary.")

    # Check that the DataFrame is not empty
    if model_performance_data.empty:
        raise ValueError("The model_performance_data DataFrame is empty.")

    # Ensure the DataFrame contains numerical values
    if not np.issubdtype(model_performance_data.dtypes[0], np.number):
        raise ValueError(
            "The model_performance_data must contain numerical values.")

    # Normalize score preference
    score_preference = normalize_preference(score_preference)
    ascending = score_preference == 'lower is better'

    # Calculate the average rank for each model
    ranks = model_performance_data.rank(axis=0, ascending=ascending).mean()

    # Sort the models by their average rank (best to worst)
    sorted_ranks = ranks.sort_values(ascending=ascending)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=fig_size or (10, 5))
    bars = ax.bar(sorted_ranks.index, sorted_ranks.values, color='skyblue')
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Rank')
    ax.set_title('Model Rankings')
    
    # Invert y-axis if 'lower is better' to have the best (lowest rank) at the top
    if ascending:
        ax.invert_yaxis()

    # Add the rank values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2),
                ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to fit all labels
    plt.show()
    
@isdf     
@ensure_pkg("scikit_posthocs", extra = ( 
    "Post-hocs test expects 'scikit-posthocs' to be installed."), 
    dist_name="scikit-posthocs"
    ) 
@ensure_pkg("statsmodels")
def perform_posthoc_test(
    model_performance_data: DataFrame, 
    test_method: str='tukey', 
    alpha: str=0.05, 
    metric: str='accuracy', 
    score_preference:str='higher is better'
    ):
    """
    Performs post-hoc analysis on model performance data to identify specific
    differences between models after a significant Friedman test result.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame where each row represents a dataset and each column
        represents a model. Cells contain performance metrics (e.g., accuracy)
        of models on datasets.
    test_method : str, optional
        The post-hoc test method to apply. Supported methods are 'tukey' for
        Tukey's Honestly Significant Difference (HSD) test and 'nemenyi' for
        the Nemenyi test. Defaults to 'tukey'.
    alpha : float, optional
        Significance level for determining statistically significant differences,
        with a default value of 0.05.
    metric : str, optional
        The name of the performance metric used in `model_performance_data`,
        such as 'accuracy', for labeling purposes. Defaults to 'accuracy'.

    Returns
    -------
    posthoc_result : pd.DataFrame or other suitable format
        The results of the post-hoc test, including pairwise comparisons and
        significance levels. The format depends on the selected `test_method`.

    Notes
    -----
    - Tukey's HSD test is useful for comparing all possible pairs of groups
      to identify significantly different pairs. It controls the Type I error
      rate across multiple comparisons.

      .. math:: Q = \\frac{\\bar{Y}_i - \\bar{Y}_j}{SE}

    - The Nemenyi test is a non-parametric equivalent of Tukey's test that
      uses rank sums. It's suitable when the assumptions of normality and
      homoscedasticity are violated.

      .. math:: CD = q_\\alpha\\sqrt{\\frac{k(k+1)}{6N}}

    These tests are essential when the Friedman test indicates a significant
    difference, allowing researchers to pinpoint where these differences lie.
    Tukey's test is preferred when data meet parametric assumptions, while
    the Nemenyi test is more versatile for non-parametric conditions.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_posthoc_test
    >>> df = pd.DataFrame({
    ...    'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...    'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...    'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> result = perform_posthoc_test(df, test_method='tukey')
    >>> print(result)

    The choice between Tukey's and Nemenyi depends on the dataset's characteristics
    and the assumptions underlying the data.
    """
    class TurkeyTest: 
        def __init__(self, posthoc_result): 
            self.posthoc_result = posthoc_result
        def __str__(self): 
            return self.posthoc_result.__str__() 
        def __repr__(self): 
            return ("<TurkeyTest object containing data."
                    " Use print() to see contents.>")
        
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import scikit_posthocs as sp
    
    # Input validation:The decorator 'isdf' handles this part. 
    # Shoud be removed next.  
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("model_performance_data must be a pandas DataFrame.")
    
    valid_methods = {'tukey': 'pairwise_tukeyhsd', 'nemenyi': 'posthoc_nemenyi_friedman'}
    if test_method.lower() not in valid_methods:
        raise ValueError(f"Unsupported test method '{test_method}'."
                         f" Available methods are {list(valid_methods.keys())}.")
    score_preference = normalize_preference(score_preference)
    # Define a function mapper for scalability
    method_function_mapper = {
        'tukey': lambda data, groups, alpha: pairwise_tukeyhsd(
            endog=data[metric], groups=data['Model'], alpha=alpha).summary(),
        'nemenyi': lambda ranks: sp.posthoc_nemenyi_friedman(ranks)
    }
    if test_method.lower() == 'tukey':
        # Flatten the DataFrame for Tukey's HSD
        data_flat = model_performance_data.melt(var_name='Model', value_name=metric)
        
    else:
        # Prepare data for Nemenyi or other rank-based tests
        encoder = LabelEncoder()
        encoded_models = encoder.fit_transform(model_performance_data.columns)
        ranks = ( 
            np.array([rankdata(-1 * performance) 
                      for performance in model_performance_data.values])
            )
        data_flat = pd.DataFrame(ranks, columns=encoded_models)

    # Execute the selected post-hoc test
    posthoc_result = ( 
        method_function_mapper[test_method.lower()](data_flat, data_flat['Model'], alpha)
        if test_method.lower() == 'tukey' 
        else method_function_mapper[test_method.lower()](ranks)
        )
    
    if test_method.lower() == 'tukey': 
        return TurkeyTest (posthoc_result)
    
    # remake dataframe for pair index and columns 
    posthoc_result = pd.DataFrame(
        posthoc_result.values, columns=model_performance_data.columns, 
        index=model_performance_data.columns 
        ) 
    posthoc_result= DataFrameFormatter(
        f"Posthoc {test_method} Results", descriptor="NemenyiTest").add_df (
        posthoc_result)
    
    return posthoc_result

@isdf
def calculate_pairwise_mean_differences(
    performance_data: DataFrame, 
    return_group: bool=False,
    unique_combinations: bool=True):
    """
    Calculates and returns pairwise mean differences between models based
    on their performance data. Offers the option to return results either
    as a matrix or a condensed list format with unique pairings.

    Parameters
    ----------
    performance_data : pandas.DataFrame
        DataFrame where each column represents a model and rows represent
        performance observations.
    return_group : bool, optional
        If True, results are returned in a DataFrame listing each unique
        pair of models and their mean differences. Defaults to False.
    unique_combinations : bool, optional
        If True, avoids repeated pair calculations (e.g., A vs B and B vs A).
        Only unique pairs (A vs B) are considered. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of pairwise mean differences. If return_group is True,
        returns a DataFrame with columns ['group1', 'group2', 'meandiff'].
        Otherwise, returns a matrix where each entry (i, j) represents the
        mean difference between models i and j.

    Examples
    --------
    >>> from gofast.stats.model_comparisons import calculate_pairwise_mean_differences
    >>> performance_data = pd.DataFrame({
    ...     'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...     'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...     'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> calculate_pairwise_mean_differences(performance_data)
              Model_A  Model_B  Model_C
    Model_A     0.000   0.0020   0.0040
    Model_B    -0.002   0.0000   0.0020
    Model_C    -0.004  -0.0020   0.0000

    >>> calculate_pairwise_mean_differences(performance_data, return_group=True)
       group1   group2  meandiff
    0  Model_A  Model_B     0.002
    1  Model_A  Model_C     0.004
    2  Model_B  Model_C     0.002
    """
    # Calculate mean performance for each model
    mean_performances = performance_data.mean()

    # Initialize a DataFrame to store the pairwise mean differences
    mean_diffs = pd.DataFrame(index=mean_performances.index,
                              columns=mean_performances.index)

    # Define a helper function to update mean differences matrix
    def update_mean_diffs(model1, model2):
        mean_diff = mean_performances[model1] - mean_performances[model2]
        mean_diffs.at[model1, model2] = mean_diff
        mean_diffs.at[model2, model1] = -mean_diff

    # Compute pairwise mean differences
    if unique_combinations:
        # Use combinations to avoid repetitive pairings
        for model1, model2 in itertools.combinations(mean_performances.index, 2):
            update_mean_diffs(model1, model2)
    else:
        # Compute for all possible pairs including self-comparison
        for model1 in mean_performances.index:
            for model2 in mean_performances.index:
                if model1 == model2:
                    mean_diffs.at[model1, model2] = 0
                else:
                    update_mean_diffs(model1, model2)

    if return_group:
        # Prepare data in a group format
        records = []
        if unique_combinations:
            for model1, model2 in itertools.combinations(mean_performances.index, 2):
                records.append({
                    'group1': model1, 'group2': model2, 
                    'meandiff': mean_diffs.at[model1, model2]
                    })
        else:
            for model1 in mean_performances.index:
                for model2 in mean_performances.index:
                    if model1 != model2:
                        records.append({
                            'group1': model1,
                            'group2': model2, 
                            'meandiff': mean_diffs.at[model1, model2]
                        })
        return pd.DataFrame(records)

    return mean_diffs

def handle_error(message, error_mode, exception_type=Exception):
    """Handle errors according to the specified mode."""
    if error_mode == "raise":
        raise exception_type(message)
    elif error_mode == "warn":
        warnings.warn(message)

def _compute_adjusted_p_values(data, method, error_mode):
    """Attempt to compute adjusted p-values, handling errors according 
    to the specified mode."""
    try:
        return compute_stats_comparisons(data, test_type=method).result
    except Exception as e:
        temp = ( "Using a simplified estimation based on mean differences." 
                if error_mode == 'warn' else '')
        message = (f"Sample sizes are not provided while computing p-values"
                   f" using the '{method}' method. {temp} Error: {str(e)}")
        handle_error(message, error_mode, ValueError)
        return None

def _retrieve_precomputed_p_values(p_adj_result, group1, group2, error_mode):
    """Retrieve precomputed p-values, handling errors."""
    try:
        return get_p_adj_for_groups(p_adj_result, group1, group2)
    except Exception as e:
        temp = "Falling back to direct computation." if error_mode == 'warn' else ''
        message = ("Failed to retrieve precomputed p-values for groups"
                   f" '{group1}' and '{group2}'. {temp} Error: {str(e)}")
        handle_error(message, error_mode, ValueError)
        return None

@isdf
def perform_posthoc_analysis(
    performance_data:DataFrame, 
    test_method: str='tukey',
    significance_level: float=0.05, 
    ci_multiplier: float=1.96,
    sample_sizes: Series=None, 
    error="warn"
    ):
    """
    Performs posthoc analysis using specified statistical tests to compare 
    mean differences between groups based on provided performance data.

    Parameters
    ----------
    performance_data : pandas.DataFrame
        DataFrame where columns represent different models and rows 
        represent performance metrics from multiple tests or experiments.
    test_method : str, default 'tukey'
        Specifies the statistical test for comparison. Accepts 'tukey', 
        'nemenyi' or or 'wilcoxon' as valid options.
    significance_level : float, default 0.05
        The significance level used to determine the critical value for 
        hypothesis testing.
    ci_multiplier : float, default 1.96
        Multiplier used for computing the confidence interval around the mean 
        difference. Default corresponds to approximately 95% confidence.
    sample_sizes : pandas.Series or None, optional
        A series containing sample sizes for each model. This is required for 
        more accurate standard error calculation. If not provided, a simpler 
        estimation is used based on the mean differences.
    error : str, default 'warn'
        Error handling mode. Options are 'raise' to throw exceptions directly, 
        or 'warn' to issue warnings without halting the execution. 
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the posthoc analysis results with columns for 
        each pair of models compared ('group1', 'group2'), the mean difference 
        ('meandiff'), adjusted p-values ('p-adj'), confidence intervals 
        ('lower', 'upper'), and a boolean indicating whether the null 
        hypothesis is rejected ('reject').
        
    Notes
    -----
    This function interprets statistical terms in the context of posthoc testing, 
    particularly following ANOVA. Each term used in the output has specific 
    statistical implications:

    - **meandiff (Mean Difference)**:
      - **Description**: Represents the average difference between the means of 
        two groups being compared.
      - **Interpretation**: A positive value indicates the first group's mean is 
        higher than the second's. The magnitude reflects the extent of the 
        difference.

    - **p-adj (Adjusted P-value)**:
      - **Description**: Measures the probability that the observed difference 
        is due to chance, under the null hypothesis of no actual difference.
        Adjustment is made to account for the increased risk of Type I error 
        due to multiple comparisons.
      - **Interpretation**: Values below the significance level (typically 0.05) 
        suggest significant evidence against the null hypothesis, indicating a 
        likely genuine difference.

    - **lower (Lower Bound of Confidence Interval)**:
      - **Description**: The lower limit of the range within which the true 
        mean difference is estimated to lie, with a specified level of confidence
        (usually 95%).
      - **Interpretation**: Provides a minimum expected value for the mean 
        difference, contributing to the assessment of result precision and 
        reliability.

    - **upper (Upper Bound of Confidence Interval)**:
      - **Description**: The upper limit of the confidence interval.
      - **Interpretation**: Offers a maximum expected value, similar to the 
        lower bound, helping to frame the potential variability in the mean 
        difference.

    - **reject (Reject the Null Hypothesis)**:
      - **Description**: A boolean indicating whether the null hypothesis 
        (no difference) should be rejected based on the statistical test.
      - **Interpretation**: `True` implies significant statistical evidence 
        exists to suggest a difference between groups, while `False` suggests 
        insufficient evidence to reject the null hypothesis.

    These definitions help contextualize the results from pairwise comparisons, 
    enabling informed conclusions based on the provided performance data.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_posthoc_analysis 
    >>> performance_data = pd.DataFrame({
    ...     'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...     'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...     'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> sample_sizes = pd.Series({'Model_A': 100, 'Model_B': 100, 'Model_C': 100})
    >>> result = perform_posthoc_analysis(performance_data, 'tukey', 
    ...                                   sample_sizes=sample_sizes)
    >>> result
    <Tukeytest object containing table data. Use print() to view contents.>

    >>>  print(result ) 
    
           Multiple Comparison Of Means - Tukey, Fwer=0.05      
     ===========================================================
      group1   group2  meandiff   p-adj    lower   upper  reject
     -----------------------------------------------------------
     Model_A  Model_B    0.0020  0.6046  -0.0050  0.0090   False
     Model_A  Model_C    0.0060  0.1419  -0.0004  0.0124   False
     Model_B  Model_C    0.0040  0.4050  -0.0044  0.0124   False
     ===========================================================
    
     >>> result.result 
         group1   group2  meandiff     p-adj     lower     upper  reject
     0  Model_A  Model_B     0.002  0.604603 -0.004985  0.008985   False
     1  Model_A  Model_C     0.006  0.141927 -0.000441  0.012441   False
     2  Model_B  Model_C     0.004  0.405023 -0.004430  0.012430   False
     
    """
    # Validate test method
    test_method = parameter_validator(
        "test_method", target_strs= ['tukey', 'nemenyi','wilcoxon' ]
        )(test_method)

    # Calculate pairwise mean differences using the previously defined function
    mean_diffs = calculate_pairwise_mean_differences(
        performance_data, return_group=True)
    
    p_adj_result = None
    if sample_sizes is None and test_method == "nemenyi":
        p_adj_result = _compute_adjusted_p_values(
            performance_data, test_method, error)

    # Prepare to store results
    records = []
    # Compute statistics for each pair
    for index, row in mean_diffs.iterrows():
        group1, group2, meandiff = row['group1'], row['group2'], row['meandiff']
  
        if sample_sizes is not None:
            # Calculate standard error from sample sizes
            n1 = sample_sizes[group1]
            n2 = sample_sizes[group2]
            se = np.sqrt(performance_data[group1].var()/n1 + performance_data[group2].var()/n2)
        else:
            # Use a placeholder standard error if sample sizes are not provided
            se = np.abs(meandiff) / ci_multiplier
        
        # Calculate confidence intervals
        lower = meandiff - ci_multiplier * se
        upper = meandiff + ci_multiplier * se
        
        # Calculate p-value based on the test method
        if test_method == 'tukey':
            # degrees of freedom for the test
            df = len(performance_data) - 1
            p_adj = 2 * (1 - statst.cdf(np.abs(meandiff) / se, df))
    
        elif test_method == 'nemenyi':
            if not p_adj_result.empty:
                p_adj = _retrieve_precomputed_p_values(p_adj_result, group1, group2, error)
            if p_adj is None:
                z_score = np.abs(meandiff) / se
                p_adj = 2 * (1 - statsnorm.cdf(z_score))
            
        elif test_method == 'wilcoxon':
            # Wilcoxon test for paired samples (non-parametric)
            stat, p_adj = wilcoxon(performance_data[group1], performance_data[group2])
        # Determine rejection of the null hypothesis
        reject = p_adj < significance_level

        records.append({
            'group1': group1,
            'group2': group2,
            'meandiff': meandiff,
            'p-adj': p_adj,
            'lower': lower,
            'upper': upper,
            'reject': reject
        })
    
    title = f"Multiple Comparison of Means - {test_method.title()}, FWER={significance_level}"
    analysis_result= DataFrameFormatter(
        title=title, descriptor=f"{test_method}Test", 
        keyword ='result').add_df (
        pd.DataFrame(records))
    
    return analysis_result

@isdf 
def get_p_adj_for_groups(df, group1, group2):
    """
    Retrieves the p-adj value for a specified pair of groups from a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing the comparison results with columns 
        'group1', 'group2', and 'p-adj'.
    group1 : str
        The name of the first group in the comparison.
    group2 : str
        The name of the second group in the comparison.

    Returns:
    -------
    float
        The p-adj value for the specified group pair.

    Raises:
    ------
    ValueError
        If no matching pair is found for the specified groups.

    Examples:
    --------
    >>> import pandas as pd 
    >>> data = pd.DataFrame({
    ...     'group1': ['Model_A', 'Model_A', 'Model_B'],
    ...     'group2': ['Model_B', 'Model_C', 'Model_C'],
    ...     'p-adj': [0.50, 1.00, 0.75]
    ... })
    >>> print(get_p_adj_for_groups(data, 'Model_A', 'Model_B'))
    0.5

    >>> print(get_p_adj_for_groups(data, 'Model_B', 'Model_C'))
    0.75
    """
    # Attempt to find the row with the matching groups
    mask = ((df['group1'] == group1) & (df['group2'] == group2)) | (
        (df['group1'] == group2) & (df['group2'] == group1))
    results = df[mask]

    if results.empty:
        raise ValueError(f"No matching pair found for {group1} and {group2}.")
    
    return results['p-adj'].iloc[0]


@isdf
def transform_comparison_data(data, /):
    """
    Transforms a square DataFrame of pairwise comparisons into a long format DataFrame
    listing unique pairings and their associated values.

    Parameters:
    ----------
    data : pandas.DataFrame
        A square DataFrame where both rows and columns represent models and the cells
        contain the pairwise comparison metric (e.g., p-values, correlations).

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with columns 'group1', 'group2', and 'p-adj', where each row 
        represents the comparison metric between a unique pair of models.
    """
    records = []  # List to store the results

    # Iterate over the upper triangle of the DataFrame to avoid duplicates
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            records.append({
                'group1': data.index[i],
                'group2': data.columns[j],
                'p-adj': data.iloc[i, j]
            })
    
    return pd.DataFrame(records)

def compute_stats_comparisons(data_or_result, test_type='wilcoxon'):
    """
    Computes statistical comparisons using specified tests and processes 
    the results into a formatted DataFrame showing pairwise comparison results.

    Parameters:
    ----------
    data_or_result : pandas.DataFrame
        DataFrame containing performance data or already computed results.
    test_type : str, default 'wilcoxon'
        Specifies the statistical test to use for comparison. Supported tests
        are 'wilcoxon' and 'nemenyi'.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the formatted pairwise comparison results with 
        columns 'group1', 'group2', and 'p-adj'.

    Raises:
    ------
    ValueError
        If an unsupported test is specified or other input validation fails.

    Examples:
    --------
    >>> from gofast.stats.model_comparisons import compute_stats_comparisons
    >>> from gofast.stats.model_comparisons import perform_wilcoxon_test2
    >>> performance_data = pd.DataFrame({
    ...     'Model_A': [1.0, 0.9, 0.78],
    ...     'Model_B': [0.9, 1.0, 0.9],
    ...     'Model_C': [0.78, 0.9, 1.0]
    ... }, index=['Model_A', 'Model_B', 'Model_C'])
    >>> print(compute_stats_comparisons(performance_data))
       group1   group2     p-adj
    0  Model_A  Model_B  0.900000
    1  Model_A  Model_C  0.781714
    2  Model_B  Model_C  0.900000
    
    >>> wilcoxon_result = perform_wilcoxon_test2 (performance_data)
    >>> print(wilcoxon_result.result)
            Model_A Model_B Model_C
    Model_A    <NA>     0.5     1.0
    Model_B     0.5    <NA>    0.75
    Model_C     1.0    0.75    <NA>
    >>> compute_stats_comparisons(wilcoxon_result)
        group1   group2  p-adj
    0  Model_A  Model_B   0.50
    1  Model_A  Model_C   1.00
    2  Model_B  Model_C   0.75
    """

    # Function mapper for scalability
    method_function_mapper = {
        'wilcoxon': lambda data: perform_wilcoxon_test2(data_or_result),
        'nemenyi': lambda data: perform_nemenyi_posthoc_test2(data_or_result)
    }

    valid_tests = {'wilcoxon', 'nemenyi'}
    test_type = parameter_validator(
        "test_type", target_strs= valid_tests, 
        error_msg=f"Unsupported test type specified. Use {valid_tests} instead." 
    ) (test_type)

    # Perform the selected test
    if isinstance(data_or_result, pd.DataFrame):
        data_or_result = method_function_mapper[test_type](data_or_result)
    
    # Validate and transform the results
    result_df = formatter_validator(data_or_result, attributes='df')
    result_df = validate_comparison_data(result_df)
    result_df = transform_comparison_data(result_df)

    comparison_results = DataFrameFormatter( 
        title =f"{test_type} Results", keyword='result', 
        descriptor=f"{test_type}Test").add_df (result_df)
    
    return comparison_results

@isdf 
@ensure_pkg (
    "scikit_posthocs", 
    extra= "Post-hocs tests need 'scikit-posthocs' package to be installed.", 
    partial_check=True,
    condition=lambda *args, **kwargs: kwargs.get("test_method") =='nemenyi'
   )
@ensure_pkg (
    "statsmodels", 
    extra= "Post-hocs test need 'statsmodels' to be installed.", 
    partial_check=True,
    condition=lambda *args, **kwargs: kwargs.get("test_method") =='tukey'
   )
def perform_posthoc_test2(
    model_performance_data: DataFrame,
    test_method: str='tukey',
    alpha: float=0.05,
    metric: str='accuracy'
    ):
    """
    Performs post-hoc analysis on model performance data to investigate pairwise
    differences between models after a significant Friedman test result.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the model performance metrics, where each row
        represents a dataset, and each column represents a model.
    test_method : str, optional
        The post-hoc test to apply. Options include 'tukey' for Tukey's HSD test
        and 'nemenyi' for Nemenyi test. Defaults to 'tukey'.
    alpha : float, optional
        The significance level for determining statistically significant differences.
        Defaults to 0.05.
    metric : str, optional
        The performance metric name used in `model_performance_data`. Defaults to
        'accuracy', used for labeling purposes.

    Returns
    -------
    posthoc_result : dict or pd.DataFrame
        The result of the post-hoc analysis. The format and content depend on the
        selected `test_method`.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_posthoc_test
    >>> model_performance_data = pd.DataFrame({
    ...     'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...     'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...     'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> posthoc_result = perform_posthoc_test(model_performance_data, test_method='tukey')
    >>> print(posthoc_result)

    Raises
    ------
    ValueError
        If an unsupported `test_method` is specified.
    """
    test_method = str(test_method)
    if test_method.lower() == 'tukey':
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        # Flatten the DataFrame for Tukey's HSD test
        data_flat = model_performance_data.melt(var_name='Model', value_name=metric)
        posthoc_result = pairwise_tukeyhsd(
            endog=data_flat[metric],
            groups=data_flat['Model'],
            alpha=alpha
        ).summary()
    elif test_method.lower() == 'nemenyi':
        from scikit_posthocs import posthoc_nemenyi_friedman
        # Nemenyi test with ranks from Friedman test setup
        ranks = np.array([rankdata(-1 * performance) 
                          for performance in model_performance_data.values])
        posthoc_result = posthoc_nemenyi_friedman(ranks)
    else:
        raise ValueError(f"Unsupported test method '{test_method}'."
                         " Choose 'tukey' or 'nemenyi'.")
    posthoc_result= DataFrameFormatter(f"Posthoc {test_method} Results").add_df (
        posthoc_result)
    return posthoc_result

@isdf 
def visualize_model_performance(
    model_performance_data: DataFrame, 
    plot_type: str='scatter', 
    **kwargs: Any):
    """
    Visualizes the performance of different models using a variety of plot types.
    This function aims to provide a versatile way to graphically compare model
    performances across multiple datasets or performance metrics.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics of models. Each column
        represents a different model, and each row represents a performance
        metric or a dataset.
    plot_type : str, optional
        The type of plot to generate. Available options are:
        - 'paired_box': Paired box plots for each model.
        - 'violin': Violin plots showing the distribution of the performance metrics.
        - 'difference': Bar plot of the difference in performance between models.
        - 'scatter': Scatter plot of performance metrics for each model.
        - 'cdf': Cumulative distribution function plot for model performances.
        Default is 'scatter'.
    **kwargs : dict
        Additional keyword arguments passed to the underlying matplotlib or
        seaborn plotting function.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.stats.model_comparisons import visualize_model_performance
    >>> # Example DataFrame of model performances
    >>> data = {'Model A': [0.8, 0.82, 0.78],
    ...         'Model B': [0.79, 0.84, 0.76],
    ...         'Model C': [0.81, 0.83, 0.77]}
    >>> model_performance_df = pd.DataFrame(data)
    >>> # Visualize using a scatter plot
    >>> visualize_model_performance(model_performance_df, plot_type='scatter')

    Notes
    -----
    - The 'paired_box' and 'violin' plots are useful for visualizing the
      spread and distribution of performance metrics across models.
    - The 'difference' plot is particularly helpful for directly comparing
      the performance of two models across multiple metrics or datasets.
    - The 'scatter' plot provides a granular look at each performance metric
      or dataset, offering insights into the variability across models.
    - The 'cdf' plot helps in understanding the proportion of datasets where
      models achieve certain performance thresholds, offering a cumulative
      perspective.

    Each plot type can be further customized through `**kwargs`, providing
    flexibility in adjusting plot aesthetics to best suit the data visualization
    needs.
    """

    plot_type = str(plot_type).lower()
    valid_plot_types = ['paired_box', 'violin', 'difference', 'scatter', 'cdf']
    
    if plot_type not in valid_plot_types:
        raise ValueError(f"Unsupported plot type '{plot_type}'. "
                         f"Available plot types are {valid_plot_types}.")
    
    if plot_type == 'paired_box' or plot_type == 'violin':
        fig, ax = plt.subplots()
        if plot_type == 'paired_box':
            sns.boxplot(data=model_performance_data, ax=ax, **kwargs)
        else:
            sns.violinplot(data=model_performance_data, ax=ax, **kwargs)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Performance Metric')

    elif plot_type == 'difference':
        differences = model_performance_data.diff(axis=1).iloc[:, 1]
        differences.plot(kind='bar', **kwargs)
        plt.title('Differences in Model Performances')
        plt.ylabel('Difference in Performance Metric')

    elif plot_type == 'scatter':
        pairs = pd.melt(model_performance_data.reset_index(), id_vars=['index'],
                        value_vars=model_performance_data.columns)
        sns.scatterplot(data=pairs, x='variable', y='value', hue='index', **kwargs)
        plt.title('Scatter Plot of Model Performances')
        plt.ylabel('Performance Metric')

    elif plot_type == 'cdf':
        for column in model_performance_data.columns:
            sns.ecdfplot(model_performance_data[column], label=column, **kwargs)
        plt.title('Cumulative Distribution of Model Performances')
        plt.ylabel('CDF')
        plt.legend()

    plt.xlabel('Datasets')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
@isdf 
def visualize_wilcoxon_test(
    model_performance_data: DataFrame, 
    alpha: float=0.05, 
    model_pairs: Optional[ Tuple[str, str]]=None,
    return_result: bool=False, 
    **kwargs: Any
    ):
    """
    Visualizes the results of the Wilcoxon signed-rank tests between pairs of
    models, highlighting significant differences in model performances.
    
    Function helps in identifying models with statistically different 
    performance metrics, thereby assisting in model selection and comparison
    processes.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A pandas DataFrame where rows represent different datasets or iterations,
        and columns represent different models. Each cell contains the performance
        metric (e.g., accuracy, F1 score) of a model on a dataset.
    alpha : float, optional
        The significance level to determine if the difference between model 
        performances is statistically significant. Defaults to 0.05.
    model_pairs : list of tuples, optional
        A list where each tuple contains two model names (as strings) to be 
        compared. If `None`, all possible pairs are compared. Defaults to `None`.
    return_result: bool, default=False, 
        Return the Wilcoxon result `Bunch` object and use print to see 
        the contents. 
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function, such as
        `figsize` to adjust the size of the plot.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the Wilcoxon test results, including the model 
        pairs, the Wilcoxon statistic values, p-values, and a flag indicating 
        significant differences.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import visualize_wilcoxon_test
    >>> df = pd.DataFrame({
    ...    'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...    'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...    'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> results= visualize_wilcoxon_test(df, alpha=0.05, return_result=True)
    >>> print(results)
       Model 1  Model 2  Statistic  p-value  Significant
    0  Model_A  Model_B        5.5   0.8125        False
    1  Model_A  Model_C        5.0   0.6250        False
    2  Model_B  Model_C        6.0   0.8125        False
    
    Results Interpretation 
    ----------------------
    The results from the `visualize_wilcoxon_test` function show the outcome
    of the Wilcoxon signed-rank tests between pairs of models. 
    For each pair, we have a Wilcoxon statistic and a corresponding p-value. 
    The 'Significant' column indicates whether the difference between the models
    is statistically significant based on the alpha level (0.05 in this case).

    Here's how to interpret the results:
    
    1. **Model_A vs. Model_B**:
       - **Statistic**: 5.5
       - **p-value**: 0.8125
       - **Significant**: False
       
       This means there is no statistically significant difference in the 
       performance of Model_A and Model_B. The high p-value 
       (greater than the alpha level of 0.05) suggests that any observed 
       difference in performance could be due to random chance rather than 
       a systematic difference between the models.
    
    2. **Model_A vs. Model_C**:
       - **Statistic**: 5.0
       - **p-value**: 0.6250
       - **Significant**: False
       
       Similarly, there is no statistically significant difference in the 
       performance of Model_A and Model_C. The p-value is again much higher 
       than the alpha level.
    
    3. **Model_B vs. Model_C**:
       - **Statistic**: 6.0
       - **p-value**: 0.8125
       - **Significant**: False
       
       Once more, no statistically significant difference is observed between
       Model_B and Model_C's performances.
    
    The bar plot would typically visualize these results by displaying the 
    Wilcoxon statistic for each model pair, with color-coding or some other
    visual indicator for whether the differences are statistically significant.
    Since all p-values are above the significance threshold (alpha = 0.05), 
    the plot suggests that there's no compelling evidence to prefer one model
    over the others based on the provided performance data.
    
    Notes
    -----
    The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test 
    used when comparing two related samples or repeated measurements on a single 
    sample to assess whether their population mean ranks differ. It's an alternative 
    to the paired Student's t-test when the population cannot be assumed to be 
    normally distributed.
    """
    # Prepare the plot
    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))

    # Results container
    results = []
    
    # Generate models pair if not given 
    model_pairs = generate_model_pairs(model_performance_data, model_pairs)
    # Ensure all model pairs exist in DataFrame columns
    for model_a, model_b in model_pairs:
        if ( 
                model_a not in model_performance_data.columns 
                or model_b not in model_performance_data.columns
            ):
            raise ValueError(
                f"One or both models in pair ({model_a}, {model_b})"
                " do not exist in the DataFrame."
                )
    # Perform Wilcoxon tests and collect results
    for model_a, model_b in model_pairs:
        performance_a = model_performance_data[model_a]
        performance_b = model_performance_data[model_b]
        stat, p_value = wilcoxon(performance_a, performance_b)
        results.append({'Model 1': model_a, 'Model 2': model_b, 
                        'Statistic': stat, 'p-value': p_value})

    # Convert results to DataFrame for easy plotting
    results_df = pd.DataFrame(results)

    # Add a column indicating significant differences
    results_df['Significant'] = results_df['p-value'] < alpha

    # Plotting
    # Plotting
    sns.barplot(x='Model 1', y='Statistic', hue='Significant', data=results_df, 
                ax=ax, **kwargs)
    ax.set_title('Wilcoxon Test Results Between Model Pairs')
    ax.set_ylabel('Wilcoxon Statistic')
    ax.set_xlabel('Model Pairs')
    ax.legend(title='Significant Difference', loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    if return_result : 
        return DataFrameFormatter(
            "wilcoxon Results", descriptor="WilcoxonTest").add_df (results_df)

@isdf 
def generate_model_pairs(
    model_performance_data: Union[Dict[str, List[float]], DataFrame], 
    model_pairs:Optional[ List[Tuple[str, str]]]=None
    ):
    """
    Generates pairs of models for comparison if not explicitly provided.

    This function utilizes itertools to automatically generate all possible
    unique pairs of models for comparison based on the columns of the input
    DataFrame, model_performance_data.

    Parameters:
    -----------
    model_performance_data: pd.DataFrame
        A DataFrame where rows represent datasets and columns represent models.
        Each cell should contain the performance metric of a model on a dataset.
    model_pairs: list of tuple, optional
        Explicitly specified list of model pairs to compare. If None, all
        unique pairs will be generated based on the DataFrame's columns.

    Returns:
    --------
    list of tuples
        A list of tuples, where each tuple contains the names of two models to
        compare. If model_pairs is given, returns it unmodified.

    Examples:
    ---------
    >>> from gofast.stats.model_comparisons import generate_model_pairs
    >>> model_performance_data = pd.DataFrame({
    ...     'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...     'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...     'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> model_pairs = generate_model_pairs(model_performance_data)
    >>> print(model_pairs)
    [('Model_A', 'Model_B'), ('Model_A', 'Model_C'), ('Model_B', 'Model_C')]
    """
    if model_pairs is not None:
        return model_pairs

    # Automatically generate all unique pairs of models from the DataFrame columns
    model_names = model_performance_data.columns
    generated_model_pairs = list(itertools.combinations(model_names, 2))
    
    return generated_model_pairs

@isdf 
def compute_model_summary(
    model_performance_data:Union[Dict[str, List[float]], DataFrame], 
    higher_is_better: bool =True
    ):
    
    """ 
    Compute a summary statistic for model performance.
    
    The `compute_model_summary` function is designed to evaluate the performance 
    of various machine learning models by computing a summary statistic for 
    each model. 
    
    This statistic is a combination of the mean performance metric of the model
    across multiple datasets and its variability (as measured by variance). 
    The purpose of adjusting the mean by the variance is to provide a single 
    measure that takes into account both the central tendency and the 
    reliability of the model's performance.
    
    Parameters:
    -----------
    - `model_performance_data`: A pandas DataFrame containing the performance 
       metrics for each model. Each column in the DataFrame corresponds to a 
       different model, and each row corresponds to the model's performance 
       metric on a different dataset.
       
    - `higher_is_better`: A boolean flag indicating the nature of the 
    performance metric. If set to `True`, the function assumes that a higher 
    metric indicates better performance (e.g., accuracy). Conversely, if set 
    to `False`, it assumes that a lower metric is better (e.g., error rate).
    
    The function modifies the mean performance based on the variance for each 
    model. If higher performance metrics are better, it subtracts the variance 
    from the mean, thereby penalizing models with more variable performance. 
    If lower metrics are better, it adds the variance, penalizing models with 
    less consistent performance. 
    
    The resultant summary statistics are then sorted from best to worst, based
    on the adjusted mean metric. By default, the function sorts in descending 
    order for metrics where higher is better and in ascending order where
    lower is better.
    
    Returns
    --------
    - A pandas DataFrame that has been formatted for readability using the 
     `DataFrameFormatter` class, which presents the summary statistics in a 
     structured table format. This DataFrame is wrapped in a Bunch object 
     (similar to sklearn's Bunch), allowing users to print it to see its 
      contents neatly tabulated.

    Notes
    -----
    
    - The summary statistic provided by this function gives a quick way to 
      compare models when considering both their average performance and their 
      consistency.
    - This approach can be particularly useful when choosing between models 
      where performance is close to each other and a decision cannot be made 
      on mean performance alone.
    - Care should be taken when interpreting the summary statistic as it 
      heavily penalizes models with high variability, which might not always 
      be appropriate depending on the context and the domain-specific 
      requirements.
      
    Examples
    ----------
    >>> from gofast.stats.model_comparisons import compute_model_summary
    >>> # Example performance data for 5 models across 5 datasets
    >>> model_performance_data = pd.DataFrame({
    ...    'LR': [86.07, 87.90, 88.57, 90.95, 83.33],
    ...    'DT': [77.58, 83.86, 78.33, 86.47, 85.61],
    ... })
    
    >>> # Compute and print summary statistics assuming higher performance is better
    >>> summary = compute_model_summary(model_performance_data)
    >>> print(summary)
    """
    if higher_is_better:
        # For metrics where higher is better, we negate the variance because we want
        # to subtract it from the mean, as higher variance is undesirable.
        summary = model_performance_data.mean() - model_performance_data.var()
    else:
        # For metrics where lower is better, we add the variance because we want
        # to penalize higher variance.
        summary = model_performance_data.mean() + model_performance_data.var()
    
    # Return the models ranked by the summary statistic
    summary= summary.sort_values(ascending= not higher_is_better)
    formatted_summary = DataFrameFormatter('Summary Statistics', 
        descriptor="ModelSummary", keyword='summary', 
        series_name = "average_score").add_df (summary) 
    
    return formatted_summary # Bunch object and print to see content 

def plot_model_summary(
    summary: Union [Series, DataFrame, DataFrameFormatter], 
    higher_is_better: bool = True,
    plot_type: str = 'bar',  
    figsize: Tuple[int, int] = (10, 6),   
    color='blue', 
):
    """
    Visualizes the summary of machine learning model performance data.

    This function creates a plot to visualize the performance of multiple 
    machine learning models based on a summary statistic. The summary can 
    either be directly provided as a pandas DataFrame or through a custom 
    formatter class that contains the performance summary. This function 
    supports bar, horizontal bar, and line plots to cater to different 
    visualization preferences.

    Parameters
    ----------
    summary : Union[Dict[str, List[float]], pandas.DataFrame, DataFrameFormatter]
        The model performance data to visualize. This can be a dictionary of 
        lists, a pandas DataFrame, or an instance of a custom formatter class
        (e.g., DataFrameFormatter) containing model performance metrics. Each 
        key or column should represent a different model, and each value or row 
        corresponds to the model's performance metric across different datasets 
        or evaluation metrics.
    higher_is_better : bool, optional
        A boolean indicating whether higher values of the performance metric 
        represent better performance. For metrics where a higher value is better
        (e.g., accuracy), this should be True. For metrics where a lower value 
        is better (e.g., error rate), this should be False. Default is True.
    plot_type : str, optional
        The type of plot to generate, which can be 'bar' for vertical bar plots,
        'barh' for horizontal bar plots, or 'line' for line plots. Default is 
        'bar'.
    figsize : Tuple[int, int], optional
        The size of the figure to create, specified as (width, height) in 
        inches. Default is (10, 6).

    Raises
    ------
    ValueError
        - Raised if `summary` is None or an unsupported type.
        - Raised if `plot_type` is not one of 'bar', 'barh', or 'line'.
    TypeError
        Raised if `summary` is not an instance of a supported type 
        (pandas DataFrame or custom formatter class).
    AttributeError
        Raised if a custom formatter class instance does not have a 'df' or 
        compatible attribute containing the summary data.

    Examples
    --------
    Using a pandas DataFrame as input:
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import plot_model_summary
    >>> performance_data = pd.DataFrame({
    ...     'Model A': [0.8, 0.82, 0.85],
    ...     'Model B': [0.75, 0.78, 0.8]
    ... })
    >>> plot_model_summary(performance_data, plot_type='bar')

    Using a custom DataFrameFormatter instance:
    >>> import gofast.api.formatter import DataFrameFormatter
    >>> formatter = DataFrameFormatter().add_df(performance_data)
    >>> plot_model_summary(formatter, higher_is_better=False, plot_type='line')

    Notes
    -----
    - This function is designed to provide a flexible means of visualizing 
      model performance across different evaluation metrics or datasets. It 
      abstracts the complexity of data preparation and plotting, offering a
      streamlined interface for users.
    - Custom formatter classes like DataFrameFormatter are assumed to have a 
      `.df` attribute or similar, which contains the summary data in a pandas 
      DataFrame format. If using such a class, ensure it is compatible with this
      function's requirements.
    - The visualization can be customized with various `plot_type` and `figsize`
      parameters to suit different preferences and display needs. The choice 
      of plot type and figure size should be guided by the context of the
      data presentation and the audience's needs.
    """
    summary_data = None
    if isinstance (summary, pd.Series): 
        # consider series as model_performance summary result
        # with index the model names.
        summary= summary.to_frame () 
        
    if isinstance(summary, pd.DataFrame): 
        if len(summary.columns)> 1:
            # consider as model_performance_data and compute model summary
            summary_data = compute_model_summary(summary, higher_is_better).df
        else: summary_data = summary.copy() 
        
    elif hasattr(summary, 'df'):  
        # Assuming DataFrameFormatter or similar with 'df' attribute
        summary_data = summary.df.astype (float)
    elif hasattr(summary, 'dfs'):  # Assuming MultiFrameFormatter with 'dfs'
        if len(summary.dfs) > 1:
            warnings.warn("Multiple DataFrames found; only the first will be used.")
        summary_data = summary.dfs[0].astype (float)
    else:
        raise ValueError(
            f"Unsupported summary input {type(summary).__name__!r}. Expect"
            " a pandas DataFrame or an instance of DataFrameFormatter.")

    # Sorting the summary data to reflect the best models accordingly
    if higher_is_better:
        summary_data = summary_data.sort_values(
            ascending=False, by = list(summary_data.columns))
    else:
        summary_data = summary_data.sort_values(
            by = list(summary_data.columns))
        
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type in ['bar', 'barh']:
        # Plot data
        bars = summary_data.plot(kind=plot_type, edgecolor='k', color=color,
                          rot=45 if plot_type == 'bar' else 0, legend=True, ax=ax)
        # Add text annotations for bars
        for bar in bars.patches:
            # Get the correct coordinate based on the plot type
            if plot_type == 'bar':
                height = bar.get_height()  # Height of the bar
                # Position text at the top of the bar, adjust y value as needed
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', 
                        ha='center', va='bottom')
            elif plot_type == 'barh':
                width = bar.get_width() # Width of the bar
                # Position text at the end of the bar, adjust x value as needed
                ax.text(width, bar.get_y() + bar.get_height() / 2.0, f' {width:.2f}', 
                        ha='left', va='center')
    
    elif plot_type == 'line':
        # Plot data for line
        summary_data.plot(kind='line', marker='o', linestyle='-', color=color, 
                          legend=True, ax=ax)
        # Add text annotations for each point on the line
        for x, y in enumerate(summary_data.values):
            print(x, y)
            ax.text(x, y, f' {float(y):.2f}', ha='center', va='bottom')
    else:
        raise ValueError(
            f"Unsupported plot_type: {plot_type}. Choose from 'bar', 'line', 'barh'.")
    
    ax.set_title('Model Performance Summary', fontsize=14)
    
    # Swap the X and Y labels because it's a horizontal bar chart
    ax.set_xlabel('Adjusted Performance Metric' if plot_type =='barh' else 'Models',
                  fontsize=12)
    ax.set_ylabel('Models' if plot_type=='barh' else 'Adjusted Performance Metric', 
                  fontsize=12)
    
    if 'bar' in plot_type:
        if plot_type == 'barh':
            # For horizontal bar charts, set the y-tick labels
            # (model names) on the vertical axis
            ax.set_yticks(range(len(summary_data.index)))
            ax.set_yticklabels(summary_data.index)  
        else:
            # For vertical bar charts, set the x-tick labels
            # (model names) on the horizontal axis
            ax.set_xticks(range(len(summary_data.index)))
            ax.set_xticklabels(summary_data.index, rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # Sample dataset
    data = {
        'LR': [86.07, 87.90, 88.57, 90.95, 83.33],
        'DT': [77.58, 83.86, 78.33, 86.47, 85.61],
        'RF': [86.52, 88.65, 90.50, 87.54, 83.46],
        'XGB': [79.53, 90.52, 86.21, 83.48, 82.50],
        'SVM': [81.29, 77.03, 80.15, 82.63, 79.56],
        'NB': [82.94, 97.93, 78.57, 85.97, 92.85],
        'KNN': [82.88, 79.16, 83.80, 82.47, 78.85]
    }
    
    df = pd.DataFrame(data)
    
    # Compute the summary statistic
    model_summary = compute_model_summary(df)
    model_summary

    # Example usage with mock data
    model_performance_mock_data = pd.DataFrame({
        'Model_A': np.random.rand(10),
        'Model_B': np.random.rand(10),
        'Model_C': np.random.rand(10),
        'Model_D': np.random.rand(10)
    })

    plot_model_rankings(model_performance_mock_data)
    # Example usage
    model_performance_data = pd.DataFrame({
        'Model_A': [0.90, 0.85, 0.88],
        'Model_B': [0.92, 0.83, 0.89],
        'Model_C': [0.91, 0.84, 0.87]
    })

    ranks = model_performance_data.rank().mean()
    cd_value = 0.5  # Example CD value
    plot_cd_diagram(ranks, cd_value, title='Example CD Diagram')
    
    # Sample data: Model performance on 5 datasets (rows) for 3 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
        'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
        'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    }
    model_performance_data = pd.DataFrame(data)
    
    # Perform the Wilcoxon signed-rank test
    wilcoxon_results = perform_wilcoxon_test2(model_performance_data)
    print(wilcoxon_results)


    # Sample data: Model performance on 3 datasets (rows) for 4 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78],
        'Model_B': [0.79, 0.84, 0.76],
        'Model_C': [0.81, 0.83, 0.77],
        'Model_D': [0.8, 0.85, 0.75]
    }
    model_performance_data = pd.DataFrame(data)
    
    # Compute ranks
    ranks = compute_model_ranks(model_performance_data)
    print(ranks)
    
    #
    # Sample data: Model performance on 3 datasets (rows) for 4 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78],
        'Model_B': [0.79, 0.84, 0.76],
        'Model_C': [0.81, 0.83, 0.77],
        'Model_D': [0.8, 0.85, 0.75]
    }
    model_performance_data = pd.DataFrame(data)
    pairwise_comparisons = perform_nemenyi_posthoc_test2(model_performance_data)
    print(pairwise_comparisons)


    # Sample data: Model performance on 3 datasets (rows) for 4 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78],
        'Model_B': [0.79, 0.84, 0.76],
        'Model_C': [0.81, 0.83, 0.77],
        'Model_D': [0.8, 0.85, 0.75]
    }
    model_performance_data = pd.DataFrame(data)
    result = perform_friedman_test2(model_performance_data)
    print(result)
