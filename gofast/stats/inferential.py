# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from ..api.formatter import DataFrameFormatter 
from ..api.types import Optional, List, Dict, Union, Tuple, Callable
from ..api.types import NumPyFunction, DataFrame, ArrayLike, Array1D, Series
from ..decorators import DynamicMethod
from ..tools.validator import assert_xy_in, is_frame, check_consistent_length 
from ..tools.coreutils import process_and_extract_data, to_series_if 
from ..tools.coreutils import get_colors_and_alphas, normalize_string 
from ..tools.coreutils import smart_format, check_uniform_type
from ..tools.funcutils import make_data_dynamic, ensure_pkg
from ..tools.funcutils import convert_and_format_data

from .utils import validate_stats_plot_type, fix_rm_anova_dataset

__all__=[
      "anova_test",
      "bootstrap",
      "check_and_fix_rm_anova_data",
      "check_anova_assumptions",
      "chi2_test",
      "cronbach_alpha",
      "friedman_test",
      "kolmogorov_smirnov_test",
      "kruskal_wallis_test",
      "levene_test",
      "mcnemar_test",
      "mixed_effects_model",
      "paired_t_test",
      "t_test_independent",
      "wilcoxon_signed_rank_test",
      "statistical_tests",
    ]

@make_data_dynamic('numeric', dynamize= False )
@ensure_pkg(
    "statsmodels", 
    extra="'statsmodels' needs to be installed for checking anova assumptions."
 )
def check_anova_assumptions(
    data: Union[DataFrame, Dict[str, List[float]]],
    significance_level: float = 0.05,
    view: bool = False,
    verbose: bool = True
) -> DataFrameFormatter:
    """
    Performs checks on the critical assumptions required for the validity
    of ANOVA (Analysis of Variance) tests. These assumptions include the 
    normality of residuals, the homogeneity of variances, and the independence 
    of observations. This function tests these assumptions and provides a 
    summary of the findings, optionally displaying a Q-Q plot for a graphical 
    assessment of the normality of residuals.

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, List[float]]]
        A collection of data points grouped by category. This can be provided as
        a pandas DataFrame with each column representing a group or a dictionary 
        with group names as keys and lists of data points as values.

    significance_level : float, default 0.05
        The threshold p-value below which the assumptions are considered violated.
        The conventional level is 0.05, indicating a 5% risk of concluding that 
        an assumption is violated when it is true.

    view : bool, default False
        A flag that, when set to True, triggers the display of a Q-Q plot to 
        visually inspect the normality of residuals. 

    verbose : bool, default True
        When True, detailed results from the homogeneity and normality tests 
        are printed to the console.

    Returns
    -------
    DataFrameFormatter
    
        A formatted DataFrame containing the statistical test results, including 
        the test statistics and p-values for both the Levene test of homogeneity 
        of variances and the Shapiro-Wilk test for normality of residuals. It also 
        includes a boolean indication of whether each assumption is met.

    Examples
    --------
    >>> from gofast.stats.inferential import check_anova_assumptions
    >>> data = {
            'Group1': [20, 21, 19, 20, 21],
            'Group2': [30, 31, 29, 30, 31],
            'Group3': [40, 41, 39, 40, 41]
        }
    >>> check_anova_assumptions(data, view=True)

    Notes
    -----
    ANOVA is sensitive to deviations from these assumptions, which may lead to 
    incorrect conclusions. Therefore, prior to performing ANOVA, it is essential 
    to ensure that these assumptions are not violated. If they are, transformations 
    or alternative non-parametric methods may be considered.

    The function internally reshapes the data suitable for ANOVA using `statsmodels`
    and conducts Levene's test and Shapiro-Wilk test, two commonly used tests for 
    checking ANOVA assumptions. If `view` is set to True, it also uses `statsmodels`
    graphics for the Q-Q plot to assess the normality of residuals.

    The function's output is valuable in guiding the appropriate analysis approach 
    and ensuring the integrity of ANOVA test results.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Make sure data is a pandas DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    # Reshape data for ANOVA
    melted_data = pd.melt(data.reset_index(), id_vars=['index'], 
                          value_vars=data.columns)
    melted_data.columns = ['index', 'groups', 'value']

    # Ensure all data is numeric
    melted_data['value'] = pd.to_numeric(melted_data['value'], errors='coerce')
    melted_data = melted_data.dropna(subset=['value'])

    # Levene's test for homogeneity of variances
    groups = melted_data.groupby('groups')['value'].apply(list)
    anova_stat, p_homogeneity = stats.levene(*groups)
    
    homogeneity_met = p_homogeneity >= significance_level
    
    if verbose:
        print_homogeneity_results(anova_stat, p_homogeneity, homogeneity_met)
    
    # OLS model for ANOVA with correct formula
    formula = 'value ~ C(groups)'
    anova_model = smf.ols(formula, data=melted_data).fit()
    residuals = anova_model.resid
  
    # Shapiro-Wilk test for normality on residuals
    shapiro_stat, p_normality = stats.shapiro(residuals)
    normality_met = p_normality >= significance_level
    if verbose:
        print_normality_results(shapiro_stat, p_normality, normality_met)
        
    # Visual inspection of normality: Q-Q plot
    if view:
        sm.qqplot(residuals, line='s') # sm.qqplot(residuals, line='45')
        plt.title('Q-Q Plot of Residuals')
        plt.show()
        
    # Compile results
    results = {
        'Levene_Statistic': anova_stat,
        'Levene_p_value': p_homogeneity,
        'Levene_Assumption_Met': homogeneity_met,
        'Shapiro_Statistic': shapiro_stat,
        'Shapiro_p_value': p_normality,
        'Shapiro_Assumption_Met': normality_met
    }
    formatted_results = DataFrameFormatter(
        "ANOVA Assumptions Check").add_df(pd.DataFrame([results]))
    
    return formatted_results

def print_homogeneity_results(stat: float, p_value: float, met: bool) -> None:
    """Prints the homogeneity test results."""
    message = ( "Levene's Test for Homogeneity of Variances:"
               f"Statistic={stat:.4f}, p-value={p_value:.4f}"
               )
    if met:
        message += " Assumption of homogeneity of variances is met."
    else:
        message += " Assumption of homogeneity of variances is violated."
    print(message)

def print_normality_results(stat: float, p_value: float, met: bool) -> None:
    """Prints the normality test results."""
    message =( "\nShapiro-Wilk Test for Normality of Residuals:"
              f" Statistic={stat:.4f}, p-value={p_value:.4f}"
              )
    if met:
        message += " Assumption of normality is met."
    else:
        message += " Assumption of normality is violated."
    print(message)


def wilcoxon_signed_rank_test(
     *samples: Array1D|DataFrame|str, 
     data: Optional [DataFrame]=None, 
     alternative:str='two-sided', 
     zero_method:str='auto', 
     as_frame:bool=False, 
     view:bool=False, 
     cmap:str='viridis', 
     fig_size:Tuple[int, int]=(10, 6), 
     **wilcoxon_kws
    ):
    """
    Perform the Wilcoxon Signed-Rank Test on two related samples and optionally
    visualize the distribution of differences between pairs.

    The Wilcoxon Signed-Rank Test is a non-parametric test used to compare two
    related samples, matched samples, or repeated measurements on a single
    sample to assess whether their population mean ranks differ. It is a paired
    difference test that can be used as an alternative to the paired Student's
    t-test when the data cannot be assumed to be normally distributed.

    Parameters
    ----------
    *samples : array-like or str
        The two sets of related samples as arrays, column names if `data` is 
        provided, or a single DataFrame.
    data : DataFrame, optional
        DataFrame containing the data if `samples` are specified as column names.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Specifies the alternative hypothesis to test against the null hypothesis
        that there is no difference between the paired samples. The options are:

        - ``two-sided``: Tests for any difference between the pairs without 
          assuming a direction (i.e., it tests for the possibility of the 
                                differences being either positive or negative).
          This is the most common choice for hypothesis testing as it does 
          not require a prior assumption about the direction of the effect.

        - ``greater``: Tests for the possibility that the differences between 
          the pairs are consistently greater than zero. This option is selected
          when there is a theoretical basis or prior evidence to suggest that 
          the first sample is expected to be larger than the second.

        - ``less``: Tests for the possibility that the differences between the 
          pairs are consistently less than zero. This option is appropriate 
          when the first sample is theorized or known to be smaller than the 
          second based on prior knowledge or evidence.

        The choice of the alternative hypothesis affects the interpretation 
        of the test results and should be made based on the specific research 
        question and the directionality of the expected effect. The default is 
        ``two-sided``, which does not assume any direction of the effect and 
        allows for testing differences in both directions.

    zero_method : {'pratt', 'wilcox', 'zsplit', 'auto'}, optional
        Defines how to handle zero differences between paired samples, which 
        can occur when the measurements for both samples are identical. 
        The options are:

        - ``pratt``: Includes zero differences in the ranking process, adjusting
           ranks accordingly.
        - ``wilcox``: Discards all zero differences before the test without 
           considering them for ranking.
        - ``zsplit``: Splits zero differences evenly between positive and 
           negative ranks.
        - ``auto``: Automatically selects between 'zsplit' and 'pratt' based on 
          the presence of zero differences in the data. If zero differences 
          are detected, ``zsplit`` is used to ensure that the test accounts 
          for these observations without excluding them. If no zero differences
          are present, ``pratt`` is used to include all non-zero differences 
          in the ranking process. This option aims to balance sensitivity and 
          specificity by adapting to the data characteristics.

        The choice of method can affect the test's sensitivity to differences
        and is particularly relevant in small samples or when a significant 
        proportion of the data pairs are identical.
        The default method is ``auto``, which provides a data-driven approach 
        to handling zero differences.
    
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    view : bool, default=False
        If True, generates a distribution plot of the differences with a zero line
        indicating no difference. Default is False.
    cmap : str, default='viridis'
        Colormap for the distribution plot.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    **wilcoxon_kws : keyword arguments
        Additional keyword arguments passed to :func:`scipy.stats.wilcoxon`.

    Returns
    -------
    statistic, p_value : float or pd.Series
        The Wilcoxon Signed-Rank test statistic and the associated p-value. Returns as
        a tuple or pandas Series based on the value of `as_frame`.
    
    Notes
    -----
    The test statistic is the sum of the ranks of the differences between the paired
    samples, where the ranks are taken with respect to the absolute values of the
    differences.

    .. math:: 
        W = \\sum_{i=1}^{n} \\text{sgn}(x_{2i} - x_{1i})R_i

    where :math:`x_{1i}` and :math:`x_{2i}` are the observations in the first and
    second sample respectively, :math:`\\text{sgn}` is the sign function, and
    :math:`R_i` is the rank of the absolute difference :math:`|x_{2i} - x_{1i}|`.

    Examples
    --------
    Performing a Wilcoxon Signed-Rank Test:

    >>> from gofast.stats.inferential import wilcoxon_signed_rank_test
    >>> data1 = np.random.normal(loc=10, scale=2, size=30)
    >>> data2 = data1 + np.random.normal(loc=0, scale=1, size=30)
    >>> statistic, p_value = wilcoxon_signed_rank_test(data1, data2)
    >>> print(statistic, p_value)

    Visualizing the distribution of differences:

    >>> wilcoxon_signed_rank_test(data1, data2, view=True, as_frame=True)
    """
    # Extract samples from DataFrame if specified
    if isinstance(data, pd.DataFrame) and all(isinstance(s, str) for s in samples):
        if len(samples) != 2:
            raise ValueError("Two column names must be provided with `data`.")
        data1, data2 = data[samples[0]].values, data[samples[1]].values
    elif len(samples) == 2 and all(isinstance(s, np.ndarray) for s in samples):
        data1, data2 = samples
    else:
        try: 
            data1, data2 = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise ValueError(
                "Samples must be two arrays or two column names with `data`.")

    # Check for zero differences and adjust zero_method if 'auto'
    differences = data2 - data1
    if zero_method == 'auto':
        if np.any(differences == 0):
            zero_method = 'zsplit'
            print("Zero differences detected. Using 'zsplit' method for zero_method.")
        else:
            zero_method = 'pratt'

    # Perform the Wilcoxon Signed-Rank Test
    try:
        statistic, p_value = stats.wilcoxon(
            data1, data2, zero_method=zero_method, alternative=alternative, 
            **wilcoxon_kws)
    except ValueError as e:
        raise ValueError(f"An error occurred during the Wilcoxon test: {e}")

    # Visualization
    if view:
        _visualize_differences(data1, data2, cmap, fig_size)

    # Return results
    if as_frame:
        return pd.Series({"W-statistic": statistic, "P-value": p_value},
                         name='Wilcoxon_Signed_Rank_test')
    return statistic, p_value

def _visualize_differences(data1, data2, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the distribution of differences between paired samples using a
    distribution plot with a line indicating no difference (zero).

    Parameters
    ----------
    data1, data2 : array-like
        The two sets of related samples.
    cmap : str, default='viridis'
        Colormap for the distribution plot.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    """
    differences = data2 - data1
    plt.figure(figsize=fig_size)
    sns.histplot(differences, kde=True, color=cmap)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Differences - Wilcoxon Signed-Rank Test')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.show()
    
def t_test_independent(
    sample1: Union[List[float], List[int], str],
    sample2: Union[List[float], List[int], str],
    alpha: float = 0.05, 
    data: DataFrame = None, 
    as_frame: bool=True, 
    view: bool = False, 
    plot_type: str = 'box', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6),  
    **kws
) -> Tuple[float, float, bool]:
    r"""
    Conducts an independent two-sample t-test to evaluate the difference in
    means between two independent samples. 
    
    This statistical test assesses whether there are statistically significant
    differences between the means of two independent samples.

    The t-statistic is computed as:
    
    .. math:: 
        t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
    
    
    Where:
    - \(\bar{X}_1\) and \(\bar{X}_2\) are the sample means,
    - \(s_1^2\) and \(s_2^2\) are the sample variances,
    - \(n_1\) and \(n_2\) are the sample sizes.
    
    The function returns the t-statistic, the two-tailed p-value, and a boolean
    indicating if the null hypothesis 
    (the hypothesis that the two samples have identical average values) 
    can be rejected at the given significance level (\(\alpha\)).

    Parameters
    ----------
    sample1 : Union[List[float], List[int], str]
        The first sample or the name of a column in `data` if provided.
    sample2 : Union[List[float], List[int], str]
        The second sample or the name of a column in `data` if provided.
    alpha : float, optional
        The significance level, default is 0.05.
    data : pd.DataFrame, optional
        DataFrame containing the data if column names are provided for 
        `sample1` or `sample2`.
    as_frame : bool, optional
        If True, returns results as a pandas DataFrame/Series.
    view : bool, optional
        If True, generates a plot to visualize the sample distributions.
    plot_type : str, optional
        The type of plot for visualization ('box' or 'hist').
    cmap : str, optional
        Color map for the plot.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the plot.
    **kwargs : dict
        Additional arguments passed to `stats.ttest_ind`.

    Returns
    -------
    Tuple[float, float, bool]
        t_stat : float
            The calculated t-statistic.
        p_value : float
            The two-tailed p-value.
        reject_null : bool
            A boolean indicating if the null hypothesis can be rejected 
            (True) or not (False).

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.inferential import t_test_independent
    >>> sample1 = [22, 23, 25, 27, 29]
    >>> sample2 = [18, 20, 21, 20, 19]
    >>> t_stat, p_value, reject_null = t_test_independent(sample1, sample2)
    >>> print(f"T-statistic: {t_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    >>> df = pd.DataFrame({'Group1': [22, 23, 25, 27, 29], 'Group2': [18, 20, 21, 20, 19]})
    >>> t_stat, p_value, reject_null = t_test_independent('Group1', 'Group2', data=df)
    >>> print(f"T-statistic: {t_stat}, P-value: {p_value}, Reject Null: {reject_null}")
    
    >>> df = pd.DataFrame({'Group1': [22, 23, 25, 27, 29], 'Group2': [18, 20, 21, 20, 19]})
    >>> t_stat, p_value, reject_null = t_test_independent('Group1', 'Group2', data=df)
    >>> print(f"T-statistic: {t_stat}, P-value: {p_value}, Reject Null: {reject_null}")
    Note
    ----
    This function is particularly useful for comparing the means of two 
    independent samples, especially in assessing differences under various 
    conditions or treatments. Ensure a DataFrame is passed when `sample1` a
    nd `sample2` are specified as column names.
    """

    if isinstance(sample1, str) or isinstance (sample2, str): 
        if data is None:
            raise ValueError(
                "Data cannot be None when 'x' or 'y' is specified as a column name.")
        # Validate that data is a DataFrame
        is_frame(data, df_only=True, raise_exception=True)  
        sample1 = data[sample1] if isinstance(sample1, str) else sample1 
        sample2 = data[sample2] if isinstance(sample2, str) else  sample2 

    check_consistent_length(sample1, sample2)
    
    t_stat, p_value = stats.ttest_ind(sample1, sample2, **kws )
    reject_null = p_value < alpha

    if view:
        
        plot_type= validate_stats_plot_type(
            plot_type, target_strs= ['box', 'hist'],
            raise_exception =True)
        colors, alphas = get_colors_and_alphas(2, cmap)
        if plot_type == 'box':
            sns.boxplot(data=[sample1, sample2], palette=cmap)
            plt.title('Sample Distributions - Boxplot')
        elif plot_type == 'hist':
            sns.histplot(sample1, color=colors[0], alpha=alphas[0], kde=True, 
                         label='Sample 1')
            sns.histplot(sample2, color=colors[1], alpha=alphas[1], kde=True, 
                         label='Sample 2')
            plt.title('Sample Distributions - Histogram')
        
        plt.xlabel("Value")
        plt.ylabel ("Frequency")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.show()
    
    if as_frame: 
        return to_series_if(
            t_stat, p_value, reject_null, 
            value_names= ["T-statistic", "P-value","Reject-Null-Hypothesis"],
            name ="t_test_independent")
    
    return t_stat, p_value, reject_null

def paired_t_test(
    *samples: Array1D|DataFrame|str, 
    data: Optional [DataFrame]=None, 
    as_frame: bool=False, 
    alternative:str='two-sided', 
    view:bool=False, 
    cmap:str='viridis', 
    fig_size:Tuple[int, int]=(10, 6), 
    **paired_test_kws
    ):
    """
    Perform the Paired t-Test on two related samples and optionally visualize
    the distribution of differences between pairs.

    The Paired t-Test is a parametric test used to compare two related samples,
    matched samples, or repeated measurements on a single sample to assess
    whether their population mean ranks differ. It assumes that the differences
    between the pairs are normally distributed.

    Parameters
    ----------
    *samples : array-like or str
        The two sets of related samples as arrays, column names if `data` is 
        provided, or a single DataFrame.
    data : DataFrame, optional
        DataFrame containing the data if `samples` are specified as column names.
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. The default is 'two-sided'.
    view : bool, default=False
        If True, generates a distribution plot of the differences with a zero line
        indicating no difference. Default is False.
    cmap : str, default='viridis'
        Colormap for the distribution plot.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    **paired_test_kws : keyword arguments
        Additional keyword arguments passed to :func:`scipy.stats.ttest_rel`.
    Returns
    -------
    statistic, p_value : float or pd.Series
        The Paired t-Test statistic and the associated p-value. Returns as
        a tuple or pandas Series based on the value of `as_frame`.

    Notes
    -----
    The Paired t-Test is based on the differences between the pairs of 
    observations.The test statistic is computed as follows:

    .. math::
        t = \\frac{\\bar{d}}{s_{d}/\\sqrt{n}}

    where :math:`\\bar{d}` is the mean of the differences between all pairs, 
    :math:`s_{d}` is the standard deviation of these differences, and 
    :math:`n` is the number of pairs. This formula assumes that the differences 
    between pairs are normally distributed.

    The null hypothesis for the test is that the mean difference between the paired 
    samples is zero. Depending on the alternative hypothesis specified, the test can 
    be two-tailed (default), left-tailed, or right-tailed.

    Examples
    --------
    Performing a Paired t-Test:

    >>> from gofast.stats.inferential import paired_t_test
    >>> data1 = np.random.normal(loc=10, scale=2, size=30)
    >>> data2 = data1 + np.random.normal(loc=0, scale=1, size=30)
    >>> statistic, p_value = paired_t_test(data1, data2)
    >>> print(statistic, p_value)

    Visualizing the distribution of differences:

    >>> paired_t_test(data1, data2, view=True, as_frame=True)
    """
    # Extract samples from DataFrame if necessary
    if data is not None:
        if len(samples) == 2 and all(isinstance(s, str) for s in samples):
            data1, data2 = data[samples[0]], data[samples[1]]
        else:
            raise ValueError("If `data` is provided, `samples`"
                             " must be two column names.")
    elif len(samples) == 2 and all(isinstance(s, np.ndarray) for s in samples):
        data1, data2 = samples
    else:
        try: 
            data1, data2  = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise ValueError("`samples` must be two arrays or "
                             "two column names with `data`.")
    
    # Perform the Paired t-Test
    statistic, p_value = stats.ttest_rel(
        data1, data2, alternative=alternative, **paired_test_kws)

    # Visualization
    if view:
        _visualize_paired_ttest_differences(data1, data2, cmap=cmap, fig_size=fig_size)

    # Return results
    if as_frame:
        return pd.Series({"T-statistic": statistic, "P-value": p_value},
                         name='Paired_T_Test')
    return statistic, p_value

def _visualize_paired_ttest_differences(
        data1, data2, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the distribution of differences between paired samples using a
    distribution plot with a line indicating no difference (zero).

    Parameters
    ----------
    data1, data2 : array-like
        The two sets of related samples.
    cmap : str, default='viridis'
        Colormap for the distribution plot. This will select a color from the colormap.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    """
    differences = data2 - data1
    plt.figure(figsize=fig_size)
    
    # Select a color from the colormap
    color = plt.get_cmap(cmap)(0.5)  # 0.5 denotes the midpoint of the colormap
    
    sns.histplot(differences, kde=True, color=color)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Differences - Paired t-Test')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.show()

@ensure_pkg ("statsmodels")
def mixed_effects_model(
    data: DataFrame, 
    formula: str, 
    groups: str, 
    re_formula: Optional[str] = None,
    data_transforms: Optional[List[Union[str, callable]]] = None,
    categorical: Optional[List[str]] = None,
    treatment: Optional[str] = None,
    order: Optional[List[str]] = None,
    summary: bool = True
) :
    """
    Fits a mixed-effects linear model to the data, accommodating both fixed 
    and random effects. 
    
    This approach is particularly useful for analyzing datasets with nested 
    structures or hierarchical levels, such as measurements taken from the 
    same subject over time or data clustered by groups. 
    
    Mixed-effects models account for both within-group (or subject) variance 
    and between-group variance.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the dependent variable, independent variables, 
        subject identifiers, and other covariates.
    formula : str
        A Patsy formula string specifying the fixed effects in the model. 
        E.g., 'score ~ time + treatment'.
    groups : str
        The column name in `data` that identifies the clustering unit or subject.
        Random effects are grouped by this identifier.
    re_formula : Optional[str], default None
        A Patsy formula string defining the structure of the random effects. 
        E.g., '~time' for random slopes for time.
        If None, only random intercepts for `groups` are included.
    data_transforms : Optional[List[Union[str, callable]]], default None
        Transformations to apply to the dataset before fitting the model. 
        This can be a list of column names to convert to categorical or callable
        functions that take the DataFrame as input and return a modified
        DataFrame.
    categorical : Optional[List[str]], default None
        Columns to convert to categorical variables. This is useful for 
        ensuring that categorical predictors use the
        correct data type.
    treatment : Optional[str], default None
        If specified, indicates the column to be treated as an ordered 
        categorical variable, useful for ordinal predictors.
    order : Optional[List[str]], default None
        The order of categories for the `treatment` column, necessary if 
        `treatment` is specified. Defines the levels and their order for an 
        ordered categorical variable.
    summary : bool, default True
        If True, prints a summary of the fitted model. Otherwise, returns 
        the model fit object.

    Returns
    -------
    sm.regression.mixed_linear_model.MixedLMResults
        The results instance for the fitted mixed-effects model.

    Mathematical Formulation
    ------------------------
    The model can be described by the equation:

    .. math:: y = X\\beta + Z\\gamma + \\epsilon

    where :math:`y` is the dependent variable, :math:`X` and :math:`Z` are 
    matrices of covariates for fixed and random effects,
    :math:`\\beta` and :math:`\\gamma` are vectors of fixed and random effects 
    coefficients, and :math:`\\epsilon` is the error term.

    Usage and Application Areas
    ---------------------------
    Mixed-effects models are particularly useful in studies where data are 
    collected in groups or hierarchies, such as longitudinal studies, clustered 
    randomized trials, or when analyzing repeated measures data. They allow for
    individual variation in response to treatments and can handle unbalanced 
    datasets or missing data more gracefully than traditional repeated measures 
    ANOVA.

    Examples
    --------
    Fitting a mixed-effects model to a dataset where scores are measured across 
    different times and treatments for each subject, with subjects as a 
    random effect:
        
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.inferential import mixed_effects_model
    >>> df = pd.DataFrame({
    ...     'subject_id': [1, 1, 2, 2],
    ...     'score': [5.5, 6.5, 5.0, 6.0],
    ...     'time': ['pre', 'post', 'pre', 'post'],
    ...     'treatment': ['A', 'A', 'B', 'B']
    ... })
    >>> mixed_effects_model(df, 'score ~ time * treatment', 'subject_id',
    ...                            re_formula='~time')
    In this example, 'score' is modeled as a function of time, treatment, 
    and their interaction, with random slopes for time grouped by 'subject_id'.
    """
    
    import statsmodels.formula.api as smf
    # Apply data transformations if specified
    if data_transforms:
        for transform in data_transforms:
            if callable(transform):
                data = transform(data)
            elif transform in data.columns:
                data[transform] = data[transform].astype('category')
    
    # Convert specified columns to categorical if requested
    if categorical:
        for col in categorical:
            data[col] = data[col].astype('category')
    
    # Set treatment column as ordered categorical if requested
    if treatment and order:
        data[treatment] = pd.Categorical(data[treatment], categories=order, ordered=True)
    
    # Fit the model
    model = smf.mixedlm(formula, data, groups=data[groups], re_formula=re_formula)
    model_fit = model.fit()
    
    # Print or return summary
    if summary:
        print(model_fit.summary())
    else:
        return model_fit


def levene_test(
    *samples: Union[List[Array1D], DataFrame], 
    columns: Optional[List[str]] = None,
    center: str = 'median', 
    proportiontocut: float = 0.05, 
    as_frame=True, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None,
    **kws
):
    """
    Perform Levene's test for equal variances across multiple samples, with an
    option to visualize the sample distributions.

    Levene's test is used to assess the equality of variances for a variable
    calculated for two or more groups. It is more robust to departures from
    normality than Bartlett's test. The test statistic is based on a measure
    of central tendency (mean, median, or trimmed mean).

    .. math:: 
        
        W = \\frac{(N - k)}{(k - 1)} \\frac{\\sum_{i=1}^{k} n_i (Z_{i\\cdot} - Z_{\\cdot\\cdot})^2}
        \; {\\sum_{i=1}^{k} \\sum_{j=1}^{n_i} (Z_{ij} - Z_{i\\cdot})^2}

    where :math:`W` is the test statistic, :math:`N` is the total number of 
    observations,:math:`k` is the number of groups, :math:`n_i` is the number
    of observations in group :math:`i`, :math:`Z_{ij}` is the deviation from 
    the group mean or median, depending on the centering
    method used.

    Parameters
    ----------
    *samples : Union[List[np.ndarray], pd.DataFrame]
        The sample data, possibly with different lengths. If a DataFrame is
        provided and `columns` are specified, it extracts data for each
        column name to compose samples.
    columns : List[str], optional
        Column names to extract from the DataFrame to compose samples.
        Ignored if direct arrays are provided.
    center : {'median', 'mean', 'trimmed'}, optional
        Specifies which measure of central tendency to use in computing the
        test statistic. Default is 'median'.
    proportiontocut : float, optional
        Proportion (0 to 0.5) of data to cut from each end when center is 
        'trimmed'. Default is 0.05.
    as_frame : bool, optional
        If True, returns the results as a pandas DataFrame. Default is True.
    view : bool, optional
        If True, displays a boxplot of the samples. Default is False.
    cmap : str, optional
        Colormap for the boxplot. Currently unused but included for future 
        compatibility.
    fig_size : Tuple[int, int], optional
        Size of the figure for the boxplot if `view` is True.
    **kws : dict
        Additional keyword arguments passed to `stats.levene`.

    Returns
    -------
    statistic : float
        The test statistic for Levene's test.
    p_value : float
        The p-value for the test.

    Examples
    --------
    Using direct array inputs:

    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.inferential import levene_test
    >>> sample1 = np.random.normal(loc=0, scale=1, size=50)
    >>> sample2 = np.random.normal(loc=0.5, scale=1.5, size=50)
    >>> sample3 = np.random.normal(loc=-0.5, scale=0.5, size=50)
    >>> statistic, p_value = levene_test(sample1, sample2, sample3, view=True)
    >>> print(f"Statistic: {statistic}, p-value: {p_value}")

    Using a DataFrame and specifying columns:

    >>> df = pd.DataFrame({'A': np.random.normal(0, 1, 50),
    ...                    'B': np.random.normal(0, 2, 50),
    ...                    'C': np.random.normal(0, 1.5, 50)})
    >>> statistic, p_value = levene_test(df, columns=['A', 'B', 'C'], view=True)
    >>> print(f"Statistic: {statistic}, p-value: {p_value}")
    """
    # Check if *samples contains a single DataFrame and columns are specified
    samples = process_and_extract_data(
        *samples, columns =columns, allow_split= True ) 
    statistic, p_value = stats.levene(
        *samples, center=center, proportiontocut=proportiontocut, **kws)

    if view:
        _visualize_samples(samples, columns=columns, fig_size=fig_size)

    if as_frame: # return series by default 
        return to_series_if(
            statistic, p_value, value_names=['L-statistic', "P-value"], 
            name ='levene_test'
            )
    return statistic, p_value

def _visualize_samples(samples, columns=None, fig_size=None):
    """
    Visualizes sample distributions using boxplots.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    plt.boxplot(samples, patch_artist=True, vert=True)
    labels = columns if columns else [f'Sample {i+1}' for i in range(len(samples))]
    plt.xticks(ticks=np.arange(1, len(samples) + 1), labels=labels)
    plt.title('Sample Distributions - Levene’s Test for Equal Variances')
    plt.ylabel('Values')
    plt.xlabel('Samples/Groups')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


@ensure_pkg("statsmodels")
def mcnemar_test(
    *samples: [Array1D, DataFrame, str], 
    data: Optional [DataFrame]=None, 
    as_frame:bool=False, 
    exact:bool=True, 
    correction:bool=True, 
    view:bool=False, 
    cmap: str='viridis', 
    fig_size: Tuple [int, int]=(10, 6)
    ):
    """
    Perform McNemar's test to compare two related samples on categorical data,
    with an option to visualize the contingency table.

    McNemar's test is a non-parametric method used to determine whether there 
    are differences between two related samples. It is suitable for binary
    categorical data to compare the proportion of discrepant observations.

    Parameters
    ----------
    *samples : str or array-like
        Names of columns in `data` DataFrame or two arrays containing the samples.
        When `data` is provided, `samples` must be the names of the columns to compare.
    data : DataFrame, optional
        DataFrame containing the samples if column names are specified in `samples`.
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    exact : bool, default=True
        If True, uses the exact binomial distribution for the test. Otherwise, 
        an asymptotic chi-squared approximation is used.
    correction : bool, default=True
        If True, applies continuity correction in the chi-squared approximation
        of the test statistic.
    view : bool, default=False
        If True, visualizes the contingency table as a heatmap.
    cmap : str, default='viridis'
        Colormap for the heatmap visualization.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the heatmap.

    Returns
    -------
    statistic, p_value : float or pd.Series
        The test statistic and p-value of McNemar's test. Returns as a tuple
        or pandas Series based on the value of `as_frame`.
    
    Raises
    ------
    TypeError
        If `data` is not a DataFrame when column names are specified in `samples`.
    ValueError
        If the number of samples provided is not equal to two.

    Notes
    -----
    McNemar's test evaluates the null hypothesis that the row and column marginal
    frequencies are equal. It is commonly used in before-after studies, matched
    pair studies, or repeated measures design where the subjects are the same.

    The test statistic is calculated as follows:

    .. math:: 
        Q = \\frac{(b - c)^2}{b + c}

    where :math:`b` and :math:`c` are the off-diagonal elements of the 2x2 
    contingency table formed by the two samples.

    Examples
    --------
    Performing McNemar's test with array inputs:

    >>> from gofast.stats.inferential import mcnemar_test
    >>> sample1 = [0, 1, 0, 1]
    >>> sample2 = [1, 0, 1, 1]
    >>> statistic, p_value = mcnemar_test(sample1, sample2)
    >>> print(statistic, p_value)

    Performing McNemar's test with DataFrame column names:

    >>> df = pd.DataFrame({'before': sample1, 'after': sample2})
    >>> result = mcnemar_test('before', 'after', data=df, view=True, as_frame=True)
    >>> print(result)
    """
    from statsmodels.stats.contingency_tables import mcnemar
    
    # Process input samples
    if isinstance(data, pd.DataFrame) and all(isinstance(s, str) for s in samples):
        samples = [data[col] for col in samples]
    elif not isinstance(data, pd.DataFrame) and len(samples) == 2:
        samples = list(samples)
    else:
        try: 
            samples = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise TypeError("Invalid input: `data` must be a DataFrame and `samples`"
                            " must be column names, or `samples` must be two sequences.")

    # Ensure there are exactly two samples
    if len(samples) != 2:
        raise ValueError("McNemar's test requires exactly two related samples.")

    # Create the contingency table and perform McNemar's test
    contingency_table = pd.crosstab(samples[0], samples[1])
    result = mcnemar(contingency_table, exact=exact, correction=correction)

    # Visualization
    if view:
        _visualize_contingency_table(contingency_table, cmap=cmap, fig_size=fig_size)

    # Return results
    if as_frame:
        return pd.Series({"M-statistic": result.statistic, "P-value": result.pvalue},
                         name='McNemar_test')
    
    return result.statistic, result.pvalue

def _visualize_contingency_table(
        contingency_table, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the contingency table of McNemar's test as a heatmap.
    """
    plt.figure(figsize=fig_size)
    sns.heatmap(contingency_table, annot=True, cmap=cmap, fmt='d')
    plt.title("McNemar's Test Contingency Table")
    plt.ylabel('Sample 1')
    plt.xlabel('Sample 2')
    plt.show()

def kruskal_wallis_test(
    *samples: Array1D|DataFrame|str, 
    data: Optional [DataFrame]=None, 
    as_frame:bool=False, 
    view:bool=False, 
    cmap: str='viridis', 
    fig_size: Tuple [int, int]=(10, 6),
    **kruskal_kws
    ):
    """
    Perform the Kruskal-Wallis H test for comparing more than two independent samples
    to determine if there are statistically significant differences between their 
    population medians. Optionally, visualize the distribution of each sample.

    The Kruskal-Wallis H test is a non-parametric version of ANOVA. It's used when the 
    assumptions of ANOVA are not met, especially the assumption of normally distributed 
    data. It ranks all data points together and then compares the sums of ranks between 
    groups.

    Parameters
    ----------
    *samples : sequence of array-like or str
        Input data for the test. When `data` is a DataFrame, `samples` can be
        column names.
    data : DataFrame, optional
        DataFrame containing the data if column names are specified in `samples`.
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    view : bool, default=False
        If True, generates boxplots of the sample distributions. Default is False.
    cmap : str, default='viridis'
        Colormap for the boxplot visualization.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the boxplot visualization.
    kruskal_kws: dict, 
        Keywords arguments passed to :func:`scipy.stats.kruskal`.
        
    Returns
    -------
    statistic, p_value : float or pd.Series
        The Kruskal-Wallis H statistic and the associated p-value. Returns as a tuple
        or pandas Series based on the value of `as_frame`.
    
    Raises
    ------
    TypeError
        If `data` is not a DataFrame when column names are specified in `samples`.
    ValueError
        If less than two samples are provided.

    Notes
    -----
    The Kruskal-Wallis test evaluates the null hypothesis that the population medians 
    of all groups are equal. It is recommended for use with ordinal data or when the 
    assumptions of one-way ANOVA are not met.

    The test statistic is calculated as follows:

    .. math:: 
        H = \\frac{12}{N(N+1)} \\sum_{i=1}^{g} \\frac{R_i^2}{n_i} - 3(N+1)

    where :math:`N` is the total number of observations across all groups, :math:`g` 
    is the number of groups, :math:`n_i` is the number of observations in the i-th 
    group, and :math:`R_i` is the sum of ranks in the i-th group.

    Examples
    --------
    Performing a Kruskal-Wallis H Test with array inputs:
    
    >>> import numpy as np 
    >>> from gofast.stats.inferential import kruskal_wallis_test
    >>> sample1 = np.random.normal(loc=10, scale=2, size=30)
    >>> sample2 = np.random.normal(loc=12, scale=2, size=30)
    >>> sample3 = np.random.normal(loc=11, scale=2, size=30)
    >>> statistic, p_value = kruskal_wallis_test(sample1, sample2, sample3)
    >>> print(statistic, p_value)

    Performing a Kruskal-Wallis H Test with DataFrame column names:

    >>> df = pd.DataFrame({'group1': sample1, 'group2': sample2, 'group3': sample3})
    >>> result = kruskal_wallis_test('group1', 'group2', 'group3', data=df, view=True, as_frame=True)
    >>> print(result)
    """
    # Process input samples
    if isinstance(data, pd.DataFrame) and all(isinstance(s, str) for s in samples):
        samples = [data[col] for col in samples]
    elif not isinstance(data, pd.DataFrame) and len(samples) >= 2:
        samples = list(samples)
    else:
        try: 
            samples = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise TypeError("Invalid input: `data` must be a DataFrame and `samples`"
                        " must be column names, or `samples` must be two or more sequences.")

    # Ensure there are at least two samples
    if len(samples) < 2:
        raise ValueError("Kruskal-Wallis H test requires at least two independent samples.")

    # Perform the Kruskal-Wallis H test
    statistic, p_value = stats.kruskal(*samples, **kruskal_kws)

    # Visualization
    if view:
        _visualize_sample_distributions(samples, cmap=cmap, fig_size=fig_size)

    # Return results
    if as_frame:
        return pd.Series({"H-statistic": statistic, "P-value": p_value},
                         name='Kruskal_Wallis_test')
    return statistic, p_value

def _visualize_sample_distributions(samples, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the distribution of each sample using boxplots.

    Parameters
    ----------
    samples : list of array-like
        The samples to visualize.
    cmap : str, default='viridis'
        Colormap for the boxplot visualization.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the boxplot visualization.
    """
    plt.figure(figsize=fig_size)
    plt.boxplot(samples, patch_artist=True)
    plt.xticks(range(1, len(samples) + 1), ['Sample ' + str(i) for i in range(
        1, len(samples) + 1)])
    plt.title('Sample Distributions - Kruskal-Wallis H Test')
    plt.ylabel('Values')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def kolmogorov_smirnov_test(
    data1: Union[Array1D, str],
    data2: Union[Array1D, str],  
    as_frame: bool = False, 
    alternative: str = 'two-sided',
    data: Optional[DataFrame] = None, 
    method: str = 'auto', 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None,
):
    """
    Perform the Kolmogorov-Smirnov (KS) test for goodness of fit between two
    samples, with an option to visualize the cumulative distribution functions
    (CDFs).

    The KS test is a nonparametric test that compares the empirical 
    distribution functions of two samples to assess whether they come from 
    the same distribution. It is based on the maximum difference between the 
    two cumulative distributions.

    .. math::
        D_{n,m} = \\sup_x |F_{1,n}(x) - F_{2,m}(x)|

    where :math:`\\sup_x` is the supremum of the set of distances, 
    :math:`F_{1,n}` and :math:`F_{2,m}` are the empirical distribution 
    functions of the first and second sample, respectively, and
    :math:`n` and :math:`m` are the sizes of the first and second sample.

    Parameters
    ----------
    data1, data2 : Union[np.ndarray, str]
        The sample observations, assumed to be drawn from a continuous distribution.
        If strings are provided, they should correspond to column names in `data`.
    as_frame : bool, optional
        If True, returns the test results as a pandas DataFrame if shape is 
        greater than 1 and pandas Series otherwise.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null hypothesis ('two-sided' default).
    data : pd.DataFrame, optional
        DataFrame containing the columns referred to by `data1` and `data2`.
    method : {'auto', 'exact', 'approx'}, optional
        Method used to compute the p-value.
    view : bool, optional
        If True, visualizes the cumulative distribution functions of both samples.
    cmap : str, optional
        Colormap for the visualization (currently not used).
    fig_size : Tuple[int, int], optional
        Size of the figure for the visualization if `view` is True.

    Returns
    -------
    statistic : float
        The KS statistic.
    p_value : float
        The p-value for the test.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.stats.inferential import kolmogorov_smirnov_test
    >>> data1 = np.random.normal(loc=0, scale=1, size=100)
    >>> data2 = np.random.normal(loc=0.5, scale=1.5, size=100)
    >>> statistic, p_value = kolmogorov_smirnov_test(data1, data2)
    >>> print(f"KS statistic: {statistic}, p-value: {p_value}")

    Using a DataFrame and specifying columns:

    >>> import pandas as pd
    >>> df = pd.DataFrame({'group1': np.random.normal(0, 1, 100),
                           'group2': np.random.normal(0.5, 1, 100)})
    >>> statistic, p_value = kolmogorov_smirnov_test('group1', 'group2', data=df, view=True)
    >>> print(f"KS statistic: {statistic}, p-value: {p_value}")
    """

    data1, data2 = assert_xy_in(data1, data2 , data=data, xy_numeric= True ) 
    statistic, p_value = stats.ks_2samp(
        data1, data2, alternative=alternative, mode=method)

    if view:
        _visualize_cdf_comparison(data1, data2, fig_size)

    return to_series_if(
        statistic, p_value, value_names=['K-statistic', "P-value"], 
        name ='levene_test'
        ) if as_frame else ( statistic, p_value) 

def _visualize_cdf_comparison(data1, data2, fig_size):
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    for sample, label in zip([data1, data2], ['Data1', 'Data2']):
        ecdf = lambda x: np.arange(1, len(x) + 1) / len(x)
        plt.step(sorted(sample), ecdf(sample), label=f'CDF of {label}')
    plt.title('CDF Comparison')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

def friedman_test(
    *samples: Union[np.ndarray, pd.DataFrame], 
    columns: Optional[List[str]] = None, 
    method: str = 'auto', 
    as_frame: bool = False, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = None
) -> Union[Tuple[float, float], Series]:
    """
    Perform the Friedman test, a non-parametric statistical test used to detect 
    differences between groups on a dependent variable across multiple test 
    attempts.

    The Friedman test [1]_ is used when the data violate the assumptions of parametric 
    tests, such as the normality and homoscedasticity assumptions required for 
    repeated measures ANOVA.

    The test statistic is calculated as follows:

    .. math:: 
        Q = \\frac{12N}{k(k+1)}\\left[\\sum_{j=1}^{k}R_j^2 - \\frac{k(k+1)^2}{4}\\right]

    where :math:`N` is the number of subjects, :math:`k` is the number of groups, 
    and :math:`R_j` is the sum of ranks for the :math:`j`th group.

    Parameters
    ----------
    *samples : Union[np.ndarray, pd.DataFrame]
        The sample groups as separate arrays or a single DataFrame. If a DataFrame 
        is provided and `columns` are specified, it will extract data for each 
        column name to compose samples.
    columns : Optional[List[str]], optional
        Column names to extract from the DataFrame to compose samples, 
        by default None.
    method : {'auto', 'exact', 'asymptotic'}, optional
        The method used for the test:
        - 'auto' : Use exact method for small sample sizes and asymptotic
          method for larger samples.
        - 'exact' : Use the exact distribution of the test statistic.
        - 'asymptotic' : Use the asymptotic distribution of the test statistic.
        The method to use for computing p-values, by default 'auto'.
        The actual friedmanchisquare only supports sample sizes and asymptotic
        method for larger samples.
    as_frame : bool, optional
        If True, returns the test statistic and p-value as a pandas Series, 
        by default False.
    view : bool, optional
        If True, displays a box plot of the sample distributions, by default False.
    cmap : str, optional
        Colormap for the box plot, by default 'viridis'. Currently unused.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the box plot, by default None.

    Returns
    -------
    Union[Tuple[float, float], pd.Series]
        The Friedman statistic and p-value, either as a tuple or as a pandas 
        Series if `as_frame` is True.

    Notes
    -----
    The Friedman test is widely used in scenarios where you want to compare
    the effects of different treatments or conditions on the same subjects,
    especially in medical, psychological, and other scientific research.
    It is an alternative to ANOVA when the normality assumption is not met.

    References
    ----------
    .. [1] Friedman, Milton. (1937). The use of ranks to avoid the assumption
          of normality implicit in the analysis of variance. Journal of the
          American Statistical Association.
    
    Examples
    --------
    Using array inputs:
     
    >>> from gofast.stats import friedman_test
    >>> group1 = [20, 21, 19, 20, 21]
    >>> group2 = [19, 20, 18, 21, 20]
    >>> group3 = [21, 22, 20, 22, 21]
    >>> statistic, p_value = friedman_test(group1, group2, group3, method='auto')
    >>> print(f'Friedman statistic: {statistic}, p-value: {p_value}')

    Using DataFrame input:

    >>> df = pd.DataFrame({'group1': group1, 'group2': group2, 'group3': group3})
    >>> statistic, p_value = friedman_test(df, columns=['group1', 'group2', 'group3'], view=True)
    >>> print(f'Friedman statistic with DataFrame input: {statistic}, p-value: {p_value}')

    """
    # Check if *samples contains a single DataFrame and columns are specified
    samples = process_and_extract_data(
        *samples, columns =columns, allow_split= True ) 
    # Convert all inputs to numpy arrays for consistency
    samples = [np.asarray(sample) for sample in samples]

    # Perform the Friedman test
    statistic, p_value = stats.friedmanchisquare(*samples)

    if view:
        _visualize_friedman_test_samples(samples, columns, fig_size)

    return to_series_if(
        statistic, p_value, value_names=["F-statistic", "P-value"],name="friedman_test"
        )if as_frame else ( statistic, p_value )

def _visualize_friedman_test_samples(samples, columns, fig_size):
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    # Assume samples are already in the correct format for plotting
    plt.boxplot(samples, patch_artist=True, labels=columns if columns else [
        f'Group {i+1}' for i in range(len(samples))])
    plt.title('Sample Distributions - Friedman Test')
    plt.xlabel('Groups')
    plt.ylabel('Scores')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

@make_data_dynamic(capture_columns= True, dynamize =False )
def cronbach_alpha(
    items_scores: ArrayLike,
    columns: Optional[list] = None,
    as_frame: bool = False, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = None
) -> Union[float, pd.Series]:
    """
    Calculate Cronbach's Alpha for assessing the internal consistency or 
    reliability of a set of test or survey items.

    Cronbach's Alpha is defined as:

    .. math:: 
        
        \\alpha = \\frac{N \\bar{\\sigma_{item}^2}}{\\sigma_{total}^2}
        \; (1 - \\frac{\\sum_{i=1}^{N} \\sigma_{item_i}^2}{\\sigma_{total}^2})

    where :math:`N` is the number of items, :math:`\\sigma_{item}^2` is the variance 
    of each item, and :math:`\\sigma_{total}^2` is the total variance of the scores.
    
    Parameters
    ----------
    items_scores : Union[np.ndarray, pd.DataFrame]
        A 2D array or DataFrame where rows represent scoring for each item and 
        columns represent items.
    columns : Optional[list], default=None
        Specific columns to use if `items_scores` is a DataFrame. If None, 
        all columns are used.
    as_frame : bool, default=False
        If True, returns the results as a pandas DataFrame or Series.
    view : bool, default=False
        If True, displays a bar plot showing the variance of each item.
    cmap : str, default='viridis'
        Colormap for the bar plot. Currently unused but included for future 
        compatibility.
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the bar plot. If None, defaults to matplotlib's default.

    Returns
    -------
    float or pd.Series
        Cronbach's Alpha as a float or as a pandas Series if `as_frame` is True.

    Notes
    -----
    Cronbach's Alpha values range from 0 to 1, with higher values indicating 
    greater internal consistency of the items. Values above 0.7 are typically 
    considered acceptable, though this threshold may vary depending on the context.
    
    Examples
    --------
    Using a numpy array:
    
    >>> import numpy as np 
    >>> from gofast.stats.inferential import cronbach_alpha
    >>> scores = np.array([[2, 3, 4], [4, 4, 5], [3, 5, 4]])
    >>> cronbach_alpha(scores)
    0.75

    Using a pandas DataFrame:
        
    >>> import pandas as pd 
    >>> df_scores = pd.DataFrame({'item1': [2, 4, 3], 'item2': [3, 4, 5], 'item3': [4, 5, 4]})
    >>> cronbach_alpha(df_scores)
    0.75

    Visualizing item variances:

    >>> cronbach_alpha(df_scores, view=True)
    Displays a bar plot of item variances.

    """
    items_scores = np.asarray(items_scores)
    if items_scores.ndim == 1:
        items_scores = items_scores.reshape(-1, 1)

    item_variances = items_scores.var(axis=0, ddof=1)
    total_variance = items_scores.sum(axis=1).var(ddof=1)
    n_items = items_scores.shape[1]

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    
    if view:
        _visualize_item_variances(item_variances, n_items, columns, fig_size, cmap)
        
    if as_frame:
        return pd.Series([alpha], index=['Cronbach\'s Alpha'],
                         name="cronbach_alpha")
    return alpha

def _visualize_item_variances(item_variances, n_items, columns, fig_size, cmap):
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    colors = plt.cm.get_cmap(cmap, n_items)
    plt.bar(range(1, n_items + 1), item_variances, color=[
        colors(i) for i in range(n_items)])
    plt.title("Item Variances")
    plt.xlabel("Item")
    plt.ylabel("Variance")
    plt.xticks(ticks=range(1, n_items + 1), labels=columns if columns else [
        f'Item {i}' for i in range(1, n_items + 1)])
    plt.show()

@make_data_dynamic('numeric', capture_columns=True)
def chi2_test(
    data: Union[ArrayLike, DataFrame],
    alpha: float = 0.05, 
    columns: List[str] = None,
    as_frame=True, 
    view: bool = False,
    plot_type=None, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5),
    **kwargs
) -> Tuple[float, float, bool]:
    """
    Performs a Chi-Squared test for independence to assess the relationship 
    between two categorical variables represented in a contingency table.

    The Chi-Squared test evaluates whether there is a significant association 
    between two categorical variables by comparing the observed frequencies in 
    the contingency table with the frequencies that would be expected if there 
    was no association between the variables.

    Parameters
    ----------
    data : array_like or pd.DataFrame
        The contingency table where rows represent categories of one variable
        and columns represent categories of the other variable. If `data` is 
        array_like and `as_frame` is True, `columns` must be provided to convert 
        it into a DataFrame.
    alpha : float, optional
        The significance level for determining if the null hypothesis can be
        rejected, default is 0.05.
    columns : List[str], optional
        Column names when converting `data` from array_like to a DataFrame.
        Required if `data` is array_like and `as_frame` is True.
    as_frame : bool, optional
        If True, returns the results in a pandas Series. Default is False.
    view : bool, optional
        If True, displays a heatmap of the contingency table. Default is False.
    plot_type : str or None, optional
        The type of plot to display. Currently not implemented; reserved for future use.
    cmap : str, optional
        Colormap for the heatmap. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the heatmap. Default is (12, 5).
    **kwargs : dict
        Additional keyword arguments to pass to `stats.chi2_contingency`.

    Returns
    -------
    chi2_stat : float
        The Chi-squared statistic.
    p_value : float
        The p-value of the test.
    reject_null : bool
        Indicates whether to reject the null hypothesis (True) or not (False),
        based on the comparison between `p_value` and `alpha`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> from gofast.stats.inferential import chi2_test 
    >>> data = pd.DataFrame({'A': [10, 20, 30], 'B': [20, 15, 30]})
    >>> chi2_stat, p_value, reject_null = chi2_test(data)
    >>> print(f"Chi2 Statistic: {chi2_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    Notes
    -----
    The mathematical formulation for the Chi-Squared test statistic is:

    .. math::
        \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}

    where :math:`O_i` are the observed frequencies, and :math:`E_i` are the
    expected frequencies under the null hypothesis that the variables are
    independent.

    The Chi-Squared test is a statistical method used to determine if there is a 
    significant association between two categorical variables. It's commonly 
    used in hypothesis testing to analyze the independence of variables in 
    contingency tables.
    """
    chi2_stat, p_value, _, _ = stats.chi2_contingency(data, **kwargs)
    reject_null = p_value < alpha
    
    if view:
        plt.figure(figsize=fig_size)
        sns.heatmap(data, annot=True, cmap=cmap, fmt="g")
        plt.title('Contingency Table')
        plt.show()
        
    if as_frame: 
        return to_series_if(
            chi2_stat, p_value, reject_null, 
            value_names=['Chi2-statistic', 'P-value', "Reject-Null-Hypothesis"], 
            name="chi_squared_test"
            )
    return chi2_stat, p_value, reject_null



@DynamicMethod( 
   'categorical',
    capture_columns=False, 
    treat_int_as_categorical=True, 
    encode_categories= True
  )
def anova_test(
    data: Union[Dict[str, List[float]], ArrayLike], 
    groups: Optional[Union[List[str], np.ndarray]]=None, 
    columns: List[str] = None,
    alpha: float = 0.05,
    view: bool = False,
    as_frame=True, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
):
    """
    Perform an Analysis of Variance (ANOVA) test to compare means across 
    multiple groups.

    ANOVA is used to determine whether there are any statistically significant 
    differences between the means of three or more independent (unrelated)
    groups.

    .. math::
        F = \frac{\text{Between-group variability}}{\text{Within-group variability}}

    Parameters
    ----------
    data : dict, np.ndarray, pd.DataFrame
        The input data. Can be a dictionary with group names as keys and lists 
        of values as values, a numpy array if `groups` are specified as indices,
        or a pandas DataFrame with `groups` specifying column names.
    groups : optional, List[str] or np.ndarray
        The names or indices of the groups to compare, extracted from `data`.
        If `data` is a DataFrame and `groups` is not provided, all columns 
        are used.
    columns : List[str], optional
        Specifies the column names for converting `data` from an array-like 
        format to a DataFrame. Note that this parameter does not influence 
        the function's behavior; it is included for API consistency.
    alpha : float, optional
        The significance level for the ANOVA test. Default is 0.05.
    view : bool, optional
        If True, generates a box plot of the group distributions. Default is False.
    as_frame : bool, optional
        If True, returns the result as a pandas Series. Default is True.
    cmap : str, optional
        The colormap for the box plot. Not directly used for box plots in current
        implementations but reserved for future compatibility. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the box plot. Default is (12, 5).

    Returns
    -------
    f_stat : float
        The calculated F-statistic from the ANOVA test.
    p_value : float
        The p-value from the ANOVA test, indicating probability of 
        observing the data if the null hypothesis is true.
    reject_null : bool
        Indicates whether the null hypothesis can be rejected based 
        on the alpha level.

    Examples
    --------
    >>> from scipy import stats
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from gofast.stats.inferential import goanova_test  
    >>> data = {'group1': [1, 2, 3], 'group2': [4, 5, 6], 'group3': [7, 8, 9]}
    >>> f_stat, p_value, reject_null = anova_test(data, alpha=0.05, view=True)
    >>> print(f"F-statistic: {f_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    Notes
    -----
    The F-statistic is calculated as the ratio of the variance between the 
    groups to the variance within the groups. A higher F-statistic indicates a
    greater degree of separation between the group means.
    """

    if isinstance(data, pd.DataFrame):
        if groups:
            groups_data = [data[group].dropna().values for group in groups]
        else:
            groups_data = [data[col].dropna().values for col in data.columns]
    elif isinstance(data, np.ndarray):
        groups_data = [data[group].flatten() for group in groups]  
    else:
        raise ValueError("Unsupported data type for `data` parameter.")

    f_stat, p_value = stats.f_oneway(*groups_data)
    reject_null = p_value < alpha

    if view:
        plt.figure(figsize=fig_size)
        if isinstance(data, (dict, pd.DataFrame)):
            plot_data = pd.DataFrame(
                groups_data, index=groups if groups else data.keys()).T.melt()
            sns.boxplot(x='variable', y='value', data=plot_data, palette=cmap)
        else:
            sns.boxplot(data=np.array(groups_data), palette=cmap)
        plt.title('Group Comparisons via ANOVA')
        plt.show()
        
    if as_frame: 
        return to_series_if(
            f_stat, p_value, reject_null, 
            value_names=['F-statistic', 'P-value', 'Reject-Null-Hypothesis'], 
            name='anova_test')
    return f_stat, p_value, reject_null

@make_data_dynamic("numeric", capture_columns=True)
def bootstrap(
    data: ArrayLike,
    n: int = 1000,
    columns: Optional[List[str]] = None,
    func: Callable | NumPyFunction = np.mean,
    as_frame: bool = True, 
    view: bool = False,
    alpha: float = .7, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (10, 6),
    random_state: Optional[int] = None,
    return_ci: bool = False,
    ci: float = 0.95
) -> Union[Array1D, DataFrame, Tuple[Union[Array1D, DataFrame],
                                     Tuple[float, float]]]:
    """
    Perform bootstrapping to estimate the distribution of a statistic and 
    optionally its confidence interval.
    
    Bootstrapping is a resampling technique used to estimate statistics on a 
    population by sampling a dataset with replacement. This method allows for 
    the estimation of the sampling distribution of almost any statistic.

    Given a dataset :math:`D` of size :math:`N`, bootstrapping generates 
    :math:`B` new datasets :math:`\{D_1, D_2, \ldots, D_B\}`, each of size
    :math:`N` by sampling with replacement from :math:`D`. A statistic :math:`T` 
    is computed on each of these bootstrapped datasets.

    Parameters
    ----------
    data : DataFrame or array_like
        The data to bootstrap. If a DataFrame is provided and `columns` is 
        specified, only the selected columns are used.
    n : int, optional
        Number of bootstrap samples to generate, default is 1000.
    columns : List[str], optional
        Specific columns to use if `data` is a DataFrame.
    func : callable, optional
        The statistic to compute from the resampled data, default is np.mean.
    as_frame : bool, optional
        If True, returns results in a pandas DataFrame. Default is True.
    view : bool, optional
        If True, displays a histogram of the bootstrapped statistics. 
        Default is True.
    alpha : float, optional
        Transparency level of the histogram bars. Default is 0.7.
    cmap : str, optional
        Colormap for the histogram. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the histogram. Default is (10, 6).
    random_state : int, optional
        Seed for the random number generator for reproducibility. 
        Default is None.
    return_ci : bool, optional
        If True, returns a tuple with bootstrapped statistics and their 
        confidence interval. Default is False.
    ci : float, optional
        The confidence level for the interval. Default is 0.95.

    Returns
    -------
    bootstrapped_stats : ndarray or DataFrame
        Array or DataFrame of bootstrapped statistic values. If `return_ci` 
        is True, also returns a tuple containing
        the lower and upper bounds of the confidence interval.

    Examples
    --------
    >>> from gofast.stats.inferential import bootstrap
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> data = np.arange(10)
    >>> stats = bootstrap(data, n=100, func=np.mean)
    >>> print(stats[:5])

    Using a DataFrame, returning confidence intervals:
    >>> df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)})
    >>> stats, ci = bootstrap(df, n=1000, func=np.median, columns=['A'],
                              view=True, return_ci=True, ci=0.95)
    >>> print(f"Median CI: {ci}")
    """
    if random_state is not None:
        np.random.seed(random_state)
    data = data.to_numpy().flatten()

    bootstrapped_stats = [
        func(np.random.choice(data, size=len(data), replace=True)
             ) for _ in range(n)]

    if view:
        colors, alphas = get_colors_and_alphas(
            bootstrapped_stats, cmap, convert_to_named_color=True)
        plt.figure(figsize=fig_size)
        plt.hist(bootstrapped_stats, bins='auto', color=colors[0],
                 alpha=alpha, rwidth=0.85)
        plt.title('Distribution of Bootstrapped Statistics')
        plt.xlabel('Statistic Value')
        plt.ylabel('Frequency')
        plt.show()

    if return_ci:
        lower_bound = np.percentile(bootstrapped_stats, (1 - ci) / 2 * 100)
        upper_bound = np.percentile(bootstrapped_stats, (1 + ci) / 2 * 100)
        result = (bootstrapped_stats, (lower_bound, upper_bound))
    else:
        result = bootstrapped_stats

    if as_frame:
        return convert_and_format_data( 
            result if not return_ci else result[0], return_df=True,
            series_name="bootstrap_stats"
            ) if as_frame else np.array(bootstrapped_stats) 
    
    return result

@ensure_pkg(
    "statsmodels", 
    extra="'rm_anova' and 'mcnemar' tests expect statsmodels' to be installed.",
    partial_check=True,
    condition= lambda *args, **kwargs: kwargs.get("test_type") in [
        "rm_anova", "mcnemar"]
    )
def statistical_tests(
    *args, 
    test_type="mcnemar", 
    data: Optional[DataFrame]=None, 
    error_type: str = 'ci', 
    confidence_interval: float = 0.95, 
    error_bars: bool = True, 
    annot: bool = True, 
    depvar:str=None, 
    subject:str=None, 
    within: List[str] =None, 
    showmeans: bool = True, 
    split: bool = True, 
    trend_line: bool = True, 
    density_overlay: bool = False,
    as_frame: bool=False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kwargs
   ):
    """
    Perform a variety of statistical tests to analyze data and assess hypotheses.
    
    Function supports both parametric and non-parametric tests, catering
    to datasets with different characteristics and research designs.

    Parameters
    ----------
    *args : sequence of array-like or DataFrame
        Input data for performing the statistical test. Each array-like object
        represents a group or condition in the analysis. When `data` is a 
        DataFrame,`args` should be the names of columns in the DataFrame that 
        represent the groups or conditions to be analyzed. This flexible input
        format allows for easy integration of the function within data analysis
        workflows.
    
    test_type : str, optional
        The specific statistical test to be performed. This parameter determines
        the statistical methodology and assumptions underlying the analysis. 
        Supported tests and their applications are as follows:
            
        - ``rm_anova``: Repeated Measures ANOVA, for comparing means across more 
          than two related groups over time or in different conditions.
        - ``cochran_q``: Cochran’s Q Test, for comparing binary outcomes across 
          more than two related groups.
        - ``mcnemar``: McNemar’s Test, for comparing binary outcomes in paired 
          samples.
        - ``kruskal_wallis``: Kruskal-Wallis H Test, a non-parametric test for 
          comparing more than two independent groups.
        - ``wilcoxon``: Wilcoxon Signed-Rank Test, a non-parametric test for 
          comparing paired samples.
        - ``ttest_paired``: Paired t-Test, for comparing means of paired samples.
        - ``ttest_indep``: Independent t-Test, for comparing means of two 
          independent groups.
          
        The default test is ``mcnemar``, which is suitable for categorical data
        analysis.
    
    data : DataFrame, optional
        A pandas DataFrame containing the dataset if column names are specified 
        in `args`. This parameter allows the function to directly interface with
        DataFrame structures, facilitating the extraction and manipulation of 
        specific columns for analysis. If `data` is provided, `args` should 
        correspond to column names within this DataFrame.
        It must not be None when `test_type` is set to ``rm_anova``.
        
    depvar : str
        The name of the dependent variable within the dataset. This variable 
        is what you are trying to predict or explain, and is the main focus 
        of the ANOVA test. It should be numeric and typically represents the 
        outcome or measure that varies across different conditions or groups.
        It must not be None when `test_type` is set to ``rm_anova``.
        
    subject : str
        The name of the variable in the dataset that identifies the subject or
        participant. This variable is used to indicate which observations 
        belong to the same subject, as repeated measures ANOVA assumes multiple
        measurements are taken from the same subjects. Identifying subjects 
        allows the model to account for intra-subject variability, treating it 
        as a random effect.
        It must not be None when `test_type` is set to ``rm_anova``.
        
    within : list of str
        A list of strings where each string is the name of a within-subject 
        factor in the dataset. Within-subject factors are conditions or groups 
        that all subjects are exposed to, allowing the analysis to examine the
        effects of these factors on the dependent variable. Each factor 
        must have two or more levels (e.g., pre-test and post-test), and the 
        analysis will assess how the dependent variable changes in relation to 
        these levels, taking into account the  repeated measures nature of the data.
        
    as_frame: bool, optional 
        Returns a pandas Series or DataFrame based on number of items that 
        may compose the colums. 
       
    view : bool, optional
        Controls the generation of visualizations for the data distributions 
        or test results. If set to ``True``, the function will produce plots 
        that offer graphical representations of the analysis, enhancing 
        interpretability and insight into the data. Default is ``False``.
    
    cmap : str, optional
        Specifies the colormap to be used in the visualizations. This parameter 
        allows for customization of the plot aesthetics, providing flexibility 
        in matching visualizations to the overall theme or style of the analysis.
        Default colormap is ``viridis``.
    
    fig_size : tuple, optional
        Determines the size of the figure for the generated visualizations. 
        This tuple should contain two values representing the width and height 
        of the figure. Specifying `fig_size` allows for control over the 
        appearance of the plots, ensuring that they are appropriately sized for
        the context in which they are presented. Default is None, which will use
        matplotlib's default figure size.
    
    **kwargs : dict
        Additional keyword arguments that are specific to the chosen statistical
        test. These arguments allow for fine-tuning of the test parameters 
        and control over aspects of the analysis that are unique to each 
        statistical method. The availability and effect of these
        parameters vary depending on the `test_type` selected.

    Returns
    -------
    result : Result object
        The result of the statistical test, including the test statistic and the
        p-value. The exact structure of this result object may vary depending on the
        specific test performed, but it generally provides key information needed
        for interpretation of the test outcomes.
    
    Test Details
    ------------
    - Repeated Measures ANOVA ('rm_anova'):
        Used for comparing the means of three or more groups on the same subjects,
        commonly in experiments where subjects undergo multiple treatments. The 
        test statistic is calculated based on the within-subject variability and
        between-group differences [1]_.
        
        .. math::
            F = \\frac{MS_{between}}{MS_{within}}
    
    - Cochran’s Q Test ('cochran_q'):
        A non-parametric test for comparing three or more matched groups on binary
        outcomes. It extends McNemar's test for situations with more than two 
        related groups.
        
        .. math::
            Q = \\frac{12}{nk(k-1)} \\sum_{j=1}^{k} (T_j - \\bar{T})^2
    
    - McNemar’s Test ('mcnemar'):
        Used for binary classification to compare the proportion of misclassified
        instances between two models on the same dataset [2]_.
        
        .. math::
            b + c - |b - c| \\over 2
    
    - Kruskal-Wallis H Test ('kruskal_wallis'):
        A non-parametric version of ANOVA for comparing two or more independent
        groups. Suitable for data that do not meet the assumptions of normality
        required for ANOVA [3]_.
        
        .. math::
            H = \\frac{12}{N(N+1)} \\sum_{i=1}^{k} \\frac{R_i^2}{n_i} - 3(N+1)
    
    - Wilcoxon Signed-Rank Test ('wilcoxon'):
        A non-parametric test to compare two related samples, used when the
        population cannot be assumed to be normally distributed [4]_.
        
        .. math::
            W = \\sum_{i=1}^{n} rank(|x_i - y_i|) \\cdot sign(x_i - y_i)
    
    - Paired t-Test ('ttest_paired'):
        Compares the means of two related groups, such as in before-and-after
        studies, using the same subjects in both groups.
        
        .. math::
            t = \\frac{\\bar{d}}{s_d / \\sqrt{n}}
    
    - Independent t-Test ('ttest_indep'):
        Compares the means of two independent groups, used when different subjects
        are in each group or condition.
        
        .. math::
            t = \\frac{\\bar{X}_1 - \\bar{X}_2}{\\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}}
    
    Notes:
    - The formulas provided are simplified representations of the test statistics
      used in each respective test. They serve as a conceptual guide to understanding
      the mathematical foundations of the tests.
    - The specific assumptions and conditions under which each test is appropriate
      should be carefully considered when interpreting the results.

    Examples
    --------
    Using the function for a paired t-test:
    
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.inferential import statistical_tests
    >>> data1 = np.random.normal(loc=10, scale=2, size=30)
    >>> data2 = np.random.normal(loc=12, scale=2, size=30)
    >>> result = statistical_tests(data1, data2, test_type='ttest_paired')
    >>> print(result)
    
    Performing a Kruskal-Wallis H Test with DataFrame input:
    
    >>> df = pd.DataFrame({'group1': np.random.normal(10, 2, 30),
    ...                       'group2': np.random.normal(12, 2, 30),
    ...                       'group3': np.random.normal(11, 2, 30)})
    >>> result = statistical_tests(df, test_type='kruskal_wallis', 
                                   columns=['group1', 'group2', 'group3'])
    >>> print(result)
    # Sample dataset
    >>> data = {
    ...     'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'score': [5, 3, 8, 4, 6, 7, 6, 5, 8],
    ...     'time': ['pre', 'mid', 'post', 'pre', 'mid', 'post', 'pre', 'mid', 'post'],
    ...     'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    ... }
    >>> df = pd.DataFrame(data)
    # Perform repeated measures ANOVA
    >>> result = statistical_tests(df, depvar='score', subject='subject_id',
    ...                 within=['time', 'treatment'], test_type ='rm_anova')
    # Display the ANOVA table
    >>> print(result)

    Notes
    -----
    - The `rm_anova` and `mcnemar` tests require the `statsmodels` package.
    - Visualization is supported for all tests but is particularly informative
      for distribution-based tests like the Kruskal-Wallis H Test and Wilcoxon 
      Signed-Rank Test.
    
    The choice of test depends on the research question, data characteristics, 
    and assumptions about the data. It is crucial to select the appropriate test 
    to ensure valid and reliable results.
    
    See Also
    --------
    - `scipy.stats.ttest_rel` : For details on the paired t-test.
    - `scipy.stats.kruskal` : For details on the Kruskal-Wallis H Test.
    - `statsmodels.stats.anova.AnovaRM` : For details on Repeated Measures ANOVA.
    - `statsmodels.stats.contingency_tables.mcnemar` : For details on McNemar's Test.
    
    References
    ----------
    .. [1] Friedman, M. (1937). The use of ranks to avoid the assumption of 
           normality implicit in the analysis of variance. 
           *Journal of the American Statistical Association*, 32(200), 675-701.
    .. [2] McNemar, Q. (1947). Note on the sampling error of the difference
          between correlated proportions or percentages. *Psychometrika*, 
          12(2), 153-157.
    .. [3] Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion 
           variance analysis. *Journal of the American Statistical Association*,
           47(260), 583-621.
    .. [4] Wilcoxon, F. (1945). Individual comparisons by ranking methods. 
          *Biometrics Bulletin*, 1(6), 80-83.
    """
    available_tests = ["cochran_q", "kruskal_wallis", "wilcoxon", "ttest_paired", 
        "ttest_indep", "rm_anova", "mcnemar"]
    error_msg = ( 
        f"Invalid test type '{test_type}'. Supported tests"
        f" are: {smart_format(available_tests)}"
        )
    test_type= normalize_string(test_type, target_strs= available_tests, 
                                match_method='contains', return_target_only=True, 
                                raise_exception=True, error_msg= error_msg,
                                deep=True)
    
    # Define test functions
    test_functions = {
        'cochran_q': lambda: stats.cochran.q(*args, **kwargs),
        'kruskal_wallis': lambda: stats.kruskal(*args),
        'wilcoxon': lambda: stats.wilcoxon(*args),
        'ttest_paired': lambda: stats.ttest_rel(*args),
        'ttest_indep': lambda: stats.ttest_ind(*args)
    }
    if isinstance (data, pd.DataFrame): 
        if all ([ isinstance (arg, str) for arg in args]): 
            # use the args as columns of the dataframe and 
            # extract arrays from this datasets 
            args = process_and_extract_data(
                *[data], columns =args, allow_split= True ) 
        else: 
            raise
            
    if test_type in ["rm_anova", "mcnemar"]: 
        from statsmodels.stats.anova import AnovaRM
        from statsmodels.stats.contingency_tables import mcnemar
        if test_type =='rm_anova': 
           test_result= AnovaRM(*args, depvar=depvar, subject=subject, 
                                within=within, **kwargs).fit() 
        elif test_type =="mcnemar": 
            test_result=mcnemar(*args, **kwargs)
    else: 
        # Execute the specified test                       
        try:
            test_result = test_functions[test_type]()
        except KeyError:
            raise ValueError(f"Invalid test type '{test_type}' specified.")
        except Exception as e : 
            raise e 
            
    # Visualization part
    if view:
        plt.figure(figsize=fig_size if fig_size else (10, 6))
        if test_type == 'mcnemar':
            sns.heatmap(test_result.table, annot=annot, cmap=cmap)
            plt.title('McNemar\'s Test Contingency Table')

        elif test_type == 'kruskal_wallis':
            sns.boxplot(data=np.array(args).T, showmeans=showmeans)
            plt.title('Kruskal-Wallis H-test')

        elif test_type == 'wilcoxon':
            if split:
                data = np.array(args[0]) - np.array(args[1])
                sns.violinplot(data=data, split=split)
            plt.title('Wilcoxon Signed-Rank Test')

        elif test_type == 'ttest_paired':
            x, y = args[0], args[1]
            plt.scatter(x, y)
            if trend_line:
                sns.regplot(x=np.array(x), y=np.array(y), ci=confidence_interval)
            plt.title('Paired T-test')

        elif test_type == 'ttest_indep':
            x, y = args[0], args[1]
            sns.histplot(x, kde=density_overlay, color='blue', 
                         label='Group 1', alpha=0.6)
            sns.histplot(y, kde=density_overlay, color='red', 
                         label='Group 2', alpha=0.6)
            plt.legend()
            plt.title('Independent Samples T-test')

        elif test_type == 'rm_anova':
            # Assuming the input data structure is suitable 
            # for this visualization
            sns.lineplot(data=np.array(args[0]),
                         err_style=error_type, 
                         ci=confidence_interval)
            plt.title('Repeated Measures ANOVA')

        elif test_type == 'cochran_q':
            # Assuming args[0] is a binary matrix or similar structure
            positive_responses = np.sum(args[0], axis=0)
            sns.barplot(x=np.arange(len(positive_responses)), 
                        y=positive_responses, 
                        ci=confidence_interval)
            plt.title('Cochran\'s Q Test')

        plt.show()
   
    return _extract_statistical_test_results (test_result, as_frame )

def _extract_statistical_test_results(
        test_result_object, return_as_frame):
    """
    Extracts statistical test results, including the statistic and p-value,
    from a given object.
    
    Parameters
    ----------
    test_result_object : object
        The object containing the statistical test results. It must 
        have attributes for the statistic value (`statistic` or similar) 
        and the p-value (`p_value` or `pvalue`).
    return_as_frame : bool
        Determines whether to return the results as a pandas DataFrame. 
        If False, the results are returned as a tuple.
    
    Returns
    -------
    tuple or pandas.DataFrame
        The statistical test results. If `return_as_frame` is True, returns
        a pandas DataFrame with columns ["Statistic", "P-value"]. Otherwise, 
        returns a tuple containing the statistic and p-value.
    
    Examples
    --------
    >>> from scipy.stats import ttest_1samp
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, size=100)
    >>> test_result = ttest_1samp(data, 0)
    >>> extract_statistical_test_results(test_result, False)
    (statistic_value, p_value)
    
    >>> extract_statistical_test_results(test_result, True)
        Statistic    P-value
    0   statistic_value p_value
    """
    statistic = None
    p_value = None
    # Extract statistic and p-value from the test_result_object
    if hasattr(test_result_object, "statistic"):
        statistic = test_result_object.statistic
    if hasattr(test_result_object, "p_value"):
        p_value = test_result_object.p_value
    elif hasattr(test_result_object, 'pvalue'):
        p_value = test_result_object.pvalue
    
    # Determine the name based on object class or a custom name attribute
    name = getattr(test_result_object, '__class__', None).__name__.lower(
        ) if hasattr(test_result_object, '__class__') else getattr(
            test_result_object, 'name', '').lower()
    name = name.replace("result", "").replace("test", "") + '_test'
    
    if statistic is not None and p_value is not None:
        test_results = (statistic, p_value)
    else:
        test_results = (test_result_object,)
    
    # Convert to pandas DataFrame if requested
    if return_as_frame:
        test_results_df = pd.DataFrame([test_results], columns=["Statistic", "P-value"])
        test_results_df.rename(index={0: name}, inplace=True)
        return test_results_df
    else:
        return test_results

def check_and_fix_rm_anova_data(
    data: DataFrame, 
    depvar: str, 
    subject: str, 
    within: List[str], 
    fix_issues: bool = False,
    strategy: str ="mean", 
    fill_value: Optional[Union[str, float, int]]=None, 
) -> DataFrame:
    """
    Checks and optionally fixes a DataFrame for repeated measures ANOVA analysis.

    This function verifies if each subject in the dataset has measurements for every 
    combination of within-subject factors. If `fix_issues` is set to True, the dataset 
    will be modified to include missing combinations, assigning `None` to the dependent 
    variable values of these new rows.

    Parameters
    ----------
    data : DataFrame
        The pandas DataFrame containing the data for ANOVA analysis.
    depvar : str
        The name of the dependent variable column in `data`.
    subject : str
        The name of the column identifying subjects in `data`.
    within : List[str]
        A list of column names representing within-subject factors.
    fix_issues : bool, optional
        If True, the dataset will be altered to include missing combinations 
        of within-subject factors for each subject. Default is False.
     strategy : str, optional
         The strategy to use for filling missing depvar values. Options are "mean",
         "median", or None. Default is "mean".
     fill_value : Optional[Union[str, float, int]], optional
         A specific value to fill missing depvar values if the strategy is None.
         Default is None, which leaves missing values as None.
    Returns
    -------
    DataFrame
        The original `data` DataFrame if `fix_issues` is False or no issues are found. 
        A modified DataFrame with issues fixed if `fix_issues` is True.

    Raises
    ------
    TypeError
        If input types for `data`, `depvar`, `subject`, or `within` are incorrect.
    ValueError
        If columns specified by `depvar`, `subject`, or `within` do not exist in `data`.

    Notes
    -----
    The mathematical formulation for identifying missing combinations involves 
    creating a Cartesian product of unique values within each within-subject 
    factor and then verifying these combinations against the existing dataset. 
    
    .. math::
    
        S = \\{s_1, s_2, ..., s_n\\} \quad \\text{(Subjects)}
        
        W_i = \\{w_{i1}, w_{i2}, ..., w_{im}\\} \quad \\text{(Within-subject factors for } i \\text{th factor)}
        
        C = W_1 \\times W_2 \\times ... \\times W_k \quad \\text{(Cartesian product of all within-subject factors)}
        
        \\text{For each subject } s \\text{ in } S, \\text{ verify } (s, c) \\in D \\text{ for all } c \\in C
        
    If combinations are missing, new rows are appended to the dataset to include 
    these missing combinations, ensuring that every subject has measurements 
    across all factor levels.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.stats.inferential import check_and_fix_rm_anova_data
    >>> data = {
    ...     'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'score': [5, 3, 8, 4, 6, 7, 6, 5, 8],
    ...     'time': ['pre', 'mid', 'post', 'pre', 'mid', 'post', 'pre', 'mid', 'post'],
    ...     'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    ... }
    >>> df = pd.DataFrame(data)
    >>> fixed_df = check_and_fix_rm_anova_data(
    ...     df, depvar='score', subject='subject_id', within=['time', 'treatment'],
    ...     fix_issues=True)
    >>> fixed_df
    """
    # Validate input types
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected 'data' to be a DataFrame,"
                        f" but got {type(data).__name__}.")
    
    if not isinstance(depvar, str):
        raise TypeError("'depvar' should be a string,"
                        f" but got {type(depvar).__name__}.")
    
    if not isinstance(subject, str):
        raise TypeError("'subject' should be a string,"
                        f" but got {type(subject).__name__}.")
    
    if not check_uniform_type (within):
        raise TypeError("All items in 'within' should be strings.")

    # Check for necessary columns in the DataFrame
    missing_columns = [col for col in [depvar, subject] + within if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}.")

    # Check combinations
    combinations_df = data.groupby([subject] + within).size().reset_index(name='counts')
    expected_combinations = len(data[within[0]].unique()) * len(data[within[1]].unique())
    subject_combination_counts = combinations_df.groupby(subject).size()
    
    missing_combinations_subjects = subject_combination_counts[
        subject_combination_counts < expected_combinations]
    
    if missing_combinations_subjects.empty:
        print("All subjects have measurements for every combination of within-subject factors.")
    else:
        missing_info = ", ".join(map(str, missing_combinations_subjects.index.tolist()))
        print(f"Subjects with missing combinations: {missing_info}")
        
        if fix_issues:
            fixed_data = fix_rm_anova_dataset(data, depvar, subject, within)
            print("Dataset issues fixed.")
            return fixed_data
    
    return data

