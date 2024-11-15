.. _inferential:

Inferential Statistics
=====================

.. currentmodule:: gofast.stats.inferential

The :mod:`gofast.stats.inferential` module provides a comprehensive suite of statistical testing and inference tools. This module implements both parametric and non-parametric statistical tests, bootstrap methods, and advanced modeling techniques for rigorous statistical analysis.

Key Features
------------
- **Parametric Tests**: Suite of classical statistical tests including t-tests, ANOVA, and chi-square tests
- **Non-parametric Tests**: Robust alternatives including Wilcoxon, Kruskal-Wallis, and Friedman tests
- **Bootstrap Methods**: Resampling techniques for parameter estimation and hypothesis testing
- **Mixed Effects Models**: Tools for analyzing hierarchical and longitudinal data
- **Test Selection Assistance**: Automated test selection based on data characteristics
- **Comprehensive Reporting**: Detailed statistical reports with effect sizes and confidence intervals

Function Descriptions
--------------------

t_test_independent
~~~~~~~~~~~~~~~~~
Performs independent samples t-test to compare means between two groups.

Parameters:
    - group1 (ArrayLike): First group's data
    - group2 (ArrayLike): Second group's data
    - equal_var (bool): Assume equal variances (default: True)
    - alternative (str): Alternative hypothesis ('two-sided', 'less', 'greater')

Returns:
    - Tuple[float, float]: T-statistic and p-value

.. math::

    t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}

where :math:`s_p` is the pooled standard deviation.

Examples:

.. code-block:: python

    from gofast.stats.inferential import t_test_independent
    import numpy as np

    # Generate sample data
    group1 = np.random.normal(0, 1, 30)
    group2 = np.random.normal(0.5, 1, 30)
    
    # Perform t-test
    t_stat, p_value = t_test_independent(group1, group2)
    print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

paired_t_test
~~~~~~~~~~~~
Conducts paired samples t-test for dependent samples.

Parameters:
    - x (ArrayLike): First measurement
    - y (ArrayLike): Second measurement
    - alternative (str): Alternative hypothesis

Returns:
    - Tuple[float, float]: T-statistic and p-value

.. math::

    t = \frac{\bar{d}}{s_d/\sqrt{n}}

where :math:`\bar{d}` is the mean difference and :math:`s_d` is the standard deviation of differences.

anova_test
~~~~~~~~~
Performs one-way ANOVA test for comparing means across multiple groups.

Parameters:
    - *args: Variable number of sample groups
    - data (Optional[DataFrame]): DataFrame containing the data
    - between (Optional[str]): Between-subjects factor
    - within (Optional[str]): Within-subjects factor

Returns:
    - Tuple[float, float]: F-statistic and p-value

.. math::

    F = \frac{MS_{between}}{MS_{within}}

chi2_test
~~~~~~~~
Performs chi-square test of independence for categorical variables.

Parameters:
    - observed: Observed frequencies
    - expected (Optional): Expected frequencies
    - correction (bool): Apply Yates' correction

Returns:
    - Tuple[float, float]: Chi-square statistic and p-value

.. math::

    \chi^2 = \sum\frac{(O-E)^2}{E}

bootstrap
~~~~~~~~
Implements bootstrap resampling for parameter estimation.

Parameters:
    - data (ArrayLike): Input data
    - n_bootstraps (int): Number of bootstrap samples
    - statistic (Callable): Statistic to compute
    - ci_level (float): Confidence interval level

Returns:
    - BootstrapResult: Bootstrap estimates and confidence intervals

Examples:

.. code-block:: python

    from gofast.stats.inferential import bootstrap
    
    # Generate sample data
    data = np.random.normal(0, 1, 100)
    
    # Perform bootstrap for mean estimation
    result = bootstrap(data, n_bootstraps=1000, statistic=np.mean)
    print(f"Bootstrap estimate: {result.estimate:.4f}")
    print(f"95% CI: ({result.ci_lower:.4f}, {result.ci_upper:.4f})")

levene_test
~~~~~~~~~~
Performs Levene's test for homogeneity of variances.

Parameters:
    - *args: Variable number of sample groups
    - center (str): Which center measure to use ('mean', 'median', 'trimmed')

Returns:
    - Tuple[float, float]: Test statistic and p-value

kolmogorov_smirnov_test
~~~~~~~~~~~~~~~~~~~~~~
Performs Kolmogorov-Smirnov test for distribution comparison.

Parameters:
    - data1 (ArrayLike): First sample
    - data2 (ArrayLike): Second sample or theoretical distribution
    - alternative (str): Alternative hypothesis

Returns:
    - Tuple[float, float]: KS statistic and p-value

friedman_test
~~~~~~~~~~~~
Conducts Friedman test for repeated measures analysis.

Parameters:
    - *args: Variable number of related samples
    - data (Optional[DataFrame]): DataFrame containing the data

Returns:
    - Tuple[float, float]: Chi-square statistic and p-value

.. math::

    \chi^2_r = \frac{12}{bk(k+1)}\sum_{j=1}^k R_j^2 - 3b(k+1)

where b is the number of blocks and k is the number of treatments.

wilcoxon_signed_rank_test
~~~~~~~~~~~~~~~~~~~~~~~
Performs Wilcoxon signed-rank test for paired samples.

Parameters:
    - x (ArrayLike): First sample
    - y (ArrayLike): Second sample
    - alternative (str): Alternative hypothesis

Returns:
    - Tuple[float, float]: Test statistic and p-value

kruskal_wallis_test
~~~~~~~~~~~~~~~~~
Conducts Kruskal-Wallis H-test for independent samples.

Parameters:
    - *args: Variable number of sample groups
    - data (Optional[DataFrame]): DataFrame containing the data

Returns:
    - Tuple[float, float]: H-statistic and p-value

mcnemar_test
~~~~~~~~~~~
Performs McNemar's test for paired nominal data.

Parameters:
    - x (ArrayLike): First sample
    - y (ArrayLike): Second sample
    - correction (bool): Apply continuity correction

Returns:
    - Tuple[float, float]: Test statistic and p-value

mixed_effects_model
~~~~~~~~~~~~~~~~~
Fits a linear mixed-effects model for hierarchical data.

Parameters:
    - formula (str): Model formula
    - data (DataFrame): Input data
    - groups (str): Grouping variable
    - random_effects (List[str]): Random effects specification

Returns:
    - MixedModelResult: Fitted model results

Examples:

.. code-block:: python

    from gofast.stats.inferential import mixed_effects_model
    import pandas as pd
    
    # Create sample data
    data = pd.DataFrame({
        'subject': np.repeat(range(20), 5),
        'time': np.tile(range(5), 20),
        'treatment': np.random.choice(['A', 'B'], 100),
        'response': np.random.normal(0, 1, 100)
    })
    
    # Fit mixed-effects model
    model = mixed_effects_model(
        'response ~ treatment + time',
        data=data,
        groups='subject',
        random_effects=['time']
    )
    print(model.summary())

statistical_tests
~~~~~~~~~~~~~~~
Provides a unified interface for performing various statistical tests.

Parameters:
    - data: Input data
    - test_type (str): Type of test to perform
    - **kwargs: Additional test-specific parameters

Returns:
    - StatisticalTestResult: Test results and diagnostics

See Also
--------
- :mod:`gofast.stats.descriptive`: For descriptive statistics calculations
- :mod:`gofast.stats.model_comparisons`: For model comparison and selection
- :mod:`gofast.stats.relationships`: For correlation and regression analysis