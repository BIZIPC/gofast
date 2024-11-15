.. _model_comparisons:

Model Comparisons
===============

.. currentmodule:: gofast.stats.model_comparisons

The :mod:`gofast.stats.model_comparisons` module provides comprehensive tools for statistical comparison and evaluation of model performance. This module implements various statistical tests and visualization methods specifically designed for comparing multiple models or algorithms across different datasets and experimental conditions.

Key Features
------------
- **Statistical Tests**:
  Methods for rigorous statistical comparison of model performance, including parametric and non-parametric tests.

  - :func:`~gofast.stats.model_comparisons.perform_friedman_test`: Non-parametric test for comparing multiple models across datasets.
  - :func:`~gofast.stats.model_comparisons.perform_wilcoxon_test`: Pairwise comparison of model performance.

- **Effect Size Analysis**:
  Tools for quantifying the magnitude and practical significance of performance differences.

  - :func:`~gofast.stats.model_comparisons.calculate_effect_size`: Computes various effect size measures.
  - :func:`~gofast.stats.model_comparisons.analyze_effect_significance`: Evaluates practical significance of differences.

- **Visualization Tools**:
  Advanced plotting functions for comparative analysis and result interpretation.

  - :func:`~gofast.stats.model_comparisons.plot_critical_difference`: Creates critical difference diagrams.
  - :func:`~gofast.stats.model_comparisons.plot_model_comparison`: Generates comparative performance visualizations.

- **Performance Metrics**:
  Comprehensive suite of metrics for model evaluation and comparison.

  - :func:`~gofast.stats.model_comparisons.calculate_ranking_matrix`: Computes model rankings across datasets.
  - :func:`~gofast.stats.model_comparisons.compare_cross_validation`: Analyzes cross-validation results.

Function Descriptions
--------------------

perform_friedman_test
~~~~~~~~~~~~~~~~~~~
Performs the Friedman test to determine if there are statistically significant differences between multiple models. This non-parametric test is particularly useful when comparing multiple models across different datasets.

Mathematical Expression:

.. math::

    \chi^2_F = \frac{12N}{k(k+1)}\left[\sum_{j=1}^k R_j^2 - \frac{k(k+1)^2}{4}\right]

where:
- N is the number of datasets
- k is the number of models
- R_j is the sum of ranks for the j-th model

Parameters:
    - performance_data (DataFrame): Performance metrics for each model
    - alpha (float): Significance level for the test
    - return_rankings (bool): Whether to return average rankings

Returns:
    - FriedmanTestResult: Named tuple containing test statistics and rankings

Examples:

.. code-block:: python

    from gofast.stats.model_comparisons import perform_friedman_test
    import pandas as pd

    # Example 1: Basic Friedman test
    data = pd.DataFrame({
        'Model_A': [0.85, 0.83, 0.82, 0.85, 0.84],
        'Model_B': [0.80, 0.82, 0.81, 0.83, 0.82],
        'Model_C': [0.88, 0.87, 0.86, 0.88, 0.87]
    })
    result = perform_friedman_test(data)
    print(f"Friedman statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.4f}")

    # Example 2: Friedman test with rankings
    result = perform_friedman_test(data, return_rankings=True)
    print("\nAverage Rankings:")
    for model, rank in zip(data.columns, result.rankings):
        print(f"{model}: {rank:.2f}")

    # Example 3: Custom significance level
    result = perform_friedman_test(data, alpha=0.01)
    if result.pvalue < 0.01:
        print("\nSignificant differences detected at α=0.01")

perform_wilcoxon_test
~~~~~~~~~~~~~~~~~~~
Conducts pairwise Wilcoxon signed-rank tests between models, with optional correction for multiple comparisons. This test is particularly useful for detailed pairwise comparisons after a significant Friedman test.

Mathematical Expression:

.. math::

    W = \sum_{i=1}^{N} [sgn(x_{2i} - x_{1i}) \cdot R_i]

where:
- N is the number of pairs
- R_i is the rank of the absolute difference
- sgn is the sign function

Parameters:
    - data (DataFrame): Performance metrics for each model
    - alpha (float): Significance level
    - adjust_method (str): Method for p-value adjustment ('bonferroni', 'holm', 'fdr')

Returns:
    - WilcoxonTestResult: Comprehensive results of pairwise comparisons

Examples:

.. code-block:: python

    from gofast.stats.model_comparisons import perform_wilcoxon_test

    # Example 1: Basic pairwise comparison
    wilcoxon_results = perform_wilcoxon_test(data)
    print("Pairwise Comparisons:")
    print(wilcoxon_results.p_values)

    # Example 2: With Bonferroni correction
    wilcoxon_results = perform_wilcoxon_test(
        data, 
        adjust_method='bonferroni'
    )
    print("\nAdjusted p-values:")
    print(wilcoxon_results.adjusted_p_values)

    # Example 3: Detailed comparison report
    wilcoxon_results = perform_wilcoxon_test(
        data, 
        return_details=True
    )
    print("\nDetailed Statistics:")
    print(wilcoxon_results.statistics)

calculate_effect_size
~~~~~~~~~~~~~~~~~~~
Calculates effect sizes for model comparisons using various metrics. This function helps quantify the practical significance of performance differences between models.

Parameters:
    - data1 (array-like): Performance metrics for first model
    - data2 (array-like): Performance metrics for second model
    - method (str): Effect size measure ('cohen_d', 'hedges_g', 'glass_delta')

Mathematical Expressions:

For Cohen's d:
.. math::

    d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}

For Hedges' g:
.. math::

    g = d \cdot (1 - \frac{3}{4(n_1 + n_2) - 9})

Examples:

.. code-block:: python

    from gofast.stats.model_comparisons import calculate_effect_size
    
    # Example 1: Cohen's d calculation
    effect_size = calculate_effect_size(
        data['Model_A'], 
        data['Model_B'], 
        method='cohen_d'
    )
    print(f"Cohen's d: {effect_size:.4f}")

    # Example 2: Multiple effect sizes
    effect_sizes = {
        method: calculate_effect_size(
            data['Model_A'], 
            data['Model_B'], 
            method=method
        )
        for method in ['cohen_d', 'hedges_g', 'glass_delta']
    }
    print("\nEffect Sizes:")
    for method, size in effect_sizes.items():
        print(f"{method}: {size:.4f}")

plot_critical_difference
~~~~~~~~~~~~~~~~~~~~~~
Creates a critical difference diagram for comparing multiple classifiers, based on their average ranks across multiple datasets.

Parameters:
    - ranks (array-like): Average ranks of classifiers
    - names (list): Names of classifiers
    - cd (float): Critical difference
    - figsize (tuple): Figure dimensions
    - title (str): Plot title

Examples:

.. code-block:: python

    from gofast.stats.model_comparisons import plot_critical_difference

    # Example 1: Basic CD diagram
    ranks = [2.1, 1.5, 2.4]  # Example ranks
    names = ['Model_A', 'Model_B', 'Model_C']
    plot_critical_difference(ranks, names, cd=0.5)

    # Example 2: Customized visualization
    plot_critical_difference(
        ranks, 
        names, 
        cd=0.5,
        figsize=(10, 6),
        title="Model Comparison with Critical Difference"
    )

compare_cross_validation
~~~~~~~~~~~~~~~~~~~~~~
Analyzes and compares cross-validation results across multiple models, providing statistical summaries and visualization options.

Parameters:
    - cv_results (dict): Cross-validation results for each model
    - metric (str): Performance metric to compare
    - alpha (float): Significance level for statistical tests
    - plot (bool): Whether to create visualization

Returns:
    - CVComparisonResult: Statistical summary of comparisons

Examples:

.. code-block:: python

    from gofast.stats.model_comparisons import compare_cross_validation

    # Example 1: Basic CV comparison
    cv_results = {
        'Model_A': {'scores': [0.85, 0.82, 0.86]},
        'Model_B': {'scores': [0.90, 0.88, 0.87]},
        'Model_C': {'scores': [0.78, 0.84, 0.82]}
    }
    comparison = compare_cross_validation(
        cv_results, 
        metric='accuracy'
    )
    print(comparison.summary)

    # Example 2: With visualization
    comparison = compare_cross_validation(
        cv_results,
        metric='accuracy',
        plot=True
    )

    # Example 3: Detailed statistical analysis
    comparison = compare_cross_validation(
        cv_results,
        metric='accuracy',
        detailed_stats=True
    )
    print("\nDetailed Statistics:")
    print(comparison.statistics)

Best Practices
-------------
1. **Choosing Statistical Tests**:
   - Use Friedman test for initial multiple model comparison
   - Follow with pairwise tests (e.g., Wilcoxon) for detailed analysis
   - Consider effect sizes alongside p-values

2. **Multiple Comparison Correction**:
   - Always use correction methods when performing multiple comparisons
   - Choose correction method based on desired control (FWER vs FDR)

3. **Visualization Guidelines**:
   - Use critical difference diagrams for comparing multiple models
   - Include confidence intervals in plots
   - Consider both statistical and practical significance

See Also
--------
- :mod:`gofast.stats.inferential`: For additional statistical tests
- :mod:`gofast.metrics`: For performance metric calculations
- :mod:`gofast.visualization`: For additional plotting utilities

References
----------
.. [1] Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
       Journal of Machine Learning Research, 7, 1-30.

.. [2] García, S., et al. (2010). Advanced nonparametric tests for multiple comparisons in
       the design of experiments in computational intelligence and data mining.
       Information Sciences, 180(10), 2044-2064.