.. _relationships:

Statistical Relationships
========================

.. currentmodule:: gofast.stats.relationships

The :mod:`gofast.stats.relationships` module provides sophisticated tools for analyzing relationships between variables. This module implements various correlation measures, regression techniques, and advanced methods for exploring variable associations and dependencies.

Key Features
------------
- **Correlation Analysis**:
  Comprehensive suite of correlation measures and tests.

  - :func:`~gofast.stats.relationships.correlation`: Multiple correlation coefficient calculations
  - :func:`~gofast.stats.relationships.partial_correlation`: Partial correlation analysis
  - :func:`~gofast.stats.relationships.rank_correlation`: Non-parametric correlation measures

- **Regression Analysis**:
  Advanced regression modeling and diagnostics.

  - :func:`~gofast.stats.relationships.perform_linear_regression`: Linear regression with comprehensive diagnostics
  - :func:`~gofast.stats.relationships.robust_regression`: Robust regression methods
  - :func:`~gofast.stats.relationships.polynomial_regression`: Polynomial regression analysis

- **Clustering and Similarity**:
  Tools for measuring similarity and grouping related variables.

  - :func:`~gofast.stats.relationships.perform_kmeans`: K-means clustering implementation
  - :func:`~gofast.stats.relationships.spectral_clustering`: Spectral clustering analysis
  - :func:`~gofast.stats.relationships.hierarchical_clustering`: Hierarchical clustering methods

- **Association Measures**:
  Methods for quantifying variable associations.

  - :func:`~gofast.stats.relationships.mutual_information`: Information-theoretic association measure
  - :func:`~gofast.stats.relationships.chi_square_test`: Chi-square test of independence
  - :func:`~gofast.stats.relationships.cramer_v`: Cramér's V association measure

Function Descriptions
--------------------

correlation
~~~~~~~~~~
Calculates correlation coefficients between variables using various methods.

Mathematical Expressions:

Pearson correlation:
.. math::

    r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
    {\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}

Spearman correlation:
.. math::

    \rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}

Parameters:
    - x (array-like): First variable
    - y (array-like): Second variable
    - method (str): Correlation method ('pearson', 'spearman', 'kendall')
    - alpha (float): Significance level for hypothesis testing

Returns:
    - CorrelationResult: Correlation coefficient, p-value, and confidence interval

Examples:

.. code-block:: python

    from gofast.stats.relationships import correlation
    import numpy as np

    # Example 1: Basic correlation analysis
    x = np.random.normal(0, 1, 100)
    y = 2*x + np.random.normal(0, 0.5, 100)
    result = correlation(x, y)
    print(f"Correlation: {result.coefficient:.4f}")
    print(f"P-value: {result.pvalue:.4f}")

    # Example 2: Different correlation methods
    methods = ['pearson', 'spearman', 'kendall']
    for method in methods:
        result = correlation(x, y, method=method)
        print(f"\n{method.capitalize()} correlation:")
        print(f"Coefficient: {result.coefficient:.4f}")
        print(f"95% CI: ({result.ci_lower:.4f}, {result.ci_upper:.4f})")

    # Example 3: Correlation matrix for multiple variables
    data = np.column_stack([x, y, 2*x + y])
    corr_matrix = correlation(data, matrix=True)
    print("\nCorrelation Matrix:")
    print(corr_matrix)

perform_linear_regression
~~~~~~~~~~~~~~~~~~~~~~
Performs linear regression analysis with comprehensive diagnostics.

Mathematical Expression:

.. math::

    y = X\beta + \epsilon, \quad \epsilon \sim N(0, \sigma^2I)

Parameters:
    - X (array-like): Independent variables
    - y (array-like): Dependent variable
    - fit_intercept (bool): Whether to fit intercept
    - diagnostics (bool): Whether to compute regression diagnostics

Returns:
    - RegressionResult: Comprehensive regression results and diagnostics

Examples:

.. code-block:: python

    from gofast.stats.relationships import perform_linear_regression

    # Example 1: Simple linear regression
    X = np.random.normal(0, 1, (100, 1))
    y = 2*X.ravel() + 1 + np.random.normal(0, 0.5, 100)
    result = perform_linear_regression(X, y)
    print("Coefficients:", result.coefficients)
    print("R-squared:", result.r_squared)

    # Example 2: Multiple regression with diagnostics
    X = np.random.normal(0, 1, (100, 3))
    y = np.sum(X, axis=1) + np.random.normal(0, 0.5, 100)
    result = perform_linear_regression(X, y, diagnostics=True)
    print("\nDiagnostics:")
    print("Condition Number:", result.condition_number)
    print("VIF:", result.vif)

    # Example 3: Regression with prediction intervals
    result = perform_linear_regression(X, y, compute_intervals=True)
    print("\nPrediction Intervals:")
    print("95% CI:", result.confidence_intervals)
    print("95% PI:", result.prediction_intervals)

perform_kmeans
~~~~~~~~~~~~
Implements k-means clustering with initialization options and diagnostics.

Mathematical Expression:

.. math::

    J = \sum_{j=1}^k \sum_{i \in C_j} \|x_i - \mu_j\|^2

Parameters:
    - data (array-like): Input data for clustering
    - n_clusters (int): Number of clusters
    - init (str): Initialization method ('k-means++', 'random')
    - n_init (int): Number of initializations

Returns:
    - KMeansResult: Clustering results including labels and metrics

Examples:

.. code-block:: python

    from gofast.stats.relationships import perform_kmeans

    # Example 1: Basic clustering
    data = np.random.randn(100, 2)
    result = perform_kmeans(data, n_clusters=3)
    print("Cluster Labels:", result.labels)
    print("Inertia:", result.inertia)

    # Example 2: Advanced clustering with multiple initializations
    result = perform_kmeans(data, n_clusters=3, n_init=10)
    print("\nBest Inertia:", result.best_inertia)
    print("Iterations:", result.n_iter)

spectral_clustering
~~~~~~~~~~~~~~~~
Performs spectral clustering for complex cluster structures.

Parameters:
    - affinity_matrix (array-like): Similarity matrix
    - n_clusters (int): Number of clusters
    - eigen_solver (str): Method for eigenvalue decomposition

Examples:

.. code-block:: python

    from gofast.stats.relationships import spectral_clustering

    # Example: Spectral clustering on similarity matrix
    affinity = np.random.rand(100, 100)
    affinity = (affinity + affinity.T)/2  # Make symmetric
    labels = spectral_clustering(affinity, n_clusters=3)
    print("Cluster Assignments:", np.unique(labels, return_counts=True))

mutual_information
~~~~~~~~~~~~~~~
Calculates mutual information between variables.

Mathematical Expression:

.. math::

    I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)

Parameters:
    - x (array-like): First variable
    - y (array-like): Second variable
    - normalize (bool): Whether to normalize the result

Examples:

.. code-block:: python

    from gofast.stats.relationships import mutual_information

    # Example: Calculate mutual information
    x = np.random.normal(0, 1, 1000)
    y = x + np.random.normal(0, 0.5, 1000)
    mi = mutual_information(x, y, normalize=True)
    print("Normalized Mutual Information:", mi)

partial_correlation
~~~~~~~~~~~~~~~~
Computes partial correlation controlling for confounding variables.

Mathematical Expression:

.. math::

    \rho_{xy|z} = \frac{\rho_{xy} - \rho_{xz}\rho_{yz}}
    {\sqrt{(1-\rho_{xz}^2)(1-\rho_{yz}^2)}}

Parameters:
    - data (DataFrame): Input data
    - x (str): First variable name
    - y (str): Second variable name
    - controlling (list): Variables to control for

Examples:

.. code-block:: python

    from gofast.stats.relationships import partial_correlation
    import pandas as pd

    # Example: Partial correlation analysis
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'z': np.random.normal(0, 1, 100)
    })
    result = partial_correlation(data, 'x', 'y', ['z'])
    print("Partial Correlation:", result.coefficient)
    print("P-value:", result.pvalue)

Best Practices
-------------
1. **Correlation Analysis**:
   - Check assumptions of normality for Pearson correlation
   - Use non-parametric methods for non-normal data
   - Consider partial correlation for controlling confounders

2. **Regression Analysis**:
   - Check for multicollinearity using VIF
   - Examine residual plots for heteroscedasticity
   - Use robust methods for outlier-contaminated data

3. **Clustering**:
   - Validate number of clusters using silhouette analysis
   - Try multiple initializations for k-means
   - Consider spectral clustering for complex structures

See Also
--------
- :mod:`gofast.stats.inferential`: For statistical inference methods
- :mod:`gofast.stats.model_comparisons`: For model comparison tools
- :mod:`gofast.visualization`: For plotting relationships

References
----------
.. [1] Cohen, J., et al. (2003). Applied Multiple Regression/Correlation
       Analysis for the Behavioral Sciences.

.. [2] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements
       of Statistical Learning.

.. [3] von Luxburg, U. (2007). A Tutorial on Spectral Clustering.
       Statistics and Computing.