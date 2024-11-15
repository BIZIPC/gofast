.. _stats:

*********
Stats
*********

.. currentmodule:: gofast.stats

The :mod:`gofast.stats` module provides a comprehensive suite of statistical functions and tools tailored for high-performance data analysis. These submodules offer a broad range of functionalities from basic descriptive statistics to advanced probabilistic modeling, designed for efficient and accurate statistical computations.

Submodules
----------

- :mod:`~gofast.stats.descriptive`: Provides fundamental statistical measures including:
    - Central tendency (mean, median, mode)
    - Variability measures (variance, standard deviation, range)
    - Distribution properties (quartiles, IQR)
    - Shape statistics (skewness, kurtosis)
    - Summary statistics with the `describe` function

- :mod:`~gofast.stats.inferential`: Comprehensive tools for statistical testing:
    - T-tests for independent samples
    - Chi-square tests for independence
    - ANOVA testing for group differences
    - Levene's test for variance homogeneity
    - Kolmogorov-Smirnov tests for distribution fitting
    - Friedman test for repeated measures
    - Bootstrap methods for parameter estimation

- :mod:`~gofast.stats.model_comparisons`: Advanced tools for statistical model evaluation:
    - Model performance metrics
    - Cross-validation utilities
    - Information criteria (AIC, BIC)
    - Model selection frameworks
    - Ensemble model comparison tools
    - DCA (Decision Curve Analysis)

- :mod:`~gofast.stats.probs`: Probability-related computations including:
    - Normal distribution functions (PDF, CDF)
    - Binomial probability mass functions
    - Poisson distribution utilities
    - Uniform sampling methods
    - Stochastic process modeling
    - Hierarchical probability models

- :mod:`~gofast.stats.relationships`: Tools for analyzing variable associations:
    - Correlation analysis
    - Linear regression modeling
    - Multidimensional scaling
    - Spectral clustering
    - Association measures
    - Dependency testing

- :mod:`~gofast.stats.survival_reliability`: Specialized tools for time-to-event analysis:
    - Kaplan-Meier survival analysis
    - Reliability metrics
    - Hazard modeling
    - Survival curves
    - Time-to-event predictions
    - Censoring handling

- :mod:`~gofast.stats.utils`: General statistical utilities including:
    - Data transformation functions
    - Statistical test helpers
    - Coefficient calculations (e.g., Gini)
    - Reliability measures (e.g., Cronbach's alpha)
    - Sampling utilities
    - Statistical report generation

.. toctree::
   :maxdepth: 2
   :titlesonly:

   descriptive
   inferential
   model_comparisons
   probs
   relationships
   survival_reliability
   utils