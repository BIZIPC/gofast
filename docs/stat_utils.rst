.. _stats_utils:

Statistical Utilities (stats_utils)
====================================

.. currentmodule:: gofast.stats.utils

The :mod:`gofast.stats.utils` module enhances data analysis and statistical modeling projects 
with a wide array of statistical functions and tests. This module simplifies the application of 
statistical methods, from basic descriptive statistics to complex statistical tests, facilitating 
rigorous data analysis and interpretation.

Key Features
------------

- **Descriptive Statistics**: Functions for computing central tendency, variability, and distribution properties.
- **Statistical Tests**: Comprehensive suite of tests for hypothesis testing, including t-tests, ANOVA, and non-parametric tests.
- **Correlation and Regression**: Tools for examining relationships between variables and for predictive modeling.
- **Clustering and Dimensionality Reduction**: Methods for identifying patterns in data and reducing the number of variables.
- **Survival Analysis**: Utilities for analyzing time-to-event data.
- **Bootstrap Methods**: Techniques for estimating statistical properties through resampling.

Function Descriptions
---------------------
.. _stats_utils:


anova_test
~~~~~~~~~~

Performs a one-way ANOVA test to determine if there are any statistically significant differences 
between the means of three or more independent (unrelated) groups.

.. math::

   F = \frac{\text{Between-group variability}}{\text{Within-group variability}}

where :math:`F` is the calculated F-statistic for the ANOVA test.

.. code-block:: python

    from gofast.stats.utils import anova_test
    group1 = [20, 21, 22, 23, 24]
    group2 = [28, 30, 29, 29, 28]
    group3 = [18, 17, 21, 20, 19]
    print("ANOVA Test Result:", anova_test(group1, group2, group3))
    # Output: ANOVA Test Result: (F-statistic, p-value)

bootstrap
~~~~~~~~~

Performs bootstrap resampling on a dataset, generating a specified number of bootstrap samples. 
This technique is useful for estimating the distribution of a statistic (e.g., mean, median) 
without making any assumptions about the population.

.. code-block:: python

    from gofast.stats.utils import bootstrap
    data = [1, 2, 3, 4, 5]
    bootstrap_samples = bootstrap(data, n_bootstraps=1000, func=np.mean)
    print("Bootstrap Mean Estimates:", bootstrap_samples)
    # Output: Bootstrap Mean Estimates: [array of bootstrap mean estimates]

chi2_test
~~~~~~~~~

Conducts a Chi-squared test of independence to examine the relationship between two categorical 
variables in a contingency table.

.. math::

   \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}

where :math:`\chi^2` is the Chi-squared statistic, :math:`O_i` are the observed frequencies, 
and :math:`E_i` are the expected frequencies under the null hypothesis of independence.

.. code-block:: python

    from gofast.stats.utils import chi2_test
    table = [[10, 20, 30], [20, 15, 35]]
    print("Chi-squared Test Result:", chi2_test(table))
    # Output: Chi-squared Test Result: (Chi2-statistic, p-value)

corr
~~~~

Calculates the Pearson correlation coefficient between two variables, measuring the linear 
relationship between them.

.. math::

   r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}

where :math:`r` is the Pearson correlation coefficient, :math:`x_i` and :math:`y_i` are the individual 
sample points, and :math:`\bar{x}` and :math:`\bar{y}` are the sample means of :math:`x` and :math:`y`, 
respectively.

.. code-block:: python

    from gofast.stats.utils import corr
    x = [1, 2, 3, 4, 5]
    y = [2, 2.5, 3, 3.5, 4]
    print("Pearson Correlation:", corr(x, y))
    # Output: Pearson Correlation: Correlation coefficient

correlation
~~~~~~~~~~~

A wrapper function that can calculate various types of correlation coefficients between two variables, 
such as Pearson, Spearman, or Kendall, depending on the method specified.

.. code-block:: python

    from gofast.stats.utils import correlation
    x = [1, 2, 3, 4, 5]
    y = [2, 2.5, 3, 3.5, 4]
    print("Spearman Correlation:", correlation(x, y, method='spearman'))
    # Output: Spearman Correlation: Correlation coefficient
    
cronbach_alpha
~~~~~~~~~~~~~~

Calculates Cronbach's alpha coefficient, a measure of internal consistency or reliability of 
a psychometric test score for a sample of examinees.

.. math::

   \alpha = \frac{K}{K-1} \left(1 - \frac{\sum_{i=1}^{K} \sigma_{y_i}^2}{\sigma_{x}^2}\right)

where :math:`K` is the number of items, :math:`\sigma_{y_i}^2` is the variance of item :math:`i`, 
and :math:`\sigma_{x}^2` is the total variance of the scores.

.. code-block:: python

    from gofast.stats.utils import cronbach_alpha
    scores = [[2, 3, 4, 5], [4, 4, 3, 3], [5, 5, 4, 4]]
    print("Cronbach's Alpha:", cronbach_alpha(scores))
    # Output: Cronbach's Alpha: Cronbach's alpha coefficient

dca_analysis
~~~~~~~~~~~~

Performs Detrended Correspondence Analysis (DCA), a multivariate statistical technique used 
to analyze ecological community data, highlighting gradients in species composition.

.. math::

   DCA is calculated based on iterative solutions to eigenvalue problems, aiming to maximize 
   the variance explained by the ordination axes.

.. code-block:: python

    from gofast.stats.utils import dca_analysis
    data = [[1, 2, 0], [0, 3, 1], [4, 0, 2]]
    dca_result = dca_analysis(data)
    print("DCA Axis Scores:", dca_result.axes)
    # Output: DCA Axis Scores: [Array of scores on DCA axes]

describe
~~~~~~~~

Generates descriptive statistics that summarize the central tendency, dispersion, and shape of a 
dataset’s distribution, excluding NaN values.

.. code-block:: python

    from gofast.stats.utils import describe
    data = [1, 2, 3, 4, 5, np.nan]
    description = describe(data)
    print("Descriptive Statistics:", description)
    # Output: Descriptive Statistics: {count, mean, std, min, 25%, 50%, 75%, max}

friedman_test
~~~~~~~~~~~~~

Conducts the Friedman test, a non-parametric statistical test used to detect differences 
in treatments across multiple test attempts. Particularly useful in cases where the data 
violates the assumptions of parametric tests such as ANOVA.

.. math::

   Q = \frac{12N}{k(k+1)}\sum_{j=1}^{k}R_j^2 - 3N(k+1)

where :math:`N` is the number of subjects, :math:`k` is the number of groups, 
and :math:`R_j` is the sum of ranks for the :math:`j`th group.

.. code-block:: python

    from gofast.stats.utils import friedman_test
    data = [[1, 2, 3], [2, 3, 4], [2, 2, 2]]
    statistic, p_value = friedman_test(*data)
    print(f"Friedman statistic: {statistic}, p-value: {p_value}")

gini_coeffs
~~~~~~~~~~~

Calculates the Gini coefficient, a measure of statistical dispersion intended to 
represent the income or wealth distribution of a nation's residents. It's most 
commonly used in economics to measure inequality.

.. math::

   G = \frac{\sum_{i=1}^{n}\sum_{j=1}^{n}|x_i - x_j|}{2n^2\bar{x}}

where :math:`n` is the number of values, :math:`x_i` and :math:`x_j` are values 
for individuals :math:`i` and :math:`j`, and :math:`\bar{x}` is the mean value.

.. code-block:: python

    from gofast.stats.utils import gini_coeffs
    incomes = [20, 30, 40, 50, 100]
    gini = gini_coeffs(incomes)
    print(f"Gini Coefficient: {gini}")

get_range
~~~~~~~~~

Returns the range of the data, providing a measure of the spread or dispersion 
of a set of data values.

.. code-block:: python

    from gofast.stats.utils import get_range
    data = [1, 9, 4, 7, 5]
    range_of_data = get_range(data)
    print(f"Range: {range_of_data}")
    # Output: Range: 8

hmean
~~~~~

Calculates the harmonic mean of data, suitable for averaging ratios or rates. 
The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals of the data.

.. math::

   H = \frac{n}{\sum_{i=1}^{n}\frac{1}{x_i}}

where :math:`n` is the number of observations, and :math:`x_i` is each 
individual observation.

.. code-block:: python

    from gofast.stats.utils import hmean
    speeds = [60, 70, 80, 90, 100]
    harmonic_mean = hmean(speeds)
    print(f"Harmonic Mean: {harmonic_mean}")
    # Output: Harmonic Mean: Harmonic mean value

iqr
~~~

Computes the interquartile range (IQR) of the data, a measure of statistical 
dispersion and variability based on dividing a data set into quartiles. 
Quartiles divide a rank-ordered data set into four equal parts.

.. math::

   IQR = Q_3 - Q_1

where :math:`Q_3` is the third quartile and :math:`Q_1` is the first quartile.

.. code-block:: python

    from gofast.stats.utils import iqr
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"IQR: {iqr(data)}")
    # Example with DataFrame not provided due to function constraint.

kaplan_meier_analysis
~~~~~~~~~~~~~~~~~~~~~

Performs Kaplan-Meier analysis, a non-parametric statistic used to estimate the 
survival function from lifetime data. It's particularly useful in medical research 
for estimating the survival time of treatment or risk factors.

.. code-block:: python

    # Example usage with hypothetical function
    from gofast.stats.utils import kaplan_meier_analysis
    times = [2, 3, 5, 10, 12]
    events = [1, 1, 0, 1, 0]  # 1 if the event of interest (e.g., death) occurred, 0 if censored
    kaplan_meier_analysis(times, events)
    # Detailed example and mathematical formulation omitted due to complexity.

kolmogorov_smirnov_test
~~~~~~~~~~~~~~~~~~~~~~~

Conducts the Kolmogorov-Smirnov test for comparing the distribution of two samples 
or a sample with a reference probability distribution. It quantifies the distance 
between the empirical distribution functions of two samples.

.. math::

   D = \max |F_{1,n}(x) - F_{2,m}(x)|

where :math:`F_{1,n}` and :math:`F_{2,m}` are the empirical distribution functions 
of the first and the second sample, respectively.

.. code-block:: python

    from gofast.stats.utils import kolmogorov_smirnov_test
    data1 = [1, 2, 3, 4, 5]
    data2 = [2, 3, 4, 5, 6]
    print(kolmogorov_smirnov_test(data1, data2))
    # DataFrame example not provided due to simplicity of function input.

kruskal_wallis_test
~~~~~~~~~~~~~~~~~~~

Performs the Kruskal-Wallis H test, a non-parametric method for testing whether samples 
originate from the same distribution. It's used for comparing more than two samples that 
are independent, or not related.

.. math::

   H = \frac{12}{N(N+1)}\sum_{i=1}^{g}\frac{R_i^2}{n_i} - 3(N+1)

where :math:`N` is the total number of observations, :math:`g` is the number of groups, 
:math:`n_i` is the number of observations in group :math:`i`, and :math:`R_i` is 
the sum of ranks in group :math:`i`.

.. code-block:: python

    from gofast.stats.utils import kruskal_wallis_test
    group1 = [1, 2, 3, 4, 5]
    group2 = [2, 3, 4, 5, 6]
    group3 = [4, 5, 6, 7, 8]
    print(kruskal_wallis_test(group1, group2, group3))
    # Example with DataFrame
    import pandas as pd
    df = pd.DataFrame({'Group1': group1, 'Group2': group2, 'Group3': group3})
    print(kruskal_wallis_test('group1', 'group2', 'group3', data=df))

kurtosis
~~~~~~~~

Calculates the kurtosis of a dataset, a measure of the "tailedness" of the 
probability distribution of a real-valued random variable. Higher kurtosis 
indicates a distribution with heavier tails and sharper peak.

.. math::

   Kurtosis = \frac{N(N+1)}{(N-1)(N-2)(N-3)} \sum_{i=1}^{N} \left( \frac{x_i - \bar{x}}{s} \right)^4 - \frac{3(N-1)^2}{(N-2)(N-3)}

where :math:`N` is the sample size, :math:`x_i` are the sample observations,
 :math:`\bar{x}` is the sample mean, and :math:`s` is the sample standard deviation.

.. code-block:: python

    from gofast.stats.utils import kurtosis
    data = [1, 2, 2, 4, 4, 4, 5, 5, 5, 5]
    print(f"Kurtosis: {kurtosis(data)}")
    # Example with DataFrame not provided due to function constraint.

levene_test
~~~~~~~~~~~

Performs Levene's test for homogeneity of variances, useful for assessing if 
different samples have equal variances, an assumption of various statistical tests.

.. math::

   W = \frac{(N-k)}{(k-1)} \frac{\sum_{i=1}^{k} n_i(Z_{i\cdot} - Z_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - Z_{i\cdot})^2}

where :math:`N` is the total number of observations, :math:`k` is the number 
of groups, :math:`n_i` is the number of observations in group :math:`i`, 
:math:`Z_{ij}` is the deviation of the :math:`j^{th}` observation in the :math:`i^{th}` group 
from its group mean, and :math:`Z_{\cdot\cdot}` is the grand mean of the :math:`Z_{ij}` values.

.. code-block:: python

    from gofast.stats.utils import levene_test
    group1 = [1, 2, 3, 4, 5]
    group2 = [2, 3, 4, 5, 6]
    group3 = [4, 5, 6, 7, 8]
    print(levene_test(group1, group2, group3))
    # Example with DataFrame
    import pandas as pd
    df = pd.DataFrame({'Group1': group1, 'Group2': group2, 'Group3': group3})
    print(levene_test(data=df, columns=['Group1', 'Group2', 'Group3']))

mean
~~~~

Calculates the arithmetic mean of a dataset, providing a measure of central tendency. 
The mean is defined as the sum of all data points divided by the number of points.

.. math::

   \mu = \frac{1}{N}\sum_{i=1}^{N}x_i

where :math:`\mu` is the mean, :math:`N` is the total number of observations, 
and :math:`x_i` are the individual data points.

.. code-block:: python

    from gofast.stats.utils import mean
    data = [1, 2, 3, 4, 5]
    print("Mean:", mean(data))
    # Output: Mean: 3.0
    
mds_similarity
~~~~~~~~~~~~~~

Performs Multidimensional Scaling (MDS) to visualize the similarity or dissimilarity 
of data points. MDS aims to place each object in N-dimensional space such that the 
between-object distances are preserved as well as possible.

.. code-block:: python

    from gofast.stats.utils import mds_similarity
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    mds_result = mds_similarity(X, n_components=2, view=True)
    # This function generates a scatter plot if `view=True`.

median
~~~~~~

Determines the median value of a dataset, which represents the middle value 
when the data is sorted. If the dataset has an even number of observations, 
the median is the average of the two middle numbers.

.. math::

   \text{median} = 
   \begin{cases} 
   x_{\frac{N+1}{2}}, & \text{if } N \text{ is odd} \\
   \frac{x_{\frac{N}{2}} + x_{\frac{N}{2}+1}}{2}, & \text{if } N \text{ is even}
   \end{cases}

where :math:`N` is the total number of observations and :math:`x` are the 
sorted data points.

.. code-block:: python

    from gofast.stats.utils import median
    data = [7, 2, 10, 9, 8]
    print("Median:", median(data))
    # Output: Median: 8
    
mcnemar_test
~~~~~~~~~~~~

Conducts McNemar's test, a non-parametric method used on nominal data to determine 
whether the row and column marginal frequencies are equal. It is applied to 2x2 
contingency tables with a dichotomous trait with matched pairs of subjects.

.. math::

   \chi^2 = \frac{(b-c)^2}{b+c}

where :math:`b` and :math:`c` are the off-diagonal elements of the 2x2 contingency table.

.. code-block:: python

    from gofast.stats.utils import mcnemar_test
    data1 = [0, 1, 0, 1, 0, 1, 0, 1]
    data2 = [0, 1, 1, 1, 0, 0, 0, 1]
    df = pd.DataFrame({'Before': data1, 'After': data2})
    print(mcnemar_test('Before', 'After', data=df))
    print(mcnemar_test(data1, data2))
    # Accept direct arrays or DataFrame with column names.

mode
~~~~

Identifies the mode(s) of a dataset, which are the values that appear most 
frequently. A dataset may have one mode (unimodal), more than one mode (multimodal), or no mode.

.. code-block:: python

    from gofast.stats.utils import mode
    data = [1, 2, 2, 3, 4]
    print("Mode:", mode(data))
    # Output: Mode: [2]
    
perform_kmeans_clustering
~~~~~~~~~~~~~~~~~~~~~~~~~

Performs K-Means clustering, partitioning n observations into k clusters in which 
each observation belongs to the cluster with the nearest mean.

.. code-block:: python

    from gofast.stats.utils import perform_kmeans_clustering
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    clustering_result = perform_kmeans_clustering(X, n_clusters=4, view=True)
    # This function generates a scatter plot if `view=True`, showcasing the clustering.

perform_linear_regression
~~~~~~~~~~~~~~~~~~~~~~~~~

Performs linear regression to model the relationship between a scalar response 
and one or more explanatory variables.

.. math::

   Y = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n + \epsilon

where :math:`Y` is the dependent variable, :math:`X_i` are the independent 
variables, :math:`\beta_i` are the coefficients, and :math:`\epsilon` is the error term.

.. code-block:: python

    import numpy as np 
    from gofast.stats.utils import perform_linear_regression
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    model, coefficients, intercept = perform_linear_regression(X, y)
    print(f"Coefficients: {coefficients}, Intercept: {intercept}")
    # Using DataFrame
    df = pd.DataFrame({'X': np.random.rand(100), 'Y': 2.5 * np.random.rand(100) + np.random.normal(0, 0.5, 100)})
    regression_results= perform_linear_regression('X', 'Y', data=df, as_frame=True)
    print(regression_results)

perform_spectral_clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performs Spectral Clustering, a technique to partition a graph into clusters based on the graph's Laplacian eigenvalues. It is particularly useful for discovering clusters that are not necessarily globular and can capture complex cluster structures.

.. code-block:: python

    from gofast.stats.utils import perform_spectral_clustering
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    labels = perform_spectral_clustering(X, n_clusters=2, view=True)
    # This function generates a scatter plot if `view=True`, visualizing the clustering.

quantile
~~~~~~~~

Calculates the q-th quantile of the data along the specified axis, which is the 
value below which a given percentage of observations in a group of observations fall.

.. math::

   Q(q) = (1-q)X_{(j)} + qX_{(j+1)}

where :math:`X_{(j)}` and :math:`X_{(j+1)}` are the j-th and (j+1)-th data points 
when the data is sorted, and :math:`q` is the quantile.

.. code-block:: python

    from gofast.stats.utils import quantile
    data = [1, 2, 3, 4, 5]
    print(quantile(data, q=0.5))
    # Output: 3

quartiles
~~~~~~~~~

Calculates the three quartiles of the dataset, dividing the data set into four equal parts. 
Quartiles are special cases of quantiles (25th, 50th, and 75th percentiles).

.. code-block:: python

    from gofast.stats.utils import quartiles
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(quartiles(data))
    # Output: (2.75, 5.5, 8.25)

skew
~~~~

Measures the skewness of the data distribution, which quantifies the degree of asymmetry
 of the distribution around its mean. Positive skew indicates a tail on the right side, while 
 negative skew indicates a tail on the left.

.. math::

   Skewness = \frac{E[(X - \mu)^3]}{\sigma^3}

where :math:`E` is the expected value operator, :math:`\mu` is the mean, and :math:`\sigma` is 
the standard deviation.

.. code-block:: python

    from gofast.stats.utils import skew
    data = [1, 2, 3, 4, 5, 6, 7, 8, 20]
    print(skew(data))
    # Output: Positive value indicating right-skewed distribution
  
statistical_tests
~~~~~~~~~~~~~~~~~

A utility function that provides an interface to conduct various statistical tests 
based on the input data and the specified test type. This function aims to simplify 
the statistical analysis process by encapsulating multiple test methods within a 
single function call.

.. code-block:: python

    from gofast.stats.utils import statistical_tests
    # Example for performing a t-test for independent samples
    data1 = [1, 2, 3, 4, 5]
    data2 = [2, 3, 4, 5, 6]
    result = statistical_tests(data1, data2, test_type="ttest_indep")
    print(result)

std
~~~

Computes the standard deviation of a dataset, measuring the amount of variation or 
dispersion from the mean. The standard deviation is the square root of the variance.

.. math::

   \sigma = \sqrt{\sigma^2}

where :math:`\sigma` is the standard deviation and :math:`\sigma^2` is the variance.

.. code-block:: python

    from gofast.stats.utils import std
    data = [1, 2, 3, 4, 5]
    print("Standard Deviation:", std(data))
    # Output: Standard Deviation: 1.5811388300841898
    
t_test_independent
~~~~~~~~~~~~~~~~~~

Performs an independent samples t-test which compares the means of two independent 
groups in order to determine whether there is statistical evidence that the 
associated population means are significantly different.

.. math::

   t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}

where :math:`\bar{x}_1` and :math:`\bar{x}_2` are the sample means, :math:`s_1^2` 
and :math:`s_2^2` are the sample variances, and :math:`n_1` and :math:`n_2` are the sample sizes.

.. code-block:: python

    from gofast.stats.utils import t_test_independent
    group1 = [20, 22, 19, 24, 25]
    group2 = [28, 32, 30, 29, 27]
    t_statistic, p_value = t_test_independent(group1, group2)
    print(f"T-statistic: {t_statistic}, P-value: {p_value}")


var
~~~

Calculates the variance of a dataset, quantifying the degree to which the data 
points diverge from the mean. The variance is the average of the squared differences 
from the Mean.

.. math::

   \sigma^2 = \frac{1}{N-1}\sum_{i=1}^{N}(x_i - \mu)^2

where :math:`\sigma^2` is the variance, :math:`N` is the total number of observations, 
:math:`x_i` are the individual data points, and :math:`\mu` is the mean.

.. code-block:: python

    from gofast.stats.utils import var
    data = [1, 2, 3, 4, 5]
    print("Variance:", var(data))
    # Output: Variance: 2.5
    
wmedian
~~~~~~~

Calculates the weighted median of the given data. The weighted median is a measure of 
central tendency which considers the weight associated with each value.

.. math::

   \text{Find } x \text{ such that } \sum_{x_i < x} w_i < \frac{1}{2} \sum w_i \text{ and } \sum_{x_i > x} w_i \leq \frac{1}{2} \sum w_i

where :math:`x_i` are the data points and :math:`w_i` are their corresponding weights.

.. code-block:: python

    from gofast.stats.utils import wmedian
    data = [1, 2, 3, 4, 5]
    weights = [1, 1, 1, 1, 5]
    print(wmedian(data, weights))
    # Output: 5, since its weight pushes the median towards itself.

wilcoxon_signed_rank_test
~~~~~~~~~~~~~~~~~~~~~~~~~

Conducts the Wilcoxon Signed-Rank Test to assess whether the median of differences 
between two paired samples is zero. It is a non-parametric alternative to the 
paired t-test when the data do not follow a normal distribution.

.. math::

   W = \sum_{i=1}^{N} \text{rank}(|x_{i1} - x_{i2}|) \cdot \text{sign}(x_{i1} - x_{i2})

where :math:`x_{i1}` and :math:`x_{i2}` are the observations in the first and 
second samples, respectively, for the :math:`i^{th}` pair, and :math:`N` is 
the number of non-zero differences.

.. code-block:: python

    from gofast.stats.utils import wilcoxon_signed_rank_test
    data1 = [1, 2, 3, 4, 5]
    data2 = [2, 2, 4, 4, 5]
    statistic, p_value = wilcoxon_signed_rank_test(data1, data2)
    print(f"Statistic: {statistic}, P-value: {p_value}")

z_scores
~~~~~~~~

Calculates the z-scores of the given data, standardizing it to have a mean of 
0 and a standard deviation of 1. Z-scores indicate how many standard deviations 
an element is from the mean.

.. math::

   z = \frac{x - \mu}{\sigma}

where :math:`x` is the observation, :math:`\mu` is the mean of the dataset, 
and :math:`\sigma` is the standard deviation.

.. code-block:: python

    from gofast.stats.utils import z_scores
    data = [1, 2, 3, 4, 5]
    print(z_scores(data))
    # Output: array of z-scores

paired_t_test
~~~~~~~~~~~~~

Performs a paired t-test to determine whether there is a statistically 
significant difference between the means of two related samples.

.. math::

   t = \frac{\bar{d}}{s_{\bar{d}} / \sqrt{n}}

where :math:`\bar{d}` is the mean of the differences between the paired 
observations, :math:`s_{\bar{d}}` is the standard deviation of these differences, 
and :math:`n` is the number of pairs.

.. code-block:: python

    from gofast.stats.utils import paired_t_test
    pre_test_scores = [85, 86, 88, 75, 78, 94, 98, 79, 84, 88]
    post_test_scores = [88, 87, 86, 76, 81, 95, 99, 80, 85, 90]
    statistic, p_value = paired_t_test(pre_test_scores, post_test_scores)
    print(f"Statistic: {statistic}, P-value: {p_value}")

