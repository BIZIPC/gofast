.. _descriptive:

Descriptive
===========

.. currentmodule:: gofast.stats.descriptive

The :mod:`gofast.stats.descriptive` module provides a comprehensive set of functions for calculating descriptive statistics, including measures of central tendency, dispersion, and shape. This module is designed for efficient statistical computations on both array-like structures and pandas DataFrames.

Key Features
------------
- **Central Tendency Measures**: Functions for computing mean, median, mode, and weighted statistics
- **Dispersion Statistics**: Tools for calculating variance, standard deviation, range, and quartiles
- **Distribution Shape Analysis**: Methods for computing skewness, kurtosis, and other shape statistics
- **Comprehensive Summary Statistics**: Flexible describe function with visualization capabilities

Function Descriptions
--------------------

describe
~~~~~~~~
Generates comprehensive descriptive statistics for numerical data. Supports both array-like structures and pandas DataFrames, with options for visualization and customization.

Parameters:
    - data (Union[ArrayLike, DataFrame]): Input data for analysis
    - as_frame (bool): Whether to return results as DataFrame
    - view (bool): Whether to display visualization
    - plot_type (str): Type of plot ('box', 'hist', 'density')
    - cmap (str): Color map for visualization

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.stats.descriptive import describe

    # Example 1: Basic usage with numpy array
    data = np.random.rand(100, 4)
    stats = describe(data, as_frame=True)
    
    # Example 2: With DataFrame and visualization
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    describe(df, view=True, plot_type='hist')

mean
~~~~
Computes the arithmetic mean of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data
    - axis (Optional[int]): Axis along which to compute mean

Examples:

.. code-block:: python

    from gofast.stats.descriptive import mean
    
    data = [1, 2, 3, 4, 5]
    result = mean(data)

median
~~~~~~
Calculates the median value of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data
    - axis (Optional[int]): Axis along which to compute median

mode
~~~~
Identifies the mode (most frequent value) in the dataset.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

std
~~~
Calculates the standard deviation of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data
    - ddof (int): Delta degrees of freedom

var
~~~
Computes the variance of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data
    - ddof (int): Delta degrees of freedom

get_range
~~~~~~~~~
Calculates the range (difference between maximum and minimum values) of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

quartiles
~~~~~~~~
Computes the quartiles (25th, 50th, and 75th percentiles) of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

iqr
~~~
Calculates the Interquartile Range (IQR) of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

skew
~~~~
Computes the skewness of the data distribution.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

kurtosis
~~~~~~~~
Calculates the kurtosis of the data distribution.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

hmean
~~~~~
Computes the harmonic mean of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

wmedian
~~~~~~~
Calculates the weighted median of the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data
    - weights (Optional[ArrayLike]): Weights for each data point

z_scores
~~~~~~~~
Computes z-scores (standard scores) for the data.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

gini_coeffs
~~~~~~~~~~
Calculates the Gini coefficient as a measure of statistical dispersion.

Parameters:
    - data (Union[ArrayLike, Series]): Input data

Examples:

.. code-block:: python

    import numpy as np
    from gofast.stats.descriptive import *

    # Generate sample data
    data = np.random.normal(0, 1, 1000)
    
    # Calculate various descriptive statistics
    print(f"Mean: {mean(data)}")
    print(f"Median: {median(data)}")
    print(f"Standard Deviation: {std(data)}")
    print(f"Range: {get_range(data)}")
    print(f"Quartiles: {quartiles(data)}")
    print(f"IQR: {iqr(data)}")
    print(f"Skewness: {skew(data)}")
    print(f"Kurtosis: {kurtosis(data)}")
    
    # Calculate z-scores
    z_scores_data = z_scores(data)
    
    # Calculate Gini coefficient
    gini = gini_coeffs(data)