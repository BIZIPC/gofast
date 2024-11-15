.. _quality:

Quality
=======

.. currentmodule:: gofast.dataops.quality

The :mod:`gofast.dataops.quality` module focuses on assessing and improving the quality of the data. This module provides functions to 
handle outliers, missing data, categorical features, and other quality control measures.

Key Features
------------
- **Outlier Handling**:
  Methods to identify and handle outliers in the dataset, ensuring that extreme values do not skew the analysis or model performance.
  
  - :func:`~gofast.dataops.quality.assess_outlier_impact`: Evaluates the impact of outliers on dataset statistics and optionally visualizes the results.
  - :func:`~gofast.dataops.quality.drop_correlated_features`: Analyzes and removes highly correlated features from a DataFrame to reduce multicollinearity.

- **Missing Data Handling**:
  Techniques to manage missing data by either imputing or removing missing values, thus maintaining the integrity of the dataset.
  
  - :func:`~gofast.dataops.quality.handle_missing_data`: Analyzes and handles patterns of missing data in the DataFrame.
  - :func:`~gofast.dataops.quality.check_missing_data`: Checks for missing data in a DataFrame and optionally visualizes the distribution.

- **Skewness Handling**:
  Functions to correct skewness in numerical data, improving the performance of statistical models and machine learning algorithms.
  
  - :func:`~gofast.dataops.quality.handle_skew`: Applies transformations to numeric columns in the DataFrame to correct for skewness.
  - :func:`~gofast.dataops.quality.validate_skew_method`: Validates the appropriateness of a skewness correction method based on the characteristics of the data.
  - :func:`~gofast.dataops.quality.check_skew_methods_applicability`: Evaluates which skew correction methods are applicable to the numeric columns in a DataFrame.

- **Duplicate Handling**:
  Tools to identify and handle duplicate rows in the dataset, ensuring data uniqueness and integrity.
  
  - :func:`~gofast.dataops.quality.handle_duplicates`: Handles duplicate rows in a DataFrame based on user-specified options.

- **Correlation Analysis**:
  Methods to compute and analyze the correlation between features, helping to identify and manage multicollinearity.
  
  - :func:`~gofast.dataops.quality.analyze_data_corr`: Computes the correlation matrix for specified columns and optionally visualizes it using a heatmap.
  - :func:`~gofast.dataops.quality.correlation_ops`: Performs correlation analysis on a given DataFrame and classifies the correlations into specified categories.

- **Comprehensive Quality Control**:
  A holistic approach to data quality checks, covering missing data, outliers, skewness, duplicates, and more.
  
  - :func:`~gofast.dataops.quality.quality_control`: Perform comprehensive data quality checks on a DataFrame and optionally cleans and sanitizes the data.

- **General Utility Functions**:
  Supporting functions for various data quality operations.
  
  - :func:`~gofast.dataops.quality.merge_frames_on_index`: Merges multiple DataFrames based on a specified column set as the index.
  - :func:`~gofast.dataops.quality.check_unique_values`: Checks for unique values in a DataFrame and provides detailed analysis.

Function Descriptions
---------------------

audit_data
~~~~~~~~~~
Audits and preprocesses a DataFrame for analytical consistency. This function streamlines the data cleaning process by handling various 
aspects of data quality, such as outliers, missing values, and data scaling. It provides flexibility to choose specific preprocessing 
steps according to the needs of the analysis or modeling.
This function helps ensure that the data is clean, consistent, and ready for analysis or modeling, providing a comprehensive report of 
the actions taken.

Examples:

.. code-block:: python

    import pandas as pd 
    from gofast.dataops.quality import audit_data

    # Example 1: Basic usage with outlier handling
    data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    audited_data, report = audit_data(data, handle_outliers=True, return_report=True)
    print(audited_data)
    print(report)

    # Example 2: Handling missing data and scaling
    data = pd.DataFrame({'A': [1, 2, None, 4], 'B': [4, None, 6, 8]})
    audited_data = audit_data(data, handle_missing=True, handle_scaling=True)
    print(audited_data)

    # Example 3: Comprehensive audit with multiple options
    data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50], 'C': [None, '2020-01-01', '2020-01-02', '2020-01-03']})
    audited_data, report = audit_data(data, handle_outliers=True, handle_missing=True, handle_scaling=True, handle_date_features=True, date_features=['C'], return_report=True, view=True)
    print(audited_data)
    print(report)

handle_categorical_features
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts numerical columns with a limited number of unique values to categorical columns in the DataFrame and optionally visualizes the data distribution before and after the conversion.

Examples:

.. code-block:: python

    import pandas as pd 
    from gofast.dataops.quality import handle_categorical_features

    # Example 1: Basic conversion to categorical
    data = pd.DataFrame({'A': [1, 2, 1, 3], 'B': range(4)})
    updated_data, report = handle_categorical_features(data, categorical_threshold=3, return_report=True, view=True)
    print(updated_data)
    print(report.converted_columns)

    # Example 2: Conversion with a higher threshold
    data = pd.DataFrame({'A': [1, 2, 1, 3, 4, 5], 'B': range(6)})
    updated_data = handle_categorical_features(data, categorical_threshold=5, view=True)
    print(updated_data)

    # Example 3: Comprehensive conversion with report generation
    data = pd.DataFrame({'A': [1, 2, 1, 3, 4, 5], 'B': ['a', 'b', 'c', 'a', 'b', 'c'], 'C': [1, 1, 1, 2, 2, 2]})
    updated_data, report = handle_categorical_features(data, categorical_threshold=2, return_report=True, view=True)
    print(updated_data)
    print(report.converted_columns)

This function is useful for transforming numerical columns into categorical types when they have a limited number of unique values, 
which can improve the performance and interpretability of certain models.

convert_date_features
~~~~~~~~~~~~~~~~~~~~~
Converts specified columns in the DataFrame to datetime and extracts relevant features. Optionally returns a report of the transformations and visualizing the data distribution before and after conversion.
This function helps ensure that date features are properly formatted and useful date components are extracted, enhancing the dataset's 
usability in time series analysis.
Examples:

.. code-block:: python

    import pandas as pd 
    from gofast.dataops.quality import convert_date_features

    # Example 1: Basic conversion with day of week and quarter extraction
    data = pd.DataFrame({'date': ['2021-01-01', '2021-01-02']})
    updated_data, report = convert_date_features(data, ['date'], day_of_week=True, quarter=True, return_report=True, view=True)
    print(updated_data)
    print(report)

    # Example 2: Handling custom date format
    data = pd.DataFrame({'date': ['01-01-2021', '02-01-2021']})
    updated_data = convert_date_features(data, ['date'], format='%d-%m-%Y', day_of_week=True, view=True)
    print(updated_data)

    # Example 3: Extracting additional features without visualization
    data = pd.DataFrame({'date': ['2021-01-01', '2021-01-02']})
    updated_data = convert_date_features(data, ['date'], day_of_week=True, quarter=True)
    print(updated_data)


scale_data
~~~~~~~~~~
Scales numerical columns in the DataFrame using the specified scaling method. Optionally returns a report on the scaling process along with visualization.
This function ensures that numerical data is scaled appropriately, which is crucial for certain analyses and modeling techniques.

Examples:

.. code-block:: python

    from gofast.dataops.quality import scale_data

    # Example 1: Scaling using min-max method
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    scaled_data, report = scale_data(data, method='minmax', return_report=True, view=True)
    print(scaled_data)
    print(report)

    # Example 2: Standard scaling
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    scaled_data = scale_data(data, method='standard', view=True)
    print(scaled_data)

    # Example 3: Scaling with scikit-learn
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    scaled_data = scale_data(data, method='standard', use_sklearn=True)
    print(scaled_data)

handle_outliers_in
~~~~~~~~~~~~~~~~~~
Handles outliers in numerical columns of the DataFrame using various methods. Optionally, displays a comparative plot showing the data 
distribution before and after outlier handling.
This function provides multiple methods to handle outliers, ensuring that extreme values do not distort the dataset.

Examples:

.. code-block:: python

    import pandas as pd 
    from gofast.dataops.quality import handle_outliers_in

    # Example 1: Clipping outliers
    data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    data, report = handle_outliers_in(data, method='clip', view=True, return_report=True)
    print(data)
    print(report)

    # Example 2: Removing outliers
    data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    data, report = handle_outliers_in(data, method='remove', view=True, return_report=True)
    print(data)
    print(report)

    # Example 3: Replacing outliers with median
    data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    data, report = handle_outliers_in(data, method='replace', replace_with='median', view=True, return_report=True)
    print(data)
    print(report)

handle_missing_data
~~~~~~~~~~~~~~~~~~~
Analyzes and handles patterns of missing data in the DataFrame. This function ensures the appropriate handling of missing data based on 
the specified method. If no method is provided, forward fill ('ffill') is used by default.
The application of the threshold is given as: 

.. math::

    \text{Threshold} = \text{dropna_threshold} \times \text{number of columns/rows}
    
Examples:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from gofast.dataops.quality import handle_missing_data

    # Example 1: Fill missing values with mean
    data = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
    updated_data, report = handle_missing_data(data, view=True, method='fill_mean', return_report=True)
    print(updated_data)
    print(report)

    # Example 2: Drop columns with more than 50% missing values
    data = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6], 'C': [np.nan, np.nan, 6]})
    updated_data = handle_missing_data(data, method='drop_cols', dropna_threshold=0.5, view=True)
    print(updated_data)

    # Example 3: Fill missing values with a specific value
    data = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
    updated_data = handle_missing_data(data, method='fill_value', fill_value=0, view=True)
    print(updated_data)


assess_outlier_impact
~~~~~~~~~~~~~~~~~~~~~
Assess the impact of outliers on dataset statistics and optionally visualize the results. This function calculates basic statistics with and without outliers, allows handling of NaN values, and can visualize data points and outliers using a box plot and scatter plot.
The threshold for considering a data point an outlier is defined as:

.. math::

    z = \frac{X - \mu}{\sigma}

where \( X \) is the data point, \( \mu \) is the mean, and \( \sigma \) is the standard deviation. Data points with \( |z| \) greater than the specified `outlier_threshold` are considered outliers.

Examples:

.. code-block:: python

    import numpy as np
    from gofast.dataops.quality import assess_outlier_impact

    # Example 1: Basic assessment with visualization
    data = np.random.normal(0, 1, 100)
    data[::10] += np.random.normal(0, 10, 10)  # Add some outliers
    results = assess_outlier_impact(data, view=True)
    print(results)

    # Example 2: Handling NaN values by filling with mean
    data = np.array([1, 2, np.nan, 4, 5, 100])
    results = assess_outlier_impact(data, handle_na='fill', view=True)
    print(results)

    # Example 3: Assessing a pandas DataFrame column
    import pandas as pd
    data = pd.DataFrame({'values': np.random.normal(0, 1, 100)})
    data.loc[::10, 'values'] += np.random.normal(0, 10, 10)  # Add some outliers
    results = assess_outlier_impact(data['values'], view=True)
    print(results)

merge_frames_on_index
~~~~~~~~~~~~~~~~~~~~~
Merges multiple DataFrames based on a specified column set as the index. This function is useful for combining datasets that share a 
common key, ensuring that all relevant data is aligned properly.
This function sets the specified column as the index for each DataFrame if it is not already. If the column specified is not present in 
any of the DataFrames, a KeyError will be raised.

Examples:

.. code-block:: python

    import pandas as pd 
    from gofast.dataops.quality import merge_frames_on_index

    # Example 1: Merging two DataFrames on a common column
    df1 = pd.DataFrame({'A': [1, 2, 3], 'Key': ['K0', 'K1', 'K2']})
    df2 = pd.DataFrame({'B': [4, 5, 6], 'Key': ['K0', 'K1', 'K2']})
    merged_df = merge_frames_on_index(df1, df2, index_col='Key')
    print(merged_df)
    
    # Example 2: Merging three DataFrames with an outer join
    df3 = pd.DataFrame({'C': [7, 8, 9], 'Key': ['K0', 'K1', 'K3']})
    merged_df = merge_frames_on_index(df1, df2, df3, index_col='Key', join_type='outer')
    print(merged_df)

    # Example 3: Merging DataFrames and ignoring the index
    df4 = pd.DataFrame({'D': [10, 11, 12], 'Key': ['K0', 'K1', 'K2']})
    merged_df = merge_frames_on_index(df1, df2, df3, df4, index_col='Key', ignore_index=True)
    print(merged_df)

check_missing_data
~~~~~~~~~~~~~~~~~~
Check for missing data in a DataFrame and optionally visualize the distribution of missing data with a pie chart.
This function can provide a pie chart visualization of the missing data distribution. The explode parameter can be set to 'auto' to 
dynamically highlight the slice with the highest percentage of missing data. If view is enabled and there are no missing values, no plot 
will be displayed.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import check_missing_data

    # Example 1: Checking missing data with visualization
    data = pd.DataFrame({'A': [1, 2, None, 4], 'B': [None, 2, 3, 4], 'C': [1, 2, 3, 4]})
    missing_stats = check_missing_data(data, view=True, explode='auto', shadow=True, startangle=270, verbose=1)
    print(missing_stats)

    # Example 2: Checking missing data without visualization
    missing_stats = check_missing_data(data)
    print(missing_stats)

    # Example 3: Checking missing data with specific explode settings
    explode = (0.1, 0, 0)  # Exploding the first slice
    missing_stats = check_missing_data(data, view=True, explode=explode)
    print(missing_stats)

data_assistant
~~~~~~~~~~~~~~
Performs an in-depth analysis of a pandas DataFrame, providing insights, identifying data quality issues, and suggesting corrective 
actions to optimize the data for statistical modeling and analysis. The function generates a report that includes recommendations and 
possible actions based on the analysis.
The `data_assistant` function is designed to assist in the preliminary analysis phase of data processing, offering a diagnostic view of 
the data's quality and structure. It is particularly useful for identifying and addressing common data issues before proceeding to more complex data modeling or analysis tasks.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import data_assistant

    # Example 1: Running the data assistant on a DataFrame with mixed data types
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, None],
        'Salary': [50000, 60000, 70000, 80000, 90000],
        'City': ['New York', 'Los Angeles', 'San Francisco', 'Houston', 'Seattle']
    })
    data_assistant(df, view=True)

    # Example 2: Running the data assistant without visualization
    df = pd.DataFrame({
        'Temperature': [30, 25, 27, None, 28],
        'Humidity': [60, 65, 63, 68, 70]
    })
    data_assistant(df)

    # Example 3: Running the data assistant on a large DataFrame
    df = pd.DataFrame({
        'Product': ['A', 'B', 'C', 'D', 'E'],
        'Sales': [100, 200, None, 150, 300],
        'Region': ['North', 'South', 'East', 'West', 'Central']
    })
    data_assistant(df, view=True)

check_unique_values
~~~~~~~~~~~~~~~~~~~
Checks for unique values in a pandas DataFrame and provides detailed analysis. This function helps identify columns with unique or near-unique values, which can be useful for data cleaning and preprocessing tasks.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import check_unique_values

    # Example 1: Basic usage with verbose output
    data = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['x', 'y', 'y', 'z'], 'C': [1.0, 2.0, 2.0, 3.0]})
    check_unique_values(data, verbose=True)
    # Output:
    # Unique value counts per column:
    # A    3
    # B    3
    # C    3
    # dtype: int64

    # Example 2: Automatic threshold determination
    check_unique_values(data, unique_threshold='auto', verbose=True)
    # Output:
    # Unique value counts per column:
    # A    3
    # B    3
    # C    3
    # dtype: int64

    # Example 3: Returning only columns with unique values
    unique_data = check_unique_values(data, only_unique=True, unique_threshold='auto')
    print(unique_data)
    # Output:
    #    A  B    C
    # 0  1  x  1.0
    # 1  2  y  2.0
    # 2  2  y  2.0
    # 3  3  z  3.0

    # Example 4: Returning a list of columns with unique values
    unique_cols = check_unique_values(data, return_unique_cols=True, unique_threshold='auto')
    print(unique_cols)
    # Output:
    # ['A', 'B', 'C']

This function analyzes the uniqueness of values in a DataFrame. The threshold for uniqueness can be set manually or determined 
automatically based on data types. 

- If the column is integer: count repetitive values.
- If the column is float and all values are integer-like: count unique values.
- If the column is float and not integer-like: ignore.
- If the column is categorical: apply standard uniqueness check.

check_correlated_features
~~~~~~~~~~~~~~~~~~~~~~~~~
Check for correlated features in a DataFrame. This function identifies pairs of features that are highly correlated, which can help in 
reducing multicollinearity in modeling tasks.
The correlation coefficient, :math:`r`, measures the strength and direction of a linear relationship between two features. It is defined as:

.. math::
    r = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}
    {\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}
    \\sqrt{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}}

Highly correlated features (with an absolute correlation coefficient greater than the specified `threshold`) can introduce 
multicollinearity into machine learning models, leading to unreliable model coefficients.

Examples:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from gofast.dataops.quality import check_correlated_features

    # Example 1: Basic usage with a view of the correlation matrix
    data = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    correlated_features = check_correlated_features(data, threshold=0.75, view=True)
    print("Correlated Features:", correlated_features)
    # Output: Correlated Features: [(...)]

    # Example 2: Returning correlated pairs without visualization
    correlated_features = check_correlated_features(data, threshold=0.75, view=False, return_correlated_pairs=True)
    print("Correlated Pairs:", correlated_features)
    # Output: Correlated Pairs: [(...)]

    # Example 3: Using a different correlation method (Spearman)
    correlated_features = check_correlated_features(data, method='spearman', threshold=0.75, view=True)
    print("Correlated Features:", correlated_features)
    # Output: Correlated Features: [(...)]

analyze_data_corr
~~~~~~~~~~~~~~~~~
Computes the correlation matrix for specified columns in a pandas DataFrame and optionally visualizes it using a heatmap. This function 
can also symbolically represent correlation values and selectively hide diagonal elements in the visualization and interpretability.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import analyze_data_corr

    # Example 1: Basic usage with visualization
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6]
    })
    corr_summary = analyze_data_corr(data, view=True)
    print(corr_summary)
    # Output:
    #      Correlation Table       
    # ==============================
    #        A        B        C   
    #  ----------------------------
    # A |           -1.0000   1.0000
    # B |  -1.0000           -1.0000
    # C |   1.0000  -1.0000         
    # ==============================
    #
    # <Summary: Populated. Use print() to see the contents.>
    
    # Example 2: Interpreting the correlation matrix symbolically
    corr_summary = analyze_data_corr(data, view=False, interpret=True)
    print(corr_summary)
    # Output:
    #  Correlation Table  
    # =====================
    #      A     B     C  
    #  -------------------
    # A |        --    ++  
    # B |  --          --  
    # C |  ++    --        
    # =====================
    #
    # .....................
    # Legend : ++: Strong
    #          positive,
    #          --: Strong
    #          negative,
    #          -+: Moderate
    # .....................

Note: The correlation coefficient, :math:`r`, measures the strength and direction of a linear relationship between two features. 
It is defined as:

.. math::
    r = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}
    {\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}
    \\sqrt{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}}

Highly correlated features (with an absolute correlation coefficient greater than the specified `threshold`) can introduce 
multicollinearity into machine learning models, leading to unreliable model coefficients.

correlation_ops
~~~~~~~~~~~~~~~
Performs correlation analysis on a given DataFrame and classifies the correlations into specified categories. Depending on the 
`correlation_type`, this function can categorize correlations as strong positive, strong negative, or moderate. It can also display 
the correlation matrix and returns a formatted report of the findings.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import correlation_ops

    # Example 1: Strong positive correlations
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 2, 3, 4, 4],
        'C': [5, 3, 2, 1, 1]
    })
    result = correlation_ops(data, corr_type='strong positive')
    print(result)
    # Output: (Formatted report of strong positive correlations)

    # Example 2: All correlations with visualization
    result = correlation_ops(data, corr_type='all', view=True)
    print(result)
    # Output: (Formatted report of all correlations with a heatmap)

    # Example 3: Moderate correlations
    result = correlation_ops(data, corr_type='moderate')
    print(result)
    # Output: (Formatted report of moderate correlations)

The correlation threshold parameters (`min_corr` and `high_corr`) help in fine-tuning which correlations are reported based on their 
strength. This is particularly useful in datasets with many variables, where focusing on highly correlated pairs is often more insightful.

drop_correlated_features
~~~~~~~~~~~~~~~~~~~~~~~~
Analyzes and removes highly correlated features from a DataFrame to reduce multicollinearity, improving the reliability and performance 
of subsequent statistical models. This function allows for customization of the correlation computation method and the threshold for 
feature removal.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import drop_correlated_features

    # Example 1: Basic usage with default settings
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 2, 3, 4, 4],
        'C': [5, 3, 2, 1, 1]
    })
    reduced_data = drop_correlated_features(data, threshold=0.8, strategy='first')
    print(reduced_data)
    # Output:
    #    B  C
    # 0  2  5
    # 1  2  3
    # 2  3  2
    # 3  4  1
    # 4  4  1

    # Example 2: Using custom moderate correlation values
    reduced_data = drop_correlated_features(
        data, corr_type='moderate', min_corr=0.3, threshold=0.7)
    print(reduced_data)
    # Output:
    #    B  C
    # 0  2  5
    # 1  2  3
    # 2  3  2
    # 3  4  1
    # 4  4  1

    # Example 3: Using default moderate correlation values
    reduced_data = drop_correlated_features(data, corr_type='moderate', use_default=True)
    print(reduced_data)
    # Output:
    #    B  C
    # 0  2  5
    # 1  2  3
    # 2  3  2
    # 3  4  1
    # 4  4  1


Removing correlated features is a common preprocessing step to avoid multicollinearity, which can distort the estimates of a model's parameters and affect the interpretability of variable importance. This function is particularly useful in the preprocessing steps for statistical modeling and machine learning.
The choice of correlation method, threshold, and strategy should be guided by specific analytical needs and the nature of the dataset. Lower thresholds increase the number of features removed, potentially simplifying the model but at the risk of losing important information.

handle_skew
~~~~~~~~~~~
Applies a specified transformation to numeric columns in the DataFrame to correct for skewness. This function supports logarithmic, square root, and Box-Cox transformations, helping to normalize data distributions and improve the performance of many statistical models and machine learning algorithms.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import handle_skew

    # Example 1: Using logarithmic transformation
    data = pd.DataFrame({
        'A': [0.1, 1.5, 3.0, 4.5, 10.0],
        'B': [-1, 2, 5, 7, 9]
    })
    transformed_data = handle_skew(data, method='log', view=True)
    print(transformed_data)
    # Output:
    #            A         B
    # 0  -2.302585  1.098612
    # 1   0.405465  1.791759
    # 2   1.098612  1.945910
    # 3   1.504077  2.079442
    # 4   2.302585  2.197225

    # Example 2: Using square root transformation
    transformed_data = handle_skew(data, method='sqrt', view=True)
    print(transformed_data)
    # Output:
    #           A         B
    # 0  0.316228  1.000000
    # 1  1.224745  1.414214
    # 2  1.732051  1.581139
    # 3  2.121320  1.732051
    # 4  3.162278  1.879049

    # Example 3: Using Box-Cox transformation
    transformed_data = handle_skew(data, method='box-cox', view=True)
    print(transformed_data)
    # Output:
    #           A         B
    # 0 -0.168824  1.098612
    # 1  0.635123  1.791759
    # 2  1.079148  1.945910
    # 3  1.398716  2.079442
    # 4  1.621374  2.197225

Skewness in a dataset can lead to biases in machine learning and statistical models, especially those that assume normality of the data 
distribution. By transforming skewed data, this function helps mitigate such issues, enhancing model accuracy and robustness.
It is important to understand the nature of your data and the requirements of your specific models when choosing a transformation method. 
Some methods, like 'log', cannot handle zero or negative values without adjustments.

validate_skew_method
~~~~~~~~~~~~~~~~~~~~
Validates the appropriateness of a skewness correction method based on the characteristics of the data provided. It ensures that the chosen method can be applied given the nature of the data's distribution, such as the presence of non-positive values which may affect certain transformations.
This function raises a `ValueError` if the selected method is not suitable for the data based on its values.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import validate_skew_method

    # Example 1: Validating log transformation
    data = pd.Series([0.1, 1.5, 3.0, 4.5, 10.0])
    print(validate_skew_method(data, 'log'))
    # Output:
    # The log transformation is appropriate for this data.

    # Example 2: Validating log transformation with zero values
    data_with_zeros = pd.Series([0, 1, 2, 3, 4])
    print(validate_skew_method(data_with_zeros, 'log'))
    # Output:
    # ValueError: Log transformation requires all data points to be positive. 
    # Consider using 'sqrt' or 'box-cox' method instead.


check_skew_methods_applicability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates each numeric column in a DataFrame to determine which skew correction methods are applicable based on the data's characteristics. Utilizes the `validate_skew_method` function to check the applicability of 'log', 'sqrt', and 'box-cox' transformations for each column.
This function raises a `ValueError` if a method is not applicable. It provides detailed feedback on each method's applicability and 
helps in identifying the best method for the most skewed column.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import check_skew_methods_applicability

    # Example 1: Check applicable skew methods for each column
    df = pd.DataFrame({
        'A': [0.1, 1.5, 3.0, 4.5, 10.0],
        'B': [1, 2, 3, 4, 5],
        'C': [-1, -2, -3, -4, -5],
        'D': [0, 0, 1, 2, 3]
    })
    applicable_methods = check_skew_methods_applicability(df)
    print(applicable_methods)
    # Output:
    # {'A': ['log', 'sqrt', 'box-cox'], 'B': ['log', 'sqrt', 'box-cox'], 
    # 'C': [], 'D': ['sqrt', 'box-cox']}

    # Example 2: Return the best skew correction method
    best_method = check_skew_methods_applicability(df, return_best_method=True)
    print(best_method)
    # Output:
    # log


handle_duplicates
~~~~~~~~~~~~~~~~~
Handles duplicate rows in a DataFrame based on user-specified options. This function can return a DataFrame containing duplicate rows, the indices of these rows, or remove duplicates based on the specified operation.
This function is designed to handle duplicates flexibly, allowing users to explore, identify, or clean duplicates based on their specific requirements. It is useful in data cleaning processes where duplicates might skew the results or affect the performance of data analysis and predictive modeling.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.quality import handle_duplicates

    # Example 1: Return DataFrame of duplicate rows
    data = pd.DataFrame({
        'A': [1, 2, 2, 4, 5, 1],
        'B': [1, 2, 2, 4, 5, 1]
    })
    duplicates = handle_duplicates(data, return_duplicate_rows=True)
    print(duplicates)
    # Output:
    #    A  B
    # 0  1  1
    # 2  2  2
    # 5  1  1

    # Example 2: Return indices of duplicate rows
    indices = handle_duplicates(data, return_indices=True)
    print(indices)
    # Output:
    # [0, 2, 5]

    # Example 3: Remove duplicate rows
    cleaned_data = handle_duplicates(data, operation='drop')
    print(cleaned_data)
    # Output:
    #    A  B
    # 0  1  1
    # 1  2  2
    # 3  4  4
    # 4  5  5


quality_control
~~~~~~~~~~~~~~~
Perform comprehensive data quality checks on a pandas DataFrame and, if specified, cleans and sanitizes the DataFrame based on identified 
issues using the `polish=True` parameter. This function is designed to enhance data integrity before further processing or analysis.

Examples:

.. code-block:: python

    from gofast.dataops.quality import quality_control
    import pandas as pd

    # Example 1: Basic quality control with data polishing
    data = pd.DataFrame({
        'A': [1, 2, 3, None, 5],
        'B': [1, 2, 100, 3, 4],
        'C': ['abc', 'def', '123', 'xyz', 'ab']
    })
    qc = quality_control(data,
                         value_ranges={'A': (0, 10)},
                         unique_value_columns=['B'],
                         string_patterns={'C': r'^[a-zA-Z]+$'},
                         polish=True)
    print(qc)
    # Output:
    #              Quality Control
    # ==========================================
    # missing_data    : {'A': '20.0 %'}
    # outliers        : {'B': [100]}
    # string_pattern_violations : {'C': ['123']}
    # ==========================================
    #
    # <QualityControl: 3 checks performed, data polished. Use print() to see the contents.>


Data cleaning and sanitization can significantly alter your dataset. It is essential to understand the implications of each step and 
adjust the thresholds and methods according to your data analysis goals.
