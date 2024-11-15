.. _inspection:

Inspection
==========

.. currentmodule:: gofast.dataops.inspection

The :mod:`gofast.dataops.inspection` module performs broad data checks and preprocessing of the data. This module provides functions 
to ensure data integrity and perform comprehensive inspections, offering detailed reports and actionable recommendations for data 
cleaning and preprocessing.

Key Features
------------
- **Data Integrity Verification**: Functions to check for missing values, duplicates, and outliers, ensuring the reliability of the dataset.
- **Comprehensive Data Inspection**: Tools to provide detailed statistical summaries, correlation analysis, and recommendations for data cleaning and preprocessing.
- **Actionable Reports**: Generates reports that offer insights and suggestions for improving data quality and preparing it for analysis or modeling.

Function Descriptions
---------------------

verify_data_integrity
~~~~~~~~~~~~~~~~~~~~~
Verifies the integrity of data within a DataFrame. Data integrity checks include looking for missing values, detecting duplicates, 
and identifying outliers, which are common issues that can lead to misleading analysis or model training results.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.inspection import verify_data_integrity

    # Example 1: Simple data integrity check
    data = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6], 'C': [7, 8, 8]})
    is_valid, report = verify_data_integrity(data)
    print(f"Data is valid: {is_valid}\nReport: {report}")

    # Example 2: Data with missing values and duplicates
    data_with_issues = pd.DataFrame({
        'A': [1, 2, None, 4, 5, 5],
        'B': [4, 5, 6, 4, 5, 6],
        'C': [7, 8, 8, 7, 8, 8]
    })
    is_valid, report = verify_data_integrity(data_with_issues)
    print(f"Data is valid: {is_valid}\nReport: {report}")

    # Example 3: Data with outliers
    data_with_outliers = pd.DataFrame({
        'A': [1, 2, 3, 4, 100],
        'B': [4, 5, 6, 7, 8],
        'C': [7, 8, 9, 10, 11]
    })
    is_valid, report = verify_data_integrity(data_with_outliers)
    print(f"Data is valid: {is_valid}\nReport: {report}")

inspect_data
~~~~~~~~~~~~
Performs an exhaustive inspection of a DataFrame. This function evaluates data integrity, provides detailed statistics, and offers 
tailored recommendations to ensure data quality for analysis or modeling.

Examples:

.. code-block:: python

    from gofast.dataops.inspection import inspect_data
    import numpy as np
    import pandas as pd

    # Example 1: Inspecting a DataFrame without returning a report object
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['x', 'y', 'y']})
    inspect_data(df)

    # Example 2: Inspecting a DataFrame and retrieving a report object for further analysis
    data = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(5, 2, 100),
        'C': np.random.randint(0, 100, 100)
    })
    data.iloc[0, 0] = np.nan  # Introduce a missing value
    data.iloc[1] = data.iloc[0]  # Introduce a duplicate row
    report = inspect_data(data, return_report=True)
    print(report.integrity_report)

    # Example 3: Inspecting data with high correlation and imbalance in categorical variables
    correlated_data = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(0, 1, 100),
        'C': np.random.choice(['A', 'B', 'C'], 100, p=[0.1, 0.1, 0.8])
    })
    correlated_data['B'] = correlated_data['A'] * 0.8 + np.random.normal(0, 0.1, 100)
    inspect_data(correlated_data, correlation_threshold=0.7)

    # Example 4: Inspecting data and including statistics table in the report
    detailed_data = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(5, 2, 100),
        'C': np.random.randint(0, 100, 100)
    })
    report = inspect_data(detailed_data, include_stats_table=True, return_report=True)
    print(report.stats_report)
