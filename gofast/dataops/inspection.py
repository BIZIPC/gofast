# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Performs broad data checks and preprocessing of the data."""
from __future__ import annotations, print_function 
import numpy as np 

from ..api.formatter import MultiFrameFormatter
from ..api.summary import ReportFactory
from ..api.summary import assemble_reports
from ..api.types import Any, DataFrame
from ..api.types import Dict, Tuple
from ..api.util import get_table_size 
from ..decorators import isdf
from ..tools.coreutils import to_numeric_dtypes
from ..tools.validator import is_frame

TW = get_table_size() 

__all__= [ "verify_data_integrity", "inspect_data",]

def verify_data_integrity(data: DataFrame, /) -> Tuple[bool, dict]:
    """
    Verifies the integrity of data within a DataFrame. 
    
    Data integrity checks are crucial in data analysis and machine learning 
    to ensure the reliability and correctness of any conclusions drawn from 
    the data. This function performs several checks including looking for 
    missing values, detecting duplicates, and identifying outliers, which
    are common issues that can lead to misleading analysis or model training 
    results.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to verify.

    Returns
    -------
    Tuple[bool, Report]
        A tuple containing:
        - A boolean indicating if the data passed all integrity checks 
          (True if no issues are found).
        - A dictionary with the details of the checks, including counts of 
        missing values, duplicates, and outliers by column.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.dataops.inspection import verify_data_integrity
    >>> data = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6], 'C': [7, 8, 8]})
    >>> is_valid, report = verify_data_integrity(data)
    >>> print(f"Data is valid: {is_valid}\nReport: {report}")

    Notes
    -----
    Checking for missing values is essential as they can indicate data 
    collection issues or errors in data processing. Identifying duplicates is 
    important to prevent skewed analysis results, especially in cases where 
    data should be unique (e.g., unique user IDs). Outlier detection is 
    critical in identifying data points that are significantly different from 
    the majority of the data, which might indicate data entry errors or other 
    anomalies.
    
    - The method used for outlier detection in this function is the 
      Interquartile Range (IQR) method. It's a simple approach that may not be
      suitable for all datasets, especially those with non-normal distributions 
      or where more sophisticated methods are required.
    - The function does not modify the original DataFrame.
    """
    report = {}
    is_valid = True
    # check whether dataframe is passed
    is_frame (data, df_only=True, raise_exception=True )
    data = to_numeric_dtypes(data)
    # Check for missing values
    missing_values = data.isnull().sum()
    report['missing_values'] = missing_values
    if missing_values.any():
        is_valid = False

    # Check for duplicates
    duplicates = data.duplicated().sum()
    report['duplicates'] = duplicates
    if duplicates > 0:
        is_valid = False

    # Check for potential outliers
    outlier_report = {}
    for col in data.select_dtypes(include=['number']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_report[col] = len(outliers)
        if len(outliers) > 0:
            is_valid = False

    report['outliers'] = outlier_report
    report['integrity_checks']='Passed' if is_valid else 'Failed'
    # make a report obj 
    report_obj= ReportFactory(title ="Data Integrity", **report )
    report_obj.add_mixed_types(report, table_width= TW)
    
    return is_valid, report_obj

@isdf
def inspect_data(
    data: DataFrame, /, 
    correlation_threshold: float = 0.8, 
    categorical_threshold: float = 0.75, 
    include_stats_table=False, 
    return_report: bool=False
) -> None:
    """
    Performs an exhaustive inspection of a DataFrame. 
    
    Funtion evaluates data integrity,provides detailed statistics, and offers
    tailored recommendations to ensure data quality for analysis or modeling.

    This function is integral for identifying and understanding various aspects
    of data quality such as missing values, duplicates, outliers, imbalances, 
    and correlations. It offers insights into the data's distribution, 
    variability, and potential issues, guiding users towards effective data 
    cleaning and preprocessing strategies.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be inspected.

    correlation_threshold : float, optional
        The threshold for flagging high correlation between numeric features.
        Features with a correlation above this threshold will be highlighted.
        Default is 0.8.

    categorical_threshold : float, optional
        The threshold for detecting imbalance in categorical variables. If the
        proportion of the most frequent category exceeds this threshold, it will
        be flagged as imbalanced. Default is 0.75.
        
    include_stats_table: bool, default=False 
       If ``True`` include the table of the calculated statistic in the report. 
       Otherwise include the dictionnary values of basic statistics. 
       
    return_report: bool, default=False
        If set to ``True``, the function returns a ``Report`` object
        containing the comprehensive analysis of the data inspection. This
        report includes data integrity assessment, detailed statistics,
        and actionable recommendations for data preprocessing. This option
        provides programmatic access to the inspection outcomes, allowing
        further custom analysis or documentation. If ``False``, the function
        only prints the inspection report to the console without returning
        an object. This mode is suitable for interactive exploratory data
        analysis where immediate visual feedback is desired.
        
    Returns
    -------
    None or Report
        If `return_report` is set to `False`, the function prints a comprehensive
        report to the console, including assessments of data integrity, detailed
        statistics, and actionable recommendations for data preprocessing. This
        mode facilitates immediate, visual inspection of the data quality and
        characteristics without returning an object.
        
        If `return_report` is set to `True`, instead of printing, the function
        returns a `Report` object. This object encapsulates all findings from the
        data inspection, structured into sections that cover data integrity
        assessments, statistical summaries, and preprocessing recommendations.
        The `Report` object allows for programmatic exploration and manipulation
        of the inspection results, enabling users to integrate data quality
        checks into broader data processing and analysis workflows.
  
    Notes
    -----
    - The returned ``Report`` object is a dynamic entity providing structured
      access to various aspects of the data inspection process, such as
      integrity checks, statistical summaries, and preprocessing recommendations.
    - This feature is particularly useful for workflows that require
      a detailed examination and documentation of dataset characteristics
      and quality before proceeding with further data processing or analysis.
    - Utilizing the ``return_report`` option enhances reproducibility and
      traceability of data preprocessing steps, facilitating a transparent
      and accountable data analysis pipeline.
    
    Examples
    --------
    >>> from gofast.dataops.inspection import inspect_data
    >>> import numpy as np
    >>> import pandas as pd
    
    Inspecting a DataFrame without returning a report object:
        
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['x', 'y', 'y']})
    >>> inspect_data(df)
    # Prints an exhaustive report to the console
    
    Inspecting a DataFrame and retrieving a report object for further analysis:
    
    >>> report = inspect_data(df, return_report=True)
    # A Report object is returned, enabling programmatic access to inspection details
    >>> print(report.integrity_report)
    # Access and print the integrity report part of the returned Report object
    
    >>> data = pd.DataFrame({
    >>>     'A': np.random.normal(0, 1, 100),
    >>>     'B': np.random.normal(5, 2, 100),
    >>>     'C': np.random.randint(0, 100, 100)
    >>> })
    >>> data.iloc[0, 0] = np.nan  # Introduce a missing value
    >>> data.iloc[1] = data.iloc[0]  # Introduce a duplicate row
    >>> report = inspect_data(data, return_report=True )
    >>> report.integrity_report
    <Report: Print to see the content>
    >>> report.integrity_report.outliers 
    {'A': 1, 'B': 0, 'C': 0}
    >>> report.stats_report
    Out[59]: 
        {'mean': A    -0.378182
         B     4.797963
         C    50.520000
         dtype: float64,
         'std_dev': A     1.037539
         B     2.154528
         C    31.858107
         dtype: float64,
         'percentiles':         0.25       0.50       0.75
         A  -1.072044  -0.496156   0.331585
         B   3.312453   4.481422   6.379643
         C  23.000000  48.000000  82.250000,
         'min': A   -3.766243
         B    0.054963
         C    0.000000
         dtype: float64,
         'max': A     2.155752
         B    10.751358
         C    99.000000
         dtype: float64}
    >>> report = inspect_data(data, include_stats_table=True, return_report=True)
    >>> report.stats_report
    <MultiFrame object with dataframes. Use print() to view.>

    >>> print(report.stats_report)
               Mean Values           
    =================================
               A       B        C
    ---------------------------------
    0    -0.1183  4.8666  48.3300
    =================================
            Standard Deviation       
    =================================
              A       B        C
    ---------------------------------
    0    0.9485  1.9769  29.5471
    =================================
               Percentitles          
    =================================
            0.25      0.5     0.75
    ---------------------------------
    A    -0.7316  -0.0772   0.4550
    B     3.7423   5.0293   5.9418
    C    18.7500  52.0000  73.2500
    =================================
              Minimum Values         
    =================================
               A        B       C
    ---------------------------------
    0    -2.3051  -0.1032  0.0000
    =================================
              Maximum Values         
    =================================
              A       B        C
    ---------------------------------
    0    2.3708  9.6736  99.0000
    =================================
    """  
    def calculate_statistics(d: DataFrame) -> Dict[str, Any]:
        """
        Calculates various statistics for the numerical columns of 
        the DataFrame.
        """
        stats = {}
        numeric_cols = d.select_dtypes(include=[np.number])

        stats['mean'] = numeric_cols.mean()
        stats['std_dev'] = numeric_cols.std()
        stats['percentiles'] = numeric_cols.quantile([0.25, 0.5, 0.75]).T
        stats['min'] = numeric_cols.min()
        stats['max'] = numeric_cols.max()

        return stats
    report ={}
    recommendations ={}
    is_frame( data, df_only=True, raise_exception=True,
             objname="Data for inspection")
    is_valid, integrity_report = verify_data_integrity(data)
    stats_report = calculate_statistics(data)
    if stats_report and include_stats_table: 
        # contruct a multiframe objects from stats_report  
        stats_titles = ['Mean Values', 'Standard Deviation', 'Percentitles', 
                        'Minimum Values', 'Maximum Values' ]
        keywords, stats_data =  zip (*stats_report.items() )
        stats_report=  MultiFrameFormatter(
            stats_titles, keywords, descriptor="BasicStats").add_dfs(
            *stats_data)
       
    report['integrity_status']= f"Checked ~ {integrity_report.integrity_checks}"
    report ['integrity_report']=integrity_report
    report ['stats_report'] = stats_report
    
    # Recommendations based on the report
    if not is_valid:
        print("YES")
        # report ["Recommendations"] = '-' *62
        
        if integrity_report['missing_values'].any():
            recommendations['rec_missing_values']= (
                "- Consider handling missing values using imputation or removal."
                )
        if integrity_report['duplicates'] > 0:
            recommendations['rec_duplicates']=(
                "- Check for and remove duplicate rows to ensure data uniqueness.")
        if any(count > 0 for count in integrity_report['outliers'].values()):
            recommendations['rec_outliers']=(
                "- Investigate potential outliers. Consider removal or transformation.")
        
        # Additional checks and recommendations
        # Check for columns with a single unique value
        single_value_columns = [col for col in data.columns if 
                                data[col].nunique() == 1]
        if single_value_columns:
            recommendations['rec_single_value_columns']=(
                "- Columns with a single unique value detected:"
                  f" {single_value_columns}. Consider removing them"
                  " as they do not provide useful information for analysis."
                  )
    
        # Check for data imbalance in categorical variables
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            if data[col].value_counts(normalize=True).max() > categorical_threshold:
                recommendations['rec_imbalance_data']=(
                    f"- High imbalance detected in categorical column '{col}'."
                    " Consider techniques to address imbalance, like sampling"
                    " methods or specialized models."
                    )
        # Check for skewness in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if abs(data[col].skew()) > 1:
                recommendations['rec_skewness']=(
                    f"- High skewness detected in numeric column '{col}'."
                      " Consider transformations like log, square root, or "
                      "Box-Cox to normalize the distribution.")
        # Normalization for numerical columns
        report['normalization_evaluation']=(
            "- Evaluate if normalization (scaling between 0 and 1) is"
            " necessary for numerical features, especially for distance-based"
             " algorithms.")
        report['normalization_status'] = True 
        # Correlation check
        correlation_threshold = correlation_threshold  # Arbitrary threshold
        corr_matrix = data[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(
            corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(col1, col2) for col1, col2 in zip(
            *np.where(upper_triangle > correlation_threshold))]
    
        report['correlation_checks'] = 'Passed' 
        if high_corr_pairs:
            report['high_corr_pairs'] =("- Highly correlated features detected:")
            for idx1, idx2 in high_corr_pairs:
                col1, col2 = numeric_cols[idx1], numeric_cols[idx2]
                recommendations['rec_high_corr_pairs']=(
                    f"- {col1} and {col2} (Correlation > {correlation_threshold})")
        
        # Data type conversions
        recommendations['rec_data_type_conversions'] = (
            "- Review data types of columns for appropriate conversions"
            " (e.g., converting float to int where applicable).")
    
    inspection_reports =[]
    report_obj= ReportFactory(title ="Data Inspection", **report, 
                              descriptor="Inspection")
    report_obj.add_mixed_types(report, table_width= TW)
    inspection_reports.append (report_obj )
    if recommendations: 
        recommendation_report = ReportFactory("Recommendations").add_mixed_types(
            recommendations, table_width= TW  )
        inspection_reports.append(recommendation_report)
        
    if return_report: 
        return report_obj # return for retrieving attributes. 
    report_obj = assemble_reports( *inspection_reports, display=True)




