# -*- coding: utf-8 -*-
# testing.py 

import pandas as pd

from gofast.api.extension import isinstance_ 
from gofast.api.formatter import DataFrameFormatter, MultiFrameFormatter
from gofast.api.formatter import MetricFormatter, BoxFormatter, DescriptionFormatter
from gofast.api.formatter import to_snake_case 
from gofast.api.summary import Summary, ReportFactory, format_text 

def assert_summary_correlation_matrix(
        summary_instance, expected_presence=True, msg=None):
    """
    Asserts the presence or absence of a correlation matrix in the summary report.
    
    Parameters:
    - summary_instance (Summary): The instance of the Summary class being tested.
    - expected_presence (bool): If True, expects a correlation matrix in the summary.
                                If False, expects no correlation matrix.
    - msg (str): Optional message to display on assertion failure.
    """
    assert isinstance_(summary_instance, Summary),( 
        "Input must be an instance of Summary class." )
    summary_content = summary_instance.__str__()
    presence = "Correlation Matrix" in summary_content
    assert presence == expected_presence,( 
        msg or "Correlation matrix presence does not match expectation.")

def assert_summary_data_sample(
        summary_instance, expected_sample_size=None, msg=None):
    """
    Asserts the inclusion and size of a data sample in the summary report.
    
    Parameters:
    - summary_instance (Summary): The instance of the Summary class being tested.
    - expected_sample_size (int): The expected number of rows in the data sample. If None,
                                  the test will only check for the presence of a data sample.
    - msg (str): Optional message to display on assertion failure.
    """
    assert isinstance_(summary_instance, Summary), "Input must be an instance of Summary class."
    summary_content = summary_instance.__str__()
    sample_presence = "Sample Data" in summary_content
    assert sample_presence, msg or "Data sample is not present in the summary."
    if expected_sample_size is not None:
        # Assuming a specific format for displaying the sample size, this may need adjustment.
        sample_size_line = [
            line for line in summary_content.split('\n') if "Sample Size" in line]
        assert sample_size_line, "Sample size information is missing."
        actual_size = int(sample_size_line[0].split(":")[1].strip())
        assert actual_size == expected_sample_size, ( 
            msg or f"Expected sample size {expected_sample_size}, got {actual_size}."
    )

def assert_summary_completeness(summary_instance, expected_sections, msg=None):
    """
    Asserts the completeness of the summary report, including specified sections.
    
    Parameters:
    - summary_instance (Summary): The instance of the Summary class being tested.
    - expected_sections (list): A list of section titles expected to appear in the summary.
    - msg (str): Optional message to display on assertion failure.
    """
    assert isinstance_(summary_instance, Summary), "Input must be an instance of Summary class."
    summary_content = summary_instance.__str__()
    missing_sections = [
        section for section in expected_sections if section not in summary_content]
    assert not missing_sections, msg or f"Missing sections: {', '.join(missing_sections)}"
    
def assert_summary_empty_dataframe(summary_instance, msg=None):
    """
    Asserts the handling of an empty DataFrame by the Summary class.
    
    Parameters:
    - summary_instance (Summary): The instance of the Summary class being tested.
    - msg (str): Optional message to display on assertion failure.
    """
    assert isinstance_(summary_instance, Summary), "Input must be an instance of Summary class."
    summary_content = summary_instance.__str__()
    assert "No data" in summary_content or "Empty" in summary_content, \
        msg or "Summary does not correctly handle an empty DataFrame."

def assert_summary_basic_statistics(summary, df, expected_output, msg=None):
    """
    Asserts that the Summary object generates the correct basic statistics report
    for the provided DataFrame.
    
    Parameters:
    -----------
    summary : Summary
        The Summary instance to test.
    df : pandas.DataFrame
        The DataFrame being summarized.
    expected_output : str
        The expected basic statistics output of the Summary.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(summary, Summary), "Object is not an instance of Summary."
    summary.basic_statistics(df)
    actual_output = str(summary)
    assert actual_output == expected_output, msg or( 
        "Basic statistics summary does not match the expected output.") 

def assert_summary_unique_counts(summary, df, expected_output, msg=None):
    """
    Asserts that the Summary object correctly generates counts of unique values
    for categorical columns in the provided DataFrame.
    
    Parameters:
    -----------
    summary : Summary
        The Summary instance to test.
    df : pandas.DataFrame
        The DataFrame being summarized.
    expected_output : str
        The expected unique counts output of the Summary.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(summary, Summary), "Object is not an instance of Summary."
    summary.unique_counts(df)
    actual_output = str(summary)
    assert actual_output == expected_output, msg or ( 
        "Unique counts summary does not match the expected output.")

def assert_summary_model(summary, model_results, expected_output, msg=None):
    """
    Asserts that the Summary object generates the correct model summary report
    based on provided model results or a scikit-learn model object.
    
    Parameters:
    -----------
    summary : Summary
        The Summary instance to test.
    model_results : dict or sklearn model
        Model results dictionary or a scikit-learn model with necessary attributes.
    expected_output : str
        The expected model summary output of the Summary.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(summary, Summary), "Object is not an instance of Summary."
    summary.model_summary(model_results=model_results)
    actual_output = str(summary)
    assert actual_output == expected_output, ( 
        msg or "Model summary does not match the expected output.") 

def assert_summary_content_integrity(
        summary, expected_content_attributes, msg=None):
    """
    Asserts that the Summary object correctly reflects the content attributes
    provided in its initialization and method invocations.
    
    Parameters:
    -----------
    summary : Summary
        The Summary instance to test.
    expected_content_attributes : dict
        A dictionary where each key-value pair represents an expected attribute
        name and its corresponding content within the Summary.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(summary, Summary), "Object is not an instance of Summary."
    for attr, expected_content in expected_content_attributes.items():
        actual_content = getattr(summary, attr, None)
        assert actual_content == expected_content, ( 
            msg or f"Summary attribute '{attr}' does not contain the expected content.") 

def assert_report_mixed_types(report_factory, report_data, msg=None):
    """
    Asserts that the ReportFactory object's report contains mixed data types,
    including but not limited to strings, floats, and integers. It is expected
    that the report should also handle other iterable data types or structures
    indicating the presence of mixed types data.

    Parameters:
    -----------
    report_factory : ReportFactory
        The instance of ReportFactory to be tested.
    report_data : dict
        A dictionary representing the mixed data types intended for report generation.
    msg : str, optional
        Custom message to display on assertion failure.
    """
    if not isinstance(report_factory, ReportFactory):
        raise AssertionError("The provided object is not an instance of ReportFactory.")

    # Trigger mixed types summary generation within the report factory.
    report_factory.mixed_types_summary(report_data)

    # Determine if mixed types are present, focusing on iterable types excluding strings.
    contains_mixed_types = any(
        not isinstance(value, (str, float, int)) for value in report_factory.report.values()
    )

    # Assert the presence of mixed types, raise an AssertionError 
    # with a custom message if the check fails.
    assert contains_mixed_types, msg or "Report does not contain expected mixed data types."

 
def assert_report_recommendations(
        report_factory, recommendations, keys=None,
        key_length=15, max_char_text=70,  msg=None):
    """
    Asserts that the ReportFactory object correctly formats and includes a text-based
    recommendations section within the report, considering recommendations could be a string,
    list of strings, or a dictionary.

    Parameters:
    -----------
    report_factory : ReportFactory
        The ReportFactory instance to test.
    recommendations : str, list of str, or dict
        The recommendations to be included in the report. Can be a single string, a list of
        strings, or a dictionary with keys as headings.
    keys : list of str, optional
        The keys associated with each recommendation if recommendations are provided as a
        list or a single string.
    key_length : int, optional
        Maximum key length for formatting before the ':'. Make sure to indicate 
        for a precise formatted of `report_factory`. Defaults to 15.
    max_char_text : int, optional
        Number of text characters to tolerate before wrapping to a new line.
        Make sure to precise identical of for formatting `report_factory` 
        for correct formattage. Defaults to 70.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    if not isinstance_(report_factory, ReportFactory):
        raise AssertionError("The provided object is not an instance of ReportFactory.")
    
    # Simulate adding recommendations to the report to trigger formatting.
    report_factory.add_recommendations(recommendations, keys=keys)

    # Validate the structure of the formatted recommendations within report_str.
    if isinstance(recommendations, dict):
        # If recommendations are a dict, verify that each key-value pair is formatted correctly.
        formatted_recommendations = {
            key: format_text(text, key, max_char_text=max_char_text, key_length=key_length)
            for key, text in recommendations.items()
        }
    elif isinstance(recommendations, (list, str)):
        # If recommendations are a list or a single string, use the provided keys or default keys.
        if not keys:
            keys = [f"Key{i+1}" for i in range(len(recommendations))] if isinstance(
                recommendations, list) else ["Key1"]
        recommendations = recommendations if isinstance(recommendations, list) else [recommendations]
        formatted_recommendations = {
            key: format_text(text, key, max_char_text=max_char_text, key_length=key_length)
            for key, text in zip(keys, recommendations)
        }
    
    # Construct the expected formatted string representation of recommendations.
    expected_formatted_str = '\n'.join(formatted_recommendations.values())
    # Ensure the actual formatted string contains the expected formatted recommendations.
    assert expected_formatted_str in report_factory.report_str, ( 
        msg or "Recommendations section is not formatted as expected." ) 


def assert_report_model_performance(
        report_factory, model_results, expected_output, msg=None
        ):
    """
    Asserts that the ReportFactory object correctly formats a model performance summary.
    
    Parameters:
    -----------
    report_factory : ReportFactory
        The ReportFactory instance to test.
    model_results : dict
        The model results data to be formatted into a report section.
    expected_output : str
        The expected formatted report output for the model performance summary.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(report_factory, ReportFactory), "Object is not an instance of ReportFactory."
    report_factory.model_performance_summary(model_results)
    actual_output = str(report_factory)
    assert actual_output == expected_output, ( 
        msg or "Model performance summary does not match expected output.") 

def assert_report_data_summary(report_factory, df, expected_output, msg=None):
    """
    Asserts that the ReportFactory object correctly formats a DataFrame summary.
    
    Parameters:
    -----------
    report_factory : ReportFactory
        The ReportFactory instance to test.
    df : pandas.DataFrame
        The DataFrame to be summarized into a report section.
    expected_output : str
        The expected formatted report output for the DataFrame summary.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(report_factory, ReportFactory), ( 
        "Object is not an instance of ReportFactory." )
    report_factory.data_summary(df)
    actual_output = str(report_factory)
    assert actual_output == expected_output,(
        msg or "DataFrame summary does not match expected output."
        )

def assert_metric_formatter(metric_formatter, expected_output, msg=None):
    """
    Asserts that a MetricFormatter object formats its metrics correctly,
    comparing the actual formatted string against the expected output.

    This function tests the correctness of MetricFormatter's functionality
    in producing formatted string representations of metrics.

    Parameters:
    -----------
    metric_formatter : MetricFormatter
        The instance of MetricFormatter to be tested. It should have been
        initialized and populated with metrics before calling this function.
        
    expected_output : str
        The expected formatted string output by the MetricFormatter. This string
        should represent exactly how the metrics, potentially along with a title,
        are expected to be formatted by the MetricFormatter instance.
        
    msg : str, optional
        An optional message to display upon an assertion failure. This can be
        used to provide additional context to help diagnose failed tests.

    Raises:
    ------
    AssertionError
        If the input object is not an instance of MetricFormatter, or if the actual
        formatted string output by the MetricFormatter does not match the expected
        output, an AssertionError is raised. A custom message is included if provided.
    """
    # Verify that the input object is an instance of MetricFormatter
    assert isinstance_(metric_formatter, MetricFormatter), msg or (
        "The provided object is not an instance of MetricFormatter."
    )

    # Generate the actual output by converting the MetricFormatter object to string
    actual_output = str(metric_formatter)
    
    # Check if the actual output matches the expected output
    assert actual_output == expected_output, msg or (
        "MetricFormatter output does not match expected output.\n"
        "Expected:\n"
        f"{expected_output}\n"
        "Actual:\n"
        f"{actual_output}"
    )

def assert_metric_attributes_present(
        metric_formatter, expected_metrics, expected_title=None, msg=None):
    """
    Asserts that the expected metrics and title are present in the 
    MetricFormatter instance.

    Parameters:
    -----------
    metric_formatter : MetricFormatter
        The MetricFormatter instance to test.
    expected_metrics : dict
        A dictionary of expected metric names and their values.
    expected_title : str, optional
        The expected title of the MetricFormatter. If None, the title is not checked.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    # Verify that the input object is an instance of MetricFormatter
    assert isinstance_(metric_formatter, MetricFormatter), msg or (
        "The provided object is not an instance of MetricFormatter."
    )
    # Check for the title if expected
    if expected_title is not None:
        assert metric_formatter.title == expected_title, msg or (
            f"Expected title '{expected_title}' but found '{metric_formatter.title}'.")

    # Check each metric is present and matches the expected value
    for metric_name, expected_value in expected_metrics.items():
        actual_value = getattr(metric_formatter, metric_name, None)
        assert actual_value is not None, msg or (
            f"Metric '{metric_name}' not found in MetricFormatter.")
        assert actual_value == expected_value, msg or (
            f"Metric '{metric_name}' value mismatch."
            f" Expected: {expected_value}, Actual: {actual_value}")
        
def assert_metric_formatter_includes_keywords(
        metric_formatter, expected_keywords, msg=None):
    """
    Asserts that the MetricFormatter instance includes the expected keywords
    as attributes, transformed into snake_case, enabling attribute-style access.

    Parameters:
    -----------
    metric_formatter : MetricFormatter
        The MetricFormatter instance to test.
    expected_keywords : list of str
        A list of expected keyword strings that should be accessible as attributes.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    # Verify that the input object is an instance of MetricFormatter
    assert isinstance_(metric_formatter, MetricFormatter), msg or (
        "The provided object is not an instance of MetricFormatter."
    )
    
    for keyword in expected_keywords:
        snake_case_keyword = keyword.replace(" ", "_").lower()
        assert hasattr(metric_formatter, snake_case_keyword), msg or (
            f"Expected keyword '{keyword}' (as '{snake_case_keyword}')"
            " not found as attribute in MetricFormatter.")

def assert_box_formatter(box_formatter, expected_output, msg=None):
    """
    Asserts that a BoxFormatter object formats its content as expected.

    Parameters:
    -----------
    box_formatter : BoxFormatter
        The BoxFormatter instance to test.
    expected_output : str
        The expected string output of the BoxFormatter.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    # Verify that the input object is an instance of MetricFormatter
    assert isinstance_(box_formatter, BoxFormatter), msg or (
        "The provided object is not an instance of BoxFormatter."
    )
    actual_output = str(box_formatter)
    assert actual_output == expected_output, msg or ( 
        "BoxFormatter output does not match expected output.")
    
def assert_description_formatter(description_formatter, expected_output, msg=None):
    """
    Asserts that a DescriptionFormatter object formats its content as expected.

    Parameters:
    -----------
    description_formatter : DescriptionFormatter
        The DescriptionFormatter instance to test.
    expected_output : str
        The expected string output of the DescriptionFormatter.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    validate_formatter_instance(description_formatter, DescriptionFormatter)
    actual_output = str(description_formatter)
    assert actual_output == expected_output, msg or (
        "DescriptionFormatter output does not match expected output.")

def assert_dataframe_formatter(dataframe_formatter, expected_output, msg=None):
    """
    Asserts that a DataFrameFormatter object formats its DataFrame as expected,
    and verifies that the input object is an instance of DataFrameFormatter. This 
    function is useful for testing the visual presentation and formatting 
    capabilities of the DataFrameFormatter class.

    Parameters:
    -----------
    dataframe_formatter : DataFrameFormatter
        The DataFrameFormatter instance to test. This function first checks if 
        the input object is indeed an instance of DataFrameFormatter to ensure 
        that the appropriate methods and attributes are available for testing.
    expected_output : str
        The expected string output of the DataFrameFormatter. This is the 
        formatted representation of the DataFrame that should result from 
        applying the DataFrameFormatter's formatting logic.
    msg : str, optional
        A custom message to display on assertion failure. This can be used to 
        provide additional context or details about the test case or the 
        expected outcome, helping to diagnose failures more effectively.

    Raises:
    -------
    AssertionError
        If the actual formatted output of the DataFrameFormatter does not match 
        the expected output, or if the input object is not an instance of 
        DataFrameFormatter. This ensures that only valid and correctly formatted 
        outputs pass the test, promoting consistency and reliability in formatting 
        behavior.

    Examples:
    ---------
    >>> import pandas as pd
    >>> from gofast.api.formatter import DataFrameFormatter
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> formatter = DataFrameFormatter()
    >>> formatter.add_df(df)
    >>> expected_output = "A formatted string representing the DataFrame"
    >>> assert_dataframe_formatter(formatter, expected_output)
    """
    # Verify that the input is an instance of DataFrameFormatter
    assert isinstance_(dataframe_formatter, DataFrameFormatter), (
        "The input is not an instance of DataFrameFormatter."
    )
    
    # Convert the DataFrameFormatter instance to a string for comparison
    actual_output = str(dataframe_formatter)
    
    # Assert that the actual output matches the expected output
    assert actual_output == expected_output, msg or (
        "DataFrameFormatter output does not match expected output."
    )
def assert_multiframe_formatter(multiframe_formatter, expected_output, msg=None):
    """
    Asserts that a MultiFrameFormatter object formats its DataFrames as expected.

    Parameters:
    -----------
    multiframe_formatter : MultiFrameFormatter
        The MultiFrameFormatter instance to test.
    expected_output : str
        The expected string output of the MultiFrameFormatter.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    validate_formatter_instance(multiframe_formatter, MultiFrameFormatter)
    actual_output = str(multiframe_formatter)
    assert actual_output == expected_output, msg or (
        "MultiFrameFormatter output does not match expected output."
        )
    
def assert_frame_column_width_adjustment(
        dataframe_formatter, column_name, expected_width, msg=None):
    """
    Asserts that the specified column of the DataFrameFormatter object has been
    adjusted to the expected width.
    
    Parameters:
    -----------
    dataframe_formatter : DataFrameFormatter
        The DataFrameFormatter instance being tested.
    column_name : str
        The name of the column to check the width of.
    expected_width : int
        The expected width of the column after formatting.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(dataframe_formatter, DataFrameFormatter), (
        "The input is not an instance of DataFrameFormatter."
    )

    actual_width = dataframe_formatter._column_name_mapping[column_name]
    assert len(actual_width) == expected_width, msg or (
        f"Width of column '{column_name}' does not match the expected width."
    )
def assert_frame_title_formatting(
        dataframe_formatter, expected_title, msg=None):
    """
    Asserts that the title of the DataFrameFormatter object is properly centered
    and formatted.
    
    Parameters:
    -----------
    dataframe_formatter : DataFrameFormatter
        The DataFrameFormatter instance being tested.
    expected_title : str
        The expected title after formatting.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(dataframe_formatter, DataFrameFormatter), (
        "The input is not an instance of DataFrameFormatter."
    )

    # Assuming _format_header() or similar method exists and returns formatted title
    actual_title = dataframe_formatter._format_header()[0]
    assert actual_title.strip() == expected_title.strip(), msg or (
        "The formatted title does not match the expected title."
    )
    
def assert_frame_numeric_formatting(
        dataframe_formatter, column_name, row_index, expected_value, msg=None):
    """
    Asserts that a numeric value in the DataFrameFormatter object is formatted
    to the correct precision.
    
    Parameters:
    -----------
    dataframe_formatter : DataFrameFormatter
        The DataFrameFormatter instance being tested.
    column_name : str
        The name of the column containing the numeric value.
    row_index : int
        The index of the row containing the numeric value.
    expected_value : float or int
        The expected formatted value.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(dataframe_formatter, DataFrameFormatter), (
        "The input is not an instance of DataFrameFormatter."
    )

    actual_value = dataframe_formatter.df.loc[row_index, column_name]
    assert actual_value == expected_value, msg or (
        f"The value at {column_name}[{row_index}] is not formatted correctly."
    )
    
def assert_frame_attribute_access(
        dataframe_formatter, attribute_name, expected_column_name, msg=None):
    """
    Asserts that a DataFrame column can be accessed as an attribute of the
    DataFrameFormatter object using the snake_case version of its name.
    
    Parameters:
    -----------
    dataframe_formatter : DataFrameFormatter
        The DataFrameFormatter instance being tested.
    attribute_name : str
        The snake_case name of the attribute to access.
    expected_column_name : str
        The original name of the column expected to be accessed.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    assert isinstance_(dataframe_formatter, DataFrameFormatter), (
        "The input is not an instance of DataFrameFormatter."
    )

    actual_column = getattr(dataframe_formatter, attribute_name)
    assert actual_column.equals(dataframe_formatter.df[expected_column_name]), msg or (
        f"Attribute '{attribute_name}' does not correctly"
        f" access the column '{expected_column_name}'."
    )
    
def assert_dataframe_presence(multiframe_formatter, dataframe, msg=None):
    """
    Asserts that a specific DataFrame is present within the MultiFrameFormatter instance.
    
    Parameters:
    -----------
    multiframe_formatter : MultiFrameFormatter
        The MultiFrameFormatter instance being tested.
    dataframe : pandas.DataFrame
        The DataFrame expected to be present in the MultiFrameFormatter.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    validate_formatter_instance(multiframe_formatter, MultiFrameFormatter)
    assert dataframe in multiframe_formatter.dfs, ( 
        msg or "DataFrame not found in MultiFrameFormatter."
        )

def assert_multiframe_column_accessibility(
        multiframe_formatter, keyword, column_name, msg=None):
    """
    Asserts that DataFrame columns can be accessed through attributes generated 
    from keywords and column names in the MultiFrameFormatter instance.
    
    Parameters:
    -----------
    multiframe_formatter : MultiFrameFormatter
        The MultiFrameFormatter instance being tested.
    keyword : str
        The keyword associated with the DataFrame whose column is to be accessed.
    column_name : str
        The name of the column to access through the generated attribute.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    validate_formatter_instance(multiframe_formatter, MultiFrameFormatter)
    attribute_name = f"{to_snake_case(column_name)}_{to_snake_case(keyword)}"
    assert hasattr(multiframe_formatter, attribute_name), ( 
        msg or f"Attribute '{attribute_name}' not accessible.") 

def assert_multiframe_title_display(
        multiframe_formatter, title, expected_in_output=True, msg=None):
    """
    Asserts that the title is correctly displayed (or not displayed) in the 
    string output of the MultiFrameFormatter instance.
    
    Parameters:
    -----------
    multiframe_formatter : MultiFrameFormatter
        The MultiFrameFormatter instance being tested.
    title : str
        The title to check for in the MultiFrameFormatter output.
    expected_in_output : bool, optional
        Indicates whether the title is expected to be found in the output.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    validate_formatter_instance(multiframe_formatter, MultiFrameFormatter)
    output_contains_title = title in str(multiframe_formatter)
    assert output_contains_title == expected_in_output, msg or (
        f"Title '{title}' presence in output is not as expected.")

def assert_multiframe_correct_formatting_strategy(
        multiframe_formatter, expected_strategy, msg=None):
    """
    Asserts that the MultiFrameFormatter uses the correct formatting strategy 
    (same columns vs. different columns) based on the added DataFrames.
    
    Parameters:
    -----------
    multiframe_formatter : MultiFrameFormatter
        The MultiFrameFormatter instance being tested.
    expected_strategy : str
        The expected formatting strategy: 'same_columns' or 'different_columns'.
    msg : str, optional
        A custom message to display on assertion failure.
    """
    validate_formatter_instance(multiframe_formatter, MultiFrameFormatter)
    if expected_strategy == 'same_columns':
        strategy_func = multiframe_formatter.dataframe_with_same_columns
    elif expected_strategy == 'different_columns':
        strategy_func = multiframe_formatter.dataframe_with_different_columns
    else:
        raise ValueError("Invalid expected_strategy. Choose"
                         " 'same_columns' or 'different_columns'.")
    
    actual_output = strategy_func()
    assert isinstance(actual_output, str), ( 
        msg or "The formatting strategy did not produce the expected output type."
        )

def validate_formatter_instance(formatter_instance, cls):
    """
    Validates that the provided formatter instance is an instance of 
    the specified class.
    
    Parameters:
    -----------
    formatter_instance : object
        The formatter instance to be validated.
    cls : type
        The class type against which the formatter instance is to be validated.
    
    Raises:
    -------
    AssertionError
        If the formatter instance is not an instance of the specified class.
    """
    assert isinstance_(formatter_instance, cls), (
        f"The input is not an instance of {cls.__name__}."
    )

if __name__=="__main__": 
    
    # import pytest 
    import unittest
    # from gofast.api.formatter import MetricFormatter
    # from testing import assert_metric_formatter
    
    class TestMetricFormatterUsage(unittest.TestCase):
        def test_metric_formatter_output(self):
            metrics = MetricFormatter(accuracy=0.95, precision=0.93, recall=0.92)
            expected_output = """
    ====================================================
    accuracy    : 0.95
    precision   : 0.93
    recall      : 0.92
    ====================================================
    """     # Note: The expected_output should match the actual format including whitespace
            assert_metric_formatter(metrics, expected_output.strip())
    
    # example_usage_test.py (continued)
    
    class TestDataFrameFormatterUsage(unittest.TestCase):
        def test_dataframe_formatter_output(self):
            df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            formatter = DataFrameFormatter()
            formatter.add_df(df)
            expected_output = """
    A    B
    --  --
    1    3
    2    4
    """     # Note: Adjust the expected output to match the actual format
            assert_dataframe_formatter(formatter, expected_output.strip())
    class TestMultiFrameFormatterUsage(unittest.TestCase):
        def test_multiframe_formatter_output(self):
            df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
            multiframe_formatter = MultiFrameFormatter()
            multiframe_formatter.add_dfs(df1, df2)
            expected_output = """
    DataFrame 1
    A    B
    --  --
    1    3
    2    4
    
    DataFrame 2
    C    D
    --  --
    5    7
    6    8
    """     # Note: Adjust the expected output to match the actual format
            assert_multiframe_formatter(multiframe_formatter, expected_output.strip())
    # pytest.main ([__file_])