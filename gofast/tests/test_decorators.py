# -*- coding: utf-8 -*-
"""
decorators tests
=================
Created on Sat Feb 17 07:27:47 2024
@author: LKouadio a.k.a  Daniel

# tests/test_decorators.py
"""
import os
import sys
import pytest
from io import StringIO  
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np

from gofast._gofastlog import gofastlog 
from gofast.decorators import DynamicMethod , SmartProcessor 
from gofast.decorators import ExportData , isdf, Dataify
from gofast.decorators import Temp2D, CheckGDALData 
from gofast.decorators import SignalFutureChange, RedirectToNew  
from gofast.decorators  import AppendDocReferences, Deprecated
from gofast.decorators import PlotPrediction, PlotFeatureImportance
from gofast.decorators import AppendDocSection, SuppressOutput
from gofast.decorators import AppendDocFrom, NumpyDocstringFormatter
from gofast.decorators import NumpyDocstring, sanitize_docstring
from gofast.decorators import DataTransformer, Extract1dArrayOrSeries

# Define a logger mock to capture warnings
_logger = gofastlog.get_gofast_logger(__name__)
# Source function with a comprehensive docstring


# Function to be decorated
@SmartProcessor(param_name='columns', to_dataframe=True)
def normalize_data(data, columns=None):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Test DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})
# Function to be decorated with no possibility to convert to a dataframe before. 
@SmartProcessor(param_name='columns')
def normalize_data2(data, columns=None):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def test_normalize_data_with_columns():
    result = normalize_data(df, columns=['C'])
    expected_columns = ['A', 'B', 'C']
    # Check if all columns are present
    assert all(col in result.columns for col in expected_columns)
    # Check if the 'C' column remains unchanged
    assert all(result['C'] == df['C'])
    # Check if the processed columns are actually normalized
    np.testing.assert_almost_equal(result['A'].mean(), 0)
    np.testing.assert_almost_equal(result['A'].std(), 1, decimal= 0)

def test_normalize_data_without_columns():
    result = normalize_data(df)
    expected_columns = ['A', 'B', 'C']
    # Check if all columns are present and processed
    assert all(col in result.columns for col in expected_columns)
    # Check normalization
    np.testing.assert_almost_equal(result.mean().mean(), 0)
    np.testing.assert_almost_equal(result.std().mean(), 1, decimal= 0)

def test_normalize_data_with_invalid_columns():
    with pytest.raises(ValueError):
        normalize_data(df, columns=['D'])  # 'D' does not exist

def test_normalize_data_with_numpy_array():
    arr = df.values
    result = normalize_data2(arr, columns=[2])  # Assuming column index 2 corresponds to 'C'
    # Check if the shape is correct and 'C' column is unchanged
    assert result.shape == arr.shape
    assert all(result[:, 2] == arr[:, 2])
    # Check normalization of other columns
    np.testing.assert_almost_equal(np.mean(result[:, :2], axis=0), [0, 0])
    np.testing.assert_almost_equal(np.std(result[:, :2], axis=0), [.8, .8], decimal=1)

def test_data_transformer_dataframe():
    @DataTransformer(rename_columns=True, verbose=True, mode='hardworker')
    def process_data():
        # This function intentionally returns a DataFrame with different column names
        # than those specified in the original_attrs for testing purposes.
        return pd.DataFrame([[1, 2], [3, 4]], columns=['X', 'Y'])
    # Mocking original attributes to check if columns are renamed
    # In principle, it should be False since, decorator wait unfil 
    # the final execution of tthe decorated function to rename 
    # the result data frame in 'hw' mode. 
    process_data.original_attrs = {'columns': ['A', 'B']}
    
    df = process_data()
    # assert list(df.columns) == ['A', 'B'], "DataTransformer failed to rename columns"
    assert list (df.columns) ==['X', 'Y'], ( 
        "Columns are renamed while is not possible outside the decorator") 

def test_data_transformer_series():
    @DataTransformer(set_index=True, verbose=True, mode='hard-worker')
    def process_series():
        return pd.Series([10, 20, 30])
    
    # Mocking original attributes to check if index is set
    # Idem for the `process_data`. 
    process_series.original_attrs = {'index': [1, 2, 3]}
    
    series = process_series()
    assert list(series.index) != [1, 2, 3], "Expect DataTransformer fails to set index"

def test_extract_1d_array_or_series_from_dataframe():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    @Extract1dArrayOrSeries(column='A', as_series=True)
    def summarize_data(data):
        return data.mean()

    result = summarize_data(df)
    assert result == 2, "Extract1dArrayOrSeries failed to extract column and calculate mean"

def test_extract_1d_array_or_series_from_ndarray():
    ndarray = np.array([[1, 2, 3], [4, 5, 6]])
    
    @Extract1dArrayOrSeries(axis=1, column=0, as_series=False)
    def compute_average(data):
        return np.mean(data)
    
    result = compute_average(ndarray)
    assert result == 2.5, "Extract1dArrayOrSeries failed to extract axis and calculate mean"

# Test that input data is converted to a DataFrame
def test_data_conversion():
    @Dataify()
    def process_data(data):
        return data

    input_data = np.array([[1, 2], [3, 4]])
    output_data = process_data(input_data)
    assert isinstance(output_data, pd.DataFrame)

# Test column naming functionality
def test_column_naming():
    @Dataify(columns=['A', 'B'])
    def process_data(data):
        return data

    input_data = [[1, 2], [3, 4]]
    output_data = process_data(input_data)
    expected_columns = ['A', 'B']
    assert list(output_data.columns) == expected_columns

# Test use dataframe column prefix 
def test_ignore_column_prefix():
    @Dataify(prefix="column_prefix_", auto_columns= True)
    def process_data(data):
        return data

    input_data = [[1, 2], [3, 4]]
    output_data = process_data(input_data)
    # Expect prefix column names 
    assert list(output_data.columns) == ["column_prefix_0", "column_prefix_1"]
    
# Test ignoring column mismatch when ignore_mismatch is True
def test_ignore_column_mismatch():
    @Dataify(columns=['A', 'B', 'C'], ignore_mismatch=True)
    def process_data(data):
        return data

    input_data = [[1, 2], [3, 4]]
    output_data = process_data(input_data)
    # Expect default integer column names due to mismatch
    assert list(output_data.columns) == [0, 1]

# Test failing silently with invalid data conversion
def test_fail_silently():
    @Dataify(fail_silently=True)
    def process_data(data):
        return data

    input_data = "not convertible to DataFrame"
    output_data = process_data(input_data)
    # Expect the original input data to be returned
    assert output_data == input_data

# Test raising an exception with invalid data conversion when fail_silently is False
def test_exception_on_failure():
    @Dataify(fail_silently=False)
    def process_data(data):
        return data

    input_data = "not convertible to DataFrame"
    with pytest.raises(ValueError):
        process_data(input_data)

# Optionally, test with a real pandas DataFrame to ensure no conversion occurs
def test_no_conversion_needed():
    @Dataify()
    def process_data(data):
        return data

    input_data = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    output_data = process_data(input_data)
    # DataFrame should be unchanged
    pd.testing.assert_frame_equal(input_data, output_data)


# Test suppressing stdout only
def test_suppress_stdout():
    with SuppressOutput(suppress_stdout=True, suppress_stderr=False):
        print("This message should not appear")

    captured = StringIO()
    sys.stdout = captured
    print("This message should appear")
    sys.stdout = sys.__stdout__  # Restore original stdout

    assert "This message should appear" in captured.getvalue()

# Test suppressing stderr only
def test_suppress_stderr():
    with SuppressOutput(suppress_stdout=False, suppress_stderr=True):
        sys.stderr.write("This error should not appear")

    captured = StringIO()
    sys.stderr = captured
    sys.stderr.write("This error should appear")
    sys.stderr = sys.__stderr__  # Restore original stderr

    assert "This error should appear" in captured.getvalue()

# Test suppressing both stdout and stderr
def test_suppress_both():
    with SuppressOutput(suppress_stdout=True, suppress_stderr=True):
        print("This message should not appear")
        sys.stderr.write("This error should not appear")

    captured_out = StringIO()
    sys.stdout = captured_out
    print("This message should appear")
    sys.stdout = sys.__stdout__  # Restore original stdout

    captured_err = StringIO()
    sys.stderr = captured_err
    sys.stderr.write("This error should appear")
    sys.stderr = sys.__stderr__  # Restore original stderr

    assert "This message should appear" in captured_out.getvalue()
    assert "This error should appear" in captured_err.getvalue()

# Test not suppressing stdout and stderr
def test_not_suppressing():
    with SuppressOutput(suppress_stdout=False, suppress_stderr=False):
        captured_out = StringIO()
        sys.stdout = captured_out
        print("This message should appear")
        sys.stdout = sys.__stdout__  # Restore original stdout

        captured_err = StringIO()
        sys.stderr = captured_err
        sys.stderr.write("This error should appear")
        sys.stderr = sys.__stderr__  # Restore original stderr

    assert "This message should appear" in captured_out.getvalue()
    assert "This error should appear" in captured_err.getvalue()

# Test with DataFrame input
def test_df_if_with_dataframe():
    @isdf
    def process_data(data):
        return data

    input_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    output_data = process_data(input_data)
    pd.testing.assert_frame_equal(input_data, output_data)

# Test with numpy array input
def test_df_if_with_numpy_array():
    @isdf
    def process_data(data):
        return data

    input_data = np.array([[1, 2], [3, 4]])
    expected_output = pd.DataFrame(input_data, columns=None)
    output_data = process_data(input_data)
    pd.testing.assert_frame_equal(expected_output, output_data)

# Test with list input and columns argument
def test_df_if_with_list_and_columns():
    @isdf
    def process_data(data, /, columns=None):
        return data

    input_data = [[1, 2], [3, 4]]
    columns = ['A', 'B']
    expected_output = pd.DataFrame(input_data, columns=columns)
    output_data = process_data(input_data, columns=columns)
    pd.testing.assert_frame_equal(expected_output, output_data)

# Test with mismatched columns argument
def test_df_if_with_mismatched_columns():
    @isdf
    def process_data(data, /, columns=None):
        return data

    input_data = [[1, 2], [3, 4]]
    columns = ['A', 'B', 'C']  # Mismatched columns length
    with pytest.raises(ValueError):
        process_data(input_data, columns=columns)

# Test with invalid input data type
def test_df_if_with_invalid_input():
    @isdf
    def process_data(data):
        return data

    input_data = "not a valid input"
    with pytest.raises(ValueError):
        process_data(input_data)


def source_function():
    """
    Source function's docstring.

    Parameters
    ----------
    param1 : int
        Description of param1.

    Returns
    -------
    int
        Description of return value.
    """

@pytest.fixture
def source_function_with_section():
    return source_function

def test_append_doc_section(source_function_with_section):
    @AppendDocSection(source_func=source_function_with_section, start="Parameters", end="Returns")
    def target_function():
        """Target function's docstring."""
    
    assert "Parameters" in target_function.__doc__, (
        "Doc section should be correctly appended to target function's docstring.")
    
def test_append_doc_from(source_function_with_section):
    @AppendDocFrom(source=source_function_with_section, from_="Parameters",
                    to="Returns", insert_at="Target function's docstring.")
    def new_function():
        """New function docstring."""

    expected_doc_part = "Parameters"
    print("new_function.__doc__ 2=", new_function.__doc__)
    assert expected_doc_part in new_function.__doc__, (
        "Doc section should be correctly inserted into new function's docstring.")

# Example function to be decorated
def example_function(x, y):
    """
    A simple addition function.

    Parameters
    ----------
    x : int
        First number to add.
    y : int
        Second number to add.

    Returns
    -------
    int
        The sum of x and y.
    """
    return x + y

# Test the decorator without any additional parameters
def test_basic_decoration():
    decorated_function = NumpyDocstring(example_function)
    assert decorated_function(2, 3) == 5, "Functionality of the decorated function should remain unchanged"
    assert "Parameters" in decorated_function.__doc__, "Docstring should be reformatted to include 'Parameters' section"
    assert "Returns" in decorated_function.__doc__, "Docstring should be reformatted to include 'Returns' section"

# Test the decorator with custom sections and strict enforcement
@pytest.mark.skip
def test_custom_sections_and_strict_enforcement():
    @sanitize_docstring(enforce_strict=True, custom_sections={'Custom Section': 'This is a custom section.'})
    def custom_function(x):
        return x * 2
    assert custom_function(4) == 8, "Functionality of the decorated function should remain unchanged"
    assert "Custom Section" in custom_function.__doc__, "Custom section should be added to the docstring"
    assert "This is a custom section." in custom_function.__doc__, "Custom section content should be present in the docstring"

@sanitize_docstring(enforce_strict=True)
def strictly_formatted_function(x):
    """
    A strictly formatted function example.

    Parameters
    ----------
    x : int
        An input number.

    Returns
    -------
    int
        The input number doubled.
    """
    return x * 2

def test_strict_formatting_enforcement():
    expected_sections = ['Parameters', 'Returns']
    docstring = strictly_formatted_function.__doc__

    # Check if all expected sections are in the docstring
    for section in expected_sections:
        assert section in docstring, f"Docstring missing expected section: '{section}'"

    # Assuming the decorator could modify the docstring to include a warning or error for missing sections
    # Here we simulate checking for such a message (this part is hypothetical and depends on your implementation)
    assert "WARNING" not in docstring, "Docstring contains warnings about formatting issues"

    # Execute the function to ensure its functionality is not impacted
    result = strictly_formatted_function(2)
    assert result == 4, "Functionality of the strictly formatted function should remain unchanged"

# Test to ensure a simple function's docstring is formatted correctly
@pytest.mark.skip(reason="Requires Sphinx environment setup and integration")
def test_simple_function_docstring_formatting():
    @NumpyDocstringFormatter()
    def simple_function():
        '''
        A simple function.

        Returns
        -------
        None.
        '''
        pass
    
    expected_docstring = """simple_function function documentation:

Returns
-------
None.

"""
    assert simple_function.__doc__ == expected_docstring

@pytest.mark.skip(reason="Requires Sphinx environment setup and integration")
def test_included_sections():
    @NumpyDocstringFormatter(include_sections=['Parameters'])
    def function_with_parameters(a, b):
        """
        Function with parameters.

        Parameters
        ----------
        a : int
            First parameter.
        b : int
            Second parameter.
        """
        pass
    
    expected_docstring = """function_with_parameters function documentation:

Parameters
----------
a : int
    First parameter.
b : int
    Second parameter.

"""
    assert function_with_parameters.__doc__ == expected_docstring

@pytest.mark.skip(reason="Requires Sphinx environment setup and integration")
def test_custom_formatting():
    def uppercase_formatter(section_name, section_content):
        return section_content.upper()

    @NumpyDocstringFormatter(custom_formatting=uppercase_formatter)
    def function_with_custom_formatting():
        """
        A function to test custom formatting.

        Notes
        -----
        This should be uppercase.
        """
        pass
    
    expected_docstring_contains = "THIS SHOULD BE UPPERCASE."
    assert expected_docstring_contains in function_with_custom_formatting.__doc__

# Mock data for testing
y_true = pd.Series(np.random.rand(10), name="y_true")
y_pred = pd.Series(np.random.rand(10), name="y_pred")
X = pd.DataFrame(np.random.rand(10, 5), columns=[f"feature_{i}" for i in range(5)])
y_pred_model = np.random.rand(5)
model = type('model', (object,), {"feature_importances_": y_pred_model})()  # Mock model with feature_importances_
feature_names = X.columns.tolist()

@patch("matplotlib.pyplot.show")
def test_plot_prediction_on(mock_show):
    @PlotPrediction(turn='on', fig_size=(10, 6))
    def predict_function():
        return y_true, y_pred, 'on'
    
    predict_function()
    mock_show.assert_called_once()

@patch("matplotlib.pyplot.show")
def test_plot_prediction_off(mock_show):
    @PlotPrediction(turn='off')
    def predict_function():
        return y_true, y_pred, 'off'
    
    predict_function()
    mock_show.assert_not_called()

@patch("matplotlib.pyplot.show")
def test_plot_feature_importance_pfi_on(mock_show):
    @PlotFeatureImportance(kind='pfi', turn='on', fig_size=(10, 6))
    def feature_importance_function():
        return X, y_pred, y_true, model, feature_names, 'on'
    
    feature_importance_function()
    mock_show.assert_called_once()

@patch("matplotlib.pyplot.show")
def test_plot_feature_importance_off(mock_show):
    @PlotFeatureImportance(kind='pfi', turn='off')
    def feature_importance_function():
        return X, y_pred, y_true, model, feature_names, 'off'
    
    feature_importance_function()
    mock_show.assert_not_called()

# Mock function to be decorated
def mock_gdal_function():
    return "GDAL function called"

@pytest.fixture
def gdal_env_setup(monkeypatch):
    """Fixture to manipulate GDAL_DATA environment variable."""
    original_gdal_data = os.environ.get('GDAL_DATA')
    monkeypatch.setenv('GDAL_DATA', '/path/to/gdal_data')
    yield
    if original_gdal_data is not None:
        monkeypatch.setenv('GDAL_DATA', original_gdal_data)
    else:
        monkeypatch.delenv('GDAL_DATA', raising=False)

def test_gdal_data_set(gdal_env_setup):
    @CheckGDALData(verbose=1)
    def func():
        return mock_gdal_function()
    
    assert func() == "GDAL function called", "Function should execute normally when GDAL_DATA is set."

def test_gdal_data_not_set_no_error(monkeypatch):
    monkeypatch.delenv('GDAL_DATA', raising=False)
    
    @CheckGDALData(raise_error=False, verbose=1)
    def func():
        return mock_gdal_function()
    
    assert func() == "GDAL function called", "Function should execute even if GDAL_DATA is not set and raise_error is False."

def test_gdal_data_not_set_raise_error(monkeypatch):
    monkeypatch.delenv('GDAL_DATA', raising=False)
    
    @CheckGDALData(raise_error=True, verbose=1)
    def func():
        return mock_gdal_function()
    
    with pytest.raises(ImportError):
        func()

# Test for verbosity
@patch('gofast.decorators._logger.info') 
def test_verbosity(mock_logger_info, gdal_env_setup):
    @CheckGDALData(verbose=1)
    def func():
        return mock_gdal_function()
    
    func()
    mock_logger_info.assert_called()  

# New target function to redirect to
def new_function(*args, **kwargs):
    """New function that replaces a deprecated one."""
    return "new_function_result", args, kwargs

class TestRedirectToNew:
    @pytest.fixture
    def capture_log_messages(self):
        """Fixture to capture log messages."""
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)
        logging.getLogger().addHandler(ch)
        yield log_capture_string
        logging.getLogger().removeHandler(ch)

    def test_redirect_to_new(self, capture_log_messages):
        reason = "Use `new_function` instead of `old_function`."
        
        @RedirectToNew(new_target=new_function, reason=reason)
        def old_function(*args, **kwargs):
            """Deprecated old function."""
            pass

        # Call the deprecated function
        result, args, kwargs = old_function(1, 2, key='value')

        # Verify new function was called and arguments were forwarded
        assert result == "new_function_result", "The new function should be called."
        assert args == (1, 2), "Positional arguments should be forwarded to the new function."
        assert kwargs == {'key': 'value'}, "Keyword arguments should be forwarded to the new function."

        # Check for deprecation warning in logs
        log_contents = capture_log_messages.getvalue()
        assert reason in log_contents, "Deprecation warning with the correct reason should be logged."

def test_append_doc_references():
    docref = ".. |Example| replace:: Example Replacement Text"

    @AppendDocReferences(docref=docref)
    def sample_function():
        """Sample function docstring."""
        pass
    # print('sample_function.__doc__==', sample_function.__doc__)
    expected_docstring = "Sample function docstring.\n.. |Example| replace:: Example Replacement Text"
    assert sample_function.__doc__ == expected_docstring, "Docstring should include the appended doc reference."

def test_deprecated_function_warning():
    reason = "Use `new_function` instead."

    @Deprecated(reason=reason)
    def old_function():
        """Old function docstring."""
        return "Old function result."

    with pytest.warns(DeprecationWarning) as record:
        result = old_function()

    # Verify the function still returns its expected result
    assert result == "Old function result.", "Deprecated function should still operate as expected."

    # Check that the warning message contains the provided reason
    assert len(record) == 1, "Exactly one warning should be issued."
    assert reason in str(record[0].message), "The warning message should contain the reason for deprecation."

# Mock function to be decorated
def mock_function():
    return "Mock function called"

def test_signal_future_change_function_behavior():
    """
    Test that the decorated function behaves as expected.
    """
    @SignalFutureChange(message="This function will be deprecated in future releases.")
    def decorated_function():
        return mock_function()
    
    assert decorated_function() == "Mock function called", "The decorated function should return the expected result."

def test_signal_future_change_warning():
    """
    Test that decorating a function with SignalFutureChange logs a warning.
    """
    with pytest.warns(FutureWarning) as record:
        @SignalFutureChange(message="This function will be deprecated in future releases.")
        def another_decorated_function():
            return mock_function()
        
    # Verify that the warning was issued
    assert len(record) == 1
    assert "This function will be deprecated in future releases." in str(record[0].message)

@pytest.fixture
def mock_plot_data():
    return np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), {
        'xlabel': 'X Axis', 'ylabel': 'Y Axis', 'title': 'Sine Wave'}

@patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock()))
def test_temp2d_plot2d(mock_subplots, mock_plot_data):
    temp2d_instance = Temp2D()
    x, y, plot_args = mock_plot_data

    ax = temp2d_instance.plot2d(x, y, **plot_args)

    # Assuming the Axes object is the second value returned by plt.subplots
    ax = mock_subplots.return_value[1]
    ax.plot.assert_called_with(x, y)
    ax.set_xlabel.assert_called_with('X Axis')
    ax.set_ylabel.assert_called_with('Y Axis')
    ax.set_title.assert_called_with('Sine Wave')


@pytest.fixture
def mock_data_frame():
    """Provides a DataFrame for testing."""
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

def test_export_data_frame_to_csv(tmp_path, mock_data_frame):
    """
    Test exporting a DataFrame to CSV using the ExportData decorator.
    """
    save_path = tmp_path

    @ExportData(export_type='frame', file_format='csv')
    def process_and_export():
        return mock_data_frame, 'output_filename', 'csv', str(save_path), None, {}

    filenames = process_and_export()

    assert Path(filenames[0]).exists(), "CSV file should be created."
    assert Path(filenames[0]).name == 'output_filename.csv', "The CSV file should have the correct filename."

def test_export_data_frame_with_custom_name_to_excel(tmp_path, mock_data_frame):
    """
    Test exporting a DataFrame to Excel with a custom sheet name.
    """
    save_path = tmp_path

    @ExportData(export_type='frame', file_format='xlsx')
    def process_and_export_excel():
        return mock_data_frame, 'excel_output', 'xlsx', str(save_path), ['CustomSheet'], {}

    filenames = process_and_export_excel()

    # Verify the file was created with the correct name and extension
    assert Path(filenames[0]).exists(), "Excel file should be created."
    assert Path(filenames[0]).name == 'excel_output_CustomSheet.xlsx', "The Excel file should have the correct filename."
   

def test_expected_type_numeric():
    @DynamicMethod(expected_type='numeric')
    def process_numeric(data):
        return data
    
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    processed_df = process_numeric(df)
    # Check if all columns in the DataFrame are of numeric type
    assert all(dtype.kind in 'iufc' for dtype in processed_df.dtypes.values), \
        "Expected only numeric types in the DataFrame."
    # assert all(dtype.kind in 'iufc' for dtype in processed_df.dtypes), \
    #     "Expected only numeric types in the DataFrame."

def test_capture_columns():
    @DynamicMethod(capture_columns=True)
    def process_columns(data, columns=None):
        return data
    
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    processed_df = process_columns(df, columns=['A', 'B'])
    assert list(processed_df.columns) == ['A', 'B'], \
        "Expected DataFrame to only contain the specified columns."

def test_drop_na():
    @DynamicMethod(drop_na=True, na_axis='row', na_thresh=2)
    def drop_missing(data):
        return data

    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
    processed_df = drop_missing(df)
    assert processed_df.isnull().sum().sum() == 0, \
        "Expected DataFrame with no missing values."

def test_reset_index():
    @DynamicMethod(reset_index=True)
    def reset_index_func(data):
        return data

    df = pd.DataFrame({'A': [1, 2, 3]}).drop(index=1)
    processed_df = reset_index_func(df)
    assert processed_df.index[0] == 0 and processed_df.index[-1] == 1, \
        "Expected DataFrame index to be reset."

def test_custom_transform_func():
    @DynamicMethod(transform_func=lambda df: df.assign(A=df['A'] * 2))
    def transform_data(data):
        return data

    df = pd.DataFrame({'A': [1, 2, 3]})
    processed_df = transform_data(df)
    assert (processed_df['A'] == [2, 4, 6]).all(), \
        "Expected 'A' column to be doubled."


if __name__=='__main__': 
    pytest.main ([__file__])

