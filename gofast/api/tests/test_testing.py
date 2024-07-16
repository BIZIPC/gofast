# -*- coding: utf-8 -*-

import pytest 
import numpy as np 
import pandas as pd 

from gofast.api.summary import Summary, ReportFactory, ModelSummary 

from gofast.api.testing import assert_box_formatter, assert_dataframe_formatter 
from gofast.api.testing import assert_dataframe_presence, assert_description_formatter 
from gofast.api.testing import assert_frame_attribute_access 
from gofast.api.testing import assert_frame_column_width_adjustment
from gofast.api.testing import assert_frame_numeric_formatting, assert_frame_title_formatting 
from gofast.api.testing import assert_metric_attributes_present, assert_summary_empty_dataframe
from gofast.api.testing import assert_summary_correlation_matrix
from gofast.api.testing import assert_summary_completeness
from gofast.api.testing import assert_summary_data_sample
from gofast.api.testing import assert_summary_basic_statistics
from gofast.api.testing import assert_summary_unique_counts
from gofast.api.testing import assert_report_mixed_types
from gofast.api.testing import assert_summary_model
from gofast.api.testing import assert_report_recommendations
from gofast.api.testing import assert_summary_content_integrity

# Sample data for testing
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Age': [25, 30, 35, 40, np.nan],
        'Salary': [50000, 60000, 70000, 80000, 90000],
        'gender': ['male', 'female', 'male', 'female', 'female']
    })

# The actual tests
def test_summary_basic_statistics(sample_dataframe):
    summary = Summary()
    summary.add_basic_statistics(sample_dataframe)
    assert_summary_correlation_matrix(summary, expected_presence=False)
    summary.add_basic_statistics(sample_dataframe, include_correlation=True)
    assert_summary_correlation_matrix(summary, expected_presence=True)

def test_summary_unique_counts(sample_dataframe):
    summary = Summary()
    summary.add_unique_counts(sample_dataframe, include_sample=True, sample_size=3)
    assert_summary_data_sample(summary, expected_sample_size=3)
    
    assert_summary_completeness(summary, expected_sections=["Unique Counts", "Sample Data"])

def test_summary_with_title(sample_dataframe):
    title = "Test Summary Report"
    summary = Summary(title=title)
    summary.add_basic_statistics(sample_dataframe)
    assert "Basic Statistics" in summary.__str__(), "Basic Statistics title is not in the summary report."
    assert_summary_completeness(summary, expected_sections=["Basic Statistics"])

# More fixtures if needed for different kinds of DataFrames
@pytest.fixture
def empty_dataframe():
    return pd.DataFrame()

@pytest.fixture
def categorical_dataframe():
    return pd.DataFrame({
        'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR'],
        'Age': [28, 34, 45, 32, 41]
    })

# The actual tests
def test_summary_with_empty_dataframe(empty_dataframe):
    summary = Summary()
    summary.add_basic_statistics(empty_dataframe)
    assert_summary_empty_dataframe(summary)

def test_summary_basic_statistics_correctness(categorical_dataframe):
    summary = Summary()
    expected_output = ["Basic Statistics"]
    assert_summary_basic_statistics(summary, expected_output, categorical_dataframe)

def test_summary_unique_counts_correctness(categorical_dataframe):
    summary = Summary()
    expected_output = ["Unique Counts"]
    assert_summary_unique_counts(summary, expected_output, categorical_dataframe)


# Sample model results for testing
@pytest.fixture
def sample_model_results():
    # Mock up of what model results might look like
    return {
        'best_estimator_': 'BestEstimator',
        'best_params_': {'param1': 'value1'},
        'cv_results_': {
        'split0_test_score': [0.6789, 0.8],
        'split1_test_score': [0.5678, 0.9],
        'split2_test_score': [0.9807, 0.95],
        'split3_test_score': [0.8541, 0.85],
        'mean_test_score': [0.8, 0.75], 
        'std_test_score': [0.05, 0.07], 
        'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}]
        }
    }

# The actual tests
def test_summary_model_output(sample_model_results):
    summary = ModelSummary()
    # use some keys expected in output  for validation.
    expected_output = [ "Model Results", "Tuning Results"] 
    assert_summary_model(summary, sample_model_results, expected_output)

def test_report_factory_mixed_types():
    report_factory = ReportFactory()
    report_data = {
        'string': 'value',
        'float': 1.0,
        'int': 1,
        'list': [1, 2, 3],
        'dict': {'key': 'value'}
    }
    assert_report_mixed_types(report_factory, report_data)

def test_report_factory_recommendations():
    report_factory = ReportFactory()
    recommendations = "Always use version control."
    keys = ["Version Control"]
    assert_report_recommendations(report_factory, recommendations, keys)


if __name__=='__main__': 
    pytest.main ([__file__])