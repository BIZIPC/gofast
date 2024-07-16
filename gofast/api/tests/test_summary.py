# -*- coding: utf-8 -*-
import pytest 
import unittest
import numpy as np 
import pandas as pd
import gofast.api as gf 
from gofast.api.summary import ReportFactory
from gofast.api.summary import Summary
from gofast.api.summary import ModelSummary 
from importlib import reload
import gofast.api.testing
reload(gofast.api.testing)

class TestModelSummary(unittest.TestCase):
    def setUp(self):
        """Initialize a ModelSummary instance before each test."""
        self.summary = ModelSummary(title="Test Model Summary")
         
        self.model_results = {
        'best_parameters_': {'C': 1, 'gamma': 0.1},
        'best_estimator_': "SVC",
            'cv_results_': {
                'split0_test_score': [0.6789, 0.8],
                'split1_test_score': [0.5678, 0.9],
                'split2_test_score': [0.9807, 0.95],
                'split3_test_score': [0.8541, 0.85],
                'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}],
            },
            'scoring': 'accuracy',
        }
    def test_summary_has_correct_title(self):
        """Test whether the title is set correctly."""
        gf.testing.assert_model_summary_has_title(self.summary, "Test Model Summary")

    def test_summary_method_functionality(self):
        """Test the functionality of the summary method."""
        model_results = {
        'best_parameters_': {'C': 1, 'penalty': 'l2'},
        'best_estimator_': "Logistic Regression",
            'cv_results_': {
                'split0_test_score': [0.6789, 0.8],
                'split1_test_score': [0.5678, 0.9],
                'split2_test_score': [0.9807, 0.95],
                'split3_test_score': [0.8541, 0.85],
                'params': [{'C': 1, 'penalty':'l1'}, {'C': 10, 'penalty':'l1'}],
            },
            'scoring': 'accuracy',
        }
        expected_output = ["Logistic Regression"]
        gf.testing.assert_model_summary_method_functionality(
            self.summary, expected_output, model_results,)

    def test_add_multi_contents_functionality(self):
        """Test adding multiple contents."""
        contents = [
            {"Model1": {"Accuracy": 0.95, "Precision": 0.93}},
            {"Model2": {"Accuracy": 0.90, "Precision": 0.88}}
        ]
        expected_output = ["Model1"]
        gf.testing.assert_model_summary_add_multi_contents(
            self.summary,  expected_output, contents,titles=["Test Summary"])

    def test_add_performance_functionality(self):
        """Test the add_performance method."""
        performance_data = {
            'best_parameters_': {'C': 1, 'gamma': 0.1},
            'best_estimator_': "SVM",
            'cv_results_': {
                'split0_test_score': [0.6789, 0.8],
                'split1_test_score': [0.5678, 0.9],
                'split2_test_score': [0.9807, 0.95],
                'split3_test_score': [0.8541, 0.85],
                'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}],
            },
            'scoring': 'accuracy',
         }
        expected_output = ["SVM"]
        gf.testing.assert_model_summary_add_performance(
            self.summary,expected_output, performance_data,)


class TestReportFactory(unittest.TestCase):
    def setUp(self):
        # This method will be run before each test
        self.report_factory = ReportFactory(title="Test Report")

    def test_mixed_types_summary(self):
        report_data = {
            'Total Sales': 123456.789,
            'Average Rating': 4.321,
            'Number of Reviews': 987
        }
        gf.testing.assert_report_mixed_types(self.report_factory, report_data)

    def test_recommendations(self):
        recommendations_text = "Consider increasing marketing budget."
        gf.testing.assert_report_recommendations(
            self.report_factory, recommendations_text)

    def test_data_summary(self):
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Salary': [50000, 60000, 70000, 80000, 90000]
        })
        # Assuming format_dataframe function provides a specific formatted string output
        expected_output = "Age Salary".split() 
        gf.testing.assert_report_data_summary(
            self.report_factory, df, expected_output)

class TestSummary(unittest.TestCase):
    def setUp(self):
        # Setup runs before each test method
        self.summary = Summary(title="Employee Data Overview")

    def test_basic_statistics(self):
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40, np.nan],
            'Salary': [50000, 60000, 70000, 80000, 90000]
        })
        expected_output = ["Basic Statistics"]
        self.summary.add_basic_statistics(df, include_correlation=True)
        gf.testing.assert_summary_basic_statistics(self.summary, expected_output)

    def test_unique_counts(self):
        df = pd.DataFrame({
            'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR'],
            'Age': [28, 34, 45, 32, 41]
        })
        expected_output = ["Unique Counts"]
        self.summary.add_unique_counts(df, include_sample=True, sample_size=3)
        gf.testing.assert_summary_unique_counts(self.summary, expected_output)


class TestSummary2(unittest.TestCase):
    def setUp(self):
        # Setup runs before each test method
        self.data = {
            'Age': [25, 30, 35, 40, np.nan],
            'Salary': [50000, 60000, 70000, 80000, 90000],
            'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR']
        }
        self.df = pd.DataFrame(self.data)
        self.summary = Summary(title="Employee Data Overview")

    def test_correlation_matrix_inclusion(self):
        # Test whether the correlation matrix is correctly included
        self.summary.add_basic_statistics(self.df, include_correlation=True)
        gf.testing.assert_summary_correlation_matrix(self.summary)

    def test_data_sample_inclusion(self):
        # Test whether the data sample is correctly included and formatted
        expected_sample_size = 2
        self.summary.add_unique_counts(self.df, include_sample=True, sample_size=2)
        gf.testing.assert_summary_data_sample(self.summary, expected_sample_size)

    def test_summary_completeness(self):
        # Test the completeness of the summary, combining various components
        expected_sections = "Unique Counts-Sample Data".split('-')
        self.summary.add_basic_statistics(self.df, include_correlation=True)
        self.summary.add_unique_counts(self.df, include_sample=True, sample_size=2)
        # ssert_summary_completeness is a comprehensive test function
        gf.testing.assert_summary_completeness(self.summary, expected_sections)

    # Add tests for any edge cases or specific scenarios
    def test_empty_dataframe(self):
        # Test handling of empty DataFrame
        empty_df = pd.DataFrame()
        self.summary.add_basic_statistics(empty_df)
        # assert_summary_empty_dataframe is tailored for this scenario
        gf.testing.assert_summary_empty_dataframe(self.summary)

    # Test for expected failures or warnings
    def test_invalid_input(self):
        # Test handling of invalid input (not a DataFrame)
        invalid_data = "DataFrame".split()
        with self.assertRaises(ValueError):
            self.summary.add_basic_statistics(invalid_data)



if __name__ == '__main__':
    pytest.main([__file__])
