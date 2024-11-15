# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer 
from sklearn.pipeline import Pipeline

from gofast.mlops.testing import (
    PipelineTest,
    ModelQuality,
    OverfittingDetection,
    DataIntegrity,
    BiasDetection,
    ModelVersionCompliance,
    PerformanceRegression,
    CIIntegration,
)


def test_pipeline_test():
    # Sample pipeline
    pipeline = Pipeline([
        ('clf', DecisionTreeClassifier())
    ])

    # Sample data
    X, y = load_iris(return_X_y=True)

    # scroing function 
    scorer = make_scorer(accuracy_score)
    # Initialize the PipelineTest
    pipeline_test = PipelineTest(pipeline=pipeline,scoring= scorer)

    # Run the test
    pipeline_test.run(X, y)

    # Check if the test ran successfully
    assert hasattr(pipeline_test, 'results_'), "PipelineTest did not produce results."


def test_model_quality():
    # Sample model
    model = LogisticRegression()

    # Sample data
    X, y = load_iris(return_X_y=True)

    # Initialize the ModelQuality test
    quality_test = ModelQuality(model=model, metrics={'accuracy': accuracy_score})

    # Fit the model
    quality_test.fit(X, y)

    # Check if the test ran successfully
    assert hasattr(quality_test, 'results_'), "ModelQuality did not produce results."
    assert 'accuracy' in quality_test.results_['test_metrics'], "Accuracy metric not computed."


def test_overfitting_detection():
    # Sample model
    model = DecisionTreeClassifier(max_depth=None)

    # Sample data
    X, y = load_iris(return_X_y=True)

    # Initialize the OverfittingDetection test
    overfit_test = OverfittingDetection(model=model)

    # Fit the model
    overfit_test.fit(X, y)

    # Check if the test ran successfully
    assert hasattr(overfit_test, 'results_'), "OverfittingDetection did not produce results."
    assert 'overfitting_detected' in overfit_test.results_, "Overfitting detection result missing."


def test_data_integrity():
    # Sample data with missing values
    data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4],
        'feature2': [np.nan, 2, 3, 4],
        'feature3': ['A', 'B', 'A', 'B']
    })

    # Initialize the DataIntegrity test
    integrity_test = DataIntegrity(
        missing_value_threshold=0.1,
        unique_check_columns=['feature3'],
        data_types={'feature1': float, 'feature2': float},
        range_checks={'feature1': (0, 5)}
    )

    # Run the test
    integrity_test.run(data)

    # Check if the test ran successfully
    assert hasattr(integrity_test, 'issues_'), "DataIntegrity did not produce issues."
    assert isinstance(integrity_test.issues_, list), "Issues should be a list."


def test_bias_detection():
    # Sample model
    model = LogisticRegression()

    # Sample data
    df = pd.DataFrame({
        'feature1': [0, 1, 2, 3],
        'gender': [0, 1, 0, 1],
        'target': [0, 1, 0, 1]
    })
    X = df[['feature1', 'gender']]
    y = df['target']

    def fairness_metric(predictions, y_true, sensitive_values):
        group_0 = sensitive_values == 0
        group_1 = sensitive_values == 1
        acc_0 = accuracy_score(y_true[group_0], predictions[group_0])
        acc_1 = accuracy_score(y_true[group_1], predictions[group_1])
        return abs(acc_0 - acc_1)

    # Initialize the BiasDetection test
    bias_test = BiasDetection(
        model=model,
        sensitive_feature='gender',
        fairness_metric=fairness_metric,
        bias_threshold=0.1
    )

    # Fit the model
    bias_test.fit(X, y)

    # Detect bias
    bias_test.detect_bias(X, y)

    # Check if the test ran successfully
    assert hasattr(bias_test, 'results_'), "BiasDetection did not produce results."
    assert 'is_biased' in bias_test.results_, "Bias detection result missing."


def test_model_version_compliance():
    # Mock model with get_version method
    class MockModel:
        def get_version(self):
            return '1.0.0'

    model = MockModel()

    # Initialize the ModelVersionCompliance test
    version_test = ModelVersionCompliance(
        expected_version='1.0.0',
        allow_minor_version_mismatch=True,
        check_deprecation=True
    )

    # Run the version compliance check
    version_test.run(model)

    # Check if the test ran successfully
    assert hasattr(version_test, 'results_'), "ModelVersionCompliance did not produce results."
    assert 'version_match' in version_test.results_, "Version match result missing."


def test_performance_regression():
    # Sample models
    current_model = DecisionTreeClassifier(random_state=42)
    baseline_model = DecisionTreeClassifier(max_depth=2, random_state=42)

    # Sample data
    X, y = load_iris(return_X_y=True)

    # Initialize the PerformanceRegression test
    regression_test = PerformanceRegression(
        model=current_model,
        baseline_model=baseline_model,
        metrics={'accuracy': accuracy_score},
        threshold=0.01
    )

    # Fit the models
    regression_test.fit(X, y)

    # Evaluate performance
    regression_test.evaluate(X, y)

    # Check if the test ran successfully
    assert hasattr(regression_test, 'results_'), "PerformanceRegression did not produce results."
    assert 'regression_detected' in regression_test.results_, "Regression detection result missing."


def test_ci_integration():
    # Mock CI tool
    class MockCITool:
        def __init__(self, name):
            self.name = name

        def trigger_action(self, project_name, action, timeout=None, **kwargs):
            # Mock response object
            class Response:
                def __init__(self):
                    self.status_code = 200

                def json(self):
                    return {'success': True}

            return Response()

    ci_tool = MockCITool(name='MockCI')

    # Initialize the CIIntegration test
    ci_test = CIIntegration(
        ci_tool=ci_tool,
        project_name="MyProject",
        trigger_action="build"
    )

    # Run the CI action
    ci_test.run()

    # Check if the test ran successfully
    assert hasattr(ci_test, 'results_'), "CIIntegration did not produce results."
    assert ci_test.results_['success'] is True, "CI action did not succeed."

if __name__=='__main__': 
    pytest.main( [__file__])