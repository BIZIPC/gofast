# -*- coding: utf-8 -*-
"""
test_metrics.py

@author: LKouadio ~@Daniel
"""
import numpy as np
import pytest
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression

from gofast.metrics import assess_classifier_metrics 
from gofast.metrics import assess_regression_metrics  
from gofast.metrics import mean_squared_log_error, balanced_accuracy
from gofast.metrics import information_value, geo_information_value 
from gofast.metrics import mae_flex, mse_flex, rmse_flex, r2_flex
from gofast.metrics import adjusted_r2_score  
from gofast.metrics import precision_recall_tradeoff, roc_tradeoff
from gofast.metrics import evaluate_confusion_matrix, display_precision_recall
from gofast.metrics import display_confusion_matrix  
from gofast.metrics import likelihood_score, log_likelihood_score 
from gofast.metrics import mean_absolute_percentage_error
from gofast.metrics import explained_variance_score, median_absolute_error
from gofast.metrics import max_error_score, mean_absolute_deviation
from gofast.metrics import mean_poisson_deviance, mean_gamma_deviance
from gofast.metrics import madev_flex, dice_similarity_score, gini_score 
from gofast.metrics import hamming_loss, fowlkes_mallows_score
from gofast.metrics import root_mean_squared_log_error, mean_percentage_error
from gofast.metrics import spearmans_rank_score, precision_at_k, ndcg_at_k
from gofast.metrics import ndcg_at_k_with_controls, mean_reciprocal_score
from gofast.metrics import jaccard_flex, balanced_accuracy_score

def test_ndcg_at_k_with_controls():
    y_true = [[3, 2, 3], [2, 1, 2]]
    y_pred = [[1, 2, 3], [1, 2, 3]]
    k = 3
    # Test basic functionality
    result = ndcg_at_k_with_controls(y_true, y_pred, k)
    assert isinstance(result, float)
    assert 0 <= result <= 1

    # Test with NaN policy
    y_pred_with_nan = [[np.nan, 2, 3], [1, np.nan, 3]]
    with pytest.raises(ValueError):
        ndcg_at_k_with_controls(y_true, y_pred_with_nan, k, nan_policy='raise')

def test_mean_reciprocal_score():
    y_true = [2, 1]
    y_pred = [[1, 2, 3], [1, 2, 3]]
    # Basic functionality
    result = mean_reciprocal_score(y_true, y_pred)
    assert isinstance(result, float)
    assert result > 0

    # Test zero division handling
    y_true_zero = [4]  # no matching rank
    y_pred_zero = [[1, 2, 3]]
    with pytest.warns(UserWarning):
        result_zero = mean_reciprocal_score(y_true_zero, y_pred_zero, zero_division='warn')
    assert result_zero == 0

def test_jaccard_flex():
    y_true = [0, 1, 2, 1, 2, 0]
    y_pred = [0, 2, 1, 0, 0, 1]
    # Test basic functionality
    result = jaccard_flex(y_true, y_pred, average='macro')
    assert hasattr ( result, 'score')  
    assert 0 <= result.score <= 1

    # Test detailed output
    result_detailed = jaccard_flex(y_true, y_pred, average='macro', detailed=True)
    assert 'intersection' in result_detailed
    assert 'union' in result_detailed

def test_balanced_accuracy_score():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    # Basic functionality
    result = balanced_accuracy_score(y_true, y_pred)
    assert isinstance(result, float)
    assert 0 <= result <= 1

    # Strategy parameter
    y_true_multi = [0, 1, 2, 2, 1]
    y_pred_multi = [0, 2, 2, 1, 1]
    result_ovr = balanced_accuracy_score(
        y_true_multi, y_pred_multi, strategy='ovr')
    result_ovo = balanced_accuracy_score(
        y_true_multi, y_pred_multi, strategy='ovo')
    assert (0 <= result_ovr <=1)  ==  (0 <= result_ovo <=1)

    # Test with sample weights
    weights = [0.5, 1, 1, 1, 1]
    weighted_result = balanced_accuracy_score(
        y_true_multi, y_pred_multi, sample_weight=weights)
    assert weighted_result != result_ovr

def test_root_mean_squared_log_error():
    y_true = np.array([3, 5, 7.5, 10])
    y_pred = np.array([2.5, 5, 7, 9.5])
    # Expected to run without errors
    assert np.isclose(root_mean_squared_log_error(y_true, y_pred), 0.0741, atol=1e-2)

    # Test with zero values where clipping should prevent log(0)
    y_true = np.array([0, 5, 7.5, 10])
    y_pred = np.array([0, 5, 7, 9.5])
    assert root_mean_squared_log_error(y_true, y_pred, clip_value=1) > 0

def test_mean_percentage_error():
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 210, 240])
    # Test typical input
    assert mean_percentage_error(y_true, y_pred) == pytest.approx(1.0867, 0.01)

    # Handling zero division
    y_true = np.array([0, 150, 200, 250])
    with pytest.warns(RuntimeWarning):
        assert np.isnan(mean_percentage_error(y_true, y_pred))

def test_spearmans_rank_score():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 6, 7, 8, 7])
    assert spearmans_rank_score(y_true, y_pred) == pytest.approx(0.8208, 0.01)

    # Test with NaNs
    y_true = np.array([1, 2, np.nan, 4])
    y_pred = np.array([4, np.nan, 3, 1])
    assert not np.isnan(spearmans_rank_score(y_true, y_pred, nan_policy='omit'))

def test_precision_at_k():
    y_true = [[1, 2, 3], [1, 2, 3]]
    y_pred = [[2, 1, 4], [1, 2, 3]]
    k = 2
    assert precision_at_k(y_true, y_pred, k) == 1.0

    # Test invalid k
    with pytest.raises(ValueError):
        precision_at_k(y_true, y_pred, k=5)

def test_ndcg_at_k():
    y_true = [[1, 2, 3], [1, 2, 3]]
    y_pred = [[2, 1, 4], [1, 2, 3]]
    k = 2
    assert ndcg_at_k(y_true, y_pred, k) > 0

    # Test with incomplete lists
    y_pred = [[2], [1]]
    assert ndcg_at_k(y_true, y_pred, k) < 1

def test_hamming_loss2():
    y_true = np.array([[1, 1], [1, 0]])
    y_pred = np.array([[1, 0], [1, 1]])
    assert hamming_loss(y_true, y_pred) == 0.5

    # Test normalization
    assert hamming_loss(y_true, y_pred, normalize=False) == 1

def test_madev_flex():
    data = np.array([1, 2, 3, 4, 5])
    result = madev_flex(data)
    assert np.isclose(result.score, 1.2), "MADEv calculation error"

def test_dice_similarity_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0])
    result = dice_similarity_score(y_true, y_pred)
    expected_score = 2 * (2 / (3 + 3))
    assert np.isclose(result, expected_score), "Dice similarity score calculation error"

def test_gini_score():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    result = gini_score(y_true, y_pred)
    expected =  np.sum (np.abs(np.subtract.outer(y_true, y_pred))) /(
        2 * len(y_true) * np.sum(y_true)) 
    assert np.isclose(result, expected), "Gini score calculation error for identical inputs"

def test_hamming_loss():
    y_true = np.array([[1, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1]])
    result = hamming_loss(y_true, y_pred, normalize=True)
    assert np.isclose(result, 0.25), "Hamming loss calculation error"

def test_fowlkes_mallows_score():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 2, 1, 2])
    result = fowlkes_mallows_score(y_true, y_pred)
    # Calculating expected FMI
    tp = 2  # Two true positives as no cluster pairs are identical
    fp = 2  # Two instances were placed in different clusters in y_true but the same in y_pred
    fn = 2  # Two instances were placed in the same cluster in y_true but different in y_pred
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 
    expected_fmi = np.sqrt(precision * recall)
    assert np.isclose(result, expected_fmi), "Fowlkes-Mallows score calculation error"


def test_mean_poisson_deviance():
    y_true = np.array([3, 5, 2, 7])
    y_pred = np.array([2, 6, 3, 8])
    expected_value= 8.7
    # Test normal case
    assert np.isclose(
        mean_poisson_deviance(y_true, y_pred), expected_value, atol=1e-1)

    # Test with zeros and epsilon adjustment
    y_pred = np.array([0, 0, 0, 0])
    assert mean_poisson_deviance(y_true, y_pred, zero_division='ignore') is not np.nan

    # Test with sample weights
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    weighted_deviance = mean_poisson_deviance(y_true, y_pred, sample_weight=weights)
    assert weighted_deviance is not None

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        mean_poisson_deviance(y_true, "invalid_pred")

def test_mean_gamma_deviance():
    y_true = np.array([0.5, 1.5, 2.0, 2.5])
    y_pred = np.array([0.5, 1.5, 2.0, 2.5])

    # Assuming this should pass as they are the same
    assert np.isclose(mean_gamma_deviance(y_true, y_pred), 0, atol=1e-4)

    # Test with clip value
    assert mean_gamma_deviance(y_true, y_pred, clip_value=0.1) is not np.nan

    # Test with NaN policy
    y_true[1] = np.nan
    assert mean_gamma_deviance(y_true, y_pred, nan_policy='omit') is not np.nan

def test_mean_absolute_deviation():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([10, 20, 30, 40])

    # Perfect prediction
    assert mean_absolute_deviation(y_true, y_pred) == 0

    # With NaNs
    y_true = np.array([np.nan, 20, np.nan, 40])
    y_pred = np.array([10, 20, 30, 40])
    assert mean_absolute_deviation(y_true, y_pred, nan_policy='omit') == 0


def test_max_error_score_basic():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    max_error = np.max(np.abs(y_true - y_pred))
    assert np.isclose(max_error_score(y_true, y_pred), max_error)

def test_max_error_score_with_weights():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, -1, 1, 9])
    weights = np.array([1, 1, 1, 2])
    weighted_errors = np.abs(y_true - y_pred) * weights
    max_error = np.max(weighted_errors)
    assert np.isclose(max_error_score(y_true, y_pred, sample_weight=weights), max_error)

def test_max_error_score_multioutput():
    y_true = np.array([[1, 2], [3, 4], [5, 6]])
    y_pred = np.array([[1.1, 1.9], [2.9, 4.1], [5.1, 5.9]])
    max_error = np.max(np.abs(y_true - y_pred), axis=0)
    result = max_error_score(y_true, y_pred, multioutput='raw_values')
    assert np.allclose(result, max_error)

def test_max_error_score_uniform_average():
    y_true = np.array([[1, 2], [3, 4], [5, 6]])
    y_pred = np.array([[1.1, 1.9], [2.9, 4.1], [5.1, 5.9]])
    max_error = np.mean(np.max(np.abs(y_true - y_pred), axis=0))
    result = max_error_score(y_true, y_pred, multioutput='uniform_average')
    assert np.isclose(result, max_error)

def test_max_error_nan_policy():
    y_true = np.array([3, np.nan, 2, 7])
    y_pred = np.array([2.5, 2.0, 2, 8])
    # Expecting test to ignore NaN and proceed with other values
    max_error = np.nanmax(np.abs(y_true - y_pred))
    result = max_error_score(y_true, y_pred, nan_policy='omit')
    assert np.isclose(result, max_error)

def test_median_absolute_error_basic():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    absolute_errors = np.abs(y_true - y_pred)
    expected_medae = np.median(absolute_errors)
    assert np.isclose(median_absolute_error(y_true, y_pred), expected_medae)

def test_median_absolute_error_with_weights(): # 1
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    weights = np.array([1, 2, 1, 2])
    expected_medae = .5 # Adjust calculation as needed
    assert np.isclose(
        median_absolute_error(y_true, y_pred, sample_weight=weights),
        expected_medae, )

def test_median_absolute_error_multioutput():
    y_true = np.array([[1, 2], [3, 4]])
    y_pred = np.array([[1.1, 1.9], [2.9, 4.1]])
    result = median_absolute_error(y_true, y_pred, multioutput='raw_values')
    expected_result = np.median(np.abs(y_true - y_pred), axis=0)
    assert np.allclose(result, expected_result)

def test_explained_variance_score_basic():
    y_true = np.array([3, 5, 2.5, 7])
    y_pred = np.array([2.5, 5, 2, 8])
    var_res = np.var(y_true - y_pred)
    var_true = np.var(y_true)
    expected_evs = 1 - var_res / var_true
    assert np.isclose(explained_variance_score(y_true, y_pred), expected_evs)

def test_explained_variance_zero_division_warn(): # 2
    y_true = np.array([4, 4, 4, 4])
    y_pred = np.array([3, 3, 3, 3])
    with pytest.warns(RuntimeWarning):
        result = explained_variance_score(y_true, y_pred, zero_division='warn')
    assert result == 1. 

def test_explained_variance_zero_division_raise():
    y_true = np.array([4, 4, 4, 4])
    y_pred = np.array([3, 3, 3, 3])
    with pytest.raises(ZeroDivisionError):
        explained_variance_score(y_true, y_pred, zero_division='raise')

def test_basic_mape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 205, 295])
    expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    assert np.isclose(mean_absolute_percentage_error(y_true, y_pred), expected_mape)

def test_mape_with_zero_division_warn(): # 3
    y_true = np.array([0, 200, 300])
    y_pred = np.array([10, 200, 300])
    with pytest.warns(RuntimeWarning):
        result =mean_absolute_percentage_error(y_true, y_pred, zero_division='warn')
    assert np.isnan(result)

def test_mape_with_zero_division_raise():
    y_true = np.array([0, 200, 300])
    y_pred = np.array([10, 200, 300])
    with pytest.raises(ValueError):
        mean_absolute_percentage_error(y_true, y_pred, zero_division='raise')

def test_mape_with_sample_weight():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 205, 295])
    sample_weight = np.array([0.5, 1.5, 1.0])
    expected_mape = np.average(np.abs((y_true - y_pred) / y_true), weights=sample_weight) * 100
    assert np.isclose(mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight), expected_mape)

def test_mape_multioutput_raw_values(): # 4
    y_true = np.array([[100, 200], [200, 300]])
    y_pred = np.array([[90, 210], [190, 295]])
    result = mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    expected_result = np.array([np.mean(np.abs((y_true[:, 0] - y_pred[:, 0]) / y_true[:, 0])),
                                np.mean(np.abs((y_true[:, 1] - y_pred[:, 1]) / y_true[:, 1]))]) * 100
    assert np.allclose(result, expected_result)

def test_mape_multioutput_uniform_average():
    y_true = np.array([[100, 200], [200, 300]])
    y_pred = np.array([[90, 210], [190, 295]])
    result = mean_absolute_percentage_error(y_true, y_pred, multioutput='uniform_average')
    expected_result = np.mean([np.mean(np.abs((y_true[:, 0] - y_pred[:, 0]) / y_true[:, 0])),
                                np.mean(np.abs((y_true[:, 1] - y_pred[:, 1]) / y_true[:, 1]))]) * 100
    assert np.isclose(result, expected_result)

# Run tests with pytest from the command line

def test_log_likelihood_positive_consensus():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 1, 2, 1, 0])
    result = log_likelihood_score(y_true, y_pred, consensus='positive', strategy='ovr')
    assert np.isclose(result, 1.2296264763713125), ( 
        "The LLR calculation does not match expected output."
        )

def test_log_likelihood_negative_consensus():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 0])
    result = log_likelihood_score(y_true, y_pred, consensus='negative', strategy='ovr')
    assert result < 10., "Negative consensus LLR should be less than 10." # CAP value of LLR

def test_likelihood_score_simple_binary():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 1, 1, 0, 0]
    result = likelihood_score(y_true, y_pred)
    expected = 1.3333222223259247
    assert np.isclose(result, expected), ( 
        "Likelihood score for binary classification did not match."
        )

def test_likelihood_score_with_weights():
    y_true = [0, 1, 2, 2, 1, 0]
    y_pred = [0, 2, 1, 2, 1, 0]
    weights = [0.5, 1, 1, 0.5, 1, 1]
    result = likelihood_score(y_true, y_pred, sample_weight=weights, strategy='ovr')
    # Expected result needs to be calculated based on the expected behavior
    assert isinstance(result, float), ( 
        "Result should be a float representing the weighted likelihood ratio."
        )

def test_zero_division_warning():
    y_true = [1, 1, 1, 1]
    y_pred = [0, 0, 0, 0]  # Completely incorrect prediction causing zero division in specificity
    with pytest.warns(RuntimeWarning):
        log_likelihood_score(y_true, y_pred, zero_division='warn')

def test_detailed_output():
    y_true = [0, 1, 1, 0, 0]
    y_pred = [0, 0, 1, 0, 1]
    result = likelihood_score(y_true, y_pred, detailed_output=True)
    assert hasattr(result, 'lr_score') and hasattr(result, 'sensitivity'
                                                    ) and hasattr(result, 'specificity'), \
        "Detailed output should include lr_score, sensitivity, and specificity."

def test_binary_classification_positive():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0])
    assert np.isclose(likelihood_score(
        y_true, y_pred, consensus='positive', epsilon=0.001), 1.5, atol=0.3)

def test_binary_classification_negative():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0])
    assert np.isclose(likelihood_score(
        y_true, y_pred, consensus='negative', epsilon=0.001), 0.633, atol=0.3)

def test_multiclass_classification_ovr():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 2, 0])
    # Assume hypothetical expected result, 
    # verify with actual expected value
    expected_result = 4.
    assert np.isclose(likelihood_score(y_true, y_pred, strategy='ovr'
                          ), expected_result, atol=0.1)

@pytest.fixture
def binary_classification_data():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def trained_model(binary_classification_data):
    X_train, _, y_train, _ = binary_classification_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def test_precision_recall_tradeoff(binary_classification_data, trained_model):
    _, X_test, _, y_test = binary_classification_data
    # Assuming y_scores are provided
    y_scores = trained_model.decision_function(X_test)
    result = precision_recall_tradeoff(y_test, y_scores=y_scores, display_chart=False)
    assert 'f1_score' in result and 'precision_score' in result and 'recall_score' in result,( 
        "Required metrics are missing")

def test_roc_tradeoff(binary_classification_data, trained_model):
    _, X_test, _, y_test = binary_classification_data
    y_scores = trained_model.decision_function(X_test)
    result = roc_tradeoff(y_test, y_scores=y_scores, display_chart=False)
    assert 'auc_score' in result, "AUC score missing from results"

def test_evaluate_confusion_matrix(binary_classification_data, trained_model):
    _, X_test, _, y_test = binary_classification_data
    result = evaluate_confusion_matrix(y_test, classifier=trained_model, X=X_test, display=False)
    assert 'confusion_matrix' in result, "Confusion matrix missing from results"

def test_display_precision_recall_does_not_raise_error():
    precisions, recalls, thresholds = precision_recall_curve([0, 1], [0.1, 0.9])
    try:
        display_precision_recall(precisions, recalls, thresholds)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")

def test_display_confusion_matrix_does_not_raise_error():
    cm = confusion_matrix([0, 1], [1, 1])
    try:
        display_confusion_matrix(cm, labels=['Neg', 'Pos'])
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")

def test_geo_information_value_basic():
    Xp = np.array([1, 2, 3])
    Np = np.array([100, 200, 300])
    Sp = np.array([10, 20, 30])
    geo_iv = geo_information_value(Xp, Np, Sp, aggregate=True)
    assert isinstance(geo_iv, float)
    assert geo_iv >= 0, "Geo-IV should be positive for this input"

def test_geo_information_value_aggregate_false():
    Xp = np.array([1, 2, 3])
    Np = np.array([100, 200, 300])
    Sp = np.array([10, 20, 30])
    geo_iv = geo_information_value(Xp, Np, Sp, aggregate=False)
    assert len(geo_iv) == len(Xp), "The output should match the number of causative factors"
    assert all(iv >= 0 for iv in geo_iv), "All Geo-IV values should be positive for this input"

def test_geo_information_value_clip_upper_bound():
    Xp = np.array([1, 2])
    Np = np.array([1e-16, 200])  # Extremely small value that should be clipped
    Sp = np.array([1, 20])
    geo_iv = geo_information_value(Xp, Np, Sp, clip_upper_bound=100, aggregate=True)
    assert isinstance(geo_iv, float), "Output should be a float when aggregate is True"

def test_adjusted_r2_score_basic():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    X = np.random.rand(4, 2)  # Assuming 2 predictors
    adjusted_r2 = adjusted_r2_score(y_true, y_pred, X)
    basic_r2 = r2_score(y_true, y_pred)
    assert adjusted_r2 < basic_r2, "Adjusted R² should be less than R² for this example"

def test_adjusted_r2_score_negative_denominator():
    y_true = np.array([1, 2])
    y_pred = np.array([1.1, 1.9])
    X = np.random.rand(2, 5)  # More predictors than samples
    with pytest.warns(UserWarning):
        adjusted_r2 = adjusted_r2_score(y_true, y_pred, X)
    assert np.isnan(adjusted_r2), "Adjusted R² should be NaN when denominator is negative or zero"

def test_assess_regression_metrics():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = assess_regression_metrics(y_test, model=model, X=X_test)
    # return Bunch object 
    assert 'mean_absolute_error' in metrics, "Dictionary should contain MAE"
    assert 'mean_squared_error' in metrics, "Dictionary should contain MSE"
    assert 'rmse' in metrics, "Dictionary should contain RMSE"
    assert 'r2_score' in metrics, "Dictionary should contain R² score"
    assert metrics.mean_absolute_error >= 0, "MAE should be non-negative"
    assert metrics.mean_squared_error >= 0, "MSE should be non-negative"
    assert metrics.rmse >= 0, "RMSE should be non-negative"
    assert -1 <= metrics.r2_score <= 1, "R² score should be within [-1, 1]"

def test_assess_classifier_metrics():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    metrics = assess_classifier_metrics(y_test, model=model, X=X_test, average='macro')

    assert 'accuracy_score' in metrics, "Dictionary should contain accuracy score"
    assert 'recall_score' in metrics, "Dictionary should contain recall score"
    assert 'precision_score' in metrics, "Dictionary should contain precision score"
    assert 'f1_score' in metrics, "Dictionary should contain F1 score"
    # ROC AUC might be skipped in multiclass scenarios without predict_proba
    # assert 'roc_auc_score' in metrics, "Dictionary should contain ROC-AUC score"
    assert 0 <= metrics.accuracy_score <= 1, "Accuracy score should be within [0, 1]"
    assert 0 <= metrics.recall_score <= 1, "Recall score should be within [0, 1]"
    assert 0 <= metrics.precision_score <= 1, "Precision score should be within [0, 1]"
    assert 0 <= metrics.f1_score <= 1, "F1 score should be within [0, 1]"

# Sample data for testing
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
def test_flexible_mae():
    result = mae_flex(y_true, y_pred)
    assert 'score' in result
    assert result.score >= 0  # MAE should always be non-negative

def test_flexible_mse():
    result = mse_flex(y_true, y_pred)
    assert 'score' in result
    assert result.score>= 0  # MSE should always be non-negative

def test_flexible_rmse():
    result = rmse_flex(y_true, y_pred)
    assert 'score' in result
    assert result.score >= 0  # RMSE should always be non-negative

def test_flexible_r2():
    result = r2_flex(y_true, y_pred)
    assert  'score' in result
    assert -1 <= result.score <= 1  # R2 should be in the range [-1, 1]

    # Test with adjustment for the number of predictors
    result_adjusted = r2_flex(y_true, y_pred, adjust_for_n=True, n_predictors=1)
    assert 'adjusted_score' in result_adjusted
    # adjusted R2 should also be in the range [-1, 1]
    assert -1 <= result_adjusted.adjusted_score <= 1 

def test_mean_squared_log_error_basic():
    y_true = np.array([3, 5, 2.5, 7])
    y_pred = np.array([2.5, 5, 4, 8])
    expected_msle = 0.03973  
    msle = mean_squared_log_error(y_true, y_pred)
    assert np.isclose(msle, expected_msle, atol=1e-4), f"Expected {expected_msle}, got {msle}"

def test_mean_squared_log_error_with_clip():
    y_true = [3, 5, 2.5, 1]
    y_pred = [2.5, 5, 4, -8] # this is simulation that not fit the reality for non-negative values. 
    msle = mean_squared_log_error(y_true, y_pred, clip_value=0.01)
    expected_msle = 0.15295
    assert np.isclose(msle, expected_msle, atol=1e-4), f"Expected {expected_msle}, got {msle}"

def test_mean_squared_log_error_negative_values():
    y_true = [3, 5, 2.5, -7]
    y_pred = [2, -5, 4, 8]
    with pytest.raises(ValueError):
        mean_squared_log_error(y_true, y_pred)

def test_balanced_accuracy_binary():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    expected_ba = 0.75
    ba = balanced_accuracy(y_true, y_pred)
    assert np.isclose(ba, expected_ba, atol=1e-4), f"Expected {expected_ba}, got {ba}"

def test_balanced_accuracy_multiclass_ovr():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 1, 1]
    expected_ba = 0.666  
    ba = balanced_accuracy(y_true, y_pred, strategy='ovr')
    assert np.isclose(ba, expected_ba, atol=1e-2), f"Expected {expected_ba}, got {ba}"

def test_balanced_accuracy_invalid_strategy():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 1, 1]
    with pytest.raises(ValueError):
        balanced_accuracy(y_true, y_pred, strategy='invalid')

def test_information_value_binary_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.7, 0.2])
    iv = information_value(y_true, y_pred, problem_type='binary', scale=None)

    assert isinstance(iv, float), "IV should be a float"
    assert iv > 0, "IV should be positive for binary classification with these inputs"

def test_information_value_multiclass_classification():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.1, 0.7, 0.2]])
    iv = information_value(y_true, y_pred, problem_type='multiclass', scale='binary_scale')
    assert isinstance(iv, float), "IV should be a float"
    assert iv < 0, ( 
        "IV should be negative for multiclass classification"
        " with these inputs and binary_scale"
        )

def test_information_value_multilabel_classification():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.8, 0.2], [0.4, 0.6]])
    iv = information_value(y_true, y_pred, problem_type='multilabel', scale='none')

    assert isinstance(iv, float), "IV should be a float"
    assert iv > 0, "IV should be positive for multilabel classification with"
    " these inputs and binary_scale"

def test_information_value_binary_classification2():
    y_true_binary = np.array([0, 1, 0, 1, 1, 0])
    y_pred_binary = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    iv = information_value(y_true_binary, y_pred_binary, problem_type='binary', )
    assert isinstance(iv, float), "IV should be a float"
    assert iv > 0, "IV should be positive for binary classification with these inputs"

def test_information_value_multilabel_classification2():
    y_true_multilabel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1]
    ])
    y_pred_multilabel = np.array([
        [0.8, 0.1, 0.7],
        [0.2, 0.9, 0.2],
        [0.6, 0.2, 0.8],
        [0.1, 0.3, 0.9]
    ])
    iv = information_value(y_true_multilabel, y_pred_multilabel,
                            problem_type='multilabel', scale='binary_scale')
    assert isinstance(iv, float), "IV should be a float"
    # Here, the expectation for IV being positive or negative depends 
    # on  interpretation.
    # If we expect IV to reflect predictive power similarly across problem
    # types, consider if the IV calculation
    # logic inversely correlates the IV value with predictive power
    # for multilabel classification.
    assert iv > 0, ( "IV should be positive for multilabel classification"
                    " with these inputs and binary_scale")

def test_information_value_regression():
    y_true = np.array([2.5, 0.0, 2.0, 8.0])
    y_pred = np.array([3.0, 0.5, 2.0, 7.5])
    iv = information_value(y_true, y_pred, problem_type='regression')
    assert isinstance(iv, float), "IV should be a float"
    assert iv < 0, "IV should be negative for regression with these inputs"

def test_information_value_invalid_problem_type():
    y_true = [0, 1]
    y_pred = [0.5, 0.5]
    with pytest.raises(ValueError):
        information_value(y_true, y_pred, problem_type='invalid_type')


if __name__=='__main__': 
    pytest.main([__file__])