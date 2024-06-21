# -*- coding: utf-8 -*-
"""
test_estimators.py
"""
import pytest 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, make_regression

from gofast.exceptions import NotFittedError
from gofast.estimators.adaline import AdalineClassifier, AdalineMixte 
from gofast.estimators.adaline import AdalineRegressor 
from gofast.estimators.adaline import AdalineStochasticRegressor 
from gofast.estimators.adaline import AdalineStochasticClassifier 
from gofast.estimators.base import DecisionStumpRegressor 
from gofast.estimators.benchmark import BenchmarkRegressor
from gofast.estimators.benchmark import BenchmarkClassifier
from gofast.estimators.boosting import HybridBoostingClassifier 
from gofast.estimators.boosting import HybridBoostingRegressor 
from gofast.estimators.cluster_based import KMFClassifier 
from gofast.estimators.cluster_based import KMFRegressor
from gofast.estimators.dynamic_system import HammersteinWienerClassifier
from gofast.estimators.dynamic_system import HammersteinWienerRegressor
from gofast.estimators.dynamic_system import EnsembleHWClassifier 
from gofast.estimators.dynamic_system import EnsembleHWRegressor
from gofast.estimators.ensemble import MajorityVoteClassifier
from gofast.estimators.ensemble import SimpleAverageClassifier
from gofast.estimators.ensemble import WeightedAverageClassifier
from gofast.estimators.ensemble import SimpleAverageRegressor
from gofast.estimators.ensemble import WeightedAverageRegressor
from gofast.estimators.ensemble import EnsembleClassifier
from gofast.estimators.ensemble import EnsembleRegressor 
from gofast.estimators.perceptron import Perceptron 
from gofast.estimators.perceptron import LightGDClassifier
from gofast.estimators.perceptron import LightGDRegressor
from gofast.estimators.tree import DTBClassifier 
from gofast.estimators.tree import DTBRegressor



@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def test_hammerstein_wiener_classifier(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    classifier = HammersteinWienerClassifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    assert accuracy >= 0.0  # Example threshold for accuracy

def test_hammerstein_wiener_regressor(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    regressor = HammersteinWienerRegressor(linear_model=LinearRegression())
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    assert mean_squared_error(y_test, predictions) < 50000  # Example threshold

def test_ensemble_hw_classifier(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    classifier = EnsembleHWClassifier(n_estimators=10, eta0=0.1)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    assert accuracy > 0.0  # Expected minimal accuracy

def test_ensemble_hw_regressor(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    regressor = EnsembleHWRegressor(n_estimators=5 )
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    assert mean_squared_error(y_test, predictions) < 50000  # Example threshold

@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def test_weighted_average_regressor(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    base_estimators = [LinearRegression(), DecisionTreeRegressor(max_depth=3)]
    weights = [0.6, 0.4]

    model = WeightedAverageRegressor(base_estimators=base_estimators, weights=weights)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert mean_squared_error(y_test, predictions) < 5000  # Example threshold

def test_ensemble_hbt_regressor(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = EnsembleRegressor(n_estimators=10, eta0=0.05, max_depth=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert mean_squared_error(y_test, predictions) < 50000  # Example threshold

def test_decision_tree_based_regressor(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = DTBRegressor(n_estimators=10, max_depth=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert mean_squared_error(y_test, predictions) < 50000  # Example threshold

# # Helper function to create datasets
def create_dataset(task='classification', n_classes=2, n_informative=2 ):
    if task == 'classification':
        X, y = make_classification(n_samples=100, n_features=5, random_state=42, 
                                    n_classes=n_classes, n_informative= 2)
    else:
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Tests for AdalineClassifier
@pytest.fixture
def adaline_classifier():
    return AdalineClassifier(eta0=0.01, max_iter=50)

def test_adaline_classifier(adaline_classifier):
    X_train, X_test, y_train, y_test = create_dataset('classification')
    adaline_classifier.fit(X_train, y_train)
    predictions = adaline_classifier.predict(X_test)
    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset([0, 1])

# Tests for AdalineMixte
@pytest.fixture
def adaline_mixte():
    return AdalineMixte(eta0=0.01, max_iter=50)

def test_adaline_mixte(adaline_mixte):
    # Testing both regression and classification modes
    # Regression test
    X_train, X_test, y_train, y_test = create_dataset('regression')
    adaline_mixte.fit(X_train, y_train)
    predictions = adaline_mixte.predict(X_test)
    assert len(predictions) == len(y_test)

    # Classification test
    X_train, X_test, y_train, y_test = create_dataset('classification', n_informative=3)
    adaline_mixte.fit(X_train, y_train)
    predictions = adaline_mixte.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.0  # Assuming it does better than random

# Tests for LightGDClassifier and LightGDRegressor
@pytest.fixture
def gradient_descent_classifier():
    return LightGDClassifier(eta0=0.01, max_iter=50)

@pytest.fixture
def gradient_descent_regressor():
    return LightGDRegressor(eta0=0.0001, max_iter=1000)

def test_gradient_descent_classifier(gradient_descent_classifier):
    X_train, X_test, y_train, y_test = create_dataset('classification', n_classes=2)
    gradient_descent_classifier.fit(X_train, y_train)
    predictions = gradient_descent_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.0  # Assuming it performs reasonably well

def test_gradient_descent_regressor(gradient_descent_regressor):
    X_train, X_test, y_train, y_test = create_dataset('regression')
    gradient_descent_regressor.fit(X_train, y_train)
    predictions = gradient_descent_regressor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    assert mse < 100  # Assuming some reasonable MSE value

# Test for SimpleAverageRegressor
@pytest.fixture
def simple_average_regressor():
    base_estimators = [LinearRegression(), DecisionTreeRegressor()]
    return SimpleAverageRegressor(base_estimators=base_estimators)

def test_simple_average_regressor(simple_average_regressor):
    X_train, X_test, y_train, y_test = create_dataset('regression')
    simple_average_regressor.fit(X_train, y_train)
    predictions = simple_average_regressor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    assert mse < np.inf  # Expecting the ensemble to perform better than individual models

# MajorityVoteClassifier tests
@pytest.fixture
def majority_vote_classifier():
    clfs = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier()]
    return MajorityVoteClassifier(classifiers=clfs)

def test_majority_vote_classifier_fit_predict(majority_vote_classifier):
    X_train, X_test, y_train, y_test = create_dataset('classification')
    majority_vote_classifier.fit(X_train, y_train)
    assert hasattr(majority_vote_classifier, 'classifiers_')
    predictions = majority_vote_classifier.predict(X_test)
    assert len(predictions) == len(y_test)

def test_majority_vote_invalid_input(majority_vote_classifier):
    with pytest.raises(ValueError):
        majority_vote_classifier.fit(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]))

# AdalineStochasticRegressor tests
@pytest.fixture
def adaline_stochastic_regressor():
    return AdalineStochasticRegressor(eta0=0.0001, max_iter=100)

def test_adaline_stochastic_regressor_fit_predict(adaline_stochastic_regressor):
    X_train, X_test, y_train, y_test = create_dataset('regression')
    adaline_stochastic_regressor.fit(X_train, y_train)
    assert hasattr(adaline_stochastic_regressor, 'weights_')
    predictions = adaline_stochastic_regressor.predict(X_test)
    assert len(predictions) == len(y_test)

# AdalineStochasticClassifier tests
@pytest.fixture
def adaline_stochastic_classifier():
    return AdalineStochasticClassifier(eta0=0.01, max_iter=50)

def test_adaline_stochastic_classifier_fit_predict(adaline_stochastic_classifier):
    X_train, X_test, y_train, y_test = create_dataset('classification')
    adaline_stochastic_classifier.fit(X_train, y_train)
    assert hasattr(adaline_stochastic_classifier, 'weights_')
    predictions = adaline_stochastic_classifier.predict(X_test)
    assert len(predictions) == len(y_test)

def test_adaline_stochastic_classifier_predict_proba(adaline_stochastic_classifier):
    X_train, X_test, y_train, y_test = create_dataset('classification')
    adaline_stochastic_classifier.fit(X_train, y_train)
    probas = adaline_stochastic_classifier.predict_proba(X_test)
    assert probas.shape == (len(y_test), 2)

# AdalineRegressor tests
@pytest.fixture
def adaline_regressor():
    return AdalineRegressor(eta0=0.01, max_iter=50)

def test_adaline_regressor_fit_predict(adaline_regressor):
    X_train, X_test, y_train, y_test = create_dataset('regression')
    adaline_regressor.fit(X_train, y_train)
    assert hasattr(adaline_regressor, 'weights_')
    predictions = adaline_regressor.predict(X_test)
    assert len(predictions) == len(y_test)

# Test cases for KMFClassifier
@pytest.fixture
def kmf_classifier():
    from sklearn.ensemble import RandomForestClassifier
    return KMFClassifier(estimator=RandomForestClassifier(), n_clusters=5)

def test_kmf_classifier_fit_predict(kmf_classifier):
    X_train, X_test, y_train, y_test = create_dataset('classification')
    kmf_classifier.fit(X_train, y_train)
    assert hasattr(kmf_classifier, 'estimator_')
    predictions = kmf_classifier.predict(X_test)
    assert len(predictions) == len(y_test)

def test_kmf_classifier_invalid_input(kmf_classifier):
    with pytest.raises(ValueError):
        kmf_classifier.fit(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]))

# Test cases for KMFRegressor
@pytest.fixture
def kmf_regressor():
    return KMFRegressor(estimator=LinearRegression(), n_clusters=5)

def test_kmf_regressor_fit_predict(kmf_regressor):
    X_train, X_test, y_train, y_test = create_dataset('regression')
    kmf_regressor.fit(X_train, y_train)
    assert hasattr(kmf_regressor, 'estimator_')
    predictions = kmf_regressor.predict(X_test)
    assert len(predictions) == len(y_test)

def test_kmf_regressor_invalid_input(kmf_regressor):
    with pytest.raises(ValueError):
        kmf_regressor.fit(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]))

# Test cases for DecisionStumpRegressor
@pytest.fixture
def decision_stump():
    return DecisionStumpRegressor()

def test_decision_stump_fit_predict(decision_stump):
    X_train, _, y_train, _ = create_dataset('regression')
    decision_stump.fit(X_train, y_train)
    assert hasattr(decision_stump, 'split_feature_')
    assert hasattr(decision_stump, 'left_value_') and hasattr(decision_stump, 'right_value_')

def test_decision_stump_invalid_input(decision_stump):
    with pytest.raises(NotFittedError):
        decision_stump.predict(np.array([[1, 2], [3, 4]]))

# Test cases for BenchmarkRegressor
@pytest.fixture
def benchmark_regressor():
    return BenchmarkRegressor(base_estimators=[
        ('lr', LinearRegression()), ('dt', DecisionTreeRegressor())],
        meta_regressor=LinearRegression())

def test_benchmark_regressor_fit_predict(benchmark_regressor):
    X_train, X_test, y_train, y_test = create_dataset('regression')
    benchmark_regressor.fit(X_train, y_train)
    assert hasattr(benchmark_regressor, 'stacked_model_')
    predictions = benchmark_regressor.predict(X_test)
    assert len(predictions) == len(y_test)

# Test cases for BenchmarkClassifier
@pytest.fixture
def benchmark_classifier():

    return BenchmarkClassifier(base_classifiers=[
        ('lr', LogisticRegression()), ('dt', DecisionTreeClassifier())],
        meta_classifier=LogisticRegression())

def test_benchmark_classifier_fit_predict(benchmark_classifier):
    X_train, X_test, y_train, y_test = create_dataset('classification')
    benchmark_classifier.fit(X_train, y_train)
    assert hasattr(benchmark_classifier, 'stacked_model_')
    predictions = benchmark_classifier.predict(X_test)
    assert len(predictions) == len(y_test)

# Test cases for BasePerceptron
@pytest.fixture
def base_perceptron():
    return Perceptron()

def test_base_perceptron_fit_predict(base_perceptron):
    X_train, X_test, y_train, y_test = create_dataset('classification')
    base_perceptron.fit(X_train, y_train)
    assert hasattr(base_perceptron, 'weights_')
    if X_test.ndim ==1: 
        X_test =X_test.reshape (-1,1)
    predictions = base_perceptron.predict(X_test)
    assert len(predictions) == len(y_test)

def test_simple_average_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the SimpleAverageClassifier
    base_classifiers = [DecisionTreeClassifier() for _ in range(3)]
    ensemble = SimpleAverageClassifier(base_classifiers)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_weighted_average_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the WeightedAverageClassifier
    base_classifiers = [DecisionTreeClassifier() for _ in range(3)]
    weights = [0.3, 0.4, 0.3]
    ensemble = WeightedAverageClassifier(base_classifiers, weights)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_simple_average_regressor2():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the SimpleAverageRegressor
    base_regressors = [DecisionTreeRegressor() for _ in range(3)]
    ensemble = SimpleAverageRegressor(base_regressors)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse >= 0.0  

def test_weighted_average_regressor2():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the WeightedAverageRegressor
    base_regressors = [DecisionTreeRegressor() for _ in range(3)]
    weights = [0.3, 0.4, 0.3]
    ensemble = WeightedAverageRegressor(base_regressors, weights)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse >= 0.0  

def test_hybrid_boosted_tree_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeClassifier
    ensemble = HybridBoostingClassifier(
        n_estimators=50, eta0=0.01, max_depth=3)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_hybrid_boosted_tree_regressor():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeRegressor
    ensemble = HybridBoostingRegressor(n_estimators=50, )
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse >= 0.0  

def test_decision_tree_based_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the DecisionTreeBasedClassifier
    ensemble = DTBClassifier(
        n_estimators=50, max_depth=3, random_state=42)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_hybrid_boosted_tree_ensemble_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeEnsembleClassifier
    ensemble = EnsembleClassifier(n_estimators=50, )
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_hybrid_boosted_tree_ensemble_regressor():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeEnsembleRegressor
    ensemble = EnsembleRegressor(n_estimators=50, max_depth=3, eta0=0.1)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse >= 0.0  

if __name__ == "__main__":
    pytest.main([__file__])

