# -*- coding: utf-8 -*-


import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error

from gofast.estimators.base import DecisionStumpClassifier
from gofast.estimators.base import DecisionStumpRegressor 
from gofast.estimators.boosting import BoostingTreeClassifier
from gofast.estimators.boosting import BoostingTreeRegressor
from gofast.estimators.boosting import HybridBoostingClassifier
from gofast.estimators.boosting import HybridBoostingRegressor


def test_regressor_fit_predict():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    stump = DecisionStumpRegressor()
    stump.fit(X, y)
    predictions = stump.predict(X)
    assert predictions.shape == (100,), "The shape of the predictions should match the number of samples"

def test_regressor_score():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    stump = DecisionStumpRegressor()
    stump.fit(X, y)
    score = stump.score(X, y)
    assert 0 <= score <= 1, "Score should be a valid probability"

def test_regressor_no_fit():
    """ Test that predict throws an error if called before fit. """
    stump = DecisionStumpRegressor()
    X = np.random.randn(10, 1)
    with pytest.raises(Exception):
        stump.predict(X)

def test_classifier_fit_predict():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    stump = DecisionStumpClassifier()
    stump.fit(X, y)
    predictions = stump.predict(X)
    assert len(predictions) == 100, "The number of predictions must be equal to the number of samples"
    assert set(np.unique(predictions)) <= {0, 1}, "Predictions should be binary"

def test_classifier_score():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    stump = DecisionStumpClassifier()
    stump.fit(X, y)
    score = stump.score(X, y)
    assert 0 <= score <= 1, "Score should be a valid probability"

def test_classifier_predict_proba():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    stump = DecisionStumpClassifier()
    stump.fit(X, y)
    proba = stump.predict_proba(X)
    assert proba.shape == (100, 2), "Probability output shape should match (n_samples, n_classes)"
    assert np.allclose(proba.sum(axis=1), 1), "Probabilities must sum to 1"

def test_classifier_no_fit():
    """ Test that predict throws an error if called before fit. """
    stump = DecisionStumpClassifier()
    X = np.random.randn(10, 2)
    with pytest.raises(Exception):
        stump.predict(X)

@pytest.fixture
def data():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    return X, y

def test_hybrid_boosting_classifier_fit_predict(data):
    X, y = data
    
    clf = HybridBoostingClassifier(
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        class_weight=None, 
        ccp_alpha=0.,
        random_state=42
    )
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    
    assert y_pred.shape == y.shape
    
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.8

def test_hybrid_boosting_classifier_incorrect_shape(data):
    X, y = data
    clf = HybridBoostingClassifier(
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        class_weight=None, 
        ccp_alpha=0.,
        random_state=42
    )
    
    with pytest.raises(ValueError):
        clf.fit(X, y.reshape(-1, 1, -1))  # Incorrect shape for y

def test_hybrid_boosting_classifier_predict_before_fit(data):
    X, _ = data
    clf = HybridBoostingClassifier(
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        class_weight=None, 
        ccp_alpha=0.,
        random_state=42
    )
    
    with pytest.raises(Exception):
        clf.predict(X)  # Model is not yet fitted

def test_hybrid_boosting_classifier_predict_proba(data):
    X, y = data
    
    clf = HybridBoostingClassifier(
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        class_weight=None, 
        ccp_alpha=0.,
        random_state=42
    )
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1)

def test_hybrid_boosting_classifier_predict_proba_before_fit(data):
    X, _ = data
    clf = HybridBoostingClassifier(
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        class_weight=None, 
        ccp_alpha=0.,
        random_state=42
    )
    
    with pytest.raises(Exception):
        clf.predict_proba(X)  # Model is not yet fitted

def _normalize_y( y): 
    return ( y-y.min() )/ (y.max() - y.min())

@pytest.fixture
def datar():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    return X, y

def test_hybrid_boosting_regressor_fit_predict(datar):
    X, y = datar
    y= _normalize_y (y)
    reg = HybridBoostingRegressor(
        n_estimators=100, 
        eta0=0.1, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        ccp_alpha=0.,
        random_state=42
    )
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    
    assert y_pred.shape == y.shape
    
    mse = mean_squared_error(y, y_pred)
    assert mse < 0.1

def test_hybrid_boosting_regressor_incorrect_shape(datar):
    X, y = datar
    y= _normalize_y (y)
    reg = HybridBoostingRegressor(
        n_estimators=100, 
        eta0=0.1, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        ccp_alpha=0.,
        random_state=42
    )
    
    with pytest.raises(ValueError):
        reg.fit(X, y.reshape(-1, 1, -1))  # Incorrect shape for y

def test_hybrid_boosting_regressor_predict_before_fit(datar):
    X, _ = datar
    reg = HybridBoostingRegressor(
        n_estimators=100, 
        eta0=0.1, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        ccp_alpha=0.,
        random_state=42
    )
    
    with pytest.raises(Exception):
        reg.predict(X)  # Model is not yet fitted

def test_hybrid_boosting_regressor_score(datar):
    X, y = datar
    y= _normalize_y (y)
    reg = HybridBoostingRegressor(
        n_estimators=100, 
        eta0=0.1, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        ccp_alpha=0.,
        random_state=42
    )
    reg.fit(X, y)
    
    score = reg.score(X, y)

    assert score < 1.  # Mean squared error should be low

             
def test_boosted_tree_regressor_fit_predict():
    # Generate synthetic regression data
    
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    
    y = (y- y.min()) / (y.max() -y.min() ) # normalize 
    # Initialize and fit BoostingTreeRegressor
    reg = BoostingTreeRegressor(n_estimators=100, max_depth=3, eta0=0.1)
    reg.fit(X, y)
    
    # Predict using the trained model
    y_pred = reg.predict(X)
   
    # Ensure the predictions have the correct shape
    assert y_pred.shape == y.shape
    
    # Ensure the model makes reasonable predictions (MSE should be low)
    mse = mean_squared_error(y, y_pred)
    assert mse < 0.1

def test_boosted_tree_regressor_incorrect_shape():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    reg = BoostingTreeRegressor(n_estimators=100, max_depth=3, eta0=0.1)
    
    with pytest.raises(ValueError):
        reg.fit(X, y.reshape(-1, 1, -1))  # Incorrect shape for y

def test_boosted_tree_regressor_predict_before_fit():
    X, _ = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    reg = BoostingTreeRegressor(n_estimators=100, max_depth=3, eta0=0.1)
    
    with pytest.raises(Exception):
        reg.predict(X)  # Model is not yet fitted

def test_boosted_tree_classifier_fit_predict():
    # Generate synthetic classification data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    
    # Initialize and fit BoostingTreeClassifier
    clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
    clf.fit(X, y)
    
    # Predict using the trained model
    y_pred = clf.predict(X)
    
    # Ensure the predictions have the correct shape
    assert y_pred.shape == y.shape
    
    # Ensure the model makes reasonable predictions (accuracy should be high)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.9

def test_boosted_tree_classifier_incorrect_shape():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
    
    with pytest.raises(ValueError):
        clf.fit(X, y.reshape(-1, 1, -1))  # Incorrect shape for y

def test_boosted_tree_classifier_predict_before_fit():
    X, _ = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
    
    with pytest.raises(Exception):
        clf.predict(X)  # Model is not yet fitted

def test_boosted_tree_classifier_predict_proba():
    # Generate synthetic classification data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    
    # Initialize and fit BoostingTreeClassifier
    clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
    clf.fit(X, y)
    
    # Predict probabilities using the trained model
    proba = clf.predict_proba(X)
    
    # Ensure the probabilities have the correct shape
    assert proba.shape == (X.shape[0], 2)
    
    # Ensure the probabilities sum to 1 for each sample
    assert np.allclose(proba.sum(axis=1), 1)

def test_boosted_tree_classifier_predict_proba_before_fit():
    X, _ = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
    
    with pytest.raises(Exception):
        clf.predict_proba(X)  # Model is not yet fitted

if __name__ == "__main__":
    pytest.main([__file__])