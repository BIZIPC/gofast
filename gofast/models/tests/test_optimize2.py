# -*- coding: utf-8 -*-

import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from gofast.models.optimize import OptimizeHyperparams, Optimizer
from gofast.models.optimize import Optimizer2, ParallelizeSearch, OptimizeSearch

# Fixtures for data
@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Fixtures for estimators and param_grids
@pytest.fixture
def estimators_and_param_grids():
    estimators = {
        'SVC': SVC(),
        'SGDClassifier': SGDClassifier()
    }
    param_grids = {
        'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
        'SGDClassifier': {'max_iter': [100, 500], 'alpha': [0.0001, 0.001]}
    }
    return estimators, param_grids

@pytest.fixture
def single_estimator_and_param_grid():
    estimator = SVC()
    param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
    return estimator, param_grid

@pytest.fixture
def single_tree_estimator_and_param_grid():
    estimator = DecisionTreeClassifier()
    param_grid = {'max_depth': [3, 5, None], 'criterion': ['gini', 'entropy']}
    return estimator, param_grid

def test_optimize_hyperparams(data, single_estimator_and_param_grid):
    X_train, X_test, y_train, y_test = data
    estimator, param_grid = single_estimator_and_param_grid
    optimizer = OptimizeHyperparams(estimator, param_grid, strategy='GSCV', n_jobs=1)
    summary = optimizer.fit(X_train, y_train)
    assert summary is not None
    assert 'best_estimator_' in summary.keys()
    assert 'best_params_' in summary.keys()

def test_optimizer(data, estimators_and_param_grids):
    X_train, X_test, y_train, y_test = data
    estimators, param_grids = estimators_and_param_grids
    optimizer = Optimizer(estimators, param_grids, strategy='GSCV', n_jobs=1)
    summary = optimizer.fit(X_train, y_train)
    assert summary is not None
    assert 'SVC' in summary.keys()
    assert 'SGDClassifier' in summary.keys()

def test_optimizer2(data, estimators_and_param_grids):
    X_train, X_test, y_train, y_test = data
    estimators, param_grids = estimators_and_param_grids
    optimizer = Optimizer2(estimators, param_grids, strategy='GSCV', n_jobs=1)
    summary = optimizer.fit(X_train, y_train)
    assert summary is not None
    assert 'SVC' in summary.keys()
    assert 'SGDClassifier' in summary.keys()

def test_parallelize_search(data, estimators_and_param_grids):
    X_train, X_test, y_train, y_test = data
    estimators = [SVC(), SGDClassifier()]
    param_grids = [
        {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
        {'max_iter': [100, 500], 'alpha': [0.0001, 0.001]}
    ]
    optimizer = ParallelizeSearch(estimators, param_grids, strategy='RSCV', n_jobs=4)
    summary = optimizer.fit(X_train, y_train)
    assert summary is not None
    assert 'SVC' in summary.keys()
    assert 'SGDClassifier' in summary.keys()

def test_optimize_search(data, estimators_and_param_grids):
    X_train, X_test, y_train, y_test = data
    estimators, param_grids = estimators_and_param_grids
    optimizer = OptimizeSearch(estimators, param_grids, strategy='GSCV', n_jobs=1)
    summary = optimizer.fit(X_train, y_train)
    assert summary is not None
    assert 'SVC' in summary.keys()
    assert 'SGDClassifier' in summary.keys()
    
    
if __name__=='__main__': 
    pytest.main([__file__])
