# -*- coding: utf-8 -*-
"""
test_search.py
"""

import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import LeaveOneOut 
from gofast.models.search import MultipleSearch, CrossValidator, BaseSearch
from gofast.models.search import SearchMultiple, BaseEvaluation

@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y

@pytest.fixture
def param_grid():
    return {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}

@pytest.fixture
def svc():
    return SVC()

@pytest.fixture
def random_forest():
    return RandomForestClassifier(random_state=42)

@pytest.fixture
def logistic_regression():
    return LogisticRegression()

@pytest.fixture
def decision_tree():
    return DecisionTreeClassifier()

@pytest.fixture
def linear_svc():
    return LinearSVC()

def test_multiple_search(iris_data, svc, random_forest, param_grid):
    X, y = iris_data
    estimators = {'svc': svc, 'rf': random_forest}
    param_grids = {'svc': param_grid, 'rf': {'n_estimators': [100, 200], 'max_depth': [10, 20]}}
    strategies = ['grid', 'random']
    ms = MultipleSearch(estimators, param_grids, strategies, cv=3, n_iter=5)
    ms.fit(X, y)
    assert ms.best_params_ is not None

def test_cross_validator(iris_data, logistic_regression):
    X, y = iris_data
    cv = CrossValidator(logistic_regression, cv=5, scoring='accuracy')
    cv.fit(X, y)
    assert cv.calculateMeanScore() > 0

def test_base_search(iris_data, decision_tree, param_grid):
    X, y = iris_data
    grid_params = {'max_depth': [2, 4, 6], 'min_samples_split': [2, 5, 10]}
    search = BaseSearch(base_estimator=decision_tree, grid_params=grid_params, cv=3,
                        strategy='GridSearchCV')
    search.fit(X, y)
    assert search.best_params_ is not None
    assert search.cv_results_ is not None

def test_search_multiple(classification_data, logistic_regression, decision_tree, param_grid):
    X, y = classification_data
    estimators = [logistic_regression, decision_tree]
    grid_params = [
        {'C': [1, 10, 100]},
        {'max_depth': [2, 4, 6], 'min_samples_split': [2, 5, 10]}
    ]
    sm = SearchMultiple(estimators=estimators, scoring='accuracy', 
                        grid_params=grid_params, cv=3)
    sm.fit(X, y)
    assert len(sm.best_estimators_) == len(estimators)
    # Check if best_params_ is not None for the first estimator
    assert sm.best_estimators_[0][2] is not None  

def test_base_evaluation(iris_data, random_forest):
    X, y = iris_data
    evaluator = BaseEvaluation(estimator=random_forest, cv=3, 
                               scoring='accuracy')
    evaluator.fit(X, y)
    assert evaluator.cv_scores_ is not None
    assert evaluator.cv_scores_.mean() > 0

def test_cross_validator_with_custom_metrics(iris_data, logistic_regression):
    X, y = iris_data
    cv = CrossValidator(logistic_regression, cv=5, scoring='accuracy')
    additional_metrics = {'precision': precision_score, }
    cv.setCVStrategy('stratified', n_splits=5)
    metrics_kws = {"precision": {"average": "macro"}}
    cv.applyCVStrategy(X, y, metrics=additional_metrics, metrics_kwargs=metrics_kws, 
                       display_results=True)
    assert cv.cv_results_['additional_metrics']['precision'] is not None

def test_multiple_search_with_savejob(classification_data, svc, random_forest, param_grid):
    X, y = classification_data
    estimators = {'svc': svc, 'rf': random_forest}
    param_grids = {'svc': param_grid, 'rf': {'n_estimators': [100, 200], 'max_depth': [10, 20]}}
    strategies = ['grid', 'random']
    ms = MultipleSearch(estimators, param_grids, strategies, cv=3, n_iter=5, 
                        savejob=True, filename='test_joblib.pkl')
    ms.fit(X, y)
    assert ms.best_params_ is not None

def test_base_search_with_random_search(iris_data, logistic_regression, param_grid):
    X, y = iris_data
    grid_params = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
    search = BaseSearch(base_estimator=logistic_regression, grid_params=grid_params,
                        cv=3, strategy='RandomizedSearchCV', n_iter=5)
    search.fit(X, y)
    assert search.best_params_ is not None

def test_cross_validator_with_leaveoneout(classification_data, svc):
    X, y = classification_data
    cv = CrossValidator(svc, cv=LeaveOneOut(), scoring='accuracy')
    cv.fit(X, y)
    assert cv.calculateMeanScore() > 0

if __name__=='__main__': 
    
    pytest.main([__file__])