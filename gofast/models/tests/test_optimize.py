# -*- coding: utf-8 -*-
"""
@author: LKouadio <etanoyau@gmail.com>
"""
import pytest # noqa 
from joblib import load
from importlib import reload
#from concurrent.futures import ThreadPoolExecutor
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import gofast
import gofast.api as gf 
reload(gofast.api.testing)

from gofast.models.optimize import optimize_hyperparams, parallelize_search
from gofast.models.optimize import optimize_search, optimize_search2

X, y = load_iris(return_X_y=True)

@pytest.mark.skip ("Skip testing the Parameter searches of extensive computations costs.")
def test_parallelize_estimators ( optimizer = 'RSCV', pack_models =False ): 
    estimators = [SVC(), DecisionTreeClassifier()]
    param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
                       {'max_depth': [3, 5, None], 'criterion': ['gini', 'entropy']}
                       ]
    o=parallelize_search(estimators, param_grids, X, y, optimizer =optimizer, 
                           pack_models = pack_models )
    return o

# Define the fixture for the dataset
@pytest.fixture
def iris_data():
    return load_iris(return_X_y=True)

# Define the fixture for the classifiers and parameter grids
@pytest.fixture
def model_fixtures():
    estimators = {'rf': RandomForestClassifier(), 'svc': SVC()}
    param_grids = {
        'rf': {'n_estimators': [10, 100], 'max_depth': [None, 10]},
        'svc': {'C': [1, 10], 'kernel': ['linear', 'rbf']}
    }
    return estimators, param_grids

# Test normal operation
def test_optimize_search_normal_operation(iris_data, model_fixtures):
    X, y = iris_data
    estimators, param_grids = model_fixtures
    results = optimize_search(estimators, param_grids, X, y)
    expected_output = ['rf', 'svc', 'Best estimator', 'Best parameters']
    gf.testing.assert_model_summary_results(results, expected_output)
  
# Test handling mismatched keys
def test_optimize_search_mismatched_keys(iris_data):
    X, y = iris_data
    estimators = {'rf': RandomForestClassifier()}
    param_grids = {'rf': {'n_estimators': [10]}, 'svc': {'C': [1]}}
    with pytest.raises(ValueError):
        optimize_search(estimators, param_grids, X, y)

# Test optimize_search2 with a simple scenario
def test_optimize_search2_simple(iris_data):
    X, y = iris_data
    estimators = [RandomForestClassifier()]
    param_grids = [{'n_estimators': [100, 200], 'max_depth': [10, 20]}]
    result = optimize_search2(estimators, param_grids, X, y)
    expected_output = ['RandomForestClassifier', 'Tuning Results',
                       'Best estimator', 'Best parameters']
    gf.testing.assert_model_summary_has_title(
        result, expected_title="RandomForestClassifier")                              
    gf.testing.assert_model_summary_results(result, expected_output)

# Additional fixture for splitting the dataset
@pytest.fixture
def split_iris_data(iris_data):
    X, y = iris_data
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Test optimize_hyperparams with correct inputs
def test_optimize_hyperparams_correct_input(split_iris_data):
    X_train, X_test, y_train, y_test = split_iris_data
    estimator = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    results = optimize_hyperparams(estimator, param_grid, X_train, y_train, cv=3, n_jobs=1)
    assert 'best_estimator_' in results
    assert 'best_params_' in results
    assert 'cv_results_' in results

# Test parallelize_search with correct inputs
def test_parallelize_search_correct_input(iris_data):
    X, y = iris_data
    estimators = [SVC(), DecisionTreeClassifier()]
    param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']},
                   {'max_depth': [3, 5, None], 'criterion': ['gini', 'entropy']}]
    results = parallelize_search(estimators, param_grids, X, y, n_jobs=2)
    assert 'SVC' in results
    assert 'DecisionTreeClassifier' in results
    assert 'best_params_' in results['SVC']
    assert 'best_params_' in results['DecisionTreeClassifier']

# Test for handling invalid input types
@pytest.mark.parametrize("estimator, param_grid", [
    (None, {'n_estimators': [10]}),  # None is not a valid estimator
    (RandomForestClassifier(), None)  # None is not a valid parameter grid
])
def test_optimize_hyperparams_invalid_inputs(estimator, param_grid, split_iris_data):
    X_train, X_test, y_train, y_test = split_iris_data
    with pytest.raises(TypeError):
        optimize_hyperparams(estimator, param_grid, X_train, y_train)

# Test saving results in optimize_hyperparams
def test_optimize_hyperparams_saving_results(split_iris_data, tmp_path):
    X_train, X_test, y_train, y_test = split_iris_data
    estimator = SVC()
    param_grid = {'C': [1, 10], 'kernel': ['linear']}
    file_path = tmp_path / "svc.joblib"
    results = optimize_hyperparams(estimator, param_grid, X_train, y_train,
                                   savejob=True, savefile=str(file_path))
    loaded_results = load(file_path)
    assert 'best_estimator_' in loaded_results
    assert 'best_params_' in loaded_results


if __name__=='__main__': 
    pytest.main([__file__])
    #o= test_parallelize_estimators(optimizer ='GSCV', pack_models= True)
    