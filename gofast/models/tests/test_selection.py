# -*- coding: utf-8 -*-
"""
test_selection.py 
"""

import pytest
from scipy.stats import expon
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from gofast.models.selection import SwarmSearchCV, GradientSearchCV, AnnealingSearchCV
from gofast.models.selection import GeneticSearchCV, EvolutionarySearchCV, SequentialSearchCV

@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y

@pytest.fixture
def param_space():
    return {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}

@pytest.fixture
def svc():
    return SVC()

def test_swarm_search_cv(iris_data, param_space, svc):
    X, y = iris_data
    search = SwarmSearchCV(estimator=svc, param_space=param_space, max_iter=5,
                           n_particles=10, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_gradient_search_cv(iris_data, param_space, svc):
    X, y = iris_data
    param_space_gradient = {'C': (0.1, 10), 'gamma': (0.001, 0.1)}
    search = GradientSearchCV(estimator=svc, param_space=param_space_gradient,
                              max_iter=5, alpha=0.1, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_annealing_search_cv(iris_data, param_space, svc):
    X, y = iris_data
    search = AnnealingSearchCV(estimator=svc, param_space=param_space, 
                               max_iter=5, init_temp=1.0, alpha=0.9, 
                               random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0


def test_swarm_search_cv0(iris_data, param_space, svc):
    X, y = iris_data
    search = SwarmSearchCV(estimator=svc, param_space=param_space, max_iter=5,
                           n_particles=10, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_gradient_search_cv0(iris_data, param_space, svc):
    X, y = iris_data
    param_space_gradient = {'C': (0.1, 10), 'gamma': (0.001, 0.1)}
    search = GradientSearchCV(estimator=svc, param_space=param_space_gradient, 
                              max_iter=5, alpha=0.1, random_state=42, 
                              verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_annealing_search_cv0(iris_data, param_space, svc):
    X, y = iris_data
    search = AnnealingSearchCV(estimator=svc, param_space=param_space, max_iter=5,
                               init_temp=1.0, alpha=0.9, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_genetic_search_cv(iris_data, param_space, svc):
    X, y = iris_data
    search = GeneticSearchCV(estimator=svc, param_space=param_space, 
                             n_population=10, n_generations=5, random_state=42,
                             verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_evolutionary_search_cv(iris_data, param_space, svc):
    X, y = iris_data
    search = EvolutionarySearchCV(estimator=svc, param_space=param_space, 
                                  n_population=10, n_generations=5, 
                                  random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_sequential_search_cv(iris_data, param_space, svc):
    X, y = iris_data
    from scipy.stats import expon
    param_space_sequential = {'C': expon(scale=100), 'gamma': expon(scale=0.1)}
    search = SequentialSearchCV(estimator=svc, param_space=param_space_sequential,
                                n_iter=10, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0


def test_swarm_search_cv1(iris_data, param_space, svc):
    X, y = iris_data
    search = SwarmSearchCV(estimator=svc, param_space=param_space, max_iter=10,
                           n_particles=15, random_state=42, verbose=1)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_gradient_search_cv1(iris_data, param_space, svc):
    X, y = iris_data
    param_space_gradient = {'C': (0.1, 50), 'gamma': (0.001, 0.05)}
    search = GradientSearchCV(estimator=svc, param_space=param_space_gradient,
                              max_iter=10, alpha=0.05, random_state=42, verbose=1)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_annealing_search_cv1(iris_data, param_space, svc):
    X, y = iris_data
    search = AnnealingSearchCV(estimator=svc, param_space=param_space, 
                               max_iter=10, init_temp=0.8, alpha=0.85, 
                               random_state=42, verbose=1)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_genetic_search_cv1(iris_data, param_space, svc):
    X, y = iris_data
    search = GeneticSearchCV(estimator=svc, param_space=param_space, 
                             n_population=15, n_generations=7, random_state=42,
                             verbose=1)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_evolutionary_search_cv1(iris_data, param_space, svc):
    X, y = iris_data
    search = EvolutionarySearchCV(estimator=svc, param_space=param_space, 
                                  n_population=15, n_generations=7, 
                                  random_state=42, verbose=1)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_sequential_search_cv1(iris_data, svc):
    X, y = iris_data
    param_space_sequential = {'C': expon(scale=50), 'gamma': expon(scale=0.05)}
    search = SequentialSearchCV(estimator=svc, param_space=param_space_sequential,
                                n_iter=15, random_state=42, verbose=1)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_swarm_search_cv2(iris_data, param_space, svc):
    X, y = iris_data
    search = SwarmSearchCV(estimator=svc, param_space=param_space, max_iter=3,
                           n_particles=5, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_gradient_search_cv2(iris_data, param_space, svc):
    X, y = iris_data
    param_space_gradient = {'C': (1, 20), 'gamma': (0.01, 0.1)}
    search = GradientSearchCV(estimator=svc, param_space=param_space_gradient, 
                              max_iter=3, alpha=0.2, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_annealing_search_cv2(iris_data, param_space, svc):
    X, y = iris_data
    search = AnnealingSearchCV(estimator=svc, param_space=param_space, max_iter=3,
                               init_temp=1.2, alpha=0.7, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_genetic_search_cv2(iris_data, param_space, svc):
    X, y = iris_data
    search = GeneticSearchCV(estimator=svc, param_space=param_space, 
                             n_population=5, n_generations=3, random_state=42,
                             verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_evolutionary_search_cv2(iris_data, param_space, svc):
    X, y = iris_data
    search = EvolutionarySearchCV(estimator=svc, param_space=param_space, 
                                  n_population=5, n_generations=3, 
                                  random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

def test_sequential_search_cv2(iris_data, svc):
    X, y = iris_data
    param_space_sequential = {'C': expon(scale=20), 'gamma': expon(scale=0.01)}
    search = SequentialSearchCV(estimator=svc, param_space=param_space_sequential,
                                n_iter=5, random_state=42, verbose=0)
    search.fit(X, y)
    assert search.best_params_ is not None
    assert isinstance(search.best_score_, float)
    assert search.best_score_ > 0

if __name__ == "__main__":
    pytest.main([__file__])
