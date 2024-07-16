# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides a set of classes for model selection and hyperparameter tuning, 
including tools for cross-validation and automated search strategies to 
optimize model performance."""

from __future__ import annotations 

import inspect
import warnings  
import joblib
from pprint import pprint 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, accuracy_score 
from sklearn.model_selection import  KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .._gofastlog import gofastlog
from ..api.structures import Boxspace
from ..api.docstring import DocstringComponents, _core_docs 
from ..api.property import BaseClass 
from ..api.summary import ModelSummary, ResultSummary 
from ..api.types import _F, List,ArrayLike, NDArray, Dict, Any, Optional, Union
from ..exceptions import EstimatorError, NotFittedError
from ..tools.coreutils import save_job, get_params 
from ..tools.coreutils import listing_items_format, validate_ratio 
from ..tools.validator import check_X_y, check_array, check_consistent_length 
from ..tools.validator import get_estimator_name, filter_valid_kwargs 

from .utils import get_scorers, dummy_evaluation, get_strategy_name
from .utils import _standardize_input , get_strategy_method 
from .utils import align_estimators_with_params, process_performance_data 
from .utils import update_if_higher 
 
_logger = gofastlog().get_gofast_logger(__name__)

__all__=["BaseEvaluation", "BaseSearch", "SearchMultiple",
         "CrossValidator", "MultipleSearch","PerformanceTuning"
    ]

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    )
class MultipleSearch(BaseClass):
    r"""
    A class for concurrently performing parameter tuning across multiple  
    estimators using different search strategies.

    This class allows users to specify multiple machine learning estimators  
    and perform parameter tuning using various strategies like GridSearchCV, 
    RandomizedSearchCV, or BayesSearchCV simultaneously in parallel. The best 
    parameters for each estimator can be saved to a file.

    Parameters
    ----------
    estimators : dict
        A dictionary of estimators to tune. 
        Format: {'estimator_name': estimator_object}.
    param_grids : dict
        A dictionary of parameter grids for the corresponding estimators. 
        Format: {'estimator_name': param_grid}.
    strategies : list of str
        List of strategies to use for parameter tuning. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.
        
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Default is 4.
    n_iter : int, optional
        Number of iterations for random or Bayes search. Default is 10.
    scoring : str, optional
        Scoring strategy to evaluate the models. Refer to scikit-learn's scoring 
        parameter documentation for valid values.
    savejob : bool, optional
        Whether to save the tuning results to a joblib file. Default is False.
    filename : str, optional
        The filename for saving the joblib file. Required if savejob is True.

    Attributes
    ----------
    best_params_ : dict
        The best found parameters for each estimator and strategy combination.
        
    summary_ : ModelSummary
        Summary of the tuning process, including detailed information about 
        the results of the parameter searches for each estimator and strategy.

    results_ : ResultSummary
        Contains the best parameters found for each estimator, along with 
        additional results such as the best estimators and cross-validation 
        results.

    Methods
    -------
    fit(X, y):
        Run the parameter tuning for each estimator using each strategy in
        parallel on the given data.

    Notes
    -----
    The optimization problem for parameter tuning can be formulated as:
        
    .. math::
        \underset{\theta}{\text{argmin}} \; f(\theta)
        
    where :math:`\theta` represents the parameters of the model, and 
    :math:`f(\theta)` is the objective function, typically the cross-validated 
    performance of the model.

    The class utilizes concurrent processing to run multiple parameter search 
    operations in parallel, enhancing efficiency, especially for large datasets 
    and complex models.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.models.search import MultipleSearch
    >>> from gofast.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5)
    >>> estimators = {'rf': RandomForestClassifier()}
    >>> param_grids = {'rf': {'n_estimators': [100, 200], 'max_depth': [10, 20]}}
    >>> strategies = ['GSCV', 'RSCV']
    >>> ms = MultipleSearch(estimators, param_grids, strategies)
    >>> ms.fit(X, y)
    >>> print(ms.best_params_)

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gofast.models.search import MultipleSearch
    >>> from gofast.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5)
    >>> estimators = {'rf': RandomForestClassifier(), 'dtc': DecisionTreeClassifier()}
    >>> param_grids = {'rf': {'n_estimators': [100, 200], 'max_depth': [10, 20]}, 
    ...          'dtc': {"max_depth": [ 11, 22, 30]}}
    >>> strategies = ['GSCV', 'RSCV']
    >>> ms = MultipleSearch(estimators, param_grids, strategies, scoring='accuracy')
    >>> ms.fit(X, y)
    >>> print(ms.summary_) 
                        Optimized Results                     
    ==========================================================
    |                 RandomForestClassifier                 |
    ----------------------------------------------------------
                          Model Results                       
    ==========================================================
    Best estimator       : RandomForestClassifier
    Best parameters      : {'max_depth': 10, 'n_estimators'...
    Best score           : 0.9800
    Scoring              : accuracy
    nCV                  : 4
    Params combinations  : 4
    ==========================================================

                     Tuning Results (*=score)                 
    ==========================================================
        Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    ----------------------------------------------------------
    0 (10, 100) 0.9800 0.0200        0.9800        0.0200    1
    1 (10, 200) 0.9800 0.0200        0.9800        0.0200    1
    2 (20, 100) 0.9800 0.0200        0.9800        0.0200    1
    3 (20, 200) 0.9800 0.0200        0.9800        0.0200    1
    ==========================================================


    ==========================================================
    |                 DecisionTreeClassifier                 |
    ----------------------------------------------------------
                          Model Results                       
    ==========================================================
    Best estimator       : DecisionTreeClassifier
    Best parameters      : {'max_depth': 22}
    Best score           : 0.9700
    Scoring              : accuracy
    nCV                  : 4
    Params combinations  : 3
    ==========================================================

                     Tuning Results (*=score)                 
    ==========================================================
        Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    ----------------------------------------------------------
    0      (11) 0.9600 0.0000        0.9600        0.0000    2
    1      (22) 0.9700 0.0173        0.9700        0.0173    1
    2      (30) 0.9600 0.0000        0.9600        0.0000    2
    ==========================================================
    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on 
        hyperparameters.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """

    def __init__(self, 
        estimators, 
        param_grids, 
        strategies, 
        cv=4, 
        n_iter=10,
        scoring=None,
        savejob=False, 
        filename=None
        ):
        self.estimators = estimators
        self.param_grids = param_grids
        self.strategies = strategies
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.savejob = savejob
        self.filename = filename
        
    def fit(self, X, y):
        r"""
        Run the parameter tuning for each estimator using each strategy in parallel 
        on the given data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
    
        Notes
        -----
        This method performs parameter tuning by applying the specified strategies
        to each estimator in parallel. The best parameters found for each estimator
        and strategy are stored in the `best_params_` attribute.
    
        The method leverages concurrent processing to execute multiple parameter search
        operations in parallel, improving efficiency and reducing the total tuning time.
    
        The optimization problem for parameter tuning can be formulated as:
        
        .. math::
            \underset{\theta}{\text{argmin}} \; f(\theta)
            
        where :math:`\theta` represents the parameters of the model, and 
        :math:`f(\theta)` is the objective function, typically the cross-validated 
        performance of the model.
    
        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from gofast.models.search import MultipleSearch
        >>> from gofast.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=5)
        >>> estimators = {'rf': RandomForestClassifier()}
        >>> param_grids = {'rf': {'n_estimators': [100, 200], 'max_depth': [10, 20]}}
        >>> strategies = ['GSCV', 'RSCV']
        >>> ms = MultipleSearch(estimators, param_grids, strategies)
        >>> ms.fit(X, y)
        >>> print(ms.best_params_)
    
        See Also
        --------
        sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
            parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV : Randomized search on 
            hyperparameters.
        
        References
        ----------
        .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
               Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
               and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
               Journal of Machine Learning Research, 12, 2825-2830.
        """
        results = {}
        self.best_params_ = {}
        
        estimators, param_grids = align_estimators_with_params(
            self.param_grids, self.estimators)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ThreadPoolExecutor(max_workers=len(self.strategies)) as executor:
                futures = []
                for strategy in self.strategies:
                    for estimator, param_grid in zip(estimators, param_grids):
                        future = executor.submit(
                            self._search, estimator,
                            param_grid, strategy, X, y, self.scoring)
                        futures.append(future)
                
                results, self.best_params_ =self._update_results_based_on_score(
                    futures, results, self.best_params_ )

        if self.savejob: 
            self.filename = self.filename or "ms_results.joblib"
            joblib.dump(self.best_params_, self.filename)
    
        self.summary_ = ModelSummary(
            descriptor="MultipleSearch", **results).summary(results)
        self.results_ = ResultSummary(name="BestParams").add_results(
            self.best_params_)
        
        return self
    
    def _search(self, estimator, param_grid, strategy, X, y, scoring):
        """
        Conducts a parameter search using the specified strategy on the
        given estimator.

        Parameters
        ----------
        estimator : estimator object
            The machine learning estimator to tune.
        param_grid : dict
            The parameter grid for tuning.
        strategy : str
            The strategy to use for parameter tuning.
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        dict
            The best parameters found for the estimator using the strategy.
        """
        strategy_name = get_strategy_name(strategy, error='ignore')
        strategy_ = get_strategy_method(strategy)
        search_kwargs = {"n_iter": self.n_iter} 
        search_kwargs = filter_valid_kwargs(estimator, search_kwargs)
        search = strategy_(estimator, param_grid, cv=self.cv, scoring=scoring, 
                           **search_kwargs)
        search.fit(X, y)
        best_params = {estimator.__class__.__name__: search.best_params_} 
        result = {estimator.__class__.__name__: {
                    "best_estimator_": search.best_estimator_, 
                    "best_params_": search.best_params_, 
                    "best_score_": search.best_score_, 
                    "scoring": _standardize_input(scoring), 
                    "strategy": strategy_name,
                    "cv_results_": search.cv_results_ ,
                    }
                }
        return best_params, result
    
    def _update_results_based_on_score(
            self, futures, results=None, best_params=None):
        """
        Updates results and `self.best_params_` based on the best score 
        of each estimator.
    
        Parameters
        ----------
        futures : list
            List of futures containing the results of the parameter search.
        results : dict, optional
            A dictionary to store the updated results. If not provided, a new 
            dictionary is created.
        best_params : dict, optional
            A dictionary to store the best parameters of each estimator. If not 
            provided, a new dictionary is created.
    
        Returns
        -------
        results : dict
            The updated results dictionary.
        best_params : dict
            The updated best parameters dictionary.
    
        Notes
        -----
        This method iterates through the futures, retrieves the results of 
        the parameter search, and updates the instance's `results` and 
        `best_params_` attributes based on the highest score for each estimator. 
        It uses the `update_if_higher` function to ensure that only results with 
        higher scores are retained.
    
        Examples
        --------
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> import numpy as np
        >>> class DummyClass:
        ...     def __init__(self):
        ...         self.best_params_ = {}
        ...         self.results_ = {}
        >>> dummy_instance = DummyClass()
        >>> def dummy_search_function(estimator, param_grid, strategy, X, y):
        ...     best_params = {estimator.__class__.__name__: {'n_estimators': 100}}
        ...     result = {
        ...         estimator.__class__.__name__: {
        ...             "best_estimator_": estimator,
        ...             "best_params_": {'n_estimators': 100},
        ...             "best_score_": np.random.rand(),
        ...             "strategy": strategy,
        ...             "cv_results_": None,
        ...         }
        ...     }
        ...     return best_params, result
        >>> X, y = make_classification(n_samples=100, n_features=5)
        >>> estimators = [RandomForestClassifier()]
        >>> param_grids = [{'n_estimators': [100, 200], 'max_depth': [10, 20]}]
        >>> strategies = ['GSCV', 'RSCV']
        >>> futures = []
        >>> with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
        ...     for strategy in strategies:
        ...         for estimator, param_grid in zip(estimators, param_grids):
        ...             future = executor.submit(
        ...                 dummy_search_function, estimator, param_grid, strategy, X, y
        ...             )
        ...             futures.append(future)
        >>> updated_results, updated_best_params = dummy_instance._update_results_based_on_score(
        ...      futures)
        >>> print(updated_results)
        >>> print(updated_best_params)
        """
        results = results or {}
        best_params = best_params or {}
        for future in tqdm(futures, desc='Optimizing parameters', ncols=100, 
                           ascii=True, unit='search'):
            best_param, result = future.result()
            estimator_name = list(best_param.keys())[0]
            results, best_params = update_if_higher(
                results, 
                estimator_name, 
                result[estimator_name]["best_score_"],
                result, 
                best_params
            )
        return results, best_params


class PerformanceTuning(BaseClass):
    """
    Fine-tune multiple estimators and create performance data for model comparison.

    This class provides functionalities to fine-tune multiple estimators, 
    evaluate their performance using cross-validation, and create performance 
    data for model comparison using statistical tests such as Friedman test, 
    Nemenyi post-hoc test, and Wilcoxon signed-rank test.

    Parameters
    ----------
    estimators : list of BaseEstimator, optional
        List of estimators to fine-tune and evaluate. Each estimator should 
        implement a `fit` method.
    
    param_grids : list of dict, optional
        List of parameter grids to be used for fine-tuning each estimator.
    
    scoring : str, optional
        Scoring strategy to evaluate the performance of the cross-validated model.
    
    strategy : str, default="GSCV"
        Strategy for hyperparameter search. The search strategy to apply for
        hyperparameter optimization. Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.
    
    cv : int, optional
        Number of cross-validation folds.
    
    tuning_depth : str, default="base"
        Depth of fine-tuning. If "deep", a more thorough fine-tuning is performed.
    
    n_jobs : int, default=1
        Number of jobs to run in parallel.

    **search_kwargs : dict, optional
        Additional keyword arguments passed to the search method.

    Methods
    -------
    fit(X, y):
        Fits the models to the data (X, y) and performs cross-validation.
    
    _cv_performance_data_base(X, y):
        Performs base cross-validation for performance data.
    
    _cv_performance_deep(X, y):
        Performs deep fine-tuning and cross-validation for performance data.
    
    _run_cross_validator(X, y):
        Runs cross-validation for each estimator and returns performance data
        and mean scores.
    
    _make_summary(performance, scores):
        Creates a summary of the performance data and scores.
    
    @staticmethod
    def evaluate_models_on_datasets(estimators, datasets, scoring, cv):
        Evaluates each model on multiple datasets and constructs a performance 
        data DataFrame.

    Examples
    --------
    >>> from gofast.models.search import PerformanceTuning
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import load_iris
    >>> clf1 = DecisionTreeClassifier()
    >>> clf2 = LogisticRegression()
    >>> param_grids = [
    ...     {'max_depth': [3, 5, 7]},
    ...     {'C': [0.1, 1, 10]}
    ... ]
    >>> tuning = PerformanceTuning(
    ...     estimators=[clf1, clf2],
    ...     param_grids=param_grids,
    ...     scoring='accuracy',
    ...     cv=5,
    ...     tuning_depth='deep'
    ... )
    >>> X, y = load_iris(return_X_y=True)
    >>> tuning.fit(X, y)
    >>> PerformanceTuning.evaluate_models_on_datasets(
    ...     estimators=[clf1, clf2],
    ...     datasets=[(X, y), (X, y)],
    ...     scoring='accuracy',
    ...     cv=5
    ... )

    Notes
    -----
    The `PerformanceTuning` class allows for a flexible and comprehensive approach to
    model evaluation by supporting various hyperparameter search strategies and
    cross-validation techniques. By comparing multiple models on different datasets,
    it provides valuable insights into model performance and robustness.

    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyperparameters.
    
    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """

    def __init__(
        self, 
        estimators=None, 
        param_grids=None, 
        scoring=None,
        strategy: str="GSCV", 
        cv=None,
        tuning_depth='base',
        n_jobs=1, 
        **search_kwargs 
        ): 
        self.estimators = estimators
        self.param_grids = param_grids 
        self.scoring = scoring 
        self.strategy = strategy 
        self.tuning_depth = tuning_depth 
        self.cv = cv 
        self.n_jobs = n_jobs
        self.search_kwargs = search_kwargs
        
    def fit(self, X, y):
        """
        Fits the models to the data (X, y) and performs cross-validation.
    
        This method fine-tunes multiple estimators, evaluates their performance
        using cross-validation, and creates performance data for model comparison.
        The tuning can be performed at different depths (base or deep) based on 
        the `tuning_depth` parameter.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix representing the input data.
            Each row corresponds to a sample, and each column corresponds to a feature.
        
        y : array-like of shape (n_samples,)
            The target values corresponding to the input data. It represents the 
            true labels or values that the model aims to predict.
    
        Returns
        -------
        self : object
            Returns the instance itself.
    
        Notes
        -----
        The method processes the estimators and parameter grids, then performs 
        cross-validation based on the specified tuning depth. If `tuning_depth` 
        is set to "deep", it performs a more thorough fine-tuning, otherwise, 
        it performs a base-level cross-validation.
    
        The cross-validation strategy ensures that each model is evaluated 
        consistently, providing a comprehensive comparison of their performance 
        on the given dataset.
    
        Examples
        --------
        >>> from gofast.models.search import PerformanceTuning
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>> clf1 = DecisionTreeClassifier()
        >>> clf2 = LogisticRegression()
        >>> param_grids = [
        ...     {'max_depth': [3, 5, 7]},
        ...     {'C': [0.1, 1, 10]}
        ... ]
        >>> tuning = PerformanceTuning(
        ...     estimators=[clf1, clf2],
        ...     param_grids=param_grids,
        ...     scoring='accuracy',
        ...     cv=5,
        ...     tuning_depth='deep'
        ... )
        >>> X, y = load_iris(return_X_y=True)
        >>> tuning.fit(X, y)
        
        See Also
        --------
        sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
            parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV : Randomized search on hyperparameters.
        
        References
        ----------
        .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
               Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
               and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
               Journal of Machine Learning Research, 12, 2825-2830.
        """
        
        if self.estimators is not None and not isinstance(
                self.estimators, (list, tuple)):
            self.estimators = [self.estimators]
        if self.param_grids is not None and not isinstance(
                self.param_grids, (list, tuple)):
            self.param_grids = [self.param_grids]
        
        self.tuning_depth = str(self.tuning_depth).lower()
        if self.tuning_depth == 'deep':
            self._cv_performance_deep(X, y)
        else:
            self._cv_performance_data_base(X, y)
        
        return self

    def _cv_performance_data_base(self, X, y): 
        
        if self.estimators is None: 
            if self.param_grids is not None: 
                # check whether estimators are gives as tuple ( estim, params )
                self.estimators, self.param_grids = align_estimators_with_params(
                    self.param_grids)
            raise ValueError("Estimators cannot be None for base tuning.")
            
        performance_data, results_mean = self._run_cross_validator(X, y)
        self._make_summary(performance_data, results_mean)
        return self
    
    def _make_summary(self, performance, scores): 
        self.results_ = ResultSummary(name='PerformanceData', pad_keys="auto"
                                      ).add_results(performance)
        self.scores_ = ResultSummary(name='Scores', pad_keys="auto"
                                     ).add_results(scores)
        self.performance_data_ = pd.DataFrame(performance)
        
    def _run_cross_validator(self, X, y): 
        performance_data = {}
        scores_mean = {}
        for estimator in self.estimators: 
            cv = CrossValidator(estimator, cv=self.cv, scoring=self.scoring)
            cv.fit(X, y) 
            scores, mean_score = cv.score_results_
            performance_data[f"{get_estimator_name(estimator)}"] = [
                round(score, 4) for score in scores ]
            scores_mean[f"{get_estimator_name(estimator)}"] = scores
        return performance_data, scores_mean
    
    def _cv_performance_deep(self, X, y): 
        self.strategy_ = get_strategy_method(self.strategy)
        estimators, param_grids = align_estimators_with_params(
            self.param_grids, estimators=self.estimators)
        search_kwargs = filter_valid_kwargs(
            self.strategy_, self.search_kwargs)
        
        best_estimators = [] 
        for estimator, param_grid in zip(estimators, param_grids): 
            search = self.strategy_(estimator, param_grid, scoring=self.scoring,
                                    cv=self.cv, n_jobs=self.n_jobs, **search_kwargs)
            search.fit(X, y)
            best_estimators.append(search.best_estimator_)
        
        self.estimators = best_estimators
        performance_data, scores_mean = self._run_cross_validator(X, y)
        self._make_summary(performance_data, scores_mean)
        
    @staticmethod
    def evaluate_models_on_datasets(
            estimators, datasets, target=None, scoring=None, cv=None,
            n_jobs=None, error='warn', mode='average', on='@data', 
            **cross_val_kwargs):
        """
        Evaluates each model on multiple datasets and constructs a performance
        data DataFrame.
    
        This method evaluates multiple estimators on multiple datasets and 
        compiles their performance into a pandas DataFrame. It supports handling 
        datasets with or without target labels and provides flexibility in 
        handling different scoring strategies and cross-validation methods.
    
        Parameters
        ----------
        estimators : list of BaseEstimator
            List of estimators to evaluate. Each estimator should implement 
            a `fit` method.
        
        datasets : list of tuple
            List of datasets to evaluate the estimators on. Each tuple contains 
            (X, y) or just X if target is provided.
        
        target : array-like, optional
            Target values to be used if not provided in datasets.
        
        scoring : str, optional
            Scoring strategy to evaluate the models. Refer to scikit-learn's 
            scoring parameter documentation for valid values.
        
        cv : int, optional
            Number of cross-validation folds.
        
        n_jobs : int, optional
            Number of jobs to run in parallel. Defaults to `None`, meaning 1.
        
        error : str, default='warn'
            Error handling strategy when target is provided but y exists in datasets.
            Options are 'warn' to issue a warning or 'raise' to raise an error.
    
        mode : str, optional, default='average'
            Processing mode. Options:
            - 'average': Averages each row array value at each index.
            - 'rowX': Processes data based on the specific row index (
                e.g., 'row0', 'row1').
            - 'dX': Same as 'rowX'.
            - 'cvX': Processes data based on the specific column index (
                e.g., 'cv0', 'cv1').
            - 'cX': Same as 'cvX'.
    
        on : str, optional, default='@data'
            Specifies the axis of processing.
            - '@data': Processes data based on rows (i.e. number of datasets).
            - '@cv': Processes data based on columns ( i.e. based on cv splits).
        
        **cross_val_kwargs : dict, optional
            Additional keyword arguments passed to `cross_val_score`.
    
        Returns
        -------
        performance_data : pd.DataFrame
            DataFrame containing the performance data for each model on each dataset.
            The DataFrame has model names as columns and lists of cross-validation
            scores for each dataset as rows.
    
        Notes
        -----
        If `target` is provided and datasets contain target values, the method will
        either warn or raise an error based on the `error` parameter. This ensures
        consistency in the use of target values.
    
        The method uses cross-validation to evaluate the models, making it robust 
        to overfitting and providing a reliable measure of model performance.
    
        Examples
        --------
        >>> from gofast.models.search import PerformanceTuning
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> clf1 = DecisionTreeClassifier()
        >>> clf2 = LogisticRegression()
        >>> datasets = [X, (X, y), (X, None)]
        >>> PerformanceTuning.evaluate_models_on_datasets(
        ...     estimators=[clf1, clf2],
        ...     datasets=datasets,
        ...     target=y,
        ...     scoring='accuracy',
        ...     cv=5
        ... )
    
        See Also
        --------
        sklearn.model_selection.cross_val_score : Evaluate a score by cross-validation.
        
        References
        ----------
        .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
               Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
               and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
               Journal of Machine Learning Research, 12, 2825-2830.
        """
        if target is not None:
            # Process datasets based on the presence of target and dataset structure
            processed_datasets = []
            for dataset in datasets:
                if isinstance(dataset, tuple):
                    X, y = dataset
                    if y is not None:
                        if error == 'warn':
                            warnings.warn(
                                "Target `y` is provided in datasets; ignoring"
                                " `target` parameter.")
                        elif error == 'raise':
                            raise ValueError(
                                "Target `y` is provided in datasets; `target`"
                                " parameter should be ignored.")
                    elif y is None: 
                        y = target 
                        
                else:
                    X = dataset
                    y = target
                    if y is None:
                        raise ValueError("Expected target `y` for model evaluation.")
                processed_datasets.append((X, y))
        else:
            # Ensure all datasets have y
            processed_datasets = []
            for dataset in datasets:
                if isinstance(dataset, tuple):
                    X, y = dataset
                else:
                    raise ValueError("Expected target `y` for model evaluation.")
                processed_datasets.append((X, y))
        
        performance_data = {}
        for estimator in estimators:
            estimator_name = get_estimator_name(estimator)
            scores = []
            for X, y in processed_datasets:
                
                cv_scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, 
                                            **cross_val_kwargs)
                scores.append(list(np.around(cv_scores, 4)))
    
            performance_data[estimator_name] = scores
        performance_df = pd.DataFrame(performance_data)
        
        try: 
            performance_df = process_performance_data(performance_df, mode=mode, on=on)
        except Exception as e: 
            if error == 'warn': 
                warnings.warn(str(e)) 
            elif error == 'raise': 
                raise ValueError(
                    "An error occurred during performance data processing.") from e
        
        return performance_df
  
    def __repr__(self):
        """
        Provide a string representation of the PerformanceTuning object, 
        displaying its parameters in a formatted manner for better readability.
    
        Returns
        -------
        repr : str
            String representation of the PerformanceTuning object with its parameters.
        """
        parameters = get_params (self)
        params = ",\n    ".join(f"{key}={val}" for key, val in parameters.items())
        return f"{self.__class__.__name__}(\n    {params}\n)"


class CrossValidator(BaseClass):
    """
    A class for handling cross-validation of machine learning models.

    This class provides functionalities for performing cross-validation on
    various classifiers or regressors using different scoring strategies.

    Parameters
    ----------
    estimator : BaseEstimator
        The machine learning model to be used for cross-validation. It must 
        implement `fit` and `predict` methods.
    
    cv : int, optional
        The number of folds for cross-validation (default is 5).
    
    scoring : str, optional
        The scoring strategy to evaluate the model (default is 'accuracy').
    
    Attributes
    ----------
    results : Optional[Tuple[np.ndarray, float]]
        Stored results of the cross-validation, including scores and mean score.
    
    Methods
    -------
    fit(X, y=None):
        Fits the model to the data (X, y) and performs cross-validation.
    
    calculate_mean_score():
        Calculates and returns the mean of the cross-validation scores.
    
    display_results():
        Prints the cross-validation scores and their mean.
    
    setCVStrategy(cv_strategy, n_splits=5, random_state=None):
        Sets the cross-validation strategy for the model.
    
    applyCVStrategy(X, y, metrics=None, metrics_kwargs={}, display_results=False):
        Applies the configured cross-validation strategy to the given dataset 
        and evaluates the model.
    
    _display_results():
        Displays the results of cross-validation.
    
    Examples
    --------
    >>> from gofast.models.search import CrossValidator
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gofast.datasets import make_classification
    >>> clf = DecisionTreeClassifier()
    >>> cross_validator = CrossValidator(clf, cv=5, scoring='accuracy')
    >>> X, y = make_classification(n_samples=100, n_features=7)
    >>> cross_validator.fit(X, y)
    >>> cross_validator.displayResults()
    Result(
      {
           Classifier/Regressor : DecisionTreeClassifier
           Cross-validation scores : [0.85, 1.0, 0.8, 0.95, 0.85]
           Mean score : 0.8900
      }
    )
    [ 3 entries ]

    See Also
    --------
    sklearn.model_selection.cross_val_score : Evaluate a score by cross-validation.
    sklearn.model_selection.KFold : K-Folds cross-validator.
    sklearn.model_selection.StratifiedKFold : Stratified K-Folds cross-validator.
    sklearn.model_selection.LeaveOneOut : Leave-One-Out cross-validator.
    
    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """

    def __init__(
        self, estimator: BaseEstimator,
        cv: int = 5, 
        scoring: str = 'accuracy'
        ):
  
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring

    def fit(self, X: Union[ArrayLike, list],
            y: Optional[Union[ArrayLike, list]] = None):
        """
        Fits the model to the data (X, y) and performs cross-validation.

        Parameters
        ----------
        X : np.ndarray or list
            The feature dataset for training the model.
        y : np.ndarray or list, optional
            The target labels for the dataset (default is None).

        Raises
        ------
        ValueError
            If the target labels `y` are not provided for supervised 
            learning models.
        """
        if y is None and issubclass(self.estimator.__class__, BaseEstimator):
            raise ValueError("Target labels `y` must be provided for"
                             " supervised learning models.")
        
        scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
        mean_score = scores.mean()
        self.score_results_ = (scores, mean_score)

        return self 
    
    def calculateMeanScore(self) -> float:
        """
        Calculate and return the mean score from the cross-validation results.

        Returns
        -------
        float
            Mean of the cross-validation scores.

        Raises
        ------
        ValueError
            If cross-validation has not been performed yet.
        """
        self.inspect 
        if self.score_results_ is None:
            raise ValueError("Cross-validation has not been performed yet.")
        return self.score_results_[1]
 
    def displayResults(self):
        """
        Display the cross-validation scores and their mean.

        Raises
        ------
        ValueError
            If cross-validation has not been performed yet.
        """
        self.inspect 
        if self.score_results_ is None:
            raise ValueError("Cross-validation has not been performed yet.")
        
        scores, mean_score = self.score_results_
        estimator_name = self.estimator.__class__.__name__
        result = {'Classifier/Regressor':f'{estimator_name}', 
                   "Cross-validation scores": f"{list(scores)}", 
                   "Mean score": f"{mean_score:.4f}"
                   }
        
        summary = ResultSummary().add_results(result)
        print(summary)

    def setCVStrategy(self, cv_strategy: Union[int, str, object], 
                        n_splits: int = 5, random_state: int = None):
        """
        Sets the cross-validation strategy for the model.

        Parameters
        ----------
        cv_strategy : int, str, or cross-validation generator object
            The cross-validation strategy to be used. If an integer is provided,
            KFold is assumed with that many splits. If a string is provided,
            it must be one of 'kfold', 'stratified', or 'leaveoneout'.
            Alternatively, a custom cross-validation generator object can be provided.
        n_splits : int, optional
            The number of splits for KFold or StratifiedKFold (default is 5).
        random_state : int, optional
            Random state for reproducibility (default is None).
        
        Returns
        -------
        cv : cross-validation generator object
            The configured cross-validation generator instance.
            
        Notes
        -----
        - 'kfold': KFold divides all the samples into 'n_splits' number of groups,
           called folds, of equal sizes (if possible). Each fold is then used as
           a validation set once while the remaining folds form the training set.
        - 'stratified': StratifiedKFold is a variation of KFold that returns
           stratified folds. Each set contains approximately the same percentage
           of samples of each target class as the complete set.
        - 'leaveoneout': LeaveOneOut (LOO) is a simple cross-validation. Each
           learning set is created by taking all the samples except one, the test
           set being the sample left out.
           
        Raises
        ------
        ValueError
            If an invalid cross-validation strategy or type is provided.
            
        Examples 
        ---------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> clf = DecisionTreeClassifier()
        >>> cross_validator = CrossValidator(clf, cv=5, scoring='accuracy')
        >>> cross_validator.setCVStrategy('stratified', n_splits=5)
        >>> X, y = # Load your dataset here
        >>> cross_validator.fit(X, y)
        >>> cross_validator.display_results()
        """
        if isinstance(cv_strategy, int):
            self.cv = KFold(n_splits=cv_strategy, random_state=random_state, shuffle=True)
        elif isinstance(cv_strategy, str):
            cv_strategy = cv_strategy.lower()
            if cv_strategy == 'kfold':
                self.cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            elif cv_strategy == 'stratified':
                self.cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, 
                                          shuffle=True)
            elif cv_strategy == 'leaveoneout':
                self.cv = LeaveOneOut()
            else:
                raise ValueError(f"Invalid cross-validation strategy: {cv_strategy}")
        elif hasattr(cv_strategy, 'split'):
            self.cv = cv_strategy
        else:
            raise ValueError("cv_strategy must be an integer, a string,"
                             " or a cross-validation generator object.")
        
        return self.cv 

    def applyCVStrategy(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        metrics: Optional[Dict[str, _F[[ArrayLike, ArrayLike], float]]] = None, 
        metrics_kwargs: dict={},
        display_results: bool = False, 
        
        ):
        """
        Applies the configured cross-validation strategy to the given dataset
        and evaluates the model.

        This method performs cross-validation using the strategy set by the
        `setCVStrategy` method. It fits the model on each training set and
        evaluates it on each corresponding test set. The results, including
        scores and other metrics, are stored in the `cv_results_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to be used for cross-validation.
        y : array-like of shape (n_samples,)
            Target values corresponding to X.
        metrics : dict, optional
            A dictionary where keys are metric names and values are functions
            that compute the metric. Each function should accept two arguments:
            true labels and predicted labels. Default is None.
            
        display_results : bool, optional
            Whether to print the cross-validation results. Default is False.
        metrics_kwargs: dict, 
            Dictionnary containing each each metric name listed in `metrics`. 
            
        Stores
        -------
        cv_results_ : dict
            Dictionary containing the results of cross-validation. This includes
            scores for each fold, mean score, and standard deviation.

        Notes
        -----
        - KFold: Suitable for large datasets and is not stratified.
        - StratifiedKFold: Suitable for imbalanced datasets, as it preserves
          the percentage of samples for each class.
        - LeaveOneOut: Suitable for small datasets but computationally expensive.
        - Ensure that the cross-validation strategy is appropriately chosen
          for your dataset and problem. For example, StratifiedKFold is more
          suited for datasets with imbalanced class distributions.
        - The metrics functions should match the signature of sklearn.metrics
          functions, i.e., they accept two arguments: y_true and y_pred.
          
        Raises
        ------
        ValueError
            If the cross-validation strategy has not been set prior to calling
            this method.
            
        Examples
        --------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = DecisionTreeClassifier()
        >>> cross_validator = CrossValidator(clf)
        >>> cross_validator.setCVStrategy('kfold', n_splits=5)
        >>> cross_validator.applyCVStrategy(X, y, display_results=True)
        >>> from sklearn.metrics import accuracy_score, precision_score
        >>> additional_metrics = {
        ...     'accuracy': accuracy_score,
        ...     'precision': precision_score
        ... }
        >>> cross_validator.applyCVStrategy(X, y, metrics=additional_metrics)
        """
        if not hasattr(self, 'cv'):
            raise ValueError("Cross-validation strategy not set."
                             " Please call setCVStrategy first.")

        self.cv_results_ = {'scores': [], 'additional_metrics': {}}
        if metrics:
            for metric in metrics:
                self.cv_results_['additional_metrics'][metric] = []

        for train_index, test_index in self.cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model_clone = clone(self.estimator)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            score = model_clone.score(X_test, y_test)
            self.cv_results_['scores'].append(score)

            if metrics:
                for metric_name, metric_func in metrics.items():
                    if metrics_kwargs: 
                        metric_kws = ( metrics_kwargs[metric_name] 
                            if metric_name in metrics_kwargs else {}
                            )
                    else:
                        metric_kws ={} 
                        
                    metric_score = metric_func(y_test, y_pred, **metric_kws)
                    self.cv_results_['additional_metrics'][metric_name].append(metric_score)

        self.cv_results_['mean_score'] = np.mean(self.cv_results_['scores'])
        self.cv_results_['std_dev'] = np.std(self.cv_results_['scores'])

        if metrics:
            for metric in self.cv_results_['additional_metrics']:
                mean_metric = np.mean(self.cv_results_['additional_metrics'][metric])
                self.cv_results_['additional_metrics'][metric] = mean_metric

        self.summary_ = ResultSummary(
            name ='CVSummary').add_results(self.cv_results_)
        if display_results:
            self._display_results()

        return self

    def _display_results(self):
        """
       Displays the results of cross-validation.

        This method prints the mean score, standard deviation, and individual
        fold scores from the cross-validation.

        Raises
        ------
        ValueError
            If this method is called before applying cross-validation.
        """
        if not hasattr(self, 'cv_results_'):
            raise ValueError("Cross-validation not applied. Please call apply_cv_strategy.")
        
        result= {"CV Mean Score":f"{self.cv_results_['mean_score']:.4f}", 
                 "CV Standard Deviation":f"{self.cv_results_['std_dev']:.4f}", 
                 "Scores per fold": list(np.around(self.cv_results_['scores'], 4))
                 }
        summary =ResultSummary(name='CVResult').add_results(result
            )
        print(summary)
    

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, "score_results_" ): 
            raise NotFittedError(msg.format(expobj=self))
        return 1 

class BaseSearch (BaseClass): 
    __slots__=(
        '_base_estimator',
        'grid_params', 
        'scoring',
        'cv', 
        '_strategy', 
        'grid_kws', 
        'best_params_',
        'cv_results_',
        'feature_importances_',
        'best_estimator_',
        'verbose',
        'savejob', 
        'grid_kws',
        )
    def __init__(
        self,
        base_estimator:_F,
        grid_params:Dict[str,Any],
        cv:int =4,
        strategy:str ='GSCV',
        scoring:str = 'nmse',
        savejob:bool=False, 
        filename:str=None, 
        verbose:int=0, 
        **grid_kws
        ): 
        
        self.base_estimator = base_estimator 
        self.grid_params = grid_params 
        self.scoring = scoring 
        self.cv = cv 
        self.best_params_ =None 
        self.cv_results_= None
        self.feature_importances_= None
        self.grid_kws = grid_kws 
        self.strategy = strategy 
        self.savejob= savejob
        self.verbose=verbose
        self.filename=filename
        
    def fit(self, X, y): 
        """
        Fit method using base Estimator and populate BaseSearch attributes.
    
        This method performs hyperparameter tuning on the provided estimator using 
        the specified search strategy and cross-validation. It stores the best 
        parameters, best estimator, and cross-validation results.
    
        Parameters
        ----------
        X : ndarray of shape (M, N)
            Training set; Denotes data that is observed at training and prediction 
            time, used as independent variables in learning. Each sample may be 
            represented by a feature vector, or a vector of precomputed (dis)similarity 
            with each training sample. :code:`X` may also not be a matrix, and may 
            require a feature extractor or a pairwise metric to turn it into one 
            before learning a model.
        
        y : array-like of shape (M,)
            Target values; Denotes data that may be observed at training time as the 
            dependent variable in learning, but which is unavailable at prediction 
            time, and is usually the target of prediction.
    
        Returns
        -------
        self : BaseSearch
            Returns an instance of :class:`~.BaseSearch`.
    
        Notes
        -----
        This method performs hyperparameter tuning by applying the specified search 
        strategy to the base estimator using the provided parameter grid and cross-
        validation. It then stores the best found parameters, the best estimator, and 
        the cross-validation results in the instance attributes.
    
        Examples
        --------
        >>> from gofast.models.search import BaseSearch
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> estimator = RandomForestClassifier()
        >>> param_grid = {'n_estimators': [10, 50], 'max_depth': [5, 10]}
        >>> search = BaseSearch(base_estimator=estimator, grid_params=param_grid,
        ...                       strategy='GSCV')
        >>> search.fit(X, y)
        >>> print(search.best_params_)
        {'n_estimators': 50, 'max_depth': 10}
    
        See Also
        --------
        sklearn.model_selection.GridSearchCV :
            Exhaustive search over specified parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV :
            Randomized search on hyperparameters.
    
        References
        ----------
        .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
               Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
               and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
               Journal of Machine Learning Research, 12, 2825-2830.
        """
        base_estimator = clone(self.base_estimator)
        self.strategy_ = get_strategy_method(self.strategy)
        
        if self.scoring in ('nmse', None): 
            self.scoring = 'neg_mean_squared_error'
        # Assert scoring values 
        get_scorers(scorer=self.scoring, check_scorer=True, error='raise')
         
        gridObj = self.strategy_(
            base_estimator, 
            self.grid_params,
            scoring=self.scoring, 
            cv=self.cv,
            **self.grid_kws
        )
        gridObj.fit(X, y)
        
        params = ('best_params_', 'best_estimator_', 'best_score_', 'cv_results_')
        params_values = [getattr(gridObj, param, None) for param in params] 
        
        for param, param_value in zip(params, params_values):
            setattr(self, param, param_value)
        # Set feature_importances if exists 
        try: 
            attr_value = gridObj.best_estimator_.feature_importances_
        except AttributeError: 
            setattr(self, 'feature_importances_', None)
        else: 
            setattr(self, 'feature_importances_', attr_value)
        
        self.data_ = {f"{get_estimator_name(base_estimator)}": params_values}
        if self.savejob: 
            self.filename = self.filename or get_estimator_name(
                base_estimator) + '.results'
            save_job(job=self.data_, savefile=self.filename) 
    
        results = {get_estimator_name(base_estimator): {
            "best_params_": self.best_params_, 
            "best_estimator_": self.best_estimator_, 
            "best_score_": self.best_score_, 
            "cv_results_": self.cv_results_, 
            }
        }
        self.summary_ = ModelSummary(
            descriptor=f"{self.__class__.__name__}", **results
        ).summary(results)
    
        return self

BaseSearch.__doc__="""\
Fine-tune hyperparameters using grid search methods.

This class performs hyperparameter tuning for a given estimator 
using either Grid Search or Randomized Search. It evaluates the 
estimator and stores the results, allowing for selection of the 
best model.

Parameters
----------
base_estimator : Callable
    Estimator for training set and label evaluation; typically a 
    class that implements a `fit` method. Refer to 
    https://scikit-learn.org/stable/modules/classes.html for 
    more details.

grid_params : list of dict
    List of hyperparameter grids to be fine-tuned. For instance::

        param_grid = [dict(
            kpca__gamma=np.linspace(0.03, 0.05, 10),
            kpca__kernel=["rbf", "sigmoid"]
        )]

{params.core.cv}
    The default is ``4``.

strategy : str, default='GSCV'
        The search strategy to apply for hyperparameter optimization. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.

scoring : str, default='nmse'
    Scoring strategy to evaluate the performance of the cross-validated 
    model on the test set. Should be a single string or a callable.

savejob : bool, default=False
    Save the model parameters to an external file using 'joblib' or 
    Python's 'pickle' module.

filename : str, optional
    Name of the file to save the cross-validation results. If not 
    provided, it will be generated using the estimator name.

verbose : int, default=0
    Controls the verbosity of the output.


{params.core.random_state}

**grid_kws : dict, optional
    Additional keyword arguments passed to the grid search method.

Attributes
----------
best_params_ : dict
    Best hyperparameters found during the search.

best_estimator_ : estimator
    Estimator that was chosen by the search, i.e. estimator which gave 
    highest score (or smallest loss if specified) on the left-out data.

best_score_ : float
    Score of `best_estimator_` on the left-out data.

cv_results_ : dict of numpy (masked) ndarrays
    Cross-validation results.

feature_importances_ : array, shape (n_features,)
    The feature importances (if supported by the `base_estimator`).

data_ : dict
    Dictionary storing the search results.

summary_ : ModelSummary
    Summary of the model search results.

Notes
-----
This class uses either GridSearchCV or RandomizedSearchCV to find the 
best parameters for the provided estimator. The cross-validation 
results and best parameters are stored and can be saved to a file if 
required.

Examples
--------
>>> from pprint import pprint
>>> from gofast.datasets import fetch_data
>>> from gofast.models.search import BaseSearch
>>> from sklearn.ensemble import RandomForestClassifier
>>> X_prepared, y_prepared = fetch_data('bagoue prepared')
>>> grid_params = [
...     dict(n_estimators=[3, 10, 30], max_features=[2, 4, 6, 8]),
...     dict(bootstrap=[False], n_estimators=[3, 10], max_features=[2, 3, 4])
... ]
>>> forest_clf = RandomForestClassifier()
>>> grid_search = BaseSearch(forest_clf, grid_params)
>>> grid_search.fit(X=X_prepared, y=y_prepared)
>>> pprint(grid_search.best_params_)
{{'max_features': 8, 'n_estimators': 30}}
>>> pprint(grid_search.cv_results_)

See Also
--------
sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
    parameter values for an estimator.
sklearn.model_selection.RandomizedSearchCV : Randomized search on 
    hyperparameters.

References
----------
.. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
       Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
       Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
       and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
       Journal of Machine Learning Research, 12, 2825-2830.
""".format (params=_param_docs,
)
   
class SearchMultiple(BaseClass):
    def __init__ (
        self, 
        estimators: _F, 
        scoring:str,  
        grid_params: Dict[str, Any],
        *, 
        strategy:str ='GSCV', 
        cv: int =7, 
        random_state:int =42,
        savejob:bool =False,
        filename: str=None, 
        verbose:int =0,
        **grid_kws, 
        ):
        self.estimators = estimators 
        self.scoring=scoring 
        self.grid_params=grid_params
        self.strategy=strategy 
        self.cv=cv
        self.savejob=savejob
        self.filename=filename 
        self.verbose=verbose 
        self.grid_kws=grid_kws
        
    def fit(self, X: NDArray, y:ArrayLike, ):
        """ Fit methods, evaluate each estimator and store models results.
        
        Parameters 
        -----------
        {params.core.X}
        {params.core.y}
        
        Returns 
        --------
        {returns.self}

        """.format( 
            params =_param_docs , 
            returns = _core_docs['returns'] 
        ) 
        err_msg = (" Each estimator must have its corresponding grid params,"
                   " i.e estimators and grid params must have the same length."
                   " Please provide the appropriate arguments.")
        try: 
            check_consistent_length(self.estimators, self.grid_params)
        except ValueError as err : 
            raise ValueError (str(err) +f". {err_msg}")

        self.best_estimators_ =[] 
        self.data_ = {} 
        models_= {}
        msg =''
        
        self.filename = self.filename or '__'.join(
            [get_estimator_name(b) for b in self.estimators ])
        
        for j, estm in enumerate(self.estimators):
            estm_name = get_estimator_name(estm)
            msg = f'{estm_name} is evaluated with {self.strategy}.'
            searchObj = BaseSearch(base_estimator=estm, 
                                    grid_params= self.grid_params[j], 
                                    cv = self.cv, 
                                    strategy=self.strategy, 
                                    scoring=self.scoring, 
                                    **self.grid_kws
                                    )
            searchObj.fit(X, y)
            best_model_clf = searchObj.best_estimator_ 
            
            if self.verbose > 7 :
                msg += ( 
                    f"\End {self.strategy} search. Set estimator {estm_name!r}"
                    " best parameters, cv_results and other importances" 
                    " attributes\n'"
                 )
            self.data_[estm_name]= {
                                'best_model_':searchObj.best_estimator_ ,
                                'best_params_':searchObj.best_params_ , 
                                'cv_results_': searchObj.cv_results_,
                                'grid_params':self.grid_params[j],
                                'scoring':self.scoring, 
                                "grid_kws": self.grid_kws
                                    }
            
            models_[estm_name] = searchObj
            
            
            msg += ( f"Cross-evaluatation the {estm_name} best model."
                    f" with KFold ={self.cv}"
                   )
            bestim_best_scores, _ = dummy_evaluation(
                best_model_clf, 
                X,
                y,
                cv = self.cv, 
                scoring = self.scoring,
                display ='on' if self.verbose > 7 else 'off', 
                )
            # store the best scores 
            self.data_[f'{estm_name}']['best_scores']= bestim_best_scores
    
            self.best_estimators_.append((estm, searchObj.best_estimator_,
                          searchObj.best_params_, 
                          bestim_best_scores) 
                        )
            
        # save models into a Box 
        d = {**models_, ** dict( 
            keys_ = list (models_.values() ), 
            values_ = list (models_.values() ), 
            models_= models_, 
            )
            
            }
        self.models= Boxspace(**d) 
        
        if self.savejob:
            msg += ('\Serialize the dict of fine-tuned '
                    f'parameters to `{self.filename}`.')
            save_job (job= self.data_ , savefile = self.filename )
            _logger.info(f'Dumping models `{self.filename}`!')
            
            if self.verbose: 
                pprint(msg)
                bg = ("Job is successfully saved. Try to fetch your job from "
                       f"{self.filename!r} using")
                lst =[ "{}.load('{}') or ".format('joblib', self.filename ),
                      "{}.load('{}')".format('pickle', self.filename)]
                
                listing_items_format(lst, bg )
    
        if self.verbose:  
            pprint(msg)    
            
        self.summary_= ModelSummary(
            descriptor = f"{self.__class__.__name__}",  **self.data_
            ).summary(self.data_)

        return self 

SearchMultiple.__doc__="""\
Search and find the best parameters for multiple estimators.

This class performs hyperparameter tuning for multiple estimators
using either Grid Search or Randomized Search. It evaluates each 
estimator and stores the results, allowing for comparison and 
selection of the best models.

Parameters
----------
estimators: list of callable obj 
    list of estimator objects to fine-tune their hyperparameters 
    For instance::
        
    random_state=42
    # build estimators
    logreg_clf = LogisticRegression(random_state =random_state)
    linear_svc_clf = LinearSVC(random_state =random_state)
    sgd_clf = SGDClassifier(random_state = random_state)
    svc_clf = SVC(random_state =random_state) 
               )
    estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
 
grid_params: list 
    List of parameter grids to search over. Each dictionary in the list 
    corresponds to one estimator and contains the parameters to tune.
    For instance::
        
        grid_params= ([
        dict(C=[1e-2, 1e-1, 1, 10, 100], gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],
                     kernel=['rbf']), 
        dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3], 
         'C': [1e-2, 1e-1, 1, 10, 100])], 
        [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
        [dict()], [dict()]
        )
{params.core.cv} 

{params.core.scoring}
   
strategy : str, default='GridSearchCV'
    The search strategy to apply for hyperparameter optimization. 
    Supported strategies include:
    
    - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
    - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
      Validation.
    - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
    - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
    - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
    - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
      Optimization.
    - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
      Search.
    - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
    - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.
    
{params.core.random_state} 

savejob: bool, default=False
    Save your model parameters to external file using 'joblib' or Python 
    persistent 'pickle' module. Default sorted to 'joblib' format. 
    
filename : str, optional
    Name of the file to save the cross-validation results. If not 
    provided, it will be generated using the estimator names.

{params.core.verbose} 

**grid_kws : dict, optional
    Additional keyword arguments passed to the grid search method.
   
Attributes
----------
best_estimators_ : list
    List of tuples containing the estimator, the best estimator found, 
    the best parameters, and the best scores.

data_ : dict
    Dictionary storing the results for each estimator.

models : Boxspace
    Boxspace object containing all the models.

summary_ : ModelSummary
    Summary of the model search results.
    
Notes
-----
This class uses either GridSearchCV or RandomizedSearchCV to find the 
best parameters for each estimator. The cross-validation results and 
best parameters are stored and can be saved to a file if required.

See Also
--------
sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
    parameter values for an estimator.
sklearn.model_selection.RandomizedSearchCV : Randomized search on 
    hyperparameters.

References
----------
.. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
       Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
       Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
       and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
       Journal of Machine Learning Research, 12, 2825-2830.
.. [2] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE 
       WESCON Convention Record, New York, 96-104.
       
Examples
--------
>>> from gofast.search import SearchMultiple , displayFineTunedResults
>>> from sklearn.svm import SVC, LinearSVC 
>>> from sklearn.linear_model import SGDClassifier,LogisticRegression
>>> X, y  = gf.fetch_data ('bagoue prepared') 
>>> X
... <344x18 sparse matrix of type '<class 'numpy.float64'>'
... with 2752 stored elements in Compressed Sparse Row format>
>>> # As example, we can build 04 estimators and provide their 
>>> # grid parameters range for fine-tuning as ::
>>> random_state=42
>>> logreg_clf = LogisticRegression(random_state =random_state)
>>> linear_svc_clf = LinearSVC(random_state =random_state)
>>> sgd_clf = SGDClassifier(random_state = random_state)
>>> svc_clf = SVC(random_state =random_state) 
>>> estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
>>> grid_params= (
...     [dict(C=[1e-2, 1e-1, 1, 10, 100], 
...             gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],kernel=['rbf']), 
...         dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3],
...                        C= [1e-2, 1e-1, 1, 10, 100])],
...     [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
...     [dict()], # we just no provided parameter for demo
...     [dict()]
...    )
>>> search = SearchMultiple(
...     estimators=estimators,
...     grid_params=grid_params,
...     cv=4,
...     scoring='accuracy',
...     verbose=1,
...     savejob=False,
...     strategy='GridSearchCV'
... ).fit(X, y)
 
""".format (params=_param_docs,
)
    
class BaseEvaluation (BaseClass): 
    def __init__(
        self, 
        estimator: _F,
        cv: int = 4,  
        pipeline: List[_F]= None, 
        prefit: bool = False, 
        scoring: str = 'nmse',
        random_state: int = 42, 
        batch_ratio: float = 0.75, 
        verbose: int = 0, 
    ): 
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        
        self.estimator = estimator
        self.cv = cv 
        self.pipeline = pipeline
        self.prefit = prefit 
        self.scoring = scoring
        self.batch_ratio = batch_ratio 
        self.random_state = random_state
        self.verbose = verbose 
        
    def fit(self, X, y):
        """
        Quick method used to evaluate the estimator, display the error results 
        as well as the sample model predictions.
        
        This method performs a quick evaluation of the given estimator using 
        cross-validation. It samples a portion of the provided data, fits the 
        estimator, and computes relevant evaluation metrics such as mean squared 
        error (MSE) and root mean squared error (RMSE).
    
        Parameters
        ----------
        X : ndarray of shape (M, N)
            Training set; denotes data that is observed at training and prediction 
            time, used as independent variables in learning. When a matrix, each 
            sample may be represented by a feature vector, or a vector of precomputed 
            (dis)similarity with each training sample. :code:`X` may also not be a 
            matrix, and may require a feature extractor or a pairwise metric to turn 
            it into one before learning a model.
        
        y : array-like, shape (M,)
            Target values; denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction.
        
        batch_ratio : float, default=0.75
            The ratio to sample `X` and `y`. The default samples 75% of the 
            data. If given, will sample the `X` and `y`. If ``None``, will sample 
            half of the data.
    
        Returns
        -------
        self : BaseEvaluation
            Returns an instance of :class:`~.BaseEvaluation`.
    
        Notes
        -----
        This method checks the estimator, samples the data based on the 
        `batch_ratio` parameter, and performs cross-validation to compute 
        evaluation metrics such as mean squared error (MSE) and root mean squared 
        error (RMSE). If a pipeline is provided, it transforms the data 
        accordingly.
    
        The following steps are performed:
        1. Data sampling based on `batch_ratio`.
        2. Estimator validation and initialization.
        3. Data transformation using the provided pipeline (if any).
        4. Cross-validation and metric computation.
    
        Examples
        --------
        >>> from gofast.models.search import BaseEvaluation
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.datasets import fetch_california_housing
        >>> X, y = fetch_california_housing(return_X_y=True)
        >>> estimator = RandomForestRegressor()
        >>> evaluation = BaseEvaluation(estimator=estimator)
        >>> evaluation.fit(X, y)
        >>> print(evaluation.cv_scores_)
        [-0.4164067  -0.38688256 -0.42903042 -0.5582246 ]
    
        See Also
        --------
        sklearn.model_selection.cross_val_score : Evaluate a score by cross-validation.
        sklearn.metrics.mean_squared_error : Mean squared error regression loss.
    
        References
        ----------
        .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
               Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
               and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
               Journal of Machine Learning Research, 12, 2825-2830.
        """
        # Pass when pipeline is supplied. Expect data to be transformed into numeric dtype
        dtype = object if self.pipeline is not None else "numeric"
        X, y = check_X_y(X, y, to_frame=True, dtype=dtype,
                         estimator=get_estimator_name(self.estimator))
    
        self.estimator = self._check_callable_estimator(self.estimator)
    
        self._logging.info(
            'Quick estimation using the %r estimator with config %r arguments %s.'
            % (repr(self.estimator), self.__class__.__name__, 
               inspect.getfullargspec(self.__init__)))
    
        batch_ratio = validate_ratio(
            self.batch_ratio, bounds=(0, 1.),  param_name="batch_ratio")
        # Sampling train data. Use 75% by default among data
        n = int(batch_ratio * len(X))
        if hasattr(X, 'columns'):
            X = X.iloc[:n]
        else:
            X = X[:n, :]
        y = y[:n]
    
        if self.pipeline is not None:
            X = self.pipeline.fit_transform(X)
    
        if not self.prefit:
            # For consistency
            if self.scoring is None:
                warnings.warn("'neg_mean_squared_error' scoring is used when"
                              " scoring parameter is ``None``.")
                self.scoring = 'neg_mean_squared_error'
            self.scoring = "neg_mean_squared_error" if self.scoring in (
                None, 'nmse') else self.scoring
    
            self.mse_, self.rmse_, self.cv_scores_ = self._fit(
                X, y, 
                self.estimator, 
                cv_scores=True,
                scoring=self.scoring
            )
            if not isinstance (self.scoring, str): 
                scoring_name = get_estimator_name(self.scoring) 
            else: scoring_name = str(self.scoring)
            
            self.results_= ResultSummary("PrefitResults").add_results(
                {f"{scoring_name}": np.average (self.rmse_), 
                 "cv_scores": self.cv_scores_}
                )
            
        return self
    
    def predict(self, X): 
        """
        Quick prediction and get the scores.
        
        Parameters 
        ----------
        X : Ndarray (M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Test set; Denotes data that is observed at testing and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. `X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one 
            before learning a model.
            
        Returns 
        -------
        y : array-like, shape (M, ) ``M=m-samples``
            Test predicted target values.
        
        Notes
        -----
        This method ensures that the `X` array is properly formatted and preprocessed
        before making predictions using the fitted estimator. If a pipeline is provided,
        it applies the pipeline transformations to `X` before prediction. The method
        raises a `NotFittedError` if the estimator has not been fitted yet.
        
        Examples
        --------
        >>> from gofast.models.search import BaseEvaluation
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> X, y = data.data, data.target
        >>> evaluator = BaseEvaluation(estimator=DecisionTreeClassifier())
        >>> evaluator.fit(X, y)
        >>> X_test = X[:10]
        >>> predictions = evaluator.predict(X_test)
        >>> print(predictions)
        
        See Also
        --------
        sklearn.base.BaseEstimator : Base class for all estimators in scikit-learn.
        
        References
        ----------
        .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
               Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
               and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
               Journal of Machine Learning Research, 12, 2825-2830.
        """
        self.inspect
        
        dtype = object if self.pipeline is not None else "numeric"
        
        X = check_array(
            X, accept_sparse=False, 
            input_name='X', dtype=dtype, 
            estimator=get_estimator_name(self.estimator),
        )
        
        if self.pipeline is not None: 
            X = self.pipeline.fit_transform(X) 
    
        return self.estimator.predict(X)

    def _fit(self, X, y, estimator, cv_scores=True,
             scoring='neg_mean_squared_error'):
        """
        Fit data once verified and compute performance scores.
    
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) 
            Training data for fitting.
        y: array-like of shape (n_samples,) 
            Target for training.
        estimator: callable or scikit-learn estimator 
            Callable or something that has a fit method. Can build your 
            own estimator following the API reference via 
            https://scikit-learn.org/stable/modules/classes.html.
        cv_scores: bool, default=True 
            Compute the cross-validation scores.
        scoring: str, default='neg_mean_squared_error' 
            Metric for scores evaluation. 
            Type of scoring for cross-validation. Please refer to 
            :doc:`~.sklearn.model_selection.cross_val_score` for further details.
                
        Returns
        -------
        tuple
            - mse: Mean Squared Error (for regression) or None (for classification).
            - rmse: Root Mean Squared Error (for regression) or accuracy (for classification).
            - scores: Cross-validation scores.
        """
        mse = rmse = None  
        
        def display_scores(scores): 
            """ Display scores..."""
            n = ("Scores:", "Means:", "RMSE/Accuracy scores:", "Standard Deviation:")
            p = (scores, scores.mean(), np.sqrt(scores) if 
                 self.scoring == 'neg_mean_squared_error' 
                 else scores.mean(), scores.std())
            for k, v in zip(n, p): 
                print(f"{k} {v}")
                    
        self._logging.info("Fit data with a supplied pipeline or using purely estimator")
    
        estimator.fit(X, y)
        y_pred = estimator.predict(X)
    
        if self.scoring in ('neg_mean_squared_error', 'r2'):  # if regression task
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
        else:  # if classification task
            rmse = accuracy_score(y, y_pred)
        
        scores = None 
        if cv_scores: 
            scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring)
            if self.scoring == 'neg_mean_squared_error': 
                rmse = np.sqrt(-scores)
            elif self.scoring in ('accuracy', 'precision', 'recall', 'f1'):
                rmse = scores
            if self.verbose:
                if self.scoring == 'neg_mean_squared_error': 
                    scores = -scores 
                display_scores(scores)   
                    
        return mse, rmse, scores

    def _check_callable_estimator(self, base_est):
        """
        Check whether the estimator is callable or not.
    
        If callable, use the default parameter for initialization.
    
        Parameters
        ----------
        base_est : object
            The estimator to check and possibly initialize.
    
        Returns
        -------
        estimator : object
            The validated and possibly initialized estimator.
    
        Raises
        ------
        EstimatorError
            If the estimator does not have a `fit` method.
    
        Notes
        -----
        This method ensures that the estimator is suitable for use by 
        checking if it has a `fit` method and initializing it if it is callable.
        """
        if not hasattr(base_est, 'fit'):
            raise EstimatorError(
                f"Wrong estimator {get_estimator_name(base_est)!r}. Each"
                " estimator must have a fit method. Refer to scikit-learn"
                " https://scikit-learn.org/stable/modules/classes.html API"
                " reference to build your own estimator.")
    
        return base_est() if callable(base_est) else base_est

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'cv_scores_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
  
BaseEvaluation.__doc__="""\
Base class for evaluating machine learning models using cross-validation and
pipeline transformations.

This class provides methods for quick evaluation of estimators, including
cross-validation and performance metrics calculation. It supports integration
with pipelines for data preprocessing and transformation.

Parameters
----------
estimator : _F
    The estimator object to evaluate. Must implement a `fit` method.


{params.core.cv}
    The number of folds for cross-validation. The default is ``4``.
    
pipeline : List[_F], optional
    A list of pipeline steps for preprocessing and transforming the data.
    Each step must implement a `fit_transform` method.

prefit : bool, optional, default=False
    If `True`, the estimator is assumed to be pre-fitted and will not be
    refitted during evaluation.

scoring : str, optional, default='nmse'
    The scoring strategy to evaluate the model performance. Default is 
    'neg_mean_squared_error' ('nmse'). Refer to scikit-learn's scoring 
    parameter documentation for valid values.

random_state : int, optional, default=42
    The random seed for reproducibility of results.

batch_ratio : float, optional, default=0.75
    The ratio of the data to be used for evaluation. Default is 0.75,
    meaning 75% of the data will be used for training and evaluation.
    Must be between 0 and 1.

verbose : int, optional, default=0
    The verbosity level of logging information. Higher values indicate
    more verbose output.

Attributes
----------
estimator : _F
    The estimator object being evaluated.

cv : int
    The number of folds for cross-validation.

pipeline : List[_F]
    The list of pipeline steps for preprocessing and transforming the data.

prefit : bool
    Indicates whether the estimator is pre-fitted.

scoring : str
    The scoring strategy used for model evaluation.

batch_ratio : float
    The ratio of the data used for evaluation.

random_state : int
    The random seed for reproducibility of results.

verbose : int
    The verbosity level of logging information.
    
results_ : ResultSummary
    Contains the scores ( averaged and cv) for the estimator. This attribute is'
    available only if `prefit` is set to ``False``.
    
Methods
-------
fit(X, y):
    Fit the model to the data and evaluate performance using cross-validation.

predict(X):
    Make predictions on the test data.

Notes
-----
The class supports both regression and classification tasks. For regression
tasks, the default scoring is 'neg_mean_squared_error' (nmse).

The evaluation process involves the following steps:
1. Data preprocessing using the provided pipeline.
2. Fitting the estimator to the training data.
3. Evaluating the performance using cross-validation.

Examples
--------
>>> from gofast.models.search import BaseEvaluation
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.datasets import load_iris
>>> data = load_iris()
>>> X, y = data.data, data.target
>>> evaluator = BaseEvaluation(estimator=DecisionTreeClassifier())
>>> evaluator.fit(X, y)
>>> X_test = X[:10]
>>> predictions = evaluator.predict(X_test)
>>> print(predictions)

Example 2
>>> import gofast as gf 
>>> from gofast.datasets import load_bagoue 
>>> from gofast.models import BaseEvaluation 
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.model_selection import train_test_split
>>> X, y = load_bagoue (as_frame =True ) 
>>> # categorizing the labels 
>>> yc = gf.smart_label_classifier (y , values = [1, 3, 10 ], 
                                 # labels =['FR0', 'FR1', 'FR2', 'FR4'] 
                                 ) 
>>> # drop the subjective columns ['num', 'name'] 
>>> X = X.drop (columns = ['num', 'name']) 
>>> # X = gf.cleaner (X , columns = 'num name', mode='drop') 
>>> X.columns 
Index(['shape', 'type', 'geol', 'east', 'north', 'power', 'magnitude', 'sfi',
       'ohmS', 'lwi'],
      dtype='object')
>>> X =  gf.naive_imputer ( X, mode ='bi-impute') # impute data 
>>> # create a pipeline for X 
>>> pipe = gf.make_naive_pipe (X) 
>>> Xtrain, Xtest, ytrain, ytest = train_test_split(X, yc) 
>>> b = BaseEvaluation (estimator= RandomForestClassifier, 
                        scoring = 'accuracy', pipeline = pipe)
>>> b.fit(Xtrain, ytrain ) # accepts only array 
>>> b.cv_scores_ 
Out[174]: array([0.75409836, 0.72131148, 0.73333333, 0.78333333])
>>> ypred = b.predict(Xtest)
>>> scores = gf.sklearn.accuracy_score (ytest, ypred) 
0.7592592592592593

See Also
--------
sklearn.model_selection.cross_val_score : Evaluate a score by cross-validation.

References
----------
.. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
       Grisel, O., Blondel, P., Prettenhofer, P., Weiss, R., Dubourg, V., 
       Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
       and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
       Journal of Machine Learning Research, 12, 2825-2830.
""".format (params=_param_docs,
)
