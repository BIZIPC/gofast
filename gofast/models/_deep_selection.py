# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np
from math import log
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.model_selection._split import check_cv
from sklearn.utils.validation import indexable
from sklearn.utils.parallel import Parallel, delayed

from .utils import aggregate_cv_results 

__all__=["HyperbandSearchCV"]

class HyperbandSearchCV(BaseSearchCV):
    r"""
    Performs hyperparameter optimization using the Hyperband algorithm, 
    a bandit-based approach to efficiently identify the best hyperparameters 
    for a given model by dynamically allocating and early-stopping resources 
    for low-performing configurations.

    The Hyperband algorithm optimizes computational resources through 
    successive halving, effectively balancing the exploration of the hyperparameter 
    space with the exploitation of promising configurations.

    The Hyperband algorithm is based on the idea of dynamically allocating 
    resources to different configurations of hyperparameters based on their 
    performance. Given a maximum amount of resource `R` that can be allocated 
    to a single configuration and a downsampling rate `eta`, Hyperband 
    computes two key values:
    
    - `s_max` = :math:`\lfloor \log_{\eta}(R) \rfloor`, the maximum number of 
      iterations/brackets.
    - `B` = :math:`(s\_max + 1) \cdot R`, the total budget to be used across 
      all brackets.
    
    For each bracket `s` in :math:`\{s\_max, s\_max - 1, \ldots, 0\}`:
    
    - The number of configurations evaluated is :math:`n = \lceil \frac{B}{R} 
      \cdot \frac{\eta^s}{(s+1)} \rceil`.
    - The amount of resource allocated to each configuration is 
      :math:`r = R \cdot \eta^{-s}`.

    This process allows for both exploration of the hyperparameter space at 
    lower resource levels and intensive exploitation of promising configurations 
    at higher resource levels.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn estimator. Must implement the fit method.
    param_distributions : dict
        Dictionary where the keys are parameters and values are distributions 
        or lists from which to sample. Distributions must provide a `rvs` 
        method for sampling.
    max_iter : int, default=81
        Maximum number of iterations/epochs per configuration. Acts as the 
        maximum resource level for the Hyperband algorithm.
    eta : int, default=3
        The reduction factor for pruning configurations in each round of 
        successive halving.
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
    scoring : string, callable, list/tuple, dict or None, default=None
        A single string or a callable to evaluate the predictions on the test 
        set. For evaluating multiple metrics, either give a list of (unique) 
        strings or a dict with names as keys and callables as values.
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a 
        joblib.parallel_backend context. -1 means using all processors.
    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
    pre_dispatch : int, or string, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel 
        execution.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator state used for random uniform sampling 
        from lists of possible values instead of scipy.stats distributions.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=True
        Whether to include train scores.
    
    Attributes
    ----------
    best_params_ : dict
        The parameters that have been chosen by the search process as the best.
    best_estimator_ : estimator object
        The estimator that was chosen by the search, fitted with the best-found 
        parameters.
    cv_results_ : dict
        A dict that describes the fit results for each configuration evaluated.
        The keys are column headers and the values are columns, where each entry
        corresponds to one configuration.
    s_max_ : int
        The maximum number of configurations that can be evaluated, derived from
        `max_iter` and `eta`.
    B_ : int
        The total budget for all configurations across all brackets, calculated
        as `(s_max_ + 1) * max_iter`.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from gofast.experimental import enable_hyperband_selection 
    >>> from gofast.models.deep_selection import HyperbandSearchCV
    >>> X, y = load_iris(return_X_y=True)
    >>> param_distributions = {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto']}
    >>> hyperband = HyperbandSearchCV(estimator=SVC(), 
                                      param_distributions=param_distributions, 
    ...                               max_iter=4, random_state=42)
    >>> hyperband.fit(X, y)
    >>> print(hyperband.best_params_)
    
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import HyperbandSearchCV
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
    ...                            random_state=42)
    >>> param_distributions = {'C': [0.1, 1, 10, 100]}
    >>> hyperband = HyperbandSearchCV(estimator=LogisticRegression(),
    ...                               param_distributions=param_distributions,
    ...                               max_iter=4, cv=5)
    >>> hyperband.fit(X, y)
    >>> print(hyperband.best_params_)
    

    Note
    ----
    HyperbandSearchCV uses the Hyperband algorithm for efficient hyperparameter
    optimization. It dynamically allocates and prunes resources, allowing for 
    a more effective search over the hyperparameter space.
    The actual computation within the method involves training models on subsets 
    of the dataset for varying amounts of resources and iteratively pruning 
    less promising models. The `fit` method supports classification, regression, 
    and clustering estimators following the scikit-learn API.
    
    """

    def __init__(self, 
        estimator, 
        param_distributions, 
        max_iter=81, 
        eta=3, 
        cv=5, 
        scoring=None, 
        n_jobs=None, 
        refit=True, 
        verbose=0, 
        pre_dispatch='2*n_jobs', 
        random_state=None, 
        error_score=np.nan, 
        return_train_score=True
        ):
        self.param_distributions = param_distributions
        self.max_iter = max_iter 
        self.eta = eta 
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state
        self.error_score = error_score
        self.return_train_score = return_train_score
        
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, #iid='deprecated', 
            refit=refit, cv=cv, verbose=verbose, 
            pre_dispatch=pre_dispatch, 
            error_score=error_score, 
            return_train_score=return_train_score
            )
        
    def _get_candidate_params(self):
        """
        Generates a list of parameter candidates for hyperparameter optimization.
    
        This method utilizes the scikit-learn `ParameterSampler` to create a 
        list of parameter settings. It samples from the specified 
        `param_distributions`. The sampling allows for both distributions that 
        can generate random samples (via the `rvs` method) and lists of 
        potential values. The number of parameter settings is determined by 
        `n_candidates`.
    
        Returns
        -------
        param_iterable : list of dict
            A list of dictionaries, where each dictionary represents a unique 
            combination of parameters to be evaluated by the hyperparameter 
            optimization process.
    
        Examples
        --------
        Assuming `self.param_distributions` is defined as:
        ```
        self.param_distributions = {
            'C': scipy.stats.expon(scale=100), 
            'gamma': scipy.stats.expon(scale=.1),
            'kernel': ['rbf'], 
            'class_weight':['balanced', None]
        }
        ```
        and `self.n_candidates` is set to 10, this method will generate a list 
        of 10 dictionaries, each with a unique combination of parameters 
        sampled from the distributions or lists provided in `param_distributions`.
    
        Note
        ----
        The actual sampling behavior and the resulting parameter combinations
        are determined by the `ParameterSampler` class from scikit-learn, which 
        may produce different results for each invocation if a `random_state` 
        is not set or if the distributions support random sampling.
        """
        param_iterable = list(ParameterSampler(
            self.param_distributions, self.n_candidates,
            random_state=self.random_state))
        return param_iterable

    
    def _hyperband_resource_allocation(self):
        """
        Calculates the resource allocation for Hyperband's configurations, 
        including the maximum number of iterations (resources) each 
        configuration should use and the total budget B.
    
        This internal method computes `s_max`, the maximum number of configurations, 
        and `B`, the total budget for all configurations across all brackets, based 
        on the `max_iter` and `eta` parameters of the HyperbandSearchCV instance.
    
        Returns
        -------
        s_max : int
            The maximum number of configurations that can be evaluated.
        B : int
            The total budget for all configurations across all brackets, 
            calculated as `(s_max + 1) * max_iter`. It represents the sum of 
            the maximum resources (max_iter) that can be allocated to each 
            configuration.
    
        Note
        ----
        This method is designed to support the Hyperband algorithm's internal 
        workings by determining how resources are allocated to configurations 
        at different stages of the optimization process. The returned `s_max` 
        and `B` values are used to calculate the number of configurations (`n`) 
        and the amount of resource (`r`) for each bracket in the Hyperband 
        execution loop.
        """
        logeta = lambda x: log(x) / log(self.eta)
        s_max = int(logeta(self.max_iter))
        B = (s_max + 1) * self.max_iter
        return s_max, B
   

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Fits the hyperparameter optimization process using the Hyperband 
        algorithm.
    
        The method takes a dataset, potentially along with labels and groups, 
        and performs hyperparameter optimization by evaluating various 
        configurations across different levels of computational resources. 
        It dynamically allocates more resources to promising configurations 
        using a bandit-based strategy and successive halving.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and 
            n_features is the number of features.
        y : array-like, shape (n_samples,), optional
            Target relative to X for classification or regression; None for 
            unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of the estimator, if any.
    
        Returns
        -------
        self : object
            Instance of the fitted HyperbandSearchCV.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from gofast.models.deep_selection import HyperbandSearchCV
        >>> X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
        ...                            random_state=42)
        >>> param_distributions = {'C': [0.1, 1, 10, 100]}
        >>> hyperband = HyperbandSearchCV(estimator=LogisticRegression(),
        ...                               param_distributions=param_distributions,
        ...                               max_iter=100, cv=5)
        >>> hyperband.fit(X, y)
        >>> print(hyperband.best_params_)
    
        Notes
        -----
        - The actual computation of the method involves training models on subsets of
          the dataset for varying amounts of resources and iteratively pruning less
          promising models.
        - The `fit` method supports classification, regression, and clustering estimators
          following the scikit-learn API.
        """
        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.s_max_, self.B_ = self._hyperband_resource_allocation()

        all_candidate_params = []
        all_outs = []

        for s in reversed(range(self.s_max_ + 1)):
            # Compute initial number of configurations and resources
            n_configs = int(np.ceil(self.B_ / self.max_iter / (s + 1) * self.eta ** s))
            resource = self.max_iter * self.eta ** (-s)

            # Begin Successive Halving for each bracket
            for i in range(s + 1):
                # Update the number of configs and resources for this round
                n_candidates = n_configs * self.eta ** (-i)
                resource = resource * self.eta ** (i)
                
                # Sample candidate parameters for this round
                candidate_params = list(ParameterSampler(
                    self.param_distributions, n_candidates, 
                    random_state=self.random_state))
                
                # Ensure the scoring parameter is correctly set up
                scoring = check_scoring(self.estimator, scoring=self.scoring)
                # Evaluate candidate parameters
                out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(_fit_and_score)(
                        clone(self.estimator), X, y, scoring, train_idx, test_idx,
                        self.verbose, params, fit_params=fit_params,
                        return_train_score=self.return_train_score,
                        error_score=self.error_score,
                        return_parameters=True, 
                        return_times =True, 
                        return_n_test_samples =True, 
                        )
                    for params in candidate_params
                    for train_idx, test_idx in cv.split(X, y, groups)
                )
                # Select top candidates based on current results if not the last round
                if i < s:
                    out= aggregate_cv_results(out)
                    candidate_params = self._select_top_candidates(
                        out, n_configs // self.eta)
      
                # Extend the cumulative lists
                all_candidate_params.extend(candidate_params)
                all_outs.extend(out)
 
        self.cv_results_ = self._format_results(all_outs, all_candidate_params)
        self.best_index_ = np.nanargmax([res['mean_test_score'] for res in self.cv_results_])
        self.best_params_ = self.cv_results_[self.best_index_]['params']
        self.best_score_ = self.cv_results_[self.best_index_]['mean_test_score']
        
        # Refit the best model on the full dataset
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)
        
        return self

    def _format_results(self, outs, candidate_params):
        """
        Formats the raw results from the hyperparameter optimization process 
        into a structured list.
    
        This method takes the output from `_fit_and_score` alongside the 
        corresponding candidate parameters and organizes this information into 
        a list of dictionaries. Each dictionary contains detailed results for 
        a single hyperparameter configuration, including the mean test score, 
        standard deviation of the test score, number of test samples, execution 
        time, and the status of the execution.
    
        Parameters
        ----------
        outs : list of tuples
            The raw output from `_fit_and_score`. Each tuple in the list should 
            contain the mean test score, standard deviation of the test score, 
            number of test samples, execution time in seconds, and execution 
            status for a given hyperparameter configuration.
        candidate_params : list of dicts
            The list of hyperparameter configurations that were evaluated. 
            Each configuration is represented as a dictionary.
    
        Returns
        -------
        formatted_results : list of dicts
            A list where each dictionary represents the results for a single 
            hyperparameter configuration. Keys in the dictionary include 
            'params' (the hyperparameter configuration), 
            'mean_test_score' (the mean score on the test set), 
            'std_test_score' (the standard deviation 
            of the test score), 'n_test_samples' (the number of test samples used),
            'time_sec' (the execution time in seconds), and 'status' 
            (the execution status).
    
        Examples
        --------
        >>> out = [
        ...     (0.95, 0.02, 100, 60, "OK"),
        ...     (0.90, 0.03, 100, 45, "OK")
        ... ]
        >>> candidate_params = [
        ...     {'C': 10, 'kernel': 'linear'},
        ...     {'C': 1, 'kernel': 'linear'}
        ... ]
        >>> formatted_results = self._format_results(out, candidate_params)
        >>> print(formatted_results)
        [
            {'params': {'C': 10, 'kernel': 'linear'}, 'mean_test_score': 0.95, 
             'std_test_score': 0.02, 
             'n_test_samples': 100, 'time_sec': 60, 'status': 'OK'},
            {'params': {'C': 1, 'kernel': 'linear'}, 'mean_test_score': 0.90,
             'std_test_score': 0.03, 
             'n_test_samples': 100, 'time_sec': 45, 'status': 'OK'}
        ]
    
        Note
        ----
        The 'status' field in the result dictionaries can be used to identify 
        configurations that failed during evaluation (e.g., due to convergence issues) 
        and handle them accordingly in the hyperparameter optimization process.
        """
        formatted_results = []
        for score, candidate_param in zip(outs, candidate_params):
            # Unpack with flexibility, accommodating different lengths of score tuples
            # The first element should always be mean_test_score
            mean_test_score = score.get('mean_test_score', np.nan)  
            std_test_score = score.get('std_test_score', np.nan) if len(score) > 1 else np.nan
            n_test_samples = score.get('n_test_samples', None) if len(score) > 2 else np.nan
            time_sec = score.get('fit_times', np.nan) if len(score) > 3 else np.nan
            status = score if len(score) > 4 else 'OK'  # Assume 'OK' if not provided
            
            result = {
                'params': candidate_param,
                'mean_test_score': mean_test_score,
                'std_test_score': std_test_score,
                'n_test_samples': n_test_samples,
                'fit_times': time_sec,
                'status': status
            }
            formatted_results.append(result)

        return formatted_results

    def _select_top_candidates(self, results, n_candidates):
        """
        Selects the top-performing hyperparameter configurations based on 
        the mean test score.
    
        This method sorts the evaluated hyperparameter configurations by their
        performance, measured by the mean test score, in descending order. It 
        then selects the top `n_candidates` configurations for further evaluation 
        or final selection.
    
        Parameters
        ----------
        results : list of dicts
            The list of dictionaries containing the results for each evaluated 
            hyperparameter configuration. Each dictionary must include at least 
            the 'params' key with the hyperparameter configuration and the 
            'mean_test_score' key with the configuration's performance metric.
        n_candidates : int
            The number of top-performing configurations to select.
    
        Returns
        -------
        top_candidate_params : list of dicts
            A list of dictionaries, where each dictionary represents the 
            parameters of a top-performing hyperparameter configuration. The 
            list is ordered by descending performance, with the best 
            configuration first.
    
        Examples
        --------
        >>> results = [
        ...     {'params': {'C': 10, 'kernel': 'linear'}, 'mean_test_score': 0.95},
        ...     {'params': {'C': 1, 'kernel': 'linear'}, 'mean_test_score': 0.90},
        ...     {'params': {'C': 0.1, 'kernel': 'linear'}, 'mean_test_score': 0.85}
        ... ]
        >>> n_candidates = 2
        >>> top_candidate_params = self._select_top_candidates(results, n_candidates)
        >>> print(top_candidate_params)
        [{'C': 10, 'kernel': 'linear'}, {'C': 1, 'kernel': 'linear'}]
    
        Note
        ----
        The selection of top candidates is solely based on the 'mean_test_score' 
        value. In the case of ties, the method selects the configurations that 
        appear first in the `results` list, which does not account for potential
        differences in other metrics such as standard deviation of scores or 
        computational resources used.
        """
        sorted_results = sorted(results, key=lambda x: x['mean_test_score'], reverse=True)
        top_candidates = sorted_results[:n_candidates]
        top_candidate_params = [candidate['params'] for candidate in top_candidates]
        
        return top_candidate_params

