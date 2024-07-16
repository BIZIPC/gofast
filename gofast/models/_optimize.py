# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Optimize Base Classes
"""

import joblib
from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from tqdm import tqdm 
import numpy as np 

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator

from ..api.summary import ModelSummary, ResultSummary
from ..api.types import Any, Dict, List,Union, Tuple, Optional, ArrayLike
from ..tools.coreutils import get_params, smart_format
from ..tools.validator import filter_valid_kwargs, get_estimator_name
from .utils import get_strategy_method, align_estimators_with_params


class BaseOptimizer(metaclass=ABCMeta):
    """
    Base class for hyperparameter optimization of multiple estimators.

    This abstract base class provides common functionality for optimizing 
    hyperparameters of machine learning estimators. Subclasses should 
    implement specific optimization strategies and call the methods provided 
    by this base class.

    Parameters
    ----------
    estimators : dict of str, estimator objects
        A dictionary where keys are estimator names and values are estimator 
        instances. Each estimator is an instance of a model to be optimized.

    param_grids : dict of str, dict
        A dictionary where keys are estimator names (matching those in 
        `estimators`) and values are parameter grids. Each parameter grid is 
        a dictionary where the keys are parameter names and the values are 
        lists of parameter settings to try for that parameter.

    strategy : str, default='GSCV'
        The search strategy to apply for hyperparameter optimization. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWSCV', 'PSOSCV' for Swarm Optimization.
        - 'SQSCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or a scorer callable 
        object / function with signature `scorer(estimator, X, y)`.

    cv : int or cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs 
        for `cv` are:
          - None, to use the default 5-fold cross-validation,
          - int, to specify the number of folds in a (Stratified)KFold,
          - CV splitter,
          - An iterable yielding (train, test) splits as arrays of indices.

    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for the optimization process. 
        `-1` means using all processors.

    **search_kwargs : dict, optional
        Additional keyword arguments to pass to the search constructor.

    Methods
    -------
    _control_strategy()
        Control and validate the strategy class and filter search keyword 
        arguments.

    _validate_parameters()
        Validate and align the estimators and parameter grids to ensure they 
        are properly configured for hyperparameter optimization.
    
    _validate_search_params()
        Validate and align the search parameters, ensuring that the estimators, 
        parameter grids, and the search strategy are properly configured before 
        performing hyperparameter optimization.

    save_results_to_file(results_dict, filename=None)
        Save the optimization results to a joblib file.

    construct_model_summary(results_dict, descriptor="BaseOptimizer")
        Construct a `ModelSummary` object from the optimization results.

    __repr__()
        Return a string representation of the BaseOptimizer object.

    __str__()
        Return a user-friendly string representation of the BaseOptimizer 
        object.

    Notes
    -----
    This class is intended to be inherited by specific optimizer classes. 
    It provides common functionality such as saving results and constructing 
    summaries, which helps to avoid code repetition.

    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
        parameters.
    ModelSummary : Class for summarizing model results.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.
    """
    
    @abstractmethod
    def __init__(
        self, 
        estimators, 
        param_grids, 
        strategy='GSCV', 
        scoring=None,
        cv=None, 
        save_results=False, 
        n_jobs=-1, 
        **search_kwargs
        ):
        self.estimators = estimators
        self.param_grids = param_grids
        self.strategy = strategy
        self.scoring = scoring
        self.cv = cv
        self.save_results = save_results
        self.n_jobs = n_jobs
        self.search_kwargs = search_kwargs
        self.summary_ = None
        
    def _control_strategy(self):
        """
        Control and validate the strategy class and filter search keyword 
        arguments.
    
        This method sets the strategy class and filters the search keyword 
        arguments to ensure they are valid for the chosen strategy.
        """
        # Just control strategy class and filter search kwargs
        self.strategy_ = get_strategy_method(self.strategy)
        self.search_kwargs = filter_valid_kwargs(self.strategy_, self.search_kwargs)
        
    def _validate_parameters(self):
        """
        Validate and align the estimators and parameter grids.
    
        This method ensures that the estimators and their corresponding parameter 
        grids are properly aligned. It uses the `align_estimators_with_params` 
        function to achieve this alignment.
    
        Notes
        -----
        This method modifies the `self.estimators` and `self.param_grids` 
        attributes to ensure they are correctly aligned. This is a crucial step 
        before performing hyperparameter optimization to ensure that each 
        estimator has the appropriate set of parameters to be optimized.
    
        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> from sklearn.linear_model import SGDClassifier
        >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
        >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
        ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}}
        >>> optimizer = Optimizer(estimators, param_grids)
        >>> optimizer._validate_parameters()
        >>> print(optimizer.estimators)
        >>> print(optimizer.param_grids)
    
        See Also
        --------
        align_estimators_with_params : Function that aligns parameter grids with 
            estimators.
        """
        self.estimators, self.param_grids = align_estimators_with_params(
            self.param_grids, self.estimators)
        
    def _validate_search_params(self):
        """
        Validate and align the search parameters.
    
        This method validates the estimators and their parameter grids, and 
        controls and validates the chosen strategy and search keyword arguments.
    
        Notes
        -----
        This method ensures that the estimators, parameter grids, and the search 
        strategy are properly configured before performing hyperparameter 
        optimization.
        """
        self._validate_parameters()
        self._control_strategy()
        self.estimators_nickname 
        
    def save_results_to_file(self, results_dict, filename=None):
        """
        Save the optimization results to a joblib file.
    
        Parameters
        ----------
        results_dict : dict
            The dictionary containing the optimization results.
    
        filename : str, optional
            The filename to save the results. If not provided, the default 
            "optimization_results.joblib" will be used.
        """
        if self.save_results:
            filename = filename or "optimization_results.joblib"
            joblib.dump(results_dict, filename)
            print(f"Results saved to {filename}")

    def construct_model_summary(self, results_dict, descriptor="BaseOptimizer"):
        """
        Construct a `ModelSummary` object from the optimization results.
    
        Parameters
        ----------
        results_dict : dict
            The dictionary containing the optimization results.
    
        descriptor : str, default='BaseOptimizer'
            A descriptor for the model summary.
    
        Returns
        -------
        summary_ : ModelSummary
            The constructed `ModelSummary` object.
        """
        self.summary_ = ModelSummary(descriptor=descriptor, **results_dict)
        self.summary_.summary(results_dict)
        return self.summary_
    
    def get_estimator_shortname(self, estimator, existing_short_names):
        """
        Generate a short name for an estimator based on its class name.
    
        Parameters
        ----------
        estimator : estimator object
            The estimator instance for which to generate a short name.
    
        existing_short_names : set
            A set of short names that are already in use to ensure uniqueness.
    
        Returns
        -------
        short_name : str
            The generated short name for the estimator.
        """
        name_map = {
            'LinearRegression': 'LinReg',
            'LogisticRegression': 'LogReg',
            'RandomForestClassifier': 'RF',
            'GradientBoostingClassifier': 'GBM',
            'GradientBoostingRegressor': 'GBM',
            'SupportVectorClassifier': 'SVM',
            'SupportVectorRegressor': 'SVM',
            'KNeighborsClassifier': 'KNN',
            'DecisionTreeClassifier': 'DT',
            'DecisionTreeRegressor': 'DT',
            'AdaBoostClassifier': 'AB',
            'AdaBoostRegressor': 'AB',
            'ExtraTreesClassifier': 'ET',
            'ExtraTreesRegressor': 'ET',
            'MultinomialNB': 'MNB',
            'BernoulliNB': 'BNB',
            'GaussianNB': 'GNB',
            'LinearDiscriminantAnalysis': 'LDA',
            'QuadraticDiscriminantAnalysis': 'QDA', 
            'XGBClassifier': "XGB"
            # ...
        }
    
        estimator_name = estimator.__class__.__name__
        short_name = name_map.get(estimator_name, '')
    
        if not short_name:
            # Create a short name by taking the first letter of each capitalized word
            short_name = ''.join([word[0] for word in estimator_name.split() if word[0].isupper()])
            if not short_name:
                short_name = estimator_name[:3]
    
        # Ensure the short name is unique
        original_short_name = short_name
        counter = 1
        while short_name in existing_short_names:
            short_name = f"{original_short_name}{counter}"
            counter += 1
        
        return short_name
    
    def generate_estimator_shortnames(self):
        """
        Generate short names for a list of estimators.
    
        Parameters
        ----------
        estimators : list of estimator objects
            The list of estimator instances for which to generate short names.
    
        Returns
        -------
        short_names : dict
            A dictionary where the keys are the original estimator class names 
            and the values are the generated short names.
        """
        short_names = {}
        existing_short_names = set()
        for estimator in self.estimators:
            original_name = get_estimator_name(estimator)
            short_name = self.get_estimator_shortname(estimator, existing_short_names)
            short_names[original_name] = short_name
            existing_short_names.add(short_name)
        return short_names
    
    @property
    def estimators_nickname(self):
        """
        Generate a concatenated string of short names for all estimators.

        This property generates short names for each estimator and concatenates
        them into a single string, separated by dots.

        Returns
        -------
        nickname : str
            The concatenated short names of all estimators.
        """
        self.estimators_nickname_ = '.'.join(
            self.generate_estimator_shortnames().values())
        return self.estimators_nickname_
        
    def __repr__(self):
        """
        Return a string representation of the BaseOptimizer object.
    
        This method provides a concise summary of the BaseOptimizer instance, 
        including the names of the estimators and the type of optimization 
        technique being used. If the optimization has been performed, it also 
        includes a summary of the results.
    
        Returns
        -------
        str
            A string representation of the BaseOptimizer object.
        """
        if self.summary_ is None:
            param_values = get_params(self)
            summary = ResultSummary(name=self.__class__.__name__, mute_note=True)
            summary.add_results(param_values)
            message = (f"[{self.__class__.__name__} has not been fit yet. Please fit the"
                       " object to get tuning results.]")
            return summary.__str__() + "\n\n" + message

        return self.summary_.__repr__()

    def __str__(self):
        """
        Return a user-friendly string representation of the BaseOptimizer object.
    
        This method provides a detailed summary of the optimization process, 
        including the best parameters and scores for each estimator if the 
        optimization has been completed. If the optimization has not been 
        performed, it provides a summary of the setup.
    
        Returns
        -------
        str
            A detailed string representation of the BaseOptimizer object.
        """
        if self.summary_ is None:
            return f"<{self.__class__.__name__}: Object with no summary yet.>"

        return self.summary_.__str__()

def _optimize_search2(
    X: ArrayLike,
    y: ArrayLike,
    param_grids: List[Union[Dict[str, List[Any]], Tuple[BaseEstimator, Dict[str, List[Any]]]]],
    estimators: Optional[List[BaseEstimator]] = None,
    strategy: str = 'GridSearchCV',
    n_jobs: int = -1,
    **search_params: Any
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization across multiple estimators using a 
    specifiedstrategy.

    Parameters
    ----------
    X : np.ndarray
        Training vectors of shape (n_samples, n_features).
    y : np.ndarray
        Target values (labels) of shape (n_samples,).
    param_grids : List[Union[Dict, Tuple[BaseEstimator, Dict]]]
        Parameter grids to explore for each estimator. Can include tuples of
        estimators and their respective parameter grids.
    estimators : Optional[List[BaseEstimator]], default=None
        List of estimator objects, required if param_grids does not include them.
   strategy : str, default='GridSearchCV'
        Optimization technique to apply ('GridSearchCV', 'RandomizedSearchCV', 'BayesSearchCV').
    n_jobs : int, default=-1
        Number of jobs to run in parallel. `-1` means using all processors.
    search_params : Any
        Additional parameters to pass to the strategy.

    Returns
    -------
    Dict[str, Any]
        Optimization results for each estimator, with estimator names as keys.

    Raises
    ------
    ValueError
        If invalidstrategy is specified.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
    >>> param_grids = [{'n_estimators': [100, 200], 'max_depth': [10, 20]}]
    >>> results = optimize_search(X, y, param_grids, [RandomForestClassifier()],
                                 strategy='RandomizedSearchCV')
    """
    estimators, param_grids = _process_estimators_and_params(param_grids, estimators)
    OptimizeMethod= get_strategy_method(strategy )
    def calculate_grid_length(param_grid):
        # n_combinations = len(list(itertools.product(*param_grid.values())))
        return np.prod([len(v) for v in param_grid.values()])
        # return n_combinations 
    def run_search(estimator, param_grid, X, y, name):
        total_combinations = calculate_grid_length(param_grid)
        with tqdm(total=total_combinations, desc=f"{name:<30}", unit= "it", #"cfg",
                  leave=True, ncols =100,
                  ) as pbar:
            search = OptimizeMethod(estimator, param_grid, n_jobs=1, **search_params)
            search.fit(X, y)
            pbar.update(total_combinations)
            return search

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_search)(est, grid, X, y, est.__class__.__name__)
        for est, grid in zip(estimators, param_grids)
    )
    results_dict = {est.__class__.__name__: {'best_estimator_': result.best_estimator_, 
                          'best_params_': result.best_params_,
                          'best_score_': result.best_score_, 
                          'cv_results_': result.cv_results_, 
                          }
                  for est, result in zip(estimators, results)}
    
    return results_dict


def _validate_parameters(param_grids, estimators):
    """
    Align estimators with their corresponding parameter grids.

    This function ensures that the estimators and their parameter grids are 
    correctly aligned and compatible for optimization.

    Parameters
    ----------
    param_grids : dict
        Dictionary of parameter grids for each estimator.
    
    estimators : dict
        Dictionary of estimator names to estimator instances.

    Returns
    -------
    tuple
        A tuple containing aligned lists of estimators and parameter grids.
    """
    return align_estimators_with_params(param_grids, estimators)


def _initialize_search(strategy, estimator, param_grid, scoring, cv, **search_kwargs):
    """
    Initialize thestrategy search method.

    This function initializes the specifiedstrategy with the given estimator 
    and parameter grid, filtering valid keyword arguments for thestrategy class.

    Parameters
    ----------
    strategy : str
        The optimization technique to apply (e.g., 'GridSearchCV').
    
    estimator : estimator instance
        The estimator instance to be optimized.
    
    param_grid : dict
        Dictionary of parameter grids for the estimator.
    
    scoring : str or callable
        Scoring method to evaluate the predictions on the test set.
    
    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.

    search_kwargs : dict
        Additional keyword arguments for thestrategy.

    Returns
    -------
    search_instance :strategy class instance
        An instance of thestrategy class initialized with the provided parameters.
    """
    strategy_class = get_strategy_method(strategy)
    search_kwargs = filter_valid_kwargs(strategy_class, search_kwargs)
    return strategy_class(
        estimator, param_grid, scoring=scoring, cv=cv, **search_kwargs)

def _perform_search(
        name, estimator, param_grid,strategy, X, y, scoring, cv, 
        search_kwargs, progress_bar_desc):
    """
    Perform the hyperparameter search.

    This function performs the hyperparameter optimization for a given estimator 
    and parameter grid using the specifiedstrategy.

    Parameters
    ----------
    name : str
        Name of the estimator.
    
    estimator : estimator instance
        The estimator instance to be optimized.
    
    param_grid : dict
        Dictionary of parameter grids for the estimator.
    
    strategy : str
        The optimization technique to apply (e.g., 'GridSearchCV').
    
    X : array-like of shape (n_samples, n_features)
        Training data.
    
    y : array-like of shape (n_samples,)
        Target values.
    
    scoring : str or callable
        Scoring method to evaluate the predictions on the test set.
    
    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.
    
    search_kwargs : dict
        Additional keyword arguments for thestrategy.
    
    progress_bar_desc : str
        Description for the progress bar.

    Returns
    -------
    tuple
        A tuple containing the name of the estimator, best estimator, best parameters, 
        best score, and cross-validation results.
    """
    search = _initialize_search(strategy, estimator, param_grid, scoring, cv, 
                                **search_kwargs)
    search.fit(X, y)
    n_combinations = len(list(ParameterGrid(param_grid)))
    pbar = tqdm(total=n_combinations, desc=progress_bar_desc, ncols=103,
                ascii=True, position=0, leave=True)
    for _ in range(n_combinations):
        pbar.update(1)
    pbar.close()

    return (
        name,
        search.best_estimator_,
        search.best_params_,
        search.best_score_,
        search.cv_results_
    )

def _process_estimators_and_params(
    param_grids: List[Union[Dict[str, List[Any]], Tuple[BaseEstimator, Dict[str, List[Any]]]]],
    estimators: Optional[List[BaseEstimator]] = None
) -> Tuple[List[BaseEstimator], List[Dict[str, List[Any]]]]:
    """
    Process and separate estimators and their corresponding parameter grids.

    This function handles two cases:
    1. `param_grids` contains tuples of estimators and their parameter grids.
    2. `param_grids` only contains parameter grids, and `estimators` are 
    provided separately.

    Parameters
    ----------
    param_grids : List[Union[Dict[str, List[Any]],
                             Tuple[BaseEstimator, Dict[str, List[Any]]]]]
        A list containing either parameter grids or tuples of estimators and 
        their parameter grids.

    estimators : List[BaseEstimator], optional
        A list of estimator objects. Required if `param_grids` only contains 
        parameter grids.

    Returns
    -------
    Tuple[List[BaseEstimator], List[Dict[str, List[Any]]]]
        Two lists: the first containing the estimators, and the second containing
        the corresponding parameter grids.

    Raises
    ------
    ValueError
        If `param_grids` does not contain estimators and `estimators` is None.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> param_grids = [
    ...     (SVC(), {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}),
    ...     (RandomForestClassifier(), {'n_estimators': [10, 50, 100],
                                        'max_depth': [5, 10, None]})
    ... ]
    >>> estimators, grids = process_estimators_and_params(param_grids)
    >>> print(estimators)
    [SVC(), RandomForestClassifier()]
    >>> print(grids)
    [{'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}, {'n_estimators': [10, 50, 100],
                                                        'max_depth': [5, 10, None]}]
    """
    if all(isinstance(grid, (tuple, list)) for grid in param_grids):
        # Extract estimators and parameter grids from tuples
        estimators, param_grids = zip(*param_grids)
        return list(estimators), list(param_grids)
    elif estimators is not None:
        # Use provided estimators and param_grids
        return estimators, param_grids
    else:
        raise ValueError("Estimators are missing. They must be provided either "
                         "in param_grids or as a separate list.")

def _get_strategy_method(strategy: str) -> Any:
    """
    Returns the correctstrategy class based on the inputstrategy string,
    ignoring case sensitivity.

    Parameters
    ----------
    strategy : str
        The name or abbreviation of thestrategy.

    Returns
    -------
    Any
        Thestrategy class corresponding to the providedstrategy string.

    Raises
    ------
    ValueError
        If no matchingstrategy is found.

    Examples
    --------
    >>>strategy_class = get_strategy_method('RSCV')
    >>> print(strategy_class)
    <class 'sklearn.model_selection.RandomizedSearchCV'>
    """

    # Mapping ofstrategy names to their possible abbreviations and variations
    opt_dict = { 
        'RandomizedSearchCV': ['RSCV', 'RandomizedSearchCV'], 
        'GridSearchCV': ['GSCV', 'GridSearchCV'], 
        'BayesSearchCV': ['BSCV', 'BayesSearchCV'], 
        'AnnealingSearchCV': ['ASCV',"AnnealingSearchCV" ], 
        'SwarmSearchCV': ['SWCV', 'PSOSCV', 'SwarmSearchCV'], 
        'SequentialSearchCV': ['SQCV', 'SMBOSearchCV'], 
        'EvolutionarySearchCV': ['EVSCV', 'EvolutionarySearchCV'], 
        'GradientSearchCV':['GBSCV', 'GradientBasedSearchCV'], 
        'GeneticSearchCV': ['GENSCV', 'GeneticSearchCV']
    }

    # Mapping ofstrategy names to their respective classes
    strategy_dict = {
        'GridSearchCV': GridSearchCV,
        'RandomizedSearchCV': RandomizedSearchCV,
    }
    try: from skopt import BayesSearchCV
    except: pass 
    else :strategy_dict["BayesSearchCV"]= BayesSearchCV
    # Normalize the inputstrategy string to ignore case
    strategy_lower =strategy.lower()

    # Search for the correspondingstrategy class
    for key, aliases in opt_dict.items():
        if strategy_lower in [alias.lower() for alias in aliases]:
            return strategy_dict[key]

    # Raise an error if no matchingstrategy is found
    raise ValueError(f"Invalid 'strategy' parameter '{strategy}'."
                     f" Choose from {smart_format(opt_dict.keys(), 'or')}.") 