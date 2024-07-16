# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides a comprehensive set of utilities for evaluating, visualizing, and 
analyzing machine learning models, including tools for cross-validation, 
performance metrics, and results presentation."""

from __future__ import annotations 
import re
import inspect
import warnings
import itertools 
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.covariance import ShrunkCovariance
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression  
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline 
from sklearn.utils.multiclass import type_of_target  

from ..api.types import Tuple,_F, ArrayLike, NDArray, Dict, Union, Any
from ..api.types import  List, Optional, Type, DataFrame, Series 
from ..api.summary import ModelSummary 
from ..tools.coreutils import smart_format
from ..tools.validator import get_estimator_name, check_X_y 
from ..tools._dependency import import_optional_dependency 
from .._gofastlog import gofastlog

_logger = gofastlog().get_gofast_logger(__name__)

__all__= [
    'find_best_C', 
    'get_cv_mean_std_scores',  
    'get_split_best_scores', 
    'display_model_max_details',
    'display_fine_tuned_results', 
    'display_cv_tables', 
    'get_scorers', 
    'dummy_evaluation', 
    "calculate_aggregate_scores", 
    "analyze_score_distribution", 
    "estimate_confidence_interval", 
    "rank_cv_scores", 
    "filter_scores", 
    "visualize_score_distribution", 
    "performance_over_time", 
    "calculate_custom_metric", 
    "handle_missing_in_scores", 
    "export_cv_results", 
    "comparative_analysis", 
    "base_evaluation", 
    "plot_parameter_importance", 
    "plot_hyperparameter_heatmap", 
    "visualize_learning_curve", 
    "plot_validation_curve", 
    "plot_feature_importance",
    "plot_roc_curve_per_fold", 
    "plot_confidence_intervals", 
    "plot_pairwise_model_comparison",
    "plot_feature_correlation", 
    "get_best_kPCA_params", 
    "shrink_covariance_cv_score",
  ]

class NoneHandler:
    """
    A utility class to handle `None` values in hyperparameters for various
    scikit-learn estimators. This class provides a mechanism to assign
    appropriate default values to hyperparameters that are set to `None`,
    based on the type of estimator being used.

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of a scikit-learn estimator. The type of the estimator
        is used to determine the appropriate default values for its
        hyperparameters.

    Attributes
    ----------
    estimator : BaseEstimator
        The scikit-learn estimator instance provided during initialization.
    estimator_type : str
        The name of the estimator class.
    none_handlers : dict
        A dictionary where keys are compiled regular expressions matching
        estimator types, and values are lambda functions that provide
        default values for `None` hyperparameters.

    Methods
    -------
    handle_none(param, value)
        Handle `None` values for the given hyperparameter based on the
        estimator type.
    default_handler(param)
        Provide a default value for unknown estimators based on common
        hyperparameters.

    Examples
    --------
    >>> from gofast.models.utils import NoneHandler
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> handler = NoneHandler(DecisionTreeClassifier())
    >>> handler.handle_none('max_depth', None)
    None

    Notes
    -----
    This class uses regular expressions to match estimator types and
    assigns default values to hyperparameters that are set to `None`.
    For example, `max_depth` in `DecisionTreeClassifier` is assigned
    `float('inf')` if it is `None`.

    The assignment of default values is based on the type of estimator
    and the specific hyperparameter. For decision trees and related
    estimators, the `max_depth` parameter is set to `float('inf')` when
    `None` is encountered:
    
    .. math::
        \text{max\_depth} = \infty \text{ if } \text{max\_depth} = \text{None}

    For support vector machines (SVMs), the `gamma` parameter is set to
    `'scale'` if it is `None`:
    
    .. math::
        \text{gamma} = \text{'scale'} \text{ if } \text{gamma} = \text{None}

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier : Decision tree classifier.
    sklearn.svm.SVC : Support vector classification.
    sklearn.ensemble.RandomForestClassifier : Random forest classifier.
    sklearn.ensemble.GradientBoostingClassifier : Gradient boosting
        classifier.
    
    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
       Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R.,
       Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher,
       M., Perrot, M., Duchesnay, E. (2011). Scikit-learn: Machine Learning
       in Python. Journal of Machine Learning Research, 12, 2825-2830.
    """
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimator_type = type(estimator).__name__
        self.none_handlers = self._initialize_handlers()

    def _initialize_handlers(self):
        handlers = {
            re.compile(r'DecisionTree.*'): lambda param: ( # float('int'))
                None if param == 'max_depth' else None
            ),
            re.compile(r'SV.*'): lambda param: (
                'scale' if param == 'gamma' else None
            ),
            re.compile(r'RandomForest.*|ExtraTrees.*'): lambda param: (
                None if param == 'max_depth' else None
            ),
            re.compile(r'GradientBoosting.*|HistGradientBoosting.*'): lambda param: (
                None if param == 'max_depth' else None
            ),
            # We can add more estimators and their None handlers as needed
        }
        return handlers

    def handle_none(self, param, value):
        """
        Handle `None` values for the given hyperparameter based on the
        estimator type.

        Parameters
        ----------
        param : str
            The name of the hyperparameter.
        value : any
            The value of the hyperparameter.

        Returns
        -------
        any
            The default value for the hyperparameter if it is `None`,
            otherwise the original value.
        """
        if value is not None:
            return value

        for pattern, handler in self.none_handlers.items():
            if pattern.match(self.estimator_type):
                return handler(param)
        
        return self.default_handler(param)

    def default_handler(self, param):
        """
        Provide a default value for unknown estimators based on common
        hyperparameters.

        Parameters
        ----------
        param : str
            The name of the hyperparameter.

        Returns
        -------
        any
            The default value for the hyperparameter.
        """
        default_values = {
            'max_depth': None,  # Use None for max_depth instead of inf
            'gamma': 'scale',  # Default value for gamma in SVMs
            'n_estimators': 100,  # Common default for ensemble methods
            'learning_rate': 0.1,  # Common default for boosting methods
            # Add more common hyperparameters as needed
        }
        return default_values.get(param, None)
    
    
def align_estimators_with_params(param_grids, estimators=None):
    """
    Reorganize estimators and their corresponding parameter grids.

    This function ensures that the estimators and parameter grids are properly 
    aligned,particularly when explicit names are given to estimators. It 
    supports different formats of estimator and parameter grid inputs, such 
    as lists, dictionaries, or tuples.

    Parameters
    ----------
    param_grids : dict, list of dict, or list of tuple
        Parameter grids to be used for each estimator. If it's a single dictionary,
        it's converted into a list. If it's a list of tuples, each tuple should
        contain an estimator name and its corresponding parameter grid.
        
    estimators : list, dict, tuple, or estimator, default=None
        Estimators to be used. It can be a single estimator, a list of estimators,
        a dictionary with estimator names as keys, or a list of tuples where each
        tuple contains an estimator name and the estimator itself.

    Returns
    -------
    tuple of (list, list)
        A tuple containing two lists: the first list contains the estimators, and
        the second list contains the corresponding parameter grids.

    Raises
    ------
    ValueError
        If the length of estimators and parameter grids does not match or if
        there is a mismatch between named estimators and parameter grids.

    Notes
    -----
    This function is particularly useful in scenarios where estimators are named
    and need to be matched with corresponding named parameter grids, ensuring
    consistency in hyperparameter tuning processes.
    
    Examples 
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from gofast.models.utils import align_estimators_with_params 

    >>> estimators1 = [{"rf": RandomForestClassifier()}, {"svc": SVC()}]
    >>> param_grids1 = [("rf", {'n_estimators': [100, 200], 'max_depth': [10, 20]}), 
                    ("svc", {"C": [1, 10], "gamma": [.001, .01, .00001]})]
    >>> new_estimators1, new_param_grids1 = align_estimators_with_params(
        param_grids1, estimators1)
    >>> print(new_estimators1)
    >>> print(new_param_grids1)
    [RandomForestClassifier(), SVC()]
    [{'n_estimators': [100, 200], 'max_depth': [10, 20]}, 
     {'C': [1, 10], 'gamma': [0.001, 0.01, 1e-05]}]
    
    >>> estimators2 = [RandomForestClassifier(), SVC()]
    >>> param_grids2 = [{'n_estimators': [100, 200], 'max_depth': [10, 20]}, 
                    {"C": [1, 10], "gamma": [.001, .01, .00001]}]
    >>> new_estimators2, new_param_grids2 = align_estimators_with_params(
        param_grids2, estimators2)
    >>> print(new_estimators2)
    >>> print(new_param_grids2)
    [RandomForestClassifier(), SVC()]
    [{'n_estimators': [100, 200], 'max_depth': [10, 20]},
     {'C': [1, 10], 'gamma': [0.001, 0.01, 1e-05]}]
    
    >>> estimators3 = [{"rf": RandomForestClassifier()}, {"svc": SVC()}]
    >>> param_grids3 = [("svc", {"C": [1, 10], "gamma": [.001, .01, .00001]}), 
                    ("rf", {'n_estimators': [100, 200], 'max_depth': [10, 20]})]
    >>> new_estimators3, new_param_grids3 = align_estimators_with_params(
        param_grids3, estimators3)
    >>> print(new_estimators3)
    >>> print(new_param_grids3)
    [SVC(), RandomForestClassifier()]
    [{'C': [1, 10], 'gamma': [0.001, 0.01, 1e-05]},
     {'n_estimators': [100, 200], 'max_depth': [10, 20]}]

    >>> estimators4 = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
    >>> param_grids4 = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
                        'SGDClassifier': {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}}
    >>> new_estimators4, new_param_grids4 = align_estimators_with_params(
         param_grids4, estimators4)
    >>> print(new_estimators4)
    >>> print(new_param_grids4)
    
    """
    if estimators is None:
        return process_estimators_and_params(param_grids)
    
    if estimators is not None: 
        estimators, param_grids= parse_estimators_and_params(
            estimators, param_grids, control="passthrough")

    param_grids = [param_grids] if isinstance(param_grids, dict) else param_grids

    if len(estimators) != len(param_grids):
        raise ValueError("Estimators and param_grid must have consistent length."
                         f" Got {len(estimators)} and {len(param_grids)} respectively.")

    estimators, estimator_names = _unpack_estimators(estimators)
    param_grids, param_grid_names = _unpack_param_grids(param_grids)

    if estimator_names and param_grid_names:
        estimators = _match_estimators_to_grids(
            estimators, estimator_names, param_grid_names)

    return estimators, param_grids

def parse_estimators_and_params(
        estimators, param_grids, control='passthrough'):
    """
    Parse and validate estimators and parameter grids.

    This function checks the provided estimators and parameter grids for 
    consistency and validity. It ensures that the lengths of the estimators 
    and parameter grids match, that the keys in both dictionaries are the same, 
    and that each estimator implements the `fit` method.

    Parameters
    ----------
    estimators : dict or list
        Dictionary of estimator names to estimator instances, or a list of estimators.
    
    param_grids : dict or list
        Dictionary of estimator names to parameter grids, or a list of parameter grids.
    
    control : str, default='passthrough'
        Control the behavior of the function. If 'strict', the function will 
        raise errors for invalid inputs. If 'passthrough', the function will 
        return the inputs as they are if they are not dictionaries.
    
    Returns
    -------
    tuple
        A tuple containing a list of estimators and a list of parameter grids, 
        ordered by the estimator names.
    
    Raises
    ------
    ValueError
        If the lengths of the estimators and parameter grids do not match, 
        if the keys in both dictionaries are not the same, or if any estimator 
        does not implement the `fit` method when control is 'strict'.
    
    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import SGDClassifier
    >>> from gofast.models.utils import parse_estimators_and_params
    >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
    >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
    ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}}
    >>> parse_estimators_and_params(estimators, param_grids)
    ([SGDClassifier(), SVC()], 
     [{'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}, 
      {'C': [1, 10], 'kernel': ['linear', 'rbf']}])
    """
    if isinstance(estimators, dict) and isinstance(param_grids, dict):
        if len(estimators) != len(param_grids):
            raise ValueError(
                "The number of estimators and parameter grids must be the same.")
        
        estimator_keys = set(estimators.keys())
        param_grid_keys = set(param_grids.keys())

        if estimator_keys != param_grid_keys:
            raise ValueError(
                "The keys in estimators and param_grids must be the same.")

        ordered_estimators = []
        ordered_param_grids = []

        for key in estimators:
            estimator = estimators[key]
            if not hasattr(estimator, 'fit'):
                raise ValueError(
                    f"The estimator {key} does not implement a 'fit' method.")
            ordered_estimators.append(estimator)
            ordered_param_grids.append(param_grids[key])

        return ordered_estimators, ordered_param_grids

    elif control == 'strict':
        raise ValueError(
            "Both estimators and param_grids must be dictionaries in strict mode.")
    
    elif control == 'passthrough':
        if isinstance(estimators, dict):
            # Transform estimator dict to list and check if each implements fit method
            ordered_estimators = []
            for key in estimators:
                estimator = estimators[key]
                if not hasattr(estimator, 'fit'):
                    raise ValueError(
                        f"The estimator {key} does not implement a 'fit' method.")
                ordered_estimators.append(estimator)
        else:
            ordered_estimators = estimators

        if isinstance(param_grids, dict):
            # Transform param_grid dict to list of param grids
            ordered_param_grids = [param_grids[key] for key in param_grids]
        else:
            ordered_param_grids = param_grids

        # Check if lengths match
        if len(ordered_estimators) != len(ordered_param_grids):
            raise ValueError(
                "The number of estimators and parameter grids must be the same.")

        return ordered_estimators, ordered_param_grids

    else:
        raise ValueError("Unknown control value. Use 'strict' or 'passthrough'.")

def _unpack_estimators(estimators):
    """
    Unpack the estimators into a consistent format and extract names if provided.

    This function handles various formats of input estimators (single estimator,
    list of estimators, dictionary of named estimators, or list of named tuples) and
    converts them into a uniform format (list of estimators) with optional extracted names.

    Parameters
    ----------
    estimators : estimator, list, dict, or list of tuples
        The input estimators in various formats.

    Returns
    -------
    tuple of (list, list or None)
        A tuple containing a list of unpacked estimators and a list of names
        if provided, otherwise None.
    """
    if hasattr(estimators, 'fit') or isinstance(estimators, dict):
        estimators = [estimators]

    if all(hasattr(estimator, 'fit') for estimator in estimators):
        return estimators, None

    if all(isinstance(estimator, dict) for estimator in estimators):
        names = [name for estimator in estimators for name in estimator]
        values = [value for estimator in estimators for value in estimator.values()]
        return values, names

    if all(isinstance(estimator, tuple) for estimator in estimators):
        names, values = zip(*estimators)
        return list(values), list(names)

    raise ValueError("Invalid format of estimators provided.")

def _unpack_param_grids(param_grids):
    """
    Unpack the parameter grids into a consistent format and extract names if provided.

    This function handles various formats of input parameter grids (single grid,
    list of grids, or list of named tuples) and converts them into a uniform format
    (list of parameter grids) with optional extracted names.

    Parameters
    ----------
    param_grids : dict, list of dict, or list of tuples
        The input parameter grids in various formats.

    Returns
    -------
    tuple of (list, list or None)
        A tuple containing a list of unpacked parameter grids and a list of names
        if provided, otherwise None.
    """
    if isinstance(param_grids, dict):
        param_grids = [param_grids]

    if all(isinstance(param_grid, dict) for param_grid in param_grids):
        return param_grids, None

    if all(isinstance(param_grid, tuple) for param_grid in param_grids):
        names, grids = zip(*param_grids)
        return list(grids), list(names)

    raise ValueError("Invalid format of parameter grids provided.")

def _match_estimators_to_grids(estimators, estimator_names, param_grid_names):
    """
    Match the estimators to their corresponding parameter grids based on names.

    This function aligns the estimators with their respective parameter grids by names.
    It is useful when estimators and parameter grids are explicitly named and need
    to be matched accurately.

    Parameters
    ----------
    estimators : list
        The list of estimators.
    estimator_names : list
        The list of names for each estimator.
    param_grid_names : list
        The list of names for each parameter grid.

    Returns
    -------
    list
        A list of matched estimators according to the parameter grid names.

    Raises
    ------
    ValueError
        If there is a mismatch between the named estimators and parameter grids.
    """
    if not estimator_names or not param_grid_names:
        return estimators

    matched_estimators = []
    for grid_name in param_grid_names:
        if grid_name not in estimator_names:
            raise ValueError(f"Estimator name '{grid_name}' not found "
                             "among provided estimators.")
        index = estimator_names.index(grid_name)
        matched_estimators.append(estimators[index])

    if len(matched_estimators) != len(estimators):
        raise ValueError("Mismatch between estimator names and parameter grid names.")

    return matched_estimators

def params_combinations(param_space):
    """
    Generate combinations of parameters from a parameter space.

    Parameters:
    -----------
    param_space : dict
        A dictionary where keys are parameter names and values are lists
        of possible values for each parameter.

    Yields:
    -------
    dict
        A dictionary representing a combination of parameters.

    Examples:
    --------
    >>> param_space = {
    ...     'C': [1, 10, 100],
    ...     'gamma': [0.001, 0.0001],
    ...     'kernel': ['linear', 'rbf']
    ... }
    >>> combinations_generator = parameter_combinations(param_space)
    >>> for combination in combinations_generator:
    ...     print(combination)
    {'C': 1, 'gamma': 0.001, 'kernel': 'linear'}
    {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}
    {'C': 1, 'gamma': 0.0001, 'kernel': 'linear'}
    {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}
    {'C': 10, 'gamma': 0.001, 'kernel': 'linear'}
    {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
    {'C': 10, 'gamma': 0.0001, 'kernel': 'linear'}
    {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
    {'C': 100, 'gamma': 0.001, 'kernel': 'linear'}
    {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    {'C': 100, 'gamma': 0.0001, 'kernel': 'linear'}
    {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}
    """
    if not isinstance (param_space, dict): 
        raise TypeError ( "Expect a dictionnary for 'param_space';"
                         f" got {type(param_space).__name__!r}")
    keys = param_space.keys()
    values = param_space.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def get_strategy_method(strategy: str) -> Type[BaseEstimator]:
    """
    Returns the corresponding strategy class based on the provided strategy 
    string.
    
    This function accounts for standard strategies as well as custom strategys 
    defined in gofast.

    Parameters
    ----------
    strategy : str
        The name or abbreviation of the strategy.

    Returns
    -------
    Type[BaseEstimator]
        The class of the strategy corresponding to the provided strategy 
        string.

    Raises
    ------
    ImportError
        If a required external strategy class (e.g., BayesSearchCV) is not 
        installed.
    ValueError
        If no matching strategy is found or the strategy name is unrecognized.

    Examples
    --------
    >>> from gofast.models.utils import get_strategy_method
    >>> strategy_class = get_strategy_method('RSCV')
    >>> print(strategy_class)
    <class 'sklearn.model_selection.RandomizedSearchCV'>
    >>> strategy_class = get_strategy_method('GESCV')
    >>> print(strategy_class)
    <class 'gofast.models.selection.GeneticSearchCV'>
    """
    # Ensure the strategy name is standardized
    strategy = validate_strategy(strategy) 
    
    # Mapping of strategy names to their respective classes
    # Standard strategy dictionary
    standard_strategy_dict = {
        'GridSearchCV': GridSearchCV,
        'RandomizedSearchCV': RandomizedSearchCV,
    }
    try: from skopt import BayesSearchCV
    except: 
        if strategy =='BayesSearchCV': 
            emsg= ("scikit-optimize is required for 'BayesSearchCV'"
                   " but not installed.")
            import_optional_dependency('skopt', extra= emsg )
        pass 
    else : standard_strategy_dict["BayesSearchCV"]= BayesSearchCV
    
    # Update standard strategy with gofast strategies if 
    # not exist previously.
    if strategy not in standard_strategy_dict.keys(): 
        from gofast.models.selection import ( 
            SwarmSearchCV, 
            SequentialSearchCV, 
            AnnealingSearchCV, 
            EvolutionarySearchCV, 
            GradientSearchCV,
            GeneticSearchCV 
            ) 
        gofast_strategy_dict = { 
            'SwarmSearchCV': SwarmSearchCV,
            'SequentialSearchCV': SequentialSearchCV,
            'AnnealingSearchCV': AnnealingSearchCV,
            'EvolutionarySearchCV': EvolutionarySearchCV,
            'GradientSearchCV': GradientSearchCV,
            'GeneticSearchCV': GeneticSearchCV,
            }
        standard_strategy_dict ={**standard_strategy_dict,**gofast_strategy_dict }
        
    # Search for the corresponding strategy class
    return standard_strategy_dict.get(strategy)

def get_strategy_name(strategy, error='raise'):
    """
    Retrieve the name of the strategy based on an input string.
    
    This function searches for known strategy identifiers within the input
    string using regular expressions and returns the formal name of the
    strategy if a match is found. If no match is found, it handles the
    situation based on the specified error handling mode ('raise', 'ignore',
    'warn').

    Parameters
    ----------
    strategy : str
        The input string potentially containing an strategy name.
    error : str, optional
        Error handling mode: 'raise' (default) to raise an exception,
        'ignore' to return a default message, and 'warn' to issue a warning.
    
    Returns
    -------
    str
        The formal name of the strategy if found, or a default message
        based on the error handling mode.
    
    Raises
    ------
    ValueError
        If no strategy is found and error mode is 'raise'.
    
    Notes
    -----
    The function uses regular expressions for flexible and efficient matching.
    The pattern matching checks for exact words and common abbreviations or
    acronyms related to strategy names.

    Examples
    --------
    >>> from gofast.models.utils import get_strategy_name
    >>> get_strategy_name("I used random search CV for optimization")
    'RandomizedSearchCV'

    >>> get_strategy_name("What is GSCV?", error='warn')
    UserWarning: Optimizer not found. Valid options include: RandomizedSearchCV, 
    GridSearchCV, etc.

    >>> get_strategy_name("optimize with GASCV")
    'GeneticSearchCV'
    
    """
    strategy = _standardize_input(strategy )
    opt_dict = {
       'RandomizedSearchCV': r"\b(random|RSCV|RandomizedSearchCV)\b",
       'GridSearchCV': r"\b(grid|GSCV|GridSearchCV)\b",
       'BayesSearchCV': r"\b(bayes|BSCV|BayesSearchCV)\b",
       'AnnealingSearchCV': r"\b(annealing|ASCV|AnnealingSearchCV)\b",
       'SwarmSearchCV': r"\b(swarm|pso|SWCV|PSOSCV|SwarmSearchCV)\b",
       'SequentialSearchCV': r"\b(sequential|SSCV|SMBOSearchCV)\b",
       'EvolutionarySearchCV': r"\b(evolution(?:ary)?|ESCV|EvolutionarySearchCV)\b",
       'GradientSearchCV': r"\b(gradient|GBSCV|GradientBasedSearchCV)\b",
       'GeneticSearchCV': r"\b(genetic|GASCV|GeneticSearchCV)\b"
   }
    
    strategy_input = str(strategy).lower()
    for key, pattern in opt_dict.items():
        if re.search(pattern.lower(), strategy_input):
            return key

    valid_opts = ', '.join(opt_dict.keys())
    error_message = f"Optimizer not found. Valid options include: {valid_opts}."

    if error == 'raise':
        raise ValueError(error_message)
    elif error == 'warn':
        warnings.warn(error_message, UserWarning)
    
    return "Optimizer not found"

def _standardize_input(input_obj):
    """
    Standardizes the input to be a string. If the input is a class or an 
    instance of a class, it retrieves the class name. If it's a string, 
    it returns it as is.
    """
    if isinstance ( input_obj, str ): 
        return input_obj
    if inspect.isclass(input_obj):
        # It's a class type
        return input_obj.__name__
    
    elif hasattr(input_obj, '__class__'):
        # It's an instance of a class
        return input_obj.__class__.__name__
    else:
        # It's something else
        return str(input_obj)

def process_estimators_and_params(
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
    >>> from sklearn.models.utils import process_estimators_and_params 
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
        
def validate_strategy(strategy: Union[str, _F]) -> str:
    """
    Check whether the given strategy is a recognized strategy type.

    This function validates if the provided strategy, either as a string 
    or an instance of a class derived from BaseEstimator, corresponds to a 
    known strategy type. If the strategy is recognized, its standardized 
    name is returned. Otherwise, a ValueError is raised.

    Parameters
    ----------
    strategy : Union[str, _F]
        The strategy to validate. This can be a string name or an instance 
        of an strategy class.

    Returns
    -------
    str
        The standardized name of the strategy.

    Raises
    ------
    ValueError
        If the strategy is not recognized.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier 
    >>> from gofast.models.selection import AnnealingSearchCV
    >>> from gofast.models.utils import validate_strategy
    >>> validate_strategy("RSCV")
    'RandomizedSearchCV'
    >>> validate_strategy(AnnealingSearchCV)
    'AnnealingSearchCV'
    >>> validate_strategy (RandomForestClassifier)
    ValueError ...
    """
    # Mapping of strategy names to their possible abbreviations and variations
    opt_dict = {
        'RandomizedSearchCV': ['RSCV', 'RandomizedSearchCV'], 
        'GridSearchCV': ['GSCV', 'GridSearchCV'], 
        'BayesSearchCV': ['BSCV', 'BayesSearchCV'], 
        'AnnealingSearchCV': ['ASCV', "AnnealingSearchCV"], 
        'SwarmSearchCV': ['SWSCV', 'SwarmSearchCV'], 
        'SequentialSearchCV': ['SQSCV', 'SequentialSearchCV'], 
        'EvolutionarySearchCV': ['EVSCV', 'EvolutionarySearchCV'], 
        'GradientSearchCV': ['GBSCV', 'GradientSearchCV'], 
        'GeneticSearchCV': ['GENSCV', 'GeneticSearchCV']
    }


    strategy_name = strategy if isinstance(
        strategy, str) else get_estimator_name(strategy)

    for key, values in opt_dict.items():
        if strategy_name.lower() in [v.lower() for v in values]:
            return key

    valid_strategys = [v1[1] for v1 in opt_dict.values()]
    raise ValueError(f"Invalid 'strategy' parameter '{strategy_name}'."
                     f" Choose from {smart_format(valid_strategys, 'or')}.")
    
def find_best_C(X, y, C_range, cv=5, scoring='accuracy', 
                scoring_reg='neg_mean_squared_error'):
    """
    Find the best C regularization parameter for an SVM, automatically determining
    whether the task is classification or regression based on the target variable.

     Mathematically, the formula can be expressed as: 

     .. math::
         \\text{Regularization Path: } C_i \\in \\{C_1, C_2, ..., C_n\\}
         \\text{For each } C_i:\\
             \\text{Evaluate } \\frac{1}{k} \\sum_{i=1}^{k} \\text{scoring}(\\text{SVM}(C_i, \\text{fold}_i))
         \\text{Select } C = \\arg\\max_{C_i} \\text{mean cross-validated score}
         
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and 
        n_features is the number of features.
    y : array-like, shape (n_samples,)
        Target values, used to determine if the task is classification or 
        regression.
    C_range : array-like
        The range of C values to explore.
    cv : int, default=5
        Number of folds in cross-validation.
    scoring : str, default='accuracy'
        A string to determine the cross-validation scoring metric 
        for classification.
    scoring_reg : str, default='neg_mean_squared_error'
        A string to determine the cross-validation scoring metric 
        for regression.

    Returns
    -------
    best_C : float
        The best C parameter found in C_range.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> C_range = np.logspace(-4, 4, 20)
    >>> best_C = find_best_C(X, y, C_range)
    >>> print(f"Best C value: {best_C}")
    """
    X, y = check_X_y(X,  y, to_frame= True, )
    task_type = type_of_target(y)
    best_score = ( 0 if task_type == 'binary' or task_type == 'multiclass'
                  else float('inf') )
    best_C = None

    for C in C_range:
        if task_type == 'binary' or task_type == 'multiclass':
            model = SVC(C=C)
            score_function = scoring
        else:  # regression
            model = SVR(C=C)
            score_function = scoring_reg

        scores = cross_val_score(model, X, y, cv=cv, scoring=score_function)
        mean_score = np.mean(scores)
        if (task_type == 'binary' or task_type == 'multiclass' and mean_score > best_score) or \
           (task_type != 'binary' and task_type != 'multiclass' and mean_score < best_score):
            best_score = mean_score
            best_C = C

    return best_C

def get_cv_mean_std_scores(
    cvres: Dict[str, ArrayLike],
    score_type: str = 'test_score',
    aggregation_method: str = 'mean',
    ignore_convergence_problem: bool = False
) -> Tuple[float, float]:
    """
    Retrieve the global aggregated score and its standard deviation from 
    cross-validation results.

    This function computes the overall aggregated score and its standard 
    deviation from the results of cross-validation for a specified score type. 
    It also provides options to handle situations where convergence issues 
    might have occurred during model training.

    Parameters
    ----------
    cvres : Dict[str, ArrayLike]
        A dictionary containing the cross-validation results. Expected to have 
        keys including 'mean_test_score', 'std_test_score', and potentially 
        others depending on the metrics used during cross-validation.
    score_type : str, default='test_score'
        The type of score to aggregate. Typical values include 'test_score' 
        and 'train_score'. The function expects corresponding 'mean' and 'std' 
        keys in `cvres` (e.g., 'mean_test_score' for 'test_score').
    aggregation_method : str, default='mean'
        Method to aggregate scores across cross-validation folds. 
        Options include 'mean' and 'median'.
    ignore_convergence_problem : bool, default=False
        If True, NaN values that might have resulted from convergence 
        issues during model training are ignored in the aggregation process. 
        If False, NaN values contribute to the final aggregation as NaN.

    Returns
    -------
    Tuple[float, float]
        A tuple containing two float values:
        - The first element is the aggregated score across all 
          cross-validation folds.
        - The second element is the mean of the standard deviations of the 
          scores across all cross-validation folds.

    Raises
    ------
    ValueError
        If the specified score type does not exist in `cvres`.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> from gofast.models import get_cv_mean_std_scores
    >>> iris = load_iris()
    >>> clf = DecisionTreeClassifier()
    >>> scores = cross_val_score(clf, iris.data, iris.target, cv=5,
    ...                          scoring='accuracy', return_train_score=True)
    >>> cvres = {'mean_test_score': scores, 'std_test_score': np.std(scores)}
    >>> mean_score, mean_std = get_cv_mean_std_scores(cvres, score_type='test_score')
    >>> print(f"Mean score: {mean_score}, Mean standard deviation: {mean_std}")

    """
    mean_key = f'mean_{score_type}'
    std_key = f'std_{score_type}'

    if mean_key not in cvres or std_key not in cvres:
        raise ValueError(f"Score type '{score_type}' not found in cvres.")

    if ignore_convergence_problem:
        mean_aggregate = ( np.nanmean(cvres[mean_key]) if aggregation_method == 'mean' 
                          else np.nanmedian(cvres[mean_key]))
        std_aggregate = np.nanmean(cvres[std_key])
    else:
        mean_aggregate = ( cvres[mean_key].mean() if aggregation_method == 'mean' 
                          else np.median(cvres[mean_key])
                          )
        std_aggregate = cvres[std_key].mean()

    return mean_aggregate, std_aggregate

def get_split_best_scores(cvres:Dict[str, ArrayLike], 
                       split:int=0)->Dict[str, float]: 
    """Get the best score at each split from cross-validation results.
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    split: int, default=1 
        The number of split to fetch parameters. 
        The number of split must be  the number of cross-validation (cv) 
        minus one.
        
    Returns
    -------
    bests: Dict, 
        Dictionnary of the best parameters at the corresponding `split` 
        in the cross-validation. 
        
    """
    # get the split score 
    split_score = cvres[f'split{split}_test_score'] 
    # take the max score of the split 
    max_sc = split_score.max() 
    ix_max = split_score.argmax()
    mean_score= split_score.mean()
    # get parm and mean score 
    bests ={'param': cvres['params'][ix_max], 
        'accuracy_score':cvres['mean_test_score'][ix_max], 
        'std_score':cvres['std_test_score'][ix_max],
        f"CV{split}_score": max_sc , 
        f"CV{split}_mean_score": mean_score,
        }
    return bests 

def display_model_max_details(cvres:Dict[str, ArrayLike], cv:int =4):
    """ Display the max details of each stored model from cross-validation.
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    cv: int, default=1 
        The number of KFlod during the fine-tuning models parameters. 

    """
    texts= {}
    for k in range (cv):
        b= get_split_best_scores(cvres, split =k)
        texts ["split = {k}"]= b 

    globalmeansc , globalstdsc= get_cv_mean_std_scores(cvres)
    texts["Global split ~ mean scores"]= globalmeansc
    texts["Global split ~ std. scores"]= globalstdsc
    
    summary = ModelSummary().add_multi_contents(
        *[texts] , titles = ["Model Max Details"], max_width= 90 )
    print(summary )

    
def display_fine_tuned_results ( cvmodels: list[_F] ): 
    """Display fined -tuning results.
    
    Parameters 
    -----------
    cvmnodels: list
        list of fined-tuned models.
    """
    keys = [] 
    values = [] 
    bsi_bestestimators = [model.best_estimator_ for model in cvmodels ]
    mnames = ( get_estimator_name(n) for n in bsi_bestestimators)
    bsi_bestparams = [model.best_params_ for model in cvmodels]

    for nam, param , estimator in zip(mnames, bsi_bestparams, 
                                      bsi_bestestimators): 
        keys.append (nam )
        values.append ( { 'Best Parameters': param, 
                         ' Best Estimator': estimator
                         }) 
    dict_contents = dict ( zip ( keys, values ))
    
    summary = ModelSummary().add_multi_contents(*dict_contents, )
    print(summary )
        
def display_cv_tables(cvres:Dict[str, ArrayLike],  cvmodels:list[_F] ): 
    """ Display the cross-validation results from all models at each 
    k-fold. 
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    cvmnodels: list
        list of fined-tuned models.
        
    Examples 
    ---------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.models import GridSearchMultiple, display_cv_tables
    >>> X, y  = fetch_data ('bagoue prepared') 
    >>> gobj =GridSearchMultiple(estimators = estimators, 
                                 grid_params = grid_params ,
                                 cv =4, scoring ='accuracy', 
                                 verbose =1,  savejob=False , 
                                 kind='GridSearchCV')
    >>> gobj.fit(X, y) 
    >>> display_cv_tables (cvmodels=[gobj.models.SVC] ,
                         cvres= [gobj.models.SVC.cv_results_ ])
    ... 
    """
    modelnames = (get_estimator_name(model.best_estimator_ ) 
                  for model in cvmodels  )
    for name,  mdetail, model in zip(modelnames, cvres, cvmodels): 
        print(name, ':')
        display_model_max_details(cvres=mdetail)
        
        print('BestParams: ', model.best_params_)
        try:
            print("Best scores:", model.best_score_)
        except: pass 
        finally: print()
        
def calculate_aggregate_scores(cv_scores):
    """
    Calculate various aggregate measures from cross-validation scores.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.

    Returns
    -------
    aggregates : dict
        Dictionary containing various aggregate measures of the scores.
    """
    aggregates = {
        'mean': np.mean(cv_scores),
        'median': np.median(cv_scores),
        'std': np.std(cv_scores),
        'min': np.min(cv_scores),
        'max': np.max(cv_scores)
    }
    return aggregates

def analyze_score_distribution(cv_scores):
    """
    Analyze the distribution of cross-validation scores.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.

    Returns
    -------
    distribution_analysis : dict
        Dictionary containing analysis of the score distribution.
    """
    distribution_analysis = {
        'skewness': scipy.stats.skew(cv_scores),
        'kurtosis': scipy.stats.kurtosis(cv_scores)
    }
    return distribution_analysis

def estimate_confidence_interval(cv_scores, confidence_level=0.95):
    """
    Estimate the confidence interval for cross-validation scores.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.
    confidence_level : float, optional
        The confidence level for the interval.

    Returns
    -------
    confidence_interval : tuple
        Tuple containing lower and upper bounds of the confidence interval.
    """
    mean_score = np.mean(cv_scores)
    std_error = scipy.stats.sem(cv_scores)
    margin = std_error * scipy.stats.t.ppf((1 + confidence_level) / 2., len(cv_scores) - 1)
    return (mean_score - margin, mean_score + margin)

def rank_cv_scores(cv_scores):
    """
    Rank cross-validation scores.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.

    Returns
    -------
    ranked_scores : ndarray
        Array of scores ranked in descending order.
    """
    ranked_scores = np.argsort(cv_scores)[::-1]
    return cv_scores[ranked_scores]

def filter_scores(cv_scores, threshold):
    """
    Filter cross-validation scores based on a threshold.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.
    threshold : float
        Threshold value for filtering scores.

    Returns
    -------
    filtered_scores : ndarray
        Array of scores that are above the threshold.
    """
    return cv_scores[cv_scores > threshold]

def visualize_score_distribution(
    scores, 
    ax=None,
    plot_type='histogram', 
    bins=30, 
    density=True, 
    title='Score Distribution', 
    xlabel='Score', 
    ylabel='Frequency', 
    color='skyblue',
    edge_color='black'
    ):
    """
    Visualize the distribution of scores.

    Parameters
    ----------
    scores : array-like
        Array of score values to be visualized.
    ax : matplotlib.axes.Axes, optional
        Predefined Matplotlib axes for the plot. If None, a new figure and 
        axes will be created.
    plot_type : str, optional
        Type of plot to display ('histogram' or 'density').
    bins : int, optional
        The number of bins for the histogram. Ignored if plot_type is 'density'.
    density : bool, optional
        Normalize histogram to form a probability density (True) or to show 
        frequencies (False).
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    color : str, optional
        Color of the histogram bars or density line.
    edge_color : str, optional
        Color of the histogram bar edges.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if plot_type == 'histogram':
        ax.hist(scores, bins=bins, density=density, color=color, edgecolor=edge_color)
    elif plot_type == 'density':
        sns.kdeplot(scores, ax=ax, color=color, fill=True)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()
    return ax

def performance_over_time(
    cv_results,
    ax=None,
    title='Performance Over Time', 
    xlabel='Timestamp', 
    ylabel='Score', 
    line_color='blue',
    line_style='-', 
    line_width=2,
    grid=True):
    """
    Analyze and visualize performance over time from cross-validation results.

    Parameters
    ----------
    cv_results : dict
        Dictionary of cross-validation results with 'timestamps' and 'scores'.
    ax : matplotlib.axes.Axes, optional
        Predefined Matplotlib axes for the plot. If None, a new figure and 
        axes will be created.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    line_color : str, optional
        Color of the line plot.
    line_style : str, optional
        Style of the line plot.
    line_width : float, optional
        Width of the line plot.
    grid : bool, optional
        Whether to show grid lines.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    timestamps = cv_results['timestamps']
    scores = cv_results['scores']

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(timestamps, scores, color=line_color,
            linestyle=line_style, linewidth=line_width)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True)

    plt.show()
    return ax
    
def calculate_custom_metric(cv_scores, metric_function):
    """
    Calculate a custom metric from cross-validation scores.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.
    metric_function : callable
        Function to calculate the custom metric.

    Returns
    -------
    metric : float
        Calculated metric.
    """
    return metric_function(cv_scores)

def handle_missing_in_scores(cv_scores, fill_value=np.nan):
    """
    Handle missing or incomplete data in cross-validation scores.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.
    fill_value : float, optional
        Value to replace missing or incomplete data.

    Returns
    -------
    filled_scores : ndarray
        Array of scores with missing data handled.
    """
    return np.nan_to_num(cv_scores, nan=fill_value)

def export_cv_results(cv_scores, filename):
    """
    Export cross-validation scores to a file.

    Parameters
    ----------
    cv_scores : array-like
        Array of cross-validation scores.
    filename : str
        Name of the file to export the scores.

    Returns
    -------
    None
    """

    pd.DataFrame(cv_scores).to_csv(filename, index=False)

def comparative_analysis(cv_scores_dict):
    """
    Perform a comparative analysis of cross-validation scores from 
    different models.

    Parameters
    ----------
    cv_scores_dict : dict
        Dictionary with model names as keys and arrays of scores as values.

    Returns
    -------
    comparison_results : dict
        Dictionary with comparative analysis results.
    """
    analysis_results = {}
    for model, scores in cv_scores_dict.items():
        analysis_results[model] = {
            'mean_score': np.mean(scores),
            'std_dev': np.std(scores)
        }
    return analysis_results

def get_scorers (*, scorer:str=None, check_scorer:bool=False, 
                 error:str='ignore')-> Tuple[str] | bool: 
    """ Fetch the list of available metrics from scikit-learn or verify 
    whether the scorer exist in that list of metrics. 
    This is prior necessary before  the model evaluation. 
    
    :param scorer: str, 
        Must be an metrics for model evaluation. Refer to :mod:`sklearn.metrics`
    :param check_scorer:bool, default=False
        Returns bool if ``True`` whether the scorer exists in the list of 
        the metrics for the model evaluation. Note that `scorer`can not be 
        ``None`` if `check_scorer` is set to ``True``.
    :param error: str, ['raise', 'ignore']
        raise a `ValueError` if `scorer` not found in the list of metrics 
        and `check_scorer `is ``True``. 
        
    :returns: 
        scorers: bool, tuple 
            ``True`` if scorer is in the list of metrics provided that 
            ` scorer` is not ``None``, or the tuple of scikit-metrics. 
            :mod:`sklearn.metrics`
    """
    from sklearn import metrics
    try:
        scorers = tuple(metrics.SCORERS.keys()) 
    except: scorers = tuple (metrics.get_scorer_names()) 
    
    if check_scorer and scorer is None: 
        raise ValueError ("Can't check the scorer while the scorer is None."
                          " Provide the name of the scorer or get the list of"
                          " scorer by setting 'check_scorer' to 'False'")
    if scorer is not None and check_scorer: 
        scorers = scorer in scorers 
        if not scorers and error =='raise': 
            raise ValueError(
                f"Wrong scorer={scorer!r}. Supports only scorers:"
                f" {tuple(metrics.SCORERS.keys())}")
            
    return scorers 

def plot_parameter_importance(
        cv_results, param_name, metric='mean_test_score'):
    """
    Visualizes the impact of a hyperparameter on model performance.

    This function creates a plot showing how different values of a single
    hyperparameter affect the specified performance metric.

    Parameters
    ----------
    cv_results : dict
        The cross-validation results returned by a model selection process, 
        such as GridSearchCV or RandomizedSearchCV.
    param_name : str
        The name of the hyperparameter to analyze.
    metric : str, optional
        The performance metric to visualize, by default 'mean_test_score'.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from gofast.models.utils import plot_parameter_importance
    >>> grid_search = GridSearchCV(estimator, param_grid, cv=5)
    >>> grid_search.fit(X, y)
    >>> plot_parameter_importance(grid_search.cv_results_, 'param_n_estimators')

    """
    param_values = cv_results['param_' + param_name]
    scores = cv_results[metric]

    plt.figure()
    plt.plot(param_values, scores, marker='o')
    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title('Parameter Importance')
    plt.show()

def plot_hyperparameter_heatmap(
        cv_results, param1, param2, metric='mean_test_score'):
    """
    Creates a heatmap for visualizing performance scores for combinations
    of two hyperparameters.

    This utility is useful for models with two key hyperparameters, 
    to show how different combinations affect the model's performance.

    Parameters
    ----------
    cv_results : dict
        The cross-validation results from GridSearchCV or RandomizedSearchCV.
    param1 : str
        The name of the first hyperparameter.
    param2 : str
        The name of the second hyperparameter.
    metric : str, optional
        The performance metric to plot, by default 'mean_test_score'.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from gofast.models.utils import plot_hyperparameter_heatmap
    >>> grid_search = GridSearchCV(estimator, param_grid, cv=5)
    >>> grid_search.fit(X, y)
    >>> plot_hyperparameter_heatmap(grid_search.cv_results_, 'param_C', 'param_gamma')

    """
    
    p1_values = cv_results['param_' + param1]
    p2_values = cv_results['param_' + param2]
    scores = cv_results[metric]

    heatmap_data = {}
    for p1, p2, score in zip(p1_values, p2_values, scores):
        heatmap_data.setdefault(p1, {})[p2] = score

    sns.heatmap(data=heatmap_data)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title('Hyperparameter Heatmap')
    plt.show()


def visualize_learning_curve(estimator, X, y, cv=None, train_sizes=None):
    """
    Generates a plot of the test and training learning curve.

    This function helps to assess how the model benefits from increasing 
    amounts of training data.

    Parameters
    ----------
    estimator : object
        A model instance implementing 'fit' and 'predict'.
    X : array-like, shape (n_samples, n_features)
        Training vector.
    y : array-like, shape (n_samples,)
        Target relative to X.
    cv : int, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.models.utils import plot_learning_curve
    >>> plot_learning_curve(RandomForestClassifier(), X, y, cv=5)

    """
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()


def plot_validation_curve(estimator, X, y, 
                          param_name, param_range, 
                          cv=None):
    """
    Generates a plot of the test and training scores for varying parameter 
    values.

    This function helps to understand how a particular hyperparameter affects
    learning performance.

    Parameters
    ----------
    estimator : object
        A model instance implementing 'fit' and 'predict'.
    X : array-like, shape (n_samples, n_features)
        Training vector.
    y : array-like, shape (n_samples,)
        Target relative to X.
    param_name : str
        Name of the hyperparameter to vary.
    param_range : array-like
        The values of the parameter that will be evaluated.
    cv : int, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from gofast.models.utils import plot_validation_curve
    >>> param_range = np.logspace(-6, -1, 5)
    >>> plot_validation_curve(SVC(), X, y, 'gamma', param_range, cv=5)

    """
    from sklearn.model_selection import validation_curve

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Visualizes the feature importances of a fitted model.

    This function is applicable for models that provide a feature_importances_ 
    attribute.

    Parameters
    ----------
    model : estimator object
        A fitted estimator that provides feature_importances_ attribute.
    feature_names : list
        List of feature names corresponding to the importances.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.models.utils import plot_feature_importance
    >>> model = RandomForestClassifier().fit(X, y)
    >>> plot_feature_importance(model, feature_names)

    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], color="r",
            align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
               rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()

def plot_roc_curve_per_fold(cv_results, fold_indices, y, metric='roc_auc'):
    """
    Plots ROC curves and calculates AUC for each fold in cross-validation.

    Parameters
    ----------
    cv_results : dict
        The cross-validation results returned by a model selection process.
    fold_indices : list of tuples
        List of tuples, where each tuple contains train and test indices for each fold.
    y : array-like, shape (n_samples,)
        True binary class labels.
    metric : str, optional
        The metric to use for calculating scores, default is 'roc_auc'.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_predict
    >>> from gofast.models.utils import plot_roc_curve_per_fold
    >>> y_scores = cross_val_predict(estimator, X, y, cv=5, method='predict_proba')
    >>> plot_roc_curve_per_fold(cv_results, fold_indices, y)

    """
    from sklearn.metrics import roc_curve, auc

    plt.figure()

    for i, (train_idx, test_idx) in enumerate(fold_indices):
        y_true = y[test_idx]
        y_scores = cv_results['y_scores'][i]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per Fold")
    plt.legend(loc="lower right")
    plt.show()

def plot_confidence_intervals(cv_results, metric='mean_test_score'):
    """
    Calculates and plots confidence intervals for cross-validation scores.

    Parameters
    ----------
    cv_results : dict
        The cross-validation results returned by a model selection process.
    metric : str, optional
        The performance metric to plot, by default 'mean_test_score'.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from gofast.models.utils import plot_confidence_intervals
    >>> scores = cross_val_score(estimator, X, y, cv=5)
    >>> plot_confidence_intervals(scores)

    """
    import scipy.stats as stats

    mean_score = np.mean(cv_results[metric])
    std_score = np.std(cv_results[metric])
    conf_interval = stats.norm.interval(
        0.95, loc=mean_score,scale=std_score / np.sqrt(len(cv_results[metric])))

    plt.figure()
    plt.errorbar(x=0, y=mean_score, yerr=[mean_score - conf_interval[0]], fmt='o')
    plt.xlim(-1, 1)
    plt.title("Confidence Interval for Score")
    plt.ylabel(metric)
    plt.show()

def plot_pairwise_model_comparison(
        cv_results_list, model_names, metric='mean_test_score'):
    """
    Compares performance between different models visually.

    Parameters
    ----------
    cv_results_list : list of dicts
        A list containing cross-validation results for each model.
    model_names : list of str
        Names of the models corresponding to the results in cv_results_list.
    metric : str, optional
        The performance metric for comparison, by default 'mean_test_score'.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from gofast.models.utils import plot_pairwise_model_comparison
    >>> results_list = [GridSearchCV(model, param_grid, cv=5).fit(X, y).cv_results_
                        for model in models]
    >>> plot_pairwise_model_comparison(results_list, ['Model1', 'Model2', 'Model3'])

    """
    scores = [np.mean(results[metric]) for results in cv_results_list]
    stds = [np.std(results[metric]) for results in cv_results_list]

    plt.figure()
    plt.bar(model_names, scores, yerr=stds, align='center', alpha=0.5,
            ecolor='black', capsize=10)
    plt.ylabel(metric)
    plt.title("Model Comparison")
    plt.show()

def plot_feature_correlation(cv_results, X, y):
    """
    Analyzes feature correlation with the target variable in different folds.

    Parameters
    ----------
    cv_results : dict
        The cross-validation results from a model selection process.
    X : array-like, shape (n_samples, n_features)
        Feature matrix used in cross-validation.
    y : array-like, shape (n_samples,)
        Target variable used in cross-validation.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from gofast.models.utils import plot_feature_correlation
    >>> plot_feature_correlation(cv_results, X, y)

    """
    correlations = []
    for train_idx, test_idx in cv_results['split_indices']:
        X_train, y_train = X[train_idx], y[train_idx]
        df = pd.DataFrame(X_train, columns=X.columns)
        df['target'] = y_train
        correlation = df.corr()['target'].drop('target')
        correlations.append(correlation)

    avg_corr = pd.DataFrame(correlations).mean()
    sns.heatmap(avg_corr.to_frame(), annot=True)
    plt.title("Feature Correlation with Target")
    plt.show()

def base_evaluation(
    model: BaseEstimator,
    X: NDArray,
    y: NDArray,
    cv: int = 5,
    scoring: str = 'accuracy', 
    display: bool = False,
    random_state: int = None,
    n_jobs: int = -1,
    return_std: bool = False,
    **kwargs: Dict[str, Any]
) -> Tuple[NDArray, float, float]:
    """
    Evaluate a machine learning model (classifier or regressor) 
    using cross-validation.

    Parameters
    ----------
    model : BaseEstimator
        The machine learning model to evaluate.
    X : NDArray
        Feature matrix with shape (n_samples, n_features).
    y : NDArray
        Target vector with shape (n_samples,).
    cv : int, optional
        Number of folds in cross-validation (default is 5).
    scoring : str, optional
        Scoring metric name. Defaults to 'accuracy' for classifiers and 
        'r2' for regressors.
    display : bool, optional
        If True, prints the model name, scores, and their mean (default is False).
    random_state : int, optional
        Random state for reproducibility (default is None).
    n_jobs : int, optional
        Number of jobs to run in parallel (-1 means using all processors, default is -1).
    return_std : bool, optional
        If True, returns the standard deviation of the cross-validation scores
        (default is False).
    **kwargs : dict
        Additional keyword arguments for `cross_val_score`.

    Returns
    -------
    scores : NDArray
        Cross-validation scores for each fold.
    mean_score : float
        Mean of the cross-validation scores.
    std_score : float, optional
        Standard deviation of the cross-validation scores, 
        returned if `return_std` is True.
        
    Examples 
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gofast.models.search import base_evaluation
    >>> X, y = gf.fetch_data('bagoue prepared', return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> scores, mean_score = base_evaluation(clf, X, y, cv=4, display=True)
    clf: DecisionTreeClassifier
    scores: [0.6279 0.7674 0.7093 0.593 ]
    scores.mean: 0.6744
    """
    # Determine if the model is a classifier or regressor
    if isinstance(model, ClassifierMixin):
        model_type = 'classifier'
    elif isinstance(model, RegressorMixin):
        model_type = 'regressor'
        scoring = 'r2' if scoring == 'accuracy' else scoring  
    else:
        raise ValueError("Model must be a classifier or regressor.")

    # Validate scoring metric
    try:
        get_scorer(scoring)
    except ValueError:
        raise ValueError(f"The scoring metric '{scoring}' is not appropriate"
                         f" for a {model_type}.")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, 
                             n_jobs=n_jobs, random_state=random_state, **kwargs)
    mean_score = scores.mean()
    std_score = np.std(scores) if return_std else None

    if display:
        model_name = model.__class__.__name__
        print(f'Model: {model_name}')
        print(f'Scores: {scores}')
        print(f'Mean score: {mean_score:.4f}')
        if return_std:
            print(f'Std deviation: {std_score:.4f}')

    return (scores, mean_score, std_score) if return_std else (scores, mean_score)

def dummy_evaluation(
    model: BaseEstimator,
    X: NDArray,
    y: NDArray,
    cv: int = 7,
    scoring: str = 'accuracy',
    display: bool = False,
    **kwargs: Dict[str, Any]
) -> Tuple[NDArray, float]:
    """
    Perform a quick evaluation of a machine learning model using cross-validation.

    Parameters
    ----------
    model : BaseEstimator
        The machine learning model to be evaluated.
    X : NDArray
        Feature matrix with shape (n_samples, n_features).
    y : NDArray
        Target vector with shape (n_samples,).
    cv : int
        Number of folds in cross-validation (default is 7).
    scoring : str
        Scoring metric to use (default is 'accuracy').
    display : bool
        If True, print the model name, scores, and their mean (default is False).
    **kwargs : dict
        Additional keyword arguments passed to `cross_val_score`.

    Returns
    -------
    scores : NDArray
        Array of scores of the estimator for each run of the cross-validation.
    mean_score : float
        Mean of the cross-validation scores.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> from gofast.models.utils import dummy_evaluation
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> scores, mean_score = dummy_evaluation(clf, X, y, cv=4, display=True)
    Model: DecisionTreeClassifier
    Scores: [0.95, 0.92, 0.95, 0.98]
    Mean score: 0.95
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, **kwargs)
    mean_score = np.mean(scores)

    if display:
        model_name = model.__class__.__name__
        print(f'Model: {model_name}')
        print(f'Scores: {scores}')
        print(f'Mean score: {mean_score:.4f}')

    return scores, mean_score

def shrink_covariance_cv_score(
    X: Union[np.ndarray, pd.DataFrame],
    shrink_space: Tuple[float, float, int] = (-2, 0, 30),
    cv: int = 5,
    scoring: str = 'neg_log_loss',
    n_jobs: int = -1,
    return_estimator: bool = False
) -> Union[float, Tuple[float, ShrunkCovariance]]:
    """
    Evaluate the performance of ShrunkCovariance estimator on the data `X`
    by tuning the 'shrinkage' parameter using GridSearchCV and cross-validation.

    Parameters
    ----------
    X : array_like or pandas.DataFrame
        Input data where rows represent samples and columns represent features.
    shrink_space : tuple of (float, float, int), default=(-2, 0, 30)
        The range and number of points for 'shrinkage' parameter space, 
        specified as (start, stop, num) for np.logspace to generate shrinkages.
    cv : int, default=5
        Number of folds in cross-validation.
    scoring : str, default='neg_log_loss'
        Scoring metric to use for the cross-validation.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.
    return_estimator : bool, default=False
        Whether to return the best estimator along with the score.

    Returns
    -------
    score : float
        Mean cross-validation score of the best estimator.
    best_estimator : ShrunkCovariance, optional
        The best ShrunkCovariance estimator from GridSearchCV, returned if
        return_estimator is True.
        
    Examples
    --------
    >>> from sklearn.datasets import make_spd_matrix
    >>> from gofast.models.utils import shrink_covariance_cv_score
    >>> # Generate a symmetric positive-definite matrix
    >>> X = make_spd_matrix(n_dim=100, random_state=42)  
    >>> score, best_estimator = shrink_covariance_cv_score(
        X, cv=3, scoring='neg_log_loss', return_estimator=True)
    >>> print(f"Best CV score: {score:.4f}")
    >>> print(f"Best shrinkage parameter: {best_estimator.shrinkage}")
    """
    shrinkages = np.logspace(*shrink_space)  # Define shrinkage values
    cv_estimator = GridSearchCV(
        ShrunkCovariance(), 
        {'shrinkage': shrinkages}, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=n_jobs
    )
    
    # Fit the GridSearchCV to find the best shrinkage parameter
    cv_estimator.fit(X)
    
    # Calculate the cross-validation score for the best estimator
    score = np.mean(cross_val_score(cv_estimator.best_estimator_, X, cv=cv,
                                    scoring=scoring, n_jobs=n_jobs))
    
    return (score, cv_estimator.best_estimator_) if return_estimator else score 

def get_best_kPCA_params(
    X: Union[NDArray, DataFrame],
    n_components: Union[float, int] = 2,
    *,
    y: Optional[Union[ArrayLike, Series]] = None,
    param_grid: Optional[Dict[str, Any]] = None,
    clf: Optional[Pipeline] = None,
    cv: int = 7,
    **grid_kws
) -> Dict[str, Any]:
    """
    Select the kernel and hyperparameters for Kernel PCA (kPCA) using 
    GridSearchCV. 
    
    GridSearchCV lead to the best performance in a subsequent supervised 
    learning task, typically classification.
    
    As kPCA( unsupervised learning algorithm), there is obvious performance
    measure to help selecting the best kernel and hyperparameters values. 
    However dimensionality reduction is often a preparation step for a 
    supervised task(e.g. classification). So we can use grid search to select
    the kernel and hyperparameters that lead the best performance on that 
    task. By default implementation we create two steps pipeline. First reducing 
    dimensionality to two dimension using kPCA, then applying the 
    `LogisticRegression` for classification. AFter use Grid searchCV to find 
    the best ``kernel`` and ``gamma`` value for kPCA in oder to get the best 
    clasification accuracy at the end of the pipeline.

    Parameters
    ----------
    X : NDArray or DataFrame
        Input data for dimensionality reduction.
    n_components : int or float, default=2
        Number of components to preserve. If a float in the range (0, 1), 
        it indicates the ratio of variance to preserve.
    y : ArrayLike or Series, optional
        Target variable for supervised learning tasks.
    param_grid : Dict[str, Any], optional
        Dictionary with parameters names (`str`) as keys and lists of parameter 
        settings to try as values.
    clf : Pipeline, optional
        A sklearn Pipeline where kPCA is followed by a classifier. Default is 
        a pipeline with KernelPCA and LogisticRegression.
    cv : int, default=7
        Number of folds in cross-validation.
    grid_kws : dict
        Additional keyword arguments passed to GridSearchCV.

    Returns
    -------
    best_params_ : Dict[str, Any]
        Dictionary containing the best parameters found on the grid.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.models.utils import get_best_kPCA_params
    >>> X, y = make_classification(n_features=20, n_redundant=0, 
                                   n_informative=2, random_state=1)
    >>> param_grid = {
    ...     'kpca__gamma': np.linspace(0.03, 0.05, 10),
    ...     'kpca__kernel': ["rbf", "sigmoid"]
    ... }
    >>> best_params = get_best_kPCA_params(X, y=y, param_grid=param_grid)
    >>> print(best_params)
    """
    from sklearn.decomposition import KernelPCA

    if param_grid is None:
        param_grid = {
            'kpca__gamma': np.linspace(0.03, 0.05, 10),
            'kpca__kernel': ["rbf", "sigmoid"]
        }

    if clf is None:
        clf = Pipeline([
            ('kpca', KernelPCA(n_components=n_components)),
            ('log_reg', LogisticRegression())
        ])
    
    grid_search = GridSearchCV(clf, param_grid, cv=cv, **grid_kws)
    grid_search.fit(X, y)
    
    return grid_search.best_params_
    
def compile_cv_results(params_list, results_list):
    """
    Compiles cross-validation results into a structured list, each containing the parameter
    set and its corresponding mean test score.

    This function takes a list of parameter dictionaries and a corresponding list of
    result dictionaries, computes the mean of the 'test_scores' for each parameter set,
    and returns a list of dictionaries summarizing the results.

    Parameters
    ----------
    params_list : list of dict
        A list where each element is a dictionary of parameter settings.
    results_list : list of dict
        A list where each element is a dictionary containing the results of testing
        each parameter set. Expected keys include 'test_scores', which should be a
        list of scores, or `nan` if not available.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the 'params' dictionary for a parameter
        set and its 'mean_test_score'.

    Examples
    --------
    >>> params = [{'C': 0.1}, {'C': 1}, {'C': 10}, {'C': 100}]
    >>> results = [
    ...     {'fit_error': None, 'test_scores': [0.8, 0.82, 0.81], 'train_scores': [0.9, 0.92, 0.91]},
    ...     {'fit_error': None, 'test_scores': [0.85, 0.87, 0.86], 'train_scores': [0.95, 0.97, 0.96]},
    ...     {'fit_error': None, 'test_scores': [0.9, 0.92, 0.91], 'train_scores': [0.98, 0.99, 0.97]},
    ...     {'fit_error': None, 'test_scores': [0.93, 0.95, 0.94], 'train_scores': [0.99, 1.0, 0.98]}
    ... ]
    >>> compiled_results = compile_cv_results(params, results)
    >>> for res in compiled_results:
    ...     print(res)
    {'params': {'C': 0.1}, 'mean_test_score': 0.81}
    {'params': {'C': 1}, 'mean_test_score': 0.86}
    {'params': {'C': 10}, 'mean_test_score': 0.91}
    {'params': {'C': 100}, 'mean_test_score': 0.94}

    Notes
    -----
    - The function assumes that each entry in `results_list` corresponds to the parameter
      set in the same position in `params_list`.
    - It uses `numpy.nanmean` for calculating the mean test score, which safely ignores
      any `nan` values. This is particularly useful in situations where some trials might
      not have produced valid scores (e.g., due to errors during model fitting).
    - In the absence of any valid test scores for a parameter set (i.e., all scores are `nan`),
      the mean test score will also be `nan`.
    """
    compiled_results = []
    for param, result in zip(params_list, results_list):
        # Filter out 'nan' values from test_scores and compute the mean
        test_scores = result.get('test_scores')
        if test_scores is not np.nan:  # Assuming 'test_scores' could be an array/list or 'nan'
            mean_test_score = np.nanmean(test_scores)  # Safely compute mean, ignoring 'nan'
        else:
            mean_test_score = np.nan
        
        compiled_result = {
            'params': param,
            'mean_test_score': mean_test_score
        }
        compiled_results.append(compiled_result)
    return compiled_results

def aggregate_cv_results(cv_results):
    """
    Aggregates cross-validation (CV) results for each unique parameter set, computing
    mean scores, fit errors, fit times, and score times across all CV folds.

    This function processes a list of dictionaries, each representing the results of
    a single CV run, and aggregates these results by parameter set. It computes the
    mean test score, mean train score, mean fit error, mean fit time, and mean score
    time for each unique set of parameters.

    Parameters
    ----------
    cv_results : list of dict
        A list where each element is a dictionary containing the results of a single
        CV run. Expected keys in each dictionary include 'parameters' (a dict of
        parameter values), 'test_scores' (a list or a scalar 'nan'), 'train_scores'
        (a list or a scalar 'nan'), 'fit_error' (None or a numeric value),
        'fit_time', and 'score_time'.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary contains a unique 'params'
        field (dict of parameter values) along with the aggregated metrics:
        'mean_test_score', 'mean_train_score', 'mean_fit_error',
        'mean_fit_times', and 'mean_score_times'.

    Examples
    --------
    >>> cv_results = [
    ...     {'parameters': {'C': 0.1}, 'test_scores': [0.8, 0.82],
    ...      'train_scores': [0.9, 0.92], 'fit_error': None, 'fit_time': 0.1,
    ...      'score_time': 0.01, 'n_test_samples': 200},
    ...     {'parameters': {'C': 0.1}, 'test_scores': [0.81, 0.83],
    ...      'train_scores': [0.91, 0.93], 'fit_error': None, 'fit_time': 0.2,
    ...      'score_time': 0.02, 'n_test_samples': 200}
    ... ]
    >>> results = aggregate_cv_results(cv_results)
    >>> print(results)
    [{'params': {'C': 0.1}, 'mean_test_score': 0.815, 'mean_train_score': 0.915,
      'mean_fit_error': nan, 'mean_fit_times': 0.15, 'mean_score_times': 0.015, 
      'n_test_samples': 200}]

    Notes
    -----
    - The function assumes that 'test_scores' and 'train_scores' can either be lists
      of numeric values or a scalar 'nan' to indicate missing data.
    - 'fit_error' is handled as optional, with None indicating no error and numeric
      values indicating some form of error measurement. If all fit errors are None,
      'mean_fit_error' will be reported as `nan`.
    - This function is designed to be flexible, accommodating any set of parameters
      provided in the 'parameters' dictionary of each CV result.
    - It uses `numpy.nanmean` to safely compute mean values while ignoring `nan`s,
      allowing for robust aggregation even in the presence of incomplete data.
    """
    def to_list( value): 
        """ Convert float of non interable value into an iterable value."""
        if not isinstance ( value, list): 
            value = [ value] 
         
        return value 
        
    # Initialize a dictionary to hold aggregated results
    aggregated_results = {}
    
    for result in cv_results:
        # Extract parameter values as a hashable tuple for uniqueness
        params_tuple = tuple(result['parameters'].items())
        
        if params_tuple not in aggregated_results:
            aggregated_results[params_tuple] = {
                'params': result['parameters'],
                'n_test_samples': [], 
                'fit_errors': [],
                'test_scores': [],
                'train_scores': [],
                'fit_times': [],
                'score_times': []
            }
        
        aggregated_results[params_tuple]['fit_errors'].append(result.get('fit_error', np.nan))
        aggregated_results[params_tuple]['n_test_samples'].append(result.get('n_test_samples', np.nan))
        aggregated_results[params_tuple]['test_scores'].extend(
            to_list(result.get('test_scores', [np.nan])))
        aggregated_results[params_tuple]['train_scores'].extend(
            to_list(result.get('train_scores', [np.nan]))) 
        aggregated_results[params_tuple]['fit_times'].append(result.get('fit_time', np.nan))
        aggregated_results[params_tuple]['score_times'].append(result.get('score_time', np.nan))
    
    # Convert aggregated results to the desired format
    final_results = []
    for _, aggregated in aggregated_results.items():
        # Calculate mean and standard deviation of test scores
        mean_test_score = np.nanmean(aggregated['test_scores'])
        std_test_score = np.nanstd(aggregated['test_scores'])
        
        # Calculate other aggregated metrics
        mean_train_score = np.nanmean(aggregated['train_scores'])
        mean_fit_times = np.nanmean(aggregated['fit_times'])
        mean_score_times = np.nanmean(aggregated['score_times'])
        mean_n_test_samples = np.nanmean(aggregated['n_test_samples'])
        fit_error= np.nan if all(x is None for x in aggregated['fit_errors']) else np.nanmean(
                                            [x for x in aggregated['fit_errors'] if x is not None]),
        
        # Append the formatted result
        final_results.append({
            'params': aggregated['params'],
            'mean_test_score': mean_test_score,
            'std_test_score': std_test_score,
            'mean_train_score': mean_train_score,
            'mean_fit_times': mean_fit_times,
            'mean_score_times': mean_score_times,
            'mean_n_test_samples': mean_n_test_samples,
            'fit_error': fit_error
        })
    
    return final_results

def get_param_types2(estimator: BaseEstimator) -> dict:
    """
    Get the parameter types for a given estimator.
    
    Parameters
    ----------
    estimator : BaseEstimator
        An instance of a scikit-learn estimator.
    
    Returns
    -------
    param_types : dict
        A dictionary mapping parameter names to their types.
    """
    params = estimator.get_params()
    param_types = {param: type(value) for param, value in params.items()}
    return param_types

def resolve_param_type(value, expected_types):
    """
    Resolve the type of a parameter value based on the expected types.
    
    Parameters
    ----------
    value : any
        The parameter value to be resolved.
    expected_types : tuple
        A tuple of expected types.
    
    Returns
    -------
    resolved_value : any
        The parameter value converted to the appropriate type if possible,
        otherwise the original value.
    """
    for expected_type in expected_types:
        try:
            if expected_type == str and value in {'scale', 'auto'}:
                return value
            return expected_type(value)
        except (ValueError, TypeError):
            continue
    return value

def apply_param_types2(estimator: BaseEstimator, param_dict: dict) -> dict:
    """
    Apply the parameter types to the values in the given dictionary.
    
    Parameters
    ----------
    estimator : BaseEstimator
        An instance of a scikit-learn estimator.
    param_dict : dict
        A dictionary of hyperparameters.
    
    Returns
    -------
    new_param_dict : dict
        A new dictionary with values converted to the expected types.
    """
    param_types = get_param_types(estimator)
    new_param_dict = {}
    
    for param, value in param_dict.items():
        if param in param_types:
            expected_type = param_types[param]
            if expected_type in {str, float}:
                new_param_dict[param] = resolve_param_type(value, (str, float))
            else:
                new_param_dict[param] = expected_type(value)
        else:
            new_param_dict[param] = value  # keep original if param not found
    
    return new_param_dict

                                                                                                                      
def get_param_types(estimator: BaseEstimator) -> dict:
    """
    Get the parameter types for a given estimator.
    
    Parameters
    ----------
    estimator : BaseEstimator
        An instance of a scikit-learn estimator.
    
    Returns
    -------
    param_types : dict
        A dictionary mapping parameter names to their types.
    """
    params = estimator.get_params()
    param_types = {param: type(value) for param, value in params.items()}
    return param_types

def apply_param_types(estimator: BaseEstimator, param_dict: dict) -> dict:
    """
    Apply the parameter types to the values in the given dictionary.
    
    Parameters
    ----------
    estimator : BaseEstimator
        An instance of a scikit-learn estimator.
    param_dict : dict
        A dictionary of hyperparameters.
    
    Returns
    -------
    new_param_dict : dict
        A new dictionary with values converted to the expected types.
    """
    param_types = get_param_types(estimator)
    new_param_dict = {}
    for param, value in param_dict.items():
        if param in param_types:
            expected_type = param_types[param]
            if value is None:
                new_param_dict[param] = value
            elif isinstance(value, expected_type):
                new_param_dict[param] = value
            else:
                try:
                    new_param_dict[param] = expected_type(value)
                except (TypeError, ValueError):
                    new_param_dict[param] = value
        else:
            new_param_dict[param] = value  # keep original if param not found
        
        # Attempt to convert string to float if applicable
        if isinstance(new_param_dict[param], str):
            try:
                new_param_dict[param] = float(new_param_dict[param])
            except ValueError:
                pass
            
    return new_param_dict



def process_performance_data(df, mode='average', on='@data'):
    """
    Process performance data based on specified parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing performance data with each cell being a list of scores.
        If the DataFrame contains only numerical values, it will be automatically
        converted to a DataFrame with each value wrapped in a list.
    
    mode : str, optional, default='average'
        Processing mode. Options:
        - 'average': Averages each row array value at each index.
        - 'rowX': Processes data based on the specific row index (e.g., 'row0', 'row1').
        - 'dX': Same as 'rowX'.
        - 'cvX': Processes data based on the specific column index (e.g., 'cv0', 'cv1').
        - 'cX': Same as 'cvX'.
    
    on : str, optional, default='@data'
        Specifies the axis of processing.
        - '@data': Processes data based on rows.
        - '@cv': Processes data based on columns.
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns names and processed values.
    
    Raises
    ------
    ValueError
        If the parameters for 'mode' or 'on' are invalid, or if the specified
        row or column index is out of bounds.
    
    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.models.utils import process_performance_data
    >>> df = pd.DataFrame({
    ...     'DecisionTreeClassifier': [[0.9667, 0.9667, 0.9, 0.9667, 1.0],
    ...                                [0.9667, 0.9667, 0.9, 0.9667, 1.0],
    ...                                [0.9667, 0.9667, 0.9, 0.9667, 1.0]],
    ...     'LogisticRegression': [[0.9667, 1.0, 0.9333, 0.9667, 1.0],
    ...                             [0.9667, 1.0, 0.9333, 0.9667, 1.0],
    ...                             [0.9667, 1.0, 0.9333, 0.9667, 1.0]]
    ... })
    >>> process_performance_data(df, mode='average')
        DecisionTreeClassifier  LogisticRegression
     0                 0.96002             0.97334
     1                 0.96002             0.97334
     2                 0.96002             0.97334
    >>> process_performance_data(df, mode='row0')
        DecisionTreeClassifier  LogisticRegression
     0                  0.9667              0.9667
     1                  0.9667              0.9667
     2                  0.9667              0.9667
    >>> process_performance_data(df, mode='cv1', on='@cv')
        DecisionTreeClassifier  LogisticRegression
     0                  0.9667                 1.0
     1                  0.9667                 1.0
     2                  0.9667                 1.0
    """
    # Check if each cell is a list of scores
    if not all(isinstance(i, list) for col in df.columns for i in df[col]):
        # If the DataFrame contains only numerical values, wrap each value in a list
        if df.applymap(lambda x: isinstance(x, (int, float))).all().all():
            df = df.applymap(lambda x: [x])
        else:
            raise ValueError("DataFrame should contain lists of scores in each cell.")

    if mode == 'average':
        # Average each row array value at each index
        if on == '@cv':
            # Average across cross-validation folds
            processed_data = {}
            for col in df.columns:
                # Transpose the list of lists to get columns of CV results
                transposed_cv = list(zip(*df[col]))
                processed_data[col] = [sum(cv_fold) / len(cv_fold) for cv_fold in transposed_cv]
            return pd.DataFrame(processed_data)
        else:
            # Average each row array value at each index
            processed_data = {col: df[col].apply(lambda x: sum(x) / len(x)) for col in df.columns}
            
        return pd.DataFrame(processed_data)

    if on == '@data':
        # Process based on row index
        if mode.startswith('row') or mode.startswith('d'):
            index = int(mode[3]) if mode.startswith('row') else int(mode[1])
            if index >= len(df):
                raise ValueError(f"Invalid row index: {index}. DataFrame only has {len(df)} rows.")
            processed_data = {col: df[col].apply(lambda x: x[index]) for col in df.columns}
            return pd.DataFrame(processed_data)

    if on == '@cv':
        # Process based on column index
        if mode.startswith('cv') or mode.startswith('c'):
            index = int(mode[2]) if mode.startswith('cv') else int(mode[1])
            max_len = max(df[col].apply(len).max() for col in df.columns)
            if index >= max_len:
                raise ValueError(
                    f"Invalid column index: {index}. DataFrame columns have"
                    f" a maximum of {max_len} values.")
            processed_data = {col: [row[index] for row in df[col]] for col in df.columns}
            return pd.DataFrame(processed_data)

    raise ValueError("Invalid parameters for 'mode' or 'on'")
    
def update_if_higher(
    results_dict, 
    estimator_name, 
    new_score, 
    result_data, 
    best_params_dict=None
    ):
    """
    Updates the results dictionary with the new score if it is higher than the 
    current score and updates the best_params dictionary accordingly.

    Parameters
    ----------
    results_dict : dict
        The dictionary containing the results of each estimator.
    estimator_name : str
        The key in the dictionary to update (name of the estimator).
    new_score : float
        The new score to compare and potentially update.
    result_data : dict
        The result dictionary containing additional details to update.
    best_params_dict : dict, optional
        The dictionary containing the best parameters for each estimator.

    Returns
    -------
    results_dict : dict
        The updated results dictionary.
    best_params_dict : dict
        The updated best parameters dictionary.

    Notes
    -----
    This function ensures that the `results_dict` and `best_params_dict` are 
    updated only if the new score is higher than the existing score for a 
    given estimator. If the estimator is not already in the `results_dict`, it 
    adds the estimator and its corresponding result data.

    Examples
    --------
    >>> results = {}
    >>> best_params = {}
    >>> estimator_name = 'RandomForest'
    >>> new_score = 0.85
    >>> result_data = {
    ...     'RandomForest': {
    ...         'best_estimator_': rf_best_estimator,
    ...         'best_params_': rf_best_params,
    ...         'best_score_': 0.85,
    ...         'scoring': 'accuracy',
    ...         'strategy': 'GridSearchCV',
    ...         'cv_results_': rf_cv_results,
    ...     }
    ... }
    >>> update_if_higher(results, estimator_name, new_score, result_data, best_params)
    """
    best_params_dict = best_params_dict or {}
    if estimator_name in results_dict:
        if new_score > results_dict[estimator_name]['best_score_']:
            results_dict[estimator_name] = result_data[estimator_name]
            best_params_dict[estimator_name] = result_data[estimator_name]['best_params_']
    else:
        results_dict[estimator_name] = result_data[estimator_name]
        best_params_dict[estimator_name] = result_data[estimator_name]['best_params_']
        
    return results_dict, best_params_dict

def prepare_estimators_and_param_grids(
        estimators, param_grids, alignment_mode='soft'):
    """
    Prepare and associate estimators and their corresponding parameter grids.

    This function takes in estimators and parameter grids in various formats and
    ensures they are correctly associated and aligned. It supports both dictionary
    and list inputs for flexibility and versatility.

    Parameters
    ----------
    estimators : dict or list
        Estimators can be provided in two formats:
        - As a dictionary where keys are estimator names and values are estimator
          objects.
          Example: ``{'rf': RandomForestClassifier(), 'svc': SVC()}``
        - As a list of estimator objects. The function will use the class names 
          as keys.
          Example: ``[RandomForestClassifier(), SVC()]``

    param_grids : dict or list
        Parameter grids can be provided in two formats:
        - As a dictionary where keys match the estimator names and values 
        are parameter grids.
          Example: ``{'rf': {'n_estimators': [10, 100], 'max_depth': [None, 10]},
                      'svc': {'C': [1, 10], 'kernel': ['linear', 'rbf']}}``
        - As a list of parameter grids. The function will align them with 
          estimators by order if alignment_mode is 'soft'.
          Example: ``[{'n_estimators': [10, 100], 'max_depth': [None, 10]}, 
                      {'C': [1, 10], 'kernel': ['linear', 'rbf']}]``

    alignment_mode : str, optional
        Specifies the alignment mode for associating parameter grids with 
        estimators when provided as lists.
        - 'soft': Align parameter grids with estimators by order.
        - 'strict': Raise an error if parameter grids are provided as lists.
        Default is 'soft'.

    Returns
    -------
    est_dict : dict
        A dictionary where keys are estimator names and values are estimator 
        objects.

    param_dict : dict
        A dictionary where keys are estimator names and values are parameter 
        grids.

    Raises
    ------
    ValueError
        If the length of estimators and parameter grids do not match, or if 
        the keys of the dictionaries do not match, or if the alignment mode is invalid.

    Notes
    -----
    This function ensures that the estimators and parameter grids are correctly
    aligned and associated.
    If both estimators and parameter grids are provided as dictionaries, their 
    keys must match.
    If they are provided as lists, the function uses the order to align them in 
    'soft' mode,and raises an error in 'strict' mode.

    Examples
    --------
    >>> from gofast.models.utils import prepare_estimators_and_param_grids
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> estimators = {'rf': RandomForestClassifier(), 'svc': SVC()}
    >>> param_grids = {'rf': {'n_estimators': [10, 100], 'max_depth': [None, 10]},
    ...                'svc': {'C': [1, 10], 'kernel': ['linear', 'rbf']}}
    >>> est_dict, param_dict = prepare_estimators_and_param_grids(estimators, param_grids)
    >>> print(est_dict)
    {'rf': RandomForestClassifier(), 'svc': SVC()}
    >>> print(param_dict)
    {'rf': {'n_estimators': [10, 100], 'max_depth': [None, 10]}, 
    'svc': {'C': [1, 10], 'kernel': ['linear', 'rbf']}}

    See Also
    --------
    sklearn.ensemble.RandomForestClassifier : A random forest classifier.
    sklearn.svm.SVC : A support vector classifier.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., 
       Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J.,
       Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). 
       Scikit-learn: Machine Learning in Python. Journal of Machine Learning 
       Research, 12, 2825-2830.
    """
    if isinstance(estimators, dict):
        est_dict = estimators
    elif isinstance(estimators, list):
        est_dict = {get_estimator_name(est): est for est in estimators}
    else:
        raise ValueError("Estimators should be a dictionary or a list.")

    if isinstance(param_grids, dict):
        param_dict = param_grids
    elif isinstance(param_grids, list):
        if alignment_mode == 'soft':
            if len(param_grids) != len(est_dict):
                raise ValueError(
                    "Length of parameter grids does not match the length of estimators.")
            param_dict = {name: param_grids[i] for i, name in enumerate(est_dict.keys())}
        elif alignment_mode == 'strict':
            raise ValueError("In strict mode, param_grids should be a dictionary"
                             " with matching keys to estimators.")
        else:
            raise ValueError("Invalid alignment_mode. Use 'soft' or 'strict'.")
    else:
        raise ValueError("param_grids should be a dictionary or a list.")

    if est_dict.keys() != param_dict.keys():
        raise ValueError("Keys of estimators and param_grids must match.")

    return est_dict, param_dict

if __name__=='__main__': 
    from sklearn.ensemble import RandomForestClassifier
    # from gofast.models.utils import prepare_estimators_and_param_grids

    # Example usage:
    estimators = {'rf': RandomForestClassifier(), 'svc': SVC()}
    param_grids = {'rf': {'n_estimators': [10, 100], 'max_depth': [None, 10]},
                    'svc': {'C': [1, 10], 'kernel': ['linear', 'rbf']}}
    
    estimators_list, param_grids_list = [
        RandomForestClassifier(), SVC()], [{'n_estimators': [10, 100], 
                                            'max_depth': [None, 10]}, 
                                            {'C': [1, 10], 'kernel': ['linear', 'rbf']}]
    
    # Testing the function
    est_dict, param_dict = prepare_estimators_and_param_grids(estimators, param_grids)
    print(est_dict)
    print(param_dict)
    
    est_dict, param_dict = prepare_estimators_and_param_grids(
        estimators_list, param_grids_list)
    print(est_dict)
    print(param_dict)

