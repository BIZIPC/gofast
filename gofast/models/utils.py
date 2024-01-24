# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator 
from sklearn.covariance import ShrunkCovariance
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV  
from sklearn.svm import SVC, SVR
from sklearn.utils.multiclass import type_of_target

from .._typing import Tuple,_F, ArrayLike, NDArray, Dict, Union, Any
from .._typing import  List, Optional, Type
from ..tools.funcutils import smart_format
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
    'display_fine_tuned_results',
    'display_cv_tables', 
    'get_scorers', 
    'naive_evaluation', 
    "calculate_aggregate_scores", 
    "analyze_score_distribution", 
    "estimate_confidence_interval", 
    "rank_cv_scores", 
    "filter_scores", 
    "visualize_score_distribution", 
    "performance_over_time", 
    "calculate_custom_metric", 
    "handle_missing_data", 
    "export_cv_results", 
    "comparative_analysis", 
    "plot_parameter_importance", 
    "plot_hyperparameter_heatmap", 
    "plot_learning_curve", 
    "plot_validation_curve", 
    "plot_feature_importance",
    "plot_roc_curve_per_fold", 
    "plot_confidence_intervals", 
    "plot_pairwise_model_comparison",
    "plot_feature_correlation", 
    "quick_evaluation", 
    "validate_optimizer", 
    "get_optimizer_method", 
    "process_estimators_and_params", 
  ]

def get_optimizer_method(optimizer: str) -> Type[BaseEstimator]:
    """
    Returns the corresponding optimizer class based on the provided optimizer 
    string.
    
    This function accounts for standard optimizers as well as custom optimizers 
    defined in gofast.

    Parameters
    ----------
    optimizer : str
        The name or abbreviation of the optimizer.

    Returns
    -------
    Type[BaseEstimator]
        The class of the optimizer corresponding to the provided optimizer 
        string.

    Raises
    ------
    ImportError
        If a required external optimizer class (e.g., BayesSearchCV) is not 
        installed.
    ValueError
        If no matching optimizer is found or the optimizer name is unrecognized.

    Examples
    --------
    >>> from gofast.models.utils import get_optimizer_method
    >>> optimizer_class = get_optimizer_method('RSCV')
    >>> print(optimizer_class)
    <class 'sklearn.model_selection.RandomizedSearchCV'>
    >>> optimizer_class = get_optimizer_method('GASCV')
    >>> print(optimizer_class)
    <class 'gofast.models.selection.GeneticSearchCV'>
    """
    # Ensure the optimizer name is standardized
    optimizer = validate_optimizer(optimizer) 
    
    # Mapping of optimizer names to their respective classes
    # Standard optimizer dictionary
    standard_optimizer_dict = {
        'GridSearchCV': GridSearchCV,
        'RandomizedSearchCV': RandomizedSearchCV,
    }
    try: from skopt import BayesSearchCV
    except: 
        if optimizer =='BayesSearchCV': 
            emsg= ("scikit-optimize is required for 'BayesSearchCV'"
                   " but not installed.")
            import_optional_dependency('skopt', extra= emsg )
        pass 
    else : standard_optimizer_dict["BayesSearchCV"]= BayesSearchCV
    
    # Update standard optimizer with gofast optimizers if 
    # not exist previously.
    if optimizer not in standard_optimizer_dict.keys(): 
        from gofast.models.selection import ( 
            PSOSearchCV, 
            SMBOSearchCV, 
            AnnealingSearchCV, 
            EvolutionarySearchCV, 
            GradientBasedSearchCV,
            GeneticSearchCV 
            ) 
        gofast_optimizer_dict = { 
            'PSOSearchCV': PSOSearchCV,'SMBOSearchCV': SMBOSearchCV,
            'AnnealingSearchCV': AnnealingSearchCV,
            'EvolutionarySearchCV': EvolutionarySearchCV,
            'GradientBasedSearchCV': GradientBasedSearchCV,
            'GeneticSearchCV': GeneticSearchCV,
            }
        standard_optimizer_dict ={**standard_optimizer_dict,**gofast_optimizer_dict }
        
    # Search for the corresponding optimizer class
    return standard_optimizer_dict.get(optimizer)
    
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
        
def validate_optimizer(optimizer: Union[str, _F]) -> str:
    """
    Check whether the given optimizer is a recognized optimizer type.

    This function validates if the provided optimizer, either as a string 
    or an instance of a class derived from BaseEstimator, corresponds to a 
    known optimizer type. If the optimizer is recognized, its standardized 
    name is returned. Otherwise, a ValueError is raised.

    Parameters
    ----------
    optimizer : Union[str, _F]
        The optimizer to validate. This can be a string name or an instance 
        of an optimizer class.

    Returns
    -------
    str
        The standardized name of the optimizer.

    Raises
    ------
    ValueError
        If the optimizer is not recognized.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier 
    >>> from gofast.models.selection import AnnealingSearchCV
    >>> from gofast.models.utils import validate_optimizer
    >>> validate_optimizer("RSCV")
    'RandomizedSearchCV'
    >>> validate_optimizer(AnnealingSearchCV)
    'AnnealingSearchCV'
    >>> validate_optimizer (RandomForestClassifier)
    ValueError ...
    """
    # Mapping of optimizer names to their possible abbreviations and variations
    opt_dict = {
        'RandomizedSearchCV': ['RSCV', 'RandomizedSearchCV'], 
        'GridSearchCV': ['GSCV', 'GridSearchCV'], 
        'BayesSearchCV': ['BSCV', 'BayesSearchCV'], 
        'AnnealingSearchCV': ['ASCV', "AnnealingSearchCV"], 
        'PSOSearchCV': ['PSCV', 'PSOSearchCV'], 
        'SMBOSearchCV': ['SSCV', 'SMBOSearchCV'], 
        'EvolutionarySearchCV': ['ESCV', 'EvolutionarySearchCV'], 
        'GradientBasedSearchCV':['GBSCV', 'GradientBasedSearchCV'], 
        'GeneticSearchCV': ['GASCV', 'GeneticSearchCV']
    }

    optimizer_name = optimizer if isinstance(
        optimizer, str) else get_estimator_name(optimizer)

    for key, values in opt_dict.items():
        if optimizer_name.lower() in [v.lower() for v in values]:
            return key

    valid_optimizers = [v1[1] for v1 in opt_dict.values()]
    raise ValueError(f"Invalid 'optimizer' parameter '{optimizer_name}'."
                     f" Choose from {smart_format(valid_optimizers, 'or')}.")
    
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
    """ Get the best score at each split from cross-validation results
    
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
    #if split ==0: split =1 
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
    for k in range (cv):
        print(f'split = {k}:')
        b= get_split_best_scores(cvres, split =k)
        print( b)

    globalmeansc , globalstdsc= get_cv_mean_std_scores(cvres)
    print("Global split scores:")
    print('mean=', globalmeansc , 'std=',globalstdsc)


def display_fine_tuned_results ( cvmodels: list[_F] ): 
    """Display fined -tuning results 
    
    Parameters 
    -----------
    cvmnodels: list
        list of fined-tuned models.
    """
    bsi_bestestimators = [model.best_estimator_ for model in cvmodels ]
    mnames = ( get_estimator_name(n) for n in bsi_bestestimators)
    bsi_bestparams = [model.best_params_ for model in cvmodels]

    for nam, param , estimator in zip(mnames, bsi_bestparams, 
                                      bsi_bestestimators): 
        print("MODEL NAME =", nam)
        print('BEST PARAM =', param)
        print('BEST ESTIMATOR =', estimator)
        print()

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
    >>> from gofast.models import GridSearchMultiple, displayCVTables
    >>> X, y  = fetch_data ('bagoue prepared') 
    >>> gobj =GridSearchMultiple(estimators = estimators, 
                                 grid_params = grid_params ,
                                 cv =4, scoring ='accuracy', 
                                 verbose =1,  savejob=False , 
                                 kind='GridSearchCV')
    >>> gobj.fit(X, y) 
    >>> displayCVTables (cvmodels=[gobj.models.SVC] ,
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
    cv_results, ax=None,
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

def handle_missing_data(cv_scores, fill_value=np.nan):
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


def plot_learning_curve(estimator, X, y, cv=None, train_sizes=None):
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


def quick_evaluation(
    clf: _F,
    X: NDArray,
    y: ArrayLike,
    cv: int = 7,
    scoring: str = 'accuracy',
    display: bool = False,
    **kws
) -> Tuple[ArrayLike, float]:
    """
    Perform a quick evaluation of a classifier using cross-validation.

    This function calculates cross-validation scores for a given classifier
    on provided data. It optionally prints the scores and their mean.

    Parameters
    ----------
    clf : _F
        Classifier to be evaluated.
    X : NDArray
        Training data, where n_samples is the number of samples and
        n_features is the number of features.
    y : ArrayLike
        Target labels corresponding to X.
    cv : int, optional
        Number of folds for cross-validation. Default is 7.
    scoring : str, optional
        Scoring metric to use. Default is 'accuracy'.
    display : bool, optional
        Whether to print the scores and their mean. Default is False.
    **kws : dict
        Additional keyword arguments passed to `cross_val_score`.

    Returns
    -------
    scores : np.ndarray
        Array of scores of the estimator for each run of the cross-validation.
    mean_score : float
        Mean of the cross-validation scores.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gofast.models.search import naive_evaluation
    >>> X, y = gf.fetch_data('bagoue data prepared')
    >>> clf = DecisionTreeClassifier()
    >>> scores, mean_score = naive_evaluation(clf, X, y, cv=4, display=True)
    clf: DecisionTreeClassifier
    scores: [0.6279 0.7674 0.7093 0.593 ]
    scores.mean: 0.6744
    """
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring, **kws)
    mean_score = scores.mean()

    if display:
        clf_name = clf.__class__.__name__
        print(f'clf: {clf_name}')
        print(f'scores: {scores}')
        print(f'scores.mean: {mean_score:.4f}')

    return scores, mean_score
   
def naive_evaluation(
        clf: _F,
        X:NDArray,
        y:ArrayLike,
        cv:int =7,
        scoring:str  ='accuracy', 
        display: str ='off', 
        **kws
        ): 
    scores = cross_val_score(clf , X, y, cv = cv, scoring=scoring, **kws)
                         
    if display is True or display =='on':
        print('clf=:', clf.__class__.__name__)
        print('scores=:', scores )
        print('scores.mean=:', scores.mean())
    
    return scores , scores.mean()

naive_evaluation.__doc__="""\
Quick scores evaluation using cross validation. 

Parameters
----------
clf: callable 
    Classifer for testing default data. 
X: ndarray
    trainset data 
    
y: array_like 
    label data 
cv: int 
    KFold for data validation.
    
scoring: str 
    type of error visualization. 
    
display: str or bool, 
    show the show on the stdout
kws: dict, 
    Additional keywords arguments passed to 
    :func:`gofast.exlib.slearn.cross_val_score`.
Returns 
---------
scores, mean_core: array_like, float 
    scaore after evaluation and mean of the score
    
Examples 
---------
>>> import gofast as gf 
>>> from gofast.models.search import naive_evaluation
>>> X,  y = gf.fetch_data ('bagoue data prepared') 
>>> clf = gf.sklearn.DecisionTreeClassifier() 
>>> naive_evaluation(clf, X, y , cv =4 , display ='on' )
clf=: DecisionTreeClassifier
scores=: [0.6279 0.7674 0.7093 0.593 ]
scores.mean=: 0.6744186046511629
Out[57]: (array([0.6279, 0.7674, 0.7093, 0.593 ]), 0.6744186046511629)
"""

def shrink_covariance_cv_score(X, skrink_space =( -2, 0, 30 )):
    shrinkages = np.logspace(*skrink_space)  # Fit the models
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))

shrink_covariance_cv_score.__doc__="""\
shrunk the covariance scores from validating X using 
GridSearchCV.
 
Parameters 
-----------
X : array_like, pandas.DataFrame 
    Input data where rows represent samples and 
    columns represent features.

Returns
-----------
score: score of covariance estimator (best ) with shrinkage

"""